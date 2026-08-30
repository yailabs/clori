#!/bin/sh
# Exercise the scrollback-preserving console and deterministic CLI isolation through real PTYs.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEX_TEST_HOST=${YVEX_TEST_HOST:-build/tests/openai_host}
. tests/support/cleanup.sh

case "$YVEX_BIN" in
    /*) ;;
    *) YVEX_BIN="$(pwd -P)/${YVEX_BIN#./}" ;;
esac
case "$YVEX_TEST_HOST" in
    /*) ;;
    *) YVEX_TEST_HOST="$(pwd -P)/${YVEX_TEST_HOST#./}" ;;
esac

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-repl-pty.XXXXXX")
runtime="$root/runtime"
config="$root/config"
models="$root/models"
registry="$root/models.local.json"
socket="$runtime/yvex/yvexd.sock"
host_pid=
console_job=
client_pid=
mkdir -m 700 "$runtime" "$runtime/yvex" "$config" "$models"
printf '{"schema":"yvex.models.local.v6","models":[]}\n' >"$registry"

cleanup()
{
    status=$?
    if test "$status" -ne 0; then
        for transcript in "$root"/*.typescript; do
            test -f "$transcript" && tail -c 12000 "$transcript" >&2 || true
        done
    fi
    if test -n "$console_job" && kill -0 "$console_job" 2>/dev/null; then
        kill "$console_job" 2>/dev/null || true
        wait "$console_job" 2>/dev/null || true
    fi
    if test -n "$host_pid" && kill -0 "$host_pid" 2>/dev/null; then
        kill "$host_pid" 2>/dev/null || true
        wait "$host_pid" 2>/dev/null || true
    fi
    yvex_test_cleanup "$root"
    return "$status"
}
trap cleanup EXIT HUP INT TERM

wait_for()
{
    file=$1
    needle=$2
    attempt=0
    while test "$attempt" -lt 400; do
        test -f "$file" && grep -F "$needle" "$file" >/dev/null 2>&1 && return 0
        attempt=$((attempt + 1))
        sleep 0.01
    done
    echo "timed out waiting for '$needle' in $file" >&2
    return 1
}

wait_count()
{
    file=$1
    needle=$2
    minimum=$3
    attempt=0
    while test "$attempt" -lt 400; do
        count=$(grep -F -c "$needle" "$file" 2>/dev/null || true)
        test "$count" -ge "$minimum" && return 0
        attempt=$((attempt + 1))
        sleep 0.01
    done
    echo "timed out waiting for $minimum occurrences of '$needle' in $file" >&2
    return 1
}

start_host()
{
    "$YVEX_TEST_HOST" "$socket" 2>>"$root/host.err" &
    host_pid=$!
    attempt=0
    while test "$attempt" -lt 400 && test ! -S "$socket"; do
        kill -0 "$host_pid"
        attempt=$((attempt + 1))
        sleep 0.01
    done
    test -S "$socket"
}

stop_host()
{
    test -n "$host_pid"
    kill "$host_pid"
    wait "$host_pid"
    host_pid=
    attempt=0
    while test "$attempt" -lt 400 && test -e "$socket"; do
        attempt=$((attempt + 1))
        sleep 0.01
    done
    test ! -e "$socket"
}

find_console_client()
{
    for candidate in $(pgrep -x yvex 2>/dev/null || true); do
        if tr '\000' '\n' <"/proc/$candidate/environ" 2>/dev/null |
            grep -Fqx "XDG_RUNTIME_DIR=$runtime"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

start_console()
{
    name=$1
    rows=$2
    columns=$3
    command=$4
    color=$5
    fifo="$root/$name.input"
    transcript="$root/$name.typescript"
    mkfifo "$fifo"
    if test "$color" = color; then
        color_env='env -u NO_COLOR'
    else
        color_env='env NO_COLOR=1'
    fi
    $color_env TERM=xterm-256color XDG_RUNTIME_DIR="$runtime" \
        YVEX_MODELS_REGISTRY="$registry" YVEX_MODELS_ROOT="$models" \
        YVEX_CONFIG_DIR="$config" XDG_CONFIG_HOME="$config" \
        script -q -f -e \
        -c "cd $root; stty rows $rows cols $columns; exec $YVEX_BIN $command" \
        "$transcript" <"$fifo" >"$root/$name.stdout" 2>"$root/$name.stderr" &
    console_job=$!
    exec 3>"$fifo"
    wait_for "$transcript" 'deepseek4-v4-flash-dspark>'
    attempt=0
    while test "$attempt" -lt 400; do
        client_pid=$(find_console_client || true)
        test -n "$client_pid" && return 0
        attempt=$((attempt + 1))
        sleep 0.01
    done
    echo 'timed out finding linear console client' >&2
    return 1
}

finish_console()
{
    exec 3>&-
    wait "$console_job"
    console_job=
    client_pid=
}

assert_linear_terminal()
{
    transcript=$1
    esc=$(printf '\033')
    ! grep -F "${esc}[?1049h" "$transcript" >/dev/null
    ! grep -F "${esc}[?1049l" "$transcript" >/dev/null
    ! grep -F "${esc}[?25l" "$transcript" >/dev/null
    ! grep -F "${esc}[?25h" "$transcript" >/dev/null
    ! grep -F "${esc}[48;" "$transcript" >/dev/null
    ! grep -F "${esc}[40m" "$transcript" >/dev/null
    grep -F "${esc}[?2004h" "$transcript" >/dev/null
    grep -F "${esc}[?2004l" "$transcript" >/dev/null
}

start_host

# Deterministic and redirected commands never emit terminal control sequences.
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --reasoning high \
    --max-new-tokens 3 --strategy greedy REASONING_STREAM \
    >"$root/raw.out" 2>"$root/raw.err"
printf 'I need to compare the constraints...\nThe valid result is 42.' >"$root/raw.expected"
cmp "$root/raw.expected" "$root/raw.out"
! grep "$(printf '\033')" "$root/raw.out" >/dev/null

# The root console refuses non-terminal input; the old explicit spelling is retired.
set +e
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" chat </dev/null \
    >"$root/non-tty.out" 2>"$root/non-tty.err"
chat_status=$?
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" </dev/null \
    >"$root/bare.out" 2>"$root/bare.err"
bare_status=$?
set -e
test "$chat_status" -eq 2
test "$bare_status" -eq 2
grep -F 'removed command: chat' "$root/non-tty.err" >/dev/null
grep -F 'run `yvex` for the interactive console' "$root/non-tty.err" >/dev/null
grep -F 'chat requires a terminal' "$root/bare.err" >/dev/null
! grep "$(printf '\033')" "$root/non-tty.out" "$root/non-tty.err" \
    "$root/bare.out" "$root/bare.err" >/dev/null

# Root options preserve scrollback, stream output, and restore bracketed paste mode.
start_console explicit 24 100 '--session linear' nocolor
printf 'hello\r' >&3
wait_for "$root/explicit.typescript" 'hello from yvex'
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/explicit.typescript"

# Bare yvex is the same console. Exercise completion, UTF-8 editing, paste, history, and resize.
start_console bare 32 150 '' color
printf '/sta\t\r' >&3
wait_for "$root/bare.typescript" '/status'
printf '\033[200~hello\nworld 🌍\033[201~\r' >&3
wait_count "$root/bare.typescript" 'hello from yvex' 1
printf '\033[A\r' >&3
wait_count "$root/bare.typescript" 'hello from yvex' 2
printf 'world\033[Hhello \033[F 🌍\r' >&3
wait_for "$root/bare.typescript" 'hello world'
wait_count "$root/bare.typescript" 'hello from yvex' 3
printf 'draft-resize' >&3
kill -WINCH "$client_pid"
wait_for "$root/bare.typescript" 'draft-resize'
kill -INT "$client_pid"
wait_for "$root/bare.typescript" '^C'
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/bare.typescript"

# A transport loss leaves the draft loop alive; the next request reconnects to a restarted host.
start_console reconnect 24 100 '--session reconnect' nocolor
stop_host
printf 'first while offline\r' >&3
wait_for "$root/reconnect.typescript" '[disconnected]'
start_host
printf 'hello after restart\r' >&3
wait_for "$root/reconnect.typescript" 'reconnected'
wait_for "$root/reconnect.typescript" 'hello from yvex'
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/reconnect.typescript"

# Active generation Ctrl-C crosses the canonical cancellation operation.
start_console cancel 24 100 '--session cancel' nocolor
printf 'WAIT_PREFILL_CANCEL\r' >&3
wait_for "$root/cancel.typescript" 'processing 4 input tokens · 0/4'
kill -INT "$client_pid"
wait_for "$root/host.err" 'generation.cancel cancel'
wait_for "$root/cancel.typescript" 'cancelled'
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/cancel.typescript"

# Ctrl-D exits with normal terminal restoration and preserved scrollback.
start_console eof 18 88 '--session eof' nocolor
printf 'preserved-unsubmitted\004' >&3
finish_console
assert_linear_terminal "$root/eof.typescript"

echo 'linear console PTY lifecycle: pass'
