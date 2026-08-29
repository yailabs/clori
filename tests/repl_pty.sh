#!/bin/sh
# Exercise the official full-screen client and deterministic CLI isolation through real PTYs.
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

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-tui-pty.XXXXXX")
runtime="$root/runtime"
config="$root/config"
models="$root/models"
registry="$root/models.local.json"
socket="$runtime/yvex/yvexd.sock"
host_pid=
tui_pid=
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
    if test -n "$tui_pid" && kill -0 "$tui_pid" 2>/dev/null; then
        kill "$tui_pid" 2>/dev/null || true
        wait "$tui_pid" 2>/dev/null || true
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

find_tui_client()
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

start_tui()
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
    tui_pid=$!
    exec 3>"$fifo"
    wait_for "$transcript" 'YVEX'
    attempt=0
    while test "$attempt" -lt 400; do
        client_pid=$(find_tui_client || true)
        test -n "$client_pid" && return 0
        attempt=$((attempt + 1))
        sleep 0.01
    done
    echo 'timed out finding native TUI client' >&2
    return 1
}

finish_tui()
{
    exec 3>&-
    wait "$tui_pid"
    tui_pid=
    client_pid=
}

assert_restored()
{
    transcript=$1
    esc=$(printf '\033')
    grep -F "${esc}[?1049h" "$transcript" >/dev/null
    grep -F "${esc}[?1049l" "$transcript" >/dev/null
    grep -F "${esc}[?25l" "$transcript" >/dev/null
    grep -F "${esc}[?25h" "$transcript" >/dev/null
    grep -F "${esc}[?2004h" "$transcript" >/dev/null
    grep -F "${esc}[?2004l" "$transcript" >/dev/null
}

assert_terminal_background()
{
    transcript=$1
    esc=$(printf '\033')
    ! grep -F "${esc}[48;" "$transcript" >/dev/null
    ! grep -F "${esc}[40m" "$transcript" >/dev/null
}

# Offline transcript preserves the same composer and exposes model selection.
start_tui offline 24 100 '' nocolor
wait_for "$root/offline.typescript" 'No model loaded'
wait_for "$root/offline.typescript" 'Ask YVEX anything'
kill -INT "$client_pid"
finish_tui
assert_restored "$root/offline.typescript"
assert_terminal_background "$root/offline.typescript"

"$YVEX_TEST_HOST" "$socket" 2>"$root/host.err" &
host_pid=$!
attempt=0
while test "$attempt" -lt 400 && test ! -S "$socket"; do
    kill -0 "$host_pid"
    attempt=$((attempt + 1))
    sleep 0.01
done
test -S "$socket"

# Deterministic commands remain outside alternate-screen mode.
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --reasoning high \
    --max-new-tokens 3 --strategy greedy REASONING_STREAM \
    >"$root/raw.out" 2>"$root/raw.err"
printf 'I need to compare the constraints...\nThe valid result is 42.' >"$root/raw.expected"
cmp "$root/raw.expected" "$root/raw.out"
! grep "$(printf '\033')" "$root/raw.out" >/dev/null

# Non-TTY interactive entrypoints refuse before mutating terminal state.
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
grep -F 'chat requires a terminal' "$root/non-tty.err" >/dev/null
grep -F 'chat requires a terminal' "$root/bare.err" >/dev/null
! grep "$(printf '\033')" "$root/non-tty.out" "$root/non-tty.err" \
    "$root/bare.out" "$root/bare.err" >/dev/null

# Explicit chat remains the linear console and never enters alternate-screen mode.
start_tui linear 24 100 'chat --session linear' nocolor
wait_for "$root/linear.typescript" 'deepseek4-v4-flash-dspark>'
printf 'hello\r' >&3
wait_for "$root/linear.typescript" 'hello from yvex'
printf '/quit\r' >&3
finish_tui
esc=$(printf '\033')
! grep -F "${esc}[?1049h" "$root/linear.typescript" >/dev/null
! grep -F "${esc}[?1049l" "$root/linear.typescript" >/dev/null
grep -F "${esc}[?2004h" "$root/linear.typescript" >/dev/null
grep -F "${esc}[?2004l" "$root/linear.typescript" >/dev/null

# Connected mode remains one transcript; slash discovery replaces screen tabs.
start_tui main 32 150 '' color
wait_for "$root/main.typescript" 'deepseek4-v4-flash-dspark'
wait_for "$root/main.typescript" '>_ YVEX'
wait_for "$root/main.typescript" 'Ask YVEX anything'
printf 'hello\r' >&3
wait_for "$root/main.typescript" 'hello from yvex'
printf '/sta' >&3
wait_for "$root/main.typescript" 'Commands'
printf '\t' >&3
wait_for "$root/main.typescript" '/status'
kill -INT "$client_pid"
wait_for "$root/main.typescript" 'Composer cleared'
printf '\033[200~hello\nworld 🌍\033[201~' >&3
wait_for "$root/main.typescript" 'world 🌍'
printf 'draft-resize' >&3
kill -WINCH "$client_pid"
wait_for "$root/main.typescript" 'draft-resize'
kill -INT "$client_pid"
wait_for "$root/main.typescript" 'Composer cleared'
kill -INT "$client_pid"
finish_tui
assert_restored "$root/main.typescript"
assert_terminal_background "$root/main.typescript"

# Active generation Ctrl-C crosses the canonical cancellation operation.
start_tui cancel 24 100 '' nocolor
wait_for "$root/cancel.typescript" 'deepseek4-v4-flash-dspark'
printf 'WAIT_PREFILL_CANCEL\r' >&3
wait_for "$root/cancel.typescript" 'Esc to interrupt'
kill -INT "$client_pid"
wait_for "$root/host.err" 'generation.cancel main'
wait_for "$root/cancel.typescript" 'native generation cancellation admitted'
kill -INT "$client_pid"
finish_tui
assert_restored "$root/cancel.typescript"

# Bare `yvex`, EOF, and SIGTERM all restore the terminal transaction.
start_tui compact 8 40 '' nocolor
printf '\020quit\r' >&3
finish_tui
assert_restored "$root/compact.typescript"

start_tui eof 18 88 '' nocolor
printf 'preserved-unsubmitted\004' >&3
finish_tui
assert_restored "$root/eof.typescript"

start_tui terminate 20 96 '' nocolor
kill -TERM "$client_pid"
finish_tui
assert_restored "$root/terminate.typescript"

echo 'TUI PTY lifecycle: pass'
