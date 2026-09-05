#!/bin/sh
# Exercise the scrollback-preserving console and deterministic CLI isolation through real PTYs.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEX_TEST_HOST=${YVEX_TEST_HOST:-build/tests/openai_host}
YVEX_TEST_PTY_WAIT_ATTEMPTS=${YVEX_TEST_PTY_WAIT_ATTEMPTS:-2000}
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
    while test "$attempt" -lt "$YVEX_TEST_PTY_WAIT_ATTEMPTS"; do
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
    while test "$attempt" -lt "$YVEX_TEST_PTY_WAIT_ATTEMPTS"; do
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
    while test "$attempt" -lt "$YVEX_TEST_PTY_WAIT_ATTEMPTS" &&
        test ! -S "$socket"; do
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
    while test "$attempt" -lt "$YVEX_TEST_PTY_WAIT_ATTEMPTS" &&
        test -e "$socket"; do
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
    while test "$attempt" -lt "$YVEX_TEST_PTY_WAIT_ATTEMPTS"; do
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

# Bare YVEX is deterministic help; chat is explicit and terminal-bound.
set +e
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" chat </dev/null \
    >"$root/non-tty.out" 2>"$root/non-tty.err"
chat_status=$?
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" </dev/null \
    >"$root/bare.out" 2>"$root/bare.err"
bare_status=$?
set -e
test "$chat_status" -eq 2
test "$bare_status" -eq 0
grep -F 'chat requires a terminal' "$root/non-tty.err" >/dev/null
grep -F 'YVEX inference runtime' "$root/bare.out" >/dev/null
! grep "$(printf '\033')" "$root/non-tty.out" "$root/non-tty.err" \
    "$root/bare.out" "$root/bare.err" >/dev/null

# An interactive chat with no host fails as a client and gives the host-start action.
set +e
XDG_RUNTIME_DIR="$runtime" NO_COLOR=1 TERM=xterm-256color \
    script -q -e -c "$YVEX_BIN chat" "$root/no-host.typescript" </dev/null \
    >"$root/no-host.stdout" 2>"$root/no-host.stderr"
no_host_status=$?
set -e
test "$no_host_status" -eq 1
grep -F 'start one with:' "$root/no-host.typescript" >/dev/null
grep -F 'yvex serve' "$root/no-host.typescript" >/dev/null

# Retired one-shot generation refuses and never contacts or starts a host.
set +e
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run REASONING_STREAM \
    >"$root/run.out" 2>"$root/run.err"
run_status=$?
set -e
test "$run_status" -eq 2
grep -F 'removed command: run' "$root/run.err" >/dev/null

start_host
XDG_RUNTIME_DIR="$runtime" YVEX_MODELS_REGISTRY="$registry" \
    YVEX_MODELS_ROOT="$models" YVEX_CONFIG_DIR="$config" XDG_CONFIG_HOME="$config" \
    python3 tests/replai_consumer.py --binary "$YVEX_BIN" --host-log "$root/host.err" \
    --output "$root" ${YVEX_REPL_MEMCHECK:+--memcheck "$YVEX_REPL_MEMCHECK"}
image="$root/attachment.png"
audio="$root/attachment.wav"
printf '\211PNG\r\n\032\nfixture' >"$image"
printf 'RIFF\004\000\000\000WAVE' >"$audio"

# Explicit chat preserves scrollback, streams output, and restores bracketed paste mode.
start_console explicit 24 100 'chat --session linear' nocolor
wait_for "$root/explicit.typescript" 'commands'
wait_for "$root/explicit.typescript" '/help'
wait_for "$root/explicit.typescript" '/status'
wait_for "$root/explicit.typescript" '/context'
wait_for "$root/explicit.typescript" '/cancel'
wait_for "$root/explicit.typescript" '/sessions'
wait_for "$root/explicit.typescript" '/attach'
wait_for "$root/explicit.typescript" '/attachments'
wait_for "$root/explicit.typescript" '/use'
wait_for "$root/explicit.typescript" '/reset'
wait_for "$root/explicit.typescript" '/quit'
printf '/attach %s\r' "$image" >&3
wait_for "$root/explicit.typescript" 'attached · image'
printf '/attach %s\r' "$audio" >&3
wait_for "$root/explicit.typescript" 'next turn 2/31'
printf '/attachments\r' >&3
wait_for "$root/explicit.typescript" 'attachments · 2 staged for next turn'
printf 'hello\r' >&3
wait_for "$root/explicit.typescript" 'hello from yvex'
wait_for "$root/explicit.typescript" 'output adaptive · envelope 256'
wait_for "$root/explicit.typescript" 'stop EOS'
printf '/attachments\r' >&3
wait_for "$root/explicit.typescript" 'attachments · none staged'
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/explicit.typescript"

# Omitted completion length stays adaptive/server-owned; an explicit limit is
# carried separately and its terminal stop is distinguishable from EOS.
start_console envelope 24 100 'chat --session envelope --max-new-tokens 3' nocolor
printf 'explicit envelope\r' >&3
wait_for "$root/envelope.typescript" 'output explicit 3 · envelope 3'
wait_for "$root/envelope.typescript" 'stop maximum tokens'
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/envelope.typescript"

# A fragment without a newline is visible while the provider is still working;
# rendering adds neither an artificial typewriter delay nor line buffering.
start_console progressive 24 100 'chat --session progressive' nocolor
printf 'PROGRESSIVE_STREAM\r' >&3
wait_for "$root/progressive.typescript" 'Letters arrive now'
! grep -F 'and finish later.' "$root/progressive.typescript" >/dev/null
wait_for "$root/progressive.typescript" 'Letters arrive now and finish later.'
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/progressive.typescript"

# Typed channels, fragmented UTF-8 and bounded Markdown stay readable while
# ordinary text streams immediately. Reasoning remains visually secondary.
start_console rendering 24 100 'chat --session rendering' nocolor
printf 'MARKDOWN_STREAM\r' >&3
wait_for "$root/rendering.typescript" 'deterministic transcript identity.'
! grep -F 'answer' "$root/rendering.typescript" >/dev/null
grep -F 'CUDA' "$root/rendering.typescript" >/dev/null
grep -F 'code · cuda' "$root/rendering.typescript" >/dev/null
grep -F '  // 🌍' "$root/rendering.typescript" >/dev/null
grep -F 'Use int safely.' "$root/rendering.typescript" >/dev/null
grep -F 'ESC: \x1b[31mnot-control' "$root/rendering.typescript" >/dev/null
grep -F 'A readable paragraph stays within the terminal prose measure' \
    "$root/rendering.typescript" >/dev/null
grep -F 'protocol consumers and deterministic transcript identity.' \
    "$root/rendering.typescript" >/dev/null
! grep -F '# CUDA' "$root/rendering.typescript" >/dev/null
! grep -F '```cuda' "$root/rendering.typescript" >/dev/null
printf 'REASONING_STREAM\r' >&3
wait_for "$root/rendering.typescript" 'The valid result is 42.'
grep -F 'reasoning' "$root/rendering.typescript" >/dev/null
grep -F '│ Plan' "$root/rendering.typescript" >/dev/null
grep -F '│ • Compare constraints carefully.' "$root/rendering.typescript" >/dev/null
! grep -F 'answer' "$root/rendering.typescript" >/dev/null
grep -F 'Result' "$root/rendering.typescript" >/dev/null
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/rendering.typescript"

# During an active response the terminal has one output owner. Editing bytes are
# neither echoed into model output nor carried into the following prompt.
start_console async 24 100 'chat --session async' nocolor
printf 'WAIT_ASYNC_KEYS\r' >&3
wait_for "$root/async.typescript" 'processing 4 input tokens · 4/4'
cycle=0
while test "$cycle" -lt 3; do
    printf '\033[A\033[B\033[C\033[D\033[H\033[F\033[3~async-keys-🌍\177\014' >&3
    kill -WINCH "$client_pid"
    cycle=$((cycle + 1))
done
wait_for "$root/async.typescript" 'hello from yvex'
! grep -F 'async-keys-🌍' "$root/async.typescript" >/dev/null
for suffix in '[A' '[B' '[C' '[D' '[H' '[F' '[3~'; do
    ! grep -F "$(printf '\033%s' "$suffix")" "$root/async.typescript" >/dev/null
done
printf 'after async\r' >&3
wait_count "$root/async.typescript" 'hello from yvex' 2
! grep -F 'INPUT_CONTAMINATED' "$root/async.typescript" >/dev/null
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/async.typescript"

# Exercise completion, UTF-8 editing, paste, history, and resize in the one public REPL.
start_console bare 32 150 'chat' color
printf '/sta\t\r' >&3
wait_for "$root/bare.typescript" '/status'
wait_count "$root/bare.typescript" "$(printf '\033[?2004h')" 2
printf '\033[200~hello\nworld 🌍\033[201~\r' >&3
wait_count "$root/bare.typescript" 'hello from yvex' 1
wait_count "$root/bare.typescript" "$(printf '\033[?2004h')" 3
printf '\033[A\r' >&3
wait_count "$root/bare.typescript" 'hello from yvex' 2
wait_count "$root/bare.typescript" "$(printf '\033[?2004h')" 4
printf 'world\033[Hhello \033[F 🌍\r' >&3
wait_for "$root/bare.typescript" 'hello world'
wait_count "$root/bare.typescript" 'hello from yvex' 3
wait_count "$root/bare.typescript" "$(printf '\033[?2004h')" 5
printf 'draft-resize' >&3
kill -WINCH "$client_pid"
wait_for "$root/bare.typescript" 'draft-resize'
kill -INT "$client_pid"
wait_for "$root/bare.typescript" '^C'
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/bare.typescript"

# A transport loss leaves the draft loop alive; the next request reconnects to a restarted host.
start_console reconnect 24 100 'chat --session reconnect' nocolor
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
start_console cancel 24 100 'chat --session cancel' nocolor
printf 'WAIT_PREFILL_CANCEL\r' >&3
wait_for "$root/cancel.typescript" 'processing 4 input tokens · 0/4'
kill -INT "$client_pid"
wait_for "$root/host.err" 'generation.cancel cancel'
wait_for "$root/cancel.typescript" 'cancelled'
printf '/quit\r' >&3
finish_console
assert_linear_terminal "$root/cancel.typescript"

# Ctrl-D exits with normal terminal restoration and preserved scrollback.
start_console eof 18 88 'chat --session eof' nocolor
printf '\004' >&3
finish_console
assert_linear_terminal "$root/eof.typescript"

echo 'linear console PTY lifecycle: pass'
