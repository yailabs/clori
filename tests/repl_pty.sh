#!/bin/sh
# Exercise the line editor and server-backed console through a real PTY.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEX_TEST_HOST=${YVEX_TEST_HOST:-build/tests/openai_host}
. tests/support/cleanup.sh

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-repl-pty.XXXXXX")
runtime="$root/runtime"
socket="$runtime/yvex/yvexd.sock"
host_pid=
repl_pid=
mkdir -m 700 "$runtime" "$runtime/yvex"
cleanup()
{
    status=$?
    if test -n "$repl_pid" && kill -0 "$repl_pid" 2>/dev/null; then
        kill "$repl_pid" 2>/dev/null || true
        wait "$repl_pid" 2>/dev/null || true
    fi
    if test -n "$host_pid" && kill -0 "$host_pid" 2>/dev/null; then
        kill "$host_pid" 2>/dev/null || true
        wait "$host_pid" 2>/dev/null || true
    fi
    yvex_test_cleanup "$root"
    return "$status"
}
trap cleanup EXIT HUP INT TERM

find_repl_client()
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

"$YVEX_TEST_HOST" "$socket" 2>"$root/host.err" &
host_pid=$!
attempt=0
while test "$attempt" -lt 100 && test ! -S "$socket"; do
    kill -0 "$host_pid"
    attempt=$((attempt + 1))
    sleep 0.01
done
test -S "$socket"

# Noninteractive output is the exact concatenation of canonical typed payloads;
# terminal separation and completion measurements belong on stderr.
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --reasoning high \
    --max-new-tokens 3 --strategy greedy REASONING_STREAM \
    >"$root/raw.out" 2>"$root/raw.err"
printf 'I need to compare the constraints...\nThe valid result is 42.' \
    >"$root/raw.expected"
cmp "$root/raw.expected" "$root/raw.out"
grep -F 'reasoning 2 tokens' "$root/raw.err" >/dev/null
grep -F 'final 1 tokens' "$root/raw.err" >/dev/null
! grep "$(printf '\033')" "$root/raw.out" >/dev/null

mkfifo "$root/input"
env -u NO_COLOR TERM=xterm-256color XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN chat --session pty" "$root/typescript" \
    <"$root/input" >"$root/stdout" 2>"$root/stderr" &
repl_pid=$!
exec 3>"$root/input"
attempt=0
while test "$attempt" -lt 100; do
    test -f "$root/typescript" && grep -F 'yvex>' "$root/typescript" >/dev/null && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'draft\014\177\177\177\177\177' >&3
printf '\033[200~hello\nworld 🌍\033[201~\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'hello from yvex' "$root/typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '\033[A\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    count=$(grep -c 'hello from yvex' "$root/typescript" 2>/dev/null || true)
    test "$count" -ge 2 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'MARKDOWN_STREAM\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    prompts=$(grep -c 'yvex>' "$root/typescript" 2>/dev/null || true)
    grep -F 'not-control' "$root/typescript" >/dev/null 2>&1 && \
        test "$prompts" -ge 4 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '/think\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    prompts=$(grep -c 'yvex>' "$root/typescript" 2>/dev/null || true)
    grep -F 'enabled for the next turn' "$root/typescript" \
        >/dev/null 2>&1 && test "$prompts" -ge 5 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '/nothink\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    prompts=$(grep -c 'yvex>' "$root/typescript" 2>/dev/null || true)
    grep -F 'disabled for the next turn' "$root/typescript" \
        >/dev/null 2>&1 && test "$prompts" -ge 6 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '/think-max\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    prompts=$(grep -c 'yvex>' "$root/typescript" 2>/dev/null || true)
    grep -F 'maximum for the next turn' "$root/typescript" \
        >/dev/null 2>&1 && test "$prompts" -ge 7 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'REASONING_STREAM\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'The valid result is 42.' "$root/typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'PARTIAL_FENCE\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'reset required (/reset)' "$root/typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '\004' >&3
exec 3>&-
wait "$repl_pid"
repl_pid=

esc=$(printf '\033')
clear=$(printf '\033[2J\033[H')
redrawn=$(printf '\033[2J\033[H\r\033[2K\033[38;5;81myvex>\033[0m draft')
sed "s/${esc}\\[[0-9;]*m//g" "$root/typescript" | tr -d '\r' \
    >"$root/typescript.plain"
grep -F 'YVEX 0.1.0 · protocol 8' "$root/typescript.plain" >/dev/null
grep -F '  model      deepseek4-v4-flash-dspark' \
    "$root/typescript.plain" >/dev/null
grep -F '  variant    dddddddddddd' "$root/typescript.plain" >/dev/null
grep -F '  runtime    ● ready · attached to resident runtime · CUDA · target-only' \
    "$root/typescript.plain" >/dev/null
grep -F '  session    pty · position 0 · turns 0' "$root/typescript.plain" >/dev/null
grep -F '  context    0/4096' "$root/typescript.plain" >/dev/null
grep -F '  memory     0.00 GiB host · 0.00 GiB device' \
    "$root/typescript.plain" >/dev/null
grep -F '  OpenAI     disabled' "$root/typescript.plain" >/dev/null
grep -Fx 'commands' "$root/typescript.plain" >/dev/null
grep -F '  /help        Discover canonical commands and operations.' \
    "$root/typescript.plain" >/dev/null
grep -F '  /status      Return one composed runtime and attached-session snapshot.' \
    "$root/typescript.plain" >/dev/null
grep -F '  /think       Enable explicit model-emitted reasoning for the next turn.' \
    "$root/typescript.plain" >/dev/null
grep -F '  /think-max   Enable the source-authored maximum reasoning policy.' \
    "$root/typescript.plain" >/dev/null
grep -F '  /nothink     Disable explicit model-emitted reasoning for the next turn.' \
    "$root/typescript.plain" >/dev/null
grep -F '  Ctrl-L       clear and redraw input' "$root/typescript.plain" >/dev/null
test "$(awk '/^commands$/ { catalog = 1; next }
             catalog && /^$/ { exit }
             catalog && /^  \// { count++ }
             END { print count + 0 }' "$root/typescript.plain")" -eq 17
! grep -F 'commands ·' "$root/typescript.plain" >/dev/null
grep -F "$clear" "$root/typescript" >/dev/null
grep -F "$redrawn" "$root/typescript" >/dev/null
grep -F 'yvex>' "$root/typescript" >/dev/null
grep -F 'processing 4 input tokens · 2/4 · 50.0%' "$root/typescript" >/dev/null
grep -F 'processing 4 input tokens · 4/4 · 100%' "$root/typescript" >/dev/null
grep -F 'hello from yvex' "$root/typescript" >/dev/null
grep -F '[cuda]' "$root/typescript.plain" >/dev/null
grep -F '__global__ void add() {' "$root/typescript.plain" >/dev/null
grep -F '🌍' "$root/typescript.plain" >/dev/null
grep -F 'Use int safely.' "$root/typescript.plain" >/dev/null
grep -F '\x1b[31mnot-control' "$root/typescript.plain" >/dev/null
grep -F 'I need to compare the constraints...' "$root/typescript.plain" >/dev/null
grep -F 'The valid result is 42.' "$root/typescript.plain" >/dev/null
grep -F 'reasoning · enabled for the next turn' "$root/typescript.plain" >/dev/null
grep -F 'reasoning · disabled for the next turn' "$root/typescript.plain" >/dev/null
grep -F 'reasoning · maximum for the next turn' "$root/typescript.plain" >/dev/null
grep -F 'int value = ' "$root/typescript.plain" >/dev/null
grep -F 'partial · 2 committed tokens · position 6 · reset required (/reset)' \
    "$root/typescript.plain" >/dev/null
! grep -F '```' "$root/typescript.plain" >/dev/null
! grep -F "${esc}[31mnot-control" "$root/typescript" >/dev/null
grep -F "${esc}[38;5;245mI need to compare the constraints..." \
    "$root/typescript" >/dev/null
grep -E '4 new/5 prompt/1 reused.*3 tokens.*TTFT 2\.50 s.*context 8/4096.*stop maximum tokens' \
    "$root/typescript" >/dev/null
! grep -F 'prefill      ' "$root/typescript" >/dev/null
! grep -F 'generation   ' "$root/typescript" >/dev/null
! grep -F 'KV unavailable' "$root/typescript" >/dev/null
! grep -F 'you>' "$root/typescript" >/dev/null
! grep -F 'assistant>' "$root/typescript" >/dev/null
grep -F "${esc}[38;5;81m" "$root/typescript" >/dev/null
grep -F "${esc}[38;5;114m" "$root/typescript" >/dev/null
grep -F "${esc}[?2004h" "$root/typescript" >/dev/null
grep -F "${esc}[?2004l" "$root/typescript" >/dev/null

# Ctrl-D discards an unfinished line and exits without submitting a turn.
mkfifo "$root/eof.input"
NO_COLOR=1 XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN chat --session eof-partial" "$root/eof.typescript" \
    <"$root/eof.input" >"$root/eof.stdout" 2>"$root/eof.stderr" &
repl_pid=$!
exec 3>"$root/eof.input"
attempt=0
while test "$attempt" -lt 100; do
    test -f "$root/eof.typescript" && \
        grep -F 'yvex> ' "$root/eof.typescript" >/dev/null && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'discard this\004' >&3
exec 3>&-
wait "$repl_pid"
repl_pid=
! grep -F 'hello from yvex' "$root/eof.typescript" >/dev/null
grep -F "${esc}[?2004l" "$root/eof.typescript" >/dev/null

# Ctrl-C uses the server cancellation operation during both prefill and decode,
# then returns to a restored prompt. The second idle Ctrl-C exits.
mkfifo "$root/cancel.input"
NO_COLOR=1 XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN chat --session cancellation" "$root/cancel.typescript" \
    <"$root/cancel.input" >"$root/cancel.stdout" 2>"$root/cancel.stderr" &
repl_pid=$!
exec 3>"$root/cancel.input"
attempt=0
while test "$attempt" -lt 100; do
    client_pid=$(find_repl_client || true)
    test -n "$client_pid" && grep -F 'yvex> ' "$root/cancel.typescript" \
        >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'WAIT_PREFILL_CANCEL\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'processing 4 input tokens · 0/4 · 0%' "$root/cancel.typescript" \
        >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
kill -INT "$client_pid"
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'cancelled' "$root/cancel.typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'WAIT_DECODE_CANCEL\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'processing 4 input tokens · 4/4 · 100%' "$root/cancel.typescript" \
        >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
kill -INT "$client_pid"
attempt=0
while test "$attempt" -lt 100; do
    count=$(grep -c 'cancelled' "$root/cancel.typescript" 2>/dev/null || true)
    test "$count" -ge 2 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'discard on resize' >&3
kill -WINCH "$client_pid"
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'discard on resize' "$root/cancel.typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
kill -INT "$client_pid"
attempt=0
while test "$attempt" -lt 100; do
    grep -F '^C' "$root/cancel.typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
kill -INT "$client_pid"
exec 3>&-
wait "$repl_pid"
repl_pid=
grep -F 'generation.cancel cancellation' "$root/host.err" >/dev/null
count=$(grep -c 'generation.cancel cancellation' "$root/host.err")
test "$count" -eq 2
grep -F "${esc}[?2004l" "$root/cancel.typescript" >/dev/null
! grep -F "${esc}[3" "$root/cancel.typescript" >/dev/null
! grep -F 'hello from yvex' "$root/cancel.typescript" >/dev/null

# Cancellation after an explicit reasoning fragment never fabricates a final
# channel, and leaves the terminal ready for the next request.
mkfifo "$root/reasoning-cancel.input"
NO_COLOR=1 XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN chat --session reasoning-cancellation" \
    "$root/reasoning-cancel.typescript" \
    <"$root/reasoning-cancel.input" >"$root/reasoning-cancel.stdout" \
    2>"$root/reasoning-cancel.stderr" &
repl_pid=$!
exec 3>"$root/reasoning-cancel.input"
attempt=0
while test "$attempt" -lt 100; do
    client_pid=$(find_repl_client || true)
    test -n "$client_pid" && \
        grep -F 'yvex> ' "$root/reasoning-cancel.typescript" \
            >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '/think\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'enabled for the next turn' \
        "$root/reasoning-cancel.typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'WAIT_REASONING_CANCEL\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'reasoning before cancellation' \
        "$root/reasoning-cancel.typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
kill -INT "$client_pid"
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'cancelled' "$root/reasoning-cancel.typescript" \
        >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
kill -INT "$client_pid"
exec 3>&-
wait "$repl_pid"
repl_pid=
grep -F 'generation.cancel reasoning-cancellation' "$root/host.err" \
    >/dev/null
! grep -F 'The valid result is 42.' "$root/reasoning-cancel.typescript" \
    >/dev/null

# A failure after reasoning but before final content preserves the typed
# reasoning fragment, ends its visual projection, and marks the turn partial.
mkfifo "$root/reasoning-partial.input"
NO_COLOR=1 XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN chat --session reasoning-partial" \
    "$root/reasoning-partial.typescript" \
    <"$root/reasoning-partial.input" >"$root/reasoning-partial.stdout" \
    2>"$root/reasoning-partial.stderr" &
repl_pid=$!
exec 3>"$root/reasoning-partial.input"
attempt=0
while test "$attempt" -lt 100; do
    test -f "$root/reasoning-partial.typescript" && \
        grep -F 'yvex> ' "$root/reasoning-partial.typescript" \
            >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '/think-max\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'maximum for the next turn' \
        "$root/reasoning-partial.typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'PARTIAL_REASONING\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'reset required (/reset)' "$root/reasoning-partial.typescript" \
        >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '\004' >&3
exec 3>&-
wait "$repl_pid"
repl_pid=
grep -F 'reasoning committed before failure' \
    "$root/reasoning-partial.typescript" >/dev/null
grep -F 'partial · 1 committed token · position 5 · reset required (/reset)' \
    "$root/reasoning-partial.typescript" >/dev/null
! grep -F 'The valid result is 42.' "$root/reasoning-partial.typescript" \
    >/dev/null

# Slash completion is projected from the canonical registry, not a second list.
mkfifo "$root/completion.input"
NO_COLOR=1 XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN chat --session completion" "$root/completion.typescript" \
    <"$root/completion.input" >"$root/completion.stdout" \
    2>"$root/completion.stderr" &
repl_pid=$!
exec 3>"$root/completion.input"
attempt=0
while test "$attempt" -lt 100; do
    test -f "$root/completion.typescript" && \
        grep -F 'yvex> ' "$root/completion.typescript" >/dev/null && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '/sta\t\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'live aaaaaaaaaaaa' "$root/completion.typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '/unknown\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'unknown command: /unknown' "$root/completion.typescript" \
        >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '/attach\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'invalid arguments for /attach' "$root/completion.typescript" \
        >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '\004' >&3
exec 3>&-
wait "$repl_pid"
repl_pid=
grep -F 'live aaaaaaaaaaaa' "$root/completion.typescript" >/dev/null

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server log >"$root/log"
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server log --json >"$root/log.jsonl"
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server status >"$root/status"
grep -F 'YVEX server ·' "$root/log" >/dev/null
grep -F 'server log · operational history and live events · Ctrl-C to stop' \
    "$root/log" >/dev/null
grep -E 'REQUEST[[:space:]]+fixture/fixture-request' "$root/log" >/dev/null
grep -E 'RUNTIME[[:space:]]+runtime shutdown complete' "$root/log" >/dev/null
! grep -F 'kernel launches 4511 · stream syncs 63' "$root/log" >/dev/null
! grep -F 'client disconnected' "$root/log" >/dev/null
! grep -E '^#[0-9]+' "$root/log" >/dev/null
! grep -E '(^|[[:space:]])[ab]=' "$root/log" >/dev/null
! grep -F "$esc" "$root/log" "$root/status" >/dev/null
grep -F '"schema":3' "$root/log.jsonl" >/dev/null
grep -F '"kind":"generation.profile"' "$root/log.jsonl" >/dev/null

# Losing the server during a turn restores the prompt and terminal before exit.
mkfifo "$root/disconnect.input"
NO_COLOR=1 XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN chat --session disconnect" "$root/disconnect.typescript" \
    <"$root/disconnect.input" >"$root/disconnect.stdout" \
    2>"$root/disconnect.stderr" &
repl_pid=$!
exec 3>"$root/disconnect.input"
attempt=0
while test "$attempt" -lt 100; do
    test -f "$root/disconnect.typescript" && \
        grep -F 'yvex> ' "$root/disconnect.typescript" >/dev/null && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf 'WAIT_DECODE_CANCEL\n' >&3
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'processing 4 input tokens · 4/4 · 100%' \
        "$root/disconnect.typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
kill "$host_pid"
wait "$host_pid" 2>/dev/null || true
host_pid=
attempt=0
while test "$attempt" -lt 100; do
    grep -F 'yvex:' "$root/disconnect.typescript" >/dev/null 2>&1 && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
printf '\004' >&3
exec 3>&-
wait "$repl_pid"
repl_pid=
grep -F "${esc}[?2004l" "$root/disconnect.typescript" >/dev/null

# A PTY selects chat, but connection refusal remains typed when no server exists.
set +e
printf '\004' | XDG_RUNTIME_DIR="$runtime" \
    script -q -e -c "$YVEX_BIN chat --session pty" "$root/absent.typescript" \
    >"$root/absent.stdout" 2>"$root/absent.stderr"
status=$?
set -e

test "$status" -eq 1
! grep -F 'chat requires a terminal' "$root/absent.typescript" >/dev/null
grep -F '`yvex server MODEL`' "$root/absent.typescript" >/dev/null
printf 'test: repl_pty\n'
