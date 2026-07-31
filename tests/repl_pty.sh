#!/bin/sh
# Exercise the line editor and daemon-backed console through a real PTY.
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

mkfifo "$root/input"
XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN chat --session pty" "$root/typescript" \
    <"$root/input" >"$root/stdout" 2>"$root/stderr" &
repl_pid=$!
exec 3>"$root/input"
attempt=0
while test "$attempt" -lt 100; do
    test -f "$root/typescript" && grep -F 'yvex> ' "$root/typescript" >/dev/null && break
    attempt=$((attempt + 1))
    sleep 0.01
done
test "$attempt" -lt 100
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
printf '\004' >&3
exec 3>&-
wait "$repl_pid"
repl_pid=

grep -F 'YVEX ' "$root/typescript" >/dev/null
grep -F 'deepseek-v4-flash · CUDA · variant dddddddddddd' "$root/typescript" >/dev/null
grep -F 'runtime ready · attached to resident runtime' "$root/typescript" >/dev/null
grep -F 'session pty · position 0 · turns 0 · context 0/4096' "$root/typescript" >/dev/null
grep -F 'yvex> ' "$root/typescript" >/dev/null
grep -F 'processing 4 input tokens · 2/4 · 50.0%' "$root/typescript" >/dev/null
grep -F 'processing 4 input tokens · 4/4 · 100%' "$root/typescript" >/dev/null
grep -F 'hello from yvex' "$root/typescript" >/dev/null
grep -F 'prefill      4 new · 5 prompt · 1 reused · 2.00 s · 2.00 tok/s' \
    "$root/typescript" >/dev/null
grep -F 'generation   3 tokens · 3.00 s · 1.00 tok/s' "$root/typescript" >/dev/null
grep -F 'TTFT         2.50 s' "$root/typescript" >/dev/null
grep -F 'context      8 / 4096' "$root/typescript" >/dev/null
grep -F 'stop         maximum tokens' "$root/typescript" >/dev/null
! grep -F 'you>' "$root/typescript" >/dev/null
! grep -F 'assistant>' "$root/typescript" >/dev/null
esc=$(printf '\033')
grep -F "${esc}[?2004h" "$root/typescript" >/dev/null
grep -F "${esc}[?2004l" "$root/typescript" >/dev/null

# Ctrl-D discards an unfinished line and exits without submitting a turn.
mkfifo "$root/eof.input"
XDG_RUNTIME_DIR="$runtime" script -q -f -e \
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
    grep -F '[cancelled]' "$root/cancel.typescript" >/dev/null 2>&1 && break
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
    count=$(grep -c '\[cancelled\]' "$root/cancel.typescript" 2>/dev/null || true)
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

# Slash completion is projected from the canonical registry, not a second list.
mkfifo "$root/completion.input"
XDG_RUNTIME_DIR="$runtime" script -q -f -e \
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
    grep -F 'live model   aaaa' "$root/completion.typescript" >/dev/null 2>&1 && break
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
grep -F 'live model   aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
    "$root/completion.typescript" >/dev/null

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime watch >"$root/watch"
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime trace >"$root/trace"
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime trace --json >"$root/trace.jsonl"
grep -F 'request started · session fixture · request fixture-request' "$root/watch" >/dev/null
grep -F 'kernel launches 4511 · stream syncs 63' "$root/watch" >/dev/null
! grep -E '^#[0-9]+' "$root/watch" >/dev/null
grep -E '^#[0-9]+ info[[:space:]]+request started' "$root/trace" >/dev/null
grep -F 'phase launches · kernel launches 4511 · stream syncs 63' "$root/trace" >/dev/null
! grep -E '(^|[[:space:]])[ab]=' "$root/watch" >/dev/null
! grep -E '(^|[[:space:]])[ab]=' "$root/trace" >/dev/null
grep -F '"schema":2' "$root/trace.jsonl" >/dev/null
grep -F '"kind":"generation.profile"' "$root/trace.jsonl" >/dev/null

# Losing yvexd during a turn restores the prompt and terminal before exit.
mkfifo "$root/disconnect.input"
XDG_RUNTIME_DIR="$runtime" script -q -f -e \
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

# A PTY selects chat, but connection refusal remains typed when no daemon exists.
set +e
printf '\004' | XDG_RUNTIME_DIR="$runtime" \
    script -q -e -c "$YVEX_BIN chat --session pty" "$root/absent.typescript" \
    >"$root/absent.stdout" 2>"$root/absent.stderr"
status=$?
set -e

test "$status" -eq 1
! grep -F 'chat requires a terminal' "$root/absent.typescript" >/dev/null
grep -F '`yvex runtime start`' "$root/absent.typescript" >/dev/null
printf 'test: repl_pty\n'
