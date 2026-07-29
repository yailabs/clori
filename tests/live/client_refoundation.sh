#!/bin/sh
# Purpose: prove the exact product binaries share one daemon model across two KV-reusing turns,
# expose corresponding raw/operational events, preserve a detached session, reset exactly, and
# close the runtime once. All runtime output remains in an untracked temporary directory.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEXD_BIN=${YVEXD_BIN:-./yvexd}
ARTIFACT=${YVEX_MODEL_ARTIFACT:?YVEX_MODEL_ARTIFACT is required}
BINDING=${YVEX_RUNTIME_BINDING:?YVEX_RUNTIME_BINDING is required}
. tests/support/cleanup.sh

test -f "$ARTIFACT"
test -f "$BINDING"
root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-client-live.XXXXXX")
runtime="$root/runtime"
mkdir -m 700 "$runtime"
daemon_pid=
watch_pid=
trace_pid=
cancel_pid=
repl_pid=
cleanup()
{
    status=$?
    if test -n "$cancel_pid" && kill -0 "$cancel_pid" 2>/dev/null; then
        kill "$cancel_pid" 2>/dev/null || true
        wait "$cancel_pid" 2>/dev/null || true
    fi
    if test -n "$repl_pid" && kill -0 "$repl_pid" 2>/dev/null; then
        kill "$repl_pid" 2>/dev/null || true
        wait "$repl_pid" 2>/dev/null || true
    fi
    if test -n "$daemon_pid" && kill -0 "$daemon_pid" 2>/dev/null; then
        XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime stop >/dev/null 2>&1 || true
        kill "$daemon_pid" 2>/dev/null || true
        wait "$daemon_pid" 2>/dev/null || true
    fi
    test -z "$watch_pid" || wait "$watch_pid" 2>/dev/null || true
    test -z "$trace_pid" || wait "$trace_pid" 2>/dev/null || true
    if test "$status" -ne 0; then
        printf 'client refoundation live failure; diagnostics: %s\n' "$root" >&2
        for file in status.json status.after.json status.final.json daemon.err \
            cancel.err cancel.out; do
            test ! -f "$root/$file" || {
                printf '%s\n' "[$file]" >&2
                tail -40 "$root/$file" >&2
            }
        done
    fi
    if test "${YVEX_KEEP_TEST_OUTPUT:-0}" = 1; then
        printf 'client refoundation live output retained: %s\n' "$root" >&2
    else
        yvex_test_cleanup "$root"
    fi
    return "$status"
}
trap cleanup EXIT HUP INT TERM

XDG_RUNTIME_DIR="$runtime" "$YVEXD_BIN" \
    --model "$ARTIFACT" --runtime-binding "$BINDING" \
    --backend cuda --context 128 --console raw --trace-level tokens \
    >"$root/raw.jsonl" 2>"$root/daemon.err" &
daemon_pid=$!

ready=0
attempt=0
while test "$attempt" -lt 600; do
    if XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime status --json \
        >"$root/status.json" 2>/dev/null; then
        ready=1
        break
    fi
    kill -0 "$daemon_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 1
done
test "$ready" -eq 1 || {
    sed -n '1,80p' "$root/daemon.err" >&2
    exit 1
}
grep -F '"model_open_count":1' "$root/status.json" >/dev/null

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime watch >"$root/engine.log" &
watch_pid=$!
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime trace --follow \
    >"$root/trace.jsonl" &
trace_pid=$!

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session new main >"$root/session.new"
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session main \
    --max-new-tokens 1 --strategy greedy Hi >"$root/turn1"
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session main \
    --max-new-tokens 1 --strategy greedy 'How are you?' >"$root/turn2"
grep -F '14 prompt · 6 reused' "$root/turn2" >/dev/null

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session detach main >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show main >"$root/detached"
grep -F 'detached' "$root/detached" >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session attach main >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime status --json >"$root/status.after.json"
grep -F '"model_open_count":1' "$root/status.after.json" >/dev/null
grep -E '"resident_device_bytes":[1-9][0-9]*' "$root/status.after.json" >/dev/null
grep -F '"output_head_upload_count":1' "$root/status.after.json" >/dev/null

mkfifo "$root/repl.input"
XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN chat --session repl-live --max-new-tokens 1" \
    "$root/repl.typescript" <"$root/repl.input" >/dev/null &
repl_pid=$!
exec 3>"$root/repl.input"
attempt=0
while test "$attempt" -lt 100; do
    test -f "$root/repl.typescript" && grep -F 'you>' "$root/repl.typescript" >/dev/null && break
    attempt=$((attempt + 1))
    sleep 0.1
done
test "$attempt" -lt 100
printf 'Hi\n' >&3
attempt=0
while test "$attempt" -lt 900; do
    prompts=$(grep -o 'you>' "$root/repl.typescript" 2>/dev/null | wc -l)
    test "$prompts" -ge 2 && break
    kill -0 "$repl_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.1
done
test "$attempt" -lt 900
printf '/quit\n' >&3
exec 3>&-
wait "$repl_pid"
repl_pid=
grep -F 'YVEX · local runtime · session repl-live' "$root/repl.typescript" >/dev/null
grep -F 'assistant>' "$root/repl.typescript" >/dev/null
grep -F '1 generated' "$root/repl.typescript" >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show repl-live >"$root/repl.session"
grep -F 'detached' "$root/repl.session" >/dev/null

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session reset main >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show main >"$root/reset"
grep -F 'position=0 turns=0' "$root/reset" >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session main \
    --max-new-tokens 1 --strategy greedy Hi >"$root/turn1.after-reset"
test "$(sed -n '1p' "$root/turn1")" = "$(sed -n '1p' "$root/turn1.after-reset")"

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --max-new-tokens 1 \
    --strategy greedy Hello >"$root/oneshot"

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session new cancel-live >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session cancel-live \
    --max-new-tokens 16 --strategy greedy 'Explain attention briefly.' \
    >"$root/cancel.out" 2>"$root/cancel.err" &
cancel_pid=$!
cancel_started=0
attempt=0
while test "$attempt" -lt 200; do
    if grep -F '"kind":"request.started"' "$root/raw.jsonl" |
        grep -F '"session":"cancel-live"' >/dev/null; then
        cancel_started=1
        break
    fi
    kill -0 "$cancel_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.1
done
test "$cancel_started" -eq 1
kill -INT "$cancel_pid"
set +e
wait "$cancel_pid"
cancel_status=$?
set -e
cancel_pid=
test "$cancel_status" -eq 130

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime status --json >"$root/status.final.json"
grep -F '"model_open_count":1' "$root/status.final.json" >/dev/null
grep -E '"cancelled_requests":[1-9][0-9]*' "$root/status.final.json" >/dev/null

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" runtime stop >/dev/null
wait "$daemon_pid"
daemon_pid=
wait "$watch_pid"
watch_pid=
wait "$trace_pid"
trace_pid=

grep -F '"kind":"runtime.ready"' "$root/raw.jsonl" >/dev/null
grep -F '"kind":"runtime.shutdown.complete"' "$root/raw.jsonl" |
    grep -F '"a":1,"b":1' >/dev/null
grep -F 'request.started' "$root/engine.log" >/dev/null
grep -F 'generation.completed' "$root/engine.log" >/dev/null
grep -F '"kind":"generation.completed"' "$root/trace.jsonl" >/dev/null
grep -F '"kind":"generation.cancelled"' "$root/raw.jsonl" |
    grep -F '"session":"cancel-live"' >/dev/null
grep -F '"kind":"tokenizer.completed"' "$root/raw.jsonl" | grep -F '"a":14,"b":6' >/dev/null
! grep -F 'How are you?' "$root/raw.jsonl" >/dev/null
! grep -F 'How are you?' "$root/engine.log" >/dev/null
! test -e "$runtime/yvex/yvexd.sock"

printf 'test: runtime_client_refoundation_live\n'
