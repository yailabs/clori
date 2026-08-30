#!/bin/sh
# The product clients share one server model across two KV-reusing turns, preserve detached state,
# reset exactly, and close the runtime once. Runtime output remains in an untracked directory.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
ARTIFACT=${YVEX_MODEL_ARTIFACT:?YVEX_MODEL_ARTIFACT is required}
BINDING=${YVEX_RUNTIME_BINDING:?YVEX_RUNTIME_BINDING is required}
. tests/support/cleanup.sh

test -f "$ARTIFACT"
test -f "$BINDING"
root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-client-live.XXXXXX")
runtime="$root/runtime"
home="$root/home"
profile=deepseek4-v4-flash-dspark-runtime-iq2xxs
repl_prompt="$profile> "
mkdir -m 700 "$runtime" "$home"
mkdir -p "$home/.local/share/yvex"
cat >"$home/.local/share/yvex/models.local.json" <<EOF
{
  "schema": "yvex.models.local.v6",
  "models": [{
    "alias": "$profile",
    "path": "$ARTIFACT",
    "runtime_binding": "$BINDING",
    "runtime_target": "deepseek4-v4-flash-dspark",
    "runtime_backend": "cuda",
    "runtime_engine_kind": "text",
    "runtime_execution_strategy": "speculative",
    "runtime_context": 128
  }]
}
EOF
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
        XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server stop >/dev/null 2>&1 || true
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

HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server "$profile" \
    --backend cuda --ctx 128 --console raw --trace-level tokens \
    --openai off \
    >"$root/raw.jsonl" 2>"$root/daemon.err" &
daemon_pid=$!

ready=0
attempt=0
while test "$attempt" -lt 600; do
    if XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server status --json \
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

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server log >"$root/engine.log" &
watch_pid=$!
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server log --json \
    >"$root/trace.log" &
trace_pid=$!

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session new main >"$root/session.new"
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session main \
    --max-new-tokens 1 --strategy greedy Hi \
    >"$root/turn1" 2>"$root/turn1.metrics"
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session main \
    --max-new-tokens 1 --strategy greedy 'How are you?' \
    >"$root/turn2" 2>"$root/turn2.metrics"
grep -F '14 prompt/6 reused' "$root/turn2.metrics" >/dev/null

state_path="$root/main-state.yvex"
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session state save main "$state_path" \
    >"$root/state.save"
test -s "$state_path"
grep -E '^state checkpoint saved position=[1-9][0-9]* bytes=[1-9][0-9]* digest=[0-9a-f]{64}$' \
    "$root/state.save" >/dev/null
if XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session state restore main \
    "$state_path" 1 >"$root/state.restore.bounded" 2>&1; then
    exit 1
fi
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session state restore main \
    "$state_path" 1073741824 >"$root/state.restore"
grep -E '^state checkpoint restored position=[1-9][0-9]* bytes=[1-9][0-9]* digest=[0-9a-f]{64}$' \
    "$root/state.restore" >/dev/null
test "$(sed -E 's/^.* digest=//' "$root/state.save")" = \
    "$(sed -E 's/^.* digest=//' "$root/state.restore")"

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session detach main >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show main >"$root/detached"
grep -F 'detached' "$root/detached" >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session attach main >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server status --json >"$root/status.after.json"
grep -F '"model_open_count":1' "$root/status.after.json" >/dev/null
grep -E '"resident_device_bytes":[1-9][0-9]*' "$root/status.after.json" >/dev/null
grep -E '"output_head_upload_count":[01](,|})' "$root/status.after.json" >/dev/null

mkfifo "$root/repl.input"
NO_COLOR=1 XDG_RUNTIME_DIR="$runtime" script -q -f -e \
    -c "$YVEX_BIN --session repl-live --max-new-tokens 1" \
    "$root/repl.typescript" <"$root/repl.input" >/dev/null &
repl_pid=$!
exec 3>"$root/repl.input"
attempt=0
while test "$attempt" -lt 100; do
    test -f "$root/repl.typescript" && grep -F "$repl_prompt" "$root/repl.typescript" >/dev/null && break
    attempt=$((attempt + 1))
    sleep 0.1
done
test "$attempt" -lt 100
printf 'Hi\n' >&3
attempt=0
while test "$attempt" -lt 900; do
    prompts=$(grep -o "$repl_prompt" "$root/repl.typescript" 2>/dev/null | wc -l)
    test "$prompts" -ge 2 && break
    kill -0 "$repl_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.1
done
test "$attempt" -lt 900
printf '\004' >&3
exec 3>&-
wait "$repl_pid"
repl_pid=
grep -F '● ready' "$root/repl.typescript" >/dev/null
grep -F 'attached to resident runtime' "$root/repl.typescript" >/dev/null
grep -F 'session repl-live' "$root/repl.typescript" >/dev/null
grep -F 'generation' "$root/repl.typescript" >/dev/null
grep -F '1 tokens' "$root/repl.typescript" >/dev/null
grep -F 'prefill' "$root/repl.typescript" >/dev/null
! grep -F 'assistant>' "$root/repl.typescript" >/dev/null
! grep -F 'you>' "$root/repl.typescript" >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show repl-live >"$root/repl.session"
grep -F 'detached' "$root/repl.session" >/dev/null

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session reset main >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show main >"$root/reset"
grep -F 'position=0 turns=0' "$root/reset" >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session main \
    --max-new-tokens 1 --strategy greedy Hi \
    >"$root/turn1.after-reset" 2>"$root/turn1.after-reset.metrics"
test "$(sed -n '1p' "$root/turn1")" = "$(sed -n '1p' "$root/turn1.after-reset")"

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --max-new-tokens 1 \
    --strategy greedy Hello >"$root/oneshot"

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session new reasoning-live >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session reasoning-live \
    --max-new-tokens 64 --strategy greedy --reasoning high \
    'What is 2 plus 2? Answer briefly.' \
    >"$root/reasoning.out" 2>"$root/reasoning.metrics"
grep -E 'reasoning [1-9][0-9]* tokens' "$root/reasoning.metrics" >/dev/null
grep -E 'final [1-9][0-9]* tokens' "$root/reasoning.metrics" >/dev/null
! grep -F '</think>' "$root/reasoning.out" >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show reasoning-live \
    >"$root/reasoning.session"
grep -F 'detached' "$root/reasoning.session" >/dev/null

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

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show cancel-live >"$root/cancel.session"
grep -F 'partial' "$root/cancel.session" >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session reset cancel-live >/dev/null
XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session cancel-live \
    --max-new-tokens 1 --strategy greedy 'Continue after cancellation.' \
    >"$root/cancel.retry" 2>"$root/cancel.retry.metrics"
grep -F 'generation 1 tokens' "$root/cancel.retry.metrics" >/dev/null

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server status --json >"$root/status.final.json"
grep -F '"model_open_count":1' "$root/status.final.json" >/dev/null
grep -E '"cancelled_requests":[1-9][0-9]*' "$root/status.final.json" >/dev/null

XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server stop >/dev/null
wait "$daemon_pid"
daemon_pid=
wait "$watch_pid"
watch_pid=
wait "$trace_pid"
trace_pid=

grep -F '"kind":"runtime.ready"' "$root/raw.jsonl" >/dev/null
grep -F '"kind":"runtime.shutdown.complete"' "$root/raw.jsonl" |
    grep -F '"a":1,"b":1' >/dev/null
grep -E 'REQUEST[[:space:]]+main/' "$root/engine.log" >/dev/null
grep -E 'COMPLETE[[:space:]]+[1-9][0-9]* token' "$root/engine.log" >/dev/null
grep -F '"kind":"request.started"' "$root/trace.log" >/dev/null
grep -F '"kind":"generation.completed"' "$root/trace.log" >/dev/null
grep -F '"schema":3' "$root/trace.log" >/dev/null
! grep -E '(^|[[:space:]])[ab]=' "$root/engine.log" >/dev/null
! grep -E '(^|[[:space:]])[ab]=' "$root/trace.log" >/dev/null
grep -F '"kind":"generation.cancelled"' "$root/raw.jsonl" |
    grep -F '"session":"cancel-live"' >/dev/null
grep -F '"kind":"tokenizer.completed"' "$root/raw.jsonl" | grep -F '"a":14,"b":6' >/dev/null
! grep -F 'How are you?' "$root/raw.jsonl" >/dev/null
! grep -F 'How are you?' "$root/engine.log" >/dev/null
! test -e "$runtime/yvex/yvexd.sock"

printf 'test: runtime_client_refoundation_live\n'
