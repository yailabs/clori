#!/bin/sh
# Exercise the production artifact-to-CLI path with no external model or accelerator.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
TINY_COMPILER=${TINY_COMPILER:-build/tests/tiny_compile}
TINY_GENERATOR=${TINY_GENERATOR:-tests/integration/tiny_model.py}
. tests/support/cleanup.sh

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-tiny-vertical.XXXXXX")
runtime="$root/runtime"
home="$root/home"
first="$root/first"
second="$root/second"
corrupt="$root/corrupt"
mkdir -m 700 "$runtime" "$home" "$first" "$second"
mkdir -p "$home/.local/share/yvex" "$first/bindings" "$second/bindings" "$corrupt"
server_pid=
log_pid=

cleanup()
{
    status=$?
    trap - EXIT HUP INT TERM
    if test -n "$server_pid" && kill -0 "$server_pid" 2>/dev/null; then
        HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server stop \
            >/dev/null 2>&1 || true
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if test -n "$log_pid"; then
        wait "$log_pid" 2>/dev/null || true
    fi
    if test "$status" -ne 0 || test "${YVEX_KEEP_TEST_OUTPUT:-0}" = 1; then
        printf 'tiny vertical output: %s\n' "$root" >&2
        test ! -f "$root/server.err" || tail -80 "$root/server.err" >&2
        test ! -f "$root/run.err" || tail -80 "$root/run.err" >&2
    else
        yvex_test_cleanup "$root"
    fi
    exit "$status"
}
trap cleanup EXIT HUP INT TERM

python3 "$TINY_GENERATOR" "$first/tiny.gguf"
python3 "$TINY_GENERATOR" "$second/tiny.gguf"
cmp "$first/tiny.gguf" "$second/tiny.gguf"
"$TINY_COMPILER" "$first/tiny.gguf" "$first/bindings" >"$first/compile.out"
"$TINY_COMPILER" "$second/tiny.gguf" "$second/bindings" >"$second/compile.out"

first_artifact=$(sed -n 's/^artifact_identity=//p' "$first/compile.out")
second_artifact=$(sed -n 's/^artifact_identity=//p' "$second/compile.out")
first_binding=$(sed -n 's/^binding_identity=//p' "$first/compile.out")
second_binding=$(sed -n 's/^binding_identity=//p' "$second/compile.out")
binding_path=$(sed -n 's/^binding_path=//p' "$first/compile.out")
test -n "$first_artifact" && test "$first_artifact" = "$second_artifact"
test -n "$first_binding" && test "$first_binding" = "$second_binding"
test -f "$binding_path"

artifact=$(realpath "$first/tiny.gguf")
binding=$(realpath "$binding_path")
if "$YVEX_BIN" execute transformer generate \
    --target tiny-executable --artifact "$artifact" \
    --runtime-binding "$binding" --backend cpu \
    --generation-mode target-only --text a --context-capacity 9 \
    --prefill-chunk-tokens 1 --max-new-tokens 1 --max-output-bytes 16 \
    --strategy greedy --progress off --output json \
    >"$root/context-refusal.json" 2>"$root/context-refusal.err"; then
    printf 'oversized tiny context was admitted\n' >&2
    exit 1
fi
grep -F '"status": "refused"' "$root/context-refusal.json" >/dev/null
grep -F '"reason": "requested context exceeds the model-authored semantic maximum"' \
    "$root/context-refusal.json" >/dev/null
cat >"$home/.local/share/yvex/models.local.json" <<EOF
{
  "schema": "yvex.models.local.v3",
  "models": [{
    "alias": "tiny-executable-cpu-complete",
    "path": "$artifact",
    "runtime_binding": "$binding",
    "runtime_target": "tiny-executable",
    "runtime_backend": "cpu",
    "runtime_mode": "target-only",
    "runtime_context": 8
  }]
}
EOF

HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server tiny-executable-cpu-complete \
    --backend cpu --ctx 8 --generation-mode target-only --max-new-tokens 1 \
    --parallel 2 --console off --openai off >"$root/server.out" 2>"$root/server.err" &
server_pid=$!

ready=0
attempt=0
while test "$attempt" -lt 100; do
    if HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server status --json \
        >"$root/status.json" 2>"$root/status.err"; then
        ready=1
        break
    fi
    kill -0 "$server_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.05
done
test "$ready" -eq 1
grep -F '"model_open_count":1' "$root/status.json" >/dev/null
grep -F '"context_capacity":8' "$root/status.json" >/dev/null
grep -F '"parallel":2' "$root/status.json" >/dev/null
grep -F '"independent_session_scheduling":true' "$root/status.json" >/dev/null
grep -F '"continuous_batching":false' "$root/status.json" >/dev/null
grep -E '"capacity_plan_identity":"[0-9a-f]{64}"' "$root/status.json" >/dev/null
grep -F 'requested ctx=8 · parallel=2' "$root/server.out" >/dev/null

HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server log --json \
    >"$root/server.log.jsonl" 2>"$root/server.log.err" &
log_pid=$!
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session new persisted \
    >"$root/session.new"
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session new independent \
    >"$root/session.independent.new"
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session persisted a \
    --strategy greedy --max-new-tokens 1 >"$root/run.out" 2>"$root/run.err" &
first_run_pid=$!
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session independent a \
    --strategy greedy --max-new-tokens 1 \
    >"$root/run.independent.out" 2>"$root/run.independent.err" &
second_run_pid=$!
wait "$first_run_pid"
wait "$second_run_pid"
grep -Fx 'ok' "$root/run.out" >/dev/null
grep -Fx 'ok' "$root/run.independent.out" >/dev/null
grep -F 'generation 1 token' "$root/run.err" >/dev/null
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show persisted \
    >"$root/prefix.source.before"
if HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session fork \
    persisted fork-too-small 1 >"$root/prefix.small.out" \
    2>"$root/prefix.small.err"; then
    printf 'bounded prefix fork unexpectedly succeeded\n' >&2
    exit 1
fi
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session fork \
    persisted forked 1048576 >"$root/prefix.fork.out"
grep -E '^forked[[:space:]]+(ready|detached)[[:space:]]+position=[1-9][0-9]* turns=[1-9][0-9]*$' \
    "$root/prefix.fork.out" >/dev/null
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show forked \
    >"$root/prefix.child.before"
source_position=$(sed -n 's/^.*position=\([0-9][0-9]*\).*$/\1/p' \
    "$root/prefix.source.before")
child_position=$(sed -n 's/^.*position=\([0-9][0-9]*\).*$/\1/p' \
    "$root/prefix.child.before")
test -n "$source_position" && test "$source_position" = "$child_position"
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session forked a \
    --strategy greedy --max-new-tokens 1 \
    >"$root/run.forked.out" 2>"$root/run.forked.err"
grep -Fx 'ok' "$root/run.forked.out" >/dev/null
grep -E '[1-9][0-9]* reused' "$root/run.forked.err" >/dev/null
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show persisted \
    >"$root/prefix.source.after"
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show forked \
    >"$root/prefix.child.after"
test "$source_position" = "$(sed -n 's/^.*position=\([0-9][0-9]*\).*$/\1/p' \
    "$root/prefix.source.after")"
test "$(sed -n 's/^.*position=\([0-9][0-9]*\).*$/\1/p' \
    "$root/prefix.child.after")" -gt "$child_position"
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session new reasoning-limit \
    >"$root/session.reasoning.new"
if HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run \
    --session reasoning-limit --reasoning high --strategy greedy \
    --max-new-tokens 1 a >"$root/reasoning-limit.out" \
    2>"$root/reasoning-limit.err"; then
    printf 'unfinished tiny reasoning unexpectedly succeeded\n' >&2
    exit 1
fi
grep -F 'thinking ended before its source delimiter; reset and retry with a larger token limit' \
    "$root/reasoning-limit.err" >/dev/null
grep -F 'partial · 1 committed token' "$root/reasoning-limit.err" >/dev/null
state_path="$root/persisted-state.yvex"
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session state save \
    persisted "$state_path" >"$root/state.save"
test -s "$state_path"
grep -E '^state checkpoint saved position=[1-9][0-9]* bytes=[1-9][0-9]* digest=[0-9a-f]{64}$' \
    "$root/state.save" >/dev/null
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show persisted \
    >"$root/session.before"

HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server stop >/dev/null
wait "$server_pid"
server_pid=
wait "$log_pid"
log_pid=
grep -F '"kind":"generation.completed"' "$root/server.log.jsonl" >/dev/null
grep -F '"phase":"graphs"' "$root/server.log.jsonl" >/dev/null
grep -F '"phase":"tensorcore"' "$root/server.log.jsonl" >/dev/null

HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server tiny-executable-cpu-complete \
    --backend cpu --ctx 8 --generation-mode target-only --max-new-tokens 1 \
    --parallel 2 --console off --openai off >>"$root/server.out" 2>>"$root/server.err" &
server_pid=$!
ready=0
attempt=0
while test "$attempt" -lt 100; do
    if HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server status --json \
        >"$root/status.restart.json" 2>"$root/status.restart.err"; then
        ready=1
        break
    fi
    kill -0 "$server_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.05
done
test "$ready" -eq 1
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session new persisted \
    >"$root/session.restart.new"
if HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session state restore \
    persisted "$state_path" 1 >"$root/state.restore.bounded" 2>&1; then
    printf 'bounded state restore unexpectedly succeeded\n' >&2
    exit 1
fi
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session state restore \
    persisted "$state_path" 1048576 >"$root/state.restore"
grep -E '^state checkpoint restored position=[1-9][0-9]* bytes=[1-9][0-9]* digest=[0-9a-f]{64}$' \
    "$root/state.restore" >/dev/null
test "$(sed -n 's/^.* digest=//p' "$root/state.save")" = \
    "$(sed -n 's/^.* digest=//p' "$root/state.restore")"
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" session show persisted \
    >"$root/session.after"
test "$(sed -n 's/^.*position=\([0-9][0-9]*\) turns=\([0-9][0-9]*\).*$/\1:\2/p' \
        "$root/session.before")" = \
    "$(sed -n 's/^.*position=\([0-9][0-9]*\) turns=\([0-9][0-9]*\).*$/\1:\2/p' \
        "$root/session.after")"
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run --session persisted a \
    --strategy greedy --max-new-tokens 1 \
    >"$root/run.after-restore.out" 2>"$root/run.after-restore.err"
grep -Fx 'ok' "$root/run.after-restore.out" >/dev/null
grep -E '[1-9][0-9]* reused' "$root/run.after-restore.err" >/dev/null
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server stop >/dev/null
wait "$server_pid"
server_pid=

python3 "$TINY_GENERATOR" "$corrupt/tiny.gguf" --corrupt
corrupt_artifact=$(realpath "$corrupt/tiny.gguf")
cat >"$home/.local/share/yvex/models.local.json" <<EOF
{
  "schema": "yvex.models.local.v3",
  "models": [{
    "alias": "tiny-executable-cpu-complete",
    "path": "$corrupt_artifact",
    "runtime_binding": "$binding",
    "runtime_target": "tiny-executable",
    "runtime_backend": "cpu",
    "runtime_mode": "target-only",
    "runtime_context": 8
  }]
}
EOF
if HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server tiny-executable-cpu-complete \
    --backend cpu --ctx 8 --generation-mode target-only --max-new-tokens 1 \
    --console off --openai off >"$root/corrupt.out" 2>"$root/corrupt.err"; then
    printf 'corrupt tiny artifact was admitted\n' >&2
    exit 1
fi
grep -F 'artifact admission failed' "$root/corrupt.err" >/dev/null

printf 'tiny vertical: artifact=%s binding=%s output=ok ctx=8\n' \
    "$first_artifact" "$first_binding"
