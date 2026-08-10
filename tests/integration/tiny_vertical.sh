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
    --console off --openai off >"$root/server.out" 2>"$root/server.err" &
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

HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server log --json \
    >"$root/server.log.jsonl" 2>"$root/server.log.err" &
log_pid=$!
HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" run a \
    --strategy greedy --max-new-tokens 1 >"$root/run.out" 2>"$root/run.err"
grep -Fx 'ok' "$root/run.out" >/dev/null
grep -F 'generation 1 token' "$root/run.err" >/dev/null

HOME="$home" XDG_RUNTIME_DIR="$runtime" "$YVEX_BIN" server stop >/dev/null
wait "$server_pid"
server_pid=
wait "$log_pid"
log_pid=
grep -F '"kind":"generation.completed"' "$root/server.log.jsonl" >/dev/null

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
