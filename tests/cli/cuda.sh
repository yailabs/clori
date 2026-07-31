#!/bin/sh
# CUDA CLI smoke: device admission, backend facts, and bounded materialization.

set -u

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-${OUT_DIR:-build/tests/cli/cuda}}
FIXTURE=tests/fixtures/gguf/valid-tokenizer-simple.gguf

mkdir -p "$OUT_DIR"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains() {
    file=$1
    value=$2
    grep -F -- "$value" "$file" >/dev/null || fail "$file missing: $value"
}

"$YVEX_BIN" system cuda >"$OUT_DIR/cuda_info.out" 2>"$OUT_DIR/cuda_info.err"
rc=$?
if [ "$rc" -eq 5 ]; then
    contains "$OUT_DIR/cuda_info.out" "cuda: unavailable"
    contains "$OUT_DIR/cuda_info.out" "status: cuda-unavailable"
    printf 'cli cuda smoke: skip\n'
    exit 77
fi
[ "$rc" -eq 0 ] || fail "cuda-info exit code was $rc"
contains "$OUT_DIR/cuda_info.out" "cuda: available"
contains "$OUT_DIR/cuda_info.out" "kernel_bundle: admitted"
contains "$OUT_DIR/cuda_info.out" "status: cuda-info"

"$YVEX_BIN" inspect backend cuda >"$OUT_DIR/backend.out" 2>"$OUT_DIR/backend.err"
rc=$?
[ "$rc" -eq 0 ] || fail "backend cuda exit code was $rc"
contains "$OUT_DIR/backend.out" "backend: cuda"
contains "$OUT_DIR/backend.out" "status: ready"
contains "$OUT_DIR/backend.out" "kernel_bundle: admitted"
contains "$OUT_DIR/backend.out" "status: backend-capabilities"

"$YVEX_BIN" artifact materialize --model "$FIXTURE" --backend cuda \
    >"$OUT_DIR/materialize.out" 2>"$OUT_DIR/materialize.err"
rc=$?
[ "$rc" -eq 0 ] || fail "materialize cuda exit code was $rc"
contains "$OUT_DIR/materialize.out" "materialization status: materialized"
contains "$OUT_DIR/materialize.out" "backend: cuda"
contains "$OUT_DIR/materialize.out" "status: weights-materialized"

"$YVEX_BIN" help >"$OUT_DIR/help.out" 2>"$OUT_DIR/help.err"
rc=$?
[ "$rc" -eq 0 ] || fail "help cuda-info exit code was $rc"
contains "$OUT_DIR/help.out" "yvex inspect target|model|moe|backend|cuda"

printf 'cli cuda smoke: ok\n'
