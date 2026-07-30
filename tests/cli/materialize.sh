#!/bin/sh
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-materialize}
FIXTURE=tests/fixtures/gguf/valid-tokenizer-simple.gguf

mkdir -p "$OUT_DIR"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains() {
    file=$1
    text=$2
    grep -F "$text" "$file" >/dev/null || fail "$file missing: $text"
}

"$YVEX_BIN" artifact materialize --model "$FIXTURE" --backend cpu >"$OUT_DIR/materialize_cpu.out" 2>"$OUT_DIR/materialize_cpu.err" ||
    fail "materialize cpu exited non-zero"
contains "$OUT_DIR/materialize_cpu.out" "materialization status: materialized"
contains "$OUT_DIR/materialize_cpu.out" "model: yvex-tokenizer-test"
contains "$OUT_DIR/materialize_cpu.out" "backend: cpu"
contains "$OUT_DIR/materialize_cpu.out" "tensors_total: 1"
contains "$OUT_DIR/materialize_cpu.out" "tensors_materialized: 1"
contains "$OUT_DIR/materialize_cpu.out" "bytes_materialized: 128"
contains "$OUT_DIR/materialize_cpu.out" "backend_allocated_bytes: 128"
contains "$OUT_DIR/materialize_cpu.out" "execution_ready: false"
contains "$OUT_DIR/materialize_cpu.out" "status: weights-materialized"

"$YVEX_BIN" artifact materialize --help >"$OUT_DIR/help_materialize.out" 2>"$OUT_DIR/help_materialize.err" ||
    fail "help materialize exited non-zero"
contains "$OUT_DIR/help_materialize.out" "usage: yvex artifact materialize --model FILE_OR_ALIAS"

echo "cli materialize smoke: ok"
