#!/bin/sh
#
# YVEX - controlled GGUF emitter CLI smoke test

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/gguf-emit}
OUT="$OUT_DIR/yvex-owned.gguf"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$YVEX_BIN" compile artifact emit \
    --out "$OUT" \
    --model-name yvex-owned-gguf-test \
    --arch llama \
    --overwrite > "$OUT_DIR/emit.out" 2> "$OUT_DIR/emit.err" || fail "emit failed"

test -f "$OUT" || fail "emitted file missing"
grep 'gguf emit: controlled' "$OUT_DIR/emit.out" >/dev/null || fail "missing emit heading"
grep 'status: gguf-written' "$OUT_DIR/emit.out" >/dev/null || fail "missing written status"
grep 'roundtrip_validated: yes' "$OUT_DIR/emit.out" >/dev/null || fail "missing roundtrip proof"

"$YVEX_BIN" artifact show "$OUT" > "$OUT_DIR/inspect.out" 2> "$OUT_DIR/inspect.err" || fail "inspect failed"
grep 'format: gguf' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect format"
grep 'version: 3' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect version"
grep 'architecture: llama' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect arch"
grep 'model_name: yvex-owned-gguf-test' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect model"
grep 'tensor_count: 1' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect tensor count"
grep 'known_tensor_bytes: 128' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect bytes"
grep 'status: descriptor-only' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect status"

"$YVEX_BIN" inspect artifact metadata "$OUT" > "$OUT_DIR/metadata.out" 2> "$OUT_DIR/metadata.err" || fail "metadata failed"
grep 'metadata_count: 12' "$OUT_DIR/metadata.out" >/dev/null || fail "missing metadata count"

"$YVEX_BIN" inspect artifact tensors "$OUT" > "$OUT_DIR/tensors.out" 2> "$OUT_DIR/tensors.err" || fail "tensors failed"
grep 'token_embd.weight' "$OUT_DIR/tensors.out" >/dev/null || fail "missing tensor name"
grep 'dims=\[4,8\]' "$OUT_DIR/tensors.out" >/dev/null || fail "missing tensor dims"
grep 'dtype=F32' "$OUT_DIR/tensors.out" >/dev/null || fail "missing tensor dtype"

"$YVEX_BIN" artifact materialize --model "$OUT" --backend cpu > "$OUT_DIR/materialize-cpu.out" 2> "$OUT_DIR/materialize-cpu.err" || fail "cpu materialize failed"
grep 'materialization status: materialized' "$OUT_DIR/materialize-cpu.out" >/dev/null || fail "missing materialized status"
grep 'tensors_materialized: 1' "$OUT_DIR/materialize-cpu.out" >/dev/null || fail "missing materialized tensor count"
grep 'bytes_materialized: 128' "$OUT_DIR/materialize-cpu.out" >/dev/null || fail "missing materialized bytes"
grep 'execution_ready: false' "$OUT_DIR/materialize-cpu.out" >/dev/null || fail "missing execution false"
grep 'status: weights-materialized' "$OUT_DIR/materialize-cpu.out" >/dev/null || fail "missing weights status"

"$YVEX_BIN" compile artifact emit --help > "$OUT_DIR/help.out" 2> "$OUT_DIR/help.err" || fail "help failed"
grep 'usage: yvex compile artifact emit' "$OUT_DIR/help.out" >/dev/null || fail "missing help usage"
