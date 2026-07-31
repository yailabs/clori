#!/bin/sh
set -eu

YVEX_BIN="${YVEX_BIN:-./yvex}"
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/model-gate-cli}
mkdir -p "$OUT_DIR"

fail() {
  echo "cli model-gate: $*" >&2
  exit 1
}

"$YVEX_BIN" compile emit artifact controlled \
  --out "$OUT_DIR/selected.gguf" \
  --model-name model-gate-test \
  --arch llama \
  --overwrite > "$OUT_DIR/emit.out"

"$YVEX_BIN" execute artifact model-gate check \
  --model "$OUT_DIR/selected.gguf" \
  --label fixture-selected \
  --family llama \
  --expect-tensor token_embd.weight \
  --expect-rank 2 \
  --expect-dims 4,8 \
  --expect-dtype F32 \
  --expect-bytes 128 \
  --backend cpu \
  --require-cpu \
  --report-out "$OUT_DIR/report.txt" > "$OUT_DIR/check.out"

grep 'status: model-gate-pass' "$OUT_DIR/report.txt" >/dev/null || fail "missing pass status"
grep 'support_level: selected-tensor-materialized' "$OUT_DIR/report.txt" >/dev/null || fail "missing selected support"
grep 'execution_ready: false' "$OUT_DIR/report.txt" >/dev/null || fail "execution_ready changed"

if "$YVEX_BIN" execute artifact model-gate check \
  --model "$OUT_DIR/selected.gguf" \
  --label bad-shape \
  --family llama \
  --expect-tensor token_embd.weight \
  --expect-rank 2 \
  --expect-dims 8,4 \
  --expect-dtype F32 \
  --expect-bytes 128 \
  --backend cpu \
  --report-out "$OUT_DIR/bad-report.txt" > "$OUT_DIR/bad-check.out"; then
  fail "bad shape unexpectedly passed"
fi

grep 'status: model-gate-fail' "$OUT_DIR/bad-report.txt" >/dev/null || fail "missing fail status"
"$YVEX_BIN" execute artifact model-gate --help > "$OUT_DIR/help.out"
grep 'model-gate check' "$OUT_DIR/help.out" >/dev/null || fail "missing help"
