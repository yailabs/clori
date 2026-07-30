#!/bin/sh
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-materialization-integrity-gate}
FIXTURE=tests/fixtures/gguf/valid-tokenizer-simple.gguf
RANGE_BAD=tests/fixtures/gguf/tensor-offset-out-of-bounds.gguf
SHAPE_BAD=tests/fixtures/gguf/tensor-dim-zero.gguf
BAD_MAGIC=tests/fixtures/gguf/bad-magic.gguf
MODEL="$OUT_DIR/deepseek4-v4-flash-selected-embed-F16-noimatrix-yvex-v1.gguf"
REG="$OUT_DIR/models.local.json"
DRIFT_REG="$OUT_DIR/drift.models.local.json"
ALIAS=deepseek4-v4-flash-selected-embed

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains() {
    file=$1
    text=$2
    grep -F "$text" "$file" >/dev/null || fail "$file missing: $text"
}

not_contains() {
    file=$1
    text=$2
    if grep -F "$text" "$file" >/dev/null; then
        fail "$file unexpectedly contained: $text"
    fi
}

mutate_registry() {
    src=$1
    dst=$2
    key=$3
    value=$4
    python3 - "$src" "$dst" "$key" "$value" <<'PY'
import json
import sys

src, dst, key, value = sys.argv[1:]
with open(src, "r", encoding="utf-8") as fh:
    data = json.load(fh)
entry = data["models"][0]
entry[key] = value
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY
}

mutate_file_byte() {
    path=$1
    python3 - "$path" <<'PY'
import sys

path = sys.argv[1]
with open(path, "r+b") as fh:
    fh.seek(-1, 2)
    b = fh.read(1)
    fh.seek(-1, 2)
    fh.write(bytes([b[0] ^ 1]))
PY
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$YVEX_BIN" artifact materialize --model "$FIXTURE" --backend cpu \
  >"$OUT_DIR/valid-materialize.out" 2>"$OUT_DIR/valid-materialize.err"
contains "$OUT_DIR/valid-materialize.out" "materialization_gate: pass"
contains "$OUT_DIR/valid-materialize.out" "materialization_phase: complete"
contains "$OUT_DIR/valid-materialize.out" "integrity_status: pass"
contains "$OUT_DIR/valid-materialize.out" "shape_status: pass"
contains "$OUT_DIR/valid-materialize.out" "range_status: pass"
contains "$OUT_DIR/valid-materialize.out" "backend_status: ready"
contains "$OUT_DIR/valid-materialize.out" "allocation_attempted: true"
contains "$OUT_DIR/valid-materialize.out" "transfer_attempted: true"
contains "$OUT_DIR/valid-materialize.out" "cleanup_status: not-needed"
contains "$OUT_DIR/valid-materialize.out" "status: weights-materialized"

"$YVEX_BIN" artifact materialize --model "$BAD_MAGIC" --backend cpu \
  >"$OUT_DIR/bad-magic.out" 2>"$OUT_DIR/bad-magic.err" && \
  fail "bad magic materialize unexpectedly passed" || true
contains "$OUT_DIR/bad-magic.out" "materialization_gate: fail"
contains "$OUT_DIR/bad-magic.out" "materialization_phase: preflight"
contains "$OUT_DIR/bad-magic.out" "integrity_status: fail"
contains "$OUT_DIR/bad-magic.out" "allocation_attempted: false"
contains "$OUT_DIR/bad-magic.out" "status: materialization-integrity-fail"

"$YVEX_BIN" artifact materialize --model "$RANGE_BAD" --backend cpu \
  >"$OUT_DIR/range-bad.out" 2>"$OUT_DIR/range-bad.err" && \
  fail "range-bad materialize unexpectedly passed" || true
contains "$OUT_DIR/range-bad.out" "materialization_phase: preflight"
contains "$OUT_DIR/range-bad.out" "allocation_attempted: false"
contains "$OUT_DIR/range-bad.out" "status: materialization-integrity-fail"

"$YVEX_BIN" artifact materialize --model "$SHAPE_BAD" --backend cpu \
  >"$OUT_DIR/shape-bad.out" 2>"$OUT_DIR/shape-bad.err" && \
  fail "shape-bad materialize unexpectedly passed" || true
contains "$OUT_DIR/shape-bad.out" "materialization_phase: preflight"
contains "$OUT_DIR/shape-bad.out" "allocation_attempted: false"
contains "$OUT_DIR/shape-bad.out" "status: materialization-integrity-fail"

"$YVEX_BIN" artifact emit controlled \
  --out "$MODEL" \
  --model-name materialization-gate-selected \
  --arch deepseek \
  --target-qtype F16 \
  --overwrite >"$OUT_DIR/emit.out" 2>"$OUT_DIR/emit.err"

"$YVEX_BIN" evidence models add \
  --path "$MODEL" \
  --alias "$ALIAS" \
  --registry "$REG" \
  --support-level selected-tensor-materialized \
  >"$OUT_DIR/add.out" 2>"$OUT_DIR/add.err"

cp "$REG" "$OUT_DIR/stale.models.local.json"
mutate_file_byte "$MODEL"
YVEX_MODELS_REGISTRY="$OUT_DIR/stale.models.local.json" "$YVEX_BIN" artifact materialize \
  --model "$ALIAS" \
  --backend cpu \
  >"$OUT_DIR/stale.out" 2>"$OUT_DIR/stale.err" && \
  fail "stale alias materialize unexpectedly passed" || true
contains "$OUT_DIR/stale.out" "identity_status: fail"
contains "$OUT_DIR/stale.out" "allocation_attempted: false"
contains "$OUT_DIR/stale.out" "status: materialization-integrity-fail"

"$YVEX_BIN" artifact emit controlled \
  --out "$MODEL" \
  --model-name materialization-gate-selected \
  --arch deepseek \
  --target-qtype F16 \
  --overwrite >"$OUT_DIR/emit-reset.out" 2>"$OUT_DIR/emit-reset.err"

mutate_registry "$REG" "$DRIFT_REG" "primary_tensor_dtype" "F32"
YVEX_MODELS_REGISTRY="$DRIFT_REG" "$YVEX_BIN" artifact materialize \
  --model "$ALIAS" \
  --backend cpu \
  >"$OUT_DIR/metadata-drift.out" 2>"$OUT_DIR/metadata-drift.err" && \
  fail "metadata drift materialize unexpectedly passed" || true
contains "$OUT_DIR/metadata-drift.out" "metadata_status: fail"
contains "$OUT_DIR/metadata-drift.out" "allocation_attempted: false"
contains "$OUT_DIR/metadata-drift.out" "status: materialization-integrity-fail"
not_contains "$OUT_DIR/metadata-drift.out" "status: weights-materialized"

YVEX_TEST_FAIL_MATERIALIZE_AFTER_TRANSFER=1 "$YVEX_BIN" artifact materialize \
  --model "$FIXTURE" \
  --backend cpu \
  >"$OUT_DIR/injected-transfer.out" 2>"$OUT_DIR/injected-transfer.err" && \
  fail "injected transfer materialize unexpectedly passed" || true
contains "$OUT_DIR/injected-transfer.out" "materialization_gate: fail"
contains "$OUT_DIR/injected-transfer.out" "materialization_phase: transfer"
contains "$OUT_DIR/injected-transfer.out" "allocation_attempted: true"
contains "$OUT_DIR/injected-transfer.out" "transfer_attempted: true"
contains "$OUT_DIR/injected-transfer.out" "cleanup_attempted: true"
contains "$OUT_DIR/injected-transfer.out" "cleanup_status: pass"
contains "$OUT_DIR/injected-transfer.out" "status: materialization-failed-cleaned"

"$YVEX_BIN" artifact materialize --model "$FIXTURE" --backend cpu \
  >"$OUT_DIR/repeat-after-failure.out" 2>"$OUT_DIR/repeat-after-failure.err"
contains "$OUT_DIR/repeat-after-failure.out" "materialization_gate: pass"
contains "$OUT_DIR/repeat-after-failure.out" "status: weights-materialized"

"$YVEX_BIN" artifact materialize-gate check \
  --model "$FIXTURE" \
  --label fixture-selected \
  --family test \
  --scope selected-tensor \
  --backend cpu \
  --require-cpu \
  --repeat 2 \
  --check-cleanup \
  --report-out "$OUT_DIR/gate-pass-report.txt" \
  >"$OUT_DIR/gate-pass.out"
contains "$OUT_DIR/gate-pass-report.txt" "materialization_gate: pass"
contains "$OUT_DIR/gate-pass-report.txt" "materialization_phase: complete"
contains "$OUT_DIR/gate-pass-report.txt" "cleanup_status: pass"
contains "$OUT_DIR/gate-pass-report.txt" "status: materialize-gate-pass"

YVEX_TEST_FAIL_MATERIALIZE_AFTER_TRANSFER=1 "$YVEX_BIN" artifact materialize-gate check \
  --model "$FIXTURE" \
  --label fixture-selected \
  --family test \
  --scope selected-tensor \
  --backend cpu \
  --require-cpu \
  --check-cleanup \
  --report-out "$OUT_DIR/gate-fail-report.txt" \
  >"$OUT_DIR/gate-fail.out" && \
  fail "injected materialize-gate unexpectedly passed" || true
contains "$OUT_DIR/gate-fail-report.txt" "materialization_gate: fail"
contains "$OUT_DIR/gate-fail-report.txt" "materialization_phase: transfer"
contains "$OUT_DIR/gate-fail-report.txt" "allocation_attempted: true"
contains "$OUT_DIR/gate-fail-report.txt" "transfer_attempted: true"
contains "$OUT_DIR/gate-fail-report.txt" "cleanup_attempted: true"
contains "$OUT_DIR/gate-fail-report.txt" "cleanup_status: pass"
contains "$OUT_DIR/gate-fail-report.txt" "status: materialize-gate-fail"

echo "cli materialization integrity gate: ok"
