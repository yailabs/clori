#!/bin/sh
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-integrity-report}
MODEL="$OUT_DIR/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf"
REG="$OUT_DIR/models.local.json"
STALE_REG="$OUT_DIR/stale.models.local.json"
DTYPE_REG="$OUT_DIR/dtype.models.local.json"
ALIAS=deepseek4-v4-flash-dspark-selected-embed

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
data["models"][0][key] = value
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
    value = fh.read(1)
    fh.seek(-1, 2)
    fh.write(bytes([value[0] ^ 1]))
PY
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$YVEX_BIN" compile artifact emit \
  --out "$MODEL" \
  --model-name integrity-report-f16 \
  --arch deepseek \
  --target-qtype F16 \
  --overwrite >"$OUT_DIR/emit.out" 2>"$OUT_DIR/emit.err"

"$YVEX_BIN" artifact verify integrity \
  "$MODEL" \
  --backend cpu \
  --require-token-embedding \
  --partial-token 0 \
  >"$OUT_DIR/raw-backend-normal.out" 2>"$OUT_DIR/raw-backend-normal.err"
contains "$OUT_DIR/raw-backend-normal.out" "integrity: pass model=$MODEL backend=cpu"
contains "$OUT_DIR/raw-backend-normal.out" "boundary: integrity gate only, generation unsupported"
contains "$OUT_DIR/raw-backend-normal.out" "status: integrity-report-pass"
test "$(wc -l < "$OUT_DIR/raw-backend-normal.out")" -le 8

"$YVEX_BIN" artifact verify integrity \
  "$MODEL" \
  --backend cpu \
  --output nope \
  >"$OUT_DIR/raw-bad-output.out" 2>"$OUT_DIR/raw-bad-output.err" && \
  fail "bad integrity output mode unexpectedly passed" || true
contains "$OUT_DIR/raw-bad-output.err" "unsupported output mode: nope"

"$YVEX_BIN" artifact verify integrity \
  "$MODEL" \
  --require-token-embedding \
  --partial-token 0 \
  --audit \
  >"$OUT_DIR/raw-pass.out" 2>"$OUT_DIR/raw-pass.err"
contains "$OUT_DIR/raw-pass.out" "artifact_integrity_report: summary"
contains "$OUT_DIR/raw-pass.out" "model_input_kind: path"
contains "$OUT_DIR/raw-pass.out" "identity_status: unregistered"
contains "$OUT_DIR/raw-pass.out" "digest_status: not-requested"
contains "$OUT_DIR/raw-pass.out" "sha256: unavailable"
contains "$OUT_DIR/raw-pass.out" "integrity_status: pass"
contains "$OUT_DIR/raw-pass.out" "selected_embedding_ready: true"
contains "$OUT_DIR/raw-pass.out" "backend_status: not-checked"
contains "$OUT_DIR/raw-pass.out" "materialization_preflight: not-checked"
contains "$OUT_DIR/raw-pass.out" "execution_ready: false"
contains "$OUT_DIR/raw-pass.out" "generation: unsupported"
contains "$OUT_DIR/raw-pass.out" "status: integrity-report-pass"

"$YVEX_BIN" artifact verify integrity \
  "$MODEL" \
  --backend cpu \
  --require-token-embedding \
  --partial-token 0 \
  --audit \
  >"$OUT_DIR/raw-backend-pass.out" 2>"$OUT_DIR/raw-backend-pass.err"
contains "$OUT_DIR/raw-backend-pass.out" "backend_status: ready"
contains "$OUT_DIR/raw-backend-pass.out" "materialization_preflight: pass"
contains "$OUT_DIR/raw-backend-pass.out" "materialization_gate: pass"
contains "$OUT_DIR/raw-backend-pass.out" "status: integrity-report-pass"

"$YVEX_BIN" profile create \
  --path "$MODEL" \
  --alias "$ALIAS" \
  --support-level selected-tensor-materialized \
  --registry "$REG" >"$OUT_DIR/add.out" 2>"$OUT_DIR/add.err"

YVEX_MODELS_REGISTRY="$REG" "$YVEX_BIN" artifact verify integrity \
  "$ALIAS" \
  --backend cpu \
  --require-token-embedding \
  --partial-token 0 \
  --audit \
  >"$OUT_DIR/alias-pass.out" 2>"$OUT_DIR/alias-pass.err"
contains "$OUT_DIR/alias-pass.out" "model_input_kind: alias"
contains "$OUT_DIR/alias-pass.out" "identity_status: pass"
contains "$OUT_DIR/alias-pass.out" "digest_status: pass"
contains "$OUT_DIR/alias-pass.out" "metadata_status: pass"
contains "$OUT_DIR/alias-pass.out" "readiness_status: pass"
contains "$OUT_DIR/alias-pass.out" "support_level: selected-tensor-materialized"
contains "$OUT_DIR/alias-pass.out" "status: integrity-report-pass"

"$YVEX_BIN" artifact verify integrity tests/fixtures/gguf/bad-magic.gguf --audit \
  >"$OUT_DIR/corrupt.out" 2>"$OUT_DIR/corrupt.err" && \
  fail "corrupt report unexpectedly passed" || true
contains "$OUT_DIR/corrupt.out" "integrity_status: fail"
contains "$OUT_DIR/corrupt.out" "report_status: fail"
contains "$OUT_DIR/corrupt.out" "status: integrity-report-fail"

cp "$REG" "$STALE_REG"
mutate_file_byte "$MODEL"
YVEX_MODELS_REGISTRY="$STALE_REG" "$YVEX_BIN" artifact verify integrity \
  "$ALIAS" \
  --audit \
  >"$OUT_DIR/stale.out" 2>"$OUT_DIR/stale.err" && \
  fail "stale alias report unexpectedly passed" || true
contains "$OUT_DIR/stale.out" "identity_status: fail"
contains "$OUT_DIR/stale.out" "digest_status: fail"
contains "$OUT_DIR/stale.out" "report_status: fail"
contains "$OUT_DIR/stale.out" "status: integrity-report-fail"

"$YVEX_BIN" compile artifact emit \
  --out "$MODEL" \
  --model-name integrity-report-f16 \
  --arch deepseek \
  --target-qtype F16 \
  --overwrite >"$OUT_DIR/emit-reset.out" 2>"$OUT_DIR/emit-reset.err"

mutate_registry "$REG" "$DTYPE_REG" "primary_tensor_dtype" "F32"
YVEX_MODELS_REGISTRY="$DTYPE_REG" "$YVEX_BIN" artifact verify integrity \
  "$ALIAS" \
  --audit \
  >"$OUT_DIR/metadata-drift.out" 2>"$OUT_DIR/metadata-drift.err" && \
  fail "metadata drift report unexpectedly passed" || true
contains "$OUT_DIR/metadata-drift.out" "identity_status: pass"
contains "$OUT_DIR/metadata-drift.out" "metadata_status: fail"
contains "$OUT_DIR/metadata-drift.out" "metadata_issue_0_code: primary-tensor-dtype-mismatch"
contains "$OUT_DIR/metadata-drift.out" "report_status: fail"
contains "$OUT_DIR/metadata-drift.out" "status: integrity-report-fail"

"$YVEX_BIN" artifact verify integrity \
  tests/fixtures/gguf/valid-minimal.gguf \
  --require-token-embedding \
  --partial-token 0 \
  --audit \
  >"$OUT_DIR/readiness-missing.out" 2>"$OUT_DIR/readiness-missing.err" && \
  fail "readiness missing report unexpectedly passed" || true
contains "$OUT_DIR/readiness-missing.out" "selected_embedding_ready: false"
contains "$OUT_DIR/readiness-missing.out" "readiness_status: fail"
contains "$OUT_DIR/readiness-missing.out" "status: integrity-report-fail"
not_contains "$OUT_DIR/readiness-missing.out" "execution_""ready: true"

echo "cli integrity report: ok"
