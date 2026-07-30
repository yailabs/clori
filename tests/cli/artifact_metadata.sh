#!/bin/sh
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-artifact-metadata}
MODEL="$OUT_DIR/deepseek4-v4-flash-selected-embed-F16-noimatrix-yvex-v1.gguf"
F32_MODEL="$OUT_DIR/deepseek4-v4-flash-selected-embed-F32-noimatrix-yvex-v1.gguf"
REG="$OUT_DIR/models.local.json"
OLD_REG="$OUT_DIR/old.models.local.json"
DTYPE_REG="$OUT_DIR/dtype.models.local.json"
DIMS_REG="$OUT_DIR/dims.models.local.json"
ARCH_REG="$OUT_DIR/arch.models.local.json"
READY_REG="$OUT_DIR/readiness.models.local.json"
ALIAS="deepseek4-v4-flash-selected-embed"

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
if value == "true":
    entry[key] = True
elif value == "false":
    entry[key] = False
else:
    try:
        entry[key] = int(value)
    except ValueError:
        entry[key] = value
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY
}

remove_metadata_fields() {
    src=$1
    dst=$2
    python3 - "$src" "$dst" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
with open(src, "r", encoding="utf-8") as fh:
    data = json.load(fh)
entry = data["models"][0]
for key in [
    "primary_tensor_role",
    "primary_tensor_rank",
    "selected_embedding_ready",
    "selected_embedding_hidden_size",
    "selected_embedding_vocab_size",
    "selected_embedding_output_count",
    "selected_embedding_slice_bytes",
]:
    entry.pop(key, None)
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY
}

force_selected_embedding_ready() {
    src=$1
    dst=$2
    python3 - "$src" "$dst" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
with open(src, "r", encoding="utf-8") as fh:
    data = json.load(fh)
entry = data["models"][0]
entry["support_level"] = "selected-tensor-materialized"
entry["selected_embedding_ready"] = True
entry["selected_embedding_hidden_size"] = 4
entry["selected_embedding_vocab_size"] = 8
entry["selected_embedding_output_count"] = 4
entry["selected_embedding_slice_bytes"] = 8
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$YVEX_BIN" artifact emit controlled \
  --out "$MODEL" \
  --model-name metadata-controlled-f16 \
  --arch deepseek \
  --target-qtype F16 \
  --overwrite >"$OUT_DIR/emit-f16.out" 2>"$OUT_DIR/emit-f16.err"

"$YVEX_BIN" artifact emit controlled \
  --out "$F32_MODEL" \
  --model-name metadata-controlled-f32 \
  --arch deepseek \
  --target-qtype F32 \
  --overwrite >"$OUT_DIR/emit-f32.out" 2>"$OUT_DIR/emit-f32.err"

"$YVEX_BIN" evidence models add \
  --path "$MODEL" \
  --registry "$REG" \
  --support-level selected-tensor-materialized \
  >"$OUT_DIR/add.out" 2>"$OUT_DIR/add.err"
contains "$OUT_DIR/add.out" "registered_architecture: deepseek"
contains "$OUT_DIR/add.out" "registered_tensor_count: 1"
contains "$OUT_DIR/add.out" "registered_known_tensor_bytes: 64"
contains "$OUT_DIR/add.out" "registered_primary_tensor: token_embd.weight"
contains "$OUT_DIR/add.out" "registered_primary_role: token_embedding"
contains "$OUT_DIR/add.out" "registered_primary_dtype: F16"
contains "$OUT_DIR/add.out" "registered_primary_dims: [4,8]"
contains "$OUT_DIR/add.out" "registered_selected_embedding_ready: true"

"$YVEX_BIN" evidence models verify "$ALIAS" --registry "$REG" --audit \
  >"$OUT_DIR/verify-pass.out" 2>"$OUT_DIR/verify-pass.err"
contains "$OUT_DIR/verify-pass.out" "identity_status: pass"
contains "$OUT_DIR/verify-pass.out" "metadata_status: pass"
contains "$OUT_DIR/verify-pass.out" "readiness_status: pass"
contains "$OUT_DIR/verify-pass.out" "registered_primary_tensor: token_embd.weight"
contains "$OUT_DIR/verify-pass.out" "current_primary_tensor: token_embd.weight"
contains "$OUT_DIR/verify-pass.out" "registered_primary_dtype: F16"
contains "$OUT_DIR/verify-pass.out" "current_primary_dtype: F16"
contains "$OUT_DIR/verify-pass.out" "registered_primary_dims: [4,8]"
contains "$OUT_DIR/verify-pass.out" "current_primary_dims: [4,8]"
contains "$OUT_DIR/verify-pass.out" "status: models-identity-pass"

remove_metadata_fields "$REG" "$OLD_REG"
"$YVEX_BIN" evidence models verify "$ALIAS" --registry "$OLD_REG" --audit \
  >"$OUT_DIR/verify-old.out" 2>"$OUT_DIR/verify-old.err" && \
  fail "old registry metadata unexpectedly passed" || true
contains "$OUT_DIR/verify-old.out" "metadata_status: missing"
contains "$OUT_DIR/verify-old.out" "registered-metadata-missing"
contains "$OUT_DIR/verify-old.out" "status: models-metadata-missing"

mutate_registry "$REG" "$DTYPE_REG" "primary_tensor_dtype" "F32"
"$YVEX_BIN" evidence models verify "$ALIAS" --registry "$DTYPE_REG" --audit \
  >"$OUT_DIR/verify-dtype.out" 2>"$OUT_DIR/verify-dtype.err" && \
  fail "dtype drift unexpectedly passed" || true
contains "$OUT_DIR/verify-dtype.out" "metadata_status: fail"
contains "$OUT_DIR/verify-dtype.out" "primary-tensor-dtype-mismatch"
contains "$OUT_DIR/verify-dtype.out" "status: models-metadata-drift"

mutate_registry "$REG" "$DIMS_REG" "primary_tensor_dims" "[4,7]"
"$YVEX_BIN" evidence models verify "$ALIAS" --registry "$DIMS_REG" --audit \
  >"$OUT_DIR/verify-dims.out" 2>"$OUT_DIR/verify-dims.err" && \
  fail "dims drift unexpectedly passed" || true
contains "$OUT_DIR/verify-dims.out" "metadata_status: fail"
contains "$OUT_DIR/verify-dims.out" "primary-tensor-dims-mismatch"
contains "$OUT_DIR/verify-dims.out" "status: models-metadata-drift"

mutate_registry "$REG" "$ARCH_REG" "architecture" "qwen"
"$YVEX_BIN" evidence models verify "$ALIAS" --registry "$ARCH_REG" --audit \
  >"$OUT_DIR/verify-arch.out" 2>"$OUT_DIR/verify-arch.err" && \
  fail "architecture drift unexpectedly passed" || true
contains "$OUT_DIR/verify-arch.out" "metadata_status: fail"
contains "$OUT_DIR/verify-arch.out" "architecture-mismatch"
contains "$OUT_DIR/verify-arch.out" "status: models-metadata-drift"

"$YVEX_BIN" evidence models add \
  --path "$F32_MODEL" \
  --registry "$READY_REG" \
  --support-level selected-tensor-materialized \
  >"$OUT_DIR/add-f32.out" 2>"$OUT_DIR/add-f32.err"
contains "$OUT_DIR/add-f32.out" "registered_primary_dtype: F32"
contains "$OUT_DIR/add-f32.out" "registered_selected_embedding_ready: false"
force_selected_embedding_ready "$READY_REG" "$READY_REG"
"$YVEX_BIN" evidence models verify "$ALIAS" --registry "$READY_REG" --audit \
  >"$OUT_DIR/verify-readiness.out" 2>"$OUT_DIR/verify-readiness.err" && \
  fail "readiness drift unexpectedly passed" || true
contains "$OUT_DIR/verify-readiness.out" "metadata_status: fail"
contains "$OUT_DIR/verify-readiness.out" "readiness_status: fail"
contains "$OUT_DIR/verify-readiness.out" "selected-embedding-readiness-mismatch"
contains "$OUT_DIR/verify-readiness.out" "status: models-metadata-drift"

YVEX_MODELS_REGISTRY="$DTYPE_REG" "$YVEX_BIN" artifact materialize \
  --model "$ALIAS" \
  --backend cpu \
  >"$OUT_DIR/materialize-drift.out" 2>"$OUT_DIR/materialize-drift.err" && \
  fail "materialize metadata drift unexpectedly passed" || true
contains "$OUT_DIR/materialize-drift.out" "metadata_status: fail"
contains "$OUT_DIR/materialize-drift.out" "status: models-metadata-drift"
not_contains "$OUT_DIR/materialize-drift.out" "status: weights-materialized"

"$YVEX_BIN" artifact verify check --model "$MODEL" \
  >"$OUT_DIR/integrity-raw.out" 2>"$OUT_DIR/integrity-raw.err"
contains "$OUT_DIR/integrity-raw.out" "digest_status: not-requested"
contains "$OUT_DIR/integrity-raw.out" "sha256: unavailable"
contains "$OUT_DIR/integrity-raw.out" "integrity_status: pass"
contains "$OUT_DIR/integrity-raw.out" "status: artifact-integrity-pass"

echo "cli artifact metadata: ok"
