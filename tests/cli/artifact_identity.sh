#!/bin/sh
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-artifact-identity}
MODEL="$OUT_DIR/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf"
STALE_DIR="$OUT_DIR/stale"
STALE_MODEL="$STALE_DIR/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf"
REG="$OUT_DIR/models.local.json"
STALE_REG="$OUT_DIR/stale.models.local.json"
ALIAS="deepseek4-v4-flash-dspark-selected-embed"
BAD_SHA="0000000000000000000000000000000000000000000000000000000000000000"

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

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR" "$STALE_DIR"

"$YVEX_BIN" compile emit artifact controlled \
  --out "$MODEL" \
  --model-name identity-controlled-f16 \
  --arch deepseek \
  --target-qtype F16 \
  --overwrite >"$OUT_DIR/emit.out" 2>"$OUT_DIR/emit.err"
cp "$MODEL" "$STALE_MODEL"

"$YVEX_BIN" model registry add \
  --path "$MODEL" \
  --registry "$REG" \
  --support-level selected-tensor-materialized \
  >"$OUT_DIR/add.out" 2>"$OUT_DIR/add.err"
contains "$OUT_DIR/add.out" "registered_file_size:"
contains "$OUT_DIR/add.out" "registered_sha256:"
contains "$OUT_DIR/add.out" "registered_format: gguf"
contains "$OUT_DIR/add.out" "registered_architecture: deepseek"
contains "$OUT_DIR/add.out" "registered_tensor_count: 1"
contains "$OUT_DIR/add.out" "registered_known_tensor_bytes: 64"
contains "$OUT_DIR/add.out" "registered_primary_tensor: token_embd.weight"
contains "$OUT_DIR/add.out" "registered_primary_role: token_embedding"
contains "$OUT_DIR/add.out" "registered_primary_dtype: F16"
contains "$OUT_DIR/add.out" "registered_primary_dims: [4,8]"
contains "$OUT_DIR/add.out" "registered_selected_embedding_ready: true"
contains "$OUT_DIR/add.out" "identity_status: recorded"
contains "$OUT_DIR/add.out" "status: models-added"

GOOD_SHA=$(awk '/^registered_sha256: / { print $2 }' "$OUT_DIR/add.out")
test -n "$GOOD_SHA" || fail "missing registered sha"

"$YVEX_BIN" model registry verify "$ALIAS" --registry "$REG" --audit \
  >"$OUT_DIR/verify-pass.out" 2>"$OUT_DIR/verify-pass.err"
contains "$OUT_DIR/verify-pass.out" "digest_status: pass"
contains "$OUT_DIR/verify-pass.out" "identity_status: pass"
contains "$OUT_DIR/verify-pass.out" "metadata_status: pass"
contains "$OUT_DIR/verify-pass.out" "readiness_status: pass"
contains "$OUT_DIR/verify-pass.out" "registered_primary_tensor: token_embd.weight"
contains "$OUT_DIR/verify-pass.out" "current_primary_tensor: token_embd.weight"
contains "$OUT_DIR/verify-pass.out" "registered_primary_dtype: F16"
contains "$OUT_DIR/verify-pass.out" "current_primary_dtype: F16"
contains "$OUT_DIR/verify-pass.out" "status: models-identity-pass"

"$YVEX_BIN" artifact verify "$MODEL" --expect-sha256 "$GOOD_SHA" \
  >"$OUT_DIR/integrity-expected-pass.out" 2>"$OUT_DIR/integrity-expected-pass.err"
contains "$OUT_DIR/integrity-expected-pass.out" "identity_checked: true"
contains "$OUT_DIR/integrity-expected-pass.out" "expected_sha256: $GOOD_SHA"
contains "$OUT_DIR/integrity-expected-pass.out" "digest_status: pass"
contains "$OUT_DIR/integrity-expected-pass.out" "status: artifact-integrity-pass"

"$YVEX_BIN" artifact verify "$MODEL" --expect-sha256 "$BAD_SHA" \
  >"$OUT_DIR/integrity-expected-fail.out" 2>"$OUT_DIR/integrity-expected-fail.err" && \
  fail "bad expected sha unexpectedly passed" || true
contains "$OUT_DIR/integrity-expected-fail.out" "digest_status: fail"
contains "$OUT_DIR/integrity-expected-fail.out" "error_0_code: digest-mismatch"
contains "$OUT_DIR/integrity-expected-fail.out" "expected_sha256: $BAD_SHA"
contains "$OUT_DIR/integrity-expected-fail.out" "actual_sha256: $GOOD_SHA"
contains "$OUT_DIR/integrity-expected-fail.out" "status: artifact-integrity-fail"

"$YVEX_BIN" artifact verify "$MODEL" \
  >"$OUT_DIR/integrity-raw.out" 2>"$OUT_DIR/integrity-raw.err"
contains "$OUT_DIR/integrity-raw.out" "registered_sha256: absent"
contains "$OUT_DIR/integrity-raw.out" "digest_status: not-requested"
contains "$OUT_DIR/integrity-raw.out" "sha256: unavailable"
contains "$OUT_DIR/integrity-raw.out" "status: artifact-integrity-pass"

"$YVEX_BIN" model registry add \
  --path "$STALE_MODEL" \
  --registry "$STALE_REG" \
  --support-level selected-tensor-materialized \
  >"$OUT_DIR/stale-add.out" 2>"$OUT_DIR/stale-add.err"
printf Z | dd of="$STALE_MODEL" bs=1 seek=128 conv=notrunc status=none

"$YVEX_BIN" model registry verify "$ALIAS" --registry "$STALE_REG" --audit \
  >"$OUT_DIR/verify-stale.out" 2>"$OUT_DIR/verify-stale.err" && \
  fail "mutated registered artifact unexpectedly verified" || true
contains "$OUT_DIR/verify-stale.out" "digest_status: fail"
contains "$OUT_DIR/verify-stale.out" "identity_status: fail"
contains "$OUT_DIR/verify-stale.out" "status: models-identity-fail"

YVEX_MODELS_REGISTRY="$STALE_REG" "$YVEX_BIN" artifact materialize \
  --model "$ALIAS" \
  --backend cpu \
  >"$OUT_DIR/materialize-stale.out" 2>"$OUT_DIR/materialize-stale.err" && \
  fail "materialize stale alias unexpectedly passed" || true
contains "$OUT_DIR/materialize-stale.out" "identity_status: fail"
contains "$OUT_DIR/materialize-stale.out" "status: models-identity-fail"
not_contains "$OUT_DIR/materialize-stale.out" "status: weights-materialized"

"$YVEX_BIN" execute artifact model-gate check \
  --model "$MODEL" \
  --label identity-bad-sha \
  --family deepseek \
  --expect-tensor token_embd.weight \
  --expect-rank 2 \
  --expect-dims 4,8 \
  --expect-dtype F16 \
  --expect-bytes 64 \
  --sha256 "$BAD_SHA" \
  --report-out "$OUT_DIR/model-gate-bad-sha.txt" \
  >"$OUT_DIR/model-gate-bad-sha.out" 2>"$OUT_DIR/model-gate-bad-sha.err" && \
  fail "model-gate bad sha unexpectedly passed" || true
contains "$OUT_DIR/model-gate-bad-sha.txt" "expected_sha256: $BAD_SHA"
contains "$OUT_DIR/model-gate-bad-sha.txt" "actual_sha256: $GOOD_SHA"
contains "$OUT_DIR/model-gate-bad-sha.txt" "digest_status: fail"
contains "$OUT_DIR/model-gate-bad-sha.txt" "identity_status: fail"

"$YVEX_BIN" execute artifact materialize-gate check \
  --model "$MODEL" \
  --label identity-bad-sha \
  --family deepseek \
  --scope selected-tensor \
  --expect-tensor token_embd.weight \
  --expect-rank 2 \
  --expect-dims 4,8 \
  --expect-dtype F16 \
  --expect-bytes 64 \
  --sha256 "$BAD_SHA" \
  --report-out "$OUT_DIR/materialize-gate-bad-sha.txt" \
  >"$OUT_DIR/materialize-gate-bad-sha.out" 2>"$OUT_DIR/materialize-gate-bad-sha.err" && \
  fail "materialize-gate bad sha unexpectedly passed" || true
contains "$OUT_DIR/materialize-gate-bad-sha.txt" "expected_sha256: $BAD_SHA"
contains "$OUT_DIR/materialize-gate-bad-sha.txt" "actual_sha256: $GOOD_SHA"
contains "$OUT_DIR/materialize-gate-bad-sha.txt" "digest_status: fail"
contains "$OUT_DIR/materialize-gate-bad-sha.txt" "identity_status: fail"
contains "$OUT_DIR/materialize-gate-bad-sha.txt" "failure_class: hash_mismatch"

echo "cli artifact identity: ok"
