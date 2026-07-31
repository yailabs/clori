#!/bin/sh

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/physical-variant-refusal}
: "${YVEX_DEEPSEEK_SOURCE:?DeepSeek source is required}"
: "${YVEX_DEEPSEEK_MODELS_ROOT:?DeepSeek models root is required}"
: "${YVEX_DEEPSEEK_SOURCE_MANIFEST:?DeepSeek compile source manifest is required}"
: "${YVEX_IMATRIX_PATH:?DeepSeek imatrix is required}"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

expect_plan_refusal() {
    name=$1
    policy=$2
    set +e
    "$YVEX_BIN" compile quant plan \
        --target deepseek4-v4-flash \
        --source "$YVEX_DEEPSEEK_SOURCE" \
        --models-root "$YVEX_DEEPSEEK_MODELS_ROOT" \
        --source-manifest "$YVEX_DEEPSEEK_SOURCE_MANIFEST" \
        --policy "$policy" \
        --backend cpu \
        --out-plan "$OUT_DIR/$name.plan" \
        > "$OUT_DIR/$name.out" 2> "$OUT_DIR/$name.err"
    rc=$?
    set -e
    test "$rc" -ne 0 || fail "$name unexpectedly planned"
    test ! -e "$OUT_DIR/$name.plan" || fail "$name published a refused plan"
}

expect_calibrated_plan_refusal() {
    name=$1
    policy=$2
    set +e
    "$YVEX_BIN" compile quant plan \
        --target deepseek4-v4-flash \
        --source "$YVEX_DEEPSEEK_SOURCE" \
        --models-root "$YVEX_DEEPSEEK_MODELS_ROOT" \
        --source-manifest "$YVEX_DEEPSEEK_SOURCE_MANIFEST" \
        --policy "$policy" \
        --imatrix-manifest "$YVEX_IMATRIX_PATH" \
        --backend cuda \
        --out-plan "$OUT_DIR/$name.plan" \
        > "$OUT_DIR/$name.out" 2> "$OUT_DIR/$name.err"
    rc=$?
    set -e
    test "$rc" -ne 0 || fail "$name unexpectedly planned"
    test ! -e "$OUT_DIR/$name.plan" || fail "$name published a refused plan"
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$YVEX_BIN" compile quant plan \
    --target deepseek4-v4-flash \
    --source "$YVEX_DEEPSEEK_SOURCE" \
    --models-root "$YVEX_DEEPSEEK_MODELS_ROOT" \
    --source-manifest "$YVEX_DEEPSEEK_SOURCE_MANIFEST" \
    --preset deepseek-v4-flash-ds4-like-q2-v1 \
    --imatrix-manifest "$YVEX_IMATRIX_PATH" \
    --backend cuda \
    --out-plan "$OUT_DIR/accepted.plan" \
    > "$OUT_DIR/accepted.out"
grep '^terminal_decisions: 1360$' "$OUT_DIR/accepted.out" >/dev/null ||
    fail "complete terminal count missing"
grep '^artifact_emittable: 1$' "$OUT_DIR/accepted.out" >/dev/null ||
    fail "artifact compatibility missing"
grep '^cpu_runtime_executable: 1$' "$OUT_DIR/accepted.out" >/dev/null ||
    fail "CPU compatibility missing"
grep '^cuda_runtime_executable: 1$' "$OUT_DIR/accepted.out" >/dev/null ||
    fail "CUDA compatibility missing"

cat > "$OUT_DIR/conflict.json" <<'JSON'
{
  "schema": "yvex.quant_policy.v2",
  "name": "conflicting-actions",
  "architecture": "deepseek4-v4-flash",
  "rules": [
    {"match":{"physical_class":"quantizable"},"action":{"qtype":"Q8_0","calibration":"none","requires_cpu_compute":true,"requires_cuda_compute":false},"priority":100},
    {"match":{"physical_class":"quantizable"},"action":{"qtype":"Q2_K","calibration":"none","requires_cpu_compute":true,"requires_cuda_compute":false},"priority":100}
  ]
}
JSON

cat > "$OUT_DIR/missing-default.json" <<'JSON'
{
  "schema": "yvex.quant_policy.v2",
  "name": "missing-default",
  "architecture": "deepseek4-v4-flash",
  "rules": [
    {"match":{"role":"moe_expert_gate"},"action":{"qtype":"Q8_0","calibration":"none","requires_cpu_compute":true,"requires_cuda_compute":false},"priority":100}
  ]
}
JSON

cat > "$OUT_DIR/exact-override.json" <<'JSON'
{
  "schema": "yvex.quant_policy.v2",
  "name": "exact-override",
  "architecture": "deepseek4-v4-flash",
  "rules": [
    {"match":{"physical_class":"exact"},"action":{"qtype":"Q8_0","calibration":"none","requires_cpu_compute":true,"requires_cuda_compute":false},"priority":100},
    {"match":{"physical_class":"quantizable"},"action":{"qtype":"Q8_0","calibration":"none","requires_cpu_compute":true,"requires_cuda_compute":false},"priority":10}
  ]
}
JSON

cat > "$OUT_DIR/unsupported.json" <<'JSON'
{
  "schema": "yvex.quant_policy.v2",
  "name": "unsupported-codec",
  "architecture": "deepseek4-v4-flash",
  "rules": [
    {"match":{"physical_class":"quantizable"},"action":{"qtype":"Q4_K","calibration":"none","requires_cpu_compute":true,"requires_cuda_compute":true},"priority":10}
  ]
}
JSON

cat > "$OUT_DIR/iq2-wrong-operation.json" <<'JSON'
{
  "schema": "yvex.quant_policy.v2",
  "name": "iq2-wrong-operation",
  "architecture": "deepseek4-v4-flash",
  "rules": [
    {"match":{"role":"attention_q_a"},"action":{"qtype":"IQ2_XXS","calibration":"required","requires_cpu_compute":true,"requires_cuda_compute":true},"priority":100},
    {"match":{"physical_class":"quantizable"},"action":{"qtype":"Q8_0","calibration":"none","requires_cpu_compute":true,"requires_cuda_compute":true},"priority":10}
  ]
}
JSON

expect_plan_refusal conflict "$OUT_DIR/conflict.json"
expect_plan_refusal missing-default "$OUT_DIR/missing-default.json"
expect_plan_refusal exact-override "$OUT_DIR/exact-override.json"
expect_plan_refusal unsupported "$OUT_DIR/unsupported.json"
expect_calibrated_plan_refusal iq2-wrong-operation "$OUT_DIR/iq2-wrong-operation.json"

printf 'physical variant refusal live: ok\n'
