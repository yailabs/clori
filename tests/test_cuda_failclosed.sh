#!/bin/sh
set -eu

YVEX_BIN=${YVEX_BIN:-build/no-nvcc/yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cuda-failclosed}

mkdir -p "$OUT_DIR"

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

contains() {
    grep -F "$2" "$1" >/dev/null || fail "$1 missing: $2"
}

if grep -RIn -E 'Fallback embedded PTX|\.visible[[:space:]]+\.entry' \
    src/backend/cuda --include='*.c' >/dev/null; then
    fail "production C source contains embedded CUDA entry points"
fi

for contract in \
    'max_host_bytes' \
    'peak_host_bytes' \
    'cuda.deepseek_attention.validate.geometry' \
    'cuda.deepseek_attention.validate.alias' \
    'cuda.deepseek_attention.validate.host_budget' \
    'cuda.deepseek_attention.context'; do
    grep -R -F "$contract" include/yvex/backend.h \
        src/backend/cuda/families/deepseek_v4.c >/dev/null ||
        fail "encoded-attention admission missing: $contract"
done
grep -F 'atomicCAS(status, 0, 2)' src/backend/cuda/kernels.cu >/dev/null ||
    fail "encoded-attention kernels do not publish contract failures"
if grep -E 'yvex_(attention|graph)|cpu_(chunk|probe|reference)' \
    src/backend/cuda/families/deepseek_v4.c >/dev/null; then
    fail "encoded-attention CUDA owner contains a CPU numerical fallback"
fi

set +e
"$YVEX_BIN" inspect backend cuda >"$OUT_DIR/backend.out" 2>"$OUT_DIR/backend.err"
rc=$?
set -e

if [ "$rc" -eq 5 ]; then
    contains "$OUT_DIR/backend.out" "backend: cuda"
    contains "$OUT_DIR/backend.out" "status: unsupported"
    contains "$OUT_DIR/backend.out" "status: backend-unsupported"
    echo "cuda no-nvcc fail-closed: driver unavailable"
    exit 0
fi
[ "$rc" -eq 0 ] || fail "backend cuda returned $rc"

contains "$OUT_DIR/backend.out" "status: context-ready"
contains "$OUT_DIR/backend.out" "tensor_alloc: yes"
contains "$OUT_DIR/backend.out" "tensor_read_write: yes"
contains "$OUT_DIR/backend.out" "op_embed: no"
contains "$OUT_DIR/backend.out" "op_rms_norm: no"
contains "$OUT_DIR/backend.out" "op_rope: no"
contains "$OUT_DIR/backend.out" "op_attention: no"
contains "$OUT_DIR/backend.out" "op_matmul: no"
contains "$OUT_DIR/backend.out" "op_mlp: no"
contains "$OUT_DIR/backend.out" "context_available: yes"
contains "$OUT_DIR/backend.out" "kernel_bundle: absent"
contains "$OUT_DIR/backend.out" "kernel_bundle_reason: kernel-bundle-absent"
contains "$OUT_DIR/backend.out" "embed-f32-to-f32: unsupported (kernel-bundle-absent)"
contains "$OUT_DIR/backend.out" "attention-noncausal-f32: unsupported (kernel-bundle-absent)"
contains "$OUT_DIR/backend.out" "qtype-row-dot: unsupported (kernel-bundle-absent)"
contains "$OUT_DIR/backend.out" "encoded-attention: unsupported (kernel-bundle-absent)"
contains "$OUT_DIR/backend.out" "status: backend-capabilities"

set +e
"$YVEX_BIN" execute attention run --target deepseek4-v4-flash-dspark --backend cuda \
    --runtime-binding "$OUT_DIR/missing.yvex-runtime-binding" \
    --artifact "$OUT_DIR/missing.gguf" --output json \
    >"$OUT_DIR/graph.out" 2>"$OUT_DIR/graph.err"
rc=$?
set -e
[ "$rc" -ne 0 ] || fail "bundle-less CUDA attention unexpectedly succeeded"
python3 - "$OUT_DIR/graph.out" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["status"] == "refused"
assert result["backend"] == "cuda"
assert result["persistent_kv_ready"] is False
assert result["transformer_ready"] is False
assert result["runtime_generation_ready"] is False
assert result["failure_where"] == "runtime.model"
assert "runtime binding" in result["reason"]
PY
contains "$OUT_DIR/graph.err" "runtime binding"

echo "cuda no-nvcc fail-closed: ok"
