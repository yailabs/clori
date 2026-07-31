#!/bin/sh
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-artifact-integrity}
MODEL="$OUT_DIR/integrity-controlled-F16.gguf"
MODEL_F32="$OUT_DIR/integrity-controlled-F32.gguf"
RANGE_SHORT="$OUT_DIR/range-one-byte-short.gguf"

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
mkdir -p "$OUT_DIR"

"$YVEX_BIN" compile emit artifact controlled \
  --out "$MODEL" \
  --model-name integrity-controlled-f16 \
  --arch deepseek \
  --target-qtype F16 \
  --overwrite >"$OUT_DIR/emit.out" 2>"$OUT_DIR/emit.err"

"$YVEX_BIN" compile emit artifact controlled \
  --out "$MODEL_F32" \
  --model-name integrity-controlled-f32 \
  --arch deepseek \
  --target-qtype F32 \
  --overwrite >"$OUT_DIR/emit-f32.out" 2>"$OUT_DIR/emit-f32.err"

python3 - "$RANGE_SHORT" <<'PY'
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])

def u32(v):
    return struct.pack("<I", v)

def u64(v):
    return struct.pack("<Q", v)

def s(text):
    data = text.encode("utf-8")
    return u64(len(data)) + data

def kv_string(key, value):
    return s(key) + u32(8) + s(value)

def tensor(name, dims, dtype, offset):
    out = s(name) + u32(len(dims))
    for dim in dims:
        out += u64(dim)
    return out + u32(dtype) + u64(offset)

data = b"GGUF" + u32(3) + u64(1) + u64(1)
data += kv_string("general.architecture", "deepseek")
data += tensor("token_embd.weight", [4, 8], 0, 0)
data += b"\0" * ((32 - (len(data) % 32)) % 32)
data += b"\0" * 127
path.write_bytes(data)
PY

"$YVEX_BIN" artifact verify "$MODEL" --require-token-embedding --partial-token 0 \
  >"$OUT_DIR/integrity-pass.out" 2>"$OUT_DIR/integrity-pass.err"
contains "$OUT_DIR/integrity-pass.out" "artifact_integrity: check"
contains "$OUT_DIR/integrity-pass.out" "format: gguf"
contains "$OUT_DIR/integrity-pass.out" "version: 3"
contains "$OUT_DIR/integrity-pass.out" "architecture: deepseek"
contains "$OUT_DIR/integrity-pass.out" "tensor_count: 1"
contains "$OUT_DIR/integrity-pass.out" "known_tensor_bytes: 64"
contains "$OUT_DIR/integrity-pass.out" "tensor_ranges_checked: 1"
contains "$OUT_DIR/integrity-pass.out" "tensor_ranges_valid: 1"
contains "$OUT_DIR/integrity-pass.out" "tensor_ranges_invalid: 0"
contains "$OUT_DIR/integrity-pass.out" "tensor_shapes_checked: 1"
contains "$OUT_DIR/integrity-pass.out" "tensor_shapes_valid: 1"
contains "$OUT_DIR/integrity-pass.out" "tensor_shapes_invalid: 0"
contains "$OUT_DIR/integrity-pass.out" "tensor_dtypes_checked: 1"
contains "$OUT_DIR/integrity-pass.out" "tensor_dtypes_valid: 1"
contains "$OUT_DIR/integrity-pass.out" "tensor_dtypes_invalid: 0"
contains "$OUT_DIR/integrity-pass.out" "tensor_byte_counts_checked: 1"
contains "$OUT_DIR/integrity-pass.out" "tensor_byte_counts_invalid: 0"
contains "$OUT_DIR/integrity-pass.out" "selected_embedding_shape: valid"
contains "$OUT_DIR/integrity-pass.out" "selected_embedding_hidden_size: 4"
contains "$OUT_DIR/integrity-pass.out" "selected_embedding_vocab_size: 8"
contains "$OUT_DIR/integrity-pass.out" "selected_embedding_output_count: 4"
contains "$OUT_DIR/integrity-pass.out" "selected_embedding_output_bytes: 16"
contains "$OUT_DIR/integrity-pass.out" "selected_embedding_slice_bytes: 8"
contains "$OUT_DIR/integrity-pass.out" "integrity_status: pass"
contains "$OUT_DIR/integrity-pass.out" "status: artifact-integrity-pass"

"$YVEX_BIN" artifact verify --help >"$OUT_DIR/help.out" 2>"$OUT_DIR/help.err"
contains "$OUT_DIR/help.out" "usage: yvex artifact verify artifact"

"$YVEX_BIN" artifact verify tests/fixtures/gguf/bad-magic.gguf \
  >"$OUT_DIR/bad-magic.out" 2>"$OUT_DIR/bad-magic.err" && fail "bad magic passed" || true
contains "$OUT_DIR/bad-magic.out" "integrity_status: fail"
contains "$OUT_DIR/bad-magic.out" "error_0_code: bad-magic"
contains "$OUT_DIR/bad-magic.out" "status: artifact-integrity-fail"

"$YVEX_BIN" artifact verify tests/fixtures/gguf/short-header.gguf \
  >"$OUT_DIR/truncated.out" 2>"$OUT_DIR/truncated.err" && fail "truncated file passed" || true
contains "$OUT_DIR/truncated.out" "integrity_status: fail"
contains "$OUT_DIR/truncated.out" "error_0_code: file-too-small"

"$YVEX_BIN" artifact verify "$RANGE_SHORT" \
  >"$OUT_DIR/range.out" 2>"$OUT_DIR/range.err" && fail "range-bad file passed" || true
contains "$OUT_DIR/range.out" "integrity_status: fail"
contains "$OUT_DIR/range.out" "error_0_code: tensor-payload-truncated"
contains "$OUT_DIR/range.out" "tensor_ranges_checked: 1"
contains "$OUT_DIR/range.out" "tensor_ranges_invalid: 1"
contains "$OUT_DIR/range.out" "error_0_relative_offset:"
contains "$OUT_DIR/range.out" "error_0_absolute_offset:"
contains "$OUT_DIR/range.out" "error_0_tensor_bytes:"
contains "$OUT_DIR/range.out" "error_0_file_size:"

"$YVEX_BIN" artifact verify tests/fixtures/gguf/tensor-dim-zero.gguf \
  >"$OUT_DIR/zero-dim.out" 2>"$OUT_DIR/zero-dim.err" && fail "zero-dim file passed" || true
contains "$OUT_DIR/zero-dim.out" "integrity_status: fail"
contains "$OUT_DIR/zero-dim.out" "error_0_code: zero-dimension"

"$YVEX_BIN" artifact verify "$MODEL_F32" --require-token-embedding --partial-token 0 \
  >"$OUT_DIR/f32-selected.out" 2>"$OUT_DIR/f32-selected.err" && fail "F32 selected embedding readiness passed" || true
contains "$OUT_DIR/f32-selected.out" "integrity_status: fail"
contains "$OUT_DIR/f32-selected.out" "error_0_code: required-tensor-dtype-invalid"

"$YVEX_BIN" artifact verify tests/fixtures/gguf/valid-minimal.gguf --require-token-embedding \
  >"$OUT_DIR/missing-required.out" 2>"$OUT_DIR/missing-required.err" && fail "missing required passed" || true
contains "$OUT_DIR/missing-required.out" "integrity_status: fail"
contains "$OUT_DIR/missing-required.out" "error_0_code: required-tensor-missing"

"$YVEX_BIN" artifact verify "$MODEL" --partial-token 99 \
  >"$OUT_DIR/token-range.out" 2>"$OUT_DIR/token-range.err" && fail "token out of range passed" || true
contains "$OUT_DIR/token-range.out" "integrity_status: fail"
contains "$OUT_DIR/token-range.out" "error_0_code: token-out-of-range"

"$YVEX_BIN" artifact materialize --model tests/fixtures/gguf/tensor-offset-out-of-bounds.gguf --backend cpu \
  >"$OUT_DIR/materialize-range.out" 2>"$OUT_DIR/materialize-range.err" && fail "materialize corrupt range passed" || true
contains "$OUT_DIR/materialize-range.err" "first-offset-not-zero"
not_contains "$OUT_DIR/materialize-range.out" "status: weights-materialized"

echo "cli artifact integrity: ok"
