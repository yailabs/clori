#!/bin/sh
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-artifact-corruption}
GEN_DIR="$OUT_DIR/generated"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains() {
    file=$1
    text=$2
    grep -F "$text" "$file" >/dev/null || fail "$file missing: $text"
}

contains_any_code() {
    file=$1
    codes=$2
    old_ifs=$IFS
    IFS=,
    for code in $codes; do
        if grep -F "error_0_code: $code" "$file" >/dev/null; then
            IFS=$old_ifs
            return 0
        fi
    done
    IFS=$old_ifs
    fail "$file missing one of error codes: $codes"
}

not_contains() {
    file=$1
    text=$2
    if grep -F "$text" "$file" >/dev/null; then
        fail "$file unexpectedly contained: $text"
    fi
}

run_reject() {
    name=$1
    surface=$2
    success_text=$3
    shift 3
    out="$OUT_DIR/$name.$surface.out"
    err="$OUT_DIR/$name.$surface.err"

    if "$@" >"$out" 2>"$err"; then
        fail "$name $surface unexpectedly passed"
    fi
    not_contains "$out" "$success_text"
    if [ ! -s "$out" ] && [ ! -s "$err" ]; then
        fail "$name $surface failed without output"
    fi
}

run_accept() {
    name=$1
    surface=$2
    expected_text=$3
    shift 3
    out="$OUT_DIR/$name.$surface.out"
    err="$OUT_DIR/$name.$surface.err"

    "$@" >"$out" 2>"$err"
    contains "$out" "$expected_text"
}

expect_integrity_failure() {
    name=$1
    path=$2
    codes=$3
    shift 3
    out="$OUT_DIR/$name.integrity.out"
    err="$OUT_DIR/$name.integrity.err"

    if "$YVEX_BIN" artifact verify "$path" "$@" >"$out" 2>"$err"; then
        fail "$name integrity unexpectedly passed"
    fi
    contains "$out" "integrity_status: fail"
    contains "$out" "status: artifact-integrity-fail"
    contains_any_code "$out" "$codes"
}

exercise_structural_case() {
    name=$1
    path=$2
    codes=$3

    printf '%s|fail|fail|fail|fail\n' "$name" >>"$OUT_DIR/refusal-matrix.txt"
    expect_integrity_failure "$name" "$path" "$codes"
    run_reject "$name" inspect "status: descriptor-only" \
        "$YVEX_BIN" artifact show "$path"
    run_reject "$name" tensors "tensor_count:" \
        "$YVEX_BIN" inspect artifact tensors "$path"
    run_reject "$name" materialize "status: weights-materialized" \
        "$YVEX_BIN" artifact materialize --model "$path" --backend cpu
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$GEN_DIR"

python3 - "$GEN_DIR" <<'PY'
import struct
import sys
from pathlib import Path

out = Path(sys.argv[1])
magic = b"GGUF"
version = 3

uint32 = 4
string = 8
ggml_f32 = 0

def u32(value):
    return struct.pack("<I", value)

def u64(value):
    return struct.pack("<Q", value)

def gguf_string(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return u64(len(value)) + value

def kv_string(key, value):
    return gguf_string(key) + u32(string) + gguf_string(value)

def kv_u32(key, value):
    return gguf_string(key) + u32(uint32) + u32(value)

def tensor(name, dims, dtype, offset):
    data = gguf_string(name)
    data += u32(len(dims))
    for dim in dims:
        data += u64(dim)
    data += u32(dtype)
    data += u64(offset)
    return data

def align(data, alignment=32):
    pad = (alignment - (len(data) % alignment)) % alignment
    return data + (b"\0" * pad)

def file_bytes(metadata, tensors, payload=b"", alignment=32):
    data = magic + u32(version) + u64(len(tensors)) + u64(len(metadata))
    data += b"".join(metadata)
    data += b"".join(tensors)
    if tensors:
        data = align(data, alignment)
        data += payload
    return data

def write(name, data):
    (out / name).write_bytes(data)

meta = [
    kv_string("general.architecture", "deepseek"),
    kv_u32("general.alignment", 32),
]

write(
    "truncated-metadata.gguf",
    magic + u32(version) + u64(0) + u64(1) + u64(12) + b"abc",
)
write(
    "truncated-tensor-directory.gguf",
    magic + u32(version) + u64(1) + u64(len(meta)) + b"".join(meta) + u64(12) + b"abc",
)
write(
    "invalid-metadata-count.gguf",
    magic + u32(version) + u64(0) + u64(2) + kv_string("general.architecture", "deepseek"),
)
write(
    "invalid-tensor-count.gguf",
    magic + u32(version) + u64(2) + u64(len(meta)) + b"".join(meta)
    + tensor("token_embd.weight", [4, 8], ggml_f32, 0),
)
write(
    "malformed-string.gguf",
    magic + u32(version) + u64(0) + u64(1)
    + gguf_string("general.name") + u32(string) + u64(4096),
)
write(
    "duplicate-tensor-name.gguf",
    file_bytes(
        meta,
        [
            tensor("token_embd.weight", [4, 8], ggml_f32, 0),
            tensor("token_embd.weight", [4, 8], ggml_f32, 128),
        ],
        b"\0" * 256,
    ),
)
write(
    "empty-tensor-name.gguf",
    file_bytes(meta, [tensor("", [4, 8], ggml_f32, 0)], b"\0" * 128),
)
write(
    "unknown-dtype.gguf",
    file_bytes(meta, [tensor("token_embd.weight", [4, 8], 999, 0)], b"\0" * 128),
)
write(
    "tensor-range-out-of-file.gguf",
    file_bytes(meta, [tensor("token_embd.weight", [4, 8], ggml_f32, 0)], b"\0" * 16),
)
write(
    "tensor-range-one-byte-short.gguf",
    file_bytes(meta, [tensor("token_embd.weight", [4, 8], ggml_f32, 0)], b"\0" * 127),
)
write(
    "tensor-byte-count-overflow.gguf",
    file_bytes(meta, [tensor("token_embd.weight", [(1 << 64) // 4 + 1], ggml_f32, 0)], b""),
)
write(
    "tensor-absolute-offset-overflow.gguf",
    file_bytes(meta, [tensor("token_embd.weight", [4, 8], ggml_f32, (1 << 64) - 32)], b""),
)
PY

printf 'case|integrity|inspect|tensors|materialize\n' >"$OUT_DIR/refusal-matrix.txt"

exercise_structural_case bad-magic \
    tests/fixtures/gguf/bad-magic.gguf bad-magic
exercise_structural_case unsupported-version \
    tests/fixtures/gguf/unsupported-version.gguf unsupported-version
exercise_structural_case truncated-header \
    tests/fixtures/gguf/short-header.gguf file-too-small
exercise_structural_case truncated-metadata \
    "$GEN_DIR/truncated-metadata.gguf" metadata-parse-failed,malformed-string,tensor-range-out-of-file
exercise_structural_case truncated-tensor-directory \
    "$GEN_DIR/truncated-tensor-directory.gguf" tensor-directory-parse-failed,malformed-string
exercise_structural_case invalid-metadata-count \
    "$GEN_DIR/invalid-metadata-count.gguf" metadata-parse-failed,tensor-directory-parse-failed
exercise_structural_case invalid-tensor-count \
    "$GEN_DIR/invalid-tensor-count.gguf" tensor-directory-parse-failed,empty-tensor-name,malformed-string
exercise_structural_case malformed-string \
    "$GEN_DIR/malformed-string.gguf" malformed-string
exercise_structural_case duplicate-tensor-name \
    "$GEN_DIR/duplicate-tensor-name.gguf" duplicate-tensor-name
exercise_structural_case empty-tensor-name \
    "$GEN_DIR/empty-tensor-name.gguf" empty-tensor-name
exercise_structural_case invalid-rank \
    tests/fixtures/gguf/tensor-rank-unsupported.gguf rank-out-of-range
exercise_structural_case zero-dimension \
    tests/fixtures/gguf/tensor-dim-zero.gguf zero-dimension
exercise_structural_case huge-dimension-overflow \
    tests/fixtures/gguf/tensor-dim-overflow.gguf row-byte-overflow
exercise_structural_case byte-count-overflow \
    "$GEN_DIR/tensor-byte-count-overflow.gguf" row-byte-overflow
exercise_structural_case unknown-dtype \
    "$GEN_DIR/unknown-dtype.gguf" unknown-dtype
exercise_structural_case tensor-offset-out-of-file \
    tests/fixtures/gguf/tensor-offset-out-of-bounds.gguf first-offset-not-zero
exercise_structural_case tensor-range-out-of-file \
    "$GEN_DIR/tensor-range-out-of-file.gguf" tensor-payload-truncated
exercise_structural_case tensor-range-one-byte-short \
    "$GEN_DIR/tensor-range-one-byte-short.gguf" tensor-payload-truncated
exercise_structural_case tensor-absolute-offset-overflow \
    "$GEN_DIR/tensor-absolute-offset-overflow.gguf" tensor-absolute-offset-overflow
exercise_structural_case misaligned-tensor-offset \
    tests/fixtures/gguf/tensor-offset-misaligned.gguf tensor-alignment-invalid

printf 'missing-token-embd-weight|fail-when-required|pass|pass|pass\n' >>"$OUT_DIR/refusal-matrix.txt"
expect_integrity_failure missing-token-embd-weight \
    tests/fixtures/gguf/valid-minimal.gguf required-tensor-missing --require-token-embedding --partial-token 0
run_accept missing-token-embd-weight inspect "status: descriptor-only" \
    "$YVEX_BIN" artifact show tests/fixtures/gguf/valid-minimal.gguf
run_accept missing-token-embd-weight tensors "tensor_count: 0" \
    "$YVEX_BIN" inspect artifact tensors tests/fixtures/gguf/valid-minimal.gguf
run_accept missing-token-embd-weight materialize "status: weights-partial" \
    "$YVEX_BIN" artifact materialize --model tests/fixtures/gguf/valid-minimal.gguf --backend cpu
echo "cli artifact corruption: ok"
