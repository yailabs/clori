#!/bin/sh
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/artifact-integrity-regression}
GEN_DIR="$OUT_DIR/generated"
MATRIX="$OUT_DIR/matrix.txt"
F16_MODEL="$OUT_DIR/deepseek4-v4-flash-selected-embed-F16-noimatrix-yvex-v1.gguf"
F32_MODEL="$OUT_DIR/deepseek4-v4-flash-fixture-embed-F32-noimatrix-yvex-v1.gguf"
REG="$OUT_DIR/models.local.json"
OLD_REG="$OUT_DIR/old.models.local.json"
STALE_REG="$OUT_DIR/stale.models.local.json"
DTYPE_REG="$OUT_DIR/dtype.models.local.json"
DIMS_REG="$OUT_DIR/dims.models.local.json"
ARCH_REG="$OUT_DIR/arch.models.local.json"
READY_REG="$OUT_DIR/readiness.models.local.json"
STALE_MODEL="$OUT_DIR/stale/deepseek4-v4-flash-selected-embed-F16-noimatrix-yvex-v1.gguf"
BAD_MAGIC=tests/fixtures/gguf/bad-magic.gguf
ALIAS=deepseek4-v4-flash-selected-embed
BAD_SHA=0000000000000000000000000000000000000000000000000000000000000000

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

append_matrix_row() {
    printf '%s|%s|%s|%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" >>"$MATRIX"
}

run_expect_fail() {
    name=$1
    surface=$2
    status=$3
    phase=$4
    allocation=$5
    dispatch=$6
    cleanup=$7
    shift 7
    out="$OUT_DIR/$name.$surface.out"
    err="$OUT_DIR/$name.$surface.err"

    if "$@" >"$out" 2>"$err"; then
        fail "$name $surface unexpectedly passed"
    fi
    contains "$out" "$status"
    append_matrix_row "$name" "$surface" "$status" "$phase" "$allocation" "$dispatch" "$cleanup"
}

run_expect_pass() {
    name=$1
    surface=$2
    status=$3
    phase=$4
    allocation=$5
    dispatch=$6
    cleanup=$7
    shift 7
    out="$OUT_DIR/$name.$surface.out"
    err="$OUT_DIR/$name.$surface.err"

    "$@" >"$out" 2>"$err"
    contains "$out" "$status"
    append_matrix_row "$name" "$surface" "$status" "$phase" "$allocation" "$dispatch" "$cleanup"
}

run_reject_smoke() {
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
    append_matrix_row "$name" "$surface" "clean-rejection" "preflight" "false" "false" "not-needed"
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

remove_digest_identity() {
    src=$1
    dst=$2
    python3 - "$src" "$dst" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
with open(src, "r", encoding="utf-8") as fh:
    data = json.load(fh)
entry = data["models"][0]
for key in ("sha256", "file_size"):
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
mkdir -p "$GEN_DIR" "$OUT_DIR/stale"
printf 'case|surface|expected_status|expected_phase|allocation_attempted|dispatch_attempted|cleanup_status\n' >"$MATRIX"

python3 - "$GEN_DIR" <<'PY'
import struct
import sys
from pathlib import Path

out = Path(sys.argv[1])
magic = b"GGUF"
version = 3
ggml_f32 = 0
uint32 = 4
string = 8

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
    return data + (b"\0" * ((alignment - (len(data) % alignment)) % alignment))

def file_bytes(metadata, tensors, payload=b"", alignment=32):
    data = magic + u32(version) + u64(len(tensors)) + u64(len(metadata))
    data += b"".join(metadata)
    data += b"".join(tensors)
    if tensors:
        data = align(data, alignment)
        data += payload
    return data

meta = [
    kv_string("general.architecture", "deepseek"),
    kv_u32("general.alignment", 32),
]

(out / "duplicate-tensor-name.gguf").write_bytes(
    file_bytes(
        meta,
        [
            tensor("token_embd.weight", [4, 8], ggml_f32, 0),
            tensor("token_embd.weight", [4, 8], ggml_f32, 128),
        ],
        b"\0" * 256,
    )
)
(out / "empty-tensor-name.gguf").write_bytes(
    file_bytes(meta, [tensor("", [4, 8], ggml_f32, 0)], b"\0" * 128)
)
(out / "unknown-dtype.gguf").write_bytes(
    file_bytes(meta, [tensor("token_embd.weight", [4, 8], 999, 0)], b"\0" * 128)
)
(out / "tensor-range-out-of-file.gguf").write_bytes(
    file_bytes(meta, [tensor("token_embd.weight", [4, 8], ggml_f32, 0)], b"\0" * 16)
)
PY

"$YVEX_BIN" artifact emit controlled \
  --out "$F16_MODEL" \
  --model-name integrity-regression-f16 \
  --arch deepseek \
  --target-qtype F16 \
  --overwrite >"$OUT_DIR/emit-f16.out" 2>"$OUT_DIR/emit-f16.err"
"$YVEX_BIN" artifact emit controlled \
  --out "$F32_MODEL" \
  --model-name integrity-regression-f32 \
  --arch deepseek \
  --target-qtype F32 \
  --overwrite >"$OUT_DIR/emit-f32.out" 2>"$OUT_DIR/emit-f32.err"

for item in \
    "bad-magic:$BAD_MAGIC" \
    "unsupported-version:tests/fixtures/gguf/unsupported-version.gguf" \
    "truncated-header:tests/fixtures/gguf/short-header.gguf" \
    "malformed-metadata:tests/fixtures/gguf/metadata-string-oob.gguf" \
    "duplicate-tensor-name:$GEN_DIR/duplicate-tensor-name.gguf" \
    "empty-tensor-name:$GEN_DIR/empty-tensor-name.gguf" \
    "zero-dimension:tests/fixtures/gguf/tensor-dim-zero.gguf" \
    "unknown-dtype:$GEN_DIR/unknown-dtype.gguf" \
    "tensor-range-out-of-file:$GEN_DIR/tensor-range-out-of-file.gguf"
do
    name=${item%%:*}
    path=${item#*:}
    run_expect_fail "$name" integrity "status: artifact-integrity-fail" preflight false false not-needed \
        "$YVEX_BIN" artifact verify check --model "$path"
    contains "$OUT_DIR/$name.integrity.out" "integrity_status: fail"
done

run_reject_smoke bad-magic inspect "status: descriptor-only" \
    "$YVEX_BIN" artifact show "$BAD_MAGIC"
run_reject_smoke bad-magic tensors "tensor_count:" \
    "$YVEX_BIN" artifact tensors "$BAD_MAGIC"
run_expect_fail bad-magic materialize "status: materialization-integrity-fail" preflight false false not-needed \
    "$YVEX_BIN" artifact materialize --model "$BAD_MAGIC" --backend cpu
contains "$OUT_DIR/bad-magic.materialize.out" "materialization_gate: fail"
contains "$OUT_DIR/bad-magic.materialize.out" "materialization_phase: preflight"
contains "$OUT_DIR/bad-magic.materialize.out" "allocation_attempted: false"

if "$YVEX_BIN" artifact materialize-gate check \
    --model "$BAD_MAGIC" \
    --label regression-bad-magic \
    --family test \
    --scope selected-tensor \
    --backend cpu \
    --require-cpu \
    --report-out "$OUT_DIR/bad-magic.materialize-gate.report" \
    >"$OUT_DIR/bad-magic.materialize-gate.out" 2>"$OUT_DIR/bad-magic.materialize-gate.err"; then
    fail "bad magic materialize-gate unexpectedly passed"
fi
contains "$OUT_DIR/bad-magic.materialize-gate.report" "materialization_gate: fail"
contains "$OUT_DIR/bad-magic.materialize-gate.report" "materialization_phase: preflight"
contains "$OUT_DIR/bad-magic.materialize-gate.report" "allocation_attempted: false"
contains "$OUT_DIR/bad-magic.materialize-gate.report" "status: materialize-gate-fail"
append_matrix_row bad-magic materialize-gate "status: materialize-gate-fail" preflight false false not-needed

"$YVEX_BIN" evidence models add \
  --path "$F16_MODEL" \
  --alias "$ALIAS" \
  --support-level selected-tensor-materialized \
  --registry "$REG" >"$OUT_DIR/models-add.out" 2>"$OUT_DIR/models-add.err"
GOOD_SHA=$(awk '/^registered_sha256: / { print $2 }' "$OUT_DIR/models-add.out")
test -n "$GOOD_SHA" || fail "missing registered sha"

run_expect_pass identity-pass models-verify "status: models-identity-pass" identity false false not-needed \
    "$YVEX_BIN" evidence models verify "$ALIAS" --registry "$REG" --audit
contains "$OUT_DIR/identity-pass.models-verify.out" "identity_status: pass"
contains "$OUT_DIR/identity-pass.models-verify.out" "metadata_status: pass"

run_expect_fail expected-sha-mismatch integrity "status: artifact-integrity-fail" identity false false not-needed \
    "$YVEX_BIN" artifact verify check --model "$F16_MODEL" --expect-sha256 "$BAD_SHA"
contains "$OUT_DIR/expected-sha-mismatch.integrity.out" "digest_status: fail"
contains "$OUT_DIR/expected-sha-mismatch.integrity.out" "error_0_code: digest-mismatch"

remove_digest_identity "$REG" "$OLD_REG"
run_expect_fail old-registry-missing-digest models-verify "status: models-identity-missing" identity false false not-needed \
    "$YVEX_BIN" evidence models verify "$ALIAS" --registry "$OLD_REG" --audit
contains "$OUT_DIR/old-registry-missing-digest.models-verify.out" "identity_status: missing"

cp "$F16_MODEL" "$STALE_MODEL"
"$YVEX_BIN" evidence models add \
  --path "$STALE_MODEL" \
  --alias "$ALIAS" \
  --support-level selected-tensor-materialized \
  --registry "$STALE_REG" >"$OUT_DIR/stale-add.out" 2>"$OUT_DIR/stale-add.err"
mutate_file_byte "$STALE_MODEL"
run_expect_fail stale-alias models-verify "status: models-identity-fail" identity false false not-needed \
    "$YVEX_BIN" evidence models verify "$ALIAS" --registry "$STALE_REG" --audit
contains "$OUT_DIR/stale-alias.models-verify.out" "digest_status: fail"
contains "$OUT_DIR/stale-alias.models-verify.out" "identity_status: fail"

run_expect_fail stale-alias materialize "status: materialization-integrity-fail" preflight false false not-needed \
    env YVEX_MODELS_REGISTRY="$STALE_REG" "$YVEX_BIN" artifact materialize --model "$ALIAS" --backend cpu
contains "$OUT_DIR/stale-alias.materialize.out" "identity_status: fail"
contains "$OUT_DIR/stale-alias.materialize.out" "allocation_attempted: false"

mutate_registry "$REG" "$DTYPE_REG" "primary_tensor_dtype" "F32"
mutate_registry "$REG" "$DIMS_REG" "primary_tensor_dims" "[4,7]"
mutate_registry "$REG" "$ARCH_REG" "architecture" "qwen"
"$YVEX_BIN" artifact emit controlled \
  --out "$OUT_DIR/readiness-F32.gguf" \
  --model-name integrity-regression-readiness \
  --arch deepseek \
  --target-qtype F32 \
  --overwrite >"$OUT_DIR/readiness-emit.out" 2>"$OUT_DIR/readiness-emit.err"
"$YVEX_BIN" evidence models add \
  --path "$OUT_DIR/readiness-F32.gguf" \
  --alias "$ALIAS" \
  --support-level selected-tensor-materialized \
  --registry "$READY_REG" >"$OUT_DIR/readiness-add.out" 2>"$OUT_DIR/readiness-add.err"
force_selected_embedding_ready "$READY_REG" "$READY_REG"

for item in \
    "metadata-dtype:$DTYPE_REG:primary-tensor-dtype-mismatch" \
    "metadata-dims:$DIMS_REG:primary-tensor-dims-mismatch" \
    "metadata-architecture:$ARCH_REG:architecture-mismatch" \
    "metadata-readiness:$READY_REG:selected-embedding-readiness-mismatch"
do
    name=$(printf '%s' "$item" | cut -d: -f1)
    reg=$(printf '%s' "$item" | cut -d: -f2)
    issue=$(printf '%s' "$item" | cut -d: -f3)
    run_expect_fail "$name" models-verify "status: models-metadata-drift" metadata false false not-needed \
        "$YVEX_BIN" evidence models verify "$ALIAS" --registry "$reg" --audit
    contains "$OUT_DIR/$name.models-verify.out" "metadata_status: fail"
    contains "$OUT_DIR/$name.models-verify.out" "$issue"
done

run_expect_fail metadata-dtype materialize "status: materialization-integrity-fail" preflight false false not-needed \
    env YVEX_MODELS_REGISTRY="$DTYPE_REG" "$YVEX_BIN" artifact materialize --model "$ALIAS" --backend cpu
contains "$OUT_DIR/metadata-dtype.materialize.out" "metadata_status: fail"
contains "$OUT_DIR/metadata-dtype.materialize.out" "allocation_attempted: false"

run_expect_pass valid-f16 materialize "status: weights-materialized" complete true false not-needed \
    "$YVEX_BIN" artifact materialize --model "$F16_MODEL" --backend cpu
contains "$OUT_DIR/valid-f16.materialize.out" "materialization_gate: pass"
contains "$OUT_DIR/valid-f16.materialize.out" "allocation_attempted: true"

run_expect_fail materialize-injected materialize "status: materialization-failed-cleaned" transfer true false pass \
    env YVEX_TEST_FAIL_MATERIALIZE_AFTER_TRANSFER=1 "$YVEX_BIN" artifact materialize --model "$F16_MODEL" --backend cpu
contains "$OUT_DIR/materialize-injected.materialize.out" "cleanup_attempted: true"
contains "$OUT_DIR/materialize-injected.materialize.out" "cleanup_status: pass"

run_expect_pass materialize-repeat materialize "status: weights-materialized" complete true false not-needed \
    "$YVEX_BIN" artifact materialize --model "$F16_MODEL" --backend cpu

"$YVEX_BIN" artifact materialize-gate check \
  --model "$F16_MODEL" \
  --label regression-valid \
  --family deepseek \
  --scope selected-tensor \
  --backend cpu \
  --require-cpu \
  --check-cleanup \
  --report-out "$OUT_DIR/valid.materialize-gate.report" \
  >"$OUT_DIR/valid.materialize-gate.out" 2>"$OUT_DIR/valid.materialize-gate.err"
contains "$OUT_DIR/valid.materialize-gate.report" "materialization_gate: pass"
contains "$OUT_DIR/valid.materialize-gate.report" "cleanup_status: pass"
contains "$OUT_DIR/valid.materialize-gate.report" "status: materialize-gate-pass"
append_matrix_row valid-f16 materialize-gate "status: materialize-gate-pass" complete true false pass

contains "$MATRIX" "bad-magic|integrity|status: artifact-integrity-fail"
contains "$MATRIX" "stale-alias|models-verify|status: models-identity-fail"
contains "$MATRIX" "metadata-dtype|models-verify|status: models-metadata-drift"
contains "$MATRIX" "materialize-injected|materialize|status: materialization-failed-cleaned"
echo "cli artifact integrity regression: ok"
