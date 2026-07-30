#!/bin/sh
#
# YVEX - Native weights CLI smoke test
#
# File: tests/cli/native_weights.sh
# Layer: test

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/native-weights-cli}

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 - "$OUT_DIR/model-00001.safetensors" <<'PY'
import json
import struct
import sys

path = sys.argv[1]
header = {
    "__metadata__": {"format": "pt"},
    "tiny.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
}
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"12345678")
PY

"$YVEX_BIN" source native --source "$OUT_DIR" > "$OUT_DIR/native.out" 2> "$OUT_DIR/native.err" || fail "native-weights failed"
grep 'native weights: safetensors' "$OUT_DIR/native.out" >/dev/null || fail "missing heading"
grep 'shards: 1' "$OUT_DIR/native.out" >/dev/null || fail "missing shard count"
grep 'tensors: 1' "$OUT_DIR/native.out" >/dev/null || fail "missing tensor count"
grep 'tiny.weight' "$OUT_DIR/native.out" >/dev/null || fail "missing tensor row"
grep 'status: native-weights' "$OUT_DIR/native.out" >/dev/null || fail "missing status"

"$YVEX_BIN" source native --source "$OUT_DIR" --tensor tiny.weight > "$OUT_DIR/tensor.out" 2> "$OUT_DIR/tensor.err" || fail "native-weights tensor failed"
grep 'tiny.weight' "$OUT_DIR/tensor.out" >/dev/null || fail "missing filtered tensor"

mkdir -p "$OUT_DIR/empty"
"$YVEX_BIN" source native --source "$OUT_DIR/empty" > "$OUT_DIR/empty.out" 2> "$OUT_DIR/empty.err" || fail "native-weights empty failed"
grep 'shards: 0' "$OUT_DIR/empty.out" >/dev/null || fail "missing empty shard count"
grep 'status: native-weights-empty' "$OUT_DIR/empty.out" >/dev/null || fail "missing empty status"
