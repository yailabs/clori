#!/bin/sh
#
# YVEX - Tensor map CLI smoke test
#
# File: tests/cli/tensor_map.sh
# Layer: test

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/tensor-map-cli}

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR/native"

python3 - "$OUT_DIR/native/model-00001.safetensors" <<'PY'
import json
import struct
import sys

path = sys.argv[1]
header = {
    "__metadata__": {"format": "pt"},
    "embed.weight": {"dtype": "F16", "shape": [8, 4], "data_offsets": [0, 64]},
}
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"0" * 64)
PY

"$YVEX_BIN" tensor map --arch deepseek4 --native-source "$OUT_DIR/native" --limit 10 > "$OUT_DIR/map.out" 2> "$OUT_DIR/map.err" || fail "tensor-map failed"
grep 'tensor map: deepseek4' "$OUT_DIR/map.out" >/dev/null || fail "missing heading"
grep 'native=embed.weight' "$OUT_DIR/map.out" >/dev/null || fail "missing embed row"
grep 'role=token_embedding' "$OUT_DIR/map.out" >/dev/null || fail "missing role"
grep 'target=token_embd.weight' "$OUT_DIR/map.out" >/dev/null || fail "missing target"
grep 'status=mapped' "$OUT_DIR/map.out" >/dev/null || fail "missing mapped status"
grep 'target_shape=unknown' "$OUT_DIR/map.out" >/dev/null || fail "missing no-template target shape"
grep 'status: tensor-map' "$OUT_DIR/map.out" >/dev/null || fail "missing command status"

"$YVEX_BIN" tensor map \
  --arch deepseek4 \
  --native-source "$OUT_DIR/native" \
  --template tests/fixtures/gguf/valid-tokenizer-simple.gguf \
  --tensor embed.weight > "$OUT_DIR/template.out" 2> "$OUT_DIR/template.err" || fail "tensor-map template failed"
grep 'native=embed.weight' "$OUT_DIR/template.out" >/dev/null || fail "missing filtered embed row"
grep 'target=token_embd.weight' "$OUT_DIR/template.out" >/dev/null || fail "missing template target"
grep 'target_shape=\[4,8\]' "$OUT_DIR/template.out" >/dev/null || fail "missing template shape"
grep 'transform=transpose' "$OUT_DIR/template.out" >/dev/null || fail "missing transpose"

"$YVEX_BIN" tensor map --help > "$OUT_DIR/help.out" 2> "$OUT_DIR/help.err" || fail "help tensor-map failed"
grep 'usage: yvex tensor map' "$OUT_DIR/help.out" >/dev/null || fail "missing help"
