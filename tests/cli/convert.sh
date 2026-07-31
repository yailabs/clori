#!/bin/sh
#
# YVEX - conversion CLI smoke test

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/convert-cli}
NATIVE="$OUT_DIR/native"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$NATIVE"

OUT_DIR="$OUT_DIR" python3 - <<'PY'
import json, struct, pathlib
import os
root = pathlib.Path(os.environ["OUT_DIR"]) / "native"
payload = bytes(range(64))
header = {
    "model.embed_tokens.weight": {
        "dtype": "F16",
        "shape": [8, 4],
        "data_offsets": [0, len(payload)],
    }
}
raw = json.dumps(header, separators=(",", ":")).encode()
(root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
PY

cat > "$OUT_DIR/policy.json" <<'JSON'
{
  "schema": "yvex.quant_policy.v1",
  "name": "test-qwen-policy",
  "architecture": "qwen3",
  "rules": [
    {"selector_kind": "role", "selector": "token_embedding", "qtype": "F32", "requires_imatrix": false}
  ]
}
JSON

"$YVEX_BIN" inspect qtype > "$OUT_DIR/qtype.out" 2> "$OUT_DIR/qtype.err" || fail "qtype support failed"
grep 'status: qtype-support' "$OUT_DIR/qtype.out" >/dev/null || fail "missing qtype status"

"$YVEX_BIN" compile quant convert plan \
  --arch qwen3 \
  --native-source "$NATIVE" \
  --quant-policy "$OUT_DIR/policy.json" \
  --out-plan "$OUT_DIR/plan.json" > "$OUT_DIR/plan.out" 2> "$OUT_DIR/plan.err" || fail "plan failed"
test -f "$OUT_DIR/plan.json" || fail "plan missing"
grep 'status: conversion-plan-written' "$OUT_DIR/plan.out" >/dev/null || fail "missing plan status"

"$YVEX_BIN" compile quant convert emit \
  --arch qwen3 \
  --native-source "$NATIVE" \
  --tensor model.embed_tokens.weight \
  --target-qtype F32 \
  --out "$OUT_DIR/qwen3-8b-selected-embed-F32-noimatrix-yvex-v1.gguf" \
  --overwrite > "$OUT_DIR/emit.out" 2> "$OUT_DIR/emit.err" || fail "emit failed"
grep 'status: conversion-gguf-written' "$OUT_DIR/emit.out" >/dev/null || fail "missing emit status"

"$YVEX_BIN" artifact show "$OUT_DIR/qwen3-8b-selected-embed-F32-noimatrix-yvex-v1.gguf" > "$OUT_DIR/inspect.out" 2> "$OUT_DIR/inspect.err" || fail "inspect failed"
grep 'status: descriptor-only' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect status"
"$YVEX_BIN" artifact materialize --model "$OUT_DIR/qwen3-8b-selected-embed-F32-noimatrix-yvex-v1.gguf" --backend cpu > "$OUT_DIR/materialize.out" 2> "$OUT_DIR/materialize.err" || fail "materialize failed"
grep 'status: weights-materialized' "$OUT_DIR/materialize.out" >/dev/null || fail "missing materialize status"
"$YVEX_BIN" compile quant convert --help > "$OUT_DIR/help.out" 2> "$OUT_DIR/help.err" || fail "help failed"
grep 'usage: yvex compile quant convert' "$OUT_DIR/help.out" >/dev/null || fail "missing help"
