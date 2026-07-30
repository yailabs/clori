#!/bin/sh
#
# YVEX - Quant policy CLI smoke test
#
# File: tests/cli/quant_policy.sh
# Layer: test

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/quant-policy-cli}

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/policy.json" <<'JSON'
{
  "schema": "yvex.quant_policy.v1",
  "name": "test-policy",
  "architecture": "deepseek4",
  "rules": [
    {"selector_kind": "role", "selector": "token_embedding", "qtype": "Q8_0", "requires_imatrix": false},
    {"selector_kind": "pattern", "selector": "blk.*.ffn.experts.*", "qtype": "Q2_K", "requires_imatrix": true}
  ]
}
JSON

"$YVEX_BIN" quant policy inspect --policy "$OUT_DIR/policy.json" > "$OUT_DIR/inspect.out" 2> "$OUT_DIR/inspect.err" || fail "inspect failed"
grep 'quant policy: inspect' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect heading"
grep 'selector=role:token_embedding qtype=Q8_0' "$OUT_DIR/inspect.out" >/dev/null || fail "missing role rule"
grep 'requires_imatrix=yes' "$OUT_DIR/inspect.out" >/dev/null || fail "missing imatrix flag"

"$YVEX_BIN" quant policy validate --policy "$OUT_DIR/policy.json" > "$OUT_DIR/validate.out" 2> "$OUT_DIR/validate.err" || fail "validate failed"
grep 'quant policy: validate' "$OUT_DIR/validate.out" >/dev/null || fail "missing validate heading"
grep 'status: quant-policy-' "$OUT_DIR/validate.out" >/dev/null || fail "missing validate status"

"$YVEX_BIN" quant policy derive \
  --template tests/fixtures/gguf/valid-tokenizer-simple.gguf \
  --arch llama \
  --out "$OUT_DIR/derived.json" > "$OUT_DIR/derive.out" 2> "$OUT_DIR/derive.err" || fail "derive failed"
test -f "$OUT_DIR/derived.json" || fail "derived policy missing"
grep 'quant policy: derived' "$OUT_DIR/derive.out" >/dev/null || fail "missing derive heading"
grep 'status: quant-policy-written' "$OUT_DIR/derive.out" >/dev/null || fail "missing derive status"

"$YVEX_BIN" quant policy validate --policy "$OUT_DIR/derived.json" --template tests/fixtures/gguf/valid-tokenizer-simple.gguf > "$OUT_DIR/derived-validate.out" 2> "$OUT_DIR/derived-validate.err" || fail "derived validate failed"
grep 'status: quant-policy-' "$OUT_DIR/derived-validate.out" >/dev/null || fail "missing derived validate status"

"$YVEX_BIN" quant policy --help > "$OUT_DIR/help.out" 2> "$OUT_DIR/help.err" || fail "help failed"
grep 'usage: yvex quant policy' "$OUT_DIR/help.out" >/dev/null || fail "missing help"
