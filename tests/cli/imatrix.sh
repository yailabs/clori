#!/bin/sh
#
# YVEX - Imatrix CLI smoke test
#
# File: tests/cli/imatrix.sh
# Layer: test

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/imatrix-cli}

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

printf 'fake-imatrix' > "$OUT_DIR/fake.dat"

"$YVEX_BIN" compile quant imatrix create \
  --name test-imatrix \
  --arch deepseek4 \
  --imatrix "$OUT_DIR/fake.dat" \
  --format routed_moe_dat \
  --status present \
  --dataset test-dataset \
  --producer test \
  --out "$OUT_DIR/imatrix.json" > "$OUT_DIR/create.out" 2> "$OUT_DIR/create.err" || fail "create failed"

test -f "$OUT_DIR/imatrix.json" || fail "manifest missing"
grep 'imatrix manifest: written' "$OUT_DIR/create.out" >/dev/null || fail "missing create heading"
grep 'file_exists: yes' "$OUT_DIR/create.out" >/dev/null || fail "missing file_exists"
grep 'status: imatrix-manifest-written' "$OUT_DIR/create.out" >/dev/null || fail "missing create status"

"$YVEX_BIN" compile quant imatrix inspect --manifest "$OUT_DIR/imatrix.json" > "$OUT_DIR/inspect.out" 2> "$OUT_DIR/inspect.err" || fail "inspect failed"
grep 'imatrix: inspect' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect heading"
grep 'status: imatrix-manifest' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect status"

"$YVEX_BIN" compile quant imatrix validate --manifest "$OUT_DIR/imatrix.json" > "$OUT_DIR/validate.out" 2> "$OUT_DIR/validate.err" || fail "validate failed"
grep 'imatrix: validate' "$OUT_DIR/validate.out" >/dev/null || fail "missing validate heading"
grep 'requires_imatrix_rules: 0' "$OUT_DIR/validate.out" >/dev/null || fail "missing requires count"
grep 'covered_rules: 0' "$OUT_DIR/validate.out" >/dev/null || fail "missing covered count"
grep 'status: imatrix-valid' "$OUT_DIR/validate.out" >/dev/null || fail "missing validate status"

"$YVEX_BIN" compile quant imatrix --help > "$OUT_DIR/help.out" 2> "$OUT_DIR/help.err" || fail "help failed"
grep 'usage: yvex compile quant imatrix' "$OUT_DIR/help.out" >/dev/null || fail "missing help"
