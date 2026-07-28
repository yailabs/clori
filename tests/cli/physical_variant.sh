#!/bin/sh
#
# YVEX - Physical-variant CLI admission smoke test
#

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/physical-variant-cli}

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

expect_rc() {
    expected=$1
    shift
    set +e
    "$@"
    actual=$?
    set -e
    test "$actual" -eq "$expected" || fail "expected rc $expected, got $actual: $*"
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$YVEX_BIN" quant preset list > "$OUT_DIR/list.out" 2> "$OUT_DIR/list.err" ||
    fail "preset list failed"
grep '^source-faithful$' "$OUT_DIR/list.out" >/dev/null || fail "source-faithful missing"
grep '^deepseek-v4-flash-q8_0-q2_k-v1$' "$OUT_DIR/list.out" >/dev/null ||
    fail "release preset missing"
grep '^deepseek-v4-flash-ds4-like-q2-v1$' "$OUT_DIR/list.out" >/dev/null ||
    fail "DS4-like preset missing"

"$YVEX_BIN" quant preset show deepseek-v4-flash-ds4-like-q2-v1 \
    > "$OUT_DIR/show.out" 2> "$OUT_DIR/show.err" || fail "preset show failed"
grep '^schema_version: 2$' "$OUT_DIR/show.out" >/dev/null || fail "schema v2 missing"
grep '^imatrix_rules: 3$' "$OUT_DIR/show.out" >/dev/null || fail "imatrix rules missing"

"$YVEX_BIN" quant --help > "$OUT_DIR/help.out" 2> "$OUT_DIR/help.err" ||
    fail "quant help failed"
grep 'yvex quant plan' "$OUT_DIR/help.out" >/dev/null || fail "plan grammar missing"
grep 'Materialization never chooses qtypes' "$OUT_DIR/help.out" >/dev/null ||
    fail "ownership boundary missing"

expect_rc 1 "$YVEX_BIN" quant preset show no-such-preset \
    > "$OUT_DIR/unknown.out" 2> "$OUT_DIR/unknown.err"
expect_rc 2 "$YVEX_BIN" quant plan --target deepseek4-v4-flash \
    > "$OUT_DIR/incomplete.out" 2> "$OUT_DIR/incomplete.err"
expect_rc 2 "$YVEX_BIN" quant nope > "$OUT_DIR/bad-action.out" 2> "$OUT_DIR/bad-action.err"

printf 'physical variant cli: ok\n'
