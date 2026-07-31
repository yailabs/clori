#!/bin/sh
#
# YVEX - GGUF template CLI smoke test
#
# File: tests/cli/gguf_template.sh
# Layer: test

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/gguf-template-cli}
FIX=tests/fixtures/gguf/valid-tokenizer-simple.gguf

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$YVEX_BIN" compile emit template inspect --template "$FIX" > "$OUT_DIR/inspect.out" 2> "$OUT_DIR/inspect.err" || fail "inspect failed"
grep 'gguf template: inspect' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect heading"
grep 'status: template-' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect status"

"$YVEX_BIN" compile emit template validate --template "$FIX" > "$OUT_DIR/validate.out" 2> "$OUT_DIR/validate.err" || fail "validate failed"
grep 'gguf template: validate' "$OUT_DIR/validate.out" >/dev/null || fail "missing validate heading"
grep 'status: template-' "$OUT_DIR/validate.out" >/dev/null || fail "missing validate status"
grep 'issues:' "$OUT_DIR/validate.out" >/dev/null || fail "missing issues"

"$YVEX_BIN" compile emit template --help > "$OUT_DIR/help.out" 2> "$OUT_DIR/help.err" || fail "help failed"
grep 'usage: yvex compile emit template' "$OUT_DIR/help.out" >/dev/null || fail "missing help usage"
