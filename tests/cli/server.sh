#!/bin/sh
# Purpose: prove the current yvexd process grammar and fail-closed listener admission.
set -eu

. tests/support/cleanup.sh

YVEXD_BIN=${YVEXD_BIN:-./yvexd}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-server}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

fail()
{
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains()
{
    grep -F -- "$2" "$1" >/dev/null || fail "$1 missing: $2"
}

"$YVEXD_BIN" --help >"$OUT_DIR/help.out" 2>"$OUT_DIR/help.err"
contains "$OUT_DIR/help.out" 'usage: yvexd --model ARTIFACT --runtime-binding FILE'
contains "$OUT_DIR/help.out" '[--openai on|off]'
contains "$OUT_DIR/help.out" 'loopback OpenAI listener'
"$YVEXD_BIN" --version >"$OUT_DIR/version.out" 2>"$OUT_DIR/version.err"
contains "$OUT_DIR/version.out" '0.1.0 protocol=3'

set +e
"$YVEXD_BIN" >"$OUT_DIR/missing.out" 2>"$OUT_DIR/missing.err"
missing_status=$?
"$YVEXD_BIN" --model missing --runtime-binding missing --openai remote \
    >"$OUT_DIR/remote.out" 2>"$OUT_DIR/remote.err"
remote_status=$?
"$YVEXD_BIN" --model missing --runtime-binding missing --openai-port 0 \
    >"$OUT_DIR/port.out" 2>"$OUT_DIR/port.err"
port_status=$?
"$YVEXD_BIN" --model missing --runtime-binding missing --openai on --openai off \
    >"$OUT_DIR/duplicate.out" 2>"$OUT_DIR/duplicate.err"
duplicate_status=$?
set -e
test "$missing_status" -eq 2
test "$remote_status" -eq 2
test "$port_status" -eq 2
test "$duplicate_status" -eq 2
contains "$OUT_DIR/missing.err" '--model and --runtime-binding are required'
contains "$OUT_DIR/port.err" 'invalid or duplicate --openai-port'
contains "$OUT_DIR/duplicate.err" 'duplicate --openai option'

printf 'cli server grammar: ok\n'
