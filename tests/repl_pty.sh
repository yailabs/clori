#!/bin/sh
# Purpose: prove the product chat distinguishes a PTY from piped input and reaches daemon admission.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
. tests/support/cleanup.sh

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-repl-pty.XXXXXX")
runtime="$root/runtime"
mkdir -m 700 "$runtime"
cleanup()
{
    yvex_test_cleanup "$root"
}
trap cleanup EXIT HUP INT TERM

set +e
printf '/quit\n' | XDG_RUNTIME_DIR="$runtime" \
    script -q -e -c "$YVEX_BIN chat --session pty" "$root/typescript" \
    >"$root/stdout" 2>"$root/stderr"
status=$?
set -e

test "$status" -eq 1
! grep -F 'chat requires a terminal' "$root/typescript" >/dev/null
grep -F 'start it with `yvex runtime start' "$root/typescript" >/dev/null
printf 'test: repl_pty\n'
