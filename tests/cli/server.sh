#!/bin/sh
# Verifies the direct foreground server grammar and fail-closed model admission.
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-server}
HOME_ROOT=$OUT_DIR/home
SOCKET_ROOT=$OUT_DIR/runtime
SOCKET_PATH=$SOCKET_ROOT/yvex.sock
PROFILE=deepseek4-v4-flash-dspark-runtime-iq2xxs-q2k-mxfp4-b9825a07-sm121-tc

yvex_test_cleanup "$OUT_DIR" "$SOCKET_PATH"
mkdir -p "$OUT_DIR" "$HOME_ROOT/.local/share/yvex" "$SOCKET_ROOT"
chmod 0700 "$SOCKET_ROOT"
SOCKET_PATH=$(realpath "$SOCKET_ROOT")/yvex.sock

fail()
{
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains()
{
    grep -F -- "$2" "$1" >/dev/null || fail "$1 missing: $2"
}

artifact=$OUT_DIR/current.gguf
binding=$OUT_DIR/current.binding
printf 'artifact fixture\n' >"$artifact"
printf 'binding fixture\n' >"$binding"
artifact=$(realpath "$artifact")
binding=$(realpath "$binding")
cat >"$HOME_ROOT/.local/share/yvex/models.local.json" <<EOF
{
  "schema": "yvex.models.local.v3",
  "models": [{
    "alias": "$PROFILE",
    "path": "$artifact",
    "runtime_binding": "$binding",
    "runtime_target": "deepseek4-v4-flash-dspark",
    "runtime_backend": "cpu",
    "runtime_mode": "target-only",
    "runtime_context": 4096
  }]
}
EOF

"$YVEX_BIN" server --help >"$OUT_DIR/help.out" 2>"$OUT_DIR/help.err"
contains "$OUT_DIR/help.out" 'usage: yvex server MODEL [options]'
contains "$OUT_DIR/help.out" 'Run one model server in the foreground.'
contains "$OUT_DIR/help.out" '--ctx'
contains "$OUT_DIR/help.out" '--parallel'
! grep -F -- '--context' "$OUT_DIR/help.out" >/dev/null
contains "$OUT_DIR/help.out" '--openai'

set +e
HOME="$HOME_ROOT" "$YVEX_BIN" server >"$OUT_DIR/missing.out" 2>"$OUT_DIR/missing.err"
missing_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server absent >"$OUT_DIR/absent.out" 2>"$OUT_DIR/absent.err"
absent_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server "$PROFILE" --openai remote \
    >"$OUT_DIR/remote.out" 2>"$OUT_DIR/remote.err"
remote_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server "$PROFILE" --openai-port 0 \
    >"$OUT_DIR/port.out" 2>"$OUT_DIR/port.err"
port_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server "$PROFILE" --openai on --openai off \
    >"$OUT_DIR/duplicate.out" 2>"$OUT_DIR/duplicate.err"
duplicate_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server "$PROFILE" --generation-mode invalid \
    >"$OUT_DIR/mode.out" 2>"$OUT_DIR/mode.err"
mode_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server "$PROFILE" --context 8192 \
    >"$OUT_DIR/context.out" 2>"$OUT_DIR/context.err"
context_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server "$PROFILE" \
    --ctx 8192 --parallel 2 --socket "$SOCKET_PATH" --openai off \
    >"$OUT_DIR/admission.out" 2>"$OUT_DIR/admission.err"
admission_status=$?
set -e

test "$missing_status" -eq 2
test "$absent_status" -eq 1
test "$remote_status" -eq 2
test "$port_status" -eq 2
test "$duplicate_status" -eq 2
test "$mode_status" -eq 2
test "$context_status" -eq 2
test "$admission_status" -eq 1
contains "$OUT_DIR/missing.err" 'usage: yvex server MODEL [options]'
contains "$OUT_DIR/absent.err" 'model is not registered: absent'
contains "$OUT_DIR/remote.err" 'invalid value for --openai: remote'
contains "$OUT_DIR/port.err" 'invalid value for --openai-port: 0'
contains "$OUT_DIR/duplicate.err" 'duplicate flag: --openai'
contains "$OUT_DIR/mode.err" 'invalid value for --generation-mode: invalid'
contains "$OUT_DIR/context.err" 'unknown flag: --context'
contains "$OUT_DIR/admission.out" 'YVEX server · foreground'
contains "$OUT_DIR/admission.out" "profile $PROFILE"
contains "$OUT_DIR/admission.out" 'backend=cpu · mode=target-only · requested ctx=8192 · parallel=2'
contains "$OUT_DIR/admission.out" "artifact $artifact"
contains "$OUT_DIR/admission.out" "binding $binding"
contains "$OUT_DIR/admission.out" 'stop with Ctrl-C or `yvex server stop`'
contains "$OUT_DIR/admission.err" 'startup refused before readiness (elapsed '
yvex_test_cleanup "$SOCKET_PATH"

printf 'cli server grammar: ok\n'
