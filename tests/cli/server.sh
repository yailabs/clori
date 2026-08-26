#!/bin/sh
# Verifies the persistent host grammar and fail-closed engine admission.
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-server}
HOME_ROOT=$OUT_DIR/home
SOCKET_ROOT=$OUT_DIR/runtime
SOCKET_PATH=$SOCKET_ROOT/yvex/yvexd.sock
PROFILE=deepseek4-v4-flash-dspark-runtime-iq2xxs-q2k-mxfp4-b9825a07-sm121-tc
server_pid=

finish()
{
    status=$?
    trap - EXIT HUP INT TERM
    if test -n "$server_pid" && kill -0 "$server_pid" 2>/dev/null; then
        HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" \
            "$YVEX_BIN" server stop >/dev/null 2>&1 || true
        wait "$server_pid" 2>/dev/null || true
    fi
    yvex_test_cleanup_preserving_status "$status" "$OUT_DIR"
}
trap finish EXIT HUP INT TERM

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$HOME_ROOT/.local/share/yvex" "$SOCKET_ROOT"
HOME_ROOT=$(realpath "$HOME_ROOT")
SOCKET_ROOT=$(realpath "$SOCKET_ROOT")
SOCKET_PATH=$SOCKET_ROOT/yvex/yvexd.sock
chmod 0700 "$SOCKET_ROOT"

fail()
{
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains()
{
    grep -F -- "$2" "$1" >/dev/null || fail "$1 missing: $2"
}

run_client()
{
    HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" "$YVEX_BIN" "$@"
}

artifact=$OUT_DIR/current.gguf
binding=$OUT_DIR/current.binding
printf 'artifact fixture\n' >"$artifact"
printf 'binding fixture\n' >"$binding"
artifact=$(realpath "$artifact")
binding=$(realpath "$binding")
cat >"$HOME_ROOT/.local/share/yvex/models.local.json" <<EOF
{
  "schema": "yvex.models.local.v5",
  "models": [{
    "alias": "$PROFILE",
    "family": "deepseek4",
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
contains "$OUT_DIR/help.out" 'usage: yvex server [options]'
contains "$OUT_DIR/help.out" 'Run the persistent multi-engine host in the foreground.'
contains "$OUT_DIR/help.out" '--workers'
contains "$OUT_DIR/help.out" '--openai'
! grep -F -- '--ctx' "$OUT_DIR/help.out" >/dev/null
! grep -F -- '--backend' "$OUT_DIR/help.out" >/dev/null
! grep -F -- '--generation-mode' "$OUT_DIR/help.out" >/dev/null
! grep -F -- '--media-artifact-root' "$OUT_DIR/help.out" >/dev/null

"$YVEX_BIN" server load --help >"$OUT_DIR/load-help.out"
"$YVEX_BIN" server unload --help >"$OUT_DIR/unload-help.out"
"$YVEX_BIN" server models --help >"$OUT_DIR/models-help.out"
contains "$OUT_DIR/load-help.out" 'usage: yvex server load MODEL'
contains "$OUT_DIR/unload-help.out" 'usage: yvex server unload MODEL'
contains "$OUT_DIR/models-help.out" 'usage: yvex server models [options]'

set +e
"$YVEX_BIN" server --backend cpu >"$OUT_DIR/backend.out" 2>"$OUT_DIR/backend.err"
backend_status=$?
"$YVEX_BIN" server --workers 0 >"$OUT_DIR/workers.out" 2>"$OUT_DIR/workers.err"
workers_status=$?
"$YVEX_BIN" server --openai remote >"$OUT_DIR/remote.out" 2>"$OUT_DIR/remote.err"
remote_status=$?
"$YVEX_BIN" server --openai-port 0 >"$OUT_DIR/port.out" 2>"$OUT_DIR/port.err"
port_status=$?
"$YVEX_BIN" server --openai on --openai off \
    >"$OUT_DIR/duplicate.out" 2>"$OUT_DIR/duplicate.err"
duplicate_status=$?
set -e

test "$backend_status" -eq 2
test "$workers_status" -eq 2
test "$remote_status" -eq 2
test "$port_status" -eq 2
test "$duplicate_status" -eq 2
contains "$OUT_DIR/backend.err" 'unknown flag: --backend'
contains "$OUT_DIR/workers.err" 'invalid value for --workers: 0'
contains "$OUT_DIR/remote.err" 'invalid value for --openai: remote'
contains "$OUT_DIR/port.err" 'invalid value for --openai-port: 0'
contains "$OUT_DIR/duplicate.err" 'duplicate flag: --openai'

HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" \
    "$YVEX_BIN" server --console off --openai off --workers 2 \
    >"$OUT_DIR/host.out" 2>"$OUT_DIR/host.err" &
server_pid=$!

ready=0
attempt=0
while test "$attempt" -lt 100; do
    if run_client server status --json >"$OUT_DIR/status.json" 2>"$OUT_DIR/status.err"; then
        ready=1
        break
    fi
    kill -0 "$server_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.02
done
test "$ready" -eq 1 || fail 'persistent host did not become ready'
contains "$OUT_DIR/host.out" 'YVEX server · persistent host'
contains "$OUT_DIR/host.out" 'engines 0/'
contains "$OUT_DIR/host.out" 'load with `yvex server load MODEL`'
contains "$OUT_DIR/status.json" '"protocol":13'
contains "$OUT_DIR/status.json" '"status":2'
contains "$OUT_DIR/status.json" '"host_ready":true'
contains "$OUT_DIR/status.json" '"engine_count":0'
contains "$OUT_DIR/status.json" '"loaded_engine_count":0'
contains "$OUT_DIR/status.json" '"workers":2'
contains "$OUT_DIR/status.json" '"model_open_count":0'
contains "$OUT_DIR/status.json" '"openai_enabled":false'

run_client server models --json >"$OUT_DIR/models-empty.json"
contains "$OUT_DIR/models-empty.json" '"schema":"yvex.server.engines.v1"'
contains "$OUT_DIR/models-empty.json" '"engines":[]'

set +e
run_client server load absent >"$OUT_DIR/load-absent.out" 2>"$OUT_DIR/load-absent.err"
absent_status=$?
run_client server load "$PROFILE" >"$OUT_DIR/load.out" 2>"$OUT_DIR/load.err"
load_status=$?
set -e
test "$absent_status" -eq 1
test "$load_status" -eq 1
contains "$OUT_DIR/load-absent.err" 'model is not registered: absent'
contains "$OUT_DIR/load.err" 'runtime binding open failed'

run_client server status --json >"$OUT_DIR/status-after-failure.json"
contains "$OUT_DIR/status-after-failure.json" '"host_ready":true'
contains "$OUT_DIR/status-after-failure.json" '"engine_count":1'
contains "$OUT_DIR/status-after-failure.json" '"loaded_engine_count":0'
contains "$OUT_DIR/status-after-failure.json" '"model_open_count":0'
run_client server models --json >"$OUT_DIR/models-failed.json"
contains "$OUT_DIR/models-failed.json" "\"alias\":\"$PROFILE\""
contains "$OUT_DIR/models-failed.json" '"generation":1'
contains "$OUT_DIR/models-failed.json" '"state":"failed"'
contains "$OUT_DIR/models-failed.json" '"execution_ready":false'

set +e
run_client server unload "$PROFILE" >"$OUT_DIR/unload.out" 2>"$OUT_DIR/unload.err"
unload_status=$?
set -e
test "$unload_status" -eq 1
contains "$OUT_DIR/unload.err" 'requested model engine is not loaded'

run_client server stop >"$OUT_DIR/stop.out" 2>"$OUT_DIR/stop.err"
wait "$server_pid"
server_pid=
test ! -e "$SOCKET_PATH"

printf 'cli persistent server lifecycle: ok\n'
