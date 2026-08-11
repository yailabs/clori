#!/bin/sh
# Runs pinned official SDKs against the production adapter and protocol-v10 fixture host.
set -eu

YVEX_OPENAI_ADAPTER=${YVEX_OPENAI_ADAPTER:-build/tests/openai_adapter}
YVEX_OPENAI_HOST=${YVEX_OPENAI_HOST:-build/tests/openai_host}
YVEX_NODE_BIN=${YVEX_NODE_BIN:-/home/dgmothx/lab/bet-tennis/.tools/node/bin/node}
YVEX_NPM_BIN=${YVEX_NPM_BIN:-/home/dgmothx/lab/bet-tennis/.tools/node/bin/npm}
. tests/support/cleanup.sh

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-openai-sdk.XXXXXX")
socket=$root/yvexd.sock
host_pid=
gateway_pid=
cleanup()
{
    status=$?
    trap - EXIT HUP INT TERM
    test -z "$gateway_pid" || kill "$gateway_pid" 2>/dev/null || true
    test -z "$host_pid" || kill "$host_pid" 2>/dev/null || true
    test -z "$gateway_pid" || wait "$gateway_pid" 2>/dev/null || true
    test -z "$host_pid" || wait "$host_pid" 2>/dev/null || true
    if test "$status" -ne 0; then
        test ! -f "$root/gateway.err" || cat "$root/gateway.err" >&2
    fi
    yvex_test_cleanup "$root" || test "$status" -ne 0
    exit "$status"
}
trap cleanup EXIT HUP INT TERM

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
"$YVEX_OPENAI_HOST" "$socket" >"$root/host.out" 2>"$root/host.err" &
host_pid=$!
attempt=0
while test "$attempt" -lt 100; do
    test -S "$socket" && break
    sleep 0.02
    attempt=$((attempt + 1))
done
test -S "$socket"
"$YVEX_OPENAI_ADAPTER" --host 127.0.0.1 --port "$port" --yvex-socket "$socket" \
    >"$root/gateway.out" 2>"$root/gateway.err" &
gateway_pid=$!
base=http://127.0.0.1:$port
attempt=0
while test "$attempt" -lt 100; do
    if curl -fsS "$base/health" >/dev/null 2>&1; then break; fi
    sleep 0.02
    attempt=$((attempt + 1))
done
test "$attempt" -lt 100

YVEX_OPENAI_BASE_URL=$base uv run --no-project --with openai==2.50.0 \
    python tests/integration/openai_sdk.py
PATH=$(dirname "$YVEX_NODE_BIN"):$PATH "$YVEX_NPM_BIN" install \
    --prefix "$root/node" --no-save --ignore-scripts --silent openai@7.1.0
YVEX_OPENAI_BASE_URL=$base \
YVEX_OPENAI_NODE_MODULE=$root/node/node_modules/openai \
    "$YVEX_NODE_BIN" tests/integration/openai_sdk.mjs
