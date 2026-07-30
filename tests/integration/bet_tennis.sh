#!/bin/sh
# Purpose: prove the unchanged external bet-tennis provider through configuration only.
set -eu

YVEX_OPENAI_ADAPTER=${YVEX_OPENAI_ADAPTER:-build/tests/openai_adapter}
YVEX_OPENAI_HOST=${YVEX_OPENAI_HOST:-build/tests/openai_host}
YVEX_BET_TENNIS_ROOT=${YVEX_BET_TENNIS_ROOT:-/home/dgmothx/lab/bet-tennis}
. tests/support/cleanup.sh

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-openai-bet-tennis.XXXXXX")
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

test -x "$YVEX_BET_TENNIS_ROOT/.venv/bin/python"
before=$(git -C "$YVEX_BET_TENNIS_ROOT" status --porcelain)
commit=$(git -C "$YVEX_BET_TENNIS_ROOT" rev-parse HEAD)
snapshot=$root/bet-tennis
mkdir -p "$snapshot"
git -C "$YVEX_BET_TENNIS_ROOT" archive "$commit" | tar -x -C "$snapshot"
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

PYTHONPATH="$snapshot/src" YVEX_OPENAI_BASE_URL=$base \
    "$YVEX_BET_TENNIS_ROOT/.venv/bin/python" tests/integration/bet_tennis.py
after=$(git -C "$YVEX_BET_TENNIS_ROOT" status --porcelain)
test "$before" = "$after"
printf 'bet-tennis commit %s: unchanged provider acceptance passed\n' "$commit"
