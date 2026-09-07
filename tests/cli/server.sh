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
LEGACY_PROFILE=deepseek4-v4-flash-dspark-runtime-iq2xxs-legacy
server_pid=
logs_pid=

finish()
{
    status=$?
    trap - EXIT HUP INT TERM
    if test -n "$server_pid" && kill -0 "$server_pid" 2>/dev/null; then
        HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" \
            "$YVEX_BIN" host stop >/dev/null 2>&1 || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if test -n "$logs_pid" && kill -0 "$logs_pid" 2>/dev/null; then
        kill "$logs_pid" 2>/dev/null || true
        wait "$logs_pid" 2>/dev/null || true
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

# Link the actual renderer; section GC excludes unrelated porcelain consumers.
${CC:-cc} -std=c11 -Wall -Wextra -Werror -D_POSIX_C_SOURCE=200809L -I. -Iinclude \
    -Ibuild/generated -ffunction-sections -fdata-sections \
    tests/cli/logs.c src/cli/io/out.c src/core/status.c \
    -Wl,--gc-sections -o "$OUT_DIR/log-renderer"
NO_COLOR=1 "$OUT_DIR/log-renderer" >"$OUT_DIR/log-renderer.out"
env -u NO_COLOR TERM=xterm script -q -e -c "$OUT_DIR/log-renderer" \
    "$OUT_DIR/log-renderer.tty" >"$OUT_DIR/log-renderer.color"
python3 - "$OUT_DIR/log-renderer.out" "$OUT_DIR/log-renderer.color" <<'PY'
import pathlib, re, sys
plain, color = [pathlib.Path(path).read_bytes().decode() for path in sys.argv[1:]]
assert '\x1b' not in plain and '\r' not in plain
assert '\x1b[' in color
assert re.sub(r'\x1b\[[0-9;]*m', '', color).replace('\r\n', '\n') == plain
PY

fail()
{
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains()
{
    grep -F -- "$2" "$1" >/dev/null || fail "$1 missing: $2"
}

not_contains()
{
    if grep -F -- "$2" "$1" >/dev/null; then
        fail "$1 unexpectedly contains: $2"
    fi
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
  "schema": "yvex.models.local.v6",
  "models": [{
    "alias": "$LEGACY_PROFILE",
    "family": "deepseek4",
    "path": "$artifact",
    "runtime_binding": "$binding",
    "runtime_target": "deepseek4-v4-flash-dspark",
    "runtime_backend": "cpu",
    "runtime_engine_kind": "text",
    "runtime_execution_strategy": "target-only",
    "runtime_context": 4096
  }, {
    "alias": "$PROFILE",
    "family": "deepseek4",
    "path": "$artifact",
    "runtime_binding": "$binding",
    "runtime_target": "deepseek4-v4-flash-dspark",
    "runtime_backend": "cpu",
    "runtime_engine_kind": "text",
    "runtime_execution_strategy": "target-only",
    "runtime_context": 4096
  }]
}
EOF

"$YVEX_BIN" serve --help >"$OUT_DIR/help.out" 2>"$OUT_DIR/help.err"
contains "$OUT_DIR/help.out" 'usage: yvex serve [options]'
contains "$OUT_DIR/help.out" 'Run the persistent YVEX host in the foreground.'
contains "$OUT_DIR/help.out" '--workers'
contains "$OUT_DIR/help.out" '--max-engines'
contains "$OUT_DIR/help.out" '--openai'
! grep -F -- '--ctx' "$OUT_DIR/help.out" >/dev/null
! grep -F -- '--backend' "$OUT_DIR/help.out" >/dev/null
! grep -F -- '--generation-mode' "$OUT_DIR/help.out" >/dev/null
! grep -F -- '--media-artifact-root' "$OUT_DIR/help.out" >/dev/null

"$YVEX_BIN" engine load --help >"$OUT_DIR/load-help.out"
"$YVEX_BIN" engine unload --help >"$OUT_DIR/unload-help.out"
"$YVEX_BIN" engine list --help >"$OUT_DIR/models-help.out"
"$YVEX_BIN" model load --help >"$OUT_DIR/model-load-help.out"
contains "$OUT_DIR/load-help.out" 'usage: yvex engine load [PROFILE]'
contains "$OUT_DIR/unload-help.out" 'usage: yvex engine unload ENGINE'
contains "$OUT_DIR/models-help.out" 'usage: yvex engine list [options]'
contains "$OUT_DIR/model-load-help.out" 'usage: yvex model load [MODEL]'

set +e
run_client engine load >"$OUT_DIR/load-nontty.out" 2>"$OUT_DIR/load-nontty.err"
load_nontty_status=$?
HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" NO_COLOR=1 \
    TERM=xterm-256color script -q -e -c "$YVEX_BIN engine load" \
    "$OUT_DIR/engine-load.typescript" </dev/null >"$OUT_DIR/engine-load.out" \
    2>"$OUT_DIR/engine-load.err"
engine_load_tty_status=$?
run_client model load >"$OUT_DIR/model-load-nontty.out" \
    2>"$OUT_DIR/model-load-nontty.err"
model_load_nontty_status=$?
printf '1\n1\n' | HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" NO_COLOR=1 \
    TERM=xterm-256color script -q -e -c "$YVEX_BIN model load" \
    "$OUT_DIR/load-selector.typescript" >"$OUT_DIR/load-selector.out" \
    2>"$OUT_DIR/load-selector.err"
load_selector_status=$?
set -e
test "$load_nontty_status" -eq 2
test "$engine_load_tty_status" -eq 2
test "$model_load_nontty_status" -eq 2
test "$load_selector_status" -eq 1
contains "$OUT_DIR/load-nontty.err" 'engine load requires PROFILE'
contains "$OUT_DIR/load-nontty.err" 'yvex model load [MODEL]'
contains "$OUT_DIR/engine-load.typescript" 'engine load requires PROFILE'
not_contains "$OUT_DIR/engine-load.typescript" 'Select model'
contains "$OUT_DIR/model-load-nontty.err" 'model load requires MODEL when input is not a terminal'
contains "$OUT_DIR/load-selector.typescript" 'no launchable models are known locally'
not_contains "$OUT_DIR/load-selector.typescript" 'Select model'
not_contains "$OUT_DIR/load-selector.typescript" 'Select representation and deployment'
not_contains "$OUT_DIR/load-selector.typescript" "$PROFILE"
not_contains "$OUT_DIR/load-selector.typescript" "$LEGACY_PROFILE"

set +e
"$YVEX_BIN" serve --backend cpu >"$OUT_DIR/backend.out" 2>"$OUT_DIR/backend.err"
backend_status=$?
"$YVEX_BIN" serve --workers 0 >"$OUT_DIR/workers.out" 2>"$OUT_DIR/workers.err"
workers_status=$?
"$YVEX_BIN" serve --max-engines 0 >"$OUT_DIR/engines.out" 2>"$OUT_DIR/engines.err"
engines_status=$?
"$YVEX_BIN" serve --openai remote >"$OUT_DIR/remote.out" 2>"$OUT_DIR/remote.err"
remote_status=$?
"$YVEX_BIN" serve --openai-port 0 >"$OUT_DIR/port.out" 2>"$OUT_DIR/port.err"
port_status=$?
"$YVEX_BIN" serve --openai on --openai off \
    >"$OUT_DIR/duplicate.out" 2>"$OUT_DIR/duplicate.err"
duplicate_status=$?
set -e

test "$backend_status" -eq 2
test "$workers_status" -eq 2
test "$engines_status" -eq 2
test "$remote_status" -eq 2
test "$port_status" -eq 2
test "$duplicate_status" -eq 2
contains "$OUT_DIR/backend.err" 'unknown flag: --backend'
contains "$OUT_DIR/workers.err" 'invalid value for --workers: 0'
contains "$OUT_DIR/engines.err" 'invalid value for --max-engines: 0'
contains "$OUT_DIR/remote.err" 'invalid value for --openai: remote'
contains "$OUT_DIR/port.err" 'invalid value for --openai-port: 0'
contains "$OUT_DIR/duplicate.err" 'duplicate flag: --openai'

HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" \
    "$YVEX_BIN" serve --logs off --openai off --workers 2 --max-engines 2 \
    >"$OUT_DIR/host.out" 2>"$OUT_DIR/host.err" &
server_pid=$!

ready=0
attempt=0
while test "$attempt" -lt 100; do
    if run_client host status --json >"$OUT_DIR/status.json" 2>"$OUT_DIR/status.err"; then
        ready=1
        break
    fi
    kill -0 "$server_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.02
done
test "$ready" -eq 1 || fail 'persistent host did not become ready'
contains "$OUT_DIR/host.out" 'YVEX HOST · verified inference runtime'
contains "$OUT_DIR/host.out" 'protocol 20'
contains "$OUT_DIR/host.out" '0/2 engines · 2 workers'
contains "$OUT_DIR/host.out" 'events lifecycle · progress · resources'
contains "$OUT_DIR/host.out" 'host ready · Ctrl-C to stop'
contains "$OUT_DIR/status.json" '"schema":"yvex.host.status.v1"'
contains "$OUT_DIR/status.json" '"protocol":20'
contains "$OUT_DIR/status.json" '"status":2'
contains "$OUT_DIR/status.json" '"host_ready":true'
contains "$OUT_DIR/status.json" '"engine_count":0'
contains "$OUT_DIR/status.json" '"loaded_engine_count":0'
contains "$OUT_DIR/status.json" '"maximum_engines":2'
contains "$OUT_DIR/status.json" '"workers":2'
contains "$OUT_DIR/status.json" '"model_open_count":0'
contains "$OUT_DIR/status.json" '"openai_enabled":false'

run_client host memory >"$OUT_DIR/memory.out"
run_client host memory --json >"$OUT_DIR/memory.json"
contains "$OUT_DIR/memory.out" 'MEMORY'
contains "$OUT_DIR/memory.out" 'Explicit device'
contains "$OUT_DIR/memory.out" 'Physical pages'
contains "$OUT_DIR/memory.out" 'not reported'
contains "$OUT_DIR/memory.json" '"schema":"yvex.host.memory.v2"'
contains "$OUT_DIR/memory.json" '"resident_device_bytes":0'
contains "$OUT_DIR/memory.json" '"physical_residency_known":false'

run_client engine list --json >"$OUT_DIR/models-empty.json"
contains "$OUT_DIR/models-empty.json" '"schema":"yvex.engine.list.v1"'
contains "$OUT_DIR/models-empty.json" '"engines":[]'

set +e
run_client engine load absent >"$OUT_DIR/load-absent.out" 2>"$OUT_DIR/load-absent.err"
absent_status=$?
run_client engine load "$PROFILE" >"$OUT_DIR/load.out" 2>"$OUT_DIR/load.err"
load_status=$?
set -e
test "$absent_status" -eq 1
test "$load_status" -eq 1
contains "$OUT_DIR/load-absent.err" 'profile is not registered: absent'
contains "$OUT_DIR/load.err" 'deployment is not current (malformed-binding)'

run_client host status --json >"$OUT_DIR/status-after-failure.json"
contains "$OUT_DIR/status-after-failure.json" '"host_ready":true'
contains "$OUT_DIR/status-after-failure.json" '"engine_count":0'
contains "$OUT_DIR/status-after-failure.json" '"loaded_engine_count":0'
contains "$OUT_DIR/status-after-failure.json" '"model_open_count":0'
run_client engine list --json >"$OUT_DIR/models-failed.json"
contains "$OUT_DIR/models-failed.json" '"engines":[]'

set +e
run_client engine unload "$PROFILE" >"$OUT_DIR/unload.out" 2>"$OUT_DIR/unload.err"
unload_status=$?
set -e
test "$unload_status" -eq 1
contains "$OUT_DIR/unload.err" 'requested engine is not loaded'

# A second foreground invocation reports the existing host and exits.  It does
# not compete for listeners or open a second stdin-driven command surface.
set +e
HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" NO_COLOR=1 TERM=xterm-256color \
    script -q -e -c \
        "stty cols 132 rows 44; $YVEX_BIN serve" \
        "$OUT_DIR/attached.typescript" </dev/null \
        >"$OUT_DIR/attached.out" 2>"$OUT_DIR/attached.err"
duplicate_host_status=$?
set -e
test "$duplicate_host_status" -eq 1
contains "$OUT_DIR/attached.typescript" 'yvex: host already running'
contains "$OUT_DIR/attached.typescript" 'yvex host status'
not_contains "$OUT_DIR/attached.typescript" 'YVEX HOST · VERIFIED INFERENCE'
not_contains "$OUT_DIR/attached.typescript" 'Interactive host console'
not_contains "$OUT_DIR/attached.typescript" 'yvex[host] >'
not_contains "$OUT_DIR/attached.typescript" 'listener reservation failed'
kill -0 "$server_pid"
run_client host status --json >"$OUT_DIR/status-after-probe.json"
contains "$OUT_DIR/status-after-probe.json" '"host_ready":true'

# A plain logs command is a finite retained-history snapshot.  Continuous
# intent is explicit, format-independent, and exits cleanly on host shutdown.
run_client host logs >"$OUT_DIR/logs-snapshot.out"
run_client host logs --json >"$OUT_DIR/logs-snapshot.jsonl"
contains "$OUT_DIR/logs-snapshot.out" 'host logs · recent retained history'
not_contains "$OUT_DIR/logs-snapshot.out" 'HOST'
not_contains "$OUT_DIR/logs-snapshot.out" 'ENDPOINTS'
contains "$OUT_DIR/logs-snapshot.jsonl" '"kind":"runtime.ready"'
contains "$OUT_DIR/logs-snapshot.jsonl" '"kind":"engine.load.failed"'
run_client host logs --json --follow >"$OUT_DIR/logs-follow.jsonl" &
logs_pid=$!
ready=0
attempt=0
while test "$attempt" -lt 100; do
    if grep -F '"kind":"runtime.ready"' "$OUT_DIR/logs-follow.jsonl" >/dev/null 2>&1; then
        ready=1
        break
    fi
    kill -0 "$logs_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.02
done
test "$ready" -eq 1 || fail 'continuous host log subscriber did not become ready'

run_client host stop >"$OUT_DIR/stop.out" 2>"$OUT_DIR/stop.err"
wait "$server_pid"
server_pid=
wait "$logs_pid"
logs_pid=
contains "$OUT_DIR/logs-follow.jsonl" '"kind":"runtime.shutdown.complete"'
test ! -e "$SOCKET_PATH"

# A foreground TTY owns only server logs.  Lifecycle control remains the same
# deterministic command plane from another terminal.
HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" NO_COLOR=1 TERM=xterm-256color \
    script -q -f -e -c \
        "stty cols 132 rows 44; $YVEX_BIN serve --openai off --workers 2 --max-engines 2" \
        "$OUT_DIR/server-terminal.typescript" </dev/null \
        >"$OUT_DIR/server-terminal.out" 2>"$OUT_DIR/server-terminal.err" &
server_pid=$!
ready=0
attempt=0
while test "$attempt" -lt 100; do
    if run_client host status --json >"$OUT_DIR/terminal-status.json" 2>/dev/null; then
        ready=1
        break
    fi
    kill -0 "$server_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.02
done
test "$ready" -eq 1 || fail 'terminal foreground host did not become ready'
set +e
run_client engine load "$PROFILE" >"$OUT_DIR/terminal-load.out" \
    2>"$OUT_DIR/terminal-load.err"
load_status=$?
set -e
test "$load_status" -eq 1
contains "$OUT_DIR/terminal-load.err" 'deployment is not current (malformed-binding)'
run_client host stop >/dev/null
wait "$server_pid"
server_pid=
contains "$OUT_DIR/server-terminal.typescript" 'YVEX HOST'
contains "$OUT_DIR/server-terminal.typescript" '▀██████████████████████▀'
contains "$OUT_DIR/server-terminal.typescript" 'Y V E X'
contains "$OUT_DIR/server-terminal.typescript" 'STATE      ● STARTING'
contains "$OUT_DIR/server-terminal.typescript" 'NATIVE'
not_contains "$OUT_DIR/server-terminal.typescript" 'LOAD   deepseek4-v4-flash-dspark · g1'
contains "$OUT_DIR/server-terminal.typescript" 'FAIL'
contains "$OUT_DIR/server-terminal.typescript" 'deepseek4-v4-flash-dspark-'
contains "$OUT_DIR/server-terminal.typescript" ' generation=0 backend=CPU'
contains "$OUT_DIR/server-terminal.typescript" 'EVENTS     lifecycle · progress · resources'
contains "$OUT_DIR/server-terminal.typescript" 'host ready · Ctrl-C to stop'
not_contains "$OUT_DIR/server-terminal.typescript" 'CONTROL'
not_contains "$OUT_DIR/server-terminal.typescript" 'OPERATE'
not_contains "$OUT_DIR/server-terminal.typescript" 'type help'
not_contains "$OUT_DIR/server-terminal.typescript" 'Interactive host console'
not_contains "$OUT_DIR/server-terminal.typescript" 'yvex[host] >'
not_contains "$OUT_DIR/server-terminal.typescript" 'yvex[multi-engine] >'
test ! -e "$SOCKET_PATH"

# Narrow terminals use a smaller solid reduction of the same canonical mark.
HOME="$HOME_ROOT" XDG_RUNTIME_DIR="$SOCKET_ROOT" NO_COLOR=1 TERM=xterm-256color \
    script -q -f -e -c \
        "stty cols 80 rows 30; $YVEX_BIN serve --openai off" \
        "$OUT_DIR/server-compact.typescript" </dev/null \
        >"$OUT_DIR/server-compact.out" 2>"$OUT_DIR/server-compact.err" &
server_pid=$!
ready=0
attempt=0
while test "$attempt" -lt 100; do
    if run_client host status --json >"$OUT_DIR/compact-status.json" 2>/dev/null; then
        ready=1
        break
    fi
    kill -0 "$server_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.02
done
test "$ready" -eq 1 || fail 'compact terminal host did not become ready'
run_client host stop >/dev/null
wait "$server_pid"
server_pid=
contains "$OUT_DIR/server-compact.typescript" '▀████████████████▀'
contains "$OUT_DIR/server-compact.typescript" '▄██▀  ██  ▀██▄'
contains "$OUT_DIR/server-compact.typescript" 'Y V E X'
not_contains "$OUT_DIR/server-compact.typescript" 'Interactive host console'
test ! -e "$SOCKET_PATH"

printf 'cli persistent server lifecycle: ok\n'
