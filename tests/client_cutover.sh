#!/bin/sh
# Purpose: prove one yvex dispatch surface, engineering-route absorption, and runtime-lane isolation.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEX_CLIENT_LANE_OBJ=${YVEX_CLIENT_LANE_OBJ:-build/obj/src/cli/io/client.o}
. tests/support/cleanup.sh

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-client-cutover.XXXXXX")
config_root=$root/config-root
mkdir -m 700 "$config_root"
cleanup()
{
    status=$?
    trap - EXIT HUP INT TERM
    if test "$status" -ne 0; then
        test ! -f "$root/err" || cat "$root/err" >&2
        test ! -f "$root/out" || cat "$root/out" >&2
    fi
    yvex_test_cleanup "$root" || test "$status" -ne 0
    exit "$status"
}
trap cleanup EXIT HUP INT TERM

"$YVEX_BIN" help >"$root/help"
for expected in \
    'yvex chat' \
    'yvex runtime start|stop|status|watch|trace' \
    'yvex graph' \
    'yvex artifact show|verify|metadata' \
    'yvex quant preset|plan|emit' \
    'yvex tokenizer show|encode|decode|prompt' \
    'yvex evidence target|model|moe|backend|cuda'
do
    grep -F "$expected" "$root/help" >/dev/null
done
! grep -F 'yvex dev' "$root/help" >/dev/null
! grep -F 'yvex-dev' "$root/help" >/dev/null
! grep -F 'yvex-openai' "$root/help" >/dev/null

# Every former engineering executable route is reached by the sole yvex dispatcher.
for arguments in \
    'artifact show --help' \
    'artifact verify --help' \
    'artifact metadata --help' \
    'artifact tensors --help' \
    'artifact materialize --help' \
    'artifact materialize-gate --help' \
    'artifact model-gate --help' \
    'artifact template --help' \
    'artifact emit --help' \
    'graph --help' \
    'quant plan --help' \
    'quant emit --help' \
    'quant summarize --help' \
    'quant explain --help' \
    'quant policy --help' \
    'quant imatrix --help' \
    'quant job --help' \
    'quant qtype --help' \
    'quant convert --help' \
    'runtime input --help' \
    'runtime context --help' \
    'tokenizer show --help' \
    'tokenizer encode --help' \
    'tokenizer decode --help' \
    'tokenizer prompt --help' \
    'source manifest --help' \
    'source native --help' \
    'tensor map --help' \
    'tensor collection --help' \
    'evidence target --help' \
    'evidence model --help' \
    'evidence moe --help' \
    'evidence backend --help' \
    'evidence cuda --help' \
    'evidence accounts --help' \
    'evidence paths --help' \
    'evidence models --help'
do
    # shellcheck disable=SC2086
    "$YVEX_BIN" $arguments >"$root/out" 2>"$root/err"
done
"$YVEX_BIN" quant preset list >"$root/out" 2>"$root/err"

# The retired flat spellings remain unknown; no forwarding alias executes.
for command in materialize quant-policy tokenizer metadata tensor-map model-target fullmodel; do
    set +e
    "$YVEX_BIN" "$command" >"$root/out" 2>"$root/err"
    status=$?
    set -e
    test "$status" -eq 2
    grep -F "unknown command: $command" "$root/err" >/dev/null
done

# Artifact routes execute artifact owners, not the removed runtime-status facade.
set +e
"$YVEX_BIN" artifact show "$root/missing.gguf" >"$root/out" 2>"$root/err"
show_status=$?
"$YVEX_BIN" artifact verify "$root/missing.gguf" >"$root/out2" 2>"$root/err2"
verify_status=$?
set -e
test "$show_status" -ne 0
test "$verify_status" -ne 0
grep -F 'failed to open' "$root/err" >/dev/null
grep -F 'failed to open' "$root/out2" >/dev/null
! grep -F 'runtime socket' "$root/err" "$root/err2" >/dev/null

set +e
printf 'hello\n' | "$YVEX_BIN" chat >"$root/out" 2>"$root/err"
status=$?
set -e
test "$status" -eq 2
grep -F 'chat requires a terminal' "$root/err" >/dev/null

for arguments in \
    'runtime status --bogus' \
    'runtime watch --bogus' \
    'runtime trace --bogus' \
    'runtime stop --bogus' \
    'session list --bogus' \
    'session show main --json' \
    'model show --bogus'
do
    set +e
    # shellcheck disable=SC2086
    "$YVEX_BIN" $arguments >"$root/out" 2>"$root/err"
    status=$?
    set -e
    test "$status" -eq 2
done

XDG_CONFIG_HOME="$config_root" "$YVEX_BIN" model use current \
    --artifact /models/current.gguf --runtime-binding /bindings/current.binding \
    --target deepseek4-v4-flash --backend cuda --context 4096 >"$root/out"
grep -F 'selected model: current' "$root/out" >/dev/null
test "$(stat -c '%a' "$config_root/yvex/model.conf")" = 600
XDG_CONFIG_HOME="$config_root" "$YVEX_BIN" model show >"$root/out"
grep -F 'backend=cuda context=4096' "$root/out" >/dev/null

# The whole ELF intentionally links offline engine owners; the runtime-client object does not.
test -f "$YVEX_CLIENT_LANE_OBJ"
! nm -u "$YVEX_CLIENT_LANE_OBJ" | grep -E \
    'yvex_(runtime_model_open|artifact_materialize|runtime_transformer|runtime_generation_operator_execute|backend_cuda)' \
    >/dev/null
printf 'test: client_cutover\n'
