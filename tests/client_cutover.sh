#!/bin/sh
# Purpose: prove the product grammar, retired-command refusal, non-TTY chat refusal,
# and thin-client symbol boundary without requiring a model host.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEX_DEV_BIN=${YVEX_DEV_BIN:-./yvex-dev}
. tests/support/cleanup.sh

"$YVEX_BIN" help > /tmp/yvex-client-help.$$
config_root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-client-config.XXXXXX")
mkdir -m 700 "$config_root/config"
cleanup()
{
    rm -f /tmp/yvex-client-help.$$ /tmp/yvex-client-out.$$ /tmp/yvex-client-err.$$
    yvex_test_cleanup "$config_root"
}
trap cleanup EXIT HUP INT TERM
grep -F 'yvex chat' /tmp/yvex-client-help.$$ >/dev/null
grep -F 'yvex runtime start|stop|status|watch|trace' /tmp/yvex-client-help.$$ >/dev/null
! grep -F 'yvex graph' /tmp/yvex-client-help.$$ >/dev/null

for command in graph materialize quant-policy tokenizer metadata tensor-map model-target fullmodel; do
    set +e
    "$YVEX_BIN" "$command" > /tmp/yvex-client-out.$$ 2> /tmp/yvex-client-err.$$
    status=$?
    set -e
    test "$status" -eq 2
    grep -F "unknown command: $command" /tmp/yvex-client-err.$$ >/dev/null
done

set +e
printf 'hello\n' | "$YVEX_BIN" chat > /tmp/yvex-client-out.$$ 2> /tmp/yvex-client-err.$$
status=$?
set -e
test "$status" -eq 2
grep -F 'chat requires a terminal' /tmp/yvex-client-err.$$ >/dev/null

for arguments in \
    'runtime status --bogus' \
    'runtime watch --bogus' \
    'runtime trace --bogus' \
    'runtime stop --bogus' \
    'session list --bogus' \
    'session show main --json' \
    'model show --bogus' \
    'artifact verify --bogus'; do
    set +e
    # The product grammar refuses trailing options instead of silently changing semantics.
    # shellcheck disable=SC2086
    "$YVEX_BIN" $arguments > /tmp/yvex-client-out.$$ 2> /tmp/yvex-client-err.$$
    status=$?
    set -e
    test "$status" -eq 2
done

XDG_CONFIG_HOME="$config_root/config" "$YVEX_BIN" model use current \
    --artifact /models/current.gguf --runtime-binding /bindings/current.binding \
    --target deepseek4-v4-flash --backend cuda --context 4096 \
    > /tmp/yvex-client-out.$$
grep -F 'selected model: current' /tmp/yvex-client-out.$$ >/dev/null
test "$(stat -c '%a' "$config_root/config/yvex/model.conf")" = 600
XDG_CONFIG_HOME="$config_root/config" "$YVEX_BIN" model show > /tmp/yvex-client-out.$$
grep -F 'current' /tmp/yvex-client-out.$$ >/dev/null
grep -F 'backend=cuda context=4096' /tmp/yvex-client-out.$$ >/dev/null
grep -F '/models/current.gguf' /tmp/yvex-client-out.$$ >/dev/null

"$YVEX_DEV_BIN" help > /tmp/yvex-client-out.$$
grep -F 'yvex-dev graph' /tmp/yvex-client-out.$$ >/dev/null
grep -F 'yvex-dev quant preset|plan|emit' /tmp/yvex-client-out.$$ >/dev/null

! nm "$YVEX_BIN" | grep -E 'yvex_runtime_model_open|yvex_artifact_materialize|yvex_runtime_generation_operator_execute|yvex_backend_cuda_graph_execute' >/dev/null
printf 'test: client_cutover\n'
