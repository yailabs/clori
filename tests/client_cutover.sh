#!/bin/sh
# Verifies registry dispatch, canonical command placement, and runtime-lane isolation.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEX_CLIENT_LANE_OBJ=${YVEX_CLIENT_LANE_OBJ:-build/obj/src/cli/io/client.o}
. tests/support/cleanup.sh

root=$(mktemp -d "${TMPDIR:-/tmp}/yvex-command-architecture.XXXXXX")
home_root=$root/home
mkdir -m 700 "$home_root" "$home_root/.config"
mkdir -m 775 "$home_root/.config/yvex"
registry="$home_root/.local/share/yvex/models.local.json"
mkdir -p "$home_root/.local/share/yvex"
printf '{"schema":"yvex.models.local.v6","models":[]}\n' >"$registry"
export HOME="$home_root"
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
grep -F 'YVEX inference/compiler/runtime' "$root/help" >/dev/null
for expected in \
    'USE' 'HOST' 'BUILD' 'INSPECT' 'META' \
    'chat' 'serve' 'host' 'engine' 'session' \
    'model' 'source' 'artifact' 'profile' 'compile' 'inspect' 'bench'
do
    grep -F "$expected" "$root/help" >/dev/null
done
for retired in 'yvex run' 'yvex server' 'yvex-dev' 'yvex-openai'; do
    ! grep -F "$retired" "$root/help" >/dev/null
done

"$YVEX_BIN" help --advanced >"$root/advanced"
for expected in \
    'yvex inspect artifact metadata' \
    'yvex inspect source tensors' \
    'yvex inspect tokenizer encode' \
    'yvex bench attention execute' \
    'yvex bench attention profile' \
    'yvex inspect cuda'
do
    grep -F "$expected" "$root/advanced" >/dev/null
done

"$YVEX_BIN" help host logs >"$root/leaf-help"
grep -F 'operation: host.logs' "$root/leaf-help" >/dev/null
"$YVEX_BIN" help --json >"$root/discovery.json"
python3 - "$root/discovery.json" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload['schema'] == 'yvex.command.discovery.v1'
assert len(payload['registry_identity']) == 64
paths = {row['command_path'] for row in payload['operations'] if row['projections']['cli']}
assert '' not in paths
assert 'chat' in paths
assert 'host status' in paths
assert 'serve' in paths
assert 'engine load' in paths
assert 'engine unload' in paths
assert 'engine list' in paths
assert 'runtime status' not in paths
assert 'inspect tokenizer encode' in paths
assert 'model list' in paths and 'source list' in paths
assert 'artifact list' in paths and 'profile list' in paths
assert not any(path.split(' ', 1)[0] in {'run', 'server'} for path in paths)
assert not any(path.split(' ', 1)[0] in {
    'dev', 'evidence', 'execute', 'graph', 'integrate', 'quant', 'system',
    'tensor', 'tokenizer', 'eval'
} for path in paths)
PY
"$YVEX_BIN" help completion bash >"$root/yvex.bash"
bash -n "$root/yvex.bash"
grep -F 'serve' "$root/yvex.bash" >/dev/null
grep -F 'host status' "$root/yvex.bash" >/dev/null
! grep -F 'server status' "$root/yvex.bash" >/dev/null
grep -F -- '--ctx' "$root/yvex.bash" >/dev/null
grep -F -- '--backend' "$root/yvex.bash" >/dev/null

# Every absorbed engineering capability has one registry-selected canonical route.
for arguments in \
    'artifact show --help' \
    'artifact verify --help' \
    'artifact materialize --help' \
    'inspect artifact metadata --help' \
    'inspect artifact tensors --help' \
    'artifact verify materialization --help' \
    'artifact verify model --help' \
    'compile artifact template --help' \
    'compile artifact emit --help' \
    'inspect attention describe --help' \
    'bench attention execute --help' \
    'bench attention profile --help' \
    'compile quant plan --help' \
    'compile quant emit --help' \
    'compile quant probe --help' \
    'inspect quant summary --help' \
    'inspect quant decision --help' \
    'compile quant policy --help' \
    'compile quant imatrix --help' \
    'compile quant job --help' \
    'inspect qtype --help' \
    'compile quant convert --help' \
    'inspect input --help' \
    'inspect context --help' \
    'inspect tokenizer --help' \
    'inspect tokenizer encode --help' \
    'inspect tokenizer decode --help' \
    'inspect tokenizer prompt --help' \
    'compile source manifest --help' \
    'inspect source tensors --help' \
    'compile tensor map --help' \
    'inspect tensor collection --help' \
    'inspect target --help' \
    'inspect model full --help' \
    'inspect moe --help' \
    'inspect backend --help' \
    'inspect backend cuda --help' \
    'inspect cuda --help' \
    'source accounts --help' \
    'inspect paths --help' \
    'model list --help' \
    'source list --help' \
    'artifact list --help' \
    'profile list --help'
do
    # shellcheck disable=SC2086
    "$YVEX_BIN" $arguments >"$root/out" 2>"$root/err"
done

# Removed namespaces and one-shot/server grammar refuse without forwarding.
for command in run server evidence execute graph quant system tensor tokenizer; do
    set +e
    "$YVEX_BIN" "$command" >"$root/out" 2>"$root/err"
    status=$?
    set -e
    test "$status" -eq 2
    grep -F "removed command: $command" "$root/err" >/dev/null
    grep -F 'hint:' "$root/err" >/dev/null
done
for arguments in 'runtime input' 'runtime context' 'runtime start' \
    'runtime status' 'runtime model' 'runtime memory' 'runtime watch' \
    'runtime trace' 'runtime stop' 'model select' 'model selected'; do
    set +e
    # shellcheck disable=SC2086
    "$YVEX_BIN" $arguments >"$root/out" 2>"$root/err"
    status=$?
    set -e
    test "$status" -eq 2
    grep -F 'removed command:' "$root/err" >/dev/null
done

# Retired flat spellings never forward to an offline adapter.
for command in materialize quant-policy metadata tensor-map model-target fullmodel; do
    set +e
    "$YVEX_BIN" "$command" >"$root/out" 2>"$root/err"
    status=$?
    set -e
    test "$status" -eq 2
    grep -F "unknown command: $command" "$root/err" >/dev/null
done

# Artifact routes execute artifact owners, not one removed runtime-status facade.
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
printf 'hello\n' | "$YVEX_BIN" >"$root/out" 2>"$root/err"
status=$?
set -e
test "$status" -eq 0
grep -F 'YVEX inference/compiler/runtime' "$root/out" >/dev/null
set +e
printf 'hello\n' | "$YVEX_BIN" chat >"$root/out" 2>"$root/err"
status=$?
set -e
test "$status" -eq 2
grep -F 'chat requires a terminal' "$root/err" >/dev/null

for arguments in \
    'host status --bogus' \
    'host logs --bogus' \
    'host stop --bogus' \
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

# One registry-driven parser owns help bypass, types, ranges, duplicates, and relations.
"$YVEX_BIN" serve -h >"$root/out" 2>"$root/err"
grep -F 'operation: host.serve' "$root/out" >/dev/null
for arguments in \
    'serve --workers' \
    'serve --workers 0' \
    'serve current extra' \
    'serve current --context 4096' \
    'compile --out artifact.gguf --out-dir artifacts'
do
    set +e
    # shellcheck disable=SC2086
    "$YVEX_BIN" $arguments >"$root/out" 2>"$root/err"
    status=$?
    set -e
    test "$status" -eq 2
    grep -F 'usage: yvex' "$root/err" >/dev/null
done
set +e
"$YVEX_BIN" serve statu >"$root/out" 2>"$root/err"
status=$?
set -e
test "$status" -eq 2
grep -F 'usage: yvex serve [options]' "$root/err" >/dev/null

# Registry discovery remains distinct from the model hosted by a running server. No command writes
# an implicit startup selection.
artifact="$root/current.gguf"
binding="$root/current.binding"
printf 'artifact fixture\n' >"$artifact"
printf 'binding fixture\n' >"$binding"
artifact=$(realpath "$artifact")
binding=$(realpath "$binding")
cat >"$registry" <<EOF
{
  "schema": "yvex.models.local.v6",
  "models": [{
    "alias": "current-model-runtime-profile",
    "family": "deepseek4",
    "path": "$artifact",
    "runtime_binding": "$binding",
    "runtime_target": "deepseek4-v4-flash-dspark",
    "runtime_backend": "cuda",
    "runtime_engine_kind": "text",
    "runtime_execution_strategy": "speculative",
    "runtime_context": 4096
  }]
}
EOF
HOME="$home_root" "$YVEX_BIN" model list >"$root/out"
grep -F 'v4-flash-dspark' "$root/out" >/dev/null
grep -F '1 artifact · 1 profile (1 runnable)' "$root/out" >/dev/null
HOME="$home_root" "$YVEX_BIN" profile list >"$root/profiles"
grep -F 'current-model-runtime-profile' "$root/profiles" >/dev/null
grep -F 'cuda/text/speculative' "$root/profiles" >/dev/null
test ! -e "$home_root/.config/yvex/model.conf"

set +e
HOME="$home_root" XDG_RUNTIME_DIR="$root/absent-runtime" \
    "$YVEX_BIN" engine load current-model-runtime-profile \
    >"$root/out2" 2>"$root/err"
status=$?
set -e
test "$status" -eq 1
grep -F 'local runtime socket is absent' "$root/err" >/dev/null
! grep -F '/models/current.gguf' "$root/out2" "$root/err" >/dev/null

# The whole ELF links finite offline owners; the runtime-client object has no engine edge.
test -f "$YVEX_CLIENT_LANE_OBJ"
! nm -u "$YVEX_CLIENT_LANE_OBJ" | grep -E \
    'yvex_(runtime_model_open|artifact_materialize|runtime_transformer|runtime_generation_operator_execute|backend_cuda)' \
    >/dev/null
! nm -u "$YVEX_CLIENT_LANE_OBJ" | grep -E 'execv|execvp' >/dev/null
printf 'test: client_command_architecture\n'
