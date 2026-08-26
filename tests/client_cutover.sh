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
printf '{"schema":"yvex.models.local.v5","models":[]}\n' >"$registry"
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
for expected in \
    'yvex chat' \
    'yvex run' \
    'yvex server                                     Run the persistent multi-engine host' \
    'yvex server load MODEL' \
    'yvex server models' \
    'yvex server status' \
    'yvex session list' \
    'yvex compile source manifest' \
    'yvex compile quant plan' \
    'yvex artifact verify' \
    'yvex completion'
do
    grep -F "$expected" "$root/help" >/dev/null
done
for retired in 'yvex dev' 'yvex-dev' 'yvex-openai' 'yvex evidence' \
    'yvex graph' 'yvex quant' 'yvex source' 'yvex tensor' 'yvex tokenizer'; do
    ! grep -F "$retired" "$root/help" >/dev/null
done

"$YVEX_BIN" help --advanced >"$root/advanced"
for expected in \
    'yvex inspect artifact metadata' \
    'yvex inspect source' \
    'yvex execute tokenizer encode' \
    'yvex execute attention run' \
    'yvex profile attention run' \
    'yvex system cuda'
do
    grep -F "$expected" "$root/advanced" >/dev/null
done

"$YVEX_BIN" help server log >"$root/leaf-help"
grep -F 'operation: server.log' "$root/leaf-help" >/dev/null
"$YVEX_BIN" help --json >"$root/discovery.json"
python3 - "$root/discovery.json" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload['schema'] == 'yvex.command.discovery.v1'
assert len(payload['registry_identity']) == 64
paths = {row['command_path'] for row in payload['operations'] if row['projections']['cli']}
assert 'server status' in paths
assert 'server' in paths
assert 'server load' in paths
assert 'server unload' in paths
assert 'server models' in paths
assert 'runtime status' not in paths
assert 'execute tokenizer encode' in paths
assert not any(path.split(' ', 1)[0] in {
    'evidence', 'graph', 'quant', 'source', 'tensor', 'tokenizer', 'eval', 'bench'
} for path in paths)
PY
"$YVEX_BIN" completion bash >"$root/yvex.bash"
bash -n "$root/yvex.bash"
grep -F 'server' "$root/yvex.bash" >/dev/null
grep -F -- '--ctx' "$root/yvex.bash" >/dev/null
grep -F -- '--backend' "$root/yvex.bash" >/dev/null

# Every absorbed engineering capability has one registry-selected canonical route.
for arguments in \
    'artifact show --help' \
    'artifact verify --help' \
    'artifact materialize --help' \
    'inspect artifact metadata --help' \
    'inspect artifact tensors --help' \
    'execute artifact materialize-gate --help' \
    'execute artifact model-gate check --help' \
    'compile emit template --help' \
    'compile emit artifact --help' \
    'inspect attention describe --help' \
    'execute attention run --help' \
    'profile attention run --help' \
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
    'execute input --help' \
    'inspect context --help' \
    'inspect tokenizer --help' \
    'execute tokenizer encode --help' \
    'execute tokenizer decode --help' \
    'execute tokenizer prompt --help' \
    'compile source manifest --help' \
    'inspect source --help' \
    'compile map --help' \
    'inspect tensor collection --help' \
    'inspect target --help' \
    'inspect model full --help' \
    'inspect moe --help' \
    'inspect backend --help' \
    'system cuda --help' \
    'system accounts --help' \
    'system paths --help' \
    'model list --help'
do
    # shellcheck disable=SC2086
    "$YVEX_BIN" $arguments >"$root/out" 2>"$root/err"
done

# Removed namespaces refuse and provide migration direction without executing aliases.
for command in evidence graph quant source tensor tokenizer; do
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
printf 'hello\n' | "$YVEX_BIN" chat >"$root/out" 2>"$root/err"
status=$?
set -e
test "$status" -eq 2
grep -F 'chat requires a terminal' "$root/err" >/dev/null

for arguments in \
    'server status --bogus' \
    'server log --bogus' \
    'server stop --bogus' \
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
"$YVEX_BIN" server -h >"$root/out" 2>"$root/err"
grep -F 'operation: server.host' "$root/out" >/dev/null
for arguments in \
    'server --workers' \
    'server --workers 0' \
    'server current extra' \
    'server current --context 4096' \
    'compile artifact prepare --out artifact.gguf --out-dir artifacts'
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
"$YVEX_BIN" server statu >"$root/out" 2>"$root/err"
status=$?
set -e
test "$status" -eq 2
grep -F 'usage: yvex server [options]' "$root/err" >/dev/null

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
  "schema": "yvex.models.local.v5",
  "models": [{
    "alias": "current-model-runtime-profile",
    "family": "deepseek4",
    "path": "$artifact",
    "runtime_binding": "$binding",
    "runtime_target": "deepseek4-v4-flash-dspark",
    "runtime_backend": "cuda",
    "runtime_mode": "dspark",
    "runtime_context": 4096
  }]
}
EOF
HOME="$home_root" "$YVEX_BIN" model list >"$root/out"
grep -F 'current-model-runtime-profile' "$root/out" >/dev/null
grep -F 'cuda' "$root/out" >/dev/null
grep -F 'package-ready' "$root/out" >/dev/null
test ! -e "$home_root/.config/yvex/model.conf"

set +e
HOME="$home_root" XDG_RUNTIME_DIR="$root/absent-runtime" \
    "$YVEX_BIN" server load current-model-runtime-profile \
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
