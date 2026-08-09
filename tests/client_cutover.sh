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
    'yvex runtime status' \
    'yvex session list' \
    'yvex model selected' \
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

"$YVEX_BIN" help runtime trace >"$root/leaf-help"
grep -F 'operation: runtime.trace' "$root/leaf-help" >/dev/null
"$YVEX_BIN" help --json >"$root/discovery.json"
python3 - "$root/discovery.json" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload['schema'] == 'yvex.command.discovery.v1'
assert len(payload['registry_identity']) == 64
paths = {row['command_path'] for row in payload['operations'] if row['projections']['cli']}
assert 'runtime status' in paths
assert 'execute tokenizer encode' in paths
assert not any(path.split(' ', 1)[0] in {
    'evidence', 'graph', 'quant', 'source', 'tensor', 'tokenizer', 'eval', 'bench'
} for path in paths)
PY
"$YVEX_BIN" completion bash >"$root/yvex.bash"
bash -n "$root/yvex.bash"
grep -F 'runtime' "$root/yvex.bash" >/dev/null
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
for arguments in 'runtime input' 'runtime context' 'runtime trace --follow'; do
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

# One registry-driven parser owns help bypass, types, ranges, duplicates, and relations.
"$YVEX_BIN" model select -h >"$root/out" 2>"$root/err"
grep -F 'operation: model.select' "$root/out" >/dev/null
for arguments in \
    'model select' \
    'model select current extra' \
    'model select current --artifact /models/current.gguf' \
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
"$YVEX_BIN" runtime statu >"$root/out" 2>"$root/err"
status=$?
set -e
test "$status" -eq 2
grep -F 'did you mean `yvex runtime status`' "$root/err" >/dev/null

# Selected startup configuration resolves one complete registry profile and remains distinct from
# live daemon state.
artifact="$root/current.gguf"
binding="$root/current.binding"
registry="$home_root/.local/share/yvex/models.local.json"
mkdir -p "$home_root/.local/share/yvex"
printf 'artifact fixture\n' >"$artifact"
printf 'binding fixture\n' >"$binding"
artifact=$(realpath "$artifact")
binding=$(realpath "$binding")
cat >"$registry" <<EOF
{
  "schema": "yvex.models.local.v3",
  "models": [{
    "alias": "current-model-runtime-profile",
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
grep -F '4096' "$root/out" >/dev/null
grep -F 'yes' "$root/out" >/dev/null
HOME="$home_root" \
    "$YVEX_BIN" model select current-model-runtime-profile >"$root/out"
grep -F 'selected model: current-model-runtime-profile' "$root/out" >/dev/null
test "$(stat -c '%a' "$home_root/.config/yvex/model.conf")" = 600
test "$(stat -c '%a' "$home_root/.config/yvex")" = 700
HOME="$home_root" "$YVEX_BIN" model selected >"$root/out"
grep -F 'target deepseek4-v4-flash-dspark · backend=cuda · mode=dspark · context=4096' \
    "$root/out" >/dev/null
! grep -F 'artifact=' "$root/out" >/dev/null
! grep -F 'binding=' "$root/out" >/dev/null

# A flag-free start projects the selected profile into yvexd's startup vector. Keep this proof
# isolated from the resident daemon by placing the client beside a recording test double.
mkdir "$root/product-bin"
cp "$YVEX_BIN" "$root/product-bin/yvex"
cat >"$root/product-bin/yvexd" <<'EOF'
#!/bin/sh
printf '%s\n' "$@" >"$YVEX_TEST_DAEMON_ARGS"
EOF
chmod 700 "$root/product-bin/yvexd"
HOME="$home_root" YVEX_TEST_DAEMON_ARGS="$root/daemon-arguments" \
    "$root/product-bin/yvex" runtime start >"$root/start.out"
grep -F 'starting selected model current-model-runtime-profile' "$root/start.out" >/dev/null
grep -F 'target deepseek4-v4-flash-dspark · CUDA · DSpark · context 4096' \
    "$root/start.out" >/dev/null
grep -F 'foreground host · leave this terminal open · readiness follows model admission' \
    "$root/start.out" >/dev/null
cat >"$root/expected-daemon-arguments" <<EOF
--model
$artifact
--runtime-binding
$binding
--target
deepseek4-v4-flash-dspark
--backend
cuda
--generation-mode
dspark
--context
4096
EOF
cmp "$root/expected-daemon-arguments" "$root/daemon-arguments"

set +e
HOME="$home_root" XDG_RUNTIME_DIR="$root/absent-runtime" \
    "$YVEX_BIN" runtime model >"$root/out2" 2>"$root/err"
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
printf 'test: client_command_architecture\n'
