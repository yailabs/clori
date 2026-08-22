#!/bin/sh
# Verifies the direct foreground server grammar and fail-closed model admission.
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli-server}
HOME_ROOT=$OUT_DIR/home
SOCKET_PATH=${TMPDIR:-/tmp}/yvex-cli-server-$$.sock

yvex_test_cleanup "$OUT_DIR" "$SOCKET_PATH"
mkdir -p "$OUT_DIR" "$HOME_ROOT/.local/share/yvex"

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
    "alias": "current-model-runtime-profile",
    "family": "deepseek4",
    "path": "$artifact",
    "runtime_binding": "$binding",
    "runtime_target": "deepseek4-v4-flash-dspark",
    "runtime_backend": "cpu",
    "runtime_mode": "target-only",
    "runtime_context": 4096
  }, {
    "alias": "minimax-h3-fl2va-runtime-media",
    "family": "minimax-h3",
    "path": "$artifact"
  }]
}
EOF

MEDIA_ROOT=$OUT_DIR/media-root
MEDIA_OUTPUT=$OUT_DIR/media-output
MEDIA_RUNTIME=$OUT_DIR/media-runtime
mkdir -p "$MEDIA_ROOT" "$MEDIA_OUTPUT" "$MEDIA_RUNTIME"
chmod 700 "$MEDIA_RUNTIME"
MEDIA_ROOT=$(realpath "$MEDIA_ROOT")
MEDIA_OUTPUT=$(realpath "$MEDIA_OUTPUT")
MEDIA_RUNTIME=$(realpath "$MEDIA_RUNTIME")
MEDIA_SOCKET=$MEDIA_RUNTIME/yvexd.sock

"$YVEX_BIN" server --help >"$OUT_DIR/help.out" 2>"$OUT_DIR/help.err"
contains "$OUT_DIR/help.out" 'usage: yvex server MODEL [options]'
contains "$OUT_DIR/help.out" 'Run one model server in the foreground.'
contains "$OUT_DIR/help.out" '--ctx'
! grep -F -- '--context' "$OUT_DIR/help.out" >/dev/null
contains "$OUT_DIR/help.out" '--openai'
contains "$OUT_DIR/help.out" '--generation-mode        target-only|dspark|media'
contains "$OUT_DIR/help.out" '--media-artifact-root'
contains "$OUT_DIR/help.out" '--output-root'

set +e
HOME="$HOME_ROOT" "$YVEX_BIN" server >"$OUT_DIR/missing.out" 2>"$OUT_DIR/missing.err"
missing_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server absent >"$OUT_DIR/absent.out" 2>"$OUT_DIR/absent.err"
absent_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server current-model-runtime-profile --openai remote \
    >"$OUT_DIR/remote.out" 2>"$OUT_DIR/remote.err"
remote_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server current-model-runtime-profile --openai-port 0 \
    >"$OUT_DIR/port.out" 2>"$OUT_DIR/port.err"
port_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server current-model-runtime-profile --openai on --openai off \
    >"$OUT_DIR/duplicate.out" 2>"$OUT_DIR/duplicate.err"
duplicate_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server current-model-runtime-profile --generation-mode invalid \
    >"$OUT_DIR/mode.out" 2>"$OUT_DIR/mode.err"
mode_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server current-model-runtime-profile --context 8192 \
    >"$OUT_DIR/context.out" 2>"$OUT_DIR/context.err"
context_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server minimax-h3-fl2va-runtime-media --generation-mode media \
    >"$OUT_DIR/media-missing.out" 2>"$OUT_DIR/media-missing.err"
media_missing_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server minimax-h3-fl2va-runtime-media --generation-mode media \
    --media-artifact-root "$MEDIA_ROOT" --output-root "$MEDIA_OUTPUT" --backend cpu \
    >"$OUT_DIR/media-cpu.out" 2>"$OUT_DIR/media-cpu.err"
media_cpu_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server minimax-h3-fl2va-runtime-media --generation-mode media \
    --media-artifact-root "$MEDIA_ROOT" --output-root "$MEDIA_OUTPUT" --openai on \
    >"$OUT_DIR/media-openai.out" 2>"$OUT_DIR/media-openai.err"
media_openai_status=$?
HOME="$HOME_ROOT" "$YVEX_BIN" server current-model-runtime-profile \
    --ctx 8192 --socket "$SOCKET_PATH" --openai off \
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
test "$media_missing_status" -eq 2
test "$media_cpu_status" -eq 2
test "$media_openai_status" -eq 2
test "$admission_status" -eq 1
contains "$OUT_DIR/missing.err" 'usage: yvex server MODEL [options]'
contains "$OUT_DIR/absent.err" 'model is not registered: absent'
contains "$OUT_DIR/remote.err" 'invalid value for --openai: remote'
contains "$OUT_DIR/port.err" 'invalid value for --openai-port: 0'
contains "$OUT_DIR/duplicate.err" 'duplicate flag: --openai'
contains "$OUT_DIR/mode.err" 'invalid value for --generation-mode: invalid'
contains "$OUT_DIR/context.err" 'unknown flag: --context'
contains "$OUT_DIR/media-missing.err" 'media mode requires --media-artifact-root and --output-root'
contains "$OUT_DIR/media-cpu.err" 'media mode requires the admitted CUDA backend'
contains "$OUT_DIR/media-openai.err" 'media mode requires the admitted CUDA backend and OpenAI disabled'
contains "$OUT_DIR/admission.out" 'YVEX server · foreground'
contains "$OUT_DIR/admission.out" 'profile current-model-runtime-profile'
contains "$OUT_DIR/admission.out" 'backend=cpu · mode=target-only · requested ctx=8192'
contains "$OUT_DIR/admission.out" "artifact $artifact"
contains "$OUT_DIR/admission.out" "binding $binding"
contains "$OUT_DIR/admission.out" 'stop with Ctrl-C or `yvex server stop`'
contains "$OUT_DIR/admission.err" 'model admission in progress (elapsed 0 s)'
contains "$OUT_DIR/admission.err" 'model admission failed (elapsed '

set +e
HOME="$HOME_ROOT" timeout --signal=TERM 2 "$YVEX_BIN" server minimax-h3-fl2va-runtime-media \
    --media-artifact-root "$MEDIA_ROOT" \
    --output-root "$MEDIA_OUTPUT" --socket "$MEDIA_SOCKET" --openai off \
    >"$OUT_DIR/media-host.out" 2>"$OUT_DIR/media-host.err"
media_host_status=$?
set -e
test "$media_host_status" -eq 1
contains "$OUT_DIR/media-host.out" 'profile minimax-h3-fl2va-runtime-media'
contains "$OUT_DIR/media-host.out" 'mode=media'
contains "$OUT_DIR/media-host.out" "component root $MEDIA_ROOT"
contains "$OUT_DIR/media-host.out" "output root $MEDIA_OUTPUT"
contains "$OUT_DIR/media-host.err" 'model admission failed'
contains "$OUT_DIR/media-host.err" 'failed to open'
test ! -e "$MEDIA_SOCKET"
yvex_test_cleanup "$SOCKET_PATH"

printf 'cli server grammar: ok\n'
