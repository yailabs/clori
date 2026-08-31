#!/bin/sh
# MiniMax-H3 Audio VAE operator discovery and fail-closed input admission.

set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli/minimax-audio}

mkdir -p "$OUT_DIR"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

run_code() {
    name=$1
    expected=$2
    shift 2
    set +e
    "$@" >"$OUT_DIR/$name.out" 2>"$OUT_DIR/$name.err"
    actual=$?
    set -e
    [ "$actual" -eq "$expected" ] || fail "$name exited $actual, expected $expected"
}

contains() {
    grep -F -- "$2" "$1" >/dev/null || fail "$1 missing: $2"
}

run_code help 0 "$YVEX_BIN" bench component audio-vae --help
contains "$OUT_DIR/help.out" "operation: execute.graph.component.audio-vae"
contains "$OUT_DIR/help.out" "--latent-steps"

run_code missing 2 "$YVEX_BIN" bench component audio-vae
contains "$OUT_DIR/missing.err" \
    "requires target, artifact, backend, input file, latent steps, and output path"

run_code wrong_target 5 "$YVEX_BIN" bench component audio-vae \
    --target wrong --artifact /tmp/missing.gguf --backend cpu \
    --input-file /tmp/missing.f32 --latent-steps 1 --out "$OUT_DIR/wrong-target.f32"
contains "$OUT_DIR/wrong_target.err" \
    "no admitted component execution binding matches target and component"

run_code wrong_backend 2 "$YVEX_BIN" bench component audio-vae \
    --target minimax-h3-fl2va --artifact /tmp/missing.gguf --backend vulkan \
    --input-file /tmp/missing.f32 --latent-steps 1 --out "$OUT_DIR/wrong-backend.f32"
contains "$OUT_DIR/wrong_backend.err" "unknown backend kind: vulkan"

run_code cuda_backend 5 "$YVEX_BIN" bench component audio-vae \
    --target minimax-h3-fl2va --artifact /tmp/missing.gguf --backend cuda \
    --input-file /tmp/missing.f32 --latent-steps 1 --out "$OUT_DIR/cuda-budget.f32"
contains "$OUT_DIR/cuda_backend.err" \
    "component execution binding does not admit the requested backend"

run_code unsafe_input 3 "$YVEX_BIN" bench component audio-vae \
    --target minimax-h3-fl2va --artifact /tmp/missing.gguf --backend cpu \
    --input-file /dev/null --latent-steps 1 --out "$OUT_DIR/unsafe-input.f32"
contains "$OUT_DIR/unsafe_input.err" "file could not be opened safely"
test ! -e "$OUT_DIR/unsafe-input.f32" || fail "refused execution published output"

printf 'cli minimax audio: ok\n'
