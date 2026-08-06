#!/bin/sh
# MiniMax-H3 Visual VAE operator discovery and fail-closed reduced geometry.

set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli/minimax-video}

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

run_code help 0 "$YVEX_BIN" execute component video-vae --help
contains "$OUT_DIR/help.out" "operation: execute.graph.component.video-vae"
contains "$OUT_DIR/help.out" "--latent-frames"
contains "$OUT_DIR/help.out" "--latent-height"
contains "$OUT_DIR/help.out" "--latent-width"

run_code missing 2 "$YVEX_BIN" execute component video-vae
contains "$OUT_DIR/missing.err" \
    "requires target, artifact, backend, input file, latent geometry, and output path"

run_code wrong_geometry 2 "$YVEX_BIN" execute component video-vae \
    --target minimax-h3-fl2va --artifact /tmp/missing.gguf --backend cpu \
    --input-file /tmp/missing.f32 --latent-frames 1 --latent-height 1 --latent-width 2 \
    --out "$OUT_DIR/wrong-geometry.f32"
contains "$OUT_DIR/wrong_geometry.err" "exact reduced geometry [1,24,1,1,1]"

run_code wrong_target 2 "$YVEX_BIN" execute component video-vae \
    --target wrong --artifact /tmp/missing.gguf --backend cpu \
    --input-file /tmp/missing.f32 --latent-frames 1 --latent-height 1 --latent-width 1 \
    --out "$OUT_DIR/wrong-target.f32"
contains "$OUT_DIR/wrong_target.err" "video-vae component requires minimax-h3-fl2va"

run_code wrong_backend 2 "$YVEX_BIN" execute component video-vae \
    --target minimax-h3-fl2va --artifact /tmp/missing.gguf --backend cuda \
    --input-file /tmp/missing.f32 --latent-frames 1 --latent-height 1 --latent-width 1 \
    --out "$OUT_DIR/wrong-backend.f32"
contains "$OUT_DIR/wrong_backend.err" "Visual VAE currently admits only backend cpu"

run_code unsafe_input 3 "$YVEX_BIN" execute component video-vae \
    --target minimax-h3-fl2va --artifact /tmp/missing.gguf --backend cpu \
    --input-file /dev/null --latent-frames 1 --latent-height 1 --latent-width 1 \
    --out "$OUT_DIR/unsafe-input.f32"
contains "$OUT_DIR/unsafe_input.err" "file could not be opened safely"
test ! -e "$OUT_DIR/unsafe-input.f32" || fail "refused execution published output"

printf 'cli minimax video: ok\n'
