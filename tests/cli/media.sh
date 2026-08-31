#!/bin/sh
# Native synchronized media operator publication and refusal coverage.

set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli/media}

mkdir -p "$OUT_DIR"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains() {
    grep -F -- "$2" "$1" >/dev/null || fail "$1 missing: $2"
}

truncate -s 144 "$OUT_DIR/video.f32"
truncate -s 32040 "$OUT_DIR/audio.f32"
rm -f "$OUT_DIR/first.avi" "$OUT_DIR/second.avi"

"$YVEX_BIN" bench media publish --help >"$OUT_DIR/help.out"
contains "$OUT_DIR/help.out" "operation: execute.media.publish"
contains "$OUT_DIR/help.out" "--video-file"
contains "$OUT_DIR/help.out" "--audio-file"
contains "$OUT_DIR/help.out" "--audio-samples"

"$YVEX_BIN" bench media generate --help >"$OUT_DIR/generate-help.out"
contains "$OUT_DIR/generate-help.out" "operation: execute.media.generate"
contains "$OUT_DIR/generate-help.out" "--prompt"
contains "$OUT_DIR/generate-help.out" "--transformer-artifact"
contains "$OUT_DIR/generate-help.out" "--steps"

set +e
"$YVEX_BIN" bench media generate \
    --target refused-family --prompt hello \
    --text-artifact missing --transformer-artifact missing \
    --video-artifact missing --audio-artifact missing \
    --frames 124 --width 32 --height 32 --out "$OUT_DIR/refused.avi" \
    >"$OUT_DIR/generate-refused.out" 2>"$OUT_DIR/generate-refused.err"
status=$?
set -e
[ "$status" -ne 0 ] || fail "unknown generation target was admitted"
contains "$OUT_DIR/generate-refused.err" "the requested target has no admitted media adapter"
[ ! -e "$OUT_DIR/refused.avi" ] || fail "refused generation published media"

"$YVEX_BIN" bench media publish \
    --video-file "$OUT_DIR/video.f32" --audio-file "$OUT_DIR/audio.f32" \
    --frames 3 --width 2 --height 2 --audio-samples 4005 \
    --max-host-bytes 1048576 --max-output-bytes 1048576 \
    --out "$OUT_DIR/first.avi" --output audit >"$OUT_DIR/first.out"
"$YVEX_BIN" bench media publish \
    --video-file "$OUT_DIR/video.f32" --audio-file "$OUT_DIR/audio.f32" \
    --frames 3 --width 2 --height 2 --audio-samples 4005 \
    --max-host-bytes 1048576 --max-output-bytes 1048576 \
    --out "$OUT_DIR/second.avi" --output audit >"$OUT_DIR/second.out"

contains "$OUT_DIR/first.out" "status: media-published"
contains "$OUT_DIR/first.out" "container: avi-bgr24-pcm-s16le"
contains "$OUT_DIR/first.out" "video_fps: 24/1"
contains "$OUT_DIR/first.out" "audio_samples_used: 4000"
contains "$OUT_DIR/first.out" "audio_samples_trimmed: 5"
contains "$OUT_DIR/first.out" "file_bytes: 16524"
contains "$OUT_DIR/first.out" "duration: 3/24 seconds"
contains "$OUT_DIR/first.out" "end_user_path_available: false"
[ "$(stat -c %s "$OUT_DIR/first.avi")" -eq 16524 ] || fail "AVI extent differs"
[ "$(dd if="$OUT_DIR/first.avi" bs=1 count=4 2>/dev/null)" = RIFF ] ||
    fail "AVI RIFF signature missing"
cmp "$OUT_DIR/first.avi" "$OUT_DIR/second.avi" || fail "repeat AVI differs"

set +e
"$YVEX_BIN" bench media publish \
    --video-file "$OUT_DIR/video.f32" --audio-file "$OUT_DIR/audio.f32" \
    --frames 3 --width 2 --height 2 --audio-samples 4005 \
    --max-host-bytes 1048576 --max-output-bytes 1048576 \
    --out "$OUT_DIR/first.avi" >"$OUT_DIR/collision.out" 2>"$OUT_DIR/collision.err"
status=$?
set -e
[ "$status" -ne 0 ] || fail "existing AVI was overwritten"
cmp "$OUT_DIR/first.avi" "$OUT_DIR/second.avi" || fail "collision changed AVI"

printf 'cli media: ok\n'
