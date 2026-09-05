#!/bin/sh
# Exact artifact-bound MiniMax-H3 Qwen2 tokenizer operator conformance.

set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
ARTIFACT=${MINIMAX_H3_TEXT_ARTIFACT:?MINIMAX_H3_TEXT_ARTIFACT is required}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli/minimax-tokenizer}

mkdir -p "$OUT_DIR"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains() {
    grep -F -- "$2" "$1" >/dev/null || fail "$1 missing: $2"
}

"$YVEX_BIN" inspect tokenizer "$ARTIFACT" >"$OUT_DIR/inspect.out"
contains "$OUT_DIR/inspect.out" "support: artifact-bpe"
contains "$OUT_DIR/inspect.out" "base_vocab_size: 151643"
contains "$OUT_DIR/inspect.out" "merge_count: 151387"
contains "$OUT_DIR/inspect.out" "added_token_count: 26"
contains "$OUT_DIR/inspect.out" "special_token_count: 14"
contains "$OUT_DIR/inspect.out" "chat_template: verbatim-no-special"
contains "$OUT_DIR/inspect.out" "bos_token_id: absent"
contains "$OUT_DIR/inspect.out" "eos_token_id: 151645"
contains "$OUT_DIR/inspect.out" "pad_token_id: 151643"

"$YVEX_BIN" inspect tokenizer encode "$ARTIFACT" \
    --text 'A red fox jumps over a blue river.' >"$OUT_DIR/prompt-first.out"
"$YVEX_BIN" inspect tokenizer encode "$ARTIFACT" \
    --text 'A red fox jumps over a blue river.' >"$OUT_DIR/prompt-second.out"
cmp "$OUT_DIR/prompt-first.out" "$OUT_DIR/prompt-second.out" ||
    fail "repeated prompt encoding differs"
contains "$OUT_DIR/prompt-first.out" "tokens: 9"
contains "$OUT_DIR/prompt-first.out" "ids: 32 2518 38835 34208 916 264 6303 14796 13"
contains "$OUT_DIR/prompt-first.out" "tokenizer_runtime_ready: true"

"$YVEX_BIN" inspect tokenizer encode "$ARTIFACT" --text '2026' >"$OUT_DIR/numbers.out"
contains "$OUT_DIR/numbers.out" "ids: 17 15 17 21"
"$YVEX_BIN" inspect tokenizer encode "$ARTIFACT" \
    --text "can't WE'RE I'd she'll" >"$OUT_DIR/contractions.out"
contains "$OUT_DIR/contractions.out" "ids: 4814 944 19677 94153 358 4172 1340 3278"
"$YVEX_BIN" inspect tokenizer encode "$ARTIFACT" \
    --text ' spaces   around ' >"$OUT_DIR/space-runs.out"
contains "$OUT_DIR/space-runs.out" "ids: 12621 256 2163 220"
"$YVEX_BIN" inspect tokenizer encode "$ARTIFACT" \
    --text 'abc123xyz' >"$OUT_DIR/letter-number.out"
contains "$OUT_DIR/letter-number.out" "ids: 13683 16 17 18 28854"

decomposed=$(printf 'citta\314\200')
"$YVEX_BIN" inspect tokenizer encode "$ARTIFACT" --text 'città' \
    >"$OUT_DIR/nfc-composed.out"
"$YVEX_BIN" inspect tokenizer encode "$ARTIFACT" --text "$decomposed" \
    >"$OUT_DIR/nfc-decomposed.out"
composed_ids=$(sed -n 's/^ids: //p' "$OUT_DIR/nfc-composed.out")
decomposed_ids=$(sed -n 's/^ids: //p' "$OUT_DIR/nfc-decomposed.out")
[ -n "$composed_ids" ] || fail "composed NFC input produced no token IDs"
[ "$composed_ids" = "$decomposed_ids" ] ||
    fail "canonically equivalent Unicode input produced different token IDs"
contains "$OUT_DIR/nfc-composed.out" "ids: 66 1442 6362"
contains "$OUT_DIR/nfc-composed.out" "tokenizer_runtime_ready: true"

"$YVEX_BIN" inspect tokenizer encode "$ARTIFACT" --text '中文 日本語' \
    >"$OUT_DIR/multilingual.out"
contains "$OUT_DIR/multilingual.out" "tokenizer_runtime_ready: true"

set +e
"$YVEX_BIN" inspect tokenizer prompt "$ARTIFACT" --user hello \
    >"$OUT_DIR/chat.out" 2>"$OUT_DIR/chat.err"
chat_status=$?
set -e
[ "$chat_status" -ne 0 ] || fail "verbatim FL2VA tokenizer silently admitted chat rendering"
contains "$OUT_DIR/chat.err" "verbatim encoding rather than chat rendering"

printf 'cli minimax tokenizer: ok\n'
