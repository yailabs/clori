#!/bin/sh
# Typed token-input smoke independent from graph execution.

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/token-input}
MODEL="$OUT_DIR/token-input-F16.gguf"
TOKENIZER_FIXTURE=tests/fixtures/gguf/valid-tokenizer-simple.gguf

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains() {
    file=$1
    value=$2
    grep -F -- "$value" "$file" >/dev/null || fail "$file missing: $value"
}

expect_fail() {
    name=$1
    shift
    if "$@" >"$OUT_DIR/$name.out" 2>"$OUT_DIR/$name.err"; then
        fail "$name unexpectedly passed"
    fi
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$YVEX_BIN" compile emit artifact controlled --out "$MODEL" --model-name token-input \
    --arch deepseek --target-qtype F16 --overwrite \
    >"$OUT_DIR/emit.out" 2>"$OUT_DIR/emit.err"

"$YVEX_BIN" execute input tokens --model "$MODEL" --tokens 0,1 \
    >"$OUT_DIR/tokens.out" 2>"$OUT_DIR/tokens.err"
contains "$OUT_DIR/tokens.out" "token_input_status: pass"
contains "$OUT_DIR/tokens.out" "token_input_kind: explicit"
contains "$OUT_DIR/tokens.out" "token_count: 2"
contains "$OUT_DIR/tokens.out" "token_bounds_status: pass"
contains "$OUT_DIR/tokens.out" "prefill_ready: false"
contains "$OUT_DIR/tokens.out" "generation: unsupported"
contains "$OUT_DIR/tokens.out" "status: token-input-pass"

expect_fail empty "$YVEX_BIN" execute input tokens --model "$MODEL" --tokens ""
contains "$OUT_DIR/empty.err" "token-list-empty"
expect_fail double_comma "$YVEX_BIN" execute input tokens --model "$MODEL" --tokens 1,,2
contains "$OUT_DIR/double_comma.err" "token-parse-invalid"
expect_fail alpha "$YVEX_BIN" execute input tokens --model "$MODEL" --tokens abc
contains "$OUT_DIR/alpha.err" "token-parse-invalid"
expect_fail negative "$YVEX_BIN" execute input tokens --model "$MODEL" --tokens -1
contains "$OUT_DIR/negative.err" "token-parse-invalid"
expect_fail overflow "$YVEX_BIN" execute input tokens --model "$MODEL" \
    --tokens 184467440737095516160
contains "$OUT_DIR/overflow.err" "token-id-overflow"
expect_fail out_of_vocab "$YVEX_BIN" execute input tokens --model "$MODEL" --tokens 8
contains "$OUT_DIR/out_of_vocab.out" "token_bounds_status: fail"
contains "$OUT_DIR/out_of_vocab.err" "token-out-of-vocab"

"$YVEX_BIN" execute input prompt --model "$TOKENIZER_FIXTURE" --text "hello world" \
    >"$OUT_DIR/prompt.out" 2>"$OUT_DIR/prompt.err"
contains "$OUT_DIR/prompt.out" "token_input_kind: prompt-text"
contains "$OUT_DIR/prompt.out" "token_count: 3"
contains "$OUT_DIR/prompt.out" "status: token-input-pass"

expect_fail tokenizer_missing "$YVEX_BIN" execute input prompt --model "$MODEL" --text hello
contains "$OUT_DIR/tokenizer_missing.out" "tokenizer_status: unsupported"
contains "$OUT_DIR/tokenizer_missing.err" "tokenizer-metadata-missing"

printf 'cli token input: ok\n'
