#!/bin/sh
# Core CLI smoke: unified help, nested offline routes, and retired-alias refusal.

set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/cli/core}
FIXTURE=tests/fixtures/gguf/valid-tokenizer-simple.gguf

mkdir -p "$OUT_DIR"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

run_ok() {
    name=$1
    shift
    "$@" >"$OUT_DIR/$name.out" 2>"$OUT_DIR/$name.err" || fail "$name exited non-zero"
}

run_code() {
    name=$1
    expected=$2
    shift 2
    set +e
    "$@" >"$OUT_DIR/$name.out" 2>"$OUT_DIR/$name.err"
    rc=$?
    set -e
    [ "$rc" -eq "$expected" ] || fail "$name exit code was $rc, expected $expected"
}

contains() {
    file=$1
    value=$2
    grep -F -- "$value" "$file" >/dev/null || fail "$file missing: $value"
}

omits() {
    file=$1
    value=$2
    if grep -F -- "$value" "$file" >/dev/null; then
        fail "$file unexpectedly contains: $value"
    fi
}

run_code no_args 2 "$YVEX_BIN"
contains "$OUT_DIR/no_args.err" "chat requires a terminal"
contains "$OUT_DIR/no_args.err" "yvex run TEXT"

run_ok help "$YVEX_BIN" --help
contains "$OUT_DIR/help.out" "YVEX local inference"
contains "$OUT_DIR/help.out" "yvex artifact show|verify"
contains "$OUT_DIR/help.out" "yvex evidence target|model|moe|backend|cuda"

run_ok version_option "$YVEX_BIN" --version
contains "$OUT_DIR/version_option.out" "yvex 0.1.0"
run_ok version_command "$YVEX_BIN" version
contains "$OUT_DIR/version_command.out" "yvex 0.1.0"

run_ok help_graph "$YVEX_BIN" graph --help
contains "$OUT_DIR/help_graph.out" "yvex graph attention"
run_ok help_input "$YVEX_BIN" runtime input --help
contains "$OUT_DIR/help_input.out" "yvex runtime input tokens"
run_ok help_paths "$YVEX_BIN" evidence paths --help
contains "$OUT_DIR/help_paths.out" "yvex evidence paths"

run_ok inspect "$YVEX_BIN" artifact show "$FIXTURE"
contains "$OUT_DIR/inspect.out" "format: gguf"
contains "$OUT_DIR/inspect.out" "status: descriptor-only"

run_ok metadata "$YVEX_BIN" artifact metadata "$FIXTURE"
contains "$OUT_DIR/metadata.out" "general.architecture"
run_ok tensors "$YVEX_BIN" artifact tensors "$FIXTURE"
contains "$OUT_DIR/tensors.out" "token_embd.weight"

run_ok tokenizer "$YVEX_BIN" tokenizer show "$FIXTURE"
contains "$OUT_DIR/tokenizer.out" "status: tokenizer-descriptor"
run_ok tokenize "$YVEX_BIN" tokenizer encode "$FIXTURE" --text "hello world"
contains "$OUT_DIR/tokenize.out" "ids: 3 4 5"
run_ok detokenize "$YVEX_BIN" tokenizer decode "$FIXTURE" --ids 3,4,5
contains "$OUT_DIR/detokenize.out" "text: \"hello world\""
run_ok prompt "$YVEX_BIN" tokenizer prompt "$FIXTURE" --user "hello world"
contains "$OUT_DIR/prompt.out" "status: rendered"

run_ok materialize "$YVEX_BIN" artifact materialize --model "$FIXTURE" --backend cpu
contains "$OUT_DIR/materialize.out" "status: weights-materialized"
contains "$OUT_DIR/materialize.out" "execution_ready: false"

run_ok backend "$YVEX_BIN" evidence backend cpu
contains "$OUT_DIR/backend.out" "status: backend-capabilities"
run_ok paths "$YVEX_BIN" evidence paths
contains "$OUT_DIR/paths.out" "models_root:"

run_code unknown 2 "$YVEX_BIN" unknown
contains "$OUT_DIR/unknown.err" "unknown command: unknown"
run_code unknown_help 2 "$YVEX_BIN" help unknown
contains "$OUT_DIR/unknown_help.err" "unknown help topic: unknown"

for retired in inspect materialize quant-policy tokenizer metadata tensor-map model-target fullmodel; do
    run_code "retired_$retired" 2 "$YVEX_BIN" "$retired"
    contains "$OUT_DIR/retired_$retired.err" "unknown command: $retired"
done

printf 'cli core smoke: ok\n'
