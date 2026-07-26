#!/bin/sh
# Core CLI smoke: catalog, stable process status, and bounded production adapters.

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

run_ok no_args "$YVEX_BIN"
contains "$OUT_DIR/no_args.out" "yvex - local-first inference engine"
contains "$OUT_DIR/no_args.out" "yvex graph attention execute"
omits "$OUT_DIR/no_args.out" "yvex generate"

run_ok help "$YVEX_BIN" --help
contains "$OUT_DIR/help.out" "command groups:"

run_ok version_option "$YVEX_BIN" --version
contains "$OUT_DIR/version_option.out" "yvex 0.1.0"
run_ok version_command "$YVEX_BIN" version
contains "$OUT_DIR/version_command.out" "yvex 0.1.0"

run_ok commands "$YVEX_BIN" commands
contains "$OUT_DIR/commands.out" "YVEX COMMAND CATALOG"
contains "$OUT_DIR/commands.out" "graph              graph"
contains "$OUT_DIR/commands.out" "input              runtime"
contains "$OUT_DIR/commands.out" "materialize        artifact"
for retired in engine session run chat prefill kv decode logits sample generate; do
    omits "$OUT_DIR/commands.out" "$retired             runtime"
done

run_ok help_graph "$YVEX_BIN" help graph
contains "$OUT_DIR/help_graph.out" "yvex graph attention"
contains "$OUT_DIR/help_graph.out" "Production attention, MoE, and transformer graph execution."
run_ok help_input "$YVEX_BIN" help input
contains "$OUT_DIR/help_input.out" "yvex input tokens"
run_ok help_paths "$YVEX_BIN" help paths
contains "$OUT_DIR/help_paths.out" "yvex paths"

run_ok inspect "$YVEX_BIN" inspect "$FIXTURE"
contains "$OUT_DIR/inspect.out" "format: gguf"
contains "$OUT_DIR/inspect.out" "status: descriptor-only"

run_ok metadata "$YVEX_BIN" metadata "$FIXTURE"
contains "$OUT_DIR/metadata.out" "general.architecture"
run_ok tensors "$YVEX_BIN" tensors "$FIXTURE"
contains "$OUT_DIR/tensors.out" "token_embd.weight"

run_ok tokenizer "$YVEX_BIN" tokenizer "$FIXTURE"
contains "$OUT_DIR/tokenizer.out" "status: tokenizer-descriptor"
run_ok tokenize "$YVEX_BIN" tokenize "$FIXTURE" --text "hello world"
contains "$OUT_DIR/tokenize.out" "ids: 3 4 5"
run_ok detokenize "$YVEX_BIN" detokenize "$FIXTURE" --ids 3,4,5
contains "$OUT_DIR/detokenize.out" "text: \"hello world\""
run_ok prompt "$YVEX_BIN" prompt "$FIXTURE" --user "hello world"
contains "$OUT_DIR/prompt.out" "status: rendered"

run_ok materialize "$YVEX_BIN" materialize --model "$FIXTURE" --backend cpu
contains "$OUT_DIR/materialize.out" "status: weights-materialized"
contains "$OUT_DIR/materialize.out" "execution_ready: false"

run_ok backend "$YVEX_BIN" backend cpu
contains "$OUT_DIR/backend.out" "status: backend-capabilities"
run_ok paths "$YVEX_BIN" paths
contains "$OUT_DIR/paths.out" "models_root:"

run_code unknown 2 "$YVEX_BIN" unknown
contains "$OUT_DIR/unknown.err" "unknown command: unknown"
run_code unknown_help 2 "$YVEX_BIN" help unknown
contains "$OUT_DIR/unknown_help.err" "unknown help topic: unknown"

for retired in engine session run chat prefill kv decode logits sample generate; do
    run_code "retired_$retired" 2 "$YVEX_BIN" "$retired"
    contains "$OUT_DIR/retired_$retired.err" "unknown command: $retired"
done

printf 'cli core smoke: ok\n'
