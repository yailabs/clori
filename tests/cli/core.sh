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

run_ok no_args "$YVEX_BIN"
contains "$OUT_DIR/no_args.out" "YVEX inference/compiler/runtime"
contains "$OUT_DIR/no_args.out" "HOST"
contains "$OUT_DIR/no_args.out" "serve"
contains "$OUT_DIR/no_args.out" "chat"
omits "$OUT_DIR/no_args.out" "yvex run"

run_code chat_non_tty 2 "$YVEX_BIN" chat
contains "$OUT_DIR/chat_non_tty.err" "chat requires a terminal"
contains "$OUT_DIR/chat_non_tty.err" "configured provider API"

run_ok help "$YVEX_BIN" --help
contains "$OUT_DIR/help.out" "YVEX inference/compiler/runtime"
contains "$OUT_DIR/help.out" "USE"
contains "$OUT_DIR/help.out" "BUILD"
contains "$OUT_DIR/help.out" "INSPECT"
omits "$OUT_DIR/help.out" "yvex run"
omits "$OUT_DIR/help.out" "yvex server"

run_ok version_option "$YVEX_BIN" --version
contains "$OUT_DIR/version_option.out" "yvex 0.1.0"
run_ok version_command "$YVEX_BIN" version
contains "$OUT_DIR/version_command.out" "yvex 0.1.0"

run_ok help_graph "$YVEX_BIN" help bench attention --advanced
contains "$OUT_DIR/help_graph.out" "yvex bench attention execute"
run_ok help_input "$YVEX_BIN" inspect input --help
contains "$OUT_DIR/help_input.out" "operation: input.prepare"
run_ok help_paths "$YVEX_BIN" inspect paths --help
contains "$OUT_DIR/help_paths.out" "operation: system.paths"

run_ok inspect "$YVEX_BIN" artifact show "$FIXTURE"
contains "$OUT_DIR/inspect.out" "format: gguf"
contains "$OUT_DIR/inspect.out" "status: descriptor-only"

run_ok metadata "$YVEX_BIN" inspect artifact metadata "$FIXTURE"
contains "$OUT_DIR/metadata.out" "general.architecture"
run_ok tensors "$YVEX_BIN" inspect artifact tensors "$FIXTURE"
contains "$OUT_DIR/tensors.out" "token_embd.weight"

run_ok tokenizer "$YVEX_BIN" inspect tokenizer "$FIXTURE"
contains "$OUT_DIR/tokenizer.out" "status: tokenizer-descriptor"
run_ok tokenize "$YVEX_BIN" inspect tokenizer encode "$FIXTURE" --text "hello world"
contains "$OUT_DIR/tokenize.out" "ids: 3 4 5"
run_ok detokenize "$YVEX_BIN" inspect tokenizer decode "$FIXTURE" --ids 3,4,5
contains "$OUT_DIR/detokenize.out" "text: \"hello world\""
run_ok prompt "$YVEX_BIN" inspect tokenizer prompt "$FIXTURE" --user "hello world"
contains "$OUT_DIR/prompt.out" "status: rendered"

run_ok materialize "$YVEX_BIN" artifact materialize --model "$FIXTURE" --backend cpu
contains "$OUT_DIR/materialize.out" "status: weights-materialized"
contains "$OUT_DIR/materialize.out" "execution_ready: false"

run_ok backend "$YVEX_BIN" inspect backend cpu
contains "$OUT_DIR/backend.out" "status: backend-capabilities"
run_ok paths "$YVEX_BIN" inspect paths
contains "$OUT_DIR/paths.out" "models_root:"

run_code retired_run 2 "$YVEX_BIN" run
contains "$OUT_DIR/retired_run.err" "removed command: run"
contains "$OUT_DIR/retired_run.err" "yvex chat"
run_code retired_server 2 "$YVEX_BIN" server status
contains "$OUT_DIR/retired_server.err" "removed command: server"
contains "$OUT_DIR/retired_server.err" "yvex host"

run_code unknown 2 "$YVEX_BIN" unknown
contains "$OUT_DIR/unknown.err" "unknown command: unknown"
run_code unknown_help 2 "$YVEX_BIN" help unknown
contains "$OUT_DIR/unknown_help.err" "unknown help path unknown"

for retired in inspect materialize quant-policy metadata tensor-map model-target fullmodel; do
    run_code "retired_$retired" 2 "$YVEX_BIN" "$retired"
    contains "$OUT_DIR/retired_$retired.err" "unknown command: $retired"
done
for retired in evidence execute graph quant system tensor tokenizer; do
    run_code "removed_$retired" 2 "$YVEX_BIN" "$retired"
    contains "$OUT_DIR/removed_$retired.err" "removed command: $retired"
done

printf 'cli core smoke: ok\n'
