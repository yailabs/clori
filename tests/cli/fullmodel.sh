#!/bin/sh
# Fullmodel CLI smoke for inventory/materialization planning only.

set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
ROOT=${YVEX_TEST_OUT_DIR:-build/tests/fullmodel-cli}
MODEL=tests/fixtures/gguf/valid-tokenizer-simple.gguf

mkdir -p "$ROOT"

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains() {
    file=$1
    value=$2
    grep -F -- "$value" "$file" >/dev/null || fail "$file missing: $value"
}

"$YVEX_BIN" evidence model --help >"$ROOT/help.out" 2>"$ROOT/help.err"
contains "$ROOT/help.out" "usage: yvex evidence model report"
contains "$ROOT/help.out" "no full model execution"

"$YVEX_BIN" evidence model report --model "$MODEL" --backend cpu --audit \
    >"$ROOT/report.out" 2>"$ROOT/report.err"
contains "$ROOT/report.out" "status: fullmodel-report"
contains "$ROOT/report.out" "tensor_inventory_status: pass"
contains "$ROOT/report.out" "full_runtime_model: false"
contains "$ROOT/report.out" "generation: unsupported-full-model"

"$YVEX_BIN" evidence model materialization-plan --model "$MODEL" --backend cpu --audit \
    >"$ROOT/plan.out" 2>"$ROOT/plan.err"
contains "$ROOT/plan.out" "materialization"
contains "$ROOT/plan.out" "generation: unsupported-full-model"

"$YVEX_BIN" evidence model descriptor --model "$MODEL" --backend cpu --audit \
    >"$ROOT/descriptor.out" 2>"$ROOT/descriptor.err"
contains "$ROOT/descriptor.out" "status: fullmodel-descriptor"
contains "$ROOT/descriptor.out" "generation_ready: false"

if "$YVEX_BIN" evidence model report --model "$ROOT/missing.gguf" --audit \
    >"$ROOT/missing.out" 2>"$ROOT/missing.err"; then
    fail "missing fullmodel artifact unexpectedly passed"
fi
contains "$ROOT/missing.out" "artifact_exists: false"

if "$YVEX_BIN" evidence model report --model "$MODEL" --output nope \
    >"$ROOT/output.out" 2>"$ROOT/output.err"; then
    fail "invalid fullmodel output mode unexpectedly passed"
fi
contains "$ROOT/output.err" "unsupported output mode"

printf 'cli fullmodel smoke: ok\n'
