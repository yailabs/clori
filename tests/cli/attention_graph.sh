#!/bin/sh
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_PARENT=${YVEX_TEST_OUT_DIR:-build/tests/cli/attention-graph}
case "$OUT_PARENT" in
    /*) ;;
    *) OUT_PARENT="$PWD/$OUT_PARENT" ;;
esac
yvex_test_cleanup_validate "$OUT_PARENT"
mkdir -p "$OUT_PARENT"
test -d "$OUT_PARENT" && test ! -L "$OUT_PARENT" || {
    printf 'FAIL: unsafe attention test root: %s\n' "$OUT_PARENT" >&2
    exit 1
}
OUT_PARENT=$(cd "$OUT_PARENT" && pwd -P)
OUT_DIR=$(mktemp -d "$OUT_PARENT/run.XXXXXX")
yvex_test_cleanup_validate "$OUT_DIR"

cleanup() {
    status=$?
    trap - 0 HUP INT TERM
    set +e
    yvex_test_cleanup_preserving_status "$status" "$OUT_DIR"
    final_status=$?
    exit "$final_status"
}

trap cleanup 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

contains() {
    grep -F -- "$2" "$1" >/dev/null || fail "$1 missing: $2"
}

expect_status() {
    expected=$1
    shift
    set +e
    "$@"
    actual=$?
    set -e
    [ "$actual" -eq "$expected" ] || fail "expected status $expected, got $actual: $*"
}

expect_nonzero() {
    set +e
    "$@"
    actual=$?
    set -e
    [ "$actual" -ne 0 ] || fail "expected nonzero status: $*"
}

"$YVEX_BIN" help graph >"$OUT_DIR/help.out" 2>"$OUT_DIR/help.err"
"$YVEX_BIN" graph attention --help >"$OUT_DIR/attention-help.out" \
    2>"$OUT_DIR/attention-help.err"
"$YVEX_BIN" graph attention execute --help >"$OUT_DIR/execute-help.out" \
    2>"$OUT_DIR/execute-help.err"
"$YVEX_BIN" graph moe execute --help >"$OUT_DIR/moe-help.out" \
    2>"$OUT_DIR/moe-help.err"
"$YVEX_BIN" graph transformer execute --help >"$OUT_DIR/transformer-help.out" \
    2>"$OUT_DIR/transformer-help.err"
for action in prepare describe capabilities plan execute compare; do
    "$YVEX_BIN" graph attention "$action" --help \
        >"$OUT_DIR/$action-help.out" 2>"$OUT_DIR/$action-help.err"
    contains "$OUT_DIR/$action-help.out" "graph attention $action"
done
for action in "state inspect" "state validate" "state exercise" \
              "residency inspect" capture replay "cuda-graph list" \
              "cuda-graph inspect" "cuda-graph warmup" "cuda-graph update" \
              "cuda-graph invalidate" "cuda-graph release" \
              trace profile benchmark "benchmark compare" qualify; do
    # shellcheck disable=SC2086
    "$YVEX_BIN" graph attention $action --help \
        >"$OUT_DIR/action-help.out" 2>"$OUT_DIR/action-help.err"
    contains "$OUT_DIR/action-help.out" "graph attention"
done
"$YVEX_BIN" commands >"$OUT_DIR/catalog.out" 2>"$OUT_DIR/catalog.err"
contains "$OUT_DIR/catalog.out" "Production attention, MoE, and transformer graph execution."
contains "$OUT_DIR/help.out" "yvex graph attention execute --target deepseek4-v4-flash"
contains "$OUT_DIR/help.out" "--backend cpu|cuda"
contains "$OUT_DIR/help.out" "--compare-backends"
contains "$OUT_DIR/help.out" "--scope quick|full"
contains "$OUT_DIR/help.out" "graph attention prepare"
contains "$OUT_DIR/help.out" "graph attention describe"
contains "$OUT_DIR/help.out" "graph attention capabilities"
contains "$OUT_DIR/help.out" "graph attention plan"
contains "$OUT_DIR/help.out" "graph attention compare"
contains "$OUT_DIR/help.out" "graph attention state inspect|validate|exercise"
contains "$OUT_DIR/help.out" \
    "graph attention cuda-graph list|inspect|warmup|update|invalidate|release"
contains "$OUT_DIR/help.out" "graph attention trace|profile|benchmark|qualify"
contains "$OUT_DIR/help.out" "graph attention benchmark compare"
contains "$OUT_DIR/help.out" "--runtime-binding FILE"
contains "$OUT_DIR/help.out" "--runtime-binding-dir DIR"
contains "$OUT_DIR/help.out" "--models-root DIR"
contains "$OUT_DIR/help.out" "--phase prefill|decode|mixed|verify"
contains "$OUT_DIR/help.out" "--mode eager|piecewise|full|auto"
contains "$OUT_DIR/help.out" "--operation-scope core|envelope|release-attention-set"
contains "$OUT_DIR/help.out" "--tokens N"
contains "$OUT_DIR/help.out" "--warmup N"
contains "$OUT_DIR/help.out" "--repeat N"
contains "$OUT_DIR/help.out" "--trace-level none|summary|stages|full"
contains "$OUT_DIR/help.out" "--max-host-bytes N"
contains "$OUT_DIR/help.out" "--max-device-bytes N"
contains "$OUT_DIR/help.out" "--require-mode"
contains "$OUT_DIR/help.out" "reserved controls refuse until their typed runtime owners exist"
contains "$OUT_DIR/help.out" "--input tensor-file"
contains "$OUT_DIR/help.out" "--input-file FILE"
contains "$OUT_DIR/help.out" "--chunk-tokens N"
contains "$OUT_DIR/help.out" "--context-capacity N"
contains "$OUT_DIR/help.out" "--capture-bucket ID"
contains "$OUT_DIR/help.out" "--baseline FILE"
contains "$OUT_DIR/help.out" "--current FILE"
contains "$OUT_DIR/help.out" "--max-regression-bps N"
contains "$OUT_DIR/help.out" "--write-baseline"
contains "$OUT_DIR/help.out" "--chart PATH.svg"
contains "$OUT_DIR/help.out" "--output normal|table|audit|json|csv"
contains "$OUT_DIR/help.out" "not prompt execution"
contains "$OUT_DIR/attention-help.out" "yvex graph attention execute"
contains "$OUT_DIR/execute-help.out" "yvex graph attention execute"
contains "$OUT_DIR/moe-help.out" "yvex graph moe execute"
contains "$OUT_DIR/moe-help.out" "--input tensor-file --input-file FILE"
contains "$OUT_DIR/moe-help.out" "token IDs are numeric routing input"
contains "$OUT_DIR/transformer-help.out" "yvex graph transformer execute"
contains "$OUT_DIR/transformer-help.out" "--input token-ids --input-file FILE"
contains "$OUT_DIR/transformer-help.out" "do not establish tokenizer, logits, decode, or generation"

expect_status 2 "$YVEX_BIN" graph transformer execute \
    >"$OUT_DIR/transformer-missing.out" 2>"$OUT_DIR/transformer-missing.err"
contains "$OUT_DIR/transformer-missing.err" \
    "requires target, artifact, runtime binding, backend, token input, chunk tokens, and context capacity"

expect_status 2 "$YVEX_BIN" graph transformer execute --target deepseek4-v4-flash \
    --artifact /tmp/missing.gguf --runtime-binding /tmp/missing.binding \
    --backend cpu --phase decode --input token-ids --input-file /tmp/missing.input \
    --chunk-tokens 1 --context-capacity 1 --progress off --output json \
    >"$OUT_DIR/transformer-phase.out" 2>"$OUT_DIR/transformer-phase.err"
contains "$OUT_DIR/transformer-phase.err" "supports only --phase prefill"

expect_status 3 "$YVEX_BIN" graph transformer execute --target deepseek4-v4-flash \
    --artifact /tmp/missing.gguf --runtime-binding /tmp/missing.binding \
    --backend cpu --phase prefill --input token-ids --input-file /tmp/missing.input \
    --chunk-tokens 1 --context-capacity 1 --progress off --output csv \
    >"$OUT_DIR/transformer-refusal.csv" 2>"$OUT_DIR/transformer-refusal.err"
contains "$OUT_DIR/transformer-refusal.csv" "field,value"
contains "$OUT_DIR/transformer-refusal.csv" '"command","graph transformer execute"'
contains "$OUT_DIR/transformer-refusal.csv" '"status","refused"'
contains "$OUT_DIR/transformer-refusal.err" "runtime binding open failed"

expect_status 2 "$YVEX_BIN" graph moe execute \
    >"$OUT_DIR/moe-missing-options.out" 2>"$OUT_DIR/moe-missing-options.err"
contains "$OUT_DIR/moe-missing-options.err" \
    "requires target, artifact, runtime binding, backend, and tensor-file input"

expect_status 2 "$YVEX_BIN" graph moe execute --target deepseek4-v4-flash \
    --artifact /tmp/missing.gguf --runtime-binding /tmp/missing.binding \
    --backend cpu --input canonical --input-file /tmp/missing.input \
    --scope full --progress off --output json \
    >"$OUT_DIR/moe-input-class.out" 2>"$OUT_DIR/moe-input-class.err"
contains "$OUT_DIR/moe-input-class.err" "requires --input tensor-file"

expect_status 2 "$YVEX_BIN" graph moe execute --target deepseek4-v4-flash \
    --artifact /tmp/missing.gguf --runtime-binding /tmp/missing.binding \
    --backend cpu --input tensor-file --input-file /tmp/missing.input \
    --scope quick --progress off --output json \
    >"$OUT_DIR/moe-scope.out" 2>"$OUT_DIR/moe-scope.err"
contains "$OUT_DIR/moe-scope.err" "supports only --scope full"

expect_status 3 "$YVEX_BIN" graph moe execute --target deepseek4-v4-flash \
    --artifact /tmp/missing.gguf --runtime-binding /tmp/missing.binding \
    --backend cpu --input tensor-file --input-file /tmp/missing.input \
    --scope full --progress off --output json \
    >"$OUT_DIR/moe-runtime-refusal.json" 2>"$OUT_DIR/moe-runtime-refusal.err"
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); \
    assert r["status"]=="refused" and not r["completed"] \
    and not r["moe_block_ready"] and not r["transformer_ready"] \
    and not r["generation_ready"]' "$OUT_DIR/moe-runtime-refusal.json"
contains "$OUT_DIR/moe-runtime-refusal.err" "runtime binding open failed"

expect_status 2 "$YVEX_BIN" graph attention qualify \
    --target deepseek4-v4-flash \
    >"$OUT_DIR/qualify-backend.out" 2>"$OUT_DIR/qualify-backend.err"
contains "$OUT_DIR/qualify-backend.err" "requires --backend cpu|cuda"

expect_status 2 "$YVEX_BIN" graph attention benchmark compare \
    --baseline "$OUT_DIR/missing-baseline.yvex-benchmark" \
    >"$OUT_DIR/benchmark-compare-paths.out" 2>"$OUT_DIR/benchmark-compare-paths.err"
contains "$OUT_DIR/benchmark-compare-paths.err" "requires --baseline FILE and --current FILE"

expect_status 2 "$YVEX_BIN" graph attention benchmark \
    --target deepseek4-v4-flash --backend cpu \
    --max-regression-bps 0 \
    >"$OUT_DIR/benchmark-threshold-owner.out" 2>"$OUT_DIR/benchmark-threshold-owner.err"
contains "$OUT_DIR/benchmark-threshold-owner.err" \
    "regression thresholds require graph attention benchmark compare"

expect_status 2 "$YVEX_BIN" graph attention \
    >"$OUT_DIR/missing-action.out" 2>"$OUT_DIR/missing-action.err"
contains "$OUT_DIR/missing-action.err" "graph attention requires an action"

expect_status 2 "$YVEX_BIN" graph attention inspect \
    >"$OUT_DIR/unknown-action.out" 2>"$OUT_DIR/unknown-action.err"
contains "$OUT_DIR/unknown-action.err" "unknown graph attention action: inspect"

expect_status 2 "$YVEX_BIN" graph attention execute --backend cpu \
    >"$OUT_DIR/missing-target.out" 2>"$OUT_DIR/missing-target.err"
contains "$OUT_DIR/missing-target.err" "requires --target TARGET"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --compare-backends \
    >"$OUT_DIR/backend-conflict.out" 2>"$OUT_DIR/backend-conflict.err"
contains "$OUT_DIR/backend-conflict.err" "cannot be combined with --backend"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --probe fixture \
    >"$OUT_DIR/probe-refusal.out" 2>"$OUT_DIR/probe-refusal.err"
contains "$OUT_DIR/probe-refusal.err" "unsupported attention probe: fixture"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --scope reduced \
    >"$OUT_DIR/scope-refusal.out" 2>"$OUT_DIR/scope-refusal.err"
contains "$OUT_DIR/scope-refusal.err" "unsupported attention scope: reduced"

expect_status 2 "$YVEX_BIN" graph attention capabilities \
    --target deepseek4-v4-flash --output json \
    >"$OUT_DIR/capability-backend.out" 2>"$OUT_DIR/capability-backend.err"
contains "$OUT_DIR/capability-backend.err" "requires --backend cpu|cuda"

expect_status 2 "$YVEX_BIN" graph attention plan \
    --target deepseek4-v4-flash --backend metal --output json \
    >"$OUT_DIR/plan-backend.out" 2>"$OUT_DIR/plan-backend.err"
contains "$OUT_DIR/plan-backend.err" "unknown backend kind: metal"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --phase generation \
    >"$OUT_DIR/phase-refusal.out" 2>"$OUT_DIR/phase-refusal.err"
contains "$OUT_DIR/phase-refusal.err" "unsupported attention phase: generation"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --mode fallback \
    >"$OUT_DIR/mode-refusal.out" 2>"$OUT_DIR/mode-refusal.err"
contains "$OUT_DIR/mode-refusal.err" "unsupported attention mode: fallback"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --operation-scope transformer \
    >"$OUT_DIR/operation-scope.out" 2>"$OUT_DIR/operation-scope.err"
contains "$OUT_DIR/operation-scope.err" "unsupported attention operation scope: transformer"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --scope quick \
    --operation-scope release-attention-set \
    >"$OUT_DIR/release-scope.out" 2>"$OUT_DIR/release-scope.err"
contains "$OUT_DIR/release-scope.err" \
    "release attention set requires complete unfiltered layer coverage"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --trace-level everything \
    >"$OUT_DIR/trace-refusal.out" 2>"$OUT_DIR/trace-refusal.err"
contains "$OUT_DIR/trace-refusal.err" "unsupported attention trace level: everything"

expect_status 3 "$YVEX_BIN" graph attention trace \
    --target deepseek4-v4-flash --backend cpu --trace-level stages \
    --runtime-binding "$OUT_DIR/missing-stage-trace.yvex-runtime-binding" \
    >"$OUT_DIR/stage-trace.out" 2>"$OUT_DIR/stage-trace.err"
contains "$OUT_DIR/stage-trace.err" "runtime binding"

for control in "--layer-start 0 --layer-count 2" \
               "--local-capacity 0" \
               "--compressed-capacity 0" "--indexer-capacity 0"; do
    # shellcheck disable=SC2086
    expect_status 2 "$YVEX_BIN" graph attention execute \
        --target deepseek4-v4-flash --backend cpu $control \
        >"$OUT_DIR/control-refusal.out" 2>"$OUT_DIR/control-refusal.err"
    contains "$OUT_DIR/control-refusal.err" "unavailable until"
done
expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --input tensor-file \
    >"$OUT_DIR/input-file-required.out" 2>"$OUT_DIR/input-file-required.err"
contains "$OUT_DIR/input-file-required.err" \
    "--input tensor-file and --input-file are required together"
expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --input-file "$OUT_DIR/missing.activations" \
    >"$OUT_DIR/input-class-required.out" 2>"$OUT_DIR/input-class-required.err"
contains "$OUT_DIR/input-class-required.err" \
    "--input tensor-file and --input-file are required together"
expect_status 3 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --phase prefill --mode eager \
    --scope full --operation-scope core --input tensor-file \
    --input-file "$OUT_DIR/missing.activations" --chunk-tokens 1 \
    --context-capacity 1 \
    --runtime-binding "$OUT_DIR/missing-input.yvex-runtime-binding" \
    >"$OUT_DIR/input-runtime-refusal.out" 2>"$OUT_DIR/input-runtime-refusal.err"
contains "$OUT_DIR/input-runtime-refusal.err" "runtime binding"
expect_status 3 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --layer 0 \
    --runtime-binding "$OUT_DIR/missing-layer.yvex-runtime-binding" \
    >"$OUT_DIR/layer.out" 2>"$OUT_DIR/layer.err"
contains "$OUT_DIR/layer.err" "runtime binding"
expect_status 3 "$YVEX_BIN" graph attention state exercise \
    --target deepseek4-v4-flash --backend cpu --class csa --history-tokens 4 --tokens 2 \
    --runtime-binding "$OUT_DIR/missing-state.yvex-runtime-binding" \
    >"$OUT_DIR/state-selection.out" 2>"$OUT_DIR/state-selection.err"
contains "$OUT_DIR/state-selection.err" "runtime binding"
expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --capture-bucket decode-1 \
    >"$OUT_DIR/capture-bucket-refusal.out" 2>"$OUT_DIR/capture-bucket-refusal.err"
contains "$OUT_DIR/capture-bucket-refusal.err" \
    "--capture-bucket requires CUDA piecewise, full, or auto mode"

expect_status 2 "$YVEX_BIN" graph attention capture \
    --target deepseek4-v4-flash --mode eager \
    >"$OUT_DIR/capture-eager.out" 2>"$OUT_DIR/capture-eager.err"
contains "$OUT_DIR/capture-eager.err" \
    "CUDA graph actions require piecewise, full, or auto mode"

expect_status 3 "$YVEX_BIN" graph attention capture \
    --target deepseek4-v4-flash --mode piecewise --capture-bucket decode-1 \
    --runtime-binding "$OUT_DIR/missing-capture.yvex-runtime-binding" --output json \
    >"$OUT_DIR/capture-piecewise.json" 2>"$OUT_DIR/capture-piecewise.err"
contains "$OUT_DIR/capture-piecewise.err" "runtime binding"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --baseline baseline.yvex-benchmark \
    >"$OUT_DIR/baseline-action.out" 2>"$OUT_DIR/baseline-action.err"
contains "$OUT_DIR/baseline-action.err" "require graph attention benchmark or profile"

expect_status 2 "$YVEX_BIN" graph attention benchmark \
    --target deepseek4-v4-flash --backend cpu --write-baseline \
    >"$OUT_DIR/baseline-path.out" 2>"$OUT_DIR/baseline-path.err"
contains "$OUT_DIR/baseline-path.err" "--write-baseline requires --baseline FILE"

expect_status 2 "$YVEX_BIN" graph attention benchmark \
    --target deepseek4-v4-flash --backend cpu --chart benchmark.png \
    >"$OUT_DIR/chart-suffix.out" 2>"$OUT_DIR/chart-suffix.err"
contains "$OUT_DIR/chart-suffix.err" "--chart path must end in .svg"

expect_status 2 "$YVEX_BIN" graph attention benchmark \
    --target deepseek4-v4-flash --backend cpu --chart benchmark.svg \
    >"$OUT_DIR/chart-relative.out" 2>"$OUT_DIR/chart-relative.err"
contains "$OUT_DIR/chart-relative.err" \
    "benchmark assets require canonical absolute paths outside the source repository"

expect_status 2 "$YVEX_BIN" graph attention benchmark \
    --target deepseek4-v4-flash --backend cpu --chart "$PWD/benchmark.svg" \
    >"$OUT_DIR/chart-repository.out" 2>"$OUT_DIR/chart-repository.err"
contains "$OUT_DIR/chart-repository.err" \
    "benchmark assets require canonical absolute paths outside the source repository"

expect_status 2 "$YVEX_BIN" graph attention benchmark \
    --target deepseek4-v4-flash --backend cpu \
    --baseline benchmark.yvex-benchmark --write-baseline \
    >"$OUT_DIR/baseline-relative.out" 2>"$OUT_DIR/baseline-relative.err"
contains "$OUT_DIR/baseline-relative.err" \
    "benchmark assets require canonical absolute paths outside the source repository"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --max-device-bytes 1 \
    >"$OUT_DIR/device-budget.out" 2>"$OUT_DIR/device-budget.err"
contains "$OUT_DIR/device-budget.err" "--max-device-bytes requires a CUDA execution path"

expect_status 3 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --max-host-bytes 1 --require-mode \
    --runtime-binding "$OUT_DIR/missing-budget.yvex-runtime-binding" --output json \
    >"$OUT_DIR/budget-refusal.json" 2>"$OUT_DIR/budget-refusal.err"

for action in "state exercise" capture replay "cuda-graph list" "cuda-graph inspect" \
              "cuda-graph warmup" "cuda-graph update" "cuda-graph invalidate" \
              "cuda-graph release"; do
    # shellcheck disable=SC2086
    expect_status 3 "$YVEX_BIN" graph attention $action \
        --target deepseek4-v4-flash \
        --runtime-binding "$OUT_DIR/missing-action.yvex-runtime-binding" --output json \
        >"$OUT_DIR/action-refusal.json" 2>"$OUT_DIR/action-refusal.err"
    python3 - "$OUT_DIR/action-refusal.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["status"] == "refused"
assert result["runtime_generation_ready"] is False
assert result["command"].startswith("graph attention ")
PY
done
for action in trace profile benchmark qualify; do
    expect_status 3 "$YVEX_BIN" graph attention "$action" \
        --target deepseek4-v4-flash --backend cpu \
        --runtime-binding "$OUT_DIR/missing-$action.yvex-runtime-binding" --output json \
        >"$OUT_DIR/action-refusal.json" 2>"$OUT_DIR/action-refusal.err"
done

expect_status 3 "$YVEX_BIN" graph attention benchmark \
    --target deepseek4-v4-flash --backend cpu --warmup 1 --repeat 2 \
    --runtime-binding "$OUT_DIR/missing-benchmark.yvex-runtime-binding" --output csv \
    >"$OUT_DIR/benchmark-refusal.csv" 2>"$OUT_DIR/benchmark-refusal.err"
contains "$OUT_DIR/benchmark-refusal.csv" 'field,value'
contains "$OUT_DIR/benchmark-refusal.csv" '"command","graph attention benchmark"'
if rg 'benchmark_(path|baseline|cold_delta|chart)' "$OUT_DIR/benchmark-refusal.csv" \
    >"$OUT_DIR/benchmark-refusal-optionals.out"; then
    fail "refused CSV benchmark exposed unavailable baseline or chart fields"
fi

for phase in prefill decode; do
    expect_status 3 "$YVEX_BIN" graph attention execute \
        --target deepseek4-v4-flash --backend cpu --phase "$phase" \
        --runtime-binding "$OUT_DIR/missing-$phase.yvex-runtime-binding" --output json \
        >"$OUT_DIR/phase-$phase.json" 2>"$OUT_DIR/phase-$phase.err"
done
for phase in mixed verify; do
    expect_status 5 "$YVEX_BIN" graph attention execute \
        --target deepseek4-v4-flash --backend cpu --phase "$phase" \
        --runtime-binding "$OUT_DIR/missing-$phase.yvex-runtime-binding" --output json \
        >"$OUT_DIR/phase-$phase.json" 2>"$OUT_DIR/phase-$phase.err"
done
for mode in eager auto; do
    expect_status 3 "$YVEX_BIN" graph attention execute \
        --target deepseek4-v4-flash --backend cpu --mode "$mode" \
        --runtime-binding "$OUT_DIR/missing-$mode.yvex-runtime-binding" --output json \
        >"$OUT_DIR/mode-$mode.json" 2>"$OUT_DIR/mode-$mode.err"
done
for mode in piecewise full; do
    expect_status 5 "$YVEX_BIN" graph attention execute \
        --target deepseek4-v4-flash --backend cpu --mode "$mode" \
        --runtime-binding "$OUT_DIR/missing-$mode.yvex-runtime-binding" --output json \
        >"$OUT_DIR/mode-$mode.json" 2>"$OUT_DIR/mode-$mode.err"
done
for operation_scope in core envelope release-attention-set; do
    coverage=quick
    if test "$operation_scope" = release-attention-set; then coverage=full; fi
    expect_status 3 "$YVEX_BIN" graph attention execute \
        --target deepseek4-v4-flash --backend cpu --scope "$coverage" \
        --operation-scope "$operation_scope" \
        --runtime-binding "$OUT_DIR/missing-$operation_scope.yvex-runtime-binding" --output json \
        >"$OUT_DIR/scope-$operation_scope.json" 2>"$OUT_DIR/scope-$operation_scope.err"
done
python3 - "$OUT_DIR" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
paths = []
for prefix in ("phase-", "mode-", "scope-"):
    paths.extend(glob.glob(os.path.join(root, prefix + "*.json")))
assert len(paths) == 11
for path in paths:
    with open(path, encoding="utf-8") as stream:
        result = json.load(stream)
    assert result["status"] == "refused"
    assert result["runtime_generation_ready"] is False
    name = os.path.basename(path)
    if name in ("phase-mixed.json", "phase-verify.json",
                "mode-piecewise.json", "mode-full.json"):
        assert result["failure_code"] == "YVEX_ERR_UNSUPPORTED"
    else:
        assert result["failure_code"] == "YVEX_ERR_IO"
PY

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --tokens 0 \
    >"$OUT_DIR/token-refusal.out" 2>"$OUT_DIR/token-refusal.err"
contains "$OUT_DIR/token-refusal.err" "--tokens requires a positive integer"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --repeat 0 \
    >"$OUT_DIR/repeat-refusal.out" 2>"$OUT_DIR/repeat-refusal.err"
contains "$OUT_DIR/repeat-refusal.err" "--repeat requires a positive integer"

expect_status 2 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu \
    --runtime-binding "$OUT_DIR/one.yvex-runtime-binding" \
    --runtime-binding-dir "$OUT_DIR" \
    >"$OUT_DIR/binding-conflict.out" 2>"$OUT_DIR/binding-conflict.err"
contains "$OUT_DIR/binding-conflict.err" "--runtime-binding conflicts with --runtime-binding-dir"

expect_status 2 "$YVEX_BIN" graph attention compare \
    --target deepseek4-v4-flash --backend cpu \
    >"$OUT_DIR/compare-backend.out" 2>"$OUT_DIR/compare-backend.err"
contains "$OUT_DIR/compare-backend.err" "compare does not accept --backend"

expect_status 3 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu --phase prefill --mode eager \
    --operation-scope envelope --runtime-binding "$OUT_DIR/missing.yvex-runtime-binding" \
    --output json >"$OUT_DIR/missing-binding.json" 2>"$OUT_DIR/missing-binding.err"
python3 - "$OUT_DIR/missing-binding.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["status"] == "refused"
assert result["failure_code"] == "YVEX_ERR_IO"
assert result["failure_where"] == "runtime.model"
assert result["runtime_generation_ready"] is False
assert "layers_executed" not in result
for key in (
    "benchmark_path", "benchmark_baseline_written", "benchmark_file_bytes",
    "benchmark_baseline_identity", "benchmark_baseline_compatible",
    "benchmark_cold_delta_seconds", "benchmark_chart_generated",
    "benchmark_chart_path", "benchmark_chart_identity", "benchmark_chart_file_bytes",
):
    assert key not in result
PY

printf 'stale-runtime-binding\n' >"$OUT_DIR/stale.yvex-runtime-binding"
expect_status 4 "$YVEX_BIN" graph attention execute \
    --target deepseek4-v4-flash --backend cpu \
    --runtime-binding "$OUT_DIR/stale.yvex-runtime-binding" --output json \
    >"$OUT_DIR/stale-binding.json" 2>"$OUT_DIR/stale-binding.err"
python3 - "$OUT_DIR/stale-binding.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["status"] == "refused"
assert result["failure_code"] == "YVEX_ERR_BOUNDS"
assert result["failure_where"] == "runtime.model"
assert result["runtime_generation_ready"] is False
PY

mkdir "$OUT_DIR/ambiguous-bindings"
printf one >"$OUT_DIR/ambiguous-bindings/one.yvex-runtime-binding"
printf two >"$OUT_DIR/ambiguous-bindings/two.yvex-runtime-binding"
expect_status 1 "$YVEX_BIN" graph attention describe \
    --target deepseek4-v4-flash \
    --runtime-binding-dir "$OUT_DIR/ambiguous-bindings" --output json \
    >"$OUT_DIR/ambiguous-binding.out" 2>"$OUT_DIR/ambiguous-binding.err"
test ! -s "$OUT_DIR/ambiguous-binding.out" || fail "ambiguous binding polluted JSON stdout"
contains "$OUT_DIR/ambiguous-binding.err" "runtime binding registry is ambiguous"

if rg -n '\b(system|popen|fork|exec[lvpe]*)[[:space:]]*\(' \
    src/cli/commands/graph.c >"$OUT_DIR/indirection-functions.out"; then
    fail "graph attention CLI gained process indirection"
fi
if rg -n 'tests/|build/tests|make (test|check)|deepseek_attention_reference' \
    src/cli/commands/graph.c >"$OUT_DIR/indirection-paths.out"; then
    fail "graph attention CLI references test or oracle ownership"
fi
if nm -a "$YVEX_BIN" 2>/dev/null | rg 'deepseek_attention_reference' \
    >"$OUT_DIR/oracle-linkage.out"; then
    fail "production yvex binary links the test-only attention oracle"
fi
python3 - src/cli/render/graph.c <<'PY'
import pathlib
import sys

source = " ".join(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").split())
rules = source.split("static const attention_presence_rule attention_presence_rules[]", 1)[1]
rules = rules.split("#undef ATTENTION_GROUP", 1)[0]
assert "ATTENTION_FIELDS_BENCHMARK_BASELINE" in rules
baseline_rule = rules.split("ATTENTION_FIELDS_BENCHMARK_BASELINE", 1)[1]
baseline_rule = baseline_rule.split("},", 1)[0]
assert "benchmark.baseline_identity" in baseline_rule
assert "ATTENTION_PRESENCE_TEXT" in baseline_rule
assert "delta_seconds" not in rules
PY

expect_status 5 "$YVEX_BIN" graph attention execute \
    --target qwen3-8b --backend cpu --scope quick --output json \
    >"$OUT_DIR/target-refusal.json" 2>"$OUT_DIR/target-refusal.err"
contains "$OUT_DIR/target-refusal.err" "unsupported attention target: qwen3-8b"
python3 - "$OUT_DIR/target-refusal.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["status"] == "refused"
assert result["target"] == "qwen3-8b"
assert result["failure_code"] == "YVEX_ERR_UNSUPPORTED"
assert result["failure_where"] == "runtime.attention"
assert result["operator_command_available"] is True
assert result["runtime_generation_ready"] is False
assert "family" not in result
assert "execution_class" not in result
assert "weights_class" not in result
assert "layers_executed" not in result
PY

if [ "${YVEX_ATTENTION_LIVE:-0}" = 1 ]; then
    MODELS_ROOT=${YVEX_ATTENTION_MODELS_ROOT:?attention live models root is required}
    ARTIFACT=${YVEX_ATTENTION_ARTIFACT:?attention live artifact is required}
    CPU_QUICK_ONLY=${YVEX_ATTENTION_CPU_QUICK_ONLY:-0}
    case "$CPU_QUICK_ONLY" in
        0|1) ;;
        *) fail "YVEX_ATTENTION_CPU_QUICK_ONLY must be 0 or 1" ;;
    esac
    BINDING_DIR="$OUT_DIR/runtime-bindings"
    mkdir "$BINDING_DIR"

    if YVEX_TEST_RUNTIME_BINDING_VALIDATE_FAILURE=1 \
        "$YVEX_BIN" graph attention prepare --target deepseek4-v4-flash \
            --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
            --runtime-binding-dir "$BINDING_DIR" --output json \
            >"$OUT_DIR/prepare-validation-failure.json" \
            2>"$OUT_DIR/prepare-validation-failure.err"; then
        fail "runtime-binding candidate validation failure returned success"
    fi
    if find "$BINDING_DIR" -mindepth 1 -print -quit | grep -q .; then
        fail "runtime-binding validation failure retained a candidate or temporary file"
    fi

    "$YVEX_BIN" graph attention prepare --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding-dir "$BINDING_DIR" --output json \
        >"$OUT_DIR/prepare.json" 2>"$OUT_DIR/prepare.err"
    BINDING=$(python3 - "$OUT_DIR/prepare.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["command"] == "graph attention prepare"
assert result["status"] == "complete"
assert len(result["runtime_binding_identity"]) == 64
assert len(result["artifact_identity"]) == 64
assert result["runtime_generation_ready"] is False
print(result["runtime_binding_path"])
PY
    )
    test -f "$BINDING" && test ! -L "$BINDING" || fail "prepared binding is not regular"
    case "$BINDING" in
        "$BINDING_DIR"/*) ;;
        *) fail "prepared binding escaped its owned directory" ;;
    esac

    run_quick_cpu() {
        "$YVEX_BIN" graph attention execute --target deepseek4-v4-flash \
            --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
            --runtime-binding "$BINDING" --backend cpu --probe canonical --scope quick \
            --phase prefill --mode eager --operation-scope envelope --tokens 2 --repeat 2 \
            --output json >"$OUT_DIR/quick-cpu.json" 2>"$OUT_DIR/quick-cpu.err"
        python3 - "$OUT_DIR/quick-cpu.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["status"] == "complete"
assert result["backend"] == "cpu"
assert result["phase"] == "prefill"
assert result["selected_mode"] == "eager"
assert result["operation_scope"] == "envelope"
assert result["layers_executed"] == 3
assert result["swa_layers_executed"] == 1
assert result["csa_layers_executed"] == 1
assert result["hca_layers_executed"] == 1
assert result["topk_selected"] == 512
assert result["hca_ratio"] == 128
assert result["bindings_executed"] > 0
assert result["repeat_count"] == 2
assert result["runtime_source_headers_read"] == 0
assert result["runtime_source_payload_bytes_read"] == 0
assert result["runtime_transform_plans_built"] == 0
assert result["runtime_quant_plans_built"] == 0
assert result["runtime_writer_plans_built"] == 0
assert result["artifact_hash_passes"] == 1
assert result["warm_artifact_hash_passes"] == 0
assert len(result["tensor_output_digest"]) == 64
assert len(result["state_delta_digest"]) == 64
assert len(result["execution_evidence_digest"]) == 64
assert len(result["execution_identity"]) == 64
assert result["persistent_kv_ready"] is False
assert result["transformer_ready"] is False
assert result["runtime_generation_ready"] is False
PY
    }

    if [ "$CPU_QUICK_ONLY" = 1 ]; then
        run_quick_cpu
        printf 'cli attention graph live: instrumented CPU production path ok\n'
    else

    "$YVEX_BIN" graph attention describe --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --output json \
        >"$OUT_DIR/describe.json" 2>"$OUT_DIR/describe.err"
    "$YVEX_BIN" graph attention plan --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding-dir "$BINDING_DIR" --backend cpu \
        --phase prefill --mode auto --operation-scope envelope --tokens 4 --output json \
        >"$OUT_DIR/plan.json" 2>"$OUT_DIR/plan.err"
    "$YVEX_BIN" graph attention plan --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --layer 0 \
        --phase prefill --mode auto --operation-scope envelope --tokens 4 --output json \
        >"$OUT_DIR/plan-layer-0.json" 2>"$OUT_DIR/plan-layer-0.err"
    python3 - "$OUT_DIR/prepare.json" "$OUT_DIR/describe.json" "$OUT_DIR/plan.json" \
        "$OUT_DIR/plan-layer-0.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    prepared = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    described = json.load(stream)
with open(sys.argv[3], encoding="utf-8") as stream:
    planned = json.load(stream)
with open(sys.argv[4], encoding="utf-8") as stream:
    planned_layer = json.load(stream)
identity = prepared["runtime_binding_identity"]
assert described["command"] == "graph attention describe"
assert planned["command"] == "graph attention plan"
assert described["runtime_binding_identity"] == identity
assert planned["runtime_binding_identity"] == identity
assert planned["backend"] == "cpu"
assert planned["phase"] == "prefill"
assert planned["requested_mode"] == "auto"
assert planned["operation_scope"] == "envelope"
assert len(planned["runtime_model_identity"]) == 64
assert len(planned["execution_descriptor_identity"]) == 64
assert len(planned["execution_identity"]) == 64
assert planned["execution_dispatch_count"] == 0
assert planned["layers_executed"] == 0
assert planned["runtime_generation_ready"] is False
assert planned_layer["status"] == "complete"
assert planned_layer["execution_descriptor_identity"] != planned["execution_descriptor_identity"]
assert planned_layer["execution_identity"] != planned["execution_identity"]
PY

    for action in inspect validate; do
        "$YVEX_BIN" graph attention state "$action" --target deepseek4-v4-flash \
            --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
            --runtime-binding "$BINDING" --output json \
            >"$OUT_DIR/state-$action.json" 2>"$OUT_DIR/state-$action.err"
    done
    python3 - "$OUT_DIR/state-inspect.json" "$OUT_DIR/state-validate.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    inspected = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    validated = json.load(stream)
for result in (inspected, validated):
    assert result["status"] == "complete"
    assert result["backend"] == "cpu"
    assert len(result["state_layout_identity"]) == 64
    assert result["state_sealed"] is True
    assert result["state_transaction_active"] is False
    assert result["execution_dispatch_count"] == 0
assert inspected["state_validation_passed"] is False
assert validated["state_validation_passed"] is True
PY

    "$YVEX_BIN" graph attention capabilities --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --output json \
        >"$OUT_DIR/capabilities.json" 2>"$OUT_DIR/capabilities.err"
    python3 - "$OUT_DIR/capabilities.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["command"] == "graph attention capabilities"
assert result["artifact_hash_passes"] == 1
assert result["runtime_model_builds"] == 1
assert result["runtime_descriptor_builds"] == 1
assert result["semantic_graph_builds"] == 1
assert result["executable_graph_builds"] == 1
assert result["attention_semantics_ready"] is True
assert result["attention_core_ready"] is True
assert result["attention_envelope_ready"] is True
assert result["cpu_prefill_eager_ready"] is True
assert result["cpu_decode_eager_ready"] is True
assert result["cuda_prefill_eager_ready"] is False
assert result["cuda_prefill_full_graph_ready"] is False
assert result["attention_workspace_ready"] is True
assert result["mixed_attention_ready"] is False
assert result["speculative_attention_ready"] is False
assert result["runtime_generation_ready"] is False
PY

    for action in capabilities "residency inspect"; do
        # shellcheck disable=SC2086
        "$YVEX_BIN" graph attention $action --target deepseek4-v4-flash \
            --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
            --runtime-binding "$BINDING" --backend cuda --output json \
            >"$OUT_DIR/cuda-${action%% *}.json" 2>"$OUT_DIR/cuda-${action%% *}.err"
    done
    python3 - "$OUT_DIR/cuda-capabilities.json" "$OUT_DIR/cuda-residency.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as stream: result = json.load(stream)
    assert result["status"] == "complete"
    assert result["backend"] == "cuda"
    assert result["cuda_device"] == "NVIDIA GB10"
    assert result["compute_capability_major"] == 12
    assert result["compute_capability_minor"] == 1
    assert result["bindings_total"] == 634
    assert result["resident_binding_count"] == 806
    assert result["device_resident_bytes"] > 0
    assert result["workspace_bytes"] > 0
    assert result["pinned_host_residency"] is True
    assert result["pinned_host_bytes"] > 0
    assert result["attention_semantics_ready"] is True
    assert result["attention_envelope_ready"] is True
    assert result["cuda_prefill_eager_ready"] is True
    assert result["cuda_decode_eager_ready"] is True
    assert result["cuda_prefill_piecewise_graph_ready"] is True
    assert result["cuda_decode_piecewise_graph_ready"] is True
    assert result["cuda_prefill_full_graph_ready"] is True
    assert result["cuda_decode_full_graph_ready"] is True
    assert result["attention_workspace_ready"] is True
    assert result["mixed_attention_ready"] is False
    assert result["speculative_attention_ready"] is False
PY

    "$YVEX_BIN" graph attention trace --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --probe canonical --scope quick \
        --trace-level stages --output json \
        >"$OUT_DIR/trace-stages.json" 2>"$OUT_DIR/trace-stages.err"
    "$YVEX_BIN" graph attention trace --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --probe canonical --scope quick \
        --trace-level summary --output json \
        >"$OUT_DIR/trace-summary.json" 2>"$OUT_DIR/trace-summary.err"
    "$YVEX_BIN" graph attention trace --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --probe canonical --scope quick \
        --trace-level full --output json \
        >"$OUT_DIR/trace-full.json" 2>"$OUT_DIR/trace-full.err"
    python3 - "$OUT_DIR/trace-summary.json" "$OUT_DIR/trace-stages.json" \
        "$OUT_DIR/trace-full.json" <<'PY'
import json
import sys

results = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as stream:
        results.append(json.load(stream))
summary, stages, full = results
for result, policy in zip(results, ("summary", "stages", "full")):
    assert result["status"] == "complete"
    assert result["trace_policy"] == policy
    assert result["trace_stage_count"] > result["layers_executed"]
    assert result["trace_value_count"] > 0
    assert len(result["execution_evidence_digest"]) == 64
    assert result["execution_dispatch_count"] == 1
assert summary["trace_stage_count"] < stages["trace_stage_count"]
assert stages["trace_stage_count"] < full["trace_stage_count"]
assert summary["trace_value_count"] < stages["trace_value_count"]
assert stages["trace_value_count"] < full["trace_value_count"]
assert len({result["execution_evidence_digest"] for result in results}) == 3
assert len({result["tensor_output_digest"] for result in results}) == 1
assert len({result["state_delta_digest"] for result in results}) == 1
PY

    cp "$BINDING" "$OUT_DIR/stale-valid.yvex-runtime-binding"
    python3 - "$OUT_DIR/stale-valid.yvex-runtime-binding" <<'PY'
import sys

path = sys.argv[1]
with open(path, "r+b") as stream:
    stream.seek(96)
    value = stream.read(1)
    assert value
    stream.seek(96)
    stream.write(bytes([value[0] ^ 0x5A]))
PY
    expect_nonzero "$YVEX_BIN" graph attention execute \
        --target deepseek4-v4-flash --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$OUT_DIR/stale-valid.yvex-runtime-binding" \
        --backend cpu --output json \
        >"$OUT_DIR/stale-valid.json" 2>"$OUT_DIR/stale-valid.err"
    python3 - "$OUT_DIR/stale-valid.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["status"] == "refused"
assert result["failure_where"] == "runtime.model"
assert result["runtime_generation_ready"] is False
PY

    run_quick_cpu

    "$YVEX_BIN" graph attention state exercise --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --class csa \
        --history-tokens 2052 --tokens 2 --output json \
        >"$OUT_DIR/state-csa-513.json" 2>"$OUT_DIR/state-csa-513.err"
    "$YVEX_BIN" graph attention state exercise --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --class hca \
        --position 127 --tokens 2 --output json \
        >"$OUT_DIR/state-hca-128.json" 2>"$OUT_DIR/state-hca-128.err"
    python3 - "$OUT_DIR/state-csa-513.json" "$OUT_DIR/state-hca-128.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    csa = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    hca = json.load(stream)
for result in (csa, hca):
    assert result["command"] == "graph attention state exercise"
    assert result["status"] == "complete"
    assert result["backend"] == "cpu"
    assert result["selected_mode"] == "eager"
    assert result["layers_executed"] == 1
    assert result["comparison_passed"] is True
    assert result["comparison_maximum_absolute_error"] == 0
    assert result["comparison_maximum_relative_error"] == 0
    assert result["comparison_rmse"] == 0
    assert result["bitwise_equality_observed"] is True
    assert len(result["tensor_output_digest"]) == 64
    assert len(result["state_delta_digest"]) == 64
    assert result["runtime_generation_ready"] is False
assert csa["swa_layers_executed"] == 0
assert csa["csa_layers_executed"] == 1
assert csa["hca_layers_executed"] == 0
assert csa["topk_selected"] == 512
assert hca["swa_layers_executed"] == 0
assert hca["csa_layers_executed"] == 0
assert hca["hca_layers_executed"] == 1
assert hca["hca_ratio"] == 128
PY

    expect_status 3 "$YVEX_BIN" graph attention execute --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --probe canonical --scope quick --output json \
        >/dev/full 2>"$OUT_DIR/render-failure.err"
    contains "$OUT_DIR/render-failure.err" "attention result rendering failed"

    "$YVEX_BIN" graph attention execute --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --probe canonical --scope full \
        --phase decode --mode eager --operation-scope release-attention-set --output audit \
        >"$OUT_DIR/full-cpu-first.out" 2>"$OUT_DIR/full-cpu-first.err"
    "$YVEX_BIN" graph attention execute --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cpu --probe canonical --scope full \
        --phase decode --mode eager --operation-scope release-attention-set --output audit \
        >"$OUT_DIR/full-cpu-second.out" 2>"$OUT_DIR/full-cpu-second.err"
    cmp "$OUT_DIR/full-cpu-first.out" "$OUT_DIR/full-cpu-second.out" >/dev/null ||
        fail "full CPU operator output is not deterministic"
    contains "$OUT_DIR/full-cpu-first.out" "layers_executed: 43"
    contains "$OUT_DIR/full-cpu-first.out" "bindings_executed: 634"
    contains "$OUT_DIR/full-cpu-first.out" "swa_layers_executed: 2"
    contains "$OUT_DIR/full-cpu-first.out" "csa_layers_executed: 21"
    contains "$OUT_DIR/full-cpu-first.out" "hca_layers_executed: 20"

    "$YVEX_BIN" graph attention execute --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cuda --probe canonical --scope quick \
        --phase decode --mode eager --operation-scope core --output audit \
        >"$OUT_DIR/quick-cuda.out" 2>"$OUT_DIR/quick-cuda.err"
    contains "$OUT_DIR/quick-cuda.out" "status: complete"
    contains "$OUT_DIR/quick-cuda.out" "backend: cuda"
    contains "$OUT_DIR/quick-cuda.out" "layers_executed: 3"
    contains "$OUT_DIR/quick-cuda.out" "swa_layers_executed: 1"
    contains "$OUT_DIR/quick-cuda.out" "csa_layers_executed: 1"
    contains "$OUT_DIR/quick-cuda.out" "hca_layers_executed: 1"
    contains "$OUT_DIR/quick-cuda.out" "topk_selected: 512"
    contains "$OUT_DIR/quick-cuda.out" "hca_ratio: 128"
    contains "$OUT_DIR/quick-cuda.out" "attention_cuda_execution_ready: true"

    for mode in eager piecewise full; do
        trace_level=none
        if [ "$mode" = full ]; then trace_level=full; fi
        "$YVEX_BIN" graph attention execute --target deepseek4-v4-flash \
            --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
            --runtime-binding "$BINDING" --backend cuda --probe canonical --scope quick \
            --phase prefill --mode "$mode" --operation-scope envelope --tokens 4 --repeat 2 \
            --require-mode --trace-level "$trace_level" --progress off --output json \
            >"$OUT_DIR/prefill-cuda-$mode.json" 2>"$OUT_DIR/prefill-cuda-$mode.err"
    done
    python3 - "$OUT_DIR/prefill-cuda-eager.json" \
        "$OUT_DIR/prefill-cuda-piecewise.json" "$OUT_DIR/prefill-cuda-full.json" <<'PY'
import json
import sys

results = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as stream:
        results.append(json.load(stream))
for result, mode in zip(results, ("eager", "piecewise", "full")):
    assert result["command"] == "graph attention execute"
    assert result["status"] == "complete"
    assert result["backend"] == "cuda"
    assert result["phase"] == "prefill"
    assert result["requested_mode"] == mode
    assert result["selected_mode"] == mode
    assert result["operation_scope"] == "envelope"
    assert result["repeat_count"] == 2
    assert result["execution_dispatch_count"] == 2
    assert result["layers_executed"] == 3
    assert result["swa_layers_executed"] == 1
    assert result["csa_layers_executed"] == 1
    assert result["hca_layers_executed"] == 1
    assert result["bindings_executed"] > 0
    assert result["trace_policy"] == ("full" if mode == "full" else "none")
    assert result["h2d_bytes"] > 0
    assert result["d2h_bytes"] > 0
    assert result["warm_h2d_bytes"] > 0
    assert result["warm_d2h_bytes"] > 0
    assert len(result["tensor_output_digest"]) == 64
    assert len(result["state_delta_digest"]) == 64
    assert result["artifact_hash_passes"] == 1
    assert result["warm_artifact_hash_passes"] == 0
    assert result["runtime_source_headers_read"] == 0
    assert result["runtime_source_payload_bytes_read"] == 0
    assert result["runtime_transform_plans_built"] == 0
    assert result["runtime_quant_plans_built"] == 0
    assert result["runtime_writer_plans_built"] == 0
    assert result["warm_weight_artifact_reads"] == 0
    assert result["warm_weight_upload_bytes"] == 0
    assert result["warm_host_allocations"] == 0
    assert result["warm_device_allocations"] == 0
    assert result["warm_device_frees"] == 0
    assert result["persistent_kv_ready"] is False
    assert result["transformer_ready"] is False
    assert result["runtime_generation_ready"] is False
eager, piecewise, full = results
assert eager["cuda_graph_count"] == 0
assert eager["cuda_graph_capture_count"] == 0
assert eager["cuda_graph_replay_count"] == 0
assert eager["kernel_launches"] > 0
for result in (piecewise, full):
    assert result["cuda_graph_count"] > 0
    assert result["cuda_graph_capture_count"] == result["cuda_graph_count"]
    assert result["cuda_graph_replay_count"] == 2 * result["cuda_graph_count"]
    assert result["cuda_graph_kernel_node_count"] > 0
assert piecewise["cuda_graph_piece_count"] == piecewise["cuda_graph_count"]
assert piecewise["cuda_graph_count"] > full["cuda_graph_count"]
assert full["d2h_bytes"] > piecewise["d2h_bytes"]
assert full["tensor_output_digest"] == piecewise["tensor_output_digest"]
assert full["state_delta_digest"] == piecewise["state_delta_digest"]
PY

    "$YVEX_BIN" graph attention capture --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --mode full --probe canonical --scope quick \
        --output json >"$OUT_DIR/capture.json" 2>"$OUT_DIR/capture.err"
    "$YVEX_BIN" graph attention capture --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --mode piecewise --probe canonical --scope quick \
        --output json >"$OUT_DIR/capture-piecewise.json" \
        2>"$OUT_DIR/capture-piecewise.err"
    "$YVEX_BIN" graph attention replay --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --mode full --probe canonical --scope quick \
        --repeat 2 --output json >"$OUT_DIR/replay.json" 2>"$OUT_DIR/replay.err"
    "$YVEX_BIN" graph attention cuda-graph warmup --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --mode full --probe canonical --scope quick \
        --warmup 2 --output json >"$OUT_DIR/graph-warmup.json" \
        2>"$OUT_DIR/graph-warmup.err"
    python3 - "$OUT_DIR/capture.json" "$OUT_DIR/capture-piecewise.json" \
        "$OUT_DIR/replay.json" "$OUT_DIR/graph-warmup.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    captured = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    piecewise = json.load(stream)
with open(sys.argv[3], encoding="utf-8") as stream:
    replayed = json.load(stream)
with open(sys.argv[4], encoding="utf-8") as stream:
    warmed = json.load(stream)
for result in (captured, piecewise, replayed, warmed):
    assert result["status"] == "complete"
    assert result["backend"] == "cuda"
    assert result["cuda_graph_count"] > 0
    assert result["cuda_graph_capture_count"] > 0
    assert result["cuda_graph_replay_count"] > 0
    assert result["cuda_graph_kernel_node_count"] > 0
assert captured["selected_mode"] == "full"
assert captured["layers_executed"] == 3
assert captured["swa_layers_executed"] == 1
assert captured["csa_layers_executed"] == 1
assert captured["hca_layers_executed"] == 1
assert captured["topk_selected"] == 512
assert captured["hca_ratio"] == 128
assert captured["cuda_graph_count"] == 3
assert captured["cuda_graph_capture_count"] == 3
assert captured["cuda_graph_replay_count"] == 3
assert captured["execution_dispatch_count"] == 1
assert piecewise["selected_mode"] == "piecewise"
assert piecewise["layers_executed"] == 3
assert piecewise["swa_layers_executed"] == 1
assert piecewise["csa_layers_executed"] == 1
assert piecewise["hca_layers_executed"] == 1
assert piecewise["topk_selected"] == 512
assert piecewise["hca_ratio"] == 128
assert piecewise["cuda_graph_count"] == 8
assert piecewise["cuda_graph_piece_count"] == 8
assert piecewise["cuda_graph_capture_count"] == 8
assert piecewise["cuda_graph_replay_count"] == 8
assert piecewise["execution_dispatch_count"] == 1
assert replayed["selected_mode"] == "full"
assert replayed["warmup_count"] == 1
assert replayed["repeat_count"] == 2
assert replayed["execution_dispatch_count"] == 3
assert replayed["cuda_graph_count"] == 3
assert replayed["cuda_graph_capture_count"] == 3
assert replayed["cuda_graph_replay_count"] == 9
assert warmed["selected_mode"] == "full"
assert warmed["warmup_count"] == 2
assert warmed["execution_dispatch_count"] == 3
assert warmed["cuda_graph_count"] == 3
assert warmed["cuda_graph_capture_count"] == 3
assert warmed["cuda_graph_replay_count"] == 9
PY

    "$YVEX_BIN" graph attention cuda-graph list --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --output json \
        >"$OUT_DIR/graph-list.json" 2>"$OUT_DIR/graph-list.err"
    "$YVEX_BIN" graph attention cuda-graph inspect \
        --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --output json \
        >"$OUT_DIR/graph-inspect.json" 2>"$OUT_DIR/graph-inspect.err"
    "$YVEX_BIN" graph attention cuda-graph update --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --output json \
        >"$OUT_DIR/graph-update.json" 2>"$OUT_DIR/graph-update.err"
    "$YVEX_BIN" graph attention cuda-graph invalidate --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --output json \
        >"$OUT_DIR/graph-invalidate.json" 2>"$OUT_DIR/graph-invalidate.err"
    "$YVEX_BIN" graph attention cuda-graph release --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --output json \
        >"$OUT_DIR/graph-release.json" 2>"$OUT_DIR/graph-release.err"
    python3 - "$OUT_DIR/graph-list.json" "$OUT_DIR/graph-inspect.json" \
        "$OUT_DIR/graph-update.json" "$OUT_DIR/graph-invalidate.json" \
        "$OUT_DIR/graph-release.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream: listed = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream: inspected = json.load(stream)
with open(sys.argv[3], encoding="utf-8") as stream: updated = json.load(stream)
with open(sys.argv[4], encoding="utf-8") as stream: invalidated = json.load(stream)
with open(sys.argv[5], encoding="utf-8") as stream: released = json.load(stream)
for result in (listed, updated, invalidated, released):
    assert result["status"] == "complete"
    assert result["cuda_graph_registry_scope"] == "ephemeral-process-session"
assert listed["execution_dispatch_count"] == 1
assert listed["cuda_graph_count"] == 3
assert listed["cuda_graph_registry_count"] == 3
assert inspected["status"] == "complete"
assert inspected["execution_dispatch_count"] == 1
assert inspected["cuda_graph_registry_count"] == 3
assert inspected["cuda_graph_entry_compatibility_identity"]
assert updated["warmup_count"] == 2
assert updated["execution_dispatch_count"] == 3
assert updated["cuda_graph_update_count"] == 3
assert updated["cuda_graph_update_pending_count"] == 0
assert updated["cuda_graph_registry_affected_count"] == 3
assert updated["cuda_graph_last_update_elapsed_ns"] > 0
assert invalidated["execution_dispatch_count"] == 1
assert invalidated["cuda_graph_count"] == 0
assert invalidated["cuda_graph_registry_count"] == 3
assert invalidated["cuda_graph_registry_affected_count"] == 3
assert invalidated["cuda_graph_invalidation_count"] == 3
assert released["execution_dispatch_count"] == 1
assert released["cuda_graph_count"] == 0
assert released["cuda_graph_registry_count"] == 0
assert released["cuda_graph_registry_affected_count"] == 3
PY

    "$YVEX_BIN" graph attention execute --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cuda --probe canonical --scope full \
        --phase decode --mode eager --operation-scope release-attention-set --output audit \
        >"$OUT_DIR/full-cuda.out" 2>"$OUT_DIR/full-cuda.err"
    contains "$OUT_DIR/full-cuda.out" "layers_executed: 43"
    contains "$OUT_DIR/full-cuda.out" "bindings_executed: 634"
    contains "$OUT_DIR/full-cuda.out" "kernel_launches:"

    YVEX_TEST_RUNTIME_AUTO_DISABLE_FULL=1 \
        "$YVEX_BIN" graph attention execute --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cuda --probe canonical --scope full \
        --phase decode --mode auto --operation-scope release-attention-set \
        --output json >"$OUT_DIR/full-cuda-piecewise.json" \
        2>"$OUT_DIR/full-cuda-piecewise.err"
    "$YVEX_BIN" graph attention execute --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cuda --probe canonical --scope full \
        --phase decode --mode full --operation-scope release-attention-set \
        --output json >"$OUT_DIR/full-cuda-full.json" 2>"$OUT_DIR/full-cuda-full.err"
    python3 - "$OUT_DIR/full-cuda-piecewise.json" "$OUT_DIR/full-cuda-full.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    piecewise = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    full = json.load(stream)
for result, mode, graph_count in (
    (piecewise, "piecewise", 127),
    (full, "full", 43),
):
    assert result["status"] == "complete"
    assert result["backend"] == "cuda"
    assert result["selected_mode"] == mode
    assert result["layers_executed"] == 43
    assert result["bindings_executed"] == 634
    assert result["swa_layers_executed"] == 2
    assert result["csa_layers_executed"] == 21
    assert result["hca_layers_executed"] == 20
    assert result["cuda_graph_count"] == graph_count
    assert result["cuda_graph_capture_count"] > 0
    assert result["cuda_graph_replay_count"] > 0
    assert result["cuda_graph_kernel_node_count"] > 0
    assert len(result["tensor_output_digest"]) == 64
    assert len(result["state_delta_digest"]) == 64
    assert result["runtime_generation_ready"] is False
assert piecewise["cuda_graph_piece_count"] == 127
assert piecewise["requested_mode"] == "auto"
assert piecewise["selection_reason"] == (
    "auto-selected admitted CUDA piecewise graph mode after full refusal"
)
PY

    for mode in eager piecewise full; do
        "$YVEX_BIN" graph attention compare --target deepseek4-v4-flash \
            --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
            --runtime-binding "$BINDING" --probe canonical --scope full \
            --phase decode --mode "$mode" --operation-scope release-attention-set \
            --output json >"$OUT_DIR/full-compare-$mode.json" \
            2>"$OUT_DIR/full-compare-$mode.err"
    done
    python3 - "$OUT_DIR/full-compare-eager.json" \
        "$OUT_DIR/full-compare-piecewise.json" "$OUT_DIR/full-compare-full.json" <<'PY'
import json
import sys

results = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as stream:
        results.append(json.load(stream))
for result, mode in zip(results, ("eager", "piecewise", "full")):
    assert result["status"] == "complete"
    assert result["backend"] == "compare"
    assert result["selected_mode"] == mode
    assert result["layers_executed"] == 43
    assert result["bindings_executed"] == 634
    assert result["swa_layers_executed"] == 2
    assert result["csa_layers_executed"] == 21
    assert result["hca_layers_executed"] == 20
    assert result["comparison_passed"] is True
    assert result["comparison_output_values"] == 43 * 16384
    assert result["comparison_state_values"] > 0
    assert result["comparison_values"] == (
        result["comparison_output_values"] + result["comparison_state_values"]
    )
    assert result["comparison_finite_values"] == result["comparison_values"]
    assert result["comparison_nonfinite_values"] == 0
    assert result["comparison_maximum_absolute_error"] == 0
    assert result["comparison_maximum_relative_error"] == 0
    assert result["comparison_rmse"] == 0
    assert len(result["comparison_contract_identity"]) == 64
    assert result["cpu_output_digest"] == result["cuda_output_digest"]
    assert result["cpu_state_delta_digest"] == result["cuda_state_delta_digest"]
    assert result["output_digest_equal"] is True
    assert result["state_delta_digest_equal"] is True
    assert result["bitwise_equality_observed"] is True
    assert result["artifact_identity_verified"] is True
    assert result["artifact_bytes_hashed"] == 102408545440
    assert result["physical_payload_compatible"] is True
    assert result["artifact_rebuild_required"] is False
    assert result["materialization_rebuild_required"] is False
    assert result["operator_command_available"] is True
    assert result["end_user_generation_available"] is False
    assert result["runtime_generation_ready"] is False
    assert result["runtime_source_headers_read"] == 0
    assert result["runtime_source_payload_bytes_read"] == 0
    assert result["runtime_transform_plans_built"] == 0
    assert result["runtime_quant_plans_built"] == 0
    assert result["runtime_writer_plans_built"] == 0
    assert result["warm_host_allocations"] == 0
    assert result["warm_device_allocations"] == 0
    assert result["warm_device_frees"] == 0
    assert result["warm_weight_artifact_reads"] == 0
    assert result["warm_weight_upload_bytes"] == 0
    if mode == "piecewise":
        assert result["cuda_graph_capture_count"] == 127
        assert result["cuda_graph_piece_count"] == 127
        assert result["cuda_graph_replay_count"] == 127
    elif mode == "full":
        assert result["cuda_graph_capture_count"] == 43
        assert result["cuda_graph_replay_count"] == 43
PY

    "$YVEX_BIN" graph attention profile --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cuda --probe canonical --scope full \
        --phase decode --mode full --operation-scope release-attention-set \
        --warmup 1 --repeat 2 --require-mode --progress off --output json \
        >"$OUT_DIR/full-profile.json" 2>"$OUT_DIR/full-profile.err"
    python3 - "$OUT_DIR/full-profile.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["command"] == "graph attention profile"
assert result["status"] == "complete"
assert result["backend"] == "cuda"
assert result["phase"] == "decode"
assert result["requested_mode"] == "full"
assert result["selected_mode"] == "full"
assert result["operation_scope"] == "release-attention-set"
assert result["warmup_count"] == 1
assert result["repeat_count"] == 2
assert result["benchmark_sample_count"] == 2
assert result["execution_dispatch_count"] == 3
assert result["layers_executed"] == 43
assert result["bindings_executed"] == 634
assert result["swa_layers_executed"] == 2
assert result["csa_layers_executed"] == 21
assert result["hca_layers_executed"] == 20
assert result["cuda_graph_count"] == 43
assert result["cuda_graph_capture_count"] == 43
assert result["cuda_graph_replay_count"] == 129
assert result["cuda_graph_kernel_node_count"] > 0
assert result["benchmark_minimum_seconds"] > 0
assert result["benchmark_p50_seconds"] > 0
assert result["benchmark_p90_seconds"] > 0
assert result["benchmark_p95_seconds"] > 0
assert result["benchmark_p99_seconds"] > 0
assert result["benchmark_maximum_seconds"] > 0
assert result["benchmark_mean_seconds"] > 0
assert result["benchmark_standard_deviation_seconds"] >= 0
assert len(result["benchmark_identity"]) == 64
assert result["attention_profile_ready"] is True
assert result["artifact_hash_passes"] == 1
assert result["warm_artifact_hash_passes"] == 0
assert result["runtime_source_headers_read"] == 0
assert result["runtime_source_payload_bytes_read"] == 0
assert result["runtime_transform_plans_built"] == 0
assert result["runtime_quant_plans_built"] == 0
assert result["runtime_writer_plans_built"] == 0
assert result["warm_weight_artifact_reads"] == 0
assert result["warm_weight_upload_bytes"] == 0
assert result["warm_host_allocations"] == 0
assert result["warm_device_allocations"] == 0
assert result["warm_device_frees"] == 0
assert result["persistent_kv_ready"] is True
assert result["transformer_ready"] is False
assert result["runtime_generation_ready"] is False
PY

    "$YVEX_BIN" graph attention qualify --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cuda --probe canonical --scope full \
        --phase decode --mode full --operation-scope release-attention-set \
        --require-mode --progress off --output json \
        >"$OUT_DIR/qualify.json" 2>"$OUT_DIR/qualify.err"
    python3 - "$OUT_DIR/qualify.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["command"] == "graph attention qualify"
assert result["status"] == "complete"
assert result["software_contract_status"] == "pass"
assert result["numerical_conformance_status"] == "pass"
assert result["runtime_qualification_status"] == "pass"
assert result["component_benchmark_status"] == "available"
assert result["warmup_count"] == 1
assert result["execution_dispatch_count"] == 2
assert result["layers_executed"] == 43
assert result["bindings_executed"] == 634
assert result["cuda_graph_count"] == 43
assert result["cuda_graph_capture_count"] == 43
assert result["cuda_graph_replay_count"] == 86
assert result["warm_host_allocations"] == 0
assert result["warm_device_allocations"] == 0
assert result["warm_device_frees"] == 0
assert result["warm_weight_artifact_reads"] == 0
assert result["warm_weight_upload_bytes"] == 0
assert len(result["qualification_identity"]) == 64
assert len(result["quality_matrix_identity"]) == 64
assert result["model_behavior_evaluation_available"] is False
assert result["model_quality_evaluation_available"] is False
assert result["agent_runtime_available"] is False
assert result["agent_evaluation_available"] is False
assert result["release_qualification_available"] is False
assert result["persistent_kv_ready"] is True
assert result["runtime_generation_ready"] is False
PY

    CONTROL_ARTIFACT=$(printf '%s\001%s' "$OUT_DIR/missing-control" ".gguf")
    expect_status 3 "$YVEX_BIN" graph attention execute \
        --target deepseek4-v4-flash --models-root "$MODELS_ROOT" \
        --artifact "$CONTROL_ARTIFACT" --runtime-binding "$BINDING" \
        --backend cpu --scope quick --output json \
        >"$OUT_DIR/control-refusal.json" 2>"$OUT_DIR/control-refusal.err"
    python3 - "$OUT_DIR/control-refusal.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["status"] == "refused"
assert "\x01" in result["artifact_path"]
assert result["failure_code"] == "YVEX_ERR_IO"
assert result["failure_where"]
assert "layers_executed" not in result
PY

    "$YVEX_BIN" graph attention benchmark --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cuda --probe canonical --scope quick \
        --phase decode --mode piecewise --operation-scope envelope \
        --position 384 --history-tokens 384 --warmup 0 --repeat 5 \
        --baseline "$OUT_DIR/attention.yvex-benchmark" --write-baseline \
        --progress off --output json \
        >"$OUT_DIR/benchmark-write.json" 2>"$OUT_DIR/benchmark-write.err"
    "$YVEX_BIN" graph attention benchmark --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cuda --probe canonical --scope quick \
        --phase decode --mode piecewise --operation-scope envelope \
        --position 384 --history-tokens 384 --warmup 0 --repeat 5 \
        --baseline "$OUT_DIR/attention.yvex-benchmark" --progress off --output json \
        >"$OUT_DIR/benchmark-compare.json" 2>"$OUT_DIR/benchmark-compare.err"
    "$YVEX_BIN" graph attention benchmark --target deepseek4-v4-flash \
        --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
        --runtime-binding "$BINDING" --backend cuda --probe canonical --scope quick \
        --phase decode --mode piecewise --operation-scope envelope \
        --position 384 --history-tokens 384 --warmup 0 --repeat 1 \
        --baseline "$OUT_DIR/attention-short.yvex-benchmark" --write-baseline \
        --chart "$OUT_DIR/attention.svg" --progress off --output json \
        >"$OUT_DIR/benchmark-chart.json" 2>"$OUT_DIR/benchmark-chart.err"
    "$YVEX_BIN" graph attention benchmark compare \
        --baseline "$OUT_DIR/attention.yvex-benchmark" \
        --current "$OUT_DIR/attention.yvex-benchmark" \
        --max-regression-bps 0 \
        --chart "$OUT_DIR/file-comparison.svg" \
        --output json \
        >"$OUT_DIR/benchmark-file-compare.json" \
        2>"$OUT_DIR/benchmark-file-compare.err"
    expect_status 1 "$YVEX_BIN" graph attention benchmark compare \
        --baseline "$OUT_DIR/attention.yvex-benchmark" \
        --current "$OUT_DIR/attention-short.yvex-benchmark" --output json \
        >"$OUT_DIR/benchmark-file-incompatible.json" \
        2>"$OUT_DIR/benchmark-file-incompatible.err"
    python3 - "$OUT_DIR/benchmark-write.json" "$OUT_DIR/benchmark-compare.json" \
        "$OUT_DIR/benchmark-chart.json" "$OUT_DIR/attention.yvex-benchmark" \
        "$OUT_DIR/attention.svg" "$OUT_DIR/benchmark-file-compare.json" \
        "$OUT_DIR/file-comparison.svg" "$OUT_DIR/benchmark-file-incompatible.json" <<'PY'
import json
import hashlib
import pathlib
import sys

documents = []
for path in sys.argv[1:4]:
    with open(path, encoding="utf-8") as stream:
        documents.append(json.load(stream))
written, compared, charted = documents
baseline, chart = map(pathlib.Path, sys.argv[4:6])
short_baseline = chart.with_name("attention-short.yvex-benchmark")
with open(sys.argv[6], encoding="utf-8") as stream:
    file_compared = json.load(stream)
file_comparison_chart = pathlib.Path(sys.argv[7])
with open(sys.argv[8], encoding="utf-8") as stream:
    file_incompatible = json.load(stream)
for result in documents:
    assert result["status"] == "complete"
    assert result["selected_mode"] == "piecewise"
    assert result["warmup_count"] == 0
    assert result["layers_executed"] == 3
    assert result["swa_layers_executed"] == 1
    assert result["csa_layers_executed"] == 1
    assert result["hca_layers_executed"] == 1
    assert result["topk_selected"] == 512
    assert result["hca_ratio"] == 128
    assert result["warm_host_allocations"] == 0
    assert result["warm_device_allocations"] == 0
    assert result["warm_weight_artifact_reads"] == 0
    assert result["warm_weight_upload_bytes"] == 0
    assert result["state_allocated_bytes"] > 0
    assert result["workspace_bytes"] > 0
    assert result["benchmark_current_source_state"] in ("clean", "dirty")
    assert len(result["benchmark_identity"]) == 64
    assert result["benchmark_scope"] == "attention_component"
    assert result["correctness_status"] == "pass"
    assert result["structural_runtime_status"] == "pass"
    assert result["performance_status"] == "measured"
    assert result["benchmark_correctness_precondition_passed"] is True
    assert result["benchmark_runtime_precondition_passed"] is True
    assert result["benchmark_p95_seconds"] > 0
assert written["benchmark_sample_count"] == 5
assert written["repeat_count"] == 5
assert written["execution_dispatch_count"] == 6
assert written["cuda_graph_capture_count"] == 8
assert written["cuda_graph_replay_count"] == 48
assert written["benchmark_baseline_written"] is True
assert baseline.is_file() and baseline.stat().st_size == written["benchmark_file_bytes"]
for key in (
    "benchmark_baseline_identity", "benchmark_baseline_compatible",
    "benchmark_cold_delta_seconds", "benchmark_chart_generated",
    "benchmark_chart_path", "benchmark_chart_identity", "benchmark_chart_file_bytes",
):
    assert key not in written
assert compared["benchmark_sample_count"] == 5
assert compared["benchmark_baseline_compatible"] is True
assert len(compared["benchmark_baseline_identity"]) == 64
assert compared["benchmark_path"] == str(baseline)
for key in (
    "benchmark_cold_delta_seconds", "benchmark_minimum_delta_seconds",
    "benchmark_p50_delta_seconds", "benchmark_p90_delta_seconds",
    "benchmark_p95_delta_seconds", "benchmark_p99_delta_seconds",
    "benchmark_maximum_delta_seconds",
    "benchmark_mean_delta_seconds",
):
    assert key in compared and isinstance(compared[key], (int, float))
for key in (
    "benchmark_baseline_written", "benchmark_file_bytes", "benchmark_chart_generated",
    "benchmark_chart_path", "benchmark_chart_identity", "benchmark_chart_file_bytes",
):
    assert key not in compared
assert charted["benchmark_sample_count"] == 1
assert charted["repeat_count"] == 1
assert charted["benchmark_chart_generated"] is True
assert len(charted["benchmark_chart_identity"]) == 64
assert chart.is_file() and chart.stat().st_size == charted["benchmark_chart_file_bytes"]
assert chart.read_text(encoding="utf-8").startswith("<svg")
assert hashlib.sha256(chart.read_bytes()).hexdigest() == charted["benchmark_chart_identity"]
assert charted["benchmark_path"] == str(short_baseline)
assert charted["benchmark_baseline_written"] is True
assert short_baseline.is_file()
assert short_baseline.stat().st_size == charted["benchmark_file_bytes"]
for key in (
    "benchmark_baseline_identity", "benchmark_baseline_compatible",
    "benchmark_cold_delta_seconds",
):
    assert key not in charted
fields = {}
for line in baseline.read_text(encoding="utf-8").splitlines():
    if "\t" in line:
        key, value = line.split("\t", 1)
        fields[key] = value
assert int(fields["state_bytes"], 16) == written["state_allocated_bytes"]
assert int(fields["host_workspace_bytes"], 16) == written["workspace_bytes"]
assert int(fields["peak_host_bytes"], 16) == (
    written["host_resident_bytes"] + written["workspace_bytes"] +
    written["pinned_host_peak_bytes"] + written["state_allocated_bytes"]
)
assert fields["build_source_state"] == written["benchmark_current_source_state"]
assert len(fields["source_delta_identity"]) == 64
assert len(fields["build_identity"]) == 64
assert fields["device"] != "cpu"
assert fields["cuda_build"] != "not-applicable"
assert fields["benchmark_scope"] == "attention_component"
assert file_compared["command"] == "graph attention benchmark compare"
assert file_compared["status"] == "complete"
assert file_compared["benchmark_scope"] == "attention_component"
assert file_compared["component_benchmark_status"] == "compared"
assert file_compared["benchmark_baseline_compatible"] is True
assert file_compared["performance_status"] == "pass"
assert file_compared["benchmark_regression_policy_enabled"] is True
assert len(file_compared["benchmark_regression_policy_identity"]) == 64
assert len(file_compared["benchmark_comparison_identity"]) == 64
assert file_compared["benchmark_regression_basis_points"] == 0
assert file_compared["benchmark_performance_passed"] is True
assert file_compared["benchmark_chart_generated"] is True
assert file_compared["benchmark_chart_path"] == str(file_comparison_chart)
assert len(file_compared["benchmark_chart_identity"]) == 64
assert file_comparison_chart.is_file()
assert hashlib.sha256(file_comparison_chart.read_bytes()).hexdigest() == (
    file_compared["benchmark_chart_identity"]
)
assert file_incompatible["command"] == "graph attention benchmark compare"
assert file_incompatible["status"] == "refused"
assert file_incompatible["component_benchmark_status"] == "unavailable"
assert file_incompatible["correctness_status"] == "not_evaluated"
assert file_incompatible["structural_runtime_status"] == "not_evaluated"
assert file_incompatible["performance_status"] == "not_measured"
assert file_incompatible["failure_code"] == "YVEX_ERR_STATE"
assert file_incompatible["failure_where"] == "runtime_benchmark"
cold_keys = (
    "artifact_open_seconds", "artifact_hash_seconds", "artifact_admission_seconds",
    "binding_open_seconds", "materialization_open_seconds", "runtime_model_seal_seconds",
    "resident_weight_prepare_seconds", "backend_open_seconds", "workspace_prepare_seconds",
    "graph_warmup_seconds", "graph_capture_seconds", "graph_instantiate_seconds",
)
cold_seconds = sum(written[key] for key in cold_keys)
assert abs(int(fields["cold_total_ns"], 16) / 1_000_000_000 - cold_seconds) < 1e-6
lifecycle_keys = cold_keys + ("execution_seconds", "publication_seconds", "cleanup_seconds")
assert sum(written[key] for key in lifecycle_keys) <= written["total_seconds"] + 1e-6
PY
    printf 'cli attention graph live: CPU/CUDA, benchmark optionals, and SVG ok\n'
    fi
fi

printf 'cli attention graph: production grammar and retired proof refusal ok\n'
