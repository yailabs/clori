#!/bin/sh
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_ROOT=${YVEX_TEST_OUT_DIR:-build/tests/cli}

run_section() {
    name=$1
    script=$2

    printf 'cli: %s\n' "$name"
    YVEX_BIN="$YVEX_BIN" \
    YVEX_TEST_OUT_DIR="$OUT_ROOT/$name" \
        sh "$script"
}

if [ "${1:-}" = "--cuda" ]; then
    run_section cuda tests/cli/cuda.sh
    exit 0
fi

run_section core tests/cli/core.sh
run_section server tests/cli/server.sh
run_section artifact-integrity tests/cli/artifact_integrity.sh
run_section integrity-report tests/cli/integrity_report.sh
run_section artifact-corruption tests/cli/artifact_corruption.sh
run_section artifact-identity tests/cli/artifact_identity.sh
run_section artifact-metadata tests/cli/artifact_metadata.sh
run_section accounts tests/cli/accounts.sh
run_section materialize tests/cli/materialize.sh
run_section materialize-gate tests/cli/materialize_gate.sh
run_section materialization-integrity-gate tests/cli/materialize_gate.sh
run_section attention-graph tests/cli/attention_graph.sh
run_section minimax-audio tests/cli/minimax_audio.sh
run_section minimax-video tests/cli/minimax_video.sh
run_section media tests/cli/media.sh
run_section token-input tests/cli/token_input.sh
run_section fullmodel tests/cli/fullmodel.sh
run_section artifact-integrity-regression tests/cli/artifact_integrity_regression.sh
run_section source-manifest tests/cli/source_manifest.sh
run_section native-weights tests/cli/native_weights.sh
run_section gguf-template tests/cli/gguf_template.sh
run_section gguf-emit tests/cli/gguf_emit.sh
run_section tensor-map tests/cli/tensor_map.sh
run_section convert tests/cli/convert.sh
run_section model-gate tests/cli/model_gate.sh
run_section models tests/cli/models.sh
run_section model-workflow tests/cli/model_workflow.sh
run_section quant-job tests/cli/quant_job.sh
run_section quant-policy tests/cli/quant_policy.sh
run_section physical-variant tests/cli/physical_variant.sh
run_section imatrix tests/cli/imatrix.sh

printf 'cli smoke: ok\n'
