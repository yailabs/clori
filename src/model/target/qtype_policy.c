/*
 * Project source dtype and canonical numeric capability policy facts.
 *
 * Qtype policy facts are planning/report-only and do not perform quantization, write GGUF
 * artifacts, or mark runtime paths ready. Qtype policy reporting is not quantization, artifact
 * emission, runtime support, generation readiness, benchmark evidence, or release readiness.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/quant_numeric.h>

#include <stdio.h>
#include <string.h>

typedef struct {
    yvex_model_target_source_profile source;
    const char *mapping_gate_status;
    const char *top_blocker;
    const char *next_row;
    const char *status;
    const char *bracket;
} qtype_policy_state;

static const unsigned int policy_qtypes[] = {
    YVEX_GGUF_QTYPE_F16, YVEX_GGUF_QTYPE_BF16,
    YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_Q8_0,
    YVEX_GGUF_QTYPE_Q2_K, YVEX_GGUF_QTYPE_Q4_K,
    YVEX_GGUF_QTYPE_IQ2_XXS
};

static const char *const policy_prefix_rows[] = {
    "tensor_map_status: naming-map-profiled",
    "output_head_map_status: output-head-profiled"
};

static const char *const policy_suffix_rows[] = {
    "missing_role_report_status: missing-role-report-blocked",
    "qtype_policy_basis: header-only-source-metadata+canonical-numeric-registry",
    "qtype_policy_status: reported"
};

static const char *const policy_downstream_rows[] = {
    "refusal_reasons: Q4_K:encoder-unavailable IQ2_XXS:calibration-required",
    "artifact_identity_status: missing",
    "runtime_descriptor_status: missing",
    "graph_consumer_status: missing",
    "backend_residency_status: missing",
    "downstream_blockers: family_quantization_plan=missing artifact_emit=missing "
    "artifact_identity=missing runtime_descriptor=missing graph_consumer=missing "
    "backend_residency=missing generation_runtime=missing eval_benchmark=missing"
};

static const yvex_model_target_request_rules qtype_policy_rules = {
    YVEX_MODEL_TARGET_COMMAND_QUANT_POLICY,
    "qtype-policy-fail",
    "qtype policy report requires quant-policy command kind",
    "quant-policy",
    1
};

static void qtype_policy_block(qtype_policy_state *state,
                               const char *mapping_status,
                               const char *blocker)
{
    state->mapping_gate_status = mapping_status;
    state->top_blocker = blocker;
    state->next_row = "V010.MAP.9";
    state->status = "blocked";
    state->bracket = "blocked";
}

static void qtype_policy_build_state(const yvex_model_target_request *request,
                                     const char *family,
                                     qtype_policy_state *state)
{
    memset(state, 0, sizeof(*state));
    state->mapping_gate_status = "passed-for-artifact-planning";
    state->top_blocker = "family-quantization-plan-unimplemented";
    state->next_row = "not-scheduled";
    state->status = "policy-reported";
    state->bracket = "reported";

    yvex_model_target_probe_source_profile(request, family, &state->source);
    if (!state->source.source_requested) {
        return;
    }
    if (!state->source.source_directory_present) {
        qtype_policy_block(
            state, "blocked-missing-source",
            strcmp(family, "gemma") == 0
                ? "missing-gemma-source-path" : "missing-qwen-source-path");
        return;
    }

    if (!state->source.header_present) {
        qtype_policy_block(state, "blocked-missing-dtype",
                           "missing-source-dtype-profile");
    } else {
        if (!state->source.attention_k_present) {
            qtype_policy_block(state, "blocked-missing-runtime-roles",
                               "missing-source-role-attention-k");
        } else if (!state->source.output_head_present) {
            qtype_policy_block(state, "blocked-missing-runtime-roles",
                               "missing-output-head-tensor");
        } else if (state->source.output_head_ambiguous) {
            qtype_policy_block(state, "blocked-missing-runtime-roles",
                               "ambiguous-output-head-tensor");
        }
    }

    if (state->source.header_present && !state->source.metadata_present &&
        strcmp(state->status, "policy-reported") == 0) {
        qtype_policy_block(state, "blocked-missing-runtime-roles",
                           "missing-tokenizer-sidecars");
    }
}

static void qtype_policy_prepare(const yvex_model_target_request *request,
                                 const qtype_policy_state *state,
                                 yvex_model_target_report *report)
{
    const char *target = request->target_id[0] ? request->target_id : "";
    const char *family = yvex_model_target_family_key(target);
    const yvex_model_target_report_profile profile = {
        .status = state->status, .target_id = target, .family = family,
        .stage = "report-only", .qtype_policy_status = state->status,
        .artifact_status = "missing", .runtime_status = "unsupported",
        .generation_status = "unsupported-full-model", .benchmark_status = "not-measured",
        .next_row = state->next_row, .reason = state->top_blocker,
        .boundary = "report-only; no quantization/artifact/runtime"
    };

    yvex_model_target_report_prepare(report, request, &profile);
}

static int qtype_policy_validate(const yvex_model_target_request *request,
                                 yvex_model_target_report *report)
{
    const yvex_model_target_record *record;
    const char *target = request->target_id;
    const char *family;

    if (!yvex_model_target_validate_request_shape(
            request, report, &qtype_policy_rules, request->release)) {
        return 1;
    }

    record = yvex_model_target_find(target);
    family = yvex_model_target_family_key(target);
    if (!record) {
        report->status = "unsupported-target";
        report->exit_code = 2;
        if (request->output_contract[0]) {
            yvex_model_target_report_add_row(report, "status: unsupported-target");
            return 1;
        }
        yvex_model_target_report_add_row(report, "qtype-policy: %s [unsupported]",
                                         target);
        yvex_model_target_report_add_row(report, "top_blocker: unsupported-target");
        yvex_model_target_report_add_error(report, "unsupported target: %s", target);
        return 1;
    }
    if (strcmp(family, "qwen") != 0 && strcmp(family, "gemma") != 0) {
        report->status = strcmp(family, "deepseek") == 0 ? "blocked" : "unsupported";
        yvex_model_target_report_add_row(
            report, "qtype-policy: %s [%s]", target,
            strcmp(family, "deepseek") == 0 ? "blocked" : "unsupported");
        yvex_model_target_report_add_row(report, "family: %s", family);
        yvex_model_target_report_add_row(
            report, "top_blocker: %s",
            strcmp(family, "deepseek") == 0
                ? "unsupported-target-class"
                : "unsupported-family");
        if (strcmp(family, "deepseek") != 0) {
            report->status = "unsupported-family";
        }
        return 1;
    }
    return 0;
}

static void qtype_policy_add_contract(const yvex_model_target_request *request,
                                      yvex_model_target_report *report)
{
    if (!request->output_contract[0]) {
        return;
    }
    if (strcmp(request->output_contract, "missing") == 0) {
        report->status = "parser-error";
        report->exit_code = 2;
        yvex_model_target_report_add_row(report, "status: parser-error");
        return;
    }
    if (strcmp(request->output_contract, "normal") != 0 &&
        strcmp(request->output_contract, "table") != 0 &&
        strcmp(request->output_contract, "audit") != 0) {
        report->status = "unsupported-mode";
        report->exit_code = 2;
        yvex_model_target_report_add_row(report, "status: unsupported-mode");
        return;
    }
    yvex_model_target_report_add_output_contract(
        report, "qtype-policy", request->output_contract);
}

static void qtype_policy_numeric_lists(char candidates[96],
                                       char refused[96])
{
    unsigned int index;

    candidates[0] = '\0';
    refused[0] = '\0';
    for (index = 0u; index < sizeof(policy_qtypes) /
                              sizeof(policy_qtypes[0]); ++index) {
        const yvex_quant_numeric_capability *capability =
            yvex_quant_numeric_capability_at(policy_qtypes[index]);
        const char *name = yvex_gguf_qtype_name(policy_qtypes[index]);
        char *target = capability && capability->encoder_available &&
                capability->reference_decoder_available &&
                capability->dedicated_cpu_compute_available &&
                capability->dedicated_cuda_compute_available
            ? candidates : refused;
        size_t capacity = 96u;
        size_t used = strlen(target);

        if (used < capacity)
            (void)snprintf(target + used, capacity - used, "%s%s",
                           used ? "," : "", name);
    }
}

static void qtype_policy_add_table(const qtype_policy_state *state,
                                   yvex_model_target_report *report)
{
    char candidates[96];
    char refused[96];

    qtype_policy_numeric_lists(candidates, refused);
    yvex_model_target_report_add_row(report, "QTYPE POLICY");
    yvex_model_target_report_add_row(
        report,
        "TARGET  FAMILY  SOURCE_DTYPE  POLICY  PREFERRED  CANDIDATES  REFUSED  STATUS  NEXT");
    yvex_model_target_report_add_row(
        report,
        "%s  %s  F32=%lu F16=%lu BF16=%lu other=%lu  artifact-planning-storage-policy  F16  %s  %s  %s  %s",
        report->target_id, report->family, state->source.f32_count,
        state->source.f16_count, state->source.bf16_count,
        state->source.other_count, candidates, refused,
        state->status, state->next_row);
}

static void qtype_policy_add_audit(const qtype_policy_state *state,
                                   yvex_model_target_report *report)
{
    const yvex_quant_numeric_capability *q8 =
        yvex_quant_numeric_capability_at(YVEX_GGUF_QTYPE_Q8_0);
    const yvex_quant_numeric_capability *q2 =
        yvex_quant_numeric_capability_at(YVEX_GGUF_QTYPE_Q2_K);
    yvex_model_target_report_add_row(report, "source_dtype_profile_status: %s",
                                     state->source.header_present ? "profiled" : "missing");
    yvex_model_target_report_add_row(report, "source_dtype_counts: F32=%lu,F16=%lu,BF16=%lu",
                                     state->source.f32_count, state->source.f16_count,
                                     state->source.bf16_count);
    yvex_model_target_report_add_row(report, "source_tensor_count: %lu",
                                     state->source.tensor_count);
    yvex_model_target_report_add_row(report, "mapping_gate_status: %s",
                                     state->mapping_gate_status);
    yvex_model_target_report_add_rows(
        report, policy_prefix_rows,
        sizeof(policy_prefix_rows) / sizeof(policy_prefix_rows[0]));
    yvex_model_target_report_add_row(
        report, "tokenizer_metadata_map_status: %s",
        state->source.metadata_present ? "present-report-only" : "missing");
    yvex_model_target_report_add_rows(
        report, policy_suffix_rows,
        sizeof(policy_suffix_rows) / sizeof(policy_suffix_rows[0]));
    yvex_model_target_report_add_row(
        report,
        "numeric_capability.Q8_0: encoder=%s decoder=%s cpu=%s cuda=%s calibration=%s",
        q8 && q8->encoder_available ? "available" : "unavailable",
        q8 && q8->reference_decoder_available ? "available" : "unavailable",
        q8 && q8->dedicated_cpu_compute_available ? "available" : "unavailable",
        q8 && q8->dedicated_cuda_compute_available ? "available" : "unavailable",
        q8 ? yvex_quant_calibration_name(q8->calibration) : "unknown");
    yvex_model_target_report_add_row(
        report,
        "numeric_capability.Q2_K: encoder=%s decoder=%s cpu=%s cuda=%s calibration=%s",
        q2 && q2->encoder_available ? "available" : "unavailable",
        q2 && q2->reference_decoder_available ? "available" : "unavailable",
        q2 && q2->dedicated_cpu_compute_available ? "available" : "unavailable",
        q2 && q2->dedicated_cuda_compute_available ? "available" : "unavailable",
        q2 ? yvex_quant_calibration_name(q2->calibration) : "unknown");
    yvex_model_target_report_add_rows(
        report, policy_downstream_rows,
        sizeof(policy_downstream_rows) / sizeof(policy_downstream_rows[0]));
    yvex_model_target_report_add_row(report, "next_required_rows: %s",
                                     state->next_row);
    yvex_model_target_report_common_tail(report);
}

int yvex_qtype_policy_report_build(const yvex_model_target_request *request,
                                   yvex_model_target_report *report,
                                   yvex_error *err)
{
    qtype_policy_state state;
    char candidates[96];
    char refused[96];
    const char *family;

    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "qtype_policy_report",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    family = yvex_model_target_family_key(request->target_id);
    qtype_policy_build_state(request, family, &state);
    qtype_policy_prepare(request, &state, report);
    if (qtype_policy_validate(request, report)) {
        return YVEX_OK;
    }
    if (request->output_contract[0]) {
        qtype_policy_add_contract(request, report);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        qtype_policy_add_table(&state, report);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        qtype_policy_add_audit(&state, report);
        return YVEX_OK;
    }

    yvex_model_target_report_add_row(report, "qtype-policy: %s [%s]",
                                     report->target_id, state.bracket);
    yvex_model_target_report_add_row(report, "family: %s  mapping_gate: %s",
                                     report->family, state.mapping_gate_status);
    yvex_model_target_report_add_row(
        report, "source_dtype: F32=%lu F16=%lu BF16=%lu other=%lu",
        state.source.f32_count, state.source.f16_count,
        state.source.bf16_count, state.source.other_count);
    if (strcmp(state.status, "policy-reported") == 0) {
        qtype_policy_numeric_lists(candidates, refused);
        yvex_model_target_report_add_row(report,
                                         "policy: artifact-planning-storage-policy");
        yvex_model_target_report_add_row(report, "preferred: F16");
        yvex_model_target_report_add_row(report, "candidates: %s",
                                         candidates);
        yvex_model_target_report_add_row(report, "refused: %s", refused);
    }
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     state.top_blocker);
    yvex_model_target_report_add_row(report, "next: %s", state.next_row);
    if (strcmp(state.status, "policy-reported") == 0) {
        yvex_model_target_report_add_row(report,
                                         "boundary: report-only; no quantization/artifact/runtime");
    }
    return YVEX_OK;
}
