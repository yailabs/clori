/*
 * Evaluate typed mapping-gate evidence from bounded source facts.
 *
 * The DeepSeek release path consumes the canonical immutable map; legacy family gates remain
 * header/sidecar evidence. Neither path marks payload, artifact, runtime, or generation behavior
 * ready. Mapping gate status is not quantization, artifact emission, runtime readiness, generation
 * readiness, benchmark evidence, or release readiness.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/source.h>

#include <string.h>

typedef struct {
    yvex_model_target_source_profile source;
    int source_observed;
    int source_missing;
    int source_ambiguous;
    int metadata_observed;
    int metadata_missing;
    const char *missing_roles;
    const char *ambiguous_roles;
    const char *top_blocker;
    const char *next_row;
    const char *status;
    const char *result;
} mapping_gate_state;

typedef struct {
    int source_observed;
    int source_missing;
    int source_ambiguous;
    int metadata_observed;
    int metadata_missing;
    const char *missing_roles;
    const char *ambiguous_roles;
    const char *top_blocker;
    const char *status;
} mapping_gate_block;

typedef struct {
    const char *status;
    const char *result;
    const char *target;
    const char *family;
    const char *metadata;
    const char *missing;
    const char *ambiguous;
    const char *next;
    int source_observed;
    int metadata_observed;
} mapping_gate_audit_facts;

#define MAPPING_LITERAL(text) \
    { YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u }
#define MAPPING_STRING(field, format) \
    { YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(mapping_gate_audit_facts, field) }
#define MAPPING_INT(field, format) \
    { YVEX_MODEL_TARGET_ROW_INT, (format), offsetof(mapping_gate_audit_facts, field) }

static const yvex_model_target_row_spec mapping_gate_audit_rows[] = {
    MAPPING_STRING(status, "tensor_mapping_gate_status: %s"),
    MAPPING_STRING(result, "tensor_mapping_gate_result: %s"),
    MAPPING_STRING(target, "tensor_mapping_gate_target_id: %s"),
    MAPPING_STRING(family, "tensor_mapping_gate_family: %s"),
    MAPPING_LITERAL("tensor_naming_map_status: naming-map-profiled"),
    MAPPING_LITERAL("output_head_map_status: output-head-profiled"),
    MAPPING_STRING(metadata, "tokenizer_metadata_map_status: %s"),
    MAPPING_LITERAL("missing_role_report_status: missing-role-report-blocked"),
    MAPPING_LITERAL("expected_source_role_count: 12"),
    MAPPING_INT(source_observed, "observed_source_role_count: %d"),
    MAPPING_LITERAL("expected_metadata_role_count: 4"),
    MAPPING_INT(metadata_observed, "observed_metadata_role_count: %d"),
    MAPPING_STRING(missing, "missing_roles: %s"),
    MAPPING_STRING(ambiguous, "ambiguous_roles: %s"),
    MAPPING_LITERAL(
        "downstream_blockers: artifact_contract=missing qtype_policy=missing "
        "runtime_descriptor=missing graph_consumer=missing backend_residency=missing "
        "logits_runtime=missing tokenizer_runtime=missing generation_runtime=missing "
        "eval_benchmark=missing"),
    MAPPING_STRING(next, "next_required_rows: %s"),
    MAPPING_LITERAL("payload_bytes_read: false"),
    MAPPING_LITERAL("artifact_emitted: false"),
    MAPPING_LITERAL("runtime_descriptor_constructed: false"),
    MAPPING_LITERAL("graph_consumer_fed: false")
};

static const mapping_gate_block mapping_source_qwen_block = {
    0, 12, 0, 0, 4, "all-source-roles", "none",
    "missing-qwen-source-path", "blocked-missing-source"
};
static const mapping_gate_block mapping_source_gemma_block = {
    0, 12, 0, 0, 4, "all-source-roles", "none",
    "missing-gemma-source-path", "blocked-missing-source"
};
static const mapping_gate_block mapping_attention_k_block = {
    11, 1, 0, 4, 0, "attention_k", "none",
    "missing-source-role-attention-k", "blocked-missing-runtime-roles"
};
static const mapping_gate_block mapping_output_ambiguous_block = {
    11, 0, 1, 4, 0, "none", "output_head",
    "ambiguous-output-head-tensor", "blocked-missing-runtime-roles"
};
static const mapping_gate_block mapping_output_missing_block = {
    11, 1, 0, 4, 0, "output_head", "none",
    "missing-output-head-tensor", "blocked-missing-runtime-roles"
};
static const mapping_gate_block mapping_metadata_block = {
    12, 0, 0, 0, 4,
    "tokenizer_metadata,config_metadata,generation_metadata,special_tokens",
    "none", "missing-tokenizer-sidecars", "blocked-missing-runtime-roles"
};

static const yvex_model_target_request_rules mapping_gate_rules = {
    YVEX_MODEL_TARGET_COMMAND_TENSOR_MAP,
    "mapping-gate-fail",
    "mapping gate report requires tensor-map command kind",
    NULL,
    0
};

#undef MAPPING_LITERAL
#undef MAPPING_STRING
#undef MAPPING_INT

static void mapping_gate_apply_block(mapping_gate_state *state,
                                     const mapping_gate_block *block)
{
    state->source_observed = block->source_observed;
    state->source_missing = block->source_missing;
    state->source_ambiguous = block->source_ambiguous;
    state->metadata_observed = block->metadata_observed;
    state->metadata_missing = block->metadata_missing;
    state->missing_roles = block->missing_roles;
    state->ambiguous_roles = block->ambiguous_roles;
    state->top_blocker = block->top_blocker;
    state->status = block->status;
    state->next_row = "V010.MAP.9";
    state->result = "block";
}

int yvex_model_mapping_report_deepseek(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    yvex_source_verify_options source_options;
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *architecture = NULL;
    yvex_deepseek_tensor_coverage *coverage = NULL;
    yvex_transform_ir *transform_ir = NULL;
    yvex_deepseek_gguf_map *map = NULL;
    yvex_deepseek_v4_ir_failure architecture_failure;
    yvex_deepseek_tensor_coverage_failure coverage_failure;
    yvex_transform_failure transform_failure;
    yvex_deepseek_gguf_map_failure map_failure;
    const char *refusal_stage = NULL;
    const char *refusal_reason = NULL;
    const char *refusal_source = "none";
    const char *refusal_emitted = "none";
    char models_root[512];
    char source_path[512];
    int rc;

    if (!yvex_model_target_release_source_paths(
            request, models_root, sizeof(models_root), source_path,
            sizeof(source_path))) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "mapping_gate_report",
                       "DeepSeek source path exceeds report bounds");
        return YVEX_ERR_BOUNDS;
    }
    memset(&source_options, 0, sizeof(source_options));
    memset(&verification, 0, sizeof(verification));
    memset(&architecture_failure, 0, sizeof(architecture_failure));
    memset(&coverage_failure, 0, sizeof(coverage_failure));
    memset(&transform_failure, 0, sizeof(transform_failure));
    memset(&map_failure, 0, sizeof(map_failure));
    source_options.identity = yvex_source_release_identity();
    source_options.source_path = source_path;
    source_options.models_root = models_root;
    source_options.promote_manifest = 0;
    rc = yvex_source_verify_with_snapshot(
        &source_options, &verification, &snapshot, err);
    if (rc != YVEX_OK || !verification.verified || !snapshot) {
        refusal_stage = "source-verification";
        refusal_reason = yvex_source_verification_status(&verification);
        refusal_source = source_path;
        if (rc == YVEX_OK) rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    rc = yvex_model_register_deepseek_v4()->ir.build(
        &architecture, &verification, &architecture_failure, err);
    if (rc != YVEX_OK) {
        refusal_stage = "architecture";
        refusal_reason = yvex_model_register_deepseek_v4()->ir.failure_name(
            architecture_failure.code);
        refusal_source = architecture_failure.field
            ? architecture_failure.field : "architecture-ir";
        goto cleanup;
    }
    rc = yvex_model_register_deepseek_v4()->coverage.build(
        &coverage, &verification, architecture, snapshot, NULL,
        &coverage_failure, err);
    if (rc != YVEX_OK) {
        refusal_stage = "source-coverage";
        refusal_reason = yvex_model_register_deepseek_v4()->coverage.failure_name(
            coverage_failure.code);
        refusal_source = coverage_failure.tensor_name[0]
            ? coverage_failure.tensor_name : "source-tensor";
        goto cleanup;
    }
    rc = yvex_model_register_deepseek_v4()->transform.build(
        &transform_ir, &verification, architecture, coverage, NULL,
        &transform_failure, err);
    if (rc != YVEX_OK) {
        refusal_stage = "transformation-ir";
        refusal_reason = yvex_transform_failure_name(transform_failure.code);
        goto cleanup;
    }
    rc = yvex_model_register_deepseek_v4()->lowering.build(
        &map, architecture, transform_ir, &map_failure, err);
    if (rc != YVEX_OK) {
        refusal_stage = "gguf-lowering";
        refusal_reason = yvex_model_register_deepseek_v4()->lowering.failure_name(map_failure.code);
        refusal_source = map_failure.source_name[0]
            ? map_failure.source_name : "none";
        refusal_emitted = map_failure.emitted_name[0]
            ? map_failure.emitted_name : "none";
    }

cleanup:
    yvex_transform_ir_release(&transform_ir);
    yvex_model_register_deepseek_v4()->ir.close(architecture);
    yvex_source_tensor_snapshot_release(snapshot);
    if (rc != YVEX_OK) {
        yvex_model_register_deepseek_v4()->lowering.close(map);
        yvex_model_register_deepseek_v4()->coverage.close(coverage);
        report->status = "mapping-plan-blocked";
        report->exit_code = 5;
        yvex_model_target_report_add_error(
            report,
            "model-target mapping-gate: DeepSeek %s refused: %s source=%s emitted=%s",
            refusal_stage ? refusal_stage : "mapping-plan",
            refusal_reason ? refusal_reason : "invalid-lifecycle-state",
            refusal_source, refusal_emitted);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    report->family_coverage = coverage;
    report->family_lowering = map;
    {
        const yvex_model_target_report_profile profile = {
            .status = "deepseek-gguf-mapping-complete",
            .target_id = request->target_id, .family = "deepseek", .stage = "header-only",
            .tensor_map_status = "complete", .runtime_status = "unsupported",
            .generation_status = "unsupported", .next_row = "V010.SOURCE.PAYLOAD.STREAM.0",
            .boundary = "logical GGUF names, shapes, source contributions, transforms, "
                        "and metadata are complete; no payload, writer, artifact, or runtime claim"
        };

        yvex_model_target_report_prepare(report, request, &profile);
    }
    return YVEX_OK;
}

static void mapping_gate_build_state(const yvex_model_target_request *request,
                                     const char *family,
                                     mapping_gate_state *state)
{
    memset(state, 0, sizeof(*state));
    state->source_observed = 12;
    state->metadata_observed = 4;
    state->missing_roles = "none";
    state->ambiguous_roles = "none";
    state->top_blocker = "missing-qtype-policy-report";
    state->next_row = "V010.QUANT.0";
    state->status = "passed-for-artifact-planning";
    state->result = "pass";

    yvex_model_target_probe_source_profile(request, family, &state->source);
    if (!state->source.source_requested) {
        return;
    }

    if (!state->source.header_present) {
        mapping_gate_apply_block(
            state, strcmp(family, "gemma") == 0
                       ? &mapping_source_gemma_block
                       : &mapping_source_qwen_block);
        return;
    }

    if (!state->source.attention_k_present) {
        mapping_gate_apply_block(state, &mapping_attention_k_block);
    } else if (state->source.output_head_ambiguous) {
        mapping_gate_apply_block(state, &mapping_output_ambiguous_block);
    } else if (!state->source.output_head_present) {
        mapping_gate_apply_block(state, &mapping_output_missing_block);
    } else if (!state->source.metadata_present) {
        mapping_gate_apply_block(state, &mapping_metadata_block);
    }
}

static void mapping_gate_prepare(const yvex_model_target_request *request,
                                 const mapping_gate_state *state,
                                 yvex_model_target_report *report)
{
    const char *target = request->target_id[0] ? request->target_id : "qwen3-8b";
    const char *family = yvex_model_target_family_key(target);
    const yvex_model_target_report_profile profile = {
        .status = state->status, .target_id = target, .family = family,
        .stage = "report-only",
        .eligibility = strcmp(state->result, "pass") == 0 ? "report-pass" : "blocked",
        .artifact_status = "missing", .tensor_map_status = "naming-map-profiled",
        .qtype_policy_status = strcmp(state->result, "pass") == 0 ? "missing" : "blocked",
        .runtime_status = "unsupported", .generation_status = "unsupported-full-model",
        .benchmark_status = "not-measured", .next_row = state->next_row,
        .boundary = "report-only; no artifact/runtime/generation",
        .reason = state->top_blocker
    };

    yvex_model_target_report_prepare(report, request, &profile);
}

static int mapping_gate_validate(const yvex_model_target_request *request,
                                 yvex_model_target_report *report)
{
    const char *target = request->target_id[0] ? request->target_id : "qwen3-8b";

    if (!yvex_model_target_validate_request_shape(
            request, report, &mapping_gate_rules, request->gate)) {
        return 1;
    }
    if (!yvex_model_target_supported_source_target(target)) {
        report->status = "unsupported-target";
        report->exit_code = 2;
        yvex_model_target_report_add_row(report, "status: unsupported-target");
        yvex_model_target_report_add_row(report, "target_id: %s", target);
        yvex_model_target_report_add_error(report, "unsupported target: %s", target);
        return 1;
    }
    return 0;
}

static void mapping_gate_add_table(const mapping_gate_state *state,
                                   yvex_model_target_report *report)
{
    yvex_model_target_report_add_row(report, "TENSOR MAPPING GATE");
    yvex_model_target_report_add_row(
        report,
        "TARGET  FAMILY  GATE  SOURCE_ROLES  META_ROLES  MISSING  AMBIG  TOP_BLOCKER  STATUS  NEXT");
    yvex_model_target_report_add_row(
        report,
        "%s  %s  v0.1.0  %d/12  %d/4  %d  %d  %s  %s  %s",
        report->target_id, report->family, state->source_observed,
        state->metadata_observed, state->source_missing + state->metadata_missing,
        state->source_ambiguous, state->top_blocker, state->status,
        state->next_row);
}

static void mapping_gate_add_audit(const mapping_gate_state *state,
                                   yvex_model_target_report *report)
{
    mapping_gate_audit_facts facts = {
        state->status, state->result, report->target_id, report->family,
        state->source.metadata_present ? "present-report-only" : "missing",
        state->missing_roles, state->ambiguous_roles, state->next_row,
        state->source_observed, state->metadata_observed
    };

    yvex_model_target_report_project_rows(
        report, mapping_gate_audit_rows,
        sizeof(mapping_gate_audit_rows) / sizeof(mapping_gate_audit_rows[0]),
        &facts);
    yvex_model_target_report_common_tail(report);
}

int yvex_mapping_gate_report_build(const yvex_model_target_request *request,
                                   yvex_model_target_report *report,
                                   yvex_error *err)
{
    mapping_gate_state state;
    const char *family;

    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "mapping_gate_report",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (yvex_source_is_release_target(request->target_id)) {
        memset(&state, 0, sizeof(state));
        state.status = "mapping-plan-check";
        state.result = "block";
        state.next_row = "V010.SOURCE.PAYLOAD.STREAM.0";
        state.top_blocker = "mapping-plan-not-evaluated";
        mapping_gate_prepare(request, &state, report);
        if (mapping_gate_validate(request, report)) return YVEX_OK;
        return yvex_model_mapping_report_deepseek(request, report, err);
    }
    family = yvex_model_target_family_key(
        request->target_id[0] ? request->target_id : "qwen3-8b");
    mapping_gate_build_state(request, family, &state);
    mapping_gate_prepare(request, &state, report);
    if (mapping_gate_validate(request, report)) {
        return YVEX_OK;
    }
    if (request->output_contract[0]) {
        yvex_model_target_report_add_output_contract(
            report, "mapping-gate", request->output_contract);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        mapping_gate_add_table(&state, report);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        mapping_gate_add_audit(&state, report);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        yvex_model_target_report_add_row(
            report,
            "{\"status\":\"%s\",\"target_id\":\"%s\",\"top_blocker\":\"%s\",\"next\":\"%s\"}",
            state.status, report->target_id, state.top_blocker, state.next_row);
        return YVEX_OK;
    }

    yvex_model_target_report_add_row(
        report, "tensor-mapping-gate: %s [%s]", report->target_id,
        strcmp(state.result, "pass") == 0 ? "reported" : "blocked");
    yvex_model_target_report_add_row(report, "gate: v0.1.0  family: %s",
                                     report->family);
    yvex_model_target_report_add_row(
        report,
        "roles: source %d/12, metadata %d/4, missing %d, ambiguous %d",
        state.source_observed, state.metadata_observed,
        state.source_missing + state.metadata_missing, state.source_ambiguous);
    if (strcmp(state.missing_roles, "none") != 0) {
        yvex_model_target_report_add_row(report, "missing: %s",
                                         state.missing_roles);
    }
    if (strcmp(state.ambiguous_roles, "none") != 0) {
        yvex_model_target_report_add_row(report, "ambiguous: %s",
                                         state.ambiguous_roles);
    }
    yvex_model_target_report_add_row(report, "result: %s", state.result);
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     state.top_blocker);
    yvex_model_target_report_add_row(report, "next: %s", state.next_row);
    yvex_model_target_report_add_row(report,
                                     "boundary: report-only; no artifact/runtime/generation");
    return YVEX_OK;
}
