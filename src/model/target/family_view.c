/*
 * Adapt family-owned inspection lifecycles into the bounded model-target view model.
 *
 * Family construction, refusal details, and cleanup terminate here. Report coordination and CLI
 * rendering see only copied projections and never borrow concrete family object layouts.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/compilation.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/source.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *attention_name(yvex_attention_class value)
{
    static const char *const names[] = {"swa", "csa", "hca"};

    return value <= YVEX_ATTENTION_CLASS_HCA ? names[value] : "unknown";
}

static int projection_refuse(yvex_error *err, const char *reason)
{
    yvex_error_set(err, YVEX_ERR_BOUNDS, "model_target_family_projection", reason);
    return YVEX_ERR_BOUNDS;
}

static int project_transform_coverage(
    yvex_model_target_report *report, const yvex_transform_ir *transform,
    const yvex_source_tensor_snapshot_facts *facts, yvex_error *err);

int yvex_model_target_report_release_coverage(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    const char *operation,
    const char *error_where,
    const char *success_status,
    const char *success_boundary,
    yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    yvex_source_verify_options source_options = {0};
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_source_tensor_snapshot_facts facts = {0};
    yvex_deepseek_v4_ir *architecture = NULL;
    yvex_deepseek_v4_ir_failure architecture_failure = {0};
    yvex_transform_ir *transform = NULL;
    yvex_transform_failure transform_failure = {0};
    char models_root[512];
    char source_path[512];
    int rc;

    if (!request || !report || !operation || !error_where || !success_status ||
        !success_boundary || !family) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_report",
                       "DeepSeek coverage report arguments are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_model_target_release_source_paths(
            request, models_root, sizeof(models_root), source_path,
            sizeof(source_path))) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, error_where,
                       "DeepSeek source path exceeds report bounds");
        return YVEX_ERR_BOUNDS;
    }
    memset(&verification, 0, sizeof(verification));
    source_options.identity = yvex_source_release_identity();
    source_options.source_path = source_path;
    source_options.models_root = models_root;
    rc = yvex_source_verify_with_snapshot(
        &source_options, &verification, &snapshot, err);
    if (rc == YVEX_OK && verification.verified && snapshot)
        rc = family->ir.build(
            &architecture, &verification, &architecture_failure, err);
    if (rc == YVEX_OK)
        rc = family->transform.build(
            &transform, &verification, architecture, snapshot, NULL,
            &transform_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_source_tensor_snapshot_facts_get(snapshot, &facts, err);
    if (rc != YVEX_OK || !verification.verified || !snapshot) {
        report->status = "tensor-coverage-blocked";
        report->exit_code = 5;
        yvex_model_target_report_add_error(
            report,
            "model-target %s: compiled source coverage refused: %s",
            operation,
            transform_failure.code
                ? yvex_transform_failure_name(transform_failure.code)
                : (architecture_failure.code
                       ? family->ir.failure_name(architecture_failure.code)
                       : yvex_source_verification_status(&verification)));
        yvex_transform_ir_release(&transform);
        family->ir.close(architecture);
        yvex_source_tensor_snapshot_release(snapshot);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = project_transform_coverage(report, transform, &facts, err);
    yvex_transform_ir_release(&transform);
    family->ir.close(architecture);
    yvex_source_tensor_snapshot_release(snapshot);
    if (rc != YVEX_OK) return rc;
    {
        const yvex_model_target_report_profile profile = {
            .status = success_status,
            .target_id = request->target_id,
            .family = "deepseek",
            .stage = "header-only",
            .tensor_map_status = "blocked",
            .runtime_status = "unsupported",
            .generation_status = "unsupported",
            .next_row = "V010.SOURCE.PAYLOAD.STREAM.0",
            .boundary = success_boundary};

        yvex_model_target_report_prepare(report, request, &profile);
    }
    return YVEX_OK;
}

static int path_has_suffix(const char *path, const char *suffix)
{
    size_t path_length;
    size_t suffix_length;

    if (!path || !suffix) return 0;
    path_length = strlen(path);
    suffix_length = strlen(suffix);
    return path_length >= suffix_length &&
           strcmp(path + path_length - suffix_length, suffix) == 0;
}

static int deepseek_models_root(
    const yvex_model_target_request *request, char *out, size_t cap)
{
    static const char suffix[] = "/hf/deepseek/DeepSeek-V4-Flash-DSpark";
    const char *environment;
    size_t source_length;
    size_t suffix_length = sizeof(suffix) - 1u;
    int n;

    if (!request || !out || cap == 0u) return 0;
    out[0] = '\0';
    if (request->models_root[0]) {
        n = snprintf(out, cap, "%s", request->models_root);
        return n >= 0 && (size_t)n < cap;
    }
    environment = getenv("YVEX_MODELS_ROOT");
    if (environment && environment[0]) {
        n = snprintf(out, cap, "%s", environment);
        return n >= 0 && (size_t)n < cap;
    }
    source_length = strlen(request->source_path);
    if (path_has_suffix(request->source_path, suffix) &&
        source_length > suffix_length && source_length - suffix_length < cap) {
        memcpy(out, request->source_path, source_length - suffix_length);
        out[source_length - suffix_length] = '\0';
        return 1;
    }
    n = snprintf(out, cap, "%s", "models");
    return n >= 0 && (size_t)n < cap;
}

static int deepseek_source(
    const yvex_model_target_request *request,
    const char *models_root,
    char *out,
    size_t cap)
{
    if (!out || cap == 0u) return 0;
    if (request->source_path[0]) {
        int n = snprintf(out, cap, "%s", request->source_path);

        return n >= 0 && (size_t)n < cap;
    }
    return yvex_source_target_path(
        out, cap, models_root, yvex_source_release_identity());
}

static void deepseek_ir_refusal(
    const yvex_deepseek_v4_ir_failure *failure,
    yvex_model_target_report *report)
{
    report->status = "architecture-ir-refused";
    report->exit_code = 5;
    yvex_model_target_report_add_error(
        report,
        "model-target class-profile: architecture IR refused: %s:%s field=%s layer=%llu",
        yvex_model_register_deepseek_v4()->ir.component_name(failure->component),
        yvex_model_register_deepseek_v4()->ir.failure_name(failure->code),
        failure->field ? failure->field : "none", failure->layer_index);
}

static int deepseek_from_verification(
    const yvex_model_target_request *request,
    const yvex_source_verification *verification,
    yvex_model_target_report *report,
    yvex_error *err)
{
    yvex_deepseek_v4_ir_failure failure;
    yvex_deepseek_v4_ir *architecture = NULL;
    int rc;

    if (!request || !verification || !report ||
        !yvex_source_is_release_target(request->target_id)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "deepseek_architecture_profile",
                       "canonical target, verification, and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_model_register_deepseek_v4()->ir.build(
        &architecture, verification, &failure, err);
    if (rc != YVEX_OK) {
        deepseek_ir_refusal(&failure, report);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    report->family_architecture = architecture;
    report->family_architecture_kind = YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_DEEPSEEK;
    rc = yvex_model_target_report_project_family_detail(report, err);
    if (rc != YVEX_OK) {
        yvex_model_register_deepseek_v4()->ir.close(architecture);
        report->family_architecture = NULL;
        report->family_architecture_kind = YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_NONE;
        return rc;
    }
    {
        const yvex_model_target_report_profile profile = {
            .status = "typed-architecture-specified",
            .target_id = request->target_id,
            .family = "deepseek",
            .stage = "typed-architecture-specification",
            .runtime_status = "unsupported",
            .generation_status = "unsupported",
            .next_row = "V010.SOURCE.PAYLOAD.STREAM.0",
            .boundary = "typed architecture specification; mapping is owned by the "
                        "canonical map plan and payload/runtime remain separate"};

        yvex_model_target_report_prepare(report, request, &profile);
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int minimax_class_profile(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const yvex_minimax_h3_api *api = yvex_model_register_minimax_h3();
    yvex_minimax_h3_open_options options;
    yvex_minimax_h3_failure failure;
    yvex_minimax_h3_target *target = NULL;
    yvex_transform_ir *transformation = NULL;
    int rc;

    if (!request->source_path[0]) {
        report->status = "source-acquisition-required";
        report->exit_code = 5;
        yvex_model_target_report_add_error(
            report, "model-target class-profile: MiniMax-H3 requires --source DIR");
        return YVEX_OK;
    }
    options.source_root = request->source_path;
    rc = api->open(&target, &options, &failure, err);
    if (rc != YVEX_OK) {
        report->status = "source-or-transformation-ir-refused";
        report->exit_code = 5;
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
            yvex_model_target_report_add_row(
                report,
                "{\"status\":\"source-or-transformation-ir-refused\","
                "\"target_id\":\"%s\",\"failure\":\"%s\","
                "\"runtime\":\"unsupported\",\"generation\":\"unsupported\"}",
                request->target_id, api->failure_name(failure.code));
        } else {
            yvex_model_target_report_add_error(
                report, "model-target class-profile: MiniMax-H3 refused: %s tensor=%s",
                api->failure_name(failure.code),
                failure.source_name[0] ? failure.source_name : "none");
        }
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_model_minimax_h3_transform_api()->build(
        &transformation, report->family_derivation_identity, target, err);
    if (rc != YVEX_OK) {
        api->close(&target);
        report->status = "transformation-ir-refused";
        report->exit_code = 5;
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
            yvex_model_target_report_add_row(
                report,
                "{\"status\":\"transformation-ir-refused\","
                "\"target_id\":\"%s\",\"runtime\":\"unsupported\","
                "\"generation\":\"unsupported\"}",
                request->target_id);
        } else {
            yvex_model_target_report_add_error(
                report, "model-target class-profile: MiniMax-H3 Transformation IR refused");
        }
        yvex_error_clear(err);
        return YVEX_OK;
    }
    report->family_architecture = target;
    report->family_transformation = transformation;
    report->family_architecture_kind = YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_MINIMAX_H3;
    rc = yvex_model_target_report_project_family_detail(report, err);
    if (rc != YVEX_OK) {
        yvex_transform_ir_release(&transformation);
        api->close(&target);
        report->family_architecture = NULL;
        report->family_transformation = NULL;
        report->family_architecture_kind = YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_NONE;
        return rc;
    }
    {
        const yvex_model_target_report_profile profile = {
            .status = "transformation-ir-admitted",
            .target_id = request->target_id,
            .family = "minimax-h3",
            .model = "MiniMax-H3 Base FL2VA",
            .target_class = "composite-source-model",
            .stage = "source-verified-architecture-and-transformation-ir",
            .source_status = "exact-immutable-source-verified",
            .artifact_status = "not-produced",
            .tensor_map_status = "complete-source-to-logical-role-map",
            .runtime_status = "unsupported",
            .generation_status = "unsupported",
            .benchmark_status = "not-measured",
            .next_row = "component physical variants and artifact emission",
            .boundary = "source, composite logical target, architecture, tensor roles, and "
                        "artifact-neutral Transformation IR only"};

        yvex_model_target_report_prepare(report, request, &profile);
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int deepseek_class_profile(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    yvex_source_verify_options options;
    yvex_source_verification verification;
    char models_root[512];
    char source_path[512];
    int rc;

    if (!deepseek_models_root(request, models_root, sizeof(models_root))) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "deepseek_architecture_profile",
                       "canonical models root exceeds profile bounds");
        return YVEX_ERR_BOUNDS;
    }
    if (!deepseek_source(request, models_root, source_path, sizeof(source_path))) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "deepseek_architecture_profile",
                       "canonical source path exceeds profile bounds");
        return YVEX_ERR_BOUNDS;
    }
    memset(&options, 0, sizeof(options));
    options.identity = yvex_source_release_identity();
    options.source_path = source_path;
    options.models_root = models_root;
    rc = yvex_source_verify(&options, &verification, err);
    if (rc != YVEX_OK) return rc;
    if (!verification.verified) {
        const char *blocker = verification.blocker_count
                                  ? verification.blockers[0]
                                  : "source-verification-incomplete";

        report->status = "architecture-ir-blocked";
        report->exit_code = 5;
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
            yvex_model_target_report_add_row(
                report,
                "{\"status\":\"architecture-ir-blocked\",\"target_id\":\"%s\","
                "\"source_verification\":\"blocked\",\"reason\":\"%s\","
                "\"runtime\":\"unsupported\",\"generation\":\"unsupported\"}",
                request->target_id, blocker);
        } else if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
            yvex_model_target_report_add_table_row(
                report, 4u, "TARGET", "SOURCE", "IR", "REASON",
                NULL, NULL, NULL, NULL);
            yvex_model_target_report_add_table_row(
                report, 4u, request->target_id, "blocked", "not-built",
                blocker, NULL, NULL, NULL, NULL);
        } else if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
            yvex_model_target_report_add_row(report, "architecture_ir_status: blocked");
            yvex_model_target_report_add_row(report, "target_id: %s", request->target_id);
            yvex_model_target_report_add_row(report, "source_path: %s", source_path);
            yvex_model_target_report_add_row(report, "source_verification_status: blocked");
            yvex_model_target_report_add_row(report, "reason: %s", blocker);
            yvex_model_target_report_add_row(report, "runtime_execution: unsupported");
            yvex_model_target_report_add_row(report, "generation: unsupported");
        } else {
            yvex_model_target_report_add_row(report, "model-class: deepseek");
            yvex_model_target_report_add_row(report, "target: %s", request->target_id);
            yvex_model_target_report_add_row(report, "status: architecture-ir-blocked");
            yvex_model_target_report_add_row(report, "reason: %s", blocker);
            yvex_model_target_report_add_row(
                report,
                "boundary: source verification required; runtime/generation unsupported");
        }
        return YVEX_OK;
    }
    return deepseek_from_verification(request, &verification, report, err);
}

int yvex_model_target_family_class_profile_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    int *handled,
    yvex_error *err)
{
    if (!request || !report || !handled) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_family_profile",
                       "request, report, and handled result are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *handled = 1;
    if (yvex_source_is_release_target(request->target_id))
        return deepseek_class_profile(request, report, err);
    if (strcmp(request->target_id, YVEX_MINIMAX_H3_TARGET_ID) == 0)
        return minimax_class_profile(request, report, err);
    *handled = 0;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_target_family_mapping_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    yvex_source_verify_options source_options;
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *architecture = NULL;
    yvex_transform_ir *transform_ir = NULL;
    yvex_deepseek_gguf_map *map = NULL;
    yvex_deepseek_v4_ir_failure architecture_failure;
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
    rc = family->ir.build(&architecture, &verification, &architecture_failure, err);
    if (rc != YVEX_OK) {
        refusal_stage = "architecture";
        refusal_reason = family->ir.failure_name(architecture_failure.code);
        refusal_source = architecture_failure.field
                             ? architecture_failure.field
                             : "architecture-ir";
        goto cleanup;
    }
    rc = family->transform.build(
        &transform_ir, &verification, architecture, snapshot, NULL,
        &transform_failure, err);
    if (rc != YVEX_OK) {
        refusal_stage = "transformation-ir";
        refusal_reason = yvex_transform_failure_name(transform_failure.code);
        goto cleanup;
    }
    rc = family->lowering.build(&map, architecture, transform_ir, &map_failure, err);
    if (rc != YVEX_OK) {
        refusal_stage = "gguf-lowering";
        refusal_reason = family->lowering.failure_name(map_failure.code);
        refusal_source = map_failure.source_name[0] ? map_failure.source_name : "none";
        refusal_emitted = map_failure.emitted_name[0] ? map_failure.emitted_name : "none";
    }

cleanup:
    yvex_transform_ir_release(&transform_ir);
    family->ir.close(architecture);
    yvex_source_tensor_snapshot_release(snapshot);
    if (rc != YVEX_OK) {
        family->lowering.close(map);
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
    report->family_lowering = map;
    rc = yvex_model_target_report_project_family_detail(report, err);
    if (rc != YVEX_OK) {
        family->lowering.close(map);
        report->family_lowering = NULL;
        return rc;
    }
    {
        const yvex_model_target_report_profile profile = {
            .status = "deepseek-gguf-mapping-complete",
            .target_id = request->target_id,
            .family = "deepseek",
            .stage = "header-only",
            .tensor_map_status = "complete",
            .runtime_status = "unsupported",
            .generation_status = "unsupported",
            .next_row = "V010.SOURCE.PAYLOAD.STREAM.0",
            .boundary = "logical GGUF names, shapes, source contributions, transforms, "
                        "and metadata are complete; no payload, writer, artifact, or runtime "
                        "claim"};

        yvex_model_target_report_prepare(report, request, &profile);
    }
    return YVEX_OK;
}

static int project_map(yvex_model_target_report *report, yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_deepseek_gguf_map_summary *source = family->lowering.summary(
        (const yvex_deepseek_gguf_map *)report->family_lowering);
    yvex_model_target_map_projection *out = &report->detail.map;
    unsigned int index;

    if (!source || YVEX_TENSOR_COLLECTION_COUNT > YVEX_MODEL_TARGET_COLLECTION_CAP)
        return projection_refuse(err, "tensor-map projection exceeds its bounded view");
    memset(out, 0, sizeof(*out));
    out->source_contributions = source->source_contribution_count;
    out->descriptors = source->descriptor_count;
    out->trunk_descriptors = source->trunk_descriptor_count;
    out->draft_descriptors = source->draft_descriptor_count;
    out->pinned_standard_names = source->pinned_standard_count;
    out->semantic_standard_names = source->semantic_standard_count;
    out->extension_names = source->extension_count;
    out->metadata = source->metadata_count;
    out->header_scans = source->header_scan_count;
    out->payload_bytes = source->payload_bytes_read;
    out->source_identity = source->source_identity;
    out->coverage_identity = source->coverage_identity;
    out->mapping_identity = source->mapping_identity;
    out->collection_count = YVEX_TENSOR_COLLECTION_COUNT;
    for (index = 0u; index < out->collection_count; ++index) {
        out->collections[index].count = source->collection_counts[index];
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_TENSOR_MAP;
    yvex_error_clear(err);
    return YVEX_OK;
}

static const char *transform_collection_name(unsigned int collection)
{
    static const char *const names[] = {
        "global", "attention", "compressor", "indexer", "norm", "mhc",
        "router", "routed-expert", "shared-expert", "auxiliary"};

    return collection < sizeof(names) / sizeof(names[0])
               ? names[collection]
               : NULL;
}

static int transform_collection_index(
    yvex_transform_subsystem subsystem, unsigned int *collection)
{
    if (!collection) return 0;
    if (subsystem <= YVEX_TRANSFORM_SUBSYSTEM_SHARED_EXPERT) {
        *collection = (unsigned int)subsystem;
        return 1;
    }
    if (subsystem == YVEX_TRANSFORM_SUBSYSTEM_OUTPUT) {
        *collection = YVEX_TENSOR_COLLECTION_GLOBAL;
        return 1;
    }
    if (subsystem == YVEX_TRANSFORM_SUBSYSTEM_AUXILIARY) {
        *collection = YVEX_TENSOR_COLLECTION_AUXILIARY;
        return 1;
    }
    return 0;
}

static int project_transform_coverage(
    yvex_model_target_report *report, const yvex_transform_ir *transform,
    const yvex_source_tensor_snapshot_facts *facts, yvex_error *err)
{
    const yvex_transform_ir_summary *source =
        yvex_transform_ir_summary_get(transform);
    yvex_model_target_coverage_projection *out;
    unsigned long long source_index;
    unsigned int index;

    if (!report || !source || !facts || !source->complete ||
        YVEX_TENSOR_COLLECTION_COUNT > YVEX_MODEL_TARGET_COLLECTION_CAP)
        return projection_refuse(err, "tensor-coverage projection exceeds its bounded view");
    out = &report->detail.coverage;
    memset(out, 0, sizeof(*out));
    out->source_tensors = facts->tensor_count;
    out->required_tensors = source->source_value_count;
    out->matched_tensors = source->source_value_count;
    out->header_scans = source->header_scan_count;
    out->payload_bytes = source->payload_bytes_read;
    out->source_lookups = facts->lookup_count;
    out->source_collisions = facts->collision_count;
    out->source_maximum_probe = facts->maximum_probe;
    out->source_identity = source->source_snapshot_identity;
    out->coverage_identity = source->coverage_identity;
    out->collection_count = YVEX_TENSOR_COLLECTION_COUNT;
    for (index = 0u; index < out->collection_count; ++index) {
        yvex_core_text_copy(out->collections[index].name,
                            sizeof(out->collections[index].name),
                            transform_collection_name(index));
    }
    for (source_index = 0u; source_index < source->source_value_count;
         ++source_index) {
        const yvex_transform_source_value *value =
            yvex_transform_ir_source_at(transform, source_index);
        unsigned int collection;

        if (!value || !transform_collection_index(value->subsystem, &collection))
            return projection_refuse(
                err, "compiled source has no bounded tensor collection");
        out->collections[collection].count++;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_TENSOR_COVERAGE;
    yvex_error_clear(err);
    return YVEX_OK;
}

static void project_architecture_model(
    yvex_model_target_architecture_projection *out,
    const yvex_deepseek_v4_model_spec *source)
{
#define COPY_TEXT(member) \
    yvex_core_text_copy(out->member, sizeof(out->member), source->member)
    COPY_TEXT(target_id);
    COPY_TEXT(family);
    COPY_TEXT(architecture);
    COPY_TEXT(repository);
    COPY_TEXT(revision);
    COPY_TEXT(verification_stage);
    COPY_TEXT(paper_revision);
    COPY_TEXT(sglang_revision);
    COPY_TEXT(vllm_revision);
#undef COPY_TEXT
    yvex_core_text_copy(out->tokenizer_class, sizeof(out->tokenizer_class),
                        source->tokenizer.tokenizer_class);
    yvex_core_text_copy(out->tokenizer_model_type, sizeof(out->tokenizer_model_type),
                        source->tokenizer.model_type);
    out->hidden_size = source->hidden_size;
    out->vocabulary_size = source->vocabulary_size;
    out->maximum_context = source->maximum_context;
    out->target_layers = source->main_layer_count;
    out->draft_layers = source->auxiliary_layer_count;
    out->swa_layers = source->swa_layer_count;
    out->csa_layers = source->csa_layer_count;
    out->hca_layers = source->hca_layer_count;
    out->hash_router_layers = source->hash_router_layer_count;
    out->learned_router_layers = source->learned_router_layer_count;
    out->dspark_block_size = source->dspark.block_size;
    out->dspark_noise_token_id = source->dspark.noise_token_id;
    out->dspark_markov_rank = source->dspark.markov_rank;
    out->dspark_confidence_available = source->dspark.confidence_available;
    out->mhc_residual_streams = source->final_mhc.residual_streams;
    out->mhc_expanded_width = source->final_mhc.expanded_width;
    out->mhc_mixing_rows = source->final_mhc.mixing_rows;
    out->mhc_mixing_columns = source->final_mhc.mixing_columns;
    out->mhc_sinkhorn_iterations = source->final_mhc.sinkhorn_iterations;
    out->final_mhc_post_required = source->final_mhc_post_required;
    out->final_mhc_head_required = source->final_mhc_head_required;
    out->final_norm_after_mhc_head = source->final_norm_after_mhc_head;
    out->tokenizer_vocabulary_size = source->tokenizer.vocabulary_size;
    out->tokenizer_base_vocab_entries = source->tokenizer.base_vocab_entries;
    out->tokenizer_added_token_entries = source->tokenizer.added_token_entries;
    out->bos_token_id = source->tokenizer.bos_token_id;
    out->eos_token_id = source->tokenizer.eos_token_id;
    out->output_head_required = source->output.required;
    out->output_head_tied = source->output.tied_to_embedding;
    out->source_quant_block_rows = source->source_constraint.quant_block_rows;
    out->source_quant_block_columns = source->source_constraint.quant_block_columns;
    out->source_header_scans = source->source_header_scan_count;
    out->source_header_tensors = source->source_header_tensor_count;
    out->source_payload_bytes = source->source_payload_bytes_read;
}

static int project_architecture(yvex_model_target_report *report, yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_deepseek_v4_ir *ir = report->family_architecture;
    const yvex_deepseek_v4_model_spec *model = family->ir.model(ir);
    yvex_model_target_architecture_projection *out = &report->detail.architecture;
    unsigned long long index;

    if (!model || family->ir.layer_count(ir) > YVEX_MODEL_TARGET_LAYER_CAP ||
        family->ir.auxiliary_count(ir) > YVEX_MODEL_TARGET_LAYER_CAP ||
        model->dspark.target_layer_count > YVEX_MODEL_TARGET_FEATURE_CAP)
        return projection_refuse(err, "architecture projection exceeds its bounded view");
    memset(out, 0, sizeof(*out));
    project_architecture_model(out, model);
    yvex_core_text_copy(out->source_weight_dtype, sizeof(out->source_weight_dtype),
                        family->ir.source_weight_dtype_name(model->source_constraint.weight_dtype));
    yvex_core_text_copy(out->source_expert_dtype, sizeof(out->source_expert_dtype),
                        family->ir.source_expert_dtype_name(model->source_constraint.expert_dtype));
    yvex_core_text_copy(out->source_quantization, sizeof(out->source_quantization),
                        family->ir.source_quantization_name(model->source_constraint.quantization));
    out->dspark_feature_layer_count = (unsigned int)model->dspark.target_layer_count;
    memcpy(out->dspark_feature_layers, model->dspark.target_layer_ids,
           out->dspark_feature_layer_count * sizeof(out->dspark_feature_layers[0]));
    out->layer_count = (unsigned int)family->ir.layer_count(ir);
    for (index = 0u; index < out->layer_count; ++index) {
        const yvex_deepseek_v4_layer_spec *source = family->ir.layer_at(ir, index);
        yvex_model_target_layer_projection *layer = &out->layers[index];

        layer->index = source->layer_index;
        layer->compression_ratio = source->compression_ratio;
        yvex_core_text_copy(layer->attention, sizeof(layer->attention),
                            attention_name(source->attention_class));
        yvex_core_text_copy(layer->kv, sizeof(layer->kv), family->ir.kv_name(source->kv.class_id));
        yvex_core_text_copy(layer->router, sizeof(layer->router),
                            family->ir.router_name(source->moe.router_class));
        yvex_core_text_copy(layer->mhc_entry, sizeof(layer->mhc_entry),
                            source->mhc.entry == YVEX_DEEPSEEK_V4_MHC_STANDALONE_PRE
                                ? "standalone-pre" : "fused-prior-post-pre");
        if (!index) {
            out->query_heads = source->query_heads;
            out->kv_heads = source->kv_heads;
            out->head_dimension = source->head_dimension;
            out->rope_head_dimension = source->rope_head_dimension;
            out->routed_experts = source->moe.routed_experts;
            out->experts_per_token = source->moe.experts_per_token;
            out->shared_experts = source->moe.shared_experts;
        }
    }
    out->draft_count = (unsigned int)family->ir.auxiliary_count(ir);
    for (index = 0u; index < out->draft_count; ++index) {
        const yvex_deepseek_v4_auxiliary_spec *source = family->ir.auxiliary_at(ir, index);
        yvex_model_target_draft_projection *draft = &out->drafts[index];

        draft->predictor_index = source->predictor_index;
        draft->layer_index = source->layer.layer_index;
        draft->compression_ratio = source->layer.compression_ratio;
        yvex_core_text_copy(draft->attention, sizeof(draft->attention),
                            attention_name(source->layer.attention_class));
        yvex_core_text_copy(draft->router, sizeof(draft->router),
                            family->ir.router_name(source->layer.moe.router_class));
        draft->feature_projection = source->has_feature_projection;
        draft->markov = source->has_markov_head;
        draft->confidence = source->has_confidence_head;
        draft->shared_head = source->shares_output_head;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_MODEL_ARCHITECTURE;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int project_composite(yvex_model_target_report *report, yvex_error *err)
{
    const yvex_minimax_h3_api *api = yvex_model_register_minimax_h3();
    const yvex_minimax_h3_target *target = report->family_architecture;
    const yvex_minimax_h3_summary *source = api->summary(target);
    const yvex_transform_ir_summary *transform = yvex_transform_ir_summary_get(
        (const yvex_transform_ir *)report->family_transformation);
    yvex_model_target_composite_projection *out = &report->detail.composite;
    unsigned long long index;

    if (!source || !api->architecture(target) || !transform ||
        source->component_count > YVEX_MODEL_TARGET_COMPONENT_CAP ||
        source->phase_edge_count > YVEX_MODEL_TARGET_EDGE_CAP)
        return projection_refuse(err, "composite architecture exceeds its bounded view");
    memset(out, 0, sizeof(*out));
#define COPY_TEXT(destination, value) \
    yvex_core_text_copy((destination), sizeof(destination), (value))
    COPY_TEXT(out->repository, YVEX_MINIMAX_H3_REPOSITORY);
    COPY_TEXT(out->revision, YVEX_MINIMAX_H3_REVISION);
    COPY_TEXT(out->subtree, YVEX_MINIMAX_H3_SUBTREE);
    COPY_TEXT(out->source_snapshot_identity, source->source_snapshot_identity);
    COPY_TEXT(out->component_manifest_identity, source->component_manifest_identity);
    COPY_TEXT(out->phase_dag_identity, source->phase_dag_identity);
    COPY_TEXT(out->architecture_identity, source->architecture_identity);
    COPY_TEXT(out->role_map_identity, source->role_map_identity);
    COPY_TEXT(out->transformation_identity, transform->transform_identity);
    COPY_TEXT(out->unresolved_requirements_identity, source->unresolved_requirements_identity);
#undef COPY_TEXT
    out->components = source->component_count;
    out->weighted_components = source->weighted_component_count;
    out->phase_edges = source->phase_edge_count;
    out->shards = source->shard_count;
    out->tensors = source->tensor_count;
    out->elements = source->element_count;
    out->payload_bytes = source->payload_bytes;
    out->payload_execution_bytes = transform->payload_bytes_read;
    out->component_count = (unsigned int)source->component_count;
    for (index = 0u; index < out->component_count; ++index) {
        const yvex_minimax_h3_component *component = api->component_at(target, index);
        yvex_model_target_component_projection *row = &out->component[index];

        if (!component) return projection_refuse(err, "composite component projection is absent");
        yvex_core_text_copy(row->canonical_id, sizeof(row->canonical_id), component->canonical_id);
        yvex_core_text_copy(row->identity, sizeof(row->identity), component->identity);
        row->shards = component->shard_count;
        row->tensors = component->tensor_count;
        row->phase = (unsigned int)component->phase;
        row->weighted = component->weighted;
        row->release_after_phase = component->release_after_phase;
    }
    out->edge_count = (unsigned int)source->phase_edge_count;
    for (index = 0u; index < out->edge_count; ++index) {
        const yvex_minimax_h3_phase_edge *edge = api->phase_edge_at(index);
        yvex_model_target_edge_projection *row = &out->edge[index];

        if (!edge) return projection_refuse(err, "composite phase-edge projection is absent");
        row->source_phase = (unsigned int)edge->source_phase;
        row->destination_phase = (unsigned int)edge->destination_phase;
        row->data_classes = edge->data_classes;
        row->lifetime = (unsigned int)edge->lifetime;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_COMPOSITE_ARCHITECTURE;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_target_report_project_family_detail(
    yvex_model_target_report *report, yvex_error *err)
{
    if (!report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_family_projection",
                       "report is required");
        return YVEX_ERR_INVALID_ARG;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_NONE;
    memset(&report->detail, 0, sizeof(report->detail));
    if (report->family_lowering) return project_map(report, err);
    if (report->family_architecture_kind == YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_DEEPSEEK)
        return project_architecture(report, err);
    if (report->family_architecture_kind == YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_MINIMAX_H3)
        return project_composite(report, err);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_model_target_report_close_family_detail(yvex_model_target_report *report)
{
    if (!report) return;
    yvex_model_register_deepseek_v4()->lowering.close(
        (yvex_deepseek_gguf_map *)report->family_lowering);
    yvex_transform_ir_release((yvex_transform_ir **)&report->family_transformation);
    if (report->family_architecture_kind == YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_DEEPSEEK) {
        yvex_model_register_deepseek_v4()->ir.close(
            (yvex_deepseek_v4_ir *)report->family_architecture);
    } else if (report->family_architecture_kind ==
               YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_MINIMAX_H3) {
        yvex_minimax_h3_target *target = report->family_architecture;

        yvex_model_register_minimax_h3()->close(&target);
    }
}
