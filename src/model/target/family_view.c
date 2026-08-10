/*
 * Adapt family-owned inspection lifecycles into the bounded model-target view model.
 *
 * Family construction, refusal details, and cleanup terminate here. Report coordination and CLI
 * rendering see only copied projections and never borrow concrete family object layouts.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/artifact_lowering.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/compiler.h>
#include <yvex/internal/compiler_source.h>
#include <yvex/internal/core.h>
#include <yvex/internal/family_catalog.h>
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
    const yvex_compilation_source_summary *source, yvex_error *err);
static int project_map(
    yvex_model_target_report *report,
    const yvex_artifact_lowering_map *map, yvex_error *err);

int yvex_model_target_report_release_coverage(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    const char *operation,
    const char *error_where,
    const char *success_status,
    const char *success_boundary,
    yvex_error *err)
{
    yvex_family_source_products products = {0};
    yvex_compilation_runtime_binding_request compilation = {0};
    char models_root[512];
    char source_path[512];
    int rc;

    if (!request || !report || !operation || !error_where || !success_status ||
        !success_boundary) {
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
    compilation.source_path = source_path;
    compilation.models_root = models_root;
    rc = yvex_family_source_compile(
        request->target_id, &compilation, &products, err);
    if (rc != YVEX_OK || !products.source_summary || !products.transform_ir) {
        report->status = "tensor-coverage-blocked";
        report->exit_code = 5;
        yvex_model_target_report_add_error(
            report,
            "model-target %s: compiled source coverage refused: %s",
            operation, yvex_error_is_set(err)
                           ? yvex_error_message(err) : "source-compiler-incomplete");
        yvex_family_source_products_release(&products);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = project_transform_coverage(
        report, products.transform_ir, products.source_summary, err);
    yvex_family_source_products_release(&products);
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

static int deepseek_from_verification(
    const yvex_model_target_request *request,
    const yvex_source_verification *verification,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const yvex_graph_execution_binding *execution;
    const yvex_family_binding_pipeline *pipeline;
    yvex_semantic_model_ir *semantic = NULL;
    int rc;

    if (!request || !verification || !report ||
        !yvex_source_is_release_target(request->target_id)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "deepseek_architecture_profile",
                       "canonical target, verification, and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    execution = yvex_graph_execution_find(0ull, 0ull, request->target_id);
    pipeline = execution && execution->compiler
                   ? execution->compiler->binding_pipeline : NULL;
    if (!pipeline || !pipeline->semantic_model_build) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "deepseek_architecture_profile",
                       "target has no semantic compiler adapter");
        rc = YVEX_ERR_UNSUPPORTED;
    } else {
        rc = pipeline->semantic_model_build(&semantic, verification, err);
    }
    if (rc != YVEX_OK) {
        report->status = "architecture-ir-refused";
        report->exit_code = 5;
        yvex_model_target_report_add_error(
            report, "model-target class-profile: Semantic Model IR refused: %s",
            yvex_error_is_set(err) ? yvex_error_message(err) : "compiler-refusal");
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_model_target_report_project_semantic_detail(
        report, semantic, verification, NULL, err);
    yvex_semantic_model_ir_close(&semantic);
    if (rc != YVEX_OK) return rc;
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
    yvex_compilation_runtime_binding_request compilation = {0};
    yvex_family_source_products products = {0};
    int rc;

    if (!request->source_path[0]) {
        report->status = "source-acquisition-required";
        report->exit_code = 5;
        yvex_model_target_report_add_error(
            report, "model-target class-profile: MiniMax-H3 requires --source DIR");
        return YVEX_OK;
    }
    compilation.source_path = request->source_path;
    rc = yvex_family_source_compile(
        request->target_id, &compilation, &products, err);
    if (rc != YVEX_OK) {
        report->status = "source-or-transformation-ir-refused";
        report->exit_code = 5;
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
            yvex_model_target_report_add_row(
                report,
                "{\"status\":\"source-or-transformation-ir-refused\","
                "\"target_id\":\"%s\",\"failure\":\"%s\","
                "\"runtime\":\"unsupported\",\"generation\":\"unsupported\"}",
                request->target_id,
                yvex_error_is_set(err) ? yvex_error_message(err) : "compiler-refusal");
        } else {
            yvex_model_target_report_add_error(
                report, "model-target class-profile: MiniMax-H3 refused: %s",
                yvex_error_is_set(err) ? yvex_error_message(err) : "compiler-refusal");
        }
        yvex_family_source_products_release(&products);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_model_target_report_project_semantic_detail(
        report, products.semantic_model, products.verification,
        products.transform_ir, err);
    yvex_core_text_copy(report->derivation_identity,
                        sizeof(report->derivation_identity),
                        products.derivation_identity);
    yvex_family_source_products_release(&products);
    if (rc != YVEX_OK) return rc;
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
    if (strcmp(yvex_model_target_family_key(request->target_id), "minimax-h3") == 0)
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
    yvex_compilation_runtime_binding_request compilation = {0};
    yvex_family_source_products products = {0};
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
    compilation.source_path = source_path;
    compilation.models_root = models_root;
    rc = yvex_family_source_compile(
        request->target_id, &compilation, &products, err);
    if (rc != YVEX_OK || !products.lowering) {
        report->status = "mapping-plan-blocked";
        report->exit_code = 5;
        yvex_model_target_report_add_error(
            report,
            "model-target mapping-gate: compiled lowering refused: %s",
            yvex_error_is_set(err) ? yvex_error_message(err)
                                   : "lowering-projection-absent");
        yvex_family_source_products_release(&products);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = project_map(report, products.lowering, err);
    yvex_family_source_products_release(&products);
    if (rc != YVEX_OK) return rc;
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

static int project_map(
    yvex_model_target_report *report,
    const yvex_artifact_lowering_map *map, yvex_error *err)
{
    const yvex_artifact_lowering_summary *source =
        yvex_artifact_lowering_operations.summary(map);
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
    const yvex_compilation_source_summary *compiled_source, yvex_error *err)
{
    const yvex_transform_ir_summary *source =
        yvex_transform_ir_summary_get(transform);
    yvex_model_target_coverage_projection *out;
    unsigned long long source_index;
    unsigned int index;

    if (!report || !source || !compiled_source || !compiled_source->complete ||
        !source->complete ||
        YVEX_TENSOR_COLLECTION_COUNT > YVEX_MODEL_TARGET_COLLECTION_CAP)
        return projection_refuse(err, "tensor-coverage projection exceeds its bounded view");
    out = &report->detail.coverage;
    memset(out, 0, sizeof(*out));
    out->source_tensors = compiled_source->source_tensor_count;
    out->required_tensors = source->source_value_count;
    out->matched_tensors = source->source_value_count;
    out->header_scans = compiled_source->source_header_scan_count;
    out->payload_bytes = compiled_source->source_payload_bytes_read;
    out->source_lookups = compiled_source->source_lookup_count;
    out->source_collisions = compiled_source->source_collision_count;
    out->source_maximum_probe = compiled_source->source_maximum_probe;
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
    const yvex_semantic_model_ir *model_ir,
    const yvex_semantic_model_ir_summary *semantic,
    const yvex_source_verification *source,
    const yvex_semantic_attention_layer *last)
{
#define COPY_TEXT(member, value) \
    yvex_core_text_copy(out->member, sizeof(out->member), (value))
    const yvex_model_execution_descriptor *model = &semantic->execution_descriptor;

    COPY_TEXT(target_id, semantic->target_id);
    COPY_TEXT(family, yvex_model_target_family_key(semantic->target_id));
    COPY_TEXT(architecture, source->architecture);
    COPY_TEXT(repository, source->repository_id);
    COPY_TEXT(revision, source->revision);
    COPY_TEXT(verification_stage, source->verification_stage);
    COPY_TEXT(source_weight_dtype, source->torch_dtype);
    COPY_TEXT(source_expert_dtype, source->expert_dtype);
    COPY_TEXT(source_quantization, source->quant_method[0]
                                       ? source->quant_method : source->quant_format);
    COPY_TEXT(paper_revision,
              yvex_semantic_model_ir_reference(model_ir, "architecture-paper"));
    COPY_TEXT(sglang_revision,
              yvex_semantic_model_ir_reference(model_ir, "sglang-reference"));
    COPY_TEXT(vllm_revision,
              yvex_semantic_model_ir_reference(model_ir, "vllm-reference"));
    yvex_core_text_copy(out->tokenizer_class, sizeof(out->tokenizer_class),
                        source->tokenizer_class);
    yvex_core_text_copy(out->tokenizer_model_type, sizeof(out->tokenizer_model_type),
                        source->tokenizer_model_type);
#undef COPY_TEXT
    out->hidden_size = model->hidden_width;
    out->vocabulary_size = model->vocabulary_size;
    out->maximum_context = model->maximum_context;
    out->target_layers = model->layer_count;
    out->draft_layers = model->draft_layer_count;
    out->swa_layers = model->swa_layers;
    out->csa_layers = model->csa_layers;
    out->hca_layers = model->hca_layers;
    out->hash_router_layers = model->hash_router_layer_count;
    out->learned_router_layers = model->layer_count - model->hash_router_layer_count;
    out->query_heads = model->attention_heads;
    out->kv_heads = model->kv_heads;
    out->head_dimension = model->head_width;
    out->rope_head_dimension = last ? last->rope_head_dimension : 0ull;
    out->routed_experts = model->routed_experts;
    out->experts_per_token = model->experts_per_row;
    out->shared_experts = model->shared_experts;
    out->dspark_block_size = model->proposal_width;
    out->dspark_noise_token_id = model->draft_noise_token_id;
    out->dspark_markov_rank = model->markov_rank;
    out->dspark_confidence_available = model->confidence_width != 0ull;
    if (last) {
        out->mhc_residual_streams = last->residual_stream_count;
        out->mhc_expanded_width = last->residual_expanded_width;
        out->mhc_mixing_rows = last->mhc_mixing_rows;
        out->mhc_mixing_columns = last->mhc_mixing_columns;
        out->mhc_sinkhorn_iterations = last->mhc_sinkhorn_iterations;
        out->final_mhc_post_required = last->mhc_attention_pre_and_post;
        out->final_mhc_head_required = last->residual_stream_count > 1ull;
        out->final_norm_after_mhc_head = last->attention_input_norm_required;
    }
    out->tokenizer_vocabulary_size = source->tokenizer_effective_vocab_size;
    out->tokenizer_base_vocab_entries = source->tokenizer_base_vocab_count;
    out->tokenizer_added_token_entries = source->tokenizer_added_token_count;
    out->bos_token_id = model->bos_token_id;
    out->eos_token_id = model->eos_token_id;
    out->output_head_required = model->output_vocabulary_size != 0ull;
    out->output_head_tied = source->tie_word_embeddings;
    out->source_quant_block_rows = source->quant_block_rows;
    out->source_quant_block_columns = source->quant_block_columns;
    out->source_header_scans = source->header_scan_count;
    out->source_header_tensors = source->header_tensor_count;
    out->source_payload_bytes = 0ull;
}

static const char *state_name(yvex_attention_class attention)
{
    static const char *const names[] = {
        "swa-state", "csa-state-core-indexer", "hca-state-core"};

    return attention <= YVEX_ATTENTION_CLASS_HCA ? names[attention] : "unknown";
}

static int project_architecture(
    yvex_model_target_report *report, const yvex_semantic_model_ir *semantic,
    const yvex_source_verification *verification, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *summary =
        yvex_semantic_model_ir_summary_get(semantic);
    const yvex_model_execution_descriptor *model =
        summary ? &summary->execution_descriptor : NULL;
    const yvex_semantic_attention_layer *layers = NULL, *drafts = NULL;
    unsigned long long layer_count = 0ull, draft_count = 0ull;
    yvex_model_target_architecture_projection *out = &report->detail.architecture;
    unsigned long long index;

    if (!summary || !verification || !verification->verified ||
        model->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
        !yvex_semantic_model_ir_attention_view(
            semantic, YVEX_TENSOR_SCOPE_MAIN_LAYER, &layers, &layer_count) ||
        !yvex_semantic_model_ir_attention_view(
            semantic, YVEX_TENSOR_SCOPE_DRAFT, &drafts, &draft_count) ||
        layer_count > YVEX_MODEL_TARGET_LAYER_CAP ||
        draft_count > YVEX_MODEL_TARGET_LAYER_CAP ||
        model->target_feature_count > YVEX_MODEL_TARGET_FEATURE_CAP)
        return projection_refuse(err, "architecture projection exceeds its bounded view");
    memset(out, 0, sizeof(*out));
    project_architecture_model(
        out, semantic, summary, verification, &layers[layer_count - 1ull]);
    out->dspark_feature_layer_count = (unsigned int)model->target_feature_count;
    memcpy(out->dspark_feature_layers, model->target_feature_layers,
           out->dspark_feature_layer_count * sizeof(out->dspark_feature_layers[0]));
    out->layer_count = (unsigned int)layer_count;
    for (index = 0u; index < out->layer_count; ++index) {
        const yvex_semantic_attention_layer *source = &layers[index];
        yvex_model_target_layer_projection *layer = &out->layers[index];

        layer->index = source->layer_index;
        layer->compression_ratio = source->compression_ratio;
        yvex_core_text_copy(layer->attention, sizeof(layer->attention),
                            attention_name(source->attention_class));
        yvex_core_text_copy(layer->kv, sizeof(layer->kv),
                            state_name(source->attention_class));
        yvex_core_text_copy(layer->router, sizeof(layer->router),
                            source->layer_index < model->hash_router_layer_count
                                ? "hash-token-id" : "learned-hidden-noaux-tc");
        yvex_core_text_copy(layer->mhc_entry, sizeof(layer->mhc_entry),
                            source->mhc_entry_policy == 0u
                                ? "standalone-pre" : "fused-prior-post-pre");
    }
    out->draft_count = (unsigned int)draft_count;
    for (index = 0u; index < out->draft_count; ++index) {
        const yvex_semantic_attention_layer *source = &drafts[index];
        yvex_model_target_draft_projection *draft = &out->drafts[index];

        draft->predictor_index = source->predictor_index;
        draft->layer_index = source->layer_index;
        draft->compression_ratio = source->compression_ratio;
        yvex_core_text_copy(draft->attention, sizeof(draft->attention),
                            attention_name(source->attention_class));
        yvex_core_text_copy(draft->router, sizeof(draft->router),
                            "learned-hidden-noaux-tc");
        draft->feature_projection = model->target_feature_count != 0ull;
        draft->markov = model->markov_rank != 0ull;
        draft->confidence = model->confidence_width != 0ull;
        draft->shared_head = model->output_vocabulary_size != 0ull;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_MODEL_ARCHITECTURE;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int project_composite(
    yvex_model_target_report *report, const yvex_semantic_model_ir *semantic,
    const yvex_transform_ir *transformation, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *semantic_summary =
        yvex_semantic_model_ir_summary_get(semantic);
    const yvex_semantic_composite_summary *source =
        semantic_summary ? &semantic_summary->composite : NULL;
    const yvex_transform_ir_summary *transform =
        yvex_transform_ir_summary_get(transformation);
    const yvex_semantic_component *components = NULL;
    const yvex_semantic_phase_edge *edges = NULL;
    unsigned long long component_count = 0ull, edge_count = 0ull;
    yvex_model_target_composite_projection *out = &report->detail.composite;
    unsigned long long index;

    if (!source || !transform ||
        !yvex_semantic_model_ir_composite_view(
            semantic, &components, &component_count, &edges, &edge_count) ||
        source->component_count > YVEX_MODEL_TARGET_COMPONENT_CAP ||
        source->phase_edge_count > YVEX_MODEL_TARGET_EDGE_CAP)
        return projection_refuse(err, "composite architecture exceeds its bounded view");
    memset(out, 0, sizeof(*out));
#define COPY_TEXT(destination, value) \
    yvex_core_text_copy((destination), sizeof(destination), (value))
    COPY_TEXT(out->repository, source->repository);
    COPY_TEXT(out->revision, source->revision);
    COPY_TEXT(out->subtree, source->subtree);
    COPY_TEXT(out->source_snapshot_identity, source->source_snapshot_identity);
    COPY_TEXT(out->component_manifest_identity, source->component_manifest_identity);
    COPY_TEXT(out->phase_dag_identity, source->phase_dag_identity);
    COPY_TEXT(out->architecture_identity, source->architecture_identity);
    COPY_TEXT(out->role_map_identity, source->role_map_identity);
    COPY_TEXT(out->transformation_identity, transform->transform_identity);
    COPY_TEXT(out->unresolved_requirements_identity, source->unresolved_requirements_identity);
#undef COPY_TEXT
    out->components = component_count;
    out->weighted_components = source->weighted_component_count;
    out->phase_edges = edge_count;
    out->shards = source->shards;
    out->tensors = source->tensors;
    out->elements = source->elements;
    out->payload_bytes = source->payload_bytes;
    out->payload_execution_bytes = transform->payload_bytes_read;
    out->component_count = (unsigned int)component_count;
    for (index = 0u; index < out->component_count; ++index) {
        const yvex_semantic_component *component = &components[index];
        yvex_model_target_component_projection *row = &out->component[index];

        yvex_core_text_copy(row->canonical_id, sizeof(row->canonical_id), component->canonical_id);
        yvex_core_text_copy(row->identity, sizeof(row->identity), component->identity);
        row->shards = component->shards;
        row->tensors = component->tensors;
        row->phase = component->phase;
        row->weighted = component->weighted;
        row->release_after_phase = component->release_after_phase;
    }
    out->edge_count = (unsigned int)edge_count;
    for (index = 0u; index < out->edge_count; ++index) {
        const yvex_semantic_phase_edge *edge = &edges[index];
        yvex_model_target_edge_projection *row = &out->edge[index];

        row->source_phase = edge->source_phase;
        row->destination_phase = edge->destination_phase;
        row->data_classes = edge->data_classes;
        row->lifetime = edge->lifetime;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_COMPOSITE_ARCHITECTURE;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_target_report_project_semantic_detail(
    yvex_model_target_report *report,
    const yvex_semantic_model_ir *semantic,
    const yvex_source_verification *verification,
    const yvex_transform_ir *transform, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *summary =
        yvex_semantic_model_ir_summary_get(semantic);

    if (!report || !summary) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_family_projection",
                       "report and sealed Semantic Model IR are required");
        return YVEX_ERR_INVALID_ARG;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_NONE;
    memset(&report->detail, 0, sizeof(report->detail));
    if (summary->composite.component_count)
        return project_composite(report, semantic, transform, err);
    return project_architecture(report, semantic, verification, err);
}
