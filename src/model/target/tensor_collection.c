/*
 * Bind canonical family tensor recipes to retained source inventory rows and expose deterministic
 * lookup/accounting to compilation consumers.
 *
 * Each verified source tensor matches exactly one requirement; complete coverage retains one
 * immutable snapshot and reads zero payload. Lexical collection facts never become role mapping,
 * while exact DeepSeek coverage is only an input to sealed transformation planning.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/source.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/source.h>

typedef struct {
    const char *status;
    const char *family;
    const char *target;
    const char *source;
    const char *moe_status;
    unsigned long long tensors;
    unsigned long long layers;
    unsigned long long embed;
    unsigned long long attention_q;
    unsigned long long attention_k;
    unsigned long long attention_v;
    unsigned long long attention_o;
    unsigned long long attention_complete;
    unsigned long long mlp_gate;
    unsigned long long mlp_up;
    unsigned long long mlp_down;
    unsigned long long mlp_complete;
    unsigned long long norm;
    unsigned long long head;
    unsigned long long moe;
} collection_audit_facts;

#define COLLECTION_LITERAL(text) \
    { YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u }
#define COLLECTION_STRING(field, format) \
    { YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(collection_audit_facts, field) }
#define COLLECTION_U64(field, format) \
    { YVEX_MODEL_TARGET_ROW_U64, (format), offsetof(collection_audit_facts, field) }

static const yvex_model_target_row_spec collection_audit_rows[] = {
    COLLECTION_STRING(status, "tensor_collection_status: %s"),
    COLLECTION_STRING(family, "tensor_collection_family: %s"),
    COLLECTION_STRING(target, "tensor_collection_target_id: %s"),
    COLLECTION_LITERAL("tensor_collection_stage: header-collection-inventory"),
    COLLECTION_LITERAL("tensor_collection_evidence_basis: header-metadata-only"),
    COLLECTION_STRING(source, "tensor_collection_source_status: %s"),
    COLLECTION_LITERAL("tensor_collection_manifest_status: not-checked"),
    COLLECTION_STRING(source, "tensor_collection_config_status: %s"),
    COLLECTION_STRING(source, "tensor_collection_tokenizer_status: %s"),
    COLLECTION_U64(tensors, "tensor_collection_tensor_count: %llu"),
    COLLECTION_U64(layers, "tensor_collection_layer_count_observed: %llu"),
    COLLECTION_LITERAL("tensor_collection_embedding_status: candidate"),
    COLLECTION_U64(embed, "tensor_collection_embedding_tensor_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_attention_status: candidate"),
    COLLECTION_U64(attention_q, "tensor_collection_attention_q_count: %llu"),
    COLLECTION_U64(attention_k, "tensor_collection_attention_k_count: %llu"),
    COLLECTION_U64(attention_v, "tensor_collection_attention_v_count: %llu"),
    COLLECTION_U64(attention_o, "tensor_collection_attention_o_count: %llu"),
    COLLECTION_U64(attention_complete,
                   "tensor_collection_attention_complete_qkvo_layer_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_mlp_status: candidate"),
    COLLECTION_U64(mlp_gate, "tensor_collection_mlp_gate_count: %llu"),
    COLLECTION_U64(mlp_up, "tensor_collection_mlp_up_count: %llu"),
    COLLECTION_U64(mlp_down, "tensor_collection_mlp_down_count: %llu"),
    COLLECTION_U64(mlp_complete, "tensor_collection_mlp_complete_gud_layer_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_norm_status: candidate"),
    COLLECTION_U64(norm, "tensor_collection_norm_tensor_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_output_head_status: candidate"),
    COLLECTION_U64(head, "tensor_collection_output_head_tensor_count: %llu"),
    COLLECTION_STRING(moe_status, "tensor_collection_moe_status: %s"),
    COLLECTION_U64(moe, "tensor_collection_moe_router_count: %llu"),
    COLLECTION_U64(moe, "tensor_collection_moe_expert_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_tokenizer_collection_status: sidecar-observed"),
    COLLECTION_LITERAL(
        "tensor_collection_kv_runtime_state_status: runtime-state-required-not-implemented"),
    COLLECTION_LITERAL("tensor_collection_validation_status: lexical-and-header-only"),
    COLLECTION_LITERAL("tensor_collection_role_mapping_status: not-implemented"),
    COLLECTION_LITERAL("tensor_collection_runtime_descriptor_status: not-implemented"),
    COLLECTION_LITERAL("tensor_collection_graph_consumer_status: not-implemented")
};

static const char *const coverage_scope_names[] = {
    "global", "main-layer", "mtp"
};

static const char *const coverage_expert_projections[] = {
    "w1", "w2", "w3"
};

static const char *const coverage_mhc_suffixes[] = {
    "hc_head_fn", "hc_head_base", "hc_head_scale"
};

static const char *const coverage_collection_names[] = {
    "global", "attention", "compressor", "indexer", "norm", "mhc",
    "router", "routed-expert", "shared-expert", "auxiliary"
};

static const char *const coverage_failure_names[] = {
    "none", "invalid-argument", "wrong-source-identity",
    "invalid-inventory-authority", "inventory-drift",
    "architecture-incomplete", "missing-requirement", "ambiguous-match",
    "unexpected-source", "invalid-index", "rank-mismatch", "shape-mismatch",
    "dtype-mismatch", "scale-companion-mismatch", "arithmetic-overflow",
    "resource-limit", "allocation-failure"
};

#undef COLLECTION_LITERAL
#undef COLLECTION_STRING
#undef COLLECTION_U64

static void collection_family_facts(const char *family,
                                    const char **top_blocker,
                                    const char **source_blocker)
{
    if (strcmp(family, "gemma") == 0) {
        *top_blocker = "missing-gemma-tensor-role-map";
        *source_blocker = "missing-gemma-source-path";
    } else {
        *top_blocker = "missing-qwen-tensor-role-map";
        *source_blocker = "missing-qwen-source-path";
    }
}

static void collection_audit_common(yvex_model_target_report *report,
                                    const yvex_model_target_request *request,
                                    const char *family,
                                    const yvex_model_target_source_scan *scan,
                                    const char *status)
{
    collection_audit_facts facts = {
        status, family, request->target_id,
        scan->source_present ? "present" : "missing",
        scan->moe == 0 ? "not-observed" : "candidate",
        scan->tensors, scan->layers, scan->embed,
        scan->attn >= 1 ? 1ull : 0ull, scan->attn >= 2 ? 1ull : 0ull,
        scan->attn >= 3 ? 1ull : 0ull, scan->attn >= 4 ? 1ull : 0ull,
        scan->attn >= 4 ? 1ull : 0ull, scan->mlp >= 1 ? 1ull : 0ull,
        scan->mlp >= 2 ? 1ull : 0ull, scan->mlp >= 3 ? 1ull : 0ull,
        scan->mlp >= 3 ? 1ull : 0ull, scan->norm, scan->head, scan->moe
    };

    yvex_model_target_report_project_rows(
        report, collection_audit_rows,
        sizeof(collection_audit_rows) / sizeof(collection_audit_rows[0]), &facts);
    yvex_model_target_report_common_tail(report);
    yvex_model_target_report_add_row(report, "next_required_rows: V010.MAP.8");
}

int yvex_tensor_collection_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const char *family;
    const char *top_blocker;
    const char *source_blocker;
    yvex_model_target_source_scan scan;
    const char *status;

    if (!request || !report ||
        request->kind != YVEX_MODEL_TARGET_COMMAND_TENSOR_COLLECTION) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tensor_collection",
                       "tensor collection report requires tensor-collection command kind");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_model_target_validate_supported(
            request, report, "tensor-collection", 0)) {
        return YVEX_OK;
    }
    if (yvex_source_is_release_target(request->target_id)) {
        return yvex_model_target_report_release_coverage(
            request, report, "tensor-collection", "tensor_collection",
            "exact-source-tensor-covered",
            "exact source coverage consumed by the canonical map; payload, artifact, "
            "runtime, and generation remain blocked",
            err);
    }
    family = yvex_model_target_family_key(request->target_id);
    collection_family_facts(family, &top_blocker, &source_blocker);
    yvex_model_target_scan_source(request, family, &scan);
    status = scan.source_present ? "collection-profiled" : "source-missing";
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report, "TENSOR COLLECTION INVENTORY");
        yvex_model_target_report_add_row(
            report, "FAMILY  TARGET  STATUS  EMBED  ATTN_QKVO  MLP_GUD  NORM  "
                    "HEAD  MOE  LAYERS  NEXT");
        yvex_model_target_report_add_row(report, "%s  %s  %s  %llu  %llu  %llu  %llu  %llu  %llu  %llu  V010.MAP.8",
                                         family, request->target_id, status,
                                         scan.embed, scan.attn >= 4 ? 1ull : 0ull,
                                         scan.mlp >= 3 ? 1ull : 0ull,
                                         scan.norm, scan.head, scan.moe,
                                         scan.layers);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        collection_audit_common(report, request, family, &scan, status);
        return YVEX_OK;
    }
    yvex_model_target_report_add_row(report, "tensor-collection: %s", family);
    yvex_model_target_report_add_row(report, "target: %s", request->target_id);
    yvex_model_target_report_add_row(report, "status: %s", status);
    yvex_model_target_report_add_row(report, "stage: header-collection-inventory");
    yvex_model_target_report_add_row(report, "evidence: header-metadata-only");
    yvex_model_target_report_add_row(report,
                                     "collections: embedding=%llu attention_qkvo=%llu "
                                     "mlp_gud=%llu norm=%llu head=%llu moe=%llu",
                                     scan.embed, scan.attn >= 4 ? 1ull : 0ull,
                                     scan.mlp >= 3 ? 1ull : 0ull, scan.norm,
                                     scan.head, scan.moe);
    yvex_model_target_report_add_row(report, "layers_observed: %llu", scan.layers);
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     scan.source_present ? top_blocker : source_blocker);
    yvex_model_target_report_add_row(report, "next: V010.MAP.8");
    yvex_model_target_report_add_row(
        report, "boundary: tensor collection inventory only; no role "
                "mapping/runtime/generation");
    return YVEX_OK;
}
/* Exact coverage binds every typed requirement to one retained source row. */

#define DEEPSEEK_COVERAGE_DEFAULT_LIMIT 100000ull

/* Coverage lifecycle and diagnostics used by construction failure paths. */
static void coverage_close(yvex_deepseek_tensor_coverage *coverage);
static const char *coverage_collection_name(
    yvex_tensor_collection collection);
static const char *coverage_failure_name(
    yvex_deepseek_tensor_coverage_failure_code code);

static const yvex_model_family_ir_api *coverage_family_ir(void)
{
    return &yvex_model_register_deepseek_v4()->ir;
}

struct yvex_deepseek_tensor_coverage {
    yvex_deepseek_tensor_coverage_options options;
    yvex_source_tensor_snapshot *snapshot;
    yvex_deepseek_tensor_coverage_row *rows;
    const yvex_deepseek_tensor_coverage_row **row_by_source;
    yvex_deepseek_tensor_coverage_summary summary;
};

typedef struct {
    yvex_deepseek_tensor_coverage *coverage;
    unsigned char *matched;
    yvex_deepseek_tensor_coverage_failure *failure;
    yvex_error *err;
} coverage_builder;

static void *coverage_allocate(size_t size, void *context)
{
    (void)context;
    return malloc(size);
}

static void coverage_release(void *allocation, void *context)
{
    (void)context;
    free(allocation);
}

static void coverage_failure_clear(
    yvex_deepseek_tensor_coverage_failure *failure)
{
    if (!failure) return;
    memset(failure, 0, sizeof(*failure));
    failure->layer_index = YVEX_DEEPSEEK_TENSOR_NO_INDEX;
    failure->expert_index = YVEX_DEEPSEEK_TENSOR_NO_INDEX;
    failure->dimension_index = YVEX_DEEPSEEK_TENSOR_NO_INDEX;
}

static int coverage_reject(
    yvex_deepseek_tensor_coverage_failure *failure,
    yvex_deepseek_tensor_coverage_failure_code code,
    yvex_tensor_collection collection,
    yvex_tensor_scope scope,
    const char *name,
    unsigned long long layer,
    unsigned long long expert,
    unsigned long long dimension,
    unsigned long long expected,
    unsigned long long actual,
    yvex_error *err)
{
    yvex_status status = code == YVEX_DEEPSEEK_COVERAGE_FAILURE_ALLOCATION
                             ? YVEX_ERR_NOMEM
                             : (code == YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_ARGUMENT
                                    ? YVEX_ERR_INVALID_ARG
                                    : YVEX_ERR_FORMAT);

    if (failure) {
        coverage_failure_clear(failure);
        failure->code = code;
        failure->collection = collection;
        failure->scope = scope;
        yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), name ? name : "");
        failure->layer_index = layer;
        failure->expert_index = expert;
        failure->dimension_index = dimension;
        failure->expected = expected;
        failure->actual = actual;
    }
    yvex_error_setf(err, status, "deepseek_tensor_coverage",
                    "%s tensor=%s layer=%llu expert=%llu dimension=%llu expected=%llu actual=%llu",
                    coverage_failure_name(code),
                    name ? name : "none", layer, expert, dimension,
                    expected, actual);
    return status;
}

static const char *coverage_scope_name(yvex_tensor_scope scope)
{
    return scope <= YVEX_TENSOR_SCOPE_MTP
               ? coverage_scope_names[scope]
               : "unknown";
}

static int coverage_require(coverage_builder *builder,
                            const char *name,
                            yvex_tensor_collection collection,
                            yvex_tensor_scope scope,
                            unsigned long long layer,
                            unsigned long long expert,
                            yvex_native_dtype dtype,
                            unsigned int rank,
                            const unsigned long long *dims)
{
    yvex_deepseek_tensor_coverage *coverage = builder->coverage;
    const yvex_native_weight_info *source;
    unsigned long long source_index;
    unsigned int dimension;
    unsigned long long row_index = coverage->summary.required_tensor_count;

    if (!yvex_source_tensor_snapshot_find_index(
            coverage->snapshot, name, &source_index)) {
        coverage->summary.missing_count++;
        return coverage_reject(
            builder->failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_MISSING_REQUIREMENT,
            collection, scope, name, layer, expert,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, builder->err);
    }
    if (row_index >= coverage->summary.source_tensor_count) {
        return coverage_reject(
            builder->failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_RESOURCE_LIMIT,
            collection, scope, name, layer, expert,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            coverage->summary.source_tensor_count, row_index + 1u,
            builder->err);
    }
    source = yvex_source_tensor_snapshot_at(coverage->snapshot, source_index);
    if (builder->matched[source_index]) {
        coverage->summary.ambiguous_count++;
        return coverage_reject(
            builder->failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_AMBIGUOUS_MATCH,
            collection, scope, name, layer, expert,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 2u, builder->err);
    }
    if (source->rank != rank) {
        return coverage_reject(
            builder->failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_RANK_MISMATCH,
            collection, scope, name, layer, expert,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, rank, source->rank, builder->err);
    }
    for (dimension = 0u; dimension < rank; ++dimension) {
        if (source->dims[dimension] != dims[dimension]) {
            return coverage_reject(
                builder->failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_SHAPE_MISMATCH,
                collection, scope, name, layer, expert, dimension,
                dims[dimension], source->dims[dimension], builder->err);
        }
    }
    if (source->dtype != dtype) {
        return coverage_reject(
            builder->failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_DTYPE_MISMATCH,
            collection, scope, name, layer, expert,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, (unsigned long long)dtype,
            (unsigned long long)source->dtype, builder->err);
    }
    builder->matched[source_index] = 1u;
    coverage->rows[row_index].source = source;
    coverage->rows[row_index].collection = collection;
    coverage->rows[row_index].scope = scope;
    coverage->rows[row_index].layer_index = layer;
    coverage->rows[row_index].expert_index = expert;
    coverage->row_by_source[source_index] = &coverage->rows[row_index];
    coverage->summary.required_tensor_count++;
    coverage->summary.matched_tensor_count++;
    coverage->summary.collection_counts[collection]++;
    return YVEX_OK;
}

static int coverage_vector(coverage_builder *builder,
                           const char *name,
                           yvex_tensor_collection collection,
                           yvex_tensor_scope scope,
                           unsigned long long layer,
                           unsigned long long expert,
                           yvex_native_dtype dtype,
                           unsigned long long width)
{
    unsigned long long dims[1] = {width};
    return coverage_require(builder, name, collection, scope, layer, expert,
                            dtype, 1u, dims);
}

static int coverage_matrix(coverage_builder *builder,
                           const char *name,
                           yvex_tensor_collection collection,
                           yvex_tensor_scope scope,
                           unsigned long long layer,
                           unsigned long long expert,
                           yvex_native_dtype dtype,
                           unsigned long long rows,
                           unsigned long long columns)
{
    unsigned long long dims[2] = {rows, columns};
    return coverage_require(builder, name, collection, scope, layer, expert,
                            dtype, 2u, dims);
}

static int coverage_companion_matrix(
    coverage_builder *builder,
    const char *name,
    yvex_tensor_collection collection,
    yvex_tensor_scope scope,
    unsigned long long layer,
    unsigned long long expert,
    yvex_native_dtype dtype,
    unsigned long long rows,
    unsigned long long columns)
{
    if (!yvex_source_tensor_snapshot_find(builder->coverage->snapshot, name)) {
        builder->coverage->summary.missing_count++;
        return coverage_reject(
            builder->failure,
            YVEX_DEEPSEEK_COVERAGE_FAILURE_SCALE_COMPANION,
            collection, scope, name, layer, expert,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, builder->err);
    }
    return coverage_matrix(builder, name, collection, scope, layer, expert,
                           dtype, rows, columns);
}

static int coverage_fp8_pair(coverage_builder *builder,
                             const char *base,
                             yvex_tensor_collection collection,
                             yvex_tensor_scope scope,
                             unsigned long long layer,
                             unsigned long long expert,
                             unsigned long long rows,
                             unsigned long long columns,
                             const yvex_deepseek_v4_source_constraint *storage)
{
    char name[256];
    int rc;

    if (!storage->quant_block_rows || !storage->quant_block_columns ||
        rows % storage->quant_block_rows != 0u ||
        columns % storage->quant_block_columns != 0u) {
        return coverage_reject(
            builder->failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_SCALE_COMPANION,
            collection, scope, base, layer, expert,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 0u, 0u, builder->err);
    }
    (void)snprintf(name, sizeof(name), "%s.weight", base);
    rc = coverage_matrix(builder, name, collection, scope, layer, expert,
                         YVEX_NATIVE_DTYPE_F8_E4M3, rows, columns);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s.scale", base);
    return coverage_companion_matrix(
        builder, name, collection, scope, layer, expert, storage->scale_dtype,
        rows / storage->quant_block_rows,
        columns / storage->quant_block_columns);
}

static int coverage_fp4_pair(coverage_builder *builder,
                             const char *base,
                             yvex_tensor_scope scope,
                             unsigned long long layer,
                             unsigned long long expert,
                             unsigned long long rows,
                             unsigned long long columns,
                             const yvex_deepseek_v4_source_constraint *storage)
{
    char name[256];
    int rc;

    if (!storage->fp4_packing_factor || !storage->fp4_scale_group_width ||
        columns % storage->fp4_packing_factor != 0u ||
        columns % storage->fp4_scale_group_width != 0u) {
        return coverage_reject(
            builder->failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_SCALE_COMPANION,
            YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, scope, base,
            layer, expert, YVEX_DEEPSEEK_TENSOR_NO_INDEX, 0u, columns,
            builder->err);
    }
    (void)snprintf(name, sizeof(name), "%s.weight", base);
    rc = coverage_matrix(
        builder, name, YVEX_TENSOR_COLLECTION_ROUTED_EXPERT,
        scope, layer, expert, storage->fp4_physical_dtype, rows,
        columns / storage->fp4_packing_factor);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s.scale", base);
    return coverage_companion_matrix(
        builder, name, YVEX_TENSOR_COLLECTION_ROUTED_EXPERT,
        scope, layer, expert, storage->scale_dtype, rows,
        columns / storage->fp4_scale_group_width);
}

static int coverage_experts(coverage_builder *builder, const char *prefix,
                            const yvex_deepseek_v4_layer_spec *layer,
                            yvex_tensor_scope scope,
                            const yvex_deepseek_v4_model_spec *model)
{
    unsigned long long expert;
    unsigned int index;

    for (expert = 0u; expert < layer->moe.routed_experts; ++expert) {
        for (index = 0u; index < 3u; ++index) {
            unsigned long long rows = index == 1u ? model->hidden_size
                                                  : layer->moe.expert_intermediate_size;
            unsigned long long columns = index == 1u ? layer->moe.expert_intermediate_size
                                                     : model->hidden_size;
            char base[256];
            int rc;

            (void)snprintf(base, sizeof(base), "%s.ffn.experts.%llu.%s",
                           prefix, expert, coverage_expert_projections[index]);
            rc = coverage_fp4_pair(builder, base, scope, layer->layer_index, expert, rows, columns,
                                   &model->source_constraint);
            if (rc != YVEX_OK) return rc;
        }
    }
    return YVEX_OK;
}

static int coverage_recipe_phase(coverage_builder *builder, const char *prefix,
                                 const yvex_deepseek_v4_layer_spec *layer,
                                 yvex_tensor_scope scope,
                                 const yvex_deepseek_v4_model_spec *model,
                                 unsigned int phase)
{
    const yvex_model_family_ir_api *family_ir = coverage_family_ir();
    unsigned long long index;

    for (index = 0u; index < family_ir->recipe_count(); ++index) {
        const yvex_deepseek_tensor_recipe *recipe = family_ir->recipe_at(index);
        unsigned long long dims[2];
        char name[256];
        int rc;

        if (!recipe || recipe->phase != phase ||
            !family_ir->recipe_enabled(recipe, layer)) continue;
        dims[0] = family_ir->recipe_dimension(recipe, 0u, layer, model);
        dims[1] = recipe->rank == 2u
            ? family_ir->recipe_dimension(recipe, 1u, layer, model) : 0u;
        (void)snprintf(name, sizeof(name), "%s.%s", prefix, recipe->suffix);
        if (recipe->kind == YVEX_DEEPSEEK_RECIPE_FP8_PAIR) {
            rc = coverage_fp8_pair(builder, name, recipe->collection, scope, layer->layer_index,
                                   YVEX_DEEPSEEK_TENSOR_NO_INDEX, dims[0], dims[1],
                                   &model->source_constraint);
        } else if (recipe->rank == 1u) {
            rc = coverage_vector(builder, name, recipe->collection, scope, layer->layer_index,
                                 YVEX_DEEPSEEK_TENSOR_NO_INDEX, recipe->dtype, dims[0]);
        } else {
            rc = coverage_matrix(builder, name, recipe->collection, scope, layer->layer_index,
                                 YVEX_DEEPSEEK_TENSOR_NO_INDEX, recipe->dtype, dims[0], dims[1]);
        }
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

static int coverage_layer(coverage_builder *builder, const char *prefix,
                          const yvex_deepseek_v4_layer_spec *layer,
                          yvex_tensor_scope scope,
                          const yvex_deepseek_v4_model_spec *model)
{
    int rc = coverage_recipe_phase(builder, prefix, layer, scope, model, 0u);
    if (rc == YVEX_OK) rc = coverage_experts(builder, prefix, layer, scope, model);
    if (rc == YVEX_OK) rc = coverage_recipe_phase(builder, prefix, layer, scope, model, 1u);
    return rc;
}

static int coverage_mhc_head(coverage_builder *builder, const char *prefix,
                             yvex_tensor_scope scope, unsigned long long layer,
                             const yvex_deepseek_v4_mhc_head_spec *head,
                             yvex_tensor_collection collection)
{
    unsigned long long dims[3][2] = {{head->function_rows, head->function_columns},
                                    {head->base_width, 0u}, {head->scale_width, 0u}};
    unsigned int index;

    if (!head->required) return YVEX_OK;
    for (index = 0u; index < 3u; ++index) {
        char name[256];
        int rc;
        (void)snprintf(name, sizeof(name), "%s%s", prefix,
                       coverage_mhc_suffixes[index]);
        rc = index == 0u
                 ? coverage_matrix(builder, name, collection, scope, layer,
                                   YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_F32,
                                   dims[index][0], dims[index][1])
                 : coverage_vector(builder, name, collection, scope, layer,
                                   YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_F32,
                                   dims[index][0]);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

static int coverage_build_requirements(coverage_builder *builder,
                                       const yvex_deepseek_v4_ir *ir)
{
    const yvex_deepseek_v4_model_spec *model = coverage_family_ir()->model(ir);
    unsigned long long index;
    int rc;

    rc = coverage_matrix(builder, "embed.weight", YVEX_TENSOR_COLLECTION_GLOBAL,
                         YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
                         YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_BF16,
                         model->embedding.vocabulary_size, model->embedding.hidden_size);
    if (rc == YVEX_OK)
        rc = coverage_vector(builder, "norm.weight", YVEX_TENSOR_COLLECTION_GLOBAL,
                             YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
                             YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_BF16,
                             model->hidden_size);
    if (rc == YVEX_OK)
        rc = coverage_matrix(builder, "head.weight", YVEX_TENSOR_COLLECTION_GLOBAL,
                             YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
                             YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_BF16,
                             model->output.vocabulary_size, model->output.input_width);
    if (rc == YVEX_OK)
        rc = coverage_mhc_head(builder, "", YVEX_TENSOR_SCOPE_GLOBAL,
                               YVEX_DEEPSEEK_TENSOR_NO_INDEX, &model->final_mhc_head,
                               YVEX_TENSOR_COLLECTION_GLOBAL);
    for (index = 0u; rc == YVEX_OK && index < model->main_layer_count; ++index) {
        char prefix[64];
        (void)snprintf(prefix, sizeof(prefix), "layers.%llu", index);
        rc = coverage_layer(builder, prefix, coverage_family_ir()->layer_at(ir, index),
                            YVEX_TENSOR_SCOPE_MAIN_LAYER, model);
    }
    for (index = 0u; rc == YVEX_OK && index < model->auxiliary_layer_count; ++index) {
        const yvex_deepseek_v4_auxiliary_spec *aux = coverage_family_ir()->auxiliary_at(ir, index);
        static const char *norms[] = {"enorm.weight", "hnorm.weight", "norm.weight"};
        unsigned long long widths[] = {aux->embedding_projection_input,
                                      aux->hidden_projection_input, model->hidden_size};
        char prefix[64], name[256];
        unsigned int item;

        (void)snprintf(prefix, sizeof(prefix), "mtp.%llu", index);
        rc = coverage_layer(builder, prefix, &aux->layer, YVEX_TENSOR_SCOPE_MTP, model);
        for (item = 0u; rc == YVEX_OK && item < 2u; ++item) {
            unsigned long long rows = item ? aux->hidden_projection_output
                                           : aux->embedding_projection_output;
            unsigned long long columns = item ? aux->hidden_projection_input
                                              : aux->embedding_projection_input;
            (void)snprintf(name, sizeof(name), "%s.%s", prefix, item ? "h_proj" : "e_proj");
            rc = coverage_fp8_pair(builder, name, YVEX_TENSOR_COLLECTION_AUXILIARY,
                                   YVEX_TENSOR_SCOPE_MTP, aux->layer.layer_index,
                                   YVEX_DEEPSEEK_TENSOR_NO_INDEX, rows, columns,
                                   &model->source_constraint);
        }
        for (item = 0u; rc == YVEX_OK && item < 3u; ++item) {
            (void)snprintf(name, sizeof(name), "%s.%s", prefix, norms[item]);
            rc = coverage_vector(builder, name, YVEX_TENSOR_COLLECTION_AUXILIARY,
                                 YVEX_TENSOR_SCOPE_MTP, aux->layer.layer_index,
                                 YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_BF16,
                                 widths[item]);
        }
        (void)snprintf(name, sizeof(name), "%s.", prefix);
        if (rc == YVEX_OK)
            rc = coverage_mhc_head(builder, name, YVEX_TENSOR_SCOPE_MTP,
                                   aux->layer.layer_index, &aux->mhc_head,
                                   YVEX_TENSOR_COLLECTION_AUXILIARY);
    }
    return rc;
}

static int coverage_parse_name_index(const char *text,
                                     unsigned long long *value,
                                     const char **end)
{
    unsigned long long parsed = 0u;
    const char *cursor = text;

    if (!cursor || *cursor < '0' || *cursor > '9') return 0;
    while (*cursor >= '0' && *cursor <= '9') {
        unsigned long long digit = (unsigned long long)(*cursor - '0');
        if (parsed > (ULLONG_MAX - digit) / 10u) return -1;
        parsed = parsed * 10u + digit;
        cursor++;
    }
    *value = parsed;
    *end = cursor;
    return 1;
}

static int coverage_reject_unexpected(coverage_builder *builder,
                                      const yvex_deepseek_v4_ir *ir,
                                      const yvex_native_weight_info *source)
{
    const yvex_deepseek_v4_model_spec *model =
        coverage_family_ir()->model(ir);
    const yvex_deepseek_v4_layer_spec *layer_spec = NULL;
    const yvex_deepseek_v4_auxiliary_spec *aux =
        coverage_family_ir()->auxiliary_at(ir, 0u);
    yvex_tensor_scope scope = YVEX_TENSOR_SCOPE_GLOBAL;
    unsigned long long layer = YVEX_DEEPSEEK_TENSOR_NO_INDEX;
    unsigned long long expert = YVEX_DEEPSEEK_TENSOR_NO_INDEX;
    const char *tail = NULL;
    const char *expert_text;
    int parsed;

    if (!source || !source->name || !model) {
        return coverage_reject(
            builder->failure,
            YVEX_DEEPSEEK_COVERAGE_FAILURE_UNEXPECTED_SOURCE,
            YVEX_TENSOR_COLLECTION_GLOBAL, scope, "unknown",
            layer, expert, YVEX_DEEPSEEK_TENSOR_NO_INDEX, 0u, 1u,
            builder->err);
    }
    if (strncmp(source->name, "layers.", 7u) == 0) {
        scope = YVEX_TENSOR_SCOPE_MAIN_LAYER;
        parsed = coverage_parse_name_index(source->name + 7u, &layer, &tail);
        if (parsed < 0) goto arithmetic_overflow;
        if (parsed > 0 && *tail == '.' && layer >= model->main_layer_count) {
            return coverage_reject(
                builder->failure,
                YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_INDEX,
                YVEX_TENSOR_COLLECTION_GLOBAL, scope, source->name,
                layer, expert, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
                model->main_layer_count - 1u, layer, builder->err);
        }
        if (parsed > 0 && *tail == '.')
            layer_spec = coverage_family_ir()->layer_at(ir, layer);
    } else if (strncmp(source->name, "mtp.", 4u) == 0) {
        unsigned long long predictor = 0u;
        scope = YVEX_TENSOR_SCOPE_MTP;
        parsed = coverage_parse_name_index(source->name + 4u,
                                           &predictor, &tail);
        if (parsed < 0) goto arithmetic_overflow;
        layer = aux ? aux->layer.layer_index : YVEX_DEEPSEEK_TENSOR_NO_INDEX;
        if (parsed > 0 && *tail == '.' &&
            (!aux || predictor != aux->predictor_index)) {
            return coverage_reject(
                builder->failure,
                YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_INDEX,
                YVEX_TENSOR_COLLECTION_AUXILIARY, scope,
                source->name, layer, predictor,
                YVEX_DEEPSEEK_TENSOR_NO_INDEX,
                aux ? aux->predictor_index : 0u, predictor, builder->err);
        }
        if (parsed > 0 && *tail == '.' && aux) layer_spec = &aux->layer;
    }
    expert_text = tail ? strstr(tail, ".ffn.experts.") : NULL;
    if (expert_text && layer_spec) {
        parsed = coverage_parse_name_index(
            expert_text + strlen(".ffn.experts."), &expert, &tail);
        if (parsed < 0) goto arithmetic_overflow;
        if (parsed > 0 && *tail == '.' &&
            expert >= layer_spec->moe.routed_experts) {
            return coverage_reject(
                builder->failure,
                YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_INDEX,
                YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, scope,
                source->name, layer, expert,
                YVEX_DEEPSEEK_TENSOR_NO_INDEX,
                layer_spec->moe.routed_experts - 1u, expert, builder->err);
        }
    }
    return coverage_reject(
        builder->failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_UNEXPECTED_SOURCE,
        YVEX_TENSOR_COLLECTION_GLOBAL, scope, source->name,
        layer, expert, YVEX_DEEPSEEK_TENSOR_NO_INDEX, 0u, 1u, builder->err);

arithmetic_overflow:
    return coverage_reject(
        builder->failure,
        YVEX_DEEPSEEK_COVERAGE_FAILURE_ARITHMETIC_OVERFLOW,
        YVEX_TENSOR_COLLECTION_GLOBAL, scope, source->name,
        layer, expert, YVEX_DEEPSEEK_TENSOR_NO_INDEX, ULLONG_MAX, 0u,
        builder->err);
}

static int coverage_validate_inputs(
    const yvex_source_verification *verification,
    const yvex_deepseek_v4_ir *ir,
    yvex_source_tensor_snapshot *snapshot,
    yvex_source_tensor_snapshot_facts *facts,
    yvex_deepseek_tensor_coverage_failure *failure,
    yvex_error *err)
{
    const yvex_source_target_identity *identity =
        yvex_source_release_identity();
    const yvex_deepseek_v4_model_spec *model = coverage_family_ir()->model(ir);

    if (!verification || !ir || !snapshot || !model) {
        return coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_ARGUMENT,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "coverage-input",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, err);
    }
    if (!verification->verified || verification->blocker_count != 0u ||
        strcmp(verification->manifest_target_id, identity->target_id) != 0 ||
        strcmp(verification->repository_id, identity->upstream_repo_id) != 0 ||
        strcmp(verification->revision, identity->upstream_revision) != 0 ||
        strcmp(model->target_id, identity->target_id) != 0 ||
        strcmp(model->revision, identity->upstream_revision) != 0) {
        return coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_SOURCE_IDENTITY,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "pinned-source-identity",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, err);
    }
    if (strcmp(verification->inventory_authority, "upstream-index") != 0 ||
        !verification->upstream_index_identity_verified ||
        strcmp(verification->upstream_index_oid,
               identity->upstream_index_oid) != 0 ||
        strcmp(verification->local_index_oid,
               identity->upstream_index_oid) != 0) {
        return coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVENTORY_AUTHORITY,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "pinned-upstream-index",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, err);
    }
    if (yvex_source_tensor_snapshot_facts_get(snapshot, facts, err) != YVEX_OK)
        return yvex_error_code(err);
    if (facts->tensor_count != verification->header_tensor_count ||
        facts->shard_count != verification->header_shard_count ||
        facts->header_scan_count != verification->header_scan_count ||
        facts->payload_bytes_read != 0u) {
        return coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVENTORY_DRIFT,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "snapshot-verification-facts",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, verification->header_tensor_count,
            facts->tensor_count, err);
    }
    if (model->main_layer_count != 43u || model->auxiliary_layer_count != 1u ||
        model->source_constraint.quant_block_rows != 128u ||
        model->source_constraint.quant_block_columns != 128u ||
        model->source_constraint.fp4_packing_factor != 2u ||
        model->source_constraint.fp4_scale_group_width != 32u ||
        model->source_constraint.scale_dtype != YVEX_NATIVE_DTYPE_F8_E8M0 ||
        !model->final_mhc_head.required) {
        return coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_ARCHITECTURE_INCOMPLETE,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "tensor-relevant-ir",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, err);
    }
    return YVEX_OK;
}

static int coverage_build(
    yvex_deepseek_tensor_coverage **out,
    const yvex_source_verification *verification,
    const yvex_deepseek_v4_ir *ir,
    yvex_source_tensor_snapshot *snapshot,
    const yvex_deepseek_tensor_coverage_options *options,
    yvex_deepseek_tensor_coverage_failure *failure,
    yvex_error *err)
{
    yvex_deepseek_tensor_coverage_options actual;
    yvex_source_tensor_snapshot_facts source_facts;
    yvex_deepseek_tensor_coverage *coverage;
    coverage_builder builder;
    unsigned long long index;
    unsigned long long hash = 1469598103934665603ull;
    int rc;

    if (!out) return coverage_reject(
        failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_ARGUMENT,
        YVEX_TENSOR_COLLECTION_GLOBAL,
        YVEX_TENSOR_SCOPE_GLOBAL, "output",
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, err);
    *out = NULL;
    coverage_failure_clear(failure);
    memset(&source_facts, 0, sizeof(source_facts));
    rc = coverage_validate_inputs(verification, ir, snapshot, &source_facts,
                                  failure, err);
    if (rc != YVEX_OK) return rc;
    actual.allocate = coverage_allocate;
    actual.release = coverage_release;
    actual.context = NULL;
    actual.maximum_tensors = DEEPSEEK_COVERAGE_DEFAULT_LIMIT;
    if (options) {
        if ((options->allocate == NULL) != (options->release == NULL)) {
            return coverage_reject(
                failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_ARGUMENT,
                YVEX_TENSOR_COLLECTION_GLOBAL,
                YVEX_TENSOR_SCOPE_GLOBAL, "allocator",
                YVEX_DEEPSEEK_TENSOR_NO_INDEX,
                YVEX_DEEPSEEK_TENSOR_NO_INDEX,
                YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, err);
        }
        if (options->allocate) {
            actual.allocate = options->allocate;
            actual.release = options->release;
            actual.context = options->context;
        }
        if (options->maximum_tensors)
            actual.maximum_tensors = options->maximum_tensors;
    }
    if (source_facts.tensor_count > actual.maximum_tensors ||
        source_facts.tensor_count > (unsigned long long)(SIZE_MAX /
                                     sizeof(yvex_deepseek_tensor_coverage_row))) {
        return coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_RESOURCE_LIMIT,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "tensor-count",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, actual.maximum_tensors,
            source_facts.tensor_count, err);
    }
    coverage = (yvex_deepseek_tensor_coverage *)actual.allocate(
        sizeof(*coverage), actual.context);
    if (!coverage) return coverage_reject(
        failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_ALLOCATION,
        YVEX_TENSOR_COLLECTION_GLOBAL,
        YVEX_TENSOR_SCOPE_GLOBAL, "coverage",
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, sizeof(*coverage), 0u, err);
    memset(coverage, 0, sizeof(*coverage));
    coverage->options = actual;
    coverage->rows = (yvex_deepseek_tensor_coverage_row *)actual.allocate(
        (size_t)source_facts.tensor_count * sizeof(coverage->rows[0]),
        actual.context);
    coverage->row_by_source =
        (const yvex_deepseek_tensor_coverage_row **)actual.allocate(
            (size_t)source_facts.tensor_count *
                sizeof(coverage->row_by_source[0]), actual.context);
    builder.matched = (unsigned char *)actual.allocate(
        (size_t)source_facts.tensor_count, actual.context);
    if (!coverage->rows || !coverage->row_by_source || !builder.matched) {
        if (builder.matched) actual.release(builder.matched, actual.context);
        coverage_close(coverage);
        return coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_ALLOCATION,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "coverage-tables",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, source_facts.tensor_count, 0u, err);
    }
    memset(coverage->rows, 0, (size_t)source_facts.tensor_count *
                                    sizeof(coverage->rows[0]));
    memset(coverage->row_by_source, 0, (size_t)source_facts.tensor_count *
                                             sizeof(coverage->row_by_source[0]));
    memset(builder.matched, 0, (size_t)source_facts.tensor_count);
    coverage->snapshot = snapshot;
    yvex_source_tensor_snapshot_retain(snapshot);
    coverage->summary.source_tensor_count = source_facts.tensor_count;
    coverage->summary.main_layer_count = 43u;
    coverage->summary.auxiliary_layer_count = 1u;
    coverage->summary.header_scan_count = source_facts.header_scan_count;
    coverage->summary.payload_bytes_read = source_facts.payload_bytes_read;
    coverage->summary.source_identity = source_facts.identity;
    builder.coverage = coverage;
    builder.failure = failure;
    builder.err = err;
    rc = coverage_build_requirements(&builder, ir);
    if (rc == YVEX_OK) {
        for (index = 0u; index < source_facts.tensor_count; ++index) {
            if (!builder.matched[index]) {
                const yvex_native_weight_info *unexpected =
                    yvex_source_tensor_snapshot_at(snapshot, index);
                coverage->summary.unexpected_count++;
                rc = coverage_reject_unexpected(&builder, ir, unexpected);
                break;
            }
        }
    }
    actual.release(builder.matched, actual.context);
    if (rc != YVEX_OK) {
        coverage_close(coverage);
        return rc;
    }
    if (coverage->summary.required_tensor_count != source_facts.tensor_count ||
        coverage->summary.matched_tensor_count != source_facts.tensor_count) {
        unsigned long long matched = coverage->summary.matched_tensor_count;
        coverage_close(coverage);
        return coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVENTORY_DRIFT,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "one-to-one-count",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, source_facts.tensor_count,
            matched, err);
    }
    hash = yvex_core_hash_mix_u64(hash, source_facts.identity);
    for (index = 0u; index < coverage->summary.required_tensor_count; ++index) {
        const yvex_native_weight_info *source = coverage->rows[index].source;
        const char *collection = coverage_collection_name(
            coverage->rows[index].collection);
        const char *scope = coverage_scope_name(coverage->rows[index].scope);
        hash = yvex_core_hash_mix_bytes(hash, source->name, strlen(source->name) + 1u);
        hash = yvex_core_hash_mix_bytes(hash, collection, strlen(collection) + 1u);
        hash = yvex_core_hash_mix_bytes(hash, scope, strlen(scope) + 1u);
        hash = yvex_core_hash_mix_u64(hash, coverage->rows[index].layer_index);
        hash = yvex_core_hash_mix_u64(hash, coverage->rows[index].expert_index);
    }
    coverage->summary.routed_expert_count =
        coverage->summary.collection_counts[
            YVEX_TENSOR_COLLECTION_ROUTED_EXPERT];
    coverage->summary.shared_expert_count =
        coverage->summary.collection_counts[
            YVEX_TENSOR_COLLECTION_SHARED_EXPERT];
    if (yvex_source_tensor_snapshot_facts_get(snapshot, &source_facts, err) !=
        YVEX_OK) {
        coverage_close(coverage);
        return yvex_error_code(err);
    }
    coverage->summary.source_lookup_count = source_facts.lookup_count;
    coverage->summary.source_collision_count = source_facts.collision_count;
    coverage->summary.source_maximum_probe = source_facts.maximum_probe;
    coverage->summary.coverage_identity = hash;
    coverage->summary.complete = 1;
    *out = coverage;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int coverage_open_verified_source(
    yvex_deepseek_tensor_coverage **out,
    yvex_source_verification *verification,
    const char *source_path,
    const char *models_root,
    yvex_deepseek_tensor_coverage_failure *failure,
    yvex_error *err)
{
    yvex_source_verify_options source_options;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure ir_failure;
    int rc;

    if (!out || !verification || !source_path || !source_path[0]) {
        return coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_ARGUMENT,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "verified-source-path",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, err);
    }
    *out = NULL;
    memset(&source_options, 0, sizeof(source_options));
    source_options.identity = yvex_source_release_identity();
    source_options.source_path = source_path;
    source_options.models_root = models_root && models_root[0]
                                     ? models_root
                                     : "models";
    rc = yvex_source_verify_with_snapshot(&source_options, verification,
                                          &snapshot, err);
    if (rc != YVEX_OK) goto cleanup;
    if (!verification->verified || !snapshot) {
        rc = coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_SOURCE_IDENTITY,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL, "strict-source-verification",
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, 1u, 0u, err);
        goto cleanup;
    }
    rc = coverage_family_ir()->build(&ir, verification, &ir_failure, err);
    if (rc != YVEX_OK) {
        rc = coverage_reject(
            failure, YVEX_DEEPSEEK_COVERAGE_FAILURE_ARCHITECTURE_INCOMPLETE,
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL,
            ir_failure.field ? ir_failure.field : "architecture-ir",
            ir_failure.layer_index, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, ir_failure.expected,
            ir_failure.actual, err);
        goto cleanup;
    }
    rc = coverage_build(
        out, verification, ir, snapshot, NULL, failure, err);
cleanup:
    coverage_family_ir()->close(ir);
    yvex_source_tensor_snapshot_release(snapshot);
    return rc;
}

static void coverage_close(
    yvex_deepseek_tensor_coverage *coverage)
{
    yvex_deepseek_tensor_coverage_options options;

    if (!coverage) return;
    options = coverage->options;
    yvex_source_tensor_snapshot_release(coverage->snapshot);
    if (coverage->row_by_source)
        options.release((void *)coverage->row_by_source, options.context);
    if (coverage->rows) options.release(coverage->rows, options.context);
    options.release(coverage, options.context);
}

static const yvex_deepseek_tensor_coverage_summary *
coverage_summary(
    const yvex_deepseek_tensor_coverage *coverage)
{
    return coverage ? &coverage->summary : NULL;
}

static const yvex_deepseek_tensor_coverage_row *
coverage_at(
    const yvex_deepseek_tensor_coverage *coverage,
    unsigned long long index)
{
    if (!coverage || index >= coverage->summary.required_tensor_count)
        return NULL;
    return &coverage->rows[index];
}

static const yvex_deepseek_tensor_coverage_row *
coverage_find(
    const yvex_deepseek_tensor_coverage *coverage,
    const char *source_name)
{
    unsigned long long index;

    if (!coverage || !source_name ||
        !yvex_source_tensor_snapshot_find_index(coverage->snapshot,
                                                source_name, &index))
        return NULL;
    return coverage->row_by_source[index];
}

static int coverage_find_index(
    const yvex_deepseek_tensor_coverage *coverage,
    const char *source_name,
    unsigned long long *row_index)
{
    unsigned long long source_index;
    const yvex_deepseek_tensor_coverage_row *row;

    if (!coverage || !source_name || !row_index || !coverage->rows ||
        !yvex_source_tensor_snapshot_find_index(coverage->snapshot,
                                                source_name, &source_index))
        return 0;
    row = coverage->row_by_source[source_index];
    if (!row) return 0;
    *row_index = (unsigned long long)(row - coverage->rows);
    return 1;
}

static int coverage_find_source_index(
    const yvex_deepseek_tensor_coverage *coverage,
    const char *source_name,
    unsigned long long *source_index)
{
    if (!coverage || !source_name || !source_index ||
        !yvex_source_tensor_snapshot_find_index(coverage->snapshot,
                                                source_name, source_index))
        return 0;
    return coverage->row_by_source[*source_index] != NULL;
}

static const char *coverage_collection_name(
    yvex_tensor_collection collection)
{
    return collection < YVEX_TENSOR_COLLECTION_COUNT
               ? coverage_collection_names[collection]
               : "unknown";
}

static const char *coverage_failure_name(
    yvex_deepseek_tensor_coverage_failure_code code)
{
    return code <= YVEX_DEEPSEEK_COVERAGE_FAILURE_ALLOCATION
               ? coverage_failure_names[code]
               : "unknown";
}

/*
 * Publish the immutable exact-coverage operations consumed by the family registration and
 * transformation builder.
 *
 * Returns process-lifetime immutable storage; no allocation or I/O. Coverage proves source
 * ownership but does not read payload bytes.
 */
const yvex_model_family_coverage_api *yvex_model_deepseek_coverage_api(void)
{
    static const yvex_model_family_coverage_api api = {
        coverage_build,
        coverage_open_verified_source,
        coverage_close,
        coverage_summary,
        coverage_at,
        coverage_find,
        coverage_find_index,
        coverage_find_source_index,
        coverage_collection_name,
        coverage_failure_name
    };

    return &api;
}
