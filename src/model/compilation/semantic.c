/*
 * Seal the family-neutral semantic identity and context capability before graph lowering.
 *
 * The sealed representation owns compiler-projected topology and carries no process-local family
 * implementation state across the compiler boundary.
 */
#include <yvex/internal/compiler.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct yvex_semantic_model_ir {
    yvex_semantic_model_ir_summary summary;
    yvex_semantic_attention_layer *attention_layers;
    unsigned long long attention_layer_count;
    yvex_semantic_attention_layer *draft_attention_layers;
    unsigned long long draft_attention_layer_count;
};

static int semantic_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "compilation.semantic-model", reason);
    return status;
}

static int semantic_hash_f64(yvex_sha256 *hash, double value)
{
    uint64_t bits;

    if (!isfinite(value)) return 0;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

static int semantic_layer_geometry_identity(
    yvex_sha256 *hash, const yvex_semantic_attention_layer *layer)
{
    return yvex_sha256_update_u64(hash, layer->ordinal) &&
           yvex_sha256_update_u64(hash, layer->layer_index) &&
           yvex_sha256_update_u64(hash, layer->predictor_index) &&
           yvex_sha256_update_u64(hash, layer->tensor_scope) &&
           yvex_sha256_update_u64(hash, layer->attention_class) &&
           yvex_sha256_update_u64(hash, layer->compute_contract) &&
           yvex_sha256_update_u64(hash, layer->compression_ratio) &&
           yvex_sha256_update_u64(hash, layer->sliding_window) &&
           yvex_sha256_update_u64(hash, layer->query_heads) &&
           yvex_sha256_update_u64(hash, layer->kv_heads) &&
           yvex_sha256_update_u64(hash, layer->head_dimension) &&
           yvex_sha256_update_u64(hash, layer->rope_head_dimension) &&
           yvex_sha256_update_u64(hash, layer->query_lora_rank) &&
           yvex_sha256_update_u64(hash, layer->output_lora_rank) &&
           yvex_sha256_update_u64(hash, layer->output_groups) &&
           yvex_sha256_update_u64(hash, layer->output_group_input_width) &&
           yvex_sha256_update_u64(hash, layer->hidden_dimension) &&
           yvex_sha256_update_u64(hash, layer->indexer_heads) &&
           yvex_sha256_update_u64(hash, layer->indexer_head_dimension) &&
           yvex_sha256_update_u64(hash, layer->indexer_topk) &&
           yvex_sha256_update_u64(hash, layer->compressor_ape_columns) &&
           yvex_sha256_update_u64(hash, layer->indexer_ape_columns) &&
           semantic_hash_f64(hash, layer->rms_norm_epsilon);
}

static int semantic_layer_mhc_identity(
    yvex_sha256 *hash, const yvex_semantic_attention_layer *layer)
{
    return yvex_sha256_update_u64(hash, layer->residual_stream_count) &&
           yvex_sha256_update_u64(hash, layer->residual_stream_width) &&
           yvex_sha256_update_u64(hash, layer->residual_expanded_width) &&
           yvex_sha256_update_u64(hash, layer->mhc_mixing_rows) &&
           yvex_sha256_update_u64(hash, layer->mhc_mixing_columns) &&
           yvex_sha256_update_u64(hash, layer->mhc_base_width) &&
           yvex_sha256_update_u64(hash, layer->mhc_scale_width) &&
           yvex_sha256_update_u64(hash, layer->mhc_sinkhorn_iterations) &&
           yvex_sha256_update_u64(hash, layer->attention_input_norm_width) &&
           semantic_hash_f64(hash, layer->mhc_epsilon) &&
           semantic_hash_f64(hash, layer->mhc_residual_post_multiplier) &&
           yvex_sha256_update_u64(hash, layer->mhc_entry_policy) &&
           yvex_sha256_update_u64(hash, layer->mhc_attention_pre_and_post) &&
           yvex_sha256_update_u64(hash, layer->attention_input_norm_required) &&
           yvex_sha256_update_u64(hash, layer->attention_input_norm_role) &&
           yvex_sha256_update_u64(hash, layer->mhc_function_role) &&
           yvex_sha256_update_u64(hash, layer->mhc_base_role) &&
           yvex_sha256_update_u64(hash, layer->mhc_scale_role) &&
           yvex_sha256_update_u64(hash, layer->compressor_required) &&
           yvex_sha256_update_u64(hash, layer->indexer_required);
}

static int semantic_topology_identity(
    yvex_sha256 *hash, const char *scope,
    const yvex_semantic_attention_layer *layers, unsigned long long count)
{
    unsigned long long index;

    if (!yvex_sha256_update_text(hash, scope) ||
        !yvex_sha256_update_u64(hash, count)) return 0;
    for (index = 0ull; index < count; ++index) {
        const yvex_semantic_attention_layer *layer = &layers[index];
        if (!semantic_layer_geometry_identity(hash, layer) ||
            !semantic_layer_mhc_identity(hash, layer) ||
            !yvex_model_position_identity_update(hash, &layer->position) ||
            !yvex_model_activation_identity_update(hash, &layer->attention_kv_activation) ||
            !yvex_model_activation_identity_update(hash, &layer->compressor_activation) ||
            !yvex_model_activation_identity_update(
                hash, &layer->compressor_rotated_activation) ||
            !yvex_model_activation_identity_update(hash, &layer->indexer_query_activation) ||
            !yvex_model_topk_identity_update(hash, &layer->sparse_topk)) return 0;
    }
    return 1;
}

static int semantic_identity(yvex_semantic_model_ir *model)
{
    yvex_semantic_model_ir_summary *summary = &model->summary;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.semantic-model-ir.v1") ||
        !yvex_sha256_update_u64(&hash, summary->schema_version) ||
        !yvex_sha256_update_u64(&hash, summary->family_adapter_id) ||
        !yvex_sha256_update_u64(&hash, summary->family_adapter_version) ||
        !yvex_sha256_update_text(&hash, summary->target_id) ||
        !yvex_sha256_update_text(&hash, summary->source_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->logical_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->semantic_payload_identity) ||
        !yvex_sha256_update_u64(&hash, summary->execution_descriptor.schema_version != 0u) ||
        !yvex_sha256_update_u64(&hash, summary->execution_descriptor.maximum_context) ||
        !yvex_sha256_update_u64(&hash, summary->execution_descriptor.original_context) ||
        !yvex_sha256_update_u64(&hash, summary->attention_layer_count) ||
        !yvex_sha256_update_u64(&hash, summary->draft_attention_layer_count) ||
        !yvex_sha256_update_u64(&hash, summary->numeric_contract.schema_version) ||
        !yvex_sha256_update_u64(&hash, summary->numeric_contract.numeric_schema_version) ||
        !yvex_sha256_update_text(&hash, summary->numeric_contract.identity) ||
        !yvex_sha256_update_text(&hash, summary->numeric_contract.algorithm_revision) ||
        !yvex_sha256_update_u64(&hash, summary->numeric_contract.compute_policy_count) ||
        !yvex_sha256_update_u64(&hash, summary->numeric_contract.activation_policy_count) ||
        !yvex_sha256_update_u64(&hash, summary->numeric_contract.sparse_topk_policy_count) ||
        !semantic_topology_identity(
            &hash, "main", model->attention_layers,
            model->attention_layer_count) ||
        !semantic_topology_identity(
            &hash, "draft", model->draft_attention_layers,
            model->draft_attention_layer_count) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, summary->identity);
    return 1;
}

static int semantic_execution_validate(
    const yvex_semantic_model_ir_request *request, yvex_error *err)
{
    unsigned char wire[YVEX_MODEL_EXECUTION_WIRE_BYTES];
    yvex_model_execution_descriptor decoded;
    const yvex_model_execution_descriptor *execution = request->execution_descriptor;
    int rc;

    if (request->numeric_contract &&
        (request->numeric_contract->schema_version !=
             YVEX_SEMANTIC_NUMERIC_CONTRACT_SCHEMA_V1 ||
         !request->numeric_contract->numeric_schema_version ||
         !yvex_sha256_hex_valid(request->numeric_contract->identity) ||
         !request->numeric_contract->algorithm_revision[0]))
        return semantic_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "semantic numeric contract is incomplete");
    if (!execution) return YVEX_OK;
    rc = yvex_model_execution_descriptor_encode(execution, wire, err);
    if (rc == YVEX_OK)
        rc = yvex_model_execution_descriptor_decode(wire, sizeof(wire), &decoded, err);
    if (rc != YVEX_OK) return rc;
    if (strcmp(execution->identity, request->semantic_payload_identity) != 0 ||
        strcmp(execution->source_model_identity, request->source_model_identity) != 0 ||
        strcmp(execution->logical_model_identity, request->logical_model_identity) != 0)
        return semantic_refuse(err, YVEX_ERR_INVALID_ARG,
            "execution geometry must match the sealed semantic identities");
    return YVEX_OK;
}

static int semantic_layers_build(
    yvex_semantic_attention_layer **out,
    const void *context, yvex_semantic_attention_layer_fn project,
    unsigned long long layer_count, yvex_error *err)
{
    unsigned long long index;

    *out = NULL;
    if (!layer_count) return YVEX_OK;
    if (!context || !project || layer_count > SIZE_MAX / sizeof(**out))
        return semantic_refuse(
            err, YVEX_ERR_BOUNDS,
            "semantic attention topology exceeds platform capacity");
    *out = calloc((size_t)layer_count, sizeof(**out));
    if (!*out)
        return semantic_refuse(
            err, YVEX_ERR_NOMEM,
            "semantic attention topology allocation failed");
    for (index = 0ull; index < layer_count; ++index) {
        if (!project(context, index, &(*out)[index]) ||
            (*out)[index].ordinal != index) {
            free(*out);
            *out = NULL;
            return semantic_refuse(
                err, YVEX_ERR_FORMAT,
                "semantic attention topology projection failed");
        }
    }
    return YVEX_OK;
}

int yvex_semantic_model_ir_seal(
    yvex_semantic_model_ir **out,
    const yvex_semantic_model_ir_request *request, yvex_error *err)
{
    yvex_semantic_model_ir *model;
    if (out) *out = NULL;
    if (!out || !request ||
        request->schema_version != YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1 ||
        !request->family_adapter_id || !request->family_adapter_version ||
        !request->target_id || !request->target_id[0] ||
        strlen(request->target_id) >=
            sizeof(((yvex_semantic_model_ir_summary *)0)->target_id) ||
        !yvex_sha256_hex_valid(request->source_model_identity) ||
        !yvex_sha256_hex_valid(request->logical_model_identity) ||
        !yvex_sha256_hex_valid(request->semantic_payload_identity) ||
        (!!request->attention_layer != !!request->attention_layer_count) ||
        (!!request->draft_attention_layer != !!request->draft_attention_layer_count) ||
        ((request->attention_layer_count || request->draft_attention_layer_count) &&
         !request->attention_context))
        return semantic_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "complete immutable semantic facts and balanced payload ownership are required");
    int rc = semantic_execution_validate(request, err);
    if (rc != YVEX_OK) return rc;
    model = calloc(1u, sizeof(*model));
    if (!model)
        return semantic_refuse(err, YVEX_ERR_NOMEM, "semantic model allocation failed");
    model->summary.schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1;
    model->summary.family_adapter_id = request->family_adapter_id;
    model->summary.family_adapter_version = request->family_adapter_version;
    model->summary.attention_layer_count = request->attention_layer_count;
    model->summary.draft_attention_layer_count = request->draft_attention_layer_count;
    if (request->execution_descriptor)
        model->summary.execution_descriptor = *request->execution_descriptor;
    if (request->numeric_contract)
        model->summary.numeric_contract = *request->numeric_contract;
    yvex_core_text_copy(model->summary.target_id, sizeof(model->summary.target_id),
                        request->target_id);
    yvex_core_text_copy(model->summary.source_model_identity,
                        sizeof(model->summary.source_model_identity),
                        request->source_model_identity);
    yvex_core_text_copy(model->summary.logical_model_identity,
                        sizeof(model->summary.logical_model_identity),
                        request->logical_model_identity);
    yvex_core_text_copy(model->summary.semantic_payload_identity,
                        sizeof(model->summary.semantic_payload_identity),
                        request->semantic_payload_identity);
    rc = semantic_layers_build(
        &model->attention_layers, request->attention_context,
        request->attention_layer,
        request->attention_layer_count, err);
    if (rc == YVEX_OK)
        rc = semantic_layers_build(
            &model->draft_attention_layers, request->attention_context,
            request->draft_attention_layer,
            request->draft_attention_layer_count, err);
    if (rc != YVEX_OK) {
        free(model->attention_layers);
        free(model->draft_attention_layers);
        free(model);
        return rc;
    }
    model->attention_layer_count = request->attention_layer_count;
    model->draft_attention_layer_count = request->draft_attention_layer_count;
    if (!semantic_identity(model)) {
        free(model->attention_layers);
        free(model->draft_attention_layers);
        free(model);
        return semantic_refuse(err, YVEX_ERR_STATE,
                               "semantic model identity derivation failed");
    }
    *out = model;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_semantic_model_ir_attention_view(
    const yvex_semantic_model_ir *model, yvex_tensor_scope tensor_scope,
    const yvex_semantic_attention_layer **layers,
    unsigned long long *layer_count)
{
    if (layers) *layers = NULL;
    if (layer_count) *layer_count = 0ull;
    if (!model || !layers || !layer_count ||
        (tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER &&
         tensor_scope != YVEX_TENSOR_SCOPE_DRAFT))
        return 0;
    if (tensor_scope == YVEX_TENSOR_SCOPE_MAIN_LAYER) {
        *layers = model->attention_layers;
        *layer_count = model->attention_layer_count;
    } else {
        *layers = model->draft_attention_layers;
        *layer_count = model->draft_attention_layer_count;
    }
    return *layers != NULL && *layer_count != 0ull;
}

const yvex_semantic_model_ir_summary *yvex_semantic_model_ir_summary_get(
    const yvex_semantic_model_ir *model)
{
    return model ? &model->summary : NULL;
}

void yvex_semantic_model_ir_close(yvex_semantic_model_ir **model)
{
    yvex_semantic_model_ir *owner;
    if (!model || !*model) return;
    owner = *model;
    free(owner->attention_layers);
    free(owner->draft_attention_layers);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *model = NULL;
}
