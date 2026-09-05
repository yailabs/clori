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
    yvex_semantic_decoder_layer *decoder_layers;
    unsigned long long decoder_layer_count;
    yvex_semantic_component *components;
    unsigned long long component_count;
    yvex_semantic_phase_edge *phase_edges;
    unsigned long long phase_edge_count;
    yvex_semantic_reference *references;
    unsigned long long reference_count;
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

static int semantic_decoder_identity(
    yvex_sha256 *hash, const yvex_semantic_decoder_layer *layers,
    unsigned long long count)
{
    unsigned long long index;

    if (!yvex_sha256_update_text(hash, "decoder") ||
        !yvex_sha256_update_u64(hash, count)) return 0;
    for (index = 0ull; index < count; ++index) {
        const yvex_semantic_decoder_layer *layer = &layers[index];
        char identity[YVEX_SHA256_HEX_BYTES];
        const char *mixer_identity = "full-causal-attention";

        if (layer->mixer == YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA) {
            if (!yvex_semantic_gated_delta_requirement_identity(
                    &layer->gated_delta, identity))
                return 0;
            mixer_identity = identity;
        }
        if (!yvex_sha256_update_u64(hash, layer->ordinal) ||
            !yvex_sha256_update_u64(hash, layer->layer_index) ||
            !yvex_sha256_update_u64(hash, layer->tensor_scope) ||
            !yvex_sha256_update_u64(hash, layer->mixer) ||
            !yvex_sha256_update_u64(hash, layer->feed_forward) ||
            !yvex_sha256_update_u64(hash, layer->hidden_width) ||
            !yvex_sha256_update_u64(hash, layer->intermediate_width) ||
            !yvex_sha256_update_u64(
                hash, layer->normalization_weight_convention) ||
            !semantic_hash_f64(hash, layer->normalization_epsilon) ||
            !yvex_sha256_update_u64(hash,
                                    (unsigned int)layer->mixer_output_gate) ||
            !yvex_sha256_update_text(hash, mixer_identity))
            return 0;
    }
    return 1;
}

static int semantic_composite_identity(
    yvex_sha256 *hash, const yvex_semantic_model_ir *model)
{
    const yvex_semantic_composite_summary *summary = &model->summary.composite;
    unsigned long long index;

    if (!summary->component_count) return 1;
    if (!yvex_sha256_update_text(hash, "composite") ||
        !yvex_sha256_update_text(hash, summary->repository) ||
        !yvex_sha256_update_text(hash, summary->revision) ||
        !yvex_sha256_update_text(hash, summary->subtree) ||
        !yvex_sha256_update_text(hash, summary->source_snapshot_identity) ||
        !yvex_sha256_update_text(hash, summary->component_manifest_identity) ||
        !yvex_sha256_update_text(hash, summary->phase_dag_identity) ||
        !yvex_sha256_update_text(hash, summary->architecture_identity) ||
        !yvex_sha256_update_text(hash, summary->role_map_identity) ||
        !yvex_sha256_update_text(hash, summary->unresolved_requirements_identity) ||
        !yvex_sha256_update_u64(hash, summary->component_count) ||
        !yvex_sha256_update_u64(hash, summary->weighted_component_count) ||
        !yvex_sha256_update_u64(hash, summary->phase_edge_count) ||
        !yvex_sha256_update_u64(hash, summary->shards) ||
        !yvex_sha256_update_u64(hash, summary->tensors) ||
        !yvex_sha256_update_u64(hash, summary->elements) ||
        !yvex_sha256_update_u64(hash, summary->payload_bytes)) return 0;
    for (index = 0ull; index < model->component_count; ++index) {
        const yvex_semantic_component *component = &model->components[index];

        if (!yvex_sha256_update_text(hash, component->canonical_id) ||
            !yvex_sha256_update_text(hash, component->identity) ||
            !yvex_sha256_update_u64(hash, component->shards) ||
            !yvex_sha256_update_u64(hash, component->tensors) ||
            !yvex_sha256_update_u64(hash, component->phase) ||
            !yvex_sha256_update_u64(hash, (unsigned int)component->weighted) ||
            !yvex_sha256_update_u64(hash, (unsigned int)component->release_after_phase))
            return 0;
    }
    for (index = 0ull; index < model->phase_edge_count; ++index) {
        const yvex_semantic_phase_edge *edge = &model->phase_edges[index];

        if (!yvex_sha256_update_u64(hash, edge->source_phase) ||
            !yvex_sha256_update_u64(hash, edge->destination_phase) ||
            !yvex_sha256_update_u64(hash, edge->data_classes) ||
            !yvex_sha256_update_u64(hash, edge->lifetime)) return 0;
    }
    return 1;
}

static int semantic_identity(yvex_semantic_model_ir *model)
{
    yvex_semantic_model_ir_summary *summary = &model->summary;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, summary->schema_version == YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2
                       ? "yvex.semantic-model-ir.v2"
                       : "yvex.semantic-model-ir.v1") ||
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
        (summary->schema_version == YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2 &&
         !semantic_decoder_identity(
             &hash, model->decoder_layers, model->decoder_layer_count)) ||
        !semantic_composite_identity(&hash, model) ||
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

static int semantic_decoder_layers_build(
    yvex_semantic_decoder_layer **out, const void *context,
    yvex_semantic_decoder_layer_fn project,
    unsigned long long layer_count, yvex_error *err)
{
    unsigned long long index;

    *out = NULL;
    if (!layer_count) return YVEX_OK;
    if (!context || !project || layer_count > SIZE_MAX / sizeof(**out))
        return semantic_refuse(
            err, YVEX_ERR_BOUNDS,
            "semantic decoder topology exceeds platform capacity");
    *out = calloc((size_t)layer_count, sizeof(**out));
    if (!*out)
        return semantic_refuse(
            err, YVEX_ERR_NOMEM,
            "semantic decoder topology allocation failed");
    for (index = 0ull; index < layer_count; ++index) {
        if (!project(context, index, &(*out)[index]) ||
            (*out)[index].ordinal != index) {
            free(*out);
            *out = NULL;
            return semantic_refuse(
                err, YVEX_ERR_FORMAT,
                "semantic decoder topology projection failed");
        }
    }
    return YVEX_OK;
}

static int semantic_decoder_validate(
    const yvex_semantic_model_ir *model, yvex_error *err)
{
    const yvex_model_execution_descriptor *execution =
        &model->summary.execution_descriptor;
    unsigned long long index, attention = 0ull, recurrent = 0ull;

    if (model->summary.schema_version != YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2 ||
        execution->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2 ||
        model->decoder_layer_count != execution->layer_count)
        return semantic_refuse(
            err, YVEX_ERR_FORMAT,
            "versioned decoder topology and execution geometry disagree");
    for (index = 0ull; index < model->decoder_layer_count; ++index) {
        const yvex_semantic_decoder_layer *layer = &model->decoder_layers[index];
        char identity[YVEX_SHA256_HEX_BYTES];

        if (layer->layer_index != index ||
            layer->tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER ||
            layer->feed_forward !=
                YVEX_SEMANTIC_DECODER_FFN_DENSE_SILU_GATED ||
            layer->hidden_width != execution->hidden_width ||
            layer->intermediate_width != execution->dense_ffn_width ||
            (layer->normalization_weight_convention !=
                 YVEX_NORMALIZATION_WEIGHT_DIRECT &&
             layer->normalization_weight_convention !=
                 YVEX_NORMALIZATION_WEIGHT_ONE_PLUS) ||
            !isfinite(layer->normalization_epsilon) ||
            layer->normalization_epsilon <= 0.0 ||
            (layer->mixer_output_gate != 0 && layer->mixer_output_gate != 1))
            return semantic_refuse(
                err, YVEX_ERR_FORMAT,
                "one semantic decoder layer has invalid geometry");
        if (layer->mixer ==
            YVEX_SEMANTIC_DECODER_MIXER_FULL_CAUSAL_ATTENTION) {
            if (layer->gated_delta.schema_version != 0u ||
                attention >= model->attention_layer_count ||
                model->attention_layers[attention].layer_index != index)
                return semantic_refuse(
                    err, YVEX_ERR_FORMAT,
                    "full-causal decoder layer has no matching attention semantics");
            attention++;
        } else if (layer->mixer ==
                   YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA) {
            if (!yvex_semantic_gated_delta_requirement_identity(
                    &layer->gated_delta, identity))
                return semantic_refuse(
                    err, YVEX_ERR_FORMAT,
                    "stateful decoder layer has invalid gated-delta semantics");
            recurrent++;
        } else {
            return semantic_refuse(
                err, YVEX_ERR_FORMAT,
                "semantic decoder token mixer is unsupported");
        }
    }
    if (attention != model->attention_layer_count ||
        recurrent != execution->sequence_mixer_layers)
        return semantic_refuse(
            err, YVEX_ERR_FORMAT,
            "decoder mixer populations disagree with semantic execution facts");
    return YVEX_OK;
}

static int semantic_composite_build(
    yvex_semantic_model_ir *model,
    const yvex_semantic_composite_request *source, yvex_error *err)
{
    yvex_semantic_composite_summary *summary = &model->summary.composite;
    unsigned long long index;

    if (!source) return YVEX_OK;
    if (!source->repository || !source->repository[0] || !source->revision ||
        !source->revision[0] || !source->subtree || !source->subtree[0] ||
        !yvex_sha256_hex_valid(source->source_snapshot_identity) ||
        !yvex_sha256_hex_valid(source->component_manifest_identity) ||
        !yvex_sha256_hex_valid(source->phase_dag_identity) ||
        !yvex_sha256_hex_valid(source->architecture_identity) ||
        !yvex_sha256_hex_valid(source->role_map_identity) ||
        !yvex_sha256_hex_valid(source->unresolved_requirements_identity) ||
        !source->component_count || source->weighted_components > source->component_count ||
        !source->components || (!!source->phase_edges != !!source->phase_edge_count))
        return semantic_refuse(err, YVEX_ERR_INVALID_ARG,
                               "composite semantic topology is incomplete");
    for (index = 0ull; index < source->component_count; ++index)
        if (!source->components[index].canonical_id[0] ||
            !yvex_sha256_hex_valid(source->components[index].identity))
            return semantic_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "composite component identity is incomplete");
    model->components = calloc((size_t)source->component_count, sizeof(*model->components));
    if (source->phase_edge_count)
        model->phase_edges = calloc((size_t)source->phase_edge_count,
                                    sizeof(*model->phase_edges));
    if (!model->components || (source->phase_edge_count && !model->phase_edges))
        return semantic_refuse(err, YVEX_ERR_NOMEM,
                               "composite semantic topology allocation failed");
    memcpy(model->components, source->components,
           (size_t)source->component_count * sizeof(*model->components));
    if (source->phase_edge_count)
        memcpy(model->phase_edges, source->phase_edges,
               (size_t)source->phase_edge_count * sizeof(*model->phase_edges));
    model->component_count = source->component_count;
    model->phase_edge_count = source->phase_edge_count;
#define COPY(field) yvex_core_text_copy(summary->field, sizeof(summary->field), source->field)
    COPY(repository); COPY(revision); COPY(subtree); COPY(source_snapshot_identity);
    COPY(component_manifest_identity); COPY(phase_dag_identity); COPY(architecture_identity);
    COPY(role_map_identity); COPY(unresolved_requirements_identity);
#undef COPY
    summary->component_count = source->component_count;
    summary->weighted_component_count = source->weighted_components;
    summary->phase_edge_count = source->phase_edge_count;
    summary->shards = source->shards;
    summary->tensors = source->tensors;
    summary->elements = source->elements;
    summary->payload_bytes = source->payload_bytes;
    return YVEX_OK;
}

static int semantic_references_build(
    yvex_semantic_model_ir *model, const yvex_semantic_model_ir_request *request,
    yvex_error *err)
{
    unsigned long long index, prior;

    if (!request->reference_count)
        return request->references
                   ? semantic_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "semantic references have no count")
                   : YVEX_OK;
    if (!request->references ||
        request->reference_count > (unsigned long long)SIZE_MAX / sizeof(*model->references))
        return semantic_refuse(err, YVEX_ERR_INVALID_ARG,
                               "semantic references are unbounded");
    model->references = calloc((size_t)request->reference_count, sizeof(*model->references));
    if (!model->references)
        return semantic_refuse(err, YVEX_ERR_NOMEM,
                               "semantic reference allocation failed");
    for (index = 0ull; index < request->reference_count; ++index) {
        const yvex_semantic_reference_request *source = &request->references[index];

        if (!source->key || !source->key[0] || !source->value || !source->value[0] ||
            strlen(source->key) >= sizeof(model->references[index].key) ||
            strlen(source->value) >= sizeof(model->references[index].value))
            return semantic_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "semantic reference is invalid");
        for (prior = 0ull; prior < index; ++prior)
            if (strcmp(model->references[prior].key, source->key) == 0)
                return semantic_refuse(err, YVEX_ERR_INVALID_ARG,
                                       "semantic reference key is duplicated");
        yvex_core_text_copy(model->references[index].key,
                            sizeof(model->references[index].key), source->key);
        yvex_core_text_copy(model->references[index].value,
                            sizeof(model->references[index].value), source->value);
    }
    model->reference_count = request->reference_count;
    return YVEX_OK;
}

static void semantic_model_release(yvex_semantic_model_ir *model)
{
    if (!model) return;
    free(model->attention_layers);
    free(model->draft_attention_layers);
    free(model->decoder_layers);
    free(model->components);
    free(model->phase_edges);
    free(model->references);
    memset(model, 0, sizeof(*model));
    free(model);
}

int yvex_semantic_model_ir_seal(
    yvex_semantic_model_ir **out,
    const yvex_semantic_model_ir_request *request, yvex_error *err)
{
    yvex_semantic_model_ir *model;
    if (out) *out = NULL;
    if (!out || !request ||
        (request->schema_version != YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1 &&
         request->schema_version != YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2) ||
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
         !request->attention_context) ||
        (!!request->decoder_layer != !!request->decoder_layer_count) ||
        (request->decoder_layer_count && !request->decoder_context) ||
        (request->schema_version == YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1 &&
         request->decoder_layer_count) ||
        (request->schema_version == YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2 &&
         !request->decoder_layer_count))
        return semantic_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "complete immutable semantic facts and balanced payload ownership are required");
    int rc = semantic_execution_validate(request, err);
    if (rc != YVEX_OK) return rc;
    model = calloc(1u, sizeof(*model));
    if (!model)
        return semantic_refuse(err, YVEX_ERR_NOMEM, "semantic model allocation failed");
    model->summary.schema_version = request->schema_version;
    model->summary.family_adapter_id = request->family_adapter_id;
    model->summary.family_adapter_version = request->family_adapter_version;
    model->summary.attention_layer_count = request->attention_layer_count;
    model->summary.draft_attention_layer_count = request->draft_attention_layer_count;
    model->summary.decoder_layer_count = request->decoder_layer_count;
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
    if (rc == YVEX_OK)
        rc = semantic_decoder_layers_build(
            &model->decoder_layers, request->decoder_context,
            request->decoder_layer, request->decoder_layer_count, err);
    model->attention_layer_count = request->attention_layer_count;
    model->draft_attention_layer_count = request->draft_attention_layer_count;
    model->decoder_layer_count = request->decoder_layer_count;
    if (rc == YVEX_OK && request->schema_version ==
                             YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2)
        rc = semantic_decoder_validate(model, err);
    if (rc == YVEX_OK)
        rc = semantic_composite_build(model, request->composite, err);
    if (rc == YVEX_OK)
        rc = semantic_references_build(model, request, err);
    if (rc != YVEX_OK) {
        semantic_model_release(model);
        return rc;
    }
    model->attention_layer_count = request->attention_layer_count;
    model->draft_attention_layer_count = request->draft_attention_layer_count;
    if (!semantic_identity(model)) {
        semantic_model_release(model);
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

int yvex_semantic_model_ir_decoder_view(
    const yvex_semantic_model_ir *model,
    const yvex_semantic_decoder_layer **layers,
    unsigned long long *layer_count)
{
    if (layers) *layers = NULL;
    if (layer_count) *layer_count = 0ull;
    if (!model || !layers || !layer_count || !model->decoder_layers ||
        !model->decoder_layer_count)
        return 0;
    *layers = model->decoder_layers;
    *layer_count = model->decoder_layer_count;
    return 1;
}

int yvex_semantic_model_ir_composite_view(
    const yvex_semantic_model_ir *model,
    const yvex_semantic_component **components, unsigned long long *component_count,
    const yvex_semantic_phase_edge **phase_edges, unsigned long long *phase_edge_count)
{
    if (components) *components = NULL;
    if (component_count) *component_count = 0ull;
    if (phase_edges) *phase_edges = NULL;
    if (phase_edge_count) *phase_edge_count = 0ull;
    if (!model || !components || !component_count || !phase_edges || !phase_edge_count ||
        !model->components || !model->component_count) return 0;
    *components = model->components;
    *component_count = model->component_count;
    *phase_edges = model->phase_edges;
    *phase_edge_count = model->phase_edge_count;
    return 1;
}

const char *yvex_semantic_model_ir_reference(
    const yvex_semantic_model_ir *model, const char *key)
{
    unsigned long long index;

    if (!model || !key || !key[0]) return NULL;
    for (index = 0ull; index < model->reference_count; ++index)
        if (strcmp(model->references[index].key, key) == 0)
            return model->references[index].value;
    return NULL;
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
    semantic_model_release(owner);
    *model = NULL;
}
