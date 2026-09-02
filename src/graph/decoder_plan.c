/* Seal heterogeneous decoder topology and its exact recurrent-state projection. */
#include <yvex/internal/decoder_plan.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>

struct yvex_decoder_plan {
    yvex_decoder_plan_summary summary;
    yvex_decoder_layer_plan *layers;
    yvex_sequence_state_binding *recurrent;
};

static int decoder_refuse(yvex_error *err, yvex_status status,
                          const char *reason)
{
    yvex_error_set(err, status, "graph.decoder-plan", reason);
    return status;
}

static int decoder_layer_identity(
    const yvex_decoder_plan_summary *summary, yvex_decoder_layer_plan *layer)
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    uint64_t epsilon;

    memcpy(&epsilon, &layer->normalization_epsilon, sizeof(epsilon));
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.decoder-layer.v1") ||
        !yvex_sha256_update_text(&hash, summary->semantic_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->operator_graph_identity) ||
        !yvex_sha256_update_u64(&hash, layer->ordinal) ||
        !yvex_sha256_update_u64(&hash, layer->layer_index) ||
        !yvex_sha256_update_u64(&hash, layer->attention_ordinal) ||
        !yvex_sha256_update_u64(&hash, layer->mixer) ||
        !yvex_sha256_update_u64(&hash, layer->feed_forward) ||
        !yvex_sha256_update_u64(&hash, layer->hidden_width) ||
        !yvex_sha256_update_u64(&hash, layer->intermediate_width) ||
        !yvex_sha256_update_u64(&hash, epsilon) ||
        !yvex_sha256_update_u64(&hash, (unsigned int)layer->mixer_output_gate) ||
        !yvex_sha256_update_text(
            &hash, layer->mixer == YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA
                       ? layer->gated_delta.identity : "attention") ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, layer->identity);
    return 1;
}

static int decoder_plan_identity(yvex_decoder_plan *plan)
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_decoder_plan_summary *summary = &plan->summary;
    yvex_sha256 hash;
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.decoder-plan.v1") ||
        !yvex_sha256_update_u64(&hash, summary->family_adapter_id) ||
        !yvex_sha256_update_u64(&hash, summary->family_adapter_version) ||
        !yvex_sha256_update_u64(&hash, summary->layer_count) ||
        !yvex_sha256_update_u64(&hash, summary->attention_layer_count) ||
        !yvex_sha256_update_u64(&hash, summary->recurrent_layer_count) ||
        !yvex_sha256_update_u64(&hash, summary->hidden_width) ||
        !yvex_sha256_update_u64(&hash, summary->intermediate_width) ||
        !yvex_sha256_update_u64(&hash, summary->vocabulary_size) ||
        !yvex_sha256_update_u64(&hash, summary->maximum_context) ||
        !yvex_sha256_update_u64(&hash, summary->convolution_state_bytes) ||
        !yvex_sha256_update_u64(&hash, summary->recurrent_state_bytes) ||
        !yvex_sha256_update_text(&hash, summary->logical_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->semantic_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->model_execution_identity) ||
        !yvex_sha256_update_text(&hash, summary->operator_graph_identity))
        return 0;
    for (index = 0ull; index < summary->layer_count; ++index)
        if (!yvex_sha256_update_text(&hash, plan->layers[index].identity))
            return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, summary->decoder_plan_identity);
    return 1;
}

static int decoder_storage_open(
    yvex_decoder_plan **out, unsigned long long layers,
    unsigned long long recurrent, yvex_error *err)
{
    yvex_decoder_plan *plan;

    if (!out || !layers || layers > SIZE_MAX / sizeof(*plan->layers) ||
        recurrent > SIZE_MAX / sizeof(*plan->recurrent))
        return decoder_refuse(
            err, YVEX_ERR_BOUNDS, "decoder plan extent exceeds this host");
    plan = calloc(1u, sizeof(*plan));
    if (plan) plan->layers = calloc((size_t)layers, sizeof(*plan->layers));
    if (plan && recurrent)
        plan->recurrent = calloc((size_t)recurrent, sizeof(*plan->recurrent));
    if (!plan || !plan->layers || (recurrent && !plan->recurrent)) {
        yvex_decoder_plan_close(&plan);
        return decoder_refuse(
            err, YVEX_ERR_NOMEM, "decoder plan allocation failed");
    }
    *out = plan;
    return YVEX_OK;
}

static int decoder_layers_project(
    yvex_decoder_plan *plan, const yvex_semantic_decoder_layer *semantic,
    yvex_error *err)
{
    unsigned long long index, attention = 0ull, recurrent = 0ull;
    unsigned long long convolution_bytes = 0ull, recurrent_bytes = 0ull;

    for (index = 0ull; index < plan->summary.layer_count; ++index) {
        const yvex_semantic_decoder_layer *source = &semantic[index];
        yvex_decoder_layer_plan *layer = &plan->layers[index];

        *layer = (yvex_decoder_layer_plan){
            .schema_version = YVEX_DECODER_LAYER_PLAN_SCHEMA_V1,
            .ordinal = source->ordinal,
            .layer_index = source->layer_index,
            .attention_ordinal = YVEX_DECODER_NO_ATTENTION,
            .mixer = source->mixer,
            .feed_forward = source->feed_forward,
            .hidden_width = source->hidden_width,
            .intermediate_width = source->intermediate_width,
            .normalization_epsilon = source->normalization_epsilon,
            .mixer_output_gate = source->mixer_output_gate};
        if (source->mixer ==
            YVEX_SEMANTIC_DECODER_MIXER_FULL_CAUSAL_ATTENTION) {
            layer->attention_ordinal = attention++;
        } else if (source->mixer ==
                   YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA) {
            yvex_sequence_state_binding *binding = &plan->recurrent[recurrent++];

            if (yvex_gated_delta_plan_seal(
                    &layer->gated_delta, &source->gated_delta, err) != YVEX_OK)
                return yvex_error_code(err);
            binding->layer_index = source->layer_index;
            binding->plan = layer->gated_delta;
            if (!yvex_core_u64_add(
                    convolution_bytes, layer->gated_delta.convolution_state_bytes,
                    &convolution_bytes) ||
                !yvex_core_u64_add(
                    recurrent_bytes, layer->gated_delta.recurrent_state_bytes,
                    &recurrent_bytes))
                return decoder_refuse(
                    err, YVEX_ERR_BOUNDS,
                    "decoder recurrent state accounting overflowed");
        } else {
            return decoder_refuse(
                err, YVEX_ERR_FORMAT, "decoder mixer semantics are unsupported");
        }
        if (!decoder_layer_identity(&plan->summary, layer))
            return decoder_refuse(
                err, YVEX_ERR_STATE, "decoder layer identity failed");
    }
    plan->summary.convolution_state_bytes = convolution_bytes;
    plan->summary.recurrent_state_bytes = recurrent_bytes;
    if (attention != plan->summary.attention_layer_count ||
        recurrent != plan->summary.recurrent_layer_count)
        return decoder_refuse(
            err, YVEX_ERR_FORMAT, "decoder mixer populations changed during planning");
    return YVEX_OK;
}

static int decoder_plan_validate(yvex_decoder_plan *plan, yvex_error *err)
{
    yvex_decoder_plan_summary *summary = plan ? &plan->summary : NULL;
    char summary_identity[YVEX_SHA256_HEX_BYTES] = {0};
    unsigned long long index, attention = 0ull, recurrent = 0ull;
    unsigned long long convolution = 0ull, recurrence = 0ull;

    if (!summary || summary->schema_version != YVEX_DECODER_PLAN_SCHEMA_V1 ||
        !summary->family_adapter_id || !summary->family_adapter_version ||
        !summary->layer_count || !summary->hidden_width ||
        !summary->intermediate_width || !summary->vocabulary_size ||
        !summary->maximum_context ||
        summary->attention_layer_count + summary->recurrent_layer_count !=
            summary->layer_count ||
        !yvex_sha256_hex_valid(summary->logical_model_identity) ||
        !yvex_sha256_hex_valid(summary->semantic_model_identity) ||
        !yvex_sha256_hex_valid(summary->model_execution_identity) ||
        !yvex_sha256_hex_valid(summary->operator_graph_identity) ||
        !yvex_sha256_hex_valid(summary->decoder_plan_identity))
        return decoder_refuse(
            err, YVEX_ERR_FORMAT, "decoder plan summary is malformed");
    yvex_core_text_copy(summary_identity, sizeof(summary_identity),
                        summary->decoder_plan_identity);
    for (index = 0ull; index < summary->layer_count; ++index) {
        yvex_decoder_layer_plan copy = plan->layers[index];
        char layer_identity[YVEX_SHA256_HEX_BYTES];

        if (copy.schema_version != YVEX_DECODER_LAYER_PLAN_SCHEMA_V1 ||
            copy.ordinal != index || copy.layer_index != index ||
            copy.feed_forward != YVEX_SEMANTIC_DECODER_FFN_DENSE_SILU_GATED ||
            copy.hidden_width != summary->hidden_width ||
            copy.intermediate_width != summary->intermediate_width ||
            !isfinite(copy.normalization_epsilon) ||
            copy.normalization_epsilon <= 0.0 ||
            (copy.mixer_output_gate != 0 && copy.mixer_output_gate != 1))
            goto malformed;
        if (copy.mixer ==
            YVEX_SEMANTIC_DECODER_MIXER_FULL_CAUSAL_ATTENTION) {
            if (copy.attention_ordinal != attention++ ||
                copy.gated_delta.schema_version) goto malformed;
        } else if (copy.mixer ==
                   YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA) {
            const yvex_sequence_state_binding *binding =
                &plan->recurrent[recurrent++];
            if (copy.attention_ordinal != YVEX_DECODER_NO_ATTENTION ||
                yvex_gated_delta_plan_validate(&copy.gated_delta, err) != YVEX_OK ||
                binding->layer_index != index ||
                strcmp(binding->plan.identity, copy.gated_delta.identity) != 0 ||
                !yvex_core_u64_add(
                    convolution, copy.gated_delta.convolution_state_bytes,
                    &convolution) ||
                !yvex_core_u64_add(
                    recurrence, copy.gated_delta.recurrent_state_bytes,
                    &recurrence)) goto malformed;
        } else goto malformed;
        yvex_core_text_copy(layer_identity, sizeof(layer_identity), copy.identity);
        copy.identity[0] = '\0';
        if (!decoder_layer_identity(summary, &copy) ||
            strcmp(copy.identity, layer_identity) != 0) goto malformed;
    }
    if (attention != summary->attention_layer_count ||
        recurrent != summary->recurrent_layer_count ||
        convolution != summary->convolution_state_bytes ||
        recurrence != summary->recurrent_state_bytes)
        goto malformed;
    summary->decoder_plan_identity[0] = '\0';
    if (!decoder_plan_identity(plan) ||
        strcmp(summary->decoder_plan_identity, summary_identity) != 0)
        goto malformed;
    yvex_error_clear(err);
    return YVEX_OK;
malformed:
    yvex_core_text_copy(summary->decoder_plan_identity,
                        sizeof(summary->decoder_plan_identity), summary_identity);
    return decoder_refuse(
        err, YVEX_ERR_FORMAT, "decoder plan layers or identity are malformed");
}

int yvex_decoder_plan_compile(
    yvex_decoder_plan **out, const yvex_semantic_model_ir *semantic_model,
    const yvex_operator_graph_ir *operator_graph, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *semantic =
        yvex_semantic_model_ir_summary_get(semantic_model);
    const yvex_operator_graph_summary *operators =
        yvex_operator_graph_ir_summary(operator_graph);
    const yvex_model_execution_descriptor *execution =
        semantic ? &semantic->execution_descriptor : NULL;
    const yvex_semantic_decoder_layer *layers = NULL;
    unsigned long long layer_count = 0ull;
    yvex_decoder_plan *plan = NULL;
    int rc;

    if (out) *out = NULL;
    if (!out || !semantic || !operators || !execution ||
        semantic->schema_version != YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2 ||
        execution->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2 ||
        !yvex_semantic_model_ir_decoder_view(
            semantic_model, &layers, &layer_count) ||
        layer_count != execution->layer_count ||
        operators->family_adapter_id != semantic->family_adapter_id ||
        operators->family_adapter_version != semantic->family_adapter_version ||
        strcmp(operators->semantic_model_identity, semantic->identity) != 0 ||
        operators->target_layer_count != layer_count ||
        operators->maximum_context != execution->maximum_context ||
        operators->state_class_mask != execution->persistent_state_class_mask)
        return decoder_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "sealed semantic model and matching operator graph are required");
    rc = decoder_storage_open(
        &plan, layer_count, execution->sequence_mixer_layers, err);
    if (rc != YVEX_OK) return rc;
    plan->summary = (yvex_decoder_plan_summary){
        .schema_version = YVEX_DECODER_PLAN_SCHEMA_V1,
        .family_adapter_id = semantic->family_adapter_id,
        .family_adapter_version = semantic->family_adapter_version,
        .layer_count = layer_count,
        .attention_layer_count = semantic->attention_layer_count,
        .recurrent_layer_count = execution->sequence_mixer_layers,
        .hidden_width = execution->hidden_width,
        .intermediate_width = execution->dense_ffn_width,
        .vocabulary_size = execution->vocabulary_size,
        .maximum_context = execution->maximum_context};
    yvex_core_text_copy(plan->summary.logical_model_identity,
                        sizeof(plan->summary.logical_model_identity),
                        semantic->logical_model_identity);
    yvex_core_text_copy(plan->summary.semantic_model_identity,
                        sizeof(plan->summary.semantic_model_identity), semantic->identity);
    yvex_core_text_copy(plan->summary.model_execution_identity,
                        sizeof(plan->summary.model_execution_identity), execution->identity);
    yvex_core_text_copy(plan->summary.operator_graph_identity,
                        sizeof(plan->summary.operator_graph_identity), operators->identity);
    rc = decoder_layers_project(plan, layers, err);
    if (rc == YVEX_OK && !decoder_plan_identity(plan))
        rc = decoder_refuse(err, YVEX_ERR_STATE, "decoder plan identity failed");
    if (rc == YVEX_OK) *out = plan;
    else yvex_decoder_plan_close(&plan);
    return rc;
}

int yvex_decoder_plan_import(
    yvex_decoder_plan **out, const yvex_decoder_plan_summary *summary,
    const yvex_decoder_layer_plan *layers, yvex_error *err)
{
    yvex_decoder_plan *plan = NULL;
    unsigned long long index, recurrent = 0ull;
    int rc;

    if (out) *out = NULL;
    if (!out || !summary || !layers || !summary->layer_count ||
        summary->attention_layer_count + summary->recurrent_layer_count !=
            summary->layer_count)
        return decoder_refuse(
            err, YVEX_ERR_INVALID_ARG, "decoder import records are incomplete");
    rc = decoder_storage_open(
        &plan, summary->layer_count, summary->recurrent_layer_count, err);
    if (rc != YVEX_OK) return rc;
    plan->summary = *summary;
    memcpy(plan->layers, layers,
           (size_t)summary->layer_count * sizeof(*plan->layers));
    for (index = 0ull; index < summary->layer_count; ++index) {
        if (plan->layers[index].mixer !=
            YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA) continue;
        plan->recurrent[recurrent].layer_index = plan->layers[index].layer_index;
        plan->recurrent[recurrent].plan = plan->layers[index].gated_delta;
        recurrent++;
    }
    rc = recurrent == summary->recurrent_layer_count
             ? decoder_plan_validate(plan, err)
             : decoder_refuse(
                   err, YVEX_ERR_FORMAT,
                   "decoder import recurrent population is malformed");
    if (rc == YVEX_OK) *out = plan;
    else yvex_decoder_plan_close(&plan);
    return rc;
}

const yvex_decoder_plan_summary *yvex_decoder_plan_summary_get(
    const yvex_decoder_plan *plan)
{
    return plan ? &plan->summary : NULL;
}

const yvex_decoder_layer_plan *yvex_decoder_plan_layer_at(
    const yvex_decoder_plan *plan, unsigned long long ordinal)
{
    return plan && ordinal < plan->summary.layer_count
               ? &plan->layers[ordinal] : NULL;
}

int yvex_decoder_plan_sequence_state(
    const yvex_decoder_plan *plan, yvex_sequence_state_plan *out,
    yvex_error *err)
{
    if (out) memset(out, 0, sizeof(*out));
    if (!plan || !out || !plan->summary.recurrent_layer_count ||
        !plan->recurrent)
        return decoder_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "decoder has no recurrent sequence-state projection");
    *out = (yvex_sequence_state_plan){
        .schema_version = YVEX_SEQUENCE_STATE_SCHEMA_V1,
        .bindings = plan->recurrent,
        .binding_count = plan->summary.recurrent_layer_count};
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_decoder_plan_close(yvex_decoder_plan **owner)
{
    yvex_decoder_plan *plan = owner ? *owner : NULL;

    if (!plan) return;
    free(plan->recurrent);
    free(plan->layers);
    memset(plan, 0, sizeof(*plan));
    free(plan);
    *owner = NULL;
}
