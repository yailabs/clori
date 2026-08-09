/* Compile, persist, and reopen the immutable operator execution plan before model-open. */
#include <yvex/internal/compiler.h>

#include <yvex/internal/core.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/transformer.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_PLAN_SCHEMA_V1 1u
#define MODEL_PLAN_MAX_LAYERS 65536ull

typedef struct {
    const unsigned char *data;
    size_t count, offset;
} model_plan_cursor;

struct yvex_compiled_model_plan {
    yvex_moe_plan *moe, *draft_moe;
    yvex_transformer_plan *transformer, *draft_transformer;
    yvex_runtime_logits_plan_summary output_head;
};

static int model_plan_refuse(yvex_error *err, yvex_status status,
                             const char *reason)
{
    yvex_error_set(err, status, "runtime.model-plan", reason);
    return status;
}

static int plan_put_u64(yvex_core_bytes *bytes, unsigned long long value)
{
    unsigned char encoded[8];
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        encoded[index] = (unsigned char)(value >> (index * 8u));
    return yvex_core_bytes_append(bytes, encoded, sizeof(encoded));
}

static int plan_put_values(yvex_core_bytes *bytes,
                           const unsigned long long *values, size_t count)
{
    size_t index;
    for (index = 0u; index < count; ++index)
        if (!plan_put_u64(bytes, values[index])) return 0;
    return 1;
}

static int plan_put_double(yvex_core_bytes *bytes, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return plan_put_u64(bytes, bits);
}

static int plan_put_text(yvex_core_bytes *bytes, const char *text)
{
    size_t count = text ? strnlen(text, YVEX_SHA256_HEX_CAP) : 0u;
    return count < YVEX_SHA256_HEX_CAP && plan_put_u64(bytes, count) &&
           yvex_core_bytes_append(bytes, text, count);
}

static int plan_get_u64(model_plan_cursor *cursor, unsigned long long *value)
{
    unsigned long long decoded = 0ull;
    unsigned int index;
    if (!cursor || !value || cursor->offset > cursor->count ||
        cursor->count - cursor->offset < 8u)
        return 0;
    for (index = 0u; index < 8u; ++index)
        decoded |= (unsigned long long)cursor->data[cursor->offset + index]
                   << (index * 8u);
    cursor->offset += 8u;
    *value = decoded;
    return 1;
}

static int plan_get_values(model_plan_cursor *cursor,
                           unsigned long long *values, size_t count)
{
    size_t index;
    for (index = 0u; index < count; ++index)
        if (!plan_get_u64(cursor, &values[index])) return 0;
    return 1;
}

static int plan_get_double(model_plan_cursor *cursor, double *value)
{
    unsigned long long bits;
    if (!plan_get_u64(cursor, &bits)) return 0;
    memcpy(value, &bits, sizeof(*value));
    return 1;
}

static int plan_get_text(model_plan_cursor *cursor,
                         char output[YVEX_SHA256_HEX_CAP])
{
    unsigned long long count;
    if (!plan_get_u64(cursor, &count) || !count ||
        count >= YVEX_SHA256_HEX_CAP || cursor->offset > cursor->count ||
        count > cursor->count - cursor->offset)
        return 0;
    memset(output, 0, YVEX_SHA256_HEX_CAP);
    memcpy(output, cursor->data + cursor->offset, (size_t)count);
    cursor->offset += (size_t)count;
    return 1;
}

static int moe_summary_write(yvex_core_bytes *bytes,
                             const yvex_moe_plan_summary *summary)
{
    const unsigned long long values[] = {
        summary->schema_version, summary->tensor_scope,
        summary->family_adapter_id, summary->family_adapter_version,
        summary->layer_count, summary->hash_router_layer_count,
        summary->learned_router_layer_count, summary->routed_experts,
        summary->shared_experts, summary->experts_per_token,
        summary->required_binding_count, summary->expert_subview_count};
    return plan_put_values(bytes, values, sizeof(values) / sizeof(values[0])) &&
           plan_put_text(bytes, summary->artifact_identity) &&
           plan_put_text(bytes, summary->materialization_identity) &&
           plan_put_text(bytes, summary->logical_model_identity) &&
           plan_put_text(bytes, summary->runtime_numeric_identity) &&
           plan_put_text(bytes, summary->runtime_descriptor_identity) &&
           plan_put_text(bytes, summary->attention_plan_identity) &&
           plan_put_text(bytes, summary->moe_plan_identity);
}

static int moe_layer_write(yvex_core_bytes *bytes,
                           const yvex_moe_layer_plan *layer)
{
    const unsigned long long values[] = {
        layer->schema_version, layer->ordinal, layer->layer_index,
        layer->predictor_index, layer->tensor_scope, layer->router_class,
        layer->scoring, layer->topk_policy, layer->activation,
        layer->hidden_width, layer->residual_streams, layer->expanded_width,
        layer->mhc_mixing_rows, layer->mhc_sinkhorn_iterations,
        layer->routed_experts, layer->shared_experts,
        layer->experts_per_token, layer->expert_intermediate_width,
        layer->shared_intermediate_width, layer->hash_table_rows,
        layer->hash_table_columns, layer->correction_bias_width,
        (unsigned int)layer->requires_token_ids,
        (unsigned int)layer->requires_correction_bias,
        (unsigned int)layer->normalize_topk_probabilities};
    unsigned long long slot;
    if (!plan_put_values(bytes, values, sizeof(values) / sizeof(values[0])) ||
        !plan_put_double(bytes, layer->rms_epsilon) ||
        !plan_put_double(bytes, layer->mhc_epsilon) ||
        !plan_put_double(bytes, layer->mhc_post_multiplier) ||
        !plan_put_double(bytes, layer->routed_scaling_factor) ||
        !plan_put_double(bytes, layer->activation_limit))
        return 0;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        if (!plan_put_u64(bytes, layer->tensor_ids[slot])) return 0;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        if (!plan_put_u64(bytes, layer->qtypes[slot])) return 0;
    return plan_put_text(bytes, layer->layer_identity);
}

static int moe_plan_write(yvex_core_bytes *bytes, const yvex_moe_plan *plan)
{
    const yvex_moe_plan_summary *summary = yvex_moe_plan_summary_get(plan);
    unsigned long long index;
    if (!summary || !moe_summary_write(bytes, summary)) return 0;
    for (index = 0ull; index < summary->layer_count; ++index)
        if (!moe_layer_write(bytes, yvex_moe_plan_layer_at(plan, index))) return 0;
    return 1;
}

static int moe_summary_read(model_plan_cursor *cursor,
                            yvex_moe_plan_summary *summary)
{
    unsigned long long v[12];
    memset(summary, 0, sizeof(*summary));
    if (!plan_get_values(cursor, v, sizeof(v) / sizeof(v[0])) ||
        v[0] > UINT_MAX || v[1] > UINT_MAX)
        return 0;
    summary->schema_version = (unsigned int)v[0];
    summary->tensor_scope = (yvex_tensor_scope)v[1];
    summary->family_adapter_id = v[2];
    summary->family_adapter_version = v[3];
    summary->layer_count = v[4];
    summary->hash_router_layer_count = v[5];
    summary->learned_router_layer_count = v[6];
    summary->routed_experts = v[7];
    summary->shared_experts = v[8];
    summary->experts_per_token = v[9];
    summary->required_binding_count = v[10];
    summary->expert_subview_count = v[11];
    return plan_get_text(cursor, summary->artifact_identity) &&
           plan_get_text(cursor, summary->materialization_identity) &&
           plan_get_text(cursor, summary->logical_model_identity) &&
           plan_get_text(cursor, summary->runtime_numeric_identity) &&
           plan_get_text(cursor, summary->runtime_descriptor_identity) &&
           plan_get_text(cursor, summary->attention_plan_identity) &&
           plan_get_text(cursor, summary->moe_plan_identity);
}

static int moe_layer_read(model_plan_cursor *cursor, yvex_moe_layer_plan *layer)
{
    unsigned long long v[25], slot;
    memset(layer, 0, sizeof(*layer));
    if (!plan_get_values(cursor, v, sizeof(v) / sizeof(v[0])) ||
        v[0] > UINT_MAX || v[4] > UINT_MAX || v[5] > UINT_MAX ||
        v[6] > UINT_MAX || v[7] > UINT_MAX || v[8] > UINT_MAX ||
        v[22] > 1ull || v[23] > 1ull || v[24] > 1ull)
        return 0;
    layer->schema_version = (unsigned int)v[0];
    layer->ordinal = v[1];
    layer->layer_index = v[2];
    layer->predictor_index = v[3];
    layer->tensor_scope = (yvex_tensor_scope)v[4];
    layer->router_class = (yvex_moe_router_class)v[5];
    layer->scoring = (yvex_moe_scoring_policy)v[6];
    layer->topk_policy = (yvex_moe_topk_policy)v[7];
    layer->activation = (yvex_moe_activation)v[8];
    layer->hidden_width = v[9];
    layer->residual_streams = v[10];
    layer->expanded_width = v[11];
    layer->mhc_mixing_rows = v[12];
    layer->mhc_sinkhorn_iterations = v[13];
    layer->routed_experts = v[14];
    layer->shared_experts = v[15];
    layer->experts_per_token = v[16];
    layer->expert_intermediate_width = v[17];
    layer->shared_intermediate_width = v[18];
    layer->hash_table_rows = v[19];
    layer->hash_table_columns = v[20];
    layer->correction_bias_width = v[21];
    layer->requires_token_ids = (int)v[22];
    layer->requires_correction_bias = (int)v[23];
    layer->normalize_topk_probabilities = (int)v[24];
    if (!plan_get_double(cursor, &layer->rms_epsilon) ||
        !plan_get_double(cursor, &layer->mhc_epsilon) ||
        !plan_get_double(cursor, &layer->mhc_post_multiplier) ||
        !plan_get_double(cursor, &layer->routed_scaling_factor) ||
        !plan_get_double(cursor, &layer->activation_limit))
        return 0;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        if (!plan_get_u64(cursor, &layer->tensor_ids[slot])) return 0;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot) {
        unsigned long long qtype;
        if (!plan_get_u64(cursor, &qtype) || qtype > UINT_MAX) return 0;
        layer->qtypes[slot] = (unsigned int)qtype;
    }
    return plan_get_text(cursor, layer->layer_identity);
}

static int moe_plan_read(model_plan_cursor *cursor, yvex_moe_plan **out,
                         yvex_error *err)
{
    yvex_moe_plan_summary summary;
    yvex_moe_layer_plan *layers = NULL;
    unsigned long long index;
    int rc;
    if (!moe_summary_read(cursor, &summary) || !summary.layer_count ||
        summary.layer_count > MODEL_PLAN_MAX_LAYERS ||
        summary.layer_count > SIZE_MAX / sizeof(*layers))
        return model_plan_refuse(err, YVEX_ERR_FORMAT,
                                 "compiled MoE summary is malformed");
    layers = (yvex_moe_layer_plan *)calloc((size_t)summary.layer_count,
                                            sizeof(*layers));
    if (!layers)
        return model_plan_refuse(err, YVEX_ERR_NOMEM,
                                 "compiled MoE directory allocation failed");
    for (index = 0ull; index < summary.layer_count; ++index)
        if (!moe_layer_read(cursor, &layers[index])) {
            free(layers);
            return model_plan_refuse(err, YVEX_ERR_FORMAT,
                                     "compiled MoE layer is malformed");
        }
    rc = yvex_moe_plan_import(out, &summary, layers, err);
    free(layers);
    return rc;
}

static int transformer_summary_write(
    yvex_core_bytes *bytes, const yvex_transformer_plan_summary *summary)
{
    const unsigned long long values[] = {
        summary->schema_version, summary->tensor_scope,
        summary->family_adapter_id, summary->family_adapter_version,
        summary->layer_count, summary->hidden_width, summary->residual_streams,
        summary->expanded_width, summary->maximum_context,
        summary->vocabulary_size, summary->initial_policy,
        summary->final_policy, summary->sinkhorn_iterations};
    unsigned long long slot;
    if (!plan_put_values(bytes, values, sizeof(values) / sizeof(values[0])) ||
        !plan_put_double(bytes, summary->mhc_epsilon) ||
        !plan_put_double(bytes, summary->output_norm_epsilon) ||
        !plan_put_text(bytes, summary->artifact_identity) ||
        !plan_put_text(bytes, summary->materialization_identity) ||
        !plan_put_text(bytes, summary->logical_model_identity) ||
        !plan_put_text(bytes, summary->runtime_numeric_identity) ||
        !plan_put_text(bytes, summary->runtime_descriptor_identity) ||
        !plan_put_text(bytes, summary->attention_plan_identity) ||
        !plan_put_text(bytes, summary->moe_plan_identity) ||
        !plan_put_text(bytes, summary->transformer_plan_identity))
        return 0;
    for (slot = 0ull; slot < YVEX_TRANSFORMER_WEIGHT_COUNT; ++slot) {
        const yvex_transformer_weight_binding *weight = &summary->weights[slot];
        const unsigned long long fields[] = {
            weight->tensor_id, weight->row_width, weight->row_count,
            weight->encoded_bytes, weight->role, weight->tensor_scope,
            weight->layer_index, weight->predictor_index, weight->qtype};
        if (!plan_put_values(bytes, fields, sizeof(fields) / sizeof(fields[0])))
            return 0;
    }
    return 1;
}

static int transformer_plan_write(yvex_core_bytes *bytes,
                                  const yvex_transformer_plan *plan)
{
    const yvex_transformer_plan_summary *summary =
        yvex_transformer_plan_summary_get(plan);
    unsigned long long index;
    if (!summary || !transformer_summary_write(bytes, summary)) return 0;
    for (index = 0ull; index < summary->layer_count; ++index) {
        const yvex_transformer_layer_plan *layer =
            yvex_transformer_plan_layer_at(plan, index);
        const unsigned long long fields[] = {
            layer ? layer->ordinal : ULLONG_MAX,
            layer ? layer->layer_index : ULLONG_MAX,
            layer ? layer->predictor_index : ULLONG_MAX,
            layer ? layer->tensor_scope : ULLONG_MAX};
        if (!layer || !plan_put_values(bytes, fields,
                                       sizeof(fields) / sizeof(fields[0])) ||
            !plan_put_text(bytes, layer->moe_layer_identity) ||
            !plan_put_text(bytes, layer->layer_identity))
            return 0;
    }
    return 1;
}

static int transformer_summary_read(
    model_plan_cursor *cursor, yvex_transformer_plan_summary *summary)
{
    unsigned long long v[13], slot;
    memset(summary, 0, sizeof(*summary));
    if (!plan_get_values(cursor, v, sizeof(v) / sizeof(v[0])) ||
        v[0] > UINT_MAX || v[1] > UINT_MAX || v[10] > UINT_MAX ||
        v[11] > UINT_MAX)
        return 0;
    summary->schema_version = (unsigned int)v[0];
    summary->tensor_scope = (yvex_tensor_scope)v[1];
    summary->family_adapter_id = v[2];
    summary->family_adapter_version = v[3];
    summary->layer_count = v[4];
    summary->hidden_width = v[5];
    summary->residual_streams = v[6];
    summary->expanded_width = v[7];
    summary->maximum_context = v[8];
    summary->vocabulary_size = v[9];
    summary->initial_policy = (yvex_transformer_initial_policy)v[10];
    summary->final_policy = (yvex_transformer_final_policy)v[11];
    summary->sinkhorn_iterations = v[12];
    if (!plan_get_double(cursor, &summary->mhc_epsilon) ||
        !plan_get_double(cursor, &summary->output_norm_epsilon) ||
        !plan_get_text(cursor, summary->artifact_identity) ||
        !plan_get_text(cursor, summary->materialization_identity) ||
        !plan_get_text(cursor, summary->logical_model_identity) ||
        !plan_get_text(cursor, summary->runtime_numeric_identity) ||
        !plan_get_text(cursor, summary->runtime_descriptor_identity) ||
        !plan_get_text(cursor, summary->attention_plan_identity) ||
        !plan_get_text(cursor, summary->moe_plan_identity) ||
        !plan_get_text(cursor, summary->transformer_plan_identity))
        return 0;
    for (slot = 0ull; slot < YVEX_TRANSFORMER_WEIGHT_COUNT; ++slot) {
        unsigned long long f[9];
        yvex_transformer_weight_binding *weight = &summary->weights[slot];
        if (!plan_get_values(cursor, f, sizeof(f) / sizeof(f[0])) ||
            f[4] > UINT_MAX || f[5] > UINT_MAX || f[8] > UINT_MAX)
            return 0;
        weight->tensor_id = f[0];
        weight->row_width = f[1];
        weight->row_count = f[2];
        weight->encoded_bytes = f[3];
        weight->role = (yvex_tensor_role)f[4];
        weight->tensor_scope = (yvex_tensor_scope)f[5];
        weight->layer_index = f[6];
        weight->predictor_index = f[7];
        weight->qtype = (unsigned int)f[8];
    }
    return 1;
}

static int transformer_plan_read(model_plan_cursor *cursor,
                                 yvex_transformer_plan **out, yvex_error *err)
{
    yvex_transformer_plan_summary summary;
    yvex_transformer_layer_plan *layers = NULL;
    unsigned long long index;
    int rc;
    if (!transformer_summary_read(cursor, &summary) || !summary.layer_count ||
        summary.layer_count > MODEL_PLAN_MAX_LAYERS ||
        summary.layer_count > SIZE_MAX / sizeof(*layers))
        return model_plan_refuse(err, YVEX_ERR_FORMAT,
                                 "compiled transformer summary is malformed");
    layers = (yvex_transformer_layer_plan *)calloc(
        (size_t)summary.layer_count, sizeof(*layers));
    if (!layers)
        return model_plan_refuse(
            err, YVEX_ERR_NOMEM,
            "compiled transformer directory allocation failed");
    for (index = 0ull; index < summary.layer_count; ++index) {
        unsigned long long v[4];
        if (!plan_get_values(cursor, v, sizeof(v) / sizeof(v[0])) ||
            v[3] > UINT_MAX ||
            !plan_get_text(cursor, layers[index].moe_layer_identity) ||
            !plan_get_text(cursor, layers[index].layer_identity)) {
            free(layers);
            return model_plan_refuse(err, YVEX_ERR_FORMAT,
                                     "compiled transformer layer is malformed");
        }
        layers[index].ordinal = v[0];
        layers[index].layer_index = v[1];
        layers[index].predictor_index = v[2];
        layers[index].tensor_scope = (yvex_tensor_scope)v[3];
    }
    rc = yvex_transformer_plan_import(out, &summary, layers, err);
    free(layers);
    return rc;
}

static int output_head_write(yvex_core_bytes *bytes,
                             const yvex_runtime_logits_plan_summary *summary)
{
    const unsigned long long values[] = {
        summary->schema_version, summary->family_adapter_id,
        summary->family_adapter_version, summary->output_head_tensor_id,
        summary->row_width, summary->row_count, summary->row_bytes,
        summary->encoded_bytes, summary->vocabulary_size,
        summary->hidden_width, summary->role, summary->qtype,
        (unsigned int)summary->separate_output_head,
        (unsigned int)summary->output_head_bias};
    return plan_put_values(bytes, values, sizeof(values) / sizeof(values[0])) &&
           plan_put_text(bytes, summary->artifact_identity) &&
           plan_put_text(bytes, summary->materialization_identity) &&
           plan_put_text(bytes, summary->logical_model_identity) &&
           plan_put_text(bytes, summary->runtime_numeric_identity) &&
           plan_put_text(bytes, summary->runtime_descriptor_identity) &&
           plan_put_text(bytes, summary->transformer_plan_identity) &&
           plan_put_text(bytes, summary->output_head_plan_identity);
}

static int output_head_read(model_plan_cursor *cursor,
                            yvex_runtime_logits_plan_summary *summary,
                            yvex_error *err)
{
    unsigned long long v[14];
    memset(summary, 0, sizeof(*summary));
    if (!plan_get_values(cursor, v, sizeof(v) / sizeof(v[0])) ||
        v[0] > UINT_MAX || v[10] > UINT_MAX || v[11] > UINT_MAX ||
        v[12] > 1ull || v[13] > 1ull)
        return model_plan_refuse(err, YVEX_ERR_FORMAT,
                                 "compiled output-head fields are malformed");
    summary->schema_version = (unsigned int)v[0];
    summary->family_adapter_id = v[1];
    summary->family_adapter_version = v[2];
    summary->output_head_tensor_id = v[3];
    summary->row_width = v[4];
    summary->row_count = v[5];
    summary->row_bytes = v[6];
    summary->encoded_bytes = v[7];
    summary->vocabulary_size = v[8];
    summary->hidden_width = v[9];
    summary->role = (yvex_tensor_role)v[10];
    summary->qtype = (unsigned int)v[11];
    summary->separate_output_head = (int)v[12];
    summary->output_head_bias = (int)v[13];
    if (!plan_get_text(cursor, summary->artifact_identity) ||
        !plan_get_text(cursor, summary->materialization_identity) ||
        !plan_get_text(cursor, summary->logical_model_identity) ||
        !plan_get_text(cursor, summary->runtime_numeric_identity) ||
        !plan_get_text(cursor, summary->runtime_descriptor_identity) ||
        !plan_get_text(cursor, summary->transformer_plan_identity) ||
        !plan_get_text(cursor, summary->output_head_plan_identity))
        return model_plan_refuse(err, YVEX_ERR_FORMAT,
                                 "compiled output-head identities are malformed");
    return yvex_output_head_plan_validate(summary, err);
}

void yvex_compiled_model_plan_close(yvex_compiled_model_plan **owner)
{
    yvex_compiled_model_plan *plans = owner ? *owner : NULL;
    if (!plans) return;
    yvex_transformer_plan_close(&plans->draft_transformer);
    yvex_transformer_plan_close(&plans->transformer);
    yvex_moe_plan_close(&plans->draft_moe);
    yvex_moe_plan_close(&plans->moe);
    memset(plans, 0, sizeof(*plans));
    free(plans);
    *owner = NULL;
}

int yvex_compiled_model_plan_build(
    yvex_compiled_model_plan **out,
    const yvex_compiled_model_plan_request *request, yvex_error *err)
{
    yvex_compiled_model_plan *plan = NULL;
    int compile_execution;
    int rc;
    if (out) *out = NULL;
    if (!out || !request || !request->materialization ||
        !request->descriptor || !request->attention)
        return model_plan_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "compiled model-plan inputs are required");
    plan = (yvex_compiled_model_plan *)calloc(1u, sizeof(*plan));
    if (!plan)
        return model_plan_refuse(err, YVEX_ERR_NOMEM,
                                 "compiled model-plan allocation failed");
    compile_execution = request->capabilities.moe_plan_ready ||
                        request->capabilities.transformer_ready ||
                        request->capabilities.logits_ready;
    if (!compile_execution) {
        *out = plan;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (!request->capabilities.moe_plan_ready ||
        !request->capabilities.transformer_ready ||
        !request->capabilities.logits_ready) {
        yvex_compiled_model_plan_close(&plan);
        return model_plan_refuse(
            err, YVEX_ERR_FORMAT,
            "partial compiled execution capabilities are inconsistent");
    }
    rc = yvex_moe_plan_build(
        &plan->moe, request->family_adapter_id,
        request->family_adapter_version, request->materialization,
        request->descriptor, request->attention, err);
    if (rc == YVEX_OK)
        rc = yvex_transformer_plan_compile(
            &plan->transformer, &request->transformer_policy,
            request->family_adapter_id, request->family_adapter_version,
            request->materialization, request->descriptor,
            request->attention, plan->moe, YVEX_TENSOR_SCOPE_GLOBAL, err);
    if (rc == YVEX_OK)
        rc = yvex_output_head_plan_build(
            &plan->output_head, request->family_adapter_id,
            request->family_adapter_version, request->materialization,
            request->descriptor, plan->transformer,
            &request->logits_policy, err);
    if (rc == YVEX_OK && request->draft_attention)
        rc = yvex_moe_plan_build(
            &plan->draft_moe, request->family_adapter_id,
            request->family_adapter_version, request->materialization,
            request->descriptor, request->draft_attention, err);
    if (rc == YVEX_OK && request->draft_attention)
        rc = yvex_transformer_plan_compile(
            &plan->draft_transformer, &request->transformer_policy,
            request->family_adapter_id, request->family_adapter_version,
            request->materialization, request->descriptor,
            request->draft_attention, plan->draft_moe,
            YVEX_TENSOR_SCOPE_DRAFT, err);
    if (rc == YVEX_OK) *out = plan;
    else yvex_compiled_model_plan_close(&plan);
    return rc;
}

int yvex_compiled_model_plan_encode(
    const yvex_compiled_model_plan *plans, yvex_core_bytes *bytes,
    yvex_error *err)
{
    int target_present = plans && plans->moe && plans->transformer;
    int draft_present = plans && plans->draft_moe && plans->draft_transformer;
    if (!plans || !bytes ||
        (plans->moe != NULL) != (plans->transformer != NULL) ||
        (target_present != (plans->output_head.schema_version != 0u)) ||
        (plans->draft_moe != NULL) != (plans->draft_transformer != NULL) ||
        (!target_present && draft_present) ||
        !plan_put_text(bytes, "yvex.compiled-model-plan.v1") ||
        !plan_put_u64(bytes, MODEL_PLAN_SCHEMA_V1) ||
        !plan_put_u64(bytes, (unsigned int)target_present) ||
        (target_present &&
         (!moe_plan_write(bytes, plans->moe) ||
          !transformer_plan_write(bytes, plans->transformer) ||
          !output_head_write(bytes, &plans->output_head))) ||
        !plan_put_u64(bytes, (unsigned int)draft_present) ||
        (draft_present &&
         (!moe_plan_write(bytes, plans->draft_moe) ||
          !transformer_plan_write(bytes, plans->draft_transformer))))
        return model_plan_refuse(err, YVEX_ERR_NOMEM,
                                 "compiled model-plan encoding failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_compiled_model_plan_decode(
    yvex_compiled_model_plan **out, const unsigned char *data, size_t count,
    yvex_error *err)
{
    model_plan_cursor cursor = {data, count, 0u};
    yvex_compiled_model_plan *plan = NULL;
    char domain[YVEX_SHA256_HEX_CAP];
    unsigned long long schema, target_present, draft_present;
    int rc;
    if (out) *out = NULL;
    if (!out || !data || !count || !plan_get_text(&cursor, domain) ||
        strcmp(domain, "yvex.compiled-model-plan.v1") != 0 ||
        !plan_get_u64(&cursor, &schema) || schema != MODEL_PLAN_SCHEMA_V1)
        return model_plan_refuse(err, YVEX_ERR_FORMAT,
                                 "compiled model-plan header is malformed");
    plan = (yvex_compiled_model_plan *)calloc(1u, sizeof(*plan));
    if (!plan)
        return model_plan_refuse(err, YVEX_ERR_NOMEM,
                                 "compiled model-plan allocation failed");
    if (!plan_get_u64(&cursor, &target_present) || target_present > 1ull)
        rc = model_plan_refuse(err, YVEX_ERR_FORMAT,
                               "compiled target-plan presence is malformed");
    else rc = YVEX_OK;
    if (rc == YVEX_OK && target_present)
        rc = moe_plan_read(&cursor, &plan->moe, err);
    if (rc == YVEX_OK && target_present)
        rc = transformer_plan_read(&cursor, &plan->transformer, err);
    if (rc == YVEX_OK && target_present)
        rc = output_head_read(&cursor, &plan->output_head, err);
    if (rc == YVEX_OK &&
        (!plan_get_u64(&cursor, &draft_present) || draft_present > 1ull))
        rc = model_plan_refuse(err, YVEX_ERR_FORMAT,
                               "compiled draft-plan presence is malformed");
    if (rc == YVEX_OK && draft_present)
        rc = moe_plan_read(&cursor, &plan->draft_moe, err);
    if (rc == YVEX_OK && draft_present)
        rc = transformer_plan_read(&cursor, &plan->draft_transformer, err);
    if (rc == YVEX_OK && cursor.offset != cursor.count)
        rc = model_plan_refuse(err, YVEX_ERR_FORMAT,
                               "compiled model-plan has trailing bytes");
    if (rc == YVEX_OK) *out = plan;
    else yvex_compiled_model_plan_close(&plan);
    return rc;
}

int yvex_compiled_model_plan_admit(
    const yvex_compiled_model_plan *plans,
    const yvex_compiled_model_plan_admission *admission)
{
    yvex_compiled_context_envelope context;
    yvex_error err;
    const yvex_moe_plan_summary *moe;
    const yvex_moe_plan_summary *draft_moe;
    const yvex_transformer_plan_summary *transformer;
    const yvex_transformer_plan_summary *draft_transformer;
    const yvex_runtime_logits_plan_summary *output;
    if (!plans || !admission || !admission->capabilities) return 0;
    moe = yvex_moe_plan_summary_get(plans->moe);
    draft_moe = yvex_moe_plan_summary_get(plans->draft_moe);
    transformer = yvex_transformer_plan_summary_get(plans->transformer);
    draft_transformer =
        yvex_transformer_plan_summary_get(plans->draft_transformer);
    output = &plans->output_head;
    if (!moe || !transformer || !output->schema_version)
        return !moe && !transformer && !draft_moe && !draft_transformer &&
               !output->schema_version &&
               !admission->capabilities->moe_plan_ready &&
               !admission->capabilities->transformer_ready &&
               !admission->capabilities->logits_ready;
    if (yvex_compiled_model_plan_context_envelope(
            plans, admission->model, &context, &err) != YVEX_OK ||
        !admission->capabilities->moe_plan_ready ||
        !admission->capabilities->transformer_ready ||
        !admission->capabilities->logits_ready ||
        moe->family_adapter_id != admission->family_adapter_id ||
        moe->family_adapter_version != admission->family_adapter_version ||
        moe->layer_count != admission->layer_count ||
        strcmp(moe->artifact_identity, admission->artifact_identity) != 0 ||
        strcmp(moe->materialization_identity,
               admission->materialization_identity) != 0 ||
        strcmp(moe->runtime_descriptor_identity,
               admission->runtime_descriptor_identity) != 0 ||
        strcmp(moe->attention_plan_identity,
               admission->attention_plan_identity) != 0 ||
        strcmp(moe->moe_plan_identity, admission->moe_plan_identity) != 0 ||
        strcmp(transformer->moe_plan_identity, moe->moe_plan_identity) != 0 ||
        strcmp(transformer->attention_plan_identity,
               moe->attention_plan_identity) != 0 ||
        strcmp(transformer->transformer_plan_identity,
               admission->transformer_plan_identity) != 0 ||
        output->output_head_tensor_id >= admission->tensor_count ||
        strcmp(output->transformer_plan_identity,
               transformer->transformer_plan_identity) != 0 ||
        strcmp(output->output_head_plan_identity,
               admission->output_head_plan_identity) != 0)
        return 0;
    if (!admission->draft_layer_count)
        return !draft_moe && !draft_transformer;
    return draft_moe && draft_transformer &&
           draft_moe->layer_count == admission->draft_layer_count &&
           strcmp(draft_moe->attention_plan_identity,
                  admission->draft_attention_plan_identity) == 0 &&
           strcmp(draft_moe->moe_plan_identity,
                  admission->draft_moe_plan_identity) == 0 &&
           strcmp(draft_transformer->moe_plan_identity,
                  draft_moe->moe_plan_identity) == 0 &&
           strcmp(draft_transformer->attention_plan_identity,
                  draft_moe->attention_plan_identity) == 0 &&
           strcmp(draft_transformer->transformer_plan_identity,
                  admission->draft_transformer_plan_identity) == 0;
}

int yvex_compiled_model_plan_context_envelope(
    const yvex_compiled_model_plan *plan,
    const yvex_model_execution_descriptor *model,
    yvex_compiled_context_envelope *envelope, yvex_error *err)
{
    const yvex_transformer_plan_summary *target =
        yvex_transformer_plan_summary_get(
            yvex_compiled_model_plan_transformer(plan, 0));
    const yvex_transformer_plan_summary *draft =
        yvex_transformer_plan_summary_get(
            yvex_compiled_model_plan_transformer(plan, 1));
    if (envelope) memset(envelope, 0, sizeof(*envelope));
    if (!envelope || !model ||
        model->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
        !model->maximum_context || !yvex_sha256_hex_valid(model->identity) ||
        !target || target->maximum_context != model->maximum_context ||
        !yvex_sha256_hex_valid(target->transformer_plan_identity) ||
        (!!draft != !!model->draft_layer_count) ||
        (draft && (draft->maximum_context != model->maximum_context ||
                   !yvex_sha256_hex_valid(draft->transformer_plan_identity))))
        return model_plan_refuse(
            err, YVEX_ERR_FORMAT,
            "compiled context envelope does not match semantic model capability");
    envelope->schema_version = YVEX_COMPILED_CONTEXT_ENVELOPE_SCHEMA_V1;
    envelope->semantic_maximum_context = model->maximum_context;
    envelope->target_maximum_context = target->maximum_context;
    envelope->draft_available = draft != NULL;
    envelope->draft_maximum_context = draft ? draft->maximum_context : 0ull;
    yvex_core_text_copy(envelope->model_execution_identity,
                        sizeof(envelope->model_execution_identity), model->identity);
    yvex_core_text_copy(envelope->target_transformer_identity,
                        sizeof(envelope->target_transformer_identity),
                        target->transformer_plan_identity);
    if (draft)
        yvex_core_text_copy(envelope->draft_transformer_identity,
                            sizeof(envelope->draft_transformer_identity),
                            draft->transformer_plan_identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_compiled_context_envelope_admit(
    const yvex_compiled_context_envelope *envelope,
    unsigned long long requested_context, int require_draft, yvex_error *err)
{
    unsigned long long maximum;
    if (!envelope ||
        envelope->schema_version != YVEX_COMPILED_CONTEXT_ENVELOPE_SCHEMA_V1 ||
        !envelope->semantic_maximum_context || !envelope->target_maximum_context ||
        envelope->semantic_maximum_context != envelope->target_maximum_context ||
        !yvex_sha256_hex_valid(envelope->model_execution_identity) ||
        !yvex_sha256_hex_valid(envelope->target_transformer_identity) ||
        (envelope->draft_available &&
         (envelope->draft_maximum_context != envelope->semantic_maximum_context ||
          !yvex_sha256_hex_valid(envelope->draft_transformer_identity))) ||
        (!envelope->draft_available &&
         (envelope->draft_maximum_context || envelope->draft_transformer_identity[0])) ||
        (require_draft != 0 && require_draft != 1) || !requested_context)
        return model_plan_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "compiled context admission requires one bounded runtime request");
    if (require_draft && !envelope->draft_available)
        return model_plan_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "compiled context envelope does not admit draft execution");
    maximum = require_draft ? envelope->draft_maximum_context
                            : envelope->target_maximum_context;
    if (requested_context > maximum)
        return model_plan_refuse(
            err, YVEX_ERR_BOUNDS,
            "requested context exceeds the compiled semantic maximum");
    yvex_error_clear(err);
    return YVEX_OK;
}

const yvex_moe_plan *yvex_compiled_model_plan_moe(
    const yvex_compiled_model_plan *plan, int draft)
{
    return plan ? (draft ? plan->draft_moe : plan->moe) : NULL;
}

const yvex_transformer_plan *yvex_compiled_model_plan_transformer(
    const yvex_compiled_model_plan *plan, int draft)
{
    return plan ? (draft ? plan->draft_transformer : plan->transformer) : NULL;
}

const yvex_runtime_logits_plan_summary *yvex_compiled_model_plan_output_head(
    const yvex_compiled_model_plan *plan)
{
    return plan && plan->output_head.schema_version ? &plan->output_head : NULL;
}
