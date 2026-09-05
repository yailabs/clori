/* Execute a heterogeneous autoregressive decoder over session-owned mixed sequence state. */
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/backend.h>
#include <yvex/internal/component.h>
#include <yvex/internal/core.h>
#include <yvex/internal/decoder_execution.h>
#include <yvex/internal/decoder_plan.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/sequence_mixer.h>
#include <yvex/internal/stateful_attention.h>

enum {
    DECODER_LINEAR_ATTENTION_Q = 0,
    DECODER_LINEAR_ATTENTION_KV,
    DECODER_LINEAR_ATTENTION_OUT,
    DECODER_LINEAR_FFN_UP,
    DECODER_LINEAR_FFN_DOWN,
    DECODER_LINEAR_DELTA_QKV,
    DECODER_LINEAR_DELTA_VALUE,
    DECODER_LINEAR_DELTA_HEAD,
    DECODER_LINEAR_DELTA_OUT,
    DECODER_LINEAR_COUNT
};

enum {
    DECODER_BUFFER_HIDDEN_A = 0,
    DECODER_BUFFER_HIDDEN_B,
    DECODER_BUFFER_NORMALIZED,
    DECODER_BUFFER_UPDATE,
    DECODER_BUFFER_ATTENTION_Q_COMBINED,
    DECODER_BUFFER_ATTENTION_Q,
    DECODER_BUFFER_ATTENTION_GATE,
    DECODER_BUFFER_ATTENTION_K,
    DECODER_BUFFER_ATTENTION_V,
    DECODER_BUFFER_MIXER_OUTPUT,
    DECODER_BUFFER_DELTA_QKV,
    DECODER_BUFFER_DELTA_GATE,
    DECODER_BUFFER_DELTA_BETA,
    DECODER_BUFFER_DELTA_DECAY,
    DECODER_BUFFER_FFN_GATE,
    DECODER_BUFFER_FFN_UP,
    DECODER_BUFFER_FFN_PRODUCT,
    DECODER_BUFFER_COSINE,
    DECODER_BUFFER_SINE,
    DECODER_BUFFER_COUNT
};

typedef struct {
    const yvex_materialized_tensor_binding *binding;
    yvex_component_encoded_weight encoded;
} decoder_weight;

typedef struct {
    const yvex_materialized_tensor_binding *binding;
    float *host;
    unsigned long long count;
    yvex_device_tensor *device;
} decoder_small_weight;

typedef struct {
    decoder_small_weight input_norm, ffn_norm;
    decoder_weight ffn_gate, ffn_up, ffn_down;
    decoder_weight attention_q, attention_k, attention_v, attention_out;
    decoder_small_weight attention_q_norm, attention_k_norm;
    decoder_weight delta_qkv, delta_gate, delta_beta, delta_decay, delta_out;
    decoder_small_weight delta_convolution, delta_decay_log;
    decoder_small_weight delta_time_bias, delta_output_norm;
} decoder_layer_resources;

typedef struct {
    yvex_transformer_linear_requirement requirement;
    const char *domain;
    yvex_transformer_linear_executable *single;
    yvex_transformer_linear_executable *multiple;
    unsigned long long multiple_rows;
} decoder_linear_owner;

struct yvex_runtime_decoder_execution_context {
    yvex_model_engine *model;
    yvex_runtime_execution_session *session;
    const yvex_model_engine_view *model_view;
    const yvex_runtime_session_view *session_view;
    const yvex_decoder_plan *plan;
    const yvex_decoder_plan_summary *summary;
    const yvex_backend_transformer_operations *operations;
    yvex_runtime_decoder_execution_options options;
    decoder_weight embedding;
    decoder_small_weight output_norm;
    decoder_layer_resources *layers;
    decoder_linear_owner linears[DECODER_LINEAR_COUNT];
    yvex_device_tensor *buffers[DECODER_BUFFER_COUNT];
    yvex_device_tensor hidden_publication;
    float *state_workspace;
    unsigned long long state_workspace_values;
    unsigned long long *position_workspace;
    float *rope_workspace;
    unsigned long long rope_workspace_values;
    unsigned long long host_bytes, device_bytes;
    pthread_mutex_t mutex;
    int mutex_ready, busy, invalidated;
};

typedef struct {
    yvex_runtime_decoder_execution_context *context;
    const yvex_runtime_decoder_execution_request *request;
    yvex_runtime_decoder_execution_result *result;
    yvex_device_tensor current, next, normalized, update;
} decoder_layer_run;

static int decoder_refuse(yvex_error *err, yvex_status status, const char *where,
                   const char *reason)
{
    if (!yvex_error_is_set(err)) yvex_error_set(err, status, where, reason);
    return err && yvex_error_is_set(err) ? yvex_error_code(err) : status;
}

static int decoder_bytes_account(
    unsigned long long *total, unsigned long long bytes,
    unsigned long long maximum, const char *where, const char *reason,
    yvex_error *err)
{
    unsigned long long updated;
    if (!total || !yvex_core_u64_add(*total, bytes, &updated) ||
        (maximum && updated > maximum))
        return decoder_refuse(
            err, YVEX_ERR_BOUNDS, where, reason);
    *total = updated;
    return YVEX_OK;
}

static const yvex_runtime_tensor_binding *decoder_descriptor_binding(
    const yvex_runtime_decoder_execution_context *context,
    yvex_tensor_role role, unsigned long long layer)
{
    yvex_tensor_scope scope = layer == YVEX_ATTENTION_NO_LAYER
                                  ? YVEX_TENSOR_SCOPE_GLOBAL
                                  : YVEX_TENSOR_SCOPE_MAIN_LAYER;
    return context && context->model_view
               ? yvex_runtime_descriptor_find_role(
                     context->model_view->descriptor, role, scope, layer,
                     YVEX_ATTENTION_NO_TENSOR_INDEX)
               : NULL;
}

static int decoder_weight_bind(yvex_runtime_decoder_execution_context *context,
                        yvex_tensor_role role, unsigned long long layer,
                        decoder_weight *out, yvex_error *err)
{
    const yvex_runtime_tensor_binding *row;
    const yvex_materialized_tensor_binding *binding;
    const unsigned char *encoded = NULL;
    unsigned long long bytes = 0ull, row_bytes;

    if (out) memset(out, 0, sizeof(*out));
    row = decoder_descriptor_binding(context, role, layer);
    binding = row ? row->binding : NULL;
    if (!context || !out || !binding || !binding->row_count ||
        !binding->row_width || binding->encoded_bytes % binding->row_count ||
        !(row_bytes = binding->encoded_bytes / binding->row_count) ||
        yvex_runtime_residency_binding_view(
            context->model_view->residency, binding, &encoded, &bytes,
            err) != YVEX_OK ||
        bytes != binding->encoded_bytes)
        return decoder_refuse(
            err, YVEX_ERR_FORMAT, "runtime.decoder.weight",
            "one exact resident decoder tensor binding is required");
    out->binding = binding;
    out->encoded = (yvex_component_encoded_weight){
        .encoded = encoded,
        .encoded_bytes = binding->encoded_bytes,
        .row_count = binding->row_count,
        .row_width = binding->row_width,
        .row_bytes = row_bytes,
        .qtype = binding->qtype};
    return YVEX_OK;
}

static int decoder_decode_weight(const decoder_weight *weight, float *decoded,
                                 unsigned long long count, yvex_error *err)
{
    const yvex_gguf_qtype_geometry *geometry;
    yvex_quant_failure failure = {0};
    unsigned long long blocks, block;

    geometry = weight && weight->binding
                   ? yvex_gguf_qtype_geometry_find(weight->binding->qtype)
                   : NULL;
    if (!geometry || !decoded || !count || !geometry->block_size ||
        !geometry->bytes_per_block || count % geometry->block_size ||
        !(blocks = count / geometry->block_size) ||
        blocks > SIZE_MAX / geometry->bytes_per_block ||
        blocks * geometry->bytes_per_block != weight->encoded.encoded_bytes)
        return decoder_refuse(
            err, YVEX_ERR_FORMAT, "runtime.decoder.weight-decode",
            "decoder parameter qtype geometry is not block exact");
    for (block = 0ull; block < blocks; ++block) {
        memset(&failure, 0, sizeof(failure));
        if (yvex_quant_decode_block(
                weight->binding->qtype,
                weight->encoded.encoded + block * geometry->bytes_per_block,
                geometry->bytes_per_block,
                decoded + block * geometry->block_size,
                geometry->block_size, &failure, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return YVEX_OK;
}

static int decoder_small_weight_bind(
    yvex_runtime_decoder_execution_context *context, yvex_tensor_role role,
    unsigned long long layer, unsigned long long expected_count, int one_plus,
    decoder_small_weight *out, yvex_error *err)
{
    decoder_weight weight = {0};
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long count, bytes, index;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    rc = decoder_weight_bind(context, role, layer, &weight, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_mul(weight.encoded.row_count, weight.encoded.row_width,
                           &count) ||
        count != expected_count ||
        !yvex_core_u64_mul(count, sizeof(float), &bytes) || bytes > SIZE_MAX)
        return decoder_refuse(
            err, YVEX_ERR_FORMAT, "runtime.decoder.small-weight",
            "decoder parameter shape differs from its semantic plan");
    out->host = malloc((size_t)bytes);
    if (!out->host)
        return decoder_refuse(err, YVEX_ERR_NOMEM,
                              "runtime.decoder.small-weight",
                              "decoder parameter allocation failed");
    out->binding = weight.binding;
    out->count = count;
    rc = decoder_decode_weight(&weight, out->host, count, err);
    if (rc == YVEX_OK && one_plus)
        for (index = 0ull; index < count; ++index) out->host[index] += 1.0f;
    descriptor.name = "decoder-small-parameter";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = count;
    descriptor.bytes = bytes;
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_alloc(context->session_view->backend,
                                       &descriptor, &out->device, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(context->session_view->backend,
                                       out->device, out->host, bytes, err);
    if (rc == YVEX_OK)
        rc = decoder_bytes_account(
            &context->host_bytes, bytes, context->options.maximum_host_bytes,
            "runtime.decoder.small-weight",
            "decoder parameters exceed the host budget", err);
    if (rc == YVEX_OK)
        rc = decoder_bytes_account(
            &context->device_bytes, bytes,
            context->options.maximum_device_bytes,
            "runtime.decoder.small-weight",
            "decoder parameters exceed the device budget", err);
    if (rc != YVEX_OK) {
        if (out->device)
            (void)yvex_backend_tensor_release(context->session_view->backend,
                                              &out->device, NULL);
        free(out->host);
        memset(out, 0, sizeof(*out));
    }
    return rc;
}

static int decoder_weight_shape(const decoder_weight *weight,
                                unsigned long long rows,
                                unsigned long long columns)
{
    return weight && weight->binding && weight->encoded.row_count == rows &&
           weight->encoded.row_width == columns;
}

static void decoder_linear_configure(decoder_linear_owner *owner,
                                     const char *domain,
                                     yvex_transformer_linear_operation operation,
                                     unsigned long long input,
                                     unsigned long long output)
{
    if (owner->requirement.input_width) return;
    owner->domain = domain;
    owner->requirement = (yvex_transformer_linear_requirement){
        .operation = operation,
        .publication_contract =
            YVEX_TRANSFORMER_LINEAR_NUMERIC_BF16_F32_ACCUMULATION,
        .source_dtype = YVEX_DTYPE_BF16,
        .input_dtype = YVEX_DTYPE_F32,
        .accumulation_dtype = YVEX_DTYPE_F32,
        .output_dtype = YVEX_DTYPE_F32,
        .publication_dtype = YVEX_DTYPE_BF16,
        .input_width = input,
        .output_width = output};
}

static int decoder_bind_common_layer(
    yvex_runtime_decoder_execution_context *context,
    const yvex_decoder_layer_plan *plan, decoder_layer_resources *resources,
    yvex_error *err)
{
    int rc = decoder_small_weight_bind(
        context, YVEX_TENSOR_ROLE_ATTENTION_NORM, plan->layer_index,
        plan->hidden_width,
        plan->normalization_weight_convention ==
            YVEX_NORMALIZATION_WEIGHT_ONE_PLUS,
        &resources->input_norm, err);
    if (rc == YVEX_OK)
        rc = decoder_small_weight_bind(
            context, YVEX_TENSOR_ROLE_FFN_NORM, plan->layer_index,
            plan->hidden_width,
            plan->normalization_weight_convention ==
                YVEX_NORMALIZATION_WEIGHT_ONE_PLUS,
            &resources->ffn_norm, err);
    if (rc == YVEX_OK)
        rc = decoder_weight_bind(context, YVEX_TENSOR_ROLE_FFN_GATE,
                                 plan->layer_index, &resources->ffn_gate, err);
    if (rc == YVEX_OK)
        rc = decoder_weight_bind(context, YVEX_TENSOR_ROLE_FFN_UP,
                                 plan->layer_index, &resources->ffn_up, err);
    if (rc == YVEX_OK)
        rc = decoder_weight_bind(context, YVEX_TENSOR_ROLE_FFN_DOWN,
                                 plan->layer_index, &resources->ffn_down, err);
    if (rc == YVEX_OK &&
        (!decoder_weight_shape(&resources->ffn_gate,
                               plan->intermediate_width, plan->hidden_width) ||
         !decoder_weight_shape(&resources->ffn_up,
                               plan->intermediate_width, plan->hidden_width) ||
         !decoder_weight_shape(&resources->ffn_down,
                               plan->hidden_width, plan->intermediate_width)))
        rc = decoder_refuse(err, YVEX_ERR_FORMAT,
                            "runtime.decoder.ffn-weights",
                            "dense FFN bindings differ from the decoder plan");
    decoder_linear_configure(
        &context->linears[DECODER_LINEAR_FFN_UP], "decoder.dense-ffn.up",
        YVEX_TRANSFORMER_LINEAR_OPERATION_GATE_UP, plan->hidden_width,
        plan->intermediate_width);
    decoder_linear_configure(
        &context->linears[DECODER_LINEAR_FFN_DOWN], "decoder.dense-ffn.down",
        YVEX_TRANSFORMER_LINEAR_OPERATION_DOWN, plan->intermediate_width,
        plan->hidden_width);
    return rc;
}

static int decoder_bind_attention_layer(
    yvex_runtime_decoder_execution_context *context,
    const yvex_decoder_layer_plan *plan, decoder_layer_resources *resources,
    yvex_error *err)
{
    const yvex_attention_layer_plan *attention = yvex_attention_plan_layer_at(
        context->model_view->attention, plan->attention_ordinal);
    unsigned long long query_width, kv_width, query_projection;
    int rc;

    if (!attention || attention->layer_index != plan->layer_index ||
        !yvex_core_u64_mul(attention->query_heads,
                           attention->head_dimension, &query_width) ||
        !yvex_core_u64_mul(attention->kv_heads,
                           attention->head_dimension, &kv_width) ||
        !yvex_core_u64_mul(query_width, 2ull, &query_projection))
        return decoder_refuse(err, YVEX_ERR_FORMAT,
                              "runtime.decoder.attention-plan",
                              "full-attention layer geometry is inconsistent");
    rc = decoder_weight_bind(context, YVEX_TENSOR_ROLE_ATTENTION_Q,
                             plan->layer_index, &resources->attention_q, err);
    if (rc == YVEX_OK)
        rc = decoder_weight_bind(context, YVEX_TENSOR_ROLE_ATTENTION_K,
                                 plan->layer_index, &resources->attention_k, err);
    if (rc == YVEX_OK)
        rc = decoder_weight_bind(context, YVEX_TENSOR_ROLE_ATTENTION_V,
                                 plan->layer_index, &resources->attention_v, err);
    if (rc == YVEX_OK)
        rc = decoder_weight_bind(context, YVEX_TENSOR_ROLE_ATTENTION_OUT,
                                 plan->layer_index, &resources->attention_out, err);
    if (rc == YVEX_OK)
        rc = decoder_small_weight_bind(
            context, YVEX_TENSOR_ROLE_ATTENTION_Q_NORM, plan->layer_index,
            attention->head_dimension, 1, &resources->attention_q_norm, err);
    if (rc == YVEX_OK)
        rc = decoder_small_weight_bind(
            context, YVEX_TENSOR_ROLE_ATTENTION_K_NORM, plan->layer_index,
            attention->head_dimension, 1, &resources->attention_k_norm, err);
    if (rc == YVEX_OK &&
        (!decoder_weight_shape(&resources->attention_q, query_projection,
                               plan->hidden_width) ||
         !decoder_weight_shape(&resources->attention_k, kv_width,
                               plan->hidden_width) ||
         !decoder_weight_shape(&resources->attention_v, kv_width,
                               plan->hidden_width) ||
         !decoder_weight_shape(&resources->attention_out, plan->hidden_width,
                               query_width)))
        rc = decoder_refuse(err, YVEX_ERR_FORMAT,
                            "runtime.decoder.attention-weights",
                            "full-attention bindings differ from the semantic geometry");
    decoder_linear_configure(
        &context->linears[DECODER_LINEAR_ATTENTION_Q],
        "decoder.full-attention.query-gate",
        YVEX_TRANSFORMER_LINEAR_OPERATION_QKV, plan->hidden_width,
        query_projection);
    decoder_linear_configure(
        &context->linears[DECODER_LINEAR_ATTENTION_KV],
        "decoder.full-attention.key-value",
        YVEX_TRANSFORMER_LINEAR_OPERATION_QKV, plan->hidden_width, kv_width);
    decoder_linear_configure(
        &context->linears[DECODER_LINEAR_ATTENTION_OUT],
        "decoder.full-attention.output",
        YVEX_TRANSFORMER_LINEAR_OPERATION_ATTENTION_OUTPUT, query_width,
        plan->hidden_width);
    return rc;
}

static int decoder_bind_delta_layer(
    yvex_runtime_decoder_execution_context *context,
    const yvex_decoder_layer_plan *plan, decoder_layer_resources *resources,
    yvex_error *err)
{
    const yvex_gated_delta_plan *delta = &plan->gated_delta;
    unsigned long long convolution_count;
    int rc;

    if (!yvex_core_u64_mul(delta->qkv_width,
                           delta->requirement.convolution_kernel,
                           &convolution_count))
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "runtime.decoder.delta-plan",
                              "gated-delta parameter geometry overflowed");
#define BIND_WEIGHT(role, field)                                                \
    do {                                                                        \
        if (rc == YVEX_OK)                                                      \
            rc = decoder_weight_bind(context, role, plan->layer_index,          \
                                     &resources->field, err);                    \
    } while (0)
#define BIND_SMALL(role, count, field)                                          \
    do {                                                                        \
        if (rc == YVEX_OK)                                                      \
            rc = decoder_small_weight_bind(context, role, plan->layer_index,    \
                                           count, 0, &resources->field, err);    \
    } while (0)
    rc = YVEX_OK;
    BIND_WEIGHT(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_QKV_PROJECTION, delta_qkv);
    BIND_WEIGHT(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_OUTPUT_GATE, delta_gate);
    BIND_WEIGHT(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_BETA_PROJECTION, delta_beta);
    BIND_WEIGHT(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_DECAY_PROJECTION, delta_decay);
    BIND_WEIGHT(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_OUTPUT, delta_out);
    BIND_SMALL(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_CONVOLUTION,
               convolution_count, delta_convolution);
    BIND_SMALL(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_DECAY_LOG,
               delta->requirement.value_heads, delta_decay_log);
    BIND_SMALL(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_TIME_BIAS,
               delta->requirement.value_heads, delta_time_bias);
    BIND_SMALL(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_OUTPUT_NORM,
               delta->requirement.value_head_dimension, delta_output_norm);
#undef BIND_SMALL
#undef BIND_WEIGHT
    if (rc == YVEX_OK &&
        (!decoder_weight_shape(&resources->delta_qkv, delta->qkv_width,
                               plan->hidden_width) ||
         !decoder_weight_shape(&resources->delta_gate, delta->value_width,
                               plan->hidden_width) ||
         !decoder_weight_shape(&resources->delta_beta,
                               delta->requirement.value_heads,
                               plan->hidden_width) ||
         !decoder_weight_shape(&resources->delta_decay,
                               delta->requirement.value_heads,
                               plan->hidden_width) ||
         !decoder_weight_shape(&resources->delta_out, plan->hidden_width,
                               delta->value_width)))
        rc = decoder_refuse(err, YVEX_ERR_FORMAT,
                            "runtime.decoder.delta-weights",
                            "gated-delta bindings differ from the semantic geometry");
    decoder_linear_configure(
        &context->linears[DECODER_LINEAR_DELTA_QKV],
        "decoder.gated-delta.qkv", YVEX_TRANSFORMER_LINEAR_OPERATION_QKV,
        plan->hidden_width, delta->qkv_width);
    decoder_linear_configure(
        &context->linears[DECODER_LINEAR_DELTA_VALUE],
        "decoder.gated-delta.value", YVEX_TRANSFORMER_LINEAR_OPERATION_PROJECTION,
        plan->hidden_width, delta->value_width);
    decoder_linear_configure(
        &context->linears[DECODER_LINEAR_DELTA_HEAD],
        "decoder.gated-delta.head", YVEX_TRANSFORMER_LINEAR_OPERATION_PROJECTION,
        plan->hidden_width, delta->requirement.value_heads);
    decoder_linear_configure(
        &context->linears[DECODER_LINEAR_DELTA_OUT],
        "decoder.gated-delta.output",
        YVEX_TRANSFORMER_LINEAR_OPERATION_ATTENTION_OUTPUT,
        delta->value_width, plan->hidden_width);
    return rc;
}

static int decoder_resources_bind(
    yvex_runtime_decoder_execution_context *context, yvex_error *err)
{
    unsigned long long layer;
    int rc = decoder_weight_bind(context, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
                                 YVEX_ATTENTION_NO_LAYER,
                                 &context->embedding, err);
    if (rc == YVEX_OK &&
        !decoder_weight_shape(&context->embedding, context->summary->vocabulary_size,
                              context->summary->hidden_width))
        rc = decoder_refuse(err, YVEX_ERR_FORMAT,
                            "runtime.decoder.embedding",
                            "token embedding differs from the decoder plan");
    if (rc == YVEX_OK)
        rc = decoder_small_weight_bind(
            context, YVEX_TENSOR_ROLE_OUTPUT_NORM, YVEX_ATTENTION_NO_LAYER,
            context->summary->hidden_width, 1, &context->output_norm, err);
    for (layer = 0ull; rc == YVEX_OK && layer < context->summary->layer_count;
         ++layer) {
        const yvex_decoder_layer_plan *plan =
            yvex_decoder_plan_layer_at(context->plan, layer);
        if (!plan)
            return decoder_refuse(err, YVEX_ERR_STATE,
                                  "runtime.decoder.layer-plan",
                                  "decoder layer plan is missing");
        rc = decoder_bind_common_layer(context, plan, &context->layers[layer], err);
        if (rc == YVEX_OK &&
            plan->mixer == YVEX_SEMANTIC_DECODER_MIXER_FULL_CAUSAL_ATTENTION)
            rc = decoder_bind_attention_layer(
                context, plan, &context->layers[layer], err);
        else if (rc == YVEX_OK &&
                 plan->mixer == YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA)
            rc = decoder_bind_delta_layer(
                context, plan, &context->layers[layer], err);
        else if (rc == YVEX_OK)
            rc = decoder_refuse(err, YVEX_ERR_UNSUPPORTED,
                                "runtime.decoder.layer-plan",
                                "decoder mixer is not executable");
    }
    return rc;
}

typedef struct {
    unsigned long long width[DECODER_BUFFER_COUNT];
    unsigned long long rotary_width, kv_width;
} decoder_buffer_geometry;

static void decoder_width_admit(unsigned long long *maximum,
                                unsigned long long candidate)
{
    if (candidate > *maximum) *maximum = candidate;
}

static int decoder_buffer_geometry_build(
    const yvex_runtime_decoder_execution_context *context,
    decoder_buffer_geometry *geometry)
{
    unsigned long long layer;
    memset(geometry, 0, sizeof(*geometry));
    geometry->width[DECODER_BUFFER_HIDDEN_A] = context->summary->hidden_width;
    geometry->width[DECODER_BUFFER_HIDDEN_B] = context->summary->hidden_width;
    geometry->width[DECODER_BUFFER_NORMALIZED] = context->summary->hidden_width;
    geometry->width[DECODER_BUFFER_UPDATE] = context->summary->hidden_width;
    geometry->width[DECODER_BUFFER_FFN_GATE] = context->summary->intermediate_width;
    geometry->width[DECODER_BUFFER_FFN_UP] = context->summary->intermediate_width;
    geometry->width[DECODER_BUFFER_FFN_PRODUCT] = context->summary->intermediate_width;
    for (layer = 0ull; layer < context->summary->layer_count; ++layer) {
        const yvex_decoder_layer_plan *plan =
            yvex_decoder_plan_layer_at(context->plan, layer);
        if (!plan) return 0;
        if (plan->mixer == YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA) {
            const yvex_gated_delta_plan *delta = &plan->gated_delta;
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_DELTA_QKV], delta->qkv_width);
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_DELTA_GATE], delta->value_width);
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_DELTA_BETA],
                delta->requirement.value_heads);
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_DELTA_DECAY],
                delta->requirement.value_heads);
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_MIXER_OUTPUT],
                delta->value_width);
        } else {
            const yvex_attention_layer_plan *attention =
                yvex_attention_plan_layer_at(
                    context->model_view->attention, plan->attention_ordinal);
            unsigned long long query, query_combined, kv;
            if (!attention ||
                !yvex_core_u64_mul(attention->query_heads,
                                   attention->head_dimension, &query) ||
                !yvex_core_u64_mul(query, 2ull, &query_combined) ||
                !yvex_core_u64_mul(attention->kv_heads,
                                   attention->head_dimension, &kv))
                return 0;
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_ATTENTION_Q_COMBINED],
                query_combined);
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_ATTENTION_Q], query);
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_ATTENTION_GATE], query);
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_ATTENTION_K], kv);
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_ATTENTION_V], kv);
            decoder_width_admit(
                &geometry->width[DECODER_BUFFER_MIXER_OUTPUT], query);
            decoder_width_admit(&geometry->rotary_width,
                                attention->rope_head_dimension);
            decoder_width_admit(&geometry->kv_width, kv);
        }
    }
    geometry->width[DECODER_BUFFER_COSINE] = geometry->rotary_width;
    geometry->width[DECODER_BUFFER_SINE] = geometry->rotary_width;
    return geometry->rotary_width && geometry->kv_width;
}

static int decoder_buffer_open(yvex_runtime_decoder_execution_context *context,
                               unsigned int slot, unsigned long long width,
                               yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long elements, bytes;
    if (!width || !yvex_core_u64_mul(context->options.token_capacity, width,
                                     &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes))
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "runtime.decoder.buffer",
                              "decoder buffer geometry overflowed");
    descriptor.name = "decoder-workspace";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 2u;
    descriptor.dims[0] = context->options.token_capacity;
    descriptor.dims[1] = width;
    descriptor.bytes = bytes;
    {
        int rc = yvex_backend_tensor_alloc(
            context->session_view->backend, &descriptor,
            &context->buffers[slot], err);
        if (rc == YVEX_OK)
            rc = decoder_bytes_account(
                &context->device_bytes, bytes,
                context->options.maximum_device_bytes,
                "runtime.decoder.buffer",
                "decoder workspace exceeds the device budget", err);
        if (rc != YVEX_OK && context->buffers[slot])
            (void)yvex_backend_tensor_release(
                context->session_view->backend, &context->buffers[slot],
                NULL);
        return rc;
    }
}

static int decoder_buffers_open(yvex_runtime_decoder_execution_context *context,
                                yvex_error *err)
{
    decoder_buffer_geometry geometry;
    unsigned long long values, positions_bytes, rope_values, rope_bytes;
    unsigned int slot;
    int rc = decoder_buffer_geometry_build(context, &geometry)
                 ? YVEX_OK
                 : decoder_refuse(err, YVEX_ERR_FORMAT,
                                  "runtime.decoder.buffer-plan",
                                  "decoder buffer maxima are incomplete");
    for (slot = 0u; rc == YVEX_OK && slot < DECODER_BUFFER_COUNT; ++slot)
        rc = decoder_buffer_open(context, slot, geometry.width[slot], err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(context->options.token_capacity,
                            geometry.kv_width, &values) ||
         !yvex_core_u64_mul(values, 4ull, &context->state_workspace_values) ||
         context->state_workspace_values > SIZE_MAX / sizeof(float) ||
         !yvex_core_u64_mul(context->options.token_capacity,
                            sizeof(unsigned long long), &positions_bytes) ||
         positions_bytes > SIZE_MAX ||
         !yvex_core_u64_mul(context->options.token_capacity,
                            geometry.rotary_width, &rope_values) ||
         !yvex_core_u64_mul(rope_values, 2ull,
                            &context->rope_workspace_values) ||
         !yvex_core_u64_mul(context->rope_workspace_values, sizeof(float),
                            &rope_bytes) || rope_bytes > SIZE_MAX))
        rc = decoder_refuse(err, YVEX_ERR_BOUNDS,
                            "runtime.decoder.host-workspace",
                            "decoder publication workspace overflowed");
    if (rc == YVEX_OK) {
        context->state_workspace = calloc(
            (size_t)context->state_workspace_values, sizeof(float));
        context->position_workspace = malloc((size_t)positions_bytes);
        context->rope_workspace = malloc((size_t)rope_bytes);
        if (!context->state_workspace || !context->position_workspace ||
            !context->rope_workspace)
            rc = decoder_refuse(err, YVEX_ERR_NOMEM,
                                "runtime.decoder.host-workspace",
                                "decoder publication workspace allocation failed");
    }
    if (rc == YVEX_OK)
        rc = decoder_bytes_account(
            &context->host_bytes,
            context->state_workspace_values * sizeof(float),
            context->options.maximum_host_bytes,
            "runtime.decoder.host-workspace",
            "decoder workspace exceeds the host budget", err);
    if (rc == YVEX_OK)
        rc = decoder_bytes_account(
            &context->host_bytes, positions_bytes,
            context->options.maximum_host_bytes,
            "runtime.decoder.host-workspace",
            "decoder workspace exceeds the host budget", err);
    if (rc == YVEX_OK)
        rc = decoder_bytes_account(
            &context->host_bytes, rope_bytes,
            context->options.maximum_host_bytes,
            "runtime.decoder.host-workspace",
            "decoder workspace exceeds the host budget", err);
    return rc;
}

static int decoder_operations_valid(
    const yvex_backend_transformer_operations *operations)
{
    return operations && operations->attention_execute &&
           operations->gated_delta_execute && operations->linear_compile &&
           operations->linear_execute && operations->linear_release &&
           operations->rotary_half_f32 &&
           operations->split_interleaved_two_f32 &&
           operations->silu_product_bf16 &&
           operations->sigmoid_product_bf16 && operations->add_bf16 &&
           operations->bf16_round;
}

int yvex_runtime_decoder_execution_context_open(
    yvex_runtime_decoder_execution_context **out, yvex_model_engine *model,
    yvex_runtime_execution_session *session,
    const yvex_runtime_decoder_execution_options *options, yvex_error *err)
{
    yvex_runtime_decoder_execution_context *context;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (!out || !model || !session || !options || !options->context_capacity ||
        !options->token_capacity ||
        options->token_capacity > options->context_capacity ||
        !options->execution_profile)
        return decoder_refuse(err, YVEX_ERR_INVALID_ARG,
                              "runtime.decoder.open",
                              "decoder model, session, capacity, and execution profile are required");
    context = calloc(1u, sizeof(*context));
    if (!context)
        return decoder_refuse(err, YVEX_ERR_NOMEM, "runtime.decoder.open",
                              "decoder context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_model_engine_view_get(model);
    context->session_view = yvex_runtime_session_view_get(session);
    context->options = *options;
    context->plan = context->model_view ? context->model_view->decoder : NULL;
    context->summary = yvex_decoder_plan_summary_get(context->plan);
    context->operations = context->session_view
                              ? yvex_backend_transformer_operations_get(
                                    context->session_view->backend)
                              : NULL;
    if (!context->model_view || !context->session_view ||
        context->session_view->engine != model || !context->summary ||
        context->summary->maximum_context < options->context_capacity ||
        yvex_backend_kind_of(context->session_view->backend) !=
            YVEX_BACKEND_KIND_CUDA ||
        !context->session_view->sequence_state ||
        !decoder_operations_valid(context->operations) ||
        pthread_mutex_init(&context->mutex, NULL) != 0)
        rc = decoder_refuse(err, YVEX_ERR_STATE, "runtime.decoder.open",
                            "admitted CUDA decoder resources are unavailable");
    else
        context->mutex_ready = 1;
    if (rc == YVEX_OK) {
        context->layers = calloc((size_t)context->summary->layer_count,
                                 sizeof(*context->layers));
        if (!context->layers)
            rc = decoder_refuse(err, YVEX_ERR_NOMEM, "runtime.decoder.open",
                                "decoder layer directory allocation failed");
    }
    if (rc == YVEX_OK) rc = decoder_resources_bind(context, err);
    if (rc == YVEX_OK) rc = decoder_buffers_open(context, err);
    if (rc != YVEX_OK) {
        (void)yvex_runtime_decoder_execution_context_close(&context, NULL);
        return rc;
    }
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
}

const yvex_decoder_plan *yvex_runtime_decoder_execution_plan(
    const yvex_runtime_decoder_execution_context *context)
{
    return context ? context->plan : NULL;
}

static int decoder_small_weight_close(yvex_backend *backend,
                                      decoder_small_weight *weight,
                                      yvex_error *err)
{
    if (!weight) return YVEX_OK;
    if (weight->device &&
        yvex_backend_tensor_release(backend, &weight->device, err) != YVEX_OK)
        return yvex_error_code(err);
    free(weight->host);
    memset(weight, 0, sizeof(*weight));
    return YVEX_OK;
}

int yvex_runtime_decoder_execution_context_close(
    yvex_runtime_decoder_execution_context **context_ptr, yvex_error *err)
{
    yvex_runtime_decoder_execution_context *context;
    yvex_backend *backend;
    unsigned long long layer;
    unsigned int index;
    int rc = YVEX_OK;
    if (!context_ptr || !*context_ptr) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    context = *context_ptr;
    if (context->mutex_ready && pthread_mutex_lock(&context->mutex) == 0) {
        if (context->busy) {
            (void)pthread_mutex_unlock(&context->mutex);
            return decoder_refuse(err, YVEX_ERR_STATE,
                                  "runtime.decoder.close",
                                  "busy decoder context cannot close");
        }
        (void)pthread_mutex_unlock(&context->mutex);
    }
    backend = context->session_view ? context->session_view->backend : NULL;
    for (index = 0u; backend && index < DECODER_LINEAR_COUNT; ++index) {
        if (context->linears[index].single && context->operations)
            rc = context->operations->linear_release(
                backend, &context->linears[index].single, err);
        if (rc == YVEX_OK && context->linears[index].multiple &&
            context->operations)
            rc = context->operations->linear_release(
                backend, &context->linears[index].multiple, err);
        if (rc != YVEX_OK) return rc;
    }
    for (index = 0u; backend && index < DECODER_BUFFER_COUNT; ++index)
        if (context->buffers[index] &&
            yvex_backend_tensor_release(backend, &context->buffers[index],
                                        err) != YVEX_OK)
            return yvex_error_code(err);
    rc = decoder_small_weight_close(backend, &context->output_norm, err);
    for (layer = 0ull; context->layers && layer < context->summary->layer_count;
         ++layer) {
        decoder_layer_resources *resources = &context->layers[layer];
        decoder_small_weight *small[] = {
            &resources->input_norm, &resources->ffn_norm,
            &resources->attention_q_norm, &resources->attention_k_norm,
            &resources->delta_convolution, &resources->delta_decay_log,
            &resources->delta_time_bias, &resources->delta_output_norm};
        unsigned int small_index;
        for (small_index = 0u;
             rc == YVEX_OK &&
             small_index < sizeof(small) / sizeof(small[0]);
             ++small_index)
            rc = decoder_small_weight_close(backend, small[small_index], err);
    }
    if (rc != YVEX_OK) return rc;
    free(context->state_workspace);
    free(context->position_workspace);
    free(context->rope_workspace);
    free(context->layers);
    if (context->mutex_ready) (void)pthread_mutex_destroy(&context->mutex);
    memset(context, 0, sizeof(*context));
    free(context);
    *context_ptr = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int decoder_tensor_view(yvex_device_tensor *source, unsigned long long count,
                        unsigned long long rows, unsigned long long columns,
                        yvex_device_tensor *out)
{
    unsigned long long expected;
    if (!source || !out || !rows || !columns ||
        !yvex_core_u64_mul(rows, columns, &expected) || expected != count ||
        !yvex_backend_tensor_f32_subview(source, 0ull, count, out))
        return 0;
    out->rank = 2u;
    out->dims[0] = rows;
    out->dims[1] = columns;
    return 1;
}

static int decoder_linear_compile(
    yvex_runtime_decoder_execution_context *context,
    decoder_linear_owner *owner, unsigned long long rows,
    yvex_transformer_linear_executable **out, yvex_error *err)
{
    yvex_transformer_linear_compile_request request = {0};
    yvex_transformer_linear_executable_summary summary = {0};
    int rc;

    if (!owner || !owner->requirement.input_width || !owner->domain || !rows)
        return decoder_refuse(err, YVEX_ERR_STATE,
                              "runtime.decoder.linear-compile",
                              "sealed decoder linear requirement is missing");
    if (*out) {
        rc = context->operations->linear_release(
            context->session_view->backend, out, err);
        if (rc != YVEX_OK) return rc;
    }
    request.semantic_domain = owner->domain;
    request.requirement = &owner->requirement;
    request.input_rows = rows;
    rc = context->operations->linear_compile(
        context->session_view->backend, &request, out, &summary, err);
    if (rc == YVEX_OK &&
        (!summary.exact || summary.input_rows != rows ||
         !yvex_sha256_hex_valid(summary.identity)))
        rc = decoder_refuse(err, YVEX_ERR_STATE,
                            "runtime.decoder.linear-compile",
                            "H30 returned an inexact decoder linear executable");
    return rc;
}

static int decoder_linear_execute(yvex_runtime_decoder_execution_context *context,
                           unsigned int slot, unsigned long long rows,
                           const decoder_weight *weight,
                           const yvex_device_tensor *input,
                           yvex_device_tensor *output,
                           yvex_backend_operation_facts *facts,
                           yvex_error *err)
{
    decoder_linear_owner *owner;
    yvex_transformer_linear_executable **executable;
    yvex_transformer_linear_execution_request request = {0};
    int rc;

    if (facts) memset(facts, 0, sizeof(*facts));
    if (!context || slot >= DECODER_LINEAR_COUNT || !rows || !weight ||
        !weight->binding || !input || !output || !facts)
        return decoder_refuse(err, YVEX_ERR_INVALID_ARG,
                              "runtime.decoder.linear",
                              "decoder linear execution owners are incomplete");
    owner = &context->linears[slot];
    executable = rows == 1ull ? &owner->single : &owner->multiple;
    if (!*executable || (rows != 1ull && owner->multiple_rows != rows)) {
        rc = decoder_linear_compile(context, owner, rows, executable, err);
        if (rc != YVEX_OK) return rc;
        if (rows != 1ull) owner->multiple_rows = rows;
    }
    request.executable = *executable;
    request.weight = &weight->encoded;
    request.input = input;
    request.output = output;
    return context->operations->linear_execute(
        context->session_view->backend, &request, facts, err);
}

static int decoder_result_facts_add(yvex_runtime_decoder_execution_result *result,
                             const yvex_backend_operation_facts *facts,
                             yvex_error *err)
{
    if (!result || !facts ||
        !yvex_core_u64_add(result->kernel_launches, facts->kernel_launches,
                           &result->kernel_launches) ||
        !yvex_core_u64_add(result->h2d_bytes, facts->h2d_bytes,
                           &result->h2d_bytes) ||
        !yvex_core_u64_add(result->d2h_bytes, facts->d2h_bytes,
                           &result->d2h_bytes) ||
        !yvex_core_u64_add(result->d2d_bytes, facts->d2d_bytes,
                           &result->d2d_bytes) ||
        !yvex_core_u64_add(result->accelerated_matrix_operations,
                           facts->accelerated_matrix_launches,
                           &result->accelerated_matrix_operations))
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "runtime.decoder.operation-facts",
                              "decoder execution accounting overflowed");
    return YVEX_OK;
}

static int decoder_operation_add(decoder_layer_run *run,
                                 const yvex_backend_operation_facts *facts,
                                 yvex_error *err)
{
    return decoder_result_facts_add(run->result, facts, err);
}

static int decoder_round(decoder_layer_run *run, yvex_device_tensor *tensor,
                         unsigned long long count, yvex_error *err)
{
    yvex_backend_operation_facts facts = {0};
    int rc = run->context->operations->bf16_round(
        run->context->session_view->backend, tensor, count, &facts, err);
    return rc == YVEX_OK ? decoder_operation_add(run, &facts, err) : rc;
}

static int decoder_normalize(decoder_layer_run *run,
                      const yvex_device_tensor *input,
                      const decoder_small_weight *weight, double epsilon,
                      yvex_device_tensor *output, unsigned long long rows,
                      unsigned long long width, yvex_error *err)
{
    int rc;
    output->rank = 2u;
    output->dims[0] = rows;
    output->dims[1] = width;
    rc = yvex_backend_op_rms_norm(
        run->context->session_view->backend, input, weight->device,
        (float)epsilon, output, err);
    return rc == YVEX_OK ? decoder_round(run, output, rows * width, err) : rc;
}

static int decoder_linear(decoder_layer_run *run, unsigned int slot,
                          const decoder_weight *weight,
                          const yvex_device_tensor *input,
                          yvex_device_tensor *output, yvex_error *err)
{
    yvex_backend_operation_facts facts = {0};
    int rc = decoder_linear_execute(
        run->context, slot, run->request->token_count, weight, input, output,
        &facts, err);
    if (rc == YVEX_OK) {
        run->result->linear_operations++;
        rc = decoder_operation_add(run, &facts, err);
    }
    return rc;
}

static int decoder_binary(decoder_layer_run *run,
                          int (*operation)(
                              yvex_backend *, const yvex_device_tensor *,
                              const yvex_device_tensor *, yvex_device_tensor *,
                              unsigned long long,
                              yvex_backend_operation_facts *, yvex_error *),
                          const yvex_device_tensor *left,
                          const yvex_device_tensor *right,
                          yvex_device_tensor *output, unsigned long long count,
                          yvex_error *err)
{
    yvex_backend_operation_facts facts = {0};
    int rc = operation(run->context->session_view->backend, left, right,
                       output, count, &facts, err);
    return rc == YVEX_OK ? decoder_operation_add(run, &facts, err) : rc;
}

static int decoder_add(decoder_layer_run *run,
                       const yvex_device_tensor *left,
                       const yvex_device_tensor *right,
                       yvex_device_tensor *output, unsigned long long rows,
                       unsigned long long width, yvex_error *err)
{
    yvex_backend_operation_facts facts = {0};
    int rc = run->context->operations->add_bf16(
        run->context->session_view->backend, left, right, output, rows, width,
        &facts, err);
    return rc == YVEX_OK ? decoder_operation_add(run, &facts, err) : rc;
}

static int decoder_rope_tables(decoder_layer_run *run,
                               const yvex_attention_layer_plan *attention,
                               yvex_device_tensor *cosine,
                               yvex_device_tensor *sine, yvex_error *err)
{
    unsigned long long tokens = run->request->token_count;
    unsigned long long width = attention->rope_head_dimension;
    unsigned long long values, required_values, bytes, token, coordinate;
    float *cosines = run->context->rope_workspace;
    float *sines;
    int rc;

    if (!width || (width & 1ull) ||
        !yvex_core_u64_mul(tokens, width, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &bytes) ||
        !yvex_core_u64_mul(values, 2ull, &required_values) ||
        required_values > run->context->rope_workspace_values)
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "runtime.decoder.rope",
                              "partial rotary table exceeds its workspace");
    sines = cosines + values;
    for (token = 0ull; token < tokens; ++token) {
        unsigned long long position = run->request->token_start + token;
        for (coordinate = 0ull; coordinate < width; ++coordinate) {
            unsigned long long pair = coordinate % (width / 2ull);
            double frequency = pow(
                (double)attention->position.theta,
                -(double)(2ull * pair) / (double)width);
            double angle = (double)position * frequency;
            unsigned long long index = token * width + coordinate;
            cosines[index] = yvex_quant_bf16_decode(
                yvex_quant_bf16_encode((float)cos(angle)));
            sines[index] = yvex_quant_bf16_decode(
                yvex_quant_bf16_encode((float)sin(angle)));
        }
    }
    rc = yvex_backend_tensor_write(run->context->session_view->backend,
                                   cosine, cosines, bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->context->session_view->backend,
                                       sine, sines, bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->result->h2d_bytes, bytes,
                            &run->result->h2d_bytes) ||
         !yvex_core_u64_add(run->result->h2d_bytes, bytes,
                            &run->result->h2d_bytes)))
        rc = decoder_refuse(err, YVEX_ERR_BOUNDS,
                            "runtime.decoder.rope",
                            "rotary table accounting overflowed");
    return rc;
}

static int decoder_rotary(decoder_layer_run *run, yvex_device_tensor *values,
                          const yvex_device_tensor *cosine,
                          const yvex_device_tensor *sine,
                          unsigned long long heads,
                          unsigned long long head_dimension,
                          unsigned long long rotary_dimension,
                          yvex_error *err)
{
    yvex_backend_operation_facts facts = {0};
    int rc = run->context->operations->rotary_half_f32(
        run->context->session_view->backend, values, cosine, sine,
        run->request->token_count, heads, head_dimension, rotary_dimension,
        &facts, err);
    if (rc == YVEX_OK) rc = decoder_operation_add(run, &facts, err);
    if (rc == YVEX_OK)
        rc = decoder_round(run, values,
                           run->request->token_count * heads * head_dimension,
                           err);
    return rc;
}

static int decoder_attention_projections(
    decoder_layer_run *run, const yvex_attention_layer_plan *attention,
    decoder_layer_resources *resources, yvex_device_tensor *query,
    yvex_device_tensor *gate, yvex_device_tensor *key,
    yvex_device_tensor *value, yvex_error *err)
{
    unsigned long long tokens = run->request->token_count;
    unsigned long long query_width = attention->query_heads *
                                     attention->head_dimension;
    unsigned long long kv_width = attention->kv_heads *
                                  attention->head_dimension;
    yvex_device_tensor combined;
    yvex_backend_operation_facts facts = {0};
    int rc;

    if (!decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_ATTENTION_Q_COMBINED],
            tokens * query_width * 2ull, tokens, query_width * 2ull,
            &combined) ||
        !decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_ATTENTION_Q],
            tokens * query_width, tokens * attention->query_heads,
            attention->head_dimension, query) ||
        !decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_ATTENTION_GATE],
            tokens * query_width, tokens, query_width, gate) ||
        !decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_ATTENTION_K],
            tokens * kv_width, tokens * attention->kv_heads,
            attention->head_dimension, key) ||
        !decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_ATTENTION_V],
            tokens * kv_width, tokens, kv_width, value))
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "runtime.decoder.attention-projections",
                              "attention projection views exceed sealed buffers");
    rc = decoder_linear(run, DECODER_LINEAR_ATTENTION_Q,
                        &resources->attention_q, &run->normalized, &combined,
                        err);
    if (rc == YVEX_OK)
        rc = run->context->operations->split_interleaved_two_f32(
            run->context->session_view->backend, &combined, query, gate,
            tokens, attention->query_heads, attention->head_dimension, &facts,
            err);
    if (rc == YVEX_OK) rc = decoder_operation_add(run, &facts, err);
    if (rc == YVEX_OK)
        rc = decoder_linear(run, DECODER_LINEAR_ATTENTION_KV,
                            &resources->attention_k, &run->normalized, key,
                            err);
    if (rc == YVEX_OK)
        rc = decoder_linear(run, DECODER_LINEAR_ATTENTION_KV,
                            &resources->attention_v, &run->normalized, value,
                            err);
    if (rc == YVEX_OK)
        rc = decoder_normalize(
            run, query, &resources->attention_q_norm, 1.0e-6, query,
            tokens * attention->query_heads, attention->head_dimension, err);
    if (rc == YVEX_OK)
        rc = decoder_normalize(
            run, key, &resources->attention_k_norm, 1.0e-6, key,
            tokens * attention->kv_heads, attention->head_dimension, err);
    return rc;
}

static int decoder_full_attention(
    decoder_layer_run *run, const yvex_decoder_layer_plan *plan,
    decoder_layer_resources *resources, yvex_error *err)
{
    const yvex_attention_layer_plan *attention = yvex_attention_plan_layer_at(
        run->context->model_view->attention, plan->attention_ordinal);
    const yvex_attention_summary *summary = yvex_attention_plan_summary(
        run->context->model_view->attention);
    unsigned long long tokens = run->request->token_count;
    unsigned long long query_width, rotary_width;
    yvex_device_tensor query, gate, key, value, mixer, gated, cosine, sine;
    yvex_runtime_stateful_attention_request request = {0};
    yvex_runtime_stateful_attention_result result = {0};
    yvex_attention_failure failure = {0};
    int rc;

    if (!attention || !summary ||
        !(query_width = attention->query_heads * attention->head_dimension) ||
        !(rotary_width = attention->rope_head_dimension) ||
        !decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_MIXER_OUTPUT],
            tokens * query_width, tokens, query_width, &mixer) ||
        !decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_ATTENTION_Q_COMBINED],
            tokens * query_width, tokens, query_width, &gated) ||
        !decoder_tensor_view(run->context->buffers[DECODER_BUFFER_COSINE],
                             tokens * rotary_width, tokens, rotary_width,
                             &cosine) ||
        !decoder_tensor_view(run->context->buffers[DECODER_BUFFER_SINE],
                             tokens * rotary_width, tokens, rotary_width,
                             &sine))
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "runtime.decoder.attention",
                              "full-attention workspace views are invalid");
    rc = decoder_attention_projections(
        run, attention, resources, &query, &gate, &key, &value, err);
    if (rc == YVEX_OK)
        rc = decoder_rope_tables(run, attention, &cosine, &sine, err);
    if (rc == YVEX_OK)
        rc = decoder_rotary(run, &query, &cosine, &sine,
                            attention->query_heads,
                            attention->head_dimension, rotary_width, err);
    if (rc == YVEX_OK)
        rc = decoder_rotary(run, &key, &cosine, &sine,
                            attention->kv_heads,
                            attention->head_dimension, rotary_width, err);
    request.backend = run->context->session_view->backend;
    request.state = run->context->session_view->attention_state_provider;
    request.residency = run->context->session_view->state_residency;
    request.layer = attention;
    request.attention_plan_identity = summary->attention_plan_identity;
    request.input_identity = run->request->input_identity;
    request.layer_ordinal = plan->attention_ordinal;
    request.token_position = run->request->token_start;
    request.token_count = tokens;
    request.query = &query;
    request.key = &key;
    request.value = &value;
    request.output = &mixer;
    request.host_workspace = run->context->state_workspace;
    request.host_workspace_values = run->context->state_workspace_values;
    request.host_positions = run->context->position_workspace;
    request.host_position_capacity = run->context->options.token_capacity;
    request.cancellation.requested = run->context->options.cancel_requested;
    request.cancellation.context = run->context->options.cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_stateful_attention_execute(
            &request, &result, &failure, err);
    if (rc == YVEX_OK) rc = decoder_operation_add(run, &result.attention, err);
    if (rc == YVEX_OK)
        rc = decoder_binary(
            run, run->context->operations->sigmoid_product_bf16, &mixer,
            &gate, &gated, tokens * query_width, err);
    if (rc == YVEX_OK)
        rc = decoder_linear(run, DECODER_LINEAR_ATTENTION_OUT,
                            &resources->attention_out, &gated, &run->update,
                            err);
    if (rc == YVEX_OK) run->result->attention_layers++;
    return rc;
}

static int decoder_delta_projections(
    decoder_layer_run *run, const yvex_gated_delta_plan *delta,
    decoder_layer_resources *resources, yvex_device_tensor *qkv,
    yvex_device_tensor *gate, yvex_device_tensor *beta,
    yvex_device_tensor *decay, yvex_device_tensor *output, yvex_error *err)
{
    unsigned long long tokens = run->request->token_count;
    unsigned long long heads = delta->requirement.value_heads;
    if (!decoder_tensor_view(run->context->buffers[DECODER_BUFFER_DELTA_QKV],
                             tokens * delta->qkv_width, tokens,
                             delta->qkv_width, qkv) ||
        !decoder_tensor_view(run->context->buffers[DECODER_BUFFER_DELTA_GATE],
                             tokens * delta->value_width, tokens,
                             delta->value_width, gate) ||
        !decoder_tensor_view(run->context->buffers[DECODER_BUFFER_DELTA_BETA],
                             tokens * heads, tokens, heads, beta) ||
        !decoder_tensor_view(run->context->buffers[DECODER_BUFFER_DELTA_DECAY],
                             tokens * heads, tokens, heads, decay) ||
        !decoder_tensor_view(run->context->buffers[DECODER_BUFFER_MIXER_OUTPUT],
                             tokens * delta->value_width, tokens,
                             delta->value_width, output))
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "runtime.decoder.delta-projections",
                              "gated-delta projection views exceed sealed buffers");
    if (decoder_linear(run, DECODER_LINEAR_DELTA_QKV, &resources->delta_qkv,
                       &run->normalized, qkv, err) != YVEX_OK ||
        decoder_linear(run, DECODER_LINEAR_DELTA_VALUE, &resources->delta_gate,
                       &run->normalized, gate, err) != YVEX_OK ||
        decoder_linear(run, DECODER_LINEAR_DELTA_HEAD, &resources->delta_beta,
                       &run->normalized, beta, err) != YVEX_OK ||
        decoder_linear(run, DECODER_LINEAR_DELTA_HEAD, &resources->delta_decay,
                       &run->normalized, decay, err) != YVEX_OK)
        return yvex_error_code(err);
    return YVEX_OK;
}

static int decoder_gated_delta(
    decoder_layer_run *run, const yvex_decoder_layer_plan *plan,
    decoder_layer_resources *resources, yvex_error *err)
{
    const yvex_gated_delta_plan *delta = &plan->gated_delta;
    yvex_sequence_device_state_view committed = {0};
    yvex_sequence_device_state_output candidate = {0};
    yvex_gated_delta_device_request request = {0};
    yvex_gated_delta_device_result result = {0};
    yvex_backend_operation_facts facts = {0};
    yvex_device_tensor qkv, gate, beta, decay, output;
    int rc = decoder_delta_projections(
        run, delta, resources, &qkv, &gate, &beta, &decay, &output, err);
    if (rc == YVEX_OK)
        rc = yvex_sequence_state_device_layer(
            run->context->session_view->sequence_state, plan->layer_index,
            &committed, &candidate, err);
    request.token_count = run->request->token_count;
    request.projected_qkv = &qkv;
    request.projected_output_gate = &gate;
    request.projected_beta = &beta;
    request.projected_decay = &decay;
    request.convolution_weight = resources->delta_convolution.device;
    request.decay_log = resources->delta_decay_log.device;
    request.time_bias = resources->delta_time_bias.device;
    request.normalization_weight = resources->delta_output_norm.device;
    request.convolution_state = committed.convolution;
    request.recurrent_state = committed.recurrent;
    request.next_convolution_state = candidate.convolution;
    request.next_recurrent_state = candidate.recurrent;
    request.output = &output;
    request.cancel_requested = run->context->options.cancel_requested;
    request.cancel_context = run->context->options.cancel_context;
    if (rc == YVEX_OK)
        rc = run->context->operations->gated_delta_execute(
            run->context->session_view->backend, delta, &request, &result,
            &facts, err);
    if (rc == YVEX_OK) rc = decoder_operation_add(run, &facts, err);
    if (rc == YVEX_OK)
        rc = decoder_round(run, &output,
                           run->request->token_count * delta->value_width,
                           err);
    if (rc == YVEX_OK)
        rc = yvex_sequence_state_stage(
            run->context->session_view->sequence_state, plan->layer_index,
            err);
    if (rc == YVEX_OK)
        rc = decoder_linear(run, DECODER_LINEAR_DELTA_OUT,
                            &resources->delta_out, &output, &run->update, err);
    if (rc == YVEX_OK) run->result->recurrent_layers++;
    return rc;
}

static int decoder_ffn(decoder_layer_run *run,
                       const yvex_decoder_layer_plan *plan,
                       decoder_layer_resources *resources, yvex_error *err)
{
    unsigned long long tokens = run->request->token_count;
    unsigned long long values = tokens * plan->intermediate_width;
    yvex_device_tensor gate, up, product;
    int rc;
    if (!decoder_tensor_view(run->context->buffers[DECODER_BUFFER_FFN_GATE],
                             values, tokens, plan->intermediate_width, &gate) ||
        !decoder_tensor_view(run->context->buffers[DECODER_BUFFER_FFN_UP],
                             values, tokens, plan->intermediate_width, &up) ||
        !decoder_tensor_view(run->context->buffers[DECODER_BUFFER_FFN_PRODUCT],
                             values, tokens, plan->intermediate_width,
                             &product))
        return decoder_refuse(err, YVEX_ERR_BOUNDS, "runtime.decoder.ffn",
                              "dense FFN workspace views are invalid");
    rc = decoder_linear(run, DECODER_LINEAR_FFN_UP, &resources->ffn_gate,
                        &run->normalized, &gate, err);
    if (rc == YVEX_OK)
        rc = decoder_linear(run, DECODER_LINEAR_FFN_UP, &resources->ffn_up,
                            &run->normalized, &up, err);
    if (rc == YVEX_OK)
        rc = decoder_binary(run, run->context->operations->silu_product_bf16,
                            &gate, &up, &product, values, err);
    if (rc == YVEX_OK)
        rc = decoder_linear(run, DECODER_LINEAR_FFN_DOWN,
                            &resources->ffn_down, &product, &run->update, err);
    return rc;
}

static void decoder_swap_hidden(decoder_layer_run *run)
{
    yvex_device_tensor swap = run->current;
    run->current = run->next;
    run->next = swap;
}

static int decoder_layer_execute(decoder_layer_run *run,
                          const yvex_decoder_layer_plan *plan,
                          decoder_layer_resources *resources,
                          yvex_error *err)
{
    unsigned long long rows = run->request->token_count;
    unsigned long long width = plan->hidden_width;
    int rc = decoder_normalize(
        run, &run->current, &resources->input_norm,
        plan->normalization_epsilon, &run->normalized, rows, width, err);
    if (rc == YVEX_OK &&
        plan->mixer == YVEX_SEMANTIC_DECODER_MIXER_FULL_CAUSAL_ATTENTION)
        rc = decoder_full_attention(run, plan, resources, err);
    else if (rc == YVEX_OK &&
             plan->mixer == YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA)
        rc = decoder_gated_delta(run, plan, resources, err);
    if (rc == YVEX_OK)
        rc = decoder_add(run, &run->current, &run->update, &run->next, rows,
                         width, err);
    if (rc == YVEX_OK) decoder_swap_hidden(run);
    if (rc == YVEX_OK)
        rc = decoder_normalize(
            run, &run->current, &resources->ffn_norm,
            plan->normalization_epsilon, &run->normalized, rows, width, err);
    if (rc == YVEX_OK) rc = decoder_ffn(run, plan, resources, err);
    if (rc == YVEX_OK)
        rc = decoder_add(run, &run->current, &run->update, &run->next, rows,
                         width, err);
    if (rc == YVEX_OK) {
        decoder_swap_hidden(run);
        run->result->layers_executed++;
    }
    return rc;
}

static int decoder_enter(yvex_runtime_decoder_execution_context *context,
                         yvex_error *err)
{
    if (!context || !context->mutex_ready ||
        pthread_mutex_lock(&context->mutex) != 0)
        return decoder_refuse(err, YVEX_ERR_STATE, "runtime.decoder.enter",
                              "synchronized decoder context is required");
    if (context->busy || context->invalidated) {
        (void)pthread_mutex_unlock(&context->mutex);
        return decoder_refuse(err, YVEX_ERR_STATE, "runtime.decoder.enter",
                              "decoder context is busy or invalidated");
    }
    context->busy = 1;
    (void)pthread_mutex_unlock(&context->mutex);
    return YVEX_OK;
}

static void decoder_leave(yvex_runtime_decoder_execution_context *context)
{
    yvex_runtime_session_summary session = {0};
    int invalidated =
        context &&
        yvex_runtime_session_summary_copy(context->session, &session, NULL) ==
            YVEX_OK &&
        session.invalidated;
    if (!context || !context->mutex_ready ||
        pthread_mutex_lock(&context->mutex) != 0)
        return;
    context->invalidated |= invalidated;
    context->busy = 0;
    (void)pthread_mutex_unlock(&context->mutex);
}

static int decoder_state_summary(
    const yvex_runtime_decoder_execution_context *context,
    yvex_graph_attention_state_summary *attention,
    yvex_sequence_state_summary *sequence, yvex_error *err)
{
    const yvex_attention_state_provider *provider =
        context && context->session_view
            ? context->session_view->attention_state_provider
            : NULL;
    if (!provider || !provider->summary ||
        provider->summary(provider->context, attention, err) != YVEX_OK ||
        yvex_sequence_state_summary_copy(context->session_view->sequence_state,
                                         sequence, err) != YVEX_OK)
        return decoder_refuse(err, YVEX_ERR_STATE, "runtime.decoder.state",
                              "mixed decoder sequence state is unavailable");
    return YVEX_OK;
}

static int decoder_prepare_attention_state(
    yvex_runtime_decoder_execution_context *context,
    const yvex_runtime_decoder_execution_request *request,
    yvex_graph_attention_state_summary *attention,
    yvex_sequence_state_summary *sequence, yvex_error *err)
{
    yvex_graph_attention_capacity_request capacity_request = {0};
    yvex_graph_attention_capacity_plan *capacity = NULL;
    yvex_attention_failure failure = {0};
    int rc;

    if (!context || !request || !attention || !sequence)
        return decoder_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.decoder.state-prepare",
            "decoder state preparation requires complete execution facts");
    if (attention->prepared_layer_count) return YVEX_OK;
    if (request->token_start)
        return decoder_refuse(
            err, YVEX_ERR_STATE, "runtime.decoder.state-prepare",
            "nonzero decoder start requires committed attention state");

    capacity_request.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
    capacity_request.token_count = context->options.context_capacity;
    capacity_request.execution_count = 1ull;
    capacity_request.use_requested_position = 1;
    rc = yvex_graph_attention_capacity_plan_build(
        &capacity, context->model_view->attention, &capacity_request, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_prepare_attention_scope_state(
            context->session, context->model, YVEX_TENSOR_SCOPE_GLOBAL,
            capacity, &failure, err);
    yvex_graph_attention_capacity_plan_close(&capacity);
    if (rc == YVEX_OK)
        rc = decoder_state_summary(context, attention, sequence, err);
    return rc;
}

static int decoder_request_validate(
    const yvex_runtime_decoder_execution_context *context,
    const yvex_runtime_decoder_execution_request *request,
    const yvex_graph_attention_state_summary *attention,
    const yvex_sequence_state_summary *sequence, yvex_error *err)
{
    unsigned long long index, end;
    if (!context || !request || !request->token_ids || !request->token_count ||
        request->token_count > context->options.token_capacity ||
        !yvex_sha256_hex_valid(request->input_identity) ||
        !yvex_core_u64_add(request->token_start, request->token_count, &end) ||
        end > context->options.context_capacity ||
        !attention || attention->transaction_active ||
        attention->prepared_layer_count != attention->layer_count ||
        !attention->position_consistent ||
        attention->next_position != request->token_start ||
        attention->capacity < end ||
        !sequence || sequence->transaction_active ||
        sequence->committed_position != request->token_start)
        return decoder_refuse(
            err, YVEX_ERR_STATE, "runtime.decoder.request",
            "decoder input must extend the exact committed mixed-state position");
    for (index = 0ull; index < request->token_count; ++index)
        if ((unsigned long long)request->token_ids[index] >=
            context->summary->vocabulary_size)
            return decoder_refuse(err, YVEX_ERR_BOUNDS,
                                  "runtime.decoder.token",
                                  "decoder token exceeds the admitted vocabulary");
    return YVEX_OK;
}

static int decoder_embedding(decoder_layer_run *run, yvex_error *err)
{
    yvex_backend_operation_facts facts = {0};
    unsigned long long rows = run->request->token_count;
    unsigned long long width = run->context->summary->hidden_width;
    int rc;
    if (!decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_HIDDEN_A], rows * width,
            rows, width, &run->current) ||
        !decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_HIDDEN_B], rows * width,
            rows, width, &run->next) ||
        !decoder_tensor_view(
            run->context->buffers[DECODER_BUFFER_NORMALIZED], rows * width,
            rows, width, &run->normalized) ||
        !decoder_tensor_view(run->context->buffers[DECODER_BUFFER_UPDATE],
                             rows * width, rows, width, &run->update))
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "runtime.decoder.embedding",
                              "decoder hidden-state views exceed sealed buffers");
    rc = yvex_backend_encoded_gather(
        run->context->session_view->backend,
        run->context->embedding.encoded.encoded,
        run->context->embedding.encoded.encoded_bytes,
        run->context->embedding.encoded.qtype,
        run->context->embedding.encoded.row_count,
        run->context->embedding.encoded.row_width,
        run->context->embedding.encoded.row_bytes, run->request->token_ids,
        rows, &run->current, &facts, err);
    if (rc == YVEX_OK)
        rc = decoder_result_facts_add(run->result, &facts, err);
    return rc;
}

static int decoder_persistent_identity(
    const yvex_graph_attention_state_summary *attention,
    const yvex_sequence_state_summary *sequence,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!attention || !sequence || !output ||
        !yvex_sha256_hex_valid(attention->state_content_identity) ||
        !yvex_sha256_hex_valid(sequence->plan_identity) ||
        !yvex_sha256_update_text(
            &hash, "yvex.runtime.decoder.mixed-state.v1") ||
        !yvex_sha256_update_text(&hash, attention->state_content_identity) ||
        !yvex_sha256_update_text(&hash, sequence->plan_identity) ||
        !yvex_sha256_update_u64(&hash, attention->generation) ||
        !yvex_sha256_update_u64(&hash, sequence->generation) ||
        !yvex_sha256_update_u64(&hash, attention->next_position) ||
        !yvex_sha256_update_u64(&hash, sequence->committed_position) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int decoder_execution_identity(
    const yvex_runtime_decoder_execution_result *result,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!result || !output ||
        !yvex_sha256_hex_valid(result->decoder_plan_identity) ||
        !yvex_sha256_hex_valid(result->input_identity) ||
        !yvex_sha256_hex_valid(result->persistent_state_identity) ||
        !yvex_sha256_hex_valid(result->normalized_hidden_digest) ||
        !yvex_sha256_update_text(
            &hash, "yvex.runtime.decoder.execution.v1") ||
        !yvex_sha256_update_text(&hash, result->decoder_plan_identity) ||
        !yvex_sha256_update_text(&hash, result->input_identity) ||
        !yvex_sha256_update_text(&hash, result->persistent_state_identity) ||
        !yvex_sha256_update_text(&hash, result->normalized_hidden_digest) ||
        !yvex_sha256_update_u64(&hash, result->token_start) ||
        !yvex_sha256_update_u64(&hash, result->token_count) ||
        !yvex_sha256_update_u64(&hash, result->layers_executed) ||
        !yvex_sha256_update_u64(&hash, result->linear_operations) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int decoder_hidden_digest(
    const yvex_runtime_decoder_execution_result *result,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!result || !output ||
        !yvex_sha256_hex_valid(result->decoder_plan_identity) ||
        !yvex_sha256_hex_valid(result->persistent_state_identity) ||
        !yvex_sha256_update_text(
            &hash, "yvex.decoder.device-normalized-hidden.v1") ||
        !yvex_sha256_update_text(&hash, result->decoder_plan_identity) ||
        !yvex_sha256_update_text(&hash, result->persistent_state_identity) ||
        !yvex_sha256_update_u64(&hash, result->token_start) ||
        !yvex_sha256_update_u64(&hash, result->token_count) ||
        !yvex_sha256_update_u64(&hash, result->position_after) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int decoder_publish_result(
    decoder_layer_run *run,
    const yvex_graph_attention_state_summary *attention,
    const yvex_sequence_state_summary *sequence, yvex_error *err)
{
    yvex_runtime_decoder_execution_context *context = run->context;
    yvex_runtime_decoder_execution_result *result = run->result;
    unsigned long long rows = run->request->token_count;
    unsigned long long width = context->summary->hidden_width;
    int rc;
    context->hidden_publication = run->normalized;
    result->schema_version = YVEX_RUNTIME_DECODER_EXECUTION_SCHEMA_V1;
    result->token_start = run->request->token_start;
    result->token_count = rows;
    result->position_after = run->request->token_start + rows;
    result->convolution_state_bytes = sequence->convolution_state_bytes;
    result->recurrent_state_bytes = sequence->recurrent_state_bytes;
    yvex_runtime_identity_copy(result->decoder_plan_identity,
                               context->summary->decoder_plan_identity);
    yvex_runtime_identity_copy(result->input_identity,
                               run->request->input_identity);
    if (!decoder_persistent_identity(attention, sequence,
                                     result->persistent_state_identity))
        return decoder_refuse(err, YVEX_ERR_STATE,
                              "runtime.decoder.result",
                              "mixed sequence-state identity derivation failed");
    if (!decoder_hidden_digest(result, result->normalized_hidden_digest))
        return decoder_refuse(
            err, YVEX_ERR_STATE, "runtime.decoder.result",
            "decoder normalized-hidden digest derivation failed");
    rc = yvex_runtime_device_view_bind(
        &result->device_hidden, YVEX_EXECUTION_DEVICE_HIDDEN, context->model,
        context->session, context->session_view->attention_state_provider,
        context->options.execution_profile, &context->hidden_publication, 0ull,
        rows, width, err);
    if (rc == YVEX_OK &&
        !decoder_execution_identity(result, result->execution_identity))
        rc = decoder_refuse(err, YVEX_ERR_STATE, "runtime.decoder.result",
                            "decoder execution identity derivation failed");
    if (rc == YVEX_OK) result->completed = 1;
    return rc;
}

static int decoder_execute_locked(
    yvex_runtime_decoder_execution_context *context,
    const yvex_runtime_decoder_execution_request *request,
    yvex_runtime_decoder_execution_result *result, yvex_error *err)
{
    yvex_graph_attention_state_summary attention_before = {0};
    yvex_graph_attention_state_summary attention_after = {0};
    yvex_sequence_state_summary sequence_before = {0}, sequence_after = {0};
    yvex_model_engine_failure failure = {0};
    decoder_layer_run run = {0};
    unsigned long long layer, started;
    int session_owned = 0, rc;

    rc = decoder_state_summary(context, &attention_before, &sequence_before,
                               err);
    if (rc == YVEX_OK)
        rc = decoder_prepare_attention_state(
            context, request, &attention_before, &sequence_before, err);
    if (rc == YVEX_OK)
        rc = decoder_request_validate(context, request, &attention_before,
                                      &sequence_before, err);
    if (rc == YVEX_OK) {
        rc = yvex_runtime_session_begin(context->session, &failure, err);
        session_owned = rc == YVEX_OK;
    }
    if (rc == YVEX_OK)
        rc = yvex_sequence_state_begin(context->session_view->sequence_state,
                                       request->token_start,
                                       request->token_count, err);
    run.context = context;
    run.request = request;
    run.result = result;
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK) rc = decoder_embedding(&run, err);
    if (rc == YVEX_OK)
        result->embedding_nanoseconds = yvex_core_monotonic_ns() - started;
    started = yvex_core_monotonic_ns();
    for (layer = 0ull; rc == YVEX_OK && layer < context->summary->layer_count;
         ++layer) {
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context))
            rc = decoder_refuse(err, YVEX_ERR_CANCELLED,
                                "runtime.decoder.cancel",
                                "decoder execution cancelled between layers");
        if (rc == YVEX_OK)
            rc = decoder_layer_execute(
                &run, yvex_decoder_plan_layer_at(context->plan, layer),
                &context->layers[layer], err);
    }
    if (rc == YVEX_OK)
        result->layer_nanoseconds = yvex_core_monotonic_ns() - started;
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = decoder_normalize(&run, &run.current, &context->output_norm,
                               yvex_decoder_plan_layer_at(context->plan, 0ull)
                                   ->normalization_epsilon,
                               &run.normalized, request->token_count,
                               context->summary->hidden_width, err);
    if (rc == YVEX_OK)
        result->final_nanoseconds = yvex_core_monotonic_ns() - started;
    if (session_owned)
        rc = yvex_runtime_session_finish_coordinated(
            context->session, rc, NULL, 0u, err);
    if (rc == YVEX_OK)
        rc = decoder_state_summary(context, &attention_after, &sequence_after,
                                   err);
    if (rc == YVEX_OK &&
        (attention_after.next_position !=
             request->token_start + request->token_count ||
         sequence_after.committed_position != attention_after.next_position ||
         result->layers_executed != context->summary->layer_count ||
         result->attention_layers != context->summary->attention_layer_count ||
         result->recurrent_layers != context->summary->recurrent_layer_count))
        rc = decoder_refuse(err, YVEX_ERR_STATE, "runtime.decoder.commit",
                            "decoder mixed-state publication is incomplete");
    if (rc == YVEX_OK)
        rc = decoder_publish_result(&run, &attention_after, &sequence_after,
                                    err);
    return rc;
}

int yvex_runtime_decoder_execution_execute(
    yvex_runtime_decoder_execution_context *context,
    const yvex_runtime_decoder_execution_request *request,
    yvex_runtime_decoder_execution_result *result, yvex_error *err)
{
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !request || !result)
        return decoder_refuse(err, YVEX_ERR_INVALID_ARG,
                              "runtime.decoder.execute",
                              "decoder context, request, and result are required");
    rc = decoder_enter(context, err);
    if (rc != YVEX_OK) return rc;
    rc = decoder_execute_locked(context, request, result, err);
    decoder_leave(context);
    return rc;
}
