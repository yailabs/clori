/* Compose transformer owners under one transaction through final hidden-state publication. */
#include <yvex/internal/transformer.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/execution.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>
typedef struct {
    unsigned char *bytes;
    unsigned long long capacity;
    float *decoded;
    unsigned long long decoded_count;
} transformer_weight_owner;
static const yvex_tensor_role transformer_runtime_roles[YVEX_TRANSFORMER_WEIGHT_COUNT] = {
    YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION,
    YVEX_TENSOR_ROLE_HC_HEAD_BASE, YVEX_TENSOR_ROLE_HC_HEAD_SCALE,
    YVEX_TENSOR_ROLE_OUTPUT_NORM};
struct yvex_runtime_transformer_context {
    yvex_runtime_model *model;
    yvex_runtime_execution_session *session;
    const yvex_runtime_model_view *model_view;
    const yvex_runtime_session_view *session_view;
    yvex_runtime_transformer_options options;
    yvex_runtime_moe_context *moe;
    yvex_transformer_plan *plan;
    transformer_weight_owner global[YVEX_TRANSFORMER_WEIGHT_COUNT];
    unsigned char *embedding_encoded;
    unsigned long long embedding_row_bytes;
    yvex_device_tensor *device_embedding_encoded, *device_embedding;
    yvex_device_tensor *device_residual[2], *device_attention, *device_hidden;
    /* Max-shaped workspaces publish only the exact final rows completed by the kernel. */
    yvex_device_tensor device_hidden_publication, device_pre_normalized_publication;
    yvex_device_tensor *device_global[YVEX_TRANSFORMER_WEIGHT_COUNT];
    float *embedding, *expanded_a, *expanded_b, *candidate_hidden;
    float *moe_combined, *moe_post, *moe_combination, *moe_routed, *moe_shared;
    unsigned long long token_capacity, host_bytes, final_weight_bytes, execution_count, moe_workspace_bytes;
    char workspace_identity[YVEX_SHA256_HEX_CAP];
    pthread_mutex_t mutex;
    int mutex_ready, busy, invalidated;
};
static const yvex_attention_plan *transformer_runtime_attention(
    const yvex_runtime_model_view *view, yvex_tensor_scope scope)
{
    if (!view) return NULL;
    return scope == YVEX_TENSOR_SCOPE_DRAFT ? view->draft_attention : view->attention;
}
static int transformer_device_value_digest(const char *domain, const yvex_transformer_plan_summary *plan,
    unsigned long long layer, const char *routing, char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!domain || !plan || !routing || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_sha256_update_text(&hash, plan->transformer_plan_identity) ||
        !yvex_sha256_update_u64(&hash, layer) ||
        !yvex_sha256_update_text(&hash, routing) || !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
typedef struct {
    const float *host_input;
    const yvex_device_tensor *device_input;
    yvex_device_tensor *device_output;
    unsigned long long token_count, input_width, layer_ordinal;
    int fixed_input;
} transformer_activation_source;
typedef struct {
    yvex_runtime_transformer_context *owner;
    const yvex_runtime_transformer_request *request;
    yvex_runtime_transformer_output *output;
    const unsigned int *tokens;
    unsigned long long token_offset, token_count, token_start, layer_ordinal;
    unsigned long long feature_next;
    float *current, *next;
    yvex_device_tensor device_current, device_next, device_attention, device_hidden;
    yvex_sha256 layer_hash, routing_hash, embedding_hash;
    yvex_backend_kind backend;
    yvex_runtime_transformer_result *result;
    transformer_activation_source activation;
    char last_expanded_digest[YVEX_SHA256_HEX_CAP];
} transformer_chunk_context;
static int transformer_runtime_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.transformer", reason);
    return status;
}
int yvex_runtime_transformer_cuda_facts_add(yvex_runtime_transformer_result *result,
    const yvex_backend_cuda_operation_facts *facts, unsigned long long h2d_bytes,
    unsigned long long download_count, unsigned long long device_synchronizations, yvex_error *err)
{
    unsigned long long downloads, uploads, synchronizations;
    if (!result || !facts)
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG, "CUDA physical facts are required");
    if (yvex_execution_memory_facts_add(&result->memory, facts->active_weight_bytes,
            facts->state_bytes, facts->activation_bytes, facts->temporary_bytes,
            facts->compulsory_memory_facts_available,
            !facts->compulsory_memory_facts_available, err) != YVEX_OK)
        return yvex_error_code(err);
    if (!yvex_core_u64_add(facts->upload_count, h2d_bytes != 0ull, &uploads) ||
        !yvex_core_u64_add(facts->download_count, download_count, &downloads) ||
        !yvex_core_u64_add(facts->device_synchronizations, device_synchronizations, &synchronizations) ||
        !yvex_core_u64_add(result->h2d_bytes, h2d_bytes, &result->h2d_bytes) ||
        !yvex_core_u64_add(result->d2h_bytes, facts->d2h_bytes, &result->d2h_bytes) ||
        !yvex_core_u64_add(result->d2d_bytes, facts->d2d_bytes, &result->d2d_bytes) ||
        !yvex_core_u64_add(result->kernel_launches, facts->kernel_launches, &result->kernel_launches) ||
        !yvex_core_u64_add(result->upload_count, uploads, &result->upload_count) ||
        !yvex_core_u64_add(result->download_count, downloads, &result->download_count) ||
        !yvex_core_u64_add(result->stream_synchronizations, facts->stream_synchronizations,
                           &result->stream_synchronizations) ||
        !yvex_core_u64_add(result->device_synchronizations, synchronizations,
                           &result->device_synchronizations))
        return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS, "CUDA physical accounting overflowed");
    return YVEX_OK;
}
static int transformer_moe_complete(transformer_chunk_context *chunk, int barrier_observed,
                                    int primary_rc, yvex_error *err)
{
    yvex_moe_row_batch_result completion;
    yvex_runtime_transformer_result *result = chunk ? chunk->result : NULL;
    yvex_error primary = err ? *err : (yvex_error){0};
    int rc;
    if (!chunk || !result)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE, "transformer MoE completion owner is invalid");
    rc = yvex_runtime_moe_rows(
        chunk->owner->moe,
        &(yvex_moe_rows_request){YVEX_MOE_ROWS_COMPLETE, 0ull, NULL, NULL, barrier_observed},
        &completion, err);
    if (primary_rc != YVEX_OK) {
        if (err) *err = primary;
        return primary_rc;
    }
    if (rc != YVEX_OK) return rc;
    if (yvex_execution_memory_facts_merge(&result->memory, &completion.memory, err) != YVEX_OK)
        return yvex_error_code(err);
    if (!yvex_core_u64_add(result->unique_experts, completion.unique_experts,
                           &result->unique_experts) ||
        !yvex_core_u64_add(result->expert_weight_bytes, completion.encoded_bytes_read,
                           &result->expert_weight_bytes) ||
        !yvex_core_u64_add(result->stream_synchronizations, completion.stream_synchronizations,
                           &result->stream_synchronizations) ||
        !yvex_core_u64_add(result->device_synchronizations, completion.device_synchronizations,
                           &result->device_synchronizations) ||
        !yvex_core_u64_add(result->synchronization_ns, completion.synchronization_ns,
                           &result->synchronization_ns) ||
        !yvex_core_u64_add(result->moe_ns, completion.synchronization_ns, &result->moe_ns))
        return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS, "transformer MoE completion facts overflowed");
    return YVEX_OK;
}
static int transformer_feature_request_validate(
    const yvex_transformer_plan_summary *plan, const yvex_transformer_input_summary *input,
    const yvex_runtime_transformer_request *request, const yvex_runtime_transformer_output *output,
    yvex_attention_evidence_level evidence, unsigned long long *feature_elements, yvex_error *err)
{
    unsigned long long index, rows, hidden_elements;
    *feature_elements = 0ull;
    if (request->transaction_disposition > YVEX_ATTENTION_TRANSACTION_STAGE)
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG, "transformer transaction disposition is invalid");
    if (request->candidate_block_visible &&
        (plan->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT || input->token_count < 2ull))
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
            "candidate-block attention requires a multi-token draft request");
    if (request->retain_prefix_checkpoints &&
        (plan->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT ||
         request->transaction_disposition != YVEX_ATTENTION_TRANSACTION_STAGE ||
         input->token_count < 2ull))
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
            "prefix checkpoints require a staged multi-token target request");
    if (!yvex_core_u64_mul(input->token_count, plan->hidden_width, &hidden_elements) ||
        (output->pre_normalized_hidden &&
         output->pre_normalized_capacity < hidden_elements) ||
        (!output->pre_normalized_hidden && output->pre_normalized_capacity))
        return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                          "transformer pre-normalized output capacity is invalid");
    if (!request->feature_layer_count)
        return request->feature_layer_ordinals || output->features ||
                       output->feature_capacity || output->device_features ||
        output->device_feature_row_offset || output->device_feature_row_stride
        ? transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
              "transformer feature outputs require requested layers")
                   : YVEX_OK;
    if (!request->feature_layer_ordinals ||
        !yvex_core_u64_mul(input->token_count, request->feature_layer_count,
                           &rows) ||
        !yvex_core_u64_mul(rows, plan->hidden_width, feature_elements) ||
        (output->features && output->feature_capacity < *feature_elements) ||
        (!output->features &&
         (output->feature_capacity || request->backend != YVEX_BACKEND_KIND_CUDA ||
          !output->device_features || evidence == YVEX_ATTENTION_EVIDENCE_FULL)))
        return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                          "transformer feature output capacity is insufficient");
    if ((output->device_features &&
         (request->backend != YVEX_BACKEND_KIND_CUDA ||
          output->device_features->dtype != YVEX_DTYPE_F32 ||
          output->device_feature_row_stride !=
              *feature_elements / input->token_count)) ||
        (!output->device_features && (output->device_feature_row_offset ||
                                      output->device_feature_row_stride)))
        return transformer_runtime_refuse(err, YVEX_ERR_FORMAT,
                                          "transformer device feature output is incompatible");
    for (index = 0ull; index < request->feature_layer_count; ++index)
        if (request->feature_layer_ordinals[index] >= plan->layer_count ||
            (index && request->feature_layer_ordinals[index] <=
                          request->feature_layer_ordinals[index - 1ull]))
            return transformer_runtime_refuse(
                err, YVEX_ERR_FORMAT, "transformer feature layers must be unique and ordered");
    return YVEX_OK;
}
static const yvex_materialized_tensor_binding *transformer_runtime_binding(
    const yvex_runtime_transformer_context *context, yvex_transformer_weight_slot slot)
{
    const yvex_transformer_plan_summary *summary = context
        ? yvex_transformer_plan_summary_get(context->plan) : NULL;
    return summary && slot < YVEX_TRANSFORMER_WEIGHT_COUNT
               ? yvex_materialization_session_tensor_at(
                     context->model_view->materialization, summary->weights[slot].tensor_id)
               : NULL;
}
static int transformer_runtime_binding_project(
    const yvex_runtime_model_view *view, yvex_tensor_role role,
    yvex_tensor_scope scope, unsigned long long layer_index, unsigned long long predictor_index,
    yvex_transformer_weight_binding *out, yvex_error *err)
{
    const yvex_runtime_tensor_binding *runtime;
    const yvex_materialized_tensor_binding *binding;
    const yvex_quant_numeric_capability *capability;
    runtime = yvex_runtime_descriptor_find_role(
        view ? view->descriptor : NULL, role, scope, layer_index, predictor_index);
    binding = runtime ? yvex_materialization_session_tensor_at(
        view->materialization, runtime->tensor_id) : NULL;
    capability = binding ? yvex_quant_numeric_capability_at(binding->qtype) : NULL;
    if (!binding || binding->role != role || !binding->encoded_bytes ||
        binding->expert_count > 1ull || !binding->backend_compatible)
        return transformer_runtime_refuse(err, YVEX_ERR_FORMAT,
                                          "transformer global binding is unavailable");
    if (!capability || !capability->reference_decoder_available ||
        !capability->dedicated_cpu_compute_available ||
        !capability->dedicated_cuda_compute_available)
        return transformer_runtime_refuse(err, YVEX_ERR_UNSUPPORTED,
            "transformer global binding qtype lacks CPU/CUDA execution");
    *out = (yvex_transformer_weight_binding){
        .tensor_id = binding->tensor_id, .row_width = binding->row_width,
        .row_count = binding->row_count, .encoded_bytes = binding->encoded_bytes,
        .role = binding->role, .tensor_scope = scope,
        .layer_index = layer_index, .predictor_index = predictor_index,
        .qtype = binding->qtype};
    return YVEX_OK;
}
static int transformer_runtime_plan_facts(
    const yvex_runtime_model_view *view, yvex_tensor_scope scope,
    yvex_transformer_plan_facts *facts, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *runtime = view
        ? yvex_runtime_descriptor_summary_get(view->descriptor) : NULL;
    const yvex_materialization_summary *material = view
        ? yvex_materialization_session_summary(view->materialization) : NULL;
    const yvex_attention_plan *attention = transformer_runtime_attention(view, scope);
    const yvex_attention_summary *attention_summary = yvex_attention_plan_summary(attention);
    const yvex_attention_layer_plan *last;
    unsigned long long slot;
    memset(facts, 0, sizeof(*facts));
    if (!view || !runtime || !view->adapter || !view->adapter->transformer_policy ||
        !view->adapter->transformer_policy(runtime, &facts->policy) || !material ||
        !attention_summary || !attention_summary->layer_count)
        return transformer_runtime_refuse(err, YVEX_ERR_FORMAT,
                                          "transformer runtime plan facts are unavailable");
    facts->family_adapter_id = view->adapter->adapter_id;
    facts->family_adapter_version = view->adapter->adapter_version;
    /* Runtime GLOBAL selects the target execution lane; its per-layer
     * attention and MoE plans are bound to MAIN_LAYER tensors. Draft is both
     * an execution lane and a physical tensor scope. */
    facts->tensor_scope = scope == YVEX_TENSOR_SCOPE_GLOBAL
                              ? YVEX_TENSOR_SCOPE_MAIN_LAYER : YVEX_TENSOR_SCOPE_DRAFT;
    facts->layer_count = attention_summary->layer_count;
    facts->vocabulary_size = runtime->vocabulary_size;
    facts->artifact_identity = material->artifact_identity;
    facts->materialization_identity = material->plan_identity;
    facts->logical_model_identity = runtime->logical_model_identity;
    facts->runtime_numeric_identity = runtime->runtime_numeric_identity;
    facts->runtime_descriptor_identity = runtime->runtime_descriptor_identity;
    last = yvex_attention_plan_layer_at(attention, attention_summary->layer_count - 1ull);
    if (!last)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer final layer is unavailable");
    for (slot = 0ull; slot < YVEX_TRANSFORMER_WEIGHT_COUNT; ++slot) {
        yvex_tensor_role role = transformer_runtime_roles[slot];
        yvex_tensor_scope weight_scope = YVEX_TENSOR_SCOPE_GLOBAL;
        unsigned long long layer = YVEX_MATERIALIZATION_NO_INDEX;
        unsigned long long predictor = YVEX_MATERIALIZATION_NO_INDEX;
        if (scope == YVEX_TENSOR_SCOPE_DRAFT &&
            slot != YVEX_TRANSFORMER_WEIGHT_EMBEDDING) {
            weight_scope = YVEX_TENSOR_SCOPE_DRAFT;
            layer = last->layer_index;
            predictor = last->predictor_index;
            if (slot == YVEX_TRANSFORMER_WEIGHT_OUTPUT_NORM)
                role = YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM;
        }
        if (transformer_runtime_binding_project(
                view, role, weight_scope, layer, predictor,
                &facts->weights[slot], err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return YVEX_OK;
}
static int transformer_runtime_read(yvex_runtime_transformer_context *context,
                                    const yvex_materialized_tensor_binding *binding,
                                    unsigned long long offset, unsigned long long bytes,
                                    unsigned char *destination, yvex_error *err)
{
    yvex_materialization_failure failure;
    if (!binding || !destination || !bytes || offset > binding->encoded_bytes ||
        bytes > binding->encoded_bytes - offset || bytes > SIZE_MAX)
        return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                          "transformer weight range is invalid");
    memset(&failure, 0, sizeof(failure));
    return yvex_materialization_session_read(context->model_view->materialization, binding,
                                              offset, destination, (size_t)bytes, &failure, err);
}
static int transformer_runtime_decode(const yvex_materialized_tensor_binding *binding,
                                      const unsigned char *encoded, unsigned long long bytes,
                                      float *decoded, unsigned long long count, yvex_error *err)
{
    const yvex_gguf_qtype_geometry *geometry;
    yvex_quant_failure failure;
    unsigned long long block, blocks;
    if (!binding || !encoded || !decoded || !count || bytes > SIZE_MAX)
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                          "transformer decode arguments are invalid");
    geometry = yvex_gguf_qtype_geometry_find(binding->qtype);
    if (!geometry || !geometry->block_size || !geometry->bytes_per_block ||
        count % geometry->block_size ||
        (blocks = count / geometry->block_size) > SIZE_MAX / geometry->bytes_per_block ||
        blocks * geometry->bytes_per_block != bytes)
        return transformer_runtime_refuse(err, YVEX_ERR_FORMAT,
                                          "transformer qtype extent is not block-exact");
    for (block = 0ull; block < blocks; ++block) {
        memset(&failure, 0, sizeof(failure));
        if (yvex_quant_decode_block(
                binding->qtype, encoded + block * geometry->bytes_per_block,
                geometry->bytes_per_block, decoded + block * geometry->block_size,
                geometry->block_size, &failure, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return YVEX_OK;
}
static int transformer_runtime_globals(yvex_runtime_transformer_context *context, yvex_error *err)
{
    unsigned long long slot, total = sizeof(*context);
    for (slot = YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION;
         slot < YVEX_TRANSFORMER_WEIGHT_COUNT; ++slot) {
        const yvex_materialized_tensor_binding *binding = transformer_runtime_binding(
            context, (yvex_transformer_weight_slot)slot);
        transformer_weight_owner *owner = &context->global[slot];
        unsigned long long count, decoded_bytes;
        if (!binding || !yvex_core_u64_mul(binding->row_width, binding->row_count, &count) ||
            !yvex_core_u64_mul(count, sizeof(float), &decoded_bytes) ||
            !yvex_core_u64_add(total, binding->encoded_bytes, &total) ||
            !yvex_core_u64_add(total, decoded_bytes, &total) ||
            !yvex_core_u64_add(context->final_weight_bytes, decoded_bytes,
                               &context->final_weight_bytes) ||
            binding->encoded_bytes > SIZE_MAX || count > SIZE_MAX / sizeof(float))
            return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                              "transformer global weight extent overflowed");
        owner->capacity = binding->encoded_bytes;
        owner->decoded_count = count;
        owner->bytes = (unsigned char *)malloc((size_t)binding->encoded_bytes);
        owner->decoded = (float *)malloc((size_t)decoded_bytes);
        if (!owner->bytes || !owner->decoded)
            return transformer_runtime_refuse(err, YVEX_ERR_NOMEM,
                                              "transformer global weight allocation failed");
        if (transformer_runtime_read(context, binding, 0ull, binding->encoded_bytes,
                                     owner->bytes, err) != YVEX_OK ||
            transformer_runtime_decode(binding, owner->bytes, binding->encoded_bytes,
                                       owner->decoded, count, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    if (context->options.maximum_host_bytes && total > context->options.maximum_host_bytes)
        return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                          "transformer immutable globals exceed host budget");
    context->host_bytes = total;
    return YVEX_OK;
}
static int transformer_device_tensor_open(
    yvex_runtime_transformer_context *context, yvex_device_tensor **out,
    const char *name, yvex_dtype dtype, unsigned long long elements, unsigned long long bytes,
    yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    descriptor.name = name;
    descriptor.dtype = dtype;
    descriptor.rank = 1u;
    descriptor.dims[0] = elements;
    descriptor.bytes = bytes;
    return yvex_backend_tensor_alloc(context->session_view->backend, &descriptor, out, err);
}
/* Seal stable CUDA transformer resources; partial resources remain owned for deterministic close. */
static int transformer_device_buffers(yvex_runtime_transformer_context *context,
                                      unsigned long long hidden, unsigned long long expanded,
                                      yvex_error *err)
{
    const yvex_materialized_tensor_binding *embedding = transformer_runtime_binding(
        context, YVEX_TRANSFORMER_WEIGHT_EMBEDDING);
    struct {
        yvex_device_tensor **owner;
        const char *name;
        unsigned long long elements;
    } buffers[] = {
        {&context->device_embedding, "transformer-embedding", hidden},
        {&context->device_residual[0], "transformer-residual-a", expanded},
        {&context->device_residual[1], "transformer-residual-b", expanded},
        {&context->device_attention, "transformer-attention", expanded},
        {&context->device_hidden, "transformer-hidden", hidden}};
    unsigned long long encoded, index, slot;
    int rc;
    if (yvex_backend_kind_of(context->session_view->backend) != YVEX_BACKEND_KIND_CUDA)
        return YVEX_OK;
    if (!embedding || !embedding->row_count || embedding->encoded_bytes % embedding->row_count ||
        !(context->embedding_row_bytes = embedding->encoded_bytes / embedding->row_count) ||
        !yvex_core_u64_mul(context->token_capacity, context->embedding_row_bytes, &encoded) ||
        encoded > SIZE_MAX)
        return transformer_runtime_refuse(err, YVEX_ERR_FORMAT,
                                          "CUDA embedding row extent is malformed");
    context->embedding_encoded = (unsigned char *)malloc((size_t)encoded);
    if (!context->embedding_encoded)
        return transformer_runtime_refuse(err, YVEX_ERR_NOMEM,
                                          "CUDA embedding staging allocation failed");
    rc = transformer_device_tensor_open(context, &context->device_embedding_encoded,
        "transformer-embedding-encoded", YVEX_DTYPE_I8, encoded, encoded, err);
    for (index = 0ull; rc == YVEX_OK && index < sizeof(buffers) / sizeof(buffers[0]); ++index)
        rc = transformer_device_tensor_open(
            context, buffers[index].owner, buffers[index].name, YVEX_DTYPE_F32,
            buffers[index].elements, buffers[index].elements * sizeof(float), err);
    for (slot = YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION;
         rc == YVEX_OK && slot < YVEX_TRANSFORMER_WEIGHT_COUNT; ++slot) {
        transformer_weight_owner *owner = &context->global[slot];
        rc = transformer_device_tensor_open(context, &context->device_global[slot],
            "transformer-final-weight", YVEX_DTYPE_F32, owner->decoded_count,
            owner->decoded_count * sizeof(float), err);
        if (rc == YVEX_OK)
            rc = yvex_backend_tensor_write(context->session_view->backend,
                context->device_global[slot], owner->decoded,
                owner->decoded_count * sizeof(float), err);
    }
    return rc;
}
static int transformer_runtime_buffers(yvex_runtime_transformer_context *context,
                                       unsigned long long tokens, yvex_error *err)
{
    const yvex_transformer_plan_summary *s = yvex_transformer_plan_summary_get(context->plan);
    unsigned long long hidden, expanded, post, combination, total, bytes;
    if (context->token_capacity)
        return tokens <= context->token_capacity
                   ? YVEX_OK
                   : transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                                "transformer chunk exceeds sealed buffer capacity");
    if (!s || !tokens || !yvex_core_u64_mul(tokens, s->hidden_width, &hidden) ||
        !yvex_core_u64_mul(tokens, s->expanded_width, &expanded) ||
        !yvex_core_u64_mul(tokens, s->residual_streams, &post) ||
        !yvex_core_u64_mul(post, s->residual_streams, &combination) ||
        !yvex_core_u64_mul(hidden, 5ull, &total) ||
        !yvex_core_u64_add(total, expanded, &total) ||
        !yvex_core_u64_add(total, expanded, &total) ||
        !yvex_core_u64_add(total, post, &total) ||
        !yvex_core_u64_add(total, combination, &total) ||
        !yvex_core_u64_mul(total, sizeof(float), &bytes) ||
        context->host_bytes > ULLONG_MAX - bytes ||
        (context->options.maximum_host_bytes &&
         context->host_bytes + bytes > context->options.maximum_host_bytes))
        return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                          "transformer chunk buffers exceed their budget");
    context->embedding = (float *)calloc((size_t)hidden, sizeof(float));
    context->expanded_a = (float *)calloc((size_t)expanded, sizeof(float));
    context->expanded_b = (float *)calloc((size_t)expanded, sizeof(float));
    context->candidate_hidden = (float *)calloc((size_t)hidden, sizeof(float));
    context->moe_combined = (float *)calloc((size_t)hidden, sizeof(float));
    context->moe_routed = (float *)calloc((size_t)hidden, sizeof(float));
    context->moe_shared = (float *)calloc((size_t)hidden, sizeof(float));
    context->moe_post = (float *)calloc((size_t)post, sizeof(float));
    context->moe_combination = (float *)calloc((size_t)combination, sizeof(float));
    if (!context->embedding || !context->expanded_a || !context->expanded_b ||
        !context->candidate_hidden || !context->moe_combined || !context->moe_routed ||
        !context->moe_shared || !context->moe_post || !context->moe_combination)
        return transformer_runtime_refuse(err, YVEX_ERR_NOMEM,
                                          "transformer chunk buffer allocation failed");
    context->token_capacity = tokens;
    context->host_bytes += bytes;
    return transformer_device_buffers(context, hidden, expanded, err);
}
static int transformer_encoded_subview(const yvex_device_tensor *source,
                                       unsigned long long bytes, yvex_device_tensor *view)
{
    if (!source || !view || source->dtype != YVEX_DTYPE_I8 || !bytes || bytes > source->bytes)
        return 0;
    *view = *source;
    view->rank = 1u;
    view->dims[0] = bytes;
    view->bytes = bytes;
    return 1;
}
static int transformer_runtime_embedding(transformer_chunk_context *chunk, yvex_error *err)
{
    yvex_runtime_transformer_context *context = chunk->owner;
    const yvex_transformer_plan_summary *s = yvex_transformer_plan_summary_get(context->plan);
    const yvex_materialized_tensor_binding *binding = transformer_runtime_binding(
        context, YVEX_TRANSFORMER_WEIGHT_EMBEDDING);
    unsigned long long token;
    if (!binding || !binding->row_count || binding->encoded_bytes % binding->row_count ||
        !(context->embedding_row_bytes = binding->encoded_bytes / binding->row_count) ||
        context->embedding_row_bytes > SIZE_MAX)
        return transformer_runtime_refuse(err, YVEX_ERR_FORMAT,
                                          "token embedding rows are malformed");
    if (!context->embedding_encoded) {
        unsigned long long bytes;
        if (!yvex_core_u64_mul(context->token_capacity, context->embedding_row_bytes, &bytes) ||
            bytes > SIZE_MAX)
            return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                              "embedding staging extent overflowed");
        context->embedding_encoded = (unsigned char *)malloc((size_t)bytes);
        if (!context->embedding_encoded)
            return transformer_runtime_refuse(err, YVEX_ERR_NOMEM,
                                              "embedding row staging allocation failed");
    }
    for (token = 0ull; token < chunk->token_count; ++token) {
        unsigned long long id = chunk->tokens[chunk->token_offset + token];
        unsigned long long offset;
        unsigned char *encoded = context->embedding_encoded +
                                 token * context->embedding_row_bytes;
        if (id >= binding->row_count ||
            !yvex_core_u64_mul(id, context->embedding_row_bytes, &offset) ||
            transformer_runtime_read(context, binding, offset, context->embedding_row_bytes,
                                     encoded, err) != YVEX_OK ||
            transformer_runtime_decode(binding, encoded, context->embedding_row_bytes,
                                       context->embedding + token * s->hidden_width,
                                       s->hidden_width, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    chunk->result->embedding_rows += chunk->token_count;
    chunk->result->embedding_bytes += chunk->token_count * context->embedding_row_bytes;
    if (context->options.evidence_level == YVEX_ATTENTION_EVIDENCE_NONE) {
        for (token = 0ull; token < chunk->token_count; ++token)
            if (!yvex_sha256_update_u64(
                    &chunk->embedding_hash, chunk->tokens[chunk->token_offset + token]))
                return transformer_runtime_refuse(err, YVEX_ERR_STATE, "transformer embedding identity update failed");
    } else if (!yvex_execution_f32_hash_update(
                   &chunk->embedding_hash, context->embedding,
                   chunk->token_count * s->hidden_width))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer embedding digest update failed");
    if (chunk->backend == YVEX_BACKEND_KIND_CUDA) {
        unsigned long long bytes = chunk->token_count * context->embedding_row_bytes;
        yvex_backend_cuda_operation_facts facts;
        yvex_device_tensor encoded_view;
        int rc;
        if (!transformer_encoded_subview(context->device_embedding_encoded, bytes, &encoded_view))
            return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                "transformer CUDA embedding upload view is invalid");
        rc = yvex_backend_tensor_write(context->session_view->backend, &encoded_view,
                                       context->embedding_encoded, bytes, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_transformer_cuda_initial(
                context->session_view->backend, &encoded_view,
                binding->qtype, chunk->token_count, s->hidden_width,
                s->residual_streams, context->device_embedding,
                context->device_residual[0], &facts, err);
        if (rc != YVEX_OK) return rc;
        rc = yvex_runtime_transformer_cuda_facts_add(chunk->result, &facts, bytes, 0ull, 1ull, err);
        if (rc != YVEX_OK) return rc;
    }
    return yvex_transformer_initial_residual(context->plan, context->embedding,
                                             chunk->token_count, context->expanded_a, err);
}
static int transformer_activation_view(void *opaque, unsigned long long layer_ordinal,
                                       unsigned long long token_count, const float **input,
                                       unsigned long long *stride, yvex_error *err)
{
    transformer_activation_source *source = (transformer_activation_source *)opaque;
    if (!source || !source->host_input || !source->input_width || !input || !stride ||
        token_count != source->token_count ||
        (!source->fixed_input && layer_ordinal != source->layer_ordinal))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer attention activation order is invalid");
    *input = source->host_input;
    *stride = source->input_width;
    return YVEX_OK;
}
static int transformer_device_view(void *opaque, unsigned long long layer_ordinal,
    unsigned long long token_count, const yvex_device_tensor **input,
    yvex_device_tensor **output, yvex_error *err)
{
    transformer_activation_source *source = (transformer_activation_source *)opaque;
    if (!source || !source->device_input || !source->device_output || !input || !output ||
        !source->device_input->is_written || token_count != source->token_count ||
        (!source->fixed_input && layer_ordinal != source->layer_ordinal))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer CUDA activation order is invalid");
    *input = source->device_input;
    *output = source->device_output;
    return YVEX_OK;
}
static int transformer_feature_capture(transformer_chunk_context *chunk,
                                       unsigned long long completed_layer, yvex_error *err)
{
    const yvex_transformer_plan_summary *plan = yvex_transformer_plan_summary_get(chunk->owner->plan);
    unsigned long long feature_index, resident_row_offset, token, hidden, stream;
    float *destination;
    if (!chunk->request->feature_layer_count ||
        chunk->feature_next >= chunk->request->feature_layer_count ||
        chunk->request->feature_layer_ordinals[chunk->feature_next] != completed_layer)
        return YVEX_OK;
    feature_index = chunk->feature_next++;
    destination = chunk->output->features
        ? chunk->output->features +
              (chunk->token_offset * chunk->request->feature_layer_count +
               feature_index) * plan->hidden_width
        : NULL;
    if (chunk->backend == YVEX_BACKEND_KIND_CUDA &&
        chunk->owner->options.evidence_level != YVEX_ATTENTION_EVIDENCE_FULL) {
        yvex_backend_cuda_operation_facts facts = {0};
        if (!yvex_core_u64_add(chunk->output->device_feature_row_offset, chunk->token_offset,
                               &resident_row_offset))
            return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                               "transformer device feature row overflowed");
        int rc = yvex_backend_transformer_cuda_feature_mean(
            chunk->owner->session_view->backend, &chunk->device_current,
            chunk->token_count, plan->hidden_width, plan->residual_streams,
            &chunk->device_hidden, chunk->output->device_features,
            resident_row_offset, chunk->output->device_feature_row_stride,
            feature_index * plan->hidden_width,
            destination ? chunk->owner->candidate_hidden : NULL, &facts, err);
        if (rc != YVEX_OK) return rc;
        rc = yvex_runtime_transformer_cuda_facts_add(chunk->result, &facts, 0ull, 0ull, 0ull, err);
        if (rc != YVEX_OK) return rc;
        if (destination)
            for (token = 0ull; token < chunk->token_count; ++token)
                memcpy(destination + token * chunk->request->feature_layer_count * plan->hidden_width,
                       chunk->owner->candidate_hidden + token * plan->hidden_width,
                       (size_t)plan->hidden_width * sizeof(float));
        return YVEX_OK;
    }
    for (token = 0ull; token < chunk->token_count; ++token) {
        for (hidden = 0ull; hidden < plan->hidden_width; ++hidden) {
            double sum = 0.0;
            for (stream = 0ull; stream < plan->residual_streams; ++stream)
                sum += chunk->current[
                    token * plan->expanded_width +
                    stream * plan->hidden_width + hidden];
            destination[hidden] = (float)(sum / (double)plan->residual_streams);
        }
        destination += chunk->request->feature_layer_count * plan->hidden_width;
    }
    return YVEX_OK;
}
/* Complete one ordered block while the coordinator retains attention and KV commit authority. */
int yvex_runtime_transformer_execute_block(
    yvex_runtime_transformer_context *context, unsigned long long layer_ordinal,
    const unsigned int *token_ids, unsigned long long token_count,
    yvex_backend_kind backend, const yvex_attention_publication *attention,
    const yvex_device_tensor *device_attention, yvex_device_tensor *device_output,
    float *expanded_output, yvex_runtime_transformer_block_result *result, yvex_error *err)
{
    const yvex_transformer_plan_summary *s = context ? yvex_transformer_plan_summary_get(context->plan) : NULL;
    const yvex_moe_plan *moe_plan = context ? yvex_runtime_moe_context_plan(context->moe) : NULL;
    const yvex_moe_layer_plan *layer = moe_plan ? yvex_moe_plan_layer_at(moe_plan, layer_ordinal) : NULL;
    yvex_sha256 output_hash, identity_hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_moe_row_batch batch = {0};
    yvex_moe_row_batch_output batch_output = {0};
    yvex_moe_row_batch_result moe_result;
    unsigned long long output_elements, hidden_elements, post_elements, combination_elements;
    unsigned long long started_ns;
    int normal_cuda;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !s || !layer || !token_ids || !token_count || !attention ||
        !attention->complete || attention->layer_index != layer->layer_index ||
        attention->token_count != token_count ||
        attention->envelope_output_width != s->expanded_width ||
        !attention->envelope_output || !expanded_output || !result ||
        !yvex_core_u64_mul(token_count, s->expanded_width, &output_elements) ||
        !yvex_core_u64_mul(token_count, s->hidden_width, &hidden_elements) ||
        !yvex_core_u64_mul(token_count, s->residual_streams, &post_elements) ||
        !yvex_core_u64_mul(post_elements, s->residual_streams, &combination_elements) ||
        backend != yvex_backend_kind_of(context->session_view->backend) ||
        (backend == YVEX_BACKEND_KIND_CUDA && (!device_attention || !device_output)))
        return transformer_runtime_refuse(err, YVEX_ERR_FORMAT,
                                          "transformer attention publication is incompatible");
    started_ns = yvex_core_monotonic_ns();
    normal_cuda = backend == YVEX_BACKEND_KIND_CUDA &&
                  context->options.evidence_level != YVEX_ATTENTION_EVIDENCE_FULL;
    batch.schema_version = YVEX_MOE_ROW_BATCH_SCHEMA_V1;
    batch.row_count = token_count;
    batch.row_width = batch.row_stride = s->expanded_width;
    batch.expanded_rows = attention->envelope_output;
    batch.device_rows = backend == YVEX_BACKEND_KIND_CUDA ? device_attention : NULL;
    batch.device_outputs = backend == YVEX_BACKEND_KIND_CUDA ? device_output : NULL;
    batch.token_ids = token_ids;
    batch.token_ids_present = 1;
    batch.execution_class = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    if (normal_cuda && context->busy && context->options.execution_profile &&
        !context->options.execution_profile->token_local_moe_reference)
        batch.execution_class = YVEX_EXECUTION_CLASS_DEVICE_NATIVE;
    batch.execution_profile_identity = context->options.execution_profile
                                           ? context->options.execution_profile->identity : NULL;
    batch_output.combined_rows = context->moe_combined;
    batch_output.combined_capacity = hidden_elements;
    batch_output.routed_rows = context->moe_routed;
    batch_output.routed_capacity = hidden_elements;
    batch_output.shared_rows = context->moe_shared;
    batch_output.shared_capacity = hidden_elements;
    batch_output.post_rows = context->moe_post;
    batch_output.post_capacity = post_elements;
    batch_output.combination_rows = context->moe_combination;
    batch_output.combination_capacity = combination_elements;
    rc = yvex_runtime_moe_rows(
        context->moe,
        &(yvex_moe_rows_request){YVEX_MOE_ROWS_EXECUTE, layer_ordinal,
                                 &batch, &batch_output, 0},
        &moe_result, err);
    if (rc != YVEX_OK) return rc;
    result->hash_routers = token_count *
        (layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID);
    result->learned_routers = token_count *
        (layer->router_class == YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE);
    result->routed_experts = moe_result.row_expert_pairs;
    result->shared_experts = moe_result.shared_expert_operations;
    result->row_expert_pairs = moe_result.row_expert_pairs;
    result->unique_experts = moe_result.unique_experts;
    result->grouped_expert_operations = moe_result.grouped_expert_operations;
    result->expert_subviews_accessed = moe_result.expert_subviews_accessed;
    result->expert_weight_bytes = moe_result.encoded_bytes_read;
    result->memory = moe_result.memory;
    result->h2d_bytes = moe_result.h2d_bytes;
    result->d2h_bytes = moe_result.d2h_bytes;
    result->d2d_bytes = moe_result.d2d_bytes;
    result->kernel_launches = moe_result.kernel_launches;
    result->upload_count = moe_result.upload_count;
    result->download_count = moe_result.download_count;
    result->cache_hits = moe_result.cache_hits;
    result->cache_misses = moe_result.cache_misses;
    result->stream_synchronizations = moe_result.stream_synchronizations;
    result->device_synchronizations = moe_result.device_synchronizations;
    result->moe_ns = moe_result.total_ns;
    result->synchronization_ns = moe_result.synchronization_ns;
    yvex_runtime_identity_copy(result->routing_digest, moe_result.routing_digest);
    if (!normal_cuda) {
        rc = yvex_transformer_deferred_post(
            context->plan, attention->envelope_output, context->moe_combined,
            context->moe_post, context->moe_combination, token_count, expanded_output, err);
        if (rc != YVEX_OK) return rc;
        yvex_sha256_init(&output_hash);
        if (!yvex_sha256_update_text(&output_hash, "yvex.transformer.block.output.v1") ||
            !yvex_execution_f32_hash_update(&output_hash, expanded_output, output_elements) ||
            !yvex_sha256_final(&output_hash, digest))
            return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                              "transformer block output digest failed");
        yvex_sha256_hex(digest, result->expanded_digest);
    } else if (!transformer_device_value_digest(
                   "yvex.transformer.device-block.v1", s, layer_ordinal,
                   result->routing_digest, result->expanded_digest))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer device block identity failed");
    yvex_sha256_init(&identity_hash);
    if (!yvex_sha256_update_text(&identity_hash, "yvex.transformer.block.execution.v1") ||
        !yvex_sha256_update_text(&identity_hash, s->transformer_plan_identity) ||
        !yvex_sha256_update_u64(&identity_hash, layer_ordinal) ||
        !yvex_sha256_update_text(&identity_hash, result->routing_digest) ||
        !yvex_sha256_update_text(&identity_hash, result->expanded_digest) ||
        !yvex_sha256_update_u64(&identity_hash, backend) ||
        !yvex_sha256_update_u64(&identity_hash, context->options.evidence_level) ||
        !yvex_sha256_final(&identity_hash, digest))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer block execution identity failed");
    yvex_sha256_hex(digest, result->execution_identity);
    result->layer_ordinal = layer_ordinal;
    result->token_count = token_count;
    if (!result->moe_ns) result->moe_ns = yvex_core_monotonic_ns() - started_ns;
    result->completed = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Complete one ordered block; any evidence or final-stage refusal aborts the outer transaction. */
static int transformer_layer_evidence(void *opaque, yvex_backend_kind backend,
                                      const yvex_attention_publication *publication,
                                      yvex_error *err)
{
    transformer_chunk_context *chunk = (transformer_chunk_context *)opaque;
    yvex_runtime_transformer_context *context = chunk ? chunk->owner : NULL;
    const yvex_transformer_plan_summary *s = context
        ? yvex_transformer_plan_summary_get(context->plan) : NULL;
    yvex_runtime_transformer_block_result block;
    int rc;
    if (!chunk || !s)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer block context is unavailable");
    rc = yvex_runtime_transformer_execute_block(
        context, chunk->layer_ordinal, chunk->tokens + chunk->token_offset,
        chunk->token_count, backend, publication,
        backend == YVEX_BACKEND_KIND_CUDA ? &chunk->device_attention : NULL,
        backend == YVEX_BACKEND_KIND_CUDA ? &chunk->device_next : NULL,
        chunk->next, &block, err);
    if (rc != YVEX_OK) return rc;
    if (backend == YVEX_BACKEND_KIND_CUDA &&
        context->options.evidence_level != YVEX_ATTENTION_EVIDENCE_FULL) {
        if (!yvex_sha256_update_text(&chunk->layer_hash, block.expanded_digest))
            return transformer_runtime_refuse(err, YVEX_ERR_STATE, "transformer device layer identity update failed");
    } else if (!yvex_execution_f32_hash_update(
                   &chunk->layer_hash, chunk->next,
                   chunk->token_count * s->expanded_width))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE, "transformer layer digest update failed");
    yvex_runtime_identity_copy(chunk->last_expanded_digest, block.expanded_digest);
    if (!yvex_sha256_update_text(&chunk->routing_hash, block.routing_digest))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer routing digest update failed");
    chunk->result->hash_routers += block.hash_routers;
    chunk->result->learned_routers += block.learned_routers;
    chunk->result->routed_experts += block.routed_experts;
    chunk->result->shared_experts += block.shared_experts;
    chunk->result->row_expert_pairs += block.row_expert_pairs;
    chunk->result->unique_experts += block.unique_experts;
    chunk->result->grouped_expert_operations += block.grouped_expert_operations;
    chunk->result->expert_subviews_accessed += block.expert_subviews_accessed;
    chunk->result->expert_weight_bytes += block.expert_weight_bytes;
    if (yvex_execution_memory_facts_merge(
            &chunk->result->memory, &block.memory, err) != YVEX_OK)
        return yvex_error_code(err);
    chunk->result->h2d_bytes += block.h2d_bytes;
    chunk->result->d2h_bytes += block.d2h_bytes;
    chunk->result->d2d_bytes += block.d2d_bytes;
    chunk->result->upload_count += block.upload_count;
    chunk->result->download_count += block.download_count;
    chunk->result->cache_hits += block.cache_hits;
    chunk->result->cache_misses += block.cache_misses;
    chunk->result->stream_synchronizations += block.stream_synchronizations;
    chunk->result->device_synchronizations += block.device_synchronizations;
    chunk->result->kernel_launches += block.kernel_launches;
    chunk->result->moe_ns += block.moe_ns;
    chunk->result->synchronization_ns += block.synchronization_ns;
    chunk->result->layers_executed++;
    chunk->layer_ordinal++;
    {
        float *swap = chunk->current;
        yvex_device_tensor device_swap = chunk->device_current;
        chunk->current = chunk->next;
        chunk->next = swap;
        chunk->device_current = chunk->device_next;
        chunk->device_next = device_swap;
    }
    chunk->activation.host_input = chunk->current;
    chunk->activation.device_input = &chunk->device_current;
    chunk->activation.layer_ordinal = chunk->layer_ordinal;
    rc = transformer_feature_capture(chunk, chunk->layer_ordinal - 1ull, err);
    if (rc != YVEX_OK) return rc;
    if (chunk->layer_ordinal == s->layer_count)
        chunk->result->final_weight_bytes += context->final_weight_bytes;
    if (chunk->layer_ordinal == s->layer_count && backend == YVEX_BACKEND_KIND_CUDA) {
        unsigned long long expanded_bytes = chunk->token_count * s->expanded_width * sizeof(float);
        unsigned long long hidden_bytes = chunk->token_count * s->hidden_width * sizeof(float);
        unsigned long long started_ns = yvex_core_monotonic_ns();
        unsigned long long read_count = 0ull;
        yvex_backend_cuda_operation_facts facts = {0};
        yvex_device_tensor device_pre_normalized = {0};
        int full = context->options.evidence_level == YVEX_ATTENTION_EVIDENCE_FULL;
        int device_pre = context->options.device_pre_normalized_output;
        int reference_pre = chunk->output->pre_normalized_hidden && full && !device_pre;
        if (reference_pre) {
            rc = yvex_backend_tensor_read(context->session_view->backend, &chunk->device_current,
                                          chunk->current, expanded_bytes, err);
            if (rc == YVEX_OK)
                rc = yvex_transformer_final_stage_capture(
                    context->plan, chunk->current, chunk->token_count,
                    context->global[YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION].decoded,
                    context->global[YVEX_TRANSFORMER_WEIGHT_FINAL_BASE].decoded,
                    context->global[YVEX_TRANSFORMER_WEIGHT_FINAL_SCALE].decoded,
                    context->global[YVEX_TRANSFORMER_WEIGHT_OUTPUT_NORM].decoded,
                    chunk->output->pre_normalized_hidden +
                        chunk->token_offset * s->hidden_width,
                    context->candidate_hidden, err);
        } else {
            /* Final attention storage is dead here; reuse it for the optional pre-normalized row. */
            if ((chunk->output->pre_normalized_hidden || device_pre) &&
                !yvex_backend_tensor_f32_subview(&chunk->device_attention, 0ull,
                    chunk->token_count * s->hidden_width, &device_pre_normalized))
                rc = transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                    "transformer final pre-normalized view is invalid");
            if (rc == YVEX_OK)
                rc = yvex_backend_transformer_cuda_final(
                    context->session_view->backend, &chunk->device_current,
                    context->device_global[YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION],
                    context->device_global[YVEX_TRANSFORMER_WEIGHT_FINAL_BASE],
                    context->device_global[YVEX_TRANSFORMER_WEIGHT_FINAL_SCALE],
                    context->device_global[YVEX_TRANSFORMER_WEIGHT_OUTPUT_NORM],
                    chunk->token_count, s->hidden_width, s->residual_streams,
                    s->output_norm_epsilon, s->mhc_epsilon,
                    chunk->output->pre_normalized_hidden || device_pre
                        ? &device_pre_normalized : NULL,
                    &chunk->device_hidden, &facts, err);
            if (rc == YVEX_OK) {
                context->device_hidden_publication = chunk->device_hidden;
                if (chunk->output->pre_normalized_hidden || device_pre)
                    context->device_pre_normalized_publication =
                        device_pre_normalized;
            }
            if (rc == YVEX_OK && full)
                rc = yvex_backend_tensor_read(context->session_view->backend,
                    &chunk->device_current, chunk->current, expanded_bytes, err);
            if (rc == YVEX_OK && (chunk->output->normalized_hidden || full))
                rc = yvex_backend_tensor_read(context->session_view->backend,
                    &chunk->device_hidden, context->candidate_hidden, hidden_bytes, err);
            if (rc == YVEX_OK && chunk->output->pre_normalized_hidden)
                rc = yvex_backend_tensor_read(context->session_view->backend,
                    &device_pre_normalized, chunk->output->pre_normalized_hidden +
                    chunk->token_offset * s->hidden_width, hidden_bytes, err);
        }
        read_count = reference_pre ? 1ull
            : (unsigned long long)full +
                  (unsigned long long)(chunk->output->normalized_hidden != NULL || full) +
                  (unsigned long long)(chunk->output->pre_normalized_hidden != NULL);
        if (rc == YVEX_OK)
            rc = yvex_runtime_transformer_cuda_facts_add(
                chunk->result, &facts, 0ull, read_count, read_count, err);
        if (rc == YVEX_OK) {
            chunk->result->d2h_bytes += reference_pre ? expanded_bytes
                : (unsigned long long)full * expanded_bytes +
                      (unsigned long long)(chunk->output->normalized_hidden != NULL || full) *
                          hidden_bytes +
                      (unsigned long long)(chunk->output->pre_normalized_hidden != NULL) *
                          hidden_bytes;
            chunk->result->final_ns += yvex_core_monotonic_ns() - started_ns;
        }
        rc = transformer_moe_complete(
            chunk, rc == YVEX_OK &&
                       (read_count || facts.stream_synchronizations ||
                        facts.device_synchronizations), rc, err);
        return rc;
    }
    if (chunk->layer_ordinal == s->layer_count)
        return yvex_transformer_final_stage_capture(
            context->plan, chunk->current, chunk->token_count,
            context->global[YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION].decoded,
            context->global[YVEX_TRANSFORMER_WEIGHT_FINAL_BASE].decoded,
            context->global[YVEX_TRANSFORMER_WEIGHT_FINAL_SCALE].decoded,
            context->global[YVEX_TRANSFORMER_WEIGHT_OUTPUT_NORM].decoded,
            chunk->output->pre_normalized_hidden
                ? chunk->output->pre_normalized_hidden +
                      chunk->token_offset * s->hidden_width
                : NULL,
            context->candidate_hidden, err);
    return YVEX_OK;
}
static int transformer_state_summary(const yvex_runtime_transformer_context *context,
                                     yvex_graph_attention_state_summary *summary,
                                     yvex_error *err)
{
    const yvex_attention_state_provider *provider = context && context->session_view
        ? (context->options.tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
               ? context->session_view->draft_attention_state_provider
               : context->session_view->attention_state_provider)
        : NULL;
    if (!provider || !provider->summary ||
        provider->summary(provider->context, summary, err) != YVEX_OK)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer persistent state is unavailable");
    return YVEX_OK;
}
static int transformer_capacity_build(yvex_graph_attention_capacity_plan **out,
                                      const yvex_runtime_model_view *model,
                                      yvex_tensor_scope scope,
                                      unsigned long long start, unsigned long long tokens,
                                      yvex_error *err)
{
    yvex_graph_attention_capacity_request request;
    const yvex_graph_family_api *graph = model && model->adapter && model->adapter->graph
        ? model->adapter->graph() : NULL;
    memset(&request, 0, sizeof(request));
    request.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
    request.history_tokens = request.start_position = start;
    request.token_count = tokens;
    request.execution_count = 1ull;
    request.use_requested_position = 1;
    return yvex_graph_attention_capacity_plan_build(out, graph,
                                                     transformer_runtime_attention(model, scope),
                                                     &request, err);
}
static yvex_execution_context_band transformer_context_band(
    unsigned long long position, unsigned long long capacity)
{
    if (position < capacity / 8ull) return YVEX_EXECUTION_CONTEXT_SHORT;
    if (position < capacity / 2ull) return YVEX_EXECUTION_CONTEXT_MEDIUM;
    if (position < capacity - capacity / 8ull)
        return YVEX_EXECUTION_CONTEXT_LONG;
    return YVEX_EXECUTION_CONTEXT_NEAR_CAPACITY;
}
static unsigned long long transformer_required_capacity(
    const yvex_graph_attention_state_component_summary *component, unsigned long long width)
{
    unsigned long long required;
    if (!component->capacity || component->entry_count >= component->capacity)
        return component->capacity;
    required = component->entry_count + width;
    return required < component->capacity ? required : component->capacity;
}
static int transformer_shape_admit(
    yvex_runtime_transformer_context *context, const yvex_transformer_input_summary *input,
    const yvex_runtime_transformer_request *request, const yvex_graph_attention_state_summary *state,
    const yvex_runtime_session_summary *session, yvex_error *err)
{
    static const char *const capacity_names[] = {
        "unregistered", "local", "compressed", "indexer",
        "rolling", "candidate", "context", "workspace"};
    const yvex_transformer_plan_summary *plan =
        yvex_transformer_plan_summary_get(context->plan);
    const yvex_execution_shape *selected = NULL;
    yvex_execution_shape admitted = {0}, required;
    yvex_execution_shape_failure failure = {0};
    unsigned long long width = input->token_count < request->chunk_tokens
                                   ? input->token_count : request->chunk_tokens;
    int rc;
    if (!context->options.execution_profile && !context->options.shape_registry)
        return YVEX_OK;
    if (!context->options.execution_profile || !context->options.shape_registry ||
        !plan || !yvex_sha256_hex_valid(session->workspace_identity) ||
        !yvex_sha256_hex_valid(state->state_layout_identity))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
            "compiled execution profile and shape registry must remain paired");
    admitted.schema_version = YVEX_EXECUTION_SHAPE_SCHEMA_V1;
    admitted.target_scope = context->options.tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                                ? YVEX_EXECUTION_SCOPE_DRAFT : YVEX_EXECUTION_SCOPE_TARGET;
    admitted.phase = admitted.target_scope == YVEX_EXECUTION_SCOPE_DRAFT
                         ? YVEX_EXECUTION_PHASE_DRAFT
                         : request->retain_prefix_checkpoints
                               ? YVEX_EXECUTION_PHASE_VERIFY
                               : request->phase == YVEX_TRANSFORMER_PHASE_DECODE
                                     ? YVEX_EXECUTION_PHASE_DECODE : YVEX_EXECUTION_PHASE_PREFILL;
    admitted.operation_scope = YVEX_EXECUTION_OPERATION_ENVELOPE;
    admitted.token_width = width;
    admitted.candidate_visible = request->candidate_block_visible;
    admitted.context_band = transformer_context_band(
        input->token_start, context->options.context_capacity);
    admitted.context_capacity = context->options.context_capacity;
    admitted.local_capacity =
        state->components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].capacity;
    admitted.compressed_capacity =
        state->components[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY].capacity;
    admitted.indexer_capacity =
        state->components[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY].capacity;
    admitted.rolling_capacity =
        state->components[YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING].capacity;
    if (state->components[YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING].capacity >
        admitted.rolling_capacity)
        admitted.rolling_capacity =
            state->components[YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING].capacity;
    admitted.candidate_capacity = context->options.workspace_token_capacity;
    admitted.workspace_generation = session->workspace_generation;
    admitted.evidence = context->options.execution_profile->evidence;
    yvex_runtime_identity_copy(
        admitted.execution_profile_identity, context->options.execution_profile->identity);
    yvex_runtime_identity_copy(admitted.attention_plan_identity, plan->attention_plan_identity);
    yvex_runtime_identity_copy(admitted.state_layout_identity, state->state_layout_identity);
    yvex_runtime_identity_copy(admitted.kernel_bundle_identity,
                               context->options.execution_profile->kernel_bundle_identity);
    yvex_runtime_identity_copy(admitted.workspace_identity, session->workspace_identity);
    rc = yvex_execution_shape_seal(&admitted, err);
    if (rc == YVEX_OK)
        rc = yvex_execution_shape_registry_register(
            context->options.shape_registry, &admitted, err);
    required = admitted;
    required.position = input->token_start;
    required.local_capacity = transformer_required_capacity(
        &state->components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY], width);
    required.compressed_capacity = transformer_required_capacity(
        &state->components[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY], width);
    required.indexer_capacity = transformer_required_capacity(
        &state->components[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY], width);
    required.candidate_capacity = width;
    if (rc == YVEX_OK) rc = yvex_execution_shape_seal(&required, err);
    if (rc == YVEX_OK)
        rc = yvex_execution_shape_registry_select(
            context->options.shape_registry, &required, &selected, &failure, err);
    if (rc != YVEX_OK) {
        const char *component = failure.component <
                                        sizeof(capacity_names) / sizeof(capacity_names[0])
                                    ? capacity_names[failure.component]
                                    : "unknown";
        yvex_error_setf(err, (yvex_status)rc, "runtime.transformer.shape",
            "%s capacity configured=%llu required=%llu scope=%u phase=%u width=%llu "
            "position=%llu shape=%.16s workspace=%.16s state=%.16s",
            component, failure.configured, failure.required,
            (unsigned int)required.target_scope, (unsigned int)required.phase,
            required.token_width, required.position, failure.shape_identity,
            required.workspace_identity, required.state_layout_identity);
    }
    if (rc == YVEX_OK && request->backend == YVEX_BACKEND_KIND_CUDA &&
        !context->options.execution_profile->eager_attention_reference) {
        rc = yvex_backend_cuda_attention_configure(
            context->session_view->backend, (yvex_backend_attention_phase)selected->phase,
            YVEX_BACKEND_CUDA_ATTENTION_FULL, context->options.execution_profile->identity, "generation",
            selected->local_capacity, selected->compressed_capacity,
            selected->indexer_capacity, err);
    }
    return rc;
}
static int transformer_prepare(yvex_runtime_transformer_context *context,
    const yvex_transformer_input_summary *input, const yvex_runtime_transformer_request *request,
    yvex_graph_attention_state_summary *state, yvex_error *err)
{
    yvex_graph_attention_capacity_plan *capacity = NULL;
    yvex_runtime_model_failure failure;
    yvex_runtime_session_summary session;
    yvex_attention_failure attention_failure;
    yvex_runtime_execution_mode mode;
    unsigned long long final, workspace_tokens, workspace_bytes;
    int rc;
    if (request->phase != YVEX_TRANSFORMER_PHASE_PREFILL &&
        request->phase != YVEX_TRANSFORMER_PHASE_DECODE)
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                          "transformer execution phase is invalid");
    if (request->phase == YVEX_TRANSFORMER_PHASE_DECODE &&
        (input->token_count != 1ull || request->chunk_tokens != 1ull ||
         !input->token_start))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
            "decode phase requires one token over a nonzero committed prefix");
    if (!yvex_core_u64_add(input->token_start, input->token_count, &final) ||
        final > context->options.context_capacity)
        return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                          "transformer request exceeds context capacity");
    if (!state->prepared_layer_count) {
        if (input->token_start)
            return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                              "nonzero transformer start requires committed state");
        rc = transformer_capacity_build(&capacity, context->model_view,
                                        context->options.tensor_scope, 0ull,
                                        context->options.context_capacity, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_session_prepare_attention_scope_state(
                context->session, context->model, context->options.tensor_scope,
                capacity, &attention_failure, err);
        yvex_graph_attention_capacity_plan_close(&capacity);
        if (rc != YVEX_OK || transformer_state_summary(context, state, err) != YVEX_OK)
            return rc != YVEX_OK ? rc : yvex_error_code(err);
    }
    if (state->prepared_layer_count != state->layer_count || !state->position_consistent ||
        (state->extension_ready ? state->staged_next_position : state->next_position) !=
            input->token_start ||
        state->capacity < final)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer state position/capacity is incompatible");
    if (request->backend == YVEX_BACKEND_KIND_CPU) return YVEX_OK;
    mode = context->options.execution_profile &&
                   !context->options.execution_profile->eager_attention_reference
               ? YVEX_RUNTIME_MODE_FULL : YVEX_RUNTIME_MODE_EAGER;
    workspace_bytes = context->moe_workspace_bytes;
    if (workspace_bytes < context->options.minimum_device_workspace_bytes)
        workspace_bytes = context->options.minimum_device_workspace_bytes;
    rc = yvex_runtime_session_summary_copy(context->session, &session, err);
    if (rc != YVEX_OK) return rc;
    /* Replanning while correction extends verification would discard the open candidate;
     * reuse is therefore confined to the already wider admitted workspace. */
    if (state->extension_ready) {
        if (!state->transaction_active || !state->prefix_selected ||
            context->options.tensor_scope != YVEX_TENSOR_SCOPE_GLOBAL ||
            input->token_count != 1ull || request->chunk_tokens != 1ull ||
            request->retain_prefix_checkpoints || !session.busy ||
            !session.host_workspace_owned || !session.host_workspace_pinned ||
            session.device_workspace_bytes < workspace_bytes ||
            !yvex_sha256_hex_valid(session.workspace_identity))
            return transformer_runtime_refuse(
                err, YVEX_ERR_STATE,
                "target prefix extension cannot reuse the admitted CUDA workspace");
    } else {
        workspace_tokens = request->chunk_tokens < input->token_count ? request->chunk_tokens : input->token_count;
        rc = transformer_capacity_build(&capacity, context->model_view,
                                        context->options.tensor_scope,
                                        context->options.context_capacity - workspace_tokens,
                                        workspace_tokens, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_session_prepare_attention_workspace(
                context->session, mode, YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE,
                YVEX_ATTENTION_EVIDENCE_NONE, capacity, workspace_bytes, &failure, err);
        yvex_graph_attention_capacity_plan_close(&capacity);
        if (rc == YVEX_OK)
            rc = yvex_runtime_session_summary_copy(context->session, &session, err);
        if (rc == YVEX_OK && (!session.host_workspace_owned || !session.host_workspace_pinned ||
                              !yvex_sha256_hex_valid(session.workspace_identity)))
            rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                "transformer CUDA workspace did not publish stable ownership");
    }
    if (rc != YVEX_OK) return rc;
    yvex_runtime_identity_copy(context->workspace_identity, session.workspace_identity);
    return transformer_shape_admit(context, input, request, state, &session, err);
}
static int transformer_core_features_execute(
    yvex_runtime_transformer_context *context, const unsigned int *token_ids, unsigned long long token_start,
    const float *features, const yvex_device_tensor *device_features,
    const char *feature_identity, unsigned long long token_count,
    yvex_attention_transaction_disposition disposition,
    yvex_runtime_transformer_core_commit_result *result, yvex_error *err)
{
    const yvex_transformer_plan_summary *plan = context ? yvex_transformer_plan_summary_get(context->plan) : NULL;
    yvex_transformer_input_summary input = {0};
    yvex_runtime_transformer_request request = {0};
    yvex_graph_attention_state_summary before = {0}, after = {0};
    yvex_attention_execution_request execution = {0};
    yvex_attention_probe_result probe = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_execution_device_view device_view = {0};
    transformer_activation_source activation = {0};
    yvex_device_tensor device_input = {0}, device_output = {0};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long value_count;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !token_ids || !plan || plan->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT ||
        (disposition != YVEX_ATTENTION_TRANSACTION_COMMIT &&
         disposition != YVEX_ATTENTION_TRANSACTION_STAGE) ||
        (!features && !device_features) || !token_count || !result ||
        !yvex_core_u64_mul(token_count, plan->hidden_width, &value_count) ||
        !yvex_sha256_hex_valid(feature_identity))
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                          "draft core-feature commit geometry is invalid");
    yvex_runtime_identity_copy(result->input_digest, feature_identity);
    if (device_features && (!device_features->is_written || !context->options.execution_profile))
        return transformer_runtime_refuse(err, YVEX_ERR_FORMAT,
                                          "draft core-feature device view is incompatible");
    if (pthread_mutex_lock(&context->mutex) != 0)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE, "transformer context lock failed");
    if (context->busy || context->invalidated) {
        (void)pthread_mutex_unlock(&context->mutex);
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer context is busy or invalidated");
    }
    context->busy = 1;
    (void)pthread_mutex_unlock(&context->mutex);
    rc = transformer_state_summary(context, &before, err);
    input = (yvex_transformer_input_summary){
        .token_start = token_start, .token_count = token_count};
    request = (yvex_runtime_transformer_request){
        .backend = yvex_backend_kind_of(context->session_view->backend),
        .phase = YVEX_TRANSFORMER_PHASE_PREFILL, .chunk_tokens = token_count,
        .transaction_disposition = disposition};
    if (rc == YVEX_OK) rc = transformer_prepare(context, &input, &request, &before, err);
    if (rc == YVEX_OK && device_features)
        rc = yvex_runtime_device_view_bind(
            &device_view, YVEX_EXECUTION_DEVICE_FEATURE_TAP, context->model,
            context->session, context->session_view->draft_attention_state_provider,
            context->options.execution_profile, device_features, 0ull,
            token_count, plan->hidden_width, err);
    activation.host_input = features;
    activation.token_count = token_count;
    activation.input_width = plan->hidden_width;
    activation.fixed_input = 1;
    if (rc == YVEX_OK && device_features &&
        (!yvex_backend_tensor_f32_subview(device_view.tensor,
                                          device_view.element_offset,
                                          value_count, &device_input) ||
         !yvex_backend_tensor_f32_subview(context->device_attention, 0ull,
                                          value_count, &device_output)))
        rc = transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                        "draft core-feature device activation extent is invalid");
    activation.device_input = device_features ? &device_input : NULL;
    activation.device_output = device_features ? &device_output : NULL;
    execution = (yvex_attention_execution_request){
        .backend = request.backend, .tensor_scope = YVEX_TENSOR_SCOPE_DRAFT,
        .execution_phase = YVEX_EXECUTION_PHASE_DRAFT,
        .probe = YVEX_ATTENTION_PROBE_UNSPECIFIED, .scope = YVEX_ATTENTION_PROBE_SCOPE_FULL,
        .operation_scope = YVEX_ATTENTION_OPERATION_CORE, .token_count = token_count,
        .token_position = token_start, .select_position = 1,
        .input_identity = result->input_digest, .token_ids = token_ids,
        .activation_view = features ? transformer_activation_view : NULL,
        .device_view = device_features ? transformer_device_view : NULL,
        .activation_context = &activation, .cancel_requested = context->options.cancel_requested,
        .cancel_context = context->options.cancel_context,
        .evidence_level = context->options.evidence_level, .transaction_disposition = disposition};
    if (rc == YVEX_OK)
        rc = yvex_runtime_attention_probe_execute(context->session, context->model,
                                                  &execution, &probe, &failure, err);
    if (rc == YVEX_OK) rc = transformer_state_summary(context, &after, err);
    if (rc == YVEX_OK &&
        yvex_execution_physical_facts_add(
             &result->physical, &probe.memory, probe.h2d_bytes, probe.d2h_bytes,
             probe.d2d_bytes, probe.kernel_launches, probe.stream_synchronizations,
             probe.device_synchronizations, err) != YVEX_OK)
        rc = yvex_error_is_set(err) ? yvex_error_code(err)
                                    : transformer_runtime_refuse(
                                          err, YVEX_ERR_BOUNDS,
                                          "draft core-feature physical accounting overflowed");
    if (rc == YVEX_OK) result->device_input_consumed = device_features != NULL;
    if (rc == YVEX_OK && disposition == YVEX_ATTENTION_TRANSACTION_COMMIT &&
        (after.next_position != token_start + token_count ||
         after.generation != before.generation + 1ull))
        rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                        "draft core-feature transaction did not commit exact state");
    if (rc == YVEX_OK && disposition == YVEX_ATTENTION_TRANSACTION_STAGE &&
        (after.next_position != before.next_position ||
         after.generation != before.generation || !after.transaction_active ||
         !after.staged_layer_count || !after.staged_batch_complete ||
         after.staged_next_position != token_start + token_count ||
         after.staged_generation != before.generation + 1ull))
        rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                        "draft core-feature transaction was not staged privately");
    if (rc == YVEX_OK) {
        result->token_start = token_start;
        result->token_count = token_count;
        result->position_before = before.next_position;
        result->position_after = disposition == YVEX_ATTENTION_TRANSACTION_STAGE
                                     ? after.staged_next_position
                                     : after.next_position;
        result->generation_before = before.generation;
        result->generation_after = disposition == YVEX_ATTENTION_TRANSACTION_STAGE
                                       ? after.staged_generation
                                       : after.generation;
        yvex_runtime_identity_copy(
            result->persistent_state_digest,
            disposition == YVEX_ATTENTION_TRANSACTION_STAGE
                ? after.staged_state_content_identity
                : after.state_content_identity);
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update_text(&hash, "yvex.runtime.transformer.core-feature-commit.v2") ||
            !yvex_sha256_update_text(&hash, plan->transformer_plan_identity) ||
            !yvex_sha256_update_u64(&hash, disposition) ||
            !yvex_sha256_update_text(&hash, result->input_digest) ||
            !yvex_sha256_update_text(&hash, result->persistent_state_digest) ||
            !yvex_sha256_update_u64(&hash, result->position_before) ||
            !yvex_sha256_update_u64(&hash, result->position_after) ||
            !yvex_sha256_final(&hash, digest))
            rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                            "draft core-feature commit identity derivation failed");
        else {
            yvex_sha256_hex(digest, result->execution_identity);
            result->completed = 1;
        }
    }
    if (pthread_mutex_lock(&context->mutex) == 0) {
        context->busy = 0;
        if (rc == YVEX_OK) context->execution_count++;
        (void)pthread_mutex_unlock(&context->mutex);
    }
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
int yvex_runtime_transformer_stage_core_features(
    yvex_runtime_transformer_context *context, const unsigned int *token_ids, unsigned long long token_start,
    const float *features, const yvex_device_tensor *device_features,
    const char *feature_identity, unsigned long long token_count,
    yvex_runtime_transformer_core_commit_result *result, yvex_error *err)
{
    return transformer_core_features_execute(
        context, token_ids, token_start, features, device_features, feature_identity, token_count,
        YVEX_ATTENTION_TRANSACTION_STAGE, result, err);
}
/* Allocate and seal one transformer context over a model/session pair with complete rollback. */
int yvex_runtime_transformer_context_open(yvex_runtime_transformer_context **out,
                                          yvex_runtime_model *model, yvex_runtime_execution_session *session,
                                          const yvex_runtime_transformer_options *options, yvex_error *err)
{
    yvex_runtime_transformer_context *context;
    yvex_runtime_moe_options moe_options;
    yvex_transformer_plan_facts facts;
    const yvex_moe_plan *moe_plan;
    int rc;
    if (out) *out = NULL;
    if (!out || !model || !session || !options || !options->context_capacity ||
        (options->tensor_scope != YVEX_TENSOR_SCOPE_GLOBAL &&
         options->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT) ||
        options->workspace_token_capacity > options->context_capacity)
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                          "transformer context owners/options are required");
    context = (yvex_runtime_transformer_context *)calloc(1u, sizeof(*context));
    if (!context) return transformer_runtime_refuse(err, YVEX_ERR_NOMEM,
                                                    "transformer context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_runtime_model_view_get(model);
    context->session_view = yvex_runtime_session_view_get(session);
    context->options = *options;
    if (!context->model_view || !context->session_view || context->session_view->model != model ||
        !context->model_view->binding->capabilities.transformer_ready ||
        pthread_mutex_init(&context->mutex, NULL) != 0) {
        rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                        "transformer model/session capability is unavailable");
        goto failure;
    }
    context->mutex_ready = 1;
    memset(&moe_options, 0, sizeof(moe_options));
    moe_options.maximum_host_bytes = options->maximum_host_bytes;
    moe_options.maximum_device_bytes = options->maximum_device_bytes;
    moe_options.row_capacity = options->workspace_token_capacity
                                   ? options->workspace_token_capacity : 1ull;
    moe_options.tensor_scope = options->tensor_scope;
    moe_options.cancel_requested = options->cancel_requested;
    moe_options.cancel_context = options->cancel_context;
    moe_options.defer_cuda_workspace = 1;
    moe_options.evidence_level = options->evidence_level;
    moe_options.execution_profile = options->execution_profile;
    rc = yvex_runtime_moe_context_open(&context->moe, model, session, &moe_options,
                                       &context->moe_workspace_bytes, err);
    moe_plan = yvex_runtime_moe_context_plan(context->moe);
    if (rc == YVEX_OK)
        rc = transformer_runtime_plan_facts(
            context->model_view, options->tensor_scope, &facts, err);
    if (rc == YVEX_OK)
        rc = yvex_transformer_plan_build(&context->plan, &facts,
                                         transformer_runtime_attention(
                                             context->model_view, options->tensor_scope),
                                         moe_plan, err);
    if (rc == YVEX_OK) rc = transformer_runtime_globals(context, err);
    if (rc == YVEX_OK && options->workspace_token_capacity)
        rc = transformer_runtime_buffers(context, options->workspace_token_capacity, err);
    if (rc != YVEX_OK) goto failure;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    (void)yvex_runtime_transformer_context_close(&context, NULL);
    return rc;
}
const yvex_transformer_plan *yvex_runtime_transformer_context_plan(const yvex_runtime_transformer_context *context)
{
    return context ? context->plan : NULL;
}
const yvex_runtime_execution_session *yvex_runtime_transformer_context_session(
    const yvex_runtime_transformer_context *context)
{
    return context ? context->session : NULL;
}
static int transformer_execution_identity(
    const yvex_transformer_plan_summary *plan, const yvex_runtime_transformer_request *request,
    yvex_runtime_transformer_result *result, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.transformer.execution.v3") ||
        !yvex_sha256_update_text(&hash, plan->transformer_plan_identity) ||
        !yvex_sha256_update_text(&hash, result->input_identity) ||
        !yvex_sha256_update_text(&hash, result->pre_normalized_hidden_digest) ||
        !yvex_sha256_update_text(&hash, result->normalized_hidden_digest) ||
        !yvex_sha256_update_text(&hash, result->feature_digest) ||
        !yvex_sha256_update_text(&hash, result->persistent_state_digest) ||
        !yvex_sha256_update_u64(&hash, request->phase) ||
        !yvex_sha256_update_u64(&hash, request->backend) ||
        !yvex_sha256_update_u64(&hash, request->chunk_tokens) ||
        !yvex_sha256_update_u64(&hash, request->transaction_disposition) ||
        !yvex_sha256_update_u64(&hash, request->candidate_block_visible) ||
        !yvex_sha256_update_u64(&hash, request->retain_prefix_checkpoints) ||
        !yvex_sha256_update_u64(&hash, request->feature_layer_count))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE, "transformer execution identity failed");
    for (index = 0ull; index < request->feature_layer_count; ++index)
        if (!yvex_sha256_update_u64(&hash, request->feature_layer_ordinals[index]))
            return transformer_runtime_refuse(err, YVEX_ERR_STATE, "transformer feature identity derivation failed");
    if (!yvex_sha256_final(&hash, digest))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE, "transformer execution identity failed");
    yvex_sha256_hex(digest, result->execution_identity);
    return YVEX_OK;
}
int yvex_runtime_transformer_context_validate_input(const yvex_runtime_transformer_context *context,
    const yvex_transformer_input *input, yvex_error *err)
{
    if (!context || !input)
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                          "transformer input validation owners are required");
    return yvex_transformer_input_validate(input, context->plan,
                                           context->model_view->binding, err);
}
static int transformer_device_output_publish(yvex_runtime_transformer_context *context,
    const yvex_transformer_plan_summary *plan,
    const transformer_chunk_context *chunk, const char *digest_domain,
    yvex_device_tensor *tensor, char digest[YVEX_SHA256_HEX_CAP],
    yvex_execution_device_view *view, yvex_error *err)
{
    const yvex_attention_state_provider *provider = context->options.tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
            ? context->session_view->draft_attention_state_provider
            : context->session_view->attention_state_provider;
    if (digest_domain && !transformer_device_value_digest(
            digest_domain, plan, chunk->result->position_after,
            chunk->result->persistent_state_digest, digest))
        return YVEX_ERR_STATE;
    return yvex_runtime_device_view_bind(view, YVEX_EXECUTION_DEVICE_HIDDEN,
        context->model, context->session, provider,
        context->options.execution_profile, tensor, 0ull, chunk->token_count,
        plan->hidden_width, err);
}
static int transformer_execution_finish(
    yvex_runtime_transformer_context *context, const yvex_transformer_input_summary *input_summary,
    const yvex_transformer_plan_summary *plan,
    const yvex_runtime_transformer_request *request,
    yvex_runtime_transformer_output *output, unsigned long long output_count,
    unsigned long long feature_elements, transformer_chunk_context *chunk,
    yvex_runtime_transformer_result *result, int rc, yvex_error *err)
{
    yvex_graph_attention_state_summary after = {0};
    unsigned char layer_digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_error primary = err ? *err : (yvex_error){0}, summary_error;
    yvex_error_clear(&summary_error);
    if (transformer_state_summary(context, &after, &summary_error) != YVEX_OK &&
        rc == YVEX_OK) {
        rc = yvex_error_code(&summary_error);
        if (err) *err = summary_error;
    } else if (rc != YVEX_OK && err) {
        *err = primary;
    }
    if (rc == YVEX_OK &&
        request->transaction_disposition == YVEX_ATTENTION_TRANSACTION_STAGE &&
        !after.staged_batch_complete)
        rc = transformer_runtime_refuse(
            err, YVEX_ERR_STATE, "transformer staged state is incomplete");
    result->committed_prefix = result->position_after =
        request->transaction_disposition == YVEX_ATTENTION_TRANSACTION_STAGE &&
                after.staged_batch_complete
            ? after.staged_next_position
            : after.next_position;
    result->generation_after =
        request->transaction_disposition == YVEX_ATTENTION_TRANSACTION_STAGE &&
                after.staged_batch_complete
            ? after.staged_generation
            : after.generation;
    yvex_runtime_identity_copy(
        result->persistent_state_digest,
        request->transaction_disposition == YVEX_ATTENTION_TRANSACTION_STAGE &&
                after.staged_batch_complete
            ? after.staged_state_content_identity
            : after.state_content_identity);
    if (rc == YVEX_OK && yvex_sha256_final(&chunk->layer_hash, layer_digest))
        yvex_sha256_hex(layer_digest, result->layer_digest);
    else if (rc == YVEX_OK)
        rc = transformer_runtime_refuse(
            err, YVEX_ERR_STATE, "transformer layer digest finalization failed");
    if (rc == YVEX_OK && yvex_sha256_final(&chunk->routing_hash, layer_digest))
        yvex_sha256_hex(layer_digest, result->routing_digest);
    else if (rc == YVEX_OK)
        rc = transformer_runtime_refuse(
            err, YVEX_ERR_STATE, "transformer routing digest finalization failed");
    if (rc == YVEX_OK &&
        yvex_sha256_final(&chunk->embedding_hash, layer_digest))
        yvex_sha256_hex(layer_digest, result->embedding_digest);
    else if (rc == YVEX_OK)
        rc = transformer_runtime_refuse(
            err, YVEX_ERR_STATE, "transformer embedding digest finalization failed");
    if (rc == YVEX_OK && request->backend == YVEX_BACKEND_KIND_CUDA &&
        context->options.evidence_level != YVEX_ATTENTION_EVIDENCE_FULL)
        yvex_runtime_identity_copy(result->final_expanded_digest,
                                   chunk->last_expanded_digest);
    else if (rc == YVEX_OK &&
             !yvex_execution_f32_digest(
                 "yvex.transformer.final-expanded.v1", chunk->current,
                 chunk->token_count * plan->expanded_width,
                 result->final_expanded_digest))
        rc = transformer_runtime_refuse(
            err, YVEX_ERR_STATE, "transformer expanded digest derivation failed");
    if (rc == YVEX_OK && output->pre_normalized_hidden) {
        if (!yvex_execution_f32_digest(
                "yvex.transformer.pre-normalized-hidden.v1",
                output->pre_normalized_hidden, output_count,
                result->pre_normalized_hidden_digest))
            rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                "transformer pre-normalized output digest derivation failed");
        else
            result->pre_normalized_hidden_host_available = 1;
    }
    if (rc == YVEX_OK && context->options.device_pre_normalized_output &&
        transformer_device_output_publish(
            context, plan, chunk, output->pre_normalized_hidden ? NULL
                : "yvex.transformer.device-pre-normalized-hidden.v1",
            &context->device_pre_normalized_publication,
            result->pre_normalized_hidden_digest,
            &result->device_pre_normalized_hidden, err) != YVEX_OK)
        rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
            "transformer device pre-normalized publication failed");
    else if (rc == YVEX_OK && context->options.device_pre_normalized_output)
        result->pre_normalized_hidden_device_available = 1;
    if (rc == YVEX_OK && output->normalized_hidden) {
        if (!yvex_execution_f32_digest(
                "yvex.transformer.normalized-hidden.v1",
                output->normalized_hidden, output_count,
                result->normalized_hidden_digest))
            rc = transformer_runtime_refuse(
                err, YVEX_ERR_STATE, "transformer output digest derivation failed");
        else
            result->normalized_hidden_host_available = 1;
    } else if (rc == YVEX_OK &&
               (!context->options.device_hidden_output ||
                transformer_device_output_publish(
                    context, plan, chunk, "yvex.transformer.device-hidden.v1",
                    &context->device_hidden_publication,
                    result->normalized_hidden_digest,
                    &result->device_hidden, err) != YVEX_OK))
        rc = transformer_runtime_refuse(
            err, YVEX_ERR_STATE, "transformer device-hidden publication failed");
    else if (rc == YVEX_OK)
        result->normalized_hidden_device_available = 1;
    if (rc == YVEX_OK && feature_elements) {
        int sealed = output->features
            ? yvex_execution_f32_digest("yvex.transformer.target-features.v1",
                  output->features, feature_elements, result->feature_digest)
            : transformer_device_value_digest("yvex.transformer.device-target-features.v1",
                  plan, result->position_after, result->layer_digest, result->feature_digest);
        if (!sealed) rc = transformer_runtime_refuse(
            err, YVEX_ERR_STATE, "transformer feature digest derivation failed");
    }
    if (rc == YVEX_OK) {
        result->feature_layer_count = request->feature_layer_count;
        result->feature_row_count = input_summary->token_count;
        rc = transformer_execution_identity(plan, request, result, err);
    }
    if (pthread_mutex_lock(&context->mutex) == 0) {
        context->busy = 0;
        if (rc == YVEX_OK) context->execution_count++;
        (void)pthread_mutex_unlock(&context->mutex);
    }
    if (rc == YVEX_OK) result->completed = 1;
    return rc;
}
/* Publish normalized rows only after each identity-bound full-stack chunk commits. */
int yvex_runtime_transformer_execute(yvex_runtime_transformer_context *context,
                                     const yvex_transformer_input *input,
                                     const yvex_runtime_transformer_request *request,
                                     yvex_runtime_transformer_output *output,
                                     yvex_runtime_transformer_result *result,
                                     yvex_error *err)
{
    const yvex_transformer_input_summary *input_summary =
        yvex_transformer_input_summary_get(input);
    const yvex_transformer_plan_summary *plan = context ? yvex_transformer_plan_summary_get(context->plan) : NULL;
    const unsigned int *tokens = yvex_transformer_input_token_ids(input);
    yvex_graph_attention_state_summary before = {0};
    transformer_chunk_context chunk;
    unsigned long long offset = 0ull, output_count, feature_elements;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !input || !input_summary || !plan || !tokens || !request || !output ||
        !result || !request->chunk_tokens ||
        request->backend != yvex_backend_kind_of(context->session_view->backend) ||
        ((context->options.device_hidden_output ||
          context->options.device_pre_normalized_output) &&
         input_summary->token_count > request->chunk_tokens) ||
        (context->options.device_pre_normalized_output &&
         request->backend != YVEX_BACKEND_KIND_CUDA) ||
        !yvex_core_u64_mul(input_summary->token_count, plan->hidden_width, &output_count) ||
        (output->normalized_hidden && output->capacity < output_count) ||
        (!output->normalized_hidden &&
         (!context->options.device_hidden_output || output->capacity ||
          request->backend != YVEX_BACKEND_KIND_CUDA)))
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                          "transformer execution owners/capacity are invalid");
    rc = transformer_feature_request_validate(
        plan, input_summary, request, output, context->options.evidence_level,
        &feature_elements, err);
    if (rc != YVEX_OK) return rc;
    if (pthread_mutex_lock(&context->mutex) != 0)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE, "transformer context lock failed");
    if (context->busy || context->invalidated) {
        (void)pthread_mutex_unlock(&context->mutex);
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer context is busy or invalidated");
    }
    context->busy = 1;
    (void)pthread_mutex_unlock(&context->mutex);
    rc = yvex_runtime_transformer_context_validate_input(context, input, err);
    if (rc == YVEX_OK) rc = transformer_runtime_buffers(context, request->chunk_tokens, err);
    if (rc == YVEX_OK) rc = transformer_state_summary(context, &before, err);
    if (rc == YVEX_OK) rc = transformer_prepare(context, input_summary, request, &before, err);
    result->token_start = result->position_before = input_summary->token_start;
    result->token_count = input_summary->token_count;
    result->phase = request->phase;
    result->generation_before = before.generation;
    yvex_runtime_identity_copy(result->input_identity, input_summary->input_identity);
    memset(&chunk, 0, sizeof(chunk));
    chunk.owner = context;
    chunk.request = request;
    chunk.output = output;
    chunk.tokens = tokens;
    chunk.result = result;
    yvex_sha256_init(&chunk.layer_hash);
    yvex_sha256_init(&chunk.routing_hash);
    yvex_sha256_init(&chunk.embedding_hash);
    (void)yvex_sha256_update_text(&chunk.layer_hash, "yvex.transformer.layer-stack.v1");
    (void)yvex_sha256_update_text(&chunk.routing_hash, "yvex.transformer.routing-stack.v1");
    (void)yvex_sha256_update_text(&chunk.embedding_hash, "yvex.transformer.embedding.v1");
    while (rc == YVEX_OK && offset < input_summary->token_count) {
        yvex_attention_execution_request execution;
        yvex_attention_probe_result attention_result;
        yvex_runtime_model_failure failure;
        unsigned long long remaining = input_summary->token_count - offset;
        unsigned long long count = remaining < request->chunk_tokens
                                       ? remaining : request->chunk_tokens;
        unsigned long long started_ns;
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context)) {
            rc = transformer_runtime_refuse(err, YVEX_ERR_CANCELLED,
                                            "transformer execution cancelled between chunks");
            break;
        }
        rc = yvex_runtime_transformer_context_validate_input(context, input, err);
        memset(&execution, 0, sizeof(execution));
        memset(&attention_result, 0, sizeof(attention_result));
        chunk.token_offset = offset;
        chunk.token_count = count;
        chunk.token_start = input_summary->token_start + offset;
        chunk.layer_ordinal = 0ull;
        chunk.feature_next = 0ull;
        chunk.backend = request->backend;
        chunk.current = context->expanded_a;
        chunk.next = context->expanded_b;
        started_ns = yvex_core_monotonic_ns();
        if (rc == YVEX_OK) rc = transformer_runtime_embedding(&chunk, err);
        if (rc == YVEX_OK)
            result->embedding_ns += yvex_core_monotonic_ns() - started_ns;
        if (rc == YVEX_OK && request->backend == YVEX_BACKEND_KIND_CUDA &&
            (!yvex_backend_tensor_f32_subview(context->device_residual[0], 0ull,
                                         count * plan->expanded_width,
                                         &chunk.device_current) ||
             !yvex_backend_tensor_f32_subview(context->device_residual[1], 0ull,
                                         count * plan->expanded_width,
                                         &chunk.device_next) ||
             !yvex_backend_tensor_f32_subview(context->device_attention, 0ull,
                                         count * plan->expanded_width,
                                         &chunk.device_attention) ||
             !yvex_backend_tensor_f32_subview(context->device_hidden, 0ull,
                                         count * plan->hidden_width,
                                         &chunk.device_hidden)))
            rc = transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                            "transformer CUDA chunk views are invalid");
        if (rc != YVEX_OK) break;
        chunk.activation = (transformer_activation_source){
            .host_input = chunk.current,
            .device_input = request->backend == YVEX_BACKEND_KIND_CUDA
                                ? &chunk.device_current : NULL,
            .device_output = request->backend == YVEX_BACKEND_KIND_CUDA
                                 ? &chunk.device_attention : NULL,
            .token_count = count, .input_width = plan->expanded_width};
        execution.backend = request->backend;
        execution.tensor_scope = context->options.tensor_scope;
        execution.execution_phase = context->options.tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                                        ? YVEX_EXECUTION_PHASE_DRAFT
                                        : request->retain_prefix_checkpoints
                                              ? YVEX_EXECUTION_PHASE_VERIFY
                                              : request->phase;
        execution.probe = YVEX_ATTENTION_PROBE_UNSPECIFIED;
        execution.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
        execution.operation_scope = YVEX_ATTENTION_OPERATION_ENVELOPE;
        execution.token_count = count;
        execution.token_position = chunk.token_start;
        execution.select_position = 1;
        execution.input_identity = input_summary->input_identity;
        execution.token_ids = tokens + offset;
        execution.activation_view = transformer_activation_view;
        execution.device_view = request->backend == YVEX_BACKEND_KIND_CUDA
                                    ? transformer_device_view : NULL;
        execution.activation_context = &chunk.activation;
        execution.cancel_requested = context->options.cancel_requested;
        execution.cancel_context = context->options.cancel_context;
        execution.evidence_level = context->options.evidence_level;
        execution.evidence = transformer_layer_evidence;
        execution.evidence_context = &chunk;
        execution.transaction_disposition = request->transaction_disposition;
        execution.candidate_block_visible = request->candidate_block_visible;
        execution.retain_prefix_checkpoints = request->retain_prefix_checkpoints;
        started_ns = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = yvex_runtime_attention_probe_execute(
                context->session, context->model, &execution,
                &attention_result, &failure, err);
        if (rc == YVEX_OK)
            result->attention_ns += yvex_core_monotonic_ns() - started_ns;
        if (request->backend == YVEX_BACKEND_KIND_CUDA)
            rc = transformer_moe_complete(&chunk, 0, rc, err);
        if (rc == YVEX_OK &&
            (chunk.layer_ordinal != plan->layer_count ||
             chunk.feature_next != request->feature_layer_count ||
                              attention_result.layers_executed != plan->layer_count))
            rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                            "transformer chunk skipped one or more layers");
        if (rc == YVEX_OK) {
            if (output->normalized_hidden)
            memcpy(output->normalized_hidden + offset * plan->hidden_width,
                   context->candidate_hidden,
                   (size_t)(count * plan->hidden_width) * sizeof(float));
            result->swa_layers += attention_result.swa_layers_executed;
            result->csa_layers += attention_result.csa_layers_executed;
            result->hca_layers += attention_result.hca_layers_executed;
            result->attention_weight_bytes += attention_result.payload_bytes_read;
            if (yvex_execution_memory_facts_merge(
                    &result->memory, &attention_result.memory, err) != YVEX_OK) {
                rc = yvex_error_code(err);
                break;
            }
            result->h2d_bytes += attention_result.h2d_bytes;
            result->d2h_bytes += attention_result.d2h_bytes;
            result->d2d_bytes += attention_result.d2d_bytes;
            result->stream_synchronizations += attention_result.stream_synchronizations;
            result->device_synchronizations += attention_result.device_synchronizations;
            result->kernel_launches += attention_result.kernel_launches;
            result->attention_device_ns += attention_result.cuda_device_execution_elapsed_ns;
            result->chunk_count++;
            offset += count;
        }
    }
    return transformer_execution_finish(
        context, input_summary, plan, request, output, output_count,
        feature_elements, &chunk, result, rc, err);
}
int yvex_runtime_transformer_context_close(yvex_runtime_transformer_context **context,
                                           yvex_error *err)
{
    yvex_device_tensor **buffers[6];
    unsigned long long index;
    int rc = YVEX_OK;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if ((*context)->mutex_ready && pthread_mutex_lock(&(*context)->mutex) == 0) {
        if ((*context)->busy) {
            (void)pthread_mutex_unlock(&(*context)->mutex);
            return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                              "busy transformer context cannot close");
        }
        (void)pthread_mutex_unlock(&(*context)->mutex);
    }
    buffers[0] = &(*context)->device_embedding_encoded;
    buffers[1] = &(*context)->device_embedding;
    buffers[2] = &(*context)->device_residual[0];
    buffers[3] = &(*context)->device_residual[1];
    buffers[4] = &(*context)->device_attention;
    buffers[5] = &(*context)->device_hidden;
    for (index = 0ull; rc == YVEX_OK && index < sizeof(buffers) / sizeof(buffers[0]); ++index)
        if (*buffers[index])
            rc = yvex_backend_tensor_release(
                (*context)->session_view->backend, buffers[index], err);
    for (index = 0ull; rc == YVEX_OK && index < YVEX_TRANSFORMER_WEIGHT_COUNT; ++index)
        if ((*context)->device_global[index])
            rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                             &(*context)->device_global[index], err);
    if (rc != YVEX_OK) return rc;
    rc = yvex_runtime_moe_context_close(&(*context)->moe, err);
    if (rc != YVEX_OK) return rc;
    yvex_transformer_plan_close(&(*context)->plan);
    for (index = 0ull; index < YVEX_TRANSFORMER_WEIGHT_COUNT; ++index) {
        free((*context)->global[index].bytes);
        free((*context)->global[index].decoded);
    }
    free((*context)->embedding_encoded);
    free((*context)->embedding); free((*context)->expanded_a); free((*context)->expanded_b);
    free((*context)->candidate_hidden); free((*context)->moe_combined);
    free((*context)->moe_post); free((*context)->moe_combination);
    free((*context)->moe_routed); free((*context)->moe_shared);
    if ((*context)->mutex_ready) (void)pthread_mutex_destroy(&(*context)->mutex);
    memset(*context, 0, sizeof(**context));
    free(*context);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
static int transformer_runtime_cleanup(void **opaque, yvex_error *err)
{
    return yvex_runtime_transformer_context_close(
        (yvex_runtime_transformer_context **)opaque, err);
}
static void transformer_operator_refuse(yvex_transformer_operator_result *result,
                                        const yvex_error *err)
{
    yvex_core_text_copy(result->status, sizeof(result->status), "refused");
    yvex_core_text_copy(result->reason, sizeof(result->reason),
                        err && yvex_error_is_set(err) ? yvex_error_message(err)
                                                     : "transformer execution refused");
}
int yvex_transformer_operator_execute(const yvex_transformer_operator_request *request,
                                      yvex_transformer_operator_result *result,
                                      yvex_runtime_cleanup_lease **retained_cleanup,
                                      yvex_error *err)
{
    yvex_runtime_model_open_request model_request = {0};
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_transformer_options options = {0};
    yvex_runtime_transformer_request execution_request = {0};
    yvex_runtime_transformer_output output = {0};
    yvex_transformer_input_limits limits = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_transformer_context *context = NULL;
    yvex_transformer_input *input = NULL;
    const yvex_runtime_model_view *model_view = NULL;
    const yvex_transformer_plan_summary *plan = NULL;
    const yvex_transformer_input_summary *input_summary = NULL;
    yvex_error primary = {0};
    unsigned long long output_count;
    int rc, cleanup_rc, adopted = 0;
    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !retained_cleanup || *retained_cleanup ||
        !request->target || !request->artifact_path || !request->runtime_binding_path ||
        !request->input_path || !request->chunk_tokens || !request->context_capacity ||
        (request->backend != YVEX_BACKEND_KIND_CPU &&
         request->backend != YVEX_BACKEND_KIND_CUDA)) {
        rc = transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                        "complete transformer operator arguments are required");
        if (result) transformer_operator_refuse(result, err);
        return rc;
    }
    yvex_core_text_copy(result->command, sizeof(result->command),
                        "execute transformer run");
    yvex_core_text_copy(result->target, sizeof(result->target), request->target);
    yvex_core_text_copy(result->backend, sizeof(result->backend),
                        request->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    yvex_core_text_copy(result->phase, sizeof(result->phase), "prefill");
    model_request.artifact_path = request->artifact_path;
    model_request.runtime_binding_path = request->runtime_binding_path;
    model_request.target_id = request->target;
    model_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.backend = request->backend;
    session_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.maximum_device_bytes = request->maximum_device_bytes;
    rc = yvex_runtime_cleanup_lease_acquire(&cleanup, &model_request, &session_request,
                                            &model, &session, &failure, err);
    limits.maximum_file_bytes = request->maximum_host_bytes
                                    ? request->maximum_host_bytes : 1ull << 30u;
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_file(&input, request->input_path, &limits, err);
    options.maximum_host_bytes = request->maximum_host_bytes;
    options.maximum_device_bytes = request->maximum_device_bytes;
    options.context_capacity = request->context_capacity;
    options.cancel_requested = request->cancel_requested;
    options.cancel_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_context_open(&context, model, session, &options, err);
    if (rc == YVEX_OK) {
        rc = yvex_runtime_cleanup_lease_adopt(cleanup, context,
                                              transformer_runtime_cleanup, err);
        adopted = rc == YVEX_OK;
    }
    model_view = yvex_runtime_model_view_get(model);
    plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(context));
    input_summary = yvex_transformer_input_summary_get(input);
    if (rc == YVEX_OK &&
        (!model_view || !plan || !input_summary ||
         !yvex_core_u64_mul(input_summary->token_count, plan->hidden_width,
                            &output_count) || output_count > SIZE_MAX / sizeof(float)))
        rc = transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                        "transformer operator output extent overflowed");
    if (rc == YVEX_OK) {
        output.normalized_hidden = (float *)calloc((size_t)output_count, sizeof(float));
        output.capacity = output_count;
        if (!output.normalized_hidden)
            rc = transformer_runtime_refuse(err, YVEX_ERR_NOMEM,
                                            "transformer operator output allocation failed");
    }
    execution_request.backend = request->backend;
    execution_request.chunk_tokens = request->chunk_tokens;
    execution_request.phase = YVEX_TRANSFORMER_PHASE_PREFILL;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_execute(context, input, &execution_request,
                                              &output, &result->execution, err);
    if (rc == YVEX_OK) {
        yvex_core_text_copy(result->family, sizeof(result->family),
                            model_view->adapter->family_name);
        yvex_runtime_identity_copy(result->artifact_identity,
                                   model_view->binding->artifact_identity);
        yvex_runtime_identity_copy(result->runtime_binding_identity,
                                   model_view->binding->identity);
        yvex_runtime_identity_copy(result->transformer_plan_identity,
                                   plan->transformer_plan_identity);
        result->hidden_width = plan->hidden_width;
        result->expanded_width = plan->expanded_width;
        result->layer_count = plan->layer_count;
        result->embedding_ready = result->transformer_plan_ready = 1;
        result->transformer_block_ready = result->transformer_stack_ready = 1;
        result->transformer_final_head_ready = result->transformer_final_norm_ready = 1;
        result->transformer_hidden_state_ready = result->full_model_prefill_ready = 1;
        result->transformer_ready = 1;
        result->single_token_transformer_component_ready = input_summary->token_count == 1ull;
    }
    free(output.normalized_hidden);
    yvex_transformer_input_close(&input);
    primary = err ? *err : (yvex_error){0};
    if (!adopted && context) {
        cleanup_rc = yvex_runtime_transformer_context_close(&context, err);
        if (rc == YVEX_OK && cleanup_rc != YVEX_OK) rc = cleanup_rc;
    }
    cleanup_rc = yvex_runtime_cleanup_lease_close(&cleanup, err);
    if (cleanup_rc != YVEX_OK) rc = cleanup_rc;
    else if (rc != YVEX_OK && err) *err = primary;
    if (cleanup) *retained_cleanup = cleanup;
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(result->status, sizeof(result->status), "complete");
        yvex_error_clear(err);
    } else {
        transformer_operator_refuse(result, err);
    }
    return rc;
}
