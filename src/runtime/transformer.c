/* Owner: runtime transformer execution.
 * Owns: transformer context, exact global-weight access, chunk coordination, state transaction, and publication.
 * Does not own: family semantics, attention/MoE mathematics, tokenizer, logits, repeated decode, CLI, or rendering.
 * Invariants: one attention transaction spans all 43 blocks and final hidden validation before one state commit.
 * Boundary: production numeric-token to normalized-hidden backbone execution over one runtime session.
 * Purpose: compose admitted embedding, attention, MoE, residual, and final-stage owners without rebuilding plans.
 * Inputs: sealed model/session, identity-bound token input, backend/chunk policy, and caller output capacity.
 * Effects: reads selected/global weights, reuses bounded buffers, commits complete chunks, and publishes results.
 * Failure: cancellation, drift, state, resource, or numeric error commits no failing chunk and no hidden output. */
#include <yvex/internal/transformer.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
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
    yvex_device_tensor *device_global[YVEX_TRANSFORMER_WEIGHT_COUNT];
    float *embedding, *expanded_a, *expanded_b, *candidate_hidden;
    float *moe_combined, *moe_post, *moe_combination, *moe_routed, *moe_shared;
    unsigned long long token_capacity, host_bytes, execution_count;
    char workspace_identity[YVEX_SHA256_HEX_CAP];
    pthread_mutex_t mutex;
    int mutex_ready, busy, invalidated;
};
static int transformer_hash_values(yvex_sha256 *hash, const float *values,
                                   unsigned long long count);
/* Purpose: identify a completed device-resident block when normal serving omits host numerics.
 * Inputs: exact plan/layer/routing facts. Effects: writes one evidence digest.
 * Failure: identity construction returns false. Boundary: this is not a numerical output digest. */
static int transformer_device_block_digest(
    const yvex_transformer_plan_summary *plan, unsigned long long layer,
    const char *routing, char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!plan || !routing || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.device-block.v1") ||
        !yvex_sha256_update_text(&hash, plan->transformer_plan_identity) ||
        !yvex_sha256_update_u64(&hash, layer) ||
        !yvex_sha256_update_text(&hash, routing) || !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
typedef struct {
    yvex_runtime_transformer_context *owner;
    const unsigned int *tokens;
    unsigned long long token_offset, token_count, token_start, layer_ordinal;
    float *current, *next;
    yvex_device_tensor device_current, device_next, device_attention, device_hidden;
    yvex_sha256 layer_hash, routing_hash, embedding_hash;
    yvex_backend_kind backend;
    yvex_runtime_transformer_result *result;
    char last_expanded_digest[YVEX_SHA256_HEX_CAP];
} transformer_chunk_context;
/* Purpose: publish one stable runtime transformer refusal. */
static int transformer_runtime_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.transformer", reason);
    return status;
}
/* Purpose: resolve one plan binding to its exact immutable materialization record. */
static const yvex_materialized_tensor_binding *transformer_runtime_binding(
    const yvex_runtime_transformer_context *context, yvex_transformer_weight_slot slot)
{
    const yvex_transformer_plan_summary *summary =
        context ? yvex_transformer_plan_summary_get(context->plan) : NULL;
    return summary && slot < YVEX_TRANSFORMER_WEIGHT_COUNT
               ? yvex_materialization_session_tensor_at(
                     context->model_view->materialization, summary->weights[slot].tensor_id)
               : NULL;
}
/* Purpose: project one exact descriptor/materialization binding into pointer-free plan facts.
 * Inputs: model view and required global role. Effects: fills one caller-owned binding fact.
 * Failure: rejects missing geometry or unavailable CPU/CUDA qtype compute.
 * Boundary: runtime import only; no payload bytes are read. */
static int transformer_runtime_binding_project(
    const yvex_runtime_model_view *view, yvex_tensor_role role,
    yvex_transformer_weight_binding *out, yvex_error *err)
{
    const yvex_runtime_tensor_binding *runtime;
    const yvex_materialized_tensor_binding *binding;
    const yvex_quant_numeric_capability *capability;
    runtime = yvex_runtime_descriptor_find_role(
        view ? view->descriptor : NULL, role, YVEX_TENSOR_SCOPE_GLOBAL,
        YVEX_MATERIALIZATION_NO_INDEX, YVEX_MATERIALIZATION_NO_INDEX);
    binding = runtime ? yvex_materialization_session_tensor_at(
                            view->materialization, runtime->tensor_id) : NULL;
    capability = binding ? yvex_quant_numeric_capability_at(binding->qtype) : NULL;
    if (!binding || binding->role != role || !binding->encoded_bytes ||
        binding->expert_count > 1ull || !binding->backend_compatible)
        return transformer_runtime_refuse(
            err, YVEX_ERR_FORMAT, "transformer global binding is unavailable");
    if (!capability || !capability->reference_decoder_available ||
        !capability->dedicated_cpu_compute_available ||
        !capability->dedicated_cuda_compute_available)
        return transformer_runtime_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "transformer global binding qtype lacks CPU/CUDA execution");
    *out = (yvex_transformer_weight_binding){
        binding->tensor_id, binding->row_width, binding->row_count,
        binding->encoded_bytes, binding->role, binding->qtype};
    return YVEX_OK;
}
/* Purpose: assemble typed family/runtime facts for the family-neutral graph plan.
 * Inputs: sealed model view and registered adapter. Effects: fills pointer-free facts.
 * Failure: missing adapter policy or identities refuse before plan allocation.
 * Boundary: runtime owns adapter/descriptor projection; graph owns plan identity. */
static int transformer_runtime_plan_facts(
    const yvex_runtime_model_view *view, yvex_transformer_plan_facts *facts,
    yvex_error *err)
{
    const yvex_runtime_descriptor_summary *runtime = view
        ? yvex_runtime_descriptor_summary_get(view->descriptor) : NULL;
    const yvex_materialization_summary *material = view
        ? yvex_materialization_session_summary(view->materialization) : NULL;
    unsigned long long slot;
    memset(facts, 0, sizeof(*facts));
    if (!view || !view->adapter || !view->adapter->transformer_policy ||
        !view->adapter->transformer_policy(&facts->policy) || !runtime || !material)
        return transformer_runtime_refuse(
            err, YVEX_ERR_FORMAT, "transformer runtime plan facts are unavailable");
    facts->family_adapter_id = view->adapter->adapter_id;
    facts->family_adapter_version = view->adapter->adapter_version;
    facts->layer_count = runtime->layer_count;
    facts->vocabulary_size = runtime->vocabulary_size;
    facts->artifact_identity = material->artifact_identity;
    facts->materialization_identity = material->plan_identity;
    facts->logical_model_identity = runtime->logical_model_identity;
    facts->runtime_numeric_identity = runtime->runtime_numeric_identity;
    facts->runtime_descriptor_identity = runtime->runtime_descriptor_identity;
    for (slot = 0ull; slot < YVEX_TRANSFORMER_WEIGHT_COUNT; ++slot)
        if (transformer_runtime_binding_project(
                view, transformer_runtime_roles[slot], &facts->weights[slot], err) != YVEX_OK)
            return yvex_error_code(err);
    return YVEX_OK;
}
/* Purpose: read one exact admitted transformer binding range.
 * Inputs: context, binding, checked range, and destination. Effects: fills destination.
 * Failure: typed range/I/O refusal. Boundary: immutable materialization bytes only. */
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
                                              offset, destination, (size_t)bytes,
                                              &failure, err);
}
/* Purpose: decode one exact qtype extent through the canonical decoder.
 * Inputs: binding and block-exact encoded/output ranges. Effects: fills decoded values.
 * Failure: typed geometry/codec refusal. Boundary: no whole-collection decode. */
static int transformer_runtime_decode(const yvex_materialized_tensor_binding *binding,
                                      const unsigned char *encoded, unsigned long long bytes,
                                      float *decoded, unsigned long long count,
                                      yvex_error *err)
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
/* Purpose: allocate and decode final transformer globals once per context.
 * Inputs: sealed context and budgets. Effects: owns immutable decoded global weights.
 * Failure: typed allocation/read/decode refusal. Boundary: excludes embedding rows and layer weights. */
static int transformer_runtime_globals(yvex_runtime_transformer_context *context,
                                       yvex_error *err)
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
/* Purpose: allocate one stable backend tensor with exact byte/element geometry.
 * Inputs: CUDA context, name, dtype, element count, and byte extent.
 * Effects: returns one context-owned device allocation. Failure: typed backend refusal.
 * Boundary: allocation only; no weight or activation bytes are uploaded. */
static int transformer_device_tensor_open(
    yvex_runtime_transformer_context *context, yvex_device_tensor **out,
    const char *name, yvex_dtype dtype, unsigned long long elements,
    unsigned long long bytes, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    descriptor.name = name;
    descriptor.dtype = dtype;
    descriptor.rank = 1u;
    descriptor.dims[0] = elements;
    descriptor.bytes = bytes;
    return yvex_backend_tensor_alloc(context->session_view->backend, &descriptor, out, err);
}
/* Purpose: seal all stable CUDA transformer activation and final-weight resources.
 * Inputs: exact chunk extents and admitted global/embedding bindings.
 * Effects: allocates device tensors once and uploads decoded immutable final weights.
 * Failure: partial resources remain context-owned for deterministic close.
 * Boundary: no attention, MoE, token execution, or output publication. */
static int transformer_device_buffers(yvex_runtime_transformer_context *context,
                                      unsigned long long hidden,
                                      unsigned long long expanded,
                                      yvex_error *err)
{
    const yvex_materialized_tensor_binding *embedding = transformer_runtime_binding(
        context, YVEX_TRANSFORMER_WEIGHT_EMBEDDING);
    unsigned long long encoded, slot;
    int rc;
    if (yvex_backend_kind_of(context->session_view->backend) != YVEX_BACKEND_KIND_CUDA)
        return YVEX_OK;
    if (!embedding || !embedding->row_count ||
        embedding->encoded_bytes % embedding->row_count ||
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
    if (rc == YVEX_OK)
        rc = transformer_device_tensor_open(context, &context->device_embedding,
            "transformer-embedding", YVEX_DTYPE_F32, hidden,
            hidden * sizeof(float), err);
    if (rc == YVEX_OK)
        rc = transformer_device_tensor_open(context, &context->device_residual[0],
            "transformer-residual-a", YVEX_DTYPE_F32, expanded,
            expanded * sizeof(float), err);
    if (rc == YVEX_OK)
        rc = transformer_device_tensor_open(context, &context->device_residual[1],
            "transformer-residual-b", YVEX_DTYPE_F32, expanded,
            expanded * sizeof(float), err);
    if (rc == YVEX_OK)
        rc = transformer_device_tensor_open(context, &context->device_attention,
            "transformer-attention", YVEX_DTYPE_F32, expanded,
            expanded * sizeof(float), err);
    if (rc == YVEX_OK)
        rc = transformer_device_tensor_open(context, &context->device_hidden,
            "transformer-hidden", YVEX_DTYPE_F32, hidden, hidden * sizeof(float), err);
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
/* Purpose: allocate all chunk buffers once and refuse later growth.
 * Inputs: requested token capacity. Effects: allocates stable host publication/scratch ranges.
 * Failure: no incomplete capacity is admitted. Boundary: no execution or state mutation. */
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
/* Purpose: borrow an exact leading byte span from one backend-owned encoded tensor.
 * Inputs: parent device tensor, requested bytes, and output view. Effects: publishes a borrowed subview.
 * Failure: returns false for absent or out-of-bounds storage. Boundary: never allocates or transfers. */
static int transformer_encoded_subview(const yvex_device_tensor *source,
                                       unsigned long long bytes,
                                       yvex_device_tensor *view)
{
    if (!source || !view || source->dtype != YVEX_DTYPE_I8 || !bytes ||
        bytes > source->bytes)
        return 0;
    *view = *source;
    view->rank = 1u;
    view->dims[0] = bytes;
    view->bytes = bytes;
    return 1;
}
/* Purpose: prepare selected embedding rows and exact initial expanded state.
 * Inputs: token chunk and admitted embedding binding. Effects: reads selected rows and fills scratch.
 * Failure: typed range/qtype/numeric refusal. Boundary: never reads the complete vocabulary matrix. */
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
            transformer_runtime_decode(binding, encoded,
                                       context->embedding_row_bytes,
                                       context->embedding + token * s->hidden_width,
                                       s->hidden_width, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    chunk->result->embedding_rows += chunk->token_count;
    chunk->result->embedding_bytes += chunk->token_count * context->embedding_row_bytes;
    if (!transformer_hash_values(&chunk->embedding_hash, context->embedding,
                                 chunk->token_count * s->hidden_width))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer embedding digest update failed");
    if (chunk->backend == YVEX_BACKEND_KIND_CUDA) {
        unsigned long long bytes = chunk->token_count * context->embedding_row_bytes;
        yvex_device_tensor encoded_view;
        int rc;
        if (!transformer_encoded_subview(context->device_embedding_encoded,
                                         bytes, &encoded_view))
            return transformer_runtime_refuse(
                err, YVEX_ERR_BOUNDS,
                "transformer CUDA embedding upload view is invalid");
        rc = yvex_backend_tensor_write(context->session_view->backend,
                                       &encoded_view,
                                       context->embedding_encoded, bytes, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_transformer_cuda_initial(
                context->session_view->backend, &encoded_view,
                binding->qtype, chunk->token_count, s->hidden_width,
                s->residual_streams, context->device_embedding,
                context->device_residual[0], err);
        if (rc != YVEX_OK) return rc;
        chunk->result->h2d_bytes += bytes;
        chunk->result->kernel_launches++;
    }
    return yvex_transformer_initial_residual(context->plan, context->embedding,
                                             chunk->token_count, context->expanded_a, err);
}
/* Purpose: hash finite F32 values by canonical bits. Inputs: values/hash. Effects: updates hash.
 * Failure: false on non-finite/hash refusal. Boundary: excludes pointers and native aggregate bytes. */
static int transformer_hash_values(yvex_sha256 *hash, const float *values,
                                   unsigned long long count)
{
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        if (!isfinite(values[index])) return 0;
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(hash, bits)) return 0;
    }
    return 1;
}
/* Purpose: derive one checked borrowed F32 range without creating device ownership.
 * Inputs: one owned tensor plus element offset/count. Effects: fills a non-owning alias.
 * Failure: false on owner/dtype/range mismatch. Boundary: alias must never be released. */
static int transformer_device_subview(const yvex_device_tensor *source,
                                      unsigned long long offset,
                                      unsigned long long count,
                                      yvex_device_tensor *view)
{
    unsigned long long end;
    if (!source || !view || source->dtype != YVEX_DTYPE_F32 ||
        !yvex_core_u64_add(offset, count, &end) ||
        end > source->bytes / sizeof(float)) return 0;
    *view = *source;
    view->rank = 1u;
    view->dims[0] = count;
    view->bytes = count * sizeof(float);
    view->data = source->data + offset * sizeof(float);
    return 1;
}
/* Purpose: borrow the current internally generated expanded activation for one ordered layer. */
static int transformer_activation_view(void *opaque, unsigned long long layer_ordinal,
                                       unsigned long long token_count, const float **input,
                                       unsigned long long *stride, yvex_error *err)
{
    transformer_chunk_context *chunk = (transformer_chunk_context *)opaque;
    const yvex_transformer_plan_summary *s = chunk && chunk->owner
        ? yvex_transformer_plan_summary_get(chunk->owner->plan) : NULL;
    if (!chunk || !s || !input || !stride || layer_ordinal != chunk->layer_ordinal ||
        token_count != chunk->token_count || layer_ordinal >= s->layer_count)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer attention activation order is invalid");
    *input = chunk->current;
    *stride = s->expanded_width;
    return YVEX_OK;
}
/* Purpose: borrow the current and next device-resident activation views for attention.
 * Inputs: exact ordered layer/chunk coordinates. Effects: returns non-owning device aliases.
 * Failure: typed order or geometry refusal. Boundary: CUDA transformer composition only. */
static int transformer_device_view(void *opaque, unsigned long long layer_ordinal,
                                   unsigned long long token_count,
                                   const yvex_device_tensor **input,
                                   yvex_device_tensor **output, yvex_error *err)
{
    transformer_chunk_context *chunk = (transformer_chunk_context *)opaque;
    if (!chunk || !input || !output || layer_ordinal != chunk->layer_ordinal ||
        token_count != chunk->token_count || chunk->backend != YVEX_BACKEND_KIND_CUDA)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer CUDA activation order is invalid");
    *input = &chunk->device_current;
    *output = &chunk->device_attention;
    return YVEX_OK;
}
/* Purpose: complete one ordered transformer block from its staged attention publication.
 * Inputs: exact layer/token coordinates, active-transaction publication, and output workspace.
 * Effects: executes MoE/deferred mHC post and publishes field-wise block evidence.
 * Failure: cancellation, geometry, routing, backend, or numeric refusal publishes no block result.
 * Boundary: internal production block API; the request coordinator owns attention and KV commit. */
int yvex_runtime_transformer_execute_block(
    yvex_runtime_transformer_context *context, unsigned long long layer_ordinal,
    const unsigned int *token_ids, unsigned long long token_count,
    yvex_backend_kind backend, const yvex_attention_publication *attention,
    const yvex_device_tensor *device_attention, yvex_device_tensor *device_output,
    float *expanded_output, yvex_runtime_transformer_block_result *result,
    yvex_error *err)
{
    const yvex_transformer_plan_summary *s = context
        ? yvex_transformer_plan_summary_get(context->plan) : NULL;
    const yvex_moe_plan *moe_plan = context
        ? yvex_runtime_moe_context_plan(context->moe) : NULL;
    const yvex_moe_layer_plan *layer = moe_plan
        ? yvex_moe_plan_layer_at(moe_plan, layer_ordinal) : NULL;
    yvex_sha256 routing_hash, output_hash, identity_hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long token, output_elements;
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
        backend != yvex_backend_kind_of(context->session_view->backend) ||
        (backend == YVEX_BACKEND_KIND_CUDA && (!device_attention || !device_output)))
        return transformer_runtime_refuse(err, YVEX_ERR_FORMAT,
                                          "transformer attention publication is incompatible");
    started_ns = yvex_core_monotonic_ns();
    normal_cuda = backend == YVEX_BACKEND_KIND_CUDA &&
                  context->options.evidence_level != YVEX_ATTENTION_EVIDENCE_FULL;
    yvex_sha256_init(&routing_hash);
    (void)yvex_sha256_update_text(&routing_hash, "yvex.transformer.block.routing.v1");
    for (token = 0ull; token < token_count; ++token) {
        yvex_moe_layer_result moe_result;
        yvex_device_tensor device_input, device_next_view;
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context))
            return transformer_runtime_refuse(err, YVEX_ERR_CANCELLED,
                                              "transformer block execution cancelled");
        memset(&moe_result, 0, sizeof(moe_result));
        moe_result.combined_output = context->moe_combined + token * s->hidden_width;
        moe_result.combined_capacity = s->hidden_width;
        moe_result.routed_output = context->moe_routed + token * s->hidden_width;
        moe_result.routed_capacity = s->hidden_width;
        moe_result.shared_output = context->moe_shared + token * s->hidden_width;
        moe_result.shared_capacity = s->hidden_width;
        moe_result.post = context->moe_post + token * s->residual_streams;
        moe_result.post_capacity = s->residual_streams;
        moe_result.combination = context->moe_combination +
            token * s->residual_streams * s->residual_streams;
        moe_result.combination_capacity = s->residual_streams * s->residual_streams;
        if (backend == YVEX_BACKEND_KIND_CUDA &&
            (!transformer_device_subview(device_attention,
                                         token * s->expanded_width, s->expanded_width,
                                         &device_input) ||
             !transformer_device_subview(device_output,
                                         token * s->expanded_width, s->expanded_width,
                                         &device_next_view)))
            return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                              "transformer CUDA MoE subview is invalid");
        rc = backend == YVEX_BACKEND_KIND_CUDA
                 ? yvex_runtime_moe_execute_layer_device_borrowed(
                       context->moe, layer_ordinal,
                       attention->envelope_output + token * s->expanded_width,
                       &device_input, &device_next_view,
                       token_ids[token], 1, &moe_result, err)
                 : yvex_runtime_moe_execute_layer_borrowed(
                       context->moe, layer_ordinal,
                       attention->envelope_output + token * s->expanded_width,
                       token_ids[token], 1, &moe_result, err);
        if (rc != YVEX_OK) return rc;
        result->hash_routers += layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID;
        result->learned_routers +=
            layer->router_class == YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE;
        result->routed_experts += moe_result.router.selected_count;
        result->shared_experts += layer->shared_experts;
        result->h2d_bytes += moe_result.host_to_device_bytes;
        result->d2h_bytes += moe_result.device_to_host_bytes;
        result->kernel_launches += moe_result.kernel_launches;
        result->d2d_bytes += moe_result.device_to_device_bytes;
        result->upload_count += moe_result.upload_count;
        result->download_count += moe_result.download_count;
        result->cache_hits += moe_result.cache_hits;
        result->cache_misses += moe_result.cache_misses;
        result->stream_synchronizations += moe_result.stream_synchronizations;
        result->device_synchronizations += moe_result.device_synchronizations;
        result->moe_ns += moe_result.total_ns;
        result->synchronization_ns += moe_result.synchronization_ns;
        if (!yvex_sha256_update_text(&routing_hash, moe_result.routing_digest))
            return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                              "transformer block routing digest failed");
    }
    if (!yvex_sha256_final(&routing_hash, digest))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer block routing digest failed");
    yvex_sha256_hex(digest, result->routing_digest);
    if (!normal_cuda) {
        rc = yvex_transformer_deferred_post(
            context->plan, attention->envelope_output, context->moe_combined,
            context->moe_post, context->moe_combination, token_count, expanded_output, err);
        if (rc != YVEX_OK) return rc;
        yvex_sha256_init(&output_hash);
        if (!yvex_sha256_update_text(&output_hash, "yvex.transformer.block.output.v1") ||
            !transformer_hash_values(&output_hash, expanded_output, output_elements) ||
            !yvex_sha256_final(&output_hash, digest))
            return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                              "transformer block output digest failed");
        yvex_sha256_hex(digest, result->expanded_digest);
    } else if (!transformer_device_block_digest(
                   s, layer_ordinal, result->routing_digest, result->expanded_digest))
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
/* Purpose: complete and advance one ordered block inside the active all-layer transaction.
 * Inputs: ordered callback context, exact backend, and completed attention publication.
 * Effects: executes the block API, accumulates evidence, swaps residual state, and may run final norm.
 * Failure: block, digest, download, or final-stage refusal aborts the outer transaction.
 * Boundary: request coordinator callback only; it neither begins nor commits persistent state. */
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
            return transformer_runtime_refuse(
                err, YVEX_ERR_STATE, "transformer device layer identity update failed");
    } else if (!transformer_hash_values(
                   &chunk->layer_hash, chunk->next,
                   chunk->token_count * s->expanded_width))
        return transformer_runtime_refuse(
            err, YVEX_ERR_STATE, "transformer layer digest update failed");
    yvex_runtime_identity_copy(chunk->last_expanded_digest, block.expanded_digest);
    if (!yvex_sha256_update_text(&chunk->routing_hash, block.routing_digest))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer routing digest update failed");
    chunk->result->hash_routers += block.hash_routers;
    chunk->result->learned_routers += block.learned_routers;
    chunk->result->routed_experts += block.routed_experts;
    chunk->result->shared_experts += block.shared_experts;
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
    if (chunk->layer_ordinal == s->layer_count && backend == YVEX_BACKEND_KIND_CUDA) {
        unsigned long long expanded_bytes = chunk->token_count * s->expanded_width * sizeof(float);
        unsigned long long hidden_bytes = chunk->token_count * s->hidden_width * sizeof(float);
        unsigned long long started_ns = yvex_core_monotonic_ns();
        rc = yvex_backend_transformer_cuda_final(
            context->session_view->backend, &chunk->device_current,
            context->device_global[YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION],
            context->device_global[YVEX_TRANSFORMER_WEIGHT_FINAL_BASE],
            context->device_global[YVEX_TRANSFORMER_WEIGHT_FINAL_SCALE],
            context->device_global[YVEX_TRANSFORMER_WEIGHT_OUTPUT_NORM],
            chunk->token_count, s->hidden_width, s->residual_streams,
            s->output_norm_epsilon, s->mhc_epsilon, &chunk->device_hidden, err);
        if (rc == YVEX_OK && context->options.evidence_level == YVEX_ATTENTION_EVIDENCE_FULL)
            rc = yvex_backend_tensor_read(context->session_view->backend,
                                          &chunk->device_current, chunk->current,
                                          expanded_bytes, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_tensor_read(context->session_view->backend,
                &chunk->device_hidden, context->candidate_hidden, hidden_bytes, err);
        if (rc == YVEX_OK) {
            chunk->result->d2h_bytes += hidden_bytes;
            if (context->options.evidence_level == YVEX_ATTENTION_EVIDENCE_FULL)
                chunk->result->d2h_bytes += expanded_bytes;
            chunk->result->kernel_launches++;
            chunk->result->final_ns += yvex_core_monotonic_ns() - started_ns;
        }
        return rc;
    }
    if (chunk->layer_ordinal == s->layer_count)
        return yvex_transformer_final_stage(
            context->plan, chunk->current, chunk->token_count,
            context->global[YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION].decoded,
            context->global[YVEX_TRANSFORMER_WEIGHT_FINAL_BASE].decoded,
            context->global[YVEX_TRANSFORMER_WEIGHT_FINAL_SCALE].decoded,
            context->global[YVEX_TRANSFORMER_WEIGHT_OUTPUT_NORM].decoded,
            context->candidate_hidden, err);
    return YVEX_OK;
}
/* Purpose: copy the complete session persistent-state summary. */
static int transformer_state_summary(const yvex_runtime_execution_session *session,
                                     yvex_graph_attention_state_summary *summary,
                                     yvex_error *err)
{
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    if (!view || !view->attention_state_provider || !view->attention_state_provider->summary ||
        view->attention_state_provider->summary(view->attention_state_provider->context,
                                                summary, err) != YVEX_OK)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer persistent state is unavailable");
    return YVEX_OK;
}
/* Purpose: build one exact full-layer attention capacity plan.
 * Inputs: model, start, and token extent. Effects: allocates caller-owned plan.
 * Failure: typed graph refusal. Boundary: no state allocation or execution. */
static int transformer_capacity_build(yvex_graph_attention_capacity_plan **out,
                                      const yvex_runtime_model_view *model,
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
                                                     model ? model->attention : NULL,
                                                     &request, err);
}
/* Purpose: prepare persistent-state capacity and CUDA attention workspace before mutation.
 * Inputs: context, admitted input/request, and state summary. Effects: seals reusable capacities.
 * Failure: typed budget/position/backend refusal. Boundary: no transformer chunk has begun. */
static int transformer_prepare(yvex_runtime_transformer_context *context,
                               const yvex_transformer_input_summary *input,
                               const yvex_runtime_transformer_request *request,
                               yvex_graph_attention_state_summary *state,
                               yvex_error *err)
{
    yvex_graph_attention_capacity_plan *capacity = NULL;
    yvex_runtime_model_failure failure;
    yvex_runtime_session_summary session;
    yvex_attention_failure attention_failure;
    unsigned long long final, workspace_tokens, workspace_start;
    int rc;
    if (request->phase != YVEX_TRANSFORMER_PHASE_PREFILL &&
        request->phase != YVEX_TRANSFORMER_PHASE_DECODE)
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                          "transformer execution phase is invalid");
    if (request->phase == YVEX_TRANSFORMER_PHASE_DECODE &&
        (input->token_count != 1ull || request->chunk_tokens != 1ull ||
         !input->token_start))
        return transformer_runtime_refuse(
            err, YVEX_ERR_STATE,
            "decode phase requires one token over a nonzero committed prefix");
    if (!yvex_core_u64_add(input->token_start, input->token_count, &final) ||
        final > context->options.context_capacity)
        return transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                          "transformer request exceeds context capacity");
    if (!state->prepared_layer_count) {
        if (input->token_start)
            return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                              "nonzero transformer start requires committed state");
        rc = transformer_capacity_build(&capacity, context->model_view, 0ull,
                                        context->options.context_capacity, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_session_prepare_attention_probe_state(
                context->session, context->model, capacity, &attention_failure, err);
        yvex_graph_attention_capacity_plan_close(&capacity);
        if (rc != YVEX_OK || transformer_state_summary(context->session, state, err) != YVEX_OK)
            return rc != YVEX_OK ? rc : yvex_error_code(err);
    }
    if (state->prepared_layer_count != state->layer_count || !state->position_consistent ||
        state->next_position != input->token_start || state->capacity < final)
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer state position/capacity is incompatible");
    if (request->backend == YVEX_BACKEND_KIND_CPU) return YVEX_OK;
    rc = yvex_runtime_session_summary_copy(context->session, &session, err);
    if (rc != YVEX_OK) return rc;
    if (context->workspace_identity[0])
        return session.host_workspace_owned && session.host_workspace_pinned &&
                       session.host_workspace_bytes && session.device_workspace_bytes &&
                       strcmp(session.workspace_identity,
                              context->workspace_identity) == 0
                   ? YVEX_OK
                   : transformer_runtime_refuse(
                         err, YVEX_ERR_STATE,
                         "transformer CUDA workspace identity changed after sealing");
    workspace_tokens = request->chunk_tokens < input->token_count
                           ? request->chunk_tokens : input->token_count;
    workspace_start = context->options.context_capacity - workspace_tokens;
    rc = transformer_capacity_build(&capacity, context->model_view, workspace_start,
                                    workspace_tokens, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_prepare_attention_workspace(
            context->session, YVEX_RUNTIME_MODE_EAGER, YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE,
            YVEX_ATTENTION_EVIDENCE_NONE, capacity, YVEX_MOE_CUDA_WORKSPACE_BYTES,
            &failure, err);
    yvex_graph_attention_capacity_plan_close(&capacity);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_summary_copy(context->session, &session, err);
    if (rc == YVEX_OK && (!session.host_workspace_owned ||
                          !session.host_workspace_pinned ||
                          !yvex_sha256_hex_valid(session.workspace_identity)))
        rc = transformer_runtime_refuse(
            err, YVEX_ERR_STATE,
            "transformer CUDA workspace did not publish stable ownership");
    if (rc == YVEX_OK)
        yvex_runtime_identity_copy(context->workspace_identity,
                                   session.workspace_identity);
    return rc;
}
/* Purpose: finalize one canonical F32 digest. */
static int transformer_values_digest(const char *domain, const float *values,
                                     unsigned long long count,
                                     char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !transformer_hash_values(&hash, values, count) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
/* Purpose: allocate and seal one transformer execution context over a model/session pair.
 * Inputs: matching model/session and budgets. Effects: owns plan, MoE context, weights, and mutex.
 * Failure: typed refusal with complete rollback. Boundary: no token execution or state mutation. */
int yvex_runtime_transformer_context_open(yvex_runtime_transformer_context **out,
                                          yvex_runtime_model *model,
                                          yvex_runtime_execution_session *session,
                                          const yvex_runtime_transformer_options *options,
                                          yvex_error *err)
{
    yvex_runtime_transformer_context *context;
    yvex_runtime_moe_options moe_options;
    yvex_transformer_plan_facts facts;
    const yvex_moe_plan *moe_plan;
    int rc;
    if (out) *out = NULL;
    if (!out || !model || !session || !options || !options->context_capacity ||
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
    moe_options.cancel_requested = options->cancel_requested;
    moe_options.cancel_context = options->cancel_context;
    moe_options.defer_cuda_workspace = 1;
    moe_options.evidence_level = options->evidence_level;
    rc = yvex_runtime_moe_context_open(&context->moe, model, session, &moe_options, err);
    moe_plan = yvex_runtime_moe_context_plan(context->moe);
    if (rc == YVEX_OK)
        rc = transformer_runtime_plan_facts(context->model_view, &facts, err);
    if (rc == YVEX_OK)
        rc = yvex_transformer_plan_build(&context->plan, &facts,
                                         context->model_view->attention, moe_plan, err);
    if (rc == YVEX_OK) rc = transformer_runtime_globals(context, err);
    if (rc == YVEX_OK && options->workspace_token_capacity)
        rc = transformer_runtime_buffers(
            context, options->workspace_token_capacity, err);
    if (rc != YVEX_OK) goto failure;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    (void)yvex_runtime_transformer_context_close(&context, NULL);
    return rc;
}
/* Purpose: borrow one context-owned plan. Inputs: context. Effects: none. Failure: NULL.
 * Boundary: borrowed lifetime ends when the context closes. */
const yvex_transformer_plan *yvex_runtime_transformer_context_plan(
    const yvex_runtime_transformer_context *context)
{
    return context ? context->plan : NULL;
}
/* Purpose: borrow the exact execution session paired with one transformer context.
 * Inputs: live context. Effects: none. Failure: NULL. Boundary: enables lifecycle coordinators
 * to prove owner pairing without exposing transformer-private resources. */
const yvex_runtime_execution_session *yvex_runtime_transformer_context_session(
    const yvex_runtime_transformer_context *context)
{
    return context ? context->session : NULL;
}
/* Purpose: seal the semantic execution identity after state and output publication.
 * Inputs: admitted plan/request and one complete mutable result.
 * Effects: writes only the result execution identity.
 * Failure: malformed hash input returns a typed runtime refusal.
 * Boundary: movement and timing evidence do not enter semantic execution identity. */
static int transformer_execution_identity(
    const yvex_transformer_plan_summary *plan,
    const yvex_runtime_transformer_request *request,
    yvex_runtime_transformer_result *result, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.transformer.execution.v2") ||
        !yvex_sha256_update_text(&hash, plan->transformer_plan_identity) ||
        !yvex_sha256_update_text(&hash, result->input_identity) ||
        !yvex_sha256_update_text(&hash, result->normalized_hidden_digest) ||
        !yvex_sha256_update_text(&hash, result->persistent_state_digest) ||
        !yvex_sha256_update_u64(&hash, request->phase) ||
        !yvex_sha256_update_u64(&hash, request->backend) ||
        !yvex_sha256_update_u64(&hash, request->chunk_tokens) ||
        !yvex_sha256_final(&hash, digest))
        return transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                          "transformer execution identity failed");
    yvex_sha256_hex(digest, result->execution_identity);
    return YVEX_OK;
}
/* Purpose: validate one token input against the exact model binding and transformer plan.
 * Inputs: live context and admitted memory/file input. Effects: checks file drift when applicable.
 * Failure: propagates typed identity, payload, or snapshot refusal. Boundary: validation only. */
int yvex_runtime_transformer_context_validate_input(
    const yvex_runtime_transformer_context *context,
    const yvex_transformer_input *input, yvex_error *err)
{
    if (!context || !input)
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                          "transformer input validation owners are required");
    return yvex_transformer_input_validate(input, context->plan,
                                           context->model_view->binding, err);
}
/* Purpose: execute an identity-bound token request as independently committed chunks.
 * Inputs: matching context/input/backend, output capacity, and deterministic chunk size.
 * Effects: commits each complete full-stack chunk and publishes normalized hidden rows after commit.
 * Failure: preserves prior chunks and leaves the failing chunk unpublished.
 * Boundary: normalized hidden state only; no logits, decode loop, tokenizer, or generation. */
int yvex_runtime_transformer_execute(yvex_runtime_transformer_context *context,
                                     const yvex_transformer_input *input,
                                     const yvex_runtime_transformer_request *request,
                                     yvex_runtime_transformer_output *output,
                                     yvex_runtime_transformer_result *result,
                                     yvex_error *err)
{
    const yvex_transformer_input_summary *input_summary =
        yvex_transformer_input_summary_get(input);
    const yvex_transformer_plan_summary *plan = context
        ? yvex_transformer_plan_summary_get(context->plan) : NULL;
    const unsigned int *tokens = yvex_transformer_input_token_ids(input);
    yvex_graph_attention_state_summary before = {0}, after = {0};
    yvex_error primary = {0}, summary_error;
    transformer_chunk_context chunk;
    unsigned long long offset = 0ull, output_count;
    unsigned char layer_digest[YVEX_SHA256_DIGEST_BYTES];
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !input || !input_summary || !plan || !tokens || !request || !output ||
        !output->normalized_hidden || !result || !request->chunk_tokens ||
        request->backend != yvex_backend_kind_of(context->session_view->backend) ||
        !yvex_core_u64_mul(input_summary->token_count, plan->hidden_width, &output_count) ||
        output->capacity < output_count)
        return transformer_runtime_refuse(err, YVEX_ERR_INVALID_ARG,
                                          "transformer execution owners/capacity are invalid");
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
    if (rc == YVEX_OK) rc = transformer_state_summary(context->session, &before, err);
    if (rc == YVEX_OK) rc = transformer_prepare(context, input_summary, request, &before, err);
    result->token_start = result->position_before = input_summary->token_start;
    result->token_count = input_summary->token_count;
    result->phase = request->phase;
    result->generation_before = before.generation;
    yvex_runtime_identity_copy(result->input_identity, input_summary->input_identity);
    memset(&chunk, 0, sizeof(chunk));
    chunk.owner = context;
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
        chunk.backend = request->backend;
        chunk.current = context->expanded_a;
        chunk.next = context->expanded_b;
        started_ns = yvex_core_monotonic_ns();
        if (rc == YVEX_OK) rc = transformer_runtime_embedding(&chunk, err);
        if (rc == YVEX_OK)
            result->embedding_ns += yvex_core_monotonic_ns() - started_ns;
        if (rc == YVEX_OK && request->backend == YVEX_BACKEND_KIND_CUDA &&
            (!transformer_device_subview(context->device_residual[0], 0ull,
                                         count * plan->expanded_width,
                                         &chunk.device_current) ||
             !transformer_device_subview(context->device_residual[1], 0ull,
                                         count * plan->expanded_width,
                                         &chunk.device_next) ||
             !transformer_device_subview(context->device_attention, 0ull,
                                         count * plan->expanded_width,
                                         &chunk.device_attention) ||
             !transformer_device_subview(context->device_hidden, 0ull,
                                         count * plan->hidden_width,
                                         &chunk.device_hidden)))
            rc = transformer_runtime_refuse(err, YVEX_ERR_BOUNDS,
                                            "transformer CUDA chunk views are invalid");
        if (rc != YVEX_OK) break;
        execution.backend = request->backend;
        execution.probe = YVEX_ATTENTION_PROBE_UNSPECIFIED;
        execution.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
        execution.operation_scope = YVEX_ATTENTION_OPERATION_ENVELOPE;
        execution.token_count = count;
        execution.token_position = chunk.token_start;
        execution.select_position = 1;
        execution.input_identity = input_summary->input_identity;
        execution.activation_view = transformer_activation_view;
        execution.device_view = request->backend == YVEX_BACKEND_KIND_CUDA
                                    ? transformer_device_view : NULL;
        execution.activation_context = &chunk;
        execution.cancel_requested = context->options.cancel_requested;
        execution.cancel_context = context->options.cancel_context;
        execution.evidence_level = context->options.evidence_level;
        execution.evidence = transformer_layer_evidence;
        execution.evidence_context = &chunk;
        started_ns = yvex_core_monotonic_ns();
        rc = yvex_runtime_attention_probe_execute(context->session, context->model,
                                                  &execution, &attention_result,
                                                  &failure, err);
        if (rc == YVEX_OK)
            result->attention_ns += yvex_core_monotonic_ns() - started_ns;
        if (rc == YVEX_OK && (chunk.layer_ordinal != plan->layer_count ||
                              attention_result.layers_executed != plan->layer_count))
            rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                            "transformer chunk skipped one or more layers");
        if (rc == YVEX_OK) {
            memcpy(output->normalized_hidden + offset * plan->hidden_width,
                   context->candidate_hidden,
                   (size_t)(count * plan->hidden_width) * sizeof(float));
            result->swa_layers += attention_result.swa_layers_executed;
            result->csa_layers += attention_result.csa_layers_executed;
            result->hca_layers += attention_result.hca_layers_executed;
            result->h2d_bytes += attention_result.h2d_bytes;
            result->d2h_bytes += attention_result.d2h_bytes;
            result->kernel_launches += attention_result.kernel_launches;
            result->attention_device_ns +=
                attention_result.cuda_device_execution_elapsed_ns;
            result->chunk_count++;
            offset += count;
        }
    }
    if (rc != YVEX_OK && err) primary = *err;
    yvex_error_clear(&summary_error);
    if (transformer_state_summary(context->session, &after, &summary_error) != YVEX_OK &&
        rc == YVEX_OK) {
        rc = yvex_error_code(&summary_error);
        if (err) *err = summary_error;
    } else if (rc != YVEX_OK && err) {
        *err = primary;
    }
    result->committed_prefix = result->position_after = after.next_position;
    result->generation_after = after.generation;
    yvex_runtime_identity_copy(result->persistent_state_digest, after.state_content_identity);
    if (rc == YVEX_OK && yvex_sha256_final(&chunk.layer_hash, layer_digest))
        yvex_sha256_hex(layer_digest, result->layer_digest);
    else if (rc == YVEX_OK)
        rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                        "transformer layer digest finalization failed");
    if (rc == YVEX_OK && yvex_sha256_final(&chunk.routing_hash, layer_digest))
        yvex_sha256_hex(layer_digest, result->routing_digest);
    else if (rc == YVEX_OK)
        rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                        "transformer routing digest finalization failed");
    if (rc == YVEX_OK && yvex_sha256_final(&chunk.embedding_hash, layer_digest))
        yvex_sha256_hex(layer_digest, result->embedding_digest);
    else if (rc == YVEX_OK)
        rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                        "transformer embedding digest finalization failed");
    if (rc == YVEX_OK && request->backend == YVEX_BACKEND_KIND_CUDA &&
        context->options.evidence_level != YVEX_ATTENTION_EVIDENCE_FULL)
        yvex_runtime_identity_copy(result->final_expanded_digest,
                                   chunk.last_expanded_digest);
    else if (rc == YVEX_OK &&
             !transformer_values_digest("yvex.transformer.final-expanded.v1", chunk.current,
                                        chunk.token_count * plan->expanded_width,
                                        result->final_expanded_digest))
        rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                        "transformer expanded digest derivation failed");
    if (rc == YVEX_OK &&
        !transformer_values_digest("yvex.transformer.normalized-hidden.v1",
                                   output->normalized_hidden, output_count,
                                   result->normalized_hidden_digest))
        rc = transformer_runtime_refuse(err, YVEX_ERR_STATE,
                                        "transformer output digest derivation failed");
    if (rc == YVEX_OK)
        rc = transformer_execution_identity(plan, request, result, err);
    if (pthread_mutex_lock(&context->mutex) == 0) {
        context->busy = 0;
        if (rc == YVEX_OK) context->execution_count++;
        (void)pthread_mutex_unlock(&context->mutex);
    }
    if (rc == YVEX_OK) result->completed = 1;
    return rc;
}
/* Purpose: release transformer context resources in reverse ownership order.
 * Inputs: idle context owner. Effects: closes MoE and frees weights/scratch/mutex.
 * Failure: typed retryable dependent cleanup refusal. Boundary: does not close model/session. */
int yvex_runtime_transformer_context_close(yvex_runtime_transformer_context **context,
                                           yvex_error *err)
{
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
    if ((*context)->device_embedding_encoded)
        rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                         &(*context)->device_embedding_encoded, err);
    if (rc == YVEX_OK && (*context)->device_embedding)
        rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                         &(*context)->device_embedding, err);
    for (index = 0ull; rc == YVEX_OK && index < 2ull; ++index)
        if ((*context)->device_residual[index])
            rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                             &(*context)->device_residual[index], err);
    if (rc == YVEX_OK && (*context)->device_attention)
        rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                         &(*context)->device_attention, err);
    if (rc == YVEX_OK && (*context)->device_hidden)
        rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                         &(*context)->device_hidden, err);
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
/* Purpose: adapt one transformer context to the typed cleanup-lease protocol. */
static int transformer_runtime_cleanup(void **opaque, yvex_error *err)
{
    return yvex_runtime_transformer_context_close(
        (yvex_runtime_transformer_context **)opaque, err);
}
/* Purpose: publish one operator refusal without converting the domain status. */
static void transformer_operator_refuse(yvex_transformer_operator_result *result,
                                        const yvex_error *err)
{
    yvex_core_text_copy(result->status, sizeof(result->status), "refused");
    yvex_core_text_copy(result->reason, sizeof(result->reason),
                        err && yvex_error_is_set(err) ? yvex_error_message(err)
                                                     : "transformer execution refused");
}
/* Purpose: execute one production transformer request through operator-owned resources.
 * Inputs: exact artifact/binding/token paths, backend, capacity, and budgets.
 * Effects: opens one model/session/context, executes, closes, and publishes typed facts.
 * Failure: retains only a cleanup lease when deterministic cleanup itself refuses.
 * Boundary: no command parsing, rendering, tokenizer, logits, decode loop, or generation. */
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
