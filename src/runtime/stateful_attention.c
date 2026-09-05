/* Append direct K/V projections once, execute H28, and stage the matching logical delta. */
#include <yvex/internal/stateful_attention.h>

#include <stdint.h>
#include <string.h>

#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>

static int stateful_refuse(yvex_error *err, yvex_status status,
                           const char *reason)
{
    if (!yvex_error_is_set(err))
        yvex_error_set(err, status, "runtime.stateful-attention", reason);
    return err && yvex_error_is_set(err) ? yvex_error_code(err) : status;
}

typedef struct {
    unsigned long long query_width, query_stride, kv_width, state_width;
    unsigned long long values, workspace_values, projection_bytes;
    unsigned long long position_bytes, staged_bytes;
    unsigned long long history_tokens, history_view_bytes;
} stateful_geometry;

static int stateful_tensor_view(const yvex_device_tensor *source,
                                unsigned long long offset,
                                unsigned long long bytes,
                                yvex_device_tensor *view)
{
    if (!source || !view || offset > source->bytes ||
        bytes > source->bytes - offset || bytes % sizeof(float))
        return 0;
    *view = *source;
    view->rank = 1u;
    memset(view->dims, 0, sizeof(view->dims));
    view->dims[0] = bytes / sizeof(float);
    view->bytes = bytes;
    view->data += offset;
    return 1;
}

static int stateful_tensor_exact(
    const yvex_runtime_stateful_attention_request *request,
    const yvex_device_tensor *tensor, unsigned long long bytes, int written)
{
    return yvex_backend_tensor_owned_by(request->backend, tensor) &&
           tensor->dtype == YVEX_DTYPE_F32 && tensor->bytes == bytes &&
           (!written || tensor->is_written);
}

static int stateful_geometry_build(
    const yvex_runtime_stateful_attention_request *request,
    stateful_geometry *geometry, yvex_error *err)
{
    unsigned long long query_elements, output_elements, view_elements;
    unsigned long long query_bytes, output_bytes, projection_pair_bytes;

    if (!geometry)
        return stateful_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "stateful attention geometry storage is required");
    memset(geometry, 0, sizeof(*geometry));
    if (!request || !request->backend || !request->state ||
        !request->residency)
        return stateful_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "stateful attention requires backend, logical state, and physical residency owners");
    if (!request->state->begin || !request->state->stage ||
        !request->state->view || !request->state->summary)
        return stateful_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "stateful attention requires a complete transactional state owner");
    if (!request->layer || !request->query || !request->key ||
        !request->value || !request->output || !request->token_count ||
        !request->host_workspace || !request->host_positions)
        return stateful_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "stateful attention requires one layer and bounded projection storage");
    if (yvex_backend_kind_of(request->backend) != YVEX_BACKEND_KIND_CUDA)
        return stateful_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "stateful attention currently requires the CUDA backend");
    if (!yvex_sha256_hex_valid(request->attention_plan_identity) ||
        !yvex_sha256_hex_valid(request->input_identity))
        return stateful_refuse(
            err, YVEX_ERR_FORMAT,
            "stateful attention identities are incomplete");
    if (request->layer->ordinal != request->layer_ordinal)
        return stateful_refuse(
            err, YVEX_ERR_FORMAT,
            "stateful attention layer ordinal does not match the compiled decoder");
    if (
        !request->layer->query_heads || !request->layer->kv_heads ||
        request->layer->query_heads % request->layer->kv_heads ||
        !request->layer->head_dimension ||
        !yvex_core_u64_mul(request->layer->query_heads,
                           request->layer->head_dimension,
                           &geometry->query_width) ||
        !yvex_core_u64_mul(request->layer->kv_heads,
                           request->layer->head_dimension,
                           &geometry->kv_width) ||
        !yvex_core_u64_mul(geometry->kv_width, 2ull,
                           &geometry->state_width) ||
        yvex_attention_layer_local_state_width(
            request->layer, &view_elements, err) != YVEX_OK ||
        view_elements != geometry->state_width)
        return stateful_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "direct-Q/K/V attention geometry does not match its state recipe");
    geometry->query_stride = request->query_token_stride
                                 ? request->query_token_stride
                                 : geometry->query_width;
    if (geometry->query_stride < geometry->query_width ||
        !yvex_core_u64_mul(request->token_count - 1ull,
                           geometry->query_stride, &query_elements) ||
        !yvex_core_u64_add(query_elements, geometry->query_width,
                           &query_elements) ||
        !yvex_core_u64_mul(request->token_count, geometry->query_width,
                           &output_elements) ||
        !yvex_core_u64_mul(request->token_count, geometry->kv_width,
                           &geometry->values) ||
        !yvex_core_u64_mul(geometry->values, 4ull,
                           &geometry->workspace_values) ||
        !yvex_core_u64_mul(geometry->values, sizeof(float),
                           &geometry->projection_bytes) ||
        !yvex_core_u64_mul(geometry->projection_bytes, 2ull,
                           &projection_pair_bytes) ||
        !yvex_core_u64_mul(request->token_count,
                           sizeof(unsigned long long),
                           &geometry->position_bytes) ||
        !yvex_core_u64_add(projection_pair_bytes, geometry->position_bytes,
                           &geometry->staged_bytes) ||
        !yvex_core_u64_add(request->token_position, request->token_count,
                           &geometry->history_tokens) ||
        !yvex_core_u64_mul(geometry->history_tokens - 1ull,
                           geometry->state_width, &view_elements) ||
        !yvex_core_u64_add(view_elements, geometry->kv_width,
                           &view_elements) ||
        !yvex_core_u64_mul(view_elements, sizeof(float),
                           &geometry->history_view_bytes) ||
        !yvex_core_u64_mul(query_elements, sizeof(float), &query_bytes) ||
        !yvex_core_u64_mul(output_elements, sizeof(float), &output_bytes) ||
        request->host_workspace_values < geometry->workspace_values ||
        request->host_position_capacity < request->token_count ||
        !stateful_tensor_exact(request, request->query, query_bytes, 1) ||
        !stateful_tensor_exact(
            request, request->key, geometry->projection_bytes, 1) ||
        !stateful_tensor_exact(
            request, request->value, geometry->projection_bytes, 1) ||
        !stateful_tensor_exact(request, request->output, output_bytes, 0))
        return stateful_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "bounded F32 projection tensors and reusable publication storage are required");
    return YVEX_OK;
}

static int stateful_identity(
    const yvex_runtime_stateful_attention_request *request,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const yvex_attention_layer_plan *layer = request->layer;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.runtime.stateful-exact-attention.v1") ||
        !yvex_sha256_update_text(&hash, request->attention_plan_identity) ||
        !yvex_sha256_update_text(&hash, request->input_identity) ||
        !yvex_sha256_update_u64(&hash, request->layer_ordinal) ||
        !yvex_sha256_update_u64(&hash, layer->layer_index) ||
        !yvex_sha256_update_u64(&hash, request->token_position) ||
        !yvex_sha256_update_u64(&hash, request->token_count) ||
        !yvex_sha256_update_u64(&hash, layer->query_heads) ||
        !yvex_sha256_update_u64(&hash, layer->kv_heads) ||
        !yvex_sha256_update_u64(&hash, layer->head_dimension) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int stateful_copy_projection(
    const yvex_runtime_stateful_attention_request *request,
    yvex_runtime_state_history_device_view *history,
    const stateful_geometry *geometry, yvex_error *err)
{
    const unsigned long long bytes = geometry->projection_bytes /
                                     request->token_count;
    const unsigned long long state_stride = history->value_width * sizeof(float);
    unsigned long long token;

    for (token = 0ull; token < request->token_count; ++token) {
        yvex_device_tensor source, target;
        unsigned long long source_offset = token * bytes;
        unsigned long long target_offset =
            (history->visible_tokens + token) * state_stride;
        int rc;

        if (!stateful_tensor_view(request->key, source_offset, bytes, &source) ||
            !stateful_tensor_view(
                &history->values, target_offset, bytes, &target))
            return stateful_refuse(
                err, YVEX_ERR_BOUNDS,
                "key projection exceeded its admitted state view");
        rc = yvex_backend_tensor_copy_async(
            request->backend, &target, &source, err);
        if (rc != YVEX_OK) return rc;
        if (!stateful_tensor_view(request->value, source_offset, bytes, &source) ||
            !stateful_tensor_view(
                &history->values, target_offset + bytes, bytes, &target))
            return stateful_refuse(
                err, YVEX_ERR_BOUNDS,
                "value projection exceeded its admitted state view");
        rc = yvex_backend_tensor_copy_async(
            request->backend, &target, &source, err);
        if (rc != YVEX_OK) return rc;
        request->host_positions[token] = request->token_position + token;
    }
    {
        yvex_device_tensor positions = history->positions;
        unsigned long long offset =
            history->visible_tokens * sizeof(unsigned long long);
        int rc;
        if (offset > positions.bytes || geometry->position_bytes >
                                          positions.bytes - offset)
            return stateful_refuse(
                err, YVEX_ERR_BOUNDS,
                "token positions exceeded their admitted state view");
        positions.data += offset;
        positions.bytes = geometry->position_bytes;
        positions.rank = 1u;
        memset(positions.dims, 0, sizeof(positions.dims));
        positions.dims[0] = request->token_count;
        rc = yvex_backend_tensor_write(
            request->backend, &positions, request->host_positions,
            positions.bytes, err);
        if (rc != YVEX_OK) return rc;
    }
    history->values.is_written = 1;
    history->positions.is_written = 1;
    return YVEX_OK;
}

static int stateful_host_delta(
    const yvex_runtime_stateful_attention_request *request,
    const stateful_geometry *geometry, float **raw_kv, yvex_error *err)
{
    float *key = request->host_workspace + 2ull * geometry->values;
    float *value = key + geometry->values;
    unsigned long long token, column;
    int rc;

    rc = yvex_backend_tensor_read(
        request->backend, request->key, key, geometry->projection_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(
            request->backend, request->value, value,
            geometry->projection_bytes, err);
    if (rc != YVEX_OK) return rc;
    *raw_kv = request->host_workspace;
    for (token = 0ull; token < request->token_count; ++token)
        for (column = 0ull; column < geometry->kv_width; ++column) {
            (*raw_kv)[token * geometry->state_width + column] =
                key[token * geometry->kv_width + column];
            (*raw_kv)[token * geometry->state_width +
                      geometry->kv_width + column] =
                value[token * geometry->kv_width + column];
        }
    return YVEX_OK;
}

int yvex_runtime_stateful_attention_execute(
    const yvex_runtime_stateful_attention_request *request,
    yvex_runtime_stateful_attention_result *result,
    yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_backend_transformer_operations *operations;
    const yvex_attention_history_view *candidate = NULL;
    yvex_runtime_state_history_device_view history;
    yvex_transformer_attention_request attention = {0};
    yvex_attention_publication publication = {0};
    yvex_device_tensor key_history, value_history;
    stateful_geometry geometry;
    float *raw_kv = NULL;
    int rc;

    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!result) return stateful_refuse(
        err, YVEX_ERR_INVALID_ARG, "stateful attention result storage is required");
    rc = stateful_geometry_build(request, &geometry, err);
    if (rc != YVEX_OK) return rc;
    operations = yvex_backend_transformer_operations_get(request->backend);
    if (!operations || !operations->attention_execute)
        return stateful_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "backend exact-attention execution is unavailable");
    rc = request->state->begin(
        request->state->context, request->layer_ordinal, request->layer, NULL,
        request->token_position, request->token_count,
        request->cancellation.requested ? &request->cancellation : NULL,
        &candidate, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_state_residency_transition(
            request->residency, request->state, NULL, request->layer_ordinal,
            request->token_count, YVEX_RUNTIME_STATE_BEGIN, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_state_residency_candidate_history(
            request->residency, request->layer_ordinal,
            YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY, &history, err);
    if (rc == YVEX_OK &&
        (!candidate || candidate->token_count != request->token_position ||
         candidate->local_tail_count != request->token_position ||
         history.value_width != geometry.state_width ||
         history.visible_tokens != request->token_position ||
         history.admitted_tokens < geometry.history_tokens))
        rc = stateful_refuse(
            err, YVEX_ERR_STATE,
            "transactional K/V history does not match the requested position");
    if (rc == YVEX_OK)
        rc = stateful_copy_projection(request, &history, &geometry, err);
    if (rc == YVEX_OK)
        rc = stateful_host_delta(request, &geometry, &raw_kv, err);
    if (rc == YVEX_OK) {
        if (!stateful_tensor_view(
                &history.values, 0ull, geometry.history_view_bytes,
                &key_history) ||
            !stateful_tensor_view(
                &history.values, geometry.kv_width * sizeof(float),
                geometry.history_view_bytes, &value_history))
            rc = stateful_refuse(
                err, YVEX_ERR_BOUNDS,
                "admitted K/V history cannot expose the requested prefix");
    }
    if (rc == YVEX_OK) {
        attention.requirement = (yvex_transformer_attention_requirement){
            .query_tokens = request->token_count,
            .key_value_tokens = geometry.history_tokens,
            .query_start = request->token_position,
            .query_heads = request->layer->query_heads,
            .key_value_heads = request->layer->kv_heads,
            .head_dimension = request->layer->head_dimension,
            .query_token_stride = geometry.query_stride,
            .key_token_stride = geometry.state_width,
            .value_token_stride = geometry.state_width,
            .query_dtype = YVEX_DTYPE_F32, .key_dtype = YVEX_DTYPE_F32,
            .value_dtype = YVEX_DTYPE_F32, .output_dtype = YVEX_DTYPE_F32,
            .layout = YVEX_TRANSFORMER_ATTENTION_LAYOUT_TOKEN_HEAD_DIM,
            .mask = YVEX_TRANSFORMER_ATTENTION_MASK_CAUSAL,
            .numeric_contract = YVEX_TRANSFORMER_ATTENTION_NUMERIC_EXACT_F32,
            .deterministic = 1};
        attention.query = request->query;
        attention.key = &key_history;
        attention.value = &value_history;
        attention.output = request->output;
        rc = operations->attention_execute(
            request->backend, &attention, &result->attention, err);
    }
    if (rc == YVEX_OK && !stateful_identity(request, result->execution_identity))
        rc = stateful_refuse(
            err, YVEX_ERR_STATE,
            "stateful exact-attention identity derivation failed");
    if (rc == YVEX_OK) {
        publication.complete = 1;
        publication.device_state_staged = 1;
        publication.layer_index = request->layer->layer_index;
        publication.attention_class = request->layer->attention_class;
        publication.token_position = request->token_position;
        publication.token_count = request->token_count;
        publication.kv_width = geometry.state_width;
        publication.raw_kv = raw_kv;
        publication.device_state_staged_bytes = geometry.staged_bytes;
        yvex_runtime_identity_copy(
            publication.execution_identity, result->execution_identity);
        rc = request->state->stage(
            request->state->context, &publication,
            request->cancellation.requested ? &request->cancellation : NULL,
            result->state_delta_identity, failure, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_state_residency_transition(
            request->residency, request->state, &publication,
            request->layer_ordinal, 0ull, YVEX_RUNTIME_STATE_STAGE, err);
    if (rc != YVEX_OK) return rc;
    result->schema_version = YVEX_STATEFUL_ATTENTION_RESULT_SCHEMA_V1;
    result->completed = 1;
    result->layer_ordinal = request->layer_ordinal;
    result->token_position = request->token_position;
    result->token_count = request->token_count;
    result->history_tokens = geometry.history_tokens;
    result->state_staged_bytes = publication.device_state_staged_bytes;
    result->h2d_bytes = geometry.position_bytes;
    result->d2h_bytes = geometry.projection_bytes * 2ull;
    result->d2d_bytes = geometry.projection_bytes * 2ull;
    yvex_error_clear(err);
    return YVEX_OK;
}
