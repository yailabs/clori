/* Execute one admitted gated-delta sequence mixer over transactional device state. */
#include "src/backend/cuda/private.h"
#include "src/backend/cuda/transformer_ops.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

enum {
    GATED_DELTA_CONVOLUTION_BLOCK = 256u,
    GATED_DELTA_RECURRENCE_BLOCK = 128u,
    GATED_DELTA_TOKEN_CHUNK = 64u,
    GATED_DELTA_WORK_ALIGNMENT = 256u
};

typedef struct {
    unsigned long long qkv_values, output_values, head_values;
    unsigned long long convolution_weight_values;
    unsigned long long matrix_updates, accumulated_values;
    unsigned long long activation_bytes, weight_bytes, state_bytes;
    unsigned long long chunk_tokens, chunk_values, workspace_bytes;
} gated_delta_geometry;

static int gated_delta_refuse(yvex_error *err, yvex_status status,
                              const char *where, const char *reason)
{
    yvex_error_set(err, status, where, reason);
    return status;
}

static int gated_delta_workspace_add(unsigned long long *cursor,
                                     unsigned long long bytes)
{
    unsigned long long aligned;
    if (!cursor || !bytes ||
        !yvex_core_u64_add(*cursor, GATED_DELTA_WORK_ALIGNMENT - 1ull, &aligned))
        return 0;
    aligned &= ~(GATED_DELTA_WORK_ALIGNMENT - 1ull);
    return yvex_core_u64_add(aligned, bytes, cursor);
}

static int gated_delta_geometry_build(const yvex_gated_delta_plan *plan,
                                      unsigned long long token_count,
                                      gated_delta_geometry *geometry)
{
    const yvex_gated_delta_requirement *requirement;
    unsigned long long parameter_values, state_bank_bytes, twice_value_heads;
    if (!plan || !geometry || !token_count) return 0;
    memset(geometry, 0, sizeof(*geometry));
    requirement = &plan->requirement;
    geometry->chunk_tokens = token_count < GATED_DELTA_TOKEN_CHUNK
                                 ? token_count : GATED_DELTA_TOKEN_CHUNK;
    if (!yvex_core_u64_mul(token_count, plan->qkv_width,
                           &geometry->qkv_values) ||
        !yvex_core_u64_mul(token_count, plan->value_width,
                           &geometry->output_values) ||
        !yvex_core_u64_mul(token_count, requirement->value_heads,
                           &geometry->head_values) ||
        !yvex_core_u64_mul(plan->qkv_width, requirement->convolution_kernel,
                           &geometry->convolution_weight_values) ||
        !yvex_core_u64_mul(token_count, requirement->value_heads,
                           &geometry->matrix_updates) ||
        !yvex_core_u64_mul(geometry->matrix_updates,
                           requirement->key_head_dimension,
                           &geometry->accumulated_values) ||
        !yvex_core_u64_mul(geometry->accumulated_values,
                           requirement->value_head_dimension,
                           &geometry->accumulated_values) ||
        !yvex_core_u64_mul(geometry->chunk_tokens, plan->qkv_width,
                           &geometry->chunk_values) ||
        !yvex_core_u64_mul(geometry->chunk_values, sizeof(float),
                           &geometry->workspace_bytes) ||
        !gated_delta_workspace_add(&geometry->workspace_bytes, sizeof(int)) ||
        !yvex_core_u64_add(geometry->qkv_values, geometry->output_values,
                           &parameter_values) ||
        !yvex_core_u64_add(parameter_values, geometry->output_values,
                           &parameter_values) ||
        !yvex_core_u64_add(parameter_values, geometry->head_values,
                           &parameter_values) ||
        !yvex_core_u64_add(parameter_values, geometry->head_values,
                           &parameter_values) ||
        !yvex_core_u64_mul(parameter_values, sizeof(float),
                           &geometry->activation_bytes) ||
        !yvex_core_u64_mul(2ull, requirement->value_heads,
                           &twice_value_heads) ||
        !yvex_core_u64_add(geometry->convolution_weight_values,
                           twice_value_heads,
                           &parameter_values) ||
        !yvex_core_u64_add(parameter_values,
                           requirement->value_head_dimension,
                           &parameter_values) ||
        !yvex_core_u64_mul(parameter_values, sizeof(float),
                           &geometry->weight_bytes) ||
        !yvex_core_u64_add(plan->convolution_state_bytes,
                           plan->recurrent_state_bytes, &state_bank_bytes) ||
        !yvex_core_u64_mul(state_bank_bytes, 2ull, &geometry->state_bytes))
        return 0;
    return geometry->workspace_bytes <= SIZE_MAX &&
           geometry->chunk_values <= SIZE_MAX / sizeof(float);
}

static int gated_delta_tensor(const yvex_backend *backend,
                              const yvex_device_tensor *tensor,
                              unsigned long long elements, int written)
{
    return backend_tensor_owner_is(backend, tensor) &&
           tensor->dtype == YVEX_DTYPE_F32 &&
           (!written || tensor->is_written) &&
           backend_tensor_f32_elements(tensor, elements);
}

static int gated_delta_ranges_overlap(const yvex_device_tensor *first,
                                      const yvex_device_tensor *second)
{
    uintptr_t first_start, second_start;
    if (!first || !second || !first->data || !second->data) return 0;
    first_start = (uintptr_t)first->data;
    second_start = (uintptr_t)second->data;
    if (first->bytes > UINTPTR_MAX - first_start ||
        second->bytes > UINTPTR_MAX - second_start) return 1;
    return first_start < second_start + (uintptr_t)second->bytes &&
           second_start < first_start + (uintptr_t)first->bytes;
}

static int gated_delta_mutable_ranges_valid(
    const yvex_gated_delta_device_request *request)
{
    const yvex_device_tensor *read_only[] = {
        request->projected_qkv, request->projected_output_gate,
        request->projected_beta, request->projected_decay,
        request->convolution_weight, request->decay_log, request->time_bias,
        request->normalization_weight, request->convolution_state,
        request->recurrent_state};
    yvex_device_tensor *mutable[] = {
        request->next_convolution_state, request->next_recurrent_state,
        request->output};
    size_t left, right;
    for (left = 0u; left < sizeof(mutable) / sizeof(mutable[0]); ++left) {
        for (right = left + 1u; right < sizeof(mutable) / sizeof(mutable[0]); ++right)
            if (gated_delta_ranges_overlap(mutable[left], mutable[right])) return 0;
        for (right = 0u; right < sizeof(read_only) / sizeof(read_only[0]); ++right)
            if (gated_delta_ranges_overlap(mutable[left], read_only[right])) return 0;
    }
    return 1;
}

static int gated_delta_request_validate(
    yvex_backend *backend, const yvex_gated_delta_plan *plan,
    const yvex_gated_delta_device_request *request,
    const gated_delta_geometry *geometry, yvex_error *err)
{
    const yvex_gated_delta_requirement *r = &plan->requirement;
    if (!backend || !request || !geometry || request->token_count == 0ull ||
        request->token_count != geometry->qkv_values / plan->qkv_width ||
        r->key_head_dimension > GATED_DELTA_RECURRENCE_BLOCK ||
        r->value_head_dimension > GATED_DELTA_RECURRENCE_BLOCK ||
        plan->qkv_width >
            (unsigned long long)UINT_MAX * GATED_DELTA_CONVOLUTION_BLOCK ||
        r->value_heads > UINT_MAX ||
        !gated_delta_tensor(backend, request->projected_qkv,
                            geometry->qkv_values, 1) ||
        !gated_delta_tensor(backend, request->projected_output_gate,
                            geometry->output_values, 1) ||
        !gated_delta_tensor(backend, request->projected_beta,
                            geometry->head_values, 1) ||
        !gated_delta_tensor(backend, request->projected_decay,
                            geometry->head_values, 1) ||
        !gated_delta_tensor(backend, request->convolution_weight,
                            geometry->convolution_weight_values, 1) ||
        !gated_delta_tensor(backend, request->decay_log, r->value_heads, 1) ||
        !gated_delta_tensor(backend, request->time_bias, r->value_heads, 1) ||
        !gated_delta_tensor(backend, request->normalization_weight,
                            r->value_head_dimension, 1) ||
        !gated_delta_tensor(backend, request->next_convolution_state,
                            plan->convolution_state_values, 0) ||
        !gated_delta_tensor(backend, request->next_recurrent_state,
                            plan->recurrent_state_values, 0) ||
        !gated_delta_tensor(backend, request->output,
                            geometry->output_values, 0) ||
        ((request->convolution_state == NULL) !=
         (request->recurrent_state == NULL)) ||
        (request->convolution_state &&
         (!gated_delta_tensor(backend, request->convolution_state,
                              plan->convolution_state_values, 1) ||
          !gated_delta_tensor(backend, request->recurrent_state,
                              plan->recurrent_state_values, 1))) ||
        !gated_delta_mutable_ranges_valid(request))
        return gated_delta_refuse(
            err, YVEX_ERR_FORMAT, "cuda.sequence-mixer.gated-delta",
            "bounded non-overlapping F32 projections, parameters, and state are required");
    return YVEX_OK;
}

int yvex_cuda_gated_delta_workspace_required(
    const yvex_gated_delta_plan *plan, unsigned long long token_count,
    unsigned long long *bytes, yvex_error *err)
{
    gated_delta_geometry geometry;
    int rc;
    if (bytes) *bytes = 0ull;
    if (!bytes || !token_count)
        return gated_delta_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.sequence-mixer.gated-delta.workspace",
            "a non-empty gated-delta workspace query is required");
    rc = yvex_gated_delta_plan_validate(plan, err);
    if (rc != YVEX_OK) return rc;
    if (plan->requirement.key_head_dimension > GATED_DELTA_RECURRENCE_BLOCK ||
        plan->requirement.value_head_dimension > GATED_DELTA_RECURRENCE_BLOCK)
        return gated_delta_refuse(
            err, YVEX_ERR_UNSUPPORTED, "cuda.sequence-mixer.gated-delta.workspace",
            "CUDA gated-delta currently admits head dimensions through 128");
    if (!gated_delta_geometry_build(plan, token_count, &geometry))
        return gated_delta_refuse(
            err, YVEX_ERR_BOUNDS, "cuda.sequence-mixer.gated-delta.workspace",
            "gated-delta workspace geometry overflowed");
    *bytes = geometry.workspace_bytes;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int gated_delta_launch_chunk(
    yvex_backend *backend, yvex_cuda_backend_state *state,
    const yvex_gated_delta_plan *plan,
    const yvex_gated_delta_device_request *request,
    CUdeviceptr activated, CUdeviceptr status,
    unsigned long long token_offset, unsigned long long chunk_tokens,
    int initialize, int has_committed, yvex_error *err)
{
    const yvex_gated_delta_requirement *r = &plan->requirement;
    CUdeviceptr projected_qkv = yvex_cuda_tensor_ptr(request->projected_qkv);
    CUdeviceptr projected_output_gate =
        yvex_cuda_tensor_ptr(request->projected_output_gate);
    CUdeviceptr projected_beta = yvex_cuda_tensor_ptr(request->projected_beta);
    CUdeviceptr projected_decay = yvex_cuda_tensor_ptr(request->projected_decay);
    CUdeviceptr convolution_weight =
        yvex_cuda_tensor_ptr(request->convolution_weight);
    CUdeviceptr decay_log = yvex_cuda_tensor_ptr(request->decay_log);
    CUdeviceptr time_bias = yvex_cuda_tensor_ptr(request->time_bias);
    CUdeviceptr normalization_weight =
        yvex_cuda_tensor_ptr(request->normalization_weight);
    CUdeviceptr convolution_state = initialize && has_committed
        ? yvex_cuda_tensor_ptr(request->convolution_state)
        : yvex_cuda_tensor_ptr(request->next_convolution_state);
    CUdeviceptr recurrent_state = initialize && has_committed
        ? yvex_cuda_tensor_ptr(request->recurrent_state)
        : yvex_cuda_tensor_ptr(request->next_recurrent_state);
    CUdeviceptr next_convolution =
        yvex_cuda_tensor_ptr(request->next_convolution_state);
    CUdeviceptr next_recurrent =
        yvex_cuda_tensor_ptr(request->next_recurrent_state);
    CUdeviceptr output = yvex_cuda_tensor_ptr(request->output);
    float qk_epsilon = (float)r->qk_normalization_epsilon;
    float output_epsilon = (float)r->output_normalization_epsilon;
    float query_scale = (float)r->query_scale;
    int normalization_one_plus =
        r->output_normalization_weight_convention ==
        YVEX_NORMALIZATION_WEIGHT_ONE_PLUS;
    unsigned long long qkv_width = plan->qkv_width;
    unsigned long long convolution_kernel = r->convolution_kernel;
    unsigned long long query_heads = r->query_heads;
    unsigned long long key_heads = r->key_heads;
    unsigned long long value_heads = r->value_heads;
    unsigned long long key_dimension = r->key_head_dimension;
    unsigned long long value_dimension = r->value_head_dimension;
    unsigned long long query_width = plan->query_width;
    unsigned long long key_width = plan->key_width;
    unsigned long long value_width = plan->value_width;
    unsigned int convolution_grid = (unsigned int)(
        (plan->qkv_width + GATED_DELTA_CONVOLUTION_BLOCK - 1ull) /
        GATED_DELTA_CONVOLUTION_BLOCK);
    int rc;
    {
        void *parameters[] = {
            &projected_qkv, &convolution_weight, &convolution_state,
            &next_convolution, &activated, &token_offset, &chunk_tokens,
            &qkv_width, &convolution_kernel, &initialize,
            &has_committed, &status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->gated_delta_convolution_function, convolution_grid,
            GATED_DELTA_CONVOLUTION_BLOCK, 0u, parameters,
            "cuda.sequence-mixer.gated-delta.convolution", err);
    }
    if (rc == YVEX_OK) {
        void *parameters[] = {
            &activated, &projected_output_gate, &projected_beta,
            &projected_decay, &decay_log, &time_bias, &normalization_weight,
            &recurrent_state, &next_recurrent, &output, &token_offset,
            &chunk_tokens, &query_heads, &key_heads, &value_heads,
            &key_dimension, &value_dimension, &query_width,
            &key_width, &value_width, &qk_epsilon,
            &output_epsilon, &query_scale, &normalization_one_plus,
            &initialize, &has_committed, &status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->gated_delta_recurrence_function,
            (unsigned int)r->value_heads, GATED_DELTA_RECURRENCE_BLOCK, 0u,
            parameters, "cuda.sequence-mixer.gated-delta.recurrence", err);
    }
    return rc;
}

static void gated_delta_publish_facts(
    const gated_delta_geometry *geometry,
    const yvex_gated_delta_device_request *request,
    const yvex_gated_delta_device_result *result,
    unsigned long long workspace_bytes, unsigned long long queue_syncs,
    unsigned long long device_syncs, yvex_backend_operation_facts *facts)
{
    facts->d2h_bytes = result->execution_chunks * sizeof(int);
    facts->download_count = result->execution_chunks;
    facts->kernel_launches = result->execution_chunks * 2ull;
    facts->queue_synchronizations = queue_syncs;
    facts->device_synchronizations = device_syncs;
    facts->active_weight_bytes = geometry->weight_bytes;
    facts->state_bytes = request->convolution_state
                             ? geometry->state_bytes
                             : geometry->state_bytes / 2ull;
    facts->activation_bytes = geometry->activation_bytes;
    facts->temporary_bytes = workspace_bytes;
    facts->compulsory_memory_facts_available = 1;
}

int yvex_cuda_gated_delta_execute(
    yvex_backend *backend, const yvex_gated_delta_plan *plan,
    const yvex_gated_delta_device_request *request,
    yvex_gated_delta_device_result *result,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    gated_delta_geometry geometry;
    yvex_cuda_work work = {0};
    yvex_error cleanup;
    CUdeviceptr activated = 0ull, status = 0ull;
    unsigned long long offset = 0ull, workspace_bytes = 0ull;
    unsigned long long queue_syncs = 0ull, device_syncs = 0ull;
    int host_status = 0, cleanup_rc, rc, has_committed;
    if (result) memset(result, 0, sizeof(*result));
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !result || !facts)
        return gated_delta_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.sequence-mixer.gated-delta",
            "backend, semantic result, and operation facts are required");
    rc = yvex_gated_delta_plan_validate(plan, err);
    if (rc != YVEX_OK) return rc;
    if (!request || !gated_delta_geometry_build(
                        plan, request->token_count, &geometry))
        return gated_delta_refuse(
            err, YVEX_ERR_BOUNDS, "cuda.sequence-mixer.gated-delta",
            "gated-delta execution geometry overflowed");
    rc = gated_delta_request_validate(backend, plan, request, &geometry, err);
    if (rc != YVEX_OK) return rc;
    request->next_convolution_state->is_written = 0;
    request->next_recurrent_state->is_written = 0;
    request->output->is_written = 0;
    if (request->cancel_requested &&
        request->cancel_requested(request->cancel_context)) {
        result->cancelled = 1;
        return gated_delta_refuse(
            err, YVEX_ERR_CANCELLED, "cuda.sequence-mixer.gated-delta",
            "gated-delta execution cancelled before candidate-state mutation");
    }
    backend_workspace_reset(backend);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    rc = yvex_cuda_work_allocate(
        &work, &activated, (size_t)(geometry.chunk_values * sizeof(float)),
        NULL, 0, "cuda.sequence-mixer.gated-delta.workspace", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            &work, &status, sizeof(host_status), NULL, 1,
            "cuda.sequence-mixer.gated-delta.status", NULL, err);
    workspace_bytes = backend->workspace_device_tensor
                          ? backend->workspace_cursor : work.peak_bytes;
    has_committed = request->convolution_state != NULL;
    while (rc == YVEX_OK && offset < request->token_count) {
        unsigned long long chunk = request->token_count - offset;
        int device_wide = 0;
        if (request->cancel_requested &&
            request->cancel_requested(request->cancel_context)) {
            result->cancelled = 1;
            rc = YVEX_ERR_CANCELLED;
            yvex_error_set(err, rc, "cuda.sequence-mixer.gated-delta",
                           "gated-delta execution cancelled before candidate publication");
            break;
        }
        if (chunk > geometry.chunk_tokens) chunk = geometry.chunk_tokens;
        rc = gated_delta_launch_chunk(
            backend, state, plan, request, activated, status, offset, chunk,
            offset == 0ull, has_committed, err);
        if (rc == YVEX_OK)
            rc = yvex_cuda_launch_synchronize(
                backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &device_wide,
                "cuda.sequence-mixer.gated-delta.sync", err);
        if (rc == YVEX_OK) {
            if (device_wide) device_syncs++;
            else queue_syncs++;
            rc = yvex_cuda_status(
                &state->driver,
                state->driver.cuMemcpyDtoH_v2(
                    &host_status, status, sizeof(host_status)),
                "cuda.sequence-mixer.gated-delta.status", err);
        }
        if (rc == YVEX_OK && host_status)
            rc = gated_delta_refuse(
                err, YVEX_ERR_FORMAT, "cuda.sequence-mixer.gated-delta.status",
                "gated-delta CUDA recurrence produced invalid numerics");
        if (rc == YVEX_OK) {
            offset += chunk;
            result->token_count += chunk;
            result->execution_chunks++;
        }
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    backend_workspace_reset(backend);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc == YVEX_OK) {
        result->output_values = geometry.output_values;
        result->convolution_state_values = plan->convolution_state_values;
        result->recurrent_state_values = plan->recurrent_state_values;
        result->recurrent_matrix_updates = geometry.matrix_updates;
        result->accumulated_values = geometry.accumulated_values;
        result->complete = 1;
        request->next_convolution_state->is_written = 1;
        request->next_recurrent_state->is_written = 1;
        request->output->is_written = 1;
        gated_delta_publish_facts(
            &geometry, request, result, workspace_bytes,
            queue_syncs, device_syncs, facts);
        yvex_error_clear(err);
    }
    return rc;
}
