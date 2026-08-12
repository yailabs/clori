#include "src/backend/cuda/private.h"
#include <yvex/internal/transformer.h>
#include <yvex/quant.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

enum { TRANSFORMER_BLOCK = 128u };

static int status_transaction_open(yvex_backend *backend, yvex_cuda_work *work,
                                      int begin, const char *stage, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    unsigned long long address = 0ull;
    int acquired, rc;
    if (!backend || !state || !work || (begin != 0 && begin != 1)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "CUDA status transaction owner is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    *work = (yvex_cuda_work){.backend = backend, .state = state,
                             .variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED};
    if (!begin && state->deferred_status) {
        work->status = state->deferred_status;
        work->status_deferred = 1;
        return YVEX_OK;
    }
    acquired = begin && backend->workspace_device_tensor
        ? yvex_backend_workspace_acquire(backend, sizeof(int), 256ull, &address)
        : YVEX_BACKEND_RESIDENT_MISS;
    if (acquired == YVEX_BACKEND_RESIDENT_HIT) {
        work->status = (CUdeviceptr)address;
        rc = yvex_cuda_work_initialize(work, work->status, sizeof(int), NULL, 1, stage, err);
        if (rc == YVEX_OK) {
            state->deferred_status = work->status;
            work->status_deferred = 1;
        }
        return rc;
    }
    if (begin && backend->workspace_device_tensor) {
        yvex_error_set(err, acquired == YVEX_BACKEND_RESIDENT_INVALID
                                ? YVEX_ERR_BOUNDS : YVEX_ERR_NOMEM,
                       stage, "CUDA status transaction exceeds the sealed workspace");
        return yvex_error_code(err);
    }
    return yvex_cuda_work_allocate(
        work, &work->status, sizeof(int), NULL, 1, stage, NULL, err);
}
static int status_transaction_close(yvex_cuda_work *work, int wait, int complete, int rc,
                                       yvex_backend_cuda_operation_facts *facts,
                                       const char *stage, yvex_error *err)
{
    yvex_cuda_backend_state *state;
    yvex_error cleanup;
    int host_status = 0, cleanup_rc, waited = 0, device_wide = 0;
    if (!work || !work->backend || !facts ||
        (wait != 0 && wait != 1) || (complete != 0 && complete != 1)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "CUDA status transaction completion is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    state = work->state;
    if (rc == YVEX_OK && wait) {
        rc = yvex_cuda_launch_synchronize(
            work->backend, work->variant, &device_wide, stage, err);
        waited = rc == YVEX_OK;
    }
    if (rc == YVEX_OK && wait)
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuMemcpyDtoH_v2(
                                  &host_status, work->status, sizeof(host_status)),
                              stage, err);
    if ((complete || rc != YVEX_OK) && work->status_deferred)
        state->deferred_status = 0ull;
    if (rc == YVEX_OK && host_status) {
        yvex_error_set(err, YVEX_ERR_FORMAT, stage,
                       "CUDA status transaction reported invalid numerics");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    facts->temporary_bytes = sizeof(host_status);
    if (waited) {
        facts->d2h_bytes += sizeof(host_status);
        facts->download_count++;
        if (device_wide) facts->device_synchronizations++;
        else facts->stream_synchronizations++;
    }
    return rc;
}
static int cuda_transformer_refuse(yvex_error *err, yvex_status status,
                                   const char *where, const char *reason)
{
    yvex_error_set(err, status, where, reason);
    return status;
}
/*
 * Decode selected encoded embedding rows and initialize repeated mHC streams on CUDA.
 *
 * Writes device embedding and expanded tensors; allocates only transaction scratch. Ownership,
 * launch, copy, status, sync, or cleanup refusal leaves output unadmitted. Transformer embedding
 * initialization only; no tokenizer or host numerical fallback.
 */
int yvex_backend_transformer_cuda_initial(
    yvex_backend *backend, const yvex_device_tensor *encoded, unsigned int qtype,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, yvex_device_tensor *embedding,
    yvex_device_tensor *expanded, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    const yvex_gguf_qtype_geometry *geometry = yvex_gguf_qtype_geometry_find(qtype);
    yvex_cuda_work work;
    CUstream stream = yvex_cuda_launch_stream(backend);
    CUdeviceptr encoded_ptr, embedding_ptr, expanded_ptr;
    unsigned long long count, expanded_count, encoded_required, token, residual_stream;
    size_t embedding_bytes, expanded_bytes, row_bytes;
    unsigned long long activation_bytes;
    unsigned int grid;
    int rc;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !geometry || !geometry->block_size || !geometry->bytes_per_block || !facts ||
        !encoded || !embedding || !expanded || !token_count || !hidden_width ||
        !residual_streams || !yvex_core_u64_mul(token_count, hidden_width, &count) ||
        !yvex_core_u64_mul(count, residual_streams, &expanded_count) ||
        !yvex_core_u64_mul(count / geometry->block_size,
                           geometry->bytes_per_block, &encoded_required) ||
        !yvex_cuda_work_checked_bytes(count, sizeof(float), &embedding_bytes) ||
        !yvex_cuda_work_checked_bytes(expanded_count, sizeof(float), &expanded_bytes) ||
        !yvex_cuda_work_checked_bytes(hidden_width, sizeof(float), &row_bytes) ||
        !yvex_core_u64_add((unsigned long long)embedding_bytes,
                           (unsigned long long)expanded_bytes, &activation_bytes) ||
        !backend_tensor_owner_is(backend, encoded) ||
        count % geometry->block_size ||
        encoded->bytes < encoded_required ||
        !backend_tensor_owner_is(backend, embedding) || embedding->dtype != YVEX_DTYPE_F32 ||
        embedding->bytes < embedding_bytes ||
        !backend_tensor_owner_is(backend, expanded) || expanded->dtype != YVEX_DTYPE_F32 ||
        expanded->bytes < expanded_bytes ||
        count > UINT_MAX * (unsigned long long)TRANSFORMER_BLOCK)
        return cuda_transformer_refuse(err, YVEX_ERR_FORMAT, "cuda.transformer.initial",
                                       "CUDA transformer embedding geometry is incompatible");
    rc = status_transaction_open(
        backend, &work, 1, "cuda.transformer.initial.status", err);
    encoded_ptr = (CUdeviceptr)encoded->data;
    embedding_ptr = (CUdeviceptr)embedding->data;
    expanded_ptr = (CUdeviceptr)expanded->data;
    grid = (unsigned int)((count + TRANSFORMER_BLOCK - 1ull) / TRANSFORMER_BLOCK);
    if (rc == YVEX_OK) {
        void *params[] = {&encoded_ptr, &count, &qtype, &embedding_ptr, &work.status};
        rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                              state->encoded_row_decode_function, grid, TRANSFORMER_BLOCK,
                              0u, params, "cuda.transformer.embedding", err);
    }
    for (token = 0ull; rc == YVEX_OK && token < token_count; ++token)
        for (residual_stream = 0ull;
             rc == YVEX_OK && residual_stream < residual_streams; ++residual_stream) {
            CUdeviceptr source = embedding_ptr + token * hidden_width * sizeof(float);
            CUdeviceptr target = expanded_ptr +
                (token * residual_streams + residual_stream) * hidden_width * sizeof(float);
            CUresult copied = stream && state->driver.cuMemcpyDtoDAsync_v2
                                  ? state->driver.cuMemcpyDtoDAsync_v2(
                                        target, source, row_bytes, stream)
                                  : !stream ? state->driver.cuMemcpyDtoD_v2(
                                        target, source, row_bytes) : (CUresult)1;
            rc = yvex_cuda_status(&state->driver, copied,
                                  "cuda.transformer.initial.repeat", err);
        }
    rc = status_transaction_close(
        &work, !work.status_deferred, 0, rc, facts,
        "cuda.transformer.initial.status", err);
    if (rc == YVEX_OK) {
        embedding->is_written = 1;
        expanded->is_written = 1;
        facts->d2d_bytes = expanded_bytes;
        facts->kernel_launches = 1ull;
        facts->active_weight_bytes = encoded_required;
        facts->activation_bytes = activation_bytes;
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}

/*
 * Collapse target-feature residual streams on CUDA and publish compact and optional resident rows.
 *
 * The caller supplies reusable device storage and may request compact host evidence. Bounded
 * status and resident rows become visible together; expanded input never leaves the device.
 */
int yvex_backend_transformer_cuda_feature_mean(
    yvex_backend *backend, const yvex_device_tensor *expanded,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, yvex_device_tensor *device_output,
    yvex_device_tensor *resident_output, unsigned long long resident_row_offset,
    unsigned long long resident_row_stride, unsigned long long resident_column_offset,
    float *host_output, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work;
    CUdeviceptr input_ptr, output_ptr, resident_ptr = 0ull;
    unsigned long long input_count, output_count, activation_count, resident_rows, resident_elements;
    size_t output_bytes, activation_bytes;
    unsigned int grid;
    int rc;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !state->transformer_feature_mean_function || !facts ||
        !token_count || !hidden_width || !residual_streams ||
        !yvex_core_u64_mul(token_count, hidden_width, &output_count) ||
        !yvex_core_u64_mul(output_count, residual_streams, &input_count) ||
        !yvex_core_u64_add(input_count, output_count, &activation_count) ||
        (resident_output &&
         (!yvex_core_u64_add(activation_count, output_count, &activation_count) ||
          !resident_row_stride || resident_column_offset > resident_row_stride ||
          hidden_width > resident_row_stride - resident_column_offset ||
          !yvex_core_u64_add(resident_row_offset, token_count, &resident_rows) ||
          !yvex_core_u64_mul(resident_rows, resident_row_stride, &resident_elements) ||
          resident_elements > ULLONG_MAX / sizeof(float) ||
          !backend_tensor_owner_is(backend, resident_output) ||
          resident_output->dtype != YVEX_DTYPE_F32 ||
          resident_output->data == expanded->data || resident_output->data == device_output->data ||
          resident_output->bytes < resident_elements * sizeof(float))) ||
        (!resident_output &&
         (resident_row_offset || resident_row_stride || resident_column_offset)) ||
        !yvex_cuda_work_checked_bytes(output_count, sizeof(float), &output_bytes) ||
        !yvex_cuda_work_checked_bytes(activation_count, sizeof(float), &activation_bytes) ||
        output_count > UINT_MAX * (unsigned long long)TRANSFORMER_BLOCK ||
        !backend_tensor_f32_elements(expanded, input_count) ||
        !backend_tensor_f32_elements(device_output, output_count))
        return cuda_transformer_refuse(
            err, YVEX_ERR_FORMAT, "cuda.transformer.feature-mean",
            "CUDA transformer feature geometry is incompatible");
    device_output->is_written = 0;
    if (resident_output) resident_output->is_written = 0;
    rc = status_transaction_open(
        backend, &work, 0, "cuda.transformer.feature-mean.status", err);
    input_ptr = (CUdeviceptr)expanded->data;
    output_ptr = (CUdeviceptr)device_output->data;
    if (resident_output) resident_ptr = (CUdeviceptr)resident_output->data;
    grid = (unsigned int)((output_count + TRANSFORMER_BLOCK - 1ull) /
                          TRANSFORMER_BLOCK);
    if (rc == YVEX_OK) {
        void *params[] = {
            &input_ptr, &token_count, &residual_streams, &hidden_width,
            &output_ptr, &resident_ptr, &resident_row_offset,
            &resident_row_stride, &resident_column_offset, &work.status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->transformer_feature_mean_function, grid, TRANSFORMER_BLOCK,
            0u, params, "cuda.transformer.feature-mean", err);
    }
    rc = status_transaction_close(
        &work, !work.status_deferred || host_output, 0, rc, facts,
        "cuda.transformer.feature-mean.status", err);
    if (rc == YVEX_OK && host_output)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemcpyDtoH_v2(host_output, output_ptr, output_bytes),
            "cuda.transformer.feature-mean.output", err);
    if (rc == YVEX_OK) {
        device_output->is_written = 1;
        if (resident_output) resident_output->is_written = 1;
        facts->d2h_bytes += host_output ? output_bytes : 0u;
        facts->kernel_launches = 1ull;
        facts->download_count += host_output != NULL;
        facts->activation_bytes = activation_bytes;
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_backend_transformer_cuda_final(
    yvex_backend *backend, const yvex_device_tensor *expanded,
    const yvex_device_tensor *function, const yvex_device_tensor *base,
    const yvex_device_tensor *scale, const yvex_device_tensor *norm,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, double epsilon, double mhc_epsilon,
    yvex_device_tensor *pre_normalized, yvex_device_tensor *output,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work;
    CUdeviceptr input_ptr, function_ptr, base_ptr, scale_ptr, norm_ptr;
    CUdeviceptr pre_output_ptr = 0ull, output_ptr;
    unsigned long long expanded_width, expanded_count, function_count, output_count;
    unsigned long long weight_count, activation_count;
    size_t weight_bytes, activation_bytes;
    int rc;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !facts || !token_count || !hidden_width || !residual_streams ||
        !yvex_core_u64_mul(hidden_width, residual_streams, &expanded_width) ||
        !yvex_core_u64_mul(token_count, expanded_width, &expanded_count) ||
        !yvex_core_u64_mul(residual_streams, expanded_width, &function_count) ||
        !yvex_core_u64_mul(token_count, hidden_width, &output_count) ||
        !yvex_core_u64_add(function_count, residual_streams, &weight_count) ||
        !yvex_core_u64_add(weight_count, 1ull, &weight_count) ||
        !yvex_core_u64_add(weight_count, hidden_width, &weight_count) ||
        !yvex_core_u64_add(expanded_count, output_count, &activation_count) ||
        (pre_normalized &&
         !yvex_core_u64_add(activation_count, output_count, &activation_count)) ||
        !yvex_cuda_work_checked_bytes(weight_count, sizeof(float), &weight_bytes) ||
        !yvex_cuda_work_checked_bytes(activation_count, sizeof(float), &activation_bytes) ||
        token_count > UINT_MAX || !backend_tensor_f32_elements(expanded, expanded_count) ||
        !backend_tensor_f32_elements(function, function_count) ||
        !backend_tensor_f32_elements(base, residual_streams) ||
        !backend_tensor_f32_elements(scale, 1ull) ||
        !backend_tensor_f32_elements(norm, hidden_width) ||
        !backend_tensor_f32_elements(output, output_count) ||
        (pre_normalized &&
         (!backend_tensor_f32_elements(pre_normalized, output_count) ||
          pre_normalized->data == output->data)))
        return cuda_transformer_refuse(err, YVEX_ERR_FORMAT, "cuda.transformer.final",
                                       "CUDA transformer final geometry is incompatible");
    rc = status_transaction_open(
        backend, &work, 0, "cuda.transformer.final.status", err);
    input_ptr = (CUdeviceptr)expanded->data; function_ptr = (CUdeviceptr)function->data;
    base_ptr = (CUdeviceptr)base->data; scale_ptr = (CUdeviceptr)scale->data;
    norm_ptr = (CUdeviceptr)norm->data;
    if (pre_normalized) pre_output_ptr = (CUdeviceptr)pre_normalized->data;
    output_ptr = (CUdeviceptr)output->data;
    if (rc == YVEX_OK) {
        void *params[] = {&input_ptr, &function_ptr, &base_ptr, &scale_ptr, &norm_ptr,
                          &token_count, &residual_streams, &hidden_width, &epsilon,
                          &mhc_epsilon, &pre_output_ptr, &output_ptr, &work.status};
        rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->transformer_final_function, (unsigned int)token_count,
            TRANSFORMER_BLOCK, TRANSFORMER_BLOCK * (unsigned int)sizeof(double),
            params, "cuda.transformer.final", err);
    }
    rc = status_transaction_close(
        &work, 1, 1, rc, facts, "cuda.transformer.final.status", err);
    if (rc == YVEX_OK) {
        if (pre_normalized) pre_normalized->is_written = 1;
        output->is_written = 1;
        facts->kernel_launches = 1ull;
        facts->active_weight_bytes = weight_bytes;
        facts->activation_bytes = activation_bytes;
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}


static int transformer_tensor(const yvex_backend *backend, const yvex_device_tensor *tensor,
                              unsigned long long elements, int require_written)
{
    return backend_tensor_owner_is(backend, tensor) &&
           (!require_written || tensor->is_written) && tensor->dtype == YVEX_DTYPE_F32 &&
           backend_tensor_f32_elements(tensor, elements);
}

static int transformer_launch(yvex_backend *backend, CUfunction function,
                              unsigned int grid, unsigned int shared_bytes,
                              void **parameters, const char *stage,
                              yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    int rc;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!backend || !function || !grid || !parameters || !facts) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "complete CUDA transformer launch facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                          function, grid, TRANSFORMER_BLOCK, shared_bytes,
                          parameters, stage, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, stage, err);
    if (rc == YVEX_OK) {
        facts->kernel_launches = 1ull;
        facts->device_synchronizations = 1ull;
        facts->compulsory_memory_facts_available = 1;
    }
    return rc;
}

int yvex_cuda_transformer_rotary_half(
    yvex_backend *backend, yvex_device_tensor *values,
    const yvex_device_tensor *cosines, const yvex_device_tensor *sines,
    unsigned long long tokens, unsigned long long heads, unsigned long long head_dim,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    unsigned long long vectors, elements, table_elements, tasks;
    CUdeviceptr value_ptr, cosine_ptr, sine_ptr;
    unsigned int grid;
    int rc;
    if (!state || !tokens || !heads || !head_dim || (head_dim & 1ull) ||
        !yvex_core_u64_mul(tokens, heads, &vectors) ||
        !yvex_core_u64_mul(vectors, head_dim, &elements) ||
        !yvex_core_u64_mul(tokens, head_dim, &table_elements) ||
        !yvex_core_u64_mul(vectors, head_dim / 2ull, &tasks) ||
        !transformer_tensor(backend, values, elements, 1) ||
        !transformer_tensor(backend, cosines, table_elements, 1) ||
        !transformer_tensor(backend, sines, table_elements, 1) ||
        tasks > (unsigned long long)UINT_MAX * TRANSFORMER_BLOCK) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.rotary",
                       "packed F32 vectors and explicit rotary tables are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)((tasks + TRANSFORMER_BLOCK - 1ull) / TRANSFORMER_BLOCK);
    value_ptr = yvex_cuda_tensor_ptr(values);
    cosine_ptr = yvex_cuda_tensor_ptr(cosines);
    sine_ptr = yvex_cuda_tensor_ptr(sines);
    {
        void *parameters[] = {&value_ptr, &cosine_ptr, &sine_ptr, &tokens, &heads, &head_dim};
        rc = transformer_launch(backend, state->rotary_half_function, grid, 0u,
                                parameters, "cuda.transformer.rotary", facts, err);
    }
    return rc;
}

int yvex_cuda_transformer_gqa(
    yvex_backend *backend, const yvex_device_tensor *query,
    const yvex_device_tensor *key, const yvex_device_tensor *value,
    yvex_device_tensor *output, unsigned long long tokens,
    unsigned long long query_heads, unsigned long long kv_heads,
    unsigned long long head_dim, yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    unsigned long long query_elements, kv_elements, rows;
    CUdeviceptr query_ptr, key_ptr, value_ptr, output_ptr;
    float scale;
    int rc;
    if (!state || !tokens || !query_heads || !kv_heads || query_heads % kv_heads ||
        !head_dim || head_dim > TRANSFORMER_BLOCK ||
        !yvex_core_u64_mul(tokens, query_heads, &rows) ||
        !yvex_core_u64_mul(rows, head_dim, &query_elements) ||
        !yvex_core_u64_mul(tokens, kv_heads, &kv_elements) ||
        !yvex_core_u64_mul(kv_elements, head_dim, &kv_elements) || rows > UINT_MAX ||
        !transformer_tensor(backend, query, query_elements, 1) ||
        !transformer_tensor(backend, key, kv_elements, 1) ||
        !transformer_tensor(backend, value, kv_elements, 1) ||
        !transformer_tensor(backend, output, query_elements, 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.gqa",
                       "bounded packed Q/K/V geometry is required");
        return YVEX_ERR_FORMAT;
    }
    query_ptr = yvex_cuda_tensor_ptr(query);
    key_ptr = yvex_cuda_tensor_ptr(key);
    value_ptr = yvex_cuda_tensor_ptr(value);
    output_ptr = yvex_cuda_tensor_ptr(output);
    scale = 1.0f / sqrtf((float)head_dim);
    {
        void *parameters[] = {&query_ptr, &key_ptr, &value_ptr, &output_ptr,
                              &tokens, &query_heads, &kv_heads, &head_dim, &scale};
        rc = transformer_launch(
            backend, state->gqa_causal_function, (unsigned int)rows,
            (TRANSFORMER_BLOCK + 2u) * sizeof(float), parameters,
            "cuda.transformer.gqa", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_silu_product_bf16(
    yvex_backend *backend, const yvex_device_tensor *gate,
    const yvex_device_tensor *up, yvex_device_tensor *output,
    unsigned long long count, yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr gate_ptr, up_ptr, output_ptr;
    unsigned long long tasks;
    unsigned int grid;
    int rc;
    if (!state || !count || !transformer_tensor(backend, gate, count, 1) ||
        !transformer_tensor(backend, up, count, 1) ||
        !transformer_tensor(backend, output, count, 0) ||
        !yvex_core_u64_add(count, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.silu-product-bf16",
                       "equal F32 storage with BF16 gate policy is required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    gate_ptr = yvex_cuda_tensor_ptr(gate);
    up_ptr = yvex_cuda_tensor_ptr(up);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *parameters[] = {&gate_ptr, &up_ptr, &output_ptr, &count};
        rc = transformer_launch(
            backend, state->silu_product_function, grid, 0u, parameters,
            "cuda.transformer.silu-product-bf16", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_bf16_round(
    yvex_backend *backend, yvex_device_tensor *values, unsigned long long count,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work;
    CUdeviceptr values_ptr;
    unsigned long long tasks;
    unsigned int grid;
    int rc;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !facts || !count || !transformer_tensor(backend, values, count, 1) ||
        !yvex_core_u64_add(count, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.bf16-round",
                       "one finite bounded F32 activation is required");
        return YVEX_ERR_FORMAT;
    }
    rc = status_transaction_open(
        backend, &work, 1, "cuda.transformer.bf16-round.status", err);
    values_ptr = yvex_cuda_tensor_ptr(values);
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    if (rc == YVEX_OK) {
        void *parameters[] = {&values_ptr, &count, &work.status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->attention_bf16_round_function, grid, TRANSFORMER_BLOCK, 0u,
            parameters, "cuda.transformer.bf16-round", err);
    }
    rc = status_transaction_close(
        &work, 1, 1, rc, facts, "cuda.transformer.bf16-round.status", err);
    if (rc == YVEX_OK) {
        facts->kernel_launches = 1ull;
        facts->compulsory_memory_facts_available = 1;
    }
    return rc;
}
int yvex_cuda_transformer_rms_norm_bf16(
    yvex_backend *backend, const yvex_device_tensor *input,
    const yvex_device_tensor *weight, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width, float epsilon,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    unsigned long long elements;
    CUdeviceptr input_ptr, weight_ptr, output_ptr;
    int rc;
    if (!state || !rows || !width || rows > UINT_MAX || !isfinite(epsilon) || epsilon <= 0.0f ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !transformer_tensor(backend, input, elements, 1) ||
        !transformer_tensor(backend, weight, width, 1) ||
        !transformer_tensor(backend, output, elements, 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.rms-norm-bf16",
                       "packed F32 storage with exact BF16 normalization policy is required");
        return YVEX_ERR_FORMAT;
    }
    input_ptr = yvex_cuda_tensor_ptr(input);
    weight_ptr = yvex_cuda_tensor_ptr(weight);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *parameters[] = {
            &input_ptr, &weight_ptr, &output_ptr, &width, &rows, &epsilon,
        };
        rc = transformer_launch(
            backend, state->rms_norm_bf16_policy_function, (unsigned int)rows,
            TRANSFORMER_BLOCK * sizeof(float), parameters,
            "cuda.transformer.rms-norm-bf16", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}
