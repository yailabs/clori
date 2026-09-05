#include "src/backend/cuda/private.h"
#include "src/backend/cuda/transformer_ops.h"
#include <yvex/internal/component.h>
#include <yvex/quant.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    TRANSFORMER_BLOCK = 128u,
    GQA_HEAD_DIMENSION_MAX = 256u,
    GQA_QUERIES_PER_BLOCK = 4u,
    GQA_BLAS_BLOCK = 256u,
    GQA_WORK_ALIGNMENT = 256u,
    /* One in-place F32 score tile keeps the canonical 768p workspace below the superseded
       pack-and-two-score-buffer path without fragmenting execution into 64-row submissions. */
    GQA_BLAS_QUERY_CHUNK = 512u
};
#define GQA_BLAS_OP_N 0
#define GQA_BLAS_OP_T 1
#define GQA_BLAS_R_32F 0
#define GQA_BLAS_R_16BF 14
#define GQA_BLAS_COMPUTE_32F 68
#define GQA_BLAS_DEFAULT -1

static int status_storage_ensure(yvex_backend *backend, yvex_cuda_backend_state *state,
                                 const char *stage, yvex_error *err)
{
    int rc;
    if (state->transformer_status) return YVEX_OK;
    rc = yvex_backend_memory_can_add(backend, sizeof(int), "CUDA", stage, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuMemAlloc_v2(
                                  &state->transformer_status, sizeof(int)),
                              stage, err);
    if (rc == YVEX_OK) backend_memory_acquire(backend, sizeof(int));
    return rc;
}

static int status_transaction_open(yvex_backend *backend, yvex_cuda_work *work,
                                      int begin, const char *stage, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc;
    if (!backend || !state || !work || (begin != 0 && begin != 1)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "CUDA status transaction owner is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    *work = (yvex_cuda_work){.backend = backend, .state = state,
                             .variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED};
    if (!begin && state->status_transaction_active) {
        work->status = state->transformer_status;
        work->status_deferred = 1;
        return YVEX_OK;
    }
    if (begin && backend->workspace_device_tensor) {
        rc = status_storage_ensure(backend, state, stage, err);
        if (rc != YVEX_OK) return rc;
        work->status = state->transformer_status;
        rc = yvex_cuda_work_initialize(work, work->status, sizeof(int), NULL, 1, stage, err);
        state->status_transaction_active = rc == YVEX_OK;
        work->status_deferred = rc == YVEX_OK;
        return rc;
    }
    return yvex_cuda_work_allocate(
        work, &work->status, sizeof(int), NULL, 1, stage, NULL, err);
}
static int status_transaction_close(yvex_cuda_work *work, int wait, int complete, int rc,
                                       yvex_backend_operation_facts *facts,
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
        state->status_transaction_active = 0;
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
        else facts->queue_synchronizations++;
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
int yvex_cuda_transformer_initial(
    yvex_backend *backend, const yvex_device_tensor *encoded, unsigned int qtype,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, yvex_device_tensor *embedding,
    yvex_device_tensor *expanded, yvex_backend_operation_facts *facts,
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
int yvex_cuda_transformer_feature_mean(
    yvex_backend *backend, const yvex_device_tensor *expanded,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, yvex_device_tensor *device_output,
    yvex_device_tensor *resident_output, unsigned long long resident_row_offset,
    unsigned long long resident_row_stride, unsigned long long resident_column_offset,
    float *host_output, yvex_backend_operation_facts *facts,
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

int yvex_cuda_transformer_final(
    yvex_backend *backend, const yvex_device_tensor *expanded,
    const yvex_device_tensor *function, const yvex_device_tensor *base,
    const yvex_device_tensor *scale, const yvex_device_tensor *norm,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, double epsilon, double mhc_epsilon,
    yvex_device_tensor *pre_normalized, yvex_device_tensor *output,
    yvex_backend_operation_facts *facts, yvex_error *err)
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
                              yvex_backend_operation_facts *facts, yvex_error *err)
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

static int transformer_rotary_half(
    yvex_backend *backend, yvex_device_tensor *values,
    const yvex_device_tensor *cosines, const yvex_device_tensor *sines,
    unsigned long long tokens, unsigned long long heads, unsigned long long head_dim,
    unsigned long long rotary_dim, CUfunction function, const char *stage,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    unsigned long long vectors, elements, table_elements, tasks;
    CUdeviceptr value_ptr, cosine_ptr, sine_ptr;
    unsigned int grid;
    int rc;
    if (!function || !stage || !tokens || !heads || !head_dim || !rotary_dim ||
        rotary_dim > head_dim || (rotary_dim & 1ull) ||
        !yvex_core_u64_mul(tokens, heads, &vectors) ||
        !yvex_core_u64_mul(vectors, head_dim, &elements) ||
        !yvex_core_u64_mul(tokens, rotary_dim, &table_elements) ||
        !yvex_core_u64_mul(vectors, rotary_dim / 2ull, &tasks) ||
        !transformer_tensor(backend, values, elements, 1) ||
        !transformer_tensor(backend, cosines, table_elements, 1) ||
        !transformer_tensor(backend, sines, table_elements, 1) ||
        tasks > (unsigned long long)UINT_MAX * TRANSFORMER_BLOCK) {
        yvex_error_set(err, YVEX_ERR_FORMAT, stage,
                       "packed F32 vectors and explicit rotary tables are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)((tasks + TRANSFORMER_BLOCK - 1ull) / TRANSFORMER_BLOCK);
    value_ptr = yvex_cuda_tensor_ptr(values);
    cosine_ptr = yvex_cuda_tensor_ptr(cosines);
    sine_ptr = yvex_cuda_tensor_ptr(sines);
    {
        void *parameters[] = {
            &value_ptr, &cosine_ptr, &sine_ptr, &tokens, &heads, &head_dim, &rotary_dim,
        };
        rc = transformer_launch(backend, function, grid, 0u,
                                parameters, stage, facts, err);
    }
    return rc;
}

int yvex_cuda_transformer_rotary_half(
    yvex_backend *backend, yvex_device_tensor *values,
    const yvex_device_tensor *cosines, const yvex_device_tensor *sines,
    unsigned long long tokens, unsigned long long heads, unsigned long long head_dim,
    unsigned long long rotary_dim, yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    return transformer_rotary_half(
        backend, values, cosines, sines, tokens, heads, head_dim, rotary_dim,
        state ? state->rotary_half_function : NULL, "cuda.transformer.rotary-bf16", facts, err);
}

int yvex_cuda_transformer_rotary_half_f32(
    yvex_backend *backend, yvex_device_tensor *values,
    const yvex_device_tensor *cosines, const yvex_device_tensor *sines,
    unsigned long long tokens, unsigned long long heads, unsigned long long head_dim,
    unsigned long long rotary_dim, yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    return transformer_rotary_half(
        backend, values, cosines, sines, tokens, heads, head_dim, rotary_dim,
        state ? state->rotary_half_plain_function : NULL, "cuda.transformer.rotary-f32", facts, err);
}

static int gqa_enqueue(yvex_backend *backend, CUfunction function, unsigned int grid,
                       unsigned int block, unsigned int shared_bytes, void **parameters,
                       const char *stage, yvex_error *err)
{
    if (!function || !grid || !block || !parameters) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "complete tiled-attention launch facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    return yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                            function, grid, block, shared_bytes, parameters, stage, err);
}

typedef struct {
    yvex_backend *backend;
    yvex_cuda_backend_state *state;
    CUdeviceptr query, key, value, output, scores, status;
    unsigned long long query_tokens, key_value_tokens, query_start;
    unsigned long long heads, head_dim;
    int causal;
} gqa_tile_context;
static int gqa_tile_execute(gqa_tile_context *chunk, unsigned long long local_query_start,
                            unsigned long long query_rows, yvex_error *err)
{
    unsigned long long score_stride, matrix_stride, query_offset, softmax_rows;
    unsigned long long absolute_query_start;
    CUdeviceptr query_chunk, output_chunk;
    const float scale = 1.0f / sqrtf((float)chunk->head_dim), zero = 0.0f, one = 1.0f;
    int blas_status, rc = YVEX_OK;
    if (!yvex_core_u64_mul(query_rows, chunk->key_value_tokens, &score_stride) ||
        !yvex_core_u64_mul(chunk->heads, chunk->head_dim, &matrix_stride) ||
        !yvex_core_u64_mul(local_query_start, matrix_stride, &query_offset) ||
        !yvex_core_u64_mul(query_offset, sizeof(float), &query_offset) ||
        !yvex_core_u64_mul(chunk->heads, query_rows, &softmax_rows) ||
        !yvex_core_u64_add(chunk->query_start, local_query_start,
                           &absolute_query_start) ||
        score_stride > LLONG_MAX || matrix_stride > INT_MAX || softmax_rows > UINT_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.transformer.gqa.blas",
                       "tiled attention launch geometry overflowed");
        return YVEX_ERR_BOUNDS;
    }
    query_chunk = chunk->query + query_offset;
    output_chunk = chunk->output + query_offset;
    blas_status = chunk->state->blas.gemm_strided_batched_ex(
        chunk->state->blas.handle, GQA_BLAS_OP_T, GQA_BLAS_OP_N,
        (int)chunk->key_value_tokens, (int)query_rows, (int)chunk->head_dim, &scale,
        (const void *)(uintptr_t)chunk->key, GQA_BLAS_R_32F,
        (int)matrix_stride, (long long)chunk->head_dim,
        (const void *)(uintptr_t)query_chunk, GQA_BLAS_R_32F, (int)matrix_stride,
        (long long)chunk->head_dim, &zero,
        (void *)(uintptr_t)chunk->scores, GQA_BLAS_R_32F,
        (int)chunk->key_value_tokens,
        (long long)score_stride, (int)chunk->heads, GQA_BLAS_COMPUTE_32F,
        GQA_BLAS_DEFAULT);
    if (blas_status != 0) {
        yvex_error_setf(err, YVEX_ERR_BACKEND, "cuda.transformer.gqa.qk",
                        "batched F32 Q/K projection failed with status %d", blas_status);
        rc = YVEX_ERR_BACKEND;
    }
    if (rc == YVEX_OK) {
        CUfunction function = chunk->key_value_tokens <= 1024ull
                                  ? chunk->state->gqa_softmax_warp_function
                                  : chunk->state->gqa_softmax_function;
        unsigned int block = chunk->key_value_tokens <= 1024ull
                                 ? 32u : GQA_BLAS_BLOCK;
        unsigned int shared = chunk->key_value_tokens <= 1024ull
                                  ? 0u : GQA_BLAS_BLOCK * sizeof(float);
        void *parameters[] = {
            &chunk->scores, &chunk->scores, &query_rows,
            &chunk->key_value_tokens,
            &absolute_query_start, &chunk->causal, &chunk->status,
        };
        rc = gqa_enqueue(chunk->backend, function, (unsigned int)softmax_rows, block, shared,
                         parameters, "cuda.transformer.gqa.softmax", err);
    }
    if (rc == YVEX_OK) {
        blas_status = chunk->state->blas.gemm_strided_batched_ex(
            chunk->state->blas.handle, GQA_BLAS_OP_N, GQA_BLAS_OP_N,
            (int)chunk->head_dim, (int)query_rows,
            (int)chunk->key_value_tokens, &one,
            (const void *)(uintptr_t)chunk->value, GQA_BLAS_R_32F,
            (int)matrix_stride, (long long)chunk->head_dim,
            (const void *)(uintptr_t)chunk->scores, GQA_BLAS_R_32F,
            (int)chunk->key_value_tokens, (long long)score_stride, &zero,
            (void *)(uintptr_t)output_chunk, GQA_BLAS_R_32F, (int)matrix_stride,
            (long long)chunk->head_dim, (int)chunk->heads,
            GQA_BLAS_COMPUTE_32F, GQA_BLAS_DEFAULT);
        if (blas_status != 0) {
            yvex_error_setf(err, YVEX_ERR_BACKEND, "cuda.transformer.gqa.pv",
                            "batched F32 probability/value projection failed with status %d",
                            blas_status);
            rc = YVEX_ERR_BACKEND;
        }
    }
    return rc;
}

static int gqa_tiled_execute(
    yvex_backend *backend, const yvex_device_tensor *query,
    const yvex_device_tensor *key, const yvex_device_tensor *value,
    yvex_device_tensor *output, unsigned long long query_tokens,
    unsigned long long key_value_tokens, unsigned long long query_start,
    unsigned long long heads, unsigned long long head_dim, int causal,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    CUdeviceptr scores = 0ull, status = 0ull;
    unsigned long long score_elements, output_elements, validation_tasks;
    unsigned long long chunks = 0ull, query_tile;
    unsigned long long score_bytes;
    unsigned long long temporary_bytes = 0ull;
    unsigned long long local_query_start;
    int host_status = 0, rc = YVEX_OK, cleanup_rc;
    yvex_error cleanup;
    query_tile = query_tokens < GQA_BLAS_QUERY_CHUNK
                     ? query_tokens : GQA_BLAS_QUERY_CHUNK;
    if (!state || !state->blas.ready || !state->blas.gemm_strided_batched_ex ||
        !state->gqa_softmax_function || !state->gqa_softmax_warp_function ||
        !state->attention_validate_function || query_tokens > INT_MAX ||
        key_value_tokens > INT_MAX ||
        heads > INT_MAX || head_dim > INT_MAX ||
        !yvex_core_u64_mul(heads, query_tile, &score_elements) ||
        !yvex_core_u64_mul(score_elements, key_value_tokens, &score_elements) ||
        !yvex_core_u64_mul(score_elements, sizeof(float), &score_bytes) ||
        !yvex_core_u64_mul(query_tokens, heads, &output_elements) ||
        !yvex_core_u64_mul(output_elements, head_dim, &output_elements) ||
        !yvex_core_u64_add(output_elements, GQA_BLAS_BLOCK - 1ull, &validation_tasks) ||
        score_elements > LLONG_MAX || score_bytes > SIZE_MAX ||
        validation_tasks / GQA_BLAS_BLOCK > UINT_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.transformer.gqa.blas",
                       "tiled exact attention exceeds admitted integer geometry");
        return YVEX_ERR_BOUNDS;
    }
    backend_workspace_reset(backend);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
#define GQA_ALLOC(pointer, bytes, zeroed, label) \
    do { \
        if (rc == YVEX_OK) \
            rc = yvex_cuda_work_allocate(&work, &(pointer), (size_t)(bytes), NULL, (zeroed), \
                                         "cuda.transformer.gqa.alloc." label, NULL, err); \
    } while (0)
    GQA_ALLOC(scores, score_bytes, 0, "scores");
    GQA_ALLOC(status, sizeof(host_status), 1, "status");
#undef GQA_ALLOC
    if (rc == YVEX_OK)
        rc = yvex_cuda_blas_bind_launch_stream(backend, "cuda.transformer.gqa.stream", err);
    if (rc == YVEX_OK)
        temporary_bytes = backend->workspace_device_tensor
                              ? backend->workspace_cursor
                              : work.peak_bytes;
    {
        gqa_tile_context chunk = {
            backend, state, yvex_cuda_tensor_ptr(query), yvex_cuda_tensor_ptr(key),
            yvex_cuda_tensor_ptr(value), yvex_cuda_tensor_ptr(output), scores, status,
            query_tokens, key_value_tokens, query_start, heads, head_dim, causal};
        for (local_query_start = 0ull;
             rc == YVEX_OK && local_query_start < query_tokens;
             local_query_start += GQA_BLAS_QUERY_CHUNK) {
            unsigned long long query_rows = query_tokens - local_query_start;
            if (query_rows > GQA_BLAS_QUERY_CHUNK) query_rows = GQA_BLAS_QUERY_CHUNK;
            rc = gqa_tile_execute(&chunk, local_query_start, query_rows, err);
            ++chunks;
        }
    }
    if (rc == YVEX_OK) {
        CUdeviceptr output_ptr = yvex_cuda_tensor_ptr(output);
        void *parameters[] = {&output_ptr, &output_elements, &status};
        rc = gqa_enqueue(
            backend, state->attention_validate_function,
            (unsigned int)(validation_tasks / GQA_BLAS_BLOCK), GQA_BLAS_BLOCK, 0u,
            parameters, "cuda.transformer.gqa.validate", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   "cuda.transformer.gqa.sync", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuMemcpyDtoH_v2(&host_status, status,
                                                            sizeof(host_status)),
                              "cuda.transformer.gqa.status", err);
    if (rc == YVEX_OK && host_status) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.gqa.status",
                       "tiled exact attention produced invalid numerics");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    backend_workspace_reset(backend);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc != YVEX_OK) return rc;
    facts->kernel_launches = chunks * 3ull + 1ull;
    facts->download_count = 1ull;
    facts->d2h_bytes = sizeof(host_status);
    facts->temporary_bytes = temporary_bytes;
    facts->accelerated_matrix_launches = chunks * 2ull;
    facts->device_synchronizations = 1ull;
    facts->compulsory_memory_facts_available = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int gqa_workspace_add(unsigned long long *cursor, unsigned long long bytes)
{
    unsigned long long aligned;
    if (!cursor || !bytes ||
        !yvex_core_u64_add(*cursor, GQA_WORK_ALIGNMENT - 1ull, &aligned))
        return 0;
    aligned &= ~(GQA_WORK_ALIGNMENT - 1ull);
    return yvex_core_u64_add(aligned, bytes, cursor);
}

int yvex_cuda_transformer_gqa_workspace_required(
    unsigned long long query_tokens, unsigned long long key_value_tokens,
    unsigned long long query_heads,
    unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long *bytes, yvex_error *err)
{
    unsigned long long score_elements, scores, query_tile;
    unsigned long long total = 0ull;
    if (bytes) *bytes = 0ull;
    if (!bytes || !query_tokens || !key_value_tokens || !query_heads ||
        !kv_heads || !head_dim ||
        query_heads % kv_heads) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.transformer.gqa.workspace",
                       "bounded grouped-query attention geometry is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (query_heads != kv_heads) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    query_tile = query_tokens < GQA_BLAS_QUERY_CHUNK
                     ? query_tokens : GQA_BLAS_QUERY_CHUNK;
    if (!yvex_core_u64_mul(query_heads, query_tile, &score_elements) ||
        !yvex_core_u64_mul(score_elements, key_value_tokens, &score_elements) ||
        !yvex_core_u64_mul(score_elements, sizeof(float), &scores) ||
        !gqa_workspace_add(&total, scores) ||
        !gqa_workspace_add(&total, sizeof(int)) || total > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.transformer.gqa.workspace",
                       "tiled exact-attention workspace geometry overflowed");
        return YVEX_ERR_BOUNDS;
    }
    *bytes = total;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_transformer_gqa_strided(
    yvex_backend *backend, const yvex_device_tensor *query,
    const yvex_device_tensor *key, const yvex_device_tensor *value,
    yvex_device_tensor *output, unsigned long long query_tokens,
    unsigned long long key_value_tokens, unsigned long long query_start,
    unsigned long long query_heads, unsigned long long kv_heads,
    unsigned long long head_dim, unsigned long long query_stride,
    unsigned long long key_stride, unsigned long long value_stride, int causal,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    unsigned long long query_elements, key_elements, value_elements, output_elements;
    unsigned long long query_width, kv_width;
    unsigned long long rows, grid_rows, query_end;
    float scale;
    int rc;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !facts || !query_tokens || !key_value_tokens ||
        !yvex_core_u64_add(query_start, query_tokens, &query_end) ||
        query_end > key_value_tokens || !query_heads || !kv_heads ||
        query_heads % kv_heads ||
        (causal != 0 && causal != 1) ||
        !head_dim || head_dim > GQA_HEAD_DIMENSION_MAX ||
        !yvex_core_u64_mul(query_tokens, query_heads, &rows) ||
        !yvex_core_u64_mul(query_heads, head_dim, &query_width) ||
        !yvex_core_u64_mul(kv_heads, head_dim, &kv_width) ||
        (query_stride && query_stride < query_width) ||
        (key_stride && key_stride < kv_width) ||
        (value_stride && value_stride < kv_width) ||
        !(query_stride = query_stride ? query_stride : query_width) ||
        !(key_stride = key_stride ? key_stride : kv_width) ||
        !(value_stride = value_stride ? value_stride : kv_width) ||
        !yvex_core_u64_mul(query_tokens - 1ull, query_stride, &query_elements) ||
        !yvex_core_u64_add(query_elements, query_width, &query_elements) ||
        !yvex_core_u64_mul(query_tokens, query_width, &output_elements) ||
        !yvex_core_u64_mul(key_value_tokens - 1ull, key_stride, &key_elements) ||
        !yvex_core_u64_add(key_elements, kv_width, &key_elements) ||
        !yvex_core_u64_mul(key_value_tokens - 1ull, value_stride, &value_elements) ||
        !yvex_core_u64_add(value_elements, kv_width, &value_elements) ||
        !yvex_core_u64_mul(query_tokens / GQA_QUERIES_PER_BLOCK +
                               (query_tokens % GQA_QUERIES_PER_BLOCK != 0ull),
                           query_heads, &grid_rows) || grid_rows > UINT_MAX ||
        !transformer_tensor(backend, query, query_elements, 1) ||
        !transformer_tensor(backend, key, key_elements, 1) ||
        !transformer_tensor(backend, value, value_elements, 1) ||
        !transformer_tensor(backend, output, output_elements, 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.gqa",
                       "bounded Q/K/V views matching the declared token strides are required");
        return YVEX_ERR_FORMAT;
    }
    if (query_heads == kv_heads && query_stride == query_width &&
        key_stride == kv_width && value_stride == kv_width && state->blas.ready &&
        state->blas.gemm_strided_batched_ex && state->gqa_softmax_function &&
        state->gqa_softmax_warp_function && state->attention_validate_function &&
        !yvex_cuda_capture_active(backend)) {
        rc = gqa_tiled_execute(
            backend, query, key, value, output, query_tokens,
            key_value_tokens, query_start, query_heads, head_dim,
            causal, facts, err);
        if (rc == YVEX_OK) output->is_written = 1;
        return rc;
    }
    backend_workspace_reset(backend);
    CUdeviceptr query_ptr = yvex_cuda_tensor_ptr(query), key_ptr = yvex_cuda_tensor_ptr(key);
    CUdeviceptr value_ptr = yvex_cuda_tensor_ptr(value), output_ptr = yvex_cuda_tensor_ptr(output);
    scale = 1.0f / sqrtf((float)head_dim);
    {
        void *parameters[] = {
            &query_ptr, &key_ptr, &value_ptr, &output_ptr,
            &query_tokens, &key_value_tokens, &query_start,
            &query_heads, &kv_heads, &head_dim, &query_stride, &key_stride,
            &value_stride, &scale, &causal,
        };
        rc = transformer_launch(
            backend, head_dim > TRANSFORMER_BLOCK ? state->gqa_wide_function
                                                  : state->gqa_function,
            (unsigned int)grid_rows,
            2u * (unsigned int)head_dim * sizeof(float), parameters,
            "cuda.transformer.gqa", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_gqa(
    yvex_backend *backend, const yvex_device_tensor *query,
    const yvex_device_tensor *key, const yvex_device_tensor *value,
    yvex_device_tensor *output, unsigned long long query_tokens,
    unsigned long long key_value_tokens, unsigned long long query_start,
    unsigned long long query_heads, unsigned long long kv_heads,
    unsigned long long head_dim, int causal, yvex_backend_operation_facts *facts,
    yvex_error *err)
{
    return yvex_cuda_transformer_gqa_strided(
        backend, query, key, value, output, query_tokens, key_value_tokens,
        query_start, query_heads, kv_heads, head_dim, 0ull, 0ull, 0ull,
        causal, facts, err);
}

int yvex_cuda_transformer_silu_product_bf16(
    yvex_backend *backend, const yvex_device_tensor *gate,
    const yvex_device_tensor *up, yvex_device_tensor *output,
    unsigned long long count, yvex_backend_operation_facts *facts, yvex_error *err)
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

int yvex_cuda_transformer_silu(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *output, unsigned long long count, int bf16_output,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr, output_ptr;
    unsigned long long tasks;
    unsigned int grid;
    int rc;
    if (!state || !count || (bf16_output != 0 && bf16_output != 1) ||
        !transformer_tensor(backend, input, count, 1) ||
        !transformer_tensor(backend, output, count, 0) ||
        !yvex_core_u64_add(count, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.silu",
                       "bounded F32 activation and explicit output policy are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    input_ptr = yvex_cuda_tensor_ptr(input);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *parameters[] = {&input_ptr, &output_ptr, &count, &bf16_output};
        rc = transformer_launch(backend, state->silu_function, grid, 0u,
                                parameters, "cuda.transformer.silu", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_timestep_embedding(
    yvex_backend *backend, const yvex_device_tensor *timesteps,
    yvex_device_tensor *output, unsigned long long rows,
    unsigned long long half_width, float maximum_period,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr timestep_ptr, output_ptr;
    unsigned long long output_width, output_values, tasks;
    unsigned int grid;
    int rc;
    if (!state || !rows || !half_width || !isfinite(maximum_period) ||
        maximum_period <= 1.0f || !yvex_core_u64_mul(half_width, 2ull, &output_width) ||
        !yvex_core_u64_mul(rows, output_width, &output_values) ||
        !transformer_tensor(backend, timesteps, rows, 1) ||
        !transformer_tensor(backend, output, output_values, 0) ||
        !yvex_core_u64_add(rows * half_width, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.timestep-embedding",
                       "bounded F32 timestep embedding geometry is required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    timestep_ptr = yvex_cuda_tensor_ptr(timesteps);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *parameters[] = {&timestep_ptr, &output_ptr, &rows,
                              &half_width, &maximum_period};
        rc = transformer_launch(
            backend, state->timestep_embedding_function, grid, 0u, parameters,
            "cuda.transformer.timestep-embedding", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_split_three(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *first, yvex_device_tensor *second, yvex_device_tensor *third,
    unsigned long long rows, unsigned long long width,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr, first_ptr, second_ptr, third_ptr;
    unsigned long long output_elements, input_elements, tasks;
    unsigned int grid;
    int rc;
    if (!state || !rows || !width || !yvex_core_u64_mul(rows, width, &output_elements) ||
        !yvex_core_u64_mul(output_elements, 3ull, &input_elements) ||
        !yvex_core_u64_add(output_elements, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX ||
        !transformer_tensor(backend, input, input_elements, 1) ||
        !transformer_tensor(backend, first, output_elements, 0) ||
        !transformer_tensor(backend, second, output_elements, 0) ||
        !transformer_tensor(backend, third, output_elements, 0) ||
        first == second || first == third || second == third || input == first ||
        input == second || input == third) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.split-three",
                       "one packed input and three independent outputs are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    input_ptr = yvex_cuda_tensor_ptr(input);
    first_ptr = yvex_cuda_tensor_ptr(first);
    second_ptr = yvex_cuda_tensor_ptr(second);
    third_ptr = yvex_cuda_tensor_ptr(third);
    {
        void *parameters[] = {
            &input_ptr, &first_ptr, &second_ptr, &third_ptr, &rows, &width,
        };
        rc = transformer_launch(backend, state->split_three_function, grid, 0u,
                                parameters, "cuda.transformer.split-three", facts, err);
    }
    if (rc == YVEX_OK) first->is_written = second->is_written = third->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_split_interleaved_three(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *first, yvex_device_tensor *second, yvex_device_tensor *third,
    unsigned long long rows, unsigned long long heads, unsigned long long head_dim,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr, first_ptr, second_ptr, third_ptr;
    unsigned long long width, output_elements, input_elements, tasks;
    unsigned int grid;
    int rc;
    if (!state || !rows || !heads || !head_dim ||
        !yvex_core_u64_mul(heads, head_dim, &width) ||
        !yvex_core_u64_mul(rows, width, &output_elements) ||
        !yvex_core_u64_mul(output_elements, 3ull, &input_elements) ||
        !yvex_core_u64_add(output_elements, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX ||
        !transformer_tensor(backend, input, input_elements, 1) ||
        !transformer_tensor(backend, first, output_elements, 0) ||
        !transformer_tensor(backend, second, output_elements, 0) ||
        !transformer_tensor(backend, third, output_elements, 0) ||
        first == second || first == third || second == third || input == first ||
        input == second || input == third) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.split-interleaved",
                       "interleaved per-head Q/K/V input and independent outputs are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    input_ptr = yvex_cuda_tensor_ptr(input);
    first_ptr = yvex_cuda_tensor_ptr(first);
    second_ptr = yvex_cuda_tensor_ptr(second);
    third_ptr = yvex_cuda_tensor_ptr(third);
    {
        void *parameters[] = {
            &input_ptr, &first_ptr, &second_ptr, &third_ptr, &rows, &heads, &head_dim,
        };
        rc = transformer_launch(
            backend, state->split_interleaved_function, grid, 0u, parameters,
            "cuda.transformer.split-interleaved", facts, err);
    }
    if (rc == YVEX_OK) first->is_written = second->is_written = third->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_swiglu_split_bf16(
    yvex_backend *backend, const yvex_device_tensor *input, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr, output_ptr;
    unsigned long long output_elements, input_elements, tasks;
    unsigned int grid;
    int rc;
    if (!state || !rows || !width || !yvex_core_u64_mul(rows, width, &output_elements) ||
        !yvex_core_u64_mul(output_elements, 2ull, &input_elements) ||
        !yvex_core_u64_add(output_elements, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX ||
        !transformer_tensor(backend, input, input_elements, 1) ||
        !transformer_tensor(backend, output, output_elements, 0) || input == output) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.swiglu-split-bf16",
                       "packed BF16-policy SwiGLU storage is required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    input_ptr = yvex_cuda_tensor_ptr(input);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *parameters[] = {&input_ptr, &output_ptr, &rows, &width};
        rc = transformer_launch(backend, state->swiglu_split_function, grid, 0u,
                                parameters, "cuda.transformer.swiglu-split-bf16", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_swiglu_split_f32(
    yvex_backend *backend, const yvex_device_tensor *input, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width, int gate_first,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr, output_ptr;
    unsigned long long output_elements, input_elements, tasks;
    unsigned int grid;
    int rc;
    if (!state || !rows || !width || (gate_first != 0 && gate_first != 1) ||
        !yvex_core_u64_mul(rows, width, &output_elements) ||
        !yvex_core_u64_mul(output_elements, 2ull, &input_elements) ||
        !yvex_core_u64_add(output_elements, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX ||
        !transformer_tensor(backend, input, input_elements, 1) ||
        !transformer_tensor(backend, output, output_elements, 0) || input == output) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.swiglu-split-f32",
                       "packed F32 SwiGLU storage and explicit gate order are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    input_ptr = yvex_cuda_tensor_ptr(input);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *parameters[] = {&input_ptr, &output_ptr, &rows, &width, &gate_first};
        rc = transformer_launch(
            backend, state->swiglu_split_f32_function, grid, 0u,
            parameters, "cuda.transformer.swiglu-split-f32", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

static int transformer_indices_validate(const unsigned int *indices,
                                        unsigned long long rows,
                                        unsigned long long table_rows)
{
    unsigned long long row;
    if (!indices || !rows || !table_rows) return 0;
    for (row = 0ull; row < rows; ++row)
        if ((unsigned long long)indices[row] >= table_rows) return 0;
    return 1;
}

int yvex_cuda_transformer_modulate_bf16(
    yvex_backend *backend, const yvex_device_tensor *input,
    const yvex_device_tensor *table, const unsigned int *row_indices,
    yvex_device_tensor *output, unsigned long long rows, unsigned long long width,
    unsigned long long table_rows, unsigned long long parameters,
    unsigned int shift_slot, unsigned int scale_slot,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    CUdeviceptr input_ptr, table_ptr, indices_ptr = 0ull, output_ptr;
    unsigned long long elements, table_elements, index_bytes, tasks;
    unsigned int grid;
    int rc, cleanup_rc;
    yvex_error cleanup;
    if (!state || !rows || !width || !parameters || shift_slot >= parameters ||
        scale_slot >= parameters || !transformer_indices_validate(row_indices, rows, table_rows) ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_mul(table_rows, parameters, &table_elements) ||
        !yvex_core_u64_mul(table_elements, width, &table_elements) ||
        !yvex_core_u64_mul(rows, sizeof(*row_indices), &index_bytes) || index_bytes > SIZE_MAX ||
        !yvex_core_u64_add(elements, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX ||
        !transformer_tensor(backend, input, elements, 1) ||
        !transformer_tensor(backend, table, table_elements, 1) ||
        !transformer_tensor(backend, output, elements, 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.modulate-bf16",
                       "bounded modulation table and row selection are required");
        return YVEX_ERR_FORMAT;
    }
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    rc = yvex_cuda_work_allocate(&work, &indices_ptr, (size_t)index_bytes, row_indices, 0,
                                 "cuda.transformer.modulate-bf16.indices", NULL, err);
    input_ptr = yvex_cuda_tensor_ptr(input);
    table_ptr = yvex_cuda_tensor_ptr(table);
    output_ptr = yvex_cuda_tensor_ptr(output);
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    if (rc == YVEX_OK) {
        void *arguments[] = {
            &input_ptr, &table_ptr, &indices_ptr, &output_ptr, &rows, &width,
            &table_rows, &parameters, &shift_slot, &scale_slot,
        };
        rc = transformer_launch(backend, state->modulation_function, grid, 0u,
                                arguments, "cuda.transformer.modulate-bf16", facts, err);
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    if (rc == YVEX_OK) {
        output->is_written = 1;
        facts->h2d_bytes = index_bytes;
        facts->temporary_bytes = index_bytes;
        facts->upload_count = 1ull;
    }
    return rc;
}

int yvex_cuda_transformer_gated_residual_bf16(
    yvex_backend *backend, const yvex_device_tensor *residual,
    const yvex_device_tensor *table, const unsigned int *row_indices,
    const yvex_device_tensor *update, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width, unsigned long long table_rows,
    unsigned long long parameters, unsigned int gate_slot,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    CUdeviceptr residual_ptr, table_ptr, indices_ptr = 0ull, update_ptr, output_ptr;
    unsigned long long elements, table_elements, index_bytes, tasks;
    unsigned int grid;
    int rc, cleanup_rc;
    yvex_error cleanup;
    if (!state || !rows || !width || !parameters || gate_slot >= parameters ||
        !transformer_indices_validate(row_indices, rows, table_rows) ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_mul(table_rows, parameters, &table_elements) ||
        !yvex_core_u64_mul(table_elements, width, &table_elements) ||
        !yvex_core_u64_mul(rows, sizeof(*row_indices), &index_bytes) || index_bytes > SIZE_MAX ||
        !yvex_core_u64_add(elements, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX ||
        !transformer_tensor(backend, residual, elements, 1) ||
        !transformer_tensor(backend, table, table_elements, 1) ||
        !transformer_tensor(backend, update, elements, 1) ||
        !transformer_tensor(backend, output, elements, 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.gated-residual-bf16",
                       "bounded residual, update, gate table, and row selection are required");
        return YVEX_ERR_FORMAT;
    }
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    rc = yvex_cuda_work_allocate(&work, &indices_ptr, (size_t)index_bytes, row_indices, 0,
                                 "cuda.transformer.gated-residual-bf16.indices", NULL, err);
    residual_ptr = yvex_cuda_tensor_ptr(residual);
    table_ptr = yvex_cuda_tensor_ptr(table);
    update_ptr = yvex_cuda_tensor_ptr(update);
    output_ptr = yvex_cuda_tensor_ptr(output);
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    if (rc == YVEX_OK) {
        void *arguments[] = {
            &residual_ptr, &table_ptr, &indices_ptr, &update_ptr, &output_ptr,
            &rows, &width, &table_rows, &parameters, &gate_slot,
        };
        rc = transformer_launch(backend, state->gated_residual_function, grid, 0u,
                                arguments, "cuda.transformer.gated-residual-bf16", facts, err);
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    if (rc == YVEX_OK) {
        output->is_written = 1;
        facts->h2d_bytes = index_bytes;
        facts->temporary_bytes = index_bytes;
        facts->upload_count = 1ull;
    }
    return rc;
}

int yvex_cuda_transformer_bias(
    yvex_backend *backend, const yvex_device_tensor *input,
    const yvex_device_tensor *bias, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width, int bf16_output,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr, bias_ptr, output_ptr;
    unsigned long long elements, tasks;
    unsigned int grid;
    int rc;
    if (!state || !rows || !width || (bf16_output != 0 && bf16_output != 1) ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_add(elements, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX ||
        !transformer_tensor(backend, input, elements, 1) ||
        !transformer_tensor(backend, bias, width, 1) ||
        !transformer_tensor(backend, output, elements, 0) || bias == output) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.bias",
                       "bounded row-major activation and independent bias are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    input_ptr = yvex_cuda_tensor_ptr(input);
    bias_ptr = yvex_cuda_tensor_ptr(bias);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *arguments[] = {
            &input_ptr, &bias_ptr, &output_ptr, &rows, &width, &bf16_output,
        };
        rc = transformer_launch(backend, state->bias_function, grid, 0u,
                                arguments, "cuda.transformer.bias", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_add_bf16(
    yvex_backend *backend, const yvex_device_tensor *left,
    const yvex_device_tensor *right, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr left_ptr, right_ptr, output_ptr;
    unsigned long long elements, tasks;
    unsigned int grid;
    int rc;
    if (!state || !rows || !width || !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_add(elements, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX ||
        !transformer_tensor(backend, left, elements, 1) ||
        !transformer_tensor(backend, right, elements, 1) ||
        !transformer_tensor(backend, output, elements, 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.add-bf16",
                       "bounded BF16-compatible activation rows are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    left_ptr = yvex_cuda_tensor_ptr(left);
    right_ptr = yvex_cuda_tensor_ptr(right);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *arguments[] = {
            &left_ptr, &right_ptr, &output_ptr, &rows, &width,
        };
        rc = transformer_launch(backend, state->add_bf16_function, grid, 0u,
                                arguments, "cuda.transformer.add-bf16", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_scaled_residual_f32(
    yvex_backend *backend, const yvex_device_tensor *residual,
    const yvex_device_tensor *update, const yvex_device_tensor *scale,
    yvex_device_tensor *output, unsigned long long rows, unsigned long long width,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr residual_ptr, update_ptr, scale_ptr, output_ptr;
    unsigned long long elements, tasks;
    unsigned int grid;
    int rc;
    if (!state || !rows || !width || !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_add(elements, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX ||
        !transformer_tensor(backend, residual, elements, 1) ||
        !transformer_tensor(backend, update, elements, 1) ||
        !transformer_tensor(backend, scale, width, 1) ||
        !transformer_tensor(backend, output, elements, 0) || scale == output) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.scaled-residual-f32",
                       "bounded F32 residual, update, and scale tensors are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    residual_ptr = yvex_cuda_tensor_ptr(residual);
    update_ptr = yvex_cuda_tensor_ptr(update);
    scale_ptr = yvex_cuda_tensor_ptr(scale);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *arguments[] = {
            &residual_ptr, &update_ptr, &scale_ptr, &output_ptr, &rows, &width,
        };
        rc = transformer_launch(
            backend, state->scaled_residual_f32_function, grid, 0u, arguments,
            "cuda.transformer.scaled-residual-f32", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_layer_norm_f32(
    yvex_backend *backend, const yvex_device_tensor *input,
    const yvex_device_tensor *weight, const yvex_device_tensor *bias,
    yvex_device_tensor *output, unsigned long long rows, unsigned long long width,
    float epsilon, yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr, weight_ptr, bias_ptr, output_ptr;
    unsigned long long elements;
    int rc;
    if (!state || !rows || rows > UINT_MAX || !width || epsilon <= 0.0f ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !transformer_tensor(backend, input, elements, 1) ||
        !transformer_tensor(backend, weight, width, 1) ||
        !transformer_tensor(backend, bias, width, 1) ||
        !transformer_tensor(backend, output, elements, 0) ||
        weight == output || bias == output) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.layer-norm-f32",
                       "bounded F32 rows and independent affine vectors are required");
        return YVEX_ERR_FORMAT;
    }
    input_ptr = yvex_cuda_tensor_ptr(input);
    weight_ptr = yvex_cuda_tensor_ptr(weight);
    bias_ptr = yvex_cuda_tensor_ptr(bias);
    output_ptr = yvex_cuda_tensor_ptr(output);
    {
        void *arguments[] = {
            &input_ptr, &weight_ptr, &bias_ptr, &output_ptr, &rows, &width, &epsilon,
        };
        rc = transformer_launch(
            backend, state->layer_norm_f32_function, (unsigned int)rows,
            TRANSFORMER_BLOCK * sizeof(float), arguments,
            "cuda.transformer.layer-norm-f32", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

int yvex_cuda_transformer_bf16_round(
    yvex_backend *backend, yvex_device_tensor *values, unsigned long long count,
    yvex_backend_operation_facts *facts, yvex_error *err)
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
    yvex_backend_operation_facts *facts, yvex_error *err)
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
        rc = transformer_launch(backend, state->rms_norm_bf16_policy_function,
                                (unsigned int)rows, 4u * sizeof(float), parameters,
                                "cuda.transformer.rms-norm-bf16", facts, err);
    }
    if (rc == YVEX_OK) output->is_written = 1;
    return rc;
}

typedef enum {
    DENSE_HIDDEN = 0,
    DENSE_NORM,
    DENSE_VECTOR,
    DENSE_ONES,
    DENSE_QKV,
    DENSE_QUERY,
    DENSE_KEY,
    DENSE_VALUE,
    DENSE_COSINE,
    DENSE_SINE,
    DENSE_ATTENTION,
    DENSE_UPDATE,
    DENSE_FUSED,
    DENSE_GATED,
    DENSE_BIAS,
    DENSE_OUTPUT,
    DENSE_DEVICE_COUNT
} dense_decoder_device_slot;

typedef struct {
    yvex_backend *backend;
    const yvex_transformer_dense_decoder_request *request;
    yvex_device_tensor *device[DENSE_DEVICE_COUNT];
    yvex_backend_operation_facts facts;
    unsigned long long device_bytes;
} dense_decoder_run;

static int dense_refuse(yvex_error *err, yvex_status status,
                        const char *stage, const char *message)
{
    yvex_error_set(err, status, stage, message);
    return status;
}

static int dense_facts_add(dense_decoder_run *run,
                           const yvex_backend_operation_facts *part)
{
    return run && part && part->compulsory_memory_facts_available &&
           yvex_core_u64_add(run->facts.kernel_launches, part->kernel_launches,
                             &run->facts.kernel_launches) &&
           yvex_core_u64_add(run->facts.h2d_bytes, part->h2d_bytes,
                             &run->facts.h2d_bytes) &&
           yvex_core_u64_add(run->facts.d2h_bytes, part->d2h_bytes,
                             &run->facts.d2h_bytes);
}

static int dense_weight_valid(const yvex_transformer_encoded_weight *weight,
                              unsigned long long rows, unsigned long long width)
{
    unsigned long long row_bytes, bytes;
    return weight && weight->encoded && weight->qtype == YVEX_GGUF_QTYPE_F32 &&
           weight->row_count == rows && weight->row_width == width &&
           yvex_core_u64_mul(width, sizeof(float), &row_bytes) &&
           row_bytes == weight->row_bytes &&
           yvex_core_u64_mul(rows, row_bytes, &bytes) && bytes == weight->encoded_bytes;
}

static int dense_request_valid(const yvex_transformer_dense_decoder_request *request)
{
    unsigned long long block, width3, ffn2, output_values;
    if (!request || !request->block_weights || !request->final_norm_weight ||
        !request->final_norm_bias || !request->output_weight || !request->output_bias ||
        !request->hidden || !request->cosines || !request->sines || !request->output ||
        !request->rows || !request->output_rows || request->output_rows > request->rows ||
        !request->width || !request->heads || !request->head_dim ||
        request->heads > ULLONG_MAX / request->head_dim ||
        request->heads * request->head_dim != request->width ||
        !request->rotary_dim || request->rotary_dim > request->head_dim ||
        (request->rotary_dim & 1ull) || !request->ffn_width || !request->block_count ||
        !request->output_width || !isfinite(request->epsilon) || request->epsilon <= 0.0f ||
        !yvex_core_u64_mul(request->width, 3ull, &width3) ||
        !yvex_core_u64_mul(request->ffn_width, 2ull, &ffn2) ||
        !yvex_core_u64_mul(request->output_rows, request->output_width, &output_values) ||
        request->output_capacity < output_values ||
        !dense_weight_valid(request->final_norm_weight, 1ull, request->width) ||
        !dense_weight_valid(request->final_norm_bias, 1ull, request->width) ||
        !dense_weight_valid(request->output_weight, request->output_width, request->width) ||
        !dense_weight_valid(request->output_bias, 1ull, request->output_width)) return 0;
    for (block = 0ull; block < request->block_count; ++block) {
        const yvex_transformer_encoded_weight *weights =
            request->block_weights + block * YVEX_TRANSFORMER_DENSE_DECODER_BLOCK_WEIGHT_COUNT;
        if (!dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_NORM1, 1ull, request->width) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_QKV_WEIGHT, width3,
                                request->width) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_QKV_BIAS, 1ull, width3) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_ATTENTION_WEIGHT,
                                request->width, request->width) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_ATTENTION_BIAS,
                                1ull, request->width) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_SCALE1, 1ull, request->width) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_NORM2, 1ull, request->width) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_FF1_WEIGHT,
                                ffn2, request->width) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_FF1_BIAS, 1ull, ffn2) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_FF2_WEIGHT,
                                request->width, request->ffn_width) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_FF2_BIAS,
                                1ull, request->width) ||
            !dense_weight_valid(weights + YVEX_TRANSFORMER_DENSE_SCALE2,
                                1ull, request->width)) return 0;
    }
    return 1;
}

static int dense_tensor_allocate(dense_decoder_run *run, dense_decoder_device_slot slot,
                                 const char *name, unsigned long long rows,
                                 unsigned long long width, int rank_one, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long elements, bytes, next;
    if (!run || slot >= DENSE_DEVICE_COUNT || !name || !rows || !width ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) ||
        !yvex_core_u64_add(run->device_bytes, bytes, &next))
        return dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.allocate",
                            "dense decoder activation geometry overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = rank_one ? 1u : 2u;
    descriptor.dims[0] = rank_one ? width : rows;
    descriptor.dims[1] = rank_one ? 0ull : width;
    descriptor.bytes = bytes;
    if (yvex_backend_tensor_alloc(run->backend, &descriptor, &run->device[slot], err) != YVEX_OK)
        return yvex_error_code(err);
    run->device_bytes = next;
    return YVEX_OK;
}

static int dense_devices_prepare(dense_decoder_run *run, yvex_error *err)
{
    const yvex_transformer_dense_decoder_request *r = run->request;
    unsigned long long width3, ffn2, bias_width;
    int rc;
    if (!yvex_core_u64_mul(r->width, 3ull, &width3) ||
        !yvex_core_u64_mul(r->ffn_width, 2ull, &ffn2))
        return dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.allocate",
                            "dense decoder workspace geometry overflowed");
    bias_width = width3 > ffn2 ? width3 : ffn2;
    if (r->output_width > bias_width) bias_width = r->output_width;
#define ALLOC(slot, name, rows, width, rank_one) \
    if (rc == YVEX_OK) rc = dense_tensor_allocate(run, slot, name, rows, width, rank_one, err)
    rc = dense_tensor_allocate(run, DENSE_HIDDEN, "dense-hidden", r->rows, r->width, 0, err);
    ALLOC(DENSE_NORM, "dense-norm", r->rows, r->width, 0);
    ALLOC(DENSE_VECTOR, "dense-vector", 1ull, r->width, 1);
    ALLOC(DENSE_ONES, "dense-ones", 1ull, r->head_dim, 1);
    ALLOC(DENSE_QKV, "dense-qkv", r->rows, width3, 0);
    ALLOC(DENSE_QUERY, "dense-query", r->rows * r->heads, r->head_dim, 0);
    ALLOC(DENSE_KEY, "dense-key", r->rows * r->heads, r->head_dim, 0);
    ALLOC(DENSE_VALUE, "dense-value", r->rows * r->heads, r->head_dim, 0);
    ALLOC(DENSE_COSINE, "dense-cosine", r->rows, r->rotary_dim, 0);
    ALLOC(DENSE_SINE, "dense-sine", r->rows, r->rotary_dim, 0);
    ALLOC(DENSE_ATTENTION, "dense-attention", r->rows, r->width, 0);
    ALLOC(DENSE_UPDATE, "dense-update", r->rows, r->width, 0);
    ALLOC(DENSE_FUSED, "dense-fused", r->rows, ffn2, 0);
    ALLOC(DENSE_GATED, "dense-gated", r->rows, r->ffn_width, 0);
    ALLOC(DENSE_BIAS, "dense-bias", 1ull, bias_width, 1);
    ALLOC(DENSE_OUTPUT, "dense-output", r->output_rows, r->output_width, 0);
#undef ALLOC
    return rc;
}

static int dense_devices_release(dense_decoder_run *run, int rc, yvex_error *err)
{
    unsigned int count = DENSE_DEVICE_COUNT;
    while (count) {
        yvex_error cleanup;
        int cleanup_rc;
        --count;
        if (!run->device[count]) continue;
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(
            run->backend, &run->device[count], &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    return rc;
}

static int dense_gather(dense_decoder_run *run,
                        const yvex_transformer_encoded_weight *weight,
                        yvex_device_tensor *output, yvex_error *err)
{
    static const unsigned int row = 0u;
    yvex_backend_operation_facts facts;
    int rc = yvex_backend_encoded_gather(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes,
        &row, 1ull, output, &facts, err);
    if (rc == YVEX_OK && !dense_facts_add(run, &facts))
        rc = dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.facts",
                          "dense decoder gather accounting overflowed");
    return rc;
}

static int dense_project(dense_decoder_run *run,
                         const yvex_transformer_encoded_weight *weight,
                         unsigned long long rows, const yvex_device_tensor *input,
                         yvex_device_tensor *output, yvex_error *err)
{
    yvex_backend_operation_facts facts;
    int rc = yvex_backend_encoded_matvec(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes, rows,
        input, NULL, 0ull, NULL, output, 0, &facts, err);
    if (rc == YVEX_OK && !dense_facts_add(run, &facts))
        rc = dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.facts",
                          "dense decoder projection accounting overflowed");
    return rc;
}

static int dense_primitive(dense_decoder_run *run, int rc,
                           const yvex_backend_operation_facts *facts,
                           yvex_error *err)
{
    if (rc == YVEX_OK && !dense_facts_add(run, facts))
        return dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.facts",
                            "dense decoder primitive accounting overflowed");
    return rc;
}

static int dense_norm(dense_decoder_run *run,
                      const yvex_transformer_encoded_weight *weight,
                      yvex_device_tensor *input, yvex_device_tensor *output,
                      yvex_error *err)
{
    int rc = dense_gather(run, weight, run->device[DENSE_VECTOR], err);
    if (rc == YVEX_OK)
        rc = yvex_backend_op_rms_norm(
            run->backend, input, run->device[DENSE_VECTOR],
            run->request->epsilon, output, err);
    if (rc == YVEX_OK) run->facts.kernel_launches++;
    return rc;
}

static int dense_bias(dense_decoder_run *run,
                      const yvex_transformer_encoded_weight *weight,
                      yvex_device_tensor *values, unsigned long long width,
                      yvex_error *err)
{
    yvex_backend_operation_facts facts;
    yvex_device_tensor bias_view = {0};
    int rc = dense_gather(run, weight, run->device[DENSE_BIAS], err);
    if (rc == YVEX_OK && !yvex_backend_tensor_f32_subview(
            run->device[DENSE_BIAS], 0ull, width, &bias_view))
        rc = dense_refuse(err, YVEX_ERR_FORMAT, "cuda.dense-decoder.bias-input",
                          "dense decoder bias prefix is outside its owned tensor");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(
            run->backend, values, &bias_view, values,
            run->request->rows, width, 0, &facts, err);
    if (rc != YVEX_OK)
        yvex_error_setf(err, (yvex_status)rc, "cuda.dense-decoder.bias",
                        "bias application failed for %s at width %llu",
                        yvex_device_tensor_name(values), width);
    return dense_primitive(run, rc, &facts, err);
}

static int dense_attention(dense_decoder_run *run,
                           const yvex_transformer_encoded_weight *weights,
                           yvex_error *err)
{
    const yvex_transformer_dense_decoder_request *r = run->request;
    yvex_backend_operation_facts facts;
    unsigned long long qkv_width;
    int rc;
    if (!yvex_core_u64_mul(r->width, 3ull, &qkv_width))
        return dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.attention",
                            "dense decoder QKV width overflowed");
    rc = dense_project(run, weights + YVEX_TRANSFORMER_DENSE_QKV_WEIGHT,
                       r->rows, run->device[DENSE_NORM], run->device[DENSE_QKV], err);
    if (rc == YVEX_OK)
        rc = dense_bias(run, weights + YVEX_TRANSFORMER_DENSE_QKV_BIAS,
                        run->device[DENSE_QKV], qkv_width, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_split_interleaved_three(
            run->backend, run->device[DENSE_QKV], run->device[DENSE_QUERY],
            run->device[DENSE_KEY], run->device[DENSE_VALUE], r->rows,
            r->heads, r->head_dim, &facts, err);
    rc = dense_primitive(run, rc, &facts, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_op_rms_norm(
            run->backend, run->device[DENSE_QUERY], run->device[DENSE_ONES],
            r->epsilon, run->device[DENSE_QUERY], err);
    if (rc == YVEX_OK) {
        run->facts.kernel_launches++;
        rc = yvex_backend_op_rms_norm(
            run->backend, run->device[DENSE_KEY], run->device[DENSE_ONES],
            r->epsilon, run->device[DENSE_KEY], err);
    }
    if (rc == YVEX_OK) run->facts.kernel_launches++;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half_f32(
            run->backend, run->device[DENSE_QUERY], run->device[DENSE_COSINE],
            run->device[DENSE_SINE], r->rows, r->heads, r->head_dim,
            r->rotary_dim, &facts, err);
    rc = dense_primitive(run, rc, &facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half_f32(
            run->backend, run->device[DENSE_KEY], run->device[DENSE_COSINE],
            run->device[DENSE_SINE], r->rows, r->heads, r->head_dim,
            r->rotary_dim, &facts, err);
    rc = dense_primitive(run, rc, &facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gqa(
            run->backend, run->device[DENSE_QUERY], run->device[DENSE_KEY],
            run->device[DENSE_VALUE], run->device[DENSE_ATTENTION],
            r->rows, r->rows, 0ull,
            r->heads, r->heads, r->head_dim, 0, &facts, err);
    return dense_primitive(run, rc, &facts, err);
}

static int dense_mlp(dense_decoder_run *run,
                     const yvex_transformer_encoded_weight *weights,
                     yvex_error *err)
{
    const yvex_transformer_dense_decoder_request *r = run->request;
    yvex_backend_operation_facts facts;
    unsigned long long fused_width;
    int rc;
    if (!yvex_core_u64_mul(r->ffn_width, 2ull, &fused_width))
        return dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.mlp",
                            "dense decoder FFN width overflowed");
    rc = dense_project(run, weights + YVEX_TRANSFORMER_DENSE_FF1_WEIGHT,
                       r->rows, run->device[DENSE_NORM], run->device[DENSE_FUSED], err);
    if (rc == YVEX_OK)
        rc = dense_bias(run, weights + YVEX_TRANSFORMER_DENSE_FF1_BIAS,
                        run->device[DENSE_FUSED], fused_width, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_swiglu_split_f32(
            run->backend, run->device[DENSE_FUSED], run->device[DENSE_GATED],
            r->rows, r->ffn_width, 1, &facts, err);
    rc = dense_primitive(run, rc, &facts, err);
    if (rc == YVEX_OK)
        rc = dense_project(run, weights + YVEX_TRANSFORMER_DENSE_FF2_WEIGHT,
                           r->rows, run->device[DENSE_GATED], run->device[DENSE_UPDATE], err);
    if (rc == YVEX_OK)
        rc = dense_bias(run, weights + YVEX_TRANSFORMER_DENSE_FF2_BIAS,
                        run->device[DENSE_UPDATE], r->width, err);
    return rc;
}

static int dense_block_execute(dense_decoder_run *run, unsigned long long block,
                               yvex_error *err)
{
    const yvex_transformer_dense_decoder_request *r = run->request;
    const yvex_transformer_encoded_weight *weights =
        r->block_weights + block * YVEX_TRANSFORMER_DENSE_DECODER_BLOCK_WEIGHT_COUNT;
    yvex_backend_operation_facts facts;
    int rc;
    if (r->cancel_requested && r->cancel_requested(r->cancel_context))
        return dense_refuse(err, YVEX_ERR_CANCELLED, "cuda.dense-decoder.cancel",
                            "dense decoder execution was cancelled between blocks");
    rc = dense_norm(run, weights + YVEX_TRANSFORMER_DENSE_NORM1,
                    run->device[DENSE_HIDDEN], run->device[DENSE_NORM], err);
    if (rc == YVEX_OK) rc = dense_attention(run, weights, err);
    if (rc == YVEX_OK)
        rc = dense_project(run, weights + YVEX_TRANSFORMER_DENSE_ATTENTION_WEIGHT,
                           r->rows, run->device[DENSE_ATTENTION],
                           run->device[DENSE_UPDATE], err);
    if (rc == YVEX_OK)
        rc = dense_bias(run, weights + YVEX_TRANSFORMER_DENSE_ATTENTION_BIAS,
                        run->device[DENSE_UPDATE], r->width, err);
    if (rc == YVEX_OK)
        rc = dense_gather(run, weights + YVEX_TRANSFORMER_DENSE_SCALE1,
                          run->device[DENSE_VECTOR], err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_scaled_residual_f32(
            run->backend, run->device[DENSE_HIDDEN], run->device[DENSE_UPDATE],
            run->device[DENSE_VECTOR], run->device[DENSE_HIDDEN],
            r->rows, r->width, &facts, err);
    rc = dense_primitive(run, rc, &facts, err);
    if (rc == YVEX_OK)
        rc = dense_norm(run, weights + YVEX_TRANSFORMER_DENSE_NORM2,
                        run->device[DENSE_HIDDEN], run->device[DENSE_NORM], err);
    if (rc == YVEX_OK) rc = dense_mlp(run, weights, err);
    if (rc == YVEX_OK)
        rc = dense_gather(run, weights + YVEX_TRANSFORMER_DENSE_SCALE2,
                          run->device[DENSE_VECTOR], err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_scaled_residual_f32(
            run->backend, run->device[DENSE_HIDDEN], run->device[DENSE_UPDATE],
            run->device[DENSE_VECTOR], run->device[DENSE_HIDDEN],
            r->rows, r->width, &facts, err);
    return dense_primitive(run, rc, &facts, err);
}

static int dense_inputs_write(dense_decoder_run *run, yvex_error *err)
{
    const yvex_transformer_dense_decoder_request *r = run->request;
    unsigned long long hidden_values, rotary_values, index;
    float *ones;
    int rc;
    if (!yvex_core_u64_mul(r->rows, r->width, &hidden_values) ||
        !yvex_core_u64_mul(r->rows, r->rotary_dim, &rotary_values) ||
        r->head_dim > SIZE_MAX / sizeof(float))
        return dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.input",
                            "dense decoder input geometry overflowed");
    ones = (float *)malloc((size_t)r->head_dim * sizeof(float));
    if (!ones)
        return dense_refuse(err, YVEX_ERR_NOMEM, "cuda.dense-decoder.input",
                            "dense decoder Q/K normalization vector allocation failed");
    for (index = 0ull; index < r->head_dim; ++index) ones[index] = 1.0f;
    rc = yvex_backend_tensor_write(
        run->backend, run->device[DENSE_HIDDEN], r->hidden,
        hidden_values * sizeof(float), err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(
            run->backend, run->device[DENSE_COSINE], r->cosines,
            rotary_values * sizeof(float), err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(
            run->backend, run->device[DENSE_SINE], r->sines,
            rotary_values * sizeof(float), err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(
            run->backend, run->device[DENSE_ONES], ones,
            r->head_dim * sizeof(float), err);
    free(ones);
    if (rc == YVEX_OK) {
        run->facts.h2d_bytes =
            (hidden_values + rotary_values * 2ull + r->head_dim) * sizeof(float);
    }
    return rc;
}

static int dense_output_execute(dense_decoder_run *run, float *staged,
                                yvex_error *err)
{
    const yvex_transformer_dense_decoder_request *r = run->request;
    yvex_backend_operation_facts facts;
    yvex_device_tensor bias_view = {0};
    unsigned long long values, index;
    int rc = dense_gather(run, r->final_norm_weight, run->device[DENSE_VECTOR], err);
    if (rc == YVEX_OK)
        rc = dense_gather(run, r->final_norm_bias, run->device[DENSE_BIAS], err);
    if (rc == YVEX_OK && !yvex_backend_tensor_f32_subview(
            run->device[DENSE_BIAS], 0ull, r->width, &bias_view))
        rc = dense_refuse(err, YVEX_ERR_FORMAT, "cuda.dense-decoder.final-bias",
                          "dense decoder final norm bias prefix is invalid");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_layer_norm_f32(
            run->backend, run->device[DENSE_HIDDEN], run->device[DENSE_VECTOR],
            &bias_view, run->device[DENSE_NORM], r->rows,
            r->width, r->epsilon, &facts, err);
    rc = dense_primitive(run, rc, &facts, err);
    if (rc == YVEX_OK)
        rc = dense_project(run, r->output_weight, r->output_rows,
                           run->device[DENSE_NORM], run->device[DENSE_OUTPUT], err);
    if (rc == YVEX_OK)
        rc = dense_gather(run, r->output_bias, run->device[DENSE_BIAS], err);
    if (rc == YVEX_OK && !yvex_backend_tensor_f32_subview(
            run->device[DENSE_BIAS], 0ull, r->output_width, &bias_view))
        rc = dense_refuse(err, YVEX_ERR_FORMAT, "cuda.dense-decoder.output-bias",
                          "dense decoder output bias prefix is invalid");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(
            run->backend, run->device[DENSE_OUTPUT], &bias_view,
            run->device[DENSE_OUTPUT], r->output_rows, r->output_width,
            0, &facts, err);
    rc = dense_primitive(run, rc, &facts, err);
    if (!yvex_core_u64_mul(r->output_rows, r->output_width, &values))
        return dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.output",
                            "dense decoder output geometry overflowed");
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(
            run->backend, run->device[DENSE_OUTPUT], staged,
            values * sizeof(float), err);
    if (rc == YVEX_OK) {
        run->facts.d2h_bytes += values * sizeof(float);
        for (index = 0ull; index < values; ++index)
            if (!isfinite(staged[index]))
                return dense_refuse(err, YVEX_ERR_FORMAT, "cuda.dense-decoder.output",
                                    "dense decoder output contains a non-finite value");
    }
    return rc;
}

int yvex_cuda_transformer_dense_decoder_execute(
    yvex_backend *backend, const yvex_transformer_dense_decoder_request *request,
    yvex_transformer_dense_decoder_result *result, yvex_error *err)
{
    dense_decoder_run run = {0};
    unsigned long long output_values, block;
    float *staged = NULL;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!backend || !result || !dense_request_valid(request) ||
        yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA)
        return dense_refuse(err, YVEX_ERR_INVALID_ARG, "cuda.dense-decoder.validate",
                            "one valid resident F32 dense decoder request is required");
    if (!yvex_core_u64_mul(request->output_rows, request->output_width, &output_values) ||
        output_values > SIZE_MAX / sizeof(float) ||
        !(staged = (float *)malloc((size_t)output_values * sizeof(float))))
        return dense_refuse(err, YVEX_ERR_NOMEM, "cuda.dense-decoder.output",
                            "dense decoder transactional output allocation failed");
    run.backend = backend;
    run.request = request;
    run.facts.compulsory_memory_facts_available = 1;
    rc = dense_devices_prepare(&run, err);
    if (rc == YVEX_OK) rc = dense_inputs_write(&run, err);
    for (block = 0ull; rc == YVEX_OK && block < request->block_count; ++block)
        rc = dense_block_execute(&run, block, err);
    if (rc == YVEX_OK && request->cancel_requested &&
        request->cancel_requested(request->cancel_context))
        rc = dense_refuse(err, YVEX_ERR_CANCELLED, "cuda.dense-decoder.cancel",
                          "dense decoder execution was cancelled before publication");
    if (rc == YVEX_OK) rc = dense_output_execute(&run, staged, err);
    if (rc == YVEX_OK) {
        memcpy(request->output, staged, (size_t)output_values * sizeof(float));
        result->rows = request->rows;
        result->output_rows = request->output_rows;
        result->block_count = request->block_count;
        result->output_values = output_values;
        result->kernel_launches = run.facts.kernel_launches;
        result->h2d_bytes = run.facts.h2d_bytes;
        result->d2h_bytes = run.facts.d2h_bytes;
        result->device_bytes = run.device_bytes;
        result->complete = 1;
        yvex_error_clear(err);
    }
    rc = dense_devices_release(&run, rc, err);
    free(staged);
    if (rc != YVEX_OK) memset(result, 0, sizeof(*result));
    return rc;
}
