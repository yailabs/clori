/* Execute reusable dense-transformer activation operations on admitted CUDA tensors. */
#include "src/backend/cuda/private.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

enum { TRANSFORMER_BLOCK = 128u };

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
    yvex_cuda_work work = {0};
    CUdeviceptr values_ptr, status = 0ull;
    unsigned long long tasks;
    unsigned int grid;
    int host_status = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !facts || !count || !transformer_tensor(backend, values, count, 1) ||
        !yvex_core_u64_add(count, TRANSFORMER_BLOCK - 1ull, &tasks) ||
        tasks / TRANSFORMER_BLOCK > UINT_MAX) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.bf16-round",
                       "one finite bounded F32 activation is required");
        return YVEX_ERR_FORMAT;
    }
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    rc = yvex_cuda_work_allocate(&work, &status, sizeof(int), NULL, 1,
                                 "cuda.transformer.bf16-round.status", NULL, err);
    values_ptr = yvex_cuda_tensor_ptr(values);
    grid = (unsigned int)(tasks / TRANSFORMER_BLOCK);
    if (rc == YVEX_OK) {
        void *parameters[] = {&values_ptr, &count, &status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->attention_bf16_round_function, grid, TRANSFORMER_BLOCK, 0u,
            parameters, "cuda.transformer.bf16-round", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            "cuda.transformer.bf16-round", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemcpyDtoH_v2(&host_status, status, sizeof(host_status)),
            "cuda.transformer.bf16-round.status", err);
    if (rc == YVEX_OK && host_status) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.transformer.bf16-round",
                       "activation contains a non-finite value");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc == YVEX_OK) {
        facts->d2h_bytes = sizeof(host_status);
        facts->kernel_launches = 1ull;
        facts->download_count = 1ull;
        facts->device_synchronizations = 1ull;
        facts->temporary_bytes = sizeof(int);
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
