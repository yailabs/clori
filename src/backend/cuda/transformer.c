/* Execute reusable dense-transformer activation operations on admitted CUDA tensors. */
#include "src/backend/cuda/private.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
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

static int transformer_rotary_half(
    yvex_backend *backend, yvex_device_tensor *values,
    const yvex_device_tensor *cosines, const yvex_device_tensor *sines,
    unsigned long long tokens, unsigned long long heads, unsigned long long head_dim,
    unsigned long long rotary_dim, CUfunction function, const char *stage,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    unsigned long long rotary_dim, yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    unsigned long long rotary_dim, yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    return transformer_rotary_half(
        backend, values, cosines, sines, tokens, heads, head_dim, rotary_dim,
        state ? state->rotary_half_plain_function : NULL, "cuda.transformer.rotary-f32", facts, err);
}

int yvex_cuda_transformer_gqa(
    yvex_backend *backend, const yvex_device_tensor *query,
    const yvex_device_tensor *key, const yvex_device_tensor *value,
    yvex_device_tensor *output, unsigned long long tokens,
    unsigned long long query_heads, unsigned long long kv_heads,
    unsigned long long head_dim, int causal, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    unsigned long long query_elements, kv_elements, rows;
    CUdeviceptr query_ptr, key_ptr, value_ptr, output_ptr;
    float scale;
    int rc;
    if (!state || !tokens || !query_heads || !kv_heads || query_heads % kv_heads ||
        (causal != 0 && causal != 1) ||
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
        void *parameters[] = {
            &query_ptr, &key_ptr, &value_ptr, &output_ptr, &tokens,
            &query_heads, &kv_heads, &head_dim, &scale, &causal,
        };
        rc = transformer_launch(
            backend, state->gqa_function, (unsigned int)rows,
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

int yvex_cuda_transformer_silu(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *output, unsigned long long count, int bf16_output,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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

int yvex_cuda_transformer_split_three(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *first, yvex_device_tensor *second, yvex_device_tensor *third,
    unsigned long long rows, unsigned long long width,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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

int yvex_cuda_transformer_scaled_residual_f32(
    yvex_backend *backend, const yvex_device_tensor *residual,
    const yvex_device_tensor *update, const yvex_device_tensor *scale,
    yvex_device_tensor *output, unsigned long long rows, unsigned long long width,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    float epsilon, yvex_backend_cuda_operation_facts *facts, yvex_error *err)
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
    yvex_backend_cuda_operation_facts facts;
    unsigned long long device_bytes;
} dense_decoder_run;

static int dense_refuse(yvex_error *err, yvex_status status,
                        const char *stage, const char *message)
{
    yvex_error_set(err, status, stage, message);
    return status;
}

static int dense_facts_add(dense_decoder_run *run,
                           const yvex_backend_cuda_operation_facts *part)
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
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_gather(
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
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_matvec(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes, rows,
        input, NULL, 0ull, NULL, output, 0, &facts, err);
    if (rc == YVEX_OK && !dense_facts_add(run, &facts))
        rc = dense_refuse(err, YVEX_ERR_BOUNDS, "cuda.dense-decoder.facts",
                          "dense decoder projection accounting overflowed");
    return rc;
}

static int dense_primitive(dense_decoder_run *run, int rc,
                           const yvex_backend_cuda_operation_facts *facts,
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
    yvex_backend_cuda_operation_facts facts;
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
    yvex_backend_cuda_operation_facts facts;
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
            run->device[DENSE_VALUE], run->device[DENSE_ATTENTION], r->rows,
            r->heads, r->heads, r->head_dim, 0, &facts, err);
    return dense_primitive(run, rc, &facts, err);
}

static int dense_mlp(dense_decoder_run *run,
                     const yvex_transformer_encoded_weight *weights,
                     yvex_error *err)
{
    const yvex_transformer_dense_decoder_request *r = run->request;
    yvex_backend_cuda_operation_facts facts;
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
    yvex_backend_cuda_operation_facts facts;
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
    yvex_backend_cuda_operation_facts facts;
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
