/* Execute reusable CUDA elementwise operations required by heterogeneous decoders. */
#include "src/backend/cuda/private.h"
#include "src/backend/cuda/transformer_ops.h"

#include <limits.h>
#include <string.h>

#include <yvex/internal/core.h>

#define DECODER_BLOCK 256u

int yvex_cuda_decoder_split_interleaved_two_f32(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *first, yvex_device_tensor *second,
    unsigned long long rows, unsigned long long heads,
    unsigned long long head_dimension, yvex_backend_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_address, first_address, second_address;
    unsigned long long width, output_values, input_values, output_bytes;
    unsigned long long input_bytes, tasks;
    unsigned int grid;
    int rc;

    if (facts) memset(facts, 0, sizeof(*facts));
    if (!backend || !state || !state->split_interleaved_two_function ||
        !facts || !rows || !heads || !head_dimension ||
        !yvex_core_u64_mul(heads, head_dimension, &width) ||
        !yvex_core_u64_mul(rows, width, &output_values) ||
        !yvex_core_u64_mul(output_values, 2ull, &input_values) ||
        !yvex_core_u64_mul(output_values, sizeof(float), &output_bytes) ||
        !yvex_core_u64_mul(input_values, sizeof(float), &input_bytes) ||
        !yvex_core_u64_add(output_values, DECODER_BLOCK - 1ull, &tasks) ||
        tasks / DECODER_BLOCK > UINT_MAX ||
        !yvex_backend_tensor_owned_by(backend, input) || !input->is_written ||
        input->dtype != YVEX_DTYPE_F32 || input->bytes != input_bytes ||
        !yvex_backend_tensor_owned_by(backend, first) ||
        first->dtype != YVEX_DTYPE_F32 || first->bytes != output_bytes ||
        !yvex_backend_tensor_owned_by(backend, second) ||
        second->dtype != YVEX_DTYPE_F32 || second->bytes != output_bytes ||
        input == first || input == second || first == second) {
        yvex_error_set(err, YVEX_ERR_FORMAT,
                       "cuda.decoder.split-interleaved-two",
                       "one per-head interleaved F32 pair and independent outputs are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / DECODER_BLOCK);
    input_address = yvex_cuda_tensor_ptr(input);
    first_address = yvex_cuda_tensor_ptr(first);
    second_address = yvex_cuda_tensor_ptr(second);
    {
        void *parameters[] = {&input_address, &first_address, &second_address,
                              &rows, &heads, &head_dimension};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->split_interleaved_two_function, grid, DECODER_BLOCK, 0u,
            parameters, "cuda.decoder.split-interleaved-two", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            "cuda.decoder.split-interleaved-two", err);
    if (rc == YVEX_OK) {
        first->is_written = second->is_written = 1;
        facts->kernel_launches = 1ull;
        facts->device_synchronizations = 1ull;
        facts->activation_bytes = input_bytes + 2ull * output_bytes;
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_cuda_decoder_sigmoid_product_bf16(
    yvex_backend *backend, const yvex_device_tensor *values,
    const yvex_device_tensor *gate, yvex_device_tensor *output,
    unsigned long long count, yvex_backend_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr values_address, gate_address, output_address;
    unsigned long long bytes, tasks;
    unsigned int grid;
    int rc;

    if (facts) memset(facts, 0, sizeof(*facts));
    if (!backend || !state || !state->sigmoid_product_function || !facts ||
        !count || !yvex_core_u64_mul(count, sizeof(float), &bytes) ||
        !yvex_core_u64_add(count, DECODER_BLOCK - 1ull, &tasks) ||
        tasks / DECODER_BLOCK > UINT_MAX ||
        !yvex_backend_tensor_owned_by(backend, values) || !values->is_written ||
        values->dtype != YVEX_DTYPE_F32 || values->bytes != bytes ||
        !yvex_backend_tensor_owned_by(backend, gate) || !gate->is_written ||
        gate->dtype != YVEX_DTYPE_F32 || gate->bytes != bytes ||
        !yvex_backend_tensor_owned_by(backend, output) ||
        output->dtype != YVEX_DTYPE_F32 || output->bytes != bytes) {
        yvex_error_set(err, YVEX_ERR_FORMAT,
                       "cuda.decoder.sigmoid-product",
                       "equal bounded F32 decoder activations are required");
        return YVEX_ERR_FORMAT;
    }
    grid = (unsigned int)(tasks / DECODER_BLOCK);
    values_address = yvex_cuda_tensor_ptr(values);
    gate_address = yvex_cuda_tensor_ptr(gate);
    output_address = yvex_cuda_tensor_ptr(output);
    {
        void *parameters[] = {
            &values_address, &gate_address, &output_address, &count};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->sigmoid_product_function, grid, DECODER_BLOCK, 0u,
            parameters, "cuda.decoder.sigmoid-product", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            "cuda.decoder.sigmoid-product", err);
    if (rc == YVEX_OK) {
        output->is_written = 1;
        facts->kernel_launches = 1ull;
        facts->device_synchronizations = 1ull;
        facts->activation_bytes = 3ull * bytes;
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}
