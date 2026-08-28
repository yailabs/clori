/* Compare both generic CUDA Conv1D algorithms with an independent exact F32 reference. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>
#include <yvex/internal/convolution.h>

#include "src/backend/cuda/private.h"
#include "tests/test.h"

#define CONVOLUTION_TEST_BLOCK 256u

typedef struct {
    yvex_graph_conv1d_geometry geometry;
    int bias, scale;
} convolution_case;

static int open_cuda_or_skip(yvex_backend **out)
{
    yvex_backend_options options = {0};
    yvex_error err;
    int rc;

    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(out, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) return 77;
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open CUDA convolution backend");
    return 0;
}

static float fixture_value(size_t index, unsigned int salt, float divisor)
{
    int value = (int)((index * 13u + salt * 7u) % 17u) - 8;
    return (float)value / divisor;
}

static void convolution_reference(
    const convolution_case *test, const float *input, const float *weight,
    const float *bias, const float *scale, unsigned long long output_length,
    float *output)
{
    const yvex_graph_conv1d_geometry *geometry = &test->geometry;
    unsigned long long batch_index, output_channel, output_position;

    for (batch_index = 0ull; batch_index < geometry->batch; ++batch_index)
        for (output_channel = 0ull; output_channel < geometry->output_channels;
             ++output_channel)
            for (output_position = 0ull; output_position < output_length;
                 ++output_position) {
                float sum = bias ? bias[output_channel] : 0.0f;
                unsigned long long input_channel;
                for (input_channel = 0ull; input_channel < geometry->input_channels;
                     ++input_channel) {
                    float factor = scale
                                       ? scale[geometry->transposed ? input_channel
                                                                    : output_channel]
                                       : 1.0f;
                    if (geometry->transposed) {
                        unsigned long long input_position;
                        for (input_position = 0ull;
                             input_position < geometry->input_length; ++input_position) {
                            unsigned long long kernel;
                            float value =
                                input[(batch_index * geometry->input_channels + input_channel) *
                                          geometry->input_length +
                                      input_position] *
                                factor;
                            for (kernel = 0ull; kernel < geometry->kernel_size; ++kernel) {
                                unsigned long long projected =
                                    input_position * geometry->stride +
                                    kernel * geometry->dilation;
                                unsigned long long weight_index;
                                if (projected < geometry->padding ||
                                    projected - geometry->padding != output_position)
                                    continue;
                                weight_index =
                                    (input_channel * geometry->output_channels +
                                     output_channel) *
                                        geometry->kernel_size +
                                    kernel;
                                sum += value * weight[weight_index];
                            }
                        }
                    } else {
                        unsigned long long kernel;
                        for (kernel = 0ull; kernel < geometry->kernel_size; ++kernel) {
                            unsigned long long projected =
                                output_position * geometry->stride +
                                kernel * geometry->dilation;
                            unsigned long long input_position, weight_index;
                            if (projected < geometry->padding) continue;
                            input_position = projected - geometry->padding;
                            if (input_position >= geometry->input_length) continue;
                            weight_index =
                                (output_channel * geometry->input_channels + input_channel) *
                                    geometry->kernel_size +
                                kernel;
                            sum += input[(batch_index * geometry->input_channels +
                                          input_channel) *
                                             geometry->input_length +
                                         input_position] *
                                   weight[weight_index] * factor;
                        }
                    }
                }
                output[(batch_index * geometry->output_channels + output_channel) *
                           output_length +
                       output_position] = sum;
            }
}

static int run_convolution_case(yvex_backend *backend, const convolution_case *test)
{
    const yvex_graph_conv1d_geometry *geometry = &test->geometry;
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *arena = NULL;
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    unsigned long long output_length, input_count, weight_count, output_count;
    unsigned long long batch, input_channels, output_channels, input_length;
    unsigned long long kernel_size, stride, dilation, padding;
    size_t input_offset = 0u, weight_offset, bias_offset, scale_offset, output_offset, total;
    float *host = NULL, *observed = NULL, *reference = NULL;
    CUdeviceptr base, input, weight, bias = 0ull, scale = 0ull, output;
    CUfunction function;
    unsigned long long tasks;
    unsigned int grid;
    int device_wide = 0, rc;
    yvex_error err;

    YVEX_TEST_ASSERT(
        yvex_graph_conv1d_output_length(geometry, &output_length, &err) == YVEX_OK,
        "derive generic convolution test output length");
    input_count = geometry->batch * geometry->input_channels * geometry->input_length;
    weight_count = geometry->input_channels * geometry->output_channels * geometry->kernel_size;
    output_count = geometry->batch * geometry->output_channels * output_length;
    weight_offset = (size_t)input_count;
    bias_offset = weight_offset + (size_t)weight_count;
    scale_offset = bias_offset + (test->bias ? (size_t)geometry->output_channels : 0u);
    output_offset = scale_offset +
                    (test->scale ? (size_t)(geometry->transposed
                                                 ? geometry->input_channels
                                                 : geometry->output_channels)
                                 : 0u);
    total = output_offset + (size_t)output_count;
    host = (float *)calloc(total, sizeof(*host));
    observed = (float *)calloc(total, sizeof(*observed));
    reference = (float *)calloc((size_t)output_count, sizeof(*reference));
    YVEX_TEST_ASSERT(host && observed && reference, "allocate generic convolution fixture");
    for (size_t index = 0u; index < (size_t)input_count; ++index)
        host[input_offset + index] = fixture_value(index, 1u, 16.0f);
    for (size_t index = 0u; index < (size_t)weight_count; ++index)
        host[weight_offset + index] = fixture_value(index, 2u, 32.0f);
    if (test->bias)
        for (size_t index = 0u; index < (size_t)geometry->output_channels; ++index)
            host[bias_offset + index] = fixture_value(index, 3u, 64.0f);
    if (test->scale) {
        size_t count = (size_t)(geometry->transposed ? geometry->input_channels
                                                     : geometry->output_channels);
        for (size_t index = 0u; index < count; ++index)
            host[scale_offset + index] = index % 2u ? 0.5f : 1.0f;
    }
    convolution_reference(test, host + input_offset, host + weight_offset,
                          test->bias ? host + bias_offset : NULL,
                          test->scale ? host + scale_offset : NULL,
                          output_length, reference);
    descriptor.name = "cuda-convolution-matrix";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = total;
    descriptor.bytes = total * sizeof(*host);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &arena, &err) == YVEX_OK &&
            yvex_backend_tensor_write(backend, arena, host, descriptor.bytes, &err) == YVEX_OK,
        "upload generic convolution fixture");
    base = yvex_cuda_tensor_ptr(arena);
    input = base + input_offset * sizeof(*host);
    weight = base + weight_offset * sizeof(*host);
    if (test->bias) bias = base + bias_offset * sizeof(*host);
    if (test->scale) scale = base + scale_offset * sizeof(*host);
    output = base + output_offset * sizeof(*host);
    tasks = output_count;
    grid = (unsigned int)((tasks + CONVOLUTION_TEST_BLOCK - 1ull) /
                          CONVOLUTION_TEST_BLOCK);
    function = geometry->transposed ? state->conv1d_transposed_function
                                    : state->conv1d_function;
    batch = geometry->batch;
    input_channels = geometry->input_channels;
    output_channels = geometry->output_channels;
    input_length = geometry->input_length;
    kernel_size = geometry->kernel_size;
    stride = geometry->stride;
    dilation = geometry->dilation;
    padding = geometry->padding;
    for (unsigned int repeat = 0u; repeat < 2u; ++repeat) {
        void *parameters[] = {
            &input, &weight, &bias, &scale, &output, &batch,
            &input_channels, &output_channels, &input_length, &output_length,
            &kernel_size, &stride, &dilation, &padding,
        };
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, function,
            grid, CONVOLUTION_TEST_BLOCK, 0u, parameters,
            geometry->transposed ? "cuda.test.conv1d-transposed" : "cuda.test.conv1d",
            &err);
        if (rc == YVEX_OK)
            rc = yvex_cuda_launch_synchronize(
                backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &device_wide,
                geometry->transposed ? "cuda.test.conv1d-transposed" : "cuda.test.conv1d",
                &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK, "execute generic CUDA convolution matrix case");
        YVEX_TEST_ASSERT(
            yvex_backend_tensor_read(backend, arena, observed, descriptor.bytes, &err) ==
                YVEX_OK &&
                memcmp(observed + output_offset, reference,
                       (size_t)output_count * sizeof(*reference)) == 0,
            "generic CUDA convolution is byte-exact with its independent reference");
    }
    YVEX_TEST_ASSERT(yvex_backend_tensor_release(backend, &arena, &err) == YVEX_OK,
                     "release generic convolution fixture");
    free(reference);
    free(observed);
    free(host);
    return 0;
}

int yvex_cuda_test_convolution(void)
{
    static const convolution_case cases[] = {
        {{1ull, 2ull, 2ull, 5ull, 3ull, 1ull, 1ull, 1ull, 0ull, 0}, 1, 1},
        {{2ull, 3ull, 2ull, 7ull, 2ull, 2ull, 1ull, 1ull, 0ull, 0}, 0, 0},
        {{1ull, 1ull, 1ull, 1ull, 1ull, 1ull, 1ull, 0ull, 0ull, 1}, 1, 1},
        {{2ull, 3ull, 4ull, 7ull, 3ull, 1ull, 1ull, 1ull, 0ull, 1}, 1, 1},
        {{1ull, 5ull, 3ull, 9ull, 4ull, 2ull, 1ull, 1ull, 0ull, 1}, 0, 1},
        {{1ull, 4ull, 2ull, 8ull, 5ull, 3ull, 2ull, 4ull, 0ull, 1}, 1, 0},
        {{2ull, 2ull, 3ull, 6ull, 4ull, 2ull, 3ull, 2ull, 0ull, 1}, 1, 1},
        {{1ull, 3ull, 2ull, 5ull, 1ull, 4ull, 1ull, 0ull, 0ull, 1}, 0, 0},
        {{1ull, 2ull, 2ull, 9ull, 3ull, 3ull, 1ull, 1ull, 0ull, 1}, 1, 1},
        {{1ull, 2ull, 3ull, 11ull, 6ull, 2ull, 2ull, 1ull, 0ull, 1}, 1, 1},
        {{1ull, 2ull, 2ull, 3ull, 3ull, 2ull, 1ull, 1ull, 1ull, 1}, 1, 1},
    };
    yvex_backend *backend = NULL;
    int rc = open_cuda_or_skip(&backend);

    if (rc == 77) return rc;
    YVEX_TEST_ASSERT(yvex_cuda_state(backend)->conv1d_function &&
                         yvex_cuda_state(backend)->conv1d_transposed_function,
                     "ordinary and transposed convolution kernels are admitted separately");
    for (size_t index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index)
        if (run_convolution_case(backend, &cases[index]) != 0) return 1;
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&backend, NULL) == YVEX_OK && !backend,
                     "close generic convolution CUDA backend");
    return 0;
}
