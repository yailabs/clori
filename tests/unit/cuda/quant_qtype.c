/*
 * All encoded inputs come from canonical TRACK.QUANT codecs. Qtype primitive parity is not
 * transformer or generation support.
 */
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "src/backend/cuda/private.h"
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/transformer.h>

#include "tests/test.h"

static int quant_cuda_encode_row(unsigned int qtype,
                                 const float *source,
                                 unsigned long long elements,
                                 unsigned char **encoded,
                                 size_t *encoded_bytes)
{
    const yvex_gguf_qtype_geometry *geometry =
        yvex_gguf_qtype_geometry_find(qtype);
    yvex_gguf_qtype_storage_result storage;
    yvex_quant_failure failure;
    yvex_error err;
    unsigned long long dims[1];
    unsigned long long block;

    *encoded = NULL;
    *encoded_bytes = 0u;
    dims[0] = elements;
    if (!geometry || yvex_gguf_qtype_tensor_storage(
            qtype, dims, 1u, &storage) != YVEX_GGUF_QTYPE_STORAGE_OK ||
        storage.total_bytes > SIZE_MAX) return 0;
    *encoded = (unsigned char *)malloc((size_t)storage.total_bytes);
    if (!*encoded) return 0;
    for (block = 0u; block < elements / geometry->block_size; ++block) {
        float calibration[YVEX_QUANT_IQ2_XXS_ELEMENTS];
        size_t wrote = 0u;
        unsigned int index;
        int rc;
        for (index = 0u; index < geometry->block_size; ++index)
            calibration[index] = 0.5f + (float)((block + index) % 17u) * 0.125f;
        rc = qtype == YVEX_GGUF_QTYPE_IQ2_XXS
                 ? yvex_quant_encode_block_weighted(
                       qtype, source + block * geometry->block_size, calibration,
                       geometry->block_size,
                       *encoded + (size_t)block * geometry->bytes_per_block,
                       geometry->bytes_per_block, &wrote, &failure, &err)
                 : yvex_quant_encode_block(
                       qtype, source + block * geometry->block_size, geometry->block_size,
                       *encoded + (size_t)block * geometry->bytes_per_block,
                       geometry->bytes_per_block, &wrote, &failure, &err);
        if (rc != YVEX_OK || wrote != geometry->bytes_per_block) {
            free(*encoded);
            *encoded = NULL;
            return 0;
        }
    }
    *encoded_bytes = (size_t)storage.total_bytes;
    return 1;
}

static int quant_cuda_parity(yvex_backend *backend,
                             unsigned int qtype,
                             unsigned long long elements,
                             unsigned int row_seed,
                             double *maximum_difference,
                             double *maximum_relative_difference)
{
    float *source = (float *)malloc((size_t)elements * sizeof(float));
    float *vector = (float *)malloc((size_t)elements * sizeof(float));
    unsigned char *encoded = NULL;
    size_t encoded_bytes = 0u;
    yvex_quant_failure failure;
    yvex_error err;
    float cpu = 0.0f;
    float cuda = 0.0f;
    unsigned long long index;
    unsigned int repeat;

    YVEX_TEST_ASSERT(source && vector, "CUDA qtype host vectors allocate");
    for (index = 0u; index < elements; ++index) {
        int centered = (int)((index + row_seed * 11u) % 31u) - 15;
        source[index] = qtype == YVEX_GGUF_QTYPE_I32
            ? (float)centered
            : (float)centered / (float)(1u + (index % 7u));
        vector[index] =
            (float)((int)((index + row_seed * 5u) % 19u) - 9) / 9.0f;
    }
    YVEX_TEST_ASSERT(quant_cuda_encode_row(
                         qtype, source, elements, &encoded, &encoded_bytes),
                     "CUDA qtype row encodes canonically");
    YVEX_TEST_ASSERT(yvex_quant_cpu_dot(
                         qtype, encoded, encoded_bytes, vector, elements,
                         &cpu, &failure, &err) == YVEX_OK,
                     "CPU qtype row-dot reference succeeds");
    for (repeat = 0u; repeat < 3u; ++repeat) {
        double difference;
        double relative_difference;
        YVEX_TEST_ASSERT(yvex_cuda_quant_row_dot(
                             backend, qtype, encoded, encoded_bytes,
                             vector, elements, &cuda, &failure, &err) ==
                             YVEX_OK,
                         "CUDA qtype row-dot launch succeeds");
        difference = fabs((double)cuda - (double)cpu);
        if (difference > *maximum_difference)
            *maximum_difference = difference;
        relative_difference = difference /
            fmax(fabs((double)cpu), 1e-12);
        if (relative_difference > *maximum_relative_difference)
            *maximum_relative_difference = relative_difference;
        YVEX_TEST_ASSERT(difference <=
                             1e-6 * (1.0 + fabs((double)cpu)),
                         "CUDA qtype direct arithmetic matches CPU reference");
    }
    memset(source, 0, (size_t)elements * sizeof(float));
    free(encoded);
    encoded = NULL;
    YVEX_TEST_ASSERT(quant_cuda_encode_row(
                         qtype, source, elements, &encoded, &encoded_bytes),
                     "CUDA qtype zero row encodes");
    YVEX_TEST_ASSERT(yvex_cuda_quant_row_dot(
                         backend, qtype, encoded, encoded_bytes,
                         vector, elements, &cuda, &failure, &err) ==
                         YVEX_OK && cuda == 0.0f,
                     "CUDA qtype zero block returns exact zero");
    free(encoded);
    encoded = NULL;
    if (qtype == YVEX_GGUF_QTYPE_F16) {
        memset(source, 0, (size_t)elements * sizeof(float));
        memset(vector, 0, (size_t)elements * sizeof(float));
        source[0] = yvex_quant_f16_decode(1u);
        vector[0] = 1.0f;
        YVEX_TEST_ASSERT(quant_cuda_encode_row(
                             qtype, source, elements, &encoded,
                             &encoded_bytes),
                         "CUDA F16 subnormal row encodes canonically");
        YVEX_TEST_ASSERT(yvex_quant_cpu_dot(
                             qtype, encoded, encoded_bytes, vector, elements,
                             &cpu, &failure, &err) == YVEX_OK &&
                             cpu == source[0],
                         "CPU F16 row dot preserves the minimum subnormal");
        YVEX_TEST_ASSERT(yvex_cuda_quant_row_dot(
                             backend, qtype, encoded, encoded_bytes,
                             vector, elements, &cuda, &failure, &err) ==
                             YVEX_OK && cuda == cpu,
                         "CUDA F16 row dot preserves the minimum subnormal");
    }
    free(encoded);
    free(vector);
    free(source);
    return 0;
}

static void quant_q8_reference(const float input[512], float output[512])
{
    unsigned int block;
    for (block = 0u; block < 2u; ++block) {
        unsigned int index, maximum = 0u;
        float absolute = 0.0f, inverse;
        for (index = 0u; index < 256u; ++index) {
            float candidate = fabsf(input[block * 256u + index]);
            if (candidate > absolute) {
                absolute = candidate;
                maximum = index;
            }
        }
        inverse = absolute == 0.0f ? 0.0f :
                  -127.0f / input[block * 256u + maximum];
        for (index = 0u; index < 256u; ++index) {
            int quantized = inverse == 0.0f ? 0 :
                            (int)nearbyintf(inverse * input[block * 256u + index]);
            if (quantized > 127) quantized = 127;
            if (quantized < -128) quantized = -128;
            output[block * 256u + index] = inverse == 0.0f ? 0.0f :
                                                   (float)quantized / inverse;
        }
    }
}

static int quant_cuda_q8_matvec(yvex_backend *backend, unsigned int qtype)
{
    enum { ROWS = 3, INPUT_ROWS = 3, WIDTH = 512 };
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *resident = NULL, *input = NULL, *output = NULL;
    unsigned char *mapped = NULL, *row = NULL;
    float source[ROWS * WIDTH], vectors[INPUT_ROWS * WIDTH];
    float q8_vectors[INPUT_ROWS * WIDTH];
    float exact[INPUT_ROWS * ROWS], expected[INPUT_ROWS * ROWS];
    float actual[INPUT_ROWS * ROWS];
    yvex_quant_failure failure;
    yvex_error err;
    size_t row_bytes = 0u;
    yvex_backend_cuda_operation_facts facts;
    unsigned long long index;
    unsigned int input_index, row_index;
    int rc;

    for (input_index = 0u; input_index < INPUT_ROWS; ++input_index) {
        for (index = 0ull; index < WIDTH; ++index)
            vectors[input_index * WIDTH + index] =
                (float)((int)((index + input_index * 5u) % 29ull) - 14) /
                (float)(13u + input_index);
        quant_q8_reference(vectors + input_index * WIDTH,
                           q8_vectors + input_index * WIDTH);
    }
    for (index = 0ull; index < ROWS * WIDTH; ++index)
        source[index] = (float)((int)((index * 7ull + 3ull) % 41ull) - 20) /
                        (float)(2ull + index % 11ull);
    for (row_index = 0u; row_index < ROWS; ++row_index) {
        size_t current_bytes = 0u;
        YVEX_TEST_ASSERT(quant_cuda_encode_row(
                             qtype, source + row_index * WIDTH, WIDTH,
                             &row, &current_bytes),
                         "Q8 activation matvec row encodes");
        if (!row_index) row_bytes = current_bytes;
        YVEX_TEST_ASSERT(current_bytes == row_bytes,
                         "Q8 activation matvec rows share exact geometry");
        if (!mapped) {
            descriptor.name = "q8_activation_encoded";
            descriptor.dtype = YVEX_DTYPE_I8;
            descriptor.rank = 1u;
            descriptor.dims[0] = descriptor.bytes = ROWS * row_bytes;
            YVEX_TEST_ASSERT(backend->vtable->resident_alloc(
                                 backend, &descriptor, &resident, &mapped, &err) == YVEX_OK,
                             "Q8 activation resident matrix allocates");
        }
        memcpy(mapped + row_index * row_bytes, row, row_bytes);
        free(row);
        row = NULL;
        for (input_index = 0u; input_index < INPUT_ROWS; ++input_index) {
            unsigned int output_index = input_index * ROWS + row_index;
            YVEX_TEST_ASSERT(
                yvex_quant_cpu_dot(
                    qtype, mapped + row_index * row_bytes, row_bytes,
                    vectors + input_index * WIDTH, WIDTH, &exact[output_index],
                    &failure, &err) == YVEX_OK &&
                    yvex_quant_cpu_dot(
                        qtype, mapped + row_index * row_bytes, row_bytes,
                        q8_vectors + input_index * WIDTH, WIDTH,
                        &expected[output_index], &failure, &err) == YVEX_OK,
                "Q8 activation CPU reference succeeds for every input row");
        }
    }
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, mapped, descriptor.bytes, resident, 11ull, &err) == YVEX_OK,
                     "Q8 activation resident matrix attaches");
    descriptor.name = "q8_activation_input";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.dims[0] = INPUT_ROWS * WIDTH;
    descriptor.bytes = sizeof(vectors);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &input, &err) == YVEX_OK &&
                         yvex_backend_tensor_write(
                             backend, input, vectors, sizeof(vectors), &err) == YVEX_OK,
                     "Q8 activation row batch uploads once");
    descriptor.name = "q8_activation_output";
    descriptor.dims[0] = INPUT_ROWS * ROWS;
    descriptor.bytes = sizeof(actual);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &output, &err) == YVEX_OK,
                     "Q8 activation output allocates");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, descriptor.bytes ? ROWS * row_bytes : 0u, qtype,
        ROWS, WIDTH, row_bytes, INPUT_ROWS, input, output, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && facts.kernel_launches == 2ull &&
                         facts.d2h_bytes == sizeof(int) &&
                         facts.device_synchronizations == 1ull &&
                         facts.compulsory_memory_facts_available &&
                         facts.active_weight_bytes == ROWS * row_bytes &&
                         facts.state_bytes == 0ull &&
                         facts.activation_bytes == sizeof(vectors) + sizeof(actual) &&
                         facts.temporary_bytes ==
                             sizeof(int) + INPUT_ROWS * (WIDTH / 256u) * 292u &&
                         yvex_backend_tensor_read(
                             backend, output, actual, sizeof(actual), &err) == YVEX_OK,
                     "Q8 activation production row batch uses one quantize and projection launch");
    for (index = 0ull; index < INPUT_ROWS * ROWS; ++index) {
        double difference = fabs((double)actual[index] - expected[index]);
        double exact_difference = fabs((double)actual[index] - exact[index]);
        double approximation = 0.1 * (1.0 + fabs((double)exact[index]));
        YVEX_TEST_ASSERT(difference <= 1e-5 * (1.0 + fabs((double)expected[index])),
                         "Q8 activation CUDA row batch matches independent codec reference");
        YVEX_TEST_ASSERT(exact_difference <= approximation,
                         "Q8 activation matvec remains within bounded execution approximation");
    }
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
                     "Q8 activation matvec releases all CUDA ownership");
    return 0;
}

static int quant_cuda_tensor(yvex_backend *backend, const char *name,
                             unsigned int dtype, const void *source,
                             unsigned long long bytes,
                             yvex_device_tensor **tensor, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    descriptor.name = name;
    descriptor.dtype = dtype;
    descriptor.rank = 1u;
    descriptor.dims[0] = dtype == YVEX_DTYPE_F32 ? bytes / sizeof(float) : bytes;
    descriptor.bytes = bytes;
    return yvex_backend_tensor_alloc(backend, &descriptor, tensor, err) == YVEX_OK &&
           (!source || yvex_backend_tensor_write(
                           backend, *tensor, source, bytes, err) == YVEX_OK);
}

static int quant_cuda_transformer_facts(yvex_backend *backend)
{
    enum { TOKENS = 2, HIDDEN = 32, STREAMS = 2 };
    yvex_device_tensor *encoded_device = NULL, *embedding_device = NULL;
    yvex_device_tensor *expanded_device = NULL, *function_device = NULL;
    yvex_device_tensor *base_device = NULL, *scale_device = NULL;
    yvex_device_tensor *norm_device = NULL, *output_device = NULL;
    unsigned char *row = NULL, *encoded = NULL;
    float source[TOKENS * HIDDEN] = {0};
    float embedding[TOKENS * HIDDEN], expanded[TOKENS * HIDDEN * STREAMS];
    float function[STREAMS * STREAMS * HIDDEN] = {0};
    float base[STREAMS] = {0}, scale[1] = {1.0f}, norm[HIDDEN];
    float output[TOKENS * HIDDEN];
    yvex_backend_cuda_operation_facts facts;
    yvex_error err;
    size_t row_bytes = 0u, current_bytes = 0u;
    unsigned long long index;
    int rc;

    for (index = 0ull; index < HIDDEN; ++index) norm[index] = 1.0f;
    for (index = 0ull; index < TOKENS; ++index) {
        YVEX_TEST_ASSERT(quant_cuda_encode_row(
                             YVEX_GGUF_QTYPE_Q8_0, source + index * HIDDEN,
                             HIDDEN, &row, &current_bytes),
                         "transformer embedding row encodes");
        if (!index) {
            row_bytes = current_bytes;
            encoded = (unsigned char *)malloc(TOKENS * row_bytes);
        }
        YVEX_TEST_ASSERT(encoded && current_bytes == row_bytes,
                         "transformer embedding encoding has stable rows");
        memcpy(encoded + index * row_bytes, row, row_bytes);
        free(row);
        row = NULL;
    }
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "transformer_encoded", YVEX_DTYPE_I8,
                          encoded, TOKENS * row_bytes, &encoded_device, &err) &&
            quant_cuda_tensor(backend, "transformer_embedding", YVEX_DTYPE_F32,
                              NULL, sizeof(embedding), &embedding_device, &err) &&
            quant_cuda_tensor(backend, "transformer_expanded", YVEX_DTYPE_F32,
                              NULL, sizeof(expanded), &expanded_device, &err),
        "transformer initial tensors allocate");
    rc = yvex_backend_transformer_cuda_initial(
        backend, encoded_device, YVEX_GGUF_QTYPE_Q8_0, TOKENS, HIDDEN, STREAMS,
        embedding_device, expanded_device, &facts, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && facts.compulsory_memory_facts_available &&
            facts.active_weight_bytes == TOKENS * row_bytes && !facts.state_bytes &&
            facts.activation_bytes == sizeof(embedding) + sizeof(expanded) &&
            facts.temporary_bytes == sizeof(int) &&
            yvex_backend_tensor_read(backend, embedding_device, embedding,
                                     sizeof(embedding), &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, expanded_device, expanded,
                                     sizeof(expanded), &err) == YVEX_OK,
        "transformer initial reports exact compulsory memory spans");
    for (index = 0ull; index < TOKENS * HIDDEN * STREAMS; ++index)
        YVEX_TEST_ASSERT(expanded[index] == 0.0f,
                         "transformer initial publishes finite repeated streams");

    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "transformer_function", YVEX_DTYPE_F32,
                          function, sizeof(function), &function_device, &err) &&
            quant_cuda_tensor(backend, "transformer_base", YVEX_DTYPE_F32,
                              base, sizeof(base), &base_device, &err) &&
            quant_cuda_tensor(backend, "transformer_scale", YVEX_DTYPE_F32,
                              scale, sizeof(scale), &scale_device, &err) &&
            quant_cuda_tensor(backend, "transformer_norm", YVEX_DTYPE_F32,
                              norm, sizeof(norm), &norm_device, &err) &&
            quant_cuda_tensor(backend, "transformer_output", YVEX_DTYPE_F32,
                              NULL, sizeof(output), &output_device, &err),
        "transformer final tensors allocate");
    rc = yvex_backend_transformer_cuda_final(
        backend, expanded_device, function_device, base_device, scale_device,
        norm_device, TOKENS, HIDDEN, STREAMS, 1e-6, 1e-6, output_device,
        &facts, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && facts.compulsory_memory_facts_available &&
            facts.active_weight_bytes == sizeof(function) + sizeof(base) +
                                             sizeof(scale) + sizeof(norm) &&
            !facts.state_bytes &&
            facts.activation_bytes == sizeof(expanded) + sizeof(output) &&
            facts.temporary_bytes == sizeof(int) &&
            yvex_backend_tensor_read(backend, output_device, output,
                                     sizeof(output), &err) == YVEX_OK,
        "transformer final reports exact compulsory memory spans");
    for (index = 0ull; index < TOKENS * HIDDEN; ++index)
        YVEX_TEST_ASSERT(output[index] == 0.0f,
                         "transformer final publishes finite normalized rows");

    free(encoded);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &output_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &norm_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &scale_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &base_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &function_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &expanded_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &embedding_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &encoded_device, &err) == YVEX_OK,
        "transformer fact tensors release");
    return 0;
}

/* Proves typed geometry, alignment, capability, and failure cleanup refusals. */
static int quant_cuda_refusals(const yvex_backend_options *options)
{
    yvex_backend *backend = NULL;
    yvex_quant_failure failure;
    yvex_error err;
    float source[32] = {0};
    float vector[33] = {0};
    float output = 0.0f;
    unsigned char *encoded = NULL;
    unsigned char *misaligned = NULL;
    size_t encoded_bytes = 0u;
    int rc;

    YVEX_TEST_ASSERT(yvex_backend_open(&backend, options, &err) == YVEX_OK,
                     "CUDA refusal backend opens");
    YVEX_TEST_ASSERT(quant_cuda_encode_row(
                         YVEX_GGUF_QTYPE_Q8_0, source, 32u,
                         &encoded, &encoded_bytes),
                     "CUDA refusal row encodes");
    rc = yvex_cuda_quant_row_dot(
        backend, YVEX_GGUF_QTYPE_Q8_0, encoded, encoded_bytes,
        vector, 33u, &output, &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         failure.code == YVEX_QUANT_FAILURE_ROW_DIVISIBILITY,
                     "CUDA refuses non-divisible qtype row");
    rc = yvex_cuda_quant_row_dot(
        backend, YVEX_GGUF_QTYPE_Q4_K, encoded, encoded_bytes,
        vector, 32u, &output, &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         failure.code ==
                             YVEX_QUANT_FAILURE_CUDA_COMPUTE_UNAVAILABLE,
                     "CUDA refuses qtype without canonical arithmetic");
    misaligned = (unsigned char *)malloc(32u * sizeof(float) + 1u);
    YVEX_TEST_ASSERT(misaligned, "misaligned CUDA refusal storage allocates");
    rc = yvex_cuda_quant_row_dot(
        backend, YVEX_GGUF_QTYPE_F32, misaligned + 1u,
        32u * sizeof(float), vector, 32u, &output, &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         failure.code == YVEX_QUANT_FAILURE_INVALID_ARGUMENT,
                     "CUDA refuses misaligned scalar encoding");

    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_QTYPE_COPY_FAILURE", "input", 1) == 0,
                     "CUDA qtype copy failure seam sets");
    rc = yvex_cuda_quant_row_dot(
        backend, YVEX_GGUF_QTYPE_Q8_0, encoded, encoded_bytes,
        vector, 32u, &output, &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         failure.code == YVEX_QUANT_FAILURE_WORKER,
                     "CUDA input copy failure cleans temporary allocations");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_QTYPE_COPY_FAILURE") == 0,
                     "CUDA qtype copy failure seam clears");
    YVEX_TEST_ASSERT(yvex_cuda_quant_row_dot(
                         backend, YVEX_GGUF_QTYPE_Q8_0, encoded,
                         encoded_bytes, vector, 32u, &output,
                         &failure, &err) == YVEX_OK,
                     "CUDA qtype operation recovers after copy refusal");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_QTYPE_COPY_FAILURE", "output", 1) == 0,
                     "CUDA qtype output-copy failure seam sets");
    rc = yvex_cuda_quant_row_dot(
        backend, YVEX_GGUF_QTYPE_Q8_0, encoded, encoded_bytes,
        vector, 32u, &output, &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         failure.code == YVEX_QUANT_FAILURE_WORKER,
                     "CUDA output copy failure cleans temporary allocations");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_QTYPE_COPY_FAILURE") == 0,
                     "CUDA qtype output-copy failure seam clears");
    YVEX_TEST_ASSERT(yvex_cuda_quant_row_dot(
                         backend, YVEX_GGUF_QTYPE_Q8_0, encoded,
                         encoded_bytes, vector, 32u, &output,
                         &failure, &err) == YVEX_OK,
                     "CUDA qtype operation recovers after output-copy refusal");
    yvex_backend_close(backend);

    backend = NULL;
    YVEX_TEST_ASSERT(yvex_backend_open(&backend, options, &err) == YVEX_OK,
                     "CUDA launch-failure backend opens");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_LAUNCH_FAILURE",
                            "qtype-row-dot", 1) == 0,
                     "CUDA qtype launch failure seam sets");
    rc = yvex_cuda_quant_row_dot(
        backend, YVEX_GGUF_QTYPE_Q8_0, encoded, encoded_bytes,
        vector, 32u, &output, &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         failure.code == YVEX_QUANT_FAILURE_WORKER,
                     "CUDA launch failure cleans and demotes qtype variant");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_LAUNCH_FAILURE") == 0,
                     "CUDA qtype launch failure seam clears");
    yvex_backend_close(backend);

    backend = NULL;
    YVEX_TEST_ASSERT(yvex_backend_open(&backend, options, &err) == YVEX_OK,
                     "CUDA cleanup-failure backend opens");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_CLEANUP_FAILURE",
                            "qtype-row-dot", 1) == 0,
                     "CUDA qtype cleanup failure seam sets");
    rc = yvex_cuda_quant_row_dot(
        backend, YVEX_GGUF_QTYPE_Q8_0, encoded, encoded_bytes,
        vector, 32u, &output, &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         failure.code == YVEX_QUANT_FAILURE_CLEANUP,
                     "CUDA cleanup refusal is typed and fail-closed");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_CLEANUP_FAILURE") == 0,
                     "CUDA qtype cleanup failure seam clears");
    yvex_backend_close(backend);

    free(misaligned);
    free(encoded);
    return 0;
}

int yvex_cuda_test_quant_qtype(void)
{
    static const struct {
        unsigned int qtype;
        unsigned long long elements;
    } cases[] = {
        {YVEX_GGUF_QTYPE_F32, 64u},
        {YVEX_GGUF_QTYPE_F16, 64u},
        {YVEX_GGUF_QTYPE_BF16, 64u},
        {YVEX_GGUF_QTYPE_I32, 64u},
        {YVEX_GGUF_QTYPE_Q8_0, 64u},
        {YVEX_GGUF_QTYPE_Q2_K, 512u},
        {YVEX_GGUF_QTYPE_IQ2_XXS, 512u},
        {YVEX_GGUF_QTYPE_MXFP4, 64u}
    };
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_error err;
    unsigned int index;
    unsigned int row;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    YVEX_TEST_ASSERT(yvex_backend_open(&backend, &options, &err) == YVEX_OK,
                     "CUDA qtype parity backend opens");
    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        double maximum_difference = 0.0;
        double maximum_relative_difference = 0.0;
        for (row = 0u; row < 2u; ++row)
            YVEX_TEST_ASSERT(quant_cuda_parity(
                                 backend, cases[index].qtype,
                                 cases[index].elements, row,
                                 &maximum_difference,
                                 &maximum_relative_difference) == 0,
                             "canonical CUDA qtype multi-row parity case");
        fprintf(stderr,
                "cuda qtype %s max_abs_difference=%.9g "
                "max_relative_difference=%.9g\n",
                yvex_gguf_qtype_name(cases[index].qtype),
                maximum_difference, maximum_relative_difference);
    }
    YVEX_TEST_ASSERT(quant_cuda_q8_matvec(backend, YVEX_GGUF_QTYPE_Q8_0) == 0,
                     "Q8_0 production Q8 activation matvec");
    YVEX_TEST_ASSERT(quant_cuda_q8_matvec(backend, YVEX_GGUF_QTYPE_Q2_K) == 0,
                     "Q2_K production Q8 activation matvec");
    YVEX_TEST_ASSERT(quant_cuda_q8_matvec(backend, YVEX_GGUF_QTYPE_IQ2_XXS) == 0,
                     "IQ2_XXS production Q8 activation matvec");
    YVEX_TEST_ASSERT(quant_cuda_transformer_facts(backend) == 0,
                     "transformer envelope physical facts");
    yvex_backend_close(backend);
    return quant_cuda_refusals(&options);
}
