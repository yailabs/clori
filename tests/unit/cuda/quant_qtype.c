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

static void quant_q8_reference(const float *input, float *output,
                               unsigned int width)
{
    unsigned int block;
    for (block = 0u; block < width / 256u; ++block) {
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
    enum { ROWS = 5, INPUT_ROWS = 3, WIDTH = 4096, HEAD = 173 };
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *resident = NULL, *input = NULL, *additive = NULL, *output = NULL;
    yvex_device_tensor *split_head = NULL, *split_tail = NULL;
    yvex_device_tensor split_output = {0};
    unsigned char *mapped = NULL, *row = NULL;
    float source[ROWS * WIDTH], vectors[INPUT_ROWS * WIDTH];
    float q8_vectors[INPUT_ROWS * WIDTH];
    float exact[INPUT_ROWS * ROWS], expected[INPUT_ROWS * ROWS];
    float additive_values[INPUT_ROWS * ROWS], actual[INPUT_ROWS * ROWS];
    float head[HEAD], tail[WIDTH - HEAD], split_reference[WIDTH];
    float split_expected[ROWS], split_actual[ROWS];
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
                           q8_vectors + input_index * WIDTH, WIDTH);
    }
    for (index = 0ull; index < WIDTH; ++index)
        split_reference[index] = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(vectors[index]));
    memcpy(head, vectors, sizeof(head));
    memcpy(tail, vectors + HEAD, sizeof(tail));
    for (index = 0ull; index < ROWS * WIDTH; ++index)
        source[index] = (float)((int)((index * 7ull + 3ull) % 41ull) - 20) /
                        (float)(2ull + index % 11ull);
    for (index = 0ull; index < INPUT_ROWS * ROWS; ++index)
        additive_values[index] = (float)((int)index - 4) * 0.25f;
    for (row_index = 0u; row_index < ROWS; ++row_index) {
        size_t current_bytes = 0u;
        YVEX_TEST_ASSERT(quant_cuda_encode_row(
                             qtype, source + row_index * WIDTH, WIDTH,
                             &row, &current_bytes),
                         "Q8 activation matvec row encodes");
        if (!row_index) row_bytes = current_bytes;
        YVEX_TEST_ASSERT(current_bytes == row_bytes, "Q8 activation matvec rows share exact geometry");
        if (!mapped) {
            descriptor.name = "q8_activation_encoded";
            descriptor.dtype = YVEX_DTYPE_I8;
            descriptor.rank = 1u;
            descriptor.dims[0] = descriptor.bytes = ROWS * row_bytes;
            YVEX_TEST_ASSERT(yvex_backend_resident_alloc(
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
            if (!input_index)
                YVEX_TEST_ASSERT(yvex_quant_cpu_dot(
                    qtype, mapped + row_index * row_bytes, row_bytes, split_reference,
                    WIDTH, &split_expected[row_index], &failure, &err) == YVEX_OK,
                    "split-input matvec reference uses the canonical codec");
        }
    }
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, mapped, descriptor.bytes, resident, 11ull, &err) == YVEX_OK,
                     "Q8 activation resident matrix attaches");
    descriptor.name = "q8_activation_input";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.dims[0] = INPUT_ROWS * WIDTH;
    descriptor.bytes = sizeof(vectors);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &input, &err) == YVEX_OK,
                     "Q8 activation row batch allocates");
    descriptor.name = "q8_activation_output";
    descriptor.dims[0] = INPUT_ROWS * ROWS;
    descriptor.bytes = sizeof(actual);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &output, &err) == YVEX_OK,
                     "Q8 activation output allocates");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, ROWS * row_bytes, qtype, ROWS, WIDTH, row_bytes,
        INPUT_ROWS, input, NULL, 0ull, NULL, output, 1, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !facts.kernel_launches,
                     "encoded matvec refuses an unpublished input before launch");
    YVEX_TEST_ASSERT(yvex_backend_tensor_write(
                         backend, input, vectors, sizeof(vectors), &err) == YVEX_OK,
                     "Q8 activation row batch uploads once");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, descriptor.bytes ? ROWS * row_bytes : 0u, qtype,
        ROWS, WIDTH, row_bytes, INPUT_ROWS, input, NULL, 0ull, NULL,
        output, 1, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && facts.kernel_launches == 2ull &&
                         facts.tensor_core_launches ==
                             (qtype == YVEX_GGUF_QTYPE_Q8_0 ? 1ull : 0ull) &&
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
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, ROWS * row_bytes, qtype, ROWS, WIDTH, row_bytes,
        INPUT_ROWS, input, NULL, 0ull, NULL, output, 0, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && facts.kernel_launches == 1ull &&
                         facts.tensor_core_launches == 0ull &&
                         facts.temporary_bytes == sizeof(int) &&
                         yvex_backend_tensor_read(
                             backend, output, actual, sizeof(actual), &err) == YVEX_OK,
                     "F32 activation policy selects one uncompressed projection launch");
    for (index = 0ull; index < INPUT_ROWS * ROWS; ++index)
        YVEX_TEST_ASSERT(fabs((double)actual[index] - exact[index]) <=
                                 1e-5 * (1.0 + fabs((double)exact[index])),
                             "F32 activation CUDA row batch matches the reference tolerance");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, ROWS * row_bytes, qtype, ROWS, WIDTH, row_bytes,
        INPUT_ROWS, input, NULL, 0ull, NULL, output, 2, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !facts.kernel_launches,
                     "encoded matvec refuses an unknown activation policy before launch");
    descriptor.name = "q8_activation_additive";
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &additive, &err) == YVEX_OK,
                     "Q8 activation additive row batch allocates");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, ROWS * row_bytes, qtype, ROWS, WIDTH, row_bytes,
        INPUT_ROWS, input, NULL, 0ull, additive, output, 1, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !facts.kernel_launches,
                     "fused encoded matvec refuses an unpublished additive before launch");
    YVEX_TEST_ASSERT(yvex_backend_tensor_write(backend, additive, additive_values,
                                               sizeof(additive_values), &err) == YVEX_OK,
                     "Q8 activation additive row batch uploads once");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, ROWS * row_bytes, qtype, ROWS, WIDTH, row_bytes,
        INPUT_ROWS, input, NULL, 0ull, additive, output, 1, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && facts.kernel_launches == 2ull &&
                         facts.tensor_core_launches ==
                             (qtype == YVEX_GGUF_QTYPE_Q8_0 ? 1ull : 0ull) &&
                         facts.d2h_bytes == sizeof(int) &&
                         facts.device_synchronizations == 1ull &&
                         facts.activation_bytes == sizeof(vectors) + 2ull * sizeof(actual) &&
                         yvex_backend_tensor_read(backend, output, actual,
                                                  sizeof(actual), &err) == YVEX_OK,
                     "fused encoded matvec adds resident rows without another launch");
    for (index = 0ull; index < INPUT_ROWS * ROWS; ++index)
        YVEX_TEST_ASSERT(fabs((double)actual[index] - expected[index] - additive_values[index]) <=
                             1e-5 * (1.0 + fabs((double)expected[index])),
                         "fused encoded matvec matches the independent additive reference");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, ROWS * row_bytes, qtype, ROWS, WIDTH, row_bytes,
        INPUT_ROWS, input, NULL, 0ull, output, output, 1, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !facts.kernel_launches,
                     "fused encoded matvec refuses aliased additive and output ownership");
    descriptor.name = "split_head";
    descriptor.dims[0] = HEAD;
    descriptor.bytes = sizeof(head);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &split_head, &err) == YVEX_OK &&
        yvex_backend_tensor_write(backend, split_head, head, sizeof(head), &err) == YVEX_OK,
                     "split-input head becomes device resident");
    descriptor.name = "split_tail";
    descriptor.dims[0] = WIDTH - HEAD;
    descriptor.bytes = sizeof(tail);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &split_tail, &err) == YVEX_OK,
                     "split-input tail allocates");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, ROWS * row_bytes, qtype, ROWS, WIDTH, row_bytes, 1ull,
        split_head, split_tail, HEAD, NULL, output, 1, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !facts.kernel_launches,
                     "split-input projection refuses an unpublished tail");
    YVEX_TEST_ASSERT(yvex_backend_tensor_write(
                         backend, split_tail, tail, sizeof(tail), &err) == YVEX_OK,
                     "split-input tail becomes device resident");
    YVEX_TEST_ASSERT(yvex_backend_tensor_f32_subview(output, 0ull, ROWS, &split_output),
                     "split-input projection owns one bounded output view");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, ROWS * row_bytes, qtype, ROWS, WIDTH, row_bytes, 1ull,
        split_head, split_tail, HEAD, NULL, &split_output, 1, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "split-input encoded projection executes");
    YVEX_TEST_ASSERT(facts.kernel_launches == 1ull &&
                         facts.activation_bytes == sizeof(head) + sizeof(tail) +
                                                       sizeof(split_actual),
                     "split-input projection accounts one direct launch");
    YVEX_TEST_ASSERT(yvex_backend_tensor_read(backend, &split_output, split_actual,
                                              sizeof(split_actual), &err) == YVEX_OK,
                     "split-input projection publishes its bounded rows");
    for (index = 0ull; index < ROWS; ++index)
        YVEX_TEST_ASSERT(fabs((double)split_actual[index] - split_expected[index]) <=
                             1e-4 * (1.0 + fabs((double)split_expected[index])),
                         "split-input CUDA projection matches the independent reference");
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &split_tail, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &split_head, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &additive, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
                     "Q8 activation matvec releases all CUDA ownership");
    return 0;
}

static int quant_cuda_q8_grouped_matvec(yvex_backend *backend,
                                        unsigned int qtype,
                                        unsigned int width)
{
    enum { ROWS = 2, MAX_WIDTH = 1536 };
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *resident = NULL, *input = NULL, *output = NULL;
    unsigned char *mapped = NULL, *encoded = NULL;
    float source[ROWS * MAX_WIDTH], vector[MAX_WIDTH], q8_vector[MAX_WIDTH];
    float expected[ROWS], actual[ROWS];
    yvex_backend_cuda_operation_facts facts;
    yvex_quant_failure failure;
    yvex_error err;
    size_t row_bytes = 0u;
    unsigned int row;
    unsigned long long index;

    YVEX_TEST_ASSERT(width == 768u || width == 1536u,
                     "grouped Q8 activation width is an admitted short-row shape");
    for (index = 0ull; index < width; ++index)
        vector[index] = (float)((int)(index % 31ull) - 15) /
                        (float)(11ull + index % 3ull);
    quant_q8_reference(vector, q8_vector, width);
    for (row = 0u; row < ROWS; ++row) {
        size_t current_bytes = 0u;
        for (index = 0ull; index < width; ++index)
            source[row * width + index] =
                (float)((int)((index * 5ull + row * 7ull) % 37ull) - 18) /
                (float)(3ull + index % 7ull);
        YVEX_TEST_ASSERT(quant_cuda_encode_row(
                             qtype, source + row * width, width,
                             &encoded, &current_bytes),
                         "grouped Q8 activation row encodes");
        if (!row) row_bytes = current_bytes;
        YVEX_TEST_ASSERT(current_bytes == row_bytes,
                         "grouped Q8 activation rows share geometry");
        if (!mapped) {
            descriptor.name = "q8_grouped_encoded";
            descriptor.dtype = YVEX_DTYPE_I8;
            descriptor.rank = 1u;
            descriptor.dims[0] = descriptor.bytes = ROWS * row_bytes;
            YVEX_TEST_ASSERT(yvex_backend_resident_alloc(
                                 backend, &descriptor, &resident, &mapped, &err) == YVEX_OK,
                             "grouped Q8 activation matrix allocates");
        }
        memcpy(mapped + row * row_bytes, encoded, row_bytes);
        free(encoded);
        encoded = NULL;
        YVEX_TEST_ASSERT(yvex_quant_cpu_dot(
                             qtype, mapped + row * row_bytes, row_bytes,
                             q8_vector, width, &expected[row], &failure, &err) == YVEX_OK,
                         "grouped Q8 activation reference succeeds");
    }
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, mapped, ROWS * row_bytes, resident, 17ull, &err) == YVEX_OK,
                     "grouped Q8 activation matrix attaches");
    descriptor.name = "q8_grouped_input";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.dims[0] = width;
    descriptor.bytes = width * sizeof(float);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &input, &err) == YVEX_OK &&
                         yvex_backend_tensor_write(
                             backend, input, vector, descriptor.bytes, &err) == YVEX_OK,
                     "grouped Q8 activation input becomes resident");
    descriptor.name = "q8_grouped_output";
    descriptor.dims[0] = ROWS;
    descriptor.bytes = sizeof(actual);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &output, &err) == YVEX_OK,
                     "grouped Q8 activation output allocates");
    YVEX_TEST_ASSERT(yvex_backend_cuda_encoded_matvec(
                         backend, mapped, ROWS * row_bytes, qtype, ROWS, width,
                         row_bytes, 1ull, input, NULL, 0ull, NULL, output, 1,
                         &facts, &err) == YVEX_OK &&
                         facts.kernel_launches == 2ull &&
                         yvex_backend_tensor_read(
                             backend, output, actual, sizeof(actual), &err) == YVEX_OK,
                     "grouped Q8 activation projection executes once");
    for (row = 0u; row < ROWS; ++row)
        YVEX_TEST_ASSERT(fabs((double)actual[row] - expected[row]) <=
                                 1e-5 * (1.0 + fabs((double)expected[row])),
                             "grouped Q8 activation preserves the canonical block result");
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
                     "grouped Q8 activation releases CUDA ownership");
    return 0;
}

static int quant_cuda_bf16_gemm(yvex_backend *backend)
{
    enum { ROWS = 5, INPUT_ROWS = 3, WIDTH = 64 };
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *resident = NULL, *input = NULL, *output = NULL;
    unsigned char *mapped = NULL, *encoded_row = NULL;
    float weights[ROWS * WIDTH], inputs[INPUT_ROWS * WIDTH];
    float expected[INPUT_ROWS * ROWS], actual[INPUT_ROWS * ROWS];
    yvex_backend_cuda_operation_facts facts;
    yvex_error err;
    size_t row_bytes = 0u;
    unsigned long long row, input_row, column;
    int rc;

    YVEX_TEST_ASSERT(state && state->blas.ready,
                     "cuBLAS mixed projection is admitted on the CUDA host");
    for (column = 0ull; column < ROWS * WIDTH; ++column)
        weights[column] = (float)((int)((column * 7ull + 3ull) % 29ull) - 14) /
                          (float)(3ull + column % 7ull);
    for (column = 0ull; column < INPUT_ROWS * WIDTH; ++column)
        inputs[column] = (float)((int)((column * 5ull + 1ull) % 23ull) - 11) /
                         (float)(5ull + column % 3ull);
    for (row = 0ull; row < ROWS; ++row) {
        size_t current_bytes = 0u;
        YVEX_TEST_ASSERT(quant_cuda_encode_row(
                             YVEX_GGUF_QTYPE_BF16, weights + row * WIDTH, WIDTH,
                             &encoded_row, &current_bytes),
                         "BF16 GEMM row encodes canonically");
        if (!row) {
            row_bytes = current_bytes;
            descriptor.name = "bf16_gemm_resident";
            descriptor.dtype = YVEX_DTYPE_I8;
            descriptor.rank = 1u;
            descriptor.dims[0] = descriptor.bytes = ROWS * row_bytes;
            YVEX_TEST_ASSERT(yvex_backend_resident_alloc(
                                 backend, &descriptor, &resident, &mapped, &err) == YVEX_OK,
                             "BF16 GEMM resident matrix allocates");
        }
        YVEX_TEST_ASSERT(current_bytes == row_bytes,
                         "BF16 GEMM rows share exact geometry");
        memcpy(mapped + row * row_bytes, encoded_row, row_bytes);
        free(encoded_row);
        encoded_row = NULL;
    }
    for (input_row = 0ull; input_row < INPUT_ROWS; ++input_row)
        for (row = 0ull; row < ROWS; ++row) {
            double value = 0.0;
            for (column = 0ull; column < WIDTH; ++column) {
                unsigned short bits;
                memcpy(&bits, mapped + row * row_bytes + column * sizeof(bits), sizeof(bits));
                value += (double)yvex_quant_bf16_decode(bits) *
                         (double)yvex_quant_bf16_decode(yvex_quant_bf16_encode(
                             inputs[input_row * WIDTH + column]));
            }
            expected[input_row * ROWS + row] = (float)value;
        }
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, mapped, descriptor.bytes, resident, 17ull, &err) == YVEX_OK,
                     "BF16 GEMM resident matrix attaches");
    descriptor.name = "bf16_gemm_input";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.dims[0] = INPUT_ROWS * WIDTH;
    descriptor.bytes = sizeof(inputs);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &input, &err) == YVEX_OK &&
                         yvex_backend_tensor_write(
                             backend, input, inputs, sizeof(inputs), &err) == YVEX_OK,
                     "BF16 GEMM input becomes device resident");
    descriptor.name = "bf16_gemm_output";
    descriptor.dims[0] = INPUT_ROWS * ROWS;
    descriptor.bytes = sizeof(actual);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &output, &err) == YVEX_OK,
                     "BF16 GEMM output allocates");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, ROWS * row_bytes, YVEX_GGUF_QTYPE_BF16,
        ROWS, WIDTH, row_bytes, INPUT_ROWS, input, NULL, 0ull,
        NULL, output, 1, &facts, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "BF16 cuBLAS refusal: %s (%s)\n",
                yvex_error_message(&err), yvex_error_where(&err));
    YVEX_TEST_ASSERT(rc == YVEX_OK && facts.kernel_launches == 2ull &&
                         facts.d2h_bytes == sizeof(int) &&
                         facts.temporary_bytes ==
                             INPUT_ROWS * WIDTH * sizeof(unsigned short) + sizeof(int) &&
                         facts.device_synchronizations == 1ull,
                     "BF16 row batch selects one pack plus mixed cuBLAS GEMM");
    YVEX_TEST_ASSERT(yvex_backend_tensor_read(
                         backend, output, actual, sizeof(actual), &err) == YVEX_OK,
                     "BF16 GEMM result downloads");
    for (column = 0ull; column < INPUT_ROWS * ROWS; ++column)
        YVEX_TEST_ASSERT(fabs((double)actual[column] - expected[column]) <=
                             2e-4 * (1.0 + fabs((double)expected[column])),
                         "mixed cuBLAS GEMM matches the decoded BF16 reference");
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
                     "BF16 GEMM releases all CUDA ownership");
    return 0;
}

static int quant_cuda_f32_gemm(yvex_backend *backend)
{
    enum { ROWS = 7, INPUT_ROWS = 4, WIDTH = 96 };
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *resident = NULL, *input = NULL, *output = NULL;
    unsigned char *mapped = NULL;
    float weights[ROWS * WIDTH], inputs[INPUT_ROWS * WIDTH];
    float expected[INPUT_ROWS * ROWS], actual[INPUT_ROWS * ROWS];
    yvex_backend_cuda_operation_facts facts;
    yvex_error err;
    unsigned long long row, input_row, column;
    int rc;

    YVEX_TEST_ASSERT(state && state->blas.ready,
                     "cuBLAS F32 projection is admitted on the CUDA host");
    for (column = 0ull; column < ROWS * WIDTH; ++column)
        weights[column] = (float)((int)((column * 11ull + 5ull) % 41ull) - 20) /
                          (float)(7ull + column % 5ull);
    for (column = 0ull; column < INPUT_ROWS * WIDTH; ++column)
        inputs[column] = (float)((int)((column * 3ull + 2ull) % 31ull) - 15) /
                         (float)(6ull + column % 7ull);
    for (input_row = 0ull; input_row < INPUT_ROWS; ++input_row)
        for (row = 0ull; row < ROWS; ++row) {
            double value = 0.0;
            for (column = 0ull; column < WIDTH; ++column)
                value += (double)weights[row * WIDTH + column] *
                         (double)inputs[input_row * WIDTH + column];
            expected[input_row * ROWS + row] = (float)value;
        }
    descriptor.name = "f32_gemm_resident";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = sizeof(weights);
    YVEX_TEST_ASSERT(backend->vtable->resident_alloc(
                         backend, &descriptor, &resident, &mapped, &err) == YVEX_OK,
                     "F32 GEMM resident matrix allocates");
    memcpy(mapped, weights, sizeof(weights));
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, mapped, sizeof(weights), resident, 19ull, &err) == YVEX_OK,
                     "F32 GEMM resident matrix attaches");
    descriptor.name = "f32_gemm_input";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.dims[0] = INPUT_ROWS * WIDTH;
    descriptor.bytes = sizeof(inputs);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &input, &err) == YVEX_OK &&
                         yvex_backend_tensor_write(
                             backend, input, inputs, sizeof(inputs), &err) == YVEX_OK,
                     "F32 GEMM input becomes device resident");
    descriptor.name = "f32_gemm_output";
    descriptor.dims[0] = INPUT_ROWS * ROWS;
    descriptor.bytes = sizeof(actual);
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &output, &err) == YVEX_OK,
                     "F32 GEMM output allocates");
    rc = yvex_backend_cuda_encoded_matvec(
        backend, mapped, sizeof(weights), YVEX_GGUF_QTYPE_F32,
        ROWS, WIDTH, WIDTH * sizeof(float), INPUT_ROWS, input, NULL, 0ull,
        NULL, output, 0, &facts, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "F32 cuBLAS refusal: %s (%s)\n",
                yvex_error_message(&err), yvex_error_where(&err));
    YVEX_TEST_ASSERT(rc == YVEX_OK && facts.kernel_launches == 1ull &&
                         !facts.d2h_bytes && !facts.temporary_bytes &&
                         facts.device_synchronizations == 1ull,
                     "F32 row batch selects one cuBLAS GEMM");
    YVEX_TEST_ASSERT(yvex_backend_tensor_read(
                         backend, output, actual, sizeof(actual), &err) == YVEX_OK,
                     "F32 GEMM result downloads");
    for (column = 0ull; column < INPUT_ROWS * ROWS; ++column)
        YVEX_TEST_ASSERT(fabs((double)actual[column] - expected[column]) <=
                             2e-5 * (1.0 + fabs((double)expected[column])),
                         "cuBLAS F32 GEMM matches the independent reference");
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
                     "F32 GEMM releases all CUDA ownership");
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

static int quant_cuda_encoded_gather(yvex_backend *backend)
{
    enum { ROWS = 4, SELECTED = 3, WIDTH = 64 };
    static const unsigned int row_ids[SELECTED] = {3u, 1u, 3u};
    static const unsigned int invalid_ids[SELECTED] = {3u, 4u, 1u};
    const yvex_gguf_qtype_geometry *geometry =
        yvex_gguf_qtype_geometry_find(YVEX_GGUF_QTYPE_Q8_0);
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *resident = NULL, *output = NULL;
    unsigned char *mapped = NULL, *encoded_row = NULL;
    float source[ROWS * WIDTH], expected[SELECTED * WIDTH], actual[SELECTED * WIDTH];
    yvex_backend_cuda_operation_facts facts;
    yvex_quant_failure failure;
    yvex_error err;
    size_t row_bytes = 0u;
    unsigned long long row, block, index;
    int rc;

    YVEX_TEST_ASSERT(geometry, "encoded gather qtype geometry resolves");
    for (index = 0ull; index < ROWS * WIDTH; ++index)
        source[index] = (float)((int)((index * 5ull + 7ull) % 37ull) - 18) /
                        (float)(3ull + index % 5ull);
    for (row = 0ull; row < ROWS; ++row) {
        size_t current_bytes = 0u;
        YVEX_TEST_ASSERT(quant_cuda_encode_row(
                             YVEX_GGUF_QTYPE_Q8_0, source + row * WIDTH, WIDTH,
                             &encoded_row, &current_bytes),
                         "encoded gather row encodes canonically");
        if (!row) {
            row_bytes = current_bytes;
            descriptor.name = "encoded_gather_resident";
            descriptor.dtype = YVEX_DTYPE_I8;
            descriptor.rank = 1u;
            descriptor.dims[0] = descriptor.bytes = ROWS * row_bytes;
            YVEX_TEST_ASSERT(yvex_backend_resident_alloc(
                                 backend, &descriptor, &resident, &mapped, &err) == YVEX_OK,
                             "encoded gather resident matrix allocates");
        }
        YVEX_TEST_ASSERT(current_bytes == row_bytes,
                         "encoded gather rows share exact geometry");
        memcpy(mapped + row * row_bytes, encoded_row, row_bytes);
        free(encoded_row);
        encoded_row = NULL;
    }
    for (row = 0ull; row < SELECTED; ++row) {
        for (block = 0ull; block < WIDTH / geometry->block_size; ++block) {
            YVEX_TEST_ASSERT(
                yvex_quant_decode_block(
                    YVEX_GGUF_QTYPE_Q8_0,
                    mapped + (unsigned long long)row_ids[row] * row_bytes +
                        block * geometry->bytes_per_block,
                    geometry->bytes_per_block,
                    expected + row * WIDTH + block * geometry->block_size,
                    geometry->block_size, &failure, &err) == YVEX_OK,
                "encoded gather independent row decode succeeds");
        }
    }
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "encoded_gather_output", YVEX_DTYPE_F32,
                          NULL, sizeof(actual), &output, &err),
        "encoded gather output allocates");
    rc = yvex_backend_cuda_encoded_gather(
        backend, mapped, ROWS * row_bytes, YVEX_GGUF_QTYPE_Q8_0,
        ROWS, WIDTH, row_bytes, row_ids, SELECTED, output, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !facts.kernel_launches,
                     "encoded gather refuses a matrix outside resident authority");
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, mapped, ROWS * row_bytes, resident, 17ull, &err) == YVEX_OK,
                     "encoded gather resident matrix attaches");
    rc = yvex_backend_cuda_encoded_gather(
        backend, mapped, ROWS * row_bytes, YVEX_GGUF_QTYPE_Q8_0,
        ROWS, WIDTH, row_bytes, row_ids, SELECTED, output, &facts, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && facts.h2d_bytes == sizeof(row_ids) &&
            facts.d2h_bytes == sizeof(int) && facts.kernel_launches == 1ull &&
            facts.upload_count == 1ull && facts.download_count == 1ull &&
            facts.device_synchronizations == 1ull &&
            facts.active_weight_bytes == SELECTED * row_bytes &&
            facts.activation_bytes == sizeof(actual) &&
            facts.temporary_bytes == sizeof(row_ids) + sizeof(int) &&
            facts.compulsory_memory_facts_available &&
            yvex_backend_tensor_read(backend, output, actual, sizeof(actual), &err) == YVEX_OK,
        "encoded gather publishes exact rows and compulsory physical facts");
    for (index = 0ull; index < SELECTED * WIDTH; ++index)
        YVEX_TEST_ASSERT(actual[index] == expected[index],
                         "encoded gather matches the independent qtype decoder exactly");
    rc = yvex_backend_cuda_encoded_gather(
        backend, mapped, ROWS * row_bytes, YVEX_GGUF_QTYPE_Q8_0,
        ROWS, WIDTH, row_bytes, invalid_ids, SELECTED, output, &facts, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS && !output->is_written && !facts.kernel_launches,
                     "encoded gather refuses an invalid row before publication or launch");
    YVEX_TEST_ASSERT(
        yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
        "encoded gather releases resident and output ownership");
    return 0;
}

static int quant_cuda_transformer_facts(yvex_backend *backend)
{
    enum { TOKENS = 2, HIDDEN = 32, STREAMS = 2 };
    yvex_device_tensor *encoded_device = NULL, *embedding_device = NULL;
    yvex_device_tensor *expanded_device = NULL, *function_device = NULL;
    yvex_device_tensor *base_device = NULL, *scale_device = NULL;
    yvex_device_tensor *norm_device = NULL, *pre_device = NULL, *output_device = NULL;
    yvex_device_tensor *feature_device = NULL, *resident_feature_device = NULL;
    unsigned char *row = NULL, *encoded = NULL;
    float source[TOKENS * HIDDEN] = {0};
    float embedding[TOKENS * HIDDEN], expanded[TOKENS * HIDDEN * STREAMS];
    float function[STREAMS * STREAMS * HIDDEN] = {0};
    float base[STREAMS] = {0}, scale[1] = {1.0f}, norm[HIDDEN];
    float pre[TOKENS * HIDDEN], output[TOKENS * HIDDEN], features[TOKENS * HIDDEN];
    float resident_features[TOKENS * HIDDEN * 2] = {0};
    float reference_pre[TOKENS * HIDDEN], reference_output[TOKENS * HIDDEN];
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

    for (index = 0ull; index < TOKENS * HIDDEN * STREAMS; ++index)
        expanded[index] = (float)((index / HIDDEN) * 4ull + index % HIDDEN);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_write(backend, expanded_device, expanded,
                                  sizeof(expanded), &err) == YVEX_OK &&
            quant_cuda_tensor(backend, "transformer_features", YVEX_DTYPE_F32,
                              NULL, sizeof(features), &feature_device, &err) &&
            quant_cuda_tensor(backend, "transformer_resident_features", YVEX_DTYPE_F32,
                              resident_features, sizeof(resident_features),
                              &resident_feature_device, &err),
        "transformer feature tensors prepare");
    YVEX_TEST_ASSERT(
        yvex_backend_transformer_cuda_feature_mean(
            backend, expanded_device, TOKENS, HIDDEN, STREAMS, feature_device,
            feature_device, 0ull, HIDDEN, 0ull, features, &facts, &err) ==
            YVEX_ERR_FORMAT,
        "transformer feature mean refuses aliased compact and resident publication");
    rc = yvex_backend_transformer_cuda_feature_mean(
        backend, expanded_device, TOKENS, HIDDEN, STREAMS, feature_device,
        resident_feature_device, 0ull, HIDDEN * 2ull, HIDDEN,
        features, &facts, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && facts.compulsory_memory_facts_available &&
            !facts.active_weight_bytes && !facts.state_bytes &&
            facts.activation_bytes == sizeof(expanded) + 2ull * sizeof(features) &&
            facts.temporary_bytes == sizeof(int) &&
            facts.d2h_bytes == sizeof(features) + sizeof(int) &&
            facts.kernel_launches == 1ull && facts.download_count == 2ull &&
            facts.device_synchronizations == 1ull && feature_device->is_written &&
            resident_feature_device->is_written &&
            yvex_backend_tensor_read(backend, resident_feature_device,
                                     resident_features,
                                     sizeof(resident_features), &err) == YVEX_OK,
        "transformer feature mean reports bounded physical facts");
    for (index = 0ull; index < TOKENS * HIDDEN; ++index) {
        unsigned long long token = index / HIDDEN, lane = index % HIDDEN;
        float expected = (float)(token * STREAMS * 4ull + lane + 2ull);
        YVEX_TEST_ASSERT(features[index] == expected,
                         "transformer feature mean matches independent reduction");
        YVEX_TEST_ASSERT(resident_features[token * HIDDEN * 2ull + lane] == 0.0f &&
                             resident_features[token * HIDDEN * 2ull + HIDDEN + lane] == expected,
                         "transformer feature mean publishes the requested resident columns");
    }
    memset(features, 0, sizeof(features));
    memset(resident_features, 0, sizeof(resident_features));
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_write(backend, resident_feature_device,
                                  resident_features, sizeof(resident_features), &err) == YVEX_OK &&
            yvex_backend_transformer_cuda_feature_mean(
                backend, expanded_device, TOKENS, HIDDEN, STREAMS, feature_device,
                resident_feature_device, 0ull, HIDDEN * 2ull, HIDDEN,
                NULL, &facts, &err) == YVEX_OK &&
            facts.d2h_bytes == sizeof(int) && facts.download_count == 1ull &&
            facts.kernel_launches == 1ull && facts.device_synchronizations == 1ull &&
            yvex_backend_tensor_read(backend, feature_device, features,
                                     sizeof(features), &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, resident_feature_device,
                                     resident_features,
                                     sizeof(resident_features), &err) == YVEX_OK,
        "transformer feature mean publishes device-only rows with bounded status transfer");
    for (index = 0ull; index < TOKENS * HIDDEN; ++index) {
        unsigned long long token = index / HIDDEN, lane = index % HIDDEN;
        float expected = (float)(token * STREAMS * 4ull + lane + 2ull);
        YVEX_TEST_ASSERT(features[index] == expected &&
                             resident_features[token * HIDDEN * 2ull + HIDDEN + lane] == expected,
                         "device-only feature mean retains exact compact and resident values");
    }
    expanded[0] = NAN;
    features[0] = 77.0f;
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_write(backend, expanded_device, expanded,
                                  sizeof(expanded), &err) == YVEX_OK &&
            yvex_backend_transformer_cuda_feature_mean(
                backend, expanded_device, TOKENS, HIDDEN, STREAMS, feature_device,
                resident_feature_device, 0ull, HIDDEN * 2ull, HIDDEN,
                features, &facts, &err) == YVEX_ERR_FORMAT &&
            features[0] == 77.0f && !feature_device->is_written &&
            !resident_feature_device->is_written,
        "transformer feature mean refuses non-finite output before publication");
    expanded[0] = 0.0f;
    YVEX_TEST_ASSERT(yvex_backend_tensor_write(
                         backend, expanded_device, expanded, sizeof(expanded), &err) == YVEX_OK,
                     "transformer final input resets");
    for (index = 0ull; index < TOKENS; ++index) {
        unsigned long long lane, stream;
        double squares = 0.0;
        for (lane = 0ull; lane < HIDDEN; ++lane) {
            double collapsed = 0.0;
            for (stream = 0ull; stream < STREAMS; ++stream)
                collapsed += (0.5 + 1e-6) *
                    expanded[(index * STREAMS + stream) * HIDDEN + lane];
            reference_pre[index * HIDDEN + lane] = yvex_quant_bf16_decode(
                yvex_quant_bf16_encode((float)collapsed));
            squares += (double)reference_pre[index * HIDDEN + lane] *
                       reference_pre[index * HIDDEN + lane];
        }
        for (lane = 0ull; lane < HIDDEN; ++lane)
            reference_output[index * HIDDEN + lane] = yvex_quant_bf16_decode(
                yvex_quant_bf16_encode((float)(
                    reference_pre[index * HIDDEN + lane] /
                    sqrt(squares / (double)HIDDEN + 1e-6))));
    }

    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "transformer_function", YVEX_DTYPE_F32,
                          function, sizeof(function), &function_device, &err) &&
            quant_cuda_tensor(backend, "transformer_base", YVEX_DTYPE_F32,
                              base, sizeof(base), &base_device, &err) &&
            quant_cuda_tensor(backend, "transformer_scale", YVEX_DTYPE_F32,
                              scale, sizeof(scale), &scale_device, &err) &&
            quant_cuda_tensor(backend, "transformer_norm", YVEX_DTYPE_F32,
                              norm, sizeof(norm), &norm_device, &err) &&
            quant_cuda_tensor(backend, "transformer_pre", YVEX_DTYPE_F32,
                              NULL, sizeof(pre), &pre_device, &err) &&
            quant_cuda_tensor(backend, "transformer_output", YVEX_DTYPE_F32,
                              NULL, sizeof(output), &output_device, &err),
        "transformer final tensors allocate");
    YVEX_TEST_ASSERT(
        yvex_backend_transformer_cuda_final(
            backend, expanded_device, function_device, base_device, scale_device,
            norm_device, TOKENS, HIDDEN, STREAMS, 1e-6, 1e-6, output_device,
            output_device, &facts, &err) == YVEX_ERR_FORMAT &&
            !facts.kernel_launches && !facts.d2h_bytes,
        "transformer final refuses aliased pre-normalized publication");
    rc = yvex_backend_transformer_cuda_final(
        backend, expanded_device, function_device, base_device, scale_device,
        norm_device, TOKENS, HIDDEN, STREAMS, 1e-6, 1e-6, pre_device,
        output_device, &facts, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && facts.compulsory_memory_facts_available &&
            facts.active_weight_bytes == sizeof(function) + sizeof(base) +
                                             sizeof(scale) + sizeof(norm) &&
            !facts.state_bytes &&
            facts.activation_bytes == sizeof(expanded) + sizeof(pre) + sizeof(output) &&
            facts.temporary_bytes == sizeof(int) &&
            yvex_backend_tensor_read(backend, pre_device, pre,
                                     sizeof(pre), &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, output_device, output,
                                     sizeof(output), &err) == YVEX_OK,
        "transformer final reports exact compulsory memory spans");
    for (index = 0ull; index < TOKENS * HIDDEN; ++index)
        YVEX_TEST_ASSERT(pre[index] == reference_pre[index] &&
                             output[index] == reference_output[index],
                         "transformer final preserves pre-normalized and normalized rows");

    free(encoded);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &resident_feature_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &feature_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &output_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &pre_device, &err) == YVEX_OK &&
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

static int quant_cuda_dense_transformer(yvex_backend *backend)
{
    yvex_device_tensor *rotary = NULL, *cosines = NULL, *sines = NULL;
    yvex_device_tensor *query = NULL, *key = NULL, *value = NULL, *attention = NULL;
    yvex_device_tensor *gate = NULL, *up = NULL, *product = NULL;
    yvex_device_tensor *norm_input = NULL, *norm_weight = NULL, *norm_output = NULL;
    float rotary_input[4] = {-1.34375f, -2.0625f, 3.0f, 4.0f};
    float cosine_input[2] = {0.63671875f, 0.63671875f};
    float sine_input[2] = {0.546875f, 0.546875f};
    float rotary_output[4], query_input[8], key_input[4], value_input[4];
    float attention_output[8], gate_input[4] = {-2.0f, -0.5f, 0.5f, 2.0f};
    float up_input[4] = {3.0f, 4.0f, 5.0f, 6.0f}, product_output[4];
    float norm_values[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    float norm_scales[4] = {0.5f, 1.0f, 1.5f, 2.0f}, norm_result[4];
    yvex_backend_cuda_operation_facts facts;
    yvex_error err;
    unsigned long long index;
    int rc;

    for (index = 0ull; index < 4ull; ++index) {
        query_input[index * 2ull] = 1.0f;
        query_input[index * 2ull + 1ull] = 0.0f;
    }
    key_input[0] = 1.0f;
    key_input[1] = 0.0f;
    key_input[2] = 0.0f;
    key_input[3] = 1.0f;
    value_input[0] = 2.0f;
    value_input[1] = 4.0f;
    value_input[2] = 6.0f;
    value_input[3] = 8.0f;
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "rotary", YVEX_DTYPE_F32, rotary_input,
                          sizeof(rotary_input), &rotary, &err) &&
            quant_cuda_tensor(backend, "cosines", YVEX_DTYPE_F32, cosine_input,
                              sizeof(cosine_input), &cosines, &err) &&
            quant_cuda_tensor(backend, "sines", YVEX_DTYPE_F32, sine_input,
                              sizeof(sine_input), &sines, &err),
        "dense transformer rotary tensors allocate");
    rc = yvex_cuda_transformer_rotary_half(
        backend, rotary, cosines, sines, 1ull, 1ull, 4ull, 2ull, &facts, &err);
    {
        float first_cosine = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(rotary_input[0] * cosine_input[0]));
        float second_sine = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(rotary_input[1] * sine_input[0]));
        float second_cosine = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(rotary_input[1] * cosine_input[0]));
        float first_sine = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(rotary_input[0] * sine_input[0]));
        float expected_first = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(first_cosine - second_sine));
        float expected_second = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(second_cosine + first_sine));
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && facts.kernel_launches == 1ull &&
                facts.device_synchronizations == 1ull &&
                yvex_backend_tensor_read(
                    backend, rotary, rotary_output, sizeof(rotary_output), &err) == YVEX_OK &&
                rotary_output[0] == expected_first && rotary_output[1] == expected_second &&
                rotary_output[2] == 3.0f && rotary_output[3] == 4.0f,
            "rotate-half matches staged BF16 products and preserves the non-rotary suffix");
    }
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "norm-input", YVEX_DTYPE_F32, norm_values,
                          sizeof(norm_values), &norm_input, &err) &&
            quant_cuda_tensor(backend, "norm-weight", YVEX_DTYPE_F32, norm_scales,
                              sizeof(norm_scales), &norm_weight, &err) &&
            quant_cuda_tensor(backend, "norm-output", YVEX_DTYPE_F32, NULL,
                              sizeof(norm_result), &norm_output, &err) &&
            yvex_cuda_transformer_rms_norm_bf16(
                backend, norm_input, norm_weight, norm_output, 1ull, 4ull, 1e-6f,
                &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, norm_output, norm_result,
                                     sizeof(norm_result), &err) == YVEX_OK,
        "dense transformer Qwen RMS policy executes");
    for (index = 0ull; index < 4ull; ++index) {
        float inverse = 1.0f / sqrtf(7.5f + 1e-6f);
        float normalized = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(norm_values[index] * inverse));
        float expected = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(normalized * norm_scales[index]));
        YVEX_TEST_ASSERT(norm_result[index] == expected,
                         "dense transformer Qwen RMS preserves the intermediate BF16 cast");
    }
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "query", YVEX_DTYPE_F32, query_input,
                          sizeof(query_input), &query, &err) &&
            quant_cuda_tensor(backend, "key", YVEX_DTYPE_F32, key_input,
                              sizeof(key_input), &key, &err) &&
            quant_cuda_tensor(backend, "value", YVEX_DTYPE_F32, value_input,
                              sizeof(value_input), &value, &err) &&
            quant_cuda_tensor(backend, "attention", YVEX_DTYPE_F32, NULL,
                              sizeof(attention_output), &attention, &err),
        "dense transformer GQA tensors allocate");
    rc = yvex_cuda_transformer_gqa(
        backend, query, key, value, attention, 2ull, 2ull, 1ull, 2ull, 1, &facts, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && facts.kernel_launches == 1ull &&
            facts.device_synchronizations == 1ull &&
            yvex_backend_tensor_read(backend, attention, attention_output,
                                     sizeof(attention_output), &err) == YVEX_OK,
        "causal grouped-query attention publishes one bounded output");
    for (index = 0ull; index < 2ull; ++index)
        YVEX_TEST_ASSERT(attention_output[index * 2ull] == 2.0f &&
                             attention_output[index * 2ull + 1ull] == 4.0f,
                         "first causal row sees only the first value");
    {
        float probability = expf(1.0f / sqrtf(2.0f));
        float expected0 = (probability * 2.0f + 6.0f) / (probability + 1.0f);
        float expected1 = (probability * 4.0f + 8.0f) / (probability + 1.0f);
        for (index = 2ull; index < 4ull; ++index)
            YVEX_TEST_ASSERT(fabsf(attention_output[index * 2ull] - expected0) < 1e-6f &&
                                 fabsf(attention_output[index * 2ull + 1ull] - expected1) < 1e-6f,
                             "second causal row uses stable grouped-query softmax");
        rc = yvex_cuda_transformer_gqa(
            backend, query, key, value, attention, 2ull, 2ull, 1ull, 2ull, 0,
            &facts, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && yvex_backend_tensor_read(
                                 backend, attention, attention_output,
                                 sizeof(attention_output), &err) == YVEX_OK,
            "full grouped-query attention publishes one bounded output");
        for (index = 0ull; index < 4ull; ++index)
            YVEX_TEST_ASSERT(fabsf(attention_output[index * 2ull] - expected0) < 1e-6f &&
                                 fabsf(attention_output[index * 2ull + 1ull] - expected1) < 1e-6f,
                             "non-causal rows attend the complete packed sequence");
    }
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "gate", YVEX_DTYPE_F32, gate_input,
                          sizeof(gate_input), &gate, &err) &&
            quant_cuda_tensor(backend, "up", YVEX_DTYPE_F32, up_input,
                              sizeof(up_input), &up, &err) &&
            quant_cuda_tensor(backend, "product", YVEX_DTYPE_F32, NULL,
                              sizeof(product_output), &product, &err) &&
            yvex_cuda_transformer_silu_product_bf16(
                backend, gate, up, product, 4ull, &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, product, product_output,
                                     sizeof(product_output), &err) == YVEX_OK,
        "dense transformer SiLU product executes");
    for (index = 0ull; index < 4ull; ++index) {
        float activated = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
            gate_input[index] / (1.0f + expf(-gate_input[index]))));
        float expected = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(activated * up_input[index]));
        YVEX_TEST_ASSERT(product_output[index] == expected,
                         "dense transformer SiLU product matches the BF16 scalar oracle");
    }
    YVEX_TEST_ASSERT(
        yvex_cuda_transformer_bf16_round(
            backend, product, 4ull, &facts, &err) == YVEX_OK &&
            facts.kernel_launches == 1ull && facts.d2h_bytes == sizeof(int) &&
            yvex_backend_tensor_read(backend, product, product_output,
                                     sizeof(product_output), &err) == YVEX_OK,
        "dense transformer BF16 activation round reports exact status transfer");
    for (index = 0ull; index < 4ull; ++index) {
        float activated = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
            gate_input[index] / (1.0f + expf(-gate_input[index]))));
        float expected = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(activated * up_input[index]));
        YVEX_TEST_ASSERT(product_output[index] == expected,
                         "dense transformer BF16 activation round matches the scalar codec");
    }
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &rotary, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &cosines, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &sines, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &query, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &key, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &value, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &attention, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &norm_input, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &norm_weight, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &norm_output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &gate, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &up, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &product, &err) == YVEX_OK,
        "dense transformer test tensors release cleanly");
    return 0;
}

static int quant_cuda_video_transformer(yvex_backend *backend)
{
    yvex_device_tensor *input = NULL, *first = NULL, *second = NULL, *third = NULL;
    yvex_device_tensor *cosines = NULL, *sines = NULL, *fused_device = NULL;
    yvex_device_tensor *swiglu_device = NULL, *residual_device = NULL, *output = NULL;
    yvex_device_tensor *scale = NULL, *weight = NULL, *bias = NULL;
    float interleaved[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float first_values[4], second_values[4], third_values[4];
    float cosine_values[2] = {0.75f, 0.75f}, sine_values[2] = {0.5f, 0.5f};
    float fused[4] = {-1.0f, 2.0f, 3.0f, -4.0f}, values[4];
    float residual[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float scales[2] = {0.5f, -2.0f};
    float norm_weight[2] = {1.5f, 0.5f}, norm_bias[2] = {-0.25f, 0.75f};
    yvex_backend_cuda_operation_facts facts;
    yvex_error err;
    unsigned long long index;

    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "video-interleaved", YVEX_DTYPE_F32,
                          interleaved, sizeof(interleaved), &input, &err) &&
            quant_cuda_tensor(backend, "video-first", YVEX_DTYPE_F32,
                              NULL, sizeof(first_values), &first, &err) &&
            quant_cuda_tensor(backend, "video-second", YVEX_DTYPE_F32,
                              NULL, sizeof(second_values), &second, &err) &&
            quant_cuda_tensor(backend, "video-third", YVEX_DTYPE_F32,
                              NULL, sizeof(third_values), &third, &err) &&
            yvex_cuda_transformer_split_interleaved_three(
                backend, input, first, second, third, 1ull, 2ull, 2ull,
                &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, first, first_values,
                                     sizeof(first_values), &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, second, second_values,
                                     sizeof(second_values), &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, third, third_values,
                                     sizeof(third_values), &err) == YVEX_OK,
        "video per-head Q/K/V split executes");
    YVEX_TEST_ASSERT(
        first_values[0] == 1.0f && first_values[1] == 2.0f &&
            first_values[2] == 7.0f && first_values[3] == 8.0f &&
            second_values[0] == 3.0f && second_values[2] == 9.0f &&
            third_values[0] == 5.0f && third_values[2] == 11.0f,
        "video split preserves interleaved head order");
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "video-cosines", YVEX_DTYPE_F32,
                          cosine_values, sizeof(cosine_values), &cosines, &err) &&
            quant_cuda_tensor(backend, "video-sines", YVEX_DTYPE_F32,
                              sine_values, sizeof(sine_values), &sines, &err) &&
            yvex_cuda_transformer_rotary_half_f32(
                backend, first, cosines, sines, 1ull, 2ull, 2ull, 2ull,
                &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, first, first_values,
                                     sizeof(first_values), &err) == YVEX_OK,
        "video F32 rotate-half executes");
    YVEX_TEST_ASSERT(first_values[0] == -0.25f && first_values[1] == 2.0f &&
                         first_values[2] == 1.25f && first_values[3] == 9.5f,
                     "video F32 rotate-half matches the scalar reference");
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "video-fused", YVEX_DTYPE_F32,
                          fused, sizeof(fused), &fused_device, &err) &&
            quant_cuda_tensor(backend, "video-swiglu", YVEX_DTYPE_F32,
                              NULL, 2ull * sizeof(float), &swiglu_device, &err) &&
            yvex_cuda_transformer_swiglu_split_f32(
                backend, fused_device, swiglu_device, 1ull, 2ull, 1, &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, swiglu_device, values,
                                     2ull * sizeof(float), &err) == YVEX_OK,
        "video gate-first F32 SwiGLU executes");
    for (index = 0ull; index < 2ull; ++index) {
        float expected = fused[index] / (1.0f + expf(-fused[index])) * fused[2ull + index];
        YVEX_TEST_ASSERT(fabsf(values[index] - expected) < 1e-6f,
                         "video gate-first SwiGLU matches the scalar reference");
    }
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "video-residual", YVEX_DTYPE_F32,
                          residual, sizeof(residual), &residual_device, &err) &&
            quant_cuda_tensor(backend, "video-scale", YVEX_DTYPE_F32,
                              scales, sizeof(scales), &scale, &err) &&
            quant_cuda_tensor(backend, "video-output", YVEX_DTYPE_F32,
                              NULL, sizeof(values), &output, &err) &&
            yvex_cuda_transformer_scaled_residual_f32(
                backend, residual_device, fused_device, scale, output,
                2ull, 2ull, &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, output, values, sizeof(values), &err) == YVEX_OK,
        "video scaled residual executes");
    for (index = 0ull; index < 4ull; ++index)
        YVEX_TEST_ASSERT(values[index] == residual[index] + fused[index] * scales[index % 2ull],
                         "video scaled residual matches the scalar reference");
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "video-norm-weight", YVEX_DTYPE_F32,
                          norm_weight, sizeof(norm_weight), &weight, &err) &&
            quant_cuda_tensor(backend, "video-norm-bias", YVEX_DTYPE_F32,
                              norm_bias, sizeof(norm_bias), &bias, &err) &&
            yvex_cuda_transformer_layer_norm_f32(
                backend, residual_device, weight, bias, output, 2ull, 2ull, 1e-5f,
                &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, output, values, sizeof(values), &err) == YVEX_OK,
        "video affine LayerNorm executes");
    for (index = 0ull; index < 4ull; ++index) {
        unsigned long long row = index / 2ull, lane = index % 2ull;
        float mean = (residual[row * 2ull] + residual[row * 2ull + 1ull]) * 0.5f;
        float delta = residual[index] - mean;
        float expected = delta / sqrtf(delta * delta + 1e-5f) *
                         norm_weight[lane] + norm_bias[lane];
        YVEX_TEST_ASSERT(fabsf(values[index] - expected) < 1e-6f,
                         "video affine LayerNorm matches the scalar reference");
    }
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &bias, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &weight, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &scale, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &residual_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &swiglu_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &fused_device, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &sines, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &cosines, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &third, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &second, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &first, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK,
        "video transformer primitive tensors release cleanly");
    return 0;
}

static float *quant_dense_weight(yvex_transformer_encoded_weight *weight,
                                 float **cursor, unsigned long long rows,
                                 unsigned long long width, float value)
{
    float *start = *cursor;
    unsigned long long index, elements = rows * width;
    for (index = 0ull; index < elements; ++index) start[index] = value;
    weight->encoded = (const unsigned char *)start;
    weight->encoded_bytes = elements * sizeof(float);
    weight->row_count = rows;
    weight->row_width = width;
    weight->row_bytes = width * sizeof(float);
    weight->qtype = YVEX_GGUF_QTYPE_F32;
    *cursor += elements;
    return start;
}

static int quant_dense_cancel(void *context)
{
    return context && *(const int *)context;
}

static int quant_cuda_dense_decoder(yvex_backend *backend)
{
    enum { ROWS = 2, WIDTH = 2, HEADS = 1, HEAD_DIM = 2, FFN = 2, OUTPUT = 1 };
    enum { WEIGHT_VALUES = 57 };
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *resident = NULL;
    yvex_transformer_encoded_weight weights[
        YVEX_TRANSFORMER_DENSE_DECODER_BLOCK_WEIGHT_COUNT] = {0};
    yvex_transformer_encoded_weight final_norm = {0}, final_bias = {0};
    yvex_transformer_encoded_weight output_weight = {0}, output_bias = {0};
    yvex_transformer_dense_decoder_request request = {0};
    yvex_transformer_dense_decoder_result result;
    unsigned char *mapped = NULL;
    float *cursor, *projection;
    float hidden[ROWS * WIDTH] = {1.0f, 3.0f, 4.0f, 0.0f};
    float cosines[ROWS * HEAD_DIM] = {1.0f, 1.0f, 1.0f, 1.0f};
    float sines[ROWS * HEAD_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
    float output[ROWS * OUTPUT] = {7.0f, 9.0f};
    float expected[ROWS * OUTPUT];
    unsigned long long row;
    int cancel = 0;
    yvex_error err;
    int rc;

    descriptor.name = "dense-decoder-weights";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = WEIGHT_VALUES * sizeof(float);
    YVEX_TEST_ASSERT(
        backend->vtable->resident_alloc(
            backend, &descriptor, &resident, &mapped, &err) == YVEX_OK,
        "dense decoder resident F32 weights allocate");
    cursor = (float *)mapped;
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_NORM1,
                       &cursor, 1ull, WIDTH, 1.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_QKV_WEIGHT,
                       &cursor, 3ull * WIDTH, WIDTH, 0.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_QKV_BIAS,
                       &cursor, 1ull, 3ull * WIDTH, 0.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_ATTENTION_WEIGHT,
                       &cursor, WIDTH, WIDTH, 0.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_ATTENTION_BIAS,
                       &cursor, 1ull, WIDTH, 0.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_SCALE1,
                       &cursor, 1ull, WIDTH, 1.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_NORM2,
                       &cursor, 1ull, WIDTH, 1.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_FF1_WEIGHT,
                       &cursor, 2ull * FFN, WIDTH, 0.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_FF1_BIAS,
                       &cursor, 1ull, 2ull * FFN, 0.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_FF2_WEIGHT,
                       &cursor, WIDTH, FFN, 0.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_FF2_BIAS,
                       &cursor, 1ull, WIDTH, 0.0f);
    quant_dense_weight(weights + YVEX_TRANSFORMER_DENSE_SCALE2,
                       &cursor, 1ull, WIDTH, 1.0f);
    quant_dense_weight(&final_norm, &cursor, 1ull, WIDTH, 1.0f);
    quant_dense_weight(&final_bias, &cursor, 1ull, WIDTH, 0.0f);
    projection = quant_dense_weight(&output_weight, &cursor, OUTPUT, WIDTH, 0.0f);
    projection[0] = 1.0f;
    projection[1] = -1.0f;
    quant_dense_weight(&output_bias, &cursor, 1ull, OUTPUT, 0.5f);
    YVEX_TEST_ASSERT((unsigned char *)cursor == mapped + descriptor.bytes,
                     "dense decoder fixture accounts every resident weight byte");
    YVEX_TEST_ASSERT(
        yvex_backend_resident_attach(
            backend, mapped, descriptor.bytes, resident, 29ull, &err) == YVEX_OK,
        "dense decoder weights attach as one immutable residency");
    request.block_weights = weights;
    request.final_norm_weight = &final_norm;
    request.final_norm_bias = &final_bias;
    request.output_weight = &output_weight;
    request.output_bias = &output_bias;
    request.hidden = hidden;
    request.cosines = cosines;
    request.sines = sines;
    request.rows = ROWS;
    request.output_rows = ROWS;
    request.width = WIDTH;
    request.heads = HEADS;
    request.head_dim = HEAD_DIM;
    request.rotary_dim = HEAD_DIM;
    request.ffn_width = FFN;
    request.block_count = 1ull;
    request.output_width = OUTPUT;
    request.output_capacity = ROWS * OUTPUT;
    request.epsilon = 1.0e-5f;
    request.output = output;
    request.cancel_requested = quant_dense_cancel;
    request.cancel_context = &cancel;
    rc = yvex_cuda_transformer_dense_decoder_execute(
        backend, &request, &result, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "dense decoder failed: %s (%s)\n",
                yvex_error_message(&err), yvex_error_where(&err));
    YVEX_TEST_ASSERT(rc == YVEX_OK && result.complete && result.rows == ROWS &&
                         result.output_rows == ROWS && result.block_count == 1ull &&
                         result.output_values == ROWS * OUTPUT &&
                         result.kernel_launches > 0ull && result.device_bytes > 0ull,
                     "one dense video decoder block executes transactionally");
    for (row = 0ull; row < ROWS; ++row) {
        float mean = (hidden[row * WIDTH] + hidden[row * WIDTH + 1ull]) * 0.5f;
        float first = hidden[row * WIDTH] - mean;
        float second = hidden[row * WIDTH + 1ull] - mean;
        float variance = (first * first + second * second) * 0.5f;
        expected[row] = (first - second) / sqrtf(variance + request.epsilon) + 0.5f;
        YVEX_TEST_ASSERT(fabsf(output[row] - expected[row]) < 1.0e-5f,
                         "dense decoder CUDA output matches independent F32 LayerNorm reference");
    }
    cancel = 1;
    output[0] = 7.0f;
    output[1] = 9.0f;
    rc = yvex_cuda_transformer_dense_decoder_execute(
        backend, &request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_CANCELLED && !result.complete &&
                         output[0] == 7.0f && output[1] == 9.0f,
                     "dense decoder cancellation preserves transactional output");
    cancel = 0;
    output_weight.row_width++;
    rc = yvex_cuda_transformer_dense_decoder_execute(
        backend, &request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !result.complete,
                     "dense decoder refuses mismatched physical weight geometry");
    output_weight.row_width--;
    YVEX_TEST_ASSERT(
        yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
        "dense decoder releases resident and activation ownership");
    return 0;
}

static int quant_cuda_omni_transformer(yvex_backend *backend)
{
    yvex_device_tensor *input = NULL, *output = NULL, *first = NULL, *second = NULL;
    yvex_device_tensor *third = NULL, *table = NULL, *update = NULL, *residual = NULL;
    float silu_input[4] = {-2.0f, -0.5f, 0.5f, 2.0f}, silu_output[4];
    float split_input[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float swiglu_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float split_first[4], split_second[4], split_third[4], swiglu_output[4];
    float modulation_input[4] = {1, 2, 3, 4}, modulation_output[4];
    float modulation_table[24] = {
        0.25f, -0.5f, 0.5f, 0.25f, 2, 3, 0, 0, 0, 0, 0, 0,
        -1, 1, -0.25f, 0.75f, -2, 4, 0, 0, 0, 0, 0, 0,
    };
    float update_values[4] = {2, 3, 4, 5}, residual_values[4] = {10, 20, 30, 40};
    float gated_output[4];
    unsigned int indices[2] = {1u, 0u};
    yvex_backend_cuda_operation_facts facts;
    yvex_error err;
    unsigned long long index;

    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "omni-silu-input", YVEX_DTYPE_F32, silu_input,
                          sizeof(silu_input), &input, &err) &&
            quant_cuda_tensor(backend, "omni-silu-output", YVEX_DTYPE_F32, NULL,
                              sizeof(silu_output), &output, &err) &&
            yvex_cuda_transformer_silu(
                backend, input, output, 4ull, 1, &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(
                backend, output, silu_output, sizeof(silu_output), &err) == YVEX_OK,
        "Omni timestep/AdaLN SiLU executes with explicit BF16 publication");
    for (index = 0ull; index < 4ull; ++index) {
        float expected = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
            silu_input[index] / (1.0f + expf(-silu_input[index]))));
        YVEX_TEST_ASSERT(silu_output[index] == expected,
                         "Omni SiLU matches the independent BF16 scalar policy");
    }
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
            quant_cuda_tensor(backend, "omni-split-input", YVEX_DTYPE_F32, split_input,
                              sizeof(split_input), &input, &err) &&
            quant_cuda_tensor(backend, "omni-split-first", YVEX_DTYPE_F32, NULL,
                              sizeof(split_first), &first, &err) &&
            quant_cuda_tensor(backend, "omni-split-second", YVEX_DTYPE_F32, NULL,
                              sizeof(split_second), &second, &err) &&
            quant_cuda_tensor(backend, "omni-split-third", YVEX_DTYPE_F32, NULL,
                              sizeof(split_third), &third, &err) &&
            yvex_cuda_transformer_split_three(
                backend, input, first, second, third, 2ull, 2ull, &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(
                backend, first, split_first, sizeof(split_first), &err) == YVEX_OK &&
            yvex_backend_tensor_read(
                backend, second, split_second, sizeof(split_second), &err) == YVEX_OK &&
            yvex_backend_tensor_read(
                backend, third, split_third, sizeof(split_third), &err) == YVEX_OK,
        "Omni fused QKV rows split without crossing row boundaries");
    YVEX_TEST_ASSERT(
        split_first[0] == 1 && split_first[1] == 2 && split_first[2] == 7 && split_first[3] == 8 &&
            split_second[0] == 3 && split_second[1] == 4 && split_second[2] == 9 && split_second[3] == 10 &&
            split_third[0] == 5 && split_third[1] == 6 && split_third[2] == 11 && split_third[3] == 12,
        "Omni fused QKV split preserves each terminal role");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
            quant_cuda_tensor(backend, "omni-swiglu-input", YVEX_DTYPE_F32,
                              swiglu_input, sizeof(swiglu_input), &input, &err) &&
        yvex_cuda_transformer_swiglu_split_bf16(
            backend, input, first, 2ull, 2ull, &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(
                backend, first, swiglu_output, sizeof(swiglu_output), &err) == YVEX_OK,
        "Omni combined FFN projection executes source-order SwiGLU");
    for (index = 0ull; index < 4ull; ++index) {
        unsigned long long row = index / 2ull, column = index % 2ull;
        float hidden = swiglu_input[row * 4ull + column];
        float gate = swiglu_input[row * 4ull + 2ull + column];
        float activated = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
            gate / (1.0f + expf(-gate))));
        float expected = yvex_quant_bf16_decode(yvex_quant_bf16_encode(hidden * activated));
        YVEX_TEST_ASSERT(swiglu_output[index] == expected,
                         "Omni source-order SwiGLU matches the BF16 scalar policy");
    }
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &third, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &second, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &first, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
            quant_cuda_tensor(backend, "omni-modulation-input", YVEX_DTYPE_F32,
                              modulation_input, sizeof(modulation_input), &input, &err) &&
            quant_cuda_tensor(backend, "omni-modulation-table", YVEX_DTYPE_F32,
                              modulation_table, sizeof(modulation_table), &table, &err) &&
            quant_cuda_tensor(backend, "omni-modulation-output", YVEX_DTYPE_F32, NULL,
                              sizeof(modulation_output), &output, &err) &&
            yvex_cuda_transformer_modulate_bf16(
                backend, input, table, indices, output, 2ull, 2ull, 2ull, 6ull,
                0u, 1u, &facts, &err) == YVEX_OK && facts.h2d_bytes == sizeof(indices) &&
            yvex_backend_tensor_read(backend, output, modulation_output,
                                     sizeof(modulation_output), &err) == YVEX_OK,
        "Omni row-indexed AdaLN modulation executes transactionally");
    for (index = 0ull; index < 4ull; ++index) {
        unsigned long long row = index / 2ull, column = index % 2ull;
        unsigned long long base = indices[row] * 12ull;
        float factor = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
            1.0f + modulation_table[base + 2ull + column]));
        float value = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
            modulation_input[index] * factor));
        float expected = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
            value + modulation_table[base + column]));
        YVEX_TEST_ASSERT(modulation_output[index] == expected,
                         "Omni AdaLN modulation matches the selected BF16 table row");
    }
    YVEX_TEST_ASSERT(
        quant_cuda_tensor(backend, "omni-update", YVEX_DTYPE_F32, update_values,
                          sizeof(update_values), &update, &err) &&
            quant_cuda_tensor(backend, "omni-residual", YVEX_DTYPE_F32, residual_values,
                              sizeof(residual_values), &residual, &err) &&
            yvex_cuda_transformer_gated_residual_bf16(
                backend, residual, table, indices, update, output, 2ull, 2ull, 2ull, 6ull,
                2u, &facts, &err) == YVEX_OK &&
            yvex_backend_tensor_read(
                backend, output, gated_output, sizeof(gated_output), &err) == YVEX_OK,
        "Omni row-indexed gated residual executes transactionally");
    for (index = 0ull; index < 4ull; ++index) {
        unsigned long long row = index / 2ull, column = index % 2ull;
        unsigned long long base = indices[row] * 12ull + 4ull + column;
        float gated = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
            modulation_table[base] * update_values[index]));
        float expected = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
            residual_values[index] + gated));
        YVEX_TEST_ASSERT(gated_output[index] == expected,
                         "Omni gated residual matches the selected BF16 table row");
    }
    YVEX_TEST_ASSERT(
        yvex_cuda_transformer_modulate_bf16(
            backend, input, table, (unsigned int[]){2u, 0u}, output,
            2ull, 2ull, 2ull, 6ull, 0u, 1u, &facts, &err) == YVEX_ERR_FORMAT,
        "Omni AdaLN refuses an out-of-range row before device mutation");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &residual, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &update, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &table, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK,
        "Omni transformer activation tensors release cleanly");
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
    unsigned int matvec_grid, matvec_block;
    int block_row;
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &options, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "CUDA qtype backend open failed: %s (%s)\n",
                yvex_error_message(&err), yvex_error_where(&err));
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "CUDA qtype parity backend opens");
    YVEX_TEST_ASSERT(
        yvex_cuda_qtype_matvec_geometry(
            24ull, 16384ull, 3ull, YVEX_GGUF_QTYPE_F32, 1,
            &matvec_grid, &matvec_block, &block_row) &&
            matvec_grid == 72u && matvec_block == 256u && block_row,
        "wide narrow F32 projection assigns one reduction block per row and input");
    YVEX_TEST_ASSERT(
        yvex_cuda_qtype_matvec_geometry(
            24ull, 16384ull, 3ull, YVEX_GGUF_QTYPE_F32, 0,
            &matvec_grid, &matvec_block, &block_row) &&
            matvec_grid == 12u && matvec_block == 192u && !block_row,
        "reference geometry retains canonical warp-owned rows");
    YVEX_TEST_ASSERT(
        yvex_cuda_qtype_matvec_geometry(
            4096ull, 8192ull, 1ull, YVEX_GGUF_QTYPE_Q8_0, 1,
            &matvec_grid, &matvec_block, &block_row) &&
            matvec_grid == 512u && matvec_block == 256u && !block_row,
        "encoded Q8 projection cannot enter the F32 reduction class");
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
    YVEX_TEST_ASSERT(quant_cuda_q8_matvec(backend, YVEX_GGUF_QTYPE_MXFP4) == 0,
                     "MXFP4 production Q8 activation matvec");
    for (index = 4u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        YVEX_TEST_ASSERT(quant_cuda_q8_grouped_matvec(
                             backend, cases[index].qtype, 1536u) == 0,
                         "six-block production Q8 activation matvec");
        YVEX_TEST_ASSERT(quant_cuda_q8_grouped_matvec(
                             backend, cases[index].qtype, 768u) == 0,
                         "three-block production Q8 activation matvec");
    }
    YVEX_TEST_ASSERT(quant_cuda_bf16_gemm(backend) == 0,
                     "BF16 production row batch GEMM");
    YVEX_TEST_ASSERT(quant_cuda_f32_gemm(backend) == 0,
                     "F32 production row batch GEMM");
    YVEX_TEST_ASSERT(quant_cuda_encoded_gather(backend) == 0,
                     "resident qtype row gather");
    YVEX_TEST_ASSERT(quant_cuda_transformer_facts(backend) == 0,
                     "transformer envelope physical facts");
    YVEX_TEST_ASSERT(quant_cuda_dense_transformer(backend) == 0,
                     "dense transformer activation primitives");
    YVEX_TEST_ASSERT(quant_cuda_video_transformer(backend) == 0,
                     "video transformer activation primitives");
    YVEX_TEST_ASSERT(quant_cuda_dense_decoder(backend) == 0,
                     "resident dense video decoder execution");
    YVEX_TEST_ASSERT(quant_cuda_omni_transformer(backend) == 0,
                     "Omni transformer activation primitives");
    yvex_backend_close(backend);
    return quant_cuda_refusals(&options);
}
