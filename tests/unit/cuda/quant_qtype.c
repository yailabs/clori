/*
 * quant_qtype.c - CUDA encoded-qtype arithmetic parity and failure tests.
 *
 * Owner: tests/unit/cuda.
 * Owns: bounded CPU/CUDA row-dot parity, admission refusals, repeatability,
 *   copy/launch/cleanup fault evidence, and backend cleanup assertions.
 * Does not own: model execution, artifact emission, or capability policy.
 * Invariants: all encoded inputs come from canonical TRACK.QUANT codecs.
 * Boundary: qtype primitive parity is not transformer or generation support.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "src/backend/cuda/private.h"
#include <yvex/internal/quant_numeric.h>

#include "tests/test.h"

/* Encodes a complete row one canonical block at a time into owned storage. */
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

/* Proves one qtype direct kernel against the canonical bounded CPU primitive. */
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
    yvex_backend_close(backend);
    return quant_cuda_refusals(&options);
}
