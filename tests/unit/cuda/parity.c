/*
 * Compares the CUDA backend CUDA F32 embedding op against the backend layer CPU reference op.
 * Returns 77 when CUDA is unavailable.
 */
#include <stdio.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static void make_desc(yvex_backend_tensor_desc *desc,
                      const char *name,
                      unsigned long long d0,
                      unsigned long long d1,
                      unsigned long long bytes)
{
    memset(desc, 0, sizeof(*desc));
    desc->name = name;
    desc->dtype = YVEX_DTYPE_F32;
    desc->rank = 2;
    desc->dims[0] = d0;
    desc->dims[1] = d1;
    desc->bytes = bytes;
}

static int open_cuda_or_skip(yvex_backend **out)
{
    yvex_backend_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(out, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        fprintf(stderr, "SKIP: CUDA unavailable: %s\n", yvex_error_message(&err));
        return 77;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open cuda");
    return 0;
}

int yvex_cuda_test_parity(void)
{
    yvex_backend *cpu = NULL;
    yvex_backend *cuda = NULL;
    yvex_device_tensor *cpu_embedding = NULL;
    yvex_device_tensor *cpu_out = NULL;
    yvex_device_tensor *cuda_embedding = NULL;
    yvex_device_tensor *cuda_out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_error err;
    float embedding_data[12] = {
        0.0f, 1.0f, 2.0f, 3.0f,
        10.0f, 11.0f, 12.0f, 13.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
    };
    float cpu_data[8];
    float cuda_data[8];
    unsigned int token_ids[2] = {0, 2};
    double max_abs_diff = 0.0;
    int rc;
    unsigned long i;

    rc = open_cuda_or_skip(&cuda);
    if (rc == 77) return 77;
    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&cpu, &err) == YVEX_OK, "open cpu");

    make_desc(&desc, "embedding", 4, 3, sizeof(embedding_data));
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(cpu, &desc, &cpu_embedding, &err) == YVEX_OK,
                     "alloc cpu embedding");
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(cuda, &desc, &cuda_embedding, &err) == YVEX_OK,
                     "alloc cuda embedding");
    YVEX_TEST_ASSERT(yvex_backend_tensor_write(cpu, cpu_embedding, embedding_data,
                                               sizeof(embedding_data), &err) == YVEX_OK,
                     "write cpu embedding");
    YVEX_TEST_ASSERT(yvex_backend_tensor_write(cuda, cuda_embedding, embedding_data,
                                               sizeof(embedding_data), &err) == YVEX_OK,
                     "write cuda embedding");

    make_desc(&desc, "out", 2, 4, sizeof(cpu_data));
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(cpu, &desc, &cpu_out, &err) == YVEX_OK,
                     "alloc cpu out");
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(cuda, &desc, &cuda_out, &err) == YVEX_OK,
                     "alloc cuda out");
    YVEX_TEST_ASSERT(yvex_backend_op_embed(cpu, cpu_embedding, token_ids, 2, cpu_out, &err) == YVEX_OK,
                     "cpu embed");
    YVEX_TEST_ASSERT(yvex_backend_op_embed(cuda, cuda_embedding, token_ids, 2, cuda_out, &err) == YVEX_OK,
                     "cuda embed");
    YVEX_TEST_ASSERT(yvex_backend_tensor_read(cpu, cpu_out, cpu_data, sizeof(cpu_data), &err) == YVEX_OK,
                     "read cpu out");
    YVEX_TEST_ASSERT(yvex_backend_tensor_read(cuda, cuda_out, cuda_data, sizeof(cuda_data), &err) == YVEX_OK,
                     "read cuda out");

    for (i = 0; i < 8; ++i) {
        double diff = (double)cpu_data[i] - (double)cuda_data[i];
        if (diff < 0.0) {
            diff = -diff;
        }
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }
    YVEX_TEST_ASSERT(max_abs_diff == 0.0, "cpu/cuda embed parity exact");
    printf("cpu_cuda_embed_parity: pass\n");
    printf("max_abs_diff: %.0f\n", max_abs_diff);
    printf("status: cuda-parity\n");

    yvex_backend_tensor_free(cuda, cuda_out);
    yvex_backend_tensor_free(cuda, cuda_embedding);
    yvex_backend_tensor_free(cpu, cpu_out);
    yvex_backend_tensor_free(cpu, cpu_embedding);
    yvex_backend_close(cuda);
    yvex_backend_close(cpu);
    return 0;
}
