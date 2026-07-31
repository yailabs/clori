
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int open_fixture(yvex_artifact **artifact,
                        yvex_gguf **gguf,
                        yvex_tensor_table **tensors)
{
    yvex_artifact_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.readonly = 1;
    options.map = 1;

    rc = yvex_artifact_open(artifact, &options, &err);
    if (rc == YVEX_OK) {
        rc = yvex_gguf_open(gguf, *artifact, &err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_tensor_table_from_gguf(tensors, *gguf, &err);
    }
    if (rc != YVEX_OK) {
        fprintf(stderr, "fixture open failed: %s: %s\n",
                yvex_error_where(&err), yvex_error_message(&err));
        yvex_tensor_table_close(*tensors);
        yvex_gguf_close(*gguf);
        yvex_artifact_close(*artifact);
        return 1;
    }
    return 0;
}

int yvex_cuda_test_materialize_cuda(void)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_backend *backend = NULL;
    yvex_weight_table *weights = NULL;
    yvex_backend_options backend_options;
    yvex_materialize_options materialize_options;
    const yvex_tensor_info *tensor;
    const yvex_materialized_weight *weight;
    unsigned char *readback;
    yvex_error err;
    int rc;

    memset(&backend_options, 0, sizeof(backend_options));
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &backend_options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        return 0;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open cuda backend");
    YVEX_TEST_ASSERT(open_fixture(&artifact, &gguf, &tensors) == 0, "open fixture");

    memset(&materialize_options, 0, sizeof(materialize_options));
    materialize_options.backend_name = "cuda";
    rc = yvex_weight_table_materialize(&weights,
                                       artifact,
                                       gguf,
                                       tensors,
                                       backend,
                                       &materialize_options,
                                       &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "materialize cuda");
    weight = yvex_weight_table_find(weights, "token_embd.weight");
    tensor = yvex_tensor_table_at(tensors, 0);
    YVEX_TEST_ASSERT(weight != NULL && tensor != NULL, "weight and source tensor");
    YVEX_TEST_ASSERT(yvex_weight_residency_of(weight) == YVEX_WEIGHT_RESIDENCY_CUDA_BACKEND,
                     "cuda residency");

    readback = (unsigned char *)malloc((size_t)yvex_weight_bytes(weight));
    YVEX_TEST_ASSERT(readback != NULL, "readback alloc");
    rc = yvex_backend_tensor_read(backend,
                                  yvex_weight_device_tensor(weight),
                                  readback,
                                  yvex_weight_bytes(weight),
                                  &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda materialized tensor");
    YVEX_TEST_ASSERT(memcmp(readback,
                            yvex_artifact_data(artifact) + tensor->absolute_offset,
                            (size_t)tensor->storage_bytes) == 0,
                     "cuda bytes match artifact tensor bytes");
    free(readback);

    yvex_weight_table_close(weights);
    yvex_backend_close(backend);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}
