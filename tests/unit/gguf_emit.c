
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <yvex/api.h>

#include "tests/test.h"

static int open_emitted(const char *path,
                        yvex_artifact **artifact,
                        yvex_gguf **gguf,
                        yvex_tensor_table **tensors)
{
    yvex_artifact_options artifact_options;
    yvex_error err;
    int rc;

    memset(&artifact_options, 0, sizeof(artifact_options));
    yvex_error_clear(&err);
    artifact_options.path = path;
    artifact_options.readonly = 1;
    artifact_options.map = 1;

    rc = yvex_artifact_open(artifact, &artifact_options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(gguf, *artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(tensors, *gguf, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "open emitted failed: %s: %s\n",
                yvex_error_where(&err), yvex_error_message(&err));
        yvex_tensor_table_close(*tensors);
        yvex_gguf_close(*gguf);
        yvex_artifact_close(*artifact);
        return 1;
    }
    return 0;
}

static int test_controlled_emit_roundtrip(void)
{
    const char *path = "build/tests/gguf-emit/yvex-owned-core.gguf";
    yvex_gguf_emit_options options;
    yvex_gguf_emit_summary summary;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_backend *backend = NULL;
    yvex_weight_table *weights = NULL;
    yvex_materialize_options materialize_options;
    yvex_materialize_summary materialize_summary;
    const yvex_gguf_header *header;
    const yvex_tensor_info *tensor;
    yvex_error err;
    int rc;

    (void)mkdir("build/tests/gguf-emit", 0777);
    remove(path);
    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.out_path = path;
    options.model_name = "yvex-owned-gguf-test";
    options.architecture = "llama";
    options.overwrite = 1;
    options.transpose_2d = 1;

    rc = yvex_gguf_emit_controlled(&options, &summary, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "controlled emit failed: %s: %s\n",
                yvex_error_where(&err), yvex_error_message(&err));
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "controlled emit succeeds");
    YVEX_TEST_ASSERT(summary.status == YVEX_GGUF_EMIT_STATUS_WRITTEN, "summary status");
    YVEX_TEST_ASSERT(summary.metadata_count == 12, "metadata count summary");
    YVEX_TEST_ASSERT(summary.tensor_count == 1, "tensor count summary");
    YVEX_TEST_ASSERT(summary.tensor_payload_bytes == 128, "payload bytes summary");
    YVEX_TEST_ASSERT(summary.alignment == 32, "alignment summary");
    YVEX_TEST_ASSERT(summary.bytes_written > 128, "bytes written summary");
    YVEX_TEST_ASSERT(summary.roundtrip_validated == 1, "roundtrip validated");
    YVEX_TEST_ASSERT_STREQ(yvex_gguf_emit_status_name(summary.status),
                           "gguf-written",
                           "status name");

    YVEX_TEST_ASSERT(open_emitted(path, &artifact, &gguf, &tensors) == 0, "open emitted");
    header = yvex_gguf_header_view(gguf);
    YVEX_TEST_ASSERT(header != NULL, "header");
    YVEX_TEST_ASSERT(header->version == 3, "version");
    YVEX_TEST_ASSERT(header->metadata_count == 12, "metadata count");
    YVEX_TEST_ASSERT(header->tensor_count == 1, "tensor count");

    tensor = yvex_tensor_table_find(tensors, "token_embd.weight");
    YVEX_TEST_ASSERT(tensor != NULL, "tensor exists");
    YVEX_TEST_ASSERT(tensor->rank == 2, "tensor rank");
    YVEX_TEST_ASSERT(tensor->dims[0] == 4 && tensor->dims[1] == 8, "tensor dims");
    YVEX_TEST_ASSERT(tensor->dtype == YVEX_DTYPE_F32, "tensor dtype");
    YVEX_TEST_ASSERT(tensor->storage_bytes == 128, "tensor payload bytes");
    YVEX_TEST_ASSERT(tensor->role == YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, "tensor role");

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu");
    memset(&materialize_options, 0, sizeof(materialize_options));
    materialize_options.backend_name = "cpu";
    rc = yvex_weight_table_materialize(&weights,
                                       artifact,
                                       gguf,
                                       tensors,
                                       backend,
                                       &materialize_options,
                                       &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "materialize emitted");
    rc = yvex_weight_table_get_summary(weights, &materialize_summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "materialize summary");
    YVEX_TEST_ASSERT(materialize_summary.status == YVEX_WEIGHT_STATUS_MATERIALIZED,
                     "materialized status");
    YVEX_TEST_ASSERT(materialize_summary.tensors_materialized == 1,
                     "materialized one tensor");
    YVEX_TEST_ASSERT(materialize_summary.bytes_materialized == 128,
                     "materialized bytes");
    YVEX_TEST_ASSERT(materialize_summary.execution_ready == 0,
                     "execution remains false");

    yvex_weight_table_close(weights);
    yvex_backend_close(backend);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

static int test_overwrite_refused(void)
{
    const char *path = "build/tests/gguf-emit/yvex-owned-core.gguf";
    yvex_gguf_emit_options options;
    yvex_gguf_emit_summary summary;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.out_path = path;
    rc = yvex_gguf_emit_controlled(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "overwrite refused");
    return 0;
}

static int test_invalid_out_path_fails(void)
{
    yvex_gguf_emit_options options;
    yvex_gguf_emit_summary summary;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.out_path = "build/tests/gguf-emit/missing-dir/out.gguf";
    options.overwrite = 1;
    rc = yvex_gguf_emit_controlled(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "invalid output path fails");
    YVEX_TEST_ASSERT(summary.status == YVEX_GGUF_EMIT_STATUS_FAILED,
                     "failed summary status");
    return 0;
}

int yvex_test_gguf_emit(void)
{
    if (test_controlled_emit_roundtrip() != 0) return 1;
    if (test_overwrite_refused() != 0) return 1;
    if (test_invalid_out_path_fails() != 0) return 1;
    return 0;
}
