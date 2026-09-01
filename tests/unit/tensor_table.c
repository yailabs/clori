/*
 * Exercises model layer builds an owned YVEX tensor table from the GGUF parser GGUF tensor
 * directory without claiming model execution.
 */
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int open_valid_gguf(yvex_artifact **artifact, yvex_gguf **gguf)
{
    yvex_artifact_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.path = "tests/fixtures/gguf/valid-metadata-tensors.gguf";
    options.readonly = 1;

    rc = yvex_artifact_open(artifact, &options, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "artifact open failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        return 1;
    }

    rc = yvex_gguf_open(gguf, *artifact, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "gguf open failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        yvex_artifact_close(*artifact);
        *artifact = NULL;
        return 1;
    }
    return 0;
}

static int test_table_from_fixture(void)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *table = NULL;
    const yvex_tensor_info *tensor;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_valid_gguf(&artifact, &gguf) == 0, "open valid gguf");
    rc = yvex_tensor_table_from_gguf(&table, gguf, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "tensor table builds");

    YVEX_TEST_ASSERT(yvex_tensor_table_count(table) == 1, "tensor table count");
    tensor = yvex_tensor_table_at(table, 0);
    YVEX_TEST_ASSERT(tensor != NULL, "tensor 0 exists");
    YVEX_TEST_ASSERT_STREQ(tensor->name, "token_embd.weight", "tensor name");
    YVEX_TEST_ASSERT(tensor->rank == 2, "tensor rank");
    YVEX_TEST_ASSERT(tensor->dims[0] == 4, "dim 0");
    YVEX_TEST_ASSERT(tensor->dims[1] == 8, "dim 1");
    YVEX_TEST_ASSERT(tensor->element_count == 32, "element count");
    YVEX_TEST_ASSERT(tensor->dtype == YVEX_DTYPE_F32, "dtype mapped");
    YVEX_TEST_ASSERT(tensor->ggml_type == 0, "raw type preserved");
    YVEX_TEST_ASSERT(tensor->storage_bytes == 128, "storage bytes");
    YVEX_TEST_ASSERT(tensor->role == YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, "role classified");
    YVEX_TEST_ASSERT(tensor->relative_offset == 0, "relative offset");
    YVEX_TEST_ASSERT(tensor->absolute_offset == yvex_gguf_tensor_data_offset(gguf), "absolute offset preserved");

    tensor = yvex_tensor_table_find(table, "token_embd.weight");
    YVEX_TEST_ASSERT(tensor != NULL, "find tensor by name");
    YVEX_TEST_ASSERT(yvex_tensor_table_find(table, "missing.weight") == NULL, "missing tensor null");
    YVEX_TEST_ASSERT(yvex_tensor_table_at(table, 99) == NULL, "out of range null");

    yvex_tensor_table_close(table);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

static int test_role_patterns(void)
{
    unsigned long long dims[2] = {4, 8};

    YVEX_TEST_ASSERT_STREQ(yvex_tensor_role_name(YVEX_TENSOR_ROLE_TOKEN_EMBEDDING),
                           "token_embedding", "role name");
    YVEX_TEST_ASSERT_STREQ(
        yvex_tensor_role_name(YVEX_TENSOR_ROLE_SEQUENCE_MIXER_QKV_PROJECTION),
        "sequence_mixer_qkv_projection", "sequence mixer role name");
    YVEX_TEST_ASSERT(yvex_tensor_role_classify("llama", "model.layers.0.self_attn.q_proj.weight",
                                               2, dims, YVEX_DTYPE_F32) == YVEX_TENSOR_ROLE_ATTENTION_Q,
                     "q projection role");
    YVEX_TEST_ASSERT(yvex_tensor_role_classify("llama", "model.layers.0.mlp.down_proj.weight",
                                               2, dims, YVEX_DTYPE_F32) == YVEX_TENSOR_ROLE_FFN_DOWN,
                     "ffn down role");
    YVEX_TEST_ASSERT(yvex_tensor_role_classify("llama", "unclassified.weight",
                                               2, dims, YVEX_DTYPE_F32) == YVEX_TENSOR_ROLE_UNKNOWN,
                     "unknown role");
    return 0;
}

int yvex_test_tensor_table(void)
{
    if (test_table_from_fixture() != 0) {
        return 1;
    }
    if (test_role_patterns() != 0) {
        return 1;
    }
    return 0;
}
