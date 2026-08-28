
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <yvex/api.h>

#include "tests/test.h"

static int mkdir_ok(const char *p) { return mkdir(p, 0777) == 0 || errno == EEXIST; }

static int write_st(const char *path)
{
    const char *json = "{\"model.embed_tokens.weight\":{\"dtype\":\"F16\",\"shape\":[8,4],\"data_offsets\":[0,64]}}";
    unsigned long long len = (unsigned long long)strlen(json);
    unsigned char b[8];
    FILE *fp = fopen(path, "wb");
    unsigned long long i;
    if (!fp) return 0;
    for (i = 0; i < 8; ++i) b[i] = (unsigned char)((len >> (i * 8u)) & 0xffu);
    fwrite(b, 1, 8, fp);
    fwrite(json, 1, (size_t)len, fp);
    for (i = 0; i < 64; ++i) fputc((int)(i & 0xffu), fp);
    return fclose(fp) == 0;
}

int yvex_test_conversion_payload(void)
{
    yvex_conversion_options options;
    yvex_conversion_summary summary;
    yvex_artifact_options ao;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    const yvex_tensor_info *tensor;
    const yvex_gguf_value *metadata;
    const char *architecture;
    unsigned long long architecture_len;
    unsigned long long semantic_context;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/conversion-payload") == 0,
                     "clear conversion payload fixture");
    YVEX_TEST_ASSERT(mkdir_ok("build") && mkdir_ok("build/tests") && mkdir_ok("build/tests/conversion-payload"), "mkdir");
    YVEX_TEST_ASSERT(write_st("build/tests/conversion-payload/model.safetensors"), "write st");

    memset(&options, 0, sizeof(options));
    options.architecture = "qwen3";
    options.native_source_dir = "build/tests/conversion-payload";
    options.tensor_name = "model.embed_tokens.weight";
    options.target_qtype = "F32";
    options.out_path = "build/tests/conversion-payload/qwen3-8b-selected-embed-F32-noimatrix-yvex-v1.gguf";
    options.overwrite = 1;
    yvex_error_clear(&err);
    rc = yvex_conversion_emit_gguf(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "emit succeeds");
    YVEX_TEST_ASSERT(summary.bytes_read == 64, "bytes read");
    YVEX_TEST_ASSERT(summary.bytes_written > 128, "bytes written");
    YVEX_TEST_ASSERT(summary.roundtrip_validated, "roundtrip");

    memset(&ao, 0, sizeof(ao));
    ao.path = options.out_path;
    ao.readonly = 1;
    rc = yvex_artifact_open(&artifact, &ao, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "parse emitted");
    tensor = yvex_tensor_table_find(tensors, "token_embd.weight");
    YVEX_TEST_ASSERT(tensor && tensor->dims[0] == 4 && tensor->dims[1] == 8 && tensor->dtype == YVEX_DTYPE_F32, "tensor shape dtype");
    metadata = yvex_gguf_metadata_find(gguf, "general.architecture");
    YVEX_TEST_ASSERT(metadata && yvex_gguf_value_as_string(metadata, &architecture,
                                                           &architecture_len) == YVEX_OK,
                     "qwen architecture metadata");
    YVEX_TEST_ASSERT(architecture_len == 4 && strncmp(architecture, "qwen", 4) == 0,
                     "qwen canonical architecture");
    metadata = yvex_gguf_metadata_find(gguf, "qwen.context_length");
    YVEX_TEST_ASSERT(metadata &&
                         yvex_gguf_value_as_u64(metadata, &semantic_context) == YVEX_OK &&
                         semantic_context == 32768u,
                     "qwen semantic context metadata");
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    tensors = NULL;
    gguf = NULL;
    artifact = NULL;

    options.architecture = "deepseek4";
    options.out_path =
        "build/tests/conversion-payload/deepseek-selected-embed-F32-noimatrix-yvex-v1.gguf";
    rc = yvex_conversion_emit_gguf(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "DeepSeek emit succeeds");
    ao.path = options.out_path;
    rc = yvex_artifact_open(&artifact, &ao, &err);
    if (rc == YVEX_OK)
        rc = yvex_gguf_open(&gguf, artifact, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "parse DeepSeek emission");
    metadata = yvex_gguf_metadata_find(gguf, "general.architecture");
    YVEX_TEST_ASSERT(metadata && yvex_gguf_value_as_string(metadata, &architecture,
                                                           &architecture_len) == YVEX_OK,
                     "DeepSeek architecture metadata");
    YVEX_TEST_ASSERT(architecture_len == 8 && strncmp(architecture, "deepseek", 8) == 0,
                     "DeepSeek canonical architecture");
    metadata = yvex_gguf_metadata_find(gguf, "deepseek.context_length");
    YVEX_TEST_ASSERT(metadata &&
                         yvex_gguf_value_as_u64(metadata, &semantic_context) == YVEX_OK &&
                         semantic_context == 1048576u,
                     "DeepSeek semantic maximum is not the selected runtime context");
    YVEX_TEST_ASSERT(yvex_gguf_metadata_find(gguf, "qwen.context_length") == NULL,
                     "DeepSeek proof does not use Qwen metadata");
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);

    options.overwrite = 0;
    rc = yvex_conversion_emit_gguf(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "overwrite false rejects");
    return 0;
}
