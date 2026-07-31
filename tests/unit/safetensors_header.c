
#include "tests/test.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <yvex/source.h>
#include <yvex/internal/core.h>

static int make_dir(const char *path)
{
    return mkdir(path, 0777) == 0 || errno == EEXIST;
}

static int write_safetensors(const char *path, const char *json, unsigned long long payload_len)
{
    FILE *fp = fopen(path, "wb");
    unsigned long long len = (unsigned long long)strlen(json);
    unsigned char b[8];
    unsigned long long i;

    if (!fp) return 0;
    for (i = 0; i < 8; ++i) {
        b[i] = (unsigned char)((len >> (8u * i)) & 0xffu);
    }
    if (fwrite(b, 1, 8, fp) != 8) return 0;
    if (fwrite(json, 1, (size_t)len, fp) != (size_t)len) return 0;
    for (i = 0; i < payload_len; ++i) {
        fputc((int)(i & 0xffu), fp);
    }
    return fclose(fp) == 0;
}

static int write_short(const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fputs("bad", fp);
    return fclose(fp) == 0;
}

int yvex_test_safetensors_header(void)
{
    const char *root = "build/tests/safetensors-header";
    yvex_native_weight_options options;
    yvex_native_weight_table *table = NULL;
    yvex_native_weight_summary summary;
    const yvex_native_weight_info *row;
    yvex_core_execution_observation observed_before, observed_after, observed_delta;
    yvex_error err;
    int rc;

    system("rm -rf build/tests/safetensors-header");
    YVEX_TEST_ASSERT(make_dir("build"), "make build");
    YVEX_TEST_ASSERT(make_dir("build/tests"), "make build/tests");
    YVEX_TEST_ASSERT(make_dir(root), "make fixture root");
    YVEX_TEST_ASSERT(write_safetensors("build/tests/safetensors-header/model-00001.safetensors",
        "{\"__metadata__\":{\"format\":\"pt\"},\"layer.weight\":{\"dtype\":\"F16\",\"shape\":[2,3],\"data_offsets\":[0,12]},\"layer.bias\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[12,24]}}",
        24), "write valid safetensors");

    memset(&options, 0, sizeof(options));
    options.source_dir = root;
    options.recursive = 1;
    yvex_error_clear(&err);
    yvex_core_execution_observation_snapshot(&observed_before);
    rc = yvex_native_weight_table_open(&table, &options, &err);
    yvex_core_execution_observation_snapshot(&observed_after);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "valid safetensors header parses");
    YVEX_TEST_ASSERT(yvex_core_execution_observation_delta(
                         &observed_before, &observed_after, &observed_delta) &&
                         observed_delta.source_headers_read == 1ull,
                     "successful source owner advances the observed header counter");
    YVEX_TEST_ASSERT(yvex_native_weight_table_summary(table, &summary, &err) == YVEX_OK, "summary works");
    YVEX_TEST_ASSERT(summary.shard_count == 1, "shard count matches");
    YVEX_TEST_ASSERT(summary.tensor_count == 2, "tensor count matches");
    YVEX_TEST_ASSERT(summary.total_tensor_bytes == 24, "bytes computed");
    row = yvex_native_weight_table_find(table, "layer.weight");
    YVEX_TEST_ASSERT(row != NULL, "find tensor");
    YVEX_TEST_ASSERT(row->dtype == YVEX_NATIVE_DTYPE_F16, "dtype parses");
    YVEX_TEST_ASSERT(row->rank == 2, "shape rank parses");
    YVEX_TEST_ASSERT(row->dims[0] == 2 && row->dims[1] == 3, "shape dims parse");
    YVEX_TEST_ASSERT(row->data_start == 0 && row->data_end == 12 && row->data_bytes == 12, "data offsets parse");
    yvex_native_weight_table_close(table);

    system("rm -rf build/tests/safetensors-header");
    YVEX_TEST_ASSERT(make_dir(root), "make bad short root");
    YVEX_TEST_ASSERT(write_short("build/tests/safetensors-header/model-00001.safetensors"), "write short file");
    table = NULL;
    rc = yvex_native_weight_table_open(&table, &options, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "bad short file rejected");

    system("rm -rf build/tests/safetensors-header");
    YVEX_TEST_ASSERT(make_dir(root), "make bad json root");
    YVEX_TEST_ASSERT(write_safetensors("build/tests/safetensors-header/model-00001.safetensors",
        "{\"x\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,4]}", 4), "write bad json");
    rc = yvex_native_weight_table_open(&table, &options, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "bad JSON rejected");

    system("rm -rf build/tests/safetensors-header");
    YVEX_TEST_ASSERT(make_dir(root), "make bad offsets root");
    YVEX_TEST_ASSERT(write_safetensors("build/tests/safetensors-header/model-00001.safetensors",
        "{\"x\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[4,2]}}", 4), "write bad offsets");
    rc = yvex_native_weight_table_open(&table, &options, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "bad offsets rejected");

    system("rm -rf build/tests/safetensors-header");
    YVEX_TEST_ASSERT(make_dir(root), "make unknown dtype root");
    YVEX_TEST_ASSERT(write_safetensors("build/tests/safetensors-header/model-00001.safetensors",
        "{\"x\":{\"dtype\":\"WEIRD\",\"shape\":[2],\"data_offsets\":[0,2]}}", 2), "write unknown dtype");
    rc = yvex_native_weight_table_open(&table, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "unknown dtype accepted as other");
    YVEX_TEST_ASSERT(yvex_native_weight_table_summary(table, &summary, &err) == YVEX_OK, "unknown summary works");
    YVEX_TEST_ASSERT(summary.unknown_dtype_count == 1, "unknown dtype counted");
    yvex_native_weight_table_close(table);

    return 0;
}
