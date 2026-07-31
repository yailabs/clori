
#include "tests/test.h"
#include <yvex/internal/source.h>
#include <yvex/internal/source_payload.h>

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <yvex/source.h>

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
    for (i = 0; i < 8; ++i) b[i] = (unsigned char)((len >> (8u * i)) & 0xffu);
    if (fwrite(b, 1, 8, fp) != 8) return 0;
    if (fwrite(json, 1, (size_t)len, fp) != (size_t)len) return 0;
    for (i = 0; i < payload_len; ++i) fputc('x', fp);
    return fclose(fp) == 0;
}

int yvex_test_native_weights(void)
{
    const char *root = "build/tests/native-weights-fixture";
    yvex_native_weight_options options;
    yvex_native_weight_table *table = NULL;
    yvex_native_weight_summary summary;
    const yvex_native_weight_info *row;
    yvex_error err;
    int rc;

    system("rm -rf build/tests/native-weights-fixture");
    YVEX_TEST_ASSERT(make_dir("build"), "make build");
    YVEX_TEST_ASSERT(make_dir("build/tests"), "make build/tests");
    YVEX_TEST_ASSERT(make_dir(root), "make native fixture root");
    YVEX_TEST_ASSERT(write_safetensors("build/tests/native-weights-fixture/model-00001-of-00002.safetensors",
        "{\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[4,4],\"data_offsets\":[0,32]}}", 32), "write shard 1");
    YVEX_TEST_ASSERT(write_safetensors("build/tests/native-weights-fixture/model-00002-of-00002.safetensors",
        "{\"scale.weight\":{\"dtype\":\"F8_E8M0\",\"shape\":[1],\"data_offsets\":[16,17]},\"b.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[8,2],\"data_offsets\":[0,16]}}", 17), "write shard 2");

    memset(&options, 0, sizeof(options));
    options.source_dir = root;
    options.recursive = 1;
    yvex_error_clear(&err);
    rc = yvex_native_weight_table_open(&table, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open native weight table");
    YVEX_TEST_ASSERT(yvex_native_weight_table_summary(table, &summary, &err) == YVEX_OK, "summary");
    YVEX_TEST_ASSERT(summary.shard_count == 2, "summary shard_count=2");
    YVEX_TEST_ASSERT(summary.tensor_count == 3, "summary tensor_count expected");
    YVEX_TEST_ASSERT(summary.total_tensor_bytes == 49, "summary bytes");
    YVEX_TEST_ASSERT_STREQ(yvex_native_weight_table_at(table, 0)->name,
                           "a.weight", "inventory order is deterministic");
    row = yvex_native_weight_table_find(table, "b.weight");
    YVEX_TEST_ASSERT(row != NULL, "find tensor by name");
    YVEX_TEST_ASSERT_STREQ(row->shard_path, "model-00002-of-00002.safetensors", "shard path copied");
    YVEX_TEST_ASSERT(row->dtype == YVEX_NATIVE_DTYPE_F8_E4M3, "F8 dtype parses");
    row = yvex_native_weight_table_find(table, "scale.weight");
    YVEX_TEST_ASSERT(row && row->dtype == YVEX_NATIVE_DTYPE_F8_E8M0,
                     "E8M0 scale dtype parses canonically");
    yvex_native_weight_table_close(table);

    rc = yvex_native_weight_table_open(&table, &(yvex_native_weight_options){ "build/tests/missing-native-source", 1, 0 }, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "missing source dir fails");

    system("rm -rf build/tests/native-weights-fixture");
    YVEX_TEST_ASSERT(make_dir(root), "make duplicate root");
    YVEX_TEST_ASSERT(write_safetensors("build/tests/native-weights-fixture/model-00001.safetensors",
        "{\"dup.weight\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}", 2), "write dup shard 1");
    YVEX_TEST_ASSERT(write_safetensors("build/tests/native-weights-fixture/model-00002.safetensors",
        "{\"dup.weight\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}", 2), "write dup shard 2");
    rc = yvex_native_weight_table_open(&table, &options, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "duplicate tensor detection");

    {
        yvex_native_weight_table empty_table;
        memset(&empty_table, 0, sizeof(empty_table));
        YVEX_TEST_ASSERT(
            yvex_native_weight_table_finalize(&empty_table, &err) == YVEX_OK,
            "empty indexed inventory finalizes without a null sort base");
        YVEX_TEST_ASSERT(empty_table.finalized == 1,
                         "empty indexed inventory records finalization");
        free(empty_table.name_slots);
    }

    {
        yvex_native_weight_table overflow_table;
        unsigned long long dims[1] = {1u};
        memset(&overflow_table, 0, sizeof(overflow_table));
        overflow_table.count = ULLONG_MAX;
        YVEX_TEST_ASSERT(
            yvex_native_weight_table_add(
                &overflow_table, "overflow.weight", "overflow.safetensors",
                "F32", 1u, dims, 0u, 4u, &err) == YVEX_ERR_BOUNDS,
            "indexed inventory count overflow fails closed");
    }

    return 0;
}
