
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int write_u64_le(FILE *fp, unsigned long long value)
{
    unsigned int i;

    for (i = 0; i < 8; ++i) {
        if (fputc((int)((value >> (i * 8u)) & 0xffu), fp) == EOF) return 0;
    }
    return 1;
}

static int write_safetensors(const char *path, const char *tensor_name)
{
    char header[512];
    FILE *fp;
    size_t header_len;

    snprintf(header, sizeof(header),
             "{\"__metadata__\":{\"format\":\"pt\"},\"%s\":{\"dtype\":\"F16\",\"shape\":[8,4],\"data_offsets\":[0,64]}}",
             tensor_name);
    header_len = strlen(header);
    fp = fopen(path, "wb");
    if (!fp) return 0;
    if (!write_u64_le(fp, (unsigned long long)header_len)) {
        fclose(fp);
        return 0;
    }
    if (fwrite(header, 1, header_len, fp) != header_len) {
        fclose(fp);
        return 0;
    }
    if (fwrite("0000000000000000000000000000000000000000000000000000000000000000", 1, 64, fp) != 64) {
        fclose(fp);
        return 0;
    }
    return fclose(fp) == 0;
}

static int test_names(void)
{
    YVEX_TEST_ASSERT(strcmp(yvex_weight_mapping_status_name(YVEX_WEIGHT_MAPPING_STATUS_MAPPED), "mapped") == 0,
                     "mapped status name");
    YVEX_TEST_ASSERT(strcmp(yvex_weight_mapping_status_name(YVEX_WEIGHT_MAPPING_STATUS_UNMAPPED), "unmapped") == 0,
                     "unmapped status name");
    YVEX_TEST_ASSERT(strcmp(yvex_weight_mapping_issue_kind_name(YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_NATIVE_NAME),
                            "unknown_native_name") == 0,
                     "issue name");
    return 0;
}

static int test_invalid_arch_fails(void)
{
    yvex_weight_mapping_options options;
    yvex_weight_mapping_table *table = NULL;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.architecture = "badarch";
    options.native_source_dir = ".";
    yvex_error_clear(&err);
    rc = yvex_weight_mapping_table_build(&table, &options, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "invalid arch fails");
    yvex_weight_mapping_table_close(table);
    return 0;
}

static int test_unmapped_behavior(void)
{
    const char *dir = "build/tests/weight-mapping-unknown";
    const char *path = "build/tests/weight-mapping-unknown/model-00001.safetensors";
    yvex_weight_mapping_options options;
    yvex_weight_mapping_table *table = NULL;
    const yvex_weight_mapping_info *row;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/weight-mapping-unknown && mkdir -p build/tests/weight-mapping-unknown") == 0,
                     "create unknown fixture dir");
    YVEX_TEST_ASSERT(write_safetensors(path, "unknown.weight"), "write unknown safetensors");

    memset(&options, 0, sizeof(options));
    options.architecture = "deepseek4";
    options.native_source_dir = dir;
    yvex_error_clear(&err);
    rc = yvex_weight_mapping_table_build(&table, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open mapping table");
    YVEX_TEST_ASSERT(yvex_weight_mapping_table_count(table) == 1, "one mapping row");
    row = yvex_weight_mapping_table_find_native(table, "unknown.weight");
    YVEX_TEST_ASSERT(row != NULL, "find unknown row");
    YVEX_TEST_ASSERT(row->status == YVEX_WEIGHT_MAPPING_STATUS_UNMAPPED, "unknown row unmapped");
    YVEX_TEST_ASSERT(row->issue == YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_NATIVE_NAME, "unknown row issue");
    yvex_weight_mapping_table_close(table);
    return 0;
}

static int test_embed_maps_with_template_transpose(void)
{
    const char *dir = "build/tests/weight-mapping-embed";
    const char *path = "build/tests/weight-mapping-embed/model-00001.safetensors";
    yvex_weight_mapping_options options;
    yvex_weight_mapping_table *table = NULL;
    const yvex_weight_mapping_info *row;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/weight-mapping-embed && mkdir -p build/tests/weight-mapping-embed") == 0,
                     "create embed fixture dir");
    YVEX_TEST_ASSERT(write_safetensors(path, "embed.weight"), "write embed safetensors");

    memset(&options, 0, sizeof(options));
    options.architecture = "deepseek4";
    options.native_source_dir = dir;
    options.template_path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.compare_template = 1;
    yvex_error_clear(&err);
    rc = yvex_weight_mapping_table_build(&table, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open mapping table with template");
    row = yvex_weight_mapping_table_find_native(table, "embed.weight");
    YVEX_TEST_ASSERT(row != NULL, "find embed row");
    YVEX_TEST_ASSERT(row->role == YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, "embed role");
    YVEX_TEST_ASSERT(strcmp(row->target_name, "token_embd.weight") == 0, "embed target");
    YVEX_TEST_ASSERT(row->status == YVEX_WEIGHT_MAPPING_STATUS_MAPPED, "embed mapped");
    YVEX_TEST_ASSERT(row->target_rank == 2, "target rank");
    YVEX_TEST_ASSERT(row->requires_transpose == 1, "transpose required");
    yvex_weight_mapping_table_close(table);
    return 0;
}

int yvex_test_weight_mapping(void)
{
    if (test_names() != 0) return 1;
    if (test_invalid_arch_fails() != 0) return 1;
    if (test_unmapped_behavior() != 0) return 1;
    if (test_embed_maps_with_template_transpose() != 0) return 1;
    return 0;
}
