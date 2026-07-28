/*
 * YVEX - Imatrix manifest tests
 *
 * File: tests/test_imatrix.c
 * Layer: test
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int write_policy(const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fprintf(fp,
            "{\n"
            "  \"schema\": \"yvex.quant_policy.v1\",\n"
            "  \"name\": \"test-policy\",\n"
            "  \"architecture\": \"deepseek4\",\n"
            "  \"rules\": [\n"
            "    {\"selector_kind\": \"pattern\", \"selector\": \"blk.*.ffn.experts.*\", \"qtype\": \"Q2_K\", \"requires_imatrix\": true}\n"
            "  ]\n"
            "}\n");
    return fclose(fp) == 0;
}

static int write_file(const char *path, const char *text)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fputs(text, fp);
    return fclose(fp) == 0;
}

static int write_u32(FILE *fp, unsigned int value)
{
    unsigned char bytes[4] = {
        (unsigned char)value,
        (unsigned char)(value >> 8),
        (unsigned char)(value >> 16),
        (unsigned char)(value >> 24)
    };
    return fwrite(bytes, 1u, sizeof(bytes), fp) == sizeof(bytes);
}

static int write_f32(FILE *fp, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return write_u32(fp, bits);
}

static int write_data_entry(FILE *fp, const char *name, unsigned int values, float base)
{
    unsigned int index;
    size_t length = strlen(name);
    if (!write_u32(fp, (unsigned int)length) || fwrite(name, 1u, length, fp) != length ||
        !write_u32(fp, 3u) || !write_u32(fp, values))
        return 0;
    for (index = 0u; index < values; ++index)
        if (!write_f32(fp, base + (float)index * 0.25f))
            return 0;
    return 1;
}

static int write_imatrix_data(const char *path)
{
    const char *dataset = "tests/reference/calibration.txt";
    FILE *fp = fopen(path, "wb");
    int ok;
    if (!fp) return 0;
    ok = write_u32(fp, 2u) &&
         write_data_entry(fp, "blk.0.ffn_gate_exps.weight", 8u, 1.0f) &&
         write_data_entry(fp, "blk.0.ffn_down_exps.weight", 4u, 2.0f) &&
         write_u32(fp, 17u) && write_u32(fp, (unsigned int)strlen(dataset)) &&
         fwrite(dataset, 1u, strlen(dataset), fp) == strlen(dataset);
    return fclose(fp) == 0 && ok;
}

static int write_duplicate_imatrix_data(const char *path)
{
    const char *dataset = "tests/reference/calibration.txt";
    FILE *fp = fopen(path, "wb");
    int ok;
    if (!fp) return 0;
    ok = write_u32(fp, 2u) &&
         write_data_entry(fp, "blk.0.ffn_gate_exps.weight", 8u, 1.0f) &&
         write_data_entry(fp, "blk.0.ffn_gate_exps.weight", 8u, 2.0f) &&
         write_u32(fp, 17u) && write_u32(fp, (unsigned int)strlen(dataset)) &&
         fwrite(dataset, 1u, strlen(dataset), fp) == strlen(dataset);
    return fclose(fp) == 0 && ok;
}

static int test_data_snapshot(void)
{
    const char *path = "build/tests/imatrix/weights.dat";
    yvex_imatrix_data_options options;
    yvex_imatrix_data *data = NULL;
    yvex_imatrix_data_summary summary;
    yvex_imatrix_entry_summary entry;
    float values[3];
    FILE *fp;
    yvex_error err;

    YVEX_TEST_ASSERT(write_imatrix_data(path), "write canonical imatrix data");
    memset(&options, 0, sizeof(options));
    options.path = path;
    options.source_model_identity = "source-model-identity";
    options.calibration_dataset_identity = "calibration-dataset-identity";
    options.producer = "llama.cpp";
    options.producer_version = 1u;
    options.maximum_mapped_bytes = 1024u * 1024u;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_imatrix_data_open(&data, &options, &err) == YVEX_OK,
                     "open imatrix data");
    YVEX_TEST_ASSERT(yvex_imatrix_data_get_summary(data, &summary, &err) == YVEX_OK,
                     "imatrix data summary");
    YVEX_TEST_ASSERT(summary.complete && summary.snapshot_stable, "stable complete data");
    YVEX_TEST_ASSERT(summary.entry_count == 2u && summary.value_count == 12u,
                     "entry and value counts");
    YVEX_TEST_ASSERT(summary.calibration_chunk_count == 17u, "calibration chunks");
    YVEX_TEST_ASSERT_STREQ(summary.dataset_name, "tests/reference/calibration.txt",
                           "dataset trailer");
    YVEX_TEST_ASSERT(yvex_imatrix_data_find(data, "blk.0.ffn_gate_exps.weight", &entry,
                                            &err) == YVEX_OK,
                     "find imatrix entry");
    YVEX_TEST_ASSERT(entry.value_count == 8u, "entry extent");
    YVEX_TEST_ASSERT(yvex_imatrix_data_read(data, entry.ordinal, 2u, values, 3u, &err) ==
                         YVEX_OK,
                     "read imatrix interval");
    YVEX_TEST_ASSERT(values[0] == 1.5f && values[2] == 2.0f, "decoded little-endian values");
    fp = fopen(path, "ab");
    YVEX_TEST_ASSERT(fp && fputc(0, fp) != EOF && fclose(fp) == 0, "mutate admitted file");
    YVEX_TEST_ASSERT(yvex_imatrix_data_validate(data, &err) != YVEX_OK,
                     "snapshot drift refusal");
    yvex_imatrix_data_close(data);
    return 0;
}

static int test_duplicate_data_entry(void)
{
    const char *path = "build/tests/imatrix/duplicate.dat";
    yvex_imatrix_data_options options;
    yvex_imatrix_data *data = NULL;
    yvex_error err;

    YVEX_TEST_ASSERT(write_duplicate_imatrix_data(path), "write duplicate imatrix data");
    memset(&options, 0, sizeof(options));
    options.path = path;
    options.source_model_identity = "source-model-identity";
    options.calibration_dataset_identity = "calibration-dataset-identity";
    options.producer = "llama.cpp";
    options.producer_version = 1u;
    options.maximum_mapped_bytes = 1024u * 1024u;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_imatrix_data_open(&data, &options, &err) == YVEX_ERR_FORMAT,
                     "duplicate imatrix entry refuses");
    YVEX_TEST_ASSERT(data == NULL, "duplicate imatrix publishes no owner");
    return 0;
}

static int test_names(void)
{
    YVEX_TEST_ASSERT_STREQ(yvex_imatrix_status_name(YVEX_IMATRIX_STATUS_PRESENT), "present", "status name");
    YVEX_TEST_ASSERT_STREQ(yvex_imatrix_format_name(YVEX_IMATRIX_FORMAT_ROUTED_MOE_DAT), "routed_moe_dat", "format name");
    YVEX_TEST_ASSERT_STREQ(yvex_imatrix_coverage_kind_name(YVEX_IMATRIX_COVERAGE_ROUTED_MOE), "routed_moe", "coverage name");
    YVEX_TEST_ASSERT_STREQ(yvex_imatrix_issue_kind_name(YVEX_IMATRIX_ISSUE_FILE_MISSING), "file_missing", "issue name");
    YVEX_TEST_ASSERT(yvex_imatrix_status_from_name("present") == YVEX_IMATRIX_STATUS_PRESENT, "status parse");
    YVEX_TEST_ASSERT(yvex_imatrix_format_from_name("routed_moe_dat") == YVEX_IMATRIX_FORMAT_ROUTED_MOE_DAT, "format parse");
    return 0;
}

static int test_create_write_open_present(void)
{
    const char *dir = "build/tests/imatrix";
    const char *policy_path = "build/tests/imatrix/policy.json";
    const char *dat_path = "build/tests/imatrix/fake.dat";
    const char *manifest_path = "build/tests/imatrix/imatrix.json";
    yvex_imatrix_manifest_options options;
    yvex_imatrix_manifest *manifest = NULL;
    yvex_imatrix_manifest *opened = NULL;
    yvex_imatrix_summary summary;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/imatrix && mkdir -p build/tests/imatrix") == 0,
                     "create imatrix dir");
    (void)dir;
    YVEX_TEST_ASSERT(write_policy(policy_path), "write policy");
    YVEX_TEST_ASSERT(write_file(dat_path, "fake-imatrix"), "write fake dat");

    memset(&options, 0, sizeof(options));
    options.name = "test-imatrix";
    options.architecture = "deepseek4";
    options.quant_policy_path = policy_path;
    options.imatrix_path = dat_path;
    options.calibration_dataset = "test-dataset";
    options.producer = "test";
    options.format = YVEX_IMATRIX_FORMAT_ROUTED_MOE_DAT;
    options.status = YVEX_IMATRIX_STATUS_PRESENT;

    yvex_error_clear(&err);
    rc = yvex_imatrix_manifest_create(&manifest, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "create manifest");
    YVEX_TEST_ASSERT(yvex_imatrix_manifest_validate(manifest, &err) == YVEX_OK, "validate manifest");
    YVEX_TEST_ASSERT(yvex_imatrix_manifest_get_summary(manifest, &summary, &err) == YVEX_OK, "summary");
    YVEX_TEST_ASSERT(summary.file_exists == 1, "file exists");
    YVEX_TEST_ASSERT(summary.requires_imatrix_rule_count == 1, "requires rule count");
    YVEX_TEST_ASSERT(summary.covered_rule_count == 1, "covered rule count");
    YVEX_TEST_ASSERT(summary.uncovered_rule_count == 0, "uncovered rule count");
    YVEX_TEST_ASSERT(yvex_imatrix_manifest_write_json(manifest_path, manifest, &err) == YVEX_OK, "write manifest");
    yvex_imatrix_manifest_close(manifest);

    rc = yvex_imatrix_manifest_open(&opened, manifest_path, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open manifest");
    YVEX_TEST_ASSERT(yvex_imatrix_manifest_get_summary(opened, &summary, &err) == YVEX_OK, "opened summary");
    YVEX_TEST_ASSERT_STREQ(summary.name, "test-imatrix", "opened name");
    YVEX_TEST_ASSERT(summary.file_exists == 1, "opened file exists");
    yvex_imatrix_manifest_close(opened);
    return 0;
}

static int test_missing_file(void)
{
    yvex_imatrix_manifest_options options;
    yvex_imatrix_manifest *manifest = NULL;
    yvex_imatrix_summary summary;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.name = "missing-imatrix";
    options.architecture = "deepseek4";
    options.imatrix_path = "build/tests/imatrix/missing.dat";
    options.format = YVEX_IMATRIX_FORMAT_ROUTED_MOE_DAT;
    options.status = YVEX_IMATRIX_STATUS_PRESENT;

    yvex_error_clear(&err);
    rc = yvex_imatrix_manifest_create(&manifest, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "create missing manifest");
    YVEX_TEST_ASSERT(yvex_imatrix_manifest_validate(manifest, &err) == YVEX_OK, "validate missing manifest");
    YVEX_TEST_ASSERT(yvex_imatrix_manifest_get_summary(manifest, &summary, &err) == YVEX_OK, "missing summary");
    YVEX_TEST_ASSERT(summary.file_exists == 0, "missing file exists false");
    YVEX_TEST_ASSERT(summary.status == YVEX_IMATRIX_STATUS_MISSING, "missing status");
    YVEX_TEST_ASSERT(summary.issue_count > 0, "missing issue");
    yvex_imatrix_manifest_close(manifest);
    return 0;
}

static int test_invalid_args(void)
{
    yvex_imatrix_manifest_options options;
    yvex_imatrix_manifest *manifest = NULL;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    rc = yvex_imatrix_manifest_create(&manifest, &options, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "invalid args fail");
    return 0;
}

int yvex_test_imatrix(void)
{
    if (test_names() != 0) return 1;
    if (test_create_write_open_present() != 0) return 1;
    if (test_data_snapshot() != 0) return 1;
    if (test_duplicate_data_entry() != 0) return 1;
    if (test_missing_file() != 0) return 1;
    if (test_invalid_args() != 0) return 1;
    return 0;
}
