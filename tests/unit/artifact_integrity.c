
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <yvex/api.h>

#include "tests/test.h"

static int first_issue_is(const yvex_artifact_integrity_report *report,
                          const char *code)
{
    const yvex_integrity_issue *issue = yvex_artifact_integrity_issue_at(report, 0);

    return issue && strcmp(issue->code, code) == 0;
}

static void append_u32(unsigned char *buf, size_t *len, unsigned int value)
{
    buf[(*len)++] = (unsigned char)(value & 0xffu);
    buf[(*len)++] = (unsigned char)((value >> 8) & 0xffu);
    buf[(*len)++] = (unsigned char)((value >> 16) & 0xffu);
    buf[(*len)++] = (unsigned char)((value >> 24) & 0xffu);
}

static void append_u64(unsigned char *buf, size_t *len, unsigned long long value)
{
    unsigned int i;

    for (i = 0; i < 8u; ++i) {
        buf[(*len)++] = (unsigned char)((value >> (i * 8u)) & 0xffu);
    }
}

static void append_bytes(unsigned char *buf, size_t *len, const void *data, size_t data_len)
{
    memcpy(buf + *len, data, data_len);
    *len += data_len;
}

static void append_string(unsigned char *buf, size_t *len, const char *text)
{
    size_t text_len = strlen(text);

    append_u64(buf, len, (unsigned long long)text_len);
    append_bytes(buf, len, text, text_len);
}

static void append_metadata_string(unsigned char *buf, size_t *len,
                                   const char *key,
                                   const char *value)
{
    append_string(buf, len, key);
    append_u32(buf, len, 8u);
    append_string(buf, len, value);
}

static void append_tensor(unsigned char *buf, size_t *len,
                          const char *name,
                          unsigned long long offset)
{
    append_string(buf, len, name);
    append_u32(buf, len, 2u);
    append_u64(buf, len, 4ull);
    append_u64(buf, len, 8ull);
    append_u32(buf, len, 0u);
    append_u64(buf, len, offset);
}

static int write_duplicate_tensor_fixture(const char *path)
{
    unsigned char buf[1024];
    size_t len = 0u;
    FILE *fp;

    append_bytes(buf, &len, "GGUF", 4u);
    append_u32(buf, &len, 3u);
    append_u64(buf, &len, 2ull);
    append_u64(buf, &len, 1ull);
    append_metadata_string(buf, &len, "general.architecture", "deepseek");
    append_tensor(buf, &len, "token_embd.weight", 0ull);
    append_tensor(buf, &len, "token_embd.weight", 128ull);
    while ((len % 32u) != 0u) {
        buf[len++] = 0u;
    }
    memset(buf + len, 0, 256u);
    len += 256u;

    fp = fopen(path, "wb");
    if (!fp) {
        return 0;
    }
    if (fwrite(buf, 1u, len, fp) != len) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

static int write_one_byte_short_range_fixture(const char *path)
{
    unsigned char buf[512];
    size_t len = 0u;
    FILE *fp;

    append_bytes(buf, &len, "GGUF", 4u);
    append_u32(buf, &len, 3u);
    append_u64(buf, &len, 1ull);
    append_u64(buf, &len, 1ull);
    append_metadata_string(buf, &len, "general.architecture", "deepseek");
    append_tensor(buf, &len, "token_embd.weight", 0ull);
    while ((len % 32u) != 0u) {
        buf[len++] = 0u;
    }
    memset(buf + len, 0, 127u);
    len += 127u;

    fp = fopen(path, "wb");
    if (!fp) {
        return 0;
    }
    if (fwrite(buf, 1u, len, fp) != len) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

static int write_canonical_zero_tensor_fixture(const char *path)
{
    unsigned char buf[32] = {0};
    size_t len = 0u;
    FILE *fp;
    append_bytes(buf, &len, "GGUF", 4u);
    append_u32(buf, &len, 3u);
    append_u64(buf, &len, 0ull);
    append_u64(buf, &len, 0ull);
    fp = fopen(path, "wb");
    if (!fp) return 0;
    if (fwrite(buf, 1u, sizeof(buf), fp) != sizeof(buf)) {
        fclose(fp);
        return 0;
    }
    return fclose(fp) == 0;
}

static int test_valid_fixture_passes(void)
{
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_artifact_integrity_check_path("tests/fixtures/gguf/valid-tokenizer-simple.gguf",
                                            NULL,
                                            &report,
                                            &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "valid tiny GGUF passes");
    YVEX_TEST_ASSERT(report.passed == 1, "valid report passes");
    YVEX_TEST_ASSERT(report.file_size > 0, "valid file size reported");
    YVEX_TEST_ASSERT(report.tensor_count == 1, "valid tensor count");
    YVEX_TEST_ASSERT(report.known_tensor_bytes == 128, "valid tensor bytes");
    YVEX_TEST_ASSERT(report.tensor_shapes_checked == 1, "valid shape checked");
    YVEX_TEST_ASSERT(report.tensor_shapes_valid == 1, "valid shape valid");
    YVEX_TEST_ASSERT(report.tensor_shapes_invalid == 0, "valid shape invalid count");
    YVEX_TEST_ASSERT(report.tensor_dtypes_checked == 1, "valid dtype checked");
    YVEX_TEST_ASSERT(report.tensor_dtypes_valid == 1, "valid dtype valid");
    YVEX_TEST_ASSERT(report.tensor_dtypes_invalid == 0, "valid dtype invalid count");
    YVEX_TEST_ASSERT(report.tensor_byte_counts_checked == 1, "valid byte count checked");
    YVEX_TEST_ASSERT(report.tensor_byte_counts_invalid == 0, "valid byte count invalid count");
    YVEX_TEST_ASSERT(report.tensor_ranges_checked == 1, "valid range checked");
    YVEX_TEST_ASSERT(report.tensor_ranges_valid == 1, "valid range valid");
    YVEX_TEST_ASSERT(report.tensor_ranges_invalid == 0, "valid range invalid count");
    return 0;
}

static int test_bad_magic_fails(void)
{
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_artifact_integrity_check_path("tests/fixtures/gguf/bad-magic.gguf",
                                            NULL,
                                            &report,
                                            &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "bad magic fails");
    YVEX_TEST_ASSERT(first_issue_is(&report, "bad-magic"), "bad magic code");
    return 0;
}

static int test_range_out_of_file_fails(void)
{
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_artifact_integrity_check_path("tests/fixtures/gguf/tensor-offset-out-of-bounds.gguf",
                                            NULL,
                                            &report,
                                            &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "range out of file fails");
    YVEX_TEST_ASSERT(first_issue_is(&report, "first-offset-not-zero"), "global offset code");
    return 0;
}

static int test_one_byte_short_range_reports_range_fields(void)
{
    const char *path = "build/tests/artifact-integrity/range-one-byte-short.gguf";
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    (void)mkdir("build", 0777);
    (void)mkdir("build/tests", 0777);
    (void)mkdir("build/tests/artifact-integrity", 0777);
    remove(path);
    YVEX_TEST_ASSERT(write_one_byte_short_range_fixture(path), "write one-byte short fixture");
    yvex_error_clear(&err);
    rc = yvex_artifact_integrity_check_path(path, NULL, &report, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "one-byte short range fails");
    YVEX_TEST_ASSERT(first_issue_is(&report, "tensor-payload-truncated"),
                     "one-byte short range code");
    YVEX_TEST_ASSERT(report.tensor_ranges_checked == 1, "range checked count");
    YVEX_TEST_ASSERT(report.tensor_ranges_invalid == 1, "range invalid count");
    YVEX_TEST_ASSERT(report.issues[0].has_range == 1, "range fields present");
    YVEX_TEST_ASSERT(report.issues[0].file_size > 0, "range file size present");
    return 0;
}

static int test_tensor_range_helper_validates_token_slices(void)
{
    const char *path = "build/tests/artifact-integrity/range-helper-f16.gguf";
    yvex_gguf_emit_options emit_options;
    yvex_gguf_emit_summary summary;
    yvex_artifact_options artifact_options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *table = NULL;
    const yvex_tensor_info *tensor;
    yvex_tensor_shape_accounting accounting;
    yvex_selected_embedding_shape selected_shape;
    yvex_tensor_range range;
    yvex_tensor_slice_range token0;
    yvex_tensor_slice_range token1;
    yvex_error err;
    int rc;

    memset(&emit_options, 0, sizeof(emit_options));
    memset(&summary, 0, sizeof(summary));
    yvex_error_clear(&err);
    (void)mkdir("build", 0777);
    (void)mkdir("build/tests", 0777);
    (void)mkdir("build/tests/artifact-integrity", 0777);
    remove(path);
    emit_options.out_path = path;
    emit_options.model_name = "range-helper-f16";
    emit_options.architecture = "deepseek";
    emit_options.target_qtype = "F16";
    emit_options.overwrite = 1;
    rc = yvex_gguf_emit_controlled(&emit_options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "emit range helper fixture");

    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = path;
    artifact_options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &artifact_options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open range helper artifact");
    rc = yvex_gguf_open(&gguf, artifact, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open range helper gguf");
    rc = yvex_tensor_table_from_gguf(&table, gguf, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open range helper tensor table");
    tensor = yvex_tensor_table_find(table, "token_embd.weight");
    YVEX_TEST_ASSERT(tensor != NULL, "range helper tensor exists");

    memset(&accounting, 0, sizeof(accounting));
    rc = yvex_tensor_shape_accounting_validate(tensor, &accounting, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "shape accounting validates");
    YVEX_TEST_ASSERT(accounting.shape_valid == 1, "shape accounting shape valid");
    YVEX_TEST_ASSERT(accounting.dtype_valid == 1, "shape accounting dtype valid");
    YVEX_TEST_ASSERT(accounting.byte_count_valid == 1, "shape accounting byte count valid");
    YVEX_TEST_ASSERT(accounting.element_count == 32, "shape accounting element count");
    YVEX_TEST_ASSERT(accounting.storage_unit_bytes == 2, "shape accounting storage unit bytes");
    YVEX_TEST_ASSERT(accounting.storage_byte_count == 64, "shape accounting storage bytes");
    YVEX_TEST_ASSERT(accounting.compute_supported_for_selected_embedding == 1,
                     "F16 selected embedding compute support flag");
    YVEX_TEST_ASSERT(accounting.compute_supported_for_fixture_embedding == 0,
                     "F16 fixture compute support flag");

    memset(&selected_shape, 0, sizeof(selected_shape));
    rc = yvex_selected_embedding_shape_validate(tensor, 0u, &selected_shape, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "selected embedding shape validates");
    YVEX_TEST_ASSERT(selected_shape.shape_valid == 1, "selected embedding shape valid flag");
    YVEX_TEST_ASSERT(selected_shape.hidden_size == 4, "selected embedding hidden size");
    YVEX_TEST_ASSERT(selected_shape.vocab_size == 8, "selected embedding vocab size");
    YVEX_TEST_ASSERT(selected_shape.output_count == 4, "selected embedding output count");
    YVEX_TEST_ASSERT(selected_shape.output_bytes == 16, "selected embedding output bytes");
    YVEX_TEST_ASSERT(selected_shape.slice_bytes == 8, "selected embedding slice bytes");

    memset(&range, 0, sizeof(range));
    rc = yvex_tensor_range_validate(artifact, gguf, tensor, &range, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "tensor range validates");
    YVEX_TEST_ASSERT(range.range_valid == 1, "range valid flag");
    YVEX_TEST_ASSERT(range.element_count == 32, "range element count");
    YVEX_TEST_ASSERT(range.dtype_size == 2, "range dtype size");
    YVEX_TEST_ASSERT(range.tensor_bytes == 64, "range tensor bytes");
    YVEX_TEST_ASSERT(range.tensor_absolute_offset >= range.tensor_data_offset,
                     "range absolute offset after data");
    YVEX_TEST_ASSERT(range.tensor_end_offset <= range.file_size,
                     "range end inside file");
    YVEX_TEST_ASSERT(range.aligned == 1, "range alignment");

    memset(&token0, 0, sizeof(token0));
    rc = yvex_tensor_embedding_slice_range_validate(&range, 0u, &token0, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "token 0 slice validates");
    YVEX_TEST_ASSERT(token0.range_valid == 1, "token 0 range valid");
    YVEX_TEST_ASSERT(token0.slice_bytes == 8, "token 0 slice bytes");
    YVEX_TEST_ASSERT(token0.slice_absolute_offset == range.tensor_absolute_offset,
                     "token 0 slice starts at tensor");
    YVEX_TEST_ASSERT(token0.slice_end_offset <= range.tensor_end_offset,
                     "token 0 slice inside tensor");

    memset(&token1, 0, sizeof(token1));
    rc = yvex_tensor_embedding_slice_range_validate(&range, 1u, &token1, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "token 1 slice validates");
    YVEX_TEST_ASSERT(token1.range_valid == 1, "token 1 range valid");
    YVEX_TEST_ASSERT(token1.slice_bytes == 8, "token 1 slice bytes");
    YVEX_TEST_ASSERT(token1.slice_absolute_offset == range.tensor_absolute_offset + 8ull,
                     "token 1 slice offset");
    YVEX_TEST_ASSERT(token1.slice_end_offset <= range.tensor_end_offset,
                     "token 1 slice inside tensor");

    yvex_tensor_table_close(table);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

static int test_selected_embedding_f32_dtype_fails(void)
{
    yvex_artifact_integrity_options options;
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.require_token_embedding = 1;
    options.token_id = 0u;
    yvex_error_clear(&err);
    rc = yvex_artifact_integrity_check_path("tests/fixtures/gguf/valid-tokenizer-simple.gguf",
                                            &options,
                                            &report,
                                            &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "F32 selected embedding readiness fails");
    YVEX_TEST_ASSERT(first_issue_is(&report, "required-tensor-dtype-invalid"),
                     "F32 selected embedding dtype code");
    return 0;
}

static int test_zero_dimension_fails(void)
{
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_artifact_integrity_check_path("tests/fixtures/gguf/tensor-dim-zero.gguf",
                                            NULL,
                                            &report,
                                            &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "zero dim fails");
    YVEX_TEST_ASSERT(first_issue_is(&report, "zero-dimension"), "zero dimension code");
    return 0;
}

static int test_required_tensor_missing_fails(void)
{
    const char *path = "build/tests/artifact-integrity/canonical-zero.gguf";
    yvex_artifact_integrity_options options;
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.require_token_embedding = 1;
    (void)mkdir("build", 0777);
    (void)mkdir("build/tests", 0777);
    (void)mkdir("build/tests/artifact-integrity", 0777);
    remove(path);
    YVEX_TEST_ASSERT(write_canonical_zero_tensor_fixture(path),
                     "write canonical zero tensor fixture");
    yvex_error_clear(&err);
    rc = yvex_artifact_integrity_check_path(path,
                                            &options,
                                            &report,
                                            &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "missing required tensor fails");
    YVEX_TEST_ASSERT(first_issue_is(&report, "required-tensor-missing"), "missing tensor code");
    remove(path);
    return 0;
}

static int test_token_out_of_range_fails(void)
{
    const char *path = "build/tests/artifact-integrity/controlled-f16.gguf";
    yvex_gguf_emit_options emit_options;
    yvex_gguf_emit_summary summary;
    yvex_artifact_integrity_options options;
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    memset(&emit_options, 0, sizeof(emit_options));
    memset(&summary, 0, sizeof(summary));
    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    (void)mkdir("build", 0777);
    (void)mkdir("build/tests", 0777);
    (void)mkdir("build/tests/artifact-integrity", 0777);
    remove(path);
    emit_options.out_path = path;
    emit_options.model_name = "integrity-f16-fixture";
    emit_options.architecture = "deepseek";
    emit_options.target_qtype = "F16";
    emit_options.overwrite = 1;
    rc = yvex_gguf_emit_controlled(&emit_options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "emit f16 fixture");

    options.require_token_embedding = 1;
    options.token_id = 99u;
    rc = yvex_artifact_integrity_check_path(path, &options, &report, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "token out of range fails");
    YVEX_TEST_ASSERT(first_issue_is(&report, "token-out-of-range"), "token range code");
    return 0;
}

static int test_duplicate_tensor_name_fails(void)
{
    const char *path = "build/tests/artifact-integrity/duplicate-tensor.gguf";
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    (void)mkdir("build", 0777);
    (void)mkdir("build/tests", 0777);
    (void)mkdir("build/tests/artifact-integrity", 0777);
    remove(path);
    YVEX_TEST_ASSERT(write_duplicate_tensor_fixture(path), "write duplicate fixture");
    yvex_error_clear(&err);
    rc = yvex_artifact_integrity_check_path(path, NULL, &report, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "duplicate tensor fails");
    YVEX_TEST_ASSERT(first_issue_is(&report, "duplicate-tensor-name"), "duplicate code");
    return 0;
}

static int test_qtype_rows_use_complete_shape(void)
{
    yvex_tensor_info tensor;
    yvex_tensor_shape_accounting accounting;
    yvex_error err;
    int rc;

    memset(&tensor, 0, sizeof(tensor));
    tensor.name = "q4.weight";
    tensor.dtype = YVEX_DTYPE_Q4_0;
    tensor.ggml_type = YVEX_GGUF_QTYPE_Q4_0;
    tensor.rank = 2u;
    tensor.dims[0] = 32ull;
    tensor.dims[1] = 2ull;
    tensor.storage_bytes = 36ull;

    yvex_error_clear(&err);
    rc = yvex_tensor_shape_accounting_validate(&tensor, &accounting, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "integrity accepts complete Q4_0 rows");
    YVEX_TEST_ASSERT(accounting.storage_byte_count == 36ull,
                     "integrity consumes canonical row bytes");

    tensor.dims[0] = 16ull;
    tensor.storage_bytes = 0ull;
    rc = yvex_tensor_shape_accounting_validate(&tensor, &accounting, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT,
                     "integrity refuses flattened-only block alignment");
    YVEX_TEST_ASSERT(strstr(yvex_error_message(&err),
                            "row-width-block-mismatch") != NULL,
                     "integrity preserves typed row refusal");
    return 0;
}

int yvex_test_artifact_integrity(void)
{
    if (test_valid_fixture_passes() != 0) return 1;
    if (test_bad_magic_fails() != 0) return 1;
    if (test_range_out_of_file_fails() != 0) return 1;
    if (test_one_byte_short_range_reports_range_fields() != 0) return 1;
    if (test_tensor_range_helper_validates_token_slices() != 0) return 1;
    if (test_selected_embedding_f32_dtype_fails() != 0) return 1;
    if (test_zero_dimension_fails() != 0) return 1;
    if (test_required_tensor_missing_fails() != 0) return 1;
    if (test_token_out_of_range_fails() != 0) return 1;
    if (test_duplicate_tensor_name_fails() != 0) return 1;
    if (test_qtype_rows_use_complete_shape() != 0) return 1;
    return 0;
}
