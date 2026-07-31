
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/api.h>
#include <yvex/internal/artifact.h>

#include "tests/test.h"
#include <yvex/gguf.h>

#define LAYOUT_TEST_DIR "build/tests/gguf-layout"

typedef struct {
    const char *name;
    unsigned int qtype;
    unsigned int rank;
    unsigned long long dims[4];
    unsigned long long offset;
} layout_tensor_spec;

static int write_bytes(FILE *fp, const void *data, size_t len)
{
    return fwrite(data, 1u, len, fp) == len;
}

static int write_u32(FILE *fp, unsigned int value)
{
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
    return write_bytes(fp, bytes, sizeof(bytes));
}

static int write_u64(FILE *fp, unsigned long long value)
{
    unsigned char bytes[8];
    unsigned int i;
    for (i = 0u; i < 8u; ++i) {
        bytes[i] = (unsigned char)((value >> (i * 8u)) & 0xffu);
    }
    return write_bytes(fp, bytes, sizeof(bytes));
}

static int write_string(FILE *fp, const char *text)
{
    size_t len = strlen(text);
    return write_u64(fp, (unsigned long long)len) && write_bytes(fp, text, len);
}

static int write_header(FILE *fp, unsigned long long tensors)
{
    return write_bytes(fp, "GGUF", 4u) && write_u32(fp, 3u) &&
           write_u64(fp, tensors) && write_u64(fp, 1ull);
}

static int write_alignment_metadata(FILE *fp, unsigned int alignment)
{
    return write_string(fp, "general.alignment") &&
           write_u32(fp, YVEX_GGUF_VALUE_UINT32) && write_u32(fp, alignment);
}

static int write_tensor_record(FILE *fp, const layout_tensor_spec *tensor)
{
    unsigned int i;
    if (!write_string(fp, tensor->name) || !write_u32(fp, tensor->rank)) return 0;
    for (i = 0u; i < tensor->rank; ++i) {
        if (!write_u64(fp, tensor->dims[i])) return 0;
    }
    return write_u32(fp, tensor->qtype) && write_u64(fp, tensor->offset);
}

static unsigned long long align_generic(unsigned long long value,
                                        unsigned int alignment)
{
    unsigned long long remainder = value % (unsigned long long)alignment;
    return remainder == 0ull ? value : value + (unsigned long long)alignment - remainder;
}

static int tensor_size(const layout_tensor_spec *tensor,
                       unsigned long long *out)
{
    yvex_gguf_qtype_storage_result storage;
    return yvex_gguf_qtype_tensor_storage(tensor->qtype, tensor->dims,
                                          tensor->rank, &storage) ==
               YVEX_GGUF_QTYPE_STORAGE_OK
               ? (*out = storage.total_bytes, 1) : 0;
}

/*
 * Writes metadata and directory bytes, then creates a sparse data span. Payload
 * holes remain unread by layout validation; only requested corrupt padding is
 * materialized.
 */
static int write_layout_fixture(const char *path,
                                unsigned int alignment,
                                const layout_tensor_spec *tensors,
                                size_t tensor_count,
                                unsigned long long data_span,
                                unsigned long long tail_bytes,
                                int nonzero_directory_padding,
                                int nonzero_tensor_padding)
{
    FILE *fp = fopen(path, "wb+");
    unsigned long long directory_end;
    unsigned long long data_offset;
    unsigned long long final_size;
    size_t i;
    int ok = fp != NULL;

    if (!fp) return 0;
    ok = write_header(fp, (unsigned long long)tensor_count) &&
         write_alignment_metadata(fp, alignment);
    for (i = 0u; ok && i < tensor_count; ++i) ok = write_tensor_record(fp, &tensors[i]);
    directory_end = (unsigned long long)ftello(fp);
    data_offset = align_generic(directory_end, alignment);
    while (ok && (unsigned long long)ftello(fp) < data_offset) {
        unsigned char value = nonzero_directory_padding &&
                              (unsigned long long)ftello(fp) == directory_end ? 1u : 0u;
        ok = write_bytes(fp, &value, 1u);
    }
    final_size = data_offset + data_span + tail_bytes;
    ok = ok && fflush(fp) == 0 && final_size <= (unsigned long long)LLONG_MAX &&
         ftruncate(fileno(fp), (off_t)final_size) == 0;
    if (ok && nonzero_tensor_padding && tensor_count > 0u) {
        unsigned long long raw_size;
        unsigned long long padding_offset;
        unsigned char value = 0x5au;
        ok = tensor_size(&tensors[0], &raw_size);
        padding_offset = data_offset + tensors[0].offset + raw_size;
        ok = ok && padding_offset < final_size &&
             pwrite(fileno(fp), &value, 1u, (off_t)padding_offset) == 1;
    }
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int run_layout(const char *path, yvex_gguf_layout_result *result)
{
    yvex_artifact_options options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_error err;
    int rc;
    memset(&options, 0, sizeof(options));
    options.path = path;
    options.readonly = 1;
    yvex_error_clear(&err);
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_layout_validate(artifact, gguf, result, &err);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc;
}

static int expect_layout_code(const char *path, yvex_gguf_layout_code code)
{
    yvex_gguf_layout_result result;
    int rc = run_layout(path, &result);
    return rc != YVEX_OK && result.code == code && !result.accepted &&
           result.tensor_payload_bytes_read == 0ull;
}

static int prepare_dir(void)
{
    (void)mkdir("build", 0777);
    (void)mkdir("build/tests", 0777);
    return mkdir(LAYOUT_TEST_DIR, 0777) == 0 || access(LAYOUT_TEST_DIR, F_OK) == 0;
}

static int test_valid_mixed_layout(void)
{
    const char *path = LAYOUT_TEST_DIR "/valid-mixed.gguf";
    const layout_tensor_spec tensors[] = {
        {"layout.f32", YVEX_GGUF_QTYPE_F32, 1u, {1ull}, 0ull},
        {"layout.q4", YVEX_GGUF_QTYPE_Q4_0, 2u, {32ull, 2ull}, 32ull},
        {"layout.f16", YVEX_GGUF_QTYPE_F16, 1u, {3ull}, 96ull}
    };
    yvex_gguf_layout_result result;
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, tensors, 3u, 128ull,
                                          0ull, 0, 0), "write valid mixed layout");
    YVEX_TEST_ASSERT(run_layout(path, &result) == YVEX_OK, "valid mixed layout passes");
    YVEX_TEST_ASSERT(result.accepted && result.code == YVEX_GGUF_LAYOUT_OK,
                     "valid mixed typed result");
    YVEX_TEST_ASSERT(result.tensors_validated == 3ull, "all mixed tensors validated");
    YVEX_TEST_ASSERT(result.raw_tensor_bytes == 46ull, "mixed raw byte total");
    YVEX_TEST_ASSERT(result.data_section_span == 128ull, "mixed padded data span");
    YVEX_TEST_ASSERT(result.tensor_payload_bytes_read == 0ull, "mixed payload unread");
    YVEX_TEST_ASSERT(result.padding_bytes_read == result.total_padding_bytes,
                     "only required padding is read");
    remove(path);
    return 0;
}

static int test_alignment_matrix(void)
{
    const unsigned int valid[] = {8u, 16u, 32u, 64u};
    const char *path = LAYOUT_TEST_DIR "/alignment.gguf";
    size_t i;
    for (i = 0u; i < sizeof(valid) / sizeof(valid[0]); ++i) {
        yvex_gguf_layout_result result;
        remove(path);
        YVEX_TEST_ASSERT(write_layout_fixture(path, valid[i], NULL, 0u, 0ull,
                                              0ull, 0, 0), "write valid alignment");
        YVEX_TEST_ASSERT(run_layout(path, &result) == YVEX_OK,
                         "power-of-two alignment passes");
        YVEX_TEST_ASSERT(result.alignment == valid[i], "alignment preserved");
    }
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 24u, NULL, 0u, 0ull,
                                          0ull, 0, 0), "write alignment 24");
    YVEX_TEST_ASSERT(expect_layout_code(path,
                                       YVEX_GGUF_LAYOUT_ALIGNMENT_NOT_POWER_OF_TWO),
                     "alignment 24 fails closed");
    remove(path);
    return 0;
}

static int test_offset_failure_matrix(void)
{
    const char *path = LAYOUT_TEST_DIR "/offset-case.gguf";
    layout_tensor_spec tensors[3];

    memset(tensors, 0, sizeof(tensors));
    tensors[0] = (layout_tensor_spec){"a", YVEX_GGUF_QTYPE_F32, 1u, {1ull}, 32ull};
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, tensors, 1u, 64ull, 0ull, 0, 0),
                     "write first-offset case");
    YVEX_TEST_ASSERT(expect_layout_code(path, YVEX_GGUF_LAYOUT_FIRST_OFFSET_NOT_ZERO),
                     "first offset must be zero");

    tensors[0] = (layout_tensor_spec){"a", YVEX_GGUF_QTYPE_F32, 1u, {1ull}, 0ull};
    tensors[1] = (layout_tensor_spec){"b", YVEX_GGUF_QTYPE_F32, 1u, {1ull}, 64ull};
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, tensors, 2u, 96ull, 0ull, 0, 0),
                     "write forward gap");
    YVEX_TEST_ASSERT(expect_layout_code(path, YVEX_GGUF_LAYOUT_UNEXPECTED_GAP),
                     "forward gap is distinct");

    tensors[1].offset = 0ull;
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, tensors, 2u, 32ull, 0ull, 0, 0),
                     "write duplicate start");
    YVEX_TEST_ASSERT(expect_layout_code(path, YVEX_GGUF_LAYOUT_DUPLICATE_START),
                     "duplicate start is distinct");

    tensors[0] = (layout_tensor_spec){"a", YVEX_GGUF_QTYPE_F32, 1u, {16ull}, 0ull};
    tensors[1] = (layout_tensor_spec){"b", YVEX_GGUF_QTYPE_F32, 1u, {16ull}, 32ull};
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, tensors, 2u, 96ull, 0ull, 0, 0),
                     "write partial overlap");
    YVEX_TEST_ASSERT(expect_layout_code(path, YVEX_GGUF_LAYOUT_PARTIAL_OVERLAP),
                     "partial overlap is distinct");

    tensors[0] = (layout_tensor_spec){"a", YVEX_GGUF_QTYPE_F32, 1u, {32ull}, 0ull};
    tensors[1] = (layout_tensor_spec){"b", YVEX_GGUF_QTYPE_F32, 1u, {1ull}, 32ull};
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, tensors, 2u, 128ull, 0ull, 0, 0),
                     "write complete overlap");
    YVEX_TEST_ASSERT(expect_layout_code(path, YVEX_GGUF_LAYOUT_COMPLETE_OVERLAP),
                     "complete overlap is distinct");

    tensors[0] = (layout_tensor_spec){"a", YVEX_GGUF_QTYPE_F32, 1u, {8ull}, 0ull};
    tensors[1] = (layout_tensor_spec){"b", YVEX_GGUF_QTYPE_F32, 1u, {16ull}, 32ull};
    tensors[2] = (layout_tensor_spec){"c", YVEX_GGUF_QTYPE_F32, 1u, {1ull}, 0ull};
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, tensors, 3u, 96ull, 0ull, 0, 0),
                     "write reversed order");
    YVEX_TEST_ASSERT(expect_layout_code(path, YVEX_GGUF_LAYOUT_OFFSET_REVERSED),
                     "reversed order is distinct");
    remove(path);
    return 0;
}

static int test_padding_truncation_and_tail(void)
{
    const char *path = LAYOUT_TEST_DIR "/span-case.gguf";
    const layout_tensor_spec tensor =
        {"span", YVEX_GGUF_QTYPE_F32, 1u, {1ull}, 0ull};

    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, &tensor, 1u, 32ull,
                                          0ull, 1, 0), "write nonzero directory padding");
    YVEX_TEST_ASSERT(expect_layout_code(path,
                                       YVEX_GGUF_LAYOUT_DIRECTORY_PADDING_NONZERO),
                     "nonzero directory padding fails");
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, &tensor, 1u, 32ull,
                                          0ull, 0, 1), "write nonzero tensor padding");
    YVEX_TEST_ASSERT(expect_layout_code(path, YVEX_GGUF_LAYOUT_TENSOR_PADDING_NONZERO),
                     "nonzero tensor padding fails");
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, &tensor, 1u, 3ull,
                                          0ull, 0, 0), "write payload truncation");
    YVEX_TEST_ASSERT(expect_layout_code(path,
                                       YVEX_GGUF_LAYOUT_TENSOR_PAYLOAD_TRUNCATED),
                     "one-byte payload truncation fails");
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, &tensor, 1u, 31ull,
                                          0ull, 0, 0), "write padding truncation");
    YVEX_TEST_ASSERT(expect_layout_code(path, YVEX_GGUF_LAYOUT_PADDING_TRUNCATED),
                     "final padding truncation fails");
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, &tensor, 1u, 32ull,
                                          1ull, 0, 0), "write trailing byte");
    YVEX_TEST_ASSERT(expect_layout_code(path, YVEX_GGUF_LAYOUT_NONCANONICAL_TAIL),
                     "trailing bytes are explicit noncanonical state");
    remove(path);
    return 0;
}

static int test_zero_tensor_file(void)
{
    const char *path = LAYOUT_TEST_DIR "/zero-tensor.gguf";
    yvex_gguf_layout_result result;
    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, NULL, 0u, 0ull,
                                          0ull, 0, 0), "write zero-tensor file");
    YVEX_TEST_ASSERT(run_layout(path, &result) == YVEX_OK, "zero-tensor layout passes");
    YVEX_TEST_ASSERT(result.tensor_count == 0ull && result.data_section_span == 0ull,
                     "zero-tensor span is explicit");
    YVEX_TEST_ASSERT(result.tensor_payload_bytes_read == 0ull, "zero payload reads");
    remove(path);
    return 0;
}

static int test_checked_arithmetic(void)
{
    unsigned long long raw_end = 0ull;
    unsigned long long padded_end = 0ull;
    unsigned long long total = 0ull;
    YVEX_TEST_ASSERT(yvex_gguf_layout_interval_measure(32ull, 36ull, 32u,
                                                       &raw_end, &padded_end) ==
                         YVEX_GGUF_LAYOUT_OK,
                     "interval arithmetic accepts normal span");
    YVEX_TEST_ASSERT(raw_end == 68ull && padded_end == 96ull,
                     "interval raw and padded ends");
    YVEX_TEST_ASSERT(yvex_gguf_layout_interval_measure(ULLONG_MAX - 3ull, 8ull,
                                                       32u, &raw_end, &padded_end) ==
                         YVEX_GGUF_LAYOUT_RAW_END_OVERFLOW,
                     "raw end overflow typed");
    YVEX_TEST_ASSERT(yvex_gguf_layout_interval_measure(ULLONG_MAX - 16ull, 1ull,
                                                       32u, &raw_end, &padded_end) ==
                         YVEX_GGUF_LAYOUT_PADDED_END_OVERFLOW,
                     "padded end overflow typed");
    YVEX_TEST_ASSERT(yvex_gguf_layout_sum_checked(ULLONG_MAX, 1ull, &total) ==
                         YVEX_GGUF_LAYOUT_AGGREGATE_OVERFLOW,
                     "aggregate overflow typed");
    return 0;
}

static int write_scale_fixture(const char *path, unsigned long long count)
{
    FILE *fp = fopen(path, "wb+");
    unsigned long long i;
    unsigned long long directory_end;
    unsigned long long data_offset;
    char name[64];
    int ok = fp != NULL;
    if (!fp) return 0;
    ok = write_header(fp, count) && write_alignment_metadata(fp, 32u);
    for (i = 0ull; ok && i < count; ++i) {
        layout_tensor_spec tensor;
        (void)snprintf(name, sizeof(name), "scale.tensor_%05llu", i);
        memset(&tensor, 0, sizeof(tensor));
        tensor.name = name;
        tensor.qtype = YVEX_GGUF_QTYPE_F32;
        tensor.rank = 1u;
        tensor.dims[0] = 1ull;
        tensor.offset = i * 32ull;
        ok = write_tensor_record(fp, &tensor);
    }
    directory_end = (unsigned long long)ftello(fp);
    data_offset = align_generic(directory_end, 32u);
    while (ok && (unsigned long long)ftello(fp) < data_offset) {
        unsigned char zero = 0u;
        ok = write_bytes(fp, &zero, 1u);
    }
    ok = ok && fflush(fp) == 0 &&
         ftruncate(fileno(fp), (off_t)(data_offset + count * 32ull)) == 0;
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int test_target_scale_linear_pass(void)
{
    const char *path = LAYOUT_TEST_DIR "/scale-69187.gguf";
    const unsigned long long count = 69187ull;
    yvex_gguf_layout_result result;
    remove(path);
    YVEX_TEST_ASSERT(write_scale_fixture(path, count), "write target-scale sparse layout");
    YVEX_TEST_ASSERT(run_layout(path, &result) == YVEX_OK, "target-scale layout passes");
    YVEX_TEST_ASSERT(result.tensors_validated == count, "target-scale tensor count");
    YVEX_TEST_ASSERT(result.raw_tensor_bytes == count * 4ull, "target-scale raw bytes");
    YVEX_TEST_ASSERT(result.padding_read_calls <= count + 1ull,
                     "target-scale pass remains linear");
    YVEX_TEST_ASSERT(result.tensor_payload_bytes_read == 0ull,
                     "target-scale payload remains unread");
    remove(path);
    return 0;
}

static int test_open_snapshot_drift(void)
{
    const char *path = LAYOUT_TEST_DIR "/drift.gguf";
    const char *old_path = LAYOUT_TEST_DIR "/drift.old";
    const char *replacement = LAYOUT_TEST_DIR "/drift.replacement";
    yvex_artifact_options options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_gguf_layout_result result;
    yvex_error err;
    int rc;

    remove(path); remove(old_path); remove(replacement);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, NULL, 0u, 0ull,
                                          0ull, 0, 0), "write drift source");
    YVEX_TEST_ASSERT(write_layout_fixture(replacement, 32u, NULL, 0u, 0ull,
                                          0ull, 0, 0), "write drift replacement");
    memset(&options, 0, sizeof(options));
    options.path = path;
    options.readonly = 1;
    yvex_error_clear(&err);
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open drift source snapshot");
    YVEX_TEST_ASSERT(rename(path, old_path) == 0 && rename(replacement, path) == 0,
                     "replace path after parse");
    rc = yvex_gguf_layout_validate(artifact, gguf, &result, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                     result.code == YVEX_GGUF_LAYOUT_FILE_IDENTITY_DRIFT,
                     "path replacement fails as identity drift");
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    remove(path); remove(old_path); remove(replacement);
    return 0;
}

static int test_materialization_consumes_layout(void)
{
    const char *path = LAYOUT_TEST_DIR "/materialize-gap.gguf";
    const layout_tensor_spec tensors[] = {
        {"layout.a", YVEX_GGUF_QTYPE_F32, 1u, {1ull}, 0ull},
        {"layout.b", YVEX_GGUF_QTYPE_F32, 1u, {1ull}, 64ull}
    };
    yvex_artifact_options artifact_options;
    yvex_materialize_options materialize_options;
    yvex_backend_memory_stats stats;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *table = NULL;
    yvex_weight_table *weights = NULL;
    yvex_backend *backend = NULL;
    yvex_error err;
    int rc;

    remove(path);
    YVEX_TEST_ASSERT(write_layout_fixture(path, 32u, tensors, 2u, 96ull,
                                          0ull, 0, 0),
                     "write materialization gap fixture");
    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = path;
    artifact_options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &artifact_options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&table, gguf, &err);
    if (rc == YVEX_OK) rc = yvex_backend_open_cpu(&backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open materialization preflight inputs");

    memset(&materialize_options, 0, sizeof(materialize_options));
    materialize_options.backend_name = "cpu";
    rc = yvex_weight_table_materialize(&weights, artifact, gguf, table, backend,
                                       &materialize_options, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK && weights == NULL,
                     "materialization refuses noncanonical layout before output");
    YVEX_TEST_ASSERT(yvex_backend_get_memory_stats(backend, &stats, &err) == YVEX_OK,
                     "read backend stats after refusal");
    YVEX_TEST_ASSERT(stats.allocated_bytes == 0ull,
                     "layout refusal precedes backend payload allocation");

    yvex_weight_table_close(weights);
    yvex_backend_close(backend);
    yvex_tensor_table_close(table);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    remove(path);
    return 0;
}

static int test_digest_is_explicit(void)
{
    const char *path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    yvex_artifact_file_identity identity;
    yvex_artifact_integrity_options options;
    yvex_artifact_integrity_report report;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_artifact_integrity_check_path(path, NULL, &report, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "layout integrity without digest passes");
    YVEX_TEST_ASSERT(report.identity_checked == 0 && report.sha256[0] == '\0',
                     "digest is not computed implicitly");
    YVEX_TEST_ASSERT(strcmp(report.digest_status, "not-requested") == 0,
                     "digest status is not-requested");
    rc = yvex_artifact_identity_read(path, &identity, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "compute explicit expected digest");
    memset(&options, 0, sizeof(options));
    options.expect_sha256 = identity.sha256;
    rc = yvex_artifact_integrity_check_path(path, &options, &report, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "explicit digest validation passes");
    YVEX_TEST_ASSERT(report.identity_checked == 1 &&
                     strcmp(report.digest_status, "pass") == 0,
                     "explicit digest uses opened snapshot");
    YVEX_TEST_ASSERT(report.layout.tensor_payload_bytes_read == 0ull,
                     "layout payload counter remains zero with separate digest");
    return 0;
}

/* Proves a fresh snapshot cannot make same-size mutated bytes match prior admission. */
static int test_admission_identity_binds_bytes(void)
{
    static const unsigned char contents[] = {
        0x59u, 0x56u, 0x45u, 0x58u, 0x2du, 0x61u, 0x64u, 0x6du,
        0x69u, 0x73u, 0x73u, 0x69u, 0x6fu, 0x6eu, 0x2du, 0x76u,
        0x31u
    };
    char root[] = LAYOUT_TEST_DIR "/admission-identity-XXXXXX";
    char path[sizeof(root) + 32u];
    yvex_artifact_options options;
    yvex_artifact_file_identity identity;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_artifact *artifact = NULL;
    yvex_error err;
    FILE *stream = NULL;
    const char *failure = NULL;
    int root_owned = 0;
    int file_owned = 0;
    int written;
    int rc;

    if (!mkdtemp(root)) {
        failure = "create collision-free admission identity root";
        goto cleanup;
    }
    root_owned = 1;
    written = snprintf(path, sizeof(path), "%s/artifact.bin", root);
    if (written < 0 || (size_t)written >= sizeof(path)) {
        failure = "construct owned admission identity path";
        goto cleanup;
    }
    stream = fopen(path, "wb");
    if (!stream) {
        failure = "create owned admission identity fixture";
        goto cleanup;
    }
    file_owned = 1;
    if (!write_bytes(stream, contents, sizeof(contents)) || fflush(stream) != 0) {
        failure = "write exact admission identity fixture";
        goto cleanup;
    }
    if (fclose(stream) != 0) {
        stream = NULL;
        failure = "close exact admission identity fixture";
        goto cleanup;
    }
    stream = NULL;

    yvex_error_clear(&err);
    memset(&identity, 0, sizeof(identity));
    if (yvex_artifact_identity_read(path, &identity, &err) != YVEX_OK) {
        failure = "derive exact admission identity";
        goto cleanup;
    }
    memset(&options, 0, sizeof(options));
    options.path = path;
    options.readonly = 1;
    if (yvex_artifact_open(&artifact, &options, &err) != YVEX_OK) {
        failure = "open exact admission fixture";
        goto cleanup;
    }
    memset(&admission, 0, sizeof(admission));
    admission.file_bytes = identity.file_size;
    admission.complete = 1;
    (void)snprintf(admission.artifact_identity, sizeof(admission.artifact_identity), "%s",
                   identity.sha256);
    if (yvex_artifact_snapshot_get(artifact, &admission.file_snapshot, &err) != YVEX_OK) {
        failure = "capture exact admission snapshot";
        goto cleanup;
    }
    memset(&admission_failure, 0, sizeof(admission_failure));
    rc = yvex_artifact_admission_identity_verify(
        artifact, &admission, NULL, NULL, &admission_failure, &err);
    if (rc != YVEX_OK || !admission.artifact_identity_verified ||
        admission.artifact_bytes_hashed != sizeof(contents)) {
        failure = "exact admission bytes must verify";
        goto cleanup;
    }
    yvex_artifact_close(artifact);
    artifact = NULL;

    stream = fopen(path, "r+b");
    if (!stream || fseeko(stream, 8, SEEK_SET) != 0 || fputc(0x58, stream) == EOF ||
        fflush(stream) != 0 || fsync(fileno(stream)) != 0) {
        failure = "mutate one owned artifact byte without changing size";
        goto cleanup;
    }
    if (fclose(stream) != 0) {
        stream = NULL;
        failure = "close same-size artifact mutation";
        goto cleanup;
    }
    stream = NULL;
    if (yvex_artifact_open(&artifact, &options, &err) != YVEX_OK ||
        yvex_artifact_size(artifact) != identity.file_size ||
        yvex_artifact_snapshot_get(artifact, &admission.file_snapshot, &err) != YVEX_OK) {
        failure = "reopen same-size mutation with a fresh snapshot";
        goto cleanup;
    }
    admission.artifact_identity_verified = 0;
    admission.artifact_bytes_hashed = 0ull;
    memset(&admission_failure, 0, sizeof(admission_failure));
    rc = yvex_artifact_admission_identity_verify(
        artifact, &admission, NULL, NULL, &admission_failure, &err);
    if (rc != YVEX_ERR_FORMAT ||
        admission_failure.code != YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH ||
        strcmp(admission_failure.field, "artifact-byte-identity") != 0 ||
        admission.artifact_identity_verified || admission.artifact_bytes_hashed != 0ull) {
        failure = "fresh snapshot must not admit a same-size byte mutation";
        goto cleanup;
    }

cleanup:
    if (stream && fclose(stream) != 0 && !failure)
        failure = "close admission identity fixture";
    yvex_artifact_close(artifact);
    if (file_owned && unlink(path) != 0 && !failure)
        failure = "remove owned admission identity fixture";
    if (root_owned && rmdir(root) != 0 && !failure)
        failure = "remove owned admission identity root";
    if (failure)
        YVEX_TEST_FAIL(failure);
    return 0;
}

int yvex_test_gguf_layout_integrity(void)
{
    YVEX_TEST_ASSERT(prepare_dir(), "prepare GGUF layout test directory");
    if (test_valid_mixed_layout() != 0) return 1;
    if (test_alignment_matrix() != 0) return 1;
    if (test_offset_failure_matrix() != 0) return 1;
    if (test_padding_truncation_and_tail() != 0) return 1;
    if (test_zero_tensor_file() != 0) return 1;
    if (test_checked_arithmetic() != 0) return 1;
    if (test_target_scale_linear_pass() != 0) return 1;
    if (test_open_snapshot_drift() != 0) return 1;
    if (test_materialization_consumes_layout() != 0) return 1;
    if (test_digest_is_explicit() != 0) return 1;
    if (test_admission_identity_binds_bytes() != 0) return 1;
    return 0;
}
