/*
 * It proves storage facts only; quantization, dequantization, emission, compute, and runtime
 * remain separate.
 */
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <yvex/qtype.h>

#include "tests/test.h"
#include <yvex/internal/gguf.h>

typedef struct {
    const char *name;
    yvex_gguf_qtype_identity_status identity;
    unsigned int block_size;
    unsigned int block_bytes;
} pinned_qtype;

static const pinned_qtype pinned[] = {
    {"F32", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 1u, 4u},
    {"F16", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 1u, 2u},
    {"Q4_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 32u, 18u},
    {"Q4_1", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 32u, 20u},
    {"Q4_2", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, 0u, 0u},
    {"Q4_3", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, 0u, 0u},
    {"Q5_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 32u, 22u},
    {"Q5_1", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 32u, 24u},
    {"Q8_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 32u, 34u},
    {"Q8_1", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 32u, 36u},
    {"Q2_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 84u},
    {"Q3_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 110u},
    {"Q4_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 144u},
    {"Q5_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 176u},
    {"Q6_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 210u},
    {"Q8_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 292u},
    {"IQ2_XXS", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 66u},
    {"IQ2_XS", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 74u},
    {"IQ3_XXS", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 98u},
    {"IQ1_S", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 50u},
    {"IQ4_NL", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 32u, 18u},
    {"IQ3_S", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 110u},
    {"IQ2_S", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 82u},
    {"IQ4_XS", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 136u},
    {"I8", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 1u, 1u},
    {"I16", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 1u, 2u},
    {"I32", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 1u, 4u},
    {"I64", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 1u, 8u},
    {"F64", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 1u, 8u},
    {"IQ1_M", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 56u},
    {"BF16", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 1u, 2u},
    {"Q4_0_4_4", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, 0u, 0u},
    {"Q4_0_4_8", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, 0u, 0u},
    {"Q4_0_8_8", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, 0u, 0u},
    {"TQ1_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 54u},
    {"TQ2_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 256u, 66u},
    {"IQ4_NL_4_4", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, 0u, 0u},
    {"IQ4_NL_4_8", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, 0u, 0u},
    {"IQ4_NL_8_8", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, 0u, 0u},
    {"MXFP4", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, 32u, 17u},
    {"NVFP4", YVEX_GGUF_QTYPE_IDENTITY_OUTSIDE_BASELINE, 64u, 36u},
    {"Q1_0", YVEX_GGUF_QTYPE_IDENTITY_OUTSIDE_BASELINE, 128u, 18u},
    {"Q2_0", YVEX_GGUF_QTYPE_IDENTITY_OUTSIDE_BASELINE, 64u, 18u}
};

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

    for (i = 0u; i < 8u; ++i) {
        buf[(*len)++] = (unsigned char)((value >> (8u * i)) & 0xffu);
    }
}

static void append_string(unsigned char *buf, size_t *len, const char *text)
{
    size_t text_len = strlen(text);

    append_u64(buf, len, text_len);
    memcpy(buf + *len, text, text_len);
    *len += text_len;
}

static int write_q4_fixture(const char *path, unsigned long long row_width)
{
    unsigned char buf[512];
    size_t len = 0u;
    FILE *fp;

    memcpy(buf + len, "GGUF", 4u);
    len += 4u;
    append_u32(buf, &len, 3u);
    append_u64(buf, &len, 1ull);
    append_u64(buf, &len, 1ull);
    append_string(buf, &len, "general.architecture");
    append_u32(buf, &len, 8u);
    append_string(buf, &len, "deepseek");
    append_string(buf, &len, "q4.weight");
    append_u32(buf, &len, 2u);
    append_u64(buf, &len, row_width);
    append_u64(buf, &len, 2ull);
    append_u32(buf, &len, YVEX_GGUF_QTYPE_Q4_0);
    append_u64(buf, &len, 0ull);
    while ((len % 32u) != 0u) buf[len++] = 0u;
    memset(buf + len, 0, 36u);
    len += 36u;
    while ((len % 32u) != 0u) buf[len++] = 0u;

    fp = fopen(path, "wb");
    if (!fp) return 0;
    if (fwrite(buf, 1u, len, fp) != len) {
        fclose(fp);
        return 0;
    }
    return fclose(fp) == 0;
}

static int inspect_fixture(const char *path, yvex_gguf_layout_result *layout,
                           yvex_gguf_parse_result *parse)
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
    if (rc == YVEX_OK)
        rc = yvex_gguf_open_ex(&gguf, artifact, NULL, parse, &err);
    if (rc == YVEX_OK && layout)
        rc = yvex_gguf_layout_validate(artifact, gguf, layout, &err);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc;
}

static int test_pinned_registry_parity(void)
{
    size_t i;

    YVEX_TEST_ASSERT_STREQ(YVEX_GGUF_QTYPE_ABI_UPSTREAM_COMMIT,
                           "af97976c7810cdabb1863172f31c432dab767de7",
                           "pinned ggml revision");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_geometry_count() ==
                     sizeof(pinned) / sizeof(pinned[0]),
                     "pinned qtype row count");
    for (i = 0u; i < sizeof(pinned) / sizeof(pinned[0]); ++i) {
        const yvex_gguf_qtype_geometry *geometry =
            yvex_gguf_qtype_geometry_at(i);
        YVEX_TEST_ASSERT(geometry != NULL, "pinned qtype row exists");
        YVEX_TEST_ASSERT(geometry->numeric_capability_reserved == 0u,
                         "storage geometry must not own numeric capability");
        YVEX_TEST_ASSERT(geometry->qtype == i, "pinned qtype numeric identity");
        YVEX_TEST_ASSERT_STREQ(geometry->name, pinned[i].name,
                               "pinned qtype canonical name");
        YVEX_TEST_ASSERT(geometry->identity_status == pinned[i].identity,
                         "pinned qtype identity status");
        YVEX_TEST_ASSERT(geometry->block_size == pinned[i].block_size,
                         "pinned qtype block width");
        YVEX_TEST_ASSERT(geometry->bytes_per_block == pinned[i].block_bytes,
                         "pinned qtype block bytes");
    }
    return 0;
}

static int test_identity_admission(void)
{
    const yvex_quant_numeric_capability *q2_k;
    const yvex_quant_numeric_capability *q4_0;
    const yvex_quant_numeric_capability *q8_0;
    yvex_gguf_qtype_storage_result result;
    unsigned long long dims[] = {32ull};

    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_Q4_2_REMOVED, dims, 1u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_REMOVED_ID,
                     "removed qtype refused distinctly");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_NVFP4_OUTSIDE_BASELINE, dims, 1u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_OUTSIDE_BASELINE,
                     "outside-baseline qtype refused distinctly");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(9999u, dims, 1u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_UNKNOWN_ID,
                     "unknown qtype refused distinctly");
    YVEX_TEST_ASSERT_STREQ(yvex_gguf_qtype_identity_status_name(
                               YVEX_GGUF_QTYPE_IDENTITY_RESERVED),
                           "reserved", "reserved identity has stable state");
    q4_0 = yvex_quant_numeric_capability_at(YVEX_GGUF_QTYPE_Q4_0);
    q8_0 = yvex_quant_numeric_capability_at(YVEX_GGUF_QTYPE_Q8_0);
    q2_k = yvex_quant_numeric_capability_at(YVEX_GGUF_QTYPE_Q2_K);
    YVEX_TEST_ASSERT(q4_0 && q4_0->reference_decoder_available == 0,
                     "known geometry does not imply reference decoder");
    YVEX_TEST_ASSERT(q8_0 && q8_0->reference_decoder_available == 1 && q2_k &&
                         q2_k->reference_decoder_available == 1,
                     "implemented codecs project canonical reference decoders");
    return 0;
}

static int expect_tensor_bytes(unsigned int qtype,
                               const unsigned long long *dims,
                               unsigned int rank,
                               unsigned long long expected)
{
    yvex_gguf_qtype_storage_result result;

    if (yvex_gguf_qtype_tensor_storage(qtype, dims, rank, &result) !=
        YVEX_GGUF_QTYPE_STORAGE_OK) return 0;
    return result.total_bytes == expected;
}

static int test_row_aware_geometry(void)
{
    yvex_gguf_qtype_storage_result result;
    unsigned long long f32[] = {32ull};
    unsigned long long q4_two_rows[] = {32ull, 2ull};
    unsigned long long q4_flat_only[] = {16ull, 2ull};
    unsigned long long q4_partial[] = {33ull};
    unsigned long long q4k[] = {256ull, 2ull};
    unsigned long long iq2[] = {256ull};
    unsigned long long mxfp4[] = {32ull};
    unsigned long long zero[] = {0ull};
    unsigned long long too_many[] = {1ull, 1ull, 1ull, 1ull, 1ull};

    YVEX_TEST_ASSERT(expect_tensor_bytes(YVEX_GGUF_QTYPE_F32, f32, 1u, 128ull),
                     "scalar row bytes");
    YVEX_TEST_ASSERT(expect_tensor_bytes(YVEX_GGUF_QTYPE_Q4_0,
                                         q4_two_rows, 2u, 36ull),
                     "Q4_0 two complete rows");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_Q4_0, q4_flat_only, 2u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_ROW_BLOCK_MISMATCH,
                     "flattened alignment cannot admit partial rows");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_Q4_0, q4_partial, 1u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_ROW_BLOCK_MISMATCH,
                     "partial row is not padded");
    YVEX_TEST_ASSERT(expect_tensor_bytes(YVEX_GGUF_QTYPE_Q4_K, q4k, 2u, 288ull),
                     "K-family row geometry");
    YVEX_TEST_ASSERT(expect_tensor_bytes(YVEX_GGUF_QTYPE_IQ2_XXS, iq2, 1u, 66ull),
                     "IQ-family row geometry");
    YVEX_TEST_ASSERT(expect_tensor_bytes(YVEX_GGUF_QTYPE_TQ1_0, iq2, 1u, 54ull),
                     "TQ-family row geometry");
    YVEX_TEST_ASSERT(expect_tensor_bytes(YVEX_GGUF_QTYPE_MXFP4, mxfp4, 1u, 17ull),
                     "MXFP4 row geometry");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_F32, zero, 1u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_INVALID_DIMENSION,
                     "zero dimension refused");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_F32, f32, 0u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_INVALID_RANK,
                     "zero rank refused");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_F32, too_many, 5u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_INVALID_RANK,
                     "excess rank refused");
    return 0;
}

static int test_overflow_statuses(void)
{
    yvex_gguf_qtype_storage_result result;
    unsigned long long aligned_max = ULLONG_MAX - (ULLONG_MAX % 32ull);
    unsigned long long row_bytes[] = {0ull};
    unsigned long long row_count[] = {32ull, ULLONG_MAX, 2ull};
    unsigned long long elements[] = {0ull, 2ull};
    unsigned long long total[] = {32ull, ULLONG_MAX / 32ull};

    row_bytes[0] = aligned_max;
    elements[0] = aligned_max;
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_Q8_1, row_bytes, 1u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_ROW_BYTE_OVERFLOW,
                     "row byte overflow is typed");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_Q4_0, row_count, 3u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_ROW_COUNT_OVERFLOW,
                     "row count overflow is typed");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_Q4_0, elements, 2u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_ELEMENT_COUNT_OVERFLOW,
                     "element count overflow is typed");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(
                         YVEX_GGUF_QTYPE_Q8_1, total, 2u, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_TOTAL_BYTE_OVERFLOW,
                     "total byte overflow is typed");
    return 0;
}

static int test_legacy_row_and_range_validation(void)
{
    yvex_gguf_qtype_storage_result result;
    const char *reason = NULL;
    unsigned long long dims[] = {32ull, 2ull};
    unsigned long long bytes = 0ull;

    YVEX_TEST_ASSERT(yvex_gguf_qtype_storage_bytes(
                         YVEX_GGUF_QTYPE_Q4_0, 32ull, &bytes, &reason) == 1,
                     "single-row compatibility calculation succeeds");
    YVEX_TEST_ASSERT(bytes == 18ull, "single-row compatibility bytes");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_storage_bytes(
                         YVEX_GGUF_QTYPE_Q4_0, 33ull, &bytes, &reason) == 0,
                     "single-row compatibility refuses partial block");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_validate_tensor_storage(
                         YVEX_GGUF_QTYPE_Q4_0, dims, 2u, 35ull, &result) ==
                     YVEX_GGUF_QTYPE_STORAGE_EXPECTED_ACTUAL_MISMATCH,
                     "actual span mismatch is typed");
    YVEX_TEST_ASSERT(result.total_bytes == 36ull,
                     "range mismatch preserves expected bytes");
    return 0;
}

static int test_reader_integration(void)
{
    const yvex_quant_numeric_capability *capability;
    const yvex_gguf_qtype_geometry *geometry;
    yvex_gguf_qtype_storage_result storage;
    yvex_gguf_layout_result layout;
    yvex_gguf_parse_result parse;
    unsigned long long dims[] = {32ull, 2ull};

    memset(&layout, 0, sizeof(layout));
    memset(&parse, 0, sizeof(parse));
    YVEX_TEST_ASSERT(inspect_fixture(
                         "tests/fixtures/gguf/valid-metadata-tensors.gguf",
                         &layout, &parse) == YVEX_OK,
                     "canonical reader and layout accept fixture");
    YVEX_TEST_ASSERT(layout.tensors_validated == 1ull,
                     "layout checks every qtype tensor");
    YVEX_TEST_ASSERT(layout.raw_tensor_bytes == 128ull,
                     "layout owns aggregate qtype bytes");

    geometry = yvex_gguf_qtype_geometry_find(YVEX_GGUF_QTYPE_Q4_0);
    capability = yvex_quant_numeric_capability_at(YVEX_GGUF_QTYPE_Q4_0);
    YVEX_TEST_ASSERT(geometry != NULL, "report geometry exists");
    YVEX_TEST_ASSERT_STREQ(yvex_gguf_qtype_identity_status_name(geometry->identity_status),
                           "admitted", "report identity");
    YVEX_TEST_ASSERT(yvex_gguf_qtype_tensor_storage(geometry->qtype, dims, 2u, &storage) ==
                         YVEX_GGUF_QTYPE_STORAGE_OK,
                     "report storage status");
    YVEX_TEST_ASSERT(capability && capability->reference_decoder_available == 0,
                     "report decoder boundary");
    YVEX_TEST_ASSERT(storage.total_bytes == 36ull,
                     "report uses row-aware bytes");
    return 0;
}

static int test_reader_and_layout_preserve_rows(void)
{
    const char *valid_path = "build/tests/gguf-qtype/q4-valid.gguf";
    const char *invalid_path = "build/tests/gguf-qtype/q4-invalid.gguf";
    yvex_gguf_layout_result layout;
    yvex_gguf_parse_result parse;

    (void)mkdir("build", 0777);
    (void)mkdir("build/tests", 0777);
    (void)mkdir("build/tests/gguf-qtype", 0777);
    YVEX_TEST_ASSERT(write_q4_fixture(valid_path, 32ull),
                     "write row-aware Q4_0 fixture");
    YVEX_TEST_ASSERT(write_q4_fixture(invalid_path, 16ull),
                     "write partial-row Q4_0 fixture");

    memset(&layout, 0, sizeof(layout));
    memset(&parse, 0, sizeof(parse));
    YVEX_TEST_ASSERT(inspect_fixture(valid_path, &layout, &parse) == YVEX_OK,
                     "reader accepts two complete Q4_0 rows");
    YVEX_TEST_ASSERT(layout.raw_tensor_bytes == 36ull,
                     "layout aggregate uses row geometry");

    memset(&parse, 0, sizeof(parse));
    YVEX_TEST_ASSERT(inspect_fixture(invalid_path, NULL, &parse) != YVEX_OK,
                     "reader propagates qtype geometry refusal");
    YVEX_TEST_ASSERT(parse.code == YVEX_GGUF_PARSE_REFUSED_QTYPE &&
                         parse.section == YVEX_GGUF_PARSE_SECTION_QTYPE,
                     "partial Q4_0 row has typed qtype refusal");
    return 0;
}

int yvex_test_gguf_qtype_abi(void)
{
    if (test_pinned_registry_parity() != 0) return 1;
    if (test_identity_admission() != 0) return 1;
    if (test_row_aware_geometry() != 0) return 1;
    if (test_overflow_statuses() != 0) return 1;
    if (test_legacy_row_and_range_validation() != 0) return 1;
    if (test_reader_integration() != 0) return 1;
    if (test_reader_and_layout_preserve_rows() != 0) return 1;
    return 0;
}
