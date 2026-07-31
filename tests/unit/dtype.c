/*
 * Exercises model layer dtype/qtype name mapping, GGML raw type mapping, and storage projection
 * to the canonical row-aware GGUF storage ABI.
 */
#include <limits.h>

#include <yvex/api.h>

#include "tests/test.h"

static int test_names_and_mapping(void)
{
    const yvex_dtype_info *info;

    YVEX_TEST_ASSERT_STREQ(yvex_dtype_name(YVEX_DTYPE_F32), "F32", "F32 name");
    YVEX_TEST_ASSERT_STREQ(yvex_dtype_name(YVEX_DTYPE_Q4_0), "Q4_0", "Q4_0 name");
    YVEX_TEST_ASSERT_STREQ(yvex_dtype_name((yvex_dtype)999), "UNKNOWN", "unknown dtype name");

    info = yvex_dtype_get_info(YVEX_DTYPE_F32);
    YVEX_TEST_ASSERT(info != NULL, "F32 info exists");
    YVEX_TEST_ASSERT(info->ggml_type == YVEX_GGUF_QTYPE_F32, "F32 raw type");
    YVEX_TEST_ASSERT(yvex_dtype_storage_supported(YVEX_DTYPE_F32) == 1,
                     "F32 accounting supported");
    YVEX_TEST_ASSERT(yvex_dtype_is_quantized(YVEX_DTYPE_F32) == 0,
                     "F32 is scalar");
    YVEX_TEST_ASSERT(yvex_dtype_is_quantized(YVEX_DTYPE_Q4_0) == 1,
                     "Q4_0 is block quantized");

    info = yvex_dtype_from_ggml_type(8);
    YVEX_TEST_ASSERT(info != NULL, "Q8_0 info exists");
    YVEX_TEST_ASSERT(info->dtype == YVEX_DTYPE_Q8_0, "Q8_0 raw type mapping");

    info = yvex_dtype_from_ggml_type(9999);
    YVEX_TEST_ASSERT(info != NULL, "unknown raw type info exists");
    YVEX_TEST_ASSERT(info->dtype == YVEX_DTYPE_UNKNOWN, "unknown raw type maps unknown");

    return 0;
}

static int expect_bytes(yvex_dtype dtype, unsigned long long elements, unsigned long long expected)
{
    yvex_error err;
    unsigned long long actual = 0;
    int rc;

    rc = yvex_dtype_storage_bytes(dtype, elements, &actual, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "storage failed for %s: %s: %s\n",
                yvex_dtype_name(dtype), yvex_error_where(&err), yvex_error_message(&err));
        return 1;
    }
    if (actual != expected) {
        fprintf(stderr, "storage mismatch for %s: expected %llu got %llu\n",
                yvex_dtype_name(dtype), expected, actual);
        return 1;
    }
    return 0;
}

static int expect_tensor_bytes(yvex_dtype dtype,
                               const unsigned long long *dims,
                               unsigned int rank,
                               unsigned long long expected)
{
    yvex_error err;
    unsigned long long actual = 0ull;
    int rc = yvex_dtype_tensor_storage_bytes(dtype, dims, rank, &actual, &err);

    if (rc != YVEX_OK) return 0;
    return actual == expected;
}

static int test_storage_formulas(void)
{
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_F32, 32, 128) == 0, "F32 bytes");
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_F16, 32, 64) == 0, "F16 bytes");
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_BF16, 32, 64) == 0, "BF16 bytes");
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_F64, 4, 32) == 0, "F64 bytes");
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_I8, 9, 9) == 0, "I8 bytes");
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_I16, 9, 18) == 0, "I16 bytes");
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_I32, 9, 36) == 0, "I32 bytes");
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_I64, 9, 72) == 0, "I64 bytes");
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_Q4_0, 32, 18) == 0, "Q4_0 exact row");
    YVEX_TEST_ASSERT(expect_bytes(YVEX_DTYPE_Q8_0, 32, 34) == 0, "Q8_0 one block");
    {
        unsigned long long q4_rows[] = {32ull, 2ull};
        unsigned long long q4_bad_rows[] = {16ull, 2ull};
        yvex_error err;
        unsigned long long bytes = 0ull;
        int rc;

        YVEX_TEST_ASSERT(expect_tensor_bytes(YVEX_DTYPE_Q4_0,
                                             q4_rows, 2u, 36ull),
                         "Q4_0 dtype projection preserves rows");
        rc = yvex_dtype_tensor_storage_bytes(YVEX_DTYPE_Q4_0,
                                             q4_bad_rows, 2u, &bytes, &err);
        YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT,
                         "Q4_0 dtype projection refuses partial rows");
    }
    return 0;
}

static int test_error_cases(void)
{
    yvex_error err;
    unsigned long long bytes = 123;
    int rc;

    rc = yvex_dtype_storage_bytes(YVEX_DTYPE_Q4_K, 256, &bytes, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && bytes == 144ull,
                     "K-family formula comes from canonical registry");

    rc = yvex_dtype_storage_bytes(YVEX_DTYPE_Q4_0, 33, &bytes, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT,
                     "legacy block query refuses partial row");

    rc = yvex_dtype_storage_bytes(YVEX_DTYPE_F32, (ULLONG_MAX / 4u) + 1u, &bytes, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "dense byte overflow rejected");

    rc = yvex_dtype_storage_bytes(YVEX_DTYPE_F32, 1, NULL, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG, "null output rejected");

    return 0;
}

int yvex_test_dtype(void)
{
    if (test_names_and_mapping() != 0) {
        return 1;
    }
    if (test_storage_formulas() != 0) {
        return 1;
    }
    if (test_error_cases() != 0) {
        return 1;
    }
    return 0;
}
