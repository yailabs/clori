/*
 * YVEX - Status tests
 *
 * File: tests/test_status.c
 * Layer: test
 *
 * Purpose:
 *   Proves that every core status code maps to a stable string and that
 *   status predicates behave deterministically.
 *
 * Covers:
 *   - yvex_status_name
 *   - yvex_status_is_ok
 *   - yvex_status_is_error
 *
 * Commands:
 *   - make test-core
 *   - build/tests/test_status
 *
 * Expected:
 *   - exits 0 on success
 *   - prints concise failure to stderr
 */
#include <yvex/core.h>
#include <yvex/internal/core.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "tests/test.h"

int yvex_test_status(void)
{
    unsigned long long arithmetic = 17ull;
    const unsigned char hash_bytes[8] = {0x08u, 0x07u, 0x06u, 0x05u,
                                         0x04u, 0x03u, 0x02u, 0x01u};
    const unsigned long long hash_seed = 1469598103934665603ull;
    char *copy;
    yvex_sha256 canonical_hash;
    yvex_sha256 explicit_hash;
    unsigned char canonical_digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned char explicit_digest[YVEX_SHA256_DIGEST_BYTES];

    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_OK), "YVEX_OK", "YVEX_OK name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR), "YVEX_ERR", "YVEX_ERR name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_NOMEM), "YVEX_ERR_NOMEM", "YVEX_ERR_NOMEM name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_IO), "YVEX_ERR_IO", "YVEX_ERR_IO name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_FORMAT), "YVEX_ERR_FORMAT", "YVEX_ERR_FORMAT name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_UNSUPPORTED), "YVEX_ERR_UNSUPPORTED", "YVEX_ERR_UNSUPPORTED name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_BACKEND), "YVEX_ERR_BACKEND", "YVEX_ERR_BACKEND name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_BOUNDS), "YVEX_ERR_BOUNDS", "YVEX_ERR_BOUNDS name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_STATE), "YVEX_ERR_STATE", "YVEX_ERR_STATE name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_CANCELLED), "YVEX_ERR_CANCELLED", "YVEX_ERR_CANCELLED name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_INVALID_ARG), "YVEX_ERR_INVALID_ARG", "YVEX_ERR_INVALID_ARG name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name(YVEX_ERR_TIMEOUT), "YVEX_ERR_TIMEOUT", "YVEX_ERR_TIMEOUT name");
    YVEX_TEST_ASSERT_STREQ(yvex_status_name((yvex_status)1234), "YVEX_STATUS_UNKNOWN", "unknown status name");

    YVEX_TEST_ASSERT(yvex_status_is_ok(YVEX_OK), "YVEX_OK is ok");
    YVEX_TEST_ASSERT(!yvex_status_is_ok(YVEX_ERR), "YVEX_ERR is not ok");
    YVEX_TEST_ASSERT(!yvex_status_is_ok((yvex_status)1234), "unknown positive status is not ok");
    YVEX_TEST_ASSERT(!yvex_status_is_error(YVEX_OK), "YVEX_OK is not error");
    YVEX_TEST_ASSERT(yvex_status_is_error(YVEX_ERR_INVALID_ARG), "YVEX_ERR_INVALID_ARG is error");
    YVEX_TEST_ASSERT(yvex_status_is_error((yvex_status)1234), "unknown positive status is error");

    YVEX_TEST_ASSERT(yvex_core_u64_add(1ull, 2ull, &arithmetic) && arithmetic == 3ull,
                     "checked u64 addition admits an exact sum");
    arithmetic = 17ull;
    YVEX_TEST_ASSERT(!yvex_core_u64_add(ULLONG_MAX, 1ull, &arithmetic) &&
                         arithmetic == 17ull,
                     "checked u64 addition refuses overflow without output mutation");
    YVEX_TEST_ASSERT(!yvex_core_u64_add(1ull, 2ull, NULL),
                     "checked u64 addition refuses a missing output");
    YVEX_TEST_ASSERT(yvex_core_u64_mul(7ull, 6ull, &arithmetic) && arithmetic == 42ull,
                     "checked u64 multiplication admits an exact product");
    arithmetic = 23ull;
    YVEX_TEST_ASSERT(!yvex_core_u64_mul(ULLONG_MAX, 2ull, &arithmetic) &&
                         arithmetic == 23ull,
                     "checked u64 multiplication refuses overflow without output mutation");
    YVEX_TEST_ASSERT(yvex_core_u64_mul(0ull, ULLONG_MAX, &arithmetic) && arithmetic == 0ull,
                     "checked u64 multiplication admits the zero product");
    YVEX_TEST_ASSERT(!yvex_core_u64_mul(1ull, 2ull, NULL),
                     "checked u64 multiplication refuses a missing output");
    YVEX_TEST_ASSERT(
        yvex_core_hash_mix_bytes(hash_seed, hash_bytes, sizeof(hash_bytes)) ==
            yvex_core_hash_mix_u64(hash_seed, 0x0102030405060708ull),
        "u64 hashing uses the canonical little-endian byte encoding");
    YVEX_TEST_ASSERT(yvex_core_hash_mix_bytes(hash_seed, NULL, 0u) == hash_seed,
                     "empty byte hashing preserves the input hash");
    YVEX_TEST_ASSERT(yvex_core_hash_mix_bytes(hash_seed, NULL, 1u) == hash_seed,
                     "invalid byte hashing preserves the input hash");
    copy = yvex_core_strdup("canonical");
    YVEX_TEST_ASSERT(copy && strcmp(copy, "canonical") == 0,
                     "core string duplication returns an exact owned copy");
    free(copy);
    copy = yvex_core_strdup(NULL);
    YVEX_TEST_ASSERT(copy && copy[0] == '\0',
                     "core string duplication canonicalizes null as empty text");
    free(copy);
    yvex_sha256_init(&canonical_hash);
    yvex_sha256_init(&explicit_hash);
    YVEX_TEST_ASSERT(yvex_sha256_update_u64(&canonical_hash,
                                             0x0102030405060708ull) &&
                         yvex_sha256_update(&explicit_hash, hash_bytes,
                                            sizeof(hash_bytes)) &&
                         yvex_sha256_final(&canonical_hash, canonical_digest) &&
                         yvex_sha256_final(&explicit_hash, explicit_digest) &&
                         memcmp(canonical_digest, explicit_digest,
                                sizeof(canonical_digest)) == 0,
                     "SHA-256 scalar encoding is canonical little endian");
    yvex_sha256_init(&canonical_hash);
    yvex_sha256_init(&explicit_hash);
    YVEX_TEST_ASSERT(yvex_sha256_update_text(&canonical_hash, NULL) &&
                         yvex_sha256_update_u64(&explicit_hash, 0ull) &&
                         yvex_sha256_final(&canonical_hash, canonical_digest) &&
                         yvex_sha256_final(&explicit_hash, explicit_digest) &&
                         memcmp(canonical_digest, explicit_digest,
                                sizeof(canonical_digest)) == 0,
                     "SHA-256 text encoding canonicalizes null as empty text");
    return 0;
}
