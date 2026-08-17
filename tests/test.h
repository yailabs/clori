/*
 * Provides tiny assertion helpers for core C tests. This is intentionally small and
 * dependency-free.
 */
#ifndef YVEX_TEST_H
#define YVEX_TEST_H

#include <stdio.h>
#include <string.h>

#include "qa/test_declarations.h"

/* Cross-owner fixture helper; it is not a runner registration. */
struct yvex_artifact_lowering_map;
int yvex_test_deepseek_map_fixture_build(struct yvex_artifact_lowering_map **out);

#define YVEX_TEST_FAIL(msg) \
    do { \
        fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
        return 1; \
    } while (0)

#define YVEX_TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            YVEX_TEST_FAIL(msg); \
        } \
    } while (0)

#define YVEX_TEST_ASSERT_STREQ(actual, expected, msg) \
    do { \
        const char *yvex_test_actual = (actual); \
        const char *yvex_test_expected = (expected); \
        if (!yvex_test_actual || !yvex_test_expected || strcmp(yvex_test_actual, yvex_test_expected) != 0) { \
            fprintf(stderr, "FAIL: %s:%d: %s: expected '%s', got '%s'\n", \
                    __FILE__, __LINE__, (msg), \
                    yvex_test_expected ? yvex_test_expected : "(null)", \
                    yvex_test_actual ? yvex_test_actual : "(null)"); \
            return 1; \
        } \
    } while (0)

#endif /* YVEX_TEST_H */
