/*
 * Exercises fixed-size YVEX error objects are safe to clear, set, truncate, inspect, and use
 * with null inputs.
 */
#include <string.h>

#include <yvex/core.h>

#include "tests/test.h"

int yvex_test_error(void)
{
    yvex_error err;
    char long_message[400];
    unsigned long i;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(err.code == YVEX_OK, "clear code");
    YVEX_TEST_ASSERT_STREQ(err.where, "", "clear where");
    YVEX_TEST_ASSERT_STREQ(err.message, "", "clear message");
    YVEX_TEST_ASSERT(!yvex_error_is_set(&err), "clear is not set");
    YVEX_TEST_ASSERT(yvex_error_code(&err) == YVEX_OK, "clear code accessor");
    YVEX_TEST_ASSERT_STREQ(yvex_error_where(&err), "", "clear where accessor");
    YVEX_TEST_ASSERT_STREQ(yvex_error_message(&err), "", "clear message accessor");

    yvex_error_set(&err, YVEX_ERR_INVALID_ARG, "test_error", "bad argument");
    YVEX_TEST_ASSERT(err.code == YVEX_ERR_INVALID_ARG, "set code");
    YVEX_TEST_ASSERT_STREQ(err.where, "test_error", "set where");
    YVEX_TEST_ASSERT_STREQ(err.message, "bad argument", "set message");
    YVEX_TEST_ASSERT(yvex_error_is_set(&err), "set is set");
    YVEX_TEST_ASSERT(yvex_error_code(&err) == YVEX_ERR_INVALID_ARG, "set code accessor");
    YVEX_TEST_ASSERT_STREQ(yvex_error_where(&err), "test_error", "set where accessor");
    YVEX_TEST_ASSERT_STREQ(yvex_error_message(&err), "bad argument", "set message accessor");

    yvex_error_set(&err, YVEX_ERR, NULL, NULL);
    YVEX_TEST_ASSERT_STREQ(err.where, "unknown", "null where fallback");
    YVEX_TEST_ASSERT_STREQ(err.message, "", "null message fallback");

    yvex_error_setf(&err, YVEX_ERR_BOUNDS, "bounds", "index %d", 7);
    YVEX_TEST_ASSERT(err.code == YVEX_ERR_BOUNDS, "setf code");
    YVEX_TEST_ASSERT_STREQ(err.where, "bounds", "setf where");
    YVEX_TEST_ASSERT_STREQ(err.message, "index 7", "setf message");

    for (i = 0; i < sizeof(long_message) - 1; ++i) {
        long_message[i] = 'x';
    }
    long_message[sizeof(long_message) - 1] = '\0';

    yvex_error_set(&err, YVEX_ERR, long_message, long_message);
    YVEX_TEST_ASSERT(err.where[YVEX_ERROR_WHERE_CAP - 1] == '\0', "where null termination");
    YVEX_TEST_ASSERT(err.message[YVEX_ERROR_MESSAGE_CAP - 1] == '\0', "message null termination");
    YVEX_TEST_ASSERT(strlen(err.where) == YVEX_ERROR_WHERE_CAP - 1, "where truncation");
    YVEX_TEST_ASSERT(strlen(err.message) == YVEX_ERROR_MESSAGE_CAP - 1, "message truncation");

    yvex_error_clear(NULL);
    yvex_error_set(NULL, YVEX_ERR, "ignored", "ignored");
    yvex_error_setf(NULL, YVEX_ERR, "ignored", "ignored");
    YVEX_TEST_ASSERT(!yvex_error_is_set(NULL), "null error is not set");
    YVEX_TEST_ASSERT(yvex_error_code(NULL) == YVEX_OK, "null error code");
    YVEX_TEST_ASSERT_STREQ(yvex_error_where(NULL), "", "null error where");
    YVEX_TEST_ASSERT_STREQ(yvex_error_message(NULL), "", "null error message");
    return 0;
}
