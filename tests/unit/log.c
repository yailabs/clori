/*
 * Exercises core log level and domain names are stable and that unknown values use the fallback
 * string.
 */
#include <yvex/core.h>

#include "tests/test.h"

int yvex_test_log(void)
{
    YVEX_TEST_ASSERT_STREQ(yvex_log_level_name(YVEX_LOG_ERROR), "error", "error level");
    YVEX_TEST_ASSERT_STREQ(yvex_log_level_name(YVEX_LOG_WARN), "warn", "warn level");
    YVEX_TEST_ASSERT_STREQ(yvex_log_level_name(YVEX_LOG_INFO), "info", "info level");
    YVEX_TEST_ASSERT_STREQ(yvex_log_level_name(YVEX_LOG_DEBUG), "debug", "debug level");
    YVEX_TEST_ASSERT_STREQ(yvex_log_level_name(YVEX_LOG_TRACE), "trace", "trace level");
    YVEX_TEST_ASSERT_STREQ(yvex_log_level_name((yvex_log_level)1234), "unknown", "unknown level");

    YVEX_TEST_ASSERT_STREQ(yvex_log_domain_name(YVEX_LOG_CORE), "core", "core domain");
    YVEX_TEST_ASSERT_STREQ(yvex_log_domain_name(YVEX_LOG_CLI), "cli", "cli domain");
    YVEX_TEST_ASSERT_STREQ(yvex_log_domain_name((yvex_log_domain)1234), "unknown", "unknown domain");
    return 0;
}
