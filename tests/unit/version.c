/* Exercises the core version API returns stable compile-time values without runtime setup. */
#include <yvex/core.h>

#include "tests/test.h"

int yvex_test_version(void)
{
    YVEX_TEST_ASSERT_STREQ(yvex_version_string(), "0.1.0", "version string");
    YVEX_TEST_ASSERT(yvex_version_major() == 0, "version major");
    YVEX_TEST_ASSERT(yvex_version_minor() == 1, "version minor");
    YVEX_TEST_ASSERT(yvex_version_patch() == 0, "version patch");
    return 0;
}
