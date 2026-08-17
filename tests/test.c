/* Registry-generated unit runner; config/qa/registry.json owns membership. */

#include "tests/test.h"
#include "tests/support/runner.h"

#include "qa/unit_registry.inc"

int main(void)
{
    return yvex_test_runner_run(yvex_unit_tests, yvex_unit_test_count, "YVEX_TEST_FILTER",
                                "test", 1);
}
