/* Registry-generated quant sanitizer runner. */

#include "tests/test.h"
#include "tests/support/runner.h"

#include "qa/quant_registry.inc"

int main(void)
{
    return yvex_test_runner_run(yvex_quant_tests, yvex_quant_test_count,
                                "YVEX_QUANT_TEST_FILTER", "quant test", 1);
}
