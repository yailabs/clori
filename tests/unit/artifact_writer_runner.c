/* Registry-generated artifact sanitizer runner. */

#include "tests/test.h"
#include "tests/support/runner.h"

#include "qa/artifact_registry.inc"

int main(void)
{
    return yvex_test_runner_run(yvex_artifact_tests, yvex_artifact_test_count,
                                "YVEX_ARTIFACT_TEST_FILTER", "artifact test", 1);
}
