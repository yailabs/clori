/* Registry-generated CUDA runner; config/qa/registry.json owns membership. */

#include "tests/test.h"
#include "tests/support/runner.h"

#include "qa/cuda_registry.inc"

int main(void)
{
    return yvex_test_runner_run(yvex_cuda_tests, yvex_cuda_test_count,
                                "YVEX_CUDA_TEST_FILTER", "cuda test", 0);
}
