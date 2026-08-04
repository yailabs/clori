/* This runner groups CUDA probe, tensor, kernel parity, and materialization coverage. */

#include "tests/test.h"

#include <stdlib.h>

static int run_cuda_test(const char *name, int (*fn)(void))
{
    int rc;

    fprintf(stderr, "cuda test: %s\n", name);
    rc = fn();
    if (rc != 0) {
        fprintf(stderr, "FAIL: %s exited %d\n", name, rc);
    }
    return rc;
}

static int cuda_test_selected(const char *filter, const char *name)
{
    return !filter || filter[0] == '\0' || strcmp(filter, name) == 0;
}

int main(void)
{
    const char *filter = getenv("YVEX_CUDA_TEST_FILTER");

    if (cuda_test_selected(filter, "info") &&
        run_cuda_test("info", yvex_cuda_test_info) != 0) return 1;
    if (cuda_test_selected(filter, "graph") &&
        run_cuda_test("graph", yvex_cuda_test_graph) != 0) return 1;
    if (cuda_test_selected(filter, "tensor") &&
        run_cuda_test("tensor", yvex_cuda_test_tensor) != 0) return 1;
    if (cuda_test_selected(filter, "ops") &&
        run_cuda_test("ops", yvex_cuda_test_ops) != 0) return 1;
    if (cuda_test_selected(filter, "parity") &&
        run_cuda_test("parity", yvex_cuda_test_parity) != 0) return 1;
    if (cuda_test_selected(filter, "quant_qtype") &&
        run_cuda_test("quant_qtype", yvex_cuda_test_quant_qtype) != 0) return 1;
    if (cuda_test_selected(filter, "sampling") &&
        run_cuda_test("sampling", yvex_cuda_test_sampling) != 0) return 1;
    if (cuda_test_selected(filter, "materialize_cuda") &&
        run_cuda_test("materialize_cuda", yvex_cuda_test_materialize_cuda) != 0) return 1;
    return 0;
}
