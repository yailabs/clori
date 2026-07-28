/*
 * quant_runner.c - Focused quantization and trusted-input test runner.
 *
 * Owner: TRACK.QUANT focused validation.
 * Owns: deterministic orchestration of the qtype ABI, Transformation IR,
 *        trusted payload binding, numeric codec, executor, and compatibility
 *        projection unit suites.
 * Does not own: production behavior, source files, artifacts, CUDA execution,
 *               operator output, or capability promotion.
 * Invariants: every invoked suite is independently owned; first failure stops
 *             the runner and produces a nonzero process status.
 * Boundary: test-only runner used by normal and sanitizer validation.
 */

#include "tests/test.h"

/*
 * run_test executes one owned suite without retaining suite state.  It
 * allocates no memory, mutates no production state, performs no payload IO on
 * its own, and reports only test-runner progress to stderr.
 */
static int run_test(const char *name, int (*fn)(void))
{
    int rc;

    fprintf(stderr, "quant test: %s\n", name);
    rc = fn();
    if (rc != 0) {
        fprintf(stderr, "FAIL: %s exited %d\n", name, rc);
    }
    return rc;
}

/*
 * main sequences the complete focused quantization dependency surface.  It
 * owns no resources beyond those released by each suite and returns on the
 * first refusal so sanitizer diagnostics remain local to the failing owner.
 */
int main(void)
{
    if (run_test("gguf_qtype_abi", yvex_test_gguf_qtype_abi) != 0) return 1;
    if (run_test("source_payload", yvex_test_source_payload) != 0) return 1;
    if (run_test("transform_ir", yvex_test_transform_ir) != 0) return 1;
    if (run_test("deepseek_tensor_coverage",
                 yvex_test_deepseek_tensor_coverage) != 0) return 1;
    if (run_test("quant_numeric", yvex_test_quant_numeric) != 0) return 1;
    if (run_test("quant_execute", yvex_test_quant_execute) != 0) return 1;
    if (run_test("gguf_writer_artifact",
                 yvex_test_gguf_writer_artifact) != 0) return 1;
    if (run_test("qtype_support", yvex_test_qtype_support) != 0) return 1;
    if (run_test("quant_policy", yvex_test_quant_policy) != 0) return 1;
    if (run_test("imatrix", yvex_test_imatrix) != 0) return 1;
    return 0;
}
