
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static void set_expected(yvex_materialize_expected_tensor *expected)
{
    memset(expected, 0, sizeof(*expected));
    expected->name = "token_embd.weight";
    expected->dtype = "F32";
    expected->rank = 2;
    expected->dims[0] = 4;
    expected->dims[1] = 8;
    expected->bytes = 128;
}

static int test_names(void)
{
    YVEX_TEST_ASSERT_STREQ(yvex_materialize_gate_status_name(YVEX_MATERIALIZE_GATE_PASS),
                           "materialize-gate-pass", "status name");
    YVEX_TEST_ASSERT_STREQ(yvex_materialize_scope_name(YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR),
                           "selected-tensor", "scope name");
    YVEX_TEST_ASSERT_STREQ(yvex_materialize_backend_status_name(YVEX_MATERIALIZE_BACKEND_PASS),
                           "pass", "backend name");
    YVEX_TEST_ASSERT_STREQ(yvex_materialize_failure_class_name(YVEX_MATERIALIZE_FAILURE_HASH_MISMATCH),
                           "hash_mismatch", "failure name");
    return 0;
}

static int test_invalid_options(void)
{
    yvex_materialize_gate_summary summary;
    yvex_error err;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_materialize_gate_check(NULL, &summary, &err) == YVEX_ERR_INVALID_ARG,
                     "null options rejected");
    return 0;
}

static int test_missing_file(void)
{
    yvex_materialize_gate_options options;
    yvex_materialize_gate_summary summary;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.model_path = "build/tests/materialize-gate/missing.gguf";
    options.label = "missing";
    options.family = "llama";
    options.scope = YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR;
    rc = yvex_materialize_gate_check(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "missing file rejected");
    YVEX_TEST_ASSERT(summary.status == YVEX_MATERIALIZE_GATE_BLOCKED, "missing file blocked");
    YVEX_TEST_ASSERT(summary.failure_class == YVEX_MATERIALIZE_FAILURE_MISSING_FILE,
                     "missing file failure class");
    return 0;
}

static int test_hash_mismatch(void)
{
    yvex_materialize_gate_options options;
    yvex_materialize_gate_summary summary;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.model_path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.label = "bad-hash";
    options.family = "llama";
    options.scope = YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR;
    options.sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    rc = yvex_materialize_gate_check(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "hash mismatch rejected");
    YVEX_TEST_ASSERT(summary.status == YVEX_MATERIALIZE_GATE_BLOCKED, "hash blocked");
    YVEX_TEST_ASSERT(summary.failure_class == YVEX_MATERIALIZE_FAILURE_HASH_MISMATCH,
                     "hash failure class");
    return 0;
}

static int test_tensor_mismatch(void)
{
    yvex_materialize_gate_options options;
    yvex_materialize_expected_tensor expected;
    yvex_materialize_gate_summary summary;
    yvex_error err;
    int rc;

    set_expected(&expected);
    expected.dims[0] = 8;
    expected.dims[1] = 4;
    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.model_path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.label = "bad-shape";
    options.family = "llama";
    options.scope = YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR;
    options.expected_tensors = &expected;
    options.expected_tensor_count = 1;
    rc = yvex_materialize_gate_check(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "tensor mismatch rejected");
    YVEX_TEST_ASSERT(summary.status == YVEX_MATERIALIZE_GATE_FAIL, "mismatch fail");
    YVEX_TEST_ASSERT(summary.failure_class == YVEX_MATERIALIZE_FAILURE_TENSOR_SPEC_MISMATCH,
                     "mismatch failure class");
    return 0;
}

static int test_selected_tensor_cpu_repeat_cleanup(void)
{
    yvex_materialize_gate_options options;
    yvex_materialize_expected_tensor expected;
    yvex_materialize_gate_summary summary;
    yvex_error err;
    int rc;

    set_expected(&expected);
    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.model_path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.label = "fixture-selected";
    options.family = "llama";
    options.scope = YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR;
    options.expected_tensors = &expected;
    options.expected_tensor_count = 1;
    options.check_cpu = 1;
    options.require_cpu = 1;
    options.repeat_count = 2;
    options.check_cleanup = 1;
    rc = yvex_materialize_gate_check(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "gate passes");
    YVEX_TEST_ASSERT(summary.status == YVEX_MATERIALIZE_GATE_PASS, "pass status");
    YVEX_TEST_ASSERT(summary.scope == YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR, "selected scope");
    YVEX_TEST_ASSERT(summary.cpu_status == YVEX_MATERIALIZE_BACKEND_PASS, "cpu pass");
    YVEX_TEST_ASSERT(summary.repeat_count == 2, "repeat count");
    YVEX_TEST_ASSERT(summary.cleanup_verified == 1, "cleanup verified");
    YVEX_TEST_ASSERT(summary.execution_ready == 0, "execution not ready");
    return 0;
}

int yvex_test_materialize_gate(void)
{
    if (test_names() != 0) return 1;
    if (test_invalid_options() != 0) return 1;
    if (test_missing_file() != 0) return 1;
    if (test_hash_mismatch() != 0) return 1;
    if (test_tensor_mismatch() != 0) return 1;
    if (test_selected_tensor_cpu_repeat_cleanup() != 0) return 1;
    return 0;
}
