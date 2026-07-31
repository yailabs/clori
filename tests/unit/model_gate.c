
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static void set_expected(yvex_model_gate_expected_tensor *expected)
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
    YVEX_TEST_ASSERT_STREQ(yvex_model_gate_status_name(YVEX_MODEL_GATE_PASS),
                           "model-gate-pass", "gate pass name");
    YVEX_TEST_ASSERT_STREQ(yvex_model_support_level_name(YVEX_MODEL_SUPPORT_SELECTED_TENSOR_MATERIALIZED),
                           "selected-tensor-materialized", "support name");
    YVEX_TEST_ASSERT_STREQ(yvex_model_gate_backend_status_name(YVEX_MODEL_GATE_BACKEND_PASS),
                           "pass", "backend pass name");
    return 0;
}

static int test_invalid_options(void)
{
    yvex_model_gate_summary summary;
    yvex_error err;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_gate_check(NULL, &summary, &err) == YVEX_ERR_INVALID_ARG,
                     "null options rejected");
    return 0;
}

static int test_missing_path_blocks(void)
{
    yvex_model_gate_options options;
    yvex_model_gate_summary summary;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.model_path = "build/tests/model-gate/missing.gguf";
    rc = yvex_model_gate_check(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "missing path rejected");
    YVEX_TEST_ASSERT(summary.status == YVEX_MODEL_GATE_BLOCKED, "missing path blocked");
    return 0;
}

static int test_selected_tensor_gate(void)
{
    yvex_model_gate_options options;
    yvex_model_gate_expected_tensor expected;
    yvex_model_gate_summary summary;
    yvex_error err;
    int rc;

    set_expected(&expected);
    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.model_path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.model_label = "fixture";
    options.family = "llama";
    options.expected_tensors = &expected;
    options.expected_tensor_count = 1;
    options.check_cpu = 1;
    options.require_cpu = 1;

    rc = yvex_model_gate_check(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "model gate passes fixture");
    YVEX_TEST_ASSERT(summary.status == YVEX_MODEL_GATE_PASS, "gate pass");
    YVEX_TEST_ASSERT(summary.support_level == YVEX_MODEL_SUPPORT_SELECTED_TENSOR_MATERIALIZED,
                     "selected tensor support");
    YVEX_TEST_ASSERT(summary.tensor_count == 1, "tensor count");
    YVEX_TEST_ASSERT(summary.expected_tensor_matches == 1, "expected match");
    YVEX_TEST_ASSERT(summary.expected_tensor_mismatches == 0, "no mismatch");
    YVEX_TEST_ASSERT(summary.cpu_status == YVEX_MODEL_GATE_BACKEND_PASS, "cpu pass");
    YVEX_TEST_ASSERT(summary.execution_ready == 0, "execution not ready");
    return 0;
}

static int test_tensor_mismatch_fails(void)
{
    yvex_model_gate_options options;
    yvex_model_gate_expected_tensor expected;
    yvex_model_gate_summary summary;
    yvex_error err;
    int rc;

    set_expected(&expected);
    expected.dims[0] = 8;
    expected.dims[1] = 4;
    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.model_path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.model_label = "bad-shape";
    options.family = "llama";
    options.expected_tensors = &expected;
    options.expected_tensor_count = 1;

    rc = yvex_model_gate_check(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "mismatch fails");
    YVEX_TEST_ASSERT(summary.status == YVEX_MODEL_GATE_FAIL, "mismatch status fail");
    YVEX_TEST_ASSERT(summary.expected_tensor_mismatches == 1, "mismatch counted");
    return 0;
}

static int test_hash_mismatch_blocks(void)
{
    yvex_model_gate_options options;
    yvex_model_gate_expected_tensor expected;
    yvex_model_gate_summary summary;
    yvex_error err;
    int rc;

    set_expected(&expected);
    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    options.model_path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.model_label = "bad-hash";
    options.family = "llama";
    options.artifact_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    options.expected_tensors = &expected;
    options.expected_tensor_count = 1;

    rc = yvex_model_gate_check(&options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "hash mismatch fails");
    YVEX_TEST_ASSERT(summary.status == YVEX_MODEL_GATE_BLOCKED, "hash mismatch blocked");
    return 0;
}

int yvex_test_model_gate(void)
{
    if (test_names() != 0) return 1;
    if (test_invalid_options() != 0) return 1;
    if (test_missing_path_blocks() != 0) return 1;
    if (test_selected_tensor_gate() != 0) return 1;
    if (test_tensor_mismatch_fails() != 0) return 1;
    if (test_hash_mismatch_blocks() != 0) return 1;
    return 0;
}
