/*
 * Exercises model layer builds descriptor-only model summaries from parsed GGUF metadata and
 * tensor tables without creating executable model state.
 */
#include <string.h>

#include <yvex/api.h>
#include <yvex/internal/model.h>

#include "tests/test.h"

static int open_gguf(const char *path, yvex_artifact **artifact, yvex_gguf **gguf)
{
    yvex_artifact_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.path = path;
    options.readonly = 1;

    rc = yvex_artifact_open(artifact, &options, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "artifact open failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        return 1;
    }

    rc = yvex_gguf_open(gguf, *artifact, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "gguf open failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        yvex_artifact_close(*artifact);
        *artifact = NULL;
        return 1;
    }
    return 0;
}

static int build_descriptor(const char *path,
                            yvex_artifact **artifact,
                            yvex_gguf **gguf,
                            yvex_tensor_table **table,
                            yvex_model_descriptor **model)
{
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_gguf(path, artifact, gguf) == 0, "open gguf");

    rc = yvex_tensor_table_from_gguf(table, *gguf, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "tensor table failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        return 1;
    }

    rc = yvex_model_descriptor_from_gguf(model, *gguf, *table, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "descriptor failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        return 1;
    }
    return 0;
}

static void close_descriptor_stack(yvex_artifact *artifact,
                                   yvex_gguf *gguf,
                                   yvex_tensor_table *table,
                                   yvex_model_descriptor *model)
{
    yvex_model_descriptor_close(model);
    yvex_tensor_table_close(table);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
}

static int test_descriptor_from_c1_fixture(void)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *table = NULL;
    yvex_model_descriptor *model = NULL;

    YVEX_TEST_ASSERT(build_descriptor("tests/fixtures/gguf/valid-metadata-tensors.gguf",
                                      &artifact, &gguf, &table, &model) == 0,
                     "build descriptor");

    YVEX_TEST_ASSERT(yvex_model_arch(model) == YVEX_ARCH_LLAMA, "architecture llama");
    YVEX_TEST_ASSERT_STREQ(yvex_arch_name(yvex_model_arch(model)), "llama", "architecture name");
    YVEX_TEST_ASSERT_STREQ(yvex_model_name(model), "yvex-test", "model name");
    YVEX_TEST_ASSERT(yvex_model_context_length(model) == 4096, "context length");
    YVEX_TEST_ASSERT(yvex_model_tensor_count(model) == 1, "tensor count");
    YVEX_TEST_ASSERT(yvex_model_total_storage_bytes(model) == 128, "known tensor bytes");
    YVEX_TEST_ASSERT(yvex_model_unsupported_tensor_accounting_count(model) == 0,
                     "unsupported tensor accounting count");
    YVEX_TEST_ASSERT(yvex_model_role_count(model, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING) == 1,
                     "token embedding role count");

    close_descriptor_stack(artifact, gguf, table, model);
    return 0;
}

static int test_minimal_descriptor_is_unknown(void)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *table = NULL;
    yvex_model_descriptor *model = NULL;

    YVEX_TEST_ASSERT(build_descriptor("tests/fixtures/gguf/valid-minimal.gguf",
                                      &artifact, &gguf, &table, &model) == 0,
                     "build minimal descriptor");

    YVEX_TEST_ASSERT(yvex_model_arch(model) == YVEX_ARCH_UNKNOWN, "minimal architecture unknown");
    YVEX_TEST_ASSERT_STREQ(yvex_arch_name(yvex_model_arch(model)), "unknown", "unknown architecture name");
    YVEX_TEST_ASSERT_STREQ(yvex_model_name(model), "", "minimal name empty");
    YVEX_TEST_ASSERT(yvex_model_context_length(model) == 0, "minimal context absent");
    YVEX_TEST_ASSERT(yvex_model_tensor_count(model) == 0, "minimal tensor count");
    YVEX_TEST_ASSERT(yvex_model_total_storage_bytes(model) == 0, "minimal known bytes");

    close_descriptor_stack(artifact, gguf, table, model);
    return 0;
}

static int test_attention_numeric_policy_validation(void)
{
    yvex_attention_activation_policy activation = {
        .required = 1,
        .stage = YVEX_ATTENTION_ACTIVATION_KV_NON_ROPE,
        .quantization = YVEX_ATTENTION_QUANT_FP8_E4M3_UE8M0_FAKE_DEQUANT,
        .block_axis = YVEX_ATTENTION_AXIS_FINAL_DIMENSION,
        .block_width = 64ull,
        .scale_format = YVEX_ATTENTION_SCALE_UE8M0,
        .scale_dtype = YVEX_NATIVE_DTYPE_F8_E8M0,
        .pre_transform = YVEX_ATTENTION_TRANSFORM_NONE,
        .tail_policy = YVEX_ATTENTION_TAIL_EXACT_OR_SHORT_FINAL_BLOCK,
        .nonfinite_policy = YVEX_ATTENTION_NONFINITE_REFUSE,
        .fake_quant_inplace = 1};
    yvex_attention_topk_policy topk = {
        .required = 1,
        .version = 1u,
        .policy = YVEX_ATTENTION_TOPK_SCORE_DESC_ORDINAL_ASC_V1,
        .k = 6ull,
        .reject_nonfinite = 1,
        .score_descending = 1,
        .equal_score_ordinal_ascending = 1,
        .plus_zero_equals_minus_zero = 1,
        .duplicate_ordinal_refused = 1,
        .output_ranked_order = 1};
    const yvex_attention_activation_policy *policies[] = {&activation};
    yvex_attention_numeric_mismatch mismatch;

    YVEX_TEST_ASSERT(
        yvex_model_attention_numeric_validate(
            YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
            YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1, policies, 1ull, &topk,
            64ull, 32ull, 1u, &mismatch),
        "complete attention numeric policy validates");
    YVEX_TEST_ASSERT(
        !yvex_model_attention_numeric_validate(
            YVEX_ATTENTION_COMPUTE_UNKNOWN,
            YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1, policies, 1ull, &topk,
            64ull, 32ull, 1u, &mismatch) &&
            mismatch.code == YVEX_ATTENTION_NUMERIC_MISMATCH_COMPUTE,
        "compute mismatch is typed");
    activation.block_width = 63ull;
    YVEX_TEST_ASSERT(
        !yvex_model_attention_numeric_validate(
            YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
            YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1, policies, 1ull, &topk,
            64ull, 32ull, 1u, &mismatch) &&
            mismatch.code == YVEX_ATTENTION_NUMERIC_MISMATCH_ACTIVATION &&
            mismatch.policy_index == 0ull && mismatch.expected == 64ull &&
            mismatch.actual == 63ull,
        "activation block mismatch retains exact evidence");
    activation.block_width = 64ull;
    topk.version = 2u;
    YVEX_TEST_ASSERT(
        !yvex_model_attention_numeric_validate(
            YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
            YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1, policies, 1ull, &topk,
            64ull, 32ull, 1u, &mismatch) &&
            mismatch.code == YVEX_ATTENTION_NUMERIC_MISMATCH_TOPK &&
            mismatch.expected == 1ull && mismatch.actual == 0ull,
        "top-k mismatch retains the admitted refusal evidence");
    return 0;
}

int yvex_test_model_descriptor(void)
{
    if (test_descriptor_from_c1_fixture() != 0) {
        return 1;
    }
    if (test_minimal_descriptor_is_unknown() != 0) {
        return 1;
    }
    if (test_attention_numeric_policy_validation() != 0) {
        return 1;
    }
    return 0;
}
