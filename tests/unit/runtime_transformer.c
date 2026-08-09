/*
 * Exercises the bounded pointer-free transformer plan/input and composed numerical contracts.
 * Reference equations do not call production initial/deferred/final transformer mechanisms.
 * Focused internal-ABI evidence; no fixture enters production objects.
 */
#define _GNU_SOURCE
#include "tests/test.h"

#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/transformer.h>

static void transformer_test_identity(char output[YVEX_SHA256_HEX_CAP], unsigned int value)
{
    (void)snprintf(output, YVEX_SHA256_HEX_CAP, "%064x", value);
}

/* Construct and import one tiny identity-bearing transformer plan. */
static int transformer_test_plan(yvex_transformer_plan **out,
                                 yvex_transformer_plan_summary *summary_out)
{
    yvex_transformer_plan_summary summary;
    yvex_transformer_layer_plan layer;
    yvex_error err;
    unsigned int slot;
    memset(&summary, 0, sizeof(summary));
    memset(&layer, 0, sizeof(layer));
    summary.schema_version = YVEX_TRANSFORMER_PLAN_SCHEMA_V2;
    summary.family_adapter_id = 1ull;
    summary.family_adapter_version = 1ull;
    summary.layer_count = 1ull;
    summary.hidden_width = 2ull;
    summary.residual_streams = 2ull;
    summary.expanded_width = 4ull;
    summary.maximum_context = 8ull;
    summary.vocabulary_size = 16ull;
    summary.initial_policy = YVEX_TRANSFORMER_INITIAL_REPEAT_STREAMS;
    summary.final_policy = YVEX_TRANSFORMER_FINAL_SIGMOID_MHC_RMS;
    summary.sinkhorn_iterations = 2ull;
    summary.mhc_epsilon = 1e-6;
    summary.output_norm_epsilon = 1e-5;
    transformer_test_identity(summary.artifact_identity, 1u);
    transformer_test_identity(summary.materialization_identity, 2u);
    transformer_test_identity(summary.logical_model_identity, 3u);
    transformer_test_identity(summary.runtime_numeric_identity, 4u);
    transformer_test_identity(summary.runtime_descriptor_identity, 5u);
    transformer_test_identity(summary.attention_plan_identity, 6u);
    transformer_test_identity(summary.moe_plan_identity, 7u);
    for (slot = 0u; slot < YVEX_TRANSFORMER_WEIGHT_COUNT; ++slot) {
        summary.weights[slot].tensor_id = slot;
        summary.weights[slot].role = (yvex_tensor_role)(YVEX_TENSOR_ROLE_TOKEN_EMBEDDING + slot);
        summary.weights[slot].qtype = YVEX_GGUF_QTYPE_F32;
        summary.weights[slot].row_width = 1ull + slot;
        summary.weights[slot].row_count = 1ull;
        summary.weights[slot].encoded_bytes = 4ull * (1ull + slot);
    }
    layer.ordinal = layer.layer_index = 0ull;
    transformer_test_identity(layer.moe_layer_identity, 8u);
    transformer_test_identity(layer.layer_identity, 9u);
    YVEX_TEST_ASSERT(yvex_transformer_plan_seal(&summary, &layer, &err) == YVEX_OK,
                     "tiny transformer plan seals field-by-field");
    YVEX_TEST_ASSERT(yvex_transformer_plan_import(out, &summary, &layer, &err) == YVEX_OK,
                     "tiny transformer plan independently reopens");
    if (summary_out) *summary_out = summary;
    return 0;
}

static int transformer_test_family(void)
{
    const yvex_family_compiler_adapter *adapter =
        yvex_compiler_family_deepseek_v4();
    yvex_runtime_descriptor_summary runtime = {0};
    yvex_transformer_family_policy policy;
    runtime.model_execution = (yvex_model_execution_descriptor){
        .schema_version = YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1,
        .maximum_context = 1048576ull,
        .hidden_width = 4096ull,
        .residual_streams = 4ull,
        .mhc_sinkhorn_iterations = 20ull,
        .mhc_epsilon = 1e-6,
        .normalization_epsilon = 1e-6};
    YVEX_TEST_ASSERT(adapter && adapter->transformer_policy &&
                         adapter->transformer_policy(&runtime, &policy),
                     "DeepSeek transformer policy is adapter-projected");
    YVEX_TEST_ASSERT(adapter->adapter_version == 7ull && policy.residual_streams == 4ull &&
                         policy.hidden_width == 4096ull && policy.expanded_width == 16384ull &&
                         policy.mhc_epsilon == 1e-6 &&
                         policy.output_norm_epsilon == 1e-6 &&
                         policy.attention_then_moe && policy.deferred_ffn_post &&
                         policy.final_norm_after_head,
                     "DeepSeek adapter fixes exact four-stream ordered backbone semantics");
    memset(&runtime.model_execution, 0, sizeof(runtime.model_execution));
    YVEX_TEST_ASSERT(!adapter->transformer_policy(&runtime, &policy),
                     "transformer policy refuses an uncompiled execution descriptor");
    return 0;
}

static int transformer_test_numeric(void)
{
    yvex_transformer_plan *plan = NULL;
    float embedding[] = {1.0f, 2.0f}, expanded[4], next[4], normalized[2];
    float pre_normalized[2], pre_expected[2];
    float feature[] = {3.0f, -4.0f}, feature_expected[2];
    float feature_norm[] = {0.5f, 1.5f};
    float residual[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float combined[] = {5.0f, 6.0f}, post[] = {0.5f, 1.0f};
    float combination[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float function[8] = {0.0f}, base[] = {0.0f, 0.0f}, scale[] = {1.0f};
    float norm[] = {1.0f, 2.0f}, expected[2];
    double square, inverse;
    yvex_error err;
    if (transformer_test_plan(&plan, NULL) != 0) return 1;
    YVEX_TEST_ASSERT(yvex_transformer_initial_residual(
                         plan, embedding, 1ull, expanded, &err) == YVEX_OK &&
                         expanded[0] == 1.0f && expanded[1] == 2.0f &&
                         expanded[2] == 1.0f && expanded[3] == 2.0f,
                     "initial residual repeats the admitted embedding across streams");
    YVEX_TEST_ASSERT(yvex_transformer_deferred_post(
                         plan, residual, combined, post, combination, 1ull, next, &err) == YVEX_OK &&
                         next[0] == yvex_quant_bf16_decode(yvex_quant_bf16_encode(3.5f)) &&
                         next[1] == yvex_quant_bf16_decode(yvex_quant_bf16_encode(5.0f)) &&
                         next[2] == yvex_quant_bf16_decode(yvex_quant_bf16_encode(8.0f)) &&
                         next[3] == yvex_quant_bf16_decode(yvex_quant_bf16_encode(10.0f)),
                     "deferred FFN post matches independent residual combination");
    expected[0] = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
        (float)((0.5 + 1e-6) * (next[0] + next[2]))));
    expected[1] = yvex_quant_bf16_decode(yvex_quant_bf16_encode(
        (float)((0.5 + 1e-6) * (next[1] + next[3]))));
    pre_expected[0] = expected[0];
    pre_expected[1] = expected[1];
    square = ((double)expected[0] * expected[0] + (double)expected[1] * expected[1]) / 2.0;
    inverse = 1.0 / sqrt(square + 1e-5);
    expected[0] = yvex_quant_bf16_decode(yvex_quant_bf16_encode((float)(expected[0] * inverse)));
    expected[1] = yvex_quant_bf16_decode(yvex_quant_bf16_encode((float)(expected[1] * inverse * 2.0)));
    YVEX_TEST_ASSERT(yvex_transformer_final_stage(
                         plan, next, 1ull, function, base, scale, norm, normalized, &err) == YVEX_OK &&
                         normalized[0] == expected[0] && normalized[1] == expected[1],
                     "final mHC head and RMSNorm match an independent full equation");
    YVEX_TEST_ASSERT(
        yvex_transformer_final_stage_capture(
            plan, next, 1ull, function, base, scale, norm, pre_normalized,
            normalized, &err) == YVEX_OK &&
            pre_normalized[0] == pre_expected[0] &&
            pre_normalized[1] == pre_expected[1] &&
            normalized[0] == expected[0] && normalized[1] == expected[1],
        "feature capture preserves the exact pre-output-normalized target state");
    square = ((double)feature[0] * feature[0] +
              (double)feature[1] * feature[1]) /
             2.0;
    inverse = 1.0 / sqrt(square + 1e-6);
    feature_expected[0] = yvex_quant_bf16_decode(
        yvex_quant_bf16_encode((float)(feature[0] * inverse * feature_norm[0])));
    feature_expected[1] = yvex_quant_bf16_decode(
        yvex_quant_bf16_encode((float)(feature[1] * inverse * feature_norm[1])));
    YVEX_TEST_ASSERT(
        yvex_transformer_feature_normalize(feature, 2ull, feature_norm, 1e-6,
                                           &err) == YVEX_OK &&
            feature[0] == feature_expected[0] &&
            feature[1] == feature_expected[1],
        "draft feature normalization matches an independent RMS equation");
    feature[0] = NAN;
    YVEX_TEST_ASSERT(
        yvex_transformer_feature_normalize(feature, 2ull, feature_norm, 1e-6,
                                           &err) == YVEX_ERR_FORMAT,
        "non-finite draft feature normalization refuses");
    next[0] = NAN;
    YVEX_TEST_ASSERT(yvex_transformer_final_stage(
                         plan, next, 1ull, function, base, scale, norm, normalized, &err) ==
                         YVEX_ERR_FORMAT,
                     "non-finite final input refuses");
    yvex_transformer_plan_close(&plan);
    return 0;
}

static int transformer_test_router_identity(void)
{
    yvex_moe_router_result router;
    char first[YVEX_SHA256_HEX_CAP], second[YVEX_SHA256_HEX_CAP];
    memset(&router, 0, sizeof(router));
    router.selected_count = 1ull;
    router.selected_experts[0] = 1ull;
    router.router_logits[0] = 0.5f;
    router.router_logits[1] = 1.0f;
    router.router_scores[0] = 0.75f;
    router.router_scores[1] = 1.25f;
    router.selected_weights[0] = 1.0f;
    YVEX_TEST_ASSERT(yvex_moe_router_result_identity(&router, 2ull, first),
                     "router identity seals canonical meaningful fields");
    router.selected_experts[5] = 99ull;
    YVEX_TEST_ASSERT(yvex_moe_router_result_identity(&router, 2ull, second) &&
                         strcmp(first, second) == 0,
                     "unused native capacity cannot affect routing identity");
    router.router_scores[1] += 0.25f;
    YVEX_TEST_ASSERT(yvex_moe_router_result_identity(&router, 2ull, second) &&
                         strcmp(first, second) != 0,
                     "meaningful score mutation changes routing identity");
    return 0;
}

/* Prove bounded memory/file token admission, exact identities, drift, and refusal. */
static int transformer_test_input(void)
{
    yvex_transformer_plan *plan = NULL;
    yvex_transformer_plan_summary plan_summary;
    yvex_transformer_input_summary summary;
    yvex_transformer_input *input = NULL;
    yvex_runtime_binding_summary binding;
    yvex_transformer_input_limits limits = {.maximum_file_bytes = 4096ull};
    unsigned int tokens[] = {3u, 7u};
    char root[] = "/tmp/yvex-transformer-input.XXXXXX";
    char path[256], link_path[256];
    int fd;
    unsigned char byte = 0u;
    yvex_error err;
    if (transformer_test_plan(&plan, &plan_summary) != 0) return 1;
    memset(&summary, 0, sizeof(summary));
    summary.schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
    summary.token_start = 0ull;
    summary.token_count = 2ull;
    summary.vocabulary_size = plan_summary.vocabulary_size;
    yvex_runtime_identity_copy(summary.logical_model_identity,
                               plan_summary.logical_model_identity);
    yvex_runtime_identity_copy(summary.runtime_numeric_identity,
                               plan_summary.runtime_numeric_identity);
    yvex_runtime_identity_copy(summary.runtime_descriptor_identity,
                               plan_summary.runtime_descriptor_identity);
    yvex_runtime_identity_copy(summary.transformer_plan_identity,
                               plan_summary.transformer_plan_identity);
    YVEX_TEST_ASSERT(yvex_transformer_input_seal(&summary, tokens, &err) == YVEX_OK &&
                         yvex_transformer_input_open_memory(&input, &summary, tokens, &err) == YVEX_OK,
                     "identity-bound in-memory numeric token input admits");
    memset(&binding, 0, sizeof(binding));
    yvex_runtime_identity_copy(binding.logical_model_identity, summary.logical_model_identity);
    yvex_runtime_identity_copy(binding.runtime_numeric_identity, summary.runtime_numeric_identity);
    yvex_runtime_identity_copy(binding.runtime_descriptor_identity,
                               summary.runtime_descriptor_identity);
    YVEX_TEST_ASSERT(yvex_transformer_input_validate(input, plan, &binding, &err) == YVEX_OK,
                     "token input validates against exact plan and runtime identities");
    yvex_transformer_input_close(&input);
    tokens[1] = 16u;
    YVEX_TEST_ASSERT(yvex_transformer_input_seal(&summary, tokens, &err) == YVEX_ERR_BOUNDS,
                     "out-of-vocabulary numeric token refuses");
    tokens[1] = 7u;
    YVEX_TEST_ASSERT(mkdtemp(root) != NULL &&
                         snprintf(path, sizeof(path), "%s/input%s", root,
                                  YVEX_TRANSFORMER_INPUT_SUFFIX) < (int)sizeof(path) &&
                         snprintf(link_path, sizeof(link_path), "%s/link%s", root,
                                  YVEX_TRANSFORMER_INPUT_SUFFIX) < (int)sizeof(link_path),
                     "transformer input temporary paths fit");
    YVEX_TEST_ASSERT(yvex_transformer_input_write(path, &summary, tokens, &err) == YVEX_OK &&
                         yvex_transformer_input_open_file(&input, path, &limits, &err) == YVEX_OK &&
                         yvex_transformer_input_validate(input, plan, &binding, &err) == YVEX_OK,
                     "portable regular transformer token file admits");
    fd = open(path, O_WRONLY | O_CLOEXEC);
    YVEX_TEST_ASSERT(fd >= 0 && pwrite(fd, &byte, 1u, 512) == 1 && close(fd) == 0 &&
                         yvex_transformer_input_validate(input, plan, &binding, &err) ==
                         YVEX_ERR_STATE,
                     "post-admission transformer input drift refuses");
    yvex_transformer_input_close(&input);
    YVEX_TEST_ASSERT(symlink(path, link_path) == 0 &&
                         yvex_transformer_input_open_file(&input, link_path, &limits, &err) ==
                         YVEX_ERR_INVALID_ARG,
                     "symlink transformer token input refuses");
    YVEX_TEST_ASSERT(unlink(link_path) == 0 && unlink(path) == 0 && rmdir(root) == 0,
                     "transformer token fixtures clean exactly");
    yvex_transformer_plan_close(&plan);
    return 0;
}

static int transformer_test_block_api_refusal(void)
{
    yvex_runtime_transformer_block_result result;
    yvex_error err;
    YVEX_TEST_ASSERT(yvex_runtime_transformer_execute_block(
                         NULL, 0ull, NULL, 1ull, YVEX_BACKEND_KIND_CPU,
                         NULL, NULL, NULL, NULL, &result, &err) == YVEX_ERR_FORMAT &&
                         !result.completed,
                     "single-block production API refuses an absent active transaction");
    return 0;
}

int yvex_test_runtime_transformer(void)
{
    if (transformer_test_family() != 0) return 1;
    if (transformer_test_numeric() != 0) return 1;
    if (transformer_test_router_identity() != 0) return 1;
    if (transformer_test_input() != 0) return 1;
    if (transformer_test_block_api_refusal() != 0) return 1;
    return 0;
}
