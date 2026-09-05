/* Generic Gated DeltaNet geometry, exact recurrence, cached decode, and rollback evidence. */
#include "tests/test.h"

#include <math.h>
#include <string.h>

#include <yvex/internal/sequence_mixer.h>

static yvex_gated_delta_requirement tiny_requirement(void)
{
    return (yvex_gated_delta_requirement){
        .schema_version = YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V2,
        .output_normalization_weight_convention =
            YVEX_NORMALIZATION_WEIGHT_DIRECT,
        .query_heads = 1ull,
        .key_heads = 1ull,
        .value_heads = 2ull,
        .key_head_dimension = 2ull,
        .value_head_dimension = 2ull,
        .convolution_kernel = 2ull,
        .projected_dtype = YVEX_DTYPE_F32,
        .convolution_state_dtype = YVEX_DTYPE_F32,
        .recurrent_state_dtype = YVEX_DTYPE_F32,
        .accumulation_dtype = YVEX_DTYPE_F32,
        .output_dtype = YVEX_DTYPE_F32,
        .numeric_contract = YVEX_SEQUENCE_MIXER_NUMERIC_F32_RECURRENCE,
        .qk_normalization_epsilon = 1e-6,
        .output_normalization_epsilon = 1e-6,
        .query_scale = 0.70710678118654752440,
        .deterministic = 1};
}

static int mixer_near(const float *left, const float *right,
                      unsigned long long count, float tolerance)
{
    unsigned long long index;
    for (index = 0ull; index < count; ++index)
        if (!isfinite(left[index]) || !isfinite(right[index]) ||
            fabsf(left[index] - right[index]) > tolerance)
            return 0;
    return 1;
}

static int mixer_test_qwen_geometry(void)
{
    yvex_gated_delta_requirement requirement = tiny_requirement();
    yvex_gated_delta_plan plan;
    yvex_error err;

    requirement.query_heads = 16ull;
    requirement.key_heads = 16ull;
    requirement.value_heads = 48ull;
    requirement.key_head_dimension = 128ull;
    requirement.value_head_dimension = 128ull;
    requirement.convolution_kernel = 4ull;
    requirement.query_scale = 1.0 / sqrt(128.0);
    YVEX_TEST_ASSERT(yvex_gated_delta_plan_seal(&plan, &requirement, &err) == YVEX_OK,
                     "Qwen-class gated-delta geometry seals");
    YVEX_TEST_ASSERT(plan.query_width == 2048ull && plan.key_width == 2048ull &&
                         plan.value_width == 6144ull && plan.qkv_width == 10240ull,
                     "Qwen-class projected widths remain exact");
    YVEX_TEST_ASSERT(plan.convolution_state_values == 30720ull &&
                         plan.convolution_state_bytes == 122880ull &&
                         plan.recurrent_state_values == 786432ull &&
                         plan.recurrent_state_bytes == 3145728ull,
                     "convolution and recurrent state bytes remain distinct");
    YVEX_TEST_ASSERT(yvex_gated_delta_plan_validate(&plan, &err) == YVEX_OK,
                     "sealed gated-delta identity revalidates");
    return 0;
}

static void mixer_fixture(float qkv[24], float gate[12], float beta[6],
                          float decay[6], float conv[16])
{
    unsigned long long index;
    for (index = 0ull; index < 24ull; ++index)
        qkv[index] = (float)((long long)(index % 9ull) - 4ll) * 0.125f;
    for (index = 0ull; index < 12ull; ++index)
        gate[index] = (float)((long long)(index % 5ull) - 2ll) * 0.2f;
    for (index = 0ull; index < 6ull; ++index) {
        beta[index] = -0.35f + (float)index * 0.17f;
        decay[index] = -0.20f + (float)index * 0.09f;
    }
    for (index = 0ull; index < 8ull; ++index) {
        conv[index * 2ull] = 0.25f;
        conv[index * 2ull + 1ull] = 0.75f;
    }
}

static int mixer_run(const yvex_gated_delta_plan *plan,
                     const float *qkv, const float *gate, const float *beta,
                     const float *decay, const float *conv,
                     unsigned long long tokens, yvex_sequence_state_view state,
                     float *next_conv, float *next_recurrent, float *output)
{
    const float decay_log[2] = {-0.8f, -0.35f};
    const float time_bias[2] = {0.1f, -0.15f};
    const float norm_weight[2] = {1.0f, 1.25f};
    yvex_gated_delta_cpu_request request = {
        .token_count = tokens,
        .projected_qkv = qkv,
        .projected_qkv_capacity = tokens * plan->qkv_width,
        .projected_output_gate = gate,
        .projected_output_gate_capacity = tokens * plan->value_width,
        .projected_beta = beta,
        .projected_beta_capacity = tokens * plan->requirement.value_heads,
        .projected_decay = decay,
        .projected_decay_capacity = tokens * plan->requirement.value_heads,
        .convolution_weight = conv,
        .convolution_weight_capacity = plan->qkv_width *
                                       plan->requirement.convolution_kernel,
        .decay_log = decay_log,
        .decay_log_capacity = plan->requirement.value_heads,
        .time_bias = time_bias,
        .time_bias_capacity = plan->requirement.value_heads,
        .normalization_weight = norm_weight,
        .normalization_weight_capacity = plan->requirement.value_head_dimension,
        .state = state,
        .next_state = {next_conv, plan->convolution_state_values,
                       next_recurrent, plan->recurrent_state_values},
        .output = output,
        .output_capacity = tokens * plan->value_width};
    yvex_gated_delta_cpu_result result;
    yvex_error err;

    YVEX_TEST_ASSERT(yvex_gated_delta_execute_cpu(plan, &request, &result, &err) == YVEX_OK,
                     "exact gated-delta CPU execution completes");
    YVEX_TEST_ASSERT(result.complete && result.token_count == tokens &&
                         result.output_values == tokens * plan->value_width,
                     "execution publishes exact bounded counts");
    return 0;
}

static int mixer_test_prefill_decode(void)
{
    yvex_gated_delta_requirement requirement = tiny_requirement();
    yvex_gated_delta_plan plan;
    yvex_error err;
    float qkv[24], gate[12], beta[6], decay[6], conv[16];
    float prefill_conv[8], prefill_recurrent[8], prefill_output[12];
    float decode_conv[2][8] = {{0}}, decode_recurrent[2][8] = {{0}};
    float decode_output[12] = {0};
    unsigned int current = 0u, candidate = 1u;
    unsigned long long token;

    YVEX_TEST_ASSERT(yvex_gated_delta_plan_seal(&plan, &requirement, &err) == YVEX_OK,
                     "tiny gated-delta plan seals");
    mixer_fixture(qkv, gate, beta, decay, conv);
    if (mixer_run(&plan, qkv, gate, beta, decay, conv, 3ull,
                  (yvex_sequence_state_view){0}, prefill_conv,
                  prefill_recurrent, prefill_output) != 0)
        return 1;
    for (token = 0ull; token < 3ull; ++token) {
        yvex_sequence_state_view state = token
            ? (yvex_sequence_state_view){decode_conv[current],
                                            decode_recurrent[current]}
            : (yvex_sequence_state_view){0};
        if (mixer_run(&plan, qkv + token * plan.qkv_width,
                      gate + token * plan.value_width,
                      beta + token * requirement.value_heads,
                      decay + token * requirement.value_heads, conv, 1ull, state,
                      decode_conv[candidate], decode_recurrent[candidate],
                      decode_output + token * plan.value_width) != 0)
            return 1;
        current = candidate;
        candidate ^= 1u;
    }
    YVEX_TEST_ASSERT(mixer_near(prefill_output, decode_output, 12ull, 1e-7f),
                     "multi-token prefill equals cached single-token decode");
    YVEX_TEST_ASSERT(mixer_near(prefill_conv, decode_conv[current],
                                plan.convolution_state_values, 0.0f) &&
                         mixer_near(prefill_recurrent, decode_recurrent[current],
                                    plan.recurrent_state_values, 1e-7f),
                     "prefill and decode publish identical mixed sequence state");
    YVEX_TEST_ASSERT(fabsf(prefill_output[1] + 0.158362687f) < 1e-6f &&
                         fabsf(prefill_output[11] - 0.152190059f) < 1e-6f,
                     "fixed numerical fixture remains stable");
    return 0;
}

static int mixer_test_normalization_convention(void)
{
    yvex_gated_delta_requirement direct_requirement = tiny_requirement();
    yvex_gated_delta_requirement one_plus_requirement = tiny_requirement();
    yvex_gated_delta_plan direct, one_plus;
    yvex_error err;
    float qkv[24], gate[12], beta[6], decay[6], conv[16];
    float direct_conv[8], direct_recurrent[8], direct_output[4];
    float one_plus_conv[8], one_plus_recurrent[8], one_plus_output[4];
    unsigned long long lane;

    one_plus_requirement.output_normalization_weight_convention =
        YVEX_NORMALIZATION_WEIGHT_ONE_PLUS;
    YVEX_TEST_ASSERT(
        yvex_gated_delta_plan_seal(&direct, &direct_requirement, &err) == YVEX_OK &&
            yvex_gated_delta_plan_seal(&one_plus, &one_plus_requirement, &err) == YVEX_OK &&
            strcmp(direct.identity, one_plus.identity) != 0,
        "normalization convention participates in gated-delta identity");
    mixer_fixture(qkv, gate, beta, decay, conv);
    if (mixer_run(&direct, qkv, gate, beta, decay, conv, 1ull,
                  (yvex_sequence_state_view){0}, direct_conv,
                  direct_recurrent, direct_output) != 0 ||
        mixer_run(&one_plus, qkv, gate, beta, decay, conv, 1ull,
                  (yvex_sequence_state_view){0}, one_plus_conv,
                  one_plus_recurrent, one_plus_output) != 0)
        return 1;
    YVEX_TEST_ASSERT(
        mixer_near(direct_conv, one_plus_conv, 8ull, 0.0f) &&
            mixer_near(direct_recurrent, one_plus_recurrent, 8ull, 0.0f),
        "output-weight convention cannot alter recurrent state");
    for (lane = 0ull; lane < 4ull; ++lane) {
        float ratio = lane % 2ull ? 1.8f : 2.0f;
        YVEX_TEST_ASSERT(fabsf(one_plus_output[lane] -
                               direct_output[lane] * ratio) < 1e-6f,
                         "one-plus output normalization is applied exactly");
    }
    return 0;
}

static int mixer_cancel(void *context)
{
    (void)context;
    return 1;
}

static int mixer_test_cancel(void)
{
    yvex_gated_delta_requirement requirement = tiny_requirement();
    yvex_gated_delta_plan plan;
    yvex_error err;
    float qkv[24], gate[12], beta[6], decay[6], conv[16];
    float committed_conv[8], committed_recurrent[8], candidate_conv[8];
    float candidate_recurrent[8], output[12];
    const float decay_log[2] = {-0.8f, -0.35f};
    const float time_bias[2] = {0.1f, -0.15f};
    const float norm_weight[2] = {1.0f, 1.25f};
    yvex_gated_delta_cpu_request request;
    yvex_gated_delta_cpu_result result;
    unsigned long long index;

    YVEX_TEST_ASSERT(yvex_gated_delta_plan_seal(&plan, &requirement, &err) == YVEX_OK,
                     "cancellation fixture plan seals");
    mixer_fixture(qkv, gate, beta, decay, conv);
    for (index = 0ull; index < 8ull; ++index) {
        committed_conv[index] = 1.0f + (float)index;
        committed_recurrent[index] = 2.0f + (float)index;
    }
    request = (yvex_gated_delta_cpu_request){
        .token_count = 3ull, .projected_qkv = qkv,
        .projected_qkv_capacity = 24ull,
        .projected_output_gate = gate, .projected_beta = beta,
        .projected_output_gate_capacity = 12ull,
        .projected_beta_capacity = 6ull,
        .projected_decay = decay, .convolution_weight = conv,
        .projected_decay_capacity = 6ull,
        .convolution_weight_capacity = 16ull,
        .decay_log = decay_log, .time_bias = time_bias,
        .decay_log_capacity = 2ull, .time_bias_capacity = 2ull,
        .normalization_weight = norm_weight,
        .normalization_weight_capacity = 2ull,
        .state = {committed_conv, committed_recurrent},
        .next_state = {candidate_conv, 8ull, candidate_recurrent, 8ull},
        .output = output, .output_capacity = 12ull,
        .cancel_requested = mixer_cancel};
    YVEX_TEST_ASSERT(yvex_gated_delta_execute_cpu(&plan, &request, &result, &err) ==
                         YVEX_ERR_CANCELLED && result.cancelled && !result.complete,
                     "cancellation refuses candidate publication");
    for (index = 0ull; index < 8ull; ++index)
        YVEX_TEST_ASSERT(committed_conv[index] == 1.0f + (float)index &&
                             committed_recurrent[index] == 2.0f + (float)index,
                         "cancellation cannot mutate committed sequence state");
    return 0;
}

int yvex_test_sequence_mixer(void)
{
    if (mixer_test_qwen_geometry() != 0) return 1;
    if (mixer_test_prefill_decode() != 0) return 1;
    if (mixer_test_normalization_convention() != 0) return 1;
    if (mixer_test_cancel() != 0) return 1;
    return 0;
}
