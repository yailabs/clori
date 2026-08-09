#include <yvex/internal/decode.h>
#include <yvex/internal/sampling.h>

#include <math.h>
#include <stdint.h>

#include "tests/test.h"

static unsigned int reference_sample(const float *target, const float *draft,
                                     unsigned long long count, double uniform,
                                     int residual)
{
    double total = 0.0, cumulative = 0.0;
    unsigned long long index;
    for (index = 0ull; index < count; ++index)
        total += residual ? fmax((double)target[index] - draft[index], 0.0)
                          : (double)target[index];
    if (residual && total <= 1e-8) {
        residual = 0;
        total = 0.0;
        for (index = 0ull; index < count; ++index)
            total += target[index];
    }
    if (total <= 0.0) return UINT32_MAX;
    for (index = 0ull; index < count; ++index) {
        double value = residual
                           ? fmax((double)target[index] - draft[index], 0.0)
                           : (double)target[index];
        cumulative += value / total;
        if (uniform < cumulative || index + 1ull == count)
            return (unsigned int)index;
    }
    return UINT32_MAX;
}

static void reference_stochastic_accept(
    const unsigned int candidates[2], const float draft[6],
    const float target[9], const double draws[2], double correction,
    unsigned int committed[3], unsigned long long *accepted,
    unsigned long long *rejected, unsigned long long *committed_count)
{
    unsigned long long index;
    *accepted = 0ull;
    *rejected = 0ull;
    *committed_count = 0ull;
    for (index = 0ull; index < 2ull; ++index) {
        const float *q = draft + index * 3ull;
        const float *p = target + index * 3ull;
        unsigned int candidate = candidates[index];
        if (draws[index] >=
            fmin(1.0, (double)p[candidate] / (double)q[candidate])) {
            committed[(*committed_count)++] =
                reference_sample(p, q, 3ull, correction, 1);
            *rejected = 2ull - index;
            return;
        }
        committed[(*committed_count)++] = candidate;
        (*accepted)++;
    }
    committed[(*committed_count)++] =
        reference_sample(target + 6ull, NULL, 3ull, correction, 0);
}

static int test_greedy_acceptance(void)
{
    const unsigned int candidates[] = {1u, 2u, 3u};
    const unsigned int targets[] = {1u, 2u, 3u, 4u};
    unsigned int committed[4] = {0};
    yvex_speculation_acceptance_request request = {
        .schema_version = YVEX_SPECULATION_SCHEMA_V1,
        .kind = YVEX_SPECULATION_ACCEPT_GREEDY,
        .candidate_count = 3ull,
        .vocabulary_size = 8ull,
        .distribution_stride = 8ull,
        .candidate_token_ids = candidates,
        .target_token_ids = targets};
    yvex_speculation_acceptance_result result;
    yvex_error err;
    YVEX_TEST_ASSERT(yvex_speculation_accept(&request, committed, 4ull, &result, &err) ==
                         YVEX_OK,
                     "all-accepted greedy speculation");
    YVEX_TEST_ASSERT(result.accepted_draft_count == 3ull &&
                         result.rejected_draft_count == 0ull &&
                         result.committed_count == 4ull && result.all_candidates_accepted &&
                         result.bonus_present && !result.correction_present &&
                         committed[0] == 1u && committed[1] == 2u &&
                         committed[2] == 3u && committed[3] == 4u &&
                         yvex_sha256_hex_valid(result.policy_identity) &&
                         yvex_sha256_hex_valid(result.acceptance_identity),
                     "greedy accepted-prefix and bonus facts");
    return 0;
}

static int test_greedy_rejection(void)
{
    const unsigned int candidates[] = {1u, 7u, 3u};
    const unsigned int alternate_suffix[] = {1u, 7u, 6u};
    const unsigned int targets[] = {1u, 2u, 5u, 4u};
    unsigned int committed[4] = {0};
    yvex_speculation_acceptance_request request = {
        YVEX_SPECULATION_SCHEMA_V1, YVEX_SPECULATION_ACCEPT_GREEDY,
        3ull, 8ull, 8ull, candidates, targets, NULL, NULL, NULL, 0.0};
    yvex_speculation_acceptance_result result, alternate;
    yvex_error err;
    YVEX_TEST_ASSERT(yvex_speculation_accept(&request, committed, 4ull, &result, &err) ==
                         YVEX_OK,
                     "partially accepted greedy speculation");
    YVEX_TEST_ASSERT(result.accepted_draft_count == 1ull &&
                         result.rejected_draft_count == 2ull &&
                         result.rejection_index == 1ull &&
                         result.committed_count == 2ull && result.correction_present &&
                         !result.bonus_present && committed[0] == 1u && committed[1] == 2u,
                     "greedy rejection commits only accepted prefix and correction");
    request.candidate_token_ids = alternate_suffix;
    YVEX_TEST_ASSERT(
        yvex_speculation_accept(&request, committed, 4ull, &alternate, &err) ==
                YVEX_OK &&
            strcmp(result.acceptance_identity, alternate.acceptance_identity) == 0,
        "an uncommitted rejected suffix cannot perturb the admitted result identity");
    return 0;
}

static int test_stochastic_acceptance(void)
{
    const unsigned int candidates[] = {1u, 0u};
    const float draft[] = {0.1f, 0.8f, 0.1f, 0.7f, 0.2f, 0.1f};
    const float target[] = {
        0.1f, 0.4f, 0.5f,
        0.1f, 0.2f, 0.7f,
        0.2f, 0.3f, 0.5f};
    const double draws[] = {0.2, 0.9};
    unsigned int committed[3] = {0};
    yvex_speculation_acceptance_request request = {
        YVEX_SPECULATION_SCHEMA_V1, YVEX_SPECULATION_ACCEPT_STOCHASTIC,
        2ull, 3ull, 3ull, candidates, NULL, draft, target, draws, 0.5};
    yvex_speculation_acceptance_result first, second;
    yvex_error err;
    YVEX_TEST_ASSERT(yvex_speculation_accept(&request, committed, 3ull, &first, &err) ==
                         YVEX_OK,
                     "stochastic speculative acceptance");
    YVEX_TEST_ASSERT(first.accepted_draft_count == 1ull &&
                         first.rejected_draft_count == 1ull &&
                         first.committed_count == 2ull && committed[0] == 1u &&
                         committed[1] == 2u && first.correction_present,
                     "residual sampling preserves target distribution boundary");
    YVEX_TEST_ASSERT(yvex_speculation_accept(&request, committed, 3ull, &second, &err) ==
                         YVEX_OK &&
                         strcmp(first.acceptance_identity, second.acceptance_identity) == 0,
                     "stochastic acceptance is deterministic for explicit draws");
    second = first;
    second.policy_identity[0] = '\0';
    second.acceptance_identity[0] = '\0';
    YVEX_TEST_ASSERT(
        yvex_speculation_acceptance_seal(3ull, committed, 3ull, &second, &err) == YVEX_OK &&
            strcmp(first.policy_identity, second.policy_identity) == 0 &&
            strcmp(first.acceptance_identity, second.acceptance_identity) == 0,
        "bounded device facts seal to the canonical stochastic acceptance identity");
    second.rejection_index = 0ull;
    YVEX_TEST_ASSERT(
        yvex_speculation_acceptance_seal(3ull, committed, 3ull, &second, &err) ==
            YVEX_ERR_FORMAT,
        "inconsistent bounded acceptance facts refuse before identity publication");
    second = first;
    second.bonus_present = 2;
    second.correction_present = -1;
    YVEX_TEST_ASSERT(
        yvex_speculation_acceptance_seal(3ull, committed, 3ull, &second, &err) ==
            YVEX_ERR_FORMAT,
        "non-Boolean bounded acceptance facts refuse before identity publication");
    return 0;
}

static int test_stochastic_bonus(void)
{
    const unsigned int candidates[] = {0u, 1u};
    const float draft[] = {0.7f, 0.2f, 0.1f, 0.1f, 0.8f, 0.1f};
    const float target[] = {
        0.8f, 0.1f, 0.1f,
        0.1f, 0.8f, 0.1f,
        0.2f, 0.3f, 0.5f};
    const double draws[] = {0.99, 0.25};
    unsigned int committed[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    yvex_speculation_acceptance_request request = {
        YVEX_SPECULATION_SCHEMA_V1, YVEX_SPECULATION_ACCEPT_STOCHASTIC,
        2ull, 3ull, 3ull, candidates, NULL, draft, target, draws, 0.75};
    yvex_speculation_acceptance_result result;
    yvex_error err;
    YVEX_TEST_ASSERT(
        yvex_speculation_accept(&request, committed, 3ull, &result, &err) ==
                YVEX_OK &&
            result.all_candidates_accepted && result.bonus_present &&
            !result.correction_present && result.accepted_draft_count == 2ull &&
            result.committed_count == 3ull && committed[0] == 0u &&
            committed[1] == 1u && committed[2] == 2u,
        "all-accepted stochastic speculation samples one target bonus");
    return 0;
}

static int test_stochastic_residual_fallback(void)
{
    const unsigned int candidates[] = {0u};
    const float draft[] = {5e-9f, 1.0f};
    const float target[] = {1e-9f, 1.0f, 1.0f, 0.0f};
    const double draws[] = {0.9};
    unsigned int committed[2] = {UINT32_MAX, UINT32_MAX};
    yvex_speculation_acceptance_request request = {
        YVEX_SPECULATION_SCHEMA_V1, YVEX_SPECULATION_ACCEPT_STOCHASTIC,
        1ull, 2ull, 2ull, candidates, NULL, draft, target, draws, 0.0};
    yvex_speculation_acceptance_result result;
    yvex_error err;
    YVEX_TEST_ASSERT(
        yvex_speculation_accept(&request, committed, 2ull, &result, &err) ==
                YVEX_OK &&
            result.correction_present && result.committed_count == 1ull &&
            committed[0] == 0u,
        "near-empty residual follows the official target fallback");
    return 0;
}

static int test_stochastic_tiny_draft_mass(void)
{
    const unsigned int candidates[] = {0u};
    const float draft[] = {1e-9f, 1.0f};
    const float target[] = {5e-9f, 1.0f, 1.0f, 0.0f};
    const double draws[] = {0.75};
    unsigned int committed[2] = {UINT32_MAX, UINT32_MAX};
    yvex_speculation_acceptance_request request = {
        YVEX_SPECULATION_SCHEMA_V1, YVEX_SPECULATION_ACCEPT_STOCHASTIC,
        1ull, 2ull, 2ull, candidates, NULL, draft, target, draws, 0.25};
    yvex_speculation_acceptance_result result;
    yvex_error err;

    YVEX_TEST_ASSERT(
        yvex_speculation_accept(&request, committed, 2ull, &result, &err) ==
                YVEX_OK &&
            result.accepted_draft_count == 1ull &&
            result.rejected_draft_count == 0ull &&
            result.all_candidates_accepted && result.bonus_present &&
            committed[0] == 0u && committed[1] == 0u,
        "positive draft mass uses the exact p/q acceptance ratio");
    return 0;
}

static int test_stochastic_reference_matrix(void)
{
    static const float draft[] = {
        0.6f, 0.3f, 0.1f,
        0.1f, 0.7f, 0.2f};
    static const float target[] = {
        0.2f, 0.5f, 0.3f,
        0.4f, 0.2f, 0.4f,
        0.3f, 0.2f, 0.5f};
    static const double uniforms[] = {0.0, 0.2, 0.5, 0.9};
    static const double corrections[] = {0.1, 0.7};
    unsigned int candidates[2], expected[3], actual[3];
    unsigned long long first, second, draw_a, draw_b, correction_index;
    yvex_error err;

    for (first = 0ull; first < 3ull; ++first)
        for (second = 0ull; second < 3ull; ++second)
            for (draw_a = 0ull; draw_a < 4ull; ++draw_a)
                for (draw_b = 0ull; draw_b < 4ull; ++draw_b)
                    for (correction_index = 0ull; correction_index < 2ull;
                         ++correction_index) {
                        double draws[2] = {uniforms[draw_a], uniforms[draw_b]};
                        unsigned long long accepted, rejected, count;
                        yvex_speculation_acceptance_result result;
                        yvex_speculation_acceptance_request request = {
                            YVEX_SPECULATION_SCHEMA_V1,
                            YVEX_SPECULATION_ACCEPT_STOCHASTIC,
                            2ull, 3ull, 3ull, candidates, NULL, draft, target,
                            draws, corrections[correction_index]};
                        candidates[0] = (unsigned int)first;
                        candidates[1] = (unsigned int)second;
                        memset(expected, 0, sizeof(expected));
                        memset(actual, 0, sizeof(actual));
                        reference_stochastic_accept(
                            candidates, draft, target, draws,
                            corrections[correction_index], expected, &accepted,
                            &rejected, &count);
                        YVEX_TEST_ASSERT(
                            yvex_speculation_accept(
                                &request, actual, 3ull, &result, &err) == YVEX_OK &&
                                result.accepted_draft_count == accepted &&
                                result.rejected_draft_count == rejected &&
                                result.committed_count == count &&
                                memcmp(actual, expected,
                                       (size_t)count * sizeof(*actual)) == 0,
                            "production stochastic acceptance matches the independent matrix");
                    }
    return 0;
}

static int test_refusals(void)
{
    const unsigned int candidates[] = {0u};
    const float malformed[] = {0.2f, 0.2f};
    const float target[] = {0.5f, 0.5f, 0.5f, 0.5f};
    const double draws[] = {0.1};
    unsigned int committed[2] = {0};
    yvex_speculation_acceptance_request request = {
        YVEX_SPECULATION_SCHEMA_V1, YVEX_SPECULATION_ACCEPT_STOCHASTIC,
        1ull, 2ull, 2ull, candidates, NULL, malformed, target, draws, 0.5};
    yvex_speculation_acceptance_result result;
    yvex_error err;
    YVEX_TEST_ASSERT(yvex_speculation_accept(&request, committed, 2ull, &result, &err) ==
                         YVEX_ERR_FORMAT,
                     "malformed draft distribution refuses");
    request.draft_probabilities = target;
    YVEX_TEST_ASSERT(yvex_speculation_accept(&request, committed, 1ull, &result, &err) ==
                         YVEX_ERR_INVALID_ARG,
                     "undersized commit directory refuses");
    {
        const float zero_draft[] = {0.0f, 1.0f};
        committed[0] = UINT32_MAX;
        committed[1] = UINT32_MAX;
        request.draft_probabilities = zero_draft;
        request.candidate_token_ids = candidates;
        YVEX_TEST_ASSERT(
            yvex_speculation_accept(&request, committed, 2ull, &result, &err) ==
                    YVEX_ERR_FORMAT &&
                committed[0] == UINT32_MAX && committed[1] == UINT32_MAX,
            "a candidate outside the draft distribution refuses before output");
    }
    request.draft_probabilities = target;
    request.distribution_stride = SIZE_MAX;
    YVEX_TEST_ASSERT(
        yvex_speculation_accept(&request, committed, 2ull, &result, &err) ==
            YVEX_ERR_INVALID_ARG,
        "overflowing probability geometry refuses");
    request.distribution_stride = 2ull;
    {
        const double invalid_draws[] = {NAN};
        request.acceptance_uniforms = invalid_draws;
        YVEX_TEST_ASSERT(
            yvex_speculation_accept(&request, committed, 2ull, &result, &err) ==
                YVEX_ERR_FORMAT,
            "non-finite acceptance draw refuses");
    }
    return 0;
}

static int test_candidate_extent(void)
{
    unsigned long long count = 99ull;
    yvex_error err;

    YVEX_TEST_ASSERT(
        yvex_speculation_candidate_extent(5ull, 12ull, 20ull, &count, &err) ==
                YVEX_OK &&
            count == 5ull,
        "full DSpark extent keeps the admitted block");
    YVEX_TEST_ASSERT(
        yvex_speculation_candidate_extent(5ull, 4ull, 20ull, &count, &err) ==
                YVEX_OK &&
            count == 2ull,
        "output budget reserves anchor and target bonus");
    YVEX_TEST_ASSERT(
        yvex_speculation_candidate_extent(5ull, 20ull, 3ull, &count, &err) ==
                YVEX_OK &&
            count == 1ull,
        "context budget reserves anchor and target bonus");
    YVEX_TEST_ASSERT(
        yvex_speculation_candidate_extent(5ull, 2ull, 20ull, &count, &err) ==
                YVEX_OK &&
            count == 0ull,
        "two output positions use the bounded anchor-only path");
    YVEX_TEST_ASSERT(
        yvex_speculation_candidate_extent(5ull, 20ull, 2ull, &count, &err) ==
                YVEX_OK &&
            count == 0ull,
        "two context positions use the bounded anchor-only path");
    YVEX_TEST_ASSERT(
        yvex_speculation_candidate_extent(
            YVEX_SPECULATION_MAX_BLOCK + 1ull, 20ull, 20ull, &count, &err) ==
                YVEX_ERR_INVALID_ARG &&
            count == 0ull,
        "candidate extent refuses an unsupported policy block");
    return 0;
}

static int test_commit_plan(void)
{
    yvex_speculation_acceptance_result acceptance = {
        .schema_version = YVEX_SPECULATION_SCHEMA_V1,
        .accepted_draft_count = 3ull,
        .committed_count = 4ull};
    yvex_speculation_commit_plan plan;
    yvex_error err;

    YVEX_TEST_ASSERT(
        yvex_speculation_commit_plan_build(&acceptance, 5ull, &plan, &err) ==
                YVEX_OK &&
            plan.state_prefix_count == 5ull &&
            plan.publication_token_count == 5ull && !plan.terminal,
        "all-accepted verification commits the target bonus atomically");
    acceptance.accepted_draft_count = 1ull;
    acceptance.committed_count = 2ull;
    YVEX_TEST_ASSERT(
        yvex_speculation_commit_plan_build(&acceptance, 3ull, &plan, &err) ==
                YVEX_OK &&
            plan.state_prefix_count == 3ull &&
            plan.publication_token_count == 3ull,
        "rejection commits the target correction with the admitted prefix");
    YVEX_TEST_ASSERT(
        yvex_speculation_commit_plan_build(&acceptance, 1ull, &plan, &err) ==
                YVEX_OK &&
            plan.state_prefix_count == 1ull &&
            plan.publication_token_count == 2ull && plan.terminal,
        "a terminal draft token joins publication but stays outside persistent target state");
    YVEX_TEST_ASSERT(
        yvex_speculation_commit_plan_build(&acceptance, 0ull, &plan, &err) ==
            YVEX_ERR_STATE,
        "a terminal conditioning token refuses the verification commit path");
    return 0;
}

static int test_family_policy_compatibility(void)
{
    const yvex_runtime_family_adapter *adapter =
        yvex_runtime_family_adapter_find("deepseek4-v4-flash-dspark");
    yvex_runtime_descriptor_summary runtime = {0};
    yvex_speculation_family_policy policy;

    YVEX_TEST_ASSERT(adapter && adapter->speculation_policy &&
                         !adapter->speculation_policy(&runtime, &policy),
                     "DSpark policy refuses an uncompiled execution descriptor");
    runtime.model_execution.schema_version = YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1;
    runtime.model_execution.proposal_width = 4ull;
    runtime.model_execution.draft_noise_token_id = 17ull;
    runtime.model_execution.target_feature_count = 2ull;
    runtime.model_execution.target_feature_layers[0] = 7ull;
    runtime.model_execution.target_feature_layers[1] = 9ull;
    runtime.model_execution.target_feature_width = 16ull;
    runtime.model_execution.draft_layer_count = 2ull;
    runtime.model_execution.markov_rank = 8ull;
    YVEX_TEST_ASSERT(adapter->speculation_policy(&runtime, &policy) &&
                         policy.block_size == 4ull && policy.noise_token_id == 17ull &&
                         policy.target_feature_layer_count == 2ull &&
                         policy.target_feature_layers[0] == 7ull &&
                         policy.target_feature_layers[1] == 9ull &&
                         policy.concatenated_feature_width == 32ull,
                     "binding v8 DSpark policy consumes model-derived geometry");
    return 0;
}

int yvex_test_runtime_speculation(void)
{
    if (test_greedy_acceptance() != 0) return 1;
    if (test_greedy_rejection() != 0) return 1;
    if (test_stochastic_acceptance() != 0) return 1;
    if (test_stochastic_bonus() != 0) return 1;
    if (test_stochastic_residual_fallback() != 0) return 1;
    if (test_stochastic_tiny_draft_mass() != 0) return 1;
    if (test_stochastic_reference_matrix() != 0) return 1;
    if (test_refusals() != 0) return 1;
    if (test_candidate_extent() != 0) return 1;
    if (test_commit_plan() != 0) return 1;
    return test_family_policy_compatibility();
}
