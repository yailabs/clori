/* Speculative acceptance is an independent numerical oracle over explicit target/draft facts. */
#include <yvex/internal/decode.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

static int acceptance_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.speculation", reason);
    return status;
}

static int acceptance_hash_finish(
    yvex_sha256 *hash, char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_speculation_distribution_valid(const float *row, unsigned long long count)
{
    double sum = 0.0;
    unsigned long long index;
    if (!row || !count) return 0;
    for (index = 0ull; index < count; ++index) {
        if (!isfinite(row[index]) || row[index] < 0.0f || row[index] > 1.0f) return 0;
        sum += row[index];
    }
    return fabs(sum - 1.0) <= 1e-5;
}

unsigned int yvex_speculation_distribution_sample(
    const float *target, const float *draft, unsigned long long count,
    double uniform, int residual)
{
    double total = 0.0, cumulative = 0.0;
    unsigned long long index;
    for (index = 0ull; index < count; ++index)
        total += residual ? fmax((double)target[index] - draft[index], 0.0)
                          : (double)target[index];
    /* Rounded-away residual mass falls back to the target distribution rather than failing. */
    if (residual && total <= 1e-8) {
        residual = 0;
        total = 0.0;
        for (index = 0ull; index < count; ++index) total += target[index];
    }
    if (total <= 0.0) return UINT32_MAX;
    for (index = 0ull; index < count; ++index) {
        double value = residual ? fmax((double)target[index] - draft[index], 0.0)
                                : (double)target[index];
        cumulative += value / total;
        if (uniform < cumulative || index + 1ull == count) return (unsigned int)index;
    }
    return UINT32_MAX;
}

static int acceptance_policy_identity(const yvex_speculation_acceptance_request *request,
                                      char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.speculation.policy.fixed.v1") &&
           yvex_sha256_update_u64(&hash, request->schema_version) &&
           yvex_sha256_update_u64(&hash, request->kind) &&
           yvex_sha256_update_u64(&hash, request->candidate_count) &&
           yvex_sha256_update_u64(&hash, request->vocabulary_size) &&
           acceptance_hash_finish(&hash, output);
}

static int acceptance_identity(
    const unsigned int *committed, const yvex_speculation_acceptance_result *result,
    char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.speculation.acceptance.v1") ||
        !yvex_sha256_update_text(&hash, result->policy_identity) ||
        !yvex_sha256_update_u64(&hash, result->accepted_draft_count) ||
        !yvex_sha256_update_u64(&hash, result->rejected_draft_count) ||
        !yvex_sha256_update_u64(&hash, result->committed_count) ||
        !yvex_sha256_update_u64(&hash, result->rejection_index) ||
        !yvex_sha256_update_u64(&hash, result->correction_present) ||
        !yvex_sha256_update_u64(&hash, result->bonus_present)) return 0;
    /* Rejected suffixes remain cycle evidence but cannot alter committed-state identity. */
    for (index = 0ull; index < result->committed_count; ++index)
        if (!yvex_sha256_update_u64(&hash, committed[index])) return 0;
    return acceptance_hash_finish(&hash, output);
}

static int acceptance_validate(const yvex_speculation_acceptance_request *request,
                               unsigned long long committed_capacity,
                               yvex_error *err)
{
    unsigned long long row;
    if (!request || request->schema_version != YVEX_SPECULATION_SCHEMA_V1 ||
        request->kind > YVEX_SPECULATION_ACCEPT_STOCHASTIC ||
        !request->candidate_count || request->candidate_count > YVEX_SPECULATION_MAX_BLOCK ||
        !request->vocabulary_size || request->vocabulary_size > UINT32_MAX ||
        request->distribution_stride < request->vocabulary_size ||
        request->distribution_stride > SIZE_MAX / sizeof(float) ||
        request->candidate_count + 1ull > SIZE_MAX / request->distribution_stride ||
        !request->candidate_token_ids || committed_capacity < request->candidate_count + 1ull)
        return acceptance_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "speculative acceptance geometry is incomplete");
    for (row = 0ull; row < request->candidate_count; ++row)
        if (request->candidate_token_ids[row] >= request->vocabulary_size)
            return acceptance_refuse(err, YVEX_ERR_BOUNDS,
                                     "candidate token is outside the vocabulary");
    if (request->kind == YVEX_SPECULATION_ACCEPT_GREEDY) {
        if (!request->target_token_ids)
            return acceptance_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "greedy acceptance requires target token IDs");
        for (row = 0ull; row <= request->candidate_count; ++row)
            if (request->target_token_ids[row] >= request->vocabulary_size)
                return acceptance_refuse(err, YVEX_ERR_BOUNDS,
                                         "target token is outside the vocabulary");
        return YVEX_OK;
    }
    if (!request->draft_probabilities || !request->target_probabilities ||
        !request->acceptance_uniforms || !isfinite(request->correction_uniform) ||
        request->correction_uniform < 0.0 || request->correction_uniform >= 1.0)
        return acceptance_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "stochastic acceptance requires distributions and draws");
    for (row = 0ull; row < request->candidate_count; ++row) {
        unsigned int candidate = request->candidate_token_ids[row];
        if (!isfinite(request->acceptance_uniforms[row]) ||
            request->acceptance_uniforms[row] < 0.0 ||
            request->acceptance_uniforms[row] >= 1.0 ||
            !yvex_speculation_distribution_valid(
                request->draft_probabilities + row * request->distribution_stride,
                request->vocabulary_size) ||
            !yvex_speculation_distribution_valid(
                request->target_probabilities + row * request->distribution_stride,
                request->vocabulary_size) ||
            request->draft_probabilities[
                row * request->distribution_stride + candidate] <= 0.0f)
            return acceptance_refuse(err, YVEX_ERR_FORMAT,
                                     "speculative probability row is malformed");
    }
    if (!yvex_speculation_distribution_valid(
            request->target_probabilities +
                request->candidate_count * request->distribution_stride,
            request->vocabulary_size))
        return acceptance_refuse(err, YVEX_ERR_FORMAT,
                                 "target bonus distribution is malformed");
    return YVEX_OK;
}

int yvex_speculation_accept(
    const yvex_speculation_acceptance_request *request,
    unsigned int *committed_token_ids, unsigned long long committed_capacity,
    yvex_speculation_acceptance_result *result, yvex_error *err)
{
    unsigned long long index;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!committed_token_ids || !result)
        return acceptance_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "speculative acceptance outputs are required");
    rc = acceptance_validate(request, committed_capacity, err);
    if (rc != YVEX_OK) return rc;
    result->schema_version = YVEX_SPECULATION_SCHEMA_V1;
    result->kind = request->kind;
    result->proposed_count = request->candidate_count;
    result->rejection_index = request->candidate_count;
    if (!acceptance_policy_identity(request, result->policy_identity))
        return acceptance_refuse(err, YVEX_ERR_STATE,
                                 "speculative policy identity derivation failed");
    for (index = 0ull; index < request->candidate_count; ++index) {
        unsigned int candidate = request->candidate_token_ids[index];
        int accepted;
        if (request->kind == YVEX_SPECULATION_ACCEPT_GREEDY)
            accepted = candidate == request->target_token_ids[index];
        else {
            const float *draft = request->draft_probabilities +
                                 index * request->distribution_stride;
            const float *target = request->target_probabilities +
                                  index * request->distribution_stride;
            accepted = request->acceptance_uniforms[index] <
                       fmin(1.0, (double)target[candidate] / draft[candidate]);
        }
        if (!accepted) {
            unsigned int correction;
            result->rejection_index = index;
            result->rejected_draft_count = request->candidate_count - index;
            correction = request->kind == YVEX_SPECULATION_ACCEPT_GREEDY
                ? request->target_token_ids[index]
                : yvex_speculation_distribution_sample(
                      request->target_probabilities + index * request->distribution_stride,
                      request->draft_probabilities + index * request->distribution_stride,
                      request->vocabulary_size, request->correction_uniform, 1);
            if (correction == UINT32_MAX)
                return acceptance_refuse(err, YVEX_ERR_FORMAT,
                                         "residual target distribution is empty");
            committed_token_ids[result->committed_count++] = correction;
            result->correction_present = 1;
            result->correction_or_bonus_token_id = correction;
            break;
        }
        committed_token_ids[result->committed_count++] = candidate;
        result->accepted_draft_count++;
    }
    if (result->accepted_draft_count == request->candidate_count) {
        unsigned int bonus = request->kind == YVEX_SPECULATION_ACCEPT_GREEDY
            ? request->target_token_ids[request->candidate_count]
            : yvex_speculation_distribution_sample(
                  request->target_probabilities +
                      request->candidate_count * request->distribution_stride,
                  NULL, request->vocabulary_size, request->correction_uniform, 0);
        if (bonus == UINT32_MAX)
            return acceptance_refuse(err, YVEX_ERR_FORMAT,
                                     "target bonus distribution is empty");
        committed_token_ids[result->committed_count++] = bonus;
        result->all_candidates_accepted = 1;
        result->bonus_present = 1;
        result->correction_or_bonus_token_id = bonus;
    }
    if (!acceptance_identity(committed_token_ids, result, result->acceptance_identity))
        return acceptance_refuse(err, YVEX_ERR_STATE,
                                 "speculative acceptance identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_speculation_candidate_extent(
    unsigned long long policy_block_size,
    unsigned long long remaining_output_tokens,
    unsigned long long remaining_context_tokens,
    unsigned long long *candidate_count, yvex_error *err)
{
    unsigned long long output_limit, context_limit, admitted;
    if (candidate_count) *candidate_count = 0ull;
    if (!candidate_count || !policy_block_size ||
        policy_block_size > YVEX_SPECULATION_MAX_BLOCK ||
        !remaining_output_tokens || !remaining_context_tokens)
        return acceptance_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "speculative candidate extent requires bounded output and context");
    /* Reserve anchor and bonus before drafting so full acceptance cannot overrun capacity. */
    if (remaining_output_tokens <= 2ull || remaining_context_tokens <= 2ull) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    output_limit = remaining_output_tokens - 2ull;
    context_limit = remaining_context_tokens - 2ull;
    admitted = policy_block_size;
    if (admitted > output_limit) admitted = output_limit;
    if (admitted > context_limit) admitted = context_limit;
    *candidate_count = admitted;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_speculation_commit_plan_build(
    const yvex_speculation_acceptance_result *acceptance,
    unsigned long long terminal_index,
    yvex_speculation_commit_plan *plan, yvex_error *err)
{
    unsigned long long output_count;
    if (plan) memset(plan, 0, sizeof(*plan));
    if (!acceptance || !plan ||
        acceptance->schema_version != YVEX_SPECULATION_SCHEMA_V1 ||
        !acceptance->committed_count ||
        acceptance->committed_count != acceptance->accepted_draft_count + 1ull ||
        !yvex_core_u64_add(acceptance->committed_count, 1ull, &output_count) ||
        terminal_index > output_count)
        return acceptance_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "speculative commit plan is malformed");
    /* Correction or bonus remains in the atomic result so RNG cannot outrun durable state. */
    if (terminal_index < output_count) {
        if (!terminal_index)
            return acceptance_refuse(
                err, YVEX_ERR_STATE,
                "a terminal conditioning token cannot enter verification");
        plan->state_prefix_count = terminal_index;
        plan->publication_token_count = terminal_index + 1ull;
        plan->terminal = 1;
    } else {
        plan->state_prefix_count = output_count;
        plan->publication_token_count = output_count;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}
