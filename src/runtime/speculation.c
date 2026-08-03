/*
 * Speculative candidates remain untrusted until this owner compares them with complete target
 * distributions. The result describes only the prefix that a caller may commit; it never mutates
 * model, RNG, tokenizer, transcript, or session state.
 */
#include <yvex/internal/decode.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/artifact.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/sampling.h>
#include <yvex/internal/transformer.h>

typedef struct {
    const yvex_materialized_tensor_binding *binding;
    const unsigned char *encoded;
    unsigned long long encoded_bytes, row_bytes;
} speculation_weight;

struct yvex_runtime_speculation_context {
    yvex_runtime_model *model;
    yvex_runtime_execution_session *session;
    const yvex_runtime_model_view *model_view;
    yvex_runtime_transformer_context *target_transformer;
    yvex_runtime_transformer_context *draft_transformer;
    yvex_runtime_logits_context *target_logits;
    yvex_runtime_sampling_context *target_sampling;
    yvex_runtime_sampling_context *draft_sampling;
    yvex_runtime_sampling_transaction *target_rng;
    yvex_runtime_sampling_transaction *draft_rng;
    yvex_runtime_sampling_policy sampling_policy;
    yvex_runtime_speculation_options options;
    yvex_speculation_family_policy policy;
    speculation_weight feature_projection, feature_norm, markov_embedding;
    speculation_weight markov_output, confidence;
    float *feature_projected, *feature_norm_weights, *target_features;
    float *draft_hidden, *draft_pre_normalized, *target_hidden;
    float *base_logits, *adjusted_logits, *markov_bias;
    float *draft_probabilities, *target_probabilities, *markov_embedding_values;
    unsigned int *draft_input_ids;
    unsigned long long vocabulary_size, hidden_width, workspace_bytes;
    unsigned long long pending_position, pending_committed_count;
    unsigned int pending_tokens[YVEX_SPECULATION_MAX_BLOCK + 2u];
    char pending_source_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char pending_sampling_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char pending_cycle_identity[YVEX_SPECULATION_IDENTITY_CAP];
    const yvex_runtime_commit_participant *publication;
    int publication_prepared, cycle_pending;
};

static void speculation_pending_clear(yvex_runtime_speculation_context *context);

static int speculation_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.speculation", reason);
    return status;
}

static int speculation_hash_finish(
    yvex_sha256 *hash, char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int speculation_probability_row(const float *row, unsigned long long count)
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

static unsigned int speculation_sample(const float *target, const float *draft,
                                       unsigned long long count, double uniform,
                                       int residual)
{
    double total = 0.0, cumulative = 0.0;
    unsigned long long index;
    for (index = 0ull; index < count; ++index)
        total += residual ? fmax((double)target[index] - draft[index], 0.0)
                          : (double)target[index];
    /* DeepSpec falls back to the target distribution when float rounding
     * collapses the residual mass. This preserves a valid correction draw
     * instead of turning a numerically degenerate rejection into a runtime
     * failure. */
    if (residual && total <= 1e-8) {
        residual = 0;
        total = 0.0;
        for (index = 0ull; index < count; ++index)
            total += target[index];
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

static int speculation_policy_identity(const yvex_speculation_acceptance_request *request,
                                       char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.speculation.policy.fixed.v1") &&
           yvex_sha256_update_u64(&hash, request->schema_version) &&
           yvex_sha256_update_u64(&hash, request->kind) &&
           yvex_sha256_update_u64(&hash, request->candidate_count) &&
           yvex_sha256_update_u64(&hash, request->vocabulary_size) &&
           speculation_hash_finish(&hash, output);
}

static int speculation_acceptance_identity(
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
    /* This identity follows the admitted result, not the diagnostic proposal.
     * Rejected suffix tokens remain visible through the cycle evidence but
     * cannot perturb identities attached to committed model state or text. */
    for (index = 0ull; index < result->committed_count; ++index)
        if (!yvex_sha256_update_u64(&hash, committed[index])) return 0;
    return speculation_hash_finish(&hash, output);
}

static int speculation_validate(const yvex_speculation_acceptance_request *request,
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
        request->candidate_count + 1ull >
            SIZE_MAX / request->distribution_stride ||
        !request->candidate_token_ids || committed_capacity < request->candidate_count + 1ull)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "speculative acceptance geometry is incomplete");
    for (row = 0ull; row < request->candidate_count; ++row)
        if (request->candidate_token_ids[row] >= request->vocabulary_size)
            return speculation_refuse(err, YVEX_ERR_BOUNDS,
                                      "candidate token is outside the vocabulary");
    if (request->kind == YVEX_SPECULATION_ACCEPT_GREEDY) {
        if (!request->target_token_ids)
            return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                      "greedy acceptance requires target token IDs");
        for (row = 0ull; row <= request->candidate_count; ++row)
            if (request->target_token_ids[row] >= request->vocabulary_size)
                return speculation_refuse(err, YVEX_ERR_BOUNDS,
                                          "target token is outside the vocabulary");
        return YVEX_OK;
    }
    if (!request->draft_probabilities || !request->target_probabilities ||
        !request->acceptance_uniforms || !isfinite(request->correction_uniform) ||
        request->correction_uniform < 0.0 ||
        request->correction_uniform >= 1.0)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "stochastic acceptance requires distributions and draws");
    for (row = 0ull; row < request->candidate_count; ++row) {
        unsigned int candidate = request->candidate_token_ids[row];
        if (!isfinite(request->acceptance_uniforms[row]) ||
            request->acceptance_uniforms[row] < 0.0 ||
            request->acceptance_uniforms[row] >= 1.0 ||
            !speculation_probability_row(
                request->draft_probabilities + row * request->distribution_stride,
                request->vocabulary_size) ||
            !speculation_probability_row(
                request->target_probabilities + row * request->distribution_stride,
                request->vocabulary_size) ||
            request->draft_probabilities[
                row * request->distribution_stride + candidate] <= 0.0f)
            return speculation_refuse(err, YVEX_ERR_FORMAT,
                                      "speculative probability row is malformed");
    }
    if (!speculation_probability_row(
            request->target_probabilities +
                request->candidate_count * request->distribution_stride,
            request->vocabulary_size))
        return speculation_refuse(err, YVEX_ERR_FORMAT,
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
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "speculative acceptance outputs are required");
    rc = speculation_validate(request, committed_capacity, err);
    if (rc != YVEX_OK) return rc;
    result->schema_version = YVEX_SPECULATION_SCHEMA_V1;
    result->kind = request->kind;
    result->proposed_count = request->candidate_count;
    result->rejection_index = request->candidate_count;
    if (!speculation_policy_identity(request, result->policy_identity))
        return speculation_refuse(err, YVEX_ERR_STATE,
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
            double q = draft[candidate];
            double p = target[candidate];
            accepted = request->acceptance_uniforms[index] < fmin(1.0, p / q);
        }
        if (!accepted) {
            unsigned int correction;
            result->rejection_index = index;
            result->rejected_draft_count = request->candidate_count - index;
            if (request->kind == YVEX_SPECULATION_ACCEPT_GREEDY)
                correction = request->target_token_ids[index];
            else
                correction = speculation_sample(
                    request->target_probabilities + index * request->distribution_stride,
                    request->draft_probabilities + index * request->distribution_stride,
                    request->vocabulary_size, request->correction_uniform, 1);
            if (correction == UINT32_MAX)
                return speculation_refuse(err, YVEX_ERR_FORMAT,
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
                                 : speculation_sample(
                                       request->target_probabilities +
                                           request->candidate_count *
                                               request->distribution_stride,
                                       NULL, request->vocabulary_size,
                                       request->correction_uniform, 0);
        if (bonus == UINT32_MAX)
            return speculation_refuse(err, YVEX_ERR_FORMAT,
                                      "target bonus distribution is empty");
        committed_token_ids[result->committed_count++] = bonus;
        result->all_candidates_accepted = 1;
        result->bonus_present = 1;
        result->correction_or_bonus_token_id = bonus;
    }
    if (!speculation_acceptance_identity(committed_token_ids, result,
                                         result->acceptance_identity))
        return speculation_refuse(err, YVEX_ERR_STATE,
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
        return speculation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "speculative candidate extent requires bounded output and context");

    /* A surviving block always adds one target bonus after the target-authored
     * anchor. Reserve both positions before drafting so an all-accepted cycle
     * cannot overrun the caller's token array or the session context. */
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
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "speculative commit plan is malformed");
    /* Every non-terminal token selected by target verification belongs to the
     * same atomic result, including its correction or bonus. Keeping that last
     * token outside the commit would advance stochastic RNG state past the
     * durable model/token prefix when a later draft is cancelled. */
    if (terminal_index < output_count) {
        if (!terminal_index)
            return speculation_refuse(
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

static int speculation_draft_sampling_policy(
    const yvex_runtime_sampling_policy *target_policy,
    unsigned long long vocabulary_size,
    yvex_runtime_sampling_policy *draft_policy, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int index;
    unsigned long long seed = 0ull;

    if (draft_policy) memset(draft_policy, 0, sizeof(*draft_policy));
    if (!target_policy || !draft_policy || !vocabulary_size ||
        !yvex_sha256_hex_valid(target_policy->policy_identity))
        return speculation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "draft sampling requires one sealed target policy");
    *draft_policy = *target_policy;
    if (target_policy->strategy == YVEX_SAMPLING_STRATEGY_STOCHASTIC) {
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update_text(
                &hash, "yvex.speculation.draft-rng-domain.v1") ||
            !yvex_sha256_update_text(&hash, target_policy->policy_identity) ||
            !yvex_sha256_final(&hash, digest))
            return speculation_refuse(
                err, YVEX_ERR_STATE,
                "draft sampling seed derivation failed");
        for (index = 0u; index < sizeof(seed); ++index)
            seed = (seed << 8u) | digest[index];
        draft_policy->seed = seed;
    }
    draft_policy->policy_identity[0] = '\0';
    return yvex_runtime_sampling_policy_seal(
        draft_policy, vocabulary_size, err);
}

static int speculation_values_digest(const char *domain, const float *values,
                                     unsigned long long count,
                                     char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!domain || !values || !count || !yvex_sha256_update_text(&hash, domain))
        return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        if (!isfinite(values[index])) return 0;
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int speculation_weight_bind(yvex_runtime_speculation_context *context,
                                   yvex_tensor_role role,
                                   speculation_weight *out, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *summary =
        yvex_runtime_descriptor_summary_get(context->model_view->descriptor);
    const yvex_runtime_tensor_binding *match = NULL;
    unsigned long long index, bytes = 0ull;
    if (!summary || !out)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "speculation weight owner is unavailable");
    for (index = 0ull; index < summary->tensor_count; ++index) {
        const yvex_runtime_tensor_binding *row =
            yvex_runtime_descriptor_tensor_at(context->model_view->descriptor,
                                              index);
        if (row && row->role == role && row->scope == YVEX_TENSOR_SCOPE_DRAFT) {
            if (match)
                return speculation_refuse(err, YVEX_ERR_FORMAT,
                                          "speculation role has duplicate bindings");
            match = row;
        }
    }
    if (!match || !match->binding || !match->binding->row_count ||
        match->binding->encoded_bytes % match->binding->row_count ||
        yvex_runtime_residency_binding_view(
            context->model_view->residency, match->binding, &out->encoded,
            &bytes, err) != YVEX_OK || bytes != match->binding->encoded_bytes)
        return speculation_refuse(err, YVEX_ERR_STATE,
                                  "speculation role is not resident and exact");
    out->binding = match->binding;
    out->encoded_bytes = bytes;
    out->row_bytes = bytes / match->binding->row_count;
    return YVEX_OK;
}

static int speculation_weight_decode_row(const speculation_weight *weight,
                                         unsigned long long row, float *values,
                                         unsigned long long capacity,
                                         yvex_error *err)
{
    const yvex_gguf_qtype_geometry *geometry;
    yvex_quant_failure failure;
    unsigned long long blocks, block;
    if (!weight || !weight->binding || !values ||
        row >= weight->binding->row_count ||
        capacity < weight->binding->row_width)
        return speculation_refuse(err, YVEX_ERR_BOUNDS,
                                  "speculation decoded row capacity is invalid");
    geometry = yvex_gguf_qtype_geometry_find(weight->binding->qtype);
    if (!geometry || !geometry->block_size ||
        weight->binding->row_width % geometry->block_size ||
        (blocks = weight->binding->row_width / geometry->block_size) == 0ull ||
        blocks * geometry->bytes_per_block != weight->row_bytes)
        return speculation_refuse(err, YVEX_ERR_FORMAT,
                                  "speculation weight row is not block exact");
    for (block = 0ull; block < blocks; ++block) {
        memset(&failure, 0, sizeof(failure));
        if (yvex_quant_decode_block(
                weight->binding->qtype,
                weight->encoded + row * weight->row_bytes +
                    block * geometry->bytes_per_block,
                geometry->bytes_per_block,
                values + block * geometry->block_size,
                geometry->block_size, &failure, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return YVEX_OK;
}

static int speculation_context_geometry(yvex_runtime_speculation_context *context,
                                        yvex_error *err)
{
    const yvex_runtime_logits_plan_summary *logits =
        yvex_runtime_logits_plan_summary_get(context->target_logits);
    if (!logits || logits->vocabulary_size != context->vocabulary_size ||
        logits->hidden_width != context->hidden_width ||
        context->feature_projection.binding->row_width !=
            context->policy.concatenated_feature_width ||
        context->feature_projection.binding->row_count != context->hidden_width ||
        context->feature_norm.binding->row_width != context->hidden_width ||
        context->feature_norm.binding->row_count != 1ull ||
        context->markov_embedding.binding->row_count != context->vocabulary_size ||
        context->markov_embedding.binding->row_width != context->policy.markov_rank ||
        context->markov_output.binding->row_count != context->vocabulary_size ||
        context->markov_output.binding->row_width != context->policy.markov_rank ||
        context->confidence.binding->row_count != 1ull ||
        context->confidence.binding->row_width !=
            context->hidden_width + context->policy.markov_rank)
        return speculation_refuse(err, YVEX_ERR_FORMAT,
                                  "DSpark runtime weight geometry is inconsistent");
    return YVEX_OK;
}

static int speculation_context_buffers(yvex_runtime_speculation_context *context,
                                       yvex_error *err)
{
    unsigned long long block = context->policy.block_size;
    unsigned long long hidden_rows, target_hidden_rows, feature_rows;
    unsigned long long draft_rows, target_rows, probabilities;
    unsigned long long markov_values, total = 0ull;
#define SPEC_ADD(count_) \
    do { \
        unsigned long long bytes_; \
        if (!yvex_core_u64_mul((count_), sizeof(float), &bytes_) || \
            !yvex_core_u64_add(total, bytes_, &total)) goto overflow; \
    } while (0)
    if (!yvex_core_u64_mul(block, context->hidden_width, &hidden_rows) ||
        !yvex_core_u64_mul(block + 2ull, context->hidden_width,
                           &target_hidden_rows) ||
        !yvex_core_u64_mul(block + 2ull,
                           context->policy.concatenated_feature_width,
                           &feature_rows) ||
        !yvex_core_u64_mul(block, context->vocabulary_size, &draft_rows) ||
        !yvex_core_u64_mul(block + 1ull, context->vocabulary_size,
                           &target_rows) ||
        !yvex_core_u64_add(draft_rows, target_rows, &probabilities) ||
        !yvex_core_u64_mul(block, context->policy.markov_rank,
                           &markov_values))
        goto overflow;
    SPEC_ADD(target_hidden_rows);
    SPEC_ADD(context->hidden_width);
    SPEC_ADD(feature_rows);
    SPEC_ADD(hidden_rows);
    SPEC_ADD(hidden_rows);
    SPEC_ADD(target_hidden_rows);
    SPEC_ADD(draft_rows);
    SPEC_ADD(context->vocabulary_size);
    SPEC_ADD(context->vocabulary_size);
    SPEC_ADD(probabilities);
    SPEC_ADD(markov_values);
    if (!yvex_core_u64_add(total, block * sizeof(unsigned int), &total) ||
        total > SIZE_MAX ||
        (context->options.maximum_host_bytes &&
         total > context->options.maximum_host_bytes))
        goto overflow;
    context->feature_projected = yvex_core_calloc(
        (size_t)target_hidden_rows, sizeof(float));
    context->feature_norm_weights = yvex_core_calloc(
        (size_t)context->hidden_width, sizeof(float));
    context->target_features = yvex_core_calloc((size_t)feature_rows, sizeof(float));
    context->draft_hidden = yvex_core_calloc((size_t)hidden_rows, sizeof(float));
    context->draft_pre_normalized = yvex_core_calloc((size_t)hidden_rows, sizeof(float));
    context->target_hidden = yvex_core_calloc(
        (size_t)target_hidden_rows, sizeof(float));
    context->base_logits = yvex_core_calloc((size_t)draft_rows, sizeof(float));
    context->adjusted_logits = yvex_core_calloc(
        (size_t)context->vocabulary_size, sizeof(float));
    context->markov_bias = yvex_core_calloc(
        (size_t)context->vocabulary_size, sizeof(float));
    context->draft_probabilities = yvex_core_calloc((size_t)probabilities,
                                                    sizeof(float));
    context->target_probabilities = context->draft_probabilities + draft_rows;
    context->markov_embedding_values = yvex_core_calloc(
        (size_t)markov_values, sizeof(float));
    context->draft_input_ids = yvex_core_calloc((size_t)block,
                                                sizeof(unsigned int));
    if (!context->feature_projected || !context->feature_norm_weights ||
        !context->target_features ||
        !context->draft_hidden || !context->draft_pre_normalized ||
        !context->target_hidden || !context->base_logits ||
        !context->adjusted_logits || !context->markov_bias ||
        !context->draft_probabilities || !context->markov_embedding_values ||
        !context->draft_input_ids)
        return speculation_refuse(err, YVEX_ERR_NOMEM,
                                  "DSpark runtime workspace allocation failed");
    context->workspace_bytes = total;
    return speculation_weight_decode_row(&context->feature_norm, 0ull,
                                         context->feature_norm_weights,
                                         context->hidden_width, err);
overflow:
    return speculation_refuse(err, YVEX_ERR_BOUNDS,
                              "DSpark runtime workspace extent overflowed");
#undef SPEC_ADD
}

int yvex_runtime_speculation_context_open(
    yvex_runtime_speculation_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session,
    yvex_runtime_transformer_context *target_transformer,
    yvex_runtime_logits_context *target_logits,
    yvex_runtime_sampling_context *target_sampling,
    const yvex_runtime_sampling_policy *sampling_policy,
    const yvex_runtime_speculation_options *options, yvex_error *err)
{
    yvex_runtime_speculation_context *context = NULL;
    yvex_runtime_transformer_options transformer_options = {0};
    yvex_runtime_sampling_options sampling_options = {0};
    yvex_runtime_sampling_policy draft_policy = {0};
    const yvex_transformer_plan_summary *target_plan;
    const yvex_runtime_descriptor_summary *descriptor;
    unsigned long long draft_context_capacity;
    int rc;
    if (out) *out = NULL;
    if (!out || !model || !session || !target_transformer || !target_logits ||
        !target_sampling || !sampling_policy || !options ||
        (options->backend != YVEX_BACKEND_KIND_CPU &&
         options->backend != YVEX_BACKEND_KIND_CUDA) ||
        !options->context_capacity)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "complete DSpark runtime owners are required");
    context = yvex_core_calloc(1u, sizeof(*context));
    if (!context)
        return speculation_refuse(err, YVEX_ERR_NOMEM,
                                  "DSpark context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_runtime_model_view_get(model);
    context->target_transformer = target_transformer;
    context->target_logits = target_logits;
    context->target_sampling = target_sampling;
    context->sampling_policy = *sampling_policy;
    context->options = *options;
    descriptor = context->model_view
        ? yvex_runtime_descriptor_summary_get(context->model_view->descriptor) : NULL;
    target_plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(target_transformer));
    if (!context->model_view || !descriptor || !target_plan ||
        yvex_runtime_transformer_context_session(target_transformer) != session ||
        !context->model_view->adapter ||
        !context->model_view->adapter->speculation_policy ||
        !context->model_view->adapter->speculation_policy(&context->policy) ||
        context->policy.schema_version !=
            YVEX_SPECULATION_FAMILY_POLICY_SCHEMA_V1 ||
        !context->policy.target_verification_required ||
        !context->policy.parallel_block_backbone ||
        !context->policy.sequential_markov ||
        !context->policy.shares_output_head ||
        !context->model_view->draft_attention ||
        descriptor->draft_layer_count != context->policy.draft_layer_count ||
        target_plan->tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER ||
        context->policy.block_size < 2ull ||
        context->policy.block_size > YVEX_SPECULATION_MAX_BLOCK) {
        rc = speculation_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "runtime binding does not admit complete DSpark execution");
        (void)yvex_runtime_speculation_context_close(&context, NULL);
        return rc;
    }
    if (!target_plan->maximum_context ||
        options->context_capacity > target_plan->maximum_context ||
        !yvex_core_u64_add(options->context_capacity,
                           context->policy.block_size + 2ull,
                           &draft_context_capacity) ||
        draft_context_capacity > target_plan->maximum_context) {
        rc = speculation_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "DSpark requires bounded draft lookahead beyond the admitted target context");
        (void)yvex_runtime_speculation_context_close(&context, NULL);
        return rc;
    }
    context->vocabulary_size = target_plan->vocabulary_size;
    context->hidden_width = target_plan->hidden_width;
    transformer_options.maximum_host_bytes = options->maximum_host_bytes;
    transformer_options.maximum_device_bytes = options->maximum_device_bytes;
    /* The five draft queries attend one another, so a shortened backbone pass
     * would change even the first proposal. Reserve ephemeral lookahead beyond
     * the target-visible context; accepted target state remains bounded by the
     * caller's original context_capacity. */
    transformer_options.context_capacity = draft_context_capacity;
    transformer_options.workspace_token_capacity = context->policy.block_size + 2ull;
    transformer_options.tensor_scope = YVEX_TENSOR_SCOPE_DRAFT;
    transformer_options.cancel_requested = options->cancel_requested;
    transformer_options.cancel_context = options->cancel_context;
    transformer_options.evidence_level = YVEX_ATTENTION_EVIDENCE_NONE;
    rc = yvex_runtime_transformer_context_open(
        &context->draft_transformer, model, session, &transformer_options, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_admit_shared_draft_plan(
            target_logits,
            yvex_runtime_transformer_context_plan(context->draft_transformer), err);
    sampling_options.maximum_vocabulary_size = context->vocabulary_size;
    sampling_options.maximum_rows = context->policy.block_size;
    sampling_options.maximum_host_bytes = options->maximum_host_bytes;
    sampling_options.cancel_requested = options->cancel_requested;
    sampling_options.cancel_context = options->cancel_context;
    if (rc == YVEX_OK)
        rc = speculation_draft_sampling_policy(
            sampling_policy, context->vocabulary_size, &draft_policy, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_open(
            &context->draft_sampling,
            yvex_runtime_logits_plan_summary_get(target_logits), &draft_policy,
            &sampling_options, err);
#define BIND(role_, member_) \
    if (rc == YVEX_OK) rc = speculation_weight_bind(context, (role_), \
                                                     &context->member_, err)
    BIND(context->policy.feature_projection_role, feature_projection);
    BIND(context->policy.feature_norm_role, feature_norm);
    BIND(context->policy.markov_embedding_role, markov_embedding);
    BIND(context->policy.markov_output_role, markov_output);
    BIND(context->policy.confidence_role, confidence);
#undef BIND
    if (rc == YVEX_OK) rc = speculation_context_geometry(context, err);
    if (rc == YVEX_OK) rc = speculation_context_buffers(context, err);
    if (rc != YVEX_OK) {
        (void)yvex_runtime_speculation_context_close(&context, NULL);
        return rc;
    }
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
}

const yvex_speculation_family_policy *yvex_runtime_speculation_policy_get(
    const yvex_runtime_speculation_context *context)
{
    return context ? &context->policy : NULL;
}

static int speculation_project_target_features(
    yvex_runtime_speculation_context *context, const float *features,
    unsigned long long token_count, yvex_error *err)
{
    unsigned long long token, row;
    for (token = 0ull; token < token_count; ++token) {
        const float *source =
            features + token * context->policy.concatenated_feature_width;
        float *projected =
            context->feature_projected + token * context->hidden_width;
        for (row = 0ull; row < context->hidden_width; ++row) {
            yvex_quant_failure failure = {0};
            if (context->options.cancel_requested &&
                context->options.cancel_requested(context->options.cancel_context))
                return speculation_refuse(err, YVEX_ERR_CANCELLED,
                                          "DSpark feature projection was cancelled");
            if (yvex_quant_cpu_dot(
                    context->feature_projection.binding->qtype,
                    context->feature_projection.encoded +
                        row * context->feature_projection.row_bytes,
                    (size_t)context->feature_projection.row_bytes, source,
                    context->policy.concatenated_feature_width, &projected[row],
                    &failure, err) != YVEX_OK)
                return yvex_error_code(err);
        }
        if (yvex_transformer_feature_normalize(
                projected, context->hidden_width,
                context->feature_norm_weights,
                yvex_transformer_plan_summary_get(
                    yvex_runtime_transformer_context_plan(
                        context->draft_transformer))->output_norm_epsilon,
                err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return YVEX_OK;
}

static int speculation_weight_matvec(
    yvex_runtime_speculation_context *context,
    const speculation_weight *weight, const float *input, float *output,
    yvex_error *err)
{
    unsigned long long row;
    for (row = 0ull; row < weight->binding->row_count; ++row) {
        yvex_quant_failure failure = {0};
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context))
            return speculation_refuse(err, YVEX_ERR_CANCELLED,
                                      "DSpark matrix projection was cancelled");
        if (yvex_quant_cpu_dot(
                weight->binding->qtype,
                weight->encoded + row * weight->row_bytes,
                (size_t)weight->row_bytes, input, weight->binding->row_width,
                &output[row], &failure, err) != YVEX_OK)
            return yvex_error_code(err);
        if (!isfinite(output[row]))
            return speculation_refuse(err, YVEX_ERR_FORMAT,
                                      "DSpark matrix projection is non-finite");
    }
    return YVEX_OK;
}

static int speculation_transformer_execute(
    yvex_runtime_speculation_context *context,
    yvex_runtime_transformer_context *transformer,
    const unsigned int *token_ids, unsigned long long token_count,
    unsigned long long position, int candidate_block_visible,
    yvex_attention_transaction_disposition disposition,
    const unsigned long long *feature_layers,
    unsigned long long feature_layer_count, float *normalized,
    float *pre_normalized, float *features,
    yvex_runtime_transformer_result *result, yvex_error *err)
{
    const yvex_transformer_plan_summary *plan =
        yvex_transformer_plan_summary_get(
            yvex_runtime_transformer_context_plan(transformer));
    yvex_transformer_input_summary summary = {0};
    yvex_transformer_input *input = NULL;
    yvex_runtime_transformer_request request = {0};
    yvex_runtime_transformer_output output = {0};
    unsigned long long value_count, feature_count = 0ull;
    int rc;
    if (!plan || !token_ids || !token_count || !normalized || !result ||
        !yvex_core_u64_mul(token_count, plan->hidden_width, &value_count) ||
        (features && (!feature_layers || !feature_layer_count ||
                      !yvex_core_u64_mul(value_count, feature_layer_count,
                                         &feature_count))) ||
        (!features && feature_layer_count))
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "speculative transformer geometry is invalid");
    summary.schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
    summary.token_start = position;
    summary.token_count = token_count;
    summary.vocabulary_size = plan->vocabulary_size;
    yvex_runtime_identity_copy(summary.logical_model_identity,
                               plan->logical_model_identity);
    yvex_runtime_identity_copy(summary.runtime_numeric_identity,
                               plan->runtime_numeric_identity);
    yvex_runtime_identity_copy(summary.runtime_descriptor_identity,
                               plan->runtime_descriptor_identity);
    yvex_runtime_identity_copy(summary.transformer_plan_identity,
                               plan->transformer_plan_identity);
    rc = yvex_transformer_input_seal(&summary, token_ids, err);
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_memory(&input, &summary, token_ids, err);
    request.chunk_tokens = token_count;
    request.backend = context->options.backend;
    request.phase = YVEX_TRANSFORMER_PHASE_PREFILL;
    request.transaction_disposition = disposition;
    request.candidate_block_visible = candidate_block_visible;
    request.feature_layer_ordinals = feature_layers;
    request.feature_layer_count = feature_layer_count;
    output.normalized_hidden = normalized;
    output.capacity = value_count;
    output.pre_normalized_hidden = pre_normalized;
    output.pre_normalized_capacity = pre_normalized ? value_count : 0ull;
    output.features = features;
    output.feature_capacity = feature_count;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_execute(
            transformer, input, &request, &output, result, err);
    yvex_transformer_input_close(&input);
    if (rc == YVEX_OK && disposition == YVEX_ATTENTION_TRANSACTION_ABORT &&
        (result->position_before != position ||
         result->position_after != position ||
         result->generation_after != result->generation_before))
        rc = speculation_refuse(
            err, YVEX_ERR_STATE,
            "speculative transformer candidate state escaped abort");
    return rc;
}

static int speculation_project_draft_base(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_transformer_result *draft,
    yvex_runtime_logits_row_result *rows, unsigned long long count,
    yvex_error *err)
{
    const yvex_transformer_plan *plan =
        yvex_runtime_transformer_context_plan(context->draft_transformer);
    unsigned long long row;
    for (row = 0ull; row < count; ++row) {
        yvex_runtime_logits_source source;
        int rc = yvex_runtime_logits_source_from_draft(
            context->target_logits, &source, plan, draft,
            context->draft_hidden, count * context->hidden_width, row, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_logits_project(
                context->target_logits, &source, context->options.backend,
                context->base_logits + row * context->vocabulary_size,
                context->vocabulary_size, &rows[row], err);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

static unsigned int speculation_argmax(const float *values,
                                       unsigned long long count);

static int speculation_draft_one(
    yvex_runtime_speculation_context *context, unsigned long long ordinal,
    unsigned int previous_token,
    const yvex_runtime_logits_row_result *base_result,
    double uniform, unsigned int *selected, float *confidence, yvex_error *err)
{
    float *markov = context->markov_embedding_values +
                    ordinal * context->policy.markov_rank;
    float *confidence_input = context->feature_projected;
    yvex_runtime_logits_row_result adjusted;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_distribution_result distribution;
    unsigned int token = UINT32_MAX;
    int rc = speculation_weight_decode_row(
        &context->markov_embedding, previous_token, markov,
        context->policy.markov_rank, err);
    if (rc == YVEX_OK)
        rc = speculation_weight_matvec(context, &context->markov_output,
                                       markov, context->markov_bias, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_additive_adjust(
            context->target_logits,
            context->base_logits + ordinal * context->vocabulary_size,
            context->vocabulary_size, base_result, context->markov_bias,
            context->vocabulary_size, context->adjusted_logits,
            context->vocabulary_size, &adjusted, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_source_from_logits(
            context->draft_sampling, &source, context->adjusted_logits,
            context->vocabulary_size, &adjusted, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_distribution(
            context->draft_sampling, &source,
            context->draft_probabilities + ordinal * context->vocabulary_size,
            context->vocabulary_size, &distribution, err);
    if (rc == YVEX_OK)
        token = context->sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY
                    ? speculation_argmax(
                          context->draft_probabilities +
                              ordinal * context->vocabulary_size,
                          context->vocabulary_size)
                    : speculation_sample(
                          context->draft_probabilities +
                              ordinal * context->vocabulary_size,
                          NULL, context->vocabulary_size, uniform, 0);
    if (rc == YVEX_OK && token == UINT32_MAX)
        rc = speculation_refuse(err, YVEX_ERR_FORMAT,
                                "draft distribution has no selectable mass");
    if (rc == YVEX_OK) {
        memcpy(confidence_input,
               context->draft_pre_normalized + ordinal * context->hidden_width,
               (size_t)context->hidden_width * sizeof(float));
        memcpy(confidence_input + context->hidden_width, markov,
               (size_t)context->policy.markov_rank * sizeof(float));
        rc = speculation_weight_matvec(context, &context->confidence,
                                       confidence_input, confidence, err);
    }
    if (rc == YVEX_OK) *selected = token;
    return rc;
}

static unsigned int speculation_argmax(const float *values,
                                       unsigned long long count)
{
    unsigned long long index, selected = 0ull;
    for (index = 1ull; index < count; ++index)
        if (values[index] > values[selected]) selected = index;
    return (unsigned int)selected;
}

static int speculation_execute_draft(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result, yvex_error *err)
{
    yvex_runtime_transformer_result draft = {0};
    yvex_runtime_logits_row_result rows[YVEX_SPECULATION_MAX_BLOCK] = {{0}};
    yvex_runtime_sampling_uniform_result draw_result = {0};
    yvex_sha256 hash;
    unsigned long long started, index, draft_count = context->policy.block_size;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    int rc;
    context->draft_input_ids[0] = request->conditioning_token_id;
    for (index = 1ull; index < draft_count; ++index)
        context->draft_input_ids[index] = (unsigned int)context->policy.noise_token_id;
    started = yvex_core_monotonic_ns();
    /* The target features at position - 1 come from the input token whose target
     * distribution produced the conditioning token at position. The reference
     * stores those main_x features first, then embeds that target-authored output
     * token beside the noise queries. Reprocessing the conditioning token before
     * this block would shift both caches and every proposal by one position.
     *
     * Always execute and sample the complete source-authored block. Candidate
     * attention is non-causal inside that block, so running only a bounded output
     * prefix would alter the hidden states being verified. The request extent
     * limits only the prefix handed to the target verifier. */
    rc = speculation_transformer_execute(
        context, context->draft_transformer, context->draft_input_ids,
        draft_count, request->position, 1,
        YVEX_ATTENTION_TRANSACTION_ABORT, NULL, 0ull,
        context->draft_hidden, context->draft_pre_normalized, NULL, &draft, err);
    if (rc == YVEX_OK)
        rc = speculation_project_draft_base(
            context, &draft, rows, draft_count, err);
    if (rc == YVEX_OK &&
        context->sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_STOCHASTIC)
        rc = yvex_runtime_sampling_transaction_begin(
            context->draft_sampling, &context->draft_rng, err);
    for (index = 0ull; rc == YVEX_OK && index < draft_count; ++index) {
        unsigned int previous = index ? result->candidate_token_ids[index - 1ull]
                                      : request->conditioning_token_id;
        double uniform = 0.0;
        if (context->draft_rng)
            rc = yvex_runtime_sampling_transaction_uniforms(
                context->draft_rng, &uniform, 1ull, &draw_result, err);
        if (rc == YVEX_OK) result->draft_rng_draw_count += draw_result.draw_count;
        if (rc == YVEX_OK)
            rc = speculation_draft_one(
                context, index, previous, &rows[index], uniform,
                &result->candidate_token_ids[index],
                &result->confidence_logits[index], err);
    }
    result->draft_ns = yvex_core_monotonic_ns() - started;
    yvex_sha256_init(&hash);
    if (rc == YVEX_OK &&
        (!yvex_sha256_update_text(&hash, "yvex.runtime.speculation.draft.v1") ||
         !yvex_sha256_update_text(&hash, context->policy.policy_identity) ||
         !yvex_sha256_update_text(&hash, draft.execution_identity) ||
         !yvex_sha256_update_u64(&hash, request->position) ||
         !yvex_sha256_update_u64(&hash, draft_count) ||
         !yvex_sha256_update_u64(&hash, request->candidate_count)))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "draft execution identity initialization failed");
    for (index = 0ull; rc == YVEX_OK && index < draft_count; ++index) {
        char probabilities[YVEX_SPECULATION_IDENTITY_CAP];
        if (!speculation_values_digest(
                "yvex.runtime.speculation.draft-distribution.v1",
                context->draft_probabilities + index * context->vocabulary_size,
                context->vocabulary_size, probabilities) ||
            !yvex_sha256_update_u64(&hash, result->candidate_token_ids[index]) ||
            !yvex_sha256_update_text(&hash, probabilities))
            rc = speculation_refuse(err, YVEX_ERR_STATE,
                                    "draft distribution identity failed");
    }
    if (rc == YVEX_OK) {
        char confidence[YVEX_SPECULATION_IDENTITY_CAP];
        if (!speculation_values_digest(
                "yvex.runtime.speculation.confidence.v1",
                result->confidence_logits, draft_count, confidence) ||
            !yvex_sha256_update_text(&hash, confidence))
            rc = speculation_refuse(err, YVEX_ERR_STATE,
                                    "draft confidence identity failed");
    }
    if (rc == YVEX_OK && !yvex_sha256_final(&hash, digest))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "draft execution identity finalization failed");
    if (rc == YVEX_OK)
        yvex_sha256_hex(digest, result->draft_execution_identity);
    return rc;
}

static int speculation_verify_target(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result, yvex_error *err)
{
    yvex_runtime_transformer_result target = {0};
    unsigned int verification_tokens[YVEX_SPECULATION_MAX_BLOCK + 1u] = {0};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long started, row;
    int rc;
    verification_tokens[0] = request->conditioning_token_id;
    memcpy(verification_tokens + 1u, result->candidate_token_ids,
           (size_t)request->candidate_count * sizeof(*verification_tokens));
    started = yvex_core_monotonic_ns();
    /* Row zero is the target distribution for the first draft candidate after
     * consuming the target-authored conditioning token. The final row supplies
     * the correction or bonus distribution when the complete draft survives. */
    rc = speculation_transformer_execute(
        context, context->target_transformer, verification_tokens,
        request->candidate_count + 1ull, request->position, 0,
        YVEX_ATTENTION_TRANSACTION_ABORT, NULL, 0ull,
        context->target_hidden, NULL, NULL, &target, err);
    yvex_sha256_init(&hash);
    if (rc == YVEX_OK &&
        (!yvex_sha256_update_text(&hash, "yvex.runtime.speculation.verify.v1") ||
         !yvex_sha256_update_text(&hash, context->pending_source_identity) ||
         !yvex_sha256_update_text(&hash, target.execution_identity)))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "target verification identity initialization failed");
    for (row = 0ull; rc == YVEX_OK && row <= request->candidate_count; ++row) {
        yvex_runtime_logits_source logits_source = {0};
        yvex_runtime_logits_row_result logits = {0};
        yvex_runtime_sampling_source sampling_source = {0};
        yvex_runtime_sampling_distribution_result distribution = {0};
        rc = yvex_runtime_logits_source_from_transformer(
            context->target_logits, &logits_source, &target,
            context->target_hidden,
            (request->candidate_count + 1ull) * context->hidden_width, row, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_logits_project(
                context->target_logits, &logits_source, context->options.backend,
                context->adjusted_logits, context->vocabulary_size, &logits, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_sampling_source_from_logits(
                context->target_sampling, &sampling_source,
                context->adjusted_logits, context->vocabulary_size, &logits, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_sampling_distribution(
                context->target_sampling, &sampling_source,
                context->target_probabilities + row * context->vocabulary_size,
                context->vocabulary_size, &distribution, err);
        if (rc == YVEX_OK &&
            !yvex_sha256_update_text(&hash, distribution.distribution_identity))
            rc = speculation_refuse(err, YVEX_ERR_STATE,
                                    "target distribution identity update failed");
    }
    result->verification_ns = yvex_core_monotonic_ns() - started;
    if (rc == YVEX_OK && !yvex_sha256_final(&hash, digest))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "target verification identity finalization failed");
    if (rc == YVEX_OK) {
        yvex_sha256_hex(digest, result->verification_execution_identity);
        result->target_verification_count = 1ull;
    }
    return rc;
}

static int speculation_target_draws(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result,
    double uniforms[YVEX_SPECULATION_MAX_BLOCK], double *correction,
    yvex_error *err)
{
    yvex_runtime_sampling_uniform_result draw_result = {0};
    unsigned long long index;
    int rc = context->target_rng
                 ? YVEX_OK
                 : yvex_runtime_sampling_transaction_begin(
                       context->target_sampling, &context->target_rng, err);
    /* DeepSpec draws the whole acceptance mask before taking its contiguous
     * prefix. Consuming every position keeps the target RNG contract stable
     * even when an early rejection makes later draws observationally inert. */
    for (index = 0ull; rc == YVEX_OK && index < request->candidate_count;
         ++index) {
        unsigned int candidate = result->candidate_token_ids[index];
        double q;
        rc = yvex_runtime_sampling_transaction_uniforms(
            context->target_rng, &uniforms[index], 1ull, &draw_result, err);
        if (rc == YVEX_OK) result->target_rng_draw_count += draw_result.draw_count;
        q = context->draft_probabilities[
            index * context->vocabulary_size + candidate];
        if (rc == YVEX_OK && q <= 0.0)
            rc = speculation_refuse(err, YVEX_ERR_FORMAT,
                                    "draft proposal has zero admitted probability");
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_transaction_uniforms(
            context->target_rng, correction, 1ull, &draw_result, err);
    if (rc == YVEX_OK) result->target_rng_draw_count += draw_result.draw_count;
    return rc;
}

static int speculation_accept_cycle(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result, yvex_error *err)
{
    yvex_speculation_acceptance_request acceptance = {0};
    unsigned int target_tokens[YVEX_SPECULATION_MAX_BLOCK + 1u] = {0};
    unsigned int accepted[YVEX_SPECULATION_MAX_BLOCK + 1u] = {0};
    double uniforms[YVEX_SPECULATION_MAX_BLOCK] = {0.0};
    double correction = 0.0;
    unsigned long long started, index;
    int rc = YVEX_OK;
    acceptance.schema_version = YVEX_SPECULATION_SCHEMA_V1;
    acceptance.kind = context->sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY
                          ? YVEX_SPECULATION_ACCEPT_GREEDY
                          : YVEX_SPECULATION_ACCEPT_STOCHASTIC;
    acceptance.candidate_count = request->candidate_count;
    acceptance.vocabulary_size = context->vocabulary_size;
    acceptance.distribution_stride = context->vocabulary_size;
    acceptance.candidate_token_ids = result->candidate_token_ids;
    acceptance.draft_probabilities = context->draft_probabilities;
    acceptance.target_probabilities = context->target_probabilities;
    acceptance.acceptance_uniforms = uniforms;
    if (acceptance.kind == YVEX_SPECULATION_ACCEPT_GREEDY) {
        for (index = 0ull; index <= request->candidate_count; ++index)
            target_tokens[index] = speculation_argmax(
                context->target_probabilities + index * context->vocabulary_size,
                context->vocabulary_size);
        acceptance.target_token_ids = target_tokens;
    } else {
        rc = speculation_target_draws(
            context, request, result, uniforms, &correction, err);
        acceptance.correction_uniform = correction;
    }
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = yvex_speculation_accept(
            &acceptance, accepted,
            YVEX_SPECULATION_MAX_BLOCK + 1ull, &result->acceptance, err);
    result->acceptance_ns = yvex_core_monotonic_ns() - started;
    if (rc == YVEX_OK) {
        memcpy(result->committed_token_ids, accepted,
               (size_t)result->acceptance.committed_count * sizeof(*accepted));
        result->committed_count = result->acceptance.committed_count;
    }
    return rc;
}

static int speculation_cycle_identity(
    const yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result)
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.runtime.speculation.cycle.v1") &&
           yvex_sha256_update_text(&hash, context->policy.policy_identity) &&
           yvex_sha256_update_text(&hash, context->pending_cycle_identity) &&
           yvex_sha256_update_u64(&hash, request->position) &&
           yvex_sha256_update_u64(&hash, request->conditioning_token_id) &&
           yvex_sha256_update_u64(&hash, result->draft_proposed_count) &&
           yvex_sha256_update_u64(&hash, result->candidate_count) &&
           yvex_sha256_update_text(&hash, result->draft_execution_identity) &&
           yvex_sha256_update_text(&hash, result->verification_execution_identity) &&
           yvex_sha256_update_text(&hash,
                                   result->acceptance.acceptance_identity) &&
           speculation_hash_finish(&hash, result->cycle_identity);
}

int yvex_runtime_speculation_cycle(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result, yvex_error *err)
{
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !request || !result || !context->cycle_pending ||
        context->pending_committed_count != 1ull ||
        context->pending_position != request->position ||
        context->pending_tokens[0] != request->conditioning_token_id ||
        !request->candidate_count ||
        request->candidate_count > context->policy.block_size ||
        request->candidate_count > context->policy.accepted_prefix_maximum ||
        request->conditioning_token_id >= context->vocabulary_size ||
        !yvex_sha256_hex_valid(context->pending_source_identity) ||
        !yvex_sha256_hex_valid(context->pending_sampling_identity))
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "DSpark cycle request does not continue its target anchor");
    result->draft_proposed_count = context->policy.block_size;
    result->candidate_count = request->candidate_count;
    result->conditioning_token_id = request->conditioning_token_id;
    result->draft_started = 1;
    rc = request->phase_sink
             ? request->phase_sink(
                   request->phase_context,
                   YVEX_RUNTIME_SPECULATION_PHASE_DRAFT_STARTED, 0ull, err)
             : YVEX_OK;
    if (rc == YVEX_OK)
        rc = speculation_execute_draft(context, request, result, err);
    if (rc == YVEX_OK) result->draft_completed = 1;
    if (rc == YVEX_OK && request->phase_sink)
        rc = request->phase_sink(
            request->phase_context,
            YVEX_RUNTIME_SPECULATION_PHASE_DRAFT_COMPLETED,
            result->draft_ns, err);
    if (rc == YVEX_OK) result->verification_started = 1;
    if (rc == YVEX_OK && request->phase_sink)
        rc = request->phase_sink(
            request->phase_context,
            YVEX_RUNTIME_SPECULATION_PHASE_VERIFICATION_STARTED, 0ull, err);
    if (rc == YVEX_OK)
        rc = speculation_verify_target(context, request, result, err);
    if (rc == YVEX_OK) result->verification_completed = 1;
    if (rc == YVEX_OK && request->phase_sink)
        rc = request->phase_sink(
            request->phase_context,
            YVEX_RUNTIME_SPECULATION_PHASE_VERIFICATION_COMPLETED,
            result->verification_ns, err);
    if (rc == YVEX_OK)
        rc = speculation_accept_cycle(context, request, result, err);
    if (rc == YVEX_OK && !speculation_cycle_identity(context, request, result))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "DSpark cycle identity derivation failed");
    if (rc == YVEX_OK) {
        result->completed = 1;
        context->pending_committed_count = result->committed_count + 1ull;
        memcpy(context->pending_tokens + 1u, result->committed_token_ids,
               (size_t)result->committed_count *
                   sizeof(*context->pending_tokens));
        yvex_runtime_identity_copy(context->pending_cycle_identity,
                                   result->cycle_identity);
        context->cycle_pending = 1;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
    (void)yvex_runtime_sampling_transaction_abort(&context->draft_rng, NULL);
    speculation_pending_clear(context);
    return rc;
}

int yvex_runtime_speculation_target_step(
    yvex_runtime_speculation_context *context, unsigned long long position,
    const float *target_probabilities, unsigned long long probability_capacity,
    const char *distribution_identity,
    yvex_runtime_speculation_target_step_result *result, yvex_error *err)
{
    yvex_runtime_sampling_uniform_result draw = {0};
    yvex_sha256 hash;
    double uniform = 0.0;
    unsigned int token;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !result || context->cycle_pending ||
        !target_probabilities || probability_capacity < context->vocabulary_size ||
        position >= context->options.context_capacity ||
        !yvex_sha256_hex_valid(distribution_identity) ||
        !speculation_probability_row(target_probabilities,
                                     context->vocabulary_size))
        return speculation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "target step requires one admitted distribution and no pending cycle");
    if (context->sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY)
        token = speculation_argmax(target_probabilities,
                                   context->vocabulary_size);
    else {
        rc = yvex_runtime_sampling_transaction_begin(
            context->target_sampling, &context->target_rng, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_sampling_transaction_uniforms(
                context->target_rng, &uniform, 1ull, &draw, err);
        token = rc == YVEX_OK
                    ? speculation_sample(target_probabilities, NULL,
                                         context->vocabulary_size, uniform, 0)
                    : UINT32_MAX;
    }
    if (rc == YVEX_OK && token == UINT32_MAX)
        rc = speculation_refuse(err, YVEX_ERR_FORMAT,
                                "target distribution has no selectable mass");
    yvex_sha256_init(&hash);
    if (rc == YVEX_OK &&
        (!yvex_sha256_update_text(
             &hash, "yvex.runtime.speculation.target-step.v1") ||
         !yvex_sha256_update_text(&hash, context->policy.policy_identity) ||
         !yvex_sha256_update_text(&hash, distribution_identity) ||
         !yvex_sha256_update_u64(&hash, position) ||
         !yvex_sha256_update_u64(&hash, token) ||
         !speculation_hash_finish(&hash, result->cycle_identity)))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "target step identity derivation failed");
    if (rc == YVEX_OK) {
        result->completed = 1;
        result->position = position;
        result->token_id = token;
        result->target_rng_draw_count = draw.draw_count;
        yvex_runtime_identity_copy(result->source_distribution_identity,
                                   distribution_identity);
        yvex_runtime_identity_copy(result->sampling_identity,
                                   result->cycle_identity);
        context->pending_position = position;
        context->pending_committed_count = 1ull;
        context->pending_tokens[0] = token;
        yvex_runtime_identity_copy(context->pending_source_identity,
                                   distribution_identity);
        yvex_runtime_identity_copy(context->pending_sampling_identity,
                                   result->cycle_identity);
        yvex_runtime_identity_copy(context->pending_cycle_identity,
                                   result->cycle_identity);
        context->cycle_pending = 1;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
    return rc;
}

static void speculation_pending_clear(yvex_runtime_speculation_context *context)
{
    if (!context) return;
    context->cycle_pending = 0;
    context->pending_position = 0ull;
    context->pending_committed_count = 0ull;
    memset(context->pending_tokens, 0, sizeof(context->pending_tokens));
    context->pending_source_identity[0] = '\0';
    context->pending_sampling_identity[0] = '\0';
    context->pending_cycle_identity[0] = '\0';
}

static int speculation_rng_prepare(void *opaque, yvex_error *err) {
    yvex_runtime_speculation_context *context = opaque;
    int rc = YVEX_OK;
    if (context->target_rng)
        rc = yvex_runtime_sampling_transaction_prepare_commit(
            context->target_rng, err);
    if (rc == YVEX_OK && context->draft_rng)
        rc = yvex_runtime_sampling_transaction_prepare_commit(
            context->draft_rng, err);
    if (rc == YVEX_OK && context->publication) {
        rc = context->publication->prepare(context->publication->context, err);
        context->publication_prepared = rc == YVEX_OK;
    }
    if (rc != YVEX_OK) {
        (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
        (void)yvex_runtime_sampling_transaction_abort(&context->draft_rng, NULL);
    }
    return rc;
}

static void speculation_rng_publish(void *opaque) {
    yvex_runtime_speculation_context *context = opaque;
    yvex_runtime_sampling_transaction_publish_commit(&context->target_rng);
    yvex_runtime_sampling_transaction_publish_commit(&context->draft_rng);
    if (context->publication_prepared)
        context->publication->publish(context->publication->context);
    context->publication = NULL;
    context->publication_prepared = 0;
    speculation_pending_clear(context);
}

static void speculation_rng_cancel(void *opaque) {
    yvex_runtime_speculation_context *context = opaque;
    (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
    (void)yvex_runtime_sampling_transaction_abort(&context->draft_rng, NULL);
    if (context->publication)
        context->publication->cancel(context->publication->context);
    context->publication = NULL;
    context->publication_prepared = 0;
    speculation_pending_clear(context);
}

static int speculation_stage_tokens(
    yvex_runtime_speculation_context *context, const unsigned int *token_ids,
    unsigned long long token_start, unsigned long long token_count,
    yvex_runtime_transformer_result *target,
    yvex_runtime_transformer_core_commit_result *draft, yvex_error *err)
{
    int rc = speculation_transformer_execute(
        context, context->target_transformer, token_ids,
        token_count, token_start, 0,
        YVEX_ATTENTION_TRANSACTION_STAGE,
        context->policy.target_feature_layers,
        context->policy.target_feature_layer_count, context->target_hidden,
        NULL, context->target_features, target, err);
    if (rc == YVEX_OK)
        rc = speculation_project_target_features(
            context, context->target_features, token_count, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_stage_core_features(
            context->draft_transformer, token_start,
            context->feature_projected, token_count, draft, err);
    return rc;
}

int yvex_runtime_speculation_prefill(
    yvex_runtime_speculation_context *context, const unsigned int *token_ids,
    unsigned long long token_start, unsigned long long token_count,
    float *normalized_hidden, unsigned long long normalized_hidden_capacity,
    yvex_runtime_transformer_result *target_result,
    yvex_runtime_speculation_feature_result *draft_result, yvex_error *err)
{
    yvex_runtime_transformer_result target = {0};
    yvex_runtime_transformer_core_commit_result draft = {0};
    yvex_runtime_speculation_feature_result prepared = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_sha256 hash;
    unsigned long long feature_values;
    int acquired = 0, rc;
    if (target_result) memset(target_result, 0, sizeof(*target_result));
    if (draft_result) memset(draft_result, 0, sizeof(*draft_result));
    if (!context || !token_ids || !token_count ||
        token_count > context->policy.block_size + 2ull || !normalized_hidden ||
        !target_result || !draft_result || context->cycle_pending ||
        token_count > context->options.context_capacity ||
        token_start > context->options.context_capacity - token_count ||
        !yvex_core_u64_mul(token_count, context->hidden_width, &feature_values) ||
        normalized_hidden_capacity < feature_values)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "DSpark prefill extent is not admitted");
    rc = yvex_runtime_session_begin(context->session, &failure, err);
    acquired = rc == YVEX_OK;
    if (rc == YVEX_OK)
        rc = speculation_stage_tokens(context, token_ids, token_start, token_count,
                                      &target, &draft, err);
    if (rc == YVEX_OK) {
        prepared.token_start = token_start;
        prepared.token_count = token_count;
        prepared.position_after = draft.position_after;
        yvex_runtime_identity_copy(prepared.persistent_state_digest,
                                   draft.persistent_state_digest);
        if (target.position_after != token_start + token_count ||
            draft.position_after != token_start + token_count ||
            !yvex_sha256_hex_valid(target.persistent_state_digest) ||
            !yvex_sha256_hex_valid(prepared.persistent_state_digest) ||
            !speculation_values_digest(
                "yvex.runtime.speculation.projected-features.v2",
                context->feature_projected, feature_values,
                prepared.projected_feature_digest))
            rc = speculation_refuse(
                err, YVEX_ERR_STATE,
                "DSpark prefill staged identity is incomplete");
    }
    if (rc == YVEX_OK) {
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update_text(
                &hash, "yvex.runtime.speculation.prefill.v1") ||
            !yvex_sha256_update_text(&hash, context->policy.policy_identity) ||
            !yvex_sha256_update_text(&hash, target.execution_identity) ||
            !yvex_sha256_update_text(&hash, draft.execution_identity) ||
            !yvex_sha256_update_text(&hash, prepared.projected_feature_digest) ||
            !yvex_sha256_update_text(&hash, prepared.persistent_state_digest) ||
            !yvex_sha256_update_u64(&hash, token_start) ||
            !yvex_sha256_update_u64(&hash, token_count) ||
            !speculation_hash_finish(&hash, prepared.execution_identity))
            rc = speculation_refuse(
                err, YVEX_ERR_STATE,
                "DSpark prefill identity derivation failed");
    }
    if (acquired)
        rc = yvex_runtime_session_finish_coordinated(context->session, rc, NULL, err);
    if (rc != YVEX_OK) return rc;
    memcpy(normalized_hidden, context->target_hidden,
           (size_t)feature_values * sizeof(float));
    *target_result = target;
    prepared.completed = 1;
    *draft_result = prepared;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int speculation_commit_identity(
    const yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_commit_result *result,
    char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(
               &hash, "yvex.runtime.speculation.commit.v1") &&
           yvex_sha256_update_text(&hash, context->policy.policy_identity) &&
           yvex_sha256_update_text(
               &hash, result->target_execution_identity) &&
           yvex_sha256_update_text(
               &hash, result->draft_execution_identity) &&
           yvex_sha256_update_text(&hash, result->target_state_identity) &&
           yvex_sha256_update_text(&hash, result->draft_state_identity) &&
           yvex_sha256_update_u64(&hash, result->token_start) &&
           yvex_sha256_update_u64(&hash, result->token_count) &&
           speculation_hash_finish(&hash, output);
}

int yvex_runtime_speculation_commit_prefix(
    yvex_runtime_speculation_context *context,
    unsigned long long committed_count, float *final_hidden,
    unsigned long long final_hidden_capacity,
    const yvex_runtime_commit_participant *publication,
    yvex_runtime_speculation_commit_result *result, yvex_error *err)
{
    yvex_runtime_transformer_result target = {0};
    yvex_runtime_transformer_core_commit_result draft = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_commit_participant participant = {
        .context = context, .prepare = speculation_rng_prepare,
        .publish = speculation_rng_publish, .cancel = speculation_rng_cancel};
    unsigned long long final_values, started;
    int acquired = 0, rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !result || !context->cycle_pending || !committed_count ||
        !final_hidden ||
        !yvex_core_u64_mul(committed_count, context->hidden_width,
                           &final_values) ||
        final_values > SIZE_MAX / sizeof(*final_hidden) ||
        final_hidden_capacity < final_values ||
        (publication && (!publication->prepare || !publication->publish ||
                         !publication->cancel)) ||
        committed_count > context->pending_committed_count ||
        committed_count > context->policy.block_size + 2ull ||
        committed_count > context->options.context_capacity ||
        context->pending_position > context->options.context_capacity - committed_count)
        return speculation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "committed prefix is not an admitted pending DSpark result");
    result->token_start = context->pending_position;
    result->token_count = committed_count;
    yvex_runtime_identity_copy(result->cycle_identity,
                               context->pending_cycle_identity);
    context->publication = publication;
    context->publication_prepared = 0;
    started = yvex_core_monotonic_ns();
    rc = yvex_runtime_session_begin(context->session, &failure, err);
    acquired = rc == YVEX_OK;
    if (rc == YVEX_OK)
        rc = speculation_stage_tokens(
            context, context->pending_tokens, context->pending_position,
            committed_count, &target, &draft, err);
    if (rc == YVEX_OK) {
        result->position_after = context->pending_position + committed_count;
        result->target_result = target;
        yvex_runtime_identity_copy(result->target_execution_identity,
                                   target.execution_identity);
        yvex_runtime_identity_copy(result->draft_execution_identity,
                                   draft.execution_identity);
        yvex_runtime_identity_copy(result->target_state_identity,
                                   target.persistent_state_digest);
        yvex_runtime_identity_copy(result->draft_state_identity,
                                   draft.persistent_state_digest);
        if (target.position_after != result->position_after ||
            draft.position_after != result->position_after ||
            !yvex_sha256_hex_valid(result->target_state_identity) ||
            !yvex_sha256_hex_valid(result->draft_state_identity) ||
            !speculation_commit_identity(context, result,
                                         result->commit_identity))
            rc = speculation_refuse(
                err, YVEX_ERR_STATE,
                "speculative staged prefix identity is incomplete");
    }
    if (acquired)
        rc = yvex_runtime_session_finish_coordinated(
            context->session, rc, &participant, err);
    else
        speculation_rng_cancel(context);
    if (rc != YVEX_OK) return rc;
    result->commit_ns = yvex_core_monotonic_ns() - started;
    /* The transformer result authenticates the complete committed prefix. Keep
     * every matching hidden row until the next anchor distribution is sealed;
     * retaining only the last row would break that producer/digest contract on
     * the second speculative cycle. */
    memcpy(final_hidden, context->target_hidden,
           (size_t)final_values * sizeof(*final_hidden));
    result->completed = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_speculation_cycle_abort(
    yvex_runtime_speculation_context *context, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!context)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "DSpark cycle owner is required");
    if (!context->cycle_pending) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_runtime_sampling_transaction_abort(&context->target_rng, err);
    if (context->draft_rng) {
        int abort_rc = yvex_runtime_sampling_transaction_abort(
            &context->draft_rng, rc == YVEX_OK ? err : NULL);
        if (rc == YVEX_OK) rc = abort_rc;
    }
    speculation_pending_clear(context);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_runtime_speculation_finish_terminal(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_commit_participant *publication, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!context || !context->cycle_pending ||
        !context->pending_committed_count || !publication ||
        !publication->prepare || !publication->publish ||
        !publication->cancel)
        return speculation_refuse(
            err, YVEX_ERR_STATE,
            "terminal completion requires pending target and publication state");
    if (context->target_rng)
        rc = yvex_runtime_sampling_transaction_prepare_commit(
            context->target_rng, err);
    if (rc == YVEX_OK && context->draft_rng)
        rc = yvex_runtime_sampling_transaction_prepare_commit(
            context->draft_rng, err);
    if (rc == YVEX_OK)
        rc = publication->prepare(publication->context, err);
    if (rc == YVEX_OK) {
        yvex_runtime_sampling_transaction_publish_commit(&context->target_rng);
        yvex_runtime_sampling_transaction_publish_commit(&context->draft_rng);
        publication->publish(publication->context);
        speculation_pending_clear(context);
        yvex_error_clear(err);
    } else {
        publication->cancel(publication->context);
        (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
        (void)yvex_runtime_sampling_transaction_abort(&context->draft_rng, NULL);
    }
    return rc;
}

int yvex_runtime_speculation_context_close(
    yvex_runtime_speculation_context **context, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if ((*context)->cycle_pending) {
        rc = yvex_runtime_speculation_cycle_abort(*context, err);
        if (rc != YVEX_OK) return rc;
    }
    if ((*context)->draft_sampling)
        rc = yvex_runtime_sampling_context_close(&(*context)->draft_sampling, err);
    if (rc == YVEX_OK && (*context)->draft_transformer)
        rc = yvex_runtime_transformer_context_close(
            &(*context)->draft_transformer, err);
    if (rc != YVEX_OK) return rc;
    yvex_core_free((*context)->draft_input_ids);
    yvex_core_free((*context)->markov_embedding_values);
    yvex_core_free((*context)->draft_probabilities);
    yvex_core_free((*context)->markov_bias);
    yvex_core_free((*context)->adjusted_logits);
    yvex_core_free((*context)->base_logits);
    yvex_core_free((*context)->target_hidden);
    yvex_core_free((*context)->draft_pre_normalized);
    yvex_core_free((*context)->draft_hidden);
    yvex_core_free((*context)->target_features);
    yvex_core_free((*context)->feature_norm_weights);
    yvex_core_free((*context)->feature_projected);
    memset(*context, 0, sizeof(**context));
    yvex_core_free(*context);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
