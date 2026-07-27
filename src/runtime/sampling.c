/* Owner: runtime.sampling.
 * Owns: immutable sampling policy, real-logits admission, canonical filters, PCG state, and token publication.
 * Does not own: model/artifact open, logits math, KV, decode, token append, tokenizer, stop policy, or generation.
 * Invariants: all vocabulary values participate and failed samples commit neither a token nor an RNG transition.
 * Boundary: family-neutral host sampling over one complete admitted logits row.
 * Purpose: provide reusable deterministic and explicitly seeded stochastic token selection.
 * Inputs: logits plan/row identities, immutable F32 logits, bounded policy, and caller-owned result storage.
 * Effects: reuses fixed workspace and commits only sampling-local RNG state after successful publication.
 * Failure: typed refusal preserves logits, runtime state, prior samples, and uncommitted RNG state. */
#include <yvex/internal/sampling.h>
#include <yvex/internal/core.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

#define SAMPLING_PCG_MULTIPLIER UINT64_C(6364136223846793005)
#define SAMPLING_PCG_INCREMENT UINT64_C(1442695040888963407)
#define SAMPLING_NORMALIZATION_TOLERANCE 1.0e-12

typedef struct {
    unsigned int token_id;
    float logit;
    double probability, deviation;
} sampling_candidate;

struct yvex_runtime_sampling_context {
    yvex_runtime_logits_plan_summary logits_plan;
    yvex_runtime_sampling_policy policy;
    yvex_runtime_sampling_options options;
    sampling_candidate *candidates, *scratch;
    uint64_t rng_state, rng_increment;
    unsigned long long successful_draws;
    pthread_mutex_t mutex;
    yvex_runtime_sampling_context_summary summary;
    int mutex_ready, busy;
};

/* Purpose: publish one typed sampling refusal. */
static int sampling_refuse(yvex_error *err, yvex_status status,
                           const char *message)
{
    yvex_error_set(err, status, "runtime.sampling", message);
    return status;
}

/* Purpose: append canonical F32 bits to one semantic identity.
 * Inputs: active hash and finite value. Effects: advances only the hash.
 * Failure: returns false on hash refusal. Boundary: excludes native struct layout. */
static int sampling_hash_f32(yvex_sha256 *hash, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

/* Purpose: append canonical F64 bits to one semantic identity.
 * Inputs: active hash and finite value. Effects: advances only the hash.
 * Failure: returns false on hash refusal. Boundary: excludes native struct layout. */
static int sampling_hash_f64(yvex_sha256 *hash, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

/* Purpose: seal one SHA-256 context into its canonical lowercase identity.
 * Inputs: active hash and bounded output. Effects: finalizes the hash into text.
 * Failure: returns false on finalization refusal. Boundary: identity formatting only. */
static int sampling_hash_finish(yvex_sha256 *hash,
                                char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

/* Purpose: derive the immutable sampling policy identity field by field.
 * Inputs: complete policy and identity output. Effects: writes only the output identity.
 * Failure: returns false on absent input or hash refusal. Boundary: excludes padding and pointers. */
static int sampling_policy_identity(
    const yvex_runtime_sampling_policy *policy,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    if (!policy ||
        !yvex_sha256_update_text(&hash, "yvex.runtime.sampling.policy.v1") ||
        !yvex_sha256_update_u64(&hash, policy->schema_version) ||
        !yvex_sha256_update_u64(&hash, policy->strategy) ||
        !sampling_hash_f64(&hash, policy->temperature) ||
        !yvex_sha256_update_u64(&hash, policy->top_k) ||
        !sampling_hash_f64(&hash, policy->top_p) ||
        !sampling_hash_f64(&hash, policy->min_p) ||
        !sampling_hash_f64(&hash, policy->typical_p) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)policy->seed_present) ||
        !yvex_sha256_update_u64(&hash, policy->seed) ||
        !yvex_sha256_update_u64(&hash, policy->rng_algorithm) ||
        !yvex_sha256_update_u64(&hash, policy->rng_version) ||
        !yvex_sha256_update_u64(&hash, policy->filter_order_version) ||
        !sampling_hash_finish(&hash, output)) return 0;
    return 1;
}

/* Purpose: validate and seal one explicit immutable policy for an exact vocabulary.
 * Inputs: mutable policy, vocabulary bound, and error. Effects: fills version and identity fields.
 * Failure: invalid strategy, filters, or seed refuse. Boundary: no logits or RNG state is consumed. */
int yvex_runtime_sampling_policy_seal(
    yvex_runtime_sampling_policy *policy, unsigned long long vocabulary_size,
    yvex_error *err)
{
    if (!policy || !vocabulary_size || vocabulary_size > UINT_MAX)
        return sampling_refuse(err, YVEX_ERR_INVALID_ARG,
                               "sampling policy and bounded vocabulary are required");
    if (policy->schema_version != YVEX_RUNTIME_SAMPLING_SCHEMA_V1 ||
        policy->strategy > YVEX_SAMPLING_STRATEGY_STOCHASTIC)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling policy schema or strategy is unsupported");
    if (policy->strategy == YVEX_SAMPLING_STRATEGY_GREEDY) {
        if (policy->temperature != 1.0 || policy->top_k || policy->top_p != 1.0 ||
            policy->min_p != 0.0 || policy->typical_p != 1.0 ||
            policy->seed_present)
            return sampling_refuse(err, YVEX_ERR_FORMAT,
                                   "greedy sampling requires neutral filters and no seed");
        policy->seed = 0ull;
    } else if (!policy->seed_present || !isfinite(policy->temperature) ||
               policy->temperature <= 0.0 || policy->top_k > vocabulary_size ||
               !isfinite(policy->top_p) || policy->top_p <= 0.0 ||
               policy->top_p > 1.0 || !isfinite(policy->min_p) ||
               policy->min_p < 0.0 || policy->min_p > 1.0 ||
               !isfinite(policy->typical_p) || policy->typical_p <= 0.0 ||
               policy->typical_p > 1.0)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "stochastic sampling policy values are invalid");
    if (policy->min_p == 0.0) policy->min_p = 0.0;
    policy->rng_algorithm = YVEX_SAMPLING_RNG_PCG_XSH_RR_64_32;
    policy->rng_version = YVEX_SAMPLING_RNG_VERSION_V1;
    policy->filter_order_version = YVEX_SAMPLING_FILTER_ORDER_V1;
    if (!sampling_policy_identity(policy, policy->policy_identity))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling policy identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: advance the versioned private PCG-XSH-RR 64/32 state once. */
static uint32_t sampling_pcg_next(uint64_t *state, uint64_t increment)
{
    uint64_t old = *state;
    uint32_t shifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    unsigned int rotation = (unsigned int)(old >> 59u);
    *state = old * SAMPLING_PCG_MULTIPLIER + increment;
    return (shifted >> rotation) | (shifted << ((0u - rotation) & 31u));
}

/* Purpose: establish the exact PCG v1 seed and fixed-stream policy. */
static void sampling_pcg_seed(uint64_t seed, uint64_t *state, uint64_t *increment)
{
    *state = 0ull;
    *increment = SAMPLING_PCG_INCREMENT;
    (void)sampling_pcg_next(state, *increment);
    *state += seed;
    (void)sampling_pcg_next(state, *increment);
}

/* Purpose: identify one RNG transition state without exposing platform layout. */
static int sampling_rng_identity(const yvex_runtime_sampling_context *context,
                                 uint64_t state, unsigned long long draws,
                                 char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return context &&
           yvex_sha256_update_text(&hash, "yvex.runtime.sampling.rng.v1") &&
           yvex_sha256_update_u64(&hash, context->policy.rng_algorithm) &&
           yvex_sha256_update_u64(&hash, context->policy.rng_version) &&
           yvex_sha256_update_u64(&hash, context->policy.seed) &&
           yvex_sha256_update_u64(&hash, state) &&
           yvex_sha256_update_u64(&hash, context->rng_increment) &&
           yvex_sha256_update_u64(&hash, draws) &&
           sampling_hash_finish(&hash, output);
}

/* Purpose: open one fixed-workspace sampling context without model/session ownership.
 * Inputs: logits plan, sealed policy, resource limits, and empty output.
 * Effects: allocates stable workspaces, mutex, and private RNG. Failure: publishes no context.
 * Boundary: borrows plan identities and never opens model, artifact, session, or CUDA owners. */
int yvex_runtime_sampling_context_open(
    yvex_runtime_sampling_context **out,
    const yvex_runtime_logits_plan_summary *logits_plan,
    const yvex_runtime_sampling_policy *policy,
    const yvex_runtime_sampling_options *options, yvex_error *err)
{
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_policy canonical;
    unsigned long long one_bytes, workspace_bytes, owned_bytes;
    if (out) *out = NULL;
    if (!out || !logits_plan || !policy || !options ||
        !logits_plan->vocabulary_size ||
        logits_plan->vocabulary_size != logits_plan->row_count ||
        !yvex_sha256_hex_valid(logits_plan->output_head_plan_identity) ||
        options->maximum_vocabulary_size < logits_plan->vocabulary_size ||
        !options->maximum_rows ||
        !yvex_core_u64_mul(logits_plan->vocabulary_size,
                           sizeof(sampling_candidate), &one_bytes) ||
        !yvex_core_u64_mul(one_bytes, 2ull, &workspace_bytes) ||
        !yvex_core_u64_add(workspace_bytes, sizeof(*context), &owned_bytes) ||
        workspace_bytes > SIZE_MAX ||
        (options->maximum_host_bytes &&
         owned_bytes > options->maximum_host_bytes))
        return sampling_refuse(err, YVEX_ERR_NOMEM,
                               "sampling context geometry or workspace budget is invalid");
    canonical = *policy;
    canonical.policy_identity[0] = '\0';
    if (yvex_runtime_sampling_policy_seal(
            &canonical, logits_plan->vocabulary_size, err) != YVEX_OK ||
        strcmp(canonical.policy_identity, policy->policy_identity) != 0)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling policy identity is stale");
    context = (yvex_runtime_sampling_context *)yvex_core_calloc(1u, sizeof(*context));
    if (!context) return sampling_refuse(err, YVEX_ERR_NOMEM,
                                         "sampling context allocation failed");
    context->candidates = (sampling_candidate *)yvex_core_malloc((size_t)one_bytes);
    context->scratch = (sampling_candidate *)yvex_core_malloc((size_t)one_bytes);
    if (!context->candidates || !context->scratch) {
        yvex_core_free(context->scratch);
        yvex_core_free(context->candidates);
        yvex_core_free(context);
        return sampling_refuse(err, YVEX_ERR_NOMEM,
                               "sampling workspace allocation failed");
    }
    context->logits_plan = *logits_plan;
    context->policy = canonical;
    context->options = *options;
    sampling_pcg_seed(canonical.seed, &context->rng_state, &context->rng_increment);
    if (pthread_mutex_init(&context->mutex, NULL) != 0) {
        yvex_core_free(context->scratch);
        yvex_core_free(context->candidates);
        yvex_core_free(context);
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling context synchronization failed");
    }
    context->mutex_ready = 1;
    context->summary.schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    context->summary.vocabulary_size = logits_plan->vocabulary_size;
    context->summary.maximum_rows = options->maximum_rows;
    context->summary.workspace_bytes = workspace_bytes;
    context->summary.workspace_generation = 1ull;
    context->summary.cold_workspace_allocations = 2ull;
    yvex_runtime_identity_copy(context->summary.output_head_plan_identity,
                               logits_plan->output_head_plan_identity);
    yvex_runtime_identity_copy(context->summary.policy_identity,
                               canonical.policy_identity);
    if (!sampling_rng_identity(context, context->rng_state, 0ull,
                               context->summary.rng_state_identity)) {
        (void)pthread_mutex_destroy(&context->mutex);
        yvex_core_free(context->scratch);
        yvex_core_free(context->candidates);
        yvex_core_free(context);
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "initial sampling RNG identity failed");
    }
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: derive one field-wise sampling source identity over a borrowed complete logits row. */
static int sampling_source_identity(yvex_runtime_sampling_source *source)
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return source &&
           yvex_sha256_update_text(&hash, "yvex.runtime.sampling.source.v1") &&
           yvex_sha256_update_u64(&hash, source->schema_version) &&
           yvex_sha256_update_u64(&hash, source->source_phase) &&
           yvex_sha256_update_u64(&hash, source->source_position) &&
           yvex_sha256_update_u64(&hash, source->vocabulary_size) &&
           yvex_sha256_update_text(&hash, source->raw_logits_digest) &&
           yvex_sha256_update_text(&hash, source->logits_row_identity) &&
           yvex_sha256_update_text(&hash, source->output_head_plan_identity) &&
           yvex_sha256_update_text(&hash, source->source_hidden_digest) &&
           yvex_sha256_update_text(&hash, source->backend_execution_identity) &&
           sampling_hash_finish(&hash, source->source_identity);
}

/* Purpose: admit one exact immutable logits publication into the sampling boundary.
 * Inputs: context plan, caller-owned complete row, capacity, and logits-owner result.
 * Effects: publishes one borrowed identity-bound source without taking ownership.
 * Failure: logits-owner validation, extent, or identity mismatch publishes no source.
 * Boundary: does not normalize, rank, mutate logits, or access output-head weights. */
int yvex_runtime_sampling_source_from_logits(
    const yvex_runtime_sampling_context *context,
    yvex_runtime_sampling_source *source, const float *logits,
    unsigned long long logits_capacity,
    const yvex_runtime_logits_row_result *row, yvex_error *err)
{
    if (source) memset(source, 0, sizeof(*source));
    if (!context || !source || !logits || !row ||
        yvex_runtime_logits_row_validate(&context->logits_plan, logits,
                                         logits_capacity, row, err) != YVEX_OK)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling requires one admitted complete logits row");
    source->schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    source->source_phase = row->source_phase;
    source->source_position = row->source_position;
    source->vocabulary_size = row->vocabulary_size;
    source->logits_capacity = logits_capacity;
    source->logits = logits;
    yvex_runtime_identity_copy(source->raw_logits_digest, row->raw_logits_digest);
    yvex_runtime_identity_copy(source->logits_row_identity, row->logits_row_identity);
    yvex_runtime_identity_copy(source->output_head_plan_identity,
                               row->output_head_plan_identity);
    yvex_runtime_identity_copy(source->source_hidden_digest,
                               row->source_hidden_digest);
    yvex_runtime_identity_copy(source->backend_execution_identity,
                               row->backend_execution_identity);
    if (!sampling_source_identity(source)) {
        memset(source, 0, sizeof(*source));
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling source identity derivation failed");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: revalidate one borrowed source immediately before candidate construction.
 * Inputs: context and immutable source. Effects: rehashes without mutating caller logits.
 * Failure: stale identity, extent, or non-finite values refuse. Boundary: no sampling occurs. */
static int sampling_source_validate(
    const yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source, yvex_error *err)
{
    yvex_runtime_sampling_source canonical;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char raw_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long index;
    if (!context || !source ||
        source->schema_version != YVEX_RUNTIME_SAMPLING_SCHEMA_V1 ||
        source->source_phase > YVEX_LOGITS_SOURCE_DECODE ||
        source->vocabulary_size != context->logits_plan.vocabulary_size ||
        source->logits_capacity < source->vocabulary_size || !source->logits ||
        strcmp(source->output_head_plan_identity,
               context->logits_plan.output_head_plan_identity) != 0 ||
        !yvex_sha256_hex_valid(source->logits_row_identity) ||
        !yvex_sha256_hex_valid(source->source_hidden_digest) ||
        !yvex_sha256_hex_valid(source->backend_execution_identity))
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling source geometry or identity is stale");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.raw-logits.v1"))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling raw-logits validation initialization failed");
    for (index = 0ull; index < source->vocabulary_size; ++index) {
        uint32_t bits;
        if (!isfinite(source->logits[index]))
            return sampling_refuse(err, YVEX_ERR_FORMAT,
                                   "sampling source contains non-finite logits");
        memcpy(&bits, &source->logits[index], sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits))
            return sampling_refuse(err, YVEX_ERR_STATE,
                                   "sampling raw-logits validation failed");
    }
    if (!yvex_sha256_final(&hash, digest))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling raw-logits digest finalization failed");
    yvex_sha256_hex(digest, raw_digest);
    if (strcmp(raw_digest, source->raw_logits_digest) != 0)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling source logits were mutated after sealing");
    canonical = *source;
    canonical.source_identity[0] = '\0';
    if (!sampling_source_identity(&canonical) ||
        strcmp(canonical.source_identity, source->source_identity) != 0)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling source identity is not canonical");
    return YVEX_OK;
}

/* Purpose: impose probability-descending, token-ascending total order. */
static int sampling_probability_compare(const void *left, const void *right)
{
    const sampling_candidate *a = (const sampling_candidate *)left;
    const sampling_candidate *b = (const sampling_candidate *)right;
    if (a->probability > b->probability) return -1;
    if (a->probability < b->probability) return 1;
    return a->token_id < b->token_id ? -1 : a->token_id > b->token_id;
}

/* Purpose: impose typical-deviation, probability, token total order. */
static int sampling_typical_compare(const void *left, const void *right)
{
    const sampling_candidate *a = (const sampling_candidate *)left;
    const sampling_candidate *b = (const sampling_candidate *)right;
    if (a->deviation < b->deviation) return -1;
    if (a->deviation > b->deviation) return 1;
    return sampling_probability_compare(left, right);
}

/* Purpose: impose canonical ascending token-ID draw and identity order. */
static int sampling_token_compare(const void *left, const void *right)
{
    const sampling_candidate *a = (const sampling_candidate *)left;
    const sampling_candidate *b = (const sampling_candidate *)right;
    return a->token_id < b->token_id ? -1 : a->token_id > b->token_id;
}

/* Purpose: sort candidates deterministically through the context-owned scratch buffer.
 * Inputs: exclusive context, bounded candidate count, and total-order comparator.
 * Effects: applies a stable bottom-up merge sort and leaves the result in candidates.
 * Failure: none for context-open geometry. Boundary: performs no warm allocation. */
static void sampling_stable_sort(
    yvex_runtime_sampling_context *context, unsigned long long count,
    int (*compare)(const void *, const void *))
{
    sampling_candidate *source = context->candidates;
    sampling_candidate *target = context->scratch;
    unsigned long long width = 1ull;
    while (width < count) {
        unsigned long long base = 0ull;
        while (base < count) {
            unsigned long long left = base;
            unsigned long long middle = base + width < count ? base + width : count;
            unsigned long long right = middle;
            unsigned long long end = middle + width < count ? middle + width : count;
            unsigned long long output = base;
            while (left < middle && right < end) {
                if (compare(&source[left], &source[right]) <= 0)
                    target[output++] = source[left++];
                else
                    target[output++] = source[right++];
            }
            while (left < middle) target[output++] = source[left++];
            while (right < end) target[output++] = source[right++];
            base = end;
        }
        {
            sampling_candidate *swap = source;
            source = target;
            target = swap;
        }
        width = width > count / 2ull ? count : width * 2ull;
    }
    if (source != context->candidates)
        memcpy(context->candidates, source,
               (size_t)count * sizeof(*context->candidates));
}

/* Purpose: normalize candidate weights and enforce the v1 sum contract.
 * Inputs: bounded candidate set and optional error output. Effects: replaces weights with probabilities.
 * Failure: empty, negative, non-finite, or zero mass refuses. Boundary: sampling-local workspace only. */
static int sampling_normalize(sampling_candidate *candidates,
                              unsigned long long count,
                              double *normalization_error, yvex_error *err)
{
    double total = 0.0, normalized = 0.0;
    unsigned long long index;
    if (!candidates || !count)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling filter removed every candidate");
    for (index = 0ull; index < count; ++index) {
        if (!isfinite(candidates[index].probability) ||
            candidates[index].probability < 0.0)
            return sampling_refuse(err, YVEX_ERR_FORMAT,
                                   "sampling candidate weight is negative or non-finite");
        total += candidates[index].probability;
    }
    if (!isfinite(total) || total <= 0.0)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling probability total is invalid");
    for (index = 0ull; index < count; ++index) {
        candidates[index].probability /= total;
        normalized += candidates[index].probability;
    }
    if (!isfinite(normalized) ||
        fabs(normalized - 1.0) > SAMPLING_NORMALIZATION_TOLERANCE)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling normalization tolerance was exceeded");
    if (normalization_error) *normalization_error = fabs(normalized - 1.0);
    return YVEX_OK;
}

/* Purpose: build the stable full-vocabulary temperature softmax in double precision.
 * Inputs: context policy, immutable logits source, result evidence, and count output.
 * Effects: fills sampling-local candidates. Failure: non-finite or invalid normalization refuses.
 * Boundary: consumes every vocabulary value and publishes no probability vector. */
static int sampling_softmax(yvex_runtime_sampling_context *context,
                            const yvex_runtime_sampling_source *source,
                            yvex_runtime_sampling_result *result,
                            unsigned long long *count, yvex_error *err)
{
    double maximum = -DBL_MAX;
    unsigned long long index;
    *count = source->vocabulary_size;
    for (index = 0ull; index < *count; ++index) {
        double scaled = (double)source->logits[index] / context->policy.temperature;
        if (!isfinite(scaled))
            return sampling_refuse(err, YVEX_ERR_FORMAT,
                                   "temperature scaling produced a non-finite value");
        context->candidates[index].token_id = (unsigned int)index;
        context->candidates[index].logit = source->logits[index];
        context->candidates[index].probability = scaled;
        context->candidates[index].deviation = 0.0;
        if (scaled > maximum) maximum = scaled;
    }
    result->maximum_logit = source->logits[0];
    for (index = 0ull; index < *count; ++index) {
        double weight = exp(context->candidates[index].probability - maximum);
        if (!isfinite(weight) || weight < 0.0)
            return sampling_refuse(err, YVEX_ERR_FORMAT,
                                   "stable softmax produced an invalid exponential");
        context->candidates[index].probability = weight;
        if (source->logits[index] > result->maximum_logit)
            result->maximum_logit = source->logits[index];
    }
    return sampling_normalize(context->candidates, *count,
                              &result->normalization_error, err);
}

/* Purpose: apply deterministic exact-count top-k and renormalize survivors. */
static int sampling_filter_top_k(yvex_runtime_sampling_context *context,
                                 unsigned long long *count,
                                 yvex_runtime_sampling_result *result,
                                 yvex_error *err)
{
    result->effective_top_k = context->policy.top_k;
    if (context->policy.top_k && context->policy.top_k < *count) {
        sampling_stable_sort(context, *count, sampling_probability_compare);
        *count = context->policy.top_k;
        if (sampling_normalize(context->candidates, *count,
                               &result->normalization_error, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    result->candidates_after_top_k = *count;
    return YVEX_OK;
}

/* Purpose: apply inclusive relative-to-maximum min-p filtering.
 * Inputs: current candidates, count, policy, and result. Effects: compacts and renormalizes workspace.
 * Failure: empty or invalid mass refuses. Boundary: threshold is relative, never absolute. */
static int sampling_filter_min_p(yvex_runtime_sampling_context *context,
                                 unsigned long long *count,
                                 yvex_runtime_sampling_result *result,
                                 yvex_error *err)
{
    unsigned long long read, write = 0ull;
    double maximum = 0.0;
    result->effective_min_p = context->policy.min_p;
    if (context->policy.min_p == 0.0) {
        result->candidates_after_min_p = *count;
        return YVEX_OK;
    }
    for (read = 0ull; read < *count; ++read)
        if (context->candidates[read].probability > maximum)
            maximum = context->candidates[read].probability;
    result->min_p_threshold = context->policy.min_p * maximum;
    for (read = 0ull; read < *count; ++read)
        if (context->candidates[read].probability >= result->min_p_threshold)
            context->candidates[write++] = context->candidates[read];
    if (!write || sampling_normalize(context->candidates, write,
                                     &result->normalization_error, err) != YVEX_OK)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "min-p filtering produced no valid candidates");
    *count = write;
    result->candidates_after_min_p = *count;
    return YVEX_OK;
}

/* Purpose: apply locally typical prefix filtering with stable total ordering.
 * Inputs: current normalized candidates, policy, result, and count. Effects: ranks and renormalizes workspace.
 * Failure: nonpositive mass or invalid entropy refuses. Boundary: includes the threshold-crossing candidate. */
static int sampling_filter_typical(yvex_runtime_sampling_context *context,
                                   unsigned long long *count,
                                   yvex_runtime_sampling_result *result,
                                   yvex_error *err)
{
    unsigned long long index, retained;
    double entropy = 0.0, cumulative = 0.0;
    result->effective_typical_p = context->policy.typical_p;
    if (context->policy.typical_p == 1.0) {
        result->candidates_after_typical_p = *count;
        return YVEX_OK;
    }
    for (index = 0ull; index < *count; ++index) {
        double probability = context->candidates[index].probability;
        if (!isfinite(probability) || probability <= 0.0)
            return sampling_refuse(err, YVEX_ERR_FORMAT,
                                   "typical sampling requires positive survivor probabilities");
        entropy -= probability * log(probability);
    }
    if (!isfinite(entropy))
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "typical sampling entropy is non-finite");
    result->entropy = entropy;
    for (index = 0ull; index < *count; ++index)
        context->candidates[index].deviation =
            fabs(-log(context->candidates[index].probability) - entropy);
    sampling_stable_sort(context, *count, sampling_typical_compare);
    retained = 0ull;
    while (retained < *count && cumulative < context->policy.typical_p) {
        cumulative += context->candidates[retained].probability;
        retained++;
    }
    if (!retained) retained = 1ull;
    result->typical_retained_mass = cumulative;
    if (sampling_normalize(context->candidates, retained,
                           &result->normalization_error, err) != YVEX_OK)
        return yvex_error_code(err);
    *count = retained;
    result->candidates_after_typical_p = *count;
    return YVEX_OK;
}

/* Purpose: apply stable nucleus prefix filtering and its final renormalization.
 * Inputs: current candidates, top-p policy, result, and count. Effects: ranks and renormalizes workspace.
 * Failure: invalid mass refuses. Boundary: includes the threshold-crossing candidate. */
static int sampling_filter_top_p(yvex_runtime_sampling_context *context,
                                 unsigned long long *count,
                                 yvex_runtime_sampling_result *result,
                                 yvex_error *err)
{
    unsigned long long retained;
    double cumulative = 0.0;
    result->effective_top_p = context->policy.top_p;
    if (context->policy.top_p == 1.0) {
        result->candidates_after_top_p = *count;
        return sampling_normalize(context->candidates, *count,
                                  &result->normalization_error, err);
    }
    sampling_stable_sort(context, *count, sampling_probability_compare);
    retained = 0ull;
    while (retained < *count && cumulative < context->policy.top_p) {
        cumulative += context->candidates[retained].probability;
        retained++;
    }
    if (!retained) retained = 1ull;
    result->top_p_retained_mass = cumulative;
    if (sampling_normalize(context->candidates, retained,
                           &result->normalization_error, err) != YVEX_OK)
        return yvex_error_code(err);
    *count = retained;
    result->candidates_after_top_p = *count;
    return YVEX_OK;
}

/* Purpose: bind the final ordered survivor token IDs to source and policy. */
static int sampling_candidate_identity(
    const yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source,
    const sampling_candidate *candidates, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!context || !source || !candidates || !count ||
        !yvex_sha256_update_text(&hash, "yvex.runtime.sampling.candidates.v1") ||
        !yvex_sha256_update_text(&hash, source->raw_logits_digest) ||
        !yvex_sha256_update_text(&hash, context->policy.policy_identity) ||
        !yvex_sha256_update_u64(&hash, context->policy.filter_order_version) ||
        !yvex_sha256_update_u64(&hash, count)) return 0;
    for (index = 0ull; index < count; ++index)
        if (!yvex_sha256_update_u64(&hash, candidates[index].token_id)) return 0;
    return sampling_hash_finish(&hash, output);
}

/* Purpose: derive the selected-token identity from canonical result fields. */
static int sampling_selected_identity(
    const yvex_runtime_sampling_result *result,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return result &&
           yvex_sha256_update_text(&hash, "yvex.runtime.sampling.selected-token.v1") &&
           yvex_sha256_update_u64(&hash, result->strategy) &&
           yvex_sha256_update_text(&hash, result->candidate_set_identity) &&
           yvex_sha256_update_u64(&hash, result->selected_token_id) &&
           sampling_hash_f32(&hash, result->selected_logit) &&
           sampling_hash_f64(&hash, result->selected_probability) &&
           yvex_sha256_update_text(&hash, result->rng_state_before_identity) &&
           yvex_sha256_update_text(&hash, result->rng_state_after_identity) &&
           sampling_hash_finish(&hash, output);
}

/* Purpose: derive one complete per-row sampling execution identity. */
static int sampling_execution_identity(
    const yvex_runtime_sampling_result *result,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return result &&
           yvex_sha256_update_text(&hash, "yvex.runtime.sampling.execution.v1") &&
           yvex_sha256_update_u64(&hash, result->schema_version) &&
           yvex_sha256_update_u64(&hash, result->source_phase) &&
           yvex_sha256_update_u64(&hash, result->source_position) &&
           yvex_sha256_update_text(&hash, result->policy_identity) &&
           yvex_sha256_update_text(&hash, result->source_identity) &&
           yvex_sha256_update_text(&hash, result->selected_token_identity) &&
           yvex_sha256_update_u64(&hash, result->final_candidate_count) &&
           yvex_sha256_update_u64(&hash, result->rng_draw_count) &&
           sampling_hash_finish(&hash, output);
}

/* Purpose: enter one sampling context exclusively. */
static int sampling_enter(yvex_runtime_sampling_context *context, yvex_error *err)
{
    if (!context || !context->mutex_ready ||
        pthread_mutex_lock(&context->mutex) != 0)
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling context lock failed");
    if (context->busy) {
        context->summary.failure_count++;
        (void)pthread_mutex_unlock(&context->mutex);
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling context is already in use");
    }
    context->busy = 1;
    (void)pthread_mutex_unlock(&context->mutex);
    return YVEX_OK;
}

/* Purpose: leave one sampling context and account its completed/refused lifecycle. */
static void sampling_leave(yvex_runtime_sampling_context *context, int rc,
                           unsigned long long completed)
{
    if (!context || !context->mutex_ready ||
        pthread_mutex_lock(&context->mutex) != 0) return;
    context->busy = 0;
    context->summary.successful_samples += completed;
    if (rc != YVEX_OK) {
        context->summary.failure_count++;
        if (rc == YVEX_ERR_CANCELLED) context->summary.cancellation_count++;
    }
    (void)pthread_mutex_unlock(&context->mutex);
}

/* Purpose: finish field-wise identities before any selected-token publication.
 * Inputs: context, source, and staged result. Effects: seals identities and completion on the staged result.
 * Failure: canonical identity mismatch refuses. Boundary: caller publication and RNG commit follow success. */
static int sampling_result_finish(
    const yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source,
    yvex_runtime_sampling_result *result, yvex_error *err)
{
    yvex_runtime_sampling_result canonical;
    yvex_runtime_identity_copy(result->policy_identity,
                               context->policy.policy_identity);
    yvex_runtime_identity_copy(result->source_identity, source->source_identity);
    if (!sampling_selected_identity(result, result->selected_token_identity) ||
        !sampling_execution_identity(result, result->execution_identity))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling result identity derivation failed");
    canonical = *result;
    canonical.selected_token_identity[0] = '\0';
    canonical.execution_identity[0] = '\0';
    if (!sampling_selected_identity(&canonical,
                                    canonical.selected_token_identity) ||
        !sampling_execution_identity(&canonical,
                                     canonical.execution_identity) ||
        strcmp(canonical.selected_token_identity,
               result->selected_token_identity) != 0 ||
        strcmp(canonical.execution_identity, result->execution_identity) != 0)
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling result identities are not canonical");
    result->completed = 1;
    return YVEX_OK;
}

/* Purpose: execute the complete-vocabulary deterministic maximum path.
 * Inputs: context workspace, admitted source, and staged result. Effects: fills selection evidence only.
 * Failure: identity derivation refuses. Boundary: scans all logits, consumes zero RNG, and low token wins ties. */
static int sampling_select_greedy(
    yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source,
    yvex_runtime_sampling_result *result, yvex_error *err)
{
    unsigned long long index, selected = 0ull, ties = 1ull;
    float maximum = source->logits[0];
    for (index = 0ull; index < source->vocabulary_size; ++index) {
        float value = source->logits[index];
        context->candidates[index].token_id = (unsigned int)index;
        context->candidates[index].logit = value;
        context->candidates[index].probability = 0.0;
        context->candidates[index].deviation = 0.0;
        if (index && value > maximum) {
            maximum = value;
            selected = index;
            ties = 1ull;
        } else if (index && value == maximum) {
            ties++;
        }
    }
    result->maximum_logit = maximum;
    result->tied_maximum_count = ties;
    result->selected_token_id = (unsigned int)selected;
    result->selected_logit = maximum;
    result->selected_probability = 1.0;
    result->selected_log_probability = 0.0;
    result->candidates_after_top_k = result->candidates_after_min_p =
        result->candidates_after_typical_p = result->candidates_after_top_p =
            result->final_candidate_count = source->vocabulary_size;
    if (!sampling_candidate_identity(context, source, context->candidates,
                                     source->vocabulary_size,
                                     result->candidate_set_identity))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "greedy candidate identity derivation failed");
    if (!sampling_rng_identity(context, context->rng_state,
                               context->successful_draws,
                               result->rng_state_before_identity))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "greedy RNG identity derivation failed");
    yvex_runtime_identity_copy(result->rng_state_after_identity,
                               result->rng_state_before_identity);
    return YVEX_OK;
}

/* Purpose: remove exact zero-mass candidates before final categorical ordering. */
static int sampling_remove_zero_mass(
    yvex_runtime_sampling_context *context, unsigned long long *count,
    yvex_runtime_sampling_result *result, yvex_error *err)
{
    unsigned long long read, write = 0ull;
    for (read = 0ull; read < *count; ++read)
        if (context->candidates[read].probability > 0.0)
            context->candidates[write++] = context->candidates[read];
    if (!write || sampling_normalize(context->candidates, write,
                                     &result->normalization_error, err) != YVEX_OK)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling has no positive final probability mass");
    *count = write;
    return YVEX_OK;
}

/* Purpose: execute canonical softmax/filter/draw using a transactional PCG copy.
 * Inputs: exclusive context, source, staged result, and candidate RNG output.
 * Effects: mutates workspace and copied RNG only. Failure: publishes neither token nor RNG transition.
 * Boundary: the caller commits the candidate RNG only after complete result sealing. */
static int sampling_select_stochastic(
    yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source,
    yvex_runtime_sampling_result *result, uint64_t *committed_state,
    yvex_error *err)
{
    unsigned long long count, index, selected = ULLONG_MAX;
    uint64_t next_state = context->rng_state;
    uint32_t random_value;
    double uniform, cumulative = 0.0;
    int rc = sampling_softmax(context, source, result, &count, err);
    if (rc == YVEX_OK) rc = sampling_filter_top_k(context, &count, result, err);
    if (rc == YVEX_OK) rc = sampling_filter_min_p(context, &count, result, err);
    if (rc == YVEX_OK) rc = sampling_filter_typical(context, &count, result, err);
    if (rc == YVEX_OK) rc = sampling_filter_top_p(context, &count, result, err);
    if (rc == YVEX_OK) rc = sampling_remove_zero_mass(context, &count, result, err);
    if (rc != YVEX_OK) return rc;
    sampling_stable_sort(context, count, sampling_token_compare);
    result->final_candidate_count = count;
    if (!sampling_candidate_identity(context, source, context->candidates, count,
                                     result->candidate_set_identity) ||
        !sampling_rng_identity(context, context->rng_state,
                               context->successful_draws,
                               result->rng_state_before_identity))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "stochastic candidate or RNG identity failed");
    if (context->options.cancel_requested &&
        context->options.cancel_requested(context->options.cancel_context))
        return sampling_refuse(err, YVEX_ERR_CANCELLED,
                               "sampling was cancelled before RNG publication");
    random_value = sampling_pcg_next(&next_state, context->rng_increment);
    uniform = ((double)random_value + 0.5) / 4294967296.0;
    for (index = 0ull; index < count; ++index) {
        cumulative += context->candidates[index].probability;
        if (uniform < cumulative) {
            selected = index;
            break;
        }
    }
    if (selected == ULLONG_MAX) {
        selected = count - 1ull;
        result->numeric_fallback_used = 1;
    }
    if (context->candidates[selected].probability <= 0.0 ||
        !isfinite(context->candidates[selected].probability))
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "categorical draw selected invalid probability mass");
    result->selected_token_id = context->candidates[selected].token_id;
    result->selected_logit = context->candidates[selected].logit;
    result->selected_probability = context->candidates[selected].probability;
    result->selected_log_probability = log(result->selected_probability);
    result->rng_draw_count = 1ull;
    if (!isfinite(result->selected_log_probability) ||
        !sampling_rng_identity(context, next_state,
                               context->successful_draws + 1ull,
                               result->rng_state_after_identity))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "stochastic selected probability or RNG identity failed");
    *committed_state = next_state;
    return YVEX_OK;
}

/* Purpose: select one token while the caller owns context exclusion.
 * Inputs: busy reusable context, admitted source, and caller result.
 * Effects: commits one RNG transition only after complete stochastic result sealing.
 * Failure: publishes no completed result and restores the prior RNG authority.
 * Boundary: never appends, decodes, changes KV/session state, or interprets token text. */
static int sampling_select_owned(
    yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source,
    yvex_runtime_sampling_result *result, yvex_error *err)
{
    yvex_runtime_sampling_result staged;
    uint64_t committed_state = 0ull;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    memset(&staged, 0, sizeof(staged));
    if (!result) rc = sampling_refuse(err, YVEX_ERR_INVALID_ARG,
                                      "sampling result storage is required");
    if (rc == YVEX_OK && context->options.cancel_requested &&
        context->options.cancel_requested(context->options.cancel_context))
        rc = sampling_refuse(err, YVEX_ERR_CANCELLED,
                             "sampling was cancelled before source admission");
    if (rc == YVEX_OK) rc = sampling_source_validate(context, source, err);
    if (rc == YVEX_OK) {
        staged.schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
        staged.strategy = context->policy.strategy;
        staged.source_phase = source->source_phase;
        staged.source_position = source->source_position;
        staged.vocabulary_size = staged.values_considered =
            staged.candidates_before = source->vocabulary_size;
        staged.greedy_tie_policy = YVEX_SAMPLING_GREEDY_LOWEST_TOKEN_ID;
        staged.temperature = context->policy.temperature;
        staged.effective_top_p = context->policy.top_p;
        staged.effective_min_p = context->policy.min_p;
        staged.effective_typical_p = context->policy.typical_p;
        if (context->policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY)
            rc = sampling_select_greedy(context, source, &staged, err);
        else
            rc = sampling_select_stochastic(context, source, &staged,
                                            &committed_state, err);
    }
    if (rc == YVEX_OK)
        rc = sampling_result_finish(context, source, &staged, err);
    if (rc == YVEX_OK) {
        if (staged.rng_draw_count) {
            context->rng_state = committed_state;
            context->successful_draws++;
            context->summary.stochastic_draws = context->successful_draws;
            yvex_runtime_identity_copy(context->summary.rng_state_identity,
                                       staged.rng_state_after_identity);
        }
        *result = staged;
        yvex_error_clear(err);
    }
    return rc;
}

/* Purpose: select one token transactionally from one exact complete logits source.
 * Inputs: reusable context, admitted source, and caller result.
 * Effects: excludes concurrent use and delegates one complete selection.
 * Failure: publishes no completed result and leaves the context reusable.
 * Boundary: one-shot wrapper over the same owned mechanism used by ordered execution. */
int yvex_runtime_sampling_select(
    yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source,
    yvex_runtime_sampling_result *result, yvex_error *err)
{
    int rc = sampling_enter(context, err);
    if (result) memset(result, 0, sizeof(*result));
    if (rc != YVEX_OK) return rc;
    rc = sampling_select_owned(context, source, result, err);
    sampling_leave(context, rc, rc == YVEX_OK ? 1ull : 0ull);
    return rc;
}

/* Purpose: validate selected-token and execution identities after publication.
 * Inputs: completed result and error output. Effects: derives only temporary canonical identities.
 * Failure: malformed fields or identity mutation refuses. Boundary: no source, RNG, or model state changes. */
int yvex_runtime_sampling_result_validate(
    const yvex_runtime_sampling_result *result, yvex_error *err)
{
    char selected[YVEX_SHA256_HEX_CAP], execution[YVEX_SHA256_HEX_CAP];
    if (!result || !result->completed ||
        result->schema_version != YVEX_RUNTIME_SAMPLING_SCHEMA_V1 ||
        result->strategy > YVEX_SAMPLING_STRATEGY_STOCHASTIC ||
        !result->vocabulary_size ||
        result->selected_token_id >= result->vocabulary_size ||
        !isfinite(result->selected_logit) ||
        !isfinite(result->selected_probability) ||
        result->selected_probability <= 0.0 ||
        !isfinite(result->selected_log_probability) ||
        !result->final_candidate_count ||
        !yvex_sha256_hex_valid(result->candidate_set_identity) ||
        !yvex_sha256_hex_valid(result->policy_identity) ||
        !yvex_sha256_hex_valid(result->source_identity) ||
        !yvex_sha256_hex_valid(result->rng_state_before_identity) ||
        !yvex_sha256_hex_valid(result->rng_state_after_identity) ||
        !sampling_selected_identity(result, selected) ||
        !sampling_execution_identity(result, execution) ||
        strcmp(selected, result->selected_token_identity) != 0 ||
        strcmp(execution, result->execution_identity) != 0)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling result identity or fields are invalid");
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: seal one repeated sampling result over ordered successful token identities.
 * Inputs: aggregate and completed result prefix. Effects: publishes aggregate identities and status.
 * Failure: hash refusal leaves aggregate incomplete. Boundary: failing and later rows never participate. */
static int sampling_execution_finish(
    yvex_runtime_sampling_execution *execution,
    const yvex_runtime_sampling_result *results, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.runtime.sampling.selected-sequence.v1") ||
        !yvex_sha256_update_u64(&hash, execution->completed_samples))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling sequence identity initialization failed");
    for (index = 0ull; index < execution->completed_samples; ++index)
        if (!yvex_sha256_update_text(&hash,
                                     results[index].selected_token_identity))
            return sampling_refuse(err, YVEX_ERR_STATE,
                                   "sampling sequence identity update failed");
    if (!yvex_sha256_final(&hash, digest))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling sequence identity finalization failed");
    yvex_sha256_hex(digest, execution->ordered_selected_token_digest);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.sampling.aggregate.v1") ||
        !yvex_sha256_update_u64(&hash, execution->requested_samples) ||
        !yvex_sha256_update_u64(&hash, execution->completed_samples) ||
        !yvex_sha256_update_u64(&hash, execution->first_incomplete_sample) ||
        !yvex_sha256_update_text(&hash,
                                 execution->initial_rng_state_identity) ||
        !yvex_sha256_update_text(&hash, execution->final_rng_state_identity) ||
        !yvex_sha256_update_text(
            &hash, execution->ordered_selected_token_digest) ||
        !sampling_hash_finish(&hash, execution->aggregate_sampling_identity))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "aggregate sampling identity derivation failed");
    execution->completed =
        execution->completed_samples == execution->requested_samples;
    execution->partial = !execution->completed && execution->completed_samples != 0ull;
    return YVEX_OK;
}

/* Purpose: sample ordered rows with row-atomic publication and transactional RNG progress.
 * Inputs: one context, ordered sources, and complete caller result directory.
 * Effects: retains prior successful results and stops at the first failed row.
 * Failure: reports exact partial progress; the failing row and its RNG transition remain absent.
 * Boundary: request atomicity is intentionally not provided. */
int yvex_runtime_sampling_execute(
    yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *sources, unsigned long long source_count,
    yvex_runtime_sampling_result *results, unsigned long long result_capacity,
    yvex_runtime_sampling_execution *execution, yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    if (execution) memset(execution, 0, sizeof(*execution));
    if (!context || !sources || !source_count ||
        source_count > context->options.maximum_rows || !results ||
        result_capacity < source_count || !execution ||
        source_count > SIZE_MAX / sizeof(*results))
        return sampling_refuse(err, YVEX_ERR_INVALID_ARG,
                               "ordered sampling request or result capacity is invalid");
    memset(results, 0, (size_t)source_count * sizeof(*results));
    execution->schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    execution->requested_samples = source_count;
    execution->first_incomplete_sample = source_count;
    rc = sampling_enter(context, err);
    if (rc != YVEX_OK) return rc;
    if (!sampling_rng_identity(context, context->rng_state,
                               context->successful_draws,
                               execution->initial_rng_state_identity)) {
        rc = sampling_refuse(err, YVEX_ERR_STATE,
                             "initial repeated RNG identity failed");
        goto leave;
    }
    for (index = 0ull; index < source_count; ++index) {
        rc = sampling_select_owned(context, &sources[index],
                                   &results[index], err);
        if (rc != YVEX_OK) {
            execution->first_incomplete_sample = index;
            break;
        }
        execution->completed_samples++;
    }
    if (!sampling_rng_identity(context, context->rng_state,
                               context->successful_draws,
                               execution->final_rng_state_identity)) {
        rc = sampling_refuse(err, YVEX_ERR_STATE,
                             "final repeated RNG identity failed");
        goto leave;
    }
    if (sampling_execution_finish(execution, results, err) != YVEX_OK) {
        rc = yvex_error_code(err);
        goto leave;
    }
    if (rc == YVEX_OK) yvex_error_clear(err);
leave:
    sampling_leave(context, rc, execution->completed_samples);
    return rc;
}

/* Purpose: inspect stable workspace, policy, counters, and current RNG authority.
 * Inputs: live context and result output. Effects: copies a synchronized summary.
 * Failure: lock or identity derivation refusal clears output. Boundary: no workspace or RNG mutation. */
int yvex_runtime_sampling_context_snapshot(
    const yvex_runtime_sampling_context *context,
    yvex_runtime_sampling_context_summary *summary, yvex_error *err)
{
    yvex_runtime_sampling_context *mutable =
        (yvex_runtime_sampling_context *)context;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!mutable || !summary || !mutable->mutex_ready ||
        pthread_mutex_lock(&mutable->mutex) != 0)
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling context snapshot failed");
    if (mutable->busy) {
        (void)pthread_mutex_unlock(&mutable->mutex);
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "busy sampling context cannot be inspected");
    }
    *summary = mutable->summary;
    if (!sampling_rng_identity(mutable, mutable->rng_state,
                               mutable->successful_draws,
                               summary->rng_state_identity)) {
        (void)pthread_mutex_unlock(&mutable->mutex);
        memset(summary, 0, sizeof(*summary));
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling snapshot RNG identity failed");
    }
    (void)pthread_mutex_unlock(&mutable->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: release fixed sampling workspace without touching borrowed logits or runtime state.
 * Inputs: context owner and error output. Effects: destroys mutex and frees fixed workspaces.
 * Failure: busy context refuses and remains owned. Boundary: idempotent after successful close. */
int yvex_runtime_sampling_context_close(
    yvex_runtime_sampling_context **context, yvex_error *err)
{
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if ((*context)->mutex_ready) {
        if (pthread_mutex_lock(&(*context)->mutex) != 0)
            return sampling_refuse(err, YVEX_ERR_STATE,
                                   "sampling context close lock failed");
        if ((*context)->busy) {
            (void)pthread_mutex_unlock(&(*context)->mutex);
            return sampling_refuse(err, YVEX_ERR_STATE,
                                   "busy sampling context cannot close");
        }
        (void)pthread_mutex_unlock(&(*context)->mutex);
        if (pthread_mutex_destroy(&(*context)->mutex) != 0)
            return sampling_refuse(err, YVEX_ERR_STATE,
                                   "sampling context synchronization cleanup failed");
        (*context)->mutex_ready = 0;
    }
    yvex_core_free((*context)->scratch);
    yvex_core_free((*context)->candidates);
    memset(*context, 0, sizeof(**context));
    yvex_core_free(*context);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: publish sampling readiness only after real-logits selection completes.
 * Inputs: successful operator result and context summary. Effects: copies resource and capability facts.
 * Failure: none for admitted inputs. Boundary: cannot promote tokenizer, append, CUDA sampling, or generation. */
static void sampling_operator_publish(
    yvex_sampling_operator_result *result,
    const yvex_runtime_sampling_context_summary *summary)
{
    result->sample_count = result->execution.completed_samples;
    result->prefill_samples = result->sample_count ? 1ull : 0ull;
    result->decode_samples = result->sample_count - result->prefill_samples;
    result->workspace_bytes = summary->workspace_bytes;
    result->workspace_generation = summary->workspace_generation;
    result->cold_workspace_allocations = summary->cold_workspace_allocations;
    result->warm_workspace_allocations = summary->warm_workspace_allocations;
    result->sampling_source_contract_ready = 1;
    result->sampling_policy_ready = 1;
    result->sampling_greedy_ready = 1;
    result->sampling_temperature_ready = 1;
    result->sampling_top_k_ready = 1;
    result->sampling_top_p_ready = 1;
    result->sampling_min_p_ready = 1;
    result->sampling_typical_ready = 1;
    result->sampling_stochastic_ready = 1;
    result->sampling_seed_reproducibility_ready = 1;
    result->sampling_real_logits_ready = 1;
    result->sampling_partial_progress_ready = 1;
    result->sampling_ready = 1;
    result->persistent_state_unchanged = 1;
}

/* Purpose: execute the admitted logits workflow and sample every completed real row.
 * Inputs: exact logits operator request, explicit policy, host workspace budget, and cancellation.
 * Effects: retains bounded logits and sampling evidence while never appending selected tokens.
 * Failure: preserves typed logits/sampling partial progress and any retained runtime cleanup lease.
 * Boundary: this is an operator adapter, not a generation loop or a model/session owner. */
int yvex_runtime_sampling_operator_execute(
    const yvex_sampling_operator_request *request,
    yvex_sampling_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err)
{
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_source *sources = NULL;
    yvex_runtime_sampling_context_summary summary = {0};
    yvex_runtime_sampling_options options = {0};
    yvex_runtime_sampling_policy policy;
    yvex_error cleanup_error;
    unsigned long long index, row_count, vocabulary_size, logits_extent;
    int rc, close_rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !retained_cleanup || *retained_cleanup)
        return sampling_refuse(err, YVEX_ERR_INVALID_ARG,
                               "sampling operator request and empty cleanup output are required");
    yvex_core_text_copy(result->command, sizeof(result->command),
                        "graph transformer sample");
    yvex_core_text_copy(result->target, sizeof(result->target),
                        request->logits.target);
    yvex_core_text_copy(result->logits_backend, sizeof(result->logits_backend),
                        request->logits.backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    yvex_core_text_copy(result->sampling_execution_kind,
                        sizeof(result->sampling_execution_kind), "common-host");
    rc = yvex_runtime_logits_operator_execute(
        &request->logits, &result->logits, retained_cleanup, err);
    yvex_core_text_copy(result->family, sizeof(result->family),
                        result->logits.family);
    if (rc != YVEX_OK) goto finish;
    row_count = result->logits.row_count;
    vocabulary_size = result->logits.plan.vocabulary_size;
    if (!row_count || !vocabulary_size ||
        !yvex_core_u64_mul(row_count, vocabulary_size, &logits_extent) ||
        !result->logits.rows ||
        !result->logits.raw_logits ||
        result->logits.raw_logits_count < logits_extent) {
        rc = sampling_refuse(err, YVEX_ERR_STATE,
                             "sampling operator received no complete real logits rows");
        goto finish;
    }
    policy = request->policy;
    if (yvex_runtime_sampling_policy_seal(&policy, vocabulary_size, err) != YVEX_OK) {
        rc = yvex_error_code(err);
        goto finish;
    }
    yvex_core_text_copy(result->strategy, sizeof(result->strategy),
                        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY ? "greedy" : "stochastic");
    result->policy = policy;
    options.maximum_vocabulary_size = vocabulary_size;
    options.maximum_rows = row_count;
    options.maximum_host_bytes = request->maximum_sampling_host_bytes;
    options.cancel_requested = request->cancel_requested;
    options.cancel_context = request->cancel_context;
    rc = yvex_runtime_sampling_context_open(
        &context, &result->logits.plan, &policy, &options, err);
    if (rc != YVEX_OK) goto finish;
    if (row_count > SIZE_MAX / sizeof(*sources) ||
        row_count > SIZE_MAX / sizeof(*result->samples)) {
        rc = sampling_refuse(err, YVEX_ERR_BOUNDS,
                             "sampling operator directory extent overflowed");
        goto close_context;
    }
    sources = (yvex_runtime_sampling_source *)yvex_core_calloc(
        (size_t)row_count, sizeof(*sources));
    result->samples = (yvex_runtime_sampling_result *)yvex_core_calloc(
        (size_t)row_count, sizeof(*result->samples));
    if (!sources || !result->samples) {
        rc = sampling_refuse(err, YVEX_ERR_NOMEM,
                             "sampling operator directory allocation failed");
        goto close_context;
    }
    for (index = 0ull; index < row_count && rc == YVEX_OK; ++index)
        rc = yvex_runtime_sampling_source_from_logits(
            context, &sources[index],
            result->logits.raw_logits + index * vocabulary_size,
            vocabulary_size, &result->logits.rows[index], err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_execute(
            context, sources, row_count, result->samples, row_count,
            &result->execution, err);
    if (yvex_runtime_sampling_context_snapshot(context, &summary,
                                               &cleanup_error) == YVEX_OK &&
        (rc == YVEX_OK || result->execution.completed_samples))
        sampling_operator_publish(result, &summary);
close_context:
    yvex_error_clear(&cleanup_error);
    close_rc = yvex_runtime_sampling_context_close(&context, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        if (err) *err = cleanup_error;
    }
finish:
    yvex_core_free(sources);
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(result->status, sizeof(result->status), "complete");
        yvex_error_clear(err);
    } else {
        yvex_core_text_copy(result->status, sizeof(result->status), "refused");
        yvex_core_text_copy(result->reason, sizeof(result->reason),
                            err && yvex_error_is_set(err)
                                ? yvex_error_message(err)
                                : "sampling execution refused");
    }
    return rc;
}

/* Purpose: release sampling/operator directories in dependency order.
 * Inputs: operator result. Effects: frees samples and retained raw-logits evidence.
 * Failure: none. Boundary: does not touch runtime/session state already closed by the operator. */
void yvex_runtime_sampling_operator_result_release(
    yvex_sampling_operator_result *result)
{
    if (!result) return;
    yvex_core_free(result->samples);
    result->samples = NULL;
    result->sample_count = 0ull;
    yvex_runtime_logits_operator_result_release(&result->logits);
}
