/*
 * Provide reusable deterministic and explicitly seeded stochastic token selection.
 *
 * All vocabulary values participate and failed samples commit neither a token nor an RNG
 * transition. Family-neutral host sampling over one complete admitted logits row.
 */
#include <yvex/internal/sampling.h>
#include <yvex/internal/core.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>

#define SAMPLING_PCG_MULTIPLIER UINT64_C(6364136223846793005)
#define SAMPLING_PCG_INCREMENT UINT64_C(1442695040888963407)
#define SAMPLING_LIFECYCLE_ACTIVE 1u
#define SAMPLING_LIFECYCLE_CLOSING 2u
#define SAMPLING_LIFECYCLE_CLOSED 6u

typedef struct {
    unsigned int token_id;
    float logit;
    double probability, deviation;
} sampling_candidate;

typedef struct {
    double sum, correction;
} sampling_compensated_sum;

struct yvex_runtime_sampling_context {
    yvex_runtime_logits_plan_summary logits_plan;
    yvex_runtime_sampling_policy policy;
    yvex_runtime_sampling_options options;
    sampling_candidate *candidates, *scratch;
    uint64_t rng_state, rng_increment;
    unsigned long long successful_draws;
    atomic_uint lifecycle;
    atomic_ullong admission_failures;
    pthread_mutex_t drain_mutex;
    pthread_cond_t drain_condition;
    yvex_runtime_sampling_context_summary summary;
    int drain_mutex_ready, drain_condition_ready;
};

static int sampling_enter(yvex_runtime_sampling_context *context,
                          yvex_error *err);
static void sampling_leave(yvex_runtime_sampling_context *context, int rc,
                           unsigned long long completed);

static int sampling_refuse(yvex_error *err, yvex_status status,
                           const char *message)
{
    yvex_error_set(err, status, "runtime.sampling", message);
    return status;
}

static int sampling_hash_f32(yvex_sha256 *hash, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

static int sampling_hash_f64(yvex_sha256 *hash, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

static int sampling_hash_finish(yvex_sha256 *hash,
                                char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

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

/*
 * Validate and seal one explicit immutable policy for an exact vocabulary.
 *
 * Fills version and identity fields.
 */
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
    policy->filter_order_version = YVEX_SAMPLING_FILTER_ORDER_V2;
    if (!sampling_policy_identity(policy, policy->policy_identity))
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling policy identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

static uint32_t sampling_pcg_next(uint64_t *state, uint64_t increment)
{
    uint64_t old = *state;
    uint32_t shifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    unsigned int rotation = (unsigned int)(old >> 59u);
    *state = old * SAMPLING_PCG_MULTIPLIER + increment;
    return (shifted >> rotation) | (shifted << ((0u - rotation) & 31u));
}

static void sampling_pcg_seed(uint64_t seed, uint64_t *state, uint64_t *increment)
{
    *state = 0ull;
    *increment = SAMPLING_PCG_INCREMENT;
    (void)sampling_pcg_next(state, *increment);
    *state += seed;
    (void)sampling_pcg_next(state, *increment);
}

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

/* Open one fixed-workspace sampling context without model/session ownership. */
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
    atomic_init(&context->lifecycle, 0u);
    atomic_init(&context->admission_failures, 0ull);
    if (pthread_mutex_init(&context->drain_mutex, NULL) != 0) {
        yvex_core_free(context->scratch);
        yvex_core_free(context->candidates);
        yvex_core_free(context);
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling context synchronization failed");
    }
    context->drain_mutex_ready = 1;
    if (pthread_cond_init(&context->drain_condition, NULL) != 0) {
        (void)pthread_mutex_destroy(&context->drain_mutex);
        yvex_core_free(context->scratch);
        yvex_core_free(context->candidates);
        yvex_core_free(context);
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling context drain synchronization failed");
    }
    context->drain_condition_ready = 1;
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
        (void)pthread_cond_destroy(&context->drain_condition);
        (void)pthread_mutex_destroy(&context->drain_mutex);
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

/*
 * Admit one exact immutable logits publication into the sampling boundary.
 *
 * Publishes one borrowed identity-bound source without taking ownership. Logits-owner validation,
 * extent, or identity mismatch publishes no source.
 */
int yvex_runtime_sampling_source_from_logits(
    const yvex_runtime_sampling_context *context,
    yvex_runtime_sampling_source *source, const float *logits,
    unsigned long long logits_capacity,
    const yvex_runtime_logits_row_result *row, yvex_error *err)
{
    yvex_runtime_sampling_context *mutable =
        (yvex_runtime_sampling_context *)context;
    int rc;
    if (source) memset(source, 0, sizeof(*source));
    if (!mutable || !source || !logits || !row)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling requires one admitted complete logits row");
    rc = sampling_enter(mutable, err);
    if (rc != YVEX_OK) return rc;
    if (yvex_runtime_logits_row_validate(&mutable->logits_plan, logits,
                                         logits_capacity, row, err) != YVEX_OK) {
        sampling_leave(mutable, YVEX_ERR_FORMAT, 0ull);
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling requires one admitted complete logits row");
    }
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
        sampling_leave(mutable, YVEX_ERR_STATE, 0ull);
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling source identity derivation failed");
    }
    sampling_leave(mutable, YVEX_OK, 0ull);
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Revalidate one borrowed source immediately before candidate construction.
 *
 * Stale identity, extent, or non-finite values refuse.
 */
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

static int sampling_probability_compare(const void *left, const void *right)
{
    const sampling_candidate *a = (const sampling_candidate *)left;
    const sampling_candidate *b = (const sampling_candidate *)right;
    if (a->probability > b->probability) return -1;
    if (a->probability < b->probability) return 1;
    return a->token_id < b->token_id ? -1 : a->token_id > b->token_id;
}

static int sampling_typical_compare(const void *left, const void *right)
{
    const sampling_candidate *a = (const sampling_candidate *)left;
    const sampling_candidate *b = (const sampling_candidate *)right;
    if (a->deviation < b->deviation) return -1;
    if (a->deviation > b->deviation) return 1;
    return sampling_probability_compare(left, right);
}

static int sampling_token_compare(const void *left, const void *right)
{
    const sampling_candidate *a = (const sampling_candidate *)left;
    const sampling_candidate *b = (const sampling_candidate *)right;
    return a->token_id < b->token_id ? -1 : a->token_id > b->token_id;
}

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

static void sampling_compensated_add(sampling_compensated_sum *total,
                                     double value)
{
    double next = total->sum + value;
    if (fabs(total->sum) >= fabs(value))
        total->correction += (total->sum - next) + value;
    else
        total->correction += (value - next) + total->sum;
    total->sum = next;
}

static double sampling_normalization_tolerance(unsigned long long count)
{
    double depth = count > 1ull ? ceil(log2((double)count)) : 0.0;
    return 8.0 * DBL_EPSILON * (3.0 + depth);
}

static int sampling_normalize(sampling_candidate *candidates,
                              unsigned long long count,
                              double *normalization_error, yvex_error *err)
{
    sampling_compensated_sum total = {0.0, 0.0};
    sampling_compensated_sum normalized = {0.0, 0.0};
    double mass, verified, observed, tolerance;
    unsigned long long index;
    if (!candidates || !count)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling filter removed every candidate");
    for (index = 0ull; index < count; ++index) {
        if (!isfinite(candidates[index].probability) ||
            candidates[index].probability < 0.0)
            return sampling_refuse(err, YVEX_ERR_FORMAT,
                                   "sampling candidate weight is negative or non-finite");
        sampling_compensated_add(&total, candidates[index].probability);
    }
    mass = total.sum + total.correction;
    if (!isfinite(mass) || mass <= 0.0)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling probability total is invalid");
    for (index = 0ull; index < count; ++index) {
        candidates[index].probability /= mass;
        sampling_compensated_add(&normalized,
                                 candidates[index].probability);
    }
    verified = normalized.sum + normalized.correction;
    observed = fabs(verified - 1.0);
    tolerance = sampling_normalization_tolerance(count);
    if (!isfinite(verified) || observed > tolerance)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling normalization tolerance was exceeded");
    if (normalization_error && observed > *normalization_error)
        *normalization_error = observed;
    return YVEX_OK;
}

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

static int sampling_filter_typical(yvex_runtime_sampling_context *context,
                                   unsigned long long *count,
                                   yvex_runtime_sampling_result *result,
                                   yvex_error *err)
{
    unsigned long long index, retained;
    sampling_compensated_sum entropy_sum = {0.0, 0.0};
    double entropy, cumulative = 0.0;
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
        sampling_compensated_add(&entropy_sum,
                                 -probability * log(probability));
    }
    entropy = entropy_sum.sum + entropy_sum.correction;
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
        !yvex_sha256_update_text(&hash, "yvex.runtime.sampling.candidates.v2") ||
        !yvex_sha256_update_text(&hash, source->raw_logits_digest) ||
        !yvex_sha256_update_text(&hash, context->policy.policy_identity) ||
        !yvex_sha256_update_u64(&hash, context->policy.filter_order_version) ||
        !yvex_sha256_update_u64(&hash, count)) return 0;
    for (index = 0ull; index < count; ++index)
        if (!yvex_sha256_update_u64(&hash, candidates[index].token_id)) return 0;
    return sampling_hash_finish(&hash, output);
}

static int sampling_selected_identity(
    const yvex_runtime_sampling_result *result,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return result &&
           yvex_sha256_update_text(&hash, "yvex.runtime.sampling.selected-token.v2") &&
           yvex_sha256_update_u64(&hash, result->strategy) &&
           yvex_sha256_update_text(&hash, result->candidate_set_identity) &&
           yvex_sha256_update_u64(&hash, result->selected_token_id) &&
           sampling_hash_f32(&hash, result->selected_logit) &&
           sampling_hash_f64(&hash, result->selected_probability) &&
           sampling_hash_f64(&hash, result->selected_log_probability) &&
           yvex_sha256_update_text(&hash, result->rng_state_before_identity) &&
           yvex_sha256_update_text(&hash, result->rng_state_after_identity) &&
           sampling_hash_finish(&hash, output);
}

/*
 * Derive one complete per-row sampling execution identity.
 *
 * Writes only the output identity.
 */
static int sampling_execution_identity(
    const yvex_runtime_sampling_result *result,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return result &&
           yvex_sha256_update_text(&hash, "yvex.runtime.sampling.execution.v2") &&
           yvex_sha256_update_u64(&hash, result->schema_version) &&
           yvex_sha256_update_u64(&hash, (unsigned int)result->completed) &&
           yvex_sha256_update_u64(&hash,
                                  (unsigned int)result->numeric_fallback_used) &&
           yvex_sha256_update_u64(&hash, result->strategy) &&
           yvex_sha256_update_u64(&hash, result->source_phase) &&
           yvex_sha256_update_u64(&hash, result->source_position) &&
           yvex_sha256_update_u64(&hash, result->vocabulary_size) &&
           yvex_sha256_update_u64(&hash, result->values_considered) &&
           yvex_sha256_update_u64(&hash, result->candidates_before) &&
           yvex_sha256_update_u64(&hash, result->candidates_after_top_k) &&
           yvex_sha256_update_u64(&hash, result->candidates_after_min_p) &&
           yvex_sha256_update_u64(&hash,
                                  result->candidates_after_typical_p) &&
           yvex_sha256_update_u64(&hash, result->candidates_after_top_p) &&
           yvex_sha256_update_u64(&hash, result->final_candidate_count) &&
           yvex_sha256_update_u64(&hash, result->selected_token_id) &&
           yvex_sha256_update_u64(&hash, result->greedy_tie_policy) &&
           sampling_hash_f32(&hash, result->selected_logit) &&
           sampling_hash_f32(&hash, result->maximum_logit) &&
           sampling_hash_f64(&hash, result->temperature) &&
           sampling_hash_f64(&hash, result->selected_probability) &&
           sampling_hash_f64(&hash, result->selected_log_probability) &&
           yvex_sha256_update_u64(&hash, result->tied_maximum_count) &&
           yvex_sha256_update_u64(&hash, result->effective_top_k) &&
           yvex_sha256_update_u64(&hash, result->rng_draw_count) &&
           sampling_hash_f64(&hash, result->effective_top_p) &&
           sampling_hash_f64(&hash, result->effective_min_p) &&
           sampling_hash_f64(&hash, result->effective_typical_p) &&
           sampling_hash_f64(&hash, result->min_p_threshold) &&
           sampling_hash_f64(&hash, result->entropy) &&
           sampling_hash_f64(&hash, result->typical_retained_mass) &&
           sampling_hash_f64(&hash, result->top_p_retained_mass) &&
           sampling_hash_f64(&hash, result->normalization_error) &&
           yvex_sha256_update_text(&hash, result->rng_state_before_identity) &&
           yvex_sha256_update_text(&hash, result->rng_state_after_identity) &&
           yvex_sha256_update_text(&hash, result->policy_identity) &&
           yvex_sha256_update_text(&hash, result->source_identity) &&
           yvex_sha256_update_text(&hash, result->candidate_set_identity) &&
           yvex_sha256_update_text(&hash, result->selected_token_identity) &&
           sampling_hash_finish(&hash, output);
}

static int sampling_enter(yvex_runtime_sampling_context *context, yvex_error *err)
{
    unsigned int expected = 0u;
    if (context && atomic_compare_exchange_strong_explicit(
                       &context->lifecycle, &expected,
                       SAMPLING_LIFECYCLE_ACTIVE, memory_order_acq_rel,
                       memory_order_acquire))
        return YVEX_OK;
    if (context)
        (void)atomic_fetch_add_explicit(&context->admission_failures, 1ull,
                                        memory_order_relaxed);
    return sampling_refuse(err, YVEX_ERR_STATE,
                           expected & SAMPLING_LIFECYCLE_CLOSING
                               ? "sampling context is closing"
                               : "sampling context is already in use");
}

static void sampling_leave(yvex_runtime_sampling_context *context, int rc,
                           unsigned long long completed)
{
    unsigned int observed;
    if (!context) return;
    context->summary.successful_samples += completed;
    if (rc != YVEX_OK) {
        context->summary.failure_count++;
        if (rc == YVEX_ERR_CANCELLED) context->summary.cancellation_count++;
    }
    observed = atomic_load_explicit(&context->lifecycle, memory_order_acquire);
    if (observed & SAMPLING_LIFECYCLE_CLOSING) {
        if (context->drain_mutex_ready &&
            pthread_mutex_lock(&context->drain_mutex) == 0) {
            (void)atomic_fetch_and_explicit(&context->lifecycle,
                                            ~SAMPLING_LIFECYCLE_ACTIVE,
                                            memory_order_release);
            if (context->drain_condition_ready)
                (void)pthread_cond_broadcast(&context->drain_condition);
            (void)pthread_mutex_unlock(&context->drain_mutex);
            return;
        }
    }
    (void)atomic_fetch_and_explicit(&context->lifecycle,
                                    ~SAMPLING_LIFECYCLE_ACTIVE,
                                    memory_order_release);
}

/*
 * Finish field-wise identities before any selected-token publication.
 *
 * Canonical identity mismatch refuses. Caller publication and RNG commit follow success.
 */
static int sampling_result_finish(
    const yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source,
    yvex_runtime_sampling_result *result, yvex_error *err)
{
    yvex_runtime_sampling_result canonical;
    yvex_runtime_identity_copy(result->policy_identity,
                               context->policy.policy_identity);
    yvex_runtime_identity_copy(result->source_identity, source->source_identity);
    result->completed = 1;
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
    return YVEX_OK;
}

/*
 * Execute the complete-vocabulary deterministic maximum path.
 *
 * Identity derivation refuses.
 */
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

static int sampling_remove_zero_mass(
    yvex_runtime_sampling_context *context, unsigned long long *count,
    yvex_runtime_sampling_result *result, yvex_error *err)
{
    unsigned long long read, write = 0ull;
    for (read = 0ull; read < *count; ++read) {
        double probability = context->candidates[read].probability;
        if (!isfinite(probability) || probability < 0.0)
            return sampling_refuse(err, YVEX_ERR_FORMAT,
                                   "sampling zero-mass compaction found invalid probability");
        if (probability > 0.0)
            context->candidates[write++] = context->candidates[read];
    }
    if (!write || sampling_normalize(context->candidates, write,
                                     &result->normalization_error, err) != YVEX_OK)
        return sampling_refuse(err, YVEX_ERR_FORMAT,
                               "sampling has no positive final probability mass");
    *count = write;
    return YVEX_OK;
}

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
    if (rc == YVEX_OK) rc = sampling_remove_zero_mass(context, &count, result, err);
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

/*
 * Select one token transactionally from one exact complete logits source.
 *
 * One-shot wrapper over the same owned mechanism used by ordered execution.
 */
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

/*
 * Validate structural relations among all authoritative result evidence.
 *
 * Identity validation remains separate and follows this bounded structural check.
 */
static int sampling_result_structure_valid(
    const yvex_runtime_sampling_result *result)
{
    double tolerance;
    if (!result || !result->completed ||
        result->schema_version != YVEX_RUNTIME_SAMPLING_SCHEMA_V1 ||
        result->strategy > YVEX_SAMPLING_STRATEGY_STOCHASTIC ||
        !result->vocabulary_size ||
        result->values_considered != result->vocabulary_size ||
        result->candidates_before != result->vocabulary_size ||
        result->selected_token_id >= result->vocabulary_size ||
        result->greedy_tie_policy != YVEX_SAMPLING_GREEDY_LOWEST_TOKEN_ID ||
        result->candidates_after_top_k > result->vocabulary_size ||
        result->candidates_after_min_p > result->candidates_after_top_k ||
        result->candidates_after_typical_p > result->candidates_after_min_p ||
        result->candidates_after_top_p > result->candidates_after_typical_p ||
        result->final_candidate_count != result->candidates_after_top_p ||
        !result->final_candidate_count ||
        !isfinite(result->selected_logit) ||
        !isfinite(result->maximum_logit) ||
        result->selected_logit > result->maximum_logit ||
        !isfinite(result->selected_probability) ||
        result->selected_probability <= 0.0 ||
        result->selected_probability > 1.0 ||
        !isfinite(result->selected_log_probability) ||
        result->selected_log_probability != log(result->selected_probability) ||
        !isfinite(result->normalization_error) ||
        result->normalization_error < 0.0 ||
        !isfinite(result->entropy) || result->entropy < 0.0 ||
        !isfinite(result->min_p_threshold) || result->min_p_threshold < 0.0 ||
        !isfinite(result->typical_retained_mass) ||
        result->typical_retained_mass < 0.0 ||
        !isfinite(result->top_p_retained_mass) ||
        result->top_p_retained_mass < 0.0 ||
        (result->numeric_fallback_used != 0 &&
         result->numeric_fallback_used != 1))
        return 0;
    tolerance = sampling_normalization_tolerance(result->vocabulary_size);
    if (result->normalization_error > tolerance ||
        result->typical_retained_mass > 1.0 + tolerance ||
        result->top_p_retained_mass > 1.0 + tolerance)
        return 0;
    if (result->strategy == YVEX_SAMPLING_STRATEGY_GREEDY)
        return result->selected_probability == 1.0 &&
               result->selected_log_probability == 0.0 &&
               result->selected_logit == result->maximum_logit &&
               result->tied_maximum_count != 0ull &&
               result->temperature == 1.0 && result->effective_top_k == 0ull &&
               result->effective_top_p == 1.0 && result->effective_min_p == 0.0 &&
               result->effective_typical_p == 1.0 && result->rng_draw_count == 0ull &&
               !result->numeric_fallback_used && result->min_p_threshold == 0.0 &&
               result->entropy == 0.0 && result->typical_retained_mass == 0.0 &&
               result->top_p_retained_mass == 0.0 && result->normalization_error == 0.0 &&
               strcmp(result->rng_state_before_identity,
                      result->rng_state_after_identity) == 0;
    return isfinite(result->temperature) && result->temperature > 0.0 &&
           result->effective_top_k <= result->vocabulary_size &&
           isfinite(result->effective_top_p) && result->effective_top_p > 0.0 &&
           result->effective_top_p <= 1.0 && isfinite(result->effective_min_p) &&
           result->effective_min_p >= 0.0 && result->effective_min_p <= 1.0 &&
           isfinite(result->effective_typical_p) && result->effective_typical_p > 0.0 &&
           result->effective_typical_p <= 1.0 && result->tied_maximum_count == 0ull &&
           result->rng_draw_count == 1ull &&
           strcmp(result->rng_state_before_identity,
                  result->rng_state_after_identity) != 0 &&
           (result->effective_min_p != 0.0 || result->min_p_threshold == 0.0) &&
           (result->effective_typical_p != 1.0 || result->typical_retained_mass == 0.0) &&
           (result->effective_top_p != 1.0 || result->top_p_retained_mass == 0.0);
}

/*
 * Validate every authoritative field plus selected-token and execution identities after
 * publication.
 *
 * Malformed fields or identity mutation refuses.
 */
int yvex_runtime_sampling_result_validate(
    const yvex_runtime_sampling_result *result, yvex_error *err)
{
    char selected[YVEX_SHA256_HEX_CAP], execution[YVEX_SHA256_HEX_CAP];
    if (!sampling_result_structure_valid(result) ||
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

static int sampling_execution_finish(
    yvex_runtime_sampling_execution *execution,
    const yvex_runtime_sampling_result *results, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.runtime.sampling.selected-sequence.v2") ||
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
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.sampling.aggregate.v2") ||
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

/*
 * Sample ordered rows with row-atomic publication and transactional RNG progress.
 *
 * Reports exact partial progress; the failing row and its RNG transition remain absent.
 */
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

/*
 * Inspect stable workspace, policy, counters, and current RNG authority.
 *
 * Lock or identity derivation refusal clears output.
 */
int yvex_runtime_sampling_context_snapshot(
    const yvex_runtime_sampling_context *context,
    yvex_runtime_sampling_context_summary *summary, yvex_error *err)
{
    yvex_runtime_sampling_context *mutable =
        (yvex_runtime_sampling_context *)context;
    int rc;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!mutable || !summary)
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling context snapshot failed");
    rc = sampling_enter(mutable, err);
    if (rc != YVEX_OK) return rc;
    *summary = mutable->summary;
    summary->failure_count += atomic_load_explicit(
        &mutable->admission_failures, memory_order_relaxed);
    if (!sampling_rng_identity(mutable, mutable->rng_state,
                               mutable->successful_draws,
                               summary->rng_state_identity)) {
        memset(summary, 0, sizeof(*summary));
        sampling_leave(mutable, YVEX_ERR_STATE, 0ull);
        return sampling_refuse(err, YVEX_ERR_STATE,
                               "sampling snapshot RNG identity failed");
    }
    sampling_leave(mutable, YVEX_OK, 0ull);
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Release fixed sampling workspace without touching borrowed logits or runtime state.
 *
 * Idempotent after successful close; callers must stop initiating operations before transferring
 * the unique close ownership.
 */
int yvex_runtime_sampling_context_close(
    yvex_runtime_sampling_context **context, yvex_error *err)
{
    yvex_runtime_sampling_context *owner;
    unsigned int observed, desired;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owner = *context;
    observed = atomic_load_explicit(&owner->lifecycle, memory_order_acquire);
    while (!(observed & SAMPLING_LIFECYCLE_CLOSING)) {
        desired = observed | SAMPLING_LIFECYCLE_CLOSING;
        if (atomic_compare_exchange_weak_explicit(
                &owner->lifecycle, &observed, desired,
                memory_order_acq_rel, memory_order_acquire))
            break;
    }
    if (owner->drain_mutex_ready) {
        if (pthread_mutex_lock(&owner->drain_mutex) != 0)
            return sampling_refuse(err, YVEX_ERR_STATE,
                                   "sampling context close drain lock failed");
        while (atomic_load_explicit(&owner->lifecycle,
                                    memory_order_acquire) &
               SAMPLING_LIFECYCLE_ACTIVE) {
            if (!owner->drain_condition_ready ||
                pthread_cond_wait(&owner->drain_condition,
                                  &owner->drain_mutex) != 0) {
                (void)pthread_mutex_unlock(&owner->drain_mutex);
                return sampling_refuse(err, YVEX_ERR_STATE,
                                       "sampling context close drain failed");
            }
        }
        (void)pthread_mutex_unlock(&owner->drain_mutex);
    }
    if (owner->drain_condition_ready) {
        if (pthread_cond_destroy(&owner->drain_condition) != 0)
            return sampling_refuse(err, YVEX_ERR_STATE,
                                   "sampling context synchronization cleanup failed");
        owner->drain_condition_ready = 0;
    }
    if (owner->drain_mutex_ready) {
        if (pthread_mutex_destroy(&owner->drain_mutex) != 0)
            return sampling_refuse(err, YVEX_ERR_STATE,
                                   "sampling context drain cleanup failed");
        owner->drain_mutex_ready = 0;
    }
    atomic_store_explicit(&owner->lifecycle, SAMPLING_LIFECYCLE_CLOSED,
                          memory_order_release);
    yvex_core_free(owner->scratch);
    yvex_core_free(owner->candidates);
    memset(owner, 0, sizeof(*owner));
    yvex_core_free(owner);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Publish sampling readiness only after real-logits selection completes. */
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

/*
 * Execute the admitted logits workflow and sample every completed real row.
 *
 * Retains bounded logits and sampling evidence while never appending selected tokens. Preserves
 * typed logits/sampling partial progress and any retained runtime cleanup lease.
 */
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
                        "execute transformer sample");
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
        result->logits.raw_logits_count != logits_extent) {
        rc = sampling_refuse(err, YVEX_ERR_STATE,
                             "sampling operator received no complete real logits rows");
        goto finish;
    }
    rc = yvex_runtime_logits_result_validate(
        &result->logits.plan, result->logits.raw_logits,
        result->logits.raw_logits_count, result->logits.rows,
        result->logits.row_count, &result->logits.execution, err);
    if (rc != YVEX_OK) goto finish;
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

void yvex_runtime_sampling_operator_result_release(
    yvex_sampling_operator_result *result)
{
    if (!result) return;
    yvex_core_free(result->samples);
    result->samples = NULL;
    result->sample_count = 0ull;
    yvex_runtime_logits_operator_result_release(&result->logits);
}
