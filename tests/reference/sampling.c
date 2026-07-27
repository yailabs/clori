/* Owner: tests.reference.sampling.
 * Owns: independent sampling arithmetic used only as a conformance oracle.
 * Does not own: production policy, identities, lifecycle, CLI, or capability.
 * Invariants: no production sampling helper is called.
 * Boundary: test-only greedy, filtering, PCG, and categorical selection.
 * Purpose: independently verify canonical sampling results.
 * Inputs: finite logits, sealed policy, and test-owned RNG state.
 * Effects: allocates temporary test storage and advances only the supplied oracle RNG.
 * Failure: invalid arithmetic returns false without publishing a result. */
#include "tests/reference/sampling.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define REF_PCG_MULTIPLIER UINT64_C(6364136223846793005)
#define REF_PCG_INCREMENT UINT64_C(1442695040888963407)

typedef struct {
    unsigned int token;
    double probability, deviation;
} ref_candidate;

/* Purpose: independently advance PCG-XSH-RR 64/32. */
static uint32_t ref_pcg_next(uint64_t *state, uint64_t increment)
{
    uint64_t old = *state;
    uint32_t shifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    unsigned int rotation = (unsigned int)(old >> 59u);
    *state = old * REF_PCG_MULTIPLIER + increment;
    return (shifted >> rotation) | (shifted << ((0u - rotation) & 31u));
}

/* Purpose: independently establish the fixed-stream PCG seed contract. */
void yvex_test_sampling_reference_seed(unsigned long long seed,
                                       yvex_test_sampling_rng *rng)
{
    if (!rng) return;
    memset(rng, 0, sizeof(*rng));
    rng->increment = REF_PCG_INCREMENT;
    (void)ref_pcg_next(&rng->state, rng->increment);
    rng->state += (uint64_t)seed;
    (void)ref_pcg_next(&rng->state, rng->increment);
}

/* Purpose: impose probability descending and token ascending ties. */
static int ref_probability_compare(const void *left, const void *right)
{
    const ref_candidate *a = (const ref_candidate *)left;
    const ref_candidate *b = (const ref_candidate *)right;
    if (a->probability > b->probability) return -1;
    if (a->probability < b->probability) return 1;
    return a->token < b->token ? -1 : a->token > b->token;
}

/* Purpose: impose typical deviation, probability, and token ordering. */
static int ref_typical_compare(const void *left, const void *right)
{
    const ref_candidate *a = (const ref_candidate *)left;
    const ref_candidate *b = (const ref_candidate *)right;
    if (a->deviation < b->deviation) return -1;
    if (a->deviation > b->deviation) return 1;
    return ref_probability_compare(left, right);
}

/* Purpose: impose ascending token order for the categorical draw. */
static int ref_token_compare(const void *left, const void *right)
{
    const ref_candidate *a = (const ref_candidate *)left;
    const ref_candidate *b = (const ref_candidate *)right;
    return a->token < b->token ? -1 : a->token > b->token;
}

/* Purpose: independently normalize one nonempty candidate set. */
static int ref_normalize(ref_candidate *candidates, unsigned long long count)
{
    double total = 0.0;
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        if (!isfinite(candidates[index].probability) ||
            candidates[index].probability < 0.0)
            return 0;
        total += candidates[index].probability;
    }
    if (!isfinite(total) || total <= 0.0) return 0;
    for (index = 0ull; index < count; ++index)
        candidates[index].probability /= total;
    return 1;
}

/* Purpose: independently produce stable full-vocabulary probabilities. */
static int ref_softmax(const float *logits, unsigned long long count,
                       double temperature, ref_candidate *candidates)
{
    double maximum = -INFINITY, total = 0.0;
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        double scaled = (double)logits[index] / temperature;
        if (!isfinite(scaled)) return 0;
        if (scaled > maximum) maximum = scaled;
    }
    for (index = 0ull; index < count; ++index) {
        candidates[index].token = (unsigned int)index;
        candidates[index].probability = exp((double)logits[index] / temperature - maximum);
        candidates[index].deviation = 0.0;
        if (!isfinite(candidates[index].probability)) return 0;
        total += candidates[index].probability;
    }
    return isfinite(total) && total > 0.0 && ref_normalize(candidates, count);
}

/* Purpose: independently remove exact zero mass before entropy-bearing filters. */
static unsigned long long ref_compact_positive(ref_candidate *candidates,
                                               unsigned long long count)
{
    unsigned long long read, write = 0ull;
    for (read = 0ull; read < count; ++read) {
        if (!isfinite(candidates[read].probability) ||
            candidates[read].probability < 0.0)
            return 0ull;
        if (candidates[read].probability > 0.0)
            candidates[write++] = candidates[read];
    }
    return write && ref_normalize(candidates, write) ? write : 0ull;
}

/* Purpose: independently retain one cumulative prefix including its crossing candidate. */
static unsigned long long ref_mass_prefix(ref_candidate *candidates,
                                          unsigned long long count,
                                          double requested)
{
    double mass = 0.0;
    unsigned long long keep;
    for (keep = 0ull; keep < count; ++keep) {
        mass += candidates[keep].probability;
        if (mass >= requested) return keep + 1ull;
    }
    return count;
}

/* Purpose: independently select greedily or through the canonical stochastic pipeline. */
int yvex_test_sampling_reference_select(
    const float *logits, unsigned long long count,
    const yvex_runtime_sampling_policy *policy,
    yvex_test_sampling_rng *rng,
    yvex_test_sampling_reference_result *result)
{
    ref_candidate *candidates;
    unsigned long long index, write, keep = count;
    double entropy = 0.0, maximum_probability, threshold, cumulative = 0.0;
    uint32_t random_value;
    if (result) memset(result, 0, sizeof(*result));
    if (!logits || !count || !policy || !rng || !result) return 0;
    if (policy->strategy == YVEX_SAMPLING_STRATEGY_GREEDY) {
        unsigned int selected = 0u;
        for (index = 0ull; index < count; ++index) {
            if (!isfinite(logits[index])) return 0;
            if (logits[index] > logits[selected]) selected = (unsigned int)index;
        }
        result->selected_token_id = selected;
        result->selected_probability = 1.0;
        result->candidate_count = count;
        result->rng_state_after = rng->state;
        result->rng_draws_after = rng->draws;
        return 1;
    }
    candidates = (ref_candidate *)calloc((size_t)count, sizeof(*candidates));
    if (!candidates || !ref_softmax(logits, count, policy->temperature, candidates)) {
        free(candidates);
        return 0;
    }
    keep = ref_compact_positive(candidates, keep);
    if (!keep) goto fail;
    if (policy->top_k && policy->top_k < keep) {
        qsort(candidates, (size_t)keep, sizeof(*candidates), ref_probability_compare);
        keep = policy->top_k;
        if (!ref_normalize(candidates, keep)) goto fail;
    }
    if (policy->min_p > 0.0) {
        maximum_probability = 0.0;
        for (index = 0ull; index < keep; ++index)
            if (candidates[index].probability > maximum_probability)
                maximum_probability = candidates[index].probability;
        threshold = policy->min_p * maximum_probability;
        for (index = write = 0ull; index < keep; ++index)
            if (candidates[index].probability >= threshold)
                candidates[write++] = candidates[index];
        keep = write;
        if (!ref_normalize(candidates, keep)) goto fail;
    }
    if (policy->typical_p < 1.0) {
        for (index = 0ull; index < keep; ++index) {
            if (candidates[index].probability <= 0.0) goto fail;
            entropy -= candidates[index].probability * log(candidates[index].probability);
        }
        for (index = 0ull; index < keep; ++index)
            candidates[index].deviation =
                fabs(-log(candidates[index].probability) - entropy);
        qsort(candidates, (size_t)keep, sizeof(*candidates), ref_typical_compare);
        keep = ref_mass_prefix(candidates, keep, policy->typical_p);
        if (!ref_normalize(candidates, keep)) goto fail;
    }
    if (policy->top_p < 1.0) {
        qsort(candidates, (size_t)keep, sizeof(*candidates), ref_probability_compare);
        keep = ref_mass_prefix(candidates, keep, policy->top_p);
        if (!ref_normalize(candidates, keep)) goto fail;
    }
    keep = ref_compact_positive(candidates, keep);
    if (!keep) goto fail;
    qsort(candidates, (size_t)keep, sizeof(*candidates), ref_token_compare);
    random_value = ref_pcg_next(&rng->state, rng->increment);
    rng->draws++;
    threshold = ((double)random_value + 0.5) / 4294967296.0;
    for (index = 0ull; index < keep; ++index) {
        cumulative += candidates[index].probability;
        if (threshold < cumulative) break;
    }
    if (index == keep) index = keep - 1ull;
    result->selected_token_id = candidates[index].token;
    result->selected_probability = candidates[index].probability;
    result->candidate_count = keep;
    result->rng_state_after = rng->state;
    result->rng_draws_after = rng->draws;
    free(candidates);
    return 1;
fail:
    free(candidates);
    return 0;
}
