/* Test-only independent sampling arithmetic; production never includes this header. */
#ifndef TESTS_REFERENCE_SAMPLING_H_INCLUDED
#define TESTS_REFERENCE_SAMPLING_H_INCLUDED

#include <stdint.h>
#include <yvex/internal/sampling.h>

typedef struct {
    uint64_t state, increment;
    unsigned long long draws;
} yvex_test_sampling_rng;

typedef struct {
    unsigned int selected_token_id;
    double selected_probability;
    unsigned long long candidate_count;
    uint64_t rng_state_after;
    unsigned long long rng_draws_after;
} yvex_test_sampling_reference_result;

void yvex_test_sampling_reference_seed(unsigned long long seed,
                                       yvex_test_sampling_rng *rng);
int yvex_test_sampling_reference_select(
    const float *logits, unsigned long long count,
    const yvex_runtime_sampling_policy *policy,
    yvex_test_sampling_rng *rng,
    yvex_test_sampling_reference_result *result);

#endif
