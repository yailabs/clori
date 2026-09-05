/*
 * Fixtures are sealed as complete logits rows before production admission. Unit/integration
 * evidence over the common real-logits sampling owner.
 */
#include <yvex/internal/sampling.h>

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tests/reference/sampling.h"
#include "tests/test.h"

static void sampling_test_identity(char output[YVEX_SHA256_HEX_CAP], char digit)
{
    memset(output, digit, YVEX_SHA256_HEX_CAP - 1u);
    output[YVEX_SHA256_HEX_CAP - 1u] = '\0';
}

static int sampling_test_raw_digest(const float *values, unsigned long long count,
                                    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.raw-logits.v1")) return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

/* Seal one synthetic but fully identity-bound logits publication. */
static int sampling_test_row(const yvex_runtime_logits_plan_summary *plan,
                             const float *values, unsigned long long count,
                             unsigned long long position,
                             yvex_runtime_logits_row_result *row)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    memset(row, 0, sizeof(*row));
    row->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    row->completed = 1;
    row->host_values_available = 1;
    row->finite_count_available = 1;
    row->range_available = 1;
    row->raw_digest_available = 1;
    row->evidence_profile = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    row->source_phase = position ? YVEX_LOGITS_SOURCE_DECODE : YVEX_LOGITS_SOURCE_PREFILL;
    row->source_position = position;
    row->vocabulary_size = row->logits_count = row->finite_count = count;
    row->minimum_logit = row->maximum_logit = values[0];
    for (index = 0ull; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
        if (values[index] < row->minimum_logit) row->minimum_logit = values[index];
        if (values[index] > row->maximum_logit) row->maximum_logit = values[index];
    }
    sampling_test_identity(row->source_hidden_digest, 'b');
    sampling_test_identity(row->output_head_residency_identity, 'c');
    sampling_test_identity(row->backend_execution_identity, 'd');
    yvex_runtime_identity_copy(row->output_head_plan_identity,
                               plan->output_head_plan_identity);
    if (!sampling_test_raw_digest(values, count, row->raw_logits_digest)) return 0;
    yvex_sha256_init(&hash);
    row->full_array_host_scan_bytes = count * sizeof(float);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.row.v2") ||
        !yvex_sha256_update_u64(&hash, row->source_phase) ||
        !yvex_sha256_update_u64(&hash, row->source_position) ||
        !yvex_sha256_update_u64(&hash, row->vocabulary_size) ||
        !yvex_sha256_update_u64(&hash, row->host_values_available) ||
        !yvex_sha256_update_u64(&hash, row->device_values_available) ||
        !yvex_sha256_update_u64(&hash, row->evidence_profile) ||
        !yvex_sha256_update_text(&hash, row->source_hidden_digest) ||
        !yvex_sha256_update_text(&hash, row->output_head_plan_identity) ||
        !yvex_sha256_update_text(&hash, row->output_head_residency_identity) ||
        !yvex_sha256_update_text(&hash, row->backend_execution_identity) ||
        !yvex_sha256_update_text(&hash, row->raw_logits_digest) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, row->logits_row_identity);
    return 1;
}

static void sampling_test_plan(yvex_runtime_logits_plan_summary *plan,
                               unsigned long long vocabulary_size)
{
    memset(plan, 0, sizeof(*plan));
    plan->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    plan->vocabulary_size = plan->row_count = vocabulary_size;
    plan->row_width = plan->hidden_width = 4ull;
    sampling_test_identity(plan->output_head_plan_identity, 'a');
}

static int sampling_test_select_fixture(
    const float *logits, unsigned long long count,
    yvex_runtime_sampling_policy policy,
    yvex_runtime_sampling_result *result, yvex_error *err)
{
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_options options = {
        .maximum_rows = 1ull};
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_source source;
    yvex_error cleanup;
    int rc, close_rc;
    sampling_test_plan(&plan, count);
    options.maximum_vocabulary_size = count;
    if (!sampling_test_row(&plan, logits, count, 1ull, &row))
        return YVEX_ERR_FORMAT;
    rc = yvex_runtime_sampling_policy_seal(&policy, count, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_open(
            &context, &plan, &policy, &options, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_source_from_logits(
            context, &source, logits, count, &row, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_select(context, NULL, &source, result, err);
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_sampling_context_close(&context, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        *err = cleanup;
    }
    return rc;
}

static yvex_runtime_sampling_policy sampling_test_neutral_stochastic(void)
{
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0,
        .seed_present = 1, .seed = 42ull};
    return policy;
}

static int sampling_test_numeric_boundaries(void)
{
    const unsigned long long vocabulary = 129280ull;
    const float underflow[4] = {0.0f, -1000.0f, -1001.0f, -1002.0f};
    yvex_runtime_sampling_policy policy = sampling_test_neutral_stochastic();
    yvex_runtime_sampling_result result;
    yvex_test_sampling_rng rng;
    yvex_test_sampling_reference_result reference;
    yvex_error err;
    float *uniform = (float *)calloc((size_t)vocabulary, sizeof(*uniform));
    YVEX_TEST_ASSERT(uniform != NULL, "uniform full-vocabulary fixture allocates");
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         uniform, vocabulary, policy, &result, &err) == YVEX_OK &&
                         result.completed && result.values_considered == vocabulary &&
                         result.final_candidate_count == vocabulary &&
                         result.normalization_error >= 0.0,
                     "129280 equal logits normalize and sample successfully");
    free(uniform);
    policy.typical_p = 0.9;
    yvex_test_sampling_reference_seed(policy.seed, &rng);
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         underflow, 4ull, policy, &result, &err) == YVEX_OK &&
                         result.completed && result.selected_token_id == 0u &&
                         result.final_candidate_count == 1ull && isfinite(result.entropy) &&
                         yvex_test_sampling_reference_select(
                             underflow, 4ull, &policy, &rng, &reference) &&
                         reference.selected_token_id == result.selected_token_id &&
                         reference.candidate_count == result.final_candidate_count,
                     "typical-p removes exact zero softmax mass before logarithms");
    return 0;
}

/* Prove explicit strategy validation and canonical identity mutation. */
static int sampling_test_policy(void)
{
    yvex_runtime_sampling_policy greedy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0};
    yvex_runtime_sampling_policy stochastic = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC,
        .temperature = 0.8, .top_k = 4ull, .top_p = 0.9,
        .min_p = 0.05, .typical_p = 0.95, .seed_present = 1, .seed = 0ull};
    char original[YVEX_SHA256_HEX_CAP];
    yvex_error err;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&greedy, 8ull, &err) == YVEX_OK,
                     "explicit neutral greedy policy seals");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&stochastic, 8ull, &err) == YVEX_OK,
                     "seed zero is valid for stochastic sampling");
    yvex_runtime_identity_copy(original, stochastic.policy_identity);
    stochastic.temperature = 0.7;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&stochastic, 8ull, &err) == YVEX_OK &&
                         strcmp(original, stochastic.policy_identity) != 0,
                     "temperature mutation changes policy identity");
    greedy.top_k = 1ull;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&greedy, 8ull, &err) ==
                         YVEX_ERR_FORMAT,
                     "greedy refuses hidden stochastic filters");
    stochastic.seed_present = 0;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&stochastic, 8ull, &err) ==
                         YVEX_ERR_FORMAT,
                     "stochastic sampling refuses an absent explicit seed");
    stochastic.seed_present = 1;
    stochastic.temperature = 0.0;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&stochastic, 8ull, &err) ==
                         YVEX_ERR_FORMAT,
                     "temperature zero cannot switch sampling strategy");
    stochastic.temperature = 1.0;
    stochastic.top_p = 0.0;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&stochastic, 8ull, &err) ==
                         YVEX_ERR_FORMAT,
                     "zero top-p refuses instead of removing every candidate");
    stochastic.top_p = 1.0;
    stochastic.min_p = 1.1;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&stochastic, 8ull, &err) ==
                         YVEX_ERR_FORMAT,
                     "out-of-range min-p refuses");
    stochastic.min_p = 0.0;
    stochastic.typical_p = 0.0;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&stochastic, 8ull, &err) ==
                         YVEX_ERR_FORMAT,
                     "zero typical-p refuses");
    return 0;
}

static int sampling_test_greedy(void)
{
    float logits[6] = {-9.0f, 1.0f, 4.0f, 4.0f, -2.0f, 3.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0};
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 6ull, .maximum_rows = 1ull};
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_result result;
    yvex_runtime_sampling_context_summary before, after;
    yvex_error err;
    sampling_test_plan(&plan, 6ull);
    YVEX_TEST_ASSERT(sampling_test_row(&plan, logits, 6ull, 0ull, &row),
                     "test logits row seals");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&policy, 6ull, &err) == YVEX_OK &&
                         yvex_runtime_sampling_context_open(
                             &context, &plan, &policy, &options, &err) == YVEX_OK,
                     "greedy context opens");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_source_from_logits(
                         context, &source, logits, 6ull, &row, &err) == YVEX_OK,
                     "complete logits row admits as a sampling source");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_snapshot(
                         context, &before, &err) == YVEX_OK,
                     "greedy context snapshots before selection");
    if (yvex_runtime_sampling_select(context, NULL, &source, &result, &err) != YVEX_OK) {
        fprintf(stderr, "sampling greedy refusal: %s\n", yvex_error_message(&err));
        YVEX_TEST_FAIL("complete admitted logits row samples greedily");
    }
    YVEX_TEST_ASSERT(result.selected_token_id == 2u &&
                         result.values_considered == 6ull &&
                         result.tied_maximum_count == 2ull &&
                         result.final_candidate_count == 6ull &&
                         result.rng_draw_count == 0ull,
                     "greedy scans all values and lowest token ID wins ties");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
                         before.stochastic_draws == after.stochastic_draws &&
                         before.workspace_generation == after.workspace_generation &&
                         after.warm_workspace_allocations == 0ull,
                     "greedy reuses workspace without RNG advancement");
    logits[2] = -10.0f;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_select(context, NULL, &source, &result, &err) ==
                         YVEX_ERR_FORMAT && !result.completed,
                     "unsealed logits mutation refuses before selection");
    logits[2] = NAN;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_select(context, NULL, &source, &result, &err) ==
                         YVEX_ERR_FORMAT && !result.completed,
                     "non-finite logits refuse before candidate publication");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_close(&context, &err) == YVEX_OK && !context,
                     "greedy context closes deterministically");
    return 0;
}

static int sampling_test_padded_output_vocabulary(void)
{
    const float logits[6] = {1.0f, 5.0f, 4.0f, 2.0f, 100.0f, 99.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0};
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 4ull,
        .selection_vocabulary_size = 4ull,
        .maximum_rows = 1ull};
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_result result;
    yvex_error err;

    sampling_test_plan(&plan, 6ull);
    YVEX_TEST_ASSERT(
        sampling_test_row(&plan, logits, 6ull, 0ull, &row) &&
            yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_open(
                &context, &plan, &policy, &options, &err) == YVEX_OK &&
            yvex_runtime_sampling_source_from_logits(
                context, &source, logits, 6ull, &row, &err) == YVEX_OK &&
            source.vocabulary_size == 4ull && source.logits_stride == 6ull &&
            yvex_runtime_sampling_select(
                context, NULL, &source, &result, &err) == YVEX_OK &&
            result.selected_token_id == 1u && result.values_considered == 4ull,
        "logical tokenizer vocabulary excludes padded output-head rows from sampling");
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_context_close(&context, &err) == YVEX_OK,
        "padded output-vocabulary sampling context closes");
    options.selection_vocabulary_size = 7ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_context_open(
            &context, &plan, &policy, &options, &err) == YVEX_ERR_NOMEM &&
            !context,
        "logical selection vocabulary cannot exceed the physical output envelope");
    return 0;
}

/* Exercise each canonical filter, endpoint tie order, and stable softmax edge. */
static int sampling_test_filter_matrix(void)
{
    const float equal[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float maxima[4] = {2.0f, 2.0f, 0.0f, -1.0f};
    const float large[4] = {1000.0f, 999.0f, -999.0f, -1000.0f};
    const float first_max[4] = {5.0f, 1.0f, 0.0f, -1.0f};
    const float last_max[4] = {-1.0f, 0.0f, 1.0f, 5.0f};
    yvex_runtime_sampling_policy policy;
    yvex_runtime_sampling_result neutral, filtered, endpoint;
    yvex_error err;
    policy = sampling_test_neutral_stochastic();
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         equal, 4ull, policy, &neutral, &err) == YVEX_OK &&
                         neutral.final_candidate_count == 4ull &&
                         fabs(neutral.selected_probability - 0.25) < 1.0e-15 &&
                         neutral.normalization_error <= 1.0e-12,
                     "equal-logit stable softmax preserves four normalized candidates");
    policy.top_k = 1ull;
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         equal, 4ull, policy, &filtered, &err) == YVEX_OK &&
                         filtered.final_candidate_count == 1ull &&
                         filtered.selected_token_id == 0u,
                     "top-k one resolves equal probabilities by lowest token ID");
    policy.top_k = 4ull;
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         equal, 4ull, policy, &filtered, &err) == YVEX_OK &&
                         filtered.final_candidate_count == 4ull,
                     "top-k equal to vocabulary extent is neutral");
    policy = sampling_test_neutral_stochastic();
    policy.min_p = 1.0;
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         maxima, 4ull, policy, &filtered, &err) == YVEX_OK &&
                         filtered.final_candidate_count == 2ull,
                     "inclusive min-p threshold retains every tied maximum");
    policy = sampling_test_neutral_stochastic();
    policy.typical_p = 0.5;
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         equal, 4ull, policy, &filtered, &err) == YVEX_OK &&
                         filtered.final_candidate_count == 2ull &&
                         fabs(filtered.entropy - log(4.0)) < 1.0e-15,
                     "typical-p includes the cumulative crossing candidate with stable ties");
    policy = sampling_test_neutral_stochastic();
    policy.top_p = 0.5;
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         equal, 4ull, policy, &filtered, &err) == YVEX_OK &&
                         filtered.final_candidate_count == 2ull,
                     "top-p includes the exact cumulative crossing candidate");
    policy = sampling_test_neutral_stochastic();
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         large, 4ull, policy, &filtered, &err) == YVEX_OK &&
                         filtered.final_candidate_count >= 1ull,
                     "maximum subtraction stabilizes large positive and negative logits");
    policy = (yvex_runtime_sampling_policy){
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0};
    YVEX_TEST_ASSERT(sampling_test_select_fixture(
                         first_max, 4ull, policy, &endpoint, &err) == YVEX_OK &&
                         endpoint.selected_token_id == 0u &&
                         sampling_test_select_fixture(
                             last_max, 4ull, policy, &endpoint, &err) == YVEX_OK &&
                         endpoint.selected_token_id == 3u,
                     "greedy unique maxima at both vocabulary endpoints are observed");
    YVEX_TEST_ASSERT(strcmp(neutral.candidate_set_identity,
                            filtered.candidate_set_identity) != 0,
                     "effective policy and source changes alter candidate identity");
    return 0;
}

static int sampling_test_stochastic(void)
{
    const float logits[8] = {3.0f, 2.0f, 1.5f, 1.0f, 0.5f, 0.0f, -1.0f, -2.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC,
        .temperature = 0.8, .top_k = 6ull, .top_p = 0.9,
        .min_p = 0.05, .typical_p = 0.95, .seed_present = 1, .seed = 42ull};
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 8ull, .maximum_rows = 2ull};
    yvex_runtime_sampling_context *first = NULL, *second = NULL;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_result a, b;
    yvex_runtime_sampling_context_summary summary;
    yvex_test_sampling_rng rng;
    yvex_test_sampling_reference_result reference, divergent;
    yvex_error err;
    sampling_test_plan(&plan, 8ull);
    YVEX_TEST_ASSERT(sampling_test_row(&plan, logits, 8ull, 1ull, &row) &&
                         yvex_runtime_sampling_policy_seal(&policy, 8ull, &err) == YVEX_OK,
                     "filtered stochastic fixture seals");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_open(
                         &first, &plan, &policy, &options, &err) == YVEX_OK &&
                         yvex_runtime_sampling_context_open(
                             &second, &plan, &policy, &options, &err) == YVEX_OK &&
                         yvex_runtime_sampling_source_from_logits(
                             first, &source, logits, 8ull, &row, &err) == YVEX_OK,
                     "isolated stochastic contexts open over one source");
    yvex_test_sampling_reference_seed(42ull, &rng);
    YVEX_TEST_ASSERT(yvex_runtime_sampling_select(first, NULL, &source, &a, &err) == YVEX_OK &&
                         yvex_runtime_sampling_select(second, NULL, &source, &b, &err) == YVEX_OK &&
                         yvex_test_sampling_reference_select(
                             logits, 8ull, &policy, &rng, &reference),
                     "production and independent filtered samplers execute");
    YVEX_TEST_ASSERT(yvex_test_sampling_reference_first_divergence(
                         &a, &reference, 1.0e-15) == NULL &&
                         a.selected_token_id == b.selected_token_id &&
                         strcmp(a.selected_token_identity, b.selected_token_identity) == 0,
                     "reference and same-seed contexts match through every sampling stage");
    divergent = reference;
    divergent.candidates_after_min_p++;
    YVEX_TEST_ASSERT(strcmp(yvex_test_sampling_reference_first_divergence(
                                &a, &divergent, 1.0e-15),
                            "min-p") == 0,
                     "sampling differential names the first mismatching stage");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_snapshot(first, &summary, &err) == YVEX_OK &&
                         summary.stochastic_draws == 1ull &&
                         summary.successful_samples == 1ull &&
                         summary.warm_workspace_allocations == 0ull,
                     "one successful stochastic sample commits exactly one warm draw");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_close(&first, &err) == YVEX_OK &&
                         yvex_runtime_sampling_context_close(&second, &err) == YVEX_OK,
                     "stochastic contexts close independently");
    return 0;
}

static int sampling_test_rng_vectors(void)
{
    const float logits[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy first_policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0,
        .seed_present = 1, .seed = 42ull};
    yvex_runtime_sampling_policy second_policy = first_policy;
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 8ull, .maximum_rows = 1ull};
    yvex_runtime_sampling_context *first = NULL, *second = NULL;
    yvex_runtime_sampling_source first_source, second_source;
    yvex_runtime_sampling_result first_result, second_result;
    yvex_error err;
    second_policy.seed = 41ull;
    sampling_test_plan(&plan, 8ull);
    YVEX_TEST_ASSERT(sampling_test_row(&plan, logits, 8ull, 1ull, &row) &&
                         yvex_runtime_sampling_policy_seal(
                             &first_policy, 8ull, &err) == YVEX_OK &&
                         yvex_runtime_sampling_policy_seal(
                             &second_policy, 8ull, &err) == YVEX_OK &&
                         yvex_runtime_sampling_context_open(
                             &first, &plan, &first_policy, &options, &err) == YVEX_OK &&
                         yvex_runtime_sampling_context_open(
                             &second, &plan, &second_policy, &options, &err) == YVEX_OK,
                     "versioned PCG fixture contexts open");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_source_from_logits(
                         first, &first_source, logits, 8ull, &row, &err) == YVEX_OK &&
                         yvex_runtime_sampling_source_from_logits(
                             second, &second_source, logits, 8ull, &row, &err) == YVEX_OK &&
                         yvex_runtime_sampling_select(
                             first, NULL, &first_source, &first_result, &err) == YVEX_OK &&
                         yvex_runtime_sampling_select(
                             second, NULL, &second_source, &second_result, &err) == YVEX_OK,
                     "versioned PCG fixture selections execute");
    YVEX_TEST_ASSERT(first_result.selected_token_id == 6u &&
                         second_result.selected_token_id == 1u &&
                         strcmp(first_result.rng_state_after_identity,
                                second_result.rng_state_after_identity) != 0,
                     "fixed PCG vectors and distinct seed sequence are stable");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_close(&first, &err) == YVEX_OK &&
                         yvex_runtime_sampling_context_close(&second, &err) == YVEX_OK,
                     "PCG fixture contexts close");
    return 0;
}

static int sampling_test_checkpoint(void)
{
    const float logits[8] = {3.0f, 2.0f, 1.5f, 1.0f, 0.5f, 0.0f, -1.0f, -2.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy policy = sampling_test_neutral_stochastic();
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 8ull, .maximum_rows = 2ull};
    yvex_runtime_sampling_context *context = NULL, *incompatible = NULL;
    yvex_runtime_sampling_transaction *transaction = NULL;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_result initial, expected, advanced, restored;
    yvex_runtime_sampling_checkpoint checkpoint, corrupt;
    yvex_runtime_sampling_context_summary before, after;
    yvex_error err;
    sampling_test_plan(&plan, 8ull);
    YVEX_TEST_ASSERT(
        sampling_test_row(&plan, logits, 8ull, 1ull, &row) &&
            yvex_runtime_sampling_policy_seal(&policy, 8ull, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_open(
                &context, &plan, &policy, &options, &err) == YVEX_OK &&
            yvex_runtime_sampling_source_from_logits(
                context, &source, logits, 8ull, &row, &err) == YVEX_OK &&
            yvex_runtime_sampling_select(
                context, NULL, &source, &initial, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_checkpoint(
                context, &checkpoint, &err) == YVEX_OK &&
            yvex_runtime_sampling_select(
                context, NULL, &source, &expected, &err) == YVEX_OK &&
            yvex_runtime_sampling_select(
                context, NULL, &source, &advanced, &err) == YVEX_OK,
        "sampling checkpoint fixture advances beyond a sealed RNG state");
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_context_restore(context, &checkpoint, &err) == YVEX_OK &&
            yvex_runtime_sampling_select(
                context, NULL, &source, &restored, &err) == YVEX_OK &&
            expected.selected_token_id == restored.selected_token_id &&
            strcmp(expected.execution_identity, restored.execution_identity) == 0 &&
            strcmp(expected.rng_state_after_identity,
                   restored.rng_state_after_identity) == 0,
        "restored sampling checkpoint reproduces the exact next stochastic selection");
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_context_snapshot(context, &before, &err) == YVEX_OK,
        "sampling checkpoint refusal fixture snapshots its committed authority");
    corrupt = checkpoint;
    corrupt.rng_state++;
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_context_restore(context, &corrupt, &err) == YVEX_ERR_FORMAT &&
            yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
            strcmp(before.rng_state_identity, after.rng_state_identity) == 0,
        "corrupt sampling checkpoint refuses without changing RNG authority");
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_transaction_begin(context, &transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_checkpoint(context, &corrupt, &err) ==
                YVEX_ERR_STATE &&
            yvex_runtime_sampling_context_restore(context, &checkpoint, &err) ==
                YVEX_ERR_STATE &&
            yvex_runtime_sampling_transaction_abort(&transaction, &err) == YVEX_OK,
        "open RNG transaction prevents checkpoint capture and restore");
    policy.seed++;
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_policy_seal(&policy, 8ull, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_open(
                &incompatible, &plan, &policy, &options, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_restore(incompatible, &checkpoint, &err) ==
                YVEX_ERR_FORMAT,
        "sampling checkpoint refuses a different immutable policy identity");
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_context_close(&context, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_close(&incompatible, &err) == YVEX_OK,
        "sampling checkpoint fixture contexts close");
    return 0;
}

static int sampling_test_rng_transactions(void)
{
    const float logits[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy policy = sampling_test_neutral_stochastic();
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 8ull, .maximum_rows = 8ull};
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_transaction *transaction = NULL, *stale = NULL;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_result aborted, retried;
    yvex_runtime_sampling_uniform_result draw;
    yvex_runtime_sampling_context_summary before, after;
    double values[2] = {0.0, 0.0}, stale_value = 0.0;
    yvex_error err;

    sampling_test_plan(&plan, 8ull);
    YVEX_TEST_ASSERT(
        sampling_test_row(&plan, logits, 8ull, 1ull, &row) &&
            yvex_runtime_sampling_policy_seal(&policy, 8ull, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_open(
                &context, &plan, &policy, &options, &err) == YVEX_OK &&
            yvex_runtime_sampling_source_from_logits(
                context, &source, logits, 8ull, &row, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_snapshot(context, &before, &err) == YVEX_OK,
        "transactional RNG fixture opens");

    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_transaction_begin(context, &transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_uniforms(
                transaction, values, 2ull, &draw, &err) == YVEX_OK &&
            draw.completed && draw.draw_count == 2ull &&
            yvex_runtime_sampling_transaction_abort(&transaction, &err) == YVEX_OK &&
            !transaction &&
            yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
            after.stochastic_draws == before.stochastic_draws &&
            strcmp(after.rng_state_identity, before.rng_state_identity) == 0,
        "aborted RNG draws leave the sampling authority unchanged");

    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_transaction_begin(context, &transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_select(
                context, transaction, &source, &aborted, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_abort(&transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
            after.successful_samples == before.successful_samples &&
            after.stochastic_draws == before.stochastic_draws &&
            yvex_runtime_sampling_transaction_begin(context, &transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_select(
                context, transaction, &source, &retried, &err) == YVEX_OK &&
            strcmp(aborted.execution_identity, retried.execution_identity) == 0 &&
            yvex_runtime_sampling_transaction_prepare_commit(transaction, &err) == YVEX_OK,
        "transaction-owned selection retries the exact staged RNG draw");
    yvex_runtime_sampling_transaction_publish_commit(&transaction);
    YVEX_TEST_ASSERT(
        !transaction && yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
            after.successful_samples == before.successful_samples + 1ull &&
            after.stochastic_draws == before.stochastic_draws + 1ull &&
            strcmp(after.rng_state_identity, retried.rng_state_after_identity) == 0,
        "transaction-owned selection publishes exactly one staged RNG transition");
    before = after;

    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_transaction_begin(context, &transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_uniforms(
                transaction, values, 1ull, &draw, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_prepare_commit(transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_abort(&transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
            after.stochastic_draws == before.stochastic_draws &&
            strcmp(after.rng_state_identity, before.rng_state_identity) == 0,
        "cancelled prepared RNG publication preserves the earlier state");

    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_transaction_begin(context, &transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_begin(context, &stale, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_uniforms(
                transaction, values, 1ull, &draw, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_uniforms(
                stale, &stale_value, 1ull, &draw, &err) == YVEX_OK &&
            values[0] == stale_value &&
            yvex_runtime_sampling_transaction_prepare_commit(transaction, &err) == YVEX_OK,
        "concurrent transactions retain one immutable base state");
    yvex_runtime_sampling_transaction_publish_commit(&transaction);
    YVEX_TEST_ASSERT(
        !transaction &&
            yvex_runtime_sampling_select(
                context, stale, &source, &retried, &err) == YVEX_ERR_STATE && !retried.completed &&
            yvex_runtime_sampling_transaction_prepare_commit(stale, &err) == YVEX_ERR_STATE &&
            yvex_runtime_sampling_transaction_abort(&stale, &err) == YVEX_OK && !stale &&
            yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
            after.stochastic_draws == before.stochastic_draws + 1ull &&
            strcmp(after.rng_state_identity, before.rng_state_identity) != 0,
        "a stale RNG transaction cannot overwrite a published successor");

    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_transaction_begin(context, &transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_uniforms(
                transaction, values, 2ull, &draw, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_prepare_commit(transaction, &err) == YVEX_OK,
        "RNG transaction prepares a bounded multi-draw publication");
    yvex_runtime_sampling_transaction_publish_commit(&transaction);
    YVEX_TEST_ASSERT(
        !transaction &&
            yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
            after.stochastic_draws == before.stochastic_draws + 3ull &&
            yvex_runtime_sampling_context_close(&context, &err) == YVEX_OK && !context,
        "published RNG transactions advance exactly their committed draws");
    return 0;
}

typedef struct {
    yvex_runtime_sampling_context *context;
    const yvex_runtime_sampling_source *source;
    int attempted, refusal;
} sampling_test_reentry;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int entered, release;
} sampling_test_thread_gate;

typedef struct {
    yvex_runtime_sampling_context *context;
    const yvex_runtime_sampling_source *source;
    sampling_test_thread_gate *gate;
    unsigned long long attempts;
    int rc, closing_refusal;
} sampling_test_select_thread;

typedef struct {
    yvex_runtime_sampling_context **context;
    int rc;
} sampling_test_close_thread;

static int sampling_test_reenter(void *opaque)
{
    sampling_test_reentry *state = (sampling_test_reentry *)opaque;
    yvex_runtime_sampling_result result;
    yvex_error err;
    if (!state->attempted) {
        state->attempted = 1;
        state->refusal = yvex_runtime_sampling_select(
            state->context, NULL, state->source, &result, &err);
    }
    return 0;
}

static int sampling_test_hold_active(void *opaque)
{
    sampling_test_thread_gate *gate = (sampling_test_thread_gate *)opaque;
    if (pthread_mutex_lock(&gate->mutex) != 0) return 1;
    gate->entered = 1;
    (void)pthread_cond_broadcast(&gate->condition);
    while (!gate->release)
        (void)pthread_cond_wait(&gate->condition, &gate->mutex);
    (void)pthread_mutex_unlock(&gate->mutex);
    return 0;
}

static void *sampling_test_select_thread_main(void *opaque)
{
    sampling_test_select_thread *thread = (sampling_test_select_thread *)opaque;
    yvex_runtime_sampling_result result;
    yvex_error err;
    thread->rc = yvex_runtime_sampling_select(
        thread->context, NULL, thread->source, &result, &err);
    return NULL;
}

static void *sampling_test_contender_thread_main(void *opaque)
{
    sampling_test_select_thread *thread = (sampling_test_select_thread *)opaque;
    yvex_runtime_sampling_result result;
    yvex_error err;
    for (thread->attempts = 1ull; thread->attempts <= 1000000ull;
         ++thread->attempts) {
        thread->rc = yvex_runtime_sampling_select(
            thread->context, NULL, thread->source, &result, &err);
        if (thread->rc == YVEX_ERR_STATE &&
            strcmp(yvex_error_message(&err), "sampling context is closing") == 0) {
            thread->closing_refusal = 1;
            break;
        }
        (void)sched_yield();
    }
    return NULL;
}

static void *sampling_test_close_thread_main(void *opaque)
{
    sampling_test_close_thread *thread = (sampling_test_close_thread *)opaque;
    yvex_error err;
    thread->rc = yvex_runtime_sampling_context_close(thread->context, &err);
    return NULL;
}

static int sampling_test_close_entry_race(void)
{
    const float logits[4] = {1.0f, 3.0f, 2.0f, 0.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0};
    sampling_test_thread_gate gate;
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 4ull, .maximum_rows = 1ull,
        .cancel_requested = sampling_test_hold_active, .cancel_context = &gate};
    yvex_runtime_sampling_context *context = NULL, *borrowed;
    yvex_runtime_sampling_source source;
    sampling_test_select_thread selector, contender;
    sampling_test_close_thread closer;
    pthread_t selector_id, contender_id, closer_id;
    yvex_error err;
    memset(&gate, 0, sizeof(gate));
    YVEX_TEST_ASSERT(pthread_mutex_init(&gate.mutex, NULL) == 0 &&
                         pthread_cond_init(&gate.condition, NULL) == 0,
                     "close-race test synchronization initializes");
    sampling_test_plan(&plan, 4ull);
    YVEX_TEST_ASSERT(sampling_test_row(&plan, logits, 4ull, 1ull, &row) &&
                         yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK &&
                         yvex_runtime_sampling_context_open(
                             &context, &plan, &policy, &options, &err) == YVEX_OK &&
                         yvex_runtime_sampling_source_from_logits(
                             context, &source, logits, 4ull, &row, &err) == YVEX_OK,
                     "close-race production context opens");
    borrowed = context;
    selector = (sampling_test_select_thread){.context = borrowed, .source = &source,
                                             .gate = &gate, .rc = YVEX_ERR_STATE};
    contender = selector;
    closer = (sampling_test_close_thread){.context = &context, .rc = YVEX_ERR_STATE};
    YVEX_TEST_ASSERT(pthread_create(&selector_id, NULL,
                                    sampling_test_select_thread_main, &selector) == 0,
                     "active selector thread starts");
    (void)pthread_mutex_lock(&gate.mutex);
    while (!gate.entered) (void)pthread_cond_wait(&gate.condition, &gate.mutex);
    (void)pthread_mutex_unlock(&gate.mutex);
    YVEX_TEST_ASSERT(pthread_create(&closer_id, NULL,
                                    sampling_test_close_thread_main, &closer) == 0 &&
                         pthread_create(&contender_id, NULL,
                                        sampling_test_contender_thread_main, &contender) == 0,
                     "close and contender threads start while selection is active");
    (void)pthread_join(contender_id, NULL);
    (void)pthread_mutex_lock(&gate.mutex);
    gate.release = 1;
    (void)pthread_cond_broadcast(&gate.condition);
    (void)pthread_mutex_unlock(&gate.mutex);
    (void)pthread_join(selector_id, NULL);
    (void)pthread_join(closer_id, NULL);
    YVEX_TEST_ASSERT(selector.rc == YVEX_OK && contender.rc == YVEX_ERR_STATE &&
                         contender.closing_refusal && contender.attempts <= 1000000ull &&
                         closer.rc == YVEX_OK && context == NULL,
                     "close drains active selection and typed CLOSING excludes later entry");
    if (context) (void)yvex_runtime_sampling_context_close(&context, &err);
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    return 0;
}

static int sampling_test_lifecycle(void)
{
    const float logits[4] = {1.0f, 3.0f, 2.0f, 0.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0};
    sampling_test_reentry reentry = {0};
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 4ull, .maximum_rows = 1ull,
        .maximum_host_bytes = 1ull};
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_result result;
    yvex_runtime_sampling_context_summary summary;
    yvex_error err;
    sampling_test_plan(&plan, 4ull);
    YVEX_TEST_ASSERT(sampling_test_row(&plan, logits, 4ull, 1ull, &row) &&
                         yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK,
                     "lifecycle fixture seals");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_open(
                         &context, &plan, &policy, &options, &err) == YVEX_ERR_NOMEM &&
                         !context,
                     "sampling context refuses insufficient workspace budget");
    options.maximum_host_bytes = 0ull;
    options.cancel_requested = sampling_test_reenter;
    options.cancel_context = &reentry;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_open(
                         &context, &plan, &policy, &options, &err) == YVEX_OK &&
                         yvex_runtime_sampling_source_from_logits(
                             context, &source, logits, 4ull, &row, &err) == YVEX_OK,
                     "bounded sampling context and source open");
    reentry.context = context;
    reentry.source = &source;
    YVEX_TEST_ASSERT(yvex_runtime_sampling_select(
                         context, NULL, &source, &result, &err) == YVEX_OK &&
                         result.selected_token_id == 1u && reentry.attempted &&
                         reentry.refusal == YVEX_ERR_STATE,
                     "same-context concurrent use refuses while outer selection completes");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_snapshot(
                         context, &summary, &err) == YVEX_OK &&
                         summary.successful_samples == 1ull && summary.failure_count == 1ull &&
                         summary.warm_workspace_allocations == 0ull,
                     "lifecycle counters distinguish completion and concurrency refusal");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_close(&context, &err) == YVEX_OK,
                     "reused sampling context closes deterministically");
    return 0;
}

typedef struct { unsigned int calls, cancel_at; } sampling_test_cancel;

static int sampling_test_cancel_requested(void *opaque)
{
    sampling_test_cancel *state = (sampling_test_cancel *)opaque;
    state->calls++;
    return state->calls >= state->cancel_at;
}

/* Prove repeated partial progress retains only published rows and committed RNG draws. */
static int sampling_test_partial(void)
{
    const float logits[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result rows[2];
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0,
        .seed_present = 1, .seed = 7ull};
    sampling_test_cancel cancel = {.cancel_at = 3u};
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 4ull, .maximum_rows = 2ull,
        .cancel_requested = sampling_test_cancel_requested, .cancel_context = &cancel};
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_source sources[2];
    yvex_runtime_sampling_result results[2];
    yvex_runtime_sampling_execution execution;
    yvex_runtime_sampling_context_summary summary;
    yvex_error err;
    sampling_test_plan(&plan, 4ull);
    YVEX_TEST_ASSERT(sampling_test_row(&plan, logits, 4ull, 1ull, &rows[0]) &&
                         sampling_test_row(&plan, logits, 4ull, 2ull, &rows[1]) &&
                         yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK &&
                         yvex_runtime_sampling_context_open(
                             &context, &plan, &policy, &options, &err) == YVEX_OK,
                     "partial-progress context opens");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_source_from_logits(
                         context, &sources[0], logits, 4ull, &rows[0], &err) == YVEX_OK &&
                         yvex_runtime_sampling_source_from_logits(
                             context, &sources[1], logits, 4ull, &rows[1], &err) == YVEX_OK,
                     "ordered sources admit");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_execute(
                         context, sources, 2ull, results, 1ull,
                         &execution, &err) == YVEX_ERR_INVALID_ARG &&
                         execution.completed_samples == 0ull,
                     "insufficient repeated result capacity refuses before RNG use");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_execute(
                         context, sources, 2ull, results, 2ull,
                         &execution, &err) == YVEX_ERR_CANCELLED &&
                         execution.completed_samples == 1ull && execution.partial &&
                         execution.first_incomplete_sample == 1ull &&
                         results[0].completed && !results[1].completed,
                     "cancellation after one row publishes exact partial progress");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_snapshot(context, &summary, &err) == YVEX_OK &&
                         summary.stochastic_draws == 1ull &&
                         summary.successful_samples == 1ull &&
                         summary.cancellation_count == 1ull,
                     "failed row advances neither result nor RNG");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_close(&context, &err) == YVEX_OK,
                     "partial context closes");
    return 0;
}

static int sampling_test_partial_logits_prefix(void)
{
    float logits[8] = {0.0f, 1.0f, 2.0f, 3.0f, 91.0f, 92.0f, 93.0f, 94.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result rows[2];
    yvex_logits_operator_result result;
    yvex_runtime_sampling_policy policy = sampling_test_neutral_stochastic();
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 4ull, .maximum_rows = 1ull};
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_result sample;
    yvex_error err;
    sampling_test_plan(&plan, 4ull);
    memset(rows, 0, sizeof(rows));
    memset(&result, 0, sizeof(result));
    YVEX_TEST_ASSERT(sampling_test_row(&plan, logits, 4ull, 0ull, &rows[0]),
                     "partial logits completed prefix seals");
    result.plan = plan;
    result.rows = rows;
    result.raw_logits = logits;
    result.row_count = 1ull;
    result.raw_logits_count = 8ull;
    result.execution.schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    result.execution.requested_rows = 2ull;
    result.execution.completed_rows = 1ull;
    result.execution.first_incomplete_row = 1ull;
    result.execution.partial = 1;
    YVEX_TEST_ASSERT(yvex_runtime_logits_result_validate(
                         &result.plan, result.raw_logits,
                         result.raw_logits_count, result.rows, 2ull,
                         &result.execution, &err) == YVEX_ERR_FORMAT,
                     "partial logits total requested extent is not publishable");
    result.raw_logits_count = 4ull;
    YVEX_TEST_ASSERT(yvex_runtime_logits_result_validate(
                         &result.plan, result.raw_logits,
                         result.raw_logits_count, result.rows, 2ull,
                         &result.execution, &err) == YVEX_OK,
                     "partial logits validator admits only the completed prefix");
    result.execution.grouped_execution = 1;
    result.execution.grouped_rows = 2ull;
    result.execution.physical.memory.activation_bytes =
        2ull * (plan.hidden_width + plan.vocabulary_size) * sizeof(float);
    result.execution.physical.memory.temporary_bytes = sizeof(int);
    result.execution.physical.memory.measured_operations = 1ull;
    result.execution.physical.memory.complete = 1;
    result.execution.physical.kernel_count = 2ull;
    result.execution.physical.synchronization_count = 3ull;
    YVEX_TEST_ASSERT(yvex_runtime_logits_result_validate(
                         &result.plan, result.raw_logits,
                         result.raw_logits_count, result.rows, 2ull,
                         &result.execution, &err) == YVEX_OK,
                     "partial grouped logits retain complete physical batch facts");
    result.execution.grouped_rows = 1ull;
    YVEX_TEST_ASSERT(yvex_runtime_logits_result_validate(
                         &result.plan, result.raw_logits,
                         result.raw_logits_count, result.rows, 2ull,
                         &result.execution, &err) == YVEX_ERR_FORMAT,
                     "grouped logits reject physical geometry smaller than the request");
    result.execution.grouped_execution = 0;
    result.execution.grouped_rows = 0ull;
    memset(&result.execution.physical, 0, sizeof(result.execution.physical));
    YVEX_TEST_ASSERT(yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK &&
                         yvex_runtime_sampling_context_open(
                             &context, &plan, &policy, &options, &err) == YVEX_OK &&
                         yvex_runtime_sampling_source_from_logits(
                             context, &source, result.raw_logits,
                             result.raw_logits_count, &rows[0], &err) == YVEX_OK &&
                         yvex_runtime_sampling_select(
                             context, NULL, &source, &sample, &err) == YVEX_OK &&
                         sample.completed,
                     "sampling consumes exactly the completed logits prefix");
    YVEX_TEST_ASSERT(yvex_runtime_sampling_context_close(&context, &err) == YVEX_OK,
                     "partial-prefix sampling context closes");
    return 0;
}

static int sampling_test_device_context(void)
{
    const float logits[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy policy = sampling_test_neutral_stochastic();
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 4ull, .maximum_rows = 1ull,
        .device_selection = 1};
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_context_summary summary;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_result result;
    yvex_error err;
    sampling_test_plan(&plan, 4ull);
    options.device_selection = 2;
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_context_open(
            &context, &plan, &policy, &options, &err) == YVEX_ERR_INVALID_ARG && !context,
        "device sampling refuses a non-boolean selection contract");
    options.device_selection = 1;
    YVEX_TEST_ASSERT(
        sampling_test_row(&plan, logits, 4ull, 1ull, &row) &&
            yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_open(
                &context, &plan, &policy, &options, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_snapshot(context, &summary, &err) == YVEX_OK &&
            summary.workspace_bytes == sizeof(unsigned int) + sizeof(float) +
                                           sizeof(unsigned long long) &&
            summary.cold_workspace_allocations == 3ull,
        "device sampling owns only bounded reusable row-result storage");
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_source_from_logits(
            context, &source, logits, 4ull, &row, &err) == YVEX_OK &&
            yvex_runtime_sampling_select(context, NULL, &source, &result, &err) ==
                YVEX_ERR_FORMAT &&
            !result.completed,
        "device sampling context refuses host-authoritative source substitution");
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_context_close(&context, &err) == YVEX_OK && !context,
        "device sampling context closes without host candidate ownership");
    return 0;
}

int yvex_test_runtime_sampling(void)
{
    if (sampling_test_policy()) return 1;
    if (sampling_test_numeric_boundaries()) return 1;
    if (sampling_test_greedy()) return 1;
    if (sampling_test_padded_output_vocabulary()) return 1;
    if (sampling_test_filter_matrix()) return 1;
    if (sampling_test_stochastic()) return 1;
    if (sampling_test_rng_vectors()) return 1;
    if (sampling_test_checkpoint()) return 1;
    if (sampling_test_rng_transactions()) return 1;
    if (sampling_test_lifecycle()) return 1;
    if (sampling_test_close_entry_race()) return 1;
    if (sampling_test_partial()) return 1;
    if (sampling_test_partial_logits_prefix()) return 1;
    if (sampling_test_device_context()) return 1;
    return 0;
}
