#include "tests/test.h"

#include <limits.h>
#include <math.h>
#include <string.h>

#include <yvex/internal/execution_batch.h>

static void worklist_test_identity(char output[YVEX_SHA256_HEX_CAP], char digit)
{
    memset(output, digit, YVEX_SHA256_HEX_CAP - 1u);
    output[YVEX_SHA256_HEX_CAP - 1u] = '\0';
}

static int worklist_test_compatibility(void)
{
    yvex_execution_compatibility_key first = {0}, same, different;
    yvex_error err;
    first.schema_version = YVEX_EXECUTION_COMPATIBILITY_SCHEMA_V1;
    first.phase = YVEX_EXECUTION_PHASE_DECODE;
    first.backend_kind = 1u;
    first.tensor_scope = 2u;
    first.execution_class = 1u;
    first.publication_contract = 1u;
    first.model_generation = 7ull;
    first.layer_ordinal = 3ull;
    first.row_width = 14336ull;
    first.admitted_width = 8ull;
    worklist_test_identity(first.runtime_model_identity, '1');
    worklist_test_identity(first.runtime_binding_identity, '2');
    worklist_test_identity(first.physical_variant_identity, '3');
    worklist_test_identity(first.execution_profile_identity, '4');
    worklist_test_identity(first.operation_identity, '5');
    YVEX_TEST_ASSERT(
        yvex_execution_compatibility_key_seal(&first, &err) == YVEX_OK &&
            yvex_execution_compatibility_key_validate(&first, &err) == YVEX_OK,
        "complete execution compatibility facts should seal");
    same = first;
    YVEX_TEST_ASSERT(
        yvex_execution_compatibility_keys_match(&first, &same, &err),
        "identical compiled operations should be compatible");
    different = first;
    different.layer_ordinal++;
    YVEX_TEST_ASSERT(
        yvex_execution_compatibility_key_seal(&different, &err) == YVEX_OK &&
            !yvex_execution_compatibility_keys_match(&first, &different, &err),
        "equal row geometry must not merge distinct compiled operations");
    different = first;
    different.admitted_width = 0ull;
    YVEX_TEST_ASSERT(
        yvex_execution_compatibility_key_seal(&different, &err) ==
            YVEX_ERR_INVALID_ARG,
        "unbounded compatibility width should refuse");
    different = first;
    different.runtime_binding_identity[0] = 'f';
    YVEX_TEST_ASSERT(
        yvex_execution_compatibility_key_validate(&different, &err) ==
            YVEX_ERR_FORMAT,
        "stale compatibility identity should refuse");
    return 0;
}

static int worklist_test_build(void)
{
    const unsigned long long selected[] = {3ull, 1ull, 7ull, 1ull, 3ull, 5ull,
                                           7ull, 1ull, 5ull, 3ull, 7ull, 5ull};
    const float weights[] = {0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f,
                             0.31f, 0.32f, 0.33f, 0.41f, 0.42f, 0.43f};
    unsigned long long experts[8], offsets[9], populations[8];
    unsigned long long pairs[12], rows[12], destinations[12];
    float ordered_weights[12];
    yvex_execution_batch_source source = {4ull, 12ull, {0}};
    yvex_execution_batch_row batch_rows[4] = {0};
    yvex_execution_batch batch = {0};
    yvex_expert_worklist_policy policy = {0};
    yvex_expert_worklist_request request = {0};
    yvex_expert_worklist_storage storage = {0};
    yvex_expert_worklist first, repeated;
    yvex_error err;

    batch.schema_version = YVEX_EXECUTION_BATCH_SCHEMA_V1;
    batch.provenance = YVEX_EXECUTION_BATCH_SPECULATIVE_VERIFICATION;
    batch.phase = 3u;
    batch.row_count = 4ull;
    batch.source_count = 1ull;
    batch.model_generation = 9ull;
    batch.sources = &source;
    batch.rows = batch_rows;
    worklist_test_identity(source.identity, '0');
    for (unsigned long long row = 0ull; row < 4ull; ++row) {
        batch_rows[row].source_row = row;
        batch_rows[row].sequence_position = 20ull + row;
        batch_rows[row].candidate_present = 1;
        batch_rows[row].candidate_ordinal = row;
        batch_rows[row].publication_ordinal = row;
    }
    worklist_test_identity(batch.runtime_model_identity, '1');
    worklist_test_identity(batch.runtime_binding_identity, '2');
    worklist_test_identity(batch.physical_variant_identity, '3');
    worklist_test_identity(batch.execution_profile_identity, '4');
    worklist_test_identity(batch.operation_identity, '5');
    YVEX_TEST_ASSERT(yvex_execution_batch_seal(&batch, &err) == YVEX_OK,
                     "verification execution batch should seal");

    policy.schema_version = YVEX_EXPERT_WORKLIST_POLICY_SCHEMA_V1;
    policy.supported_width_mask = 0x1feull;
    policy.tensor_core_minimum = 3ull;
    strcpy(policy.narrow_kernel_family, "exact-narrow");
    strcpy(policy.tensor_core_kernel_family, "exact-wide");
    YVEX_TEST_ASSERT(yvex_expert_worklist_policy_seal(&policy, &err) == YVEX_OK,
                     "compiled expert worklist policy should seal");

    request.schema_version = YVEX_EXPERT_WORKLIST_SCHEMA_V1;
    request.batch = &batch;
    request.policy = &policy;
    request.expert_count = 8ull;
    request.experts_per_row = 3ull;
    request.pair_count = 12ull;
    request.selected_experts = selected;
    request.route_weights = weights;
    storage.expert_ids = experts;
    storage.bucket_offsets = offsets;
    storage.bucket_populations = populations;
    storage.source_pairs = pairs;
    storage.source_rows = rows;
    storage.destination_rows = destinations;
    storage.route_weights = ordered_weights;
    storage.bucket_capacity = 8ull;
    storage.pair_capacity = 12ull;

    YVEX_TEST_ASSERT(yvex_expert_worklist_build(&request, &storage, &first, &err) == YVEX_OK,
                     "expert-major worklist should build");
    YVEX_TEST_ASSERT(first.bucket_count == 4ull && first.pair_count == 12ull &&
                         first.actual_width == 4ull &&
                         first.maximum_bucket_population == 3ull &&
                         first.tensor_core_eligible_pairs == 12ull &&
                         first.narrow_pairs == 0ull && first.tail_rows == 20ull &&
                         first.population_histogram[3] == 4ull,
                     "worklist should expose exact populations and admitted tails");
    YVEX_TEST_ASSERT(experts[0] == 1ull && experts[1] == 3ull && experts[2] == 5ull &&
                         experts[3] == 7ull && offsets[0] == 0ull && offsets[1] == 3ull &&
                         offsets[4] == 12ull && populations[0] == 3ull &&
                         pairs[0] == 1ull && pairs[1] == 3ull && pairs[2] == 7ull &&
                         rows[0] == 0ull && rows[2] == 2ull && destinations[2] == 7ull &&
                         memcmp(&ordered_weights[2], &weights[7], sizeof(float)) == 0,
                     "worklist should preserve exact route/source associations");
    repeated = first;
    YVEX_TEST_ASSERT(yvex_expert_worklist_build(&request, &storage, &repeated, &err) == YVEX_OK &&
                         strcmp(first.identity, repeated.identity) == 0,
                     "worklist construction should be deterministic");
    return 0;
}

static int worklist_test_refusals(void)
{
    unsigned long long selected[] = {0ull, 1ull};
    float weights[] = {0.5f, 0.5f};
    unsigned long long experts[2], offsets[3], populations[2];
    unsigned long long pairs[2], rows[2], destinations[2];
    float ordered[2];
    yvex_execution_batch_source source = {1ull, 1ull, {0}};
    yvex_execution_batch_row batch_rows[2] = {
        {0ull, 0ull, 10ull, 0ull, 0ull, 0},
        {0ull, 1ull, 11ull, 0ull, 1ull, 0}};
    yvex_execution_batch batch = {0};
    yvex_expert_worklist_policy policy = {0};
    yvex_expert_worklist_request request = {0};
    yvex_expert_worklist_storage storage = {
        experts, offsets, populations, pairs, rows, destinations, ordered, 2ull, 2ull};
    yvex_expert_worklist worklist;
    yvex_error err;

    batch.schema_version = YVEX_EXECUTION_BATCH_SCHEMA_V1;
    batch.provenance = YVEX_EXECUTION_BATCH_SINGLE_ROW;
    batch.row_count = batch.source_count = batch.model_generation = 1ull;
    batch.sources = &source;
    batch.rows = batch_rows;
    worklist_test_identity(source.identity, '0');
    worklist_test_identity(batch.runtime_model_identity, 'a');
    worklist_test_identity(batch.runtime_binding_identity, 'b');
    worklist_test_identity(batch.physical_variant_identity, 'c');
    worklist_test_identity(batch.execution_profile_identity, 'd');
    worklist_test_identity(batch.operation_identity, 'e');
    YVEX_TEST_ASSERT(yvex_execution_batch_seal(&batch, &err) == YVEX_OK,
                     "single-row execution batch should seal");
    policy.schema_version = YVEX_EXPERT_WORKLIST_POLICY_SCHEMA_V1;
    policy.supported_width_mask = 2ull;
    strcpy(policy.narrow_kernel_family, "exact-narrow");
    YVEX_TEST_ASSERT(yvex_expert_worklist_policy_seal(&policy, &err) == YVEX_OK,
                     "narrow-only worklist policy should seal");
    request = (yvex_expert_worklist_request){YVEX_EXPERT_WORKLIST_SCHEMA_V1, &batch,
        &policy, 2ull, 2ull, 2ull, selected, weights};
    YVEX_TEST_ASSERT(yvex_expert_worklist_build(&request, &storage, &worklist, &err) == YVEX_OK,
                     "valid width-one buckets should build");
    source.execution_generation++;
    YVEX_TEST_ASSERT(yvex_execution_batch_validate(&batch, &err) == YVEX_ERR_FORMAT &&
                         yvex_expert_worklist_build(
                             &request, &storage, &worklist, &err) == YVEX_ERR_INVALID_ARG,
                     "stale execution-batch identity should refuse before construction");
    source.execution_generation--;
    YVEX_TEST_ASSERT(yvex_execution_batch_seal(&batch, &err) == YVEX_OK,
                     "restored execution batch should reseal");
    policy.supported_width_mask = 6ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_policy_validate(&policy, &err) == YVEX_ERR_FORMAT,
                     "stale compiled width policy should refuse");
    policy.supported_width_mask = 2ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_policy_seal(&policy, &err) == YVEX_OK,
                     "restored worklist policy should reseal");
    policy.supported_width_mask = 0ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_policy_seal(
                         &policy, &err) == YVEX_ERR_INVALID_ARG,
                     "zero-width worklist policy should refuse");
    policy.supported_width_mask = 2ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_policy_seal(&policy, &err) == YVEX_OK,
                     "narrow worklist policy should reseal after refusal");
    batch.row_count = 2ull;
    batch.provenance = YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE;
    request.experts_per_row = 1ull;
    YVEX_TEST_ASSERT(yvex_execution_batch_seal(&batch, &err) == YVEX_OK &&
                         yvex_expert_worklist_build(
                             &request, &storage, &worklist, &err) == YVEX_ERR_INVALID_ARG,
                     "runtime width outside the compiled mask should refuse");
    batch.row_count = 1ull;
    batch.provenance = YVEX_EXECUTION_BATCH_SINGLE_ROW;
    request.experts_per_row = 2ull;
    YVEX_TEST_ASSERT(yvex_execution_batch_seal(&batch, &err) == YVEX_OK,
                     "single-row execution batch should reseal after width refusal");
    storage.pair_capacity = 1ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_build(
                         &request, &storage, &worklist, &err) == YVEX_ERR_INVALID_ARG,
                     "undersized worklist storage should refuse");
    storage.pair_capacity = 2ull;
    weights[1] = NAN;
    YVEX_TEST_ASSERT(yvex_expert_worklist_build(
                         &request, &storage, &worklist, &err) == YVEX_ERR_FORMAT,
                     "non-finite route weight should refuse");
    weights[1] = 0.5f;
    YVEX_TEST_ASSERT(yvex_expert_worklist_build(
                         &request, &storage, &worklist, &err) == YVEX_OK,
                     "valid worklist should rebuild before mutation testing");
    destinations[0] = 1ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_validate(&request, &worklist, &err) ==
                         YVEX_ERR_FORMAT,
                     "duplicate or stale destination mapping should refuse");
    selected[1] = 2ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_build(&request, &storage, &worklist, &err) ==
                         YVEX_ERR_FORMAT,
                     "out-of-range expert should refuse before execution");
    selected[1] = 0ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_build(&request, &storage, &worklist, &err) ==
                         YVEX_OK && worklist.maximum_bucket_population == 2ull &&
                         worklist.narrow_pairs == 2ull,
                     "large buckets may split into real admitted narrow tiles");
    return 0;
}

static int worklist_test_observation(void)
{
    yvex_expert_worklist_observation facts = {0}, delta = {0};
    yvex_error err;
    delta.schema_version = YVEX_EXPERT_WORKLIST_OBSERVATION_SCHEMA_V1;
    delta.worklist_count = 1ull;
    delta.pair_count = delta.narrow_pairs = 6ull;
    delta.bucket_count = 3ull;
    delta.maximum_bucket_population = 3ull;
    delta.width_histogram[4] = 1ull;
    delta.population_histogram[1] = 1ull;
    delta.population_histogram[2] = 1ull;
    delta.population_histogram[3] = 1ull;
    delta.provenance_counts[YVEX_EXECUTION_BATCH_SPECULATIVE_VERIFICATION] = 1ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_observation_add(
                         &facts, &delta, &err) == YVEX_OK &&
                         facts.width_histogram[4] == 1ull &&
                         facts.population_histogram[3] == 1ull,
                     "worklist observations should preserve real width and population");
    facts.pair_count = ULLONG_MAX;
    YVEX_TEST_ASSERT(yvex_expert_worklist_observation_add(
                         &facts, &delta, &err) == YVEX_ERR_BOUNDS,
                     "worklist observation overflow should refuse atomically");
    delta.width_histogram[4] = 0ull;
    YVEX_TEST_ASSERT(yvex_expert_worklist_observation_add(
                         &(yvex_expert_worklist_observation){0}, &delta, &err) ==
                         YVEX_ERR_FORMAT,
                     "worklist observation width mismatch should refuse");
    return 0;
}

static int worklist_test_multi_session_sources(void)
{
    yvex_execution_batch_source sources[2] = {
        {3ull, 7ull, {0}}, {8ull, 11ull, {0}}};
    yvex_execution_batch_row rows[2] = {
        {0ull, 0ull, 20ull, 0ull, 0ull, 0},
        {1ull, 0ull, 34ull, 0ull, 1ull, 0}};
    yvex_execution_batch batch = {0};
    yvex_error err;
    worklist_test_identity(sources[0].identity, '6');
    worklist_test_identity(sources[1].identity, '7');
    batch.schema_version = YVEX_EXECUTION_BATCH_SCHEMA_V1;
    batch.provenance = YVEX_EXECUTION_BATCH_MULTI_SESSION;
    batch.phase = YVEX_EXECUTION_PHASE_DECODE;
    batch.row_count = batch.source_count = batch.model_generation = 2ull;
    batch.sources = sources;
    batch.rows = rows;
    worklist_test_identity(batch.runtime_model_identity, 'a');
    worklist_test_identity(batch.runtime_binding_identity, 'b');
    worklist_test_identity(batch.physical_variant_identity, 'c');
    worklist_test_identity(batch.execution_profile_identity, 'd');
    worklist_test_identity(batch.operation_identity, 'e');
    YVEX_TEST_ASSERT(yvex_execution_batch_seal(&batch, &err) == YVEX_OK,
                     "multi-session rows should seal under one typed batch");
    rows[1].source_index = 0ull;
    YVEX_TEST_ASSERT(yvex_execution_batch_seal(&batch, &err) == YVEX_ERR_INVALID_ARG,
                     "duplicate source-row ownership should refuse");
    rows[1].source_index = 1ull;
    memcpy(sources[1].identity, sources[0].identity, sizeof(sources[1].identity));
    YVEX_TEST_ASSERT(yvex_execution_batch_seal(&batch, &err) == YVEX_ERR_INVALID_ARG,
                     "ambiguous multi-session source identity should refuse");
    return 0;
}

int yvex_test_expert_worklist(void)
{
    if (worklist_test_compatibility() != 0) return 1;
    if (worklist_test_build() != 0) return 1;
    if (worklist_test_refusals() != 0) return 1;
    if (worklist_test_multi_session_sources() != 0) return 1;
    return worklist_test_observation();
}
