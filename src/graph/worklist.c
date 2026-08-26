/* Deterministically lower routed pairs into one expert-major physical execution contract. */
#include <yvex/internal/execution_batch.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>

static int worklist_refuse(yvex_error *err, yvex_status status,
                           const char *reason)
{
    yvex_error_set(err, status, "graph.worklist", reason);
    return status;
}

static int worklist_hash_finish(yvex_sha256 *hash,
                                char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int worklist_identity_valid(const char *identity)
{
    return identity && yvex_sha256_hex_valid(identity);
}

static int execution_batch_sources_valid(const yvex_execution_batch *batch)
{
    unsigned long long left, right;
    if (!batch->source_count || batch->source_count > batch->row_count ||
        !batch->sources || !batch->rows ||
        (batch->provenance == YVEX_EXECUTION_BATCH_MULTI_SESSION &&
         batch->source_count < 2ull) ||
        (batch->provenance != YVEX_EXECUTION_BATCH_MULTI_SESSION &&
         batch->source_count != 1ull))
        return 0;
    for (left = 0ull; left < batch->source_count; ++left) {
        const yvex_execution_batch_source *source = &batch->sources[left];
        if (!source->execution_generation ||
            !worklist_identity_valid(source->identity))
            return 0;
        for (right = left + 1ull; right < batch->source_count; ++right)
            if (!strcmp(source->identity, batch->sources[right].identity))
                return 0;
    }
    for (left = 0ull; left < batch->row_count; ++left) {
        const yvex_execution_batch_row *row = &batch->rows[left];
        if (row->source_index >= batch->source_count ||
            (!row->candidate_present && row->candidate_ordinal))
            return 0;
        for (right = left + 1ull; right < batch->row_count; ++right)
            if (row->source_index == batch->rows[right].source_index &&
                row->source_row == batch->rows[right].source_row)
                return 0;
    }
    return 1;
}

static int execution_batch_sources_hash(yvex_sha256 *hash,
                                        const yvex_execution_batch *batch)
{
    unsigned long long index;
    for (index = 0ull; index < batch->source_count; ++index)
        if (!yvex_sha256_update_u64(hash, batch->sources[index].execution_generation) ||
            !yvex_sha256_update_u64(hash, batch->sources[index].state_generation) ||
            !yvex_sha256_update_text(hash, batch->sources[index].identity))
            return 0;
    for (index = 0ull; index < batch->row_count; ++index)
        if (!yvex_sha256_update_u64(hash, batch->rows[index].source_index) ||
            !yvex_sha256_update_u64(hash, batch->rows[index].source_row) ||
            !yvex_sha256_update_u64(hash, batch->rows[index].sequence_position) ||
            !yvex_sha256_update_u64(hash, batch->rows[index].candidate_present) ||
            !yvex_sha256_update_u64(hash, batch->rows[index].candidate_ordinal) ||
            !yvex_sha256_update_u64(hash, batch->rows[index].publication_ordinal))
            return 0;
    return 1;
}

int yvex_execution_batch_seal(yvex_execution_batch *batch, yvex_error *err)
{
    yvex_sha256 hash;
    if (!batch || batch->schema_version != YVEX_EXECUTION_BATCH_SCHEMA_V1 ||
        batch->provenance > YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE ||
        batch->phase >= YVEX_EXECUTION_PHASE_COUNT ||
        !batch->row_count || batch->row_count >= 64ull ||
        (batch->provenance == YVEX_EXECUTION_BATCH_SINGLE_ROW &&
         batch->row_count != 1ull) ||
        (batch->provenance == YVEX_EXECUTION_BATCH_SPECULATIVE_VERIFICATION &&
         batch->phase != YVEX_EXECUTION_PHASE_VERIFY) ||
        (batch->provenance == YVEX_EXECUTION_BATCH_MULTI_SESSION &&
         batch->row_count < 2ull) ||
        (batch->provenance == YVEX_EXECUTION_BATCH_PREFILL &&
         batch->phase != YVEX_EXECUTION_PHASE_PREFILL) ||
        !batch->engine_generation || !execution_batch_sources_valid(batch) ||
        !worklist_identity_valid(batch->runtime_model_identity) ||
        !worklist_identity_valid(batch->runtime_binding_identity) ||
        !worklist_identity_valid(batch->physical_variant_identity) ||
        !worklist_identity_valid(batch->execution_profile_identity) ||
        !worklist_identity_valid(batch->operation_identity))
        return worklist_refuse(
            err, YVEX_ERR_INVALID_ARG, "execution batch identity or provenance is incomplete");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.execution-batch.v1") ||
        !yvex_sha256_update_u64(&hash, batch->schema_version) ||
        !yvex_sha256_update_u64(&hash, batch->provenance) ||
        !yvex_sha256_update_u64(&hash, batch->phase) ||
        !yvex_sha256_update_u64(&hash, batch->row_count) ||
        !yvex_sha256_update_u64(&hash, batch->source_count) ||
        !yvex_sha256_update_u64(&hash, batch->engine_generation) ||
        !execution_batch_sources_hash(&hash, batch) ||
        !yvex_sha256_update_text(&hash, batch->runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, batch->runtime_binding_identity) ||
        !yvex_sha256_update_text(&hash, batch->physical_variant_identity) ||
        !yvex_sha256_update_text(&hash, batch->execution_profile_identity) ||
        !yvex_sha256_update_text(&hash, batch->operation_identity) ||
        !worklist_hash_finish(&hash, batch->identity))
        return worklist_refuse(err, YVEX_ERR_STATE,
                               "execution batch identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_batch_validate(const yvex_execution_batch *batch,
                                  yvex_error *err)
{
    yvex_execution_batch expected;
    char identity[YVEX_SHA256_HEX_CAP];
    if (!batch || !worklist_identity_valid(batch->identity))
        return worklist_refuse(err, YVEX_ERR_INVALID_ARG,
                               "execution batch identity is unavailable");
    expected = *batch;
    yvex_core_text_copy(identity, sizeof(identity), batch->identity);
    if (yvex_execution_batch_seal(&expected, err) != YVEX_OK)
        return yvex_error_code(err);
    if (strcmp(identity, expected.identity) != 0)
        return worklist_refuse(err, YVEX_ERR_FORMAT,
                               "execution batch identity is stale");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_compatibility_key_validate(
    const yvex_execution_compatibility_key *key, yvex_error *err)
{
    if (!key || key->schema_version != YVEX_EXECUTION_COMPATIBILITY_SCHEMA_V2 ||
        key->phase >= YVEX_EXECUTION_PHASE_COUNT ||
        key->operation >= YVEX_EXECUTION_COMPATIBILITY_OPERATION_COUNT ||
        !key->engine_generation || !key->row_width || !key->admitted_width ||
        key->admitted_width >= 64ull)
        return worklist_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "execution compatibility geometry or engine handle is incomplete");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_compatibility_keys_match(
    const yvex_execution_compatibility_key *left,
    const yvex_execution_compatibility_key *right, yvex_error *err)
{
    if (yvex_execution_compatibility_key_validate(left, err) != YVEX_OK ||
        yvex_execution_compatibility_key_validate(right, err) != YVEX_OK)
        return 0;
    yvex_error_clear(err);
    return left->phase == right->phase &&
           left->operation == right->operation &&
           left->backend_kind == right->backend_kind &&
           left->tensor_scope == right->tensor_scope &&
           left->execution_class == right->execution_class &&
           left->engine_generation == right->engine_generation &&
           left->layer_ordinal == right->layer_ordinal &&
           left->row_width == right->row_width &&
           left->admitted_width == right->admitted_width;
}

int yvex_expert_worklist_policy_seal(yvex_expert_worklist_policy *policy,
                                     yvex_error *err)
{
    yvex_sha256 hash;
    int tensor_core;
    if (!policy || policy->schema_version != YVEX_EXPERT_WORKLIST_POLICY_SCHEMA_V1 ||
        !(policy->supported_width_mask & 2ull) ||
        (policy->supported_width_mask & 1ull) ||
        (policy->supported_width_mask >> 63u) ||
        policy->narrow_implementation >= YVEX_ENGINE_IMPLEMENTATION_COUNT)
        return worklist_refuse(err, YVEX_ERR_INVALID_ARG,
                               "expert worklist width policy is incomplete");
    tensor_core = policy->tensor_core_minimum != 0ull;
    if (tensor_core != (policy->wide_implementation != YVEX_ENGINE_IMPLEMENTATION_COUNT) ||
        (tensor_core &&
         (policy->tensor_core_minimum >= 63ull ||
          !(policy->supported_width_mask & (1ull << policy->tensor_core_minimum)) ||
          policy->wide_implementation !=
              YVEX_ENGINE_IMPLEMENTATION_CUDA_SM121_MOE_TENSORCORE ||
          policy->narrow_implementation == policy->wide_implementation)))
        return worklist_refuse(err, YVEX_ERR_INVALID_ARG,
                               "expert Tensor Core width policy is inconsistent");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.expert-worklist-policy.v1") ||
        !yvex_sha256_update_u64(&hash, policy->schema_version) ||
        !yvex_sha256_update_u64(&hash, policy->supported_width_mask) ||
        !yvex_sha256_update_u64(&hash, policy->tensor_core_minimum) ||
        !yvex_sha256_update_u64(&hash, policy->narrow_implementation) ||
        !yvex_sha256_update_u64(&hash, policy->wide_implementation) ||
        !worklist_hash_finish(&hash, policy->identity))
        return worklist_refuse(err, YVEX_ERR_STATE,
                               "expert worklist policy identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_expert_worklist_policy_validate(
    const yvex_expert_worklist_policy *policy, yvex_error *err)
{
    yvex_expert_worklist_policy expected;
    char identity[YVEX_SHA256_HEX_CAP];
    if (!policy || !worklist_identity_valid(policy->identity))
        return worklist_refuse(err, YVEX_ERR_INVALID_ARG,
                               "expert worklist policy identity is unavailable");
    expected = *policy;
    yvex_core_text_copy(identity, sizeof(identity), policy->identity);
    if (yvex_expert_worklist_policy_seal(&expected, err) != YVEX_OK)
        return yvex_error_code(err);
    if (strcmp(identity, expected.identity) != 0)
        return worklist_refuse(err, YVEX_ERR_FORMAT,
                               "expert worklist policy identity is stale");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int worklist_request_valid(const yvex_expert_worklist_request *request,
                                  const yvex_expert_worklist_storage *storage)
{
    unsigned long long pairs;
    return request && storage && request->schema_version == YVEX_EXPERT_WORKLIST_SCHEMA_V1 &&
           request->batch && request->policy &&
           yvex_execution_batch_validate(request->batch, NULL) == YVEX_OK &&
           yvex_expert_worklist_policy_validate(request->policy, NULL) == YVEX_OK &&
           request->expert_count && request->expert_count <= 256ull &&
           request->experts_per_row && request->experts_per_row <= request->expert_count &&
           request->batch->row_count < 63ull &&
           (request->policy->supported_width_mask &
            (1ull << request->batch->row_count)) &&
           yvex_core_u64_mul(request->batch->row_count, request->experts_per_row, &pairs) &&
           pairs == request->pair_count && request->selected_experts &&
           request->route_weights && storage->expert_ids && storage->bucket_offsets &&
           storage->bucket_populations && storage->source_pairs && storage->source_rows &&
           storage->destination_rows && storage->route_weights &&
           storage->bucket_capacity >= request->expert_count &&
           storage->pair_capacity >= request->pair_count;
}

static unsigned long long worklist_maximum_width(unsigned long long mask)
{
    unsigned long long width = 0ull;
    while (mask >>= 1u) width++;
    return width;
}

static int worklist_seal(const yvex_expert_worklist_request *request,
                         yvex_expert_worklist *worklist)
{
    yvex_sha256 hash;
    unsigned long long bucket, pair;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.expert-worklist.v1") ||
        !yvex_sha256_update_text(&hash, worklist->batch_identity) ||
        !yvex_sha256_update_text(&hash, worklist->policy_identity) ||
        !yvex_sha256_update_u64(&hash, worklist->actual_width) ||
        !yvex_sha256_update_u64(&hash, worklist->expert_count) ||
        !yvex_sha256_update_u64(&hash, worklist->pair_count) ||
        !yvex_sha256_update_u64(&hash, worklist->bucket_count) ||
        !yvex_sha256_update_u64(&hash, worklist->maximum_bucket_population))
        return 0;
    for (bucket = 0ull; bucket < worklist->bucket_count; ++bucket)
        if (!yvex_sha256_update_u64(&hash, worklist->expert_ids[bucket]) ||
            !yvex_sha256_update_u64(&hash, worklist->bucket_offsets[bucket]) ||
            !yvex_sha256_update_u64(&hash, worklist->bucket_populations[bucket]))
            return 0;
    if (!yvex_sha256_update_u64(&hash, worklist->bucket_offsets[worklist->bucket_count]))
        return 0;
    for (pair = 0ull; pair < request->pair_count; ++pair) {
        uint32_t weight_bits;
        memcpy(&weight_bits, &worklist->route_weights[pair], sizeof(weight_bits));
        if (!yvex_sha256_update_u64(&hash, worklist->source_pairs[pair]) ||
            !yvex_sha256_update_u64(&hash, worklist->source_rows[pair]) ||
            !yvex_sha256_update_u64(&hash, worklist->destination_rows[pair]) ||
            !yvex_sha256_update_u64(&hash, weight_bits))
            return 0;
    }
    return worklist_hash_finish(&hash, worklist->identity);
}

int yvex_expert_worklist_build(const yvex_expert_worklist_request *request,
                               const yvex_expert_worklist_storage *storage,
                               yvex_expert_worklist *worklist, yvex_error *err)
{
    unsigned long long expert, pair, cursor = 0ull, bucket = 0ull;
    unsigned long long maximum_width;
    if (worklist) memset(worklist, 0, sizeof(*worklist));
    if (!worklist || !worklist_request_valid(request, storage))
        return worklist_refuse(err, YVEX_ERR_INVALID_ARG,
                               "expert worklist request or storage is invalid");
    maximum_width = worklist_maximum_width(request->policy->supported_width_mask);
    for (expert = 0ull; expert < request->expert_count; ++expert) {
        unsigned long long population = 0ull, start = cursor;
        for (pair = 0ull; pair < request->pair_count; ++pair) {
            float weight = request->route_weights[pair];
            if (request->selected_experts[pair] >= request->expert_count ||
                !isfinite(weight))
                return worklist_refuse(err, YVEX_ERR_FORMAT,
                                       "routed pair has an invalid expert or weight");
            if (request->selected_experts[pair] != expert) continue;
            storage->source_pairs[cursor] = pair;
            storage->source_rows[cursor] = pair / request->experts_per_row;
            storage->destination_rows[cursor] = pair;
            storage->route_weights[cursor] = weight;
            cursor++;
            population++;
        }
        if (!population) continue;
        storage->expert_ids[bucket] = expert;
        storage->bucket_offsets[bucket] = start;
        storage->bucket_populations[bucket] = population;
        if (population > worklist->maximum_bucket_population)
            worklist->maximum_bucket_population = population;
        worklist->population_histogram[
            population < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP
                ? population : YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP - 1u]++;
        if (request->policy->tensor_core_minimum &&
            population >= request->policy->tensor_core_minimum) {
            worklist->tensor_core_eligible_pairs += population;
            if (population % maximum_width)
                worklist->tail_rows += maximum_width - population % maximum_width;
        }
        bucket++;
    }
    storage->bucket_offsets[bucket] = cursor;
    if (cursor != request->pair_count)
        return worklist_refuse(err, YVEX_ERR_STATE,
                               "expert worklist did not emit every routed pair");
    worklist->schema_version = YVEX_EXPERT_WORKLIST_SCHEMA_V1;
    worklist->provenance = request->batch->provenance;
    worklist->phase = request->batch->phase;
    worklist->actual_width = request->batch->row_count;
    worklist->expert_count = request->expert_count;
    worklist->pair_count = request->pair_count;
    worklist->bucket_count = bucket;
    worklist->admitted_tile_width = maximum_width;
    worklist->narrow_pairs = request->pair_count - worklist->tensor_core_eligible_pairs;
    worklist->expert_ids = storage->expert_ids;
    worklist->bucket_offsets = storage->bucket_offsets;
    worklist->bucket_populations = storage->bucket_populations;
    worklist->source_pairs = storage->source_pairs;
    worklist->source_rows = storage->source_rows;
    worklist->destination_rows = storage->destination_rows;
    worklist->route_weights = storage->route_weights;
    yvex_core_text_copy(worklist->batch_identity, sizeof(worklist->batch_identity),
                        request->batch->identity);
    yvex_core_text_copy(worklist->policy_identity, sizeof(worklist->policy_identity),
                        request->policy->identity);
    if (!worklist_seal(request, worklist))
        return worklist_refuse(err, YVEX_ERR_STATE,
                               "expert worklist identity derivation failed");
    return yvex_expert_worklist_validate(request, worklist, err);
}

int yvex_expert_worklist_validate(const yvex_expert_worklist_request *request,
                                  const yvex_expert_worklist *worklist,
                                  yvex_error *err)
{
    unsigned char *seen;
    unsigned long long bucket, position, populations = 0ull;
    int rc = YVEX_OK;
    if (!request || !worklist ||
        worklist->schema_version != YVEX_EXPERT_WORKLIST_SCHEMA_V1 ||
        !worklist_identity_valid(worklist->identity) ||
        strcmp(worklist->batch_identity, request->batch->identity) != 0 ||
        strcmp(worklist->policy_identity, request->policy->identity) != 0 ||
        worklist->actual_width != request->batch->row_count ||
        !worklist->bucket_count || worklist->bucket_count > request->expert_count ||
        worklist->pair_count != request->pair_count || !worklist->expert_ids ||
        !worklist->bucket_offsets || !worklist->bucket_populations ||
        !worklist->source_pairs || !worklist->source_rows ||
        !worklist->destination_rows || !worklist->route_weights ||
        worklist->bucket_offsets[0] != 0ull ||
        worklist->bucket_offsets[worklist->bucket_count] != request->pair_count)
        return worklist_refuse(err, YVEX_ERR_INVALID_ARG,
                               "expert worklist structure or identity is invalid");
    seen = yvex_core_calloc((size_t)request->pair_count, sizeof(*seen));
    if (!seen)
        return worklist_refuse(err, YVEX_ERR_NOMEM,
                               "expert worklist validation allocation failed");
    for (bucket = 0ull; rc == YVEX_OK && bucket < worklist->bucket_count; ++bucket) {
        unsigned long long expert = worklist->expert_ids[bucket];
        unsigned long long start = worklist->bucket_offsets[bucket];
        unsigned long long end = worklist->bucket_offsets[bucket + 1ull];
        unsigned long long population = worklist->bucket_populations[bucket];
        if (expert >= request->expert_count ||
            (bucket && expert <= worklist->expert_ids[bucket - 1ull]) ||
            end < start || end - start != population || !population) {
            rc = worklist_refuse(err, YVEX_ERR_FORMAT,
                                 "expert worklist bucket is malformed");
            break;
        }
        populations += population;
        for (position = start; position < end; ++position) {
            unsigned long long pair = worklist->source_pairs[position];
            if (pair >= request->pair_count || seen[pair] ||
                request->selected_experts[pair] != expert ||
                worklist->source_rows[position] != pair / request->experts_per_row ||
                worklist->destination_rows[position] != pair ||
                memcmp(&worklist->route_weights[position],
                       &request->route_weights[pair], sizeof(float)) != 0) {
                rc = worklist_refuse(err, YVEX_ERR_FORMAT,
                                     "expert worklist pair mapping is not exact");
                break;
            }
            seen[pair] = 1u;
        }
    }
    if (rc == YVEX_OK && populations != request->pair_count)
        rc = worklist_refuse(err, YVEX_ERR_FORMAT,
                             "expert worklist population sum is incomplete");
    yvex_core_free(seen);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_expert_worklist_routing_identity(
    const char *batch_identity, const char *policy_identity,
    const char *routing_identity, char output[YVEX_SHA256_HEX_CAP],
    yvex_error *err)
{
    yvex_sha256 hash;
    if (!worklist_identity_valid(batch_identity) ||
        !worklist_identity_valid(policy_identity) ||
        !worklist_identity_valid(routing_identity) || !output)
        return worklist_refuse(err, YVEX_ERR_INVALID_ARG,
                               "expert worklist routing identities are incomplete");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.expert-worklist-routing.v1") ||
        !yvex_sha256_update_text(&hash, batch_identity) ||
        !yvex_sha256_update_text(&hash, policy_identity) ||
        !yvex_sha256_update_text(&hash, routing_identity) ||
        !worklist_hash_finish(&hash, output))
        return worklist_refuse(err, YVEX_ERR_STATE,
                               "expert worklist routing identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_expert_worklist_observation_add(
    yvex_expert_worklist_observation *facts,
    const yvex_expert_worklist_observation *delta, yvex_error *err)
{
    yvex_expert_worklist_observation candidate;
    unsigned long long widths = 0ull, populations = 0ull, provenances = 0ull;
    unsigned long long classified_pairs;
    unsigned int index;
    if (!facts || !delta ||
        delta->schema_version != YVEX_EXPERT_WORKLIST_OBSERVATION_SCHEMA_V1 ||
        !delta->worklist_count || !delta->pair_count || !delta->bucket_count ||
        !delta->maximum_bucket_population ||
        !yvex_core_u64_add(delta->tensor_core_eligible_pairs, delta->narrow_pairs,
                           &classified_pairs) || classified_pairs != delta->pair_count)
        return worklist_refuse(err, YVEX_ERR_INVALID_ARG,
                               "expert worklist observation is incomplete");
    for (index = 0u; index < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP; ++index) {
        if (!yvex_core_u64_add(widths, delta->width_histogram[index], &widths) ||
            !yvex_core_u64_add(populations, delta->population_histogram[index],
                               &populations))
            return worklist_refuse(err, YVEX_ERR_BOUNDS,
                                   "expert worklist observation histogram overflowed");
    }
    for (index = 0u; index <= YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE; ++index)
        if (!yvex_core_u64_add(provenances, delta->provenance_counts[index],
                               &provenances))
            return worklist_refuse(err, YVEX_ERR_BOUNDS,
                                   "expert worklist provenance overflowed");
    if (widths != delta->worklist_count || populations != delta->bucket_count ||
        provenances != delta->worklist_count)
        return worklist_refuse(err, YVEX_ERR_FORMAT,
                               "expert worklist observation histograms diverge");
    candidate = *facts;
    if (!candidate.schema_version)
        candidate.schema_version = YVEX_EXPERT_WORKLIST_OBSERVATION_SCHEMA_V1;
    if (candidate.schema_version != YVEX_EXPERT_WORKLIST_OBSERVATION_SCHEMA_V1 ||
        !yvex_core_u64_add(candidate.worklist_count, delta->worklist_count,
                           &candidate.worklist_count) ||
        !yvex_core_u64_add(candidate.pair_count, delta->pair_count,
                           &candidate.pair_count) ||
        !yvex_core_u64_add(candidate.bucket_count, delta->bucket_count,
                           &candidate.bucket_count) ||
        !yvex_core_u64_add(candidate.tensor_core_eligible_pairs,
                           delta->tensor_core_eligible_pairs,
                           &candidate.tensor_core_eligible_pairs) ||
        !yvex_core_u64_add(candidate.tensor_core_executed_pairs,
                           delta->tensor_core_executed_pairs,
                           &candidate.tensor_core_executed_pairs) ||
        !yvex_core_u64_add(candidate.narrow_pairs, delta->narrow_pairs,
                           &candidate.narrow_pairs) ||
        !yvex_core_u64_add(candidate.tail_rows, delta->tail_rows,
                           &candidate.tail_rows))
        return worklist_refuse(err, YVEX_ERR_BOUNDS,
                               "expert worklist observation counters overflowed");
    if (delta->maximum_bucket_population > candidate.maximum_bucket_population)
        candidate.maximum_bucket_population = delta->maximum_bucket_population;
    for (index = 0u; index < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP; ++index)
        if (!yvex_core_u64_add(candidate.width_histogram[index],
                               delta->width_histogram[index],
                               &candidate.width_histogram[index]) ||
            !yvex_core_u64_add(candidate.population_histogram[index],
                               delta->population_histogram[index],
                               &candidate.population_histogram[index]))
            return worklist_refuse(err, YVEX_ERR_BOUNDS,
                                   "expert worklist observation aggregation overflowed");
    for (index = 0u; index <= YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE; ++index)
        if (!yvex_core_u64_add(candidate.provenance_counts[index],
                               delta->provenance_counts[index],
                               &candidate.provenance_counts[index]))
            return worklist_refuse(err, YVEX_ERR_BOUNDS,
                                   "expert worklist provenance aggregation overflowed");
    *facts = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}
