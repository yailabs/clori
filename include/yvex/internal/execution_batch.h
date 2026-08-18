/*
 * Bind real execution width to deterministic expert-major work without making a backend infer
 * routing compatibility. Dynamic rows remain request-owned; this contract records their exact
 * provenance, ordering, and bounded physical grouping.
 */
#ifndef INCLUDE_YVEX_INTERNAL_EXECUTION_BATCH_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_EXECUTION_BATCH_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_EXECUTION_BATCH_SCHEMA_V1 1u
#define YVEX_EXECUTION_COMPATIBILITY_SCHEMA_V1 1u
#define YVEX_EXPERT_WORKLIST_POLICY_SCHEMA_V1 1u
#define YVEX_EXPERT_WORKLIST_SCHEMA_V1 1u
#define YVEX_EXPERT_WORKLIST_OBSERVATION_SCHEMA_V1 1u
#define YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP 17u

typedef enum {
    YVEX_EXECUTION_PHASE_PREFILL = 0,
    YVEX_EXECUTION_PHASE_DECODE,
    YVEX_EXECUTION_PHASE_DRAFT,
    YVEX_EXECUTION_PHASE_VERIFY,
    YVEX_EXECUTION_PHASE_CORRECTION,
    YVEX_EXECUTION_PHASE_RESET,
    YVEX_EXECUTION_PHASE_MIXED,
    YVEX_EXECUTION_PHASE_COUNT
} yvex_execution_phase;

typedef enum {
    YVEX_EXECUTION_BATCH_SINGLE_ROW = 0,
    YVEX_EXECUTION_BATCH_SPECULATIVE_VERIFICATION,
    YVEX_EXECUTION_BATCH_MULTI_SESSION,
    YVEX_EXECUTION_BATCH_PREFILL,
    YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE
} yvex_execution_batch_provenance;

typedef struct {
    unsigned long long execution_generation, state_generation;
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_batch_source;

typedef struct {
    unsigned long long source_index, source_row, sequence_position;
    unsigned long long candidate_ordinal, publication_ordinal;
    int candidate_present;
} yvex_execution_batch_row;

typedef struct {
    unsigned int schema_version;
    yvex_execution_batch_provenance provenance;
    yvex_execution_phase phase;
    unsigned long long row_count, source_count, model_generation;
    const yvex_execution_batch_source *sources;
    const yvex_execution_batch_row *rows;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char physical_variant_identity[YVEX_SHA256_HEX_CAP];
    char execution_profile_identity[YVEX_SHA256_HEX_CAP];
    char operation_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_batch;

/*
 * Stable compiled/runtime facts that prove rows may enter one physical operation. Dynamic
 * session, candidate, and row ownership stays in yvex_execution_batch_source/row and therefore
 * cannot be mistaken for compatibility merely because two buffers have equal extents.
 */
typedef struct {
    unsigned int schema_version;
    yvex_execution_phase phase;
    unsigned int backend_kind, tensor_scope, execution_class, publication_contract;
    unsigned long long model_generation, layer_ordinal, row_width, admitted_width;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char physical_variant_identity[YVEX_SHA256_HEX_CAP];
    char execution_profile_identity[YVEX_SHA256_HEX_CAP];
    char operation_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_compatibility_key;

typedef struct {
    unsigned int schema_version;
    unsigned long long supported_width_mask;
    unsigned long long tensor_core_minimum;
    char narrow_kernel_family[64];
    char tensor_core_kernel_family[64];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_expert_worklist_policy;

typedef struct {
    unsigned int schema_version;
    const yvex_execution_batch *batch;
    const yvex_expert_worklist_policy *policy;
    unsigned long long expert_count, experts_per_row, pair_count;
    const unsigned long long *selected_experts;
    const float *route_weights;
} yvex_expert_worklist_request;

typedef struct {
    unsigned long long *expert_ids, *bucket_offsets, *bucket_populations;
    unsigned long long *source_pairs, *source_rows, *destination_rows;
    float *route_weights;
    unsigned long long bucket_capacity, pair_capacity;
} yvex_expert_worklist_storage;

typedef struct {
    unsigned int schema_version;
    yvex_execution_batch_provenance provenance;
    yvex_execution_phase phase;
    unsigned long long actual_width, expert_count, pair_count, bucket_count;
    unsigned long long maximum_bucket_population, admitted_tile_width;
    unsigned long long tensor_core_eligible_pairs, narrow_pairs, tail_rows;
    const unsigned long long *expert_ids, *bucket_offsets, *bucket_populations;
    const unsigned long long *source_pairs, *source_rows, *destination_rows;
    const float *route_weights;
    unsigned long long population_histogram[YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP];
    char batch_identity[YVEX_SHA256_HEX_CAP];
    char policy_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_expert_worklist;

/* Pointer-free additive evidence copied after a device builder materializes the same schema. */
typedef struct {
    unsigned int schema_version;
    unsigned long long worklist_count, pair_count, bucket_count;
    unsigned long long maximum_bucket_population;
    unsigned long long tensor_core_eligible_pairs, tensor_core_executed_pairs;
    unsigned long long narrow_pairs, tail_rows;
    unsigned long long width_histogram[YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP];
    unsigned long long population_histogram[YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP];
    unsigned long long provenance_counts[YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE + 1u];
} yvex_expert_worklist_observation;

int yvex_execution_batch_seal(yvex_execution_batch *batch, yvex_error *err);
int yvex_execution_batch_validate(const yvex_execution_batch *batch,
                                  yvex_error *err);
int yvex_execution_compatibility_key_seal(
    yvex_execution_compatibility_key *key, yvex_error *err);
int yvex_execution_compatibility_key_validate(
    const yvex_execution_compatibility_key *key, yvex_error *err);
int yvex_execution_compatibility_keys_match(
    const yvex_execution_compatibility_key *left,
    const yvex_execution_compatibility_key *right, yvex_error *err);
int yvex_expert_worklist_compiled_policy_valid(
    unsigned long long supported_width_mask,
    unsigned long long tensor_core_minimum,
    const char *tensor_core_kernel_family);
int yvex_expert_worklist_policy_seal(yvex_expert_worklist_policy *policy,
                                     yvex_error *err);
int yvex_expert_worklist_policy_validate(
    const yvex_expert_worklist_policy *policy, yvex_error *err);
int yvex_expert_worklist_build(const yvex_expert_worklist_request *request,
                               const yvex_expert_worklist_storage *storage,
                               yvex_expert_worklist *worklist, yvex_error *err);
int yvex_expert_worklist_validate(const yvex_expert_worklist_request *request,
                                  const yvex_expert_worklist *worklist,
                                  yvex_error *err);
int yvex_expert_worklist_routing_identity(
    const char *batch_identity, const char *policy_identity,
    const char *routing_identity, char output[YVEX_SHA256_HEX_CAP],
    yvex_error *err);
int yvex_expert_worklist_observation_add(
    yvex_expert_worklist_observation *facts,
    const yvex_expert_worklist_observation *delta, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_EXECUTION_BATCH_H_INCLUDED */
