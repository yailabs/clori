/* Rich operator, benchmark, qualification, and forensic evidence assembled from engine facts. */
#ifndef INCLUDE_YVEX_INTERNAL_EVIDENCE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_EVIDENCE_H_INCLUDED

#include <yvex/internal/runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_PROFILE_SCHEMA_V4 4u

typedef enum {
    YVEX_RUNTIME_PROFILE_OFF = 0,
    YVEX_RUNTIME_PROFILE_SUMMARY,
    YVEX_RUNTIME_PROFILE_STAGES,
    YVEX_RUNTIME_PROFILE_DETAILED
} yvex_runtime_profile_mode;

typedef enum {
    YVEX_RUNTIME_PROFILE_STARTUP = 0,
    YVEX_RUNTIME_PROFILE_PREFILL,
    YVEX_RUNTIME_PROFILE_DECODE,
    YVEX_RUNTIME_PROFILE_GENERATION
} yvex_runtime_profile_scope;

typedef enum {
    YVEX_RUNTIME_PROFILE_QUEUE = 0,
    YVEX_RUNTIME_PROFILE_TOKENIZER,
    YVEX_RUNTIME_PROFILE_PROMPT_RENDERING,
    YVEX_RUNTIME_PROFILE_EMBEDDING,
    YVEX_RUNTIME_PROFILE_ATTENTION,
    YVEX_RUNTIME_PROFILE_MOE_INGRESS,
    YVEX_RUNTIME_PROFILE_ROUTER_PROJECTION,
    YVEX_RUNTIME_PROFILE_ROUTER_TOPK,
    YVEX_RUNTIME_PROFILE_SELECTED_EXPERT_PREPARATION,
    YVEX_RUNTIME_PROFILE_ROUTED_EXPERTS,
    YVEX_RUNTIME_PROFILE_SHARED_EXPERTS,
    YVEX_RUNTIME_PROFILE_MOE_POST,
    YVEX_RUNTIME_PROFILE_MOE_TOTAL,
    YVEX_RUNTIME_PROFILE_FINAL_NORMALIZATION,
    YVEX_RUNTIME_PROFILE_OUTPUT_HEAD,
    YVEX_RUNTIME_PROFILE_LOGITS_PUBLICATION,
    YVEX_RUNTIME_PROFILE_SAMPLING,
    YVEX_RUNTIME_PROFILE_STATE_VALIDATION,
    YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT,
    YVEX_RUNTIME_PROFILE_KV_COMMIT,
    YVEX_RUNTIME_PROFILE_DETOKENIZATION,
    YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION,
    YVEX_RUNTIME_PROFILE_TOTAL_PREFILL,
    YVEX_RUNTIME_PROFILE_FIRST_DECODE,
    YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE,
    YVEX_RUNTIME_PROFILE_TOTAL_GENERATION,
    YVEX_RUNTIME_PROFILE_PHASE_COUNT
} yvex_runtime_profile_phase;

typedef enum {
    YVEX_RUNTIME_PROFILE_HOST_PAYLOAD_READS = 0,
    YVEX_RUNTIME_PROFILE_MAPPED_BYTES_TOUCHED,
    YVEX_RUNTIME_PROFILE_H2D_BYTES,
    YVEX_RUNTIME_PROFILE_D2H_BYTES,
    YVEX_RUNTIME_PROFILE_D2D_BYTES,
    YVEX_RUNTIME_PROFILE_MANAGED_PREFETCH_BYTES,
    YVEX_RUNTIME_PROFILE_UPLOADS,
    YVEX_RUNTIME_PROFILE_DOWNLOADS,
    YVEX_RUNTIME_PROFILE_CACHE_HITS,
    YVEX_RUNTIME_PROFILE_CACHE_MISSES,
    YVEX_RUNTIME_PROFILE_CACHE_EVICTIONS,
    YVEX_RUNTIME_PROFILE_EXPERT_SUBVIEWS,
    YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES,
    YVEX_RUNTIME_PROFILE_TENSOR_CORE_LAUNCHES,
    YVEX_RUNTIME_PROFILE_GRAPH_LAUNCHES,
    YVEX_RUNTIME_PROFILE_GRAPH_CAPTURES,
    YVEX_RUNTIME_PROFILE_GRAPH_REPLAYS,
    YVEX_RUNTIME_PROFILE_STREAM_SYNCHRONIZATIONS,
    YVEX_RUNTIME_PROFILE_EVENT_SYNCHRONIZATIONS,
    YVEX_RUNTIME_PROFILE_DEVICE_SYNCHRONIZATIONS,
    YVEX_RUNTIME_PROFILE_DEVICE_ALLOCATIONS,
    YVEX_RUNTIME_PROFILE_HOST_ALLOCATIONS,
    YVEX_RUNTIME_PROFILE_WORKSPACE_RESETS,
    YVEX_RUNTIME_PROFILE_PROMPT_TOKENS,
    YVEX_RUNTIME_PROFILE_REUSED_TOKENS,
    YVEX_RUNTIME_PROFILE_NEW_PREFILL_TOKENS,
    YVEX_RUNTIME_PROFILE_GENERATED_TOKENS,
    YVEX_RUNTIME_PROFILE_TARGET_FORWARDS,
    YVEX_RUNTIME_PROFILE_TARGET_ROWS,
    YVEX_RUNTIME_PROFILE_DRAFT_FORWARDS,
    YVEX_RUNTIME_PROFILE_DRAFT_ROWS,
    YVEX_RUNTIME_PROFILE_TARGET_VERIFICATIONS,
    YVEX_RUNTIME_PROFILE_VERIFIED_ROWS,
    YVEX_RUNTIME_PROFILE_ACCEPTED_DRAFT_TOKENS,
    YVEX_RUNTIME_PROFILE_PROMOTED_TARGET_ROWS,
    YVEX_RUNTIME_PROFILE_DISCARDED_CANDIDATE_ROWS,
    YVEX_RUNTIME_PROFILE_TARGET_EXTENSIONS,
    YVEX_RUNTIME_PROFILE_REPLAYED_ACCEPTED_TARGET_ROWS,
    YVEX_RUNTIME_PROFILE_OUTPUT_HEAD_ROWS,
    YVEX_RUNTIME_PROFILE_LOGITS_H2D_BYTES,
    YVEX_RUNTIME_PROFILE_LOGITS_D2H_BYTES,
    YVEX_RUNTIME_PROFILE_LOGITS_D2D_BYTES,
    YVEX_RUNTIME_PROFILE_FULL_ARRAY_HOST_SCAN_BYTES,
    YVEX_RUNTIME_PROFILE_ROW_EXPERT_PAIRS,
    YVEX_RUNTIME_PROFILE_UNIQUE_EXPERTS,
    YVEX_RUNTIME_PROFILE_EXPERT_BYTES,
    YVEX_RUNTIME_PROFILE_COUNTER_COUNT
} yvex_runtime_profile_counter;

typedef struct {
    unsigned int schema_version, backend;
    yvex_runtime_profile_mode mode;
    yvex_runtime_profile_scope scope;
    int sealed;
    unsigned long long started_ns, completed_ns;
    unsigned long long counters[YVEX_RUNTIME_PROFILE_COUNTER_COUNT];
    unsigned long long phase_ns[YVEX_RUNTIME_PROFILE_PHASE_COUNT];
    unsigned long long phase_calls[YVEX_RUNTIME_PROFILE_PHASE_COUNT];
    char artifact_identity[YVEX_SHA256_HEX_BYTES];
    char physical_variant_identity[YVEX_SHA256_HEX_BYTES];
    char runtime_binding_identity[YVEX_SHA256_HEX_BYTES];
    char runtime_model_identity[YVEX_SHA256_HEX_BYTES];
    char execution_plan_identity[YVEX_SHA256_HEX_BYTES];
    char workload_identity[YVEX_SHA256_HEX_BYTES];
    char profile_identity[YVEX_SHA256_HEX_BYTES];
} yvex_runtime_profile_record;

int yvex_runtime_profile_begin(
    yvex_runtime_profile_record *record, yvex_runtime_profile_mode mode,
    yvex_runtime_profile_scope scope, unsigned int backend,
    const char *artifact_identity, const char *physical_variant_identity,
    const char *runtime_binding_identity, const char *runtime_model_identity,
    const char *execution_plan_identity, const char *workload_identity,
    yvex_error *err);
int yvex_runtime_profile_counter_add(
    yvex_runtime_profile_record *record, yvex_runtime_profile_counter counter,
    unsigned long long value, yvex_error *err);
int yvex_runtime_profile_phase_add(
    yvex_runtime_profile_record *record, yvex_runtime_profile_phase phase,
    unsigned long long elapsed_ns, yvex_error *err);
int yvex_runtime_profile_finish(
    yvex_runtime_profile_record *record, yvex_error *err);
int yvex_runtime_profile_validate(
    const yvex_runtime_profile_record *record, yvex_error *err);
const char *yvex_runtime_profile_mode_name(yvex_runtime_profile_mode mode);
const char *yvex_runtime_profile_phase_name(yvex_runtime_profile_phase phase);
const char *yvex_runtime_profile_counter_name(
    yvex_runtime_profile_counter counter);

#define YVEX_EXECUTION_PHASE_ROOFLINE_SCHEMA_V1 1u

typedef enum {
    YVEX_EXECUTION_ROOFLINE_PREFILL_LAYER = 0,
    YVEX_EXECUTION_ROOFLINE_DECODE_LAYER,
    YVEX_EXECUTION_ROOFLINE_VERIFY_SWEEP,
    YVEX_EXECUTION_ROOFLINE_DRAFT_SWEEP,
    YVEX_EXECUTION_ROOFLINE_OUTPUT_HEAD,
    YVEX_EXECUTION_ROOFLINE_STATE_PROMOTION,
    YVEX_EXECUTION_ROOFLINE_BATCHED_DECODE,
    YVEX_EXECUTION_ROOFLINE_PHASE_COUNT
} yvex_execution_roofline_phase;

typedef enum {
    YVEX_EXECUTION_PHASE_FACT_ACTIVE_WEIGHT = 0,
    YVEX_EXECUTION_PHASE_FACT_STATE,
    YVEX_EXECUTION_PHASE_FACT_ACTIVATION,
    YVEX_EXECUTION_PHASE_FACT_TEMPORARY,
    YVEX_EXECUTION_PHASE_FACT_MOVEMENT,
    YVEX_EXECUTION_PHASE_FACT_KERNELS,
    YVEX_EXECUTION_PHASE_FACT_SYNCHRONIZATIONS,
    YVEX_EXECUTION_PHASE_FACT_OCCUPANCY,
    YVEX_EXECUTION_PHASE_FACT_DURATION,
    YVEX_EXECUTION_PHASE_FACT_WORK,
    YVEX_EXECUTION_PHASE_FACT_COMMITTED_TOKENS,
    YVEX_EXECUTION_PHASE_FACT_COUNT
} yvex_execution_phase_fact;

#define YVEX_EXECUTION_PHASE_FACT_BIT(fact) (1ull << (unsigned int)(fact))
#define YVEX_EXECUTION_PHASE_FACT_ALL \
    ((1ull << YVEX_EXECUTION_PHASE_FACT_COUNT) - 1ull)
#define YVEX_EXECUTION_PHASE_MEMORY_FACTS                                      \
    (YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_ACTIVE_WEIGHT) |  \
     YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_STATE) |          \
     YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_ACTIVATION) |     \
     YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_TEMPORARY))

/* Memory facts count compulsory device spans once, not capacity or observed DRAM transactions. */
typedef struct {
    yvex_execution_roofline_phase phase;
    /* Zero is the original v1 representation and means that every fact is available. */
    unsigned long long fact_mask;
    unsigned long long active_weight_bytes, state_bytes, activation_bytes;
    unsigned long long temporary_bytes, h2d_bytes, d2h_bytes, d2d_bytes;
    unsigned long long kernel_count, synchronization_count;
    /* Repeated deltas publish the work-unit-weighted mean occupancy. */
    unsigned long long occupancy_parts_per_million;
    unsigned long long measured_duration_ns, work_units, committed_tokens;
} yvex_execution_phase_measurement;

typedef struct {
    yvex_execution_phase_measurement measurement;
    int available, roofline_available;
    unsigned long long missing_fact_mask;
    unsigned long long active_device_bytes, transfer_bytes;
    unsigned long long minimum_memory_time_ns, measured_bytes_per_second;
    unsigned long long roofline_utilization_parts_per_million;
    unsigned long long optimization_headroom_ns, optimization_priority;
} yvex_execution_phase_roofline;

int yvex_execution_phase_measurement_accumulate(
    yvex_execution_phase_measurement *measurements,
    unsigned long long measurement_capacity,
    unsigned long long *measurement_count,
    const yvex_execution_phase_measurement *delta, yvex_error *err);

typedef struct {
    unsigned int schema_version;
    const yvex_execution_hardware_profile *hardware;
    const char *artifact_identity, *execution_profile_identity;
    const char *kernel_bundle_identity, *workload_profile_identity;
    const yvex_execution_phase_measurement *measurements;
    unsigned long long measurement_count;
} yvex_execution_roofline_ledger_request;

typedef struct {
    unsigned int schema_version;
    unsigned long long phase_count, measured_phase_count;
    unsigned long long measured_phase_mask, missing_phase_mask, rooflined_phase_mask;
    unsigned long long measured_duration_ns, committed_tokens;
    int priority_provisional;
    yvex_execution_phase_roofline phases[YVEX_EXECUTION_ROOFLINE_PHASE_COUNT];
    char hardware_profile_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char execution_profile_identity[YVEX_SHA256_HEX_CAP];
    char kernel_bundle_identity[YVEX_SHA256_HEX_CAP];
    char workload_profile_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_roofline_ledger;

int yvex_execution_roofline_ledger_build(
    const yvex_execution_roofline_ledger_request *request,
    yvex_execution_roofline_ledger *ledger, yvex_error *err);

#define YVEX_RUNTIME_GENERATION_EVIDENCE_SCHEMA_V1 1u

/* Caller-owned observation sidecar; it never participates in committed generation identity. */
typedef struct {
    unsigned int schema_version;
    yvex_runtime_profile_record profile;
    yvex_expert_worklist_observation expert_worklists;
    int roofline_available;
    yvex_execution_roofline_ledger roofline;
} yvex_runtime_generation_evidence;

/* Ordinary authenticated engine dispatch uses compact handles and never hashes this projection. */
typedef struct {
    unsigned int schema_version;
    const char *runtime_model_identity, *runtime_binding_identity;
    const char *artifact_identity, *runtime_numeric_identity;
    const char *runtime_descriptor_identity, *semantic_graph_identity;
    const char *executable_graph_identity, *selected_mode, *capture_bucket;
    const char *residency_identity, *workspace_identity, *capacity_plan_identity;
    const char *state_layout_identity;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned int probe, probe_scope, operation_scope, phase, backend, requested_mode;
    int compare_backends;
    unsigned long long token_count, request_count, start_position;
    unsigned long long layer_start, layer_count, selection_key, binding_count;
    unsigned long long state_component_entries[YVEX_ATTENTION_STATE_BINDING_COUNT];
    unsigned long long state_component_capacities[YVEX_ATTENTION_STATE_BINDING_COUNT];
    unsigned long long maximum_compression_ratio, maximum_topk_capacity;
    unsigned int trace_policy;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    unsigned long long residency_generation, resident_binding_count;
    unsigned long long resident_encoded_bytes, workspace_bytes, workspace_generation;
    unsigned long long prepared_state_layers, state_allocated_bytes, state_generation;
    unsigned long long qtype_binding_counts[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    unsigned long long qtype_bytes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    unsigned int device_kind;
    int device_index, compute_capability_major, compute_capability_minor;
    unsigned long long total_device_bytes;
} yvex_runtime_operator_execution_facts;

int yvex_runtime_operator_execution_identity_compute(
    const yvex_runtime_operator_execution_facts *facts,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
