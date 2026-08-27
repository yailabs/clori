/* Rich operator, benchmark, qualification, and forensic evidence assembled from engine facts. */
#ifndef INCLUDE_YVEX_INTERNAL_EVIDENCE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_EVIDENCE_H_INCLUDED

#include <yvex/internal/runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

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
