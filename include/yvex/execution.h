/* Quantitative execution facts shared by runtime reports, protocol, and clients. */
#ifndef YVEX_EXECUTION_H
#define YVEX_EXECUTION_H

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_EXECUTION_CAPACITY_SCHEMA_V1 1u
#define YVEX_EXECUTION_MEASUREMENT_SCHEMA_V1 1u
#define YVEX_EXECUTION_RESOURCE_SCHEMA_V1 1u

/*
 * Runnable work is scheduler-visible logical concurrency. Physical width is
 * the number of sequences one admitted backend operation may execute together.
 * Continuous batching is reserved for dynamic membership of physical decode
 * batches; cooperative scheduling and compatible-operation coalescing do not
 * imply it. The legacy engine-summary concurrent_sequences field is the
 * configured physical-width request, not logical runnable concurrency.
 */
typedef struct {
    unsigned int schema_version;
    unsigned long long session_capacity;
    unsigned long long runnable_work_capacity;
    unsigned long long physical_sequence_width;
    int cooperative_scheduling_ready;
    int compatible_operation_batching_ready;
    int continuous_batching_ready;
} yvex_execution_capacity_summary;

typedef enum {
    YVEX_EXECUTION_SCOPE_UNAVAILABLE = 0,
    YVEX_EXECUTION_SCOPE_QUEUE,
    YVEX_EXECUTION_SCOPE_TOKENIZER,
    YVEX_EXECUTION_SCOPE_PROMPT_RENDERING,
    YVEX_EXECUTION_SCOPE_PREFILL,
    YVEX_EXECUTION_SCOPE_FIRST_DECODE,
    YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE,
    YVEX_EXECUTION_SCOPE_MODEL_FORWARD,
    YVEX_EXECUTION_SCOPE_ATTENTION,
    YVEX_EXECUTION_SCOPE_MODEL_COMPONENT,
    YVEX_EXECUTION_SCOPE_OUTPUT,
    YVEX_EXECUTION_SCOPE_LOGITS_PUBLICATION,
    YVEX_EXECUTION_SCOPE_SAMPLING,
    YVEX_EXECUTION_SCOPE_STATE_COMMIT,
    YVEX_EXECUTION_SCOPE_SYNCHRONIZATION,
    YVEX_EXECUTION_SCOPE_DETOKENIZATION,
    YVEX_EXECUTION_SCOPE_CLIENT_PUBLICATION,
    YVEX_EXECUTION_SCOPE_MODEL_LIFECYCLE,
    YVEX_EXECUTION_SCOPE_TOTAL_OPERATION,
    YVEX_EXECUTION_SCOPE_UNATTRIBUTED
} yvex_execution_measurement_scope;

typedef enum {
    YVEX_EXECUTION_CLOCK_UNAVAILABLE = 0,
    YVEX_EXECUTION_CLOCK_HOST_WALL,
    YVEX_EXECUTION_CLOCK_DEVICE,
    YVEX_EXECUTION_CLOCK_MIXED
} yvex_execution_measurement_clock;

typedef enum {
    YVEX_EXECUTION_COMPOSITION_UNAVAILABLE = 0,
    YVEX_EXECUTION_COMPOSITION_TOP_LEVEL,
    YVEX_EXECUTION_COMPOSITION_NESTED,
    YVEX_EXECUTION_COMPOSITION_ENCLOSING,
    YVEX_EXECUTION_COMPOSITION_OVERLAPPING
} yvex_execution_measurement_composition;

typedef enum {
    YVEX_EXECUTION_WORK_NONE = 0,
    YVEX_EXECUTION_WORK_TOKENS,
    YVEX_EXECUTION_WORK_BYTES,
    YVEX_EXECUTION_WORK_TENSORS,
    YVEX_EXECUTION_WORK_PLANS,
    YVEX_EXECUTION_WORK_COMPONENTS,
    YVEX_EXECUTION_WORK_EVALUATIONS,
    YVEX_EXECUTION_WORK_FRAMES,
    YVEX_EXECUTION_WORK_SAMPLES,
    YVEX_EXECUTION_WORK_OPERATIONS
} yvex_execution_work_unit;

#define YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE (1ull << 0u)
#define YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE (1ull << 1u)
#define YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE (1ull << 2u)
#define YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE (1ull << 3u)

/*
 * Rate denominators are explicit. A rolling rate covers rolling_units over
 * rolling_duration_ns and never reuses the cumulative operation denominator.
 */
typedef struct {
    unsigned int schema_version;
    yvex_execution_measurement_scope scope;
    yvex_execution_measurement_clock clock;
    yvex_execution_measurement_composition composition;
    yvex_execution_work_unit work_unit;
    unsigned long long available;
    unsigned long long completed_units, total_units, duration_ns;
    unsigned long long rolling_units, rolling_duration_ns;
    unsigned long long rolling_window_units;
    double cumulative_rate, rolling_rate;
} yvex_execution_measurement;

typedef enum {
    YVEX_EXECUTION_PLACEMENT_UNKNOWN = 0,
    YVEX_EXECUTION_PLACEMENT_EXPLICIT_HOST,
    YVEX_EXECUTION_PLACEMENT_MANAGED_UNIFIED,
    YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED,
    YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED_DEVICE_ADDRESSABLE,
    YVEX_EXECUTION_PLACEMENT_COMPOSITE
} yvex_execution_resource_placement;

#define YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE (1ull << 0u)
#define YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE (1ull << 1u)
#define YVEX_EXECUTION_RESOURCE_ARENA_AVAILABLE (1ull << 2u)
#define YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE (1ull << 3u)
#define YVEX_EXECUTION_RESOURCE_TRANSIENT_AVAILABLE (1ull << 4u)
#define YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE (1ull << 5u)
#define YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE (1ull << 6u)
#define YVEX_EXECUTION_RESOURCE_UNIFIED_MEMORY (1ull << 7u)

/*
 * Semantic ownership and physical observations are deliberately separate.
 * artifact/mapped/device-addressable spans may overlap; typed session-state
 * bytes are contained in physical_state_bytes; peak workspace/transient
 * classes may overlap in time and ownership. Consumers must not add every
 * field into a synthetic total.
 */
typedef struct {
    unsigned int schema_version;
    yvex_execution_resource_placement placement;
    unsigned long long available;
    unsigned long long component_count;
    unsigned long long model_artifact_bytes, model_mapped_bytes;
    unsigned long long model_prepared_bytes;
    unsigned long long model_explicit_host_bytes;
    unsigned long long model_explicit_device_bytes;
    unsigned long long model_device_addressable_bytes;
    unsigned long long session_attention_allocated_bytes;
    unsigned long long session_attention_resident_bytes;
    unsigned long long session_attention_virtual_bytes;
    unsigned long long session_attention_page_table_bytes;
    unsigned long long session_recurrent_state_bytes;
    unsigned long long session_convolution_state_bytes;
    unsigned long long session_candidate_state_bytes;
    unsigned long long session_physical_state_bytes;
    unsigned long long activation_arena_current_bytes;
    unsigned long long activation_arena_peak_bytes;
    unsigned long long workspace_current_bytes, workspace_peak_bytes;
    unsigned long long transient_current_bytes, transient_peak_bytes;
    unsigned long long process_rss_current_bytes, process_rss_peak_bytes;
    unsigned long long logical_upload_bytes, logical_download_bytes;
} yvex_execution_resource_summary;

#ifdef __cplusplus
}
#endif

#endif /* YVEX_EXECUTION_H */
