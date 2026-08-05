/*
 * Bind physical tensor decisions, request execution policy, device-resident values, and
 * attention shapes without allowing runtime consumers to recover those facts from names.
 */
#ifndef INCLUDE_YVEX_INTERNAL_EXECUTION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_EXECUTION_H_INCLUDED

#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/model.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_PHYSICAL_EXECUTION_SCHEMA_V1 1u
#define YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V1 1u
#define YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1 1u
#define YVEX_EXECUTION_WORKLOAD_PROFILE_SCHEMA_V1 1u
#define YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1 1u
#define YVEX_EXECUTION_PHASE_ROOFLINE_SCHEMA_V1 1u
#define YVEX_EXECUTION_SHAPE_SCHEMA_V1 1u
#define YVEX_EXECUTION_DEVICE_VIEW_SCHEMA_V1 1u
#define YVEX_EXECUTION_TEXT_CAP 64u
#define YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE (8ull * 1024ull * 1024ull * 1024ull)

typedef enum {
    YVEX_EXECUTION_EVIDENCE_PRODUCTION = 0,
    YVEX_EXECUTION_EVIDENCE_AUDIT,
    YVEX_EXECUTION_EVIDENCE_FORENSIC
} yvex_execution_evidence_profile;

typedef enum {
    YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE = 0,
    YVEX_EXECUTION_CLASS_DEVICE_NATIVE,
    YVEX_EXECUTION_CLASS_FORENSIC_REFERENCE
} yvex_execution_class;

typedef enum {
    YVEX_EXECUTION_CONSUMER_EMBEDDING = 0,
    YVEX_EXECUTION_CONSUMER_ATTENTION_PROJECTION,
    YVEX_EXECUTION_CONSUMER_ATTENTION_STATE,
    YVEX_EXECUTION_CONSUMER_MOE_ROUTER,
    YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP,
    YVEX_EXECUTION_CONSUMER_ROUTED_DOWN,
    YVEX_EXECUTION_CONSUMER_SHARED_EXPERT,
    YVEX_EXECUTION_CONSUMER_FINAL_NORMALIZATION,
    YVEX_EXECUTION_CONSUMER_OUTPUT_HEAD,
    YVEX_EXECUTION_CONSUMER_DRAFT_FEATURE_PROJECTION,
    YVEX_EXECUTION_CONSUMER_DRAFT_BACKBONE,
    YVEX_EXECUTION_CONSUMER_MARKOV,
    YVEX_EXECUTION_CONSUMER_CONFIDENCE,
    YVEX_EXECUTION_CONSUMER_COUNT
} yvex_execution_consumer_class;

typedef enum {
    YVEX_EXECUTION_LAYOUT_CANONICAL_ROW = 0,
    YVEX_EXECUTION_LAYOUT_CONTIGUOUS_DENSE,
    YVEX_EXECUTION_LAYOUT_EXPERT_MAJOR,
    YVEX_EXECUTION_LAYOUT_DERIVED_BACKEND
} yvex_execution_layout_class;

typedef enum {
    YVEX_EXECUTION_PLACEMENT_FILE_BACKED = 0,
    YVEX_EXECUTION_PLACEMENT_HOST_CANONICAL,
    YVEX_EXECUTION_PLACEMENT_CUDA_ADDRESSABLE_HOST,
    YVEX_EXECUTION_PLACEMENT_DEVICE,
    YVEX_EXECUTION_PLACEMENT_STAGED
} yvex_execution_placement_class;

typedef enum {
    YVEX_EXECUTION_SHARING_EXCLUSIVE = 0,
    YVEX_EXECUTION_SHARING_MODEL_READ_ONLY,
    YVEX_EXECUTION_SHARING_ALIAS
} yvex_execution_sharing_class;

typedef enum {
    YVEX_EXECUTION_HARDWARE_FACT_MEMORY = 0,
    YVEX_EXECUTION_HARDWARE_FACT_BANDWIDTH,
    YVEX_EXECUTION_HARDWARE_FACT_TOPOLOGY,
    YVEX_EXECUTION_HARDWARE_FACT_PAGING,
    YVEX_EXECUTION_HARDWARE_FACT_GRAPH,
    YVEX_EXECUTION_HARDWARE_FACT_NATIVE_CODE,
    YVEX_EXECUTION_HARDWARE_FACT_COUNT
} yvex_execution_hardware_fact;

#define YVEX_EXECUTION_HARDWARE_FACT_BIT(fact) (1ull << (unsigned int)(fact))

typedef enum {
    YVEX_EXECUTION_ACTIVATION_HOST_F32 = 0,
    YVEX_EXECUTION_ACTIVATION_DEVICE_F32,
    YVEX_EXECUTION_ACTIVATION_DEVICE_F16,
    YVEX_EXECUTION_ACTIVATION_DEVICE_ENCODED
} yvex_execution_activation_class;

typedef enum {
    YVEX_EXECUTION_BACKEND_ANY = 0,
    YVEX_EXECUTION_BACKEND_CPU,
    YVEX_EXECUTION_BACKEND_CUDA
} yvex_execution_backend_requirement;

typedef struct {
    unsigned int schema_version;
    unsigned long long terminal_tensor_id;
    yvex_tensor_role role;
    yvex_tensor_scope scope;
    unsigned long long layer_index, predictor_index, expert_count;
    unsigned int canonical_qtype;
    unsigned long long canonical_row_width, canonical_row_count;
    unsigned long long encoded_offset, encoded_bytes, alignment;
    yvex_execution_consumer_class consumer;
    yvex_execution_layout_class layout;
    yvex_execution_placement_class placement;
    yvex_execution_sharing_class sharing;
    yvex_execution_activation_class activation;
    unsigned long long supported_width_mask, maximum_context;
    yvex_execution_backend_requirement required_backend;
    unsigned int required_compute_major, required_compute_minor;
    yvex_execution_evidence_profile evidence;
    yvex_execution_class fallback;
    int derived_asset_required;
    char kernel_family[YVEX_EXECUTION_TEXT_CAP];
    char terminal_identity[YVEX_SHA256_HEX_CAP];
    char decision_identity[YVEX_SHA256_HEX_CAP];
} yvex_physical_execution_decision;

typedef struct {
    unsigned int schema_version;
    unsigned long long decision_count, encoded_bytes;
    unsigned long long consumer_counts[YVEX_EXECUTION_CONSUMER_COUNT];
    unsigned long long layout_counts[4], placement_counts[5];
    char physical_variant_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_physical_execution_summary;

typedef struct yvex_physical_execution_ir yvex_physical_execution_ir;

int yvex_physical_execution_ir_build(
    yvex_physical_execution_ir **out,
    const yvex_materialization_session *materialization,
    const yvex_runtime_descriptor *descriptor,
    const char *physical_variant_identity, yvex_error *err);
const yvex_physical_execution_summary *yvex_physical_execution_ir_summary(
    const yvex_physical_execution_ir *ir);
void yvex_physical_execution_ir_close(yvex_physical_execution_ir **ir);

typedef struct {
    unsigned int schema_version;
    yvex_backend_kind backend;
    unsigned long long admitted_fact_mask;
    int device_index, compute_major, compute_minor;
    unsigned long long sm_count, copy_engine_count, l2_bytes;
    unsigned long long total_memory_bytes, usable_memory_bytes;
    unsigned long long sustainable_read_bytes_per_second;
    unsigned long long sustainable_copy_bytes_per_second;
    unsigned long long host_page_bytes, device_page_bytes;
    int unified_addressing, coherent_host_memory, virtual_memory, graph_capture;
    int native_architecture_code;
    char name[YVEX_EXECUTION_TEXT_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_hardware_profile;

int yvex_execution_hardware_profile_seal(
    yvex_execution_hardware_profile *profile, yvex_error *err);

typedef enum {
    YVEX_EXECUTION_WORKLOAD_INTERACTIVE_LATENCY = 0,
    YVEX_EXECUTION_WORKLOAD_BALANCED_SERVING,
    YVEX_EXECUTION_WORKLOAD_LONG_CONTEXT,
    YVEX_EXECUTION_WORKLOAD_DEEP_CONTEXT,
    YVEX_EXECUTION_WORKLOAD_FULL_MODEL_RESEARCH
} yvex_execution_workload_profile_kind;

typedef struct {
    unsigned int schema_version;
    yvex_execution_workload_profile_kind kind;
    unsigned long long minimum_session_context, requested_session_context;
    unsigned long long concurrent_sequences, logical_batch_tokens;
    unsigned long long prefill_chunk_tokens, attention_microbatch_rows;
    unsigned long long moe_row_tile, output_head_rows;
    unsigned long long prefix_cache_bytes, persistent_state_bytes;
    unsigned long long system_reserve_bytes;
    int latency_priority, continuous_batching, prefix_sharing, durable_state;
    char name[YVEX_EXECUTION_TEXT_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_workload_profile;

int yvex_execution_workload_profile_seal(
    yvex_execution_workload_profile *profile, yvex_error *err);

typedef enum {
    YVEX_EXECUTION_STATE_EXTENT_CONTEXT = 0,
    YVEX_EXECUTION_STATE_EXTENT_FIXED,
    YVEX_EXECUTION_STATE_EXTENT_CANDIDATE,
    YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET
} yvex_execution_state_extent;

typedef struct {
    yvex_model_state_class state_class;
    yvex_execution_state_extent extent;
    unsigned long long logical_block_tokens, bytes_per_block;
    unsigned long long fixed_tokens_per_sequence;
    unsigned long long alignment_bytes, kernel_tile_tokens;
    unsigned long long promotion_granularity_tokens, page_table_entry_bytes;
    int shared, copy_on_write;
} yvex_execution_state_class_request;

typedef struct {
    yvex_model_state_class state_class;
    yvex_execution_state_extent extent;
    unsigned long long logical_block_tokens, bytes_per_block;
    unsigned long long page_tokens, page_bytes;
    unsigned long long tokens_per_sequence, pool_tokens, pool_bytes;
    unsigned long long page_count, page_table_bytes;
    unsigned long long tail_fragmentation_bytes, copy_on_write_tail_bytes;
    unsigned long long promotion_fragmentation_bytes;
    int shared, copy_on_write;
} yvex_execution_state_class_plan;

typedef struct {
    unsigned int schema_version;
    const yvex_model_execution_descriptor *model;
    const yvex_execution_hardware_profile *hardware;
    const yvex_execution_workload_profile *workload;
    unsigned long long model_bytes, derived_layout_bytes;
    const yvex_execution_state_class_request *state_classes;
    unsigned long long state_class_count;
    unsigned long long workspace_bytes, scheduler_bytes, graph_bytes;
} yvex_execution_capacity_plan_request;

typedef struct {
    unsigned int schema_version;
    unsigned long long model_maximum_context, admitted_execution_maximum;
    unsigned long long per_session_maximum, per_request_maximum;
    unsigned long long total_logical_context_tokens, physical_state_pool_tokens;
    unsigned long long candidate_reserve_tokens, concurrent_sequences;
    unsigned long long logical_batch_tokens, attention_microbatch_rows;
    unsigned long long moe_row_tile, output_head_rows;
    unsigned long long model_bytes, derived_layout_bytes, state_pool_bytes;
    unsigned long long candidate_reserve_bytes, workspace_bytes;
    unsigned long long scheduler_bytes, graph_bytes, prefix_cache_bytes;
    unsigned long long persistent_state_bytes, system_reserve_bytes;
    unsigned long long required_bytes, usable_memory_bytes, unreserved_bytes;
    unsigned long long state_class_count;
    yvex_execution_state_class_plan state_classes[YVEX_MODEL_STATE_CLASS_COUNT];
    char model_execution_identity[YVEX_SHA256_HEX_CAP];
    char hardware_profile_identity[YVEX_SHA256_HEX_CAP];
    char workload_profile_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_capacity_plan;

int yvex_execution_capacity_plan_build(
    const yvex_execution_capacity_plan_request *request,
    yvex_execution_capacity_plan *plan, yvex_error *err);

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
#define YVEX_EXECUTION_PHASE_MEMORY_FACTS                                                \
    (YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_ACTIVE_WEIGHT) |           \
     YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_STATE) |                   \
     YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_ACTIVATION) |              \
     YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_TEMPORARY))

typedef struct {
    unsigned long long active_weight_bytes, state_bytes, activation_bytes, temporary_bytes;
    unsigned long long measured_operations, missing_operations;
    int complete;
} yvex_execution_memory_facts;

int yvex_execution_memory_facts_add(
    yvex_execution_memory_facts *facts, unsigned long long active_weight_bytes,
    unsigned long long state_bytes, unsigned long long activation_bytes,
    unsigned long long temporary_bytes, unsigned long long measured_operations,
    unsigned long long missing_operations, yvex_error *err);
int yvex_execution_memory_facts_merge(
    yvex_execution_memory_facts *facts,
    const yvex_execution_memory_facts *delta, yvex_error *err);

typedef struct {
    yvex_execution_memory_facts memory;
    unsigned long long h2d_bytes, d2h_bytes, d2d_bytes;
    unsigned long long kernel_count, synchronization_count;
} yvex_execution_physical_facts;
int yvex_execution_physical_facts_add(
    yvex_execution_physical_facts *facts,
    const yvex_execution_memory_facts *memory, unsigned long long h2d_bytes,
    unsigned long long d2h_bytes, unsigned long long d2d_bytes,
    unsigned long long kernel_count, unsigned long long synchronization_count,
    yvex_error *err);
int yvex_execution_f32_hash_update(
    yvex_sha256 *hash, const float *values, unsigned long long count);
int yvex_execution_f32_digest(
    const char *domain, const float *values, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP]);

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

/* Adds one borrowed delta to caller-owned phase storage with checked counters.
 * Repeated deltas for a phase must expose exactly the same fact set. */
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

typedef enum {
    YVEX_EXECUTION_WORKLOAD_INTERACTIVE = 0,
    YVEX_EXECUTION_WORKLOAD_BATCH,
    YVEX_EXECUTION_WORKLOAD_QUALIFICATION
} yvex_execution_workload_class;

typedef enum {
    YVEX_EXECUTION_GENERATION_TARGET_ONLY = 0,
    YVEX_EXECUTION_GENERATION_SPECULATIVE
} yvex_execution_generation_mode;

typedef struct {
    unsigned int schema_version;
    const char *logical_model_identity;
    const char *physical_variant_identity;
    const char *physical_execution_identity;
    const char *artifact_identity;
    const char *materialization_identity;
    const char *runtime_binding_identity;
    const char *kernel_bundle_identity;
    const char *hardware_profile;
    yvex_backend_kind backend;
    int device_index, compute_major, compute_minor;
    unsigned long long context_capacity;
    yvex_execution_generation_mode generation_mode;
    yvex_execution_workload_class workload;
    yvex_execution_evidence_profile evidence;
    yvex_execution_class execution_class;
    int host_stochastic_reference;
    int token_local_moe_reference;
    int eager_attention_reference;
} yvex_compiled_execution_profile_request;

typedef struct {
    unsigned int schema_version;
    yvex_backend_kind backend;
    int device_index, compute_major, compute_minor;
    unsigned long long context_capacity;
    yvex_execution_generation_mode generation_mode;
    yvex_execution_workload_class workload;
    yvex_execution_evidence_profile evidence;
    yvex_execution_class execution_class;
    int host_stochastic_reference;
    int token_local_moe_reference;
    int eager_attention_reference;
    char hardware_profile[YVEX_EXECUTION_TEXT_CAP];
    char kernel_bundle_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_compiled_execution_profile;

int yvex_compiled_execution_profile_seal(
    const yvex_compiled_execution_profile_request *request,
    yvex_compiled_execution_profile *profile, yvex_error *err);

typedef enum {
    YVEX_EXECUTION_DEVICE_HIDDEN = 0,
    YVEX_EXECUTION_DEVICE_EXPANDED_RESIDUAL,
    YVEX_EXECUTION_DEVICE_FEATURE_TAP,
    YVEX_EXECUTION_DEVICE_LOGITS,
    YVEX_EXECUTION_DEVICE_PROBABILITIES,
    YVEX_EXECUTION_DEVICE_CANDIDATE_STATE,
    YVEX_EXECUTION_DEVICE_ACCEPTED_PREFIX,
    YVEX_EXECUTION_DEVICE_WORKSPACE
} yvex_execution_device_value_kind;

typedef enum {
    YVEX_EXECUTION_MATERIALIZE_NONE = 0,
    YVEX_EXECUTION_MATERIALIZE_SCALAR,
    YVEX_EXECUTION_MATERIALIZE_ROW,
    YVEX_EXECUTION_MATERIALIZE_AUDIT_DIGEST,
    YVEX_EXECUTION_MATERIALIZE_FORENSIC_FULL
} yvex_execution_materialization_policy;

typedef struct {
    unsigned int schema_version;
    yvex_execution_device_value_kind kind;
    yvex_backend *backend;
    const yvex_device_tensor *tensor;
    unsigned long long element_offset;
    unsigned long long model_generation, session_generation, state_generation;
    unsigned long long rows, columns, element_bytes;
    yvex_dtype dtype;
    int mutable_view, synchronization_required;
    yvex_execution_materialization_policy materialization;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char execution_profile_identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_device_view;

int yvex_execution_device_view_validate(
    const yvex_execution_device_view *view, yvex_error *err);

typedef enum {
    YVEX_EXECUTION_SCOPE_TARGET = 0,
    YVEX_EXECUTION_SCOPE_DRAFT
} yvex_execution_target_scope;

typedef enum {
    YVEX_EXECUTION_PHASE_PREFILL = 0,
    YVEX_EXECUTION_PHASE_DECODE,
    YVEX_EXECUTION_PHASE_DRAFT,
    YVEX_EXECUTION_PHASE_VERIFY,
    YVEX_EXECUTION_PHASE_CORRECTION,
    YVEX_EXECUTION_PHASE_RESET
} yvex_execution_phase;

typedef enum {
    YVEX_EXECUTION_CONTEXT_SHORT = 0,
    YVEX_EXECUTION_CONTEXT_MEDIUM,
    YVEX_EXECUTION_CONTEXT_LONG,
    YVEX_EXECUTION_CONTEXT_NEAR_CAPACITY
} yvex_execution_context_band;

typedef enum {
    YVEX_EXECUTION_OPERATION_CORE = 0,
    YVEX_EXECUTION_OPERATION_ENVELOPE,
    YVEX_EXECUTION_OPERATION_RELEASE_SET
} yvex_execution_operation_scope;

typedef struct {
    unsigned int schema_version;
    yvex_execution_target_scope target_scope;
    yvex_execution_phase phase;
    yvex_execution_operation_scope operation_scope;
    unsigned long long token_width;
    int candidate_visible;
    yvex_execution_context_band context_band;
    unsigned long long position, context_capacity;
    unsigned long long local_capacity, compressed_capacity, indexer_capacity;
    unsigned long long rolling_capacity, candidate_capacity;
    unsigned long long workspace_generation;
    yvex_execution_evidence_profile evidence;
    char execution_profile_identity[YVEX_SHA256_HEX_CAP];
    char attention_plan_identity[YVEX_SHA256_HEX_CAP];
    char state_layout_identity[YVEX_SHA256_HEX_CAP];
    char kernel_bundle_identity[YVEX_SHA256_HEX_CAP];
    char workspace_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_shape;

typedef enum {
    YVEX_EXECUTION_CAPACITY_NONE = 0,
    YVEX_EXECUTION_CAPACITY_LOCAL,
    YVEX_EXECUTION_CAPACITY_COMPRESSED,
    YVEX_EXECUTION_CAPACITY_INDEXER,
    YVEX_EXECUTION_CAPACITY_ROLLING,
    YVEX_EXECUTION_CAPACITY_CANDIDATE,
    YVEX_EXECUTION_CAPACITY_CONTEXT,
    YVEX_EXECUTION_CAPACITY_WORKSPACE
} yvex_execution_capacity_component;

typedef struct {
    yvex_execution_capacity_component component;
    unsigned long long configured, required, position, width;
    yvex_execution_target_scope target_scope;
    yvex_execution_phase phase;
    yvex_execution_context_band context_band;
    char shape_identity[YVEX_SHA256_HEX_CAP];
    char workspace_identity[YVEX_SHA256_HEX_CAP];
    char state_layout_identity[YVEX_SHA256_HEX_CAP];
} yvex_execution_shape_failure;

typedef struct yvex_execution_shape_registry yvex_execution_shape_registry;

typedef struct {
    unsigned long long count, capacity, hit_count, miss_count;
} yvex_execution_shape_registry_summary;

int yvex_execution_shape_seal(yvex_execution_shape *shape, yvex_error *err);
int yvex_execution_shape_registry_open(
    yvex_execution_shape_registry **out, unsigned long long capacity,
    yvex_error *err);
int yvex_execution_shape_registry_register(
    yvex_execution_shape_registry *registry,
    const yvex_execution_shape *shape, yvex_error *err);
int yvex_execution_shape_registry_select(
    yvex_execution_shape_registry *registry,
    const yvex_execution_shape *request,
    const yvex_execution_shape **selected,
    yvex_execution_shape_failure *failure, yvex_error *err);
int yvex_execution_shape_registry_summary_copy(
    const yvex_execution_shape_registry *registry,
    yvex_execution_shape_registry_summary *summary, yvex_error *err);
void yvex_execution_shape_registry_close(
    yvex_execution_shape_registry **registry);

#ifdef __cplusplus
}
#endif
#endif
