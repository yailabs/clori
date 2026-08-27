/* Hardware, workload, and capacity facts selected for one deployment and engine workload. */
#ifndef INCLUDE_YVEX_INTERNAL_DEPLOYMENT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_DEPLOYMENT_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/internal/core.h>
#include <yvex/internal/model.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1 1u
#define YVEX_EXECUTION_WORKLOAD_PROFILE_SCHEMA_V1 1u
#define YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1 1u
#define YVEX_EXECUTION_TEXT_CAP 64u
#define YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE (8ull * 1024ull * 1024ull * 1024ull)

typedef enum {
    YVEX_EXECUTION_EVIDENCE_PRODUCTION = 0,
    YVEX_EXECUTION_EVIDENCE_AUDIT,
    YVEX_EXECUTION_EVIDENCE_FORENSIC
} yvex_execution_evidence_profile;

typedef enum {
    YVEX_EXECUTION_RESOLUTION_EXACT = 0,
    YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED,
    YVEX_EXECUTION_RESOLUTION_TEMPORARILY_RESOURCE_LIMITED,
    YVEX_EXECUTION_RESOLUTION_UNSUPPORTED,
    YVEX_EXECUTION_RESOLUTION_TRUST_FAILURE
} yvex_execution_resolution;

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
    const char *model_execution_identity;
    unsigned long long semantic_maximum_context, candidate_width, semantic_state_class_mask;
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
int yvex_execution_capacity_plan_validate(
    const yvex_execution_capacity_plan *plan, yvex_error *err);

typedef enum {
    YVEX_EXECUTION_GENERATION_TARGET_ONLY = 0,
    YVEX_EXECUTION_GENERATION_SPECULATIVE
} yvex_execution_generation_mode;

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_DEPLOYMENT_H_INCLUDED */
