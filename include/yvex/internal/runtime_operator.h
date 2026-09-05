/* Engineering operator requests and evidence remain outside the model-engine core ABI. */
#ifndef INCLUDE_YVEX_INTERNAL_RUNTIME_OPERATOR_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_RUNTIME_OPERATOR_H_INCLUDED

#include <yvex/internal/benchmark.h>
#include <yvex/internal/evidence.h>
#include <yvex/internal/runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_GRAPH_ATTENTION_TEXT_CAP 128u
#define YVEX_GRAPH_ATTENTION_REASON_CAP 256u
#define YVEX_RUNTIME_CAPTURE_BUCKET_CAP 64u

typedef enum {
    YVEX_RUNTIME_OPERATOR_EXECUTE = 0,
    YVEX_RUNTIME_OPERATOR_PLAN,
    YVEX_RUNTIME_OPERATOR_STATE_INSPECT,
    YVEX_RUNTIME_OPERATOR_STATE_VALIDATE,
    YVEX_RUNTIME_OPERATOR_STATE_EXERCISE,
    YVEX_RUNTIME_OPERATOR_CAPTURE,
    YVEX_RUNTIME_OPERATOR_REPLAY,
    YVEX_RUNTIME_OPERATOR_GRAPH_LIST,
    YVEX_RUNTIME_OPERATOR_GRAPH_INSPECT,
    YVEX_RUNTIME_OPERATOR_GRAPH_WARMUP,
    YVEX_RUNTIME_OPERATOR_GRAPH_UPDATE,
    YVEX_RUNTIME_OPERATOR_GRAPH_INVALIDATE,
    YVEX_RUNTIME_OPERATOR_GRAPH_RELEASE,
    YVEX_RUNTIME_OPERATOR_TRACE,
    YVEX_RUNTIME_OPERATOR_PROFILE,
    YVEX_RUNTIME_OPERATOR_BENCHMARK,
    YVEX_RUNTIME_OPERATOR_QUALIFY,
    YVEX_RUNTIME_OPERATOR_CAPABILITIES,
    YVEX_RUNTIME_OPERATOR_RESIDENCY_INSPECT
} yvex_runtime_operator_action;

typedef struct {
    const char *target, *artifact_path;
    const char *runtime_binding_path, *activation_input_path, *capture_bucket, *attention_class;
    yvex_backend_kind backend;
    yvex_attention_probe_kind probe;
    yvex_attention_probe_scope scope;
    yvex_execution_phase phase;
    yvex_runtime_execution_mode mode;
    yvex_runtime_execution_scope operation_scope;
    yvex_runtime_trace_policy trace_policy;
    yvex_runtime_operator_action operator_action;
    unsigned long long token_count, chunk_tokens, context_capacity, warmup, repeat;
    unsigned long long layer_start, layer_count, history_tokens;
    unsigned long long maximum_host_bytes, maximum_device_bytes, selection_key;
    int compare_backends, require_mode, select_layer, select_selection_key;
    yvex_runtime_progress_callback progress;
    void *progress_context;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_graph_attention_operator_request;

typedef enum {
    YVEX_RUNTIME_QUALITY_SOFTWARE = 0,
    YVEX_RUNTIME_QUALITY_NUMERICAL,
    YVEX_RUNTIME_QUALITY_QUALIFICATION,
    YVEX_RUNTIME_QUALITY_COMPONENT_BENCHMARK,
    YVEX_RUNTIME_QUALITY_CORRECTNESS,
    YVEX_RUNTIME_QUALITY_STRUCTURAL,
    YVEX_RUNTIME_QUALITY_PERFORMANCE,
    YVEX_RUNTIME_QUALITY_STATUS_COUNT
} yvex_runtime_quality_status;

int yvex_graph_attention_operator_selection_validate(
    const yvex_graph_attention_operator_request *request, yvex_error *err);

typedef struct yvex_graph_attention_operator_result {
    int completed;
    char status[32], command[YVEX_GRAPH_ATTENTION_TEXT_CAP];
    char target[YVEX_GRAPH_ATTENTION_TEXT_CAP], family[32], backend[32], scope[32];
    char operation_scope[32], phase[32], trace_policy[16];
    char requested_mode[16], selected_mode[16], selection_reason[96];
    char capture_bucket[YVEX_RUNTIME_CAPTURE_BUCKET_CAP], input_class[64];
    char execution_class[32], weights_class[64];
    char artifact_path[YVEX_PATH_CAP], runtime_binding_path[YVEX_PATH_CAP];
    char source_snapshot_identity[17];
    char cuda_driver[32], cuda_build_identity[YVEX_SHA256_HEX_CAP];
    char payload_identity[YVEX_SHA256_HEX_CAP], artifact_identity[YVEX_SHA256_HEX_CAP];
    char artifact_transform_identity[YVEX_SHA256_HEX_CAP], logical_transform_identity[YVEX_SHA256_HEX_CAP];
    char materialization_identity[YVEX_SHA256_HEX_CAP], logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP], runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP], runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char attention_plan_identity[YVEX_SHA256_HEX_CAP];
    char semantic_graph_identity[YVEX_SHA256_HEX_CAP], executable_graph_identity[YVEX_SHA256_HEX_CAP];
    char execution_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char cuda_launch_graph_identity[YVEX_SHA256_HEX_CAP], cuda_graph_exec_identity[YVEX_SHA256_HEX_CAP];
    char cuda_graph_registry_scope[32], cuda_graph_entry_compatibility_identity[160];
    char residency_identity[YVEX_SHA256_HEX_CAP], workspace_identity[YVEX_SHA256_HEX_CAP];
    char state_layout_identity[YVEX_SHA256_HEX_CAP], state_content_identity[YVEX_SHA256_HEX_CAP],
         state_residency_identity[YVEX_SHA256_HEX_CAP];
    char execution_evidence_digest[YVEX_SHA256_HEX_CAP], execution_identity[YVEX_SHA256_HEX_CAP];
    char qualification_identity[YVEX_SHA256_HEX_CAP], quality_matrix_identity[YVEX_SHA256_HEX_CAP];
    char activation_input_identity[YVEX_SHA256_HEX_CAP], quality_status[YVEX_RUNTIME_QUALITY_STATUS_COUNT][24];
    char benchmark_scope[32], attention_class[16];
    char current_writer_plan_identity[YVEX_SHA256_HEX_CAP];
    char payload_plan_identity[YVEX_SHA256_HEX_CAP], payload_byte_identity[YVEX_SHA256_HEX_CAP];
    char reason[YVEX_GRAPH_ATTENTION_REASON_CAP], first_failing_stage[32];
    unsigned long long main_layers_total, bindings_total;
    unsigned long long repeat_count, warmup_count, benchmark_sample_count;
    unsigned long long requested_token_count, requested_history_tokens, requested_layer_start, requested_layer_count;
    unsigned long long artifact_hash_passes, warm_artifact_hash_passes;
    unsigned long long runtime_source_headers_read, runtime_source_payload_bytes_read;
    unsigned long long runtime_transform_plans_built, runtime_quant_plans_built;
    unsigned long long runtime_writer_plans_built;
    unsigned long long resident_binding_count, resident_encoded_bytes, host_resident_bytes, device_resident_bytes;
    unsigned long long workspace_bytes, pinned_host_bytes, pinned_host_peak_bytes, upload_bytes, upload_count;
    unsigned long long warm_weight_artifact_reads, warm_weight_upload_bytes;
    unsigned long long warm_h2d_bytes, warm_d2h_bytes;
    unsigned long long warm_host_allocations, warm_device_allocations, warm_device_frees;
    unsigned long long cuda_graph_count, cuda_graph_piece_count, cuda_graph_capture_count, cuda_graph_instantiate_count;
    unsigned long long cuda_graph_replay_count, cuda_graph_launch_count, cuda_graph_node_count;
    unsigned long long cuda_graph_kernel_node_count, cuda_graph_memcpy_node_count,
                       cuda_graph_memset_node_count, cuda_graph_invalidation_count;
    unsigned long long cuda_graph_update_count, cuda_graph_update_pending_count;
    unsigned long long cuda_graph_registry_count, cuda_graph_registry_index,
                       cuda_graph_registry_affected_count;
    unsigned long long cuda_graph_capture_elapsed_ns, cuda_graph_instantiate_elapsed_ns,
                       cuda_graph_last_update_elapsed_ns, cuda_graph_last_replay_elapsed_ns;
    unsigned long long execution_dispatch_count, trace_stage_count, trace_value_count;
    unsigned long long state_layer_count, state_prepared_layer_count, state_allocated_bytes;
    unsigned long long state_capacity, state_committed_sequence_length, state_next_position;
    unsigned long long state_generation, state_residency_generation, state_device_bytes;
    unsigned long long state_upload_bytes, state_upload_count;
    unsigned long long state_commit_count, state_abort_count, state_cancellation_count, state_reset_count;
    unsigned long long prefill_chunk_count, committed_prefix;
    int cuda_graph_entry_state, cuda_graph_entry_reason, cuda_graph_entry_capture_mode;
    int cuda_graph_entry_uploaded, cuda_graph_entry_update_requested, pinned_host_residency;
    int physical_payload_compatible, artifact_rebuild_required, materialization_rebuild_required;
    int tensor_inventory_equal, qtype_equal, layout_equal, offset_equal, payload_digest_equal;
    yvex_attention_probe_result probe;
    yvex_runtime_capabilities capabilities;
    int attention_cuda_execution_ready;
    int state_sealed, state_persistent, state_position_consistent, state_cuda_ready;
    int state_transaction_active, state_validation_passed, state_read_after_write_verified,
        state_clear_reuse_verified;
    int production_api_available, internal_live_runner_available, operator_command_available;
    int end_user_generation_available, model_behavior_evaluation_available;
    int model_quality_evaluation_available;
    int agent_runtime_available, agent_evaluation_available, release_qualification_available;
    int benchmark_correctness_precondition_passed, benchmark_runtime_precondition_passed;
    int activation_prefill_ready, prefill_persistent_state_ready, full_model_prefill_ready;
    unsigned long long artifact_bytes_hashed;
    double lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_COUNT], total_seconds, benchmark_first_execution_seconds;
    double benchmark_host_seconds[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
    int benchmark_device_timing_available, artifact_identity_verified;
    double benchmark_device_seconds[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
    yvex_runtime_benchmark_operator_summary benchmark;
    char failure_code[32], failure_where[YVEX_ERROR_WHERE_CAP];
} yvex_graph_attention_operator_result;

int yvex_graph_attention_operator_execute(const yvex_graph_attention_operator_request *request,
    yvex_graph_attention_operator_result *result, yvex_runtime_cleanup_lease **retained_cleanup,
    yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_RUNTIME_OPERATOR_H_INCLUDED */
