/*
 * Expose the complete transformer-backbone boundary to runtime decode and operator consumers.
 *
 * One plan binds exact lower-owner identities and one chunk owns one persistent-state transaction.
 * Internal graph/runtime/operator ABI from numeric token IDs to normalized hidden states.
 */
#ifndef INCLUDE_YVEX_INTERNAL_TRANSFORMER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_TRANSFORMER_H_INCLUDED
#include <yvex/internal/execution.h>
#include <yvex/internal/moe.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct yvex_backend_cuda_operation_facts yvex_backend_cuda_operation_facts;
#define YVEX_TRANSFORMER_PLAN_SCHEMA_V2 2u
#define YVEX_TRANSFORMER_INPUT_SCHEMA_V1 1u
#define YVEX_TRANSFORMER_INPUT_SUFFIX ".yvex-transformer-input"
#define YVEX_TRANSFORMER_WEIGHT_COUNT 5u
typedef enum {
    YVEX_TRANSFORMER_INITIAL_REPEAT_STREAMS = 0
} yvex_transformer_initial_policy;
typedef enum {
    YVEX_TRANSFORMER_FINAL_SIGMOID_MHC_RMS = 0
} yvex_transformer_final_policy;
typedef enum {
    YVEX_TRANSFORMER_WEIGHT_EMBEDDING = 0,
    YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION,
    YVEX_TRANSFORMER_WEIGHT_FINAL_BASE,
    YVEX_TRANSFORMER_WEIGHT_FINAL_SCALE,
    YVEX_TRANSFORMER_WEIGHT_OUTPUT_NORM
} yvex_transformer_weight_slot;
typedef struct yvex_transformer_family_policy {
    unsigned int schema_version;
    yvex_transformer_initial_policy initial_policy;
    yvex_transformer_final_policy final_policy;
    unsigned long long residual_streams, hidden_width, expanded_width;
    unsigned long long maximum_context, sinkhorn_iterations;
    double mhc_epsilon, output_norm_epsilon;
    int attention_then_moe, deferred_ffn_post, final_norm_after_head;
} yvex_transformer_family_policy;
typedef struct {
    unsigned long long tensor_id, row_width, row_count, encoded_bytes;
    yvex_tensor_role role;
    yvex_tensor_scope tensor_scope;
    unsigned long long layer_index, predictor_index;
    unsigned int qtype;
} yvex_transformer_weight_binding;
typedef struct {
    unsigned long long ordinal, layer_index;
    unsigned long long predictor_index;
    yvex_tensor_scope tensor_scope;
    char moe_layer_identity[YVEX_SHA256_HEX_CAP];
    char layer_identity[YVEX_SHA256_HEX_CAP];
} yvex_transformer_layer_plan;
typedef struct {
    unsigned int schema_version;
    yvex_tensor_scope tensor_scope;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long layer_count, hidden_width, residual_streams, expanded_width;
    unsigned long long maximum_context, vocabulary_size;
    yvex_transformer_initial_policy initial_policy;
    yvex_transformer_final_policy final_policy;
    unsigned long long sinkhorn_iterations;
    double mhc_epsilon, output_norm_epsilon;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char materialization_identity[YVEX_SHA256_HEX_CAP];
    char logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char attention_plan_identity[YVEX_SHA256_HEX_CAP];
    char moe_plan_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    yvex_transformer_weight_binding weights[YVEX_TRANSFORMER_WEIGHT_COUNT];
} yvex_transformer_plan_summary;
typedef struct {
    yvex_transformer_family_policy policy;
    yvex_tensor_scope tensor_scope;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long layer_count, vocabulary_size;
    const char *artifact_identity, *materialization_identity, *logical_model_identity;
    const char *runtime_numeric_identity, *runtime_descriptor_identity;
    yvex_transformer_weight_binding weights[YVEX_TRANSFORMER_WEIGHT_COUNT];
} yvex_transformer_plan_facts;
typedef struct yvex_transformer_plan yvex_transformer_plan;
int yvex_transformer_plan_build(yvex_transformer_plan **out,
                                const yvex_transformer_plan_facts *facts,
                                const yvex_attention_plan *attention,
                                const yvex_moe_plan *moe, yvex_error *err);
int yvex_transformer_plan_import(yvex_transformer_plan **out,
                                 const yvex_transformer_plan_summary *summary,
                                 const yvex_transformer_layer_plan *layers,
                                 yvex_error *err);
int yvex_transformer_plan_seal(yvex_transformer_plan_summary *summary,
                               const yvex_transformer_layer_plan *layers,
                               yvex_error *err);
const yvex_transformer_plan_summary *yvex_transformer_plan_summary_get(
    const yvex_transformer_plan *plan);
void yvex_transformer_plan_close(yvex_transformer_plan **plan);
typedef struct {
    unsigned int schema_version;
    unsigned long long token_start, token_count, vocabulary_size, payload_bytes;
    char logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    char payload_digest[YVEX_SHA256_HEX_CAP];
    char input_identity[YVEX_SHA256_HEX_CAP];
} yvex_transformer_input_summary;
typedef struct yvex_transformer_input yvex_transformer_input;
typedef struct { unsigned long long maximum_file_bytes; } yvex_transformer_input_limits;
int yvex_transformer_input_seal(yvex_transformer_input_summary *summary,
                                const unsigned int *token_ids, yvex_error *err);
int yvex_transformer_input_write(const char *path,
                                 const yvex_transformer_input_summary *summary,
                                 const unsigned int *token_ids, yvex_error *err);
int yvex_transformer_input_open_memory(yvex_transformer_input **out,
                                       const yvex_transformer_input_summary *summary,
                                       const unsigned int *token_ids, yvex_error *err);
int yvex_transformer_input_open_file(yvex_transformer_input **out, const char *path,
                                     const yvex_transformer_input_limits *limits,
                                     yvex_error *err);
int yvex_transformer_input_validate(const yvex_transformer_input *input,
                                    const yvex_transformer_plan *plan,
                                    const yvex_runtime_binding_summary *binding,
                                    yvex_error *err);
const yvex_transformer_input_summary *yvex_transformer_input_summary_get(
    const yvex_transformer_input *input);
const unsigned int *yvex_transformer_input_token_ids(const yvex_transformer_input *input);
void yvex_transformer_input_close(yvex_transformer_input **input);
int yvex_transformer_initial_residual(const yvex_transformer_plan *plan,
                                      const float *embedding, unsigned long long token_count,
                                      float *expanded, yvex_error *err);
int yvex_transformer_deferred_post(const yvex_transformer_plan *plan,
                                   const float *residual, const float *combined,
                                   const float *post, const float *combination,
                                   unsigned long long token_count, float *expanded,
                                   yvex_error *err);
int yvex_transformer_final_stage(const yvex_transformer_plan *plan,
                                 const float *expanded, unsigned long long token_count,
                                 const float *function, const float *base, const float *scale,
                                 const float *norm, float *normalized, yvex_error *err);
int yvex_transformer_final_stage_capture(
    const yvex_transformer_plan *plan, const float *expanded,
    unsigned long long token_count, const float *function, const float *base,
    const float *scale, const float *norm, float *pre_normalized,
    float *normalized, yvex_error *err);
int yvex_transformer_feature_normalize(float *values,
                                       unsigned long long value_count,
                                       const float *weights, double epsilon,
                                       yvex_error *err);
int yvex_backend_transformer_cuda_initial(
    yvex_backend *backend, const yvex_device_tensor *encoded, unsigned int qtype,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, yvex_device_tensor *embedding,
    yvex_device_tensor *expanded, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err);
int yvex_backend_transformer_cuda_final(
    yvex_backend *backend, const yvex_device_tensor *expanded,
    const yvex_device_tensor *function, const yvex_device_tensor *base,
    const yvex_device_tensor *scale, const yvex_device_tensor *norm,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, double epsilon, double mhc_epsilon,
    yvex_device_tensor *output, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err);
typedef struct yvex_runtime_transformer_context yvex_runtime_transformer_context;
typedef enum {
    YVEX_TRANSFORMER_PHASE_PREFILL = 0,
    YVEX_TRANSFORMER_PHASE_DECODE
} yvex_runtime_transformer_phase;
typedef struct {
    unsigned long long maximum_host_bytes, maximum_device_bytes, context_capacity;
    unsigned long long workspace_token_capacity, minimum_device_workspace_bytes;
    yvex_tensor_scope tensor_scope;
    int (*cancel_requested)(void *context);
    void *cancel_context;
    yvex_attention_evidence_level evidence_level;
    int device_hidden_output;
    const yvex_compiled_execution_profile *execution_profile;
    yvex_execution_shape_registry *shape_registry;
} yvex_runtime_transformer_options;
typedef struct {
    unsigned long long chunk_tokens;
    yvex_backend_kind backend;
    yvex_runtime_transformer_phase phase;
    yvex_attention_transaction_disposition transaction_disposition;
    const unsigned long long *feature_layer_ordinals;
    unsigned long long feature_layer_count;
    int candidate_block_visible, retain_prefix_checkpoints;
} yvex_runtime_transformer_request;
typedef struct {
    int completed;
    unsigned long long layer_ordinal, token_count;
    unsigned long long hash_routers, learned_routers, routed_experts, shared_experts;
    unsigned long long row_expert_pairs, unique_experts;
    unsigned long long grouped_expert_operations, expert_subviews_accessed;
    unsigned long long attention_weight_bytes, expert_weight_bytes, final_weight_bytes;
    yvex_execution_memory_facts memory;
    unsigned long long h2d_bytes, d2h_bytes, kernel_launches;
    unsigned long long d2d_bytes, upload_count, download_count, cache_hits, cache_misses;
    unsigned long long stream_synchronizations, device_synchronizations;
    unsigned long long embedding_ns, attention_ns, attention_device_ns, moe_ns, final_ns;
    unsigned long long synchronization_ns;
    char routing_digest[YVEX_SHA256_HEX_CAP];
    char expanded_digest[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_transformer_block_result;
typedef struct {
    float *normalized_hidden;
    unsigned long long capacity;
    float *pre_normalized_hidden;
    unsigned long long pre_normalized_capacity;
    float *features;
    unsigned long long feature_capacity;
} yvex_runtime_transformer_output;
typedef struct {
    int completed;
    int normalized_hidden_host_available, normalized_hidden_device_available;
    yvex_runtime_transformer_phase phase;
    unsigned long long token_start, token_count, chunk_count, committed_prefix;
    unsigned long long position_before, position_after, generation_before, generation_after;
    unsigned long long embedding_rows, embedding_bytes, layers_executed;
    unsigned long long feature_layer_count, feature_row_count;
    unsigned long long swa_layers, csa_layers, hca_layers;
    unsigned long long hash_routers, learned_routers, routed_experts, shared_experts;
    unsigned long long row_expert_pairs, unique_experts;
    unsigned long long grouped_expert_operations, expert_subviews_accessed;
    unsigned long long attention_weight_bytes, expert_weight_bytes, final_weight_bytes;
    yvex_execution_memory_facts memory;
    unsigned long long h2d_bytes, d2h_bytes, kernel_launches;
    unsigned long long d2d_bytes, upload_count, download_count, cache_hits, cache_misses;
    unsigned long long stream_synchronizations, device_synchronizations;
    unsigned long long embedding_ns, attention_ns, attention_device_ns, moe_ns, final_ns;
    unsigned long long synchronization_ns;
    unsigned long long full_array_host_scan_bytes;
    yvex_execution_device_view device_hidden;
    char input_identity[YVEX_SHA256_HEX_CAP];
    char embedding_digest[YVEX_SHA256_HEX_CAP];
    char routing_digest[YVEX_SHA256_HEX_CAP];
    char layer_digest[YVEX_SHA256_HEX_CAP];
    char final_expanded_digest[YVEX_SHA256_HEX_CAP];
    char pre_normalized_hidden_digest[YVEX_SHA256_HEX_CAP];
    char normalized_hidden_digest[YVEX_SHA256_HEX_CAP];
    char feature_digest[YVEX_SHA256_HEX_CAP];
    char persistent_state_digest[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_transformer_result;
typedef struct {
    int completed;
    unsigned long long token_start, token_count;
    unsigned long long position_before, position_after;
    unsigned long long generation_before, generation_after;
    char input_digest[YVEX_SHA256_HEX_CAP];
    char persistent_state_digest[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_transformer_core_commit_result;
int yvex_runtime_transformer_context_open(yvex_runtime_transformer_context **out,
                                          yvex_runtime_model *model,
                                          yvex_runtime_execution_session *session,
                                          const yvex_runtime_transformer_options *options,
                                          yvex_error *err);
const yvex_transformer_plan *yvex_runtime_transformer_context_plan(
    const yvex_runtime_transformer_context *context);
const yvex_runtime_execution_session *yvex_runtime_transformer_context_session(
    const yvex_runtime_transformer_context *context);
int yvex_runtime_transformer_context_validate_input(
    const yvex_runtime_transformer_context *context,
    const yvex_transformer_input *input, yvex_error *err);
int yvex_runtime_transformer_execute_block(
    yvex_runtime_transformer_context *context, unsigned long long layer_ordinal,
    const unsigned int *token_ids, unsigned long long token_count,
    yvex_backend_kind backend, const yvex_attention_publication *attention,
    const yvex_device_tensor *device_attention, yvex_device_tensor *device_output,
    float *expanded_output, yvex_runtime_transformer_block_result *result,
    yvex_error *err);
int yvex_runtime_transformer_execute(yvex_runtime_transformer_context *context,
                                     const yvex_transformer_input *input,
                                     const yvex_runtime_transformer_request *request,
                                     yvex_runtime_transformer_output *output,
                                     yvex_runtime_transformer_result *result,
                                     yvex_error *err);
int yvex_runtime_transformer_stage_core_features(
    yvex_runtime_transformer_context *context, unsigned long long token_start,
    const float *features, unsigned long long token_count,
    yvex_runtime_transformer_core_commit_result *result, yvex_error *err);
int yvex_runtime_transformer_context_close(yvex_runtime_transformer_context **context,
                                           yvex_error *err);
typedef struct {
    const char *target, *artifact_path, *runtime_binding_path, *input_path;
    yvex_backend_kind backend;
    unsigned long long chunk_tokens, context_capacity;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_transformer_operator_request;
typedef struct {
    int completed;
    char status[32], command[64], target[128], family[32], backend[16], phase[16];
    char reason[256];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    yvex_runtime_transformer_result execution;
    unsigned long long hidden_width, expanded_width, layer_count;
    int embedding_ready, transformer_plan_ready, transformer_block_ready;
    int transformer_stack_ready, transformer_final_head_ready, transformer_final_norm_ready;
    int transformer_hidden_state_ready, full_model_prefill_ready, transformer_ready;
    int single_token_transformer_component_ready;
    int model_decode_ready, logits_ready, sampling_ready, tokenizer_runtime_ready;
    int generation_ready, model_behavior_evaluation_ready, release_qualification_ready;
} yvex_transformer_operator_result;
int yvex_transformer_operator_execute(const yvex_transformer_operator_request *request,
                                      yvex_transformer_operator_result *result,
                                      yvex_runtime_cleanup_lease **retained_cleanup,
                                      yvex_error *err);
#ifdef __cplusplus
}
#endif
#endif
