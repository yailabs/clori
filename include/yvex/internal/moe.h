/* Owner: graph/runtime MoE execution contract.
 * Owns: immutable MoE plans, typed activation input, selected-expert execution, and publication.
 * Does not own: family discovery, artifact admission, transformer composition, KV, tokenizer, or generation.
 * Invariants: family policy is projected by adapter identity and only selected expert subviews are consumed.
 * Boundary: internal graph/runtime/backend ABI for one token-local MoE block.
 * Purpose: expose the complete production MoE-local boundary without CLI-shaped numerical APIs.
 * Inputs: admitted runtime identities, encoded weight views, expanded hidden activations, and token IDs.
 * Effects: owns bounded input/context resources and publishes complete typed results transactionally.
 * Failure: stale identity, malformed geometry, unavailable qtype/backend, or cancellation publishes no output. */
#ifndef INCLUDE_YVEX_INTERNAL_MOE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MOE_H_INCLUDED

#include <stddef.h>
#include <yvex/internal/graph.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_MOE_PLAN_SCHEMA_V1 1u
#define YVEX_MOE_INPUT_SCHEMA_V1 1u
#define YVEX_MOE_INPUT_SUFFIX ".yvex-moe-input"
#define YVEX_MOE_NO_TENSOR ULLONG_MAX
#define YVEX_MOE_MAX_SELECTED 16u
#define YVEX_MOE_CUDA_WORKSPACE_BYTES (16ull * 1024ull * 1024ull)

typedef struct yvex_runtime_binding_summary yvex_runtime_binding_summary;
typedef struct yvex_runtime_model yvex_runtime_model;
typedef struct yvex_runtime_execution_session yvex_runtime_execution_session;
typedef struct yvex_runtime_cleanup_lease yvex_runtime_cleanup_lease;

typedef enum {
    YVEX_MOE_ROUTER_HASH_TOKEN_ID = 0,
    YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE
} yvex_moe_router_class;
typedef enum { YVEX_MOE_SCORING_SQRT_SOFTPLUS = 0 } yvex_moe_scoring_policy;
typedef enum { YVEX_MOE_TOPK_NOAUX_TC = 0 } yvex_moe_topk_policy;
typedef enum { YVEX_MOE_ACTIVATION_SILU = 0 } yvex_moe_activation;
typedef enum {
    YVEX_MOE_WEIGHT_FFN_NORM = 0,
    YVEX_MOE_WEIGHT_MHC_FUNCTION,
    YVEX_MOE_WEIGHT_MHC_BASE,
    YVEX_MOE_WEIGHT_MHC_SCALE,
    YVEX_MOE_WEIGHT_ROUTER,
    YVEX_MOE_WEIGHT_ROUTER_TABLE,
    YVEX_MOE_WEIGHT_ROUTER_BIAS,
    YVEX_MOE_WEIGHT_ROUTED_GATE,
    YVEX_MOE_WEIGHT_ROUTED_UP,
    YVEX_MOE_WEIGHT_ROUTED_DOWN,
    YVEX_MOE_WEIGHT_SHARED_GATE,
    YVEX_MOE_WEIGHT_SHARED_UP,
    YVEX_MOE_WEIGHT_SHARED_DOWN,
    YVEX_MOE_WEIGHT_COUNT
} yvex_moe_weight_slot;

typedef struct {
    unsigned long long tensor_id, expert_index;
    yvex_tensor_role role;
    unsigned int qtype;
    const unsigned char *encoded;
    size_t encoded_bytes;
    unsigned long long row_bytes, row_width, row_count;
} yvex_moe_weight_view;

typedef struct {
    unsigned int schema_version;
    unsigned long long ordinal, layer_index;
    yvex_moe_router_class router_class;
    yvex_moe_scoring_policy scoring;
    yvex_moe_topk_policy topk_policy;
    yvex_moe_activation activation;
    unsigned long long hidden_width, residual_streams, expanded_width;
    unsigned long long mhc_mixing_rows, mhc_sinkhorn_iterations;
    unsigned long long routed_experts, shared_experts, experts_per_token;
    unsigned long long expert_intermediate_width, shared_intermediate_width;
    unsigned long long hash_table_rows, hash_table_columns, correction_bias_width;
    double rms_epsilon, mhc_epsilon, mhc_post_multiplier;
    double routed_scaling_factor, activation_limit;
    int requires_token_ids, requires_correction_bias, normalize_topk_probabilities;
    unsigned long long tensor_ids[YVEX_MOE_WEIGHT_COUNT];
    unsigned int qtypes[YVEX_MOE_WEIGHT_COUNT];
    char layer_identity[YVEX_SHA256_HEX_CAP];
} yvex_moe_layer_plan;

typedef struct {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long layer_count, hash_router_layer_count, learned_router_layer_count;
    unsigned long long routed_experts, shared_experts, experts_per_token;
    unsigned long long required_binding_count, expert_subview_count;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char materialization_identity[YVEX_SHA256_HEX_CAP];
    char logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char attention_plan_identity[YVEX_SHA256_HEX_CAP];
    char moe_plan_identity[YVEX_SHA256_HEX_CAP];
} yvex_moe_plan_summary;

typedef struct yvex_moe_plan yvex_moe_plan;
typedef struct {
    unsigned long long adapter_id, adapter_version;
    int (*project_layer)(unsigned long long, const yvex_runtime_descriptor_summary *,
                         const yvex_attention_layer_plan *, yvex_moe_layer_plan *,
                         yvex_error *);
} yvex_moe_family_api;

const yvex_moe_family_api *yvex_graph_moe_family_at(unsigned long long index);
int yvex_moe_plan_build(yvex_moe_plan **out, unsigned long long adapter_id,
                        unsigned long long adapter_version,
                        const yvex_materialization_session *materialization,
                        const yvex_runtime_descriptor *descriptor,
                        const yvex_attention_plan *attention, yvex_error *err);
const yvex_moe_plan_summary *yvex_moe_plan_summary_get(const yvex_moe_plan *plan);
const yvex_moe_layer_plan *yvex_moe_plan_layer_at(const yvex_moe_plan *plan,
                                                   unsigned long long ordinal);
void yvex_moe_plan_close(yvex_moe_plan **plan);

typedef struct {
    unsigned int schema_version;
    unsigned long long token_start, token_count, layer_count;
    unsigned long long activation_payload_bytes, token_id_payload_bytes;
    char logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char moe_plan_identity[YVEX_SHA256_HEX_CAP];
    char activation_payload_digest[YVEX_SHA256_HEX_CAP];
    char token_id_payload_digest[YVEX_SHA256_HEX_CAP];
    char input_identity[YVEX_SHA256_HEX_CAP];
} yvex_moe_input_summary;
typedef struct {
    unsigned long long ordinal, layer_index, width, stride;
    unsigned long long payload_offset, payload_bytes;
    char layer_identity[YVEX_SHA256_HEX_CAP];
} yvex_moe_input_layer_record;
typedef struct yvex_moe_input yvex_moe_input;
typedef struct { unsigned long long maximum_file_bytes; } yvex_moe_input_limits;

int yvex_moe_input_seal(yvex_moe_input_summary *summary,
                        yvex_moe_input_layer_record *records,
                        const float *activations, const unsigned int *token_ids,
                        yvex_error *err);
int yvex_moe_input_write(const char *path, const yvex_moe_input_summary *summary,
                         const yvex_moe_input_layer_record *records,
                         const float *activations, const unsigned int *token_ids,
                         yvex_error *err);
int yvex_moe_input_open_file(yvex_moe_input **out, const char *path,
                             const yvex_moe_input_limits *limits, yvex_error *err);
int yvex_moe_input_open_memory(yvex_moe_input **out, const yvex_moe_input_summary *summary,
                               const yvex_moe_input_layer_record *records,
                               const float *activations, const unsigned int *token_ids,
                               yvex_error *err);
int yvex_moe_input_validate(const yvex_moe_input *input, const yvex_moe_plan *plan,
                            const yvex_runtime_binding_summary *binding, yvex_error *err);
const yvex_moe_input_summary *yvex_moe_input_summary_get(const yvex_moe_input *input);
int yvex_moe_input_layer_view(const yvex_moe_input *input, unsigned long long ordinal,
                              const float **values, unsigned long long *stride,
                              yvex_error *err);
const unsigned int *yvex_moe_input_token_ids(const yvex_moe_input *input);
void yvex_moe_input_close(yvex_moe_input **input);

typedef struct {
    unsigned long long selected_count;
    unsigned long long selected_experts[YVEX_MOE_MAX_SELECTED];
    float router_logits[256], router_scores[256];
    float selected_weights[YVEX_MOE_MAX_SELECTED];
} yvex_moe_router_result;
int yvex_moe_router_result_identity(const yvex_moe_router_result *router,
                                    unsigned long long routed_experts,
                                    char output[YVEX_SHA256_HEX_CAP]);
typedef struct {
    const yvex_moe_layer_plan *layer;
    yvex_moe_weight_view weights[YVEX_MOE_WEIGHT_COUNT];
    const float *expanded_input;
    const yvex_device_tensor *device_input;
    yvex_device_tensor *device_output;
    unsigned int token_id;
    int token_id_present;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_moe_layer_job;
typedef struct {
    float *combined_output, *post, *combination;
    unsigned long long combined_capacity, post_capacity, combination_capacity;
    float *routed_output, *shared_output;
    unsigned long long routed_capacity, shared_capacity;
    yvex_moe_router_result router;
    unsigned long long expert_subviews_accessed, encoded_bytes_read;
    unsigned long long host_to_device_bytes, device_to_host_bytes, kernel_launches;
    unsigned long long cache_hits, cache_misses, upload_count;
    unsigned long long qtype_counts[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    char routed_digest[YVEX_SHA256_HEX_CAP], shared_digest[YVEX_SHA256_HEX_CAP];
    char combined_digest[YVEX_SHA256_HEX_CAP], routing_digest[YVEX_SHA256_HEX_CAP];
} yvex_moe_layer_result;

int yvex_moe_ffn_prepare_cpu(const yvex_moe_layer_job *job, float *normalized,
                             float *post, float *combination, yvex_error *err);
int yvex_moe_route_cpu(const yvex_moe_layer_job *job, const float *normalized,
                       yvex_moe_router_result *result, yvex_error *err);
int yvex_moe_expert_cpu(const yvex_moe_layer_plan *layer,
                        const yvex_moe_weight_view *gate,
                        const yvex_moe_weight_view *up,
                        const yvex_moe_weight_view *down, const float *input,
                        float *output, yvex_error *err);

typedef struct yvex_backend_moe_execution yvex_backend_moe_execution;
int yvex_backend_moe_begin(yvex_backend_moe_execution **out, yvex_backend *backend,
                           const yvex_moe_layer_job *job,
                           yvex_moe_layer_result *result, yvex_error *err);
int yvex_backend_moe_add_expert(yvex_backend_moe_execution *execution,
                                const yvex_moe_weight_view *gate,
                                const yvex_moe_weight_view *up,
                                const yvex_moe_weight_view *down, float route_weight,
                                int shared, yvex_error *err);
int yvex_backend_moe_finish(yvex_backend_moe_execution *execution,
                            yvex_moe_layer_result *result, yvex_error *err);
int yvex_backend_moe_close(yvex_backend_moe_execution **execution, yvex_error *err);

typedef struct yvex_runtime_moe_context yvex_runtime_moe_context;
typedef struct {
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    int (*cancel_requested)(void *context);
    void *cancel_context;
    int defer_cuda_workspace;
} yvex_runtime_moe_options;
typedef struct {
    float *combined_outputs, *post, *combination;
    unsigned long long combined_capacity, post_capacity, combination_capacity;
} yvex_runtime_moe_output;
typedef struct {
    int completed;
    unsigned long long token_count, layers_executed, hash_router_executions;
    unsigned long long learned_router_executions, routed_expert_executions;
    unsigned long long shared_expert_executions, expert_subviews_accessed;
    unsigned long long encoded_bytes_read, host_to_device_bytes, device_to_host_bytes;
    unsigned long long kernel_launches, upload_count, cache_hits, cache_misses;
    unsigned long long qtype_counts[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    char input_identity[YVEX_SHA256_HEX_CAP], routing_digest[YVEX_SHA256_HEX_CAP];
    char routed_digest[YVEX_SHA256_HEX_CAP], shared_digest[YVEX_SHA256_HEX_CAP];
    char combined_output_digest[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_moe_result;

int yvex_runtime_moe_context_open(yvex_runtime_moe_context **out, yvex_runtime_model *model,
                                  yvex_runtime_execution_session *session,
                                  const yvex_runtime_moe_options *options, yvex_error *err);
const yvex_moe_plan *yvex_runtime_moe_context_plan(const yvex_runtime_moe_context *context);
int yvex_runtime_moe_execute(yvex_runtime_moe_context *context,
                             const yvex_moe_input *input,
                             yvex_runtime_moe_output *output,
                             yvex_runtime_moe_result *result, yvex_error *err);
int yvex_runtime_moe_execute_layer(yvex_runtime_moe_context *context,
                                   unsigned long long layer_index,
                                   const float *expanded_input, unsigned int token_id,
                                   int token_id_present, yvex_moe_layer_result *result,
                                   yvex_error *err);
int yvex_runtime_moe_execute_layer_borrowed(yvex_runtime_moe_context *context,
                                            unsigned long long layer_index,
                                            const float *expanded_input, unsigned int token_id,
                                   int token_id_present,
                                   yvex_moe_layer_result *result, yvex_error *err);
int yvex_runtime_moe_execute_layer_device_borrowed(
    yvex_runtime_moe_context *context, unsigned long long layer_index,
    const float *expanded_input, const yvex_device_tensor *device_input,
    yvex_device_tensor *device_output, unsigned int token_id, int token_id_present,
    yvex_moe_layer_result *result, yvex_error *err);
int yvex_runtime_moe_context_reset(yvex_runtime_moe_context *context, yvex_error *err);
int yvex_runtime_moe_context_close(yvex_runtime_moe_context **context, yvex_error *err);

typedef struct {
    const char *target, *artifact_path, *runtime_binding_path, *input_path;
    yvex_backend_kind backend;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_moe_operator_request;
typedef struct {
    int completed;
    char status[32], command[64], target[128], family[32], backend[16], reason[256];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char moe_plan_identity[YVEX_SHA256_HEX_CAP];
    yvex_runtime_moe_result execution;
    unsigned long long layer_count, token_count, hash_router_count, learned_router_count;
    unsigned long long routed_experts, shared_experts, experts_per_token;
    int moe_plan_ready, moe_router_ready, moe_routed_expert_ready;
    int moe_shared_expert_ready, moe_block_ready;
    int moe_prefill_composed, moe_decode_composed, transformer_ready, generation_ready;
} yvex_moe_operator_result;
int yvex_runtime_moe_operator_execute(const yvex_moe_operator_request *request,
                                      yvex_moe_operator_result *result,
                                      yvex_runtime_cleanup_lease **retained_cleanup,
                                      yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_MOE_H_INCLUDED */
