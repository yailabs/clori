/* Owner: runtime repeated decode contract.
 * Owns: explicit decode phase, one-token steps, repeated orchestration, partial progress, and evidence.
 * Does not own: transformer numerics, KV storage, tokenization, logits, sampling, or generation.
 * Invariants: every successful step consumes one nonzero committed prefix and advances session state once.
 * Boundary: internal runtime/operator ABI over the existing transformer token-input and context owners.
 * Purpose: expose teacher-forced model decode without duplicating transformer or persistent-state resources.
 * Inputs: one paired transformer/session context, identity-bound numeric tokens, and caller-owned outputs.
 * Effects: commits successful steps independently and publishes ordered normalized hidden rows.
 * Failure: the failing step publishes nothing while all earlier successful steps remain authoritative. */
#ifndef INCLUDE_YVEX_INTERNAL_DECODE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_DECODE_H_INCLUDED
#include <yvex/internal/transformer.h>
#ifdef __cplusplus
extern "C" {
#endif
#define YVEX_RUNTIME_DECODE_SCHEMA_V1 1u
typedef enum {
    YVEX_RUNTIME_DECODE_STATUS_NONE = 0,
    YVEX_RUNTIME_DECODE_STATUS_COMPLETE,
    YVEX_RUNTIME_DECODE_STATUS_PARTIAL
} yvex_runtime_decode_status;
typedef struct {
    unsigned int schema_version;
    int completed;
    unsigned long long step_ordinal;
    unsigned int token_id;
    unsigned long long position_before, position_after;
    unsigned long long generation_before, generation_after;
    unsigned long long layers_executed, swa_layers, csa_layers, hca_layers;
    unsigned long long hash_routers, learned_routers, routed_experts, shared_experts;
    unsigned long long h2d_bytes, d2h_bytes, kernel_launches;
    unsigned long long d2d_bytes, upload_count, download_count, cache_hits, cache_misses;
    unsigned long long stream_synchronizations, device_synchronizations;
    unsigned long long embedding_ns, attention_ns, attention_device_ns, moe_ns, final_ns;
    unsigned long long synchronization_ns;
    char embedding_digest[YVEX_SHA256_HEX_CAP];
    char routing_digest[YVEX_SHA256_HEX_CAP];
    char layer_digest[YVEX_SHA256_HEX_CAP];
    char normalized_hidden_digest[YVEX_SHA256_HEX_CAP];
    char persistent_state_digest[YVEX_SHA256_HEX_CAP];
    char transformer_execution_identity[YVEX_SHA256_HEX_CAP];
    char decode_step_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_decode_step_result;
typedef struct {
    unsigned int schema_version;
    yvex_runtime_decode_status status;
    int completed, partial, has_incomplete_step;
    unsigned long long requested_steps, completed_steps, first_incomplete_step;
    unsigned long long initial_committed_prefix, final_committed_prefix;
    unsigned long long generation_before, generation_after;
    unsigned long long layers_executed, swa_layers, csa_layers, hca_layers;
    unsigned long long hash_routers, learned_routers, routed_experts, shared_experts;
    unsigned long long h2d_bytes, d2h_bytes, kernel_launches;
    unsigned long long d2d_bytes, upload_count, download_count, cache_hits, cache_misses;
    unsigned long long stream_synchronizations, device_synchronizations;
    unsigned long long embedding_ns, attention_ns, attention_device_ns, moe_ns, final_ns;
    unsigned long long synchronization_ns;
    char input_identity[YVEX_SHA256_HEX_CAP];
    char aggregate_hidden_digest[YVEX_SHA256_HEX_CAP];
    char aggregate_state_digest[YVEX_SHA256_HEX_CAP];
    char decode_execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_decode_result;
typedef struct {
    unsigned long long maximum_steps;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_runtime_decode_options;
typedef struct {
    yvex_backend_kind backend;
} yvex_runtime_decode_request;
typedef struct {
    float *normalized_hidden;
    unsigned long long normalized_hidden_capacity;
    yvex_runtime_decode_step_result *steps;
    unsigned long long step_capacity;
} yvex_runtime_decode_output;
typedef struct yvex_runtime_decode_context yvex_runtime_decode_context;
int yvex_runtime_decode_context_open(
    yvex_runtime_decode_context **out,
    yvex_runtime_transformer_context *transformer,
    yvex_runtime_execution_session *session,
    const yvex_runtime_decode_options *options, yvex_error *err);
int yvex_runtime_decode_step(
    yvex_runtime_decode_context *context, unsigned long long step_ordinal,
    unsigned long long expected_position, unsigned int token_id,
    yvex_backend_kind backend, float *normalized_hidden,
    unsigned long long normalized_hidden_capacity,
    yvex_runtime_decode_step_result *result, yvex_error *err);
int yvex_runtime_decode_execute(
    yvex_runtime_decode_context *context,
    const yvex_transformer_input *input,
    const yvex_runtime_decode_request *request,
    yvex_runtime_decode_output *output,
    yvex_runtime_decode_result *result, yvex_error *err);
int yvex_runtime_decode_context_close(yvex_runtime_decode_context **context,
                                      yvex_error *err);
int yvex_runtime_decode_step_identity(
    const yvex_runtime_decode_step_result *result,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_decode_result_identity(
    const yvex_runtime_decode_result *result,
    const yvex_runtime_decode_step_result *steps,
    char output[YVEX_SHA256_HEX_CAP]);
typedef struct {
    const char *target, *artifact_path, *runtime_binding_path, *input_path;
    yvex_backend_kind backend;
    unsigned long long prefill_tokens, prefill_chunk_tokens, context_capacity;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_decode_operator_request;
typedef struct {
    int completed;
    char status[32], command[64], target[128], family[32], backend[16], phase[16];
    char reason[256];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long hidden_width, layer_count, prefill_tokens_committed;
    yvex_runtime_transformer_result prefill;
    yvex_runtime_decode_result decode;
    yvex_runtime_decode_step_result *steps;
    unsigned long long step_count;
    int decode_step_ready, decode_repeat_ready, decode_hidden_state_ready;
    int decode_partial_progress_ready, moe_decode_composed, model_decode_ready;
    int logits_ready, output_head_ready, sampling_ready, tokenizer_runtime_ready;
    int generation_ready, model_behavior_evaluation_ready;
    int full_model_benchmark_ready, release_qualification_ready;
} yvex_decode_operator_result;
int yvex_runtime_decode_operator_execute(
    const yvex_decode_operator_request *request,
    yvex_decode_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err);
void yvex_runtime_decode_operator_result_release(
    yvex_decode_operator_result *result);
#ifdef __cplusplus
}
#endif
#endif
