/*
 * Expose teacher-forced model decode without duplicating transformer or persistent-state
 * resources.
 *
 * Every successful step consumes one nonzero committed prefix and advances session state once.
 * Internal runtime/operator ABI over the existing transformer token-input and context owners.
 */
#ifndef INCLUDE_YVEX_INTERNAL_DECODE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_DECODE_H_INCLUDED
#include <yvex/internal/runtime.h>
#include <yvex/internal/transformer.h>
#include <yvex/model.h>
#ifdef __cplusplus
extern "C" {
#endif
#define YVEX_RUNTIME_DECODE_SCHEMA_V2 2u
#define YVEX_RUNTIME_DECODE_SCHEMA_V1 YVEX_RUNTIME_DECODE_SCHEMA_V2
#define YVEX_SPECULATION_SCHEMA_V1 1u
typedef enum {
    YVEX_SPECULATION_ACCEPT_GREEDY = 0,
    YVEX_SPECULATION_ACCEPT_STOCHASTIC
} yvex_speculation_acceptance_kind;

typedef struct {
    unsigned int schema_version;
    yvex_speculation_acceptance_kind kind;
    unsigned long long candidate_count, vocabulary_size, distribution_stride;
    const unsigned int *candidate_token_ids;
    const unsigned int *target_token_ids;
    const float *draft_probabilities, *target_probabilities;
    const double *acceptance_uniforms;
    double correction_uniform;
} yvex_speculation_acceptance_request;

typedef struct {
    unsigned int schema_version;
    yvex_speculation_acceptance_kind kind;
    unsigned long long proposed_count, accepted_draft_count, rejected_draft_count;
    unsigned long long committed_count, rejection_index;
    int all_candidates_accepted, correction_present, bonus_present;
    unsigned int correction_or_bonus_token_id;
    char policy_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char acceptance_identity[YVEX_SPECULATION_IDENTITY_CAP];
} yvex_speculation_acceptance_result;

typedef struct {
    unsigned long long state_prefix_count, publication_token_count;
    int terminal;
} yvex_speculation_commit_plan;

typedef struct yvex_runtime_logits_context yvex_runtime_logits_context;
typedef struct yvex_runtime_sampling_context yvex_runtime_sampling_context;
struct yvex_runtime_sampling_result;
struct yvex_runtime_sampling_source;
struct yvex_runtime_sampling_policy;
typedef struct yvex_runtime_speculation_context yvex_runtime_speculation_context;

typedef struct {
    yvex_backend_kind backend;
    unsigned long long context_capacity, prefill_chunk_tokens, maximum_host_bytes, maximum_device_bytes;
    unsigned long long compatible_batch_width;
    int compatible_batching;
    const unsigned long long *execution_width;
    int (*cancel_requested)(void *context);
    void *cancel_context;
    const yvex_compiled_execution_profile *execution_profile;
    yvex_execution_shape_registry *shape_registry;
} yvex_runtime_speculation_options;

typedef struct {
    int completed;
    unsigned long long token_start, token_count, position_after;
    yvex_execution_physical_facts physical;
    char projected_feature_digest[YVEX_SPECULATION_IDENTITY_CAP];
    char persistent_state_digest[YVEX_SPECULATION_IDENTITY_CAP];
    char execution_identity[YVEX_SPECULATION_IDENTITY_CAP];
} yvex_runtime_speculation_feature_result;

typedef enum {
    YVEX_RUNTIME_SPECULATION_PHASE_DRAFT_STARTED = 0,
    YVEX_RUNTIME_SPECULATION_PHASE_DRAFT_COMPLETED,
    YVEX_RUNTIME_SPECULATION_PHASE_VERIFICATION_STARTED,
    YVEX_RUNTIME_SPECULATION_PHASE_VERIFICATION_COMPLETED
} yvex_runtime_speculation_phase;

typedef int (*yvex_runtime_speculation_phase_sink)(
    void *context, yvex_runtime_speculation_phase phase,
    unsigned long long elapsed_ns, yvex_error *err);

typedef struct {
    unsigned long long position, candidate_count;
    unsigned int conditioning_token_id;
    yvex_runtime_speculation_phase_sink phase_sink;
    void *phase_context;
} yvex_runtime_speculation_cycle_request;

typedef struct {
    int completed;
    int draft_started, draft_completed;
    int verification_started, verification_completed;
    unsigned long long draft_proposed_count, candidate_count, committed_count;
    unsigned long long target_verification_count;
    unsigned int conditioning_token_id;
    unsigned int candidate_token_ids[YVEX_SPECULATION_MAX_BLOCK];
    unsigned int committed_token_ids[YVEX_SPECULATION_MAX_BLOCK + 2u];
    float confidence_logits[YVEX_SPECULATION_MAX_BLOCK];
    unsigned long long target_rng_draw_count, draft_rng_draw_count;
    unsigned long long draft_ns, verification_ns, acceptance_ns;
    yvex_execution_physical_facts draft_physical, verification_physical;
    yvex_expert_worklist_observation draft_worklists, verification_worklists;
    yvex_speculation_acceptance_result acceptance;
    char draft_execution_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char verification_execution_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char cycle_identity[YVEX_SPECULATION_IDENTITY_CAP];
} yvex_runtime_speculation_cycle_result;

typedef struct {
    int completed;
    int durable;
    unsigned long long position;
    unsigned int token_id;
    unsigned long long target_rng_draw_count;
    char source_selection_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char sampling_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char cycle_identity[YVEX_SPECULATION_IDENTITY_CAP];
} yvex_runtime_speculation_target_step_result;

typedef struct {
    int completed;
    unsigned long long token_start, token_count, position_after, commit_ns;
    unsigned long long verified_prefix_count, promoted_target_token_count;
    unsigned long long target_extension_count, replayed_target_token_count;
    unsigned long long promotion_ns, target_extension_ns;
    yvex_runtime_state_promotion_facts promotion_physical;
    yvex_runtime_transformer_result target_result;
    yvex_expert_worklist_observation extension_worklists;
    char cycle_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char target_execution_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char draft_execution_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char target_state_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char draft_state_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char commit_identity[YVEX_SPECULATION_IDENTITY_CAP];
} yvex_runtime_speculation_commit_result;

int yvex_speculation_accept(
    const yvex_speculation_acceptance_request *request,
    unsigned int *committed_token_ids, unsigned long long committed_capacity,
    yvex_speculation_acceptance_result *result, yvex_error *err);
int yvex_speculation_acceptance_seal(
    unsigned long long vocabulary_size,
    const unsigned int *committed_token_ids,
    unsigned long long committed_capacity,
    yvex_speculation_acceptance_result *result, yvex_error *err);
unsigned int yvex_speculation_distribution_sample(
    const float *target, const float *draft, unsigned long long count,
    double uniform, int residual);
int yvex_speculation_candidate_extent(
    unsigned long long policy_block_size,
    unsigned long long remaining_output_tokens,
    unsigned long long remaining_context_tokens,
    unsigned long long *candidate_count, yvex_error *err);
int yvex_speculation_commit_plan_build(
    const yvex_speculation_acceptance_result *acceptance,
    unsigned long long terminal_index,
    yvex_speculation_commit_plan *plan, yvex_error *err);
int yvex_runtime_speculation_context_open(
    yvex_runtime_speculation_context **out, yvex_model_engine *model,
    yvex_runtime_execution_session *session,
    yvex_runtime_transformer_context *target_transformer,
    yvex_runtime_logits_context *target_logits,
    yvex_runtime_sampling_context *target_sampling,
    const struct yvex_runtime_sampling_policy *sampling_policy,
    const yvex_runtime_speculation_options *options,
    unsigned long long *workspace_bytes, yvex_error *err);
const yvex_speculation_family_policy *yvex_runtime_speculation_policy_get(
    const yvex_runtime_speculation_context *context);
int yvex_runtime_speculation_prefill(
    yvex_runtime_speculation_context *context, const unsigned int *token_ids,
    unsigned long long token_start, unsigned long long token_count,
    float *normalized_hidden, unsigned long long normalized_hidden_capacity,
    yvex_runtime_transformer_result *target_result,
    yvex_runtime_speculation_feature_result *draft_result, yvex_error *err);
int yvex_runtime_speculation_cycle(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result, yvex_error *err);
int yvex_runtime_speculation_target_step_select(
    yvex_runtime_speculation_context *context, unsigned long long position,
    const struct yvex_runtime_sampling_source *source,
    yvex_runtime_speculation_target_step_result *result,
    struct yvex_runtime_sampling_result *selection, yvex_error *err);
int yvex_runtime_speculation_commit_prefix(
    yvex_runtime_speculation_context *context,
    unsigned long long committed_count, float *final_hidden,
    unsigned long long final_hidden_capacity,
    const yvex_runtime_commit_participant *publication,
    yvex_runtime_speculation_commit_result *result, yvex_error *err);
int yvex_runtime_speculation_cycle_abort(
    yvex_runtime_speculation_context *context, yvex_error *err);
int yvex_runtime_speculation_finish_terminal(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_commit_participant *publication, yvex_error *err);
int yvex_runtime_speculation_context_close(
    yvex_runtime_speculation_context **context, yvex_error *err);
typedef enum {
    YVEX_RUNTIME_DECODE_STATUS_NONE = 0,
    YVEX_RUNTIME_DECODE_STATUS_COMPLETE,
    YVEX_RUNTIME_DECODE_STATUS_PARTIAL
} yvex_runtime_decode_status;
typedef struct {
    unsigned int schema_version;
    int completed, normalized_hidden_host_available;
    int normalized_hidden_device_available;
    unsigned long long step_ordinal;
    unsigned int token_id;
    unsigned long long position_before, position_after;
    unsigned long long generation_before, generation_after;
    unsigned long long layers_executed, swa_layers, csa_layers, hca_layers;
    unsigned long long hash_routers, learned_routers, routed_experts, shared_experts;
    unsigned long long row_expert_pairs, unique_experts;
    yvex_expert_worklist_observation expert_worklists;
    unsigned long long grouped_expert_operations, expert_subviews_accessed;
    unsigned long long embedding_weight_bytes, attention_weight_bytes;
    unsigned long long expert_weight_bytes, final_weight_bytes;
    yvex_execution_memory_facts memory;
    unsigned long long h2d_bytes, d2h_bytes, kernel_launches, tensor_core_launches;
    unsigned long long graph_launches, graph_captures, graph_replays;
    unsigned long long d2d_bytes, upload_count, download_count, cache_hits, cache_misses;
    unsigned long long stream_synchronizations, device_synchronizations;
    unsigned long long embedding_ns, attention_ns, attention_device_ns, moe_ns, final_ns;
    unsigned long long synchronization_ns;
    unsigned long long full_array_host_scan_bytes;
    yvex_execution_device_view device_hidden;
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
    unsigned long long row_expert_pairs, unique_experts;
    yvex_expert_worklist_observation expert_worklists;
    unsigned long long grouped_expert_operations, expert_subviews_accessed;
    unsigned long long embedding_weight_bytes, attention_weight_bytes;
    unsigned long long expert_weight_bytes, final_weight_bytes;
    yvex_execution_memory_facts memory;
    unsigned long long h2d_bytes, d2h_bytes, kernel_launches, tensor_core_launches;
    unsigned long long graph_launches, graph_captures, graph_replays;
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
typedef struct {
    const unsigned long long *layer_ordinals;
    unsigned long long layer_count;
    float *values;
    unsigned long long capacity, row_count;
    char digest[YVEX_SHA256_HEX_CAP];
} yvex_runtime_decode_feature_output;
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
