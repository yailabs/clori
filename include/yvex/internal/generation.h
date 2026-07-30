/* Owner: runtime autoregressive generation composition contract.
 * Owns: generation plans, prompt-to-prefill orchestration, sampled-token feedback,
 *   stop state, and evidence.
 * Does not own: tokenizer algorithms, Transformer math, KV storage, logits projection,
 *   sampling policy, or CLI rendering.
 * Invariants: every ordinary published token is the exact sampled ID committed by one decode step.
 * Boundary: internal runtime/operator ABI from exact text/messages to model-backed incremental text.
 * Purpose: compose admitted lower owners into one bounded autoregressive lifecycle.
 * Inputs: one model/session, exact prompt input, sampling policy, budgets, and caller cancellation.
 * Effects: advances only the borrowed session through prompt prefill and successful sampled-token decode commits.
 * Failure: preserves typed partial token/model/text progress and never publishes text for an uncommitted token. */
#ifndef INCLUDE_YVEX_INTERNAL_GENERATION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_GENERATION_H_INCLUDED
#include <yvex/internal/sampling.h>
#include <yvex/internal/profile.h>
#include <yvex/tokenizer.h>
#ifdef __cplusplus
extern "C" {
#endif
#define YVEX_RUNTIME_GENERATION_SCHEMA_V1 1u
#define YVEX_RUNTIME_GENERATION_SCHEMA_V2 2u
#define YVEX_RUNTIME_GENERATION_TURN_SCHEMA_V1 1u
typedef enum {
    YVEX_GENERATION_INPUT_TEXT = 0,
    YVEX_GENERATION_INPUT_MESSAGES = 1,
    YVEX_GENERATION_INPUT_PROVIDER = 2
} yvex_runtime_generation_input_kind;
typedef enum {
    YVEX_GENERATION_STOP_NONE = 0,
    YVEX_GENERATION_STOP_EOS,
    YVEX_GENERATION_STOP_TOKENIZER_TOKEN,
    YVEX_GENERATION_STOP_MAX_NEW_TOKENS,
    YVEX_GENERATION_STOP_CONTEXT_CAPACITY,
    YVEX_GENERATION_STOP_CANCELLED,
    YVEX_GENERATION_STOP_MODEL_FAILURE,
    YVEX_GENERATION_STOP_TOKENIZER_FAILURE,
    YVEX_GENERATION_STOP_OUTPUT_FAILURE
} yvex_runtime_generation_stop_reason;
typedef enum {
    YVEX_GENERATION_STATUS_NONE = 0,
    YVEX_GENERATION_STATUS_COMPLETE,
    YVEX_GENERATION_STATUS_PARTIAL,
    YVEX_GENERATION_STATUS_CANCELLED,
    YVEX_GENERATION_STATUS_FAILED
} yvex_runtime_generation_status;
typedef struct {
    unsigned int schema_version;
    yvex_backend_kind backend;
    unsigned long long context_capacity, prefill_chunk_tokens, maximum_new_tokens;
    unsigned long long maximum_output_bytes, maximum_host_bytes, maximum_device_bytes;
    yvex_runtime_trace_policy trace_policy;
    yvex_runtime_sampling_policy sampling_policy;
    const unsigned int *additional_stop_token_ids;
    unsigned long long additional_stop_token_count;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_runtime_generation_options;
typedef struct {
    unsigned int schema_version;
    yvex_runtime_generation_input_kind kind;
    const unsigned char *text;
    unsigned long long text_bytes;
    const yvex_prompt_message *messages;
    unsigned long long message_count;
    yvex_prompt_options prompt_options;
    yvex_tokenizer_encode_options encode_options;
    const yvex_provider_request *provider_request;
} yvex_runtime_generation_request;
typedef struct {
    unsigned int schema_version;
    yvex_backend_kind backend;
    unsigned long long context_capacity, prefill_chunk_tokens, maximum_new_tokens;
    unsigned long long maximum_output_bytes;
    unsigned int trace_policy;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char tokenizer_plan_identity[YVEX_SHA256_HEX_CAP];
    char prompt_policy_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    char logits_plan_identity[YVEX_SHA256_HEX_CAP];
    char sampling_policy_identity[YVEX_SHA256_HEX_CAP];
    char stop_policy_identity[YVEX_SHA256_HEX_CAP];
    char generation_plan_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_generation_plan_summary;
typedef struct {
    unsigned int schema_version;
    unsigned long long ordinal;
    unsigned int sampled_token_id;
    unsigned int decode_input_token_id;
    yvex_token_append_state sequence_state_before, sequence_state_after;
    unsigned long long position_before, position_after;
    unsigned long long persistent_generation_before, persistent_generation_after;
    unsigned long long text_byte_offset, text_byte_count;
    int sampled, decode_submitted, model_committed, detokenized, text_published;
    int terminal, suppressed;
    yvex_tokenizer_token_classification classification;
    char source_logits_identity[YVEX_SHA256_HEX_CAP];
    char sampling_result_identity[YVEX_SHA256_HEX_CAP];
    char decode_execution_identity[YVEX_SHA256_HEX_CAP];
    char persistent_state_digest[YVEX_SHA256_HEX_CAP];
    char decoder_fragment_identity[YVEX_SHA256_HEX_CAP];
    char token_step_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_generation_token_result;
typedef struct {
    unsigned int schema_version;
    yvex_runtime_generation_status status;
    yvex_runtime_generation_stop_reason stop_reason;
    int completed, partial, cancelled, failed, has_incomplete_token;
    unsigned long long prompt_bytes, prompt_token_count, prefill_chunk_count;
    unsigned long long requested_new_tokens, sampled_token_count;
    unsigned long long model_committed_token_count, text_published_token_count;
    unsigned long long terminal_token_count, suppressed_token_count;
    unsigned long long first_incomplete_token;
    unsigned long long logits_projection_count, sampling_draw_count, decode_step_count;
    unsigned long long final_position, final_persistent_generation;
    unsigned long long generated_text_bytes;
    unsigned long long initial_position, reusable_prefix_token_count;
    unsigned long long new_prefill_token_count;
    char prompt_identity[YVEX_SHA256_HEX_CAP];
    char prompt_token_identity[YVEX_SHA256_HEX_CAP];
    char reusable_prefix_identity[YVEX_SHA256_HEX_CAP];
    char initial_rng_identity[YVEX_SHA256_HEX_CAP];
    char final_rng_identity[YVEX_SHA256_HEX_CAP];
    char generated_token_identity[YVEX_SHA256_HEX_CAP];
    char generated_text_digest[YVEX_SHA256_HEX_CAP];
    char final_persistent_state_digest[YVEX_SHA256_HEX_CAP];
    char generation_plan_identity[YVEX_SHA256_HEX_CAP];
    char generation_execution_identity[YVEX_SHA256_HEX_CAP];
    yvex_runtime_profile_record profile;
} yvex_runtime_generation_result;
typedef int (*yvex_runtime_generation_fragment_sink)(
    void *context, const yvex_runtime_generation_token_result *token,
    const unsigned char *bytes, unsigned long long byte_count,
    yvex_error *err);
typedef enum {
    YVEX_GENERATION_PROGRESS_PROMPT_ACCEPTED = 0,
    YVEX_GENERATION_PROGRESS_PREFILL_STARTED,
    YVEX_GENERATION_PROGRESS_PREFILL_COMPLETED
} yvex_runtime_generation_progress_kind;
typedef int (*yvex_runtime_generation_progress_sink)(
    void *context, yvex_runtime_generation_progress_kind kind,
    unsigned long long value_a, unsigned long long value_b,
    yvex_error *err);
typedef struct {
    unsigned int schema_version;
    const yvex_runtime_generation_request *prompt;
    const unsigned int *committed_prefix_token_ids;
    unsigned long long committed_prefix_token_count;
    unsigned long long maximum_new_tokens;
    unsigned int *prompt_token_ids;
    unsigned long long prompt_token_capacity;
    yvex_runtime_generation_fragment_sink fragment_sink;
    void *fragment_context;
    yvex_runtime_generation_progress_sink progress_sink;
    void *progress_context;
} yvex_runtime_generation_turn_request;
typedef struct {
    unsigned int schema_version;
    int open, busy, closing;
    unsigned long long execution_count, failure_count, cancellation_count;
    unsigned long long token_capacity, text_capacity, workspace_bytes;
    unsigned long long artifact_reopens, model_rebuilds, output_head_reuploads;
    char generation_plan_identity[YVEX_SHA256_HEX_CAP];
    char token_sequence_identity[YVEX_SHA256_HEX_CAP];
    char rng_state_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_generation_context_summary;
typedef struct yvex_runtime_generation_context yvex_runtime_generation_context;
int yvex_runtime_generation_context_open(
    yvex_runtime_generation_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session,
    const yvex_runtime_generation_options *options, yvex_error *err);
const yvex_runtime_generation_plan_summary *yvex_runtime_generation_plan_summary_get(
    const yvex_runtime_generation_context *context);
int yvex_runtime_generation_execute(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_request *request,
    yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_error *err);
int yvex_runtime_generation_turn_execute(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_error *err);
int yvex_runtime_generation_result_validate(
    const yvex_runtime_generation_plan_summary *plan,
    const yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, const unsigned char *text,
    unsigned long long text_capacity,
    const yvex_runtime_generation_result *result, yvex_error *err);
int yvex_runtime_generation_context_close(
    yvex_runtime_generation_context **context, yvex_error *err);
typedef struct {
    const char *target, *artifact_path, *runtime_binding_path;
    yvex_backend_kind backend;
    yvex_runtime_generation_input_kind input_kind;
    const unsigned char *text;
    unsigned long long text_bytes;
    const yvex_prompt_message *messages;
    unsigned long long message_count;
    yvex_prompt_options prompt_options;
    yvex_tokenizer_encode_options encode_options;
    unsigned long long context_capacity, prefill_chunk_tokens, maximum_new_tokens;
    unsigned long long maximum_output_bytes, maximum_host_bytes, maximum_device_bytes;
    yvex_runtime_sampling_policy sampling_policy;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_generation_operator_request;
typedef struct {
    int completed;
    char status[32], command[64], target[128], family[32], backend[16];
    char sampling_execution_kind[32], tokenizer_execution_kind[32], reason[256];
    yvex_runtime_generation_plan_summary plan;
    yvex_runtime_generation_result execution;
    yvex_runtime_generation_context_summary context;
    yvex_runtime_generation_token_result *tokens;
    unsigned long long token_count;
    unsigned char *text;
    unsigned long long text_bytes;
    int generation_plan_ready, generation_prompt_ready, generation_prefill_ready;
    int generation_first_token_ready, sampled_token_feedback_ready;
    int generation_decode_loop_ready, generation_logits_loop_ready;
    int generation_sampling_loop_ready, generation_token_append_ready;
    int generation_eos_stop_ready, generation_context_stop_ready;
    int generation_incremental_text_ready, generation_partial_progress_ready;
    int generation_cpu_ready, generation_cuda_model_path_ready;
    int generation_loop_ready, generation_ready;
    int cli_generate_ready, repl_ready, interactive_chat_ready, server_generation_ready;
    int model_behavior_evaluation_ready, full_model_benchmark_ready;
    int release_qualification_ready, mtp_ready, speculative_execution_ready;
} yvex_generation_operator_result;
int yvex_runtime_generation_operator_execute(
    const yvex_generation_operator_request *request,
    yvex_generation_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err);
void yvex_runtime_generation_operator_result_release(
    yvex_generation_operator_result *result);
const char *yvex_runtime_generation_stop_reason_name(
    yvex_runtime_generation_stop_reason reason);
#ifdef __cplusplus
}
#endif
#endif
