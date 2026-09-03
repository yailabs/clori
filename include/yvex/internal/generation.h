/*
 * Compose admitted lower owners into one bounded autoregressive lifecycle.
 * Published tokens are always target-authored: ordinary generation commits one decode step at a
 * time, while speculative generation may commit a target-verified prefix atomically. This is the
 * internal runtime/operator ABI from exact text/messages to model-backed incremental text.
 */
#ifndef INCLUDE_YVEX_INTERNAL_GENERATION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_GENERATION_H_INCLUDED
#include <yvex/internal/core.h>
#include <yvex/internal/deployment.h>
#include <yvex/internal/evidence.h>
#include <yvex/internal/sampling.h>
#include <yvex/tokenizer.h>
#ifdef __cplusplus
extern "C" {
#endif
#define YVEX_RUNTIME_GENERATION_SCHEMA_V3 3u
#define YVEX_RUNTIME_GENERATION_SCHEMA_V5 5u
#define YVEX_RUNTIME_GENERATION_SCHEMA_V6 6u
#define YVEX_RUNTIME_GENERATION_PLAN_SCHEMA_V6 6u
#define YVEX_RUNTIME_GENERATION_RESULT_SCHEMA_V5 5u
#define YVEX_RUNTIME_GENERATION_TURN_SCHEMA_V1 1u
#define YVEX_RUNTIME_PARTIAL_TURN_SCHEMA_V1 1u
typedef enum {
    YVEX_GENERATION_MODE_TARGET_ONLY = 0,
    YVEX_GENERATION_MODE_SPECULATIVE
} yvex_runtime_generation_mode;
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
typedef struct yvex_runtime_generation_options {
    unsigned int schema_version;
    yvex_backend_kind backend;
    yvex_runtime_generation_mode mode;
    yvex_execution_workload_profile_kind workload_kind;
    unsigned long long context_capacity, prefill_chunk_tokens, maximum_new_tokens, concurrent_sequences;
    unsigned long long runnable_sequences;
    unsigned long long maximum_output_bytes, maximum_host_bytes, maximum_device_bytes;
    yvex_runtime_trace_policy trace_policy;
    yvex_execution_evidence_profile evidence_profile;
    yvex_runtime_sampling_policy sampling_policy;
    const unsigned int *additional_stop_token_ids;
    unsigned long long additional_stop_token_count;
    int (*cancel_requested)(void *context);
    void *cancel_context;
    int continuous_batching;
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
    yvex_execution_plan_kind producer_kind;
    yvex_backend_kind backend;
    yvex_runtime_generation_mode mode;
    unsigned long long context_capacity, prefill_chunk_tokens, maximum_new_tokens;
    unsigned long long maximum_output_bytes;
    unsigned int trace_policy;
    yvex_execution_evidence_profile evidence_profile;
    yvex_execution_class execution_class;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char tokenizer_plan_identity[YVEX_SHA256_HEX_CAP];
    char prompt_policy_identity[YVEX_SHA256_HEX_CAP];
    char producer_plan_identity[YVEX_SHA256_HEX_CAP];
    char logits_plan_identity[YVEX_SHA256_HEX_CAP];
    char sampling_policy_identity[YVEX_SHA256_HEX_CAP];
    char speculation_policy_identity[YVEX_SHA256_HEX_CAP];
    char stop_policy_identity[YVEX_SHA256_HEX_CAP];
    char kernel_bundle_identity[YVEX_SHA256_HEX_CAP];
    char execution_profile_identity[YVEX_SHA256_HEX_CAP], workload_profile_identity[YVEX_SHA256_HEX_CAP];
    char hardware_profile[YVEX_EXECUTION_TEXT_CAP];
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
/*
 * A failed turn reports the exact committed boundary independently of its failure class. Facts
 * without a current owner remain explicitly unavailable instead of being inferred from counters.
 */
typedef struct {
    unsigned int schema_version;
    int available, committed_progress, reset_required;
    int draft_state_generation_available, detokenizer_generation_available;
    int failure_status;
    yvex_runtime_generation_stop_reason stop_reason;
    unsigned long long initial_position, final_committed_position;
    unsigned long long committed_token_count, published_text_bytes;
    unsigned long long target_state_generation, draft_state_generation;
    unsigned long long rng_generation, token_ledger_generation;
    unsigned long long detokenizer_generation;
    char target_state_identity[YVEX_SHA256_HEX_CAP];
    char rng_state_identity[YVEX_SHA256_HEX_CAP];
    char token_ledger_identity[YVEX_SHA256_HEX_CAP];
    char published_text_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_partial_turn;
typedef struct {
    unsigned int schema_version;
    yvex_runtime_generation_mode execution_mode;
    yvex_runtime_generation_status status;
    yvex_runtime_generation_stop_reason stop_reason;
    int completed, partial, cancelled, failed, has_incomplete_token;
    unsigned long long prompt_bytes, prompt_token_count, prefill_chunk_count;
    unsigned long long requested_new_tokens, sampled_token_count;
    unsigned long long model_committed_token_count, text_published_token_count;
    unsigned long long terminal_token_count, suppressed_token_count;
    unsigned long long first_incomplete_token;
    /* A decode step is one committed sequence position. Target forward and
     * block-verification counts remain separate because one verification may
     * commit several positions. */
    unsigned long long logits_projection_count, sampling_draw_count, decode_step_count;
    unsigned long long draft_cycle_count, draft_forward_count, proposed_token_count;
    unsigned long long selected_verification_token_count, target_verification_count;
    unsigned long long accepted_draft_token_count, rejected_draft_token_count;
    unsigned long long discarded_draft_token_count;
    unsigned long long target_correction_or_bonus_token_count;
    unsigned long long speculation_source_boundary_token_count;
    unsigned long long maximum_accepted_prefix, confidence_logit_count;
    unsigned long long draft_ns, verification_ns, speculative_commit_ns;
    double mean_accepted_prefix, effective_committed_tokens_per_second;
    double confidence_logit_minimum, confidence_logit_maximum;
    double confidence_logit_mean;
    unsigned long long final_position, final_persistent_generation;
    unsigned long long final_rng_generation, final_token_ledger_generation;
    unsigned long long generated_text_bytes;
    unsigned long long initial_position, reusable_prefix_token_count;
    unsigned long long new_prefill_token_count;
    char prompt_identity[YVEX_SHA256_HEX_CAP];
    char prompt_token_identity[YVEX_SHA256_HEX_CAP];
    char reusable_prefix_identity[YVEX_SHA256_HEX_CAP];
    char initial_rng_identity[YVEX_SHA256_HEX_CAP];
    char final_rng_identity[YVEX_SHA256_HEX_CAP];
    char final_token_ledger_identity[YVEX_SHA256_HEX_CAP];
    char generated_token_identity[YVEX_SHA256_HEX_CAP];
    char generated_text_digest[YVEX_SHA256_HEX_CAP];
    char final_persistent_state_digest[YVEX_SHA256_HEX_CAP];
    char generation_plan_identity[YVEX_SHA256_HEX_CAP];
    char speculation_policy_identity[YVEX_SHA256_HEX_CAP];
    char generation_execution_identity[YVEX_SHA256_HEX_CAP];
    yvex_runtime_partial_turn partial_turn;
} yvex_runtime_generation_result;
typedef int (*yvex_runtime_generation_fragment_sink)(
    void *context, const yvex_runtime_generation_token_result *token,
    const unsigned char *bytes, unsigned long long byte_count,
    yvex_error *err);
typedef enum {
    YVEX_GENERATION_PROGRESS_PROMPT_ACCEPTED = 0,
    YVEX_GENERATION_PROGRESS_PREFILL_STARTED,
    YVEX_GENERATION_PROGRESS_PREFILL_PROGRESS,
    YVEX_GENERATION_PROGRESS_PREFILL_COMPLETED
} yvex_runtime_generation_progress_kind;
typedef int (*yvex_runtime_generation_progress_sink)(
    void *context, yvex_runtime_generation_progress_kind kind,
    unsigned long long value_a, unsigned long long value_b,
    yvex_error *err);
typedef enum {
    YVEX_SPECULATION_PROGRESS_DRAFT_STARTED = 0,
    YVEX_SPECULATION_PROGRESS_DRAFT_COMPLETED,
    YVEX_SPECULATION_PROGRESS_VERIFICATION_STARTED,
    YVEX_SPECULATION_PROGRESS_VERIFICATION_COMPLETED,
    YVEX_SPECULATION_PROGRESS_PREFIX_ACCEPTED,
    YVEX_SPECULATION_PROGRESS_CANDIDATE_REJECTED,
    YVEX_SPECULATION_PROGRESS_CYCLE_COMMITTED
} yvex_runtime_speculation_progress_kind;
typedef struct {
    unsigned int schema_version;
    yvex_runtime_speculation_progress_kind kind;
    unsigned long long cycle, proposed_tokens, selected_verification_tokens;
    unsigned long long accepted_tokens, rejected_tokens, discarded_tokens;
    unsigned long long verification_count;
    unsigned long long confidence_logit_count;
    double confidence_logit_minimum, confidence_logit_maximum;
    double confidence_logit_mean, seconds;
    char policy_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_speculation_progress;
typedef int (*yvex_runtime_speculation_progress_sink)(
    void *context, const yvex_runtime_speculation_progress *progress,
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
    yvex_runtime_speculation_progress_sink speculation_progress_sink;
    void *speculation_progress_context;
    yvex_runtime_generation_evidence *evidence;
} yvex_runtime_generation_turn_request;
typedef struct {
    unsigned int schema_version;
    int open, busy, closing, continuous_batching;
    unsigned long long execution_count, failure_count, cancellation_count;
    unsigned long long token_capacity, text_capacity, workspace_bytes, concurrent_sequences;
    unsigned long long capacity_required_bytes, capacity_unreserved_bytes;
    unsigned long long artifact_reopens, model_rebuilds, output_head_reuploads;
    char generation_plan_identity[YVEX_SHA256_HEX_CAP],
        capacity_plan_identity[YVEX_SHA256_HEX_CAP];
    char token_sequence_identity[YVEX_SHA256_HEX_CAP], rng_state_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_generation_context_summary;
typedef struct yvex_runtime_generation_context yvex_runtime_generation_context;
int yvex_runtime_generation_prefix_identity(
    const unsigned int *tokens, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_stop_identity(
    const yvex_tokenizer_plan_summary *tokenizer,
    const unsigned int *additional, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_plan_identity(
    const yvex_runtime_generation_plan_summary *plan,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_token_identity(
    const yvex_runtime_generation_token_result *token,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_execution_identity(
    const yvex_runtime_generation_result *result,
    const yvex_runtime_generation_token_result *tokens,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_context_summary_copy(const yvex_runtime_generation_context *context,
    yvex_runtime_generation_context_summary *summary, yvex_error *err);
int yvex_runtime_generation_context_open(yvex_runtime_generation_context **out,
    yvex_model_engine *model, yvex_runtime_execution_session *session,
    const yvex_runtime_generation_options *options, yvex_error *err);
const yvex_runtime_generation_plan_summary *yvex_runtime_generation_plan_summary_get(
    const yvex_runtime_generation_context *context);
int yvex_runtime_generation_execute(
    yvex_runtime_generation_context *context, const yvex_runtime_generation_request *request,
    yvex_runtime_generation_token_result *tokens, unsigned long long token_capacity,
    unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_runtime_generation_evidence *evidence,
    yvex_error *err);
/* Begin borrows request and output storage until mandatory finish; advance commits at most the
 * requested scheduling quanta and reports when the transactional turn can be finished. */
int yvex_runtime_generation_turn_begin(
    yvex_runtime_generation_context *context, const yvex_runtime_generation_turn_request *turn,
    yvex_runtime_generation_token_result *tokens, unsigned long long token_capacity,
    unsigned char *text, unsigned long long text_capacity,
    yvex_runtime_generation_result *result, yvex_error *err);
int yvex_runtime_generation_turn_advance(
    yvex_runtime_generation_context *context, unsigned long long work_budget,
    int *complete, yvex_error *err);
int yvex_runtime_generation_turn_finish(
    yvex_runtime_generation_context *context, yvex_error *err);
int yvex_runtime_generation_result_validate(
    const yvex_runtime_generation_plan_summary *plan,
    const yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, const unsigned char *text,
    unsigned long long text_capacity,
    const yvex_runtime_generation_result *result, yvex_error *err);
int yvex_runtime_generation_evidence_validate(
    const yvex_runtime_generation_plan_summary *plan,
    const yvex_runtime_generation_evidence *evidence, yvex_error *err);
int yvex_runtime_generation_context_close(yvex_runtime_generation_context **context, yvex_error *err);
typedef struct {
    const char *target, *artifact_path, *runtime_binding_path;
    yvex_backend_kind backend;
    yvex_runtime_generation_mode mode;
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
    yvex_runtime_generation_evidence evidence;
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
    int release_qualification_ready, speculative_execution_ready;
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
