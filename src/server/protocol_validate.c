/* Validate complete local-protocol message schemas independently of wire I/O. */
#include "src/server/private.h"

#include <math.h>
#include <string.h>

static int optional_identity_valid(const char value[YVEX_SHA256_HEX_CAP])
{
    return !value[0] || yvex_sha256_hex_valid(value);
}

static int checkpoint_fields_valid(
    const yvex_client_state_checkpoint *checkpoint)
{
    if (!checkpoint->schema_version)
        return !checkpoint->file_bytes && !checkpoint->scope_count &&
               !checkpoint->committed_sequence_length &&
               !checkpoint->runtime_model_identity[0] &&
               !checkpoint->runtime_binding_identity[0] &&
               !checkpoint->artifact_identity[0] &&
               !checkpoint->file_digest[0];
    return checkpoint->schema_version == YVEX_CLIENT_STATE_CHECKPOINT_SCHEMA_V1 &&
           checkpoint->file_bytes && checkpoint->scope_count &&
           yvex_sha256_hex_valid(checkpoint->runtime_model_identity) &&
           yvex_sha256_hex_valid(checkpoint->runtime_binding_identity) &&
           yvex_sha256_hex_valid(checkpoint->artifact_identity) &&
           yvex_sha256_hex_valid(checkpoint->file_digest);
}

static int partial_turn_fields_valid(const yvex_client_partial_turn *partial)
{
#define PARTIAL_BOOL(value) ((value) == 0 || (value) == 1)
    if (!partial || !PARTIAL_BOOL(partial->available) ||
        !PARTIAL_BOOL(partial->committed_progress) ||
        !PARTIAL_BOOL(partial->reset_required) ||
        !PARTIAL_BOOL(partial->draft_state_generation_available) ||
        !PARTIAL_BOOL(partial->detokenizer_generation_available) ||
        partial->failure_class > YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT ||
        partial->stop_reason > YVEX_GENERATION_STOP_OUTPUT_FAILURE)
        return 0;
    if (!partial->available)
        return !partial->schema_version && !partial->committed_progress &&
               !partial->reset_required && !partial->failure_status &&
               partial->failure_class == YVEX_CLIENT_FAILURE_NONE &&
               !partial->stop_reason && !partial->initial_position &&
               !partial->final_committed_position &&
               !partial->committed_token_count &&
               !partial->published_text_bytes &&
               !partial->target_state_generation &&
               !partial->draft_state_generation && !partial->rng_generation &&
               !partial->token_ledger_generation &&
               !partial->detokenizer_generation &&
               !partial->message_history_generation &&
               !partial->transcript_generation &&
               !partial->target_state_identity[0] &&
               !partial->rng_state_identity[0] &&
               !partial->token_ledger_identity[0] &&
               !partial->published_text_identity[0];
    return partial->schema_version == YVEX_CLIENT_PARTIAL_TURN_SCHEMA_V1 &&
           partial->reset_required && partial->failure_status != YVEX_OK &&
           partial->failure_class != YVEX_CLIENT_FAILURE_NONE &&
           partial->final_committed_position >= partial->initial_position &&
           (partial->draft_state_generation_available ||
            !partial->draft_state_generation) &&
           (partial->detokenizer_generation_available ||
            !partial->detokenizer_generation) &&
           optional_identity_valid(partial->target_state_identity) &&
           optional_identity_valid(partial->rng_state_identity) &&
           optional_identity_valid(partial->token_ledger_identity) &&
           optional_identity_valid(partial->published_text_identity);
#undef PARTIAL_BOOL
}

static int media_result_fields_valid(const yvex_client_media_result *result)
{
    if (!result->available)
        return !result->schema_version && !result->output_path[0] &&
               !result->width && !result->height && !result->frames &&
               !result->fps_numerator && !result->fps_denominator &&
               !result->duration_milliseconds && !result->audio_samples &&
               !result->audio_sample_rate && !result->seed &&
               !result->model_evaluations && !result->engine_generation &&
               !result->task && !result->condition_count && !result->file_bytes &&
               !result->preset_identity[0] && !result->trajectory_identity[0] &&
               !result->rng_identity[0] && !result->plan_identity[0] &&
               !result->execution_identity[0] &&
               !result->file_identity[0] && !result->publication_identity[0];
    return result->schema_version == YVEX_CLIENT_MEDIA_RESULT_SCHEMA_V2 &&
           result->output_path[0] == '/' && !strstr(result->output_path, "/../") &&
           strcmp(result->output_path + strlen(result->output_path) - 1u, "/..") &&
           result->width && result->height && result->frames && result->fps_numerator &&
           result->fps_denominator && result->duration_milliseconds &&
           result->audio_samples && result->audio_sample_rate &&
           result->model_evaluations && result->engine_generation &&
           result->task <= YVEX_CLIENT_MEDIA_TASK_FIRST_LAST &&
           result->condition_count <= YVEX_CLIENT_MEDIA_CONDITION_CAP &&
           ((result->task == YVEX_CLIENT_MEDIA_TASK_T2VA &&
             result->condition_count == 0ull) ||
            ((result->task == YVEX_CLIENT_MEDIA_TASK_FIRST ||
              result->task == YVEX_CLIENT_MEDIA_TASK_LAST) &&
             result->condition_count == 1ull) ||
            (result->task == YVEX_CLIENT_MEDIA_TASK_FIRST_LAST &&
             result->condition_count == 2ull)) &&
           result->file_bytes &&
           yvex_sha256_hex_valid(result->preset_identity) &&
           yvex_sha256_hex_valid(result->trajectory_identity) &&
           yvex_sha256_hex_valid(result->rng_identity) &&
           yvex_sha256_hex_valid(result->plan_identity) &&
           yvex_sha256_hex_valid(result->execution_identity) &&
           yvex_sha256_hex_valid(result->file_identity) &&
           yvex_sha256_hex_valid(result->publication_identity);
}

int yvex_server_protocol_message_valid(const yvex_client_message *message)
{
    const yvex_server_summary *runtime = &message->runtime;
    const yvex_console_status *console = &message->console;
    const yvex_server_engine_summary *engine = &message->engine;
#define ENUM_VALID(value, first, last) \
    ((int)(value) >= (int)(first) && (value) <= (last))
#define BOOL_VALID(value) ((value) == 0 || (value) == 1)
    return ENUM_VALID(message->kind, YVEX_CLIENT_MESSAGE_ACK,
                      YVEX_CLIENT_MESSAGE_ENGINE) &&
           ENUM_VALID(message->failure_class, YVEX_CLIENT_FAILURE_NONE,
                      YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT) &&
           ENUM_VALID(message->generation_phase, YVEX_CLIENT_PHASE_UNAVAILABLE,
                      YVEX_CLIENT_PHASE_FAILED) &&
           ENUM_VALID(message->cancellation_class, YVEX_CLIENT_CANCELLATION_NONE,
                      YVEX_CLIENT_CANCELLATION_FAILED) &&
           ENUM_VALID(message->stream_channel, YVEX_CLIENT_STREAM_UNSPECIFIED,
                      YVEX_CLIENT_STREAM_ERROR) &&
           (message->kind != YVEX_CLIENT_MESSAGE_ERROR ||
            message->stream_channel == YVEX_CLIENT_STREAM_ERROR) &&
           (message->kind != YVEX_CLIENT_MESSAGE_FRAGMENT ||
            (message->stream_channel == YVEX_CLIENT_STREAM_FINAL_TEXT &&
             message->provider_output_kind == YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT) ||
            (message->stream_channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING &&
             message->provider_output_kind == YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING) ||
            (message->stream_channel == YVEX_CLIENT_STREAM_TOOL_CALL &&
             message->provider_output_kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL) ||
            (message->stream_channel == YVEX_CLIENT_STREAM_ERROR &&
             message->provider_output_kind == YVEX_PROVIDER_OUTPUT_ERROR)) &&
           ENUM_VALID(message->engine_kind, YVEX_SERVER_ENGINE_NONE,
                      YVEX_SERVER_ENGINE_MEDIA) &&
           ENUM_VALID(message->execution_strategy,
                      YVEX_SERVER_EXECUTION_NOT_APPLICABLE,
                      YVEX_SERVER_EXECUTION_SPECULATIVE) &&
           (message->engine_kind == YVEX_SERVER_ENGINE_TEXT ||
            message->execution_strategy ==
                YVEX_SERVER_EXECUTION_NOT_APPLICABLE) &&
           (message->engine_kind != YVEX_SERVER_ENGINE_TEXT ||
            message->execution_strategy !=
                YVEX_SERVER_EXECUTION_NOT_APPLICABLE) &&
           ENUM_VALID(message->session_state, YVEX_SERVER_SESSION_CREATED,
                      YVEX_SERVER_SESSION_FAILED) &&
           ENUM_VALID(message->provider_output_kind,
                      YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
                      YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING) &&
           ENUM_VALID(message->provider_finish, YVEX_PROVIDER_FINISH_STOP,
                      YVEX_PROVIDER_FINISH_FAILED) &&
           message->stop_reason <= YVEX_GENERATION_STOP_OUTPUT_FAILURE &&
           ENUM_VALID(runtime->status, YVEX_SERVER_STATUS_CONFIGURED,
                      YVEX_SERVER_STATUS_FAILED) &&
           ENUM_VALID(runtime->trace_level, YVEX_SERVER_TRACE_SUMMARY,
                      YVEX_SERVER_TRACE_FULL) &&
           ENUM_VALID(console->backend, YVEX_BACKEND_KIND_CPU,
                      YVEX_BACKEND_KIND_ROCM) &&
           ENUM_VALID(console->session_state, YVEX_SERVER_SESSION_CREATED,
                      YVEX_SERVER_SESSION_FAILED) &&
           ENUM_VALID(console->generation_phase, YVEX_CLIENT_PHASE_UNAVAILABLE,
                      YVEX_CLIENT_PHASE_FAILED) &&
           ENUM_VALID(console->cancellation_class,
                      YVEX_CLIENT_CANCELLATION_NONE,
                      YVEX_CLIENT_CANCELLATION_FAILED) &&
           ENUM_VALID(message->event.kind, YVEX_SERVER_EVENT_PROCESS_START,
                      YVEX_SERVER_EVENT_ENGINE_UNLOAD_FAILED) &&
           ENUM_VALID(message->event.severity, YVEX_SERVER_SEVERITY_DEBUG,
                      YVEX_SERVER_SEVERITY_FATAL) &&
           BOOL_VALID(message->kv_used_available) &&
           BOOL_VALID(message->publication_timing_available) &&
           BOOL_VALID(message->output_limit_explicit) &&
           BOOL_VALID(runtime->host_ready) &&
           BOOL_VALID(runtime->openai_listener_enabled) &&
           BOOL_VALID(runtime->openai_listener_ready) &&
           BOOL_VALID(console->runtime_ready) &&
           BOOL_VALID(console->session_available) &&
           BOOL_VALID(console->attached) &&
           BOOL_VALID(console->cancel_requested) &&
           BOOL_VALID(console->kv_used_available) &&
           BOOL_VALID(console->progress_available) &&
           BOOL_VALID(console->selected_model_available) &&
           BOOL_VALID(console->explicit_reasoning_channel_supported) &&
           yvex_reasoning_policy_valid(console->reasoning_policy) &&
           partial_turn_fields_valid(&message->partial_turn) &&
           checkpoint_fields_valid(&message->state_checkpoint) &&
           media_result_fields_valid(&message->media_result) &&
           (!message->media_result.available ||
            (message->kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE &&
             message->status == YVEX_OK &&
             message->engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
             message->generation_phase == YVEX_CLIENT_PHASE_COMPLETE)) &&
           (message->kind != YVEX_CLIENT_MESSAGE_TURN_COMPLETE ||
            message->status != YVEX_OK ||
            message->engine_kind != YVEX_SERVER_ENGINE_MEDIA ||
            message->media_result.available) &&
           isfinite(message->queue_seconds) &&
           isfinite(message->prefill_seconds) &&
           isfinite(message->first_token_seconds) &&
           isfinite(message->first_reasoning_seconds) &&
           isfinite(message->first_final_seconds) &&
           isfinite(message->decode_seconds) &&
           isfinite(message->prefill_rate) &&
           isfinite(message->decode_rate) &&
           isfinite(message->reasoning_seconds) &&
           isfinite(message->final_seconds) &&
           isfinite(message->total_completion_seconds) &&
           isfinite(message->reasoning_rate) &&
           isfinite(message->final_rate) &&
           isfinite(message->total_completion_rate) &&
           isfinite(message->publication_seconds) &&
           isfinite(message->draft_seconds) &&
           isfinite(message->verification_seconds) &&
           isfinite(message->speculative_commit_seconds) &&
           isfinite(message->mean_accepted_prefix) &&
           isfinite(message->effective_committed_rate) &&
           isfinite(message->confidence_logit_minimum) &&
           isfinite(message->confidence_logit_maximum) &&
           isfinite(message->confidence_logit_mean) &&
           message->draft_forward_count <= message->draft_cycle_count &&
           message->target_verification_count <= message->draft_forward_count &&
           message->selected_verification_tokens <= message->proposed_tokens &&
           message->accepted_draft_tokens + message->rejected_draft_tokens >=
               message->accepted_draft_tokens &&
           message->accepted_draft_tokens + message->rejected_draft_tokens <=
               message->selected_verification_tokens &&
           message->accepted_draft_tokens + message->rejected_draft_tokens +
                   message->discarded_draft_tokens >=
               message->accepted_draft_tokens + message->rejected_draft_tokens &&
           message->accepted_draft_tokens + message->rejected_draft_tokens +
                   message->discarded_draft_tokens ==
               message->proposed_tokens &&
           message->maximum_accepted_prefix <=
               message->accepted_draft_tokens &&
           message->target_correction_or_bonus_tokens <=
               message->target_verification_count &&
           (message->kind != YVEX_CLIENT_MESSAGE_TURN_COMPLETE ||
            message->status != YVEX_OK ||
            message->generation_phase != YVEX_CLIENT_PHASE_COMPLETE ||
            message->execution_strategy !=
                YVEX_SERVER_EXECUTION_SPECULATIVE ||
            (message->draft_cycle_count == message->draft_forward_count &&
             message->draft_forward_count ==
                 message->target_verification_count)) &&
           message->confidence_logit_count <= message->proposed_tokens &&
           (!message->confidence_logit_count ||
            (message->confidence_logit_minimum <=
                 message->confidence_logit_mean &&
             message->confidence_logit_mean <=
                 message->confidence_logit_maximum)) &&
           (message->confidence_logit_count ||
            (!message->confidence_logit_minimum &&
             !message->confidence_logit_maximum &&
             !message->confidence_logit_mean)) &&
           isfinite(message->event.seconds) &&
           isfinite(message->event.rate) &&
           (message->kv_used_available || message->kv_used_bytes == 0u) &&
           (message->publication_timing_available ||
            message->publication_seconds == 0.0) &&
           (message->output_limit_explicit ==
            (message->requested_maximum_new_tokens != 0u)) &&
           (!message->requested_maximum_new_tokens ||
            message->resolved_maximum_new_tokens <=
                message->requested_maximum_new_tokens) &&
           (console->kv_used_available || !console->kv_used_bytes) &&
           (console->selected_model_available ||
            !console->selected_model_identity[0]) &&
           (message->kind != YVEX_CLIENT_MESSAGE_ENGINE ||
            yvex_server_engine_summary_valid(engine));
#undef BOOL_VALID
#undef ENUM_VALID
}
