/* Local clients and the persistent host exchange bounded versioned frames without sharing
 * engine pointers. Engine generations own executable resources and sessions; the host owns
 * routing, external admission, transport, and configured engine capacity. */
#ifndef YVEX_SERVER_H
#define YVEX_SERVER_H
#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/provider.h>
#ifdef __cplusplus
extern "C" {
#endif
#define YVEX_LOCAL_PROTOCOL_VERSION 13u
#define YVEX_CLIENT_MEDIA_RESULT_SCHEMA_V1 1u
#define YVEX_SERVER_OPTIONS_SCHEMA_V3 3u
#define YVEX_SERVER_ENGINE_SCHEMA_V1 1u
#define YVEX_CLIENT_PARTIAL_TURN_SCHEMA_V1 1u
#define YVEX_CLIENT_STATE_CHECKPOINT_SCHEMA_V1 1u
#define YVEX_RUNTIME_EVENT_SCHEMA_VERSION 3u
#define YVEX_RUNTIME_METRICS_SCHEMA_VERSION 3u
#define YVEX_SERVER_SESSION_NAME_CAP 64u
#define YVEX_SERVER_ID_CAP 65u
#define YVEX_SERVER_REASON_CAP 256u
#define YVEX_SERVER_FRAGMENT_CAP 4096u
#define YVEX_SERVER_SOCKET_PATH_CAP 512u
#define YVEX_SERVER_STATE_PATH_CAP 512u
#define YVEX_SERVER_FRAME_MAX_BYTES 1048576u
#define YVEX_SERVER_MODEL_ALIAS_CAP 128u
#define YVEX_SERVER_DEFAULT_MAXIMUM_ENGINES 8u
#define YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES 64u
typedef struct yvex_server yvex_server;
typedef struct yvex_client yvex_client;
typedef int (*yvex_server_model_loader)(
    void *context, yvex_server *server, const char *alias, yvex_error *err);
typedef enum {
    YVEX_SERVER_STATUS_CONFIGURED = 0,
    YVEX_SERVER_STATUS_STARTING,
    YVEX_SERVER_STATUS_READY,
    YVEX_SERVER_STATUS_STOPPING,
    YVEX_SERVER_STATUS_STOPPED,
    YVEX_SERVER_STATUS_FAILED
} yvex_server_status;
typedef enum {
    YVEX_SERVER_ENGINE_UNLOADED = 0,
    YVEX_SERVER_ENGINE_LOADING,
    YVEX_SERVER_ENGINE_LOADED,
    YVEX_SERVER_ENGINE_DRAINING,
    YVEX_SERVER_ENGINE_UNLOADING,
    YVEX_SERVER_ENGINE_FAILED
} yvex_server_engine_state;
typedef enum {
    YVEX_SERVER_TRACE_SUMMARY = 0,
    YVEX_SERVER_TRACE_STAGES,
    YVEX_SERVER_TRACE_TOKENS,
    YVEX_SERVER_TRACE_FULL
} yvex_server_trace_level;
typedef enum {
    YVEX_SERVER_CONSOLE_OFF = 0,
    YVEX_SERVER_CONSOLE_RAW,
    YVEX_SERVER_CONSOLE_HUMAN
} yvex_server_console_kind;
typedef enum {
    YVEX_SERVER_GENERATION_TARGET_ONLY = 0,
    YVEX_SERVER_GENERATION_DSPARK,
    YVEX_SERVER_GENERATION_MEDIA
} yvex_server_generation_mode;
typedef enum {
    YVEX_SERVER_SESSION_CREATED = 0,
    YVEX_SERVER_SESSION_READY,
    YVEX_SERVER_SESSION_RUNNING,
    YVEX_SERVER_SESSION_PARTIAL,
    YVEX_SERVER_SESSION_DETACHED,
    YVEX_SERVER_SESSION_RESETTING,
    YVEX_SERVER_SESSION_CLOSING,
    YVEX_SERVER_SESSION_CLOSED,
    YVEX_SERVER_SESSION_FAILED
} yvex_server_session_state;
typedef enum {
    YVEX_SERVER_EVENT_PROCESS_START = 0,
    YVEX_SERVER_EVENT_TELEMETRY_READY,
    YVEX_SERVER_EVENT_ARTIFACT_OPEN_START,
    YVEX_SERVER_EVENT_ARTIFACT_OPEN_COMPLETE,
    YVEX_SERVER_EVENT_BINDING_ADMITTED,
    YVEX_SERVER_EVENT_MATERIALIZATION_START,
    YVEX_SERVER_EVENT_MATERIALIZATION_COMPLETE,
    YVEX_SERVER_EVENT_RESIDENCY_READY,
    YVEX_SERVER_EVENT_RUNTIME_READY,
    YVEX_SERVER_EVENT_LISTENER_READY,
    YVEX_SERVER_EVENT_SESSION_CREATED,
    YVEX_SERVER_EVENT_SESSION_ATTACHED,
    YVEX_SERVER_EVENT_SESSION_DETACHED,
    YVEX_SERVER_EVENT_SESSION_RESET,
    YVEX_SERVER_EVENT_SESSION_CLOSED,
    YVEX_SERVER_EVENT_REQUEST_RECEIVED,
    YVEX_SERVER_EVENT_REQUEST_QUEUED,
    YVEX_SERVER_EVENT_REQUEST_STARTED,
    YVEX_SERVER_EVENT_TOKENIZER_COMPLETED,
    YVEX_SERVER_EVENT_PREFILL_STARTED,
    YVEX_SERVER_EVENT_PREFILL_PROGRESS,
    YVEX_SERVER_EVENT_PREFILL_COMPLETED,
    YVEX_SERVER_EVENT_DRAFT_STARTED,
    YVEX_SERVER_EVENT_DRAFT_COMPLETED,
    YVEX_SERVER_EVENT_VERIFICATION_STARTED,
    YVEX_SERVER_EVENT_VERIFICATION_COMPLETED,
    YVEX_SERVER_EVENT_PREFIX_ACCEPTED,
    YVEX_SERVER_EVENT_CANDIDATE_REJECTED,
    YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED,
    YVEX_SERVER_EVENT_GENERATION_FIRST_TOKEN,
    YVEX_SERVER_EVENT_GENERATION_FRAGMENT,
    YVEX_SERVER_EVENT_GENERATION_PROGRESS,
    YVEX_SERVER_EVENT_GENERATION_PROFILE,
    YVEX_SERVER_EVENT_GENERATION_COMPLETED,
    YVEX_SERVER_EVENT_GENERATION_CANCELLED,
    YVEX_SERVER_EVENT_GENERATION_FAILED,
    YVEX_SERVER_EVENT_CLIENT_DISCONNECTED,
    YVEX_SERVER_EVENT_TELEMETRY_DROPPED,
    YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_START,
    YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE
} yvex_server_event_kind;
typedef enum {
    YVEX_SERVER_SEVERITY_DEBUG = 0,
    YVEX_SERVER_SEVERITY_INFO,
    YVEX_SERVER_SEVERITY_WARNING,
    YVEX_SERVER_SEVERITY_ERROR,
    YVEX_SERVER_SEVERITY_FATAL
} yvex_server_event_severity;
typedef struct {
    unsigned int schema_version;
    unsigned long long sequence, wall_time_ns, monotonic_time_ns, process_id;
    yvex_server_event_kind kind;
    yvex_server_event_severity severity;
    char session_id[YVEX_SERVER_ID_CAP];
    char request_id[YVEX_SERVER_ID_CAP];
    char turn_id[YVEX_SERVER_ID_CAP];
    char phase[32];
    char provider_adapter[YVEX_PROVIDER_ADAPTER_CAP];
    char provider_request_identity[YVEX_PROVIDER_ID_CAP];
    char external_correlation_id[YVEX_PROVIDER_ID_CAP];
    unsigned long long value_a, value_b, value_c;
    yvex_server_generation_mode generation_mode;
    unsigned long long speculative_cycle, proposed_tokens;
    unsigned long long selected_verification_tokens, accepted_tokens;
    unsigned long long rejected_tokens, discarded_tokens, verification_count;
    unsigned long long confidence_logit_count;
    double confidence_logit_minimum, confidence_logit_maximum;
    double confidence_logit_mean, seconds, rate;
    char speculation_policy_identity[YVEX_SHA256_HEX_CAP];
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char variant_identity[YVEX_SHA256_HEX_CAP];
    char event_identity[YVEX_SHA256_HEX_CAP];
} yvex_server_event;
typedef struct {
    unsigned int schema_version;
    unsigned long long uptime_ns, model_open_count, model_close_count;
    unsigned long long artifact_open_count, binding_open_count;
    unsigned long long materialization_count, residency_build_count;
    unsigned long long output_head_upload_count;
    unsigned long long current_rss_bytes, peak_rss_bytes;
    unsigned long long mapped_artifact_bytes, resident_host_bytes;
    unsigned long long resident_device_bytes, queue_depth, queue_capacity;
    unsigned long long active_sessions, total_sessions;
    unsigned long long active_requests, completed_requests;
    unsigned long long failed_requests, cancelled_requests;
    unsigned long long active_http_requests, completed_http_requests;
    unsigned long long failed_http_requests, cancelled_http_requests;
    unsigned long long telemetry_dropped;
} yvex_server_metrics;
typedef struct {
    unsigned int schema_version;
    const char *socket_path;
    unsigned long long request_queue_capacity, worker_count, maximum_engines;
    unsigned long long openai_timeout_ms;
    unsigned short openai_port;
    yvex_server_trace_level trace_level;
    yvex_server_console_kind console;
    int trace_content, openai_enabled;
    yvex_server_model_loader model_loader;
    void *model_loader_context;
} yvex_server_options;
typedef struct {
    unsigned int schema_version;
    const char *alias;
    const char *artifact_path;
    const char *runtime_binding_path;
    const char *target_id;
    yvex_backend_kind backend;
    yvex_server_generation_mode generation_mode;
    unsigned long long context_capacity, prefill_chunk_tokens;
    unsigned long long maximum_new_tokens, maximum_output_bytes;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    unsigned long long maximum_sessions, concurrent_sequences;
    unsigned long long sampling_seed;
    yvex_server_trace_level trace_level;
} yvex_server_engine_options;
typedef struct {
    unsigned int schema_version;
    yvex_server_engine_state state;
    yvex_backend_kind backend;
    yvex_server_generation_mode generation_mode;
    char alias[YVEX_SERVER_MODEL_ALIAS_CAP];
    char target_id[128];
    unsigned long long generation, active_work, session_count;
    unsigned long long context_capacity, prefill_chunk_tokens;
    unsigned long long maximum_new_tokens, maximum_output_bytes;
    unsigned long long maximum_sessions, concurrent_sequences;
    unsigned long long mapped_package_bytes, resident_host_bytes;
    unsigned long long resident_device_bytes, prepared_bytes;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char specialization_identity[YVEX_SHA256_HEX_CAP];
    char capacity_plan_identity[YVEX_SHA256_HEX_CAP];
    int execution_ready, explicit_reasoning_channel_supported;
    int continuous_batching_ready;
} yvex_server_engine_summary;
typedef struct {
    unsigned int schema_version;
    yvex_server_status status;
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    unsigned long long session_count, request_count;
    unsigned long long engine_count, loaded_engine_count;
    unsigned long long draining_engine_count, maximum_engines;
    unsigned long long request_queue_capacity, worker_count;
    unsigned long long openai_timeout_ms;
    unsigned short openai_port;
    yvex_server_trace_level trace_level;
    yvex_server_metrics metrics;
    int host_ready;
    int openai_listener_enabled, openai_listener_ready;
} yvex_server_summary;
typedef enum {
    YVEX_CLIENT_OP_HANDSHAKE = 0,
    YVEX_CLIENT_OP_RUNTIME_STATUS,
    YVEX_CLIENT_OP_RUNTIME_WATCH,
    YVEX_CLIENT_OP_RUNTIME_TRACE,
    YVEX_CLIENT_OP_RUNTIME_STOP,
    YVEX_CLIENT_OP_SESSION_NEW,
    YVEX_CLIENT_OP_SESSION_LIST,
    YVEX_CLIENT_OP_SESSION_SHOW,
    YVEX_CLIENT_OP_SESSION_ATTACH,
    YVEX_CLIENT_OP_SESSION_DETACH,
    YVEX_CLIENT_OP_SESSION_RESET,
    YVEX_CLIENT_OP_SESSION_STATE_SAVE,
    YVEX_CLIENT_OP_SESSION_STATE_RESTORE,
    YVEX_CLIENT_OP_SESSION_FORK,
    YVEX_CLIENT_OP_SESSION_CLOSE,
    YVEX_CLIENT_OP_GENERATION_TURN,
    YVEX_CLIENT_OP_GENERATION_CANCEL,
    YVEX_CLIENT_OP_CONSOLE_STATUS,
    YVEX_CLIENT_OP_ENGINE_LOAD,
    YVEX_CLIENT_OP_ENGINE_LIST,
    YVEX_CLIENT_OP_ENGINE_UNLOAD
} yvex_client_operation;
typedef enum {
    YVEX_CLIENT_MESSAGE_ACK = 0,
    YVEX_CLIENT_MESSAGE_ERROR,
    YVEX_CLIENT_MESSAGE_STATUS,
    YVEX_CLIENT_MESSAGE_SESSION,
    YVEX_CLIENT_MESSAGE_SESSION_LIST,
    YVEX_CLIENT_MESSAGE_EVENT,
    YVEX_CLIENT_MESSAGE_TURN_STARTED,
    YVEX_CLIENT_MESSAGE_FRAGMENT,
    YVEX_CLIENT_MESSAGE_TURN_COMPLETE,
    YVEX_CLIENT_MESSAGE_CONSOLE_STATUS,
    YVEX_CLIENT_MESSAGE_ENGINE
} yvex_client_message_kind;
typedef enum {
    YVEX_CLIENT_FAILURE_NONE = 0,
    YVEX_CLIENT_FAILURE_INVALID_REQUEST,
    YVEX_CLIENT_FAILURE_MODEL_NOT_FOUND,
    YVEX_CLIENT_FAILURE_INCOMPATIBLE_STATE,
    YVEX_CLIENT_FAILURE_REQUEST_TOO_LARGE,
    YVEX_CLIENT_FAILURE_UNSUPPORTED_PARAMETER,
    YVEX_CLIENT_FAILURE_QUEUE_FULL,
    YVEX_CLIENT_FAILURE_CLIENT_CANCELLED,
    YVEX_CLIENT_FAILURE_INTERNAL,
    YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE,
    YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT
} yvex_client_failure_class;
typedef enum {
    YVEX_CLIENT_STOP_NONE = 0,
    YVEX_CLIENT_STOP_EOS,
    YVEX_CLIENT_STOP_TOKENIZER_TOKEN,
    YVEX_CLIENT_STOP_MAXIMUM_TOKENS,
    YVEX_CLIENT_STOP_CONTEXT_CAPACITY,
    YVEX_CLIENT_STOP_CANCELLED,
    YVEX_CLIENT_STOP_MODEL_FAILURE,
    YVEX_CLIENT_STOP_TOKENIZER_FAILURE,
    YVEX_CLIENT_STOP_OUTPUT_FAILURE
} yvex_client_stop_reason;
typedef enum {
    YVEX_CLIENT_PHASE_UNAVAILABLE = 0,
    YVEX_CLIENT_PHASE_IDLE,
    YVEX_CLIENT_PHASE_QUEUED,
    YVEX_CLIENT_PHASE_TOKENIZING,
    YVEX_CLIENT_PHASE_PREFILL,
    YVEX_CLIENT_PHASE_DECODE,
    YVEX_CLIENT_PHASE_COMPLETE,
    YVEX_CLIENT_PHASE_CANCELLED,
    YVEX_CLIENT_PHASE_FAILED
} yvex_client_generation_phase;
typedef enum {
    YVEX_CLIENT_CANCELLATION_NONE = 0,
    YVEX_CLIENT_CANCELLATION_REQUESTED,
    YVEX_CLIENT_CANCELLATION_COMPLETED,
    YVEX_CLIENT_CANCELLATION_DISCONNECT,
    YVEX_CLIENT_CANCELLATION_RUNTIME_SHUTDOWN,
    YVEX_CLIENT_CANCELLATION_FAILED
} yvex_client_cancellation_class;
typedef enum {
    YVEX_CLIENT_STREAM_UNSPECIFIED = 0,
    YVEX_CLIENT_STREAM_FINAL_TEXT,
    YVEX_CLIENT_STREAM_EXPLICIT_REASONING,
    YVEX_CLIENT_STREAM_TOOL_CALL,
    YVEX_CLIENT_STREAM_TOOL_RESULT,
    YVEX_CLIENT_STREAM_CONTROL_EVENT,
    YVEX_CLIENT_STREAM_ERROR
} yvex_client_stream_channel;

/*
 * A terminal failure may follow an atomic model-state commit. This snapshot keeps the exact
 * committed boundary distinct from the failure class and makes reset admission explicit. Facts
 * whose runtime owner cannot yet report a generation remain unavailable rather than synthesized.
 */
typedef struct {
    unsigned int schema_version;
    int available, committed_progress, reset_required;
    int draft_state_generation_available, detokenizer_generation_available;
    int failure_status;
    yvex_client_failure_class failure_class;
    unsigned int stop_reason;
    unsigned long long initial_position, final_committed_position;
    unsigned long long committed_token_count, published_text_bytes;
    unsigned long long target_state_generation, draft_state_generation;
    unsigned long long rng_generation, token_ledger_generation;
    unsigned long long detokenizer_generation;
    unsigned long long message_history_generation, transcript_generation;
    char target_state_identity[YVEX_SHA256_HEX_CAP];
    char rng_state_identity[YVEX_SHA256_HEX_CAP];
    char token_ledger_identity[YVEX_SHA256_HEX_CAP];
    char published_text_identity[YVEX_SHA256_HEX_CAP];
} yvex_client_partial_turn;

typedef struct {
    unsigned int schema_version;
    yvex_backend_kind backend;
    yvex_server_session_state session_state;
    yvex_client_generation_phase generation_phase;
    yvex_client_cancellation_class cancellation_class;
    unsigned long long position, turn_count, context_capacity, context_used;
    unsigned long long kv_used_bytes;
    char session_name[YVEX_SERVER_SESSION_NAME_CAP];
    char model_alias[YVEX_SERVER_MODEL_ALIAS_CAP];
    unsigned long long engine_generation;
    char live_model_identity[YVEX_SHA256_HEX_CAP];
    char physical_variant_identity[YVEX_SHA256_HEX_CAP];
    char selected_model_identity[YVEX_SHA256_HEX_CAP];
    int runtime_ready, session_available, attached, cancel_requested;
    int kv_used_available, progress_available, selected_model_available;
    int explicit_reasoning_channel_supported;
    yvex_reasoning_policy reasoning_policy;
} yvex_console_status;
typedef struct {
    unsigned int schema_version;
    unsigned long long file_bytes, scope_count, committed_sequence_length;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char file_digest[YVEX_SHA256_HEX_CAP];
} yvex_client_state_checkpoint;
typedef struct {
    unsigned int schema_version;
    int available;
    char output_path[YVEX_SERVER_STATE_PATH_CAP];
    unsigned long long width, height, frames;
    unsigned long long fps_numerator, fps_denominator;
    unsigned long long audio_samples, audio_sample_rate, seed, file_bytes;
    char preset_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
    char file_identity[YVEX_SHA256_HEX_CAP];
    char publication_identity[YVEX_SHA256_HEX_CAP];
} yvex_client_media_result;
typedef struct {
    unsigned int schema_version;
    yvex_client_operation operation;
    unsigned long long request_number;
    char model_alias[YVEX_SERVER_MODEL_ALIAS_CAP];
    unsigned long long engine_generation;
    char session_name[YVEX_SERVER_SESSION_NAME_CAP];
    char fork_session_name[YVEX_SERVER_SESSION_NAME_CAP];
    char state_path[YVEX_SERVER_STATE_PATH_CAP];
    const unsigned char *prompt;
    unsigned long long prompt_bytes, maximum_new_tokens;
    unsigned long long maximum_state_file_bytes;
    unsigned long long maximum_prefix_bytes;
    int stochastic, seed_present;
    unsigned long long seed;
    double temperature, top_p, min_p, typical_p;
    unsigned long long top_k, event_after_sequence;
    yvex_server_trace_level trace_level;
    int trace_content;
    yvex_reasoning_policy reasoning_policy;
    const yvex_provider_request *provider_request;
} yvex_client_request;
typedef struct {
    unsigned int schema_version;
    yvex_client_message_kind kind;
    int status;
    yvex_client_failure_class failure_class;
    unsigned long long request_number;
    char session_name[YVEX_SERVER_SESSION_NAME_CAP];
    char reason[YVEX_SERVER_REASON_CAP];
    unsigned char bytes[YVEX_SERVER_FRAGMENT_CAP];
    unsigned long long byte_count;
    unsigned long long prompt_tokens, reused_tokens, prefill_tokens;
    unsigned long long generated_tokens, final_position, turn_count;
    unsigned long long reasoning_tokens, final_tokens;
    unsigned long long context_used, kv_used_bytes;
    yvex_server_generation_mode generation_mode;
    unsigned long long draft_cycle_count, draft_forward_count;
    unsigned long long proposed_tokens, selected_verification_tokens;
    unsigned long long target_verification_count;
    unsigned long long accepted_draft_tokens, rejected_draft_tokens;
    unsigned long long discarded_draft_tokens;
    unsigned long long target_correction_or_bonus_tokens;
    unsigned long long maximum_accepted_prefix;
    unsigned long long confidence_logit_count;
    double queue_seconds, prefill_seconds, first_token_seconds, decode_seconds;
    double prefill_rate, decode_rate, publication_seconds;
    double first_reasoning_seconds, first_final_seconds;
    double reasoning_seconds, final_seconds, total_completion_seconds;
    double reasoning_rate, final_rate, total_completion_rate;
    double draft_seconds, verification_seconds, speculative_commit_seconds;
    double mean_accepted_prefix, effective_committed_rate;
    double confidence_logit_minimum, confidence_logit_maximum;
    double confidence_logit_mean;
    unsigned int stop_reason;
    yvex_client_generation_phase generation_phase;
    yvex_client_cancellation_class cancellation_class;
    yvex_client_stream_channel stream_channel;
    int kv_used_available, publication_timing_available;
    yvex_server_session_state session_state;
    char session_identity[YVEX_SHA256_HEX_CAP];
    char turn_identity[YVEX_SHA256_HEX_CAP];
    char state_digest[YVEX_SHA256_HEX_CAP];
    char generated_token_identity[YVEX_SHA256_HEX_CAP];
    char generated_text_digest[YVEX_SHA256_HEX_CAP];
    char speculation_policy_identity[YVEX_SHA256_HEX_CAP];
    yvex_provider_output_kind provider_output_kind;
    yvex_provider_finish_class provider_finish;
    unsigned long long completion_tokens, total_tokens;
    char provider_request_identity[YVEX_PROVIDER_ID_CAP];
    char external_correlation_id[YVEX_PROVIDER_ID_CAP];
    char tool_call_id[YVEX_PROVIDER_ID_CAP];
    char tool_name[YVEX_PROVIDER_TOOL_NAME_CAP];
    yvex_client_partial_turn partial_turn;
    yvex_client_state_checkpoint state_checkpoint;
    yvex_client_media_result media_result;
    yvex_server_engine_summary engine;
    yvex_server_summary runtime;
    yvex_console_status console;
    yvex_server_event event;
} yvex_client_message;
int yvex_server_create(yvex_server **out, const yvex_server_options *options,
                       yvex_error *err);
int yvex_server_start(yvex_server *server, yvex_error *err);
int yvex_server_engine_load(
    yvex_server *server, const yvex_server_engine_options *options,
    yvex_server_engine_summary *summary, yvex_error *err);
int yvex_server_engine_unload(
    yvex_server *server, const char *alias, unsigned long long generation,
    yvex_server_engine_summary *summary, yvex_error *err);
int yvex_server_engine_snapshot(
    const yvex_server *server, yvex_server_engine_summary *engines,
    unsigned long long capacity, unsigned long long *count, yvex_error *err);
int yvex_server_serve(yvex_server *server, yvex_error *err);
int yvex_server_stop(yvex_server *server, yvex_error *err);
int yvex_server_finish(yvex_server *server, yvex_error *err);
int yvex_server_get_summary(const yvex_server *server,
                            yvex_server_summary *out, yvex_error *err);
int yvex_server_event_next(yvex_server *server, unsigned long long after_sequence,
                           int wait, yvex_server_event *event, yvex_error *err);
int yvex_server_event_json(const yvex_server_event *event, char *output,
                           unsigned long long capacity, yvex_error *err);
int yvex_server_event_validate(const yvex_server_event *event, yvex_error *err);
const char *yvex_server_event_kind_name(yvex_server_event_kind kind);
const char *yvex_server_session_state_name(yvex_server_session_state state);
void yvex_server_close(yvex_server **server);
int yvex_client_connect(yvex_client **out, const char *socket_path,
                        yvex_error *err);
int yvex_client_timeout_set(yvex_client *client,
                            unsigned long long milliseconds,
                            yvex_error *err);
int yvex_client_send(yvex_client *client, const yvex_client_request *request,
                     yvex_error *err);
int yvex_client_receive(yvex_client *client, yvex_client_message *message,
                        yvex_error *err);
void yvex_client_close(yvex_client **client);
int yvex_protocol_request_encode(const yvex_client_request *request,
                                 unsigned char *output,
                                 unsigned long long capacity,
                                 unsigned long long *byte_count,
                                 yvex_error *err);
int yvex_protocol_request_decode(const unsigned char *input,
                                 unsigned long long byte_count,
                                 yvex_client_request *request,
                                 unsigned char **owned_prompt,
                                 yvex_provider_request **owned_provider,
                                 yvex_error *err);
int yvex_protocol_message_encode(const yvex_client_message *message,
                                 unsigned char *output,
                                 unsigned long long capacity,
                                 unsigned long long *byte_count,
                                 yvex_error *err);
int yvex_protocol_message_decode(const unsigned char *input,
                                 unsigned long long byte_count,
                                 yvex_client_message *message,
                                 yvex_error *err);
int yvex_server_socket_path(char output[YVEX_SERVER_SOCKET_PATH_CAP],
                            yvex_error *err);
#ifdef __cplusplus
}
#endif
#endif
