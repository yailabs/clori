/* Owner: public local runtime-host and client protocol ABI.
 * Owns: local protocol facts, host lifecycle, session operations, typed runtime events,
 *   metrics snapshots, and the thin client connection boundary.
 * Does not own: model math, generation composition, tokenizer policy, terminal rendering,
 *   public-network serving, authentication, or CLI argument parsing.
 * Invariants: clients exchange bounded versioned frames and never receive engine pointers.
 * Boundary: yvexd host and local yvex client communicate through one Unix-domain protocol.
 * Purpose: expose the process/session product topology without linking clients to inference.
 * Inputs: explicit paths, bounded requests, and caller-owned result storage.
 * Effects: host APIs own runtime resources; client APIs own only one local socket.
 * Failure: typed refusal publishes no partial frame, session, or false readiness. */
#ifndef YVEX_SERVER_H
#define YVEX_SERVER_H
#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/provider.h>
#ifdef __cplusplus
extern "C" {
#endif
#define YVEX_LOCAL_PROTOCOL_VERSION 3u
#define YVEX_RUNTIME_EVENT_SCHEMA_VERSION 2u
#define YVEX_RUNTIME_METRICS_SCHEMA_VERSION 3u
#define YVEX_SERVER_SESSION_NAME_CAP 64u
#define YVEX_SERVER_ID_CAP 65u
#define YVEX_SERVER_REASON_CAP 256u
#define YVEX_SERVER_FRAGMENT_CAP 4096u
#define YVEX_SERVER_SOCKET_PATH_CAP 512u
#define YVEX_SERVER_FRAME_MAX_BYTES 1048576u
typedef struct yvex_server yvex_server;
typedef struct yvex_client yvex_client;
typedef enum {
    YVEX_SERVER_STATUS_CONFIGURED = 0,
    YVEX_SERVER_STATUS_STARTING,
    YVEX_SERVER_STATUS_READY,
    YVEX_SERVER_STATUS_STOPPING,
    YVEX_SERVER_STATUS_STOPPED,
    YVEX_SERVER_STATUS_FAILED
} yvex_server_status;
typedef enum {
    YVEX_SERVER_TRACE_SUMMARY = 0,
    YVEX_SERVER_TRACE_STAGES,
    YVEX_SERVER_TRACE_TOKENS,
    YVEX_SERVER_TRACE_FULL
} yvex_server_trace_level;
typedef enum {
    YVEX_SERVER_CONSOLE_OFF = 0,
    YVEX_SERVER_CONSOLE_RAW
} yvex_server_console_kind;
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
    double seconds, rate;
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
    const char *artifact_path;
    const char *runtime_binding_path;
    const char *target_id;
    const char *socket_path;
    yvex_backend_kind backend;
    unsigned long long context_capacity, prefill_chunk_tokens;
    unsigned long long maximum_new_tokens, maximum_output_bytes;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    unsigned long long maximum_sessions, request_queue_capacity;
    unsigned long long sampling_seed;
    unsigned long long openai_timeout_ms;
    unsigned short openai_port;
    yvex_server_trace_level trace_level;
    yvex_server_console_kind console;
    int trace_content, openai_enabled;
} yvex_server_options;
typedef struct {
    unsigned int schema_version;
    yvex_server_status status;
    yvex_backend_kind backend;
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    char target_id[128];
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char physical_variant_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long context_capacity, session_count, request_count;
    unsigned short openai_port;
    yvex_server_metrics metrics;
    int runtime_ready, generation_ready, public_server_ready;
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
    YVEX_CLIENT_OP_SESSION_CLOSE,
    YVEX_CLIENT_OP_GENERATION_TURN,
    YVEX_CLIENT_OP_GENERATION_CANCEL,
    YVEX_CLIENT_OP_MODEL_SHOW,
    YVEX_CLIENT_OP_ARTIFACT_SHOW,
    YVEX_CLIENT_OP_ARTIFACT_VERIFY
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
    YVEX_CLIENT_MESSAGE_TURN_COMPLETE
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
typedef struct {
    unsigned int schema_version;
    yvex_client_operation operation;
    unsigned long long request_number;
    char session_name[YVEX_SERVER_SESSION_NAME_CAP];
    const unsigned char *prompt;
    unsigned long long prompt_bytes, maximum_new_tokens;
    int stochastic, seed_present;
    unsigned long long seed;
    double temperature, top_p, min_p, typical_p;
    unsigned long long top_k, event_after_sequence;
    yvex_server_trace_level trace_level;
    int trace_content;
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
    unsigned long long generated_tokens, final_position;
    double queue_seconds, prefill_seconds, first_token_seconds, decode_seconds;
    double prefill_rate, decode_rate;
    unsigned int stop_reason;
    yvex_server_session_state session_state;
    char session_identity[YVEX_SHA256_HEX_CAP];
    char turn_identity[YVEX_SHA256_HEX_CAP];
    char state_digest[YVEX_SHA256_HEX_CAP];
    char generated_token_identity[YVEX_SHA256_HEX_CAP];
    char generated_text_digest[YVEX_SHA256_HEX_CAP];
    yvex_provider_output_kind provider_output_kind;
    yvex_provider_finish_class provider_finish;
    unsigned long long completion_tokens, total_tokens;
    char provider_request_identity[YVEX_PROVIDER_ID_CAP];
    char external_correlation_id[YVEX_PROVIDER_ID_CAP];
    char tool_call_id[YVEX_PROVIDER_ID_CAP];
    char tool_name[YVEX_PROVIDER_TOOL_NAME_CAP];
    yvex_server_summary runtime;
    yvex_server_event event;
} yvex_client_message;
int yvex_server_create(yvex_server **out, const yvex_server_options *options,
                       yvex_error *err);
int yvex_server_start(yvex_server *server, yvex_error *err);
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
