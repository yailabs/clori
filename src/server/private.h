/*
 * Connect admitted server owners without exposing sockets or engine pointers publicly.
 *
 * Only server translation units consume these declarations and all mutable owners are opaque.
 * Source-local interface shared by host, protocol, telemetry, and session owners.
 */
#ifndef SRC_SERVER_PRIVATE_H_INCLUDED
#define SRC_SERVER_PRIVATE_H_INCLUDED

#include <stdatomic.h>

#include <yvex/internal/generation.h>
#include <yvex/internal/runtime_state_store.h>
#include <yvex/server.h>

typedef struct server_telemetry server_telemetry;
typedef struct server_session_registry server_session_registry;
typedef struct server_openai_listener server_openai_listener;
typedef struct server_scheduler server_scheduler;

typedef void (*server_scheduler_execute)(void *context, void *work);
typedef void (*server_scheduler_observe)(void *context,
                                         unsigned long long queued,
                                         unsigned long long capacity,
                                         unsigned long long active);

int yvex_server_scheduler_open(
    server_scheduler **out, unsigned long long queue_capacity,
    unsigned long long worker_count, server_scheduler_execute execute,
    server_scheduler_observe observe, void *context, yvex_error *err);
int yvex_server_scheduler_start(server_scheduler *scheduler, yvex_error *err);
int yvex_server_scheduler_submit(server_scheduler *scheduler, void *work,
                                 const char *serialization_key,
                                 unsigned long long *queued, yvex_error *err);
void yvex_server_scheduler_request_stop(server_scheduler *scheduler);
int yvex_server_scheduler_finish(server_scheduler *scheduler, yvex_error *err);
void yvex_server_scheduler_snapshot(const server_scheduler *scheduler,
                                    unsigned long long *queued,
                                    unsigned long long *active);
void yvex_server_scheduler_close(server_scheduler **scheduler);

#define SESSION_SCHEMA_V1 1u
#define SESSION_MAX_MESSAGES 128u
#define SESSION_TRANSCRIPT_BYTES 1048576u

typedef struct server_session {
    char name[YVEX_SERVER_SESSION_NAME_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
    yvex_server_session_state state;
    yvex_runtime_execution_session *execution;
    yvex_runtime_generation_context *generation;
    yvex_prompt_message messages[SESSION_MAX_MESSAGES];
    unsigned long long message_count;
    unsigned char *transcript;
    unsigned long long transcript_count, transcript_capacity;
    unsigned int *committed_tokens, *prompt_tokens;
    unsigned long long committed_count, token_capacity;
    yvex_runtime_generation_token_result *token_results;
    unsigned char *turn_text;
    unsigned long long text_capacity, turn_count, attached_clients;
    unsigned long long message_history_generation, transcript_generation;
    char last_turn_identity[YVEX_SHA256_HEX_CAP];
    char state_digest[YVEX_SHA256_HEX_CAP];
    char generated_token_identity[YVEX_SHA256_HEX_CAP];
    char generated_text_digest[YVEX_SHA256_HEX_CAP];
    yvex_client_partial_turn partial_turn;
    yvex_client_state_checkpoint state_checkpoint;
    yvex_runtime_sampling_policy policy;
    yvex_reasoning_policy reasoning_policy;
    yvex_runtime_generation_checkpoint pending_generation_checkpoint;
    int policy_set, pending_generation_checkpoint_present;
    atomic_int cancel_requested;
    atomic_int active_turn;
} server_session;

typedef struct {
    const char *yvex_socket;
    unsigned short port;
    unsigned long long timeout_ms;
} server_openai_options;

typedef struct {
    unsigned short port;
    int enabled, ready;
} server_openai_snapshot;

typedef int (*server_message_emit)(void *context,
                                   const yvex_client_message *message,
                                   yvex_error *err);

#define YVEX_SERVER_SESSION_STORE_SCHEMA_V1 1u
typedef struct {
    const yvex_prompt_message *messages;
    const unsigned int *committed_tokens;
    unsigned long long message_count, committed_count, turn_count;
    unsigned long long message_history_generation, transcript_generation;
    yvex_runtime_sampling_policy policy;
    yvex_reasoning_policy reasoning_policy;
    yvex_runtime_generation_checkpoint generation_checkpoint;
    int policy_set, generation_checkpoint_present;
    char last_turn_identity[YVEX_SHA256_HEX_CAP];
    char state_digest[YVEX_SHA256_HEX_CAP];
    char generated_token_identity[YVEX_SHA256_HEX_CAP];
    char generated_text_digest[YVEX_SHA256_HEX_CAP];
} server_session_store_view;

typedef struct {
    yvex_prompt_message *messages;
    unsigned char *transcript;
    unsigned int *committed_tokens;
    unsigned long long message_count, transcript_count, committed_count;
    unsigned long long turn_count, message_history_generation, transcript_generation;
    yvex_runtime_sampling_policy policy;
    yvex_reasoning_policy reasoning_policy;
    yvex_runtime_generation_checkpoint generation_checkpoint;
    int policy_set, generation_checkpoint_present;
    char last_turn_identity[YVEX_SHA256_HEX_CAP];
    char state_digest[YVEX_SHA256_HEX_CAP];
    char generated_token_identity[YVEX_SHA256_HEX_CAP];
    char generated_text_digest[YVEX_SHA256_HEX_CAP];
    char payload_identity[YVEX_SHA256_HEX_CAP];
} server_session_store_state;

int yvex_server_session_store_encode(
    const server_session_store_view *view, unsigned char **bytes,
    unsigned long long *byte_count,
    char payload_identity[YVEX_SHA256_HEX_CAP], yvex_error *err);
int yvex_server_session_store_decode(
    const unsigned char *bytes, unsigned long long byte_count,
    unsigned long long maximum_messages,
    unsigned long long maximum_transcript_bytes,
    unsigned long long maximum_tokens, unsigned long long vocabulary_size,
    server_session_store_state *state, yvex_error *err);
void yvex_server_session_store_bytes_close(unsigned char **bytes);
void yvex_server_session_store_close(server_session_store_state *state);
int yvex_server_session_state_save(
    server_session *session, const char *path,
    yvex_runtime_state_store_summary *summary, yvex_error *err);
int yvex_server_session_state_restore(
    server_session *session, const char *path,
    unsigned long long maximum_file_bytes, unsigned long long vocabulary_size,
    yvex_runtime_state_store_summary *summary, yvex_error *err);
int yvex_server_session_generation_state_restore(
    server_session *session, yvex_error *err);

yvex_client_failure_class yvex_server_failure_class_from_status(int status);

int yvex_server_protocol_receive(int fd, yvex_client_request *request,
                                 unsigned char **owned_prompt,
                                 yvex_provider_request **owned_provider,
                                 yvex_error *err);
int yvex_server_protocol_send(int fd, const yvex_client_message *message,
                              yvex_error *err);

int yvex_server_telemetry_open(server_telemetry **out, unsigned long long capacity,
                          yvex_server_generation_mode generation_mode,
                          const char *runtime_model_identity,
                          const char *artifact_identity,
                          const char *variant_identity, yvex_error *err);
int yvex_server_telemetry_emit(server_telemetry *telemetry,
                          yvex_server_event_kind kind,
                          yvex_server_event_severity severity,
                          const char *session_id, const char *request_id,
                          const char *turn_id, const char *phase,
                          unsigned long long value_a,
                          unsigned long long value_b,
                          unsigned long long value_c,
                          double seconds, double rate, yvex_error *err);
int yvex_server_telemetry_emit_provider(
    server_telemetry *telemetry, yvex_server_event_kind kind,
    yvex_server_event_severity severity, const char *session_id,
    const char *request_id, const char *turn_id, const char *phase,
    unsigned long long value_a, unsigned long long value_b,
    unsigned long long value_c, double seconds, double rate,
    const yvex_runtime_speculation_progress *speculation,
    const yvex_provider_request *provider, yvex_server_event *emitted,
    yvex_error *err);
int yvex_server_telemetry_next(server_telemetry *telemetry,
                          unsigned long long after_sequence, int wait,
                          yvex_server_event *event, yvex_error *err);
int yvex_server_telemetry_metrics_copy(server_telemetry *telemetry,
                                  yvex_server_metrics *metrics,
                                  yvex_error *err);
void yvex_server_telemetry_identities(server_telemetry *telemetry,
                                 const char *runtime_model_identity,
                                 const char *artifact_identity,
                                 const char *variant_identity);
void yvex_server_telemetry_model_opened(server_telemetry *telemetry,
                                   unsigned long long artifact_bytes,
                                   unsigned long long host_bytes,
                                   unsigned long long device_bytes,
                                   unsigned long long uploads);
void yvex_server_telemetry_model_closed(server_telemetry *telemetry);
void yvex_server_telemetry_resources(server_telemetry *telemetry,
                                 unsigned long long host_bytes,
                                 unsigned long long device_bytes,
                                 unsigned long long uploads);
void yvex_server_telemetry_queue(server_telemetry *telemetry,
                            unsigned long long depth,
                            unsigned long long capacity);
void yvex_server_telemetry_session(server_telemetry *telemetry, int active_delta,
                              int created);
void yvex_server_telemetry_request(server_telemetry *telemetry, int active_delta,
                              int completed, int failed, int cancelled);
void yvex_server_telemetry_openai_request(server_telemetry *telemetry,
                                          int active_delta, int completed,
                                          int failed, int cancelled);
void yvex_server_telemetry_close(server_telemetry **telemetry);

int yvex_server_openai_prepare(server_openai_listener **out,
                               const server_openai_options *options,
                               server_telemetry *telemetry,
                               yvex_error *err);
int yvex_server_openai_start(server_openai_listener *listener,
                             yvex_error *err);
void yvex_server_openai_activate(server_openai_listener *listener);
void yvex_server_openai_request_stop(server_openai_listener *listener);
int yvex_server_openai_finish(server_openai_listener *listener,
                              yvex_error *err);
void yvex_server_openai_snapshot(const server_openai_listener *listener,
                                 server_openai_snapshot *snapshot);
void yvex_server_openai_close(server_openai_listener **listener);

int yvex_server_sessions_open(server_session_registry **out,
                                 yvex_runtime_model *model,
                                 const yvex_server_options *options,
                                 server_telemetry *telemetry,
                                 yvex_error *err);
int yvex_server_sessions_execute(server_session_registry *registry,
                                    const yvex_client_request *request,
                                    const char *request_id,
                                    double queue_seconds,
                                    server_message_emit emit,
                                    void *emit_context, yvex_error *err);
int yvex_server_sessions_count(server_session_registry *registry,
                                  unsigned long long *count, yvex_error *err);
int yvex_server_sessions_console_status(server_session_registry *registry,
                                        const char *session_name,
                                        yvex_console_status *status,
                                        yvex_client_partial_turn *partial_turn,
                                        yvex_error *err);
int yvex_server_sessions_cancel(server_session_registry *registry,
                                   const char *session_name,
                                   yvex_error *err);
void yvex_server_sessions_cancel_all(server_session_registry *registry);
int yvex_server_sessions_close(server_session_registry **registry,
                                  yvex_error *err);

#endif
