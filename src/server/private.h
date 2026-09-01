/*
 * Connect admitted server owners without exposing sockets or engine pointers publicly.
 *
 * Only server translation units consume these declarations and all mutable owners are opaque.
 * Source-local interface shared by host, protocol, telemetry, and session owners.
 */
#ifndef SRC_SERVER_PRIVATE_H_INCLUDED
#define SRC_SERVER_PRIVATE_H_INCLUDED

#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

#include <yvex/internal/generation.h>
#include <yvex/internal/server_media.h>
#include <yvex/internal/runtime_prefix.h>
#include <yvex/internal/runtime_state_store.h>
#include <yvex/server.h>

typedef struct server_telemetry server_telemetry;
typedef struct server_session_registry server_session_registry;
typedef struct server_media_registry server_media_registry;
typedef struct server_openai_listener server_openai_listener;
typedef struct server_request_queue server_request_queue;
typedef struct server_engine_manager server_engine_manager;
typedef struct server_engine_lease {
    void *engine;
    unsigned long long generation;
} server_engine_lease;
typedef struct {
    yvex_server_engine_kind engine_kind;
    yvex_server_execution_strategy execution_strategy;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char specialization_identity[YVEX_SHA256_HEX_CAP];
} server_event_scope;
static inline void server_event_scope_from_engine(
    server_event_scope *scope, const yvex_server_engine_summary *engine)
{
    memset(scope, 0, sizeof(*scope));
    if (!engine) return;
    scope->engine_kind = engine->engine_kind;
    scope->execution_strategy = engine->execution_strategy;
    memcpy(scope->runtime_model_identity, engine->runtime_model_identity,
           sizeof(scope->runtime_model_identity));
    memcpy(scope->artifact_identity, engine->artifact_identity,
           sizeof(scope->artifact_identity));
    memcpy(scope->specialization_identity, engine->specialization_identity,
           sizeof(scope->specialization_identity));
}
#define SERVER_REQUEST_QUEUE_KEY_CAP 224u

typedef void (*server_request_queue_execute)(void *context, void *work);
typedef void (*server_request_queue_observe)(void *context,
                                             unsigned long long queued,
                                             unsigned long long capacity,
                                             unsigned long long active);

typedef struct {
    unsigned long long queued, capacity, active, workers;
} server_request_queue_summary;

int yvex_server_request_queue_open(
    server_request_queue **out, unsigned long long queue_capacity,
    unsigned long long worker_count, server_request_queue_execute execute,
    server_request_queue_observe observe, void *context, yvex_error *err);
int yvex_server_request_queue_start(server_request_queue *request_queue,
                                    yvex_error *err);
int yvex_server_request_queue_key(
    char output[SERVER_REQUEST_QUEUE_KEY_CAP],
    unsigned long long engine_generation, const char *serialization_scope,
    yvex_error *err);
int yvex_server_request_queue_submit(server_request_queue *request_queue, void *work,
                                     const char *serialization_key,
                                     unsigned long long *queued, yvex_error *err);
void yvex_server_request_queue_request_stop(server_request_queue *request_queue);
int yvex_server_request_queue_finish(server_request_queue *request_queue,
                                     yvex_error *err);
void yvex_server_request_queue_snapshot(const server_request_queue *request_queue,
                                        server_request_queue_summary *summary);
void yvex_server_request_queue_close(server_request_queue **request_queue);

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

struct server_session_registry {
    pthread_mutex_t mutex;
    yvex_model_engine *model;
    yvex_server_engine_options options;
    yvex_reasoning_policy default_reasoning_policy;
    server_telemetry *telemetry;
    server_session *sessions;
    unsigned long long capacity, count, next_id;
    int mutex_ready, closing, continuous_batching;
    unsigned long long engine_generation;
    server_event_scope event_scope;
};

typedef struct {
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char specialization_identity[YVEX_SHA256_HEX_CAP];
} server_media_summary;

typedef struct {
    const char *yvex_socket;
    unsigned short port;
    unsigned long long timeout_ms, maximum_connections;
} server_openai_options;

typedef struct {
    unsigned short port;
    int enabled, ready;
} server_openai_snapshot;

typedef int (*server_message_emit)(void *context, const yvex_client_message *message,
                                   yvex_error *err);
static inline yvex_reasoning_policy server_reasoning_automatic_policy(void)
{
    /* Capability only determines whether an explicit mode can be admitted.
       An omitted policy selects the ordinary source-authored chat mode. */
    return YVEX_REASONING_DISABLED;
}

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
int yvex_server_session_state_clone(
    server_session *source, server_session *destination,
    unsigned long long vocabulary_size, yvex_error *err);

server_session *yvex_server_session_find_locked(
    server_session_registry *registry, const char *name);
int yvex_server_session_execution_open(
    server_session_registry *registry, server_session *session,
    yvex_error *err);
int yvex_server_session_create_locked(
    server_session_registry *registry, const char *requested,
    server_session **created, yvex_error *err);
int yvex_server_session_fork_locked(
    server_session_registry *registry, server_session *source,
    const char *requested, unsigned long long maximum_shared_bytes,
    server_session **created,
    yvex_runtime_session_prefix_summary *prefix_summary, yvex_error *err);
int yvex_server_session_reset_locked(
    server_session_registry *registry, server_session *session,
    yvex_error *err);
int yvex_server_session_close_locked(
    server_session_registry *registry, server_session *session,
    yvex_error *err);

yvex_client_failure_class yvex_server_failure_class_from_status(int status);

int yvex_server_protocol_receive(int fd, yvex_client_request *request,
                                 unsigned char **owned_prompt,
                                 yvex_provider_request **owned_provider,
                                 yvex_error *err);
int yvex_server_protocol_send(int fd, const yvex_client_message *message,
                              yvex_error *err);

int yvex_server_telemetry_open(server_telemetry **out,
                               unsigned long long capacity, yvex_error *err);
int yvex_server_telemetry_emit(server_telemetry *telemetry,
                          const server_event_scope *scope,
                          yvex_server_event_kind kind,
                          yvex_server_event_severity severity,
                          const char *session_id, const char *request_id,
                          const char *turn_id, const char *phase,
                          unsigned long long value_a,
                          unsigned long long value_b,
                          unsigned long long value_c,
                          double seconds, double rate, yvex_error *err);
int yvex_server_telemetry_emit_provider(
    server_telemetry *telemetry, const server_event_scope *scope,
    yvex_server_event_kind kind,
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
int yvex_server_telemetry_latest_sequence(server_telemetry *telemetry,
                                      unsigned long long *sequence,
                                      yvex_error *err);
int yvex_server_telemetry_metrics_copy(server_telemetry *telemetry,
                                  yvex_server_metrics *metrics,
                                  yvex_error *err);
void yvex_server_telemetry_model_opened(server_telemetry *telemetry,
                                   unsigned long long mapped_artifact_bytes,
                                   unsigned long long host_bytes,
                                   unsigned long long device_bytes,
                                   unsigned long long uploads);
void yvex_server_telemetry_media_model_opened(
    server_telemetry *telemetry, unsigned long long artifact_count);
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

int yvex_server_sessions_open(server_session_registry **out, yvex_model_engine *model,
                              const yvex_server_engine_options *options,
                              unsigned long long engine_generation,
                              int continuous_batching,
                              const server_event_scope *event_scope,
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
int yvex_server_sessions_resource_bytes(
    server_session_registry *, unsigned long long *, unsigned long long *,
    yvex_error *);
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

int yvex_server_media_registry_open(
    server_media_registry **, const yvex_server_media_options *, server_telemetry *, yvex_error *);
int yvex_server_media_registry_execute(
    server_media_registry *, const yvex_client_request *, const char *, double,
    server_message_emit, void *, yvex_error *);
int yvex_server_media_registry_console_status(
    server_media_registry *, const char *, yvex_console_status *,
    yvex_client_partial_turn *, yvex_error *);
int yvex_server_media_registry_cancel(
    server_media_registry *, const char *, yvex_error *);
void yvex_server_media_registry_cancel_all(server_media_registry *);
int yvex_server_media_registry_count(
    server_media_registry *, unsigned long long *, yvex_error *);
int yvex_server_media_registry_summary(
    server_media_registry *, server_media_summary *, yvex_error *);
int yvex_server_media_registry_start(
    server_media_registry *, yvex_runtime_media_model_summary *, yvex_error *);
void yvex_server_media_registry_close(server_media_registry **);

int yvex_server_engine_manager_open(
    server_engine_manager **, unsigned long long, unsigned long long,
    unsigned long long, server_request_queue_execute, void *,
    server_telemetry *, yvex_error *);
int yvex_server_engine_summary_valid(const yvex_server_engine_summary *);
int yvex_server_engine_manager_load(
    server_engine_manager *, const yvex_server_engine_options *,
    const yvex_server_media_options *, yvex_server_engine_summary *, yvex_error *);
int yvex_server_engine_manager_unload(
    server_engine_manager *, const char *, unsigned long long,
    yvex_server_engine_summary *, yvex_error *);
int yvex_server_engine_manager_snapshot(
    server_engine_manager *, yvex_server_engine_summary *, unsigned long long,
    unsigned long long *, yvex_error *);
int yvex_server_engine_manager_acquire(
    server_engine_manager *, const char *, unsigned long long,
    server_engine_lease *, yvex_server_engine_summary *, yvex_error *);
void yvex_server_engine_manager_release(
    server_engine_manager *, server_engine_lease *);
int yvex_server_engine_lease_submit(
    server_engine_lease *, void *, const char *,
    unsigned long long *, yvex_error *);
int yvex_server_engine_lease_execute(
    server_engine_lease *, const yvex_client_request *, const char *, double,
    server_message_emit, void *, yvex_error *);
int yvex_server_engine_lease_cancel(
    server_engine_lease *, const char *, yvex_error *);
int yvex_server_engine_lease_console_status(
    server_engine_lease *, const char *, yvex_console_status *,
    yvex_client_partial_turn *, yvex_error *);
void yvex_server_engine_manager_cancel_all(server_engine_manager *);
int yvex_server_engine_manager_request_queue_snapshot(
    server_engine_manager *, server_request_queue_summary *, yvex_error *);
int yvex_server_engine_manager_close(server_engine_manager **, yvex_error *);

#endif
