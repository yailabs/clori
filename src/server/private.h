/*
 * Connect admitted server owners without exposing sockets or engine pointers publicly.
 *
 * Only server translation units consume these declarations and all mutable owners are opaque.
 * Source-local interface shared by host, protocol, telemetry, and session owners.
 */
#ifndef SRC_SERVER_PRIVATE_H_INCLUDED
#define SRC_SERVER_PRIVATE_H_INCLUDED

#include <yvex/internal/generation.h>
#include <yvex/server.h>

typedef struct server_telemetry server_telemetry;
typedef struct server_session_registry server_session_registry;
typedef struct server_openai_listener server_openai_listener;

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
                                        yvex_error *err);
int yvex_server_sessions_cancel(server_session_registry *registry,
                                   const char *session_name,
                                   yvex_error *err);
void yvex_server_sessions_cancel_all(server_session_registry *registry);
int yvex_server_sessions_close(server_session_registry **registry,
                                  yvex_error *err);

#endif
