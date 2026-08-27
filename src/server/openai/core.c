/*
 * Compose bounded HTTP/OpenAI translation with the existing local runtime-host protocol.
 *
 * Every generation uses the local protocol; ephemeral sessions close and retained state is
 * explicit. The adapter is server-owned and never links or invokes the inference engine directly.
 */
#define _POSIX_C_SOURCE 200809L
#include "src/server/openai/private.h"
#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include <yvex/internal/core.h>
typedef struct {
    const openai_gateway *gateway;
    int http_fd;
    char model_alias[YVEX_SERVER_MODEL_ALIAS_CAP];
    unsigned long long engine_generation;
    char session_name[YVEX_SERVER_SESSION_NAME_CAP];
    atomic_int stop;
    atomic_int peer_closed;
    pthread_t thread;
    int started;
} disconnect_watch;

struct server_openai_listener {
    openai_gateway gateway;
    atomic_int stop, admit, ready;
    pthread_mutex_t connection_mutex;
    pthread_cond_t connection_idle;
    pthread_t thread;
    unsigned long long active_connections, maximum_connections;
    int listen_fd, thread_started, thread_status;
    int connection_mutex_ready, connection_idle_ready;
    yvex_error thread_error;
};

typedef struct {
    server_openai_listener *listener;
    int fd;
} openai_connection;

static unsigned long long wall_seconds(void)
{
    struct timespec now;
    return clock_gettime(CLOCK_REALTIME, &now) == 0
               ? (unsigned long long)now.tv_sec : 0u;
}

static int next_request_ordinal(openai_gateway *gateway,
                                unsigned long long *ordinal, yvex_error *err)
{
    unsigned long long current;
    if (!gateway || !ordinal) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.openai.request-id",
                       "request identity owner is unavailable");
        return YVEX_ERR_INVALID_ARG;
    }
    current = atomic_load_explicit(&gateway->next_id, memory_order_relaxed);
    do {
        if (current == ULLONG_MAX) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "server.openai.request-id",
                           "request identity sequence is exhausted");
            return YVEX_ERR_BOUNDS;
        }
    } while (!atomic_compare_exchange_weak_explicit(
        &gateway->next_id, &current, current + 1ull, memory_order_relaxed,
        memory_order_relaxed));
    *ordinal = current + 1ull;
    return YVEX_OK;
}
/*
 * Grow one bounded collected output span.
 *
 * Collects committed provider bytes without model or transport ownership.
 */
static int result_append(unsigned char **bytes, unsigned long long *count,
                         unsigned long long *capacity,
                         const unsigned char *fragment,
                         unsigned long long fragment_count, yvex_error *err)
{
    unsigned long long need, grown;
    unsigned char *storage;
    if (!bytes || !count || !capacity || (!fragment && fragment_count) ||
        *count > YVEX_PROVIDER_MAX_CONTENT_BYTES - fragment_count)
        return YVEX_ERR_BOUNDS;
    need = *count + fragment_count;
    if (need > *capacity) {
        grown = *capacity ? *capacity : 4096u;
        while (grown < need) {
            if (grown > YVEX_PROVIDER_MAX_CONTENT_BYTES / 2u) {
                grown = YVEX_PROVIDER_MAX_CONTENT_BYTES;
                break;
            }
            grown *= 2u;
        }
        storage = realloc(*bytes, (size_t)grown);
        if (!storage) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "server.openai.result",
                           "provider output allocation failed");
            return YVEX_ERR_NOMEM;
        }
        *bytes = storage;
        *capacity = grown;
    }
    if (fragment_count)
        memcpy(*bytes + *count, fragment, (size_t)fragment_count);
    *count = need;
    return YVEX_OK;
}

static void result_clear(openai_generation_result *result)
{
    unsigned long long index;
    if (!result) return;
    free(result->reasoning);
    free(result->text);
    for (index = 0u; index < result->tool_call_count; ++index)
        free(result->tool_calls[index].arguments);
    memset(result, 0, sizeof(*result));
}

static int client_connect(const openai_gateway *gateway, yvex_client **client,
                          yvex_error *err)
{
    int rc = yvex_client_connect(client, gateway->yvex_socket, err);
    if (rc == YVEX_OK)
        rc = yvex_client_timeout_set(*client, gateway->yvex_timeout_ms, err);
    if (rc != YVEX_OK) yvex_client_close(client);
    return rc;
}

static int daemon_status(const openai_gateway *gateway,
                         yvex_server_summary *summary, yvex_error *err)
{
    yvex_client *client = NULL;
    yvex_client_request request = {0};
    yvex_client_message message;
    int rc = client_connect(gateway, &client, err);
    if (rc == YVEX_OK) {
        request.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
        request.operation = YVEX_CLIENT_OP_RUNTIME_STATUS;
        request.request_number = 1u;
        rc = yvex_client_send(client, &request, err);
    }
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK && (message.kind != YVEX_CLIENT_MESSAGE_STATUS ||
                          message.status != YVEX_OK ||
                          !message.runtime.host_ready)) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.openai.runtime",
                       "YVEX server is not ready");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK && summary) *summary = message.runtime;
    yvex_client_close(&client);
    return rc;
}

static int daemon_engines(const openai_gateway *gateway,
                          yvex_server_engine_summary *engines,
                          unsigned long long capacity,
                          unsigned long long *count, yvex_error *err)
{
    yvex_client *client = NULL;
    yvex_client_request request = {0};
    unsigned long long written = 0ull;
    int complete = 0;
    int rc;
    if (!gateway || (!engines && capacity) || !count) return YVEX_ERR_INVALID_ARG;
    *count = 0ull;
    rc = client_connect(gateway, &client, err);
    if (rc == YVEX_OK) {
        request.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
        request.operation = YVEX_CLIENT_OP_ENGINE_LIST;
        request.request_number = 1u;
        rc = yvex_client_send(client, &request, err);
    }
    while (rc == YVEX_OK && !complete) {
        yvex_client_message message;
        rc = yvex_client_receive(client, &message, err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_ERROR || message.status != YVEX_OK) {
            rc = message.status ? message.status : YVEX_ERR;
            yvex_error_set(err, (yvex_status)rc, "server.openai.engines",
                           message.reason);
        } else if (message.kind == YVEX_CLIENT_MESSAGE_ACK) {
            complete = 1;
        } else if (message.kind != YVEX_CLIENT_MESSAGE_ENGINE) {
            rc = YVEX_ERR_FORMAT;
            yvex_error_set(err, YVEX_ERR_FORMAT, "server.openai.engines",
                           "YVEX server returned an invalid engine list");
        } else if (message.engine.state == YVEX_SERVER_ENGINE_LOADED &&
                   message.engine.execution_ready) {
            if (written >= capacity) {
                rc = YVEX_ERR_BOUNDS;
                yvex_error_set(err, YVEX_ERR_BOUNDS, "server.openai.engines",
                               "loaded engine list exceeds adapter capacity");
            } else {
                engines[written++] = message.engine;
            }
        }
    }
    if (rc == YVEX_OK && !complete) {
        rc = YVEX_ERR_FORMAT;
        yvex_error_set(err, YVEX_ERR_FORMAT, "server.openai.engines",
                       "YVEX server engine list is incomplete");
    }
    if (rc == YVEX_OK) *count = written;
    yvex_client_close(&client);
    return rc;
}

static const yvex_server_engine_summary *engine_find(
    const yvex_server_engine_summary *engines, unsigned long long count,
    const char *alias)
{
    unsigned long long index;
    if ((!engines && count) || !alias || !alias[0]) return NULL;
    for (index = 0ull; index < count; ++index)
        if (!strcmp(engines[index].alias, alias)) return &engines[index];
    return NULL;
}

static void request_bind(yvex_client_request *request, const char *model_alias,
                         unsigned long long engine_generation)
{
    yvex_core_text_copy(request->model_alias, sizeof(request->model_alias),
                        model_alias);
    request->engine_generation = engine_generation;
}
/*
 * Append prior Responses context and current input without reconstructing hidden state.
 *
 * Clones one complete provider request graph and reseals its identity.
 */
static int context_combine(const yvex_provider_request *prior,
                           const yvex_provider_request *current,
                           yvex_provider_request **combined, yvex_error *err)
{
    yvex_provider_request candidate;
    yvex_provider_message *messages;
    unsigned long long count;
    int rc;
    *combined = NULL;
    if (!prior) return yvex_provider_request_clone(current, combined, err);
    if (strcmp(prior->model, current->model) != 0 ||
        prior->message_count > YVEX_PROVIDER_MAX_MESSAGES - current->message_count)
        return YVEX_ERR_STATE;
    count = prior->message_count + current->message_count;
    messages = calloc((size_t)count, sizeof(*messages));
    if (!messages) return YVEX_ERR_NOMEM;
    memcpy(messages, prior->messages,
           (size_t)prior->message_count * sizeof(*messages));
    memcpy(messages + prior->message_count, current->messages,
           (size_t)current->message_count * sizeof(*messages));
    candidate = *current;
    candidate.messages = messages;
    candidate.message_count = count;
    if (!candidate.tool_count) {
        candidate.tools = prior->tools;
        candidate.tool_count = prior->tool_count;
    }
    candidate.sealed = 0;
    candidate.request_identity[0] = '\0';
    rc = yvex_provider_request_seal(&candidate, err);
    if (rc == YVEX_OK)
        rc = yvex_provider_request_clone(&candidate, combined, err);
    free(messages);
    return rc;
}
/*
 * Append one authoritative assistant output to retained Responses context.
 *
 * Clones and extends the message graph, then seals a new context identity.
 */
static int context_complete(const yvex_provider_request *request,
                            const openai_generation_result *result,
                            yvex_provider_request **context, yvex_error *err)
{
    yvex_provider_request candidate = *request;
    yvex_provider_message *messages;
    yvex_provider_tool_call calls[YVEX_PROVIDER_MAX_TOOLS] = {0};
    unsigned long long index;
    int rc;
    *context = NULL;
    if (request->message_count >= YVEX_PROVIDER_MAX_MESSAGES)
        return YVEX_ERR_BOUNDS;
    messages = calloc((size_t)request->message_count + 1u, sizeof(*messages));
    if (!messages) return YVEX_ERR_NOMEM;
    memcpy(messages, request->messages,
           (size_t)request->message_count * sizeof(*messages));
    messages[request->message_count].role = YVEX_PROVIDER_ROLE_ASSISTANT;
    messages[request->message_count].reasoning_content.bytes = result->reasoning;
    messages[request->message_count].reasoning_content.count =
        result->reasoning_count;
    messages[request->message_count].content.bytes = result->text;
    messages[request->message_count].content.count = result->text_count;
    for (index = 0u; index < result->tool_call_count; ++index) {
        yvex_core_text_copy(calls[index].call_id, sizeof(calls[index].call_id),
                            result->tool_calls[index].call_id);
        yvex_core_text_copy(calls[index].name, sizeof(calls[index].name),
                            result->tool_calls[index].name);
        calls[index].arguments_json.bytes =
            result->tool_calls[index].arguments;
        calls[index].arguments_json.count =
            result->tool_calls[index].arguments_count;
    }
    messages[request->message_count].tool_calls = calls;
    messages[request->message_count].tool_call_count = result->tool_call_count;
    candidate.messages = messages;
    candidate.message_count = request->message_count + 1u;
    candidate.previous_response_id[0] = '\0';
    candidate.sealed = 0;
    candidate.request_identity[0] = '\0';
    rc = yvex_provider_request_seal(&candidate, err);
    if (rc == YVEX_OK)
        rc = yvex_provider_request_clone(&candidate, context, err);
    free(messages);
    return rc;
}

static void cancel_session(const openai_gateway *gateway,
                           const char *model_alias,
                           unsigned long long engine_generation,
                           const char *session_name)
{
    yvex_client *client = NULL;
    yvex_client_request request = {0};
    yvex_client_message response;
    yvex_error err;
    if (!session_name || !session_name[0] ||
        client_connect(gateway, &client, &err) != YVEX_OK)
        return;
    request.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request.operation = YVEX_CLIENT_OP_GENERATION_CANCEL;
    request.request_number = 1u;
    request_bind(&request, model_alias, engine_generation);
    yvex_core_text_copy(request.session_name, sizeof(request.session_name),
                        session_name);
    if (yvex_client_send(client, &request, &err) == YVEX_OK)
        (void)yvex_client_receive(client, &response, &err);
    yvex_client_close(&client);
}

static void *disconnect_watch_main(void *opaque)
{
    disconnect_watch *watch = opaque;
    while (!atomic_load_explicit(&watch->stop, memory_order_acquire)) {
        int closed = 0;
        yvex_error err;
        int rc = openai_http_peer_wait(watch->http_fd, 100u, &closed, &err);
        if ((watch->gateway->stop &&
             atomic_load_explicit(watch->gateway->stop,
                                  memory_order_acquire)) ||
            rc != YVEX_OK || closed) {
            atomic_store_explicit(&watch->peer_closed, 1,
                                  memory_order_release);
            cancel_session(watch->gateway, watch->model_alias,
                           watch->engine_generation, watch->session_name);
            break;
        }
    }
    return NULL;
}

static int disconnect_watch_open(disconnect_watch *watch,
                                 const openai_gateway *gateway, int http_fd,
                                 const char *model_alias,
                                 unsigned long long engine_generation,
                                 const char *session_name, yvex_error *err)
{
    if (!watch || !gateway || http_fd < 0 || !model_alias || !model_alias[0] ||
        !engine_generation || !session_name || !session_name[0])
        return YVEX_ERR_INVALID_ARG;
    memset(watch, 0, sizeof(*watch));
    watch->gateway = gateway;
    watch->http_fd = http_fd;
    yvex_core_text_copy(watch->model_alias, sizeof(watch->model_alias),
                        model_alias);
    watch->engine_generation = engine_generation;
    yvex_core_text_copy(watch->session_name, sizeof(watch->session_name),
                        session_name);
    atomic_init(&watch->stop, 0);
    atomic_init(&watch->peer_closed, 0);
    if (pthread_create(&watch->thread, NULL, disconnect_watch_main, watch) != 0) {
        yvex_error_set(err, YVEX_ERR_IO, "server.openai.disconnect",
                       "HTTP disconnect watcher could not start");
        return YVEX_ERR_IO;
    }
    watch->started = 1;
    return YVEX_OK;
}

static int disconnect_watch_close(disconnect_watch *watch)
{
    int peer_closed;
    if (!watch || !watch->started) return 0;
    atomic_store_explicit(&watch->stop, 1, memory_order_release);
    (void)pthread_join(watch->thread, NULL);
    peer_closed = atomic_load_explicit(&watch->peer_closed,
                                       memory_order_acquire);
    watch->started = 0;
    return peer_closed;
}

static int session_operation(yvex_client *client, yvex_client_operation operation,
                             const char *model_alias,
                             unsigned long long engine_generation,
                             const char *session_name, yvex_error *err)
{
    yvex_client_request request = {0};
    yvex_client_message response;
    int rc;
    request.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request.operation = operation;
    request.request_number = 1u;
    request_bind(&request, model_alias, engine_generation);
    yvex_core_text_copy(request.session_name, sizeof(request.session_name),
                        session_name);
    rc = yvex_client_send(client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &response, err);
    if (rc == YVEX_OK && response.status != YVEX_OK) {
        yvex_error_set(err, (yvex_status)response.status,
                       "server.openai.session", response.reason);
        rc = response.status;
    }
    return rc;
}

static int session_close(openai_gateway *gateway, const char *model_alias,
                         unsigned long long engine_generation,
                         const char *session_name, yvex_error *err)
{
    yvex_client *client = NULL;
    int rc = client_connect(gateway, &client, err);
    if (rc == YVEX_OK)
        rc = session_operation(client, YVEX_CLIENT_OP_SESSION_CLOSE,
                               model_alias, engine_generation, session_name, err);
    yvex_client_close(&client);
    return rc;
}
/* Reclaim expired state and, when required, one deterministic LRU slot. */
static int state_prepare(openai_gateway *gateway, unsigned long long now,
                         int require_free, yvex_error *err)
{
    openai_response_record *oldest = NULL;
    unsigned long long index;
    int free_slot = 0;
    for (index = 0u; index < OPENAI_RESPONSE_RECORD_MAX; ++index) {
        openai_response_record *record = &gateway->records[index];
        int expired = record->occupied && now >= record->created_seconds &&
                      now - record->created_seconds >
                          OPENAI_RESPONSE_TTL_SECONDS;
        if (expired) {
            int rc = session_close(gateway, record->model,
                                   record->engine_generation,
                                   record->session_name, err);
            if (rc != YVEX_OK) return rc;
            openai_state_remove(record);
        }
        if (!record->occupied)
            free_slot = 1;
        else if (!oldest || record->last_used_sequence <
                                oldest->last_used_sequence)
            oldest = record;
    }
    if (!require_free || free_slot) return YVEX_OK;
    if (!oldest) return YVEX_ERR_STATE;
    if (session_close(gateway, oldest->model, oldest->engine_generation,
                      oldest->session_name, err) != YVEX_OK)
        return yvex_error_code(err);
    openai_state_remove(oldest);
    return YVEX_OK;
}

static int state_sessions_close(openai_gateway *gateway, yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    for (index = 0u; index < OPENAI_RESPONSE_RECORD_MAX; ++index) {
        openai_response_record *record = &gateway->records[index];
        if (record->occupied) {
            yvex_error local;
            int close_rc = session_close(gateway, record->model,
                                         record->engine_generation,
                                         record->session_name, &local);
            if (close_rc != YVEX_OK && rc == YVEX_OK) {
                rc = close_rc;
                *err = local;
            }
            openai_state_remove(record);
        }
    }
    return rc;
}

static const char *response_sse_name(openai_response_event_kind kind)
{
    static const char *const names[] = {
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.reasoning_content.delta",
        "response.reasoning_content.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
        "response.incomplete",
        "response.failed"
    };
    return kind <= OPENAI_RESPONSE_EVENT_FAILED ? names[kind] : NULL;
}

static int response_event_emit(openai_http_sink *sink,
                               openai_response_event_kind kind,
                               const char *id, const char *model,
                               unsigned long long created,
                               const yvex_client_message *message,
                               const openai_generation_result *result,
                               unsigned long long output_index,
                               yvex_error *err)
{
    unsigned char *json = NULL;
    unsigned long long count = 0u;
    int rc = openai_json_response_event(
        kind, id, model, created, message, result,
        output_index, sink->response_sequence, &json, &count, err);
    if (rc == YVEX_OK)
        rc = openai_http_sse_event(sink->fd, response_sse_name(kind),
                                   json, count, err);
    if (rc == YVEX_OK) sink->response_sequence++;
    free(json);
    return rc;
}

static int generation_tool_call(
    openai_generation_result *result, const yvex_client_message *message,
    unsigned long long *tool_index, yvex_error *err)
{
    unsigned long long index;
    if (!message->tool_call_id[0] || !message->tool_name[0]) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "server.openai.tool",
                       "typed tool fragments require call and function identities");
        return YVEX_ERR_FORMAT;
    }
    for (index = 0u; index < result->tool_call_count; ++index)
        if (strcmp(result->tool_calls[index].call_id,
                   message->tool_call_id) == 0) {
            if (strcmp(result->tool_calls[index].name, message->tool_name) != 0) {
                yvex_error_set(err, YVEX_ERR_FORMAT, "server.openai.tool",
                               "one tool-call identity changed function name");
                return YVEX_ERR_FORMAT;
            }
            *tool_index = index;
            return YVEX_OK;
        }
    if (result->tool_call_count >= YVEX_PROVIDER_MAX_TOOLS) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.openai.tool",
                       "tool-call result exceeds provider capacity");
        return YVEX_ERR_BOUNDS;
    }
    index = result->tool_call_count++;
    yvex_core_text_copy(result->tool_calls[index].call_id,
                        sizeof(result->tool_calls[index].call_id),
                        message->tool_call_id);
    yvex_core_text_copy(result->tool_calls[index].name,
                        sizeof(result->tool_calls[index].name),
                        message->tool_name);
    *tool_index = index;
    return YVEX_OK;
}

static int response_terminal_emit(openai_http_sink *sink, const char *id,
                                  const char *model,
                                  unsigned long long created,
                                  const yvex_client_message *message,
                                  const openai_generation_result *result,
                                  yvex_error *err)
{
    openai_response_event_kind terminal =
        result->finish == YVEX_PROVIDER_FINISH_LENGTH
            ? OPENAI_RESPONSE_EVENT_INCOMPLETE
            : OPENAI_RESPONSE_EVENT_COMPLETED;
    unsigned long long index, output_index;
    int rc;
    if (result->reasoning_count) {
        rc = response_event_emit(sink, OPENAI_RESPONSE_EVENT_REASONING_DONE,
                                 id, model, created, message, result, 0u, err);
    } else {
        rc = YVEX_OK;
    }
    if (rc == YVEX_OK && (result->text_count || !result->tool_call_count)) {
        rc = response_event_emit(sink, OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DONE,
                                 id, model, created, message, result, 0u, err);
        if (rc == YVEX_OK)
            rc = response_event_emit(
                sink, OPENAI_RESPONSE_EVENT_CONTENT_PART_DONE,
                id, model, created, message, result, 0u, err);
        if (rc == YVEX_OK)
            rc = response_event_emit(
                sink, OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_DONE,
                id, model, created, message, result, 0u, err);
    }
    for (index = 0u; rc == YVEX_OK && index < result->tool_call_count;
         ++index) {
        output_index = (result->text_count ? 1u : 0u) + index;
        rc = response_event_emit(
            sink, OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DONE,
            id, model, created, message, result, output_index, err);
        if (rc == YVEX_OK)
            rc = response_event_emit(
                sink, OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_DONE,
                id, model, created, message, result, output_index, err);
    }
    if (rc == YVEX_OK)
        rc = response_event_emit(sink, terminal, id, model, created,
                                 message, result, 0u, err);
    return rc;
}

static int response_stream_complete(openai_http_sink *sink, const char *id,
                                    const char *model,
                                    unsigned long long created,
                                    const openai_generation_result *result,
                                    yvex_error *err)
{
    int rc = YVEX_OK;
    if (!sink->response_item_mask && !result->tool_call_count) {
        rc = response_event_emit(sink, OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_ADDED,
                                 id, model, created, NULL, result, 0u, err);
        if (rc == YVEX_OK)
            rc = response_event_emit(
                sink, OPENAI_RESPONSE_EVENT_CONTENT_PART_ADDED,
                id, model, created, NULL, result, 0u, err);
        if (rc == YVEX_OK) sink->response_item_mask = 1u;
    }
    if (rc == YVEX_OK)
        rc = response_terminal_emit(sink, id, model, created, NULL, result,
                                    err);
    return rc;
}

static int generation_message(openai_http_sink *sink, const char *id,
                              const char *model, unsigned long long created,
                              const yvex_client_message *message,
                              openai_generation_result *result,
                              yvex_error *err)
{
    unsigned char *json = NULL;
    unsigned long long count = 0u, tool_index = 0u, output_index = 0u;
    int rc = YVEX_OK;
    if (message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
        if (message->provider_output_kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL) {
            rc = generation_tool_call(result, message, &tool_index, err);
            if (rc == YVEX_OK) {
                openai_generation_tool_call *call = &result->tool_calls[tool_index];
                rc = result_append(&call->arguments, &call->arguments_count,
                                   &call->arguments_capacity, message->bytes,
                                   message->byte_count, err);
                output_index = (result->text_count ? 1u : 0u) + tool_index;
            }
        } else if (message->provider_output_kind ==
                   YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING) {
            rc = result_append(&result->reasoning, &result->reasoning_count,
                               &result->reasoning_capacity, message->bytes,
                               message->byte_count, err);
        } else {
            rc = result_append(&result->text, &result->text_count,
                               &result->text_capacity, message->bytes,
                               message->byte_count, err);
        }
    } else if (message->kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE) {
        result->prompt_tokens = message->prompt_tokens;
        result->completion_tokens = message->completion_tokens;
        result->total_tokens = message->total_tokens;
        result->reasoning_tokens = message->reasoning_tokens;
        result->final_tokens = message->final_tokens;
        result->first_reasoning_seconds = message->first_reasoning_seconds;
        result->first_final_seconds = message->first_final_seconds;
        result->reasoning_seconds = message->reasoning_seconds;
        result->final_seconds = message->final_seconds;
        result->total_completion_seconds = message->total_completion_seconds;
        result->reasoning_rate = message->reasoning_rate;
        result->final_rate = message->final_rate;
        result->total_completion_rate = message->total_completion_rate;
        result->finish = message->provider_finish;
        yvex_core_text_copy(result->turn_identity,
                            sizeof(result->turn_identity),
                            message->turn_identity);
        result->complete = 1;
    }
    if (rc == YVEX_OK && sink->stream &&
        sink->endpoint == OPENAI_ENDPOINT_RESPONSES) {
        if (message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT &&
            message->provider_output_kind ==
                YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING) {
            rc = response_event_emit(
                sink, OPENAI_RESPONSE_EVENT_REASONING_DELTA, id, model,
                created, message, result, 0u, err);
        } else if (message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
            unsigned long long mask = 1ull << output_index;
            if (!(sink->response_item_mask & mask)) {
                rc = response_event_emit(
                    sink, OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_ADDED,
                    id, model, created, message, result, output_index, err);
                if (rc == YVEX_OK &&
                    message->provider_output_kind !=
                        YVEX_PROVIDER_OUTPUT_FUNCTION_CALL)
                    rc = response_event_emit(
                        sink, OPENAI_RESPONSE_EVENT_CONTENT_PART_ADDED,
                        id, model, created, message, result, output_index, err);
                if (rc == YVEX_OK) sink->response_item_mask |= mask;
            }
            if (rc == YVEX_OK)
                rc = response_event_emit(
                    sink,
                    message->provider_output_kind ==
                            YVEX_PROVIDER_OUTPUT_FUNCTION_CALL
                        ? OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DELTA
                        : OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DELTA,
                    id, model, created, message, result, output_index, err);
        }
    } else if (rc == YVEX_OK && sink->stream &&
               message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
        rc = openai_json_stream_chunk(sink->endpoint, id, model, created,
                                      message, tool_index, 0, &json, &count,
                                      err);
        if (rc == YVEX_OK)
            rc = openai_http_sse_event(
                sink->fd,
                NULL,
                json, count, err);
        free(json);
    }
    return rc;
}

static int generation_execute(openai_gateway *gateway,
                              openai_http_sink *sink, const char *id,
                              const yvex_server_engine_summary *engine,
                              unsigned long long created,
                              const char *session_name,
                              yvex_provider_request *provider,
                              openai_generation_result *result,
                              yvex_error *err)
{
    yvex_client *client = NULL;
    yvex_client_request request = {0};
    yvex_client_message message;
    int rc = client_connect(gateway, &client, err);
    if (rc != YVEX_OK) return rc;
    request.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request.operation = YVEX_CLIENT_OP_GENERATION_TURN;
    request.request_number = 2u;
    request.reasoning_policy = provider->reasoning_policy;
    request.provider_request = provider;
    request_bind(&request, engine->alias, engine->generation);
    yvex_core_text_copy(request.session_name, sizeof(request.session_name),
                        session_name);
    rc = yvex_client_send(client, &request, err);
    while (rc == YVEX_OK && !result->complete) {
        rc = yvex_client_receive(client, &message, err);
        if (rc != YVEX_OK) break;
        if (message.status != YVEX_OK || message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            result->failure_class = message.failure_class;
            yvex_error_set(err, (yvex_status)(message.status ? message.status
                                                             : YVEX_ERR),
                           "server.openai.generation", message.reason);
            rc = message.status ? message.status : YVEX_ERR;
            break;
        }
        if (message.kind == YVEX_CLIENT_MESSAGE_TURN_STARTED && sink->stream &&
            !sink->headers_sent) {
            rc = openai_http_sse_begin(sink->fd, err);
            if (rc == YVEX_OK) sink->headers_sent = 1;
            if (rc == YVEX_OK && sink->endpoint == OPENAI_ENDPOINT_RESPONSES)
                rc = response_event_emit(
                    sink, OPENAI_RESPONSE_EVENT_CREATED, id, engine->alias, created,
                    NULL, result, 0u, err);
            else if (rc == YVEX_OK) {
                unsigned char *json = NULL;
                unsigned long long count = 0u;
                rc = openai_json_stream_chunk(sink->endpoint, id, engine->alias,
                                              created, NULL, 0u, 1, &json,
                                              &count, err);
                if (rc == YVEX_OK)
                    rc = openai_http_sse_event(sink->fd, NULL,
                                               json, count, err);
                free(json);
            }
        } else if (message.kind == YVEX_CLIENT_MESSAGE_FRAGMENT ||
                   message.kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE) {
            rc = generation_message(sink, id, engine->alias, created, &message,
                                    result, err);
        }
    }
    yvex_client_close(&client);
    return rc;
}

static int chat_stream_complete(openai_http_sink *sink, const char *id,
                                const char *model,
                                unsigned long long created,
                                const openai_generation_result *result,
                                int include_usage, yvex_error *err)
{
    yvex_client_message terminal = {0};
    unsigned char *json = NULL;
    unsigned long long count = 0u;
    int rc;
    terminal.kind = YVEX_CLIENT_MESSAGE_TURN_COMPLETE;
    terminal.provider_finish = result->finish;
    rc = openai_json_stream_chunk(OPENAI_ENDPOINT_CHAT, id, model, created,
                                  &terminal, 0u, 0, &json, &count, err);
    if (rc == YVEX_OK)
        rc = openai_http_sse_event(sink->fd, NULL, json, count, err);
    free(json);
    json = NULL;
    if (rc == YVEX_OK && include_usage) {
        rc = openai_json_chat_usage_chunk(id, model, created, result, &json,
                                          &count, err);
        if (rc == YVEX_OK)
            rc = openai_http_sse_event(sink->fd, NULL, json, count, err);
        free(json);
    }
    if (rc == YVEX_OK) rc = openai_http_sse_done(sink->fd, err);
    return rc;
}

static int http_status(int status, yvex_client_failure_class failure_class)
{
    switch (failure_class) {
    case YVEX_CLIENT_FAILURE_INVALID_REQUEST: return 400;
    case YVEX_CLIENT_FAILURE_MODEL_NOT_FOUND: return 404;
    case YVEX_CLIENT_FAILURE_INCOMPATIBLE_STATE: return 409;
    case YVEX_CLIENT_FAILURE_REQUEST_TOO_LARGE: return 413;
    case YVEX_CLIENT_FAILURE_UNSUPPORTED_PARAMETER: return 422;
    case YVEX_CLIENT_FAILURE_QUEUE_FULL: return 429;
    case YVEX_CLIENT_FAILURE_CLIENT_CANCELLED: return 499;
    case YVEX_CLIENT_FAILURE_INTERNAL: return 500;
    case YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE: return 503;
    case YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT: return 504;
    case YVEX_CLIENT_FAILURE_NONE: break;
    }
    switch (status) {
    case YVEX_ERR_FORMAT:
    case YVEX_ERR_INVALID_ARG: return 400;
    case YVEX_ERR_UNSUPPORTED: return 422;
    case YVEX_ERR_BOUNDS: return 413;
    case YVEX_ERR_CANCELLED: return 499;
    case YVEX_ERR_STATE: return 409;
    case YVEX_ERR_IO:
    case YVEX_ERR_BACKEND: return 503;
    case YVEX_ERR_TIMEOUT: return 504;
    default: return 500;
    }
}

static int send_error(int fd, int status, const char *message)
{
    unsigned char *json = NULL;
    unsigned long long count = 0u;
    yvex_error err;
    const char *type = status == 429 ? "rate_limit_error" :
                       status >= 500 ? "server_error" :
                       status == 422 ? "unsupported_parameter" :
                       status == 409 ? "incompatible_state" :
                                       "invalid_request_error";
    const char *code = status == 404 ? "model_not_found" :
                       status == 409 ? "incompatible_state" :
                       status == 413 ? "request_too_large" :
                       status == 422 ? "unsupported_parameter" :
                       status == 429 ? "queue_full" :
                       status == 499 ? "client_cancelled" :
                       status == 500 ? "internal_error" :
                       status == 503 ? "runtime_unavailable" :
                       status == 504 ? "gateway_timeout" :
                                       "invalid_request";
    if (openai_json_error(status, type, NULL, code,
                          message ? message : "request failed",
                          &json, &count, &err) != YVEX_OK)
        return YVEX_ERR;
    (void)openai_http_json(fd, status, json, count, &err);
    free(json);
    return YVEX_OK;
}

static int route(const openai_http_request *request,
                 openai_endpoint *endpoint, char model[YVEX_PROVIDER_MODEL_CAP])
{
    model[0] = '\0';
    if (strcmp(request->method, "GET") == 0 && strcmp(request->path, "/health") == 0)
        *endpoint = OPENAI_ENDPOINT_HEALTH;
    else if (strcmp(request->method, "GET") == 0 && strcmp(request->path, "/v1/models") == 0)
        *endpoint = OPENAI_ENDPOINT_MODELS;
    else if (strcmp(request->method, "GET") == 0 &&
             strncmp(request->path, "/v1/models/", 11u) == 0) {
        if (strlen(request->path + 11u) >= YVEX_PROVIDER_MODEL_CAP) return 0;
        strcpy(model, request->path + 11u);
        *endpoint = OPENAI_ENDPOINT_MODEL;
    } else if (strcmp(request->method, "POST") == 0 &&
               strcmp(request->path, "/v1/chat/completions") == 0)
        *endpoint = OPENAI_ENDPOINT_CHAT;
    else if (strcmp(request->method, "POST") == 0 &&
             strcmp(request->path, "/v1/responses") == 0)
        *endpoint = OPENAI_ENDPOINT_RESPONSES;
    else return 0;
    return 1;
}

static int handle_read(openai_gateway *gateway, int fd,
                       openai_endpoint endpoint, const char *requested_model)
{
    static const unsigned char healthy[] =
        "{\"status\":\"ok\",\"adapter\":\"ready\",\"server\":\"ready\","
        "\"profile\":\"" OPENAI_COMPAT_PROFILE "\"}";
    yvex_server_summary summary;
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    const yvex_server_engine_summary *selected = NULL;
    unsigned char *json = NULL;
    unsigned long long count = 0u, engine_count = 0u;
    yvex_error err;
    int rc = daemon_status(gateway, &summary, &err);
    if (rc != YVEX_OK)
        return send_error(fd, 503, "YVEX server is unavailable or not ready");
    if (endpoint == OPENAI_ENDPOINT_HEALTH)
        return openai_http_json(fd, 200, healthy, sizeof(healthy) - 1u, &err);
    rc = daemon_engines(gateway, engines, YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES,
                        &engine_count, &err);
    if (rc != YVEX_OK)
        return send_error(fd, 503, "YVEX engine catalog is unavailable");
    if (endpoint == OPENAI_ENDPOINT_MODEL) {
        selected = engine_find(engines, engine_count, requested_model);
        if (!selected)
            return send_error(fd, 404, "requested model is not loaded");
    }
    rc = openai_json_models(selected ? selected : engines,
                            selected ? 1ull : engine_count,
                            endpoint == OPENAI_ENDPOINT_MODELS, &json, &count,
                            &err);
    if (rc == YVEX_OK) rc = openai_http_json(fd, 200, json, count, &err);
    free(json);
    return rc;
}

static int generation_admit_engine(openai_gateway *gateway,
                                   const openai_http_request *http,
                                   openai_endpoint endpoint,
                                   openai_admitted_request *admitted,
                                   yvex_server_engine_summary *engine,
                                   int *error_status, yvex_error *err)
{
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    const yvex_server_engine_summary *selected;
    unsigned long long engine_count = 0ull;
    int rc = openai_json_admit(http, endpoint,
                               server_reasoning_automatic_policy(), admitted, err);
    if (rc != YVEX_OK) {
        *error_status = http_status(rc, YVEX_CLIENT_FAILURE_NONE);
        return rc;
    }
    rc = daemon_engines(gateway, engines, YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES,
                        &engine_count, err);
    if (rc != YVEX_OK) {
        *error_status = 503;
        yvex_error_set(err, rc, "server.openai.engines",
                       "YVEX engine catalog is unavailable");
        return rc;
    }
    selected = engine_find(engines, engine_count, admitted->provider->model);
    if (!selected) {
        *error_status = 404;
        yvex_error_set(err, YVEX_ERR_STATE, "server.openai.engine",
                       "requested model is not loaded");
        return YVEX_ERR_STATE;
    }
    *engine = *selected;
    return YVEX_OK;
}

static int handle_generation(openai_gateway *gateway, int fd,
                             const openai_http_request *http,
                             openai_endpoint endpoint)
{
    yvex_server_engine_summary engine = {0};
    openai_admitted_request admitted = {0};
    openai_generation_result result = {0};
    disconnect_watch watch = {0};
    openai_http_sink sink = {
        .fd = fd,
        .endpoint = endpoint
    };
    openai_response_record *prior = NULL, *retained = NULL;
    yvex_provider_request *combined = NULL, *context = NULL;
    yvex_client *session_client = NULL;
    char id[YVEX_PROVIDER_ID_CAP] = {0};
    char session[YVEX_SERVER_SESSION_NAME_CAP] = {0};
    unsigned char *json = NULL;
    unsigned long long json_count = 0u, now = wall_seconds();
    unsigned long long request_ordinal;
    int stateful = endpoint == OPENAI_ENDPOINT_RESPONSES;
    int created_session = 0, generation_started = 0, peer_closed = 0;
    int state_locked = 0, error_status = 500, rc;
    yvex_error err, failure_error;
    rc = daemon_status(gateway, NULL, &err);
    if (rc != YVEX_OK) return send_error(fd, 503, "YVEX server is unavailable or not ready");
    rc = generation_admit_engine(gateway, http, endpoint, &admitted, &engine,
                                 &error_status, &err);
    if (rc != YVEX_OK) {
        openai_admitted_request_clear(&admitted);
        return send_error(fd, error_status, yvex_error_message(&err));
    }
    rc = next_request_ordinal(gateway, &request_ordinal, &err);
    if (rc != YVEX_OK) goto failure;
    (void)snprintf(id, sizeof(id), endpoint == OPENAI_ENDPOINT_CHAT
                                      ? "chatcmpl-yvex-%012llu"
                                      : "resp_yvex_%012llu",
                   request_ordinal);
    yvex_core_text_copy(admitted.provider->adapter,
                        sizeof(admitted.provider->adapter), "openai");
    yvex_core_text_copy(admitted.provider->external_correlation_id,
                        sizeof(admitted.provider->external_correlation_id), id);
    admitted.provider->sealed = 0;
    admitted.provider->request_identity[0] = '\0';
    rc = yvex_provider_request_seal(admitted.provider, &err);
    if (rc == YVEX_OK && stateful) {
        if (!gateway->state_mutex_ready ||
            pthread_mutex_lock(&gateway->state_mutex) != 0) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(&err, rc, "server.openai.state",
                           "response-state lock is unavailable");
        } else {
            state_locked = 1;
            rc = state_prepare(gateway, now, 0, &err);
        }
    }
    if (rc == YVEX_OK && admitted.provider->previous_response_id[0]) {
        prior = openai_state_find(gateway,
                                  admitted.provider->previous_response_id, now);
        if (!prior) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(&err, rc, "server.openai.previous_response",
                           "previous_response_id is unknown or expired");
        } else if (strcmp(prior->model, engine.alias) != 0) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(&err, rc, "server.openai.previous_response",
                           "previous response belongs to another model");
        } else if (prior->engine_generation != engine.generation) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(&err, rc, "server.openai.previous_response",
                           "previous response belongs to a stale engine generation");
        }
    }
    if (rc == YVEX_OK)
        rc = context_combine(prior ? prior->context : NULL,
                             admitted.provider, &combined, &err);
    if (rc != YVEX_OK) goto failure;
    if (stateful && !prior) {
        rc = state_prepare(gateway, now, 1, &err);
        if (rc != YVEX_OK) goto failure;
    }
    sink.stream = combined->stream;
    if (prior) {
        yvex_core_text_copy(session, sizeof(session), prior->session_name);
    } else {
        (void)snprintf(session, sizeof(session), "oa-%012llu",
                       request_ordinal);
        rc = client_connect(gateway, &session_client, &err);
        if (rc == YVEX_OK)
            rc = session_operation(session_client, YVEX_CLIENT_OP_SESSION_NEW,
                                   engine.alias, engine.generation, session,
                                   &err);
        yvex_client_close(&session_client);
        if (rc != YVEX_OK) goto failure;
        created_session = 1;
    }
    yvex_core_text_copy(result.session_name, sizeof(result.session_name), session);
    rc = disconnect_watch_open(&watch, gateway, fd, engine.alias,
                               engine.generation, session, &err);
    if (rc != YVEX_OK) goto failure;
    generation_started = 1;
    rc = generation_execute(gateway, &sink, id, &engine, now, session,
                            combined, &result, &err);
    peer_closed = disconnect_watch_close(&watch);
    if (rc == YVEX_OK && peer_closed) {
        rc = YVEX_ERR_CANCELLED;
        result.failure_class = YVEX_CLIENT_FAILURE_CLIENT_CANCELLED;
        yvex_error_set(&err, rc, "server.openai.disconnect",
                       "HTTP client disconnected during generation");
    }
    if (rc != YVEX_OK) goto failure;
    if (stateful) {
        rc = context_complete(combined, &result, &context, &err);
        if (rc == YVEX_OK && prior) {
            rc = openai_state_replace(gateway, prior, id, engine.generation,
                                      context, now, &err);
            if (rc == YVEX_OK) retained = prior;
        } else if (rc == YVEX_OK) {
            rc = openai_state_store(gateway, id, session, engine.generation,
                                    context, now, &err);
            if (rc == YVEX_OK) retained = openai_state_find(gateway, id, now);
        }
        if (rc != YVEX_OK) goto failure;
    }
    if (!stateful && created_session) {
        rc = session_close(gateway, engine.alias, engine.generation, session,
                           &err);
        if (rc != YVEX_OK) goto failure;
        created_session = 0;
    }
    if (sink.stream && endpoint == OPENAI_ENDPOINT_RESPONSES) {
        rc = response_stream_complete(&sink, id, engine.alias, now,
                                      &result, &err);
        if (rc != YVEX_OK) goto failure;
    } else if (sink.stream) {
        rc = chat_stream_complete(&sink, id, engine.alias, now, &result,
                                  combined->include_usage, &err);
        if (rc != YVEX_OK) goto failure;
    } else if (!sink.stream) {
        rc = openai_json_result(endpoint, id, engine.alias, now,
                                &result, &json, &json_count, &err);
        if (rc == YVEX_OK)
            rc = openai_http_json(fd, 200, json, json_count, &err);
        if (rc != YVEX_OK) goto failure;
    }
    free(json);
    yvex_provider_request_close(&context);
    yvex_provider_request_close(&combined);
    openai_admitted_request_clear(&admitted);
    result_clear(&result);
    if (state_locked) (void)pthread_mutex_unlock(&gateway->state_mutex);
    return rc;
failure:
    peer_closed = disconnect_watch_close(&watch) || peer_closed;
    failure_error = err;
    if (generation_started && stateful && prior && prior->occupied) {
        openai_state_remove(prior);
        retained = NULL;
    } else if (retained && retained->occupied) {
        openai_state_remove(retained);
        retained = NULL;
    }
    if (created_session || (generation_started && stateful && session[0])) {
        cancel_session(gateway, engine.alias, engine.generation, session);
        (void)session_close(gateway, engine.alias, engine.generation, session,
                            &err);
    }
    if (!sink.headers_sent)
        (void)send_error(fd, http_status(rc, result.failure_class),
                         yvex_error_message(&failure_error));
    else if (endpoint == OPENAI_ENDPOINT_RESPONSES) {
        yvex_error stream_error;
        (void)response_event_emit(
            &sink, OPENAI_RESPONSE_EVENT_FAILED, id, engine.alias, now,
            NULL, &result, 0u, &stream_error);
    } else {
        yvex_error stream_error;
        unsigned char *stream_json = NULL;
        unsigned long long stream_count = 0u;
        if (openai_json_error(
                500, "internal_error", NULL, "stream_failed",
                "YVEX generation failed", &stream_json, &stream_count,
                &stream_error) == YVEX_OK)
            (void)openai_http_sse_event(fd, "error", stream_json,
                                        stream_count, &stream_error);
        free(stream_json);
    }
    free(json);
    yvex_provider_request_close(&context);
    yvex_provider_request_close(&combined);
    openai_admitted_request_clear(&admitted);
    result_clear(&result);
    if (state_locked) (void)pthread_mutex_unlock(&gateway->state_mutex);
    return rc;
}
/*
 * Admit and route one accepted loopback connection.
 *
 * Sends one bounded error when possible and always returns connection ownership. One request per
 * connection; no keep-alive or runtime ownership.
 */
static int handle_connection(openai_gateway *gateway, int fd)
{
    struct timeval timeout = {30, 0};
    openai_http_request request;
    openai_endpoint endpoint;
    char model[YVEX_PROVIDER_MODEL_CAP];
    yvex_error err;
    int rc;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    rc = openai_http_read(fd, &request, &err);
    if (rc != YVEX_OK) {
        (void)send_error(fd, http_status(rc, YVEX_CLIENT_FAILURE_NONE),
                         yvex_error_message(&err));
        return rc;
    }
    if (!route(&request, &endpoint, model)) {
        (void)send_error(fd, 404, "endpoint is outside the YVEX OpenAI profile");
        rc = YVEX_ERR_UNSUPPORTED;
    } else if (endpoint <= OPENAI_ENDPOINT_MODEL)
        rc = handle_read(gateway, fd, endpoint, model);
    else
        rc = handle_generation(gateway, fd, &request, endpoint);
    openai_http_request_clear(&request);
    return rc;
}

static void connection_release(server_openai_listener *listener)
{
    if (!listener || !listener->connection_mutex_ready ||
        pthread_mutex_lock(&listener->connection_mutex) != 0)
        return;
    if (listener->active_connections) listener->active_connections--;
    if (listener->connection_idle_ready)
        (void)pthread_cond_broadcast(&listener->connection_idle);
    (void)pthread_mutex_unlock(&listener->connection_mutex);
}

static void *connection_main(void *opaque)
{
    openai_connection *connection = opaque;
    server_openai_listener *listener = connection->listener;
    yvex_server_telemetry_openai_request(listener->gateway.telemetry,
                                         1, 0, 0, 0);
    int rc = handle_connection(&listener->gateway, connection->fd);
    yvex_server_telemetry_openai_request(
        listener->gateway.telemetry, -1, rc == YVEX_OK,
        rc != YVEX_OK && rc != YVEX_ERR_CANCELLED,
        rc == YVEX_ERR_CANCELLED);
    (void)close(connection->fd);
    connection_release(listener);
    free(connection);
    return NULL;
}

static int connection_start(server_openai_listener *listener, int fd,
                            yvex_error *err)
{
    openai_connection *connection = NULL;
    pthread_attr_t attributes;
    pthread_t thread;
    int attributes_ready = 0, rc = YVEX_OK;
    if (!listener || fd < 0 || !listener->maximum_connections ||
        !listener->connection_mutex_ready ||
        pthread_mutex_lock(&listener->connection_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.openai.connection",
                       "bounded connection ownership is required");
        return YVEX_ERR_INVALID_ARG;
    }
    while (listener->active_connections >= listener->maximum_connections &&
           !atomic_load_explicit(&listener->stop, memory_order_acquire))
        (void)pthread_cond_wait(&listener->connection_idle,
                                &listener->connection_mutex);
    if (atomic_load_explicit(&listener->stop, memory_order_acquire))
        rc = YVEX_ERR_CANCELLED;
    else {
        connection = calloc(1u, sizeof(*connection));
        if (!connection) rc = YVEX_ERR_NOMEM;
    }
    if (rc == YVEX_OK) {
        connection->listener = listener;
        connection->fd = fd;
        listener->active_connections++;
    }
    (void)pthread_mutex_unlock(&listener->connection_mutex);
    if (rc == YVEX_OK && pthread_attr_init(&attributes) == 0) {
        attributes_ready = 1;
        if (pthread_attr_setdetachstate(&attributes,
                                        PTHREAD_CREATE_DETACHED) != 0)
            rc = YVEX_ERR_STATE;
    } else if (rc == YVEX_OK) {
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK &&
        pthread_create(&thread, &attributes, connection_main, connection) != 0)
        rc = YVEX_ERR_STATE;
    if (attributes_ready) (void)pthread_attr_destroy(&attributes);
    if (rc != YVEX_OK) {
        if (connection) connection_release(listener);
        free(connection);
        yvex_error_set(
            err, (yvex_status)rc, "server.openai.connection",
            rc == YVEX_ERR_CANCELLED
                ? "OpenAI listener stopped before connection admission"
                : "OpenAI connection worker could not start");
        return rc;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static void connections_finish(server_openai_listener *listener)
{
    if (!listener || !listener->connection_mutex_ready ||
        pthread_mutex_lock(&listener->connection_mutex) != 0)
        return;
    while (listener->active_connections)
        (void)pthread_cond_wait(&listener->connection_idle,
                                &listener->connection_mutex);
    (void)pthread_mutex_unlock(&listener->connection_mutex);
}
/*
 * Serve one already-reserved loopback listener until its server owner requests stop.
 *
 * Prepared listener ownership. Accepted connections execute concurrently only up to the runtime's
 * admitted session width; each still reaches the canonical server scheduler over local protocol.
 */
static void *listener_main(void *opaque)
{
    server_openai_listener *listener = opaque;
    openai_gateway *gateway = &listener->gateway;
    struct timespec delay = {0, 1000000L};
    int rc = YVEX_OK, listener_failed = 0;
    while (!atomic_load_explicit(&listener->admit, memory_order_acquire) &&
           !atomic_load_explicit(&listener->stop, memory_order_acquire))
        (void)nanosleep(&delay, NULL);
    while (!atomic_load_explicit(&listener->stop, memory_order_acquire)) {
        fd_set readable;
        struct timeval wait = {1, 0};
        int ready;
        FD_ZERO(&readable);
        FD_SET(listener->listen_fd, &readable);
        ready = select(listener->listen_fd + 1, &readable, NULL, NULL, &wait);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) {
            if (atomic_load_explicit(&listener->stop, memory_order_acquire))
                break;
            rc = YVEX_ERR_IO;
            listener_failed = 1;
            break;
        }
        if (ready > 0) {
            struct sockaddr_in peer;
            socklen_t peer_count = sizeof(peer);
            int client = accept(listener->listen_fd,
                                (struct sockaddr *)&peer, &peer_count);
            if (client < 0 && errno == EINTR) continue;
            if (client < 0) {
                if (atomic_load_explicit(&listener->stop,
                                         memory_order_acquire))
                    break;
                rc = YVEX_ERR_IO;
                listener_failed = 1;
                break;
            }
            if (peer.sin_addr.s_addr == htonl(INADDR_LOOPBACK)) {
                yvex_error connection_error;
                int request_rc = connection_start(listener, client,
                                                  &connection_error);
                if (request_rc == YVEX_OK) client = -1;
                else if (request_rc != YVEX_ERR_CANCELLED)
                    (void)send_error(client, 503,
                                     yvex_error_message(&connection_error));
            }
            if (client >= 0) (void)close(client);
        }
    }
    if (listener->listen_fd >= 0) {
        (void)close(listener->listen_fd);
        listener->listen_fd = -1;
    }
    connections_finish(listener);
    {
        yvex_error cleanup_error;
        int cleanup_rc = state_sessions_close(gateway, &cleanup_error);
        if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
            rc = cleanup_rc;
            listener->thread_error = cleanup_error;
        }
    }
    openai_state_clear(gateway);
    if (rc != YVEX_OK && listener_failed)
        yvex_error_set(&listener->thread_error, rc,
                       "server.openai.listener", "loopback listener failed");
    listener->thread_status = rc;
    atomic_store_explicit(&listener->ready, 0, memory_order_release);
    return NULL;
}

int yvex_server_openai_prepare(server_openai_listener **out,
                               const server_openai_options *options,
                               server_telemetry *telemetry,
                               yvex_error *err)
{
    struct sockaddr_in address;
    server_openai_listener *listener;
    int one = 1;
    if (out) *out = NULL;
    if (!out || !options || !options->yvex_socket ||
        options->yvex_socket[0] != '/' || !options->port ||
        options->timeout_ms < 100u || options->timeout_ms > 86400000u ||
        !options->maximum_connections || options->maximum_connections >= 64ull ||
        strlen(options->yvex_socket) >= YVEX_SERVER_SOCKET_PATH_CAP) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.openai.config",
                       "loopback port, timeout, and YVEX socket are required");
        return YVEX_ERR_INVALID_ARG;
    }
    listener = calloc(1u, sizeof(*listener));
    if (!listener) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "server.openai.prepare",
                       "OpenAI listener allocation failed");
        return YVEX_ERR_NOMEM;
    }
    listener->maximum_connections = options->maximum_connections;
    atomic_init(&listener->gateway.next_id, 0ull);
    if (pthread_mutex_init(&listener->gateway.state_mutex, NULL) != 0)
        goto state_failure;
    listener->gateway.state_mutex_ready = 1;
    if (pthread_mutex_init(&listener->connection_mutex, NULL) != 0)
        goto state_failure;
    listener->connection_mutex_ready = 1;
    if (pthread_cond_init(&listener->connection_idle, NULL) != 0)
        goto state_failure;
    listener->connection_idle_ready = 1;
    listener->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listener->listen_fd < 0) goto io_failure;
    (void)setsockopt(listener->listen_fd, SOL_SOCKET, SO_REUSEADDR,
                     &one, sizeof(one));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(options->port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listener->listen_fd, (struct sockaddr *)&address,
             sizeof(address)) != 0 || listen(listener->listen_fd, 16) != 0)
        goto io_failure;
    strcpy(listener->gateway.host, "127.0.0.1");
    listener->gateway.port = options->port;
    listener->gateway.yvex_timeout_ms = options->timeout_ms;
    listener->gateway.telemetry = telemetry;
    strcpy(listener->gateway.yvex_socket, options->yvex_socket);
    atomic_init(&listener->stop, 0);
    atomic_init(&listener->admit, 0);
    atomic_init(&listener->ready, 0);
    listener->gateway.stop = &listener->stop;
    *out = listener;
    yvex_error_clear(err);
    return YVEX_OK;
io_failure:
    if (listener->listen_fd >= 0) (void)close(listener->listen_fd);
state_failure:
    if (listener->connection_idle_ready)
        (void)pthread_cond_destroy(&listener->connection_idle);
    if (listener->connection_mutex_ready)
        (void)pthread_mutex_destroy(&listener->connection_mutex);
    if (listener->gateway.state_mutex_ready)
        (void)pthread_mutex_destroy(&listener->gateway.state_mutex);
    free(listener);
    yvex_error_set(err, YVEX_ERR_IO, "server.openai.listener",
                   "loopback listener reservation failed");
    return YVEX_ERR_IO;
}

int yvex_server_openai_start(server_openai_listener *listener,
                             yvex_error *err)
{
    if (!listener || listener->listen_fd < 0 || listener->thread_started) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.openai.start",
                       "one prepared OpenAI listener is required");
        return YVEX_ERR_STATE;
    }
    if (pthread_create(&listener->thread, NULL, listener_main, listener) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.openai.start",
                       "OpenAI listener thread creation failed");
        return YVEX_ERR_STATE;
    }
    listener->thread_started = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Admit HTTP accepts only after the runtime-ready transaction has published. */
void yvex_server_openai_activate(server_openai_listener *listener)
{
    if (!listener || !listener->thread_started) return;
    atomic_store_explicit(&listener->ready, 1, memory_order_release);
    atomic_store_explicit(&listener->admit, 1, memory_order_release);
}

void yvex_server_openai_request_stop(server_openai_listener *listener)
{
    if (!listener) return;
    atomic_store_explicit(&listener->stop, 1, memory_order_release);
    if (listener->connection_mutex_ready &&
        pthread_mutex_lock(&listener->connection_mutex) == 0) {
        if (listener->connection_idle_ready)
            (void)pthread_cond_broadcast(&listener->connection_idle);
        (void)pthread_mutex_unlock(&listener->connection_mutex);
    }
    if (listener->listen_fd >= 0)
        (void)shutdown(listener->listen_fd, SHUT_RDWR);
}

int yvex_server_openai_finish(server_openai_listener *listener,
                              yvex_error *err)
{
    int rc;
    if (!listener) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    yvex_server_openai_request_stop(listener);
    if (!listener->thread_started) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = pthread_join(listener->thread, NULL) == 0
             ? listener->thread_status : YVEX_ERR_STATE;
    listener->thread_started = 0;
    if (rc != YVEX_OK) {
        if (yvex_error_code(&listener->thread_error) == YVEX_OK)
            yvex_error_set(&listener->thread_error, rc,
                           "server.openai.finish",
                           "OpenAI listener thread join failed");
        if (err) *err = listener->thread_error;
        return rc;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_server_openai_snapshot(const server_openai_listener *listener,
                                 server_openai_snapshot *snapshot)
{
    if (!snapshot) return;
    memset(snapshot, 0, sizeof(*snapshot));
    if (!listener) return;
    snapshot->enabled = 1;
    snapshot->ready = atomic_load_explicit(&listener->ready,
                                           memory_order_acquire);
    snapshot->port = listener->gateway.port;
}

/*
 * Release the adapter after its thread and retained protocol state have stopped.
 *
 * Finish failure is cleanup evidence but ownership is still reclaimed.
 */
void yvex_server_openai_close(server_openai_listener **listener)
{
    server_openai_listener *owner;
    yvex_error err;
    if (!listener || !*listener) return;
    owner = *listener;
    (void)yvex_server_openai_finish(owner, &err);
    if (owner->listen_fd >= 0) (void)close(owner->listen_fd);
    openai_state_clear(&owner->gateway);
    if (owner->connection_idle_ready)
        (void)pthread_cond_destroy(&owner->connection_idle);
    if (owner->connection_mutex_ready)
        (void)pthread_mutex_destroy(&owner->connection_mutex);
    if (owner->gateway.state_mutex_ready)
        (void)pthread_mutex_destroy(&owner->gateway.state_mutex);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *listener = NULL;
}
