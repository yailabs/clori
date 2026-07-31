/* Owner: server.openai.core.
 * Owns: loopback adapter lifecycle, endpoint routing, YVEX session orchestration, streaming, and cancellation.
 * Does not own: model/runtime lifecycle, JSON syntax, HTTP parsing, provider prompts, KV, or tool execution.
 * Invariants: every generation uses the local protocol; ephemeral sessions close and retained state is explicit.
 * Boundary: the adapter is server-owned and never links or invokes the inference engine directly.
 * Purpose: compose bounded HTTP/OpenAI translation with the existing local runtime-host protocol.
 * Inputs: loopback configuration, accepted HTTP requests, and a private YVEX socket path.
 * Effects: creates client protocol connections/sessions, streams committed results, and retains bounded response IDs.
 * Failure: cancels abandoned work, closes owned sessions/connections, and never reports false HTTP success. */
#define _POSIX_C_SOURCE 200809L
#include "src/server/openai/private.h"
#include <arpa/inet.h>
#include <errno.h>
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
    char session_name[YVEX_SERVER_SESSION_NAME_CAP];
    atomic_int stop;
    atomic_int peer_closed;
    pthread_t thread;
    int started;
} disconnect_watch;

struct server_openai_listener {
    openai_gateway gateway;
    atomic_int stop, admit, ready;
    pthread_t thread;
    int listen_fd, thread_started, thread_status;
    yvex_error thread_error;
};
/* Purpose: return current wall seconds for public OpenAI object timestamps and state TTL only. */
static unsigned long long wall_seconds(void)
{
    struct timespec now;
    return clock_gettime(CLOCK_REALTIME, &now) == 0
               ? (unsigned long long)now.tv_sec : 0u;
}
/* Purpose: grow one bounded collected output span.
 * Inputs: current owned bytes/count/capacity, one explicit source span, and error output.
 * Effects: reallocates and appends only after the complete extent is admitted.
 * Failure: preserves the prior owner and reports bounds or allocation failure.
 * Boundary: collects committed provider bytes without model or transport ownership. */
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
/* Purpose: release one collected generation result.
 * Inputs: a gateway-owned aggregate result.
 * Effects: frees text/argument storage and clears all public counters.
 * Failure: none; a null or cleared result is accepted.
 * Boundary: never releases daemon session or provider-request ownership. */
static void result_clear(openai_generation_result *result)
{
    if (!result) return;
    free(result->text);
    free(result->arguments);
    memset(result, 0, sizeof(*result));
}
/* Purpose: open one gateway protocol client with a bounded model-response timeout.
 * Inputs: gateway socket, client output, and error output.
 * Effects: authenticates protocol v4 and applies the configured send/receive ceiling.
 * Failure: closes partial client ownership and preserves the typed timeout/refusal.
 * Boundary: bounds gateway waiting without changing daemon generation semantics. */
static int client_connect(const openai_gateway *gateway, yvex_client **client,
                          yvex_error *err)
{
    int rc = yvex_client_connect(client, gateway->yvex_socket, err);
    if (rc == YVEX_OK)
        rc = yvex_client_timeout_set(*client, gateway->yvex_timeout_ms, err);
    if (rc != YVEX_OK) yvex_client_close(client);
    return rc;
}
/* Purpose: connect and obtain one authoritative daemon status snapshot.
 * Inputs: gateway socket configuration and caller-owned summary/error outputs.
 * Effects: opens and closes one protocol client after one status exchange.
 * Failure: publishes no summary when connection, send, or response validation fails.
 * Boundary: reads host authority without opening model or artifact resources. */
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
                          !message.runtime.runtime_ready)) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.openai.runtime",
                       "yvexd is not ready");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK && summary) *summary = message.runtime;
    yvex_client_close(&client);
    return rc;
}
/* Purpose: append prior Responses context and current input without reconstructing hidden state.
 * Inputs: optional retained request, current request, and owned combined output.
 * Effects: clones one complete provider request graph and reseals its identity.
 * Failure: closes partial storage and leaves the output null.
 * Boundary: combines typed conversation facts but never infers KV from response text. */
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
        candidate.tool_choice = prior->tool_choice;
    }
    candidate.sealed = 0;
    candidate.request_identity[0] = '\0';
    rc = yvex_provider_request_seal(&candidate, err);
    if (rc == YVEX_OK)
        rc = yvex_provider_request_clone(&candidate, combined, err);
    free(messages);
    return rc;
}
/* Purpose: append one authoritative assistant output to retained Responses context.
 * Inputs: admitted request and committed assistant text or function-call result.
 * Effects: clones and extends the message graph, then seals a new context identity.
 * Failure: destroys the candidate context and publishes no retained mapping.
 * Boundary: records provider-visible history without owning runtime session state. */
static int context_complete(const yvex_provider_request *request,
                            const openai_generation_result *result,
                            yvex_provider_request **context, yvex_error *err)
{
    yvex_provider_request candidate = *request;
    yvex_provider_message *messages;
    yvex_provider_tool_call call = {0};
    int rc;
    *context = NULL;
    if (request->message_count >= YVEX_PROVIDER_MAX_MESSAGES)
        return YVEX_ERR_BOUNDS;
    messages = calloc((size_t)request->message_count + 1u, sizeof(*messages));
    if (!messages) return YVEX_ERR_NOMEM;
    memcpy(messages, request->messages,
           (size_t)request->message_count * sizeof(*messages));
    messages[request->message_count].role = YVEX_PROVIDER_ROLE_ASSISTANT;
    messages[request->message_count].content.bytes = result->text;
    messages[request->message_count].content.count = result->text_count;
    if (result->has_tool_call) {
        yvex_core_text_copy(call.call_id, sizeof(call.call_id),
                            result->tool_call_id);
        yvex_core_text_copy(call.name, sizeof(call.name), result->tool_name);
        call.arguments_json.bytes = result->arguments;
        call.arguments_json.count = result->arguments_count;
        messages[request->message_count].tool_calls = &call;
        messages[request->message_count].tool_call_count = 1u;
    }
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
/* Purpose: request cancellation through a separate local-protocol connection. */
static void cancel_session(const openai_gateway *gateway,
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
    yvex_core_text_copy(request.session_name, sizeof(request.session_name),
                        session_name);
    if (yvex_client_send(client, &request, &err) == YVEX_OK)
        (void)yvex_client_receive(client, &response, &err);
    yvex_client_close(&client);
}
/* Purpose: observe one HTTP peer while the main gateway thread waits on model output.
 * Inputs: a fully initialized watch with immutable gateway/session facts.
 * Effects: requests typed daemon cancellation exactly once after peer closure.
 * Failure: liveness-probe failure is treated as peer loss; cancellation remains best effort.
 * Boundary: this thread never receives model output or mutates gateway response state. */
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
            cancel_session(watch->gateway, watch->session_name);
            break;
        }
    }
    return NULL;
}
/* Purpose: transfer HTTP-disconnect observation to one bounded watcher.
 * Inputs: cleared watch, gateway, peer descriptor, and exact active session name.
 * Effects: starts at most one joinable observer without changing session state.
 * Failure: reports thread creation and leaves the watch stopped and join-safe.
 * Boundary: generation remains on the gateway owner thread. */
static int disconnect_watch_open(disconnect_watch *watch,
                                 const openai_gateway *gateway, int http_fd,
                                 const char *session_name, yvex_error *err)
{
    if (!watch || !gateway || http_fd < 0 || !session_name || !session_name[0])
        return YVEX_ERR_INVALID_ARG;
    memset(watch, 0, sizeof(*watch));
    watch->gateway = gateway;
    watch->http_fd = http_fd;
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
/* Purpose: stop and join one HTTP-disconnect watcher before session cleanup.
 * Inputs: a possibly unopened watch. Effects: publishes whether peer closure was observed.
 * Failure: join failure is secondary to process ownership and reports peer state conservatively.
 * Boundary: never cancels a request after returning. */
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
/* Purpose: create or close one server-owned session over the current gateway connection. */
static int session_operation(yvex_client *client, yvex_client_operation operation,
                             const char *session_name, yvex_error *err)
{
    yvex_client_request request = {0};
    yvex_client_message response;
    int rc;
    request.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request.operation = operation;
    request.request_number = 1u;
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
/* Purpose: close one retained or ephemeral server session through protocol v4.
 * Inputs: gateway socket, exact session name, and error output.
 * Effects: opens one bounded client connection and transfers session cleanup to yvexd.
 * Failure: leaves the caller's mapping intact so cleanup can be retried or reported.
 * Boundary: closes session/KV state only; the daemon model remains open. */
static int session_close(openai_gateway *gateway, const char *session_name,
                         yvex_error *err)
{
    yvex_client *client = NULL;
    int rc = client_connect(gateway, &client, err);
    if (rc == YVEX_OK)
        rc = session_operation(client, YVEX_CLIENT_OP_SESSION_CLOSE,
                               session_name, err);
    yvex_client_close(&client);
    return rc;
}
/* Purpose: reclaim expired state and, when required, one deterministic LRU slot.
 * Inputs: gateway state, current wall seconds, free-slot requirement, and error output.
 * Effects: closes each selected daemon session before removing its application mapping.
 * Failure: preserves the first mapping whose daemon session cannot be closed.
 * Boundary: eviction never leaves an intentionally retained session unreachable. */
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
            int rc = session_close(gateway, record->session_name, err);
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
    if (session_close(gateway, oldest->session_name, err) != YVEX_OK)
        return yvex_error_code(err);
    openai_state_remove(oldest);
    return YVEX_OK;
}
/* Purpose: close every retained daemon session during graceful gateway shutdown.
 * Inputs: gateway state and error output.
 * Effects: closes and removes mappings in stable directory order.
 * Failure: returns the first cleanup failure while continuing other rows.
 * Boundary: process-local mappings never survive shutdown. */
static int state_sessions_close(openai_gateway *gateway, yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    for (index = 0u; index < OPENAI_RESPONSE_RECORD_MAX; ++index) {
        openai_response_record *record = &gateway->records[index];
        if (record->occupied) {
            yvex_error local;
            int close_rc = session_close(gateway, record->session_name,
                                         &local);
            if (close_rc != YVEX_OK && rc == YVEX_OK) {
                rc = close_rc;
                *err = local;
            }
            openai_state_remove(record);
        }
    }
    return rc;
}
/* Purpose: map one Responses event kind to its exact SSE event name. */
static const char *response_sse_name(openai_response_event_kind kind)
{
    static const char *const names[] = {
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
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
/* Purpose: render and publish one sequenced Responses event transactionally.
 * Inputs: stream sink, event kind, public facts, optional message/result, and error output.
 * Effects: advances sequence only after one complete SSE record is written.
 * Failure: frees candidate JSON and leaves the prior sequence authoritative.
 * Boundary: event projection only; model/session state is already committed. */
static int response_event_emit(openai_http_sink *sink,
                               openai_response_event_kind kind,
                               const char *id, const char *model,
                               unsigned long long created,
                               const yvex_client_message *message,
                               const openai_generation_result *result,
                               yvex_error *err)
{
    unsigned char *json = NULL;
    unsigned long long count = 0u;
    int rc = openai_json_response_event(
        kind, id, model, created, message, result,
        sink->response_sequence, &json, &count, err);
    if (rc == YVEX_OK)
        rc = openai_http_sse_event(sink->fd, response_sse_name(kind),
                                   json, count, err);
    if (rc == YVEX_OK) sink->response_sequence++;
    free(json);
    return rc;
}
/* Purpose: emit the complete Responses item terminal sequence.
 * Inputs: sink/IDs/time, terminal message, completed aggregate result, and error output.
 * Effects: seals text or arguments, item, and response in canonical order.
 * Failure: stops at the first incomplete socket event without claiming completion.
 * Boundary: all bytes and usage facts originate in protocol-v4 messages. */
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
    int rc;
    if (result->has_tool_call) {
        rc = response_event_emit(
            sink, OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DONE,
            id, model, created, message, result, err);
    } else {
        rc = response_event_emit(sink, OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DONE,
                                 id, model, created, message, result, err);
        if (rc == YVEX_OK)
            rc = response_event_emit(
                sink, OPENAI_RESPONSE_EVENT_CONTENT_PART_DONE,
                id, model, created, message, result, err);
    }
    if (rc == YVEX_OK)
        rc = response_event_emit(sink, OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_DONE,
                                 id, model, created, message, result, err);
    if (rc == YVEX_OK)
        rc = response_event_emit(sink, terminal, id, model, created,
                                 message, result, err);
    return rc;
}
/* Purpose: publish the terminal Responses sequence only after retained state is committed.
 * Inputs: stream sink, public IDs/time, completed aggregate result, and error output.
 * Effects: creates an empty output item when no fragment existed, then emits terminal events.
 * Failure: never emits response.completed before the state mapping is authoritative.
 * Boundary: provider publication follows gateway/session state commit. */
static int response_stream_complete(openai_http_sink *sink, const char *id,
                                    const char *model,
                                    unsigned long long created,
                                    const openai_generation_result *result,
                                    yvex_error *err)
{
    int rc = YVEX_OK;
    if (!sink->response_item_started) {
        rc = response_event_emit(sink, OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_ADDED,
                                 id, model, created, NULL, result, err);
        if (rc == YVEX_OK && !result->has_tool_call)
            rc = response_event_emit(
                sink, OPENAI_RESPONSE_EVENT_CONTENT_PART_ADDED,
                id, model, created, NULL, result, err);
        if (rc == YVEX_OK) sink->response_item_started = 1;
    }
    if (rc == YVEX_OK)
        rc = response_terminal_emit(sink, id, model, created, NULL, result,
                                    err);
    return rc;
}
/* Purpose: stream one admitted protocol fragment/event and retain exact reconstructed bytes.
 * Inputs: sink/profile IDs, one protocol message, aggregate result, and error output.
 * Effects: appends committed bytes/counters and may emit one valid SSE record.
 * Failure: preserves prior collected output and returns the first render/write error.
 * Boundary: translates committed protocol facts; it never invents model output. */
static int generation_message(openai_http_sink *sink, const char *id,
                              const char *model, unsigned long long created,
                              const yvex_client_message *message,
                              openai_generation_result *result,
                              yvex_error *err)
{
    unsigned char *json = NULL;
    unsigned long long count = 0u;
    int rc = YVEX_OK;
    if (message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
        if (message->provider_output_kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL) {
            result->has_tool_call = 1;
            yvex_core_text_copy(result->tool_call_id,
                                sizeof(result->tool_call_id),
                                message->tool_call_id);
            yvex_core_text_copy(result->tool_name, sizeof(result->tool_name),
                                message->tool_name);
            rc = result_append(&result->arguments, &result->arguments_count,
                               &result->arguments_capacity, message->bytes,
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
        result->finish = message->provider_finish;
        yvex_core_text_copy(result->turn_identity,
                            sizeof(result->turn_identity),
                            message->turn_identity);
        result->complete = 1;
    }
    if (rc == YVEX_OK && sink->stream &&
        sink->endpoint == OPENAI_ENDPOINT_RESPONSES) {
        if (message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
            if (!sink->response_item_started) {
                rc = response_event_emit(
                    sink, OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_ADDED,
                    id, model, created, message, result, err);
                if (rc == YVEX_OK && !result->has_tool_call)
                    rc = response_event_emit(
                        sink, OPENAI_RESPONSE_EVENT_CONTENT_PART_ADDED,
                        id, model, created, message, result, err);
                if (rc == YVEX_OK) sink->response_item_started = 1;
            }
            if (rc == YVEX_OK)
                rc = response_event_emit(
                    sink,
                    result->has_tool_call
                        ? OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DELTA
                        : OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DELTA,
                    id, model, created, message, result, err);
        }
    } else if (rc == YVEX_OK && sink->stream &&
               message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
        rc = openai_json_stream_chunk(sink->endpoint, id, model, created,
                                      message, 0, &json, &count, err);
        if (rc == YVEX_OK)
            rc = openai_http_sse_event(
                sink->fd,
                NULL,
                json, count, err);
        free(json);
    }
    return rc;
}
/* Purpose: execute one provider request through a server-owned session and committed stream.
 * Inputs: gateway socket, provider request, chosen session, public IDs, and sink.
 * Effects: sends one daemon turn, streams typed results, and records terminal facts.
 * Failure: requests cancellation and retains only facts received before failure.
 * Boundary: protocol orchestration only; model, KV, and generation remain in yvexd. */
static int generation_execute(openai_gateway *gateway,
                              openai_http_sink *sink, const char *id,
                              const char *model, unsigned long long created,
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
    request.provider_request = provider;
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
                    sink, OPENAI_RESPONSE_EVENT_CREATED, id, model, created,
                    NULL, result, err);
            else if (rc == YVEX_OK) {
                unsigned char *json = NULL;
                unsigned long long count = 0u;
                rc = openai_json_stream_chunk(sink->endpoint, id, model,
                                              created, NULL, 1, &json,
                                              &count, err);
                if (rc == YVEX_OK)
                    rc = openai_http_sse_event(sink->fd, NULL,
                                               json, count, err);
                free(json);
            }
        } else if (message.kind == YVEX_CLIENT_MESSAGE_FRAGMENT ||
                   message.kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE) {
            rc = generation_message(sink, id, model, created, &message,
                                    result, err);
        }
    }
    yvex_client_close(&client);
    return rc;
}
/* Purpose: close one Chat stream only after ephemeral session cleanup succeeds.
 * Inputs: sink, public IDs/time, completed result, usage policy, and error output.
 * Effects: emits terminal finish, optional usage, then exactly one [DONE] sentinel.
 * Failure: emits no success sentinel after an incomplete terminal publication.
 * Boundary: committed model bytes precede transport completion; session cleanup precedes success. */
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
                                  &terminal, 0, &json, &count, err);
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
/* Purpose: map one typed YVEX/gateway status into the bounded public HTTP error profile.
 * Inputs: lower status and optional authoritative protocol failure class.
 * Effects: none.
 * Failure: unknown status/class combinations become internal error responses.
 * Boundary: classification uses typed facts and never diagnostic prose. */
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
/* Purpose: send one safe public error envelope before any streaming headers exist.
 * Inputs: client descriptor, HTTP status, and bounded public message.
 * Effects: renders and writes exactly one complete JSON error response.
 * Failure: returns rendering or socket failure without reporting false HTTP success.
 * Boundary: hides internal paths and identities and never changes daemon state. */
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
/* Purpose: recognize only the documented method/path compatibility matrix.
 * Inputs: admitted HTTP request and endpoint/model outputs.
 * Effects: writes route facts only when method and path match the profile.
 * Failure: returns false for unsupported or oversized model paths.
 * Boundary: routing contains no request-body or model-execution semantics. */
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
/* Purpose: serve health or model discovery without creating a runtime session.
 * Inputs: gateway configuration, client descriptor, endpoint, and optional model.
 * Effects: queries daemon readiness and writes one bounded JSON response.
 * Failure: maps unavailable daemon/model state to a typed provider error.
 * Boundary: read-only discovery; no session or generation request is created. */
static int handle_read(openai_gateway *gateway, int fd,
                       openai_endpoint endpoint, const char *requested_model)
{
    static const unsigned char healthy[] =
        "{\"status\":\"ok\",\"adapter\":\"ready\",\"yvexd\":\"ready\","
        "\"profile\":\"" OPENAI_COMPAT_PROFILE "\"}";
    yvex_server_summary summary;
    unsigned char *json = NULL;
    unsigned long long count = 0u;
    yvex_error err;
    int rc = daemon_status(gateway, &summary, &err);
    if (rc != YVEX_OK)
        return send_error(fd, 503, "yvexd is unavailable or not ready");
    if (endpoint == OPENAI_ENDPOINT_HEALTH)
        return openai_http_json(fd, 200, healthy, sizeof(healthy) - 1u, &err);
    if (endpoint == OPENAI_ENDPOINT_MODEL &&
        strcmp(requested_model, summary.target_id) != 0)
        return send_error(fd, 404, "requested model is not loaded");
    rc = openai_json_models(&summary, summary.target_id,
                            endpoint == OPENAI_ENDPOINT_MODELS,
                            &json, &count, &err);
    if (rc == YVEX_OK) rc = openai_http_json(fd, 200, json, count, &err);
    free(json);
    return rc;
}
/* Purpose: execute one Chat/Responses HTTP request through exact local-protocol state.
 * Inputs: gateway state, client descriptor, complete HTTP request, and endpoint.
 * Effects: admits provider intent, executes one turn, renders output, and retains state if required.
 * Failure: cancels active work, closes ephemeral state, and emits no false completion.
 * Boundary: owns application translation while yvexd owns inference and KV mutation. */
static int handle_generation(openai_gateway *gateway, int fd,
                             const openai_http_request *http,
                             openai_endpoint endpoint)
{
    yvex_server_summary summary;
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
    char id[YVEX_PROVIDER_ID_CAP], session[YVEX_SERVER_SESSION_NAME_CAP];
    unsigned char *json = NULL;
    unsigned long long json_count = 0u, now = wall_seconds();
    int stateful = endpoint == OPENAI_ENDPOINT_RESPONSES;
    int created_session = 0, generation_started = 0, peer_closed = 0, rc;
    yvex_error err, failure_error;
    rc = daemon_status(gateway, &summary, &err);
    if (rc != YVEX_OK) return send_error(fd, 503, "yvexd is unavailable or not ready");
    rc = openai_json_admit(http, endpoint, summary.target_id, &admitted, &err);
    if (rc != YVEX_OK)
        return send_error(fd, http_status(rc, YVEX_CLIENT_FAILURE_NONE),
                          yvex_error_message(&err));
    (void)snprintf(id, sizeof(id), endpoint == OPENAI_ENDPOINT_CHAT
                                      ? "chatcmpl-yvex-%012llu"
                                      : "resp_yvex_%012llu",
                   ++gateway->next_id);
    yvex_core_text_copy(admitted.provider->adapter,
                        sizeof(admitted.provider->adapter), "openai");
    yvex_core_text_copy(admitted.provider->external_correlation_id,
                        sizeof(admitted.provider->external_correlation_id), id);
    admitted.provider->sealed = 0;
    admitted.provider->request_identity[0] = '\0';
    rc = yvex_provider_request_seal(admitted.provider, &err);
    if (rc == YVEX_OK)
        rc = state_prepare(gateway, now, 0, &err);
    if (rc == YVEX_OK && admitted.provider->previous_response_id[0]) {
        prior = openai_state_find(gateway,
                                  admitted.provider->previous_response_id, now);
        if (!prior) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(&err, rc, "server.openai.previous_response",
                           "previous_response_id is unknown or expired");
        } else if (strcmp(prior->model, admitted.provider->model) != 0) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(&err, rc, "server.openai.previous_response",
                           "previous response belongs to another model");
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
        (void)snprintf(session, sizeof(session), "oa-%012llu", gateway->next_id);
        rc = client_connect(gateway, &session_client, &err);
        if (rc == YVEX_OK)
            rc = session_operation(session_client, YVEX_CLIENT_OP_SESSION_NEW,
                                   session, &err);
        yvex_client_close(&session_client);
        if (rc != YVEX_OK) goto failure;
        created_session = 1;
    }
    yvex_core_text_copy(result.session_name, sizeof(result.session_name), session);
    rc = disconnect_watch_open(&watch, gateway, fd, session, &err);
    if (rc != YVEX_OK) goto failure;
    generation_started = 1;
    rc = generation_execute(gateway, &sink, id, summary.target_id, now,
                            session, combined, &result, &err);
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
            rc = openai_state_replace(gateway, prior, id, context, now, &err);
            if (rc == YVEX_OK) retained = prior;
        } else if (rc == YVEX_OK) {
            rc = openai_state_store(gateway, id, session, context, now, &err);
            if (rc == YVEX_OK) retained = openai_state_find(gateway, id, now);
        }
        if (rc != YVEX_OK) goto failure;
    }
    if (!stateful && created_session) {
        rc = session_close(gateway, session, &err);
        if (rc != YVEX_OK) goto failure;
        created_session = 0;
    }
    if (sink.stream && endpoint == OPENAI_ENDPOINT_RESPONSES) {
        rc = response_stream_complete(&sink, id, summary.target_id, now,
                                      &result, &err);
        if (rc != YVEX_OK) goto failure;
    } else if (sink.stream) {
        rc = chat_stream_complete(&sink, id, summary.target_id, now, &result,
                                  combined->include_usage, &err);
        if (rc != YVEX_OK) goto failure;
    } else if (!sink.stream) {
        rc = openai_json_result(endpoint, id, summary.target_id, now,
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
        cancel_session(gateway, session);
        (void)session_close(gateway, session, &err);
    }
    if (!sink.headers_sent)
        (void)send_error(fd, http_status(rc, result.failure_class),
                         yvex_error_message(&failure_error));
    else if (endpoint == OPENAI_ENDPOINT_RESPONSES) {
        yvex_error stream_error;
        (void)response_event_emit(
            &sink, OPENAI_RESPONSE_EVENT_FAILED, id, summary.target_id, now,
            NULL, &result, &stream_error);
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
    return rc;
}
/* Purpose: admit and route one accepted loopback connection.
 * Inputs: gateway state and one accepted loopback socket.
 * Effects: reads one request, writes one response or stream, then clears request storage.
 * Failure: sends one bounded error when possible and always returns connection ownership.
 * Boundary: one request per connection; no keep-alive or runtime ownership. */
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
/* Purpose: serve one already-reserved loopback listener until its server owner requests stop.
 * Inputs: prepared listener ownership. Effects: accepts serial bounded requests and clears response state.
 * Failure: records the first listener failure for the lifecycle joiner. Boundary: no model ownership. */
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
                int request_rc;
                yvex_server_telemetry_openai_request(gateway->telemetry,
                                                     1, 0, 0, 0);
                request_rc = handle_connection(gateway, client);
                yvex_server_telemetry_openai_request(
                    gateway->telemetry, -1, request_rc == YVEX_OK,
                    request_rc != YVEX_OK && request_rc != YVEX_ERR_CANCELLED,
                    request_rc == YVEX_ERR_CANCELLED);
            }
            (void)close(client);
        }
    }
    if (listener->listen_fd >= 0) {
        (void)close(listener->listen_fd);
        listener->listen_fd = -1;
    }
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

/* Purpose: reserve the loopback HTTP endpoint before any runtime model opens.
 * Inputs: unique output, explicit local protocol path, port, timeout, and shared telemetry.
 * Effects: allocates adapter state and binds/listens without accepting requests.
 * Failure: closes every partial descriptor and reports configuration or port collision.
 * Boundary: preparation owns no model, session, worker, or process entrypoint. */
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
    free(listener);
    yvex_error_set(err, YVEX_ERR_IO, "server.openai.listener",
                   "loopback listener reservation failed");
    return YVEX_ERR_IO;
}

/* Purpose: start the gated listener thread before runtime readiness is published.
 * Inputs: one prepared listener and writable error output.
 * Effects: creates exactly one joinable thread; HTTP admission remains disabled.
 * Failure: reports invalid lifecycle state or thread creation failure without consuming the listener.
 * Boundary: does not admit requests or mutate runtime/model state. */
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

/* Purpose: admit HTTP accepts only after the runtime-ready transaction has published.
 * Inputs: one successfully started listener. Effects: atomically publishes ready and admission.
 * Failure: invalid or unstarted listeners remain unchanged. Boundary: no socket or runtime mutation. */
void yvex_server_openai_activate(server_openai_listener *listener)
{
    if (!listener || !listener->thread_started) return;
    atomic_store_explicit(&listener->ready, 1, memory_order_release);
    atomic_store_explicit(&listener->admit, 1, memory_order_release);
}

/* Purpose: stop HTTP admission and wake the accept owner for bounded shutdown.
 * Inputs: optional prepared listener. Effects: publishes stop and shuts down the listening descriptor.
 * Failure: descriptor shutdown failure is deferred to lifecycle cleanup. Boundary: no model close. */
void yvex_server_openai_request_stop(server_openai_listener *listener)
{
    if (!listener) return;
    atomic_store_explicit(&listener->stop, 1, memory_order_release);
    if (listener->listen_fd >= 0)
        (void)shutdown(listener->listen_fd, SHUT_RDWR);
}

/* Purpose: join the HTTP owner after bounded cancellation and response-state cleanup.
 * Inputs: prepared listener and error output. Effects: joins at most once.
 * Failure: reports thread or listener cleanup failure. Boundary: no runtime owner closes here. */
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

/* Purpose: copy bounded listener lifecycle facts into the authoritative server status.
 * Inputs: optional listener and writable snapshot. Effects: writes one self-contained snapshot.
 * Failure: absent listener produces disabled zero facts. Boundary: reads no request or runtime state. */
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

/* Purpose: release the adapter after its thread and retained protocol state have stopped.
 * Inputs: unique listener-owner slot. Effects: joins, closes, clears state, frees, and nulls the slot.
 * Failure: finish failure is cleanup evidence but ownership is still reclaimed. Boundary: closes no model. */
void yvex_server_openai_close(server_openai_listener **listener)
{
    server_openai_listener *owner;
    yvex_error err;
    if (!listener || !*listener) return;
    owner = *listener;
    (void)yvex_server_openai_finish(owner, &err);
    if (owner->listen_fd >= 0) (void)close(owner->listen_fd);
    openai_state_clear(&owner->gateway);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *listener = NULL;
}
