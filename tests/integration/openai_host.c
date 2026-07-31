/* Test-only fake YVEX protocol host for OpenAI adapter integration.
 * Purpose: provide deterministic status, session, text, JSON, and tool-call protocol-v4 facts.
 * Inputs: one private Unix socket path. Effects: serves bounded local requests until signalled.
 * Failure: exits nonzero on socket/protocol errors. Boundary: never enters production objects. */

#include "src/server/private.h"

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stopped;
static unsigned long long sessions_created, sessions_closed;
static int listener_fd = -1;

static int serve_connection(int fd, yvex_error *err);

/* Purpose: request deterministic fake-host shutdown. */
static void stop_handler(int signal_number)
{
    (void)signal_number;
    stopped = 1;
}

/* Purpose: populate one successful protocol message with common correlation facts. */
static void message_base(yvex_client_message *message,
                         yvex_client_message_kind kind,
                         const yvex_client_request *request)
{
    memset(message, 0, sizeof(*message));
    message->schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message->kind = kind;
    message->status = YVEX_OK;
    message->request_number = request ? request->request_number : 0u;
    if (request) {
        strcpy(message->session_name, request->session_name);
        if (request->provider_request) {
            strcpy(message->provider_request_identity,
                   request->provider_request->request_identity);
            strcpy(message->external_correlation_id,
                   request->provider_request->external_correlation_id);
        }
    }
}

/* Purpose: send one successful handshake acknowledgement. */
static int send_ack(int fd, const yvex_client_request *request,
                    yvex_error *err)
{
    yvex_client_message message;
    message_base(&message, YVEX_CLIENT_MESSAGE_ACK, request);
    strcpy(message.reason, "protocol-v4");
    return yvex_server_protocol_send(fd, &message, err);
}

/* Purpose: send an authoritative ready status without engine ownership. */
static int send_status(int fd, const yvex_client_request *request,
                       yvex_error *err)
{
    yvex_client_message message;
    message_base(&message, YVEX_CLIENT_MESSAGE_STATUS, request);
    message.runtime.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.runtime.status = YVEX_SERVER_STATUS_READY;
    message.runtime.runtime_ready = 1;
    message.runtime.generation_ready = 1;
    strcpy(message.runtime.target_id, "deepseek-v4-flash");
    memset(message.runtime.runtime_model_identity, 'a', 64u);
    message.runtime.runtime_model_identity[64] = '\0';
    memset(message.runtime.runtime_binding_identity, 'b', 64u);
    message.runtime.runtime_binding_identity[64] = '\0';
    memset(message.runtime.artifact_identity, 'c', 64u);
    message.runtime.artifact_identity[64] = '\0';
    memset(message.runtime.physical_variant_identity, 'd', 64u);
    message.runtime.physical_variant_identity[64] = '\0';
    message.runtime.context_capacity = 4096u;
    message.runtime.metrics.model_open_count = 1u;
    message.runtime.metrics.artifact_open_count = 1u;
    message.runtime.metrics.materialization_count = 1u;
    return yvex_server_protocol_send(fd, &message, err);
}

/* Purpose: send one deterministic provider fragment from already-committed fake state. */
static int send_fragment(int fd, const yvex_client_request *request,
                         yvex_provider_output_kind kind,
                         const char *bytes, const char *call_id,
                         const char *tool_name, yvex_error *err)
{
    yvex_client_message message;
    size_t count = strlen(bytes);
    message_base(&message, YVEX_CLIENT_MESSAGE_FRAGMENT, request);
    message.provider_output_kind = kind;
    message.byte_count = count;
    memcpy(message.bytes, bytes, count);
    if (call_id) strcpy(message.tool_call_id, call_id);
    if (tool_name) strcpy(message.tool_name, tool_name);
    return yvex_server_protocol_send(fd, &message, err);
}

/* Purpose: find one ASCII proof marker inside typed provider message bytes. */
static int request_contains(const yvex_provider_request *request,
                            const char *marker)
{
    unsigned long long message, offset;
    size_t count = strlen(marker);
    if (!request || !count) return 0;
    for (message = 0u; message < request->message_count; ++message) {
        yvex_provider_span content = request->messages[message].content;
        if (content.count < count) continue;
        for (offset = 0u; offset <= content.count - count; ++offset)
            if (memcmp(content.bytes + offset, marker, count) == 0) return 1;
    }
    return 0;
}

/* Purpose: send one exact fake generation turn for text, JSON, or tool-loop pressure. */
static int send_generation(int fd, const yvex_client_request *request,
                           yvex_error *err)
{
    const yvex_provider_request *provider = request->provider_request;
    yvex_client_message message;
    unsigned long long index;
    int has_tool_result = 0, rc;

    if (request_contains(provider, "QUEUE_FULL")) {
        message_base(&message, YVEX_CLIENT_MESSAGE_ERROR, request);
        message.status = YVEX_ERR_BOUNDS;
        message.failure_class = YVEX_CLIENT_FAILURE_QUEUE_FULL;
        strcpy(message.reason, "bounded request queue is full");
        return yvex_server_protocol_send(fd, &message, err);
    }
    if (request_contains(provider, "TIMEOUT")) {
        const struct timespec delay = {0, 700000000L};
        (void)nanosleep(&delay, NULL);
        return YVEX_OK;
    }
    message_base(&message, YVEX_CLIENT_MESSAGE_TURN_STARTED, request);
    rc = yvex_server_protocol_send(fd, &message, err);
    if (rc == YVEX_OK && request_contains(provider, "DISCONNECT")) {
        struct pollfd listener = {.fd = listener_fd, .events = POLLIN};
        int ready;
        do {
            ready = poll(&listener, 1u, 2000);
        } while (ready < 0 && errno == EINTR);
        if (ready > 0 && (listener.revents & POLLIN)) {
            int cancellation = accept(listener_fd, NULL, NULL);
            if (cancellation >= 0) {
                rc = serve_connection(cancellation, err);
                close(cancellation);
            }
        }
        if (rc == YVEX_OK) {
            message_base(&message, YVEX_CLIENT_MESSAGE_ERROR, request);
            message.status = YVEX_ERR_CANCELLED;
            message.failure_class = YVEX_CLIENT_FAILURE_CLIENT_CANCELLED;
            strcpy(message.reason, "HTTP disconnect cancellation admitted");
            rc = yvex_server_protocol_send(fd, &message, err);
        }
        return rc;
    }
    for (index = 0u; provider && index < provider->message_count; ++index)
        if (provider->messages[index].role == YVEX_PROVIDER_ROLE_TOOL)
            has_tool_result = 1;
    if (rc == YVEX_OK && provider && provider->tool_count && !has_tool_result) {
        rc = send_fragment(fd, request, YVEX_PROVIDER_OUTPUT_FUNCTION_CALL,
                           "{\"match_id\":\"m1\"}", "call_fixture_1",
                           provider->tools[0].name, err);
        message.provider_finish = YVEX_PROVIDER_FINISH_TOOL_CALLS;
    } else if (rc == YVEX_OK) {
        const char *text = has_tool_result
                               ? "Match context accepted."
                               : "hello from yvex";
        if (request_contains(provider, "exactly these keys"))
            text = "{\"status\":\"ok\",\"operation_mode\":\"observe\","
                   "\"real_data\":false}";
        else if (request_contains(provider, "Select exactly one"))
            text = "{\"tool\":\"query_match_context\",\"arguments\":{"
                   "\"match_id\":\"proof-match\"},\"reason\":"
                   "\"authoritative match context requested\"}";
        else if (request_contains(provider, "DATA_UNAVAILABLE"))
            text = "DATA_UNAVAILABLE: observe-only state.";
        else if (request_contains(provider, "Enrich notifications"))
            text = "INFO_ONLY: no actionable edge.";
        else if (provider && provider->response_format ==
                                 YVEX_PROVIDER_RESPONSE_JSON_OBJECT)
            text = "{\"ok\":true}";
        if (request_contains(provider, "STREAM_OK")) {
            rc = send_fragment(fd, request,
                               YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
                               "STREAM_", NULL, NULL, err);
            if (rc == YVEX_OK)
                rc = send_fragment(fd, request,
                                   YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
                                   "OK", NULL, NULL, err);
        } else {
            rc = send_fragment(fd, request,
                               YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
                               text, NULL, NULL, err);
        }
        message.provider_finish = YVEX_PROVIDER_FINISH_STOP;
    }
    if (rc != YVEX_OK) return rc;
    message_base(&message, YVEX_CLIENT_MESSAGE_TURN_COMPLETE, request);
    message.provider_finish = provider && provider->tool_count && !has_tool_result
                                  ? YVEX_PROVIDER_FINISH_TOOL_CALLS
                                  : YVEX_PROVIDER_FINISH_STOP;
    message.prompt_tokens = 5u;
    message.completion_tokens = 3u;
    message.total_tokens = 8u;
    message.generated_tokens = 3u;
    message.final_position = 8u;
    memset(message.turn_identity, 'e', 64u);
    message.turn_identity[64] = '\0';
    return yvex_server_protocol_send(fd, &message, err);
}

/* Purpose: serve the one post-handshake request expected on a gateway client connection. */
static int serve_connection(int fd, yvex_error *err)
{
    yvex_client_request request;
    unsigned char *prompt = NULL;
    yvex_provider_request *provider = NULL;
    yvex_client_message message;
    int rc = yvex_server_protocol_receive(fd, &request, &prompt, &provider,
                                          err);
    if (rc != YVEX_OK || request.operation != YVEX_CLIENT_OP_HANDSHAKE)
        goto done;
    rc = send_ack(fd, &request, err);
    if (rc != YVEX_OK) goto done;
    free(prompt);
    prompt = NULL;
    yvex_provider_request_close(&provider);
    rc = yvex_server_protocol_receive(fd, &request, &prompt, &provider, err);
    if (rc != YVEX_OK) goto done;
    request.provider_request = provider;
    if (request.operation == YVEX_CLIENT_OP_RUNTIME_STATUS)
        rc = send_status(fd, &request, err);
    else if (request.operation == YVEX_CLIENT_OP_GENERATION_TURN)
        rc = send_generation(fd, &request, err);
    else {
        if (request.operation == YVEX_CLIENT_OP_SESSION_NEW) {
            sessions_created++;
            fprintf(stderr, "session.new %s\n", request.session_name);
            fflush(stderr);
        } else if (request.operation == YVEX_CLIENT_OP_SESSION_CLOSE) {
            sessions_closed++;
            fprintf(stderr, "session.close %s\n", request.session_name);
            fflush(stderr);
        }
        if (request.operation == YVEX_CLIENT_OP_GENERATION_CANCEL) {
            fprintf(stderr, "generation.cancel %s\n", request.session_name);
            fflush(stderr);
        }
        message_base(&message,
                     request.operation == YVEX_CLIENT_OP_SESSION_LIST
                         ? YVEX_CLIENT_MESSAGE_SESSION_LIST
                         : request.operation == YVEX_CLIENT_OP_GENERATION_CANCEL
                               ? YVEX_CLIENT_MESSAGE_ACK
                               : YVEX_CLIENT_MESSAGE_SESSION,
                     &request);
        rc = yvex_server_protocol_send(fd, &message, err);
    }
done:
    free(prompt);
    yvex_provider_request_close(&provider);
    return rc;
}

/* Purpose: run one private Unix protocol fixture until signalled. */
int main(int argc, char **argv)
{
    struct sockaddr_un address;
    struct sigaction action;
    yvex_error err;
    int listener, rc = 0;
    if (argc != 2 || strlen(argv[1]) >= sizeof(address.sun_path)) return 2;
    memset(&action, 0, sizeof(action));
    action.sa_handler = stop_handler;
    sigemptyset(&action.sa_mask);
    (void)sigaction(SIGTERM, &action, NULL);
    (void)sigaction(SIGINT, &action, NULL);
    listener = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listener < 0) return 1;
    listener_fd = listener;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    strcpy(address.sun_path, argv[1]);
    (void)unlink(argv[1]);
    if (bind(listener, (struct sockaddr *)&address, sizeof(address)) != 0 ||
        chmod(argv[1], 0600) != 0 || listen(listener, 16) != 0) {
        close(listener);
        unlink(argv[1]);
        return 1;
    }
    while (!stopped) {
        int client = accept(listener, NULL, NULL);
        if (client < 0 && errno == EINTR) continue;
        if (client < 0) { rc = 1; break; }
        if (serve_connection(client, &err) != YVEX_OK && !stopped)
            rc = 1;
        close(client);
        if (rc) break;
    }
    close(listener);
    listener_fd = -1;
    fprintf(stderr, "session.summary created=%llu closed=%llu\n",
            sessions_created, sessions_closed);
    unlink(argv[1]);
    return rc;
}
