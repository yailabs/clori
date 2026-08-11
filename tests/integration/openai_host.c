/*
 * Provide deterministic status, session, text, JSON, and tool-call protocol-v9 facts. Never
 * enters production objects.
 */

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

static void stop_handler(int signal_number)
{
    (void)signal_number;
    stopped = 1;
}

static void message_base(yvex_client_message *message,
                         yvex_client_message_kind kind,
                         const yvex_client_request *request)
{
    memset(message, 0, sizeof(*message));
    message->schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message->kind = kind;
    message->status = YVEX_OK;
    if (kind == YVEX_CLIENT_MESSAGE_ERROR)
        message->stream_channel = YVEX_CLIENT_STREAM_ERROR;
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

static int send_ack(int fd, const yvex_client_request *request,
                    yvex_error *err)
{
    yvex_client_message message;
    message_base(&message, YVEX_CLIENT_MESSAGE_ACK, request);
    strcpy(message.reason, "protocol-v9");
    return yvex_server_protocol_send(fd, &message, err);
}

static int send_status(int fd, const yvex_client_request *request,
                       yvex_error *err)
{
    yvex_client_message message;
    message_base(&message, YVEX_CLIENT_MESSAGE_STATUS, request);
    message.runtime.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.runtime.status = YVEX_SERVER_STATUS_READY;
    message.runtime.runtime_ready = 1;
    message.runtime.generation_ready = 1;
    message.runtime.explicit_reasoning_channel_supported = 1;
    message.runtime.concurrent_sequences = 1u;
    message.runtime.capacity_required_bytes = 4096u;
    message.runtime.capacity_unreserved_bytes = 8192u;
    strcpy(message.runtime.target_id, "deepseek4-v4-flash-dspark");
    memset(message.runtime.runtime_model_identity, 'a', 64u);
    message.runtime.runtime_model_identity[64] = '\0';
    memset(message.runtime.runtime_binding_identity, 'b', 64u);
    message.runtime.runtime_binding_identity[64] = '\0';
    memset(message.runtime.artifact_identity, 'c', 64u);
    message.runtime.artifact_identity[64] = '\0';
    memset(message.runtime.physical_variant_identity, 'd', 64u);
    message.runtime.physical_variant_identity[64] = '\0';
    memset(message.runtime.capacity_plan_identity, 'e', 64u);
    message.runtime.capacity_plan_identity[64] = '\0';
    message.runtime.context_capacity = 4096u;
    message.runtime.metrics.model_open_count = 1u;
    message.runtime.metrics.artifact_open_count = 1u;
    message.runtime.metrics.materialization_count = 1u;
    return yvex_server_protocol_send(fd, &message, err);
}

static int send_console_status(int fd, const yvex_client_request *request,
                               yvex_error *err)
{
    yvex_client_message message;
    message_base(&message, YVEX_CLIENT_MESSAGE_CONSOLE_STATUS, request);
    message.runtime.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.runtime.status = YVEX_SERVER_STATUS_READY;
    message.runtime.runtime_ready = 1;
    message.runtime.generation_ready = 1;
    message.runtime.explicit_reasoning_channel_supported = 1;
    message.runtime.concurrent_sequences = 1u;
    message.runtime.capacity_required_bytes = 4096u;
    message.runtime.capacity_unreserved_bytes = 8192u;
    message.runtime.backend = YVEX_BACKEND_KIND_CUDA;
    message.runtime.context_capacity = 4096u;
    strcpy(message.runtime.target_id, "deepseek4-v4-flash-dspark");
    memset(message.runtime.runtime_model_identity, 'a', 64u);
    message.runtime.runtime_model_identity[64] = '\0';
    memset(message.runtime.physical_variant_identity, 'd', 64u);
    message.runtime.physical_variant_identity[64] = '\0';
    memset(message.runtime.capacity_plan_identity, 'e', 64u);
    message.runtime.capacity_plan_identity[64] = '\0';
    message.console.schema_version = 1u;
    message.console.backend = YVEX_BACKEND_KIND_CUDA;
    message.console.session_state = YVEX_SERVER_SESSION_READY;
    message.console.generation_phase = YVEX_CLIENT_PHASE_IDLE;
    message.console.context_capacity = 4096u;
    message.console.runtime_ready = 1;
    message.console.session_available = 1;
    message.console.attached = 1;
    message.console.explicit_reasoning_channel_supported = 1;
    strcpy(message.console.session_name, request->session_name);
    memset(message.console.live_model_identity, 'a', 64u);
    message.console.live_model_identity[64] = '\0';
    memset(message.console.physical_variant_identity, 'd', 64u);
    message.console.physical_variant_identity[64] = '\0';
    return yvex_server_protocol_send(fd, &message, err);
}

static int send_native_progress(int fd, const yvex_client_request *request,
                                size_t event_count, yvex_error *err)
{
    static const yvex_server_event_kind kinds[] = {
        YVEX_SERVER_EVENT_TOKENIZER_COMPLETED,
        YVEX_SERVER_EVENT_PREFILL_STARTED,
        YVEX_SERVER_EVENT_PREFILL_PROGRESS,
        YVEX_SERVER_EVENT_PREFILL_COMPLETED};
    static const unsigned long long values[][2] = {
        {5u, 1u}, {4u, 4u}, {2u, 4u}, {4u, 1u}};
    static const char *const phases[] = {
        "tokenizer", "prefill", "prefill", "prefill"};
    static const char identity[] =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    server_telemetry *telemetry = NULL;
    size_t index;
    int rc = yvex_server_telemetry_open(&telemetry, 8u,
                                        YVEX_SERVER_GENERATION_TARGET_ONLY,
                                        identity, identity,
                                        identity, err);
    for (index = 0u; rc == YVEX_OK && index < event_count; ++index) {
        yvex_client_message message;
        message_base(&message, YVEX_CLIENT_MESSAGE_EVENT, request);
        message.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
        rc = yvex_server_telemetry_emit_provider(
            telemetry, kinds[index], YVEX_SERVER_SEVERITY_INFO,
            request->session_name, "fixture-request", "fixture-turn",
            phases[index], values[index][0], values[index][1], 0u,
            index >= 2u ? (double)(index - 1u) : 0.0,
            index >= 2u ? 2.0 : 0.0,
            NULL, NULL, &message.event, err);
        if (rc == YVEX_OK) rc = yvex_server_protocol_send(fd, &message, err);
    }
    yvex_server_telemetry_close(&telemetry);
    return rc;
}

static int send_event_stream(int fd, const yvex_client_request *request,
                             yvex_error *err)
{
    static const yvex_server_event_kind kinds[] = {
        YVEX_SERVER_EVENT_REQUEST_STARTED,
        YVEX_SERVER_EVENT_GENERATION_PROFILE,
        YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE};
    static const unsigned long long values[][3] = {
        {5u, 1u, 3u}, {4511u, 63u, 0u}, {1u, 1u, 0u}};
    static const char *const phases[] = {"generation", "launches", "runtime"};
    static const char identity[] =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    server_telemetry *telemetry = NULL;
    size_t index;
    int rc = yvex_server_telemetry_open(&telemetry, 8u,
                                        YVEX_SERVER_GENERATION_TARGET_ONLY,
                                        identity, identity,
                                        identity, err);
    for (index = 0u; rc == YVEX_OK && index < sizeof(kinds) / sizeof(kinds[0]); ++index) {
        yvex_client_message message;
        message_base(&message, YVEX_CLIENT_MESSAGE_EVENT, request);
        message.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
        rc = yvex_server_telemetry_emit_provider(
            telemetry, kinds[index], YVEX_SERVER_SEVERITY_INFO,
            index < 2u ? "fixture" : NULL, index < 2u ? "fixture-request" : NULL,
            index < 2u ? "fixture-turn" : NULL, phases[index], values[index][0],
            values[index][1], values[index][2], 0.0, 0.0, NULL, NULL,
            &message.event, err);
        if (rc == YVEX_OK) rc = yvex_server_protocol_send(fd, &message, err);
    }
    yvex_server_telemetry_close(&telemetry);
    return rc;
}

static int send_fragment_bytes(int fd, const yvex_client_request *request,
                               yvex_provider_output_kind kind,
                               yvex_client_stream_channel channel,
                               const unsigned char *bytes, size_t count,
                               const char *call_id, const char *tool_name,
                               yvex_error *err)
{
    yvex_client_message message;
    message_base(&message, YVEX_CLIENT_MESSAGE_FRAGMENT, request);
    message.provider_output_kind = kind;
    message.stream_channel = channel;
    message.byte_count = count;
    memcpy(message.bytes, bytes, count);
    if (call_id) strcpy(message.tool_call_id, call_id);
    if (tool_name) strcpy(message.tool_name, tool_name);
    return yvex_server_protocol_send(fd, &message, err);
}

static int send_fragment(int fd, const yvex_client_request *request,
                         yvex_provider_output_kind kind, const char *bytes,
                         const char *call_id, const char *tool_name,
                         yvex_error *err)
{
    yvex_client_stream_channel channel =
        kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL
            ? YVEX_CLIENT_STREAM_TOOL_CALL
        : kind == YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING
            ? YVEX_CLIENT_STREAM_EXPLICIT_REASONING
            : YVEX_CLIENT_STREAM_FINAL_TEXT;
    return send_fragment_bytes(fd, request, kind, channel,
                               (const unsigned char *)bytes, strlen(bytes),
                               call_id, tool_name, err);
}

static int send_native_markdown(int fd, const yvex_client_request *request,
                                yvex_error *err)
{
    static const unsigned char part_a[] = "# CUDA\n\n``";
    static const unsigned char part_b[] = "`cuda\n__global__ void add() {\n  // ";
    static const unsigned char part_c[] = {0xf0u, 0x9fu};
    static const unsigned char part_d[] = {0x8cu, 0x8du, '\n', '}', '\n', '`', '`'};
    static const unsigned char part_e[] = "`\nUse `int` safely.\nESC: \033[31mnot-control\n";
    const struct {
        const unsigned char *bytes;
        size_t count;
    } parts[] = {{part_a, sizeof(part_a) - 1u},
                 {part_b, sizeof(part_b) - 1u},
                 {part_c, sizeof(part_c)},
                 {part_d, sizeof(part_d)},
                 {part_e, sizeof(part_e) - 1u}};
    size_t index;
    int rc = YVEX_OK;
    for (index = 0u; rc == YVEX_OK && index < sizeof(parts) / sizeof(parts[0]); ++index)
        rc = send_fragment_bytes(fd, request, YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
                                 YVEX_CLIENT_STREAM_FINAL_TEXT, parts[index].bytes,
                                 parts[index].count, NULL, NULL, err);
    return rc;
}

static int send_native_reasoning(int fd, const yvex_client_request *request,
                                 yvex_error *err)
{
    int rc = send_fragment_bytes(
        fd, request, YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING,
        YVEX_CLIENT_STREAM_EXPLICIT_REASONING,
        (const unsigned char *)"I need to compare the constraints...\n",
        strlen("I need to compare the constraints...\n"), NULL, NULL, err);
    if (rc == YVEX_OK)
        rc = send_fragment_bytes(
            fd, request, YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
            YVEX_CLIENT_STREAM_FINAL_TEXT,
            (const unsigned char *)"The valid result is 42.",
            strlen("The valid result is 42."), NULL, NULL, err);
    return rc;
}

static int send_native_partial(int fd, const yvex_client_request *request,
                               yvex_provider_output_kind kind,
                               const char *bytes, const char *reason,
                               unsigned long long committed_tokens,
                               unsigned long long final_position,
                               yvex_error *err)
{
    yvex_client_message message;
    int rc = send_fragment(fd, request, kind, bytes, NULL, NULL, err);
    if (rc != YVEX_OK) return rc;
    message_base(&message, YVEX_CLIENT_MESSAGE_ERROR, request);
    message.status = YVEX_ERR_BACKEND;
    message.failure_class = YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE;
    message.session_state = YVEX_SERVER_SESSION_PARTIAL;
    strcpy(message.reason, reason);
    message.partial_turn.schema_version = YVEX_CLIENT_PARTIAL_TURN_SCHEMA_V1;
    message.partial_turn.available = 1;
    message.partial_turn.committed_progress = 1;
    message.partial_turn.reset_required = 1;
    message.partial_turn.failure_status = YVEX_ERR_BACKEND;
    message.partial_turn.failure_class = YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE;
    message.partial_turn.stop_reason = YVEX_CLIENT_STOP_MODEL_FAILURE;
    message.partial_turn.initial_position = 4u;
    message.partial_turn.final_committed_position = final_position;
    message.partial_turn.committed_token_count = committed_tokens;
    message.partial_turn.published_text_bytes = strlen(bytes);
    message.partial_turn.target_state_generation = committed_tokens;
    message.partial_turn.rng_generation = 0u;
    message.partial_turn.token_ledger_generation = 2u;
    message.partial_turn.message_history_generation = 1u;
    message.partial_turn.transcript_generation = 1u;
    memset(message.partial_turn.target_state_identity, 'a', 64u);
    message.partial_turn.target_state_identity[64] = '\0';
    memset(message.partial_turn.rng_state_identity, 'b', 64u);
    message.partial_turn.rng_state_identity[64] = '\0';
    memset(message.partial_turn.token_ledger_identity, 'c', 64u);
    message.partial_turn.token_ledger_identity[64] = '\0';
    memset(message.partial_turn.published_text_identity, 'd', 64u);
    message.partial_turn.published_text_identity[64] = '\0';
    return yvex_server_protocol_send(fd, &message, err);
}

static int send_native_partial_fence(int fd,
                                     const yvex_client_request *request,
                                     yvex_error *err)
{
    return send_native_partial(
        fd, request, YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
        "```cuda\nint value = ",
        "injected CUDA failure after committed output", 2u, 6u, err);
}

static int send_native_partial_reasoning(int fd,
                                         const yvex_client_request *request,
                                         yvex_error *err)
{
    return send_native_partial(
        fd, request, YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING,
        "reasoning committed before failure",
        "injected failure between explicit reasoning and final output", 1u,
        5u, err);
}

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

static int native_prompt_contains(const yvex_client_request *request,
                                  const char *marker)
{
    unsigned long long offset;
    size_t count = strlen(marker);
    if (!request || request->provider_request || !request->prompt ||
        request->prompt_bytes < count)
        return 0;
    for (offset = 0u; offset <= request->prompt_bytes - count; ++offset)
        if (memcmp(request->prompt + offset, marker, count) == 0) return 1;
    return 0;
}

static int send_native_cancellation(int fd,
                                    const yvex_client_request *request,
                                    yvex_error *err)
{
    struct pollfd listener = {.fd = listener_fd, .events = POLLIN};
    yvex_client_message message;
    int cancellation, ready, rc;
    do {
        ready = poll(&listener, 1u, 2000);
    } while (ready < 0 && errno == EINTR);
    if (ready <= 0 || !(listener.revents & POLLIN)) {
        yvex_error_set(err, YVEX_ERR_TIMEOUT, "test.openai-host.cancel",
                       "native cancellation request did not arrive");
        return YVEX_ERR_TIMEOUT;
    }
    cancellation = accept(listener_fd, NULL, NULL);
    if (cancellation < 0) {
        yvex_error_set(err, YVEX_ERR_IO, "test.openai-host.cancel",
                       "native cancellation accept failed");
        return YVEX_ERR_IO;
    }
    rc = serve_connection(cancellation, err);
    close(cancellation);
    if (rc != YVEX_OK) return rc;
    message_base(&message, YVEX_CLIENT_MESSAGE_ERROR, request);
    message.status = YVEX_ERR_CANCELLED;
    message.failure_class = YVEX_CLIENT_FAILURE_CLIENT_CANCELLED;
    strcpy(message.reason, "native generation cancellation admitted");
    return yvex_server_protocol_send(fd, &message, err);
}

static int send_generation(int fd, const yvex_client_request *request,
                           yvex_error *err)
{
    const yvex_provider_request *provider = request->provider_request;
    yvex_client_message message;
    unsigned long long index;
    int has_tool_result = 0, has_reasoning_history = 0, rc;

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
    if (rc == YVEX_OK && !provider) {
        size_t progress_events = native_prompt_contains(request, "WAIT_PREFILL_CANCEL")
                                     ? 2u : 4u;
        rc = send_native_progress(fd, request, progress_events, err);
    }
    if (rc == YVEX_OK &&
        (native_prompt_contains(request, "WAIT_PREFILL_CANCEL") ||
         native_prompt_contains(request, "WAIT_DECODE_CANCEL")))
        return send_native_cancellation(fd, request, err);
    if (rc == YVEX_OK &&
        native_prompt_contains(request, "WAIT_REASONING_CANCEL")) {
        if (request->reasoning_policy == YVEX_REASONING_DISABLED) {
            message_base(&message, YVEX_CLIENT_MESSAGE_ERROR, request);
            message.status = YVEX_ERR_STATE;
            strcpy(message.reason,
                   "reasoning cancellation fixture requires an explicit policy");
            return yvex_server_protocol_send(fd, &message, err);
        }
        rc = send_fragment(fd, request,
                           YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING,
                           "reasoning before cancellation", NULL, NULL, err);
        if (rc == YVEX_OK) return send_native_cancellation(fd, request, err);
    }
    if (rc == YVEX_OK && request_contains(provider, "DISCONNECT")) {
        struct pollfd listener = {.fd = listener_fd, .events = POLLIN};
        int ready;
        if (request_contains(provider, "REASONING_DISCONNECT")) {
            if (provider->reasoning_policy == YVEX_REASONING_DISABLED) {
                message_base(&message, YVEX_CLIENT_MESSAGE_ERROR, request);
                message.status = YVEX_ERR_STATE;
                strcpy(message.reason,
                       "reasoning disconnect fixture requires an explicit policy");
                return yvex_server_protocol_send(fd, &message, err);
            }
            rc = send_fragment(fd, request,
                               YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING,
                               "reasoning before disconnect", NULL, NULL, err);
            if (rc != YVEX_OK) return rc;
        }
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
    for (index = 0u; provider && index < provider->message_count; ++index) {
        if (provider->messages[index].role == YVEX_PROVIDER_ROLE_TOOL)
            has_tool_result = 1;
        else if (provider->messages[index].role == YVEX_PROVIDER_ROLE_ASSISTANT &&
                 provider->messages[index].reasoning_content.count)
            has_reasoning_history = 1;
    }
    if (rc == YVEX_OK && provider &&
        provider->reasoning_policy != YVEX_REASONING_DISABLED)
        rc = send_fragment(fd, request,
                           YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING,
                           "explicit model reasoning", NULL, NULL, err);
    if (rc == YVEX_OK && !provider &&
        native_prompt_contains(request, "PARTIAL_FENCE"))
        return send_native_partial_fence(fd, request, err);
    if (rc == YVEX_OK && !provider &&
        native_prompt_contains(request, "PARTIAL_REASONING")) {
        if (request->reasoning_policy == YVEX_REASONING_DISABLED) {
            message_base(&message, YVEX_CLIENT_MESSAGE_ERROR, request);
            message.status = YVEX_ERR_STATE;
            strcpy(message.reason,
                   "partial reasoning fixture requires an explicit policy");
            return yvex_server_protocol_send(fd, &message, err);
        }
        return send_native_partial_reasoning(fd, request, err);
    }
    if (rc == YVEX_OK && !provider &&
        native_prompt_contains(request, "MARKDOWN_STREAM")) {
        rc = send_native_markdown(fd, request, err);
        message.provider_finish = YVEX_PROVIDER_FINISH_STOP;
    } else if (rc == YVEX_OK && !provider &&
               native_prompt_contains(request, "REASONING_STREAM")) {
        if (request->reasoning_policy == YVEX_REASONING_DISABLED) {
            message_base(&message, YVEX_CLIENT_MESSAGE_ERROR, request);
            message.status = YVEX_ERR_STATE;
            strcpy(message.reason, "reasoning fixture requires an explicit policy");
            return yvex_server_protocol_send(fd, &message, err);
        }
        rc = send_native_reasoning(fd, request, err);
        message.provider_finish = YVEX_PROVIDER_FINISH_STOP;
    } else if (rc == YVEX_OK && provider && provider->tool_count && !has_tool_result) {
        rc = send_fragment(fd, request, YVEX_PROVIDER_OUTPUT_FUNCTION_CALL,
                           "{\"match_id\":\"m1\"}", "call_fixture_1",
                           provider->tools[0].name, err);
        if (rc == YVEX_OK && request_contains(provider, "MULTI_TOOL"))
            rc = send_fragment(fd, request,
                               YVEX_PROVIDER_OUTPUT_FUNCTION_CALL,
                               "{\"match_id\":\"m2\"}", "call_fixture_2",
                               provider->tools[0].name, err);
        message.provider_finish = YVEX_PROVIDER_FINISH_TOOL_CALLS;
    } else if (rc == YVEX_OK) {
        const char *text = has_tool_result && has_reasoning_history
                               ? "Reasoning continuity accepted."
                           : has_tool_result ? "Match context accepted."
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
    message.reused_tokens = 1u;
    message.prefill_tokens = 4u;
    message.completion_tokens = 3u;
    message.total_tokens = 8u;
    message.generated_tokens = 3u;
    if ((provider ? provider->reasoning_policy : request->reasoning_policy) !=
        YVEX_REASONING_DISABLED) {
        message.reasoning_tokens = 2u;
        message.final_tokens = 1u;
        message.first_reasoning_seconds = 2.5;
        message.first_final_seconds = 2.75;
        message.reasoning_seconds = 0.25;
        message.final_seconds = 0.25;
        message.total_completion_seconds = 0.5;
        message.reasoning_rate = 8.0;
        message.final_rate = 4.0;
        message.total_completion_rate = 6.0;
    } else {
        message.final_tokens = 3u;
    }
    message.final_position = 8u;
    message.context_used = 8u;
    message.prefill_seconds = 2.0;
    message.first_token_seconds = 2.5;
    message.decode_seconds = 3.0;
    message.prefill_rate = 2.0;
    message.decode_rate = 1.0;
    message.stop_reason = 3u;
    message.generation_phase = YVEX_CLIENT_PHASE_COMPLETE;
    message.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
    memset(message.turn_identity, 'e', 64u);
    message.turn_identity[64] = '\0';
    return yvex_server_protocol_send(fd, &message, err);
}

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
    else if (request.operation == YVEX_CLIENT_OP_CONSOLE_STATUS)
        rc = send_console_status(fd, &request, err);
    else if (request.operation == YVEX_CLIENT_OP_RUNTIME_WATCH ||
             request.operation == YVEX_CLIENT_OP_RUNTIME_TRACE)
        rc = send_event_stream(fd, &request, err);
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
        {
            int connection_status = serve_connection(client, &err);
            if (connection_status != YVEX_OK && !stopped) {
                fprintf(stderr, "connection.error status=%d where=%s reason=%s\n",
                        connection_status, yvex_error_where(&err),
                        yvex_error_message(&err));
                rc = 1;
            }
        }
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
