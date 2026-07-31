/*
 * Exercises field-wise request/message roundtrip, embedded bytes, version refusal,
 * malformed-frame refusal, and stable protocol identities without opening an engine.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <yvex/server.h>

#include "tests/test.h"

static int test_request_roundtrip(void)
{
    static const unsigned char prompt[] = {'a', 0u, 'b', 0xf0u, 0x9fu, 0x98u, 0x80u};
    unsigned char frame[2048];
    unsigned char *owned_prompt = NULL;
    yvex_provider_request *owned_provider = NULL;
    yvex_client_request source, decoded;
    unsigned long long count = 0u;
    yvex_error err;
    int rc;
    memset(&source, 0, sizeof(source));
    source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.operation = YVEX_CLIENT_OP_GENERATION_TURN;
    source.request_number = 42u;
    strcpy(source.session_name, "main");
    source.prompt = prompt;
    source.prompt_bytes = sizeof(prompt);
    source.maximum_new_tokens = 17u;
    source.stochastic = 1;
    source.seed_present = 1;
    source.seed = 0u;
    source.temperature = 0.8;
    source.top_k = 50u;
    source.top_p = 0.95;
    source.min_p = 0.05;
    source.typical_p = 1.0;
    rc = yvex_protocol_request_encode(&source, frame, sizeof(frame), &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count > 0u, "request encode");
    memset(&decoded, 0, sizeof(decoded));
    rc = yvex_protocol_request_decode(frame, count, &decoded, &owned_prompt,
                                      &owned_provider, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "request decode");
    YVEX_TEST_ASSERT(decoded.operation == source.operation, "request operation");
    YVEX_TEST_ASSERT(decoded.request_number == 42u, "request number");
    YVEX_TEST_ASSERT_STREQ(decoded.session_name, "main", "session name");
    YVEX_TEST_ASSERT(decoded.prompt_bytes == sizeof(prompt), "prompt extent");
    YVEX_TEST_ASSERT(memcmp(decoded.prompt, prompt, sizeof(prompt)) == 0,
                     "prompt bytes including NUL");
    YVEX_TEST_ASSERT(decoded.seed_present && decoded.seed == 0u, "seed zero");
    free(owned_prompt);
    yvex_provider_request_close(&owned_provider);

    source.schema_version++;
    count = 99u;
    rc = yvex_protocol_request_encode(&source, frame, sizeof(frame), &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && count == 0u,
                     "unsupported request version refuses");
    rc = yvex_protocol_request_decode(frame, 3u, &decoded, &owned_prompt,
                                      &owned_provider, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "truncated request refuses");
    return 0;
}

static int test_all_operation_roundtrips(void)
{
    unsigned char frame[2048];
    yvex_client_request source, decoded;
    yvex_provider_request *provider = NULL;
    unsigned char *prompt = NULL;
    unsigned long long count;
    yvex_error err;
    unsigned int value;
    for (value = YVEX_CLIENT_OP_HANDSHAKE;
         value <= YVEX_CLIENT_OP_CONSOLE_STATUS; ++value) {
        memset(&source, 0, sizeof(source));
        source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
        source.operation = (yvex_client_operation)value;
        source.temperature = 1.0;
        source.top_p = 1.0;
        source.typical_p = 1.0;
        count = 0u;
        YVEX_TEST_ASSERT(
            yvex_protocol_request_encode(&source, frame, sizeof(frame), &count,
                                         &err) == YVEX_OK,
            "all protocol-v4 operations encode");
        YVEX_TEST_ASSERT(
            yvex_protocol_request_decode(frame, count, &decoded, &prompt,
                                         &provider, &err) == YVEX_OK &&
                decoded.operation == (yvex_client_operation)value,
            "all protocol-v4 operations decode");
        free(prompt);
        prompt = NULL;
        yvex_provider_request_close(&provider);
    }
    return 0;
}

static int test_schema_refusals(void)
{
    unsigned char frame[2048], malformed[2048];
    unsigned char *prompt = NULL;
    yvex_provider_request *provider = NULL;
    yvex_client_request request, decoded_request;
    yvex_client_message message, decoded_message;
    unsigned long long count = 0u;
    yvex_error err;
    memset(&request, 0, sizeof(request));
    request.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request.operation = YVEX_CLIENT_OP_RUNTIME_STATUS;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&request, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK,
        "valid refusal fixture request");
    memcpy(malformed, frame, (size_t)count);
    malformed[15] = 0xffu;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_decode(malformed, count, &decoded_request,
                                     &prompt, &provider, &err) == YVEX_ERR_FORMAT,
        "unknown request operation refuses on decode");
    memcpy(malformed, frame, (size_t)count);
    malformed[2] = 1u;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_decode(malformed, count, &decoded_request,
                                     &prompt, &provider, &err) == YVEX_ERR_FORMAT,
        "nonzero TLV reserved field refuses");
    memcpy(malformed, frame, (size_t)count);
    malformed[count] = 0u;
    malformed[count + 1u] = 191u;
    memset(malformed + count + 2u, 0, 6u);
    YVEX_TEST_ASSERT(
        yvex_protocol_request_decode(malformed, count + 8u, &decoded_request,
                                     &prompt, &provider, &err) == YVEX_ERR_FORMAT,
        "unknown request field refuses");
    request.operation = (yvex_client_operation)-1;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&request, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "negative request operation refuses on encode");
    request.operation = YVEX_CLIENT_OP_RUNTIME_STATUS;
    request.temperature = NAN;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&request, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "nonfinite request sampling refuses");

    memset(&message, 0, sizeof(message));
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_ACK;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK,
        "valid refusal fixture message");
    memcpy(malformed, frame, (size_t)count);
    malformed[15] = 0xffu;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_decode(malformed, count, &decoded_message,
                                     &err) == YVEX_ERR_FORMAT,
        "unknown message kind refuses on decode");
    memcpy(malformed, frame, (size_t)count);
    malformed[count] = 0u;
    malformed[count + 1u] = 191u;
    memset(malformed + count + 2u, 0, 6u);
    YVEX_TEST_ASSERT(
        yvex_protocol_message_decode(malformed, count + 8u, &decoded_message,
                                     &err) == YVEX_ERR_FORMAT,
        "unknown message field refuses");
    message.failure_class = (yvex_client_failure_class)-1;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "negative message enum refuses on encode");
    message.failure_class = YVEX_CLIENT_FAILURE_NONE;
    message.console.runtime_ready = 2;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "nonboolean message fact refuses");
    message.console.runtime_ready = 0;
    message.decode_seconds = NAN;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "nonfinite message timing refuses");
    message.decode_seconds = 0.0;
    message.kv_used_bytes = 1u;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "unavailable message KV fact refuses nonzero bytes");
    message.kv_used_bytes = 0u;
    message.publication_seconds = 1.0;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "unavailable publication timing refuses nonzero seconds");
    message.publication_seconds = 0.0;
    message.console.kv_used_bytes = 1u;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "unavailable console KV fact refuses nonzero bytes");
    message.console.kv_used_bytes = 0u;
    strcpy(message.console.selected_model_identity, "selected-model");
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "unavailable selected model refuses a projected identity");
    return 0;
}

static int test_message_roundtrip(void)
{
    yvex_client_message source, decoded;
    unsigned char frame[8192];
    unsigned long long count = 0u;
    yvex_error err;
    int rc;
    memset(&source, 0, sizeof(source));
    source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.kind = YVEX_CLIENT_MESSAGE_STATUS;
    source.status = YVEX_OK;
    source.request_number = 9u;
    strcpy(source.session_name, "session-a");
    source.runtime.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.runtime.status = YVEX_SERVER_STATUS_READY;
    source.runtime.backend = YVEX_BACKEND_KIND_CUDA;
    strcpy(source.runtime.target_id, "deepseek4-v4-flash");
    source.runtime.runtime_ready = 1;
    source.runtime.generation_ready = 1;
    source.runtime.openai_listener_enabled = 1;
    source.runtime.openai_listener_ready = 1;
    source.runtime.explicit_reasoning_channel_supported = 1;
    source.runtime.openai_port = 8001u;
    source.runtime.context_capacity = 4096u;
    source.runtime.prefill_chunk_tokens = 64u;
    source.runtime.maximum_new_tokens = 256u;
    source.runtime.maximum_output_bytes = 1048576u;
    source.runtime.maximum_sessions = 8u;
    source.runtime.request_queue_capacity = 16u;
    source.runtime.openai_timeout_ms = 600000u;
    source.runtime.trace_level = YVEX_SERVER_TRACE_STAGES;
    source.runtime.metrics.model_open_count = 1u;
    source.runtime.metrics.artifact_open_count = 1u;
    source.runtime.metrics.queue_capacity = 16u;
    source.runtime.metrics.active_http_requests = 2u;
    source.runtime.metrics.completed_http_requests = 7u;
    source.runtime.metrics.failed_http_requests = 3u;
    source.runtime.metrics.cancelled_http_requests = 1u;
    rc = yvex_protocol_message_encode(&source, frame, sizeof(frame), &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "message encode");
    rc = yvex_protocol_message_decode(frame, count, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "message decode");
    YVEX_TEST_ASSERT(decoded.kind == YVEX_CLIENT_MESSAGE_STATUS, "message kind");
    YVEX_TEST_ASSERT(decoded.runtime.status == YVEX_SERVER_STATUS_READY,
                     "runtime status");
    YVEX_TEST_ASSERT(decoded.runtime.backend == YVEX_BACKEND_KIND_CUDA,
                     "runtime backend");
    YVEX_TEST_ASSERT(decoded.runtime.metrics.model_open_count == 1u,
                     "model-open metric");
    YVEX_TEST_ASSERT(decoded.runtime.metrics.queue_capacity == 16u,
                     "queue-capacity metric");
    YVEX_TEST_ASSERT(decoded.runtime.openai_listener_enabled &&
                         decoded.runtime.openai_listener_ready &&
                         decoded.runtime.openai_port == 8001u,
                     "integrated OpenAI listener roundtrip");
    YVEX_TEST_ASSERT(decoded.runtime.context_capacity == 4096u &&
                         decoded.runtime.prefill_chunk_tokens == 64u &&
                         decoded.runtime.maximum_new_tokens == 256u &&
                         decoded.runtime.maximum_output_bytes == 1048576u &&
                         decoded.runtime.maximum_sessions == 8u &&
                         decoded.runtime.request_queue_capacity == 16u &&
                         decoded.runtime.openai_timeout_ms == 600000u &&
                         decoded.runtime.trace_level == YVEX_SERVER_TRACE_STAGES &&
                         decoded.runtime.explicit_reasoning_channel_supported,
                     "runtime configuration and channel capability roundtrip");
    YVEX_TEST_ASSERT(decoded.runtime.metrics.active_http_requests == 2u &&
                         decoded.runtime.metrics.completed_http_requests == 7u &&
                         decoded.runtime.metrics.failed_http_requests == 3u &&
                         decoded.runtime.metrics.cancelled_http_requests == 1u,
                     "integrated HTTP metrics roundtrip");
    memset(&source, 0, sizeof(source));
    source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.kind = YVEX_CLIENT_MESSAGE_ERROR;
    source.status = YVEX_ERR_BOUNDS;
    source.failure_class = YVEX_CLIENT_FAILURE_QUEUE_FULL;
    rc = yvex_protocol_message_encode(&source, frame, sizeof(frame), &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "negative status encode");
    rc = yvex_protocol_message_decode(frame, count, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && decoded.status == YVEX_ERR_BOUNDS &&
                         decoded.failure_class == YVEX_CLIENT_FAILURE_QUEUE_FULL,
                     "negative status roundtrip");

    memset(&source, 0, sizeof(source));
    source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.kind = YVEX_CLIENT_MESSAGE_CONSOLE_STATUS;
    source.status = YVEX_OK;
    source.turn_count = 4u;
    source.context_used = 37u;
    source.kv_used_bytes = 8192u;
    source.kv_used_available = 1;
    source.publication_seconds = 0.125;
    source.publication_timing_available = 1;
    source.generation_phase = YVEX_CLIENT_PHASE_COMPLETE;
    source.cancellation_class = YVEX_CLIENT_CANCELLATION_COMPLETED;
    source.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
    source.console.schema_version = 1u;
    source.console.backend = YVEX_BACKEND_KIND_CUDA;
    source.console.session_state = YVEX_SERVER_SESSION_READY;
    source.console.generation_phase = YVEX_CLIENT_PHASE_DECODE;
    source.console.cancellation_class = YVEX_CLIENT_CANCELLATION_REQUESTED;
    source.console.position = 37u;
    source.console.turn_count = 4u;
    source.console.context_capacity = 4096u;
    source.console.context_used = 37u;
    source.console.kv_used_bytes = 8192u;
    source.console.runtime_ready = 1;
    source.console.session_available = 1;
    source.console.attached = 1;
    source.console.cancel_requested = 1;
    source.console.kv_used_available = 1;
    source.console.progress_available = 1;
    source.console.selected_model_available = 1;
    source.console.explicit_reasoning_channel_supported = 1;
    strcpy(source.console.session_name, "main");
    strcpy(source.console.live_model_identity,
           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    strcpy(source.console.physical_variant_identity,
           "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    strcpy(source.console.selected_model_identity,
           "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    rc = yvex_protocol_message_encode(&source, frame, sizeof(frame), &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "console status encode");
    rc = yvex_protocol_message_decode(frame, count, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         decoded.kind == YVEX_CLIENT_MESSAGE_CONSOLE_STATUS &&
                         decoded.turn_count == 4u && decoded.context_used == 37u &&
                         decoded.kv_used_available &&
                         decoded.publication_timing_available &&
                         decoded.publication_seconds == 0.125 &&
                         decoded.generation_phase == YVEX_CLIENT_PHASE_COMPLETE &&
                         decoded.cancellation_class == YVEX_CLIENT_CANCELLATION_COMPLETED &&
                         decoded.stream_channel == YVEX_CLIENT_STREAM_CONTROL_EVENT,
                     "complete turn facts roundtrip");
    YVEX_TEST_ASSERT(decoded.console.runtime_ready &&
                         decoded.console.session_available &&
                         decoded.console.attached &&
                         decoded.console.cancel_requested &&
                         decoded.console.kv_used_available &&
                         decoded.console.progress_available &&
                         decoded.console.selected_model_available &&
                         decoded.console.explicit_reasoning_channel_supported &&
                         decoded.console.position == 37u &&
                         decoded.console.turn_count == 4u &&
                         decoded.console.context_capacity == 4096u &&
                         decoded.console.context_used == 37u &&
                         decoded.console.kv_used_bytes == 8192u &&
                         decoded.console.generation_phase == YVEX_CLIENT_PHASE_DECODE &&
                         decoded.console.cancellation_class == YVEX_CLIENT_CANCELLATION_REQUESTED &&
                         strcmp(decoded.console.session_name, "main") == 0,
                     "console status facts roundtrip");
    rc = yvex_protocol_message_decode(frame, count - 1u, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "truncated message refuses");
    return 0;
}

typedef struct {
    int listener;
} v3_peer;

static void *v3_peer_main(void *opaque)
{
    static const unsigned char response[12] = {
        'Y', 'V', 'X', 'P', 0u, 3u, 0u, 2u, 0u, 0u, 0u, 0u};
    v3_peer *peer = opaque;
    unsigned char header[12], discard[4096];
    unsigned int length;
    int client = accept(peer->listener, NULL, NULL);
    if (client < 0 || recv(client, header, sizeof(header), MSG_WAITALL) !=
                          (ssize_t)sizeof(header)) {
        if (client >= 0) (void)close(client);
        return NULL;
    }
    length = ((unsigned int)header[8] << 24u) |
             ((unsigned int)header[9] << 16u) |
             ((unsigned int)header[10] << 8u) | header[11];
    while (length) {
        size_t chunk = length < sizeof(discard) ? length : sizeof(discard);
        ssize_t received = recv(client, discard, chunk, MSG_WAITALL);
        if (received <= 0) break;
        length -= (unsigned int)received;
    }
    (void)send(client, response, sizeof(response), 0);
    (void)close(client);
    return NULL;
}

static int test_v3_frame_refusal(void)
{
    struct sockaddr_un address;
    char path[sizeof(address.sun_path)];
    yvex_client *client = NULL;
    yvex_error err;
    pthread_t thread;
    v3_peer peer;
    int rc;
    (void)snprintf(path, sizeof(path), "build/tests/protocol-v4-%lu.sock",
                   (unsigned long)getpid());
    (void)unlink(path);
    peer.listener = socket(AF_UNIX, SOCK_STREAM, 0);
    YVEX_TEST_ASSERT(peer.listener >= 0, "v3 peer socket");
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path) + 1u);
    YVEX_TEST_ASSERT(bind(peer.listener, (struct sockaddr *)&address,
                          sizeof(address)) == 0 &&
                         chmod(path, 0600) == 0 && listen(peer.listener, 1) == 0,
                     "v3 peer bind/listen");
    YVEX_TEST_ASSERT(pthread_create(&thread, NULL, v3_peer_main, &peer) == 0,
                     "v3 peer thread");
    rc = yvex_client_connect(&client, path, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && client == NULL &&
                         strstr(yvex_error_message(&err), "version 4") != NULL,
                     "v3 frame explicitly refuses");
    YVEX_TEST_ASSERT(pthread_join(thread, NULL) == 0, "v3 peer join");
    (void)close(peer.listener);
    (void)unlink(path);
    return 0;
}

static int test_bounded_parser_mutation(void)
{
    unsigned char bytes[1024];
    unsigned int state = 0x9e3779b9u;
    unsigned long long iteration, index;
    for (iteration = 0u; iteration < 256u; ++iteration) {
        yvex_client_request request;
        yvex_client_message message;
        unsigned char *prompt = NULL;
        yvex_provider_request *provider = NULL;
        yvex_error err;
        unsigned long long count = (iteration * 37u) % sizeof(bytes);
        int request_rc, message_rc;
        for (index = 0u; index < count; ++index) {
            state ^= state << 13u;
            state ^= state >> 17u;
            state ^= state << 5u;
            bytes[index] = (unsigned char)state;
        }
        request_rc = yvex_protocol_request_decode(
            bytes, count, &request, &prompt, &provider, &err);
        YVEX_TEST_ASSERT(request_rc == YVEX_OK || request_rc < YVEX_OK,
                         "request mutation returns typed status");
        free(prompt);
        yvex_provider_request_close(&provider);
        message_rc = yvex_protocol_message_decode(bytes, count, &message, &err);
        YVEX_TEST_ASSERT(message_rc == YVEX_OK || message_rc < YVEX_OK,
                         "message mutation returns typed status");
    }
    return 0;
}

int yvex_test_protocol(void)
{
    if (test_request_roundtrip() != 0) return 1;
    if (test_all_operation_roundtrips() != 0) return 1;
    if (test_schema_refusals() != 0) return 1;
    if (test_message_roundtrip() != 0) return 1;
    if (test_v3_frame_refusal() != 0) return 1;
    if (test_bounded_parser_mutation() != 0) return 1;
    return 0;
}
