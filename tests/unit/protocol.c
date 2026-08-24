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
    source.reasoning_policy = YVEX_REASONING_MAXIMUM;
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
    YVEX_TEST_ASSERT(decoded.reasoning_policy == YVEX_REASONING_MAXIMUM,
                     "request-bound reasoning policy");
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
        if (source.operation == YVEX_CLIENT_OP_SESSION_STATE_SAVE ||
            source.operation == YVEX_CLIENT_OP_SESSION_STATE_RESTORE) {
            strcpy(source.state_path, "/tmp/session-state.yvex");
            if (source.operation == YVEX_CLIENT_OP_SESSION_STATE_RESTORE)
                source.maximum_state_file_bytes = 4096u;
        }
        if (source.operation == YVEX_CLIENT_OP_SESSION_FORK) {
            strcpy(source.fork_session_name, "child");
            source.maximum_prefix_bytes = 8192u;
        }
        count = 0u;
        YVEX_TEST_ASSERT(
            yvex_protocol_request_encode(&source, frame, sizeof(frame), &count,
                                         &err) == YVEX_OK,
            "all protocol-v12 operations encode");
        YVEX_TEST_ASSERT(
            yvex_protocol_request_decode(frame, count, &decoded, &prompt,
                                         &provider, &err) == YVEX_OK &&
                decoded.operation == (yvex_client_operation)value,
            "all protocol-v12 operations decode");
        if (source.operation == YVEX_CLIENT_OP_SESSION_FORK)
            YVEX_TEST_ASSERT(
                strcmp(decoded.fork_session_name, "child") == 0 &&
                    decoded.maximum_prefix_bytes == 8192u,
                "session fork fields roundtrip");
        free(prompt);
        prompt = NULL;
        yvex_provider_request_close(&provider);
    }
    return 0;
}

static int test_schema_refusals(void)
{
    unsigned char frame[4096], malformed[4096];
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
    malformed[count + 1u] = 255u;
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
    request.reasoning_policy = (yvex_reasoning_policy)99;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&request, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "unknown reasoning policy refuses");
    request.reasoning_policy = YVEX_REASONING_DISABLED;
    request.operation = YVEX_CLIENT_OP_SESSION_FORK;
    strcpy(request.fork_session_name, "child");
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&request, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "session fork without an explicit byte bound refuses");
    request.maximum_prefix_bytes = 4096u;
    request.fork_session_name[0] = '\0';
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&request, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "session fork without a child identity refuses");
    strcpy(request.fork_session_name, "child");
    request.operation = YVEX_CLIENT_OP_RUNTIME_STATUS;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&request, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "non-fork operations refuse fork-only fields");
    request.fork_session_name[0] = '\0';
    request.maximum_prefix_bytes = 0u;
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
    malformed[count + 1u] = 255u;
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
    memset(message.console.selected_model_identity, 0,
           sizeof(message.console.selected_model_identity));
    message.partial_turn.available = 1;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "incomplete partial-turn contract refuses");

    memset(&message, 0, sizeof(message));
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_ACK;
    message.generation_mode = YVEX_SERVER_GENERATION_DSPARK;
    message.draft_cycle_count = 1u;
    message.draft_forward_count = 1u;
    message.proposed_tokens = 257u;
    message.selected_verification_tokens = 1u;
    message.target_verification_count = 1u;
    message.accepted_draft_tokens = 1u;
    message.discarded_draft_tokens = 256u;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK,
        "valid speculative aggregate fixture");
    {
        unsigned long long offset = 0u;
        int replaced = 0;
        while (offset + 16u <= count) {
            unsigned long long length =
                ((unsigned long long)frame[offset + 4u] << 24u) |
                ((unsigned long long)frame[offset + 5u] << 16u) |
                ((unsigned long long)frame[offset + 6u] << 8u) |
                frame[offset + 7u];
            if (length == 8u && frame[offset + 8u] == 0u &&
                frame[offset + 9u] == 0u && frame[offset + 10u] == 0u &&
                frame[offset + 11u] == 0u && frame[offset + 12u] == 0u &&
                frame[offset + 13u] == 0u && frame[offset + 14u] == 1u &&
                frame[offset + 15u] == 1u) {
                memset(frame + offset + 8u, 0, 8u);
                replaced = 1;
                break;
            }
            offset += 8u + length;
        }
        YVEX_TEST_ASSERT(replaced, "speculative aggregate wire field located");
    }
    YVEX_TEST_ASSERT(
        yvex_protocol_message_decode(frame, count, &decoded_message,
                                     &err) == YVEX_ERR_FORMAT,
        "inconsistent speculative aggregate refuses on decode");

    memset(&message, 0, sizeof(message));
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_TURN_COMPLETE;
    message.status = YVEX_ERR_CANCELLED;
    message.generation_phase = YVEX_CLIENT_PHASE_CANCELLED;
    message.generation_mode = YVEX_SERVER_GENERATION_DSPARK;
    message.draft_cycle_count = 1u;
    message.draft_forward_count = 1u;
    message.proposed_tokens = 5u;
    message.selected_verification_tokens = 5u;
    message.discarded_draft_tokens = 5u;
    message.confidence_logit_count = 5u;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, count, &decoded_message,
                                         &err) == YVEX_OK &&
            decoded_message.draft_cycle_count == 1u &&
            decoded_message.draft_forward_count == 1u &&
            decoded_message.target_verification_count == 0u &&
            decoded_message.discarded_draft_tokens == 5u,
        "cancelled verification preserves completed draft accounting");

    message.status = YVEX_OK;
    message.generation_phase = YVEX_CLIENT_PHASE_COMPLETE;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "completed speculation refuses an unfinished verification cycle");
    return 0;
}

static int test_message_roundtrip(void)
{
    yvex_client_message source, decoded;
    unsigned char frame[16384];
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
    source.runtime.generation_mode = YVEX_SERVER_GENERATION_DSPARK;
    strcpy(source.runtime.target_id, "deepseek4-v4-flash-dspark");
    source.runtime.runtime_ready = 1;
    source.runtime.generation_ready = 1;
    source.runtime.openai_listener_enabled = 1;
    source.runtime.openai_listener_ready = 1;
    source.runtime.explicit_reasoning_channel_supported = 1;
    source.runtime.independent_session_scheduling_ready = 1;
    source.runtime.openai_port = 8001u;
    source.runtime.context_capacity = 4096u;
    source.runtime.prefill_chunk_tokens = 64u;
    source.runtime.maximum_new_tokens = 256u;
    source.runtime.maximum_output_bytes = 1048576u;
    source.runtime.maximum_sessions = 8u;
    source.runtime.request_queue_capacity = 16u;
    source.runtime.concurrent_sequences = 4u;
    source.runtime.capacity_required_bytes = 1024u;
    source.runtime.capacity_unreserved_bytes = 2048u;
    memset(source.runtime.runtime_model_identity, 'a', 64u);
    memset(source.runtime.runtime_binding_identity, 'b', 64u);
    memset(source.runtime.artifact_identity, 'c', 64u);
    memset(source.runtime.physical_variant_identity, 'd', 64u);
    memset(source.runtime.capacity_plan_identity, 'e', 64u);
    source.runtime.openai_timeout_ms = 600000u;
    source.runtime.trace_level = YVEX_SERVER_TRACE_STAGES;
    source.runtime.metrics.model_open_count = 1u;
    source.runtime.metrics.artifact_open_count = 1u;
    source.runtime.metrics.queue_capacity = 16u;
    source.runtime.metrics.active_http_requests = 2u;
    source.runtime.metrics.completed_http_requests = 7u;
    source.runtime.metrics.failed_http_requests = 3u;
    source.runtime.metrics.cancelled_http_requests = 1u;
    source.state_checkpoint.schema_version =
        YVEX_CLIENT_STATE_CHECKPOINT_SCHEMA_V1;
    source.state_checkpoint.file_bytes = 8192u;
    source.state_checkpoint.scope_count = 2u;
    source.state_checkpoint.committed_sequence_length = 64u;
    memset(source.state_checkpoint.runtime_model_identity, 'a', 64u);
    memset(source.state_checkpoint.runtime_binding_identity, 'b', 64u);
    memset(source.state_checkpoint.artifact_identity, 'c', 64u);
    memset(source.state_checkpoint.file_digest, 'd', 64u);
    rc = yvex_protocol_message_encode(&source, frame, sizeof(frame), &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "message encode");
    rc = yvex_protocol_message_decode(frame, count, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "message decode");
    YVEX_TEST_ASSERT(decoded.kind == YVEX_CLIENT_MESSAGE_STATUS, "message kind");
    YVEX_TEST_ASSERT(decoded.runtime.status == YVEX_SERVER_STATUS_READY,
                     "runtime status");
    YVEX_TEST_ASSERT(decoded.runtime.backend == YVEX_BACKEND_KIND_CUDA,
                     "runtime backend");
    YVEX_TEST_ASSERT(decoded.runtime.generation_mode ==
                         YVEX_SERVER_GENERATION_DSPARK,
                     "runtime generation mode");
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
                         decoded.runtime.concurrent_sequences == 4u &&
                         decoded.runtime.capacity_required_bytes == 1024u &&
                         decoded.runtime.capacity_unreserved_bytes == 2048u &&
                         decoded.runtime.independent_session_scheduling_ready &&
                         !decoded.runtime.continuous_batching_ready &&
                         strcmp(decoded.runtime.capacity_plan_identity,
                                source.runtime.capacity_plan_identity) == 0 &&
                         decoded.runtime.openai_timeout_ms == 600000u &&
                         decoded.runtime.trace_level == YVEX_SERVER_TRACE_STAGES &&
                         decoded.runtime.explicit_reasoning_channel_supported,
                     "runtime configuration and channel capability roundtrip");
    YVEX_TEST_ASSERT(decoded.runtime.metrics.active_http_requests == 2u &&
                         decoded.runtime.metrics.completed_http_requests == 7u &&
                         decoded.runtime.metrics.failed_http_requests == 3u &&
                         decoded.runtime.metrics.cancelled_http_requests == 1u,
                     "integrated HTTP metrics roundtrip");
    source.runtime.capacity_plan_identity[0] = 'x';
    source.runtime.capacity_plan_identity[1] = '\0';
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "ready status refuses an invalid capacity-plan identity");
    memset(source.runtime.capacity_plan_identity, 'e', 64u);
    source.runtime.capacity_plan_identity[64] = '\0';
    source.runtime.concurrent_sequences = 0u;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "ready status refuses zero admitted concurrency");
    source.runtime.concurrent_sequences = 4u;
    YVEX_TEST_ASSERT(
        decoded.state_checkpoint.schema_version ==
            YVEX_CLIENT_STATE_CHECKPOINT_SCHEMA_V1 &&
            decoded.state_checkpoint.file_bytes == 8192u &&
            decoded.state_checkpoint.scope_count == 2u &&
            decoded.state_checkpoint.committed_sequence_length == 64u &&
            strcmp(decoded.state_checkpoint.file_digest,
                   source.state_checkpoint.file_digest) == 0,
        "typed state checkpoint evidence roundtrip");
    memset(&source, 0, sizeof(source));
    source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.kind = YVEX_CLIENT_MESSAGE_ERROR;
    source.status = YVEX_ERR_BOUNDS;
    source.failure_class = YVEX_CLIENT_FAILURE_QUEUE_FULL;
    source.stream_channel = YVEX_CLIENT_STREAM_ERROR;
    rc = yvex_protocol_message_encode(&source, frame, sizeof(frame), &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "negative status encode");
    rc = yvex_protocol_message_decode(frame, count, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && decoded.status == YVEX_ERR_BOUNDS &&
                         decoded.failure_class == YVEX_CLIENT_FAILURE_QUEUE_FULL &&
                         decoded.stream_channel == YVEX_CLIENT_STREAM_ERROR,
                     "negative status roundtrip");
    source.stream_channel = YVEX_CLIENT_STREAM_UNSPECIFIED;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "error message without its typed stream channel refuses");
    memset(&source, 0, sizeof(source));
    source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.kind = YVEX_CLIENT_MESSAGE_FRAGMENT;
    source.status = YVEX_OK;
    source.provider_output_kind = YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING;
    source.stream_channel = YVEX_CLIENT_STREAM_EXPLICIT_REASONING;
    memcpy(source.bytes, "why", 3u);
    source.byte_count = 3u;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, count, &decoded, &err) ==
                YVEX_OK &&
            decoded.provider_output_kind ==
                YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING &&
            decoded.stream_channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING,
        "explicit reasoning channel roundtrip");
    source.stream_channel = YVEX_CLIENT_STREAM_FINAL_TEXT;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "mismatched typed output kind and stream channel refuses");

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
    source.reasoning_tokens = 7u;
    source.final_tokens = 11u;
    source.first_reasoning_seconds = 0.25;
    source.first_final_seconds = 1.5;
    source.reasoning_seconds = 1.25;
    source.final_seconds = 2.75;
    source.total_completion_seconds = 4.0;
    source.reasoning_rate = 5.6;
    source.final_rate = 4.0;
    source.total_completion_rate = 4.5;
    source.generation_phase = YVEX_CLIENT_PHASE_COMPLETE;
    source.cancellation_class = YVEX_CLIENT_CANCELLATION_COMPLETED;
    source.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
    source.generation_mode = YVEX_SERVER_GENERATION_DSPARK;
    source.draft_cycle_count = 3u;
    source.draft_forward_count = 3u;
    source.proposed_tokens = 16u;
    source.selected_verification_tokens = 15u;
    source.target_verification_count = 3u;
    source.accepted_draft_tokens = 10u;
    source.rejected_draft_tokens = 5u;
    source.discarded_draft_tokens = 1u;
    source.target_correction_or_bonus_tokens = 3u;
    source.maximum_accepted_prefix = 5u;
    source.confidence_logit_count = 16u;
    source.draft_seconds = 0.25;
    source.verification_seconds = 0.75;
    source.speculative_commit_seconds = 0.125;
    source.mean_accepted_prefix = 10.0 / 3.0;
    source.effective_committed_rate = 4.5;
    source.confidence_logit_minimum = -1.25;
    source.confidence_logit_maximum = 2.5;
    source.confidence_logit_mean = 0.75;
    strcpy(source.speculation_policy_identity,
           "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
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
    source.console.reasoning_policy = YVEX_REASONING_MAXIMUM;
    source.runtime.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.runtime.status = YVEX_SERVER_STATUS_READY;
    source.runtime.backend = YVEX_BACKEND_KIND_CUDA;
    strcpy(source.runtime.target_id, "deepseek4-v4-flash-dspark");
    strcpy(source.console.session_name, "main");
    strcpy(source.console.live_model_identity,
           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    strcpy(source.console.physical_variant_identity,
           "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    strcpy(source.console.selected_model_identity,
           "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    source.partial_turn.schema_version = YVEX_CLIENT_PARTIAL_TURN_SCHEMA_V1;
    source.partial_turn.available = 1;
    source.partial_turn.committed_progress = 1;
    source.partial_turn.reset_required = 1;
    source.partial_turn.failure_status = YVEX_ERR_BACKEND;
    source.partial_turn.failure_class = YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE;
    source.partial_turn.stop_reason = YVEX_CLIENT_STOP_MODEL_FAILURE;
    source.partial_turn.initial_position = 30u;
    source.partial_turn.final_committed_position = 37u;
    source.partial_turn.committed_token_count = 2u;
    source.partial_turn.published_text_bytes = 17u;
    source.partial_turn.target_state_generation = 8u;
    source.partial_turn.rng_generation = 3u;
    source.partial_turn.token_ledger_generation = 9u;
    source.partial_turn.message_history_generation = 4u;
    source.partial_turn.transcript_generation = 4u;
    strcpy(source.partial_turn.target_state_identity,
           "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
    strcpy(source.partial_turn.rng_state_identity,
           "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    strcpy(source.partial_turn.token_ledger_identity,
           "abababababababababababababababababababababababababababababababab");
    strcpy(source.partial_turn.published_text_identity,
           "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd");
    rc = yvex_protocol_message_encode(&source, frame, sizeof(frame), &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "console status encode");
    rc = yvex_protocol_message_decode(frame, count, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         decoded.kind == YVEX_CLIENT_MESSAGE_CONSOLE_STATUS &&
                         decoded.turn_count == 4u && decoded.context_used == 37u &&
                         decoded.kv_used_available &&
                         decoded.publication_timing_available &&
                         decoded.publication_seconds == 0.125 &&
                         decoded.reasoning_tokens == 7u &&
                         decoded.final_tokens == 11u &&
                         decoded.first_reasoning_seconds == 0.25 &&
                         decoded.first_final_seconds == 1.5 &&
                         decoded.reasoning_seconds == 1.25 &&
                         decoded.final_seconds == 2.75 &&
                         decoded.total_completion_seconds == 4.0 &&
                         decoded.reasoning_rate == 5.6 &&
                         decoded.final_rate == 4.0 &&
                         decoded.total_completion_rate == 4.5 &&
                         decoded.generation_phase == YVEX_CLIENT_PHASE_COMPLETE &&
                         decoded.cancellation_class == YVEX_CLIENT_CANCELLATION_COMPLETED &&
                         decoded.stream_channel == YVEX_CLIENT_STREAM_CONTROL_EVENT &&
                         decoded.generation_mode == YVEX_SERVER_GENERATION_DSPARK &&
                         decoded.draft_cycle_count == 3u &&
                         decoded.draft_forward_count == 3u &&
                         decoded.proposed_tokens == 16u &&
                         decoded.selected_verification_tokens == 15u &&
                         decoded.target_verification_count == 3u &&
                         decoded.accepted_draft_tokens == 10u &&
                         decoded.rejected_draft_tokens == 5u &&
                         decoded.discarded_draft_tokens == 1u &&
                         decoded.target_correction_or_bonus_tokens == 3u &&
                         decoded.maximum_accepted_prefix == 5u &&
                         decoded.confidence_logit_count == 16u &&
                         decoded.draft_seconds == 0.25 &&
                         decoded.verification_seconds == 0.75 &&
                         decoded.speculative_commit_seconds == 0.125 &&
                         decoded.confidence_logit_minimum == -1.25 &&
                         decoded.confidence_logit_maximum == 2.5 &&
                         decoded.confidence_logit_mean == 0.75 &&
                         decoded.effective_committed_rate == 4.5 &&
                         !strcmp(decoded.speculation_policy_identity,
                                 source.speculation_policy_identity) &&
                         decoded.partial_turn.available &&
                         decoded.partial_turn.reset_required &&
                         decoded.partial_turn.failure_status == YVEX_ERR_BACKEND &&
                         decoded.partial_turn.failure_class ==
                             YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE &&
                         decoded.partial_turn.final_committed_position == 37u &&
                         decoded.partial_turn.committed_token_count == 2u &&
                         decoded.partial_turn.message_history_generation == 4u &&
                         !strcmp(decoded.partial_turn.token_ledger_identity,
                                 source.partial_turn.token_ledger_identity),
                     "complete turn facts roundtrip");
    YVEX_TEST_ASSERT(decoded.console.runtime_ready &&
                         decoded.console.session_available &&
                         decoded.console.attached &&
                         decoded.console.cancel_requested &&
                         decoded.console.kv_used_available &&
                         decoded.console.progress_available &&
                         decoded.console.selected_model_available &&
                         decoded.console.explicit_reasoning_channel_supported &&
                         decoded.console.reasoning_policy ==
                             YVEX_REASONING_MAXIMUM &&
                         decoded.console.position == 37u &&
                         decoded.console.turn_count == 4u &&
                         decoded.console.context_capacity == 4096u &&
                         decoded.console.context_used == 37u &&
                         decoded.console.kv_used_bytes == 8192u &&
                         decoded.console.generation_phase == YVEX_CLIENT_PHASE_DECODE &&
                         decoded.console.cancellation_class == YVEX_CLIENT_CANCELLATION_REQUESTED &&
                         strcmp(decoded.console.session_name, "main") == 0 &&
                         strcmp(decoded.runtime.target_id, "deepseek4-v4-flash-dspark") == 0,
                     "console status facts roundtrip");
    rc = yvex_protocol_message_decode(frame, count - 1u, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "truncated message refuses");
    return 0;
}

typedef struct {
    int listener;
} v6_peer;

static void *v6_peer_main(void *opaque)
{
    static const unsigned char response[12] = {
        'Y', 'V', 'X', 'P', 0u, 6u, 0u, 2u, 0u, 0u, 0u, 0u};
    v6_peer *peer = opaque;
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

static int test_v6_frame_refusal(void)
{
    struct sockaddr_un address;
    char path[sizeof(address.sun_path)];
    yvex_client *client = NULL;
    yvex_error err;
    pthread_t thread;
    v6_peer peer;
    int rc;
    (void)snprintf(path, sizeof(path), "build/tests/protocol-v12-%lu.sock",
                   (unsigned long)getpid());
    (void)unlink(path);
    peer.listener = socket(AF_UNIX, SOCK_STREAM, 0);
    YVEX_TEST_ASSERT(peer.listener >= 0, "v6 peer socket");
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path) + 1u);
    YVEX_TEST_ASSERT(bind(peer.listener, (struct sockaddr *)&address,
                          sizeof(address)) == 0 &&
                         chmod(path, 0600) == 0 && listen(peer.listener, 1) == 0,
                     "v6 peer bind/listen");
    YVEX_TEST_ASSERT(pthread_create(&thread, NULL, v6_peer_main, &peer) == 0,
                     "v6 peer thread");
    rc = yvex_client_connect(&client, path, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && client == NULL &&
                         strstr(yvex_error_message(&err), "version 12") != NULL,
                     "v6 frame explicitly refuses");
    YVEX_TEST_ASSERT(pthread_join(thread, NULL) == 0, "v6 peer join");
    (void)close(peer.listener);
    (void)unlink(path);
    return 0;
}

static int test_capability_aware_readiness(void)
{
    yvex_client_message message = {0}, decoded;
    unsigned char frame[16384];
    unsigned long long count = 0ull;
    yvex_error err;

    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_STATUS;
    message.status = YVEX_OK;
    message.runtime.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.runtime.status = YVEX_SERVER_STATUS_READY;
    message.runtime.backend = YVEX_BACKEND_KIND_CUDA;
    message.runtime.generation_mode = YVEX_SERVER_GENERATION_MEDIA;
    message.runtime.runtime_ready = 1;
    message.runtime.generation_ready = 1;
    message.runtime.concurrent_sequences = 1ull;
    memset(message.runtime.runtime_model_identity, 'a', 64u);
    memset(message.runtime.physical_variant_identity, 'b', 64u);
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, count, &decoded, &err) == YVEX_OK &&
            decoded.runtime.runtime_ready &&
            !decoded.runtime.runtime_binding_identity[0] &&
            !decoded.runtime.artifact_identity[0] &&
            !decoded.runtime.capacity_plan_identity[0],
        "media readiness roundtrip requires only its admitted identities");
    message.runtime.capacity_required_bytes = 1ull;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "media readiness refuses an aliased text capacity plan");
    message.runtime.capacity_required_bytes = 0ull;
    memset(message.runtime.runtime_binding_identity, 'c', 64u);
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "media readiness refuses a fake runtime binding identity");
    message.runtime.runtime_binding_identity[0] = '\0';
    message.runtime.runtime_model_identity[0] = '\0';
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "media readiness refuses without a composite runtime model identity");
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

static int test_media_result_roundtrip(void)
{
    yvex_client_message message = {0}, decoded;
    unsigned char frame[16384];
    unsigned long long count = 0ull;
    yvex_error err;

    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_TURN_COMPLETE;
    message.status = YVEX_OK;
    message.generation_mode = YVEX_SERVER_GENERATION_MEDIA;
    message.generation_phase = YVEX_CLIENT_PHASE_COMPLETE;
    message.stop_reason = YVEX_CLIENT_STOP_EOS;
    message.session_state = YVEX_SERVER_SESSION_READY;
    message.media_result.schema_version = YVEX_CLIENT_MEDIA_RESULT_SCHEMA_V1;
    message.media_result.available = 1;
    strcpy(message.media_result.output_path, "/tmp/yvex-media-result.avi");
    message.media_result.width = 192ull;
    message.media_result.height = 192ull;
    message.media_result.frames = 124ull;
    message.media_result.fps_numerator = 24ull;
    message.media_result.fps_denominator = 1ull;
    message.media_result.audio_samples = 248000ull;
    message.media_result.audio_sample_rate = 48000ull;
    message.media_result.seed = 42ull;
    message.media_result.file_bytes = 123456ull;
    memset(message.media_result.preset_identity, 'a', 64u);
    memset(message.media_result.execution_identity, 'b', 64u);
    memset(message.media_result.file_identity, 'c', 64u);
    memset(message.media_result.publication_identity, 'd', 64u);
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, count, &decoded, &err) ==
                YVEX_OK &&
            decoded.media_result.available &&
            !strcmp(decoded.media_result.output_path,
                    message.media_result.output_path) &&
            decoded.media_result.width == 192ull &&
            decoded.media_result.height == 192ull &&
            decoded.media_result.frames == 124ull &&
            decoded.media_result.audio_samples == 248000ull &&
            decoded.media_result.seed == 42ull &&
            !strcmp(decoded.media_result.publication_identity,
                    message.media_result.publication_identity),
        "typed media result roundtrip preserves publication facts");
    message.media_result.output_path[0] = 'x';
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "typed media result refuses a relative output path");
    memset(&message.media_result, 0, sizeof(message.media_result));
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "successful media completion refuses without a typed media result");
    return 0;
}

int yvex_test_protocol(void)
{
    if (test_request_roundtrip() != 0) return 1;
    if (test_all_operation_roundtrips() != 0) return 1;
    if (test_schema_refusals() != 0) return 1;
    if (test_message_roundtrip() != 0) return 1;
    if (test_capability_aware_readiness() != 0) return 1;
    if (test_media_result_roundtrip() != 0) return 1;
    if (test_v6_frame_refusal() != 0) return 1;
    if (test_bounded_parser_mutation() != 0) return 1;
    return 0;
}
