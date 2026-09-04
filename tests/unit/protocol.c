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
    yvex_content_part *owned_content = NULL;
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
    source.media_condition_count = 2u;
    source.media_conditions[0].schema_version =
        YVEX_CLIENT_MEDIA_CONDITION_SCHEMA_V1;
    source.media_conditions[0].kind = YVEX_CLIENT_MEDIA_CONDITION_IMAGE;
    source.media_conditions[0].role = YVEX_CLIENT_MEDIA_CONDITION_FIRST;
    strcpy(source.media_conditions[0].source_path, "/tmp/first.png");
    source.media_conditions[1].schema_version =
        YVEX_CLIENT_MEDIA_CONDITION_SCHEMA_V1;
    source.media_conditions[1].kind = YVEX_CLIENT_MEDIA_CONDITION_IMAGE;
    source.media_conditions[1].role = YVEX_CLIENT_MEDIA_CONDITION_LAST;
    strcpy(source.media_conditions[1].source_path, "/tmp/last.png");
    source.media_execution.schema_version =
        YVEX_CLIENT_MEDIA_EXECUTION_SCHEMA_V1;
    source.media_execution.trajectory = YVEX_CLIENT_MEDIA_TRAJECTORY_RELEASED;
    source.media_execution.present = YVEX_CLIENT_MEDIA_EXECUTION_WIDTH |
                                     YVEX_CLIENT_MEDIA_EXECUTION_HEIGHT |
                                     YVEX_CLIENT_MEDIA_EXECUTION_DURATION |
                                     YVEX_CLIENT_MEDIA_EXECUTION_SEED;
    source.media_execution.width = 1344ull;
    source.media_execution.height = 768ull;
    source.media_execution.duration_milliseconds = 5000ull;
    source.media_execution.seed = 42ull;
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
                                      &owned_content, &owned_provider, &err);
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
    YVEX_TEST_ASSERT(
        decoded.media_condition_count == 2u &&
            decoded.media_conditions[0].role == YVEX_CLIENT_MEDIA_CONDITION_FIRST &&
            decoded.media_conditions[1].role == YVEX_CLIENT_MEDIA_CONDITION_LAST &&
            strcmp(decoded.media_conditions[0].source_path, "/tmp/first.png") == 0 &&
            strcmp(decoded.media_conditions[1].source_path, "/tmp/last.png") == 0,
        "typed first and last media conditions roundtrip");
    YVEX_TEST_ASSERT(
        decoded.media_execution.schema_version ==
                YVEX_CLIENT_MEDIA_EXECUTION_SCHEMA_V1 &&
            decoded.media_execution.trajectory ==
                YVEX_CLIENT_MEDIA_TRAJECTORY_RELEASED &&
            decoded.media_execution.width == 1344ull &&
            decoded.media_execution.height == 768ull &&
            decoded.media_execution.duration_milliseconds == 5000ull &&
            decoded.media_execution.seed == 42ull,
        "typed released media execution roundtrip");
    free(owned_prompt);
    yvex_content_parts_close(&owned_content, decoded.content_part_count);
    yvex_provider_request_close(&owned_provider);

    source.schema_version++;
    count = 99u;
    rc = yvex_protocol_request_encode(&source, frame, sizeof(frame), &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && count == 0u,
                     "unsupported request version refuses");
    source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.media_execution.present &= ~YVEX_CLIENT_MEDIA_EXECUTION_HEIGHT;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "media execution refuses an unpaired canvas axis");
    rc = yvex_protocol_request_decode(frame, 3u, &decoded, &owned_prompt,
                                      &owned_content, &owned_provider, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "truncated request refuses");
    return 0;
}

static int test_content_request_roundtrip(void)
{
    static const unsigned char transcript[] = "heard text";
    yvex_content_part source_parts[2] = {0}, *owned_content = NULL;
    yvex_client_request source = {0}, decoded;
    yvex_provider_request *owned_provider = NULL;
    unsigned char *owned_prompt = NULL;
    unsigned char frame[4096];
    unsigned long long count = 0u;
    char source_identity[YVEX_CONTENT_ID_CAP];
    yvex_error err;
    source_parts[0].schema_version = YVEX_CONTENT_PART_SCHEMA_V1;
    source_parts[0].kind = YVEX_CONTENT_AUDIO;
    source_parts[0].storage = YVEX_CONTENT_LOCAL_FILE;
    source_parts[0].byte_count = 7000000u;
    strcpy(source_parts[0].media_type, "audio/wav");
    strcpy(source_parts[0].reference, "/var/tmp/yvex-audio.wav");
    memset(source_parts[0].content_identity, 'a',
           sizeof(source_parts[0].content_identity) - 1u);
    source_parts[1].schema_version = YVEX_CONTENT_PART_SCHEMA_V1;
    source_parts[1].kind = YVEX_CONTENT_TEXT;
    source_parts[1].storage = YVEX_CONTENT_INLINE;
    source_parts[1].bytes = transcript;
    source_parts[1].byte_count = sizeof(transcript) - 1u;
    strcpy(source_parts[1].media_type, "text/plain;charset=utf-8");
    strcpy(source_parts[1].derived_from_content_identity,
           source_parts[0].content_identity);
    YVEX_TEST_ASSERT(
        yvex_content_part_seal(source_parts + 1u, &err) == YVEX_OK &&
            yvex_content_parts_identity(source_parts, 2u, source_identity,
                                        &err) == YVEX_OK,
        "multipart request fixture seals");
    source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.operation = YVEX_CLIENT_OP_GENERATION_TURN;
    source.request_number = 91u;
    strcpy(source.session_name, "multipart-session");
    source.content_parts = source_parts;
    source.content_part_count = 2u;
    source.temperature = 1.0;
    source.top_p = 1.0;
    source.typical_p = 1.0;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK &&
            count < sizeof(frame),
        "multipart request uses bounded reference transport");
    YVEX_TEST_ASSERT(
        yvex_protocol_request_decode(frame, count, &decoded, &owned_prompt,
                                     &owned_content, &owned_provider,
                                     &err) == YVEX_OK &&
            owned_prompt == NULL && owned_provider == NULL &&
            decoded.content_part_count == 2u &&
            decoded.content_parts == owned_content &&
            decoded.content_parts[0].byte_count == 7000000u &&
            decoded.content_parts[0].bytes == NULL &&
            !strcmp(decoded.content_parts[1].derived_from_content_identity,
                    source_parts[0].content_identity) &&
            !memcmp(decoded.content_parts[1].bytes, transcript,
                    sizeof(transcript) - 1u),
        "multipart request preserves order, payload, and provenance");
    {
        char decoded_identity[YVEX_CONTENT_ID_CAP];
        YVEX_TEST_ASSERT(
            yvex_content_parts_identity(decoded.content_parts,
                                        decoded.content_part_count,
                                        decoded_identity, &err) == YVEX_OK &&
                !strcmp(decoded_identity, source_identity),
            "multipart request identity survives protocol boundary");
    }
    yvex_content_parts_close(&owned_content, decoded.content_part_count);
    source.prompt = transcript;
    source.prompt_bytes = sizeof(transcript) - 1u;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "legacy prompt and typed content are mutually exclusive");
    return 0;
}

static int test_all_operation_roundtrips(void)
{
    unsigned char frame[2048];
    yvex_client_request source, decoded;
    yvex_provider_request *provider = NULL;
    yvex_content_part *content = NULL;
    unsigned char *prompt = NULL;
    unsigned long long count;
    yvex_error err;
    unsigned int value;
    for (value = YVEX_CLIENT_OP_HANDSHAKE;
         value <= YVEX_CLIENT_OP_ENGINE_LEASE_RELEASE; ++value) {
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
        if (source.operation == YVEX_CLIENT_OP_ENGINE_LEASE_RELEASE)
            memset(source.model_lease_identity, 'a',
                   sizeof(source.model_lease_identity) - 1u);
        count = 0u;
        YVEX_TEST_ASSERT(
            yvex_protocol_request_encode(&source, frame, sizeof(frame), &count,
                                         &err) == YVEX_OK,
            "all current protocol operations encode");
        YVEX_TEST_ASSERT(
            yvex_protocol_request_decode(frame, count, &decoded, &prompt,
                                         &content, &provider, &err) == YVEX_OK &&
                decoded.operation == (yvex_client_operation)value,
            "all current protocol operations decode");
        if (source.operation == YVEX_CLIENT_OP_SESSION_FORK)
            YVEX_TEST_ASSERT(
                strcmp(decoded.fork_session_name, "child") == 0 &&
                    decoded.maximum_prefix_bytes == 8192u,
                "session fork fields roundtrip");
        free(prompt);
        prompt = NULL;
        yvex_content_parts_close(&content, decoded.content_part_count);
        yvex_provider_request_close(&provider);
    }
    return 0;
}

static int test_schema_refusals(void)
{
    unsigned char frame[4096], malformed[4096];
    unsigned char *prompt = NULL;
    yvex_content_part *content = NULL;
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
                                     &prompt, &content, &provider,
                                     &err) == YVEX_ERR_FORMAT,
        "unknown request operation refuses on decode");
    memcpy(malformed, frame, (size_t)count);
    malformed[2] = 1u;
    YVEX_TEST_ASSERT(
        yvex_protocol_request_decode(malformed, count, &decoded_request,
                                     &prompt, &content, &provider,
                                     &err) == YVEX_ERR_FORMAT,
        "nonzero TLV reserved field refuses");
    memcpy(malformed, frame, (size_t)count);
    malformed[count] = 0u;
    malformed[count + 1u] = 255u;
    memset(malformed + count + 2u, 0, 6u);
    YVEX_TEST_ASSERT(
        yvex_protocol_request_decode(malformed, count + 8u, &decoded_request,
                                     &prompt, &content, &provider,
                                     &err) == YVEX_ERR_FORMAT,
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
    request.media_condition_count = 1u;
    request.media_conditions[0].schema_version =
        YVEX_CLIENT_MEDIA_CONDITION_SCHEMA_V1;
    request.media_conditions[0].kind = YVEX_CLIENT_MEDIA_CONDITION_IMAGE;
    request.media_conditions[0].role = YVEX_CLIENT_MEDIA_CONDITION_FIRST;
    strcpy(request.media_conditions[0].source_path, "/tmp/first.png");
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&request, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "non-generation operations refuse media conditions");
    request.operation = YVEX_CLIENT_OP_GENERATION_TURN;
    request.media_condition_count = 2u;
    request.media_conditions[1] = request.media_conditions[0];
    strcpy(request.media_conditions[1].source_path, "/tmp/duplicate.png");
    YVEX_TEST_ASSERT(
        yvex_protocol_request_encode(&request, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "duplicate media condition roles refuse");
    request.media_condition_count = 0u;
    memset(request.media_conditions, 0, sizeof(request.media_conditions));
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
    message.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    message.execution_strategy = YVEX_SERVER_EXECUTION_SPECULATIVE;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "media engine kind refuses a text execution strategy");
    message.engine_kind = YVEX_SERVER_ENGINE_TEXT;
    message.execution_strategy = YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "text engine kind requires an explicit execution strategy");

    memset(&message, 0, sizeof(message));
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_ACK;
    message.engine_kind = YVEX_SERVER_ENGINE_TEXT;
    message.execution_strategy = YVEX_SERVER_EXECUTION_SPECULATIVE;
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
    message.engine_kind = YVEX_SERVER_ENGINE_TEXT;
    message.execution_strategy = YVEX_SERVER_EXECUTION_SPECULATIVE;
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
    source.runtime.schema_version = YVEX_SERVER_SUMMARY_SCHEMA_V2;
    source.runtime.metrics.schema_version = YVEX_RUNTIME_METRICS_SCHEMA_VERSION;
    source.runtime.metrics.resources.schema_version =
        YVEX_EXECUTION_RESOURCE_SCHEMA_V1;
    source.runtime.metrics.resources.available =
        YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE;
    source.runtime.status = YVEX_SERVER_STATUS_READY;
    strcpy(source.runtime.socket_path, "/tmp/yvex.sock");
    source.runtime.host_ready = 1;
    source.runtime.openai_listener_enabled = 1;
    source.runtime.openai_listener_ready = 1;
    source.runtime.openai_port = 8001u;
    source.runtime.session_count = 3u;
    source.runtime.request_count = 11u;
    source.runtime.request_queue_capacity = 16u;
    source.runtime.worker_count = 4u;
    source.runtime.engine_count = 2u;
    source.runtime.loaded_engine_count = 1u;
    source.runtime.draining_engine_count = 1u;
    source.runtime.maximum_engines = 8u;
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
    YVEX_TEST_ASSERT(decoded.runtime.host_ready &&
                         !strcmp(decoded.runtime.socket_path, "/tmp/yvex.sock"),
                     "host readiness and endpoint roundtrip");
    YVEX_TEST_ASSERT(decoded.runtime.metrics.model_open_count == 1u,
                     "model-open metric");
    YVEX_TEST_ASSERT(decoded.runtime.metrics.queue_capacity == 16u,
                     "queue-capacity metric");
    YVEX_TEST_ASSERT(decoded.runtime.openai_listener_enabled &&
                         decoded.runtime.openai_listener_ready &&
                         decoded.runtime.openai_port == 8001u,
                     "integrated OpenAI listener roundtrip");
    YVEX_TEST_ASSERT(decoded.runtime.session_count == 3u &&
                         decoded.runtime.request_count == 11u &&
                         decoded.runtime.request_queue_capacity == 16u &&
                         decoded.runtime.worker_count == 4u &&
                         decoded.runtime.engine_count == 2u &&
                         decoded.runtime.loaded_engine_count == 1u &&
                         decoded.runtime.draining_engine_count == 1u &&
                         decoded.runtime.maximum_engines == 8u &&
                         decoded.runtime.openai_timeout_ms == 600000u &&
                         decoded.runtime.trace_level == YVEX_SERVER_TRACE_STAGES,
                     "host configuration and engine inventory roundtrip");
    YVEX_TEST_ASSERT(decoded.runtime.metrics.active_http_requests == 2u &&
                         decoded.runtime.metrics.completed_http_requests == 7u &&
                         decoded.runtime.metrics.failed_http_requests == 3u &&
                         decoded.runtime.metrics.cancelled_http_requests == 1u,
                     "integrated HTTP metrics roundtrip");
    source.runtime.host_ready = 2;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "host status refuses a nonboolean readiness fact");
    source.runtime.host_ready = 1;
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
    source.kind = YVEX_CLIENT_MESSAGE_TURN_STARTED;
    source.status = YVEX_OK;
    source.generation_phase = YVEX_CLIENT_PHASE_TOKENIZING;
    source.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
    source.initial_position = 41u;
    source.requested_maximum_new_tokens = 17u;
    source.resolved_maximum_new_tokens = 9u;
    source.output_limit_explicit = 1;
    source.content_part_count = 3u;
    memset(source.input_content_identity, 'c',
           sizeof(source.input_content_identity) - 1u);
    rc = yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                      &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "turn envelope message encode");
    rc = yvex_protocol_message_decode(frame, count, &decoded, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && decoded.initial_position == 41u &&
            decoded.requested_maximum_new_tokens == 17u &&
            decoded.resolved_maximum_new_tokens == 9u &&
            decoded.output_limit_explicit &&
            decoded.content_part_count == 3u &&
            !strcmp(decoded.input_content_identity,
                    source.input_content_identity),
        "turn envelope ownership roundtrip");
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
    source.engine_kind = YVEX_SERVER_ENGINE_TEXT;
    source.execution_strategy = YVEX_SERVER_EXECUTION_SPECULATIVE;
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
    source.console.schema_version = YVEX_CONSOLE_STATUS_SCHEMA_V1;
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
    source.runtime.schema_version = YVEX_SERVER_SUMMARY_SCHEMA_V2;
    source.runtime.metrics.schema_version = YVEX_RUNTIME_METRICS_SCHEMA_VERSION;
    source.runtime.metrics.resources.schema_version =
        YVEX_EXECUTION_RESOURCE_SCHEMA_V1;
    source.runtime.metrics.resources.available =
        YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE;
    source.runtime.status = YVEX_SERVER_STATUS_READY;
    source.runtime.host_ready = 1;
    source.runtime.maximum_engines = 4u;
    source.runtime.worker_count = 1u;
    strcpy(source.console.model_alias, "deepseek");
    source.console.engine_generation = 7ull;
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
                         decoded.engine_kind == YVEX_SERVER_ENGINE_TEXT &&
                         decoded.execution_strategy ==
                             YVEX_SERVER_EXECUTION_SPECULATIVE &&
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
                         strcmp(decoded.console.model_alias, "deepseek") == 0 &&
                         decoded.console.engine_generation == 7ull &&
                         decoded.runtime.host_ready,
                     "console status facts roundtrip");
    rc = yvex_protocol_message_decode(frame, count - 1u, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "truncated message refuses");
    return 0;
}

typedef struct {
    int listener;
} stale_peer;

static void *stale_peer_main(void *opaque)
{
    static const unsigned char response[12] = {
        'Y', 'V', 'X', 'P', 0u, 19u, 0u, 2u, 0u, 0u, 0u, 0u};
    stale_peer *peer = opaque;
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

static int test_stale_frame_refusal(void)
{
    struct sockaddr_un address;
    char path[sizeof(address.sun_path)];
    yvex_client *client = NULL;
    yvex_error err;
    pthread_t thread;
    stale_peer peer;
    int rc;
    (void)snprintf(path, sizeof(path), "build/tests/protocol-current-%lu.sock",
                   (unsigned long)getpid());
    (void)unlink(path);
    peer.listener = socket(AF_UNIX, SOCK_STREAM, 0);
    YVEX_TEST_ASSERT(peer.listener >= 0, "stale peer socket");
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path) + 1u);
    YVEX_TEST_ASSERT(bind(peer.listener, (struct sockaddr *)&address,
                          sizeof(address)) == 0 &&
                         chmod(path, 0600) == 0 && listen(peer.listener, 1) == 0,
                     "stale peer bind/listen");
    YVEX_TEST_ASSERT(pthread_create(&thread, NULL, stale_peer_main, &peer) == 0,
                     "stale peer thread");
    rc = yvex_client_connect(&client, path, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && client == NULL &&
                         strstr(yvex_error_message(&err), "version 20") != NULL,
                     "immediately prior v19 frame explicitly refuses");
    YVEX_TEST_ASSERT(pthread_join(thread, NULL) == 0, "stale peer join");
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
    message.kind = YVEX_CLIENT_MESSAGE_ENGINE;
    message.status = YVEX_OK;
    message.engine.schema_version = YVEX_SERVER_ENGINE_SCHEMA_CURRENT;
    message.engine.state = YVEX_SERVER_ENGINE_LOADED;
    message.engine.backend = YVEX_BACKEND_KIND_CUDA;
    message.engine.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    message.engine.execution_strategy =
        YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    strcpy(message.engine.alias, "minimax");
    strcpy(message.engine.target_id, "minimax-h3");
    message.engine.generation = 7ull;
    message.engine.maximum_sessions = 2ull;
    message.engine.concurrent_sequences = 1ull;
    message.engine.execution_ready = 1;
    message.engine.capacity.schema_version =
        YVEX_EXECUTION_CAPACITY_SCHEMA_V1;
    message.engine.capacity.session_capacity = 2ull;
    message.engine.capacity.runnable_work_capacity = 2ull;
    message.engine.capacity.physical_sequence_width = 1ull;
    message.engine.capacity.cooperative_scheduling_ready = 1;
    message.engine.capabilities.schema_version =
        YVEX_MODEL_CAPABILITY_SCHEMA_V1;
    message.engine.capabilities.input_kinds =
        YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_TEXT) |
        YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_IMAGE);
    message.engine.capabilities.output_kinds =
        YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_AUDIO) |
        YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_VIDEO);
    message.engine.capabilities.execution_properties =
        YVEX_MODEL_CAPABILITY_DEMAND_ACTIVATION;
    message.engine.capabilities.maximum_input_parts = 3u;
    message.engine.resources.schema_version =
        YVEX_EXECUTION_RESOURCE_SCHEMA_V1;
    message.engine.resources.placement =
        YVEX_EXECUTION_PLACEMENT_COMPOSITE;
    message.engine.resources.available =
        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE |
        YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE;
    message.engine.resources.component_count = 4ull;
    memset(message.engine.runtime_model_identity, 'a', 64u);
    memset(message.engine.specialization_identity, 'b', 64u);
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, count, &decoded, &err) == YVEX_OK &&
            decoded.engine.execution_ready &&
            decoded.engine.generation == 7ull &&
            !strcmp(decoded.engine.alias, "minimax") &&
            !decoded.engine.runtime_binding_identity[0] &&
            !decoded.engine.artifact_identity[0] &&
            !decoded.engine.capacity_plan_identity[0] &&
            decoded.engine.capacity.runnable_work_capacity == 2ull &&
            decoded.engine.capacity.physical_sequence_width == 1ull &&
            !decoded.engine.capacity.continuous_batching_ready &&
            decoded.engine.resources.component_count == 4ull &&
            decoded.engine.capabilities.input_kinds ==
                message.engine.capabilities.input_kinds &&
            decoded.engine.capabilities.output_kinds ==
                message.engine.capabilities.output_kinds &&
            decoded.engine.capabilities.maximum_input_parts == 3u,
        "media engine readiness roundtrip requires only admitted engine identities");
    message.engine.concurrent_sequences = 0ull;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "engine readiness refuses zero executable concurrency");
    message.engine.concurrent_sequences = 1ull;
    message.engine.continuous_batching_ready = 1;
    message.engine.capacity.continuous_batching_ready = 1;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "physical width one cannot claim continuous batching");
    message.engine.continuous_batching_ready = 0;
    message.engine.capacity.continuous_batching_ready = 0;
    message.engine.specialization_identity[0] = '\0';
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "loaded engine readiness refuses without a specialization identity");
    memset(message.engine.specialization_identity, 'b', 64u);
    message.engine.runtime_model_identity[0] = '\0';
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "loaded engine readiness refuses without a runtime model identity");
    return 0;
}

static int test_execution_truth_roundtrip(void)
{
    yvex_client_message source = {0}, decoded;
    yvex_execution_measurement *measurement = &source.measurement;
    yvex_execution_resource_summary *resource = &source.runtime.metrics.resources;
    unsigned char frame[16384];
    unsigned long long count = 0ull;
    yvex_error err;
    source.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    source.kind = YVEX_CLIENT_MESSAGE_ACK;
    measurement->schema_version = YVEX_EXECUTION_MEASUREMENT_SCHEMA_V1;
    measurement->scope = YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE;
    measurement->clock = YVEX_EXECUTION_CLOCK_HOST_WALL;
    measurement->composition = YVEX_EXECUTION_COMPOSITION_NESTED;
    measurement->work_unit = YVEX_EXECUTION_WORK_TOKENS;
    measurement->available =
        YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE |
        YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE |
        YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE;
    measurement->completed_units = 96ull;
    measurement->duration_ns = 48000000000ull;
    measurement->rolling_units = 32ull;
    measurement->rolling_duration_ns = 32000000000ull;
    measurement->rolling_window_units = 32ull;
    measurement->cumulative_rate = 2.0;
    measurement->rolling_rate = 1.0;
    resource->schema_version = YVEX_EXECUTION_RESOURCE_SCHEMA_V1;
    resource->placement =
        YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED_DEVICE_ADDRESSABLE;
    resource->available = YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE |
                          YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE |
                          YVEX_EXECUTION_RESOURCE_UNIFIED_MEMORY;
    resource->component_count = 1ull;
    resource->model_artifact_bytes = 64ull << 30u;
    resource->model_mapped_bytes = 64ull << 30u;
    resource->model_device_addressable_bytes = 64ull << 30u;
    resource->process_rss_current_bytes = 48ull << 30u;
    resource->process_rss_peak_bytes = 52ull << 30u;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, count, &decoded, &err) ==
                YVEX_OK &&
            decoded.measurement.scope ==
                YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE &&
            decoded.measurement.rolling_window_units == 32ull &&
            decoded.measurement.cumulative_rate == 2.0 &&
            decoded.measurement.rolling_rate == 1.0 &&
            decoded.runtime.metrics.resources.model_explicit_device_bytes ==
                0ull &&
            decoded.runtime.metrics.resources.model_device_addressable_bytes ==
                (64ull << 30u) &&
            !(decoded.runtime.metrics.resources.available &
              YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE),
        "scoped rates and UMA addressability roundtrip without fabricated residency");
    measurement->work_unit = YVEX_EXECUTION_WORK_FRAMES;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, count, &decoded, &err) ==
                YVEX_OK &&
            decoded.measurement.work_unit == YVEX_EXECUTION_WORK_FRAMES,
        "media frame work units cross the generic measurement protocol");
    measurement->work_unit = YVEX_EXECUTION_WORK_SAMPLES;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, count, &decoded, &err) ==
                YVEX_OK &&
            decoded.measurement.work_unit == YVEX_EXECUTION_WORK_SAMPLES,
        "media sample work units cross the generic measurement protocol");
    measurement->work_unit = YVEX_EXECUTION_WORK_TOKENS;
    measurement->available &=
        ~YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "rolling facts require an explicit rolling-rate availability claim");
    measurement->available |=
        YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE;
    measurement->scope = YVEX_EXECUTION_SCOPE_UNAVAILABLE;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "measurement facts require an explicit execution scope");
    measurement->scope = YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE;
    measurement->clock = YVEX_EXECUTION_CLOCK_UNAVAILABLE;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "measured duration requires an explicit clock owner");
    measurement->clock = YVEX_EXECUTION_CLOCK_HOST_WALL;
    measurement->rolling_units = measurement->completed_units + 1ull;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "rolling progress cannot exceed cumulative completed work");
    measurement->rolling_units = 32ull;
    resource->available &= ~YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "model bytes require an explicit model-resource availability claim");
    resource->available |= YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE;
    resource->model_explicit_device_bytes = 1ull;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "UMA addressability cannot be promoted to explicit device allocation");
    resource->model_explicit_device_bytes = 0ull;
    resource->available |= YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE;
    resource->session_attention_allocated_bytes = 8ull;
    resource->session_attention_resident_bytes = 8ull;
    resource->session_physical_state_bytes = 4ull;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "typed session state cannot exceed physical session ownership");
    resource->session_physical_state_bytes = 8ull;
    resource->available |= YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE;
    resource->workspace_current_bytes = 16ull;
    resource->workspace_peak_bytes = 15ull;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&source, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "resource current bytes cannot exceed their measured peak");
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
        yvex_content_part *content = NULL;
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
            bytes, count, &request, &prompt, &content, &provider, &err);
        YVEX_TEST_ASSERT(request_rc == YVEX_OK || request_rc < YVEX_OK,
                         "request mutation returns typed status");
        free(prompt);
        yvex_content_parts_close(
            &content, request_rc == YVEX_OK ? request.content_part_count : 0u);
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
    message.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    message.execution_strategy = YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    message.generation_phase = YVEX_CLIENT_PHASE_COMPLETE;
    message.stop_reason = YVEX_CLIENT_STOP_EOS;
    message.session_state = YVEX_SERVER_SESSION_READY;
    message.media_result.schema_version = YVEX_CLIENT_MEDIA_RESULT_SCHEMA_V2;
    message.media_result.available = 1;
    strcpy(message.media_result.output_path, "/tmp/yvex-media-result.avi");
    message.media_result.width = 192ull;
    message.media_result.height = 192ull;
    message.media_result.frames = 124ull;
    message.media_result.fps_numerator = 24ull;
    message.media_result.fps_denominator = 1ull;
    message.media_result.duration_milliseconds = 5166ull;
    message.media_result.audio_samples = 248000ull;
    message.media_result.audio_sample_rate = 48000ull;
    message.media_result.seed = 42ull;
    message.media_result.model_evaluations = 49ull;
    message.media_result.engine_generation = 3ull;
    message.media_result.task = YVEX_CLIENT_MEDIA_TASK_FIRST_LAST;
    message.media_result.condition_count = 2ull;
    message.media_result.file_bytes = 123456ull;
    memset(message.media_result.preset_identity, 'a', 64u);
    memset(message.media_result.trajectory_identity, 'b', 64u);
    memset(message.media_result.rng_identity, 'c', 64u);
    memset(message.media_result.plan_identity, 'd', 64u);
    memset(message.media_result.execution_identity, 'e', 64u);
    memset(message.media_result.file_identity, 'f', 64u);
    memset(message.media_result.publication_identity, '1', 64u);
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
            decoded.media_result.model_evaluations == 49ull &&
            decoded.media_result.task == YVEX_CLIENT_MEDIA_TASK_FIRST_LAST &&
            !strcmp(decoded.media_result.trajectory_identity,
                    message.media_result.trajectory_identity) &&
            !strcmp(decoded.media_result.publication_identity,
                    message.media_result.publication_identity),
        "typed media result roundtrip preserves publication facts");
    message.media_result.condition_count = 1ull;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&message, frame, sizeof(frame), &count,
                                     &err) == YVEX_ERR_INVALID_ARG,
        "typed media result refuses a task-condition mismatch");
    message.media_result.condition_count = 2ull;
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
    if (test_content_request_roundtrip() != 0) return 1;
    if (test_all_operation_roundtrips() != 0) return 1;
    if (test_schema_refusals() != 0) return 1;
    if (test_message_roundtrip() != 0) return 1;
    if (test_capability_aware_readiness() != 0) return 1;
    if (test_execution_truth_roundtrip() != 0) return 1;
    if (test_media_result_roundtrip() != 0) return 1;
    if (test_stale_frame_refusal() != 0) return 1;
    if (test_bounded_parser_mutation() != 0) return 1;
    return 0;
}
