/*
 * YVEX - bounded local protocol tests
 *
 * Purpose: prove field-wise request/message roundtrip, embedded bytes, version refusal,
 * malformed-frame refusal, and stable protocol identities without opening an engine.
 */
#include <stdlib.h>
#include <string.h>

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
    source.runtime.metrics.model_open_count = 1u;
    source.runtime.metrics.artifact_open_count = 1u;
    source.runtime.metrics.queue_capacity = 16u;
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
    rc = yvex_protocol_message_decode(frame, count - 1u, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "truncated message refuses");
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
    if (test_message_roundtrip() != 0) return 1;
    if (test_bounded_parser_mutation() != 0) return 1;
    return 0;
}
