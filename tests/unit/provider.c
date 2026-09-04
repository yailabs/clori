#define _POSIX_C_SOURCE 200809L
/*
 * Exercises bounded request identity, owned clone/wire roundtrip, JSON/tool refusal, and
 * authoritative output mutation detection without HTTP or model execution.
 */
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <yvex/content.h>
#include <yvex/provider.h>

#include "tests/test.h"

/* Construct one exact provider request used by identity and wire tests. */
static int request_build(yvex_provider_request *request,
                         yvex_provider_message messages[2],
                         yvex_provider_function_tool tools[1],
                         yvex_provider_span stops[1], yvex_error *err)
{
    static const unsigned char system_text[] = "Answer with the admitted tool when needed.";
    static const unsigned char user_text[] = "Get match m1.";
    static const unsigned char description[] = "Load one deterministic match fixture.";
    static const unsigned char schema[] =
        "{\"type\":\"object\",\"properties\":{\"match_id\":{\"type\":\"string\"}},"
        "\"required\":[\"match_id\"],\"additionalProperties\":false}";
    static const unsigned char stop[] = "STOP";
    memset(request, 0, sizeof(*request));
    memset(messages, 0, 2u * sizeof(*messages));
    memset(tools, 0, sizeof(*tools));
    memset(stops, 0, sizeof(*stops));
    messages[0].role = YVEX_PROVIDER_ROLE_SYSTEM;
    messages[0].content.bytes = system_text;
    messages[0].content.count = sizeof(system_text) - 1u;
    messages[1].role = YVEX_PROVIDER_ROLE_USER;
    messages[1].content.bytes = user_text;
    messages[1].content.count = sizeof(user_text) - 1u;
    strcpy(tools[0].name, "get_match_context");
    tools[0].description.bytes = description;
    tools[0].description.count = sizeof(description) - 1u;
    tools[0].description_present = 1;
    tools[0].parameters_json.bytes = schema;
    tools[0].parameters_json.count = sizeof(schema) - 1u;
    stops[0].bytes = stop;
    stops[0].count = sizeof(stop) - 1u;
    request->schema_version = YVEX_PROVIDER_SCHEMA_V4;
    strcpy(request->model, "deepseek4-v4-flash-dspark");
    request->messages = messages;
    request->message_count = 2u;
    request->tools = tools;
    request->tool_count = 1u;
    request->stop_strings = stops;
    request->stop_count = 1u;
    request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_AUTO;
    request->response_format = YVEX_PROVIDER_RESPONSE_JSON_OBJECT;
    request->reasoning_policy = YVEX_REASONING_ENABLED;
    request->reasoning_history_policy = YVEX_REASONING_HISTORY_DROP;
    request->drop_thinking = 1;
    request->sampling.stochastic = 1;
    request->sampling.seed_present = 1;
    request->sampling.seed = 42u;
    request->sampling.temperature = 0.8;
    request->sampling.top_k = 50u;
    request->sampling.top_p = 0.95;
    request->sampling.typical_p = 1.0;
    request->maximum_output_tokens = 64u;
    request->stream = 1;
    request->include_usage = 1;
    strcpy(request->adapter, "openai");
    strcpy(request->external_correlation_id, "req-test-1");
    return yvex_provider_request_seal(request, err);
}

/* Prove canonical identity and complete owned wire reconstruction. */
static int test_request_roundtrip(void)
{
    yvex_provider_request source, *clone = NULL, *decoded = NULL;
    yvex_provider_message messages[2];
    yvex_provider_function_tool tools[1];
    yvex_provider_span stops[1];
    unsigned char *wire = malloc(YVEX_PROVIDER_WIRE_MAX_BYTES);
    unsigned long long count = 0u;
    yvex_error err;
    int rc;
    YVEX_TEST_ASSERT(wire != NULL, "provider wire allocation");
    rc = request_build(&source, messages, tools, stops, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && source.sealed, "provider request seal");
    rc = yvex_provider_request_clone(&source, &clone, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && clone != NULL, "provider clone");
    YVEX_TEST_ASSERT_STREQ(clone->request_identity, source.request_identity,
                           "provider clone identity");
    rc = yvex_provider_request_wire_encode(
        clone, wire, YVEX_PROVIDER_WIRE_MAX_BYTES, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count > 0u, "provider wire encode");
    rc = yvex_provider_request_wire_decode(wire, count, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && decoded != NULL, "provider wire decode");
    YVEX_TEST_ASSERT_STREQ(decoded->request_identity, source.request_identity,
                           "provider wire identity");
    YVEX_TEST_ASSERT(decoded->message_count == 2u && decoded->tool_count == 1u,
                     "provider wire counts");
    YVEX_TEST_ASSERT(decoded->schema_version == YVEX_PROVIDER_SCHEMA_V4 &&
                         decoded->reasoning_policy == YVEX_REASONING_ENABLED &&
                         decoded->reasoning_history_policy ==
                             YVEX_REASONING_HISTORY_DROP &&
                         decoded->drop_thinking,
                     "provider v4 reasoning and history policy roundtrip");
    YVEX_TEST_ASSERT_STREQ(decoded->tools[0].name, "get_match_context",
                           "provider wire tool");
    YVEX_TEST_ASSERT_STREQ(decoded->adapter, "openai",
                           "provider adapter roundtrip");
    wire[count - 1u] ^= 1u;
    yvex_provider_request_close(&decoded);
    rc = yvex_provider_request_wire_decode(wire, count, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && decoded == NULL,
                     "wire identity mutation refuses");
    yvex_provider_request_close(&clone);
    free(wire);
    return 0;
}

static int test_refusal(void)
{
    yvex_provider_request request;
    yvex_provider_message messages[2];
    yvex_provider_function_tool tools[1];
    yvex_provider_tool_call calls[2] = {0};
    yvex_provider_span stops[1];
    static const unsigned char malformed[] = "{\"type\":}";
    static const unsigned char arguments[] = "{}";
    yvex_error err;
    int rc = request_build(&request, messages, tools, stops, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "provider refusal fixture");
    tools[0].parameters_json.bytes = malformed;
    tools[0].parameters_json.count = sizeof(malformed) - 1u;
    rc = yvex_provider_request_seal(&request, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !request.sealed,
                     "malformed tool schema refuses");

    rc = request_build(&request, messages, tools, stops, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "provider duplicate-call fixture");
    messages[1].role = YVEX_PROVIDER_ROLE_ASSISTANT;
    calls[0].arguments_json = (yvex_provider_span){arguments,
                                                   sizeof(arguments) - 1u};
    calls[1].arguments_json = calls[0].arguments_json;
    strcpy(calls[0].call_id, "call_duplicate");
    strcpy(calls[1].call_id, "call_duplicate");
    strcpy(calls[0].name, "get_match_context");
    strcpy(calls[1].name, "get_match_context");
    messages[1].tool_calls = calls;
    messages[1].tool_call_count = 2u;
    rc = yvex_provider_request_seal(&request, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !request.sealed,
                     "duplicate assistant tool-call IDs refuse");

    strcpy(calls[1].call_id, "call_second");
    request.schema_version = YVEX_PROVIDER_SCHEMA_V2;
    request.reasoning_history_policy =
        YVEX_REASONING_HISTORY_SOURCE_DEFAULT;
    rc = yvex_provider_request_seal(&request, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && request.sealed,
                     "provider v2 remains readable and writable");
    request.schema_version = YVEX_PROVIDER_SCHEMA_V1;
    request.reasoning_policy = YVEX_REASONING_DISABLED;
    rc = yvex_provider_request_seal(&request, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !request.sealed,
                     "provider v1 retains its single-call contract");
    return 0;
}

static int test_output_identity(void)
{
    static const unsigned char bytes[] = "ok";
    yvex_provider_output output;
    yvex_error err;
    int rc;
    memset(&output, 0, sizeof(output));
    output.schema_version = YVEX_PROVIDER_SCHEMA_V1;
    output.kind = YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT;
    output.finish = YVEX_PROVIDER_FINISH_STOP;
    output.bytes.bytes = bytes;
    output.bytes.count = sizeof(bytes) - 1u;
    output.prompt_tokens = 3u;
    output.completion_tokens = 2u;
    output.total_tokens = 5u;
    memset(output.request_identity, 'a', YVEX_PROVIDER_ID_CAP - 1u);
    output.request_identity[YVEX_PROVIDER_ID_CAP - 1u] = '\0';
    rc = yvex_provider_output_seal(&output, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "provider output seal");
    output.completion_tokens++;
    rc = yvex_provider_output_validate(&output, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE, "provider output mutation refuses");
    return 0;
}

static int test_request_defaults(void)
{
    yvex_provider_request request;
    yvex_provider_sampling sampling;

    memset(&request, 0xa5, sizeof(request));
    yvex_provider_request_default(&request);
    YVEX_TEST_ASSERT(request.schema_version == YVEX_PROVIDER_SCHEMA_V4,
                     "provider default schema");
    YVEX_TEST_ASSERT(request.response_format == YVEX_PROVIDER_RESPONSE_TEXT,
                     "provider default response format");
    YVEX_TEST_ASSERT(request.maximum_output_tokens == 0u,
                     "provider default adaptive output limit");
    YVEX_TEST_ASSERT(request.tool_choice.kind == YVEX_PROVIDER_TOOL_CHOICE_AUTO,
                     "provider default tool choice");
    YVEX_TEST_ASSERT(
        request.reasoning_policy == YVEX_REASONING_SOURCE_DEFAULT &&
            request.reasoning_history_policy ==
                YVEX_REASONING_HISTORY_SOURCE_DEFAULT &&
            !request.drop_thinking,
        "provider defaults defer reasoning and history to source policy");
    YVEX_TEST_ASSERT(!request.sampling.stochastic && !request.sampling.seed_present,
                     "provider default deterministic sampling");
    YVEX_TEST_ASSERT(request.sampling.temperature == 1.0 &&
                         request.sampling.top_p == 1.0 &&
                         request.sampling.typical_p == 1.0,
                     "provider default probability policy");
    YVEX_TEST_ASSERT(request.sampling.top_k == 0u && request.sampling.min_p == 0.0,
                     "provider default filters");
    YVEX_TEST_ASSERT(request.messages == NULL && request.tools == NULL &&
                         request.stop_strings == NULL && !request.sealed,
                     "provider defaults own no payload");

    memset(&sampling, 0xa5, sizeof(sampling));
    yvex_provider_sampling_default(&sampling);
    YVEX_TEST_ASSERT(!sampling.stochastic && !sampling.seed_present &&
                         sampling.temperature == 1.0 && sampling.top_p == 1.0 &&
                         sampling.typical_p == 1.0,
                     "sampling default constructor");
    yvex_provider_request_default(NULL);
    yvex_provider_sampling_default(NULL);
    return 0;
}

static int test_ordered_content(void)
{
    static const unsigned char audio_bytes[] = {
        'R', 'I', 'F', 'F', 4u, 0u, 0u, 0u, 'W', 'A', 'V', 'E'};
    static const unsigned char transcript[] = "spoken words";
    yvex_content_part source[2] = {0}, reversed[2], *decoded = NULL;
    yvex_model_capability_summary capability = {0};
    unsigned char *wire = malloc(YVEX_CONTENT_WIRE_MAX_BYTES);
    unsigned long long wire_bytes = 0u, decoded_count = 0u;
    char identity[YVEX_CONTENT_ID_CAP], reversed_identity[YVEX_CONTENT_ID_CAP];
    char path[] = "/tmp/yvex-content-XXXXXX";
    yvex_error err;
    int fd = mkstemp(path), rc;
    YVEX_TEST_ASSERT(fd >= 0 && wire != NULL, "content fixture allocation");
    YVEX_TEST_ASSERT(write(fd, audio_bytes, sizeof(audio_bytes)) ==
                         (ssize_t)sizeof(audio_bytes) && close(fd) == 0,
                     "content fixture publication");
    source[0].schema_version = YVEX_CONTENT_PART_SCHEMA_V1;
    source[0].kind = YVEX_CONTENT_AUDIO;
    source[0].storage = YVEX_CONTENT_LOCAL_FILE;
    source[0].byte_count = sizeof(audio_bytes);
    strcpy(source[0].media_type, "audio/wav");
    strcpy(source[0].reference, path);
    YVEX_TEST_ASSERT(yvex_content_part_seal(source, &err) == YVEX_OK,
                     "local audio identity seal");
    source[1].schema_version = YVEX_CONTENT_PART_SCHEMA_V1;
    source[1].kind = YVEX_CONTENT_TEXT;
    source[1].storage = YVEX_CONTENT_INLINE;
    source[1].bytes = transcript;
    source[1].byte_count = sizeof(transcript) - 1u;
    strcpy(source[1].media_type, "text/plain;charset=utf-8");
    strcpy(source[1].derived_from_content_identity,
           source[0].content_identity);
    YVEX_TEST_ASSERT(yvex_content_part_seal(source + 1u, &err) == YVEX_OK,
                     "derived transcript provenance seal");
    YVEX_TEST_ASSERT(
        yvex_content_parts_identity(source, 2u, identity, &err) == YVEX_OK,
        "ordered content identity");
    reversed[0] = source[1];
    reversed[1] = source[0];
    YVEX_TEST_ASSERT(
        yvex_content_parts_identity(reversed, 2u, reversed_identity, &err) ==
                YVEX_OK &&
            strcmp(identity, reversed_identity),
        "ordered content identity binds part order");
    rc = yvex_content_parts_wire_encode(
        source, 2u, wire, YVEX_CONTENT_WIRE_MAX_BYTES, &wire_bytes, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && wire_bytes < 4096u,
                     "local media remains a bounded reference on wire");
    rc = yvex_content_parts_wire_decode(wire, wire_bytes, &decoded,
                                        &decoded_count, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && decoded_count == 2u &&
            decoded[0].storage == YVEX_CONTENT_LOCAL_FILE &&
            decoded[0].bytes == NULL &&
            !strcmp(decoded[0].content_identity,
                    source[0].content_identity) &&
            !strcmp(decoded[1].derived_from_content_identity,
                    source[0].content_identity) &&
            decoded[1].byte_count == sizeof(transcript) - 1u &&
            !memcmp(decoded[1].bytes, transcript, sizeof(transcript) - 1u),
        "content wire preserves source identity and provenance");
    YVEX_TEST_ASSERT(yvex_content_part_local_verify(decoded, &err) == YVEX_OK,
                     "server-side local reference verification");
    capability.schema_version = YVEX_MODEL_CAPABILITY_SCHEMA_V1;
    capability.input_kinds = YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_TEXT) |
                             YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_AUDIO);
    capability.output_kinds = YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_TEXT);
    capability.execution_properties =
        YVEX_MODEL_CAPABILITY_ORDERED_INPUT_PARTS |
        YVEX_MODEL_CAPABILITY_DEMAND_ACTIVATION;
    capability.maximum_input_parts = 4u;
    YVEX_TEST_ASSERT(
        yvex_model_capability_admit(&capability, decoded, decoded_count,
                                    &err) == YVEX_OK,
        "directional capability admits exact ordered kinds");
    capability.input_kinds = YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_TEXT);
    YVEX_TEST_ASSERT(
        yvex_model_capability_admit(&capability, decoded, decoded_count,
                                    &err) == YVEX_ERR_UNSUPPORTED,
        "unsupported audio refuses without coercion");
    fd = open(path, O_WRONLY);
    YVEX_TEST_ASSERT(fd >= 0 && write(fd, "X", 1u) == 1 && close(fd) == 0,
                     "content mutation fixture");
    YVEX_TEST_ASSERT(
        yvex_content_part_local_verify(decoded, &err) == YVEX_ERR_STATE,
        "local content mutation refuses against sealed identity");
    yvex_content_parts_close(&decoded, decoded_count);
    free(wire);
    (void)unlink(path);
    return 0;
}

int yvex_test_provider(void)
{
    if (test_request_defaults() != 0) return 1;
    if (test_request_roundtrip() != 0) return 1;
    if (test_refusal() != 0) return 1;
    if (test_output_identity() != 0) return 1;
    if (test_ordered_content() != 0) return 1;
    return 0;
}
