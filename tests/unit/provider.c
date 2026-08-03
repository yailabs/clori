/*
 * Exercises bounded request identity, owned clone/wire roundtrip, JSON/tool refusal, and
 * authoritative output mutation detection without HTTP or model execution.
 */
#include <stdlib.h>
#include <string.h>

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
    tools[0].parameters_json.bytes = schema;
    tools[0].parameters_json.count = sizeof(schema) - 1u;
    stops[0].bytes = stop;
    stops[0].count = sizeof(stop) - 1u;
    request->schema_version = YVEX_PROVIDER_SCHEMA_V1;
    strcpy(request->model, "deepseek4-v4-flash-dspark");
    request->messages = messages;
    request->message_count = 2u;
    request->tools = tools;
    request->tool_count = 1u;
    request->stop_strings = stops;
    request->stop_count = 1u;
    request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_AUTO;
    request->response_format = YVEX_PROVIDER_RESPONSE_JSON_OBJECT;
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
    yvex_provider_span stops[1];
    static const unsigned char malformed[] = "{\"type\":}";
    yvex_error err;
    int rc = request_build(&request, messages, tools, stops, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "provider refusal fixture");
    tools[0].parameters_json.bytes = malformed;
    tools[0].parameters_json.count = sizeof(malformed) - 1u;
    rc = yvex_provider_request_seal(&request, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !request.sealed,
                     "malformed tool schema refuses");
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
    YVEX_TEST_ASSERT(request.schema_version == YVEX_PROVIDER_SCHEMA_V1,
                     "provider default schema");
    YVEX_TEST_ASSERT(request.response_format == YVEX_PROVIDER_RESPONSE_TEXT,
                     "provider default response format");
    YVEX_TEST_ASSERT(request.maximum_output_tokens == 128u,
                     "provider default output limit");
    YVEX_TEST_ASSERT(request.tool_choice.kind == YVEX_PROVIDER_TOOL_CHOICE_AUTO,
                     "provider default tool choice");
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

int yvex_test_provider(void)
{
    if (test_request_defaults() != 0) return 1;
    if (test_request_roundtrip() != 0) return 1;
    if (test_refusal() != 0) return 1;
    if (test_output_identity() != 0) return 1;
    return 0;
}
