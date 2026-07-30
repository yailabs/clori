/* Owner: OpenAI gateway focused tests.
 * Owns: bounded profile request, rendering, and HTTP admission checks.
 * Does not own: model execution, external SDK environments, or live runtime proof.
 * Invariants: profile syntax reaches provider facts and refusals publish no request.
 * Boundary: tests may inspect the gateway source-local interface without entering production objects.
 * Purpose: exercise the compatibility adapter independently of a model or daemon.
 * Inputs: deterministic JSON and socket-pair HTTP fixtures.
 * Effects: allocates only test-local request/result storage.
 * Failure: any semantic mismatch returns one focused test failure. */

#include "tests/test.h"

#include "src/gateway/openai/private.h"

#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* Purpose: admit one explicit JSON fixture through the selected profile endpoint. */
static int admit_fixture(const char *json, openai_endpoint endpoint,
                         openai_admitted_request *admitted, yvex_error *err)
{
    openai_http_request request = {0};

    request.body = (unsigned char *)json;
    request.body_count = strlen(json);
    return openai_json_admit(&request, endpoint, "deepseek-v4-flash",
                             admitted, err);
}

/* Purpose: prove Chat fields map to sealed provider-neutral facts. */
static int test_chat_admission(void)
{
    static const char basic[] =
        "{\"model\":\"deepseek-v4-flash\",\"messages\":[{"
        "\"role\":\"user\",\"content\":\"Hello\"}],\"temperature\":0}";
    static const char json[] =
        "{\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"system\",\"content\":\"Be exact.\"},"
        "{\"role\":\"user\",\"content\":\"Score?\"}],"
        "\"stream\":true,\"stream_options\":{\"include_usage\":true},"
        "\"temperature\":0,\"top_p\":1,\"seed\":42,\"max_tokens\":16,\"n\":1,"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"get_match_context\",\"description\":\"Read match context\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"match_id\":{\"type\":\"string\"}},\"required\":[\"match_id\"]},"
        "\"strict\":false}}],\"tool_choice\":\"auto\","
        "\"parallel_tool_calls\":false,"
        "\"response_format\":{\"type\":\"json_object\"}}";
    openai_admitted_request admitted = {0};
    yvex_error err;
    int rc = admit_fixture(basic, OPENAI_ENDPOINT_CHAT, &admitted, &err);

    if (rc != YVEX_OK)
        fprintf(stderr, "basic Chat admission: %s\n", yvex_error_message(&err));
    YVEX_TEST_ASSERT(rc == YVEX_OK, "basic Chat request must admit");
    openai_admitted_request_clear(&admitted);
    rc = admit_fixture(json, OPENAI_ENDPOINT_CHAT, &admitted, &err);

    YVEX_TEST_ASSERT(rc == YVEX_OK, "Chat request must admit");
    YVEX_TEST_ASSERT(admitted.provider && admitted.provider->sealed,
                     "Chat request must seal provider identity");
    YVEX_TEST_ASSERT(admitted.provider->message_count == 2u,
                     "Chat messages must preserve ordering");
    YVEX_TEST_ASSERT(admitted.provider->tool_count == 1u,
                     "function definition must remain typed");
    YVEX_TEST_ASSERT(admitted.provider->maximum_output_tokens == 16u,
                     "maximum output tokens must map exactly");
    YVEX_TEST_ASSERT(admitted.provider->stream && admitted.provider->include_usage,
                     "stream usage policy must map exactly");
    YVEX_TEST_ASSERT(admitted.provider->response_format ==
                         YVEX_PROVIDER_RESPONSE_JSON_OBJECT,
                     "JSON object policy must map exactly");
    YVEX_TEST_ASSERT(!admitted.provider->sampling.stochastic,
                     "temperature zero must select greedy generation");
    openai_admitted_request_clear(&admitted);
    return 0;
}

/* Purpose: prove unsupported, duplicate, and multimodal fields fail closed. */
static int test_request_refusals(void)
{
    static const char duplicate[] =
        "{\"model\":\"deepseek-v4-flash\",\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    static const char multiple[] =
        "{\"model\":\"deepseek-v4-flash\",\"n\":2,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    static const char multimodal[] =
        "{\"model\":\"deepseek-v4-flash\",\"messages\":[{"
        "\"role\":\"user\",\"content\":[{\"type\":\"input_image\","
        "\"image_url\":\"data:image/png;base64,AA==\"}]}]}";
    static const char ambiguous_maximum[] =
        "{\"model\":\"deepseek-v4-flash\",\"max_tokens\":4,"
        "\"max_completion_tokens\":4,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    static const char strict_tool[] =
        "{\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"f\",\"parameters\":{\"type\":\"object\"},"
        "\"strict\":true}}]}";
    openai_admitted_request admitted = {0};
    yvex_error err;

    YVEX_TEST_ASSERT(admit_fixture(duplicate, OPENAI_ENDPOINT_CHAT,
                                   &admitted, &err) != YVEX_OK,
                     "duplicate root fields must refuse");
    YVEX_TEST_ASSERT(!admitted.provider,
                     "duplicate refusal must publish no provider request");
    YVEX_TEST_ASSERT(admit_fixture(multiple, OPENAI_ENDPOINT_CHAT,
                                   &admitted, &err) == YVEX_ERR_UNSUPPORTED,
                     "n greater than one must refuse explicitly");
    YVEX_TEST_ASSERT(admit_fixture(multimodal, OPENAI_ENDPOINT_CHAT,
                                   &admitted, &err) != YVEX_OK,
                     "multimodal content must refuse");
    YVEX_TEST_ASSERT(admit_fixture(ambiguous_maximum, OPENAI_ENDPOINT_CHAT,
                                   &admitted, &err) != YVEX_OK,
                     "competing maximum-token fields must refuse");
    YVEX_TEST_ASSERT(admit_fixture(strict_tool, OPENAI_ENDPOINT_CHAT,
                                   &admitted, &err) == YVEX_ERR_UNSUPPORTED,
                     "strict tool schemas must refuse without constrained decoding");
    return 0;
}

/* Purpose: prove Responses instructions and continuity references remain typed. */
static int test_responses_admission(void)
{
    static const char json[] =
        "{\"model\":\"deepseek-v4-flash\",\"instructions\":\"Answer briefly.\","
        "\"input\":\"Hello\",\"max_output_tokens\":8,\"temperature\":0,"
        "\"previous_response_id\":\"resp_prior\",\"store\":false,"
        "\"background\":false,\"parallel_tool_calls\":false,"
        "\"tools\":[{\"type\":\"function\",\"name\":\"lookup\","
        "\"parameters\":{\"type\":\"object\"},\"strict\":false}],"
        "\"tool_choice\":{\"type\":\"function\",\"name\":\"lookup\"}}";
    openai_admitted_request admitted = {0};
    yvex_error err;

    YVEX_TEST_ASSERT(admit_fixture(json, OPENAI_ENDPOINT_RESPONSES,
                                   &admitted, &err) == YVEX_OK,
                     "Responses request must admit");
    YVEX_TEST_ASSERT(admitted.provider->message_count == 2u,
                     "instructions must prepend a system message");
    YVEX_TEST_ASSERT(admitted.provider->messages[0].role ==
                         YVEX_PROVIDER_ROLE_SYSTEM,
                     "instruction role must remain typed");
    YVEX_TEST_ASSERT_STREQ(admitted.provider->previous_response_id,
                           "resp_prior", "response reference must map");
    YVEX_TEST_ASSERT(admitted.provider->tool_choice.kind ==
                         YVEX_PROVIDER_TOOL_CHOICE_FUNCTION,
                     "Responses named function choice must remain typed");
    YVEX_TEST_ASSERT_STREQ(admitted.provider->tool_choice.function_name,
                           "lookup", "Responses flat function name must map");
    openai_admitted_request_clear(&admitted);
    return 0;
}

/* Purpose: prove both endpoint result and streaming documents are valid JSON. */
static int test_rendering(void)
{
    openai_generation_result result = {0};
    yvex_client_message fragment = {0};
    unsigned char *json = NULL;
    unsigned long long count = 0u;
    yvex_error err;

    result.text = (unsigned char *)"hello";
    result.text_count = 5u;
    result.prompt_tokens = 3u;
    result.completion_tokens = 1u;
    result.total_tokens = 4u;
    result.finish = YVEX_PROVIDER_FINISH_STOP;
    result.complete = 1;
    YVEX_TEST_ASSERT(openai_json_result(
        OPENAI_ENDPOINT_CHAT, "chatcmpl_test", "deepseek-v4-flash", 7u,
        &result, &json, &count, &err) == YVEX_OK,
        "Chat result must render");
    YVEX_TEST_ASSERT(yvex_provider_json_value_validate(json, count, 1,
                                                        &err) == YVEX_OK,
                     "Chat result must be valid JSON");
    free(json);
    json = NULL;
    YVEX_TEST_ASSERT(openai_json_result(
        OPENAI_ENDPOINT_RESPONSES, "resp_test", "deepseek-v4-flash", 7u,
        &result, &json, &count, &err) == YVEX_OK,
        "Responses result must render");
    YVEX_TEST_ASSERT(yvex_provider_json_value_validate(json, count, 1,
                                                        &err) == YVEX_OK,
                     "Responses result must be valid JSON");
    free(json);
    json = NULL;
    fragment.kind = YVEX_CLIENT_MESSAGE_FRAGMENT;
    memcpy(fragment.bytes, "hel", 3u);
    fragment.byte_count = 3u;
    fragment.provider_output_kind = YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT;
    YVEX_TEST_ASSERT(openai_json_stream_chunk(
        OPENAI_ENDPOINT_RESPONSES, "resp_test", "deepseek-v4-flash", 7u,
        &fragment, 0, &json, &count, &err) == YVEX_OK,
        "Responses delta must render");
    YVEX_TEST_ASSERT(yvex_provider_json_value_validate(json, count, 1,
                                                        &err) == YVEX_OK,
                     "Responses delta must be valid JSON");
    free(json);
    return 0;
}

/* Purpose: prove strict Content-Length admission and request-smuggling refusal. */
static int test_http_admission(void)
{
    static const char good[] =
        "POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Length: 2\r\n\r\n{}";
    static const char duplicate[] =
        "POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}";
    int pair[2];
    openai_http_request request = {0};
    yvex_error err;

    YVEX_TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0,
                     "HTTP socket pair must open");
    YVEX_TEST_ASSERT(write(pair[0], good, sizeof(good) - 1u) ==
                         (ssize_t)(sizeof(good) - 1u),
                     "complete HTTP fixture must write");
    YVEX_TEST_ASSERT(openai_http_read(pair[1], &request, &err) == YVEX_OK,
                     "bounded HTTP request must admit");
    YVEX_TEST_ASSERT_STREQ(request.method, "POST", "HTTP method must parse");
    YVEX_TEST_ASSERT_STREQ(request.path, "/v1/responses",
                           "HTTP path must parse");
    openai_http_request_clear(&request);
    close(pair[0]);
    close(pair[1]);

    YVEX_TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0,
                     "duplicate-header socket pair must open");
    YVEX_TEST_ASSERT(write(pair[0], duplicate, sizeof(duplicate) - 1u) ==
                         (ssize_t)(sizeof(duplicate) - 1u),
                     "duplicate-header fixture must write");
    YVEX_TEST_ASSERT(openai_http_read(pair[1], &request, &err) != YVEX_OK,
                     "duplicate Content-Length must refuse");
    close(pair[0]);
    close(pair[1]);
    return 0;
}

/* Purpose: prove bounded HTTP peer observation distinguishes an open socket from FIN. */
static int test_http_peer_liveness(void)
{
    int pair[2], closed = -1;
    yvex_error err;
    YVEX_TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0,
                     "HTTP peer-liveness socket pair must open");
    YVEX_TEST_ASSERT(openai_http_peer_wait(pair[1], 1u, &closed, &err) == YVEX_OK &&
                         !closed,
                     "an idle connected HTTP peer must remain live");
    close(pair[0]);
    YVEX_TEST_ASSERT(openai_http_peer_wait(pair[1], 100u, &closed, &err) == YVEX_OK &&
                         closed,
                     "HTTP peer FIN must be observed without a response write");
    close(pair[1]);
    return 0;
}

/* Purpose: run the bounded compatibility-adapter unit matrix. */
int yvex_test_openai(void)
{
    if (test_chat_admission() != 0) return 1;
    if (test_request_refusals() != 0) return 1;
    if (test_responses_admission() != 0) return 1;
    if (test_rendering() != 0) return 1;
    if (test_http_admission() != 0) return 1;
    return test_http_peer_liveness();
}
