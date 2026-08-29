/*
 * Verify mutable tokenizer request state remains isolated and transactional. Synthetic pieces
 * exercise only public contracts and never enter production objects. Sanitizer-visible invariant
 * proof; target-scale BPE parity belongs to the live lane.
 */
#include "src/tokenizer/private.h"
#include "tests/test.h"

#include <yvex/internal/core.h>
#include <yvex/internal/compiler.h>
#include <yvex/internal/family_catalog.h>
#include <yvex/internal/families/deepseek_v4.h>

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>

#define TEST_DSML "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c"

extern const yvex_family_descriptor yvex_graph_family_descriptor_minimax_h3;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int entered, release;
} decoder_test_gate;

typedef struct {
    yvex_tokenizer_decoder *decoder;
    unsigned long long attempts;
    int rc, closing_refusal;
} decoder_test_call;

typedef struct {
    yvex_tokenizer_decoder **decoder;
} decoder_test_close;

typedef struct {
    unsigned char reasoning[128], final[128];
    unsigned long long reasoning_count, final_count;
} reasoning_capture;

static int test_nfc_normalization(void)
{
    static const unsigned char decomposed_latin[] = "e\xcc\x80 a\xcc\x8a";
    static const unsigned char composed_latin[] = "\xc3\xa8 \xc3\xa5";
    static const unsigned char decomposed_hangul[] = "\xe1\x84\x80\xe1\x85\xa1";
    static const unsigned char composed_hangul[] = "\xea\xb0\x80";
    static const unsigned char multilingual[] = "\xe4\xb8\xad\xe6\x96\x87 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e";
    static const unsigned char malformed[] = {0xc3u, 0x28u};
    unsigned char *output = NULL;
    unsigned long long count = 0u;
    yvex_error err;

    YVEX_TEST_ASSERT(
        yvex_tokenizer_nfc_normalize(decomposed_latin, sizeof(decomposed_latin) - 1u,
                                     &output, &count, &err) == YVEX_OK &&
            count == sizeof(composed_latin) - 1u &&
            memcmp(output, composed_latin, (size_t)count) == 0,
        "NFC composes Latin canonical equivalents");
    free(output);
    output = NULL;
    YVEX_TEST_ASSERT(
        yvex_tokenizer_nfc_normalize(decomposed_hangul, sizeof(decomposed_hangul) - 1u,
                                     &output, &count, &err) == YVEX_OK &&
            count == sizeof(composed_hangul) - 1u &&
            memcmp(output, composed_hangul, (size_t)count) == 0,
        "NFC composes Hangul algorithmically");
    free(output);
    output = NULL;
    YVEX_TEST_ASSERT(
        yvex_tokenizer_nfc_normalize(multilingual, sizeof(multilingual) - 1u,
                                     &output, &count, &err) == YVEX_OK &&
            count == sizeof(multilingual) - 1u &&
            memcmp(output, multilingual, (size_t)count) == 0,
        "NFC preserves already-normalized multilingual input");
    free(output);
    output = (unsigned char *)1;
    count = 7u;
    YVEX_TEST_ASSERT(
        yvex_tokenizer_nfc_normalize(malformed, sizeof(malformed), &output, &count, &err) ==
                YVEX_ERR_FORMAT &&
            output == NULL && count == 0u,
        "NFC rejects malformed UTF-8 without publishing output");
    return 0;
}

static int reasoning_capture_sink(void *opaque,
                                  yvex_reasoning_segment segment,
                                  const unsigned char *bytes,
                                  unsigned long long byte_count,
                                  yvex_error *err)
{
    reasoning_capture *capture = (reasoning_capture *)opaque;
    unsigned char *target = segment == YVEX_REASONING_SEGMENT_EXPLICIT
                                ? capture->reasoning : capture->final;
    unsigned long long *count = segment == YVEX_REASONING_SEGMENT_EXPLICIT
                                    ? &capture->reasoning_count
                                    : &capture->final_count;
    if (*count > 128u - byte_count) return YVEX_ERR_BOUNDS;
    memcpy(target + *count, bytes, (size_t)byte_count);
    *count += byte_count;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int decoder_test_cancel(void *opaque)
{
    decoder_test_gate *gate = (decoder_test_gate *)opaque;
    (void)pthread_mutex_lock(&gate->mutex);
    gate->entered = 1;
    (void)pthread_cond_broadcast(&gate->condition);
    while (!gate->release)
        (void)pthread_cond_wait(&gate->condition, &gate->mutex);
    (void)pthread_mutex_unlock(&gate->mutex);
    return 1;
}

static void *decoder_test_push_main(void *opaque)
{
    decoder_test_call *call = (decoder_test_call *)opaque;
    yvex_tokenizer_fragment fragment;
    yvex_error err;
    call->rc = yvex_tokenizer_decoder_push(call->decoder, 3u, &fragment, &err);
    yvex_tokenizer_fragment_clear(&fragment);
    return NULL;
}

static void *decoder_test_contender_main(void *opaque)
{
    decoder_test_call *call = (decoder_test_call *)opaque;
    yvex_tokenizer_fragment fragment;
    yvex_error err;
    for (call->attempts = 1u; call->attempts <= 1000000u; ++call->attempts) {
        call->rc = yvex_tokenizer_decoder_push(call->decoder, 3u, &fragment, &err);
        if (call->rc == YVEX_ERR_STATE &&
            strcmp(yvex_error_message(&err), "decoder is closing") == 0) {
            call->closing_refusal = 1;
            break;
        }
        yvex_tokenizer_fragment_clear(&fragment);
        (void)sched_yield();
    }
    return NULL;
}

static void *decoder_test_close_main(void *opaque)
{
    decoder_test_close *close = (decoder_test_close *)opaque;
    yvex_tokenizer_decoder_close(close->decoder);
    return NULL;
}

static void fixture_open(yvex_tokenizer *tokenizer, yvex_token_info tokens[4])
{
    static const char identity[] =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    memset(tokenizer, 0, sizeof(*tokenizer));
    memset(tokens, 0, 4u * sizeof(*tokens));
    tokens[0] = (yvex_token_info){0u, "<eos>", 5u, 0.0f, YVEX_TOKEN_TYPE_CONTROL};
    tokens[1] = (yvex_token_info){1u, "\xf0\x9f", 2u, 0.0f, YVEX_TOKEN_TYPE_USER_DEFINED};
    tokens[2] = (yvex_token_info){2u, "\x98\x80", 2u, 0.0f, YVEX_TOKEN_TYPE_USER_DEFINED};
    tokens[3] = (yvex_token_info){3u, "ok", 2u, 0.0f, YVEX_TOKEN_TYPE_USER_DEFINED};
    tokenizer->tokens = tokens;
    tokenizer->vocab_size = 4u;
    tokenizer->eos.present = 1;
    tokenizer->eos.id = 0u;
    tokenizer->plan.sealed = 1;
    tokenizer->plan.vocabulary_size = 4u;
    tokenizer->plan.prompt_policy = YVEX_TOKENIZER_PROMPT_CONVERSATION;
    tokenizer->plan.schema_version = YVEX_TOKENIZER_PLAN_SCHEMA_CURRENT;
    tokenizer->plan.explicit_reasoning_supported = 1;
    tokenizer->plan.maximum_reasoning_supported = 1;
    tokenizer->plan.reasoning_start_token_id = 10u;
    tokenizer->plan.reasoning_end_token_id = 11u;
    (void)yvex_compiler_family_deepseek_v4()->tokenizer_policy(
        &tokenizer->compiled_policy, NULL);
    (void)yvex_tokenizer_family_policy_conversation(
        &tokenizer->compiled_policy, &tokenizer->conversation_view);
    tokenizer->conversation = &tokenizer->conversation_view;
    memcpy(tokenizer->plan.tokenizer_plan_identity, identity, sizeof(identity));
}

static yvex_provider_span text_span(const char *text)
{
    yvex_provider_span span = {
        .bytes = (const unsigned char *)text,
        .count = text ? (unsigned long long)strlen(text) : 0u};
    return span;
}

static int test_compiled_family_policy(void)
{
    const yvex_family_compiler_adapter *compiler = yvex_compiler_family_deepseek_v4();
    const yvex_family_descriptor *minimax_descriptor =
        &yvex_graph_family_descriptor_minimax_h3;
    const yvex_family_source_adapter *minimax =
        minimax_descriptor && minimax_descriptor->source
            ? minimax_descriptor->source() : NULL;
    yvex_tokenizer_family_policy first, decoded, changed, direct, direct_decoded;
    yvex_conversation_protocol view;
    yvex_core_bytes encoded = {0}, repeated = {0}, direct_encoded = {0};
    yvex_error err;
    unsigned int reasoning_offset;
    encoded.maximum = repeated.maximum = direct_encoded.maximum = 16384u;
    encoded.initial_capacity = repeated.initial_capacity = direct_encoded.initial_capacity = 4096u;
    YVEX_TEST_ASSERT(
        compiler && compiler->tokenizer_policy &&
            compiler->tokenizer_policy(&first, &err) &&
            yvex_tokenizer_family_policy_validate(&first, &err) == YVEX_OK &&
            yvex_tokenizer_family_policy_encode(&first, &encoded, &err) == YVEX_OK &&
            yvex_tokenizer_family_policy_encode(&first, &repeated, &err) == YVEX_OK &&
            encoded.count == repeated.count &&
            memcmp(encoded.data, repeated.data, encoded.count) == 0,
        "family tokenizer policy compiles deterministically");
    YVEX_TEST_ASSERT(
        yvex_tokenizer_family_policy_decode(
            &decoded, encoded.data, encoded.count, &err) == YVEX_OK &&
            strcmp(decoded.policy_identity, first.policy_identity) == 0 &&
            decoded.vocabulary_size == 129280u &&
            decoded.prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION &&
            yvex_tokenizer_family_policy_conversation(&decoded, &view) &&
            strcmp(view.thinking_start, "<think>") == 0 &&
            strcmp(view.thinking_end, "</think>") == 0,
        "compiled tokenizer policy roundtrips without family lookup");
    changed = decoded;
    reasoning_offset = changed.text_offsets[YVEX_TOKENIZER_POLICY_THINKING_END];
    changed.text[reasoning_offset] ^= 1;
    YVEX_TEST_ASSERT(
        yvex_tokenizer_family_policy_validate(&changed, &err) == YVEX_ERR_FORMAT,
        "compiled tokenizer policy rejects source-grammar drift");
    YVEX_TEST_ASSERT(
        yvex_core_bytes_append(&encoded, "x", 1u) &&
            yvex_tokenizer_family_policy_decode(
                &changed, encoded.data, encoded.count, &err) == YVEX_ERR_FORMAT,
        "compiled tokenizer policy rejects trailing encoding bytes");
    YVEX_TEST_ASSERT(
        minimax && minimax->tokenizer_policy &&
            minimax->tokenizer_policy(&direct, &err) &&
            direct.prompt_policy == YVEX_TOKENIZER_PROMPT_VERBATIM &&
            strcmp(direct.direct_prompt_name, "verbatim-no-special-v1") == 0 &&
            yvex_tokenizer_family_policy_encode(
                &direct, &direct_encoded, &err) == YVEX_OK &&
            yvex_tokenizer_family_policy_decode(
                &direct_decoded, direct_encoded.data, direct_encoded.count, &err) == YVEX_OK &&
            strcmp(direct_decoded.policy_identity, direct.policy_identity) == 0 &&
            !yvex_tokenizer_family_policy_conversation(&direct_decoded, &view),
        "direct family tokenizer policy roundtrips without fake conversation facts");
    changed = direct_decoded;
    changed.direct_prompt_name[0] ^= 1;
    YVEX_TEST_ASSERT(
        yvex_tokenizer_family_policy_validate(&changed, &err) == YVEX_ERR_FORMAT,
        "direct family tokenizer policy rejects prompt-contract drift");
    free(encoded.data);
    free(repeated.data);
    free(direct_encoded.data);
    return 0;
}

static int prompt_digest(const yvex_rendered_prompt *prompt,
                         char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!prompt || !prompt->text ||
        !yvex_sha256_update(&hash, prompt->text, (size_t)prompt->len) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int request_reseal(yvex_provider_request *request, yvex_error *err)
{
    request->sealed = 0;
    request->request_identity[0] = '\0';
    return yvex_provider_request_seal(request, err);
}

static int test_reasoning_channel(void)
{
    static const unsigned char output[] = "analysis</think>answer";
    yvex_tokenizer tokenizer;
    yvex_token_info tokens[4];
    yvex_tokenizer_reasoning_stream *stream = NULL;
    reasoning_capture capture = {0};
    yvex_error err;
    size_t index;
    fixture_open(&tokenizer, tokens);
    YVEX_TEST_ASSERT(
        yvex_tokenizer_reasoning_stream_open(
            &stream, &tokenizer, YVEX_REASONING_ENABLED,
            reasoning_capture_sink, &capture, &err) == YVEX_OK,
        "open source-authored reasoning classifier");
    for (index = 0u; index < sizeof(output) - 1u; ++index)
        YVEX_TEST_ASSERT(yvex_tokenizer_reasoning_stream_push(
                             stream, output + index, 1u, &err) == YVEX_OK,
                         "classify delimiter split at every byte");
    YVEX_TEST_ASSERT(
        yvex_tokenizer_reasoning_stream_finish(stream, &err) == YVEX_OK &&
            capture.reasoning_count == 8u && capture.final_count == 6u &&
            memcmp(capture.reasoning, "analysis", 8u) == 0 &&
            memcmp(capture.final, "answer", 6u) == 0,
        "delimiter is consumed and channels remain byte-exact");
    yvex_tokenizer_reasoning_stream_close(&stream);
    memset(&capture, 0, sizeof(capture));
    YVEX_TEST_ASSERT(
        yvex_tokenizer_reasoning_stream_open(
            &stream, &tokenizer, YVEX_REASONING_ENABLED,
            reasoning_capture_sink, &capture, &err) == YVEX_OK &&
            yvex_tokenizer_reasoning_stream_push(
                stream, (const unsigned char *)"reason", 6u, &err) ==
                YVEX_OK &&
            yvex_tokenizer_reasoning_stream_push(
                stream,
                (const unsigned char *)tokenizer.conversation->thinking_end,
                strlen(tokenizer.conversation->thinking_end), &err) == YVEX_OK &&
            yvex_tokenizer_reasoning_stream_push(
                stream, (const unsigned char *)"answer", 6u, &err) ==
                YVEX_OK &&
            yvex_tokenizer_reasoning_stream_finish(stream, &err) == YVEX_OK &&
            capture.reasoning_count == 6u && capture.final_count == 6u &&
            memcmp(capture.reasoning, "reason", 6u) == 0 &&
            memcmp(capture.final, "answer", 6u) == 0,
        "source delimiter changes the typed stream channel");
    yvex_tokenizer_reasoning_stream_close(&stream);
    memset(&capture, 0, sizeof(capture));
    YVEX_TEST_ASSERT(
        yvex_tokenizer_reasoning_stream_open(
            &stream, &tokenizer, YVEX_REASONING_DISABLED,
            reasoning_capture_sink, &capture, &err) == YVEX_OK &&
            yvex_tokenizer_reasoning_stream_push(
                stream, output, sizeof(output) - 1u, &err) == YVEX_OK &&
            yvex_tokenizer_reasoning_stream_finish(stream, &err) == YVEX_OK &&
            capture.reasoning_count == 0u &&
            capture.final_count == sizeof(output) - 1u &&
            memcmp(capture.final, output, sizeof(output) - 1u) == 0,
        "disabled policy preserves canonical final bytes without inference");
    yvex_tokenizer_reasoning_stream_close(&stream);
    memset(&capture, 0, sizeof(capture));
    YVEX_TEST_ASSERT(
        yvex_tokenizer_reasoning_stream_open(
            &stream, &tokenizer, YVEX_REASONING_MAXIMUM,
            reasoning_capture_sink, &capture, &err) == YVEX_OK &&
            yvex_tokenizer_reasoning_stream_push(
                stream, (const unsigned char *)"unfinished", 10u, &err) ==
                YVEX_OK &&
            yvex_tokenizer_reasoning_stream_finish(stream, &err) ==
                YVEX_ERR_FORMAT &&
            capture.reasoning_count == 10u && capture.final_count == 0u,
        "incomplete source reasoning is retained but fails closed");
    yvex_tokenizer_reasoning_stream_close(&stream);
    return 0;
}

static int test_source_prompt_modes(void)
{
    static const char *const expected[] = {
        "5ec6ff56a432b916c342465ba509744d13252c6c92ff10b28157b8213b4d4699",
        "35670d0fcfe106b6b61e14f28fa5b7a787bf2d76ebf0c2664daa244fb4254a4b",
        "f1d70259df3c292ccaeef08217cda85e4db5356393669056c44bfe8ef23e0ab5"};
    const yvex_reasoning_policy policies[] = {
        YVEX_REASONING_DISABLED, YVEX_REASONING_ENABLED,
        YVEX_REASONING_MAXIMUM};
    yvex_tokenizer tokenizer;
    yvex_token_info tokens[4];
    yvex_provider_message message = {0};
    yvex_provider_request request;
    yvex_rendered_prompt rendered = {0};
    yvex_prompt_message native_messages[2] = {0};
    yvex_prompt_options native_options = {
        .add_bos = 1,
        .drop_thinking = 1,
        .mode = YVEX_PROMPT_MODE_THINKING,
        .reasoning_policy = YVEX_REASONING_ENABLED};
    yvex_error err;
    unsigned int index;

    fixture_open(&tokenizer, tokens);
    YVEX_TEST_ASSERT(
        strcmp(tokenizer.conversation->source_encoding_path,
               "encoding/encoding_dsv4.py") == 0 &&
            strcmp(tokenizer.conversation->source_encoding_identity,
                   "bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0") == 0,
        "conversation grammar is bound to the pinned source encoder");
    message.role = YVEX_PROVIDER_ROLE_USER;
    message.content = text_span("hello");
    yvex_provider_request_default(&request);
    strcpy(request.model, "deepseek4-v4-flash-dspark");
    request.messages = &message;
    request.message_count = 1u;
    request.maximum_output_tokens = 16u;
    for (index = 0u; index < 3u; ++index) {
        char digest[YVEX_SHA256_HEX_CAP];
        request.reasoning_policy = policies[index];
        YVEX_TEST_ASSERT(
            request_reseal(&request, &err) == YVEX_OK &&
                yvex_tokenizer_provider_prompt(
                    &tokenizer, &request, &rendered, &err) == YVEX_OK &&
                prompt_digest(&rendered, digest) &&
                strcmp(digest, expected[index]) == 0,
            "chat, think-high, and think-max match pinned source bytes");
        yvex_rendered_prompt_free(&rendered);
    }
    native_messages[0].role = YVEX_PROMPT_ROLE_USER;
    native_messages[0].content = "quote the delimiter";
    native_messages[1].role = YVEX_PROMPT_ROLE_ASSISTANT;
    native_messages[1].content = "literal </think> remains final text";
    YVEX_TEST_ASSERT(
        yvex_prompt_render(&rendered, &tokenizer, native_messages, 2u,
                           &native_options, &err) == YVEX_OK &&
            strstr(rendered.text, "literal </think> remains final text") != NULL,
        "ordinary assistant text is never inferred to be reasoning history");
    yvex_rendered_prompt_free(&rendered);
    return 0;
}

static int test_source_multiturn_and_tools(void)
{
    static const char expected_drop[] =
        "da3f00952c8d0525107de63e954c5998c78159f60f57e6f924787242328ab87a";
    static const char expected_keep[] =
        "ce0042976a04b0a8b8a0c7f45014c4e65eb5d001e12c8f9258edc2b14bd23201";
    static const char expected_tools[] =
        "28fe1c75aa52d7e27eb7eed107131c508e668abda61bce93d9a33e5b225ce5e5";
    static const unsigned char schema[] =
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},"
        "\"threshold\":{\"type\":\"number\",\"default\":1e20},"
        "\"zero\":{\"type\":\"number\",\"default\":-0.0},"
        "\"small\":{\"type\":\"number\",\"default\":1e-7}},"
        "\"required\":[\"city\"]}";
    yvex_tokenizer tokenizer;
    yvex_token_info tokens[4];
    yvex_provider_request request;
    yvex_provider_message messages[6] = {0};
    yvex_provider_function_tool tool = {0};
    yvex_provider_tool_call calls[2] = {0};
    yvex_rendered_prompt rendered = {0};
    yvex_error err;
    char digest[YVEX_SHA256_HEX_CAP];

    fixture_open(&tokenizer, tokens);
    messages[0].role = YVEX_PROVIDER_ROLE_USER;
    messages[0].content = text_span("q1");
    messages[1].role = YVEX_PROVIDER_ROLE_ASSISTANT;
    messages[1].reasoning_content = text_span("r1");
    messages[1].content = text_span("a1");
    messages[2].role = YVEX_PROVIDER_ROLE_USER;
    messages[2].content = text_span("q2");
    yvex_provider_request_default(&request);
    strcpy(request.model, "deepseek4-v4-flash-dspark");
    request.messages = messages;
    request.message_count = 3u;
    request.maximum_output_tokens = 16u;
    request.reasoning_policy = YVEX_REASONING_ENABLED;
    YVEX_TEST_ASSERT(
        request_reseal(&request, &err) == YVEX_OK &&
            yvex_tokenizer_provider_prompt(
                &tokenizer, &request, &rendered, &err) == YVEX_OK &&
            prompt_digest(&rendered, digest) &&
            strcmp(digest, expected_drop) == 0,
        "ordinary multi-turn drops prior reasoning exactly like source");
    yvex_rendered_prompt_free(&rendered);
    request.drop_thinking = 0;
    YVEX_TEST_ASSERT(
        request_reseal(&request, &err) == YVEX_OK &&
            yvex_tokenizer_provider_prompt(
                &tokenizer, &request, &rendered, &err) == YVEX_OK &&
            prompt_digest(&rendered, digest) &&
            strcmp(digest, expected_keep) == 0,
        "explicit retained-reasoning history matches source bytes");
    yvex_rendered_prompt_free(&rendered);

    memset(messages, 0, sizeof(messages));
    messages[0].role = YVEX_PROVIDER_ROLE_SYSTEM;
    messages[0].content = text_span("You are helpful.");
    messages[1].role = YVEX_PROVIDER_ROLE_USER;
    messages[1].content = text_span("q1");
    messages[2].role = YVEX_PROVIDER_ROLE_ASSISTANT;
    messages[2].reasoning_content = text_span("r1");
    strcpy(calls[0].call_id, "call_b");
    strcpy(calls[0].name, "weather");
    calls[0].arguments_json = text_span(
        "{\"city\":\"Rome\",\"threshold\":1e20,\"zero\":-0,"
        "\"small\":1e-7,\"unit\":1.0}");
    strcpy(calls[1].call_id, "call_a");
    strcpy(calls[1].name, "weather");
    calls[1].arguments_json = text_span("{\"city\":\"Milan\"}");
    messages[2].tool_calls = calls;
    messages[2].tool_call_count = 2u;
    messages[3].role = YVEX_PROVIDER_ROLE_TOOL;
    strcpy(messages[3].tool_call_id, "call_a");
    messages[3].content = text_span("rain");
    messages[4].role = YVEX_PROVIDER_ROLE_TOOL;
    strcpy(messages[4].tool_call_id, "call_b");
    messages[4].content = text_span("sunny");
    messages[5].role = YVEX_PROVIDER_ROLE_USER;
    messages[5].content = text_span("now");
    strcpy(tool.name, "weather");
    tool.description = text_span("Weather");
    tool.description_present = 1;
    tool.parameters_json = (yvex_provider_span){schema, sizeof(schema) - 1u};
    request.messages = messages;
    request.message_count = 6u;
    request.tools = &tool;
    request.tool_count = 1u;
    request.drop_thinking = 1;
    YVEX_TEST_ASSERT(
        request_reseal(&request, &err) == YVEX_OK &&
            yvex_tokenizer_provider_prompt(
                &tokenizer, &request, &rendered, &err) == YVEX_OK &&
            prompt_digest(&rendered, digest) &&
            strcmp(digest, expected_tools) == 0,
        "tool history, JSON numbers, and result order match source bytes");
    yvex_rendered_prompt_free(&rendered);
    return 0;
}

static int test_source_developer_boundary(void)
{
    static const char expected[] =
        "2a30daa2e35c287d9956f1c1df4a2099e422d0afcb312cdcdf3ce6aaa057f0cb";
    yvex_tokenizer tokenizer;
    yvex_token_info tokens[4];
    yvex_provider_message messages[2] = {0};
    yvex_provider_request request;
    yvex_rendered_prompt rendered = {0};
    yvex_error err;
    char digest[YVEX_SHA256_HEX_CAP];

    fixture_open(&tokenizer, tokens);
    messages[0].role = YVEX_PROVIDER_ROLE_DEVELOPER;
    messages[0].content = text_span("rules");
    messages[1].role = YVEX_PROVIDER_ROLE_USER;
    messages[1].content = text_span("hello");
    yvex_provider_request_default(&request);
    strcpy(request.model, "deepseek4-v4-flash-dspark");
    request.messages = messages;
    request.message_count = 2u;
    request.maximum_output_tokens = 16u;
    YVEX_TEST_ASSERT(
        request_reseal(&request, &err) == YVEX_OK &&
            yvex_tokenizer_provider_prompt(
                &tokenizer, &request, &rendered, &err) == YVEX_OK &&
            prompt_digest(&rendered, digest) && strcmp(digest, expected) == 0,
        "developer and user boundaries match the pinned source encoder");
    yvex_rendered_prompt_free(&rendered);
    return 0;
}

static int provider_fixture(yvex_provider_request *request,
                            yvex_provider_message messages[1],
                            yvex_provider_function_tool tools[1],
                            yvex_error *err)
{
    static const unsigned char user[] = "Get match m1.";
    static const unsigned char description[] = "Load match context.";
    static const unsigned char schema[] =
        "{\"type\":\"object\",\"properties\":{\"match_id\":{"
        "\"type\":\"string\"}},\"required\":[\"match_id\"]}";
    memset(request, 0, sizeof(*request));
    memset(messages, 0, sizeof(*messages));
    memset(tools, 0, sizeof(*tools));
    messages[0].role = YVEX_PROVIDER_ROLE_USER;
    messages[0].content = (yvex_provider_span){user, sizeof(user) - 1u};
    strcpy(tools[0].name, "get_match_context");
    tools[0].description = (yvex_provider_span){description,
                                                sizeof(description) - 1u};
    tools[0].description_present = 1;
    tools[0].parameters_json = (yvex_provider_span){schema,
                                                    sizeof(schema) - 1u};
    request->schema_version = YVEX_PROVIDER_SCHEMA_V2;
    strcpy(request->model, "deepseek4-v4-flash-dspark");
    request->messages = messages;
    request->message_count = 1u;
    request->tools = tools;
    request->tool_count = 1u;
    request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_AUTO;
    request->sampling.temperature = 0.0;
    request->sampling.top_p = 1.0;
    request->sampling.typical_p = 1.0;
    request->maximum_output_tokens = 16u;
    request->reasoning_policy = YVEX_REASONING_DISABLED;
    request->drop_thinking = 1;
    return yvex_provider_request_seal(request, err);
}

static int test_provider_projection(void)
{
    static const unsigned char completion[] =
        "\n\n<" TEST_DSML "tool_calls>\n"
        "<" TEST_DSML "invoke name=\"get_match_context\">\n"
        "<" TEST_DSML "parameter name=\"match_id\" string=\"true\">m1"
        "</" TEST_DSML "parameter>\n"
        "</" TEST_DSML "invoke>\n"
        "</" TEST_DSML "tool_calls>";
    static const unsigned char prose[] = "Please call get_match_context with m1.";
    yvex_tokenizer tokenizer;
    yvex_token_info tokens[4];
    yvex_provider_request request;
    yvex_provider_message messages[1];
    yvex_provider_function_tool tools[1];
    yvex_rendered_prompt rendered = {0};
    yvex_tokenizer_provider_result result = {0};
    yvex_error err;

    fixture_open(&tokenizer, tokens);
    YVEX_TEST_ASSERT(provider_fixture(&request, messages, tools, &err) ==
                         YVEX_OK,
                     "provider fixture seals");
    YVEX_TEST_ASSERT(yvex_tokenizer_provider_prompt(
                         &tokenizer, &request, &rendered, &err) == YVEX_OK,
                     "provider prompt renders through tokenizer policy");
    YVEX_TEST_ASSERT(strstr(rendered.text, "Available Tool Schemas") != NULL &&
                         strstr(rendered.text, "get_match_context") != NULL,
                     "rendered prompt contains exact tool policy and schema");
    yvex_rendered_prompt_free(&rendered);
    YVEX_TEST_ASSERT(yvex_tokenizer_parse_provider_completion(
                         &tokenizer, &request, completion,
                         sizeof(completion) - 1u, &result, &err) == YVEX_OK &&
                         result.schema_version ==
                             YVEX_TOKENIZER_PROVIDER_RESULT_SCHEMA_V2 &&
                         result.kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL,
                     "exact DSML completion becomes one typed function call");
    YVEX_TEST_ASSERT(result.tool_call_count == 1u,
                     "one exact tool call is parsed");
    YVEX_TEST_ASSERT_STREQ(result.tool_calls[0].name, "get_match_context",
                           "tool name remains exact");
    YVEX_TEST_ASSERT(result.tool_calls[0].call_id[0] &&
                         yvex_provider_json_value_validate(
                             result.tool_calls[0].arguments_json.bytes,
                             result.tool_calls[0].arguments_json.count, 1,
                             &err) == YVEX_OK,
                     "tool arguments and call identity are valid");
    yvex_tokenizer_provider_result_clear(&result);
    YVEX_TEST_ASSERT(yvex_tokenizer_parse_provider_completion(
                         &tokenizer, &request, prose, sizeof(prose) - 1u,
                         &result, &err) == YVEX_OK &&
                         result.kind == YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
                     "ordinary prose never becomes a tool call");
    yvex_tokenizer_provider_result_clear(&result);
    request.reasoning_policy = YVEX_REASONING_ENABLED;
    YVEX_TEST_ASSERT(request_reseal(&request, &err) == YVEX_OK,
                     "thinking provider fixture reseals");
    {
        static const unsigned char thinking_completion[] =
            "reason</think>answer";
        static const unsigned char missing_delimiter[] = "reason only";
        YVEX_TEST_ASSERT(
            yvex_tokenizer_parse_provider_completion(
                &tokenizer, &request, thinking_completion,
                sizeof(thinking_completion) - 1u, &result, &err) == YVEX_OK &&
                result.reasoning_content_count == 6u &&
                memcmp(result.reasoning_content, "reason", 6u) == 0 &&
                result.content_count == 6u &&
                memcmp(result.content, "answer", 6u) == 0,
            "source delimiter separates explicit reasoning and final content");
        yvex_tokenizer_provider_result_clear(&result);
        YVEX_TEST_ASSERT(
            yvex_tokenizer_parse_provider_completion(
                &tokenizer, &request, missing_delimiter,
                sizeof(missing_delimiter) - 1u, &result, &err) ==
                YVEX_ERR_FORMAT,
            "thinking completion without source delimiter refuses");
    }
    {
        static const unsigned char multiple_calls[] =
            "r</think>\n\n<" TEST_DSML "tool_calls>\n"
            "<" TEST_DSML "invoke name=\"get_match_context\">\n"
            "<" TEST_DSML "parameter name=\"match_id\" string=\"true\">m1"
            "</" TEST_DSML "parameter>\n"
            "</" TEST_DSML "invoke>\n"
            "<" TEST_DSML "invoke name=\"get_match_context\">\n"
            "<" TEST_DSML "parameter name=\"match_id\" string=\"true\">m2"
            "</" TEST_DSML "parameter>\n"
            "</" TEST_DSML "invoke>\n"
            "</" TEST_DSML "tool_calls>";
        YVEX_TEST_ASSERT(
            yvex_tokenizer_parse_provider_completion(
                &tokenizer, &request, multiple_calls,
                sizeof(multiple_calls) - 1u, &result, &err) == YVEX_OK &&
                result.kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL &&
                result.reasoning_content_count == 1u &&
                result.tool_call_count == 2u &&
                strcmp(result.tool_calls[0].name, "get_match_context") == 0 &&
                strcmp(result.tool_calls[1].name, "get_match_context") == 0 &&
                strcmp(result.tool_calls[0].call_id,
                       result.tool_calls[1].call_id) != 0,
            "source grammar parses multiple grounded tool calls distinctly");
        yvex_tokenizer_provider_result_clear(&result);
    }
    {
        static const unsigned char joined_calls[] =
            "r</think>\n\n<" TEST_DSML "tool_calls>\n"
            "<" TEST_DSML "invoke name=\"get_match_context\">\n"
            "<" TEST_DSML "parameter name=\"match_id\" string=\"true\">m1"
            "</" TEST_DSML "parameter>\n"
            "</" TEST_DSML "invoke>"
            "<" TEST_DSML "invoke name=\"get_match_context\">\n"
            "<" TEST_DSML "parameter name=\"match_id\" string=\"true\">m2"
            "</" TEST_DSML "parameter>\n"
            "</" TEST_DSML "invoke>\n"
            "</" TEST_DSML "tool_calls>";
        YVEX_TEST_ASSERT(
            yvex_tokenizer_parse_provider_completion(
                &tokenizer, &request, joined_calls,
                sizeof(joined_calls) - 1u, &result, &err) == YVEX_ERR_FORMAT,
            "source grammar requires a newline between tool invocations");
    }
    {
        char many_parameters[16384];
        size_t offset;
        unsigned int index;
        int written;

        written = snprintf(
            many_parameters, sizeof(many_parameters),
            "r</think>\n\n<" TEST_DSML "tool_calls>\n"
            "<" TEST_DSML "invoke name=\"get_match_context\">\n");
        YVEX_TEST_ASSERT(written > 0, "many-parameter fixture prefix");
        offset = (size_t)written;
        for (index = 0u; index < 40u; ++index) {
            written = snprintf(
                many_parameters + offset, sizeof(many_parameters) - offset,
                "<" TEST_DSML "parameter name=\"p%02u\" string=\"false\">%u"
                "</" TEST_DSML "parameter>\n",
                index, index);
            YVEX_TEST_ASSERT(written > 0 &&
                                 (size_t)written < sizeof(many_parameters) - offset,
                             "many-parameter fixture body");
            offset += (size_t)written;
        }
        written = snprintf(
            many_parameters + offset, sizeof(many_parameters) - offset,
            "</" TEST_DSML "invoke>\n</" TEST_DSML "tool_calls>");
        YVEX_TEST_ASSERT(written > 0 &&
                             (size_t)written < sizeof(many_parameters) - offset,
                         "many-parameter fixture suffix");
        offset += (size_t)written;
        YVEX_TEST_ASSERT(
            yvex_tokenizer_parse_provider_completion(
                &tokenizer, &request, (const unsigned char *)many_parameters,
                offset, &result, &err) == YVEX_OK &&
                result.tool_call_count == 1u &&
                result.tool_calls[0].arguments_json.count > 32u &&
                strstr((const char *)result.tool_calls[0].arguments_json.bytes,
                       "\"p39\": 39") != NULL,
            "source parameters are bounded by bytes rather than tool count");
        yvex_tokenizer_provider_result_clear(&result);
    }
    return 0;
}

static int test_incremental_decoder(void)
{
    yvex_tokenizer tokenizer;
    yvex_token_info tokens[4];
    yvex_tokenizer_decoder *decoder = NULL;
    yvex_tokenizer_fragment fragment;
    yvex_tokenizer_decode_result batch;
    yvex_tokenizer_decode_options options = {
        .skip_special_tokens = 1, .require_complete_utf8 = 1};
    yvex_error err;
    unsigned int ids[] = {1u, 2u, 3u};
    fixture_open(&tokenizer, tokens);
    memset(&fragment, 0, sizeof(fragment));
    memset(&batch, 0, sizeof(batch));
    YVEX_TEST_ASSERT(yvex_tokenizer_decoder_open(&decoder, &tokenizer, &options, &err) == YVEX_OK,
                     "open incremental decoder");
    YVEX_TEST_ASSERT(yvex_tokenizer_decoder_push(decoder, 1u, &fragment, &err) == YVEX_OK &&
                         fragment.byte_count == 0u && fragment.pending_byte_count == 2u,
                     "retain incomplete UTF-8 prefix");
    yvex_tokenizer_fragment_clear(&fragment);
    YVEX_TEST_ASSERT(yvex_tokenizer_decoder_push(decoder, 99u, &fragment, &err) == YVEX_ERR_BOUNDS &&
                         !fragment.completed,
                     "invalid token publishes no fragment");
    YVEX_TEST_ASSERT(yvex_tokenizer_decoder_push(decoder, 2u, &fragment, &err) == YVEX_OK &&
                         fragment.byte_count == 4u && fragment.pending_byte_count == 0u &&
                         memcmp(fragment.bytes, "\xf0\x9f\x98\x80", 4u) == 0,
                     "complete split UTF-8 atomically");
    yvex_tokenizer_fragment_clear(&fragment);
    YVEX_TEST_ASSERT(yvex_tokenizer_decoder_push(decoder, 3u, &fragment, &err) == YVEX_OK &&
                         fragment.byte_count == 2u && memcmp(fragment.bytes, "ok", 2u) == 0,
                     "decoder remains reusable after refusal");
    yvex_tokenizer_fragment_clear(&fragment);
    YVEX_TEST_ASSERT(yvex_tokenizer_decoder_finish(decoder, &fragment, &err) == YVEX_OK &&
                         fragment.pending_byte_count == 0u,
                     "finish complete decoder");
    yvex_tokenizer_fragment_clear(&fragment);
    YVEX_TEST_ASSERT(yvex_tokenizer_decoder_reset(decoder, &err) == YVEX_OK &&
                         yvex_tokenizer_decoder_push(decoder, 3u, &fragment, &err) == YVEX_OK &&
                         fragment.processed_token_count == 1u && fragment.byte_count == 2u,
                     "decoder reset starts one later turn without reallocating");
    yvex_tokenizer_fragment_clear(&fragment);
    YVEX_TEST_ASSERT(yvex_tokenizer_decode(&tokenizer, ids, 3u, &options, &batch, &err) == YVEX_OK &&
                         batch.byte_count == 6u &&
                         memcmp(batch.bytes, "\xf0\x9f\x98\x80ok", 6u) == 0,
                     "batch decode matches incremental bytes");
    yvex_tokenizer_decode_result_clear(&batch);
    yvex_tokenizer_decoder_close(&decoder);
    yvex_tokenizer_decoder_close(&decoder);
    YVEX_TEST_ASSERT(!decoder, "decoder close is idempotent");
    return 0;
}

static int test_incremental_close_drain(void)
{
    yvex_tokenizer tokenizer;
    yvex_token_info tokens[4];
    decoder_test_gate gate;
    decoder_test_call active, contender;
    yvex_tokenizer_decoder *decoder = NULL, *close_owner = NULL;
    decoder_test_close close;
    yvex_tokenizer_decode_options options = {
        .skip_special_tokens = 1, .require_complete_utf8 = 1,
        .cancelled = decoder_test_cancel, .cancel_context = &gate};
    pthread_t active_id, contender_id, close_id;
    yvex_error err;
    fixture_open(&tokenizer, tokens);
    memset(&gate, 0, sizeof(gate));
    YVEX_TEST_ASSERT(pthread_mutex_init(&gate.mutex, NULL) == 0 &&
                         pthread_cond_init(&gate.condition, NULL) == 0,
                     "decoder lifecycle test synchronization initializes");
    YVEX_TEST_ASSERT(yvex_tokenizer_decoder_open(&decoder, &tokenizer, &options, &err) == YVEX_OK,
                     "open cancellable decoder");
    close_owner = decoder;
    active = (decoder_test_call){.decoder = decoder, .rc = YVEX_ERR_STATE};
    contender = active;
    close = (decoder_test_close){.decoder = &close_owner};
    YVEX_TEST_ASSERT(pthread_create(&active_id, NULL, decoder_test_push_main, &active) == 0,
                     "active decoder call starts");
    (void)pthread_mutex_lock(&gate.mutex);
    while (!gate.entered)
        (void)pthread_cond_wait(&gate.condition, &gate.mutex);
    (void)pthread_mutex_unlock(&gate.mutex);
    YVEX_TEST_ASSERT(pthread_create(&close_id, NULL, decoder_test_close_main, &close) == 0 &&
                         pthread_create(&contender_id, NULL, decoder_test_contender_main,
                                        &contender) == 0,
                     "close and contender start while decoder is active");
    (void)pthread_join(contender_id, NULL);
    (void)pthread_mutex_lock(&gate.mutex);
    gate.release = 1;
    (void)pthread_cond_broadcast(&gate.condition);
    (void)pthread_mutex_unlock(&gate.mutex);
    (void)pthread_join(active_id, NULL);
    (void)pthread_join(close_id, NULL);
    decoder = NULL;
    YVEX_TEST_ASSERT(active.rc == YVEX_ERR_CANCELLED && contender.closing_refusal &&
                         contender.rc == YVEX_ERR_STATE && contender.attempts <= 1000000u &&
                         close_owner == NULL,
                     "close drains ACTIVE and atomically excludes entry after CLOSING");
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    return 0;
}

static int test_classification_and_append(void)
{
    yvex_tokenizer tokenizer;
    yvex_token_info tokens[4];
    yvex_tokenizer_token_classification classification;
    yvex_token_sequence *first = NULL, *second = NULL;
    yvex_token_sequence_summary before, after, isolated;
    yvex_error err;
    unsigned long long ordinal;
    fixture_open(&tokenizer, tokens);
    YVEX_TEST_ASSERT(yvex_tokenizer_token_classify(&tokenizer, 0u, &classification, &err) == YVEX_OK &&
                         classification.eos && classification.stop && classification.special,
                     "EOS classification is tokenizer-owned");
    YVEX_TEST_ASSERT(yvex_token_sequence_open(&first, 2u, &err) == YVEX_OK &&
                         yvex_token_sequence_open(&second, 2u, &err) == YVEX_OK,
                     "open isolated append directories");
    YVEX_TEST_ASSERT(yvex_token_sequence_append(first, 3u, 4u, &ordinal, &err) == YVEX_OK &&
                         ordinal == 0u &&
                         yvex_token_sequence_summary_get(first, &before, &err) == YVEX_OK,
                     "append proposed token");
    YVEX_TEST_ASSERT(yvex_token_sequence_transition(first, 0u, YVEX_TOKEN_APPEND_PROPOSED,
                                                     YVEX_TOKEN_APPEND_APPENDED, &err) == YVEX_OK &&
                         yvex_token_sequence_summary_get(first, &after, &err) == YVEX_OK &&
                         after.generation == before.generation + 1u,
                     "advance one exact append state");
    YVEX_TEST_ASSERT(yvex_token_sequence_transition(first, 0u, YVEX_TOKEN_APPEND_PROPOSED,
                                                     YVEX_TOKEN_APPEND_APPENDED, &err) == YVEX_ERR_STATE &&
                         yvex_token_sequence_summary_get(first, &isolated, &err) == YVEX_OK &&
                         strcmp(isolated.state_identity, after.state_identity) == 0,
                     "stale transition preserves state");
    YVEX_TEST_ASSERT(yvex_token_sequence_summary_get(second, &isolated, &err) == YVEX_OK &&
                         isolated.count == 0u && isolated.generation == 0u,
                     "separate append state is isolated");
    YVEX_TEST_ASSERT(yvex_token_sequence_reset(first, &err) == YVEX_OK &&
                         yvex_token_sequence_summary_get(first, &isolated, &err) == YVEX_OK &&
                         isolated.count == 0u && isolated.generation > after.generation,
                     "append reset retains capacity and advances directory identity state");
    yvex_token_sequence_close(&first);
    yvex_token_sequence_close(&second);
    yvex_token_sequence_close(&second);
    return 0;
}

static int test_candidate_transactions(void)
{
    yvex_tokenizer tokenizer;
    yvex_token_info tokens[4];
    yvex_tokenizer_decode_options options = {
        .skip_special_tokens = 1, .require_complete_utf8 = 1};
    yvex_tokenizer_decoder *decoder = NULL;
    yvex_tokenizer_decoder_transaction *decoder_tx = NULL;
    yvex_token_sequence *sequence = NULL;
    yvex_token_sequence_transaction *sequence_tx = NULL;
    yvex_tokenizer_fragment fragment = {0};
    yvex_token_sequence_summary before, staged, published;
    yvex_error err;
    unsigned long long ordinal, index;
    fixture_open(&tokenizer, tokens);
    YVEX_TEST_ASSERT(
        yvex_tokenizer_decoder_open(&decoder, &tokenizer, &options, &err) ==
                YVEX_OK &&
            yvex_tokenizer_decoder_push(decoder, 3u, &fragment, &err) ==
                YVEX_OK,
        "decoder transaction fixture opens");
    yvex_tokenizer_fragment_clear(&fragment);
    YVEX_TEST_ASSERT(
        yvex_tokenizer_decoder_transaction_begin(decoder, &decoder_tx,
                                                  &err) == YVEX_OK &&
            yvex_tokenizer_decoder_transaction_push(decoder_tx, 1u,
                                                     &fragment, &err) ==
                YVEX_OK,
        "decoder candidate begins without publication");
    yvex_tokenizer_fragment_clear(&fragment);
    YVEX_TEST_ASSERT(
        yvex_tokenizer_decoder_transaction_push(decoder_tx, 2u, &fragment,
                                                 &err) == YVEX_OK &&
            fragment.byte_count == 4u &&
            yvex_tokenizer_decoder_transaction_prepare(decoder_tx, &err) ==
                YVEX_OK,
        "decoder candidate prepares after its complete staged fragment");
    yvex_tokenizer_fragment_clear(&fragment);
    YVEX_TEST_ASSERT(
        yvex_tokenizer_decoder_push(decoder, 3u, &fragment, &err) ==
            YVEX_ERR_STATE,
        "prepared decoder candidate remains exclusive");
    yvex_tokenizer_decoder_transaction_abort(&decoder_tx);
    YVEX_TEST_ASSERT(
        yvex_tokenizer_decoder_push(decoder, 3u, &fragment, &err) == YVEX_OK &&
            fragment.processed_token_count == 2u,
        "decoder abort preserves the earlier published state");
    yvex_tokenizer_fragment_clear(&fragment);
    YVEX_TEST_ASSERT(
        yvex_tokenizer_decoder_transaction_begin(decoder, &decoder_tx,
                                                  &err) == YVEX_OK &&
            yvex_tokenizer_decoder_transaction_push(decoder_tx, 3u,
                                                     &fragment, &err) ==
                YVEX_OK &&
            yvex_tokenizer_decoder_transaction_prepare(decoder_tx, &err) ==
                YVEX_OK,
        "decoder candidate prepares for publication");
    yvex_tokenizer_fragment_clear(&fragment);
    yvex_tokenizer_decoder_transaction_publish(&decoder_tx);
    YVEX_TEST_ASSERT(
        yvex_tokenizer_decoder_push(decoder, 3u, &fragment, &err) == YVEX_OK &&
            fragment.processed_token_count == 4u,
        "decoder publication advances the complete candidate once");
    yvex_tokenizer_fragment_clear(&fragment);

    YVEX_TEST_ASSERT(yvex_token_sequence_open(&sequence, 4u, &err) == YVEX_OK &&
                         yvex_token_sequence_summary_get(sequence, &before,
                                                         &err) == YVEX_OK &&
                         yvex_token_sequence_transaction_begin(
                             sequence, 2u, &sequence_tx, &err) == YVEX_OK,
                     "token-ledger candidate begins");
    for (index = 0u; index < 2u; ++index) {
        YVEX_TEST_ASSERT(
            yvex_token_sequence_transaction_append(
                sequence_tx, 2u + (unsigned int)index, 4u, &ordinal, &err) ==
                YVEX_OK,
            "token-ledger candidate appends");
        YVEX_TEST_ASSERT(
            yvex_token_sequence_transaction_transition(
                sequence_tx, ordinal, YVEX_TOKEN_APPEND_PROPOSED,
                YVEX_TOKEN_APPEND_APPENDED, &err) == YVEX_OK &&
                yvex_token_sequence_transaction_transition(
                    sequence_tx, ordinal, YVEX_TOKEN_APPEND_APPENDED,
                    YVEX_TOKEN_APPEND_SUBMITTED, &err) == YVEX_OK &&
                yvex_token_sequence_transaction_transition(
                    sequence_tx, ordinal, YVEX_TOKEN_APPEND_SUBMITTED,
                    YVEX_TOKEN_APPEND_MODEL_COMMITTED, &err) == YVEX_OK &&
                yvex_token_sequence_transaction_transition(
                    sequence_tx, ordinal, YVEX_TOKEN_APPEND_MODEL_COMMITTED,
                    YVEX_TOKEN_APPEND_DETOKENIZED, &err) == YVEX_OK &&
                yvex_token_sequence_transaction_transition(
                    sequence_tx, ordinal, YVEX_TOKEN_APPEND_DETOKENIZED,
                    YVEX_TOKEN_APPEND_TEXT_PUBLISHED, &err) == YVEX_OK,
            "token-ledger candidate reaches publication state");
    }
    YVEX_TEST_ASSERT(
        yvex_token_sequence_summary_get(sequence, &staged, &err) == YVEX_OK &&
            staged.count == before.count &&
            yvex_token_sequence_transaction_prepare(sequence_tx, &err) ==
                YVEX_OK,
        "staged token-ledger prefix remains invisible");
    yvex_token_sequence_transaction_publish(&sequence_tx);
    YVEX_TEST_ASSERT(
        yvex_token_sequence_summary_get(sequence, &published, &err) ==
                YVEX_OK &&
            published.count == 2u && published.generation > before.generation,
        "token-ledger prefix publishes atomically");
    yvex_token_sequence_close(&sequence);
    yvex_tokenizer_decoder_close(&decoder);
    return 0;
}

int yvex_test_runtime_tokenizer(void)
{
    if (test_nfc_normalization() != 0)
        return 1;
    if (test_compiled_family_policy() != 0)
        return 1;
    if (test_reasoning_channel() != 0)
        return 1;
    if (test_source_prompt_modes() != 0)
        return 1;
    if (test_source_multiturn_and_tools() != 0)
        return 1;
    if (test_source_developer_boundary() != 0)
        return 1;
    if (test_provider_projection() != 0)
        return 1;
    if (test_incremental_decoder() != 0)
        return 1;
    if (test_incremental_close_drain() != 0)
        return 1;
    if (test_candidate_transactions() != 0)
        return 1;
    return test_classification_and_append();
}
