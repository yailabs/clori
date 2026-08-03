/*
 * Verify mutable tokenizer request state remains isolated and transactional. Synthetic pieces
 * exercise only public contracts and never enter production objects. Sanitizer-visible invariant
 * proof; target-scale BPE parity belongs to the live lane.
 */
#include "src/tokenizer/private.h"
#include "tests/test.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>

#define TEST_DSML "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c"

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
    tokenizer->plan.prompt_policy = YVEX_TOKENIZER_PROMPT_DEEPSEEK_V4;
    tokenizer->plan.explicit_reasoning_supported = 1;
    tokenizer->plan.maximum_reasoning_supported = 1;
    memcpy(tokenizer->plan.tokenizer_plan_identity, identity, sizeof(identity));
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
    tools[0].parameters_json = (yvex_provider_span){schema,
                                                    sizeof(schema) - 1u};
    request->schema_version = YVEX_PROVIDER_SCHEMA_V1;
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
                         result.kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL,
                     "exact DSML completion becomes one typed function call");
    YVEX_TEST_ASSERT_STREQ(result.tool_call.name, "get_match_context",
                           "tool name remains exact");
    YVEX_TEST_ASSERT(result.tool_call.call_id[0] &&
                         yvex_provider_json_value_validate(
                             result.tool_call.arguments_json.bytes,
                             result.tool_call.arguments_json.count, 1,
                             &err) == YVEX_OK,
                     "tool arguments and call identity are valid");
    yvex_tokenizer_provider_result_clear(&result);
    YVEX_TEST_ASSERT(yvex_tokenizer_parse_provider_completion(
                         &tokenizer, &request, prose, sizeof(prose) - 1u,
                         &result, &err) == YVEX_OK &&
                         result.kind == YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
                     "ordinary prose never becomes a tool call");
    yvex_tokenizer_provider_result_clear(&result);
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
    if (test_reasoning_channel() != 0)
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
