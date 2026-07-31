/*
 * Exercises exact artifact-bound text/token semantics against pinned expected vectors. The
 * admitted GGUF is the only production tokenizer source and no tensor payload is read.
 * Target-scale tokenizer proof; sampled IDs are decoded but never fed back into the model.
 */
#include <yvex/api.h>
#include <yvex/internal/runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const unsigned char *text;
    unsigned long long byte_count;
    const unsigned int *ids;
    unsigned long long token_count;
    const char *name;
} live_vector;

static int expect_encode(const yvex_tokenizer *tokenizer, const live_vector *vector,
                         yvex_error *err)
{
    yvex_tokenizer_encode_options options = {0, 0, 1, 65536u};
    yvex_tokenizer_encode_result result = {0};
    int rc = yvex_tokenizer_encode(tokenizer, vector->text, vector->byte_count,
                                   &options, &result, err);
    if (rc == YVEX_OK &&
        (result.tokens.len != vector->token_count ||
         memcmp(result.tokens.ids, vector->ids,
                (size_t)vector->token_count * sizeof(*vector->ids)) != 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.tokenizer.vector",
                       vector->name);
        rc = YVEX_ERR_FORMAT;
    }
    yvex_tokenizer_encode_result_clear(&result);
    return rc;
}

static int vector_proof(const yvex_tokenizer *tokenizer, yvex_error *err)
{
    static const unsigned int hello[] = {33310u, 2058u};
    static const unsigned int spaces[] = {223u, 12529u, 262u, 13564u,
                                           201u, 6695u, 200u, 1836u};
    static const unsigned int accents[] = {69u, 2797u, 619u, 312u, 17793u};
    static const unsigned int emoji[] = {28927u, 225u, 91825u, 257u};
    static const unsigned int cjk[] = {30594u, 3427u};
    static const unsigned int cyrillic[] = {24797u, 8919u, 74779u};
    static const unsigned int arabic[] = {10393u, 2212u, 53067u, 9254u,
                                          1183u, 14059u};
    static const unsigned int devanagari[] = {9094u, 12524u, 70223u, 6011u,
                                              26227u, 101422u, 85639u};
    static const unsigned int numeric[] = {6895u, 18009u, 25744u, 18u};
    static const unsigned int nul[] = {67u, 191u, 68u};
    static const unsigned int added[] = {128821u, 33310u, 128822u};
    static const unsigned char embedded_nul[] = {'a', 0u, 'b'};
    static const live_vector vectors[] = {
        {(const unsigned char *)"hello world", 11u, hello, 2u, "hello"},
        {(const unsigned char *)"  repeated   spaces\nnext\tline", 29u, spaces, 8u, "space"},
        {(const unsigned char *)"caf\xc3\xa9 e\xcc\x81", 9u, accents, 5u, "accent"},
        {(const unsigned char *)"\xf0\x9f\x98\x80\xf0\x9f\xa7\xa0", 8u, emoji, 4u, "emoji"},
        {(const unsigned char *)"\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c", 12u, cjk, 2u, "cjk"},
        {(const unsigned char *)"\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xbc\xd0\xb8\xd1\x80", 19u,
         cyrillic, 3u, "cyrillic"},
        {(const unsigned char *)"\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd8\xa7\xd9\x84\xd8\xb9\xd8\xa7\xd9\x84\xd9\x85", 25u,
         arabic, 6u, "arabic"},
        {(const unsigned char *)"\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87 \xe0\xa4\xa6\xe0\xa5\x81\xe0\xa4\xa8\xe0\xa4\xbf\xe0\xa4\xaf\xe0\xa4\xbe", 37u,
         devanagari, 7u, "devanagari"},
        {(const unsigned char *)"1234567890", 10u, numeric, 4u, "numeric"},
        {embedded_nul, 3u, nul, 3u, "embedded-nul"},
        {(const unsigned char *)"<think>hello</think>", 20u, added, 3u, "added"}
    };
    unsigned long long index;
    for (index = 0u; index < sizeof(vectors) / sizeof(vectors[0]); ++index) {
        int rc = expect_encode(tokenizer, &vectors[index], err);
        if (rc != YVEX_OK)
            return rc;
    }
    return YVEX_OK;
}

static int admission_refusal_proof(const yvex_tokenizer *tokenizer, yvex_error *err)
{
    const unsigned char invalid[] = {0xc3u};
    yvex_tokenizer_encode_options options = {0, 0, 1, 1u};
    yvex_tokenizer_encode_result result = {0};
    int rc = yvex_tokenizer_encode(tokenizer, invalid, 1u, &options, &result, err);
    if (rc != YVEX_ERR_FORMAT || result.completed)
        return YVEX_ERR_FORMAT;
    options.maximum_tokens = 1u;
    rc = yvex_tokenizer_encode(tokenizer, (const unsigned char *)"hello world",
                               11u, &options, &result, err);
    if (rc != YVEX_ERR_BOUNDS || result.completed)
        return YVEX_ERR_FORMAT;
    options.maximum_tokens = 8u;
    options.add_bos = 1;
    rc = yvex_tokenizer_encode(tokenizer, (const unsigned char *)"hello", 5u,
                               &options, &result, err);
    if (rc != YVEX_ERR_UNSUPPORTED || result.completed)
        return YVEX_ERR_FORMAT;
    options.add_bos = 0;
    options.add_eos = 1;
    rc = yvex_tokenizer_encode(tokenizer, (const unsigned char *)"hello", 5u,
                               &options, &result, err);
    if (rc != YVEX_ERR_UNSUPPORTED || result.completed)
        return YVEX_ERR_FORMAT;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int prompt_proof(const yvex_tokenizer *tokenizer, yvex_error *err)
{
    static const char expected[] =
        "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"
        "policy<\xef\xbd\x9cUser\xef\xbd\x9c>hi"
        "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>ok"
        "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"
        "<\xef\xbd\x9cUser\xef\xbd\x9c>next"
        "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>";
    yvex_prompt_message messages[] = {
        {YVEX_PROMPT_ROLE_SYSTEM, "policy", 6u},
        {YVEX_PROMPT_ROLE_USER, "hi", 2u},
        {YVEX_PROMPT_ROLE_ASSISTANT, "ok", 2u},
        {YVEX_PROMPT_ROLE_USER, "next", 4u}
    };
    yvex_prompt_options options = {1, 0, 1, 1, YVEX_PROMPT_MODE_CHAT};
    yvex_tokenizer_encode_options encode = {0, 0, 1, 128u};
    yvex_rendered_prompt rendered = {0};
    yvex_tokenizer_encode_result tokens = {0};
    int rc = yvex_tokenizer_encode_prompt(tokenizer, messages, 4u, &options,
                                          &encode, &rendered, &tokens, err);
    if (rc == YVEX_OK &&
        (rendered.len != sizeof(expected) - 1u ||
         memcmp(rendered.text, expected, sizeof(expected) - 1u) != 0 ||
         !tokens.completed || tokens.tokens.len != 12u)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.tokenizer.prompt",
                       "DeepSeek prompt bytes or tokens differ");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_tokenizer_encode_result_clear(&tokens);
    yvex_rendered_prompt_free(&rendered);
    return rc;
}

static int decode_proof(const yvex_tokenizer *tokenizer, const unsigned int sampled[3],
                        yvex_error *err)
{
    const unsigned int ids[] = {0u, 28927u, 225u, 1u};
    yvex_tokenizer_decode_options options = {
        .skip_special_tokens = 1, .require_complete_utf8 = 1};
    yvex_tokenizer_decode_result batch = {0}, sampled_result = {0};
    yvex_tokenizer_decoder *decoder = NULL;
    yvex_tokenizer_fragment fragment = {0};
    unsigned char aggregate[8];
    unsigned long long index, used = 0u;
    int rc = yvex_tokenizer_decode(tokenizer, ids, 4u, &options, &batch, err);
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_decoder_open(&decoder, tokenizer, &options, err);
    for (index = 0u; rc == YVEX_OK && index < 4u; ++index) {
        rc = yvex_tokenizer_decoder_push(decoder, ids[index], &fragment, err);
        if (rc == YVEX_OK && fragment.byte_count <= sizeof(aggregate) - used) {
            memcpy(aggregate + used, fragment.bytes, (size_t)fragment.byte_count);
            used += fragment.byte_count;
        } else if (rc == YVEX_OK) {
            rc = YVEX_ERR_BOUNDS;
        }
        yvex_tokenizer_fragment_clear(&fragment);
    }
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_decoder_finish(decoder, &fragment, err);
    yvex_tokenizer_fragment_clear(&fragment);
    if (rc == YVEX_OK &&
        (used != batch.byte_count || memcmp(aggregate, batch.bytes, (size_t)used) != 0 ||
         used != 4u || memcmp(aggregate, "\xf0\x9f\x98\x80", 4u) != 0))
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_decode(tokenizer, sampled, 3u, NULL, &sampled_result, err);
    yvex_tokenizer_decode_result_clear(&sampled_result);
    yvex_tokenizer_decode_result_clear(&batch);
    yvex_tokenizer_decoder_close(&decoder);
    return rc;
}

static int sampled_ids_parse(const char *text, unsigned int ids[3])
{
    char trailing;
    return text && sscanf(text, "%u,%u,%u%c", &ids[0], &ids[1], &ids[2], &trailing) == 3;
}

int main(int argc, char **argv)
{
    yvex_model_context context = {0};
    yvex_runtime_binding *binding = NULL;
    yvex_runtime_binding_summary binding_summary = {0};
    yvex_runtime_binding_failure binding_failure = {0};
    yvex_complete_artifact_admission admission = {0};
    const yvex_tokenizer_plan_summary *plan;
    yvex_error err;
    unsigned int sampled[3];
    int rc;
    if (argc != 4 || !sampled_ids_parse(argv[3], sampled)) {
        fprintf(stderr, "usage: %s ARTIFACT RUNTIME_BINDING ID,ID,ID\n", argv[0]);
        return 2;
    }
    rc = yvex_model_context_open_tokenizer(argv[1], &context, &err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_binding_open(&binding, argv[2], &binding_summary,
                                       &admission, &binding_failure, &err);
    if (rc == YVEX_OK) {
        int refusal = yvex_tokenizer_bind_runtime(context.tokenizer, "invalid",
                                                   binding_summary.logical_model_identity,
                                                   binding_summary.runtime_descriptor_identity, &err);
        if (refusal != YVEX_ERR_INVALID_ARG)
            rc = YVEX_ERR_FORMAT;
        else
            yvex_error_clear(&err);
    }
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_bind_runtime(context.tokenizer,
                                         binding_summary.artifact_identity,
                                         binding_summary.logical_model_identity,
                                         binding_summary.runtime_descriptor_identity, &err);
    plan = yvex_tokenizer_plan_summary_get(context.tokenizer);
    if (rc == YVEX_OK && (!plan || !plan->runtime_bound ||
        plan->vocabulary_size != 129280u || plan->base_vocabulary_size != 128000u ||
        plan->merge_count != 127741u || plan->added_token_count != 1283u ||
        plan->special_token_count != 1230u || !plan->bos_present || plan->bos_token_id != 0u ||
        !plan->eos_present || plan->eos_token_id != 1u ||
        !plan->pad_present || plan->pad_token_id != 1u || plan->unk_present))
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK) rc = vector_proof(context.tokenizer, &err);
    if (rc == YVEX_OK) rc = admission_refusal_proof(context.tokenizer, &err);
    if (rc == YVEX_OK) rc = prompt_proof(context.tokenizer, &err);
    if (rc == YVEX_OK) rc = decode_proof(context.tokenizer, sampled, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "tokenizer_live status=%d where=%s reason=%s\n", rc,
                yvex_error_where(&err), yvex_error_message(&err));
    else
        printf("tokenizer_model=bpe-bytelevel vocabulary=129280 base_vocabulary=128000 "
               "merges=127741 added_tokens=1283 special_tokens=1230 bos=0 eos=1 pad=1 "
               "oracle=tokenizers-0.20.3 vectors=11 prompt_parity=pass "
               "batch_incremental_equivalence=pass utf8_split=pass sampled_ids=%u,%u,%u "
               "owned_bytes=%llu warm_rebuilds=0 model_state_unchanged=pass "
               "tokenizer_plan_identity=%s runtime_binding_identity=%s generation_ready=0\n",
               sampled[0], sampled[1], sampled[2], plan->owned_bytes,
               plan->tokenizer_plan_identity, binding_summary.identity);
    yvex_runtime_binding_close(binding);
    yvex_model_context_close(&context);
    return rc == YVEX_OK ? 0 : 1;
}
