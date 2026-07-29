/* Owner: tests.unit.runtime_tokenizer.
 * Owns: focused tokenizer runtime, incremental decoder, append, rollback, and lifecycle proof.
 * Does not own: production tokenizer policy, selected-artifact parity, model execution, or generation.
 * Invariants: synthetic pieces exercise only public contracts and never enter production objects.
 * Boundary: sanitizer-visible invariant proof; target-scale BPE parity belongs to the live lane.
 * Purpose: verify mutable tokenizer request state remains isolated and transactional.
 * Inputs: one bounded test-owned tokenizer fixture. Effects: allocates only test-local state.
 * Failure: reports the first violated tokenizer runtime contract. */
#include "src/tokenizer/private.h"
#include "tests/test.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>

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

/* Purpose: hold one admitted decoder call so close and contender transitions are observable. */
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

/* Purpose: execute the active cancellable decoder call from a test-owned thread. */
static void *decoder_test_push_main(void *opaque)
{
    decoder_test_call *call = (decoder_test_call *)opaque;
    yvex_tokenizer_fragment fragment;
    yvex_error err;
    call->rc = yvex_tokenizer_decoder_push(call->decoder, 3u, &fragment, &err);
    yvex_tokenizer_fragment_clear(&fragment);
    return NULL;
}

/* Purpose: observe typed CLOSING admission while the first call retains active ownership. */
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

/* Purpose: transfer unique close ownership and drain the active decoder call. */
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
    memcpy(tokenizer->plan.tokenizer_plan_identity, identity, sizeof(identity));
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

int yvex_test_runtime_tokenizer(void)
{
    if (test_incremental_decoder() != 0)
        return 1;
    if (test_incremental_close_drain() != 0)
        return 1;
    return test_classification_and_append();
}
