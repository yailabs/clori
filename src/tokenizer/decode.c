/*
 * Provide batch and one-token-at-a-time decoding over the exact shared tokenizer plan.
 *
 * A pushed token either publishes one complete fragment and state transition or changes nothing.
 * Converts admitted numeric IDs to bytes/events; it never advances KV or declares generation
 * complete.
 */

#include "src/tokenizer/private.h"

#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/artifact.h>
#include <yvex/internal/core.h>

enum { DECODER_OPEN = 0u, DECODER_ACTIVE = 1u, DECODER_CLOSING = 2u };

struct yvex_tokenizer_decoder {
    const yvex_tokenizer *tokenizer;
    yvex_tokenizer_decode_options options;
    unsigned char pending[4];
    unsigned long long pending_count, processed_token_count;
    char state_identity[YVEX_SHA256_HEX_CAP];
    atomic_uint lifecycle;
    atomic_uint transaction_active;
    pthread_mutex_t drain_mutex;
    pthread_cond_t drain_condition;
    int drain_mutex_ready, drain_condition_ready;
};

struct yvex_tokenizer_decoder_transaction {
    yvex_tokenizer_decoder *decoder;
    unsigned char pending[4];
    unsigned long long pending_count, processed_token_count;
    char base_identity[YVEX_SHA256_HEX_CAP];
    char state_identity[YVEX_SHA256_HEX_CAP];
    int prepared;
};

typedef struct {
    unsigned char *data;
    unsigned long long count, capacity;
} decode_builder;

static int decode_cancelled(const yvex_tokenizer_decode_options *options, yvex_error *err)
{
    if (!options || !options->cancelled || !options->cancelled(options->cancel_context))
        return YVEX_OK;
    yvex_error_set(err, YVEX_ERR_CANCELLED, "tokenizer.decode.cancelled",
                   "tokenizer decoding was cancelled before publication");
    return YVEX_ERR_CANCELLED;
}

static int codepoint_byte(uint32_t point, unsigned char *byte)
{
    unsigned int candidate, extra = 0u;
    if (point <= 255u &&
        ((point >= 33u && point <= 126u) || (point >= 161u && point <= 172u) ||
         (point >= 174u && point <= 255u))) {
        *byte = (unsigned char)point;
        return 1;
    }
    if (point < 256u)
        return 0;
    for (candidate = 0u; candidate < 256u; ++candidate) {
        if ((candidate >= 33u && candidate <= 126u) ||
            (candidate >= 161u && candidate <= 172u) ||
            (candidate >= 174u && candidate <= 255u))
            continue;
        if (256u + extra == point) {
            *byte = (unsigned char)candidate;
            return 1;
        }
        ++extra;
    }
    return 0;
}

int yvex_tokenizer_token_classify(
    const yvex_tokenizer *tokenizer, unsigned int token_id,
    yvex_tokenizer_token_classification *classification, yvex_error *err)
{
    const yvex_token_info *token;
    if (!tokenizer || !classification || token_id >= tokenizer->vocab_size) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.classify", "token ID is outside vocabulary");
        return YVEX_ERR_BOUNDS;
    }
    token = &tokenizer->tokens[token_id];
    memset(classification, 0, sizeof(*classification));
    classification->token_id = token_id;
    classification->special = token->type == YVEX_TOKEN_TYPE_CONTROL;
    classification->eos = tokenizer->eos.present && token_id == tokenizer->eos.id;
    classification->pad = tokenizer->pad.present && token_id == tokenizer->pad.id;
    classification->unknown = tokenizer->unk.present && token_id == tokenizer->unk.id;
    classification->stop = classification->eos;
    classification->suppressed_by_default = classification->special;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int tokenizer_piece_decode(const yvex_tokenizer *tokenizer, unsigned int token_id,
                                  int skip_special, unsigned char **bytes,
                                  unsigned long long *byte_count, int *suppressed,
                                  yvex_error *err)
{
    const yvex_token_info *token;
    unsigned char *output;
    unsigned long long input_offset = 0u, output_count = 0u;

    if (bytes) *bytes = NULL;
    if (byte_count) *byte_count = 0u;
    if (suppressed) *suppressed = 0;
    if (!tokenizer || !bytes || !byte_count || !suppressed || token_id >= tokenizer->vocab_size) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.decode.piece",
                       "valid tokenizer, outputs, and token ID are required");
        return YVEX_ERR_BOUNDS;
    }
    token = &tokenizer->tokens[token_id];
    if (skip_special && token->type == YVEX_TOKEN_TYPE_CONTROL) {
        *suppressed = 1;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    output = malloc(token->text_len ? (size_t)token->text_len : 1u);
    if (!output) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.decode.piece", "piece allocation failed");
        return YVEX_ERR_NOMEM;
    }
    if (token->type == YVEX_TOKEN_TYPE_CONTROL ||
        token->type == YVEX_TOKEN_TYPE_USER_DEFINED) {
        if (token->text_len)
            memcpy(output, token->text, (size_t)token->text_len);
        output_count = token->text_len;
    } else {
        while (input_offset < token->text_len) {
            uint32_t point;
            unsigned char byte;
            if (!yvex_tokenizer_utf8_next((const unsigned char *)token->text,
                                          token->text_len, &input_offset, &point) ||
                !codepoint_byte(point, &byte)) {
                free(output);
                yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.decode.piece",
                               "vocabulary piece is not canonical ByteLevel text");
                return YVEX_ERR_FORMAT;
            }
            output[output_count++] = byte;
        }
    }
    *bytes = output;
    *byte_count = output_count;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int decode_reserve(decode_builder *builder, unsigned long long add, yvex_error *err)
{
    unsigned long long need, capacity;
    unsigned char *grown;
    if (builder->count > ULLONG_MAX - add) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.decode", "decoded extent overflow");
        return YVEX_ERR_BOUNDS;
    }
    need = builder->count + add;
    if (need <= builder->capacity)
        return YVEX_OK;
    capacity = builder->capacity ? builder->capacity : 128u;
    while (capacity < need) {
        if (capacity > ULLONG_MAX / 2u || capacity * 2u > SIZE_MAX)
            return YVEX_ERR_NOMEM;
        capacity *= 2u;
    }
    grown = realloc(builder->data, (size_t)capacity);
    if (!grown) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.decode", "decoded buffer allocation failed");
        return YVEX_ERR_NOMEM;
    }
    builder->data = grown;
    builder->capacity = capacity;
    return YVEX_OK;
}

static int decode_append(decode_builder *builder, const unsigned char *bytes,
                         unsigned long long count, yvex_error *err)
{
    int rc = decode_reserve(builder, count, err);
    if (rc == YVEX_OK && count)
        memcpy(builder->data + builder->count, bytes, (size_t)count);
    if (rc == YVEX_OK)
        builder->count += count;
    return rc;
}

static int decode_identities(const yvex_tokenizer *tokenizer, const unsigned int *ids,
                             unsigned long long count, yvex_tokenizer_decode_result *result)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.token-ids.v1") ||
        !yvex_sha256_update_u64_be(&hash, count))
        return 0;
    for (index = 0u; index < count; ++index)
        if (!yvex_sha256_update_u64_be(&hash, ids[index]))
            return 0;
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, result->token_ids_identity);
    if (yvex_artifact_sha256_hex_bytes(result->bytes, result->byte_count,
                                       result->decoded_bytes_identity, NULL) != YVEX_OK)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.decode.v1") ||
        !yvex_sha256_update_text(&hash, tokenizer->plan.tokenizer_plan_identity) ||
        !yvex_sha256_update_text(&hash, result->token_ids_identity) ||
        !yvex_sha256_update_text(&hash, result->decoded_bytes_identity) ||
        !yvex_sha256_update_u64_be(&hash, result->suppressed_special_count) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, result->decoder_identity);
    return 1;
}

int yvex_tokenizer_decode(const yvex_tokenizer *tokenizer,
                          const unsigned int *ids,
                          unsigned long long count,
                          const yvex_tokenizer_decode_options *options,
                          yvex_tokenizer_decode_result *result,
                          yvex_error *err)
{
    yvex_tokenizer_decode_options defaults = {0};
    yvex_tokenizer_decode_result candidate;
    decode_builder builder = {0};
    unsigned long long index;
    int rc = YVEX_OK;
    if (!options) options = &defaults;
    if (!tokenizer || !tokenizer->plan.sealed || !result || (!ids && count)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.decode", "sealed tokenizer, IDs, and result are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(&candidate, 0, sizeof(candidate));
    for (index = 0u; index < count && rc == YVEX_OK; ++index) {
        unsigned char *piece = NULL;
        unsigned long long piece_count = 0u;
        int suppressed = 0;
        rc = decode_cancelled(options, err);
        if (rc == YVEX_OK)
            rc = tokenizer_piece_decode(tokenizer, ids[index], options->skip_special_tokens,
                                        &piece, &piece_count, &suppressed, err);
        if (rc == YVEX_OK)
            rc = decode_append(&builder, piece, piece_count, err);
        candidate.suppressed_special_count += suppressed;
        free(piece);
    }
    if (rc == YVEX_OK && options->require_complete_utf8) {
        unsigned long long offset = 0u;
        uint32_t point;
        while (offset < builder.count &&
               yvex_tokenizer_utf8_next(builder.data, builder.count, &offset, &point)) {}
        if (offset != builder.count) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.decode.utf8", "batch output is not complete UTF-8");
            rc = YVEX_ERR_FORMAT;
        }
    }
    if (rc == YVEX_OK) {
        candidate.schema_version = YVEX_TOKENIZER_EXECUTION_SCHEMA_V1;
        candidate.bytes = builder.data;
        candidate.byte_count = builder.count;
        candidate.token_count = count;
        candidate.completed = 1;
        if (!decode_identities(tokenizer, ids, count, &candidate)) {
            yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decode.identity", "decode identity derivation failed");
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc != YVEX_OK) {
        free(builder.data);
        memset(result, 0, sizeof(*result));
        return rc;
    }
    *result = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_tokenizer_decode_result_clear(yvex_tokenizer_decode_result *result)
{
    if (!result) return;
    free(result->bytes);
    memset(result, 0, sizeof(*result));
}

static int decoder_state_identity(const yvex_tokenizer_decoder *decoder,
                                  const unsigned char pending[4],
                                  unsigned long long pending_count,
                                  unsigned long long processed,
                                  char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.incremental-state.v1") ||
        !yvex_sha256_update_text(&hash, decoder->tokenizer->plan.tokenizer_plan_identity) ||
        !yvex_sha256_update_u64_be(&hash, decoder->options.skip_special_tokens) ||
        !yvex_sha256_update_u64_be(&hash, decoder->options.require_complete_utf8) ||
        !yvex_sha256_update_u64_be(&hash, processed) ||
        !yvex_sha256_update_u64_be(&hash, pending_count) ||
        !yvex_sha256_update(&hash, pending, (size_t)pending_count) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int decoder_enter(yvex_tokenizer_decoder *decoder, yvex_error *err)
{
    unsigned int expected = DECODER_OPEN;
    if (decoder && atomic_compare_exchange_strong_explicit(
                       &decoder->lifecycle, &expected, DECODER_ACTIVE,
                       memory_order_acq_rel, memory_order_acquire))
        return YVEX_OK;
    yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.lifecycle",
                   expected & DECODER_CLOSING ? "decoder is closing" : "decoder is already in use");
    return YVEX_ERR_STATE;
}

static void decoder_leave(yvex_tokenizer_decoder *decoder)
{
    unsigned int observed = atomic_load_explicit(&decoder->lifecycle, memory_order_acquire);
    if (observed & DECODER_CLOSING && decoder->drain_mutex_ready &&
        pthread_mutex_lock(&decoder->drain_mutex) == 0) {
        (void)atomic_fetch_and_explicit(&decoder->lifecycle, ~DECODER_ACTIVE,
                                        memory_order_release);
        if (decoder->drain_condition_ready)
            (void)pthread_cond_broadcast(&decoder->drain_condition);
        (void)pthread_mutex_unlock(&decoder->drain_mutex);
        return;
    }
    (void)atomic_fetch_and_explicit(&decoder->lifecycle, ~DECODER_ACTIVE,
                                    memory_order_release);
}

static int utf8_tail_valid(const unsigned char *bytes, unsigned long long count)
{
    unsigned int needed;
    unsigned long long index;
    if (!count || count > 3u)
        return 0;
    if (bytes[0] >= 0xc2u && bytes[0] <= 0xdfu)
        needed = 2u;
    else if (bytes[0] >= 0xe0u && bytes[0] <= 0xefu)
        needed = 3u;
    else if (bytes[0] >= 0xf0u && bytes[0] <= 0xf4u)
        needed = 4u;
    else
        return 0;
    if (count >= needed)
        return 0;
    for (index = 1u; index < count; ++index)
        if ((bytes[index] & 0xc0u) != 0x80u)
            return 0;
    return 1;
}

static int utf8_prefix(const unsigned char *bytes, unsigned long long count,
                       unsigned long long *prefix)
{
    unsigned long long offset = 0u, prior = 0u;
    uint32_t point;
    while (offset < count) {
        prior = offset;
        if (!yvex_tokenizer_utf8_next(bytes, count, &offset, &point)) {
            if (utf8_tail_valid(bytes + prior, count - prior)) {
                *prefix = prior;
                return 1;
            }
            return 0;
        }
    }
    *prefix = count;
    return 1;
}

static int fragment_identity(yvex_tokenizer_fragment *fragment)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.fragment.v1") ||
        !yvex_sha256_update_u64_be(&hash, fragment->token_id) ||
        !yvex_sha256_update_u64_be(&hash, fragment->byte_count) ||
        !yvex_sha256_update(&hash, fragment->bytes, (size_t)fragment->byte_count) ||
        !yvex_sha256_update_u64_be(&hash, fragment->pending_byte_count) ||
        !yvex_sha256_update_u64_be(&hash, fragment->processed_token_count) ||
        !yvex_sha256_update_u64_be(&hash, fragment->special) ||
        !yvex_sha256_update_u64_be(&hash, fragment->eos) ||
        !yvex_sha256_update_u64_be(&hash, fragment->suppressed) ||
        !yvex_sha256_update_text(&hash, fragment->state_before_identity) ||
        !yvex_sha256_update_text(&hash, fragment->state_after_identity) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, fragment->fragment_identity);
    return 1;
}

static int decoder_stage_token(
    yvex_tokenizer_decoder *decoder, const unsigned char pending[4],
    unsigned long long pending_count, unsigned long long processed_token_count,
    const char *state_identity, unsigned int token_id,
    unsigned char next_pending[4], unsigned long long *next_pending_count,
    char next_state_identity[YVEX_SHA256_HEX_CAP],
    yvex_tokenizer_fragment *fragment, yvex_error *err)
{
    yvex_tokenizer_fragment candidate = {0};
    yvex_tokenizer_token_classification classification;
    unsigned char combined[4096], *piece = NULL;
    unsigned long long piece_count = 0ull, combined_count = 0ull, prefix = 0ull;
    int suppressed = 0, rc = decode_cancelled(&decoder->options, err);
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_token_classify(
            decoder->tokenizer, token_id, &classification, err);
    if (rc == YVEX_OK)
        rc = tokenizer_piece_decode(
            decoder->tokenizer, token_id, decoder->options.skip_special_tokens,
            &piece, &piece_count, &suppressed, err);
    if (rc == YVEX_OK && piece_count > sizeof(combined) - pending_count) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.decoder.push",
                       "one token piece exceeds decoder bound");
        rc = YVEX_ERR_BOUNDS;
    }
    if (rc == YVEX_OK) {
        combined_count = pending_count + piece_count;
        memcpy(combined, pending, (size_t)pending_count);
        if (piece_count) memcpy(combined + pending_count, piece, (size_t)piece_count);
        if (!utf8_prefix(combined, combined_count, &prefix)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.decoder.push",
                           "token produces malformed UTF-8 stream");
            rc = YVEX_ERR_FORMAT;
        }
    }
    if (rc == YVEX_OK && prefix) {
        candidate.bytes = malloc((size_t)prefix);
        if (!candidate.bytes) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.decoder.push",
                           "fragment allocation failed");
            rc = YVEX_ERR_NOMEM;
        } else {
            memcpy(candidate.bytes, combined, (size_t)prefix);
        }
    }
    if (rc == YVEX_OK) {
        *next_pending_count = combined_count - prefix;
        memset(next_pending, 0, 4u);
        if (*next_pending_count)
            memcpy(next_pending, combined + prefix, (size_t)*next_pending_count);
        candidate.schema_version = YVEX_TOKENIZER_DECODER_SCHEMA_V1;
        candidate.token_id = token_id;
        candidate.byte_count = prefix;
        candidate.pending_byte_count = *next_pending_count;
        candidate.processed_token_count = processed_token_count + 1ull;
        candidate.special = classification.special;
        candidate.eos = classification.eos;
        candidate.suppressed = suppressed;
        yvex_core_text_copy(candidate.state_before_identity,
                            sizeof(candidate.state_before_identity), state_identity);
        if (!decoder_state_identity(
                decoder, next_pending, *next_pending_count,
                candidate.processed_token_count, next_state_identity)) {
            yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.push",
                           "fragment state identity failed");
            rc = YVEX_ERR_STATE;
        } else {
            yvex_core_text_copy(candidate.state_after_identity,
                                sizeof(candidate.state_after_identity),
                                next_state_identity);
            if (!fragment_identity(&candidate)) {
                yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.push",
                               "fragment identity failed");
                rc = YVEX_ERR_STATE;
            } else {
                candidate.completed = 1;
                *fragment = candidate;
            }
        }
    }
    free(piece);
    if (rc != YVEX_OK) yvex_tokenizer_fragment_clear(&candidate);
    return rc;
}

int yvex_tokenizer_decoder_open(yvex_tokenizer_decoder **out,
                                const yvex_tokenizer *tokenizer,
                                const yvex_tokenizer_decode_options *options,
                                yvex_error *err)
{
    yvex_tokenizer_decode_options defaults = {0};
    yvex_tokenizer_decoder *decoder;
    if (!out || !tokenizer || !tokenizer->plan.sealed) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.decoder.open", "sealed tokenizer and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    decoder = calloc(1u, sizeof(*decoder));
    if (!decoder) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.decoder.open", "decoder allocation failed");
        return YVEX_ERR_NOMEM;
    }
    defaults.skip_special_tokens = 1;
    defaults.require_complete_utf8 = 1;
    decoder->tokenizer = tokenizer;
    decoder->options = options ? *options : defaults;
    atomic_init(&decoder->lifecycle, DECODER_OPEN);
    atomic_init(&decoder->transaction_active, 0u);
    if (pthread_mutex_init(&decoder->drain_mutex, NULL) != 0) {
        free(decoder);
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.open", "decoder drain lock failed");
        return YVEX_ERR_STATE;
    }
    decoder->drain_mutex_ready = 1;
    if (pthread_cond_init(&decoder->drain_condition, NULL) != 0) {
        (void)pthread_mutex_destroy(&decoder->drain_mutex);
        free(decoder);
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.open", "decoder drain condition failed");
        return YVEX_ERR_STATE;
    }
    decoder->drain_condition_ready = 1;
    if (!decoder_state_identity(decoder, decoder->pending, 0u, 0u, decoder->state_identity)) {
        (void)pthread_cond_destroy(&decoder->drain_condition);
        (void)pthread_mutex_destroy(&decoder->drain_mutex);
        free(decoder);
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.open", "initial state identity failed");
        return YVEX_ERR_STATE;
    }
    *out = decoder;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Stage, validate, and atomically publish one token's incremental decoded fragment.
 *
 * Rollback.
 */
int yvex_tokenizer_decoder_push(yvex_tokenizer_decoder *decoder,
                                unsigned int token_id,
                                yvex_tokenizer_fragment *fragment,
                                yvex_error *err)
{
    unsigned char next_pending[4];
    unsigned long long next_pending_count = 0ull;
    char next_state_identity[YVEX_SHA256_HEX_CAP];
    int rc;

    if (fragment) memset(fragment, 0, sizeof(*fragment));
    if (!decoder || !fragment)
        return YVEX_ERR_INVALID_ARG;
    rc = decoder_enter(decoder, err);
    if (rc != YVEX_OK)
        return rc;
    if (atomic_load_explicit(&decoder->transaction_active, memory_order_acquire)) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.push",
                       "decoder has an active candidate transaction");
        rc = YVEX_ERR_STATE;
    } else {
        rc = decoder_stage_token(
            decoder, decoder->pending, decoder->pending_count,
            decoder->processed_token_count, decoder->state_identity, token_id,
            next_pending, &next_pending_count, next_state_identity, fragment, err);
    }
    if (rc == YVEX_OK) {
        memcpy(decoder->pending, next_pending, sizeof(decoder->pending));
        decoder->pending_count = next_pending_count;
        decoder->processed_token_count = fragment->processed_token_count;
        yvex_core_text_copy(decoder->state_identity, sizeof(decoder->state_identity),
                            next_state_identity);
    }
    decoder_leave(decoder);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_tokenizer_decoder_transaction_begin(
    yvex_tokenizer_decoder *decoder,
    yvex_tokenizer_decoder_transaction **out, yvex_error *err)
{
    yvex_tokenizer_decoder_transaction *transaction;
    unsigned int expected = 0u;
    int rc;
    if (out) *out = NULL;
    if (!decoder || !out || !atomic_compare_exchange_strong_explicit(
            &decoder->transaction_active, &expected, 1u,
            memory_order_acq_rel, memory_order_acquire)) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.transaction.begin",
                       "decoder cannot admit another candidate transaction");
        return YVEX_ERR_STATE;
    }
    rc = decoder_enter(decoder, err);
    if (rc != YVEX_OK) {
        atomic_store_explicit(&decoder->transaction_active, 0u,
                              memory_order_release);
        return rc;
    }
    transaction = calloc(1u, sizeof(*transaction));
    if (!transaction) {
        atomic_store_explicit(&decoder->transaction_active, 0u,
                              memory_order_release);
        decoder_leave(decoder);
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.decoder.transaction.begin",
                       "decoder transaction allocation failed");
        return YVEX_ERR_NOMEM;
    }
    transaction->decoder = decoder;
    memcpy(transaction->pending, decoder->pending, sizeof(transaction->pending));
    transaction->pending_count = decoder->pending_count;
    transaction->processed_token_count = decoder->processed_token_count;
    yvex_core_text_copy(transaction->base_identity,
                        sizeof(transaction->base_identity), decoder->state_identity);
    yvex_core_text_copy(transaction->state_identity,
                        sizeof(transaction->state_identity), decoder->state_identity);
    decoder_leave(decoder);
    *out = transaction;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_tokenizer_decoder_transaction_push(
    yvex_tokenizer_decoder_transaction *transaction, unsigned int token_id,
    yvex_tokenizer_fragment *fragment, yvex_error *err)
{
    unsigned char next_pending[4];
    unsigned long long next_pending_count = 0ull;
    char next_state_identity[YVEX_SHA256_HEX_CAP];
    int rc;
    if (fragment) memset(fragment, 0, sizeof(*fragment));
    if (!transaction || transaction->prepared || !fragment) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.transaction.push",
                       "mutable decoder transaction and fragment are required");
        return YVEX_ERR_STATE;
    }
    rc = decoder_stage_token(
        transaction->decoder, transaction->pending, transaction->pending_count,
        transaction->processed_token_count, transaction->state_identity, token_id,
        next_pending, &next_pending_count, next_state_identity, fragment, err);
    if (rc == YVEX_OK) {
        memcpy(transaction->pending, next_pending, sizeof(transaction->pending));
        transaction->pending_count = next_pending_count;
        transaction->processed_token_count = fragment->processed_token_count;
        yvex_core_text_copy(transaction->state_identity,
                            sizeof(transaction->state_identity), next_state_identity);
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_tokenizer_decoder_transaction_prepare(
    yvex_tokenizer_decoder_transaction *transaction, yvex_error *err)
{
    yvex_tokenizer_decoder *decoder = transaction ? transaction->decoder : NULL;
    int rc;
    if (!transaction || transaction->prepared || !decoder) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.transaction.prepare",
                       "unprepared decoder transaction is required");
        return YVEX_ERR_STATE;
    }
    rc = decoder_enter(decoder, err);
    if (rc != YVEX_OK) return rc;
    if (!atomic_load_explicit(&decoder->transaction_active, memory_order_acquire) ||
        strcmp(decoder->state_identity, transaction->base_identity) != 0) {
        decoder_leave(decoder);
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.transaction.prepare",
                       "decoder state changed while the candidate was staged");
        return YVEX_ERR_STATE;
    }
    transaction->prepared = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_tokenizer_decoder_transaction_publish(
    yvex_tokenizer_decoder_transaction **transaction)
{
    yvex_tokenizer_decoder_transaction *owner = transaction ? *transaction : NULL;
    yvex_tokenizer_decoder *decoder;
    if (!owner || !owner->prepared) return;
    decoder = owner->decoder;
    memcpy(decoder->pending, owner->pending, sizeof(decoder->pending));
    decoder->pending_count = owner->pending_count;
    decoder->processed_token_count = owner->processed_token_count;
    yvex_core_text_copy(decoder->state_identity, sizeof(decoder->state_identity),
                        owner->state_identity);
    atomic_store_explicit(&decoder->transaction_active, 0u, memory_order_release);
    decoder_leave(decoder);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *transaction = NULL;
}

void yvex_tokenizer_decoder_transaction_abort(
    yvex_tokenizer_decoder_transaction **transaction)
{
    yvex_tokenizer_decoder_transaction *owner = transaction ? *transaction : NULL;
    if (!owner) return;
    atomic_store_explicit(&owner->decoder->transaction_active, 0u,
                          memory_order_release);
    if (owner->prepared) decoder_leave(owner->decoder);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *transaction = NULL;
}

int yvex_tokenizer_decoder_finish(yvex_tokenizer_decoder *decoder,
                                  yvex_tokenizer_fragment *fragment,
                                  yvex_error *err)
{
    int rc;
    if (fragment) memset(fragment, 0, sizeof(*fragment));
    if (!decoder || !fragment)
        return YVEX_ERR_INVALID_ARG;
    rc = decoder_enter(decoder, err);
    if (rc != YVEX_OK)
        return rc;
    if (atomic_load_explicit(&decoder->transaction_active, memory_order_acquire)) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.finish",
                       "decoder has an active candidate transaction");
        rc = YVEX_ERR_STATE;
    } else {
        rc = decode_cancelled(&decoder->options, err);
    }
    if (rc != YVEX_OK) {
        decoder_leave(decoder);
        return rc;
    }
    if (decoder->pending_count && decoder->options.require_complete_utf8) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.decoder.finish", "incomplete UTF-8 tail remains");
        rc = YVEX_ERR_FORMAT;
    } else {
        fragment->schema_version = YVEX_TOKENIZER_DECODER_SCHEMA_V1;
        fragment->processed_token_count = decoder->processed_token_count;
        fragment->pending_byte_count = decoder->pending_count;
        yvex_core_text_copy(fragment->state_before_identity,
                            sizeof(fragment->state_before_identity), decoder->state_identity);
        yvex_core_text_copy(fragment->state_after_identity,
                            sizeof(fragment->state_after_identity), decoder->state_identity);
        fragment->completed = 1;
        rc = fragment_identity(fragment) ? YVEX_OK : YVEX_ERR_STATE;
    }
    decoder_leave(decoder);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

/*
 * Return one idle incremental decoder to its canonical empty-turn state.
 *
 * Busy/closing or identity failure preserves prior state.
 */
int yvex_tokenizer_decoder_reset(yvex_tokenizer_decoder *decoder,
                                 yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP];
    unsigned char pending[4] = {0};
    int rc;

    if (!decoder) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.decoder.reset",
                       "decoder is required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = decoder_enter(decoder, err);
    if (rc != YVEX_OK)
        return rc;
    if (atomic_load_explicit(&decoder->transaction_active, memory_order_acquire)) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.reset",
                       "decoder has an active candidate transaction");
        rc = YVEX_ERR_STATE;
    } else if (!decoder_state_identity(decoder, pending, 0u, 0u, identity)) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.decoder.reset",
                       "empty decoder identity derivation failed");
        rc = YVEX_ERR_STATE;
    } else {
        memset(decoder->pending, 0, sizeof(decoder->pending));
        decoder->pending_count = 0u;
        decoder->processed_token_count = 0u;
        yvex_core_text_copy(decoder->state_identity,
                            sizeof(decoder->state_identity), identity);
    }
    decoder_leave(decoder);
    if (rc == YVEX_OK)
        yvex_error_clear(err);
    return rc;
}

void yvex_tokenizer_fragment_clear(yvex_tokenizer_fragment *fragment)
{
    if (!fragment) return;
    free(fragment->bytes);
    memset(fragment, 0, sizeof(*fragment));
}

/* Transfer unique close ownership only while no push/finish call is active. */
void yvex_tokenizer_decoder_close(yvex_tokenizer_decoder **decoder)
{
    yvex_tokenizer_decoder *owner;
    unsigned int observed, desired;
    if (!decoder || !*decoder)
        return;
    owner = *decoder;
    observed = atomic_load_explicit(&owner->lifecycle, memory_order_acquire);
    while (!(observed & DECODER_CLOSING)) {
        desired = observed | DECODER_CLOSING;
        if (atomic_compare_exchange_weak_explicit(&owner->lifecycle, &observed,
                                                   desired, memory_order_acq_rel,
                                                   memory_order_acquire))
            break;
    }
    if (owner->drain_mutex_ready && pthread_mutex_lock(&owner->drain_mutex) == 0) {
        while (atomic_load_explicit(&owner->lifecycle, memory_order_acquire) & DECODER_ACTIVE)
            if (!owner->drain_condition_ready ||
                pthread_cond_wait(&owner->drain_condition, &owner->drain_mutex) != 0) {
                (void)pthread_mutex_unlock(&owner->drain_mutex);
                return;
        }
        (void)pthread_mutex_unlock(&owner->drain_mutex);
    }
    if (atomic_load_explicit(&owner->transaction_active, memory_order_acquire)) return;
    if (owner->drain_condition_ready)
        (void)pthread_cond_destroy(&owner->drain_condition);
    if (owner->drain_mutex_ready)
        (void)pthread_mutex_destroy(&owner->drain_mutex);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *decoder = NULL;
}
