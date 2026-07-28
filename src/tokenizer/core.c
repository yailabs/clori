/* Owner: tokenizer.core.
 * Owns: tokenizer metadata, vocabulary lifecycle, and retained fixture transforms.
 * Does not own: exact BPE execution, incremental decoding, prompt policy, generation loops, or CLI presentation.
 * Invariants: vocabulary views remain bound to admitted metadata and token buffers own storage.
 * Boundary: tokenizer ABI over GGUF facts; text generation remains downstream.
 * Purpose: construct tokenizer state and perform bounded text, token, and prompt transforms.
 * Inputs: admitted GGUF metadata, immutable text or token IDs, and typed prompt messages.
 * Effects: allocates tokenizer, vocabulary, token, and rendered-prompt storage explicitly.
 * Failure: rejects malformed metadata, bounds, allocation, and unsupported tokenizer behavior. */

#include "src/tokenizer/private.h"
#include <yvex/gguf.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static int tokenizer_load_vocab(yvex_tokenizer *tokenizer,
                                const yvex_gguf *gguf,
                                yvex_error *err);
static void tokenizer_free_vocab(yvex_tokenizer *tokenizer);
static int tokenizer_load_specials(yvex_tokenizer *tokenizer,
                                   const yvex_gguf *gguf,
                                   yvex_error *err);
static void tokenizer_free_metadata(yvex_tokenizer *tokenizer);

/* Purpose: copy one admitted GGUF string into owned terminated storage.
 * Inputs: nullable typed value and optional length result.
 * Effects: allocates a byte-exact copy and publishes its length.
 * Failure: wrong type, excess length, or allocation failure returns NULL.
 * Boundary: metadata copying; value admission remains GGUF-owned. */
static char *copy_string_value(const yvex_gguf_value *value, unsigned long long *out_len)
{
    const char *data;
    unsigned long long len;
    char *copy;

    if (out_len) {
        *out_len = 0;
    }
    if (!value || yvex_gguf_value_as_string(value, &data, &len) != YVEX_OK) {
        return NULL;
    }
    if (len > (unsigned long long)(SIZE_MAX - 1)) {
        return NULL;
    }
    copy = (char *)malloc((size_t)len + 1u);
    if (!copy) {
        return NULL;
    }
    if (len > 0) {
        memcpy(copy, data, (size_t)len);
    }
    copy[len] = '\0';
    if (out_len) {
        *out_len = len;
    }
    return copy;
}

/* Purpose: map tokenizer model spelling to the closed tokenizer-kind enumeration. */
static yvex_tokenizer_kind kind_from_model(const char *model)
{
    if (!model) {
        return YVEX_TOKENIZER_KIND_UNKNOWN;
    }
    if (strcmp(model, "llama") == 0) return YVEX_TOKENIZER_KIND_GGML_LLAMA;
    if (strcmp(model, "gpt2") == 0) return YVEX_TOKENIZER_KIND_GGML_GPT2;
    if (strcmp(model, "replit") == 0) return YVEX_TOKENIZER_KIND_GGML_REPLIT;
    if (strcmp(model, "rwkv") == 0) return YVEX_TOKENIZER_KIND_GGML_RWKV;
    if (strcmp(model, "huggingface-json") == 0) return YVEX_TOKENIZER_KIND_HUGGINGFACE_JSON;
    if (strcmp(model, "yvex-fixture-simple") == 0) return YVEX_TOKENIZER_KIND_FIXTURE_SIMPLE;
    return YVEX_TOKENIZER_KIND_UNKNOWN;
}

/* Purpose: project truthful implementation support for one tokenizer kind. */
static yvex_tokenizer_support support_for_kind(yvex_tokenizer_kind kind)
{
    switch (kind) {
    case YVEX_TOKENIZER_KIND_FIXTURE_SIMPLE:
        return YVEX_TOKENIZER_SUPPORT_FIXTURE_ENCODE_DECODE;
    case YVEX_TOKENIZER_KIND_GGML_LLAMA:
    case YVEX_TOKENIZER_KIND_GGML_GPT2:
    case YVEX_TOKENIZER_KIND_GGML_REPLIT:
    case YVEX_TOKENIZER_KIND_GGML_RWKV:
        return YVEX_TOKENIZER_SUPPORT_VOCAB_ONLY;
    case YVEX_TOKENIZER_KIND_HUGGINGFACE_JSON:
        return YVEX_TOKENIZER_SUPPORT_METADATA_ONLY;
    case YVEX_TOKENIZER_KIND_UNKNOWN:
        return YVEX_TOKENIZER_SUPPORT_UNSUPPORTED;
    }
    return YVEX_TOKENIZER_SUPPORT_UNSUPPORTED;
}

/* Purpose: construct tokenizer state from admitted GGUF metadata.
 * Inputs: result slot, immutable GGUF, optional model descriptor, and error output.
 * Effects: allocates tokenizer state and copies vocabulary, template, and special-token facts.
 * Failure: validation, allocation, or metadata errors publish no tokenizer.
 * Boundary: tokenizer admission; it does not execute generation. */
int yvex_tokenizer_from_gguf(yvex_tokenizer **out,
                             const yvex_gguf *gguf,
                             const yvex_model_descriptor *model,
                             yvex_error *err)
{
    yvex_tokenizer *tokenizer;
    const yvex_gguf_value *value;
    int rc;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_tokenizer_from_gguf", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;

    if (!gguf) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_tokenizer_from_gguf", "gguf is required");
        return YVEX_ERR_INVALID_ARG;
    }

    tokenizer = (yvex_tokenizer *)calloc(1, sizeof(*tokenizer));
    if (!tokenizer) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tokenizer_from_gguf", "failed to allocate tokenizer");
        return YVEX_ERR_NOMEM;
    }

    value = yvex_gguf_metadata_find(gguf, "tokenizer.ggml.model");
    tokenizer->model_name = copy_string_value(value, NULL);
    tokenizer->kind = kind_from_model(tokenizer->model_name);
    tokenizer->support = support_for_kind(tokenizer->kind);

    value = yvex_gguf_metadata_find(gguf, "tokenizer.chat_template");
    tokenizer->chat_template = copy_string_value(value, &tokenizer->chat_template_len);

    value = yvex_gguf_metadata_find(gguf, "tokenizer.huggingface.json");
    tokenizer->has_huggingface_json = value && yvex_gguf_value_type_of(value) == YVEX_GGUF_VALUE_STRING;

    rc = tokenizer_load_vocab(tokenizer, gguf, err);
    if (rc != YVEX_OK) {
        yvex_tokenizer_close(tokenizer);
        return rc;
    }

    rc = tokenizer_load_specials(tokenizer, gguf, err);
    if (rc != YVEX_OK) {
        yvex_tokenizer_close(tokenizer);
        return rc;
    }

    rc = yvex_tokenizer_execution_seal(tokenizer, gguf, model, err);
    if (rc != YVEX_OK && rc != YVEX_ERR_UNSUPPORTED) {
        yvex_tokenizer_close(tokenizer);
        return rc;
    }
    if (rc == YVEX_ERR_UNSUPPORTED) {
        yvex_error_clear(err);
    }

    *out = tokenizer;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: release all storage owned by one tokenizer.
 * Inputs: nullable owned tokenizer.
 * Effects: frees vocabulary, metadata, and root allocation.
 * Failure: none; NULL is accepted.
 * Boundary: terminal tokenizer lifecycle operation. */
void yvex_tokenizer_close(yvex_tokenizer *tokenizer)
{
    if (!tokenizer) {
        return;
    }
    yvex_tokenizer_execution_release(tokenizer);
    tokenizer_free_vocab(tokenizer);
    tokenizer_free_metadata(tokenizer);
    free(tokenizer);
}

/* Purpose: read the admitted tokenizer kind.
 * Inputs: nullable immutable tokenizer.
 * Effects: none.
 * Failure: NULL projects the unknown kind.
 * Boundary: immutable metadata view. */
yvex_tokenizer_kind yvex_tokenizer_kind_of(const yvex_tokenizer *tokenizer)
{
    return tokenizer ? tokenizer->kind : YVEX_TOKENIZER_KIND_UNKNOWN;
}

/* Purpose: read the bounded tokenizer support classification.
 * Inputs: nullable immutable tokenizer.
 * Effects: none.
 * Failure: NULL projects unsupported.
 * Boundary: support fact, not generation readiness. */
yvex_tokenizer_support yvex_tokenizer_support_of(const yvex_tokenizer *tokenizer)
{
    return tokenizer ? tokenizer->support : YVEX_TOKENIZER_SUPPORT_UNSUPPORTED;
}

/* Purpose: map tokenizer kind to its stable diagnostic spelling.
 * Inputs: typed tokenizer kind.
 * Effects: none; returned storage is static.
 * Failure: unknown values map to "unknown".
 * Boundary: label projection only. */
const char *yvex_tokenizer_kind_name(yvex_tokenizer_kind kind)
{
    switch (kind) {
    case YVEX_TOKENIZER_KIND_UNKNOWN: return "unknown";
    case YVEX_TOKENIZER_KIND_GGML_LLAMA: return "llama";
    case YVEX_TOKENIZER_KIND_GGML_GPT2: return "gpt2";
    case YVEX_TOKENIZER_KIND_GGML_REPLIT: return "replit";
    case YVEX_TOKENIZER_KIND_GGML_RWKV: return "rwkv";
    case YVEX_TOKENIZER_KIND_HUGGINGFACE_JSON: return "huggingface-json";
    case YVEX_TOKENIZER_KIND_FIXTURE_SIMPLE: return "yvex-fixture-simple";
    }
    return "unknown";
}

/* Purpose: map tokenizer support state to stable public wording.
 * Inputs: typed support state.
 * Effects: none; returned storage is static.
 * Failure: unrecognized values map to "unsupported".
 * Boundary: label projection, not capability admission. */
const char *yvex_tokenizer_support_name(yvex_tokenizer_support support)
{
    switch (support) {
    case YVEX_TOKENIZER_SUPPORT_METADATA_ONLY: return "metadata-only";
    case YVEX_TOKENIZER_SUPPORT_VOCAB_ONLY: return "vocab-only";
    case YVEX_TOKENIZER_SUPPORT_FIXTURE_ENCODE_DECODE: return "fixture-encode-decode";
    case YVEX_TOKENIZER_SUPPORT_ARTIFACT_BPE: return "artifact-bpe";
    case YVEX_TOKENIZER_SUPPORT_UNSUPPORTED: return "unsupported";
    }
    return "unsupported";
}

/* Purpose: expose the tokenizer's immutable chat-template bytes.
 * Inputs: tokenizer and required data/length outputs.
 * Effects: writes a borrowed view valid for tokenizer lifetime.
 * Failure: invalid inputs or missing template return typed refusal.
 * Boundary: metadata view; prompt rendering is separate. */
int yvex_tokenizer_chat_template(const yvex_tokenizer *tokenizer,
                                 const char **data,
                                 unsigned long long *len)
{
    if (!tokenizer || !data || !len) {
        return YVEX_ERR_INVALID_ARG;
    }
    if (!tokenizer->chat_template) {
        *data = NULL;
        *len = 0;
        return YVEX_ERR_UNSUPPORTED;
    }
    *data = tokenizer->chat_template;
    *len = tokenizer->chat_template_len;
    return YVEX_OK;
}

/* Purpose: release dynamic metadata strings owned by a tokenizer.
 * Inputs: nullable mutable tokenizer.
 * Effects: frees and clears model and chat-template storage.
 * Failure: none; cleared state is accepted.
 * Boundary: tokenizer-local cleanup. */
static void tokenizer_free_metadata(yvex_tokenizer *tokenizer)
{
    if (!tokenizer) {
        return;
    }
    free(tokenizer->model_name);
    tokenizer->model_name = NULL;
    free(tokenizer->chat_template);
    tokenizer->chat_template = NULL;
    tokenizer->chat_template_len = 0;
}

/* Purpose: release an owned token sequence.
 * Inputs: nullable mutable token container.
 * Effects: frees token IDs and restores empty state.
 * Failure: none; empty state is accepted.
 * Boundary: token-buffer lifecycle. */
void yvex_tokens_free(yvex_tokens *tokens)
{
    if (!tokens) {
        return;
    }
    free(tokens->ids);
    yvex_tokens_clear(tokens);
}

/* Purpose: decide whether one admitted token type contributes text bytes. */
static int token_emits_text(yvex_token_type type)
{
    return type == YVEX_TOKEN_TYPE_NORMAL ||
           type == YVEX_TOKEN_TYPE_UNK ||
           type == YVEX_TOKEN_TYPE_USER_DEFINED ||
           type == YVEX_TOKEN_TYPE_BYTE;
}

/* Purpose: decode admitted fixture token IDs into caller-bounded text.
 * Inputs: tokenizer, ID sequence, output storage/capacity, and error output.
 * Effects: writes a terminated concatenation of text-emitting vocabulary entries.
 * Failure: unsupported kind, invalid ID, or capacity overflow leaves typed refusal.
 * Boundary: fixture decode proof; it is not model generation. */
int yvex_detokenize_ids(const yvex_tokenizer *tokenizer,
                        const unsigned int *ids,
                        unsigned long long len,
                        char *out,
                        unsigned long long cap,
                        yvex_error *err)
{
    yvex_tokenizer_decode_result decoded;
    yvex_tokenizer_decode_options options;
    unsigned long long i;
    unsigned long long used = 0;
    int rc;

    if (!tokenizer || (!ids && len > 0) || !out || cap == 0) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "yvex_detokenize_ids",
                       "tokenizer, ids and output buffer are required");
        return YVEX_ERR_INVALID_ARG;
    }
    out[0] = '\0';

    if (tokenizer->support == YVEX_TOKENIZER_SUPPORT_ARTIFACT_BPE) {
        memset(&decoded, 0, sizeof(decoded));
        memset(&options, 0, sizeof(options));
        rc = yvex_tokenizer_decode(tokenizer, ids, len, &options, &decoded, err);
        if (rc != YVEX_OK) {
            return rc;
        }
        if (decoded.byte_count >= cap) {
            yvex_tokenizer_decode_result_clear(&decoded);
            yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_detokenize_ids",
                           "output buffer too small");
            return YVEX_ERR_BOUNDS;
        }
        if (decoded.byte_count) {
            memcpy(out, decoded.bytes, (size_t)decoded.byte_count);
        }
        out[decoded.byte_count] = '\0';
        yvex_tokenizer_decode_result_clear(&decoded);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (tokenizer->support != YVEX_TOKENIZER_SUPPORT_FIXTURE_ENCODE_DECODE) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "yvex_detokenize_ids",
                        "tokenizer kind %s is not executable in tokenizer layer",
                        yvex_tokenizer_kind_name(tokenizer->kind));
        return YVEX_ERR_UNSUPPORTED;
    }

    for (i = 0; i < len; ++i) {
        const yvex_token_info *token = yvex_tokenizer_token_at(tokenizer, ids[i]);
        if (!token) {
            yvex_error_setf(err, YVEX_ERR_BOUNDS, "yvex_detokenize_ids",
                            "token id %u is outside vocabulary", ids[i]);
            return YVEX_ERR_BOUNDS;
        }
        if (!token_emits_text(token->type)) {
            continue;
        }
        if (token->text_len > cap - used - 1u) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_detokenize_ids", "output buffer too small");
            return YVEX_ERR_BOUNDS;
        }
        if (token->text_len > 0) {
            memcpy(out + used, token->text, (size_t)token->text_len);
        }
        used += token->text_len;
        out[used] = '\0';
    }

    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: restore a token container to empty borrowed-free state.
 * Inputs: nullable mutable token container.
 * Effects: clears pointer, count, and capacity without freeing storage.
 * Failure: none; NULL is accepted.
 * Boundary: initialization/reset helper; callers own prior allocation cleanup. */
void yvex_tokens_clear(yvex_tokens *tokens)
{
    if (!tokens) {
        return;
    }
    tokens->ids = NULL;
    tokens->len = 0;
    tokens->cap = 0;
}

/* Purpose: append one ID to an owned growable token sequence.
 * Inputs: mutable token buffer, ID, and error output.
 * Effects: may reallocate storage and increments count exactly once.
 * Failure: capacity overflow or allocation failure preserves a valid buffer.
 * Boundary: tokenizer-local buffer growth. */
static int tokens_append(yvex_tokens *tokens, unsigned int id, yvex_error *err)
{
    unsigned int *next;
    unsigned long long next_cap;

    if (tokens->len == tokens->cap) {
        next_cap = tokens->cap == 0 ? 8 : tokens->cap * 2u;
        if (next_cap < tokens->cap || next_cap > (unsigned long long)(SIZE_MAX / sizeof(*tokens->ids))) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tokenize_text", "token buffer too large");
            return YVEX_ERR_NOMEM;
        }
        next = (unsigned int *)realloc(tokens->ids, (size_t)next_cap * sizeof(*tokens->ids));
        if (!next) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tokenize_text", "failed to grow token buffer");
            return YVEX_ERR_NOMEM;
        }
        tokens->ids = next;
        tokens->cap = next_cap;
    }

    tokens->ids[tokens->len++] = id;
    return YVEX_OK;
}

/* Purpose: select the longest deterministic vocabulary prefix at one text position.
 * Inputs: immutable tokenizer, remaining text, and optional matched-byte output.
 * Effects: writes match length and returns a borrowed vocabulary entry.
 * Failure: no match returns NULL with zero length.
 * Boundary: fixture tokenizer matching policy. */
static const yvex_token_info *find_longest_match(const yvex_tokenizer *tokenizer,
                                                 const char *text,
                                                 unsigned long long remaining)
{
    const yvex_token_info *best = NULL;
    unsigned long long i;

    for (i = 0; i < tokenizer->vocab_size; ++i) {
        const yvex_token_info *token = &tokenizer->tokens[i];
        if (token->type == YVEX_TOKEN_TYPE_CONTROL || token->type == YVEX_TOKEN_TYPE_UNUSED) {
            continue;
        }
        if (token->text_len == 0 || token->text_len > remaining) {
            continue;
        }
        if (memcmp(text, token->text, (size_t)token->text_len) == 0 &&
            (!best || token->text_len > best->text_len)) {
            best = token;
        }
    }

    return best;
}

/* Purpose: encode text through the admitted fixture vocabulary.
 * Inputs: tokenizer, immutable text, result token container, and error output.
 * Effects: allocates and publishes an ordered ID sequence on exact completion.
 * Failure: unsupported kind, unmatched input, overflow, or allocation aborts output.
 * Boundary: bounded tokenizer execution, not model inference. */
int yvex_tokenize_text(const yvex_tokenizer *tokenizer,
                       const char *text,
                       yvex_tokens *out,
                       yvex_error *err)
{
    yvex_tokenizer_encode_result encoded;
    yvex_tokenizer_encode_options options;
    unsigned long long len;
    unsigned long long offset = 0;
    unsigned int unk_id;
    int has_unk;
    int rc;

    if (!tokenizer || !text || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_tokenize_text", "tokenizer, text and out are required");
        return YVEX_ERR_INVALID_ARG;
    }

    yvex_tokens_clear(out);

    if (tokenizer->support == YVEX_TOKENIZER_SUPPORT_ARTIFACT_BPE) {
        memset(&encoded, 0, sizeof(encoded));
        memset(&options, 0, sizeof(options));
        options.allow_special_tokens = 1;
        options.maximum_tokens = ULLONG_MAX;
        rc = yvex_tokenizer_encode(tokenizer, (const unsigned char *)text,
                                   (unsigned long long)strlen(text), &options,
                                   &encoded, err);
        if (rc != YVEX_OK) {
            return rc;
        }
        *out = encoded.tokens;
        yvex_tokens_clear(&encoded.tokens);
        return YVEX_OK;
    }

    if (tokenizer->support != YVEX_TOKENIZER_SUPPORT_FIXTURE_ENCODE_DECODE) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "yvex_tokenize_text",
                        "tokenizer kind %s is not executable in tokenizer layer",
                        yvex_tokenizer_kind_name(tokenizer->kind));
        return YVEX_ERR_UNSUPPORTED;
    }

    len = (unsigned long long)strlen(text);
    has_unk = yvex_tokenizer_unk_id(tokenizer, &unk_id) == YVEX_OK;

    while (offset < len) {
        const yvex_token_info *match = find_longest_match(tokenizer, text + offset, len - offset);
        if (match) {
            rc = tokens_append(out, match->id, err);
            if (rc != YVEX_OK) {
                yvex_tokens_free(out);
                return rc;
            }
            offset += match->text_len;
        } else if (has_unk) {
            rc = tokens_append(out, unk_id, err);
            if (rc != YVEX_OK) {
                yvex_tokens_free(out);
                return rc;
            }
            offset += 1;
        } else {
            yvex_tokens_free(out);
            yvex_error_set(err,
                           YVEX_ERR_UNSUPPORTED,
                           "yvex_tokenize_text",
                           "no token matched and no unknown token exists");
            return YVEX_ERR_UNSUPPORTED;
        }
    }

    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: load one optional special-token ID with vocabulary bounds validation.
 * Inputs: GGUF, key, vocabulary size, destination slot, and error output.
 * Effects: marks and writes the special ID only when metadata is present.
 * Failure: malformed type or out-of-range ID returns typed refusal.
 * Boundary: tokenizer special-token admission. */
static int load_special(const yvex_gguf *gguf,
                        const char *key,
                        tokenizer_special_id *slot,
                        unsigned long long vocab_size,
                        yvex_error *err)
{
    const yvex_gguf_value *value = yvex_gguf_metadata_find(gguf, key);
    unsigned long long id;
    int rc;

    slot->present = 0;
    slot->id = 0;
    if (!value) {
        return YVEX_OK;
    }

    rc = yvex_gguf_value_as_u64(value, &id);
    if (rc != YVEX_OK) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "yvex_tokenizer_load_specials",
                        "%s must be an unsigned integer", key);
        return YVEX_ERR_FORMAT;
    }
    if (id >= vocab_size) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, "yvex_tokenizer_load_specials",
                        "%s value %llu is outside vocab size %llu", key, id, vocab_size);
        return YVEX_ERR_BOUNDS;
    }

    slot->present = 1;
    slot->id = (unsigned int)id;
    return YVEX_OK;
}

/* Purpose: admit the closed set of tokenizer special IDs.
 * Inputs: mutable tokenizer, immutable GGUF, and error output.
 * Effects: populates BOS, EOS, UNK, PAD, and SEP slots in deterministic order.
 * Failure: the first malformed or out-of-range value aborts construction.
 * Boundary: metadata population during tokenizer construction. */
static int tokenizer_load_specials(yvex_tokenizer *tokenizer,
                                   const yvex_gguf *gguf,
                                   yvex_error *err)
{
    int rc;

    rc = load_special(gguf, "tokenizer.ggml.bos_token_id", &tokenizer->bos, tokenizer->vocab_size, err);
    if (rc != YVEX_OK) return rc;
    rc = load_special(gguf, "tokenizer.ggml.eos_token_id", &tokenizer->eos, tokenizer->vocab_size, err);
    if (rc != YVEX_OK) return rc;
    rc = load_special(gguf, "tokenizer.ggml.unknown_token_id", &tokenizer->unk, tokenizer->vocab_size, err);
    if (rc != YVEX_OK) return rc;
    rc = load_special(gguf, "tokenizer.ggml.padding_token_id", &tokenizer->pad, tokenizer->vocab_size, err);
    if (rc != YVEX_OK) return rc;
    return load_special(gguf, "tokenizer.ggml.separator_token_id", &tokenizer->sep, tokenizer->vocab_size, err);
}

/* Purpose: project one optional special-ID slot through the public sentinel convention. */
static int special_id_get(const yvex_tokenizer *tokenizer,
                          const tokenizer_special_id *slot,
                          unsigned int *out)
{
    if (!tokenizer || !slot || !out) {
        return YVEX_ERR_INVALID_ARG;
    }
    if (!slot->present) {
        return YVEX_ERR_UNSUPPORTED;
    }
    *out = slot->id;
    return YVEX_OK;
}

/* Purpose: read the admitted beginning-of-sequence token ID.
 * Inputs: tokenizer and required result storage.
 * Effects: writes the ID only when present.
 * Failure: invalid or absent state returns typed refusal.
 * Boundary: special-token metadata view. */
int yvex_tokenizer_bos_id(const yvex_tokenizer *tokenizer, unsigned int *out)
{
    return special_id_get(tokenizer, tokenizer ? &tokenizer->bos : 0, out);
}

/* Purpose: read the admitted end-of-sequence token ID.
 * Inputs: tokenizer and required result storage.
 * Effects: writes the ID only when present.
 * Failure: invalid or absent state returns typed refusal.
 * Boundary: immutable EOS metadata projection. */
int yvex_tokenizer_eos_id(const yvex_tokenizer *tokenizer, unsigned int *out)
{
    return special_id_get(tokenizer, tokenizer ? &tokenizer->eos : 0, out);
}

/* Purpose: read the admitted unknown-token ID.
 * Inputs: tokenizer and required output.
 * Effects: publishes one scalar ID when available.
 * Failure: missing state is reported without output mutation.
 * Boundary: immutable UNK metadata projection. */
int yvex_tokenizer_unk_id(const yvex_tokenizer *tokenizer, unsigned int *out)
{
    return special_id_get(tokenizer, tokenizer ? &tokenizer->unk : 0, out);
}

/* Purpose: read the admitted padding-token ID.
 * Inputs: tokenizer and required output.
 * Effects: publishes one scalar ID when present.
 * Failure: invalid arguments or absent metadata return refusal.
 * Boundary: immutable PAD metadata projection. */
int yvex_tokenizer_pad_id(const yvex_tokenizer *tokenizer, unsigned int *out)
{
    return special_id_get(tokenizer, tokenizer ? &tokenizer->pad : 0, out);
}

/* Purpose: validate and expose one GGUF metadata array's element type and count.
 * Inputs: value, expected type, output, error context, and metadata key.
 * Effects: writes immutable array facts on success.
 * Failure: absent or mismatched arrays return typed format refusal.
 * Boundary: tokenizer validation over the GGUF value ABI. */
static int read_array_info(const yvex_gguf_value *value,
                           yvex_gguf_value_type expected,
                           yvex_gguf_array_info *out,
                           yvex_error *err,
                           const char *where,
                           const char *name)
{
    if (!value || yvex_gguf_value_array_info(value, out) != YVEX_OK) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, where, "%s must be an array", name);
        return YVEX_ERR_FORMAT;
    }
    if (out->element_type != expected) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, where, "%s has wrong element type", name);
        return YVEX_ERR_FORMAT;
    }
    return YVEX_OK;
}

/* Purpose: allocate a terminated copy of one bounded vocabulary token.
 * Inputs: immutable bytes and exact length.
 * Effects: allocates and returns owned string storage.
 * Failure: unrepresentable length or allocation failure returns NULL.
 * Boundary: vocabulary-entry ownership helper. */
static char *copy_text(const char *data, unsigned long long len)
{
    char *copy;

    if (len > (unsigned long long)(SIZE_MAX - 1)) {
        return NULL;
    }
    copy = (char *)malloc((size_t)len + 1u);
    if (!copy) {
        return NULL;
    }
    if (len > 0) {
        memcpy(copy, data, (size_t)len);
    }
    copy[len] = '\0';
    return copy;
}

/* Purpose: load exact vocabulary entries and optional score/type arrays from GGUF.
 * Inputs: mutable tokenizer, immutable GGUF, and error output.
 * Effects: allocates ordered token facts bound to admitted vocabulary cardinality.
 * Failure: type, count, overflow, or allocation errors leave cleanup to construction unwind.
 * Boundary: vocabulary admission; tokenizer support policy stays explicit. */
static int tokenizer_load_vocab(yvex_tokenizer *tokenizer,
                                const yvex_gguf *gguf,
                                yvex_error *err)
{
    const yvex_gguf_value *tokens_value;
    const yvex_gguf_value *scores_value;
    const yvex_gguf_value *types_value;
    yvex_gguf_array_info tokens_info;
    yvex_gguf_array_info scores_info;
    yvex_gguf_array_info types_info;
    unsigned long long i;
    int has_scores = 0;
    int has_types = 0;
    int rc;

    if (!tokenizer || !gguf) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_tokenizer_load_vocab", "tokenizer and gguf are required");
        return YVEX_ERR_INVALID_ARG;
    }

    tokens_value = yvex_gguf_metadata_find(gguf, "tokenizer.ggml.tokens");
    if (!tokens_value) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "yvex_tokenizer_load_vocab", "tokenizer.ggml.tokens is missing");
        return YVEX_ERR_UNSUPPORTED;
    }

    rc = read_array_info(tokens_value, YVEX_GGUF_VALUE_STRING, &tokens_info, err,
                         "yvex_tokenizer_load_vocab", "tokenizer.ggml.tokens");
    if (rc != YVEX_OK) {
        return rc;
    }
    if (tokens_info.count == 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_tokenizer_load_vocab", "tokenizer vocabulary is empty");
        return YVEX_ERR_FORMAT;
    }
    if (tokens_info.count > (unsigned long long)(SIZE_MAX / sizeof(*tokenizer->tokens))) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tokenizer_load_vocab", "vocabulary too large");
        return YVEX_ERR_NOMEM;
    }

    scores_value = yvex_gguf_metadata_find(gguf, "tokenizer.ggml.scores");
    if (scores_value) {
        rc = read_array_info(scores_value, YVEX_GGUF_VALUE_FLOAT32, &scores_info, err,
                             "yvex_tokenizer_load_vocab", "tokenizer.ggml.scores");
        if (rc != YVEX_OK) {
            return rc;
        }
        if (scores_info.count != tokens_info.count) {
            yvex_error_set(err,
                           YVEX_ERR_FORMAT,
                           "yvex_tokenizer_load_vocab",
                           "tokenizer score count does not match tokens");
            return YVEX_ERR_FORMAT;
        }
        has_scores = 1;
    }

    types_value = yvex_gguf_metadata_find(gguf, "tokenizer.ggml.token_type");
    if (types_value) {
        rc = read_array_info(types_value, YVEX_GGUF_VALUE_INT32, &types_info, err,
                             "yvex_tokenizer_load_vocab", "tokenizer.ggml.token_type");
        if (rc != YVEX_OK) {
            return rc;
        }
        if (types_info.count != tokens_info.count) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_tokenizer_load_vocab", "token type count does not match tokens");
            return YVEX_ERR_FORMAT;
        }
        has_types = 1;
    }

    tokenizer->tokens = (yvex_token_info *)calloc((size_t)tokens_info.count, sizeof(*tokenizer->tokens));
    if (!tokenizer->tokens) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tokenizer_load_vocab", "failed to allocate vocabulary");
        return YVEX_ERR_NOMEM;
    }
    tokenizer->vocab_size = tokens_info.count;

    for (i = 0; i < tokens_info.count; ++i) {
        const yvex_gguf_value *item = yvex_gguf_value_array_at(tokens_value, i);
        const char *text;
        unsigned long long len;
        yvex_token_info *token = &tokenizer->tokens[i];

        if (!item || yvex_gguf_value_as_string(item, &text, &len) != YVEX_OK) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_tokenizer_load_vocab", "token entry is not a string");
            return YVEX_ERR_FORMAT;
        }
        token->text = copy_text(text, len);
        if (!token->text) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tokenizer_load_vocab", "failed to copy token text");
            return YVEX_ERR_NOMEM;
        }
        token->id = (unsigned int)i;
        token->text_len = len;
        token->score = 0.0f;
        token->type = YVEX_TOKEN_TYPE_NORMAL;

        if (has_scores) {
            double score;
            item = yvex_gguf_value_array_at(scores_value, i);
            if (!item || yvex_gguf_value_as_f64(item, &score) != YVEX_OK) {
                yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_tokenizer_load_vocab", "score entry is invalid");
                return YVEX_ERR_FORMAT;
            }
            token->score = (float)score;
        }

        if (has_types) {
            long long type;
            item = yvex_gguf_value_array_at(types_value, i);
            if (!item || yvex_gguf_value_as_i64(item, &type) != YVEX_OK ||
                type < YVEX_TOKEN_TYPE_NORMAL || type > YVEX_TOKEN_TYPE_BYTE) {
                yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_tokenizer_load_vocab", "token type entry is invalid");
                return YVEX_ERR_FORMAT;
            }
            token->type = (yvex_token_type)type;
        }
    }

    return YVEX_OK;
}

/* Purpose: release every vocabulary entry and its owning array.
 * Inputs: nullable mutable tokenizer.
 * Effects: frees token strings and clears vocabulary state.
 * Failure: none; partially constructed vocabulary is accepted.
 * Boundary: construction unwind and tokenizer teardown. */
static void tokenizer_free_vocab(yvex_tokenizer *tokenizer)
{
    unsigned long long i;

    if (!tokenizer || !tokenizer->tokens) {
        return;
    }
    for (i = 0; i < tokenizer->vocab_size; ++i) {
        free((char *)tokenizer->tokens[i].text);
        tokenizer->tokens[i].text = NULL;
    }
    free(tokenizer->tokens);
    tokenizer->tokens = NULL;
    tokenizer->vocab_size = 0;
}

/* Purpose: read admitted vocabulary cardinality.
 * Inputs: nullable immutable tokenizer.
 * Effects: none.
 * Failure: NULL projects zero.
 * Boundary: scalar metadata view. */
unsigned long long yvex_tokenizer_vocab_size(const yvex_tokenizer *tokenizer)
{
    return tokenizer ? tokenizer->vocab_size : 0;
}

/* Purpose: retrieve one immutable vocabulary entry by exact ID.
 * Inputs: tokenizer and zero-based token ID.
 * Effects: none; returned view follows tokenizer lifetime.
 * Failure: invalid or out-of-range ID returns NULL.
 * Boundary: borrowed vocabulary lookup. */
const yvex_token_info *yvex_tokenizer_token_at(const yvex_tokenizer *tokenizer,
                                               unsigned long long id)
{
    if (!tokenizer || id >= tokenizer->vocab_size) {
        return NULL;
    }
    return &tokenizer->tokens[id];
}
