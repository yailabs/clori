/*
 * Execute one compiled artifact tokenizer policy without source sidecars or external runtimes.
 *
 * Accepted plans match exact tokenizer/config identities and reuse immutable bounded lookup
 * indexes. Produces numeric token sequences and prompt bytes; it never mutates model KV or chooses
 * tokens.
 */

#include "src/tokenizer/private.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>

typedef struct {
    unsigned char *data;
    unsigned long long count, capacity;
} byte_builder;

typedef struct {
    const unsigned char *bytes;
    unsigned long long count, offset;
} tokenizer_span;

typedef struct {
    unsigned int token_id;
    yvex_token_append_state state;
} token_sequence_row;

struct yvex_token_sequence {
    token_sequence_row *rows;
    unsigned long long count, capacity, generation;
    int transaction_active;
};

struct yvex_token_sequence_transaction {
    yvex_token_sequence *sequence;
    token_sequence_row *rows;
    unsigned long long base_count, base_generation;
    unsigned long long count, capacity, generation;
    int prepared;
};

static uint64_t lookup_hash(const void *data, size_t count)
{
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t value = UINT64_C(1469598103934665603);
    size_t index;

    for (index = 0u; index < count; ++index) {
        value ^= bytes[index];
        value *= UINT64_C(1099511628211);
    }
    return value ? value : 1u;
}

static int bytes_identity(const char *domain, const void *data, size_t count,
                          char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_sha256_update_u64_be(&hash, (unsigned long long)count) ||
        !yvex_sha256_update(&hash, data, count) || !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int gguf_string(const yvex_gguf *gguf, const char *key, const char **data,
                       unsigned long long *count)
{
    const yvex_gguf_value *value = yvex_gguf_metadata_find(gguf, key);
    return value && yvex_gguf_value_as_string(value, data, count) == YVEX_OK;
}

static int builder_reserve(byte_builder *builder, unsigned long long add, yvex_error *err)
{
    unsigned long long need, capacity;
    unsigned char *grown;

    if (!builder || builder->count > ULLONG_MAX - add - 1u) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.builder", "byte extent overflow");
        return YVEX_ERR_BOUNDS;
    }
    need = builder->count + add + 1u;
    if (need <= builder->capacity)
        return YVEX_OK;
    capacity = builder->capacity ? builder->capacity : 128u;
    while (capacity < need) {
        if (capacity > ULLONG_MAX / 2u || capacity * 2u > (unsigned long long)SIZE_MAX) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.builder", "byte buffer exceeds address space");
            return YVEX_ERR_NOMEM;
        }
        capacity *= 2u;
    }
    grown = (unsigned char *)realloc(builder->data, (size_t)capacity);
    if (!grown) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.builder", "byte buffer allocation failed");
        return YVEX_ERR_NOMEM;
    }
    builder->data = grown;
    builder->capacity = capacity;
    return YVEX_OK;
}

static int builder_append(byte_builder *builder, const void *data,
                          unsigned long long count, yvex_error *err)
{
    int rc = builder_reserve(builder, count, err);
    if (rc != YVEX_OK)
        return rc;
    if (count)
        memcpy(builder->data + builder->count, data, (size_t)count);
    builder->count += count;
    builder->data[builder->count] = 0u;
    return YVEX_OK;
}

static int vocab_lookup(const yvex_tokenizer *tokenizer, const void *data, size_t count,
                        unsigned int *token_id)
{
    uint64_t hash = lookup_hash(data, count);
    unsigned long long mask, slot;

    if (!tokenizer || !tokenizer->vocab_index_capacity)
        return 0;
    mask = tokenizer->vocab_index_capacity - 1u;
    slot = hash & mask;
    while (tokenizer->vocab_index[slot].occupied) {
        const yvex_token_info *token = &tokenizer->tokens[tokenizer->vocab_index[slot].id];
        if (tokenizer->vocab_index[slot].hash == hash && token->text_len == count &&
            (!count || memcmp(token->text, data, count) == 0)) {
            *token_id = token->id;
            return 1;
        }
        slot = (slot + 1u) & mask;
    }
    return 0;
}

static int vocab_insert(yvex_tokenizer *tokenizer, unsigned int token_id)
{
    const yvex_token_info *token = &tokenizer->tokens[token_id];
    uint64_t hash = lookup_hash(token->text, (size_t)token->text_len);
    unsigned long long mask = tokenizer->vocab_index_capacity - 1u;
    unsigned long long slot = hash & mask;

    while (tokenizer->vocab_index[slot].occupied) {
        const yvex_token_info *prior = &tokenizer->tokens[tokenizer->vocab_index[slot].id];
        if (tokenizer->vocab_index[slot].hash == hash && prior->text_len == token->text_len &&
            (!token->text_len || memcmp(prior->text, token->text, (size_t)token->text_len) == 0))
            return prior->id == token_id;
        slot = (slot + 1u) & mask;
    }
    tokenizer->vocab_index[slot].hash = hash;
    tokenizer->vocab_index[slot].id = token_id;
    tokenizer->vocab_index[slot].occupied = 1;
    return 1;
}

static const tokenizer_merge_slot *merge_lookup(const yvex_tokenizer *tokenizer,
                                                unsigned int left, unsigned int right)
{
    uint64_t pair = ((uint64_t)left << 32u) | right;
    unsigned long long mask = tokenizer->merge_index_capacity - 1u;
    unsigned long long slot = lookup_hash(&pair, sizeof(pair)) & mask;

    while (tokenizer->merge_index[slot].occupied) {
        if (tokenizer->merge_index[slot].pair == pair)
            return &tokenizer->merge_index[slot];
        slot = (slot + 1u) & mask;
    }
    return NULL;
}

static size_t utf8_put(uint32_t point, unsigned char output[4])
{
    if (point <= 0x7fu) {
        output[0] = (unsigned char)point;
        return 1u;
    }
    if (point <= 0x7ffu) {
        output[0] = (unsigned char)(0xc0u | (point >> 6u));
        output[1] = (unsigned char)(0x80u | (point & 0x3fu));
        return 2u;
    }
    if (point <= 0xffffu) {
        output[0] = (unsigned char)(0xe0u | (point >> 12u));
        output[1] = (unsigned char)(0x80u | ((point >> 6u) & 0x3fu));
        output[2] = (unsigned char)(0x80u | (point & 0x3fu));
        return 3u;
    }
    output[0] = (unsigned char)(0xf0u | (point >> 18u));
    output[1] = (unsigned char)(0x80u | ((point >> 12u) & 0x3fu));
    output[2] = (unsigned char)(0x80u | ((point >> 6u) & 0x3fu));
    output[3] = (unsigned char)(0x80u | (point & 0x3fu));
    return 4u;
}

static uint32_t byte_codepoint(unsigned int byte)
{
    unsigned int candidate, extra = 0u;
    if ((byte >= 33u && byte <= 126u) || (byte >= 161u && byte <= 172u) ||
        (byte >= 174u && byte <= 255u))
        return byte;
    for (candidate = 0u; candidate < byte; ++candidate)
        if (!((candidate >= 33u && candidate <= 126u) ||
              (candidate >= 161u && candidate <= 172u) ||
              (candidate >= 174u && candidate <= 255u)))
            ++extra;
    return 256u + extra;
}

static int byte_tokens_build(yvex_tokenizer *tokenizer, yvex_error *err)
{
    unsigned int byte;
    for (byte = 0u; byte < 256u; ++byte) {
        unsigned char encoded[4];
        size_t count = utf8_put(byte_codepoint(byte), encoded);
        if (!vocab_lookup(tokenizer, encoded, count, &tokenizer->byte_token_ids[byte])) {
            yvex_error_setf(err, YVEX_ERR_FORMAT, "tokenizer.plan.bytelevel",
                            "ByteLevel initial unit %u is absent from vocabulary", byte);
            return YVEX_ERR_FORMAT;
        }
    }
    return YVEX_OK;
}

/*
 * Build and identity-seal the exact ID-indexed vocabulary lookup.
 *
 * Model lifetime.
 */
static int vocabulary_build(yvex_tokenizer *tokenizer, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, capacity = 1u;

    while (capacity < tokenizer->vocab_size * 2u)
        capacity *= 2u;
    if (capacity > (unsigned long long)(SIZE_MAX / sizeof(*tokenizer->vocab_index))) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.plan.vocabulary", "vocabulary index exceeds address space");
        return YVEX_ERR_NOMEM;
    }
    tokenizer->vocab_index = calloc((size_t)capacity, sizeof(*tokenizer->vocab_index));
    if (!tokenizer->vocab_index) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.plan.vocabulary", "vocabulary index allocation failed");
        return YVEX_ERR_NOMEM;
    }
    tokenizer->vocab_index_capacity = capacity;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.vocabulary.v1"))
        return YVEX_ERR_STATE;
    for (index = 0u; index < tokenizer->vocab_size; ++index) {
        const yvex_token_info *token = &tokenizer->tokens[index];
        if (!vocab_insert(tokenizer, (unsigned int)index) ||
            !yvex_sha256_update_u64_be(&hash, index) ||
            !yvex_sha256_update_u64_be(&hash, token->text_len) ||
            !yvex_sha256_update(&hash, token->text, (size_t)token->text_len) ||
            !yvex_sha256_update_u64_be(&hash, (unsigned long long)token->type)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.vocabulary",
                           "duplicate or unhashable vocabulary entry");
            return YVEX_ERR_FORMAT;
        }
    }
    if (!yvex_sha256_final(&hash, digest))
        return YVEX_ERR_STATE;
    yvex_sha256_hex(digest, tokenizer->plan.vocabulary_identity);
    return byte_tokens_build(tokenizer, err);
}

static int merge_insert(yvex_tokenizer *tokenizer, unsigned int left, unsigned int right,
                        unsigned int merged, unsigned int rank)
{
    uint64_t pair = ((uint64_t)left << 32u) | right;
    unsigned long long mask = tokenizer->merge_index_capacity - 1u;
    unsigned long long slot = lookup_hash(&pair, sizeof(pair)) & mask;

    while (tokenizer->merge_index[slot].occupied) {
        if (tokenizer->merge_index[slot].pair == pair)
            return 0;
        slot = (slot + 1u) & mask;
    }
    tokenizer->merge_index[slot].pair = pair;
    tokenizer->merge_index[slot].rank = rank;
    tokenizer->merge_index[slot].merged_id = merged;
    tokenizer->merge_index[slot].occupied = 1;
    return 1;
}

static int merge_parse(const yvex_tokenizer *tokenizer, const char *text,
                       unsigned long long count, unsigned int *left,
                       unsigned int *right, unsigned int *merged, yvex_error *err)
{
    const char *space;
    unsigned long long left_count, right_count;
    unsigned char *joined;

    if (!count || count > (unsigned long long)SIZE_MAX ||
        !(space = memchr(text, ' ', (size_t)count)) ||
        memchr(space + 1, ' ', (size_t)(text + count - space - 1))) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.merges", "merge row is not canonical");
        return YVEX_ERR_FORMAT;
    }
    left_count = (unsigned long long)(space - text);
    right_count = count - left_count - 1u;
    if (!left_count || !right_count ||
        !vocab_lookup(tokenizer, text, (size_t)left_count, left) ||
        !vocab_lookup(tokenizer, space + 1, (size_t)right_count, right)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.merges", "merge component is absent from vocabulary");
        return YVEX_ERR_FORMAT;
    }
    joined = (unsigned char *)malloc((size_t)(left_count + right_count));
    if (!joined) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.plan.merges", "merge scratch allocation failed");
        return YVEX_ERR_NOMEM;
    }
    memcpy(joined, text, (size_t)left_count);
    memcpy(joined + left_count, space + 1, (size_t)right_count);
    if (!vocab_lookup(tokenizer, joined, (size_t)(left_count + right_count), merged)) {
        free(joined);
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.merges", "merged token is absent from vocabulary");
        return YVEX_ERR_FORMAT;
    }
    free(joined);
    return YVEX_OK;
}

/*
 * Validate, index, and field-wise identity-seal all admitted merge rows.
 *
 * Model lifetime.
 */
static int merges_build(yvex_tokenizer *tokenizer, const yvex_gguf *gguf,
                        yvex_error *err)
{
    const yvex_gguf_value *value = yvex_gguf_metadata_find(gguf, "tokenizer.ggml.merges");
    yvex_gguf_array_info info;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long capacity = 1u, rank;

    if (!value || yvex_gguf_value_array_info(value, &info) != YVEX_OK ||
        info.element_type != YVEX_GGUF_VALUE_STRING ||
        info.count != tokenizer->compiled_policy.merge_count) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.merges",
                       "merge table differs from the compiled tokenizer policy");
        return YVEX_ERR_FORMAT;
    }
    while (capacity < info.count * 2u)
        capacity *= 2u;
    tokenizer->merge_index = calloc((size_t)capacity, sizeof(*tokenizer->merge_index));
    if (!tokenizer->merge_index) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.plan.merges", "merge index allocation failed");
        return YVEX_ERR_NOMEM;
    }
    tokenizer->merge_index_capacity = capacity;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.merges.v1"))
        return YVEX_ERR_STATE;
    for (rank = 0u; rank < info.count; ++rank) {
        const yvex_gguf_value *row = yvex_gguf_value_array_at(value, rank);
        const char *text;
        unsigned long long count;
        unsigned int left, right, merged;
        int rc;

        if (!row || yvex_gguf_value_as_string(row, &text, &count) != YVEX_OK)
            rc = YVEX_ERR_FORMAT;
        else
            rc = merge_parse(tokenizer, text, count, &left, &right, &merged, err);
        if (rc != YVEX_OK || !merge_insert(tokenizer, left, right, merged, (unsigned int)rank) ||
            !yvex_sha256_update_u64_be(&hash, rank) ||
            !yvex_sha256_update_u64_be(&hash, count) ||
            !yvex_sha256_update(&hash, text, (size_t)count)) {
            if (rc == YVEX_OK)
                yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.merges",
                               "duplicate merge pair or identity failure");
            return rc == YVEX_OK ? YVEX_ERR_FORMAT : rc;
        }
    }
    if (!yvex_sha256_final(&hash, digest))
        return YVEX_ERR_STATE;
    yvex_sha256_hex(digest, tokenizer->plan.merge_table_identity);
    tokenizer->plan.merge_count = info.count;
    return YVEX_OK;
}

static int added_compare(const void *left_value, const void *right_value, void *context)
{
    const yvex_tokenizer *tokenizer = context;
    unsigned int left = *(const unsigned int *)left_value;
    unsigned int right = *(const unsigned int *)right_value;
    const yvex_token_info *a = &tokenizer->tokens[left];
    const yvex_token_info *b = &tokenizer->tokens[right];
    unsigned char af = a->text_len ? (unsigned char)a->text[0] : 0u;
    unsigned char bf = b->text_len ? (unsigned char)b->text[0] : 0u;

    if (af != bf)
        return af < bf ? -1 : 1;
    if (a->text_len != b->text_len)
        return a->text_len > b->text_len ? -1 : 1;
    return left < right ? -1 : left != right;
}

static void added_sort(yvex_tokenizer *tokenizer)
{
    unsigned long long index;
    for (index = 1u; index < tokenizer->added_token_count; ++index) {
        unsigned int value = tokenizer->added_token_ids[index];
        unsigned long long cursor = index;
        while (cursor && added_compare(&value, &tokenizer->added_token_ids[cursor - 1u], tokenizer) < 0) {
            tokenizer->added_token_ids[cursor] = tokenizer->added_token_ids[cursor - 1u];
            --cursor;
        }
        tokenizer->added_token_ids[cursor] = value;
    }
}

/* Build exact added-token buckets and their field-wise identity. */
static int added_tokens_build(yvex_tokenizer *tokenizer, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, added = 0u, special = 0u, cursor = 0u;
    unsigned int byte;

    for (index = 0u; index < tokenizer->vocab_size; ++index)
        if (tokenizer->tokens[index].type == YVEX_TOKEN_TYPE_CONTROL ||
            tokenizer->tokens[index].type == YVEX_TOKEN_TYPE_USER_DEFINED)
            ++added;
    if (added != tokenizer->compiled_policy.added_token_count ||
        added > SIZE_MAX / sizeof(unsigned int)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.added",
                       "added-token set differs from the compiled tokenizer policy");
        return YVEX_ERR_FORMAT;
    }
    tokenizer->added_token_ids = malloc((size_t)added * sizeof(*tokenizer->added_token_ids));
    if (!tokenizer->added_token_ids) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.plan.added", "added-token index allocation failed");
        return YVEX_ERR_NOMEM;
    }
    for (index = 0u; index < tokenizer->vocab_size; ++index)
        if (tokenizer->tokens[index].type == YVEX_TOKEN_TYPE_CONTROL ||
            tokenizer->tokens[index].type == YVEX_TOKEN_TYPE_USER_DEFINED) {
            tokenizer->added_token_ids[cursor++] = (unsigned int)index;
            if (tokenizer->tokens[index].type == YVEX_TOKEN_TYPE_CONTROL)
                ++special;
        }
    tokenizer->added_token_count = added;
    added_sort(tokenizer);
    cursor = 0u;
    for (byte = 0u; byte < 256u; ++byte) {
        tokenizer->added_bucket_start[byte] = cursor;
        while (cursor < added && tokenizer->tokens[tokenizer->added_token_ids[cursor]].text_len &&
               (unsigned char)tokenizer->tokens[tokenizer->added_token_ids[cursor]].text[0] == byte)
            ++cursor;
    }
    tokenizer->added_bucket_start[256] = added;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.added.v1"))
        return YVEX_ERR_STATE;
    for (index = 0u; index < added; ++index) {
        const yvex_token_info *token = &tokenizer->tokens[tokenizer->added_token_ids[index]];
        if (!yvex_sha256_update_u64_be(&hash, token->id) ||
            !yvex_sha256_update_u64_be(&hash, token->text_len) ||
            !yvex_sha256_update(&hash, token->text, (size_t)token->text_len) ||
            !yvex_sha256_update_u64_be(&hash, token->type == YVEX_TOKEN_TYPE_CONTROL))
            return YVEX_ERR_STATE;
    }
    if (!yvex_sha256_final(&hash, digest))
        return YVEX_ERR_STATE;
    yvex_sha256_hex(digest, tokenizer->plan.added_token_identity);
    tokenizer->plan.added_token_count = added;
    tokenizer->plan.special_token_count = special;
    return YVEX_OK;
}

static int special_identity_build(yvex_tokenizer *tokenizer)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const tokenizer_special_id *facts[] = {
        &tokenizer->bos, &tokenizer->eos, &tokenizer->pad, &tokenizer->unk
    };
    size_t index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.special-policy.v1"))
        return 0;
    for (index = 0u; index < sizeof(facts) / sizeof(facts[0]); ++index)
        if (!yvex_sha256_update_u64_be(&hash, facts[index]->present) ||
            !yvex_sha256_update_u64_be(&hash, facts[index]->id))
            return 0;
    if (!yvex_sha256_update_u64_be(&hash, 0u) ||
        !yvex_sha256_update_u64_be(&hash, 0u) || !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, tokenizer->plan.special_policy_identity);
    return 1;
}

static int prompt_identity_build(yvex_tokenizer *tokenizer)
{
    const yvex_conversation_protocol *conversation = tokenizer->conversation;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    size_t index;

    yvex_sha256_init(&hash);
    if (tokenizer->plan.prompt_policy != YVEX_TOKENIZER_PROMPT_CONVERSATION) {
        if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.verbatim-prompt.v1") ||
            !yvex_sha256_update_text(
                &hash, tokenizer->compiled_policy.direct_prompt_name) ||
            !yvex_sha256_final(&hash, digest))
            return 0;
        yvex_sha256_hex(digest, tokenizer->plan.prompt_policy_identity);
        return 1;
    }
    if (!conversation) return 0;
    {
        const char *facts[] = {
            conversation->source_revision, conversation->source_encoding_path,
            conversation->source_encoding_identity, conversation->bos, conversation->eos,
            conversation->user, conversation->assistant,
            conversation->latest_reminder, conversation->thinking_start,
            conversation->thinking_end, conversation->tool_result_start,
            conversation->tool_result_end, conversation->dsml,
            conversation->tool_calls_start, conversation->tool_calls_end,
            conversation->tool_invoke_start, conversation->tool_invoke_name_end,
            conversation->tool_invoke_end, conversation->tool_parameter_start,
            conversation->tool_parameter_name_end,
            conversation->tool_parameter_kind_end,
            conversation->tool_parameter_end,
            conversation->reasoning_effort_max, conversation->tools_prefix,
            conversation->tools_suffix, conversation->response_format_prefix};

    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.conversation-prompt.v3") ||
        !yvex_sha256_update_u64_be(
            &hash, conversation->drop_prior_reasoning_by_default) ||
        !yvex_sha256_update_u64_be(&hash,
                                   conversation->tools_preserve_reasoning) ||
        !yvex_sha256_update_u64_be(
            &hash, conversation->tool_results_merge_into_user))
        return 0;
    for (index = 0u; index < sizeof(facts) / sizeof(facts[0]); ++index)
        if (!yvex_sha256_update_u64_be(&hash, strlen(facts[index])) ||
            !yvex_sha256_update(&hash, facts[index], strlen(facts[index])))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, tokenizer->plan.prompt_policy_identity);
    return 1;
}

static int raw_sha256(const void *data, size_t count, char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, data, count) || !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

/* Derive the complete tokenizer plan identity field by field. */
static int plan_identity_build(yvex_tokenizer *tokenizer)
{
    yvex_tokenizer_plan_summary *plan = &tokenizer->plan;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const char *identities[] = {
        plan->artifact_identity, plan->logical_model_identity,
        plan->runtime_descriptor_identity, plan->tokenizer_json_identity,
        plan->tokenizer_config_identity, plan->vocabulary_identity,
        plan->merge_table_identity, plan->added_token_identity,
        plan->special_policy_identity, plan->prompt_policy_identity
    };
    size_t index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.plan.v4") ||
        !yvex_sha256_update_u64_be(&hash, plan->schema_version) ||
        !yvex_sha256_update_u64_be(&hash, plan->family_adapter_id) ||
        !yvex_sha256_update_u64_be(&hash, plan->family_adapter_version) ||
        !yvex_sha256_update_u64_be(&hash, plan->vocabulary_size) ||
        !yvex_sha256_update_u64_be(&hash, plan->base_vocabulary_size) ||
        !yvex_sha256_update_u64_be(&hash, plan->merge_count) ||
        !yvex_sha256_update_u64_be(&hash, plan->added_token_count) ||
        !yvex_sha256_update_u64_be(&hash, plan->special_token_count) ||
        !yvex_sha256_update_u64_be(&hash, plan->model_policy) ||
        !yvex_sha256_update_u64_be(&hash, plan->prompt_policy) ||
        !yvex_sha256_update_u64_be(&hash, plan->add_bos_token) ||
        !yvex_sha256_update_u64_be(&hash, plan->add_eos_token) ||
        !yvex_sha256_update_u64_be(&hash, plan->byte_fallback) ||
        !yvex_sha256_update_u64_be(&hash, plan->reasoning_start_token_id) ||
        !yvex_sha256_update_u64_be(&hash, plan->reasoning_end_token_id) ||
        !yvex_sha256_update_u64_be(&hash, plan->explicit_reasoning_supported) ||
        !yvex_sha256_update_u64_be(&hash, plan->maximum_reasoning_supported))
        return 0;
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index)
        if (!yvex_sha256_update_text(&hash, identities[index]))
            return 0;
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, plan->tokenizer_plan_identity);
    return 1;
}

static int exact_policy_admit(yvex_tokenizer *tokenizer, const yvex_gguf *gguf,
                              yvex_error *err)
{
    const yvex_tokenizer_family_policy *policy = &tokenizer->compiled_policy;
    const char *architecture, *pre, *json, *config;
    unsigned long long architecture_count, pre_count, json_count, config_count;

    if (yvex_tokenizer_family_policy_validate(policy, NULL) != YVEX_OK)
        return YVEX_ERR_UNSUPPORTED;
    if (!gguf_string(gguf, "general.architecture", &architecture,
                     &architecture_count) ||
        architecture_count != strlen(policy->architecture) ||
        memcmp(architecture, policy->architecture, (size_t)architecture_count) != 0)
        return YVEX_ERR_UNSUPPORTED;
    if (policy->prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION &&
        (!tokenizer->conversation || !tokenizer->conversation->source_revision ||
        !tokenizer->conversation->source_encoding_path ||
        !tokenizer->conversation->source_encoding_identity ||
        !tokenizer->conversation->bos || !tokenizer->conversation->eos ||
        !tokenizer->conversation->user || !tokenizer->conversation->assistant ||
        !tokenizer->conversation->thinking_start ||
        !tokenizer->conversation->thinking_end ||
        !tokenizer->conversation->tool_calls_start ||
        !tokenizer->conversation->tool_calls_end ||
        !tokenizer->conversation->tool_invoke_start ||
        !tokenizer->conversation->tool_invoke_name_end ||
        !tokenizer->conversation->tool_invoke_end ||
        !tokenizer->conversation->tool_parameter_start ||
        !tokenizer->conversation->tool_parameter_name_end ||
        !tokenizer->conversation->tool_parameter_kind_end ||
        !tokenizer->conversation->tool_parameter_end ||
        !tokenizer->conversation->reasoning_effort_max ||
        !tokenizer->conversation->tools_prefix ||
        !tokenizer->conversation->tools_suffix))
        return YVEX_ERR_UNSUPPORTED;
    if (tokenizer->kind != policy->tokenizer_kind ||
        !tokenizer->model_name || strcmp(tokenizer->model_name, policy->tokenizer_model) != 0 ||
        tokenizer->vocab_size != policy->vocabulary_size ||
        !gguf_string(gguf, "tokenizer.ggml.pre", &pre, &pre_count) ||
        pre_count != strlen(policy->tokenizer_pre) ||
        memcmp(pre, policy->tokenizer_pre, pre_count) != 0 ||
        !gguf_string(gguf, "tokenizer.huggingface.json", &json, &json_count) ||
        !gguf_string(gguf, "yvex.tokenizer.config.json", &config, &config_count) ||
        json_count > SIZE_MAX || config_count > SIZE_MAX ||
        !raw_sha256(json, (size_t)json_count, tokenizer->plan.tokenizer_json_identity) ||
        !raw_sha256(config, (size_t)config_count, tokenizer->plan.tokenizer_config_identity)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.admission",
                       "artifact tokenizer metadata differs from the compiled policy");
        return YVEX_ERR_FORMAT;
    }
    if (policy->prompt_policy != YVEX_TOKENIZER_PROMPT_CONVERSATION) {
        const char *prompt;
        unsigned long long prompt_count;
        if (!gguf_string(gguf, "yvex.tokenizer.prompt_policy", &prompt, &prompt_count) ||
            prompt_count != strlen(policy->direct_prompt_name) ||
            memcmp(prompt, policy->direct_prompt_name, (size_t)prompt_count) != 0) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.admission",
                           "artifact direct-prompt metadata differs from compiled policy");
            return YVEX_ERR_FORMAT;
        }
    }
    if (strcmp(tokenizer->plan.tokenizer_json_identity, policy->tokenizer_json_identity) != 0 ||
        strcmp(tokenizer->plan.tokenizer_config_identity,
               policy->tokenizer_config_identity) != 0) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "tokenizer.plan.components",
                       "tokenizer JSON/config identities differ from the compiled policy");
        return YVEX_ERR_UNSUPPORTED;
    }
    return YVEX_OK;
}

/*
 * Seal the exact immutable tokenizer plan and all reusable indexes.
 *
 * Model lifetime.
 */
int yvex_tokenizer_execution_seal(yvex_tokenizer *tokenizer, const yvex_gguf *gguf,
                                  const yvex_tokenizer_family_policy *policy,
                                  yvex_error *err)
{
    int rc;
    if (!tokenizer || !gguf)
        return YVEX_ERR_INVALID_ARG;
    if (!policy) return YVEX_ERR_UNSUPPORTED;
    memset(&tokenizer->plan, 0, sizeof(tokenizer->plan));
    rc = exact_policy_admit(tokenizer, gguf, err);
    if (rc != YVEX_OK)
        return rc;
    tokenizer->plan.schema_version = YVEX_TOKENIZER_PLAN_SCHEMA_CURRENT;
    tokenizer->plan.family_adapter_id = policy->family_adapter_id;
    tokenizer->plan.family_adapter_version = policy->family_adapter_version;
    tokenizer->plan.vocabulary_size = tokenizer->vocab_size;
    tokenizer->plan.base_vocabulary_size = policy->base_vocabulary_size;
    tokenizer->plan.model_policy = policy->model_policy;
    tokenizer->plan.prompt_policy = policy->prompt_policy;
    tokenizer->plan.add_bos_token = policy->add_bos_token;
    tokenizer->plan.add_eos_token = policy->add_eos_token;
    tokenizer->plan.byte_fallback = policy->byte_fallback;
    rc = vocabulary_build(tokenizer, err);
    if (rc == YVEX_OK)
        rc = merges_build(tokenizer, gguf, err);
    if (rc == YVEX_OK)
        rc = added_tokens_build(tokenizer, err);
    if (rc == YVEX_OK && policy->prompt_policy == YVEX_TOKENIZER_PROMPT_CONVERSATION) {
        unsigned int start_id, end_id;
        if (!vocab_lookup(
                tokenizer,
                (const unsigned char *)tokenizer->conversation->thinking_start,
                strlen(tokenizer->conversation->thinking_start), &start_id) ||
            !vocab_lookup(
                tokenizer,
                (const unsigned char *)tokenizer->conversation->thinking_end,
                strlen(tokenizer->conversation->thinking_end), &end_id)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.reasoning",
                           "source-authored reasoning delimiters are absent");
            rc = YVEX_ERR_FORMAT;
        } else {
            tokenizer->plan.reasoning_start_token_id = start_id;
            tokenizer->plan.reasoning_end_token_id = end_id;
            tokenizer->plan.explicit_reasoning_supported = 1;
            tokenizer->plan.maximum_reasoning_supported = 1;
        }
    }
    if (rc == YVEX_OK && (!special_identity_build(tokenizer) ||
                          !prompt_identity_build(tokenizer))) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.plan.identity", "policy identity derivation failed");
        rc = YVEX_ERR_STATE;
    }
    tokenizer->plan.bos_present = tokenizer->bos.present;
    tokenizer->plan.eos_present = tokenizer->eos.present;
    tokenizer->plan.pad_present = tokenizer->pad.present;
    tokenizer->plan.unk_present = tokenizer->unk.present;
    tokenizer->plan.bos_token_id = tokenizer->bos.id;
    tokenizer->plan.eos_token_id = tokenizer->eos.id;
    tokenizer->plan.pad_token_id = tokenizer->pad.id;
    tokenizer->plan.unk_token_id = tokenizer->unk.id;
    if (rc == YVEX_OK &&
        (tokenizer->plan.special_token_count != policy->special_token_count ||
         tokenizer->bos.present != policy->bos_present ||
         tokenizer->bos.id != policy->bos_token_id ||
         tokenizer->eos.present != policy->eos_present ||
         tokenizer->eos.id != policy->eos_token_id ||
         tokenizer->pad.present != policy->pad_present ||
         tokenizer->pad.id != policy->pad_token_id ||
         tokenizer->unk.present != policy->unk_present ||
         tokenizer->unk.id != policy->unk_token_id)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.plan.specials",
                       "special-token facts differ from the compiled tokenizer policy");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        tokenizer->plan.owned_bytes =
            tokenizer->vocab_index_capacity * sizeof(*tokenizer->vocab_index) +
            tokenizer->merge_index_capacity * sizeof(*tokenizer->merge_index) +
            tokenizer->added_token_count * sizeof(*tokenizer->added_token_ids);
        tokenizer->plan.sealed = 1;
        if (!plan_identity_build(tokenizer)) {
            yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.plan.identity", "plan identity derivation failed");
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc != YVEX_OK) {
        yvex_tokenizer_execution_release(tokenizer);
        return rc;
    }
    tokenizer->support = YVEX_TOKENIZER_SUPPORT_ARTIFACT_BPE;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Release exact execution indexes without touching shared vocabulary ownership.
 *
 * Model lifetime.
 */
void yvex_tokenizer_execution_release(yvex_tokenizer *tokenizer)
{
    if (!tokenizer)
        return;
    free(tokenizer->vocab_index);
    free(tokenizer->merge_index);
    free(tokenizer->added_token_ids);
    tokenizer->vocab_index = NULL;
    tokenizer->merge_index = NULL;
    tokenizer->added_token_ids = NULL;
    tokenizer->vocab_index_capacity = 0u;
    tokenizer->merge_index_capacity = 0u;
    tokenizer->added_token_count = 0u;
    memset(tokenizer->added_bucket_start, 0, sizeof(tokenizer->added_bucket_start));
    memset(&tokenizer->plan, 0, sizeof(tokenizer->plan));
}

/*
 * Expose one immutable sealed tokenizer-plan summary.
 *
 * Tokenizer lifetime.
 */
const yvex_tokenizer_plan_summary *yvex_tokenizer_plan_summary_get(const yvex_tokenizer *tokenizer)
{
    return tokenizer && tokenizer->plan.sealed ? &tokenizer->plan : NULL;
}

/* Bind the tokenizer plan to one already admitted runtime model identity triple. */
int yvex_tokenizer_bind_runtime(yvex_tokenizer *tokenizer,
                                const char *artifact_identity,
                                const char *logical_model_identity,
                                const char *runtime_descriptor_identity,
                                yvex_error *err)
{
    if (!tokenizer || !tokenizer->plan.sealed || tokenizer->plan.runtime_bound ||
        !yvex_sha256_hex_is_valid(artifact_identity) ||
        !yvex_sha256_hex_is_valid(logical_model_identity) ||
        !yvex_sha256_hex_is_valid(runtime_descriptor_identity)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.plan.runtime-bind",
                       "one unbound sealed plan and valid runtime identities are required");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_core_text_copy(tokenizer->plan.artifact_identity,
                        sizeof(tokenizer->plan.artifact_identity), artifact_identity);
    yvex_core_text_copy(tokenizer->plan.logical_model_identity,
                        sizeof(tokenizer->plan.logical_model_identity), logical_model_identity);
    yvex_core_text_copy(tokenizer->plan.runtime_descriptor_identity,
                        sizeof(tokenizer->plan.runtime_descriptor_identity), runtime_descriptor_identity);
    tokenizer->plan.runtime_bound = 1;
    if (!plan_identity_build(tokenizer)) {
        tokenizer->plan.runtime_bound = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.plan.runtime-bind", "bound plan identity failed");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int span_peek(const tokenizer_span *span, unsigned long long offset,
                     uint32_t *point, unsigned long long *next)
{
    unsigned long long cursor = offset;
    if (offset >= span->count ||
        !yvex_tokenizer_utf8_next(span->bytes, span->count, &cursor, point))
        return 0;
    if (next)
        *next = cursor;
    return 1;
}

static int is_cjk_split(uint32_t point)
{
    return (point >= 0x4e00u && point <= 0x9fa5u) ||
           (point >= 0x3040u && point <= 0x309fu) ||
           (point >= 0x30a0u && point <= 0x30ffu);
}

static int is_ascii_punctuation(uint32_t point)
{
    return (point >= 0x21u && point <= 0x2fu) ||
           (point >= 0x3au && point <= 0x40u) ||
           (point >= 0x5bu && point <= 0x60u) ||
           (point >= 0x7bu && point <= 0x7eu);
}

static unsigned long long take_class(const tokenizer_span *span, unsigned long long offset,
                                     unsigned int required)
{
    uint32_t point;
    unsigned long long next;
    while (span_peek(span, offset, &point, &next) &&
           (yvex_tokenizer_unicode_class(point) & required) != 0u)
        offset = next;
    return offset;
}

static unsigned long long take_numbers(const tokenizer_span *span, unsigned long long offset,
                                       unsigned int maximum)
{
    uint32_t point;
    unsigned long long next;
    unsigned int count = 0u;
    while (count < maximum && span_peek(span, offset, &point, &next) &&
           (yvex_tokenizer_unicode_class(point) & TOKENIZER_UNICODE_NUMBER) != 0u) {
        offset = next;
        ++count;
    }
    return offset;
}

static unsigned long long take_cjk(const tokenizer_span *span, unsigned long long offset)
{
    uint32_t point;
    unsigned long long next;
    while (span_peek(span, offset, &point, &next) && is_cjk_split(point))
        offset = next;
    return offset;
}

static unsigned long long take_word_or_symbol(const tokenizer_span *span,
                                              unsigned long long offset,
                                              uint32_t point,
                                              unsigned long long next)
{
    uint32_t following;
    unsigned long long after;
    unsigned int classification = yvex_tokenizer_unicode_class(point);

    if (is_ascii_punctuation(point) && span_peek(span, next, &following, &after) &&
        ((following >= 'A' && following <= 'Z') || (following >= 'a' && following <= 'z'))) {
        while (span_peek(span, next, &following, &after) &&
               ((following >= 'A' && following <= 'Z') ||
                (following >= 'a' && following <= 'z')))
            next = after;
        return next;
    }
    if ((classification & TOKENIZER_UNICODE_LETTER_MARK) != 0u)
        return take_class(span, offset, TOKENIZER_UNICODE_LETTER_MARK);
    if (point != '\r' && point != '\n' &&
        (classification & (TOKENIZER_UNICODE_LETTER_MARK |
                           TOKENIZER_UNICODE_PUNCT_SYMBOL)) == 0u &&
        span_peek(span, next, &following, &after) &&
        (yvex_tokenizer_unicode_class(following) & TOKENIZER_UNICODE_LETTER_MARK) != 0u)
        return take_class(span, next, TOKENIZER_UNICODE_LETTER_MARK);
    if ((classification & TOKENIZER_UNICODE_PUNCT_SYMBOL) != 0u) {
        unsigned long long end = take_class(span, offset, TOKENIZER_UNICODE_PUNCT_SYMBOL);
        while (span_peek(span, end, &following, &after) &&
               (following == '\r' || following == '\n'))
            end = after;
        return end;
    }
    return offset;
}

static unsigned long long take_space(const tokenizer_span *span, unsigned long long offset)
{
    uint32_t point, following;
    unsigned long long next, run_end = offset, prior = offset, after;
    unsigned int scalar_count = 0u;
    int has_newline = 0;

    while (span_peek(span, run_end, &point, &next) &&
           (yvex_tokenizer_unicode_class(point) & TOKENIZER_UNICODE_SPACE) != 0u) {
        prior = run_end;
        run_end = next;
        ++scalar_count;
        if (point == '\r' || point == '\n')
            has_newline = 1;
    }
    if (has_newline || run_end == span->count)
        return run_end;
    if (span_peek(span, run_end, &following, &after) &&
        (yvex_tokenizer_unicode_class(following) & TOKENIZER_UNICODE_LETTER_MARK) != 0u) {
        if (scalar_count > 1u)
            return prior;
        return take_class(span, run_end, TOKENIZER_UNICODE_LETTER_MARK);
    }
    if (span->bytes[prior] == ' ' && span_peek(span, run_end, &following, &after) &&
        (yvex_tokenizer_unicode_class(following) & TOKENIZER_UNICODE_PUNCT_SYMBOL) != 0u) {
        if (scalar_count > 1u)
            return prior;
        run_end = take_class(span, run_end, TOKENIZER_UNICODE_PUNCT_SYMBOL);
        while (span_peek(span, run_end, &following, &after) &&
               (following == '\r' || following == '\n'))
            run_end = after;
    }
    return run_end;
}

static unsigned long long next_piece(const tokenizer_span *span, unsigned long long offset,
                                     int qwen2_pretokenizer)
{
    uint32_t point;
    unsigned long long next, end;
    unsigned int classification;

    if (!span_peek(span, offset, &point, &next))
        return span->count;
    classification = yvex_tokenizer_unicode_class(point);
    if ((classification & TOKENIZER_UNICODE_NUMBER) != 0u)
        return take_numbers(span, offset, qwen2_pretokenizer ? 1u : 3u);
    if (!qwen2_pretokenizer && is_cjk_split(point))
        return take_cjk(span, offset);
    end = take_word_or_symbol(span, offset, point, next);
    if (end != offset)
        return end;
    if ((classification & TOKENIZER_UNICODE_SPACE) != 0u)
        return take_space(span, offset);
    return next;
}

/*
 * Append one token under the caller's pre-admitted result capacity.
 *
 * Encoding transaction.
 */
static int encode_append(yvex_tokens *tokens, unsigned int token_id,
                         unsigned long long maximum, yvex_error *err)
{
    unsigned int *grown;
    unsigned long long capacity;

    if (tokens->len >= maximum) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.encode", "token output capacity exceeded");
        return YVEX_ERR_BOUNDS;
    }
    if (tokens->len < tokens->cap) {
        tokens->ids[tokens->len++] = token_id;
        return YVEX_OK;
    }
    capacity = tokens->cap ? tokens->cap * 2u : 16u;
    if (capacity > maximum)
        capacity = maximum;
    if (capacity <= tokens->cap || capacity > SIZE_MAX / sizeof(*tokens->ids)) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.encode", "token output allocation overflow");
        return YVEX_ERR_NOMEM;
    }
    grown = realloc(tokens->ids, (size_t)capacity * sizeof(*tokens->ids));
    if (!grown) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.encode", "token output allocation failed");
        return YVEX_ERR_NOMEM;
    }
    tokens->ids = grown;
    tokens->cap = capacity;
    tokens->ids[tokens->len++] = token_id;
    return YVEX_OK;
}

static int bpe_piece(const yvex_tokenizer *tokenizer, const unsigned char *bytes,
                     unsigned long long count, yvex_tokens *tokens,
                     unsigned long long maximum, yvex_error *err)
{
    unsigned int *symbols;
    unsigned long long symbol_count = count, index;
    int rc = YVEX_OK;

    if (!count)
        return YVEX_OK;
    if (count > SIZE_MAX / sizeof(*symbols))
        return YVEX_ERR_NOMEM;
    symbols = malloc((size_t)count * sizeof(*symbols));
    if (!symbols) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.encode.bpe", "piece workspace allocation failed");
        return YVEX_ERR_NOMEM;
    }
    for (index = 0u; index < count; ++index)
        symbols[index] = tokenizer->byte_token_ids[bytes[index]];
    while (symbol_count > 1u) {
        const tokenizer_merge_slot *best = NULL;
        unsigned long long write = 0u;
        for (index = 0u; index + 1u < symbol_count; ++index) {
            const tokenizer_merge_slot *candidate =
                merge_lookup(tokenizer, symbols[index], symbols[index + 1u]);
            if (candidate && (!best || candidate->rank < best->rank))
                best = candidate;
        }
        if (!best)
            break;
        for (index = 0u; index < symbol_count;) {
            uint64_t pair = index + 1u < symbol_count
                ? ((uint64_t)symbols[index] << 32u) | symbols[index + 1u] : UINT64_MAX;
            if (pair == best->pair) {
                symbols[write++] = best->merged_id;
                index += 2u;
            } else {
                symbols[write++] = symbols[index++];
            }
        }
        symbol_count = write;
    }
    for (index = 0u; index < symbol_count && rc == YVEX_OK; ++index)
        rc = encode_append(tokens, symbols[index], maximum, err);
    free(symbols);
    return rc;
}

static int added_at(const yvex_tokenizer *tokenizer, const unsigned char *bytes,
                    unsigned long long count, unsigned long long offset,
                    int allow_special, unsigned int *token_id,
                    unsigned long long *matched)
{
    unsigned int first;
    unsigned long long index, end;
    if (offset >= count)
        return 0;
    first = bytes[offset];
    index = tokenizer->added_bucket_start[first];
    end = tokenizer->added_bucket_start[first + 1u];
    for (; index < end; ++index) {
        const yvex_token_info *token = &tokenizer->tokens[tokenizer->added_token_ids[index]];
        if ((!allow_special && token->type == YVEX_TOKEN_TYPE_CONTROL) ||
            !token->text_len || token->text_len > count - offset)
            continue;
        if (memcmp(bytes + offset, token->text, (size_t)token->text_len) == 0) {
            *token_id = token->id;
            *matched = token->text_len;
            return 1;
        }
    }
    return 0;
}

static unsigned long long next_added_offset(const yvex_tokenizer *tokenizer,
                                            const unsigned char *bytes,
                                            unsigned long long count,
                                            unsigned long long offset,
                                            int allow_special)
{
    unsigned int token_id;
    unsigned long long matched;
    while (offset < count) {
        if (added_at(tokenizer, bytes, count, offset, allow_special, &token_id, &matched))
            return offset;
        ++offset;
    }
    return count;
}

static int ordinary_encode(const yvex_tokenizer *tokenizer, const unsigned char *bytes,
                           unsigned long long count, yvex_tokens *tokens,
                           unsigned long long maximum, yvex_error *err)
{
    tokenizer_span span = {bytes, count, 0u};
    int qwen2_pretokenizer = strcmp(tokenizer->compiled_policy.tokenizer_pre, "qwen2") == 0;

    while (span.offset < span.count) {
        unsigned long long end = next_piece(&span, span.offset, qwen2_pretokenizer);
        int rc;
        if (end <= span.offset || end > span.count) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.encode.pretokenizer",
                           "pre-tokenizer failed to advance canonically");
            return YVEX_ERR_FORMAT;
        }
        rc = bpe_piece(tokenizer, span.bytes + span.offset, end - span.offset,
                       tokens, maximum, err);
        if (rc != YVEX_OK)
            return rc;
        span.offset = end;
    }
    return YVEX_OK;
}

static int utf8_validate(const unsigned char *bytes, unsigned long long count)
{
    unsigned long long offset = 0u;
    uint32_t point;
    while (offset < count)
        if (!yvex_tokenizer_utf8_next(bytes, count, &offset, &point))
            return 0;
    return 1;
}

static int encoding_identity_build(const yvex_tokenizer *tokenizer,
                                   yvex_tokenizer_encode_result *result)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.token-ids.v1") ||
        !yvex_sha256_update_u64_be(&hash, result->tokens.len))
        return 0;
    for (index = 0u; index < result->tokens.len; ++index)
        if (!yvex_sha256_update_u64_be(&hash, result->tokens.ids[index]))
            return 0;
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, result->token_ids_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.encoding.v1") ||
        !yvex_sha256_update_text(&hash, tokenizer->plan.tokenizer_plan_identity) ||
        !yvex_sha256_update_text(&hash, result->input_identity) ||
        !yvex_sha256_update_text(&hash, result->token_ids_identity) ||
        !yvex_sha256_update_u64_be(&hash, result->input_bytes) ||
        !yvex_sha256_update_u64_be(&hash, result->added_token_matches) ||
        !yvex_sha256_update_u64_be(&hash, result->bos_inserted) ||
        !yvex_sha256_update_u64_be(&hash, result->eos_inserted) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, result->encoding_identity);
    return 1;
}

/*
 * Encode one exact byte span through added-token, pre-tokenizer, ByteLevel, and BPE policy.
 *
 * Rollback.
 */
int yvex_tokenizer_encode(const yvex_tokenizer *tokenizer,
                          const unsigned char *bytes,
                          unsigned long long byte_count,
                          const yvex_tokenizer_encode_options *options,
                          yvex_tokenizer_encode_result *result,
                          yvex_error *err)
{
    yvex_tokenizer_encode_result candidate;
    unsigned long long offset = 0u;
    unsigned long long maximum = options && options->maximum_tokens
        ? options->maximum_tokens : ULLONG_MAX;
    const unsigned char *normalized_bytes = bytes;
    unsigned char *normalized_owner = NULL;
    unsigned long long normalized_count = byte_count;
    int allow_special = options ? options->allow_special_tokens : 1;
    int rc = YVEX_OK;

    if (!tokenizer || !result || (!bytes && byte_count) || byte_count > SIZE_MAX || !maximum) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.encode",
                       "sealed tokenizer, explicit byte span, and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!tokenizer->plan.sealed) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.encode",
                       "tokenizer execution policy is not sealed");
        return YVEX_ERR_STATE;
    }
    if (options && ((options->add_bos && !tokenizer->plan.add_bos_token) ||
                    (options->add_eos && !tokenizer->plan.add_eos_token))) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "tokenizer.encode.special-policy",
                       "requested BOS/EOS insertion is not admitted by the tokenizer policy");
        return YVEX_ERR_UNSUPPORTED;
    }
    memset(&candidate, 0, sizeof(candidate));
    if (!utf8_validate(bytes, byte_count)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.encode.utf8", "input is not canonical UTF-8");
        return YVEX_ERR_FORMAT;
    }
    if (strcmp(tokenizer->compiled_policy.tokenizer_pre, "qwen2") == 0) {
        rc = yvex_tokenizer_nfc_normalize(bytes, byte_count, &normalized_owner,
                                          &normalized_count, err);
        normalized_bytes = normalized_owner;
    }
    candidate.schema_version = YVEX_TOKENIZER_EXECUTION_SCHEMA_V1;
    candidate.input_bytes = byte_count;
    if (!bytes_identity("yvex.tokenizer.input.v1", bytes, (size_t)byte_count,
                        candidate.input_identity))
        rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK && options && options->add_bos) {
        if (!tokenizer->bos.present)
            rc = YVEX_ERR_UNSUPPORTED;
        else {
            rc = encode_append(&candidate.tokens, tokenizer->bos.id, maximum, err);
            candidate.bos_inserted = rc == YVEX_OK;
        }
    }
    while (rc == YVEX_OK && offset < normalized_count) {
        unsigned int added_id;
        unsigned long long matched;
        if (added_at(tokenizer, normalized_bytes, normalized_count, offset, allow_special,
                     &added_id, &matched)) {
            rc = encode_append(&candidate.tokens, added_id, maximum, err);
            if (rc == YVEX_OK) {
                ++candidate.added_token_matches;
                offset += matched;
            }
        } else {
            unsigned long long end = next_added_offset(tokenizer, normalized_bytes, normalized_count,
                                                       offset + 1u, allow_special);
            rc = ordinary_encode(tokenizer, normalized_bytes + offset, end - offset,
                                 &candidate.tokens, maximum, err);
            offset = end;
        }
    }
    if (rc == YVEX_OK && options && options->add_eos) {
        if (!tokenizer->eos.present)
            rc = YVEX_ERR_UNSUPPORTED;
        else {
            rc = encode_append(&candidate.tokens, tokenizer->eos.id, maximum, err);
            candidate.eos_inserted = rc == YVEX_OK;
        }
    }
    if (rc == YVEX_OK) {
        yvex_core_text_copy(candidate.tokenizer_plan_identity,
                            sizeof(candidate.tokenizer_plan_identity),
                            tokenizer->plan.tokenizer_plan_identity);
        candidate.completed = 1;
        if (!encoding_identity_build(tokenizer, &candidate)) {
            yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.encode.identity", "encoding identity failed");
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc != YVEX_OK) {
        yvex_tokens_free(&candidate.tokens);
        free(normalized_owner);
        memset(result, 0, sizeof(*result));
        return rc;
    }
    free(normalized_owner);
    *result = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_tokenizer_encode_result_clear(yvex_tokenizer_encode_result *result)
{
    if (!result)
        return;
    yvex_tokens_free(&result->tokens);
    memset(result, 0, sizeof(*result));
}

/* Map prompt roles to stable operator and identity labels. */
const char *yvex_prompt_role_name(yvex_prompt_role role)
{
    switch (role) {
    case YVEX_PROMPT_ROLE_SYSTEM: return "system";
    case YVEX_PROMPT_ROLE_USER: return "user";
    case YVEX_PROMPT_ROLE_ASSISTANT: return "assistant";
    case YVEX_PROMPT_ROLE_TOOL: return "tool";
    }
    return "unknown";
}

static int message_span(const yvex_prompt_message *message, const unsigned char **bytes,
                        unsigned long long *count)
{
    if (!message || !message->content)
        return 0;
    *bytes = (const unsigned char *)message->content;
    *count = message->content_len ? message->content_len
                                  : (unsigned long long)strlen(message->content);
    return *count <= SIZE_MAX && utf8_validate(*bytes, *count);
}

static int prompt_roles_valid(const yvex_prompt_message *messages,
                              unsigned long long count,
                              const yvex_prompt_options *options)
{
    unsigned long long index;
    yvex_prompt_role prior = YVEX_PROMPT_ROLE_SYSTEM;
    for (index = 0u; index < count; ++index) {
        yvex_prompt_role role = messages[index].role;
        if (role < YVEX_PROMPT_ROLE_SYSTEM || role > YVEX_PROMPT_ROLE_TOOL ||
            (role == YVEX_PROMPT_ROLE_SYSTEM && index != 0u) ||
            (role == YVEX_PROMPT_ROLE_ASSISTANT && index && prior != YVEX_PROMPT_ROLE_USER &&
             prior != YVEX_PROMPT_ROLE_TOOL) ||
            (role == YVEX_PROMPT_ROLE_TOOL && prior != YVEX_PROMPT_ROLE_ASSISTANT &&
             prior != YVEX_PROMPT_ROLE_TOOL))
            return 0;
        prior = role;
    }
    if (options->add_generation_prompt &&
        prior != YVEX_PROMPT_ROLE_USER && prior != YVEX_PROMPT_ROLE_TOOL)
        return 0;
    return 1;
}

static int prompt_message_append(byte_builder *builder,
                                 const yvex_conversation_protocol *conversation,
                                 const yvex_prompt_message *message,
                                 yvex_prompt_role prior,
                                 yvex_error *err)
{
    const unsigned char *content;
    unsigned long long count;
    int rc;

    if (!message_span(message, &content, &count)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.prompt", "message content is absent or invalid UTF-8");
        return YVEX_ERR_FORMAT;
    }
    if (message->role == YVEX_PROMPT_ROLE_SYSTEM)
        return builder_append(builder, content, count, err);
    if (message->role == YVEX_PROMPT_ROLE_USER) {
        rc = builder_append(builder, conversation->user,
                            strlen(conversation->user), err);
        return rc == YVEX_OK ? builder_append(builder, content, count, err) : rc;
    }
    if (message->role == YVEX_PROMPT_ROLE_ASSISTANT) {
        rc = builder_append(builder, content, count, err);
        return rc == YVEX_OK
            ? builder_append(builder, conversation->eos,
                             strlen(conversation->eos), err)
            : rc;
    }
    if (prior == YVEX_PROMPT_ROLE_ASSISTANT) {
        rc = builder_append(builder, conversation->user,
                            strlen(conversation->user), err);
    } else {
        rc = builder_append(builder, "\n\n", 2u, err);
    }
    if (rc == YVEX_OK)
        rc = builder_append(builder, conversation->tool_result_start,
                            strlen(conversation->tool_result_start), err);
    if (rc == YVEX_OK)
        rc = builder_append(builder, content, count, err);
    return rc == YVEX_OK
        ? builder_append(builder, conversation->tool_result_end,
                         strlen(conversation->tool_result_end), err)
        : rc;
}

static int prompt_assistant_transition(byte_builder *builder,
                                       const yvex_conversation_protocol *conversation,
                                       const yvex_prompt_options *options,
                                       int final_user,
                                       yvex_error *err)
{
    int rc = builder_append(builder, conversation->assistant,
                            strlen(conversation->assistant), err);
    if (rc != YVEX_OK)
        return rc;
    if (options->mode == YVEX_PROMPT_MODE_THINKING &&
        (final_user || !options->drop_thinking))
        return builder_append(builder, conversation->thinking_start,
                              strlen(conversation->thinking_start), err);
    return builder_append(builder, conversation->thinking_end,
                          strlen(conversation->thinking_end), err);
}

static int fixture_prompt_render(yvex_rendered_prompt *out,
                                 const yvex_prompt_message *messages,
                                 unsigned long long message_count,
                                 const yvex_prompt_options *options,
                                 yvex_error *err)
{
    yvex_prompt_options defaults = {
        .add_generation_prompt = 1,
        .mode = YVEX_PROMPT_MODE_CHAT,
        .reasoning_policy = YVEX_REASONING_DISABLED};
    byte_builder builder = {0};
    unsigned long long index;
    int rc = YVEX_OK;
    if (!out || !messages || !message_count) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.prompt.fixture",
                       "fixture prompt messages are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!options)
        options = &defaults;
    memset(out, 0, sizeof(*out));
    for (index = 0u; index < message_count && rc == YVEX_OK; ++index) {
        const char *role = yvex_prompt_role_name(messages[index].role);
        if (!messages[index].content || strcmp(role, "unknown") == 0) {
            rc = YVEX_ERR_INVALID_ARG;
            break;
        }
        rc = builder_append(&builder, "<", 1u, err);
        if (rc == YVEX_OK) rc = builder_append(&builder, role, strlen(role), err);
        if (rc == YVEX_OK) rc = builder_append(&builder, ">\n", 2u, err);
        if (rc == YVEX_OK)
            rc = builder_append(&builder, messages[index].content,
                                strlen(messages[index].content), err);
        if (rc == YVEX_OK) rc = builder_append(&builder, "\n</", 3u, err);
        if (rc == YVEX_OK) rc = builder_append(&builder, role, strlen(role), err);
        if (rc == YVEX_OK) rc = builder_append(&builder, ">\n", 2u, err);
    }
    if (rc == YVEX_OK && options->add_generation_prompt)
        rc = builder_append(&builder, "<assistant>\n", 12u, err);
    if (rc != YVEX_OK) {
        free(builder.data);
        yvex_error_set(err, rc, "tokenizer.prompt.fixture",
                       "fixture prompt rendering failed");
        return rc;
    }
    out->text = (char *)builder.data;
    out->len = builder.count;
    out->generation_prompt = options->add_generation_prompt;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int prompt_identities(const yvex_tokenizer *tokenizer,
                             const yvex_prompt_message *messages,
                             unsigned long long message_count,
                             const yvex_prompt_options *options,
                             yvex_rendered_prompt *prompt)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.messages.v1") ||
        !yvex_sha256_update_u64_be(&hash, message_count))
        return 0;
    for (index = 0u; index < message_count; ++index) {
        const unsigned char *bytes;
        unsigned long long count;
        if (!message_span(&messages[index], &bytes, &count) ||
            !yvex_sha256_update_u64_be(&hash, messages[index].role) ||
            !yvex_sha256_update_u64_be(&hash, count) ||
            !yvex_sha256_update(&hash, bytes, (size_t)count))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, prompt->message_sequence_identity);
    if (!bytes_identity("yvex.tokenizer.rendered.v1", prompt->text,
                        (size_t)prompt->len, prompt->rendered_bytes_identity))
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.prompt.v1") ||
        !yvex_sha256_update_text(&hash, tokenizer->plan.tokenizer_plan_identity) ||
        !yvex_sha256_update_text(&hash, tokenizer->plan.prompt_policy_identity) ||
        !yvex_sha256_update_text(&hash, prompt->message_sequence_identity) ||
        !yvex_sha256_update_text(&hash, prompt->rendered_bytes_identity) ||
        !yvex_sha256_update_u64_be(&hash, options->add_bos) ||
        !yvex_sha256_update_u64_be(&hash, options->add_eos) ||
        !yvex_sha256_update_u64_be(&hash, options->add_generation_prompt) ||
        !yvex_sha256_update_u64_be(&hash, options->drop_thinking) ||
        !yvex_sha256_update_u64_be(&hash, options->mode) ||
        !yvex_sha256_update_u64_be(&hash, options->reasoning_policy) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, prompt->prompt_identity);
    return 1;
}

/*
 * Render admitted messages through the exact bounded source-authored conversation policy.
 *
 * Rollback.
 */
int yvex_prompt_render(yvex_rendered_prompt *out,
                       const yvex_tokenizer *tokenizer,
                       const yvex_prompt_message *messages,
                       unsigned long long message_count,
                       const yvex_prompt_options *options,
                       yvex_error *err)
{
    yvex_prompt_options defaults = {
        .add_bos = 1,
        .add_generation_prompt = 1,
        .drop_thinking = 1,
        .mode = YVEX_PROMPT_MODE_CHAT,
        .reasoning_policy = YVEX_REASONING_DISABLED};
    byte_builder builder = {0};
    yvex_rendered_prompt candidate;
    unsigned long long index;
    int rc = YVEX_OK;

    if (!tokenizer || !tokenizer->plan.sealed)
        return fixture_prompt_render(out, messages, message_count, options, err);
    if (tokenizer->plan.prompt_policy != YVEX_TOKENIZER_PROMPT_CONVERSATION) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "tokenizer.prompt",
                       "this tokenizer admits verbatim encoding rather than chat rendering");
        return YVEX_ERR_UNSUPPORTED;
    }
    memset(&candidate, 0, sizeof(candidate));
    if (!options)
        options = &defaults;
    if (!out || !tokenizer || !tokenizer->plan.sealed || !messages || !message_count ||
        options->mode > YVEX_PROMPT_MODE_THINKING ||
        options->reasoning_policy > YVEX_REASONING_MAXIMUM ||
        ((options->reasoning_policy == YVEX_REASONING_DISABLED) !=
         (options->mode == YVEX_PROMPT_MODE_CHAT)) ||
        (options->reasoning_policy != YVEX_REASONING_DISABLED &&
         !tokenizer->plan.explicit_reasoning_supported) ||
        (options->reasoning_policy == YVEX_REASONING_MAXIMUM &&
         !tokenizer->plan.maximum_reasoning_supported) ||
        !prompt_roles_valid(messages, message_count, options)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.prompt",
                       "sealed tokenizer, admitted reasoning policy, and valid ordered messages are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (options->add_bos)
        rc = builder_append(&builder, tokenizer->conversation->bos,
                            strlen(tokenizer->conversation->bos), err);
    if (rc == YVEX_OK && options->reasoning_policy == YVEX_REASONING_MAXIMUM)
        rc = builder_append(&builder,
                            tokenizer->conversation->reasoning_effort_max,
                            strlen(tokenizer->conversation->reasoning_effort_max),
                            err);
    for (index = 0u; index < message_count && rc == YVEX_OK; ++index) {
        yvex_prompt_role prior = index ? messages[index - 1u].role : YVEX_PROMPT_ROLE_SYSTEM;
        rc = prompt_message_append(&builder, tokenizer->conversation,
                                   &messages[index], prior, err);
        if (rc == YVEX_OK && messages[index].role == YVEX_PROMPT_ROLE_USER &&
            index + 1u < message_count &&
            messages[index + 1u].role == YVEX_PROMPT_ROLE_ASSISTANT)
            rc = prompt_assistant_transition(&builder, tokenizer->conversation,
                                             options, 0, err);
    }
    if (rc == YVEX_OK && options->add_generation_prompt)
        rc = prompt_assistant_transition(&builder, tokenizer->conversation,
                                         options, 1, err);
    if (rc == YVEX_OK && options->add_eos)
        rc = builder_append(&builder, tokenizer->conversation->eos,
                            strlen(tokenizer->conversation->eos), err);
    if (rc != YVEX_OK) {
        free(builder.data);
        memset(out, 0, sizeof(*out));
        return rc;
    }
    candidate.text = (char *)builder.data;
    candidate.len = builder.count;
    candidate.generation_prompt = options->add_generation_prompt;
    if (!prompt_identities(tokenizer, messages, message_count, options, &candidate)) {
        yvex_rendered_prompt_free(&candidate);
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.prompt.identity", "prompt identity derivation failed");
        return YVEX_ERR_STATE;
    }
    *out = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Render and encode through one shared tokenizer path without a second formatting owner.
 *
 * Rollback.
 */
int yvex_tokenizer_encode_prompt(const yvex_tokenizer *tokenizer,
                                 const yvex_prompt_message *messages,
                                 unsigned long long message_count,
                                 const yvex_prompt_options *prompt_options,
                                 const yvex_tokenizer_encode_options *encode_options,
                                 yvex_rendered_prompt *rendered,
                                 yvex_tokenizer_encode_result *encoded,
                                 yvex_error *err)
{
    yvex_tokenizer_encode_options exact = {0, 0, 1, 0};
    int rc;
    if (!rendered || !encoded)
        return YVEX_ERR_INVALID_ARG;
    memset(rendered, 0, sizeof(*rendered));
    memset(encoded, 0, sizeof(*encoded));
    rc = yvex_prompt_render(rendered, tokenizer, messages, message_count,
                            prompt_options, err);
    if (rc != YVEX_OK)
        return rc;
    if (encode_options) {
        exact = *encode_options;
        exact.add_bos = 0;
        exact.add_eos = 0;
        exact.allow_special_tokens = 1;
    }
    rc = yvex_tokenizer_encode(tokenizer, (const unsigned char *)rendered->text,
                               rendered->len, &exact, encoded, err);
    if (rc != YVEX_OK)
        yvex_rendered_prompt_free(rendered);
    return rc;
}

void yvex_rendered_prompt_free(yvex_rendered_prompt *prompt)
{
    if (!prompt)
        return;
    free(prompt->text);
    memset(prompt, 0, sizeof(*prompt));
}

static int sequence_identity(const yvex_token_sequence *sequence,
                             yvex_token_sequence_summary *summary)
{
    yvex_sha256 ids_hash, state_hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&ids_hash);
    yvex_sha256_init(&state_hash);
    if (!yvex_sha256_update_text(&ids_hash, "yvex.tokenizer.append.ids.v1") ||
        !yvex_sha256_update_text(&state_hash, "yvex.tokenizer.append.state.v1") ||
        !yvex_sha256_update_u64_be(&ids_hash, sequence->count) ||
        !yvex_sha256_update_u64_be(&state_hash, sequence->count) ||
        !yvex_sha256_update_u64_be(&state_hash, sequence->generation))
        return 0;
    for (index = 0u; index < sequence->count; ++index)
        if (!yvex_sha256_update_u64_be(&ids_hash, sequence->rows[index].token_id) ||
            !yvex_sha256_update_u64_be(&state_hash, sequence->rows[index].token_id) ||
            !yvex_sha256_update_u64_be(&state_hash, sequence->rows[index].state))
            return 0;
    if (!yvex_sha256_final(&ids_hash, digest))
        return 0;
    yvex_sha256_hex(digest, summary->token_ids_identity);
    if (!yvex_sha256_final(&state_hash, digest))
        return 0;
    yvex_sha256_hex(digest, summary->state_identity);
    return 1;
}

int yvex_token_sequence_open(yvex_token_sequence **out,
                             unsigned long long capacity,
                             yvex_error *err)
{
    yvex_token_sequence *sequence;
    if (!out || !capacity || capacity > SIZE_MAX / sizeof(token_sequence_row)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.append.open", "bounded nonzero capacity is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    sequence = calloc(1u, sizeof(*sequence));
    if (sequence)
        sequence->rows = calloc((size_t)capacity, sizeof(*sequence->rows));
    if (!sequence || !sequence->rows) {
        free(sequence ? sequence->rows : NULL);
        free(sequence);
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.append.open", "token directory allocation failed");
        return YVEX_ERR_NOMEM;
    }
    sequence->capacity = capacity;
    *out = sequence;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_token_sequence_append(yvex_token_sequence *sequence,
                               unsigned int token_id,
                               unsigned long long vocabulary_size,
                               unsigned long long *ordinal,
                               yvex_error *err)
{
    unsigned long long next_count, next_generation;
    if (!sequence || !ordinal || sequence->transaction_active ||
        token_id >= vocabulary_size || sequence->count >= sequence->capacity) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.append", "token ID or directory capacity is invalid");
        return YVEX_ERR_BOUNDS;
    }
    if (!yvex_core_u64_add(sequence->count, 1u, &next_count) ||
        !yvex_core_u64_add(sequence->generation, 1u, &next_generation)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.append", "token directory counter overflow");
        return YVEX_ERR_BOUNDS;
    }
    *ordinal = sequence->count;
    sequence->rows[sequence->count].token_id = token_id;
    sequence->rows[sequence->count].state = YVEX_TOKEN_APPEND_PROPOSED;
    sequence->count = next_count;
    sequence->generation = next_generation;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_token_sequence_transaction_begin(
    yvex_token_sequence *sequence, unsigned long long maximum_rows,
    yvex_token_sequence_transaction **out, yvex_error *err)
{
    yvex_token_sequence_transaction *transaction;
    if (out) *out = NULL;
    if (!sequence || !out || !maximum_rows || sequence->transaction_active ||
        maximum_rows > sequence->capacity - sequence->count ||
        maximum_rows > SIZE_MAX / sizeof(token_sequence_row)) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.append.transaction.begin",
                       "token sequence cannot admit the requested transaction");
        return YVEX_ERR_STATE;
    }
    transaction = calloc(1u, sizeof(*transaction));
    if (transaction)
        transaction->rows = calloc((size_t)maximum_rows,
                                   sizeof(*transaction->rows));
    if (!transaction || !transaction->rows) {
        free(transaction ? transaction->rows : NULL);
        free(transaction);
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.append.transaction.begin",
                       "token transaction allocation failed");
        return YVEX_ERR_NOMEM;
    }
    transaction->sequence = sequence;
    transaction->base_count = sequence->count;
    transaction->base_generation = sequence->generation;
    transaction->generation = sequence->generation;
    transaction->capacity = maximum_rows;
    sequence->transaction_active = 1;
    *out = transaction;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_token_sequence_transaction_append(
    yvex_token_sequence_transaction *transaction, unsigned int token_id,
    unsigned long long vocabulary_size, unsigned long long *ordinal,
    yvex_error *err)
{
    unsigned long long next_generation;
    if (!transaction || transaction->prepared || !ordinal ||
        token_id >= vocabulary_size || transaction->count >= transaction->capacity ||
        !yvex_core_u64_add(transaction->generation, 1ull, &next_generation)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.append.transaction.append",
                       "staged token or transaction extent is invalid");
        return YVEX_ERR_BOUNDS;
    }
    *ordinal = transaction->base_count + transaction->count;
    transaction->rows[transaction->count].token_id = token_id;
    transaction->rows[transaction->count].state = YVEX_TOKEN_APPEND_PROPOSED;
    transaction->count++;
    transaction->generation = next_generation;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_token_sequence_transaction_transition(
    yvex_token_sequence_transaction *transaction, unsigned long long ordinal,
    yvex_token_append_state expected, yvex_token_append_state next,
    yvex_error *err)
{
    unsigned long long local, next_generation;
    if (!transaction || transaction->prepared || ordinal < transaction->base_count) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.append.transaction.transition",
                       "staged append transition is stale");
        return YVEX_ERR_STATE;
    }
    local = ordinal - transaction->base_count;
    if (local >= transaction->count || transaction->rows[local].state != expected ||
        next != (yvex_token_append_state)(expected + 1) ||
        !yvex_core_u64_add(transaction->generation, 1ull, &next_generation)) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.append.transaction.transition",
                       "staged append transition is non-contiguous");
        return YVEX_ERR_STATE;
    }
    transaction->rows[local].state = next;
    transaction->generation = next_generation;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_token_sequence_transaction_prepare(
    yvex_token_sequence_transaction *transaction, yvex_error *err)
{
    yvex_token_sequence *sequence = transaction ? transaction->sequence : NULL;
    if (!transaction || transaction->prepared || !transaction->count || !sequence ||
        !sequence->transaction_active || sequence->count != transaction->base_count ||
        sequence->generation != transaction->base_generation ||
        transaction->count > sequence->capacity - sequence->count) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.append.transaction.prepare",
                       "token transaction no longer matches its base state");
        return YVEX_ERR_STATE;
    }
    transaction->prepared = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_token_sequence_transaction_publish(
    yvex_token_sequence_transaction **transaction)
{
    yvex_token_sequence_transaction *owner = transaction ? *transaction : NULL;
    yvex_token_sequence *sequence;
    if (!owner || !owner->prepared) return;
    sequence = owner->sequence;
    memcpy(sequence->rows + owner->base_count, owner->rows,
           (size_t)owner->count * sizeof(*owner->rows));
    sequence->count = owner->base_count + owner->count;
    sequence->generation = owner->generation;
    sequence->transaction_active = 0;
    free(owner->rows);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *transaction = NULL;
}

void yvex_token_sequence_transaction_abort(
    yvex_token_sequence_transaction **transaction)
{
    yvex_token_sequence_transaction *owner = transaction ? *transaction : NULL;
    if (!owner) return;
    if (owner->sequence) owner->sequence->transaction_active = 0;
    free(owner->rows);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *transaction = NULL;
}

int yvex_token_sequence_transition(yvex_token_sequence *sequence,
                                   unsigned long long ordinal,
                                   yvex_token_append_state expected,
                                   yvex_token_append_state next,
                                   yvex_error *err)
{
    unsigned long long next_generation;
    if (!sequence || sequence->transaction_active || ordinal >= sequence->count ||
        sequence->rows[ordinal].state != expected ||
        next != (yvex_token_append_state)(expected + 1)) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.append.transition",
                       "append transition is non-contiguous or stale");
        return YVEX_ERR_STATE;
    }
    if (!yvex_core_u64_add(sequence->generation, 1u, &next_generation)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.append.transition",
                       "token directory generation overflow");
        return YVEX_ERR_BOUNDS;
    }
    sequence->rows[ordinal].state = next;
    sequence->generation = next_generation;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_token_sequence_summary_get(const yvex_token_sequence *sequence,
                                    yvex_token_sequence_summary *summary,
                                    yvex_error *err)
{
    if (!sequence || !summary) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.append.summary", "sequence and summary are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(summary, 0, sizeof(*summary));
    summary->schema_version = YVEX_TOKENIZER_APPEND_SCHEMA_V1;
    summary->count = sequence->count;
    summary->capacity = sequence->capacity;
    summary->generation = sequence->generation;
    if (!sequence_identity(sequence, summary)) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.append.summary", "append identity derivation failed");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Reuse one generation-local token directory for a later turn without reallocating it.
 *
 * Generation overflow preserves prior rows.
 */
int yvex_token_sequence_reset(yvex_token_sequence *sequence,
                              yvex_error *err)
{
    unsigned long long next_generation;

    if (!sequence || sequence->transaction_active) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.append.reset",
                       "token directory is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_core_u64_add(sequence->generation, 1u, &next_generation)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.append.reset",
                       "token directory generation overflow");
        return YVEX_ERR_BOUNDS;
    }
    memset(sequence->rows, 0,
           (size_t)sequence->capacity * sizeof(*sequence->rows));
    sequence->count = 0u;
    sequence->generation = next_generation;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Release one token directory deterministically and clear caller ownership. */
void yvex_token_sequence_close(yvex_token_sequence **sequence)
{
    if (!sequence || !*sequence)
        return;
    free((*sequence)->rows);
    memset(*sequence, 0, sizeof(**sequence));
    free(*sequence);
    *sequence = NULL;
}
