/*
 * Construct deterministic GGUF tokenizer metadata from verified JSON sidecars.
 *
 * JSON depth is capped, decoded strings share one owned arena, every expected token ID is present
 * once, and provider identities revalidate. This module exposes immutable artifact metadata only.
 */
#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf.h>

#define TOKENIZER_JSON_LIMIT (32u * 1024u * 1024u)
#define TOKENIZER_CONFIG_LIMIT (1024u * 1024u)
#define TOKENIZER_JSON_MAX_DEPTH 64u
#define TOKENIZER_TYPE_NORMAL 1
#define TOKENIZER_TYPE_CONTROL 3
#define TOKENIZER_TYPE_USER_DEFINED 4

typedef struct {
    size_t offset;
    size_t length;
    int present;
} tokenizer_string_ref;

typedef struct {
    unsigned long long id;
    tokenizer_string_ref content;
    int special;
} tokenizer_added;

typedef struct {
    const unsigned char *cursor;
    const unsigned char *end;
} tokenizer_json;

typedef yvex_core_bytes tokenizer_arena;

struct yvex_gguf_tokenizer_metadata {
    yvex_gguf_tokenizer_summary summary;
    yvex_source_metadata_blob tokenizer_json;
    yvex_source_metadata_blob tokenizer_config;
    tokenizer_arena arena;
    tokenizer_string_ref *tokens;
    int *token_types;
    tokenizer_string_ref *merges;
    size_t merge_capacity;
    tokenizer_added *added;
    size_t added_count;
    size_t added_capacity;
    tokenizer_string_ref config_bos;
    tokenizer_string_ref config_eos;
    tokenizer_string_ref config_pad;
};

static int tokenizer_fail(yvex_gguf_tokenizer_failure *failure, yvex_gguf_tokenizer_code code,
                          const char *field, unsigned long long index, unsigned long long expected,
                          unsigned long long actual, yvex_error *err, yvex_status status,
                          const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->record_index = index;
        failure->expected = expected;
        failure->actual = actual;
        if (field)
            yvex_core_text_copy(failure->field, sizeof(failure->field), field);
    }
    yvex_error_set(err, status, "gguf.tokenizer.metadata", message);
    return status;
}

static void tokenizer_json_space(tokenizer_json *json) {
    while (json->cursor < json->end && isspace((unsigned char)*json->cursor))
        json->cursor++;
}

static int tokenizer_json_string_raw(tokenizer_json *json, const unsigned char **begin,
                                     const unsigned char **end) {
    const unsigned char *start;

    tokenizer_json_space(json);
    if (json->cursor >= json->end || *json->cursor++ != '"')
        return 0;
    start = json->cursor;
    while (json->cursor < json->end) {
        unsigned char ch = *json->cursor++;
        if (ch == '"') {
            if (begin)
                *begin = start;
            if (end)
                *end = json->cursor - 1u;
            return 1;
        }
        if (ch < 0x20u)
            return 0;
        if (ch == '\\') {
            if (json->cursor >= json->end)
                return 0;
            ch = *json->cursor++;
            if (ch == 'u') {
                unsigned int index;
                for (index = 0u; index < 4u; ++index)
                    if (json->cursor >= json->end || !isxdigit((unsigned char)*json->cursor++))
                        return 0;
            } else if (!strchr("\"\\/bfnrt", ch)) {
                return 0;
            }
        }
    }
    return 0;
}

static int tokenizer_hex4(const unsigned char *text, uint32_t *value) {
    uint32_t result = 0u;
    unsigned int index;

    for (index = 0u; index < 4u; ++index) {
        unsigned char ch = text[index];
        unsigned int digit;
        if (ch >= '0' && ch <= '9')
            digit = ch - '0';
        else if (ch >= 'a' && ch <= 'f')
            digit = ch - 'a' + 10u;
        else if (ch >= 'A' && ch <= 'F')
            digit = ch - 'A' + 10u;
        else
            return 0;
        result = (result << 4u) | digit;
    }
    *value = result;
    return 1;
}

static int tokenizer_arena_byte(tokenizer_arena *arena, unsigned char byte) {
    return yvex_core_bytes_append(arena, &byte, 1u);
}

static int tokenizer_arena_utf8(tokenizer_arena *arena, uint32_t point) {
    if (point <= 0x7fu)
        return tokenizer_arena_byte(arena, (unsigned char)point);
    if (point <= 0x7ffu)
        return tokenizer_arena_byte(arena, (unsigned char)(0xc0u | (point >> 6u))) &&
               tokenizer_arena_byte(arena, (unsigned char)(0x80u | (point & 0x3fu)));
    if (point <= 0xffffu)
        return tokenizer_arena_byte(arena, (unsigned char)(0xe0u | (point >> 12u))) &&
               tokenizer_arena_byte(arena, (unsigned char)(0x80u | ((point >> 6u) & 0x3fu))) &&
               tokenizer_arena_byte(arena, (unsigned char)(0x80u | (point & 0x3fu)));
    if (point <= 0x10ffffu)
        return tokenizer_arena_byte(arena, (unsigned char)(0xf0u | (point >> 18u))) &&
               tokenizer_arena_byte(arena, (unsigned char)(0x80u | ((point >> 12u) & 0x3fu))) &&
               tokenizer_arena_byte(arena, (unsigned char)(0x80u | ((point >> 6u) & 0x3fu))) &&
               tokenizer_arena_byte(arena, (unsigned char)(0x80u | (point & 0x3fu)));
    return 0;
}

static int tokenizer_decode_string(tokenizer_arena *arena, const unsigned char *begin,
                                   const unsigned char *end, tokenizer_string_ref *out) {
    const unsigned char *cursor = begin;
    size_t start;

    if (!arena || !begin || !end || end < begin || !out)
        return 0;
    start = arena->count;
    while (cursor < end) {
        unsigned char ch = *cursor++;
        if (ch != '\\') {
            if (!tokenizer_arena_byte(arena, ch))
                return 0;
            continue;
        }
        if (cursor >= end)
            return 0;
        ch = *cursor++;
        if (ch == '"' || ch == '\\' || ch == '/') {
            if (!tokenizer_arena_byte(arena, ch))
                return 0;
        } else if (ch == 'b') {
            if (!tokenizer_arena_byte(arena, '\b'))
                return 0;
        } else if (ch == 'f') {
            if (!tokenizer_arena_byte(arena, '\f'))
                return 0;
        } else if (ch == 'n') {
            if (!tokenizer_arena_byte(arena, '\n'))
                return 0;
        } else if (ch == 'r') {
            if (!tokenizer_arena_byte(arena, '\r'))
                return 0;
        } else if (ch == 't') {
            if (!tokenizer_arena_byte(arena, '\t'))
                return 0;
        } else if (ch == 'u') {
            uint32_t first;
            uint32_t point;
            if ((size_t)(end - cursor) < 4u || !tokenizer_hex4(cursor, &first))
                return 0;
            cursor += 4u;
            point = first;
            if (first >= 0xd800u && first <= 0xdbffu) {
                uint32_t second;
                if ((size_t)(end - cursor) < 6u || cursor[0] != '\\' || cursor[1] != 'u' ||
                    !tokenizer_hex4(cursor + 2u, &second) || second < 0xdc00u || second > 0xdfffu)
                    return 0;
                cursor += 6u;
                point = 0x10000u + ((first - 0xd800u) << 10u) + (second - 0xdc00u);
            } else if (first >= 0xdc00u && first <= 0xdfffu) {
                return 0;
            }
            if (!tokenizer_arena_utf8(arena, point))
                return 0;
        } else {
            return 0;
        }
    }
    out->offset = start;
    out->length = arena->count - start;
    out->present = 1;
    return 1;
}

static int tokenizer_string_key(tokenizer_json *json, char *out, size_t capacity) {
    const unsigned char *begin;
    const unsigned char *end;
    tokenizer_arena arena;
    tokenizer_string_ref ref;
    int ok;

    memset(&arena, 0, sizeof(arena));
    arena.maximum = capacity ? capacity - 1u : 0u;
    arena.initial_capacity = 4096u;
    if (!tokenizer_json_string_raw(json, &begin, &end) || !capacity)
        return 0;
    ok = tokenizer_decode_string(&arena, begin, end, &ref);
    if (ok && ref.length < capacity) {
        memcpy(out, arena.data + ref.offset, ref.length);
        out[ref.length] = '\0';
    } else {
        ok = 0;
    }
    free(arena.data);
    return ok;
}

static int tokenizer_json_literal(tokenizer_json *json, const char *literal) {
    size_t length = strlen(literal);
    tokenizer_json_space(json);
    if ((size_t)(json->end - json->cursor) < length || memcmp(json->cursor, literal, length) != 0)
        return 0;
    json->cursor += length;
    return 1;
}

static int tokenizer_json_u64(tokenizer_json *json, unsigned long long *out) {
    yvex_json core;
    int accepted;

    yvex_json_init(&core, (const char *)json->cursor, (size_t)(json->end - json->cursor));
    accepted = yvex_json_u64(&core, out);
    if (accepted)
        json->cursor = (const unsigned char *)core.cursor;
    return accepted;
}

static int tokenizer_json_skip(tokenizer_json *json, unsigned int depth) {
    yvex_json core;
    int accepted;

    if (depth > TOKENIZER_JSON_MAX_DEPTH)
        return 0;
    yvex_json_init(&core, (const char *)json->cursor, (size_t)(json->end - json->cursor));
    core.depth = depth;
    accepted = yvex_json_skip_value(&core);
    if (accepted)
        json->cursor = (const unsigned char *)core.cursor;
    return accepted;
}

static int tokenizer_ref_equal(const yvex_gguf_tokenizer_metadata *metadata,
                               tokenizer_string_ref left, tokenizer_string_ref right) {
    return left.present && right.present && left.length == right.length &&
           memcmp(metadata->arena.data + left.offset, metadata->arena.data + right.offset,
                  left.length) == 0;
}

static int tokenizer_merges_grow(yvex_gguf_tokenizer_metadata *metadata) {
    size_t capacity = metadata->merge_capacity ? metadata->merge_capacity * 2u : 1024u;
    tokenizer_string_ref *grown;

    if (capacity < metadata->merge_capacity || capacity > SIZE_MAX / sizeof(*grown))
        return 0;
    grown = (tokenizer_string_ref *)realloc(metadata->merges, capacity * sizeof(*grown));
    if (!grown)
        return 0;
    metadata->merges = grown;
    metadata->merge_capacity = capacity;
    return 1;
}

static int tokenizer_added_grow(yvex_gguf_tokenizer_metadata *metadata) {
    size_t capacity = metadata->added_capacity ? metadata->added_capacity * 2u : 128u;
    tokenizer_added *grown;

    if (capacity < metadata->added_capacity || capacity > SIZE_MAX / sizeof(*grown))
        return 0;
    grown = (tokenizer_added *)realloc(metadata->added, capacity * sizeof(*grown));
    if (!grown)
        return 0;
    metadata->added = grown;
    metadata->added_capacity = capacity;
    return 1;
}

static int tokenizer_parse_vocab(yvex_gguf_tokenizer_metadata *metadata, tokenizer_json *json,
                                 yvex_gguf_tokenizer_failure *failure, yvex_error *err) {
    tokenizer_string_ref token;
    const unsigned char *begin;
    const unsigned char *end;
    unsigned long long id;

    tokenizer_json_space(json);
    if (json->cursor >= json->end || *json->cursor++ != '{')
        return 0;
    for (;;) {
        tokenizer_json_space(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            return 1;
        }
        if (!tokenizer_json_string_raw(json, &begin, &end) ||
            !tokenizer_decode_string(&metadata->arena, begin, end, &token))
            return 0;
        tokenizer_json_space(json);
        if (json->cursor >= json->end || *json->cursor++ != ':' || !tokenizer_json_u64(json, &id))
            return 0;
        if (id >= metadata->summary.token_count)
            return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_CARDINALITY, "model.vocab", id,
                                  metadata->summary.token_count, id + 1u, err, YVEX_ERR_BOUNDS,
                                  "token id exceeds expected vocabulary");
        if (metadata->tokens[id].present)
            return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_DUPLICATE_TOKEN_ID, "model.vocab",
                                  id, 1u, 2u, err, YVEX_ERR_FORMAT,
                                  "duplicate tokenizer vocabulary id");
        metadata->tokens[id] = token;
        metadata->token_types[id] = TOKENIZER_TYPE_NORMAL;
        tokenizer_json_space(json);
        if (json->cursor >= json->end)
            return 0;
        if (*json->cursor == '}')
            continue;
        if (*json->cursor++ != ',')
            return 0;
    }
}

static int tokenizer_parse_merges(yvex_gguf_tokenizer_metadata *metadata, tokenizer_json *json) {
    tokenizer_string_ref merge;
    const unsigned char *begin;
    const unsigned char *end;

    tokenizer_json_space(json);
    if (json->cursor >= json->end || *json->cursor++ != '[')
        return 0;
    for (;;) {
        tokenizer_json_space(json);
        if (json->cursor < json->end && *json->cursor == ']') {
            json->cursor++;
            return metadata->summary.merge_count != 0u;
        }
        if (*json->cursor != '"' || !tokenizer_json_string_raw(json, &begin, &end) ||
            !tokenizer_decode_string(&metadata->arena, begin, end, &merge))
            return 0;
        if (metadata->summary.merge_count >= metadata->merge_capacity &&
            !tokenizer_merges_grow(metadata))
            return 0;
        metadata->merges[metadata->summary.merge_count++] = merge;
        tokenizer_json_space(json);
        if (json->cursor >= json->end)
            return 0;
        if (*json->cursor == ']')
            continue;
        if (*json->cursor++ != ',')
            return 0;
    }
}

static int tokenizer_parse_model(yvex_gguf_tokenizer_metadata *metadata, tokenizer_json *json,
                                 yvex_gguf_tokenizer_failure *failure, yvex_error *err) {
    char key[64];
    char model_type[32] = "";
    unsigned int seen = 0u;

    tokenizer_json_space(json);
    if (json->cursor >= json->end || *json->cursor++ != '{')
        return 0;
    for (;;) {
        tokenizer_json_space(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            return (seen & 7u) == 7u && strcmp(model_type, "BPE") == 0;
        }
        if (!tokenizer_string_key(json, key, sizeof(key)))
            return 0;
        tokenizer_json_space(json);
        if (json->cursor >= json->end || *json->cursor++ != ':')
            return 0;
        if (strcmp(key, "type") == 0) {
            if ((seen & 1u) || !tokenizer_string_key(json, model_type, sizeof(model_type)))
                return 0;
            seen |= 1u;
        } else if (strcmp(key, "vocab") == 0) {
            if ((seen & 2u) || !tokenizer_parse_vocab(metadata, json, failure, err))
                return 0;
            seen |= 2u;
        } else if (strcmp(key, "merges") == 0) {
            if ((seen & 4u) || !tokenizer_parse_merges(metadata, json))
                return 0;
            seen |= 4u;
        } else if (!tokenizer_json_skip(json, 1u)) {
            return 0;
        }
        tokenizer_json_space(json);
        if (json->cursor >= json->end)
            return 0;
        if (*json->cursor == '}')
            continue;
        if (*json->cursor++ != ',')
            return 0;
    }
}

static int tokenizer_parse_added_row(yvex_gguf_tokenizer_metadata *metadata, tokenizer_json *json) {
    tokenizer_added row;
    char key[64];
    unsigned int seen = 0u;

    memset(&row, 0, sizeof(row));
    tokenizer_json_space(json);
    if (json->cursor >= json->end || *json->cursor++ != '{')
        return 0;
    for (;;) {
        tokenizer_json_space(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            if ((seen & 7u) != 7u)
                return 0;
            if (metadata->added_count >= metadata->added_capacity &&
                !tokenizer_added_grow(metadata))
                return 0;
            metadata->added[metadata->added_count++] = row;
            return 1;
        }
        if (!tokenizer_string_key(json, key, sizeof(key)))
            return 0;
        tokenizer_json_space(json);
        if (json->cursor >= json->end || *json->cursor++ != ':')
            return 0;
        if (strcmp(key, "id") == 0) {
            if ((seen & 1u) || !tokenizer_json_u64(json, &row.id))
                return 0;
            seen |= 1u;
        } else if (strcmp(key, "content") == 0) {
            const unsigned char *begin;
            const unsigned char *end;
            if ((seen & 2u) || !tokenizer_json_string_raw(json, &begin, &end) ||
                !tokenizer_decode_string(&metadata->arena, begin, end, &row.content))
                return 0;
            seen |= 2u;
        } else if (strcmp(key, "special") == 0) {
            if (seen & 4u)
                return 0;
            if (tokenizer_json_literal(json, "true"))
                row.special = 1;
            else if (tokenizer_json_literal(json, "false"))
                row.special = 0;
            else
                return 0;
            seen |= 4u;
        } else if (!tokenizer_json_skip(json, 1u)) {
            return 0;
        }
        tokenizer_json_space(json);
        if (json->cursor >= json->end)
            return 0;
        if (*json->cursor == '}')
            continue;
        if (*json->cursor++ != ',')
            return 0;
    }
}

static int tokenizer_parse_added(yvex_gguf_tokenizer_metadata *metadata, tokenizer_json *json) {
    tokenizer_json_space(json);
    if (json->cursor >= json->end || *json->cursor++ != '[')
        return 0;
    for (;;) {
        tokenizer_json_space(json);
        if (json->cursor < json->end && *json->cursor == ']') {
            json->cursor++;
            return 1;
        }
        if (!tokenizer_parse_added_row(metadata, json))
            return 0;
        tokenizer_json_space(json);
        if (json->cursor >= json->end)
            return 0;
        if (*json->cursor == ']')
            continue;
        if (*json->cursor++ != ',')
            return 0;
    }
}

static int tokenizer_apply_added(yvex_gguf_tokenizer_metadata *metadata,
                                 yvex_gguf_tokenizer_failure *failure, yvex_error *err) {
    size_t index;

    for (index = 0u; index < metadata->added_count; ++index) {
        tokenizer_added *row = &metadata->added[index];
        if (row->id >= metadata->summary.token_count)
            return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_CARDINALITY, "added_tokens", index,
                                  metadata->summary.token_count, row->id + 1u, err, YVEX_ERR_BOUNDS,
                                  "added token id exceeds expected vocabulary");
        if (metadata->tokens[row->id].present) {
            if (!tokenizer_ref_equal(metadata, metadata->tokens[row->id], row->content))
                return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_TOKEN_MISMATCH, "added_tokens",
                                      row->id, 1u, 0u, err, YVEX_ERR_FORMAT,
                                      "added token content differs from vocabulary token");
        } else {
            metadata->tokens[row->id] = row->content;
        }
        metadata->token_types[row->id] =
            row->special ? TOKENIZER_TYPE_CONTROL : TOKENIZER_TYPE_USER_DEFINED;
    }
    metadata->summary.added_token_count = metadata->added_count;
    return YVEX_OK;
}

static int tokenizer_parse_json(yvex_gguf_tokenizer_metadata *metadata,
                                yvex_gguf_tokenizer_failure *failure, yvex_error *err) {
    tokenizer_json json;
    char key[64];
    unsigned int seen = 0u;
    unsigned long long index;
    int rc;

    json.cursor = metadata->tokenizer_json.bytes;
    json.end = json.cursor + metadata->tokenizer_json.byte_count;
    tokenizer_json_space(&json);
    if (json.cursor >= json.end || *json.cursor++ != '{')
        goto malformed;
    for (;;) {
        tokenizer_json_space(&json);
        if (json.cursor < json.end && *json.cursor == '}') {
            json.cursor++;
            break;
        }
        if (!tokenizer_string_key(&json, key, sizeof(key)))
            goto malformed;
        tokenizer_json_space(&json);
        if (json.cursor >= json.end || *json.cursor++ != ':')
            goto malformed;
        if (strcmp(key, "added_tokens") == 0) {
            if ((seen & 1u) || !tokenizer_parse_added(metadata, &json))
                goto malformed;
            seen |= 1u;
        } else if (strcmp(key, "model") == 0) {
            if ((seen & 2u) || !tokenizer_parse_model(metadata, &json, failure, err)) {
                if (failure && failure->code != YVEX_GGUF_TOKENIZER_OK)
                    return yvex_error_code(err);
                goto malformed;
            }
            seen |= 2u;
        } else if (!tokenizer_json_skip(&json, 1u)) {
            goto malformed;
        }
        tokenizer_json_space(&json);
        if (json.cursor >= json.end)
            goto malformed;
        if (*json.cursor == '}')
            continue;
        if (*json.cursor++ != ',')
            goto malformed;
    }
    tokenizer_json_space(&json);
    if (json.cursor != json.end || (seen & 3u) != 3u)
        goto malformed;
    rc = tokenizer_apply_added(metadata, failure, err);
    if (rc != YVEX_OK)
        return rc;
    for (index = 0u; index < metadata->summary.token_count; ++index)
        if (!metadata->tokens[index].present)
            return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_MISSING_TOKEN_ID, "model.vocab",
                                  index, 1u, 0u, err, YVEX_ERR_FORMAT,
                                  "tokenizer vocabulary has a missing token id");
    return YVEX_OK;
malformed:
    return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_MALFORMED_JSON, "tokenizer.json",
                          (unsigned long long)(json.cursor - metadata->tokenizer_json.bytes), 1u,
                          0u, err, YVEX_ERR_FORMAT,
                          "tokenizer JSON is malformed or lacks the required BPE material");
}

static int tokenizer_config_special(yvex_gguf_tokenizer_metadata *metadata, tokenizer_json *json,
                                    tokenizer_string_ref *out) {
    char key[64];

    tokenizer_json_space(json);
    if (json->cursor < json->end && *json->cursor == '"') {
        const unsigned char *begin;
        const unsigned char *end;
        return tokenizer_json_string_raw(json, &begin, &end) &&
               tokenizer_decode_string(&metadata->arena, begin, end, out);
    }
    if (json->cursor >= json->end || *json->cursor++ != '{')
        return 0;
    for (;;) {
        tokenizer_json_space(json);
        if (json->cursor < json->end && *json->cursor == '}') {
            json->cursor++;
            return out->present;
        }
        if (!tokenizer_string_key(json, key, sizeof(key)))
            return 0;
        tokenizer_json_space(json);
        if (json->cursor >= json->end || *json->cursor++ != ':')
            return 0;
        if (strcmp(key, "content") == 0) {
            const unsigned char *begin;
            const unsigned char *end;
            if (out->present || !tokenizer_json_string_raw(json, &begin, &end) ||
                !tokenizer_decode_string(&metadata->arena, begin, end, out))
                return 0;
        } else if (!tokenizer_json_skip(json, 1u)) {
            return 0;
        }
        tokenizer_json_space(json);
        if (json->cursor >= json->end)
            return 0;
        if (*json->cursor == '}')
            continue;
        if (*json->cursor++ != ',')
            return 0;
    }
}

static int tokenizer_config_bool(tokenizer_json *json, int *out) {
    if (tokenizer_json_literal(json, "true")) {
        *out = 1;
        return 1;
    }
    if (tokenizer_json_literal(json, "false")) {
        *out = 0;
        return 1;
    }
    return 0;
}

static int tokenizer_parse_config(yvex_gguf_tokenizer_metadata *metadata,
                                  yvex_gguf_tokenizer_failure *failure, yvex_error *err) {
    tokenizer_json json;
    char key[64];
    unsigned int seen = 0u;

    json.cursor = metadata->tokenizer_config.bytes;
    json.end = json.cursor + metadata->tokenizer_config.byte_count;
    tokenizer_json_space(&json);
    if (json.cursor >= json.end || *json.cursor++ != '{')
        goto malformed;
    for (;;) {
        tokenizer_json_space(&json);
        if (json.cursor < json.end && *json.cursor == '}') {
            json.cursor++;
            break;
        }
        if (!tokenizer_string_key(&json, key, sizeof(key)))
            goto malformed;
        tokenizer_json_space(&json);
        if (json.cursor >= json.end || *json.cursor++ != ':')
            goto malformed;
        if (strcmp(key, "add_bos_token") == 0) {
            if ((seen & 1u) || !tokenizer_config_bool(&json, &metadata->summary.add_bos_token))
                goto malformed;
            seen |= 1u;
        } else if (strcmp(key, "add_eos_token") == 0) {
            if ((seen & 2u) || !tokenizer_config_bool(&json, &metadata->summary.add_eos_token))
                goto malformed;
            seen |= 2u;
        } else if (strcmp(key, "bos_token") == 0) {
            if ((seen & 4u) || !tokenizer_config_special(metadata, &json, &metadata->config_bos))
                goto malformed;
            seen |= 4u;
        } else if (strcmp(key, "eos_token") == 0) {
            if ((seen & 8u) || !tokenizer_config_special(metadata, &json, &metadata->config_eos))
                goto malformed;
            seen |= 8u;
        } else if (strcmp(key, "pad_token") == 0) {
            if ((seen & 16u) || !tokenizer_config_special(metadata, &json, &metadata->config_pad))
                goto malformed;
            seen |= 16u;
        } else if (strcmp(key, "chat_template") == 0) {
            if (seen & 32u)
                goto malformed;
            tokenizer_json_space(&json);
            if (json.cursor >= json.end)
                goto malformed;
            if (*json.cursor == 'n') {
                if (!tokenizer_json_literal(&json, "null"))
                    goto malformed;
            } else {
                const unsigned char *begin;
                const unsigned char *end;
                if (!tokenizer_json_string_raw(&json, &begin, &end))
                    goto malformed;
                metadata->summary.chat_template_present = 1;
            }
            seen |= 32u;
        } else if (!tokenizer_json_skip(&json, 1u)) {
            goto malformed;
        }
        tokenizer_json_space(&json);
        if (json.cursor >= json.end)
            goto malformed;
        if (*json.cursor == '}')
            continue;
        if (*json.cursor++ != ',')
            goto malformed;
    }
    tokenizer_json_space(&json);
    if (json.cursor != json.end || (seen & 31u) != 31u)
        goto malformed;
    return YVEX_OK;
malformed:
    return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_MALFORMED_JSON, "tokenizer_config.json",
                          (unsigned long long)(json.cursor - metadata->tokenizer_config.bytes), 1u,
                          0u, err, YVEX_ERR_FORMAT,
                          "tokenizer config is malformed or lacks special-token policy");
}

static int tokenizer_find_token(const yvex_gguf_tokenizer_metadata *metadata,
                                tokenizer_string_ref target, unsigned int *out) {
    unsigned long long index;
    for (index = 0u; index < metadata->summary.token_count; ++index)
        if (tokenizer_ref_equal(metadata, metadata->tokens[index], target)) {
            *out = (unsigned int)index;
            return 1;
        }
    return 0;
}

static int tokenizer_blob_sha(const yvex_source_metadata_blob *blob,
                              char out[YVEX_GGUF_TOKENIZER_SHA256_CAP]) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, blob->bytes, blob->byte_count) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, out);
    return 1;
}

/*
 * Construct immutable target-scale tokenizer metadata from verified sidecar facts.
 *
 * Identity, JSON, cardinality, special-token, digest, bounds, or allocation unwind fully.
 */
int yvex_gguf_tokenizer_metadata_load(yvex_gguf_tokenizer_metadata **out,
                                      const yvex_source_verification *verification,
                                      unsigned long long expected_vocab_size,
                                      const char *pre_tokenizer, size_t maximum_owned_bytes,
                                      yvex_gguf_tokenizer_failure *failure, yvex_error *err) {
    yvex_gguf_tokenizer_metadata *metadata;
    size_t token_bytes;
    size_t type_bytes;
    int rc;

    if (out)
        *out = NULL;
    if (!out || !verification || !verification->tokenizer_json_valid ||
        !verification->tokenizer_config_valid || !expected_vocab_size ||
        expected_vocab_size > UINT_MAX || !pre_tokenizer || !pre_tokenizer[0] ||
        strlen(pre_tokenizer) >= 32u || maximum_owned_bytes < 1024u ||
        expected_vocab_size > SIZE_MAX / sizeof(tokenizer_string_ref) ||
        expected_vocab_size > SIZE_MAX / sizeof(int))
        return tokenizer_fail(
            failure, YVEX_GGUF_TOKENIZER_INVALID_ARGUMENT, "load", ULLONG_MAX, 1u, 0u, err,
            YVEX_ERR_INVALID_ARG,
            "verified tokenizer facts, vocabulary, pre-tokenizer, and budget are required");
    metadata = (yvex_gguf_tokenizer_metadata *)calloc(1u, sizeof(*metadata));
    if (!metadata)
        return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_ALLOCATION, "metadata", ULLONG_MAX,
                              sizeof(*metadata), 0u, err, YVEX_ERR_NOMEM,
                              "tokenizer metadata allocation failed");
    metadata->summary.token_count = expected_vocab_size;
    metadata->arena.maximum = maximum_owned_bytes;
    metadata->arena.initial_capacity = 4096u;
    yvex_core_text_copy(metadata->summary.pre_tokenizer, sizeof(metadata->summary.pre_tokenizer), pre_tokenizer);
    rc = yvex_source_provenance_metadata_read(verification, "tokenizer.json", TOKENIZER_JSON_LIMIT,
                                              &metadata->tokenizer_json, err);
    if (rc == YVEX_OK)
        rc = yvex_source_provenance_metadata_read(verification, "tokenizer_config.json",
                                                  TOKENIZER_CONFIG_LIMIT,
                                                  &metadata->tokenizer_config, err);
    if (rc != YVEX_OK) {
        yvex_gguf_tokenizer_metadata_release(&metadata);
        return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_SOURCE_IDENTITY, "sidecar", ULLONG_MAX,
                              1u, 0u, err, (yvex_status)rc,
                              "tokenizer sidecar identity or exact read failed");
    }
    token_bytes = (size_t)expected_vocab_size * sizeof(*metadata->tokens);
    type_bytes = (size_t)expected_vocab_size * sizeof(*metadata->token_types);
    if (token_bytes > maximum_owned_bytes || type_bytes > maximum_owned_bytes - token_bytes) {
        yvex_gguf_tokenizer_metadata_release(&metadata);
        return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_RESOURCE_LIMIT, "vocabulary", ULLONG_MAX,
                              token_bytes + type_bytes, maximum_owned_bytes, err, YVEX_ERR_BOUNDS,
                              "tokenizer index exceeds its ownership budget");
    }
    metadata->arena.maximum = maximum_owned_bytes - token_bytes - type_bytes;
    metadata->tokens =
        (tokenizer_string_ref *)calloc((size_t)expected_vocab_size, sizeof(*metadata->tokens));
    metadata->token_types =
        (int *)calloc((size_t)expected_vocab_size, sizeof(*metadata->token_types));
    if (!metadata->tokens || !metadata->token_types) {
        yvex_gguf_tokenizer_metadata_release(&metadata);
        return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_ALLOCATION, "vocabulary", ULLONG_MAX,
                              token_bytes + type_bytes, 0u, err, YVEX_ERR_NOMEM,
                              "tokenizer vocabulary index allocation failed");
    }
    rc = tokenizer_parse_json(metadata, failure, err);
    if (rc == YVEX_OK)
        rc = tokenizer_parse_config(metadata, failure, err);
    if (rc == YVEX_OK &&
        (!tokenizer_find_token(metadata, metadata->config_bos, &metadata->summary.bos_token_id) ||
         !tokenizer_find_token(metadata, metadata->config_eos, &metadata->summary.eos_token_id) ||
         !tokenizer_find_token(metadata, metadata->config_pad, &metadata->summary.pad_token_id))) {
        rc = tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_SPECIAL_TOKEN, "special_token", ULLONG_MAX,
                            3u, 0u, err, YVEX_ERR_FORMAT,
                            "tokenizer special-token content has no vocabulary id");
    }
    if (rc != YVEX_OK ||
        !tokenizer_blob_sha(&metadata->tokenizer_json, metadata->summary.tokenizer_json_sha256) ||
        !tokenizer_blob_sha(&metadata->tokenizer_config,
                            metadata->summary.tokenizer_config_sha256)) {
        if (rc == YVEX_OK)
            rc = tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_SOURCE_IDENTITY, "sha256", ULLONG_MAX,
                                1u, 0u, err, YVEX_ERR_BOUNDS,
                                "tokenizer sidecar digest calculation failed");
        yvex_gguf_tokenizer_metadata_release(&metadata);
        return rc;
    }
    metadata->summary.tokenizer_json_bytes = metadata->tokenizer_json.byte_count;
    metadata->summary.tokenizer_config_bytes = metadata->tokenizer_config.byte_count;
    metadata->summary.decoded_string_bytes = metadata->arena.count;
    metadata->summary.owned_bytes =
        sizeof(*metadata) + token_bytes + type_bytes + metadata->arena.capacity +
        metadata->merge_capacity * sizeof(*metadata->merges) +
        metadata->added_capacity * sizeof(*metadata->added) + metadata->tokenizer_json.byte_count +
        metadata->tokenizer_config.byte_count;
    if (metadata->summary.owned_bytes > maximum_owned_bytes) {
        unsigned long long observed_owned_bytes = metadata->summary.owned_bytes;
        yvex_gguf_tokenizer_metadata_release(&metadata);
        return tokenizer_fail(failure, YVEX_GGUF_TOKENIZER_RESOURCE_LIMIT, "owned_bytes",
                              ULLONG_MAX, maximum_owned_bytes, observed_owned_bytes, err,
                              YVEX_ERR_BOUNDS,
                              "tokenizer metadata exceeded its final ownership budget");
    }
    memcpy(metadata->summary.tokenizer_json_git_oid,
           metadata->tokenizer_json.identity.observed_git_blob_oid,
           sizeof(metadata->summary.tokenizer_json_git_oid));
    memcpy(metadata->summary.tokenizer_config_git_oid,
           metadata->tokenizer_config.identity.observed_git_blob_oid,
           sizeof(metadata->summary.tokenizer_config_git_oid));
    metadata->summary.complete = 1;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    *out = metadata;
    return YVEX_OK;
}

/*
 * Release every tokenizer index, arena, and verified sidecar blob idempotently.
 *
 * Clears caller ownership and frees every transitive allocation. Source verification ownership
 * remains external.
 */
void yvex_gguf_tokenizer_metadata_release(yvex_gguf_tokenizer_metadata **metadata_address) {
    yvex_gguf_tokenizer_metadata *metadata;
    if (!metadata_address || !*metadata_address)
        return;
    metadata = *metadata_address;
    *metadata_address = NULL;
    yvex_source_metadata_blob_release(&metadata->tokenizer_json);
    yvex_source_metadata_blob_release(&metadata->tokenizer_config);
    free(metadata->tokens);
    free(metadata->token_types);
    free(metadata->merges);
    free(metadata->added);
    free(metadata->arena.data);
    memset(metadata, 0, sizeof(*metadata));
    free(metadata);
}

/*
 * Borrow the immutable complete tokenizer metadata summary.
 *
 * The view remains valid only for the owner lifetime.
 */
const yvex_gguf_tokenizer_summary *
yvex_gguf_tokenizer_summary_get(const yvex_gguf_tokenizer_metadata *metadata) {
    return metadata && metadata->summary.complete ? &metadata->summary : NULL;
}

int yvex_gguf_tokenizer_token_at(const yvex_gguf_tokenizer_metadata *metadata,
                                 unsigned long long index, const unsigned char **bytes,
                                 size_t *byte_count, int *token_type) {
    if (!metadata || !metadata->summary.complete || !bytes || !byte_count || !token_type ||
        index >= metadata->summary.token_count || !metadata->tokens[index].present)
        return 0;
    *bytes = metadata->arena.data + metadata->tokens[index].offset;
    *byte_count = metadata->tokens[index].length;
    *token_type = metadata->token_types[index];
    return 1;
}

int yvex_gguf_tokenizer_merge_at(const yvex_gguf_tokenizer_metadata *metadata,
                                 unsigned long long index, const unsigned char **bytes,
                                 size_t *byte_count) {
    if (!metadata || !metadata->summary.complete || !bytes || !byte_count ||
        index >= metadata->summary.merge_count)
        return 0;
    *bytes = metadata->arena.data + metadata->merges[index].offset;
    *byte_count = metadata->merges[index].length;
    return 1;
}

int yvex_gguf_tokenizer_raw_json(const yvex_gguf_tokenizer_metadata *metadata,
                                 const unsigned char **bytes, size_t *byte_count) {
    if (!metadata || !metadata->summary.complete || !bytes || !byte_count)
        return 0;
    *bytes = metadata->tokenizer_json.bytes;
    *byte_count = metadata->tokenizer_json.byte_count;
    return 1;
}

int yvex_gguf_tokenizer_raw_config(const yvex_gguf_tokenizer_metadata *metadata,
                                   const unsigned char **bytes, size_t *byte_count) {
    if (!metadata || !metadata->summary.complete || !bytes || !byte_count)
        return 0;
    *bytes = metadata->tokenizer_config.bytes;
    *byte_count = metadata->tokenizer_config.byte_count;
    return 1;
}
