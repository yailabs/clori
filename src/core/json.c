/*
 * Eliminate independent JSON cursors in source and quantization owners.
 *
 * Reads, recursion, copied keys, and allocated strings have explicit caps. Consumers assign
 * meaning only after this owner accepts the byte syntax.
 */
#include <yvex/internal/core.h>

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define YVEX_JSON_DEPTH_CAP 64u

void yvex_json_init(yvex_json *json, const char *data, size_t length)
{
    if (!json) return;
    json->cursor = data;
    json->end = data ? data + length : data;
    json->depth = 0u;
    json->extensions = 0u;
}

char *yvex_read_bounded_file(const char *path,
                             size_t cap,
                             size_t *length,
                             yvex_error *err)
{
    struct stat st;
    FILE *fp;
    char *data;
    size_t wanted;
    int read_failed;

    if (length) *length = 0u;
    if (!path || stat(path, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) {
        return NULL;
    }
    if ((unsigned long long)st.st_size > (unsigned long long)cap) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, "source_metadata_read",
                        "metadata file exceeds bounded read limit: %s", path);
        return NULL;
    }
    wanted = (size_t)st.st_size;
    data = (char *)malloc(wanted + 1u);
    if (!data) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "source_metadata_read",
                       "metadata buffer allocation failed");
        return NULL;
    }
    fp = fopen(path, "rb");
    if (!fp) {
        free(data);
        yvex_error_setf(err, YVEX_ERR_IO, "source_metadata_read",
                        "cannot read metadata file: %s", path);
        return NULL;
    }
    read_failed = (wanted && fread(data, 1u, wanted, fp) != wanted) || ferror(fp);
    if (fclose(fp) != 0) read_failed = 1;
    if (read_failed) {
        free(data);
        yvex_error_setf(err, YVEX_ERR_IO, "source_metadata_read",
                        "cannot read metadata file: %s", path);
        return NULL;
    }
    data[wanted] = '\0';
    if (length) *length = wanted;
    return data;
}

const char *yvex_json_probe_field_value(const char *text, const char *key)
{
    char needle[96];
    const char *cursor;

    if (!text || !key)
        return NULL;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    cursor = strstr(text, needle);
    if (!cursor)
        return NULL;
    cursor = strchr(cursor, ':');
    if (!cursor)
        return NULL;
    cursor++;
    while (*cursor && isspace((unsigned char)*cursor))
        cursor++;
    return cursor;
}

int yvex_json_probe_string_field(const char *text,
                                 const char *key,
                                 char *out,
                                 size_t capacity)
{
    const char *cursor;
    const char *end;
    size_t length;

    if (out && capacity > 0u)
        out[0] = '\0';
    if (!text || !key || !out || capacity == 0u)
        return 0;
    cursor = yvex_json_probe_field_value(text, key);
    if (!cursor)
        return 0;
    if (*cursor != '"')
        return 0;
    cursor++;
    end = strchr(cursor, '"');
    if (!end)
        return 0;
    length = (size_t)(end - cursor);
    if (length >= capacity)
        length = capacity - 1u;
    memcpy(out, cursor, length);
    out[length] = '\0';
    return 1;
}

const void *yvex_core_keyed_row_find(const void *rows,
                                     size_t count,
                                     size_t stride,
                                     size_t key_offset,
                                     const char *key)
{
    const unsigned char *row = (const unsigned char *)rows;
    size_t index;

    if (!row || !key || stride < key_offset || stride - key_offset < sizeof(key))
        return NULL;
    for (index = 0u; index < count; ++index, row += stride) {
        const char *row_key;

        memcpy(&row_key, row + key_offset, sizeof(row_key));
        if (row_key && strcmp(row_key, key) == 0)
            return row;
    }
    return NULL;
}

void yvex_json_space(yvex_json *json)
{
    while (json && json->cursor < json->end &&
           isspace((unsigned char)*json->cursor)) {
        json->cursor++;
    }
}

int yvex_json_iter_begin(yvex_json *json,
                         yvex_json_iter *iter,
                         yvex_json_collection collection)
{
    char opening;
    char closing;

    if (collection == YVEX_JSON_COLLECTION_OBJECT) {
        opening = '{';
        closing = '}';
    } else if (collection == YVEX_JSON_COLLECTION_ARRAY) {
        opening = '[';
        closing = ']';
    } else {
        return 0;
    }
    yvex_json_space(json);
    if (!json || !iter || json->cursor >= json->end || *json->cursor++ != opening)
        return 0;
    iter->json = json;
    iter->count = 0u;
    iter->closing = closing;
    iter->complete = 0;
    iter->trailing_separator = 0;
    return 1;
}

static yvex_json_item json_iter_advance(yvex_json_iter *iter)
{
    yvex_json *json;

    if (!iter || !iter->json || iter->complete)
        return YVEX_JSON_ITEM_ERROR;
    json = iter->json;
    yvex_json_space(json);
    if (json->cursor >= json->end)
        return YVEX_JSON_ITEM_ERROR;
    if (*json->cursor == iter->closing) {
        json->cursor++;
        iter->complete = 1;
        return YVEX_JSON_ITEM_END;
    }
    if (iter->count != 0u) {
        if (*json->cursor++ != ',')
            return YVEX_JSON_ITEM_ERROR;
        yvex_json_space(json);
        if (json->cursor >= json->end)
            return YVEX_JSON_ITEM_ERROR;
        if (*json->cursor == iter->closing) {
            json->cursor++;
            iter->complete = 1;
            iter->trailing_separator = 1;
            return YVEX_JSON_ITEM_END;
        }
    }
    iter->count++;
    return YVEX_JSON_ITEM_READY;
}

yvex_json_item yvex_json_object_member(yvex_json_iter *iter,
                                       char *key,
                                       size_t capacity)
{
    yvex_json_item item = json_iter_advance(iter);
    yvex_json *json;

    if (item != YVEX_JSON_ITEM_READY)
        return item;
    json = iter->json;
    if (!yvex_json_string(json, key, capacity))
        return YVEX_JSON_ITEM_ERROR;
    yvex_json_space(json);
    if (json->cursor >= json->end || *json->cursor++ != ':')
        return YVEX_JSON_ITEM_ERROR;
    return YVEX_JSON_ITEM_READY;
}

yvex_json_item yvex_json_array_value(yvex_json_iter *iter)
{
    return json_iter_advance(iter);
}

static int source_json_hex(char value)
{
    return (value >= '0' && value <= '9') ||
           (value >= 'a' && value <= 'f') ||
           (value >= 'A' && value <= 'F');
}

int yvex_json_string(yvex_json *json, char *out, size_t cap)
{
    size_t length = 0u;

    yvex_json_space(json);
    if (!json || json->cursor >= json->end || *json->cursor++ != '"') return 0;
    while (json->cursor < json->end) {
        unsigned char ch = (unsigned char)*json->cursor++;
        if (ch == '"') {
            if (out && cap) out[length < cap ? length : cap - 1u] = '\0';
            return !out || length < cap;
        }
        if (ch < 0x20u) return 0;
        if (ch == '\\') {
            unsigned char escaped;
            if (json->cursor >= json->end) return 0;
            escaped = (unsigned char)*json->cursor++;
            if (escaped == 'u') {
                unsigned int i;
                for (i = 0; i < 4u; ++i) {
                    if (json->cursor >= json->end ||
                        !source_json_hex(*json->cursor++)) return 0;
                }
                ch = '?';
            } else {
                switch (escaped) {
                case '"': ch = '"'; break;
                case '\\': ch = '\\'; break;
                case '/': ch = '/'; break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                default: return 0;
                }
            }
        }
        if (out && length + 1u < cap) out[length] = (char)ch;
        length++;
    }
    return 0;
}

char *yvex_json_string_dup(yvex_json *json, size_t cap)
{
    yvex_json original;
    yvex_json probe;
    const char *start;
    char *value;
    size_t encoded;

    if (!json || cap == 0u) return NULL;
    original = *json;
    probe = *json;
    yvex_json_space(&probe);
    start = probe.cursor;
    if (!yvex_json_string(&probe, NULL, 0u)) return NULL;
    encoded = (size_t)(probe.cursor - start);
    if (encoded < 2u || encoded - 1u > cap) return NULL;
    value = (char *)malloc(encoded);
    if (!value) return NULL;
    if (!yvex_json_string(json, value, encoded)) {
        *json = original;
        free(value);
        return NULL;
    }
    return value;
}

static int json_skip_array(yvex_json *json)
{
    yvex_json_iter iter;
    yvex_json_item item;

    if (!json || json->depth >= YVEX_JSON_DEPTH_CAP ||
        !yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY)) return 0;
    json->depth++;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        if (!yvex_json_skip_value(json)) return 0;
    }
    if (item != YVEX_JSON_ITEM_END || iter.trailing_separator) return 0;
    json->depth--;
    return 1;
}

static int json_skip_object(yvex_json *json)
{
    char key[YVEX_JSON_KEY_CAP];
    yvex_json_iter iter;
    yvex_json_item item;

    if (!json || json->depth >= YVEX_JSON_DEPTH_CAP ||
        !yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    json->depth++;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (!yvex_json_skip_value(json)) return 0;
    }
    if (item != YVEX_JSON_ITEM_END || iter.trailing_separator) return 0;
    json->depth--;
    return 1;
}

static int source_json_skip_number(yvex_json *json)
{
    const char *start;

    yvex_json_space(json);
    if (!json) return 0;
    start = json->cursor;
    if (json->cursor < json->end && *json->cursor == '-') json->cursor++;
    if (json->cursor >= json->end) return 0;
    if (*json->cursor == '0') {
        json->cursor++;
    } else {
        if (!isdigit((unsigned char)*json->cursor)) return 0;
        while (json->cursor < json->end &&
               isdigit((unsigned char)*json->cursor)) json->cursor++;
    }
    if (json->cursor < json->end && *json->cursor == '.') {
        json->cursor++;
        if (json->cursor >= json->end ||
            !isdigit((unsigned char)*json->cursor)) return 0;
        while (json->cursor < json->end &&
               isdigit((unsigned char)*json->cursor)) json->cursor++;
    }
    if (json->cursor < json->end &&
        (*json->cursor == 'e' || *json->cursor == 'E')) {
        json->cursor++;
        if (json->cursor < json->end &&
            (*json->cursor == '+' || *json->cursor == '-')) json->cursor++;
        if (json->cursor >= json->end ||
            !isdigit((unsigned char)*json->cursor)) return 0;
        while (json->cursor < json->end &&
               isdigit((unsigned char)*json->cursor)) json->cursor++;
    }
    return json->cursor > start;
}

static int source_json_literal(yvex_json *json, const char *literal)
{
    size_t length = strlen(literal);

    yvex_json_space(json);
    if (!json || (size_t)(json->end - json->cursor) < length ||
        memcmp(json->cursor, literal, length) != 0) return 0;
    json->cursor += length;
    return 1;
}

int yvex_json_skip_value(yvex_json *json)
{
    yvex_json_space(json);
    if (!json || json->cursor >= json->end) return 0;
    if (*json->cursor == '{') return json_skip_object(json);
    if (*json->cursor == '[') return json_skip_array(json);
    if (*json->cursor == '"') return yvex_json_string(json, NULL, 0u);
    if (*json->cursor == 't') return source_json_literal(json, "true");
    if (*json->cursor == 'f') return source_json_literal(json, "false");
    if (*json->cursor == 'n') return source_json_literal(json, "null");
    if (*json->cursor == 'I' && (json->extensions & YVEX_JSON_EXTENSION_POSITIVE_INFINITY))
        return source_json_literal(json, "Infinity");
    return source_json_skip_number(json);
}

int yvex_json_complete(yvex_json *json)
{
    yvex_json_space(json);
    return json && json->cursor == json->end && json->depth == 0u;
}

/* Parses an unsigned decimal integer with checked overflow and no coercion. */

int yvex_json_u64(yvex_json *json, unsigned long long *out)
{
    unsigned long long value = 0u;

    yvex_json_space(json);
    if (!json || !out || json->cursor >= json->end ||
        !isdigit((unsigned char)*json->cursor)) return 0;
    while (json->cursor < json->end &&
           isdigit((unsigned char)*json->cursor)) {
        unsigned int digit = (unsigned int)(*json->cursor++ - '0');
        if (value > (ULLONG_MAX - digit) / 10u) return 0;
        value = value * 10u + digit;
    }
    *out = value;
    return 1;
}

int yvex_json_bool(yvex_json *json, int *out)
{
    if (!out) return 0;
    if (source_json_literal(json, "true")) {
        *out = 1;
        return 1;
    }
    if (source_json_literal(json, "false")) {
        *out = 0;
        return 1;
    }
    return 0;
}

int yvex_json_number_text(yvex_json *json, char *out, size_t cap)
{
    const char *start;
    size_t length;

    if (!json || !out || cap == 0u) return 0;
    yvex_json_space(json);
    start = json->cursor;
    if (!source_json_skip_number(json)) return 0;
    length = (size_t)(json->cursor - start);
    if (length == 0u || length >= cap) return 0;
    memcpy(out, start, length);
    out[length] = '\0';
    return 1;
}

int yvex_json_u64_array(yvex_json *json,
                               unsigned long long *values,
                               size_t cap,
                               unsigned long long *count)
{
    unsigned long long used = 0u;
    yvex_json_iter iter;
    yvex_json_item item;

    if (!json || !values || !count || cap == 0u ||
        !yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY)) return 0;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        if (used >= (unsigned long long)cap ||
            !yvex_json_u64(json, &values[used])) return 0;
        used++;
    }
    if (item != YVEX_JSON_ITEM_END || iter.trailing_separator || used == 0u) return 0;
    *count = used;
    return 1;
}
