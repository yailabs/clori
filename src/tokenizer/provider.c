/*
 * Execute the compiled tool and response-format contract without gateway prompt
 * assembly.
 *
 * Application messages reach model syntax only through the tokenizer plan; prose is never a tool
 * call. Bridges transport-neutral provider facts to artifact-bound tokenizer prompt/completion
 * semantics.
 */

#include "src/tokenizer/private.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>

#define REASONING_EMIT_CAP 4096u
#define REASONING_DELIMITER_CAP 64u
#define PROVIDER_JSON_DEPTH_CAP 64u

typedef struct {
    unsigned char *data;
    unsigned long long count, capacity;
} provider_builder;

typedef struct {
    yvex_provider_span reasoning;
    yvex_provider_span final;
    int reasoning_complete;
} conversation_output_view;

struct yvex_tokenizer_reasoning_stream {
    yvex_tokenizer_reasoning_sink sink;
    void *sink_context;
    const unsigned char *reasoning_end;
    unsigned int reasoning_end_count;
    unsigned char pending[REASONING_DELIMITER_CAP];
    unsigned int pending_count;
    int reasoning, finished;
};

static int conversation_admitted(const yvex_tokenizer *tokenizer)
{
    return tokenizer && tokenizer->conversation == &tokenizer->conversation_view &&
           tokenizer->conversation->schema_version == YVEX_CONVERSATION_PROTOCOL_SCHEMA_V1 &&
           yvex_tokenizer_family_policy_validate(&tokenizer->compiled_policy, NULL) == YVEX_OK;
}

static int reasoning_emit(yvex_tokenizer_reasoning_stream *stream,
                          yvex_reasoning_segment segment,
                          const unsigned char *bytes,
                          unsigned long long count, yvex_error *err)
{
    return count ? stream->sink(stream->sink_context, segment, bytes, count, err)
                 : YVEX_OK;
}

int yvex_tokenizer_reasoning_stream_open(
    yvex_tokenizer_reasoning_stream **out, const yvex_tokenizer *tokenizer,
    yvex_reasoning_policy policy, yvex_tokenizer_reasoning_sink sink,
    void *sink_context, yvex_error *err)
{
    yvex_tokenizer_reasoning_stream *stream;
    if (out) *out = NULL;
    if (!out || !tokenizer || !tokenizer->plan.sealed ||
        !conversation_admitted(tokenizer) || !sink ||
        policy > YVEX_REASONING_MAXIMUM ||
        (policy != YVEX_REASONING_DISABLED &&
         !tokenizer->plan.explicit_reasoning_supported) ||
        (policy == YVEX_REASONING_MAXIMUM &&
         !tokenizer->plan.maximum_reasoning_supported)) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "tokenizer.reasoning",
                       "reasoning policy is not supported by the tokenizer plan");
        return YVEX_ERR_UNSUPPORTED;
    }
    stream = calloc(1u, sizeof(*stream));
    if (!stream) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.reasoning",
                       "reasoning stream allocation failed");
        return YVEX_ERR_NOMEM;
    }
    stream->sink = sink;
    stream->sink_context = sink_context;
    stream->reasoning_end = (const unsigned char *)
        tokenizer->conversation->thinking_end;
    stream->reasoning_end_count = (unsigned int)strlen(
        tokenizer->conversation->thinking_end);
    if (!stream->reasoning_end_count ||
        stream->reasoning_end_count > sizeof(stream->pending)) {
        free(stream);
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.reasoning",
                       "source reasoning delimiter exceeds stream capacity");
        return YVEX_ERR_BOUNDS;
    }
    stream->reasoning = policy != YVEX_REASONING_DISABLED;
    *out = stream;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_tokenizer_reasoning_stream_push(
    yvex_tokenizer_reasoning_stream *stream, const unsigned char *bytes,
    unsigned long long byte_count, yvex_error *err)
{
    unsigned char emitted[REASONING_EMIT_CAP];
    unsigned long long offset = 0u;
    unsigned int emitted_count = 0u;
    int rc = YVEX_OK;
    if (!stream || stream->finished || (!bytes && byte_count)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.reasoning",
                       "open reasoning stream and bounded bytes are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!stream->reasoning)
        return reasoning_emit(stream, YVEX_REASONING_SEGMENT_FINAL_TEXT,
                              bytes, byte_count, err);
    while (rc == YVEX_OK && offset < byte_count && stream->reasoning) {
        stream->pending[stream->pending_count++] = bytes[offset++];
        while (stream->pending_count &&
               memcmp(stream->reasoning_end, stream->pending,
                      stream->pending_count) != 0) {
            emitted[emitted_count++] = stream->pending[0];
            memmove(stream->pending, stream->pending + 1u,
                    --stream->pending_count);
            if (emitted_count == sizeof(emitted)) {
                rc = reasoning_emit(stream, YVEX_REASONING_SEGMENT_EXPLICIT,
                                    emitted, emitted_count, err);
                emitted_count = 0u;
            }
        }
        if (stream->pending_count == stream->reasoning_end_count) {
            if (emitted_count)
                rc = reasoning_emit(stream, YVEX_REASONING_SEGMENT_EXPLICIT,
                                    emitted, emitted_count, err);
            emitted_count = 0u;
            stream->pending_count = 0u;
            stream->reasoning = 0;
        }
    }
    if (rc == YVEX_OK && emitted_count)
        rc = reasoning_emit(stream, YVEX_REASONING_SEGMENT_EXPLICIT,
                            emitted, emitted_count, err);
    if (rc == YVEX_OK && offset < byte_count)
        rc = reasoning_emit(stream, YVEX_REASONING_SEGMENT_FINAL_TEXT,
                            bytes + offset, byte_count - offset, err);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_tokenizer_reasoning_stream_finish(
    yvex_tokenizer_reasoning_stream *stream, yvex_error *err)
{
    int incomplete, rc;
    if (!stream || stream->finished) {
        yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.reasoning",
                       "open reasoning stream is required");
        return YVEX_ERR_STATE;
    }
    incomplete = stream->reasoning;
    rc = reasoning_emit(stream, incomplete
                                    ? YVEX_REASONING_SEGMENT_EXPLICIT
                                    : YVEX_REASONING_SEGMENT_FINAL_TEXT,
                        stream->pending, stream->pending_count, err);
    stream->pending_count = 0u;
    stream->finished = 1;
    if (rc == YVEX_OK && incomplete) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.reasoning",
                       "thinking completion is missing its source delimiter");
        rc = YVEX_ERR_FORMAT;
    } else if (rc == YVEX_OK) {
        yvex_error_clear(err);
    }
    return rc;
}

void yvex_tokenizer_reasoning_stream_close(
    yvex_tokenizer_reasoning_stream **stream)
{
    if (!stream || !*stream) return;
    memset(*stream, 0, sizeof(**stream));
    free(*stream);
    *stream = NULL;
}

/*
 * Reserve one checked transactional provider prompt extent.
 *
 * Retains prior ownership on bounds or allocation failure.
 */
static int reserve(provider_builder *builder, unsigned long long add,
                   yvex_error *err)
{
    unsigned long long need, capacity;
    unsigned char *grown;
    if (!builder || builder->count > ULLONG_MAX - add - 1u) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.provider",
                       "provider prompt extent overflowed");
        return YVEX_ERR_BOUNDS;
    }
    need = builder->count + add + 1u;
    if (need <= builder->capacity) return YVEX_OK;
    capacity = builder->capacity ? builder->capacity : 512u;
    while (capacity < need) {
        if (capacity > ULLONG_MAX / 2u || capacity * 2u > SIZE_MAX) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "tokenizer.provider",
                           "provider prompt exceeds address space");
            return YVEX_ERR_BOUNDS;
        }
        capacity *= 2u;
    }
    grown = realloc(builder->data, (size_t)capacity);
    if (!grown) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "tokenizer.provider",
                       "provider prompt allocation failed");
        return YVEX_ERR_NOMEM;
    }
    builder->data = grown;
    builder->capacity = capacity;
    return YVEX_OK;
}

static int append(provider_builder *builder, const void *bytes,
                  unsigned long long count, yvex_error *err)
{
    int rc;
    if ((!bytes && count) || count > SIZE_MAX) return YVEX_ERR_INVALID_ARG;
    rc = reserve(builder, count, err);
    if (rc != YVEX_OK) return rc;
    if (count) memcpy(builder->data + builder->count, bytes, (size_t)count);
    builder->count += count;
    builder->data[builder->count] = '\0';
    return YVEX_OK;
}

static int literal(provider_builder *builder, const char *text, yvex_error *err)
{
    return append(builder, text, (unsigned long long)strlen(text), err);
}

/* Render one JSON string in deterministic ensure-ascii-false form. */
static int json_string(provider_builder *builder, const unsigned char *bytes,
                       unsigned long long count, yvex_error *err)
{
    unsigned long long index;
    int rc = literal(builder, "\"", err);
    for (index = 0u; rc == YVEX_OK && index < count; ++index) {
        unsigned char byte = bytes[index];
        char escaped[7];
        if (byte == '"' || byte == '\\') {
            escaped[0] = '\\';
            escaped[1] = (char)byte;
            rc = append(builder, escaped, 2u, err);
        } else if (byte == '\b') rc = literal(builder, "\\b", err);
        else if (byte == '\f') rc = literal(builder, "\\f", err);
        else if (byte == '\n') rc = literal(builder, "\\n", err);
        else if (byte == '\r') rc = literal(builder, "\\r", err);
        else if (byte == '\t') rc = literal(builder, "\\t", err);
        else if (byte < 0x20u) {
            (void)snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
            rc = append(builder, escaped, 6u, err);
        } else {
            rc = append(builder, &byte, 1u, err);
        }
    }
    return rc == YVEX_OK ? literal(builder, "\"", err) : rc;
}

static int valid_utf8(const unsigned char *bytes, unsigned long long count)
{
    unsigned long long offset = 0u;
    uint32_t point;
    if (!bytes && count) return 0;
    while (offset < count)
        if (!yvex_tokenizer_utf8_next(bytes, count, &offset, &point)) return 0;
    return 1;
}

static int hex_nibble(unsigned char byte)
{
    if (byte >= '0' && byte <= '9') return byte - '0';
    if (byte >= 'a' && byte <= 'f') return byte - 'a' + 10;
    if (byte >= 'A' && byte <= 'F') return byte - 'A' + 10;
    return -1;
}

static int codepoint_append(provider_builder *builder, unsigned int point,
                            yvex_error *err)
{
    unsigned char bytes[4];
    unsigned int count;
    if (point <= 0x7fu) {
        bytes[0] = (unsigned char)point;
        count = 1u;
    } else if (point <= 0x7ffu) {
        bytes[0] = (unsigned char)(0xc0u | point >> 6u);
        bytes[1] = (unsigned char)(0x80u | (point & 0x3fu));
        count = 2u;
    } else if (point <= 0xffffu && (point < 0xd800u || point > 0xdfffu)) {
        bytes[0] = (unsigned char)(0xe0u | point >> 12u);
        bytes[1] = (unsigned char)(0x80u | ((point >> 6u) & 0x3fu));
        bytes[2] = (unsigned char)(0x80u | (point & 0x3fu));
        count = 3u;
    } else if (point >= 0x10000u && point <= 0x10ffffu) {
        bytes[0] = (unsigned char)(0xf0u | point >> 18u);
        bytes[1] = (unsigned char)(0x80u | ((point >> 12u) & 0x3fu));
        bytes[2] = (unsigned char)(0x80u | ((point >> 6u) & 0x3fu));
        bytes[3] = (unsigned char)(0x80u | (point & 0x3fu));
        count = 4u;
    } else {
        return YVEX_ERR_FORMAT;
    }
    return append(builder, bytes, count, err);
}

static int hex_quad(yvex_json *json, unsigned int *point)
{
    unsigned int value = 0u, index;
    if (!json || !point || (size_t)(json->end - json->cursor) < 4u) return 0;
    for (index = 0u; index < 4u; ++index) {
        int nibble = hex_nibble((unsigned char)*json->cursor++);
        if (nibble < 0) return 0;
        value = (value << 4u) | (unsigned int)nibble;
    }
    *point = value;
    return 1;
}

/* Decode one JSON string exactly before ensure-ascii-false re-encoding. */
static int json_string_decode(yvex_json *json, provider_builder *decoded,
                              yvex_error *err)
{
    yvex_json_space(json);
    if (!json || !decoded || json->cursor >= json->end ||
        *json->cursor++ != '"')
        return YVEX_ERR_FORMAT;
    while (json->cursor < json->end) {
        unsigned char byte = (unsigned char)*json->cursor++;
        int rc;
        if (byte == '"') return YVEX_OK;
        if (byte < 0x20u) return YVEX_ERR_FORMAT;
        if (byte != '\\') {
            rc = append(decoded, &byte, 1u, err);
        } else {
            unsigned int point;
            if (json->cursor >= json->end) return YVEX_ERR_FORMAT;
            byte = (unsigned char)*json->cursor++;
            if (byte == 'u') {
                if (!hex_quad(json, &point)) return YVEX_ERR_FORMAT;
                if (point >= 0xd800u && point <= 0xdbffu) {
                    unsigned int low;
                    if ((size_t)(json->end - json->cursor) < 6u ||
                        json->cursor[0] != '\\' || json->cursor[1] != 'u')
                        return YVEX_ERR_FORMAT;
                    json->cursor += 2u;
                    if (!hex_quad(json, &low) || low < 0xdc00u || low > 0xdfffu)
                        return YVEX_ERR_FORMAT;
                    point = 0x10000u + ((point - 0xd800u) << 10u) +
                            (low - 0xdc00u);
                } else if (point >= 0xdc00u && point <= 0xdfffu) {
                    return YVEX_ERR_FORMAT;
                }
                rc = codepoint_append(decoded, point, err);
            } else {
                switch (byte) {
                case '"': case '\\': case '/': break;
                case 'b': byte = '\b'; break;
                case 'f': byte = '\f'; break;
                case 'n': byte = '\n'; break;
                case 'r': byte = '\r'; break;
                case 't': byte = '\t'; break;
                default: return YVEX_ERR_FORMAT;
                }
                rc = append(decoded, &byte, 1u, err);
            }
        }
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_ERR_FORMAT;
}

static int json_punctuation(yvex_json *json, char expected)
{
    yvex_json_space(json);
    if (!json || json->cursor >= json->end || *json->cursor != expected)
        return 0;
    ++json->cursor;
    return 1;
}

static int json_canonical_value(provider_builder *builder, yvex_json *json,
                                unsigned int depth, yvex_error *err);

static int same_double(double left, double right)
{
    return memcmp(&left, &right, sizeof(left)) == 0;
}

/* Match Python's finite-float repr policy used by the pinned source encoder. */
static int json_python_float(provider_builder *builder, const char *bytes,
                             size_t count, yvex_error *err)
{
    char input[96], shortest[64], scientific[64], digits[32], output[96];
    char *end, *exponent_at;
    double value, roundtrip;
    int precision, exponent, negative, digit_count = 0, written = 0;
    size_t index;

    if (!count || count >= sizeof(input)) return YVEX_ERR_BOUNDS;
    memcpy(input, bytes, count);
    input[count] = '\0';
    errno = 0;
    value = strtod(input, &end);
    if (end != input + count || errno == ERANGE || !isfinite(value))
        return YVEX_ERR_FORMAT;
    if (value == 0.0)
        return literal(builder, signbit(value) ? "-0.0" : "0.0", err);
    for (precision = 1; precision <= 17; ++precision) {
        if (snprintf(shortest, sizeof(shortest), "%.*g", precision, value) <= 0)
            return YVEX_ERR_STATE;
        errno = 0;
        roundtrip = strtod(shortest, &end);
        if (!errno && *end == '\0' && same_double(value, roundtrip)) break;
    }
    if (precision > 17 ||
        snprintf(scientific, sizeof(scientific), "%.*e", precision - 1,
                 value) <= 0)
        return YVEX_ERR_STATE;
    exponent_at = strchr(scientific, 'e');
    if (!exponent_at) return YVEX_ERR_STATE;
    errno = 0;
    exponent = (int)strtol(exponent_at + 1, &end, 10);
    if (errno || *end != '\0') return YVEX_ERR_STATE;
    negative = scientific[0] == '-';
    for (index = negative ? 1u : 0u;
         scientific + index < exponent_at; ++index)
        if (scientific[index] != '.')
            digits[digit_count++] = scientific[index];
    if (!digit_count || digit_count > 17) return YVEX_ERR_STATE;
    if (negative) output[written++] = '-';
    if (exponent >= -4 && exponent < 16) {
        int before = exponent + 1;
        if (before <= 0) {
            output[written++] = '0';
            output[written++] = '.';
            while (before++ < 0) output[written++] = '0';
            memcpy(output + written, digits, (size_t)digit_count);
            written += digit_count;
        } else {
            int copied = before < digit_count ? before : digit_count;
            memcpy(output + written, digits, (size_t)copied);
            written += copied;
            while (copied++ < before) output[written++] = '0';
            if (before < digit_count) {
                output[written++] = '.';
                memcpy(output + written, digits + before,
                       (size_t)(digit_count - before));
                written += digit_count - before;
            } else {
                output[written++] = '.';
                output[written++] = '0';
            }
        }
    } else {
        output[written++] = digits[0];
        if (digit_count > 1) {
            output[written++] = '.';
            memcpy(output + written, digits + 1, (size_t)(digit_count - 1));
            written += digit_count - 1;
        }
        written += snprintf(output + written, sizeof(output) - (size_t)written,
                            "e%c%02d", exponent < 0 ? '-' : '+',
                            exponent < 0 ? -exponent : exponent);
    }
    if (written <= 0 || (size_t)written >= sizeof(output))
        return YVEX_ERR_BOUNDS;
    return append(builder, output, (unsigned long long)written, err);
}

static int json_canonical_collection(provider_builder *builder, yvex_json *json,
                                     unsigned int depth, int object,
                                     yvex_error *err)
{
    char closing = object ? '}' : ']';
    unsigned long long count = 0u;
    int rc = literal(builder, object ? "{" : "[", err);
    ++json->cursor;
    yvex_json_space(json);
    if (json->cursor < json->end && *json->cursor == closing) {
        ++json->cursor;
        return rc == YVEX_OK ? literal(builder, object ? "}" : "]", err) : rc;
    }
    while (rc == YVEX_OK) {
        if (count++) rc = literal(builder, ", ", err);
        if (rc == YVEX_OK && object) {
            provider_builder key = {0};
            rc = json_string_decode(json, &key, err);
            if (rc == YVEX_OK) rc = json_string(builder, key.data, key.count, err);
            free(key.data);
            if (rc == YVEX_OK && !json_punctuation(json, ':'))
                rc = YVEX_ERR_FORMAT;
            if (rc == YVEX_OK) rc = literal(builder, ": ", err);
        }
        if (rc == YVEX_OK)
            rc = json_canonical_value(builder, json, depth + 1u, err);
        yvex_json_space(json);
        if (rc != YVEX_OK || json->cursor >= json->end) break;
        if (*json->cursor == closing) {
            ++json->cursor;
            return literal(builder, object ? "}" : "]", err);
        }
        if (*json->cursor++ != ',') {
            rc = YVEX_ERR_FORMAT;
            break;
        }
    }
    return rc == YVEX_OK ? YVEX_ERR_FORMAT : rc;
}

static int json_canonical_value(provider_builder *builder, yvex_json *json,
                                unsigned int depth, yvex_error *err)
{
    provider_builder decoded = {0};
    yvex_json probe;
    const char *start;
    int rc;
    if (!builder || !json || depth >= PROVIDER_JSON_DEPTH_CAP)
        return YVEX_ERR_BOUNDS;
    yvex_json_space(json);
    if (json->cursor >= json->end) return YVEX_ERR_FORMAT;
    if (*json->cursor == '{')
        return json_canonical_collection(builder, json, depth, 1, err);
    if (*json->cursor == '[')
        return json_canonical_collection(builder, json, depth, 0, err);
    if (*json->cursor == '"') {
        rc = json_string_decode(json, &decoded, err);
        if (rc == YVEX_OK)
            rc = json_string(builder, decoded.data, decoded.count, err);
        free(decoded.data);
        return rc;
    }
    start = json->cursor;
    probe = *json;
    if (!yvex_json_skip_value(&probe) || probe.cursor <= start)
        return YVEX_ERR_FORMAT;
    if ((*start == '-' || (*start >= '0' && *start <= '9')) &&
        memchr(start, '.', (size_t)(probe.cursor - start)))
        rc = json_python_float(builder, start,
                               (size_t)(probe.cursor - start), err);
    else if ((*start == '-' || (*start >= '0' && *start <= '9')) &&
             (memchr(start, 'e', (size_t)(probe.cursor - start)) ||
              memchr(start, 'E', (size_t)(probe.cursor - start))))
        rc = json_python_float(builder, start,
                               (size_t)(probe.cursor - start), err);
    else if (probe.cursor - start == 2 && start[0] == '-' && start[1] == '0')
        rc = literal(builder, "0", err);
    else
        rc = append(builder, start,
                    (unsigned long long)(probe.cursor - start), err);
    if (rc == YVEX_OK) *json = probe;
    return rc;
}

static int json_canonical_span(provider_builder *builder,
                               yvex_provider_span span, yvex_error *err)
{
    yvex_json json;
    int rc;
    if (!valid_utf8(span.bytes, span.count) || span.count > SIZE_MAX)
        return YVEX_ERR_FORMAT;
    yvex_json_init(&json, (const char *)span.bytes, (size_t)span.count);
    rc = json_canonical_value(builder, &json, 0u, err);
    return rc == YVEX_OK && yvex_json_complete(&json) ? YVEX_OK
                                                       : YVEX_ERR_FORMAT;
}

static int append_arguments(provider_builder *builder,
                            const yvex_conversation_protocol *conversation,
                            yvex_provider_span arguments, yvex_error *err)
{
    yvex_json json;
    int rc = YVEX_OK;
    if (arguments.count > SIZE_MAX ||
        !valid_utf8(arguments.bytes, arguments.count))
        return YVEX_ERR_BOUNDS;
    yvex_json_init(&json, (const char *)arguments.bytes, (size_t)arguments.count);
    if (!json_punctuation(&json, '{'))
        return YVEX_ERR_FORMAT;
    yvex_json_space(&json);
    if (json.cursor < json.end && *json.cursor == '}') {
        ++json.cursor;
        return yvex_json_complete(&json) ? YVEX_OK : YVEX_ERR_FORMAT;
    }
    while (rc == YVEX_OK && json.cursor < json.end) {
        provider_builder key = {0}, decoded = {0};
        int is_string;
        rc = json_string_decode(&json, &key, err);
        if (rc == YVEX_OK &&
            (!key.count || key.count >= YVEX_PROVIDER_TOOL_NAME_CAP ||
             memchr(key.data, '\0', (size_t)key.count) ||
             memchr(key.data, '"', (size_t)key.count)))
            rc = YVEX_ERR_FORMAT;
        if (rc == YVEX_OK && !json_punctuation(&json, ':'))
            rc = YVEX_ERR_FORMAT;
        yvex_json_space(&json);
        is_string = rc == YVEX_OK && json.cursor < json.end &&
                    *json.cursor == '"';
        if (rc == YVEX_OK &&
            (literal(builder, conversation->tool_parameter_start, err) !=
                 YVEX_OK ||
             append(builder, key.data, key.count, err) != YVEX_OK ||
             literal(builder, conversation->tool_parameter_name_end, err) !=
                 YVEX_OK ||
             literal(builder, is_string ? "true" : "false", err) != YVEX_OK ||
             literal(builder, conversation->tool_parameter_kind_end, err) !=
                 YVEX_OK))
            rc = yvex_error_code(err);
        if (rc == YVEX_OK && is_string)
            rc = json_string_decode(&json, &decoded, err);
        if (rc == YVEX_OK && is_string)
            rc = append(builder, decoded.data, decoded.count, err);
        else if (rc == YVEX_OK)
            rc = json_canonical_value(builder, &json, 0u, err);
        if (rc == YVEX_OK &&
            literal(builder, conversation->tool_parameter_end, err) != YVEX_OK)
            rc = yvex_error_code(err);
        free(decoded.data);
        free(key.data);
        if (rc != YVEX_OK) break;
        yvex_json_space(&json);
        if (json.cursor < json.end && *json.cursor == '}') {
            ++json.cursor;
            break;
        }
        if (json.cursor >= json.end || *json.cursor++ != ',')
            rc = YVEX_ERR_FORMAT;
    }
    return rc == YVEX_OK && yvex_json_complete(&json) ? YVEX_OK
                                                       : YVEX_ERR_FORMAT;
}

static int append_tool_calls(provider_builder *builder,
                             const yvex_conversation_protocol *conversation,
                             const yvex_provider_tool_call *calls,
                             unsigned long long call_count, yvex_error *err)
{
    unsigned long long index;
    int rc = literal(builder, conversation->tool_calls_start, err);
    for (index = 0u; rc == YVEX_OK && index < call_count; ++index) {
        const yvex_provider_tool_call *call = &calls[index];
        unsigned long long arguments_start;
        if (index) rc = literal(builder, "\n", err);
        if (rc == YVEX_OK)
            rc = literal(builder, conversation->tool_invoke_start, err);
        if (rc == YVEX_OK) rc = literal(builder, call->name, err);
        if (rc == YVEX_OK)
            rc = literal(builder, conversation->tool_invoke_name_end, err);
        arguments_start = builder->count;
        if (rc == YVEX_OK)
            rc = append_arguments(builder, conversation,
                                  call->arguments_json, err);
        if (rc == YVEX_OK && builder->count == arguments_start)
            rc = literal(builder, "\n", err);
        if (rc == YVEX_OK)
            rc = literal(builder, conversation->tool_invoke_end, err);
    }
    if (rc == YVEX_OK) rc = literal(builder, "\n", err);
    if (rc == YVEX_OK)
        rc = literal(builder, conversation->tool_calls_end, err);
    return rc;
}

static int append_tool_schema(provider_builder *builder,
                              unsigned int request_schema,
                              const yvex_provider_function_tool *tool,
                              yvex_error *err)
{
    int rc = literal(builder, "{\"name\": ", err);
    if (rc == YVEX_OK)
        rc = json_string(builder, (const unsigned char *)tool->name,
                         strlen(tool->name), err);
    if (rc == YVEX_OK &&
        (request_schema == YVEX_PROVIDER_SCHEMA_V1 ||
         tool->description_present))
        rc = literal(builder, ", \"description\": ", err);
    if (rc == YVEX_OK &&
        (request_schema == YVEX_PROVIDER_SCHEMA_V1 ||
         tool->description_present))
        rc = json_string(builder, tool->description.bytes,
                         tool->description.count, err);
    if (rc == YVEX_OK) rc = literal(builder, ", \"parameters\": ", err);
    if (rc == YVEX_OK)
        rc = json_canonical_span(builder, tool->parameters_json, err);
    if (rc == YVEX_OK && request_schema == YVEX_PROVIDER_SCHEMA_V2 &&
        tool->strict_present)
        rc = literal(builder, tool->strict ? ", \"strict\": true"
                                           : ", \"strict\": false", err);
    if (rc == YVEX_OK) rc = literal(builder, "}", err);
    return rc;
}

static int append_controls(provider_builder *builder,
                           const yvex_conversation_protocol *conversation,
                           const yvex_provider_request *request,
                           yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    if (request->tool_count) {
        rc = literal(builder, "\n\n", err);
        if (rc == YVEX_OK)
            rc = literal(builder, conversation->tools_prefix, err);
        for (index = 0u; rc == YVEX_OK && index < request->tool_count; ++index) {
            if (index) rc = literal(builder, "\n", err);
            if (rc == YVEX_OK)
                rc = append_tool_schema(builder, request->schema_version,
                                        &request->tools[index], err);
        }
        if (rc == YVEX_OK)
            rc = literal(builder, conversation->tools_suffix, err);
    }
    if (rc == YVEX_OK &&
        request->response_format == YVEX_PROVIDER_RESPONSE_JSON_OBJECT) {
        if (literal(builder, "\n\n", err) == YVEX_OK &&
            literal(builder, conversation->response_format_prefix, err) == YVEX_OK)
            rc = literal(builder, "{\"type\": \"json_object\"}", err);
        else
            rc = yvex_error_code(err);
    }
    return rc;
}

static int call_id_index(const yvex_provider_tool_call *calls,
                         unsigned long long count, const char *call_id,
                         unsigned long long *index)
{
    unsigned long long current;
    for (current = 0u; current < count; ++current)
        if (strcmp(calls[current].call_id, call_id) == 0) {
            if (index) *index = current;
            return 1;
        }
    return 0;
}

/* Project tool results in the source encoder's preceding-call order. */
static const yvex_provider_message *ordered_tool_result(
    const yvex_provider_request *request, unsigned long long message_index)
{
    const yvex_provider_message *message = &request->messages[message_index];
    const yvex_provider_tool_call *calls;
    unsigned long long call_count, group_start, group_end, slot = 0u;
    unsigned long long call, result, selected = 0u;
    if (message->role != YVEX_PROVIDER_ROLE_TOOL) return message;
    group_start = message_index;
    while (group_start &&
           (request->messages[group_start - 1u].role == YVEX_PROVIDER_ROLE_USER ||
            request->messages[group_start - 1u].role == YVEX_PROVIDER_ROLE_TOOL))
        --group_start;
    if (!group_start ||
        request->messages[group_start - 1u].role != YVEX_PROVIDER_ROLE_ASSISTANT)
        return NULL;
    calls = request->messages[group_start - 1u].tool_calls;
    call_count = request->messages[group_start - 1u].tool_call_count;
    if (!calls || !call_count) return NULL;
    group_end = message_index + 1u;
    while (group_end < request->message_count &&
           (request->messages[group_end].role == YVEX_PROVIDER_ROLE_USER ||
            request->messages[group_end].role == YVEX_PROVIDER_ROLE_TOOL))
        ++group_end;
    for (result = group_start; result < message_index; ++result)
        if (request->messages[result].role == YVEX_PROVIDER_ROLE_TOOL) ++slot;
    for (result = group_start; result < group_end; ++result) {
        unsigned long long unused;
        if (request->messages[result].role == YVEX_PROVIDER_ROLE_TOOL &&
            !call_id_index(calls, call_count,
                           request->messages[result].tool_call_id, &unused))
            return NULL;
    }
    for (call = 0u; call < call_count; ++call)
        for (result = group_start; result < group_end; ++result)
            if (request->messages[result].role == YVEX_PROVIDER_ROLE_TOOL &&
                strcmp(request->messages[result].tool_call_id,
                       calls[call].call_id) == 0) {
                unsigned long long duplicate;
                for (duplicate = result + 1u; duplicate < group_end; ++duplicate)
                    if (request->messages[duplicate].role ==
                            YVEX_PROVIDER_ROLE_TOOL &&
                        strcmp(request->messages[duplicate].tool_call_id,
                               calls[call].call_id) == 0)
                        return NULL;
                if (selected++ == slot) return &request->messages[result];
            }
    return NULL;
}

/*
 * Render the sealed provider request through the exact compiled conversation policy.
 *
 * Allocates exact prompt bytes and field-wise prompt/message identities.
 */
int yvex_tokenizer_provider_prompt(
    const yvex_tokenizer *tokenizer, const yvex_provider_request *request,
    yvex_rendered_prompt *rendered, yvex_error *err)
{
    provider_builder builder = {0};
    const yvex_conversation_protocol *conversation;
    unsigned long long index, controls_at = ULLONG_MAX;
    unsigned long long last_user = ULLONG_MAX;
    yvex_provider_role prior = YVEX_PROVIDER_ROLE_SYSTEM;
    int user_group = 0, effective_drop, thinking, rc;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!tokenizer || !rendered ||
        !conversation_admitted(tokenizer) ||
        tokenizer->plan.prompt_policy != YVEX_TOKENIZER_PROMPT_CONVERSATION ||
        yvex_provider_request_validate(request, err) != YVEX_OK)
        return YVEX_ERR_INVALID_ARG;
    conversation = tokenizer->conversation;
    thinking = request->schema_version == YVEX_PROVIDER_SCHEMA_V2 &&
               request->reasoning_policy != YVEX_REASONING_DISABLED;
    effective_drop = request->schema_version == YVEX_PROVIDER_SCHEMA_V1 ||
                     request->drop_thinking;
    if (request->tool_count && conversation->tools_preserve_reasoning)
        effective_drop = 0;
    memset(rendered, 0, sizeof(*rendered));
    for (index = 0u; index < request->message_count; ++index)
        if (request->messages[index].role == YVEX_PROVIDER_ROLE_SYSTEM ||
            request->messages[index].role == YVEX_PROVIDER_ROLE_DEVELOPER) {
            controls_at = index;
            break;
        }
    for (index = request->message_count; index > 0u; --index) {
        yvex_provider_role role = request->messages[index - 1u].role;
        if (role == YVEX_PROVIDER_ROLE_USER ||
            role == YVEX_PROVIDER_ROLE_DEVELOPER ||
            role == YVEX_PROVIDER_ROLE_TOOL) {
            last_user = index - 1u;
            break;
        }
    }
    rc = literal(&builder, conversation->bos, err);
    if (rc == YVEX_OK &&
        request->reasoning_policy == YVEX_REASONING_MAXIMUM)
        rc = literal(&builder, conversation->reasoning_effort_max, err);
    if (rc == YVEX_OK && controls_at == ULLONG_MAX)
        rc = append_controls(&builder, conversation, request, err);
    for (index = 0u; rc == YVEX_OK && index < request->message_count; ++index) {
        const yvex_provider_message *message =
            ordered_tool_result(request, index);
        if (!message) {
            rc = YVEX_ERR_FORMAT;
            break;
        }
        if (!valid_utf8(message->content.bytes, message->content.count) ||
            !valid_utf8(message->reasoning_content.bytes,
                        message->reasoning_content.count)) {
            rc = YVEX_ERR_FORMAT;
            break;
        }
        if (thinking && effective_drop && index < last_user &&
            message->role == YVEX_PROVIDER_ROLE_DEVELOPER)
            continue;
        if (message->role == YVEX_PROVIDER_ROLE_SYSTEM) {
            if (index != 0u) { rc = YVEX_ERR_FORMAT; break; }
            rc = append(&builder, message->content.bytes,
                        message->content.count, err);
            user_group = 0;
        } else if (message->role == YVEX_PROVIDER_ROLE_DEVELOPER) {
            rc = literal(&builder, conversation->user, err);
            if (rc == YVEX_OK)
                rc = append(&builder, message->content.bytes,
                            message->content.count, err);
            user_group = 0;
        } else if (message->role == YVEX_PROVIDER_ROLE_USER ||
                   message->role == YVEX_PROVIDER_ROLE_TOOL) {
            if (!user_group) rc = literal(&builder, conversation->user, err);
            else rc = literal(&builder, "\n\n", err);
            if (rc == YVEX_OK && message->role == YVEX_PROVIDER_ROLE_TOOL)
                rc = literal(&builder, conversation->tool_result_start, err);
            if (rc == YVEX_OK)
                rc = append(&builder, message->content.bytes,
                            message->content.count, err);
            if (rc == YVEX_OK && message->role == YVEX_PROVIDER_ROLE_TOOL)
                rc = literal(&builder, conversation->tool_result_end, err);
            user_group = 1;
        } else if (message->role == YVEX_PROVIDER_ROLE_ASSISTANT) {
            if (prior != YVEX_PROVIDER_ROLE_USER &&
                prior != YVEX_PROVIDER_ROLE_TOOL &&
                prior != YVEX_PROVIDER_ROLE_DEVELOPER) {
                rc = YVEX_ERR_FORMAT;
                break;
            }
            rc = literal(&builder, conversation->assistant, err);
            if (rc == YVEX_OK && thinking &&
                (!effective_drop || index > last_user)) {
                rc = literal(&builder, conversation->thinking_start, err);
                if (rc == YVEX_OK)
                    rc = append(&builder, message->reasoning_content.bytes,
                                message->reasoning_content.count, err);
            }
            if (rc == YVEX_OK)
                rc = literal(&builder, conversation->thinking_end, err);
            if (rc == YVEX_OK)
                rc = append(&builder, message->content.bytes,
                            message->content.count, err);
            if (rc == YVEX_OK && message->tool_call_count)
                rc = append_tool_calls(&builder, conversation,
                                       message->tool_calls,
                                       message->tool_call_count, err);
            if (rc == YVEX_OK) rc = literal(&builder, conversation->eos, err);
            user_group = 0;
        } else {
            rc = YVEX_ERR_FORMAT;
        }
        if (rc == YVEX_OK && index == controls_at)
            rc = append_controls(&builder, conversation, request, err);
        prior = message->role;
    }
    if (rc == YVEX_OK && prior != YVEX_PROVIDER_ROLE_USER &&
        prior != YVEX_PROVIDER_ROLE_TOOL && prior != YVEX_PROVIDER_ROLE_DEVELOPER)
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK) rc = literal(&builder, conversation->assistant, err);
    if (rc == YVEX_OK)
        rc = literal(&builder, thinking ? conversation->thinking_start
                                        : conversation->thinking_end,
                     err);
    if (rc != YVEX_OK) {
        free(builder.data);
        yvex_error_set(err, rc, "tokenizer.provider.prompt",
                       "provider messages do not satisfy the compiled prompt semantics");
        return rc;
    }
    rendered->text = (char *)builder.data;
    rendered->len = builder.count;
    rendered->generation_prompt = 1;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.provider.messages.v2") ||
        !yvex_sha256_update_text(&hash, request->request_identity) ||
        !yvex_sha256_final(&hash, digest)) goto identity_failure;
    yvex_sha256_hex(digest, rendered->message_sequence_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.provider.rendered.v2") ||
        !yvex_sha256_update_u64_be(&hash, rendered->len) ||
        !yvex_sha256_update(&hash, rendered->text, (size_t)rendered->len) ||
        !yvex_sha256_final(&hash, digest)) goto identity_failure;
    yvex_sha256_hex(digest, rendered->rendered_bytes_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.provider.prompt.v2") ||
        !yvex_sha256_update_text(&hash, tokenizer->plan.tokenizer_plan_identity) ||
        !yvex_sha256_update_text(&hash, request->request_identity) ||
        !yvex_sha256_update_text(&hash, rendered->rendered_bytes_identity) ||
        !yvex_sha256_final(&hash, digest)) goto identity_failure;
    yvex_sha256_hex(digest, rendered->prompt_identity);
    yvex_error_clear(err);
    return YVEX_OK;
identity_failure:
    yvex_rendered_prompt_free(rendered);
    yvex_error_set(err, YVEX_ERR_STATE, "tokenizer.provider.identity",
                   "provider prompt identity derivation failed");
    return YVEX_ERR_STATE;
}

int yvex_tokenizer_encode_provider_prompt(
    const yvex_tokenizer *tokenizer, const yvex_provider_request *request,
    const yvex_tokenizer_encode_options *encode_options,
    yvex_rendered_prompt *rendered, yvex_tokenizer_encode_result *encoded,
    yvex_error *err)
{
    yvex_tokenizer_encode_options options = {0, 0, 1, 0};
    int rc;
    if (!rendered || !encoded) return YVEX_ERR_INVALID_ARG;
    memset(rendered, 0, sizeof(*rendered));
    memset(encoded, 0, sizeof(*encoded));
    rc = yvex_tokenizer_provider_prompt(tokenizer, request, rendered, err);
    if (rc != YVEX_OK) return rc;
    if (encode_options) options = *encode_options;
    options.add_bos = 0;
    options.add_eos = 0;
    options.allow_special_tokens = 1;
    rc = yvex_tokenizer_encode(tokenizer,
                               (const unsigned char *)rendered->text,
                               rendered->len, &options, encoded, err);
    if (rc != YVEX_OK) {
        yvex_rendered_prompt_free(rendered);
        yvex_tokenizer_encode_result_clear(encoded);
    }
    return rc;
}

static const unsigned char *find_bytes(const unsigned char *bytes,
                                       unsigned long long count,
                                       const char *marker)
{
    size_t length = strlen(marker);
    unsigned long long index;
    if (!length || count < length) return NULL;
    for (index = 0u; index <= count - length; ++index)
        if (memcmp(bytes + index, marker, length) == 0) return bytes + index;
    return NULL;
}

static int consume(const unsigned char **cursor, const unsigned char *end,
                   const char *text)
{
    size_t count = strlen(text);
    if (!cursor || !*cursor || (size_t)(end - *cursor) < count ||
        memcmp(*cursor, text, count) != 0) return 0;
    *cursor += count;
    return 1;
}

static int starts_with(const unsigned char *cursor, const unsigned char *end,
                       const char *text)
{
    size_t count = strlen(text);
    return cursor && (size_t)(end - cursor) >= count &&
           memcmp(cursor, text, count) == 0;
}

static int attribute(const unsigned char **cursor, const unsigned char *end,
                     const char *delimiter, char *output, size_t capacity)
{
    const unsigned char *found = find_bytes(*cursor,
        (unsigned long long)(end - *cursor), delimiter);
    size_t count;
    if (!found) return 0;
    count = (size_t)(found - *cursor);
    if (!count || count >= capacity) return 0;
    memcpy(output, *cursor, count);
    output[count] = '\0';
    *cursor = found + strlen(delimiter);
    return 1;
}

static int tool_admitted(const yvex_provider_request *request, const char *name)
{
    unsigned long long index;
    for (index = 0u; index < request->tool_count; ++index)
        if (strcmp(request->tools[index].name, name) == 0) return 1;
    return 0;
}

static int completion_special_valid(
    const yvex_conversation_protocol *conversation,
    const conversation_output_view *view)
{
    const char *forbidden[] = {
        conversation->bos, conversation->eos, conversation->thinking_start,
        conversation->thinking_end};
    unsigned long long index;

    if (find_bytes(view->reasoning.bytes, view->reasoning.count,
                   conversation->dsml))
        return 0;
    for (index = 0u; index < sizeof(forbidden) / sizeof(forbidden[0]); ++index)
        if (find_bytes(view->reasoning.bytes, view->reasoning.count,
                       forbidden[index]) ||
            find_bytes(view->final.bytes, view->final.count,
                       forbidden[index]))
            return 0;
    return 1;
}

static int conversation_output_parse(
    const yvex_tokenizer *tokenizer, yvex_reasoning_policy policy,
    const unsigned char *bytes, unsigned long long byte_count,
    conversation_output_view *view, yvex_error *err)
{
    const unsigned char *reasoning_end;
    size_t delimiter_count;

    if (view) memset(view, 0, sizeof(*view));
    if (!tokenizer || !tokenizer->plan.sealed ||
        !conversation_admitted(tokenizer) ||
        !view || (!bytes && byte_count) || !valid_utf8(bytes, byte_count) ||
        policy > YVEX_REASONING_MAXIMUM) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.output.grammar",
                       "admitted policy and valid completion bytes are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (policy == YVEX_REASONING_DISABLED) {
        view->final.bytes = bytes;
        view->final.count = byte_count;
    } else {
        reasoning_end = find_bytes(bytes, byte_count,
                                   tokenizer->conversation->thinking_end);
        if (!reasoning_end) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.output.grammar",
                           "thinking completion is missing its source delimiter");
            return YVEX_ERR_FORMAT;
        }
        delimiter_count = strlen(tokenizer->conversation->thinking_end);
        view->reasoning.bytes = bytes;
        view->reasoning.count = (unsigned long long)(reasoning_end - bytes);
        view->final.bytes = reasoning_end + delimiter_count;
        view->final.count = byte_count - view->reasoning.count - delimiter_count;
    }
    if (!completion_special_valid(tokenizer->conversation, view)) {
        memset(view, 0, sizeof(*view));
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.output.grammar",
                       "completion contains a source control token in content");
        return YVEX_ERR_FORMAT;
    }
    view->reasoning_complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int parameter_name_seen(const provider_builder *names,
                               const char *name)
{
    unsigned long long offset = 0u;
    while (offset < names->count) {
        const char *candidate = (const char *)names->data + offset;
        size_t count = strlen(candidate);
        if (strcmp(candidate, name) == 0) return 1;
        offset += count + 1u;
    }
    return 0;
}

static int parse_invoke(const yvex_conversation_protocol *conversation,
                        const yvex_provider_request *request,
                        const unsigned char **position,
                        const unsigned char *end,
                        yvex_provider_tool_call *call, yvex_error *err)
{
    provider_builder arguments = {0};
    provider_builder parameter_names = {0};
    char parameter[YVEX_PROVIDER_TOOL_NAME_CAP];
    char string_kind[6];
    unsigned long long parameter_count = 0u;
    const unsigned char *cursor = *position;
    int rc = YVEX_OK;
    if (!consume(&cursor, end, conversation->tool_invoke_start) ||
        !attribute(&cursor, end, conversation->tool_invoke_name_end,
                   call->name, sizeof(call->name)) ||
        !tool_admitted(request, call->name) ||
        literal(&arguments, "{", err) != YVEX_OK)
        rc = YVEX_ERR_FORMAT;
    while (rc == YVEX_OK && cursor < end &&
           !starts_with(cursor, end, conversation->tool_invoke_end)) {
        const unsigned char *value, *close;
        size_t parameter_size;
        if (!consume(&cursor, end, conversation->tool_parameter_start) ||
            !attribute(&cursor, end,
                       conversation->tool_parameter_name_end, parameter,
                       sizeof(parameter)) ||
            !attribute(&cursor, end,
                       conversation->tool_parameter_kind_end, string_kind,
                       sizeof(string_kind))) {
            rc = YVEX_ERR_FORMAT;
            break;
        }
        value = cursor;
        close = find_bytes(cursor, (unsigned long long)(end - cursor),
                           conversation->tool_parameter_end);
        if (!close || (strcmp(string_kind, "true") != 0 &&
                       strcmp(string_kind, "false") != 0)) {
            rc = YVEX_ERR_FORMAT;
            break;
        }
        parameter_size = strlen(parameter) + 1u;
        if (parameter_name_seen(&parameter_names, parameter) ||
            parameter_names.count > YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES -
                                        parameter_size) {
            rc = YVEX_ERR_FORMAT;
            break;
        }
        rc = append(&parameter_names, parameter, parameter_size, err);
        if (rc == YVEX_OK && parameter_count++)
            rc = literal(&arguments, ", ", err);
        if (rc == YVEX_OK)
            rc = json_string(&arguments, (const unsigned char *)parameter,
                             strlen(parameter), err);
        if (rc == YVEX_OK) rc = literal(&arguments, ": ", err);
        if (rc == YVEX_OK && strcmp(string_kind, "true") == 0)
            rc = json_string(&arguments, value,
                             (unsigned long long)(close - value), err);
        else if (rc == YVEX_OK) {
            if (yvex_provider_json_value_validate(
                    value, (unsigned long long)(close - value), 0, err) != YVEX_OK)
                rc = YVEX_ERR_FORMAT;
            else
                rc = append(&arguments, value,
                            (unsigned long long)(close - value), err);
        }
        cursor = close;
        if (rc == YVEX_OK &&
            !consume(&cursor, end, conversation->tool_parameter_end))
            rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) rc = literal(&arguments, "}", err);
    if (rc == YVEX_OK &&
        !consume(&cursor, end, conversation->tool_invoke_end))
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK && yvex_provider_json_value_validate(
            arguments.data, arguments.count, 1, err) != YVEX_OK)
        rc = YVEX_ERR_FORMAT;
    if (rc != YVEX_OK) {
        free(arguments.data);
        free(parameter_names.data);
        if (rc == YVEX_ERR_NOMEM) return rc;
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.provider.tool",
                       "completion contains a malformed source-authored tool call");
        return YVEX_ERR_FORMAT;
    }
    free(parameter_names.data);
    call->arguments_json.bytes = arguments.data;
    call->arguments_json.count = arguments.count;
    *position = cursor;
    return YVEX_OK;
}

static int parse_calls(const yvex_conversation_protocol *conversation,
                       const yvex_provider_request *request,
                       const unsigned char *cursor, const unsigned char *end,
                       yvex_provider_tool_call **calls,
                       unsigned long long *call_count, yvex_error *err)
{
    yvex_provider_tool_call *owned;
    unsigned long long count = 0u;
    int rc = YVEX_OK;

    *calls = NULL;
    *call_count = 0u;
    if (!consume(&cursor, end, conversation->tool_calls_start))
        rc = YVEX_ERR_FORMAT;
    owned = calloc(YVEX_PROVIDER_MAX_TOOLS, sizeof(*owned));
    if (!owned) return YVEX_ERR_NOMEM;
    while (rc == YVEX_OK && cursor < end &&
           !starts_with(cursor, end, conversation->tool_calls_end)) {
        if (count >= YVEX_PROVIDER_MAX_TOOLS)
            rc = YVEX_ERR_BOUNDS;
        else
            rc = parse_invoke(conversation, request, &cursor, end,
                              &owned[count], err);
        if (rc == YVEX_OK) ++count;
        if (rc == YVEX_OK &&
            (cursor >= end || cursor[0] != '\n'))
            rc = YVEX_ERR_FORMAT;
        if (rc == YVEX_OK) ++cursor;
    }
    if (rc == YVEX_OK &&
        (!count || !consume(&cursor, end, conversation->tool_calls_end) ||
         cursor != end))
        rc = YVEX_ERR_FORMAT;
    if (rc != YVEX_OK) {
        unsigned long long index;
        for (index = 0u; index < count; ++index)
            free((void *)owned[index].arguments_json.bytes);
        free(owned);
        if (!yvex_error_code(err))
            yvex_error_set(err, rc, "tokenizer.provider.tool",
                           "completion tool block is not exact source grammar");
        return rc;
    }
    *calls = owned;
    *call_count = count;
    return YVEX_OK;
}

/*
 * Parse exact completion channels and grounded DSML calls transactionally.
 *
 * Allocates content/call bytes and seals one output identity.
 */
int yvex_tokenizer_parse_provider_completion(
    const yvex_tokenizer *tokenizer, const yvex_provider_request *request,
    const unsigned char *bytes, unsigned long long byte_count,
    yvex_tokenizer_provider_result *result, yvex_error *err)
{
    yvex_tokenizer_provider_result candidate = {0};
    conversation_output_view view;
    const unsigned char *tool;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long call;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!result || !tokenizer ||
        tokenizer->plan.prompt_policy != YVEX_TOKENIZER_PROMPT_CONVERSATION ||
        yvex_provider_request_validate(request, err) != YVEX_OK ||
        (!bytes && byte_count) || !valid_utf8(bytes, byte_count)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.provider.output",
                       "sealed request and valid UTF-8 completion are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = conversation_output_parse(
        tokenizer, request->reasoning_policy, bytes, byte_count, &view, err);
    if (rc != YVEX_OK) return rc;
    candidate.schema_version = YVEX_TOKENIZER_PROVIDER_RESULT_SCHEMA_V2;
    tool = find_bytes(view.final.bytes, view.final.count,
                      tokenizer->conversation->tool_calls_start);
    candidate.reasoning_content_count = view.reasoning.count;
    candidate.content_count = tool
                                  ? (unsigned long long)(tool - view.final.bytes)
                                  : view.final.count;
    if (candidate.reasoning_content_count) {
        candidate.reasoning_content = malloc(
            (size_t)candidate.reasoning_content_count);
        if (!candidate.reasoning_content) rc = YVEX_ERR_NOMEM;
        else memcpy(candidate.reasoning_content, view.reasoning.bytes,
                    (size_t)candidate.reasoning_content_count);
    }
    if (candidate.content_count) {
        candidate.content = malloc((size_t)candidate.content_count);
        if (!candidate.content) rc = YVEX_ERR_NOMEM;
        else memcpy(candidate.content, view.final.bytes,
                    (size_t)candidate.content_count);
    }
    if (rc == YVEX_OK &&
        find_bytes(candidate.content, candidate.content_count,
                   tokenizer->conversation->dsml))
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK && tool) {
        candidate.kind = YVEX_PROVIDER_OUTPUT_FUNCTION_CALL;
        rc = parse_calls(tokenizer->conversation, request, tool,
                         view.final.bytes + view.final.count,
                         &candidate.tool_calls, &candidate.tool_call_count, err);
    } else if (rc == YVEX_OK) {
        candidate.kind = YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT;
        if (find_bytes(view.final.bytes, view.final.count,
                       tokenizer->conversation->dsml))
            rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.provider.output.v2") ||
            !yvex_sha256_update_text(&hash, request->request_identity) ||
            !yvex_sha256_update_u64_be(&hash, candidate.kind) ||
            !yvex_sha256_update_u64_be(
                &hash, candidate.reasoning_content_count) ||
            !yvex_sha256_update(
                &hash, candidate.reasoning_content,
                (size_t)candidate.reasoning_content_count) ||
            !yvex_sha256_update_u64_be(&hash, candidate.content_count) ||
            !yvex_sha256_update(&hash, candidate.content,
                                (size_t)candidate.content_count) ||
            !yvex_sha256_update_u64_be(&hash, candidate.tool_call_count))
            rc = YVEX_ERR_STATE;
        for (call = 0u; rc == YVEX_OK &&
                        call < candidate.tool_call_count; ++call)
            if (!yvex_sha256_update_text(&hash,
                                         candidate.tool_calls[call].name) ||
                !yvex_sha256_update_u64_be(
                    &hash, candidate.tool_calls[call].arguments_json.count) ||
                !yvex_sha256_update(
                    &hash, candidate.tool_calls[call].arguments_json.bytes,
                    (size_t)candidate.tool_calls[call].arguments_json.count))
                rc = YVEX_ERR_STATE;
        if (rc == YVEX_OK && !yvex_sha256_final(&hash, digest))
            rc = YVEX_ERR_STATE;
        if (rc == YVEX_OK) {
            yvex_sha256_hex(digest, candidate.output_identity);
            for (call = 0u; call < candidate.tool_call_count; ++call)
                (void)snprintf(candidate.tool_calls[call].call_id,
                               sizeof(candidate.tool_calls[call].call_id),
                               "call_%.20s_%llu", candidate.output_identity,
                               call);
            candidate.completed = 1;
        }
    }
    if (rc != YVEX_OK) {
        yvex_tokenizer_provider_result_clear(&candidate);
        if (!yvex_error_code(err))
            yvex_error_set(err, rc, "tokenizer.provider.output",
                           "provider completion parsing failed");
        return rc;
    }
    *result = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_tokenizer_provider_result_clear(yvex_tokenizer_provider_result *result)
{
    if (!result) return;
    free(result->reasoning_content);
    free(result->content);
    if (result->tool_calls) {
        unsigned long long index;
        for (index = 0u; index < result->tool_call_count; ++index)
            free((void *)result->tool_calls[index].arguments_json.bytes);
    }
    free(result->tool_calls);
    memset(result, 0, sizeof(*result));
}
