/* Owner: tokenizer.provider.
 * Owns: provider-message projection into the admitted prompt policy and exact typed completion parsing.
 * Does not own: HTTP/OpenAI syntax, provider request admission, generation, sessions, or tool execution.
 * Invariants: application messages reach model syntax only through the tokenizer plan; prose is never a tool call.
 * Boundary: bridges transport-neutral provider facts to artifact-bound tokenizer prompt/completion semantics.
 * Purpose: execute the pinned DeepSeek DSML tool and response-format contract without gateway prompt assembly.
 * Inputs: sealed provider requests, immutable tokenizer plan, explicit completion bytes, and bounded outputs.
 * Effects: allocates transactional rendered/encoded results and typed parsed completion storage.
 * Failure: unsupported plans or malformed DSML publish no prompt, tool call, or partial identity. */

#include "src/tokenizer/private.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>

#define PROVIDER_RESULT_SCHEMA_V1 1u
#define DSML_TOKEN "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c"

static const char ds_bos[] =
    "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>";
static const char ds_eos[] =
    "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>";
static const char ds_user[] = "<\xef\xbd\x9cUser\xef\xbd\x9c>";
static const char ds_assistant[] = "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>";
static const char ds_dsml[] = DSML_TOKEN;

static const char tools_intro[] =
    "## Tools\n\n"
    "You have access to tools that can help answer the user's question. Invoke "
    "them by writing a \"<" DSML_TOKEN "tool_calls>\" block like this:\n\n"
    "<" DSML_TOKEN "tool_calls>\n"
    "<" DSML_TOKEN "invoke name=\"$TOOL_NAME\">\n"
    "<" DSML_TOKEN "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">"
    "$PARAMETER_VALUE</" DSML_TOKEN "parameter>\n"
    "...\n"
    "</" DSML_TOKEN "invoke>\n"
    "<" DSML_TOKEN "invoke name=\"$TOOL_NAME2\">\n"
    "...\n"
    "</" DSML_TOKEN "invoke>\n"
    "</" DSML_TOKEN "tool_calls>\n\n"
    "String parameters should be specified as is and set `string=\"true\"`. "
    "For all other types (numbers, booleans, arrays, objects), pass the value in "
    "JSON format and set `string=\"false\"`.\n\n"
    "If thinking_mode is enabled (triggered by <think>), you MUST output your "
    "complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n"
    "Otherwise, output directly after </think> with tool calls or final response.\n\n"
    "### Available Tool Schemas\n\n";

static const char tools_outro[] =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.";

typedef struct {
    unsigned char *data;
    unsigned long long count, capacity;
} provider_builder;

/* Purpose: reserve one checked transactional provider prompt extent.
 * Inputs: builder, requested additional bytes, and error output.
 * Effects: grows unique prompt storage while preserving the admitted prefix.
 * Failure: retains prior ownership on bounds or allocation failure.
 * Boundary: storage growth contains no prompt-policy decision. */
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

/* Purpose: append one exact byte span after complete capacity admission. */
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

/* Purpose: append one terminated literal without making it a semantic length authority. */
static int literal(provider_builder *builder, const char *text, yvex_error *err)
{
    return append(builder, text, (unsigned long long)strlen(text), err);
}

/* Purpose: render one JSON string in deterministic ensure-ascii-false form.
 * Inputs: builder, valid explicit bytes/count, and error output.
 * Effects: appends quotes and required JSON escapes.
 * Failure: returns the first builder failure without publishing a prompt.
 * Boundary: preserves UTF-8 bytes and performs no normalization. */
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

/* Purpose: confirm one explicit span is complete UTF-8 under the tokenizer decoder. */
static int valid_utf8(const unsigned char *bytes, unsigned long long count)
{
    unsigned long long offset = 0u;
    uint32_t point;
    if (!bytes && count) return 0;
    while (offset < count)
        if (!yvex_tokenizer_utf8_next(bytes, count, &offset, &point)) return 0;
    return 1;
}

/* Purpose: append exact DSML parameters projected from one JSON argument object.
 * Inputs: builder, valid arguments-object bytes, and error output.
 * Effects: parses members and appends ordered typed DSML parameter elements.
 * Failure: refuses malformed, nested-ambiguous, or unsupported JSON values.
 * Boundary: projection is tokenizer-owned and never executes a function. */
static int append_arguments(provider_builder *builder, yvex_provider_span arguments,
                            yvex_error *err)
{
    yvex_json json;
    yvex_json_iter iter;
    yvex_json_item item;
    char key[YVEX_PROVIDER_TOOL_NAME_CAP];
    if (arguments.count > SIZE_MAX) return YVEX_ERR_BOUNDS;
    yvex_json_init(&json, (const char *)arguments.bytes, (size_t)arguments.count);
    if (!yvex_json_iter_begin(&json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        const char *start;
        unsigned char *decoded = NULL;
        size_t decoded_capacity;
        int is_string;
        yvex_json_space(&json);
        start = json.cursor;
        is_string = start < json.end && *start == '"';
        if (literal(builder, "<", err) != YVEX_OK ||
            literal(builder, ds_dsml, err) != YVEX_OK ||
            literal(builder, "parameter name=\"", err) != YVEX_OK ||
            literal(builder, key, err) != YVEX_OK ||
            literal(builder, is_string ? "\" string=\"true\">" :
                                        "\" string=\"false\">", err) != YVEX_OK)
            return yvex_error_code(err);
        if (is_string) {
            decoded_capacity = (size_t)(json.end - json.cursor) + 1u;
            decoded = malloc(decoded_capacity);
            if (!decoded) return YVEX_ERR_NOMEM;
            if (!yvex_json_string(&json, (char *)decoded, decoded_capacity)) {
                free(decoded);
                return YVEX_ERR_FORMAT;
            }
            if (append(builder, decoded, strlen((char *)decoded), err) != YVEX_OK) {
                free(decoded);
                return yvex_error_code(err);
            }
            free(decoded);
        } else {
            if (!yvex_json_skip_value(&json) ||
                append(builder, start, (unsigned long long)(json.cursor - start),
                       err) != YVEX_OK)
                return YVEX_ERR_FORMAT;
        }
        if (literal(builder, "</", err) != YVEX_OK ||
            literal(builder, ds_dsml, err) != YVEX_OK ||
            literal(builder, "parameter>\n", err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
                   yvex_json_complete(&json)
               ? YVEX_OK
               : YVEX_ERR_FORMAT;
}

/* Purpose: append one exact assistant DSML tool-call block. */
static int append_tool_call(provider_builder *builder,
                            const yvex_provider_tool_call *call,
                            yvex_error *err)
{
    int rc = literal(builder, "\n\n<", err);
    if (rc == YVEX_OK) rc = literal(builder, ds_dsml, err);
    if (rc == YVEX_OK) rc = literal(builder, "tool_calls>\n<", err);
    if (rc == YVEX_OK) rc = literal(builder, ds_dsml, err);
    if (rc == YVEX_OK) rc = literal(builder, "invoke name=\"", err);
    if (rc == YVEX_OK) rc = literal(builder, call->name, err);
    if (rc == YVEX_OK) rc = literal(builder, "\">\n", err);
    if (rc == YVEX_OK) rc = append_arguments(builder, call->arguments_json, err);
    if (rc == YVEX_OK) rc = literal(builder, "</", err);
    if (rc == YVEX_OK) rc = literal(builder, ds_dsml, err);
    if (rc == YVEX_OK) rc = literal(builder, "invoke>\n</", err);
    if (rc == YVEX_OK) rc = literal(builder, ds_dsml, err);
    if (rc == YVEX_OK) rc = literal(builder, "tool_calls>", err);
    return rc;
}

/* Purpose: render one canonical tool schema line owned by the prompt adapter. */
static int append_tool_schema(provider_builder *builder,
                              const yvex_provider_function_tool *tool,
                              yvex_error *err)
{
    int rc = literal(builder, "{\"name\": ", err);
    if (rc == YVEX_OK)
        rc = json_string(builder, (const unsigned char *)tool->name,
                         strlen(tool->name), err);
    if (rc == YVEX_OK) rc = literal(builder, ", \"description\": ", err);
    if (rc == YVEX_OK)
        rc = json_string(builder, tool->description.bytes,
                         tool->description.count, err);
    if (rc == YVEX_OK) rc = literal(builder, ", \"parameters\": ", err);
    if (rc == YVEX_OK)
        rc = append(builder, tool->parameters_json.bytes,
                    tool->parameters_json.count, err);
    if (rc == YVEX_OK) rc = literal(builder, "}", err);
    return rc;
}

/* Purpose: append exact DeepSeek tools and response-format control material.
 * Inputs: builder, sealed provider request, and error output.
 * Effects: appends schemas/choice controls and bounded JSON-object instruction.
 * Failure: returns the first schema/rendering failure without a partial prompt result.
 * Boundary: only executable facts admitted by the pinned family policy are projected. */
static int append_controls(provider_builder *builder,
                           const yvex_provider_request *request,
                           yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    if (request->tool_count) {
        rc = literal(builder, "\n\n", err);
        if (rc == YVEX_OK) rc = literal(builder, tools_intro, err);
        for (index = 0u; rc == YVEX_OK && index < request->tool_count; ++index) {
            if (index) rc = literal(builder, "\n", err);
            if (rc == YVEX_OK)
                rc = append_tool_schema(builder, &request->tools[index], err);
        }
        if (rc == YVEX_OK) rc = literal(builder, tools_outro, err);
    }
    if (rc == YVEX_OK &&
        request->response_format == YVEX_PROVIDER_RESPONSE_JSON_OBJECT)
        rc = literal(builder,
                     "\n\n## Response Format:\n\nYou MUST strictly adhere to the "
                     "following schema to reply:\n{\"type\": \"json_object\"}",
                     err);
    return rc;
}

/* Purpose: render the sealed provider request through the exact admitted DeepSeek prompt policy.
 * Inputs: immutable tokenizer, sealed provider request, rendered output, and error output.
 * Effects: allocates exact prompt bytes and field-wise prompt/message identities.
 * Failure: frees candidate bytes and publishes no rendered prompt.
 * Boundary: family prompt syntax lives here, outside HTTP and generation owners. */
int yvex_tokenizer_provider_prompt(
    const yvex_tokenizer *tokenizer, const yvex_provider_request *request,
    yvex_rendered_prompt *rendered, yvex_error *err)
{
    provider_builder builder = {0};
    unsigned long long index, controls_at = ULLONG_MAX;
    yvex_provider_role prior = YVEX_PROVIDER_ROLE_SYSTEM;
    int user_group = 0, rc;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!tokenizer || !rendered ||
        tokenizer->plan.prompt_policy != YVEX_TOKENIZER_PROMPT_DEEPSEEK_V4 ||
        yvex_provider_request_validate(request, err) != YVEX_OK)
        return YVEX_ERR_INVALID_ARG;
    memset(rendered, 0, sizeof(*rendered));
    for (index = 0u; index < request->message_count; ++index)
        if (request->messages[index].role == YVEX_PROVIDER_ROLE_SYSTEM ||
            request->messages[index].role == YVEX_PROVIDER_ROLE_DEVELOPER) {
            controls_at = index;
            break;
        }
    rc = literal(&builder, ds_bos, err);
    if (rc == YVEX_OK && controls_at == ULLONG_MAX)
        rc = append_controls(&builder, request, err);
    for (index = 0u; rc == YVEX_OK && index < request->message_count; ++index) {
        const yvex_provider_message *message = &request->messages[index];
        if (!valid_utf8(message->content.bytes, message->content.count)) {
            rc = YVEX_ERR_FORMAT;
            break;
        }
        if (message->role == YVEX_PROVIDER_ROLE_SYSTEM) {
            if (index != 0u) { rc = YVEX_ERR_FORMAT; break; }
            rc = append(&builder, message->content.bytes,
                        message->content.count, err);
            user_group = 0;
        } else if (message->role == YVEX_PROVIDER_ROLE_DEVELOPER) {
            rc = literal(&builder, ds_user, err);
            if (rc == YVEX_OK)
                rc = append(&builder, message->content.bytes,
                            message->content.count, err);
            user_group = 1;
        } else if (message->role == YVEX_PROVIDER_ROLE_USER ||
                   message->role == YVEX_PROVIDER_ROLE_TOOL) {
            if (!user_group) rc = literal(&builder, ds_user, err);
            else rc = literal(&builder, "\n\n", err);
            if (rc == YVEX_OK && message->role == YVEX_PROVIDER_ROLE_TOOL)
                rc = literal(&builder, "<tool_result>", err);
            if (rc == YVEX_OK)
                rc = append(&builder, message->content.bytes,
                            message->content.count, err);
            if (rc == YVEX_OK && message->role == YVEX_PROVIDER_ROLE_TOOL)
                rc = literal(&builder, "</tool_result>", err);
            user_group = 1;
        } else if (message->role == YVEX_PROVIDER_ROLE_ASSISTANT) {
            if (prior != YVEX_PROVIDER_ROLE_USER &&
                prior != YVEX_PROVIDER_ROLE_TOOL &&
                prior != YVEX_PROVIDER_ROLE_DEVELOPER) {
                rc = YVEX_ERR_FORMAT;
                break;
            }
            rc = literal(&builder, ds_assistant, err);
            if (rc == YVEX_OK) rc = literal(&builder, "</think>", err);
            if (rc == YVEX_OK)
                rc = append(&builder, message->content.bytes,
                            message->content.count, err);
            if (rc == YVEX_OK && message->tool_call_count)
                rc = append_tool_call(&builder, &message->tool_calls[0], err);
            if (rc == YVEX_OK) rc = literal(&builder, ds_eos, err);
            user_group = 0;
        } else {
            rc = YVEX_ERR_FORMAT;
        }
        if (rc == YVEX_OK && index == controls_at)
            rc = append_controls(&builder, request, err);
        prior = message->role;
    }
    if (rc == YVEX_OK && prior != YVEX_PROVIDER_ROLE_USER &&
        prior != YVEX_PROVIDER_ROLE_TOOL && prior != YVEX_PROVIDER_ROLE_DEVELOPER)
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK) rc = literal(&builder, ds_assistant, err);
    if (rc == YVEX_OK) rc = literal(&builder, "</think>", err);
    if (rc != YVEX_OK) {
        free(builder.data);
        yvex_error_set(err, rc, "tokenizer.provider.prompt",
                       "provider messages do not satisfy DeepSeek prompt semantics");
        return rc;
    }
    rendered->text = (char *)builder.data;
    rendered->len = builder.count;
    rendered->generation_prompt = 1;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.provider.messages.v1") ||
        !yvex_sha256_update_text(&hash, request->request_identity) ||
        !yvex_sha256_final(&hash, digest)) goto identity_failure;
    yvex_sha256_hex(digest, rendered->message_sequence_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.provider.rendered.v1") ||
        !yvex_sha256_update_u64_be(&hash, rendered->len) ||
        !yvex_sha256_update(&hash, rendered->text, (size_t)rendered->len) ||
        !yvex_sha256_final(&hash, digest)) goto identity_failure;
    yvex_sha256_hex(digest, rendered->rendered_bytes_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.provider.prompt.v1") ||
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

/* Purpose: render and encode one sealed provider request through the artifact tokenizer.
 * Inputs: tokenizer/request/options, rendered/encoded outputs, and error output.
 * Effects: publishes exact rendered bytes and complete ordered token IDs.
 * Failure: clears both outputs if rendering or encoding fails.
 * Boundary: encoding reuses the admitted tokenizer and never opens source sidecars. */
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

/* Purpose: find one exact byte marker in an explicit completion span. */
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

/* Purpose: compare and consume one exact parser literal. */
static int consume(const unsigned char **cursor, const unsigned char *end,
                   const char *text)
{
    size_t count = strlen(text);
    if (!cursor || !*cursor || (size_t)(end - *cursor) < count ||
        memcmp(*cursor, text, count) != 0) return 0;
    *cursor += count;
    return 1;
}

/* Purpose: copy one bounded attribute up to its exact delimiter. */
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

/* Purpose: locate one admitted function name without treating prose as a call. */
static int tool_admitted(const yvex_provider_request *request, const char *name)
{
    unsigned long long index;
    for (index = 0u; index < request->tool_count; ++index)
        if (strcmp(request->tools[index].name, name) == 0) return 1;
    return 0;
}

/* Purpose: parse one exact single-call DSML block into canonical JSON arguments.
 * Inputs: sealed request, exact completion range, call output, and error output.
 * Effects: allocates one validated arguments object and records an admitted function name.
 * Failure: frees arguments and publishes no typed call for malformed/unlisted DSML.
 * Boundary: syntax recognition only; arbitrary prose is never interpreted as a call. */
static int parse_call(const yvex_provider_request *request,
                      const unsigned char *cursor, const unsigned char *end,
                      yvex_provider_tool_call *call, yvex_error *err)
{
    provider_builder arguments = {0};
    char parameter[YVEX_PROVIDER_TOOL_NAME_CAP];
    char string_kind[6];
    unsigned long long parameter_count = 0u;
    int rc = YVEX_OK;
    if (!consume(&cursor, end, "\n\n<") || !consume(&cursor, end, ds_dsml) ||
        !consume(&cursor, end, "tool_calls>\n<") ||
        !consume(&cursor, end, ds_dsml) ||
        !consume(&cursor, end, "invoke name=\"") ||
        !attribute(&cursor, end, "\">\n", call->name, sizeof(call->name)) ||
        !tool_admitted(request, call->name) ||
        literal(&arguments, "{", err) != YVEX_OK)
        rc = YVEX_ERR_FORMAT;
    while (rc == YVEX_OK && cursor < end &&
           !((size_t)(end - cursor) >= 2u && cursor[0] == '<' && cursor[1] == '/')) {
        const unsigned char *value, *close;
        if (!consume(&cursor, end, "<") || !consume(&cursor, end, ds_dsml) ||
            !consume(&cursor, end, "parameter name=\"") ||
            !attribute(&cursor, end, "\" string=\"", parameter,
                       sizeof(parameter)) ||
            !attribute(&cursor, end, "\">", string_kind,
                       sizeof(string_kind))) {
            rc = YVEX_ERR_FORMAT;
            break;
        }
        value = cursor;
        close = find_bytes(cursor, (unsigned long long)(end - cursor), "</");
        if (!close || (strcmp(string_kind, "true") != 0 &&
                       strcmp(string_kind, "false") != 0)) {
            rc = YVEX_ERR_FORMAT;
            break;
        }
        if (parameter_count++) rc = literal(&arguments, ", ", err);
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
            (!consume(&cursor, end, "</") || !consume(&cursor, end, ds_dsml) ||
             !consume(&cursor, end, "parameter>\n")))
            rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) rc = literal(&arguments, "}", err);
    if (rc == YVEX_OK &&
        (!consume(&cursor, end, "</") || !consume(&cursor, end, ds_dsml) ||
         !consume(&cursor, end, "invoke>\n</") ||
         !consume(&cursor, end, ds_dsml) ||
         !consume(&cursor, end, "tool_calls>") || cursor != end))
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK && yvex_provider_json_value_validate(
            arguments.data, arguments.count, 1, err) != YVEX_OK)
        rc = YVEX_ERR_FORMAT;
    if (rc != YVEX_OK) {
        free(arguments.data);
        yvex_error_set(err, YVEX_ERR_FORMAT, "tokenizer.provider.tool",
                       "completion does not contain one exact admitted DSML tool call");
        return rc;
    }
    call->arguments_json.bytes = arguments.data;
    call->arguments_json.count = arguments.count;
    return YVEX_OK;
}

/* Purpose: parse exact completion text or a grounded single DSML call transactionally.
 * Inputs: tokenizer/request, valid UTF-8 completion bytes, result output, and error output.
 * Effects: allocates content/call bytes and seals one output identity.
 * Failure: clears the candidate and publishes no partial typed result.
 * Boundary: parsing does not execute tools or alter generation/session state. */
int yvex_tokenizer_parse_provider_completion(
    const yvex_tokenizer *tokenizer, const yvex_provider_request *request,
    const unsigned char *bytes, unsigned long long byte_count,
    yvex_tokenizer_provider_result *result, yvex_error *err)
{
    static const char marker[] = "\n\n<" DSML_TOKEN "tool_calls>";
    yvex_tokenizer_provider_result candidate = {0};
    const unsigned char *tool;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!result || !tokenizer ||
        tokenizer->plan.prompt_policy != YVEX_TOKENIZER_PROMPT_DEEPSEEK_V4 ||
        yvex_provider_request_validate(request, err) != YVEX_OK ||
        (!bytes && byte_count) || !valid_utf8(bytes, byte_count)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tokenizer.provider.output",
                       "sealed request and valid UTF-8 completion are required");
        return YVEX_ERR_INVALID_ARG;
    }
    candidate.schema_version = PROVIDER_RESULT_SCHEMA_V1;
    tool = find_bytes(bytes, byte_count, marker);
    candidate.content_count = tool ? (unsigned long long)(tool - bytes) : byte_count;
    if (candidate.content_count) {
        candidate.content = malloc((size_t)candidate.content_count);
        if (!candidate.content) rc = YVEX_ERR_NOMEM;
        else memcpy(candidate.content, bytes, (size_t)candidate.content_count);
    }
    if (rc == YVEX_OK && tool) {
        candidate.kind = YVEX_PROVIDER_OUTPUT_FUNCTION_CALL;
        rc = parse_call(request, tool, bytes + byte_count,
                        &candidate.tool_call, err);
    } else if (rc == YVEX_OK) {
        candidate.kind = YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT;
    }
    if (rc == YVEX_OK) {
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update_text(&hash, "yvex.tokenizer.provider.output.v1") ||
            !yvex_sha256_update_text(&hash, request->request_identity) ||
            !yvex_sha256_update_u64_be(&hash, candidate.kind) ||
            !yvex_sha256_update_u64_be(&hash, candidate.content_count) ||
            !yvex_sha256_update(&hash, candidate.content,
                                (size_t)candidate.content_count) ||
            !yvex_sha256_update_text(&hash, candidate.tool_call.name) ||
            !yvex_sha256_update_u64_be(
                &hash, candidate.tool_call.arguments_json.count) ||
            !yvex_sha256_update(
                &hash, candidate.tool_call.arguments_json.bytes,
                (size_t)candidate.tool_call.arguments_json.count) ||
            !yvex_sha256_final(&hash, digest))
            rc = YVEX_ERR_STATE;
        else {
            yvex_sha256_hex(digest, candidate.output_identity);
            if (candidate.kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL)
                (void)snprintf(candidate.tool_call.call_id,
                               sizeof(candidate.tool_call.call_id),
                               "call_%.24s", candidate.output_identity);
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

/* Purpose: release one tokenizer-owned provider completion result.
 * Inputs: result that may own content and arguments bytes.
 * Effects: frees both allocations and clears all evidence fields.
 * Failure: none; null and cleared results are accepted.
 * Boundary: call identifiers and provider requests remain caller-owned. */
void yvex_tokenizer_provider_result_clear(yvex_tokenizer_provider_result *result)
{
    if (!result) return;
    free(result->content);
    free((void *)result->tool_call.arguments_json.bytes);
    memset(result, 0, sizeof(*result));
}
