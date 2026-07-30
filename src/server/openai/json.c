/* Owner: server.openai.json.
 * Owns: strict bounded OpenAI-profile JSON request parsing into provider-neutral typed requests.
 * Does not own: HTTP framing, response rendering, model prompt syntax, sessions, or generation.
 * Invariants: duplicate/unknown/unsupported fields refuse; strings and arrays retain explicit bounded lengths.
 * Boundary: OpenAI object names end at the provider-neutral request contract.
 * Purpose: admit the exact Chat Completions and Responses v1 request subset without silent field loss.
 * Inputs: one complete HTTP body, endpoint profile, and selected runtime model identifier.
 * Effects: allocates one fully owned sealed provider request or none.
 * Failure: malformed types, duplicates, bounds, and unsupported semantics return typed refusals. */

#include "src/server/openai/private.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <yvex/internal/core.h>

#define JSON_SEEN_MAX 64u
#define JSON_SEEN_KEY 64u

typedef struct {
    char keys[JSON_SEEN_MAX][JSON_SEEN_KEY];
    unsigned int count;
} seen_keys;

/* Purpose: publish one parsing refusal while retaining the offending public parameter. */
static int json_refuse(yvex_error *err, yvex_status status,
                       const char *message)
{
    yvex_error_set(err, status, "server.openai.request", message);
    return status;
}

/* Purpose: reject duplicate object keys after JSON escape decoding. */
static int key_unique(seen_keys *seen, const char *key, yvex_error *err)
{
    unsigned int index;
    if (!seen || !key || strlen(key) >= JSON_SEEN_KEY ||
        seen->count >= JSON_SEEN_MAX)
        return json_refuse(err, YVEX_ERR_BOUNDS,
                           "JSON object has too many or oversized fields");
    for (index = 0u; index < seen->count; ++index)
        if (strcmp(seen->keys[index], key) == 0)
            return json_refuse(err, YVEX_ERR_FORMAT,
                               "duplicate JSON fields are refused");
    strcpy(seen->keys[seen->count++], key);
    return YVEX_OK;
}

/* Purpose: validate UTF-8 independently of tokenizer/model linkage.
 * Inputs: explicit bytes and extent.
 * Effects: none.
 * Failure: returns false for NUL, overlong, surrogate, truncated, or out-of-range input.
 * Boundary: validates encoding only and performs no Unicode normalization. */
static int utf8_valid(const unsigned char *bytes, size_t count)
{
    size_t index = 0u;
    while (index < count) {
        unsigned char first = bytes[index++];
        unsigned int need, point, minimum;
        if (first < 0x80u) {
            if (!first) return 0;
            continue;
        }
        if ((first & 0xe0u) == 0xc0u) {
            need = 1u; point = first & 0x1fu; minimum = 0x80u;
        } else if ((first & 0xf0u) == 0xe0u) {
            need = 2u; point = first & 0x0fu; minimum = 0x800u;
        } else if ((first & 0xf8u) == 0xf0u) {
            need = 3u; point = first & 0x07u; minimum = 0x10000u;
        } else return 0;
        if (index + need > count) return 0;
        while (need--) {
            unsigned char byte = bytes[index++];
            if ((byte & 0xc0u) != 0x80u) return 0;
            point = (point << 6u) | (byte & 0x3fu);
        }
        if (point < minimum || point > 0x10ffffu ||
            (point >= 0xd800u && point <= 0xdfffu)) return 0;
    }
    return 1;
}

/* Purpose: decode one bounded JSON string into uniquely owned UTF-8 bytes.
 * Inputs: parser cursor, span output, semantic maximum/empty policy, and error output.
 * Effects: allocates and publishes one decoded explicit-length span.
 * Failure: frees candidate storage and clears the span.
 * Boundary: JSON decoding does not apply tokenizer normalization. */
static int string_span(yvex_json *json, yvex_provider_span *span,
                       unsigned long long maximum, int allow_empty,
                       yvex_error *err)
{
    size_t capacity;
    char *decoded;
    memset(span, 0, sizeof(*span));
    if (!json || json->cursor >= json->end) return YVEX_ERR_FORMAT;
    capacity = (size_t)(json->end - json->cursor) + 1u;
    decoded = yvex_json_string_dup(json, capacity);
    if (!decoded) return json_refuse(err, YVEX_ERR_FORMAT,
                                     "invalid JSON string");
    span->count = (unsigned long long)strlen(decoded);
    if (span->count > maximum || (!allow_empty && !span->count) ||
        !utf8_valid((const unsigned char *)decoded, (size_t)span->count)) {
        free(decoded);
        memset(span, 0, sizeof(*span));
        return json_refuse(err, YVEX_ERR_BOUNDS,
                           "JSON string is empty, invalid UTF-8, or oversized");
    }
    span->bytes = (unsigned char *)decoded;
    return YVEX_OK;
}

/* Purpose: decode one bounded JSON string into fixed identifier storage. */
static int string_fixed(yvex_json *json, char *output, size_t capacity,
                        int allow_empty, yvex_error *err)
{
    if (!yvex_json_string(json, output, capacity) ||
        (!allow_empty && !output[0]))
        return json_refuse(err, YVEX_ERR_FORMAT,
                           "invalid or oversized JSON identifier");
    return YVEX_OK;
}

/* Purpose: parse one finite JSON number without implicit string coercion. */
static int number_double(yvex_json *json, double *output, yvex_error *err)
{
    char text[64], *end = NULL;
    double value;
    if (!yvex_json_number_text(json, text, sizeof(text)))
        return json_refuse(err, YVEX_ERR_FORMAT, "JSON number is required");
    errno = 0;
    value = strtod(text, &end);
    if (errno || !end || *end || !isfinite(value))
        return json_refuse(err, YVEX_ERR_FORMAT,
                           "finite JSON number is required");
    *output = value;
    return YVEX_OK;
}

/* Purpose: copy one complete raw JSON value for tokenizer-owned schema rendering.
 * Inputs: parser cursor, span output, bound/object policy, and error output.
 * Effects: allocates the exact admitted JSON byte slice.
 * Failure: releases invalid storage and publishes no span.
 * Boundary: validates syntax but does not interpret JSON-schema semantics. */
static int raw_value(yvex_json *json, yvex_provider_span *span,
                     unsigned long long maximum, int require_object,
                     yvex_error *err)
{
    const char *start;
    yvex_json_space(json);
    start = json->cursor;
    if (!start || (require_object &&
                   (start >= json->end || *start != '{')) ||
        !yvex_json_skip_value(json))
        return json_refuse(err, YVEX_ERR_FORMAT,
                           "complete JSON object is required");
    span->count = (unsigned long long)(json->cursor - start);
    if (!span->count || span->count > maximum) return YVEX_ERR_BOUNDS;
    span->bytes = malloc((size_t)span->count);
    if (!span->bytes) return YVEX_ERR_NOMEM;
    memcpy((void *)span->bytes, start, (size_t)span->count);
    if (yvex_provider_json_value_validate(span->bytes, span->count,
                                          require_object, err) != YVEX_OK) {
        free((void *)span->bytes);
        memset(span, 0, sizeof(*span));
        return yvex_error_code(err);
    }
    return YVEX_OK;
}

/* Purpose: parse one OpenAI function payload used by definitions or calls.
 * Inputs: object parser, cleared tool-call output, and error output.
 * Effects: allocates validated JSON arguments and records the exact function name.
 * Failure: refuses duplicates/unknown fields and leaves caller cleanup ownership explicit.
 * Boundary: parsing never dispatches or executes the named function. */
static int function_call(yvex_json *json, yvex_provider_tool_call *call,
                         yvex_error *err)
{
    yvex_json_iter iter;
    yvex_json_item item;
    seen_keys seen = {0};
    char key[JSON_SEEN_KEY];
    int name_seen = 0, arguments_seen = 0;
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (key_unique(&seen, key, err) != YVEX_OK) return yvex_error_code(err);
        if (strcmp(key, "name") == 0) {
            if (string_fixed(json, call->name, sizeof(call->name), 0, err) != YVEX_OK)
                return yvex_error_code(err);
            name_seen = 1;
        } else if (strcmp(key, "arguments") == 0) {
            if (string_span(json, &call->arguments_json,
                            YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES, 0, err) != YVEX_OK)
                return yvex_error_code(err);
            if (yvex_provider_json_value_validate(
                    call->arguments_json.bytes, call->arguments_json.count,
                    1, err) != YVEX_OK)
                return yvex_error_code(err);
            arguments_seen = 1;
        } else {
            return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                               "unsupported function-call field");
        }
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
                   name_seen && arguments_seen
               ? YVEX_OK
               : json_refuse(err, YVEX_ERR_FORMAT,
                             "complete function call is required");
}

/* Purpose: parse one OpenAI assistant tool-call object with one function only.
 * Inputs: object parser, cleared call output, and error output.
 * Effects: records exact call ID/type and delegates function payload admission.
 * Failure: refuses missing, duplicate, parallel, or non-function call forms.
 * Boundary: syntax translation only; call identity remains application-owned input. */
static int tool_call(yvex_json *json, yvex_provider_tool_call *call,
                     yvex_error *err)
{
    yvex_json_iter iter;
    yvex_json_item item;
    seen_keys seen = {0};
    char key[JSON_SEEN_KEY], type[32] = {0};
    int id_seen = 0, type_seen = 0, function_seen = 0;
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (key_unique(&seen, key, err) != YVEX_OK) return yvex_error_code(err);
        if (strcmp(key, "id") == 0) {
            if (string_fixed(json, call->call_id, sizeof(call->call_id), 0, err) != YVEX_OK)
                return yvex_error_code(err);
            id_seen = 1;
        } else if (strcmp(key, "type") == 0) {
            if (string_fixed(json, type, sizeof(type), 0, err) != YVEX_OK)
                return yvex_error_code(err);
            type_seen = 1;
        } else if (strcmp(key, "function") == 0) {
            if (function_call(json, call, err) != YVEX_OK)
                return yvex_error_code(err);
            function_seen = 1;
        } else {
            return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                               "unsupported tool-call field");
        }
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator && id_seen &&
                   type_seen && function_seen && strcmp(type, "function") == 0
               ? YVEX_OK
               : json_refuse(err, YVEX_ERR_FORMAT,
                             "one typed function call is required");
}

/* Purpose: parse the admitted one-call assistant tool_calls array.
 * Inputs: array parser, target message, and error output.
 * Effects: allocates at most one typed tool call on the message.
 * Failure: refuses empty or parallel arrays and preserves explicit cleanup ownership.
 * Boundary: implements the profile's single-call limit only. */
static int tool_calls(yvex_json *json, yvex_provider_message *message,
                      yvex_error *err)
{
    yvex_json_iter iter;
    yvex_json_item item;
    yvex_provider_tool_call *call = NULL;
    unsigned long long count = 0u;
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        if (count++) return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                        "parallel tool calls are unsupported");
        call = calloc(1u, sizeof(*call));
        if (!call) return YVEX_ERR_NOMEM;
        message->tool_calls = call;
        message->tool_call_count = 1u;
        if (tool_call(json, call, err) != YVEX_OK) return yvex_error_code(err);
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator && count == 1u
               ? YVEX_OK
               : json_refuse(err, YVEX_ERR_FORMAT,
                             "one tool call is required");
}

/* Purpose: map one admitted role string into the provider-neutral role enum.
 * Inputs: terminated role label and enum output. Effects: writes one known role.
 * Failure: returns false for unknown labels.
 * Boundary: role mapping introduces no model-specific prompt markers. */
static int role_parse(const char *role, yvex_provider_role *output)
{
    if (strcmp(role, "developer") == 0) *output = YVEX_PROVIDER_ROLE_DEVELOPER;
    else if (strcmp(role, "system") == 0) *output = YVEX_PROVIDER_ROLE_SYSTEM;
    else if (strcmp(role, "user") == 0) *output = YVEX_PROVIDER_ROLE_USER;
    else if (strcmp(role, "assistant") == 0) *output = YVEX_PROVIDER_ROLE_ASSISTANT;
    else if (strcmp(role, "tool") == 0) *output = YVEX_PROVIDER_ROLE_TOOL;
    else return 0;
    return 1;
}

/* Purpose: parse one text-only Chat message without concatenating prompt syntax.
 * Inputs: message object parser, cleared provider message, and error output.
 * Effects: allocates content/tool-call bytes and records one typed role.
 * Failure: refuses unsupported content kinds or inconsistent role fields.
 * Boundary: model prompt markers remain tokenizer/family-owned. */
static int message_parse(yvex_json *json, yvex_provider_message *message,
                         yvex_error *err)
{
    yvex_json_iter iter;
    yvex_json_item item;
    seen_keys seen = {0};
    char key[JSON_SEEN_KEY], role[32] = {0};
    int role_seen = 0, content_seen = 0;
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (key_unique(&seen, key, err) != YVEX_OK) return yvex_error_code(err);
        if (strcmp(key, "role") == 0) {
            if (string_fixed(json, role, sizeof(role), 0, err) != YVEX_OK ||
                !role_parse(role, &message->role))
                return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                   "unsupported message role");
            role_seen = 1;
        } else if (strcmp(key, "content") == 0) {
            yvex_json_space(json);
            if (json->cursor < json->end && *json->cursor == 'n') {
                if (!yvex_json_skip_value(json)) return YVEX_ERR_FORMAT;
            } else if (string_span(json, &message->content,
                                   YVEX_PROVIDER_MAX_MESSAGE_BYTES, 1,
                                   err) != YVEX_OK) {
                return yvex_error_code(err);
            }
            content_seen = 1;
        } else if (strcmp(key, "tool_call_id") == 0) {
            if (string_fixed(json, message->tool_call_id,
                             sizeof(message->tool_call_id), 0, err) != YVEX_OK)
                return yvex_error_code(err);
        } else if (strcmp(key, "tool_calls") == 0) {
            if (tool_calls(json, message, err) != YVEX_OK)
                return yvex_error_code(err);
        } else if (strcmp(key, "name") == 0) {
            return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                               "named messages are unsupported");
        } else {
            return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                               "unsupported message field");
        }
    }
    if (item != YVEX_JSON_ITEM_END || iter.trailing_separator || !role_seen ||
        !content_seen)
        return json_refuse(err, YVEX_ERR_FORMAT,
                           "message role and content are required");
    return YVEX_OK;
}

/* Purpose: parse one bounded ordered message array.
 * Inputs: array parser, provider request owner, and error output.
 * Effects: allocates and fills the ordered provider message directory.
 * Failure: refuses empty/oversized arrays and leaves request cleanup authoritative.
 * Boundary: retains caller order without constructing prompt text. */
static int messages_parse(yvex_json *json, yvex_provider_request *request,
                          yvex_error *err)
{
    yvex_json_iter iter;
    yvex_json_item item;
    yvex_provider_message *messages = calloc(
        YVEX_PROVIDER_MAX_MESSAGES, sizeof(*messages));
    if (!messages) return YVEX_ERR_NOMEM;
    request->messages = messages;
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        unsigned long long index;
        if (request->message_count >= YVEX_PROVIDER_MAX_MESSAGES)
            return json_refuse(err, YVEX_ERR_BOUNDS,
                               "message count exceeds the provider limit");
        index = request->message_count++;
        if (message_parse(json, &messages[index], err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
                   request->message_count
               ? YVEX_OK
               : json_refuse(err, YVEX_ERR_FORMAT,
                             "nonempty message array is required");
}

/* Purpose: parse one OpenAI function definition into provider-neutral fields.
 * Inputs: object parser, cleared function definition, and error output.
 * Effects: allocates description/schema bytes and records exact name/strict facts.
 * Failure: refuses unknown fields, invalid schema objects, and unsupported strict mode.
 * Boundary: definition admission never executes or dynamically loads tools. */
static int function_definition(yvex_json *json,
                               yvex_provider_function_tool *tool,
                               yvex_error *err)
{
    yvex_json_iter iter;
    yvex_json_item item;
    seen_keys seen = {0};
    char key[JSON_SEEN_KEY];
    int name_seen = 0, parameters_seen = 0;
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (key_unique(&seen, key, err) != YVEX_OK) return yvex_error_code(err);
        if (strcmp(key, "name") == 0) {
            if (string_fixed(json, tool->name, sizeof(tool->name), 0, err) != YVEX_OK)
                return yvex_error_code(err);
            name_seen = 1;
        } else if (strcmp(key, "description") == 0) {
            if (string_span(json, &tool->description,
                            YVEX_PROVIDER_MAX_MESSAGE_BYTES, 1, err) != YVEX_OK)
                return yvex_error_code(err);
        } else if (strcmp(key, "parameters") == 0) {
            if (raw_value(json, &tool->parameters_json,
                          YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES, 1, err) != YVEX_OK)
                return yvex_error_code(err);
            parameters_seen = 1;
        } else if (strcmp(key, "strict") == 0) {
            int strict = 0;
            if (!yvex_json_bool(json, &strict)) return YVEX_ERR_FORMAT;
            if (strict)
                return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                   "strict function schemas require constrained decoding");
        } else {
            return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                               "unsupported function definition field");
        }
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
                   name_seen && parameters_seen
               ? YVEX_OK
               : json_refuse(err, YVEX_ERR_FORMAT,
                             "function name and parameters are required");
}

/* Purpose: parse one tools array containing only bounded functions.
 * Inputs: array parser, provider request owner, and error output.
 * Effects: allocates a bounded typed function-tool directory.
 * Failure: refuses unknown tool types, empty definitions, or count overflow.
 * Boundary: only the admitted function type enters provider-neutral state. */
static int tools_parse(yvex_json *json, yvex_provider_request *request,
                       int responses_style, yvex_error *err)
{
    yvex_json_iter array;
    yvex_json_item item;
    yvex_provider_function_tool *tools = calloc(
        YVEX_PROVIDER_MAX_TOOLS, sizeof(*tools));
    if (!tools) return YVEX_ERR_NOMEM;
    request->tools = tools;
    if (!yvex_json_iter_begin(json, &array, YVEX_JSON_COLLECTION_ARRAY))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_array_value(&array)) == YVEX_JSON_ITEM_READY) {
        yvex_json_iter object;
        yvex_json_item member;
        seen_keys seen = {0};
        char key[JSON_SEEN_KEY], type[32] = {0};
        int type_seen = 0, function_seen = 0;
        int name_seen = 0, parameters_seen = 0;
        unsigned long long tool_index;
        if (request->tool_count >= YVEX_PROVIDER_MAX_TOOLS)
            return json_refuse(err, YVEX_ERR_BOUNDS,
                               "tool count exceeds the provider limit");
        tool_index = request->tool_count++;
        if (!yvex_json_iter_begin(json, &object, YVEX_JSON_COLLECTION_OBJECT))
            return YVEX_ERR_FORMAT;
        while ((member = yvex_json_object_member(&object, key, sizeof(key))) ==
               YVEX_JSON_ITEM_READY) {
            if (key_unique(&seen, key, err) != YVEX_OK)
                return yvex_error_code(err);
            if (strcmp(key, "type") == 0) {
                if (string_fixed(json, type, sizeof(type), 0, err) != YVEX_OK)
                    return yvex_error_code(err);
                type_seen = 1;
            } else if (strcmp(key, "function") == 0) {
                if (responses_style)
                    return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                       "Responses function tools use flat definitions");
                if (function_definition(json, &tools[tool_index], err) !=
                    YVEX_OK)
                    return yvex_error_code(err);
                function_seen = 1;
            } else if (responses_style && strcmp(key, "name") == 0) {
                if (string_fixed(json, tools[tool_index].name,
                                 sizeof(tools[tool_index].name), 0, err) != YVEX_OK)
                    return yvex_error_code(err);
                name_seen = 1;
            } else if (responses_style && strcmp(key, "description") == 0) {
                if (string_span(json, &tools[tool_index].description,
                                YVEX_PROVIDER_MAX_MESSAGE_BYTES, 1, err) != YVEX_OK)
                    return yvex_error_code(err);
            } else if (responses_style && strcmp(key, "parameters") == 0) {
                if (raw_value(json, &tools[tool_index].parameters_json,
                              YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES, 1, err) != YVEX_OK)
                    return yvex_error_code(err);
                parameters_seen = 1;
            } else if (responses_style && strcmp(key, "strict") == 0) {
                int strict = 0;
                if (!yvex_json_bool(json, &strict)) return YVEX_ERR_FORMAT;
                if (strict)
                    return json_refuse(
                        err, YVEX_ERR_UNSUPPORTED,
                        "strict function schemas require constrained decoding");
            } else {
                return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                   "unsupported tool definition field");
            }
        }
        if (member != YVEX_JSON_ITEM_END || object.trailing_separator ||
            !type_seen || strcmp(type, "function") != 0 ||
            (responses_style ? (!name_seen || !parameters_seen)
                             : !function_seen))
            return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                               "only complete function tools are supported");
    }
    return item == YVEX_JSON_ITEM_END && !array.trailing_separator
               ? YVEX_OK
               : YVEX_ERR_FORMAT;
}

/* Purpose: parse none/auto/required or one named function tool choice.
 * Inputs: parser cursor, provider request policy, and error output.
 * Effects: records one deterministic tool-choice enum and optional function name.
 * Failure: refuses unknown strings, fields, types, and malformed named choices.
 * Boundary: choice is intent; selection remains model/tokenizer-owned. */
static int tool_choice_parse(yvex_json *json, yvex_provider_request *request,
                             int responses_style, yvex_error *err)
{
    yvex_json_space(json);
    if (json->cursor < json->end && *json->cursor == '"') {
        char choice[32];
        if (string_fixed(json, choice, sizeof(choice), 0, err) != YVEX_OK)
            return yvex_error_code(err);
        if (strcmp(choice, "none") == 0)
            request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_NONE;
        else if (strcmp(choice, "auto") == 0)
            request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_AUTO;
        else if (strcmp(choice, "required") == 0)
            request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_REQUIRED;
        else return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                "unsupported tool_choice string");
        return YVEX_OK;
    }
    {
        yvex_json_iter object;
        yvex_json_item item;
        seen_keys seen = {0};
        char key[JSON_SEEN_KEY], type[32] = {0};
        int function_seen = 0;
        if (!yvex_json_iter_begin(json, &object, YVEX_JSON_COLLECTION_OBJECT))
            return YVEX_ERR_FORMAT;
        while ((item = yvex_json_object_member(&object, key, sizeof(key))) ==
               YVEX_JSON_ITEM_READY) {
            if (key_unique(&seen, key, err) != YVEX_OK)
                return yvex_error_code(err);
            if (strcmp(key, "type") == 0) {
                if (string_fixed(json, type, sizeof(type), 0, err) != YVEX_OK)
                    return yvex_error_code(err);
            } else if (strcmp(key, "function") == 0) {
                yvex_json_iter function;
                yvex_json_item field;
                char function_key[JSON_SEEN_KEY];
                if (responses_style)
                    return json_refuse(
                        err, YVEX_ERR_UNSUPPORTED,
                        "Responses named tool choices use a flat function name");
                if (!yvex_json_iter_begin(json, &function,
                                          YVEX_JSON_COLLECTION_OBJECT))
                    return YVEX_ERR_FORMAT;
                field = yvex_json_object_member(&function, function_key,
                                                sizeof(function_key));
                if (field != YVEX_JSON_ITEM_READY ||
                    strcmp(function_key, "name") != 0 ||
                    string_fixed(json, request->tool_choice.function_name,
                                 sizeof(request->tool_choice.function_name),
                                 0, err) != YVEX_OK ||
                    yvex_json_object_member(&function, function_key,
                                            sizeof(function_key)) !=
                        YVEX_JSON_ITEM_END || function.trailing_separator)
                    return YVEX_ERR_FORMAT;
                function_seen = 1;
            } else if (responses_style && strcmp(key, "name") == 0) {
                if (string_fixed(json, request->tool_choice.function_name,
                                 sizeof(request->tool_choice.function_name),
                                 0, err) != YVEX_OK)
                    return yvex_error_code(err);
                function_seen = 1;
            } else return YVEX_ERR_UNSUPPORTED;
        }
        if (item != YVEX_JSON_ITEM_END || object.trailing_separator ||
            strcmp(type, "function") != 0 || !function_seen)
            return YVEX_ERR_FORMAT;
    }
    request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_FUNCTION;
    return YVEX_OK;
}

/* Purpose: parse one string or bounded array of stop strings.
 * Inputs: parser cursor, provider request owner, and error output.
 * Effects: allocates explicit stop spans in request order.
 * Failure: refuses empty, oversized, malformed, or over-count input.
 * Boundary: incremental stop matching belongs to the server output owner. */
static int stops_parse(yvex_json *json, yvex_provider_request *request,
                       yvex_error *err)
{
    yvex_provider_span *stops = calloc(YVEX_PROVIDER_MAX_STOPS, sizeof(*stops));
    if (!stops) return YVEX_ERR_NOMEM;
    request->stop_strings = stops;
    yvex_json_space(json);
    if (json->cursor < json->end && *json->cursor == '"') {
        if (string_span(json, &stops[0], YVEX_PROVIDER_MAX_STOP_BYTES,
                        0, err) != YVEX_OK)
            return yvex_error_code(err);
        request->stop_count = 1u;
        return YVEX_OK;
    }
    {
        yvex_json_iter iter;
        yvex_json_item item;
        if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY))
            return YVEX_ERR_FORMAT;
        while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
            if (request->stop_count >= YVEX_PROVIDER_MAX_STOPS)
                return YVEX_ERR_BOUNDS;
            if (string_span(json, &stops[request->stop_count],
                            YVEX_PROVIDER_MAX_STOP_BYTES, 0, err) != YVEX_OK)
                return yvex_error_code(err);
            request->stop_count++;
        }
        return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
                       request->stop_count
                   ? YVEX_OK
                   : YVEX_ERR_FORMAT;
    }
}

/* Purpose: parse the bounded json_object response-format profile.
 * Inputs: response-format object parser, provider request, and error output.
 * Effects: selects text or json_object validation policy.
 * Failure: refuses duplicates, unknown fields, json_schema, or unknown types.
 * Boundary: does not claim constrained decoding or strict schema adherence. */
static int response_format_parse(yvex_json *json,
                                 yvex_provider_request *request,
                                 yvex_error *err)
{
    yvex_json_iter iter;
    yvex_json_item item;
    seen_keys seen = {0};
    char key[JSON_SEEN_KEY], type[32] = {0};
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (key_unique(&seen, key, err) != YVEX_OK) return yvex_error_code(err);
        if (strcmp(key, "type") != 0 ||
            string_fixed(json, type, sizeof(type), 0, err) != YVEX_OK)
            return YVEX_ERR_UNSUPPORTED;
    }
    if (item != YVEX_JSON_ITEM_END || iter.trailing_separator) return YVEX_ERR_FORMAT;
    if (strcmp(type, "text") == 0)
        request->response_format = YVEX_PROVIDER_RESPONSE_TEXT;
    else if (strcmp(type, "json_object") == 0)
        request->response_format = YVEX_PROVIDER_RESPONSE_JSON_OBJECT;
    else
        return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                           "only text and json_object response formats are supported");
    return YVEX_OK;
}

/* Purpose: parse stream_options.include_usage without accepting hidden options.
 * Inputs: stream-options object parser, provider request, and error output.
 * Effects: records the exact terminal usage-chunk request.
 * Failure: refuses duplicate, unknown, or non-boolean fields.
 * Boundary: affects compatibility rendering only, not token accounting. */
static int stream_options_parse(yvex_json *json,
                                yvex_provider_request *request,
                                yvex_error *err)
{
    yvex_json_iter iter;
    yvex_json_item item;
    seen_keys seen = {0};
    char key[JSON_SEEN_KEY];
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return YVEX_ERR_FORMAT;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (key_unique(&seen, key, err) != YVEX_OK) return yvex_error_code(err);
        if (strcmp(key, "include_usage") != 0 ||
            !yvex_json_bool(json, &request->include_usage))
            return YVEX_ERR_UNSUPPORTED;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator
               ? YVEX_OK
               : YVEX_ERR_FORMAT;
}

/* Purpose: apply common root request fields shared by both admitted endpoint profiles.
 * Inputs: field name, parser cursor, provider request, handled flag, and error output.
 * Effects: records one shared model/sampling/stream/tool/stop field.
 * Failure: refuses invalid types, unsupported parallel calls, or malformed values.
 * Boundary: endpoint-specific fields remain in the root profile parser. */
static int common_field(const char *key, yvex_json *json,
                        yvex_provider_request *request,
                        openai_endpoint endpoint, int *handled,
                        yvex_error *err)
{
    *handled = 1;
    if (strcmp(key, "model") == 0)
        return string_fixed(json, request->model, sizeof(request->model), 0, err);
    if (strcmp(key, "temperature") == 0)
        return number_double(json, &request->sampling.temperature, err);
    if (strcmp(key, "top_p") == 0)
        return number_double(json, &request->sampling.top_p, err);
    if (strcmp(key, "seed") == 0) {
        if (!yvex_json_u64(json, &request->sampling.seed)) return YVEX_ERR_FORMAT;
        request->sampling.seed_present = 1;
        return YVEX_OK;
    }
    if (strcmp(key, "stream") == 0)
        return yvex_json_bool(json, &request->stream) ? YVEX_OK : YVEX_ERR_FORMAT;
    if (strcmp(key, "stop") == 0) return stops_parse(json, request, err);
    if (strcmp(key, "tools") == 0)
        return tools_parse(json, request,
                           endpoint == OPENAI_ENDPOINT_RESPONSES, err);
    if (strcmp(key, "tool_choice") == 0)
        return tool_choice_parse(json, request,
                                 endpoint == OPENAI_ENDPOINT_RESPONSES, err);
    if (strcmp(key, "parallel_tool_calls") == 0) {
        int parallel = 0;
        if (!yvex_json_bool(json, &parallel)) return YVEX_ERR_FORMAT;
        if (parallel) return json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                         "parallel tool calls are unsupported");
        return YVEX_OK;
    }
    *handled = 0;
    return YVEX_OK;
}

/* Purpose: shift one optional Responses instruction into the first system message.
 * Inputs: provider request, owned instruction span, and error output.
 * Effects: grows/shifts the message directory and transfers instruction ownership.
 * Failure: preserves existing message order and reports capacity/allocation failure.
 * Boundary: role projection only; prompt syntax remains tokenizer-owned. */
static int prepend_instruction(yvex_provider_request *request,
                               yvex_provider_span instruction,
                               yvex_error *err)
{
    yvex_provider_message *messages = (yvex_provider_message *)request->messages;
    if (!instruction.bytes) return YVEX_OK;
    if (!messages) {
        messages = calloc(YVEX_PROVIDER_MAX_MESSAGES, sizeof(*messages));
        if (!messages) return YVEX_ERR_NOMEM;
        request->messages = messages;
    }
    if (request->message_count >= YVEX_PROVIDER_MAX_MESSAGES)
        return json_refuse(err, YVEX_ERR_BOUNDS,
                           "instruction exceeds message capacity");
    memmove(messages + 1u, messages,
            (size_t)request->message_count * sizeof(*messages));
    memset(messages, 0, sizeof(*messages));
    messages[0].role = YVEX_PROVIDER_ROLE_SYSTEM;
    messages[0].content = instruction;
    request->message_count++;
    return YVEX_OK;
}

/* Purpose: parse Responses input string, message array, or function-call-output items.
 * Inputs: input parser, provider request owner, and error output.
 * Effects: allocates ordered typed messages including tool-result call IDs.
 * Failure: refuses unsupported item types, malformed fields, or count overflow.
 * Boundary: remote conversation objects and multimodal content remain unsupported. */
static int responses_input(yvex_json *json, yvex_provider_request *request,
                           yvex_error *err)
{
    yvex_json_space(json);
    if (json->cursor < json->end && *json->cursor == '"') {
        yvex_provider_message *messages = calloc(
            YVEX_PROVIDER_MAX_MESSAGES, sizeof(*messages));
        if (!messages) return YVEX_ERR_NOMEM;
        request->messages = messages;
        messages[0].role = YVEX_PROVIDER_ROLE_USER;
        if (string_span(json, &messages[0].content,
                        YVEX_PROVIDER_MAX_MESSAGE_BYTES, 0, err) != YVEX_OK)
            return yvex_error_code(err);
        request->message_count = 1u;
        return YVEX_OK;
    }
    {
        yvex_json_iter array;
        yvex_json_item item;
        yvex_provider_message *messages = calloc(
            YVEX_PROVIDER_MAX_MESSAGES, sizeof(*messages));
        if (!messages) return YVEX_ERR_NOMEM;
        request->messages = messages;
        if (!yvex_json_iter_begin(json, &array, YVEX_JSON_COLLECTION_ARRAY))
            return YVEX_ERR_FORMAT;
        while ((item = yvex_json_array_value(&array)) == YVEX_JSON_ITEM_READY) {
            const char *start = json->cursor;
            yvex_json probe = *json;
            yvex_json_iter object;
            yvex_json_item member;
            char key[JSON_SEEN_KEY], type[64] = {0};
            int typed = 0;
            unsigned long long message_index;
            if (request->message_count >= YVEX_PROVIDER_MAX_MESSAGES)
                return YVEX_ERR_BOUNDS;
            message_index = request->message_count++;
            if (!yvex_json_iter_begin(&probe, &object,
                                      YVEX_JSON_COLLECTION_OBJECT))
                return YVEX_ERR_FORMAT;
            while ((member = yvex_json_object_member(&object, key,
                                                     sizeof(key))) ==
                   YVEX_JSON_ITEM_READY) {
                if (strcmp(key, "type") == 0) {
                    if (!yvex_json_string(&probe, type, sizeof(type)))
                        return YVEX_ERR_FORMAT;
                    typed = 1;
                    break;
                }
                if (!yvex_json_skip_value(&probe)) return YVEX_ERR_FORMAT;
            }
            *json = (yvex_json){start, json->end, json->depth};
            if (typed && strcmp(type, "function_call_output") == 0) {
                yvex_provider_message *message = &messages[message_index];
                seen_keys seen = {0};
                int call_seen = 0, output_seen = 0;
                message->role = YVEX_PROVIDER_ROLE_TOOL;
                if (!yvex_json_iter_begin(json, &object,
                                          YVEX_JSON_COLLECTION_OBJECT))
                    return YVEX_ERR_FORMAT;
                while ((member = yvex_json_object_member(&object, key,
                                                         sizeof(key))) ==
                       YVEX_JSON_ITEM_READY) {
                    if (key_unique(&seen, key, err) != YVEX_OK)
                        return yvex_error_code(err);
                    if (strcmp(key, "type") == 0) {
                        if (!yvex_json_string(json, type, sizeof(type)))
                            return YVEX_ERR_FORMAT;
                    } else if (strcmp(key, "call_id") == 0) {
                        if (string_fixed(json, message->tool_call_id,
                                         sizeof(message->tool_call_id), 0,
                                         err) != YVEX_OK)
                            return yvex_error_code(err);
                        call_seen = 1;
                    } else if (strcmp(key, "output") == 0) {
                        if (string_span(json, &message->content,
                                        YVEX_PROVIDER_MAX_MESSAGE_BYTES, 1,
                                        err) != YVEX_OK)
                            return yvex_error_code(err);
                        output_seen = 1;
                    } else return YVEX_ERR_UNSUPPORTED;
                }
                if (member != YVEX_JSON_ITEM_END || object.trailing_separator ||
                    !call_seen || !output_seen) return YVEX_ERR_FORMAT;
            } else {
                if (message_parse(json, &messages[message_index], err) !=
                    YVEX_OK)
                    return yvex_error_code(err);
            }
        }
        return item == YVEX_JSON_ITEM_END && !array.trailing_separator &&
                       request->message_count
                   ? YVEX_OK
                   : YVEX_ERR_FORMAT;
    }
}

/* Purpose: parse the complete admitted endpoint root and seal its provider identity.
 * Inputs: complete HTTP body, endpoint profile, loaded model ID, output owner, and error output.
 * Effects: allocates, validates, and identity-seals one provider-neutral request graph.
 * Failure: releases every partial allocation and publishes no request.
 * Boundary: OpenAI field names end here and never enter runtime/generation owners. */
int openai_json_admit(const openai_http_request *http,
                      openai_endpoint endpoint, const char *selected_model,
                      openai_admitted_request *admitted, yvex_error *err)
{
    yvex_provider_request *request = NULL;
    yvex_provider_span instruction = {0};
    yvex_json json;
    yvex_json_iter root;
    yvex_json_item item;
    seen_keys seen = {0};
    char key[JSON_SEEN_KEY];
    int model_seen = 0, input_seen = 0, maximum_seen = 0;
    int handled, rc = YVEX_OK;
    if (admitted) memset(admitted, 0, sizeof(*admitted));
    if (!http || !admitted || !selected_model || !http->body ||
        !http->body_count || http->body_count > SIZE_MAX)
        return json_refuse(err, YVEX_ERR_INVALID_ARG,
                           "one bounded JSON request body is required");
    request = calloc(1u, sizeof(*request));
    if (!request) return YVEX_ERR_NOMEM;
    request->schema_version = YVEX_PROVIDER_SCHEMA_V1;
    request->response_format = YVEX_PROVIDER_RESPONSE_TEXT;
    request->sampling.temperature = 1.0;
    request->sampling.top_p = 1.0;
    request->sampling.typical_p = 1.0;
    request->maximum_output_tokens = 128u;
    request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_AUTO;
    request->external_correlation_id[0] = '\0';
    yvex_json_init(&json, (const char *)http->body, (size_t)http->body_count);
    if (!yvex_json_iter_begin(&json, &root, YVEX_JSON_COLLECTION_OBJECT)) {
        rc = YVEX_ERR_FORMAT;
        goto failure;
    }
    while (rc == YVEX_OK &&
           (item = yvex_json_object_member(&root, key, sizeof(key))) ==
               YVEX_JSON_ITEM_READY) {
        rc = key_unique(&seen, key, err);
        if (rc != YVEX_OK) break;
        rc = common_field(key, &json, request, endpoint, &handled, err);
        if (rc != YVEX_OK) break;
        if (handled) {
            if (strcmp(key, "model") == 0) model_seen = 1;
            continue;
        }
        if (endpoint == OPENAI_ENDPOINT_CHAT) {
            if (strcmp(key, "messages") == 0) {
                rc = messages_parse(&json, request, err);
                input_seen = rc == YVEX_OK;
            } else if (strcmp(key, "max_tokens") == 0 ||
                       strcmp(key, "max_completion_tokens") == 0) {
                if (maximum_seen)
                    rc = json_refuse(
                        err, YVEX_ERR_FORMAT,
                        "max_tokens and max_completion_tokens are mutually exclusive");
                else if (!yvex_json_u64(
                             &json, &request->maximum_output_tokens))
                    rc = YVEX_ERR_FORMAT;
                maximum_seen = rc == YVEX_OK;
            } else if (strcmp(key, "n") == 0) {
                unsigned long long n = 0u;
                if (!yvex_json_u64(&json, &n) || n != 1u)
                    rc = json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                     "only n=1 is supported");
            } else if (strcmp(key, "response_format") == 0) {
                rc = response_format_parse(&json, request, err);
            } else if (strcmp(key, "stream_options") == 0) {
                rc = stream_options_parse(&json, request, err);
            } else {
                rc = json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                 "unsupported Chat Completions parameter");
            }
        } else {
            if (strcmp(key, "input") == 0) {
                rc = responses_input(&json, request, err);
                input_seen = rc == YVEX_OK;
            } else if (strcmp(key, "instructions") == 0) {
                rc = string_span(&json, &instruction,
                                 YVEX_PROVIDER_MAX_MESSAGE_BYTES, 1, err);
            } else if (strcmp(key, "max_output_tokens") == 0) {
                if (!yvex_json_u64(&json, &request->maximum_output_tokens))
                    rc = YVEX_ERR_FORMAT;
            } else if (strcmp(key, "previous_response_id") == 0) {
                rc = string_fixed(&json, request->previous_response_id,
                                  sizeof(request->previous_response_id), 0, err);
            } else if (strcmp(key, "store") == 0) {
                int store = 0;
                if (!yvex_json_bool(&json, &store)) rc = YVEX_ERR_FORMAT;
                else if (store) rc = json_refuse(
                    err, YVEX_ERR_UNSUPPORTED, "store=true is unsupported");
            } else if (strcmp(key, "background") == 0) {
                int background = 0;
                if (!yvex_json_bool(&json, &background)) rc = YVEX_ERR_FORMAT;
                else if (background)
                    rc = json_refuse(
                        err, YVEX_ERR_UNSUPPORTED,
                        "background Responses are unsupported");
            } else {
                rc = json_refuse(err, YVEX_ERR_UNSUPPORTED,
                                 "unsupported Responses parameter");
            }
        }
    }
    if (rc == YVEX_OK &&
        (item != YVEX_JSON_ITEM_END || root.trailing_separator ||
         !yvex_json_complete(&json) || !model_seen || !input_seen))
        rc = json_refuse(err, YVEX_ERR_FORMAT,
                         "model and complete input are required");
    if (rc == YVEX_OK && strcmp(request->model, selected_model) != 0)
        rc = json_refuse(err, YVEX_ERR_STATE,
                         "requested model is not loaded by yvexd");
    if (rc == YVEX_OK && endpoint == OPENAI_ENDPOINT_RESPONSES)
        rc = prepend_instruction(request, instruction, err);
    if (rc == YVEX_OK) instruction.bytes = NULL;
    if (rc == YVEX_OK && !request->tool_count) {
        if (request->tool_choice.kind == YVEX_PROVIDER_TOOL_CHOICE_REQUIRED ||
            request->tool_choice.kind == YVEX_PROVIDER_TOOL_CHOICE_FUNCTION)
            rc = json_refuse(err, YVEX_ERR_FORMAT,
                             "tool_choice requires at least one function tool");
        else
            request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_NONE;
    }
    request->sampling.stochastic = request->sampling.temperature > 0.0;
    if (rc == YVEX_OK && request->sampling.stochastic &&
        !request->sampling.seed_present) {
        struct timespec now;
        (void)clock_gettime(CLOCK_MONOTONIC, &now);
        request->sampling.seed =
            ((unsigned long long)now.tv_sec << 32u) ^
            (unsigned long long)now.tv_nsec ^ (unsigned long long)getpid();
        request->sampling.seed_present = 1;
    }
    if (rc == YVEX_OK && !request->sampling.stochastic &&
        (request->sampling.top_p != 1.0 || request->sampling.top_k ||
         request->sampling.min_p != 0.0 || request->sampling.typical_p != 1.0))
        rc = json_refuse(err, YVEX_ERR_UNSUPPORTED,
                         "temperature=0 requires neutral sampling filters");
    if (rc == YVEX_OK) rc = yvex_provider_request_seal(request, err);
    if (rc != YVEX_OK) goto failure;
    admitted->provider = request;
    admitted->endpoint = endpoint;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    free((void *)instruction.bytes);
    yvex_provider_request_close(&request);
    if (!yvex_error_code(err))
        yvex_error_set(err, rc, "server.openai.request",
                       "OpenAI request admission failed");
    return rc;
}

/* Purpose: release one admitted request and all translated application bytes.
 * Inputs: gateway admitted-request owner.
 * Effects: closes the provider graph and clears endpoint facts.
 * Failure: none; null and cleared owners are accepted.
 * Boundary: does not affect daemon sessions or HTTP request storage. */
void openai_admitted_request_clear(openai_admitted_request *request)
{
    if (!request) return;
    yvex_provider_request_close(&request->provider);
    memset(request, 0, sizeof(*request));
}
