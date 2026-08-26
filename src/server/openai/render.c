/*
 * Render stable Chat Completions and Responses JSON without terminal-oriented printers.
 *
 * Every string is escaped, usage is authoritative, and no internal path/identity leaks by default.
 * Provider-neutral/YVEX result facts become only the documented compatibility-profile objects.
 */

#include "src/server/openai/private.h"

#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char *data;
    unsigned long long count, capacity;
} render_builder;

static int render_reserve(render_builder *builder, unsigned long long add,
                          yvex_error *err)
{
    unsigned long long need, capacity;
    unsigned char *grown;
    if (!builder || builder->count > ULLONG_MAX - add - 1u) return YVEX_ERR_BOUNDS;
    need = builder->count + add + 1u;
    if (need <= builder->capacity) return YVEX_OK;
    capacity = builder->capacity ? builder->capacity : 512u;
    while (capacity < need) {
        if (capacity > ULLONG_MAX / 2u || capacity * 2u > SIZE_MAX)
            return YVEX_ERR_BOUNDS;
        capacity *= 2u;
    }
    grown = realloc(builder->data, (size_t)capacity);
    if (!grown) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "server.openai.render",
                       "JSON response allocation failed");
        return YVEX_ERR_NOMEM;
    }
    builder->data = grown;
    builder->capacity = capacity;
    return YVEX_OK;
}

static int render_append(render_builder *builder, const void *bytes,
                         unsigned long long count, yvex_error *err)
{
    int rc = render_reserve(builder, count, err);
    if (rc != YVEX_OK) return rc;
    if (count) memcpy(builder->data + builder->count, bytes, (size_t)count);
    builder->count += count;
    builder->data[builder->count] = '\0';
    return YVEX_OK;
}

static int render_literal(render_builder *builder, const char *text,
                          yvex_error *err)
{
    return render_append(builder, text, (unsigned long long)strlen(text), err);
}

static int render_string(render_builder *builder, const unsigned char *bytes,
                         unsigned long long count, yvex_error *err)
{
    unsigned long long index;
    int rc = render_literal(builder, "\"", err);
    for (index = 0u; rc == YVEX_OK && index < count; ++index) {
        unsigned char byte = bytes[index];
        char escaped[7];
        if (byte == '"' || byte == '\\') {
            escaped[0] = '\\'; escaped[1] = (char)byte;
            rc = render_append(builder, escaped, 2u, err);
        } else if (byte == '\b') rc = render_literal(builder, "\\b", err);
        else if (byte == '\f') rc = render_literal(builder, "\\f", err);
        else if (byte == '\n') rc = render_literal(builder, "\\n", err);
        else if (byte == '\r') rc = render_literal(builder, "\\r", err);
        else if (byte == '\t') rc = render_literal(builder, "\\t", err);
        else if (byte < 0x20u) {
            (void)snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
            rc = render_append(builder, escaped, 6u, err);
        } else rc = render_append(builder, &byte, 1u, err);
    }
    return rc == YVEX_OK ? render_literal(builder, "\"", err) : rc;
}

static int render_text(render_builder *builder, const char *text,
                       yvex_error *err)
{
    return render_string(builder, (const unsigned char *)(text ? text : ""),
                         text ? strlen(text) : 0u, err);
}

static int render_finish(render_builder *builder, unsigned char **output,
                         unsigned long long *count, yvex_error *err)
{
    if (!output || !count || !builder->data) {
        free(builder->data);
        return YVEX_ERR_INVALID_ARG;
    }
    *output = builder->data;
    *count = builder->count;
    memset(builder, 0, sizeof(*builder));
    yvex_error_clear(err);
    return YVEX_OK;
}

int openai_json_error(int status, const char *type, const char *param,
                      const char *code, const char *message,
                      unsigned char **output, unsigned long long *count,
                      yvex_error *err)
{
    render_builder builder = {0};
    int rc;
    (void)status;
    if (output) *output = NULL;
    if (count) *count = 0u;
    rc = render_literal(&builder, "{\"error\":{\"message\":", err);
    if (rc == YVEX_OK) rc = render_text(&builder, message, err);
    if (rc == YVEX_OK) rc = render_literal(&builder, ",\"type\":", err);
    if (rc == YVEX_OK) rc = render_text(&builder, type, err);
    if (rc == YVEX_OK) rc = render_literal(&builder, ",\"param\":", err);
    if (rc == YVEX_OK) {
        if (param && *param) rc = render_text(&builder, param, err);
        else rc = render_literal(&builder, "null", err);
    }
    if (rc == YVEX_OK) rc = render_literal(&builder, ",\"code\":", err);
    if (rc == YVEX_OK) rc = render_text(&builder, code, err);
    if (rc == YVEX_OK) rc = render_literal(&builder, "}}", err);
    if (rc != YVEX_OK) {
        free(builder.data);
        return rc;
    }
    return render_finish(&builder, output, count, err);
}

int openai_json_models(const yvex_server_engine_summary *engines,
                       unsigned long long engine_count, int list,
                       unsigned char **output, unsigned long long *count,
                       yvex_error *err)
{
    render_builder builder = {0};
    unsigned long long index;
    int rc;
    if (output) *output = NULL;
    if (count) *count = 0u;
    if ((!engines && engine_count) || (!list && engine_count != 1u))
        return YVEX_ERR_INVALID_ARG;
    rc = render_literal(&builder, list ? "{\"object\":\"list\",\"data\":["
                                       : "", err);
    for (index = 0ull; rc == YVEX_OK && index < engine_count; ++index) {
        if (list && index) rc = render_literal(&builder, ",", err);
        if (rc == YVEX_OK)
        rc = render_literal(&builder, "{\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(&builder, engines[index].alias, err);
        if (rc == YVEX_OK)
            rc = render_literal(
                &builder,
                ",\"object\":\"model\",\"created\":0,\"owned_by\":\"yvex\","
                "\"yvex_profile\":\"" OPENAI_COMPAT_PROFILE "\"}", err);
    }
    if (rc == YVEX_OK && list) rc = render_literal(&builder, "]}", err);
    if (rc != YVEX_OK) { free(builder.data); return rc; }
    return render_finish(&builder, output, count, err);
}

static int render_chat_usage(render_builder *builder,
                             const openai_generation_result *result,
                             yvex_error *err)
{
    char values[384];
    int length = snprintf(values, sizeof(values),
        "\"usage\":{\"prompt_tokens\":%llu,\"completion_tokens\":%llu,"
        "\"total_tokens\":%llu,\"completion_tokens_details\":{"
        "\"reasoning_tokens\":%llu}}", result->prompt_tokens,
        result->completion_tokens, result->total_tokens,
        result->reasoning_tokens);
    return length > 0 && (size_t)length < sizeof(values)
               ? render_append(builder, values, (unsigned long long)length, err)
               : YVEX_ERR_BOUNDS;
}

static int render_completion_metrics(render_builder *builder,
                                     const openai_generation_result *result,
                                     yvex_error *err)
{
    char values[768];
    int length = snprintf(
        values, sizeof(values),
        "\"yvex_completion_metrics\":{\"reasoning_tokens\":%llu,"
        "\"final_tokens\":%llu,\"reasoning_tokens_per_second\":%.9g,"
        "\"final_tokens_per_second\":%.9g,\"total_tokens_per_second\":%.9g,"
        "\"time_to_first_reasoning_token\":%.9g,"
        "\"time_to_first_final_token\":%.9g,\"reasoning_seconds\":%.9g,"
        "\"final_seconds\":%.9g,\"total_completion_seconds\":%.9g}",
        result->reasoning_tokens, result->final_tokens,
        result->reasoning_rate, result->final_rate,
        result->total_completion_rate, result->first_reasoning_seconds,
        result->first_final_seconds, result->reasoning_seconds,
        result->final_seconds, result->total_completion_seconds);
    return length > 0 && (size_t)length < sizeof(values)
               ? render_append(builder, values, (unsigned long long)length, err)
               : YVEX_ERR_BOUNDS;
}

static int render_chat_message(render_builder *builder,
                               const openai_generation_result *result,
                               yvex_error *err)
{
    unsigned long long index;
    int rc = render_literal(builder, "{\"role\":\"assistant\",\"content\":", err);
    if (rc == YVEX_OK) {
        if (result->tool_call_count && !result->text_count)
            rc = render_literal(builder, "null", err);
        else
            rc = render_string(builder, result->text, result->text_count, err);
    }
    if (rc == YVEX_OK) {
        rc = render_literal(builder, ",\"reasoning_content\":", err);
        if (rc == YVEX_OK)
            rc = render_string(builder, result->reasoning,
                               result->reasoning_count, err);
    }
    if (rc == YVEX_OK && result->tool_call_count)
        rc = render_literal(builder, ",\"tool_calls\":[", err);
    for (index = 0u; rc == YVEX_OK && index < result->tool_call_count;
         ++index) {
        const openai_generation_tool_call *call = &result->tool_calls[index];
        if (index) rc = render_literal(builder, ",", err);
        if (rc == YVEX_OK) rc = render_literal(builder, "{\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, call->call_id, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                ",\"type\":\"function\",\"function\":{\"name\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, call->name, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"arguments\":", err);
        if (rc == YVEX_OK)
            rc = render_string(builder, call->arguments,
                               call->arguments_count, err);
        if (rc == YVEX_OK) rc = render_literal(builder, "}}", err);
    }
    if (rc == YVEX_OK && result->tool_call_count)
        rc = render_literal(builder, "]", err);
    return rc == YVEX_OK ? render_literal(builder, "}", err) : rc;
}

static int render_chat_result(render_builder *builder, const char *id,
                              const char *model, unsigned long long created,
                              const openai_generation_result *result,
                              yvex_error *err)
{
    char prefix[256];
    int length = snprintf(prefix, sizeof(prefix),
        "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%llu,"
        "\"model\":", id, created);
    int rc = length > 0 && (size_t)length < sizeof(prefix)
                 ? render_append(builder, prefix, (unsigned long long)length, err)
                 : YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK) rc = render_text(builder, model, err);
    if (rc == YVEX_OK) rc = render_literal(builder, ",\"choices\":[{\"index\":0,\"message\":", err);
    if (rc == YVEX_OK) rc = render_chat_message(builder, result, err);
    if (rc == YVEX_OK) rc = render_literal(builder, ",\"finish_reason\":", err);
    if (rc == YVEX_OK) rc = render_text(builder, yvex_provider_finish_name(result->finish), err);
    if (rc == YVEX_OK) rc = render_literal(builder, "}],", err);
    if (rc == YVEX_OK) rc = render_chat_usage(builder, result, err);
    if (rc == YVEX_OK) rc = render_literal(builder, ",", err);
    if (rc == YVEX_OK) rc = render_completion_metrics(builder, result, err);
    return rc == YVEX_OK ? render_literal(builder, "}", err) : rc;
}

static int render_responses_result(render_builder *builder, const char *id,
                                   const char *model, unsigned long long created,
                                   const openai_generation_result *result,
                                   yvex_error *err)
{
    char prefix[384];
    unsigned long long index, output_count = 0u;
    const char *status = result->finish == YVEX_PROVIDER_FINISH_LENGTH
                             ? "incomplete" : "completed";
    int length = snprintf(prefix, sizeof(prefix),
        "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%llu,"
        "\"status\":\"%s\",\"model\":", id, created, status);
    int rc = length > 0 && (size_t)length < sizeof(prefix)
                 ? render_append(builder, prefix, (unsigned long long)length, err)
                 : YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK) rc = render_text(builder, model, err);
    if (rc == YVEX_OK) rc = render_literal(builder, ",\"output\":[", err);
    if (rc == YVEX_OK && (result->text_count || !result->tool_call_count)) {
        char message_id[YVEX_PROVIDER_ID_CAP];
        (void)snprintf(message_id, sizeof(message_id), "msg_%.48s", id);
        rc = render_literal(builder, "{\"type\":\"message\",\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, message_id, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                ",\"status\":\"completed\",\"role\":\"assistant\","
                "\"content\":[{\"type\":\"output_text\",\"text\":", err);
        if (rc == YVEX_OK)
            rc = render_string(builder, result->text, result->text_count, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder, ",\"annotations\":[]}]}", err);
        output_count = 1u;
    }
    for (index = 0u; rc == YVEX_OK && index < result->tool_call_count;
         ++index) {
        const openai_generation_tool_call *call = &result->tool_calls[index];
        if (output_count++) rc = render_literal(builder, ",", err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                                "{\"type\":\"function_call\",\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, call->call_id, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"call_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, call->call_id, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"name\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, call->name, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"arguments\":", err);
        if (rc == YVEX_OK)
            rc = render_string(builder, call->arguments,
                               call->arguments_count, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"status\":\"completed\"}", err);
    }
    if (rc == YVEX_OK) {
        rc = render_literal(builder, "],\"reasoning_content\":", err);
        if (rc == YVEX_OK)
            rc = render_string(builder, result->reasoning,
                               result->reasoning_count, err);
    }
    if (rc == YVEX_OK) rc = render_literal(builder, ",\"usage\":{", err);
    if (rc == YVEX_OK) {
        char usage[256];
        length = snprintf(usage, sizeof(usage),
            "\"input_tokens\":%llu,\"output_tokens\":%llu,\"total_tokens\":%llu,"
            "\"output_tokens_details\":{\"reasoning_tokens\":%llu}}",
            result->prompt_tokens, result->completion_tokens,
            result->total_tokens, result->reasoning_tokens);
        rc = length > 0 && (size_t)length < sizeof(usage)
                 ? render_append(builder, usage, (unsigned long long)length, err)
                 : YVEX_ERR_BOUNDS;
    }
    if (rc == YVEX_OK) rc = render_literal(builder, ",", err);
    if (rc == YVEX_OK) rc = render_completion_metrics(builder, result, err);
    if (rc == YVEX_OK && result->finish == YVEX_PROVIDER_FINISH_LENGTH)
        rc = render_literal(builder,
            ",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}", err);
    else if (rc == YVEX_OK)
        rc = render_literal(builder, ",\"incomplete_details\":null", err);
    return rc == YVEX_OK ? render_literal(builder, "}", err) : rc;
}

int openai_json_result(openai_endpoint endpoint, const char *id,
                       const char *model, unsigned long long created,
                       const openai_generation_result *result,
                       unsigned char **output, unsigned long long *count,
                       yvex_error *err)
{
    render_builder builder = {0};
    int rc;
    if (output) *output = NULL;
    if (count) *count = 0u;
    if (!id || !model || !result || !result->complete) return YVEX_ERR_INVALID_ARG;
    rc = endpoint == OPENAI_ENDPOINT_CHAT
             ? render_chat_result(&builder, id, model, created, result, err)
             : render_responses_result(&builder, id, model, created, result, err);
    if (rc != YVEX_OK) { free(builder.data); return rc; }
    return render_finish(&builder, output, count, err);
}

static int render_chat_chunk(render_builder *builder, const char *id,
                             const char *model, unsigned long long created,
                             const yvex_client_message *message,
                             unsigned long long tool_index, int initial,
                             yvex_error *err)
{
    char prefix[320];
    int length = snprintf(prefix, sizeof(prefix),
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%llu,\"model\":\"%s\",\"choices\":[{\"index\":0,"
        "\"delta\":", id, created, model);
    int rc = length > 0 && (size_t)length < sizeof(prefix)
                 ? render_append(builder, prefix, (unsigned long long)length, err)
                 : YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK && initial)
        rc = render_literal(builder, "{\"role\":\"assistant\"}", err);
    else if (rc == YVEX_OK && message &&
             message->provider_output_kind ==
                 YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING) {
        rc = render_literal(builder, "{\"reasoning_content\":", err);
        if (rc == YVEX_OK)
            rc = render_string(builder, message->bytes,
                               message->byte_count, err);
        if (rc == YVEX_OK) rc = render_literal(builder, "}", err);
    }
    else if (rc == YVEX_OK && message &&
             message->provider_output_kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL) {
        char index_text[64];
        int index_count = snprintf(index_text, sizeof(index_text),
                                   "{\"tool_calls\":[{\"index\":%llu",
                                   tool_index);
        rc = index_count > 0 && (size_t)index_count < sizeof(index_text)
                 ? render_append(builder, index_text,
                                 (unsigned long long)index_count, err)
                 : YVEX_ERR_BOUNDS;
        if (message->tool_call_id[0]) {
            if (rc == YVEX_OK) rc = render_literal(builder, ",\"id\":", err);
            if (rc == YVEX_OK) rc = render_text(builder, message->tool_call_id, err);
            if (rc == YVEX_OK)
                rc = render_literal(builder, ",\"type\":\"function\",\"function\":{\"name\":", err);
            if (rc == YVEX_OK) rc = render_text(builder, message->tool_name, err);
            if (rc == YVEX_OK) rc = render_literal(builder, ",\"arguments\":", err);
        } else if (rc == YVEX_OK) {
            rc = render_literal(builder, ",\"function\":{\"arguments\":", err);
        }
        if (rc == YVEX_OK)
            rc = render_string(builder, message->bytes, message->byte_count, err);
        if (rc == YVEX_OK) rc = render_literal(builder, "}}]}", err);
    } else if (rc == YVEX_OK && message &&
               message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
        rc = render_literal(builder, "{\"content\":", err);
        if (rc == YVEX_OK)
            rc = render_string(builder, message->bytes, message->byte_count, err);
        if (rc == YVEX_OK) rc = render_literal(builder, "}", err);
    } else if (rc == YVEX_OK) {
        rc = render_literal(builder, "{}", err);
    }
    if (rc == YVEX_OK) rc = render_literal(builder, ",\"finish_reason\":", err);
    if (rc == YVEX_OK) {
        if (message && message->kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE)
            rc = render_text(builder, yvex_provider_finish_name(message->provider_finish), err);
        else rc = render_literal(builder, "null", err);
    }
    return rc == YVEX_OK ? render_literal(builder, "}]}", err) : rc;
}

/*
 * Render one documented Responses streaming event payload.
 *
 * Returns before partial event bytes leave builder ownership.
 */
static int render_response_chunk(render_builder *builder, const char *id,
                                 const char *model,
                                 const yvex_client_message *message,
                                 int initial, yvex_error *err)
{
    int rc;
    if (initial) {
        rc = render_literal(builder, "{\"type\":\"response.created\",\"response\":{\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, id, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                ",\"object\":\"response\",\"status\":\"in_progress\",\"model\":",
                err);
        if (rc == YVEX_OK) rc = render_text(builder, model, err);
        return rc == YVEX_OK ? render_literal(builder, "}}", err) : rc;
    }
    if (message && message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT &&
        message->provider_output_kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL) {
        rc = render_literal(builder, "{\"type\":\"response.function_call_arguments.delta\",\"item_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, message->tool_call_id, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"output_index\":0,\"delta\":", err);
        if (rc == YVEX_OK) rc = render_string(builder, message->bytes, message->byte_count, err);
        return rc == YVEX_OK ? render_literal(builder, "}", err) : rc;
    }
    if (message && message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
        char message_id[YVEX_PROVIDER_ID_CAP];
        (void)snprintf(message_id, sizeof(message_id), "msg_%.48s", id);
        rc = render_literal(builder, "{\"type\":\"response.output_text.delta\",\"item_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, message_id, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder, ",\"output_index\":0,\"content_index\":0,\"delta\":", err);
        if (rc == YVEX_OK) rc = render_string(builder, message->bytes, message->byte_count, err);
        return rc == YVEX_OK ? render_literal(builder, "}", err) : rc;
    }
    rc = render_literal(builder, "{\"type\":\"response.completed\",\"response\":{\"id\":", err);
    if (rc == YVEX_OK) rc = render_text(builder, id, err);
    if (rc == YVEX_OK)
        rc = render_literal(builder,
            ",\"object\":\"response\",\"status\":\"completed\",\"model\":",
            err);
    if (rc == YVEX_OK) rc = render_text(builder, model, err);
    return rc == YVEX_OK ? render_literal(builder, "}}", err) : rc;
}

int openai_json_stream_chunk(openai_endpoint endpoint, const char *id,
                             const char *model, unsigned long long created,
                             const yvex_client_message *message,
                             unsigned long long tool_index, int initial,
                             unsigned char **output, unsigned long long *count,
                             yvex_error *err)
{
    render_builder builder = {0};
    int rc;
    if (output) *output = NULL;
    if (count) *count = 0u;
    rc = endpoint == OPENAI_ENDPOINT_CHAT
             ? render_chat_chunk(&builder, id, model, created, message,
                                 tool_index, initial, err)
             : render_response_chunk(&builder, id, model, message,
                                     initial, err);
    if (rc != YVEX_OK) { free(builder.data); return rc; }
    return render_finish(&builder, output, count, err);
}

int openai_json_chat_usage_chunk(const char *id, const char *model,
                                 unsigned long long created,
                                 const openai_generation_result *result,
                                 unsigned char **output,
                                 unsigned long long *count, yvex_error *err)
{
    render_builder builder = {0};
    char prefix[320];
    int length, rc;
    if (output) *output = NULL;
    if (count) *count = 0u;
    if (!id || !model || !result || !result->complete)
        return YVEX_ERR_INVALID_ARG;
    length = snprintf(prefix, sizeof(prefix),
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%llu,\"model\":", id, created);
    rc = length > 0 && (size_t)length < sizeof(prefix)
             ? render_append(&builder, prefix, (unsigned long long)length, err)
             : YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK) rc = render_text(&builder, model, err);
    if (rc == YVEX_OK) rc = render_literal(&builder, ",\"choices\":[],", err);
    if (rc == YVEX_OK) rc = render_chat_usage(&builder, result, err);
    if (rc == YVEX_OK) rc = render_literal(&builder, ",", err);
    if (rc == YVEX_OK) rc = render_completion_metrics(&builder, result, err);
    if (rc == YVEX_OK) rc = render_literal(&builder, "}", err);
    if (rc != YVEX_OK) {
        free(builder.data);
        return rc;
    }
    return render_finish(&builder, output, count, err);
}

static const char *response_event_name(openai_response_event_kind kind)
{
    static const char *const names[] = {
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.reasoning_content.delta",
        "response.reasoning_content.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
        "response.incomplete",
        "response.failed"
    };
    return kind <= OPENAI_RESPONSE_EVENT_FAILED ? names[kind] : NULL;
}

static const openai_generation_tool_call *response_tool_at(
    const openai_generation_result *result, unsigned long long output_index)
{
    unsigned long long first_tool = result->text_count ? 1u : 0u;
    if (output_index < first_tool ||
        output_index - first_tool >= result->tool_call_count)
        return NULL;
    return &result->tool_calls[output_index - first_tool];
}

static void response_item_id(const char *response_id,
                             const openai_generation_result *result,
                             unsigned long long output_index,
                             char output[YVEX_PROVIDER_ID_CAP])
{
    const openai_generation_tool_call *call =
        response_tool_at(result, output_index);
    if (call)
        (void)snprintf(output, YVEX_PROVIDER_ID_CAP, "%s",
                       call->call_id);
    else
        (void)snprintf(output, YVEX_PROVIDER_ID_CAP, "msg_%.48s",
                       response_id);
}

/*
 * Append one in-progress or completed Responses output item.
 *
 * Appends one text-message or function-call item with stable identity. Returns the first bounded
 * rendering error.
 */
static int render_response_item(render_builder *builder, const char *id,
                                const openai_generation_result *result,
                                unsigned long long output_index, int completed,
                                yvex_error *err)
{
    char item_id[YVEX_PROVIDER_ID_CAP];
    const openai_generation_tool_call *call =
        response_tool_at(result, output_index);
    int rc;
    response_item_id(id, result, output_index, item_id);
    if (call) {
        rc = render_literal(builder, "{\"type\":\"function_call\",\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, item_id, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"call_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, call->call_id, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"name\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, call->name, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"arguments\":", err);
        if (rc == YVEX_OK)
            rc = completed
                     ? render_string(builder, call->arguments,
                                     call->arguments_count, err)
                     : render_literal(builder, "\"\"", err);
    } else {
        rc = render_literal(builder, "{\"type\":\"message\",\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, item_id, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder, ",\"role\":\"assistant\",\"content\":[", err);
        if (rc == YVEX_OK && completed) {
            rc = render_literal(builder, "{\"type\":\"output_text\",\"text\":", err);
            if (rc == YVEX_OK)
                rc = render_string(builder, result->text, result->text_count, err);
            if (rc == YVEX_OK)
                rc = render_literal(builder, ",\"annotations\":[]}", err);
        }
        if (rc == YVEX_OK) rc = render_literal(builder, "]", err);
    }
    if (rc == YVEX_OK)
        rc = render_literal(builder, completed
                                         ? ",\"status\":\"completed\"}"
                                         : ",\"status\":\"in_progress\"}",
                            err);
    return rc;
}

/*
 * Append the event-specific payload after common type/sequence facts.
 *
 * Appends only fields admitted for the bounded Responses stream profile. Preserves builder
 * ownership for caller cleanup.
 */
static int render_response_event_payload(
    render_builder *builder, openai_response_event_kind kind,
    const char *id, const char *model, unsigned long long created,
    const yvex_client_message *message,
    const openai_generation_result *result, unsigned long long output_index,
    yvex_error *err)
{
    char item_id[YVEX_PROVIDER_ID_CAP];
    const openai_generation_tool_call *call =
        response_tool_at(result, output_index);
    int rc = YVEX_OK;
    response_item_id(id, result, output_index, item_id);
    if (kind == OPENAI_RESPONSE_EVENT_CREATED) {
        rc = render_literal(builder, ",\"response\":{\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, id, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                ",\"object\":\"response\",\"status\":\"in_progress\","
                "\"model\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, model, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder, ",\"output\":[]}", err);
    } else if (kind == OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_ADDED ||
               kind == OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_DONE) {
        char index_text[64];
        int count = snprintf(index_text, sizeof(index_text),
                             ",\"output_index\":%llu,\"item\":", output_index);
        rc = count > 0 && (size_t)count < sizeof(index_text)
                 ? render_append(builder, index_text,
                                 (unsigned long long)count, err)
                 : YVEX_ERR_BOUNDS;
        if (rc == YVEX_OK)
            rc = render_response_item(
                builder, id, result, output_index,
                kind == OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_DONE, err);
    } else if (kind == OPENAI_RESPONSE_EVENT_CONTENT_PART_ADDED ||
               kind == OPENAI_RESPONSE_EVENT_CONTENT_PART_DONE) {
        rc = render_literal(builder, ",\"item_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, item_id, err);
        if (rc == YVEX_OK) {
            char index_text[96];
            int count = snprintf(
                index_text, sizeof(index_text),
                ",\"output_index\":%llu,\"content_index\":0,\"part\":{"
                "\"type\":\"output_text\",\"text\":", output_index);
            rc = count > 0 && (size_t)count < sizeof(index_text)
                     ? render_append(builder, index_text,
                                     (unsigned long long)count, err)
                     : YVEX_ERR_BOUNDS;
        }
        if (rc == YVEX_OK)
            rc = kind == OPENAI_RESPONSE_EVENT_CONTENT_PART_DONE
                     ? render_string(builder, result->text,
                                     result->text_count, err)
                     : render_literal(builder, "\"\"", err);
        if (rc == YVEX_OK)
            rc = render_literal(builder, ",\"annotations\":[]}", err);
    } else if (kind == OPENAI_RESPONSE_EVENT_REASONING_DELTA ||
               kind == OPENAI_RESPONSE_EVENT_REASONING_DONE) {
        rc = render_literal(builder, ",\"output_index\":0,", err);
        if (rc == YVEX_OK)
            rc = render_literal(
                builder, kind == OPENAI_RESPONSE_EVENT_REASONING_DELTA
                             ? "\"delta\":" : "\"reasoning_content\":",
                err);
        if (rc == YVEX_OK)
            rc = kind == OPENAI_RESPONSE_EVENT_REASONING_DELTA
                     ? render_string(builder, message->bytes,
                                     message->byte_count, err)
                     : render_string(builder, result->reasoning,
                                     result->reasoning_count, err);
    } else if (kind == OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DELTA ||
               kind == OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DONE) {
        rc = render_literal(builder, ",\"item_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, item_id, err);
        if (rc == YVEX_OK) {
            char index_text[80];
            int count = snprintf(index_text, sizeof(index_text),
                                 ",\"output_index\":%llu,"
                                 "\"content_index\":0,", output_index);
            rc = count > 0 && (size_t)count < sizeof(index_text)
                     ? render_append(builder, index_text,
                                     (unsigned long long)count, err)
                     : YVEX_ERR_BOUNDS;
        }
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                kind == OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DELTA
                    ? "\"delta\":" : "\"text\":", err);
        if (rc == YVEX_OK)
            rc = kind == OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DELTA
                     ? render_string(builder, message->bytes,
                                     message->byte_count, err)
                     : render_string(builder, result->text,
                                     result->text_count, err);
    } else if (kind == OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DELTA ||
               kind == OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DONE) {
        rc = render_literal(builder, ",\"item_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, item_id, err);
        if (rc == YVEX_OK) {
            char index_text[48];
            int count = snprintf(index_text, sizeof(index_text),
                                 ",\"output_index\":%llu,", output_index);
            rc = count > 0 && (size_t)count < sizeof(index_text)
                     ? render_append(builder, index_text,
                                     (unsigned long long)count, err)
                     : YVEX_ERR_BOUNDS;
        }
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                kind == OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DELTA
                    ? "\"delta\":" : "\"arguments\":", err);
        if (rc == YVEX_OK)
            rc = kind == OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DELTA
                     ? render_string(builder, message->bytes,
                                     message->byte_count, err)
                     : call
                           ? render_string(builder, call->arguments,
                                           call->arguments_count, err)
                           : YVEX_ERR_STATE;
    } else if (kind == OPENAI_RESPONSE_EVENT_COMPLETED ||
               kind == OPENAI_RESPONSE_EVENT_INCOMPLETE) {
        rc = render_literal(builder, ",\"response\":", err);
        if (rc == YVEX_OK)
            rc = render_responses_result(builder, id, model, created,
                                         result, err);
    } else {
        rc = render_literal(builder, ",\"response\":{\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, id, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                ",\"object\":\"response\",\"status\":\"failed\","
                "\"model\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, model, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                ",\"error\":{\"code\":\"server_error\","
                "\"message\":\"YVEX generation failed\"}}", err);
    }
    return rc;
}

int openai_json_response_event(openai_response_event_kind kind,
                               const char *id, const char *model,
                               unsigned long long created,
                               const yvex_client_message *message,
                               const openai_generation_result *result,
                               unsigned long long output_index,
                               unsigned long long sequence,
                               unsigned char **output,
                               unsigned long long *count, yvex_error *err)
{
    render_builder builder = {0};
    char prefix[160];
    const char *name = response_event_name(kind);
    int length, rc;
    if (output) *output = NULL;
    if (count) *count = 0u;
    if (!name || !id || !model || !result) return YVEX_ERR_INVALID_ARG;
    length = snprintf(prefix, sizeof(prefix),
                      "{\"type\":\"%s\",\"sequence_number\":%llu",
                      name, sequence);
    rc = length > 0 && (size_t)length < sizeof(prefix)
             ? render_append(&builder, prefix, (unsigned long long)length, err)
             : YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = render_response_event_payload(&builder, kind, id, model,
                                           created, message, result,
                                           output_index, err);
    if (rc == YVEX_OK) rc = render_literal(&builder, "}", err);
    if (rc != YVEX_OK) {
        free(builder.data);
        return rc;
    }
    return render_finish(&builder, output, count, err);
}
