/* Owner: server.openai.render.
 * Owns: OpenAI-profile JSON/error/model/result objects and streaming event projections.
 * Does not own: JSON request admission, HTTP framing, YVEX counters, generation, or terminal UI.
 * Invariants: every string is escaped, usage is authoritative, and no internal path/identity leaks by default.
 * Boundary: provider-neutral/YVEX result facts become only the documented compatibility-profile objects.
 * Purpose: render stable Chat Completions and Responses JSON without terminal-oriented printers.
 * Inputs: explicit IDs/model, committed fragments, terminal usage/finish facts, and bounded output owner.
 * Effects: allocates one complete JSON document per call.
 * Failure: allocation/bounds errors publish no partial JSON owner. */

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

/* Purpose: reserve one checked render extent.
 * Inputs: builder, requested additional bytes, and error output.
 * Effects: grows unique render storage while retaining existing bytes.
 * Failure: preserves the prior owner on bounds or allocation failure.
 * Boundary: byte storage only; no JSON semantics are selected here. */
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

/* Purpose: append one explicit render span. */
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

/* Purpose: append one fixed rendering literal. */
static int render_literal(render_builder *builder, const char *text,
                          yvex_error *err)
{
    return render_append(builder, text, (unsigned long long)strlen(text), err);
}

/* Purpose: render one explicit UTF-8 span as a JSON string.
 * Inputs: builder, explicit bytes/count, and error output.
 * Effects: appends quotes and deterministic JSON escaping.
 * Failure: returns the first builder failure with no published document.
 * Boundary: escaping does not normalize or reinterpret UTF-8 content. */
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

/* Purpose: append one terminated public string through the same JSON escaping path. */
static int render_text(render_builder *builder, const char *text,
                       yvex_error *err)
{
    return render_string(builder, (const unsigned char *)(text ? text : ""),
                         text ? strlen(text) : 0u, err);
}

/* Purpose: publish one completed builder into caller ownership.
 * Inputs: complete builder and cleared output pointer/count.
 * Effects: transfers allocation ownership and clears the builder.
 * Failure: frees invalid candidate storage and publishes no output.
 * Boundary: publication occurs only after the full JSON document exists. */
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

/* Purpose: render one bounded OpenAI-compatible error envelope.
 * Inputs: public status/type/parameter/code/message and output owner.
 * Effects: allocates one complete escaped error JSON object.
 * Failure: frees partial rendering and leaves output null/count zero.
 * Boundary: contains no internal path, prompt, or model-state facts. */
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

/* Purpose: render the loaded-model object or list from authoritative daemon status.
 * Inputs: daemon summary, selected public model ID, list mode, and output owner.
 * Effects: allocates one model object or list document.
 * Failure: frees incomplete JSON and publishes no discovery result.
 * Boundary: projects readiness facts but does not discover or load models itself. */
int openai_json_models(const yvex_server_summary *summary,
                       const char *selected_model, int list,
                       unsigned char **output, unsigned long long *count,
                       yvex_error *err)
{
    render_builder builder = {0};
    int rc;
    if (output) *output = NULL;
    if (count) *count = 0u;
    if (!summary || !selected_model || !selected_model[0]) return YVEX_ERR_INVALID_ARG;
    rc = render_literal(&builder, list ? "{\"object\":\"list\",\"data\":["
                                       : "", err);
    if (rc == YVEX_OK)
        rc = render_literal(&builder, "{\"id\":", err);
    if (rc == YVEX_OK) rc = render_text(&builder, selected_model, err);
    if (rc == YVEX_OK)
        rc = render_literal(&builder,
            ",\"object\":\"model\",\"created\":0,\"owned_by\":\"yvex\","
            "\"yvex_profile\":\"" OPENAI_COMPAT_PROFILE "\"}", err);
    if (rc == YVEX_OK && list) rc = render_literal(&builder, "]}", err);
    if (rc != YVEX_OK) { free(builder.data); return rc; }
    return render_finish(&builder, output, count, err);
}

/* Purpose: append standard Chat usage counters without reuse-count corruption. */
static int render_chat_usage(render_builder *builder,
                             const openai_generation_result *result,
                             yvex_error *err)
{
    char values[256];
    int length = snprintf(values, sizeof(values),
        "\"usage\":{\"prompt_tokens\":%llu,\"completion_tokens\":%llu,"
        "\"total_tokens\":%llu}", result->prompt_tokens,
        result->completion_tokens, result->total_tokens);
    return length > 0 && (size_t)length < sizeof(values)
               ? render_append(builder, values, (unsigned long long)length, err)
               : YVEX_ERR_BOUNDS;
}

/* Purpose: append one Chat assistant message with optional exact function call.
 * Inputs: builder, committed aggregate result, and error output.
 * Effects: writes role/content plus one typed tool call when present.
 * Failure: returns the first rendering failure without publishing a response.
 * Boundary: arguments remain JSON text bytes and are never executed. */
static int render_chat_message(render_builder *builder,
                               const openai_generation_result *result,
                               yvex_error *err)
{
    int rc = render_literal(builder, "{\"role\":\"assistant\",\"content\":", err);
    if (rc == YVEX_OK) {
        if (result->has_tool_call && !result->text_count)
            rc = render_literal(builder, "null", err);
        else
            rc = render_string(builder, result->text, result->text_count, err);
    }
    if (rc == YVEX_OK && result->has_tool_call) {
        rc = render_literal(builder, ",\"tool_calls\":[{\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, result->tool_call_id, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                ",\"type\":\"function\",\"function\":{\"name\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, result->tool_name, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"arguments\":", err);
        if (rc == YVEX_OK)
            rc = render_string(builder, result->arguments,
                               result->arguments_count, err);
        if (rc == YVEX_OK) rc = render_literal(builder, "}}]", err);
    }
    return rc == YVEX_OK ? render_literal(builder, "}", err) : rc;
}

/* Purpose: render one complete non-stream Chat Completions response.
 * Inputs: builder, public IDs/time, authoritative result, and error output.
 * Effects: appends choices, finish reason, and exact usage counters.
 * Failure: leaves publication to the caller only after full completion.
 * Boundary: maps provider facts without changing token-count semantics. */
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
    return rc == YVEX_OK ? render_literal(builder, "}", err) : rc;
}

/* Purpose: render one complete bounded Responses object.
 * Inputs: builder, public IDs/time, authoritative result, and error output.
 * Effects: writes message or function-call output, status, usage, and incomplete facts.
 * Failure: returns the first render failure with no externally owned document.
 * Boundary: implements only the admitted text/function Responses profile. */
static int render_responses_result(render_builder *builder, const char *id,
                                   const char *model, unsigned long long created,
                                   const openai_generation_result *result,
                                   yvex_error *err)
{
    char prefix[384];
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
    if (rc == YVEX_OK && result->has_tool_call) {
        rc = render_literal(builder, "{\"type\":\"function_call\",\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, result->tool_call_id, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"call_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, result->tool_call_id, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"name\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, result->tool_name, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"arguments\":", err);
        if (rc == YVEX_OK)
            rc = render_string(builder, result->arguments,
                               result->arguments_count, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"status\":\"completed\"}", err);
    } else if (rc == YVEX_OK) {
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
    }
    if (rc == YVEX_OK) rc = render_literal(builder, "],\"usage\":{", err);
    if (rc == YVEX_OK) {
        char usage[256];
        length = snprintf(usage, sizeof(usage),
            "\"input_tokens\":%llu,\"output_tokens\":%llu,\"total_tokens\":%llu}",
            result->prompt_tokens, result->completion_tokens,
            result->total_tokens);
        rc = length > 0 && (size_t)length < sizeof(usage)
                 ? render_append(builder, usage, (unsigned long long)length, err)
                 : YVEX_ERR_BOUNDS;
    }
    if (rc == YVEX_OK && result->finish == YVEX_PROVIDER_FINISH_LENGTH)
        rc = render_literal(builder,
            ",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}", err);
    else if (rc == YVEX_OK)
        rc = render_literal(builder, ",\"incomplete_details\":null", err);
    return rc == YVEX_OK ? render_literal(builder, "}", err) : rc;
}

/* Purpose: render one complete endpoint-specific non-stream result.
 * Inputs: endpoint, IDs/time, completed result, output owner, and error output.
 * Effects: allocates and transfers one Chat or Responses JSON document.
 * Failure: clears output and frees partial builder storage.
 * Boundary: endpoint projection only; transport framing is separate. */
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

/* Purpose: render one Chat streaming role/content/tool/terminal chunk.
 * Inputs: builder, public IDs/time, protocol message, initial flag, and error output.
 * Effects: appends exactly one chat.completion.chunk object.
 * Failure: returns before the caller emits any incomplete SSE record.
 * Boundary: consumes only model-committed protocol fragments and terminal facts. */
static int render_chat_chunk(render_builder *builder, const char *id,
                             const char *model, unsigned long long created,
                             const yvex_client_message *message, int initial,
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
             message->provider_output_kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL) {
        rc = render_literal(builder, "{\"tool_calls\":[{\"index\":0", err);
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

/* Purpose: render one documented Responses streaming event payload.
 * Inputs: builder, public IDs/model, protocol message, initial flag, and error output.
 * Effects: appends one admitted Responses event object.
 * Failure: returns before partial event bytes leave builder ownership.
 * Boundary: event sequencing is orchestrator-owned; this function renders one event. */
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

/* Purpose: render one endpoint-specific SSE data object from a protocol-v4 message.
 * Inputs: endpoint, IDs/time, protocol message, initial flag, and output owner.
 * Effects: allocates and transfers one complete stream JSON object.
 * Failure: frees partial storage and leaves output null/count zero.
 * Boundary: performs no socket I/O and never fabricates uncommitted text. */
int openai_json_stream_chunk(openai_endpoint endpoint, const char *id,
                             const char *model, unsigned long long created,
                             const yvex_client_message *message, int initial,
                             unsigned char **output, unsigned long long *count,
                             yvex_error *err)
{
    render_builder builder = {0};
    int rc;
    if (output) *output = NULL;
    if (count) *count = 0u;
    rc = endpoint == OPENAI_ENDPOINT_CHAT
             ? render_chat_chunk(&builder, id, model, created, message,
                                 initial, err)
             : render_response_chunk(&builder, id, model, message,
                                     initial, err);
    if (rc != YVEX_OK) { free(builder.data); return rc; }
    return render_finish(&builder, output, count, err);
}

/* Purpose: render the optional terminal Chat streaming usage chunk.
 * Inputs: public IDs/time, completed authoritative result, output owner, and error output.
 * Effects: allocates one chat.completion.chunk with empty choices and exact usage.
 * Failure: frees partial storage and leaves output null/count zero.
 * Boundary: usage values remain YVEX counters and exclude prefix-reuse extensions. */
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
    if (rc == YVEX_OK) rc = render_literal(&builder, "}", err);
    if (rc != YVEX_OK) {
        free(builder.data);
        return rc;
    }
    return render_finish(&builder, output, count, err);
}

/* Purpose: map one bounded Responses stream event kind to its public type. */
static const char *response_event_name(openai_response_event_kind kind)
{
    static const char *const names[] = {
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
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

/* Purpose: derive the stable output-item ID used by every event for one response. */
static void response_item_id(const char *response_id,
                             const openai_generation_result *result,
                             char output[YVEX_PROVIDER_ID_CAP])
{
    if (result->has_tool_call)
        (void)snprintf(output, YVEX_PROVIDER_ID_CAP, "%s",
                       result->tool_call_id);
    else
        (void)snprintf(output, YVEX_PROVIDER_ID_CAP, "msg_%.48s",
                       response_id);
}

/* Purpose: append one in-progress or completed Responses output item.
 * Inputs: builder, response/result facts, completion state, and error output.
 * Effects: appends one text-message or function-call item with stable identity.
 * Failure: returns the first bounded rendering error.
 * Boundary: item bytes are committed provider output and are never executed. */
static int render_response_item(render_builder *builder, const char *id,
                                const openai_generation_result *result,
                                int completed, yvex_error *err)
{
    char item_id[YVEX_PROVIDER_ID_CAP];
    int rc;
    response_item_id(id, result, item_id);
    if (result->has_tool_call) {
        rc = render_literal(builder, "{\"type\":\"function_call\",\"id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, item_id, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"call_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, result->tool_call_id, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"name\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, result->tool_name, err);
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"arguments\":", err);
        if (rc == YVEX_OK)
            rc = completed
                     ? render_string(builder, result->arguments,
                                     result->arguments_count, err)
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

/* Purpose: append the event-specific payload after common type/sequence facts.
 * Inputs: event kind, IDs/model, optional fragment/result, and error output.
 * Effects: appends only fields admitted for the bounded Responses stream profile.
 * Failure: preserves builder ownership for caller cleanup.
 * Boundary: event order is controlled by the gateway orchestration owner. */
static int render_response_event_payload(
    render_builder *builder, openai_response_event_kind kind,
    const char *id, const char *model, unsigned long long created,
    const yvex_client_message *message,
    const openai_generation_result *result, yvex_error *err)
{
    char item_id[YVEX_PROVIDER_ID_CAP];
    int rc = YVEX_OK;
    response_item_id(id, result, item_id);
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
        rc = render_literal(builder, ",\"output_index\":0,\"item\":", err);
        if (rc == YVEX_OK)
            rc = render_response_item(
                builder, id, result,
                kind == OPENAI_RESPONSE_EVENT_OUTPUT_ITEM_DONE, err);
    } else if (kind == OPENAI_RESPONSE_EVENT_CONTENT_PART_ADDED ||
               kind == OPENAI_RESPONSE_EVENT_CONTENT_PART_DONE) {
        rc = render_literal(builder, ",\"item_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, item_id, err);
        if (rc == YVEX_OK)
            rc = render_literal(
                builder, ",\"output_index\":0,\"content_index\":0,\"part\":{"
                         "\"type\":\"output_text\",\"text\":", err);
        if (rc == YVEX_OK)
            rc = kind == OPENAI_RESPONSE_EVENT_CONTENT_PART_DONE
                     ? render_string(builder, result->text,
                                     result->text_count, err)
                     : render_literal(builder, "\"\"", err);
        if (rc == YVEX_OK)
            rc = render_literal(builder, ",\"annotations\":[]}", err);
    } else if (kind == OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DELTA ||
               kind == OPENAI_RESPONSE_EVENT_OUTPUT_TEXT_DONE) {
        rc = render_literal(builder, ",\"item_id\":", err);
        if (rc == YVEX_OK) rc = render_text(builder, item_id, err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                ",\"output_index\":0,\"content_index\":0,", err);
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
        if (rc == YVEX_OK) rc = render_literal(builder, ",\"output_index\":0,", err);
        if (rc == YVEX_OK)
            rc = render_literal(builder,
                kind == OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DELTA
                    ? "\"delta\":" : "\"arguments\":", err);
        if (rc == YVEX_OK)
            rc = kind == OPENAI_RESPONSE_EVENT_FUNCTION_ARGUMENTS_DELTA
                     ? render_string(builder, message->bytes,
                                     message->byte_count, err)
                     : render_string(builder, result->arguments,
                                     result->arguments_count, err);
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

/* Purpose: render one fully sequenced Responses streaming event.
 * Inputs: event kind, public IDs/time, optional committed fragment/result, sequence, and output owner.
 * Effects: allocates one complete profile-v1 event document.
 * Failure: frees partial bytes and publishes no event.
 * Boundary: produces application syntax only and never mutates runtime state. */
int openai_json_response_event(openai_response_event_kind kind,
                               const char *id, const char *model,
                               unsigned long long created,
                               const yvex_client_message *message,
                               const openai_generation_result *result,
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
                                           created, message, result, err);
    if (rc == YVEX_OK) rc = render_literal(&builder, "}", err);
    if (rc != YVEX_OK) {
        free(builder.data);
        return rc;
    }
    return render_finish(&builder, output, count, err);
}
