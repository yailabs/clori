/*
 * Admit application intent once before any transport-specific translation or runtime handoff.
 *
 * Every published request is bounded, complete, transport-neutral, and identity-authenticated.
 * Protocol and gateway owners consume this ABI without gaining model/runtime authority.
 */

#include <yvex/provider.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>

#define PROVIDER_WIRE_MAGIC UINT32_C(0x59505231)

typedef struct {
    unsigned char *data;
    size_t capacity, count;
} provider_writer;

typedef struct {
    const unsigned char *data;
    size_t count, offset;
} provider_reader;

void yvex_provider_sampling_default(yvex_provider_sampling *sampling)
{
    if (!sampling)
        return;
    memset(sampling, 0, sizeof(*sampling));
    sampling->temperature = 1.0;
    sampling->top_p = 1.0;
    sampling->typical_p = 1.0;
}

/*
 * Initialize one complete provider request policy without owning request payloads.
 *
 * Model, messages, tools, stops, and adapter identity remain caller-supplied.
 */
void yvex_provider_request_default(yvex_provider_request *request)
{
    if (!request)
        return;
    memset(request, 0, sizeof(*request));
    request->schema_version = YVEX_PROVIDER_SCHEMA_V1;
    request->response_format = YVEX_PROVIDER_RESPONSE_TEXT;
    request->maximum_output_tokens = 128u;
    request->tool_choice.kind = YVEX_PROVIDER_TOOL_CHOICE_AUTO;
    yvex_provider_sampling_default(&request->sampling);
}

const char *yvex_provider_role_name(yvex_provider_role role)
{
    switch (role) {
    case YVEX_PROVIDER_ROLE_DEVELOPER: return "developer";
    case YVEX_PROVIDER_ROLE_SYSTEM: return "system";
    case YVEX_PROVIDER_ROLE_USER: return "user";
    case YVEX_PROVIDER_ROLE_ASSISTANT: return "assistant";
    case YVEX_PROVIDER_ROLE_TOOL: return "tool";
    }
    return "unknown";
}

const char *yvex_provider_finish_name(yvex_provider_finish_class finish)
{
    switch (finish) {
    case YVEX_PROVIDER_FINISH_STOP: return "stop";
    case YVEX_PROVIDER_FINISH_LENGTH: return "length";
    case YVEX_PROVIDER_FINISH_TOOL_CALLS: return "tool_calls";
    case YVEX_PROVIDER_FINISH_CANCELLED: return "cancelled";
    case YVEX_PROVIDER_FINISH_FAILED: return "failed";
    }
    return "unknown";
}

static int provider_refuse(yvex_error *err, yvex_status status,
                           const char *reason)
{
    yvex_error_set(err, status, "provider.request", reason);
    return status;
}

static int span_valid(yvex_provider_span span, unsigned long long maximum,
                      int allow_empty)
{
    return span.count <= maximum && (span.bytes || !span.count) &&
           (allow_empty || span.count != 0u) && span.count <= SIZE_MAX;
}

static int identifier_valid(const char *value, size_t capacity, int allow_empty)
{
    size_t index, count;
    if (!value) return 0;
    count = strnlen(value, capacity);
    if (count == capacity || (!allow_empty && !count)) return 0;
    for (index = 0u; index < count; ++index) {
        unsigned char byte = (unsigned char)value[index];
        if (!((byte >= 'a' && byte <= 'z') ||
              (byte >= 'A' && byte <= 'Z') ||
              (byte >= '0' && byte <= '9') || byte == '_' || byte == '-' ||
              byte == '.' || byte == ':'))
            return 0;
    }
    return 1;
}

int yvex_provider_json_value_validate(const unsigned char *bytes,
                                      unsigned long long byte_count,
                                      int require_object, yvex_error *err)
{
    yvex_json json;
    const unsigned char *cursor = bytes;
    unsigned long long remaining = byte_count;
    if (!bytes || !byte_count || byte_count > YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES ||
        byte_count > SIZE_MAX)
        return provider_refuse(err, YVEX_ERR_BOUNDS,
                               "bounded JSON bytes are required");
    while (remaining && (*cursor == ' ' || *cursor == '\t' ||
                         *cursor == '\r' || *cursor == '\n')) {
        ++cursor;
        --remaining;
    }
    if (!remaining || (require_object && *cursor != '{'))
        return provider_refuse(err, YVEX_ERR_FORMAT,
                               "JSON object syntax is required");
    yvex_json_init(&json, (const char *)bytes, (size_t)byte_count);
    if (!yvex_json_skip_value(&json) || !yvex_json_complete(&json))
        return provider_refuse(err, YVEX_ERR_FORMAT,
                               "one complete JSON value is required");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int tool_valid(const yvex_provider_function_tool *tool,
                      yvex_error *err)
{
    if (!tool || !identifier_valid(tool->name, sizeof(tool->name), 0) ||
        !span_valid(tool->description, YVEX_PROVIDER_MAX_MESSAGE_BYTES, 1) ||
        !span_valid(tool->parameters_json,
                    YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES, 0) ||
        (tool->strict != 0 && tool->strict != 1))
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "function tool fields are invalid or oversized");
    return yvex_provider_json_value_validate(
        tool->parameters_json.bytes, tool->parameters_json.count, 1, err);
}

/*
 * Validate one provider message and its optional single typed tool call.
 *
 * Advances the counter only for completely valid bounded spans. Returns false for invalid roles,
 * call IDs, JSON, or count overflow.
 */
static int message_valid(const yvex_provider_message *message,
                         yvex_error *err)
{
    unsigned long long index;
    if (!message || message->role > YVEX_PROVIDER_ROLE_TOOL ||
        !span_valid(message->content, YVEX_PROVIDER_MAX_MESSAGE_BYTES,
                    message->role == YVEX_PROVIDER_ROLE_ASSISTANT) ||
        message->tool_call_count > 1u ||
        (!message->tool_calls && message->tool_call_count))
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "provider message fields are invalid or oversized");
    if (message->role == YVEX_PROVIDER_ROLE_TOOL) {
        if (!identifier_valid(message->tool_call_id,
                              sizeof(message->tool_call_id), 0) ||
            message->tool_call_count)
            return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "tool results require exactly one call ID");
    } else if (message->tool_call_id[0]) {
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "only tool-result messages carry tool_call_id");
    }
    if (message->tool_call_count && message->role != YVEX_PROVIDER_ROLE_ASSISTANT)
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "only assistant messages carry tool calls");
    for (index = 0u; index < message->tool_call_count; ++index) {
        const yvex_provider_tool_call *call = &message->tool_calls[index];
        if (!identifier_valid(call->call_id, sizeof(call->call_id), 0) ||
            !identifier_valid(call->name, sizeof(call->name), 0) ||
            !span_valid(call->arguments_json,
                        YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES, 0))
            return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "assistant tool-call fields are invalid");
        if (yvex_provider_json_value_validate(
                call->arguments_json.bytes, call->arguments_json.count,
                1, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return YVEX_OK;
}

/* Validate all provider-neutral semantics without relying on an identity claim. */
static int request_fields_validate(const yvex_provider_request *request,
                                   yvex_error *err)
{
    unsigned long long index, other, total = 0u;
    int named_found = 0;
    if (!request || request->schema_version != YVEX_PROVIDER_SCHEMA_V1 ||
        !identifier_valid(request->model, sizeof(request->model), 0) ||
        !request->messages || !request->message_count ||
        request->message_count > YVEX_PROVIDER_MAX_MESSAGES ||
        request->tool_count > YVEX_PROVIDER_MAX_TOOLS ||
        (!request->tools && request->tool_count) ||
        request->stop_count > YVEX_PROVIDER_MAX_STOPS ||
        (!request->stop_strings && request->stop_count) ||
        request->tool_choice.kind > YVEX_PROVIDER_TOOL_CHOICE_FUNCTION ||
        request->tool_choice.parallel_calls ||
        request->response_format > YVEX_PROVIDER_RESPONSE_JSON_OBJECT ||
        !request->maximum_output_tokens ||
        request->maximum_output_tokens > UINT32_MAX ||
        (request->stream != 0 && request->stream != 1) ||
        (request->include_usage != 0 && request->include_usage != 1) ||
        !isfinite(request->sampling.temperature) ||
        !isfinite(request->sampling.top_p) ||
        !isfinite(request->sampling.min_p) ||
        !isfinite(request->sampling.typical_p))
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "complete bounded provider request fields are required");
    if (request->sampling.temperature < 0.0 ||
        request->sampling.temperature > 2.0 ||
        request->sampling.top_p < 0.0 || request->sampling.top_p > 1.0 ||
        request->sampling.min_p < 0.0 || request->sampling.min_p > 1.0 ||
        request->sampling.typical_p < 0.0 || request->sampling.typical_p > 1.0)
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "sampling values are outside the admitted range");
    for (index = 0u; index < request->message_count; ++index) {
        int rc = message_valid(&request->messages[index], err);
        if (rc != YVEX_OK) return rc;
        if (total > YVEX_PROVIDER_MAX_CONTENT_BYTES -
                    request->messages[index].content.count)
            return provider_refuse(err, YVEX_ERR_BOUNDS,
                                   "aggregate message bytes exceed the request limit");
        total += request->messages[index].content.count;
    }
    for (index = 0u; index < request->tool_count; ++index) {
        int rc = tool_valid(&request->tools[index], err);
        if (rc != YVEX_OK) return rc;
        for (other = 0u; other < index; ++other)
            if (strcmp(request->tools[index].name,
                       request->tools[other].name) == 0)
                return provider_refuse(err, YVEX_ERR_FORMAT,
                                       "function tool names must be unique");
        if (request->tool_choice.kind == YVEX_PROVIDER_TOOL_CHOICE_FUNCTION &&
            strcmp(request->tool_choice.function_name,
                   request->tools[index].name) == 0)
            named_found = 1;
    }
    if ((request->tool_choice.kind == YVEX_PROVIDER_TOOL_CHOICE_REQUIRED ||
         request->tool_choice.kind == YVEX_PROVIDER_TOOL_CHOICE_FUNCTION) &&
        !request->tool_count)
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "required tool choice needs a tool definition");
    if (request->tool_choice.kind == YVEX_PROVIDER_TOOL_CHOICE_FUNCTION &&
        (!identifier_valid(request->tool_choice.function_name,
                           sizeof(request->tool_choice.function_name), 0) ||
         !named_found))
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "named tool choice does not match a definition");
    for (index = 0u; index < request->stop_count; ++index)
        if (!span_valid(request->stop_strings[index],
                        YVEX_PROVIDER_MAX_STOP_BYTES, 0))
            return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "stop string is empty or oversized");
    if (request->adapter[0] &&
        !identifier_valid(request->adapter, sizeof(request->adapter), 0))
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "provider adapter identifier is invalid");
    if (request->previous_response_id[0] &&
        !identifier_valid(request->previous_response_id,
                          sizeof(request->previous_response_id), 0))
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "previous response ID is invalid");
    if (request->external_correlation_id[0] &&
        !identifier_valid(request->external_correlation_id,
                          sizeof(request->external_correlation_id), 0))
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "external correlation ID is invalid");
    return YVEX_OK;
}

static int hash_span(yvex_sha256 *hash, yvex_provider_span span)
{
    return yvex_sha256_update_u64_be(hash, span.count) &&
           yvex_sha256_update(hash, span.bytes, (size_t)span.count);
}

static int hash_double(yvex_sha256 *hash, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64_be(hash, bits);
}

/*
 * Derive the canonical field-wise request identity after complete validation.
 *
 * Writes one field-wise SHA-256 identity. Clears no ownership but returns false when canonical
 * hashing fails.
 */
static int request_identity(yvex_provider_request *request)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, call;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.provider.request.v1") ||
        !yvex_sha256_update_u64_be(&hash, request->schema_version) ||
        !yvex_sha256_update_text(&hash, request->model) ||
        !yvex_sha256_update_u64_be(&hash, request->message_count))
        return 0;
    for (index = 0u; index < request->message_count; ++index) {
        const yvex_provider_message *message = &request->messages[index];
        if (!yvex_sha256_update_u64_be(&hash, message->role) ||
            !hash_span(&hash, message->content) ||
            !yvex_sha256_update_text(&hash, message->tool_call_id) ||
            !yvex_sha256_update_u64_be(&hash, message->tool_call_count))
            return 0;
        for (call = 0u; call < message->tool_call_count; ++call)
            if (!yvex_sha256_update_text(&hash, message->tool_calls[call].call_id) ||
                !yvex_sha256_update_text(&hash, message->tool_calls[call].name) ||
                !hash_span(&hash, message->tool_calls[call].arguments_json))
                return 0;
    }
    if (!yvex_sha256_update_u64_be(&hash, request->tool_count)) return 0;
    for (index = 0u; index < request->tool_count; ++index)
        if (!yvex_sha256_update_text(&hash, request->tools[index].name) ||
            !hash_span(&hash, request->tools[index].description) ||
            !hash_span(&hash, request->tools[index].parameters_json) ||
            !yvex_sha256_update_u64_be(&hash, request->tools[index].strict))
            return 0;
    if (!yvex_sha256_update_u64_be(&hash, request->stop_count)) return 0;
    for (index = 0u; index < request->stop_count; ++index)
        if (!hash_span(&hash, request->stop_strings[index])) return 0;
    if (!yvex_sha256_update_u64_be(&hash, request->tool_choice.kind) ||
        !yvex_sha256_update_text(&hash, request->tool_choice.function_name) ||
        !yvex_sha256_update_u64_be(&hash, request->tool_choice.parallel_calls) ||
        !yvex_sha256_update_u64_be(&hash, request->response_format) ||
        !yvex_sha256_update_u64_be(&hash, request->sampling.stochastic) ||
        !yvex_sha256_update_u64_be(&hash, request->sampling.seed_present) ||
        !yvex_sha256_update_u64_be(&hash, request->sampling.seed) ||
        !yvex_sha256_update_u64_be(&hash, request->sampling.top_k) ||
        !hash_double(&hash, request->sampling.temperature) ||
        !hash_double(&hash, request->sampling.top_p) ||
        !hash_double(&hash, request->sampling.min_p) ||
        !hash_double(&hash, request->sampling.typical_p) ||
        !yvex_sha256_update_u64_be(&hash, request->maximum_output_tokens) ||
        !yvex_sha256_update_u64_be(&hash, request->stream) ||
        !yvex_sha256_update_u64_be(&hash, request->include_usage) ||
        !yvex_sha256_update_text(&hash, request->adapter) ||
        !yvex_sha256_update_text(&hash, request->previous_response_id) ||
        !yvex_sha256_update_text(&hash, request->external_correlation_id) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, request->request_identity);
    return 1;
}

/*
 * Validate and identity-seal one caller-owned provider request transactionally.
 *
 * Writes identity/sealed on success. Clears sealed/identity and reports the exact semantic
 * refusal.
 */
int yvex_provider_request_seal(yvex_provider_request *request, yvex_error *err)
{
    int rc = request_fields_validate(request, err);
    if (rc != YVEX_OK) {
        if (request) {
            request->sealed = 0;
            request->request_identity[0] = '\0';
        }
        return rc;
    }
    if (!request_identity(request))
        return provider_refuse(err, YVEX_ERR_STATE,
                               "provider request identity derivation failed");
    request->sealed = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Reconstruct and compare the complete provider request identity.
 *
 * Refuses any field mutation or malformed identity.
 */
int yvex_provider_request_validate(const yvex_provider_request *request,
                                   yvex_error *err)
{
    yvex_provider_request candidate;
    if (!request || !request->sealed)
        return provider_refuse(err, YVEX_ERR_STATE,
                               "sealed provider request is required");
    candidate = *request;
    candidate.sealed = 0;
    candidate.request_identity[0] = '\0';
    if (request_fields_validate(&candidate, err) != YVEX_OK ||
        !request_identity(&candidate) ||
        strcmp(candidate.request_identity, request->request_identity) != 0)
        return provider_refuse(err, YVEX_ERR_STATE,
                               "provider request identity mismatch");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int span_clone(yvex_provider_span source, yvex_provider_span *target)
{
    unsigned char *bytes;
    memset(target, 0, sizeof(*target));
    if (!source.count) return 1;
    bytes = malloc((size_t)source.count);
    if (!bytes) return 0;
    memcpy(bytes, source.bytes, (size_t)source.count);
    target->bytes = bytes;
    target->count = source.count;
    return 1;
}

void yvex_provider_request_close(yvex_provider_request **request)
{
    yvex_provider_request *owned;
    unsigned long long index, call;
    if (!request || !*request) return;
    owned = *request;
    for (index = 0u; index < owned->message_count; ++index) {
        yvex_provider_message *message = (yvex_provider_message *)&owned->messages[index];
        free((void *)message->content.bytes);
        for (call = 0u; call < message->tool_call_count; ++call)
            free((void *)message->tool_calls[call].arguments_json.bytes);
        free((void *)message->tool_calls);
    }
    for (index = 0u; index < owned->tool_count; ++index) {
        yvex_provider_function_tool *tool =
            (yvex_provider_function_tool *)&owned->tools[index];
        free((void *)tool->description.bytes);
        free((void *)tool->parameters_json.bytes);
    }
    for (index = 0u; index < owned->stop_count; ++index)
        free((void *)owned->stop_strings[index].bytes);
    free((void *)owned->messages);
    free((void *)owned->tools);
    free((void *)owned->stop_strings);
    free(owned);
    *request = NULL;
}

/*
 * Deep-clone one validated request into a unique lifecycle owner.
 *
 * Allocates every nested provider span and preserves exact identity.
 */
int yvex_provider_request_clone(const yvex_provider_request *source,
                                yvex_provider_request **out, yvex_error *err)
{
    yvex_provider_request *copy;
    yvex_provider_message *messages = NULL;
    yvex_provider_function_tool *tools = NULL;
    yvex_provider_span *stops = NULL;
    unsigned long long index, call;
    if (out) *out = NULL;
    if (!out || yvex_provider_request_validate(source, err) != YVEX_OK)
        return yvex_error_code(err);
    copy = calloc(1u, sizeof(*copy));
    if (!copy) return provider_refuse(err, YVEX_ERR_NOMEM,
                                      "provider request allocation failed");
    *copy = *source;
    copy->messages = NULL;
    copy->tools = NULL;
    copy->stop_strings = NULL;
    if (source->message_count)
        messages = calloc((size_t)source->message_count, sizeof(*messages));
    if (source->tool_count)
        tools = calloc((size_t)source->tool_count, sizeof(*tools));
    if (source->stop_count)
        stops = calloc((size_t)source->stop_count, sizeof(*stops));
    copy->messages = messages;
    copy->tools = tools;
    copy->stop_strings = stops;
    if ((source->message_count && !messages) || (source->tool_count && !tools) ||
        (source->stop_count && !stops)) goto no_memory;
    for (index = 0u; index < source->message_count; ++index) {
        messages[index] = source->messages[index];
        messages[index].content.bytes = NULL;
        messages[index].tool_calls = NULL;
        if (!span_clone(source->messages[index].content,
                        &messages[index].content)) goto no_memory;
        if (source->messages[index].tool_call_count) {
            yvex_provider_tool_call *calls = calloc(
                (size_t)source->messages[index].tool_call_count, sizeof(*calls));
            if (!calls) goto no_memory;
            messages[index].tool_calls = calls;
            for (call = 0u; call < source->messages[index].tool_call_count; ++call) {
                calls[call] = source->messages[index].tool_calls[call];
                calls[call].arguments_json.bytes = NULL;
                if (!span_clone(source->messages[index].tool_calls[call].arguments_json,
                                &calls[call].arguments_json)) goto no_memory;
            }
        }
    }
    for (index = 0u; index < source->tool_count; ++index) {
        tools[index] = source->tools[index];
        tools[index].description.bytes = NULL;
        tools[index].parameters_json.bytes = NULL;
        if (!span_clone(source->tools[index].description, &tools[index].description) ||
            !span_clone(source->tools[index].parameters_json,
                        &tools[index].parameters_json)) goto no_memory;
    }
    for (index = 0u; index < source->stop_count; ++index)
        if (!span_clone(source->stop_strings[index], &stops[index])) goto no_memory;
    *out = copy;
    yvex_error_clear(err);
    return YVEX_OK;
no_memory:
    yvex_provider_request_close(&copy);
    return provider_refuse(err, YVEX_ERR_NOMEM,
                           "provider request clone allocation failed");
}

static int write_u32(provider_writer *writer, uint32_t value)
{
    if (!writer || writer->count > writer->capacity ||
        writer->capacity - writer->count < 4u) return 0;
    writer->data[writer->count++] = (unsigned char)(value >> 24u);
    writer->data[writer->count++] = (unsigned char)(value >> 16u);
    writer->data[writer->count++] = (unsigned char)(value >> 8u);
    writer->data[writer->count++] = (unsigned char)value;
    return 1;
}

static int write_u64(provider_writer *writer, uint64_t value)
{
    unsigned int index;
    if (!writer || writer->count > writer->capacity ||
        writer->capacity - writer->count < 8u) return 0;
    for (index = 0u; index < 8u; ++index)
        writer->data[writer->count++] =
            (unsigned char)(value >> (56u - 8u * index));
    return 1;
}

static int write_double(provider_writer *writer, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return write_u64(writer, bits);
}

static int write_span(provider_writer *writer, yvex_provider_span span)
{
    if (span.count > UINT32_MAX || span.count > SIZE_MAX ||
        !write_u32(writer, (uint32_t)span.count) ||
        writer->count > writer->capacity || span.count > writer->capacity - writer->count)
        return 0;
    if (span.count) memcpy(writer->data + writer->count, span.bytes, (size_t)span.count);
    writer->count += (size_t)span.count;
    return 1;
}

static int write_text(provider_writer *writer, const char *text, size_t capacity)
{
    size_t count = strnlen(text, capacity);
    yvex_provider_span span = {(const unsigned char *)text, count};
    return count < capacity && write_span(writer, span);
}

/* Encode one sealed provider request in deterministic field order. */
int yvex_provider_request_wire_encode(const yvex_provider_request *request,
                                      unsigned char *output,
                                      unsigned long long capacity,
                                      unsigned long long *byte_count,
                                      yvex_error *err)
{
    provider_writer writer;
    unsigned long long index, call;
    if (byte_count) *byte_count = 0u;
    if (!output || !byte_count || capacity > SIZE_MAX ||
        capacity > YVEX_PROVIDER_WIRE_MAX_BYTES ||
        yvex_provider_request_validate(request, err) != YVEX_OK)
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "sealed request and bounded wire output are required");
    writer.data = output;
    writer.capacity = (size_t)capacity;
    writer.count = 0u;
#define W32(value) write_u32(&writer, (uint32_t)(value))
#define W64(value) write_u64(&writer, (uint64_t)(value))
    if (!W32(PROVIDER_WIRE_MAGIC) || !W32(YVEX_PROVIDER_WIRE_SCHEMA_V1) ||
        !write_text(&writer, request->model, sizeof(request->model)) ||
        !W32(request->message_count)) goto bounds;
    for (index = 0u; index < request->message_count; ++index) {
        const yvex_provider_message *message = &request->messages[index];
        if (!W32(message->role) || !write_span(&writer, message->content) ||
            !write_text(&writer, message->tool_call_id,
                        sizeof(message->tool_call_id)) ||
            !W32(message->tool_call_count)) goto bounds;
        for (call = 0u; call < message->tool_call_count; ++call)
            if (!write_text(&writer, message->tool_calls[call].call_id,
                            sizeof(message->tool_calls[call].call_id)) ||
                !write_text(&writer, message->tool_calls[call].name,
                            sizeof(message->tool_calls[call].name)) ||
                !write_span(&writer, message->tool_calls[call].arguments_json))
                goto bounds;
    }
    if (!W32(request->tool_count)) goto bounds;
    for (index = 0u; index < request->tool_count; ++index)
        if (!write_text(&writer, request->tools[index].name,
                        sizeof(request->tools[index].name)) ||
            !write_span(&writer, request->tools[index].description) ||
            !write_span(&writer, request->tools[index].parameters_json) ||
            !W32(request->tools[index].strict)) goto bounds;
    if (!W32(request->stop_count)) goto bounds;
    for (index = 0u; index < request->stop_count; ++index)
        if (!write_span(&writer, request->stop_strings[index])) goto bounds;
    if (!W32(request->tool_choice.kind) ||
        !write_text(&writer, request->tool_choice.function_name,
                    sizeof(request->tool_choice.function_name)) ||
        !W32(request->tool_choice.parallel_calls) ||
        !W32(request->response_format) || !W32(request->sampling.stochastic) ||
        !W32(request->sampling.seed_present) || !W64(request->sampling.seed) ||
        !W64(request->sampling.top_k) ||
        !write_double(&writer, request->sampling.temperature) ||
        !write_double(&writer, request->sampling.top_p) ||
        !write_double(&writer, request->sampling.min_p) ||
        !write_double(&writer, request->sampling.typical_p) ||
        !W64(request->maximum_output_tokens) || !W32(request->stream) ||
        !W32(request->include_usage) ||
        !write_text(&writer, request->adapter, sizeof(request->adapter)) ||
        !write_text(&writer, request->previous_response_id,
                    sizeof(request->previous_response_id)) ||
        !write_text(&writer, request->external_correlation_id,
                    sizeof(request->external_correlation_id)) ||
        !write_text(&writer, request->request_identity,
                    sizeof(request->request_identity))) goto bounds;
#undef W32
#undef W64
    *byte_count = writer.count;
    yvex_error_clear(err);
    return YVEX_OK;
bounds:
#undef W32
#undef W64
    return provider_refuse(err, YVEX_ERR_BOUNDS,
                           "provider request exceeds wire capacity");
}

static int read_u32(provider_reader *reader, uint32_t *value)
{
    const unsigned char *bytes;
    if (!reader || !value || reader->offset > reader->count ||
        reader->count - reader->offset < 4u) return 0;
    bytes = reader->data + reader->offset;
    *value = ((uint32_t)bytes[0] << 24u) | ((uint32_t)bytes[1] << 16u) |
             ((uint32_t)bytes[2] << 8u) | bytes[3];
    reader->offset += 4u;
    return 1;
}

static int read_u64(provider_reader *reader, uint64_t *value)
{
    unsigned int index;
    if (!reader || !value || reader->offset > reader->count ||
        reader->count - reader->offset < 8u) return 0;
    *value = 0u;
    for (index = 0u; index < 8u; ++index)
        *value = (*value << 8u) | reader->data[reader->offset++];
    return 1;
}

static int read_double(provider_reader *reader, double *value)
{
    uint64_t bits;
    if (!read_u64(reader, &bits)) return 0;
    memcpy(value, &bits, sizeof(bits));
    return 1;
}

static int read_span(provider_reader *reader, yvex_provider_span *span,
                     unsigned long long maximum)
{
    uint32_t count;
    unsigned char *copy = NULL;
    if (!read_u32(reader, &count) || count > maximum ||
        reader->offset > reader->count || count > reader->count - reader->offset)
        return 0;
    if (count) {
        copy = malloc(count);
        if (!copy) return 0;
        memcpy(copy, reader->data + reader->offset, count);
    }
    reader->offset += count;
    span->bytes = copy;
    span->count = count;
    return 1;
}

static int read_text(provider_reader *reader, char *output, size_t capacity)
{
    yvex_provider_span span = {0};
    int valid;
    if (!read_span(reader, &span, capacity ? capacity - 1u : 0u)) return 0;
    valid = span.count < capacity &&
            (!span.count || !memchr(span.bytes, '\0', (size_t)span.count));
    if (valid) {
        if (span.count) memcpy(output, span.bytes, (size_t)span.count);
        output[span.count] = '\0';
    }
    free((void *)span.bytes);
    return valid;
}

/*
 * Decode one canonical provider request into a complete owned graph.
 *
 * Allocates, validates, and publishes one identity-authenticated request.
 */
int yvex_provider_request_wire_decode(const unsigned char *input,
                                      unsigned long long byte_count,
                                      yvex_provider_request **out,
                                      yvex_error *err)
{
    provider_reader reader;
    yvex_provider_request *request = NULL;
    yvex_provider_message *messages = NULL;
    yvex_provider_function_tool *tools = NULL;
    yvex_provider_span *stops = NULL;
    char claimed[YVEX_PROVIDER_ID_CAP];
    uint32_t magic, schema, count32, value32;
    uint64_t value64;
    unsigned long long index, call;
    if (out) *out = NULL;
    if (!input || !out || !byte_count || byte_count > YVEX_PROVIDER_WIRE_MAX_BYTES ||
        byte_count > SIZE_MAX)
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "bounded provider wire bytes are required");
    reader.data = input;
    reader.count = (size_t)byte_count;
    reader.offset = 0u;
    request = calloc(1u, sizeof(*request));
    if (!request) goto no_memory;
    if (!read_u32(&reader, &magic) || !read_u32(&reader, &schema) ||
        magic != PROVIDER_WIRE_MAGIC || schema != YVEX_PROVIDER_WIRE_SCHEMA_V1 ||
        !read_text(&reader, request->model, sizeof(request->model)) ||
        !read_u32(&reader, &count32) || count32 > YVEX_PROVIDER_MAX_MESSAGES)
        goto malformed;
    request->schema_version = YVEX_PROVIDER_SCHEMA_V1;
    request->message_count = count32;
    messages = calloc(count32 ? count32 : 1u, sizeof(*messages));
    if (!messages) goto no_memory;
    request->messages = messages;
    for (index = 0u; index < request->message_count; ++index) {
        if (!read_u32(&reader, &value32) || value32 > YVEX_PROVIDER_ROLE_TOOL ||
            !read_span(&reader, &messages[index].content,
                       YVEX_PROVIDER_MAX_MESSAGE_BYTES) ||
            !read_text(&reader, messages[index].tool_call_id,
                       sizeof(messages[index].tool_call_id)) ||
            !read_u32(&reader, &count32) || count32 > 1u)
            goto malformed;
        messages[index].role = (yvex_provider_role)value32;
        messages[index].tool_call_count = count32;
        if (count32) {
            yvex_provider_tool_call *calls = calloc(count32, sizeof(*calls));
            if (!calls) goto no_memory;
            messages[index].tool_calls = calls;
            for (call = 0u; call < count32; ++call)
                if (!read_text(&reader, calls[call].call_id, sizeof(calls[call].call_id)) ||
                    !read_text(&reader, calls[call].name, sizeof(calls[call].name)) ||
                    !read_span(&reader, &calls[call].arguments_json,
                               YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES))
                    goto malformed;
        }
    }
    if (!read_u32(&reader, &count32) || count32 > YVEX_PROVIDER_MAX_TOOLS)
        goto malformed;
    request->tool_count = count32;
    tools = calloc(count32 ? count32 : 1u, sizeof(*tools));
    if (!tools) goto no_memory;
    request->tools = tools;
    for (index = 0u; index < request->tool_count; ++index) {
        if (!read_text(&reader, tools[index].name, sizeof(tools[index].name)) ||
            !read_span(&reader, &tools[index].description,
                       YVEX_PROVIDER_MAX_MESSAGE_BYTES) ||
            !read_span(&reader, &tools[index].parameters_json,
                       YVEX_PROVIDER_MAX_TOOL_SCHEMA_BYTES) ||
            !read_u32(&reader, &value32) || value32 > 1u)
            goto malformed;
        tools[index].strict = (int)value32;
    }
    if (!read_u32(&reader, &count32) || count32 > YVEX_PROVIDER_MAX_STOPS)
        goto malformed;
    request->stop_count = count32;
    stops = calloc(count32 ? count32 : 1u, sizeof(*stops));
    if (!stops) goto no_memory;
    request->stop_strings = stops;
    for (index = 0u; index < request->stop_count; ++index)
        if (!read_span(&reader, &stops[index], YVEX_PROVIDER_MAX_STOP_BYTES))
            goto malformed;
    if (!read_u32(&reader, &value32) || value32 > YVEX_PROVIDER_TOOL_CHOICE_FUNCTION)
        goto malformed;
    request->tool_choice.kind = (yvex_provider_tool_choice_kind)value32;
    if (!read_text(&reader, request->tool_choice.function_name,
                   sizeof(request->tool_choice.function_name)) ||
        !read_u32(&reader, &value32) || value32 > 1u)
        goto malformed;
    request->tool_choice.parallel_calls = (int)value32;
    if (!read_u32(&reader, &value32) || value32 > YVEX_PROVIDER_RESPONSE_JSON_OBJECT)
        goto malformed;
    request->response_format = (yvex_provider_response_format)value32;
    if (!read_u32(&reader, &value32) || value32 > 1u) goto malformed;
    request->sampling.stochastic = (int)value32;
    if (!read_u32(&reader, &value32) || value32 > 1u) goto malformed;
    request->sampling.seed_present = (int)value32;
    if (!read_u64(&reader, &value64)) goto malformed;
    request->sampling.seed = value64;
    if (!read_u64(&reader, &value64)) goto malformed;
    request->sampling.top_k = value64;
    if (!read_double(&reader, &request->sampling.temperature) ||
        !read_double(&reader, &request->sampling.top_p) ||
        !read_double(&reader, &request->sampling.min_p) ||
        !read_double(&reader, &request->sampling.typical_p) ||
        !read_u64(&reader, &value64)) goto malformed;
    request->maximum_output_tokens = value64;
    if (!read_u32(&reader, &value32) || value32 > 1u) goto malformed;
    request->stream = (int)value32;
    if (!read_u32(&reader, &value32) || value32 > 1u) goto malformed;
    request->include_usage = (int)value32;
    if (!read_text(&reader, request->adapter, sizeof(request->adapter)) ||
        !read_text(&reader, request->previous_response_id,
                   sizeof(request->previous_response_id)) ||
        !read_text(&reader, request->external_correlation_id,
                   sizeof(request->external_correlation_id)) ||
        !read_text(&reader, claimed, sizeof(claimed)) ||
        reader.offset != reader.count ||
        yvex_provider_request_seal(request, err) != YVEX_OK ||
        strcmp(claimed, request->request_identity) != 0)
        goto malformed;
    *out = request;
    yvex_error_clear(err);
    return YVEX_OK;
malformed:
    yvex_provider_request_close(&request);
    return provider_refuse(err, YVEX_ERR_FORMAT,
                           "provider wire request is malformed or unauthenticated");
no_memory:
    yvex_provider_request_close(&request);
    return provider_refuse(err, YVEX_ERR_NOMEM,
                           "provider wire request allocation failed");
}

/*
 * Identity-seal one provider output after checked usage and item semantics.
 *
 * Writes output identity/completed fact. Clears completion/identity and reports inconsistent usage
 * or item fields. Output evidence only; it does not publish bytes to an application.
 */
int yvex_provider_output_seal(yvex_provider_output *output, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!output || output->schema_version != YVEX_PROVIDER_SCHEMA_V1 ||
        output->kind > YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING ||
        output->finish > YVEX_PROVIDER_FINISH_FAILED ||
        !span_valid(output->bytes, YVEX_PROVIDER_MAX_CONTENT_BYTES, 1) ||
        output->prompt_tokens > ULLONG_MAX - output->completion_tokens ||
        output->total_tokens != output->prompt_tokens + output->completion_tokens ||
        !identifier_valid(output->request_identity,
                          sizeof(output->request_identity), 0))
        return provider_refuse(err, YVEX_ERR_INVALID_ARG,
                               "complete bounded provider output is required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.provider.output.v1") ||
        !yvex_sha256_update_u64_be(&hash, output->kind) ||
        !yvex_sha256_update_u64_be(&hash, output->finish) ||
        !hash_span(&hash, output->bytes) ||
        !yvex_sha256_update_text(&hash, output->tool_call.call_id) ||
        !yvex_sha256_update_text(&hash, output->tool_call.name) ||
        !hash_span(&hash, output->tool_call.arguments_json) ||
        !yvex_sha256_update_u64_be(&hash, output->prompt_tokens) ||
        !yvex_sha256_update_u64_be(&hash, output->completion_tokens) ||
        !yvex_sha256_update_u64_be(&hash, output->total_tokens) ||
        !yvex_sha256_update_text(&hash, output->request_identity) ||
        !yvex_sha256_final(&hash, digest))
        return provider_refuse(err, YVEX_ERR_STATE,
                               "provider output identity derivation failed");
    yvex_sha256_hex(digest, output->output_identity);
    output->completed = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Recompute and compare every authoritative provider output field.
 *
 * Rejects any semantic mutation or malformed identity.
 */
int yvex_provider_output_validate(const yvex_provider_output *output,
                                  yvex_error *err)
{
    yvex_provider_output candidate;
    if (!output || !output->completed)
        return provider_refuse(err, YVEX_ERR_STATE,
                               "completed provider output is required");
    candidate = *output;
    candidate.completed = 0;
    candidate.output_identity[0] = '\0';
    if (yvex_provider_output_seal(&candidate, err) != YVEX_OK ||
        strcmp(candidate.output_identity, output->output_identity) != 0)
        return provider_refuse(err, YVEX_ERR_STATE,
                               "provider output identity mismatch");
    yvex_error_clear(err);
    return YVEX_OK;
}
