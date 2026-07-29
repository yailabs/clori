/* Owner: server.protocol.
 * Owns: canonical local frame/TLV encoding, bounded decoding, UDS client lifecycle, and paths.
 * Does not own: server listening, sessions, generation, telemetry semantics, or terminal output.
 * Invariants: wire integers are big-endian, duplicate known fields refuse, and frames are bounded.
 * Boundary: public local client and reusable canonical message codec.
 * Purpose: transport typed client requests and server messages without engine linkage.
 * Inputs: explicit byte spans, typed requests/messages, and one verified local socket path.
 * Effects: client APIs own one AF_UNIX descriptor; codecs mutate only caller-owned buffers.
 * Failure: malformed, oversized, foreign-owned, or partial frames publish no typed message. */
#define _POSIX_C_SOURCE 200809L

#include "src/server/private.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#define FRAME_HEADER_BYTES 12u
#define FRAME_KIND_REQUEST 1u
#define FRAME_KIND_MESSAGE 2u
#define TLV_HEADER_BYTES 8u
#define TLV_U64_BYTES 8u

enum {
    TAG_OPERATION = 1,
    TAG_REQUEST_NUMBER,
    TAG_SESSION_NAME,
    TAG_PROMPT,
    TAG_MAXIMUM_NEW_TOKENS,
    TAG_FLAGS,
    TAG_SEED,
    TAG_TEMPERATURE,
    TAG_TOP_K,
    TAG_TOP_P,
    TAG_MIN_P,
    TAG_TYPICAL_P,
    TAG_EVENT_AFTER,
    TAG_TRACE_LEVEL,
    TAG_PROVIDER_REQUEST,
    TAG_MESSAGE_KIND = 32,
    TAG_STATUS,
    TAG_REASON,
    TAG_BYTES,
    TAG_PROMPT_TOKENS,
    TAG_REUSED_TOKENS,
    TAG_PREFILL_TOKENS,
    TAG_GENERATED_TOKENS,
    TAG_FINAL_POSITION,
    TAG_QUEUE_SECONDS,
    TAG_PREFILL_SECONDS,
    TAG_FIRST_TOKEN_SECONDS,
    TAG_DECODE_SECONDS,
    TAG_PREFILL_RATE,
    TAG_DECODE_RATE,
    TAG_STOP_REASON,
    TAG_SESSION_STATE,
    TAG_SESSION_IDENTITY,
    TAG_TURN_IDENTITY,
    TAG_STATE_DIGEST,
    TAG_GENERATED_TOKEN_IDENTITY,
    TAG_GENERATED_TEXT_DIGEST,
    TAG_RUNTIME_STATUS = 64,
    TAG_RUNTIME_BACKEND,
    TAG_SOCKET_PATH,
    TAG_TARGET_ID,
    TAG_RUNTIME_MODEL_ID,
    TAG_RUNTIME_BINDING_ID,
    TAG_ARTIFACT_ID,
    TAG_CONTEXT_CAPACITY,
    TAG_SESSION_COUNT,
    TAG_RUNTIME_REQUEST_COUNT,
    TAG_RUNTIME_FLAGS,
    TAG_METRICS = 80,
    TAG_EVENT_SEQUENCE = 96,
    TAG_EVENT_WALL_TIME,
    TAG_EVENT_MONOTONIC_TIME,
    TAG_EVENT_KIND,
    TAG_EVENT_SEVERITY,
    TAG_EVENT_REQUEST_ID,
    TAG_EVENT_TURN_ID,
    TAG_EVENT_PHASE,
    TAG_EVENT_VALUE_A,
    TAG_EVENT_VALUE_B,
    TAG_EVENT_VALUE_C,
    TAG_EVENT_SECONDS,
    TAG_EVENT_RATE,
    TAG_EVENT_VARIANT_ID,
    TAG_EVENT_IDENTITY,
    TAG_EVENT_PROCESS_ID,
    TAG_EVENT_RUNTIME_MODEL_ID,
    TAG_EVENT_ARTIFACT_ID,
    TAG_EVENT_SESSION_ID,
    TAG_PHYSICAL_VARIANT_ID,
    TAG_PROVIDER_OUTPUT_KIND,
    TAG_PROVIDER_FINISH,
    TAG_COMPLETION_TOKENS,
    TAG_TOTAL_TOKENS,
    TAG_PROVIDER_REQUEST_ID,
    TAG_EXTERNAL_CORRELATION_ID,
    TAG_TOOL_CALL_ID,
    TAG_TOOL_NAME,
    TAG_EVENT_PROVIDER_ADAPTER,
    TAG_EVENT_PROVIDER_REQUEST_ID,
    TAG_EVENT_EXTERNAL_CORRELATION_ID,
    TAG_FAILURE_CLASS
};

typedef struct {
    unsigned char *data;
    unsigned long long capacity, count;
} wire_writer;

typedef struct {
    const unsigned char *data;
    unsigned long long count, offset;
    uint64_t seen[2];
} wire_reader;

struct yvex_client {
    int fd;
};

_Static_assert(sizeof(double) == 8u, "local protocol requires binary64 double");
_Static_assert(TAG_FAILURE_CLASS < 128u,
               "known protocol tags must fit the duplicate-field set");

/* Purpose: publish one protocol refusal without preserving parser-local state. */
static int protocol_refuse(yvex_error *err, yvex_status status,
                           const char *reason)
{
    yvex_error_set(err, status, "server.protocol", reason);
    return status;
}

/* Purpose: encode one 16-bit integer in canonical network byte order. */
static void put_u16(unsigned char *out, uint16_t value)
{
    out[0] = (unsigned char)(value >> 8u);
    out[1] = (unsigned char)value;
}

/* Purpose: encode one 32-bit integer in canonical network byte order. */
static void put_u32(unsigned char *out, uint32_t value)
{
    out[0] = (unsigned char)(value >> 24u);
    out[1] = (unsigned char)(value >> 16u);
    out[2] = (unsigned char)(value >> 8u);
    out[3] = (unsigned char)value;
}

/* Purpose: encode one 64-bit integer in canonical network byte order. */
static void put_u64(unsigned char *out, uint64_t value)
{
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        out[index] = (unsigned char)(value >> (56u - 8u * index));
}

/* Purpose: decode one canonical 16-bit integer. */
static uint16_t get_u16(const unsigned char *input)
{
    return (uint16_t)(((uint16_t)input[0] << 8u) | input[1]);
}

/* Purpose: decode one canonical 32-bit integer. */
static uint32_t get_u32(const unsigned char *input)
{
    return ((uint32_t)input[0] << 24u) | ((uint32_t)input[1] << 16u) |
           ((uint32_t)input[2] << 8u) | input[3];
}

/* Purpose: decode one canonical 64-bit integer. */
static uint64_t get_u64(const unsigned char *input)
{
    uint64_t value = 0u;
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        value = (value << 8u) | input[index];
    return value;
}

/* Purpose: append one bounded field to a canonical TLV stream.
 * Inputs: writer, tag, explicit bytes, and extent. Effects: advances output on success.
 * Failure: returns false on capacity or invalid span. Boundary: caller owns schema/tag semantics. */
static int writer_field(wire_writer *writer, unsigned int tag,
                        const void *bytes, unsigned long long count)
{
    unsigned long long required;
    if (!writer || tag > UINT16_MAX || count > UINT32_MAX ||
        count > writer->capacity || writer->count > writer->capacity - count ||
        writer->count + count > writer->capacity - TLV_HEADER_BYTES)
        return 0;
    required = writer->count + TLV_HEADER_BYTES + count;
    if (required > writer->capacity)
        return 0;
    put_u16(writer->data + writer->count, (uint16_t)tag);
    put_u16(writer->data + writer->count + 2u, 0u);
    put_u32(writer->data + writer->count + 4u, (uint32_t)count);
    if (count)
        memcpy(writer->data + writer->count + TLV_HEADER_BYTES, bytes,
               (size_t)count);
    writer->count = required;
    return 1;
}

/* Purpose: append one canonical integer field.
 * Inputs: writer, tag, and integer. Effects: appends a fixed-width TLV.
 * Failure: returns false when output capacity is exhausted. Boundary: no native byte order leaks. */
static int writer_u64(wire_writer *writer, unsigned int tag,
                      unsigned long long value)
{
    unsigned char bytes[TLV_U64_BYTES];
    put_u64(bytes, value);
    return writer_field(writer, tag, bytes, sizeof(bytes));
}

/* Purpose: append one canonical binary64 field.
 * Inputs: writer, tag, and finite-or-schema-admitted value. Effects: appends canonical IEEE bytes.
 * Failure: returns false on output exhaustion. Boundary: field policy remains with the message owner. */
static int writer_double(wire_writer *writer, unsigned int tag, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return writer_u64(writer, tag, bits);
}

/* Purpose: append one explicit-length text field without terminator semantics.
 * Inputs: writer, tag, text bytes, and extent. Effects: appends one bounded TLV.
 * Failure: returns false on invalid span or capacity. Boundary: wire length, not strlen, is authoritative. */
static int writer_text(wire_writer *writer, unsigned int tag,
                       const char *text)
{
    return !text || !text[0] ||
           writer_field(writer, tag, text,
                        (unsigned long long)strlen(text));
}

/* Purpose: consume one TLV while rejecting duplicate known tags.
 * Inputs: reader and field outputs. Effects: advances cursor and marks the tag observed.
 * Failure: returns false for truncation, duplicate tags, or invalid tag range. Boundary: payload remains borrowed. */
static int reader_next(wire_reader *reader, unsigned int *tag,
                       const unsigned char **bytes, unsigned long long *count)
{
    uint32_t length;
    unsigned int word, bit;
    if (reader->offset == reader->count)
        return 0;
    if (reader->offset > reader->count ||
        reader->count - reader->offset < TLV_HEADER_BYTES)
        return -1;
    *tag = get_u16(reader->data + reader->offset);
    length = get_u32(reader->data + reader->offset + 4u);
    reader->offset += TLV_HEADER_BYTES;
    if (length > reader->count - reader->offset)
        return -1;
    if (*tag < 128u) {
        word = *tag / 64u;
        bit = *tag % 64u;
        if (reader->seen[word] & (UINT64_C(1) << bit))
            return -1;
        reader->seen[word] |= UINT64_C(1) << bit;
    }
    *bytes = reader->data + reader->offset;
    *count = length;
    reader->offset += length;
    return 1;
}

/* Purpose: decode one exact unsigned 64-bit field.
 * Inputs: field bytes, length, and output. Effects: writes output on exact geometry.
 * Failure: returns false for any noncanonical length. Boundary: no implicit narrowing. */
static int reader_u64(const unsigned char *bytes, unsigned long long count,
                      unsigned long long *value)
{
    if (count != TLV_U64_BYTES)
        return 0;
    *value = get_u64(bytes);
    return 1;
}

/* Purpose: decode one exact canonical binary64 field.
 * Inputs: field bytes, length, and output. Effects: writes reconstructed value.
 * Failure: returns false for any noncanonical length. Boundary: semantic finiteness is checked by caller. */
static int reader_double(const unsigned char *bytes, unsigned long long count,
                         double *value)
{
    unsigned long long canonical;
    uint64_t bits;
    if (!reader_u64(bytes, count, &canonical))
        return 0;
    bits = canonical;
    memcpy(value, &bits, sizeof(bits));
    return 1;
}

/* Purpose: copy one bounded wire text field and append a process-local terminator.
 * Inputs: destination/capacity and wire bytes/extent. Effects: writes text plus trailing NUL.
 * Failure: returns false on absent or oversized spans. Boundary: wire extent remains semantic authority. */
static int reader_text(const unsigned char *bytes, unsigned long long count,
                       char *output, size_t capacity)
{
    if (!capacity || count >= capacity || (count && memchr(bytes, '\0', (size_t)count)))
        return 0;
    if (count)
        memcpy(output, bytes, (size_t)count);
    output[count] = '\0';
    return 1;
}

/* Purpose: encode one request field by field for server-independent transport.
 * Inputs: admitted request, output bytes/capacity, and count/error outputs. Effects: writes canonical payload.
 * Failure: publishes zero count for schema, span, or capacity refusal. Boundary: adds no outer frame. */
int yvex_protocol_request_encode(const yvex_client_request *request,
                                 unsigned char *output,
                                 unsigned long long capacity,
                                 unsigned long long *byte_count,
                                 yvex_error *err)
{
    wire_writer writer = {output, capacity, 0u};
    unsigned char *provider_bytes = NULL;
    unsigned long long provider_count = 0u;
    unsigned long long flags;
    int provider_rc = YVEX_OK;
    if (byte_count) *byte_count = 0u;
    if (!request || !output || !byte_count ||
        request->schema_version != YVEX_LOCAL_PROTOCOL_VERSION ||
        request->operation > YVEX_CLIENT_OP_ARTIFACT_VERIFY ||
        request->prompt_bytes > YVEX_SERVER_FRAME_MAX_BYTES ||
        (!request->prompt && request->prompt_bytes) ||
        (request->prompt_bytes && request->provider_request))
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "complete bounded client request is required");
    if (request->provider_request) {
        provider_bytes = malloc(YVEX_PROVIDER_WIRE_MAX_BYTES);
        if (!provider_bytes)
            return protocol_refuse(err, YVEX_ERR_NOMEM,
                                   "provider request wire allocation failed");
        provider_rc = yvex_provider_request_wire_encode(
            request->provider_request, provider_bytes,
            YVEX_PROVIDER_WIRE_MAX_BYTES, &provider_count, err);
        if (provider_rc != YVEX_OK) {
            free(provider_bytes);
            return provider_rc;
        }
    }
    flags = (request->stochastic ? 1u : 0u) |
            (request->seed_present ? 2u : 0u) |
            (request->trace_content ? 4u : 0u);
    if (!writer_u64(&writer, TAG_OPERATION, request->operation) ||
        !writer_u64(&writer, TAG_REQUEST_NUMBER, request->request_number) ||
        !writer_text(&writer, TAG_SESSION_NAME, request->session_name) ||
        !writer_field(&writer, TAG_PROMPT, request->prompt,
                      request->prompt_bytes) ||
        !writer_u64(&writer, TAG_MAXIMUM_NEW_TOKENS,
                    request->maximum_new_tokens) ||
        !writer_u64(&writer, TAG_FLAGS, flags) ||
        !writer_u64(&writer, TAG_SEED, request->seed) ||
        !writer_double(&writer, TAG_TEMPERATURE, request->temperature) ||
        !writer_u64(&writer, TAG_TOP_K, request->top_k) ||
        !writer_double(&writer, TAG_TOP_P, request->top_p) ||
        !writer_double(&writer, TAG_MIN_P, request->min_p) ||
        !writer_double(&writer, TAG_TYPICAL_P, request->typical_p) ||
        !writer_u64(&writer, TAG_EVENT_AFTER, request->event_after_sequence) ||
        !writer_u64(&writer, TAG_TRACE_LEVEL, request->trace_level) ||
        !writer_field(&writer, TAG_PROVIDER_REQUEST, provider_bytes,
                      provider_count)) {
        free(provider_bytes);
        return protocol_refuse(err, YVEX_ERR_BOUNDS,
                               "request does not fit the admitted frame");
    }
    free(provider_bytes);
    *byte_count = writer.count;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: decode one complete request transactionally and own only its prompt copy.
 * Inputs: payload bytes/extent, request output, prompt-owner output, and error output. Effects: may allocate prompt.
 * Failure: frees partial prompt and clears outputs. Boundary: unknown or duplicate executable fields refuse. */
int yvex_protocol_request_decode(const unsigned char *input,
                                 unsigned long long byte_count,
                                 yvex_client_request *request,
                                 unsigned char **owned_prompt,
                                 yvex_provider_request **owned_provider,
                                 yvex_error *err)
{
    wire_reader reader = {input, byte_count, 0u, {0u, 0u}};
    yvex_client_request candidate;
    yvex_provider_request *provider = NULL;
    unsigned char *prompt = NULL;
    const unsigned char *bytes;
    unsigned long long count, value;
    unsigned int tag;
    int next, valid = 1, have_operation = 0;
    if (owned_prompt) *owned_prompt = NULL;
    if (owned_provider) *owned_provider = NULL;
    if (!input || !request || !owned_prompt || !owned_provider ||
        byte_count > YVEX_SERVER_FRAME_MAX_BYTES)
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "bounded request bytes and outputs are required");
    memset(&candidate, 0, sizeof(candidate));
    candidate.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    while ((next = reader_next(&reader, &tag, &bytes, &count)) > 0 && valid) {
        switch (tag) {
        case TAG_OPERATION:
            valid = reader_u64(bytes, count, &value) &&
                    value <= YVEX_CLIENT_OP_ARTIFACT_VERIFY;
            candidate.operation = (yvex_client_operation)value;
            have_operation = valid;
            break;
        case TAG_REQUEST_NUMBER:
            valid = reader_u64(bytes, count, &candidate.request_number);
            break;
        case TAG_SESSION_NAME:
            valid = reader_text(bytes, count, candidate.session_name,
                                sizeof(candidate.session_name));
            break;
        case TAG_PROMPT:
            if (count) {
                prompt = malloc((size_t)count);
                valid = prompt != NULL;
                if (valid) memcpy(prompt, bytes, (size_t)count);
            }
            candidate.prompt = prompt;
            candidate.prompt_bytes = count;
            break;
        case TAG_MAXIMUM_NEW_TOKENS:
            valid = reader_u64(bytes, count, &candidate.maximum_new_tokens);
            break;
        case TAG_FLAGS:
            valid = reader_u64(bytes, count, &value) && !(value & ~7u);
            candidate.stochastic = (value & 1u) != 0u;
            candidate.seed_present = (value & 2u) != 0u;
            candidate.trace_content = (value & 4u) != 0u;
            break;
        case TAG_SEED: valid = reader_u64(bytes, count, &candidate.seed); break;
        case TAG_TEMPERATURE: valid = reader_double(bytes, count, &candidate.temperature); break;
        case TAG_TOP_K: valid = reader_u64(bytes, count, &candidate.top_k); break;
        case TAG_TOP_P: valid = reader_double(bytes, count, &candidate.top_p); break;
        case TAG_MIN_P: valid = reader_double(bytes, count, &candidate.min_p); break;
        case TAG_TYPICAL_P: valid = reader_double(bytes, count, &candidate.typical_p); break;
        case TAG_EVENT_AFTER: valid = reader_u64(bytes, count, &candidate.event_after_sequence); break;
        case TAG_TRACE_LEVEL:
            valid = reader_u64(bytes, count, &value) && value <= YVEX_SERVER_TRACE_FULL;
            candidate.trace_level = (yvex_server_trace_level)value;
            break;
        case TAG_PROVIDER_REQUEST:
            valid = !provider &&
                    (!count || yvex_provider_request_wire_decode(
                        bytes, count, &provider, err) == YVEX_OK);
            candidate.provider_request = provider;
            break;
        default: break;
        }
    }
    if (next < 0 || !valid || !have_operation ||
        (candidate.prompt_bytes && candidate.provider_request)) {
        free(prompt);
        yvex_provider_request_close(&provider);
        return protocol_refuse(err, YVEX_ERR_FORMAT,
                               "request frame contains malformed or duplicate fields");
    }
    *request = candidate;
    *owned_prompt = prompt;
    *owned_provider = provider;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: append the fixed process metrics vector within one typed blob.
 * Inputs: writer and authoritative metrics snapshot. Effects: appends canonical counters in schema order.
 * Failure: returns false on capacity exhaustion. Boundary: metrics meaning is owned by telemetry. */
static int writer_metrics(wire_writer *writer, const yvex_server_metrics *metrics)
{
    unsigned char bytes[25u * 8u];
    const unsigned long long values[] = {
        metrics->schema_version, metrics->uptime_ns, metrics->model_open_count,
        metrics->model_close_count, metrics->artifact_open_count,
        metrics->binding_open_count, metrics->materialization_count,
        metrics->residency_build_count, metrics->output_head_upload_count,
        metrics->current_rss_bytes, metrics->peak_rss_bytes,
        metrics->mapped_artifact_bytes, metrics->resident_host_bytes,
        metrics->resident_device_bytes, metrics->queue_depth,
        metrics->queue_capacity, metrics->active_sessions,
        metrics->total_sessions, metrics->active_requests,
        metrics->completed_requests, metrics->failed_requests,
        metrics->cancelled_requests, metrics->telemetry_dropped};
    unsigned int index;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index)
        put_u64(bytes + index * 8u, values[index]);
    return writer_field(writer, TAG_METRICS, bytes,
                        sizeof(values) / sizeof(values[0]) * 8u);
}

/* Purpose: decode the exact process metrics vector.
 * Inputs: field bytes/extent and metrics output. Effects: replaces the output on exact geometry.
 * Failure: returns false on length mismatch. Boundary: does not infer missing metrics. */
static int reader_metrics(const unsigned char *bytes, unsigned long long count,
                          yvex_server_metrics *metrics)
{
    unsigned long long *values[] = {
        NULL, &metrics->uptime_ns,
        &metrics->model_open_count, &metrics->model_close_count,
        &metrics->artifact_open_count, &metrics->binding_open_count,
        &metrics->materialization_count, &metrics->residency_build_count,
        &metrics->output_head_upload_count, &metrics->current_rss_bytes,
        &metrics->peak_rss_bytes, &metrics->mapped_artifact_bytes,
        &metrics->resident_host_bytes, &metrics->resident_device_bytes,
        &metrics->queue_depth, &metrics->queue_capacity,
        &metrics->active_sessions, &metrics->total_sessions,
        &metrics->active_requests, &metrics->completed_requests,
        &metrics->failed_requests, &metrics->cancelled_requests,
        &metrics->telemetry_dropped};
    unsigned int index;
    if (count != sizeof(values) / sizeof(values[0]) * 8u)
        return 0;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index) {
        unsigned long long value = get_u64(bytes + index * 8u);
        if (index == 0u)
            metrics->schema_version = (unsigned int)value;
        else
            *values[index] = value;
    }
    return 1;
}

/* Purpose: encode one server message with explicit field ownership.
 * Inputs: admitted message, output bytes/capacity, and count/error outputs. Effects: writes canonical payload.
 * Failure: publishes zero count for invalid schema, fields, event identity, or capacity. Boundary: no frame I/O. */
int yvex_protocol_message_encode(const yvex_client_message *message,
                                 unsigned char *output,
                                 unsigned long long capacity,
                                 unsigned long long *byte_count,
                                 yvex_error *err)
{
    wire_writer writer = {output, capacity, 0u};
    unsigned long long runtime_flags =
        (message && message->runtime.runtime_ready ? 1u : 0u) |
        (message && message->runtime.generation_ready ? 2u : 0u) |
        (message && message->runtime.public_server_ready ? 4u : 0u);
    if (byte_count) *byte_count = 0u;
    if (!message || !output || !byte_count ||
        message->schema_version != YVEX_LOCAL_PROTOCOL_VERSION ||
        message->kind > YVEX_CLIENT_MESSAGE_TURN_COMPLETE ||
        message->byte_count > sizeof(message->bytes))
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "complete bounded server message is required");
#define WRITE_U64(tag, field) writer_u64(&writer, tag, (unsigned long long)(field))
    if (!WRITE_U64(TAG_MESSAGE_KIND, message->kind) ||
        !WRITE_U64(TAG_STATUS, (uint32_t)(int32_t)message->status) ||
        !WRITE_U64(TAG_FAILURE_CLASS, message->failure_class) ||
        !WRITE_U64(TAG_REQUEST_NUMBER, message->request_number) ||
        !writer_text(&writer, TAG_SESSION_NAME, message->session_name) ||
        !writer_text(&writer, TAG_REASON, message->reason) ||
        !writer_field(&writer, TAG_BYTES, message->bytes, message->byte_count) ||
        !WRITE_U64(TAG_PROMPT_TOKENS, message->prompt_tokens) ||
        !WRITE_U64(TAG_REUSED_TOKENS, message->reused_tokens) ||
        !WRITE_U64(TAG_PREFILL_TOKENS, message->prefill_tokens) ||
        !WRITE_U64(TAG_GENERATED_TOKENS, message->generated_tokens) ||
        !WRITE_U64(TAG_FINAL_POSITION, message->final_position) ||
        !writer_double(&writer, TAG_QUEUE_SECONDS, message->queue_seconds) ||
        !writer_double(&writer, TAG_PREFILL_SECONDS, message->prefill_seconds) ||
        !writer_double(&writer, TAG_FIRST_TOKEN_SECONDS, message->first_token_seconds) ||
        !writer_double(&writer, TAG_DECODE_SECONDS, message->decode_seconds) ||
        !writer_double(&writer, TAG_PREFILL_RATE, message->prefill_rate) ||
        !writer_double(&writer, TAG_DECODE_RATE, message->decode_rate) ||
        !WRITE_U64(TAG_STOP_REASON, message->stop_reason) ||
        !WRITE_U64(TAG_SESSION_STATE, message->session_state) ||
        !writer_text(&writer, TAG_SESSION_IDENTITY,
                     message->session_identity) ||
        !writer_text(&writer, TAG_TURN_IDENTITY, message->turn_identity) ||
        !writer_text(&writer, TAG_STATE_DIGEST, message->state_digest) ||
        !writer_text(&writer, TAG_GENERATED_TOKEN_IDENTITY,
                     message->generated_token_identity) ||
        !writer_text(&writer, TAG_GENERATED_TEXT_DIGEST,
                     message->generated_text_digest) ||
        !WRITE_U64(TAG_PROVIDER_OUTPUT_KIND, message->provider_output_kind) ||
        !WRITE_U64(TAG_PROVIDER_FINISH, message->provider_finish) ||
        !WRITE_U64(TAG_COMPLETION_TOKENS, message->completion_tokens) ||
        !WRITE_U64(TAG_TOTAL_TOKENS, message->total_tokens) ||
        !writer_text(&writer, TAG_PROVIDER_REQUEST_ID,
                     message->provider_request_identity) ||
        !writer_text(&writer, TAG_EXTERNAL_CORRELATION_ID,
                     message->external_correlation_id) ||
        !writer_text(&writer, TAG_TOOL_CALL_ID, message->tool_call_id) ||
        !writer_text(&writer, TAG_TOOL_NAME, message->tool_name) ||
        !WRITE_U64(TAG_RUNTIME_STATUS, message->runtime.status) ||
        !WRITE_U64(TAG_RUNTIME_BACKEND, message->runtime.backend) ||
        !writer_text(&writer, TAG_SOCKET_PATH, message->runtime.socket_path) ||
        !writer_text(&writer, TAG_TARGET_ID, message->runtime.target_id) ||
        !writer_text(&writer, TAG_RUNTIME_MODEL_ID,
                     message->runtime.runtime_model_identity) ||
        !writer_text(&writer, TAG_RUNTIME_BINDING_ID,
                     message->runtime.runtime_binding_identity) ||
        !writer_text(&writer, TAG_ARTIFACT_ID,
                     message->runtime.artifact_identity) ||
        !writer_text(&writer, TAG_PHYSICAL_VARIANT_ID,
                     message->runtime.physical_variant_identity) ||
        !WRITE_U64(TAG_CONTEXT_CAPACITY, message->runtime.context_capacity) ||
        !WRITE_U64(TAG_SESSION_COUNT, message->runtime.session_count) ||
        !WRITE_U64(TAG_RUNTIME_REQUEST_COUNT, message->runtime.request_count) ||
        !WRITE_U64(TAG_RUNTIME_FLAGS, runtime_flags) ||
        !writer_metrics(&writer, &message->runtime.metrics) ||
        !WRITE_U64(TAG_EVENT_SEQUENCE, message->event.sequence) ||
        !WRITE_U64(TAG_EVENT_WALL_TIME, message->event.wall_time_ns) ||
        !WRITE_U64(TAG_EVENT_MONOTONIC_TIME, message->event.monotonic_time_ns) ||
        !WRITE_U64(TAG_EVENT_PROCESS_ID, message->event.process_id) ||
        !WRITE_U64(TAG_EVENT_KIND, message->event.kind) ||
        !WRITE_U64(TAG_EVENT_SEVERITY, message->event.severity) ||
        !writer_text(&writer, TAG_EVENT_SESSION_ID,
                     message->event.session_id) ||
        !writer_text(&writer, TAG_EVENT_REQUEST_ID, message->event.request_id) ||
        !writer_text(&writer, TAG_EVENT_TURN_ID, message->event.turn_id) ||
        !writer_text(&writer, TAG_EVENT_PHASE, message->event.phase) ||
        !writer_text(&writer, TAG_EVENT_PROVIDER_ADAPTER,
                     message->event.provider_adapter) ||
        !writer_text(&writer, TAG_EVENT_PROVIDER_REQUEST_ID,
                     message->event.provider_request_identity) ||
        !writer_text(&writer, TAG_EVENT_EXTERNAL_CORRELATION_ID,
                     message->event.external_correlation_id) ||
        !WRITE_U64(TAG_EVENT_VALUE_A, message->event.value_a) ||
        !WRITE_U64(TAG_EVENT_VALUE_B, message->event.value_b) ||
        !WRITE_U64(TAG_EVENT_VALUE_C, message->event.value_c) ||
        !writer_double(&writer, TAG_EVENT_SECONDS, message->event.seconds) ||
        !writer_double(&writer, TAG_EVENT_RATE, message->event.rate) ||
        !writer_text(&writer, TAG_EVENT_VARIANT_ID,
                     message->event.variant_identity) ||
        !writer_text(&writer, TAG_EVENT_RUNTIME_MODEL_ID,
                     message->event.runtime_model_identity) ||
        !writer_text(&writer, TAG_EVENT_ARTIFACT_ID,
                     message->event.artifact_identity) ||
        !writer_text(&writer, TAG_EVENT_IDENTITY,
                     message->event.event_identity))
        return protocol_refuse(err, YVEX_ERR_BOUNDS,
                               "server message does not fit admitted frame");
#undef WRITE_U64
    *byte_count = writer.count;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: decode one message-level protocol field outside runtime/event snapshots.
 * Inputs: candidate, tag, explicit field bytes/count, and kind-presence output.
 * Effects: writes exactly one recognized message field.
 * Failure: returns negative for malformed recognized fields and zero for unknown extensions.
 * Boundary: no reader position, runtime summary, event, or socket ownership. */
static int message_base_field(yvex_client_message *candidate, unsigned int tag,
                              const unsigned char *bytes,
                              unsigned long long count, int *have_kind)
{
    unsigned long long value;
    int valid = 1;
#define BASE_U64(field) (reader_u64(bytes, count, &value) ? ((field) = value, 1) : 0)
    switch (tag) {
    case TAG_MESSAGE_KIND:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_CLIENT_MESSAGE_TURN_COMPLETE;
        candidate->kind = (yvex_client_message_kind)value;
        *have_kind = valid;
        break;
    case TAG_STATUS:
        valid = reader_u64(bytes, count, &value) && value <= UINT32_MAX;
        if (valid) candidate->status = (int)(int32_t)(uint32_t)value;
        break;
    case TAG_FAILURE_CLASS:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT;
        if (valid)
            candidate->failure_class = (yvex_client_failure_class)value;
        break;
    case TAG_REQUEST_NUMBER: valid = BASE_U64(candidate->request_number); break;
    case TAG_SESSION_NAME:
        valid = reader_text(bytes, count, candidate->session_name,
                            sizeof(candidate->session_name));
        break;
    case TAG_REASON:
        valid = reader_text(bytes, count, candidate->reason,
                            sizeof(candidate->reason));
        break;
    case TAG_BYTES:
        valid = count <= sizeof(candidate->bytes);
        if (valid && count) memcpy(candidate->bytes, bytes, (size_t)count);
        candidate->byte_count = count;
        break;
    case TAG_PROMPT_TOKENS: valid = BASE_U64(candidate->prompt_tokens); break;
    case TAG_REUSED_TOKENS: valid = BASE_U64(candidate->reused_tokens); break;
    case TAG_PREFILL_TOKENS: valid = BASE_U64(candidate->prefill_tokens); break;
    case TAG_GENERATED_TOKENS: valid = BASE_U64(candidate->generated_tokens); break;
    case TAG_FINAL_POSITION: valid = BASE_U64(candidate->final_position); break;
    case TAG_QUEUE_SECONDS: valid = reader_double(bytes, count, &candidate->queue_seconds); break;
    case TAG_PREFILL_SECONDS: valid = reader_double(bytes, count, &candidate->prefill_seconds); break;
    case TAG_FIRST_TOKEN_SECONDS:
        valid = reader_double(bytes, count, &candidate->first_token_seconds);
        break;
    case TAG_DECODE_SECONDS: valid = reader_double(bytes, count, &candidate->decode_seconds); break;
    case TAG_PREFILL_RATE: valid = reader_double(bytes, count, &candidate->prefill_rate); break;
    case TAG_DECODE_RATE: valid = reader_double(bytes, count, &candidate->decode_rate); break;
    case TAG_STOP_REASON:
        valid = reader_u64(bytes, count, &value) && value <= UINT_MAX;
        if (valid) candidate->stop_reason = (unsigned int)value;
        break;
    case TAG_SESSION_STATE: valid = BASE_U64(candidate->session_state); break;
    case TAG_SESSION_IDENTITY:
        valid = reader_text(bytes, count, candidate->session_identity,
                            sizeof(candidate->session_identity));
        break;
    case TAG_TURN_IDENTITY:
        valid = reader_text(bytes, count, candidate->turn_identity,
                            sizeof(candidate->turn_identity));
        break;
    case TAG_STATE_DIGEST:
        valid = reader_text(bytes, count, candidate->state_digest,
                            sizeof(candidate->state_digest));
        break;
    case TAG_GENERATED_TOKEN_IDENTITY:
        valid = reader_text(bytes, count, candidate->generated_token_identity,
                            sizeof(candidate->generated_token_identity));
        break;
    case TAG_GENERATED_TEXT_DIGEST:
        valid = reader_text(bytes, count, candidate->generated_text_digest,
                            sizeof(candidate->generated_text_digest));
        break;
    case TAG_PROVIDER_OUTPUT_KIND:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_PROVIDER_OUTPUT_ERROR;
        candidate->provider_output_kind = (yvex_provider_output_kind)value;
        break;
    case TAG_PROVIDER_FINISH:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_PROVIDER_FINISH_FAILED;
        candidate->provider_finish = (yvex_provider_finish_class)value;
        break;
    case TAG_COMPLETION_TOKENS: valid = BASE_U64(candidate->completion_tokens); break;
    case TAG_TOTAL_TOKENS: valid = BASE_U64(candidate->total_tokens); break;
    case TAG_PROVIDER_REQUEST_ID:
        valid = reader_text(bytes, count, candidate->provider_request_identity,
                            sizeof(candidate->provider_request_identity));
        break;
    case TAG_EXTERNAL_CORRELATION_ID:
        valid = reader_text(bytes, count, candidate->external_correlation_id,
                            sizeof(candidate->external_correlation_id));
        break;
    case TAG_TOOL_CALL_ID:
        valid = reader_text(bytes, count, candidate->tool_call_id,
                            sizeof(candidate->tool_call_id));
        break;
    case TAG_TOOL_NAME:
        valid = reader_text(bytes, count, candidate->tool_name,
                            sizeof(candidate->tool_name));
        break;
    default: return 0;
    }
#undef BASE_U64
    return valid ? 1 : -1;
}

/* Purpose: decode one authoritative runtime-summary field from a server message.
 * Inputs: candidate, tag, and explicit field bytes/count.
 * Effects: writes one recognized runtime summary field.
 * Failure: returns negative for malformed recognized fields and zero for unknown fields.
 * Boundary: does not validate event identity or message kind. */
static int message_runtime_field(yvex_client_message *candidate,
                                 unsigned int tag,
                                 const unsigned char *bytes,
                                 unsigned long long count)
{
    unsigned long long value;
    int valid = 1;
#define RUNTIME_U64(field) (reader_u64(bytes, count, &value) ? ((field) = value, 1) : 0)
    switch (tag) {
    case TAG_RUNTIME_STATUS: valid = RUNTIME_U64(candidate->runtime.status); break;
    case TAG_RUNTIME_BACKEND: valid = RUNTIME_U64(candidate->runtime.backend); break;
    case TAG_SOCKET_PATH:
        valid = reader_text(bytes, count, candidate->runtime.socket_path,
                            sizeof(candidate->runtime.socket_path));
        break;
    case TAG_TARGET_ID:
        valid = reader_text(bytes, count, candidate->runtime.target_id,
                            sizeof(candidate->runtime.target_id));
        break;
    case TAG_RUNTIME_MODEL_ID:
        valid = reader_text(bytes, count, candidate->runtime.runtime_model_identity,
                            sizeof(candidate->runtime.runtime_model_identity));
        break;
    case TAG_RUNTIME_BINDING_ID:
        valid = reader_text(bytes, count, candidate->runtime.runtime_binding_identity,
                            sizeof(candidate->runtime.runtime_binding_identity));
        break;
    case TAG_ARTIFACT_ID:
        valid = reader_text(bytes, count, candidate->runtime.artifact_identity,
                            sizeof(candidate->runtime.artifact_identity));
        break;
    case TAG_PHYSICAL_VARIANT_ID:
        valid = reader_text(bytes, count, candidate->runtime.physical_variant_identity,
                            sizeof(candidate->runtime.physical_variant_identity));
        break;
    case TAG_CONTEXT_CAPACITY: valid = RUNTIME_U64(candidate->runtime.context_capacity); break;
    case TAG_SESSION_COUNT: valid = RUNTIME_U64(candidate->runtime.session_count); break;
    case TAG_RUNTIME_REQUEST_COUNT: valid = RUNTIME_U64(candidate->runtime.request_count); break;
    case TAG_RUNTIME_FLAGS:
        valid = reader_u64(bytes, count, &value) && !(value & ~7u);
        candidate->runtime.runtime_ready = (value & 1u) != 0u;
        candidate->runtime.generation_ready = (value & 2u) != 0u;
        candidate->runtime.public_server_ready = (value & 4u) != 0u;
        break;
    case TAG_METRICS: valid = reader_metrics(bytes, count, &candidate->runtime.metrics); break;
    default: return 0;
    }
#undef RUNTIME_U64
    return valid ? 1 : -1;
}

/* Purpose: decode one authoritative typed-event field from a server message.
 * Inputs: candidate, tag, and explicit field bytes/count.
 * Effects: writes one recognized event field for later identity validation.
 * Failure: returns negative for malformed recognized fields and zero for unknown fields.
 * Boundary: event identity is validated only after the complete message is decoded. */
static int message_event_field(yvex_client_message *candidate,
                               unsigned int tag,
                               const unsigned char *bytes,
                               unsigned long long count)
{
    unsigned long long value;
    int valid = 1;
#define EVENT_U64(field) (reader_u64(bytes, count, &value) ? ((field) = value, 1) : 0)
    switch (tag) {
    case TAG_EVENT_SEQUENCE: valid = EVENT_U64(candidate->event.sequence); break;
    case TAG_EVENT_WALL_TIME: valid = EVENT_U64(candidate->event.wall_time_ns); break;
    case TAG_EVENT_MONOTONIC_TIME: valid = EVENT_U64(candidate->event.monotonic_time_ns); break;
    case TAG_EVENT_PROCESS_ID: valid = EVENT_U64(candidate->event.process_id); break;
    case TAG_EVENT_KIND: valid = EVENT_U64(candidate->event.kind); break;
    case TAG_EVENT_SEVERITY: valid = EVENT_U64(candidate->event.severity); break;
    case TAG_EVENT_SESSION_ID:
        valid = reader_text(bytes, count, candidate->event.session_id,
                            sizeof(candidate->event.session_id));
        break;
    case TAG_EVENT_REQUEST_ID:
        valid = reader_text(bytes, count, candidate->event.request_id,
                            sizeof(candidate->event.request_id));
        break;
    case TAG_EVENT_TURN_ID:
        valid = reader_text(bytes, count, candidate->event.turn_id,
                            sizeof(candidate->event.turn_id));
        break;
    case TAG_EVENT_PHASE:
        valid = reader_text(bytes, count, candidate->event.phase,
                            sizeof(candidate->event.phase));
        break;
    case TAG_EVENT_PROVIDER_ADAPTER:
        valid = reader_text(bytes, count, candidate->event.provider_adapter,
                            sizeof(candidate->event.provider_adapter));
        break;
    case TAG_EVENT_PROVIDER_REQUEST_ID:
        valid = reader_text(bytes, count,
                            candidate->event.provider_request_identity,
                            sizeof(candidate->event.provider_request_identity));
        break;
    case TAG_EVENT_EXTERNAL_CORRELATION_ID:
        valid = reader_text(bytes, count,
                            candidate->event.external_correlation_id,
                            sizeof(candidate->event.external_correlation_id));
        break;
    case TAG_EVENT_VALUE_A: valid = EVENT_U64(candidate->event.value_a); break;
    case TAG_EVENT_VALUE_B: valid = EVENT_U64(candidate->event.value_b); break;
    case TAG_EVENT_VALUE_C: valid = EVENT_U64(candidate->event.value_c); break;
    case TAG_EVENT_SECONDS: valid = reader_double(bytes, count, &candidate->event.seconds); break;
    case TAG_EVENT_RATE: valid = reader_double(bytes, count, &candidate->event.rate); break;
    case TAG_EVENT_VARIANT_ID:
        valid = reader_text(bytes, count, candidate->event.variant_identity,
                            sizeof(candidate->event.variant_identity));
        break;
    case TAG_EVENT_RUNTIME_MODEL_ID:
        valid = reader_text(bytes, count, candidate->event.runtime_model_identity,
                            sizeof(candidate->event.runtime_model_identity));
        break;
    case TAG_EVENT_ARTIFACT_ID:
        valid = reader_text(bytes, count, candidate->event.artifact_identity,
                            sizeof(candidate->event.artifact_identity));
        break;
    case TAG_EVENT_IDENTITY:
        valid = reader_text(bytes, count, candidate->event.event_identity,
                            sizeof(candidate->event.event_identity));
        break;
    default: return 0;
    }
#undef EVENT_U64
    return valid ? 1 : -1;
}

/* Purpose: validate and publish one fully decoded server-message candidate.
 * Inputs: candidate, parse state, destination, and error output. Effects: copies candidate on success.
 * Failure: refuses malformed fields or invalid event identity. Boundary: no frame or socket ownership. */
static int message_publish(const yvex_client_message *candidate, int next,
                           int valid, int have_kind,
                           yvex_client_message *message, yvex_error *err)
{
    if (next < 0 || !valid || !have_kind)
        return protocol_refuse(
            err, YVEX_ERR_FORMAT,
            "message frame contains malformed or duplicate fields");
    if (candidate->kind == YVEX_CLIENT_MESSAGE_EVENT &&
        yvex_server_event_validate(&candidate->event, err) != YVEX_OK)
        return yvex_error_code(err);
    *message = *candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: decode one complete server message transactionally.
 * Inputs: payload bytes/extent, message output, and error output. Effects: replaces output on success.
 * Failure: clears output and refuses malformed, duplicate, or invalid authoritative fields. Boundary: no socket I/O. */
int yvex_protocol_message_decode(const unsigned char *input,
                                 unsigned long long byte_count,
                                 yvex_client_message *message,
                                 yvex_error *err)
{
    wire_reader reader = {input, byte_count, 0u, {0u, 0u}};
    yvex_client_message candidate;
    const unsigned char *bytes;
    unsigned long long count;
    unsigned int tag;
    int next, valid = 1, have_kind = 0;
    if (!input || !message || byte_count > YVEX_SERVER_FRAME_MAX_BYTES)
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "bounded message bytes and output are required");
    memset(&candidate, 0, sizeof(candidate));
    candidate.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    candidate.runtime.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    candidate.event.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    while ((next = reader_next(&reader, &tag, &bytes, &count)) > 0 && valid) {
        int field = message_base_field(&candidate, tag, bytes, count,
                                       &have_kind);
        if (!field)
            field = message_runtime_field(&candidate, tag, bytes, count);
        if (!field)
            field = message_event_field(&candidate, tag, bytes, count);
        if (field < 0) valid = 0;
    }
    return message_publish(&candidate, next, valid, have_kind, message, err);
}

/* Purpose: read or write a complete byte span while preserving the first I/O error.
 * Inputs: socket descriptor, mutable byte span/count, direction, and error output.
 * Effects: advances the exact span until complete.
 * Failure: distinguishes configured timeout from peer closure and other I/O failure.
 * Boundary: raw local transport only; frame syntax remains with the caller. */
static int transfer_all(int fd, void *buffer, size_t count, int writing,
                        yvex_error *err)
{
    unsigned char *bytes = buffer;
    size_t offset = 0u;
    while (offset < count) {
        ssize_t moved = writing ? send(fd, bytes + offset, count - offset, MSG_NOSIGNAL)
                                : recv(fd, bytes + offset, count - offset, 0);
        if (moved < 0 && errno == EINTR)
            continue;
        if (moved < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            return protocol_refuse(err, YVEX_ERR_TIMEOUT,
                                   "local protocol operation timed out");
        if (moved <= 0)
            return protocol_refuse(err, YVEX_ERR_IO,
                                   writing ? "local socket write failed"
                                           : "local socket closed during frame read");
        offset += (size_t)moved;
    }
    return YVEX_OK;
}

/* Purpose: apply or clear one bounded local-protocol socket I/O timeout.
 * Inputs: connected client, milliseconds where zero clears the timeout, and error output.
 * Effects: updates both receive and send timeout policy on the owned descriptor.
 * Failure: preserves descriptor ownership and reports conversion or socket refusal.
 * Boundary: controls transport waiting only; generation cancellation remains caller-owned. */
int yvex_client_timeout_set(yvex_client *client,
                            unsigned long long milliseconds,
                            yvex_error *err)
{
    struct timeval timeout;
    if (!client || client->fd < 0 || milliseconds > 86400000u)
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "connected client and bounded timeout are required");
    timeout.tv_sec = (time_t)(milliseconds / 1000u);
    timeout.tv_usec = (suseconds_t)((milliseconds % 1000u) * 1000u);
    if (setsockopt(client->fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                   sizeof(timeout)) != 0 ||
        setsockopt(client->fd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                   sizeof(timeout)) != 0)
        return protocol_refuse(err, YVEX_ERR_IO,
                               "local protocol timeout configuration failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: send one canonical framed payload. */
static int frame_send(int fd, unsigned int kind, const unsigned char *payload,
                      unsigned long long count, yvex_error *err)
{
    unsigned char header[FRAME_HEADER_BYTES] = {'Y', 'V', 'X', 'P'};
    if (count > YVEX_SERVER_FRAME_MAX_BYTES)
        return protocol_refuse(err, YVEX_ERR_BOUNDS,
                               "local protocol frame exceeds maximum size");
    put_u16(header + 4u, YVEX_LOCAL_PROTOCOL_VERSION);
    put_u16(header + 6u, (uint16_t)kind);
    put_u32(header + 8u, (uint32_t)count);
    if (transfer_all(fd, header, sizeof(header), 1, err) != YVEX_OK)
        return yvex_error_code(err);
    return count ? transfer_all(fd, (void *)payload, (size_t)count, 1, err)
                 : YVEX_OK;
}

/* Purpose: receive one expected canonical frame into owned bounded bytes.
 * Inputs: connected descriptor, expected kind, payload/count outputs, and error output.
 * Effects: allocates payload. Failure: frees partial ownership and refuses invalid frames or I/O.
 * Boundary: payload decoding follows. */
static int frame_receive(int fd, unsigned int expected_kind,
                         unsigned char **payload, unsigned long long *count,
                         yvex_error *err)
{
    unsigned char header[FRAME_HEADER_BYTES], *bytes = NULL;
    uint32_t length;
    int rc;
    *payload = NULL;
    *count = 0u;
    rc = transfer_all(fd, header, sizeof(header), 0, err);
    if (rc != YVEX_OK) return rc;
    length = get_u32(header + 8u);
    if (memcmp(header, "YVXP", 4u) != 0 ||
        get_u16(header + 4u) != YVEX_LOCAL_PROTOCOL_VERSION ||
        get_u16(header + 6u) != expected_kind ||
        length > YVEX_SERVER_FRAME_MAX_BYTES)
        return protocol_refuse(err, YVEX_ERR_FORMAT,
                               "local protocol frame header is invalid");
    if (length) {
        bytes = malloc(length);
        if (!bytes)
            return protocol_refuse(err, YVEX_ERR_NOMEM,
                                   "local protocol frame allocation failed");
        rc = transfer_all(fd, bytes, length, 0, err);
        if (rc != YVEX_OK) {
            free(bytes);
            return rc;
        }
    }
    *payload = bytes;
    *count = length;
    return YVEX_OK;
}

/* Purpose: resolve the canonical local Unix-socket path without creating filesystem state.
 * Inputs: caller path output and error output. Effects: reads XDG_RUNTIME_DIR or selects private fallback.
 * Failure: refuses oversized or unsafe environment paths. Boundary: directory creation belongs to host. */
int yvex_server_socket_path(char output[YVEX_SERVER_SOCKET_PATH_CAP],
                            yvex_error *err)
{
    const char *runtime = getenv("XDG_RUNTIME_DIR");
    int length;
    if (!output)
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "socket path output is required");
    if (runtime && runtime[0] == '/')
        length = snprintf(output, YVEX_SERVER_SOCKET_PATH_CAP,
                          "%s/yvex/yvexd.sock", runtime);
    else
        length = snprintf(output, YVEX_SERVER_SOCKET_PATH_CAP,
                          "/tmp/yvex-%lu/yvexd.sock", (unsigned long)getuid());
    if (length < 0 || length >= (int)YVEX_SERVER_SOCKET_PATH_CAP)
        return protocol_refuse(err, YVEX_ERR_BOUNDS,
                               "canonical socket path exceeds its bound");
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: connect one thin client only to an owner-validated private Unix socket.
 * Inputs: owner output, optional path, and error output. Effects: opens and authenticates one descriptor.
 * Failure: closes partial ownership and refuses mode, owner, symlink, or connect mismatch. Boundary: local UID only. */
int yvex_client_connect(yvex_client **out, const char *socket_path,
                        yvex_error *err)
{
    yvex_client_request handshake;
    yvex_client_message response;
    yvex_client *client;
    struct sockaddr_un address;
    struct stat info;
    char canonical[YVEX_SERVER_SOCKET_PATH_CAP];
    const char *path = socket_path;
    int fd;
    if (out) *out = NULL;
    if (!out)
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "client output is required");
    if (!path) {
        if (yvex_server_socket_path(canonical, err) != YVEX_OK)
            return yvex_error_code(err);
        path = canonical;
    }
    if (strlen(path) >= sizeof(address.sun_path) || lstat(path, &info) != 0 ||
        !S_ISSOCK(info.st_mode) || info.st_uid != getuid() ||
        (info.st_mode & 0077u) != 0u)
        return protocol_refuse(err, YVEX_ERR_IO,
                               "local runtime socket is absent or not private to this user");
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return protocol_refuse(err, YVEX_ERR_IO,
                               "local client socket creation failed");
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path) + 1u);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        (void)close(fd);
        return protocol_refuse(err, YVEX_ERR_IO,
                               "cannot connect to the local YVEX runtime");
    }
    client = calloc(1u, sizeof(*client));
    if (!client) {
        (void)close(fd);
        return protocol_refuse(err, YVEX_ERR_NOMEM,
                               "local client allocation failed");
    }
    client->fd = fd;
    if (yvex_client_timeout_set(client, 30000u, err) != YVEX_OK) {
        (void)close(client->fd);
        free(client);
        return yvex_error_code(err);
    }
    memset(&handshake, 0, sizeof(handshake));
    handshake.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    handshake.operation = YVEX_CLIENT_OP_HANDSHAKE;
    handshake.temperature = 1.0;
    handshake.top_p = 1.0;
    handshake.typical_p = 1.0;
    if (yvex_client_send(client, &handshake, err) != YVEX_OK ||
        yvex_client_receive(client, &response, err) != YVEX_OK ||
        response.kind != YVEX_CLIENT_MESSAGE_ACK ||
        response.status != YVEX_OK || strcmp(response.reason, "protocol-v2") != 0) {
        (void)close(client->fd);
        memset(client, 0, sizeof(*client));
        free(client);
        if (yvex_error_code(err) == YVEX_OK)
            yvex_error_set(err, YVEX_ERR_FORMAT, "server.protocol.handshake",
                           "daemon did not admit local protocol version 2");
        return yvex_error_code(err);
    }
    if (yvex_client_timeout_set(client, 0u, err) != YVEX_OK) {
        (void)close(client->fd);
        free(client);
        return yvex_error_code(err);
    }
    *out = client;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: encode and send one typed request without retaining caller spans.
 * Inputs: connected client, immutable request, and error output. Effects: allocates a bounded payload and writes frame.
 * Failure: frees payload and returns encode or I/O refusal. Boundary: does not wait for a response. */
int yvex_client_send(yvex_client *client, const yvex_client_request *request,
                     yvex_error *err)
{
    unsigned char *payload;
    unsigned long long capacity, count;
    int rc;
    if (!client || client->fd < 0 || !request)
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "connected client and request are required");
    capacity = request->provider_request ? YVEX_SERVER_FRAME_MAX_BYTES
                                         : request->prompt_bytes + 512u;
    if (capacity > YVEX_SERVER_FRAME_MAX_BYTES)
        return protocol_refuse(err, YVEX_ERR_BOUNDS,
                               "client request exceeds frame capacity");
    payload = malloc((size_t)capacity);
    if (!payload)
        return protocol_refuse(err, YVEX_ERR_NOMEM,
                               "client request frame allocation failed");
    rc = yvex_protocol_request_encode(request, payload, capacity, &count, err);
    if (rc == YVEX_OK)
        rc = frame_send(client->fd, FRAME_KIND_REQUEST, payload, count, err);
    free(payload);
    return rc;
}

/* Purpose: receive and decode one complete server message.
 * Inputs: connected client, message output, and error output. Effects: allocates then frees one frame payload.
 * Failure: returns frame or message refusal without partial message authority. Boundary: one frame per call. */
int yvex_client_receive(yvex_client *client, yvex_client_message *message,
                        yvex_error *err)
{
    unsigned char *payload;
    unsigned long long count;
    int rc;
    if (!client || client->fd < 0 || !message)
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "connected client and message output are required");
    rc = frame_receive(client->fd, FRAME_KIND_MESSAGE, &payload, &count, err);
    if (rc == YVEX_OK)
        rc = yvex_protocol_message_decode(payload, count, message, err);
    free(payload);
    return rc;
}

/* Purpose: close one thin-client descriptor and clear unique ownership.
 * Inputs: unique client-owner pointer. Effects: closes descriptor, frees allocation, and stores NULL.
 * Failure: close errors are intentionally secondary in this destructor. Boundary: no daemon/session close. */
void yvex_client_close(yvex_client **client)
{
    if (!client || !*client)
        return;
    if ((*client)->fd >= 0)
        (void)close((*client)->fd);
    memset(*client, 0, sizeof(**client));
    free(*client);
    *client = NULL;
}

/* Purpose: receive and decode one server-side request frame.
 * Inputs: connected descriptor, request/prompt outputs, and error output. Effects: may allocate prompt ownership.
 * Failure: frees frame storage and preserves transactional decode refusal. Boundary: host consumes private ABI. */
int yvex_server_protocol_receive(int fd, yvex_client_request *request,
                                 unsigned char **owned_prompt,
                                 yvex_provider_request **owned_provider,
                                 yvex_error *err)
{
    unsigned char *payload;
    unsigned long long count;
    int rc = frame_receive(fd, FRAME_KIND_REQUEST, &payload, &count, err);
    if (rc == YVEX_OK)
        rc = yvex_protocol_request_decode(payload, count, request,
                                          owned_prompt, owned_provider, err);
    free(payload);
    return rc;
}

/* Purpose: encode and send one server-side response frame.
 * Inputs: connected descriptor, immutable message, and error output. Effects: writes one bounded frame.
 * Failure: returns canonical encode or I/O refusal. Boundary: host consumes private ABI. */
int yvex_server_protocol_send(int fd, const yvex_client_message *message,
                              yvex_error *err)
{
    unsigned char payload[16384];
    unsigned long long count;
    int rc = yvex_protocol_message_encode(message, payload, sizeof(payload),
                                          &count, err);
    if (rc == YVEX_OK)
        rc = frame_send(fd, FRAME_KIND_MESSAGE, payload, count, err);
    return rc;
}
