/*
 * Transport typed client requests and server messages without engine linkage.
 *
 * Wire integers are big-endian, duplicate known fields refuse, and frames are bounded. Public
 * local client and reusable canonical message codec.
 */
#define _POSIX_C_SOURCE 200809L
#include "src/server/private.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
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
    TAG_REASONING_POLICY,
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
    TAG_PREFILL_CHUNK_TOKENS,
    TAG_RUNTIME_MAXIMUM_NEW_TOKENS,
    TAG_RUNTIME_MAXIMUM_OUTPUT_BYTES,
    TAG_RUNTIME_MAXIMUM_SESSIONS,
    TAG_METRICS = 80,
    TAG_OPENAI_PORT,
    TAG_RUNTIME_QUEUE_CAPACITY,
    TAG_RUNTIME_OPENAI_TIMEOUT,
    TAG_RUNTIME_TRACE_LEVEL,
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
    TAG_FAILURE_CLASS,
    TAG_TURN_COUNT,
    TAG_CONTEXT_USED,
    TAG_KV_USED_BYTES,
    TAG_GENERATION_PHASE,
    TAG_CANCELLATION_CLASS,
    TAG_STREAM_CHANNEL,
    TAG_PUBLICATION_SECONDS,
    TAG_MESSAGE_AVAILABILITY_FLAGS,
    TAG_CONSOLE_FLAGS,
    TAG_CONSOLE_BACKEND,
    TAG_CONSOLE_SESSION_STATE,
    TAG_CONSOLE_POSITION,
    TAG_CONSOLE_TURN_COUNT,
    TAG_CONSOLE_CONTEXT_CAPACITY,
    TAG_CONSOLE_CONTEXT_USED,
    TAG_CONSOLE_KV_USED_BYTES,
    TAG_CONSOLE_PHASE,
    TAG_CONSOLE_CANCELLATION,
    TAG_CONSOLE_LIVE_MODEL_ID,
    TAG_CONSOLE_VARIANT_ID,
    TAG_CONSOLE_SESSION_NAME,
    TAG_CONSOLE_SELECTED_MODEL_ID,
    TAG_RUNTIME_GENERATION_MODE,
    TAG_GENERATION_MODE,
    TAG_DRAFT_CYCLE_COUNT,
    TAG_DRAFT_FORWARD_COUNT,
    TAG_PROPOSED_TOKENS,
    TAG_SELECTED_VERIFICATION_TOKENS,
    TAG_TARGET_VERIFICATION_COUNT,
    TAG_ACCEPTED_DRAFT_TOKENS,
    TAG_REJECTED_DRAFT_TOKENS,
    TAG_TARGET_CORRECTION_OR_BONUS_TOKENS,
    TAG_MAXIMUM_ACCEPTED_PREFIX,
    TAG_DRAFT_SECONDS,
    TAG_VERIFICATION_SECONDS,
    TAG_SPECULATIVE_COMMIT_SECONDS,
    TAG_MEAN_ACCEPTED_PREFIX,
    TAG_EFFECTIVE_COMMITTED_RATE,
    TAG_SPECULATION_POLICY_ID,
    TAG_EVENT_GENERATION_MODE,
    TAG_EVENT_SPECULATIVE_CYCLE,
    TAG_EVENT_PROPOSED_TOKENS,
    TAG_EVENT_SELECTED_VERIFICATION_TOKENS,
    TAG_EVENT_ACCEPTED_TOKENS,
    TAG_EVENT_REJECTED_TOKENS,
    TAG_EVENT_VERIFICATION_COUNT,
    TAG_EVENT_SPECULATION_POLICY_ID,
    TAG_DISCARDED_DRAFT_TOKENS,
    TAG_EVENT_DISCARDED_TOKENS,
    TAG_CONFIDENCE_LOGIT_COUNT,
    TAG_CONFIDENCE_LOGIT_MINIMUM,
    TAG_CONFIDENCE_LOGIT_MAXIMUM,
    TAG_CONFIDENCE_LOGIT_MEAN,
    TAG_EVENT_CONFIDENCE_LOGIT_COUNT,
    TAG_EVENT_CONFIDENCE_LOGIT_MINIMUM,
    TAG_EVENT_CONFIDENCE_LOGIT_MAXIMUM,
    TAG_EVENT_CONFIDENCE_LOGIT_MEAN,
    TAG_PARTIAL_FLAGS,
    TAG_PARTIAL_FAILURE_STATUS,
    TAG_PARTIAL_FAILURE_CLASS,
    TAG_PARTIAL_STOP_REASON,
    TAG_PARTIAL_INITIAL_POSITION,
    TAG_PARTIAL_FINAL_POSITION,
    TAG_PARTIAL_COMMITTED_TOKENS,
    TAG_PARTIAL_PUBLISHED_BYTES,
    TAG_PARTIAL_TARGET_GENERATION,
    TAG_PARTIAL_DRAFT_GENERATION,
    TAG_PARTIAL_RNG_GENERATION,
    TAG_PARTIAL_LEDGER_GENERATION,
    TAG_PARTIAL_DETOKENIZER_GENERATION,
    TAG_PARTIAL_MESSAGE_GENERATION,
    TAG_PARTIAL_TRANSCRIPT_GENERATION,
    TAG_PARTIAL_TARGET_IDENTITY,
    TAG_PARTIAL_RNG_IDENTITY,
    TAG_PARTIAL_LEDGER_IDENTITY,
    TAG_PARTIAL_TEXT_IDENTITY,
    TAG_CONSOLE_REASONING_POLICY
};
typedef struct {
    unsigned char *data;
    unsigned long long capacity, count;
} wire_writer;
typedef struct {
    const unsigned char *data;
    unsigned long long count, offset;
    uint64_t seen[4];
} wire_reader;
struct yvex_client {
    int fd;
};
_Static_assert(sizeof(double) == 8u, "local protocol requires binary64 double");
_Static_assert(TAG_CONSOLE_REASONING_POLICY < 256u,
               "known protocol tags must fit the duplicate-field set");

static int protocol_refuse(yvex_error *err, yvex_status status,
                           const char *reason)
{
    yvex_error_set(err, status, "server.protocol", reason);
    return status;
}

static void put_u16(unsigned char *out, uint16_t value)
{
    out[0] = (unsigned char)(value >> 8u);
    out[1] = (unsigned char)value;
}

static void put_u32(unsigned char *out, uint32_t value)
{
    out[0] = (unsigned char)(value >> 24u);
    out[1] = (unsigned char)(value >> 16u);
    out[2] = (unsigned char)(value >> 8u);
    out[3] = (unsigned char)value;
}

static void put_u64(unsigned char *out, uint64_t value)
{
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        out[index] = (unsigned char)(value >> (56u - 8u * index));
}

static uint16_t get_u16(const unsigned char *input)
{
    return (uint16_t)(((uint16_t)input[0] << 8u) | input[1]);
}

static uint32_t get_u32(const unsigned char *input)
{
    return ((uint32_t)input[0] << 24u) | ((uint32_t)input[1] << 16u) |
           ((uint32_t)input[2] << 8u) | input[3];
}

static uint64_t get_u64(const unsigned char *input)
{
    uint64_t value = 0u;
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        value = (value << 8u) | input[index];
    return value;
}

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

static int writer_u64(wire_writer *writer, unsigned int tag,
                      unsigned long long value)
{
    unsigned char bytes[TLV_U64_BYTES];
    put_u64(bytes, value);
    return writer_field(writer, tag, bytes, sizeof(bytes));
}

static int writer_double(wire_writer *writer, unsigned int tag, double value)
{
    uint64_t bits;
    if (!isfinite(value)) return 0;
    memcpy(&bits, &value, sizeof(bits));
    return writer_u64(writer, tag, bits);
}

static int writer_text(wire_writer *writer, unsigned int tag,
                       const char *text)
{
    return !text || !text[0] ||
           writer_field(writer, tag, text,
                        (unsigned long long)strlen(text));
}

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
    if (get_u16(reader->data + reader->offset + 2u) != 0u)
        return -1;
    length = get_u32(reader->data + reader->offset + 4u);
    reader->offset += TLV_HEADER_BYTES;
    if (length > reader->count - reader->offset)
        return -1;
    if (*tag < 192u) {
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

static int reader_u64(const unsigned char *bytes, unsigned long long count,
                      unsigned long long *value)
{
    if (count != TLV_U64_BYTES)
        return 0;
    *value = get_u64(bytes);
    return 1;
}

static int reader_double(const unsigned char *bytes, unsigned long long count,
                         double *value)
{
    unsigned long long canonical;
    uint64_t bits;
    if (!reader_u64(bytes, count, &canonical))
        return 0;
    bits = canonical;
    memcpy(value, &bits, sizeof(bits));
    return isfinite(*value);
}

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
        (int)request->operation < (int)YVEX_CLIENT_OP_HANDSHAKE ||
        request->operation > YVEX_CLIENT_OP_CONSOLE_STATUS ||
        (int)request->trace_level < (int)YVEX_SERVER_TRACE_SUMMARY ||
        request->trace_level > YVEX_SERVER_TRACE_FULL ||
        request->reasoning_policy > YVEX_REASONING_MAXIMUM ||
        (request->stochastic != 0 && request->stochastic != 1) ||
        (request->seed_present != 0 && request->seed_present != 1) ||
        (request->trace_content != 0 && request->trace_content != 1) ||
        !isfinite(request->temperature) || !isfinite(request->top_p) ||
        !isfinite(request->min_p) || !isfinite(request->typical_p) ||
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
        !writer_u64(&writer, TAG_REASONING_POLICY,
                    request->reasoning_policy) ||
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
    unsigned long long count, value = 0ull;
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
                    value <= YVEX_CLIENT_OP_CONSOLE_STATUS;
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
        case TAG_REASONING_POLICY:
            valid = reader_u64(bytes, count, &value) &&
                    value <= YVEX_REASONING_MAXIMUM;
            if (valid)
                candidate.reasoning_policy = (yvex_reasoning_policy)value;
            break;
        default: valid = 0; break;
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

static int writer_metrics(wire_writer *writer, const yvex_server_metrics *metrics)
{
    unsigned char bytes[32u * 8u];
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
        metrics->cancelled_requests, metrics->active_http_requests,
        metrics->completed_http_requests, metrics->failed_http_requests,
        metrics->cancelled_http_requests, metrics->telemetry_dropped};
    unsigned int index;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index)
        put_u64(bytes + index * 8u, values[index]);
    return writer_field(writer, TAG_METRICS, bytes,
                        sizeof(values) / sizeof(values[0]) * 8u);
}

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
        &metrics->active_http_requests, &metrics->completed_http_requests,
        &metrics->failed_http_requests, &metrics->cancelled_http_requests,
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

static int optional_identity_valid(const char value[YVEX_SHA256_HEX_CAP])
{
    return !value[0] || yvex_sha256_hex_valid(value);
}

static int partial_turn_fields_valid(const yvex_client_partial_turn *partial)
{
#define PARTIAL_BOOL(value) ((value) == 0 || (value) == 1)
    if (!partial || !PARTIAL_BOOL(partial->available) ||
        !PARTIAL_BOOL(partial->committed_progress) ||
        !PARTIAL_BOOL(partial->reset_required) ||
        !PARTIAL_BOOL(partial->draft_state_generation_available) ||
        !PARTIAL_BOOL(partial->detokenizer_generation_available) ||
        partial->failure_class > YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT ||
        partial->stop_reason > YVEX_GENERATION_STOP_OUTPUT_FAILURE)
        return 0;
    if (!partial->available)
        return !partial->schema_version && !partial->committed_progress &&
               !partial->reset_required && !partial->failure_status &&
               partial->failure_class == YVEX_CLIENT_FAILURE_NONE &&
               !partial->stop_reason && !partial->initial_position &&
               !partial->final_committed_position &&
               !partial->committed_token_count &&
               !partial->published_text_bytes &&
               !partial->target_state_generation &&
               !partial->draft_state_generation && !partial->rng_generation &&
               !partial->token_ledger_generation &&
               !partial->detokenizer_generation &&
               !partial->message_history_generation &&
               !partial->transcript_generation &&
               !partial->target_state_identity[0] &&
               !partial->rng_state_identity[0] &&
               !partial->token_ledger_identity[0] &&
               !partial->published_text_identity[0];
    return partial->schema_version == YVEX_CLIENT_PARTIAL_TURN_SCHEMA_V1 &&
           partial->reset_required && partial->failure_status != YVEX_OK &&
           partial->failure_class != YVEX_CLIENT_FAILURE_NONE &&
           partial->final_committed_position >= partial->initial_position &&
           (partial->draft_state_generation_available ||
            !partial->draft_state_generation) &&
           (partial->detokenizer_generation_available ||
            !partial->detokenizer_generation) &&
           optional_identity_valid(partial->target_state_identity) &&
           optional_identity_valid(partial->rng_state_identity) &&
           optional_identity_valid(partial->token_ledger_identity) &&
           optional_identity_valid(partial->published_text_identity);
#undef PARTIAL_BOOL
}

static int message_fields_valid(const yvex_client_message *message)
{
#define ENUM_VALID(value, first, last) \
    ((int)(value) >= (int)(first) && (value) <= (last))
#define BOOL_VALID(value) ((value) == 0 || (value) == 1)
    return ENUM_VALID(message->kind, YVEX_CLIENT_MESSAGE_ACK,
                      YVEX_CLIENT_MESSAGE_CONSOLE_STATUS) &&
           ENUM_VALID(message->failure_class, YVEX_CLIENT_FAILURE_NONE,
                      YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT) &&
           ENUM_VALID(message->generation_phase, YVEX_CLIENT_PHASE_UNAVAILABLE,
                      YVEX_CLIENT_PHASE_FAILED) &&
           ENUM_VALID(message->cancellation_class, YVEX_CLIENT_CANCELLATION_NONE,
                      YVEX_CLIENT_CANCELLATION_FAILED) &&
           ENUM_VALID(message->stream_channel, YVEX_CLIENT_STREAM_UNSPECIFIED,
                      YVEX_CLIENT_STREAM_CONTROL_EVENT) &&
           ENUM_VALID(message->generation_mode, YVEX_SERVER_GENERATION_TARGET_ONLY,
                      YVEX_SERVER_GENERATION_DSPARK) &&
           ENUM_VALID(message->session_state, YVEX_SERVER_SESSION_CREATED,
                      YVEX_SERVER_SESSION_FAILED) &&
           ENUM_VALID(message->provider_output_kind,
                      YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT,
                      YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING) &&
           ENUM_VALID(message->provider_finish, YVEX_PROVIDER_FINISH_STOP,
                      YVEX_PROVIDER_FINISH_FAILED) &&
           message->stop_reason <= YVEX_GENERATION_STOP_OUTPUT_FAILURE &&
           ENUM_VALID(message->runtime.status, YVEX_SERVER_STATUS_CONFIGURED,
                      YVEX_SERVER_STATUS_FAILED) &&
           ENUM_VALID(message->runtime.backend, YVEX_BACKEND_KIND_CPU,
                      YVEX_BACKEND_KIND_ROCM) &&
           ENUM_VALID(message->runtime.trace_level, YVEX_SERVER_TRACE_SUMMARY,
                      YVEX_SERVER_TRACE_FULL) &&
           ENUM_VALID(message->console.backend, YVEX_BACKEND_KIND_CPU,
                      YVEX_BACKEND_KIND_ROCM) &&
           ENUM_VALID(message->console.session_state, YVEX_SERVER_SESSION_CREATED,
                      YVEX_SERVER_SESSION_FAILED) &&
           ENUM_VALID(message->console.generation_phase,
                      YVEX_CLIENT_PHASE_UNAVAILABLE, YVEX_CLIENT_PHASE_FAILED) &&
           ENUM_VALID(message->console.cancellation_class,
                      YVEX_CLIENT_CANCELLATION_NONE,
                      YVEX_CLIENT_CANCELLATION_FAILED) &&
           ENUM_VALID(message->event.kind, YVEX_SERVER_EVENT_PROCESS_START,
                      YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE) &&
           ENUM_VALID(message->event.severity, YVEX_SERVER_SEVERITY_DEBUG,
                      YVEX_SERVER_SEVERITY_FATAL) &&
           BOOL_VALID(message->kv_used_available) &&
           BOOL_VALID(message->publication_timing_available) &&
           BOOL_VALID(message->runtime.runtime_ready) &&
           BOOL_VALID(message->runtime.generation_ready) &&
           BOOL_VALID(message->runtime.public_server_ready) &&
           BOOL_VALID(message->runtime.openai_listener_enabled) &&
           BOOL_VALID(message->runtime.openai_listener_ready) &&
           BOOL_VALID(message->runtime.explicit_reasoning_channel_supported) &&
           BOOL_VALID(message->console.runtime_ready) &&
           BOOL_VALID(message->console.session_available) &&
           BOOL_VALID(message->console.attached) &&
           BOOL_VALID(message->console.cancel_requested) &&
           BOOL_VALID(message->console.kv_used_available) &&
           BOOL_VALID(message->console.progress_available) &&
           BOOL_VALID(message->console.selected_model_available) &&
           BOOL_VALID(message->console.explicit_reasoning_channel_supported) &&
           ENUM_VALID(message->console.reasoning_policy,
                      YVEX_REASONING_DISABLED,
                      YVEX_REASONING_MAXIMUM) &&
           partial_turn_fields_valid(&message->partial_turn) &&
           isfinite(message->queue_seconds) &&
           isfinite(message->prefill_seconds) &&
           isfinite(message->first_token_seconds) &&
           isfinite(message->decode_seconds) && isfinite(message->prefill_rate) &&
           isfinite(message->decode_rate) &&
           isfinite(message->publication_seconds) &&
           isfinite(message->draft_seconds) &&
           isfinite(message->verification_seconds) &&
           isfinite(message->speculative_commit_seconds) &&
           isfinite(message->mean_accepted_prefix) &&
           isfinite(message->effective_committed_rate) &&
           isfinite(message->confidence_logit_minimum) &&
           isfinite(message->confidence_logit_maximum) &&
           isfinite(message->confidence_logit_mean) &&
           message->draft_forward_count <= message->draft_cycle_count &&
           message->target_verification_count <= message->draft_forward_count &&
           message->selected_verification_tokens <= message->proposed_tokens &&
           message->accepted_draft_tokens + message->rejected_draft_tokens >=
               message->accepted_draft_tokens &&
           message->accepted_draft_tokens + message->rejected_draft_tokens <=
               message->selected_verification_tokens &&
           message->accepted_draft_tokens + message->rejected_draft_tokens +
                   message->discarded_draft_tokens >=
               message->accepted_draft_tokens + message->rejected_draft_tokens &&
           message->accepted_draft_tokens + message->rejected_draft_tokens +
                   message->discarded_draft_tokens ==
               message->proposed_tokens &&
           message->maximum_accepted_prefix <=
               message->accepted_draft_tokens &&
           message->target_correction_or_bonus_tokens <=
               message->target_verification_count &&
           (message->kind != YVEX_CLIENT_MESSAGE_TURN_COMPLETE ||
            message->status != YVEX_OK ||
            message->generation_phase != YVEX_CLIENT_PHASE_COMPLETE ||
            message->generation_mode != YVEX_SERVER_GENERATION_DSPARK ||
            (message->draft_cycle_count == message->draft_forward_count &&
             message->draft_forward_count ==
                 message->target_verification_count)) &&
           message->confidence_logit_count <= message->proposed_tokens &&
           (!message->confidence_logit_count ||
            (message->confidence_logit_minimum <=
                 message->confidence_logit_mean &&
             message->confidence_logit_mean <=
                 message->confidence_logit_maximum)) &&
           (message->confidence_logit_count ||
            (!message->confidence_logit_minimum &&
             !message->confidence_logit_maximum &&
             !message->confidence_logit_mean)) &&
           isfinite(message->event.seconds) && isfinite(message->event.rate) &&
           (message->kv_used_available || message->kv_used_bytes == 0u) &&
           (message->publication_timing_available ||
            message->publication_seconds == 0.0) &&
           (message->console.kv_used_available ||
            message->console.kv_used_bytes == 0u) &&
           (message->console.selected_model_available ||
            message->console.selected_model_identity[0] == '\0');
#undef BOOL_VALID
#undef ENUM_VALID
}

static int protocol_event_write(wire_writer *writer,
                                const yvex_server_event *event)
{
#define EVENT_U64(tag, field) \
    writer_u64(writer, tag, (unsigned long long)(field))
    int valid =
        EVENT_U64(TAG_EVENT_SEQUENCE, event->sequence) &&
        EVENT_U64(TAG_EVENT_WALL_TIME, event->wall_time_ns) &&
        EVENT_U64(TAG_EVENT_MONOTONIC_TIME, event->monotonic_time_ns) &&
        EVENT_U64(TAG_EVENT_PROCESS_ID, event->process_id) &&
        EVENT_U64(TAG_EVENT_KIND, event->kind) &&
        EVENT_U64(TAG_EVENT_SEVERITY, event->severity) &&
        writer_text(writer, TAG_EVENT_SESSION_ID, event->session_id) &&
        writer_text(writer, TAG_EVENT_REQUEST_ID, event->request_id) &&
        writer_text(writer, TAG_EVENT_TURN_ID, event->turn_id) &&
        writer_text(writer, TAG_EVENT_PHASE, event->phase) &&
        writer_text(writer, TAG_EVENT_PROVIDER_ADAPTER,
                    event->provider_adapter) &&
        writer_text(writer, TAG_EVENT_PROVIDER_REQUEST_ID,
                    event->provider_request_identity) &&
        writer_text(writer, TAG_EVENT_EXTERNAL_CORRELATION_ID,
                    event->external_correlation_id) &&
        EVENT_U64(TAG_EVENT_VALUE_A, event->value_a) &&
        EVENT_U64(TAG_EVENT_VALUE_B, event->value_b) &&
        EVENT_U64(TAG_EVENT_VALUE_C, event->value_c) &&
        EVENT_U64(TAG_EVENT_GENERATION_MODE, event->generation_mode) &&
        EVENT_U64(TAG_EVENT_SPECULATIVE_CYCLE, event->speculative_cycle) &&
        EVENT_U64(TAG_EVENT_PROPOSED_TOKENS, event->proposed_tokens) &&
        EVENT_U64(TAG_EVENT_SELECTED_VERIFICATION_TOKENS,
                  event->selected_verification_tokens) &&
        EVENT_U64(TAG_EVENT_ACCEPTED_TOKENS, event->accepted_tokens) &&
        EVENT_U64(TAG_EVENT_REJECTED_TOKENS, event->rejected_tokens) &&
        EVENT_U64(TAG_EVENT_DISCARDED_TOKENS, event->discarded_tokens) &&
        EVENT_U64(TAG_EVENT_VERIFICATION_COUNT, event->verification_count) &&
        EVENT_U64(TAG_EVENT_CONFIDENCE_LOGIT_COUNT,
                  event->confidence_logit_count) &&
        writer_double(writer, TAG_EVENT_CONFIDENCE_LOGIT_MINIMUM,
                      event->confidence_logit_minimum) &&
        writer_double(writer, TAG_EVENT_CONFIDENCE_LOGIT_MAXIMUM,
                      event->confidence_logit_maximum) &&
        writer_double(writer, TAG_EVENT_CONFIDENCE_LOGIT_MEAN,
                      event->confidence_logit_mean) &&
        writer_text(writer, TAG_EVENT_SPECULATION_POLICY_ID,
                    event->speculation_policy_identity) &&
        writer_double(writer, TAG_EVENT_SECONDS, event->seconds) &&
        writer_double(writer, TAG_EVENT_RATE, event->rate) &&
        writer_text(writer, TAG_EVENT_VARIANT_ID, event->variant_identity) &&
        writer_text(writer, TAG_EVENT_RUNTIME_MODEL_ID,
                    event->runtime_model_identity) &&
        writer_text(writer, TAG_EVENT_ARTIFACT_ID, event->artifact_identity) &&
        writer_text(writer, TAG_EVENT_IDENTITY, event->event_identity);
#undef EVENT_U64
    return valid;
}

static int protocol_message_core_write(wire_writer *writer,
                                       const yvex_client_message *message,
                                       unsigned long long message_flags)
{
#define MESSAGE_U64(tag, field) \
    writer_u64(writer, tag, (unsigned long long)(field))
    int valid =
        MESSAGE_U64(TAG_MESSAGE_KIND, message->kind) &&
        MESSAGE_U64(TAG_STATUS, (uint32_t)(int32_t)message->status) &&
        MESSAGE_U64(TAG_FAILURE_CLASS, message->failure_class) &&
        MESSAGE_U64(TAG_REQUEST_NUMBER, message->request_number) &&
        writer_text(writer, TAG_SESSION_NAME, message->session_name) &&
        writer_text(writer, TAG_REASON, message->reason) &&
        writer_field(writer, TAG_BYTES, message->bytes, message->byte_count) &&
        MESSAGE_U64(TAG_PROMPT_TOKENS, message->prompt_tokens) &&
        MESSAGE_U64(TAG_REUSED_TOKENS, message->reused_tokens) &&
        MESSAGE_U64(TAG_PREFILL_TOKENS, message->prefill_tokens) &&
        MESSAGE_U64(TAG_GENERATED_TOKENS, message->generated_tokens) &&
        MESSAGE_U64(TAG_FINAL_POSITION, message->final_position) &&
        MESSAGE_U64(TAG_TURN_COUNT, message->turn_count) &&
        MESSAGE_U64(TAG_CONTEXT_USED, message->context_used) &&
        MESSAGE_U64(TAG_KV_USED_BYTES, message->kv_used_bytes) &&
        MESSAGE_U64(TAG_GENERATION_MODE, message->generation_mode) &&
        MESSAGE_U64(TAG_DRAFT_CYCLE_COUNT, message->draft_cycle_count) &&
        MESSAGE_U64(TAG_DRAFT_FORWARD_COUNT, message->draft_forward_count) &&
        MESSAGE_U64(TAG_PROPOSED_TOKENS, message->proposed_tokens) &&
        MESSAGE_U64(TAG_SELECTED_VERIFICATION_TOKENS,
                    message->selected_verification_tokens) &&
        MESSAGE_U64(TAG_TARGET_VERIFICATION_COUNT,
                    message->target_verification_count) &&
        MESSAGE_U64(TAG_ACCEPTED_DRAFT_TOKENS,
                    message->accepted_draft_tokens) &&
        MESSAGE_U64(TAG_REJECTED_DRAFT_TOKENS,
                    message->rejected_draft_tokens) &&
        MESSAGE_U64(TAG_DISCARDED_DRAFT_TOKENS,
                    message->discarded_draft_tokens) &&
        MESSAGE_U64(TAG_TARGET_CORRECTION_OR_BONUS_TOKENS,
                    message->target_correction_or_bonus_tokens) &&
        MESSAGE_U64(TAG_MAXIMUM_ACCEPTED_PREFIX,
                    message->maximum_accepted_prefix) &&
        MESSAGE_U64(TAG_CONFIDENCE_LOGIT_COUNT,
                    message->confidence_logit_count) &&
        writer_double(writer, TAG_QUEUE_SECONDS, message->queue_seconds) &&
        writer_double(writer, TAG_PREFILL_SECONDS, message->prefill_seconds) &&
        writer_double(writer, TAG_FIRST_TOKEN_SECONDS,
                      message->first_token_seconds) &&
        writer_double(writer, TAG_DECODE_SECONDS, message->decode_seconds) &&
        writer_double(writer, TAG_PREFILL_RATE, message->prefill_rate) &&
        writer_double(writer, TAG_DECODE_RATE, message->decode_rate) &&
        writer_double(writer, TAG_PUBLICATION_SECONDS,
                      message->publication_seconds) &&
        writer_double(writer, TAG_DRAFT_SECONDS, message->draft_seconds) &&
        writer_double(writer, TAG_VERIFICATION_SECONDS,
                      message->verification_seconds) &&
        writer_double(writer, TAG_SPECULATIVE_COMMIT_SECONDS,
                      message->speculative_commit_seconds) &&
        writer_double(writer, TAG_MEAN_ACCEPTED_PREFIX,
                      message->mean_accepted_prefix) &&
        writer_double(writer, TAG_EFFECTIVE_COMMITTED_RATE,
                      message->effective_committed_rate) &&
        writer_double(writer, TAG_CONFIDENCE_LOGIT_MINIMUM,
                      message->confidence_logit_minimum) &&
        writer_double(writer, TAG_CONFIDENCE_LOGIT_MAXIMUM,
                      message->confidence_logit_maximum) &&
        writer_double(writer, TAG_CONFIDENCE_LOGIT_MEAN,
                      message->confidence_logit_mean) &&
        MESSAGE_U64(TAG_STOP_REASON, message->stop_reason) &&
        MESSAGE_U64(TAG_GENERATION_PHASE, message->generation_phase) &&
        MESSAGE_U64(TAG_CANCELLATION_CLASS, message->cancellation_class) &&
        MESSAGE_U64(TAG_STREAM_CHANNEL, message->stream_channel) &&
        MESSAGE_U64(TAG_MESSAGE_AVAILABILITY_FLAGS, message_flags) &&
        MESSAGE_U64(TAG_SESSION_STATE, message->session_state) &&
        writer_text(writer, TAG_SESSION_IDENTITY,
                    message->session_identity) &&
        writer_text(writer, TAG_TURN_IDENTITY, message->turn_identity) &&
        writer_text(writer, TAG_STATE_DIGEST, message->state_digest) &&
        writer_text(writer, TAG_GENERATED_TOKEN_IDENTITY,
                    message->generated_token_identity) &&
        writer_text(writer, TAG_GENERATED_TEXT_DIGEST,
                    message->generated_text_digest) &&
        writer_text(writer, TAG_SPECULATION_POLICY_ID,
                    message->speculation_policy_identity) &&
        MESSAGE_U64(TAG_PROVIDER_OUTPUT_KIND, message->provider_output_kind) &&
        MESSAGE_U64(TAG_PROVIDER_FINISH, message->provider_finish) &&
        MESSAGE_U64(TAG_COMPLETION_TOKENS, message->completion_tokens) &&
        MESSAGE_U64(TAG_TOTAL_TOKENS, message->total_tokens) &&
        writer_text(writer, TAG_PROVIDER_REQUEST_ID,
                    message->provider_request_identity) &&
        writer_text(writer, TAG_EXTERNAL_CORRELATION_ID,
                    message->external_correlation_id) &&
        writer_text(writer, TAG_TOOL_CALL_ID, message->tool_call_id) &&
        writer_text(writer, TAG_TOOL_NAME, message->tool_name);
#undef MESSAGE_U64
    return valid;
}

static int protocol_partial_write(wire_writer *writer,
                                  const yvex_client_partial_turn *partial)
{
    unsigned long long flags = (partial->available ? 1u : 0u) |
                               (partial->committed_progress ? 2u : 0u) |
                               (partial->reset_required ? 4u : 0u) |
                               (partial->draft_state_generation_available ? 8u : 0u) |
                               (partial->detokenizer_generation_available ? 16u : 0u);
#define PARTIAL_U64(tag, field) \
    writer_u64(writer, tag, (unsigned long long)(field))
    int valid =
        PARTIAL_U64(TAG_PARTIAL_FLAGS, flags) &&
        PARTIAL_U64(TAG_PARTIAL_FAILURE_STATUS,
                    (uint32_t)(int32_t)partial->failure_status) &&
        PARTIAL_U64(TAG_PARTIAL_FAILURE_CLASS, partial->failure_class) &&
        PARTIAL_U64(TAG_PARTIAL_STOP_REASON, partial->stop_reason) &&
        PARTIAL_U64(TAG_PARTIAL_INITIAL_POSITION, partial->initial_position) &&
        PARTIAL_U64(TAG_PARTIAL_FINAL_POSITION,
                    partial->final_committed_position) &&
        PARTIAL_U64(TAG_PARTIAL_COMMITTED_TOKENS,
                    partial->committed_token_count) &&
        PARTIAL_U64(TAG_PARTIAL_PUBLISHED_BYTES,
                    partial->published_text_bytes) &&
        PARTIAL_U64(TAG_PARTIAL_TARGET_GENERATION,
                    partial->target_state_generation) &&
        PARTIAL_U64(TAG_PARTIAL_DRAFT_GENERATION,
                    partial->draft_state_generation) &&
        PARTIAL_U64(TAG_PARTIAL_RNG_GENERATION, partial->rng_generation) &&
        PARTIAL_U64(TAG_PARTIAL_LEDGER_GENERATION,
                    partial->token_ledger_generation) &&
        PARTIAL_U64(TAG_PARTIAL_DETOKENIZER_GENERATION,
                    partial->detokenizer_generation) &&
        PARTIAL_U64(TAG_PARTIAL_MESSAGE_GENERATION,
                    partial->message_history_generation) &&
        PARTIAL_U64(TAG_PARTIAL_TRANSCRIPT_GENERATION,
                    partial->transcript_generation) &&
        writer_text(writer, TAG_PARTIAL_TARGET_IDENTITY,
                    partial->target_state_identity) &&
        writer_text(writer, TAG_PARTIAL_RNG_IDENTITY,
                    partial->rng_state_identity) &&
        writer_text(writer, TAG_PARTIAL_LEDGER_IDENTITY,
                    partial->token_ledger_identity) &&
        writer_text(writer, TAG_PARTIAL_TEXT_IDENTITY,
                    partial->published_text_identity);
#undef PARTIAL_U64
    return valid;
}

static int protocol_runtime_write(wire_writer *writer,
                                  const yvex_server_summary *runtime)
{
    unsigned long long flags = (runtime->runtime_ready ? 1u : 0u) |
                               (runtime->generation_ready ? 2u : 0u) |
                               (runtime->public_server_ready ? 4u : 0u) |
                               (runtime->openai_listener_enabled ? 8u : 0u) |
                               (runtime->openai_listener_ready ? 16u : 0u) |
                               (runtime->explicit_reasoning_channel_supported ? 32u : 0u);
#define RUNTIME_U64(tag, field) \
    writer_u64(writer, tag, (unsigned long long)(field))
    int valid =
        RUNTIME_U64(TAG_RUNTIME_STATUS, runtime->status) &&
        RUNTIME_U64(TAG_RUNTIME_BACKEND, runtime->backend) &&
        RUNTIME_U64(TAG_RUNTIME_GENERATION_MODE, runtime->generation_mode) &&
        writer_text(writer, TAG_SOCKET_PATH, runtime->socket_path) &&
        writer_text(writer, TAG_TARGET_ID, runtime->target_id) &&
        writer_text(writer, TAG_RUNTIME_MODEL_ID,
                    runtime->runtime_model_identity) &&
        writer_text(writer, TAG_RUNTIME_BINDING_ID,
                    runtime->runtime_binding_identity) &&
        writer_text(writer, TAG_ARTIFACT_ID, runtime->artifact_identity) &&
        writer_text(writer, TAG_PHYSICAL_VARIANT_ID,
                    runtime->physical_variant_identity) &&
        RUNTIME_U64(TAG_CONTEXT_CAPACITY, runtime->context_capacity) &&
        RUNTIME_U64(TAG_SESSION_COUNT, runtime->session_count) &&
        RUNTIME_U64(TAG_RUNTIME_REQUEST_COUNT, runtime->request_count) &&
        RUNTIME_U64(TAG_RUNTIME_FLAGS, flags) &&
        RUNTIME_U64(TAG_PREFILL_CHUNK_TOKENS, runtime->prefill_chunk_tokens) &&
        RUNTIME_U64(TAG_RUNTIME_MAXIMUM_NEW_TOKENS,
                    runtime->maximum_new_tokens) &&
        RUNTIME_U64(TAG_RUNTIME_MAXIMUM_OUTPUT_BYTES,
                    runtime->maximum_output_bytes) &&
        RUNTIME_U64(TAG_RUNTIME_MAXIMUM_SESSIONS,
                    runtime->maximum_sessions) &&
        RUNTIME_U64(TAG_OPENAI_PORT, runtime->openai_port) &&
        RUNTIME_U64(TAG_RUNTIME_QUEUE_CAPACITY,
                    runtime->request_queue_capacity) &&
        RUNTIME_U64(TAG_RUNTIME_OPENAI_TIMEOUT, runtime->openai_timeout_ms) &&
        RUNTIME_U64(TAG_RUNTIME_TRACE_LEVEL, runtime->trace_level) &&
        writer_metrics(writer, &runtime->metrics);
#undef RUNTIME_U64
    return valid;
}

static int protocol_console_write(wire_writer *writer,
                                  const yvex_console_status *console)
{
    unsigned long long flags = (console->runtime_ready ? 1u : 0u) |
                               (console->session_available ? 2u : 0u) |
                               (console->attached ? 4u : 0u) |
                               (console->cancel_requested ? 8u : 0u) |
                               (console->kv_used_available ? 16u : 0u) |
                               (console->progress_available ? 32u : 0u) |
                               (console->selected_model_available ? 64u : 0u) |
                               (console->explicit_reasoning_channel_supported ? 128u : 0u);
#define CONSOLE_U64(tag, field) \
    writer_u64(writer, tag, (unsigned long long)(field))
    int valid =
        CONSOLE_U64(TAG_CONSOLE_FLAGS, flags) &&
        CONSOLE_U64(TAG_CONSOLE_BACKEND, console->backend) &&
        CONSOLE_U64(TAG_CONSOLE_SESSION_STATE, console->session_state) &&
        CONSOLE_U64(TAG_CONSOLE_POSITION, console->position) &&
        CONSOLE_U64(TAG_CONSOLE_TURN_COUNT, console->turn_count) &&
        CONSOLE_U64(TAG_CONSOLE_CONTEXT_CAPACITY, console->context_capacity) &&
        CONSOLE_U64(TAG_CONSOLE_CONTEXT_USED, console->context_used) &&
        CONSOLE_U64(TAG_CONSOLE_KV_USED_BYTES, console->kv_used_bytes) &&
        CONSOLE_U64(TAG_CONSOLE_PHASE, console->generation_phase) &&
        CONSOLE_U64(TAG_CONSOLE_CANCELLATION, console->cancellation_class) &&
        CONSOLE_U64(TAG_CONSOLE_REASONING_POLICY,
                    console->reasoning_policy) &&
        writer_text(writer, TAG_CONSOLE_LIVE_MODEL_ID,
                    console->live_model_identity) &&
        writer_text(writer, TAG_CONSOLE_VARIANT_ID,
                    console->physical_variant_identity) &&
        writer_text(writer, TAG_CONSOLE_SESSION_NAME, console->session_name) &&
        writer_text(writer, TAG_CONSOLE_SELECTED_MODEL_ID,
                    console->selected_model_identity);
#undef CONSOLE_U64
    return valid;
}
int yvex_protocol_message_encode(const yvex_client_message *message,
                                 unsigned char *output,
                                 unsigned long long capacity,
                                 unsigned long long *byte_count,
                                 yvex_error *err)
{
    wire_writer writer = {output, capacity, 0u};
    unsigned long long message_flags =
        (message && message->kv_used_available ? 1u : 0u) |
        (message && message->publication_timing_available ? 2u : 0u);
    if (byte_count) *byte_count = 0u;
    if (!message || !output || !byte_count ||
        message->schema_version != YVEX_LOCAL_PROTOCOL_VERSION ||
        !message_fields_valid(message) ||
        (message->kind == YVEX_CLIENT_MESSAGE_CONSOLE_STATUS &&
         message->console.schema_version != 1u) ||
        message->byte_count > sizeof(message->bytes))
        return protocol_refuse(err, YVEX_ERR_INVALID_ARG,
                               "complete bounded server message is required");
    if (message->kind == YVEX_CLIENT_MESSAGE_EVENT &&
        yvex_server_event_validate(&message->event, err) != YVEX_OK)
        return yvex_error_code(err);
    if (!protocol_message_core_write(&writer, message, message_flags) ||
        !protocol_partial_write(&writer, &message->partial_turn) ||
        !protocol_runtime_write(&writer, &message->runtime) ||
        !protocol_console_write(&writer, &message->console) ||
        !protocol_event_write(&writer, &message->event))
        return protocol_refuse(err, YVEX_ERR_BOUNDS,
                               "server message does not fit admitted frame");
    *byte_count = writer.count;
    yvex_error_clear(err);
    return YVEX_OK;
}
static int message_speculation_u64_field(yvex_client_message *candidate,
                                         unsigned int tag,
                                         const unsigned char *bytes,
                                         unsigned long long count)
{
    unsigned long long value;

    switch (tag) {
    case TAG_DRAFT_CYCLE_COUNT:
    case TAG_DRAFT_FORWARD_COUNT:
    case TAG_PROPOSED_TOKENS:
    case TAG_SELECTED_VERIFICATION_TOKENS:
    case TAG_TARGET_VERIFICATION_COUNT:
    case TAG_ACCEPTED_DRAFT_TOKENS:
    case TAG_REJECTED_DRAFT_TOKENS:
    case TAG_DISCARDED_DRAFT_TOKENS:
    case TAG_TARGET_CORRECTION_OR_BONUS_TOKENS:
    case TAG_MAXIMUM_ACCEPTED_PREFIX:
    case TAG_CONFIDENCE_LOGIT_COUNT:
        break;
    default:
        return 0;
    }
    if (!reader_u64(bytes, count, &value)) return -1;
    switch (tag) {
    case TAG_DRAFT_CYCLE_COUNT: candidate->draft_cycle_count = value; break;
    case TAG_DRAFT_FORWARD_COUNT: candidate->draft_forward_count = value; break;
    case TAG_PROPOSED_TOKENS: candidate->proposed_tokens = value; break;
    case TAG_SELECTED_VERIFICATION_TOKENS:
        candidate->selected_verification_tokens = value;
        break;
    case TAG_TARGET_VERIFICATION_COUNT:
        candidate->target_verification_count = value;
        break;
    case TAG_ACCEPTED_DRAFT_TOKENS: candidate->accepted_draft_tokens = value; break;
    case TAG_REJECTED_DRAFT_TOKENS: candidate->rejected_draft_tokens = value; break;
    case TAG_DISCARDED_DRAFT_TOKENS: candidate->discarded_draft_tokens = value; break;
    case TAG_TARGET_CORRECTION_OR_BONUS_TOKENS:
        candidate->target_correction_or_bonus_tokens = value;
        break;
    case TAG_MAXIMUM_ACCEPTED_PREFIX: candidate->maximum_accepted_prefix = value; break;
    case TAG_CONFIDENCE_LOGIT_COUNT: candidate->confidence_logit_count = value; break;
    default: return 0;
    }
    return 1;
}

static int message_base_field(yvex_client_message *candidate, unsigned int tag,
                              const unsigned char *bytes,
                              unsigned long long count, int *have_kind)
{
    unsigned long long value = 0ull;
    int valid = 1;
#define BASE_U64(field) (reader_u64(bytes, count, &value) ? ((field) = value, 1) : 0)
    switch (tag) {
    case TAG_MESSAGE_KIND:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_CLIENT_MESSAGE_CONSOLE_STATUS;
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
    case TAG_TURN_COUNT: valid = BASE_U64(candidate->turn_count); break;
    case TAG_CONTEXT_USED: valid = BASE_U64(candidate->context_used); break;
    case TAG_KV_USED_BYTES: valid = BASE_U64(candidate->kv_used_bytes); break;
    case TAG_GENERATION_MODE:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_SERVER_GENERATION_DSPARK;
        if (valid)
            candidate->generation_mode = (yvex_server_generation_mode)value;
        break;
    case TAG_QUEUE_SECONDS: valid = reader_double(bytes, count, &candidate->queue_seconds); break;
    case TAG_PREFILL_SECONDS: valid = reader_double(bytes, count, &candidate->prefill_seconds); break;
    case TAG_FIRST_TOKEN_SECONDS:
        valid = reader_double(bytes, count, &candidate->first_token_seconds);
        break;
    case TAG_DECODE_SECONDS: valid = reader_double(bytes, count, &candidate->decode_seconds); break;
    case TAG_PREFILL_RATE: valid = reader_double(bytes, count, &candidate->prefill_rate); break;
    case TAG_DECODE_RATE: valid = reader_double(bytes, count, &candidate->decode_rate); break;
    case TAG_PUBLICATION_SECONDS:
        valid = reader_double(bytes, count, &candidate->publication_seconds);
        break;
    case TAG_DRAFT_SECONDS:
        valid = reader_double(bytes, count, &candidate->draft_seconds);
        break;
    case TAG_VERIFICATION_SECONDS:
        valid = reader_double(bytes, count, &candidate->verification_seconds);
        break;
    case TAG_SPECULATIVE_COMMIT_SECONDS:
        valid = reader_double(bytes, count,
                              &candidate->speculative_commit_seconds);
        break;
    case TAG_MEAN_ACCEPTED_PREFIX:
        valid = reader_double(bytes, count, &candidate->mean_accepted_prefix);
        break;
    case TAG_EFFECTIVE_COMMITTED_RATE:
        valid = reader_double(bytes, count,
                              &candidate->effective_committed_rate);
        break;
    case TAG_CONFIDENCE_LOGIT_MINIMUM:
        valid = reader_double(bytes, count,
                              &candidate->confidence_logit_minimum);
        break;
    case TAG_CONFIDENCE_LOGIT_MAXIMUM:
        valid = reader_double(bytes, count,
                              &candidate->confidence_logit_maximum);
        break;
    case TAG_CONFIDENCE_LOGIT_MEAN:
        valid = reader_double(bytes, count,
                              &candidate->confidence_logit_mean);
        break;
    case TAG_STOP_REASON:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_GENERATION_STOP_OUTPUT_FAILURE;
        if (valid) candidate->stop_reason = (unsigned int)value;
        break;
    case TAG_GENERATION_PHASE:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_CLIENT_PHASE_FAILED;
        if (valid) candidate->generation_phase = (yvex_client_generation_phase)value;
        break;
    case TAG_CANCELLATION_CLASS:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_CLIENT_CANCELLATION_FAILED;
        if (valid)
            candidate->cancellation_class = (yvex_client_cancellation_class)value;
        break;
    case TAG_STREAM_CHANNEL:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_CLIENT_STREAM_CONTROL_EVENT;
        if (valid) candidate->stream_channel = (yvex_client_stream_channel)value;
        break;
    case TAG_MESSAGE_AVAILABILITY_FLAGS:
        valid = reader_u64(bytes, count, &value) && !(value & ~3u);
        candidate->kv_used_available = (value & 1u) != 0u;
        candidate->publication_timing_available = (value & 2u) != 0u;
        break;
    case TAG_SESSION_STATE:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_SERVER_SESSION_FAILED;
        if (valid) candidate->session_state = (yvex_server_session_state)value;
        break;
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
    case TAG_SPECULATION_POLICY_ID:
        valid = reader_text(bytes, count,
                            candidate->speculation_policy_identity,
                            sizeof(candidate->speculation_policy_identity));
        break;
    case TAG_PROVIDER_OUTPUT_KIND:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_PROVIDER_OUTPUT_EXPLICIT_REASONING;
        if (valid)
            candidate->provider_output_kind = (yvex_provider_output_kind)value;
        break;
    case TAG_PROVIDER_FINISH:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_PROVIDER_FINISH_FAILED;
        if (valid)
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
    default: return message_speculation_u64_field(candidate, tag, bytes, count);
    }
#undef BASE_U64
    return valid ? 1 : -1;
}

static int message_partial_field(yvex_client_message *candidate,
                                 unsigned int tag,
                                 const unsigned char *bytes,
                                 unsigned long long count)
{
    yvex_client_partial_turn *partial = &candidate->partial_turn;
    unsigned long long *number = NULL;
    char *identity = NULL;
    unsigned long long value = 0ull;
    int valid = 1;
    switch (tag) {
    case TAG_PARTIAL_FLAGS:
        valid = reader_u64(bytes, count, &value) && !(value & ~31u);
        partial->available = (value & 1u) != 0u;
        partial->committed_progress = (value & 2u) != 0u;
        partial->reset_required = (value & 4u) != 0u;
        partial->draft_state_generation_available = (value & 8u) != 0u;
        partial->detokenizer_generation_available = (value & 16u) != 0u;
        partial->schema_version = partial->available
                                      ? YVEX_CLIENT_PARTIAL_TURN_SCHEMA_V1
                                      : 0u;
        break;
    case TAG_PARTIAL_FAILURE_STATUS:
        valid = reader_u64(bytes, count, &value) && value <= UINT32_MAX;
        if (valid) partial->failure_status = (int)(int32_t)(uint32_t)value;
        break;
    case TAG_PARTIAL_FAILURE_CLASS:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT;
        if (valid) partial->failure_class = (yvex_client_failure_class)value;
        break;
    case TAG_PARTIAL_STOP_REASON:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_GENERATION_STOP_OUTPUT_FAILURE;
        if (valid) partial->stop_reason = (unsigned int)value;
        break;
    case TAG_PARTIAL_INITIAL_POSITION: number = &partial->initial_position; break;
    case TAG_PARTIAL_FINAL_POSITION: number = &partial->final_committed_position; break;
    case TAG_PARTIAL_COMMITTED_TOKENS: number = &partial->committed_token_count; break;
    case TAG_PARTIAL_PUBLISHED_BYTES: number = &partial->published_text_bytes; break;
    case TAG_PARTIAL_TARGET_GENERATION: number = &partial->target_state_generation; break;
    case TAG_PARTIAL_DRAFT_GENERATION: number = &partial->draft_state_generation; break;
    case TAG_PARTIAL_RNG_GENERATION: number = &partial->rng_generation; break;
    case TAG_PARTIAL_LEDGER_GENERATION: number = &partial->token_ledger_generation; break;
    case TAG_PARTIAL_DETOKENIZER_GENERATION:
        number = &partial->detokenizer_generation;
        break;
    case TAG_PARTIAL_MESSAGE_GENERATION: number = &partial->message_history_generation; break;
    case TAG_PARTIAL_TRANSCRIPT_GENERATION: number = &partial->transcript_generation; break;
    case TAG_PARTIAL_TARGET_IDENTITY: identity = partial->target_state_identity; break;
    case TAG_PARTIAL_RNG_IDENTITY: identity = partial->rng_state_identity; break;
    case TAG_PARTIAL_LEDGER_IDENTITY: identity = partial->token_ledger_identity; break;
    case TAG_PARTIAL_TEXT_IDENTITY: identity = partial->published_text_identity; break;
    default: return 0;
    }
    if (number) valid = reader_u64(bytes, count, number);
    if (identity)
        valid = reader_text(bytes, count, identity, YVEX_SHA256_HEX_CAP);
    return valid ? 1 : -1;
}
static int message_runtime_field(yvex_client_message *candidate,
                                 unsigned int tag,
                                 const unsigned char *bytes,
                                 unsigned long long count)
{
    unsigned long long value = 0ull;
    int valid = 1;
#define RUNTIME_U64(field) (reader_u64(bytes, count, &value) ? ((field) = value, 1) : 0)
    switch (tag) {
    case TAG_RUNTIME_STATUS:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_SERVER_STATUS_FAILED;
        if (valid) candidate->runtime.status = (yvex_server_status)value;
        break;
    case TAG_RUNTIME_BACKEND:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_BACKEND_KIND_ROCM;
        if (valid) candidate->runtime.backend = (yvex_backend_kind)value;
        break;
    case TAG_RUNTIME_GENERATION_MODE:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_SERVER_GENERATION_DSPARK;
        if (valid)
            candidate->runtime.generation_mode =
                (yvex_server_generation_mode)value;
        break;
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
    case TAG_PREFILL_CHUNK_TOKENS:
        valid = RUNTIME_U64(candidate->runtime.prefill_chunk_tokens);
        break;
    case TAG_RUNTIME_MAXIMUM_NEW_TOKENS:
        valid = RUNTIME_U64(candidate->runtime.maximum_new_tokens);
        break;
    case TAG_RUNTIME_MAXIMUM_OUTPUT_BYTES:
        valid = RUNTIME_U64(candidate->runtime.maximum_output_bytes);
        break;
    case TAG_RUNTIME_MAXIMUM_SESSIONS:
        valid = RUNTIME_U64(candidate->runtime.maximum_sessions);
        break;
    case TAG_RUNTIME_QUEUE_CAPACITY:
        valid = RUNTIME_U64(candidate->runtime.request_queue_capacity);
        break;
    case TAG_RUNTIME_OPENAI_TIMEOUT:
        valid = RUNTIME_U64(candidate->runtime.openai_timeout_ms);
        break;
    case TAG_RUNTIME_TRACE_LEVEL:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_SERVER_TRACE_FULL;
        if (valid) candidate->runtime.trace_level = (yvex_server_trace_level)value;
        break;
    case TAG_OPENAI_PORT:
        valid = reader_u64(bytes, count, &value) && value <= 65535u;
        if (valid) candidate->runtime.openai_port = (unsigned short)value;
        break;
    case TAG_RUNTIME_FLAGS:
        valid = reader_u64(bytes, count, &value) && !(value & ~63u);
        candidate->runtime.runtime_ready = (value & 1u) != 0u;
        candidate->runtime.generation_ready = (value & 2u) != 0u;
        candidate->runtime.public_server_ready = (value & 4u) != 0u;
        candidate->runtime.openai_listener_enabled = (value & 8u) != 0u;
        candidate->runtime.openai_listener_ready = (value & 16u) != 0u;
        candidate->runtime.explicit_reasoning_channel_supported =
            (value & 32u) != 0u;
        break;
    case TAG_METRICS: valid = reader_metrics(bytes, count, &candidate->runtime.metrics); break;
    default: return 0;
    }
#undef RUNTIME_U64
    return valid ? 1 : -1;
}

static int message_console_field(yvex_client_message *candidate,
                                 unsigned int tag,
                                 const unsigned char *bytes,
                                 unsigned long long count)
{
    unsigned long long value = 0u;
    int valid = 1;
#define CONSOLE_U64(field) (reader_u64(bytes, count, &value) ? ((field) = value, 1) : 0)
    switch (tag) {
    case TAG_CONSOLE_FLAGS:
        valid = reader_u64(bytes, count, &value) && !(value & ~255u);
        candidate->console.runtime_ready = (value & 1u) != 0u;
        candidate->console.session_available = (value & 2u) != 0u;
        candidate->console.attached = (value & 4u) != 0u;
        candidate->console.cancel_requested = (value & 8u) != 0u;
        candidate->console.kv_used_available = (value & 16u) != 0u;
        candidate->console.progress_available = (value & 32u) != 0u;
        candidate->console.selected_model_available = (value & 64u) != 0u;
        candidate->console.explicit_reasoning_channel_supported =
            (value & 128u) != 0u;
        break;
    case TAG_CONSOLE_BACKEND:
        valid = reader_u64(bytes, count, &value) && value <= YVEX_BACKEND_KIND_ROCM;
        if (valid) candidate->console.backend = (yvex_backend_kind)value;
        break;
    case TAG_CONSOLE_SESSION_STATE:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_SERVER_SESSION_FAILED;
        if (valid) candidate->console.session_state = (yvex_server_session_state)value;
        break;
    case TAG_CONSOLE_POSITION: valid = CONSOLE_U64(candidate->console.position); break;
    case TAG_CONSOLE_TURN_COUNT: valid = CONSOLE_U64(candidate->console.turn_count); break;
    case TAG_CONSOLE_CONTEXT_CAPACITY:
        valid = CONSOLE_U64(candidate->console.context_capacity);
        break;
    case TAG_CONSOLE_CONTEXT_USED:
        valid = CONSOLE_U64(candidate->console.context_used);
        break;
    case TAG_CONSOLE_KV_USED_BYTES:
        valid = CONSOLE_U64(candidate->console.kv_used_bytes);
        break;
    case TAG_CONSOLE_PHASE:
        valid = reader_u64(bytes, count, &value) && value <= YVEX_CLIENT_PHASE_FAILED;
        if (valid) candidate->console.generation_phase = (yvex_client_generation_phase)value;
        break;
    case TAG_CONSOLE_CANCELLATION:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_CLIENT_CANCELLATION_FAILED;
        if (valid)
            candidate->console.cancellation_class =
                (yvex_client_cancellation_class)value;
        break;
    case TAG_CONSOLE_REASONING_POLICY:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_REASONING_MAXIMUM;
        if (valid)
            candidate->console.reasoning_policy =
                (yvex_reasoning_policy)value;
        break;
    case TAG_CONSOLE_LIVE_MODEL_ID:
        valid = reader_text(bytes, count, candidate->console.live_model_identity,
                            sizeof(candidate->console.live_model_identity));
        break;
    case TAG_CONSOLE_VARIANT_ID:
        valid = reader_text(bytes, count,
                            candidate->console.physical_variant_identity,
                            sizeof(candidate->console.physical_variant_identity));
        break;
    case TAG_CONSOLE_SESSION_NAME:
        valid = reader_text(bytes, count, candidate->console.session_name,
                            sizeof(candidate->console.session_name));
        break;
    case TAG_CONSOLE_SELECTED_MODEL_ID:
        valid = reader_text(bytes, count,
                            candidate->console.selected_model_identity,
                            sizeof(candidate->console.selected_model_identity));
        break;
    default: return 0;
    }
#undef CONSOLE_U64
    return valid ? 1 : -1;
}
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
    case TAG_EVENT_KIND:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE;
        if (valid) candidate->event.kind = (yvex_server_event_kind)value;
        break;
    case TAG_EVENT_SEVERITY:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_SERVER_SEVERITY_FATAL;
        if (valid) candidate->event.severity = (yvex_server_event_severity)value;
        break;
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
    case TAG_EVENT_GENERATION_MODE:
        valid = reader_u64(bytes, count, &value) &&
                value <= YVEX_SERVER_GENERATION_DSPARK;
        if (valid)
            candidate->event.generation_mode =
                (yvex_server_generation_mode)value;
        break;
    case TAG_EVENT_SPECULATIVE_CYCLE:
        valid = EVENT_U64(candidate->event.speculative_cycle);
        break;
    case TAG_EVENT_PROPOSED_TOKENS:
        valid = EVENT_U64(candidate->event.proposed_tokens);
        break;
    case TAG_EVENT_SELECTED_VERIFICATION_TOKENS:
        valid = EVENT_U64(candidate->event.selected_verification_tokens);
        break;
    case TAG_EVENT_ACCEPTED_TOKENS:
        valid = EVENT_U64(candidate->event.accepted_tokens);
        break;
    case TAG_EVENT_REJECTED_TOKENS:
        valid = EVENT_U64(candidate->event.rejected_tokens);
        break;
    case TAG_EVENT_DISCARDED_TOKENS:
        valid = EVENT_U64(candidate->event.discarded_tokens);
        break;
    case TAG_EVENT_VERIFICATION_COUNT:
        valid = EVENT_U64(candidate->event.verification_count);
        break;
    case TAG_EVENT_CONFIDENCE_LOGIT_COUNT:
        valid = EVENT_U64(candidate->event.confidence_logit_count);
        break;
    case TAG_EVENT_CONFIDENCE_LOGIT_MINIMUM:
        valid = reader_double(bytes, count,
                              &candidate->event.confidence_logit_minimum);
        break;
    case TAG_EVENT_CONFIDENCE_LOGIT_MAXIMUM:
        valid = reader_double(bytes, count,
                              &candidate->event.confidence_logit_maximum);
        break;
    case TAG_EVENT_CONFIDENCE_LOGIT_MEAN:
        valid = reader_double(bytes, count,
                              &candidate->event.confidence_logit_mean);
        break;
    case TAG_EVENT_SPECULATION_POLICY_ID:
        valid = reader_text(bytes, count,
                            candidate->event.speculation_policy_identity,
                            sizeof(candidate->event.speculation_policy_identity));
        break;
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

static int message_publish(const yvex_client_message *candidate, int next,
                           int valid, int have_kind,
                           yvex_client_message *message, yvex_error *err)
{
    if (next < 0 || !valid || !have_kind || !message_fields_valid(candidate))
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
    candidate.console.schema_version = 1u;
    candidate.event.schema_version = YVEX_RUNTIME_EVENT_SCHEMA_VERSION;
    while ((next = reader_next(&reader, &tag, &bytes, &count)) > 0 && valid) {
        int field = message_base_field(&candidate, tag, bytes, count,
                                       &have_kind);
        if (!field)
            field = message_partial_field(&candidate, tag, bytes, count);
        if (!field)
            field = message_runtime_field(&candidate, tag, bytes, count);
        if (!field)
            field = message_console_field(&candidate, tag, bytes, count);
        if (!field)
            field = message_event_field(&candidate, tag, bytes, count);
        if (field <= 0) valid = 0;
    }
    return message_publish(&candidate, next, valid, have_kind, message, err);
}

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
        get_u16(header + 6u) != expected_kind ||
        length > YVEX_SERVER_FRAME_MAX_BYTES)
        return protocol_refuse(err, YVEX_ERR_FORMAT,
                               "local protocol frame header is invalid");
    if (get_u16(header + 4u) != YVEX_LOCAL_PROTOCOL_VERSION)
        return protocol_refuse(err, YVEX_ERR_FORMAT,
                               "local protocol version is incompatible; version 6 is required");
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
int yvex_client_connect(yvex_client **out, const char *socket_path,
                        yvex_error *err)
{
    yvex_client_request handshake;
    yvex_client_message response;
    yvex_provider_sampling sampling;
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
    yvex_provider_sampling_default(&sampling);
    handshake.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    handshake.operation = YVEX_CLIENT_OP_HANDSHAKE;
    handshake.temperature = sampling.temperature;
    handshake.top_p = sampling.top_p;
    handshake.typical_p = sampling.typical_p;
    if (yvex_client_send(client, &handshake, err) != YVEX_OK ||
        yvex_client_receive(client, &response, err) != YVEX_OK ||
        response.kind != YVEX_CLIENT_MESSAGE_ACK ||
        response.status != YVEX_OK || strcmp(response.reason, "protocol-v6") != 0) {
        (void)close(client->fd);
        memset(client, 0, sizeof(*client));
        free(client);
        if (yvex_error_code(err) == YVEX_OK)
            yvex_error_set(err, YVEX_ERR_FORMAT, "server.protocol.handshake",
                           "daemon did not admit local protocol version 6");
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
