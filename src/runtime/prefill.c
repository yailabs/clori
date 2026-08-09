/*
 * Admit production activation chunks and execute them through persistent runtime state.
 *
 * Records are canonical and every successful chunk commits all prepared attention layers. Consumes
 * the common runtime model/session and graph executor without reconstructing compiler truth.
 */
#define _GNU_SOURCE
#include <yvex/internal/runtime_prefill.h>
#include <yvex/internal/core.h>

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

#define ACTIVATION_MAGIC_BYTES 16u
#define ACTIVATION_IDENTITY_BYTES 64u
#define ACTIVATION_HEADER_BYTES 440u
#define ACTIVATION_RECORD_BYTES 112u

static const unsigned char activation_magic[ACTIVATION_MAGIC_BYTES] = {
    'Y', 'V', 'E', 'X', 'A', 'C', 'T', 'I', 'V', 'A', 'T', 'I', 'O', 'N', '1', '\0'};

struct yvex_runtime_activation_input {
    yvex_runtime_activation_input_summary summary;
    yvex_runtime_activation_layer_record *records;
    const float *payload;
    unsigned char *mapping;
    size_t mapping_bytes;
    int fd, file_backed;
    struct stat snapshot;
};

static int activation_refuse(yvex_error *err, yvex_status status,
                             const char *reason)
{
    yvex_error_set(err, status, "runtime.activation-input", reason);
    return status;
}

static int activation_host_supported(void)
{
    const uint32_t one = 1u;
    return sizeof(float) == 4u && *(const unsigned char *)&one == 1u;
}

static void activation_put_u32(unsigned char *out, uint32_t value)
{
    unsigned int index;
    for (index = 0u; index < 4u; ++index)
        out[index] = (unsigned char)(value >> (index * 8u));
}

static void activation_put_u64(unsigned char *out, uint64_t value)
{
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        out[index] = (unsigned char)(value >> (index * 8u));
}

static uint32_t activation_get_u32(const unsigned char *input)
{
    uint32_t value = 0u;
    unsigned int index;
    for (index = 0u; index < 4u; ++index)
        value |= (uint32_t)input[index] << (index * 8u);
    return value;
}

static uint64_t activation_get_u64(const unsigned char *input)
{
    uint64_t value = 0ull;
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        value |= (uint64_t)input[index] << (index * 8u);
    return value;
}

static void activation_identity_get(char output[YVEX_SHA256_HEX_CAP],
                                    const unsigned char *input)
{
    memcpy(output, input, ACTIVATION_IDENTITY_BYTES);
    output[ACTIVATION_IDENTITY_BYTES] = '\0';
}

static int activation_hash_u64s(yvex_sha256 *hash,
                                const unsigned long long *fields, size_t count)
{
    size_t index;
    for (index = 0u; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, fields[index])) return 0;
    return 1;
}

static int activation_payload_digest(
    const float *payload, unsigned long long bytes,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if ((!payload && bytes) || bytes > (unsigned long long)SIZE_MAX)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, payload, (size_t)bytes) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

/*
 * Derive the canonical input identity.
 *
 * Excludes paths, pointers, padding, and timing.
 */
static int activation_input_identity(
    const yvex_runtime_activation_input_summary *summary,
    const yvex_runtime_activation_layer_record *records,
    char output[YVEX_SHA256_HEX_CAP])
{
    const unsigned long long fields[] = {
        summary->schema_version, summary->operation_scope,
        summary->token_start, summary->token_count,
        summary->layer_count, summary->payload_bytes};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.activation-input.v1") ||
        !activation_hash_u64s(&hash, fields,
                              sizeof(fields) / sizeof(fields[0])) ||
        !yvex_sha256_update_text(&hash, summary->logical_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->runtime_numeric_identity) ||
        !yvex_sha256_update_text(&hash, summary->runtime_descriptor_identity) ||
        !yvex_sha256_update_text(&hash, summary->attention_plan_identity) ||
        !yvex_sha256_update_text(&hash, summary->payload_digest))
        return 0;
    for (index = 0ull; index < summary->layer_count; ++index) {
        const yvex_runtime_activation_layer_record *record = &records[index];
        const unsigned long long record_fields[] = {
            record->ordinal, record->layer_index, record->width, record->stride,
            record->payload_offset, record->payload_bytes};
        if (!activation_hash_u64s(
                &hash, record_fields,
                sizeof(record_fields) / sizeof(record_fields[0])) ||
            !yvex_sha256_update_text(&hash, record->layer_identity))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int activation_summary_valid(
    const yvex_runtime_activation_input_summary *summary)
{
    return summary &&
           summary->schema_version == YVEX_RUNTIME_ACTIVATION_INPUT_SCHEMA_V1 &&
           (summary->operation_scope == YVEX_ATTENTION_OPERATION_CORE ||
            summary->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE) &&
           summary->token_count && summary->layer_count &&
           summary->payload_bytes &&
           yvex_sha256_hex_valid(summary->logical_model_identity) &&
           yvex_sha256_hex_valid(summary->runtime_numeric_identity) &&
           yvex_sha256_hex_valid(summary->runtime_descriptor_identity) &&
           yvex_sha256_hex_valid(summary->attention_plan_identity);
}

/*
 * Validate record ordering, extents, uniqueness, and finite payload.
 *
 * Summary, record directory, payload, and identity policy.
 */
static int activation_records_validate(
    const yvex_runtime_activation_input_summary *summary,
    const yvex_runtime_activation_layer_record *records,
    const float *payload, int require_identities, yvex_error *err)
{
    unsigned long long index, offset = 0ull;
    if (!activation_summary_valid(summary) || !records || !payload)
        return activation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "complete activation summary, records, and payload are required");
    for (index = 0ull; index < summary->layer_count; ++index) {
        const yvex_runtime_activation_layer_record *record = &records[index];
        unsigned long long elements, bytes, prior;
        if (record->ordinal != index || !record->width ||
            record->stride != record->width ||
            record->payload_offset != offset ||
            !yvex_core_u64_mul(summary->token_count, record->stride, &elements) ||
            !yvex_core_u64_mul(elements, sizeof(float), &bytes) ||
            record->payload_bytes != bytes ||
            !yvex_core_u64_add(offset, bytes, &offset) ||
            (require_identities &&
             !yvex_sha256_hex_valid(record->layer_identity)))
            return activation_refuse(
                err, YVEX_ERR_FORMAT,
                "activation layer ordering, dimensions, or payload range is malformed");
        for (prior = 0ull; prior < index; ++prior)
            if (records[prior].layer_index == record->layer_index)
                return activation_refuse(
                    err, YVEX_ERR_FORMAT,
                    "activation layer identity is duplicated");
    }
    if (offset != summary->payload_bytes)
        return activation_refuse(
            err, YVEX_ERR_FORMAT,
            "activation payload ranges are not exact and contiguous");
    for (index = 0ull; index < summary->payload_bytes / sizeof(float); ++index)
        if (!isfinite(payload[index]))
            return activation_refuse(
                err, YVEX_ERR_FORMAT,
                "activation payload contains a non-finite value");
    return YVEX_OK;
}

/*
 * Derive one runtime-plan layer identity.
 *
 * Plan identity, ordinal, layer facts, and scope. No activation bytes enter this identity.
 */
int yvex_runtime_activation_layer_identity_compute(
    const char *attention_plan_identity, unsigned long long ordinal,
    const yvex_attention_layer_plan *layer,
    yvex_attention_operation_scope operation_scope,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    const unsigned long long fields[] = {
        ordinal, layer ? layer->layer_index : 0ull,
        layer ? (unsigned long long)layer->attention_class : 0ull,
        layer ? layer->hidden_dimension : 0ull,
        layer ? layer->residual_expanded_width : 0ull,
        layer ? layer->required_binding_count : 0ull,
        (unsigned long long)operation_scope};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_hex_valid(attention_plan_identity) || !layer || !output ||
        (operation_scope != YVEX_ATTENTION_OPERATION_CORE &&
         operation_scope != YVEX_ATTENTION_OPERATION_ENVELOPE))
        return activation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "sealed attention layer facts are required for activation identity");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.runtime.activation-layer.v1") ||
        !yvex_sha256_update_text(&hash, attention_plan_identity) ||
        !activation_hash_u64s(&hash, fields,
                              sizeof(fields) / sizeof(fields[0])) ||
        !yvex_sha256_final(&hash, digest))
        return activation_refuse(
            err, YVEX_ERR_STATE,
            "activation layer identity could not be derived");
    yvex_sha256_hex(digest, output);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_activation_input_seal(
    yvex_runtime_activation_input_summary *summary,
    yvex_runtime_activation_layer_record *records,
    const float *payload, yvex_error *err)
{
    char payload_digest[YVEX_SHA256_HEX_CAP];
    char input_identity[YVEX_SHA256_HEX_CAP];
    int rc;
    if (!activation_host_supported())
        return activation_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "canonical little-endian F32 activation input is unavailable");
    rc = activation_records_validate(summary, records, payload, 1, err);
    if (rc != YVEX_OK) return rc;
    if (!activation_payload_digest(payload, summary->payload_bytes,
                                   payload_digest)) {
        return activation_refuse(
            err, YVEX_ERR_STATE,
            "activation payload digest could not be derived");
    }
    yvex_runtime_identity_copy(summary->payload_digest, payload_digest);
    if (!activation_input_identity(summary, records, input_identity))
        return activation_refuse(
            err, YVEX_ERR_STATE,
            "activation input identity could not be derived");
    yvex_runtime_identity_copy(summary->input_identity, input_identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Serialize one sealed activation input.
 *
 * Never serializes native structures or padding.
 */
static int activation_serialize(
    const yvex_runtime_activation_input_summary *summary,
    const yvex_runtime_activation_layer_record *records,
    const float *payload, unsigned char **output, size_t *output_bytes,
    yvex_error *err)
{
    unsigned long long records_bytes, payload_offset, total;
    unsigned char *bytes, *cursor;
    unsigned long long index;
    if (output) *output = NULL;
    if (output_bytes) *output_bytes = 0u;
    if (!output || !output_bytes ||
        !yvex_sha256_hex_valid(summary ? summary->payload_digest : NULL) ||
        !yvex_sha256_hex_valid(summary ? summary->input_identity : NULL) ||
        activation_records_validate(summary, records, payload, 1, err) !=
            YVEX_OK ||
        !yvex_core_u64_mul(summary->layer_count, ACTIVATION_RECORD_BYTES,
                           &records_bytes) ||
        !yvex_core_u64_add(ACTIVATION_HEADER_BYTES, records_bytes,
                           &payload_offset) ||
        !yvex_core_u64_add(payload_offset, summary->payload_bytes, &total) ||
        total > (unsigned long long)SIZE_MAX)
        return yvex_error_is_set(err)
                   ? yvex_error_code(err)
                   : activation_refuse(
                         err, YVEX_ERR_BOUNDS,
                         "activation serialization extent overflowed");
    bytes = calloc((size_t)total, 1u);
    if (!bytes)
        return activation_refuse(
            err, YVEX_ERR_NOMEM,
            "activation serialization allocation failed");
    memcpy(bytes, activation_magic, sizeof(activation_magic));
    activation_put_u32(bytes + 16u, summary->schema_version);
    activation_put_u32(bytes + 20u, (uint32_t)summary->operation_scope);
    activation_put_u64(bytes + 24u, summary->token_start);
    activation_put_u64(bytes + 32u, summary->token_count);
    activation_put_u64(bytes + 40u, summary->layer_count);
    activation_put_u64(bytes + 48u, summary->payload_bytes);
    cursor = bytes + 56u;
#define PUT_IDENTITY(member)                                                   \
    do {                                                                       \
        memcpy(cursor, summary->member, ACTIVATION_IDENTITY_BYTES);            \
        cursor += ACTIVATION_IDENTITY_BYTES;                                   \
    } while (0)
    PUT_IDENTITY(logical_model_identity);
    PUT_IDENTITY(runtime_numeric_identity);
    PUT_IDENTITY(runtime_descriptor_identity);
    PUT_IDENTITY(attention_plan_identity);
    PUT_IDENTITY(payload_digest);
    PUT_IDENTITY(input_identity);
#undef PUT_IDENTITY
    cursor = bytes + ACTIVATION_HEADER_BYTES;
    for (index = 0ull; index < summary->layer_count; ++index) {
        const yvex_runtime_activation_layer_record *record = &records[index];
        activation_put_u64(cursor, record->ordinal);
        activation_put_u64(cursor + 8u, record->layer_index);
        activation_put_u64(cursor + 16u, record->width);
        activation_put_u64(cursor + 24u, record->stride);
        activation_put_u64(cursor + 32u, record->payload_offset);
        activation_put_u64(cursor + 40u, record->payload_bytes);
        memcpy(cursor + 48u, record->layer_identity,
               ACTIVATION_IDENTITY_BYTES);
        cursor += ACTIVATION_RECORD_BYTES;
    }
    memcpy(bytes + payload_offset, payload, (size_t)summary->payload_bytes);
    *output = bytes;
    *output_bytes = (size_t)total;
    return YVEX_OK;
}

int yvex_runtime_activation_input_write(
    const char *path, const yvex_runtime_activation_input_summary *summary,
    const yvex_runtime_activation_layer_record *records,
    const float *payload, yvex_error *err)
{
    yvex_core_file_result result;
    unsigned char *bytes = NULL;
    size_t count = 0u;
    int rc;
    if (!path || !path[0])
        return activation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "activation output path is required");
    rc = activation_serialize(
        summary, records, payload, &bytes, &count, err);
    if (rc == YVEX_OK)
        rc = yvex_core_file_publish_noreplace(
            path, bytes, count, NULL, NULL, NULL, &result, err);
    free(bytes);
    return rc;
}

static int activation_parse_mapping(
    yvex_runtime_activation_input *input, yvex_error *err)
{
    const unsigned char *bytes = input->mapping;
    yvex_runtime_activation_input_summary *summary = &input->summary;
    unsigned long long records_bytes, payload_offset, total, index;
    const unsigned char *cursor;
    if (input->mapping_bytes < ACTIVATION_HEADER_BYTES ||
        memcmp(bytes, activation_magic, sizeof(activation_magic)) != 0)
        return activation_refuse(
            err, YVEX_ERR_FORMAT,
            "activation tensor file magic or header is invalid");
    memset(summary, 0, sizeof(*summary));
    summary->schema_version = activation_get_u32(bytes + 16u);
    summary->operation_scope =
        (yvex_attention_operation_scope)activation_get_u32(bytes + 20u);
    summary->token_start = activation_get_u64(bytes + 24u);
    summary->token_count = activation_get_u64(bytes + 32u);
    summary->layer_count = activation_get_u64(bytes + 40u);
    summary->payload_bytes = activation_get_u64(bytes + 48u);
    cursor = bytes + 56u;
#define GET_IDENTITY(member)                                                   \
    do {                                                                       \
        activation_identity_get(summary->member, cursor);                      \
        cursor += ACTIVATION_IDENTITY_BYTES;                                   \
    } while (0)
    GET_IDENTITY(logical_model_identity);
    GET_IDENTITY(runtime_numeric_identity);
    GET_IDENTITY(runtime_descriptor_identity);
    GET_IDENTITY(attention_plan_identity);
    GET_IDENTITY(payload_digest);
    GET_IDENTITY(input_identity);
#undef GET_IDENTITY
    if (!activation_summary_valid(summary) ||
        !yvex_core_u64_mul(summary->layer_count, ACTIVATION_RECORD_BYTES,
                           &records_bytes) ||
        !yvex_core_u64_add(ACTIVATION_HEADER_BYTES, records_bytes,
                           &payload_offset) ||
        !yvex_core_u64_add(payload_offset, summary->payload_bytes, &total) ||
        total != input->mapping_bytes ||
        summary->layer_count > (unsigned long long)(SIZE_MAX / sizeof(*input->records)))
        return activation_refuse(
            err, total < input->mapping_bytes ? YVEX_ERR_FORMAT
                                              : YVEX_ERR_BOUNDS,
            total < input->mapping_bytes
                ? "activation tensor file contains trailing bytes"
                : "activation tensor file is truncated or oversized");
    input->records = calloc((size_t)summary->layer_count,
                            sizeof(*input->records));
    if (!input->records)
        return activation_refuse(
            err, YVEX_ERR_NOMEM,
            "activation record directory allocation failed");
    cursor = bytes + ACTIVATION_HEADER_BYTES;
    for (index = 0ull; index < summary->layer_count; ++index) {
        yvex_runtime_activation_layer_record *record =
            &input->records[index];
        record->ordinal = activation_get_u64(cursor);
        record->layer_index = activation_get_u64(cursor + 8u);
        record->width = activation_get_u64(cursor + 16u);
        record->stride = activation_get_u64(cursor + 24u);
        record->payload_offset = activation_get_u64(cursor + 32u);
        record->payload_bytes = activation_get_u64(cursor + 40u);
        activation_identity_get(record->layer_identity, cursor + 48u);
        cursor += ACTIVATION_RECORD_BYTES;
    }
    input->payload = (const float *)(bytes + payload_offset);
    return YVEX_OK;
}

static int activation_input_verify(
    const yvex_runtime_activation_input *input, yvex_error *err)
{
    char payload_digest[YVEX_SHA256_HEX_CAP];
    char input_identity[YVEX_SHA256_HEX_CAP];
    int rc = activation_records_validate(
        &input->summary, input->records, input->payload, 1, err);
    if (rc != YVEX_OK) return rc;
    if (!activation_payload_digest(input->payload,
                                   input->summary.payload_bytes,
                                   payload_digest) ||
        strcmp(payload_digest, input->summary.payload_digest) != 0)
        return activation_refuse(
            err, YVEX_ERR_FORMAT,
            "activation payload digest does not match payload bytes");
    if (!activation_input_identity(
            &input->summary, input->records, input_identity) ||
        strcmp(input_identity, input->summary.input_identity) != 0)
        return activation_refuse(
            err, YVEX_ERR_FORMAT,
            "activation input identity does not match canonical facts");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_activation_input_open_file(
    yvex_runtime_activation_input **out, const char *path,
    const yvex_runtime_activation_input_limits *limits, yvex_error *err)
{
    yvex_runtime_activation_input *input;
    unsigned long long maximum =
        limits ? limits->maximum_file_bytes : 0ull;
    int rc;
    if (out) *out = NULL;
    if (!out || !path || !path[0] || !maximum ||
        !activation_host_supported())
        return activation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "bounded activation tensor-file arguments are required");
    input = calloc(1u, sizeof(*input));
    if (!input)
        return activation_refuse(
            err, YVEX_ERR_NOMEM,
            "activation input owner allocation failed");
    input->fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (input->fd < 0 || fstat(input->fd, &input->snapshot) != 0 ||
        !S_ISREG(input->snapshot.st_mode) || input->snapshot.st_size <= 0) {
        rc = activation_refuse(
            err, YVEX_ERR_IO,
            "activation tensor file is not a readable regular non-symlink file");
        goto fail;
    }
    if ((unsigned long long)input->snapshot.st_size > maximum ||
        (unsigned long long)input->snapshot.st_size > (unsigned long long)SIZE_MAX) {
        rc = activation_refuse(
            err, YVEX_ERR_BOUNDS,
            "activation tensor file exceeds its resource bound");
        goto fail;
    }
    input->mapping_bytes = (size_t)input->snapshot.st_size;
    input->mapping = mmap(
        NULL, input->mapping_bytes, PROT_READ, MAP_PRIVATE, input->fd, 0);
    if (input->mapping == MAP_FAILED) {
        input->mapping = NULL;
        rc = activation_refuse(
            err, YVEX_ERR_IO,
            "activation tensor file mapping failed");
        goto fail;
    }
    input->file_backed = 1;
    rc = activation_parse_mapping(input, err);
    if (rc == YVEX_OK) rc = activation_input_verify(input, err);
    if (rc != YVEX_OK) goto fail;
    *out = input;
    return YVEX_OK;
fail:
    yvex_runtime_activation_input_close(&input);
    return rc;
}

/*
 * Admit caller-owned activation memory.
 *
 * Caller keeps payload lifetime until close.
 */
int yvex_runtime_activation_input_open_memory(
    yvex_runtime_activation_input **out,
    const yvex_runtime_activation_input_summary *summary,
    const yvex_runtime_activation_layer_record *records,
    const float *payload, yvex_error *err)
{
    yvex_runtime_activation_input *input;
    int rc;
    if (out) *out = NULL;
    if (!out || !summary || !records || !payload ||
        summary->layer_count > (unsigned long long)(SIZE_MAX / sizeof(*records)))
        return activation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "activation memory input arguments are incomplete");
    input = calloc(1u, sizeof(*input));
    if (!input)
        return activation_refuse(
            err, YVEX_ERR_NOMEM,
            "activation memory owner allocation failed");
    input->fd = -1;
    input->summary = *summary;
    input->records = calloc(
        (size_t)summary->layer_count, sizeof(*records));
    if (!input->records) {
        yvex_runtime_activation_input_close(&input);
        return activation_refuse(
            err, YVEX_ERR_NOMEM,
            "activation memory record allocation failed");
    }
    memcpy(input->records, records,
           (size_t)summary->layer_count * sizeof(*records));
    input->payload = payload;
    rc = activation_input_verify(input, err);
    if (rc != YVEX_OK) {
        yvex_runtime_activation_input_close(&input);
        return rc;
    }
    *out = input;
    return YVEX_OK;
}

/*
 * Admit input against one sealed runtime model.
 *
 * Verifies every layer identity and geometry.
 */
static int activation_input_admit(
    const yvex_runtime_activation_input *input,
    const yvex_runtime_activation_input_expectation *expectation,
    yvex_error *err)
{
    const yvex_runtime_activation_input_summary *summary =
        input ? &input->summary : NULL;
    unsigned long long index, layer_count;
    if (!input || !expectation || !expectation->attention ||
        !yvex_sha256_hex_valid(expectation->logical_model_identity) ||
        !yvex_sha256_hex_valid(expectation->runtime_numeric_identity) ||
        !yvex_sha256_hex_valid(expectation->runtime_descriptor_identity) ||
        !yvex_sha256_hex_valid(expectation->attention_plan_identity) ||
        !summary ||
        strcmp(summary->logical_model_identity,
               expectation->logical_model_identity) != 0 ||
        strcmp(summary->runtime_numeric_identity,
               expectation->runtime_numeric_identity) != 0 ||
        strcmp(summary->runtime_descriptor_identity,
               expectation->runtime_descriptor_identity) != 0 ||
        strcmp(summary->attention_plan_identity,
               expectation->attention_plan_identity) != 0 ||
        summary->operation_scope != expectation->operation_scope)
        return activation_refuse(
            err, YVEX_ERR_FORMAT,
            "activation input is incompatible with runtime identities or scope");
    layer_count = yvex_attention_plan_layer_count(expectation->attention);
    if (summary->layer_count != layer_count)
        return activation_refuse(
            err, YVEX_ERR_FORMAT,
            "activation input does not cover every admitted attention layer");
    for (index = 0ull; index < layer_count; ++index) {
        const yvex_attention_layer_plan *layer =
            yvex_attention_plan_layer_at(expectation->attention, index);
        const yvex_runtime_activation_layer_record *record =
            &input->records[index];
        char identity[YVEX_SHA256_HEX_CAP];
        unsigned long long width =
            summary->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE
                ? layer ? layer->residual_expanded_width : 0ull
                : layer ? layer->hidden_dimension : 0ull;
        if (!layer || record->layer_index != layer->layer_index ||
            record->width != width || record->stride != width ||
            yvex_runtime_activation_layer_identity_compute(
                summary->attention_plan_identity, index, layer,
                summary->operation_scope, identity, err) != YVEX_OK ||
            strcmp(record->layer_identity, identity) != 0)
            return activation_refuse(
                err, YVEX_ERR_FORMAT,
                "activation layer geometry or identity is stale");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_activation_input_validate(
    const yvex_runtime_activation_input *input, yvex_error *err)
{
    struct stat current;
    if (!input)
        return activation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "activation input is required");
    if (!input->file_backed) return activation_input_verify(input, err);
    if (input->fd < 0 || fstat(input->fd, &current) != 0 ||
        current.st_dev != input->snapshot.st_dev ||
        current.st_ino != input->snapshot.st_ino ||
        current.st_size != input->snapshot.st_size ||
        current.st_mtim.tv_sec != input->snapshot.st_mtim.tv_sec ||
        current.st_mtim.tv_nsec != input->snapshot.st_mtim.tv_nsec ||
        current.st_ctim.tv_sec != input->snapshot.st_ctim.tv_sec ||
        current.st_ctim.tv_nsec != input->snapshot.st_ctim.tv_nsec)
        return activation_refuse(
            err, YVEX_ERR_STATE,
            "activation tensor file drifted after admission");
    return activation_input_verify(input, err);
}

/*
 * Borrow the immutable typed activation summary.
 *
 * Does not transfer ownership.
 */
const yvex_runtime_activation_input_summary *
yvex_runtime_activation_input_summary_get(
    const yvex_runtime_activation_input *input)
{
    return input ? &input->summary : NULL;
}

static const yvex_runtime_activation_layer_record *
activation_input_record_at(
    const yvex_runtime_activation_input *input, unsigned long long ordinal)
{
    return input && ordinal < input->summary.layer_count
               ? &input->records[ordinal] : NULL;
}

int yvex_runtime_activation_input_view(
    const yvex_runtime_activation_input *input, unsigned long long ordinal,
    unsigned long long token_offset, unsigned long long token_count,
    const float **values, unsigned long long *stride, yvex_error *err)
{
    const yvex_runtime_activation_layer_record *record;
    unsigned long long end, element_offset, byte_end;
    if (values) *values = NULL;
    if (stride) *stride = 0ull;
    record = activation_input_record_at(input, ordinal);
    if (!record || !values || !stride || !token_count ||
        !yvex_core_u64_add(token_offset, token_count, &end) ||
        end > input->summary.token_count ||
        !yvex_core_u64_mul(token_offset, record->stride, &element_offset) ||
        !yvex_core_u64_mul(end, record->stride, &byte_end) ||
        !yvex_core_u64_mul(byte_end, sizeof(float), &byte_end) ||
        byte_end > record->payload_bytes)
        return activation_refuse(
            err, YVEX_ERR_BOUNDS,
            "activation layer token view is outside its admitted range");
    *values = input->payload + record->payload_offset / sizeof(float) +
              element_offset;
    *stride = record->stride;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Close one activation input owner.
 *
 * Unmaps, closes, releases records, and clears caller ownership.
 */
void yvex_runtime_activation_input_close(
    yvex_runtime_activation_input **input_ptr)
{
    yvex_runtime_activation_input *input;
    if (!input_ptr || !*input_ptr) return;
    input = *input_ptr;
    *input_ptr = NULL;
    free(input->records);
    if (input->mapping)
        (void)munmap(input->mapping, input->mapping_bytes);
    if (input->fd >= 0) (void)close(input->fd);
    memset(input, 0, sizeof(*input));
    free(input);
}

typedef struct {
    const yvex_runtime_activation_input *input;
    unsigned long long token_offset, token_count;
    unsigned long long view_count;
    yvex_sha256 *layer_hashes;
    unsigned long long layer_count;
} activation_prefill_context;

static int activation_prefill_view(
    void *opaque, unsigned long long layer_ordinal,
    unsigned long long token_count, const float **input,
    unsigned long long *input_stride, yvex_error *err)
{
    activation_prefill_context *context =
        (activation_prefill_context *)opaque;
    const char *injected = getenv("YVEX_TEST_RUNTIME_PREFILL_FAIL_LAYER");
    unsigned long long failure_layer = injected ? strtoull(injected, NULL, 10)
                                                : ULLONG_MAX;
    if (!context || token_count != context->token_count ||
        layer_ordinal >= context->layer_count)
        return activation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "activation prefill view request is inconsistent");
    if (injected && layer_ordinal == failure_layer)
        return activation_refuse(
            err, YVEX_ERR_STATE,
            "activation prefill layer fault was injected");
    context->view_count++;
    return yvex_runtime_activation_input_view(
        context->input, layer_ordinal, context->token_offset, token_count,
        input, input_stride, err);
}

static int activation_prefill_evidence(
    void *opaque, yvex_backend_kind backend,
    const yvex_attention_publication *publication, yvex_error *err)
{
    activation_prefill_context *context =
        (activation_prefill_context *)opaque;
    unsigned long long ordinal, width, count, index;
    const float *values;
    (void)backend;
    if (!context || !publication || !publication->complete)
        return activation_refuse(
            err, YVEX_ERR_FORMAT,
            "activation prefill publication is incomplete");
    for (ordinal = 0ull; ordinal < context->layer_count; ++ordinal)
        if (context->input->records[ordinal].layer_index ==
            publication->layer_index)
            break;
    if (ordinal == context->layer_count)
        return activation_refuse(
            err, YVEX_ERR_FORMAT,
            "activation prefill publication layer is unknown");
    values = publication->envelope_output_width
                 ? publication->envelope_output : publication->core_output;
    width = publication->envelope_output_width
                ? publication->envelope_output_width
                : publication->core_output_width;
    if (!values || !width ||
        !yvex_core_u64_mul(publication->token_count, width, &count))
        return activation_refuse(
            err, YVEX_ERR_FORMAT,
            "activation prefill output geometry is incomplete");
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        if (!isfinite(values[index]))
            return activation_refuse(
                err, YVEX_ERR_FORMAT,
                "activation prefill output is non-finite");
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(
                &context->layer_hashes[ordinal],
                (unsigned long long)bits))
            return activation_refuse(
                err, YVEX_ERR_STATE,
                "activation prefill output digest update failed");
    }
    return YVEX_OK;
}

/*
 * Allocate and initialize per-layer output hashes.
 *
 * Initialization excludes input identity and backend.
 */
static int activation_prefill_hashes_open(
    activation_prefill_context *context, yvex_error *err)
{
    unsigned long long index;
    context->layer_hashes = calloc(
        (size_t)context->layer_count, sizeof(*context->layer_hashes));
    if (!context->layer_hashes)
        return activation_refuse(
            err, YVEX_ERR_NOMEM,
            "activation prefill output hash allocation failed");
    for (index = 0ull; index < context->layer_count; ++index) {
        const yvex_runtime_activation_layer_record *record =
            &context->input->records[index];
        yvex_sha256_init(&context->layer_hashes[index]);
        if (!yvex_sha256_update_text(
                &context->layer_hashes[index],
                "yvex.runtime.activation-prefill.layer-output.v1") ||
            !yvex_sha256_update_u64(
                &context->layer_hashes[index], index) ||
            !yvex_sha256_update_u64(
                &context->layer_hashes[index], record->layer_index) ||
            !yvex_sha256_update_u64(
                &context->layer_hashes[index], record->width))
            return activation_refuse(
                err, YVEX_ERR_STATE,
                "activation prefill layer output hash initialization failed");
    }
    return YVEX_OK;
}

/*
 * Finalize ordered layer output hashes.
 *
 * Output identity is chunk/backend independent.
 */
static int activation_prefill_hashes_close(
    activation_prefill_context *context,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_sha256 aggregate;
    unsigned char layer_digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&aggregate);
    if (!yvex_sha256_update_text(
            &aggregate, "yvex.runtime.activation-prefill.output.v1"))
        goto fail;
    for (index = 0ull; index < context->layer_count; ++index)
        if (!yvex_sha256_final(
                &context->layer_hashes[index], layer_digest) ||
            !yvex_sha256_update(
                &aggregate, layer_digest, sizeof(layer_digest)))
            goto fail;
    if (!yvex_sha256_final(&aggregate, digest)) goto fail;
    yvex_sha256_hex(digest, output);
    return YVEX_OK;
fail:
    return activation_refuse(
        err, YVEX_ERR_STATE,
        "activation prefill output digest finalization failed");
}

static int activation_prefill_state_summary(
    const yvex_runtime_execution_session *session,
    yvex_graph_attention_state_summary *summary, yvex_error *err)
{
    const yvex_runtime_session_view *view =
        yvex_runtime_session_view_get(session);
    if (!view || !view->attention_state_provider ||
        !view->attention_state_provider->summary ||
        view->attention_state_provider->summary(
            view->attention_state_provider->context, summary, err) != YVEX_OK)
        return activation_refuse(
            err, YVEX_ERR_STATE,
            "activation prefill persistent state is unavailable");
    return YVEX_OK;
}

static int activation_prefill_capacity_build(
    yvex_graph_attention_capacity_plan **out,
    const yvex_runtime_model_view *model,
    unsigned long long start, unsigned long long tokens,
    yvex_error *err)
{
    yvex_graph_attention_capacity_request request;
    memset(&request, 0, sizeof(request));
    request.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
    request.history_tokens = request.start_position = start;
    request.token_count = tokens;
    request.execution_count = 1ull;
    request.use_requested_position = 1;
    return yvex_graph_attention_capacity_plan_build(
        out, model ? model->attention : NULL, &request, err);
}

static int activation_prefill_prepare(
    yvex_runtime_model *model, yvex_runtime_execution_session *session,
    const yvex_runtime_activation_prefill_request *request,
    const yvex_runtime_activation_input_summary *input,
    yvex_graph_attention_state_summary *state,
    yvex_runtime_model_failure *failure, yvex_error *err)
{
    const yvex_runtime_model_view *model_view =
        yvex_runtime_model_view_get(model);
    yvex_graph_attention_capacity_plan *state_capacity = NULL;
    yvex_graph_attention_capacity_plan *workspace_capacity = NULL;
    yvex_attention_failure graph_failure;
    unsigned long long final, state_tokens, workspace_tokens;
    int rc;
    if (!yvex_core_u64_add(input->token_start, input->token_count, &final) ||
        request->context_capacity < final)
        return activation_refuse(
            err, YVEX_ERR_BOUNDS,
            "activation prefill exceeds its declared context capacity");
    if (!state->prepared_layer_count) {
        if (input->token_start)
            return activation_refuse(
                err, YVEX_ERR_STATE,
                "nonzero activation prefill requires existing committed state");
        state_tokens = request->context_capacity;
        rc = activation_prefill_capacity_build(
            &state_capacity, model_view, 0ull, state_tokens, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_session_prepare_attention_probe_state(
                session, model, state_capacity, &graph_failure, err);
        if (rc == YVEX_OK)
            rc = activation_prefill_state_summary(session, state, err);
        yvex_graph_attention_capacity_plan_close(&state_capacity);
        if (rc != YVEX_OK) return rc;
    }
    if (state->prepared_layer_count != state->layer_count ||
        !state->position_consistent || state->next_position != input->token_start ||
        state->capacity < final)
        return activation_refuse(
            err, YVEX_ERR_STATE,
            "activation prefill position or prepared state capacity is incompatible");
    if (request->backend == YVEX_BACKEND_KIND_CPU)
        return YVEX_OK;
    {
        yvex_runtime_session_summary session_summary;
        rc = yvex_runtime_session_summary_copy(
            session, &session_summary, err);
        if (rc != YVEX_OK) return rc;
        if (session_summary.device_workspace_bytes)
            return YVEX_OK;
    }
    workspace_tokens =
        request->chunk_tokens < input->token_count
            ? request->chunk_tokens : input->token_count;
    rc = activation_prefill_capacity_build(
        &workspace_capacity, model_view, input->token_start,
        workspace_tokens, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_prepare_attention_workspace(
            session, request->mode, request->operation_scope,
            YVEX_ATTENTION_EVIDENCE_NONE, workspace_capacity, 0ull, failure, err);
    yvex_graph_attention_capacity_plan_close(&workspace_capacity);
    return rc;
}

static int activation_prefill_admit(
    yvex_runtime_model *model, yvex_runtime_execution_session *session,
    const yvex_runtime_activation_input *input,
    const yvex_runtime_activation_prefill_request *request,
    yvex_error *err)
{
    const yvex_runtime_model_view *model_view =
        yvex_runtime_model_view_get(model);
    const yvex_runtime_session_view *session_view =
        yvex_runtime_session_view_get(session);
    const yvex_runtime_binding_summary *binding =
        model_view ? model_view->binding : NULL;
    yvex_runtime_activation_input_expectation expectation;
    yvex_attention_operation_scope graph_scope;
    if (!model_view || !session_view || session_view->model != model ||
        !binding || !input || !request || !request->chunk_tokens ||
        !request->context_capacity ||
        (request->backend != YVEX_BACKEND_KIND_CPU &&
         request->backend != YVEX_BACKEND_KIND_CUDA) ||
        request->mode != YVEX_RUNTIME_MODE_EAGER ||
        (request->operation_scope != YVEX_RUNTIME_SCOPE_ATTENTION_CORE &&
         request->operation_scope != YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE) ||
        !session_view->backend ||
        yvex_backend_kind_of(session_view->backend) != request->backend)
        return activation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "activation prefill requires matching eager runtime owners");
    graph_scope =
        request->operation_scope == YVEX_RUNTIME_SCOPE_ATTENTION_CORE
            ? YVEX_ATTENTION_OPERATION_CORE
            : YVEX_ATTENTION_OPERATION_ENVELOPE;
    memset(&expectation, 0, sizeof(expectation));
    expectation.logical_model_identity = binding->logical_model_identity;
    expectation.runtime_numeric_identity = binding->runtime_numeric_identity;
    expectation.runtime_descriptor_identity =
        binding->runtime_descriptor_identity;
    expectation.attention_plan_identity = binding->attention_plan_identity;
    expectation.attention = model_view->attention;
    expectation.operation_scope = graph_scope;
    return activation_input_admit(input, &expectation, err);
}

/*
 * Derive the final prefill execution identity.
 *
 * Identity includes path policy, not local paths.
 */
static int activation_prefill_execution_identity(
    const yvex_runtime_activation_prefill_request *request,
    const yvex_runtime_activation_prefill_result *result,
    char output[YVEX_SHA256_HEX_CAP])
{
    const unsigned long long fields[] = {
        request->backend, request->mode, request->operation_scope,
        request->chunk_tokens, request->context_capacity,
        result->token_start, result->token_count, result->chunk_count,
        result->committed_prefix, result->generation_after};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.runtime.activation-prefill.execution.v1") ||
        !yvex_sha256_update_text(&hash, result->input_identity) ||
        !yvex_sha256_update_text(
            &hash, result->tensor_output_digest) ||
        !yvex_sha256_update_text(
            &hash, result->persistent_state_digest) ||
        !activation_hash_u64s(
            &hash, fields, sizeof(fields) / sizeof(fields[0])) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_runtime_activation_prefill_execute(
    yvex_runtime_model *model, yvex_runtime_execution_session *session,
    const yvex_runtime_activation_input *input,
    const yvex_runtime_activation_prefill_request *request,
    yvex_runtime_activation_prefill_result *result,
    yvex_runtime_model_failure *failure, yvex_error *err)
{
    const yvex_runtime_activation_input_summary *summary =
        yvex_runtime_activation_input_summary_get(input);
    yvex_graph_attention_state_summary before, after;
    activation_prefill_context context;
    unsigned long long offset = 0ull;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    memset(&context, 0, sizeof(context));
    rc = !result
             ? activation_refuse(
                   err, YVEX_ERR_INVALID_ARG,
                   "activation prefill result is required")
             : activation_prefill_admit(
                   model, session, input, request, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_activation_input_validate(input, err);
    if (rc == YVEX_OK)
        rc = activation_prefill_state_summary(session, &before, err);
    if (rc != YVEX_OK) return rc;
    context.input = input;
    context.layer_count = summary->layer_count;
    rc = activation_prefill_hashes_open(&context, err);
    if (rc == YVEX_OK)
        rc = activation_prefill_prepare(
            model, session, request, summary, &before, failure, err);
    result->token_start = result->position_before = summary->token_start;
    result->token_count = summary->token_count;
    result->generation_before = before.generation;
    yvex_runtime_identity_copy(
        result->input_identity, summary->input_identity);
    while (rc == YVEX_OK && offset < summary->token_count) {
        yvex_attention_execution_request execution;
        yvex_attention_probe_result chunk;
        unsigned long long remaining = summary->token_count - offset;
        unsigned long long tokens =
            remaining < request->chunk_tokens
                ? remaining : request->chunk_tokens;
        if (request->cancel_requested &&
            request->cancel_requested(request->cancel_context)) {
            rc = activation_refuse(
                err, YVEX_ERR_CANCELLED,
                "activation prefill cancelled between chunks");
            break;
        }
        memset(&execution, 0, sizeof(execution));
        context.token_offset = offset;
        context.token_count = tokens;
        context.view_count = 0ull;
        execution.backend = request->backend;
        execution.probe = YVEX_ATTENTION_PROBE_UNSPECIFIED;
        execution.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
        execution.operation_scope =
            request->operation_scope == YVEX_RUNTIME_SCOPE_ATTENTION_CORE
                ? YVEX_ATTENTION_OPERATION_CORE
                : YVEX_ATTENTION_OPERATION_ENVELOPE;
        execution.token_count = tokens;
        execution.token_position = summary->token_start + offset;
        execution.select_position = 1;
        execution.input_identity = summary->input_identity;
        execution.activation_view = activation_prefill_view;
        execution.activation_context = &context;
        execution.cancel_requested = request->cancel_requested;
        execution.cancel_context = request->cancel_context;
        execution.evidence = activation_prefill_evidence;
        execution.evidence_context = &context;
        memset(&chunk, 0, sizeof(chunk));
        rc = yvex_runtime_attention_probe_execute(
            session, model, &execution, &chunk, failure, err);
        if (rc == YVEX_OK &&
            (context.view_count != summary->layer_count ||
             chunk.layers_executed != summary->layer_count)) {
            rc = activation_refuse(
                err, YVEX_ERR_STATE,
                "activation prefill chunk did not execute every layer");
        }
        if (rc == YVEX_OK) {
            result->layers_executed += chunk.layers_executed;
            result->bindings_executed += chunk.bindings_executed;
            result->swa_layers_executed += chunk.swa_layers_executed;
            result->csa_layers_executed += chunk.csa_layers_executed;
            result->hca_layers_executed += chunk.hca_layers_executed;
            result->chunk_count++;
            offset += tokens;
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_activation_input_validate(input, err);
    if (activation_prefill_state_summary(session, &after, err) != YVEX_OK) {
        if (rc == YVEX_OK) rc = yvex_error_code(err);
        memset(&after, 0, sizeof(after));
    }
    result->committed_prefix = result->position_after = after.next_position;
    result->generation_after = after.generation;
    yvex_runtime_identity_copy(
        result->persistent_state_digest, after.state_content_identity);
    if (rc == YVEX_OK)
        rc = activation_prefill_hashes_close(
            &context, result->tensor_output_digest, err);
    if (rc == YVEX_OK &&
        (!activation_prefill_execution_identity(
             request, result, result->execution_identity) ||
         result->committed_prefix != summary->token_start +
                                         summary->token_count))
        rc = activation_refuse(
            err, YVEX_ERR_STATE,
            "activation prefill completion identity or prefix is incomplete");
    free(context.layer_hashes);
    if (rc == YVEX_OK) {
        result->completed = 1;
        if (failure) memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    return rc;
}

static int activation_prefill_operator_publish(
    const yvex_graph_attention_operator_request *request,
    const yvex_runtime_model *model,
    const yvex_runtime_execution_session *session,
    const yvex_runtime_activation_prefill_result *prefill,
    yvex_graph_attention_operator_result *result, yvex_error *err)
{
    const yvex_runtime_model_view *model_view =
        yvex_runtime_model_view_get(model);
    const yvex_runtime_session_view *session_view =
        yvex_runtime_session_view_get(session);
    const yvex_runtime_binding_summary *binding =
        model_view ? model_view->binding : NULL;
    const yvex_attention_summary *attention =
        model_view ? yvex_attention_plan_summary(model_view->attention) : NULL;
    yvex_runtime_model_summary model_summary;
    yvex_runtime_session_summary session_summary;
    yvex_graph_attention_state_summary state;
    yvex_runtime_state_residency_summary state_residency;
    if (!binding || !attention || !session_view ||
        yvex_runtime_model_summary_copy(
            model, &model_summary, err) != YVEX_OK ||
        yvex_runtime_session_summary_copy(
            session, &session_summary, err) != YVEX_OK ||
        activation_prefill_state_summary(session, &state, err) != YVEX_OK ||
        !session_view->state_residency ||
        yvex_runtime_state_residency_summary_copy(
            session_view->state_residency, &state_residency, err) != YVEX_OK)
        return activation_refuse(
            err, YVEX_ERR_STATE,
            "activation prefill operator facts are incomplete");
    yvex_core_text_copy(
        result->command, sizeof(result->command), "execute attention run");
    yvex_core_text_copy(
        result->target, sizeof(result->target), request->target);
    yvex_core_text_copy(
        result->family, sizeof(result->family),
        model_view->target_id);
    yvex_core_text_copy(
        result->backend, sizeof(result->backend),
        request->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    yvex_core_text_copy(result->scope, sizeof(result->scope), "full");
    yvex_core_text_copy(
        result->operation_scope, sizeof(result->operation_scope),
        request->operation_scope == YVEX_RUNTIME_SCOPE_ATTENTION_CORE
            ? "core" : "envelope");
    yvex_core_text_copy(result->phase, sizeof(result->phase), "prefill");
    yvex_core_text_copy(
        result->requested_mode, sizeof(result->requested_mode), "eager");
    yvex_core_text_copy(
        result->selected_mode, sizeof(result->selected_mode), "eager");
    yvex_core_text_copy(
        result->selection_reason, sizeof(result->selection_reason),
        "explicit activation-prefill eager mode");
    yvex_core_text_copy(
        result->input_class, sizeof(result->input_class),
        "typed_activation_tensor_file");
    yvex_core_text_copy(
        result->execution_class, sizeof(result->execution_class),
        "production");
    yvex_core_text_copy(
        result->weights_class, sizeof(result->weights_class),
        "admitted_external_artifact");
    yvex_core_text_copy(
        result->artifact_path, sizeof(result->artifact_path),
        request->artifact_path);
    yvex_core_text_copy(
        result->runtime_binding_path,
        sizeof(result->runtime_binding_path), request->runtime_binding_path);
#define COPY_BINDING(member)                                                   \
    yvex_runtime_identity_copy(result->member, binding->member)
    COPY_BINDING(artifact_identity);
    COPY_BINDING(materialization_identity);
    COPY_BINDING(logical_model_identity);
    COPY_BINDING(runtime_numeric_identity);
    COPY_BINDING(runtime_descriptor_identity);
    COPY_BINDING(attention_plan_identity);
    COPY_BINDING(semantic_graph_identity);
    COPY_BINDING(executable_graph_identity);
#undef COPY_BINDING
    yvex_runtime_identity_copy(
        result->runtime_binding_identity, binding->identity);
    yvex_runtime_identity_copy(
        result->runtime_model_identity,
        model_summary.runtime_model_identity);
    yvex_runtime_identity_copy(
        result->residency_identity, session_summary.residency_identity);
    yvex_runtime_identity_copy(
        result->workspace_identity, session_summary.workspace_identity);
    yvex_runtime_identity_copy(
        result->state_layout_identity, state.state_layout_identity);
    yvex_runtime_identity_copy(
        result->state_content_identity, state.state_content_identity);
    yvex_runtime_identity_copy(
        result->state_residency_identity,
        state_residency.layout_identity);
    yvex_runtime_identity_copy(
        result->activation_input_identity, prefill->input_identity);
    yvex_runtime_identity_copy(
        result->probe.tensor_output_digest,
        prefill->tensor_output_digest);
    yvex_runtime_identity_copy(
        result->probe.state_delta_digest,
        prefill->persistent_state_digest);
    yvex_runtime_identity_copy(
        result->execution_identity, prefill->execution_identity);
    yvex_runtime_identity_copy(
        result->probe.attention_execution_identity,
        prefill->execution_identity);
    result->main_layers_total = result->probe.layers_executed =
        attention->layer_count;
    result->bindings_total = result->probe.bindings_executed =
        attention->required_binding_count;
    result->probe.swa_layers_executed = attention->swa_layer_count;
    result->probe.csa_layers_executed = attention->csa_layer_count;
    result->probe.hca_layers_executed = attention->hca_layer_count;
    result->requested_token_count = prefill->token_count;
    result->execution_dispatch_count = result->prefill_chunk_count =
        prefill->chunk_count;
    result->committed_prefix = prefill->committed_prefix;
    result->state_layer_count = state.layer_count;
    result->state_prepared_layer_count = state.prepared_layer_count;
    result->state_allocated_bytes = state.allocated_bytes;
    result->state_capacity = state.capacity;
    result->state_committed_sequence_length =
        state.committed_sequence_length;
    result->state_next_position = state.next_position;
    result->state_generation = state.generation;
    result->state_residency_generation = state_residency.generation;
    result->state_device_bytes = state_residency.device_bytes;
    result->state_upload_bytes = state_residency.upload_bytes;
    result->state_upload_count = state_residency.upload_count;
    result->state_commit_count = state.commit_count;
    result->state_abort_count = state.abort_count;
    result->state_cancellation_count = state.cancellation_count;
    result->state_reset_count = state.reset_count;
    result->state_sealed = state.sealed;
    result->state_persistent = state.prepared_layer_count == state.layer_count;
    result->state_position_consistent = state.position_consistent;
    result->state_cuda_ready = state_residency.cuda_ready;
    result->state_validation_passed =
        state.sealed && !state.invalidated &&
        state.position_consistent && !state.transaction_active;
    result->resident_binding_count = session_summary.resident_binding_count;
    result->resident_encoded_bytes = session_summary.resident_encoded_bytes;
    result->host_resident_bytes = session_summary.host_resident_bytes;
    result->device_resident_bytes = session_summary.device_resident_bytes;
    result->workspace_bytes = session_summary.workspace_bytes;
    result->upload_bytes = session_summary.upload_bytes;
    result->upload_count = session_summary.upload_count;
    result->artifact_hash_passes = model_summary.artifact_hash_passes;
    result->artifact_bytes_hashed = model_summary.artifact_bytes_hashed;
    result->runtime_model_builds = model_summary.runtime_model_builds;
    result->runtime_descriptor_builds =
        model_summary.runtime_descriptor_builds;
    result->semantic_graph_builds = model_summary.semantic_graph_builds;
    result->executable_graph_builds = model_summary.executable_graph_builds;
    result->capabilities = session_summary.capabilities;
    result->attention_cuda_execution_ready =
        request->backend == YVEX_BACKEND_KIND_CUDA;
    result->activation_prefill_ready = 1;
    result->prefill_persistent_state_ready = 1;
    result->full_model_prefill_ready = 0;
    result->artifact_identity_verified = 1;
    result->production_api_available = 1;
    result->internal_live_runner_available = 1;
    result->operator_command_available = 1;
    result->end_user_generation_available = 0;
    return YVEX_OK;
}

static void activation_prefill_operator_refuse(
    yvex_graph_attention_operator_result *result, int rc,
    const yvex_error *err)
{
    yvex_status status =
        err && yvex_error_is_set(err) ? yvex_error_code(err)
                                      : (yvex_status)rc;
    yvex_core_text_copy(result->status, sizeof(result->status), "refused");
    yvex_core_text_copy(
        result->failure_code, sizeof(result->failure_code),
        yvex_status_name(status));
    yvex_core_text_copy(
        result->failure_where, sizeof(result->failure_where),
        err && yvex_error_where(err)[0] ? yvex_error_where(err)
                                        : "runtime.activation-prefill");
    yvex_core_text_copy(
        result->reason, sizeof(result->reason),
        err && yvex_error_message(err)[0]
            ? yvex_error_message(err)
            : "activation prefill refused");
}

int yvex_runtime_activation_prefill_operator_execute(
    const yvex_graph_attention_operator_request *request,
    yvex_graph_attention_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err)
{
    yvex_runtime_model_open_request model_request;
    yvex_runtime_session_open_request session_request;
    yvex_runtime_activation_input_limits limits;
    yvex_runtime_activation_prefill_request prefill_request;
    yvex_runtime_activation_prefill_result prefill;
    yvex_runtime_activation_input *input = NULL;
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_model_failure failure;
    yvex_error primary;
    const yvex_runtime_activation_input_summary *summary;
    unsigned long long final;
    int rc, cleanup_rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !retained_cleanup || *retained_cleanup ||
        !request->activation_input_path ||
        request->operator_action != YVEX_RUNTIME_OPERATOR_EXECUTE ||
        request->phase != YVEX_EXECUTION_PHASE_PREFILL ||
        request->mode != YVEX_RUNTIME_MODE_EAGER ||
        request->scope != YVEX_ATTENTION_PROBE_SCOPE_FULL ||
        request->compare_backends) {
        rc = activation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "tensor-file activation input requires full eager prefill execute");
        if (result) activation_prefill_operator_refuse(result, rc, err);
        return rc;
    }
    memset(&model_request, 0, sizeof(model_request));
    model_request.artifact_path = request->artifact_path;
    model_request.runtime_binding_path = request->runtime_binding_path;
    model_request.target_id = request->target;
    model_request.maximum_host_bytes = request->maximum_host_bytes;
    model_request.progress = request->progress;
    model_request.progress_context = request->progress_context;
    memset(&session_request, 0, sizeof(session_request));
    session_request.backend = request->backend;
    session_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.maximum_device_bytes = request->maximum_device_bytes;
    memset(&failure, 0, sizeof(failure));
    rc = yvex_runtime_cleanup_lease_acquire(
        &cleanup, &model_request, &session_request, &model, &session,
        &failure, err);
    limits.maximum_file_bytes =
        request->maximum_host_bytes ? request->maximum_host_bytes
                                    : 1ull << 30u;
    if (rc == YVEX_OK)
        rc = yvex_runtime_activation_input_open_file(
            &input, request->activation_input_path, &limits, err);
    summary = yvex_runtime_activation_input_summary_get(input);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(
             summary->token_start, summary->token_count, &final)))
        rc = activation_refuse(
            err, YVEX_ERR_BOUNDS,
            "activation prefill input range overflowed");
    if (rc == YVEX_OK) {
        memset(&prefill_request, 0, sizeof(prefill_request));
        prefill_request.backend = request->backend;
        prefill_request.mode = request->mode;
        prefill_request.operation_scope = request->operation_scope;
        prefill_request.chunk_tokens =
            request->chunk_tokens ? request->chunk_tokens
                                  : summary->token_count;
        prefill_request.context_capacity =
            request->context_capacity ? request->context_capacity : final;
        prefill_request.maximum_host_bytes = request->maximum_host_bytes;
        prefill_request.maximum_device_bytes = request->maximum_device_bytes;
        prefill_request.cancel_requested = request->cancel_requested;
        prefill_request.cancel_context = request->cancel_context;
        rc = yvex_runtime_activation_prefill_execute(
            model, session, input, &prefill_request, &prefill, &failure, err);
    }
    if (rc == YVEX_OK)
        rc = activation_prefill_operator_publish(
            request, model, session, &prefill, result, err);
    yvex_runtime_activation_input_close(&input);
    primary = err ? *err : (yvex_error){0};
    cleanup_rc = yvex_runtime_cleanup_lease_close(&cleanup, err);
    if (cleanup_rc != YVEX_OK) rc = cleanup_rc;
    else if (rc != YVEX_OK && err) *err = primary;
    if (cleanup) *retained_cleanup = cleanup;
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(
            result->status, sizeof(result->status), "complete");
        yvex_error_clear(err);
    } else {
        activation_prefill_operator_refuse(result, rc, err);
    }
    return rc;
}
