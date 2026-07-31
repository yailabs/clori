/*
 * Provide one production in-memory and external representation for MoE execution input.
 *
 * One input binds every ordered MoE layer and exact canonical little-endian payload bytes.
 * External activation files become bounded typed views, never model or tokenizer authority.
 */
#define _GNU_SOURCE
#include <yvex/internal/moe.h>

#include <yvex/internal/runtime.h>

#include <yvex/internal/core.h>

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

#define MOE_INPUT_MAGIC_BYTES 16u
#define MOE_INPUT_IDENTITY_BYTES 64u
#define MOE_INPUT_HEADER_BYTES 512u
#define MOE_INPUT_RECORD_BYTES 112u

static const unsigned char moe_input_magic[MOE_INPUT_MAGIC_BYTES] = {
    'Y', 'V', 'E', 'X', 'M', 'O', 'E', 'I', 'N', 'P', 'U', 'T', '0', '0', '1', '\0'};

struct yvex_moe_input {
    yvex_moe_input_summary summary;
    yvex_moe_input_layer_record *records;
    const float *activations;
    const unsigned int *token_ids;
    unsigned char *mapping;
    size_t mapping_bytes;
    int fd, file_backed, owns_memory;
    struct stat snapshot;
};

static int input_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.moe-input", reason);
    return status;
}

static int input_host_supported(void)
{
    const uint32_t one = 1u;
    return sizeof(float) == 4u && sizeof(unsigned int) == 4u &&
           *(const unsigned char *)&one == 1u;
}

static void input_put_u64(unsigned char *out, uint64_t value)
{
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        out[index] = (unsigned char)(value >> (index * 8u));
}

static uint64_t input_get_u64(const unsigned char *input)
{
    uint64_t value = 0ull;
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        value |= (uint64_t)input[index] << (index * 8u);
    return value;
}

static void input_get_identity(char output[YVEX_SHA256_HEX_CAP],
                               const unsigned char *input)
{
    memcpy(output, input, MOE_INPUT_IDENTITY_BYTES);
    output[MOE_INPUT_IDENTITY_BYTES] = '\0';
}

static int input_payload_digest(const void *payload, unsigned long long bytes,
                                char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if ((!payload && bytes) || bytes > (unsigned long long)SIZE_MAX) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, payload, (size_t)bytes) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int input_hash_fields(yvex_sha256 *hash, const unsigned long long *fields,
                             size_t count)
{
    size_t index;
    for (index = 0u; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, fields[index])) return 0;
    return 1;
}

/*
 * Derive the path-independent MoE input identity.
 *
 * Writes identity.
 */
static int input_identity(const yvex_moe_input_summary *summary,
                          const yvex_moe_input_layer_record *records,
                          char output[YVEX_SHA256_HEX_CAP])
{
    const unsigned long long fields[] = {
        summary->schema_version, summary->token_start, summary->token_count,
        summary->layer_count, summary->activation_payload_bytes,
        summary->token_id_payload_bytes};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.moe-input.v1") ||
        !input_hash_fields(&hash, fields, sizeof(fields) / sizeof(fields[0])) ||
        !yvex_sha256_update_text(&hash, summary->logical_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->runtime_numeric_identity) ||
        !yvex_sha256_update_text(&hash, summary->runtime_descriptor_identity) ||
        !yvex_sha256_update_text(&hash, summary->moe_plan_identity) ||
        !yvex_sha256_update_text(&hash, summary->activation_payload_digest) ||
        !yvex_sha256_update_text(&hash, summary->token_id_payload_digest)) return 0;
    for (index = 0ull; index < summary->layer_count; ++index) {
        const yvex_moe_input_layer_record *record = &records[index];
        const unsigned long long row[] = {
            record->ordinal, record->layer_index, record->width, record->stride,
            record->payload_offset, record->payload_bytes};
        if (!input_hash_fields(&hash, row, sizeof(row) / sizeof(row[0])) ||
            !yvex_sha256_update_text(&hash, record->layer_identity)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int input_summary_valid(const yvex_moe_input_summary *summary)
{
    return summary && summary->schema_version == YVEX_MOE_INPUT_SCHEMA_V1 &&
           summary->token_count && summary->layer_count &&
           summary->activation_payload_bytes && summary->token_id_payload_bytes &&
           yvex_sha256_hex_valid(summary->logical_model_identity) &&
           yvex_sha256_hex_valid(summary->runtime_numeric_identity) &&
           yvex_sha256_hex_valid(summary->runtime_descriptor_identity) &&
           yvex_sha256_hex_valid(summary->moe_plan_identity);
}

static int input_records_valid(const yvex_moe_input_summary *summary,
                               const yvex_moe_input_layer_record *records,
                               const float *activations, const unsigned int *token_ids,
                               int identities, yvex_error *err)
{
    unsigned long long index, offset = 0ull, token_bytes;
    if (!input_summary_valid(summary) || !records || !activations || !token_ids ||
        !yvex_core_u64_mul(summary->token_count, sizeof(*token_ids), &token_bytes) ||
        token_bytes != summary->token_id_payload_bytes)
        return input_refuse(err, YVEX_ERR_INVALID_ARG,
                            "complete canonical MoE input facts are required");
    for (index = 0ull; index < summary->layer_count; ++index) {
        const yvex_moe_input_layer_record *record = &records[index];
        unsigned long long elements, bytes, prior;
        if (record->ordinal != index || record->stride != record->width || !record->width ||
            record->payload_offset != offset ||
            !yvex_core_u64_mul(summary->token_count, record->stride, &elements) ||
            !yvex_core_u64_mul(elements, sizeof(float), &bytes) ||
            record->payload_bytes != bytes || !yvex_core_u64_add(offset, bytes, &offset) ||
            (identities && !yvex_sha256_hex_valid(record->layer_identity)))
            return input_refuse(err, YVEX_ERR_FORMAT,
                                "MoE input layer ordering or range is malformed");
        for (prior = 0ull; prior < index; ++prior)
            if (records[prior].layer_index == record->layer_index)
                return input_refuse(err, YVEX_ERR_FORMAT,
                                    "MoE input contains a duplicate layer");
    }
    if (offset != summary->activation_payload_bytes)
        return input_refuse(err, YVEX_ERR_FORMAT,
                            "MoE activation ranges are not exact and contiguous");
    for (index = 0ull; index < offset / sizeof(float); ++index)
        if (!isfinite(activations[index]))
            return input_refuse(err, YVEX_ERR_FORMAT,
                                "MoE input contains a non-finite activation");
    return YVEX_OK;
}

int yvex_moe_input_seal(yvex_moe_input_summary *summary,
                        yvex_moe_input_layer_record *records,
                        const float *activations, const unsigned int *token_ids,
                        yvex_error *err)
{
    char activations_digest[YVEX_SHA256_HEX_CAP];
    char tokens_digest[YVEX_SHA256_HEX_CAP], identity[YVEX_SHA256_HEX_CAP];
    int rc;
    if (!input_host_supported())
        return input_refuse(err, YVEX_ERR_UNSUPPORTED,
                            "canonical MoE scalar encoding is unavailable");
    rc = input_records_valid(summary, records, activations, token_ids, 1, err);
    if (rc != YVEX_OK) return rc;
    if (!input_payload_digest(activations, summary->activation_payload_bytes,
                              activations_digest) ||
        !input_payload_digest(token_ids, summary->token_id_payload_bytes, tokens_digest))
        return input_refuse(err, YVEX_ERR_STATE, "MoE payload digest derivation failed");
    yvex_runtime_identity_copy(summary->activation_payload_digest, activations_digest);
    yvex_runtime_identity_copy(summary->token_id_payload_digest, tokens_digest);
    if (!input_identity(summary, records, identity))
        return input_refuse(err, YVEX_ERR_STATE, "MoE input identity derivation failed");
    yvex_runtime_identity_copy(summary->input_identity, identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int input_serialize(const yvex_moe_input_summary *summary,
                           const yvex_moe_input_layer_record *records,
                           const float *activations, const unsigned int *token_ids,
                           unsigned char **out, size_t *out_bytes, yvex_error *err)
{
    unsigned long long records_bytes, activation_offset, token_offset, total, index;
    unsigned char *bytes, *cursor;
    if (out) *out = NULL;
    if (out_bytes) *out_bytes = 0u;
    if (!out || !out_bytes || !yvex_sha256_hex_valid(summary ? summary->input_identity : NULL) ||
        input_records_valid(summary, records, activations, token_ids, 1, err) != YVEX_OK ||
        !yvex_core_u64_mul(summary->layer_count, MOE_INPUT_RECORD_BYTES, &records_bytes) ||
        !yvex_core_u64_add(MOE_INPUT_HEADER_BYTES, records_bytes, &activation_offset) ||
        !yvex_core_u64_add(activation_offset, summary->activation_payload_bytes, &token_offset) ||
        !yvex_core_u64_add(token_offset, summary->token_id_payload_bytes, &total) ||
        total > (unsigned long long)SIZE_MAX)
        return yvex_error_is_set(err) ? yvex_error_code(err)
                                      : input_refuse(err, YVEX_ERR_BOUNDS,
                                                     "MoE input extent overflowed");
    bytes = (unsigned char *)calloc((size_t)total, 1u);
    if (!bytes) return input_refuse(err, YVEX_ERR_NOMEM, "MoE file allocation failed");
    memcpy(bytes, moe_input_magic, sizeof(moe_input_magic));
    input_put_u64(bytes + 16u, summary->schema_version);
    input_put_u64(bytes + 24u, summary->token_start);
    input_put_u64(bytes + 32u, summary->token_count);
    input_put_u64(bytes + 40u, summary->layer_count);
    input_put_u64(bytes + 48u, summary->activation_payload_bytes);
    input_put_u64(bytes + 56u, summary->token_id_payload_bytes);
    cursor = bytes + 64u;
#define PUT_ID(member_) do { memcpy(cursor, summary->member_, 64u); cursor += 64u; } while (0)
    PUT_ID(logical_model_identity); PUT_ID(runtime_numeric_identity);
    PUT_ID(runtime_descriptor_identity); PUT_ID(moe_plan_identity);
    PUT_ID(activation_payload_digest); PUT_ID(token_id_payload_digest); PUT_ID(input_identity);
#undef PUT_ID
    cursor = bytes + MOE_INPUT_HEADER_BYTES;
    for (index = 0ull; index < summary->layer_count; ++index) {
        const yvex_moe_input_layer_record *record = &records[index];
        input_put_u64(cursor, record->ordinal); input_put_u64(cursor + 8u, record->layer_index);
        input_put_u64(cursor + 16u, record->width); input_put_u64(cursor + 24u, record->stride);
        input_put_u64(cursor + 32u, record->payload_offset);
        input_put_u64(cursor + 40u, record->payload_bytes);
        memcpy(cursor + 48u, record->layer_identity, 64u);
        cursor += MOE_INPUT_RECORD_BYTES;
    }
    memcpy(bytes + activation_offset, activations, (size_t)summary->activation_payload_bytes);
    memcpy(bytes + token_offset, token_ids, (size_t)summary->token_id_payload_bytes);
    *out = bytes;
    *out_bytes = (size_t)total;
    return YVEX_OK;
}

/* Transactionally publish one sealed MoE tensor file without replacement. */
int yvex_moe_input_write(const char *path, const yvex_moe_input_summary *summary,
                         const yvex_moe_input_layer_record *records,
                         const float *activations, const unsigned int *token_ids,
                         yvex_error *err)
{
    yvex_core_file_result result;
    unsigned char *bytes = NULL;
    size_t count = 0u;
    int rc;
    if (!path || !path[0])
        return input_refuse(err, YVEX_ERR_INVALID_ARG, "MoE input output path is required");
    rc = input_serialize(summary, records, activations, token_ids, &bytes, &count, err);
    if (rc == YVEX_OK)
        rc = yvex_core_file_publish_noreplace(path, bytes, count, NULL, NULL, NULL, &result, err);
    free(bytes);
    return rc;
}

static int input_parse(yvex_moe_input *input, yvex_error *err)
{
    yvex_moe_input_summary *summary = &input->summary;
    const unsigned char *bytes = input->mapping, *cursor;
    unsigned long long records_bytes, activation_offset, token_offset, total, index;
    if (!bytes || input->mapping_bytes < MOE_INPUT_HEADER_BYTES ||
        memcmp(bytes, moe_input_magic, sizeof(moe_input_magic)) != 0)
        return input_refuse(err, YVEX_ERR_FORMAT, "MoE input magic or header is invalid");
    memset(summary, 0, sizeof(*summary));
    summary->schema_version = (unsigned int)input_get_u64(bytes + 16u);
    summary->token_start = input_get_u64(bytes + 24u);
    summary->token_count = input_get_u64(bytes + 32u);
    summary->layer_count = input_get_u64(bytes + 40u);
    summary->activation_payload_bytes = input_get_u64(bytes + 48u);
    summary->token_id_payload_bytes = input_get_u64(bytes + 56u);
    cursor = bytes + 64u;
#define GET_ID(member_) do { input_get_identity(summary->member_, cursor); cursor += 64u; } while (0)
    GET_ID(logical_model_identity); GET_ID(runtime_numeric_identity);
    GET_ID(runtime_descriptor_identity); GET_ID(moe_plan_identity);
    GET_ID(activation_payload_digest); GET_ID(token_id_payload_digest); GET_ID(input_identity);
#undef GET_ID
    if (!input_summary_valid(summary) || summary->layer_count > 1024ull ||
        !yvex_core_u64_mul(summary->layer_count, MOE_INPUT_RECORD_BYTES, &records_bytes) ||
        !yvex_core_u64_add(MOE_INPUT_HEADER_BYTES, records_bytes, &activation_offset) ||
        !yvex_core_u64_add(activation_offset, summary->activation_payload_bytes, &token_offset) ||
        !yvex_core_u64_add(token_offset, summary->token_id_payload_bytes, &total) ||
        total != input->mapping_bytes ||
        summary->layer_count > (unsigned long long)(SIZE_MAX / sizeof(*input->records)))
        return input_refuse(err, total < input->mapping_bytes ? YVEX_ERR_FORMAT : YVEX_ERR_BOUNDS,
                            total < input->mapping_bytes ? "MoE input contains trailing bytes"
                                                       : "MoE input is truncated or oversized");
    input->records = (yvex_moe_input_layer_record *)calloc(
        (size_t)summary->layer_count, sizeof(*input->records));
    if (!input->records)
        return input_refuse(err, YVEX_ERR_NOMEM, "MoE input record allocation failed");
    cursor = bytes + MOE_INPUT_HEADER_BYTES;
    for (index = 0ull; index < summary->layer_count; ++index) {
        yvex_moe_input_layer_record *record = &input->records[index];
        record->ordinal = input_get_u64(cursor);
        record->layer_index = input_get_u64(cursor + 8u);
        record->width = input_get_u64(cursor + 16u);
        record->stride = input_get_u64(cursor + 24u);
        record->payload_offset = input_get_u64(cursor + 32u);
        record->payload_bytes = input_get_u64(cursor + 40u);
        input_get_identity(record->layer_identity, cursor + 48u);
        cursor += MOE_INPUT_RECORD_BYTES;
    }
    input->activations = (const float *)(bytes + activation_offset);
    input->token_ids = (const unsigned int *)(bytes + token_offset);
    return YVEX_OK;
}

static int input_verify(const yvex_moe_input *input, yvex_error *err)
{
    char activation_digest[YVEX_SHA256_HEX_CAP];
    char token_digest[YVEX_SHA256_HEX_CAP], identity[YVEX_SHA256_HEX_CAP];
    int rc = input_records_valid(&input->summary, input->records, input->activations,
                                 input->token_ids, 1, err);
    if (rc != YVEX_OK) return rc;
    if (!input_payload_digest(input->activations, input->summary.activation_payload_bytes,
                              activation_digest) ||
        strcmp(activation_digest, input->summary.activation_payload_digest) != 0 ||
        !input_payload_digest(input->token_ids, input->summary.token_id_payload_bytes,
                              token_digest) ||
        strcmp(token_digest, input->summary.token_id_payload_digest) != 0)
        return input_refuse(err, YVEX_ERR_FORMAT, "MoE payload digest mismatch");
    if (!input_identity(&input->summary, input->records, identity) ||
        strcmp(identity, input->summary.input_identity) != 0)
        return input_refuse(err, YVEX_ERR_FORMAT, "MoE input identity mismatch");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int input_snapshot_valid(const yvex_moe_input *input)
{
    struct stat current;
    return !input->file_backed ||
           (fstat(input->fd, &current) == 0 && current.st_dev == input->snapshot.st_dev &&
            current.st_ino == input->snapshot.st_ino && current.st_mode == input->snapshot.st_mode &&
            current.st_size == input->snapshot.st_size &&
            current.st_mtim.tv_sec == input->snapshot.st_mtim.tv_sec &&
            current.st_mtim.tv_nsec == input->snapshot.st_mtim.tv_nsec &&
            current.st_ctim.tv_sec == input->snapshot.st_ctim.tv_sec &&
            current.st_ctim.tv_nsec == input->snapshot.st_ctim.tv_nsec);
}

int yvex_moe_input_open_file(yvex_moe_input **out, const char *path,
                             const yvex_moe_input_limits *limits, yvex_error *err)
{
    yvex_moe_input *input;
    unsigned long long maximum = limits ? limits->maximum_file_bytes : 0ull;
    int rc;
    if (out) *out = NULL;
    if (!out || !path || !path[0] || !maximum || !input_host_supported())
        return input_refuse(err, YVEX_ERR_INVALID_ARG,
                            "bounded MoE tensor-file arguments are required");
    input = (yvex_moe_input *)calloc(1u, sizeof(*input));
    if (!input) return input_refuse(err, YVEX_ERR_NOMEM, "MoE input allocation failed");
    input->fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (input->fd < 0 || fstat(input->fd, &input->snapshot) != 0 ||
        !S_ISREG(input->snapshot.st_mode) || input->snapshot.st_size <= 0) {
        rc = input_refuse(err, YVEX_ERR_IO,
                          "MoE input is not a readable regular non-symlink file");
        goto fail;
    }
    if ((unsigned long long)input->snapshot.st_size > maximum ||
        (unsigned long long)input->snapshot.st_size > (unsigned long long)SIZE_MAX) {
        rc = input_refuse(err, YVEX_ERR_BOUNDS, "MoE input exceeds its resource bound");
        goto fail;
    }
    input->mapping_bytes = (size_t)input->snapshot.st_size;
    input->mapping = (unsigned char *)mmap(NULL, input->mapping_bytes, PROT_READ,
                                           MAP_PRIVATE, input->fd, 0);
    if (input->mapping == MAP_FAILED) {
        input->mapping = NULL;
        rc = input_refuse(err, YVEX_ERR_IO, "MoE input mapping failed");
        goto fail;
    }
    input->file_backed = 1;
    rc = input_parse(input, err);
    if (rc == YVEX_OK) rc = input_verify(input, err);
    if (rc == YVEX_OK && !input_snapshot_valid(input))
        rc = input_refuse(err, YVEX_ERR_STATE, "MoE input drifted during admission");
    if (rc != YVEX_OK) goto fail;
    *out = input;
    return YVEX_OK;
fail:
    yvex_moe_input_close(&input);
    return rc;
}

/*
 * Copy caller-owned canonical memory into one immutable input owner.
 *
 * No file identity.
 */
int yvex_moe_input_open_memory(yvex_moe_input **out, const yvex_moe_input_summary *summary,
                               const yvex_moe_input_layer_record *records,
                               const float *activations, const unsigned int *token_ids,
                               yvex_error *err)
{
    yvex_moe_input *input;
    unsigned long long total;
    int rc;
    if (out) *out = NULL;
    if (!out || input_records_valid(summary, records, activations, token_ids, 1, err) != YVEX_OK ||
        !yvex_core_u64_add(summary->activation_payload_bytes, summary->token_id_payload_bytes,
                           &total) || total > (unsigned long long)SIZE_MAX)
        return yvex_error_is_set(err) ? yvex_error_code(err)
                                      : input_refuse(err, YVEX_ERR_BOUNDS,
                                                     "MoE memory input extent overflowed");
    input = (yvex_moe_input *)calloc(1u, sizeof(*input));
    if (!input) return input_refuse(err, YVEX_ERR_NOMEM, "MoE input allocation failed");
    input->fd = -1;
    input->summary = *summary;
    input->records = (yvex_moe_input_layer_record *)malloc(
        (size_t)summary->layer_count * sizeof(*records));
    input->mapping = (unsigned char *)malloc((size_t)total);
    if (!input->records || !input->mapping) {
        rc = input_refuse(err, YVEX_ERR_NOMEM, "MoE memory input copy failed");
        goto fail;
    }
    memcpy(input->records, records, (size_t)summary->layer_count * sizeof(*records));
    memcpy(input->mapping, activations, (size_t)summary->activation_payload_bytes);
    memcpy(input->mapping + summary->activation_payload_bytes, token_ids,
           (size_t)summary->token_id_payload_bytes);
    input->mapping_bytes = (size_t)total;
    input->activations = (const float *)input->mapping;
    input->token_ids = (const unsigned int *)(input->mapping + summary->activation_payload_bytes);
    input->owns_memory = 1;
    rc = input_verify(input, err);
    if (rc != YVEX_OK) goto fail;
    *out = input;
    return YVEX_OK;
fail:
    yvex_moe_input_close(&input);
    return rc;
}

/*
 * Admit one open input against the exact plan and binding identity chain.
 *
 * Runtime identity chain.
 */
int yvex_moe_input_validate(const yvex_moe_input *input, const yvex_moe_plan *plan,
                            const yvex_runtime_binding_summary *binding, yvex_error *err)
{
    const yvex_moe_plan_summary *summary = yvex_moe_plan_summary_get(plan);
    unsigned long long index, vocabulary = 0ull;
    if (!input || !summary || !binding || !input_snapshot_valid(input) ||
        input_verify(input, err) != YVEX_OK || input->summary.layer_count != summary->layer_count ||
        strcmp(input->summary.logical_model_identity, summary->logical_model_identity) != 0 ||
        strcmp(input->summary.runtime_numeric_identity, summary->runtime_numeric_identity) != 0 ||
        strcmp(input->summary.runtime_descriptor_identity,
               summary->runtime_descriptor_identity) != 0 ||
        strcmp(input->summary.moe_plan_identity, summary->moe_plan_identity) != 0 ||
        strcmp(binding->moe_plan_identity, summary->moe_plan_identity) != 0)
        return yvex_error_is_set(err) ? yvex_error_code(err)
                                      : input_refuse(err, YVEX_ERR_STATE,
                                                     "MoE input identity chain is stale");
    for (index = 0ull; index < summary->layer_count; ++index) {
        const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(plan, index);
        const yvex_moe_input_layer_record *record = &input->records[index];
        if (!layer || record->layer_index != layer->layer_index ||
            record->width != layer->expanded_width || record->stride != layer->expanded_width ||
            strcmp(record->layer_identity, layer->layer_identity) != 0)
            return input_refuse(err, YVEX_ERR_FORMAT,
                                "MoE input layer geometry or identity is incompatible");
        if (layer->requires_token_ids && layer->hash_table_rows > vocabulary)
            vocabulary = layer->hash_table_rows;
    }
    for (index = 0ull; index < input->summary.token_count; ++index)
        if ((unsigned long long)input->token_ids[index] >= vocabulary)
            return input_refuse(err, YVEX_ERR_BOUNDS,
                                "MoE input token ID exceeds the admitted vocabulary");
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Borrow immutable input summary facts.
 *
 * Borrowed owner lifetime.
 */
const yvex_moe_input_summary *yvex_moe_input_summary_get(const yvex_moe_input *input)
{
    return input ? &input->summary : NULL;
}

int yvex_moe_input_layer_view(const yvex_moe_input *input, unsigned long long ordinal,
                              const float **values, unsigned long long *stride,
                              yvex_error *err)
{
    const yvex_moe_input_layer_record *record;
    if (values) *values = NULL;
    if (stride) *stride = 0ull;
    if (!input || !values || !stride || ordinal >= input->summary.layer_count ||
        !input_snapshot_valid(input))
        return input_refuse(err, YVEX_ERR_BOUNDS, "MoE input layer view is unavailable");
    record = &input->records[ordinal];
    *values = input->activations + record->payload_offset / sizeof(float);
    *stride = record->stride;
    yvex_error_clear(err);
    return YVEX_OK;
}

const unsigned int *yvex_moe_input_token_ids(const yvex_moe_input *input)
{
    return input ? input->token_ids : NULL;
}

void yvex_moe_input_close(yvex_moe_input **input)
{
    if (!input || !*input) return;
    if ((*input)->file_backed && (*input)->mapping)
        (void)munmap((*input)->mapping, (*input)->mapping_bytes);
    else if ((*input)->owns_memory)
        free((*input)->mapping);
    if ((*input)->fd >= 0) (void)close((*input)->fd);
    free((*input)->records);
    free(*input);
    *input = NULL;
}
