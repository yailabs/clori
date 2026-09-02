/*
 * Derive sealed GGUF structure and exact physical tensor ranges.
 *
 * Plan bytes are explicit little-endian GGUF v3; qtype geometry is consumed from its canonical
 * owner; planning reads zero tensor payload. Sealed structure and predicted ranges do not prove
 * file emission.
 */
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact_lowering.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/gguf_writer.h>

#define WRITER_GGUF_VERSION 3u
#define WRITER_DEFAULT_ALIGNMENT 32u
#define WRITER_DEFAULT_BUDGET (128u * 1024u * 1024u)
#define WRITER_METADATA_CAP 96u

typedef enum {
    WRITER_META_SCALAR = 0,
    WRITER_META_TOKEN_ARRAY,
    WRITER_META_TOKEN_TYPE_ARRAY,
    WRITER_META_MERGE_ARRAY,
    WRITER_META_U64_ARRAY,
    WRITER_META_F64_ARRAY
} writer_metadata_source;

typedef struct {
    char key[128];
    unsigned int type;
    writer_metadata_source source;
    const unsigned char *string_bytes;
    size_t string_length;
    unsigned long long u64;
    double f64;
    int boolean;
    unsigned int element_type;
    unsigned long long count;
    const yvex_gguf_writer_lowering_metadata *map_entry;
} writer_metadata;

typedef yvex_core_bytes writer_buffer;

struct yvex_gguf_writer_plan {
    yvex_gguf_writer_plan_summary summary;
    const yvex_quant_plan *quant_plan;
    yvex_gguf_tokenizer_metadata *tokenizer;
    yvex_gguf_writer_tensor *tensors;
    unsigned char *prefix;
    size_t prefix_bytes;
};

static int writer_fail(yvex_gguf_writer_failure *failure, yvex_gguf_writer_code code,
                       const char *name, unsigned long long metadata_index,
                       unsigned long long tensor_index, unsigned long long expected,
                       unsigned long long actual, yvex_error *err, yvex_status status,
                       const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->metadata_index = metadata_index;
        failure->tensor_index = tensor_index;
        failure->expected = expected;
        failure->actual = actual;
        if (name)
            yvex_core_text_copy(failure->name, sizeof(failure->name), name);
    }
    yvex_error_set(err, status, "gguf.writer.plan", message);
    return status;
}

static int writer_align(unsigned long long value, unsigned int alignment, unsigned long long *out) {
    unsigned long long mask;
    if (!out || !alignment || (alignment & (alignment - 1u)) != 0u)
        return 0;
    mask = alignment - 1u;
    if (value > ULLONG_MAX - mask)
        return 0;
    *out = (value + mask) & ~mask;
    return 1;
}

static int writer_tensor_name_compare(const void *left, const void *right) {
    const yvex_gguf_writer_tensor *const *left_tensor =
        (const yvex_gguf_writer_tensor *const *)left;
    const yvex_gguf_writer_tensor *const *right_tensor =
        (const yvex_gguf_writer_tensor *const *)right;
    return strcmp((*left_tensor)->name, (*right_tensor)->name);
}

static int writer_tensor_names_unique(const yvex_gguf_writer_tensor *tensors,
                                      unsigned long long tensor_count) {
    const yvex_gguf_writer_tensor **ordered;
    unsigned long long index;
    int result = 1;
    if (!tensors || tensor_count > SIZE_MAX / sizeof(*ordered))
        return -1;
    ordered = (const yvex_gguf_writer_tensor **)malloc((size_t)tensor_count * sizeof(*ordered));
    if (!ordered)
        return -1;
    for (index = 0u; index < tensor_count; ++index)
        ordered[index] = &tensors[index];
    qsort(ordered, (size_t)tensor_count, sizeof(*ordered), writer_tensor_name_compare);
    for (index = 1u; index < tensor_count; ++index)
        if (strcmp(ordered[index - 1u]->name, ordered[index]->name) == 0) {
            result = 0;
            break;
        }
    free(ordered);
    return result;
}

static int writer_u32(writer_buffer *buffer, unsigned int value) {
    unsigned char bytes[4];
    unsigned int index;
    for (index = 0u; index < 4u; ++index)
        bytes[index] = (unsigned char)(value >> (index * 8u));
    return yvex_core_bytes_append(buffer, bytes, sizeof(bytes));
}

static int writer_u64(writer_buffer *buffer, unsigned long long value) {
    unsigned char bytes[8];
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        bytes[index] = (unsigned char)(value >> (index * 8u));
    return yvex_core_bytes_append(buffer, bytes, sizeof(bytes));
}

static int writer_f32(writer_buffer *buffer, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return writer_u32(buffer, bits);
}

static int writer_f64(writer_buffer *buffer, double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return writer_u64(buffer, bits);
}

static int writer_string(writer_buffer *buffer, const unsigned char *bytes, size_t byte_count) {
    return writer_u64(buffer, byte_count) &&
           yvex_core_bytes_append(buffer, bytes, byte_count);
}

static int writer_metadata_key_exists(const writer_metadata *entries, unsigned int count,
                                      const char *key) {
    unsigned int index;
    for (index = 0u; index < count; ++index)
        if (strcmp(entries[index].key, key) == 0)
            return 1;
    return 0;
}

static writer_metadata *writer_metadata_new(writer_metadata *entries, unsigned int *count,
                                            const char *key) {
    writer_metadata *entry;
    if (!entries || !count || !key || !key[0] || strlen(key) >= sizeof(entries[0].key) ||
        *count >= WRITER_METADATA_CAP || writer_metadata_key_exists(entries, *count, key))
        return NULL;
    entry = &entries[(*count)++];
    memset(entry, 0, sizeof(*entry));
    yvex_core_text_copy(entry->key, sizeof(entry->key), key);
    return entry;
}

static int writer_meta_string(writer_metadata *entries, unsigned int *count, const char *key,
                              const unsigned char *bytes, size_t byte_count) {
    writer_metadata *entry = writer_metadata_new(entries, count, key);
    if (!entry || (!bytes && byte_count))
        return 0;
    entry->type = YVEX_GGUF_VALUE_STRING;
    entry->string_bytes = bytes;
    entry->string_length = byte_count;
    return 1;
}

static int writer_meta_text(writer_metadata *entries, unsigned int *count, const char *key,
                            const char *text) {
    return text &&
           writer_meta_string(entries, count, key, (const unsigned char *)text, strlen(text));
}

static int writer_meta_u32(writer_metadata *entries, unsigned int *count, const char *key,
                           unsigned long long value) {
    writer_metadata *entry;
    if (value > UINT_MAX)
        return 0;
    entry = writer_metadata_new(entries, count, key);
    if (!entry)
        return 0;
    entry->type = YVEX_GGUF_VALUE_UINT32;
    entry->u64 = value;
    return 1;
}

static int writer_meta_bool(writer_metadata *entries, unsigned int *count, const char *key,
                            int value) {
    writer_metadata *entry = writer_metadata_new(entries, count, key);
    if (!entry)
        return 0;
    entry->type = YVEX_GGUF_VALUE_BOOL;
    entry->boolean = value != 0;
    return 1;
}

static int writer_meta_map(writer_metadata *entries, unsigned int *count,
                           const yvex_gguf_writer_lowering_metadata *map_entry) {
    writer_metadata *entry;
    int custom;
    if (!map_entry)
        return 0;
    entry = writer_metadata_new(entries, count, map_entry->key);
    if (!entry)
        return 0;
    entry->map_entry = map_entry;
    custom = strncmp(map_entry->key, "yvex.", 5u) == 0;
    switch (map_entry->type) {
    case YVEX_GGUF_WRITER_LOWERING_METADATA_STRING:
        entry->type = YVEX_GGUF_VALUE_STRING;
        entry->string_bytes = (const unsigned char *)map_entry->string_value;
        entry->string_length = strlen(map_entry->string_value);
        return 1;
    case YVEX_GGUF_WRITER_LOWERING_METADATA_U64:
        if (!custom && map_entry->u64_value > UINT_MAX)
            return 0;
        entry->type = custom ? YVEX_GGUF_VALUE_UINT64 : YVEX_GGUF_VALUE_UINT32;
        entry->u64 = map_entry->u64_value;
        return 1;
    case YVEX_GGUF_WRITER_LOWERING_METADATA_F64:
        if (!isfinite(map_entry->f64_value) || fabs(map_entry->f64_value) > FLT_MAX)
            return 0;
        entry->type = custom ? YVEX_GGUF_VALUE_FLOAT64 : YVEX_GGUF_VALUE_FLOAT32;
        entry->f64 = map_entry->f64_value;
        return 1;
    case YVEX_GGUF_WRITER_LOWERING_METADATA_BOOL:
        entry->type = YVEX_GGUF_VALUE_BOOL;
        entry->boolean = map_entry->bool_value != 0;
        return 1;
    case YVEX_GGUF_WRITER_LOWERING_METADATA_U64_ARRAY:
        entry->type = YVEX_GGUF_VALUE_ARRAY;
        entry->source = WRITER_META_U64_ARRAY;
        entry->element_type = custom ? YVEX_GGUF_VALUE_UINT64 : YVEX_GGUF_VALUE_UINT32;
        entry->count = map_entry->array_count;
        return entry->count != 0u;
    case YVEX_GGUF_WRITER_LOWERING_METADATA_F64_ARRAY:
        entry->type = YVEX_GGUF_VALUE_ARRAY;
        entry->source = WRITER_META_F64_ARRAY;
        entry->element_type = custom ? YVEX_GGUF_VALUE_FLOAT64 : YVEX_GGUF_VALUE_FLOAT32;
        entry->count = map_entry->array_count;
        return entry->count != 0u;
    default:
        return 0;
    }
}

static int writer_meta_dynamic_array(writer_metadata *entries, unsigned int *count, const char *key,
                                     writer_metadata_source source, unsigned int element_type,
                                     unsigned long long element_count) {
    writer_metadata *entry = writer_metadata_new(entries, count, key);
    if (!entry || !element_count)
        return 0;
    entry->type = YVEX_GGUF_VALUE_ARRAY;
    entry->source = source;
    entry->element_type = element_type;
    entry->count = element_count;
    return 1;
}

static int writer_tokenizer_metadata_add(
    writer_metadata *metadata, unsigned int *count,
    const yvex_gguf_tokenizer_summary *tokenizer, const unsigned char *raw_json,
    size_t raw_json_bytes, const unsigned char *raw_config, size_t raw_config_bytes,
    const char *prompt_policy, int standalone)
{
    int ok = (!standalone || writer_meta_text(metadata, count, "tokenizer.ggml.model", "gpt2")) &&
        writer_meta_text(metadata, count, "tokenizer.ggml.pre", tokenizer->pre_tokenizer) &&
        writer_meta_dynamic_array(metadata, count, "tokenizer.ggml.tokens",
                                  WRITER_META_TOKEN_ARRAY, YVEX_GGUF_VALUE_STRING,
                                  tokenizer->token_count) &&
        writer_meta_dynamic_array(metadata, count, "tokenizer.ggml.token_type",
                                  WRITER_META_TOKEN_TYPE_ARRAY, YVEX_GGUF_VALUE_INT32,
                                  tokenizer->token_count) &&
        writer_meta_dynamic_array(metadata, count, "tokenizer.ggml.merges",
                                  WRITER_META_MERGE_ARRAY, YVEX_GGUF_VALUE_STRING,
                                  tokenizer->merge_count);
    if (ok && standalone && tokenizer->bos_token_present)
        ok = writer_meta_u32(metadata, count, "tokenizer.ggml.bos_token_id",
                             tokenizer->bos_token_id);
    if (ok && standalone && tokenizer->eos_token_present)
        ok = writer_meta_u32(metadata, count, "tokenizer.ggml.eos_token_id",
                             tokenizer->eos_token_id);
    if (ok && tokenizer->pad_token_present)
        ok = writer_meta_u32(metadata, count, "tokenizer.ggml.padding_token_id",
                             tokenizer->pad_token_id);
    return ok && writer_meta_bool(metadata, count, "tokenizer.ggml.add_bos_token",
                                  tokenizer->add_bos_token) &&
           writer_meta_bool(metadata, count, "tokenizer.ggml.add_eos_token",
                            tokenizer->add_eos_token) &&
           (!standalone || writer_meta_text(metadata, count, "yvex.tokenizer.prompt_policy",
                                             prompt_policy)) &&
           writer_meta_string(metadata, count, "tokenizer.huggingface.json", raw_json,
                              raw_json_bytes) &&
           writer_meta_string(metadata, count, "yvex.tokenizer.config.json", raw_config,
                              raw_config_bytes) &&
           writer_meta_text(metadata, count, "yvex.tokenizer.json.sha256",
                            tokenizer->tokenizer_json_sha256) &&
           writer_meta_text(metadata, count, "yvex.tokenizer.config.sha256",
                            tokenizer->tokenizer_config_sha256) &&
           writer_meta_text(metadata, count, "yvex.tokenizer.json.git_oid",
                            tokenizer->tokenizer_json_git_oid) &&
           writer_meta_text(metadata, count, "yvex.tokenizer.config.git_oid",
                            tokenizer->tokenizer_config_git_oid);
}

/*
 * Serialize one staged GGUF metadata key/value pair.
 *
 * Appends canonical scalar or array bytes in deterministic order.
 */
static int writer_metadata_serialize(writer_buffer *buffer, const writer_metadata *entry,
                                     const yvex_gguf_tokenizer_metadata *tokenizer) {
    unsigned long long index;
    if (!writer_string(buffer, (const unsigned char *)entry->key, strlen(entry->key)) ||
        !writer_u32(buffer, entry->type))
        return 0;
    switch (entry->type) {
    case YVEX_GGUF_VALUE_UINT32:
        return writer_u32(buffer, (unsigned int)entry->u64);
    case YVEX_GGUF_VALUE_UINT64:
        return writer_u64(buffer, entry->u64);
    case YVEX_GGUF_VALUE_FLOAT32:
        return writer_f32(buffer, (float)entry->f64);
    case YVEX_GGUF_VALUE_FLOAT64:
        return writer_f64(buffer, entry->f64);
    case YVEX_GGUF_VALUE_BOOL: {
        unsigned char value = entry->boolean ? 1u : 0u;
        return yvex_core_bytes_append(buffer, &value, 1u);
    }
    case YVEX_GGUF_VALUE_STRING:
        return writer_string(buffer, entry->string_bytes, entry->string_length);
    case YVEX_GGUF_VALUE_ARRAY:
        if (!writer_u32(buffer, entry->element_type) || !writer_u64(buffer, entry->count))
            return 0;
        for (index = 0u; index < entry->count; ++index) {
            if (entry->source == WRITER_META_TOKEN_ARRAY) {
                const unsigned char *bytes;
                size_t byte_count;
                int token_type;
                if (!yvex_gguf_tokenizer_token_at(tokenizer, index, &bytes, &byte_count,
                                                  &token_type) ||
                    !writer_string(buffer, bytes, byte_count))
                    return 0;
            } else if (entry->source == WRITER_META_TOKEN_TYPE_ARRAY) {
                const unsigned char *bytes;
                size_t byte_count;
                int token_type;
                if (!yvex_gguf_tokenizer_token_at(tokenizer, index, &bytes, &byte_count,
                                                  &token_type) ||
                    !writer_u32(buffer, (unsigned int)token_type))
                    return 0;
            } else if (entry->source == WRITER_META_MERGE_ARRAY) {
                const unsigned char *bytes;
                size_t byte_count;
                if (!yvex_gguf_tokenizer_merge_at(tokenizer, index, &bytes, &byte_count) ||
                    !writer_string(buffer, bytes, byte_count))
                    return 0;
            } else if (entry->source == WRITER_META_U64_ARRAY) {
                unsigned long long value = entry->map_entry->array_values[index];
                if (entry->element_type == YVEX_GGUF_VALUE_UINT32) {
                    if (value > UINT_MAX || !writer_u32(buffer, (unsigned int)value))
                        return 0;
                } else if (!writer_u64(buffer, value))
                    return 0;
            } else if (entry->source == WRITER_META_F64_ARRAY) {
                double value = entry->map_entry->f64_array_values[index];
                if (!isfinite(value) || fabs(value) > FLT_MAX)
                    return 0;
                if (entry->element_type == YVEX_GGUF_VALUE_FLOAT32) {
                    if (!writer_f32(buffer, (float)value))
                        return 0;
                } else if (!writer_f64(buffer, value))
                    return 0;
            } else {
                return 0;
            }
        }
        return 1;
    default:
        return 0;
    }
}

static int writer_plan_identity(yvex_gguf_writer_plan *plan) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned char scalar[8];
    unsigned int index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, "yvex.gguf.writer.plan.v1", 24u) ||
        !yvex_sha256_update(&hash, plan->prefix, plan->prefix_bytes))
        return 0;
    for (index = 0u; index < 8u; ++index)
        scalar[index] = (unsigned char)(plan->summary.final_file_bytes >> (index * 8u));
    if (!yvex_sha256_update(&hash, scalar, sizeof(scalar)) ||
        !yvex_sha256_update(&hash, plan->summary.profile_identity,
                            strlen(plan->summary.profile_identity)) ||
        !yvex_sha256_update(&hash, plan->summary.payload_plan_identity,
                            strlen(plan->summary.payload_plan_identity)) ||
        !yvex_sha256_update(&hash, plan->summary.required_execution_identity,
                            strlen(plan->summary.required_execution_identity)) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, plan->summary.writer_plan_identity);
    return 1;
}

/*
 * Seed immutable provenance shared by fixture and model writer plans.
 *
 * Copies semantic facts but computes neither ranges nor identity.
 */
static void writer_plan_seed(yvex_gguf_writer_plan *plan, const yvex_quant_plan *quant_plan,
                             const yvex_quant_plan_summary *quant, unsigned long long tensor_count,
                             const yvex_gguf_writer_plan_options *options) {
    plan->quant_plan = quant_plan;
    plan->summary.schema_version = YVEX_GGUF_WRITER_SCHEMA_VERSION;
    plan->summary.gguf_version = WRITER_GGUF_VERSION;
    plan->summary.alignment = options->alignment;
    plan->summary.tensor_count = tensor_count;
    plan->summary.source_snapshot_identity = quant->source_snapshot_identity;
    plan->summary.mapping_identity = quant->mapping_identity;
    yvex_core_text_copy(plan->summary.payload_identity,
                        sizeof(plan->summary.payload_identity),
                        quant->required_payload_identity);
    yvex_core_text_copy(plan->summary.transform_identity,
                        sizeof(plan->summary.transform_identity),
                        quant->transform_identity);
    yvex_core_text_copy(plan->summary.profile_name, sizeof(plan->summary.profile_name), quant->profile_name);
    yvex_core_text_copy(plan->summary.profile_identity,
                        sizeof(plan->summary.profile_identity),
                        quant->profile_identity);
    yvex_core_text_copy(plan->summary.payload_plan_identity,
                        sizeof(plan->summary.payload_plan_identity),
                        quant->payload_plan_identity);
    if (options->required_execution_identity)
        yvex_core_text_copy(plan->summary.required_execution_identity,
                            sizeof(plan->summary.required_execution_identity),
                            options->required_execution_identity);
}

/*
 * Serialize a deterministic GGUF header, metadata set, and directory.
 *
 * Bounded buffer, admitted metadata/tokenizer, tensors, and counts. Emits no data-section padding
 * or tensor payload.
 */
static int writer_prefix_serialize(writer_buffer *buffer, const writer_metadata *metadata,
                                   unsigned int metadata_count,
                                   const yvex_gguf_tokenizer_metadata *tokenizer,
                                   const yvex_gguf_writer_tensor *tensors,
                                   unsigned long long tensor_count) {
    unsigned long long ordinal;

    if (!writer_u32(buffer, YVEX_GGUF_MAGIC) || !writer_u32(buffer, WRITER_GGUF_VERSION) ||
        !writer_u64(buffer, tensor_count) || !writer_u64(buffer, metadata_count))
        return 0;
    for (ordinal = 0u; ordinal < metadata_count; ++ordinal)
        if (!writer_metadata_serialize(buffer, &metadata[ordinal], tokenizer))
            return 0;
    for (ordinal = 0u; ordinal < tensor_count; ++ordinal) {
        const yvex_gguf_writer_tensor *tensor = &tensors[ordinal];
        unsigned int dimension;
        if (!writer_string(buffer, (const unsigned char *)tensor->name, strlen(tensor->name)) ||
            !writer_u32(buffer, tensor->rank))
            return 0;
        for (dimension = 0u; dimension < tensor->rank; ++dimension)
            if (!writer_u64(buffer, tensor->dims[dimension]))
                return 0;
        if (!writer_u32(buffer, tensor->qtype) || !writer_u64(buffer, tensor->relative_offset))
            return 0;
    }
    return 1;
}

/*
 * Align the data section and project all absolute tensor ranges.
 *
 * Partial plan, serialized prefix, alignment, and padded data span. Transfers prefix ownership and
 * finalizes structural/file geometry. Returns false on alignment, range, size, or buffer overflow.
 */
static int writer_prefix_finish(yvex_gguf_writer_plan *plan, writer_buffer *buffer,
                                unsigned int alignment, unsigned long long data_span) {
    unsigned long long structural_unaligned = buffer->count;
    unsigned long long data_offset;
    unsigned long long ordinal;

    if (!writer_align(structural_unaligned, alignment, &data_offset) || data_offset > SIZE_MAX ||
        !yvex_core_bytes_append_zero(buffer, (size_t)(data_offset - structural_unaligned)) ||
        !yvex_core_u64_add(data_offset, data_span, &plan->summary.final_file_bytes))
        return 0;
    for (ordinal = 0u; ordinal < plan->summary.tensor_count; ++ordinal) {
        yvex_gguf_writer_tensor *tensor = &plan->tensors[ordinal];
        if (!yvex_core_u64_add(data_offset, tensor->relative_offset, &tensor->absolute_offset) ||
            !yvex_core_u64_add(tensor->absolute_offset, tensor->raw_bytes, &tensor->absolute_end) ||
            !yvex_core_u64_add(data_offset, tensor->padded_end, &tensor->padded_end))
            return 0;
    }
    plan->prefix = buffer->data;
    plan->prefix_bytes = buffer->count;
    buffer->data = NULL;
    plan->summary.structural_bytes = structural_unaligned;
    plan->summary.pre_data_padding_bytes = data_offset - structural_unaligned;
    plan->summary.data_section_bytes = data_span;
    return 1;
}

/*
 * Initialize canonical writer-plan resource and alignment options.
 *
 * Resets the structure and installs bounded defaults.
 */
void yvex_gguf_writer_plan_options_default(yvex_gguf_writer_plan_options *options) {
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->alignment = WRITER_DEFAULT_ALIGNMENT;
    options->maximum_owned_bytes = WRITER_DEFAULT_BUDGET;
}

typedef enum {
    WRITER_FIXTURE_TENSOR_OK = 0,
    WRITER_FIXTURE_TENSOR_INVALID,
    WRITER_FIXTURE_TENSOR_ARITHMETIC
} writer_fixture_tensor_status;

/*
 * Project and account every explicit fixture terminal in plan order.
 *
 * Fixture names, sealed quant plan, alignment, and owned tensor array. Fills tensor rows, qtype
 * counts, padding, and total payload geometry.
 */
static writer_fixture_tensor_status writer_fixture_tensors_build(
    yvex_gguf_writer_plan *plan, const yvex_quant_plan *quant_plan,
    const yvex_gguf_writer_proof_tensor *fixtures, unsigned long long tensor_count,
    unsigned int alignment, unsigned long long *failed_ordinal,
    unsigned long long *data_span) {
    unsigned long long relative = 0u;
    unsigned long long ordinal;

    for (ordinal = 0u; ordinal < tensor_count; ++ordinal) {
        const yvex_quant_decision *decision = yvex_quant_plan_decision_at(quant_plan, ordinal);
        yvex_gguf_qtype_storage_result geometry;
        yvex_gguf_writer_tensor *tensor = &plan->tensors[ordinal];
        unsigned long long raw_end;
        unsigned int dimension;

        if (!decision || !fixtures[ordinal].name || !fixtures[ordinal].name[0] ||
            strlen(fixtures[ordinal].name) >= sizeof(tensor->name) ||
            yvex_gguf_qtype_validate_tensor_storage(decision->qtype, decision->dims,
                                                    decision->rank, decision->encoded_bytes,
                                                    &geometry) != YVEX_GGUF_QTYPE_STORAGE_OK) {
            *failed_ordinal = ordinal;
            return WRITER_FIXTURE_TENSOR_INVALID;
        }
        yvex_core_text_copy(tensor->name, sizeof(tensor->name), fixtures[ordinal].name);
        tensor->rank = decision->rank;
        tensor->qtype = decision->qtype;
        tensor->relative_offset = relative;
        tensor->raw_bytes = decision->encoded_bytes;
        for (dimension = 0u; dimension < decision->rank; ++dimension)
            tensor->dims[dimension] = decision->dims[dimension];
        if (!yvex_core_u64_add(relative, tensor->raw_bytes, &raw_end) ||
            !writer_align(raw_end, alignment, &tensor->padded_end) ||
            decision->qtype > YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID ||
            !yvex_core_u64_add(plan->summary.tensor_payload_bytes, tensor->raw_bytes,
                               &plan->summary.tensor_payload_bytes) ||
            !yvex_core_u64_add(plan->summary.tensor_padding_bytes,
                               tensor->padded_end - raw_end,
                               &plan->summary.tensor_padding_bytes) ||
            !yvex_core_u64_add(plan->summary.qtype_payload_bytes[decision->qtype],
                               tensor->raw_bytes,
                               &plan->summary.qtype_payload_bytes[decision->qtype]) ||
            plan->summary.qtype_tensor_counts[decision->qtype] == ULLONG_MAX) {
            *failed_ordinal = ordinal;
            return WRITER_FIXTURE_TENSOR_ARITHMETIC;
        }
        tensor->padded_bytes = tensor->padded_end - relative;
        relative = tensor->padded_end;
        plan->summary.qtype_tensor_counts[decision->qtype]++;
    }
    *data_span = relative;
    return WRITER_FIXTURE_TENSOR_OK;
}

static int writer_plan_build_tensor_proof(
    yvex_gguf_writer_plan **out, const yvex_quant_plan *quant_plan,
    const yvex_gguf_writer_proof_tensor *fixture_tensors, unsigned long long tensor_count,
    const yvex_gguf_writer_plan_options *options, yvex_gguf_writer_failure *failure,
    yvex_error *err) {
    const yvex_quant_plan_summary *quant = yvex_quant_plan_summary_get(quant_plan);
    yvex_gguf_writer_plan_options local;
    yvex_gguf_writer_plan *plan = NULL;
    writer_metadata metadata[2];
    writer_buffer buffer = {0};
    writer_fixture_tensor_status tensor_status;
    unsigned long long failed_ordinal = ULLONG_MAX;
    unsigned long long data_span = 0u;
    unsigned long long tensor_bytes;
    int unique;

    if (out)
        *out = NULL;
    if (!out || !quant || !quant->complete || !fixture_tensors || !tensor_count ||
        tensor_count != quant->terminal_count || tensor_count > SIZE_MAX / sizeof(*plan->tensors))
        return writer_fail(failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL, ULLONG_MAX,
                           ULLONG_MAX, quant ? quant->terminal_count : 0u, tensor_count, err,
                           YVEX_ERR_INVALID_ARG,
                           "matching explicit quant and writer fixture tensors are required");
    yvex_gguf_writer_plan_options_default(&local);
    if (options)
        local = *options;
    if (!local.alignment || (local.alignment & (local.alignment - 1u)) != 0u ||
        !local.maximum_owned_bytes)
        return writer_fail(failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL, ULLONG_MAX,
                           ULLONG_MAX, 1u, local.alignment, err, YVEX_ERR_INVALID_ARG,
                           "fixture writer options are invalid");
    plan = (yvex_gguf_writer_plan *)calloc(1u, sizeof(*plan));
    if (!plan)
        goto allocation_failure;
    plan->tensors = (yvex_gguf_writer_tensor *)calloc((size_t)tensor_count,
                                                      sizeof(*plan->tensors));
    if (!plan->tensors)
        goto allocation_failure;
    writer_plan_seed(plan, quant_plan, quant, tensor_count, &local);
    tensor_status = writer_fixture_tensors_build(plan, quant_plan, fixture_tensors, tensor_count,
                                                 local.alignment, &failed_ordinal, &data_span);
    if (tensor_status == WRITER_FIXTURE_TENSOR_INVALID)
        goto tensor_failure;
    if (tensor_status != WRITER_FIXTURE_TENSOR_OK)
        goto serialization_failure;
    unique = writer_tensor_names_unique(plan->tensors, tensor_count);
    if (unique < 0)
        goto allocation_failure;
    if (unique == 0)
        goto duplicate_failure;
    memset(metadata, 0, sizeof(metadata));
    yvex_core_text_copy(metadata[0].key, sizeof(metadata[0].key), "general.architecture");
    metadata[0].type = YVEX_GGUF_VALUE_STRING;
    metadata[0].source = WRITER_META_SCALAR;
    metadata[0].string_bytes = (const unsigned char *)"yvex-fixture";
    metadata[0].string_length = strlen("yvex-fixture");
    yvex_core_text_copy(metadata[1].key, sizeof(metadata[1].key), "general.alignment");
    metadata[1].type = YVEX_GGUF_VALUE_UINT32;
    metadata[1].source = WRITER_META_SCALAR;
    metadata[1].u64 = local.alignment;
    buffer.maximum = local.maximum_owned_bytes;
    if (!writer_prefix_serialize(&buffer, metadata, 2u, NULL, plan->tensors, tensor_count) ||
        !writer_prefix_finish(plan, &buffer, local.alignment, data_span))
        goto serialization_failure;
    plan->summary.metadata_count = 2u;
    if (!yvex_core_u64_mul(tensor_count, sizeof(*plan->tensors), &tensor_bytes) ||
        !yvex_core_u64_add(sizeof(*plan), tensor_bytes, &plan->summary.owned_bytes) ||
        !yvex_core_u64_add(plan->summary.owned_bytes, plan->prefix_bytes,
                           &plan->summary.owned_bytes))
        goto serialization_failure;
    if (plan->summary.owned_bytes > local.maximum_owned_bytes ||
        plan->summary.tensor_payload_bytes != quant->encoded_bytes || !writer_plan_identity(plan))
        goto serialization_failure;
    plan->summary.complete = 1;
    *out = plan;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;

duplicate_failure:
    yvex_gguf_writer_plan_release(&plan);
    return writer_fail(failure, YVEX_GGUF_WRITER_DUPLICATE_TENSOR, NULL, ULLONG_MAX, ULLONG_MAX,
                       tensor_count, 0u, err, YVEX_ERR_FORMAT,
                       "fixture tensor names are duplicate");
tensor_failure:
    yvex_gguf_writer_plan_release(&plan);
    return writer_fail(failure, YVEX_GGUF_WRITER_QTYPE_GEOMETRY, NULL, ULLONG_MAX,
                       failed_ordinal, 1u, 0u, err, YVEX_ERR_FORMAT,
                       "fixture tensor name or qtype geometry is invalid");
allocation_failure:
    yvex_gguf_writer_plan_release(&plan);
    return writer_fail(failure, YVEX_GGUF_WRITER_ALLOCATION, NULL, ULLONG_MAX, ULLONG_MAX,
                       tensor_count, 0u, err, YVEX_ERR_NOMEM,
                       "fixture writer allocation failed");
serialization_failure:
    free(buffer.data);
    yvex_gguf_writer_plan_release(&plan);
    return writer_fail(failure, YVEX_GGUF_WRITER_SERIALIZATION, NULL, ULLONG_MAX, ULLONG_MAX,
                       1u, 0u, err, YVEX_ERR_BOUNDS,
                       "fixture writer serialization failed");
}

typedef struct {
    const yvex_gguf_writer_component_input *input;
    const yvex_quant_plan *quant_plan;
    const yvex_quant_plan_summary *quant;
    const yvex_transform_ir *ir;
    const yvex_transform_ir_summary *transform;
    const yvex_gguf_tokenizer_summary *tokenizer;
    const unsigned char *raw_json, *raw_config;
    size_t raw_json_bytes, raw_config_bytes;
    yvex_gguf_writer_plan_options options;
    yvex_gguf_writer_plan *plan;
    writer_metadata metadata[WRITER_METADATA_CAP];
    unsigned int metadata_count;
    writer_buffer buffer;
    unsigned long long data_span;
    int physical_shape_folded;
    yvex_gguf_writer_failure *failure;
    yvex_error *err;
} writer_component_context;

static int writer_component_tokenizer_build(writer_component_context *context)
{
    const yvex_gguf_writer_tokenizer_input *input = context->input->tokenizer;
    yvex_gguf_tokenizer_failure failure;
    int rc;

    if (!input) return YVEX_OK;
    if (!input->acquisition || !input->source_root || !input->json_path ||
        !input->config_path || !input->pre_tokenizer || !input->prompt_policy ||
        !input->prompt_policy[0] || !input->token_count)
        return writer_fail(context->failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT,
                           "tokenizer", ULLONG_MAX, ULLONG_MAX, 1u, 0u,
                           context->err, YVEX_ERR_INVALID_ARG,
                           "complete acquired tokenizer input is required");
    memset(&failure, 0, sizeof(failure));
    rc = yvex_gguf_tokenizer_metadata_load_acquired(
        &context->plan->tokenizer, input->acquisition, input->source_root,
        input->json_path, input->config_path, input->token_count,
        input->pre_tokenizer, context->options.maximum_owned_bytes / 2u,
        &failure, context->err);
    context->tokenizer = yvex_gguf_tokenizer_summary_get(context->plan->tokenizer);
    if (rc != YVEX_OK || !context->tokenizer ||
        !yvex_gguf_tokenizer_raw_json(context->plan->tokenizer, &context->raw_json,
                                      &context->raw_json_bytes) ||
        !yvex_gguf_tokenizer_raw_config(context->plan->tokenizer, &context->raw_config,
                                        &context->raw_config_bytes))
        return writer_fail(context->failure, YVEX_GGUF_WRITER_METADATA_INCOMPLETE,
                           failure.field, failure.record_index, ULLONG_MAX,
                           failure.expected, failure.actual, context->err,
                           rc == YVEX_OK ? YVEX_ERR_FORMAT : (yvex_status)rc,
                           "verified component tokenizer metadata did not seal");
    return YVEX_OK;
}

static int writer_component_shape_matches(const yvex_transform_shape *logical,
                                          const yvex_quant_decision *physical,
                                          int *folded) {
    unsigned long long outer = 1ull;
    unsigned int dimension;

    if (!logical || !physical || !folded || !logical->rank || !physical->rank)
        return 0;
    if (logical->rank == physical->rank) {
        for (dimension = 0u; dimension < logical->rank; ++dimension)
            if (logical->dims[logical->rank - dimension - 1u] != physical->dims[dimension])
                return 0;
        return 1;
    }
    if (logical->rank <= YVEX_GGUF_QTYPE_MAX_DIMS ||
        physical->rank != YVEX_GGUF_QTYPE_MAX_DIMS)
        return 0;
    for (dimension = 0u; dimension + 1u < YVEX_GGUF_QTYPE_MAX_DIMS; ++dimension)
        if (logical->dims[logical->rank - dimension - 1u] != physical->dims[dimension])
            return 0;
    for (dimension = 0u;
         dimension < logical->rank - YVEX_GGUF_QTYPE_MAX_DIMS + 1u; ++dimension)
        if (!yvex_core_u64_mul(outer, logical->dims[dimension], &outer))
            return 0;
    if (outer != physical->dims[YVEX_GGUF_QTYPE_MAX_DIMS - 1u])
        return 0;
    *folded = 1;
    return 1;
}

static int writer_component_validate(writer_component_context *context) {
    const yvex_gguf_writer_component_input *input = context->input;
    const yvex_transform_ir_summary *transform = context->transform;
    const yvex_quant_plan_summary *quant = context->quant;

    if (!input || !input->architecture || !input->architecture[0] ||
        strlen(input->architecture) >= 64u || !input->target_id || !input->target_id[0] ||
        strlen(input->target_id) >= YVEX_GGUF_WRITER_NAME_CAP || !input->component_id ||
        !input->component_id[0] || strlen(input->component_id) >= YVEX_GGUF_WRITER_NAME_CAP ||
        !yvex_sha256_hex_valid(input->source_snapshot_identity) ||
        !input->source_snapshot_key ||
        !yvex_sha256_hex_valid(input->component_identity) ||
        !yvex_sha256_hex_valid(input->component_manifest_identity) ||
        !yvex_sha256_hex_valid(input->architecture_identity) ||
        !yvex_sha256_hex_valid(input->role_map_identity))
        return writer_fail(context->failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL,
                           ULLONG_MAX, ULLONG_MAX, 1u, 0u, context->err,
                           YVEX_ERR_INVALID_ARG,
                           "bounded component labels and immutable identities are required");
    if (!quant || !transform || !quant->complete || !transform->complete ||
        transform->schema_version != YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION ||
        yvex_quant_plan_transform_ir(context->quant_plan) != context->ir ||
        strcmp(transform->logical_model_identity, input->component_identity) != 0 ||
        strcmp(transform->component_manifest_identity,
               input->component_manifest_identity) != 0 ||
        strcmp(transform->architecture_identity, input->architecture_identity) != 0 ||
        strcmp(transform->role_map_identity, input->role_map_identity) != 0 ||
        strcmp(transform->transform_identity, quant->transform_identity) != 0 ||
        input->source_snapshot_key != transform->source_snapshot_identity ||
        quant->source_snapshot_identity != transform->source_snapshot_identity ||
        strcmp(quant->required_payload_identity, transform->required_payload_identity) != 0 ||
        !quant->mapping_identity || quant->terminal_count != transform->terminal_count ||
        quant->decision_count != transform->terminal_count ||
        transform->source_value_count != transform->terminal_count ||
        transform->node_count != transform->terminal_count)
        return writer_fail(context->failure, YVEX_GGUF_WRITER_IDENTITY_MISMATCH,
                           input->component_id, ULLONG_MAX, ULLONG_MAX, 1u, 0u,
                           context->err, YVEX_ERR_FORMAT,
                           "component, source, transform, and physical-plan identities diverge");
    return YVEX_OK;
}

static int writer_component_metadata_build(writer_component_context *context) {
    const yvex_gguf_writer_component_input *input = context->input;
    int ok;

    memset(context->metadata, 0, sizeof(context->metadata));
    ok = writer_meta_text(context->metadata, &context->metadata_count,
                            "general.architecture", input->architecture) &&
           writer_meta_u32(context->metadata, &context->metadata_count,
                           "general.alignment", context->options.alignment) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "general.name", input->component_id) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.logical.target", input->target_id) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.logical.component", input->component_id) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.source.snapshot.identity",
                            input->source_snapshot_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.logical.component.identity", input->component_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.logical.component_manifest.identity",
                            input->component_manifest_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.logical.architecture.identity",
                            input->architecture_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.logical.role_map.identity", input->role_map_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.logical.unresolved_requirements.identity",
                            context->transform->unresolved_requirements_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.transformation.identity",
                            context->transform->transform_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.physical.profile.name", context->quant->profile_name) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.physical.profile.identity",
                            context->quant->profile_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.physical.payload_plan.identity",
                            context->quant->payload_plan_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.payload.identity",
                            context->quant->required_payload_identity) &&
           writer_meta_text(context->metadata, &context->metadata_count,
                            "yvex.evidence.stage", "component-artifact-planned") &&
           (!context->physical_shape_folded ||
            writer_meta_text(context->metadata, &context->metadata_count,
                             "yvex.physical.shape.policy",
                             "reverse-logical-fold-outer-v1"));
    return ok && (!context->tokenizer || writer_tokenizer_metadata_add(
        context->metadata, &context->metadata_count, context->tokenizer,
        context->raw_json, context->raw_json_bytes, context->raw_config,
        context->raw_config_bytes, input->tokenizer->prompt_policy, 1));
}

static writer_fixture_tensor_status writer_component_tensors_build(
    writer_component_context *context, unsigned long long *failed_ordinal) {
    yvex_gguf_writer_proof_tensor *names;
    unsigned long long ordinal;
    writer_fixture_tensor_status status = WRITER_FIXTURE_TENSOR_INVALID;

    names = (yvex_gguf_writer_proof_tensor *)calloc(
        (size_t)context->transform->terminal_count, sizeof(*names));
    if (!names)
        return WRITER_FIXTURE_TENSOR_ARITHMETIC;
    for (ordinal = 0u; ordinal < context->transform->terminal_count; ++ordinal) {
        const yvex_transform_value *terminal = yvex_transform_ir_terminal_at(context->ir, ordinal);
        const yvex_transform_node *node =
            terminal ? yvex_transform_ir_node_at(context->ir, terminal->producer_node_id) : NULL;
        const yvex_transform_value *input =
            node ? yvex_transform_ir_node_input_at(context->ir, node, 0u) : NULL;
        const yvex_transform_source_value *source =
            input && input->kind == YVEX_TRANSFORM_VALUE_SOURCE
                ? yvex_transform_ir_source_at(context->ir, input->source_index) : NULL;
        const yvex_quant_decision *decision =
            yvex_quant_plan_decision_at(context->quant_plan, ordinal);

        if (!terminal || terminal->canonical_ordinal != ordinal || !node ||
            node->kind != YVEX_TRANSFORM_OP_IDENTITY || node->input_count != 1u ||
            node->numeric != YVEX_TRANSFORM_NUMERIC_EXACT || !source ||
            !source->source_name[0] || !decision ||
            !writer_component_shape_matches(&terminal->shape, decision,
                                            &context->physical_shape_folded)) {
            *failed_ordinal = ordinal;
            goto done;
        }
        names[ordinal].name = source->source_name;
    }
    status = writer_fixture_tensors_build(
        context->plan, context->quant_plan, names, context->transform->terminal_count,
        context->options.alignment, failed_ordinal, &context->data_span);
done:
    free(names);
    return status;
}

static int writer_plan_build_component(
    yvex_gguf_writer_plan **out, const yvex_quant_plan *quant_plan,
    const yvex_gguf_writer_component_input *input,
    const yvex_gguf_writer_plan_options *options, yvex_gguf_writer_failure *failure,
    yvex_error *err) {
    writer_component_context context;
    writer_fixture_tensor_status tensor_status;
    unsigned long long failed_ordinal = ULLONG_MAX;
    unsigned long long tensor_bytes;
    int unique;
    int rc;

    memset(&context, 0, sizeof(context));
    context.input = input;
    context.quant_plan = quant_plan;
    context.quant = yvex_quant_plan_summary_get(quant_plan);
    context.ir = yvex_quant_plan_transform_ir(quant_plan);
    context.transform = yvex_transform_ir_summary_get(context.ir);
    context.failure = failure;
    context.err = err;
    if (out)
        *out = NULL;
    if (!out || !quant_plan)
        return writer_fail(failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL, ULLONG_MAX,
                           ULLONG_MAX, 1u, 0u, err, YVEX_ERR_INVALID_ARG,
                           "one component physical plan is required");
    yvex_gguf_writer_plan_options_default(&context.options);
    if (options)
        context.options = *options;
    if (!context.options.alignment ||
        (context.options.alignment & (context.options.alignment - 1u)) != 0u ||
        context.options.maximum_owned_bytes < 1024u ||
        (context.options.required_execution_identity &&
         context.options.required_execution_identity[0] &&
         !yvex_sha256_hex_valid(context.options.required_execution_identity)))
        return writer_fail(failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL, ULLONG_MAX,
                           ULLONG_MAX, 1u, context.options.alignment, err,
                           YVEX_ERR_INVALID_ARG, "component writer options are invalid");
    rc = writer_component_validate(&context);
    if (rc != YVEX_OK)
        return rc;
    if (context.transform->terminal_count > SIZE_MAX / sizeof(*context.plan->tensors))
        goto allocation_failure;
    context.plan = (yvex_gguf_writer_plan *)calloc(1u, sizeof(*context.plan));
    if (!context.plan)
        goto allocation_failure;
    context.plan->tensors = (yvex_gguf_writer_tensor *)calloc(
        (size_t)context.transform->terminal_count, sizeof(*context.plan->tensors));
    if (!context.plan->tensors)
        goto allocation_failure;
    writer_plan_seed(context.plan, quant_plan, context.quant,
                     context.transform->terminal_count, &context.options);
    rc = writer_component_tokenizer_build(&context);
    if (rc != YVEX_OK)
        goto build_failure;
    tensor_status = writer_component_tensors_build(&context, &failed_ordinal);
    if (tensor_status == WRITER_FIXTURE_TENSOR_INVALID)
        goto tensor_failure;
    if (tensor_status != WRITER_FIXTURE_TENSOR_OK)
        goto allocation_failure;
    unique = writer_tensor_names_unique(context.plan->tensors,
                                        context.transform->terminal_count);
    if (unique < 0)
        goto allocation_failure;
    if (unique == 0)
        goto duplicate_failure;
    if (!writer_component_metadata_build(&context))
        goto metadata_failure;
    context.buffer.maximum = context.options.maximum_owned_bytes;
    if (!writer_prefix_serialize(&context.buffer, context.metadata,
                                 context.metadata_count, context.plan->tokenizer,
                                 context.plan->tensors,
                                 context.transform->terminal_count) ||
        !writer_prefix_finish(context.plan, &context.buffer, context.options.alignment,
                              context.data_span))
        goto serialization_failure;
    context.plan->summary.metadata_count = context.metadata_count;
    if (context.tokenizer) {
        context.plan->summary.tokenizer_token_count = context.tokenizer->token_count;
        context.plan->summary.tokenizer_merge_count = context.tokenizer->merge_count;
        context.plan->summary.tokenizer_embedded_bytes =
            context.raw_json_bytes + context.raw_config_bytes;
    }
    if (!yvex_core_u64_mul(context.transform->terminal_count,
                           sizeof(*context.plan->tensors), &tensor_bytes) ||
        !yvex_core_u64_add(sizeof(*context.plan), tensor_bytes,
                           &context.plan->summary.owned_bytes) ||
        !yvex_core_u64_add(context.plan->summary.owned_bytes,
                           context.plan->prefix_bytes,
                           &context.plan->summary.owned_bytes) ||
        (context.tokenizer &&
         !yvex_core_u64_add(context.plan->summary.owned_bytes,
                            context.tokenizer->owned_bytes,
                            &context.plan->summary.owned_bytes)) ||
        context.plan->summary.owned_bytes > context.options.maximum_owned_bytes ||
        context.plan->summary.tensor_payload_bytes != context.quant->encoded_bytes ||
        !writer_plan_identity(context.plan))
        goto serialization_failure;
    context.plan->summary.complete = 1;
    *out = context.plan;
    yvex_core_execution_observation_record(YVEX_CORE_OBSERVE_WRITER_PLAN, 1ull);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;

duplicate_failure:
    rc = writer_fail(failure, YVEX_GGUF_WRITER_DUPLICATE_TENSOR, NULL, ULLONG_MAX,
                     ULLONG_MAX, context.transform->terminal_count, 0u, err,
                     YVEX_ERR_FORMAT, "component tensor names are duplicate");
    goto build_failure;
metadata_failure:
    rc = writer_fail(failure, YVEX_GGUF_WRITER_DUPLICATE_METADATA, NULL,
                     context.metadata_count, ULLONG_MAX, 1u, 0u, err,
                     YVEX_ERR_FORMAT, "component metadata is duplicate or unrepresentable");
    goto build_failure;
tensor_failure:
    rc = writer_fail(failure, YVEX_GGUF_WRITER_TENSOR_DIVERGENCE, NULL, ULLONG_MAX,
                     failed_ordinal, 1u, 0u, err, YVEX_ERR_FORMAT,
                     "component terminal is not one exact source tensor");
    goto build_failure;
allocation_failure:
    rc = writer_fail(failure, YVEX_GGUF_WRITER_ALLOCATION, NULL, ULLONG_MAX,
                     ULLONG_MAX, context.transform ? context.transform->terminal_count : 0u,
                     0u, err, YVEX_ERR_NOMEM, "component writer allocation failed");
    goto build_failure;
serialization_failure:
    rc = writer_fail(failure, YVEX_GGUF_WRITER_SERIALIZATION, NULL, ULLONG_MAX,
                     ULLONG_MAX, 1u, 0u, err, YVEX_ERR_BOUNDS,
                     "component writer serialization or accounting failed");
build_failure:
    free(context.buffer.data);
    yvex_gguf_writer_plan_release(&context.plan);
    return rc;
}

typedef struct {
    yvex_gguf_writer_plan *plan;
    const yvex_quant_plan *quant_plan;
    const yvex_gguf_writer_lowering_api *lowering;
    const void *lowering_context;
    const yvex_source_verification *verification;
    const char *tokenizer_architecture;
    unsigned long long tokenizer_vocabulary_size;
    const yvex_quant_plan_summary *quant;
    yvex_gguf_writer_lowering_summary mapping;
    yvex_gguf_writer_plan_options options;
    writer_metadata metadata[WRITER_METADATA_CAP];
    yvex_gguf_writer_lowering_metadata lowering_metadata[WRITER_METADATA_CAP];
    unsigned int metadata_count;
    const yvex_gguf_tokenizer_summary *tokenizer;
    const unsigned char *raw_json;
    const unsigned char *raw_config;
    size_t raw_json_bytes;
    size_t raw_config_bytes;
    char source_identity[32];
    char mapping_identity[32];
    writer_buffer buffer;
    unsigned long long data_span;
    yvex_gguf_writer_failure *failure;
    yvex_error *err;
} writer_complete_context;

static int writer_artifact_lowering_summary(
    const void *context, yvex_gguf_writer_lowering_summary *out)
{
    const yvex_artifact_lowering_summary *summary =
        yvex_artifact_lowering_operations.summary(context);

    if (!summary || !out) return 0;
    *out = (yvex_gguf_writer_lowering_summary){
        summary->descriptor_count, summary->metadata_count,
        summary->source_identity, summary->mapping_identity, summary->complete};
    return 1;
}

static int writer_artifact_lowering_tensor(
    const void *context, unsigned long long ordinal,
    yvex_gguf_writer_lowering_tensor *out)
{
    const yvex_artifact_lowering_descriptor *row =
        yvex_artifact_lowering_operations.descriptor_at(context, ordinal);

    if (!row || !out || row->logical_rank > YVEX_GGUF_QTYPE_MAX_DIMS) return 0;
    memset(out, 0, sizeof(*out));
    yvex_core_text_copy(out->emitted_name, sizeof(out->emitted_name), row->emitted_name);
    out->logical_rank = row->logical_rank;
    memcpy(out->logical_dims, row->logical_dims,
           row->logical_rank * sizeof(out->logical_dims[0]));
    return 1;
}

static int writer_artifact_lowering_metadata(
    const void *context, unsigned long long ordinal,
    yvex_gguf_writer_lowering_metadata *out)
{
    const yvex_artifact_lowering_metadata *row =
        yvex_artifact_lowering_operations.metadata_at(context, ordinal);

    if (!row || !out || row->type > YVEX_ARTIFACT_LOWERING_METADATA_F64_ARRAY ||
        row->array_count > YVEX_ARTIFACT_LOWERING_METADATA_CAP)
        return 0;
    memset(out, 0, sizeof(*out));
    yvex_core_text_copy(out->key, sizeof(out->key), row->key);
    out->type = (yvex_gguf_writer_lowering_metadata_type)row->type;
    yvex_core_text_copy(out->string_value, sizeof(out->string_value), row->string_value);
    out->u64_value = row->u64_value;
    out->f64_value = row->f64_value;
    out->bool_value = row->bool_value;
    out->array_count = row->array_count;
    memcpy(out->array_values, row->array_values, sizeof(out->array_values));
    memcpy(out->f64_array_values, row->f64_array_values,
           sizeof(out->f64_array_values));
    return 1;
}

const yvex_gguf_writer_lowering_api *yvex_gguf_writer_artifact_lowering_api(void)
{
    static const yvex_gguf_writer_lowering_api api = {
        writer_artifact_lowering_summary,
        writer_artifact_lowering_tensor,
        writer_artifact_lowering_metadata};

    return &api;
}

static int writer_complete_plan_create(writer_complete_context *context) {
    yvex_gguf_tokenizer_failure tokenizer_failure;
    char tokenizer_message[YVEX_ERROR_MESSAGE_CAP];
    int rc;

    context->plan = (yvex_gguf_writer_plan *)calloc(1u, sizeof(*context->plan));
    if (!context->plan)
        return writer_fail(context->failure, YVEX_GGUF_WRITER_ALLOCATION, NULL, ULLONG_MAX,
                           ULLONG_MAX, sizeof(*context->plan), 0u, context->err, YVEX_ERR_NOMEM,
                           "writer plan allocation failed");
    writer_plan_seed(context->plan, context->quant_plan, context->quant,
                     context->quant->terminal_count, &context->options);
    context->plan->tensors = (yvex_gguf_writer_tensor *)calloc(
        (size_t)context->quant->terminal_count, sizeof(*context->plan->tensors));
    if (!context->plan->tensors)
        return writer_fail(context->failure, YVEX_GGUF_WRITER_ALLOCATION, NULL, ULLONG_MAX,
                           ULLONG_MAX, context->quant->terminal_count, 0u, context->err,
                           YVEX_ERR_NOMEM, "writer tensor plan allocation failed");
    rc = yvex_gguf_tokenizer_metadata_load(&context->plan->tokenizer, context->verification,
                                           context->tokenizer_vocabulary_size,
                                           context->tokenizer_architecture,
                                           context->options.maximum_owned_bytes / 2u,
                                           &tokenizer_failure, context->err);
    if (rc != YVEX_OK) {
        (void)snprintf(tokenizer_message, sizeof(tokenizer_message),
                       "verified tokenizer material is incomplete: %.180s",
                       yvex_error_message(context->err));
        return writer_fail(context->failure, YVEX_GGUF_WRITER_METADATA_INCOMPLETE,
                           tokenizer_failure.field, tokenizer_failure.record_index, ULLONG_MAX,
                           tokenizer_failure.expected, tokenizer_failure.actual, context->err,
                           (yvex_status)rc, tokenizer_message);
    }
    context->tokenizer = yvex_gguf_tokenizer_summary_get(context->plan->tokenizer);
    if (!context->tokenizer ||
        !yvex_gguf_tokenizer_raw_json(context->plan->tokenizer, &context->raw_json,
                                      &context->raw_json_bytes) ||
        !yvex_gguf_tokenizer_raw_config(context->plan->tokenizer, &context->raw_config,
                                        &context->raw_config_bytes))
        return writer_fail(context->failure, YVEX_GGUF_WRITER_METADATA_INCOMPLETE, "tokenizer",
                           ULLONG_MAX, ULLONG_MAX, 1u, 0u, context->err, YVEX_ERR_FORMAT,
                           "tokenizer metadata did not seal");
    return YVEX_OK;
}

static int writer_complete_add_lowering_metadata(writer_complete_context *context) {
    unsigned long long ordinal;

    memset(context->metadata, 0, sizeof(context->metadata));
    for (ordinal = 0u; ordinal < context->mapping.metadata_count; ++ordinal) {
        yvex_gguf_writer_lowering_metadata *entry =
            &context->lowering_metadata[ordinal];

        if (!context->lowering->metadata_at(context->lowering_context, ordinal, entry))
            return writer_fail(context->failure, YVEX_GGUF_WRITER_UNSUPPORTED_METADATA,
                               NULL, ordinal, ULLONG_MAX, 1u, 0u, context->err,
                               YVEX_ERR_FORMAT,
                               "lowering metadata projection is incomplete");
        if (!writer_meta_map(context->metadata, &context->metadata_count, entry))
            return writer_fail(
                context->failure,
                writer_metadata_key_exists(context->metadata, context->metadata_count,
                                           entry ? entry->key : "")
                    ? YVEX_GGUF_WRITER_DUPLICATE_METADATA
                    : YVEX_GGUF_WRITER_UNSUPPORTED_METADATA,
                entry ? entry->key : NULL, ordinal, ULLONG_MAX, 1u, 0u, context->err,
                YVEX_ERR_FORMAT, "lowering metadata cannot be serialized canonically");
    }
    return YVEX_OK;
}

/*
 * Add scalable tokenizer material and exact sidecar identities.
 *
 * Appends deterministic token arrays, policy fields, and digest metadata. Embeds verified
 * tokenizer evidence without changing tokenizer identity.
 */
static int writer_complete_add_tokenizer_metadata(writer_complete_context *context) {
    writer_metadata *metadata = context->metadata;
    unsigned int *count = &context->metadata_count;

    return writer_meta_u32(metadata, count, "general.alignment", context->options.alignment) &&
           writer_tokenizer_metadata_add(
               metadata, count, context->tokenizer, context->raw_json,
               context->raw_json_bytes, context->raw_config,
               context->raw_config_bytes, NULL, 0);
}

/*
 * Bind source, transform, lowering, and quant profile provenance.
 *
 * Canonical identity fields from the sealed quant plan. Stages deterministic provenance keys and
 * numeric-contract version.
 */
static int writer_complete_add_provenance_metadata(writer_complete_context *context) {
    writer_metadata *metadata = context->metadata;
    unsigned int *count = &context->metadata_count;

    (void)snprintf(context->source_identity, sizeof(context->source_identity), "%016llx",
                   context->quant->source_snapshot_identity);
    (void)snprintf(context->mapping_identity, sizeof(context->mapping_identity), "%016llx",
                   context->quant->mapping_identity);
    return writer_meta_text(metadata, count, "yvex.source.snapshot.identity",
                            context->source_identity) &&
           writer_meta_text(metadata, count, "yvex.source.payload.identity",
                            context->quant->required_payload_identity) &&
           writer_meta_text(metadata, count, "yvex.transform.identity",
                            context->quant->transform_identity) &&
           writer_meta_text(metadata, count, "yvex.gguf.mapping.identity",
                            context->mapping_identity) &&
           writer_meta_text(metadata, count, "yvex.quant.profile.name",
                            context->quant->profile_name) &&
           writer_meta_text(metadata, count, "yvex.quant.profile.identity",
                            context->quant->profile_identity) &&
           (!context->quant->policy_identity[0] ||
            (writer_meta_text(metadata, count, "yvex.quant.policy.identity",
                              context->quant->policy_identity) &&
             writer_meta_text(metadata, count, "yvex.quant.imatrix.identity",
                              context->quant->imatrix_identity))) &&
           writer_meta_u32(metadata, count, "yvex.quant.numeric_contract",
                           YVEX_QUANT_NUMERIC_CONTRACT_VERSION);
}

/*
 * Add one bijected quant/lowering tensor and advance physical geometry.
 *
 * Fills one tensor row and updates qtype/payload/padding accounting. Typed refusal covers
 * divergence, qtype geometry, and arithmetic overflow.
 */
static int writer_complete_tensor_add(writer_complete_context *context,
                                      unsigned long long ordinal,
                                      unsigned long long *relative) {
    const yvex_quant_decision *decision = yvex_quant_plan_decision_at(context->quant_plan, ordinal);
    yvex_gguf_writer_lowering_tensor projected;
    const yvex_gguf_writer_lowering_tensor *descriptor =
        context->lowering->tensor_at(context->lowering_context, ordinal, &projected)
            ? &projected
            : NULL;
    yvex_gguf_writer_tensor *tensor = &context->plan->tensors[ordinal];
    yvex_gguf_qtype_storage_result geometry;
    unsigned long long next;
    unsigned int dimension;

    if (!decision || !descriptor || decision->terminal_ordinal != ordinal ||
        descriptor->logical_rank != decision->rank || !descriptor->emitted_name[0] ||
        strlen(descriptor->emitted_name) >= sizeof(tensor->name))
        return writer_fail(context->failure, YVEX_GGUF_WRITER_TENSOR_DIVERGENCE,
                           descriptor ? descriptor->emitted_name : NULL, ULLONG_MAX, ordinal,
                           ordinal, decision ? decision->terminal_ordinal : ULLONG_MAX,
                           context->err, YVEX_ERR_FORMAT,
                           "quant decision and lowering tensor do not biject");
    memset(&geometry, 0, sizeof(geometry));
    if (yvex_gguf_qtype_validate_tensor_storage(decision->qtype, decision->dims, decision->rank,
                                                decision->encoded_bytes,
                                                &geometry) != YVEX_GGUF_QTYPE_STORAGE_OK)
        return writer_fail(context->failure, YVEX_GGUF_WRITER_QTYPE_GEOMETRY,
                           descriptor->emitted_name, ULLONG_MAX, ordinal, decision->encoded_bytes,
                           geometry.total_bytes, context->err, YVEX_ERR_FORMAT,
                           "quant decision violates canonical qtype byte geometry");
    yvex_core_text_copy(tensor->name, sizeof(tensor->name), descriptor->emitted_name);
    tensor->rank = decision->rank;
    tensor->qtype = decision->qtype;
    tensor->relative_offset = *relative;
    tensor->raw_bytes = decision->encoded_bytes;
    for (dimension = 0u; dimension < tensor->rank; ++dimension) {
        if (decision->dims[dimension] != descriptor->logical_dims[dimension])
            return writer_fail(context->failure, YVEX_GGUF_WRITER_TENSOR_DIVERGENCE, tensor->name,
                               ULLONG_MAX, ordinal, descriptor->logical_dims[dimension],
                               decision->dims[dimension], context->err, YVEX_ERR_FORMAT,
                               "quant and lowering tensor dimensions diverge");
        tensor->dims[dimension] = decision->dims[dimension];
    }
    if (!yvex_core_u64_add(*relative, tensor->raw_bytes, &next) ||
        !writer_align(next, context->options.alignment, &tensor->padded_end))
        return writer_fail(context->failure, YVEX_GGUF_WRITER_ARITHMETIC_OVERFLOW, tensor->name,
                           ULLONG_MAX, ordinal, ULLONG_MAX, *relative, context->err,
                           YVEX_ERR_BOUNDS, "tensor relative range or alignment overflowed");
    tensor->padded_bytes = tensor->padded_end - *relative;
    *relative = tensor->padded_end;
    if (decision->qtype > YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID ||
        context->plan->summary.qtype_tensor_counts[decision->qtype] == ULLONG_MAX ||
        !yvex_core_u64_add(context->plan->summary.qtype_payload_bytes[decision->qtype],
                           tensor->raw_bytes,
                           &context->plan->summary.qtype_payload_bytes[decision->qtype]) ||
        !yvex_core_u64_add(context->plan->summary.tensor_payload_bytes, tensor->raw_bytes,
                           &context->plan->summary.tensor_payload_bytes) ||
        !yvex_core_u64_add(context->plan->summary.tensor_padding_bytes,
                           tensor->padded_bytes - tensor->raw_bytes,
                           &context->plan->summary.tensor_padding_bytes))
        return writer_fail(context->failure, YVEX_GGUF_WRITER_ARITHMETIC_OVERFLOW, tensor->name,
                           ULLONG_MAX, ordinal, ULLONG_MAX, tensor->raw_bytes, context->err,
                           YVEX_ERR_BOUNDS, "aggregate tensor accounting overflowed");
    context->plan->summary.qtype_tensor_counts[decision->qtype]++;
    return YVEX_OK;
}

static int writer_complete_tensors_build(writer_complete_context *context) {
    unsigned long long relative = 0u;
    unsigned long long ordinal;
    int unique;
    int rc;

    for (ordinal = 0u; ordinal < context->quant->terminal_count; ++ordinal) {
        rc = writer_complete_tensor_add(context, ordinal, &relative);
        if (rc != YVEX_OK)
            return rc;
    }
    unique = writer_tensor_names_unique(context->plan->tensors, context->quant->terminal_count);
    if (unique <= 0)
        return writer_fail(context->failure,
                           unique == 0 ? YVEX_GGUF_WRITER_DUPLICATE_TENSOR
                                       : YVEX_GGUF_WRITER_ALLOCATION,
                           NULL, ULLONG_MAX, ULLONG_MAX, context->quant->terminal_count, 0u,
                           context->err, unique == 0 ? YVEX_ERR_FORMAT : YVEX_ERR_NOMEM,
                           unique == 0 ? "duplicate emitted tensor name refused"
                                       : "tensor name uniqueness index allocation failed");
    context->data_span = relative;
    return YVEX_OK;
}

/*
 * Serialize structural bytes and seal plan accounting and identity.
 *
 * Complete metadata/tensor context and ownership budget. Typed refusal covers serialization,
 * arithmetic, budget, or identity failure.
 */
static int writer_complete_plan_finish(writer_complete_context *context) {
    unsigned long long tensor_bytes;
    yvex_gguf_writer_plan_summary *summary = &context->plan->summary;

    memset(&context->buffer, 0, sizeof(context->buffer));
    context->buffer.maximum = context->options.maximum_owned_bytes;
    context->buffer.initial_capacity = 4096u;
    if (!writer_prefix_serialize(&context->buffer, context->metadata, context->metadata_count,
                                 context->plan->tokenizer, context->plan->tensors,
                                 context->quant->terminal_count) ||
        !writer_prefix_finish(context->plan, &context->buffer, context->options.alignment,
                              context->data_span))
        return writer_fail(context->failure, YVEX_GGUF_WRITER_SERIALIZATION, NULL, ULLONG_MAX,
                           ULLONG_MAX, 1u, 0u, context->err, YVEX_ERR_BOUNDS,
                           "GGUF structural prefix serialization overflowed or exceeded budget");
    summary->metadata_count = context->metadata_count;
    summary->tokenizer_token_count = context->tokenizer->token_count;
    summary->tokenizer_merge_count = context->tokenizer->merge_count;
    summary->tokenizer_embedded_bytes = context->raw_json_bytes + context->raw_config_bytes;
    if (!yvex_core_u64_mul(context->quant->terminal_count, sizeof(*context->plan->tensors),
                        &tensor_bytes) ||
        !yvex_core_u64_add(sizeof(*context->plan), tensor_bytes, &summary->owned_bytes) ||
        !yvex_core_u64_add(summary->owned_bytes, context->plan->prefix_bytes, &summary->owned_bytes) ||
        !yvex_core_u64_add(summary->owned_bytes, context->tokenizer->owned_bytes,
                        &summary->owned_bytes))
        return writer_fail(context->failure, YVEX_GGUF_WRITER_SERIALIZATION, NULL, ULLONG_MAX,
                           ULLONG_MAX, 1u, 0u, context->err, YVEX_ERR_BOUNDS,
                           "GGUF structural prefix serialization overflowed or exceeded budget");
    if (summary->owned_bytes > context->options.maximum_owned_bytes ||
        summary->tensor_payload_bytes != context->quant->encoded_bytes ||
        summary->final_file_bytes !=
            summary->structural_bytes + summary->pre_data_padding_bytes + context->data_span ||
        !writer_plan_identity(context->plan))
        return writer_fail(context->failure, YVEX_GGUF_WRITER_RESOURCE_LIMIT, NULL, ULLONG_MAX,
                           ULLONG_MAX, context->options.maximum_owned_bytes, summary->owned_bytes,
                           context->err, YVEX_ERR_BOUNDS,
                           "writer plan budget, payload accounting, or identity failed");
    summary->payload_bytes_read = 0u;
    summary->complete = 1;
    return YVEX_OK;
}

static int writer_plan_build_complete(
    yvex_gguf_writer_plan **out, const yvex_quant_plan *quant_plan,
    const yvex_gguf_writer_lowering_api *lowering, const void *lowering_context,
    const yvex_source_verification *verification, const char *tokenizer_architecture,
    unsigned long long tokenizer_vocabulary_size,
    const yvex_gguf_writer_plan_options *options, yvex_gguf_writer_failure *failure,
    yvex_error *err) {
    writer_complete_context context;
    int rc;

    memset(&context, 0, sizeof(context));
    context.quant_plan = quant_plan;
    context.lowering = lowering;
    context.lowering_context = lowering_context;
    context.verification = verification;
    context.tokenizer_architecture = tokenizer_architecture;
    context.tokenizer_vocabulary_size = tokenizer_vocabulary_size
                                            ? tokenizer_vocabulary_size
                                            : verification
                                                  ? verification->tokenizer_effective_vocab_size
                                                  : 0ull;
    context.quant = yvex_quant_plan_summary_get(quant_plan);
    context.failure = failure;
    context.err = err;
    if (out)
        *out = NULL;
    if (!out || !quant_plan || !lowering || !lowering_context || !verification ||
        !tokenizer_architecture || !tokenizer_architecture[0] ||
        !context.tokenizer_vocabulary_size ||
        !lowering->summary || !lowering->tensor_at || !lowering->metadata_at ||
        !lowering->summary(lowering_context, &context.mapping) || !context.quant ||
        !context.quant->complete || !context.mapping.complete ||
        context.quant->terminal_count != context.mapping.descriptor_count ||
        context.quant->terminal_count > SIZE_MAX / sizeof(*context.plan->tensors) ||
        context.mapping.metadata_count > WRITER_METADATA_CAP ||
        context.quant->mapping_identity != context.mapping.mapping_identity ||
        context.quant->source_snapshot_identity != verification->source_snapshot_identity ||
        strcmp(context.quant->required_payload_identity, verification->manifest_payload_identity) !=
            0)
        return writer_fail(
            failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL, ULLONG_MAX, ULLONG_MAX, 1u, 0u, err,
            YVEX_ERR_INVALID_ARG,
            "sealed quant plan, matching lowering, and verified source are required");
    yvex_gguf_writer_plan_options_default(&context.options);
    if (options)
        context.options = *options;
    if (!context.options.alignment ||
        (context.options.alignment & (context.options.alignment - 1u)) != 0u ||
        context.options.maximum_owned_bytes < 1024u ||
        (context.options.required_execution_identity &&
         context.options.required_execution_identity[0] &&
         !yvex_sha256_hex_valid(context.options.required_execution_identity)))
        return writer_fail(failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL, ULLONG_MAX, ULLONG_MAX,
                           1u, 0u, err, YVEX_ERR_INVALID_ARG,
                           "writer alignment, ownership budget, or execution identity is invalid");
    rc = writer_complete_plan_create(&context);
    if (rc != YVEX_OK)
        goto build_failure;
    rc = writer_complete_add_lowering_metadata(&context);
    if (rc != YVEX_OK)
        goto build_failure;
    if (!writer_complete_add_tokenizer_metadata(&context) ||
        !writer_complete_add_provenance_metadata(&context)) {
        rc = writer_fail(failure, YVEX_GGUF_WRITER_DUPLICATE_METADATA, NULL, context.metadata_count,
                         ULLONG_MAX, 1u, 0u, err, YVEX_ERR_FORMAT,
                         "required artifact metadata is duplicate or unrepresentable");
        goto build_failure;
    }

    rc = writer_complete_tensors_build(&context);
    if (rc != YVEX_OK)
        goto build_failure;

    rc = writer_complete_plan_finish(&context);
    if (rc != YVEX_OK)
        goto build_failure;
    *out = context.plan;
    yvex_core_execution_observation_record(
        YVEX_CORE_OBSERVE_WRITER_PLAN, 1ull);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;

build_failure:
    free(context.buffer.data);
    yvex_gguf_writer_plan_release(&context.plan);
    return rc;
}

int yvex_gguf_writer_plan_build(yvex_gguf_writer_plan **out,
                                const yvex_gguf_writer_plan_request *request,
                                yvex_gguf_writer_failure *failure, yvex_error *err) {
    if (out)
        *out = NULL;
    if (!out || !request)
        return writer_fail(failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL, ULLONG_MAX,
                           ULLONG_MAX, 1u, 0u, err, YVEX_ERR_INVALID_ARG,
                           "a typed writer-plan request is required");
    switch (request->input_class) {
    case YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT:
        return writer_plan_build_complete(
            out, request->quant_plan, request->input.complete.lowering,
            request->input.complete.lowering_context,
            request->input.complete.verification,
            request->input.complete.tokenizer_architecture,
            request->input.complete.tokenizer_vocabulary_size,
            request->options, failure, err);
    case YVEX_GGUF_WRITER_INPUT_TENSOR_PROOF:
        return writer_plan_build_tensor_proof(
            out, request->quant_plan, request->input.tensor_proof.tensors,
            request->input.tensor_proof.tensor_count, request->options, failure, err);
    case YVEX_GGUF_WRITER_INPUT_LOGICAL_COMPONENT:
        return writer_plan_build_component(
            out, request->quant_plan, &request->input.component, request->options,
            failure, err);
    default:
        return writer_fail(failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL, ULLONG_MAX,
                           ULLONG_MAX, YVEX_GGUF_WRITER_INPUT_TENSOR_PROOF,
                           request->input_class, err, YVEX_ERR_INVALID_ARG,
                           "writer-plan input class is unsupported");
    }
}

void yvex_gguf_writer_plan_release(yvex_gguf_writer_plan **plan_address) {
    yvex_gguf_writer_plan *plan;
    if (!plan_address || !*plan_address)
        return;
    plan = *plan_address;
    *plan_address = NULL;
    yvex_gguf_tokenizer_metadata_release(&plan->tokenizer);
    free(plan->tensors);
    free(plan->prefix);
    memset(plan, 0, sizeof(*plan));
    free(plan);
}

const yvex_gguf_writer_plan_summary *
yvex_gguf_writer_plan_summary_get(const yvex_gguf_writer_plan *plan) {
    return plan && plan->summary.complete ? &plan->summary : NULL;
}

const yvex_gguf_writer_tensor *yvex_gguf_writer_plan_tensor_at(const yvex_gguf_writer_plan *plan,
                                                               unsigned long long ordinal) {
    return plan && plan->summary.complete && ordinal < plan->summary.tensor_count
               ? &plan->tensors[ordinal]
               : NULL;
}

const unsigned char *yvex_gguf_writer_plan_prefix(const yvex_gguf_writer_plan *plan,
                                                  size_t *byte_count) {
    if (byte_count)
        *byte_count = 0u;
    if (!plan || !plan->summary.complete || !byte_count)
        return NULL;
    *byte_count = plan->prefix_bytes;
    return plan->prefix;
}
