/* Owner: gguf.artifact writer plan (TRACK.ARTIFACT).
 * Owns: exact metadata encoding, tokenizer projection, tensor directory order, checked offsets/alignment,
 *   structural prefix bytes, and plan identity.
 * Does not own: numeric execution, source reads, file creation/publication, reader roundtrip, support admission,
 *   materialization, runtime, or rendering.
 * Invariants: plan bytes are explicit little-endian GGUF v3; qtype geometry is consumed from its canonical owner;
 *   planning reads zero tensor payload.
 * Boundary: sealed structure and predicted ranges do not prove file emission.
 * Purpose: derive sealed GGUF structure and exact physical tensor ranges.
 * Inputs: immutable lowering, quantization, tokenizer, and provenance facts.
 * Effects: allocates a self-owned plan and deterministic structural prefix.
 * Failure: typed refusal releases partial ownership and publishes no plan. */
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/gguf.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>
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
    const yvex_deepseek_gguf_metadata *map_entry;
} writer_metadata;

typedef yvex_core_bytes writer_buffer;

struct yvex_gguf_writer_plan {
    yvex_gguf_writer_plan_summary summary;
    const yvex_quant_plan *quant_plan;
    const yvex_deepseek_gguf_map *map;
    yvex_gguf_tokenizer_metadata *tokenizer;
    yvex_gguf_writer_tensor *tensors;
    unsigned char *prefix;
    size_t prefix_bytes;
};

/* Purpose: publish one structured writer-plan refusal.
 * Inputs: failure coordinates, expected/actual facts, status, and diagnostic.
 * Effects: resets the failure record and updates the shared error object.
 * Failure: returns the supplied status without exposing partial plan state.
 * Boundary: centralizes diagnostics while callers retain validation policy. */
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

/* Purpose: align one file position to a checked power-of-two boundary.
 * Inputs: unaligned value, alignment, and caller-owned result.
 * Effects: writes the canonical aligned value.
 * Failure: returns false for invalid alignment, null output, or overflow.
 * Boundary: computes geometry only and emits no padding bytes. */
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

/* Purpose: order borrowed tensor pointers by canonical emitted name.
 * Inputs: qsort-compatible pointers to tensor pointers.
 * Effects: none beyond qsort comparison semantics.
 * Failure: valid writer tensors always yield a deterministic comparison.
 * Boundary: lexical ordering proves uniqueness, not directory order. */
static int writer_tensor_name_compare(const void *left, const void *right) {
    const yvex_gguf_writer_tensor *const *left_tensor =
        (const yvex_gguf_writer_tensor *const *)left;
    const yvex_gguf_writer_tensor *const *right_tensor =
        (const yvex_gguf_writer_tensor *const *)right;
    return strcmp((*left_tensor)->name, (*right_tensor)->name);
}

/* Purpose: prove emitted tensor names are unique in O(n log n).
 * Inputs: immutable tensor array and count.
 * Effects: allocates and releases a temporary pointer ordering.
 * Failure: returns zero for duplicates and negative one for invalid/allocation failure.
 * Boundary: does not reorder the canonical writer directory. */
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

/* Purpose: serialize one unsigned 32-bit value in little-endian order.
 * Inputs: owned prefix buffer and scalar value.
 * Effects: appends exactly four bytes.
 * Failure: returns false when the bounded buffer cannot grow.
 * Boundary: explicit-width encoding independent of host endianness. */
static int writer_u32(writer_buffer *buffer, unsigned int value) {
    unsigned char bytes[4];
    unsigned int index;
    for (index = 0u; index < 4u; ++index)
        bytes[index] = (unsigned char)(value >> (index * 8u));
    return yvex_core_bytes_append(buffer, bytes, sizeof(bytes));
}

/* Purpose: serialize one unsigned 64-bit value in little-endian order.
 * Inputs: owned prefix buffer and scalar value.
 * Effects: appends exactly eight bytes.
 * Failure: returns false when the bounded buffer cannot grow.
 * Boundary: explicit-width encoding independent of native structure layout. */
static int writer_u64(writer_buffer *buffer, unsigned long long value) {
    unsigned char bytes[8];
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        bytes[index] = (unsigned char)(value >> (index * 8u));
    return yvex_core_bytes_append(buffer, bytes, sizeof(bytes));
}

/* Purpose: serialize the exact IEEE-754 binary32 bit pattern.
 * Inputs: owned prefix buffer and admitted finite metadata value.
 * Effects: appends the scalar through canonical little-endian integer encoding.
 * Failure: returns false when prefix capacity is exhausted.
 * Boundary: preserves bits; metadata admission owns numeric policy. */
static int writer_f32(writer_buffer *buffer, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return writer_u32(buffer, bits);
}

/* Purpose: serialize the exact IEEE-754 binary64 bit pattern.
 * Inputs: owned prefix buffer and admitted finite metadata value.
 * Effects: appends the scalar through canonical little-endian integer encoding.
 * Failure: returns false when prefix capacity is exhausted.
 * Boundary: preserves bits without redefining floating-point policy. */
static int writer_f64(writer_buffer *buffer, double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return writer_u64(buffer, bits);
}

/* Purpose: serialize a GGUF length-prefixed byte string.
 * Inputs: owned buffer, borrowed string bytes, and exact byte count.
 * Effects: appends a 64-bit length followed by the borrowed bytes.
 * Failure: returns false when either append exceeds the prefix budget.
 * Boundary: accepts arbitrary bytes; callers own UTF-8 or tokenizer validity. */
static int writer_string(writer_buffer *buffer, const unsigned char *bytes, size_t byte_count) {
    return writer_u64(buffer, byte_count) &&
           yvex_core_bytes_append(buffer, bytes, byte_count);
}

/* Purpose: detect a duplicate key in the bounded metadata staging set.
 * Inputs: immutable staged entries, count, and candidate key.
 * Effects: none.
 * Failure: returns false when no exact duplicate exists.
 * Boundary: linear scan is bounded by the fixed metadata cap. */
static int writer_metadata_key_exists(const writer_metadata *entries, unsigned int count,
                                      const char *key) {
    unsigned int index;
    for (index = 0u; index < count; ++index)
        if (strcmp(entries[index].key, key) == 0)
            return 1;
    return 0;
}

/* Purpose: reserve one unique bounded metadata staging row.
 * Inputs: entry array, mutable count, and canonical key.
 * Effects: clears and initializes the next row while advancing count.
 * Failure: returns null for invalid, duplicate, oversized, or over-budget keys.
 * Boundary: stages metadata facts but does not serialize them. */
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

/* Purpose: stage one borrowed GGUF string metadata value.
 * Inputs: metadata set, key, borrowed bytes, and exact length.
 * Effects: records string type and borrowed byte view.
 * Failure: returns false for invalid bytes or unavailable metadata row.
 * Boundary: lifetime remains borrowed until prefix serialization completes. */
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

/* Purpose: stage one NUL-terminated text value as GGUF string metadata.
 * Inputs: metadata set, key, and borrowed C string.
 * Effects: records its byte view without retaining the terminator.
 * Failure: returns false for null text or metadata admission failure.
 * Boundary: convenience adapter over exact byte-string staging. */
static int writer_meta_text(writer_metadata *entries, unsigned int *count, const char *key,
                            const char *text) {
    return text &&
           writer_meta_string(entries, count, key, (const unsigned char *)text, strlen(text));
}

/* Purpose: stage one range-checked GGUF unsigned 32-bit metadata scalar.
 * Inputs: metadata set, key, and unsigned source value.
 * Effects: records canonical scalar type and value.
 * Failure: returns false for overflow or metadata admission failure.
 * Boundary: refuses implicit truncation from architecture-sized values. */
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

/* Purpose: stage one normalized GGUF boolean metadata scalar.
 * Inputs: metadata set, key, and truth value.
 * Effects: records a canonical zero-or-one boolean fact.
 * Failure: returns false when the metadata row cannot be admitted.
 * Boundary: owns encoding normalization, not source policy. */
static int writer_meta_bool(writer_metadata *entries, unsigned int *count, const char *key,
                            int value) {
    writer_metadata *entry = writer_metadata_new(entries, count, key);
    if (!entry)
        return 0;
    entry->type = YVEX_GGUF_VALUE_BOOL;
    entry->boolean = value != 0;
    return 1;
}

/* Purpose: project one typed DeepSeek lowering metadata row into GGUF storage.
 * Inputs: bounded metadata set and immutable canonical map entry.
 * Effects: records scalar/array source, type, and borrowed value references.
 * Failure: refuses unsupported type, invalid numeric range, or duplicate key.
 * Boundary: consumes lowering truth without reconstructing architecture facts. */
static int writer_meta_map(writer_metadata *entries, unsigned int *count,
                           const yvex_deepseek_gguf_metadata *map_entry) {
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
    case YVEX_DEEPSEEK_GGUF_METADATA_STRING:
        entry->type = YVEX_GGUF_VALUE_STRING;
        entry->string_bytes = (const unsigned char *)map_entry->string_value;
        entry->string_length = strlen(map_entry->string_value);
        return 1;
    case YVEX_DEEPSEEK_GGUF_METADATA_U64:
        if (!custom && map_entry->u64_value > UINT_MAX)
            return 0;
        entry->type = custom ? YVEX_GGUF_VALUE_UINT64 : YVEX_GGUF_VALUE_UINT32;
        entry->u64 = map_entry->u64_value;
        return 1;
    case YVEX_DEEPSEEK_GGUF_METADATA_F64:
        if (!isfinite(map_entry->f64_value) || fabs(map_entry->f64_value) > FLT_MAX)
            return 0;
        entry->type = custom ? YVEX_GGUF_VALUE_FLOAT64 : YVEX_GGUF_VALUE_FLOAT32;
        entry->f64 = map_entry->f64_value;
        return 1;
    case YVEX_DEEPSEEK_GGUF_METADATA_BOOL:
        entry->type = YVEX_GGUF_VALUE_BOOL;
        entry->boolean = map_entry->bool_value != 0;
        return 1;
    case YVEX_DEEPSEEK_GGUF_METADATA_U64_ARRAY:
        entry->type = YVEX_GGUF_VALUE_ARRAY;
        entry->source = WRITER_META_U64_ARRAY;
        entry->element_type = custom ? YVEX_GGUF_VALUE_UINT64 : YVEX_GGUF_VALUE_UINT32;
        entry->count = map_entry->array_count;
        return entry->count != 0u;
    case YVEX_DEEPSEEK_GGUF_METADATA_F64_ARRAY:
        entry->type = YVEX_GGUF_VALUE_ARRAY;
        entry->source = WRITER_META_F64_ARRAY;
        entry->element_type = custom ? YVEX_GGUF_VALUE_FLOAT64 : YVEX_GGUF_VALUE_FLOAT32;
        entry->count = map_entry->array_count;
        return entry->count != 0u;
    default:
        return 0;
    }
}

/* Purpose: stage one dynamically produced tokenizer metadata array.
 * Inputs: key, source class, element type, and exact nonzero cardinality.
 * Effects: records array production instructions in one metadata row.
 * Failure: returns false for zero count or metadata admission failure.
 * Boundary: array elements remain owned by the tokenizer metadata source. */
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

/* Purpose: serialize one staged GGUF metadata key/value pair.
 * Inputs: prefix buffer, typed metadata row, and optional tokenizer source.
 * Effects: appends canonical scalar or array bytes in deterministic order.
 * Failure: returns false for malformed source data, narrowing, or buffer limits.
 * Boundary: serializes admitted rows only and creates no metadata policy. */
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

/* Purpose: derive the deterministic writer-plan identity from semantic bytes.
 * Inputs: sealed-prefix candidate and bound profile/execution identities.
 * Effects: writes the plan identity into its owned summary.
 * Failure: returns false when canonical SHA-256 encoding cannot complete.
 * Boundary: excludes pointers, allocation order, paths, and runtime counters. */
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

/* Purpose: seed immutable provenance shared by fixture and model writer plans.
 * Inputs: empty owned plan, sealed quant summary, tensor count, and options.
 * Effects: binds identities and physical planning parameters into the summary.
 * Failure: admitted inputs make this initialization infallible.
 * Boundary: copies semantic facts but computes neither ranges nor identity. */
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

/* Purpose: serialize a deterministic GGUF header, metadata set, and directory.
 * Inputs: bounded buffer, admitted metadata/tokenizer, tensors, and counts.
 * Effects: appends structural bytes in canonical plan order.
 * Failure: returns false for malformed metadata or exhausted prefix capacity.
 * Boundary: emits no data-section padding or tensor payload. */
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

/* Purpose: align the data section and project all absolute tensor ranges.
 * Inputs: partial plan, serialized prefix, alignment, and padded data span.
 * Effects: transfers prefix ownership and finalizes structural/file geometry.
 * Failure: returns false on alignment, range, size, or buffer overflow.
 * Boundary: computes positions only and performs no file writes. */
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

/* Purpose: initialize canonical writer-plan resource and alignment options.
 * Inputs: caller-owned options storage.
 * Effects: resets the structure and installs bounded defaults.
 * Failure: a null destination is ignored.
 * Boundary: configures planning only and creates no plan or file. */
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

/* Purpose: project and account every explicit fixture terminal in plan order.
 * Inputs: fixture names, sealed quant plan, alignment, and owned tensor array.
 * Effects: fills tensor rows, qtype counts, padding, and total payload geometry.
 * Failure: returns typed local status and the first failing ordinal.
 * Boundary: fixture planning proves mechanics but cannot establish artifact admission. */
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

/* Purpose: build a structurally real GGUF plan from an explicit quant fixture.
 * Inputs: sealed fixture quant plan, names, count, options, and result records.
 * Effects: allocates a self-owned tensor directory and serialized prefix.
 * Failure: releases every partial allocation and returns typed writer refusal.
 * Boundary: omits tokenizer proof and therefore cannot enter complete admission. */
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
    yvex_gguf_writer_plan *plan;
    const yvex_quant_plan *quant_plan;
    const yvex_deepseek_gguf_map *map;
    const yvex_source_verification *verification;
    const yvex_quant_plan_summary *quant;
    const yvex_deepseek_gguf_map_summary *mapping;
    yvex_gguf_writer_plan_options options;
    writer_metadata metadata[WRITER_METADATA_CAP];
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
} writer_deepseek_context;

/* Purpose: allocate the DeepSeek plan and seal verified tokenizer material.
 * Inputs: build context with matching quant, source verification, and budget.
 * Effects: owns plan/tensor/tokenizer allocations and caches borrowed token views.
 * Failure: returns typed allocation or tokenizer-completeness refusal.
 * Boundary: consumes canonical tokenizer facts without parsing GGUF metadata. */
static int writer_deepseek_plan_create(writer_deepseek_context *context) {
    yvex_gguf_tokenizer_failure tokenizer_failure;
    int rc;

    context->plan = (yvex_gguf_writer_plan *)calloc(1u, sizeof(*context->plan));
    if (!context->plan)
        return writer_fail(context->failure, YVEX_GGUF_WRITER_ALLOCATION, NULL, ULLONG_MAX,
                           ULLONG_MAX, sizeof(*context->plan), 0u, context->err, YVEX_ERR_NOMEM,
                           "writer plan allocation failed");
    context->plan->map = context->map;
    writer_plan_seed(context->plan, context->quant_plan, context->quant,
                     context->quant->terminal_count, &context->options);
    context->plan->tensors = (yvex_gguf_writer_tensor *)calloc(
        (size_t)context->quant->terminal_count, sizeof(*context->plan->tensors));
    if (!context->plan->tensors)
        return writer_fail(context->failure, YVEX_GGUF_WRITER_ALLOCATION, NULL, ULLONG_MAX,
                           ULLONG_MAX, context->quant->terminal_count, 0u, context->err,
                           YVEX_ERR_NOMEM, "writer tensor plan allocation failed");
    rc = yvex_gguf_tokenizer_metadata_load(&context->plan->tokenizer, context->verification,
                                           context->verification->tokenizer_effective_vocab_size,
                                           "deepseek-v3", context->options.maximum_owned_bytes / 2u,
                                           &tokenizer_failure, context->err);
    if (rc != YVEX_OK)
        return writer_fail(context->failure, YVEX_GGUF_WRITER_METADATA_INCOMPLETE,
                           tokenizer_failure.field, tokenizer_failure.record_index, ULLONG_MAX,
                           tokenizer_failure.expected, tokenizer_failure.actual, context->err,
                           (yvex_status)rc, "complete verified tokenizer material is required");
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

/* Purpose: project canonical lowering metadata without reconstructing semantics.
 * Inputs: DeepSeek build context and immutable lowering metadata sequence.
 * Effects: fills the bounded metadata staging array in lowering order.
 * Failure: refuses duplicates, unsupported values, or unrepresentable scalars.
 * Boundary: lowering remains the authority for architecture metadata. */
static int writer_deepseek_add_lowering_metadata(writer_deepseek_context *context) {
    unsigned long long ordinal;

    memset(context->metadata, 0, sizeof(context->metadata));
    for (ordinal = 0u; ordinal < context->mapping->metadata_count; ++ordinal) {
        const yvex_deepseek_gguf_metadata *entry =
            yvex_model_register_deepseek_v4()->lowering.metadata_at(context->map, ordinal);
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

/* Purpose: add scalable tokenizer material and exact sidecar identities.
 * Inputs: sealed tokenizer summary plus borrowed raw JSON/config bytes.
 * Effects: appends deterministic token arrays, policy fields, and digest metadata.
 * Failure: returns false when any required row cannot be staged exactly.
 * Boundary: embeds verified tokenizer evidence without changing tokenizer identity. */
static int writer_deepseek_add_tokenizer_metadata(writer_deepseek_context *context) {
    writer_metadata *metadata = context->metadata;
    unsigned int *count = &context->metadata_count;
    const yvex_gguf_tokenizer_summary *tokenizer = context->tokenizer;

    return writer_meta_u32(metadata, count, "general.alignment", context->options.alignment) &&
           writer_meta_text(metadata, count, "tokenizer.ggml.pre", tokenizer->pre_tokenizer) &&
           writer_meta_dynamic_array(metadata, count, "tokenizer.ggml.tokens",
                                     WRITER_META_TOKEN_ARRAY, YVEX_GGUF_VALUE_STRING,
                                     tokenizer->token_count) &&
           writer_meta_dynamic_array(metadata, count, "tokenizer.ggml.token_type",
                                     WRITER_META_TOKEN_TYPE_ARRAY, YVEX_GGUF_VALUE_INT32,
                                     tokenizer->token_count) &&
           writer_meta_dynamic_array(metadata, count, "tokenizer.ggml.merges",
                                     WRITER_META_MERGE_ARRAY, YVEX_GGUF_VALUE_STRING,
                                     tokenizer->merge_count) &&
           writer_meta_u32(metadata, count, "tokenizer.ggml.padding_token_id",
                           tokenizer->pad_token_id) &&
           writer_meta_bool(metadata, count, "tokenizer.ggml.add_bos_token",
                            tokenizer->add_bos_token) &&
           writer_meta_bool(metadata, count, "tokenizer.ggml.add_eos_token",
                            tokenizer->add_eos_token) &&
           writer_meta_string(metadata, count, "tokenizer.huggingface.json", context->raw_json,
                              context->raw_json_bytes) &&
           writer_meta_string(metadata, count, "yvex.tokenizer.config.json", context->raw_config,
                              context->raw_config_bytes) &&
           writer_meta_text(metadata, count, "yvex.tokenizer.json.sha256",
                            tokenizer->tokenizer_json_sha256) &&
           writer_meta_text(metadata, count, "yvex.tokenizer.config.sha256",
                            tokenizer->tokenizer_config_sha256) &&
           writer_meta_text(metadata, count, "yvex.tokenizer.json.git_oid",
                            tokenizer->tokenizer_json_git_oid) &&
           writer_meta_text(metadata, count, "yvex.tokenizer.config.git_oid",
                            tokenizer->tokenizer_config_git_oid);
}

/* Purpose: bind source, transform, lowering, and quant profile provenance.
 * Inputs: canonical identity fields from the sealed quant plan.
 * Effects: stages deterministic provenance keys and numeric-contract version.
 * Failure: returns false for duplicate or unrepresentable metadata rows.
 * Boundary: records existing identities and never derives replacement identities. */
static int writer_deepseek_add_provenance_metadata(writer_deepseek_context *context) {
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

/* Purpose: add one bijected quant/lowering tensor and advance physical geometry.
 * Inputs: DeepSeek context, canonical ordinal, and current relative cursor.
 * Effects: fills one tensor row and updates qtype/payload/padding accounting.
 * Failure: typed refusal covers divergence, qtype geometry, and arithmetic overflow.
 * Boundary: consumes logical and physical facts but emits no payload bytes. */
static int writer_deepseek_tensor_add(writer_deepseek_context *context, unsigned long long ordinal,
                                      unsigned long long *relative) {
    const yvex_quant_decision *decision = yvex_quant_plan_decision_at(context->quant_plan, ordinal);
    const yvex_deepseek_gguf_descriptor *descriptor =
        yvex_model_register_deepseek_v4()->lowering.at(context->map, ordinal);
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

/* Purpose: build every DeepSeek tensor range and prove emitted-name uniqueness.
 * Inputs: initialized DeepSeek context with matching quant and lowering owners.
 * Effects: fills the complete tensor directory and records padded data span.
 * Failure: propagates first tensor refusal or duplicate-index allocation failure.
 * Boundary: canonical ordinal order remains unchanged by uniqueness checking. */
static int writer_deepseek_tensors_build(writer_deepseek_context *context) {
    unsigned long long relative = 0u;
    unsigned long long ordinal;
    int unique;
    int rc;

    for (ordinal = 0u; ordinal < context->quant->terminal_count; ++ordinal) {
        rc = writer_deepseek_tensor_add(context, ordinal, &relative);
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

/* Purpose: serialize structural bytes and seal plan accounting and identity.
 * Inputs: complete metadata/tensor context and ownership budget.
 * Effects: owns prefix bytes and finalizes sizes, tokenizer counts, and plan digest.
 * Failure: typed refusal covers serialization, arithmetic, budget, or identity failure.
 * Boundary: marks planning complete while payload reads remain zero. */
static int writer_deepseek_plan_finish(writer_deepseek_context *context) {
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

/* Purpose: build the complete DeepSeek writer plan from canonical owners.
 * Inputs: sealed quant plan, matching lowering, verified source, and options.
 * Effects: returns an independently owned tokenizer/prefix/tensor plan.
 * Failure: releases all partial state and returns a typed plan refusal.
 * Boundary: performs no file I/O and reads zero source payload bytes. */
static int writer_plan_build_deepseek(
    yvex_gguf_writer_plan **out, const yvex_quant_plan *quant_plan,
    const yvex_deepseek_gguf_map *map, const yvex_source_verification *verification,
    const yvex_gguf_writer_plan_options *options, yvex_gguf_writer_failure *failure,
    yvex_error *err) {
    writer_deepseek_context context;
    int rc;

    memset(&context, 0, sizeof(context));
    context.quant_plan = quant_plan;
    context.map = map;
    context.verification = verification;
    context.quant = yvex_quant_plan_summary_get(quant_plan);
    context.mapping = yvex_model_register_deepseek_v4()->lowering.summary(map);
    context.failure = failure;
    context.err = err;
    if (out)
        *out = NULL;
    if (!out || !quant_plan || !map || !verification || !context.quant || !context.mapping ||
        !context.quant->complete || !context.mapping->complete ||
        yvex_quant_plan_lowering(quant_plan) != map ||
        context.quant->terminal_count != context.mapping->descriptor_count ||
        context.quant->terminal_count != YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT ||
        context.quant->terminal_count > SIZE_MAX / sizeof(*context.plan->tensors) ||
        context.quant->mapping_identity != context.mapping->mapping_identity ||
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
    rc = writer_deepseek_plan_create(&context);
    if (rc != YVEX_OK)
        goto build_failure;
    rc = writer_deepseek_add_lowering_metadata(&context);
    if (rc != YVEX_OK)
        goto build_failure;
    if (!writer_deepseek_add_tokenizer_metadata(&context) ||
        !writer_deepseek_add_provenance_metadata(&context)) {
        rc = writer_fail(failure, YVEX_GGUF_WRITER_DUPLICATE_METADATA, NULL, context.metadata_count,
                         ULLONG_MAX, 1u, 0u, err, YVEX_ERR_FORMAT,
                         "required artifact metadata is duplicate or unrepresentable");
        goto build_failure;
    }

    rc = writer_deepseek_tensors_build(&context);
    if (rc != YVEX_OK)
        goto build_failure;

    rc = writer_deepseek_plan_finish(&context);
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

/* Purpose: build one writer plan from a tagged, immutable input class.
 * Inputs: destination, typed request, and optional failure records.
 * Effects: delegates to exactly one bounded or complete plan implementation.
 * Failure: rejects absent or unknown tags without publishing partial state.
 * Boundary: this is the sole cross-owner writer-plan constructor. */
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
        if (request->input.complete.family_adapter != yvex_model_register_deepseek_v4())
            return writer_fail(failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL,
                               ULLONG_MAX, ULLONG_MAX, 1u, 0u, err,
                               YVEX_ERR_UNSUPPORTED,
                               "writer family adapter is unsupported");
        return writer_plan_build_deepseek(
            out, request->quant_plan,
            (const yvex_deepseek_gguf_map *)request->input.complete.lowering,
            request->input.complete.verification, request->options, failure, err);
    case YVEX_GGUF_WRITER_INPUT_TENSOR_PROOF:
        return writer_plan_build_tensor_proof(
            out, request->quant_plan, request->input.tensor_proof.tensors,
            request->input.tensor_proof.tensor_count, request->options, failure, err);
    default:
        return writer_fail(failure, YVEX_GGUF_WRITER_INVALID_ARGUMENT, NULL, ULLONG_MAX,
                           ULLONG_MAX, YVEX_GGUF_WRITER_INPUT_TENSOR_PROOF,
                           request->input_class, err, YVEX_ERR_INVALID_ARG,
                           "writer-plan input class is unsupported");
    }
}

/* Purpose: release all independently owned writer-plan state idempotently.
 * Inputs: address of an optional writer-plan handle.
 * Effects: nulls the caller handle and frees tokenizer, directory, prefix, and plan.
 * Failure: absent handles are accepted; cleanup reports no recoverable error.
 * Boundary: borrowed lowering and quant owners are never released here. */
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

/* Purpose: expose the immutable summary of a sealed writer plan.
 * Inputs: optional borrowed plan.
 * Effects: none.
 * Failure: returns null for absent or incomplete plans.
 * Boundary: returned view remains owned by the plan. */
const yvex_gguf_writer_plan_summary *
yvex_gguf_writer_plan_summary_get(const yvex_gguf_writer_plan *plan) {
    return plan && plan->summary.complete ? &plan->summary : NULL;
}

/* Purpose: retrieve one immutable tensor-directory row by canonical ordinal.
 * Inputs: sealed borrowed plan and requested ordinal.
 * Effects: none.
 * Failure: returns null for incomplete plans or out-of-range ordinals.
 * Boundary: returned tensor view cannot outlive the plan. */
const yvex_gguf_writer_tensor *yvex_gguf_writer_plan_tensor_at(const yvex_gguf_writer_plan *plan,
                                                               unsigned long long ordinal) {
    return plan && plan->summary.complete && ordinal < plan->summary.tensor_count
               ? &plan->tensors[ordinal]
               : NULL;
}

/* Purpose: borrow the exact serialized structural prefix of a sealed plan.
 * Inputs: sealed plan and caller-owned byte-count output.
 * Effects: publishes prefix length without copying bytes.
 * Failure: returns null and zero count for invalid inputs or incomplete plans.
 * Boundary: borrowed bytes remain immutable and plan-owned. */
const unsigned char *yvex_gguf_writer_plan_prefix(const yvex_gguf_writer_plan *plan,
                                                  size_t *byte_count) {
    if (byte_count)
        *byte_count = 0u;
    if (!plan || !plan->summary.complete || !byte_count)
        return NULL;
    *byte_count = plan->prefix_bytes;
    return plan->prefix;
}
