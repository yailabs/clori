/* Persist portable runtime-model inputs; reopened snapshots are leases, never identity authority. */
#include "src/runtime/private.h"
#include <yvex/internal/core.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/operator_graph.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#define BINDING_MAGIC_V14 "YVRBND14"
#define BINDING_MAGIC_V15 "YVRBND15"
#define BINDING_SCHEMA_V14 14u
#define BINDING_MAGIC_BYTES 8u
#define BINDING_HEADER_BYTES (BINDING_MAGIC_BYTES + 16u + 64u)
#define BINDING_MAX_BYTES (64u * 1024u * 1024u)
#define BINDING_MAX_RECORDS 1048576ull
#define BINDING_MAX_LAYERS 65536ull
typedef yvex_core_bytes binding_bytes;
typedef struct {
    const unsigned char *data;
    size_t count, offset;
} binding_cursor;
typedef enum {
    BINDING_PARSE_FORMAT = 0,
    BINDING_PARSE_OK,
    BINDING_PARSE_BOUNDS,
    BINDING_PARSE_ALLOCATION
} binding_parse_result;
typedef struct {
    yvex_runtime_binding_failure_code code;
    yvex_status status;
    const char *reason;
} binding_parse_failure;
static const binding_parse_failure binding_parse_failures[] = {
    {YVEX_RUNTIME_BINDING_FAILURE_FORMAT, YVEX_ERR_FORMAT,
     "runtime binding canonical records are malformed"},
    {YVEX_RUNTIME_BINDING_FAILURE_NONE, YVEX_OK, NULL},
    {YVEX_RUNTIME_BINDING_FAILURE_BOUNDS, YVEX_ERR_BOUNDS,
     "runtime binding record declarations exceed their canonical byte budget"},
    {YVEX_RUNTIME_BINDING_FAILURE_ALLOCATION, YVEX_ERR_NOMEM,
     "runtime binding record allocation failed"},
};
static const yvex_runtime_binding_failure_code binding_file_codes[] = {
    /* Ordered by yvex_core_file_stage. */
    YVEX_RUNTIME_BINDING_FAILURE_FORMAT, YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT,
    YVEX_RUNTIME_BINDING_FAILURE_DIRECTORY, YVEX_RUNTIME_BINDING_FAILURE_CREATE,
    YVEX_RUNTIME_BINDING_FAILURE_WRITE, YVEX_RUNTIME_BINDING_FAILURE_SYNC,
    YVEX_RUNTIME_BINDING_FAILURE_SYNC, YVEX_RUNTIME_BINDING_FAILURE_CONFLICT,
    YVEX_RUNTIME_BINDING_FAILURE_PUBLISH, YVEX_RUNTIME_BINDING_FAILURE_SYNC,
    YVEX_RUNTIME_BINDING_FAILURE_OPEN, YVEX_RUNTIME_BINDING_FAILURE_BOUNDS,
    YVEX_RUNTIME_BINDING_FAILURE_ALLOCATION, YVEX_RUNTIME_BINDING_FAILURE_TRUNCATED,
    YVEX_RUNTIME_BINDING_FAILURE_TRUNCATED, YVEX_RUNTIME_BINDING_FAILURE_PUBLISH,
    YVEX_RUNTIME_BINDING_FAILURE_FORMAT,
};
#define binding_reject yvex_runtime_private_binding_refuse
static int bytes_put_u64(binding_bytes *bytes, unsigned long long value)
{
    unsigned char encoded[8];
    unsigned int i;
    for (i = 0u; i < 8u; ++i) encoded[i] = (unsigned char)(value >> (i * 8u));
    return yvex_core_bytes_append(bytes, encoded, sizeof(encoded));
}
static int bytes_put_text(binding_bytes *bytes, const char *text)
{
    size_t length = text ? strlen(text) : 0u;
    return bytes_put_u64(bytes, (unsigned long long)length) &&
           yvex_core_bytes_append(bytes, text, length);
}
static int cursor_take(binding_cursor *cursor, void *out, size_t count)
{
    if (!cursor || (!out && count) || cursor->offset > cursor->count ||
        count > cursor->count - cursor->offset) return 0;
    if (count) memcpy(out, cursor->data + cursor->offset, count);
    cursor->offset += count;
    return 1;
}
static int cursor_u64(binding_cursor *cursor, unsigned long long *out)
{
    unsigned char encoded[8];
    unsigned long long value = 0ull;
    unsigned int i;
    if (!out || !cursor_take(cursor, encoded, sizeof(encoded))) return 0;
    for (i = 0u; i < 8u; ++i) value |= (unsigned long long)encoded[i] << (i * 8u);
    *out = value;
    return 1;
}
static int cursor_text(binding_cursor *cursor, char *out, size_t capacity)
{
    unsigned long long length;
    if (!out || !capacity || !cursor_u64(cursor, &length) ||
        length >= (unsigned long long)capacity || length > cursor->count - cursor->offset ||
        (length && memchr(cursor->data + cursor->offset, '\0', (size_t)length)))
        return 0;
    if (length) memcpy(out, cursor->data + cursor->offset, (size_t)length);
    out[length] = '\0';
    cursor->offset += (size_t)length;
    return 1;
}
static int model_execution_write(binding_bytes *bytes,
                                 const yvex_model_execution_descriptor *descriptor)
{
    unsigned char encoded[YVEX_MODEL_EXECUTION_WIRE_BYTES];
    yvex_error err;
    return yvex_model_execution_descriptor_encode(descriptor, encoded, &err) == YVEX_OK &&
           bytes_put_u64(bytes, sizeof(encoded)) &&
           yvex_core_bytes_append(bytes, encoded, sizeof(encoded));
}
static int model_execution_read(binding_cursor *cursor, yvex_model_execution_descriptor *descriptor)
{
    unsigned char encoded[YVEX_MODEL_EXECUTION_WIRE_BYTES];
    unsigned long long byte_count;
    yvex_error err;
    return cursor_u64(cursor, &byte_count) && byte_count == sizeof(encoded) &&
           cursor_take(cursor, encoded, sizeof(encoded)) &&
           yvex_model_execution_descriptor_decode(
               encoded, sizeof(encoded), descriptor, &err) == YVEX_OK;
}
typedef enum {
    BINDING_FIELD_UNSIGNED = 0, BINDING_FIELD_SIGNED,
    BINDING_FIELD_FLOAT,
    BINDING_FIELD_TEXT
} binding_field_kind;
typedef struct {
    size_t offset, width, count;
    binding_field_kind kind;
} binding_field;
#define FIELD_U(type, member)                                                                    \
    { offsetof(type, member), sizeof(((type *)0)->member), 1u, BINDING_FIELD_UNSIGNED }
#define FIELD_S(type, member)                                                                    \
    { offsetof(type, member), sizeof(((type *)0)->member), 1u, BINDING_FIELD_SIGNED }
#define FIELD_F(type, member)                                                                    \
    { offsetof(type, member), sizeof(((type *)0)->member), 1u, BINDING_FIELD_FLOAT }
#define FIELD_T(type, member)                                                                    \
    { offsetof(type, member), sizeof(((type *)0)->member), 1u, BINDING_FIELD_TEXT }
#define FIELD_A(type, member)                                                                    \
    {                                                                                            \
        offsetof(type, member), sizeof(((type *)0)->member[0]),                                 \
            sizeof(((type *)0)->member) / sizeof(((type *)0)->member[0]),                        \
            BINDING_FIELD_UNSIGNED                                                               \
    }
#define FIELD_N(type, parent, nested, member, field_kind)                                        \
    {                                                                                            \
        offsetof(type, parent) + offsetof(nested, member), sizeof(((nested *)0)->member), 1u,   \
            field_kind                                                                           \
    }
#define FIELD_COUNT(fields) (sizeof(fields) / sizeof((fields)[0]))
static int record_count_fits(const binding_cursor *cursor, unsigned long long record_count,
                             size_t object_bytes)
{
    size_t remaining;
    if (!cursor || cursor->offset > cursor->count || !record_count || !object_bytes ||
        record_count > (unsigned long long)SIZE_MAX)
        return 0;
    remaining = cursor->count - cursor->offset;
    return (size_t)record_count <= remaining / sizeof(unsigned long long) &&
           (size_t)record_count <= BINDING_MAX_BYTES / object_bytes;
}
static int field_unsigned_load(const void *field, size_t width, unsigned long long *value)
{
    unsigned char u8;
    unsigned short u16;
    unsigned int u32;
    *value = 0ull;
    if (width == sizeof(u8)) memcpy(&u8, field, width), *value = u8;
    else if (width == sizeof(u16)) memcpy(&u16, field, width), *value = u16;
    else if (width == sizeof(u32)) memcpy(&u32, field, width), *value = u32;
    else if (width == sizeof(*value)) memcpy(value, field, width);
    else return 0;
    return 1;
}
static int field_unsigned_store(void *field, size_t width, unsigned long long value)
{
    union {
        unsigned char u8;
        unsigned short u16;
        unsigned int u32;
        unsigned long long u64;
    } converted;
    if (width == sizeof(converted.u8) && value <= UCHAR_MAX)
        converted.u8 = (unsigned char)value;
    else if (width == sizeof(converted.u16) && value <= USHRT_MAX)
        converted.u16 = (unsigned short)value;
    else if (width == sizeof(converted.u32) && value <= UINT_MAX)
        converted.u32 = (unsigned int)value;
    else if (width == sizeof(converted.u64))
        converted.u64 = value;
    else return 0;
    memcpy(field, &converted, width);
    return 1;
}
static int fields_write(binding_bytes *bytes, const void *object,
                        const binding_field *fields, size_t field_count)
{
    const unsigned char *base = (const unsigned char *)object;
    size_t i, j;
    for (i = 0u; i < field_count; ++i) {
        const binding_field *field = &fields[i];
        if (field->kind == BINDING_FIELD_TEXT) {
            if (!bytes_put_text(bytes, (const char *)(base + field->offset))) return 0;
            continue;
        }
        for (j = 0u; j < field->count; ++j) {
            const void *value = base + field->offset + j * field->width;
            unsigned long long number;
            double floating;
            long long signed_value;
            if (field->kind == BINDING_FIELD_UNSIGNED) {
                if (!field_unsigned_load(value, field->width, &number) ||
                    !bytes_put_u64(bytes, number)) return 0;
            } else if (field->kind == BINDING_FIELD_SIGNED) {
                if (field->width != sizeof(signed_value)) return 0;
                memcpy(&signed_value, value, sizeof(signed_value));
                if (!bytes_put_u64(bytes, (unsigned long long)signed_value)) return 0;
            } else {
                if (field->width != sizeof(floating)) return 0;
                memcpy(&floating, value, sizeof(floating));
                memcpy(&number, &floating, sizeof(number));
                if (!bytes_put_u64(bytes, number)) return 0;
            }
        }
    }
    return 1;
}
/* Decode a bounded field table without following pointers or validating cross-record identity. */
static int fields_read(binding_cursor *cursor, void *object,
                       const binding_field *fields, size_t field_count)
{
    unsigned char *base = (unsigned char *)object;
    size_t i, j;
    for (i = 0u; i < field_count; ++i) {
        const binding_field *field = &fields[i];
        if (field->kind == BINDING_FIELD_TEXT) {
            if (!cursor_text(cursor, (char *)(base + field->offset), field->width)) return 0;
            continue;
        }
        for (j = 0u; j < field->count; ++j) {
            void *value = base + field->offset + j * field->width;
            unsigned long long number;
            double floating;
            long long signed_value;
            if (field->kind == BINDING_FIELD_UNSIGNED) {
                if (!cursor_u64(cursor, &number) ||
                    !field_unsigned_store(value, field->width, number)) return 0;
            } else if (field->kind == BINDING_FIELD_SIGNED) {
                if (field->width != sizeof(signed_value) ||
                    !cursor_u64(cursor, &number)) return 0;
                memcpy(&signed_value, &number, sizeof(signed_value));
                memcpy(value, &signed_value, sizeof(signed_value));
            } else {
                if (field->width != sizeof(floating) || !cursor_u64(cursor, &number)) return 0;
                memcpy(&floating, &number, sizeof(floating));
                memcpy(value, &floating, sizeof(floating));
            }
        }
    }
    return 1;
}
static int record_read(binding_cursor *cursor, void *object, size_t object_size,
                       const binding_field *fields, size_t field_count)
{
    memset(object, 0, object_size);
    return fields_read(cursor, object, fields, field_count);
}
static const binding_field admission_fields[] = {
    FIELD_U(yvex_complete_artifact_admission, artifact_class),
    FIELD_U(yvex_complete_artifact_admission, metadata_count),
    FIELD_U(yvex_complete_artifact_admission, tensor_count),
    FIELD_U(yvex_complete_artifact_admission, payload_bytes),
    FIELD_U(yvex_complete_artifact_admission, file_bytes),
    FIELD_U(yvex_complete_artifact_admission, source_snapshot_identity),
    FIELD_U(yvex_complete_artifact_admission, mapping_identity),
    FIELD_N(yvex_complete_artifact_admission, file_snapshot, yvex_artifact_snapshot,
            size, BINDING_FIELD_UNSIGNED),
    FIELD_T(yvex_complete_artifact_admission, payload_identity),
    FIELD_T(yvex_complete_artifact_admission, transform_identity),
    FIELD_T(yvex_complete_artifact_admission, profile_identity),
    FIELD_T(yvex_complete_artifact_admission, profile_name),
    FIELD_T(yvex_complete_artifact_admission, quant_execution_identity),
    FIELD_T(yvex_complete_artifact_admission, payload_plan_identity),
    FIELD_T(yvex_complete_artifact_admission, payload_byte_identity),
    FIELD_T(yvex_complete_artifact_admission, writer_plan_identity),
    FIELD_T(yvex_complete_artifact_admission, artifact_identity),
    FIELD_T(yvex_complete_artifact_admission, admission_identity),
    FIELD_T(yvex_complete_artifact_admission, official_reader_revision),
    FIELD_U(yvex_complete_artifact_admission, tokenizer_complete),
    FIELD_U(yvex_complete_artifact_admission, native_reader_accepted),
    FIELD_U(yvex_complete_artifact_admission, official_reader_accepted),
    FIELD_U(yvex_complete_artifact_admission, payload_integrity_accepted),
    FIELD_U(yvex_complete_artifact_admission, materialization_input_ready),
    FIELD_U(yvex_complete_artifact_admission, runtime_supported),
    FIELD_U(yvex_complete_artifact_admission, artifact_bytes_hashed),
    FIELD_U(yvex_complete_artifact_admission, artifact_identity_verified),
    FIELD_U(yvex_complete_artifact_admission, complete),
};
static const binding_field physical_compatibility_fields[] = {
    FIELD_U(yvex_artifact_physical_compatibility, schema_version),
    FIELD_U(yvex_artifact_physical_compatibility, source_snapshot_identity),
    FIELD_U(yvex_artifact_physical_compatibility, mapping_identity),
    FIELD_U(yvex_artifact_physical_compatibility, tensor_count),
    FIELD_U(yvex_artifact_physical_compatibility, tensors_compared),
    FIELD_U(yvex_artifact_physical_compatibility, payload_bytes),
    FIELD_U(yvex_artifact_physical_compatibility, payload_bytes_read),
    FIELD_T(yvex_artifact_physical_compatibility, writer_plan_identity),
    FIELD_T(yvex_artifact_physical_compatibility, admitted_writer_plan_identity),
    FIELD_T(yvex_artifact_physical_compatibility, artifact_identity),
    FIELD_T(yvex_artifact_physical_compatibility, payload_identity),
    FIELD_T(yvex_artifact_physical_compatibility, writer_transform_identity),
    FIELD_T(yvex_artifact_physical_compatibility, admitted_transform_identity),
    FIELD_T(yvex_artifact_physical_compatibility, writer_profile_identity),
    FIELD_T(yvex_artifact_physical_compatibility, admitted_profile_identity),
    FIELD_T(yvex_artifact_physical_compatibility, quant_execution_identity),
    FIELD_T(yvex_artifact_physical_compatibility, payload_plan_identity),
    FIELD_T(yvex_artifact_physical_compatibility, payload_byte_identity),
    FIELD_U(yvex_artifact_physical_compatibility, physical_payload_compatible),
    FIELD_U(yvex_artifact_physical_compatibility, artifact_rebuild_required),
    FIELD_U(yvex_artifact_physical_compatibility, materialization_rebuild_required),
    FIELD_U(yvex_artifact_physical_compatibility, tensor_inventory_equal),
    FIELD_U(yvex_artifact_physical_compatibility, qtype_equal),
    FIELD_U(yvex_artifact_physical_compatibility, layout_equal),
    FIELD_U(yvex_artifact_physical_compatibility, offset_equal),
    FIELD_U(yvex_artifact_physical_compatibility, payload_digest_equal),
};
static const binding_field material_summary_fields[] = {
    FIELD_T(yvex_materialization_summary, artifact_identity),
    FIELD_T(yvex_materialization_summary, plan_identity),
    FIELD_U(yvex_materialization_summary, status),
    FIELD_U(yvex_materialization_summary, tensor_count),
    FIELD_U(yvex_materialization_summary, payload_bytes),
    FIELD_U(yvex_materialization_summary, file_bytes),
    FIELD_U(yvex_materialization_summary, file_backed_tensors),
    FIELD_U(yvex_materialization_summary, file_backed_bytes),
    FIELD_U(yvex_materialization_summary, staged_cache_tensors),
    FIELD_U(yvex_materialization_summary, staged_cache_bytes),
    FIELD_U(yvex_materialization_summary, backend_candidate_tensors),
    FIELD_U(yvex_materialization_summary, backend_candidate_bytes),
    FIELD_U(yvex_materialization_summary, mapped_virtual_bytes),
    FIELD_U(yvex_materialization_summary, file_backed_bytes_owned),
    FIELD_U(yvex_materialization_summary, process_resident_bytes),
    FIELD_U(yvex_materialization_summary, pageable_host_bytes),
    FIELD_U(yvex_materialization_summary, pinned_host_bytes),
    FIELD_U(yvex_materialization_summary, backend_allocated_bytes),
    FIELD_U(yvex_materialization_summary, staging_bytes),
    FIELD_U(yvex_materialization_summary, cache_bytes),
    FIELD_U(yvex_materialization_summary, graph_scratch_reserved_bytes),
    FIELD_U(yvex_materialization_summary, kv_reserved_bytes),
    FIELD_U(yvex_materialization_summary, peak_executor_owned_bytes),
    FIELD_U(yvex_materialization_summary, expert_subview_count),
    FIELD_U(yvex_materialization_summary, execution_ready),
    FIELD_A(yvex_materialization_summary, qtype_tensor_counts),
    FIELD_A(yvex_materialization_summary, qtype_bytes),
};
static const binding_field material_record_fields[] = {
    FIELD_T(yvex_materialized_tensor_binding, name),
    FIELD_U(yvex_materialized_tensor_binding, tensor_id),
    FIELD_U(yvex_materialized_tensor_binding, descriptor_index),
    FIELD_U(yvex_materialized_tensor_binding, role),
    FIELD_U(yvex_materialized_tensor_binding, collection),
    FIELD_U(yvex_materialized_tensor_binding, scope),
    FIELD_U(yvex_materialized_tensor_binding, layer_index),
    FIELD_U(yvex_materialized_tensor_binding, predictor_index),
    FIELD_U(yvex_materialized_tensor_binding, expert_count),
    FIELD_U(yvex_materialized_tensor_binding, rank),
    FIELD_A(yvex_materialized_tensor_binding, dims),
    FIELD_U(yvex_materialized_tensor_binding, qtype),
    FIELD_U(yvex_materialized_tensor_binding, storage_class),
    FIELD_U(yvex_materialized_tensor_binding, row_width),
    FIELD_U(yvex_materialized_tensor_binding, row_count),
    FIELD_U(yvex_materialized_tensor_binding, block_size),
    FIELD_U(yvex_materialized_tensor_binding, bytes_per_block),
    FIELD_U(yvex_materialized_tensor_binding, encoded_bytes),
    FIELD_U(yvex_materialized_tensor_binding, absolute_offset),
    FIELD_U(yvex_materialized_tensor_binding, absolute_end_offset),
    FIELD_U(yvex_materialized_tensor_binding, alignment),
    FIELD_U(yvex_materialized_tensor_binding, placement),
    FIELD_U(yvex_materialized_tensor_binding, access_mode),
    FIELD_U(yvex_materialized_tensor_binding, backend_compatible),
};
static const binding_field descriptor_fields[] = {
    FIELD_T(yvex_runtime_descriptor_summary, artifact_identity),
    FIELD_T(yvex_runtime_descriptor_summary, materialization_plan_identity),
    FIELD_T(yvex_runtime_descriptor_summary, logical_model_identity),
    FIELD_T(yvex_runtime_descriptor_summary, runtime_descriptor_identity),
    FIELD_T(yvex_runtime_descriptor_summary, runtime_numeric_identity),
    FIELD_T(yvex_runtime_descriptor_summary, runtime_hadamard_revision),
    FIELD_U(yvex_runtime_descriptor_summary, status),
    FIELD_U(yvex_runtime_descriptor_summary, runtime_numeric_schema_version),
    FIELD_U(yvex_runtime_descriptor_summary, runtime_compute_policy_count),
    FIELD_U(yvex_runtime_descriptor_summary, runtime_activation_policy_count),
    FIELD_U(yvex_runtime_descriptor_summary, runtime_sparse_topk_policy_count),
    FIELD_U(yvex_runtime_descriptor_summary, tensor_count),
    FIELD_U(yvex_runtime_descriptor_summary, payload_bytes),
    FIELD_U(yvex_runtime_descriptor_summary, global_bindings),
    FIELD_U(yvex_runtime_descriptor_summary, main_layer_bindings),
    FIELD_U(yvex_runtime_descriptor_summary, draft_bindings),
    FIELD_U(yvex_runtime_descriptor_summary, routed_expert_bindings),
    FIELD_U(yvex_runtime_descriptor_summary, expert_subview_count),
    FIELD_U(yvex_runtime_descriptor_summary, missing_required_bindings),
    FIELD_U(yvex_runtime_descriptor_summary, duplicate_bindings),
    FIELD_U(yvex_runtime_descriptor_summary, unexpected_bindings),
    FIELD_U(yvex_runtime_descriptor_summary, layer_count),
    FIELD_U(yvex_runtime_descriptor_summary, draft_layer_count),
    FIELD_U(yvex_runtime_descriptor_summary, routed_experts),
    FIELD_U(yvex_runtime_descriptor_summary, experts_per_token),
    FIELD_U(yvex_runtime_descriptor_summary, vocabulary_size),
    FIELD_U(yvex_runtime_descriptor_summary, tokenizer_metadata_available),
    FIELD_U(yvex_runtime_descriptor_summary, graph_execution_ready),
    FIELD_U(yvex_runtime_descriptor_summary, generation_ready),
    FIELD_A(yvex_runtime_descriptor_summary, qtype_tensor_counts),
    FIELD_A(yvex_runtime_descriptor_summary, qtype_bytes),
    FIELD_A(yvex_runtime_descriptor_summary, role_counts),
};
static const binding_field runtime_record_fields[] = {
    FIELD_U(yvex_runtime_tensor_binding, tensor_id),
    FIELD_U(yvex_runtime_tensor_binding, descriptor_index),
    FIELD_U(yvex_runtime_tensor_binding, role),
    FIELD_U(yvex_runtime_tensor_binding, collection),
    FIELD_U(yvex_runtime_tensor_binding, scope),
    FIELD_U(yvex_runtime_tensor_binding, layer_index),
    FIELD_U(yvex_runtime_tensor_binding, predictor_index),
    FIELD_U(yvex_runtime_tensor_binding, qtype),
    FIELD_U(yvex_runtime_tensor_binding, placement),
    FIELD_U(yvex_runtime_tensor_binding, access_mode),
};
static const binding_field activation_fields[] = {
    FIELD_U(yvex_attention_activation_policy, required),
    FIELD_U(yvex_attention_activation_policy, stage),
    FIELD_U(yvex_attention_activation_policy, quantization),
    FIELD_U(yvex_attention_activation_policy, block_axis),
    FIELD_U(yvex_attention_activation_policy, block_width),
    FIELD_U(yvex_attention_activation_policy, scale_format),
    FIELD_U(yvex_attention_activation_policy, scale_dtype),
    FIELD_U(yvex_attention_activation_policy, pre_transform),
    FIELD_U(yvex_attention_activation_policy, tail_policy),
    FIELD_U(yvex_attention_activation_policy, nonfinite_policy),
    FIELD_U(yvex_attention_activation_policy, fake_quant_inplace),
    FIELD_U(yvex_attention_activation_policy, zero_pad_hadamard_to_power_of_two),
};
static const binding_field topk_fields[] = {
    FIELD_U(yvex_attention_topk_policy, required),
    FIELD_U(yvex_attention_topk_policy, version),
    FIELD_U(yvex_attention_topk_policy, policy),
    FIELD_U(yvex_attention_topk_policy, k),
    FIELD_U(yvex_attention_topk_policy, reject_nonfinite),
    FIELD_U(yvex_attention_topk_policy, score_descending),
    FIELD_U(yvex_attention_topk_policy, equal_score_ordinal_ascending),
    FIELD_U(yvex_attention_topk_policy, plus_zero_equals_minus_zero),
    FIELD_U(yvex_attention_topk_policy, duplicate_ordinal_refused),
    FIELD_U(yvex_attention_topk_policy, output_ranked_order),
};
static const binding_field attention_summary_fields[] = {
    FIELD_T(yvex_attention_summary, artifact_identity),
    FIELD_T(yvex_attention_summary, materialization_plan_identity),
    FIELD_T(yvex_attention_summary, logical_model_identity),
    FIELD_T(yvex_attention_summary, runtime_descriptor_identity),
    FIELD_T(yvex_attention_summary, runtime_numeric_identity),
    FIELD_T(yvex_attention_summary, attention_plan_identity),
    FIELD_U(yvex_attention_summary, status),
    FIELD_U(yvex_attention_summary, tensor_scope),
    FIELD_U(yvex_attention_summary, layer_count),
    FIELD_U(yvex_attention_summary, auxiliary_layer_count),
    FIELD_U(yvex_attention_summary, swa_layer_count),
    FIELD_U(yvex_attention_summary, csa_layer_count),
    FIELD_U(yvex_attention_summary, hca_layer_count),
    FIELD_U(yvex_attention_summary, required_binding_count),
    FIELD_U(yvex_attention_summary, required_envelope_binding_count),
    FIELD_U(yvex_attention_summary, missing_binding_count),
    FIELD_U(yvex_attention_summary, qtype_compute_refusal_count),
    FIELD_U(yvex_attention_summary, payload_bytes_bound),
    FIELD_U(yvex_attention_summary, history_contract_ready),
    FIELD_U(yvex_attention_summary, state_delta_contract_ready),
    FIELD_U(yvex_attention_summary, cpu_reference_ready),
    FIELD_U(yvex_attention_summary, cuda_execution_ready),
    FIELD_U(yvex_attention_summary, full_execution_ready),
    FIELD_A(yvex_attention_summary, qtype_binding_counts),
};
static const binding_field capability_fields[] = {
    FIELD_U(yvex_runtime_capabilities, attention_semantics_ready),
    FIELD_U(yvex_runtime_capabilities, attention_core_ready),
    FIELD_U(yvex_runtime_capabilities, attention_envelope_ready),
    FIELD_U(yvex_runtime_capabilities, cpu_prefill_eager_ready),
    FIELD_U(yvex_runtime_capabilities, cpu_decode_eager_ready),
    FIELD_U(yvex_runtime_capabilities, cuda_prefill_eager_ready),
    FIELD_U(yvex_runtime_capabilities, cuda_decode_eager_ready),
    FIELD_U(yvex_runtime_capabilities, cuda_eager_implemented),
    FIELD_U(yvex_runtime_capabilities, cuda_piecewise_graph_implemented),
    FIELD_U(yvex_runtime_capabilities, cuda_full_graph_implemented),
    FIELD_U(yvex_runtime_capabilities, cuda_prefill_piecewise_graph_ready),
    FIELD_U(yvex_runtime_capabilities, cuda_decode_piecewise_graph_ready),
    FIELD_U(yvex_runtime_capabilities, cuda_prefill_full_graph_ready),
    FIELD_U(yvex_runtime_capabilities, cuda_decode_full_graph_ready),
    FIELD_U(yvex_runtime_capabilities, attention_weight_residency_ready),
    FIELD_U(yvex_runtime_capabilities, attention_workspace_ready),
    FIELD_U(yvex_runtime_capabilities, attention_state_delta_ready),
    FIELD_U(yvex_runtime_capabilities, attention_operator_ready),
    FIELD_U(yvex_runtime_capabilities, attention_trace_ready),
    FIELD_U(yvex_runtime_capabilities, attention_profile_ready),
    FIELD_U(yvex_runtime_capabilities, attention_benchmark_ready),
    FIELD_U(yvex_runtime_capabilities, mixed_attention_ready),
    FIELD_U(yvex_runtime_capabilities, speculative_attention_ready),
    FIELD_U(yvex_runtime_capabilities, persistent_kv_ready),
    FIELD_U(yvex_runtime_capabilities, moe_plan_ready),
    FIELD_U(yvex_runtime_capabilities, moe_router_ready),
    FIELD_U(yvex_runtime_capabilities, moe_routed_expert_ready),
    FIELD_U(yvex_runtime_capabilities, moe_shared_expert_ready),
    FIELD_U(yvex_runtime_capabilities, moe_block_ready),
    FIELD_U(yvex_runtime_capabilities, transformer_ready),
    FIELD_U(yvex_runtime_capabilities, output_head_binding_ready),
    FIELD_U(yvex_runtime_capabilities, output_head_projection_ready),
    FIELD_U(yvex_runtime_capabilities, logits_cpu_ready),
    FIELD_U(yvex_runtime_capabilities, logits_cuda_ready),
    FIELD_U(yvex_runtime_capabilities, logits_prefill_ready),
    FIELD_U(yvex_runtime_capabilities, logits_decode_ready),
    FIELD_U(yvex_runtime_capabilities, logits_full_vocabulary_ready),
    FIELD_U(yvex_runtime_capabilities, logits_hidden_contract_ready),
    FIELD_U(yvex_runtime_capabilities, logits_partial_progress_ready),
    FIELD_U(yvex_runtime_capabilities, logits_ready),
    FIELD_U(yvex_runtime_capabilities, generation_ready),
};
static const binding_field transformer_policy_fields[] = {
    FIELD_U(yvex_transformer_family_policy, schema_version),
    FIELD_U(yvex_transformer_family_policy, initial_policy),
    FIELD_U(yvex_transformer_family_policy, final_policy),
    FIELD_U(yvex_transformer_family_policy, residual_streams),
    FIELD_U(yvex_transformer_family_policy, hidden_width),
    FIELD_U(yvex_transformer_family_policy, expanded_width),
    FIELD_U(yvex_transformer_family_policy, maximum_context),
    FIELD_U(yvex_transformer_family_policy, sinkhorn_iterations),
    FIELD_F(yvex_transformer_family_policy, mhc_epsilon),
    FIELD_F(yvex_transformer_family_policy, output_norm_epsilon),
    FIELD_U(yvex_transformer_family_policy, attention_then_moe),
    FIELD_U(yvex_transformer_family_policy, deferred_ffn_post),
    FIELD_U(yvex_transformer_family_policy, final_norm_after_head),
};
static const binding_field logits_policy_fields[] = {
    FIELD_U(yvex_logits_family_policy, schema_version),
    FIELD_U(yvex_logits_family_policy, separate_output_head),
    FIELD_U(yvex_logits_family_policy, tied_output_head),
    FIELD_U(yvex_logits_family_policy, output_head_bias),
};
static const binding_field speculation_policy_fields[] = {
    FIELD_U(yvex_speculation_family_policy, schema_version),
    FIELD_U(yvex_speculation_family_policy, block_size),
    FIELD_U(yvex_speculation_family_policy, noise_token_id),
    FIELD_U(yvex_speculation_family_policy, target_feature_layer_count),
    FIELD_A(yvex_speculation_family_policy, target_feature_layers),
    FIELD_U(yvex_speculation_family_policy, target_feature_width),
    FIELD_U(yvex_speculation_family_policy, concatenated_feature_width),
    FIELD_U(yvex_speculation_family_policy, draft_layer_count),
    FIELD_U(yvex_speculation_family_policy, markov_rank),
    FIELD_U(yvex_speculation_family_policy, accepted_prefix_maximum),
    FIELD_U(yvex_speculation_family_policy, feature_projection_role),
    FIELD_U(yvex_speculation_family_policy, feature_norm_role),
    FIELD_U(yvex_speculation_family_policy, output_norm_role),
    FIELD_U(yvex_speculation_family_policy, markov_embedding_role),
    FIELD_U(yvex_speculation_family_policy, markov_output_role),
    FIELD_U(yvex_speculation_family_policy, confidence_role),
    FIELD_U(yvex_speculation_family_policy, parallel_block_backbone),
    FIELD_U(yvex_speculation_family_policy, sequential_markov),
    FIELD_U(yvex_speculation_family_policy, confidence_available),
    FIELD_U(yvex_speculation_family_policy, shares_embedding),
    FIELD_U(yvex_speculation_family_policy, shares_output_head),
    FIELD_U(yvex_speculation_family_policy, target_verification_required),
    FIELD_T(yvex_speculation_family_policy, policy_identity),
};
typedef yvex_physical_execution_summary physical_summary;
typedef yvex_physical_execution_decision physical_decision;
#define PHYSICAL_U(member) FIELD_U(physical_decision, member)
#define PHYSICAL_T(member) FIELD_T(physical_decision, member)
static const binding_field physical_summary_fields[] = {
    FIELD_U(physical_summary, schema_version), FIELD_U(physical_summary, decision_count),
    FIELD_U(physical_summary, encoded_bytes), FIELD_A(physical_summary, consumer_counts),
    FIELD_A(physical_summary, layout_counts),
    FIELD_T(physical_summary, physical_variant_identity), FIELD_T(physical_summary, identity),
};
static const binding_field physical_decision_fields[] = {
    PHYSICAL_U(schema_version), PHYSICAL_U(terminal_tensor_id), PHYSICAL_U(role), PHYSICAL_U(scope),
    PHYSICAL_U(layer_index), PHYSICAL_U(predictor_index), PHYSICAL_U(expert_count),
    PHYSICAL_U(canonical_qtype), PHYSICAL_U(canonical_row_width),
    PHYSICAL_U(canonical_row_count), PHYSICAL_U(encoded_offset), PHYSICAL_U(encoded_bytes),
    PHYSICAL_U(alignment), PHYSICAL_U(consumer), PHYSICAL_U(layout), PHYSICAL_U(sharing),
    PHYSICAL_T(terminal_identity), PHYSICAL_T(decision_identity),
};
#undef PHYSICAL_U
#undef PHYSICAL_T

#define PHYSICAL_V14_U(member) FIELD_U(yvex_runtime_binding_physical_decision_v14, member)
#define PHYSICAL_V14_T(member) FIELD_T(yvex_runtime_binding_physical_decision_v14, member)
static const binding_field physical_summary_v14_fields[] = {
    FIELD_U(yvex_runtime_binding_physical_summary_v14, schema_version),
    FIELD_U(yvex_runtime_binding_physical_summary_v14, decision_count),
    FIELD_U(yvex_runtime_binding_physical_summary_v14, encoded_bytes),
    FIELD_A(yvex_runtime_binding_physical_summary_v14, consumer_counts),
    FIELD_A(yvex_runtime_binding_physical_summary_v14, layout_counts),
    FIELD_A(yvex_runtime_binding_physical_summary_v14, placement_counts),
    FIELD_T(yvex_runtime_binding_physical_summary_v14, physical_variant_identity),
    FIELD_T(yvex_runtime_binding_physical_summary_v14, identity),
};
static const binding_field physical_decision_v14_fields[] = {
    PHYSICAL_V14_U(schema_version), PHYSICAL_V14_U(terminal_tensor_id),
    PHYSICAL_V14_U(role), PHYSICAL_V14_U(scope), PHYSICAL_V14_U(layer_index),
    PHYSICAL_V14_U(predictor_index), PHYSICAL_V14_U(expert_count),
    PHYSICAL_V14_U(canonical_qtype), PHYSICAL_V14_U(canonical_row_width),
    PHYSICAL_V14_U(canonical_row_count), PHYSICAL_V14_U(encoded_offset),
    PHYSICAL_V14_U(encoded_bytes), PHYSICAL_V14_U(alignment),
    PHYSICAL_V14_U(consumer), PHYSICAL_V14_U(layout), PHYSICAL_V14_U(placement),
    PHYSICAL_V14_U(sharing), PHYSICAL_V14_U(activation),
    PHYSICAL_V14_U(supported_width_mask), PHYSICAL_V14_U(maximum_context),
    PHYSICAL_V14_U(worklist_width_mask), PHYSICAL_V14_U(tensor_core_minimum),
    PHYSICAL_V14_U(required_backend), PHYSICAL_V14_U(required_compute_major),
    PHYSICAL_V14_U(required_compute_minor), PHYSICAL_V14_U(evidence),
    PHYSICAL_V14_U(fallback), PHYSICAL_V14_U(derived_asset_required),
    PHYSICAL_V14_T(kernel_family), PHYSICAL_V14_T(tensor_core_kernel_family),
    PHYSICAL_V14_T(terminal_identity), PHYSICAL_V14_T(decision_identity),
};
#undef PHYSICAL_V14_U
#undef PHYSICAL_V14_T
int yvex_runtime_capabilities_contract_valid(const yvex_runtime_capabilities *facts)
{
    return facts && (!facts->attention_core_ready || facts->attention_semantics_ready) &&
           (!facts->attention_envelope_ready || facts->attention_core_ready) &&
           (!(facts->cpu_prefill_eager_ready || facts->cpu_decode_eager_ready ||
              facts->cuda_eager_implemented) || facts->attention_core_ready) &&
           (!(facts->cuda_piecewise_graph_implemented || facts->cuda_full_graph_implemented) ||
            facts->cuda_eager_implemented) &&
           (!(facts->attention_operator_ready || facts->attention_trace_ready ||
              facts->attention_profile_ready || facts->attention_benchmark_ready) ||
            facts->attention_core_ready) && !facts->cuda_prefill_eager_ready &&
           !facts->cuda_decode_eager_ready && !facts->cuda_prefill_piecewise_graph_ready &&
           !facts->cuda_decode_piecewise_graph_ready && !facts->cuda_prefill_full_graph_ready &&
           !facts->cuda_decode_full_graph_ready && !facts->attention_weight_residency_ready &&
           !facts->attention_workspace_ready && !facts->mixed_attention_ready &&
           !facts->speculative_attention_ready && !facts->persistent_kv_ready &&
           (!facts->moe_router_ready || facts->moe_plan_ready) &&
           (!facts->moe_routed_expert_ready || facts->moe_router_ready) &&
           (!facts->moe_shared_expert_ready || facts->moe_plan_ready) &&
           (!facts->moe_block_ready ||
            (facts->moe_routed_expert_ready && facts->moe_shared_expert_ready)) &&
           (!facts->transformer_ready || facts->moe_block_ready) &&
           (!facts->output_head_projection_ready || facts->output_head_binding_ready) &&
           (!(facts->logits_cpu_ready || facts->logits_cuda_ready ||
              facts->logits_prefill_ready || facts->logits_decode_ready ||
              facts->logits_full_vocabulary_ready || facts->logits_hidden_contract_ready ||
              facts->logits_partial_progress_ready) ||
            facts->output_head_projection_ready) &&
           (!facts->logits_ready ||
            (facts->transformer_ready && facts->output_head_binding_ready &&
             facts->output_head_projection_ready && facts->logits_cpu_ready &&
             facts->logits_cuda_ready && facts->logits_prefill_ready &&
             facts->logits_decode_ready && facts->logits_full_vocabulary_ready &&
             facts->logits_hidden_contract_ready && facts->logits_partial_progress_ready)) &&
           !facts->generation_ready;
}
/* Hash a pre-admission capability contract field-by-field without native padding. */
int yvex_runtime_capabilities_identity(
    const yvex_runtime_capabilities *facts,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long value;
    size_t index;
    if (output) output[0] = '\0';
    if (!facts || !output || !yvex_runtime_capabilities_contract_valid(facts)) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.execution-capabilities.v2") ||
        !yvex_sha256_update_u64(
            &hash, YVEX_RUNTIME_EXECUTION_CAPABILITY_SCHEMA_V2))
        return 0;
    for (index = 0u; index < FIELD_COUNT(capability_fields); ++index) {
        const binding_field *field = &capability_fields[index];
        const unsigned char *address = (const unsigned char *)facts + field->offset;
        if (!field_unsigned_load(address, field->width, &value) || value > 1ull ||
            !yvex_sha256_update_u64(&hash, value))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
int yvex_runtime_capabilities_admitted_by(const yvex_runtime_capabilities *facts,
                                          const yvex_runtime_capabilities *maximum)
{
    size_t index;
    if (!yvex_runtime_capabilities_contract_valid(facts) ||
        !yvex_runtime_capabilities_contract_valid(maximum)) return 0;
    for (index = 0u; index < FIELD_COUNT(capability_fields); ++index) {
        const binding_field *field = &capability_fields[index];
        unsigned long long actual, admitted;
        if (!field_unsigned_load((const unsigned char *)facts + field->offset,
                                 field->width, &actual) ||
            !field_unsigned_load((const unsigned char *)maximum + field->offset,
                                 field->width, &admitted) || actual > admitted)
            return 0;
    }
    return 1;
}
static int binding_moe_unavailable_identity(
    const yvex_runtime_binding_prepare_request *request,
    const yvex_runtime_descriptor_summary *descriptor,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.moe.plan.unavailable.v1") ||
        !yvex_sha256_update_u64(&hash, request->family_adapter_id) ||
        !yvex_sha256_update_u64(&hash, request->family_adapter_version) ||
        !yvex_sha256_update_text(&hash, descriptor->runtime_descriptor_identity) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
static const binding_field layer_prefix_fields[] = {
    FIELD_U(yvex_attention_layer_plan, ordinal),
    FIELD_U(yvex_attention_layer_plan, layer_index),
    FIELD_U(yvex_attention_layer_plan, predictor_index),
    FIELD_U(yvex_attention_layer_plan, tensor_scope),
    FIELD_U(yvex_attention_layer_plan, attention_class),
    FIELD_U(yvex_attention_layer_plan, compute_contract),
    FIELD_U(yvex_attention_layer_plan, compression_ratio),
    FIELD_U(yvex_attention_layer_plan, sliding_window),
    FIELD_U(yvex_attention_layer_plan, query_heads),
    FIELD_U(yvex_attention_layer_plan, kv_heads),
    FIELD_U(yvex_attention_layer_plan, head_dimension),
    FIELD_U(yvex_attention_layer_plan, rope_head_dimension),
    FIELD_U(yvex_attention_layer_plan, query_lora_rank),
    FIELD_U(yvex_attention_layer_plan, output_lora_rank),
    FIELD_U(yvex_attention_layer_plan, output_groups),
    FIELD_U(yvex_attention_layer_plan, output_group_input_width),
    FIELD_U(yvex_attention_layer_plan, hidden_dimension),
    FIELD_U(yvex_attention_layer_plan, indexer_heads),
    FIELD_U(yvex_attention_layer_plan, indexer_head_dimension),
    FIELD_U(yvex_attention_layer_plan, indexer_topk),
    FIELD_U(yvex_attention_layer_plan, compressor_ape_columns),
    FIELD_U(yvex_attention_layer_plan, indexer_ape_columns),
    FIELD_F(yvex_attention_layer_plan, rms_norm_epsilon),
    FIELD_U(yvex_attention_layer_plan, residual_stream_count),
    FIELD_U(yvex_attention_layer_plan, residual_stream_width),
    FIELD_U(yvex_attention_layer_plan, residual_expanded_width),
    FIELD_U(yvex_attention_layer_plan, mhc_mixing_rows),
    FIELD_U(yvex_attention_layer_plan, mhc_mixing_columns),
    FIELD_U(yvex_attention_layer_plan, mhc_base_width),
    FIELD_U(yvex_attention_layer_plan, mhc_scale_width),
    FIELD_U(yvex_attention_layer_plan, mhc_sinkhorn_iterations),
    FIELD_U(yvex_attention_layer_plan, attention_input_norm_width),
    FIELD_U(yvex_attention_layer_plan, mhc_entry_policy),
    FIELD_U(yvex_attention_layer_plan, mhc_attention_pre_and_post),
    FIELD_U(yvex_attention_layer_plan, attention_input_norm_required),
    FIELD_U(yvex_attention_layer_plan, attention_input_norm_role),
    FIELD_U(yvex_attention_layer_plan, mhc_function_role),
    FIELD_U(yvex_attention_layer_plan, mhc_base_role),
    FIELD_U(yvex_attention_layer_plan, mhc_scale_role),
    FIELD_U(yvex_attention_layer_plan, compressor_required),
    FIELD_U(yvex_attention_layer_plan, indexer_required),
    FIELD_F(yvex_attention_layer_plan, mhc_epsilon),
    FIELD_F(yvex_attention_layer_plan, mhc_residual_post_multiplier),
    FIELD_N(yvex_attention_layer_plan, position, yvex_attention_position_policy,
            rope_dimension, BINDING_FIELD_UNSIGNED),
    FIELD_N(yvex_attention_layer_plan, position, yvex_attention_position_policy,
            theta, BINDING_FIELD_UNSIGNED),
    FIELD_N(yvex_attention_layer_plan, position, yvex_attention_position_policy,
            scaling_factor, BINDING_FIELD_UNSIGNED),
    FIELD_N(yvex_attention_layer_plan, position, yvex_attention_position_policy,
            original_context, BINDING_FIELD_UNSIGNED),
    FIELD_N(yvex_attention_layer_plan, position, yvex_attention_position_policy,
            beta_fast, BINDING_FIELD_UNSIGNED),
    FIELD_N(yvex_attention_layer_plan, position, yvex_attention_position_policy,
            beta_slow, BINDING_FIELD_UNSIGNED),
    FIELD_N(yvex_attention_layer_plan, position, yvex_attention_position_policy,
            maximum_context, BINDING_FIELD_UNSIGNED),
    FIELD_N(yvex_attention_layer_plan, position, yvex_attention_position_policy,
            partial_rope, BINDING_FIELD_UNSIGNED),
    FIELD_N(yvex_attention_layer_plan, position, yvex_attention_position_policy,
            inverse_output_rotation, BINDING_FIELD_UNSIGNED),
};
static const binding_field layer_tail_fields[] = {
    FIELD_U(yvex_attention_layer_plan, required_binding_count),
    FIELD_U(yvex_attention_layer_plan, qtype_compute_refusal_count),
    FIELD_U(yvex_attention_layer_plan, payload_bytes_bound),
};
static int write_attention_layer(binding_bytes *bytes, const yvex_attention_layer_plan *value)
{
    return fields_write(bytes, value, layer_prefix_fields, FIELD_COUNT(layer_prefix_fields)) &&
           fields_write(bytes, &value->attention_kv_activation,
                        activation_fields, FIELD_COUNT(activation_fields)) &&
           fields_write(bytes, &value->compressor_activation,
                        activation_fields, FIELD_COUNT(activation_fields)) &&
           fields_write(bytes, &value->compressor_rotated_activation,
                        activation_fields, FIELD_COUNT(activation_fields)) &&
           fields_write(bytes, &value->indexer_query_activation,
                        activation_fields, FIELD_COUNT(activation_fields)) &&
           fields_write(bytes, &value->sparse_topk, topk_fields, FIELD_COUNT(topk_fields)) &&
           fields_write(bytes, value, layer_tail_fields, FIELD_COUNT(layer_tail_fields));
}
static int read_attention_layer(binding_cursor *cursor, yvex_attention_layer_plan *value)
{
    memset(value, 0, sizeof(*value));
    return fields_read(cursor, value, layer_prefix_fields, FIELD_COUNT(layer_prefix_fields)) &&
           fields_read(cursor, &value->attention_kv_activation,
                       activation_fields, FIELD_COUNT(activation_fields)) &&
           fields_read(cursor, &value->compressor_activation,
                       activation_fields, FIELD_COUNT(activation_fields)) &&
           fields_read(cursor, &value->compressor_rotated_activation,
                       activation_fields, FIELD_COUNT(activation_fields)) &&
           fields_read(cursor, &value->indexer_query_activation,
                       activation_fields, FIELD_COUNT(activation_fields)) &&
           fields_read(cursor, &value->sparse_topk, topk_fields, FIELD_COUNT(topk_fields)) &&
           fields_read(cursor, value, layer_tail_fields, FIELD_COUNT(layer_tail_fields));
}
static int write_physical_execution(binding_bytes *body,
                                    const yvex_physical_execution_ir *physical)
{
    const physical_summary *summary = yvex_physical_execution_ir_summary(physical);
    unsigned long long index;
    if (!summary || !fields_write(body, summary, physical_summary_fields,
                                  FIELD_COUNT(physical_summary_fields))) return 0;
    for (index = 0ull; index < summary->decision_count; ++index) {
        const physical_decision *decision =
            yvex_physical_execution_ir_decision_at(physical, index);
        if (!decision || !fields_write(body, decision, physical_decision_fields,
                                       FIELD_COUNT(physical_decision_fields))) return 0;
    }
    return 1;
}
static binding_parse_result read_physical_execution_v15(
    binding_cursor *cursor, yvex_physical_execution_ir **out,
    unsigned long long expected_count)
{
    physical_summary summary;
    physical_decision *decisions;
    yvex_error err;
    unsigned long long index;
    if (!record_read(cursor, &summary, sizeof(summary), physical_summary_fields,
                     FIELD_COUNT(physical_summary_fields)) ||
        summary.decision_count != expected_count ||
        !record_count_fits(cursor, expected_count, sizeof(*decisions)))
        return BINDING_PARSE_FORMAT;
    decisions = calloc((size_t)expected_count, sizeof(*decisions));
    if (!decisions) return BINDING_PARSE_ALLOCATION;
    for (index = 0ull; index < expected_count; ++index)
        if (!record_read(cursor, &decisions[index], sizeof(decisions[index]),
                         physical_decision_fields, FIELD_COUNT(physical_decision_fields))) {
            free(decisions);
            return BINDING_PARSE_FORMAT;
        }
    if (yvex_physical_execution_ir_import(out, &summary, decisions,
                                          expected_count, &err) != YVEX_OK) {
        free(decisions);
        return BINDING_PARSE_FORMAT;
    }
    free(decisions);
    return BINDING_PARSE_OK;
}

static binding_parse_result read_physical_execution_v14(
    binding_cursor *cursor, yvex_physical_execution_ir **out,
    unsigned long long expected_count)
{
    yvex_runtime_binding_physical_summary_v14 summary;
    yvex_runtime_binding_physical_decision_v14 *legacy = NULL;
    yvex_error err;
    unsigned long long index;
    binding_parse_result result = BINDING_PARSE_FORMAT;
    if (!record_read(cursor, &summary, sizeof(summary), physical_summary_v14_fields,
                     FIELD_COUNT(physical_summary_v14_fields)) ||
        summary.decision_count != expected_count ||
        !record_count_fits(cursor, expected_count, sizeof(*legacy)))
        return BINDING_PARSE_FORMAT;
    legacy = calloc((size_t)expected_count, sizeof(*legacy));
    if (!legacy) return BINDING_PARSE_ALLOCATION;
    for (index = 0ull; index < expected_count; ++index)
        if (!record_read(cursor, &legacy[index], sizeof(legacy[index]),
                         physical_decision_v14_fields,
                         FIELD_COUNT(physical_decision_v14_fields)))
            goto done;
    if (yvex_runtime_private_binding_physical_v14_import(
            out, &summary, legacy, expected_count, &err) == YVEX_OK)
        result = BINDING_PARSE_OK;
done:
    free(legacy);
    return result;
}
static int binding_admission_ready(const yvex_complete_artifact_admission *admission) {
    return admission && admission->complete && admission->materialization_input_ready &&
           !admission->runtime_supported && admission->artifact_identity_verified &&
           admission->file_snapshot.size == admission->file_bytes;
}
static int binding_attention_ready(const yvex_attention_summary *attention) {
    return attention && attention->history_contract_ready &&
           attention->state_delta_contract_ready && attention->cpu_reference_ready &&
           attention->cuda_execution_ready && attention->full_execution_ready;
}
static int binding_identity_chain_valid(
    const yvex_complete_artifact_admission *admission,
    const yvex_materialization_summary *materialization,
    const yvex_runtime_descriptor_summary *descriptor,
    const yvex_attention_summary *attention)
{
    return admission && materialization && descriptor && attention &&
           strcmp(admission->artifact_identity, materialization->artifact_identity) == 0 &&
           strcmp(materialization->artifact_identity, descriptor->artifact_identity) == 0 &&
           strcmp(materialization->plan_identity,
                  descriptor->materialization_plan_identity) == 0 &&
           strcmp(descriptor->artifact_identity, attention->artifact_identity) == 0 &&
           strcmp(descriptor->materialization_plan_identity,
                  attention->materialization_plan_identity) == 0 &&
           strcmp(descriptor->logical_model_identity, attention->logical_model_identity) == 0 &&
           strcmp(descriptor->runtime_descriptor_identity,
                  attention->runtime_descriptor_identity) == 0 &&
           strcmp(descriptor->runtime_numeric_identity,
                  attention->runtime_numeric_identity) == 0;
}
/* Validate the sealed identity chain before external serialization starts. */
static int prepare_validate(const yvex_runtime_binding_prepare_request *request,
                            char semantic[YVEX_SHA256_HEX_CAP],
                            char executable[YVEX_SHA256_HEX_CAP],
                            char moe_identity[YVEX_SHA256_HEX_CAP],
                            char draft_moe_identity[YVEX_SHA256_HEX_CAP],
                            yvex_runtime_binding_failure *failure, yvex_error *err)
{
    const yvex_materialization_summary *materialization;
    const yvex_runtime_descriptor_summary *descriptor;
    const yvex_attention_summary *attention;
    const yvex_attention_summary *draft_attention;
    const yvex_operator_graph_summary *operators;
    const physical_summary *physical;
    const char *compiled_graph;
    const char *compatibility_mismatch;
    if (!request || !request->directory || !request->directory[0] ||
        !request->admission || !request->operator_graph ||
        !request->physical_execution || !request->compiled_plan ||
        !request->materialization || !request->runtime_descriptor || !request->attention_plan ||
        !request->family_adapter_id || !request->family_adapter_version ||
        !request->artifact_format || !request->artifact_format[0] ||
        strlen(request->artifact_format) >= sizeof(((yvex_runtime_binding_summary *)0)->artifact_format) ||
        !request->artifact_format_version ||
        !yvex_sha256_hex_is_valid(request->logical_transform_identity) ||
        !semantic || !executable || !moe_identity || !draft_moe_identity)
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT, "request",
            request ? request->directory : NULL, 0ull, 1ull, 0ull, YVEX_ERR_INVALID_ARG,
            "runtime binding preparation requires complete typed inputs", err);
    if (!yvex_runtime_capabilities_contract_valid(&request->capabilities))
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY,
            "execution-capabilities", request->directory, 0ull, 1ull, 0ull,
            YVEX_ERR_STATE,
            "runtime binding capabilities are not a valid compiled envelope", err);
    materialization = yvex_materialization_session_summary(request->materialization);
    descriptor = yvex_runtime_descriptor_summary_get(request->runtime_descriptor);
    operators = yvex_operator_graph_ir_summary(request->operator_graph);
    physical = yvex_physical_execution_ir_summary(request->physical_execution);
    compiled_graph = yvex_compiled_model_plan_operator_graph_identity(
        request->compiled_plan);
    attention = yvex_attention_plan_summary(request->attention_plan);
    draft_attention = yvex_attention_plan_summary(request->draft_attention_plan);
    if (!descriptor || !operators ||
        operators->family_adapter_id != request->family_adapter_id ||
        operators->family_adapter_version != request->family_adapter_version ||
        operators->maximum_context != descriptor->model_execution.maximum_context ||
        operators->target_layer_count != descriptor->model_execution.layer_count ||
        operators->draft_layer_count != descriptor->model_execution.draft_layer_count ||
        !yvex_runtime_private_binding_policies_match_model(
            &descriptor->model_execution, &request->transformer_policy,
            &request->logits_policy, &request->speculation_policy) ||
        request->tokenizer_policy.family_adapter_id != request->family_adapter_id ||
        request->tokenizer_policy.family_adapter_version != request->family_adapter_version ||
        request->tokenizer_policy.vocabulary_size !=
            descriptor->model_execution.vocabulary_size ||
        yvex_tokenizer_family_policy_validate(&request->tokenizer_policy, NULL) != YVEX_OK)
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY,
            "execution-policies", request->directory, 0ull, 1ull, 0ull,
            YVEX_ERR_FORMAT,
            "runtime binding requires a valid compiled execution policy envelope", err);
    compatibility_mismatch = yvex_artifact_physical_compatibility_mismatch(
        request->physical_compatibility, request->admission,
        request->logical_transform_identity);
    if (compatibility_mismatch)
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY,
            compatibility_mismatch, request->directory, 0ull, 1ull, 0ull,
            YVEX_ERR_STATE,
            "runtime binding requires an exact physical compatibility proof", err);
    if (!binding_admission_ready(request->admission) ||
        !materialization || !materialization->committed ||
        materialization->status != YVEX_MATERIALIZATION_STATUS_COMMITTED ||
        !descriptor || descriptor->status != YVEX_RUNTIME_DESCRIPTOR_STATUS_READY ||
        descriptor->model_execution.schema_version !=
            YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
        !binding_attention_ready(attention) ||
        (descriptor && descriptor->draft_layer_count &&
         !binding_attention_ready(draft_attention)))
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_UNSEALED_INPUT, "lifecycle",
            request->directory, 0ull, 1ull, 0ull, YVEX_ERR_STATE,
            "runtime binding inputs must be sealed and admitted", err);
    if (materialization->tensor_count == 0ull ||
        materialization->tensor_count > BINDING_MAX_RECORDS ||
        descriptor->tensor_count != materialization->tensor_count ||
        attention->layer_count == 0ull || attention->layer_count > BINDING_MAX_LAYERS ||
        attention->required_binding_count == 0ull || attention->missing_binding_count != 0ull ||
        attention->qtype_compute_refusal_count != 0ull ||
        yvex_attention_plan_layer_count(request->attention_plan) != attention->layer_count ||
        attention->tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER ||
        (descriptor->layer_count && attention->layer_count != descriptor->layer_count) ||
        (descriptor->draft_layer_count &&
         (!draft_attention || draft_attention->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT ||
          draft_attention->layer_count != descriptor->draft_layer_count ||
          draft_attention->layer_count > BINDING_MAX_LAYERS ||
          draft_attention->required_binding_count == 0ull ||
          draft_attention->missing_binding_count != 0ull ||
          draft_attention->qtype_compute_refusal_count != 0ull ||
          yvex_attention_plan_layer_count(request->draft_attention_plan) !=
              draft_attention->layer_count)) ||
        (!descriptor->draft_layer_count && request->draft_attention_plan))
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_BOUNDS, "record-count",
            request->directory, 0ull, materialization->tensor_count,
            descriptor->tensor_count, YVEX_ERR_BOUNDS,
            "runtime binding record counts are inconsistent or excessive", err);
    if (!binding_identity_chain_valid(request->admission, materialization,
                                      descriptor, attention) ||
        (draft_attention && !binding_identity_chain_valid(
                                request->admission, materialization,
                                descriptor, draft_attention)))
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY, "identity-chain",
            request->directory, 0ull, 1ull, 0ull, YVEX_ERR_STATE,
            "runtime binding inputs do not share one immutable identity chain", err);
    if (!physical || physical->decision_count != materialization->tensor_count ||
        strcmp(physical->physical_variant_identity,
               request->admission->profile_identity) != 0)
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY,
            "physical-execution", request->directory, 0ull,
            materialization->tensor_count, 0ull, YVEX_ERR_STATE,
            "runtime binding requires compiler-sealed physical execution truth", err);
    if (!compiled_graph || strcmp(compiled_graph, operators->identity) != 0)
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
            "operator-graph", request->directory, 0ull, 1ull, 0ull,
            YVEX_ERR_STATE,
            "compiled execution does not bind the canonical operator graph", err);
    if (!yvex_compiled_graph_identities(
            operators->identity,
            materialization, descriptor, attention, draft_attention,
            semantic, executable))
        return binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY, "graph-identity-inputs",
            request->directory, 0ull, 1ull, 0ull, YVEX_ERR_STATE,
            "runtime binding graph identity inputs are incomplete", err);
    if (request->capabilities.moe_plan_ready) {
        const yvex_moe_plan_summary *moe = yvex_moe_plan_summary_get(
            yvex_compiled_model_plan_moe(request->compiled_plan, 0));
        const yvex_moe_plan_summary *draft_moe = yvex_moe_plan_summary_get(
            yvex_compiled_model_plan_moe(request->compiled_plan, 1));
        if (!moe)
            return binding_reject(
                failure, YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY,
                "moe-plan",
                request->directory, 0ull, 1ull, 0ull,
                YVEX_ERR_STATE,
                "runtime binding requires an admitted target MoE plan", err);
        yvex_runtime_identity_copy(moe_identity, moe->moe_plan_identity);
        if (draft_attention && !draft_moe)
            return binding_reject(
                failure, YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY,
                "draft-moe-plan", request->directory, 0ull, 1ull, 0ull,
                YVEX_ERR_STATE,
                "runtime binding requires an admitted draft MoE plan", err);
        if (draft_moe)
            yvex_runtime_identity_copy(
                draft_moe_identity, draft_moe->moe_plan_identity);
    } else if (!binding_moe_unavailable_identity(request, descriptor, moe_identity)) {
        return binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
                              "moe-plan", request->directory, 0ull, 1ull, 0ull,
                              YVEX_ERR_STATE, "unavailable MoE plan identity could not be derived", err);
    }
    if (!draft_attention &&
        !binding_moe_unavailable_identity(request, descriptor, draft_moe_identity))
        return binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
                              "draft-moe-plan", request->directory, 0ull, 1ull, 0ull,
                              YVEX_ERR_STATE, "absent draft MoE identity could not be derived", err);
    yvex_error_clear(err);
    return YVEX_OK;
}
static int binding_body_write(const yvex_runtime_binding_prepare_request *request,
                              const char *semantic, const char *executable,
                              const char *moe_identity,
                              const char *draft_moe_identity,
                              const yvex_physical_execution_ir *physical,
                              const yvex_transformer_family_policy *transformer,
                              const yvex_logits_family_policy *logits,
                              const yvex_speculation_family_policy *speculation,
                              const yvex_core_bytes *tokenizer_policy,
                              const yvex_core_bytes *compiled_plans,
                              binding_bytes *body)
{
    yvex_materialization_summary canonical;
    const yvex_artifact_physical_compatibility *compatibility;
    const yvex_complete_artifact_admission *admission;
    const yvex_materialization_summary *materialization;
    const yvex_runtime_descriptor_summary *descriptor;
    const yvex_attention_summary *attention;
    const yvex_attention_summary *draft_attention;
    const yvex_runtime_capabilities *capabilities;
    const char *format, *logical;
    char capability_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long adapter_id, adapter_version, format_version;
    unsigned long long tensor_count, layer_count, draft_layer_count, i;
    if (!body || !request || !transformer || !logits || !speculation ||
        !tokenizer_policy || !tokenizer_policy->data || !tokenizer_policy->count ||
        !compiled_plans || !compiled_plans->data || !compiled_plans->count)
        return 0;
    body->maximum = BINDING_MAX_BYTES;
    body->initial_capacity = 4096u;
    canonical = *yvex_materialization_session_summary(request->materialization);
    canonical.status = YVEX_MATERIALIZATION_STATUS_PLANNED;
    compatibility = request->physical_compatibility;
    admission = request->admission;
    materialization = &canonical;
    descriptor = yvex_runtime_descriptor_summary_get(request->runtime_descriptor);
    attention = yvex_attention_plan_summary(request->attention_plan);
    draft_attention = yvex_attention_plan_summary(request->draft_attention_plan);
    adapter_id = request->family_adapter_id;
    adapter_version = request->family_adapter_version;
    format = request->artifact_format;
    format_version = request->artifact_format_version;
    logical = request->logical_transform_identity;
    capabilities = &request->capabilities;
    tensor_count = materialization->tensor_count;
    layer_count = attention->layer_count;
    draft_layer_count = draft_attention ? draft_attention->layer_count : 0ull;
    if (!yvex_runtime_capabilities_identity(capabilities, capability_identity) ||
        !bytes_put_text(body, "yvex.runtime.binding.payload.v15") ||
        !bytes_put_u64(body, YVEX_RUNTIME_BINDING_SCHEMA_CURRENT) ||
        !bytes_put_u64(body, adapter_id) || !bytes_put_u64(body, adapter_version) ||
        !bytes_put_text(body, format) || !bytes_put_u64(body, format_version) ||
        !bytes_put_text(body, logical) || !bytes_put_text(body, semantic) ||
        !bytes_put_text(body, executable) || !bytes_put_text(body, moe_identity) ||
        !bytes_put_text(body, draft_moe_identity) ||
        !bytes_put_text(body, capability_identity) ||
        !fields_write(body, capabilities, capability_fields,
                      FIELD_COUNT(capability_fields)) ||
        !fields_write(body, compatibility, physical_compatibility_fields,
                      FIELD_COUNT(physical_compatibility_fields)) ||
        !fields_write(body, admission, admission_fields, FIELD_COUNT(admission_fields)) ||
        !fields_write(body, materialization, material_summary_fields,
                      FIELD_COUNT(material_summary_fields)) ||
        !bytes_put_u64(body, tensor_count))
        return 0;
    for (i = 0ull; i < tensor_count; ++i) {
        const yvex_materialized_tensor_binding *record =
            yvex_materialization_session_tensor_at(request->materialization, i);
        if (!record || !fields_write(body, record, material_record_fields,
                                     FIELD_COUNT(material_record_fields))) return 0;
    }
    if (!fields_write(body, descriptor, descriptor_fields, FIELD_COUNT(descriptor_fields)) ||
        !model_execution_write(body, &descriptor->model_execution) ||
        !bytes_put_u64(body, tensor_count)) return 0;
    for (i = 0ull; i < tensor_count; ++i) {
        const yvex_runtime_tensor_binding *record =
            yvex_runtime_descriptor_tensor_at(request->runtime_descriptor, i);
        if (!record || !fields_write(body, record, runtime_record_fields,
                                     FIELD_COUNT(runtime_record_fields))) return 0;
    }
    if (!write_physical_execution(body, physical) ||
        !fields_write(body, transformer, transformer_policy_fields,
                      FIELD_COUNT(transformer_policy_fields)) ||
        !fields_write(body, logits, logits_policy_fields,
                      FIELD_COUNT(logits_policy_fields)) ||
        !fields_write(body, speculation, speculation_policy_fields,
                      FIELD_COUNT(speculation_policy_fields)) ||
        !bytes_put_u64(body, tokenizer_policy->count) ||
        !yvex_core_bytes_append(body, tokenizer_policy->data, tokenizer_policy->count) ||
        !bytes_put_u64(body, compiled_plans->count) ||
        !yvex_core_bytes_append(body, compiled_plans->data,
                                compiled_plans->count)) return 0;
    if (!fields_write(body, attention, attention_summary_fields,
                      FIELD_COUNT(attention_summary_fields)) ||
        !bytes_put_u64(body, layer_count)) return 0;
    for (i = 0ull; i < layer_count; ++i) {
        const yvex_attention_layer_plan *record =
            yvex_attention_plan_layer_at(request->attention_plan, i);
        if (!record || !write_attention_layer(body, record)) return 0;
    }
    if (!bytes_put_u64(body, draft_attention != NULL)) return 0;
    if (!draft_attention) return 1;
    if (!fields_write(body, draft_attention, attention_summary_fields,
                      FIELD_COUNT(attention_summary_fields)) ||
        !bytes_put_u64(body, draft_layer_count)) return 0;
    for (i = 0ull; i < draft_layer_count; ++i) {
        const yvex_attention_layer_plan *record =
            yvex_attention_plan_layer_at(request->draft_attention_plan, i);
        if (!record || !write_attention_layer(body, record)) return 0;
    }
    return 1;
}
static int binding_identity(unsigned int schema, const unsigned char *body, size_t body_bytes,
                            char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if ((schema != BINDING_SCHEMA_V14 &&
         schema != YVEX_RUNTIME_BINDING_SCHEMA_CURRENT) ||
        !yvex_sha256_update_text(
            &hash, schema == BINDING_SCHEMA_V14
                       ? "yvex.runtime.binding.v14"
                       : "yvex.runtime.binding.v15") ||
        !yvex_sha256_update_u64(&hash, schema) ||
        !yvex_sha256_update(&hash, body, body_bytes) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
static int build_file(const binding_bytes *body, const char *identity, binding_bytes *file)
{
    if (!body || !identity || !file) return 0;
    file->maximum = BINDING_MAX_BYTES;
    file->initial_capacity = 4096u;
    return yvex_core_bytes_append(file, BINDING_MAGIC_V15, BINDING_MAGIC_BYTES) &&
           bytes_put_u64(file, YVEX_RUNTIME_BINDING_SCHEMA_CURRENT) &&
           bytes_put_u64(file, (unsigned long long)body->count) &&
           yvex_core_bytes_append(file, identity, 64u) &&
           yvex_core_bytes_append(file, body->data, body->count);
}
static binding_parse_result parse_body(yvex_runtime_binding *binding,
                                       const unsigned char *data, size_t count,
                                       unsigned int expected_schema)
{
    binding_cursor cursor = {data, count, 0u};
    char domain[64], logical_transform[YVEX_SHA256_HEX_CAP];
    char semantic[YVEX_SHA256_HEX_CAP], executable[YVEX_SHA256_HEX_CAP];
    char moe_identity[YVEX_SHA256_HEX_CAP], draft_moe_identity[YVEX_SHA256_HEX_CAP];
    char capability_identity[YVEX_SHA256_HEX_CAP];
    char format[16];
    unsigned long long schema, family_id, family_version, format_version;
    unsigned long long material_count, runtime_count, layer_count, draft_present;
    unsigned long long tokenizer_policy_bytes, compiled_plan_bytes;
    unsigned long long draft_layer_count = 0ull, i;
    if (!cursor_text(&cursor, domain, sizeof(domain)) ||
        !cursor_u64(&cursor, &schema) || schema != expected_schema ||
        strcmp(domain, expected_schema == BINDING_SCHEMA_V14
                           ? "yvex.runtime.binding.payload.v14"
                           : "yvex.runtime.binding.payload.v15") != 0 ||
        !cursor_u64(&cursor, &family_id) || !family_id ||
        !cursor_u64(&cursor, &family_version) || !family_version ||
        !cursor_text(&cursor, format, sizeof(format)) || !format[0] ||
        !cursor_u64(&cursor, &format_version) || !format_version || format_version > UINT_MAX ||
        !cursor_text(&cursor, logical_transform, sizeof(logical_transform)) ||
        !cursor_text(&cursor, semantic, sizeof(semantic)) ||
        !cursor_text(&cursor, executable, sizeof(executable)) ||
        !cursor_text(&cursor, moe_identity, sizeof(moe_identity)) ||
        !cursor_text(&cursor, draft_moe_identity, sizeof(draft_moe_identity)) ||
        !cursor_text(&cursor, capability_identity, sizeof(capability_identity)) ||
        !yvex_sha256_hex_is_valid(logical_transform) ||
        !yvex_sha256_hex_is_valid(semantic) || !yvex_sha256_hex_is_valid(executable) ||
        !yvex_sha256_hex_is_valid(moe_identity) ||
        !yvex_sha256_hex_is_valid(draft_moe_identity) ||
        !yvex_sha256_hex_is_valid(capability_identity) ||
        !record_read(&cursor, &binding->summary.capabilities,
                     sizeof(binding->summary.capabilities), capability_fields,
                     FIELD_COUNT(capability_fields)) ||
        !record_read(&cursor, &binding->summary.physical_compatibility,
                     sizeof(binding->summary.physical_compatibility),
                     physical_compatibility_fields,
                     FIELD_COUNT(physical_compatibility_fields)) ||
        !record_read(&cursor, &binding->admission, sizeof(binding->admission),
                     admission_fields, FIELD_COUNT(admission_fields)) ||
        !record_read(&cursor, &binding->materialization, sizeof(binding->materialization),
                     material_summary_fields, FIELD_COUNT(material_summary_fields)) ||
        !cursor_u64(&cursor, &material_count))
        return BINDING_PARSE_FORMAT;
    if (material_count > BINDING_MAX_RECORDS ||
        !record_count_fits(&cursor, material_count, sizeof(*binding->materialized)))
        return BINDING_PARSE_BOUNDS;
    binding->materialized = (yvex_materialized_tensor_binding *)calloc(
        (size_t)material_count, sizeof(*binding->materialized));
    if (!binding->materialized) return BINDING_PARSE_ALLOCATION;
    for (i = 0ull; i < material_count; ++i)
        if (!record_read(&cursor, &binding->materialized[i], sizeof(binding->materialized[i]),
                         material_record_fields, FIELD_COUNT(material_record_fields)))
            return BINDING_PARSE_FORMAT;
    if (!record_read(&cursor, &binding->descriptor, sizeof(binding->descriptor),
                     descriptor_fields, FIELD_COUNT(descriptor_fields)) ||
        !model_execution_read(&cursor, &binding->descriptor.model_execution) ||
        !cursor_u64(&cursor, &runtime_count) || runtime_count != material_count)
        return BINDING_PARSE_FORMAT;
    if (!record_count_fits(&cursor, runtime_count, sizeof(*binding->runtime)))
        return BINDING_PARSE_BOUNDS;
    binding->runtime = (yvex_runtime_tensor_binding *)calloc(
        (size_t)runtime_count, sizeof(*binding->runtime));
    if (!binding->runtime) return BINDING_PARSE_ALLOCATION;
    for (i = 0ull; i < runtime_count; ++i)
        if (!record_read(&cursor, &binding->runtime[i], sizeof(binding->runtime[i]),
                         runtime_record_fields, FIELD_COUNT(runtime_record_fields)))
            return BINDING_PARSE_FORMAT;
    if ((expected_schema == BINDING_SCHEMA_V14
             ? read_physical_execution_v14(
                   &cursor, &binding->physical_execution, runtime_count)
             : read_physical_execution_v15(
                   &cursor, &binding->physical_execution, runtime_count)) !=
        BINDING_PARSE_OK)
        return BINDING_PARSE_FORMAT;
    if (!record_read(&cursor, &binding->transformer_policy,
                     sizeof(binding->transformer_policy), transformer_policy_fields,
                     FIELD_COUNT(transformer_policy_fields)) ||
        !record_read(&cursor, &binding->logits_policy,
                     sizeof(binding->logits_policy), logits_policy_fields,
                     FIELD_COUNT(logits_policy_fields)) ||
        !record_read(&cursor, &binding->speculation_policy,
                     sizeof(binding->speculation_policy), speculation_policy_fields,
                     FIELD_COUNT(speculation_policy_fields)) ||
        !cursor_u64(&cursor, &tokenizer_policy_bytes) || !tokenizer_policy_bytes ||
        tokenizer_policy_bytes > cursor.count - cursor.offset ||
        yvex_tokenizer_family_policy_decode(
            &binding->tokenizer_policy, cursor.data + cursor.offset,
            (size_t)tokenizer_policy_bytes, NULL) != YVEX_OK)
        return BINDING_PARSE_FORMAT;
    cursor.offset += (size_t)tokenizer_policy_bytes;
    if (
        !cursor_u64(&cursor, &compiled_plan_bytes) || !compiled_plan_bytes ||
        compiled_plan_bytes > cursor.count - cursor.offset ||
        yvex_compiled_model_plan_decode(
            &binding->plan, cursor.data + cursor.offset,
            (size_t)compiled_plan_bytes, NULL) != YVEX_OK)
        return BINDING_PARSE_FORMAT;
    cursor.offset += (size_t)compiled_plan_bytes;
    if (!record_read(&cursor, &binding->attention, sizeof(binding->attention),
                     attention_summary_fields, FIELD_COUNT(attention_summary_fields)) ||
        !cursor_u64(&cursor, &layer_count))
        return BINDING_PARSE_FORMAT;
    if (layer_count > BINDING_MAX_LAYERS ||
        !record_count_fits(&cursor, layer_count, sizeof(*binding->layers)))
        return BINDING_PARSE_BOUNDS;
    binding->layers = (yvex_attention_layer_plan *)calloc((size_t)layer_count,
                                                           sizeof(*binding->layers));
    if (!binding->layers) return BINDING_PARSE_ALLOCATION;
    for (i = 0ull; i < layer_count; ++i)
        if (!read_attention_layer(&cursor, &binding->layers[i]))
            return BINDING_PARSE_FORMAT;
    if (!cursor_u64(&cursor, &draft_present) || draft_present > 1ull)
        return BINDING_PARSE_FORMAT;
    if (draft_present) {
        if (!record_read(&cursor, &binding->draft_attention,
                         sizeof(binding->draft_attention), attention_summary_fields,
                         FIELD_COUNT(attention_summary_fields)) ||
            !cursor_u64(&cursor, &draft_layer_count))
            return BINDING_PARSE_FORMAT;
        if (!draft_layer_count || draft_layer_count > BINDING_MAX_LAYERS ||
            !record_count_fits(&cursor, draft_layer_count,
                               sizeof(*binding->draft_layers)))
            return BINDING_PARSE_BOUNDS;
        binding->draft_layers = (yvex_attention_layer_plan *)calloc(
            (size_t)draft_layer_count, sizeof(*binding->draft_layers));
        if (!binding->draft_layers) return BINDING_PARSE_ALLOCATION;
        for (i = 0ull; i < draft_layer_count; ++i)
            if (!read_attention_layer(&cursor, &binding->draft_layers[i]))
                return BINDING_PARSE_FORMAT;
    }
    binding->summary.schema_version = (unsigned int)schema;
    binding->summary.family_adapter_id = family_id;
    binding->summary.family_adapter_version = family_version;
    binding->summary.artifact_format_version = (unsigned int)format_version;
    binding->summary.tensor_count = material_count;
    binding->summary.layer_count = layer_count;
    binding->summary.draft_layer_count = draft_layer_count;
    binding->summary.source_snapshot_identity = binding->admission.source_snapshot_identity;
    binding->summary.mapping_identity = binding->admission.mapping_identity;
    yvex_core_text_copy(binding->summary.artifact_format, sizeof(binding->summary.artifact_format), format);
    yvex_runtime_identity_copy(binding->summary.logical_transform_identity,
                               logical_transform);
    yvex_runtime_identity_copy(binding->summary.semantic_graph_identity, semantic);
    yvex_runtime_identity_copy(binding->summary.executable_graph_identity, executable);
    yvex_runtime_identity_copy(binding->summary.moe_plan_identity, moe_identity);
    yvex_runtime_identity_copy(binding->summary.draft_moe_plan_identity,
                               draft_moe_identity);
    {
        const yvex_transformer_plan_summary *transformer =
            yvex_transformer_plan_summary_get(
                yvex_compiled_model_plan_transformer(binding->plan, 0));
        const yvex_transformer_plan_summary *draft_transformer =
            yvex_transformer_plan_summary_get(
                yvex_compiled_model_plan_transformer(binding->plan, 1));
        const yvex_runtime_logits_plan_summary *output =
            yvex_compiled_model_plan_output_head(binding->plan);
        if (transformer)
            yvex_runtime_identity_copy(
                binding->summary.transformer_plan_identity,
                transformer->transformer_plan_identity);
        if (draft_transformer)
            yvex_runtime_identity_copy(
                binding->summary.draft_transformer_plan_identity,
                draft_transformer->transformer_plan_identity);
        if (output)
            yvex_runtime_identity_copy(
                binding->summary.output_head_plan_identity,
                output->output_head_plan_identity);
    }
    yvex_runtime_identity_copy(binding->summary.execution_capability_identity,
                               capability_identity);
    return cursor.offset == cursor.count ? BINDING_PARSE_OK : BINDING_PARSE_FORMAT;
}
/* Validate parsed cross-record identities, lifecycle facts, counts, and canonical ordinals. */
static int binding_validate(const yvex_runtime_binding *binding,
                            const char **field,
                            yvex_runtime_binding_failure_code *code)
{
    char capability_identity[YVEX_SHA256_HEX_CAP];
    char semantic[YVEX_SHA256_HEX_CAP], executable[YVEX_SHA256_HEX_CAP];
    const char *compatibility;
    const physical_summary *physical;
    unsigned long long i;
    *field = "canonical-body";
    *code = YVEX_RUNTIME_BINDING_FAILURE_FORMAT;
    if (!binding) return 0;
    physical = yvex_physical_execution_ir_summary(binding->physical_execution);
    if (!physical || physical->decision_count != binding->summary.tensor_count ||
        strcmp(physical->physical_variant_identity,
               binding->admission.profile_identity) != 0) {
        *field = "physical-execution";
        *code = YVEX_RUNTIME_BINDING_FAILURE_IDENTITY;
        return 0;
    }
    for (i = 0ull; i < physical->decision_count; ++i) {
        const physical_decision *decision =
            yvex_physical_execution_ir_decision_at(binding->physical_execution, i);
        if (!decision || decision->terminal_tensor_id >= binding->summary.tensor_count ||
            decision->canonical_qtype !=
                binding->runtime[decision->terminal_tensor_id].qtype) {
            *field = "physical-execution-package";
            *code = YVEX_RUNTIME_BINDING_FAILURE_IDENTITY;
            return 0;
        }
    }
    if (!yvex_runtime_capabilities_identity(&binding->summary.capabilities,
                                            capability_identity) ||
        strcmp(capability_identity,
               binding->summary.execution_capability_identity) != 0 ||
        !yvex_runtime_capabilities_contract_valid(&binding->summary.capabilities)) {
        *field = "execution-capabilities";
        return 0;
    }
    compatibility = yvex_artifact_physical_compatibility_mismatch(
        &binding->summary.physical_compatibility, &binding->admission,
        binding->summary.logical_transform_identity);
    if (compatibility) {
        *field = compatibility;
        *code = YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY;
        return 0;
    }
    if (binding->summary.tensor_count != binding->admission.tensor_count ||
        binding->summary.tensor_count != binding->materialization.tensor_count ||
        binding->summary.tensor_count != binding->descriptor.tensor_count ||
        binding->summary.layer_count != binding->attention.layer_count ||
        binding->summary.draft_layer_count != binding->descriptor.draft_layer_count ||
        binding->summary.draft_layer_count != binding->draft_attention.layer_count ||
        binding->summary.tensor_count == 0ull || binding->summary.layer_count == 0ull ||
        !binding_admission_ready(&binding->admission) ||
        !yvex_sha256_hex_is_valid(binding->admission.transform_identity) ||
        !yvex_sha256_hex_is_valid(binding->summary.logical_transform_identity) ||
        binding->materialization.committed || binding->materialization.cleanup_complete ||
        binding->materialization.status != YVEX_MATERIALIZATION_STATUS_PLANNED ||
        binding->materialization.access_calls != 0ull ||
        binding->materialization.payload_bytes_accessed != 0ull ||
        binding->materialization.full_walks != 0ull ||
        binding->materialization.snapshot_drift_count != 0ull ||
        binding->materialization.committed_bindings != 0ull ||
        binding->materialization.aborted_bindings != 0ull ||
        binding->descriptor.status != YVEX_RUNTIME_DESCRIPTOR_STATUS_READY ||
        binding->descriptor.model_execution.schema_version !=
            YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
        !yvex_sha256_hex_is_valid(binding->descriptor.model_execution.identity) ||
        strcmp(binding->descriptor.logical_model_identity,
               binding->descriptor.model_execution.logical_model_identity) != 0 ||
        binding->descriptor.layer_count !=
            binding->descriptor.model_execution.layer_count ||
        binding->descriptor.draft_layer_count !=
            binding->descriptor.model_execution.draft_layer_count ||
        binding->descriptor.vocabulary_size !=
            binding->descriptor.model_execution.vocabulary_size ||
        !yvex_runtime_private_binding_policies_valid(binding) ||
        !yvex_runtime_private_compiled_plan_valid(binding) ||
        !binding_attention_ready(&binding->attention) ||
        binding->attention.tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER ||
        binding->attention.required_binding_count == 0ull ||
        binding->attention.missing_binding_count != 0ull ||
        binding->attention.qtype_compute_refusal_count != 0ull ||
        !binding_identity_chain_valid(&binding->admission, &binding->materialization,
                                      &binding->descriptor, &binding->attention) ||
        (binding->summary.draft_layer_count &&
         (!binding_attention_ready(&binding->draft_attention) ||
          binding->draft_attention.tensor_scope != YVEX_TENSOR_SCOPE_DRAFT ||
          !binding_identity_chain_valid(&binding->admission,
                                        &binding->materialization,
                                        &binding->descriptor,
                                        &binding->draft_attention))))
        return 0;
    if (!yvex_compiled_graph_identities(
            yvex_compiled_model_plan_operator_graph_identity(binding->plan),
            &binding->materialization, &binding->descriptor, &binding->attention,
            binding->summary.draft_layer_count ? &binding->draft_attention : NULL,
            semantic, executable)) {
        *field = "graph-identity-inputs";
        *code = YVEX_RUNTIME_BINDING_FAILURE_IDENTITY;
        return 0;
    }
    if (strcmp(binding->summary.semantic_graph_identity, semantic) != 0) {
        *field = "semantic-graph-identity";
        *code = YVEX_RUNTIME_BINDING_FAILURE_IDENTITY;
        return 0;
    }
    if (strcmp(binding->summary.executable_graph_identity, executable) != 0) {
        *field = "executable-graph-identity";
        *code = YVEX_RUNTIME_BINDING_FAILURE_IDENTITY;
        return 0;
    }
    for (i = 0ull; i < binding->summary.tensor_count; ++i)
        if (binding->materialized[i].tensor_id != i ||
            binding->runtime[i].tensor_id >= binding->summary.tensor_count) return 0;
    for (i = 0ull; i < binding->summary.layer_count; ++i)
        if (binding->layers[i].ordinal != i || binding->layers[i].layer_index != i ||
            binding->layers[i].tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER ||
            binding->layers[i].predictor_index != YVEX_MATERIALIZATION_NO_INDEX) return 0;
    for (i = 0ull; i < binding->summary.draft_layer_count; ++i)
        if (binding->draft_layers[i].ordinal != i ||
            binding->draft_layers[i].tensor_scope != YVEX_TENSOR_SCOPE_DRAFT ||
            binding->draft_layers[i].predictor_index != i) return 0;
    return 1;
}
/*
 * Project one public summary from the canonical parsed or prepared owners.
 *
 * Projects canonical identities and accounting facts from the admitted owners.
 */
static void summary_finish(yvex_runtime_binding_summary *summary,
                           const yvex_complete_artifact_admission *admission,
                           const yvex_materialization_summary *materialization,
                           const yvex_runtime_descriptor_summary *descriptor,
                           const yvex_attention_summary *attention,
                           const yvex_attention_summary *draft_attention,
                           const char *logical_transform_identity,
                           const char *identity, unsigned long long file_bytes)
{
    summary->file_bytes = file_bytes;
    yvex_runtime_identity_copy(summary->identity, identity);
    summary->source_snapshot_identity = admission->source_snapshot_identity;
    summary->mapping_identity = admission->mapping_identity;
    yvex_runtime_identity_copy(summary->payload_identity, admission->payload_identity);
    yvex_runtime_identity_copy(summary->artifact_transform_identity,
                               admission->transform_identity);
    if (logical_transform_identity != summary->logical_transform_identity)
        yvex_runtime_identity_copy(summary->logical_transform_identity,
                                   logical_transform_identity);
    yvex_runtime_identity_copy(summary->profile_identity, admission->profile_identity);
    yvex_runtime_identity_copy(summary->quant_execution_identity,
                               admission->quant_execution_identity);
    yvex_runtime_identity_copy(summary->artifact_identity, admission->artifact_identity);
    yvex_runtime_identity_copy(summary->materialization_identity,
                               materialization->plan_identity);
    yvex_runtime_identity_copy(summary->logical_model_identity,
                               descriptor->logical_model_identity);
    yvex_runtime_identity_copy(summary->runtime_numeric_identity,
                               descriptor->runtime_numeric_identity);
    yvex_runtime_identity_copy(summary->runtime_descriptor_identity,
                               descriptor->runtime_descriptor_identity);
    if (yvex_sha256_hex_is_valid(descriptor->model_execution.identity))
        yvex_runtime_identity_copy(summary->model_execution_identity,
                                   descriptor->model_execution.identity);
    summary->semantic_maximum_context = descriptor->model_execution.maximum_context;
    yvex_runtime_identity_copy(summary->attention_plan_identity,
                               attention->attention_plan_identity);
    if (draft_attention)
        yvex_runtime_identity_copy(summary->draft_attention_plan_identity,
                                   draft_attention->attention_plan_identity);
}
/*
 * Decode and authenticate one already stable runtime-binding byte snapshot.
 *
 * Borrowed exact file bytes, diagnostic path, optional expected identity, and name policy.
 * Malformed, noncanonical, stale, or incompatible bytes publish no binding.
 */
static int binding_file_decode(yvex_runtime_binding **out,
                               const unsigned char *file, size_t file_count,
                               const char *path, const char *expected_identity,
                               int require_addressed_name,
                               yvex_runtime_binding_failure *failure,
                               yvex_error *err)
{
    yvex_runtime_binding *binding = NULL;
    binding_cursor header;
    unsigned char magic[BINDING_MAGIC_BYTES];
    unsigned long long schema = 0ull, body_bytes = 0ull;
    char stored_identity[YVEX_SHA256_HEX_CAP] = {0};
    char computed_identity[YVEX_SHA256_HEX_CAP] = {0};
    char expected_name[96];
    const char *basename;
    binding_parse_result parse_result;
    const binding_parse_failure *parse_failure;
    yvex_runtime_binding_failure_code validation_code;
    const char *validation_field;
    int rc = YVEX_ERR_FORMAT;
    if (out) *out = NULL;
    if (!out || !file || !path || file_count < BINDING_HEADER_BYTES)
        return binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_BOUNDS,
                              "file-size", path, 0ull, BINDING_HEADER_BYTES,
                              file_count, YVEX_ERR_BOUNDS,
                              "runtime binding file size is outside its bound", err);
    header = (binding_cursor){file, file_count, 0u};
    if (!cursor_take(&header, magic, sizeof(magic))) {
        rc = binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
                            "file-magic", path, 0ull, 1ull, 0ull,
                            YVEX_ERR_FORMAT,
                            "runtime binding file magic is invalid", err);
        goto done;
    }
    if (!cursor_u64(&header, &schema)) {
        rc = binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_SCHEMA,
                            "schema-version", path, 0ull,
                            YVEX_RUNTIME_BINDING_SCHEMA_CURRENT, schema,
                            YVEX_ERR_FORMAT,
                            "runtime binding schema is unsupported", err);
        goto done;
    }
    if (schema >= 7u && schema < BINDING_SCHEMA_V14 &&
        memcmp(magic, "YVRBND", 6u) == 0) {
        rc = binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY,
            "model-execution", path, 0ull, YVEX_RUNTIME_BINDING_SCHEMA_CURRENT,
            schema, YVEX_ERR_FORMAT,
            "runtime binding predates the canonical operator graph and must be rebuilt", err);
        goto done;
    }
    if (!((schema == BINDING_SCHEMA_V14 &&
           memcmp(magic, BINDING_MAGIC_V14, sizeof(magic)) == 0) ||
          (schema == YVEX_RUNTIME_BINDING_SCHEMA_CURRENT &&
           memcmp(magic, BINDING_MAGIC_V15, sizeof(magic)) == 0))) {
        rc = binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_SCHEMA,
                            "schema-version", path, 0ull,
                            YVEX_RUNTIME_BINDING_SCHEMA_CURRENT, schema,
                            YVEX_ERR_FORMAT,
                            "runtime binding schema is unsupported", err);
        goto done;
    }
    if (!cursor_u64(&header, &body_bytes) ||
        !cursor_take(&header, stored_identity, 64u) ||
        !yvex_sha256_hex_is_valid(stored_identity)) {
        rc = binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
                            "file-header", path, 0ull, 1ull, 0ull,
                            YVEX_ERR_FORMAT,
                            "runtime binding header identity is invalid", err);
        goto done;
    }
    if (body_bytes != (unsigned long long)file_count - BINDING_HEADER_BYTES) {
        unsigned long long actual_body =
            (unsigned long long)file_count - BINDING_HEADER_BYTES;
        yvex_runtime_binding_failure_code code =
            actual_body < body_bytes ? YVEX_RUNTIME_BINDING_FAILURE_TRUNCATED
                                     : YVEX_RUNTIME_BINDING_FAILURE_TRAILING_DATA;
        rc = binding_reject(
            failure, code, "body-size", path, 0ull, body_bytes, actual_body,
            YVEX_ERR_FORMAT,
            actual_body < body_bytes ? "runtime binding body is truncated"
                                     : "runtime binding contains trailing bytes",
            err);
        goto done;
    }
    if (!binding_identity((unsigned int)schema, file + BINDING_HEADER_BYTES,
                          (size_t)body_bytes,
                          computed_identity) ||
        strcmp(stored_identity, computed_identity) != 0 ||
        (expected_identity && strcmp(expected_identity, computed_identity) != 0)) {
        rc = binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
                            "file-identity", path, 0ull, 1ull, 0ull,
                            YVEX_ERR_FORMAT,
                            "runtime binding header or content identity is invalid", err);
        goto done;
    }
    if (require_addressed_name) {
        basename = strrchr(path, '/');
        basename = basename ? basename + 1 : path;
        (void)snprintf(expected_name, sizeof(expected_name), "%s%s",
                       computed_identity, YVEX_RUNTIME_BINDING_SUFFIX);
        if (strcmp(basename, expected_name) != 0) {
            rc = binding_reject(
                failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
                "content-addressed-name", path, 0ull, 1ull, 0ull,
                YVEX_ERR_FORMAT,
                "runtime binding basename is not its content address", err);
            goto done;
        }
    }
    binding = (yvex_runtime_binding *)calloc(1u, sizeof(*binding));
    if (!binding) {
        rc = binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_ALLOCATION,
                            "binding", path, 0ull, sizeof(*binding), 0ull,
                            YVEX_ERR_NOMEM,
                            "runtime binding object allocation failed", err);
        goto done;
    }
    parse_result = parse_body(binding, file + BINDING_HEADER_BYTES,
                              (size_t)body_bytes, (unsigned int)schema);
    if (parse_result != BINDING_PARSE_OK) {
        parse_failure = &binding_parse_failures[parse_result];
        rc = binding_reject(failure, parse_failure->code, "canonical-body",
                            path, 0ull, body_bytes, 0ull,
                            parse_failure->status, parse_failure->reason, err);
        goto done;
    }
    if (!binding_validate(binding, &validation_field, &validation_code)) {
        rc = binding_reject(
            failure, validation_code, validation_field,
            path, 0ull, body_bytes, 0ull, YVEX_ERR_FORMAT,
            "runtime binding canonical records are inconsistent", err);
        goto done;
    }
    summary_finish(&binding->summary, &binding->admission,
                   &binding->materialization, &binding->descriptor,
                   &binding->attention,
                   binding->summary.draft_layer_count
                       ? &binding->draft_attention : NULL,
                   binding->summary.logical_transform_identity,
                   computed_identity, file_count);
    {
        const physical_summary *physical =
            yvex_physical_execution_ir_summary(binding->physical_execution);
        binding->summary.physical_execution_decision_count = physical->decision_count;
        yvex_runtime_identity_copy(binding->summary.physical_execution_identity,
                                   physical->identity);
    }
    *out = binding;
    binding = NULL;
    rc = YVEX_OK;
    yvex_error_clear(err);
    if (failure) memset(failure, 0, sizeof(*failure));
done:
    yvex_runtime_binding_close(binding);
    return rc;
}
typedef struct {
    const char *path;
    const char *expected_identity;
    yvex_runtime_binding_failure *failure;
    yvex_runtime_binding_summary summary;
} binding_candidate_context;
/*
 * Read one exact reopened publication candidate without sharing a mutable file offset.
 *
 * Allocates one byte snapshot after proving regular-file identity and stability. Short reads, size
 * mismatch, drift, or allocation failure publish no snapshot.
 */
/*
 * Authenticate the exact fsynced candidate reopened by the file lifecycle.
 *
 * Read-only descriptor, exact byte count, and expected content identity. This callback cannot
 * name, replace, or publish a filesystem object.
 */
static int binding_candidate_validate(int descriptor, size_t count,
                                      void *opaque, yvex_error *err)
{
    binding_candidate_context *context = (binding_candidate_context *)opaque;
    yvex_runtime_binding *binding = NULL;
    unsigned char *file = NULL;
    yvex_core_file_result snapshot = {0};
    int rc;
    if (!context || !context->path || !context->expected_identity)
        return binding_reject(
            context ? context->failure : NULL,
            YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT,
            "candidate-validation", context ? context->path : NULL,
            0ull, 1ull, 0ull, YVEX_ERR_INVALID_ARG,
            "runtime binding candidate validator context is incomplete", err);
    rc = count <= BINDING_MAX_BYTES
             ? yvex_core_file_read_descriptor_snapshot(
                   descriptor, count, &file, &snapshot, err)
             : YVEX_ERR_BOUNDS;
    if (rc != YVEX_OK)
        rc = binding_reject(
            context->failure, YVEX_RUNTIME_BINDING_FAILURE_FORMAT,
            "candidate-snapshot", context->path, 0ull, count, snapshot.actual,
            (yvex_status)rc,
            "runtime binding publication candidate changed or read incompletely", err);
    if (rc == YVEX_OK)
        rc = binding_file_decode(&binding, file, count, context->path,
                                 context->expected_identity, 0,
                                 context->failure, err);
    if (rc == YVEX_OK && getenv("YVEX_TEST_RUNTIME_BINDING_VALIDATE_FAILURE"))
        rc = binding_reject(
            context->failure, YVEX_RUNTIME_BINDING_FAILURE_FORMAT,
            "candidate-validation", context->path, 0ull, 1ull, 0ull,
            YVEX_ERR_FORMAT,
            "runtime binding candidate validation failure was injected", err);
    if (rc == YVEX_OK) {
        context->summary = binding->summary;
    }
    yvex_runtime_binding_close(binding);
    free(file);
    return rc;
}
static yvex_runtime_binding_failure_code binding_file_code(yvex_core_file_stage stage)
{
    return (unsigned int)stage < sizeof(binding_file_codes) /
                                    sizeof(binding_file_codes[0])
               ? binding_file_codes[stage] : YVEX_RUNTIME_BINDING_FAILURE_FORMAT;
}
/* Publish one identity-bound immutable runtime binding transactionally. */
int yvex_runtime_binding_prepare(const yvex_runtime_binding_prepare_request *request,
                                 yvex_runtime_binding_prepare_result *result,
                                 yvex_runtime_binding_failure *failure, yvex_error *err)
{
    binding_bytes body = {0}, file = {0};
    yvex_core_bytes compiled_plan_bytes = {0};
    yvex_core_bytes tokenizer_policy_bytes = {0};
    char identity[YVEX_SHA256_HEX_CAP] = {0};
    char semantic[YVEX_SHA256_HEX_CAP] = {0}, executable[YVEX_SHA256_HEX_CAP] = {0};
    char moe_identity[YVEX_SHA256_HEX_CAP] = {0};
    char draft_moe_identity[YVEX_SHA256_HEX_CAP] = {0};
    char final_name[96], final_path[YVEX_PATH_CAP];
    yvex_core_file_result file_result;
    binding_candidate_context candidate;
    yvex_transformer_family_policy transformer = {0};
    yvex_logits_family_policy logits = {0};
    yvex_speculation_family_policy speculation = {0};
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    rc = prepare_validate(request, semantic, executable, moe_identity,
                          draft_moe_identity, failure, err);
    if (!result || rc != YVEX_OK) {
        if (rc == YVEX_OK)
            rc = binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT,
                                "result", NULL, 0ull, 1ull, 0ull, YVEX_ERR_INVALID_ARG,
                                "runtime binding result is required", err);
        return rc;
    }
    if (rc == YVEX_OK) {
        transformer = request->transformer_policy;
        logits = request->logits_policy;
        speculation = request->speculation_policy;
    }
    if (rc == YVEX_OK) {
        const yvex_moe_plan_summary *moe = yvex_moe_plan_summary_get(
            yvex_compiled_model_plan_moe(request->compiled_plan, 0));
        const yvex_moe_plan_summary *draft =
            yvex_moe_plan_summary_get(
                yvex_compiled_model_plan_moe(request->compiled_plan, 1));
        if ((request->capabilities.moe_plan_ready &&
             (!moe || strcmp(moe->moe_plan_identity, moe_identity) != 0 ||
              (request->draft_attention_plan &&
               (!draft || strcmp(draft->moe_plan_identity,
                                 draft_moe_identity) != 0)))) ||
            (!request->capabilities.moe_plan_ready && (moe || draft)))
            rc = binding_reject(
                failure, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
                "compiled-model-plan", request->directory, 0ull, 1ull, 0ull,
                YVEX_ERR_STATE,
                "compiled model-plan identities disagree with graph admission", err);
    }
    compiled_plan_bytes.maximum = BINDING_MAX_BYTES;
    compiled_plan_bytes.initial_capacity = 4096u;
    tokenizer_policy_bytes.maximum = 16384u;
    tokenizer_policy_bytes.initial_capacity = 4096u;
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_family_policy_encode(
            &request->tokenizer_policy, &tokenizer_policy_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_compiled_model_plan_encode(
            request->compiled_plan, &compiled_plan_bytes, err);
    if (rc == YVEX_OK &&
        !binding_body_write(request, semantic, executable, moe_identity,
                            draft_moe_identity, request->physical_execution,
                            &transformer, &logits,
                            &speculation, &tokenizer_policy_bytes,
                            &compiled_plan_bytes, &body))
        rc = binding_reject(
            failure, YVEX_RUNTIME_BINDING_FAILURE_ALLOCATION, "canonical-body",
            request->directory, 0ull, BINDING_MAX_BYTES, body.count, YVEX_ERR_NOMEM,
            "runtime binding canonical body exceeded its allocation budget", err);
    if (rc != YVEX_OK ||
        !binding_identity(YVEX_RUNTIME_BINDING_SCHEMA_CURRENT,
                          body.data, body.count, identity) ||
        !build_file(&body, identity, &file)) {
        if (rc == YVEX_OK)
            rc = binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_ALLOCATION,
                                "file", request->directory, 0ull, BINDING_MAX_BYTES,
                                body.count, YVEX_ERR_NOMEM,
                                "runtime binding file assembly failed", err);
        goto done;
    }
    (void)snprintf(final_name, sizeof(final_name), "%s%s", identity, YVEX_RUNTIME_BINDING_SUFFIX);
    if (snprintf(final_path, sizeof(final_path), "%s%s%s", request->directory,
                 request->directory[strlen(request->directory) - 1u] == '/' ? "" : "/",
                 final_name) >= (int)sizeof(final_path)) {
        rc = binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_BOUNDS, "published-path",
                            request->directory, 0ull, sizeof(result->path), 0ull,
                            YVEX_ERR_BOUNDS, "runtime binding published path exceeds capacity", err);
        goto done;
    }
    memset(&file_result, 0, sizeof(file_result));
    memset(&candidate, 0, sizeof(candidate));
    candidate.path = final_path;
    candidate.expected_identity = identity;
    candidate.failure = failure;
    rc = yvex_core_file_publish_noreplace(
        final_path, file.data, file.count, NULL, binding_candidate_validate,
        &candidate, &file_result, err);
    if (rc != YVEX_OK) {
        if (file_result.stage != YVEX_CORE_FILE_STAGE_VALIDATE ||
            !failure || failure->code == YVEX_RUNTIME_BINDING_FAILURE_NONE)
            rc = binding_reject(
                failure, binding_file_code(file_result.stage), "file-lifecycle",
                final_path, 0ull, file_result.expected, file_result.actual,
                (yvex_status)rc, "runtime binding file lifecycle failed", err);
        goto done;
    }
    yvex_core_text_copy(result->path, sizeof(result->path), final_path);
    result->published = 1;
    result->summary = candidate.summary;
    rc = YVEX_OK;
    yvex_error_clear(err);
    if (failure) memset(failure, 0, sizeof(*failure));
done:
    free(compiled_plan_bytes.data);
    free(tokenizer_policy_bytes.data);
    free(body.data);
    free(file.data);
    return rc;
}
/*
 * Reopen a bounded content-addressed snapshot; malformed identity, drift, or trailing bytes never
 * publish a view.
 */
int yvex_runtime_binding_open(yvex_runtime_binding **out, const char *path,
                              yvex_runtime_binding_summary *summary,
                              yvex_complete_artifact_admission *admission,
                              yvex_runtime_binding_failure *failure, yvex_error *err)
{
    unsigned char *file = NULL;
    yvex_core_file_result file_result;
    size_t file_count = 0u;
    int rc;
    if (out) *out = NULL;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (admission) memset(admission, 0, sizeof(*admission));
    if (!out || !path || !path[0])
        return binding_reject(failure, YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT,
                              "path", path, 0ull, 1ull, 0ull, YVEX_ERR_INVALID_ARG,
                              "runtime binding path and output are required", err);
    memset(&file_result, 0, sizeof(file_result));
    rc = yvex_core_file_read_snapshot(path, BINDING_MAX_BYTES, &file, &file_count,
                                      &file_result, err);
    if (rc != YVEX_OK) {
        rc = binding_reject(failure, binding_file_code(file_result.stage),
                            "file-lifecycle", path, 0ull, file_result.expected,
                            file_result.actual, (yvex_status)rc,
                            "runtime binding file snapshot failed", err);
        goto done;
    }
    rc = binding_file_decode(out, file, file_count, path, NULL, 1, failure, err);
    if (rc == YVEX_OK) {
        if (summary) *summary = (*out)->summary;
        if (admission) *admission = (*out)->admission;
    }
done:
    free(file);
    return rc;
}
