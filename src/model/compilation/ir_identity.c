/*
 * Derive stable identities for generic transformation plans and admitted logical architecture
 * recipes.
 *
 * Every integer uses explicit big-endian width and every string is bounded and length-prefixed;
 * native structure bytes are never hashed. Identity seals plan semantics but does not prove
 * numerical execution.
 */
#include <yvex/internal/compilation.h>

#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int transform_identity_bytes(yvex_sha256 *hash,
                                    const void *bytes,
                                    size_t length)
{
    return yvex_sha256_update(hash, bytes, length);
}

static int transform_identity_u32(yvex_sha256 *hash, unsigned int value)
{
    unsigned char bytes[4];
    unsigned int index;

    for (index = 0u; index < 4u; ++index)
        bytes[3u - index] = (unsigned char)((value >> (index * 8u)) & 0xffu);
    return transform_identity_bytes(hash, bytes, sizeof(bytes));
}

static int transform_identity_u64(yvex_sha256 *hash,
                                  unsigned long long value)
{
    unsigned char bytes[8];
    unsigned int index;

    for (index = 0u; index < 8u; ++index)
        bytes[7u - index] =
            (unsigned char)((value >> (index * 8u)) & 0xffu);
    return transform_identity_bytes(hash, bytes, sizeof(bytes));
}

static int transform_identity_string(yvex_sha256 *hash, const char *text)
{
    size_t length;

    if (!text) return 0;
    length = strlen(text);
    return transform_identity_u64(hash, (unsigned long long)length) &&
           transform_identity_bytes(hash, text, length);
}

static int transform_identity_shape(yvex_sha256 *hash,
                                    const yvex_transform_shape *shape)
{
    unsigned int dimension;

    if (!transform_identity_u32(hash, shape->rank)) return 0;
    for (dimension = 0u; dimension < shape->rank; ++dimension)
        if (!transform_identity_u64(hash, shape->dims[dimension])) return 0;
    return 1;
}

static int transform_identity_key(yvex_sha256 *hash,
                                  const yvex_transform_logical_key *key)
{
    return transform_identity_u32(hash, (unsigned int)key->scope) &&
           transform_identity_u32(hash, (unsigned int)key->subsystem) &&
           transform_identity_u32(hash, (unsigned int)key->role) &&
           transform_identity_u64(hash, key->layer_index) &&
           transform_identity_u64(hash, key->auxiliary_index) &&
           transform_identity_u64(hash, key->group_index);
}

static int transform_identity_precision(
    yvex_sha256 *hash, const yvex_transform_precision_constraint *precision)
{
    return transform_identity_u32(hash, precision->flags) &&
           transform_identity_u32(hash,
                                  precision->allowed_physical_classes) &&
           transform_identity_u32(hash,
                                  precision->approximation_allowed != 0) &&
           transform_identity_u32(hash,
                                  precision->range_proof_required != 0) &&
           transform_identity_u32(hash,
                                  precision->reference_compute_required != 0);
}

/* Encode ir compute identity fields in canonical identity order. */

int yvex_transform_ir_compute_identity(yvex_transform_ir *ir,
                                       yvex_transform_failure *failure,
                                       yvex_error *err)
{
    static const char domain[] = "yvex.transform-ir.v1";
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    if (!ir || ir->summary.schema_version !=
                   YVEX_TRANSFORM_IR_SCHEMA_VERSION) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_IDENTITY_ENCODING,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_SCHEMA_VERSION,
            ir ? ir->summary.schema_version : 0u, 0u, err,
            "transform_ir_identity");
    }
    yvex_sha256_init(&hash);
    if (!transform_identity_string(&hash, domain) ||
        !transform_identity_u32(&hash, ir->summary.schema_version) ||
        !transform_identity_string(&hash,
                                   ir->summary.logical_model_identity) ||
        !transform_identity_u64(&hash,
                                ir->summary.source_snapshot_identity) ||
        !transform_identity_u64(&hash, ir->summary.coverage_identity) ||
        !transform_identity_string(&hash,
                                   ir->summary.required_payload_identity) ||
        !transform_identity_string(&hash, ir->summary.payload_trust_class) ||
        !transform_identity_u64(&hash, ir->summary.source_value_count) ||
        !transform_identity_u64(&hash, ir->summary.value_count) ||
        !transform_identity_u64(&hash, ir->summary.node_count) ||
        !transform_identity_u64(&hash, ir->summary.edge_count) ||
        !transform_identity_u64(&hash, ir->summary.terminal_count))
        goto encode_failure;

    for (index = 0u; index < ir->summary.source_value_count; ++index) {
        const yvex_transform_source_value *source = &ir->sources[index];
        if (!transform_identity_u64(&hash, source->value_id) ||
            !transform_identity_string(&hash, source->source_name) ||
            !transform_identity_string(&hash, source->shard_name) ||
            !transform_identity_u64(&hash, source->source_tensor_index) ||
            !transform_identity_u64(&hash, source->requirement_index) ||
            !transform_identity_u64(&hash,
                                    source->source_snapshot_identity) ||
            !transform_identity_u32(&hash,
                                    (unsigned int)source->source_dtype) ||
            !transform_identity_u32(&hash,
                                    (unsigned int)source->value_dtype) ||
            !transform_identity_shape(&hash, &source->shape) ||
            !transform_identity_u64(&hash, source->relative_begin) ||
            !transform_identity_u64(&hash, source->relative_end) ||
            !transform_identity_u64(&hash, source->requirement_identity) ||
            !transform_identity_u32(&hash, (unsigned int)source->scope) ||
            !transform_identity_u32(&hash,
                                    (unsigned int)source->subsystem) ||
            !transform_identity_u32(&hash,
                                    (unsigned int)source->role_hint) ||
            !transform_identity_u64(&hash, source->layer_index) ||
            !transform_identity_u64(&hash, source->auxiliary_index) ||
            !transform_identity_u64(&hash, source->expert_index) ||
            !transform_identity_u64(&hash, source->required_uses))
            goto encode_failure;
    }
    for (index = 0u; index < ir->summary.value_count; ++index) {
        const yvex_transform_value *value = &ir->values[index];
        if (!transform_identity_u64(&hash, value->id) ||
            !transform_identity_u32(&hash, (unsigned int)value->kind) ||
            !transform_identity_u64(&hash, value->semantic_id) ||
            !transform_identity_u64(&hash, value->canonical_ordinal) ||
            !transform_identity_u64(&hash, value->source_index) ||
            !transform_identity_shape(&hash, &value->shape) ||
            !transform_identity_u32(&hash, (unsigned int)value->dtype) ||
            !transform_identity_precision(&hash, &value->precision))
            goto encode_failure;
        if (value->kind == YVEX_TRANSFORM_VALUE_TERMINAL &&
            !transform_identity_key(&hash, &value->logical_key))
            goto encode_failure;
    }
    for (index = 0u; index < ir->summary.node_count; ++index) {
        const yvex_transform_node *node = &ir->nodes[index];
        unsigned long long input;
        unsigned int permutation;

        if (!transform_identity_u64(&hash, node->id) ||
            !transform_identity_u32(&hash, (unsigned int)node->kind) ||
            !transform_identity_u64(&hash, node->output_value_id) ||
            !transform_identity_u64(&hash, node->input_count) ||
            !transform_identity_u32(&hash, node->axis) ||
            !transform_identity_u32(&hash, node->permutation_rank) ||
            !transform_identity_u64(&hash, node->expert_count) ||
            !transform_identity_u64(&hash, node->packing_factor) ||
            !transform_identity_u64(&hash, node->scale_group_width) ||
            !transform_identity_u64(&hash, node->scale_block_rows) ||
            !transform_identity_u64(&hash, node->scale_block_columns) ||
            !transform_identity_u32(&hash, (unsigned int)node->numeric) ||
            !transform_identity_u32(&hash, (unsigned int)node->ordering) ||
            !transform_identity_u32(&hash,
                                    node->payload_execution_required != 0))
            goto encode_failure;
        for (permutation = 0u; permutation < node->permutation_rank;
             ++permutation)
            if (!transform_identity_u32(&hash,
                                        node->permutation[permutation]))
                goto encode_failure;
        for (input = 0u; input < node->input_count; ++input)
            if (!transform_identity_u64(
                    &hash, ir->edges[node->input_offset + input]))
                goto encode_failure;
    }
    for (index = 0u; index < ir->summary.node_count; ++index)
        if (!transform_identity_u64(&hash, ir->topological_order[index]))
            goto encode_failure;
    for (index = 0u; index < ir->summary.terminal_count; ++index)
        if (!transform_identity_u64(&hash, ir->terminal_values[index]))
            goto encode_failure;
    if (!yvex_sha256_final(&hash, digest)) goto encode_failure;
    yvex_sha256_hex(digest, ir->summary.transform_identity);
    return YVEX_OK;

encode_failure:
    return yvex_transform_fail(
        failure, YVEX_TRANSFORM_FAILURE_IDENTITY_ENCODING,
        YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
        YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
        YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, 0u, err,
        "transform_ir_identity");
}
#define deepseek_identity_u64 yvex_sha256_update_u64_be

static int deepseek_identity_text(yvex_sha256 *hash, const char *text)
{
    size_t length;

    if (!text) return 0;
    length = strlen(text);
    return deepseek_identity_u64(hash, (unsigned long long)length) &&
           yvex_sha256_update(hash, text, length);
}

static int deepseek_identity_double(yvex_sha256 *hash, double value)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return deepseek_identity_u64(hash, (unsigned long long)bits);
}

typedef enum {
    IDENTITY_FIELD_UNSIGNED = 0,
    IDENTITY_FIELD_SIGNED,
    IDENTITY_FIELD_DOUBLE,
    IDENTITY_FIELD_TEXT
} identity_field_kind;

typedef struct {
    size_t offset;
    size_t width;
    identity_field_kind kind;
} identity_field;

/* Encode identity unsigned fields in canonical identity order. */

static int identity_unsigned(yvex_sha256 *hash, const void *field, size_t width)
{
    uint64_t value64;
    uint32_t value32;
    uint16_t value16;
    uint8_t value8;

    if (width == sizeof(value64)) {
        memcpy(&value64, field, width);
        return deepseek_identity_u64(hash, value64);
    }
    if (width == sizeof(value32)) {
        memcpy(&value32, field, width);
        return deepseek_identity_u64(hash, value32);
    }
    if (width == sizeof(value16)) {
        memcpy(&value16, field, width);
        return deepseek_identity_u64(hash, value16);
    }
    if (width == sizeof(value8)) {
        memcpy(&value8, field, width);
        return deepseek_identity_u64(hash, value8);
    }
    return 0;
}

/* Encode identity signed fields in canonical identity order. */

static int identity_signed(yvex_sha256 *hash, const void *field, size_t width)
{
    int64_t value64;
    int32_t value32;
    int16_t value16;
    int8_t value8;

    if (width == sizeof(value64)) {
        memcpy(&value64, field, width);
        return deepseek_identity_u64(hash, (unsigned long long)value64);
    }
    if (width == sizeof(value32)) {
        memcpy(&value32, field, width);
        return deepseek_identity_u64(hash, (unsigned long long)value32);
    }
    if (width == sizeof(value16)) {
        memcpy(&value16, field, width);
        return deepseek_identity_u64(hash, (unsigned long long)value16);
    }
    if (width == sizeof(value8)) {
        memcpy(&value8, field, width);
        return deepseek_identity_u64(hash, (unsigned long long)value8);
    }
    return 0;
}

/*
 * Serialize one declared semantic field sequence in canonical order.
 *
 * Object is immutable and fields name exact typed members, never native padding.
 */
static int identity_fields(yvex_sha256 *hash,
                           const void *object,
                           const identity_field *fields,
                           size_t field_count)
{
    const unsigned char *base = (const unsigned char *)object;
    size_t index;

    if (!hash || !object || (!fields && field_count != 0u)) return 0;
    for (index = 0u; index < field_count; ++index) {
        const void *field = base + fields[index].offset;
        double value;
        int ok;

        switch (fields[index].kind) {
        case IDENTITY_FIELD_UNSIGNED:
            ok = identity_unsigned(hash, field, fields[index].width);
            break;
        case IDENTITY_FIELD_SIGNED:
            ok = identity_signed(hash, field, fields[index].width);
            break;
        case IDENTITY_FIELD_DOUBLE:
            memcpy(&value, field, sizeof(value));
            ok = deepseek_identity_double(hash, value);
            break;
        case IDENTITY_FIELD_TEXT:
            ok = deepseek_identity_text(hash, (const char *)field);
            break;
        default:
            return 0;
        }
        if (!ok) return 0;
    }
    return 1;
}

#define ID_UNSIGNED(type, field) \
    { offsetof(type, field), sizeof(((type *)0)->field), IDENTITY_FIELD_UNSIGNED }
#define ID_SIGNED(type, field) \
    { offsetof(type, field), sizeof(((type *)0)->field), IDENTITY_FIELD_SIGNED }
#define ID_DOUBLE(type, field) \
    { offsetof(type, field), sizeof(((type *)0)->field), IDENTITY_FIELD_DOUBLE }
#define ID_TEXT(type, field) \
    { offsetof(type, field), sizeof(((type *)0)->field), IDENTITY_FIELD_TEXT }

static const identity_field activation_fields[] = {
    ID_SIGNED(yvex_attention_activation_policy, required),
    ID_UNSIGNED(yvex_attention_activation_policy, stage),
    ID_UNSIGNED(yvex_attention_activation_policy, quantization),
    ID_UNSIGNED(yvex_attention_activation_policy, block_axis),
    ID_UNSIGNED(yvex_attention_activation_policy, block_width),
    ID_UNSIGNED(yvex_attention_activation_policy, scale_format),
    ID_UNSIGNED(yvex_attention_activation_policy, scale_dtype),
    ID_UNSIGNED(yvex_attention_activation_policy, pre_transform),
    ID_UNSIGNED(yvex_attention_activation_policy, tail_policy),
    ID_UNSIGNED(yvex_attention_activation_policy, nonfinite_policy),
    ID_SIGNED(yvex_attention_activation_policy, fake_quant_inplace),
    ID_SIGNED(yvex_attention_activation_policy, zero_pad_hadamard_to_power_of_two)
};

static const identity_field topk_fields[] = {
    ID_SIGNED(yvex_attention_topk_policy, required),
    ID_UNSIGNED(yvex_attention_topk_policy, version),
    ID_UNSIGNED(yvex_attention_topk_policy, policy),
    ID_UNSIGNED(yvex_attention_topk_policy, k),
    ID_SIGNED(yvex_attention_topk_policy, reject_nonfinite),
    ID_SIGNED(yvex_attention_topk_policy, score_descending),
    ID_SIGNED(yvex_attention_topk_policy, equal_score_ordinal_ascending),
    ID_SIGNED(yvex_attention_topk_policy, plus_zero_equals_minus_zero),
    ID_SIGNED(yvex_attention_topk_policy, duplicate_ordinal_refused),
    ID_SIGNED(yvex_attention_topk_policy, output_ranked_order)
};

static const identity_field main_layer_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, layer_index),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, attention_class),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, compute_contract),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, compression_ratio),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, query_heads),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, kv_heads),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, head_dimension),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, rope_head_dimension),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, non_rope_head_dimension),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, query_lora_rank),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, output_lora_rank),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, output_groups),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, output_heads_per_group),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, output_group_input_width),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, indexer_heads),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, indexer_head_dimension),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, indexer_topk),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, attention_sink_count),
    ID_DOUBLE(yvex_deepseek_v4_layer_spec, attention_dropout),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, causal),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, attention_bias),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, query_norm_required),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, kv_norm_required),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, compressor_required),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, indexer_required)
};

static const identity_field auxiliary_layer_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, compute_contract),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, compression_ratio),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, query_heads),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, kv_heads),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, head_dimension),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, rope_head_dimension),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, non_rope_head_dimension),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, query_lora_rank),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, output_lora_rank),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, output_groups),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, output_heads_per_group),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, output_group_input_width),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, indexer_heads),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, indexer_head_dimension),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, indexer_topk),
    ID_UNSIGNED(yvex_deepseek_v4_layer_spec, attention_sink_count),
    ID_DOUBLE(yvex_deepseek_v4_layer_spec, attention_dropout),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, causal),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, attention_bias),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, query_norm_required),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, kv_norm_required),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, compressor_required),
    ID_SIGNED(yvex_deepseek_v4_layer_spec, indexer_required)
};

static const identity_field position_fields[] = {
    ID_UNSIGNED(yvex_attention_position_policy, rope_dimension),
    ID_UNSIGNED(yvex_attention_position_policy, theta),
    ID_UNSIGNED(yvex_attention_position_policy, scaling_factor),
    ID_UNSIGNED(yvex_attention_position_policy, original_context),
    ID_UNSIGNED(yvex_attention_position_policy, beta_fast),
    ID_UNSIGNED(yvex_attention_position_policy, beta_slow),
    ID_UNSIGNED(yvex_attention_position_policy, maximum_context),
    ID_SIGNED(yvex_attention_position_policy, partial_rope),
    ID_SIGNED(yvex_attention_position_policy, inverse_output_rotation)
};

static const identity_field kv_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_kv_spec, class_id),
    ID_UNSIGNED(yvex_deepseek_v4_kv_spec, compression_ratio),
    ID_UNSIGNED(yvex_deepseek_v4_kv_spec, sliding_window),
    ID_SIGNED(yvex_deepseek_v4_kv_spec, requires_state_cache),
    ID_SIGNED(yvex_deepseek_v4_kv_spec, requires_uncompressed_tail),
    ID_SIGNED(yvex_deepseek_v4_kv_spec, requires_compressed_core),
    ID_SIGNED(yvex_deepseek_v4_kv_spec, requires_indexer_cache)
};

static const identity_field main_mhc_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, residual_streams),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, stream_width),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, expanded_width),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, mixing_rows),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, mixing_columns),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, base_width),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, scale_width),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, sinkhorn_iterations),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, entry),
    ID_SIGNED(yvex_deepseek_v4_mhc_spec, attention_pre_and_post),
    ID_SIGNED(yvex_deepseek_v4_mhc_spec, ffn_pre_and_deferred_post),
    ID_DOUBLE(yvex_deepseek_v4_mhc_spec, epsilon),
    ID_DOUBLE(yvex_deepseek_v4_mhc_spec, residual_post_multiplier)
};

static const identity_field auxiliary_mhc_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, residual_streams),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, stream_width),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, expanded_width),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, mixing_rows),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, mixing_columns),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, base_width),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, scale_width),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, sinkhorn_iterations),
    ID_DOUBLE(yvex_deepseek_v4_mhc_spec, epsilon),
    ID_DOUBLE(yvex_deepseek_v4_mhc_spec, residual_post_multiplier),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_spec, entry),
    ID_SIGNED(yvex_deepseek_v4_mhc_spec, attention_pre_and_post),
    ID_SIGNED(yvex_deepseek_v4_mhc_spec, ffn_pre_and_deferred_post)
};

static const identity_field main_moe_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, router_class),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, scoring),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, topk_policy),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, activation),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, routed_experts),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, shared_experts),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, experts_per_token),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, expert_intermediate_size),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, shared_intermediate_size),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, hash_table_rows),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, hash_table_columns),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, correction_bias_width),
    ID_DOUBLE(yvex_deepseek_v4_moe_spec, routed_scaling_factor),
    ID_DOUBLE(yvex_deepseek_v4_moe_spec, activation_limit),
    ID_SIGNED(yvex_deepseek_v4_moe_spec, requires_token_ids),
    ID_SIGNED(yvex_deepseek_v4_moe_spec, requires_hidden_state),
    ID_SIGNED(yvex_deepseek_v4_moe_spec, requires_correction_bias),
    ID_SIGNED(yvex_deepseek_v4_moe_spec, normalize_topk_probabilities)
};

static const identity_field auxiliary_moe_intro_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, router_class),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, routed_experts),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, expert_intermediate_size),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, shared_intermediate_size)
};

static const identity_field auxiliary_moe_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, scoring),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, topk_policy),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, activation),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, shared_experts),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, experts_per_token),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, hash_table_rows),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, hash_table_columns),
    ID_UNSIGNED(yvex_deepseek_v4_moe_spec, correction_bias_width),
    ID_DOUBLE(yvex_deepseek_v4_moe_spec, routed_scaling_factor),
    ID_DOUBLE(yvex_deepseek_v4_moe_spec, activation_limit),
    ID_SIGNED(yvex_deepseek_v4_moe_spec, requires_token_ids),
    ID_SIGNED(yvex_deepseek_v4_moe_spec, requires_hidden_state),
    ID_SIGNED(yvex_deepseek_v4_moe_spec, requires_correction_bias),
    ID_SIGNED(yvex_deepseek_v4_moe_spec, normalize_topk_probabilities)
};

static const identity_field norm_fields[] = {
    ID_SIGNED(yvex_deepseek_v4_norm_spec, required),
    ID_UNSIGNED(yvex_deepseek_v4_norm_spec, width)
};

static const identity_field tensor_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, q_a_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, q_a_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, q_b_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, q_b_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, kv_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, kv_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, o_a_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, o_a_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, o_b_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, o_b_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, compressor_ape_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, compressor_ape_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, compressor_norm_width),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, compressor_projection_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, compressor_projection_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, indexer_ape_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, indexer_ape_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, indexer_norm_width),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, indexer_projection_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, indexer_projection_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, indexer_query_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, indexer_query_columns),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, indexer_weight_rows),
    ID_UNSIGNED(yvex_deepseek_v4_attention_tensor_spec, indexer_weight_columns)
};

static const identity_field auxiliary_tail_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_auxiliary_spec, previous_hidden_width),
    ID_UNSIGNED(yvex_deepseek_v4_auxiliary_spec, embedding_projection_input),
    ID_UNSIGNED(yvex_deepseek_v4_auxiliary_spec, embedding_projection_output),
    ID_UNSIGNED(yvex_deepseek_v4_auxiliary_spec, hidden_projection_input),
    ID_UNSIGNED(yvex_deepseek_v4_auxiliary_spec, hidden_projection_output),
    ID_SIGNED(yvex_deepseek_v4_auxiliary_spec, requires_token_embedding),
    ID_SIGNED(yvex_deepseek_v4_auxiliary_spec, requires_previous_hidden_state),
    ID_SIGNED(yvex_deepseek_v4_auxiliary_spec, requires_embedding_norm),
    ID_SIGNED(yvex_deepseek_v4_auxiliary_spec, requires_hidden_norm),
    ID_SIGNED(yvex_deepseek_v4_auxiliary_spec, requires_separate_mhc_head)
};

static const identity_field mhc_head_fields[] = {
    ID_SIGNED(yvex_deepseek_v4_mhc_head_spec, required),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_head_spec, function_rows),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_head_spec, function_columns),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_head_spec, base_width),
    ID_UNSIGNED(yvex_deepseek_v4_mhc_head_spec, scale_width)
};

static const identity_field auxiliary_share_fields[] = {
    ID_SIGNED(yvex_deepseek_v4_auxiliary_spec, shares_output_head),
    ID_SIGNED(yvex_deepseek_v4_auxiliary_spec, shares_final_norm)
};

static const identity_field model_text_fields[] = {
    ID_TEXT(yvex_deepseek_v4_model_spec, target_id),
    ID_TEXT(yvex_deepseek_v4_model_spec, family),
    ID_TEXT(yvex_deepseek_v4_model_spec, architecture),
    ID_TEXT(yvex_deepseek_v4_model_spec, repository),
    ID_TEXT(yvex_deepseek_v4_model_spec, revision),
    ID_TEXT(yvex_deepseek_v4_model_spec, hadamard_revision)
};

static const identity_field model_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_model_spec, runtime_numeric_schema_version),
    ID_UNSIGNED(yvex_deepseek_v4_model_spec, runtime_compute_policy_count),
    ID_UNSIGNED(yvex_deepseek_v4_model_spec, runtime_activation_policy_count),
    ID_UNSIGNED(yvex_deepseek_v4_model_spec, runtime_sparse_topk_policy_count),
    ID_UNSIGNED(yvex_deepseek_v4_model_spec, hidden_size),
    ID_UNSIGNED(yvex_deepseek_v4_model_spec, vocabulary_size),
    ID_UNSIGNED(yvex_deepseek_v4_model_spec, maximum_context),
    ID_UNSIGNED(yvex_deepseek_v4_model_spec, main_layer_count),
    ID_UNSIGNED(yvex_deepseek_v4_model_spec, auxiliary_layer_count)
};

static const identity_field embedding_fields[] = {
    ID_SIGNED(yvex_deepseek_v4_embedding_spec, required),
    ID_UNSIGNED(yvex_deepseek_v4_embedding_spec, vocabulary_size),
    ID_UNSIGNED(yvex_deepseek_v4_embedding_spec, hidden_size)
};

static const identity_field output_fields[] = {
    ID_SIGNED(yvex_deepseek_v4_output_spec, required),
    ID_SIGNED(yvex_deepseek_v4_output_spec, tied_to_embedding),
    ID_UNSIGNED(yvex_deepseek_v4_output_spec, input_width),
    ID_UNSIGNED(yvex_deepseek_v4_output_spec, vocabulary_size)
};

static const identity_field source_constraint_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_source_constraint, weight_dtype),
    ID_UNSIGNED(yvex_deepseek_v4_source_constraint, expert_dtype),
    ID_UNSIGNED(yvex_deepseek_v4_source_constraint, quantization),
    ID_UNSIGNED(yvex_deepseek_v4_source_constraint, quant_block_rows),
    ID_UNSIGNED(yvex_deepseek_v4_source_constraint, quant_block_columns),
    ID_UNSIGNED(yvex_deepseek_v4_source_constraint, fp4_packing_factor),
    ID_UNSIGNED(yvex_deepseek_v4_source_constraint, fp4_scale_group_width),
    ID_UNSIGNED(yvex_deepseek_v4_source_constraint, fp4_physical_dtype),
    ID_UNSIGNED(yvex_deepseek_v4_source_constraint, scale_dtype)
};

static const identity_field tokenizer_text_fields[] = {
    ID_TEXT(yvex_deepseek_v4_tokenizer_spec, tokenizer_class),
    ID_TEXT(yvex_deepseek_v4_tokenizer_spec, model_type)
};

static const identity_field tokenizer_fields[] = {
    ID_UNSIGNED(yvex_deepseek_v4_tokenizer_spec, vocabulary_size),
    ID_UNSIGNED(yvex_deepseek_v4_tokenizer_spec, base_vocab_entries),
    ID_UNSIGNED(yvex_deepseek_v4_tokenizer_spec, added_token_entries),
    ID_UNSIGNED(yvex_deepseek_v4_tokenizer_spec, maximum_token_id),
    ID_UNSIGNED(yvex_deepseek_v4_tokenizer_spec, maximum_context),
    ID_UNSIGNED(yvex_deepseek_v4_tokenizer_spec, bos_token_id),
    ID_UNSIGNED(yvex_deepseek_v4_tokenizer_spec, eos_token_id),
    ID_SIGNED(yvex_deepseek_v4_tokenizer_spec, bos_required),
    ID_SIGNED(yvex_deepseek_v4_tokenizer_spec, eos_required)
};

static const identity_field model_tail_fields[] = {
    ID_DOUBLE(yvex_deepseek_v4_model_spec, final_norm_epsilon),
    ID_SIGNED(yvex_deepseek_v4_model_spec, use_cache),
    ID_SIGNED(yvex_deepseek_v4_model_spec, final_mhc_post_required),
    ID_SIGNED(yvex_deepseek_v4_model_spec, final_mhc_head_required),
    ID_SIGNED(yvex_deepseek_v4_model_spec, final_norm_after_mhc_head)
};

#undef ID_UNSIGNED
#undef ID_SIGNED
#undef ID_DOUBLE
#undef ID_TEXT

static int identity_activation(yvex_sha256 *hash,
                               const yvex_attention_activation_policy *policy)
{
    return identity_fields(hash, policy, activation_fields,
                           sizeof(activation_fields) / sizeof(activation_fields[0]));
}

static int identity_topk(yvex_sha256 *hash,
                         const yvex_attention_topk_policy *policy)
{
    return identity_fields(hash, policy, topk_fields,
                           sizeof(topk_fields) / sizeof(topk_fields[0]));
}

/* Encode identity main layer fields in canonical identity order. */

static int identity_main_layer(yvex_sha256 *hash, const yvex_deepseek_v4_layer_spec *layer)
{
    const yvex_attention_position_policy *position = &layer->position;
    const yvex_deepseek_v4_attention_tensor_spec *tensor = &layer->tensors;
    const yvex_deepseek_v4_mhc_spec *mhc = &layer->mhc;
    const yvex_deepseek_v4_moe_spec *moe = &layer->moe;
    const yvex_deepseek_v4_kv_spec *kv = &layer->kv;

    return identity_fields(hash, layer, main_layer_fields,
                           sizeof(main_layer_fields) / sizeof(main_layer_fields[0])) &&
           identity_fields(hash, position, position_fields,
                           sizeof(position_fields) / sizeof(position_fields[0])) &&
           identity_fields(hash, kv, kv_fields, sizeof(kv_fields) / sizeof(kv_fields[0])) &&
           identity_fields(hash, mhc, main_mhc_fields,
                           sizeof(main_mhc_fields) / sizeof(main_mhc_fields[0])) &&
           identity_fields(hash, moe, main_moe_fields,
                           sizeof(main_moe_fields) / sizeof(main_moe_fields[0])) &&
           identity_activation(hash, &layer->attention_kv_activation) &&
           identity_activation(hash, &layer->compressor_activation) &&
           identity_activation(hash, &layer->compressor_rotated_activation) &&
           identity_activation(hash, &layer->indexer_query_activation) &&
           identity_topk(hash, &layer->sparse_topk) &&
           identity_fields(hash, &layer->attention_input_norm, norm_fields,
                           sizeof(norm_fields) / sizeof(norm_fields[0])) &&
           identity_fields(hash, &layer->post_attention_ffn_norm, norm_fields,
                           sizeof(norm_fields) / sizeof(norm_fields[0])) &&
           identity_fields(hash, tensor, tensor_fields,
                           sizeof(tensor_fields) / sizeof(tensor_fields[0])) &&
           deepseek_identity_double(hash, layer->rms_norm_epsilon);
}

/* Encode identity auxiliary fields in canonical identity order. */

static int identity_auxiliary(yvex_sha256 *hash, const yvex_deepseek_v4_auxiliary_spec *aux)
{
    const yvex_deepseek_v4_layer_spec *layer = &aux->layer;
    const yvex_attention_position_policy *position = &layer->position;
    const yvex_deepseek_v4_attention_tensor_spec *tensor = &layer->tensors;
    const yvex_deepseek_v4_mhc_spec *mhc = &layer->mhc;
    const yvex_deepseek_v4_moe_spec *moe = &layer->moe;
    const yvex_deepseek_v4_kv_spec *kv = &layer->kv;

    return deepseek_identity_u64(hash, aux->predictor_index) &&
           deepseek_identity_u64(hash, layer->layer_index) &&
           deepseek_identity_u64(hash, layer->attention_class) &&
           identity_fields(hash, moe, auxiliary_moe_intro_fields,
                           sizeof(auxiliary_moe_intro_fields) /
                               sizeof(auxiliary_moe_intro_fields[0])) &&
           identity_fields(hash, layer, auxiliary_layer_fields,
                           sizeof(auxiliary_layer_fields) /
                               sizeof(auxiliary_layer_fields[0])) &&
           identity_fields(hash, position, position_fields,
                           sizeof(position_fields) / sizeof(position_fields[0])) &&
           identity_fields(hash, kv, kv_fields, sizeof(kv_fields) / sizeof(kv_fields[0])) &&
           identity_fields(hash, mhc, auxiliary_mhc_fields,
                           sizeof(auxiliary_mhc_fields) /
                               sizeof(auxiliary_mhc_fields[0])) &&
           identity_fields(hash, moe, auxiliary_moe_fields,
                           sizeof(auxiliary_moe_fields) /
                               sizeof(auxiliary_moe_fields[0])) &&
           identity_activation(hash, &layer->attention_kv_activation) &&
           identity_activation(hash, &layer->compressor_activation) &&
           identity_activation(hash, &layer->compressor_rotated_activation) &&
           identity_activation(hash, &layer->indexer_query_activation) &&
           identity_topk(hash, &layer->sparse_topk) &&
           identity_fields(hash, &layer->attention_input_norm, norm_fields,
                           sizeof(norm_fields) / sizeof(norm_fields[0])) &&
           identity_fields(hash, &layer->post_attention_ffn_norm, norm_fields,
                           sizeof(norm_fields) / sizeof(norm_fields[0])) &&
           identity_fields(hash, tensor, tensor_fields,
                           sizeof(tensor_fields) / sizeof(tensor_fields[0])) &&
           deepseek_identity_double(hash, layer->rms_norm_epsilon) &&
           identity_fields(hash, aux, auxiliary_tail_fields,
                           sizeof(auxiliary_tail_fields) /
                               sizeof(auxiliary_tail_fields[0])) &&
           identity_fields(hash, &aux->mhc_head, mhc_head_fields,
                           sizeof(mhc_head_fields) / sizeof(mhc_head_fields[0])) &&
           identity_fields(hash, aux, auxiliary_share_fields,
                           sizeof(auxiliary_share_fields) /
                               sizeof(auxiliary_share_fields[0]));
}

/* Encode identity model fields in canonical identity order. */

static int identity_model(yvex_sha256 *hash, const yvex_deepseek_v4_model_spec *model)
{
    const yvex_deepseek_v4_mhc_spec *mhc = &model->final_mhc;

    return identity_fields(hash, model, model_text_fields,
                           sizeof(model_text_fields) / sizeof(model_text_fields[0])) &&
           identity_fields(hash, model, model_fields,
                           sizeof(model_fields) / sizeof(model_fields[0])) &&
           identity_fields(hash, &model->embedding, embedding_fields,
                           sizeof(embedding_fields) / sizeof(embedding_fields[0])) &&
           identity_fields(hash, &model->output, output_fields,
                           sizeof(output_fields) / sizeof(output_fields[0])) &&
           identity_fields(hash, &model->source_constraint, source_constraint_fields,
                           sizeof(source_constraint_fields) /
                               sizeof(source_constraint_fields[0])) &&
           identity_fields(hash, &model->tokenizer, tokenizer_text_fields,
                           sizeof(tokenizer_text_fields) /
                               sizeof(tokenizer_text_fields[0])) &&
           identity_fields(hash, &model->tokenizer, tokenizer_fields,
                           sizeof(tokenizer_fields) / sizeof(tokenizer_fields[0])) &&
           identity_fields(hash, mhc, auxiliary_mhc_fields,
                           sizeof(auxiliary_mhc_fields) /
                               sizeof(auxiliary_mhc_fields[0])) &&
           identity_fields(hash, &model->final_mhc_head, mhc_head_fields,
                           sizeof(mhc_head_fields) / sizeof(mhc_head_fields[0])) &&
           identity_fields(hash, model, model_tail_fields,
                           sizeof(model_tail_fields) / sizeof(model_tail_fields[0]));
}

/*
 * Encode the admitted DeepSeek logical architecture without native structure bytes or
 * artifact-format facts.
 *
 * Writes one lowercase SHA-256 identity on complete success. Missing architecture facts or hash
 * failure leave no valid identity. This is logical-model identity, not Transform IR or artifact
 * identity.
 */
int yvex_transform_deepseek_architecture_identity(
    const yvex_deepseek_v4_ir *architecture,
    char output[YVEX_TRANSFORM_IR_IDENTITY_CAP])
{
    static const char domain[] = "yvex.logical-model.deepseek-v4-flash.v1";
    const yvex_model_family_ir_api *family_ir =
        &yvex_model_register_deepseek_v4()->ir;
    const yvex_deepseek_v4_model_spec *model = family_ir->model(architecture);
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    unsigned long long index;

    if (!model || !output) return 0;
    yvex_sha256_init(&hash);
    if (!deepseek_identity_text(&hash, domain) || !identity_model(&hash, model)) return 0;
    for (index = 0u; index < model->main_layer_count; ++index) {
        const yvex_deepseek_v4_layer_spec *layer =
            family_ir->layer_at(architecture, index);
        if (!layer || !identity_main_layer(&hash, layer)) return 0;
    }
    for (index = 0u; index < model->auxiliary_layer_count; ++index) {
        const yvex_deepseek_v4_auxiliary_spec *aux =
            family_ir->auxiliary_at(architecture, index);
        if (!aux || !identity_auxiliary(&hash, aux)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
