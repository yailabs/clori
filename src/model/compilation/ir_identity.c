/*
 * Derive stable identities for generic transformation plans and admitted logical architecture
 * recipes.
 *
 * Every integer uses explicit big-endian width and every string is bounded and length-prefixed;
 * native structure bytes are never hashed. Identity seals plan semantics but does not prove
 * numerical execution.
 */
#include <yvex/internal/compilation.h>
#include "src/model/compilation/private.h"

#include <yvex/internal/core.h>
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
                                  const yvex_transform_logical_key *key,
                                  unsigned int schema_version)
{
    int ok = transform_identity_u32(hash, (unsigned int)key->scope) &&
           transform_identity_u32(hash, (unsigned int)key->subsystem) &&
           transform_identity_u32(hash, (unsigned int)key->role);

    if (ok && schema_version == YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION) {
        ok = transform_identity_u64(hash, key->component_identity) &&
             transform_identity_u64(hash, key->semantic_role) &&
             transform_identity_u64(hash, key->phase_identity) &&
             transform_identity_u64(hash, key->lifetime_identity);
    }
    return ok &&
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
    const char *domain;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    if (!ir || (ir->summary.schema_version != YVEX_TRANSFORM_IR_SCHEMA_VERSION &&
                ir->summary.schema_version !=
                    YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION)) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_IDENTITY_ENCODING,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_SCHEMA_VERSION,
            ir ? ir->summary.schema_version : 0u, 0u, err,
            "transform_ir_identity");
    }
    domain = ir->summary.schema_version == YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION
                 ? "yvex.transform-ir.v2" : "yvex.transform-ir.v1";
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
    if (ir->summary.schema_version == YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION &&
        (!transform_identity_string(&hash, ir->summary.component_manifest_identity) ||
         !transform_identity_string(&hash, ir->summary.architecture_identity) ||
         !transform_identity_string(&hash, ir->summary.role_map_identity) ||
         !transform_identity_string(&hash, ir->summary.unresolved_requirements_identity)))
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
        if (ir->summary.schema_version ==
                YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION &&
            (!transform_identity_u64(&hash, source->component_identity) ||
             !transform_identity_u64(&hash, source->semantic_role) ||
             !transform_identity_u64(&hash, source->phase_identity) ||
             !transform_identity_u64(&hash, source->lifetime_identity) ||
             !transform_identity_u64(
                 &hash, source->unresolved_requirement_identity)))
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
            !transform_identity_key(&hash, &value->logical_key,
                                    ir->summary.schema_version))
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
