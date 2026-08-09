/*
 * Seal deterministic physical decisions over immutable transform plans.
 *
 * 1,409 canonical terminals and descriptors biject after complete typed-field validation;
 * construction reads zero payload bytes. This chooses physical encodings but produces no encoded
 * payload.
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/quant_numeric.h>

typedef struct {
    unsigned long long hash;
    unsigned long long ordinal_plus_one;
} quant_plan_slot;

typedef struct {
    yvex_transform_operation_kind operation;
    yvex_deepseek_gguf_transform lowering;
} operation_lowering;

static const operation_lowering operation_lowerings[] = {
    {YVEX_TRANSFORM_OP_IDENTITY, YVEX_DEEPSEEK_GGUF_TRANSFORM_DIRECT},
    {YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR, YVEX_DEEPSEEK_GGUF_TRANSFORM_FP8_E4M3_E8M0},
    {YVEX_TRANSFORM_OP_CHECKED_CAST, YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32},
    {YVEX_TRANSFORM_OP_EXPERT_AGGREGATE, YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4},
};

static const unsigned int exact_qtypes[YVEX_TRANSFORM_DTYPE_REAL + 1u] = {
    [YVEX_TRANSFORM_DTYPE_F32] = YVEX_GGUF_QTYPE_F32,
    [YVEX_TRANSFORM_DTYPE_F16] = YVEX_GGUF_QTYPE_F16,
    [YVEX_TRANSFORM_DTYPE_BF16] = YVEX_GGUF_QTYPE_BF16,
    [YVEX_TRANSFORM_DTYPE_I32] = YVEX_GGUF_QTYPE_I32,
    [YVEX_TRANSFORM_DTYPE_REAL] = YVEX_GGUF_QTYPE_F32,
};

static const yvex_quant_failure_code
    storage_failures[YVEX_GGUF_QTYPE_STORAGE_EXPECTED_ACTUAL_MISMATCH + 1u] = {
        [YVEX_GGUF_QTYPE_STORAGE_INVALID_ARGUMENT] = YVEX_QUANT_FAILURE_INVALID_ARGUMENT,
        [YVEX_GGUF_QTYPE_STORAGE_UNKNOWN_ID] = YVEX_QUANT_FAILURE_UNKNOWN_QTYPE,
        [YVEX_GGUF_QTYPE_STORAGE_REMOVED_ID] = YVEX_QUANT_FAILURE_REMOVED_QTYPE,
        [YVEX_GGUF_QTYPE_STORAGE_RESERVED_ID] = YVEX_QUANT_FAILURE_UNKNOWN_QTYPE,
        [YVEX_GGUF_QTYPE_STORAGE_OUTSIDE_BASELINE] =
            YVEX_QUANT_FAILURE_QTYPE_OUTSIDE_BASELINE,
        [YVEX_GGUF_QTYPE_STORAGE_GEOMETRY_UNAVAILABLE] = YVEX_QUANT_FAILURE_UNKNOWN_QTYPE,
        [YVEX_GGUF_QTYPE_STORAGE_INVALID_RANK] = YVEX_QUANT_FAILURE_INVALID_RANK,
        [YVEX_GGUF_QTYPE_STORAGE_INVALID_DIMENSION] = YVEX_QUANT_FAILURE_INVALID_DIMENSION,
        [YVEX_GGUF_QTYPE_STORAGE_ROW_BLOCK_MISMATCH] = YVEX_QUANT_FAILURE_ROW_DIVISIBILITY,
        [YVEX_GGUF_QTYPE_STORAGE_ELEMENT_COUNT_OVERFLOW] = YVEX_QUANT_FAILURE_ELEMENT_OVERFLOW,
};

struct yvex_quant_plan {
    yvex_quant_plan_summary summary;
    const yvex_transform_ir *ir;
    const yvex_transform_binding *binding;
    const yvex_deepseek_gguf_map *map;
    yvex_quant_decision *decisions;
    quant_plan_slot *index;
    yvex_quant_allocate_fn allocate;
    yvex_quant_release_fn release;
    void *allocator_context;
};

static void *quant_plan_default_allocate(size_t size, void *context) {
    (void)context;
    return calloc(1u, size);
}

static void quant_plan_default_release(void *allocation, void *context) {
    (void)context;
    free(allocation);
}

static int quant_plan_fail(yvex_quant_failure *failure, yvex_quant_failure_code code,
                           unsigned long long terminal, unsigned long long source,
                           unsigned long long expected, unsigned long long actual,
                           unsigned int qtype, yvex_transform_operation_kind operation,
                           yvex_error *err, int status, const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->terminal_ordinal = terminal;
        failure->source_index = source;
        failure->row_index = ULLONG_MAX;
        failure->block_index = ULLONG_MAX;
        failure->expected = expected;
        failure->actual = actual;
        failure->qtype = qtype;
        failure->operation = operation;
    }
    yvex_error_set(err, (yvex_status)status, "quant.plan", message);
    return status;
}

static unsigned long long quant_hash_u64(unsigned long long hash, unsigned long long value) {
    unsigned int index;
    for (index = 0u; index < 8u; ++index) {
        hash ^= (value >> (index * 8u)) & 0xffu;
        hash *= 1099511628211ull;
    }
    return hash;
}

static unsigned long long quant_key_hash(const yvex_transform_logical_key *key) {
    unsigned long long hash = 1469598103934665603ull;
    hash = quant_hash_u64(hash, key->scope);
    hash = quant_hash_u64(hash, key->subsystem);
    hash = quant_hash_u64(hash, key->role);
    hash = quant_hash_u64(hash, key->layer_index);
    hash = quant_hash_u64(hash, key->auxiliary_index);
    return quant_hash_u64(hash, key->group_index);
}

static int quant_key_equal(const yvex_transform_logical_key *left,
                           const yvex_transform_logical_key *right) {
    return left->scope == right->scope && left->subsystem == right->subsystem &&
           left->role == right->role && left->layer_index == right->layer_index &&
           left->auxiliary_index == right->auxiliary_index &&
           left->group_index == right->group_index;
}

static int quant_size_add(size_t left, size_t right, size_t *out) {
    if (!out)
        return 0;
    if (left > SIZE_MAX - right) {
        *out = SIZE_MAX;
        return 0;
    }
    *out = left + right;
    return 1;
}

static int quant_transform_element_count(const yvex_transform_shape *shape,
                                         unsigned long long *out) {
    unsigned long long count = 1u;
    unsigned int dimension;

    if (out)
        *out = 0u;
    if (!shape || !out || !shape->rank || shape->rank > YVEX_TRANSFORM_IR_MAX_RANK)
        return 0;
    for (dimension = 0u; dimension < shape->rank; ++dimension) {
        if (!shape->dims[dimension] || count > ULLONG_MAX / shape->dims[dimension])
            return 0;
        count *= shape->dims[dimension];
    }
    *out = count;
    return 1;
}

static yvex_tensor_scope quant_map_scope(yvex_transform_scope scope) {
    if (scope == YVEX_TRANSFORM_SCOPE_GLOBAL)
        return YVEX_TENSOR_SCOPE_GLOBAL;
    if (scope == YVEX_TRANSFORM_SCOPE_MAIN_LAYER)
        return YVEX_TENSOR_SCOPE_MAIN_LAYER;
    return YVEX_TENSOR_SCOPE_DRAFT;
}

static yvex_deepseek_gguf_transform quant_map_operation(yvex_transform_operation_kind operation) {
    size_t index;
    for (index = 0u; index < sizeof(operation_lowerings) / sizeof(operation_lowerings[0]); ++index)
        if (operation_lowerings[index].operation == operation)
            return operation_lowerings[index].lowering;
    return (yvex_deepseek_gguf_transform)-1;
}

static unsigned int quant_exact_qtype(yvex_transform_dtype dtype) {
    return ((dtype >= YVEX_TRANSFORM_DTYPE_F32 && dtype <= YVEX_TRANSFORM_DTYPE_I32) ||
            dtype == YVEX_TRANSFORM_DTYPE_REAL)
               ? exact_qtypes[dtype]
               : UINT_MAX;
}

static unsigned int quant_candidate_qtype(yvex_quant_profile_kind profile,
                                          const yvex_transform_value *terminal,
                                          const yvex_transform_node *node) {
    if (node->kind == YVEX_TRANSFORM_OP_CHECKED_CAST)
        return YVEX_GGUF_QTYPE_I32;
    if (node->kind == YVEX_TRANSFORM_OP_EXPERT_AGGREGATE)
        return profile == YVEX_QUANT_PROFILE_RELEASE_Q8_Q2 ? YVEX_GGUF_QTYPE_Q2_K
                                                           : YVEX_GGUF_QTYPE_MXFP4;
    if (terminal->precision.flags & YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT) {
        if (profile == YVEX_QUANT_PROFILE_RELEASE_Q8_Q2)
            return YVEX_GGUF_QTYPE_Q8_0;
        if (node->kind == YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR)
            return YVEX_GGUF_QTYPE_F32;
    }
    return quant_exact_qtype(terminal->dtype);
}

static int
quant_descriptor_matches(const yvex_transform_ir *ir, const yvex_transform_value *terminal,
                         const yvex_transform_node *node, const yvex_deepseek_gguf_map *map,
                         const yvex_deepseek_gguf_descriptor *descriptor,
                         unsigned long long ordinal, yvex_quant_failure *failure, yvex_error *err) {
    unsigned int dimension;
    unsigned long long input;

    if (!descriptor || descriptor->role != terminal->logical_key.role ||
        descriptor->scope != quant_map_scope(terminal->logical_key.scope) ||
        descriptor->layer_index != terminal->logical_key.layer_index ||
        descriptor->predictor_index != terminal->logical_key.auxiliary_index ||
        descriptor->transform != quant_map_operation(node->kind) ||
        descriptor->logical_rank != terminal->shape.rank ||
        descriptor->contribution_count != node->input_count) {
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_UNMATCHED_LOWERING, ordinal, ULLONG_MAX,
                               node->input_count, descriptor ? descriptor->contribution_count : 0u,
                               UINT_MAX, node->kind, err, YVEX_ERR_FORMAT,
                               "terminal and GGUF lowering descriptor do not biject");
    }
    for (dimension = 0u; dimension < terminal->shape.rank; ++dimension) {
        unsigned int terminal_axis = terminal->shape.rank - dimension - 1u;
        unsigned int source_axis = terminal_axis;

        if (node->kind == YVEX_TRANSFORM_OP_EXPERT_AGGREGATE) {
            source_axis = terminal_axis == node->axis  ? YVEX_DEEPSEEK_GGUF_AGGREGATED_AXIS
                          : terminal_axis > node->axis ? terminal_axis - 1u
                                                       : terminal_axis;
        }
        if (descriptor->logical_dims[dimension] != terminal->shape.dims[terminal_axis] ||
            descriptor->source_axis_for_logical[dimension] != source_axis) {
            return quant_plan_fail(failure, YVEX_QUANT_FAILURE_UNMATCHED_LOWERING, ordinal,
                                   ULLONG_MAX, terminal->shape.dims[terminal_axis],
                                   descriptor->logical_dims[dimension], UINT_MAX, node->kind, err,
                                   YVEX_ERR_FORMAT, "terminal and lowering physical axes diverge");
        }
    }
    for (input = 0u; input < node->input_count; ++input) {
        const yvex_transform_value *value = yvex_transform_ir_node_input_at(ir, node, input);
        const yvex_transform_source_value *source =
            value ? yvex_transform_ir_source_at(ir, value->source_index) : NULL;
        const yvex_deepseek_gguf_contribution *contribution =
            yvex_model_register_deepseek_v4()->lowering.contribution_at(
                map, descriptor->contribution_offset + input);
        if (!value || !source || !contribution || contribution->descriptor_index != ordinal ||
            strcmp(contribution->source_name, source->source_name) != 0 ||
            contribution->source_dtype != source->source_dtype ||
            contribution->expert_index != source->expert_index) {
            return quant_plan_fail(failure, YVEX_QUANT_FAILURE_UNMATCHED_LOWERING, ordinal,
                                   value ? value->source_index : ULLONG_MAX, input,
                                   contribution ? contribution->descriptor_index : ULLONG_MAX,
                                   UINT_MAX, node->kind, err, YVEX_ERR_FORMAT,
                                   "lowering contribution does not match the exact IR input");
        }
    }
    return YVEX_OK;
}

static int quant_decision_geometry_dims(yvex_quant_decision *decision,
                                        const unsigned long long *dims, unsigned int rank,
                                        yvex_quant_failure *failure, yvex_error *err) {
    yvex_gguf_qtype_storage_result storage;
    yvex_gguf_qtype_storage_status status;
    yvex_quant_failure_code code;

    status = yvex_gguf_qtype_tensor_storage(decision->qtype, dims, rank, &storage);
    if (status != YVEX_GGUF_QTYPE_STORAGE_OK) {
        code = status <= YVEX_GGUF_QTYPE_STORAGE_EXPECTED_ACTUAL_MISMATCH &&
                       storage_failures[status] != 0
                   ? storage_failures[status]
                   : YVEX_QUANT_FAILURE_BYTE_OVERFLOW;
        return quant_plan_fail(failure, code, decision->terminal_ordinal, ULLONG_MAX, 0u, status,
                               decision->qtype, decision->operation, err, YVEX_ERR_BOUNDS,
                               "selected qtype cannot represent the lowering tensor geometry");
    }
    decision->rank = rank;
    memcpy(decision->dims, dims, sizeof(decision->dims));
    decision->row_axis = 0u;
    decision->row_width = storage.row_width;
    decision->row_count = storage.row_count;
    decision->element_count = storage.element_count;
    decision->encoded_bytes = storage.total_bytes;
    return YVEX_OK;
}

static yvex_quant_failure_code
quant_capability_failure(const yvex_quant_numeric_capability *capability) {
    if (!capability || !capability->identity_known)
        return YVEX_QUANT_FAILURE_UNKNOWN_QTYPE;
    if (capability->refusal == YVEX_QUANT_REFUSAL_REMOVED_IDENTITY)
        return YVEX_QUANT_FAILURE_REMOVED_QTYPE;
    if (capability->refusal == YVEX_QUANT_REFUSAL_OUTSIDE_PINNED_BASELINE)
        return YVEX_QUANT_FAILURE_QTYPE_OUTSIDE_BASELINE;
    if (capability->encoder_available && !capability->reference_decoder_available)
        return YVEX_QUANT_FAILURE_DECODER_UNAVAILABLE;
    return YVEX_QUANT_FAILURE_ENCODER_UNAVAILABLE;
}

static int quant_decision_geometry(yvex_quant_decision *decision,
                                   const yvex_deepseek_gguf_descriptor *descriptor,
                                   yvex_quant_failure *failure, yvex_error *err) {
    return quant_decision_geometry_dims(decision, descriptor->logical_dims,
                                        descriptor->logical_rank, failure, err);
}

/*
 * Derive deterministic identity for one complete physical decision.
 *
 * Writes SHA-256 identity into the owned decision.
 */
static int quant_decision_identity(yvex_quant_decision *decision) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int dimension;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.quant.decision.v1") ||
        !yvex_sha256_update_u64(&hash, decision->logical_key.scope) ||
        !yvex_sha256_update_u64(&hash, decision->logical_key.subsystem) ||
        !yvex_sha256_update_u64(&hash, decision->logical_key.role) ||
        !yvex_sha256_update_u64(&hash, decision->logical_key.layer_index) ||
        !yvex_sha256_update_u64(&hash, decision->logical_key.auxiliary_index) ||
        !yvex_sha256_update_u64(&hash, decision->logical_key.group_index) ||
        !yvex_sha256_update_u64(&hash, decision->terminal_value_id) ||
        !yvex_sha256_update_u64(&hash, decision->operation) || !yvex_sha256_update_u64(&hash, decision->qtype) ||
        !yvex_sha256_update_u64(&hash, decision->rank))
        return 0;
    for (dimension = 0u; dimension < decision->rank; ++dimension)
        if (!yvex_sha256_update_u64(&hash, decision->dims[dimension]))
            return 0;
    if (!yvex_sha256_update_u64(&hash, decision->row_axis) ||
        !yvex_sha256_update_u64(&hash, decision->element_count) ||
        !yvex_sha256_update_u64(&hash, decision->encoded_bytes) ||
        !yvex_sha256_update_u64(&hash, decision->approximation) ||
        !yvex_sha256_update_u64(&hash, decision->calibration) ||
        !yvex_sha256_update_u64(&hash, decision->numeric_contract_version))
        return 0;
    if (decision->policy_bound &&
        (!yvex_sha256_update_text(&hash, "yvex.quant.decision.policy.v2") ||
         !yvex_sha256_update_u64(&hash, decision->collection) ||
         !yvex_sha256_update_u64(&hash, decision->policy_rule_ordinal) ||
         !yvex_sha256_update_u64(&hash, decision->policy_priority) ||
         !yvex_sha256_update_u64(&hash, (unsigned int)decision->policy_requires_imatrix) ||
         !yvex_sha256_update_text(&hash, decision->policy_label) ||
         !yvex_sha256_update_text(&hash, decision->physical_tensor_name) ||
         !yvex_sha256_update_u64(&hash, decision->physical_expert_count) ||
         !yvex_sha256_update_text(&hash, decision->policy_rule_identity)))
        return 0;
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, decision->decision_identity);
    return 1;
}

/*
 * Account one decision into its candidate profile summary.
 *
 * Returns false for invalid qtype or any counter overflow.
 */
static int quant_summary_add(yvex_quant_candidate_summary *summary,
                             const yvex_quant_decision *decision) {
    unsigned long long *class_bytes;

    if (!summary || !decision || decision->qtype > YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID ||
        ULLONG_MAX - summary->encoded_bytes < decision->encoded_bytes ||
        summary->terminal_count == ULLONG_MAX ||
        summary->qtype_tensor_counts[decision->qtype] == ULLONG_MAX)
        return 0;
    if (decision->qtype == YVEX_GGUF_QTYPE_Q8_0)
        class_bytes = &summary->q8_0_bytes;
    else if (decision->qtype == YVEX_GGUF_QTYPE_Q2_K)
        class_bytes = &summary->q2_k_bytes;
    else if (decision->qtype == YVEX_GGUF_QTYPE_IQ2_XXS)
        class_bytes = &summary->iq2_xxs_bytes;
    else if (decision->qtype == YVEX_GGUF_QTYPE_MXFP4)
        class_bytes = &summary->mxfp4_bytes;
    else
        class_bytes = &summary->exact_scalar_bytes;
    if (ULLONG_MAX - *class_bytes < decision->encoded_bytes)
        return 0;
    summary->encoded_bytes += decision->encoded_bytes;
    summary->terminal_count++;
    summary->qtype_tensor_counts[decision->qtype]++;
    *class_bytes += decision->encoded_bytes;
    if (decision->calibration == YVEX_QUANT_CALIBRATION_REQUIRED)
        summary->calibration_required = 1;
    if (!decision->cpu_compute_available || !decision->cuda_compute_available)
        summary->compute_admissible = 0;
    return 1;
}

/*
 * Build one candidate decision from typed IR and numeric capability.
 *
 * Fills decision geometry, constraints, compute facts, and identity. Typed refusal covers
 * precision, codec, binding, geometry, or identity.
 */
static int quant_build_qtype_decision(unsigned int qtype,
                                          const yvex_transform_binding *binding,
                                          const yvex_transform_value *terminal,
                                          const yvex_transform_node *node,
                                          const yvex_deepseek_gguf_descriptor *descriptor,
                                          unsigned long long ordinal, yvex_quant_decision *decision,
                                          yvex_quant_failure *failure, yvex_error *err) {
    const yvex_quant_numeric_capability *capability;
    yvex_transform_physical_decision binding_decision;
    yvex_transform_failure transform_failure;
    int rc;
    if (qtype == UINT_MAX)
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_PRECISION_CONSTRAINT, ordinal,
                               ULLONG_MAX, 1u, 0u, qtype, node->kind, err, YVEX_ERR_UNSUPPORTED,
                               "terminal dtype has no admitted physical scalar representation");
    capability = yvex_quant_numeric_capability_at(qtype);
    if (!capability || !capability->encoder_available || !capability->reference_decoder_available) {
        return quant_plan_fail(failure, quant_capability_failure(capability), ordinal, ULLONG_MAX,
                               1u, 0u, qtype, node->kind, err, YVEX_ERR_UNSUPPORTED,
                               "selected profile requires an unavailable codec");
    }
    if ((unsigned int)node->kind >= sizeof(capability->transform_kind_mask) * CHAR_BIT ||
        !(capability->transform_kind_mask & (1u << (unsigned int)node->kind)))
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_PRECISION_CONSTRAINT, ordinal,
                               ULLONG_MAX, capability->transform_kind_mask,
                               (unsigned long long)node->kind, qtype, node->kind, err,
                               YVEX_ERR_UNSUPPORTED,
                               "selected qtype does not admit the terminal operation");
    memset(decision, 0, sizeof(*decision));
    decision->logical_key = terminal->logical_key;
    decision->terminal_ordinal = ordinal;
    decision->terminal_value_id = terminal->id;
    decision->node_id = node->id;
    decision->role = terminal->logical_key.role;
    decision->collection = descriptor->collection;
    decision->scope = terminal->logical_key.scope;
    decision->operation = node->kind;
    decision->qtype = qtype;
    decision->physical_class = qtype == YVEX_GGUF_QTYPE_F32 || qtype == YVEX_GGUF_QTYPE_F16 ||
                                       qtype == YVEX_GGUF_QTYPE_BF16 || qtype == YVEX_GGUF_QTYPE_I32
                                   ? YVEX_QUANT_PHYSICAL_EXACT_SCALAR
                                   : YVEX_QUANT_PHYSICAL_BLOCK_QUANTIZED;
    decision->approximation = qtype == YVEX_GGUF_QTYPE_Q8_0 || qtype == YVEX_GGUF_QTYPE_Q2_K ||
                              qtype == YVEX_GGUF_QTYPE_IQ2_XXS ||
                              qtype == YVEX_GGUF_QTYPE_MXFP4;
    if (decision->approximation && !terminal->precision.approximation_allowed)
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_APPROXIMATION_FORBIDDEN, ordinal,
                               ULLONG_MAX, 0u, 1u, qtype, node->kind, err, YVEX_ERR_FORMAT,
                               "profile selected approximation for an exact terminal");
    memset(&binding_decision, 0, sizeof(binding_decision));
    binding_decision.physical_class = capability->physical_class_mask;
    binding_decision.encoding_id = qtype;
    binding_decision.approximation_selected = decision->approximation;
    rc = yvex_transform_binding_decision_validate(binding, ordinal, &binding_decision,
                                                  &transform_failure, err);
    if (rc != YVEX_OK)
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_PRECISION_CONSTRAINT, ordinal,
                               ULLONG_MAX, terminal->precision.allowed_physical_classes,
                               capability->physical_class_mask, qtype, node->kind, err,
                               YVEX_ERR_FORMAT,
                               "profile decision violates the terminal precision constraint");
    decision->calibration = capability->calibration;
    decision->reference_decoder_required = 1;
    decision->cpu_compute_available = capability->dedicated_cpu_compute_available;
    decision->cuda_compute_available = capability->dedicated_cuda_compute_available;
    decision->numeric_contract_version = capability->numeric_contract_version;
    rc = quant_decision_geometry(decision, descriptor, failure, err);
    if (rc != YVEX_OK)
        return rc;
    if (!quant_decision_identity(decision))
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, ordinal, ULLONG_MAX, 1u,
                               0u, qtype, node->kind, err, YVEX_ERR_BOUNDS,
                               "decision identity encoding failed");
    return YVEX_OK;
}

static int quant_build_candidate_decision(yvex_quant_profile_kind profile,
                                          const yvex_transform_binding *binding,
                                          const yvex_transform_value *terminal,
                                          const yvex_transform_node *node,
                                          const yvex_deepseek_gguf_descriptor *descriptor,
                                          unsigned long long ordinal, yvex_quant_decision *decision,
                                          yvex_quant_failure *failure, yvex_error *err) {
    return quant_build_qtype_decision(quant_candidate_qtype(profile, terminal, node), binding,
                                      terminal, node, descriptor, ordinal, decision, failure, err);
}

static int payload_u64(yvex_sha256 *hash, unsigned long long value)
{
    return yvex_sha256_update_u64(hash, value);
}

static int quant_payload_source_identity(yvex_sha256 *hash,
                                         const yvex_transform_source_value *source)
{
    unsigned int dimension;

    if (!hash || !source || !yvex_sha256_update_text(hash, source->source_name) ||
        !yvex_sha256_update_text(hash, source->shard_name) ||
        !payload_u64(hash, source->source_tensor_index) || !payload_u64(hash, source->requirement_index) ||
        !payload_u64(hash, source->source_snapshot_identity) || !payload_u64(hash, source->source_dtype) ||
        !payload_u64(hash, source->value_dtype) || !payload_u64(hash, source->shape.rank))
        return 0;
    for (dimension = 0u; dimension < source->shape.rank; ++dimension)
        if (!payload_u64(hash, source->shape.dims[dimension]))
            return 0;
    return payload_u64(hash, source->relative_begin) && payload_u64(hash, source->relative_end) &&
           payload_u64(hash, source->requirement_identity) &&
           payload_u64(hash, source->expert_index) && payload_u64(hash, source->required_uses);
}

/*
 * Bind one transformation node and its ordered byte inputs.
 *
 * Records execution geometry without the enclosing Transform IR identity.
 */
static int quant_payload_node_identity(yvex_sha256 *hash, const yvex_transform_ir *ir,
                                       const yvex_transform_node *node)
{
    unsigned long long input;
    unsigned int axis;

    if (!hash || !ir || !node || !payload_u64(hash, node->kind) || !payload_u64(hash, node->input_count) ||
        !payload_u64(hash, node->axis) || !payload_u64(hash, node->permutation_rank))
        return 0;
    for (axis = 0u; axis < node->permutation_rank; ++axis)
        if (!payload_u64(hash, node->permutation[axis]))
            return 0;
    if (!payload_u64(hash, node->expert_count) || !payload_u64(hash, node->packing_factor) ||
        !payload_u64(hash, node->scale_group_width) || !payload_u64(hash, node->scale_block_rows) ||
        !payload_u64(hash, node->scale_block_columns) || !payload_u64(hash, node->numeric) ||
        !payload_u64(hash, node->ordering) || !payload_u64(hash, node->payload_execution_required))
        return 0;
    for (input = 0ull; input < node->input_count; ++input) {
        const yvex_transform_value *value = yvex_transform_ir_node_input_at(ir, node, input);
        const yvex_transform_source_value *source;

        if (!value || !payload_u64(hash, input) || !payload_u64(hash, value->kind) ||
            !payload_u64(hash, value->dtype) || !payload_u64(hash, value->shape.rank))
            return 0;
        for (axis = 0u; axis < value->shape.rank; ++axis)
            if (!payload_u64(hash, value->shape.dims[axis]))
                return 0;
        if (value->kind != YVEX_TRANSFORM_VALUE_SOURCE) {
            if (!payload_u64(hash, value->semantic_id) || !payload_u64(hash, value->producer_node_id))
                return 0;
            continue;
        }
        source = yvex_transform_ir_source_at(ir, value->source_index);
        if (!quant_payload_source_identity(hash, source))
            return 0;
    }
    return 1;
}

static int quant_payload_decision_identity(yvex_sha256 *hash,
                                           const yvex_quant_decision *decision)
{
    unsigned int dimension;

    if (!hash || !decision || !payload_u64(hash, decision->terminal_ordinal) ||
        !payload_u64(hash, decision->qtype) || !payload_u64(hash, decision->rank))
        return 0;
    for (dimension = 0u; dimension < decision->rank; ++dimension)
        if (!payload_u64(hash, decision->dims[dimension]))
            return 0;
    return payload_u64(hash, decision->row_axis) && payload_u64(hash, decision->row_width) &&
           payload_u64(hash, decision->row_count) && payload_u64(hash, decision->element_count) &&
           payload_u64(hash, decision->encoded_bytes) && payload_u64(hash, decision->approximation) &&
           payload_u64(hash, decision->calibration) && payload_u64(hash, decision->numeric_contract_version);
}

/*
 * Derive the byte-execution recipe identity independently from semantic IR identity.
 *
 * Writes only the payload-plan identity. Excludes GGUF names, mapping provenance, and layout;
 * writer identity binds those.
 */
static int quant_payload_plan_identity(yvex_quant_plan *plan)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long ordinal;

    if (!plan || !plan->ir)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.quant.payload.plan.v2") ||
        !payload_u64(&hash, plan->summary.source_snapshot_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.required_payload_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.calibration_identity) ||
        !yvex_sha256_update_u64(&hash, plan->summary.decision_count))
        return 0;
    for (ordinal = 0ull; ordinal < plan->summary.decision_count; ++ordinal) {
        const yvex_transform_value *terminal =
            yvex_transform_ir_terminal_at(plan->ir, ordinal);
        const yvex_transform_node *node =
            terminal ? yvex_transform_ir_node_at(plan->ir, terminal->producer_node_id) : NULL;

        if (!node || !quant_payload_decision_identity(&hash, &plan->decisions[ordinal]) ||
            !quant_payload_node_identity(&hash, plan->ir, node))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, plan->summary.payload_plan_identity);
    return 1;
}

/*
 * Derive canonical identity for one complete physical quant plan.
 *
 * Writes the profile identity into the owned summary.
 */
static int quant_plan_identity(yvex_quant_plan *plan) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long ordinal;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.quant.plan.v1") ||
        !yvex_sha256_update_u64(&hash, plan->summary.schema_version) ||
        !yvex_sha256_update_text(&hash, plan->summary.profile_name) ||
        !yvex_sha256_update_text(&hash, plan->summary.transform_identity) ||
        !yvex_sha256_update_u64(&hash, plan->summary.source_snapshot_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.required_payload_identity) ||
        !yvex_sha256_update_u64(&hash, plan->summary.mapping_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.calibration_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.backend_compute_contract) ||
        !yvex_sha256_update_u64(&hash, plan->summary.decision_count))
        return 0;
    if (plan->summary.policy_identity[0] &&
        (!yvex_sha256_update_text(&hash, "yvex.quant.physical.variant.v1") ||
         !yvex_sha256_update_text(&hash, plan->summary.policy_identity) ||
         !yvex_sha256_update_text(&hash, plan->summary.imatrix_identity)))
        return 0;
    for (ordinal = 0u; ordinal < plan->summary.decision_count; ++ordinal) {
        if (!yvex_sha256_update_text(&hash, plan->decisions[ordinal].decision_identity))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, plan->summary.profile_identity);
    yvex_core_text_copy(plan->summary.physical_variant_identity,
                        sizeof(plan->summary.physical_variant_identity),
                        plan->summary.profile_identity);
    return quant_payload_plan_identity(plan);
}

typedef struct {
    const yvex_transform_ir *ir;
    const yvex_transform_binding *binding;
    const yvex_deepseek_gguf_map *map;
    const yvex_transform_ir_summary *ir_summary;
    const yvex_transform_binding_summary *binding_summary;
    yvex_quant_plan_options options;
    yvex_quant_plan *plan;
    yvex_quant_failure *failure;
    yvex_error *err;
    const char *profile_name;
    unsigned long long mapping_identity;
    int explicit_plan;
} quant_build_context;

static int quant_binding_identity_validate(quant_build_context *context) {
    if (strcmp(context->ir_summary->transform_identity,
               context->binding_summary->transform_identity) != 0)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_TRANSFORM_IDENTITY, ULLONG_MAX,
                               ULLONG_MAX, 1u, 0u, UINT_MAX, YVEX_TRANSFORM_OP_COUNT, context->err,
                               YVEX_ERR_FORMAT, "binding and IR identities diverge");
    if (context->ir_summary->source_snapshot_identity !=
        context->binding_summary->source_snapshot_identity)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_SOURCE_IDENTITY, ULLONG_MAX,
                               ULLONG_MAX, context->ir_summary->source_snapshot_identity,
                               context->binding_summary->source_snapshot_identity, UINT_MAX,
                               YVEX_TRANSFORM_OP_COUNT, context->err, YVEX_ERR_FORMAT,
                               "binding and IR source identities diverge");
    if (strcmp(context->ir_summary->required_payload_identity,
               context->binding_summary->required_payload_identity) != 0)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_PAYLOAD_IDENTITY, ULLONG_MAX,
                               ULLONG_MAX, 1u, 0u, UINT_MAX, YVEX_TRANSFORM_OP_COUNT, context->err,
                               YVEX_ERR_FORMAT, "binding and IR payload identities diverge");
    return YVEX_OK;
}

static int quant_build_allocate(quant_build_context *context,
                                const yvex_quant_plan_options *options) {
    size_t decision_bytes;
    size_t index_bytes;
    size_t owned_bytes = SIZE_MAX;
    unsigned long long count = context->ir_summary->terminal_count;
    memset(&context->options, 0, sizeof(context->options));
    context->options.allocate = quant_plan_default_allocate;
    context->options.release = quant_plan_default_release;
    context->options.maximum_owned_bytes = 16u * 1024u * 1024u;
    if (options)
        context->options = *options;
    if (!context->options.allocate)
        context->options.allocate = quant_plan_default_allocate;
    if (!context->options.release)
        context->options.release = quant_plan_default_release;
    if (!context->options.maximum_owned_bytes)
        context->options.maximum_owned_bytes = 16u * 1024u * 1024u;
    context->plan = (yvex_quant_plan *)context->options.allocate(sizeof(*context->plan),
                                                                 context->options.context);
    if (!context->plan)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_ALLOCATION, ULLONG_MAX,
                               ULLONG_MAX, sizeof(*context->plan), 0u, UINT_MAX,
                               YVEX_TRANSFORM_OP_COUNT, context->err, YVEX_ERR_NOMEM,
                               context->explicit_plan
                                   ? "explicit quantization plan allocation failed"
                                   : "quantization plan allocation failed");
    memset(context->plan, 0, sizeof(*context->plan));
    context->plan->allocate = context->options.allocate;
    context->plan->release = context->options.release;
    context->plan->allocator_context = context->options.context;
    context->plan->ir = context->ir;
    context->plan->binding = context->binding;
    context->plan->map = context->map;
    context->plan->summary.schema_version = YVEX_QUANT_PROFILE_SCHEMA_VERSION;
    context->plan->summary.state = YVEX_QUANT_PLAN_BUILDING;
    context->plan->summary.terminal_count = count;
    context->plan->summary.source_value_count = context->ir_summary->source_value_count;
    context->plan->summary.source_snapshot_identity = context->ir_summary->source_snapshot_identity;
    context->plan->summary.mapping_identity = context->mapping_identity;
    yvex_core_text_copy(context->plan->summary.profile_name,
                        sizeof(context->plan->summary.profile_name),
                        context->profile_name);
    yvex_core_text_copy(context->plan->summary.transform_identity,
                        sizeof(context->plan->summary.transform_identity),
                        context->ir_summary->transform_identity);
    yvex_core_text_copy(context->plan->summary.required_payload_identity,
                        sizeof(context->plan->summary.required_payload_identity),
                        context->ir_summary->required_payload_identity);
    yvex_core_text_copy(context->plan->summary.backend_compute_contract,
                        sizeof(context->plan->summary.backend_compute_contract),
                        "cpu-cuda-encoded-row-dot-v1");
    yvex_core_text_copy(context->plan->summary.calibration_identity,
                        sizeof(context->plan->summary.calibration_identity),
                        "no-calibration-required");
    if (!yvex_core_power_of_two_capacity(count, 1ull, 1ull, 2ull,
                                         &context->plan->summary.index_capacity) ||
        count > SIZE_MAX / sizeof(*context->plan->decisions) ||
        context->plan->summary.index_capacity > SIZE_MAX / sizeof(*context->plan->index))
        return quant_plan_fail(
            context->failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, ULLONG_MAX, ULLONG_MAX, SIZE_MAX,
            count, UINT_MAX, YVEX_TRANSFORM_OP_COUNT, context->err, YVEX_ERR_BOUNDS,
            context->explicit_plan ? "explicit plan allocation geometry overflowed"
                                   : "quantization plan allocation geometry overflowed");
    decision_bytes = (size_t)count * sizeof(*context->plan->decisions);
    index_bytes = (size_t)context->plan->summary.index_capacity * sizeof(*context->plan->index);
    if (!quant_size_add(sizeof(*context->plan), decision_bytes, &owned_bytes) ||
        !quant_size_add(owned_bytes, index_bytes, &owned_bytes) ||
        owned_bytes > context->options.maximum_owned_bytes)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_RESOURCE_BUDGET, ULLONG_MAX,
                               ULLONG_MAX, context->options.maximum_owned_bytes, owned_bytes,
                               UINT_MAX, YVEX_TRANSFORM_OP_COUNT, context->err, YVEX_ERR_BOUNDS,
                               context->explicit_plan
                                   ? "explicit plan ownership budget exceeded"
                                   : "quantization plan ownership budget exceeded");
    context->plan->decisions = (yvex_quant_decision *)context->plan->allocate(
        decision_bytes, context->plan->allocator_context);
    context->plan->index =
        (quant_plan_slot *)context->plan->allocate(index_bytes, context->plan->allocator_context);
    if (!context->plan->decisions || !context->plan->index)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_ALLOCATION, ULLONG_MAX,
                               ULLONG_MAX, owned_bytes - sizeof(*context->plan), 0u, UINT_MAX,
                               YVEX_TRANSFORM_OP_COUNT, context->err, YVEX_ERR_NOMEM,
                               context->explicit_plan
                                   ? "explicit decision/index allocation failed"
                                   : "quantization decision/index allocation failed");
    memset(context->plan->decisions, 0, decision_bytes);
    memset(context->plan->index, 0, index_bytes);
    context->plan->summary.owned_bytes = owned_bytes;
    context->plan->summary.peak_builder_bytes = owned_bytes;
    return YVEX_OK;
}

static int quant_index_add(quant_build_context *context, yvex_quant_decision *decision,
                           unsigned long long ordinal, const char *duplicate_message,
                           const char *exhausted_message) {
    unsigned long long hash = quant_key_hash(&decision->logical_key);
    unsigned long long slot = hash & (context->plan->summary.index_capacity - 1u);
    unsigned long long probe;

    for (probe = 0u; probe < context->plan->summary.index_capacity; ++probe) {
        if (!context->plan->index[slot].ordinal_plus_one)
            break;
        if (context->plan->index[slot].hash == hash &&
            quant_key_equal(
                &context->plan->decisions[context->plan->index[slot].ordinal_plus_one - 1u]
                     .logical_key,
                &decision->logical_key))
            return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_DUPLICATE_DECISION, ordinal,
                                   ULLONG_MAX, 1u, 2u, decision->qtype, decision->operation,
                                   context->err, YVEX_ERR_FORMAT, duplicate_message);
        slot = (slot + 1u) & (context->plan->summary.index_capacity - 1u);
    }
    if (probe == context->plan->summary.index_capacity)
        return quant_plan_fail(
            context->failure, YVEX_QUANT_FAILURE_RESOURCE_BUDGET, ordinal, ULLONG_MAX,
            context->plan->summary.index_capacity, context->plan->summary.index_capacity,
            decision->qtype, decision->operation, context->err, YVEX_ERR_BOUNDS, exhausted_message);
    if (ULLONG_MAX - context->plan->summary.qtype_encoded_bytes[decision->qtype] <
        decision->encoded_bytes)
        return quant_plan_fail(
            context->failure, YVEX_QUANT_FAILURE_RESOURCE_BUDGET, ordinal, ULLONG_MAX,
            ULLONG_MAX, decision->encoded_bytes, decision->qtype, decision->operation,
            context->err, YVEX_ERR_BOUNDS, "qtype byte accounting overflowed");
    context->plan->index[slot].hash = hash;
    context->plan->index[slot].ordinal_plus_one = ordinal + 1u;
    context->plan->summary.decision_count++;
    context->plan->summary.qtype_tensor_counts[decision->qtype]++;
    context->plan->summary.qtype_encoded_bytes[decision->qtype] += decision->encoded_bytes;
    if ((unsigned int)decision->role < YVEX_TENSOR_ROLE_COUNT)
        context->plan->summary.role_tensor_counts[decision->role]++;
    return YVEX_OK;
}

/*
 * Validate and materialize one caller-selected physical decision.
 *
 * Typed refusal covers binding, codec, shape, precision, or identity.
 */
static int quant_explicit_decision_build(quant_build_context *context,
                                         const yvex_quant_explicit_decision *spec,
                                         unsigned long long ordinal,
                                         yvex_quant_candidate_summary *candidate) {
    const yvex_transform_value *terminal = yvex_transform_ir_terminal_at(context->ir, ordinal);
    const yvex_transform_node *node =
        terminal ? yvex_transform_ir_node_at(context->ir, terminal->producer_node_id) : NULL;
    const yvex_transform_value *input =
        node && node->input_count == 1u
            ? yvex_transform_ir_node_input_at(context->ir, node, 0u) : NULL;
    const yvex_transform_source_value *source =
        input && input->kind == YVEX_TRANSFORM_VALUE_SOURCE
            ? yvex_transform_ir_source_at(context->ir, input->source_index) : NULL;
    const yvex_quant_numeric_capability *capability = yvex_quant_numeric_capability_at(spec->qtype);
    yvex_quant_decision *decision = &context->plan->decisions[ordinal];
    yvex_transform_physical_decision binding_decision;
    yvex_transform_failure transform_failure;
    unsigned long long logical_elements;
    int logical_geometry_ok;
    int rc;

    if (!terminal || !node || terminal->canonical_ordinal != ordinal ||
        yvex_transform_binding_terminal_at(context->binding, ordinal) != terminal ||
        yvex_transform_binding_terminal_operation(context->binding, ordinal) != node)
        return quant_plan_fail(
            context->failure, YVEX_QUANT_FAILURE_MISSING_DECISION, ordinal, ULLONG_MAX, ordinal,
            terminal ? terminal->canonical_ordinal : ULLONG_MAX, spec->qtype,
            node ? node->kind : YVEX_TRANSFORM_OP_COUNT, context->err, YVEX_ERR_FORMAT,
            "binding does not expose the canonical terminal operation");
    if (!capability || !capability->identity_known)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_UNKNOWN_QTYPE, ordinal,
                               ULLONG_MAX, 1u, 0u, spec->qtype, node->kind, context->err,
                               YVEX_ERR_UNSUPPORTED, "explicit qtype identity is unknown");
    if (!capability->encoder_available || !capability->reference_decoder_available)
        return quant_plan_fail(context->failure, quant_capability_failure(capability), ordinal,
                               ULLONG_MAX, 1u, 0u, spec->qtype, node->kind, context->err,
                               YVEX_ERR_UNSUPPORTED, "explicit plan selected an unavailable codec");
    if (node->kind >= YVEX_TRANSFORM_OP_COUNT ||
        !(capability->transform_kind_mask & (1u << node->kind)))
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_UNSUPPORTED_OPERATION, ordinal,
                               ULLONG_MAX, capability->transform_kind_mask, node->kind, spec->qtype,
                               node->kind, context->err, YVEX_ERR_UNSUPPORTED,
                               "explicit qtype does not admit the terminal operation");
    if (spec->row_axis != 0u)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_INVALID_ROW_AXIS, ordinal,
                               ULLONG_MAX, 0u, spec->row_axis, spec->qtype, node->kind,
                               context->err, YVEX_ERR_UNSUPPORTED,
                               "current physical geometry requires qtype rows on axis zero");
    memset(decision, 0, sizeof(*decision));
    decision->logical_key = terminal->logical_key;
    decision->terminal_ordinal = ordinal;
    decision->terminal_value_id = terminal->id;
    decision->node_id = node->id;
    decision->role = terminal->logical_key.role;
    decision->scope = terminal->logical_key.scope;
    decision->operation = node->kind;
    if (source)
        yvex_core_text_copy(decision->physical_tensor_name,
                            sizeof(decision->physical_tensor_name), source->source_name);
    decision->qtype = spec->qtype;
    decision->physical_class = capability->physical_class_mask == YVEX_TRANSFORM_PHYSICAL_QUANTIZED
                                   ? YVEX_QUANT_PHYSICAL_BLOCK_QUANTIZED
                                   : YVEX_QUANT_PHYSICAL_EXACT_SCALAR;
    decision->approximation = spec->approximation;
    if (decision->approximation && !terminal->precision.approximation_allowed)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_APPROXIMATION_FORBIDDEN,
                               ordinal, ULLONG_MAX, 0u, 1u, spec->qtype, node->kind, context->err,
                               YVEX_ERR_FORMAT,
                               "explicit plan selected approximation for an exact terminal");
    memset(&binding_decision, 0, sizeof(binding_decision));
    binding_decision.physical_class = capability->physical_class_mask;
    binding_decision.encoding_id = spec->qtype;
    binding_decision.approximation_selected = spec->approximation;
    rc = yvex_transform_binding_decision_validate(context->binding, ordinal, &binding_decision,
                                                  &transform_failure, context->err);
    if (rc != YVEX_OK)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_PRECISION_CONSTRAINT, ordinal,
                               ULLONG_MAX, terminal->precision.allowed_physical_classes,
                               capability->physical_class_mask, spec->qtype, node->kind,
                               context->err, YVEX_ERR_FORMAT,
                               "explicit decision violates the terminal precision constraint");
    decision->calibration = capability->calibration;
    decision->reference_decoder_required = 1;
    decision->cpu_compute_available = capability->dedicated_cpu_compute_available;
    decision->cuda_compute_available = capability->dedicated_cuda_compute_available;
    decision->numeric_contract_version = capability->numeric_contract_version;
    rc = quant_decision_geometry_dims(decision, spec->dims, spec->rank, context->failure,
                                      context->err);
    logical_geometry_ok = quant_transform_element_count(&terminal->shape, &logical_elements);
    if (rc == YVEX_OK && (!logical_geometry_ok || logical_elements != decision->element_count))
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_INVALID_DIMENSION, ordinal,
                               ULLONG_MAX, logical_geometry_ok ? logical_elements : ULLONG_MAX,
                               decision->element_count, spec->qtype, node->kind, context->err,
                               YVEX_ERR_BOUNDS,
                               "physical decision element count differs from its logical terminal");
    if (rc != YVEX_OK)
        return rc;
    if (!quant_decision_identity(decision))
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, ordinal,
                               ULLONG_MAX, 1u, 0u, spec->qtype, node->kind, context->err,
                               YVEX_ERR_BOUNDS, "explicit decision identity failed");
    if (!quant_summary_add(candidate, decision))
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, ordinal,
                               ULLONG_MAX, ULLONG_MAX, candidate->encoded_bytes, spec->qtype,
                               node->kind, context->err, YVEX_ERR_BOUNDS,
                               "explicit plan byte accounting overflowed");
    return quant_index_add(context, decision, ordinal,
                           "duplicate logical terminal decision refused",
                           "explicit decision index exhausted");
}

static void quant_summary_select(yvex_quant_plan_summary *summary,
                                 const yvex_quant_candidate_summary *candidate) {
    summary->encoded_bytes = candidate->encoded_bytes;
    summary->exact_scalar_bytes = candidate->exact_scalar_bytes;
    summary->q8_0_bytes = candidate->q8_0_bytes;
    summary->q2_k_bytes = candidate->q2_k_bytes;
    summary->iq2_xxs_bytes = candidate->iq2_xxs_bytes;
    summary->mxfp4_bytes = candidate->mxfp4_bytes;
    summary->calibration_required = candidate->calibration_required;
}

/*
 * Build and seal a caller-described physical plan over a complete binding.
 *
 * IR/binding identities, lowering identity, exact decisions, and budget. Releases partial
 * ownership and publishes a typed plan refusal.
 */
int yvex_quant_plan_build_explicit(yvex_quant_plan **out, const yvex_transform_ir *ir,
                                   const yvex_transform_binding *binding, const char *profile_name,
                                   unsigned long long lowering_identity,
                                   const yvex_quant_explicit_decision *decisions,
                                   unsigned long long decision_count,
                                   const yvex_quant_plan_options *options,
                                   yvex_quant_failure *failure, yvex_error *err) {
    quant_build_context context;
    yvex_quant_candidate_summary candidate;
    unsigned long long ordinal;
    int rc;

    memset(&context, 0, sizeof(context));
    context.ir = ir;
    context.binding = binding;
    context.ir_summary = yvex_transform_ir_summary_get(ir);
    context.binding_summary = yvex_transform_binding_summary_get(binding);
    context.profile_name = profile_name;
    context.mapping_identity = lowering_identity;
    context.explicit_plan = 1;
    context.failure = failure;
    context.err = err;
    if (out)
        *out = NULL;
    if (!out || !ir || !binding || !profile_name || !profile_name[0] ||
        strlen(profile_name) >= sizeof(((yvex_quant_plan_summary *)0)->profile_name) ||
        !lowering_identity || !decisions || !context.ir_summary || !context.binding_summary ||
        yvex_transform_binding_ir(binding) != ir || !context.ir_summary->complete ||
        !context.binding_summary->complete) {
        return quant_plan_fail(
            failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, ULLONG_MAX, ULLONG_MAX, 1u, 0u, UINT_MAX,
            YVEX_TRANSFORM_OP_COUNT, err, YVEX_ERR_INVALID_ARG,
            "complete IR/binding, profile, lowering identity, and decisions are required");
    }
    if (decision_count != context.ir_summary->terminal_count) {
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_MISSING_DECISION, ULLONG_MAX, ULLONG_MAX,
                               context.ir_summary->terminal_count, decision_count, UINT_MAX,
                               YVEX_TRANSFORM_OP_COUNT, err, YVEX_ERR_FORMAT,
                               "explicit plan must decide every canonical terminal exactly once");
    }
    rc = quant_binding_identity_validate(&context);
    if (rc != YVEX_OK)
        return rc;

    rc = quant_build_allocate(&context, options);
    if (rc != YVEX_OK) {
        yvex_quant_plan_release(&context.plan);
        return rc;
    }
    memset(&candidate, 0, sizeof(candidate));
    candidate.kind = YVEX_QUANT_PROFILE_RELEASE_Q8_Q2;
    candidate.name = context.plan->summary.profile_name;
    candidate.compute_admissible = 1;

    for (ordinal = 0u; ordinal < decision_count; ++ordinal) {
        rc = quant_explicit_decision_build(&context, &decisions[ordinal], ordinal, &candidate);
        if (rc != YVEX_OK) {
            yvex_quant_plan_release(&context.plan);
            return rc;
        }
    }
    candidate.numerically_admissible = 1;
    context.plan->summary.candidates[0] = candidate;
    quant_summary_select(&context.plan->summary, &candidate);
    if (context.plan->summary.decision_count != decision_count ||
        context.plan->summary.calibration_required || !candidate.compute_admissible ||
        !quant_plan_identity(context.plan)) {
        yvex_quant_plan_release(&context.plan);
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_INCOMPLETE, ULLONG_MAX, ULLONG_MAX,
                               decision_count, 0u, UINT_MAX, YVEX_TRANSFORM_OP_COUNT, err,
                               YVEX_ERR_FORMAT, "explicit quantization plan did not seal");
    }
    context.plan->summary.state = YVEX_QUANT_PLAN_SEALED;
    context.plan->summary.complete = 1;
    *out = context.plan;
    yvex_core_execution_observation_record(
        YVEX_CORE_OBSERVE_QUANT_PLAN, 1ull);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Preserve one identity-only Transformation IR in its exact scalar source representation.
 *
 * This is intentionally narrower than a quantization policy: every terminal must be a one-source
 * identity operation and its admitted dtype selects the corresponding lossless GGUF scalar type.
 * GGUF has four physical dimensions, so higher-rank exact tensors preserve their first three
 * axes and fold the remaining contiguous axes into the fourth. The logical rank remains in
 * Transformation IR and the physical decision identity binds this reversible container view.
 */
int yvex_quant_plan_build_source_faithful(
    yvex_quant_plan **out, const yvex_transform_ir *ir,
    const yvex_transform_binding *binding, const char *profile_name,
    unsigned long long lowering_identity, const yvex_quant_plan_options *options,
    yvex_quant_failure *failure, yvex_error *err) {
    const yvex_transform_ir_summary *summary = yvex_transform_ir_summary_get(ir);
    yvex_quant_explicit_decision *decisions = NULL;
    unsigned long long ordinal;
    int rc;

    if (out)
        *out = NULL;
    if (!out || !summary || !summary->complete || !summary->terminal_count ||
        summary->terminal_count > SIZE_MAX / sizeof(*decisions))
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, ULLONG_MAX,
                               ULLONG_MAX, 1u, summary ? summary->terminal_count : 0u, UINT_MAX,
                               YVEX_TRANSFORM_OP_COUNT, err, YVEX_ERR_INVALID_ARG,
                               "a bounded sealed Transformation IR is required");
    decisions = (yvex_quant_explicit_decision *)calloc(
        (size_t)summary->terminal_count, sizeof(*decisions));
    if (!decisions)
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_ALLOCATION, ULLONG_MAX, ULLONG_MAX,
                               summary->terminal_count, 0u, UINT_MAX, YVEX_TRANSFORM_OP_COUNT,
                               err, YVEX_ERR_NOMEM,
                               "source-faithful decision allocation failed");
    for (ordinal = 0u; ordinal < summary->terminal_count; ++ordinal) {
        const yvex_transform_value *terminal = yvex_transform_ir_terminal_at(ir, ordinal);
        const yvex_transform_node *node =
            terminal ? yvex_transform_ir_node_at(ir, terminal->producer_node_id) : NULL;
        unsigned int dimension;

        if (!terminal || !node || node->kind != YVEX_TRANSFORM_OP_IDENTITY ||
            node->input_count != 1u || node->numeric != YVEX_TRANSFORM_NUMERIC_EXACT ||
            quant_exact_qtype(terminal->dtype) == UINT_MAX) {
            free(decisions);
            return quant_plan_fail(
                failure, YVEX_QUANT_FAILURE_UNSUPPORTED_OPERATION, ordinal, ULLONG_MAX,
                YVEX_TRANSFORM_OP_IDENTITY, node ? node->kind : YVEX_TRANSFORM_OP_COUNT,
                UINT_MAX, node ? node->kind : YVEX_TRANSFORM_OP_COUNT, err,
                YVEX_ERR_UNSUPPORTED,
                "source-faithful planning admits only exact one-source identity terminals");
        }
        decisions[ordinal].qtype = quant_exact_qtype(terminal->dtype);
        decisions[ordinal].rank = terminal->shape.rank <= YVEX_GGUF_QTYPE_MAX_DIMS
                                      ? terminal->shape.rank
                                      : YVEX_GGUF_QTYPE_MAX_DIMS;
        for (dimension = 0u; dimension < decisions[ordinal].rank; ++dimension) {
            unsigned int logical_axis = terminal->shape.rank - dimension - 1u;
            decisions[ordinal].dims[dimension] = terminal->shape.dims[logical_axis];
        }
        if (terminal->shape.rank > YVEX_GGUF_QTYPE_MAX_DIMS) {
            unsigned long long folded = 1ull;
            unsigned int outer_axes =
                terminal->shape.rank - YVEX_GGUF_QTYPE_MAX_DIMS + 1u;

            for (dimension = 0u; dimension < outer_axes; ++dimension) {
                if (!yvex_core_u64_mul(folded, terminal->shape.dims[dimension], &folded)) {
                    free(decisions);
                    return quant_plan_fail(
                        failure, YVEX_QUANT_FAILURE_INVALID_DIMENSION, ordinal,
                        ULLONG_MAX, ULLONG_MAX, terminal->shape.dims[dimension],
                        decisions[ordinal].qtype, node->kind, err, YVEX_ERR_BOUNDS,
                        "source-faithful physical shape folding overflowed");
                }
            }
            decisions[ordinal].dims[YVEX_GGUF_QTYPE_MAX_DIMS - 1u] = folded;
        }
    }
    rc = yvex_quant_plan_build_explicit(
        out, ir, binding, profile_name, lowering_identity, decisions,
        summary->terminal_count, options, failure, err);
    free(decisions);
    return rc;
}

/*
 * Account both DeepSeek candidates and index one selected terminal.
 *
 * Candidate comparison remains deterministic and payload-free.
 */
static int quant_deepseek_decision_build(quant_build_context *context,
                                         yvex_quant_profile_kind profile,
                                         unsigned long long ordinal) {
    const yvex_transform_value *terminal = yvex_transform_ir_terminal_at(context->ir, ordinal);
    const yvex_transform_node *node =
        terminal ? yvex_transform_ir_node_at(context->ir, terminal->producer_node_id) : NULL;
    const yvex_deepseek_gguf_descriptor *descriptor =
        yvex_model_register_deepseek_v4()->lowering.at(context->map, ordinal);
    yvex_quant_decision reference_decision;
    yvex_quant_decision release_decision;
    yvex_quant_decision *decision = &context->plan->decisions[ordinal];
    int rc;

    if (!terminal || !node || terminal->canonical_ordinal != ordinal ||
        yvex_transform_binding_terminal_at(context->binding, ordinal) != terminal ||
        yvex_transform_binding_terminal_operation(context->binding, ordinal) != node)
        return quant_plan_fail(
            context->failure, YVEX_QUANT_FAILURE_MISSING_DECISION, ordinal, ULLONG_MAX, ordinal,
            terminal ? terminal->canonical_ordinal : ULLONG_MAX, UINT_MAX,
            node ? node->kind : YVEX_TRANSFORM_OP_COUNT, context->err, YVEX_ERR_FORMAT,
            "binding does not expose the canonical terminal operation");
    rc = quant_descriptor_matches(context->ir, terminal, node, context->map, descriptor, ordinal,
                                  context->failure, context->err);
    if (rc != YVEX_OK)
        return rc;
    rc = quant_build_candidate_decision(YVEX_QUANT_PROFILE_SOURCE_FAITHFUL, context->binding,
                                        terminal, node, descriptor, ordinal, &reference_decision,
                                        context->failure, context->err);
    if (rc != YVEX_OK)
        return rc;
    if (!quant_summary_add(&context->plan->summary.candidates[0], &reference_decision))
        return quant_plan_fail(
            context->failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, ordinal, ULLONG_MAX, ULLONG_MAX,
            context->plan->summary.candidates[0].encoded_bytes, UINT_MAX, node->kind, context->err,
            YVEX_ERR_BOUNDS, "reference candidate byte accounting overflowed");
    rc = quant_build_candidate_decision(YVEX_QUANT_PROFILE_RELEASE_Q8_Q2, context->binding,
                                        terminal, node, descriptor, ordinal, &release_decision,
                                        context->failure, context->err);
    if (rc != YVEX_OK)
        return rc;
    if (!quant_summary_add(&context->plan->summary.candidates[1], &release_decision))
        return quant_plan_fail(
            context->failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, ordinal, ULLONG_MAX, ULLONG_MAX,
            context->plan->summary.candidates[1].encoded_bytes, UINT_MAX, node->kind, context->err,
            YVEX_ERR_BOUNDS, "release candidate byte accounting overflowed");
    *decision =
        profile == YVEX_QUANT_PROFILE_SOURCE_FAITHFUL ? reference_decision : release_decision;
    return quant_index_add(context, decision, ordinal,
                           "duplicate logical terminal decision refused",
                           "quantization decision index is exhausted");
}

static int quant_deepseek_plan_seal(quant_build_context *context, yvex_quant_profile_kind profile) {
    yvex_quant_candidate_summary *selected = &context->plan->summary.candidates[profile];

    context->plan->summary.candidates[0].numerically_admissible = 1;
    context->plan->summary.candidates[1].numerically_admissible = 1;
    quant_summary_select(&context->plan->summary, selected);
    if (context->plan->summary.decision_count != context->ir_summary->terminal_count ||
        context->plan->summary.calibration_required || !selected->compute_admissible ||
        !quant_plan_identity(context->plan))
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_INCOMPLETE, ULLONG_MAX,
                               ULLONG_MAX, context->ir_summary->terminal_count, 0u, UINT_MAX,
                               YVEX_TRANSFORM_OP_COUNT, context->err, YVEX_ERR_FORMAT,
                               "quantization profile did not seal completely");
    context->plan->summary.state = YVEX_QUANT_PLAN_SEALED;
    context->plan->summary.complete = 1;
    context->plan->summary.payload_bytes_read = 0u;
    return YVEX_OK;
}

static int quant_policy_pattern_matches(const char *pattern, const char *name) {
    const char *star;
    size_t prefix;
    size_t suffix;
    size_t length;

    if (!pattern || !name)
        return 0;
    star = strchr(pattern, '*');
    if (!star)
        return strcmp(pattern, name) == 0;
    if (strchr(star + 1, '*'))
        return 0;
    prefix = (size_t)(star - pattern);
    suffix = strlen(star + 1);
    length = strlen(name);
    return length >= prefix + suffix && strncmp(pattern, name, prefix) == 0 &&
           strcmp(name + length - suffix, star + 1) == 0;
}

static unsigned int quant_policy_qtype(yvex_quant_qtype qtype) {
    switch (qtype) {
    case YVEX_QUANT_QTYPE_F32: return YVEX_GGUF_QTYPE_F32;
    case YVEX_QUANT_QTYPE_F16: return YVEX_GGUF_QTYPE_F16;
    case YVEX_QUANT_QTYPE_BF16: return YVEX_GGUF_QTYPE_BF16;
    case YVEX_QUANT_QTYPE_Q8_0: return YVEX_GGUF_QTYPE_Q8_0;
    case YVEX_QUANT_QTYPE_Q4_0: return YVEX_GGUF_QTYPE_Q4_0;
    case YVEX_QUANT_QTYPE_Q4_K: return YVEX_GGUF_QTYPE_Q4_K;
    case YVEX_QUANT_QTYPE_Q5_K: return YVEX_GGUF_QTYPE_Q5_K;
    case YVEX_QUANT_QTYPE_Q6_K: return YVEX_GGUF_QTYPE_Q6_K;
    case YVEX_QUANT_QTYPE_Q2_K: return YVEX_GGUF_QTYPE_Q2_K;
    case YVEX_QUANT_QTYPE_IQ2_XXS: return YVEX_GGUF_QTYPE_IQ2_XXS;
    case YVEX_QUANT_QTYPE_IQ2_XS: return YVEX_GGUF_QTYPE_IQ2_XS;
    case YVEX_QUANT_QTYPE_IQ3_XXS: return YVEX_GGUF_QTYPE_IQ3_XXS;
    case YVEX_QUANT_QTYPE_IQ4_NL: return YVEX_GGUF_QTYPE_IQ4_NL;
    case YVEX_QUANT_QTYPE_I32: return YVEX_GGUF_QTYPE_I32;
    default: return UINT_MAX;
    }
}

static int quant_policy_rule_matches(const yvex_quant_policy_rule *rule,
                                     const yvex_transform_value *terminal,
                                     const yvex_transform_node *node,
                                     const yvex_deepseek_gguf_descriptor *descriptor) {
    unsigned long long mask = rule->match_mask;
    yvex_quant_policy_physical_class physical =
        terminal->precision.flags & YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT
            ? YVEX_QUANT_POLICY_PHYSICAL_QUANTIZABLE
            : YVEX_QUANT_POLICY_PHYSICAL_EXACT;

    if ((mask & YVEX_QUANT_MATCH_ROLE) && rule->role != terminal->logical_key.role)
        return 0;
    if ((mask & YVEX_QUANT_MATCH_COLLECTION) && rule->collection != descriptor->collection)
        return 0;
    if ((mask & YVEX_QUANT_MATCH_SCOPE) && rule->scope != descriptor->scope)
        return 0;
    if ((mask & YVEX_QUANT_MATCH_TENSOR_NAME) &&
        strcmp(rule->tensor_name, descriptor->emitted_name) != 0)
        return 0;
    if ((mask & YVEX_QUANT_MATCH_TENSOR_PATTERN) &&
        !quant_policy_pattern_matches(rule->tensor_pattern, descriptor->emitted_name))
        return 0;
    if ((mask & YVEX_QUANT_MATCH_LAYER_RANGE) &&
        (descriptor->layer_index == YVEX_DEEPSEEK_GGUF_NO_INDEX ||
         descriptor->layer_index < rule->layer_first || descriptor->layer_index > rule->layer_last))
        return 0;
    if ((mask & YVEX_QUANT_MATCH_EXPERT_GROUP) &&
        terminal->logical_key.group_index != rule->expert_group)
        return 0;
    if ((mask & YVEX_QUANT_MATCH_OPERATION) &&
        (unsigned int)rule->operation != (unsigned int)node->kind + 1u)
        return 0;
    if ((mask & YVEX_QUANT_MATCH_PHYSICAL_CLASS) && rule->physical_class != physical)
        return 0;
    return 1;
}

static int quant_policy_actions_equal(const yvex_quant_policy_rule *left,
                                      const yvex_quant_policy_rule *right) {
    return left->qtype == right->qtype &&
           left->requires_imatrix == right->requires_imatrix &&
           left->requires_cpu_compute == right->requires_cpu_compute &&
           left->requires_cuda_compute == right->requires_cuda_compute;
}

/*
 * Resolve one highest-priority non-conflicting policy action.
 *
 * Resolution is deterministic and independent of payload bytes and rule allocation.
 */
static int quant_policy_resolve(const yvex_quant_policy *policy,
                                const yvex_transform_value *terminal,
                                const yvex_transform_node *node,
                                const yvex_deepseek_gguf_descriptor *descriptor,
                                const yvex_quant_policy_rule **selected,
                                unsigned long long *selected_ordinal,
                                yvex_quant_failure *failure, yvex_error *err) {
    unsigned long long index;

    *selected = NULL;
    *selected_ordinal = ULLONG_MAX;
    for (index = 0u; index < yvex_quant_policy_rule_count(policy); ++index) {
        const yvex_quant_policy_rule *rule = yvex_quant_policy_rule_at(policy, index);
        if (!rule || !quant_policy_rule_matches(rule, terminal, node, descriptor))
            continue;
        if (!*selected || rule->priority > (*selected)->priority) {
            *selected = rule;
            *selected_ordinal = index;
        } else if (rule->priority == (*selected)->priority &&
                   !quant_policy_actions_equal(rule, *selected)) {
            return quant_plan_fail(failure, YVEX_QUANT_FAILURE_PRECISION_CONSTRAINT,
                                   terminal->canonical_ordinal, ULLONG_MAX, (*selected)->priority,
                                   rule->priority, UINT_MAX, node->kind, err, YVEX_ERR_FORMAT,
                                   "equal-priority policy actions conflict");
        }
    }
    if (!*selected)
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_MISSING_DECISION,
                               terminal->canonical_ordinal, ULLONG_MAX, 1u, 0u, UINT_MAX,
                               node->kind, err, YVEX_ERR_FORMAT,
                               "quantizable terminal has no matching policy rule or default");
    return YVEX_OK;
}

static int quant_policy_bind_decision(yvex_quant_decision *decision,
                                      const yvex_quant_policy_summary *policy_summary,
                                      const yvex_quant_policy_rule *rule,
                                      unsigned long long rule_ordinal) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    decision->policy_bound = 1;
    decision->policy_rule_ordinal = rule_ordinal;
    decision->policy_priority = rule ? rule->priority : UINT_MAX;
    decision->policy_requires_imatrix = rule ? rule->requires_imatrix : 0;
    yvex_core_text_copy(decision->policy_label, sizeof(decision->policy_label),
                        rule && rule->label ? rule->label : "family exact physical constraint");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.quant.policy.rule.decision.v1") ||
        !yvex_sha256_update_text(&hash, policy_summary->policy_identity) ||
        !yvex_sha256_update_u64(&hash, rule_ordinal) ||
        !yvex_sha256_update_u64(&hash, decision->qtype) ||
        !yvex_sha256_update_u64(&hash, decision->terminal_ordinal) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, decision->policy_rule_identity);
    return quant_decision_identity(decision);
}

/*
 * Build baseline comparisons plus one policy-selected terminal decision.
 *
 * Incomplete lowering, policy conflict, unsupported qtype, or identity failure refuses.
 */
static int quant_deepseek_policy_decision_build(
    quant_build_context *context, const yvex_quant_policy *policy,
    const yvex_quant_policy_summary *policy_summary, unsigned long long ordinal,
    yvex_quant_candidate_summary *selected_summary) {
    const yvex_transform_value *terminal = yvex_transform_ir_terminal_at(context->ir, ordinal);
    const yvex_transform_node *node =
        terminal ? yvex_transform_ir_node_at(context->ir, terminal->producer_node_id) : NULL;
    const yvex_deepseek_gguf_descriptor *descriptor =
        yvex_model_register_deepseek_v4()->lowering.at(context->map, ordinal);
    const yvex_quant_policy_rule *rule = NULL;
    unsigned long long rule_ordinal = ULLONG_MAX;
    yvex_quant_decision baseline;
    yvex_quant_decision *decision = &context->plan->decisions[ordinal];
    unsigned int qtype;
    int exact;
    int rc;

    if (!terminal || !node || !descriptor || terminal->canonical_ordinal != ordinal)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_MISSING_DECISION, ordinal,
                               ULLONG_MAX, ordinal,
                               terminal ? terminal->canonical_ordinal : ULLONG_MAX, UINT_MAX,
                               node ? node->kind : YVEX_TRANSFORM_OP_COUNT, context->err,
                               YVEX_ERR_FORMAT, "policy plan terminal is incomplete");
    rc = quant_descriptor_matches(context->ir, terminal, node, context->map, descriptor, ordinal,
                                  context->failure, context->err);
    if (rc != YVEX_OK)
        return rc;
    rc = quant_build_candidate_decision(YVEX_QUANT_PROFILE_SOURCE_FAITHFUL, context->binding,
                                        terminal, node, descriptor, ordinal, &baseline,
                                        context->failure, context->err);
    if (rc == YVEX_OK &&
        !quant_summary_add(&context->plan->summary.candidates[0], &baseline))
        rc = YVEX_ERR_BOUNDS;
    if (rc != YVEX_OK)
        return rc;
    rc = quant_build_candidate_decision(YVEX_QUANT_PROFILE_RELEASE_Q8_Q2, context->binding,
                                        terminal, node, descriptor, ordinal, &baseline,
                                        context->failure, context->err);
    if (rc == YVEX_OK &&
        !quant_summary_add(&context->plan->summary.candidates[1], &baseline))
        rc = YVEX_ERR_BOUNDS;
    if (rc != YVEX_OK)
        return rc;
    exact = !(terminal->precision.flags & YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT);
    rc = quant_policy_resolve(policy, terminal, node, descriptor, &rule, &rule_ordinal,
                              context->failure, context->err);
    if (exact && rc != YVEX_OK) {
        yvex_error_clear(context->err);
        if (context->failure)
            memset(context->failure, 0, sizeof(*context->failure));
        rc = YVEX_OK;
    } else if (rc != YVEX_OK) {
        return rc;
    }
    if (exact) {
        if (rule && rule->qtype != YVEX_QUANT_QTYPE_SOURCE &&
            quant_policy_qtype(rule->qtype) != quant_exact_qtype(terminal->dtype))
            return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_APPROXIMATION_FORBIDDEN,
                                   ordinal, ULLONG_MAX, quant_exact_qtype(terminal->dtype),
                                   quant_policy_qtype(rule->qtype), UINT_MAX, node->kind,
                                   context->err, YVEX_ERR_FORMAT,
                                   "policy attempted to override an exact-only terminal");
        qtype = quant_exact_qtype(terminal->dtype);
        rule = NULL;
        rule_ordinal = ULLONG_MAX;
    } else if (rule->qtype == YVEX_QUANT_QTYPE_SOURCE) {
        qtype = quant_candidate_qtype(YVEX_QUANT_PROFILE_SOURCE_FAITHFUL, terminal, node);
    } else {
        qtype = quant_policy_qtype(rule->qtype);
    }
    rc = quant_build_qtype_decision(qtype, context->binding, terminal, node, descriptor, ordinal,
                                    decision, context->failure, context->err);
    if (rc != YVEX_OK)
        return rc;
    if (rule && rule->requires_imatrix)
        decision->calibration = YVEX_QUANT_CALIBRATION_REQUIRED;
    if (rule && rule->requires_cpu_compute && !decision->cpu_compute_available)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_CPU_COMPUTE_UNAVAILABLE,
                               ordinal, ULLONG_MAX, 1u, 0u, qtype, node->kind, context->err,
                               YVEX_ERR_UNSUPPORTED,
                               "policy requires CPU compute that is absent");
    if (rule && rule->requires_cuda_compute && !decision->cuda_compute_available)
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_CUDA_COMPUTE_UNAVAILABLE,
                               ordinal, ULLONG_MAX, 1u, 0u, qtype, node->kind, context->err,
                               YVEX_ERR_UNSUPPORTED,
                               "policy requires CUDA compute that is absent");
    if (decision->calibration == YVEX_QUANT_CALIBRATION_REQUIRED &&
        (!rule || !rule->requires_imatrix))
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_CALIBRATION_REQUIRED,
                               ordinal, ULLONG_MAX, 1u, 0u, qtype, node->kind, context->err,
                               YVEX_ERR_FORMAT,
                               "selected qtype requires explicit imatrix policy admission");
    yvex_core_text_copy(decision->physical_tensor_name,
                        sizeof(decision->physical_tensor_name), descriptor->emitted_name);
    decision->physical_expert_count = descriptor->expert_count;
    if (!quant_policy_bind_decision(decision, policy_summary, rule, rule_ordinal) ||
        !quant_summary_add(selected_summary, decision))
        return quant_plan_fail(context->failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, ordinal,
                               ULLONG_MAX, 1u, 0u, qtype, node->kind, context->err,
                               YVEX_ERR_BOUNDS, "policy decision identity or accounting failed");
    return quant_index_add(context, decision, ordinal, "duplicate policy decision refused",
                           "policy decision index exhausted");
}

/*
 * Resolve one sealed policy over every canonical DeepSeek terminal without payload reads.
 *
 * Complete IR/binding/lowering, sealed policy, optional imatrix identity, and budget. Returns one
 * immutable identity-bound physical-variant plan.
 */
int yvex_quant_plan_build_deepseek_policy(
    yvex_quant_plan **out, const yvex_transform_ir *ir,
    const yvex_transform_binding *binding, const yvex_deepseek_gguf_map *map,
    const yvex_quant_policy *policy, const char *imatrix_identity,
    const yvex_quant_plan_options *options, yvex_quant_failure *failure, yvex_error *err) {
    quant_build_context context;
    yvex_quant_policy_summary policy_summary;
    yvex_quant_candidate_summary selected;
    const yvex_deepseek_gguf_map_summary *map_summary =
        yvex_model_register_deepseek_v4()->lowering.summary(map);
    unsigned long long ordinal;
    int rc;

    memset(&context, 0, sizeof(context));
    memset(&selected, 0, sizeof(selected));
    if (out)
        *out = NULL;
    if (!out || !ir || !binding || !map || !policy || !map_summary ||
        yvex_quant_policy_get_summary(policy, &policy_summary, err) != YVEX_OK ||
        policy_summary.schema_version != YVEX_QUANT_POLICY_SCHEMA_VERSION ||
        policy_summary.status != YVEX_QUANT_POLICY_STATUS_VALID ||
        !yvex_sha256_hex_valid(policy_summary.policy_identity) ||
        strlen(policy_summary.name) >= sizeof(((yvex_quant_plan_summary *)0)->profile_name))
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, ULLONG_MAX,
                               ULLONG_MAX, 1u, 0u, UINT_MAX, YVEX_TRANSFORM_OP_COUNT, err,
                               YVEX_ERR_INVALID_ARG,
                               "complete DeepSeek inputs and one valid policy-v2 are required");
    rc = yvex_quant_policy_identity_validate(policy, err);
    if (rc != YVEX_OK)
        return rc;
    context.ir = ir;
    context.binding = binding;
    context.map = map;
    context.ir_summary = yvex_transform_ir_summary_get(ir);
    context.binding_summary = yvex_transform_binding_summary_get(binding);
    context.profile_name = policy_summary.name;
    context.mapping_identity = map_summary->mapping_identity;
    context.failure = failure;
    context.err = err;
    if (!context.ir_summary || !context.binding_summary || !context.ir_summary->complete ||
        !context.binding_summary->complete || !map_summary->complete ||
        yvex_transform_binding_ir(binding) != ir ||
        context.ir_summary->terminal_count != YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT)
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, ULLONG_MAX,
                               ULLONG_MAX, YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT,
                               context.ir_summary ? context.ir_summary->terminal_count : 0u,
                               UINT_MAX, YVEX_TRANSFORM_OP_COUNT, err, YVEX_ERR_FORMAT,
                               "policy resolution requires the complete canonical DeepSeek graph");
    rc = quant_binding_identity_validate(&context);
    if (rc == YVEX_OK)
        rc = quant_build_allocate(&context, options);
    if (rc != YVEX_OK) {
        yvex_quant_plan_release(&context.plan);
        return rc;
    }
    context.plan->summary.schema_version = 2u;
    yvex_core_text_copy(context.plan->summary.policy_identity,
                        sizeof(context.plan->summary.policy_identity),
                        policy_summary.policy_identity);
    yvex_core_text_copy(context.plan->summary.imatrix_identity,
                        sizeof(context.plan->summary.imatrix_identity),
                        imatrix_identity && imatrix_identity[0] ? imatrix_identity : "none");
    context.plan->summary.candidates[0].kind = YVEX_QUANT_PROFILE_SOURCE_FAITHFUL;
    context.plan->summary.candidates[0].name = YVEX_QUANT_REFERENCE_PROFILE_NAME;
    context.plan->summary.candidates[0].compute_admissible = 1;
    context.plan->summary.candidates[1].kind = YVEX_QUANT_PROFILE_RELEASE_Q8_Q2;
    context.plan->summary.candidates[1].name = YVEX_QUANT_RELEASE_PROFILE_NAME;
    context.plan->summary.candidates[1].compute_admissible = 1;
    selected.name = context.plan->summary.profile_name;
    selected.compute_admissible = 1;
    for (ordinal = 0u; ordinal < context.ir_summary->terminal_count; ++ordinal) {
        rc = quant_deepseek_policy_decision_build(&context, policy, &policy_summary, ordinal,
                                                  &selected);
        if (rc != YVEX_OK) {
            yvex_quant_plan_release(&context.plan);
            return rc;
        }
    }
    if (selected.calibration_required &&
        (!imatrix_identity || !yvex_sha256_hex_valid(imatrix_identity))) {
        yvex_quant_plan_release(&context.plan);
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_CALIBRATION_IDENTITY, ULLONG_MAX,
                               ULLONG_MAX, 1u, 0u, UINT_MAX, YVEX_TRANSFORM_OP_COUNT, err,
                               YVEX_ERR_FORMAT,
                               "calibrated physical variant requires one canonical imatrix identity");
    }
    if (selected.calibration_required)
        yvex_core_text_copy(context.plan->summary.calibration_identity,
                            sizeof(context.plan->summary.calibration_identity), imatrix_identity);
    quant_summary_select(&context.plan->summary, &selected);
    if (context.plan->summary.decision_count != context.ir_summary->terminal_count ||
        !selected.compute_admissible || !quant_plan_identity(context.plan)) {
        yvex_quant_plan_release(&context.plan);
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_INCOMPLETE, ULLONG_MAX, ULLONG_MAX,
                               context.ir_summary->terminal_count,
                               context.plan ? context.plan->summary.decision_count : 0u, UINT_MAX,
                               YVEX_TRANSFORM_OP_COUNT, err, YVEX_ERR_FORMAT,
                               "policy-driven physical variant did not seal completely");
    }
    context.plan->summary.state = YVEX_QUANT_PLAN_SEALED;
    context.plan->summary.complete = 1;
    context.plan->summary.payload_bytes_read = 0u;
    *out = context.plan;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_quant_plan_build_deepseek_profile(yvex_quant_plan **out, const yvex_transform_ir *ir,
                                           const yvex_transform_binding *binding,
                                           const yvex_deepseek_gguf_map *map,
                                           yvex_quant_profile_kind profile,
                                           const yvex_quant_plan_options *options,
                                           yvex_quant_failure *failure, yvex_error *err) {
    quant_build_context context;
    const yvex_deepseek_gguf_map_summary *map_summary =
        yvex_model_register_deepseek_v4()->lowering.summary(map);
    unsigned long long ordinal;
    int rc;

    memset(&context, 0, sizeof(context));
    context.ir = ir;
    context.binding = binding;
    context.map = map;
    context.ir_summary = yvex_transform_ir_summary_get(ir);
    context.binding_summary = yvex_transform_binding_summary_get(binding);
    context.profile_name = profile == YVEX_QUANT_PROFILE_SOURCE_FAITHFUL
                               ? YVEX_QUANT_REFERENCE_PROFILE_NAME
                               : YVEX_QUANT_RELEASE_PROFILE_NAME;
    context.mapping_identity = map_summary ? map_summary->mapping_identity : 0u;
    context.failure = failure;
    context.err = err;
    if (out)
        *out = NULL;
    if (!out || !ir || !binding || !map ||
        (profile != YVEX_QUANT_PROFILE_SOURCE_FAITHFUL &&
         profile != YVEX_QUANT_PROFILE_RELEASE_Q8_Q2) ||
        yvex_transform_binding_ir(binding) != ir || !context.ir_summary ||
        !context.binding_summary || !map_summary || !context.ir_summary->complete ||
        !context.binding_summary->complete || !map_summary->complete)
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, ULLONG_MAX, ULLONG_MAX,
                               1u, 0u, UINT_MAX, YVEX_TRANSFORM_OP_COUNT, err, YVEX_ERR_INVALID_ARG,
                               "sealed IR, complete binding, lowering, and output are required");
    rc = quant_binding_identity_validate(&context);
    if (rc != YVEX_OK)
        return rc;
    if (map_summary->mapping_identity != YVEX_DEEPSEEK_GGUF_MAPPING_IDENTITY)
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_MAPPING_IDENTITY, ULLONG_MAX, ULLONG_MAX,
                               YVEX_DEEPSEEK_GGUF_MAPPING_IDENTITY, map_summary->mapping_identity,
                               UINT_MAX, YVEX_TRANSFORM_OP_COUNT, err, YVEX_ERR_FORMAT,
                               "GGUF mapping identity is not the pinned lowering");
    if (context.ir_summary->terminal_count != YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT ||
        context.ir_summary->source_value_count != YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        map_summary->descriptor_count != context.ir_summary->terminal_count)
        return quant_plan_fail(failure, YVEX_QUANT_FAILURE_MISSING_DECISION, ULLONG_MAX, ULLONG_MAX,
                               YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT,
                               context.ir_summary->terminal_count, UINT_MAX,
                               YVEX_TRANSFORM_OP_COUNT, err, YVEX_ERR_FORMAT,
                               "target-scale terminal or source accounting is incomplete");
    rc = quant_build_allocate(&context, options);
    if (rc != YVEX_OK) {
        yvex_quant_plan_release(&context.plan);
        return rc;
    }
    context.plan->summary.candidates[0].kind = YVEX_QUANT_PROFILE_SOURCE_FAITHFUL;
    context.plan->summary.candidates[0].name = YVEX_QUANT_REFERENCE_PROFILE_NAME;
    context.plan->summary.candidates[0].compute_admissible = 1;
    context.plan->summary.candidates[1].kind = YVEX_QUANT_PROFILE_RELEASE_Q8_Q2;
    context.plan->summary.candidates[1].name = YVEX_QUANT_RELEASE_PROFILE_NAME;
    context.plan->summary.candidates[1].compute_admissible = 1;
    for (ordinal = 0u; ordinal < context.ir_summary->terminal_count; ++ordinal) {
        rc = quant_deepseek_decision_build(&context, profile, ordinal);
        if (rc != YVEX_OK) {
            yvex_quant_plan_release(&context.plan);
            return rc;
        }
    }
    rc = quant_deepseek_plan_seal(&context, profile);
    if (rc != YVEX_OK) {
        yvex_quant_plan_release(&context.plan);
        return rc;
    }
    *out = context.plan;
    yvex_core_execution_observation_record(
        YVEX_CORE_OBSERVE_QUANT_PLAN, 1ull);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_quant_plan_release(yvex_quant_plan **plan_address) {
    yvex_quant_plan *plan;
    yvex_quant_release_fn release;
    void *context;

    if (!plan_address || !*plan_address)
        return;
    plan = *plan_address;
    *plan_address = NULL;
    release = plan->release ? plan->release : quant_plan_default_release;
    context = plan->allocator_context;
    if (plan->index)
        release(plan->index, context);
    if (plan->decisions)
        release(plan->decisions, context);
    plan->summary.state = YVEX_QUANT_PLAN_RELEASED;
    release(plan, context);
}

const yvex_quant_plan_summary *yvex_quant_plan_summary_get(const yvex_quant_plan *plan) {
    return plan ? &plan->summary : NULL;
}

const yvex_quant_decision *yvex_quant_plan_decision_at(const yvex_quant_plan *plan,
                                                       unsigned long long ordinal) {
    return plan && plan->summary.state == YVEX_QUANT_PLAN_SEALED &&
                   ordinal < plan->summary.decision_count
               ? &plan->decisions[ordinal]
               : NULL;
}

const yvex_quant_decision *yvex_quant_plan_find(const yvex_quant_plan *plan,
                                                const yvex_transform_logical_key *key) {
    unsigned long long hash;
    unsigned long long slot;
    unsigned long long probe;

    if (!plan || plan->summary.state != YVEX_QUANT_PLAN_SEALED || !key)
        return NULL;
    hash = quant_key_hash(key);
    slot = hash & (plan->summary.index_capacity - 1u);
    for (probe = 0u; probe < plan->summary.index_capacity && plan->index[slot].ordinal_plus_one;
         ++probe) {
        if (plan->index[slot].hash == hash) {
            const yvex_quant_decision *decision =
                &plan->decisions[plan->index[slot].ordinal_plus_one - 1u];
            if (quant_key_equal(&decision->logical_key, key))
                return decision;
        }
        slot = (slot + 1u) & (plan->summary.index_capacity - 1u);
    }
    return NULL;
}

/*
 * Expose the transform IR borrowed by a quant plan.
 *
 * Lifetime and ownership remain with the upstream IR owner.
 */
const yvex_transform_ir *yvex_quant_plan_transform_ir(const yvex_quant_plan *plan) {
    return plan ? plan->ir : NULL;
}

const yvex_transform_binding *yvex_quant_plan_binding(const yvex_quant_plan *plan) {
    return plan ? plan->binding : NULL;
}
