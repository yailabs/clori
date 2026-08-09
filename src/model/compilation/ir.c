/*
 * Construct, seal, index, and expose deterministic transformation plans.
 *
 * Builders publish no partial IR; sealed arrays are read-only to consumers; all allocator
 * transitions are checked and released by their owning object. Family composition registers
 * artifact-neutral metadata only and never reads tensor payload bytes or selects a physical
 * encoding.
 */
#include <yvex/internal/compilation.h>

#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>
#include "src/model/compilation/private.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRANSFORM_DEFAULT_MAX_SOURCES 100000ull
#define TRANSFORM_DEFAULT_MAX_VALUES 120000ull
#define TRANSFORM_DEFAULT_MAX_NODES 10000ull
#define TRANSFORM_DEFAULT_MAX_EDGES 200000ull
#define TRANSFORM_DEFAULT_MAX_TERMINALS 10000ull
#define TRANSFORM_DEFAULT_MAX_BYTES (512u * 1024u * 1024u)

static const char *const transform_failure_names[] = {
    "none", "invalid-argument", "invalid-lifecycle-state",
    "architecture-ir-not-admitted", "source-coverage-incomplete",
    "source-snapshot-identity-mismatch", "required-payload-identity-mismatch",
    "unsupported-ir-schema", "invalid-logical-tensor-key",
    "duplicate-logical-tensor-key", "duplicate-source-input",
    "missing-source-input", "unexpected-source-input", "duplicate-value-id",
    "duplicate-node-id", "missing-producer", "multiple-producers",
    "unresolved-edge", "cycle-detected", "unsupported-operation",
    "invalid-operation-arity", "invalid-dtype-combination",
    "unsupported-source-dtype", "invalid-rank", "invalid-shape",
    "dimensional-overflow", "invalid-axis", "invalid-permutation",
    "element-count-mismatch", "invalid-aggregation-cardinality",
    "duplicate-expert-index", "missing-expert-index",
    "unconsumed-required-source", "duplicate-terminal-output",
    "missing-terminal-output", "resource-budget-exceeded",
    "allocation-failure", "identity-encoding-failure", "seal-failure",
    "gguf-lowering-divergence", "mapping-identity-mismatch", "cleanup-failure"
};

static const yvex_transform_subsystem deepseek_subsystems[] = {
    YVEX_TRANSFORM_SUBSYSTEM_GLOBAL,
    YVEX_TRANSFORM_SUBSYSTEM_ATTENTION,
    YVEX_TRANSFORM_SUBSYSTEM_COMPRESSOR,
    YVEX_TRANSFORM_SUBSYSTEM_INDEXER,
    YVEX_TRANSFORM_SUBSYSTEM_NORMALIZATION,
    YVEX_TRANSFORM_SUBSYSTEM_RESIDUAL,
    YVEX_TRANSFORM_SUBSYSTEM_ROUTER,
    YVEX_TRANSFORM_SUBSYSTEM_ROUTED_EXPERT,
    YVEX_TRANSFORM_SUBSYSTEM_SHARED_EXPERT,
    YVEX_TRANSFORM_SUBSYSTEM_AUXILIARY
};

typedef struct {
    yvex_tensor_role role;
    const char *projection;
} deepseek_expert_projection;

static const deepseek_expert_projection deepseek_expert_projections[] = {
    {YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, "w1"},
    {YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN, "w2"},
    {YVEX_TENSOR_ROLE_MOE_EXPERT_UP, "w3"}
};

static void *transform_default_allocate(size_t size, void *context)
{
    (void)context;
    return malloc(size);
}

static void transform_default_release(void *allocation, void *context)
{
    (void)context;
    free(allocation);
}

static int transform_identity_text_valid(const char *text)
{
    size_t index;

    if (!text || strlen(text) != 64u) return 0;
    for (index = 0u; index < 64u; ++index) {
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f'))) return 0;
    }
    return 1;
}

void *yvex_transform_allocate_zero(yvex_transform_allocator *allocator,
                                   size_t size)
{
    void *allocation;

    if (!allocator || !allocator->allocate || size == 0u) return NULL;
    allocation = allocator->allocate(size, allocator->context);
    if (allocation) memset(allocation, 0, size);
    return allocation;
}
/* Encode hash string fields in canonical identity order. */
unsigned long long yvex_transform_hash_string(const char *text)
{
    return text
        ? yvex_core_hash_mix_bytes(1469598103934665603ull, text,
                                    strlen(text) + 1u)
        : 0u;
}
/* Encode hash logical key fields in canonical identity order. */
unsigned long long yvex_transform_hash_logical_key(
    const yvex_transform_logical_key *key)
{
    unsigned long long hash = 1469598103934665603ull;

    if (!key) return 0u;
    hash = yvex_core_hash_mix_u64(hash, (unsigned long long)key->scope);
    hash = yvex_core_hash_mix_u64(hash, (unsigned long long)key->subsystem);
    hash = yvex_core_hash_mix_u64(hash, (unsigned long long)key->role);
    hash = yvex_core_hash_mix_u64(hash, key->component_identity);
    hash = yvex_core_hash_mix_u64(hash, key->semantic_role);
    hash = yvex_core_hash_mix_u64(hash, key->phase_identity);
    hash = yvex_core_hash_mix_u64(hash, key->lifetime_identity);
    hash = yvex_core_hash_mix_u64(hash, key->layer_index);
    hash = yvex_core_hash_mix_u64(hash, key->auxiliary_index);
    return yvex_core_hash_mix_u64(hash, key->group_index);
}
/* Maintain deterministic bounded index capacity lookup state. */
unsigned long long yvex_transform_index_capacity(unsigned long long count)
{
    unsigned long long capacity;

    return yvex_core_power_of_two_capacity(count, 8ull, 1ull, 2ull, &capacity)
               ? capacity
               : 0ull;
}

int yvex_transform_index_insert(yvex_transform_index_slot *slots,
                                unsigned long long capacity,
                                unsigned long long hash,
                                unsigned long long value)
{
    unsigned long long slot;
    unsigned long long probe;

    if (!slots || capacity == 0u || (capacity & (capacity - 1u)) != 0u)
        return 0;
    slot = hash & (capacity - 1u);
    for (probe = 0u; probe < capacity; ++probe) {
        if (slots[slot].value_plus_one == 0u) {
            slots[slot].hash = hash;
            slots[slot].value_plus_one = value + 1u;
            return 1;
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    return 0;
}

static yvex_status transform_status(yvex_transform_failure_code code)
{
    switch (code) {
    case YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT:
        return YVEX_ERR_INVALID_ARG;
    case YVEX_TRANSFORM_FAILURE_INVALID_STATE:
        return YVEX_ERR_STATE;
    case YVEX_TRANSFORM_FAILURE_ALLOCATION:
        return YVEX_ERR_NOMEM;
    case YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET:
    case YVEX_TRANSFORM_FAILURE_DIMENSION_OVERFLOW:
        return YVEX_ERR_BOUNDS;
    case YVEX_TRANSFORM_FAILURE_PAYLOAD_IDENTITY_MISMATCH:
    case YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH:
        return YVEX_ERR_STATE;
    default:
        return YVEX_ERR_FORMAT;
    }
}

int yvex_transform_fail(yvex_transform_failure *failure,
                        yvex_transform_failure_code code,
                        unsigned long long value_id,
                        unsigned long long node_id,
                        unsigned long long source_index,
                        unsigned long long terminal_ordinal,
                        unsigned long long input_index,
                        unsigned long long expected,
                        unsigned long long actual,
                        unsigned int axis,
                        yvex_error *err,
                        const char *where)
{
    yvex_status status = transform_status(code);

    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->value_id = value_id;
        failure->node_id = node_id;
        failure->source_index = source_index;
        failure->terminal_ordinal = terminal_ordinal;
        failure->input_index = input_index;
        failure->expected = expected;
        failure->actual = actual;
        failure->axis = axis;
    }
    yvex_error_setf(err, status, where ? where : "transform_ir",
                    "%s value=%llu node=%llu source=%llu terminal=%llu input=%llu expected=%llu actual=%llu axis=%u",
                    yvex_transform_failure_name(code), value_id, node_id,
                    source_index, terminal_ordinal, input_index,
                    expected, actual, axis);
    return status;
}

void yvex_transform_budget_default(yvex_transform_budget *budget)
{
    if (!budget) return;
    memset(budget, 0, sizeof(*budget));
    budget->maximum_sources = TRANSFORM_DEFAULT_MAX_SOURCES;
    budget->maximum_values = TRANSFORM_DEFAULT_MAX_VALUES;
    budget->maximum_nodes = TRANSFORM_DEFAULT_MAX_NODES;
    budget->maximum_edges = TRANSFORM_DEFAULT_MAX_EDGES;
    budget->maximum_terminals = TRANSFORM_DEFAULT_MAX_TERMINALS;
    budget->maximum_owned_bytes = TRANSFORM_DEFAULT_MAX_BYTES;
}

static int transform_grow(yvex_transform_builder *builder,
                          void **allocation,
                          unsigned long long *capacity,
                          unsigned long long required,
                          unsigned long long maximum,
                          size_t element_size,
                          unsigned long long initialized,
                          yvex_transform_failure *failure,
                          yvex_error *err)
{
    unsigned long long next;
    size_t old_bytes;
    size_t new_bytes;
    void *replacement;

    if (required <= *capacity) return YVEX_OK;
    if (required > maximum || element_size == 0u) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, maximum, required, 0u, err,
            "transform_builder_grow");
    }
    next = *capacity ? *capacity : 8u;
    while (next < required) {
        if (next > maximum / 2u) {
            next = maximum;
            break;
        }
        next *= 2u;
    }
    if (next > (unsigned long long)(SIZE_MAX / element_size)) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, SIZE_MAX, next, 0u, err,
            "transform_builder_grow");
    }
    old_bytes = (size_t)(*capacity) * element_size;
    new_bytes = (size_t)next * element_size;
    if (new_bytes > builder->budget.maximum_owned_bytes ||
        builder->owned_bytes - old_bytes >
            builder->budget.maximum_owned_bytes - new_bytes) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, builder->budget.maximum_owned_bytes,
            builder->owned_bytes - old_bytes + new_bytes, 0u, err,
            "transform_builder_grow");
    }
    replacement = yvex_transform_allocate_zero(&builder->allocator, new_bytes);
    if (!replacement) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_ALLOCATION,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, new_bytes, 0u, 0u, err,
            "transform_builder_grow");
    }
    if (*allocation && initialized)
        memcpy(replacement, *allocation, (size_t)initialized * element_size);
    if (*allocation)
        builder->allocator.release(*allocation, builder->allocator.context);
    *allocation = replacement;
    *capacity = next;
    builder->owned_bytes = builder->owned_bytes - old_bytes + new_bytes;
    if (builder->owned_bytes > builder->peak_bytes)
        builder->peak_bytes = builder->owned_bytes;
    return YVEX_OK;
}

static int transform_shape_valid(const yvex_transform_shape *shape)
{
    unsigned int index;

    if (!shape || shape->rank == 0u ||
        shape->rank > YVEX_TRANSFORM_IR_MAX_RANK) return 0;
    for (index = 0u; index < shape->rank; ++index)
        if (shape->dims[index] == 0u) return 0;
    return 1;
}

static int transform_source_dtype_matches(yvex_native_dtype source,
                                          yvex_transform_dtype value)
{
    switch (source) {
    case YVEX_NATIVE_DTYPE_F32:
        return value == YVEX_TRANSFORM_DTYPE_F32;
    case YVEX_NATIVE_DTYPE_F16:
        return value == YVEX_TRANSFORM_DTYPE_F16;
    case YVEX_NATIVE_DTYPE_BF16:
        return value == YVEX_TRANSFORM_DTYPE_BF16;
    case YVEX_NATIVE_DTYPE_I32:
        return value == YVEX_TRANSFORM_DTYPE_I32;
    case YVEX_NATIVE_DTYPE_I64:
        return value == YVEX_TRANSFORM_DTYPE_I64;
    case YVEX_NATIVE_DTYPE_F8_E4M3:
        return value == YVEX_TRANSFORM_DTYPE_FP8_E4M3;
    case YVEX_NATIVE_DTYPE_F8_E8M0:
        return value == YVEX_TRANSFORM_DTYPE_E8M0_SCALE;
    case YVEX_NATIVE_DTYPE_I8:
    case YVEX_NATIVE_DTYPE_FP4:
        return value == YVEX_TRANSFORM_DTYPE_PACKED_FP4;
    default:
        return 0;
    }
}

static int transform_logical_key_valid(const yvex_transform_logical_key *key,
                                       unsigned int schema_version)
{
    if (!key || key->scope > YVEX_TRANSFORM_SCOPE_AUXILIARY ||
        key->subsystem >= YVEX_TRANSFORM_SUBSYSTEM_COUNT) return 0;
    if (schema_version == YVEX_TRANSFORM_IR_SCHEMA_VERSION) {
        if (key->role <= YVEX_TENSOR_ROLE_UNKNOWN ||
            key->role >= YVEX_TENSOR_ROLE_COUNT || key->component_identity ||
            key->semantic_role || key->phase_identity || key->lifetime_identity) return 0;
    } else if (schema_version == YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION) {
        if (key->role >= YVEX_TENSOR_ROLE_COUNT || !key->component_identity ||
            !key->semantic_role || !key->phase_identity || !key->lifetime_identity) return 0;
    } else return 0;
    if (key->scope == YVEX_TRANSFORM_SCOPE_GLOBAL)
        return key->layer_index == YVEX_TRANSFORM_IR_NO_ID &&
               key->auxiliary_index == YVEX_TRANSFORM_IR_NO_ID;
    if (key->scope == YVEX_TRANSFORM_SCOPE_MAIN_LAYER)
        return key->layer_index != YVEX_TRANSFORM_IR_NO_ID &&
               key->auxiliary_index == YVEX_TRANSFORM_IR_NO_ID;
    return key->layer_index != YVEX_TRANSFORM_IR_NO_ID &&
           key->auxiliary_index != YVEX_TRANSFORM_IR_NO_ID;
}

static int transform_source_scope_valid(
    const yvex_transform_source_spec *source,
    unsigned int schema_version)
{
    if (!source || source->scope > YVEX_TRANSFORM_SCOPE_AUXILIARY ||
        source->subsystem >= YVEX_TRANSFORM_SUBSYSTEM_COUNT) return 0;
    if (schema_version == YVEX_TRANSFORM_IR_SCHEMA_VERSION) {
        if (source->role_hint <= YVEX_TENSOR_ROLE_UNKNOWN ||
            source->role_hint >= YVEX_TENSOR_ROLE_COUNT || source->component_identity ||
            source->semantic_role || source->phase_identity || source->lifetime_identity ||
            source->unresolved_requirement_identity) return 0;
    } else if (schema_version == YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION) {
        if (source->role_hint >= YVEX_TENSOR_ROLE_COUNT || !source->component_identity ||
            !source->semantic_role || !source->phase_identity || !source->lifetime_identity)
            return 0;
    } else return 0;
    if (source->scope == YVEX_TRANSFORM_SCOPE_GLOBAL)
        return source->layer_index == YVEX_TRANSFORM_IR_NO_ID &&
               source->auxiliary_index == YVEX_TRANSFORM_IR_NO_ID;
    if (source->scope == YVEX_TRANSFORM_SCOPE_MAIN_LAYER)
        return source->layer_index != YVEX_TRANSFORM_IR_NO_ID &&
               source->auxiliary_index == YVEX_TRANSFORM_IR_NO_ID;
    return source->layer_index != YVEX_TRANSFORM_IR_NO_ID &&
           source->auxiliary_index != YVEX_TRANSFORM_IR_NO_ID;
}

int yvex_transform_builder_create(
    yvex_transform_builder **out,
    const yvex_transform_header *header,
    const yvex_transform_builder_options *options,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    yvex_transform_allocator allocator;
    yvex_transform_budget budget;
    yvex_transform_builder *builder;

    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (!out || !header ||
        (header->schema_version != YVEX_TRANSFORM_IR_SCHEMA_VERSION &&
         header->schema_version != YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION) ||
        !transform_identity_text_valid(header->logical_model_identity) ||
        !transform_identity_text_valid(header->required_payload_identity) ||
        !header->payload_trust_class || !header->payload_trust_class[0] ||
        header->source_snapshot_identity == 0u ||
        header->expected_source_count == 0u ||
        header->expected_terminal_count == 0u ||
        (header->schema_version == YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION &&
         (!transform_identity_text_valid(header->component_manifest_identity) ||
          !transform_identity_text_valid(header->architecture_identity) ||
          !transform_identity_text_valid(header->role_map_identity) ||
          !transform_identity_text_valid(header->unresolved_requirements_identity)))) {
        return yvex_transform_fail(
            failure,
            header && header->schema_version != YVEX_TRANSFORM_IR_SCHEMA_VERSION &&
                    header->schema_version != YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION
                ? YVEX_TRANSFORM_FAILURE_SCHEMA_UNSUPPORTED
                : YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_SCHEMA_VERSION,
            header ? header->schema_version : 0u, 0u, err,
            "transform_builder_create");
    }
    allocator.allocate = transform_default_allocate;
    allocator.release = transform_default_release;
    allocator.context = NULL;
    yvex_transform_budget_default(&budget);
    if (options) {
        if ((options->allocator.allocate == NULL) !=
                (options->allocator.release == NULL)) {
            return yvex_transform_fail(
                failure, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, 0u, err,
                "transform_builder_create");
        }
        if (options->allocator.allocate) allocator = options->allocator;
        if (options->budget.maximum_sources) budget = options->budget;
    }
    if (budget.maximum_sources < header->expected_source_count ||
        budget.maximum_terminals < header->expected_terminal_count ||
        budget.maximum_values < header->expected_source_count ||
        budget.maximum_owned_bytes < sizeof(*builder)) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, header->expected_source_count,
            budget.maximum_sources, 0u, err, "transform_builder_create");
    }
    builder = (yvex_transform_builder *)yvex_transform_allocate_zero(
        &allocator, sizeof(*builder));
    if (!builder) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_ALLOCATION,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, sizeof(*builder), 0u, 0u, err,
            "transform_builder_create");
    }
    builder->state = YVEX_TRANSFORM_IR_STATE_BUILDING;
    builder->allocator = allocator;
    builder->budget = budget;
    builder->header = *header;
    yvex_core_text_copy(builder->logical_model_identity,
                        sizeof(builder->logical_model_identity),
                        header->logical_model_identity);
    yvex_core_text_copy(builder->required_payload_identity,
                        sizeof(builder->required_payload_identity),
                        header->required_payload_identity);
    yvex_core_text_copy(builder->payload_trust_class,
                        sizeof(builder->payload_trust_class),
                        header->payload_trust_class);
    yvex_core_text_copy(builder->component_manifest_identity,
                        sizeof(builder->component_manifest_identity),
                        header->component_manifest_identity
                            ? header->component_manifest_identity : "");
    yvex_core_text_copy(builder->architecture_identity,
                        sizeof(builder->architecture_identity),
                        header->architecture_identity ? header->architecture_identity : "");
    yvex_core_text_copy(builder->role_map_identity,
                        sizeof(builder->role_map_identity),
                        header->role_map_identity ? header->role_map_identity : "");
    yvex_core_text_copy(builder->unresolved_requirements_identity,
                        sizeof(builder->unresolved_requirements_identity),
                        header->unresolved_requirements_identity
                            ? header->unresolved_requirements_identity : "");
    builder->header.logical_model_identity = builder->logical_model_identity;
    builder->header.required_payload_identity =
        builder->required_payload_identity;
    builder->header.payload_trust_class = builder->payload_trust_class;
    builder->header.component_manifest_identity = builder->component_manifest_identity;
    builder->header.architecture_identity = builder->architecture_identity;
    builder->header.role_map_identity = builder->role_map_identity;
    builder->header.unresolved_requirements_identity =
        builder->unresolved_requirements_identity;
    builder->owned_bytes = sizeof(*builder);
    builder->peak_bytes = sizeof(*builder);
    *out = builder;
    return YVEX_OK;
}

int yvex_transform_builder_add_source(
    yvex_transform_builder *builder,
    const yvex_transform_source_spec *spec,
    unsigned long long *value_id,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    yvex_transform_source_value *source;
    yvex_transform_value *value;
    unsigned long long id;
    int rc;

    if (value_id) *value_id = YVEX_TRANSFORM_IR_NO_ID;
    if (!builder || builder->state != YVEX_TRANSFORM_IR_STATE_BUILDING) {
        return yvex_transform_fail(
            failure, builder ? YVEX_TRANSFORM_FAILURE_INVALID_STATE
                             : YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_STATE_BUILDING,
            builder ? builder->state : YVEX_TRANSFORM_IR_STATE_RELEASED,
            0u, err, "transform_builder_add_source");
    }
    if (!spec || !value_id || !spec->source_name || !spec->source_name[0] ||
        strlen(spec->source_name) >= YVEX_TRANSFORM_IR_SOURCE_NAME_CAP ||
        !spec->shard_name || !spec->shard_name[0] ||
        strlen(spec->shard_name) >= YVEX_TRANSFORM_IR_SHARD_NAME_CAP ||
        spec->source_snapshot_identity !=
            builder->header.source_snapshot_identity ||
        spec->source_dtype <= YVEX_NATIVE_DTYPE_UNKNOWN ||
        spec->source_dtype >= YVEX_NATIVE_DTYPE_OTHER ||
        spec->value_dtype <= YVEX_TRANSFORM_DTYPE_UNKNOWN ||
        spec->value_dtype > YVEX_TRANSFORM_DTYPE_REAL ||
        !transform_source_dtype_matches(spec->source_dtype,
                                        spec->value_dtype) ||
        !transform_source_scope_valid(spec, builder->header.schema_version) ||
        !transform_shape_valid(&spec->shape) ||
        spec->relative_end <= spec->relative_begin ||
        spec->required_uses == 0u) {
        builder->state = YVEX_TRANSFORM_IR_STATE_FAILED;
        return yvex_transform_fail(
            failure,
            spec && spec->source_snapshot_identity !=
                        builder->header.source_snapshot_identity
                ? YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH
                : (spec &&
                   !transform_source_dtype_matches(spec->source_dtype,
                                                   spec->value_dtype)
                       ? YVEX_TRANSFORM_FAILURE_UNSUPPORTED_SOURCE_DTYPE
                       : (spec && !transform_source_scope_valid(
                                      spec, builder->header.schema_version)
                              ? YVEX_TRANSFORM_FAILURE_INVALID_LOGICAL_KEY
                              : YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT)),
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            spec ? spec->source_tensor_index : YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            builder->header.source_snapshot_identity,
            spec ? spec->source_snapshot_identity : 0u, 0u, err,
            "transform_builder_add_source");
    }
    rc = transform_grow(builder, (void **)&builder->sources,
                        &builder->source_capacity, builder->source_count + 1u,
                        builder->budget.maximum_sources,
                        sizeof(builder->sources[0]), builder->source_count,
                        failure, err);
    if (rc == YVEX_OK)
        rc = transform_grow(builder, (void **)&builder->values,
                            &builder->value_capacity, builder->value_count + 1u,
                            builder->budget.maximum_values,
                            sizeof(builder->values[0]), builder->value_count,
                            failure, err);
    if (rc != YVEX_OK) {
        builder->state = YVEX_TRANSFORM_IR_STATE_FAILED;
        return rc;
    }
    id = builder->value_count;
    source = &builder->sources[builder->source_count];
    memset(source, 0, sizeof(*source));
    source->value_id = id;
    yvex_core_text_copy(source->source_name, sizeof(source->source_name), spec->source_name);
    yvex_core_text_copy(source->shard_name, sizeof(source->shard_name), spec->shard_name);
    source->source_tensor_index = spec->source_tensor_index;
    source->requirement_index = spec->requirement_index;
    source->source_snapshot_identity = spec->source_snapshot_identity;
    source->source_dtype = spec->source_dtype;
    source->value_dtype = spec->value_dtype;
    source->shape = spec->shape;
    source->relative_begin = spec->relative_begin;
    source->relative_end = spec->relative_end;
    source->requirement_identity = spec->requirement_identity;
    source->scope = spec->scope;
    source->subsystem = spec->subsystem;
    source->role_hint = spec->role_hint;
    source->component_identity = spec->component_identity;
    source->semantic_role = spec->semantic_role;
    source->phase_identity = spec->phase_identity;
    source->lifetime_identity = spec->lifetime_identity;
    source->unresolved_requirement_identity =
        spec->unresolved_requirement_identity;
    source->layer_index = spec->layer_index;
    source->auxiliary_index = spec->auxiliary_index;
    source->expert_index = spec->expert_index;
    source->required_uses = spec->required_uses;
    value = &builder->values[id];
    memset(value, 0, sizeof(*value));
    value->id = id;
    value->kind = YVEX_TRANSFORM_VALUE_SOURCE;
    value->semantic_id = spec->requirement_identity;
    value->canonical_ordinal = YVEX_TRANSFORM_IR_NO_ID;
    value->source_index = builder->source_count;
    value->producer_node_id = YVEX_TRANSFORM_IR_NO_ID;
    value->shape = spec->shape;
    value->dtype = spec->value_dtype;
    builder->source_count++;
    builder->value_count++;
    *value_id = id;
    return YVEX_OK;
}

int yvex_transform_builder_declare_value(
    yvex_transform_builder *builder,
    const yvex_transform_value_spec *spec,
    unsigned long long *value_id,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    yvex_transform_value *value;
    unsigned long long id;
    int invalid_state;
    int rc;

    if (value_id) *value_id = YVEX_TRANSFORM_IR_NO_ID;
    invalid_state = builder &&
        builder->state != YVEX_TRANSFORM_IR_STATE_BUILDING;
    if (!builder || builder->state != YVEX_TRANSFORM_IR_STATE_BUILDING ||
        !spec || !value_id ||
        (spec->kind != YVEX_TRANSFORM_VALUE_INTERMEDIATE &&
         spec->kind != YVEX_TRANSFORM_VALUE_TERMINAL) ||
        !transform_shape_valid(&spec->shape) ||
        spec->dtype <= YVEX_TRANSFORM_DTYPE_UNKNOWN ||
        spec->dtype > YVEX_TRANSFORM_DTYPE_REAL ||
        (spec->kind == YVEX_TRANSFORM_VALUE_TERMINAL &&
         (!transform_logical_key_valid(&spec->logical_key,
                                       builder->header.schema_version) ||
          spec->canonical_ordinal == YVEX_TRANSFORM_IR_NO_ID))) {
        if (builder) builder->state = YVEX_TRANSFORM_IR_STATE_FAILED;
        return yvex_transform_fail(
            failure,
            invalid_state
                ? YVEX_TRANSFORM_FAILURE_INVALID_STATE
                : (spec && spec->kind == YVEX_TRANSFORM_VALUE_TERMINAL
                   ? YVEX_TRANSFORM_FAILURE_INVALID_LOGICAL_KEY
                   : YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT),
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID,
            spec ? spec->canonical_ordinal : YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, 0u, err,
            "transform_builder_declare_value");
    }
    if (spec->kind == YVEX_TRANSFORM_VALUE_TERMINAL &&
        builder->terminal_count >= builder->budget.maximum_terminals) {
        builder->state = YVEX_TRANSFORM_IR_STATE_FAILED;
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, spec->canonical_ordinal,
            YVEX_TRANSFORM_IR_NO_ID, builder->budget.maximum_terminals,
            builder->terminal_count + 1u, 0u, err,
            "transform_builder_declare_value");
    }
    rc = transform_grow(builder, (void **)&builder->values,
                        &builder->value_capacity, builder->value_count + 1u,
                        builder->budget.maximum_values,
                        sizeof(builder->values[0]), builder->value_count,
                        failure, err);
    if (rc != YVEX_OK) {
        builder->state = YVEX_TRANSFORM_IR_STATE_FAILED;
        return rc;
    }
    id = builder->value_count++;
    value = &builder->values[id];
    memset(value, 0, sizeof(*value));
    value->id = id;
    value->kind = spec->kind;
    value->semantic_id = spec->semantic_id;
    value->canonical_ordinal = spec->canonical_ordinal;
    value->source_index = YVEX_TRANSFORM_IR_NO_ID;
    value->producer_node_id = YVEX_TRANSFORM_IR_NO_ID;
    value->shape = spec->shape;
    value->dtype = spec->dtype;
    value->precision = spec->precision;
    value->logical_key = spec->logical_key;
    if (spec->kind == YVEX_TRANSFORM_VALUE_TERMINAL)
        builder->terminal_count++;
    *value_id = id;
    return YVEX_OK;
}

int yvex_transform_builder_add_node(
    yvex_transform_builder *builder,
    const yvex_transform_node_spec *spec,
    unsigned long long *node_id,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    yvex_transform_builder_node *entry;
    unsigned long long id;
    int invalid_state;
    int rc;

    if (node_id) *node_id = YVEX_TRANSFORM_IR_NO_ID;
    invalid_state = builder &&
        builder->state != YVEX_TRANSFORM_IR_STATE_BUILDING;
    if (!builder || builder->state != YVEX_TRANSFORM_IR_STATE_BUILDING ||
        !spec || !node_id || spec->kind >= YVEX_TRANSFORM_OP_COUNT ||
        !spec->input_value_ids || spec->input_count == 0u) {
        if (builder) builder->state = YVEX_TRANSFORM_IR_STATE_FAILED;
        return yvex_transform_fail(
            failure,
            invalid_state
                ? YVEX_TRANSFORM_FAILURE_INVALID_STATE
                : YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            spec ? spec->output_value_id : YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID, 1u,
            spec ? spec->input_count : 0u, 0u, err,
            "transform_builder_add_node");
    }
    if (builder->edge_count > ULLONG_MAX - spec->input_count) {
        builder->state = YVEX_TRANSFORM_IR_STATE_FAILED;
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
            spec->output_value_id, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, ULLONG_MAX, spec->input_count, 0u, err,
            "transform_builder_add_node");
    }
    rc = transform_grow(builder, (void **)&builder->nodes,
                        &builder->node_capacity, builder->node_count + 1u,
                        builder->budget.maximum_nodes,
                        sizeof(builder->nodes[0]), builder->node_count,
                        failure, err);
    if (rc == YVEX_OK)
        rc = transform_grow(builder, (void **)&builder->edges,
                            &builder->edge_capacity,
                            builder->edge_count + spec->input_count,
                            builder->budget.maximum_edges,
                            sizeof(builder->edges[0]), builder->edge_count,
                            failure, err);
    if (rc != YVEX_OK) {
        builder->state = YVEX_TRANSFORM_IR_STATE_FAILED;
        return rc;
    }
    id = builder->node_count++;
    entry = &builder->nodes[id];
    memset(entry, 0, sizeof(*entry));
    entry->provisional_id = id;
    entry->node.id = id;
    entry->node.kind = spec->kind;
    entry->node.output_value_id = spec->output_value_id;
    entry->node.input_offset = builder->edge_count;
    entry->node.input_count = spec->input_count;
    entry->node.axis = spec->axis;
    entry->node.permutation_rank = spec->permutation_rank;
    memcpy(entry->node.permutation, spec->permutation,
           sizeof(entry->node.permutation));
    entry->node.expert_count = spec->expert_count;
    entry->node.packing_factor = spec->packing_factor;
    entry->node.scale_group_width = spec->scale_group_width;
    entry->node.scale_block_rows = spec->scale_block_rows;
    entry->node.scale_block_columns = spec->scale_block_columns;
    entry->node.numeric = spec->numeric;
    entry->node.ordering = spec->ordering;
    entry->node.payload_execution_required =
        spec->payload_execution_required != 0;
    memcpy(&builder->edges[builder->edge_count], spec->input_value_ids,
           (size_t)spec->input_count * sizeof(builder->edges[0]));
    builder->edge_count += spec->input_count;
    *node_id = id;
    return YVEX_OK;
}

int yvex_transform_builder_seal(
    yvex_transform_builder *builder,
    yvex_transform_ir **out,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    int rc;

    if (out) *out = NULL;
    if (!builder || !out || builder->state != YVEX_TRANSFORM_IR_STATE_BUILDING) {
        return yvex_transform_fail(
            failure, builder ? YVEX_TRANSFORM_FAILURE_INVALID_STATE
                             : YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_STATE_BUILDING,
            builder ? builder->state : YVEX_TRANSFORM_IR_STATE_RELEASED,
            0u, err, "transform_builder_seal");
    }
    rc = yvex_transform_ir_validate_and_seal(builder, out, failure, err);
    builder->state = rc == YVEX_OK ? YVEX_TRANSFORM_IR_STATE_SEALED
                                   : YVEX_TRANSFORM_IR_STATE_FAILED;
    if (rc == YVEX_OK)
        yvex_core_execution_observation_record(
            YVEX_CORE_OBSERVE_TRANSFORM_PLAN, 1ull);
    return rc;
}

void yvex_transform_builder_release(yvex_transform_builder **builder_ptr)
{
    yvex_transform_builder *builder;
    yvex_transform_allocator allocator;

    if (!builder_ptr || !*builder_ptr) return;
    builder = *builder_ptr;
    allocator = builder->allocator;
    if (builder->edges) allocator.release(builder->edges, allocator.context);
    if (builder->nodes) allocator.release(builder->nodes, allocator.context);
    if (builder->values) allocator.release(builder->values, allocator.context);
    if (builder->sources) allocator.release(builder->sources, allocator.context);
    builder->state = YVEX_TRANSFORM_IR_STATE_RELEASED;
    allocator.release(builder, allocator.context);
    *builder_ptr = NULL;
}

void yvex_transform_ir_release(yvex_transform_ir **ir_ptr)
{
    yvex_transform_ir *ir;
    yvex_transform_allocator allocator;

    if (!ir_ptr || !*ir_ptr) return;
    ir = *ir_ptr;
    allocator = ir->allocator;
    if (ir->terminal_index)
        allocator.release(ir->terminal_index, allocator.context);
    if (ir->source_index)
        allocator.release(ir->source_index, allocator.context);
    if (ir->terminal_values)
        allocator.release(ir->terminal_values, allocator.context);
    if (ir->topological_order)
        allocator.release(ir->topological_order, allocator.context);
    if (ir->edges) allocator.release(ir->edges, allocator.context);
    if (ir->nodes) allocator.release(ir->nodes, allocator.context);
    if (ir->values) allocator.release(ir->values, allocator.context);
    if (ir->sources) allocator.release(ir->sources, allocator.context);
    allocator.release(ir, allocator.context);
    *ir_ptr = NULL;
}

const yvex_transform_ir_summary *yvex_transform_ir_summary_get(
    const yvex_transform_ir *ir)
{
    return ir ? &ir->summary : NULL;
}

const yvex_transform_source_value *yvex_transform_ir_source_at(
    const yvex_transform_ir *ir, unsigned long long index)
{
    return ir && index < ir->summary.source_value_count
        ? &ir->sources[index] : NULL;
}

const yvex_transform_source_value *yvex_transform_ir_source_find(
    const yvex_transform_ir *ir, const char *source_name)
{
    unsigned long long hash;
    unsigned long long slot;
    unsigned long long probe;

    if (!ir || !source_name || !source_name[0] ||
        ir->source_index_capacity == 0u) return NULL;
    hash = yvex_transform_hash_string(source_name);
    slot = hash & (ir->source_index_capacity - 1u);
    for (probe = 0u; probe < ir->source_index_capacity &&
         ir->source_index[slot].value_plus_one; ++probe) {
        if (ir->source_index[slot].hash == hash) {
            const yvex_transform_source_value *source =
                &ir->sources[ir->source_index[slot].value_plus_one - 1u];
            if (strcmp(source->source_name, source_name) == 0) return source;
        }
        slot = (slot + 1u) & (ir->source_index_capacity - 1u);
    }
    return NULL;
}

static const yvex_transform_value *transform_ir_value_at(
    const yvex_transform_ir *ir, unsigned long long value_id)
{
    return ir && value_id < ir->summary.value_count
        ? &ir->values[value_id] : NULL;
}

const yvex_transform_node *yvex_transform_ir_node_at(
    const yvex_transform_ir *ir, unsigned long long node_id)
{
    return ir && node_id < ir->summary.node_count ? &ir->nodes[node_id] : NULL;
}

const yvex_transform_node *yvex_transform_ir_node_topological_at(
    const yvex_transform_ir *ir, unsigned long long ordinal)
{
    if (!ir || ordinal >= ir->summary.node_count) return NULL;
    return &ir->nodes[ir->topological_order[ordinal]];
}

const yvex_transform_value *yvex_transform_ir_terminal_at(
    const yvex_transform_ir *ir, unsigned long long ordinal)
{
    if (!ir || ordinal >= ir->summary.terminal_count) return NULL;
    return &ir->values[ir->terminal_values[ordinal]];
}

const yvex_transform_value *yvex_transform_ir_node_input_at(
    const yvex_transform_ir *ir, const yvex_transform_node *node,
    unsigned long long ordinal)
{
    unsigned long long value_id;

    if (!ir || !node || node->id >= ir->summary.node_count ||
        node != &ir->nodes[node->id] ||
        ordinal >= node->input_count ||
        node->input_count > ir->summary.edge_count ||
        node->input_offset > ir->summary.edge_count - node->input_count)
        return NULL;
    value_id = ir->edges[node->input_offset + ordinal];
    return transform_ir_value_at(ir, value_id);
}
int yvex_transform_logical_key_equal(
    const yvex_transform_logical_key *left,
    const yvex_transform_logical_key *right)
{
    return left && right && left->scope == right->scope &&
        left->subsystem == right->subsystem && left->role == right->role &&
        left->component_identity == right->component_identity &&
        left->semantic_role == right->semantic_role &&
        left->phase_identity == right->phase_identity &&
        left->lifetime_identity == right->lifetime_identity &&
        left->layer_index == right->layer_index &&
        left->auxiliary_index == right->auxiliary_index &&
        left->group_index == right->group_index;
}

int yvex_transform_shape_element_count(
    const yvex_transform_shape *shape,
    unsigned long long *out,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    unsigned long long product = 1u;
    unsigned int index;

    if (out) *out = 0u;
    if (!out || !transform_shape_valid(shape)) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_INVALID_SHAPE,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, 0u, err,
            "transform_shape_element_count");
    }
    for (index = 0u; index < shape->rank; ++index) {
        if (shape->dims[index] > ULLONG_MAX / product) {
            return yvex_transform_fail(
                failure, YVEX_TRANSFORM_FAILURE_DIMENSION_OVERFLOW,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, ULLONG_MAX, shape->dims[index],
                index, err, "transform_shape_element_count");
        }
        product *= shape->dims[index];
    }
    *out = product;
    return YVEX_OK;
}

const char *yvex_transform_failure_name(yvex_transform_failure_code code)
{
    size_t count = sizeof(transform_failure_names) /
                   sizeof(transform_failure_names[0]);

    return code >= 0 && (size_t)code < count
               ? transform_failure_names[code]
               : "unknown-transform-failure";
}

struct yvex_transform_recipe_sink {
    yvex_transform_builder *builder;
    unsigned long long source_count;
    unsigned long long terminal_ordinal;
};

int yvex_transform_recipe_sink_add(
    yvex_transform_recipe_sink *sink, const yvex_transform_recipe *recipe,
    yvex_transform_failure *failure, yvex_error *err)
{
    yvex_transform_value_spec terminal;
    yvex_transform_node_spec operation;
    unsigned long long *inputs = NULL;
    unsigned long long terminal_value = 0ull, node_id, index;
    size_t input_bytes;
    int rc;

    if (!sink || !sink->builder || !recipe || !recipe->sources ||
        !recipe->source_count ||
        recipe->source_count > (unsigned long long)(SIZE_MAX / sizeof(*inputs))) {
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, 1ull, 0ull, 0u, err,
            "transform_recipe_sink_add");
    }
    input_bytes = (size_t)recipe->source_count * sizeof(*inputs);
    inputs = sink->builder->allocator.allocate(
        input_bytes, sink->builder->allocator.context);
    if (!inputs)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_ALLOCATION,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, input_bytes, 0ull, 0u, err,
            "transform_recipe_sink_add");
    rc = YVEX_OK;
    for (index = 0ull; rc == YVEX_OK && index < recipe->source_count; ++index)
        rc = yvex_transform_builder_add_source(
            sink->builder, &recipe->sources[index], &inputs[index], failure, err);
    terminal = recipe->terminal;
    terminal.kind = YVEX_TRANSFORM_VALUE_TERMINAL;
    terminal.canonical_ordinal = sink->terminal_ordinal;
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_declare_value(
            sink->builder, &terminal, &terminal_value, failure, err);
    operation = recipe->operation;
    operation.output_value_id = terminal_value;
    operation.input_value_ids = inputs;
    operation.input_count = recipe->source_count;
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_add_node(
            sink->builder, &operation, &node_id, failure, err);
    sink->builder->allocator.release(inputs, sink->builder->allocator.context);
    if (rc == YVEX_OK) {
        sink->source_count += recipe->source_count;
        ++sink->terminal_ordinal;
    }
    return rc;
}

int yvex_transform_recipe_compile(
    yvex_transform_ir **out, const yvex_transform_header *header,
    yvex_transform_recipe_project_fn project, void *context,
    const yvex_transform_builder_options *options,
    yvex_transform_failure *failure, yvex_error *err)
{
    yvex_transform_recipe_sink sink = {0};
    const yvex_transform_ir_summary *summary;
    int rc;

    if (out) *out = NULL;
    if (!out || !header || !project)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, 1ull, 0ull, 0u, err,
            "transform_recipe_compile");
    rc = yvex_transform_builder_create(
        &sink.builder, header, options, failure, err);
    if (rc == YVEX_OK) rc = project(context, &sink, failure, err);
    if (rc == YVEX_OK &&
        (sink.source_count != header->expected_source_count ||
         sink.terminal_ordinal != header->expected_terminal_count))
        rc = yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_MISSING_TERMINAL,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, sink.terminal_ordinal,
            YVEX_TRANSFORM_IR_NO_ID, header->expected_terminal_count,
            sink.terminal_ordinal, 0u, err, "transform_recipe_compile");
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_seal(sink.builder, out, failure, err);
    yvex_transform_builder_release(&sink.builder);
    summary = rc == YVEX_OK ? yvex_transform_ir_summary_get(*out) : NULL;
    if (rc == YVEX_OK &&
        (!summary || !summary->complete ||
         summary->source_value_count != header->expected_source_count ||
         summary->terminal_count != header->expected_terminal_count ||
         summary->node_count != header->expected_terminal_count ||
         summary->payload_bytes_read != 0ull)) {
        yvex_transform_ir_release(out);
        rc = yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_SEAL,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, header->expected_terminal_count,
            summary ? summary->terminal_count : 0ull, 0u, err,
            "transform_recipe_compile");
    }
    return rc;
}
/* The family transform recipe registers semantics in the generic sealed IR. */
typedef struct {
    yvex_transform_recipe_sink *sink;
    const yvex_model_family_api *family;
    const yvex_source_verification *verification;
    const yvex_deepseek_v4_ir *architecture;
    const yvex_deepseek_v4_model_spec *model;
    const yvex_deepseek_tensor_coverage *coverage;
    const yvex_deepseek_tensor_coverage_summary *coverage_summary;
    yvex_transform_allocator temporary_allocator;
    yvex_transform_failure *failure;
    yvex_error *err;
} deepseek_transform_builder;

static void *deepseek_default_allocate(size_t size, void *context)
{
    (void)context;
    return malloc(size);
}

static void deepseek_default_release(void *allocation, void *context)
{
    (void)context;
    free(allocation);
}

static unsigned long long deepseek_hash_text(unsigned long long hash,
                                             const char *text)
{
    return yvex_core_hash_mix_bytes(hash, text, strlen(text) + 1u);
}

static yvex_transform_scope deepseek_scope(
    yvex_tensor_scope scope)
{
    if (scope == YVEX_TENSOR_SCOPE_MAIN_LAYER)
        return YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    if (scope == YVEX_TENSOR_SCOPE_DRAFT)
        return YVEX_TRANSFORM_SCOPE_AUXILIARY;
    return YVEX_TRANSFORM_SCOPE_GLOBAL;
}

static yvex_transform_subsystem deepseek_subsystem(
    yvex_tensor_collection collection)
{
    return collection < YVEX_TENSOR_COLLECTION_COUNT
        ? deepseek_subsystems[collection] : YVEX_TRANSFORM_SUBSYSTEM_COUNT;
}

static yvex_transform_dtype deepseek_dtype(yvex_native_dtype dtype,
                                           int packed_fp4)
{
    if (packed_fp4) return YVEX_TRANSFORM_DTYPE_PACKED_FP4;
    switch (dtype) {
    case YVEX_NATIVE_DTYPE_F32: return YVEX_TRANSFORM_DTYPE_F32;
    case YVEX_NATIVE_DTYPE_F16: return YVEX_TRANSFORM_DTYPE_F16;
    case YVEX_NATIVE_DTYPE_BF16: return YVEX_TRANSFORM_DTYPE_BF16;
    case YVEX_NATIVE_DTYPE_I32: return YVEX_TRANSFORM_DTYPE_I32;
    case YVEX_NATIVE_DTYPE_I64: return YVEX_TRANSFORM_DTYPE_I64;
    case YVEX_NATIVE_DTYPE_F8_E4M3: return YVEX_TRANSFORM_DTYPE_FP8_E4M3;
    case YVEX_NATIVE_DTYPE_F8_E8M0: return YVEX_TRANSFORM_DTYPE_E8M0_SCALE;
    default: return YVEX_TRANSFORM_DTYPE_UNKNOWN;
    }
}

static unsigned int deepseek_physical_classes(yvex_transform_dtype dtype)
{
    switch (dtype) {
    case YVEX_TRANSFORM_DTYPE_F32: return YVEX_TRANSFORM_PHYSICAL_F32;
    case YVEX_TRANSFORM_DTYPE_F16: return YVEX_TRANSFORM_PHYSICAL_F16 |
                                           YVEX_TRANSFORM_PHYSICAL_F32;
    case YVEX_TRANSFORM_DTYPE_BF16: return YVEX_TRANSFORM_PHYSICAL_BF16 |
                                            YVEX_TRANSFORM_PHYSICAL_F32;
    case YVEX_TRANSFORM_DTYPE_I32: return YVEX_TRANSFORM_PHYSICAL_I32;
    default: return YVEX_TRANSFORM_PHYSICAL_F32 |
                    YVEX_TRANSFORM_PHYSICAL_F16 |
                    YVEX_TRANSFORM_PHYSICAL_BF16 |
                    YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
    }
}

static int deepseek_refuse(deepseek_transform_builder *builder,
                           yvex_transform_failure_code code,
                           unsigned long long expected,
                           unsigned long long actual,
                           const char *where)
{
    return yvex_transform_fail(
        builder ? builder->failure : NULL, code,
        YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
        YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
        YVEX_TRANSFORM_IR_NO_ID, expected, actual, 0u,
        builder ? builder->err : NULL, where);
}

static int deepseek_add_source(deepseek_transform_builder *builder,
                               const char *name,
                               yvex_tensor_role role,
                               yvex_tensor_collection collection,
                               yvex_tensor_scope scope,
                               unsigned long long layer,
                               unsigned long long auxiliary,
                               unsigned long long expert,
                               yvex_native_dtype expected_dtype,
                               int packed_fp4,
                               yvex_transform_source_spec *spec)
{
    const yvex_deepseek_tensor_coverage_row *row;
    unsigned long long requirement_index;
    unsigned long long source_index;
    unsigned long long identity;
    unsigned int dimension;

    row = builder->family->coverage.find(builder->coverage, name);
    if (!row || !row->source ||
        !builder->family->coverage.find_index(
            builder->coverage, name, &requirement_index) ||
        !builder->family->coverage.find_source_index(
            builder->coverage, name, &source_index)) {
        return deepseek_refuse(builder, YVEX_TRANSFORM_FAILURE_MISSING_SOURCE,
                               1u, 0u, "deepseek_transform_source");
    }
    if (row->collection != collection || row->scope != scope ||
        row->layer_index != layer || row->source->dtype != expected_dtype ||
        (expert != YVEX_DEEPSEEK_TENSOR_NO_INDEX &&
         row->expert_index != expert)) {
        return deepseek_refuse(
            builder,
            row->source->dtype != expected_dtype
                ? YVEX_TRANSFORM_FAILURE_UNSUPPORTED_SOURCE_DTYPE
                : YVEX_TRANSFORM_FAILURE_UNEXPECTED_SOURCE,
            (unsigned long long)expected_dtype,
            (unsigned long long)row->source->dtype,
            "deepseek_transform_source");
    }
    memset(spec, 0, sizeof(*spec));
    spec->source_name = row->source->name;
    spec->shard_name = row->source->shard_path;
    spec->source_tensor_index = source_index;
    spec->requirement_index = requirement_index;
    spec->source_snapshot_identity = builder->coverage_summary->source_identity;
    spec->source_dtype = row->source->dtype;
    spec->value_dtype = deepseek_dtype(row->source->dtype, packed_fp4);
    spec->shape.rank = row->source->rank;
    for (dimension = 0u; dimension < row->source->rank; ++dimension)
        spec->shape.dims[dimension] = row->source->dims[dimension];
    spec->relative_begin = row->source->data_start;
    spec->relative_end = row->source->data_end;
    identity = deepseek_hash_text(1469598103934665603ull, name);
    identity = yvex_core_hash_mix_u64(identity,
                                 builder->coverage_summary->coverage_identity);
    identity = yvex_core_hash_mix_u64(identity, requirement_index);
    spec->requirement_identity = identity;
    spec->scope = deepseek_scope(scope);
    spec->subsystem = deepseek_subsystem(collection);
    spec->role_hint = role;
    spec->layer_index = layer;
    spec->auxiliary_index = auxiliary;
    spec->expert_index = expert;
    spec->required_uses = 1u;
    return YVEX_OK;
}

static int deepseek_add_terminal(deepseek_transform_builder *builder,
                                 yvex_tensor_role role,
                                 yvex_tensor_collection collection,
                                 yvex_tensor_scope scope,
                                 unsigned long long layer,
                                 unsigned long long auxiliary,
                                 const yvex_transform_source_spec *sources,
                                 unsigned long long source_count,
                                 const yvex_transform_shape *shape,
                                 yvex_transform_dtype dtype,
                                 const yvex_transform_precision_constraint *precision,
                                 const yvex_transform_node_spec *operation)
{
    yvex_transform_recipe recipe = {0};
    unsigned long long semantic = 1469598103934665603ull;

    semantic = yvex_core_hash_mix_u64(semantic, (unsigned long long)scope);
    semantic = yvex_core_hash_mix_u64(semantic, (unsigned long long)collection);
    semantic = yvex_core_hash_mix_u64(semantic, (unsigned long long)role);
    semantic = yvex_core_hash_mix_u64(semantic, layer);
    semantic = yvex_core_hash_mix_u64(semantic, auxiliary);
    recipe.sources = sources;
    recipe.source_count = source_count;
    recipe.terminal.semantic_id = semantic;
    recipe.terminal.shape = *shape;
    recipe.terminal.dtype = dtype;
    recipe.terminal.precision = *precision;
    recipe.terminal.logical_key.scope = deepseek_scope(scope);
    recipe.terminal.logical_key.subsystem = deepseek_subsystem(collection);
    recipe.terminal.logical_key.role = role;
    recipe.terminal.logical_key.layer_index = scope == YVEX_TENSOR_SCOPE_GLOBAL
        ? YVEX_TRANSFORM_IR_NO_ID : layer;
    recipe.terminal.logical_key.auxiliary_index =
        scope == YVEX_TENSOR_SCOPE_DRAFT
            ? auxiliary : YVEX_TRANSFORM_IR_NO_ID;
    recipe.terminal.logical_key.group_index = 0u;
    recipe.operation = *operation;
    return yvex_transform_recipe_sink_add(
        builder->sink, &recipe, builder->failure, builder->err);
}

static int deepseek_add_direct(deepseek_transform_builder *builder,
                               yvex_tensor_role role,
                               yvex_tensor_collection collection,
                               yvex_tensor_scope scope,
                               unsigned long long layer,
                               unsigned long long auxiliary,
                               const char *source_name,
                               yvex_native_dtype source_dtype,
                               int checked_cast)
{
    const yvex_deepseek_tensor_coverage_row *row =
        builder->family->coverage.find(builder->coverage, source_name);
    yvex_transform_precision_constraint precision;
    yvex_transform_node_spec node;
    yvex_transform_shape shape;
    yvex_transform_source_spec source;
    yvex_transform_dtype output_dtype;
    unsigned int dimension;
    int rc;

    if (!row || !row->source)
        return deepseek_refuse(builder, YVEX_TRANSFORM_FAILURE_MISSING_SOURCE,
                               1u, 0u, "deepseek_transform_direct");
    rc = deepseek_add_source(
        builder, source_name, role, collection, scope, layer, auxiliary,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, source_dtype, 0, &source);
    if (rc != YVEX_OK) return rc;
    memset(&shape, 0, sizeof(shape));
    shape.rank = row->source->rank;
    for (dimension = 0u; dimension < shape.rank; ++dimension)
        shape.dims[dimension] = row->source->dims[dimension];
    output_dtype = checked_cast ? YVEX_TRANSFORM_DTYPE_I32
                                : deepseek_dtype(source_dtype, 0);
    memset(&precision, 0, sizeof(precision));
    precision.allowed_physical_classes = deepseek_physical_classes(output_dtype);
    if (checked_cast) {
        precision.flags = YVEX_TRANSFORM_PRECISION_LOSSLESS |
                          YVEX_TRANSFORM_PRECISION_RANGE_PROOF |
                          YVEX_TRANSFORM_PRECISION_INTEGER_ONLY;
        precision.range_proof_required = 1;
    } else {
        precision.flags = YVEX_TRANSFORM_PRECISION_EXACT;
    }
    memset(&node, 0, sizeof(node));
    node.kind = checked_cast ? YVEX_TRANSFORM_OP_CHECKED_CAST
                             : YVEX_TRANSFORM_OP_IDENTITY;
    node.numeric = checked_cast ? YVEX_TRANSFORM_NUMERIC_RANGE_PROOF
                                : YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    return deepseek_add_terminal(builder, role, collection, scope, layer,
                                 auxiliary, &source, 1u, &shape, output_dtype,
                                 &precision, &node);
}

static int deepseek_add_fp8(deepseek_transform_builder *builder,
                            yvex_tensor_role role,
                            yvex_tensor_collection collection,
                            yvex_tensor_scope scope,
                            unsigned long long layer,
                            unsigned long long auxiliary,
                            const char *base)
{
    char weight[256];
    char scale[256];
    const yvex_deepseek_tensor_coverage_row *row;
    yvex_transform_precision_constraint precision;
    yvex_transform_node_spec node;
    yvex_transform_shape shape;
    yvex_transform_source_spec sources[2];
    unsigned int dimension;
    int rc;

    (void)snprintf(weight, sizeof(weight), "%s.weight", base);
    (void)snprintf(scale, sizeof(scale), "%s.scale", base);
    row = builder->family->coverage.find(builder->coverage, weight);
    if (!row || !row->source)
        return deepseek_refuse(builder, YVEX_TRANSFORM_FAILURE_MISSING_SOURCE,
                               1u, 0u, "deepseek_transform_fp8");
    rc = deepseek_add_source(
        builder, weight, role, collection, scope, layer, auxiliary,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_F8_E4M3, 0,
        &sources[0]);
    if (rc == YVEX_OK)
        rc = deepseek_add_source(
            builder, scale, role, collection, scope, layer, auxiliary,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_F8_E8M0, 0,
            &sources[1]);
    if (rc != YVEX_OK) return rc;
    memset(&shape, 0, sizeof(shape));
    shape.rank = row->source->rank;
    for (dimension = 0u; dimension < shape.rank; ++dimension)
        shape.dims[dimension] = row->source->dims[dimension];
    memset(&precision, 0, sizeof(precision));
    precision.flags = YVEX_TRANSFORM_PRECISION_SCALE_PAIRED |
                      YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT |
                      YVEX_TRANSFORM_PRECISION_REFERENCE_COMPUTE;
    precision.allowed_physical_classes =
        YVEX_TRANSFORM_PHYSICAL_F32 | YVEX_TRANSFORM_PHYSICAL_F16 |
        YVEX_TRANSFORM_PHYSICAL_BF16 | YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
    precision.approximation_allowed = 1;
    precision.reference_compute_required = 1;
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR;
    node.scale_block_rows = builder->model->source_constraint.quant_block_rows;
    node.scale_block_columns =
        builder->model->source_constraint.quant_block_columns;
    node.numeric = YVEX_TRANSFORM_NUMERIC_LOSSLESS;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    return deepseek_add_terminal(
        builder, role, collection, scope, layer, auxiliary, sources, 2u,
        &shape, YVEX_TRANSFORM_DTYPE_REAL, &precision, &node);
}

static int deepseek_add_experts(deepseek_transform_builder *builder,
                                yvex_tensor_role role,
                                yvex_tensor_scope scope,
                                unsigned long long layer,
                                unsigned long long auxiliary,
                                const char *prefix,
                                const char *projection,
                                unsigned long long expert_count)
{
    char weight[256];
    char scale[256];
    const yvex_deepseek_tensor_coverage_row *first;
    yvex_transform_precision_constraint precision;
    yvex_transform_node_spec node;
    yvex_transform_shape shape;
    yvex_transform_source_spec *sources = NULL;
    unsigned long long input_count;
    unsigned long long logical_width;
    unsigned long long expert;
    size_t bytes;
    int rc = YVEX_OK;

    if (!expert_count || expert_count > ULLONG_MAX / 2u)
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_INVALID_AGGREGATION,
            1u, expert_count, "deepseek_transform_experts");
    input_count = expert_count * 2u;
    if (input_count > (unsigned long long)(SIZE_MAX / sizeof(sources[0])))
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
            SIZE_MAX, input_count, "deepseek_transform_experts");
    bytes = (size_t)input_count * sizeof(sources[0]);
    sources = (yvex_transform_source_spec *)builder->temporary_allocator.allocate(
        bytes, builder->temporary_allocator.context);
    if (!sources)
        return deepseek_refuse(builder, YVEX_TRANSFORM_FAILURE_ALLOCATION,
                               bytes, 0u, "deepseek_transform_experts");
    (void)snprintf(weight, sizeof(weight), "%s.ffn.experts.0.%s.weight",
                   prefix, projection);
    first = builder->family->coverage.find(builder->coverage, weight);
    if (!first || !first->source || first->source->rank != 2u ||
        first->source->dims[1] > ULLONG_MAX /
            builder->model->source_constraint.fp4_packing_factor) {
        rc = deepseek_refuse(
            builder, first ? YVEX_TRANSFORM_FAILURE_DIMENSION_OVERFLOW
                           : YVEX_TRANSFORM_FAILURE_MISSING_SOURCE,
            1u, 0u, "deepseek_transform_experts");
        goto cleanup;
    }
    for (expert = 0u; expert < expert_count; ++expert) {
        (void)snprintf(weight, sizeof(weight),
                       "%s.ffn.experts.%llu.%s.weight", prefix, expert,
                       projection);
        (void)snprintf(scale, sizeof(scale),
                       "%s.ffn.experts.%llu.%s.scale", prefix, expert,
                       projection);
        rc = deepseek_add_source(
            builder, weight, role,
            YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, scope, layer,
            auxiliary, expert, YVEX_NATIVE_DTYPE_I8, 1,
            &sources[expert * 2u]);
        if (rc == YVEX_OK)
            rc = deepseek_add_source(
                builder, scale, role,
                YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, scope, layer,
                auxiliary, expert, YVEX_NATIVE_DTYPE_F8_E8M0, 0,
                &sources[expert * 2u + 1u]);
        if (rc != YVEX_OK) goto cleanup;
    }
    logical_width = first->source->dims[1] *
                    builder->model->source_constraint.fp4_packing_factor;
    memset(&shape, 0, sizeof(shape));
    shape.rank = 3u;
    shape.dims[0] = expert_count;
    shape.dims[1] = first->source->dims[0];
    shape.dims[2] = logical_width;
    memset(&precision, 0, sizeof(precision));
    precision.flags = YVEX_TRANSFORM_PRECISION_SCALE_PAIRED |
                      YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT |
                      YVEX_TRANSFORM_PRECISION_REFERENCE_COMPUTE;
    precision.allowed_physical_classes =
        YVEX_TRANSFORM_PHYSICAL_F32 | YVEX_TRANSFORM_PHYSICAL_F16 |
        YVEX_TRANSFORM_PHYSICAL_BF16 | YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
    precision.approximation_allowed = 1;
    precision.reference_compute_required = 1;
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_EXPERT_AGGREGATE;
    node.axis = 0u;
    node.expert_count = expert_count;
    node.packing_factor =
        builder->model->source_constraint.fp4_packing_factor;
    node.scale_group_width =
        builder->model->source_constraint.fp4_scale_group_width;
    node.numeric = YVEX_TRANSFORM_NUMERIC_LOSSLESS;
    node.ordering = YVEX_TRANSFORM_ORDER_EXPERT_INDEX;
    node.payload_execution_required = 1;
    rc = deepseek_add_terminal(
        builder, role, YVEX_TENSOR_COLLECTION_ROUTED_EXPERT,
        scope, layer, auxiliary, sources, input_count, &shape,
        YVEX_TRANSFORM_DTYPE_REAL, &precision, &node);

cleanup:
    builder->temporary_allocator.release(
        sources, builder->temporary_allocator.context);
    return rc;
}

static int deepseek_add_recipe_phase(deepseek_transform_builder *builder,
                                     const char *prefix,
                                     const yvex_deepseek_v4_layer_spec *layer,
                                     yvex_tensor_scope scope,
                                     unsigned long long auxiliary,
                                     unsigned int phase)
{
    const yvex_model_family_ir_api *family_ir = &builder->family->ir;
    unsigned long long index;

    for (index = 0u; index < family_ir->recipe_count(); ++index) {
        const yvex_deepseek_tensor_recipe *recipe =
            family_ir->recipe_at(index);
        char name[256];
        int rc;

        if (!recipe || recipe->phase != phase ||
            !family_ir->recipe_enabled(recipe, layer))
            continue;
        (void)snprintf(name, sizeof(name), "%s.%s", prefix, recipe->suffix);
        if (recipe->kind == YVEX_DEEPSEEK_RECIPE_FP8_PAIR) {
            rc = deepseek_add_fp8(builder, recipe->role, recipe->collection, scope,
                                  layer->layer_index, auxiliary, name);
        } else {
            rc = deepseek_add_direct(builder, recipe->role, recipe->collection, scope,
                                     layer->layer_index, auxiliary, name, recipe->dtype,
                                     recipe->kind == YVEX_DEEPSEEK_RECIPE_CHECKED_CAST);
        }
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

static int deepseek_add_layer(deepseek_transform_builder *builder,
                              const char *prefix,
                              const yvex_deepseek_v4_layer_spec *layer,
                              yvex_tensor_scope scope,
                              unsigned long long auxiliary)
{
    size_t index;
    int rc;

    rc = deepseek_add_recipe_phase(builder, prefix, layer, scope, auxiliary, 0u);
    for (index = 0u;
         rc == YVEX_OK &&
         index < sizeof(deepseek_expert_projections) /
                     sizeof(deepseek_expert_projections[0]);
         ++index) {
        rc = deepseek_add_experts(builder, deepseek_expert_projections[index].role,
                                  scope, layer->layer_index, auxiliary, prefix,
                                  deepseek_expert_projections[index].projection,
                                  layer->moe.routed_experts);
    }
    if (rc == YVEX_OK)
        rc = deepseek_add_recipe_phase(builder, prefix, layer, scope, auxiliary, 1u);
    return rc;
}

static int deepseek_build_graph(void *context, yvex_transform_recipe_sink *sink,
                                yvex_transform_failure *failure,
                                yvex_error *err)
{
    deepseek_transform_builder *builder = context;
    const yvex_tensor_role head_roles[3] = {
        YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION,
        YVEX_TENSOR_ROLE_HC_HEAD_BASE,
        YVEX_TENSOR_ROLE_HC_HEAD_SCALE
    };
    const char *head_names[3] = {
        "hc_head_fn", "hc_head_base", "hc_head_scale"
    };
    unsigned long long layer;
    unsigned int index;
    int rc;

    builder->sink = sink;
    builder->failure = failure;
    builder->err = err;

    rc = deepseek_add_direct(
        builder, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
        YVEX_TENSOR_COLLECTION_GLOBAL,
        YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, "embed.weight",
        YVEX_NATIVE_DTYPE_BF16, 0);
    if (rc != YVEX_OK) return rc;
    rc = deepseek_add_direct(
        builder, YVEX_TENSOR_ROLE_OUTPUT_NORM,
        YVEX_TENSOR_COLLECTION_GLOBAL,
        YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, "norm.weight",
        YVEX_NATIVE_DTYPE_BF16, 0);
    if (rc != YVEX_OK) return rc;
    rc = deepseek_add_direct(
        builder, YVEX_TENSOR_ROLE_OUTPUT_HEAD,
        YVEX_TENSOR_COLLECTION_GLOBAL,
        YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, "head.weight",
        YVEX_NATIVE_DTYPE_BF16, 0);
    if (rc != YVEX_OK) return rc;
    for (index = 0u; index < 3u; ++index) {
        rc = deepseek_add_direct(
            builder, head_roles[index],
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            head_names[index], YVEX_NATIVE_DTYPE_F32, 0);
        if (rc != YVEX_OK) return rc;
    }
    for (layer = 0u; layer < builder->model->main_layer_count; ++layer) {
        char prefix[64];
        (void)snprintf(prefix, sizeof(prefix), "layers.%llu", layer);
        rc = deepseek_add_layer(
            builder, prefix,
            builder->family->ir.layer_at(builder->architecture, layer),
            YVEX_TENSOR_SCOPE_MAIN_LAYER,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX);
        if (rc != YVEX_OK) return rc;
    }
    for (layer = 0u; layer < builder->model->auxiliary_layer_count; ++layer) {
        const yvex_deepseek_v4_auxiliary_spec *aux =
            builder->family->ir.auxiliary_at(builder->architecture, layer);
        char prefix[64];
        char name[128];
        char base[128];

        if (!aux)
            return deepseek_refuse(
                builder, YVEX_TRANSFORM_FAILURE_ARCHITECTURE_NOT_ADMITTED,
                1u, 0u, "deepseek_transform_auxiliary");
        (void)snprintf(prefix, sizeof(prefix), "mtp.%llu", layer);
        rc = deepseek_add_layer(builder, prefix, &aux->layer,
                                YVEX_TENSOR_SCOPE_DRAFT, layer);
        if (rc != YVEX_OK) return rc;
#define DRAFT_DIRECT(role_id, suffix) do {                                     \
    (void)snprintf(name, sizeof(name), "%s.%s", prefix, suffix);             \
    rc = deepseek_add_direct(                                                   \
        builder, role_id, YVEX_TENSOR_COLLECTION_AUXILIARY,          \
        YVEX_TENSOR_SCOPE_DRAFT, aux->layer.layer_index, layer, name,   \
        YVEX_NATIVE_DTYPE_BF16, 0);                                             \
    if (rc != YVEX_OK) return rc;                                               \
} while (0)
        if (aux->has_feature_projection) {
            (void)snprintf(base, sizeof(base), "%s.main_proj", prefix);
            rc = deepseek_add_fp8(
                builder, YVEX_TENSOR_ROLE_DRAFT_FEATURE_PROJECTION,
                YVEX_TENSOR_COLLECTION_AUXILIARY,
                YVEX_TENSOR_SCOPE_DRAFT, aux->layer.layer_index, layer, base);
            if (rc != YVEX_OK) return rc;
        }
        if (aux->has_feature_norm)
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM, "main_norm.weight");
        if (aux->has_output_norm)
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM, "norm.weight");
        if (aux->has_markov_head) {
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING,
                         "markov_head.markov_w1.weight");
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT,
                         "markov_head.markov_w2.weight");
        }
        if (aux->has_confidence_head)
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE,
                         "confidence_head.proj.weight");
#undef DRAFT_DIRECT
        for (index = 0u; aux->has_separate_mhc_head && index < 3u; ++index) {
            (void)snprintf(name, sizeof(name), "%s.hc_head_%s", prefix,
                           index == 0u ? "fn" :
                           (index == 1u ? "base" : "scale"));
            rc = deepseek_add_direct(
                builder, head_roles[index],
                YVEX_TENSOR_COLLECTION_AUXILIARY,
                YVEX_TENSOR_SCOPE_DRAFT, aux->layer.layer_index,
                layer, name, YVEX_NATIVE_DTYPE_F32, 0);
            if (rc != YVEX_OK) return rc;
        }
    }
    return YVEX_OK;
}

static int deepseek_validate_inputs(
    deepseek_transform_builder *builder,
    const yvex_source_verification *verification,
    const yvex_deepseek_v4_ir *architecture,
    const yvex_deepseek_tensor_coverage *coverage,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_deepseek_v4_model_spec *model =
        family->ir.model(architecture);
    const yvex_deepseek_tensor_coverage_summary *summary =
        family->coverage.summary(coverage);

    memset(builder, 0, sizeof(*builder));
    builder->failure = failure;
    builder->err = err;
    if (!verification || !architecture || !coverage || !model || !summary)
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            1u, 0u, "deepseek_transform_build");
    if (!verification->verified || verification->blocker_count != 0u ||
        model->main_layer_count != 43u ||
        model->auxiliary_layer_count != 3u) {
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_ARCHITECTURE_NOT_ADMITTED,
            46u, model->main_layer_count + model->auxiliary_layer_count,
            "deepseek_transform_build");
    }
    if (!summary->complete ||
        summary->source_tensor_count != YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT ||
        summary->matched_tensor_count != YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT ||
        summary->missing_count || summary->ambiguous_count ||
        summary->unexpected_count || summary->header_scan_count != 1u ||
        summary->payload_bytes_read != 0u) {
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_COVERAGE_INCOMPLETE,
            YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT,
            summary->matched_tensor_count, "deepseek_transform_build");
    }
    if (verification->source_snapshot_identity != summary->source_identity)
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH,
            verification->source_snapshot_identity, summary->source_identity,
            "deepseek_transform_build");
    if (!verification->manifest_payload_trusted ||
        !yvex_sha256_hex_valid(verification->manifest_payload_identity) ||
        verification->manifest_payload_source_snapshot_identity !=
            summary->source_identity ||
        (strcmp(verification->manifest_payload_trust_class,
                "upstream_payload_verified") != 0 &&
         strcmp(verification->manifest_payload_trust_class,
                "local_payload_snapshot_sealed") != 0)) {
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_PAYLOAD_IDENTITY_MISMATCH,
            summary->source_identity,
            verification->manifest_payload_source_snapshot_identity,
            "deepseek_transform_build");
    }
    builder->verification = verification;
    builder->family = family;
    builder->architecture = architecture;
    builder->model = model;
    builder->coverage = coverage;
    builder->coverage_summary = summary;
    return YVEX_OK;
}

static int deepseek_transform_build(
    yvex_transform_ir **out,
    const yvex_source_verification *verification,
    const yvex_deepseek_v4_ir *architecture,
    const yvex_deepseek_tensor_coverage *coverage,
    const yvex_transform_builder_options *options,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    deepseek_transform_builder deepseek;
    yvex_transform_header header;
    char logical_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    int rc;

    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (!out)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, 0u, err,
            "deepseek_transform_build");
    rc = deepseek_validate_inputs(&deepseek, verification, architecture,
                                  coverage, failure, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_transform_deepseek_architecture_identity(
            architecture, logical_identity))
        return deepseek_refuse(
            &deepseek, YVEX_TRANSFORM_FAILURE_IDENTITY_ENCODING,
            1u, 0u, "deepseek_transform_architecture_identity");
    deepseek.temporary_allocator.allocate = deepseek_default_allocate;
    deepseek.temporary_allocator.release = deepseek_default_release;
    deepseek.temporary_allocator.context = NULL;
    if (options && options->allocator.allocate)
        deepseek.temporary_allocator = options->allocator;
    memset(&header, 0, sizeof(header));
    header.schema_version = YVEX_TRANSFORM_IR_SCHEMA_VERSION;
    header.logical_model_identity = logical_identity;
    header.source_snapshot_identity =
        deepseek.coverage_summary->source_identity;
    header.coverage_identity = deepseek.coverage_summary->coverage_identity;
    header.required_payload_identity =
        verification->manifest_payload_identity;
    header.payload_trust_class = verification->manifest_payload_trust_class;
    header.expected_source_count = YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT;
    header.expected_terminal_count = YVEX_DEEPSEEK_TRANSFORM_TERMINAL_COUNT;
    header.header_scan_count = deepseek.coverage_summary->header_scan_count;
    rc = yvex_transform_recipe_compile(
        out, &header, deepseek_build_graph, &deepseek, options, failure, err);
    if (rc == YVEX_OK) {
        const yvex_transform_ir_summary *summary =
            yvex_transform_ir_summary_get(*out);
        if (!summary || !summary->complete ||
            summary->source_value_count !=
                YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT ||
            summary->node_count != YVEX_DEEPSEEK_TRANSFORM_TERMINAL_COUNT ||
            summary->edge_count != YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT ||
            summary->terminal_count !=
                YVEX_DEEPSEEK_TRANSFORM_TERMINAL_COUNT ||
            summary->maximum_fan_in != 512u ||
            summary->payload_bytes_read != 0u) {
            yvex_transform_ir_release(out);
            return deepseek_refuse(
                &deepseek, YVEX_TRANSFORM_FAILURE_SEAL,
                YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT,
                summary ? summary->edge_count : 0u,
                "deepseek_transform_build");
        }
    }
    return rc;
}

/*
 * Publish the immutable family transform operations used by the registration table and compilation
 * consumers.
 *
 * Returns process-lifetime immutable storage; no allocation or I/O.
 */
const yvex_model_family_transform_api *yvex_model_deepseek_transform_api(void)
{
    static const yvex_model_family_transform_api api = {
        yvex_transform_deepseek_architecture_identity,
        deepseek_transform_build
    };

    return &api;
}
