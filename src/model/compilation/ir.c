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
#include "src/model/compilation/private.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *transform_failure_name(yvex_transform_failure_code code);

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
                    transform_failure_name(code), value_id, node_id,
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

static const char *transform_failure_name(yvex_transform_failure_code code)
{
    size_t count = sizeof(transform_failure_names) /
                   sizeof(transform_failure_names[0]);

    return code >= 0 && (size_t)code < count
               ? transform_failure_names[code]
               : "unknown-transform-failure";
}

struct yvex_transform_recipe_sink {
    yvex_transform_builder *builder;
    const yvex_source_tensor_snapshot *source_snapshot;
    unsigned long long source_count;
    unsigned long long terminal_ordinal;
};

int yvex_transform_recipe_sink_resolve_source(
    yvex_transform_recipe_sink *sink,
    const yvex_transform_source_requirement *requirement,
    yvex_transform_source_spec *resolved,
    yvex_transform_failure *failure, yvex_error *err)
{
    const yvex_native_weight_info *source;
    yvex_source_tensor_snapshot_facts facts = {0};
    unsigned long long source_index;
    unsigned int dimension;

    if (resolved) memset(resolved, 0, sizeof(*resolved));
    if (!sink || !sink->builder || !sink->source_snapshot || !requirement ||
        !resolved || !requirement->source_name ||
        !yvex_source_tensor_snapshot_find_index(
            sink->source_snapshot, requirement->source_name, &source_index))
        return yvex_transform_fail(
            failure, requirement && requirement->source_name
                         ? YVEX_TRANSFORM_FAILURE_MISSING_SOURCE
                         : YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, 1ull, 0ull, 0u, err,
            "transform_recipe_resolve_source");
    source = yvex_source_tensor_snapshot_at(sink->source_snapshot, source_index);
    if (!source ||
        yvex_source_tensor_snapshot_facts_get(
            sink->source_snapshot, &facts, err) != YVEX_OK ||
        facts.identity != sink->builder->header.source_snapshot_identity)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID, source_index,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            sink->builder->header.source_snapshot_identity, facts.identity,
            0u, err, "transform_recipe_resolve_source");
    if (source->dtype != requirement->source_dtype)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_UNSUPPORTED_SOURCE_DTYPE,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID, source_index,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            requirement->source_dtype, source->dtype, 0u, err,
            "transform_recipe_resolve_source");
    if (source->rank != requirement->shape.rank)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_INVALID_RANK,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID, source_index,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            requirement->shape.rank, source->rank, 0u, err,
            "transform_recipe_resolve_source");
    for (dimension = 0u; dimension < source->rank; ++dimension)
        if (source->dims[dimension] != requirement->shape.dims[dimension])
            return yvex_transform_fail(
                failure, YVEX_TRANSFORM_FAILURE_INVALID_SHAPE,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                source_index, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, requirement->shape.dims[dimension],
                source->dims[dimension], dimension, err,
                "transform_recipe_resolve_source");
    *resolved = (yvex_transform_source_spec){
        .source_name = source->name,
        .shard_name = source->shard_path,
        .source_tensor_index = source_index,
        .requirement_index = requirement->requirement_index,
        .source_snapshot_identity = facts.identity,
        .source_dtype = source->dtype,
        .value_dtype = requirement->value_dtype,
        .shape = requirement->shape,
        .relative_begin = source->data_start,
        .relative_end = source->data_end,
        .scope = requirement->scope,
        .subsystem = requirement->subsystem,
        .role_hint = requirement->role_hint,
        .component_identity = requirement->component_identity,
        .semantic_role = requirement->semantic_role,
        .phase_identity = requirement->phase_identity,
        .lifetime_identity = requirement->lifetime_identity,
        .unresolved_requirement_identity =
            requirement->unresolved_requirement_identity,
        .layer_index = requirement->layer_index,
        .auxiliary_index = requirement->auxiliary_index,
        .expert_index = requirement->expert_index,
        .required_uses = requirement->required_uses};
    return YVEX_OK;
}

static const char *transform_coverage_scope_name(yvex_transform_scope scope)
{
    static const char *const names[] = {"global", "main-layer", "draft"};

    return scope <= YVEX_TRANSFORM_SCOPE_AUXILIARY ? names[scope] : NULL;
}

static const char *transform_coverage_subsystem_name(
    yvex_transform_subsystem subsystem)
{
    static const char *const names[] = {
        "global", "attention", "compressor", "indexer", "norm", "mhc",
        "router", "routed-expert", "shared-expert", "output", "auxiliary",
        "sequence-mixer"};

    return subsystem < YVEX_TRANSFORM_SUBSYSTEM_COUNT ? names[subsystem] : NULL;
}

static int transform_recipe_source_coverage_finalize(
    yvex_transform_recipe_sink *sink, yvex_transform_failure *failure,
    yvex_error *err)
{
    yvex_transform_builder *builder = sink->builder;
    yvex_source_tensor_snapshot_facts facts = {0};
    yvex_transform_source_value **requirements = NULL;
    unsigned long long coverage = 1469598103934665603ull;
    unsigned long long index;
    size_t bytes;
    int rc = YVEX_OK;

    if (!sink->source_snapshot) return YVEX_OK;
    if (yvex_source_tensor_snapshot_facts_get(
            sink->source_snapshot, &facts, err) != YVEX_OK ||
        facts.identity != builder->header.source_snapshot_identity ||
        facts.tensor_count != builder->source_count)
        return yvex_transform_fail(
            failure, facts.identity != builder->header.source_snapshot_identity
                         ? YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH
                         : YVEX_TRANSFORM_FAILURE_COVERAGE_INCOMPLETE,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, facts.tensor_count,
            builder->source_count, 0u, err,
            "transform_recipe_source_coverage");
    if (builder->source_count > SIZE_MAX / sizeof(*requirements))
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, SIZE_MAX, builder->source_count, 0u,
            err, "transform_recipe_source_coverage");
    bytes = (size_t)builder->source_count * sizeof(*requirements);
    requirements = builder->allocator.allocate(
        bytes, builder->allocator.context);
    if (!requirements)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_ALLOCATION,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, bytes, 0u, 0u, err,
            "transform_recipe_source_coverage");
    memset(requirements, 0, bytes);
    for (index = 0u; index < builder->source_count; ++index) {
        yvex_transform_source_value *source = &builder->sources[index];
        const yvex_native_weight_info *actual =
            yvex_source_tensor_snapshot_at(
                sink->source_snapshot, source->source_tensor_index);
        unsigned int dimension;
        int shape_matches = actual && actual->rank == source->shape.rank;

        for (dimension = 0u; shape_matches && dimension < actual->rank;
             ++dimension)
            shape_matches = actual->dims[dimension] ==
                            source->shape.dims[dimension];
        if (source->requirement_index >= builder->source_count ||
            (source->requirement_index < builder->source_count &&
             requirements[source->requirement_index]) || !actual ||
            strcmp(actual->name, source->source_name) != 0 ||
            actual->dtype != source->source_dtype || !shape_matches ||
            !actual->shard_path || !source->shard_name[0] ||
            strcmp(actual->shard_path, source->shard_name) != 0 ||
            actual->data_start != source->relative_begin ||
            actual->data_end != source->relative_end) {
            rc = yvex_transform_fail(
                failure, source->requirement_index < builder->source_count &&
                                 requirements[source->requirement_index]
                             ? YVEX_TRANSFORM_FAILURE_DUPLICATE_SOURCE
                             : YVEX_TRANSFORM_FAILURE_UNEXPECTED_SOURCE,
                source->value_id, YVEX_TRANSFORM_IR_NO_ID,
                source->source_tensor_index, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, builder->source_count,
                source->requirement_index, 0u, err,
                "transform_recipe_source_coverage");
            break;
        }
        requirements[source->requirement_index] = source;
    }
    coverage = yvex_core_hash_mix_u64(coverage, facts.identity);
    for (index = 0u; rc == YVEX_OK && index < builder->source_count; ++index) {
        yvex_transform_source_value *source = requirements[index];
        const char *subsystem = source
            ? transform_coverage_subsystem_name(source->subsystem) : NULL;
        const char *scope = source
            ? transform_coverage_scope_name(source->scope) : NULL;
        if (!source || !subsystem || !scope) {
            rc = yvex_transform_fail(
                failure, YVEX_TRANSFORM_FAILURE_COVERAGE_INCOMPLETE,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, 0u, err,
                "transform_recipe_source_coverage");
            break;
        }
        coverage = yvex_core_hash_mix_bytes(
            coverage, source->source_name, strlen(source->source_name) + 1u);
        coverage = yvex_core_hash_mix_bytes(
            coverage, subsystem, strlen(subsystem) + 1u);
        coverage = yvex_core_hash_mix_bytes(
            coverage, scope, strlen(scope) + 1u);
        coverage = yvex_core_hash_mix_u64(coverage, source->layer_index);
        coverage = yvex_core_hash_mix_u64(coverage, source->expert_index);
    }
    if (rc == YVEX_OK && builder->header.coverage_identity &&
        builder->header.coverage_identity != coverage)
        rc = yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_COVERAGE_INCOMPLETE,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, builder->header.coverage_identity,
            coverage, 0u, err, "transform_recipe_source_coverage");
    if (rc == YVEX_OK) {
        builder->header.coverage_identity = coverage;
        for (index = 0u; index < builder->source_count; ++index) {
            yvex_transform_source_value *source = requirements[index];
            unsigned long long identity =
                yvex_transform_hash_string(source->source_name);
            identity = yvex_core_hash_mix_u64(identity, coverage);
            source->requirement_identity =
                yvex_core_hash_mix_u64(identity, index);
            builder->values[source->value_id].semantic_id =
                source->requirement_identity;
        }
    }
    builder->allocator.release(requirements, builder->allocator.context);
    return rc;
}

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

static yvex_transform_scope transform_recipe_scope(yvex_tensor_scope scope)
{
    if (scope == YVEX_TENSOR_SCOPE_MAIN_LAYER) return YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    if (scope == YVEX_TENSOR_SCOPE_DRAFT) return YVEX_TRANSFORM_SCOPE_AUXILIARY;
    return YVEX_TRANSFORM_SCOPE_GLOBAL;
}

static yvex_transform_subsystem transform_recipe_subsystem(
    yvex_tensor_collection collection)
{
    static const yvex_transform_subsystem subsystems[] = {
        YVEX_TRANSFORM_SUBSYSTEM_GLOBAL,
        YVEX_TRANSFORM_SUBSYSTEM_ATTENTION,
        YVEX_TRANSFORM_SUBSYSTEM_COMPRESSOR,
        YVEX_TRANSFORM_SUBSYSTEM_INDEXER,
        YVEX_TRANSFORM_SUBSYSTEM_NORMALIZATION,
        YVEX_TRANSFORM_SUBSYSTEM_RESIDUAL,
        YVEX_TRANSFORM_SUBSYSTEM_ROUTER,
        YVEX_TRANSFORM_SUBSYSTEM_ROUTED_EXPERT,
        YVEX_TRANSFORM_SUBSYSTEM_SHARED_EXPERT,
        YVEX_TRANSFORM_SUBSYSTEM_AUXILIARY,
        YVEX_TRANSFORM_SUBSYSTEM_SEQUENCE_MIXER};

    return collection < YVEX_TENSOR_COLLECTION_COUNT
               ? subsystems[collection] : YVEX_TRANSFORM_SUBSYSTEM_COUNT;
}

static yvex_transform_dtype transform_recipe_dtype(yvex_native_dtype dtype, int packed_fp4)
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

static unsigned int transform_recipe_physical_classes(yvex_transform_dtype dtype)
{
    switch (dtype) {
    case YVEX_TRANSFORM_DTYPE_F32: return YVEX_TRANSFORM_PHYSICAL_F32;
    case YVEX_TRANSFORM_DTYPE_F16:
        return YVEX_TRANSFORM_PHYSICAL_F16 | YVEX_TRANSFORM_PHYSICAL_F32;
    case YVEX_TRANSFORM_DTYPE_BF16:
        return YVEX_TRANSFORM_PHYSICAL_BF16 | YVEX_TRANSFORM_PHYSICAL_F32;
    case YVEX_TRANSFORM_DTYPE_I32: return YVEX_TRANSFORM_PHYSICAL_I32;
    default:
        return YVEX_TRANSFORM_PHYSICAL_F32 | YVEX_TRANSFORM_PHYSICAL_F16 |
               YVEX_TRANSFORM_PHYSICAL_BF16 | YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
    }
}

static int transform_recipe_resolve_source(
    yvex_transform_recipe_sink *sink, const char *source_name, yvex_tensor_role role,
    yvex_tensor_collection collection, yvex_tensor_scope scope, unsigned long long layer,
    unsigned long long auxiliary, unsigned long long expert,
    unsigned long long requirement_index, yvex_native_dtype source_dtype, int packed_fp4,
    const yvex_transform_shape *shape, yvex_transform_source_spec *resolved,
    yvex_transform_failure *failure, yvex_error *err)
{
    yvex_transform_source_requirement requirement = {0};

    if (!shape || !shape->rank)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_INVALID_SHAPE,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, 0u, err,
            "transform_recipe_source");
    requirement.source_name = source_name;
    requirement.requirement_index = requirement_index;
    requirement.source_dtype = source_dtype;
    requirement.value_dtype = transform_recipe_dtype(source_dtype, packed_fp4);
    requirement.shape = *shape;
    requirement.scope = transform_recipe_scope(scope);
    requirement.subsystem = transform_recipe_subsystem(collection);
    requirement.role_hint = role;
    requirement.layer_index = layer;
    requirement.auxiliary_index = auxiliary;
    requirement.expert_index = expert;
    requirement.required_uses = 1u;
    return yvex_transform_recipe_sink_resolve_source(
        sink, &requirement, resolved, failure, err);
}

int yvex_transform_recipe_add_terminal(
    yvex_transform_recipe_sink *sink, yvex_tensor_role role,
    yvex_tensor_collection collection, yvex_tensor_scope scope, unsigned long long layer,
    unsigned long long auxiliary, const yvex_transform_source_spec *sources,
    unsigned long long source_count, const yvex_transform_shape *shape,
    yvex_transform_dtype dtype, const yvex_transform_precision_constraint *precision,
    const yvex_transform_node_spec *operation, yvex_transform_failure *failure,
    yvex_error *err)
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
    recipe.terminal.logical_key.scope = transform_recipe_scope(scope);
    recipe.terminal.logical_key.subsystem = transform_recipe_subsystem(collection);
    recipe.terminal.logical_key.role = role;
    recipe.terminal.logical_key.layer_index = scope == YVEX_TENSOR_SCOPE_GLOBAL
                                                    ? YVEX_TRANSFORM_IR_NO_ID : layer;
    recipe.terminal.logical_key.auxiliary_index = scope == YVEX_TENSOR_SCOPE_DRAFT
                                                        ? auxiliary : YVEX_TRANSFORM_IR_NO_ID;
    recipe.operation = *operation;
    return yvex_transform_recipe_sink_add(sink, &recipe, failure, err);
}

int yvex_transform_recipe_add_direct(
    yvex_transform_recipe_sink *sink, const yvex_transform_direct_recipe *recipe,
    yvex_transform_failure *failure, yvex_error *err)
{
    yvex_transform_precision_constraint precision = {0};
    yvex_transform_node_spec node = {0};
    yvex_transform_source_spec source;
    yvex_transform_dtype output_dtype;
    int rc;

    if (!recipe)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, 0u, err,
            "transform_recipe_direct");
    rc = transform_recipe_resolve_source(
        sink, recipe->source_name, recipe->role, recipe->collection, recipe->scope,
        recipe->layer, recipe->auxiliary, recipe->expert, recipe->requirement_index,
        recipe->source_dtype, recipe->packed_fp4, &recipe->shape, &source, failure, err);
    if (rc != YVEX_OK) return rc;
    output_dtype = recipe->checked_cast ? YVEX_TRANSFORM_DTYPE_I32
                                        : transform_recipe_dtype(
                                              recipe->source_dtype, recipe->packed_fp4);
    precision.allowed_physical_classes = transform_recipe_physical_classes(output_dtype);
    if (recipe->checked_cast) {
        precision.flags = YVEX_TRANSFORM_PRECISION_LOSSLESS |
                          YVEX_TRANSFORM_PRECISION_RANGE_PROOF |
                          YVEX_TRANSFORM_PRECISION_INTEGER_ONLY;
        precision.range_proof_required = 1;
    } else {
        precision.flags = YVEX_TRANSFORM_PRECISION_EXACT;
    }
    node.kind = recipe->checked_cast ? YVEX_TRANSFORM_OP_CHECKED_CAST
                                     : YVEX_TRANSFORM_OP_IDENTITY;
    node.numeric = recipe->checked_cast ? YVEX_TRANSFORM_NUMERIC_RANGE_PROOF
                                        : YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    return yvex_transform_recipe_add_terminal(
        sink, recipe->role, recipe->collection, recipe->scope, recipe->layer,
        recipe->auxiliary, &source, 1u, &recipe->shape, output_dtype,
        &precision, &node, failure, err);
}

int yvex_transform_recipe_add_scale_pair(
    yvex_transform_recipe_sink *sink, const yvex_transform_scale_pair_recipe *recipe,
    yvex_transform_failure *failure, yvex_error *err)
{
    char weight[YVEX_TRANSFORM_IR_SOURCE_NAME_CAP];
    char scale[YVEX_TRANSFORM_IR_SOURCE_NAME_CAP];
    yvex_transform_precision_constraint precision = {0};
    yvex_transform_node_spec node = {0};
    yvex_transform_shape shape, scale_shape;
    yvex_transform_source_spec sources[2];
    int rc;

    if (!recipe || !recipe->base_name || !recipe->block_rows ||
        !recipe->block_columns || recipe->rows % recipe->block_rows != 0u ||
        recipe->columns % recipe->block_columns != 0u)
        return yvex_transform_fail(
            failure, YVEX_TRANSFORM_FAILURE_INVALID_SHAPE,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
            YVEX_TRANSFORM_IR_NO_ID, 1u, 0u, 0u, err,
            "transform_recipe_scale_pair");
    (void)snprintf(weight, sizeof(weight), "%s.weight", recipe->base_name);
    (void)snprintf(scale, sizeof(scale), "%s.scale", recipe->base_name);
    shape = (yvex_transform_shape){.rank = 2u,
                                   .dims = {recipe->rows, recipe->columns}};
    scale_shape = (yvex_transform_shape){
        .rank = 2u,
        .dims = {recipe->rows / recipe->block_rows,
                 recipe->columns / recipe->block_columns}};
    rc = transform_recipe_resolve_source(
        sink, weight, recipe->role, recipe->collection, recipe->scope, recipe->layer,
        recipe->auxiliary, YVEX_TRANSFORM_IR_NO_ID, recipe->requirement_index,
        YVEX_NATIVE_DTYPE_F8_E4M3, 0, &shape, &sources[0], failure, err);
    if (rc == YVEX_OK)
        rc = transform_recipe_resolve_source(
            sink, scale, recipe->role, recipe->collection, recipe->scope, recipe->layer,
            recipe->auxiliary, YVEX_TRANSFORM_IR_NO_ID, recipe->requirement_index + 1u,
            YVEX_NATIVE_DTYPE_F8_E8M0, 0, &scale_shape, &sources[1], failure, err);
    if (rc != YVEX_OK) return rc;
    precision.flags = YVEX_TRANSFORM_PRECISION_SCALE_PAIRED |
                      YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT |
                      YVEX_TRANSFORM_PRECISION_REFERENCE_COMPUTE;
    precision.allowed_physical_classes =
        YVEX_TRANSFORM_PHYSICAL_F32 | YVEX_TRANSFORM_PHYSICAL_F16 |
        YVEX_TRANSFORM_PHYSICAL_BF16 | YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
    precision.approximation_allowed = 1;
    precision.reference_compute_required = 1;
    node.kind = YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR;
    node.scale_block_rows = recipe->block_rows;
    node.scale_block_columns = recipe->block_columns;
    node.numeric = YVEX_TRANSFORM_NUMERIC_LOSSLESS;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    return yvex_transform_recipe_add_terminal(
        sink, recipe->role, recipe->collection, recipe->scope, recipe->layer,
        recipe->auxiliary, sources, 2u, &shape, YVEX_TRANSFORM_DTYPE_REAL,
        &precision, &node, failure, err);
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
    sink.source_snapshot = options ? options->source_snapshot : NULL;
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
        rc = transform_recipe_source_coverage_finalize(&sink, failure, err);
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
