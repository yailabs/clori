/*
 * Own the immutable artifact-lowering map, its allocation/rollback lifecycle, deterministic
 * indexes, and validation from sealed Transformation IR into container operands.
 */
#include <yvex/internal/artifact_lowering.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static const char *const lowering_transform_names[] = {
    "direct", "fp8-e4m3-e8m0-pair", "expert-mxfp4-repack", "i64-to-i32"
};

static const char *const lowering_failure_names[] = {
    "none", "invalid-argument", "architecture-incomplete",
    "coverage-row-mismatch", "missing-source", "duplicate-source",
    "source-dtype-mismatch", "expert-sequence-mismatch", "name-refused",
    "duplicate-name", "layout-refused", "metadata-refused",
    "accounting-mismatch", "arithmetic-overflow", "allocation-failure",
    "transform-ir-refused", "lowering-divergence", "mapping-identity-mismatch"
};

static const yvex_artifact_lowering_failure cleared_map_failure = {
    .layer_index = YVEX_ARTIFACT_LOWERING_NO_INDEX,
    .predictor_index = YVEX_ARTIFACT_LOWERING_NO_INDEX,
    .expert_index = YVEX_ARTIFACT_LOWERING_NO_INDEX
};


static yvex_tensor_scope map_scope(yvex_transform_scope scope);
static void lowering_close(yvex_artifact_lowering_map *map);
static const char *lowering_failure_name(yvex_artifact_lowering_failure_code code);

typedef struct {
    unsigned long long hash;
    unsigned long long value_plus_one;
} map_index_slot;

struct yvex_artifact_lowering_map {
    yvex_artifact_lowering_allocator allocator;
    yvex_artifact_lowering_descriptor *descriptors;
    yvex_artifact_lowering_contribution *contributions;
    map_index_slot *source_index;
    map_index_slot *emitted_index;
    map_index_slot *role_index;
    unsigned long long source_index_capacity;
    unsigned long long emitted_index_capacity;
    unsigned long long role_index_capacity;
    yvex_artifact_lowering_metadata metadata[YVEX_ARTIFACT_LOWERING_METADATA_CAP];
    yvex_artifact_lowering_summary summary;
};

typedef struct {
    yvex_artifact_lowering_map *map;
    const yvex_transform_ir *transform_ir;
    const yvex_artifact_lowering_policy *policy;
    yvex_artifact_lowering_failure *failure;
    yvex_error *err;
} map_builder;

typedef struct {
    yvex_artifact_lowering_transform transform;
    unsigned int qtype;
    int supported;
} map_transform_projection;

static const yvex_tensor_collection map_collections[YVEX_TRANSFORM_SUBSYSTEM_COUNT] = {
    [YVEX_TRANSFORM_SUBSYSTEM_GLOBAL] = YVEX_TENSOR_COLLECTION_GLOBAL,
    [YVEX_TRANSFORM_SUBSYSTEM_ATTENTION] = YVEX_TENSOR_COLLECTION_ATTENTION,
    [YVEX_TRANSFORM_SUBSYSTEM_COMPRESSOR] = YVEX_TENSOR_COLLECTION_COMPRESSOR,
    [YVEX_TRANSFORM_SUBSYSTEM_INDEXER] = YVEX_TENSOR_COLLECTION_INDEXER,
    [YVEX_TRANSFORM_SUBSYSTEM_NORMALIZATION] = YVEX_TENSOR_COLLECTION_NORM,
    [YVEX_TRANSFORM_SUBSYSTEM_RESIDUAL] = YVEX_TENSOR_COLLECTION_MHC,
    [YVEX_TRANSFORM_SUBSYSTEM_ROUTER] = YVEX_TENSOR_COLLECTION_ROUTER,
    [YVEX_TRANSFORM_SUBSYSTEM_ROUTED_EXPERT] = YVEX_TENSOR_COLLECTION_ROUTED_EXPERT,
    [YVEX_TRANSFORM_SUBSYSTEM_SHARED_EXPERT] = YVEX_TENSOR_COLLECTION_SHARED_EXPERT,
    [YVEX_TRANSFORM_SUBSYSTEM_OUTPUT] = YVEX_TENSOR_COLLECTION_GLOBAL,
    [YVEX_TRANSFORM_SUBSYSTEM_AUXILIARY] = YVEX_TENSOR_COLLECTION_AUXILIARY,
    [YVEX_TRANSFORM_SUBSYSTEM_SEQUENCE_MIXER] = YVEX_TENSOR_COLLECTION_SEQUENCE_MIXER
};

static const yvex_tensor_scope map_scopes[] = {
    YVEX_TENSOR_SCOPE_GLOBAL,
    YVEX_TENSOR_SCOPE_MAIN_LAYER,
    YVEX_TENSOR_SCOPE_DRAFT
};

enum { MAP_SCOPE_COUNT = 3, MAP_SUBSYSTEM_COUNT = YVEX_TRANSFORM_SUBSYSTEM_COUNT };

static const yvex_artifact_lowering_contribution_kind map_contribution_kinds[][2] = {
    {YVEX_ARTIFACT_LOWERING_CONTRIBUTION_PRIMARY,
     YVEX_ARTIFACT_LOWERING_CONTRIBUTION_PRIMARY},
    {YVEX_ARTIFACT_LOWERING_CONTRIBUTION_PRIMARY,
     YVEX_ARTIFACT_LOWERING_CONTRIBUTION_SCALE},
    {YVEX_ARTIFACT_LOWERING_CONTRIBUTION_EXPERT_WEIGHT,
     YVEX_ARTIFACT_LOWERING_CONTRIBUTION_EXPERT_SCALE},
    {YVEX_ARTIFACT_LOWERING_CONTRIBUTION_ROUTING_TABLE,
     YVEX_ARTIFACT_LOWERING_CONTRIBUTION_ROUTING_TABLE}
};

static const map_transform_projection map_transforms[YVEX_TRANSFORM_OP_COUNT] = {
    [YVEX_TRANSFORM_OP_IDENTITY] = {
        YVEX_ARTIFACT_LOWERING_TRANSFORM_DIRECT, YVEX_GGUF_NO_FORCED_QTYPE, 1},
    [YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR] = {
        YVEX_ARTIFACT_LOWERING_TRANSFORM_FP8_E4M3_E8M0, YVEX_GGUF_NO_FORCED_QTYPE, 1},
    [YVEX_TRANSFORM_OP_CHECKED_CAST] = {
        YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32, 26u, 1},
    [YVEX_TRANSFORM_OP_EXPERT_AGGREGATE] = {
        YVEX_ARTIFACT_LOWERING_TRANSFORM_EXPERT_MXFP4, 39u, 1}
};


static void *map_default_allocate(size_t size, void *context)
{
    (void)context;
    return malloc(size);
}

static void map_default_release(void *allocation, void *context)
{
    (void)context;
    free(allocation);
}

static unsigned long long map_hash_string(const char *text)
{
    return yvex_core_hash_mix_bytes(1469598103934665603ull, text, strlen(text) + 1u);
}

static void map_failure_clear(yvex_artifact_lowering_failure *failure)
{
    if (!failure) return;
    *failure = cleared_map_failure;
}

static int map_reject(map_builder *builder,
                      yvex_artifact_lowering_failure_code code,
                      yvex_tensor_role role,
                      yvex_tensor_scope scope,
                      unsigned long long layer,
                      unsigned long long predictor,
                      unsigned long long expert,
                      const char *source_name,
                      const char *emitted_name,
                      unsigned long long expected,
                      unsigned long long actual)
{
    yvex_status status = code == YVEX_ARTIFACT_LOWERING_FAILURE_ALLOCATION
        ? YVEX_ERR_NOMEM
        : (code == YVEX_ARTIFACT_LOWERING_FAILURE_INVALID_ARGUMENT
            ? YVEX_ERR_INVALID_ARG : YVEX_ERR_FORMAT);
    yvex_artifact_lowering_failure *failure =
        builder ? builder->failure : NULL;

    if (failure) {
        map_failure_clear(failure);
        failure->code = code;
        failure->role = role;
        failure->scope = scope;
        failure->layer_index = layer;
        failure->predictor_index = predictor;
        failure->expert_index = expert;
        failure->expected = expected;
        failure->actual = actual;
        yvex_core_text_copy(failure->source_name, sizeof(failure->source_name), source_name ? source_name : "");
        yvex_core_text_copy(failure->emitted_name, sizeof(failure->emitted_name), emitted_name ? emitted_name : "");
    }
    yvex_error_setf(builder ? builder->err : NULL, status,
                    "artifact_lowering",
                    "%s role=%s source=%s emitted=%s layer=%llu expert=%llu expected=%llu actual=%llu",
                    lowering_failure_name(code),
                    yvex_tensor_role_name(role),
                    source_name ? source_name : "none",
                    emitted_name ? emitted_name : "none", layer, expert,
                    expected, actual);
    return status;
}

static int map_reject_global(map_builder *builder,
                             yvex_artifact_lowering_failure_code code,
                             const char *subject,
                             unsigned long long expected,
                             unsigned long long actual)
{
    return map_reject(builder, code, YVEX_TENSOR_ROLE_UNKNOWN,
                      YVEX_TENSOR_SCOPE_GLOBAL, YVEX_ARTIFACT_LOWERING_NO_INDEX,
                      YVEX_ARTIFACT_LOWERING_NO_INDEX, YVEX_ARTIFACT_LOWERING_NO_INDEX,
                      subject, NULL, expected, actual);
}

static int map_reject_descriptor(map_builder *builder,
                                 yvex_artifact_lowering_failure_code code,
                                 const yvex_artifact_lowering_descriptor *descriptor,
                                 unsigned long long expert,
                                 const char *source,
                                 const char *emitted,
                                 unsigned long long expected,
                                 unsigned long long actual)
{
    return map_reject(builder, code, descriptor->role, descriptor->scope,
                      descriptor->layer_index, descriptor->predictor_index,
                      expert, source, emitted, expected, actual);
}

static int map_reject_key(map_builder *builder,
                          yvex_artifact_lowering_failure_code code,
                          const yvex_transform_logical_key *key,
                          int include_location,
                          unsigned long long expected,
                          unsigned long long actual)
{
    return map_reject(builder, code, key ? key->role : YVEX_TENSOR_ROLE_UNKNOWN,
                      key ? map_scope(key->scope) : YVEX_TENSOR_SCOPE_GLOBAL,
                      key && include_location ? key->layer_index
                                              : YVEX_ARTIFACT_LOWERING_NO_INDEX,
                      key && include_location ? key->auxiliary_index
                                              : YVEX_ARTIFACT_LOWERING_NO_INDEX,
                      YVEX_ARTIFACT_LOWERING_NO_INDEX, NULL, NULL, expected, actual);
}

static void *map_allocate_zero(yvex_artifact_lowering_map *map, size_t size)
{
    void *allocation = map->allocator.allocate(size, map->allocator.context);
    if (allocation) memset(allocation, 0, size);
    return allocation;
}

static int map_index_insert(map_index_slot *slots,
                            unsigned long long capacity,
                            unsigned long long hash,
                            unsigned long long value)
{
    unsigned long long slot;
    unsigned long long probe;

    if (!slots || !capacity || (capacity & (capacity - 1u)) != 0u) return 0;
    slot = hash & (capacity - 1u);
    for (probe = 0u; probe < capacity; ++probe) {
        if (!slots[slot].value_plus_one) {
            slots[slot].hash = hash;
            slots[slot].value_plus_one = value + 1u;
            return 1;
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    return 0;
}

static int map_unique_index_equal(const yvex_artifact_lowering_map *map,
                                  int emitted,
                                  unsigned long long left,
                                  unsigned long long right)
{
    const yvex_artifact_lowering_descriptor *candidate = &map->descriptors[right];
    const yvex_artifact_lowering_descriptor *current = &map->descriptors[left];

    return emitted ? strcmp(current->emitted_name, candidate->emitted_name) == 0
                   : current->role == candidate->role &&
                         current->scope == candidate->scope &&
                         current->layer_index == candidate->layer_index &&
                         current->predictor_index == candidate->predictor_index;
}

static int map_unique_index_insert(yvex_artifact_lowering_map *map,
                                   int emitted,
                                   unsigned long long hash,
                                   unsigned long long value)
{
    map_index_slot *slots = emitted ? map->emitted_index : map->role_index;
    unsigned long long capacity = emitted ? map->emitted_index_capacity
                                          : map->role_index_capacity;
    unsigned long long slot = hash & (capacity - 1u);
    unsigned long long probe;

    for (probe = 0u; probe < capacity; ++probe) {
        map_index_slot *entry = &slots[slot];
        if (!entry->value_plus_one) {
            entry->hash = hash;
            entry->value_plus_one = value + 1u;
            return 1;
        }
        if (entry->hash == hash &&
            map_unique_index_equal(
                map, emitted, entry->value_plus_one - 1u, value)) return 0;
        slot = (slot + 1u) & (capacity - 1u);
    }
    return 0;
}

static int map_emitted_index_insert(yvex_artifact_lowering_map *map,
                                    unsigned long long hash,
                                    unsigned long long value)
{
    return map_unique_index_insert(map, 1, hash, value);
}

static int map_role_index_insert(yvex_artifact_lowering_map *map,
                                 unsigned long long hash,
                                 unsigned long long value)
{
    return map_unique_index_insert(map, 0, hash, value);
}

static yvex_tensor_scope map_scope(yvex_transform_scope scope)
{
    return (unsigned int)scope < MAP_SCOPE_COUNT ? map_scopes[(unsigned int)scope]
                                                  : YVEX_TENSOR_SCOPE_GLOBAL;
}

static yvex_tensor_collection map_collection(
    yvex_transform_subsystem subsystem)
{
    return (unsigned int)subsystem < MAP_SUBSYSTEM_COUNT
        ? map_collections[(unsigned int)subsystem] : YVEX_TENSOR_COLLECTION_COUNT;
}

static int map_transform(const yvex_transform_node *node,
                         yvex_artifact_lowering_transform *transform,
                         unsigned int *qtype)
{
    const map_transform_projection *projection;

    if (!node || !transform || !qtype) return 0;
    if ((unsigned int)node->kind >= YVEX_TRANSFORM_OP_COUNT ||
        !map_transforms[(unsigned int)node->kind].supported) return 0;
    projection = &map_transforms[(unsigned int)node->kind];
    *transform = projection->transform;
    *qtype = projection->qtype;
    return 1;
}

static yvex_artifact_lowering_contribution_kind map_contribution_kind(
    yvex_artifact_lowering_transform transform,
    unsigned long long input)
{
    unsigned int secondary = transform == YVEX_ARTIFACT_LOWERING_TRANSFORM_EXPERT_MXFP4
        ? (unsigned int)(input & 1u) : input != 0u;

    return (unsigned int)transform <= YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32
        ? map_contribution_kinds[(unsigned int)transform][secondary]
        : YVEX_ARTIFACT_LOWERING_CONTRIBUTION_PRIMARY;
}

static int map_descriptor_begin(map_builder *builder,
                                const yvex_transform_value *terminal,
                                const yvex_transform_node *node,
                                unsigned long long descriptor_index)
{
    yvex_artifact_lowering_map *map = builder->map;
    yvex_artifact_lowering_descriptor *descriptor =
        &map->descriptors[descriptor_index];
    yvex_gguf_name_provenance provenance;
    yvex_tensor_scope scope = map_scope(terminal->logical_key.scope);
    yvex_tensor_collection collection =
        map_collection(terminal->logical_key.subsystem);
    const char *reason = NULL;
    unsigned long long role_hash = 1469598103934665603ull;
    unsigned int qtype;
    unsigned int dimension;

    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->role = terminal->logical_key.role;
    descriptor->collection = collection;
    descriptor->scope = scope;
    descriptor->layer_index = terminal->logical_key.layer_index;
    descriptor->predictor_index = terminal->logical_key.auxiliary_index;
    descriptor->expert_count = node->expert_count;
    if (collection >= YVEX_TENSOR_COLLECTION_COUNT ||
        !map_transform(node, &descriptor->transform, &qtype) ||
        terminal->shape.rank > YVEX_TENSOR_MAX_DIMS) {
        return map_reject_descriptor(
            builder, YVEX_ARTIFACT_LOWERING_FAILURE_LOWERING_DIVERGENCE,
            descriptor, YVEX_ARTIFACT_LOWERING_NO_INDEX, NULL, NULL, 1u, 0u);
    }
    descriptor->forced_qtype = qtype;
    descriptor->logical_rank = terminal->shape.rank;
    descriptor->contribution_offset = map->summary.source_contribution_count;
    for (dimension = 0u; dimension < terminal->shape.rank; ++dimension) {
        unsigned int source_axis = terminal->shape.rank - dimension - 1u;
        descriptor->logical_dims[dimension] =
            terminal->shape.dims[source_axis];
        descriptor->source_axis_for_logical[dimension] = source_axis;
    }
    if (node->kind == YVEX_TRANSFORM_OP_EXPERT_AGGREGATE) {
        descriptor->source_axis_for_logical[0] = 1u;
        descriptor->source_axis_for_logical[1] = 0u;
        descriptor->source_axis_for_logical[2] =
            YVEX_ARTIFACT_LOWERING_AGGREGATED_AXIS;
    }
    if (!yvex_gguf_name_map_resolve(
            descriptor->role, scope == YVEX_TENSOR_SCOPE_DRAFT,
            descriptor->layer_index, descriptor->predictor_index,
            descriptor->emitted_name, sizeof(descriptor->emitted_name),
            &provenance, &reason)) {
        return map_reject_descriptor(
            builder, YVEX_ARTIFACT_LOWERING_FAILURE_NAME, descriptor,
            YVEX_ARTIFACT_LOWERING_NO_INDEX, NULL, reason, 1u, 0u);
    }
    descriptor->name_provenance = provenance;
    if (!yvex_gguf_layout_map_shape_supported(
            descriptor->role, qtype, descriptor->logical_rank,
            descriptor->logical_dims, &reason)) {
        return map_reject_descriptor(
            builder, YVEX_ARTIFACT_LOWERING_FAILURE_LAYOUT, descriptor,
            YVEX_ARTIFACT_LOWERING_NO_INDEX, NULL, descriptor->emitted_name, 1u, 0u);
    }
    if (!map_emitted_index_insert(
            map, map_hash_string(descriptor->emitted_name), descriptor_index)) {
        return map_reject_descriptor(
            builder, YVEX_ARTIFACT_LOWERING_FAILURE_DUPLICATE_NAME, descriptor,
            YVEX_ARTIFACT_LOWERING_NO_INDEX, NULL, descriptor->emitted_name, 1u, 2u);
    }
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->role);
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->scope);
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->layer_index);
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->predictor_index);
    if (!map_role_index_insert(map, role_hash, descriptor_index)) {
        return map_reject_descriptor(
            builder, YVEX_ARTIFACT_LOWERING_FAILURE_DUPLICATE_NAME, descriptor,
            YVEX_ARTIFACT_LOWERING_NO_INDEX, NULL, descriptor->emitted_name, 1u, 2u);
    }
    descriptor->identity = map_hash_string(descriptor->emitted_name);
    descriptor->identity = yvex_core_hash_mix_u64(descriptor->identity,
                                        descriptor->transform);
    descriptor->identity = yvex_core_hash_mix_u64(descriptor->identity, qtype);
    for (dimension = 0u; dimension < descriptor->logical_rank; ++dimension)
        descriptor->identity = yvex_core_hash_mix_u64(
            descriptor->identity, descriptor->logical_dims[dimension]);
    map->summary.descriptor_count++;
    map->summary.collection_counts[collection]++;
    if (scope == YVEX_TENSOR_SCOPE_DRAFT)
        map->summary.draft_descriptor_count++;
    else
        map->summary.trunk_descriptor_count++;
    if (provenance == YVEX_GGUF_NAME_PINNED_STANDARD)
        map->summary.pinned_standard_count++;
    else if (provenance == YVEX_GGUF_NAME_SEMANTIC_STANDARD)
        map->summary.semantic_standard_count++;
    else
        map->summary.extension_count++;
    return YVEX_OK;
}

static int map_descriptor_add_source(
    map_builder *builder,
    const yvex_transform_node *node,
    unsigned long long descriptor_index,
    unsigned long long input_index)
{
    yvex_artifact_lowering_map *map = builder->map;
    yvex_artifact_lowering_descriptor *descriptor =
        &map->descriptors[descriptor_index];
    const yvex_transform_value *value = yvex_transform_ir_node_input_at(
        builder->transform_ir, node, input_index);
    const yvex_transform_source_value *source;
    yvex_artifact_lowering_contribution *contribution;
    unsigned long long index = map->summary.source_contribution_count;
    unsigned int dimension;

    if (!value || value->kind != YVEX_TRANSFORM_VALUE_SOURCE)
        return map_reject_descriptor(
            builder, YVEX_ARTIFACT_LOWERING_FAILURE_LOWERING_DIVERGENCE,
            descriptor, YVEX_ARTIFACT_LOWERING_NO_INDEX, NULL,
            descriptor->emitted_name, 1u, 0u);
    source = yvex_transform_ir_source_at(
        builder->transform_ir, value->source_index);
    if (!source || index >= builder->policy->source_contribution_count ||
        source->requirement_index >= builder->policy->source_contribution_count ||
        source->shape.rank > YVEX_TENSOR_MAX_DIMS ||
        source->role_hint != descriptor->role ||
        map_scope(source->scope) != descriptor->scope ||
        map_collection(source->subsystem) != descriptor->collection) {
        return map_reject_descriptor(
            builder, YVEX_ARTIFACT_LOWERING_FAILURE_COVERAGE_ROW, descriptor,
            source ? source->expert_index : YVEX_ARTIFACT_LOWERING_NO_INDEX,
            source ? source->source_name : NULL, descriptor->emitted_name,
            1u, 0u);
    }
    contribution = &map->contributions[index];
    yvex_core_text_copy(contribution->source_name, sizeof(contribution->source_name), source->source_name);
    contribution->source_dtype = source->source_dtype;
    contribution->source_rank = source->shape.rank;
    for (dimension = 0u; dimension < source->shape.rank; ++dimension)
        contribution->source_dims[dimension] = source->shape.dims[dimension];
    contribution->kind = map_contribution_kind(descriptor->transform,
                                               input_index);
    contribution->source_row_index = source->requirement_index;
    contribution->descriptor_index = descriptor_index;
    contribution->expert_index = source->expert_index;
    if (!map_index_insert(map->source_index, map->source_index_capacity,
                          map_hash_string(source->source_name), index)) {
        return map_reject_descriptor(
            builder, YVEX_ARTIFACT_LOWERING_FAILURE_DUPLICATE_SOURCE,
            descriptor, source->expert_index,
            source->source_name, descriptor->emitted_name, 1u, 2u);
    }
    descriptor->contribution_count++;
    descriptor->identity = yvex_core_hash_mix_bytes(
        descriptor->identity, source->source_name,
        strlen(source->source_name) + 1u);
    map->summary.source_contribution_count++;
    return YVEX_OK;
}

static int map_build_descriptors(map_builder *builder)
{
    const yvex_transform_ir_summary *summary =
        yvex_transform_ir_summary_get(builder->transform_ir);
    unsigned long long ordinal;

    if (!summary || !summary->complete ||
        summary->state != YVEX_TRANSFORM_IR_STATE_SEALED ||
        summary->source_value_count != builder->policy->source_contribution_count ||
        summary->terminal_count != builder->policy->descriptor_count ||
        summary->edge_count != builder->policy->source_contribution_count ||
        summary->payload_bytes_read != 0u) {
        return map_reject_global(
            builder, YVEX_ARTIFACT_LOWERING_FAILURE_TRANSFORM_IR, NULL,
            builder->policy->descriptor_count,
            summary ? summary->terminal_count : 0u);
    }
    for (ordinal = 0u; ordinal < summary->terminal_count; ++ordinal) {
        const yvex_transform_value *terminal =
            yvex_transform_ir_terminal_at(builder->transform_ir, ordinal);
        const yvex_transform_node *node;
        unsigned long long input;
        int rc;

        if (!terminal || terminal->canonical_ordinal != ordinal ||
            terminal->producer_node_id >= summary->node_count) {
            return map_reject_key(
                builder, YVEX_ARTIFACT_LOWERING_FAILURE_TRANSFORM_IR,
                terminal ? &terminal->logical_key : NULL, 0, ordinal,
                terminal ? terminal->canonical_ordinal : ULLONG_MAX);
        }
        node = yvex_transform_ir_node_at(
            builder->transform_ir, terminal->producer_node_id);
        if (!node || node->output_value_id != terminal->id) {
            return map_reject_key(
                builder, YVEX_ARTIFACT_LOWERING_FAILURE_TRANSFORM_IR,
                &terminal->logical_key, 1, terminal->id,
                node ? node->output_value_id : ULLONG_MAX);
        }
        rc = map_descriptor_begin(builder, terminal, node, ordinal);
        if (rc != YVEX_OK) return rc;
        for (input = 0u; input < node->input_count; ++input) {
            rc = map_descriptor_add_source(builder, node, ordinal, input);
            if (rc != YVEX_OK) return rc;
        }
    }
    return YVEX_OK;
}

static int map_metadata_begin(map_builder *builder,
                              const char *key,
                              yvex_artifact_lowering_metadata **out)
{
    yvex_artifact_lowering_map *map = builder->map;
    unsigned long long index;

    if (!key || map->summary.metadata_count >= YVEX_ARTIFACT_LOWERING_METADATA_CAP)
        return map_reject_global(builder, YVEX_ARTIFACT_LOWERING_FAILURE_METADATA,
                                 key, YVEX_ARTIFACT_LOWERING_METADATA_CAP,
                                 map->summary.metadata_count + 1u);
    for (index = 0u; index < map->summary.metadata_count; ++index)
        if (strcmp(map->metadata[index].key, key) == 0)
            return map_reject_global(builder,
                                     YVEX_ARTIFACT_LOWERING_FAILURE_METADATA,
                                     key, 1u, 2u);
    *out = &map->metadata[map->summary.metadata_count++];
    yvex_core_text_copy((*out)->key, sizeof((*out)->key), key);
    return YVEX_OK;
}


static int map_build_metadata(map_builder *builder)
{
    unsigned long long index;

    if (!builder->policy->metadata ||
        builder->policy->metadata_count > YVEX_ARTIFACT_LOWERING_METADATA_CAP)
        return map_reject_global(builder, YVEX_ARTIFACT_LOWERING_FAILURE_METADATA, NULL,
                                 YVEX_ARTIFACT_LOWERING_METADATA_CAP,
                                 builder->policy->metadata_count);
    for (index = 0u; index < builder->policy->metadata_count; ++index) {
        const yvex_artifact_lowering_metadata *source = &builder->policy->metadata[index];
        yvex_artifact_lowering_metadata *entry = NULL;
        int rc;

        if (source->type > YVEX_ARTIFACT_LOWERING_METADATA_F64_ARRAY ||
            source->array_count > YVEX_ARTIFACT_LOWERING_METADATA_CAP)
            return map_reject_global(builder, YVEX_ARTIFACT_LOWERING_FAILURE_METADATA,
                                     source->key, YVEX_ARTIFACT_LOWERING_METADATA_CAP,
                                     source->array_count);
        rc = map_metadata_begin(builder, source->key, &entry);
        if (rc != YVEX_OK) return rc;
        *entry = *source;
    }
    return YVEX_OK;
}

static int map_finalize(map_builder *builder)
{
    yvex_artifact_lowering_map *map = builder->map;
    unsigned long long trunk[YVEX_TENSOR_COLLECTION_COUNT] = {0};
    unsigned long long identity = 1469598103934665603ull;
    unsigned long long index;

    if (map->summary.source_contribution_count != builder->policy->source_contribution_count ||
        map->summary.descriptor_count != builder->policy->descriptor_count ||
        map->summary.trunk_descriptor_count != builder->policy->trunk_descriptor_count ||
        map->summary.draft_descriptor_count != builder->policy->draft_descriptor_count ||
        map->summary.pinned_standard_count != builder->policy->pinned_standard_count ||
        map->summary.extension_count != builder->policy->extension_count)
        return map_reject_global(builder, YVEX_ARTIFACT_LOWERING_FAILURE_ACCOUNTING, NULL,
                                 builder->policy->descriptor_count,
                                 map->summary.descriptor_count);
    for (index = 0u; index < map->summary.descriptor_count; ++index) {
        const yvex_artifact_lowering_descriptor *descriptor =
            &map->descriptors[index];
        if (descriptor->scope != YVEX_TENSOR_SCOPE_DRAFT)
            trunk[descriptor->collection]++;
        identity = yvex_core_hash_mix_u64(identity, descriptor->identity);
    }
    for (index = 0u; index < YVEX_TENSOR_COLLECTION_COUNT; ++index)
        if (trunk[index] != builder->policy->trunk_collection_counts[index])
            return map_reject(
                builder, YVEX_ARTIFACT_LOWERING_FAILURE_ACCOUNTING,
                YVEX_TENSOR_ROLE_UNKNOWN, YVEX_TENSOR_SCOPE_MAIN_LAYER,
                YVEX_ARTIFACT_LOWERING_NO_INDEX, YVEX_ARTIFACT_LOWERING_NO_INDEX,
                YVEX_ARTIFACT_LOWERING_NO_INDEX, NULL, NULL,
                builder->policy->trunk_collection_counts[index], trunk[index]);
    identity = yvex_core_hash_mix_u64(identity, map->summary.source_identity);
    identity = yvex_core_hash_mix_u64(identity, map->summary.coverage_identity);
    map->summary.mapping_identity = identity;
    map->summary.complete = 1;
    return YVEX_OK;
}

static int lowering_build_with_allocator(
    yvex_artifact_lowering_map **out,
    const yvex_transform_ir *transform_ir,
    const yvex_artifact_lowering_policy *policy,
    const yvex_artifact_lowering_allocator *allocator,
    yvex_artifact_lowering_failure *failure,
    yvex_error *err)
{
    const yvex_transform_ir_summary *transform_summary;
    yvex_artifact_lowering_map *map;
    map_builder builder;
    size_t bytes;
    int rc;

    if (out) *out = NULL;
    map_failure_clear(failure);
    yvex_error_clear(err);
    memset(&builder, 0, sizeof(builder));
    builder.failure = failure;
    builder.err = err;
    if (!out || !transform_ir || !policy || !allocator ||
        !allocator->allocate || !allocator->release) {
        return map_reject_global(
            &builder, YVEX_ARTIFACT_LOWERING_FAILURE_INVALID_ARGUMENT,
            NULL, 1u, 0u);
    }
    transform_summary = yvex_transform_ir_summary_get(transform_ir);
    if (policy->schema_version != YVEX_ARTIFACT_LOWERING_POLICY_SCHEMA_V1 ||
        !policy->source_contribution_count || !policy->descriptor_count ||
        policy->trunk_descriptor_count + policy->draft_descriptor_count !=
            policy->descriptor_count ||
        policy->pinned_standard_count + policy->extension_count > policy->descriptor_count) {
        return map_reject_global(
            &builder, YVEX_ARTIFACT_LOWERING_FAILURE_ARCHITECTURE, NULL,
            YVEX_ARTIFACT_LOWERING_POLICY_SCHEMA_V1, policy->schema_version);
    }
    if (!transform_summary || !transform_summary->complete ||
        transform_summary->source_value_count != policy->source_contribution_count ||
        transform_summary->terminal_count != policy->descriptor_count) {
        return map_reject_global(
            &builder, YVEX_ARTIFACT_LOWERING_FAILURE_TRANSFORM_IR, NULL,
            policy->descriptor_count,
            transform_summary ? transform_summary->terminal_count : 0u);
    }
    map = (yvex_artifact_lowering_map *)allocator->allocate(
        sizeof(*map), allocator->context);
    if (!map)
        return map_reject_global(
            &builder, YVEX_ARTIFACT_LOWERING_FAILURE_ALLOCATION,
            "map", sizeof(*map), 0u);
    memset(map, 0, sizeof(*map));
    map->allocator = *allocator;
    builder.map = map;
    if (!yvex_core_power_of_two_capacity(policy->source_contribution_count, 8ull,
                                         1ull, 2ull, &map->source_index_capacity) ||
        !yvex_core_power_of_two_capacity(policy->descriptor_count, 8ull,
                                         1ull, 2ull, &map->emitted_index_capacity) ||
        !yvex_core_power_of_two_capacity(policy->descriptor_count, 8ull,
                                         1ull, 2ull, &map->role_index_capacity))
    {
        rc = map_reject_global(
            &builder, YVEX_ARTIFACT_LOWERING_FAILURE_ARITHMETIC_OVERFLOW,
            "mapping-index", 1u, 0u);
        lowering_close(map);
        return rc;
    }
    map->descriptors = (yvex_artifact_lowering_descriptor *)map_allocate_zero(
        map, (size_t)policy->descriptor_count * sizeof(*map->descriptors));
    map->contributions = (yvex_artifact_lowering_contribution *)map_allocate_zero(
        map, (size_t)policy->source_contribution_count * sizeof(*map->contributions));
    bytes = (size_t)map->source_index_capacity * sizeof(*map->source_index);
    map->source_index = (map_index_slot *)map_allocate_zero(map, bytes);
    bytes = (size_t)map->emitted_index_capacity * sizeof(*map->emitted_index);
    map->emitted_index = (map_index_slot *)map_allocate_zero(map, bytes);
    bytes = (size_t)map->role_index_capacity * sizeof(*map->role_index);
    map->role_index = (map_index_slot *)map_allocate_zero(map, bytes);
    builder.transform_ir = transform_ir;
    builder.policy = policy;
    if (!map->descriptors || !map->contributions || !map->source_index ||
        !map->emitted_index || !map->role_index) {
        rc = map_reject_global(
            &builder, YVEX_ARTIFACT_LOWERING_FAILURE_ALLOCATION,
            "mapping-tables", 1u, 0u);
        lowering_close(map);
        return rc;
    }
    map->summary.header_scan_count = transform_summary->header_scan_count;
    map->summary.payload_bytes_read = transform_summary->payload_bytes_read;
    map->summary.source_identity = transform_summary->source_snapshot_identity;
    map->summary.coverage_identity = transform_summary->coverage_identity;
    rc = map_build_descriptors(&builder);
    if (rc == YVEX_OK) rc = map_build_metadata(&builder);
    if (rc == YVEX_OK) rc = map_finalize(&builder);
    if (rc != YVEX_OK) {
        lowering_close(map);
        return rc;
    }
    *out = map;
    yvex_error_clear(err);
    return YVEX_OK;
}


static int lowering_build(
    yvex_artifact_lowering_map **out,
    const yvex_transform_ir *transform_ir,
    const yvex_artifact_lowering_policy *policy,
    yvex_artifact_lowering_failure *failure,
    yvex_error *err)
{
    yvex_artifact_lowering_allocator allocator;

    allocator.allocate = map_default_allocate;
    allocator.release = map_default_release;
    allocator.context = NULL;
    return lowering_build_with_allocator(
        out, transform_ir, policy, &allocator, failure, err);
}

static void lowering_close(yvex_artifact_lowering_map *map)
{
    yvex_artifact_lowering_allocator allocator;
    void *allocations[5];
    unsigned int index;

    if (!map) return;
    allocator = map->allocator;
    allocations[0] = map->role_index;
    allocations[1] = map->emitted_index;
    allocations[2] = map->source_index;
    allocations[3] = map->contributions;
    allocations[4] = map->descriptors;
    for (index = 0u; index < sizeof(allocations) / sizeof(allocations[0]); ++index)
        if (allocations[index]) allocator.release(allocations[index], allocator.context);
    allocator.release(map, allocator.context);
}

static const yvex_artifact_lowering_summary *lowering_summary(
    const yvex_artifact_lowering_map *map)
{
    return map ? &map->summary : NULL;
}

static const yvex_artifact_lowering_descriptor *lowering_descriptor_at(
    const yvex_artifact_lowering_map *map,
    unsigned long long index)
{
    return map && index < map->summary.descriptor_count ? &map->descriptors[index] : NULL;
}

static const yvex_artifact_lowering_contribution *
lowering_contribution_at(
    const yvex_artifact_lowering_map *map,
    unsigned long long index)
{
    return map && index < map->summary.source_contribution_count ? &map->contributions[index] : NULL;
}

static const yvex_artifact_lowering_descriptor *map_find_name(
    const yvex_artifact_lowering_map *map,
    const char *name,
    int emitted)
{
    const map_index_slot *slots;
    unsigned long long capacity;
    unsigned long long hash;
    unsigned long long slot;
    unsigned long long probe;

    if (!map || !name) return NULL;
    slots = emitted ? map->emitted_index : map->source_index;
    capacity = emitted ? map->emitted_index_capacity
                       : map->source_index_capacity;
    hash = map_hash_string(name);
    slot = hash & (capacity - 1u);
    for (probe = 0u; probe < capacity && slots[slot].value_plus_one; ++probe) {
        if (slots[slot].hash == hash) {
            unsigned long long value = slots[slot].value_plus_one - 1u;
            if (emitted) {
                if (strcmp(map->descriptors[value].emitted_name, name) == 0)
                    return &map->descriptors[value];
            } else if (strcmp(map->contributions[value].source_name,
                              name) == 0) {
                return &map->descriptors[
                    map->contributions[value].descriptor_index];
            }
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    return NULL;
}

static const yvex_artifact_lowering_descriptor *lowering_find_source(
    const yvex_artifact_lowering_map *map,
    const char *source_name)
{
    return map_find_name(map, source_name, 0);
}

static const yvex_artifact_lowering_descriptor *lowering_find_emitted(
    const yvex_artifact_lowering_map *map,
    const char *emitted_name)
{
    return map_find_name(map, emitted_name, 1);
}

static const yvex_artifact_lowering_descriptor *lowering_find_role(
    const yvex_artifact_lowering_map *map,
    yvex_tensor_role role,
    yvex_tensor_scope scope,
    unsigned long long layer_index,
    unsigned long long predictor_index)
{
    unsigned long long hash = 1469598103934665603ull;
    unsigned long long slot;
    unsigned long long probe;
    if (!map) return NULL;
    hash = yvex_core_hash_mix_u64(hash, role);
    hash = yvex_core_hash_mix_u64(hash, scope);
    hash = yvex_core_hash_mix_u64(hash, layer_index);
    hash = yvex_core_hash_mix_u64(hash, predictor_index);
    slot = hash & (map->role_index_capacity - 1u);
    for (probe = 0u; probe < map->role_index_capacity &&
         map->role_index[slot].value_plus_one; ++probe) {
        if (map->role_index[slot].hash == hash) {
            const yvex_artifact_lowering_descriptor *descriptor =
                &map->descriptors[map->role_index[slot].value_plus_one - 1u];
            if (descriptor->role == role && descriptor->scope == scope &&
                descriptor->layer_index == layer_index &&
                descriptor->predictor_index == predictor_index)
                return descriptor;
        }
        slot = (slot + 1u) & (map->role_index_capacity - 1u);
    }
    return NULL;
}

static const yvex_artifact_lowering_metadata *lowering_metadata_at(
    const yvex_artifact_lowering_map *map,
    unsigned long long index)
{
    return map && index < map->summary.metadata_count ? &map->metadata[index] : NULL;
}

static const yvex_artifact_lowering_metadata *lowering_metadata_find(
    const yvex_artifact_lowering_map *map,
    const char *key)
{
    unsigned long long index;
    if (!map || !key) return NULL;
    for (index = 0u; index < map->summary.metadata_count; ++index)
        if (strcmp(map->metadata[index].key, key) == 0)
            return &map->metadata[index];
    return NULL;
}

static const char *lowering_transform_name(yvex_artifact_lowering_transform transform)
{
    return transform <= YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32
        ? lowering_transform_names[transform] : "unknown";
}

static const char *lowering_failure_name(yvex_artifact_lowering_failure_code code)
{
    return code <= YVEX_ARTIFACT_LOWERING_FAILURE_MAPPING_IDENTITY
        ? lowering_failure_names[code] : "unknown";
}


const yvex_artifact_lowering_api yvex_artifact_lowering_operations = {
    lowering_build,
    lowering_build_with_allocator,
    lowering_close,
    lowering_summary,
    lowering_descriptor_at,
    lowering_contribution_at,
    lowering_find_source,
    lowering_find_emitted,
    lowering_find_role,
    lowering_metadata_at,
    lowering_metadata_find,
    lowering_transform_name,
    lowering_failure_name
};
