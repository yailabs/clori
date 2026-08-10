/*
 * Lower the admitted DeepSeek transformation graph into deterministic GGUF descriptors,
 * metadata, quantization plans, and writer projections.
 *
 * This owner is the remaining concrete family-to-container adapter. Its resource lifecycle is
 * deliberately separate from bounded tensor-map reporting while the generic compiler lowering
 * contract is established.
 */
#include <yvex/internal/core.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/gguf_writer.h>

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
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

static const yvex_deepseek_gguf_map_failure cleared_map_failure = {
    .layer_index = YVEX_DEEPSEEK_GGUF_NO_INDEX,
    .predictor_index = YVEX_DEEPSEEK_GGUF_NO_INDEX,
    .expert_index = YVEX_DEEPSEEK_GGUF_NO_INDEX
};

/* GGUF lowering projects the sealed IR without becoming semantic identity. */

#define MAP_METADATA_CAP 64u

/* Local lowering lifecycle and diagnostic operations used before definition. */
static void lowering_close(yvex_deepseek_gguf_map *map);
static const char *lowering_failure_name(
    yvex_deepseek_gguf_map_failure_code code);
static yvex_tensor_scope map_scope(yvex_transform_scope scope);

static const yvex_model_family_ir_api *lowering_family_ir(void)
{
    return &yvex_model_register_deepseek_v4()->ir;
}

typedef struct {
    unsigned long long hash;
    unsigned long long value_plus_one;
} map_index_slot;

struct yvex_deepseek_gguf_map {
    yvex_deepseek_gguf_map_allocator allocator;
    yvex_deepseek_gguf_descriptor *descriptors;
    yvex_deepseek_gguf_contribution *contributions;
    map_index_slot *source_index;
    map_index_slot *emitted_index;
    map_index_slot *role_index;
    unsigned long long source_index_capacity;
    unsigned long long emitted_index_capacity;
    unsigned long long role_index_capacity;
    yvex_deepseek_gguf_metadata metadata[MAP_METADATA_CAP];
    yvex_deepseek_gguf_map_summary summary;
};

typedef struct {
    yvex_deepseek_gguf_map *map;
    const yvex_deepseek_v4_ir *architecture;
    const yvex_transform_ir *transform_ir;
    yvex_deepseek_gguf_map_failure *failure;
    yvex_error *err;
} map_builder;

typedef struct {
    yvex_deepseek_gguf_transform transform;
    unsigned int qtype;
    int supported;
} map_transform_projection;

static const yvex_tensor_collection map_collections[YVEX_TRANSFORM_SUBSYSTEM_COUNT] = {
    YVEX_TENSOR_COLLECTION_GLOBAL,
    YVEX_TENSOR_COLLECTION_ATTENTION,
    YVEX_TENSOR_COLLECTION_COMPRESSOR,
    YVEX_TENSOR_COLLECTION_INDEXER,
    YVEX_TENSOR_COLLECTION_NORM,
    YVEX_TENSOR_COLLECTION_MHC,
    YVEX_TENSOR_COLLECTION_ROUTER,
    YVEX_TENSOR_COLLECTION_ROUTED_EXPERT,
    YVEX_TENSOR_COLLECTION_SHARED_EXPERT,
    YVEX_TENSOR_COLLECTION_GLOBAL,
    YVEX_TENSOR_COLLECTION_AUXILIARY
};

static const yvex_tensor_scope map_scopes[] = {
    YVEX_TENSOR_SCOPE_GLOBAL,
    YVEX_TENSOR_SCOPE_MAIN_LAYER,
    YVEX_TENSOR_SCOPE_DRAFT
};

enum { MAP_SCOPE_COUNT = 3, MAP_SUBSYSTEM_COUNT = YVEX_TRANSFORM_SUBSYSTEM_COUNT };

static const yvex_deepseek_gguf_contribution_kind map_contribution_kinds[][2] = {
    {YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY,
     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY},
    {YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY,
     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_SCALE},
    {YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_WEIGHT,
     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_SCALE},
    {YVEX_DEEPSEEK_GGUF_CONTRIBUTION_ROUTING_TABLE,
     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_ROUTING_TABLE}
};

static const map_transform_projection map_transforms[YVEX_TRANSFORM_OP_COUNT] = {
    [YVEX_TRANSFORM_OP_IDENTITY] = {
        YVEX_DEEPSEEK_GGUF_TRANSFORM_DIRECT, YVEX_GGUF_NO_FORCED_QTYPE, 1},
    [YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR] = {
        YVEX_DEEPSEEK_GGUF_TRANSFORM_FP8_E4M3_E8M0, YVEX_GGUF_NO_FORCED_QTYPE, 1},
    [YVEX_TRANSFORM_OP_CHECKED_CAST] = {
        YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32, 26u, 1},
    [YVEX_TRANSFORM_OP_EXPERT_AGGREGATE] = {
        YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4, 39u, 1}
};

typedef struct {
    yvex_tensor_collection collection;
    unsigned long long count;
} map_collection_expectation;

static const map_collection_expectation map_trunk_expectations[] = {
    {YVEX_TENSOR_COLLECTION_GLOBAL, 6u},
    {YVEX_TENSOR_COLLECTION_ATTENTION, 344u},
    {YVEX_TENSOR_COLLECTION_MHC, 258u},
    {YVEX_TENSOR_COLLECTION_NORM, 86u},
    {YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, 129u},
    {YVEX_TENSOR_COLLECTION_SHARED_EXPERT, 129u},
    {YVEX_TENSOR_COLLECTION_ROUTER, 86u},
    {YVEX_TENSOR_COLLECTION_COMPRESSOR, 164u},
    {YVEX_TENSOR_COLLECTION_INDEXER, 126u}
};

typedef struct {
    size_t offset;
    unsigned long long count;
} map_summary_expectation;

static const map_summary_expectation map_summary_expectations[] = {
    {offsetof(yvex_deepseek_gguf_map_summary, source_contribution_count),
     YVEX_DEEPSEEK_GGUF_SOURCE_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, descriptor_count),
     YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, trunk_descriptor_count),
     YVEX_DEEPSEEK_GGUF_TRUNK_DESCRIPTOR_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, draft_descriptor_count),
     YVEX_DEEPSEEK_GGUF_DRAFT_DESCRIPTOR_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, pinned_standard_count),
     YVEX_DEEPSEEK_GGUF_TRUNK_DESCRIPTOR_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, extension_count),
     YVEX_DEEPSEEK_GGUF_DRAFT_DESCRIPTOR_COUNT}
};

typedef enum {
    M_LIT = 0, M_MODEL, M_LAYER, M_CSA,
    M_LAYER_NUM, M_CSA_NUM, M_RATIOS, M_CLAMP, M_DSPARK_LAYERS
} map_metadata_owner;

#define M_STR YVEX_DEEPSEEK_GGUF_METADATA_STRING
#define M_U64 YVEX_DEEPSEEK_GGUF_METADATA_U64
#define M_F64 YVEX_DEEPSEEK_GGUF_METADATA_F64
#define M_BOOL YVEX_DEEPSEEK_GGUF_METADATA_BOOL
#define M_U64S YVEX_DEEPSEEK_GGUF_METADATA_U64_ARRAY
#define M_F64S YVEX_DEEPSEEK_GGUF_METADATA_F64_ARRAY

typedef yvex_deepseek_v4_model_spec model_t;
typedef yvex_deepseek_v4_layer_spec layer_t;

typedef struct {
    const char *key;
    yvex_deepseek_gguf_metadata_type type;
    map_metadata_owner owner;
    size_t offset;
    union {
        const char *string;
        unsigned long long u64;
        double f64;
    } literal;
} map_metadata_spec;

static const map_metadata_spec map_metadata_specs[] = {
    {"general.architecture", M_STR, M_LIT, 0u, {.string = "deepseek4"}},
    {"general.name", M_STR, M_LIT, 0u, {.string = "DeepSeek-V4-Flash-DSpark"}},
    {"general.source.huggingface.repository", M_STR, M_MODEL, offsetof(model_t, repository), {0}},
    {"yvex.source.revision", M_STR, M_MODEL, offsetof(model_t, revision), {0}},
    {"deepseek4.block_count", M_U64, M_MODEL, offsetof(model_t, main_layer_count), {0}},
    {"deepseek4.embedding_length", M_U64, M_MODEL, offsetof(model_t, hidden_size), {0}},
    {"deepseek4.context_length", M_U64, M_MODEL, offsetof(model_t, maximum_context), {0}},
    {"deepseek4.vocab_size", M_U64, M_MODEL, offsetof(model_t, vocabulary_size), {0}},
    {"deepseek4.attention.head_count", M_U64, M_LAYER, offsetof(layer_t, query_heads), {0}},
    {"deepseek4.attention.head_count_kv", M_U64, M_LAYER, offsetof(layer_t, kv_heads), {0}},
    {"deepseek4.attention.key_length", M_U64, M_LAYER, offsetof(layer_t, head_dimension), {0}},
    {"deepseek4.attention.value_length", M_U64, M_LAYER, offsetof(layer_t, head_dimension), {0}},
    {"deepseek4.attention.layer_norm_rms_epsilon", M_F64, M_LAYER, offsetof(layer_t, rms_norm_epsilon), {0}},
    {"deepseek4.rope.dimension_count", M_U64, M_LAYER, offsetof(layer_t, rope_head_dimension), {0}},
    {"deepseek4.rope.freq_base", M_F64, M_LAYER_NUM, offsetof(layer_t, position.theta), {0}},
    {"deepseek4.attention.q_lora_rank", M_U64, M_LAYER, offsetof(layer_t, query_lora_rank), {0}},
    {"deepseek4.attention.output_lora_rank", M_U64, M_LAYER, offsetof(layer_t, output_lora_rank), {0}},
    {"deepseek4.attention.output_group_count", M_U64, M_LAYER, offsetof(layer_t, output_groups), {0}},
    {"deepseek4.attention.compress_ratios", M_U64S, M_RATIOS, 0u, {0}},
    {"deepseek4.attention.sliding_window", M_U64, M_LAYER, offsetof(layer_t, kv.sliding_window), {0}},
    {"deepseek4.expert_count", M_U64, M_LAYER, offsetof(layer_t, moe.routed_experts), {0}},
    {"deepseek4.expert_used_count", M_U64, M_LAYER, offsetof(layer_t, moe.experts_per_token), {0}},
    {"deepseek4.expert_shared_count", M_U64, M_LAYER, offsetof(layer_t, moe.shared_experts), {0}},
    {"deepseek4.expert_feed_forward_length", M_U64, M_LAYER, offsetof(layer_t, moe.expert_intermediate_size), {0}},
    {"deepseek4.expert_weights_scale", M_F64, M_LAYER, offsetof(layer_t, moe.routed_scaling_factor), {0}},
    {"deepseek4.expert_weights_norm", M_BOOL, M_LAYER, offsetof(layer_t, moe.normalize_topk_probabilities), {0}},
    {"deepseek4.expert_gating_func", M_U64, M_LIT, 0u, {.u64 = 4u}},
    {"deepseek4.swiglu_clamp_exp", M_F64S, M_CLAMP, 0u, {0}},
    {"deepseek4.swiglu_clamp_shexp", M_F64S, M_CLAMP, 0u, {0}},
    {"deepseek4.hash_layer_count", M_U64, M_MODEL, offsetof(model_t, hash_router_layer_count), {0}},
    {"deepseek4.attention.compress_rope_freq_base", M_F64, M_CSA_NUM, offsetof(layer_t, position.theta), {0}},
    {"deepseek4.hyper_connection.count", M_U64, M_LAYER, offsetof(layer_t, mhc.residual_streams), {0}},
    {"deepseek4.hyper_connection.sinkhorn_iterations", M_U64, M_LAYER, offsetof(layer_t, mhc.sinkhorn_iterations), {0}},
    {"deepseek4.hyper_connection.epsilon", M_F64, M_LAYER, offsetof(layer_t, mhc.epsilon), {0}},
    {"deepseek4.indexer.head_count", M_U64, M_CSA, offsetof(layer_t, indexer_heads), {0}},
    {"deepseek4.indexer.key_length", M_U64, M_CSA, offsetof(layer_t, indexer_head_dimension), {0}},
    {"deepseek4.indexer.top_k", M_U64, M_CSA, offsetof(layer_t, indexer_topk), {0}},
    {"tokenizer.ggml.model", M_STR, M_LIT, 0u, {.string = "gpt2"}},
    {"tokenizer.ggml.vocab_size", M_U64, M_MODEL, offsetof(model_t, tokenizer.vocabulary_size), {0}},
    {"tokenizer.ggml.bos_token_id", M_U64, M_MODEL, offsetof(model_t, tokenizer.bos_token_id), {0}},
    {"tokenizer.ggml.eos_token_id", M_U64, M_MODEL, offsetof(model_t, tokenizer.eos_token_id), {0}},
    {"yvex.tokenizer.sidecars_verified", M_BOOL, M_LIT, 0u, {.u64 = 1u}},
    {"yvex.deepseek4.dspark.schema", M_U64, M_LIT, 0u,
     {.u64 = YVEX_GGUF_DSPARK_EXTENSION_VERSION}},
    {"yvex.deepseek4.dspark.block_size", M_U64, M_MODEL,
     offsetof(model_t, dspark.block_size), {0}},
    {"yvex.deepseek4.dspark.noise_token_id", M_U64, M_MODEL,
     offsetof(model_t, dspark.noise_token_id), {0}},
    {"yvex.deepseek4.dspark.target_layer_ids", M_U64S, M_DSPARK_LAYERS, 0u, {0}},
    {"yvex.deepseek4.dspark.draft_layer_count", M_U64, M_MODEL,
     offsetof(model_t, dspark.draft_layer_count), {0}},
    {"yvex.deepseek4.dspark.markov_rank", M_U64, M_MODEL,
     offsetof(model_t, dspark.markov_rank), {0}},
    {"yvex.deepseek4.dspark.confidence_available", M_BOOL, M_MODEL,
     offsetof(model_t, dspark.confidence_available), {0}},
    {"yvex.deepseek4.dspark.target_verification_required", M_BOOL, M_MODEL,
     offsetof(model_t, dspark.target_verification_required), {0}},
    {"yvex.deepseek4.dspark.descriptor_count", M_U64, M_LIT, 0u,
     {.u64 = YVEX_DEEPSEEK_GGUF_DRAFT_DESCRIPTOR_COUNT}},
    {"yvex.deepseek4.dspark.runtime_supported", M_BOOL, M_LIT, 0u, {.u64 = 1u}},
    {"yvex.deepseek4.dspark.name_prefix", M_STR, M_LIT, 0u,
     {.string = "yvex.draft.v1"}}
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

static void map_failure_clear(yvex_deepseek_gguf_map_failure *failure)
{
    if (!failure) return;
    *failure = cleared_map_failure;
}

static int map_reject(map_builder *builder,
                      yvex_deepseek_gguf_map_failure_code code,
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
    yvex_status status = code == YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ALLOCATION
        ? YVEX_ERR_NOMEM
        : (code == YVEX_DEEPSEEK_GGUF_MAP_FAILURE_INVALID_ARGUMENT
            ? YVEX_ERR_INVALID_ARG : YVEX_ERR_FORMAT);
    yvex_deepseek_gguf_map_failure *failure =
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
                    "deepseek_gguf_lowering",
                    "%s role=%s source=%s emitted=%s layer=%llu expert=%llu expected=%llu actual=%llu",
                    lowering_failure_name(code),
                    yvex_tensor_role_name(role),
                    source_name ? source_name : "none",
                    emitted_name ? emitted_name : "none", layer, expert,
                    expected, actual);
    return status;
}

static int map_reject_global(map_builder *builder,
                             yvex_deepseek_gguf_map_failure_code code,
                             const char *subject,
                             unsigned long long expected,
                             unsigned long long actual)
{
    return map_reject(builder, code, YVEX_TENSOR_ROLE_UNKNOWN,
                      YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_GGUF_NO_INDEX,
                      YVEX_DEEPSEEK_GGUF_NO_INDEX, YVEX_DEEPSEEK_GGUF_NO_INDEX,
                      subject, NULL, expected, actual);
}

static int map_reject_descriptor(map_builder *builder,
                                 yvex_deepseek_gguf_map_failure_code code,
                                 const yvex_deepseek_gguf_descriptor *descriptor,
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
                          yvex_deepseek_gguf_map_failure_code code,
                          const yvex_transform_logical_key *key,
                          int include_location,
                          unsigned long long expected,
                          unsigned long long actual)
{
    return map_reject(builder, code, key ? key->role : YVEX_TENSOR_ROLE_UNKNOWN,
                      key ? map_scope(key->scope) : YVEX_TENSOR_SCOPE_GLOBAL,
                      key && include_location ? key->layer_index
                                              : YVEX_DEEPSEEK_GGUF_NO_INDEX,
                      key && include_location ? key->auxiliary_index
                                              : YVEX_DEEPSEEK_GGUF_NO_INDEX,
                      YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, NULL, expected, actual);
}

static void *map_allocate_zero(yvex_deepseek_gguf_map *map, size_t size)
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

static int map_unique_index_equal(const yvex_deepseek_gguf_map *map,
                                  int emitted,
                                  unsigned long long left,
                                  unsigned long long right)
{
    const yvex_deepseek_gguf_descriptor *candidate = &map->descriptors[right];
    const yvex_deepseek_gguf_descriptor *current = &map->descriptors[left];

    return emitted ? strcmp(current->emitted_name, candidate->emitted_name) == 0
                   : current->role == candidate->role &&
                         current->scope == candidate->scope &&
                         current->layer_index == candidate->layer_index &&
                         current->predictor_index == candidate->predictor_index;
}

static int map_unique_index_insert(yvex_deepseek_gguf_map *map,
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

static int map_emitted_index_insert(yvex_deepseek_gguf_map *map,
                                    unsigned long long hash,
                                    unsigned long long value)
{
    return map_unique_index_insert(map, 1, hash, value);
}

static int map_role_index_insert(yvex_deepseek_gguf_map *map,
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
                         yvex_deepseek_gguf_transform *transform,
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

static yvex_deepseek_gguf_contribution_kind map_contribution_kind(
    yvex_deepseek_gguf_transform transform,
    unsigned long long input)
{
    unsigned int secondary = transform == YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4
        ? (unsigned int)(input & 1u) : input != 0u;

    return (unsigned int)transform <= YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32
        ? map_contribution_kinds[(unsigned int)transform][secondary]
        : YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY;
}

static int map_descriptor_begin(map_builder *builder,
                                const yvex_transform_value *terminal,
                                const yvex_transform_node *node,
                                unsigned long long descriptor_index)
{
    yvex_deepseek_gguf_map *map = builder->map;
    yvex_deepseek_gguf_descriptor *descriptor =
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
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_LOWERING_DIVERGENCE,
            descriptor, YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, NULL, 1u, 0u);
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
            YVEX_DEEPSEEK_GGUF_AGGREGATED_AXIS;
    }
    if (!yvex_gguf_name_map_resolve(
            descriptor->role, scope == YVEX_TENSOR_SCOPE_DRAFT,
            descriptor->layer_index, descriptor->predictor_index,
            descriptor->emitted_name, sizeof(descriptor->emitted_name),
            &provenance, &reason)) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_NAME, descriptor,
            YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, reason, 1u, 0u);
    }
    descriptor->name_provenance = provenance;
    if (!yvex_gguf_layout_map_shape_supported(
            descriptor->role, qtype, descriptor->logical_rank,
            descriptor->logical_dims, &reason)) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_LAYOUT, descriptor,
            YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, descriptor->emitted_name, 1u, 0u);
    }
    if (!map_emitted_index_insert(
            map, map_hash_string(descriptor->emitted_name), descriptor_index)) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_DUPLICATE_NAME, descriptor,
            YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, descriptor->emitted_name, 1u, 2u);
    }
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->role);
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->scope);
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->layer_index);
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->predictor_index);
    if (!map_role_index_insert(map, role_hash, descriptor_index)) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_DUPLICATE_NAME, descriptor,
            YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, descriptor->emitted_name, 1u, 2u);
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
    yvex_deepseek_gguf_map *map = builder->map;
    yvex_deepseek_gguf_descriptor *descriptor =
        &map->descriptors[descriptor_index];
    const yvex_transform_value *value = yvex_transform_ir_node_input_at(
        builder->transform_ir, node, input_index);
    const yvex_transform_source_value *source;
    yvex_deepseek_gguf_contribution *contribution;
    unsigned long long index = map->summary.source_contribution_count;
    unsigned int dimension;

    if (!value || value->kind != YVEX_TRANSFORM_VALUE_SOURCE)
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_LOWERING_DIVERGENCE,
            descriptor, YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL,
            descriptor->emitted_name, 1u, 0u);
    source = yvex_transform_ir_source_at(
        builder->transform_ir, value->source_index);
    if (!source || index >= YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        source->requirement_index >= YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        source->shape.rank > 2u || source->role_hint != descriptor->role ||
        map_scope(source->scope) != descriptor->scope ||
        map_collection(source->subsystem) != descriptor->collection) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_COVERAGE_ROW, descriptor,
            source ? source->expert_index : YVEX_DEEPSEEK_GGUF_NO_INDEX,
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
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_DUPLICATE_SOURCE,
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
        summary->source_value_count != YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        summary->terminal_count != YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT ||
        summary->edge_count != YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        summary->payload_bytes_read != 0u) {
        return map_reject_global(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_TRANSFORM_IR, NULL,
            YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT,
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
                builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_TRANSFORM_IR,
                terminal ? &terminal->logical_key : NULL, 0, ordinal,
                terminal ? terminal->canonical_ordinal : ULLONG_MAX);
        }
        node = yvex_transform_ir_node_at(
            builder->transform_ir, terminal->producer_node_id);
        if (!node || node->output_value_id != terminal->id) {
            return map_reject_key(
                builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_TRANSFORM_IR,
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
                              yvex_deepseek_gguf_metadata **out)
{
    yvex_deepseek_gguf_map *map = builder->map;
    unsigned long long index;

    if (!key || map->summary.metadata_count >= MAP_METADATA_CAP)
        return map_reject_global(builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_METADATA,
                                 key, MAP_METADATA_CAP,
                                 map->summary.metadata_count + 1u);
    for (index = 0u; index < map->summary.metadata_count; ++index)
        if (strcmp(map->metadata[index].key, key) == 0)
            return map_reject_global(builder,
                                     YVEX_DEEPSEEK_GGUF_MAP_FAILURE_METADATA,
                                     key, 1u, 2u);
    *out = &map->metadata[map->summary.metadata_count++];
    yvex_core_text_copy((*out)->key, sizeof((*out)->key), key);
    return YVEX_OK;
}

static int map_add_metadata_spec(map_builder *builder,
                                 const map_metadata_spec *spec,
                                 const model_t *model,
                                 const layer_t *first,
                                 const layer_t *first_csa,
                                 const unsigned long long *ratios,
                                 const double *clamp)
{
    yvex_deepseek_gguf_metadata *entry = NULL;
    const void *owner = spec->owner == M_MODEL
        ? (const void *)model
        : (spec->owner == M_CSA ||
           spec->owner == M_CSA_NUM)
            ? (const void *)first_csa : (const void *)first;
    const char *field = owner ? (const char *)owner + spec->offset : NULL;
    unsigned int count = (unsigned int)model->main_layer_count;
    int rc;

    if (!count || count > 64u)
        return map_reject_global(builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_METADATA,
                                 spec->key, 64u, count);
    rc = map_metadata_begin(builder, spec->key, &entry);
    if (rc != YVEX_OK) return rc;
    entry->type = spec->type;
    if (spec->type == M_STR) {
        const char *value = spec->owner == M_LIT
            ? spec->literal.string : field;
        yvex_core_text_copy(entry->string_value, sizeof(entry->string_value), value);
    } else if (spec->type == M_U64) {
        entry->u64_value = spec->owner == M_LIT
            ? spec->literal.u64 : *(const unsigned long long *)field;
    } else if (spec->type == M_BOOL) {
        entry->bool_value = spec->owner == M_LIT
            ? spec->literal.u64 != 0u : *(const int *)field != 0;
    } else if (spec->type == M_F64) {
        entry->f64_value = spec->owner == M_LAYER_NUM ||
                           spec->owner == M_CSA_NUM
            ? (double)*(const unsigned long long *)field : *(const double *)field;
    } else if (spec->type == M_U64S) {
        if (spec->owner == M_DSPARK_LAYERS) {
            count = (unsigned int)model->dspark.target_layer_count;
            memcpy(entry->array_values, model->dspark.target_layer_ids,
                   (size_t)count * sizeof(entry->array_values[0]));
        } else {
            memcpy(entry->array_values, ratios,
                   (size_t)count * sizeof(entry->array_values[0]));
        }
    } else {
        memcpy(entry->f64_array_values, clamp,
               (size_t)count * sizeof(entry->f64_array_values[0]));
    }
    entry->array_count = spec->type >= M_U64S
        ? count : 0u;
    return YVEX_OK;
}

static int map_build_metadata(map_builder *builder)
{
    const model_t *model =
        lowering_family_ir()->model(builder->architecture);
    const layer_t *first =
        lowering_family_ir()->layer_at(builder->architecture, 0u);
    const layer_t *first_csa =
        lowering_family_ir()->layer_at(builder->architecture, 2u);
    unsigned long long ratios[64];
    double clamp[64];
    unsigned long long index;
    int rc;
    for (index = 0u; index < model->main_layer_count; ++index) {
        const layer_t *layer =
            lowering_family_ir()->layer_at(builder->architecture, index);
        ratios[index] = layer->compression_ratio;
        clamp[index] = layer->moe.activation_limit;
    }
    for (index = 0u;
         index < sizeof(map_metadata_specs) / sizeof(map_metadata_specs[0]);
         ++index) {
        rc = map_add_metadata_spec(builder, &map_metadata_specs[index], model,
                                   first, first_csa, ratios, clamp);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

static int map_finalize(map_builder *builder)
{
    yvex_deepseek_gguf_map *map = builder->map;
    unsigned long long trunk[YVEX_TENSOR_COLLECTION_COUNT] = {0};
    unsigned long long identity = 1469598103934665603ull;
    unsigned long long index;

    for (index = 0u;
         index < sizeof(map_summary_expectations) / sizeof(map_summary_expectations[0]);
         ++index)
        if (*(const unsigned long long *)((const char *)&map->summary +
                                          map_summary_expectations[index].offset) !=
            map_summary_expectations[index].count)
            return map_reject_global(
                builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ACCOUNTING, NULL,
                YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT,
                map->summary.descriptor_count);
    for (index = 0u; index < map->summary.descriptor_count; ++index) {
        const yvex_deepseek_gguf_descriptor *descriptor =
            &map->descriptors[index];
        if (descriptor->scope != YVEX_TENSOR_SCOPE_DRAFT)
            trunk[descriptor->collection]++;
        identity = yvex_core_hash_mix_u64(identity, descriptor->identity);
    }
    for (index = 0u;
         index < sizeof(map_trunk_expectations) / sizeof(map_trunk_expectations[0]);
         ++index)
        if (trunk[map_trunk_expectations[index].collection] !=
            map_trunk_expectations[index].count)
            return map_reject(
                builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ACCOUNTING,
                YVEX_TENSOR_ROLE_UNKNOWN, YVEX_TENSOR_SCOPE_MAIN_LAYER,
                YVEX_DEEPSEEK_GGUF_NO_INDEX, YVEX_DEEPSEEK_GGUF_NO_INDEX,
                YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, NULL, 1328u, 0u);
    identity = yvex_core_hash_mix_u64(identity, map->summary.source_identity);
    identity = yvex_core_hash_mix_u64(identity, map->summary.coverage_identity);
    map->summary.mapping_identity = identity;
    map->summary.complete = 1;
    return YVEX_OK;
}

static int lowering_build_with_allocator(
    yvex_deepseek_gguf_map **out,
    const yvex_deepseek_v4_ir *architecture,
    const yvex_transform_ir *transform_ir,
    const yvex_deepseek_gguf_map_allocator *allocator,
    yvex_deepseek_gguf_map_failure *failure,
    yvex_error *err)
{
    const model_t *model;
    const yvex_transform_ir_summary *transform_summary;
    yvex_deepseek_gguf_map *map;
    map_builder builder;
    size_t bytes;
    int rc;

    if (out) *out = NULL;
    map_failure_clear(failure);
    yvex_error_clear(err);
    memset(&builder, 0, sizeof(builder));
    builder.failure = failure;
    builder.err = err;
    if (!out || !architecture || !transform_ir || !allocator ||
        !allocator->allocate || !allocator->release) {
        return map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_INVALID_ARGUMENT,
            NULL, 1u, 0u);
    }
    model = lowering_family_ir()->model(architecture);
    transform_summary = yvex_transform_ir_summary_get(transform_ir);
    if (!model || model->main_layer_count != 43u ||
        model->auxiliary_layer_count != 3u) {
        return map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ARCHITECTURE, NULL, 46u,
            model ? model->main_layer_count + model->auxiliary_layer_count : 0u);
    }
    if (!transform_summary || !transform_summary->complete ||
        transform_summary->source_value_count !=
            YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        transform_summary->terminal_count !=
            YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT) {
        return map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_TRANSFORM_IR, NULL,
            YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT,
            transform_summary ? transform_summary->terminal_count : 0u);
    }
    map = (yvex_deepseek_gguf_map *)allocator->allocate(
        sizeof(*map), allocator->context);
    if (!map)
        return map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ALLOCATION,
            "map", sizeof(*map), 0u);
    memset(map, 0, sizeof(*map));
    map->allocator = *allocator;
    builder.map = map;
    if (!yvex_core_power_of_two_capacity(YVEX_DEEPSEEK_GGUF_SOURCE_COUNT, 8ull,
                                         1ull, 2ull, &map->source_index_capacity) ||
        !yvex_core_power_of_two_capacity(YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT, 8ull,
                                         1ull, 2ull, &map->emitted_index_capacity) ||
        !yvex_core_power_of_two_capacity(YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT, 8ull,
                                         1ull, 2ull, &map->role_index_capacity))
    {
        rc = map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ARITHMETIC_OVERFLOW,
            "mapping-index", 1u, 0u);
        lowering_close(map);
        return rc;
    }
    map->descriptors = (yvex_deepseek_gguf_descriptor *)map_allocate_zero(
        map, (size_t)YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT *
             sizeof(*map->descriptors));
    map->contributions = (yvex_deepseek_gguf_contribution *)map_allocate_zero(
        map, (size_t)YVEX_DEEPSEEK_GGUF_SOURCE_COUNT *
             sizeof(*map->contributions));
    bytes = (size_t)map->source_index_capacity * sizeof(*map->source_index);
    map->source_index = (map_index_slot *)map_allocate_zero(map, bytes);
    bytes = (size_t)map->emitted_index_capacity * sizeof(*map->emitted_index);
    map->emitted_index = (map_index_slot *)map_allocate_zero(map, bytes);
    bytes = (size_t)map->role_index_capacity * sizeof(*map->role_index);
    map->role_index = (map_index_slot *)map_allocate_zero(map, bytes);
    builder.architecture = architecture;
    builder.transform_ir = transform_ir;
    if (!map->descriptors || !map->contributions || !map->source_index ||
        !map->emitted_index || !map->role_index) {
        rc = map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ALLOCATION,
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
    yvex_deepseek_gguf_map **out,
    const yvex_deepseek_v4_ir *architecture,
    const yvex_transform_ir *transform_ir,
    yvex_deepseek_gguf_map_failure *failure,
    yvex_error *err)
{
    yvex_deepseek_gguf_map_allocator allocator;
    allocator.allocate = map_default_allocate;
    allocator.release = map_default_release;
    allocator.context = NULL;
    return lowering_build_with_allocator(
        out, architecture, transform_ir, &allocator, failure, err);
}

static void lowering_close(yvex_deepseek_gguf_map *map)
{
    yvex_deepseek_gguf_map_allocator allocator;
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

static const yvex_deepseek_gguf_map_summary *lowering_summary(
    const yvex_deepseek_gguf_map *map)
{
    return map ? &map->summary : NULL;
}

static const yvex_deepseek_gguf_descriptor *lowering_at(
    const yvex_deepseek_gguf_map *map,
    unsigned long long index)
{
    return map && index < map->summary.descriptor_count ? &map->descriptors[index] : NULL;
}

static const yvex_deepseek_gguf_contribution *
lowering_contribution_at(
    const yvex_deepseek_gguf_map *map,
    unsigned long long index)
{
    return map && index < map->summary.source_contribution_count ? &map->contributions[index] : NULL;
}

static const yvex_deepseek_gguf_descriptor *map_find_name(
    const yvex_deepseek_gguf_map *map,
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

static const yvex_deepseek_gguf_descriptor *lowering_find_source(
    const yvex_deepseek_gguf_map *map,
    const char *source_name)
{
    return map_find_name(map, source_name, 0);
}

static const yvex_deepseek_gguf_descriptor *lowering_find_emitted(
    const yvex_deepseek_gguf_map *map,
    const char *emitted_name)
{
    return map_find_name(map, emitted_name, 1);
}

static const yvex_deepseek_gguf_descriptor *lowering_find_role(
    const yvex_deepseek_gguf_map *map,
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
            const yvex_deepseek_gguf_descriptor *descriptor =
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

static const yvex_deepseek_gguf_metadata *lowering_metadata_at(
    const yvex_deepseek_gguf_map *map,
    unsigned long long index)
{
    return map && index < map->summary.metadata_count ? &map->metadata[index] : NULL;
}

static const yvex_deepseek_gguf_metadata *lowering_metadata_find(
    const yvex_deepseek_gguf_map *map,
    const char *key)
{
    unsigned long long index;
    if (!map || !key) return NULL;
    for (index = 0u; index < map->summary.metadata_count; ++index)
        if (strcmp(map->metadata[index].key, key) == 0)
            return &map->metadata[index];
    return NULL;
}

static const char *lowering_transform_name(
    yvex_deepseek_gguf_transform transform)
{
    return transform <= YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32
        ? lowering_transform_names[transform] : "unknown";
}

static const char *lowering_failure_name(
    yvex_deepseek_gguf_map_failure_code code)
{
    return code <= YVEX_DEEPSEEK_GGUF_MAP_FAILURE_MAPPING_IDENTITY
        ? lowering_failure_names[code] : "unknown";
}

/*
 * Publish the immutable GGUF-lowering operation table used by the family registration without
 * exporting its implementation helpers.
 *
 * Returns process-lifetime immutable storage; no allocation or I/O.
 */
const yvex_model_family_lowering_api *yvex_model_deepseek_lowering_api(void)
{
    static const yvex_model_family_lowering_api api = {
        lowering_build,
        lowering_build_with_allocator,
        lowering_close,
        lowering_summary,
        lowering_at,
        lowering_contribution_at,
        lowering_find_source,
        lowering_find_emitted,
        lowering_find_role,
        lowering_metadata_at,
        lowering_metadata_find,
        lowering_transform_name,
        lowering_failure_name
    };

    return &api;
}

static int quant_lowering_summary(const void *context, yvex_quant_lowering_summary *out)
{
    const yvex_deepseek_gguf_map_summary *summary =
        lowering_summary((const yvex_deepseek_gguf_map *)context);

    if (!summary || !out) return 0;
    *out = (yvex_quant_lowering_summary){
        summary->source_contribution_count,
        summary->descriptor_count,
        summary->source_identity,
        summary->mapping_identity,
        summary->complete};
    return 1;
}

static int quant_lowering_qtypes(yvex_deepseek_gguf_transform transform,
                                 unsigned int *source_faithful, unsigned int *release)
{
    if (!source_faithful || !release) return 0;
    switch (transform) {
    case YVEX_DEEPSEEK_GGUF_TRANSFORM_DIRECT:
    case YVEX_DEEPSEEK_GGUF_TRANSFORM_FP8_E4M3_E8M0:
        *source_faithful = YVEX_GGUF_QTYPE_F32;
        *release = YVEX_GGUF_QTYPE_Q8_0;
        return 1;
    case YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4:
        *source_faithful = YVEX_GGUF_QTYPE_MXFP4;
        *release = YVEX_GGUF_QTYPE_Q2_K;
        return 1;
    case YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32:
        *source_faithful = YVEX_GGUF_QTYPE_I32;
        *release = YVEX_GGUF_QTYPE_I32;
        return 1;
    }
    return 0;
}

static yvex_transform_operation_kind quant_lowering_operation(
    yvex_deepseek_gguf_transform transform)
{
    switch (transform) {
    case YVEX_DEEPSEEK_GGUF_TRANSFORM_DIRECT: return YVEX_TRANSFORM_OP_IDENTITY;
    case YVEX_DEEPSEEK_GGUF_TRANSFORM_FP8_E4M3_E8M0:
        return YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR;
    case YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4:
        return YVEX_TRANSFORM_OP_EXPERT_AGGREGATE;
    case YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32: return YVEX_TRANSFORM_OP_CHECKED_CAST;
    }
    return YVEX_TRANSFORM_OP_COUNT;
}

static int quant_lowering_tensor(const void *context, unsigned long long ordinal,
                                 yvex_quant_lowering_tensor *out)
{
    const yvex_deepseek_gguf_descriptor *row =
        lowering_at((const yvex_deepseek_gguf_map *)context, ordinal);

    if (!row || !out || row->logical_rank > YVEX_GGUF_QTYPE_MAX_DIMS ||
        row->transform > YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32) return 0;
    memset(out, 0, sizeof(*out));
    out->role = row->role;
    out->collection = row->collection;
    out->scope = row->scope;
    out->layer_index = row->layer_index;
    out->predictor_index = row->predictor_index;
    out->expert_count = row->expert_count;
    yvex_core_text_copy(out->emitted_name, sizeof(out->emitted_name), row->emitted_name);
    out->operation = quant_lowering_operation(row->transform);
    if (out->operation == YVEX_TRANSFORM_OP_COUNT ||
        !quant_lowering_qtypes(row->transform, &out->source_faithful_qtype,
                               &out->release_qtype)) return 0;
    out->profile_qtype_required =
        row->transform == YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4 ||
        row->transform == YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32;
    out->logical_rank = row->logical_rank;
    memcpy(out->logical_dims, row->logical_dims, sizeof(out->logical_dims));
    memcpy(out->source_axis_for_logical, row->source_axis_for_logical,
           sizeof(out->source_axis_for_logical));
    out->contribution_offset = row->contribution_offset;
    out->contribution_count = row->contribution_count;
    return 1;
}

static int quant_lowering_contribution(const void *context, unsigned long long ordinal,
                                       yvex_quant_lowering_contribution *out)
{
    const yvex_deepseek_gguf_contribution *row =
        lowering_contribution_at((const yvex_deepseek_gguf_map *)context, ordinal);

    if (!row || !out) return 0;
    memset(out, 0, sizeof(*out));
    yvex_core_text_copy(out->source_name, sizeof(out->source_name), row->source_name);
    out->source_dtype = row->source_dtype;
    out->tensor_ordinal = row->descriptor_index;
    out->expert_index = row->expert_index;
    return 1;
}

static const yvex_quant_lowering_api *deepseek_quant_lowering_api(void)
{
    static const yvex_quant_lowering_api api = {
        YVEX_DEEPSEEK_QUANT_SOURCE_PROFILE_NAME,
        YVEX_DEEPSEEK_QUANT_RELEASE_PROFILE_NAME,
        quant_lowering_summary, quant_lowering_tensor, quant_lowering_contribution};
    return &api;
}

static void deepseek_preset_rule(
    yvex_quant_policy_rule *rule, unsigned long long match_mask, yvex_tensor_role role,
    yvex_quant_policy_operation operation, yvex_tensor_scope scope,
    yvex_quant_policy_physical_class physical_class, yvex_quant_qtype qtype, int imatrix,
    unsigned int priority, const char *label)
{
    memset(rule, 0, sizeof(*rule));
    rule->schema_version = YVEX_QUANT_POLICY_SCHEMA_VERSION;
    rule->match_mask = match_mask;
    rule->role = role;
    rule->operation = operation;
    rule->scope = scope;
    rule->physical_class = physical_class;
    rule->qtype = qtype;
    rule->requires_imatrix = imatrix;
    rule->requires_cpu_compute = 1;
    rule->requires_cuda_compute = 1;
    rule->priority = priority;
    rule->label = label;
}

unsigned long long yvex_quant_policy_preset_count(void)
{
    return 3u;
}

const char *yvex_quant_policy_preset_name(unsigned long long index)
{
    static const char *const names[] = {
        "source-faithful", YVEX_DEEPSEEK_QUANT_RELEASE_PROFILE_NAME,
        YVEX_DEEPSEEK_QUANT_DSPARK_PROFILE_NAME};
    return index < sizeof(names) / sizeof(names[0]) ? names[index] : NULL;
}

int yvex_quant_policy_preset_open(yvex_quant_policy **out, const char *name, yvex_error *err)
{
    static const yvex_tensor_role exact_roles[] = {
        YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM, YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM,
        YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING, YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT,
        YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE};
    yvex_quant_policy_rule rules[10];
    yvex_quant_policy_definition definition;
    unsigned long long count = 0u;
    unsigned long long index;

    if (out) *out = NULL;
    if (!out || !name) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_preset",
                       "out and preset name are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(name, "source-faithful") != 0 &&
        strcmp(name, YVEX_DEEPSEEK_QUANT_RELEASE_PROFILE_NAME) != 0 &&
        strcmp(name, YVEX_DEEPSEEK_QUANT_DSPARK_PROFILE_NAME) != 0) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "quant_policy_preset",
                        "unknown quantization preset: %s", name);
        return YVEX_ERR_UNSUPPORTED;
    }
    deepseek_preset_rule(
        &rules[count++], YVEX_QUANT_MATCH_PHYSICAL_CLASS, YVEX_TENSOR_ROLE_UNKNOWN,
        YVEX_QUANT_POLICY_OPERATION_ANY, YVEX_TENSOR_SCOPE_GLOBAL,
        YVEX_QUANT_POLICY_PHYSICAL_QUANTIZABLE,
        strcmp(name, "source-faithful") == 0 ? YVEX_QUANT_QTYPE_SOURCE : YVEX_QUANT_QTYPE_Q8_0,
        0, 10u, strcmp(name, "source-faithful") == 0
                      ? "preserve the admitted source physical representation"
                      : "default approximable terminal representation");
    if (strcmp(name, YVEX_DEEPSEEK_QUANT_RELEASE_PROFILE_NAME) == 0)
        deepseek_preset_rule(
            &rules[count++], YVEX_QUANT_MATCH_OPERATION, YVEX_TENSOR_ROLE_UNKNOWN,
            YVEX_QUANT_POLICY_OPERATION_EXPERT_AGGREGATE, YVEX_TENSOR_SCOPE_GLOBAL,
            YVEX_QUANT_POLICY_PHYSICAL_ANY, YVEX_QUANT_QTYPE_Q2_K, 0, 100u,
            "verified release routed-expert aggregate");
    if (strcmp(name, YVEX_DEEPSEEK_QUANT_DSPARK_PROFILE_NAME) == 0) {
        static const yvex_tensor_role expert_roles[] = {
            YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, YVEX_TENSOR_ROLE_MOE_EXPERT_UP,
            YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN};
        static const yvex_quant_qtype expert_qtypes[] = {
            YVEX_QUANT_QTYPE_IQ2_XXS, YVEX_QUANT_QTYPE_IQ2_XXS, YVEX_QUANT_QTYPE_Q2_K};
        static const char *const expert_labels[] = {
            "imatrix-weighted routed expert gate", "imatrix-weighted routed expert up",
            "imatrix-covered routed expert down"};
        for (index = 0u; index < 3u; ++index)
            deepseek_preset_rule(
                &rules[count++], YVEX_QUANT_MATCH_ROLE | YVEX_QUANT_MATCH_SCOPE |
                                     YVEX_QUANT_MATCH_OPERATION,
                expert_roles[index], YVEX_QUANT_POLICY_OPERATION_EXPERT_AGGREGATE,
                YVEX_TENSOR_SCOPE_MAIN_LAYER, YVEX_QUANT_POLICY_PHYSICAL_ANY,
                expert_qtypes[index], 1, 200u, expert_labels[index]);
        deepseek_preset_rule(
            &rules[count++], YVEX_QUANT_MATCH_SCOPE | YVEX_QUANT_MATCH_PHYSICAL_CLASS,
            YVEX_TENSOR_ROLE_UNKNOWN, YVEX_QUANT_POLICY_OPERATION_ANY,
            YVEX_TENSOR_SCOPE_DRAFT, YVEX_QUANT_POLICY_PHYSICAL_QUANTIZABLE,
            YVEX_QUANT_QTYPE_Q8_0, 0, 150u, "conservative DSpark draft representation");
        for (index = 0u; index < sizeof(exact_roles) / sizeof(exact_roles[0]); ++index)
            deepseek_preset_rule(
                &rules[count++], YVEX_QUANT_MATCH_ROLE | YVEX_QUANT_MATCH_SCOPE,
                exact_roles[index], YVEX_QUANT_POLICY_OPERATION_ANY, YVEX_TENSOR_SCOPE_DRAFT,
                YVEX_QUANT_POLICY_PHYSICAL_ANY, YVEX_QUANT_QTYPE_BF16, 0, 250u,
                "exact DSpark norm, Markov, and confidence control");
    }
    definition = (yvex_quant_policy_definition){
        name, "deepseek4-v4-flash-dspark", "built-in-preset", rules, count};
    return yvex_quant_policy_create_definition(out, &definition, err);
}

int yvex_quant_plan_build_deepseek_profile(
    yvex_quant_plan **out, const yvex_transform_ir *ir,
    const yvex_transform_binding *binding, const yvex_deepseek_gguf_map *map,
    yvex_quant_profile_kind profile, const yvex_quant_plan_options *options,
    yvex_quant_failure *failure, yvex_error *err)
{
    return yvex_quant_plan_build_profile(out, ir, binding, deepseek_quant_lowering_api(), map,
                                         profile, options, failure, err);
}

int yvex_quant_plan_build_deepseek_policy(
    yvex_quant_plan **out, const yvex_transform_ir *ir,
    const yvex_transform_binding *binding, const yvex_deepseek_gguf_map *map,
    const yvex_quant_policy *policy, const char *imatrix_identity,
    const yvex_quant_plan_options *options, yvex_quant_failure *failure, yvex_error *err)
{
    return yvex_quant_plan_build_policy(out, ir, binding, deepseek_quant_lowering_api(), map,
                                        policy, imatrix_identity, options, failure, err);
}

static int writer_lowering_summary(const void *context,
                                   yvex_gguf_writer_lowering_summary *out)
{
    const yvex_deepseek_gguf_map_summary *summary =
        lowering_summary((const yvex_deepseek_gguf_map *)context);

    if (!summary || !out) return 0;
    *out = (yvex_gguf_writer_lowering_summary){
        summary->descriptor_count,
        summary->metadata_count,
        summary->source_identity,
        summary->mapping_identity,
        summary->complete};
    return 1;
}

static int writer_lowering_tensor(const void *context, unsigned long long ordinal,
                                  yvex_gguf_writer_lowering_tensor *out)
{
    const yvex_deepseek_gguf_descriptor *row =
        lowering_at((const yvex_deepseek_gguf_map *)context, ordinal);

    if (!row || !out || row->logical_rank > YVEX_GGUF_QTYPE_MAX_DIMS) return 0;
    memset(out, 0, sizeof(*out));
    yvex_core_text_copy(out->emitted_name, sizeof(out->emitted_name), row->emitted_name);
    out->logical_rank = row->logical_rank;
    memcpy(out->logical_dims, row->logical_dims,
           row->logical_rank * sizeof(out->logical_dims[0]));
    return 1;
}

static int writer_lowering_metadata(const void *context, unsigned long long ordinal,
                                    yvex_gguf_writer_lowering_metadata *out)
{
    const yvex_deepseek_gguf_metadata *row =
        lowering_metadata_at((const yvex_deepseek_gguf_map *)context, ordinal);

    if (!row || !out || row->type > YVEX_DEEPSEEK_GGUF_METADATA_F64_ARRAY ||
        row->array_count > 64u) return 0;
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

const yvex_gguf_writer_lowering_api *yvex_model_deepseek_writer_lowering_api(void)
{
    static const yvex_gguf_writer_lowering_api api = {
        "deepseek-v3",
        writer_lowering_summary,
        writer_lowering_tensor,
        writer_lowering_metadata};

    return &api;
}
