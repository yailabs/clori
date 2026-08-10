/*
 * Lower the admitted DeepSeek transformation graph into deterministic GGUF descriptors,
 * metadata, quantization plans, and writer projections.
 *
 * This owner is the remaining concrete family-to-container adapter. Its resource lifecycle is
 * deliberately separate from bounded tensor-map reporting while the generic compiler lowering
 * contract is established.
 */
#include <yvex/internal/core.h>
#include <yvex/internal/artifact_lowering.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/gguf_writer.h>

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


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

typedef enum {
    M_LIT = 0, M_MODEL, M_LAYER, M_CSA,
    M_LAYER_NUM, M_CSA_NUM, M_RATIOS, M_CLAMP, M_DSPARK_LAYERS
} map_metadata_owner;

#define M_STR YVEX_ARTIFACT_LOWERING_METADATA_STRING
#define M_U64 YVEX_ARTIFACT_LOWERING_METADATA_U64
#define M_F64 YVEX_ARTIFACT_LOWERING_METADATA_F64
#define M_BOOL YVEX_ARTIFACT_LOWERING_METADATA_BOOL
#define M_U64S YVEX_ARTIFACT_LOWERING_METADATA_U64_ARRAY
#define M_F64S YVEX_ARTIFACT_LOWERING_METADATA_F64_ARRAY

typedef yvex_deepseek_v4_model_spec model_t;
typedef yvex_deepseek_v4_layer_spec layer_t;

typedef struct {
    const char *key;
    yvex_artifact_lowering_metadata_type type;
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


static const yvex_model_family_ir_api *lowering_family_ir(void);

static int deepseek_metadata_project(yvex_artifact_lowering_metadata *entry,
                                     const map_metadata_spec *spec,
                                     const model_t *model,
                                     const layer_t *first,
                                     const layer_t *first_csa,
                                     const unsigned long long *ratios,
                                     const double *clamp)
{
    const void *owner = spec->owner == M_MODEL
        ? (const void *)model
        : (spec->owner == M_CSA ||
           spec->owner == M_CSA_NUM)
            ? (const void *)first_csa : (const void *)first;
    const char *field = owner ? (const char *)owner + spec->offset : NULL;
    unsigned int count = (unsigned int)model->main_layer_count;

    if (!entry || !spec || !model || !first || !first_csa || !count ||
        count > YVEX_ARTIFACT_LOWERING_METADATA_CAP ||
        (!field && spec->owner != M_LIT))
        return 0;
    memset(entry, 0, sizeof(*entry));
    yvex_core_text_copy(entry->key, sizeof(entry->key), spec->key);
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
            if (count > YVEX_ARTIFACT_LOWERING_METADATA_CAP) return 0;
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
    return 1;
}

static int deepseek_lowering_policy_build(
    yvex_artifact_lowering_policy *policy,
    yvex_artifact_lowering_metadata metadata[YVEX_ARTIFACT_LOWERING_METADATA_CAP],
    const yvex_deepseek_v4_ir *architecture)
{
    const model_t *model = lowering_family_ir()->model(architecture);
    const layer_t *first = lowering_family_ir()->layer_at(architecture, 0u);
    const layer_t *first_csa = lowering_family_ir()->layer_at(architecture, 2u);
    unsigned long long ratios[YVEX_ARTIFACT_LOWERING_METADATA_CAP];
    double clamp[YVEX_ARTIFACT_LOWERING_METADATA_CAP];
    unsigned long long index;

    if (!policy || !metadata || !model || !first || !first_csa ||
        model->main_layer_count != 43u || model->auxiliary_layer_count != 3u ||
        model->main_layer_count > YVEX_ARTIFACT_LOWERING_METADATA_CAP)
        return 0;
    memset(policy, 0, sizeof(*policy));
    memset(metadata, 0, YVEX_ARTIFACT_LOWERING_METADATA_CAP * sizeof(*metadata));
    for (index = 0u; index < model->main_layer_count; ++index) {
        const layer_t *layer = lowering_family_ir()->layer_at(architecture, index);
        if (!layer) return 0;
        ratios[index] = layer->compression_ratio;
        clamp[index] = layer->moe.activation_limit;
    }
    for (index = 0u;
         index < sizeof(map_metadata_specs) / sizeof(map_metadata_specs[0]);
         ++index) {
        if (!deepseek_metadata_project(&metadata[index], &map_metadata_specs[index], model,
                                       first, first_csa, ratios, clamp))
            return 0;
    }
    policy->schema_version = YVEX_ARTIFACT_LOWERING_POLICY_SCHEMA_V1;
    policy->source_contribution_count = YVEX_DEEPSEEK_GGUF_SOURCE_COUNT;
    policy->descriptor_count = YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT;
    policy->trunk_descriptor_count = YVEX_DEEPSEEK_GGUF_TRUNK_DESCRIPTOR_COUNT;
    policy->draft_descriptor_count = YVEX_DEEPSEEK_GGUF_DRAFT_DESCRIPTOR_COUNT;
    policy->pinned_standard_count = YVEX_DEEPSEEK_GGUF_TRUNK_DESCRIPTOR_COUNT;
    policy->extension_count = YVEX_DEEPSEEK_GGUF_DRAFT_DESCRIPTOR_COUNT;
    for (index = 0u;
         index < sizeof(map_trunk_expectations) / sizeof(map_trunk_expectations[0]); ++index)
        policy->trunk_collection_counts[map_trunk_expectations[index].collection] =
            map_trunk_expectations[index].count;
    policy->metadata = metadata;
    policy->metadata_count = sizeof(map_metadata_specs) / sizeof(map_metadata_specs[0]);
    return 1;
}


static const yvex_model_family_ir_api *lowering_family_ir(void)
{
    return &yvex_model_register_deepseek_v4()->ir;
}

static int lowering_build(
    yvex_artifact_lowering_map **out,
    const yvex_deepseek_v4_ir *architecture,
    const yvex_transform_ir *transform_ir,
    yvex_artifact_lowering_failure *failure,
    yvex_error *err)
{
    yvex_artifact_lowering_metadata metadata[YVEX_ARTIFACT_LOWERING_METADATA_CAP];
    yvex_artifact_lowering_policy policy;

    memset(&policy, 0, sizeof(policy));
    if (!out || !architecture || !transform_ir)
        return yvex_artifact_lowering_operations.build(out, transform_ir, NULL, failure, err);
    if (!deepseek_lowering_policy_build(&policy, metadata, architecture))
        return yvex_artifact_lowering_operations.build(out, transform_ir, &policy, failure, err);
    return yvex_artifact_lowering_operations.build(out, transform_ir, &policy, failure, err);
}

static int lowering_build_with_allocator(
    yvex_artifact_lowering_map **out,
    const yvex_deepseek_v4_ir *architecture,
    const yvex_transform_ir *transform_ir,
    const yvex_artifact_lowering_allocator *allocator,
    yvex_artifact_lowering_failure *failure,
    yvex_error *err)
{
    yvex_artifact_lowering_metadata metadata[YVEX_ARTIFACT_LOWERING_METADATA_CAP];
    yvex_artifact_lowering_policy policy;

    memset(&policy, 0, sizeof(policy));
    if (!out || !architecture || !transform_ir || !allocator ||
        !allocator->allocate || !allocator->release)
        return yvex_artifact_lowering_operations.build_with_allocator(
            out, transform_ir, NULL, allocator, failure, err);
    if (!deepseek_lowering_policy_build(&policy, metadata, architecture))
        return yvex_artifact_lowering_operations.build_with_allocator(
            out, transform_ir, &policy, allocator, failure, err);
    return yvex_artifact_lowering_operations.build_with_allocator(
        out, transform_ir, &policy, allocator, failure, err);
}

const yvex_model_family_lowering_api *yvex_model_deepseek_lowering_api(void)
{
    static const yvex_model_family_lowering_api api = {
        lowering_build,
        lowering_build_with_allocator,
        &yvex_artifact_lowering_operations
    };

    return &api;
}
