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

static int quant_lowering_summary(const void *context, yvex_quant_lowering_summary *out)
{
    const yvex_artifact_lowering_summary *summary =
        yvex_artifact_lowering_operations.summary((const yvex_artifact_lowering_map *)context);

    if (!summary || !out) return 0;
    *out = (yvex_quant_lowering_summary){
        summary->source_contribution_count,
        summary->descriptor_count,
        summary->source_identity,
        summary->mapping_identity,
        summary->complete};
    return 1;
}

static int quant_lowering_qtypes(yvex_artifact_lowering_transform transform,
                                 unsigned int *source_faithful, unsigned int *release)
{
    if (!source_faithful || !release) return 0;
    switch (transform) {
    case YVEX_ARTIFACT_LOWERING_TRANSFORM_DIRECT:
    case YVEX_ARTIFACT_LOWERING_TRANSFORM_FP8_E4M3_E8M0:
        *source_faithful = YVEX_GGUF_QTYPE_F32;
        *release = YVEX_GGUF_QTYPE_Q8_0;
        return 1;
    case YVEX_ARTIFACT_LOWERING_TRANSFORM_EXPERT_MXFP4:
        *source_faithful = YVEX_GGUF_QTYPE_MXFP4;
        *release = YVEX_GGUF_QTYPE_Q2_K;
        return 1;
    case YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32:
        *source_faithful = YVEX_GGUF_QTYPE_I32;
        *release = YVEX_GGUF_QTYPE_I32;
        return 1;
    }
    return 0;
}

static yvex_transform_operation_kind quant_lowering_operation(
    yvex_artifact_lowering_transform transform)
{
    switch (transform) {
    case YVEX_ARTIFACT_LOWERING_TRANSFORM_DIRECT: return YVEX_TRANSFORM_OP_IDENTITY;
    case YVEX_ARTIFACT_LOWERING_TRANSFORM_FP8_E4M3_E8M0:
        return YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR;
    case YVEX_ARTIFACT_LOWERING_TRANSFORM_EXPERT_MXFP4:
        return YVEX_TRANSFORM_OP_EXPERT_AGGREGATE;
    case YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32: return YVEX_TRANSFORM_OP_CHECKED_CAST;
    }
    return YVEX_TRANSFORM_OP_COUNT;
}

static int quant_lowering_tensor(const void *context, unsigned long long ordinal,
                                 yvex_quant_lowering_tensor *out)
{
    const yvex_artifact_lowering_descriptor *row =
        yvex_artifact_lowering_operations.descriptor_at((const yvex_artifact_lowering_map *)context, ordinal);

    if (!row || !out || row->logical_rank > YVEX_GGUF_QTYPE_MAX_DIMS ||
        row->transform > YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32) return 0;
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
        row->transform == YVEX_ARTIFACT_LOWERING_TRANSFORM_EXPERT_MXFP4 ||
        row->transform == YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32;
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
    const yvex_artifact_lowering_contribution *row =
        yvex_artifact_lowering_operations.contribution_at((const yvex_artifact_lowering_map *)context, ordinal);

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
    const yvex_transform_binding *binding, const yvex_artifact_lowering_map *map,
    yvex_quant_profile_kind profile, const yvex_quant_plan_options *options,
    yvex_quant_failure *failure, yvex_error *err)
{
    return yvex_quant_plan_build_profile(out, ir, binding, deepseek_quant_lowering_api(), map,
                                         profile, options, failure, err);
}

int yvex_quant_plan_build_deepseek_policy(
    yvex_quant_plan **out, const yvex_transform_ir *ir,
    const yvex_transform_binding *binding, const yvex_artifact_lowering_map *map,
    const yvex_quant_policy *policy, const char *imatrix_identity,
    const yvex_quant_plan_options *options, yvex_quant_failure *failure, yvex_error *err)
{
    return yvex_quant_plan_build_policy(out, ir, binding, deepseek_quant_lowering_api(), map,
                                        policy, imatrix_identity, options, failure, err);
}

static int writer_lowering_summary(const void *context,
                                   yvex_gguf_writer_lowering_summary *out)
{
    const yvex_artifact_lowering_summary *summary =
        yvex_artifact_lowering_operations.summary((const yvex_artifact_lowering_map *)context);

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
    const yvex_artifact_lowering_descriptor *row =
        yvex_artifact_lowering_operations.descriptor_at((const yvex_artifact_lowering_map *)context, ordinal);

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
    const yvex_artifact_lowering_metadata *row =
        yvex_artifact_lowering_operations.metadata_at((const yvex_artifact_lowering_map *)context, ordinal);

    if (!row || !out || row->type > YVEX_ARTIFACT_LOWERING_METADATA_F64_ARRAY ||
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
