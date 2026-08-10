/*
 * Compile irreducible DeepSeek model semantics into generic transformation and execution plans.
 * Mutable plan storage and validation remain owned by generic compiler sinks; this projection
 * supplies only source roles, topology, numerical policy, and operator composition.
 */
#include "src/graph/private.h"
#include <yvex/internal/artifact_lowering.h>
#include <yvex/internal/compiler_source.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/family_catalog.h>
#include <yvex/internal/tokenizer.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/operator_graph.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/transformer.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int graph_recipe_project(const yvex_deepseek_v4_layer_spec *layer,
                                unsigned long long ordinal, yvex_tensor_scope scope,
                                unsigned long long predictor,
                                yvex_semantic_attention_layer *out)
{
    const yvex_attention_activation_policy *policies[4];

    if (!layer || !out) return 0;
    policies[0] = &layer->attention_kv_activation;
    policies[1] = &layer->compressor_activation;
    policies[2] = &layer->compressor_rotated_activation;
    policies[3] = &layer->indexer_query_activation;
    if (!yvex_model_attention_numeric_validate(
            layer->compute_contract, YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
            policies, sizeof(policies) / sizeof(policies[0]), &layer->sparse_topk,
            YVEX_DEEPSEEK_V4_RUNTIME_FP8_ACT_BLOCK,
            YVEX_DEEPSEEK_V4_RUNTIME_FP4_ACT_BLOCK,
            YVEX_DEEPSEEK_V4_RUNTIME_TOPK_POLICY_VERSION, NULL))
        return 0;
    memset(out, 0, sizeof(*out));
    out->ordinal = ordinal;
    out->layer_index = layer->layer_index;
    out->predictor_index = predictor;
    out->tensor_scope = scope;
    out->attention_class = layer->attention_class;
    out->compute_contract = layer->compute_contract;
    out->compression_ratio = layer->compression_ratio;
    out->sliding_window = layer->kv.sliding_window;
    out->query_heads = layer->query_heads;
    out->kv_heads = layer->kv_heads;
    out->head_dimension = layer->head_dimension;
    out->rope_head_dimension = layer->rope_head_dimension;
    out->query_lora_rank = layer->query_lora_rank;
    out->output_lora_rank = layer->output_lora_rank;
    out->output_groups = layer->output_groups;
    out->output_group_input_width = layer->output_group_input_width;
    out->hidden_dimension = layer->tensors.q_a_columns;
    out->indexer_heads = layer->indexer_heads;
    out->indexer_head_dimension = layer->indexer_head_dimension;
    out->indexer_topk = layer->indexer_topk;
    out->compressor_ape_columns = layer->tensors.compressor_ape_columns;
    out->indexer_ape_columns = layer->tensors.indexer_ape_columns;
    out->rms_norm_epsilon = layer->rms_norm_epsilon;
    out->residual_stream_count = layer->mhc.residual_streams;
    out->residual_stream_width = layer->mhc.stream_width;
    out->residual_expanded_width = layer->mhc.expanded_width;
    out->mhc_mixing_rows = layer->mhc.mixing_rows;
    out->mhc_mixing_columns = layer->mhc.mixing_columns;
    out->mhc_base_width = layer->mhc.base_width;
    out->mhc_scale_width = layer->mhc.scale_width;
    out->mhc_sinkhorn_iterations = layer->mhc.sinkhorn_iterations;
    out->mhc_epsilon = layer->mhc.epsilon;
    out->mhc_residual_post_multiplier = layer->mhc.residual_post_multiplier;
    out->mhc_entry_policy = (unsigned int)layer->mhc.entry;
    out->mhc_attention_pre_and_post = layer->mhc.attention_pre_and_post;
    out->attention_input_norm_required = layer->attention_input_norm.required;
    out->attention_input_norm_width = layer->attention_input_norm.width;
    out->attention_input_norm_role = YVEX_TENSOR_ROLE_ATTENTION_NORM;
    out->mhc_function_role = YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION;
    out->mhc_base_role = YVEX_TENSOR_ROLE_HC_ATTENTION_BASE;
    out->mhc_scale_role = YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE;
    out->compressor_required = layer->compressor_required;
    out->indexer_required = layer->indexer_required;
    out->position = layer->position;
    out->attention_kv_activation = layer->attention_kv_activation;
    out->compressor_activation = layer->compressor_activation;
    out->compressor_rotated_activation = layer->compressor_rotated_activation;
    out->indexer_query_activation = layer->indexer_query_activation;
    out->sparse_topk = layer->sparse_topk;
    return 1;
}
static int graph_recipe_layer(const void *context, unsigned long long index,
                              yvex_semantic_attention_layer *out)
{
    const yvex_deepseek_v4_ir *ir = context;
    return graph_recipe_project(yvex_model_register_deepseek_v4()->ir.layer_at(ir, index),
                                index, YVEX_TENSOR_SCOPE_MAIN_LAYER,
                                YVEX_ATTENTION_NO_TENSOR_INDEX, out);
}
static int graph_recipe_draft_layer(const void *context, unsigned long long index,
                                    yvex_semantic_attention_layer *out)
{
    const yvex_deepseek_v4_auxiliary_spec *draft =
        yvex_model_register_deepseek_v4()->ir.auxiliary_at(context, index);
    return draft && graph_recipe_project(&draft->layer, index, YVEX_TENSOR_SCOPE_DRAFT,
                                         draft->predictor_index, out);
}
static int graph_plan_build(
    yvex_attention_plan **out, const yvex_semantic_model_ir *semantic_model,
    const yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err)
{
    return yvex_attention_plan_build_semantic(out, semantic_model,
        YVEX_TENSOR_SCOPE_MAIN_LAYER, session, descriptor, failure, err);
}
static int graph_draft_plan_build(
    yvex_attention_plan **out, const yvex_semantic_model_ir *semantic_model,
    const yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err)
{
    return yvex_attention_plan_build_semantic(out, semantic_model,
        YVEX_TENSOR_SCOPE_DRAFT, session, descriptor, failure, err);
}
static const yvex_model_execution_descriptor *deepseek_execution_model(
    const yvex_runtime_descriptor_summary *runtime)
{
    return runtime && runtime->model_execution.schema_version == YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1
               ? &runtime->model_execution : NULL;
}
static int deepseek_moe_layer(unsigned long long index, const yvex_runtime_descriptor_summary *runtime,
                              const yvex_attention_layer_plan *attention, yvex_moe_layer_plan *out,
                              yvex_error *err)
{
    const yvex_model_execution_descriptor *model = deepseek_execution_model(runtime);
    if (!model || !attention || !out || attention->ordinal != index ||
        (attention->tensor_scope == YVEX_TENSOR_SCOPE_MAIN_LAYER &&
         (attention->layer_index >= runtime->layer_count ||
          attention->predictor_index != YVEX_MATERIALIZATION_NO_INDEX)) ||
        (attention->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT &&
         (attention->predictor_index >= runtime->draft_layer_count ||
          attention->layer_index < runtime->layer_count)) ||
        (attention->tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER &&
         attention->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.family.deepseek.moe",
                       "DeepSeek MoE projection requires one ordered admitted layer");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->schema_version = YVEX_MOE_PLAN_SCHEMA_V1; out->ordinal = index;
    out->layer_index = attention->layer_index; out->predictor_index = attention->predictor_index;
    out->tensor_scope = attention->tensor_scope;
    out->router_class = attention->tensor_scope == YVEX_TENSOR_SCOPE_MAIN_LAYER &&
                                attention->layer_index < model->hash_router_layer_count
                            ? YVEX_MOE_ROUTER_HASH_TOKEN_ID :
                                      YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE;
    out->scoring = YVEX_MOE_SCORING_SQRT_SOFTPLUS; out->topk_policy = YVEX_MOE_TOPK_NOAUX_TC;
    out->activation = YVEX_MOE_ACTIVATION_SILU; out->hidden_width = attention->hidden_dimension;
    out->residual_streams = attention->residual_stream_count;
    out->expanded_width = attention->residual_expanded_width; out->mhc_mixing_rows = attention->mhc_mixing_rows;
    out->mhc_sinkhorn_iterations = attention->mhc_sinkhorn_iterations;
    out->rms_epsilon = attention->rms_norm_epsilon; out->mhc_epsilon = attention->mhc_epsilon;
    out->mhc_post_multiplier = attention->mhc_residual_post_multiplier;
    out->routed_experts = runtime->routed_experts; out->shared_experts = model->shared_experts;
    out->experts_per_token = runtime->experts_per_token;
    out->expert_intermediate_width = model->routed_ffn_width;
    out->shared_intermediate_width = model->shared_ffn_width;
    out->hash_table_rows = runtime->vocabulary_size; out->hash_table_columns = out->experts_per_token;
    out->correction_bias_width = out->routed_experts;
    out->routed_scaling_factor = model->routed_scaling_factor; out->activation_limit = model->activation_limit;
    out->requires_token_ids = out->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID;
    out->requires_correction_bias = !out->requires_token_ids; out->normalize_topk_probabilities = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
static const yvex_moe_family_api deepseek_moe_api = {
    .adapter_id = YVEX_DEEPSEEK_V4_ADAPTER_ID, .adapter_version = YVEX_DEEPSEEK_V4_ADAPTER_VERSION,
    .project_layer = deepseek_moe_layer};
static const yvex_graph_compiler_api deepseek_graph_compiler = {
    .plan_build = graph_plan_build,
    .draft_plan_build = graph_draft_plan_build,
    .moe = &deepseek_moe_api};
static const yvex_graph_compiler_api *deepseek_graph_compile(void) {
    return &deepseek_graph_compiler;
}
static int deepseek_execution_capabilities(yvex_runtime_capabilities *out) {
    if (!out) return 0;
    *out = (yvex_runtime_capabilities){
        .attention_semantics_ready = 1, .attention_core_ready = 1,
        .attention_envelope_ready = 1, .cpu_prefill_eager_ready = 1,
        .cpu_decode_eager_ready = 1, .cuda_eager_implemented = 1,
        .cuda_piecewise_graph_implemented = 1, .cuda_full_graph_implemented = 1,
        .attention_state_delta_ready = 1, .attention_operator_ready = 1,
        .attention_trace_ready = 1, .attention_profile_ready = 1,
        .attention_benchmark_ready = 1, .moe_plan_ready = 1, .moe_router_ready = 1,
        .moe_routed_expert_ready = 1,
        .moe_shared_expert_ready = 1, .moe_block_ready = 1, .transformer_ready = 1,
        .output_head_binding_ready = 1, .output_head_projection_ready = 1,
        .logits_cpu_ready = 1, .logits_cuda_ready = 1, .logits_prefill_ready = 1,
        .logits_decode_ready = 1, .logits_full_vocabulary_ready = 1,
        .logits_hidden_contract_ready = 1, .logits_partial_progress_ready = 1, .logits_ready = 1};
    return yvex_runtime_capabilities_contract_valid(out);
}
static int deepseek_transformer_policy(const yvex_runtime_descriptor_summary *runtime,
                                       yvex_transformer_family_policy *out) {
    const yvex_model_execution_descriptor *model = deepseek_execution_model(runtime);
    unsigned long long expanded_width;
    if (!model || !out || !yvex_core_u64_mul(model->residual_streams, model->hidden_width,
                                              &expanded_width)) return 0;
    *out = (yvex_transformer_family_policy){
        .schema_version = YVEX_TRANSFORMER_PLAN_SCHEMA_V2,
        .initial_policy = YVEX_TRANSFORMER_INITIAL_REPEAT_STREAMS,
        .final_policy = YVEX_TRANSFORMER_FINAL_SIGMOID_MHC_RMS,
        .residual_streams = model->residual_streams, .hidden_width = model->hidden_width,
        .expanded_width = expanded_width, .maximum_context = model->maximum_context,
        .sinkhorn_iterations = model->mhc_sinkhorn_iterations, .mhc_epsilon = model->mhc_epsilon,
        .output_norm_epsilon = model->normalization_epsilon,
        .attention_then_moe = 1, .deferred_ffn_post = 1, .final_norm_after_head = 1};
    return 1;
}
static int deepseek_logits_policy(yvex_logits_family_policy *out) {
    if (!out) return 0;
    *out = (yvex_logits_family_policy){
        .schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1,
        .separate_output_head = 1, .tied_output_head = 0, .output_head_bias = 0};
    return 1;
}
static int deepseek_speculation_policy(const yvex_runtime_descriptor_summary *runtime,
                                       yvex_speculation_family_policy *out) {
    const yvex_model_execution_descriptor *model = deepseek_execution_model(runtime);
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, feature_width;
    if (!model || !out || model->target_feature_count > YVEX_SPECULATION_MAX_FEATURE_LAYERS ||
        !yvex_core_u64_mul(model->target_feature_count, model->target_feature_width,
                           &feature_width)) return 0;
    *out = (yvex_speculation_family_policy){
        .schema_version = YVEX_SPECULATION_FAMILY_POLICY_SCHEMA_V1,
        .block_size = model->proposal_width, .noise_token_id = model->draft_noise_token_id,
        .target_feature_layer_count = model->target_feature_count, .target_feature_width = model->target_feature_width,
        .concatenated_feature_width = feature_width,
        .draft_layer_count = model->draft_layer_count, .markov_rank = model->markov_rank,
        .accepted_prefix_maximum = model->proposal_width,
        .feature_projection_role = YVEX_TENSOR_ROLE_DRAFT_FEATURE_PROJECTION,
        .feature_norm_role = YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM,
        .output_norm_role = YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM,
        .markov_embedding_role = YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING,
        .markov_output_role = YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT,
        .confidence_role = YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE,
        .parallel_block_backbone = 1, .sequential_markov = 1, .confidence_available = 1,
        .shares_embedding = 1, .shares_output_head = 1, .target_verification_required = 1};
    memcpy(out->target_feature_layers, model->target_feature_layers, sizeof(out->target_feature_layers));
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.deepseek-v4.dspark.policy.v1") ||
        !yvex_sha256_update_u64(&hash, out->block_size) ||
        !yvex_sha256_update_u64(&hash, out->noise_token_id) ||
        !yvex_sha256_update_u64(&hash, out->target_feature_layer_count) ||
        !yvex_sha256_update_u64(&hash, out->target_feature_width) ||
        !yvex_sha256_update_u64(&hash, out->concatenated_feature_width) ||
        !yvex_sha256_update_u64(&hash, out->draft_layer_count) ||
        !yvex_sha256_update_u64(&hash, out->markov_rank)) return 0;
    for (index = 0ull; index < out->target_feature_layer_count; ++index)
        if (!yvex_sha256_update_u64(&hash, out->target_feature_layers[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, out->policy_identity);
    return 1;
}
static int deepseek_tokenizer_policy(yvex_tokenizer_family_policy *out, yvex_error *err) {
    return yvex_tokenizer_family_policy_compile(
        out, yvex_model_deepseek_v4_conversation(), YVEX_TOKENIZER_KIND_GGML_GPT2,
        YVEX_TOKENIZER_MODEL_BPE_BYTELEVEL, YVEX_TOKENIZER_PROMPT_CONVERSATION, err) == YVEX_OK;
}
static const yvex_family_compiler_adapter deepseek_compiler;
static const yvex_graph_execution_binding deepseek_execution = {
    .schema_version = YVEX_GRAPH_EXECUTION_BINDING_SCHEMA_V1,
    .adapter_id = YVEX_DEEPSEEK_V4_ADAPTER_ID,
    .adapter_version = YVEX_DEEPSEEK_V4_ADAPTER_VERSION,
    .target_id = "deepseek4-v4-flash-dspark", .family_name = "deepseek-v4-flash-dspark",
    .logical_transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .operator_family_key = "deepseek",
    .operator_artifact_filename = YVEX_SELECTED_DEEPSEEK_ARTIFACT_FILENAME,
    .source_manifest_filename = YVEX_SOURCE_RELEASE_MANIFEST_LEAF,
    .model = yvex_model_register_deepseek_v4,
    .compiler = &deepseek_compiler,
    .api = &yvex_attention_execution_api};
static int deepseek_compilation_source_open(
    yvex_family_compilation_source *out,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err);
static void deepseek_compilation_source_close(void *owner);
static int deepseek_compilation_semantic_model(
    yvex_semantic_model_ir **out,
    const yvex_source_verification *verification, yvex_error *err);
static int deepseek_compilation_descriptor(
    yvex_runtime_descriptor **out, const yvex_complete_artifact_admission *admission,
    yvex_materialization_session *materialization, const void *lowering_context,
    const yvex_semantic_model_ir *semantic_model, yvex_error *err);
static int deepseek_compilation_quant_default(
    yvex_quant_plan **out, const yvex_transform_ir *transform,
    const yvex_transform_binding *binding, const void *lowering_context,
    yvex_error *err);
static int deepseek_compilation_quant_policy(
    yvex_quant_plan **out, const yvex_transform_ir *transform,
    const yvex_transform_binding *binding, const void *lowering_context,
    const yvex_quant_policy *policy, const char *imatrix_identity,
    yvex_error *err);
static const yvex_physical_execution_policy deepseek_physical_execution_policy = {
    .schema_version = YVEX_PHYSICAL_EXECUTION_POLICY_SCHEMA_V1,
    .activation = YVEX_EXECUTION_ACTIVATION_DEVICE_F32,
    .required_backend = YVEX_EXECUTION_BACKEND_ANY,
    .evidence = YVEX_EXECUTION_EVIDENCE_PRODUCTION,
    .fallback = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE,
    .dense_kernel_family = "portable-encoded-row",
    .expert_kernel_family = "portable-expert-row"};
static const yvex_family_binding_pipeline deepseek_binding_pipeline = {
    .schema_version = YVEX_FAMILY_BINDING_PIPELINE_SCHEMA_V1,
    .source_open = deepseek_compilation_source_open,
    .source_close = deepseek_compilation_source_close,
    .artifact_admit = yvex_artifact_admit_deepseek,
    .semantic_model_build = deepseek_compilation_semantic_model,
    .runtime_descriptor_build = deepseek_compilation_descriptor,
    .quant_plan_default = deepseek_compilation_quant_default,
    .quant_plan_policy = deepseek_compilation_quant_policy,
    .tokenizer_architecture = "deepseek-v3",
    .imatrix_source_identity = YVEX_DEEPSEEK_QUANT_IMATRIX_SOURCE_IDENTITY,
    .imatrix_dataset_identity = YVEX_DEEPSEEK_QUANT_IMATRIX_DATASET_IDENTITY,
    .imatrix_producer = "llama.cpp-imatrix",
    .imatrix_producer_version = 1u};
static const yvex_family_compiler_adapter deepseek_compiler = {
    .schema_version = YVEX_FAMILY_COMPILER_SCHEMA_V2,
    .adapter_id = YVEX_DEEPSEEK_V4_ADAPTER_ID,
    .adapter_version = YVEX_DEEPSEEK_V4_ADAPTER_VERSION,
    .target_id = "deepseek4-v4-flash-dspark",
    .family = "deepseek-v4",
    .logical_transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .physical_execution_policy = &deepseek_physical_execution_policy,
    .graph = deepseek_graph_compile,
    .operator_graph_build = yvex_operator_graph_ir_build_transformer,
    .execution_capabilities = deepseek_execution_capabilities,
    .transformer_policy = deepseek_transformer_policy,
    .logits_policy = deepseek_logits_policy,
    .speculation_policy = deepseek_speculation_policy,
    .tokenizer_policy = deepseek_tokenizer_policy,
    .physical_variant = yvex_graph_physical_variant_api_get,
    .binding_pipeline = &deepseek_binding_pipeline,
    .binding_compile = yvex_family_binding_compile};
const yvex_family_compiler_adapter *yvex_compiler_family_deepseek_v4(void) {
    return &deepseek_compiler;
}
const yvex_graph_execution_binding *yvex_graph_deepseek_v4_execution_binding(void)
{
    return &deepseek_execution;
}
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

typedef struct {
    yvex_transform_recipe_sink *sink;
    const yvex_model_family_api *family;
    const yvex_deepseek_v4_ir *architecture;
    const yvex_deepseek_v4_model_spec *model;
    yvex_source_tensor_snapshot_facts source_facts;
    unsigned long long requirement_cursor;
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
                               const yvex_transform_shape *shape,
                               unsigned long long requirement_index,
                               yvex_transform_source_spec *spec)
{
    yvex_transform_source_requirement requirement;

    if (!shape || !shape->rank)
        return deepseek_refuse(builder, YVEX_TRANSFORM_FAILURE_INVALID_SHAPE,
                               1u, 0u, "deepseek_transform_source");
    memset(&requirement, 0, sizeof(requirement));
    requirement.source_name = name;
    requirement.requirement_index = requirement_index;
    requirement.source_dtype = expected_dtype;
    requirement.value_dtype = deepseek_dtype(expected_dtype, packed_fp4);
    requirement.shape = *shape;
    requirement.scope = deepseek_scope(scope);
    requirement.subsystem = deepseek_subsystem(collection);
    requirement.role_hint = role;
    requirement.layer_index = layer;
    requirement.auxiliary_index = auxiliary;
    requirement.expert_index = expert;
    requirement.required_uses = 1u;
    return yvex_transform_recipe_sink_resolve_source(
        builder->sink, &requirement, spec, builder->failure, builder->err);
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
                               int checked_cast,
                               const yvex_transform_shape *expected_shape)
{
    yvex_transform_precision_constraint precision;
    yvex_transform_node_spec node;
    yvex_transform_shape shape;
    yvex_transform_source_spec source;
    yvex_transform_dtype output_dtype;
    int rc;

    rc = deepseek_add_source(
        builder, source_name, role, collection, scope, layer, auxiliary,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, source_dtype, 0, expected_shape,
        builder->requirement_cursor++, &source);
    if (rc != YVEX_OK) return rc;
    shape = *expected_shape;
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
                            const char *base,
                            unsigned long long rows,
                            unsigned long long columns)
{
    char weight[256];
    char scale[256];
    yvex_transform_precision_constraint precision;
    yvex_transform_node_spec node;
    yvex_transform_shape shape;
    yvex_transform_source_spec sources[2];
    yvex_transform_shape scale_shape;
    int rc;

    (void)snprintf(weight, sizeof(weight), "%s.weight", base);
    (void)snprintf(scale, sizeof(scale), "%s.scale", base);
    memset(&shape, 0, sizeof(shape));
    shape.rank = 2u;
    shape.dims[0] = rows;
    shape.dims[1] = columns;
    scale_shape = shape;
    if (!builder->model->source_constraint.quant_block_rows ||
        !builder->model->source_constraint.quant_block_columns ||
        rows % builder->model->source_constraint.quant_block_rows != 0u ||
        columns % builder->model->source_constraint.quant_block_columns != 0u)
        return deepseek_refuse(builder, YVEX_TRANSFORM_FAILURE_INVALID_SHAPE,
                               1u, 0u, "deepseek_transform_fp8");
    scale_shape.dims[0] /= builder->model->source_constraint.quant_block_rows;
    scale_shape.dims[1] /= builder->model->source_constraint.quant_block_columns;
    rc = deepseek_add_source(
        builder, weight, role, collection, scope, layer, auxiliary,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_F8_E4M3, 0, &shape,
        builder->requirement_cursor++, &sources[0]);
    if (rc == YVEX_OK)
        rc = deepseek_add_source(
            builder, scale, role, collection, scope, layer, auxiliary,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_NATIVE_DTYPE_F8_E8M0, 0,
            &scale_shape, builder->requirement_cursor++, &sources[1]);
    if (rc != YVEX_OK) return rc;
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
                                unsigned long long expert_count,
                                unsigned long long expert_intermediate_size,
                                unsigned int projection_index,
                                unsigned long long requirement_base)
{
    char weight[256];
    char scale[256];
    yvex_transform_precision_constraint precision;
    yvex_transform_node_spec node;
    yvex_transform_shape shape;
    yvex_transform_source_spec *sources = NULL;
    unsigned long long input_count;
    unsigned long long logical_width;
    unsigned long long rows;
    unsigned long long columns;
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
    rows = projection_index == 1u
        ? builder->model->hidden_size
        : expert_intermediate_size;
    columns = projection_index == 1u
        ? expert_intermediate_size
        : builder->model->hidden_size;
    if (!builder->model->source_constraint.fp4_packing_factor ||
        !builder->model->source_constraint.fp4_scale_group_width ||
        columns % builder->model->source_constraint.fp4_packing_factor != 0u ||
        columns % builder->model->source_constraint.fp4_scale_group_width != 0u) {
        rc = deepseek_refuse(builder, YVEX_TRANSFORM_FAILURE_INVALID_SHAPE,
                             1u, columns, "deepseek_transform_experts");
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
                &(yvex_transform_shape){
                    .rank = 2u, .dims = {rows, columns /
                        builder->model->source_constraint.fp4_packing_factor}},
                requirement_base + expert * 6u + projection_index * 2u,
                &sources[expert * 2u]);
        if (rc == YVEX_OK)
            rc = deepseek_add_source(
                builder, scale, role,
                YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, scope, layer,
                auxiliary, expert, YVEX_NATIVE_DTYPE_F8_E8M0, 0,
                &(yvex_transform_shape){
                    .rank = 2u, .dims = {rows, columns /
                        builder->model->source_constraint.fp4_scale_group_width}},
                requirement_base + expert * 6u + projection_index * 2u + 1u,
                &sources[expert * 2u + 1u]);
        if (rc != YVEX_OK) goto cleanup;
    }
    logical_width = columns;
    memset(&shape, 0, sizeof(shape));
    shape.rank = 3u;
    shape.dims[0] = expert_count;
    shape.dims[1] = rows;
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
        yvex_transform_shape shape = {0};
        int rc;

        if (!recipe || recipe->phase != phase ||
            !family_ir->recipe_enabled(recipe, layer))
            continue;
        (void)snprintf(name, sizeof(name), "%s.%s", prefix, recipe->suffix);
        shape.rank = recipe->rank;
        shape.dims[0] = family_ir->recipe_dimension(
            recipe, 0u, layer, builder->model);
        if (shape.rank == 2u)
            shape.dims[1] = family_ir->recipe_dimension(
                recipe, 1u, layer, builder->model);
        if (recipe->kind == YVEX_DEEPSEEK_RECIPE_FP8_PAIR) {
            rc = deepseek_add_fp8(builder, recipe->role, recipe->collection, scope,
                                  layer->layer_index, auxiliary, name,
                                  shape.dims[0], shape.dims[1]);
        } else {
            rc = deepseek_add_direct(builder, recipe->role, recipe->collection, scope,
                                     layer->layer_index, auxiliary, name, recipe->dtype,
                                     recipe->kind == YVEX_DEEPSEEK_RECIPE_CHECKED_CAST,
                                     &shape);
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
    unsigned long long expert_base;
    int rc;

    rc = deepseek_add_recipe_phase(builder, prefix, layer, scope, auxiliary, 0u);
    expert_base = builder->requirement_cursor;
    for (index = 0u;
         rc == YVEX_OK &&
         index < sizeof(deepseek_expert_projections) /
                     sizeof(deepseek_expert_projections[0]);
         ++index) {
        rc = deepseek_add_experts(builder, deepseek_expert_projections[index].role,
                                  scope, layer->layer_index, auxiliary, prefix,
                                  deepseek_expert_projections[index].projection,
                                  layer->moe.routed_experts,
                                  layer->moe.expert_intermediate_size,
                                  (unsigned int)index, expert_base);
    }
    if (rc == YVEX_OK)
        builder->requirement_cursor = expert_base + layer->moe.routed_experts * 6u;
    if (rc == YVEX_OK)
        rc = deepseek_add_recipe_phase(builder, prefix, layer, scope, auxiliary, 1u);
    return rc;
}

static yvex_transform_shape deepseek_shape(
    unsigned int rank, unsigned long long first, unsigned long long second)
{
    yvex_transform_shape shape = {.rank = rank, .dims = {first, second}};

    return shape;
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
    yvex_transform_shape shape;
    unsigned long long layer;
    unsigned int index;
    int rc;

    builder->sink = sink;
    builder->failure = failure;
    builder->err = err;

    shape = deepseek_shape(2u, builder->model->embedding.vocabulary_size,
                           builder->model->embedding.hidden_size);
    rc = deepseek_add_direct(
        builder, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
        YVEX_TENSOR_COLLECTION_GLOBAL,
        YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, "embed.weight",
        YVEX_NATIVE_DTYPE_BF16, 0, &shape);
    if (rc != YVEX_OK) return rc;
    shape = deepseek_shape(1u, builder->model->hidden_size, 0u);
    rc = deepseek_add_direct(
        builder, YVEX_TENSOR_ROLE_OUTPUT_NORM,
        YVEX_TENSOR_COLLECTION_GLOBAL,
        YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, "norm.weight",
        YVEX_NATIVE_DTYPE_BF16, 0, &shape);
    if (rc != YVEX_OK) return rc;
    shape = deepseek_shape(2u, builder->model->output.vocabulary_size,
                           builder->model->output.input_width);
    rc = deepseek_add_direct(
        builder, YVEX_TENSOR_ROLE_OUTPUT_HEAD,
        YVEX_TENSOR_COLLECTION_GLOBAL,
        YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
        YVEX_DEEPSEEK_TENSOR_NO_INDEX, "head.weight",
        YVEX_NATIVE_DTYPE_BF16, 0, &shape);
    if (rc != YVEX_OK) return rc;
    for (index = 0u; index < 3u; ++index) {
        shape = deepseek_shape(
            index == 0u ? 2u : 1u,
            index == 0u ? builder->model->final_mhc_head.function_rows
                        : (index == 1u
                               ? builder->model->final_mhc_head.base_width
                               : builder->model->final_mhc_head.scale_width),
            index == 0u ? builder->model->final_mhc_head.function_columns : 0u);
        rc = deepseek_add_direct(
            builder, head_roles[index],
            YVEX_TENSOR_COLLECTION_GLOBAL,
            YVEX_TENSOR_SCOPE_GLOBAL,
            YVEX_DEEPSEEK_TENSOR_NO_INDEX, YVEX_DEEPSEEK_TENSOR_NO_INDEX,
            head_names[index], YVEX_NATIVE_DTYPE_F32, 0, &shape);
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
#define DRAFT_DIRECT(role_id, suffix, expected_shape) do {                     \
    (void)snprintf(name, sizeof(name), "%s.%s", prefix, suffix);             \
    shape = (expected_shape);                                                   \
    rc = deepseek_add_direct(                                                   \
        builder, role_id, YVEX_TENSOR_COLLECTION_AUXILIARY,          \
        YVEX_TENSOR_SCOPE_DRAFT, aux->layer.layer_index, layer, name,   \
        YVEX_NATIVE_DTYPE_BF16, 0, &shape);                                     \
    if (rc != YVEX_OK) return rc;                                               \
} while (0)
        if (aux->has_feature_projection) {
            (void)snprintf(base, sizeof(base), "%s.main_proj", prefix);
            rc = deepseek_add_fp8(
                builder, YVEX_TENSOR_ROLE_DRAFT_FEATURE_PROJECTION,
                YVEX_TENSOR_COLLECTION_AUXILIARY,
                YVEX_TENSOR_SCOPE_DRAFT, aux->layer.layer_index, layer, base,
                aux->feature_projection_output, aux->feature_projection_input);
            if (rc != YVEX_OK) return rc;
        }
        if (aux->has_feature_norm)
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM, "main_norm.weight",
                         deepseek_shape(1u, aux->feature_norm_width, 0u));
        if (aux->has_output_norm)
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM, "norm.weight",
                         deepseek_shape(1u, aux->output_norm_width, 0u));
        if (aux->has_markov_head) {
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING,
                         "markov_head.markov_w1.weight",
                         deepseek_shape(2u, aux->markov_vocabulary_size,
                                        aux->markov_rank));
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT,
                         "markov_head.markov_w2.weight",
                         deepseek_shape(2u, aux->markov_vocabulary_size,
                                        aux->markov_rank));
        }
        if (aux->has_confidence_head)
            DRAFT_DIRECT(YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE,
                         "confidence_head.proj.weight",
                         deepseek_shape(2u, aux->confidence_output_width,
                                        aux->confidence_input_width));
#undef DRAFT_DIRECT
        for (index = 0u; aux->has_separate_mhc_head && index < 3u; ++index) {
            (void)snprintf(name, sizeof(name), "%s.hc_head_%s", prefix,
                           index == 0u ? "fn" :
                           (index == 1u ? "base" : "scale"));
            shape = deepseek_shape(
                index == 0u ? 2u : 1u,
                index == 0u ? aux->mhc_head.function_rows
                            : (index == 1u ? aux->mhc_head.base_width
                                           : aux->mhc_head.scale_width),
                index == 0u ? aux->mhc_head.function_columns : 0u);
            rc = deepseek_add_direct(
                builder, head_roles[index],
                YVEX_TENSOR_COLLECTION_AUXILIARY,
                YVEX_TENSOR_SCOPE_DRAFT, aux->layer.layer_index,
                layer, name, YVEX_NATIVE_DTYPE_F32, 0, &shape);
            if (rc != YVEX_OK) return rc;
        }
    }
    return YVEX_OK;
}

static int deepseek_validate_inputs(
    deepseek_transform_builder *builder,
    const yvex_source_verification *verification,
    const yvex_deepseek_v4_ir *architecture,
    yvex_source_tensor_snapshot *snapshot,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_deepseek_v4_model_spec *model =
        family->ir.model(architecture);
    int rc;

    memset(builder, 0, sizeof(*builder));
    builder->failure = failure;
    builder->err = err;
    if (!verification || !architecture || !snapshot || !model)
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
            1u, 0u, "deepseek_transform_build");
    rc = yvex_source_tensor_snapshot_facts_get(
        snapshot, &builder->source_facts, err);
    if (rc != YVEX_OK)
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH,
            verification->source_snapshot_identity, 0u,
            "deepseek_transform_build");
    if (!verification->verified || verification->blocker_count != 0u ||
        model->main_layer_count != 43u ||
        model->auxiliary_layer_count != 3u) {
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_ARCHITECTURE_NOT_ADMITTED,
            46u, model->main_layer_count + model->auxiliary_layer_count,
            "deepseek_transform_build");
    }
    if (builder->source_facts.tensor_count !=
            YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT ||
        builder->source_facts.header_scan_count != 1u ||
        builder->source_facts.payload_bytes_read != 0u) {
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_COVERAGE_INCOMPLETE,
            YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT,
            builder->source_facts.tensor_count, "deepseek_transform_build");
    }
    if (verification->source_snapshot_identity != builder->source_facts.identity)
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH,
            verification->source_snapshot_identity, builder->source_facts.identity,
            "deepseek_transform_build");
    if (!verification->manifest_payload_trusted ||
        !yvex_sha256_hex_valid(verification->manifest_payload_identity) ||
        verification->manifest_payload_source_snapshot_identity !=
            builder->source_facts.identity ||
        (strcmp(verification->manifest_payload_trust_class,
                "upstream_payload_verified") != 0 &&
         strcmp(verification->manifest_payload_trust_class,
                "local_payload_snapshot_sealed") != 0)) {
        return deepseek_refuse(
            builder, YVEX_TRANSFORM_FAILURE_PAYLOAD_IDENTITY_MISMATCH,
            builder->source_facts.identity,
            verification->manifest_payload_source_snapshot_identity,
            "deepseek_transform_build");
    }
    builder->family = family;
    builder->architecture = architecture;
    builder->model = model;
    return YVEX_OK;
}

static int deepseek_transform_build(
    yvex_transform_ir **out,
    const yvex_source_verification *verification,
    const yvex_deepseek_v4_ir *architecture,
    yvex_source_tensor_snapshot *snapshot,
    const yvex_transform_builder_options *options,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    deepseek_transform_builder deepseek;
    yvex_transform_header header;
    yvex_transform_builder_options actual = {0};
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
                                  snapshot, failure, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_transform_deepseek_architecture_identity(
            architecture, logical_identity))
        return deepseek_refuse(
            &deepseek, YVEX_TRANSFORM_FAILURE_IDENTITY_ENCODING,
            1u, 0u, "deepseek_transform_architecture_identity");
    deepseek.temporary_allocator.allocate = deepseek_default_allocate;
    deepseek.temporary_allocator.release = deepseek_default_release;
    deepseek.temporary_allocator.context = NULL;
    if (options) actual = *options;
    actual.source_snapshot = snapshot;
    if (actual.allocator.allocate)
        deepseek.temporary_allocator = actual.allocator;
    memset(&header, 0, sizeof(header));
    header.schema_version = YVEX_TRANSFORM_IR_SCHEMA_VERSION;
    header.logical_model_identity = logical_identity;
    header.source_snapshot_identity = deepseek.source_facts.identity;
    header.required_payload_identity =
        verification->manifest_payload_identity;
    header.payload_trust_class = verification->manifest_payload_trust_class;
    header.expected_source_count = YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT;
    header.expected_terminal_count = YVEX_DEEPSEEK_TRANSFORM_TERMINAL_COUNT;
    header.header_scan_count = deepseek.source_facts.header_scan_count;
    rc = yvex_transform_recipe_compile(
        out, &header, deepseek_build_graph, &deepseek, &actual, failure, err);
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

const yvex_model_family_transform_api *yvex_model_deepseek_transform_api(void)
{
    static const yvex_model_family_transform_api api = {
        yvex_transform_deepseek_architecture_identity,
        deepseek_transform_build
    };
    return &api;
}
static int deepseek_source_lower(
    yvex_transform_ir **transform, yvex_artifact_lowering_map **lowering,
    const yvex_source_verification *verification,
    yvex_source_tensor_snapshot *snapshot,
    yvex_compilation_source_failure *failure, yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    yvex_deepseek_v4_ir_failure semantic_failure = {0};
    yvex_artifact_lowering_failure lowering_failure = {0};
    yvex_transform_failure transform_failure = {0};
    yvex_deepseek_v4_ir *semantic = NULL;
    int rc;

    if (transform) *transform = NULL;
    if (lowering) *lowering = NULL;
    rc = family->ir.build(&semantic, verification, &semantic_failure, err);
    if (rc != YVEX_OK) {
        if (failure) failure->code = YVEX_COMPILATION_SOURCE_FAILURE_SEMANTIC_MODEL;
        family->ir.close(semantic);
        return rc;
    }
    rc = family->transform.build(
        transform, verification, semantic, snapshot, NULL,
        &transform_failure, err);
    if (rc != YVEX_OK) {
        if (failure) failure->code = YVEX_COMPILATION_SOURCE_FAILURE_TRANSFORM_IR;
        family->ir.close(semantic);
        return rc;
    }
    rc = family->lowering.build(
        lowering, semantic, *transform, &lowering_failure, err);
    family->ir.close(semantic);
    if (rc != YVEX_OK && failure)
        failure->code = YVEX_COMPILATION_SOURCE_FAILURE_LOWERING;
    return rc;
}

static const void *deepseek_source_identity(void)
{
    return yvex_source_release_identity();
}

static const yvex_compilation_source_projection deepseek_source_projection = {
    .schema_version = YVEX_COMPILATION_SOURCE_PROJECTION_SCHEMA_V1,
    .expected_mapping_identity = YVEX_DEEPSEEK_PAYLOAD_MAPPING_IDENTITY,
    .required_contribution_mask =
        YVEX_COMPILATION_SOURCE_REQUIRE_DIRECT |
        YVEX_COMPILATION_SOURCE_REQUIRE_FP8_WEIGHT |
        YVEX_COMPILATION_SOURCE_REQUIRE_E8M0_SCALE |
        YVEX_COMPILATION_SOURCE_REQUIRE_EXPERT |
        YVEX_COMPILATION_SOURCE_REQUIRE_I64_ROUTER |
        YVEX_COMPILATION_SOURCE_REQUIRE_GLOBAL |
        YVEX_COMPILATION_SOURCE_REQUIRE_NORM |
        YVEX_COMPILATION_SOURCE_REQUIRE_SHARED_EXPERT |
        YVEX_COMPILATION_SOURCE_REQUIRE_OUTPUT_HEAD |
        YVEX_COMPILATION_SOURCE_REQUIRE_DRAFT,
    .source_identity = deepseek_source_identity,
    .lower = deepseek_source_lower,
    .lowering = &yvex_artifact_lowering_operations};

static int payload_open(
    yvex_deepseek_payload_handoff **out,
    const yvex_deepseek_payload_handoff_options *options,
    yvex_deepseek_payload_failure *failure, yvex_error *err)
{
    return yvex_compilation_source_operations.open(
        out, options, &deepseek_source_projection, failure, err);
}

static void payload_close(yvex_deepseek_payload_handoff *handoff)
{
    yvex_compilation_source_operations.close(handoff);
}

static const yvex_deepseek_payload_handoff_summary *payload_summary(
    const yvex_deepseek_payload_handoff *handoff)
{
    return yvex_compilation_source_operations.summary(handoff);
}

static const yvex_source_verification *payload_verification(
    const yvex_deepseek_payload_handoff *handoff)
{
    return yvex_compilation_source_operations.verification(handoff);
}

static const yvex_artifact_lowering_map *payload_map(
    const yvex_deepseek_payload_handoff *handoff)
{
    return yvex_compilation_source_operations.lowering(handoff);
}

static const yvex_transform_ir *payload_transform(
    const yvex_deepseek_payload_handoff *handoff)
{
    return yvex_compilation_source_operations.transform(handoff);
}

static const yvex_transform_binding *payload_binding(
    const yvex_deepseek_payload_handoff *handoff)
{
    return yvex_compilation_source_operations.binding(handoff);
}

static yvex_source_payload_session *payload_session(
    yvex_deepseek_payload_handoff *handoff)
{
    return yvex_compilation_source_operations.payload(handoff);
}

static const yvex_source_payload_plan *payload_plan(
    const yvex_deepseek_payload_handoff *handoff)
{
    return yvex_compilation_source_operations.plan(handoff);
}

static const char *payload_failure_name(yvex_deepseek_payload_failure_code code)
{
    return yvex_compilation_source_operations.failure_name(code);
}

const yvex_model_family_payload_api *yvex_model_deepseek_payload_api(void)
{
    static const yvex_model_family_payload_api api = {
        payload_open,
        payload_close,
        payload_summary,
        payload_verification,
        payload_map,
        payload_transform,
        payload_binding,
        payload_session,
        payload_plan,
        payload_failure_name};
    return &api;
}

static int deepseek_compilation_source_open(
    yvex_family_compilation_source *out,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err)
{
    const yvex_model_family_api *model = yvex_model_register_deepseek_v4();
    const yvex_model_family_payload_api *payload = model ? &model->payload : NULL;
    yvex_deepseek_payload_handoff_options options = {0};
    yvex_deepseek_payload_failure failure = {0};
    yvex_deepseek_payload_handoff *handoff = NULL;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    if (!out || !request || !payload || !payload->open || !payload->close) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "deepseek.compilation-source",
                       "source output, request, and family payload API are required");
        return YVEX_ERR_INVALID_ARG;
    }
    options.source_path = request->source_path;
    options.models_root = request->models_root;
    options.manifest_path = request->source_manifest_path;
    yvex_source_payload_budget_default(&options.budget);
    options.budget.maximum_open_handles = 32u;
    options.budget.maximum_streams = 16u;
    options.budget.maximum_inflight_host_bytes =
        options.budget.chunk_bytes * options.budget.maximum_streams;
    options.chunk_bytes = options.budget.chunk_bytes;
    options.page_bytes = options.budget.page_bytes;
    rc = payload->open(&handoff, &options, &failure, err);
    if (rc != YVEX_OK) return rc;
    out->owner = handoff;
    out->verification = payload->verification(handoff);
    out->transform_ir = payload->transform_ir(handoff);
    out->transform_binding = payload->binding(handoff);
    out->lowering_context = payload->map(handoff);
    if (!out->verification || !out->transform_ir || !out->transform_binding ||
        !out->lowering_context) {
        payload->close(handoff);
        memset(out, 0, sizeof(*out));
        yvex_error_set(err, YVEX_ERR_STATE, "deepseek.compilation-source",
                       "family payload did not project complete compiler inputs");
        return YVEX_ERR_STATE;
    }
    return YVEX_OK;
}
static void deepseek_compilation_source_close(void *owner)
{
    yvex_model_register_deepseek_v4()->payload.close(
        (yvex_deepseek_payload_handoff *)owner);
}
static int deepseek_descriptor_refuse(
    yvex_runtime_descriptor_failure *failure,
    yvex_runtime_descriptor_failure_code code, unsigned long long index,
    const char *reason, yvex_status status, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->tensor_index = index;
        failure->expected = 1ull;
        failure->reason = reason;
    }
    yvex_error_set(err, status, "deepseek.runtime-descriptor", reason);
    return status;
}

static int deepseek_numeric_contract_build(
    const yvex_deepseek_v4_ir *ir,
    yvex_semantic_numeric_contract *contract,
    yvex_runtime_descriptor_failure *failure, yvex_error *err)
{
    const yvex_model_family_api *api = yvex_model_register_deepseek_v4();
    const yvex_deepseek_v4_model_spec *model = api->ir.model(ir);
    const yvex_deepseek_v4_layer_spec *first = api->ir.layer_at(ir, 0ull);
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    unsigned long long count, index;

    if (!contract || !model || !first ||
        model->runtime_numeric_schema_version != 2u ||
        model->runtime_compute_policy_count != 1ull ||
        !model->runtime_activation_policy_count || !model->hadamard_revision[0])
        return deepseek_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE,
            YVEX_MATERIALIZATION_NO_INDEX, "runtime numeric authority is incomplete",
            YVEX_ERR_FORMAT, err);
    yvex_sha256_init(&hash);
    yvex_sha256_update_text(&hash, "yvex.runtime.numeric.deepseek-v4.v2");
    yvex_sha256_update_text(&hash, model->hadamard_revision);
    yvex_sha256_update_text(&hash, model->sglang_revision);
    yvex_sha256_update_u64(&hash, model->runtime_numeric_schema_version);
    yvex_sha256_update_u64(&hash, model->runtime_compute_policy_count);
    yvex_sha256_update_u64(&hash, model->runtime_activation_policy_count);
    yvex_sha256_update_u64(&hash, model->runtime_sparse_topk_policy_count);
    count = api->ir.layer_count(ir);
    yvex_sha256_update_u64(&hash, count);
    for (index = 0ull; index < count; ++index) {
        const yvex_deepseek_v4_layer_spec *layer = api->ir.layer_at(ir, index);
        if (!layer || layer->moe.routed_experts != first->moe.routed_experts ||
            layer->moe.experts_per_token != first->moe.experts_per_token)
            return deepseek_descriptor_refuse(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, index,
                "runtime numeric layer is missing", YVEX_ERR_FORMAT, err);
        yvex_sha256_update_u64(&hash, layer->layer_index);
        yvex_sha256_update_u64(&hash, (unsigned long long)layer->attention_class);
        yvex_sha256_update_u64(&hash, (unsigned long long)layer->compute_contract);
        if (!yvex_model_activation_identity_update(
                &hash, &layer->attention_kv_activation) ||
            !yvex_model_activation_identity_update(
                &hash, &layer->compressor_activation) ||
            !yvex_model_activation_identity_update(
                &hash, &layer->compressor_rotated_activation) ||
            !yvex_model_activation_identity_update(
                &hash, &layer->indexer_query_activation) ||
            !yvex_model_topk_identity_update(&hash, &layer->sparse_topk))
            return deepseek_descriptor_refuse(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, index,
                "runtime numeric identity update failed", YVEX_ERR_STATE, err);
    }
    if (!yvex_sha256_final(&hash, digest))
        return deepseek_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE,
            YVEX_MATERIALIZATION_NO_INDEX, "runtime numeric identity failed",
            YVEX_ERR_STATE, err);
    memset(contract, 0, sizeof(*contract));
    contract->schema_version = YVEX_SEMANTIC_NUMERIC_CONTRACT_SCHEMA_V1;
    contract->numeric_schema_version = model->runtime_numeric_schema_version;
    contract->compute_policy_count = model->runtime_compute_policy_count;
    contract->activation_policy_count = model->runtime_activation_policy_count;
    contract->sparse_topk_policy_count = model->runtime_sparse_topk_policy_count;
    yvex_core_text_copy(contract->algorithm_revision, sizeof(contract->algorithm_revision),
                        model->hadamard_revision);
    yvex_sha256_hex(digest, contract->identity);
    return YVEX_OK;
}

static int deepseek_descriptor_facts(
    const yvex_semantic_model_ir *semantic_model,
    yvex_runtime_descriptor_family_facts *facts,
    yvex_runtime_descriptor_failure *failure, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *semantic =
        yvex_semantic_model_ir_summary_get(semantic_model);
    const yvex_semantic_attention_layer *layers = NULL;
    const yvex_model_execution_descriptor *execution =
        semantic ? &semantic->execution_descriptor : NULL;
    const yvex_semantic_numeric_contract *numeric =
        semantic ? &semantic->numeric_contract : NULL;
    unsigned long long layer_count = 0ull;

    if (!semantic || !execution || !numeric ||
        numeric->schema_version != YVEX_SEMANTIC_NUMERIC_CONTRACT_SCHEMA_V1 ||
        !yvex_sha256_hex_valid(numeric->identity) ||
        !yvex_semantic_model_ir_attention_view(
            semantic_model, YVEX_TENSOR_SCOPE_MAIN_LAYER,
            &layers, &layer_count) ||
        layer_count != execution->layer_count || !layers)
        return deepseek_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE,
            YVEX_MATERIALIZATION_NO_INDEX,
            "semantic execution descriptor and attention topology are incomplete",
            YVEX_ERR_FORMAT, err);
    *facts = (yvex_runtime_descriptor_family_facts){
        .logical_model_identity = execution->logical_model_identity,
        .runtime_numeric_identity = numeric->identity,
        .runtime_hadamard_revision = numeric->algorithm_revision,
        .runtime_numeric_schema_version = numeric->numeric_schema_version,
        .runtime_compute_policy_count = numeric->compute_policy_count,
        .runtime_activation_policy_count = numeric->activation_policy_count,
        .runtime_sparse_topk_policy_count = numeric->sparse_topk_policy_count,
        .layer_count = execution->layer_count,
        .draft_layer_count = execution->draft_layer_count,
        .routed_experts = execution->routed_experts,
        .experts_per_token = execution->experts_per_row,
        .vocabulary_size = execution->vocabulary_size,
        .model_execution = execution};
    return YVEX_OK;
}

int yvex_runtime_descriptor_build_deepseek(
    yvex_runtime_descriptor **out,
    const yvex_complete_artifact_admission *admission,
    const yvex_materialization_session *session,
    const yvex_artifact_lowering_map *map,
    const yvex_semantic_model_ir *semantic_model,
    yvex_runtime_descriptor_failure *failure, yvex_error *err)
{
    yvex_materialization_projection projection;
    yvex_runtime_descriptor_family_facts facts;
    int rc;

    if (out) *out = NULL;
    if (!out || !map || !semantic_model)
        return deepseek_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_INVALID_ARGUMENT,
            YVEX_MATERIALIZATION_NO_INDEX,
            "map and sealed semantic model are required",
            YVEX_ERR_INVALID_ARG, err);
    rc = yvex_materialization_project_artifact_lowering(map, &projection, err);
    if (rc == YVEX_OK)
        rc = deepseek_descriptor_facts(
            semantic_model, &facts, failure, err);
    return rc == YVEX_OK
               ? yvex_runtime_descriptor_build_projected(
                     out, admission, session, &facts, &projection, failure, err)
               : rc;
}

static int deepseek_semantic_model_build(
    yvex_semantic_model_ir **out, const yvex_deepseek_v4_ir *semantic,
    yvex_error *err)
{
    const yvex_model_family_api *model = yvex_model_register_deepseek_v4();
    yvex_runtime_descriptor_failure failure = {0};
    yvex_model_execution_descriptor execution = {0};
    yvex_semantic_numeric_contract numeric = {0};
    yvex_semantic_model_ir_request request = {0};
    char logical[YVEX_SHA256_HEX_CAP];
    int rc;

    if (out) *out = NULL;
    if (!out || !model || !semantic) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "deepseek.semantic-model",
                       "semantic model output and family architecture are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = model->transform.architecture_identity(semantic, logical)
             ? YVEX_OK : YVEX_ERR_STATE;
    if (rc != YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_STATE, "deepseek.semantic-model",
                       "logical model identity derivation failed");
    }
    if (rc == YVEX_OK)
        rc = model->ir.execution_descriptor(
            semantic, logical, &execution, err);
    if (rc == YVEX_OK)
        rc = deepseek_numeric_contract_build(
            semantic, &numeric, &failure, err);
    if (rc == YVEX_OK) {
        request.schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1;
        request.family_adapter_id = YVEX_DEEPSEEK_V4_ADAPTER_ID;
        request.family_adapter_version = YVEX_DEEPSEEK_V4_ADAPTER_VERSION;
        request.target_id = "deepseek4-v4-flash-dspark";
        request.source_model_identity = execution.source_model_identity;
        request.logical_model_identity = execution.logical_model_identity;
        request.semantic_payload_identity = execution.identity;
        request.execution_descriptor = &execution;
        request.numeric_contract = &numeric;
        request.attention_context = semantic;
        request.attention_layer = graph_recipe_layer;
        request.attention_layer_count = model->ir.layer_count(semantic);
        request.draft_attention_layer = graph_recipe_draft_layer;
        request.draft_attention_layer_count =
            model->ir.auxiliary_count(semantic);
        rc = yvex_semantic_model_ir_seal(out, &request, err);
    }
    return rc;
}

static int deepseek_compilation_semantic_model(
    yvex_semantic_model_ir **out,
    const yvex_source_verification *verification, yvex_error *err)
{
    const yvex_model_family_api *model = yvex_model_register_deepseek_v4();
    yvex_deepseek_v4_ir_failure failure = {0};
    yvex_deepseek_v4_ir *semantic = NULL;
    int rc;

    if (out) *out = NULL;
    rc = model->ir.build(&semantic, verification, &failure, err);
    if (rc == YVEX_OK)
        rc = deepseek_semantic_model_build(out, semantic, err);
    model->ir.close(semantic);
    return rc;
}

static int deepseek_compilation_descriptor(
    yvex_runtime_descriptor **out, const yvex_complete_artifact_admission *admission,
    yvex_materialization_session *materialization, const void *lowering_context,
    const yvex_semantic_model_ir *semantic_model, yvex_error *err)
{
    yvex_runtime_descriptor_failure failure = {0};

    return yvex_runtime_descriptor_build_deepseek(
        out, admission, materialization,
        (const yvex_artifact_lowering_map *)lowering_context,
        semantic_model, &failure, err);
}

static int deepseek_compilation_quant_default(
    yvex_quant_plan **out, const yvex_transform_ir *transform,
    const yvex_transform_binding *binding, const void *lowering_context,
    yvex_error *err)
{
    yvex_quant_failure failure = {0};

    return yvex_quant_plan_build_deepseek_profile(
        out, transform, binding, (const yvex_artifact_lowering_map *)lowering_context,
        YVEX_QUANT_PROFILE_RELEASE_Q8_Q2, NULL, &failure, err);
}

static int deepseek_compilation_quant_policy(
    yvex_quant_plan **out, const yvex_transform_ir *transform,
    const yvex_transform_binding *binding, const void *lowering_context,
    const yvex_quant_policy *policy, const char *imatrix_identity,
    yvex_error *err)
{
    yvex_quant_failure failure = {0};

    return yvex_quant_plan_build_deepseek_policy(
        out, transform, binding, (const yvex_artifact_lowering_map *)lowering_context,
        policy, imatrix_identity, NULL, &failure, err);
}

static const yvex_quant_artifact_lowering_rule deepseek_quant_lowering_rules[] = {
    {YVEX_ARTIFACT_LOWERING_TRANSFORM_DIRECT, YVEX_TRANSFORM_OP_IDENTITY,
     YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_Q8_0, 0},
    {YVEX_ARTIFACT_LOWERING_TRANSFORM_FP8_E4M3_E8M0,
     YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR,
     YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_Q8_0, 0},
    {YVEX_ARTIFACT_LOWERING_TRANSFORM_EXPERT_MXFP4,
     YVEX_TRANSFORM_OP_EXPERT_AGGREGATE,
     YVEX_GGUF_QTYPE_MXFP4, YVEX_GGUF_QTYPE_Q2_K, 1},
    {YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32, YVEX_TRANSFORM_OP_CHECKED_CAST,
     YVEX_GGUF_QTYPE_I32, YVEX_GGUF_QTYPE_I32, 1}};

static const yvex_quant_artifact_lowering_policy deepseek_quant_lowering_policy = {
    YVEX_DEEPSEEK_QUANT_SOURCE_PROFILE_NAME,
    YVEX_DEEPSEEK_QUANT_RELEASE_PROFILE_NAME,
    deepseek_quant_lowering_rules,
    sizeof(deepseek_quant_lowering_rules) / sizeof(deepseek_quant_lowering_rules[0])};

int yvex_quant_plan_build_deepseek_profile(
    yvex_quant_plan **out, const yvex_transform_ir *ir,
    const yvex_transform_binding *binding, const yvex_artifact_lowering_map *map,
    yvex_quant_profile_kind profile, const yvex_quant_plan_options *options,
    yvex_quant_failure *failure, yvex_error *err)
{
    return yvex_quant_plan_build_artifact_lowering_profile(
        out, ir, binding, map, &deepseek_quant_lowering_policy,
        profile, options, failure, err);
}

int yvex_quant_plan_build_deepseek_policy(
    yvex_quant_plan **out, const yvex_transform_ir *ir,
    const yvex_transform_binding *binding, const yvex_artifact_lowering_map *map,
    const yvex_quant_policy *policy, const char *imatrix_identity,
    const yvex_quant_plan_options *options, yvex_quant_failure *failure, yvex_error *err)
{
    return yvex_quant_plan_build_artifact_lowering_policy(
        out, ir, binding, map, &deepseek_quant_lowering_policy,
        policy, imatrix_identity, options, failure, err);
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

static unsigned long long deepseek_quant_preset_count(void)
{
    return 3u;
}

static const char *deepseek_quant_preset_name(unsigned long long index)
{
    static const char *const names[] = {
        "source-faithful", YVEX_DEEPSEEK_QUANT_RELEASE_PROFILE_NAME,
        YVEX_DEEPSEEK_QUANT_DSPARK_PROFILE_NAME};

    return index < sizeof(names) / sizeof(names[0]) ? names[index] : NULL;
}

static int deepseek_quant_preset_open(
    yvex_quant_policy **out, const char *name, yvex_error *err)
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

const yvex_quant_preset_catalog *yvex_graph_deepseek_v4_quant_presets(void)
{
    static const yvex_quant_preset_catalog catalog = {
        YVEX_QUANT_PRESET_CATALOG_SCHEMA_V1,
        "deepseek4-v4-flash-dspark",
        deepseek_quant_preset_count,
        deepseek_quant_preset_name,
        deepseek_quant_preset_open};

    return &catalog;
}

static const yvex_complete_artifact_admission deepseek_selected_catalog = {
    .artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX,
    .metadata_count = 76ull,
    .tensor_count = 1409ull,
    .payload_bytes = 108274154488ull,
    .file_bytes = YVEX_SELECTED_DEEPSEEK_FILE_BYTES,
    .source_snapshot_identity = 0x8d8da435dea23049ull,
    .mapping_identity = 0x779aa44d104fc718ull,
    .payload_identity =
        "e05ddb86f9783bf665d05395636588f4e8dbd1ee6f1ba54be4140f84369ee939",
    .transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .profile_identity =
        "a48d43c8594999a1af3a5b1f572b34a5823042cb767832d558642bb804b036c5",
    .profile_name = "deepseek-v4-flash-dspark-bootstrap-q2-v1",
    .quant_execution_identity =
        "777559149e4e8421c34299da78f63f6b0d296a91005d7670196164c3c72b62af",
    .payload_plan_identity =
        "8d1a89e794363c0aaf1c721b07c0661ea03f9680691d0113543b2540297b69e7",
    .payload_byte_identity =
        "6dce1edb82810715687d40c6d62273e992cfe9e0aa610cb9598447e06fb7099f",
    .writer_plan_identity =
        "1ba1ceaa709862145b1a145e938cf03327cd58da27bca42ade2f884e2b2fc635",
    .artifact_identity =
        "bf80bd7372e9ff754cd61d8f6e849ca8eff2177fad40840a2dad8e840b35690a",
    .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
    .tokenizer_complete = 1,
    .native_reader_accepted = 1,
    .official_reader_accepted = 1,
    .payload_integrity_accepted = 1,
    .materialization_input_ready = 1,
};

static const yvex_complete_artifact_admission deepseek_native_drafter_catalog = {
    .artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX,
    .metadata_count = 76ull,
    .tensor_count = 1409ull,
    .payload_bytes = 98006498296ull,
    .file_bytes = 98018204640ull,
    .source_snapshot_identity = 0x8d8da435dea23049ull,
    .mapping_identity = 0x779aa44d104fc718ull,
    .payload_identity =
        "e05ddb86f9783bf665d05395636588f4e8dbd1ee6f1ba54be4140f84369ee939",
    .transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .profile_identity =
        "6a99e9f7c374e3f718cce705002bf2b799db9cc1b86f65091631857f52c1c587",
    .profile_name = "deepseek-v4-flash-dspark-native-drafter-candidate",
    .quant_execution_identity =
        "35002244d5854a2d51b877ea31614cd985c9795d11c7e0904ed3475fec7fcb77",
    .payload_plan_identity =
        "e83545c729b219d327d4a437d499b73407648c94748ba7fda13905baace15c3e",
    .payload_byte_identity =
        "c79712bb85e31ebdcbd71ef0256709a001ae4cc62c4150ba8726d5dc5722dcd0",
    .writer_plan_identity =
        "2d4694925c02c04811ea846f389a94dbf524d26809a292c93f2c46ca8f05a025",
    .artifact_identity =
        "59c4649b19bb9f3eb7c01559e12ae52c3d4fbd067957e35de0a1a851759c7cc1",
    .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
    .tokenizer_complete = 1,
    .native_reader_accepted = 1,
    .official_reader_accepted = 1,
    .payload_integrity_accepted = 1,
    .materialization_input_ready = 1,
};

static const yvex_complete_artifact_admission *deepseek_catalog_find(
    unsigned long long file_bytes)
{
    static const yvex_complete_artifact_admission *const rows[] = {
        &deepseek_selected_catalog,
        &deepseek_native_drafter_catalog,
    };
    size_t index;

    for (index = 0u; index < sizeof(rows) / sizeof(rows[0]); ++index)
        if (rows[index]->file_bytes == file_bytes) return rows[index];
    return NULL;
}

int yvex_artifact_admit_deepseek(
    const yvex_artifact *artifact, yvex_complete_artifact_admission *out,
    yvex_artifact_admission_failure *failure, yvex_error *err)
{
    const yvex_complete_artifact_admission *catalog =
        artifact ? deepseek_catalog_find(yvex_artifact_size(artifact)) : NULL;
    yvex_artifact_catalog_contract contract = {0};

    if (!artifact || !out)
        return yvex_artifact_admit_catalog(
            artifact, NULL, NULL, &contract, out, failure, err);
    if (!catalog) {
        memset(out, 0, sizeof(*out));
        if (failure) {
            memset(failure, 0, sizeof(*failure));
            failure->code = YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH;
            failure->actual = yvex_artifact_size(artifact);
            yvex_core_text_copy(failure->field, sizeof(failure->field), "file-bytes");
        }
        yvex_error_set(err, YVEX_ERR_FORMAT, "model.deepseek.artifact-catalog",
                       "artifact extent is not in the admitted DeepSeek physical catalog");
        return YVEX_ERR_FORMAT;
    }
    contract.catalog = catalog;
    return yvex_artifact_admit_catalog(
        artifact, NULL, NULL, &contract, out, failure, err);
}
