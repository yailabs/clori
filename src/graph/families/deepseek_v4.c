/*
 * Compile irreducible DeepSeek model semantics into generic transformation and execution plans.
 * Mutable plan storage and validation remain owned by generic compiler sinks; this projection
 * supplies only source roles, topology, numerical policy, and operator composition.
 */
#include "src/graph/private.h"
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/tokenizer.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/transformer.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int graph_recipe_identity(const void *context, char output[65])
{
    return yvex_model_register_deepseek_v4()->transform.architecture_identity(
        (const yvex_deepseek_v4_ir *)context, output);
}
static int graph_recipe_project(const yvex_deepseek_v4_layer_spec *layer,
                                unsigned long long ordinal, yvex_tensor_scope scope,
                                unsigned long long predictor,
                                yvex_attention_layer_plan *out)
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
                              yvex_attention_layer_plan *out)
{
    const yvex_deepseek_v4_layer_spec *layer =
        yvex_model_register_deepseek_v4()->ir.layer_at(
            (const yvex_deepseek_v4_ir *)context, index);
    return graph_recipe_project(layer, index, YVEX_TENSOR_SCOPE_MAIN_LAYER,
                                YVEX_ATTENTION_NO_TENSOR_INDEX, out);
}
static int graph_recipe_draft_layer(const void *context, unsigned long long index,
                                    yvex_attention_layer_plan *out)
{
    const yvex_deepseek_v4_auxiliary_spec *draft =
        yvex_model_register_deepseek_v4()->ir.auxiliary_at(
            (const yvex_deepseek_v4_ir *)context, index);
    return draft && graph_recipe_project(&draft->layer, index, YVEX_TENSOR_SCOPE_DRAFT,
                                         draft->predictor_index, out);
}
static int graph_plan_build(yvex_attention_plan **out, const void *family_ir,
    const yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_deepseek_v4_ir *ir = (const yvex_deepseek_v4_ir *)family_ir;
    const yvex_deepseek_v4_model_spec *model = ir
        ? yvex_model_register_deepseek_v4()->ir.model(ir) : NULL;
    yvex_attention_recipe recipe;
    if (!model) return yvex_attention_plan_build(out, NULL, session, descriptor, failure, err);
    recipe = (yvex_attention_recipe){
        .context = ir, .layer_count = yvex_model_register_deepseek_v4()->ir.layer_count(ir),
        .auxiliary_layer_count = model->auxiliary_layer_count,
        .swa_layer_count = model->swa_layer_count, .csa_layer_count = model->csa_layer_count,
        .hca_layer_count = model->hca_layer_count,
        .tensor_scope = YVEX_TENSOR_SCOPE_MAIN_LAYER,
        .identity = graph_recipe_identity, .layer = graph_recipe_layer};
    return yvex_attention_plan_build(out, &recipe, session, descriptor, failure, err);
}
static int graph_draft_plan_build(yvex_attention_plan **out, const void *family_ir,
    const yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_deepseek_v4_ir *ir = (const yvex_deepseek_v4_ir *)family_ir;
    const yvex_deepseek_v4_model_spec *model =
        ir ? yvex_model_register_deepseek_v4()->ir.model(ir) : NULL;
    yvex_attention_recipe recipe;
    if (!model || !model->dspark.present)
        return yvex_attention_plan_build(out, NULL, session, descriptor, failure, err);
    recipe = (yvex_attention_recipe){
        .context = ir, .layer_count = model->dspark.draft_layer_count,
        .auxiliary_layer_count = model->dspark.draft_layer_count,
        .swa_layer_count = model->dspark.draft_layer_count,
        .tensor_scope = YVEX_TENSOR_SCOPE_DRAFT,
        .identity = graph_recipe_identity, .layer = graph_recipe_draft_layer};
    return yvex_attention_plan_build(out, &recipe, session, descriptor, failure, err);
}
static const yvex_model_execution_descriptor *deepseek_execution_model(const yvex_runtime_descriptor_summary *runtime) {
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
static const yvex_graph_execution_binding deepseek_execution = {
    .schema_version = YVEX_GRAPH_EXECUTION_BINDING_SCHEMA_V1,
    .adapter_id = YVEX_DEEPSEEK_V4_ADAPTER_ID,
    .adapter_version = YVEX_DEEPSEEK_V4_ADAPTER_VERSION,
    .target_id = "deepseek4-v4-flash-dspark", .family_name = "deepseek-v4-flash-dspark",
    .logical_transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .operator_family_key = "deepseek",
    .operator_artifact_filename = YVEX_SELECTED_DEEPSEEK_ARTIFACT_FILENAME,
    .api = &yvex_attention_execution_api};
static int deepseek_runtime_binding_compile(
    const yvex_compilation_runtime_binding_request *request,
    yvex_runtime_binding_prepare_request *prepare, void **owner,
    yvex_error *err);
static void deepseek_runtime_binding_release(void *owner);
static const yvex_family_compiler_adapter deepseek_compiler = {
    .schema_version = YVEX_FAMILY_COMPILER_SCHEMA_V1,
    .adapter_id = YVEX_DEEPSEEK_V4_ADAPTER_ID,
    .adapter_version = YVEX_DEEPSEEK_V4_ADAPTER_VERSION,
    .target_id = "deepseek4-v4-flash-dspark",
    .logical_transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .graph = deepseek_graph_compile,
    .execution_capabilities = deepseek_execution_capabilities,
    .transformer_policy = deepseek_transformer_policy,
    .logits_policy = deepseek_logits_policy,
    .speculation_policy = deepseek_speculation_policy,
    .tokenizer_policy = deepseek_tokenizer_policy,
    .runtime_binding_compile = deepseek_runtime_binding_compile,
    .runtime_binding_release = deepseek_runtime_binding_release};
const yvex_family_compiler_adapter *yvex_compiler_family_deepseek_v4(void) {
    return &deepseek_compiler;
}
const yvex_graph_execution_binding *yvex_graph_execution_at(unsigned long long index) {
    return index == 0ull ? &deepseek_execution : NULL;
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
static const char *const payload_failure_names[] = {"none",
                                                    "invalid-argument",
                                                    "source-verification",
                                                    "architecture-ir",
                                                    "tensor-coverage",
                                                    "transform-ir",
                                                    "gguf-mapping",
                                                    "mapping-identity-mismatch",
                                                    "mapping-contribution",
                                                    "payload-range",
                                                    "transform-binding",
                                                    "payload-plan",
                                                    "allocation-failure"};

/* Payload handoff resolves typed family inputs through the common source ABI. */

/* Local composed lifecycle operation used by construction-failure unwinds. */
static void payload_close(yvex_deepseek_payload_handoff *handoff);

struct yvex_deepseek_payload_handoff {
    char *source_path;
    char *models_root;
    char *manifest_path;
    yvex_source_verify_options source_options;
    yvex_source_verification verification;
    yvex_deepseek_tensor_coverage *coverage;
    yvex_transform_ir *transform_ir;
    yvex_deepseek_gguf_map *map;
    yvex_source_payload_session *session;
    yvex_transform_binding *binding;
    yvex_source_payload_plan *plan;
    yvex_deepseek_payload_handoff_summary summary;
};

static char *handoff_strdup(const char *text) {
    size_t length;
    char *copy;

    if (!text)
        return NULL;
    length = strlen(text);
    copy = (char *)malloc(length + 1u);
    if (copy)
        memcpy(copy, text, length + 1u);
    return copy;
}

static int handoff_reject(yvex_deepseek_payload_failure *failure,
                          yvex_deepseek_payload_failure_code code, unsigned long long descriptor,
                          unsigned long long contribution, int status, yvex_error *err,
                          const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->descriptor_index = descriptor;
        failure->contribution_index = contribution;
    }
    yvex_error_set(err, (yvex_status)status, "deepseek_payload_handoff", message);
    return status;
}

static int handoff_resolve(yvex_deepseek_payload_handoff *handoff,
                           const yvex_deepseek_payload_handoff_options *options,
                           yvex_deepseek_payload_failure *failure, yvex_error *err) {
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_model_family_lowering_api *lowering = yvex_model_deepseek_lowering_api();
    const yvex_deepseek_gguf_map_summary *map_summary = lowering->summary(handoff->map);
    unsigned long long *tensor_indices;
    unsigned long long contribution_index;
    unsigned long long descriptor_index;
    char identity_message[192];
    int rc;

    if (!map_summary || !map_summary->complete ||
        map_summary->mapping_identity != YVEX_DEEPSEEK_PAYLOAD_MAPPING_IDENTITY ||
        map_summary->source_identity != handoff->verification.source_snapshot_identity) {
        (void)snprintf(
            identity_message, sizeof(identity_message),
            "canonical DeepSeek mapping identity mismatch: expected=%016llx actual=%016llx "
            "source=%016llx verified_source=%016llx",
            (unsigned long long)YVEX_DEEPSEEK_PAYLOAD_MAPPING_IDENTITY,
            map_summary ? map_summary->mapping_identity : 0ull,
            map_summary ? map_summary->source_identity : 0ull,
            handoff->verification.source_snapshot_identity);
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_MAPPING_IDENTITY, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_FORMAT, err,
                              identity_message);
    }
    if (map_summary->source_contribution_count >
        (unsigned long long)(SIZE_MAX / sizeof(tensor_indices[0]))) {
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_ALLOCATION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_BOUNDS, err,
                              "mapping contribution index allocation overflow");
    }
    tensor_indices = (unsigned long long *)calloc((size_t)map_summary->source_contribution_count,
                                                  sizeof(tensor_indices[0]));
    if (!tensor_indices) {
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_ALLOCATION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_NOMEM, err,
                              "mapping contribution index allocation failed");
    }
    handoff->summary.mapping_identity = map_summary->mapping_identity;
    yvex_core_text_copy(
        handoff->summary.transform_identity,
        sizeof(handoff->summary.transform_identity),
        yvex_transform_ir_summary_get(handoff->transform_ir)->transform_identity);
    handoff->summary.source_snapshot_identity = map_summary->source_identity;
    handoff->summary.descriptor_count = map_summary->descriptor_count;
    handoff->summary.contribution_count = map_summary->source_contribution_count;
    for (contribution_index = 0u; contribution_index < map_summary->source_contribution_count;
         ++contribution_index) {
        const yvex_deepseek_gguf_contribution *contribution =
            lowering->contribution_at(handoff->map, contribution_index);
        const yvex_source_payload_range *range;
        const yvex_deepseek_tensor_coverage_row *coverage_row;
        const yvex_deepseek_gguf_descriptor *descriptor;

        if (!contribution || contribution->descriptor_index >= map_summary->descriptor_count) {
            free(tensor_indices);
            return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_CONTRIBUTION, ULLONG_MAX,
                                  contribution_index, YVEX_ERR_FORMAT, err,
                                  "mapping contribution is incomplete");
        }
        descriptor = lowering->at(handoff->map, contribution->descriptor_index);
        coverage_row = family->coverage.at(handoff->coverage, contribution->source_row_index);
        range = yvex_source_payload_range_find(handoff->session, contribution->source_name);
        handoff->summary.range_lookup_count++;
        if (!descriptor || !coverage_row || !coverage_row->source || !range ||
            strcmp(coverage_row->source->name, contribution->source_name) != 0 ||
            range->source_snapshot_identity != map_summary->source_identity ||
            range->dtype != contribution->source_dtype ||
            range->rank != contribution->source_rank) {
            free(tensor_indices);
            return handoff_reject(
                failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_RANGE, contribution->descriptor_index,
                contribution_index, YVEX_ERR_FORMAT, err,
                "mapping contribution does not resolve to its exact source range");
        }
        tensor_indices[contribution_index] = range->source_tensor_index;
        handoff->summary.contributions_resolved++;
        if (descriptor->transform == YVEX_DEEPSEEK_GGUF_TRANSFORM_DIRECT)
            handoff->summary.direct_contributions++;
        if (contribution->kind == YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY &&
            contribution->source_dtype == YVEX_NATIVE_DTYPE_F8_E4M3)
            handoff->summary.fp8_weight_contributions++;
        if (contribution->kind == YVEX_DEEPSEEK_GGUF_CONTRIBUTION_SCALE &&
            contribution->source_dtype == YVEX_NATIVE_DTYPE_F8_E8M0)
            handoff->summary.e8m0_scale_contributions++;
        if (contribution->kind == YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_WEIGHT ||
            contribution->kind == YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_SCALE) {
            if (ULLONG_MAX - handoff->summary.routed_expert_logical_bytes < range->byte_length) {
                free(tensor_indices);
                return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_RANGE,
                                      contribution->descriptor_index, contribution_index,
                                      YVEX_ERR_BOUNDS, err,
                                      "routed expert payload accounting overflow");
            }
            handoff->summary.expert_contributions++;
            handoff->summary.routed_expert_logical_bytes += range->byte_length;
        }
        if (descriptor->transform == YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32 &&
            contribution->source_dtype == YVEX_NATIVE_DTYPE_I64)
            handoff->summary.i64_router_contributions++;
        if (descriptor->collection == YVEX_TENSOR_COLLECTION_GLOBAL)
            handoff->summary.global_contributions++;
        if (descriptor->collection == YVEX_TENSOR_COLLECTION_NORM)
            handoff->summary.norm_contributions++;
        if (descriptor->collection == YVEX_TENSOR_COLLECTION_SHARED_EXPERT)
            handoff->summary.shared_expert_contributions++;
        if (descriptor->role == YVEX_TENSOR_ROLE_OUTPUT_HEAD) {
            if (ULLONG_MAX - handoff->summary.output_head_logical_bytes < range->byte_length) {
                free(tensor_indices);
                return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_RANGE,
                                      contribution->descriptor_index, contribution_index,
                                      YVEX_ERR_BOUNDS, err,
                                      "output head payload accounting overflow");
            }
            handoff->summary.output_head_contributions++;
            handoff->summary.output_head_logical_bytes += range->byte_length;
        }
        if (descriptor->scope == YVEX_TENSOR_SCOPE_DRAFT)
            handoff->summary.draft_contributions++;
    }
    for (descriptor_index = 0u; descriptor_index < map_summary->descriptor_count;
         ++descriptor_index) {
        const yvex_deepseek_gguf_descriptor *descriptor =
            lowering->at(handoff->map, descriptor_index);
        unsigned long long end;

        if (!descriptor || descriptor->contribution_count == 0u ||
            ULLONG_MAX - descriptor->contribution_offset < descriptor->contribution_count) {
            free(tensor_indices);
            return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_CONTRIBUTION,
                                  descriptor_index, ULLONG_MAX, YVEX_ERR_FORMAT, err,
                                  "logical descriptor has no bounded source contribution set");
        }
        end = descriptor->contribution_offset + descriptor->contribution_count;
        if (end > handoff->summary.contributions_resolved) {
            free(tensor_indices);
            return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_CONTRIBUTION,
                                  descriptor_index, end, YVEX_ERR_FORMAT, err,
                                  "logical descriptor contribution span exceeds resolved mapping");
        }
        handoff->summary.descriptors_covered++;
    }
    rc = yvex_source_payload_plan_build(
        &handoff->plan, handoff->session, tensor_indices, map_summary->source_contribution_count,
        options->chunk_bytes, options->page_bytes, failure ? &failure->payload_failure : NULL, err);
    free(tensor_indices);
    if (rc != YVEX_OK) {
        if (failure)
            failure->code = YVEX_DEEPSEEK_PAYLOAD_FAILURE_PLAN;
        return rc;
    }
    handoff->summary.complete =
        handoff->summary.descriptors_covered == YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT &&
        handoff->summary.contributions_resolved == YVEX_DEEPSEEK_GGUF_SOURCE_COUNT &&
        handoff->summary.fp8_weight_contributions != 0u &&
        handoff->summary.e8m0_scale_contributions != 0u &&
        handoff->summary.expert_contributions != 0u &&
        handoff->summary.i64_router_contributions != 0u &&
        handoff->summary.global_contributions != 0u && handoff->summary.norm_contributions != 0u &&
        handoff->summary.shared_expert_contributions != 0u &&
        handoff->summary.output_head_contributions != 0u &&
        handoff->summary.draft_contributions != 0u;
    if (!handoff->summary.complete)
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_CONTRIBUTION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_FORMAT, err,
                              "mapping payload handoff lacks one required contribution class");
    return YVEX_OK;
}

static int payload_open(yvex_deepseek_payload_handoff **out,
                        const yvex_deepseek_payload_handoff_options *options,
                        yvex_deepseek_payload_failure *failure, yvex_error *err) {
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_model_family_lowering_api *lowering = yvex_model_deepseek_lowering_api();
    yvex_deepseek_payload_handoff *handoff;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure ir_failure;
    yvex_deepseek_tensor_coverage_failure coverage_failure;
    yvex_deepseek_gguf_map_failure map_failure;
    yvex_transform_failure transform_failure;
    yvex_source_payload_open_options payload_options;
    int rc;

    if (out)
        *out = NULL;
    if (!out || !options || !options->source_path || !options->source_path[0] ||
        !options->models_root || !options->models_root[0]) {
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_INVALID_ARGUMENT, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_INVALID_ARG, err,
                              "source path, models root, and output are required");
    }
    handoff = (yvex_deepseek_payload_handoff *)calloc(1u, sizeof(*handoff));
    if (!handoff)
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_ALLOCATION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_NOMEM, err, "payload handoff allocation failed");
    handoff->source_path = handoff_strdup(options->source_path);
    handoff->models_root = handoff_strdup(options->models_root);
    handoff->manifest_path = options->manifest_path ? handoff_strdup(options->manifest_path) : NULL;
    if (!handoff->source_path || !handoff->models_root ||
        (options->manifest_path && !handoff->manifest_path)) {
        payload_close(handoff);
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_ALLOCATION, ULLONG_MAX,
                              ULLONG_MAX, YVEX_ERR_NOMEM, err,
                              "payload handoff path allocation failed");
    }
    handoff->source_options.identity = yvex_source_release_identity();
    handoff->source_options.source_path = handoff->source_path;
    handoff->source_options.models_root = handoff->models_root;
    handoff->source_options.manifest_path = handoff->manifest_path;
    handoff->source_options.promote_manifest = 1;
    rc = yvex_source_verify_with_snapshot(&handoff->source_options, &handoff->verification,
                                          &snapshot, err);
    if (rc != YVEX_OK || !handoff->verification.verified || !snapshot) {
        yvex_source_tensor_snapshot_release(snapshot);
        payload_close(handoff);
        return handoff_reject(failure, YVEX_DEEPSEEK_PAYLOAD_FAILURE_SOURCE, ULLONG_MAX, ULLONG_MAX,
                              rc == YVEX_OK ? YVEX_ERR_STATE : rc, err,
                              "exact source verification did not produce a retained snapshot");
    }
    rc = family->ir.build(&ir, &handoff->verification, &ir_failure, err);
    if (rc != YVEX_OK) {
        yvex_source_tensor_snapshot_release(snapshot);
        payload_close(handoff);
        if (failure)
            failure->code = YVEX_DEEPSEEK_PAYLOAD_FAILURE_ARCHITECTURE;
        return rc;
    }
    rc = family->coverage.build(&handoff->coverage, &handoff->verification, ir, snapshot, NULL,
                                &coverage_failure, err);
    if (rc == YVEX_OK)
        rc = family->transform.build(&handoff->transform_ir, &handoff->verification, ir,
                                     handoff->coverage, NULL, &transform_failure, err);
    if (rc == YVEX_OK)
        rc = lowering->build(&handoff->map, ir, handoff->transform_ir, &map_failure, err);
    family->ir.close(ir);
    if (rc != YVEX_OK) {
        yvex_deepseek_payload_failure_code code =
            !handoff->coverage
                ? YVEX_DEEPSEEK_PAYLOAD_FAILURE_COVERAGE
                : (!handoff->transform_ir ? YVEX_DEEPSEEK_PAYLOAD_FAILURE_TRANSFORM_IR
                                          : YVEX_DEEPSEEK_PAYLOAD_FAILURE_MAPPING);
        yvex_source_tensor_snapshot_release(snapshot);
        payload_close(handoff);
        if (failure)
            failure->code = code;
        return rc;
    }
    memset(&payload_options, 0, sizeof(payload_options));
    payload_options.verification_options = &handoff->source_options;
    payload_options.verification = &handoff->verification;
    payload_options.snapshot = snapshot;
    payload_options.budget = options->budget;
    payload_options.manifest_path = handoff->verification.manifest_path;
    rc = yvex_source_payload_session_open(&handoff->session, &payload_options,
                                          failure ? &failure->payload_failure : NULL, err);
    yvex_source_tensor_snapshot_release(snapshot);
    if (rc != YVEX_OK) {
        if (failure)
            failure->code = YVEX_DEEPSEEK_PAYLOAD_FAILURE_SOURCE;
        payload_close(handoff);
        return rc;
    }
    rc = yvex_transform_binding_create(&handoff->binding, handoff->transform_ir, handoff->session,
                                       NULL, &transform_failure, err);
    if (rc != YVEX_OK) {
        if (failure)
            failure->code = YVEX_DEEPSEEK_PAYLOAD_FAILURE_BINDING;
        payload_close(handoff);
        return rc;
    }
    rc = handoff_resolve(handoff, options, failure, err);
    if (rc != YVEX_OK) {
        payload_close(handoff);
        return rc;
    }
    *out = handoff;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

static void payload_close(yvex_deepseek_payload_handoff *handoff) {
    const yvex_model_family_api *family;
    const yvex_model_family_lowering_api *lowering;

    if (!handoff)
        return;
    family = yvex_model_register_deepseek_v4();
    lowering = yvex_model_deepseek_lowering_api();
    yvex_source_payload_plan_close(handoff->plan);
    yvex_transform_binding_release(&handoff->binding);
    (void)yvex_source_payload_session_release(&handoff->session, NULL, NULL);
    lowering->close(handoff->map);
    yvex_transform_ir_release(&handoff->transform_ir);
    family->coverage.close(handoff->coverage);
    free(handoff->manifest_path);
    free(handoff->models_root);
    free(handoff->source_path);
    free(handoff);
}

static const yvex_deepseek_payload_handoff_summary *
payload_summary(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? &handoff->summary : NULL;
}

static const yvex_source_verification *
payload_verification(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? &handoff->verification : NULL;
}

static const yvex_deepseek_gguf_map *payload_map(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->map : NULL;
}

static const yvex_transform_ir *payload_transform_ir(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->transform_ir : NULL;
}

static const yvex_transform_binding *payload_binding(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->binding : NULL;
}

static yvex_source_payload_session *payload_session(yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->session : NULL;
}

static const yvex_source_payload_plan *payload_plan(const yvex_deepseek_payload_handoff *handoff) {
    return handoff ? handoff->plan : NULL;
}

static const char *payload_failure_name(yvex_deepseek_payload_failure_code code) {
    size_t count = sizeof(payload_failure_names) / sizeof(payload_failure_names[0]);

    return code >= 0 && (size_t)code < count ? payload_failure_names[code]
                                             : "unknown-handoff-failure";
}

/*
 * Publish the immutable trusted-payload handoff operation table used by the family registration.
 *
 * Returns process-lifetime immutable storage; no allocation or I/O.
 */
const yvex_model_family_payload_api *yvex_model_deepseek_payload_api(void) {
    static const yvex_model_family_payload_api api = {
        payload_open, payload_close,        payload_summary, payload_verification,
        payload_map,  payload_transform_ir, payload_binding, payload_session,
        payload_plan, payload_failure_name};

    return &api;
}

typedef struct {
    const yvex_model_family_api *model;
    const yvex_graph_compiler_api *graph;
    yvex_deepseek_payload_handoff *handoff;
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
    yvex_complete_artifact_admission admission;
    yvex_materialization_plan *materialization_plan;
    yvex_materialization_session *materialization;
    yvex_deepseek_v4_ir *architecture;
    yvex_runtime_descriptor *descriptor;
    yvex_attention_plan *attention;
    yvex_attention_plan *draft_attention;
    yvex_quant_policy *quant_policy;
    yvex_imatrix_data *imatrix;
    yvex_quant_plan *quant;
    yvex_gguf_writer_plan *writer;
    yvex_artifact_physical_compatibility compatibility;
    yvex_artifact_compatibility_failure compatibility_failure;
    yvex_deepseek_payload_handoff_options payload_options;
    yvex_deepseek_payload_failure payload_failure;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_projection materialization_projection;
    yvex_materialization_failure materialization_failure;
    yvex_runtime_descriptor_failure descriptor_failure;
    yvex_deepseek_v4_ir_failure architecture_failure;
    yvex_attention_failure attention_failure;
    yvex_quant_failure quant_failure;
    yvex_gguf_writer_failure writer_failure;
} runtime_binding_compiler;

static void runtime_binding_compiler_close(runtime_binding_compiler *compiler)
{
    if (!compiler) return;
    yvex_gguf_writer_plan_release(&compiler->writer);
    yvex_quant_plan_release(&compiler->quant);
    yvex_imatrix_data_close(compiler->imatrix);
    yvex_quant_policy_close(compiler->quant_policy);
    yvex_attention_plan_close(compiler->attention);
    yvex_attention_plan_close(compiler->draft_attention);
    yvex_runtime_descriptor_close(compiler->descriptor);
    if (compiler->model) compiler->model->ir.close(compiler->architecture);
    yvex_materialization_session_close(compiler->materialization);
    yvex_materialization_plan_close(compiler->materialization_plan);
    yvex_tensor_table_close(compiler->tensors);
    yvex_gguf_close(compiler->gguf);
    yvex_artifact_close(compiler->artifact);
    if (compiler->model) compiler->model->payload.close(compiler->handoff);
    memset(compiler, 0, sizeof(*compiler));
}

static int runtime_binding_compiler_open(
    runtime_binding_compiler *compiler,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err)
{
    yvex_artifact_options options = {0};
    int rc;

    compiler->payload_options.source_path = request->source_path;
    compiler->payload_options.models_root = request->models_root;
    compiler->payload_options.manifest_path = request->source_manifest_path;
    yvex_source_payload_budget_default(&compiler->payload_options.budget);
    compiler->payload_options.budget.maximum_open_handles = 32u;
    compiler->payload_options.budget.maximum_streams = 16u;
    compiler->payload_options.budget.maximum_inflight_host_bytes =
        compiler->payload_options.budget.chunk_bytes *
        compiler->payload_options.budget.maximum_streams;
    compiler->payload_options.chunk_bytes = compiler->payload_options.budget.chunk_bytes;
    compiler->payload_options.page_bytes = compiler->payload_options.budget.page_bytes;
    rc = compiler->model->payload.open(&compiler->handoff, &compiler->payload_options,
                                       &compiler->payload_failure, err);
    options.path = request->artifact_path;
    options.readonly = 1;
    if (rc == YVEX_OK) rc = yvex_artifact_open(&compiler->artifact, &options, err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&compiler->gguf, compiler->artifact, err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&compiler->tensors, compiler->gguf, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admit_deepseek(
            compiler->artifact, &compiler->admission, &compiler->admission_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admission_identity_verify(
            compiler->artifact, &compiler->admission, NULL, NULL,
            &compiler->admission_failure, err);
    return rc;
}

static int runtime_binding_compiler_plan(runtime_binding_compiler *compiler,
                                         const yvex_compilation_runtime_binding_request *request,
                                         yvex_error *err)
{
    const yvex_transform_ir_summary *transform = yvex_transform_ir_summary_get(
        compiler->model->payload.transform_ir(compiler->handoff));
    yvex_imatrix_data_options imatrix_options = {0};
    yvex_imatrix_data_summary imatrix_summary = {0};
    yvex_gguf_writer_plan_options writer_options;
    yvex_gguf_writer_plan_request writer_request;
    int rc;

    yvex_materialization_options_default(&compiler->materialization_options);
    compiler->materialization_options.require_terminal_projection = 1;
    compiler->materialization_options.max_chunk_bytes = 16ull * 1024ull * 1024ull;
    compiler->materialization_options.cache_budget_bytes = 256ull * 1024ull * 1024ull;
    compiler->materialization_options.future_graph_scratch_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;
    compiler->materialization_options.future_kv_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;
    rc = yvex_deepseek_materialization_projection(
        compiler->model->payload.map(compiler->handoff),
        &compiler->materialization_projection, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &compiler->materialization_plan, &compiler->admission, compiler->artifact,
            compiler->gguf, compiler->tensors, &compiler->materialization_projection,
            &compiler->materialization_options, &compiler->materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &compiler->materialization, compiler->materialization_plan, compiler->artifact,
            &compiler->materialization_options, &compiler->materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(
            compiler->materialization, &compiler->materialization_failure, err);
    if (rc == YVEX_OK)
        rc = compiler->model->ir.build(
            &compiler->architecture,
            compiler->model->payload.verification(compiler->handoff),
            &compiler->architecture_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_descriptor_build_deepseek(
            &compiler->descriptor, &compiler->admission, compiler->materialization,
            compiler->model->payload.map(compiler->handoff), compiler->architecture,
            &compiler->descriptor_failure, err);
    if (rc == YVEX_OK)
        rc = compiler->graph->plan_build(
            &compiler->attention, compiler->architecture, compiler->materialization,
            compiler->descriptor, &compiler->attention_failure, err);
    if (rc == YVEX_OK && compiler->graph->draft_plan_build)
        rc = compiler->graph->draft_plan_build(
            &compiler->draft_attention, compiler->architecture,
            compiler->materialization, compiler->descriptor,
            &compiler->attention_failure, err);
    if (rc == YVEX_OK && request->physical_variant_plan_path) {
        if (!transform) {
            yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_prepare",
                           "variant preparation requires the sealed transform identity");
            rc = YVEX_ERR_STATE;
        } else if ((request->quant_policy_path != NULL) ==
            (request->quant_preset_name != NULL)) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_prepare",
                           "variant preparation requires exactly one quant policy or preset");
            rc = YVEX_ERR_INVALID_ARG;
        } else if (request->quant_policy_path) {
            rc = yvex_quant_policy_open(&compiler->quant_policy,
                                        request->quant_policy_path, err);
        } else {
            rc = yvex_quant_policy_preset_open(&compiler->quant_policy,
                                               request->quant_preset_name, err);
        }
        if (rc == YVEX_OK && request->imatrix_path) {
            imatrix_options.path = request->imatrix_path;
            /* This request carries a path, not independent calibration provenance. Reconstruct
             * only the admitted predecessor prior used by planning and emission; a fresh
             * calibration requires an explicit provenance contract rather than inference from
             * the policy-selection mechanism. */
            imatrix_options.source_model_identity = YVEX_QUANT_DSPARK_IMATRIX_SOURCE_IDENTITY;
            imatrix_options.calibration_dataset_identity =
                YVEX_QUANT_DSPARK_IMATRIX_DATASET_IDENTITY;
            imatrix_options.producer = "llama.cpp-imatrix";
            imatrix_options.producer_version = 1u;
            imatrix_options.maximum_mapped_bytes = 1024u * 1024u * 1024u;
            rc = yvex_imatrix_data_open(&compiler->imatrix, &imatrix_options, err);
            if (rc == YVEX_OK)
                rc = yvex_imatrix_data_get_summary(compiler->imatrix,
                                                   &imatrix_summary, err);
        }
        if (rc == YVEX_OK)
            rc = yvex_quant_plan_build_deepseek_policy(
                &compiler->quant,
                compiler->model->payload.transform_ir(compiler->handoff),
                compiler->model->payload.binding(compiler->handoff),
                compiler->model->payload.map(compiler->handoff),
                compiler->quant_policy,
                imatrix_summary.complete ? imatrix_summary.imatrix_identity : NULL,
                NULL,
                &compiler->quant_failure, err);
        if (rc == YVEX_OK)
            rc = yvex_quant_plan_file_validate(request->physical_variant_plan_path,
                                               compiler->quant, err);
    } else if (rc == YVEX_OK &&
               (request->quant_policy_path || request->quant_preset_name ||
                request->imatrix_path)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_prepare",
                       "policy, preset, and imatrix require a sealed physical variant plan");
        rc = YVEX_ERR_INVALID_ARG;
    } else if (rc == YVEX_OK) {
        rc = yvex_quant_plan_build_deepseek_profile(
            &compiler->quant, compiler->model->payload.transform_ir(compiler->handoff),
            compiler->model->payload.binding(compiler->handoff),
            compiler->model->payload.map(compiler->handoff),
            YVEX_QUANT_PROFILE_RELEASE_Q8_Q2, NULL, &compiler->quant_failure, err);
    }
    if (rc != YVEX_OK) return rc;
    yvex_gguf_writer_plan_options_default(&writer_options);
    writer_options.required_execution_identity =
        request->physical_variant_plan_path ? NULL : compiler->admission.quant_execution_identity;
    memset(&writer_request, 0, sizeof(writer_request));
    writer_request.input_class = YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT;
    writer_request.quant_plan = compiler->quant;
    writer_request.options = &writer_options;
    writer_request.input.complete.family_adapter = compiler->model;
    writer_request.input.complete.lowering =
        compiler->model->payload.map(compiler->handoff);
    writer_request.input.complete.verification =
        compiler->model->payload.verification(compiler->handoff);
    return yvex_gguf_writer_plan_build(
        &compiler->writer, &writer_request, &compiler->writer_failure, err);
}

static void deepseek_runtime_binding_release(void *owner)
{
    runtime_binding_compiler *compiler = (runtime_binding_compiler *)owner;

    if (!compiler) return;
    runtime_binding_compiler_close(compiler);
    free(compiler);
}

static int deepseek_runtime_binding_compile(
    const yvex_compilation_runtime_binding_request *request,
    yvex_runtime_binding_prepare_request *prepare, void **owner,
    yvex_error *err)
{
    runtime_binding_compiler *compiler;
    const yvex_gguf_writer_plan_summary *writer = NULL;
    const yvex_transform_ir_summary *transform = NULL;
    int rc;

    if (prepare) memset(prepare, 0, sizeof(*prepare));
    if (owner) *owner = NULL;
    if (!request || !prepare || !owner || !request->source_path ||
        !request->models_root || !request->source_manifest_path ||
        !request->artifact_path) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_prepare",
                       "source, artifact, and source manifest are required");
        return YVEX_ERR_INVALID_ARG;
    }
    compiler = (runtime_binding_compiler *)calloc(1u, sizeof(*compiler));
    if (!compiler) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "graph_attention_prepare",
                       "family compilation owner allocation failed");
        return YVEX_ERR_NOMEM;
    }
    *owner = compiler;
    compiler->model = yvex_model_register_deepseek_v4();
    compiler->graph = deepseek_compiler.graph ? deepseek_compiler.graph() : NULL;
    if (!compiler->model || !compiler->graph ||
        !deepseek_compiler.execution_capabilities ||
        !deepseek_compiler.transformer_policy ||
        !deepseek_compiler.logits_policy ||
        !deepseek_compiler.speculation_policy ||
        !deepseek_compiler.tokenizer_policy) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_prepare",
                       "family compiler registration is incomplete");
        return YVEX_ERR_STATE;
    }
    rc = runtime_binding_compiler_open(compiler, request, err);
    if (rc == YVEX_OK) rc = runtime_binding_compiler_plan(compiler, request, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_physical_compatibility_validate(
            compiler->writer, &compiler->admission, compiler->artifact,
            compiler->gguf, &compiler->compatibility,
            &compiler->compatibility_failure, err);
    if (rc == YVEX_OK)
        writer = yvex_gguf_writer_plan_summary_get(compiler->writer);
    if (rc == YVEX_OK)
        transform = yvex_transform_ir_summary_get(
            compiler->model->payload.transform_ir(compiler->handoff));
    if (rc == YVEX_OK &&
        (!writer || !transform ||
         !yvex_sha256_hex_is_valid(transform->transform_identity) ||
         !compiler->compatibility.physical_payload_compatible)) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_prepare",
                       "logical transform and physical compatibility proof are required");
        rc = YVEX_ERR_STATE;
    }
    if (rc != YVEX_OK) return rc;
    prepare->directory = request->directory;
    prepare->admission = &compiler->admission;
    prepare->physical_compatibility = &compiler->compatibility;
    prepare->materialization = compiler->materialization;
    prepare->runtime_descriptor = compiler->descriptor;
    prepare->attention_plan = compiler->attention;
    prepare->draft_attention_plan = compiler->draft_attention;
    prepare->graph_compiler = compiler->graph;
    prepare->family_adapter_id = request->family_adapter_id;
    prepare->family_adapter_version = request->family_adapter_version;
    prepare->artifact_format = "gguf";
    prepare->artifact_format_version = writer->gguf_version;
    prepare->logical_transform_identity = transform->transform_identity;
    if (!deepseek_compiler.execution_capabilities(&prepare->capabilities) ||
        !yvex_runtime_capabilities_contract_valid(&prepare->capabilities) ||
        !deepseek_compiler.transformer_policy(
            yvex_runtime_descriptor_summary_get(compiler->descriptor),
            &prepare->transformer_policy) ||
        !deepseek_compiler.logits_policy(&prepare->logits_policy) ||
        !deepseek_compiler.speculation_policy(
            yvex_runtime_descriptor_summary_get(compiler->descriptor),
            &prepare->speculation_policy) ||
        !deepseek_compiler.tokenizer_policy(&prepare->tokenizer_policy, err)) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_prepare",
                       "family execution envelope compilation failed");
        return YVEX_ERR_STATE;
    }
    return YVEX_OK;
}

static int prepare_deepseek_runtime_binding(
    const yvex_compilation_runtime_binding_request *request,
    yvex_compilation_runtime_binding_result *result, yvex_error *err)
{
    if (result) memset(result, 0, sizeof(*result));
    if (!result) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_prepare",
                       "runtime binding output is required");
        return YVEX_ERR_INVALID_ARG;
    }
    return yvex_runtime_binding_compile_publish(
        &deepseek_compiler, request, result->path, &result->published, err);
}

/*
 * Enumerate compiler-plane family preparation facts and their one typed publication callback.
 *
 * Returns immutable process-lifetime storage.
 */
const yvex_graph_family_preparation *
yvex_graph_family_preparation_at(unsigned long long index)
{
    static const yvex_graph_family_preparation preparation = {
        YVEX_SOURCE_RELEASE_TARGET_ID, YVEX_SOURCE_RELEASE_MANIFEST_LEAF,
        yvex_model_register_deepseek_v4, yvex_artifact_admit_deepseek,
        prepare_deepseek_runtime_binding};
    return index == 0ull ? &preparation : NULL;
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
