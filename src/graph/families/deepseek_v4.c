#include "src/graph/private.h"
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/transformer.h>
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
static const yvex_graph_compiler_api deepseek_graph_compiler = {
    .plan_build = graph_plan_build,
    .draft_plan_build = graph_draft_plan_build};
static const yvex_graph_compiler_api *deepseek_graph_compile(void) {
    return &deepseek_graph_compiler;
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
const yvex_moe_family_api *yvex_graph_moe_family_at(unsigned long long index) {
    return index == 0ull ? &deepseek_moe_api : NULL;
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
static const yvex_graph_execution_binding deepseek_execution = {
    .schema_version = YVEX_GRAPH_EXECUTION_BINDING_SCHEMA_V1,
    .adapter_id = YVEX_DEEPSEEK_V4_ADAPTER_ID,
    .adapter_version = YVEX_DEEPSEEK_V4_ADAPTER_VERSION,
    .target_id = "deepseek4-v4-flash-dspark", .family_name = "deepseek-v4-flash-dspark",
    .logical_transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .operator_family_key = "deepseek",
    .operator_artifact_filename = YVEX_SELECTED_DEEPSEEK_ARTIFACT_FILENAME,
    .api = &yvex_attention_execution_api};
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
    .speculation_policy = deepseek_speculation_policy};
const yvex_family_compiler_adapter *yvex_compiler_family_deepseek_v4(void) {
    return &deepseek_compiler;
}
const yvex_graph_execution_binding *yvex_graph_execution_at(unsigned long long index) {
    return index == 0ull ? &deepseek_execution : NULL;
}
