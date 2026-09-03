/*
 * Exercises normalization, layer topology, typed refusal, owned lifetime, allocation cleanup,
 * and the model-class report consumer without source IO or tensor payload reads.
 */
#include "tests/test.h"

#include <yvex/internal/artifact.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/model_target.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/source.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void arch_ir_copy(char *out, size_t cap, const char *value)
{
    size_t length = strlen(value);

    if (length >= cap) length = cap - 1u;
    memcpy(out, value, length);
    out[length] = '\0';
}

static void arch_ir_verification_fixture(yvex_source_verification *source)
{
    unsigned long long i;

    memset(source, 0, sizeof(*source));
    source->verified = 1;
    source->path_verified = 1;
    source->repository_verified = 1;
    source->revision_verified = 1;
    source->manifest_verified = 1;
    source->manifest_reopened = 1;
    source->upstream_index_identity_verified = 1;
    source->config_valid = 1;
    source->tokenizer_json_valid = 1;
    source->tokenizer_config_valid = 1;
    source->generation_config_valid = 1;
    source->inference_config_valid = 1;
    source->shard_index_present = 1;
    source->shard_index_valid = 1;
    source->shard_index_headers_match = 1;
    source->header_scan_count = 1;
    source->header_shard_count = 48;
    source->header_tensor_count = 72317;
    source->shard_count = 48;
    source->indexed_tensor_count = 72317;
    arch_ir_copy(source->resolved_source_path,
                 sizeof(source->resolved_source_path),
                 "/verified/DeepSeek-V4-Flash-DSpark");
    arch_ir_copy(source->manifest_target_id,
                 sizeof(source->manifest_target_id),
                 yvex_source_release_identity()->target_id);
    arch_ir_copy(source->verification_stage,
                 sizeof(source->verification_stage),
                 "exact-source-metadata-header-verified");
    arch_ir_copy(source->inventory_authority,
                 sizeof(source->inventory_authority), "upstream-index");
    arch_ir_copy(source->upstream_index_oid,
                 sizeof(source->upstream_index_oid),
                 yvex_source_release_identity()->upstream_index_oid);
    arch_ir_copy(source->local_index_oid,
                 sizeof(source->local_index_oid),
                 yvex_source_release_identity()->upstream_index_oid);
    arch_ir_copy(source->source_kind, sizeof(source->source_kind),
                 "huggingface");
    arch_ir_copy(source->repository_id, sizeof(source->repository_id),
                 yvex_source_release_identity()->upstream_repo_id);
    arch_ir_copy(source->revision, sizeof(source->revision),
                 yvex_source_release_identity()->upstream_revision);
    arch_ir_copy(source->model_type, sizeof(source->model_type),
                 yvex_source_release_identity()->config_model_type);
    arch_ir_copy(source->architecture, sizeof(source->architecture),
                 yvex_source_release_identity()->config_architecture);
    arch_ir_copy(source->torch_dtype, sizeof(source->torch_dtype),
                 "bfloat16");
    arch_ir_copy(source->expert_dtype, sizeof(source->expert_dtype), "fp4");
    arch_ir_copy(source->hidden_act, sizeof(source->hidden_act), "silu");
    arch_ir_copy(source->scoring_func, sizeof(source->scoring_func),
                 "sqrtsoftplus");
    arch_ir_copy(source->topk_method, sizeof(source->topk_method),
                 "noaux_tc");
    arch_ir_copy(source->tokenizer_model_type,
                 sizeof(source->tokenizer_model_type), "BPE");
    arch_ir_copy(source->tokenizer_class, sizeof(source->tokenizer_class),
                 "PreTrainedTokenizerFast");
    arch_ir_copy(source->rope_scaling_type,
                 sizeof(source->rope_scaling_type), "yarn");
    arch_ir_copy(source->quant_method, sizeof(source->quant_method), "fp8");
    arch_ir_copy(source->quant_format, sizeof(source->quant_format), "e4m3");
    arch_ir_copy(source->quant_activation_scheme,
                 sizeof(source->quant_activation_scheme), "dynamic");
    arch_ir_copy(source->quant_scale_format,
                 sizeof(source->quant_scale_format), "ue8m0");
    arch_ir_copy(source->attention_dropout,
                 sizeof(source->attention_dropout), "0.0");
    arch_ir_copy(source->hc_eps, sizeof(source->hc_eps), "0.000001");
    arch_ir_copy(source->rms_norm_eps, sizeof(source->rms_norm_eps),
                 "0.000001");
    arch_ir_copy(source->routed_scaling_factor,
                 sizeof(source->routed_scaling_factor), "1.5");
    arch_ir_copy(source->swiglu_limit, sizeof(source->swiglu_limit), "10.0");
    source->hidden_size = 4096;
    source->num_hidden_layers = 43;
    source->num_attention_heads = 64;
    source->num_key_value_heads = 1;
    source->head_dim = 512;
    source->qk_rope_head_dim = 64;
    source->max_position_embeddings = 1048576;
    source->moe_intermediate_size = 2048;
    source->n_routed_experts = 256;
    source->n_shared_experts = 1;
    source->num_experts_per_tok = 6;
    source->num_hash_layers = 3;
    source->q_lora_rank = 1024;
    source->o_lora_rank = 1024;
    source->vocab_size = 129280;
    source->sliding_window = 128;
    source->bos_token_id = 0;
    source->eos_token_id = 1;
    source->compress_ratio_count = 46;
    for (i = 0u; i < source->compress_ratio_count; ++i) {
        source->compress_ratios[i] =
            i < 2u || i >= 43u ? 0u : (i % 2u == 0u ? 4u : 128u);
    }
    source->compress_rope_theta = 160000;
    source->hc_mult = 4;
    source->hc_sinkhorn_iters = 20;
    source->index_head_dim = 128;
    source->index_n_heads = 64;
    source->index_topk = 512;
    source->num_nextn_predict_layers = 1;
    source->dspark_block_size = 5;
    source->dspark_noise_token_id = 128799;
    source->dspark_target_layer_ids[0] = 40;
    source->dspark_target_layer_ids[1] = 41;
    source->dspark_target_layer_ids[2] = 42;
    source->dspark_target_layer_count = 3;
    source->dspark_markov_rank = 256;
    source->dspark_inference_layer_count = 3;
    source->o_groups = 8;
    source->rope_theta = 10000;
    source->attention_bias = 0;
    source->norm_topk_prob = 1;
    source->use_cache = 1;
    source->rope_scaling_factor = 16;
    source->rope_original_context = 65536;
    source->rope_beta_fast = 32;
    source->rope_beta_slow = 1;
    source->quant_block_rows = 128;
    source->quant_block_columns = 128;
    source->tokenizer_base_vocab_count = 128000;
    source->tokenizer_added_token_count = 1283;
    source->tokenizer_max_token_id = 129279;
    source->tokenizer_effective_vocab_size = 129280;
    source->tokenizer_model_max_length = 1048576;
    source->generation_bos_token_id = 0;
    source->generation_eos_token_id = 1;
    source->generation_from_model_config = 1;
    source->generation_do_sample = 1;
    source->tie_word_embeddings = 0;
}

static int arch_ir_build(yvex_source_verification *source,
                         yvex_deepseek_v4_ir **ir,
                         yvex_deepseek_v4_ir_failure *failure)
{
    yvex_error err;

    yvex_error_clear(&err);
    return yvex_model_register_deepseek_v4()->ir.build(ir, source, failure, &err);
}

static int arch_ir_expect_failure(
    yvex_source_verification *source,
    yvex_deepseek_v4_ir_failure_code expected,
    const char *message)
{
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure failure;

    YVEX_TEST_ASSERT(arch_ir_build(source, &ir, &failure) != YVEX_OK,
                     message);
    YVEX_TEST_ASSERT(ir == NULL, "refused architecture publishes no IR");
    YVEX_TEST_ASSERT(failure.code == expected, message);
    return 0;
}

static int test_arch_ir_golden_topology(void)
{
    yvex_source_verification source;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure failure;
    const yvex_deepseek_v4_model_spec *model;
    const yvex_deepseek_v4_layer_spec *swa;
    const yvex_deepseek_v4_layer_spec *csa;
    const yvex_deepseek_v4_layer_spec *hca;
    const yvex_deepseek_v4_auxiliary_spec *draft_first;
    const yvex_deepseek_v4_auxiliary_spec *draft_middle;
    const yvex_deepseek_v4_auxiliary_spec *draft_final;
    const yvex_graph_execution_binding *graph_execution;
    yvex_model_execution_descriptor execution;
    yvex_error err;
    int execution_rc;
    char logical_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];

    arch_ir_verification_fixture(&source);
    YVEX_TEST_ASSERT(arch_ir_build(&source, &ir, &failure) == YVEX_OK && ir,
                     "exact verified target builds architecture IR");
    model = yvex_model_register_deepseek_v4()->ir.model(ir);
    YVEX_TEST_ASSERT(model && model->main_layer_count == 43 &&
                     model->auxiliary_layer_count == 3 &&
                     model->hidden_size == 4096 &&
                     model->vocabulary_size == 129280 &&
                     model->maximum_context == 1048576,
                     "global DeepSeek-V4-Flash-DSpark geometry is normalized");
    graph_execution = yvex_graph_execution_find(
        0ull, 0ull, "deepseek4-v4-flash-dspark");
    YVEX_TEST_ASSERT(
        graph_execution && yvex_model_register_deepseek_v4()->transform.architecture_identity(
                       ir, logical_identity),
        "compiled execution and logical architecture identities are available");
    YVEX_TEST_ASSERT_STREQ(
        graph_execution->logical_transform_identity,
        YVEX_DEEPSEEK_CURRENT_LOGICAL_TRANSFORM_IDENTITY,
        "execution binding is bound to the admitted DSpark Transformation IR");
    YVEX_TEST_ASSERT(
        strcmp(logical_identity, graph_execution->logical_transform_identity) != 0,
        "logical architecture and Transformation IR identities remain distinct");
    YVEX_TEST_ASSERT(
        strcmp(logical_identity,
               YVEX_DEEPSEEK_QUANT_IMATRIX_SOURCE_IDENTITY) != 0,
        "current DSpark identity cannot impersonate retained predecessor calibration");
    execution_rc = yvex_model_register_deepseek_v4()->ir.execution_descriptor
                       ? yvex_model_register_deepseek_v4()->ir.execution_descriptor(
                             ir, logical_identity, &execution, &err)
                       : YVEX_ERR_STATE;
    if (execution_rc != YVEX_OK)
        fprintf(stderr, "model execution projection: %s\n", yvex_error_message(&err));
    YVEX_TEST_ASSERT(
        execution_rc == YVEX_OK,
        "family projection seals a source-derived execution descriptor");
    YVEX_TEST_ASSERT(
        execution.maximum_context == 1048576ull &&
            execution.original_context == 65536ull &&
            execution.rope_theta == 10000ull &&
            execution.compressed_rope_theta == 160000ull &&
            execution.layer_count == 43ull && execution.hidden_width == 4096ull &&
            execution.vocabulary_size == 129280ull &&
            execution.attention_heads == 64ull && execution.kv_heads == 1ull &&
            execution.head_width == 512ull && execution.swa_layers == 2ull &&
            execution.csa_layers == 21ull && execution.hca_layers == 20ull,
        "family projection preserves exact model and attention geometry");
    YVEX_TEST_ASSERT(
        execution.routed_experts == 256ull && execution.experts_per_row == 6ull &&
            execution.shared_experts == 1ull && execution.proposal_width == 5ull &&
            execution.verification_width_maximum == 6ull &&
            execution.target_feature_count == 3ull &&
            execution.target_feature_layers[0] == 40ull &&
            execution.target_feature_layers[1] == 41ull &&
            execution.target_feature_layers[2] == 42ull &&
            execution.markov_rank == 256ull && execution.residual_streams == 4ull &&
            execution.mhc_sinkhorn_iterations == 20ull &&
            execution.hash_router_layer_count == 3ull &&
            execution.routed_scaling_factor == 1.5 && execution.activation_limit == 10.0,
        "family projection preserves exact MoE and DSpark geometry");
    YVEX_TEST_ASSERT(
            yvex_sha256_hex_valid(execution.source_model_identity) &&
            yvex_sha256_hex_valid(execution.attention_schedule_identity) &&
            yvex_sha256_hex_valid(execution.persistent_state_identity),
        "family projection seals independent source, schedule, and state identities");
    YVEX_TEST_ASSERT(
        model->dspark.present && model->dspark.schema_version == 1u &&
            model->dspark.block_size == 5u &&
            model->dspark.noise_token_id == 128799u &&
            model->dspark.target_layer_count == 3u &&
            model->dspark.target_layer_ids[0] == 40u &&
            model->dspark.target_layer_ids[1] == 41u &&
            model->dspark.target_layer_ids[2] == 42u &&
            model->dspark.concatenated_feature_width == 12288u &&
            model->dspark.draft_layer_count == 3u &&
            model->dspark.markov_rank == 256u &&
            model->dspark.final_draft_layer == 2u &&
            model->dspark.parallel_block_backbone &&
            model->dspark.sequential_markov &&
            model->dspark.confidence_available &&
            model->dspark.target_verification_required,
        "DSpark source facts become one exact typed architecture contract");
    YVEX_TEST_ASSERT(model->swa_layer_count == 2 &&
                     model->csa_layer_count == 21 &&
                     model->hca_layer_count == 20,
                     "hybrid attention counts match exact schedule");
    YVEX_TEST_ASSERT(model->hash_router_layer_count == 3 &&
                     model->learned_router_layer_count == 40,
                     "main-layer router classes are explicit");
    YVEX_TEST_ASSERT(model->final_mhc.residual_streams == 4 &&
                     model->final_mhc.expanded_width == 16384 &&
                     model->final_mhc.mixing_rows == 24 &&
                     model->final_mhc.mixing_columns == 16384 &&
                     model->final_mhc.sinkhorn_iterations == 20,
                     "mHC dimensions and iteration policy are explicit");
    YVEX_TEST_ASSERT(model->final_mhc_post_required &&
                     model->final_mhc_head_required &&
                     model->final_norm_after_mhc_head &&
                     model->final_mhc_head.function_rows == 4 &&
                     model->final_mhc_head.function_columns == 16384 &&
                     model->final_mhc_head.base_width == 4 &&
                     model->final_mhc_head.scale_width == 1,
                     "final mHC collapse precedes final norm");
    YVEX_TEST_ASSERT(model->embedding.required && model->output.required &&
                     !model->output.tied_to_embedding &&
                     model->tokenizer.vocabulary_size == 129280 &&
                     model->tokenizer.bos_token_id == 0 &&
                     model->tokenizer.eos_token_id == 1,
                     "embedding output and tokenizer requirements agree");
    YVEX_TEST_ASSERT(model->source_constraint.quant_block_rows == 128 &&
                     model->source_constraint.quant_block_columns == 128 &&
                     model->source_constraint.fp4_packing_factor == 2 &&
                     model->source_constraint.fp4_scale_group_width == 32 &&
                     model->source_constraint.fp4_physical_dtype ==
                         YVEX_NATIVE_DTYPE_I8 &&
                     model->source_constraint.scale_dtype ==
                         YVEX_NATIVE_DTYPE_F8_E8M0 &&
                     model->source_payload_bytes_read == 0 &&
                     model->source_header_scan_count == 1,
                     "source constraints remain typed without payload reads");
    YVEX_TEST_ASSERT_STREQ(
        yvex_model_register_deepseek_v4()->ir.source_quantization_name(
            model->source_constraint.quantization),
        "fp8-e4m3-ue8m0-dynamic",
        "source quantization rendering projects the typed constraint");
    YVEX_TEST_ASSERT_STREQ(
        model->sglang_revision,
        "96a04cb13f9c3ed86028e090784a9eb059cf5318",
        "SGLang interpretation baseline is immutable");
    YVEX_TEST_ASSERT_STREQ(
        model->vllm_revision,
        "8df14cfc8c8a09b4e57f082e59593a3abce4ffb3",
        "vLLM interpretation baseline is immutable");
    YVEX_TEST_ASSERT_STREQ(
        model->hadamard_revision,
        "Dao-AILab/fast-hadamard-transform:v1.1.0.post2:"
        "e7706faf8d1c3b9f241e36860640ad1dac644ede",
        "Hadamard transform authority is pinned");
    YVEX_TEST_ASSERT(model->runtime_numeric_schema_version == 2 &&
                     model->runtime_compute_policy_count == 1 &&
                     model->runtime_activation_policy_count == 3 &&
                     model->runtime_sparse_topk_policy_count == 1,
                     "runtime numeric schema is explicit");

    swa = yvex_model_register_deepseek_v4()->ir.layer_at(ir, 0);
    csa = yvex_model_register_deepseek_v4()->ir.layer_at(ir, 2);
    hca = yvex_model_register_deepseek_v4()->ir.layer_at(ir, 3);
    YVEX_TEST_ASSERT(swa && csa && hca &&
                     swa->compute_contract ==
                         YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1 &&
                     csa->compute_contract ==
                         YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1 &&
                     hca->compute_contract ==
                         YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1 &&
                     swa->attention_class == YVEX_ATTENTION_CLASS_SWA &&
                     csa->attention_class == YVEX_ATTENTION_CLASS_CSA &&
                     hca->attention_class == YVEX_ATTENTION_CLASS_HCA,
                     "every attention class has an explicit layer descriptor");
    YVEX_TEST_ASSERT(swa->head_dimension == 512 &&
                     swa->rope_head_dimension == 64 &&
                     swa->non_rope_head_dimension == 448 &&
                     swa->position.theta == 10000 &&
                     swa->position.original_context == 0 &&
                     csa->position.theta == 160000 &&
                     csa->position.original_context == 65536 &&
                     hca->position.theta == 160000 &&
                     hca->position.original_context == 65536,
                     "SWA disables YaRN while compressed attention retains it");
    YVEX_TEST_ASSERT(csa->compressor_required && csa->indexer_required &&
                     csa->indexer_heads == 64 &&
                     csa->indexer_head_dimension == 128 &&
                     csa->indexer_topk == 512 &&
                     csa->kv.requires_indexer_cache,
                     "CSA compressor indexer and KV state are explicit");
    YVEX_TEST_ASSERT(
        swa->attention_kv_activation.required &&
        swa->attention_kv_activation.quantization ==
            YVEX_ATTENTION_QUANT_FP8_E4M3_UE8M0_FAKE_DEQUANT &&
        swa->attention_kv_activation.block_width == 64 &&
        swa->attention_kv_activation.pre_transform ==
            YVEX_ATTENTION_TRANSFORM_NONE &&
        !swa->compressor_activation.required,
        "SWA owns only runtime KV fake quantization");
    YVEX_TEST_ASSERT(
        csa->compressor_activation.required &&
        csa->compressor_activation.block_width == 64 &&
        csa->compressor_rotated_activation.required &&
        csa->compressor_rotated_activation.quantization ==
            YVEX_ATTENTION_QUANT_FP4_E2M1_UE8M0_FAKE_DEQUANT &&
        csa->compressor_rotated_activation.block_width == 32 &&
        csa->compressor_rotated_activation.pre_transform ==
            YVEX_ATTENTION_TRANSFORM_DAO_FHT_V1_1_0_POST2 &&
        csa->compressor_rotated_activation.zero_pad_hadamard_to_power_of_two,
        "CSA separates source 128x128 quantization from runtime activation quantization");
    YVEX_TEST_ASSERT(
        csa->indexer_query_activation.required &&
        csa->indexer_query_activation.block_width == 32 &&
        csa->sparse_topk.required &&
        csa->sparse_topk.policy ==
            YVEX_ATTENTION_TOPK_SCORE_DESC_ORDINAL_ASC_V1 &&
        csa->sparse_topk.k == 512 &&
        csa->sparse_topk.equal_score_ordinal_ascending &&
        csa->sparse_topk.duplicate_ordinal_refused,
        "CSA owns deterministic sparse top-k policy");
    YVEX_TEST_ASSERT(
        hca->compressor_activation.required &&
        hca->compressor_activation.block_width == 64 &&
        !hca->compressor_rotated_activation.required &&
        !hca->indexer_query_activation.required &&
        !hca->sparse_topk.required,
        "HCA owns non-rotated runtime compressor quantization without CSA indexer policy");
    YVEX_TEST_ASSERT(csa->attention_input_norm.required &&
                     csa->attention_input_norm.width == 4096 &&
                     csa->post_attention_ffn_norm.required &&
                     csa->post_attention_ffn_norm.width == 4096 &&
                     csa->tensors.compressor_ape_rows == 4 &&
                     csa->tensors.compressor_ape_columns == 1024 &&
                     csa->tensors.indexer_query_rows == 8192 &&
                     csa->tensors.indexer_weight_rows == 64,
                     "norm and compressed-attention tensor geometry is explicit");
    YVEX_TEST_ASSERT(hca->compressor_required && !hca->indexer_required &&
                     hca->kv.requires_compressed_core &&
                     !hca->kv.requires_indexer_cache,
                     "HCA uses heavy compression without sparse indexer");
    YVEX_TEST_ASSERT(swa->moe.router_class ==
                         YVEX_DEEPSEEK_V4_ROUTER_HASH_TOKEN_ID &&
                     swa->moe.requires_token_ids &&
                     swa->moe.hash_table_rows == 129280 &&
                     swa->moe.hash_table_columns == 6,
                     "hash MoE requires token-id routing table");
    YVEX_TEST_ASSERT(hca->moe.router_class ==
                         YVEX_DEEPSEEK_V4_ROUTER_LEARNED_HIDDEN_STATE &&
                     hca->moe.requires_hidden_state &&
                     hca->moe.requires_correction_bias &&
                     hca->moe.correction_bias_width == 256,
                     "learned MoE requires hidden-state noaux routing");

    draft_first = yvex_model_register_deepseek_v4()->ir.auxiliary_at(ir, 0);
    draft_middle = yvex_model_register_deepseek_v4()->ir.auxiliary_at(ir, 1);
    draft_final = yvex_model_register_deepseek_v4()->ir.auxiliary_at(ir, 2);
    YVEX_TEST_ASSERT(draft_first && draft_first->layer.layer_index == 43 &&
                     draft_first->layer.attention_class ==
                         YVEX_ATTENTION_CLASS_SWA &&
                     draft_first->layer.moe.router_class ==
                         YVEX_DEEPSEEK_V4_ROUTER_LEARNED_HIDDEN_STATE &&
                     draft_first->has_feature_projection &&
                     draft_first->feature_projection_input == 12288u &&
                     draft_first->feature_projection_output == 4096u &&
                     draft_first->has_feature_norm &&
                     !draft_first->has_markov_head &&
                     draft_middle && draft_middle->layer.layer_index == 44u &&
                     !draft_middle->has_feature_projection &&
                     !draft_middle->has_output_norm &&
                     draft_final && draft_final->layer.layer_index == 45u &&
                     draft_final->has_output_norm &&
                     draft_final->has_markov_head &&
                     draft_final->markov_rank == 256u &&
                     draft_final->markov_vocabulary_size == 129280u &&
                     draft_final->has_confidence_head &&
                     draft_final->confidence_input_width == 4352u &&
                     draft_final->has_separate_mhc_head &&
                     draft_final->mhc_head.function_rows == 4 &&
                     draft_final->mhc_head.function_columns == 16384 &&
                     draft_first->shares_embedding &&
                     draft_final->shares_output_head,
                     "three DSpark blocks own distinct source-backed roles");
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->ir.layer_at(ir, 43) == NULL &&
                     yvex_model_register_deepseek_v4()->ir.auxiliary_at(ir, 3) == NULL,
                     "borrowed accessors are bounds checked");
    yvex_model_register_deepseek_v4()->ir.close(ir);
    return 0;
}

static int test_arch_ir_position_authority(void)
{
    yvex_source_verification source;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure failure;
    unsigned long long i;

    arch_ir_verification_fixture(&source);
    source.rope_original_context = 32768;
    YVEX_TEST_ASSERT(arch_ir_build(&source, &ir, &failure) == YVEX_OK && ir,
                     "position-authority fixture builds");
    for (i = 0u; i < source.num_hidden_layers; ++i) {
        const yvex_deepseek_v4_layer_spec *layer =
            yvex_model_register_deepseek_v4()->ir.layer_at(ir, i);

        YVEX_TEST_ASSERT(layer != NULL,
                         "position-authority layer resolves");
        if (source.compress_ratios[i] == 0u) {
            YVEX_TEST_ASSERT(
                layer->attention_class == YVEX_ATTENTION_CLASS_SWA &&
                    layer->position.theta == source.rope_theta &&
                    layer->position.original_context == 0u,
                "every SWA layer uses base RoPE with YaRN disabled");
        } else {
            YVEX_TEST_ASSERT(
                layer->attention_class != YVEX_ATTENTION_CLASS_SWA &&
                    layer->position.theta == source.compress_rope_theta &&
                    layer->position.original_context ==
                        source.rope_original_context,
                "every compressed layer retains the admitted YaRN context");
        }
    }
    YVEX_TEST_ASSERT(
        yvex_model_register_deepseek_v4()->ir.auxiliary_at(ir, 0u)
                ->layer.position.original_context == 0u &&
            yvex_model_register_deepseek_v4()->ir.auxiliary_at(ir, 0u)
                ->layer.position.theta == source.rope_theta,
        "uncompressed draft attention also disables YaRN");
    yvex_model_register_deepseek_v4()->ir.close(ir);
    return 0;
}

static int test_arch_ir_refusal_matrix(void)
{
    yvex_source_verification source;

    arch_ir_verification_fixture(&source);
    source.verified = 0;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_SOURCE_NOT_VERIFIED,
            "unverified source refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    arch_ir_copy(source.repository_id, sizeof(source.repository_id),
                 "wrong/repository");
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_IDENTITY_MISMATCH,
            "wrong identity refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.config_valid = 0;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_SOURCE_NOT_VERIFIED,
            "incomplete sidecar evidence refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.header_shard_count = 47u;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_SOURCE_NOT_VERIFIED,
            "incomplete pinned shard inventory refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.dspark_block_size = 4u;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_DSPARK,
            "wrong DSpark block geometry refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.dspark_target_layer_ids[1] = 39u;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_DSPARK,
            "wrong DSpark target feature order refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.dspark_noise_token_id = source.vocab_size;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_DSPARK,
            "out-of-vocabulary DSpark noise token refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.dspark_markov_rank = 128u;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_DSPARK,
            "wrong DSpark Markov geometry refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.dspark_inference_layer_count = 2u;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_DSPARK,
            "wrong DSpark inference depth refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.compress_ratio_count = 43;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_SCHEDULE_LENGTH,
            "schedule length mismatch refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.compress_ratios[7] = 7;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_UNSUPPORTED_COMPRESSION,
            "unknown compression refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.compress_ratios[4] = 128;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_SCHEDULE_PATTERN,
            "non-alternating hybrid schedule refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.qk_rope_head_dim = 512;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_DIMENSION,
            "invalid head partition refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.o_groups = 7;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_GROUP_GEOMETRY,
            "invalid output grouping refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.rope_beta_slow = 64;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_POSITION,
            "invalid YaRN parameters refuse") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.use_cache = 0;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_POSITION,
            "missing cache-state requirement refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.hc_mult = 0;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_MHC,
            "invalid mHC dimensions refuse") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.hidden_size = ULLONG_MAX;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_ARITHMETIC_OVERFLOW,
            "mHC checked geometry overflow refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.num_hash_layers = 44;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_ROUTING,
            "excessive hash layer count refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.num_experts_per_tok = 257;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_EXPERT_TOPK,
            "invalid expert top-k refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.tokenizer_effective_vocab_size = 129279;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_TOKENIZER_OUTPUT_MISMATCH,
            "tokenizer and output vocabulary mismatch refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    arch_ir_copy(source.quant_scale_format,
                 sizeof(source.quant_scale_format), "unknown");
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_UNSUPPORTED_SOURCE_CONSTRAINT,
            "unknown source quantization refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    arch_ir_copy(source.inventory_authority,
                 sizeof(source.inventory_authority), "header-derived");
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_SOURCE_FACT_MISSING,
            "header-derived inventory cannot admit release IR") != 0) return 1;

    arch_ir_verification_fixture(&source);
    source.quant_block_rows = 64;
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_UNSUPPORTED_SOURCE_CONSTRAINT,
            "non-pinned quant block geometry refuses") != 0) return 1;

    arch_ir_verification_fixture(&source);
    arch_ir_copy(source.rms_norm_eps, sizeof(source.rms_norm_eps), "nan");
    if (arch_ir_expect_failure(&source,
            YVEX_DEEPSEEK_V4_IR_FAILURE_NUMERIC_VALUE,
            "non-finite numeric source fact refuses") != 0) return 1;
    return 0;
}

static int test_arch_ir_payload_manifest_stage(void)
{
    yvex_source_verification source;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure failure;

    arch_ir_verification_fixture(&source);
    arch_ir_copy(source.verification_stage,
                 sizeof(source.verification_stage),
                 "exact-source-payload-verified");
    source.manifest_payload_trusted = 1;
    YVEX_TEST_ASSERT(arch_ir_build(&source, &ir, &failure) == YVEX_OK && ir,
                     "trusted v3 payload manifest remains strict IR input");
    yvex_model_register_deepseek_v4()->ir.close(ir);

    source.manifest_payload_trusted = 0;
    if (arch_ir_expect_failure(
            &source, YVEX_DEEPSEEK_V4_IR_FAILURE_SOURCE_FACT_MISSING,
            "untrusted payload-stage label cannot bypass source admission") !=
        0) return 1;
    return 0;
}

typedef struct {
    unsigned int call_count;
    unsigned int fail_call;
    unsigned int live_allocations;
} arch_ir_allocator_state;

static void *arch_ir_test_allocate(size_t size, void *context)
{
    arch_ir_allocator_state *state = (arch_ir_allocator_state *)context;
    void *allocation;

    if (state->call_count++ == state->fail_call) return NULL;
    allocation = malloc(size);
    if (allocation) state->live_allocations++;
    return allocation;
}

static void arch_ir_test_release(void *allocation, void *context)
{
    arch_ir_allocator_state *state = (arch_ir_allocator_state *)context;

    if (!allocation) return;
    free(allocation);
    state->live_allocations--;
}

static int test_arch_ir_lifetime_and_allocation(void)
{
    yvex_source_verification *source;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure failure;
    yvex_error err;
    unsigned int fail_call;

    source = (yvex_source_verification *)malloc(sizeof(*source));
    YVEX_TEST_ASSERT(source != NULL, "allocate source evidence fixture");
    arch_ir_verification_fixture(source);
    YVEX_TEST_ASSERT(arch_ir_build(source, &ir, &failure) == YVEX_OK && ir,
                     "build owned lifetime IR");
    memset(source, 0, sizeof(*source));
    free(source);
    YVEX_TEST_ASSERT_STREQ(yvex_model_register_deepseek_v4()->ir.model(ir)->repository,
                           yvex_source_release_identity()->upstream_repo_id,
                           "IR remains valid after source evidence release");
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->ir.layer_at(ir, 42)->layer_index == 42,
                     "owned layer collection survives source release");
    yvex_model_register_deepseek_v4()->ir.close(ir);

    for (fail_call = 0u; fail_call < 3u; ++fail_call) {
        yvex_source_verification fixture;
        arch_ir_allocator_state state;
        yvex_deepseek_v4_ir_allocator allocator;
        int rc;

        arch_ir_verification_fixture(&fixture);
        memset(&state, 0, sizeof(state));
        state.fail_call = fail_call;
        allocator.allocate = arch_ir_test_allocate;
        allocator.release = arch_ir_test_release;
        allocator.context = &state;
        ir = NULL;
        yvex_error_clear(&err);
        rc = yvex_model_register_deepseek_v4()->ir.build_with_allocator(
            &ir, &fixture, &allocator, &failure, &err);
        YVEX_TEST_ASSERT(rc == YVEX_ERR_NOMEM && ir == NULL &&
                         failure.code ==
                             YVEX_DEEPSEEK_V4_IR_FAILURE_ALLOCATION,
                         "allocation failure refuses partial IR");
        YVEX_TEST_ASSERT(state.live_allocations == 0,
                         "partial construction releases every allocation");
    }
    return 0;
}

static int report_has(const yvex_model_target_report *report,
                      const char *text)
{
    unsigned long i;

    for (i = 0u; report && i < report->row_count; ++i) {
        if (strstr(report->rows[i].value, text)) return 1;
    }
    return 0;
}

static int test_arch_ir_report_consumer_and_family_preservation(void)
{
    yvex_model_target_request request;
    yvex_model_target_report report;
    yvex_source_verification source;
    const yvex_family_compiler_adapter *compiler =
        yvex_compiler_family_deepseek_v4();
    yvex_semantic_model_ir *semantic = NULL;
    yvex_error err;

    memset(&request, 0, sizeof(request));
    memset(&report, 0, sizeof(report));
    request.kind = YVEX_MODEL_TARGET_COMMAND_CLASS_PROFILE;
    request.mode = YVEX_MODEL_TARGET_OUTPUT_NORMAL;
    arch_ir_copy(request.target_id, sizeof(request.target_id), "qwen3-8b");
    arch_ir_copy(request.models_root, sizeof(request.models_root),
                 "build/tests/missing-arch-ir-qwen");
    YVEX_TEST_ASSERT(yvex_model_target_report_build(
                         &request, &report, &err) == YVEX_OK &&
                     report_has(&report, "qwen-source-model-class-profile"),
                     "Qwen evidence path remains intact");
    yvex_model_target_report_close(&report);

    memset(&request, 0, sizeof(request));
    memset(&report, 0, sizeof(report));
    request.kind = YVEX_MODEL_TARGET_COMMAND_CLASS_PROFILE;
    request.mode = YVEX_MODEL_TARGET_OUTPUT_NORMAL;
    arch_ir_copy(request.target_id, sizeof(request.target_id),
                 "gemma-4-12b-it");
    arch_ir_copy(request.models_root, sizeof(request.models_root),
                 "build/tests/missing-arch-ir-gemma");
    YVEX_TEST_ASSERT(yvex_model_target_report_build(
                         &request, &report, &err) == YVEX_OK &&
                     report_has(&report, "gemma-source-model-class-profile"),
                     "Gemma evidence path remains intact");
    yvex_model_target_report_close(&report);

    arch_ir_verification_fixture(&source);
    YVEX_TEST_ASSERT(
        compiler && compiler->operator_graph_build && compiler->execution_capabilities &&
            compiler->transformer_policy && compiler->physical_variant,
        "DeepSeek compiler exposes semantic lowering without deployment policy");
    YVEX_TEST_ASSERT(compiler->binding_pipeline &&
                         compiler->binding_pipeline->semantic_model_build(
                             &semantic, &source, &err) == YVEX_OK,
                     "DeepSeek fixture compiles to generic Semantic Model IR");
    memset(&report, 0, sizeof(report));
    YVEX_TEST_ASSERT(
        yvex_model_target_report_project_semantic_detail(
            &report, semantic, &source, NULL, &err) == YVEX_OK &&
            report.detail_kind == YVEX_MODEL_TARGET_DETAIL_MODEL_ARCHITECTURE &&
            report.detail.architecture.maximum_context == source.max_position_embeddings &&
            report.detail.architecture.layer_count == source.num_hidden_layers &&
            report.detail.architecture.draft_count == source.dspark_inference_layer_count &&
            report.detail.architecture.routed_experts == source.n_routed_experts &&
            report.detail.architecture.experts_per_token == source.num_experts_per_tok &&
            strcmp(report.detail.architecture.paper_revision, "arXiv:2606.19348v1") == 0 &&
            strcmp(report.detail.architecture.sglang_revision,
                   "96a04cb13f9c3ed86028e090784a9eb059cf5318") == 0 &&
            strcmp(report.detail.architecture.vllm_revision,
                   "8df14cfc8c8a09b4e57f082e59593a3abce4ffb3") == 0,
        "family-neutral report projection retains source-authored architecture facts");
    yvex_semantic_model_ir_close(&semantic);
    yvex_model_target_report_close(&report);
    return 0;
}

/* Every runtime numeric policy field is semantic input to logical identity. */
static int test_runtime_numeric_identity_field_coverage(void)
{
    yvex_source_verification source;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure failure;
    yvex_deepseek_v4_layer_spec *layer;
    yvex_attention_activation_policy activation;
    yvex_attention_topk_policy topk;
    char baseline[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char changed[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    yvex_error err;

    arch_ir_verification_fixture(&source);
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->ir.build(
                         &ir, &source, &failure, &err) == YVEX_OK,
                     "identity fixture architecture builds");
    layer = (yvex_deepseek_v4_layer_spec *)
        yvex_model_register_deepseek_v4()->ir.layer_at(ir, 2ull);
    YVEX_TEST_ASSERT(layer && yvex_model_register_deepseek_v4()->transform.architecture_identity(
                                  ir, baseline),
                     "baseline logical identity exists");
    activation = layer->compressor_rotated_activation;
    topk = layer->sparse_topk;
#define ASSERT_ID_MUTATION(change, restore, label) do {                       \
        change;                                                                \
        YVEX_TEST_ASSERT(                                                      \
            yvex_model_register_deepseek_v4()->transform.architecture_identity(ir, changed) &&     \
                strcmp(baseline, changed) != 0, label);                        \
        restore;                                                               \
    } while (0)
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.required ^= 1,
                       layer->compressor_rotated_activation = activation,
                       "activation required changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.stage =
                           YVEX_ATTENTION_ACTIVATION_KV_NON_ROPE,
                       layer->compressor_rotated_activation = activation,
                       "activation stage changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.quantization =
                           YVEX_ATTENTION_QUANT_FP8_E4M3_UE8M0_FAKE_DEQUANT,
                       layer->compressor_rotated_activation = activation,
                       "activation qtype changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.block_axis =
                           YVEX_ATTENTION_AXIS_NONE,
                       layer->compressor_rotated_activation = activation,
                       "activation axis changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.block_width += 1ull,
                       layer->compressor_rotated_activation = activation,
                       "activation block width changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.scale_format =
                           YVEX_ATTENTION_SCALE_NONE,
                       layer->compressor_rotated_activation = activation,
                       "activation scale format changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.scale_dtype =
                           YVEX_NATIVE_DTYPE_F32,
                       layer->compressor_rotated_activation = activation,
                       "activation scale dtype changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.pre_transform =
                           YVEX_ATTENTION_TRANSFORM_NONE,
                       layer->compressor_rotated_activation = activation,
                       "Hadamard policy changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.tail_policy =
                           YVEX_ATTENTION_TAIL_NONE,
                       layer->compressor_rotated_activation = activation,
                       "activation tail changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.nonfinite_policy =
                           (yvex_attention_nonfinite_policy)1,
                       layer->compressor_rotated_activation = activation,
                       "activation finite policy changes identity");
    ASSERT_ID_MUTATION(layer->compressor_rotated_activation.fake_quant_inplace ^= 1,
                       layer->compressor_rotated_activation = activation,
                       "activation publication mode changes identity");
    ASSERT_ID_MUTATION(
        layer->compressor_rotated_activation.zero_pad_hadamard_to_power_of_two ^= 1,
        layer->compressor_rotated_activation = activation,
        "Hadamard padding changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.required ^= 1,
                       layer->sparse_topk = topk,
                       "top-k requirement changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.version += 1u,
                       layer->sparse_topk = topk,
                       "top-k version changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.policy =
                           YVEX_ATTENTION_TOPK_NONE,
                       layer->sparse_topk = topk,
                       "top-k policy changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.k -= 1ull,
                       layer->sparse_topk = topk,
                       "top-k cardinality changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.reject_nonfinite ^= 1,
                       layer->sparse_topk = topk,
                       "top-k finite policy changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.score_descending ^= 1,
                       layer->sparse_topk = topk,
                       "top-k score ordering changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.equal_score_ordinal_ascending ^= 1,
                       layer->sparse_topk = topk,
                       "top-k tie ordering changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.plus_zero_equals_minus_zero ^= 1,
                       layer->sparse_topk = topk,
                       "top-k signed-zero rule changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.duplicate_ordinal_refused ^= 1,
                       layer->sparse_topk = topk,
                       "top-k duplicate rule changes identity");
    ASSERT_ID_MUTATION(layer->sparse_topk.output_ranked_order ^= 1,
                       layer->sparse_topk = topk,
                       "top-k output ordering changes identity");
#undef ASSERT_ID_MUTATION
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->transform.architecture_identity(
                         ir, changed) && strcmp(baseline, changed) == 0,
                     "restored numeric policies restore identity");
    yvex_model_register_deepseek_v4()->ir.close(ir);
    return 0;
}

int yvex_test_deepseek_arch_ir(void)
{
    if (test_arch_ir_golden_topology() != 0) return 1;
    if (test_arch_ir_position_authority() != 0) return 1;
    if (test_arch_ir_refusal_matrix() != 0) return 1;
    if (test_arch_ir_payload_manifest_stage() != 0) return 1;
    if (test_arch_ir_lifetime_and_allocation() != 0) return 1;
    if (test_runtime_numeric_identity_field_coverage() != 0) return 1;
    if (test_arch_ir_report_consumer_and_family_preservation() != 0) return 1;
    return 0;
}
