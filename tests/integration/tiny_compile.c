#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/compiler.h>
#include <yvex/internal/execution.h>
#include <yvex/internal/generation.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/model.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/operator_graph.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/tokenizer.h>

#include "src/runtime/private.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TINY_ADAPTER_ID 0x54494e59ull
#define TINY_ADAPTER_VERSION 1ull
#define TINY_MAPPING_ID 0x54494e5950524f4aull
#define TINY_SOURCE_ID 0x54494e59534f5552ull

static const char tiny_source_identity[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
static const char tiny_logical_identity[] =
    "2222222222222222222222222222222222222222222222222222222222222222";
static const char tiny_schedule_identity[] =
    "3333333333333333333333333333333333333333333333333333333333333333";
static const char tiny_state_identity[] =
    "4444444444444444444444444444444444444444444444444444444444444444";
static const char tiny_transform_identity[] =
    "5555555555555555555555555555555555555555555555555555555555555555";
static const char tiny_profile_identity[] =
    "6666666666666666666666666666666666666666666666666666666666666666";
static const char tiny_numeric_identity[] =
    "7777777777777777777777777777777777777777777777777777777777777777";

typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
    yvex_materialization_plan *materialization_plan;
    yvex_materialization_session *materialization;
    yvex_runtime_descriptor *descriptor;
    yvex_semantic_model_ir *semantic;
    yvex_attention_plan *attention;
    yvex_operator_graph_ir *operators;
    yvex_physical_execution_ir *physical;
    yvex_compiled_model_plan *compiled;
    yvex_complete_artifact_admission admission;
    yvex_artifact_physical_compatibility compatibility;
    yvex_runtime_capabilities capabilities;
    yvex_transformer_family_policy transformer;
    yvex_logits_family_policy logits;
    yvex_speculation_family_policy speculation;
    yvex_tokenizer_family_policy tokenizer;
} tiny_fixture;

typedef struct {
    const char *name;
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    unsigned long long layer;
    unsigned long long experts;
} tiny_terminal;

static const tiny_terminal tiny_terminals[] = {
    {"tiny.token_embedding", YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
     YVEX_TENSOR_COLLECTION_GLOBAL, YVEX_TENSOR_SCOPE_GLOBAL,
     YVEX_MATERIALIZATION_NO_INDEX, 0ull},
    {"tiny.hc_head_function", YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION,
     YVEX_TENSOR_COLLECTION_MHC, YVEX_TENSOR_SCOPE_GLOBAL,
     YVEX_MATERIALIZATION_NO_INDEX, 0ull},
    {"tiny.hc_head_base", YVEX_TENSOR_ROLE_HC_HEAD_BASE,
     YVEX_TENSOR_COLLECTION_MHC, YVEX_TENSOR_SCOPE_GLOBAL,
     YVEX_MATERIALIZATION_NO_INDEX, 0ull},
    {"tiny.hc_head_scale", YVEX_TENSOR_ROLE_HC_HEAD_SCALE,
     YVEX_TENSOR_COLLECTION_MHC, YVEX_TENSOR_SCOPE_GLOBAL,
     YVEX_MATERIALIZATION_NO_INDEX, 0ull},
    {"tiny.output_norm", YVEX_TENSOR_ROLE_OUTPUT_NORM,
     YVEX_TENSOR_COLLECTION_GLOBAL, YVEX_TENSOR_SCOPE_GLOBAL,
     YVEX_MATERIALIZATION_NO_INDEX, 0ull},
    {"tiny.output_head", YVEX_TENSOR_ROLE_OUTPUT_HEAD,
     YVEX_TENSOR_COLLECTION_GLOBAL, YVEX_TENSOR_SCOPE_GLOBAL,
     YVEX_MATERIALIZATION_NO_INDEX, 0ull},
    {"tiny.layer.0.attention_sinks", YVEX_TENSOR_ROLE_ATTENTION_SINKS,
     YVEX_TENSOR_COLLECTION_ATTENTION, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.attention_norm", YVEX_TENSOR_ROLE_ATTENTION_NORM,
     YVEX_TENSOR_COLLECTION_NORM, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.hc_attention_function", YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION,
     YVEX_TENSOR_COLLECTION_MHC, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.hc_attention_base", YVEX_TENSOR_ROLE_HC_ATTENTION_BASE,
     YVEX_TENSOR_COLLECTION_MHC, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.hc_attention_scale", YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE,
     YVEX_TENSOR_COLLECTION_MHC, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.attention_q_a_norm", YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM,
     YVEX_TENSOR_COLLECTION_ATTENTION, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.attention_kv_norm", YVEX_TENSOR_ROLE_ATTENTION_KV_NORM,
     YVEX_TENSOR_COLLECTION_ATTENTION, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.attention_q_a", YVEX_TENSOR_ROLE_ATTENTION_Q_A,
     YVEX_TENSOR_COLLECTION_ATTENTION, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.attention_q_b", YVEX_TENSOR_ROLE_ATTENTION_Q_B,
     YVEX_TENSOR_COLLECTION_ATTENTION, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.attention_kv", YVEX_TENSOR_ROLE_ATTENTION_KV,
     YVEX_TENSOR_COLLECTION_ATTENTION, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.attention_out_a", YVEX_TENSOR_ROLE_ATTENTION_OUT_A,
     YVEX_TENSOR_COLLECTION_ATTENTION, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.attention_out_b", YVEX_TENSOR_ROLE_ATTENTION_OUT_B,
     YVEX_TENSOR_COLLECTION_ATTENTION, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.ffn_norm", YVEX_TENSOR_ROLE_FFN_NORM,
     YVEX_TENSOR_COLLECTION_NORM, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.hc_ffn_function", YVEX_TENSOR_ROLE_HC_FFN_FUNCTION,
     YVEX_TENSOR_COLLECTION_MHC, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.hc_ffn_base", YVEX_TENSOR_ROLE_HC_FFN_BASE,
     YVEX_TENSOR_COLLECTION_MHC, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.hc_ffn_scale", YVEX_TENSOR_ROLE_HC_FFN_SCALE,
     YVEX_TENSOR_COLLECTION_MHC, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.router", YVEX_TENSOR_ROLE_MOE_ROUTER,
     YVEX_TENSOR_COLLECTION_ROUTER, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.router_table", YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE,
     YVEX_TENSOR_COLLECTION_ROUTER, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.expert_gate", YVEX_TENSOR_ROLE_MOE_EXPERT_GATE,
     YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 2ull},
    {"tiny.layer.0.expert_up", YVEX_TENSOR_ROLE_MOE_EXPERT_UP,
     YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 2ull},
    {"tiny.layer.0.expert_down", YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN,
     YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 2ull},
    {"tiny.layer.0.shared_gate", YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE,
     YVEX_TENSOR_COLLECTION_SHARED_EXPERT, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.shared_up", YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP,
     YVEX_TENSOR_COLLECTION_SHARED_EXPERT, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
    {"tiny.layer.0.shared_down", YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN,
     YVEX_TENSOR_COLLECTION_SHARED_EXPERT, YVEX_TENSOR_SCOPE_MAIN_LAYER, 0ull, 0ull},
};

static int tiny_terminal_find(const void *context, const char *name,
                              yvex_materialization_terminal *out)
{
    size_t index;
    if (context != tiny_terminals || !name || !out) return 0;
    for (index = 0u; index < sizeof(tiny_terminals) / sizeof(tiny_terminals[0]); ++index) {
        const tiny_terminal *terminal = &tiny_terminals[index];
        if (strcmp(name, terminal->name) != 0) continue;
        *out = (yvex_materialization_terminal){
            index, terminal->role, terminal->collection, terminal->scope,
            terminal->layer, YVEX_MATERIALIZATION_NO_INDEX, terminal->experts};
        return 1;
    }
    return 0;
}

static int tiny_attention_layer(const void *context, unsigned long long index,
                                yvex_semantic_attention_layer *out)
{
    (void)context;
    if (index != 0ull || !out) return 0;
    memset(out, 0, sizeof(*out));
    out->ordinal = 0ull;
    out->layer_index = 0ull;
    out->predictor_index = YVEX_MATERIALIZATION_NO_INDEX;
    out->tensor_scope = YVEX_TENSOR_SCOPE_MAIN_LAYER;
    out->attention_class = YVEX_ATTENTION_CLASS_SWA;
    out->compute_contract = YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1;
    out->compression_ratio = 0ull;
    out->sliding_window = 4ull;
    out->query_heads = 1ull;
    out->kv_heads = 1ull;
    out->head_dimension = 4ull;
    out->rope_head_dimension = 4ull;
    out->query_lora_rank = 4ull;
    out->output_lora_rank = 4ull;
    out->output_groups = 1ull;
    out->output_group_input_width = 4ull;
    out->hidden_dimension = 4ull;
    out->rms_norm_epsilon = 1e-6;
    out->residual_stream_count = 1ull;
    out->residual_stream_width = 4ull;
    out->residual_expanded_width = 4ull;
    out->mhc_mixing_rows = 3ull;
    out->mhc_mixing_columns = 4ull;
    out->mhc_base_width = 3ull;
    out->mhc_scale_width = 3ull;
    out->mhc_sinkhorn_iterations = 1ull;
    out->attention_input_norm_width = 4ull;
    out->mhc_epsilon = 1e-6;
    out->mhc_residual_post_multiplier = 1.0;
    out->attention_input_norm_required = 1;
    out->mhc_attention_pre_and_post = 1;
    out->attention_input_norm_role = YVEX_TENSOR_ROLE_ATTENTION_NORM;
    out->mhc_function_role = YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION;
    out->mhc_base_role = YVEX_TENSOR_ROLE_HC_ATTENTION_BASE;
    out->mhc_scale_role = YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE;
    out->position.rope_dimension = 4ull;
    out->position.theta = 10000ull;
    out->position.scaling_factor = 1ull;
    out->position.original_context = 0ull;
    out->position.maximum_context = 8ull;
    return 1;
}

static int tiny_moe_layer(unsigned long long index,
                          const yvex_runtime_descriptor_summary *runtime,
                          const yvex_attention_layer_plan *attention,
                          yvex_moe_layer_plan *out, yvex_error *err)
{
    if (index != 0ull || !runtime || !attention || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tiny.moe", "tiny MoE layer is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->schema_version = YVEX_MOE_PLAN_SCHEMA_V1;
    out->ordinal = index;
    out->layer_index = attention->layer_index;
    out->predictor_index = attention->predictor_index;
    out->tensor_scope = attention->tensor_scope;
    out->router_class = YVEX_MOE_ROUTER_HASH_TOKEN_ID;
    out->scoring = YVEX_MOE_SCORING_SQRT_SOFTPLUS;
    out->topk_policy = YVEX_MOE_TOPK_NOAUX_TC;
    out->activation = YVEX_MOE_ACTIVATION_SILU;
    out->hidden_width = 4ull;
    out->residual_streams = 1ull;
    out->expanded_width = 4ull;
    out->mhc_mixing_rows = 3ull;
    out->mhc_sinkhorn_iterations = 1ull;
    out->routed_experts = 2ull;
    out->shared_experts = 1ull;
    out->experts_per_token = 1ull;
    out->expert_intermediate_width = 1ull;
    out->shared_intermediate_width = 1ull;
    out->hash_table_rows = 261ull;
    out->hash_table_columns = 1ull;
    out->rms_epsilon = 1e-6;
    out->mhc_epsilon = 1e-6;
    out->mhc_post_multiplier = 1.0;
    out->routed_scaling_factor = 1.0;
    out->activation_limit = 1.0;
    out->requires_token_ids = 1;
    out->normalize_topk_probabilities = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static const yvex_moe_family_api tiny_moe_api = {
    .adapter_id = TINY_ADAPTER_ID,
    .adapter_version = TINY_ADAPTER_VERSION,
    .project_layer = tiny_moe_layer,
};

static const yvex_graph_compiler_api tiny_graph_api = {
    .moe = &tiny_moe_api,
};

static void tiny_close(tiny_fixture *fixture)
{
    if (!fixture) return;
    yvex_compiled_model_plan_close(&fixture->compiled);
    yvex_physical_execution_ir_close(&fixture->physical);
    yvex_operator_graph_ir_close(&fixture->operators);
    yvex_attention_plan_close(fixture->attention);
    yvex_semantic_model_ir_close(&fixture->semantic);
    yvex_runtime_descriptor_close(fixture->descriptor);
    yvex_materialization_session_close(fixture->materialization);
    yvex_materialization_plan_close(fixture->materialization_plan);
    yvex_tensor_table_close(fixture->tensors);
    yvex_gguf_close(fixture->gguf);
    yvex_artifact_close(fixture->artifact);
    memset(fixture, 0, sizeof(*fixture));
}

static int tiny_open(tiny_fixture *fixture, const char *path, yvex_error *err)
{
    yvex_artifact_options options = {.path = path, .readonly = 1};
    int rc = yvex_artifact_open(&fixture->artifact, &options, err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&fixture->gguf, fixture->artifact, err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&fixture->tensors, fixture->gguf, err);
    return rc;
}

static int tiny_admit(tiny_fixture *fixture, yvex_error *err)
{
    yvex_complete_artifact_admission catalog = {0};
    yvex_artifact_catalog_contract contract = {0};
    yvex_artifact_file_identity identity = {0};
    yvex_artifact_payload_identity payload = {0};
    yvex_artifact_admission_failure failure = {0};
    unsigned long long index;
    int rc;
    if (yvex_artifact_identity_read_open(fixture->artifact, &identity, err) != YVEX_OK ||
        yvex_artifact_payload_identity_compute(
            fixture->artifact, fixture->gguf, 4096u, &payload, err) != YVEX_OK)
        return yvex_error_code(err);
    catalog.artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX;
    catalog.metadata_count = yvex_gguf_metadata_count(fixture->gguf);
    catalog.tensor_count = yvex_tensor_table_count(fixture->tensors);
    for (index = 0ull; index < catalog.tensor_count; ++index) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(fixture->tensors, index);
        if (!tensor || catalog.payload_bytes > ~0ull - tensor->storage_bytes) return YVEX_ERR_BOUNDS;
        catalog.payload_bytes += tensor->storage_bytes;
    }
    catalog.file_bytes = yvex_artifact_size(fixture->artifact);
    catalog.source_snapshot_identity = TINY_SOURCE_ID;
    catalog.mapping_identity = TINY_MAPPING_ID;
    snprintf(catalog.artifact_path, sizeof(catalog.artifact_path), "%s",
             yvex_artifact_path(fixture->artifact));
    snprintf(catalog.payload_identity, sizeof(catalog.payload_identity), "%s",
             payload.payload_byte_identity);
    snprintf(catalog.transform_identity, sizeof(catalog.transform_identity), "%s",
             tiny_transform_identity);
    snprintf(catalog.profile_identity, sizeof(catalog.profile_identity), "%s",
             tiny_profile_identity);
    snprintf(catalog.profile_name, sizeof(catalog.profile_name), "tiny-cpu-v1");
    snprintf(catalog.quant_execution_identity, sizeof(catalog.quant_execution_identity), "%064x", 8);
    snprintf(catalog.payload_plan_identity, sizeof(catalog.payload_plan_identity), "%064x", 9);
    snprintf(catalog.payload_byte_identity, sizeof(catalog.payload_byte_identity), "%s",
             payload.payload_byte_identity);
    snprintf(catalog.writer_plan_identity, sizeof(catalog.writer_plan_identity), "%064x", 10);
    snprintf(catalog.artifact_identity, sizeof(catalog.artifact_identity), "%s", identity.sha256);
    snprintf(catalog.official_reader_revision, sizeof(catalog.official_reader_revision), "%s",
             YVEX_GGUF_OFFICIAL_READER_REVISION);
    catalog.tokenizer_complete = 1;
    catalog.native_reader_accepted = 1;
    catalog.official_reader_accepted = 1;
    catalog.payload_integrity_accepted = 1;
    catalog.materialization_input_ready = 1;
    contract.catalog = &catalog;
    rc = yvex_artifact_admit_catalog(
        fixture->artifact, NULL, NULL, &contract, &fixture->admission, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admission_identity_verify(
            fixture->artifact, &fixture->admission, NULL, NULL, &failure, err);
    return rc;
}

static void tiny_compatibility(tiny_fixture *fixture)
{
    yvex_artifact_physical_compatibility *proof = &fixture->compatibility;
    const yvex_complete_artifact_admission *admission = &fixture->admission;
    memset(proof, 0, sizeof(*proof));
    proof->schema_version = YVEX_ARTIFACT_PHYSICAL_COMPATIBILITY_SCHEMA_VERSION;
    proof->source_snapshot_identity = admission->source_snapshot_identity;
    proof->mapping_identity = admission->mapping_identity;
    proof->tensor_count = admission->tensor_count;
    proof->tensors_compared = admission->tensor_count;
    proof->payload_bytes = admission->payload_bytes;
    snprintf(proof->writer_plan_identity, sizeof(proof->writer_plan_identity), "%064x", 11);
    snprintf(proof->admitted_writer_plan_identity, sizeof(proof->admitted_writer_plan_identity),
             "%s", admission->writer_plan_identity);
    snprintf(proof->artifact_identity, sizeof(proof->artifact_identity), "%s",
             admission->artifact_identity);
    snprintf(proof->payload_identity, sizeof(proof->payload_identity), "%s",
             admission->payload_identity);
    snprintf(proof->writer_transform_identity, sizeof(proof->writer_transform_identity), "%s",
             tiny_transform_identity);
    snprintf(proof->admitted_transform_identity, sizeof(proof->admitted_transform_identity), "%s",
             admission->transform_identity);
    snprintf(proof->writer_profile_identity, sizeof(proof->writer_profile_identity), "%s",
             admission->profile_identity);
    snprintf(proof->admitted_profile_identity, sizeof(proof->admitted_profile_identity), "%s",
             admission->profile_identity);
    snprintf(proof->quant_execution_identity, sizeof(proof->quant_execution_identity), "%s",
             admission->quant_execution_identity);
    snprintf(proof->payload_plan_identity, sizeof(proof->payload_plan_identity), "%s",
             admission->payload_plan_identity);
    snprintf(proof->payload_byte_identity, sizeof(proof->payload_byte_identity), "%s",
             admission->payload_byte_identity);
    proof->physical_payload_compatible = 1;
    proof->tensor_inventory_equal = 1;
    proof->qtype_equal = 1;
    proof->layout_equal = 1;
    proof->offset_equal = 1;
    proof->payload_digest_equal = 1;
}

static int tiny_model_execution(yvex_model_execution_descriptor *model, yvex_error *err)
{
    yvex_model_execution_descriptor_request request = {
        .schema_version = YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1,
        .logical_model_identity = tiny_logical_identity,
        .source_model_identity = tiny_source_identity,
        .attention_schedule_identity = tiny_schedule_identity,
        .persistent_state_identity = tiny_state_identity,
        .maximum_context = 8ull,
        .original_context = 8ull,
        .rope_scaling = YVEX_MODEL_ROPE_SCALING_NONE,
        .rope_theta = 10000ull,
        .rope_scaling_factor = 1ull,
        .layer_count = 1ull,
        .hidden_width = 4ull,
        .vocabulary_size = 261ull,
        .attention_heads = 1ull,
        .kv_heads = 1ull,
        .head_width = 4ull,
        .swa_layers = 1ull,
        .sliding_window = 4ull,
        .minimum_compression_ratio = 1ull,
        .maximum_compression_ratio = 1ull,
        .residual_streams = 1ull,
        .mhc_sinkhorn_iterations = 1ull,
        .mhc_epsilon = 1e-6,
        .normalization_epsilon = 1e-6,
        .routed_experts = 2ull,
        .experts_per_row = 1ull,
        .shared_experts = 1ull,
        .routed_ffn_width = 1ull,
        .shared_ffn_width = 1ull,
        .hash_router_layer_count = 1ull,
        .routed_scaling_factor = 1.0,
        .activation_limit = 1.0,
        .output_input_width = 4ull,
        .output_vocabulary_size = 261ull,
        .persistent_state_class_mask =
            YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_SWA_RING),
        .eos_token_id = 257ull,
    };
    return yvex_model_execution_descriptor_seal(&request, model, err);
}

static int tiny_materialize(tiny_fixture *fixture, yvex_error *err)
{
    yvex_materialization_projection projection = {
        YVEX_MATERIALIZATION_PROJECTION_SCHEMA_VERSION, TINY_MAPPING_ID,
        sizeof(tiny_terminals) / sizeof(tiny_terminals[0]), tiny_terminals,
        tiny_terminal_find, 1};
    yvex_materialization_options options;
    yvex_materialization_failure failure = {0};
    int rc;
    yvex_materialization_options_default(&options);
    options.max_chunk_bytes = 4096u;
    rc = yvex_materialization_plan_build(
        &fixture->materialization_plan, &fixture->admission, fixture->artifact,
        fixture->gguf, fixture->tensors, &projection, &options, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &fixture->materialization, fixture->materialization_plan,
            fixture->artifact, &options, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(fixture->materialization, &failure, err);
    return rc;
}

static int tiny_descriptor(tiny_fixture *fixture,
                           const yvex_model_execution_descriptor *model,
                           yvex_error *err)
{
    yvex_materialization_projection projection = {
        YVEX_MATERIALIZATION_PROJECTION_SCHEMA_VERSION, TINY_MAPPING_ID,
        sizeof(tiny_terminals) / sizeof(tiny_terminals[0]), tiny_terminals,
        tiny_terminal_find, 1};
    yvex_runtime_descriptor_family_facts family = {
        .logical_model_identity = model->logical_model_identity,
        .runtime_numeric_identity = tiny_numeric_identity,
        .runtime_hadamard_revision = "tiny-deterministic-v1",
        .runtime_numeric_schema_version = 1u,
        .runtime_compute_policy_count = 1ull,
        .layer_count = 1ull,
        .routed_experts = 2ull,
        .experts_per_token = 1ull,
        .vocabulary_size = 261ull,
        .model_execution = model,
    };
    yvex_runtime_descriptor_failure failure = {0};
    return yvex_runtime_descriptor_build_projected(
        &fixture->descriptor, &fixture->admission, fixture->materialization,
        &family, &projection, &failure, err);
}

static int tiny_semantic(tiny_fixture *fixture,
                         const yvex_model_execution_descriptor *model,
                         yvex_error *err)
{
    yvex_semantic_numeric_contract numeric = {
        .schema_version = YVEX_SEMANTIC_NUMERIC_CONTRACT_SCHEMA_V1,
        .numeric_schema_version = 1u,
        .compute_policy_count = 1ull,
    };
    yvex_semantic_model_ir_request request = {
        .schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1,
        .family_adapter_id = TINY_ADAPTER_ID,
        .family_adapter_version = TINY_ADAPTER_VERSION,
        .target_id = "tiny-executable",
        .source_model_identity = model->source_model_identity,
        .logical_model_identity = model->logical_model_identity,
        .semantic_payload_identity = model->identity,
        .execution_descriptor = model,
        .numeric_contract = &numeric,
        .attention_context = model,
        .attention_layer = tiny_attention_layer,
        .attention_layer_count = 1ull,
    };
    snprintf(numeric.identity, sizeof(numeric.identity), "%s", tiny_numeric_identity);
    snprintf(numeric.algorithm_revision, sizeof(numeric.algorithm_revision),
             "tiny-deterministic-v1");
    return yvex_semantic_model_ir_seal(&fixture->semantic, &request, err);
}

static int tiny_execution(tiny_fixture *fixture, yvex_error *err)
{
    static const yvex_physical_execution_policy physical_policy = {
        .schema_version = YVEX_PHYSICAL_EXECUTION_POLICY_SCHEMA_V1,
        .activation = YVEX_EXECUTION_ACTIVATION_HOST_F32,
        .required_backend = YVEX_EXECUTION_BACKEND_CPU,
        .evidence = YVEX_EXECUTION_EVIDENCE_PRODUCTION,
        .fallback = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE,
        .dense_kernel_family = "tiny-cpu-f32",
        .expert_kernel_family = "tiny-cpu-f32",
    };
    yvex_attention_failure attention_failure = {0};
    yvex_compiled_model_plan_request compiled = {0};
    int rc = yvex_attention_plan_build_semantic(
        &fixture->attention, fixture->semantic, YVEX_TENSOR_SCOPE_MAIN_LAYER,
        fixture->materialization, fixture->descriptor, &attention_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_operator_graph_ir_build_transformer(
            &fixture->operators, fixture->semantic, fixture->attention, NULL, err);
    fixture->capabilities = (yvex_runtime_capabilities){
        .attention_semantics_ready = 1, .attention_core_ready = 1,
        .attention_envelope_ready = 1, .cpu_prefill_eager_ready = 1,
        .cpu_decode_eager_ready = 1, .cuda_eager_implemented = 1,
        .attention_state_delta_ready = 1, .attention_operator_ready = 1,
        .moe_plan_ready = 1, .moe_router_ready = 1,
        .moe_routed_expert_ready = 1, .moe_shared_expert_ready = 1,
        .moe_block_ready = 1, .transformer_ready = 1,
        .output_head_binding_ready = 1, .output_head_projection_ready = 1,
        .logits_cpu_ready = 1, .logits_cuda_ready = 1,
        .logits_prefill_ready = 1, .logits_decode_ready = 1,
        .logits_full_vocabulary_ready = 1, .logits_hidden_contract_ready = 1,
        .logits_partial_progress_ready = 1, .logits_ready = 1,
    };
    fixture->transformer = (yvex_transformer_family_policy){
        .schema_version = YVEX_TRANSFORMER_PLAN_SCHEMA_V2,
        .initial_policy = YVEX_TRANSFORMER_INITIAL_REPEAT_STREAMS,
        .final_policy = YVEX_TRANSFORMER_FINAL_SIGMOID_MHC_RMS,
        .residual_streams = 1ull, .hidden_width = 4ull, .expanded_width = 4ull,
        .maximum_context = 8ull, .sinkhorn_iterations = 1ull,
        .mhc_epsilon = 1e-6, .output_norm_epsilon = 1e-6,
        .attention_then_moe = 1, .deferred_ffn_post = 1,
        .final_norm_after_head = 1,
    };
    fixture->logits = (yvex_logits_family_policy){
        .schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V3,
        .separate_output_head = 1,
    };
    fixture->speculation = (yvex_speculation_family_policy){
        .schema_version = YVEX_SPECULATION_FAMILY_POLICY_SCHEMA_V1,
        .feature_projection_role = YVEX_TENSOR_ROLE_DRAFT_FEATURE_PROJECTION,
        .feature_norm_role = YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM,
        .output_norm_role = YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM,
        .markov_embedding_role = YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING,
        .markov_output_role = YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT,
        .confidence_role = YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE,
        .shares_embedding = 1, .shares_output_head = 1,
    };
    snprintf(fixture->speculation.policy_identity,
             sizeof(fixture->speculation.policy_identity), "%064x", 12);
    if (rc == YVEX_OK && !yvex_runtime_capabilities_contract_valid(&fixture->capabilities))
        rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK)
        rc = yvex_physical_execution_ir_build(
            &fixture->physical, fixture->materialization, fixture->descriptor,
            fixture->admission.profile_identity, &physical_policy, err);
    compiled = (yvex_compiled_model_plan_request){
        .operator_graph = fixture->operators,
        .materialization = fixture->materialization,
        .descriptor = fixture->descriptor,
        .attention = fixture->attention,
        .graph = &tiny_graph_api,
        .family_adapter_id = TINY_ADAPTER_ID,
        .family_adapter_version = TINY_ADAPTER_VERSION,
        .capabilities = fixture->capabilities,
        .transformer_policy = fixture->transformer,
        .logits_policy = fixture->logits,
    };
    if (rc == YVEX_OK)
        rc = yvex_compiled_model_plan_build(&fixture->compiled, &compiled, err);
    return rc;
}

static int tiny_tokenizer(tiny_fixture *fixture, yvex_error *err)
{
    static const char json_identity[] =
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a";
    yvex_conversation_protocol source = {
        .schema_version = YVEX_CONVERSATION_PROTOCOL_SCHEMA_V1,
        .family_adapter_id = TINY_ADAPTER_ID,
        .family_adapter_version = TINY_ADAPTER_VERSION,
        .architecture = "tiny-executable",
        .source_revision = "tiny-v1",
        .source_encoding_path = "tests/integration/tiny_model.py",
        .source_encoding_identity = tiny_source_identity,
        .bos = "", .eos = "<eos>", .user = "", .assistant = "",
        .latest_reminder = "", .thinking_start = "<think>",
        .thinking_end = "</think>", .tool_result_start = "<tool-result>",
        .tool_result_end = "</tool-result>", .dsml = "<dsml>",
        .tool_calls_start = "<tool-calls>", .tool_calls_end = "</tool-calls>",
        .tool_invoke_start = "<tool>", .tool_invoke_name_end = "</name>",
        .tool_invoke_end = "</tool>", .tool_parameter_start = "<parameter>",
        .tool_parameter_name_end = "</parameter-name>",
        .tool_parameter_kind_end = "</parameter-kind>",
        .tool_parameter_end = "</parameter>", .reasoning_effort_max = "max",
        .tools_prefix = "<tools>", .tools_suffix = "</tools>",
        .response_format_prefix = "<format>",
        .tokenizer_model = "yvex-fixture-simple", .tokenizer_pre = "tiny",
        .tokenizer_json_identity = json_identity,
        .tokenizer_config_identity = json_identity,
        .vocabulary_size = 261ull, .base_vocabulary_size = 258ull,
        .merge_count = 1ull, .added_token_count = 3ull, .special_token_count = 3ull,
        .eos_token_id = 257u, .unk_token_id = 260u,
        .eos_present = 1, .unk_present = 1,
    };
    return yvex_tokenizer_family_policy_compile(
        &fixture->tokenizer, &source, YVEX_TOKENIZER_KIND_FIXTURE_SIMPLE,
        YVEX_TOKENIZER_MODEL_BPE_BYTELEVEL, YVEX_TOKENIZER_PROMPT_CONVERSATION, err);
}

static int tiny_binding(tiny_fixture *fixture, const char *directory,
                        yvex_runtime_binding_prepare_result *result,
                        yvex_error *err)
{
    yvex_runtime_binding_prepare_request request = {
        .directory = directory,
        .admission = &fixture->admission,
        .physical_compatibility = &fixture->compatibility,
        .materialization = fixture->materialization,
        .runtime_descriptor = fixture->descriptor,
        .operator_graph = fixture->operators,
        .physical_execution = fixture->physical,
        .compiled_plan = fixture->compiled,
        .attention_plan = fixture->attention,
        .family_adapter_id = TINY_ADAPTER_ID,
        .family_adapter_version = TINY_ADAPTER_VERSION,
        .artifact_format = "GGUF",
        .artifact_format_version = 3u,
        .logical_transform_identity = tiny_transform_identity,
        .capabilities = fixture->capabilities,
        .transformer_policy = fixture->transformer,
        .logits_policy = fixture->logits,
        .speculation_policy = fixture->speculation,
        .tokenizer_policy = fixture->tokenizer,
    };
    yvex_runtime_binding_failure failure = {0};
    return yvex_runtime_binding_prepare(&request, result, &failure, err);
}

static int tiny_compile(const char *artifact_path, const char *directory,
                        yvex_runtime_binding_prepare_result *result, yvex_error *err)
{
    tiny_fixture fixture = {0};
    yvex_model_execution_descriptor model = {0};
    int rc = tiny_open(&fixture, artifact_path, err);
    if (rc == YVEX_OK) rc = tiny_admit(&fixture, err);
    if (rc == YVEX_OK) tiny_compatibility(&fixture);
    if (rc == YVEX_OK) rc = tiny_materialize(&fixture, err);
    if (rc == YVEX_OK) rc = tiny_model_execution(&model, err);
    if (rc == YVEX_OK) rc = tiny_descriptor(&fixture, &model, err);
    if (rc == YVEX_OK) rc = tiny_semantic(&fixture, &model, err);
    if (rc == YVEX_OK) rc = tiny_execution(&fixture, err);
    if (rc == YVEX_OK) rc = tiny_tokenizer(&fixture, err);
    if (rc == YVEX_OK) rc = tiny_binding(&fixture, directory, result, err);
    tiny_close(&fixture);
    return rc;
}

typedef struct {
    unsigned long long events[YVEX_RUNTIME_LIFECYCLE_COUNT];
} tiny_capacity_progress;

static int tiny_capacity_progress_collect(
    void *opaque, yvex_runtime_lifecycle_phase phase,
    unsigned long long completed, unsigned long long total)
{
    tiny_capacity_progress *progress = (tiny_capacity_progress *)opaque;
    (void)completed;
    (void)total;
    if (!progress || (unsigned int)phase >= YVEX_RUNTIME_LIFECYCLE_COUNT)
        return 0;
    progress->events[phase]++;
    return 1;
}

static int tiny_generation_capacity_refusal(
    const char *artifact_path, const char *binding_path, yvex_error *err)
{
    yvex_model_engine_open_request model_request = {
        .artifact_path = artifact_path,
        .runtime_binding_path = binding_path,
        .target_id = "tiny-executable",
        .residency_backend = YVEX_BACKEND_KIND_CPU,
    };
    yvex_runtime_session_open_request session_request = {
        .backend = YVEX_BACKEND_KIND_CPU,
    };
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5,
        .backend = YVEX_BACKEND_KIND_CPU,
        .mode = YVEX_GENERATION_MODE_TARGET_ONLY,
        .context_capacity = 8ull,
        .prefill_chunk_tokens = 1ull,
        .maximum_new_tokens = 1ull,
        .maximum_output_bytes = 64ull,
        .evidence_profile = YVEX_EXECUTION_EVIDENCE_PRODUCTION,
        .sampling_policy = {
            .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
            .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
            .temperature = 1.0,
            .top_p = 1.0,
            .typical_p = 1.0,
        },
    };
    yvex_model_engine_failure failure = {0};
    yvex_runtime_binding_failure binding_failure = {0};
    yvex_runtime_binding_summary binding_summary = {0};
    yvex_complete_artifact_admission admission = {0};
    yvex_runtime_binding *binding = NULL;
    yvex_model_engine *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *generation = NULL;
    tiny_capacity_progress progress = {0};
    yvex_error cleanup;
    char available_text[32];
    unsigned long long transient, baseline;
    int injected_system = 0, injected_process = 0;
    int rc = yvex_runtime_binding_open(
        &binding, binding_path, &binding_summary, &admission,
        &binding_failure, err);
    if (rc == YVEX_OK &&
        (!runtime_binding_maximum_tensor_bytes(
             binding, &transient) ||
         !yvex_core_u64_add(
             admission.payload_bytes,
             yvex_runtime_private_system_reserve(128ull * 1024ull * 1024ull * 1024ull),
             &baseline) ||
         !yvex_core_u64_add(baseline, transient, &baseline) ||
         snprintf(available_text, sizeof(available_text), "%llu", baseline) <= 0 ||
         setenv("YVEX_TEST_RUNTIME_TOTAL_MEMORY_BYTES", "137438953472", 1) != 0 ||
         setenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES", available_text, 1) != 0)) {
        rc = YVEX_ERR_IO;
        yvex_error_set(err, rc, "tiny.capacity",
                       "startup capacity injection failed");
    }
    model_request.startup_generation = &options;
    model_request.progress = tiny_capacity_progress_collect;
    model_request.progress_context = &progress;
    if (rc == YVEX_OK)
        rc = yvex_model_engine_open(&model, &model_request, &failure, err);
    (void)unsetenv("YVEX_TEST_RUNTIME_TOTAL_MEMORY_BYTES");
    (void)unsetenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES");
    if (rc == YVEX_ERR_BOUNDS && !model &&
        failure.code == YVEX_MODEL_ENGINE_FAILURE_ALLOCATION &&
        strcmp(failure.field, "startup-execution-capacity") == 0 &&
        failure.expected > baseline && failure.actual == baseline &&
        progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN] == 0ull) {
        rc = YVEX_OK;
        yvex_error_clear(err);
    } else if (rc == YVEX_OK) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, rc, "tiny.capacity",
                       "startup capacity reached artifact mutation");
    }
    yvex_runtime_binding_close(binding);
    if (rc == YVEX_OK) {
        memset(&progress, 0, sizeof(progress));
        memset(&failure, 0, sizeof(failure));
        rc = yvex_model_engine_open(
            &model, &model_request, &failure, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_open(
            &session, model, &session_request, &failure, err);
    if (rc == YVEX_OK) {
        injected_system = setenv(
            "YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES",
            "18446744073709551615", 1) == 0;
        injected_process = setenv(
            "YVEX_TEST_RUNTIME_CGROUP_AVAILABLE_MEMORY_BYTES", "1", 1) == 0;
        if (!injected_system || !injected_process) {
            rc = YVEX_ERR_IO;
            yvex_error_set(err, rc, "tiny.capacity",
                           "process memory injection failed");
        }
    }
    if (rc == YVEX_OK) {
        rc = yvex_runtime_generation_context_open(
            &generation, model, session, &options, err);
        if (rc == YVEX_ERR_BOUNDS && !generation &&
            strcmp(yvex_error_where(err), "runtime.generation") == 0 &&
            strcmp(yvex_error_message(err),
                   "live process memory cannot preserve the admitted runtime reserve") == 0) {
            rc = YVEX_OK;
            yvex_error_clear(err);
        } else if (rc == YVEX_OK) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(err, rc, "tiny.capacity",
                           "generation ignored live process memory capacity");
        }
    }
    if (injected_system)
        (void)unsetenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES");
    if (injected_process)
        (void)unsetenv("YVEX_TEST_RUNTIME_CGROUP_AVAILABLE_MEMORY_BYTES");
    if (generation)
        (void)yvex_runtime_generation_context_close(&generation, &cleanup);
    if (session)
        (void)yvex_runtime_session_close(&session, &cleanup);
    yvex_model_engine_close(&model);
    return rc;
}

int main(int argc, char **argv)
{
    yvex_runtime_binding_prepare_result result = {0};
    yvex_error err;
    int rc;
    if (argc != 3) {
        fprintf(stderr, "usage: %s ARTIFACT OUTPUT_DIRECTORY\n", argv[0]);
        return 2;
    }
    yvex_error_clear(&err);
    rc = tiny_compile(argv[1], argv[2], &result, &err);
    if (rc == YVEX_OK)
        rc = tiny_generation_capacity_refusal(argv[1], result.path, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "tiny compile failed: %s: %s\n",
                yvex_error_where(&err), yvex_error_message(&err));
        return 1;
    }
    printf("artifact_identity=%s\n", result.summary.artifact_identity);
    printf("binding_identity=%s\n", result.summary.identity);
    printf("binding_path=%s\n", result.path);
    return 0;
}
