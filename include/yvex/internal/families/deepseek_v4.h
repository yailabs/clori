/* Private DeepSeek architecture, lowering, and graph-recipe contract. */
#ifndef INCLUDE_YVEX_INTERNAL_FAMILIES_DEEPSEEK_V4_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_FAMILIES_DEEPSEEK_V4_H_INCLUDED
#include <stddef.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/model.h>
#include <yvex/quant.h>
#include <yvex/source.h>
#include <yvex/internal/model.h>
#define YVEX_DEEPSEEK_IDENTITY_CAP 65u
#define YVEX_DEEPSEEK_V4_ADAPTER_ID 0x44535634ull
#define YVEX_DEEPSEEK_V4_ADAPTER_VERSION 7ull
typedef struct yvex_source_verification yvex_source_verification;
typedef struct yvex_source_tensor_snapshot yvex_source_tensor_snapshot;
typedef struct yvex_source_payload_plan yvex_source_payload_plan;
typedef struct yvex_source_payload_session yvex_source_payload_session;
typedef struct yvex_transform_ir yvex_transform_ir;
typedef struct yvex_transform_binding yvex_transform_binding;
typedef struct yvex_transform_builder_options yvex_transform_builder_options;
typedef struct yvex_transform_failure yvex_transform_failure;
typedef struct yvex_artifact yvex_artifact;
typedef struct yvex_artifact_admission_failure yvex_artifact_admission_failure;
typedef struct yvex_complete_artifact_admission yvex_complete_artifact_admission;
typedef struct yvex_materialization_failure yvex_materialization_failure;
typedef struct yvex_materialization_options yvex_materialization_options;
typedef struct yvex_materialization_projection yvex_materialization_projection;
typedef struct yvex_materialization_plan yvex_materialization_plan;
typedef struct yvex_materialization_session yvex_materialization_session;
typedef struct yvex_quant_failure yvex_quant_failure;
typedef struct yvex_quant_plan yvex_quant_plan;
typedef struct yvex_quant_plan_options yvex_quant_plan_options;
typedef struct yvex_runtime_descriptor yvex_runtime_descriptor;
typedef struct yvex_runtime_descriptor_failure yvex_runtime_descriptor_failure;
typedef struct yvex_model_target_request yvex_model_target_request;
typedef struct yvex_model_target_report yvex_model_target_report;
typedef struct yvex_graph_execution_api yvex_graph_execution_api;
typedef struct yvex_graph_compiler_api yvex_graph_compiler_api;
typedef struct yvex_family_compiler_adapter yvex_family_compiler_adapter;
#define YVEX_DEEPSEEK_V4_IR_NO_LAYER (~0ull)
typedef enum {
    YVEX_DEEPSEEK_V4_IR_FAILURE_NONE = 0, YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_ARGUMENT,
    YVEX_DEEPSEEK_V4_IR_FAILURE_SOURCE_NOT_VERIFIED, YVEX_DEEPSEEK_V4_IR_FAILURE_IDENTITY_MISMATCH,
    YVEX_DEEPSEEK_V4_IR_FAILURE_SOURCE_FACT_MISSING, YVEX_DEEPSEEK_V4_IR_FAILURE_SCHEDULE_LENGTH,
    YVEX_DEEPSEEK_V4_IR_FAILURE_SCHEDULE_PATTERN, YVEX_DEEPSEEK_V4_IR_FAILURE_UNSUPPORTED_COMPRESSION,
    YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_DIMENSION, YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_GROUP_GEOMETRY,
    YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_POSITION, YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_MHC,
    YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_DSPARK,
    YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_ROUTING, YVEX_DEEPSEEK_V4_IR_FAILURE_INVALID_EXPERT_TOPK,
    YVEX_DEEPSEEK_V4_IR_FAILURE_TOKENIZER_OUTPUT_MISMATCH,
    YVEX_DEEPSEEK_V4_IR_FAILURE_UNSUPPORTED_SOURCE_CONSTRAINT,
    YVEX_DEEPSEEK_V4_IR_FAILURE_UNSUPPORTED_RUNTIME_NUMERIC, YVEX_DEEPSEEK_V4_IR_FAILURE_NUMERIC_VALUE,
    YVEX_DEEPSEEK_V4_IR_FAILURE_ARITHMETIC_OVERFLOW, YVEX_DEEPSEEK_V4_IR_FAILURE_ALLOCATION
} yvex_deepseek_v4_ir_failure_code;
typedef enum {
    YVEX_DEEPSEEK_V4_IR_COMPONENT_NONE = 0, YVEX_DEEPSEEK_V4_IR_COMPONENT_SOURCE,
    YVEX_DEEPSEEK_V4_IR_COMPONENT_IDENTITY, YVEX_DEEPSEEK_V4_IR_COMPONENT_MODEL,
    YVEX_DEEPSEEK_V4_IR_COMPONENT_ATTENTION, YVEX_DEEPSEEK_V4_IR_COMPONENT_POSITION,
    YVEX_DEEPSEEK_V4_IR_COMPONENT_MHC, YVEX_DEEPSEEK_V4_IR_COMPONENT_MOE,
    YVEX_DEEPSEEK_V4_IR_COMPONENT_OUTPUT, YVEX_DEEPSEEK_V4_IR_COMPONENT_TOKENIZER,
    YVEX_DEEPSEEK_V4_IR_COMPONENT_DSPARK, YVEX_DEEPSEEK_V4_IR_COMPONENT_SOURCE_CONSTRAINT,
    YVEX_DEEPSEEK_V4_IR_COMPONENT_RUNTIME_NUMERIC, YVEX_DEEPSEEK_V4_IR_COMPONENT_ALLOCATION
} yvex_deepseek_v4_ir_component;
typedef enum {
    YVEX_DEEPSEEK_V4_KV_SWA = 0, YVEX_DEEPSEEK_V4_KV_CSA, YVEX_DEEPSEEK_V4_KV_HCA
} yvex_deepseek_v4_kv_class;
typedef enum {
    YVEX_DEEPSEEK_V4_ROUTER_HASH_TOKEN_ID = 0, YVEX_DEEPSEEK_V4_ROUTER_LEARNED_HIDDEN_STATE
} yvex_deepseek_v4_router_class;
typedef enum {
    YVEX_DEEPSEEK_V4_MHC_STANDALONE_PRE = 0, YVEX_DEEPSEEK_V4_MHC_FUSED_PRIOR_POST_PRE
} yvex_deepseek_v4_mhc_entry;
typedef enum { YVEX_DEEPSEEK_V4_SCORING_SQRT_SOFTPLUS = 0 } yvex_deepseek_v4_scoring_policy;
typedef enum { YVEX_DEEPSEEK_V4_TOPK_NOAUX_TC = 0 } yvex_deepseek_v4_topk_policy;
typedef enum { YVEX_DEEPSEEK_V4_ACTIVATION_SILU = 0 } yvex_deepseek_v4_activation;
typedef enum { YVEX_DEEPSEEK_V4_SOURCE_WEIGHT_BF16 = 0 } yvex_deepseek_v4_source_weight_dtype;
typedef enum { YVEX_DEEPSEEK_V4_SOURCE_EXPERT_FP4 = 0 } yvex_deepseek_v4_source_expert_dtype;
typedef enum { YVEX_DEEPSEEK_V4_SOURCE_QUANT_FP8_E4M3_UE8M0_DYNAMIC = 0 } yvex_deepseek_v4_source_quantization;
typedef struct {
    yvex_deepseek_v4_ir_failure_code code;
    yvex_deepseek_v4_ir_component component;
    const char *field;
    unsigned long long layer_index, expected, actual;
} yvex_deepseek_v4_ir_failure;
typedef struct {
    yvex_deepseek_v4_kv_class class_id;
    unsigned long long compression_ratio, sliding_window;
    int requires_state_cache, requires_uncompressed_tail, requires_compressed_core;
    int requires_indexer_cache;
} yvex_deepseek_v4_kv_spec;
typedef struct {
    unsigned long long residual_streams, stream_width, expanded_width, mixing_rows, mixing_columns;
    unsigned long long base_width, scale_width, sinkhorn_iterations;
    double epsilon, residual_post_multiplier;
    yvex_deepseek_v4_mhc_entry entry;
    int attention_pre_and_post, ffn_pre_and_deferred_post;
} yvex_deepseek_v4_mhc_spec;
typedef struct {
    int required;
    unsigned long long width;
} yvex_deepseek_v4_norm_spec;
typedef struct {
    int required;
    unsigned long long function_rows, function_columns, base_width, scale_width;
} yvex_deepseek_v4_mhc_head_spec;
typedef struct {
    unsigned long long q_a_rows, q_a_columns, q_b_rows, q_b_columns, kv_rows, kv_columns, o_a_rows;
    unsigned long long o_a_columns, o_b_rows, o_b_columns, compressor_ape_rows;
    unsigned long long compressor_ape_columns, compressor_norm_width, compressor_projection_rows;
    unsigned long long compressor_projection_columns, indexer_ape_rows, indexer_ape_columns;
    unsigned long long indexer_norm_width, indexer_projection_rows, indexer_projection_columns;
    unsigned long long indexer_query_rows, indexer_query_columns, indexer_weight_rows;
    unsigned long long indexer_weight_columns;
} yvex_deepseek_v4_attention_tensor_spec;
typedef struct {
    yvex_deepseek_v4_router_class router_class;
    yvex_deepseek_v4_scoring_policy scoring;
    yvex_deepseek_v4_topk_policy topk_policy;
    yvex_deepseek_v4_activation activation;
    unsigned long long routed_experts, shared_experts, experts_per_token, expert_intermediate_size;
    unsigned long long shared_intermediate_size, hash_table_rows, hash_table_columns;
    unsigned long long correction_bias_width;
    double routed_scaling_factor, activation_limit;
    int requires_token_ids, requires_hidden_state, requires_correction_bias;
    int normalize_topk_probabilities;
} yvex_deepseek_v4_moe_spec;
typedef struct yvex_deepseek_v4_layer_spec {
    unsigned long long layer_index;
    yvex_attention_class attention_class;
    yvex_attention_compute_contract compute_contract;
    unsigned long long compression_ratio, query_heads, kv_heads, head_dimension;
    unsigned long long rope_head_dimension, non_rope_head_dimension, query_lora_rank;
    unsigned long long output_lora_rank, output_groups, output_heads_per_group;
    unsigned long long output_group_input_width, indexer_heads, indexer_head_dimension;
    unsigned long long indexer_topk, attention_sink_count;
    double attention_dropout;
    int causal, attention_bias, query_norm_required, kv_norm_required, compressor_required;
    int indexer_required;
    yvex_attention_position_policy position;
    yvex_deepseek_v4_kv_spec kv;
    yvex_deepseek_v4_mhc_spec mhc;
    yvex_deepseek_v4_moe_spec moe;
    yvex_attention_activation_policy attention_kv_activation, compressor_activation;
    yvex_attention_activation_policy compressor_rotated_activation;
    yvex_attention_activation_policy indexer_query_activation;
    yvex_attention_topk_policy sparse_topk;
    yvex_deepseek_v4_norm_spec attention_input_norm, post_attention_ffn_norm;
    yvex_deepseek_v4_attention_tensor_spec tensors;
    double rms_norm_epsilon;
} yvex_deepseek_v4_layer_spec;
typedef struct {
    yvex_deepseek_v4_layer_spec layer;
    unsigned long long predictor_index;
    int has_feature_projection, has_feature_norm, has_output_norm;
    unsigned long long feature_projection_input, feature_projection_output;
    unsigned long long feature_norm_width, output_norm_width;
    int has_markov_head;
    unsigned long long markov_rank, markov_vocabulary_size;
    int has_confidence_head;
    unsigned long long confidence_input_width, confidence_output_width;
    int has_separate_mhc_head;
    yvex_deepseek_v4_mhc_head_spec mhc_head;
    int shares_embedding, shares_output_head;
} yvex_deepseek_v4_auxiliary_spec;
typedef struct {
    int present;
    unsigned int schema_version;
    unsigned long long block_size, noise_token_id;
    unsigned long long target_layer_ids[3], target_layer_count;
    unsigned long long target_feature_width, concatenated_feature_width;
    unsigned long long draft_layer_count, markov_rank, final_draft_layer;
    int parallel_block_backbone, sequential_markov, confidence_available;
    int shares_embedding, shares_output_head, target_verification_required;
    unsigned long long accepted_prefix_maximum;
} yvex_deepseek_v4_dspark_spec;
typedef struct {
    int required;
    unsigned long long vocabulary_size, hidden_size;
} yvex_deepseek_v4_embedding_spec;
typedef struct {
    int required, tied_to_embedding;
    unsigned long long input_width, vocabulary_size;
} yvex_deepseek_v4_output_spec;
typedef struct {
    char tokenizer_class[64];
    char model_type[32];
    unsigned long long vocabulary_size, base_vocab_entries, added_token_entries, maximum_token_id;
    unsigned long long maximum_context, bos_token_id, eos_token_id;
    int bos_required, eos_required;
} yvex_deepseek_v4_tokenizer_spec;
typedef struct {
    yvex_deepseek_v4_source_weight_dtype weight_dtype;
    yvex_deepseek_v4_source_expert_dtype expert_dtype;
    yvex_deepseek_v4_source_quantization quantization;
    unsigned long long quant_block_rows, quant_block_columns, fp4_packing_factor;
    unsigned long long fp4_scale_group_width;
    yvex_native_dtype fp4_physical_dtype, scale_dtype;
} yvex_deepseek_v4_source_constraint;
typedef struct {
    char target_id[128];
    char family[32];
    char architecture[128];
    char repository[256];
    char revision[128];
    char verification_stage[64];
    char paper_revision[32], dspark_paper_revision[32];
    char deepspec_revision[64], sglang_revision[64], vllm_revision[64];
    char hadamard_revision[128];
    unsigned int runtime_numeric_schema_version;
    unsigned long long runtime_compute_policy_count, runtime_activation_policy_count;
    unsigned long long runtime_sparse_topk_policy_count;
    unsigned long long hidden_size, vocabulary_size, maximum_context, main_layer_count;
    unsigned long long auxiliary_layer_count, swa_layer_count, csa_layer_count, hca_layer_count;
    unsigned long long source_snapshot_identity;
    unsigned long long hash_router_layer_count, learned_router_layer_count;
    unsigned long long source_header_scan_count, source_header_tensor_count;
    unsigned long long source_payload_bytes_read;
    yvex_deepseek_v4_embedding_spec embedding;
    yvex_deepseek_v4_output_spec output;
    yvex_deepseek_v4_tokenizer_spec tokenizer;
    yvex_deepseek_v4_source_constraint source_constraint;
    yvex_deepseek_v4_dspark_spec dspark;
    yvex_deepseek_v4_mhc_spec final_mhc;
    yvex_deepseek_v4_mhc_head_spec final_mhc_head;
    double final_norm_epsilon;
    int use_cache, final_mhc_post_required, final_mhc_head_required, final_norm_after_mhc_head;
} yvex_deepseek_v4_model_spec;
typedef void *(*yvex_deepseek_v4_ir_allocate_fn)(size_t size, void *context);
typedef void (*yvex_deepseek_v4_ir_release_fn)(void *allocation,
                                               void *context);
typedef struct {
    yvex_deepseek_v4_ir_allocate_fn allocate;
    yvex_deepseek_v4_ir_release_fn release;
    void *context;
} yvex_deepseek_v4_ir_allocator;
typedef struct yvex_deepseek_v4_ir yvex_deepseek_v4_ir;
typedef enum {
    YVEX_DEEPSEEK_COVERAGE_FAILURE_NONE = 0, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_ARGUMENT,
    YVEX_DEEPSEEK_COVERAGE_FAILURE_SOURCE_IDENTITY, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVENTORY_AUTHORITY,
    YVEX_DEEPSEEK_COVERAGE_FAILURE_INVENTORY_DRIFT, YVEX_DEEPSEEK_COVERAGE_FAILURE_ARCHITECTURE_INCOMPLETE,
    YVEX_DEEPSEEK_COVERAGE_FAILURE_MISSING_REQUIREMENT, YVEX_DEEPSEEK_COVERAGE_FAILURE_AMBIGUOUS_MATCH,
    YVEX_DEEPSEEK_COVERAGE_FAILURE_UNEXPECTED_SOURCE, YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_INDEX,
    YVEX_DEEPSEEK_COVERAGE_FAILURE_RANK_MISMATCH, YVEX_DEEPSEEK_COVERAGE_FAILURE_SHAPE_MISMATCH,
    YVEX_DEEPSEEK_COVERAGE_FAILURE_DTYPE_MISMATCH, YVEX_DEEPSEEK_COVERAGE_FAILURE_SCALE_COMPANION,
    YVEX_DEEPSEEK_COVERAGE_FAILURE_ARITHMETIC_OVERFLOW, YVEX_DEEPSEEK_COVERAGE_FAILURE_RESOURCE_LIMIT,
    YVEX_DEEPSEEK_COVERAGE_FAILURE_ALLOCATION
} yvex_deepseek_tensor_coverage_failure_code;
#define YVEX_DEEPSEEK_TENSOR_NO_INDEX (~0ull)
typedef struct {
    yvex_deepseek_tensor_coverage_failure_code code;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    char tensor_name[256];
    unsigned long long layer_index, expert_index, dimension_index, expected, actual;
} yvex_deepseek_tensor_coverage_failure;
typedef struct {
    const yvex_native_weight_info *source;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    unsigned long long layer_index, expert_index;
} yvex_deepseek_tensor_coverage_row;
typedef struct {
    unsigned long long source_tensor_count, required_tensor_count, matched_tensor_count;
    unsigned long long missing_count, ambiguous_count, unexpected_count;
    unsigned long long collection_counts[YVEX_TENSOR_COLLECTION_COUNT];
    unsigned long long main_layer_count, auxiliary_layer_count, routed_expert_count;
    unsigned long long shared_expert_count, header_scan_count, payload_bytes_read;
    unsigned long long source_lookup_count, source_collision_count, source_maximum_probe;
    unsigned long long source_identity, coverage_identity;
    int complete;
} yvex_deepseek_tensor_coverage_summary;
typedef void *(*yvex_deepseek_tensor_coverage_allocate_fn)(size_t size,
                                                            void *context);
typedef void (*yvex_deepseek_tensor_coverage_release_fn)(void *allocation,
                                                         void *context);
typedef struct {
    yvex_deepseek_tensor_coverage_allocate_fn allocate;
    yvex_deepseek_tensor_coverage_release_fn release;
    void *context;
    unsigned long long maximum_tensors;
} yvex_deepseek_tensor_coverage_options;
typedef struct yvex_deepseek_tensor_coverage yvex_deepseek_tensor_coverage;
typedef enum {
    YVEX_DEEPSEEK_RECIPE_DIRECT = 0, YVEX_DEEPSEEK_RECIPE_FP8_PAIR,
    YVEX_DEEPSEEK_RECIPE_CHECKED_CAST
} yvex_deepseek_tensor_recipe_kind;
typedef enum {
    YVEX_DEEPSEEK_RECIPE_ALWAYS = 0, YVEX_DEEPSEEK_RECIPE_COMPRESSOR,
    YVEX_DEEPSEEK_RECIPE_INDEXER, YVEX_DEEPSEEK_RECIPE_HASH_ROUTER,
    YVEX_DEEPSEEK_RECIPE_LEARNED_ROUTER
} yvex_deepseek_tensor_recipe_condition;
typedef struct { size_t offset; int model_field; } yvex_deepseek_tensor_dimension_ref;
typedef struct {
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_deepseek_tensor_recipe_kind kind;
    yvex_deepseek_tensor_recipe_condition condition;
    unsigned int phase;
    const char *suffix;
    yvex_native_dtype dtype; unsigned int rank;
    yvex_deepseek_tensor_dimension_ref dimensions[2];
} yvex_deepseek_tensor_recipe;
#define YVEX_DEEPSEEK_TRANSFORM_SOURCE_COUNT 72317ull
#define YVEX_DEEPSEEK_TRANSFORM_TERMINAL_COUNT 1409ull
#define YVEX_DEEPSEEK_TRANSFORM_MAIN_TERMINAL_COUNT 1328ull
#define YVEX_DEEPSEEK_TRANSFORM_AUX_TERMINAL_COUNT 81ull
#define YVEX_DEEPSEEK_GGUF_NO_INDEX (~0ull)
#define YVEX_DEEPSEEK_GGUF_AGGREGATED_AXIS (~0u)
#define YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT 1409ull
#define YVEX_DEEPSEEK_GGUF_TRUNK_DESCRIPTOR_COUNT 1328ull
#define YVEX_DEEPSEEK_GGUF_DRAFT_DESCRIPTOR_COUNT 81ull
#define YVEX_DEEPSEEK_GGUF_SOURCE_COUNT 72317ull
#define YVEX_DEEPSEEK_GGUF_MAPPING_IDENTITY 0x779aa44d104fc718ull
typedef enum {
    YVEX_DEEPSEEK_GGUF_TRANSFORM_DIRECT = 0, YVEX_DEEPSEEK_GGUF_TRANSFORM_FP8_E4M3_E8M0,
    YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4, YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32
} yvex_deepseek_gguf_transform;
typedef enum {
    YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY = 0, YVEX_DEEPSEEK_GGUF_CONTRIBUTION_SCALE,
    YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_WEIGHT, YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_SCALE,
    YVEX_DEEPSEEK_GGUF_CONTRIBUTION_ROUTING_TABLE
} yvex_deepseek_gguf_contribution_kind;
typedef enum {
    YVEX_DEEPSEEK_GGUF_METADATA_STRING = 0, YVEX_DEEPSEEK_GGUF_METADATA_U64,
    YVEX_DEEPSEEK_GGUF_METADATA_F64, YVEX_DEEPSEEK_GGUF_METADATA_BOOL,
    YVEX_DEEPSEEK_GGUF_METADATA_U64_ARRAY, YVEX_DEEPSEEK_GGUF_METADATA_F64_ARRAY
} yvex_deepseek_gguf_metadata_type;
typedef enum {
    YVEX_DEEPSEEK_GGUF_MAP_FAILURE_NONE = 0, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_INVALID_ARGUMENT,
    YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ARCHITECTURE, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_COVERAGE_ROW,
    YVEX_DEEPSEEK_GGUF_MAP_FAILURE_MISSING_SOURCE, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_DUPLICATE_SOURCE,
    YVEX_DEEPSEEK_GGUF_MAP_FAILURE_SOURCE_DTYPE, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_EXPERT_SEQUENCE,
    YVEX_DEEPSEEK_GGUF_MAP_FAILURE_NAME, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_DUPLICATE_NAME,
    YVEX_DEEPSEEK_GGUF_MAP_FAILURE_LAYOUT, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_METADATA,
    YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ACCOUNTING, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ARITHMETIC_OVERFLOW,
    YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ALLOCATION, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_TRANSFORM_IR,
    YVEX_DEEPSEEK_GGUF_MAP_FAILURE_LOWERING_DIVERGENCE, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_MAPPING_IDENTITY
} yvex_deepseek_gguf_map_failure_code;
typedef struct {
    yvex_deepseek_gguf_map_failure_code code;
    yvex_tensor_role role;
    yvex_tensor_scope scope;
    unsigned long long layer_index, predictor_index, expert_index, expected, actual;
    char source_name[256];
    char emitted_name[192];
} yvex_deepseek_gguf_map_failure;
typedef struct {
    char source_name[256];
    yvex_native_dtype source_dtype;
    unsigned int source_rank;
    unsigned long long source_dims[2];
    yvex_deepseek_gguf_contribution_kind kind;
    unsigned long long source_row_index, descriptor_index, expert_index;
} yvex_deepseek_gguf_contribution;
typedef struct {
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    unsigned long long layer_index, predictor_index, expert_count;
    char emitted_name[192];
    yvex_deepseek_gguf_transform transform;
    yvex_gguf_name_provenance name_provenance;
    unsigned int forced_qtype, logical_rank;
    unsigned long long logical_dims[YVEX_TENSOR_MAX_DIMS];
    unsigned int source_axis_for_logical[YVEX_TENSOR_MAX_DIMS];
    unsigned long long contribution_offset, contribution_count, identity;
} yvex_deepseek_gguf_descriptor;
typedef struct {
    char key[128];
    yvex_deepseek_gguf_metadata_type type;
    char string_value[192];
    unsigned long long u64_value;
    double f64_value;
    int bool_value;
    unsigned long long array_values[64];
    double f64_array_values[64];
    unsigned int array_count;
} yvex_deepseek_gguf_metadata;
typedef struct {
    unsigned long long source_contribution_count, descriptor_count, trunk_descriptor_count;
    unsigned long long draft_descriptor_count, pinned_standard_count, semantic_standard_count;
    unsigned long long extension_count;
    unsigned long long collection_counts[YVEX_TENSOR_COLLECTION_COUNT];
    unsigned long long metadata_count, header_scan_count, payload_bytes_read, source_identity;
    unsigned long long coverage_identity, mapping_identity;
    int complete;
} yvex_deepseek_gguf_map_summary;
typedef void *(*yvex_deepseek_gguf_map_allocate_fn)(size_t, void *);
typedef void (*yvex_deepseek_gguf_map_release_fn)(void *, void *);
typedef struct {
    yvex_deepseek_gguf_map_allocate_fn allocate;
    yvex_deepseek_gguf_map_release_fn release;
    void *context;
} yvex_deepseek_gguf_map_allocator;
typedef struct yvex_deepseek_gguf_map yvex_deepseek_gguf_map;
#define YVEX_DEEPSEEK_PAYLOAD_MAPPING_IDENTITY \
    YVEX_DEEPSEEK_GGUF_MAPPING_IDENTITY
typedef enum {
    YVEX_DEEPSEEK_PAYLOAD_FAILURE_NONE = 0, YVEX_DEEPSEEK_PAYLOAD_FAILURE_INVALID_ARGUMENT,
    YVEX_DEEPSEEK_PAYLOAD_FAILURE_SOURCE, YVEX_DEEPSEEK_PAYLOAD_FAILURE_ARCHITECTURE,
    YVEX_DEEPSEEK_PAYLOAD_FAILURE_COVERAGE, YVEX_DEEPSEEK_PAYLOAD_FAILURE_TRANSFORM_IR,
    YVEX_DEEPSEEK_PAYLOAD_FAILURE_MAPPING, YVEX_DEEPSEEK_PAYLOAD_FAILURE_MAPPING_IDENTITY,
    YVEX_DEEPSEEK_PAYLOAD_FAILURE_CONTRIBUTION, YVEX_DEEPSEEK_PAYLOAD_FAILURE_RANGE,
    YVEX_DEEPSEEK_PAYLOAD_FAILURE_BINDING, YVEX_DEEPSEEK_PAYLOAD_FAILURE_PLAN,
    YVEX_DEEPSEEK_PAYLOAD_FAILURE_ALLOCATION
} yvex_deepseek_payload_failure_code;
typedef struct {
    yvex_deepseek_payload_failure_code code;
    unsigned long long descriptor_index, contribution_index;
    yvex_source_payload_failure payload_failure;
} yvex_deepseek_payload_failure;
typedef struct {
    unsigned long long mapping_identity;
    char transform_identity[YVEX_DEEPSEEK_IDENTITY_CAP];
    unsigned long long source_snapshot_identity, descriptor_count, descriptors_covered;
    unsigned long long contribution_count, contributions_resolved, direct_contributions;
    unsigned long long fp8_weight_contributions, e8m0_scale_contributions, expert_contributions;
    unsigned long long i64_router_contributions, global_contributions, norm_contributions;
    unsigned long long shared_expert_contributions, output_head_contributions, draft_contributions;
    unsigned long long routed_expert_logical_bytes, output_head_logical_bytes, range_lookup_count;
    int complete;
} yvex_deepseek_payload_handoff_summary;
typedef struct yvex_deepseek_payload_handoff yvex_deepseek_payload_handoff;
typedef struct {
    const char *source_path;
    const char *models_root;
    const char *manifest_path;
    yvex_source_payload_budget budget;
    size_t chunk_bytes, page_bytes;
} yvex_deepseek_payload_handoff_options;
/*
 * One immutable registration is the complete family-visible ABI.  Consumers
 * select typed operations from this table; implementation helpers remain
 * private to the family translation unit. */
typedef struct {
    int (*build)(yvex_deepseek_v4_ir **out, const struct yvex_source_verification *verification,
                 yvex_deepseek_v4_ir_failure *failure, yvex_error *err);
    int (*build_with_allocator)(yvex_deepseek_v4_ir **out,
                                const struct yvex_source_verification *verification,
                                const yvex_deepseek_v4_ir_allocator *allocator,
                                yvex_deepseek_v4_ir_failure *failure, yvex_error *err);
    void (*close)(yvex_deepseek_v4_ir *ir);
    const yvex_deepseek_v4_model_spec *(*model)(const yvex_deepseek_v4_ir *ir);
    int (*execution_descriptor)(const yvex_deepseek_v4_ir *ir, const char *logical_model_identity,
                                yvex_model_execution_descriptor *descriptor, yvex_error *err);
    unsigned long long (*layer_count)(const yvex_deepseek_v4_ir *ir);
    const yvex_deepseek_v4_layer_spec *(*layer_at)(const yvex_deepseek_v4_ir *ir,
                                                   unsigned long long index);
    unsigned long long (*auxiliary_count)(const yvex_deepseek_v4_ir *ir);
    const yvex_deepseek_v4_auxiliary_spec *(*auxiliary_at)(const yvex_deepseek_v4_ir *ir,
                                                           unsigned long long index);
    const char *(*failure_name)(yvex_deepseek_v4_ir_failure_code code);
    const char *(*component_name)(yvex_deepseek_v4_ir_component component);
    const char *(*attention_name)(yvex_attention_class class_id);
    const char *(*kv_name)(yvex_deepseek_v4_kv_class class_id);
    const char *(*router_name)(yvex_deepseek_v4_router_class class_id);
    const char *(*source_weight_dtype_name)(yvex_deepseek_v4_source_weight_dtype dtype);
    const char *(*source_expert_dtype_name)(yvex_deepseek_v4_source_expert_dtype dtype);
    const char *(*source_quantization_name)(yvex_deepseek_v4_source_quantization quantization);
    const char *(*activation_stage_name)(yvex_attention_activation_stage stage);
    const char *(*activation_quantization_name)(yvex_attention_quantization quantization);
    const char *(*runtime_transform_name)(yvex_attention_transform transform);
    const char *(*sparse_topk_policy_name)(yvex_attention_topk_policy_id policy);
    unsigned long long (*recipe_count)(void);
    const yvex_deepseek_tensor_recipe *(*recipe_at)(unsigned long long index);
    int (*recipe_enabled)(const yvex_deepseek_tensor_recipe *recipe,
                          const yvex_deepseek_v4_layer_spec *layer);
    unsigned long long (*recipe_dimension)(const yvex_deepseek_tensor_recipe *recipe,
                                           unsigned int dimension,
                                           const yvex_deepseek_v4_layer_spec *layer,
                                           const yvex_deepseek_v4_model_spec *model);
} yvex_model_family_ir_api;
typedef struct {
    int (*build)(yvex_deepseek_tensor_coverage **out, const yvex_source_verification *verification,
                 const yvex_deepseek_v4_ir *ir, yvex_source_tensor_snapshot *snapshot,
                 const yvex_deepseek_tensor_coverage_options *options,
                 yvex_deepseek_tensor_coverage_failure *failure, yvex_error *err);
    int (*open_verified_source)(yvex_deepseek_tensor_coverage **out,
                                yvex_source_verification *verification, const char *source_path,
                                const char *models_root,
                                yvex_deepseek_tensor_coverage_failure *failure, yvex_error *err);
    void (*close)(yvex_deepseek_tensor_coverage *coverage);
    const yvex_deepseek_tensor_coverage_summary *(*summary)(const yvex_deepseek_tensor_coverage *coverage);
    const yvex_deepseek_tensor_coverage_row *(*at)(const yvex_deepseek_tensor_coverage *coverage,
                                                   unsigned long long index);
    const yvex_deepseek_tensor_coverage_row *(*find)(const yvex_deepseek_tensor_coverage *coverage,
                                                     const char *source_name);
    int (*find_index)(const yvex_deepseek_tensor_coverage *coverage,
                      const char *source_name,
                      unsigned long long *row_index);
    int (*find_source_index)(const yvex_deepseek_tensor_coverage *coverage,
                             const char *source_name,
                             unsigned long long *source_index);
    const char *(*collection_name)(yvex_tensor_collection collection);
    const char *(*failure_name)(yvex_deepseek_tensor_coverage_failure_code code);
} yvex_model_family_coverage_api;
typedef struct {
    int (*architecture_identity)(const yvex_deepseek_v4_ir *architecture,
                                 char output[YVEX_DEEPSEEK_IDENTITY_CAP]);
    int (*build)(yvex_transform_ir **out, const yvex_source_verification *verification,
                 const yvex_deepseek_v4_ir *architecture,
                 const yvex_deepseek_tensor_coverage *coverage,
                 const yvex_transform_builder_options *options, yvex_transform_failure *failure,
                 yvex_error *err);
} yvex_model_family_transform_api;
typedef struct {
    int (*build)(yvex_deepseek_gguf_map **out, const yvex_deepseek_v4_ir *ir,
                 const yvex_transform_ir *transform_ir,
                 yvex_deepseek_gguf_map_failure *failure, yvex_error *err);
    int (*build_with_allocator)(yvex_deepseek_gguf_map **out, const yvex_deepseek_v4_ir *ir,
                                const yvex_transform_ir *transform_ir,
                                const yvex_deepseek_gguf_map_allocator *allocator,
                                yvex_deepseek_gguf_map_failure *failure, yvex_error *err);
    void (*close)(yvex_deepseek_gguf_map *map);
    const yvex_deepseek_gguf_map_summary *(*summary)(const yvex_deepseek_gguf_map *map);
    const yvex_deepseek_gguf_descriptor *(*at)(const yvex_deepseek_gguf_map *map,
                                               unsigned long long index);
    const yvex_deepseek_gguf_contribution *(*contribution_at)(const yvex_deepseek_gguf_map *map,
                                                              unsigned long long index);
    const yvex_deepseek_gguf_descriptor *(*find_source)(const yvex_deepseek_gguf_map *map,
                                                        const char *source_name);
    const yvex_deepseek_gguf_descriptor *(*find_emitted)(const yvex_deepseek_gguf_map *map,
                                                         const char *emitted_name);
    const yvex_deepseek_gguf_descriptor *(*find_role)(const yvex_deepseek_gguf_map *map,
                                                      yvex_tensor_role role,
                                                      yvex_tensor_scope scope,
                                                      unsigned long long layer_index,
                                                      unsigned long long predictor_index);
    const yvex_deepseek_gguf_metadata *(*metadata_at)(const yvex_deepseek_gguf_map *map,
                                                      unsigned long long index);
    const yvex_deepseek_gguf_metadata *(*metadata_find)(const yvex_deepseek_gguf_map *map,
                                                        const char *key);
    const char *(*transform_name)(yvex_deepseek_gguf_transform transform);
    const char *(*failure_name)(yvex_deepseek_gguf_map_failure_code code);
} yvex_model_family_lowering_api;
typedef struct {
    int (*open)(yvex_deepseek_payload_handoff **out,
                const yvex_deepseek_payload_handoff_options *options,
                yvex_deepseek_payload_failure *failure, yvex_error *err);
    void (*close)(yvex_deepseek_payload_handoff *handoff);
    const yvex_deepseek_payload_handoff_summary *(*summary)(const yvex_deepseek_payload_handoff *handoff);
    const yvex_source_verification *(*verification)(const yvex_deepseek_payload_handoff *handoff);
    const yvex_deepseek_gguf_map *(*map)(const yvex_deepseek_payload_handoff *handoff);
    const yvex_transform_ir *(*transform_ir)(const yvex_deepseek_payload_handoff *handoff);
    const yvex_transform_binding *(*binding)(const yvex_deepseek_payload_handoff *handoff);
    yvex_source_payload_session *(*session)(yvex_deepseek_payload_handoff *handoff);
    const yvex_source_payload_plan *(*plan)(const yvex_deepseek_payload_handoff *handoff);
    const char *(*failure_name)(yvex_deepseek_payload_failure_code code);
} yvex_model_family_payload_api;
typedef struct yvex_model_family_api {
    unsigned int schema_version;
    const char *family_key;
    yvex_model_family_ir_api ir;
    yvex_model_family_coverage_api coverage;
    yvex_model_family_transform_api transform;
    yvex_model_family_lowering_api lowering;
    yvex_model_family_payload_api payload;
} yvex_model_family_api;
const yvex_model_family_coverage_api *yvex_model_deepseek_coverage_api(void);
const yvex_model_family_transform_api *yvex_model_deepseek_transform_api(void);
const yvex_model_family_lowering_api *yvex_model_deepseek_lowering_api(void);
const yvex_model_family_payload_api *yvex_model_deepseek_payload_api(void);
int yvex_transform_deepseek_architecture_identity(
    const yvex_deepseek_v4_ir *architecture, char output[YVEX_DEEPSEEK_IDENTITY_CAP]);
int yvex_gguf_map_deepseek_name(const char *native_name, char *target,
                                size_t target_cap, yvex_tensor_role *role,
                                yvex_weight_mapping_issue_kind *issue);
int yvex_quant_plan_build_deepseek_profile(
    yvex_quant_plan **out, const yvex_transform_ir *ir,
    const yvex_transform_binding *binding, const yvex_deepseek_gguf_map *map,
    yvex_quant_profile_kind profile, const yvex_quant_plan_options *options,
    yvex_quant_failure *failure, yvex_error *err);
int yvex_quant_plan_build_deepseek_policy(
    yvex_quant_plan **out, const yvex_transform_ir *ir,
    const yvex_transform_binding *binding, const yvex_deepseek_gguf_map *map,
    const yvex_quant_policy *policy, const char *imatrix_identity,
    const yvex_quant_plan_options *options,
    yvex_quant_failure *failure, yvex_error *err);
const yvex_deepseek_gguf_map *yvex_quant_plan_lowering(
    const yvex_quant_plan *plan);
int yvex_artifact_admit_deepseek(
    const yvex_artifact *artifact, yvex_complete_artifact_admission *out,
    yvex_artifact_admission_failure *failure, yvex_error *err);
int yvex_deepseek_materialization_projection(
    const yvex_deepseek_gguf_map *map, yvex_materialization_projection *out,
    yvex_error *err);
int yvex_runtime_descriptor_build_deepseek(
    yvex_runtime_descriptor **out,
    const yvex_complete_artifact_admission *admission,
    const yvex_materialization_session *session,
    const yvex_deepseek_gguf_map *map, const yvex_deepseek_v4_ir *ir,
    yvex_runtime_descriptor_failure *failure, yvex_error *err);
int yvex_model_mapping_report_deepseek(
    const yvex_model_target_request *request, yvex_model_target_report *report, yvex_error *err);
const yvex_model_family_api *yvex_model_register_deepseek_v4(void);
const yvex_family_compiler_adapter *yvex_compiler_family_deepseek_v4(void);
#endif
