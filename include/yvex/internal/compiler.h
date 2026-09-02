/*
 * Family compilation ends at one immutable execution envelope.
 *
 * The adapter supplies semantic policy while compiler-plane owners validate and persist the
 * result. Only the binding-publication boundary invokes these callbacks; runtime execution
 * consumes the persisted records.
 */
#ifndef INCLUDE_YVEX_INTERNAL_COMPILER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_COMPILER_H_INCLUDED

#include <yvex/core.h>
#include <yvex/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/media_target.h>
#include <yvex/internal/model.h>
#include <yvex/internal/semantic_decoder.h>
#include <yvex/registry.h>
#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_FAMILY_COMPILER_SCHEMA_V2 2u
#define YVEX_RUNTIME_EXECUTION_CAPABILITY_SCHEMA_V2 2u
#define YVEX_TRANSFORMER_PLAN_SCHEMA_V2 2u
#define YVEX_RUNTIME_LOGITS_SCHEMA_V3 3u
#define YVEX_RUNTIME_LOGITS_SCHEMA_V2 YVEX_RUNTIME_LOGITS_SCHEMA_V3
#define YVEX_RUNTIME_LOGITS_SCHEMA_V1 YVEX_RUNTIME_LOGITS_SCHEMA_V3
#define YVEX_SPECULATION_FAMILY_POLICY_SCHEMA_V1 1u
#define YVEX_COMPILED_CONTEXT_ENVELOPE_SCHEMA_V1 1u
#define YVEX_SPECULATION_MAX_BLOCK 8u
#define YVEX_SPECULATION_MAX_FEATURE_LAYERS YVEX_MODEL_EXECUTION_FEATURE_LAYER_CAP
#define YVEX_SPECULATION_IDENTITY_CAP (YVEX_SHA256_HEX_BYTES + 1u)

/* Component plans seal family geometry before generic resource execution. */
#define YVEX_COMPONENT_BINDING_SCHEMA_V1 1u
#define YVEX_COMPONENT_PLAN_SCHEMA_V1 1u
#define YVEX_COMPONENT_GEOMETRY_CAP 3u
#define YVEX_COMPONENT_OUTPUT_RANK_CAP 5u
typedef enum {
    YVEX_COMPONENT_FAILURE_NONE = 0,
    YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT,
    YVEX_COMPONENT_FAILURE_UNSUPPORTED,
    YVEX_COMPONENT_FAILURE_LIFECYCLE,
    YVEX_COMPONENT_FAILURE_MISSING_TENSOR,
    YVEX_COMPONENT_FAILURE_TENSOR_CONTRACT,
    YVEX_COMPONENT_FAILURE_BUDGET,
    YVEX_COMPONENT_FAILURE_MATERIALIZATION,
    YVEX_COMPONENT_FAILURE_NUMERIC,
    YVEX_COMPONENT_FAILURE_CANCELLED
} yvex_component_failure_code;
typedef struct {
    yvex_component_failure_code code;
    char tensor_name[256];
    unsigned long long expected, actual;
    const char *reason;
} yvex_component_failure;
typedef int (*yvex_component_cancelled_fn)(void *context);
typedef struct {
    const char *target_id, *component_id;
    yvex_backend_kind backend;
    unsigned long long batch;
    unsigned int geometry_rank;
    unsigned long long geometry[YVEX_COMPONENT_GEOMETRY_CAP];
    unsigned long long maximum_host_bytes;
} yvex_component_plan_request;
typedef struct {
    unsigned int schema_version;
    unsigned long long binding_id, binding_version;
    char target_id[128], component_id[64];
    yvex_backend_kind backend;
    unsigned long long batch;
    unsigned int geometry_rank;
    unsigned long long geometry[YVEX_COMPONENT_GEOMETRY_CAP];
    unsigned long long input_values, input_bytes;
    unsigned long long output_values, output_bytes, workspace_bytes;
    unsigned int output_rank;
    unsigned long long output_dims[YVEX_COMPONENT_OUTPUT_RANK_CAP];
    char identity[YVEX_SHA256_HEX_BYTES];
    int complete;
} yvex_component_plan;
typedef struct {
    const yvex_component_plan *plan;
    const float *input;
    float *output;
    unsigned long long output_capacity;
    yvex_component_cancelled_fn cancelled;
    void *cancellation_context;
} yvex_component_execution_request;
typedef struct {
    unsigned long long batch, output_values;
    unsigned int output_rank;
    unsigned long long output_dims[YVEX_COMPONENT_OUTPUT_RANK_CAP];
    unsigned long long tensor_reads, payload_bytes_read, peak_workspace_bytes;
    char artifact_identity[YVEX_SHA256_HEX_BYTES];
    char execution_identity[YVEX_SHA256_HEX_BYTES];
    int complete;
} yvex_component_execution_result;
typedef struct yvex_runtime_capabilities {
    int attention_semantics_ready, attention_core_ready, attention_envelope_ready;
    int cpu_prefill_eager_ready, cpu_decode_eager_ready, cuda_prefill_eager_ready;
    int cuda_decode_eager_ready;
    int cuda_eager_implemented, cuda_piecewise_graph_implemented, cuda_full_graph_implemented;
    int cuda_prefill_piecewise_graph_ready, cuda_decode_piecewise_graph_ready;
    int cuda_prefill_full_graph_ready, cuda_decode_full_graph_ready;
    int attention_weight_residency_ready, attention_workspace_ready;
    int attention_state_delta_ready, attention_operator_ready, attention_trace_ready;
    int attention_profile_ready, attention_benchmark_ready, mixed_attention_ready;
    int speculative_attention_ready, persistent_kv_ready;
    int moe_plan_ready, moe_router_ready, moe_routed_expert_ready, moe_shared_expert_ready;
    int moe_block_ready;
    int transformer_ready, output_head_binding_ready, output_head_projection_ready;
    int logits_cpu_ready, logits_cuda_ready, logits_prefill_ready, logits_decode_ready;
    int logits_full_vocabulary_ready, logits_hidden_contract_ready;
    int logits_partial_progress_ready, logits_ready;
    int generation_ready;
} yvex_runtime_capabilities;

typedef enum {
    YVEX_TRANSFORMER_INITIAL_REPEAT_STREAMS = 0,
    YVEX_TRANSFORMER_INITIAL_POLICY_COUNT
} yvex_transformer_initial_policy;

typedef enum {
    YVEX_TRANSFORMER_FINAL_SIGMOID_MHC_RMS = 0,
    YVEX_TRANSFORMER_FINAL_POLICY_COUNT
} yvex_transformer_final_policy;

typedef struct yvex_transformer_family_policy {
    unsigned int schema_version;
    yvex_transformer_initial_policy initial_policy;
    yvex_transformer_final_policy final_policy;
    unsigned long long residual_streams, hidden_width, expanded_width;
    unsigned long long maximum_context, sinkhorn_iterations;
    double mhc_epsilon, output_norm_epsilon;
    int attention_then_moe, deferred_ffn_post, final_norm_after_head;
} yvex_transformer_family_policy;

typedef struct yvex_logits_family_policy {
    unsigned int schema_version;
    int separate_output_head, tied_output_head, output_head_bias;
} yvex_logits_family_policy;

typedef struct yvex_runtime_logits_plan_summary {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long output_head_tensor_id, row_width, row_count, row_bytes;
    unsigned long long encoded_bytes, vocabulary_size, hidden_width;
    yvex_tensor_role role;
    unsigned int qtype;
    int separate_output_head, output_head_bias;
    char artifact_identity[YVEX_SHA256_HEX_BYTES];
    char materialization_identity[YVEX_SHA256_HEX_BYTES];
    char logical_model_identity[YVEX_SHA256_HEX_BYTES];
    char runtime_numeric_identity[YVEX_SHA256_HEX_BYTES];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_BYTES];
    char transformer_plan_identity[YVEX_SHA256_HEX_BYTES];
    char output_head_plan_identity[YVEX_SHA256_HEX_BYTES];
} yvex_runtime_logits_plan_summary;

typedef struct yvex_speculation_family_policy {
    unsigned int schema_version;
    unsigned long long block_size, noise_token_id;
    unsigned long long target_feature_layer_count;
    unsigned long long target_feature_layers[YVEX_SPECULATION_MAX_FEATURE_LAYERS];
    unsigned long long target_feature_width, concatenated_feature_width;
    unsigned long long draft_layer_count, markov_rank, accepted_prefix_maximum;
    yvex_tensor_role feature_projection_role, feature_norm_role;
    yvex_tensor_role output_norm_role, markov_embedding_role;
    yvex_tensor_role markov_output_role, confidence_role;
    int parallel_block_backbone, sequential_markov, confidence_available;
    int shares_embedding, shares_output_head, target_verification_required;
    char policy_identity[YVEX_SPECULATION_IDENTITY_CAP];
} yvex_speculation_family_policy;

struct yvex_graph_compiler_api;
struct yvex_compilation_runtime_binding_request;
struct yvex_runtime_descriptor_summary;
struct yvex_tokenizer_family_policy;
struct yvex_artifact;
struct yvex_complete_artifact_admission;
struct yvex_artifact_admission_failure;
struct yvex_artifact_physical_compatibility;
struct yvex_materialization_session;
struct yvex_runtime_descriptor;
struct yvex_attention_plan;
struct yvex_transform_ir;
struct yvex_transform_binding;
struct yvex_source_verification;
struct yvex_quant_policy;
struct yvex_quant_plan;
struct yvex_gguf_writer_plan;
struct yvex_imatrix_data;
struct yvex_family_compilation_products;
struct yvex_compilation_source_summary;
typedef struct {
    unsigned long long ordinal, layer_index, predictor_index;
    yvex_tensor_scope tensor_scope;
    yvex_attention_class attention_class;
    yvex_attention_compute_contract compute_contract;
    unsigned long long compression_ratio, sliding_window, query_heads, kv_heads;
    unsigned long long head_dimension, rope_head_dimension, query_lora_rank;
    unsigned long long output_lora_rank, output_groups, output_group_input_width;
    unsigned long long hidden_dimension;
    unsigned long long indexer_heads, indexer_head_dimension, indexer_topk;
    unsigned long long compressor_ape_columns, indexer_ape_columns;
    double rms_norm_epsilon;
    unsigned long long residual_stream_count, residual_stream_width, residual_expanded_width;
    unsigned long long mhc_mixing_rows, mhc_mixing_columns, mhc_base_width, mhc_scale_width;
    unsigned long long mhc_sinkhorn_iterations, attention_input_norm_width;
    double mhc_epsilon, mhc_residual_post_multiplier;
    unsigned int mhc_entry_policy;
    int mhc_attention_pre_and_post, attention_input_norm_required;
    yvex_tensor_role attention_input_norm_role, mhc_function_role, mhc_base_role;
    yvex_tensor_role mhc_scale_role;
    int compressor_required, indexer_required;
    yvex_attention_position_policy position;
    yvex_attention_activation_policy attention_kv_activation;
    yvex_attention_activation_policy compressor_activation;
    yvex_attention_activation_policy compressor_rotated_activation;
    yvex_attention_activation_policy indexer_query_activation;
    yvex_attention_topk_policy sparse_topk;
} yvex_semantic_attention_layer;
typedef int (*yvex_semantic_attention_layer_fn)(
    const void *context, unsigned long long index,
    yvex_semantic_attention_layer *output);
typedef struct yvex_family_compiler_adapter yvex_family_compiler_adapter;
typedef struct yvex_operator_graph_ir yvex_operator_graph_ir;
typedef struct yvex_compiled_model_plan yvex_compiled_model_plan;

#define YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1 1u
typedef enum {
    YVEX_PHYSICAL_VARIANT_COMPLETE_MODEL = 1,
    YVEX_PHYSICAL_VARIANT_COMPONENT
} yvex_physical_variant_kind;
typedef struct {
    const char *target_id, *source_path, *models_root, *source_manifest_path;
    const char *quant_preset_name, *quant_policy_path, *imatrix_path, *backend;
    const char *component_id;
    unsigned int worker_count;
} yvex_physical_variant_request;
typedef struct {
    unsigned int schema_version;
    yvex_physical_variant_kind kind;
    char target_id[128], family[64], component_id[64], source_revision[65];
    char source_snapshot_identity[65], component_identity[65], transform_identity[65];
    unsigned long long shards, tensors, elements, payload_execution_bytes_read;
    unsigned int worker_count;
    int source_verified;
} yvex_physical_variant_summary;
typedef struct {
    const char *source_path, *component_id;
    int candidate_q8;
} yvex_component_variant_source_request;
typedef struct {
    void *owner;
    void (*close)(void *owner);
    const struct yvex_transform_ir *transform_ir;
    const struct yvex_transform_binding *transform_binding;
    char architecture[64], target_id[128], component_id[64];
    char source_snapshot_identity[65], component_identity[65];
    char component_manifest_identity[65], architecture_identity[65], role_map_identity[65];
    unsigned long long source_snapshot_key;
    yvex_physical_variant_summary summary;
} yvex_component_variant_source;
struct yvex_physical_variant_api;
typedef struct {
    unsigned int schema_version;
    const char *target_id, *family, *source_revision, *profile_name;
    const char *candidate_profile_name, *candidate_component_id;
    unsigned long long candidate_q8_semantic_role_mask;
    int (*source_open)(yvex_component_variant_source *out,
                       const yvex_component_variant_source_request *request,
                       yvex_error *err);
    const struct yvex_physical_variant_api *(*physical_variant)(void);
    int (*media_target_profile)(yvex_media_target_profile *, yvex_error *);
    const yvex_media_execution_recipe *media_execution;
} yvex_component_variant_adapter;
#define YVEX_QUANT_PRESET_CATALOG_SCHEMA_V1 1u
typedef struct {
    unsigned int schema_version;
    const char *target_id;
    unsigned long long (*count)(void);
    const char *(*name)(unsigned long long index);
    int (*open)(struct yvex_quant_policy **out, const char *name, yvex_error *err);
} yvex_quant_preset_catalog;
typedef struct yvex_physical_variant_session yvex_physical_variant_session;
typedef struct {
    const yvex_physical_variant_summary *summary;
    const struct yvex_quant_plan *plan;
    const struct yvex_gguf_writer_plan *writer;
    const struct yvex_imatrix_data *imatrix;
} yvex_physical_variant_view;
typedef struct yvex_physical_variant_api {
    unsigned int schema_version;
    int (*open)(yvex_physical_variant_session **out,
                const yvex_physical_variant_request *request,
                yvex_error *err);
    void (*close)(yvex_physical_variant_session **session);
    const yvex_physical_variant_view *(*view)(
        const yvex_physical_variant_session *session);
} yvex_physical_variant_api;
#define YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1 1u
#define YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2 2u
#define YVEX_SEMANTIC_NUMERIC_CONTRACT_SCHEMA_V1 1u
typedef struct {
    unsigned int schema_version;
    unsigned int numeric_schema_version;
    char identity[YVEX_SHA256_HEX_BYTES];
    char algorithm_revision[128];
    unsigned long long compute_policy_count;
    unsigned long long activation_policy_count;
    unsigned long long sparse_topk_policy_count;
} yvex_semantic_numeric_contract;
typedef struct {
    char canonical_id[64], identity[YVEX_SHA256_HEX_BYTES];
    unsigned long long shards, tensors;
    unsigned int phase;
    int weighted, release_after_phase;
} yvex_semantic_component;
typedef struct {
    unsigned int source_phase, destination_phase, data_classes, lifetime;
} yvex_semantic_phase_edge;
typedef struct {
    const char *repository, *revision, *subtree;
    const char *source_snapshot_identity, *component_manifest_identity;
    const char *phase_dag_identity, *architecture_identity, *role_map_identity;
    const char *unresolved_requirements_identity;
    unsigned long long weighted_components, shards, tensors, elements, payload_bytes;
    const yvex_semantic_component *components;
    unsigned long long component_count;
    const yvex_semantic_phase_edge *phase_edges;
    unsigned long long phase_edge_count;
} yvex_semantic_composite_request;
typedef struct {
    char repository[128], revision[64], subtree[128];
    char source_snapshot_identity[YVEX_SHA256_HEX_BYTES];
    char component_manifest_identity[YVEX_SHA256_HEX_BYTES];
    char phase_dag_identity[YVEX_SHA256_HEX_BYTES], architecture_identity[YVEX_SHA256_HEX_BYTES];
    char role_map_identity[YVEX_SHA256_HEX_BYTES];
    char unresolved_requirements_identity[YVEX_SHA256_HEX_BYTES];
    unsigned long long component_count, weighted_component_count, phase_edge_count;
    unsigned long long shards, tensors, elements, payload_bytes;
} yvex_semantic_composite_summary;
typedef struct { const char *key, *value; } yvex_semantic_reference_request;
typedef struct { char key[48], value[128]; } yvex_semantic_reference;
typedef struct yvex_semantic_model_ir yvex_semantic_model_ir;
typedef struct {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long attention_layer_count, draft_attention_layer_count;
    unsigned long long decoder_layer_count;
    yvex_model_execution_descriptor execution_descriptor;
    yvex_semantic_numeric_contract numeric_contract;
    char target_id[128];
    char source_model_identity[YVEX_SHA256_HEX_BYTES];
    char logical_model_identity[YVEX_SHA256_HEX_BYTES];
    char semantic_payload_identity[YVEX_SHA256_HEX_BYTES];
    char identity[YVEX_SHA256_HEX_BYTES];
    yvex_semantic_composite_summary composite;
} yvex_semantic_model_ir_summary;
typedef struct {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    const char *target_id;
    const char *source_model_identity;
    const char *logical_model_identity;
    const char *semantic_payload_identity;
    const yvex_model_execution_descriptor *execution_descriptor;
    const yvex_semantic_numeric_contract *numeric_contract;
    const void *attention_context;
    yvex_semantic_attention_layer_fn attention_layer;
    unsigned long long attention_layer_count;
    yvex_semantic_attention_layer_fn draft_attention_layer;
    unsigned long long draft_attention_layer_count;
    const void *decoder_context;
    yvex_semantic_decoder_layer_fn decoder_layer;
    unsigned long long decoder_layer_count;
    const yvex_semantic_composite_request *composite;
    const yvex_semantic_reference_request *references;
    unsigned long long reference_count;
} yvex_semantic_model_ir_request;
int yvex_semantic_model_ir_seal(
    yvex_semantic_model_ir **out,
    const yvex_semantic_model_ir_request *request, yvex_error *err);
const yvex_semantic_model_ir_summary *yvex_semantic_model_ir_summary_get(
    const yvex_semantic_model_ir *model);
int yvex_semantic_model_ir_attention_view(
    const yvex_semantic_model_ir *model, yvex_tensor_scope tensor_scope,
    const yvex_semantic_attention_layer **layers,
    unsigned long long *layer_count);
int yvex_semantic_model_ir_decoder_view(
    const yvex_semantic_model_ir *model,
    const yvex_semantic_decoder_layer **layers,
    unsigned long long *layer_count);
int yvex_semantic_model_ir_composite_view(
    const yvex_semantic_model_ir *model,
    const yvex_semantic_component **components, unsigned long long *component_count,
    const yvex_semantic_phase_edge **phase_edges, unsigned long long *phase_edge_count);
const char *yvex_semantic_model_ir_reference(
    const yvex_semantic_model_ir *model, const char *key);
void yvex_semantic_model_ir_close(yvex_semantic_model_ir **model);
#define YVEX_FAMILY_BINDING_PIPELINE_SCHEMA_V1 1u
typedef struct yvex_family_compilation_source {
    void *owner;
    const struct yvex_source_verification *verification;
    const struct yvex_transform_ir *transform_ir;
    const struct yvex_transform_binding *transform_binding;
    const struct yvex_artifact_lowering_map *artifact_lowering;
    const struct yvex_compilation_source_summary *source_summary;
    const void *lowering_context;
} yvex_family_compilation_source;
/*
 * Family callbacks project semantic facts into one generic binding-compilation lifecycle.
 * Every borrowed view remains valid until source_close; semantic_model is independently owned.
 */
typedef struct yvex_family_binding_pipeline {
    unsigned int schema_version;
    int (*source_open)(yvex_family_compilation_source *out,
                       const struct yvex_compilation_runtime_binding_request *request,
                       yvex_error *err);
    void (*source_close)(void *owner);
    int (*artifact_admit)(const struct yvex_artifact *artifact,
                          struct yvex_complete_artifact_admission *out,
                          struct yvex_artifact_admission_failure *failure,
                          yvex_error *err);
    int (*semantic_model_build)(yvex_semantic_model_ir **out,
                                const struct yvex_source_verification *verification,
                                yvex_error *err);
    int (*runtime_descriptor_build)(
        struct yvex_runtime_descriptor **out,
        const struct yvex_complete_artifact_admission *admission,
        struct yvex_materialization_session *materialization,
        const void *lowering_context,
        const yvex_semantic_model_ir *semantic_model,
        yvex_error *err);
    int (*quant_plan_default)(struct yvex_quant_plan **out,
                              const struct yvex_transform_ir *transform,
                              const struct yvex_transform_binding *binding,
                              const void *lowering_context, yvex_error *err);
    int (*quant_plan_policy)(struct yvex_quant_plan **out,
                             const struct yvex_transform_ir *transform,
                             const struct yvex_transform_binding *binding,
                             const void *lowering_context,
                             const struct yvex_quant_policy *policy,
                             const char *imatrix_identity, yvex_error *err);
    const char *tokenizer_architecture;
    const char *imatrix_source_identity;
    const char *imatrix_dataset_identity;
    const char *imatrix_producer;
    unsigned int imatrix_producer_version;
} yvex_family_binding_pipeline;

struct yvex_family_compiler_adapter {
    unsigned int schema_version;
    unsigned long long adapter_id, adapter_version;
    const char *target_id, *family, *logical_transform_identity;
    const struct yvex_graph_compiler_api *(*graph)(void);
    int (*operator_graph_build)(
        yvex_operator_graph_ir **out,
        const yvex_semantic_model_ir *semantic_model,
        const struct yvex_attention_plan *attention,
        const struct yvex_attention_plan *draft_attention,
        yvex_error *err);
    int (*execution_capabilities)(yvex_runtime_capabilities *out);
    int (*transformer_policy)(const struct yvex_runtime_descriptor_summary *,
                              yvex_transformer_family_policy *);
    int (*logits_policy)(yvex_logits_family_policy *out);
    int (*speculation_policy)(const struct yvex_runtime_descriptor_summary *,
                              yvex_speculation_family_policy *);
    int (*tokenizer_policy)(struct yvex_tokenizer_family_policy *, yvex_error *);
    const yvex_physical_variant_api *(*physical_variant)(void);
    const yvex_family_binding_pipeline *binding_pipeline;
    int (*binding_compile)(
        const struct yvex_family_compiler_adapter *adapter,
        const struct yvex_compilation_runtime_binding_request *request,
        struct yvex_family_compilation_products *products, void **owner,
        yvex_error *err);
};

/*
 * A successful compile lends every product from owner through release. The compile callback sets
 * release before publishing owned state so callers can discharge partial failures identically.
 */
typedef struct yvex_family_compilation_products {
    const char *directory;
    const struct yvex_complete_artifact_admission *admission;
    const struct yvex_artifact_physical_compatibility *physical_compatibility;
    const struct yvex_materialization_session *materialization;
    const struct yvex_runtime_descriptor *runtime_descriptor;
    const yvex_operator_graph_ir *operator_graph;
    const struct yvex_physical_execution_ir *physical_execution;
    const yvex_compiled_model_plan *compiled_plan;
    const struct yvex_attention_plan *attention_plan, *draft_attention_plan;
    unsigned long long family_adapter_id, family_adapter_version;
    const char *artifact_format, *logical_transform_identity;
    unsigned int artifact_format_version;
    const yvex_runtime_capabilities *capabilities;
    const yvex_transformer_family_policy *transformer_policy;
    const yvex_logits_family_policy *logits_policy;
    const yvex_speculation_family_policy *speculation_policy;
    const struct yvex_tokenizer_family_policy *tokenizer_policy;
    void (*release)(void *owner);
} yvex_family_compilation_products;

int yvex_family_binding_compile(
    const yvex_family_compiler_adapter *adapter,
    const struct yvex_compilation_runtime_binding_request *request,
    yvex_family_compilation_products *products, void **owner, yvex_error *err);

int yvex_runtime_capabilities_identity(
    const yvex_runtime_capabilities *facts,
    char output[YVEX_SHA256_HEX_BYTES]);
int yvex_runtime_capabilities_admitted_by(
    const yvex_runtime_capabilities *facts,
    const yvex_runtime_capabilities *maximum);
int yvex_runtime_capabilities_contract_valid(
    const yvex_runtime_capabilities *facts);

struct yvex_materialization_session;
struct yvex_runtime_descriptor;
struct yvex_attention_plan;
struct yvex_moe_plan;
struct yvex_transformer_plan;
typedef struct {
    const yvex_operator_graph_ir *operator_graph;
    const struct yvex_materialization_session *materialization;
    const struct yvex_runtime_descriptor *descriptor;
    const struct yvex_attention_plan *attention, *draft_attention;
    const struct yvex_graph_compiler_api *graph;
    unsigned long long family_adapter_id, family_adapter_version;
    yvex_runtime_capabilities capabilities;
    yvex_transformer_family_policy transformer_policy;
    yvex_logits_family_policy logits_policy;
} yvex_compiled_model_plan_request;
typedef struct {
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long tensor_count, layer_count, draft_layer_count;
    const char *model_execution_identity;
    unsigned long long semantic_maximum_context;
    const yvex_runtime_capabilities *capabilities;
    const char *artifact_identity, *materialization_identity;
    const char *runtime_descriptor_identity;
    const char *attention_plan_identity, *draft_attention_plan_identity;
    const char *moe_plan_identity, *draft_moe_plan_identity;
    const char *transformer_plan_identity, *draft_transformer_plan_identity;
    const char *output_head_plan_identity;
} yvex_compiled_model_plan_admission;
typedef struct {
    unsigned int schema_version;
    unsigned long long semantic_maximum_context;
    unsigned long long target_maximum_context, draft_maximum_context;
    int draft_available;
    char model_execution_identity[YVEX_SHA256_HEX_BYTES];
    char target_transformer_identity[YVEX_SHA256_HEX_BYTES];
    char draft_transformer_identity[YVEX_SHA256_HEX_BYTES];
} yvex_compiled_context_envelope;
int yvex_compiled_model_plan_build(
    yvex_compiled_model_plan **out,
    const yvex_compiled_model_plan_request *request, yvex_error *err);
int yvex_compiled_model_plan_encode(
    const yvex_compiled_model_plan *plan, yvex_core_bytes *bytes,
    yvex_error *err);
int yvex_compiled_model_plan_decode(
    yvex_compiled_model_plan **out, const unsigned char *data, size_t count,
    yvex_error *err);
int yvex_compiled_model_plan_admit(
    const yvex_compiled_model_plan *plan,
    const yvex_compiled_model_plan_admission *admission);
int yvex_compiled_model_plan_context_envelope(
    const yvex_compiled_model_plan *plan,
    const char *model_execution_identity,
    unsigned long long semantic_maximum_context,
    yvex_compiled_context_envelope *envelope, yvex_error *err);
int yvex_compiled_context_envelope_admit(
    const yvex_compiled_context_envelope *envelope,
    unsigned long long requested_context, int require_draft, yvex_error *err);
const struct yvex_moe_plan *yvex_compiled_model_plan_moe(
    const yvex_compiled_model_plan *plan, int draft);
const struct yvex_transformer_plan *yvex_compiled_model_plan_transformer(
    const yvex_compiled_model_plan *plan, int draft);
const struct yvex_runtime_logits_plan_summary *yvex_compiled_model_plan_output_head(
    const yvex_compiled_model_plan *plan);
const char *yvex_compiled_model_plan_operator_graph_identity(
    const yvex_compiled_model_plan *plan);
void yvex_compiled_model_plan_close(yvex_compiled_model_plan **plan);
int yvex_output_head_plan_build(
    yvex_runtime_logits_plan_summary *out,
    unsigned long long family_adapter_id,
    unsigned long long family_adapter_version,
    const struct yvex_materialization_session *materialization,
    const struct yvex_runtime_descriptor *descriptor,
    const struct yvex_transformer_plan *transformer,
    const yvex_logits_family_policy *policy, yvex_error *err);
int yvex_output_head_plan_validate(
    const yvex_runtime_logits_plan_summary *summary, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
