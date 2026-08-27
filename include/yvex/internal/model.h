/*
 * Keep model identity-bearing numeric facts upstream of graph consumers.
 *
 * Enum values are versioned identity inputs shared unchanged from architecture admission through
 * runtime and graph planning. A numeric policy value is not graph or backend capability.
 */
#ifndef INCLUDE_YVEX_INTERNAL_MODEL_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MODEL_H_INCLUDED

#include <yvex/catalog.h>
#include <yvex/model.h>
#include <yvex/source.h>
#include <yvex/internal/core.h>

#define YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 1u
#define YVEX_MODEL_EXECUTION_IDENTITY_CAP (YVEX_SHA256_HEX_BYTES + 1u)
#define YVEX_MODEL_EXECUTION_FEATURE_LAYER_CAP 8u
#define YVEX_MODEL_EXECUTION_SCALAR_COUNT 56u
#define YVEX_MODEL_EXECUTION_IDENTITY_COUNT 4u
#define YVEX_MODEL_EXECUTION_WIRE_BYTES                                                   \
    (8u + YVEX_MODEL_EXECUTION_SCALAR_COUNT * 8u +                                      \
     (YVEX_MODEL_EXECUTION_IDENTITY_COUNT + 1u) * YVEX_SHA256_HEX_BYTES)

unsigned long long yvex_remote_catalog_provider_count(const yvex_remote_catalog *catalog);
const char *yvex_remote_catalog_query(const yvex_remote_catalog *catalog);
const char *yvex_remote_model_kind_name(yvex_remote_model_kind kind);

typedef enum {
    YVEX_MODEL_ROPE_SCALING_NONE = 0,
    YVEX_MODEL_ROPE_SCALING_LINEAR,
    YVEX_MODEL_ROPE_SCALING_YARN
} yvex_model_rope_scaling;

typedef enum {
    YVEX_MODEL_STATE_SWA_RING = 0,
    YVEX_MODEL_STATE_COMPRESSED_HISTORY,
    YVEX_MODEL_STATE_HCA_HISTORY,
    YVEX_MODEL_STATE_INDEXER_HISTORY,
    YVEX_MODEL_STATE_MAIN_ROLLING,
    YVEX_MODEL_STATE_INDEXER_ROLLING,
    YVEX_MODEL_STATE_RESIDUAL_MIXING,
    YVEX_MODEL_STATE_DRAFT_PERSISTENT,
    YVEX_MODEL_STATE_CANDIDATE_DELTA,
    YVEX_MODEL_STATE_PREFIX_CHECKPOINT,
    YVEX_MODEL_STATE_CLASS_COUNT
} yvex_model_state_class;

#define YVEX_MODEL_STATE_CLASS_BIT(state_class) (1ull << (unsigned int)(state_class))

/*
 * Common execution planning consumes this sealed projection rather than a family structure.
 * Counts and dimensions are source-derived facts; their values are never capability claims.
 */
typedef struct {
    unsigned int schema_version;
    const char *logical_model_identity;
    const char *source_model_identity;
    const char *attention_schedule_identity;
    const char *persistent_state_identity;
    unsigned long long maximum_context, original_context;
    yvex_model_rope_scaling rope_scaling;
    unsigned long long rope_theta, compressed_rope_theta, rope_scaling_factor;
    unsigned long long rope_beta_fast, rope_beta_slow;
    unsigned long long layer_count, hidden_width, vocabulary_size;
    unsigned long long attention_heads, kv_heads, head_width;
    unsigned long long swa_layers, csa_layers, hca_layers, sliding_window;
    unsigned long long minimum_compression_ratio, maximum_compression_ratio;
    unsigned long long index_heads, index_head_width, index_topk;
    unsigned long long residual_streams, mhc_sinkhorn_iterations;
    double mhc_epsilon, normalization_epsilon;
    unsigned long long routed_experts, experts_per_row;
    unsigned long long shared_experts, routed_ffn_width, shared_ffn_width;
    unsigned long long hash_router_layer_count;
    double routed_scaling_factor, activation_limit;
    unsigned long long output_input_width, output_vocabulary_size;
    unsigned long long proposal_width, verification_width_maximum;
    unsigned long long draft_layer_count, target_feature_count;
    unsigned long long target_feature_layers[YVEX_MODEL_EXECUTION_FEATURE_LAYER_CAP];
    unsigned long long target_feature_width;
    unsigned long long markov_rank, confidence_width, persistent_state_class_mask;
    unsigned long long bos_token_id, eos_token_id, draft_noise_token_id;
} yvex_model_execution_descriptor_request;

typedef struct yvex_model_execution_descriptor {
    unsigned int schema_version;
    unsigned long long maximum_context, original_context;
    yvex_model_rope_scaling rope_scaling;
    unsigned long long rope_theta, compressed_rope_theta, rope_scaling_factor;
    unsigned long long rope_beta_fast, rope_beta_slow;
    unsigned long long layer_count, hidden_width, vocabulary_size;
    unsigned long long attention_heads, kv_heads, head_width;
    unsigned long long swa_layers, csa_layers, hca_layers, sliding_window;
    unsigned long long minimum_compression_ratio, maximum_compression_ratio;
    unsigned long long index_heads, index_head_width, index_topk;
    unsigned long long residual_streams, mhc_sinkhorn_iterations;
    double mhc_epsilon, normalization_epsilon;
    unsigned long long routed_experts, experts_per_row;
    unsigned long long shared_experts, routed_ffn_width, shared_ffn_width;
    unsigned long long hash_router_layer_count;
    double routed_scaling_factor, activation_limit;
    unsigned long long output_input_width, output_vocabulary_size;
    unsigned long long proposal_width, verification_width_maximum;
    unsigned long long draft_layer_count, target_feature_count;
    unsigned long long target_feature_layers[YVEX_MODEL_EXECUTION_FEATURE_LAYER_CAP];
    unsigned long long target_feature_width;
    unsigned long long markov_rank, confidence_width, persistent_state_class_mask;
    unsigned long long bos_token_id, eos_token_id, draft_noise_token_id;
    char logical_model_identity[YVEX_MODEL_EXECUTION_IDENTITY_CAP];
    char source_model_identity[YVEX_MODEL_EXECUTION_IDENTITY_CAP];
    char attention_schedule_identity[YVEX_MODEL_EXECUTION_IDENTITY_CAP];
    char persistent_state_identity[YVEX_MODEL_EXECUTION_IDENTITY_CAP];
    char identity[YVEX_MODEL_EXECUTION_IDENTITY_CAP];
} yvex_model_execution_descriptor;

int yvex_model_execution_descriptor_seal(
    const yvex_model_execution_descriptor_request *request,
    yvex_model_execution_descriptor *descriptor, yvex_error *err);
int yvex_model_execution_descriptor_encode(
    const yvex_model_execution_descriptor *descriptor,
    unsigned char output[YVEX_MODEL_EXECUTION_WIRE_BYTES], yvex_error *err);
int yvex_model_execution_descriptor_decode(
    const unsigned char *bytes, size_t byte_count,
    yvex_model_execution_descriptor *descriptor, yvex_error *err);

typedef enum {
    YVEX_ATTENTION_CLASS_SWA = 0,
    YVEX_ATTENTION_CLASS_CSA,
    YVEX_ATTENTION_CLASS_HCA
} yvex_attention_class;

typedef enum {
    YVEX_ATTENTION_ACTIVATION_NONE = 0,
    YVEX_ATTENTION_ACTIVATION_KV_NON_ROPE,
    YVEX_ATTENTION_ACTIVATION_COMPRESSOR_NON_ROTATED,
    YVEX_ATTENTION_ACTIVATION_COMPRESSOR_ROTATED,
    YVEX_ATTENTION_ACTIVATION_INDEXER_QUERY_ROTATED
} yvex_attention_activation_stage;

typedef enum {
    YVEX_ATTENTION_QUANT_NONE = 0,
    YVEX_ATTENTION_QUANT_FP8_E4M3_UE8M0_FAKE_DEQUANT,
    YVEX_ATTENTION_QUANT_FP4_E2M1_UE8M0_FAKE_DEQUANT
} yvex_attention_quantization;

typedef enum {
    YVEX_ATTENTION_AXIS_NONE = 0,
    YVEX_ATTENTION_AXIS_FINAL_DIMENSION
} yvex_attention_axis;

typedef enum {
    YVEX_ATTENTION_SCALE_NONE = 0,
    YVEX_ATTENTION_SCALE_UE8M0
} yvex_attention_scale_format;

typedef enum {
    YVEX_ATTENTION_TRANSFORM_NONE = 0,
    YVEX_ATTENTION_TRANSFORM_DAO_FHT_V1_1_0_POST2
} yvex_attention_transform;

typedef enum {
    YVEX_ATTENTION_TAIL_NONE = 0,
    YVEX_ATTENTION_TAIL_EXACT_OR_SHORT_FINAL_BLOCK
} yvex_attention_tail_policy;

typedef enum {
    YVEX_ATTENTION_NONFINITE_REFUSE = 0
} yvex_attention_nonfinite_policy;

typedef enum {
    YVEX_ATTENTION_TOPK_NONE = 0,
    YVEX_ATTENTION_TOPK_SCORE_DESC_ORDINAL_ASC_V1
} yvex_attention_topk_policy_id;

/* BF16 values are rounded-to-nearest-even at every model-visible projection,
 * normalization, position, fake-quant, attention-output, and output-projection
 * boundary. Working values are F32; scalar dot products, scores, softmax, and
 * reductions use deterministic widened accumulation before F32 publication.
 * Compressor input projections are the architecture's explicit F32 exception. */
typedef enum {
    YVEX_ATTENTION_COMPUTE_UNKNOWN = 0,
    YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1
} yvex_attention_compute_contract;

typedef struct {
    int required;
    yvex_attention_activation_stage stage;
    yvex_attention_quantization quantization;
    yvex_attention_axis block_axis;
    unsigned long long block_width;
    yvex_attention_scale_format scale_format;
    yvex_native_dtype scale_dtype;
    yvex_attention_transform pre_transform;
    yvex_attention_tail_policy tail_policy;
    yvex_attention_nonfinite_policy nonfinite_policy;
    int fake_quant_inplace;
    int zero_pad_hadamard_to_power_of_two;
} yvex_attention_activation_policy;

typedef struct {
    int required;
    unsigned int version;
    yvex_attention_topk_policy_id policy;
    unsigned long long k;
    int reject_nonfinite;
    int score_descending;
    int equal_score_ordinal_ascending;
    int plus_zero_equals_minus_zero;
    int duplicate_ordinal_refused;
    int output_ranked_order;
} yvex_attention_topk_policy;

typedef struct {
    unsigned long long rope_dimension;
    unsigned long long theta;
    unsigned long long scaling_factor;
    unsigned long long original_context;
    unsigned long long beta_fast;
    unsigned long long beta_slow;
    unsigned long long maximum_context;
    int partial_rope;
    int inverse_output_rotation;
} yvex_attention_position_policy;

typedef enum {
    YVEX_ATTENTION_NUMERIC_MISMATCH_NONE = 0,
    YVEX_ATTENTION_NUMERIC_MISMATCH_COMPUTE,
    YVEX_ATTENTION_NUMERIC_MISMATCH_ACTIVATION,
    YVEX_ATTENTION_NUMERIC_MISMATCH_TOPK
} yvex_attention_numeric_mismatch_code;

typedef struct {
    yvex_attention_numeric_mismatch_code code;
    unsigned long long policy_index, expected, actual;
} yvex_attention_numeric_mismatch;

/* Validate one family-selected numeric profile before it becomes a compiled graph fact. */
int yvex_model_attention_numeric_validate(
    yvex_attention_compute_contract compute_contract,
    yvex_attention_compute_contract expected_compute_contract,
    const yvex_attention_activation_policy *const *activation_policies,
    unsigned long long activation_policy_count,
    const yvex_attention_topk_policy *topk_policy,
    unsigned long long fp8_block_width, unsigned long long fp4_block_width,
    unsigned int topk_policy_version, yvex_attention_numeric_mismatch *mismatch);

/* Canonical identity updates keep shared numeric policy fields ordered once. */
int yvex_model_activation_identity_update(
    yvex_sha256 *hash, const yvex_attention_activation_policy *policy);
int yvex_model_topk_identity_update(
    yvex_sha256 *hash, const yvex_attention_topk_policy *policy);
int yvex_model_position_identity_update(
    yvex_sha256 *hash, const yvex_attention_position_policy *policy);

#endif
