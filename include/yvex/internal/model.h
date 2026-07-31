/*
 * Keep model identity-bearing numeric facts upstream of graph consumers.
 *
 * Enum values are versioned identity inputs shared unchanged from architecture admission through
 * runtime and graph planning. A numeric policy value is not graph or backend capability.
 */
#ifndef INCLUDE_YVEX_INTERNAL_MODEL_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MODEL_H_INCLUDED

#include <yvex/model.h>
#include <yvex/source.h>

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

#endif
