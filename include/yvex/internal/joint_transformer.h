/* Typed recipe and execution boundary for a joint-modality CUDA Transformer. */
#ifndef INCLUDE_YVEX_INTERNAL_JOINT_TRANSFORMER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_JOINT_TRANSFORMER_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/backend.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_backend yvex_backend;
#define YVEX_TRANSFORMER_JOINT_SCHEMA_V1 1u
#define YVEX_TRANSFORMER_JOINT_SCHEMA_V2 2u
#define YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT 10u
#define YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT 35u

typedef struct yvex_transformer_joint_encoded_weight {
    const unsigned char *encoded;
    unsigned long long encoded_bytes, row_count, row_width, row_bytes;
    unsigned int qtype;
} yvex_transformer_joint_encoded_weight;

typedef enum {
    YVEX_TRANSFORMER_JOINT_NORM1 = 0,
    YVEX_TRANSFORMER_JOINT_QKV,
    YVEX_TRANSFORMER_JOINT_Q_NORM,
    YVEX_TRANSFORMER_JOINT_K_NORM,
    YVEX_TRANSFORMER_JOINT_ATTENTION_OUT,
    YVEX_TRANSFORMER_JOINT_NORM2,
    YVEX_TRANSFORMER_JOINT_FC1,
    YVEX_TRANSFORMER_JOINT_FC2,
    YVEX_TRANSFORMER_JOINT_ADALN_WEIGHT,
    YVEX_TRANSFORMER_JOINT_ADALN_BIAS
} yvex_transformer_joint_weight_slot;

typedef enum {
    YVEX_TRANSFORMER_JOINT_AUDIO_WEIGHT = 0,
    YVEX_TRANSFORMER_JOINT_AUDIO_BIAS,
    YVEX_TRANSFORMER_JOINT_VIDEO_WEIGHT,
    YVEX_TRANSFORMER_JOINT_VIDEO_BIAS,
    YVEX_TRANSFORMER_JOINT_CONDITION_WEIGHT,
    YVEX_TRANSFORMER_JOINT_CONDITION_BIAS,
    YVEX_TRANSFORMER_JOINT_TIME_IN_WEIGHT,
    YVEX_TRANSFORMER_JOINT_TIME_IN_BIAS,
    YVEX_TRANSFORMER_JOINT_TIME_OUT_WEIGHT,
    YVEX_TRANSFORMER_JOINT_TIME_OUT_BIAS,
    YVEX_TRANSFORMER_JOINT_REFINER_WEIGHTS,
    YVEX_TRANSFORMER_JOINT_REFINER_FINAL = YVEX_TRANSFORMER_JOINT_REFINER_WEIGHTS + 16,
    YVEX_TRANSFORMER_JOINT_ROPE_INV_FREQ,
    YVEX_TRANSFORMER_JOINT_FINAL_NORM,
    YVEX_TRANSFORMER_JOINT_FINAL_ADALN_WEIGHT,
    YVEX_TRANSFORMER_JOINT_FINAL_ADALN_BIAS,
    YVEX_TRANSFORMER_JOINT_VIDEO_OUT_WEIGHT,
    YVEX_TRANSFORMER_JOINT_VIDEO_OUT_BIAS,
    YVEX_TRANSFORMER_JOINT_AUDIO_OUT_WEIGHT,
    YVEX_TRANSFORMER_JOINT_AUDIO_OUT_BIAS
} yvex_transformer_joint_external_slot;

typedef enum {
    YVEX_TRANSFORMER_QKV_LAYOUT_UNKNOWN = 0,
    YVEX_TRANSFORMER_QKV_LAYOUT_GLOBAL_THREE,
    YVEX_TRANSFORMER_QKV_LAYOUT_PER_HEAD_THREE
} yvex_transformer_qkv_layout;

typedef enum {
    YVEX_TRANSFORMER_SWIGLU_LAYOUT_UNKNOWN = 0,
    YVEX_TRANSFORMER_SWIGLU_LAYOUT_GATE_THEN_UP,
    YVEX_TRANSFORMER_SWIGLU_LAYOUT_UP_THEN_GATE
} yvex_transformer_swiglu_layout;

typedef struct yvex_transformer_joint_recipe {
    unsigned int schema_version;
    yvex_transformer_qkv_layout qkv_layout;
    yvex_transformer_swiglu_layout swiglu_layout;
    const char *identity_domain;
    unsigned long long hidden_width, attention_heads, head_dimension, attention_width;
    unsigned long long ffn_width, timestep_width, rotary_width;
    unsigned long long modality_count, modulation_parameters;
    unsigned long long block_count, refiner_block_count;
    unsigned long long maximum_timesteps, maximum_packed_rows;
    unsigned long long video_input_width, audio_input_width, condition_input_width;
    unsigned long long video_output_width, audio_output_width;
    yvex_backend_linear_numeric_policy video_output_numeric, audio_output_numeric;
} yvex_transformer_joint_recipe;

typedef struct yvex_transformer_joint_block_result {
    unsigned long long packed_rows, block_count, resident_bytes, kernel_launches;
    unsigned long long h2d_bytes, d2h_bytes, device_bytes, temporary_bytes;
    char residency_identity[65], execution_identity[65];
    int complete;
} yvex_transformer_joint_block_result;

typedef struct yvex_transformer_joint_block_observation {
    unsigned long long completed_blocks, packed_rows, hidden_width, value_count;
    const float *values;
} yvex_transformer_joint_block_observation;

typedef enum {
    YVEX_TRANSFORMER_JOINT_STAGE_MODULATION = 0,
    YVEX_TRANSFORMER_JOINT_STAGE_NORM1,
    YVEX_TRANSFORMER_JOINT_STAGE_MODULATED1,
    YVEX_TRANSFORMER_JOINT_STAGE_QKV_F32,
    YVEX_TRANSFORMER_JOINT_STAGE_QKV_BF16,
    YVEX_TRANSFORMER_JOINT_STAGE_QUERY,
    YVEX_TRANSFORMER_JOINT_STAGE_KEY,
    YVEX_TRANSFORMER_JOINT_STAGE_VALUE,
    YVEX_TRANSFORMER_JOINT_STAGE_QUERY_NORM,
    YVEX_TRANSFORMER_JOINT_STAGE_KEY_NORM,
    YVEX_TRANSFORMER_JOINT_STAGE_QUERY_ROTARY,
    YVEX_TRANSFORMER_JOINT_STAGE_KEY_ROTARY,
    YVEX_TRANSFORMER_JOINT_STAGE_ATTENTION_F32,
    YVEX_TRANSFORMER_JOINT_STAGE_ATTENTION_BF16,
    YVEX_TRANSFORMER_JOINT_STAGE_ATTENTION_PROJECTION_F32,
    YVEX_TRANSFORMER_JOINT_STAGE_ATTENTION_PROJECTION_BF16,
    YVEX_TRANSFORMER_JOINT_STAGE_ATTENTION_RESIDUAL,
    YVEX_TRANSFORMER_JOINT_STAGE_NORM2,
    YVEX_TRANSFORMER_JOINT_STAGE_MODULATED2,
    YVEX_TRANSFORMER_JOINT_STAGE_FC1_F32,
    YVEX_TRANSFORMER_JOINT_STAGE_FC1_BF16,
    YVEX_TRANSFORMER_JOINT_STAGE_SWIGLU,
    YVEX_TRANSFORMER_JOINT_STAGE_FC2_F32,
    YVEX_TRANSFORMER_JOINT_STAGE_FC2_BF16,
    YVEX_TRANSFORMER_JOINT_STAGE_OUTPUT,
    YVEX_TRANSFORMER_JOINT_STAGE_INPUT_HIDDEN,
    YVEX_TRANSFORMER_JOINT_STAGE_INPUT_TIME,
    YVEX_TRANSFORMER_JOINT_STAGE_TIME_INPUT,
    YVEX_TRANSFORMER_JOINT_STAGE_TIME_PROJECTION_IN,
    YVEX_TRANSFORMER_JOINT_STAGE_TIME_BIAS_IN,
    YVEX_TRANSFORMER_JOINT_STAGE_TIME_ACTIVATION,
    YVEX_TRANSFORMER_JOINT_STAGE_TIME_PROJECTION_OUT,
    YVEX_TRANSFORMER_JOINT_STAGE_CONDITION_ACTIVATION,
    YVEX_TRANSFORMER_JOINT_STAGE_FINAL_TIME_ACTIVATION,
    YVEX_TRANSFORMER_JOINT_STAGE_FINAL_ADALN,
    YVEX_TRANSFORMER_JOINT_STAGE_FINAL_NORM,
    YVEX_TRANSFORMER_JOINT_STAGE_FINAL_HIDDEN,
    YVEX_TRANSFORMER_JOINT_STAGE_COUNT
} yvex_transformer_joint_stage;

typedef enum {
    YVEX_TRANSFORMER_JOINT_SCOPE_OMNI = 0,
    YVEX_TRANSFORMER_JOINT_SCOPE_REFINER,
    YVEX_TRANSFORMER_JOINT_SCOPE_COUNT
} yvex_transformer_joint_scope;

typedef struct yvex_transformer_joint_stage_observation {
    unsigned long long block, rows, width, value_count;
    yvex_transformer_joint_scope scope;
    yvex_transformer_joint_stage stage;
    const float *values;
} yvex_transformer_joint_stage_observation;

/* Observation values are borrowed for the callback only. A callback failure aborts the
   enclosing output transaction before any result becomes visible. */
typedef int (*yvex_transformer_joint_block_observer_fn)(
    void *context, const yvex_transformer_joint_block_observation *observation,
    yvex_error *err);
typedef int (*yvex_transformer_joint_stage_observer_fn)(
    void *context, const yvex_transformer_joint_stage_observation *observation,
    yvex_error *err);

typedef struct yvex_transformer_joint_block_options {
    const float *inv_freq;
    yvex_transformer_joint_block_observer_fn block_observer;
    void *block_observer_context;
    yvex_transformer_joint_stage_observer_fn stage_observer;
    void *stage_observer_context;
    unsigned long long observed_stage_block;
    yvex_transformer_joint_scope observed_stage_scope;
    yvex_transformer_joint_stage observed_stage;
} yvex_transformer_joint_block_options;

typedef struct yvex_transformer_joint_request {
    const yvex_transformer_joint_recipe *recipe;
    const float *video, *audio, *conditioning, *timesteps, *position_ids;
    const unsigned int *video_indices, *audio_indices, *text_indices;
    const unsigned int *timestep_indices, *token_tags;
    unsigned long long video_rows, audio_rows, text_rows, timestep_count, packed_rows;
    unsigned long long block_count;
    float *video_output, *audio_output;
    unsigned long long video_output_capacity, audio_output_capacity;
    yvex_transformer_joint_block_observer_fn block_observer;
    void *block_observer_context;
    yvex_transformer_joint_stage_observer_fn stage_observer;
    void *stage_observer_context;
    yvex_transformer_joint_scope observed_stage_scope;
    unsigned long long observed_stage_block;
    yvex_transformer_joint_stage observed_stage;
} yvex_transformer_joint_request;

typedef struct yvex_transformer_joint_result {
    unsigned long long video_rows, audio_rows, text_rows, packed_rows, block_count;
    unsigned long long resident_bytes, kernel_launches, h2d_bytes, d2h_bytes, device_bytes;
    char residency_identity[65], execution_identity[65];
    int complete;
} yvex_transformer_joint_result;

int yvex_backend_transformer_joint_blocks_cuda(
    yvex_backend *backend, const yvex_transformer_joint_recipe *recipe,
    const yvex_transformer_joint_encoded_weight *weights, unsigned long long block_count,
    const char *residency_identity, unsigned long long resident_bytes,
    const float *hidden, const float *temb, unsigned long long timestep_count,
    const float *position_ids, const unsigned int *adaln_indices,
    unsigned long long packed_rows, float *output, unsigned long long output_capacity,
    yvex_transformer_joint_block_result *result,
    const yvex_transformer_joint_block_options *options, yvex_error *err);
int yvex_backend_transformer_joint_cuda(
    yvex_backend *backend, const yvex_transformer_joint_encoded_weight *external_weights,
    const yvex_transformer_joint_encoded_weight *block_weights,
    const char *residency_identity, unsigned long long resident_bytes,
    const yvex_transformer_joint_request *request,
    yvex_transformer_joint_result *result, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
