/* Typed recipe and execution boundary for a joint-modality CUDA Transformer. */
#ifndef INCLUDE_YVEX_INTERNAL_JOINT_TRANSFORMER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_JOINT_TRANSFORMER_H_INCLUDED

#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_backend yvex_backend;
#define YVEX_TRANSFORMER_JOINT_SCHEMA_V1 1u
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

typedef struct yvex_transformer_joint_recipe {
    unsigned int schema_version;
    const char *identity_domain;
    unsigned long long hidden_width, attention_heads, head_dimension, attention_width;
    unsigned long long ffn_width, timestep_width, rotary_width;
    unsigned long long modality_count, modulation_parameters;
    unsigned long long block_count, refiner_block_count;
    unsigned long long maximum_timesteps, maximum_packed_rows;
    unsigned long long video_input_width, audio_input_width, condition_input_width;
    unsigned long long video_output_width, audio_output_width;
} yvex_transformer_joint_recipe;

typedef struct yvex_transformer_joint_block_result {
    unsigned long long packed_rows, block_count, resident_bytes, kernel_launches;
    unsigned long long h2d_bytes, d2h_bytes, device_bytes, temporary_bytes;
    char residency_identity[65], execution_identity[65];
    int complete;
} yvex_transformer_joint_block_result;

typedef struct yvex_transformer_joint_request {
    const yvex_transformer_joint_recipe *recipe;
    const float *video, *audio, *conditioning, *timesteps, *position_ids;
    const unsigned int *video_indices, *audio_indices, *text_indices;
    const unsigned int *timestep_indices, *token_tags;
    unsigned long long video_rows, audio_rows, text_rows, timestep_count, packed_rows;
    unsigned long long block_count;
    float *video_output, *audio_output;
    unsigned long long video_output_capacity, audio_output_capacity;
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
    yvex_transformer_joint_block_result *result, yvex_error *err);
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
