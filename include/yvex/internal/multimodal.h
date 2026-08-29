/* Execute admitted visual encoders and inject their typed outputs into a language stack. */
#ifndef INCLUDE_YVEX_INTERNAL_MULTIMODAL_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MULTIMODAL_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/component.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_VISION_RECIPE_SCHEMA_V1 1u
#define YVEX_VISION_BLOCK_WEIGHT_COUNT 12u
#define YVEX_VISION_MERGER_WEIGHT_COUNT 6u
#define YVEX_VISION_DEEPSTACK_CAP 3u

typedef enum {
    YVEX_VISION_WEIGHT_EXTERNAL = 0,
    YVEX_VISION_WEIGHT_BLOCK,
    YVEX_VISION_WEIGHT_MERGER,
    YVEX_VISION_WEIGHT_DEEPSTACK
} yvex_vision_weight_group;

typedef enum {
    YVEX_VISION_PATCH_WEIGHT = 0,
    YVEX_VISION_PATCH_BIAS,
    YVEX_VISION_POSITION_WEIGHT,
    YVEX_VISION_EXTERNAL_WEIGHT_COUNT
} yvex_vision_external_weight_slot;

typedef enum {
    YVEX_VISION_NORM1_WEIGHT = 0,
    YVEX_VISION_NORM1_BIAS,
    YVEX_VISION_QKV_WEIGHT,
    YVEX_VISION_QKV_BIAS,
    YVEX_VISION_ATTENTION_WEIGHT,
    YVEX_VISION_ATTENTION_BIAS,
    YVEX_VISION_NORM2_WEIGHT,
    YVEX_VISION_NORM2_BIAS,
    YVEX_VISION_FF1_WEIGHT,
    YVEX_VISION_FF1_BIAS,
    YVEX_VISION_FF2_WEIGHT,
    YVEX_VISION_FF2_BIAS
} yvex_vision_block_weight_slot;

typedef enum {
    YVEX_VISION_MERGER_NORM_WEIGHT = 0,
    YVEX_VISION_MERGER_NORM_BIAS,
    YVEX_VISION_MERGER_FC1_WEIGHT,
    YVEX_VISION_MERGER_FC1_BIAS,
    YVEX_VISION_MERGER_FC2_WEIGHT,
    YVEX_VISION_MERGER_FC2_BIAS
} yvex_vision_merger_weight_slot;

typedef enum {
    YVEX_VISION_OBSERVE_PATCH = 1,
    YVEX_VISION_OBSERVE_POSITION,
    YVEX_VISION_OBSERVE_BLOCK,
    YVEX_VISION_OBSERVE_NORM1,
    YVEX_VISION_OBSERVE_QKV,
    YVEX_VISION_OBSERVE_QUERY,
    YVEX_VISION_OBSERVE_KEY,
    YVEX_VISION_OBSERVE_ATTENTION,
    YVEX_VISION_OBSERVE_ATTENTION_PROJECTION,
    YVEX_VISION_OBSERVE_NORM2,
    YVEX_VISION_OBSERVE_FF1,
    YVEX_VISION_OBSERVE_GELU,
    YVEX_VISION_OBSERVE_FF2
} yvex_vision_observation_stage;

typedef int (*yvex_vision_observer_fn)(
    void *, unsigned int, unsigned long long,
    const float *, unsigned long long, unsigned long long, yvex_error *);

typedef struct {
    unsigned int schema_version;
    const char *semantic_identity;
    unsigned long long patch_channels, temporal_patch, patch_height, patch_width;
    unsigned long long position_grid_side, hidden_width, ffn_width;
    unsigned long long heads, head_dimension, layer_count, merge;
    unsigned long long output_width, deepstack_layer_count;
    unsigned long long deepstack_layers[YVEX_VISION_DEEPSTACK_CAP];
    unsigned long long rope_theta;
    float normalization_epsilon;
} yvex_vision_recipe;

typedef int (*yvex_vision_weight_name_fn)(
    void *, int, unsigned long long, unsigned int, char[256], yvex_error *);

typedef struct {
    const yvex_vision_recipe *recipe;
    const float *patches;
    unsigned long long patch_rows, patch_capacity, image_count, grid_height, grid_width;
    yvex_vision_weight_name_fn weight_name;
    void *weight_name_context;
    float *merged, *deepstack;
    unsigned long long merged_capacity, deepstack_capacity;
    yvex_vision_observer_fn observe;
    void *observer_context;
    int (*cancel_requested)(void *);
    void *cancel_context;
} yvex_vision_request;

typedef struct {
    unsigned long long patch_rows, merged_rows, hidden_width, output_width, layer_count;
    unsigned long long kernel_launches, h2d_bytes, d2h_bytes, device_bytes;
    char residency_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_vision_result;

typedef struct {
    const yvex_vision_request *request;
    const yvex_component_encoded_weight *weights;
    unsigned long long weight_count;
    const char *residency_identity;
    unsigned long long resident_bytes;
} yvex_backend_vision_request;

typedef struct {
    const yvex_component_text_recipe *recipe;
    const char *embedding_weight_name;
    yvex_component_text_weight_name_fn layer_weight_name;
    void *weight_name_context;
    const unsigned int *token_ids;
    unsigned long long token_count, layer_count;
    const yvex_backend_text_multimodal_input *multimodal;
    float *output;
    unsigned long long output_capacity;
} yvex_component_multimodal_text_request;

int yvex_backend_vision_execute(
    yvex_backend *, const yvex_backend_vision_request *, yvex_vision_result *, yvex_error *);
int yvex_component_vision_execute(
    const yvex_component_execution *, const yvex_vision_request *,
    yvex_vision_result *, yvex_error *);
int yvex_component_multimodal_text_execute(
    const yvex_component_execution *, const yvex_component_multimodal_text_request *,
    yvex_runtime_av_conditioning_result *, yvex_error *);

#ifdef __cplusplus
}
#endif
#endif
