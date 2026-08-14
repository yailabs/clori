/* Generic one-dimensional decoder execution over admitted resident F32 weights. */
#ifndef INCLUDE_YVEX_INTERNAL_CONVOLUTION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_CONVOLUTION_H_INCLUDED

#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_backend yvex_backend;
typedef struct yvex_component_encoded_weight yvex_convolution_weight;
typedef struct yvex_component_execution_failure yvex_component_execution_failure;
typedef struct yvex_materialization_session yvex_materialization_session;

#define YVEX_ALIAS_DECODER_MAX_STAGES 8u
#define YVEX_ALIAS_DECODER_MAX_RESBLOCKS 4u
#define YVEX_ALIAS_DECODER_MAX_LAYERS 4u

typedef struct {
    unsigned long long batch, input_channels, output_channels, input_length;
    unsigned long long kernel_size, stride, dilation, padding, output_padding;
    int transposed;
} yvex_convolution_1d_geometry;
typedef yvex_convolution_1d_geometry yvex_graph_conv1d_geometry;

typedef enum {
    YVEX_ALIAS_DECODER_INPUT_WEIGHT = 0,
    YVEX_ALIAS_DECODER_INPUT_BIAS,
    YVEX_ALIAS_DECODER_PRE_WEIGHT,
    YVEX_ALIAS_DECODER_PRE_GAIN,
    YVEX_ALIAS_DECODER_PRE_BIAS,
    YVEX_ALIAS_DECODER_UP_WEIGHT,
    YVEX_ALIAS_DECODER_UP_GAIN,
    YVEX_ALIAS_DECODER_UP_BIAS,
    YVEX_ALIAS_DECODER_ACT_ALPHA,
    YVEX_ALIAS_DECODER_ACT_BETA,
    YVEX_ALIAS_DECODER_ACT_UP_FILTER,
    YVEX_ALIAS_DECODER_ACT_DOWN_FILTER,
    YVEX_ALIAS_DECODER_RES1_WEIGHT,
    YVEX_ALIAS_DECODER_RES1_GAIN,
    YVEX_ALIAS_DECODER_RES1_BIAS,
    YVEX_ALIAS_DECODER_RES2_WEIGHT,
    YVEX_ALIAS_DECODER_RES2_GAIN,
    YVEX_ALIAS_DECODER_RES2_BIAS,
    YVEX_ALIAS_DECODER_POST_ACT_ALPHA,
    YVEX_ALIAS_DECODER_POST_ACT_BETA,
    YVEX_ALIAS_DECODER_POST_ACT_UP_FILTER,
    YVEX_ALIAS_DECODER_POST_ACT_DOWN_FILTER,
    YVEX_ALIAS_DECODER_POST_WEIGHT,
    YVEX_ALIAS_DECODER_POST_GAIN,
    YVEX_ALIAS_DECODER_WEIGHT_ROLE_COUNT
} yvex_alias_decoder_weight_role;

typedef struct {
    const char *input_projection, *pre_convolution, *upsamples;
    const char *residual_blocks, *post_activation, *post_convolution;
} yvex_alias_decoder_name_templates;

typedef int (*yvex_alias_decoder_weight_name_fn)(
    void *context, yvex_alias_decoder_weight_role role,
    unsigned long long stage, unsigned long long block,
    unsigned long long layer, char output[256], yvex_error *err);
typedef int (*yvex_alias_decoder_weight_bind_fn)(
    void *context, const char *name, unsigned long long rows,
    unsigned long long width, yvex_convolution_weight *weight,
    yvex_error *err);

typedef struct {
    unsigned long long input_channels, projection_channels, input_kernel;
    unsigned long long pre_channels, pre_kernel;
    unsigned long long stage_count, residual_blocks, residual_layers;
    unsigned long long rates[YVEX_ALIAS_DECODER_MAX_STAGES];
    unsigned long long upsample_kernels[YVEX_ALIAS_DECODER_MAX_STAGES];
    unsigned long long residual_kernels[YVEX_ALIAS_DECODER_MAX_RESBLOCKS];
    unsigned long long residual_dilations[YVEX_ALIAS_DECODER_MAX_LAYERS];
    unsigned long long final_channels, final_kernel;
} yvex_alias_decoder_recipe;

typedef struct {
    const yvex_alias_decoder_recipe *recipe;
    const float *input;
    unsigned long long batch, input_length, input_count;
    float *output;
    unsigned long long output_capacity, maximum_workspace_bytes;
    yvex_alias_decoder_weight_name_fn weight_name;
    yvex_alias_decoder_weight_bind_fn weight_bind;
    void *weight_name_context, *weight_bind_context;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_alias_decoder_request;

typedef struct {
    unsigned long long output_length, output_values, weight_bindings;
    unsigned long long kernel_launches, h2d_bytes, d2h_bytes, peak_device_bytes;
    unsigned long long tensor_reads, payload_bytes_read, peak_host_bytes;
    int complete;
} yvex_alias_decoder_result;

int yvex_backend_alias_decoder_execute(
    yvex_backend *backend, const yvex_alias_decoder_request *request,
    yvex_alias_decoder_result *result, yvex_error *err);
int yvex_runtime_alias_decoder_execute_cpu(
    yvex_materialization_session *session, const yvex_alias_decoder_request *request,
    yvex_alias_decoder_result *result, yvex_component_execution_failure *failure,
    yvex_error *err);
int yvex_graph_conv1d_output_length(
    const yvex_graph_conv1d_geometry *geometry, unsigned long long *output_length,
    yvex_error *err);
int yvex_graph_conv1d_f32(
    const yvex_graph_conv1d_geometry *geometry, const float *input,
    unsigned long long input_count, const float *weight, unsigned long long weight_count,
    const float *bias, unsigned long long bias_count, const float *gain,
    unsigned long long gain_count, float *output, unsigned long long output_count,
    yvex_error *err);
int yvex_graph_alias_snake_f32(
    const float *input, unsigned long long batch, unsigned long long channels,
    unsigned long long length, const float *alpha_log, const float *beta_log,
    const float up_filter[12], const float down_filter[12], float *output,
    float *scratch, unsigned long long scratch_count, yvex_error *err);
int yvex_alias_decoder_template_name(
    void *context, yvex_alias_decoder_weight_role role,
    unsigned long long stage, unsigned long long block,
    unsigned long long layer, char output[256], yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_CONVOLUTION_H_INCLUDED */
