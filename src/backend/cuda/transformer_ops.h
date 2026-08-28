/* CUDA transformer primitives shared only by CUDA implementation owners. */
#ifndef SRC_BACKEND_CUDA_TRANSFORMER_OPS_H_INCLUDED
#define SRC_BACKEND_CUDA_TRANSFORMER_OPS_H_INCLUDED

#include <yvex/internal/transformer.h>

#ifdef __cplusplus
extern "C" {
#endif

const yvex_backend_transformer_operations *yvex_cuda_transformer_operations_get(
    const yvex_backend *backend);
int yvex_cuda_transformer_initial(
    yvex_backend *backend, const yvex_device_tensor *encoded, unsigned int qtype,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, yvex_device_tensor *embedding,
    yvex_device_tensor *expanded, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err);
int yvex_cuda_transformer_feature_mean(
    yvex_backend *backend, const yvex_device_tensor *expanded,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, yvex_device_tensor *device_output,
    yvex_device_tensor *resident_output, unsigned long long resident_row_offset,
    unsigned long long resident_row_stride, unsigned long long resident_column_offset,
    float *host_output, yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_final(
    yvex_backend *backend, const yvex_device_tensor *expanded,
    const yvex_device_tensor *function, const yvex_device_tensor *base,
    const yvex_device_tensor *scale, const yvex_device_tensor *norm,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, double epsilon, double mhc_epsilon,
    yvex_device_tensor *pre_normalized, yvex_device_tensor *output,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_gqa_workspace_required(
    unsigned long long tokens, unsigned long long query_heads,
    unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long *bytes, yvex_error *err);
int yvex_cuda_transformer_dense_decoder_execute(
    yvex_backend *backend, const yvex_transformer_dense_decoder_request *request,
    yvex_transformer_dense_decoder_result *result, yvex_error *err);
int yvex_cuda_transformer_rotary_half(
    yvex_backend *backend, yvex_device_tensor *values,
    const yvex_device_tensor *cosines, const yvex_device_tensor *sines,
    unsigned long long tokens, unsigned long long heads, unsigned long long head_dim,
    unsigned long long rotary_dim, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err);
int yvex_cuda_transformer_rotary_half_f32(
    yvex_backend *backend, yvex_device_tensor *values,
    const yvex_device_tensor *cosines, const yvex_device_tensor *sines,
    unsigned long long tokens, unsigned long long heads, unsigned long long head_dim,
    unsigned long long rotary_dim, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err);
int yvex_cuda_transformer_gqa(
    yvex_backend *backend, const yvex_device_tensor *query,
    const yvex_device_tensor *key, const yvex_device_tensor *value,
    yvex_device_tensor *output, unsigned long long tokens,
    unsigned long long query_heads, unsigned long long kv_heads,
    unsigned long long head_dim, int causal,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_silu_product_bf16(
    yvex_backend *backend, const yvex_device_tensor *gate,
    const yvex_device_tensor *up, yvex_device_tensor *output,
    unsigned long long count, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err);
int yvex_cuda_transformer_silu(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *output, unsigned long long count, int bf16_output,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_timestep_embedding(
    yvex_backend *backend, const yvex_device_tensor *timesteps,
    yvex_device_tensor *output, unsigned long long rows,
    unsigned long long half_width, float maximum_period,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_split_three(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *first, yvex_device_tensor *second,
    yvex_device_tensor *third, unsigned long long rows, unsigned long long width,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_split_interleaved_three(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *first, yvex_device_tensor *second,
    yvex_device_tensor *third, unsigned long long rows, unsigned long long heads,
    unsigned long long head_dim, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err);
int yvex_cuda_transformer_swiglu_split_bf16(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *output, unsigned long long rows, unsigned long long width,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_swiglu_split_f32(
    yvex_backend *backend, const yvex_device_tensor *input,
    yvex_device_tensor *output, unsigned long long rows, unsigned long long width,
    int gate_first, yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_modulate_bf16(
    yvex_backend *backend, const yvex_device_tensor *input,
    const yvex_device_tensor *table, const unsigned int *row_indices,
    yvex_device_tensor *output, unsigned long long rows, unsigned long long width,
    unsigned long long table_rows, unsigned long long parameters,
    unsigned int shift_slot, unsigned int scale_slot,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_gated_residual_bf16(
    yvex_backend *backend, const yvex_device_tensor *residual,
    const yvex_device_tensor *table, const unsigned int *row_indices,
    const yvex_device_tensor *update, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width,
    unsigned long long table_rows, unsigned long long parameters,
    unsigned int gate_slot, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err);
int yvex_cuda_transformer_bias(
    yvex_backend *backend, const yvex_device_tensor *input,
    const yvex_device_tensor *bias, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width, int bf16_output,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_add_bf16(
    yvex_backend *backend, const yvex_device_tensor *left,
    const yvex_device_tensor *right, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_scaled_residual_f32(
    yvex_backend *backend, const yvex_device_tensor *residual,
    const yvex_device_tensor *update, const yvex_device_tensor *scale,
    yvex_device_tensor *output, unsigned long long rows, unsigned long long width,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_layer_norm_f32(
    yvex_backend *backend, const yvex_device_tensor *input,
    const yvex_device_tensor *weight, const yvex_device_tensor *bias,
    yvex_device_tensor *output, unsigned long long rows, unsigned long long width,
    float epsilon, yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_bf16_round(
    yvex_backend *backend, yvex_device_tensor *values, unsigned long long count,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_cuda_transformer_rms_norm_bf16(
    yvex_backend *backend, const yvex_device_tensor *input,
    const yvex_device_tensor *weight, yvex_device_tensor *output,
    unsigned long long rows, unsigned long long width, float epsilon,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
