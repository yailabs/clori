/* CUDA component operations shared by capability publication, implementations, and oracles. */
#ifndef SRC_BACKEND_CUDA_COMPONENT_OPS_H_INCLUDED
#define SRC_BACKEND_CUDA_COMPONENT_OPS_H_INCLUDED

#include <yvex/internal/component.h>
#include <yvex/internal/convolution.h>
#include <yvex/internal/joint_transformer.h>

#ifdef __cplusplus
extern "C" {
#endif

const yvex_backend_component_operations *yvex_cuda_component_operations_get(
    const yvex_backend *backend);
int yvex_cuda_text_embedding_execute(
    yvex_backend *backend, const yvex_component_text_recipe *geometry,
    const unsigned char *encoded, unsigned long long encoded_bytes,
    unsigned int qtype, unsigned long long row_count, unsigned long long row_width,
    unsigned long long row_bytes, const char *residency_identity,
    unsigned long long resident_bytes, const unsigned int *token_ids,
    unsigned long long token_count, float *output, unsigned long long output_capacity,
    yvex_backend_text_execution_result *result, yvex_error *err);
int yvex_cuda_text_encoder_execute(
    yvex_backend *backend, const yvex_component_text_recipe *geometry,
    const yvex_backend_text_weight *weights, unsigned long long layer_count,
    const char *residency_identity, unsigned long long resident_bytes,
    const unsigned int *token_ids, unsigned long long token_count, float *output,
    unsigned long long output_capacity, yvex_backend_text_execution_result *result,
    yvex_error *err);
int yvex_cuda_transformer_joint_blocks_execute(
    yvex_backend *backend, const yvex_transformer_joint_recipe *recipe,
    const yvex_transformer_joint_encoded_weight *weights, unsigned long long block_count,
    const char *residency_identity, unsigned long long resident_bytes,
    const float *hidden, const float *temb, unsigned long long timestep_count,
    const float *position_ids, const unsigned int *adaln_indices,
    unsigned long long packed_rows, float *output, unsigned long long output_capacity,
    yvex_transformer_joint_block_result *result,
    const yvex_transformer_joint_block_options *options, yvex_error *err);
int yvex_cuda_transformer_joint_execute(
    yvex_backend *backend, const yvex_transformer_joint_encoded_weight *external_weights,
    const yvex_transformer_joint_encoded_weight *block_weights,
    const char *residency_identity, unsigned long long resident_bytes,
    const yvex_transformer_joint_request *request,
    yvex_transformer_joint_result *result, yvex_error *err);
int yvex_cuda_alias_decoder_execute(
    yvex_backend *backend, const yvex_alias_decoder_request *request,
    yvex_alias_decoder_result *result, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif /* SRC_BACKEND_CUDA_COMPONENT_OPS_H_INCLUDED */
