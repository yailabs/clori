/* CUDA component operations shared by capability publication, implementations, and oracles. */
#ifndef SRC_BACKEND_CUDA_COMPONENT_OPS_H_INCLUDED
#define SRC_BACKEND_CUDA_COMPONENT_OPS_H_INCLUDED

#include <yvex/internal/component.h>
#include <yvex/internal/convolution.h>
#include <yvex/internal/joint_transformer.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_cuda_execution_arena yvex_cuda_execution_arena;
typedef struct {
    const yvex_backend_tensor_desc *device;
    const unsigned long long *host_bytes;
    unsigned int device_count, host_count;
} yvex_cuda_execution_arena_plan;
typedef struct {
    unsigned long long device_bytes, host_bytes, allocation_count;
    unsigned int device_region_count, host_region_count;
} yvex_cuda_execution_arena_summary;
int yvex_cuda_execution_arena_open(
    yvex_cuda_execution_arena **, yvex_backend *,
    const yvex_cuda_execution_arena_plan *,
    yvex_cuda_execution_arena_summary *, yvex_error *);
yvex_device_tensor *yvex_cuda_execution_arena_device(
    yvex_cuda_execution_arena *, unsigned int);
yvex_device_tensor *yvex_cuda_execution_arena_device_bind(
    yvex_cuda_execution_arena *, unsigned int,
    const yvex_backend_tensor_desc *, yvex_error *);
void *yvex_cuda_execution_arena_host(
    yvex_cuda_execution_arena *, unsigned int);
int yvex_cuda_execution_arena_close(
    yvex_cuda_execution_arena **, yvex_error *);

typedef enum {
    YVEX_CUDA_JOINT_HOST_VIDEO_EMBED = 0,
    YVEX_CUDA_JOINT_HOST_AUDIO_EMBED,
    YVEX_CUDA_JOINT_HOST_TEXT_EMBED,
    YVEX_CUDA_JOINT_HOST_TEXT_REFINED,
    YVEX_CUDA_JOINT_HOST_PACKED,
    YVEX_CUDA_JOINT_HOST_BLOCK_OUTPUT,
    YVEX_CUDA_JOINT_HOST_BLOCK_STAGED,
    YVEX_CUDA_JOINT_HOST_TIME_EMBED,
    YVEX_CUDA_JOINT_HOST_NORMALIZED,
    YVEX_CUDA_JOINT_HOST_ALL_VIDEO,
    YVEX_CUDA_JOINT_HOST_ALL_AUDIO,
    YVEX_CUDA_JOINT_HOST_STAGED_VIDEO,
    YVEX_CUDA_JOINT_HOST_STAGED_AUDIO,
    YVEX_CUDA_JOINT_HOST_ADALN,
    YVEX_CUDA_JOINT_HOST_ROPE_COSINE,
    YVEX_CUDA_JOINT_HOST_ROPE_SINE,
    YVEX_CUDA_JOINT_HOST_COUNT
} yvex_cuda_joint_host_slot;

#define YVEX_CUDA_JOINT_DEVICE_REGION_COUNT 19u

struct yvex_transformer_joint_prepared {
    yvex_backend *backend;
    yvex_cuda_execution_arena *arena;
    yvex_transformer_joint_prepared_summary summary;
    yvex_transformer_linear_executable *linear[YVEX_TRANSFORMER_JOINT_LINEAR_COUNT];
    char residency_identity[YVEX_SHA256_HEX_CAP];
    char layout_identity[YVEX_SHA256_HEX_CAP];
    char condition_identity[YVEX_SHA256_HEX_CAP];
    int in_use;
};

int yvex_cuda_joint_recipe_supported(
    const yvex_transformer_joint_recipe *);
int yvex_cuda_joint_weights_validate(
    const yvex_transformer_joint_encoded_weight *, unsigned long long,
    unsigned long long *);
int yvex_cuda_joint_request_valid(
    const yvex_transformer_joint_request *, unsigned long long *,
    unsigned long long *, yvex_error *);
int yvex_cuda_joint_external_valid(
    const yvex_transformer_joint_encoded_weight *, unsigned long long *);
int yvex_cuda_joint_device_plan(
    const yvex_transformer_joint_request *, yvex_backend_tensor_desc *,
    unsigned int, yvex_error *);
int yvex_cuda_joint_prepare_invariants(
    yvex_transformer_joint_prepared *, yvex_backend *,
    const yvex_transformer_joint_encoded_weight *,
    const yvex_transformer_joint_request *, yvex_error *);
int yvex_cuda_joint_dense_plan_execute(
    yvex_transformer_joint_prepared *, yvex_transformer_joint_weight_slot,
    const yvex_transformer_joint_encoded_weight *, const yvex_device_tensor *,
    yvex_device_tensor *, yvex_transformer_joint_block_result *, int *, int *,
    yvex_error *);
int yvex_cuda_joint_execution_identity(
    const yvex_transformer_joint_request *, const char *, const char *,
    const float *, unsigned long long, const float *, unsigned long long,
    char[YVEX_SHA256_HEX_CAP]);

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
int yvex_cuda_text_encoder_multimodal_execute(
    yvex_backend *backend, const yvex_component_text_recipe *geometry,
    const yvex_backend_text_weight *weights, unsigned long long layer_count,
    const char *residency_identity, unsigned long long resident_bytes,
    const unsigned int *token_ids, unsigned long long token_count,
    const yvex_backend_text_multimodal_input *multimodal, float *output,
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
int yvex_cuda_transformer_joint_prepare(
    yvex_backend *backend, const yvex_transformer_joint_encoded_weight *external_weights,
    const yvex_transformer_joint_encoded_weight *block_weights,
    const char *residency_identity, unsigned long long resident_bytes,
    const char *prepared_identity, const yvex_transformer_joint_request *request,
    yvex_transformer_joint_prepared **prepared,
    yvex_transformer_joint_prepared_summary *summary, yvex_error *err);
int yvex_cuda_transformer_joint_prepared_execute(
    yvex_backend *backend, const yvex_transformer_joint_encoded_weight *external_weights,
    const yvex_transformer_joint_encoded_weight *block_weights,
    const char *residency_identity, unsigned long long resident_bytes,
    yvex_transformer_joint_prepared *prepared,
    const yvex_transformer_joint_request *request,
    yvex_transformer_joint_result *result, yvex_error *err);
int yvex_cuda_transformer_joint_prepared_release(
    yvex_backend *backend, yvex_transformer_joint_prepared **prepared,
    yvex_error *err);
int yvex_cuda_alias_decoder_execute(
    yvex_backend *backend, const yvex_alias_decoder_request *request,
    yvex_alias_decoder_result *result, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif /* SRC_BACKEND_CUDA_COMPONENT_OPS_H_INCLUDED */
