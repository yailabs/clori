/* Bounded host buffers and exact resident weight views for admitted component execution. */
#ifndef INCLUDE_YVEX_INTERNAL_COMPONENT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_COMPONENT_H_INCLUDED

#include <yvex/backend.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/latent.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_runtime_residency yvex_runtime_residency;
typedef struct yvex_runtime_component_session yvex_runtime_component_session;
typedef struct yvex_runtime_av_layout_output yvex_runtime_av_layout_output;
typedef struct yvex_runtime_av_layout_result yvex_runtime_av_layout_result;
typedef struct yvex_transformer_linear_physical_plan yvex_transformer_linear_physical_plan;
typedef struct yvex_transformer_joint_request yvex_transformer_joint_request;
typedef struct yvex_transformer_joint_result yvex_transformer_joint_result;

typedef struct {
    float *data;
    unsigned long long count;
} yvex_component_f32_buffer;

typedef struct yvex_component_encoded_weight {
    const unsigned char *encoded;
    unsigned long long encoded_bytes, row_count, row_width, row_bytes;
    unsigned int qtype;
} yvex_component_encoded_weight;

typedef enum {
    YVEX_COMPONENT_LOAD_NONE = 0,
    YVEX_COMPONENT_LOAD_MISSING,
    YVEX_COMPONENT_LOAD_CONTRACT,
    YVEX_COMPONENT_LOAD_BUDGET,
    YVEX_COMPONENT_LOAD_MATERIALIZATION
} yvex_component_load_code;

typedef struct {
    yvex_component_load_code code;
    char tensor_name[256];
    unsigned long long expected, actual;
    const char *reason;
} yvex_component_load_failure;

typedef enum {
    YVEX_COMPONENT_EXECUTION_NONE = 0,
    YVEX_COMPONENT_EXECUTION_INVALID_ARGUMENT,
    YVEX_COMPONENT_EXECUTION_LIFECYCLE,
    YVEX_COMPONENT_EXECUTION_MISSING_TENSOR,
    YVEX_COMPONENT_EXECUTION_TENSOR_CONTRACT,
    YVEX_COMPONENT_EXECUTION_BUDGET,
    YVEX_COMPONENT_EXECUTION_MATERIALIZATION,
    YVEX_COMPONENT_EXECUTION_NUMERIC,
    YVEX_COMPONENT_EXECUTION_CANCELLED
} yvex_component_execution_code;

typedef struct yvex_component_execution_failure {
    unsigned int code;
    char tensor_name[256];
    unsigned long long expected, actual;
    const char *reason;
} yvex_component_execution_failure;

typedef struct yvex_runtime_av_conditioning_result {
    unsigned long long token_count, hidden_width, layer_count;
    unsigned long long resident_bytes, kernel_launches, h2d_bytes, d2h_bytes, device_bytes;
    unsigned long long peak_workspace_bytes;
    char residency_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_conditioning_result;

/* Families supply stable text-stack meaning and canonical weight names. Component runtime owns
 * artifact residency, transactional publication, and backend dispatch for that recipe. */
#define YVEX_COMPONENT_TEXT_RECIPE_SCHEMA_V1 1u
#define YVEX_COMPONENT_TEXT_LAYER_WEIGHT_COUNT 11u
typedef struct yvex_component_text_recipe {
    unsigned int schema_version;
    const char *semantic_identity;
    const char *embedding_identity_domain;
    const char *encoder_identity_domain;
    unsigned long long layer_capacity, hidden_width, ffn_width;
    unsigned long long query_heads, kv_heads, head_dimension;
    unsigned long long vocabulary_size, rope_theta;
    float normalization_epsilon;
} yvex_component_text_recipe;
typedef int (*yvex_component_text_weight_name_fn)(
    void *, unsigned long long, unsigned int, char[256], yvex_error *);
typedef struct {
    const yvex_component_text_recipe *recipe;
    const char *embedding_weight_name;
    yvex_component_text_weight_name_fn layer_weight_name;
    void *weight_name_context;
    const unsigned int *token_ids;
    unsigned long long token_count, layer_count;
    float *output;
    unsigned long long output_capacity, maximum_host_bytes, maximum_device_bytes;
} yvex_component_text_request;
typedef int (*yvex_component_joint_weight_name_fn)(
    void *, unsigned long long, unsigned int, char[256], yvex_error *);

typedef struct yvex_runtime_av_audio_decode_options {
    const float *latent;
    unsigned long long batch, latent_channels, latent_steps;
    float *output;
    unsigned long long output_capacity, max_workspace_bytes;
    int (*cancelled)(void *);
    void *cancellation_context;
} yvex_runtime_av_audio_decode_options;

typedef struct yvex_runtime_av_audio_decode_result {
    unsigned long long batch, samples_per_channel, output_values;
    unsigned long long tensor_reads, payload_bytes_read, peak_workspace_bytes, kernel_launches;
    unsigned long long h2d_bytes, d2h_bytes, device_bytes;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
    char residency_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_audio_decode_result;

typedef struct yvex_runtime_av_video_decode_options {
    const float *latent;
    float *output;
    unsigned long long batch, latent_channels;
    unsigned long long latent_frames, latent_height, latent_width;
    unsigned long long output_capacity, max_workspace_bytes;
    int (*cancelled)(void *);
    void *cancellation_context;
} yvex_runtime_av_video_decode_options;

typedef struct yvex_runtime_av_video_decode_result {
    unsigned long long batch, frames, height, width, output_values;
    unsigned long long tensor_reads, payload_bytes_read, peak_workspace_bytes, kernel_launches;
    unsigned long long h2d_bytes, d2h_bytes, device_bytes;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
    char residency_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_video_decode_result;

typedef struct yvex_runtime_av_latent_context {
    yvex_runtime_component_session *transformer_session;
    const float *conditioning;
    unsigned long long conditioning_capacity;
    const yvex_runtime_av_layout_output *layout;
    const yvex_runtime_av_layout_result *layout_result;
    const yvex_transformer_linear_physical_plan *video_output_specialization;
    const yvex_transformer_linear_physical_plan *audio_output_specialization;
    unsigned int *timestep_indices;
    unsigned long long timestep_capacity, block_count;
    const char *conditioning_identity;
    int (*cancelled)(void *);
    void *cancellation_context;
    yvex_runtime_latent_observe_fn observe;
    void *observer_context;
} yvex_runtime_av_latent_context;

int yvex_component_buffer_open(
    yvex_component_f32_buffer *, unsigned long long, unsigned long long,
    unsigned long long *, unsigned long long *, const char *, const char *, yvex_error *);
void yvex_component_buffer_close(yvex_component_f32_buffer *, unsigned long long *);
int yvex_component_weight_bind_sized(
    void *, const char *, unsigned long long, unsigned long long,
    yvex_component_encoded_weight *, yvex_error *);
int yvex_component_f32_load(
    yvex_materialization_session *, const char *, unsigned int,
    const unsigned long long *, yvex_component_f32_buffer *, unsigned long long,
    unsigned long long *, unsigned long long *, unsigned long long *,
    unsigned long long *, yvex_component_load_failure *, const char *, const char *,
    yvex_error *);
int yvex_runtime_component_session_open(
    yvex_runtime_component_session **, const yvex_complete_artifact_admission *,
    const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *, yvex_backend_kind,
    unsigned long long, unsigned long long, yvex_error *);
int yvex_runtime_component_session_close(yvex_runtime_component_session **, yvex_error *);
int yvex_runtime_component_text_artifact_cuda(
    const yvex_complete_artifact_admission *, const yvex_artifact *, const yvex_gguf *,
    const yvex_tensor_table *, const yvex_component_text_request *,
    yvex_runtime_av_conditioning_result *, yvex_error *);
int yvex_runtime_component_joint_transformer_cuda(
    yvex_runtime_component_session *, const char *const *, unsigned long long,
    yvex_component_joint_weight_name_fn, void *, const yvex_transformer_joint_request *,
    yvex_transformer_joint_result *, yvex_error *);
yvex_materialization_session *yvex_runtime_component_session_materialization(
    const yvex_runtime_component_session *);
yvex_backend *yvex_runtime_component_session_backend(const yvex_runtime_component_session *);
const yvex_runtime_residency_summary *yvex_runtime_component_session_summary(
    const yvex_runtime_component_session *);

#ifdef __cplusplus
}
#endif
#endif
