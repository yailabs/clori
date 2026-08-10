/* Execute bounded paired latent evolution without owning family schedules or model evaluation. */
#ifndef INCLUDE_YVEX_INTERNAL_LATENT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_LATENT_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/sampling.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_LATENT_SCHEMA_V1 1u
#define YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1 1u

typedef int (*yvex_runtime_latent_evaluate_fn)(
    void *context, const float *video_state, unsigned long long video_values,
    const float *audio_state, unsigned long long audio_values,
    float video_timestep, float audio_timestep,
    float *video_velocity, float *audio_velocity, yvex_error *err);
typedef int (*yvex_runtime_latent_advance_fn)(
    float *output, const float *sample, const float *velocity,
    unsigned long long values, float timestep, float sigma, float sigma_next,
    yvex_error *err);

typedef struct yvex_runtime_latent_request {
    unsigned int schema_version;
    unsigned long long seed, video_values, audio_values, step_count;
    unsigned long long maximum_workspace_bytes;
    const float *video_sigmas, *audio_sigmas;
    const char *plan_identity;
    yvex_runtime_latent_evaluate_fn evaluate;
    yvex_runtime_latent_advance_fn advance;
    void *execution_context;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_runtime_latent_request;

typedef struct yvex_runtime_latent_result {
    unsigned int schema_version;
    int completed;
    unsigned long long video_values, audio_values, completed_steps;
    unsigned long long model_evaluations, peak_workspace_bytes;
    yvex_runtime_sampling_normal_result initialization;
    char initial_state_identity[YVEX_SHA256_HEX_CAP];
    char final_state_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_latent_result;

typedef struct yvex_runtime_av_layout_request {
    unsigned int schema_version;
    unsigned int text_tag, audio_tag, video_tag;
    unsigned long long text_rows, audio_steps, audio_channels;
    unsigned long long video_frames, latent_height, latent_width, patch_height, patch_width;
    unsigned long long text_start, audio_start, video_start, packed_rows;
    const unsigned long long *audio_width_indices;
    const unsigned int *temporal_pattern;
    unsigned long long temporal_pattern_count, maximum_workspace_bytes;
    double temporal_scale, spatial_scale, media_time_origin;
    const char *plan_identity;
} yvex_runtime_av_layout_request;

typedef struct yvex_runtime_av_layout_result {
    unsigned int schema_version;
    int complete;
    unsigned long long text_rows, audio_rows, video_rows, packed_rows, workspace_bytes;
    char layout_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_av_layout_result;

typedef struct yvex_runtime_av_layout_output {
    float *position_ids;
    unsigned long long position_capacity;
    unsigned int *token_tags, *video_indices, *audio_indices, *text_indices;
    unsigned long long tag_capacity, video_capacity, audio_capacity, text_capacity;
} yvex_runtime_av_layout_output;

int yvex_runtime_latent_execute(
    const yvex_runtime_latent_request *request,
    float *video_output, unsigned long long video_capacity,
    float *audio_output, unsigned long long audio_capacity,
    yvex_runtime_latent_result *result, yvex_error *err);
int yvex_runtime_latent_shifted_sigmas(
    float *output, unsigned int points, float shift, yvex_error *err);
int yvex_runtime_latent_plan_identity(
    const char *domain, const char *target, const char *source_revision,
    const unsigned long long *facts, unsigned long long fact_count,
    const float *video_sigmas, const float *audio_sigmas, unsigned int points,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err);
int yvex_runtime_av_layout_build(
    const yvex_runtime_av_layout_request *request,
    const yvex_runtime_av_layout_output *output,
    yvex_runtime_av_layout_result *result, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
