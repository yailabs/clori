/* Execute bounded paired latent evolution without owning family schedules or model evaluation. */
#ifndef INCLUDE_YVEX_INTERNAL_LATENT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_LATENT_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/core.h>
#include <yvex/internal/sampling.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_LATENT_SCHEMA_V1 1u
#define YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1 1u
#define YVEX_RUNTIME_AV_PLAN_SCHEMA_V1 1u
#define YVEX_RUNTIME_AV_UNPACK_SCHEMA_V1 1u
#define YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1 1u
#define YVEX_RUNTIME_AV_TEMPORAL_PATTERN_CAP 8u
#define YVEX_RUNTIME_AV_TILE_CAP 16u
#define YVEX_RUNTIME_LATENT_EVALUATOR_SCHEMA_V1 1u
#define YVEX_RUNTIME_LATENT_OBSERVATION_SCHEMA_V1 1u

typedef struct {
    unsigned int schema_version, rng_algorithm, rng_version;
    int completed;
    unsigned long long seed, value_count, uniform_draw_count, workspace_bytes;
    char normal_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_latent_normal_result;

typedef struct yvex_runtime_av_plan_policy {
    unsigned int schema_version, maximum_steps;
    unsigned int text_tag, audio_tag, video_tag;
    unsigned long long frame_period, frame_remainder;
    unsigned long long video_latents_per_period, video_latent_remainder;
    unsigned long long spatial_ratio, patch_height, patch_width;
    unsigned long long audio_rate_numerator, audio_rate_denominator, audio_channels;
    unsigned long long video_value_width, audio_value_width;
    unsigned int temporal_pattern[YVEX_RUNTIME_AV_TEMPORAL_PATTERN_CAP];
    unsigned int temporal_pattern_count;
    float video_sigma_shift, audio_sigma_shift;
    double temporal_scale, spatial_scale;
    const char *identity_domain, *target_identity, *source_revision;
} yvex_runtime_av_plan_policy;

typedef struct yvex_runtime_av_plan {
    unsigned int schema_version, sigma_grid_points, model_evaluations;
    unsigned int text_tag, audio_tag, video_tag;
    unsigned long long text_tokens, frames, width, height;
    unsigned long long video_latent_frames, video_latent_height, video_latent_width;
    unsigned long long audio_latent_steps, audio_rows, video_rows, packed_rows;
    unsigned long long patch_height, patch_width, audio_channels;
    unsigned long long video_value_width, audio_value_width;
    unsigned int temporal_pattern[YVEX_RUNTIME_AV_TEMPORAL_PATTERN_CAP];
    unsigned int temporal_pattern_count;
    double temporal_scale, spatial_scale;
    float video_sigmas[65], audio_sigmas[65];
    char identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_plan;

typedef int (*yvex_runtime_latent_evaluate_fn)(
    void *context, const float *video_state, unsigned long long video_values,
    const float *audio_state, unsigned long long audio_values,
    float video_timestep, float audio_timestep,
    float *video_velocity, float *audio_velocity, yvex_error *err);
typedef int (*yvex_runtime_latent_advance_fn)(
    float *output, const float *sample, const float *velocity,
    unsigned long long values, float timestep, float sigma, float sigma_next,
    yvex_error *err);

typedef enum yvex_runtime_latent_observation_stage {
    YVEX_RUNTIME_LATENT_OBSERVATION_INITIAL = 1,
    YVEX_RUNTIME_LATENT_OBSERVATION_EVALUATED = 2,
    YVEX_RUNTIME_LATENT_OBSERVATION_ADVANCED = 3,
    YVEX_RUNTIME_LATENT_OBSERVATION_FINAL = 4
} yvex_runtime_latent_observation_stage;

typedef struct yvex_runtime_latent_observation {
    unsigned int schema_version;
    yvex_runtime_latent_observation_stage stage;
    unsigned long long completed_steps, video_values, audio_values;
    float video_timestep, audio_timestep;
    const float *video_state, *audio_state;
    const float *video_velocity, *audio_velocity;
} yvex_runtime_latent_observation;

/* The view is immutable and borrowed for one synchronous call. Failure aborts latent publication. */
typedef int (*yvex_runtime_latent_observe_fn)(
    void *context, const yvex_runtime_latent_observation *observation, yvex_error *err);

typedef struct yvex_runtime_latent_request {
    unsigned int schema_version;
    unsigned long long seed, video_values, audio_values, step_count;
    unsigned long long maximum_workspace_bytes;
    const float *video_sigmas, *audio_sigmas;
    const char *plan_identity, *evaluator_identity;
    yvex_runtime_latent_evaluate_fn evaluate;
    yvex_runtime_latent_advance_fn advance;
    void *execution_context;
    int (*cancel_requested)(void *context);
    void *cancel_context;
    yvex_runtime_latent_observe_fn observe;
    void *observer_context;
} yvex_runtime_latent_request;

typedef struct yvex_runtime_latent_result {
    unsigned int schema_version;
    int completed;
    unsigned long long video_values, audio_values, completed_steps;
    unsigned long long model_evaluations, peak_workspace_bytes;
    yvex_runtime_latent_normal_result initialization;
    char initial_state_identity[YVEX_SHA256_HEX_CAP];
    char final_state_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_latent_result;

typedef struct yvex_runtime_latent_evaluator_result {
    unsigned int schema_version;
    unsigned long long model_evaluations, kernel_launches, h2d_bytes, d2h_bytes;
    unsigned long long peak_device_bytes;
    char residency_identity[YVEX_SHA256_HEX_CAP];
    char evaluator_identity[YVEX_SHA256_HEX_CAP];
    char execution_chain_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_latent_evaluator_result;

typedef struct yvex_runtime_latent_evaluator_evidence {
    yvex_runtime_latent_evaluator_result staged;
    yvex_sha256 chain;
    int active;
} yvex_runtime_latent_evaluator_evidence;

int yvex_runtime_latent_normal_f32(
    float *, unsigned long long, unsigned long long, unsigned long long,
    unsigned long long, yvex_runtime_latent_normal_result *, yvex_error *);

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

typedef struct yvex_runtime_av_unpack_request {
    unsigned int schema_version;
    const yvex_runtime_av_plan *plan;
    const float *video_rows, *audio_rows;
    unsigned long long video_row_capacity, audio_row_capacity;
    const float *video_channel_mean, *video_channel_std;
    const float *audio_channel_mean, *audio_channel_std;
    unsigned long long video_channel_count, audio_channel_count;
    unsigned long long maximum_workspace_bytes;
    const char *latent_execution_identity;
} yvex_runtime_av_unpack_request;

typedef struct yvex_runtime_av_unpack_output {
    float *video, *audio;
    unsigned long long video_capacity, audio_capacity;
} yvex_runtime_av_unpack_output;

typedef struct yvex_runtime_av_unpack_result {
    unsigned int schema_version;
    unsigned long long video_channels, video_frames, video_height, video_width;
    unsigned long long video_values, audio_batch, audio_channels, audio_steps, audio_values;
    unsigned long long peak_workspace_bytes;
    char input_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_unpack_result;

typedef struct yvex_runtime_av_video_reconstruction_request {
    unsigned int schema_version;
    unsigned long long frames, width, height;
    unsigned long long latent_frames, latent_height, latent_width;
    unsigned long long temporal_ratio, clip_length, token_drop;
    unsigned long long spatial_ratio, tile_size, minimum_tile_overlap;
    const char *source_identity;
} yvex_runtime_av_video_reconstruction_request;

typedef struct yvex_runtime_av_video_reconstruction_plan {
    unsigned int schema_version;
    unsigned long long frames, width, height;
    unsigned long long latent_frames, latent_height, latent_width;
    unsigned long long spatial_ratio;
    unsigned long long tokens_per_chunk, token_overlap, frame_pre_padding;
    unsigned long long frame_overlap, temporal_chunks, decode_latent_frames;
    unsigned long long decode_frames, pad_tokens, tile_y_count, tile_x_count;
    unsigned long long tile_y_start[YVEX_RUNTIME_AV_TILE_CAP];
    unsigned long long tile_y_length[YVEX_RUNTIME_AV_TILE_CAP];
    unsigned long long tile_y_overlap[YVEX_RUNTIME_AV_TILE_CAP - 1u];
    unsigned long long tile_x_start[YVEX_RUNTIME_AV_TILE_CAP];
    unsigned long long tile_x_length[YVEX_RUNTIME_AV_TILE_CAP];
    unsigned long long tile_x_overlap[YVEX_RUNTIME_AV_TILE_CAP - 1u];
    unsigned long long total_decode_calls;
    char identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_video_reconstruction_plan;

typedef struct yvex_runtime_av_video_decode_window {
    const float *latent;
    unsigned long long latent_channels, latent_frames, latent_height, latent_width;
    float *output;
    unsigned long long output_capacity;
} yvex_runtime_av_video_decode_window;

typedef struct yvex_runtime_av_video_decode_evidence {
    unsigned long long output_values, kernel_launches, h2d_bytes, d2h_bytes, device_bytes;
    char execution_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_video_decode_evidence;

typedef int (*yvex_runtime_av_video_decode_fn)(
    void *context, const yvex_runtime_av_video_decode_window *window,
    yvex_runtime_av_video_decode_evidence *evidence, yvex_error *err);

typedef struct yvex_runtime_av_video_reconstruction_execution {
    unsigned int schema_version;
    const yvex_runtime_av_video_reconstruction_plan *plan;
    const float *latent;
    unsigned long long latent_channels, latent_capacity, maximum_workspace_bytes;
    const float *output_channel_mean, *output_channel_std;
    unsigned long long output_channel_count;
    yvex_runtime_av_video_decode_fn decode;
    void *decode_context;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_runtime_av_video_reconstruction_execution;

typedef struct yvex_runtime_av_video_reconstruction_result {
    unsigned int schema_version;
    unsigned long long output_values, decode_calls, peak_workspace_bytes;
    unsigned long long kernel_launches, h2d_bytes, d2h_bytes, peak_device_bytes;
    char execution_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_video_reconstruction_result;

int yvex_runtime_latent_execute(
    const yvex_runtime_latent_request *request,
    float *video_output, unsigned long long video_capacity,
    float *audio_output, unsigned long long audio_capacity,
    yvex_runtime_latent_result *result, yvex_error *err);
int yvex_runtime_latent_binding_identity(
    const char *domain, const char *const *identities, unsigned long long identity_count,
    const unsigned long long *facts, unsigned long long fact_count,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err);
int yvex_runtime_latent_evaluator_begin(
    yvex_runtime_latent_evaluator_evidence *evidence, const char *domain,
    const char *evaluator_identity, yvex_error *err);
int yvex_runtime_latent_evaluator_record(
    yvex_runtime_latent_evaluator_evidence *evidence, const char *residency_identity,
    const char *execution_identity, unsigned long long kernel_launches,
    unsigned long long h2d_bytes, unsigned long long d2h_bytes,
    unsigned long long device_bytes, yvex_error *err);
int yvex_runtime_latent_evaluator_finish(
    yvex_runtime_latent_evaluator_evidence *evidence,
    unsigned long long expected_evaluations,
    yvex_runtime_latent_evaluator_result *result, yvex_error *err);
int yvex_runtime_av_plan_build(
    const yvex_runtime_av_plan_policy *policy, unsigned long long text_tokens,
    unsigned long long width, unsigned long long height, unsigned long long frames,
    unsigned int inference_steps, yvex_runtime_av_plan *out, yvex_error *err);
int yvex_runtime_av_scheduler_step(
    float *output, const float *sample, const float *velocity,
    unsigned long long values, float timestep, float sigma, float sigma_next,
    yvex_error *err);
int yvex_runtime_av_latent_execute(
    const yvex_runtime_av_plan *plan, const yvex_runtime_latent_request *template,
    float *video, unsigned long long video_capacity, float *audio,
    unsigned long long audio_capacity, yvex_runtime_latent_result *result, yvex_error *err);
int yvex_runtime_av_layout_from_plan(
    const yvex_runtime_av_plan *plan, const yvex_runtime_av_layout_output *output,
    yvex_runtime_av_layout_result *result, yvex_error *err);
int yvex_runtime_av_layout_matches_plan(
    const yvex_runtime_av_plan *plan, const yvex_runtime_av_layout_output *output,
    const yvex_runtime_av_layout_result *result, yvex_error *err);
int yvex_runtime_av_layout_build(
    const yvex_runtime_av_layout_request *request,
    const yvex_runtime_av_layout_output *output,
    yvex_runtime_av_layout_result *result, yvex_error *err);
int yvex_runtime_av_unpack(
    const yvex_runtime_av_unpack_request *request,
    const yvex_runtime_av_unpack_output *output,
    yvex_runtime_av_unpack_result *result, yvex_error *err);
int yvex_runtime_av_video_reconstruction_plan_build(
    const yvex_runtime_av_video_reconstruction_request *request,
    yvex_runtime_av_video_reconstruction_plan *plan, yvex_error *err);
int yvex_runtime_av_video_reconstruct(
    const yvex_runtime_av_video_reconstruction_execution *execution,
    float *output, unsigned long long output_capacity,
    yvex_runtime_av_video_reconstruction_result *result, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
