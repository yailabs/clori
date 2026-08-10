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
#define YVEX_RUNTIME_AV_TEMPORAL_PATTERN_CAP 8u
#define YVEX_RUNTIME_LATENT_EVALUATOR_SCHEMA_V1 1u

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

#ifdef __cplusplus
}
#endif
#endif
