/* Bind family media facts and graph execution callbacks without importing runtime ownership. */
#ifndef INCLUDE_YVEX_INTERNAL_MEDIA_TARGET_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MEDIA_TARGET_H_INCLUDED

#include <yvex/backend.h>
#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_MEDIA_TARGET_PROFILE_SCHEMA_V1 1u
#define YVEX_MEDIA_EXECUTION_RECIPE_SCHEMA_V1 1u
#define YVEX_MEDIA_TARGET_TIER_CAP 5u

typedef struct yvex_artifact yvex_artifact;
typedef struct yvex_gguf yvex_gguf;
typedef struct yvex_tensor_table yvex_tensor_table;
typedef struct yvex_complete_artifact_admission yvex_complete_artifact_admission;
typedef struct yvex_artifact_admission_failure yvex_artifact_admission_failure;
typedef struct yvex_artifact_admission_options yvex_artifact_admission_options;
typedef struct yvex_artifact_admission_evidence yvex_artifact_admission_evidence;
typedef struct yvex_runtime_av_plan yvex_runtime_av_plan;
typedef struct yvex_runtime_av_layout_output yvex_runtime_av_layout_output;
typedef struct yvex_runtime_av_layout_result yvex_runtime_av_layout_result;
typedef struct yvex_runtime_av_conditioning_result yvex_runtime_av_conditioning_result;
typedef struct yvex_runtime_av_latent_context yvex_runtime_av_latent_context;
typedef struct yvex_runtime_latent_result yvex_runtime_latent_result;
typedef struct yvex_runtime_latent_evaluator_result yvex_runtime_latent_evaluator_result;
typedef struct yvex_runtime_component_session yvex_runtime_component_session;
typedef struct yvex_runtime_av_video_decode_options yvex_runtime_av_video_decode_options;
typedef struct yvex_runtime_av_video_decode_result yvex_runtime_av_video_decode_result;
typedef struct yvex_runtime_av_audio_decode_options yvex_runtime_av_audio_decode_options;
typedef struct yvex_runtime_av_audio_decode_result yvex_runtime_av_audio_decode_result;
typedef struct yvex_component_execution_failure yvex_component_execution_failure;
typedef struct yvex_transformer_linear_requirement yvex_transformer_linear_requirement;

typedef struct {
    const char *name;
    unsigned long long width, height, maximum_frames;
    int preview_alias;
} yvex_media_target_tier;

typedef struct yvex_media_target_profile {
    unsigned int schema_version;
    const char *target, *family, *source_identity;
    const char *text_artifact, *transformer_artifact;
    const char *video_artifact, *audio_artifact;
    yvex_media_target_tier tiers[YVEX_MEDIA_TARGET_TIER_CAP];
    unsigned long long tier_count;
    unsigned long long fps_numerator, fps_denominator, audio_sample_rate, seed;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    unsigned long long maximum_workspace_bytes, maximum_file_bytes;
    unsigned long long video_temporal_ratio, video_clip_length, video_token_drop;
    unsigned long long video_spatial_ratio, video_tile_size, video_minimum_tile_overlap;
    const float *video_mean, *video_std, *audio_mean, *audio_std;
    const float *pixel_mean, *pixel_std;
    unsigned long long video_channels, audio_channels, pixel_channels;
    unsigned long long audio_output_channels, audio_samples_per_step;
    unsigned long long frames_per_chunk, frame_remainder;
    unsigned long long minimum_frames, maximum_frames;
    unsigned long long minimum_inference_steps, maximum_inference_steps;
    unsigned long long canvas_multiple, maximum_canvas_pixels;
} yvex_media_target_profile;

typedef int (*yvex_media_plan_fn)(
    yvex_runtime_av_plan *, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned int, yvex_error *);
typedef int (*yvex_media_layout_fn)(
    const yvex_runtime_av_plan *, const yvex_runtime_av_layout_output *,
    yvex_runtime_av_layout_result *, yvex_error *);
typedef int (*yvex_media_component_admit_fn)(
    const char *, const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *,
    const yvex_artifact_admission_options *, yvex_complete_artifact_admission *,
    yvex_artifact_admission_evidence *, yvex_artifact_admission_failure *, yvex_error *);
typedef int (*yvex_media_condition_fn)(
    const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *,
    const unsigned int *, unsigned long long, unsigned long long, float *, unsigned long long,
    unsigned long long, unsigned long long, yvex_runtime_av_conditioning_result *,
    yvex_error *);
typedef int (*yvex_media_latent_fn)(
    const yvex_runtime_av_plan *, const yvex_runtime_av_latent_context *,
    unsigned long long, unsigned long long, float *, unsigned long long, float *,
    unsigned long long, yvex_runtime_latent_result *,
    yvex_runtime_latent_evaluator_result *, yvex_error *);
typedef int (*yvex_media_video_fn)(
    yvex_runtime_component_session *, const yvex_runtime_av_video_decode_options *,
    yvex_runtime_av_video_decode_result *, yvex_component_execution_failure *, yvex_error *);
typedef int (*yvex_media_audio_fn)(
    const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *,
    const yvex_runtime_av_audio_decode_options *, unsigned long long,
    yvex_runtime_av_audio_decode_result *, yvex_component_execution_failure *, yvex_error *);

typedef struct yvex_media_execution_recipe {
    unsigned int schema_version;
    unsigned long long conditioning_layers, transformer_blocks;
    unsigned long long maximum_prompt_tokens, maximum_packed_rows;
    yvex_backend_kind component_backend;
    const char *output_semantic_domain;
    const yvex_transformer_linear_requirement *video_output_requirement;
    const yvex_transformer_linear_requirement *audio_output_requirement;
    yvex_media_plan_fn plan_build;
    yvex_media_layout_fn layout_build;
    yvex_media_component_admit_fn component_admit;
    yvex_media_condition_fn condition;
    yvex_media_latent_fn latent;
    yvex_media_video_fn video_decode;
    yvex_media_audio_fn audio_decode;
} yvex_media_execution_recipe;

#ifdef __cplusplus
}
#endif
#endif
