/* Compose one staged decoded-media request without owning model-family policy. */
#ifndef INCLUDE_YVEX_INTERNAL_MEDIA_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MEDIA_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/component.h>
#include <yvex/internal/io.h>
#include <yvex/internal/latent.h>
#include <yvex/internal/media_target.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/transformer.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_AV_GENERATION_SCHEMA_V1 1u
#define YVEX_RUNTIME_MEDIA_HOST_SCHEMA_V1 1u
#define YVEX_RUNTIME_MEDIA_MODEL_SCHEMA_V1 1u
#define YVEX_RUNTIME_MEDIA_MODEL_OPEN_SCHEMA_V1 1u
#define YVEX_RUNTIME_MEDIA_PRESET_SCHEMA_V1 1u
#define YVEX_RUNTIME_MEDIA_PROGRESS_SCHEMA_V1 1u
#define YVEX_RUNTIME_MEDIA_PROFILE_CAP 5u
#define YVEX_RUNTIME_MEDIA_PROFILE_NAME_CAP 32u

typedef struct yvex_runtime_media_model yvex_runtime_media_model;

typedef struct {
    char name[YVEX_RUNTIME_MEDIA_PROFILE_NAME_CAP];
    unsigned long long width, height, maximum_frames;
    int preview_alias;
} yvex_runtime_media_profile;

typedef struct {
    unsigned int schema_version;
    char name[YVEX_RUNTIME_MEDIA_PROFILE_NAME_CAP];
    char profile[YVEX_RUNTIME_MEDIA_PROFILE_NAME_CAP];
    char format[8];
    unsigned long long width, height, frames, sigma_grid_points, seed;
    char identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_media_execution_preset;

typedef enum {
    YVEX_RUNTIME_MEDIA_PROGRESS_CONDITIONING_START = 0,
    YVEX_RUNTIME_MEDIA_PROGRESS_CONDITIONING_COMPLETE,
    YVEX_RUNTIME_MEDIA_PROGRESS_LATENT_START,
    YVEX_RUNTIME_MEDIA_PROGRESS_LATENT_STEP,
    YVEX_RUNTIME_MEDIA_PROGRESS_LATENT_COMPLETE,
    YVEX_RUNTIME_MEDIA_PROGRESS_VIDEO_START,
    YVEX_RUNTIME_MEDIA_PROGRESS_VIDEO_COMPLETE,
    YVEX_RUNTIME_MEDIA_PROGRESS_AUDIO_START,
    YVEX_RUNTIME_MEDIA_PROGRESS_AUDIO_COMPLETE,
    YVEX_RUNTIME_MEDIA_PROGRESS_PUBLICATION_START,
    YVEX_RUNTIME_MEDIA_PROGRESS_PUBLICATION_COMPLETE
} yvex_runtime_media_progress_kind;
typedef struct {
    unsigned int schema_version;
    yvex_runtime_media_progress_kind kind;
    unsigned long long completed, total, value;
} yvex_runtime_media_progress;
typedef int (*yvex_runtime_media_progress_fn)(
    void *context, const yvex_runtime_media_progress *progress, yvex_error *err);

typedef struct {
    unsigned long long batch, samples_per_channel, output_values;
    unsigned long long kernel_launches, h2d_bytes, d2h_bytes, device_bytes;
    unsigned long long peak_workspace_bytes;
    char residency_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_audio_result;

typedef yvex_media_plan_fn yvex_runtime_av_plan_fn;
typedef yvex_media_layout_fn yvex_runtime_av_layout_fn;
typedef yvex_media_component_admit_fn yvex_runtime_av_component_admit_fn;
typedef yvex_media_condition_fn yvex_runtime_av_condition_fn;
typedef yvex_media_latent_fn yvex_runtime_av_latent_fn;
typedef yvex_media_video_fn yvex_runtime_av_video_fn;
typedef yvex_media_audio_fn yvex_runtime_av_audio_fn;

typedef struct {
    unsigned int schema_version;
    const char *target, *prompt, *output_path;
    const char *text_artifact_path, *transformer_artifact_path;
    const char *video_artifact_path, *audio_artifact_path;
    const char *source_identity;
    unsigned long long frames, width, height;
    unsigned long long fps_numerator, fps_denominator, audio_sample_rate;
    unsigned int inference_steps;
    unsigned long long conditioning_layers, transformer_blocks, seed;
    unsigned long long maximum_prompt_tokens, maximum_packed_rows;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    unsigned long long maximum_workspace_bytes, maximum_file_bytes;
    yvex_backend_kind component_backend;
    const char *output_semantic_domain;
    const yvex_transformer_linear_requirement *video_output_requirement;
    const yvex_transformer_linear_requirement *audio_output_requirement;
    yvex_transformer_linear_physical_plan video_output_specialization;
    yvex_transformer_linear_physical_plan audio_output_specialization;
    unsigned long long video_temporal_ratio, video_clip_length, video_token_drop;
    unsigned long long video_spatial_ratio, video_tile_size, video_minimum_tile_overlap;
    const float *video_mean, *video_std, *audio_mean, *audio_std;
    const float *pixel_mean, *pixel_std;
    unsigned long long video_channels, audio_channels, pixel_channels;
    unsigned long long audio_output_channels, audio_samples_per_step;
    yvex_runtime_av_plan_fn plan_build;
    yvex_runtime_av_layout_fn layout_build;
    yvex_runtime_av_component_admit_fn component_admit;
    yvex_runtime_av_condition_fn condition;
    yvex_runtime_av_latent_fn latent;
    yvex_runtime_av_video_fn video_decode;
    yvex_runtime_av_audio_fn audio_decode;
    int (*cancel_requested)(void *);
    void *cancel_context;
    yvex_runtime_media_progress_fn observe_progress;
    void *progress_context;
} yvex_runtime_av_generation_request;

/*
 * The compiled family adapter seals model facts and graph callbacks into this path-owning profile
 * before the server sees it. The server copies referenced strings during configuration, so callers
 * may keep this profile on their own stack for the foreground host lifetime.
 */
typedef struct {
    unsigned int schema_version;
    yvex_runtime_av_generation_request request_template;
    yvex_runtime_media_profile profiles[YVEX_RUNTIME_MEDIA_PROFILE_CAP];
    unsigned long long profile_count;
    unsigned long long frames_per_chunk, frame_remainder;
    unsigned long long minimum_frames, maximum_frames;
    unsigned long long minimum_inference_steps, maximum_inference_steps;
    unsigned long long canvas_multiple, maximum_canvas_pixels;
    char output_root[YVEX_PATH_CAP];
    char target[128], source_identity[YVEX_SHA256_HEX_CAP];
    char text_artifact[YVEX_PATH_CAP], transformer_artifact[YVEX_PATH_CAP];
    char video_artifact[YVEX_PATH_CAP], audio_artifact[YVEX_PATH_CAP];
} yvex_runtime_media_host_profile;

typedef struct {
    unsigned int schema_version;
    const char *artifact_reopen_cache_root;
    void (*observe_component)(void *context, const char *role,
                              const yvex_artifact_admission_evidence *evidence);
    void *observer_context;
} yvex_runtime_media_model_open_options;

typedef struct {
    char role[32];
    yvex_artifact_admission_evidence evidence;
} yvex_runtime_media_component_summary;

typedef struct {
    unsigned int schema_version;
    unsigned long long prompt_tokens, frames, width, height, audio_samples;
    unsigned long long model_evaluations, kernel_launches, peak_device_bytes;
    unsigned long long peak_workspace_bytes, file_bytes;
    char prompt_identity[YVEX_SHA256_HEX_CAP];
    char conditioning_identity[YVEX_SHA256_HEX_CAP];
    char plan_identity[YVEX_SHA256_HEX_CAP];
    char layout_identity[YVEX_SHA256_HEX_CAP];
    char latent_identity[YVEX_SHA256_HEX_CAP];
    char vae_input_identity[YVEX_SHA256_HEX_CAP];
    char video_identity[YVEX_SHA256_HEX_CAP];
    char audio_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
    char file_identity[YVEX_SHA256_HEX_CAP];
    char publication_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_generation_result;

typedef struct {
    unsigned int schema_version;
    unsigned long long component_count, artifact_bytes, artifact_bytes_hashed;
    char model_identity[YVEX_SHA256_HEX_CAP];
    char source_identity[YVEX_SHA256_HEX_CAP];
    yvex_runtime_media_component_summary components[4];
    int complete;
} yvex_runtime_media_model_summary;

int yvex_runtime_av_generate(const yvex_runtime_av_generation_request *,
                             yvex_runtime_av_generation_result *, yvex_error *);
int yvex_runtime_media_model_open(
    yvex_runtime_media_model **, const yvex_runtime_av_generation_request *,
    const yvex_runtime_media_model_open_options *,
    yvex_runtime_media_model_summary *, yvex_error *);
int yvex_runtime_media_model_generate(
    yvex_runtime_media_model *, const yvex_runtime_av_generation_request *,
    yvex_runtime_av_generation_result *, yvex_error *);
void yvex_runtime_media_model_close(yvex_runtime_media_model **);
int yvex_runtime_media_host_profile_build(
    yvex_runtime_media_host_profile *, const yvex_media_target_profile *,
    const yvex_media_execution_recipe *, const char *artifact_root,
    const char *output_root, yvex_error *);
int yvex_runtime_media_request_specialize(
    yvex_runtime_av_generation_request *, const char *,
    const yvex_transformer_linear_requirement *,
    const yvex_transformer_linear_requirement *,
    yvex_error *);
int yvex_runtime_media_execution_preset_build(
    const yvex_runtime_media_host_profile *, yvex_runtime_media_execution_preset *,
    yvex_error *);
int yvex_runtime_media_execution_preset_validate(
    const yvex_runtime_media_host_profile *,
    const yvex_runtime_media_execution_preset *, yvex_error *);

#ifdef __cplusplus
}
#endif
#endif
