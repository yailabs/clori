/* Compose one staged decoded-media request without owning model-family policy. */
#ifndef INCLUDE_YVEX_INTERNAL_MEDIA_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MEDIA_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/component.h>
#include <yvex/internal/io.h>
#include <yvex/internal/latent.h>
#include <yvex/internal/runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_AV_GENERATION_SCHEMA_V1 1u

typedef struct {
    unsigned long long batch, samples_per_channel, output_values;
    unsigned long long kernel_launches, h2d_bytes, d2h_bytes, device_bytes;
    unsigned long long peak_workspace_bytes;
    char residency_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_runtime_av_audio_result;

typedef int (*yvex_runtime_av_plan_fn)(
    yvex_runtime_av_plan *, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned int, yvex_error *);
typedef int (*yvex_runtime_av_layout_fn)(
    const yvex_runtime_av_plan *, const yvex_runtime_av_layout_output *,
    yvex_runtime_av_layout_result *, yvex_error *);
typedef int (*yvex_runtime_av_component_admit_fn)(
    const char *, const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *,
    yvex_complete_artifact_admission *, yvex_artifact_admission_failure *, yvex_error *);
typedef int (*yvex_runtime_av_condition_fn)(
    const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *,
    const unsigned int *, unsigned long long, unsigned long long, float *, unsigned long long,
    unsigned long long, unsigned long long, yvex_runtime_av_conditioning_result *,
    yvex_error *);
typedef int (*yvex_runtime_av_latent_fn)(
    const yvex_runtime_av_plan *, const yvex_runtime_av_latent_context *,
    unsigned long long, unsigned long long, float *,
    unsigned long long, float *, unsigned long long, yvex_runtime_latent_result *,
    yvex_runtime_latent_evaluator_result *, yvex_error *);
typedef int (*yvex_runtime_av_video_fn)(
    yvex_runtime_component_session *, const yvex_runtime_av_video_decode_options *,
    yvex_runtime_av_video_decode_result *, yvex_component_execution_failure *, yvex_error *);
typedef int (*yvex_runtime_av_audio_fn)(
    const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *,
    const yvex_runtime_av_audio_decode_options *, unsigned long long,
    yvex_runtime_av_audio_decode_result *, yvex_component_execution_failure *, yvex_error *);

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
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    unsigned long long maximum_workspace_bytes, maximum_file_bytes;
    yvex_backend_kind component_backend;
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
} yvex_runtime_av_generation_request;

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

int yvex_runtime_av_generate(const yvex_runtime_av_generation_request *,
                             yvex_runtime_av_generation_result *, yvex_error *);

#ifdef __cplusplus
}
#endif
#endif
