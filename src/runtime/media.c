/*
 * Stage one identity-bound audio-video request across independently resident components.
 *
 * Family callbacks retain numeric policy. This owner supplies the common transactional
 * lifecycle: tokenizer, conditioning, latent state, late VAE residency, synchronization,
 * and publication are advanced in order and no partial decoded output becomes visible.
 */
#include <yvex/internal/media.h>

#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/gguf.h>
#include <yvex/internal/family_catalog.h>
#include <yvex/internal/image.h>
#include <yvex/model.h>
#include <yvex/tokenizer.h>

typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
} component_view;

typedef enum {
    MEDIA_COMPONENT_TEXT = 0,
    MEDIA_COMPONENT_TRANSFORMER,
    MEDIA_COMPONENT_VIDEO,
    MEDIA_COMPONENT_AUDIO,
    MEDIA_COMPONENT_COUNT
} media_component_index;

static const char *const media_component_names[MEDIA_COMPONENT_COUNT] = {
    "text_encoder", "transformer", "video_vae", "audio_vae",
};

typedef struct {
    yvex_runtime_media_model *model;
    media_component_index component;
} media_resource_release_context;

struct yvex_runtime_media_model {
    yvex_model_context text;
    component_view transformer, video, audio;
    yvex_complete_artifact_admission admissions[MEDIA_COMPONENT_COUNT];
    yvex_runtime_av_generation_request contract;
    yvex_runtime_media_model_summary summary;
    yvex_engine_resource_catalog *resources;
    yvex_engine_resource_handle resource_handles[MEDIA_COMPONENT_COUNT];
    media_resource_release_context release_contexts[MEDIA_COMPONENT_COUNT];
};

static _Atomic unsigned long long media_engine_generation_counter;

typedef struct {
    yvex_runtime_media_model *model;
    const yvex_runtime_av_generation_request *request;
    yvex_runtime_av_plan plan;
    yvex_runtime_av_layout_output layout;
    yvex_runtime_av_layout_result layout_result;
    yvex_runtime_latent_result latent_result;
    yvex_runtime_latent_evaluator_result evaluator_result;
    yvex_runtime_av_unpack_result unpack_result;
    yvex_runtime_av_video_reconstruction_result video_result;
    yvex_runtime_av_audio_decode_result audio_result;
    yvex_media_avi_result media_result;
    yvex_runtime_av_conditioning_result conditioning_result;
    yvex_runtime_av_keyframe_result keyframe_result;
    yvex_image condition_images[YVEX_RUNTIME_MEDIA_CONDITION_CAP];
    float *conditioning, *condition_latents, *video_rows, *audio_rows;
    float *video_latent, *audio_latent;
    float *rgb, *pcm;
    unsigned int *text_tags, *timestep_indices;
    unsigned long long conditioning_values, condition_latent_values;
    unsigned long long video_row_values, audio_row_values;
    unsigned long long video_latent_values, audio_latent_values, rgb_values, pcm_values;
    unsigned long long layout_position_values;
    unsigned long long host_live, host_peak;
    char prompt_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} generation_state;

typedef struct {
    const yvex_runtime_av_generation_request *request;
    const yvex_component_execution *component;
} video_decode_context;

static int media_artifact_path(char output[YVEX_PATH_CAP], const char *root,
                               const char *relative)
{
    int length;

    if (!output || !root || root[0] != '/' || !relative || !relative[0] ||
        relative[0] == '/' || strstr(relative, ".."))
        return 0;
    length = snprintf(output, YVEX_PATH_CAP, "%s/%s", root, relative);
    return length >= 0 && length < YVEX_PATH_CAP;
}

static int media_target_validate(
    const yvex_media_target_profile *target,
    const yvex_media_execution_recipe *execution, yvex_error *err)
{
    if (!target || target->schema_version != YVEX_MEDIA_TARGET_PROFILE_SCHEMA_V2 ||
        !target->target || !target->target[0] || !target->family || !target->family[0] ||
        !target->source_identity || strlen(target->source_identity) != 64u ||
        !target->tier_count || target->tier_count > YVEX_MEDIA_TARGET_TIER_CAP ||
        !target->fps_numerator || !target->fps_denominator || !target->audio_sample_rate ||
        !target->maximum_host_bytes || !target->maximum_device_bytes ||
        !target->maximum_workspace_bytes || !target->maximum_file_bytes ||
        !target->minimum_frames || target->minimum_frames > target->maximum_frames ||
        !target->minimum_inference_steps ||
        target->minimum_inference_steps > target->maximum_inference_steps ||
        !target->released_sigma_grid_points ||
        target->released_sigma_grid_points < target->minimum_inference_steps ||
        target->released_sigma_grid_points > target->maximum_inference_steps ||
        !target->canvas_multiple || !target->canvas_short_edge ||
        !target->minimum_canvas_pixels ||
        target->minimum_canvas_pixels > target->maximum_canvas_pixels ||
        !target->released_width ||
        !target->released_height || !target->minimum_duration_milliseconds ||
        target->minimum_duration_milliseconds > target->maximum_duration_milliseconds ||
        !target->minimum_aspect_numerator || !target->minimum_aspect_denominator ||
        !target->maximum_aspect_numerator || !target->maximum_aspect_denominator ||
        !execution || execution->schema_version != YVEX_MEDIA_EXECUTION_RECIPE_SCHEMA_V1 ||
        !execution->conditioning_layers || !execution->transformer_blocks ||
        !execution->maximum_prompt_tokens || !execution->maximum_packed_rows ||
        !execution->plan_build || !execution->layout_build || !execution->component_admit ||
        !execution->condition || !execution->keyframe_encode || !execution->latent ||
        !execution->video_decode ||
        !execution->audio_decode) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.media-profile",
                       "complete target facts and execution recipe are required");
        return YVEX_ERR_INVALID_ARG;
    }
    return YVEX_OK;
}

int yvex_runtime_media_host_profile_build(
    yvex_runtime_media_host_profile *out, const yvex_media_target_profile *target,
    const yvex_media_execution_recipe *execution, const char *artifact_root,
    const char *output_root, yvex_error *err)
{
    yvex_runtime_av_generation_request *request;
    unsigned long long index;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    rc = media_target_validate(target, execution, err);
    if (rc != YVEX_OK || !out || !artifact_root || artifact_root[0] != '/' ||
        !output_root || output_root[0] != '/') {
        if (rc == YVEX_OK)
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.media-profile",
                           "output and absolute component and publication roots are required");
        return rc == YVEX_OK ? YVEX_ERR_INVALID_ARG : rc;
    }
    if (strlen(target->target) >= sizeof(out->target) ||
        strlen(output_root) >= sizeof(out->output_root) ||
        !media_artifact_path(out->text_artifact, artifact_root, target->text_artifact) ||
        !media_artifact_path(out->transformer_artifact, artifact_root,
                             target->transformer_artifact) ||
        !media_artifact_path(out->video_artifact, artifact_root, target->video_artifact) ||
        !media_artifact_path(out->audio_artifact, artifact_root, target->audio_artifact)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.media-profile",
                       "media profile path exceeds its bounded representation");
        return YVEX_ERR_BOUNDS;
    }
    for (index = 0ull; index < target->tier_count; ++index) {
        const yvex_media_target_tier *tier = target->tiers + index;
        yvex_runtime_av_plan plan = {0};

        if (!tier->name || !tier->name[0] ||
            strlen(tier->name) >= YVEX_RUNTIME_MEDIA_PROFILE_NAME_CAP ||
            !tier->width || !tier->height ||
            tier->maximum_frames < target->minimum_frames ||
            tier->maximum_frames > target->maximum_frames) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.media-profile",
                           "media tier exceeds its admitted bounds");
            return YVEX_ERR_BOUNDS;
        }
        yvex_media_plan_request plan_request = {
            .schema_version = YVEX_RUNTIME_AV_PLAN_SCHEMA_V1,
            .text_tokens = execution->maximum_prompt_tokens,
            .width = tier->width, .height = tier->height,
            .frames = tier->maximum_frames, .inference_steps = 1u,
        };
        rc = execution->plan_build(&plan, &plan_request, err);
        if (rc != YVEX_OK) return rc;
        if (plan.packed_rows > execution->maximum_packed_rows) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.media-profile",
                           "media tier exceeds the admitted packed-row capacity");
            return YVEX_ERR_BOUNDS;
        }
        yvex_core_text_copy(out->profiles[index].name,
                            sizeof(out->profiles[index].name), tier->name);
        out->profiles[index].width = tier->width;
        out->profiles[index].height = tier->height;
        out->profiles[index].maximum_frames = tier->maximum_frames;
        out->profiles[index].preview_alias = tier->preview_alias;
    }
    yvex_core_text_copy(out->target, sizeof(out->target), target->target);
    yvex_core_text_copy(out->source_identity, sizeof(out->source_identity),
                        target->source_identity);
    yvex_core_text_copy(out->output_root, sizeof(out->output_root), output_root);
    out->schema_version = YVEX_RUNTIME_MEDIA_HOST_SCHEMA_V2;
    out->profile_count = target->tier_count;
    out->frames_per_chunk = target->frames_per_chunk;
    out->frame_remainder = target->frame_remainder;
    out->minimum_frames = target->minimum_frames;
    out->maximum_frames = target->maximum_frames;
    out->minimum_inference_steps = target->minimum_inference_steps;
    out->maximum_inference_steps = target->maximum_inference_steps;
    out->released_sigma_grid_points = target->released_sigma_grid_points;
    out->default_seed = target->seed;
    out->canvas_multiple = target->canvas_multiple;
    out->canvas_short_edge = target->canvas_short_edge;
    out->minimum_canvas_pixels = target->minimum_canvas_pixels;
    out->maximum_canvas_pixels = target->maximum_canvas_pixels;
    out->released_width = target->released_width;
    out->released_height = target->released_height;
    out->minimum_duration_milliseconds = target->minimum_duration_milliseconds;
    out->maximum_duration_milliseconds = target->maximum_duration_milliseconds;
    out->minimum_aspect_numerator = target->minimum_aspect_numerator;
    out->minimum_aspect_denominator = target->minimum_aspect_denominator;
    out->maximum_aspect_numerator = target->maximum_aspect_numerator;
    out->maximum_aspect_denominator = target->maximum_aspect_denominator;
    request = &out->request_template;
    *request = (yvex_runtime_av_generation_request){
        .schema_version = YVEX_RUNTIME_AV_GENERATION_SCHEMA_V2,
        .target = out->target,
        .text_artifact_path = out->text_artifact,
        .transformer_artifact_path = out->transformer_artifact,
        .video_artifact_path = out->video_artifact,
        .audio_artifact_path = out->audio_artifact,
        .source_identity = out->source_identity,
        .fps_numerator = target->fps_numerator, .fps_denominator = target->fps_denominator,
        .audio_sample_rate = target->audio_sample_rate, .seed = target->seed,
        .keyframe_encode_seed = target->keyframe_encode_seed,
        .conditioning_layers = execution->conditioning_layers,
        .transformer_blocks = execution->transformer_blocks,
        .maximum_prompt_tokens = execution->maximum_prompt_tokens,
        .maximum_packed_rows = execution->maximum_packed_rows,
        .maximum_host_bytes = target->maximum_host_bytes,
        .maximum_device_bytes = target->maximum_device_bytes,
        .maximum_workspace_bytes = target->maximum_workspace_bytes,
        .maximum_file_bytes = target->maximum_file_bytes,
        .component_backend = execution->component_backend,
        .output_semantic_domain = execution->output_semantic_domain,
        .video_output_requirement = execution->video_output_requirement,
        .audio_output_requirement = execution->audio_output_requirement,
        .video_temporal_ratio = target->video_temporal_ratio,
        .video_clip_length = target->video_clip_length,
        .video_token_drop = target->video_token_drop,
        .video_spatial_ratio = target->video_spatial_ratio,
        .video_tile_size = target->video_tile_size,
        .video_minimum_tile_overlap = target->video_minimum_tile_overlap,
        .video_mean = target->video_mean, .video_std = target->video_std,
        .audio_mean = target->audio_mean, .audio_std = target->audio_std,
        .pixel_mean = target->pixel_mean, .pixel_std = target->pixel_std,
        .video_channels = target->video_channels, .audio_channels = target->audio_channels,
        .pixel_channels = target->pixel_channels,
        .audio_output_channels = target->audio_output_channels,
        .audio_samples_per_step = target->audio_samples_per_step,
        .plan_build = execution->plan_build, .layout_build = execution->layout_build,
        .component_admit = execution->component_admit, .condition = execution->condition,
        .keyframe_encode = execution->keyframe_encode,
        .latent = execution->latent, .video_decode = execution->video_decode,
        .audio_decode = execution->audio_decode};
    if (execution->output_semantic_domain || execution->video_output_requirement ||
        execution->audio_output_requirement)
        rc = yvex_runtime_media_request_specialize(
            request, execution->output_semantic_domain,
            execution->video_output_requirement,
            execution->audio_output_requirement, err);
    if (rc != YVEX_OK) return rc;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int generation_fail(
    yvex_error *err, yvex_status status, const char *where, const char *message);

static int media_preset_identity(
    const yvex_runtime_media_execution_preset *preset,
    char identity[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const unsigned long long facts[] = {
        preset->width, preset->height, preset->frames,
        preset->sigma_grid_points, preset->seed,
    };
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.hosted-media-preset.v1") ||
        !yvex_sha256_update_text(&hash, preset->name) ||
        !yvex_sha256_update_text(&hash, preset->profile) ||
        !yvex_sha256_update_text(&hash, preset->format))
        return generation_fail(err, YVEX_ERR_STATE, "runtime.media-preset",
                               "hosted media preset identity could not start");
    for (index = 0ull; index < sizeof(facts) / sizeof(facts[0]); ++index)
        if (!yvex_sha256_update_u64_be(&hash, facts[index]))
            return generation_fail(err, YVEX_ERR_STATE, "runtime.media-preset",
                                   "hosted media preset identity facts failed");
    if (!yvex_sha256_final(&hash, digest))
        return generation_fail(err, YVEX_ERR_STATE, "runtime.media-preset",
                               "hosted media preset identity could not finish");
    yvex_sha256_hex(digest, identity);
    return YVEX_OK;
}

int yvex_runtime_media_execution_preset_validate(
    const yvex_runtime_media_host_profile *host,
    const yvex_runtime_media_execution_preset *preset, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP];
    const yvex_runtime_media_profile *profile = NULL;
    unsigned long long index;
    int rc;

    if (!host || host->schema_version != YVEX_RUNTIME_MEDIA_HOST_SCHEMA_V2 ||
        !preset || preset->schema_version != YVEX_RUNTIME_MEDIA_PRESET_SCHEMA_V1 ||
        !preset->name[0] || !preset->profile[0] || strcmp(preset->format, "avi") ||
        !preset->width || !preset->height || !preset->frames ||
        preset->frames < host->minimum_frames || preset->frames > host->maximum_frames ||
        preset->frames % host->frames_per_chunk != host->frame_remainder ||
        preset->sigma_grid_points < host->minimum_inference_steps ||
        preset->sigma_grid_points > host->maximum_inference_steps || !preset->complete)
        return generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.media-preset",
                               "one complete admitted hosted media preset is required");
    for (index = 0ull; index < host->profile_count; ++index)
        if (!strcmp(host->profiles[index].name, preset->profile)) {
            profile = host->profiles + index;
            break;
        }
    if (!strcmp(preset->profile, "released")) {
        unsigned long long pixels, left, right;
        if (preset->width % host->canvas_multiple ||
            preset->height % host->canvas_multiple ||
            !yvex_core_u64_mul(preset->width, preset->height, &pixels) ||
            pixels < host->minimum_canvas_pixels ||
            pixels > host->maximum_canvas_pixels ||
            !yvex_core_u64_mul(preset->width,
                               host->minimum_aspect_denominator, &left) ||
            !yvex_core_u64_mul(preset->height,
                               host->minimum_aspect_numerator, &right) ||
            left < right ||
            !yvex_core_u64_mul(preset->width,
                               host->maximum_aspect_denominator, &left) ||
            !yvex_core_u64_mul(preset->height,
                               host->maximum_aspect_numerator, &right) ||
            left > right ||
            preset->sigma_grid_points != host->released_sigma_grid_points)
            return generation_fail(err, YVEX_ERR_BOUNDS,
                                   "runtime.media-preset",
                                   "released media execution exceeds its admitted canvas or trajectory");
    } else if (!profile || profile->width != preset->width ||
               profile->height != preset->height ||
               preset->frames > profile->maximum_frames)
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.media-preset",
                               "hosted media preset exceeds its admitted profile");
    rc = media_preset_identity(preset, identity, err);
    if (rc == YVEX_OK && strcmp(identity, preset->identity))
        rc = generation_fail(err, YVEX_ERR_FORMAT, "runtime.media-preset",
                             "hosted media preset identity is inconsistent");
    return rc;
}

int yvex_runtime_media_execution_preset_build(
    const yvex_runtime_media_host_profile *host,
    yvex_runtime_media_execution_preset *out, yvex_error *err)
{
    const yvex_runtime_media_profile *preview = NULL;
    unsigned long long index;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    if (!host || !out || host->schema_version != YVEX_RUNTIME_MEDIA_HOST_SCHEMA_V2)
        return generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.media-preset",
                               "one admitted media host profile is required");
    for (index = 0ull; index < host->profile_count; ++index) {
        if (!host->profiles[index].preview_alias) continue;
        if (preview)
            return generation_fail(err, YVEX_ERR_STATE, "runtime.media-preset",
                                   "hosted media policy requires one preview profile");
        preview = host->profiles + index;
    }
    if (!preview || host->minimum_frames > preview->maximum_frames)
        return generation_fail(err, YVEX_ERR_STATE, "runtime.media-preset",
                               "hosted media preview policy is not admitted");
    out->schema_version = YVEX_RUNTIME_MEDIA_PRESET_SCHEMA_V1;
    yvex_core_text_copy(out->name, sizeof(out->name), "interactive-preview-v1");
    yvex_core_text_copy(out->profile, sizeof(out->profile), preview->name);
    yvex_core_text_copy(out->format, sizeof(out->format), "avi");
    out->width = preview->width;
    out->height = preview->height;
    out->frames = host->minimum_frames;
    out->sigma_grid_points = host->minimum_inference_steps;
    out->seed = 42ull;
    out->complete = 1;
    rc = media_preset_identity(out, out->identity, err);
    if (rc == YVEX_OK) rc = yvex_runtime_media_execution_preset_validate(host, out, err);
    return rc;
}

static int released_frames_resolve(
    const yvex_runtime_media_host_profile *host,
    unsigned long long duration_milliseconds,
    unsigned long long *frames, yvex_error *err)
{
    unsigned long long numerator, denominator, requested, remainder, aligned;
    if (!duration_milliseconds ||
        duration_milliseconds < host->minimum_duration_milliseconds ||
        duration_milliseconds > host->maximum_duration_milliseconds ||
        !yvex_core_u64_mul(duration_milliseconds,
                           host->request_template.fps_numerator, &numerator) ||
        !yvex_core_u64_mul(1000ull,
                           host->request_template.fps_denominator, &denominator) ||
        !denominator)
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.media-execution",
                               "released duration is outside the admitted source contract");
    requested = numerator / denominator;
    if (numerator % denominator &&
        !yvex_core_u64_add(requested, 1ull, &requested))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.media-execution",
                               "released duration frame count overflowed");
    remainder = requested % host->frames_per_chunk;
    if (remainder <= host->frame_remainder) {
        if (!yvex_core_u64_add(requested,
                               host->frame_remainder - remainder, &aligned))
            return generation_fail(err, YVEX_ERR_BOUNDS,
                                   "runtime.media-execution",
                                   "released duration alignment overflowed");
    }
    else if (!yvex_core_u64_add(
                 requested, host->frames_per_chunk - remainder +
                                  host->frame_remainder, &aligned))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.media-execution",
                               "released duration alignment overflowed");
    if (aligned < host->minimum_frames || aligned > host->maximum_frames)
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.media-execution",
                               "released duration cannot align inside the source frame envelope");
    *frames = aligned;
    return YVEX_OK;
}

int yvex_runtime_media_execution_resolve(
    const yvex_runtime_media_host_profile *host,
    const yvex_runtime_media_execution_request *request,
    yvex_runtime_media_execution_preset *out, yvex_error *err)
{
    yvex_runtime_media_execution_kind kind;
    unsigned int allowed = YVEX_RUNTIME_MEDIA_EXECUTION_WIDTH |
                           YVEX_RUNTIME_MEDIA_EXECUTION_HEIGHT |
                           YVEX_RUNTIME_MEDIA_EXECUTION_DURATION |
                           YVEX_RUNTIME_MEDIA_EXECUTION_SEED;
    int rc;
    if (out) memset(out, 0, sizeof(*out));
    if (!host || host->schema_version != YVEX_RUNTIME_MEDIA_HOST_SCHEMA_V2 ||
        !request || request->schema_version != YVEX_RUNTIME_MEDIA_EXECUTION_SCHEMA_V1 ||
        !out || request->kind > YVEX_RUNTIME_MEDIA_EXECUTION_RELEASED ||
        request->present & ~allowed ||
        (!!(request->present & YVEX_RUNTIME_MEDIA_EXECUTION_WIDTH) !=
         !!(request->present & YVEX_RUNTIME_MEDIA_EXECUTION_HEIGHT)) ||
        (!(request->present & YVEX_RUNTIME_MEDIA_EXECUTION_WIDTH) &&
         (request->width || request->height)) ||
        (!(request->present & YVEX_RUNTIME_MEDIA_EXECUTION_DURATION) &&
         request->duration_milliseconds) ||
        (!(request->present & YVEX_RUNTIME_MEDIA_EXECUTION_SEED) && request->seed))
        return generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.media-execution",
                               "one complete typed media execution request is required");
    kind = request->kind == YVEX_RUNTIME_MEDIA_EXECUTION_DEFAULT
               ? YVEX_RUNTIME_MEDIA_EXECUTION_RELEASED : request->kind;
    if (kind == YVEX_RUNTIME_MEDIA_EXECUTION_PREVIEW) {
        if (request->present & ~(unsigned int)YVEX_RUNTIME_MEDIA_EXECUTION_SEED)
            return generation_fail(err, YVEX_ERR_UNSUPPORTED,
                                   "runtime.media-execution",
                                   "preview geometry and trajectory are immutable");
        rc = yvex_runtime_media_execution_preset_build(host, out, err);
        if (rc != YVEX_OK) return rc;
        if (request->present & YVEX_RUNTIME_MEDIA_EXECUTION_SEED) {
            out->seed = request->seed;
            rc = media_preset_identity(out, out->identity, err);
        }
        return rc;
    }
    out->schema_version = YVEX_RUNTIME_MEDIA_PRESET_SCHEMA_V1;
    yvex_core_text_copy(out->name, sizeof(out->name), "released-fl2va-v1");
    yvex_core_text_copy(out->profile, sizeof(out->profile), "released");
    yvex_core_text_copy(out->format, sizeof(out->format), "avi");
    out->width = request->present & YVEX_RUNTIME_MEDIA_EXECUTION_WIDTH
                     ? request->width : host->released_width;
    out->height = request->present & YVEX_RUNTIME_MEDIA_EXECUTION_HEIGHT
                      ? request->height : host->released_height;
    out->frames = host->minimum_frames;
    if (request->present & YVEX_RUNTIME_MEDIA_EXECUTION_DURATION) {
        rc = released_frames_resolve(host, request->duration_milliseconds,
                                     &out->frames, err);
        if (rc != YVEX_OK) return rc;
    }
    out->sigma_grid_points = host->released_sigma_grid_points;
    out->seed = request->present & YVEX_RUNTIME_MEDIA_EXECUTION_SEED
                    ? request->seed : host->default_seed;
    out->complete = 1;
    rc = media_preset_identity(out, out->identity, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_media_execution_preset_validate(host, out, err);
    return rc;
}

static int generation_fail(
    yvex_error *err, yvex_status status, const char *where, const char *message)
{
    yvex_error_set(err, status, where, message);
    return status;
}

static int generation_cancelled(
    const yvex_runtime_av_generation_request *request, yvex_error *err)
{
    if (request->cancel_requested && request->cancel_requested(request->cancel_context))
        return generation_fail(err, YVEX_ERR_CANCELLED, "runtime.av-generation",
                               "audio-video generation was cancelled");
    return YVEX_OK;
}

static int generation_progress(
    const yvex_runtime_av_generation_request *request,
    yvex_runtime_media_progress_kind kind, unsigned long long completed,
    unsigned long long total, unsigned long long value, yvex_error *err)
{
    yvex_runtime_media_progress progress = {0};
    if (!request->observe_progress) return YVEX_OK;
    progress.schema_version = YVEX_RUNTIME_MEDIA_PROGRESS_SCHEMA_V1;
    progress.kind = kind;
    progress.completed = completed;
    progress.total = total;
    progress.value = value;
    return request->observe_progress(request->progress_context, &progress, err);
}

static int latent_progress_observe(
    void *opaque, const yvex_runtime_latent_observation *observation,
    yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = opaque;
    if (!observation ||
        observation->schema_version != YVEX_RUNTIME_LATENT_OBSERVATION_SCHEMA_V1)
        return generation_fail(err, YVEX_ERR_INVALID_ARG,
                               "runtime.av-generation.progress",
                               "typed latent progress is required");
    if (observation->stage != YVEX_RUNTIME_LATENT_OBSERVATION_ADVANCED)
        return YVEX_OK;
    return generation_progress(
        request, YVEX_RUNTIME_MEDIA_PROGRESS_LATENT_STEP,
        observation->completed_steps, request->inference_steps,
        observation->video_values + observation->audio_values, err);
}

static int component_view_open(const char *path, component_view *view, yvex_error *err)
{
    yvex_artifact_options options = {0};
    int rc;
    memset(view, 0, sizeof(*view));
    options.path = path;
    options.readonly = 1;
    rc = yvex_artifact_open(&view->artifact, &options, err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&view->gguf, view->artifact, err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&view->tensors, view->gguf, err);
    return rc;
}

static void component_view_close(component_view *view)
{
    if (!view) return;
    yvex_tensor_table_close(view->tensors);
    yvex_gguf_close(view->gguf);
    yvex_artifact_close(view->artifact);
    memset(view, 0, sizeof(*view));
}

static int host_allocate(
    generation_state *state, unsigned long long count, size_t element_bytes,
    void **out, const char *label, yvex_error *err)
{
    unsigned long long bytes, next;
    if (out) *out = NULL;
    if (!state || !out || !count || !element_bytes ||
        !yvex_core_u64_mul(count, element_bytes, &bytes) || bytes > SIZE_MAX ||
        !yvex_core_u64_add(state->host_live, bytes, &next) ||
        next > state->request->maximum_workspace_bytes ||
        next > state->request->maximum_host_bytes) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, "runtime.av-generation.workspace",
                        "%s exceeds the pipeline workspace budget", label);
        return YVEX_ERR_BOUNDS;
    }
    *out = calloc(1u, (size_t)bytes);
    if (!*out) {
        yvex_error_setf(err, YVEX_ERR_NOMEM, "runtime.av-generation.workspace",
                        "%s allocation failed", label);
        return YVEX_ERR_NOMEM;
    }
    state->host_live = next;
    if (next > state->host_peak) state->host_peak = next;
    return YVEX_OK;
}

static void host_release(
    generation_state *state, void **owned, unsigned long long count, size_t element_bytes)
{
    unsigned long long bytes = 0ull;
    if (!state || !owned || !*owned) return;
    if (yvex_core_u64_mul(count, element_bytes, &bytes) && bytes <= state->host_live)
        state->host_live -= bytes;
    free(*owned);
    *owned = NULL;
}

static void generation_state_close(generation_state *state)
{
    unsigned long long index;
    if (!state) return;
    for (index = 0ull; index < YVEX_RUNTIME_MEDIA_CONDITION_CAP; ++index)
        yvex_image_close(state->condition_images + index);
    host_release(state, (void **)&state->pcm, state->pcm_values, sizeof(float));
    host_release(state, (void **)&state->rgb, state->rgb_values, sizeof(float));
    host_release(state, (void **)&state->audio_latent, state->audio_latent_values, sizeof(float));
    host_release(state, (void **)&state->video_latent, state->video_latent_values, sizeof(float));
    host_release(state, (void **)&state->audio_rows, state->audio_row_values, sizeof(float));
    host_release(state, (void **)&state->video_rows, state->video_row_values, sizeof(float));
    host_release(state, (void **)&state->conditioning,
                 state->conditioning_values, sizeof(float));
    host_release(state, (void **)&state->condition_latents,
                 state->condition_latent_values, sizeof(float));
    host_release(state, (void **)&state->text_tags,
                 state->request ? state->request->maximum_prompt_tokens : 0ull,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->timestep_indices, state->plan.packed_rows,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.text_indices, state->plan.text_tokens,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.audio_indices, state->plan.audio_rows,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.video_indices,
                 state->plan.condition_rows + state->plan.video_rows,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.token_tags, state->plan.packed_rows,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.position_ids, state->layout_position_values,
                 sizeof(float));
}

static int model_contract_validate(
    const yvex_runtime_av_generation_request *request, yvex_error *err)
{
    yvex_runtime_av_generation_request expected;
    int rc;
    if (!request ||
        (request->schema_version != YVEX_RUNTIME_AV_GENERATION_SCHEMA_V1 &&
         request->schema_version != YVEX_RUNTIME_AV_GENERATION_SCHEMA_V2) ||
        !request->target || !request->target[0] ||
        !request->text_artifact_path || !request->transformer_artifact_path ||
        !request->video_artifact_path || !request->audio_artifact_path ||
        !request->source_identity || !yvex_sha256_hex_valid(request->source_identity) ||
        !request->fps_numerator || !request->fps_denominator || !request->audio_sample_rate ||
        !request->conditioning_layers || !request->transformer_blocks ||
        !request->maximum_prompt_tokens || !request->maximum_packed_rows ||
        !request->maximum_host_bytes || !request->maximum_device_bytes ||
        !request->maximum_workspace_bytes || !request->maximum_file_bytes ||
        (request->component_backend != YVEX_BACKEND_KIND_CPU &&
         request->component_backend != YVEX_BACKEND_KIND_CUDA) ||
        !request->video_temporal_ratio || !request->video_clip_length ||
        !request->video_spatial_ratio || !request->video_tile_size ||
        !request->video_mean || !request->video_std || !request->audio_mean ||
        !request->audio_std || !request->pixel_mean || !request->pixel_std ||
        !request->video_channels || !request->audio_channels || !request->pixel_channels ||
        !request->audio_output_channels || request->audio_output_channels > 2ull ||
        !request->audio_samples_per_step || !request->plan_build ||
        !request->layout_build || !request->component_admit ||
        !request->condition || !request->keyframe_encode || !request->latent ||
        !request->video_decode ||
        !request->audio_decode)
        return generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.av-generation",
                               "one exact admitted media model contract is required");
    if (!request->output_semantic_domain && !request->video_output_requirement &&
        !request->audio_output_requirement) {
        if (request->video_output_specialization.physical_identity[0] ||
            request->audio_output_specialization.physical_identity[0])
            return generation_fail(
                err, YVEX_ERR_FORMAT, "runtime.av-generation",
                "media specialization exists without output semantics");
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (!request->output_semantic_domain || !request->video_output_requirement ||
        !request->audio_output_requirement)
        return generation_fail(
            err, YVEX_ERR_FORMAT, "runtime.av-generation",
            "media output semantics are only partially specified");
    expected = *request;
    rc = yvex_runtime_media_request_specialize(
        &expected, request->output_semantic_domain,
        request->video_output_requirement, request->audio_output_requirement, err);
    if (rc != YVEX_OK) return rc;
    if (strcmp(expected.video_output_specialization.physical_identity,
               request->video_output_specialization.physical_identity) != 0 ||
        strcmp(expected.audio_output_specialization.physical_identity,
               request->audio_output_specialization.physical_identity) != 0)
        return generation_fail(
            err, YVEX_ERR_FORMAT, "runtime.av-generation",
            "media request does not carry its compiler-sealed specialization");
    return YVEX_OK;
}

static int request_validate(
    const yvex_runtime_av_generation_request *request, yvex_error *err)
{
    unsigned long long index;
    int first = 0, last = 0;
    int rc = model_contract_validate(request, err);
    if (rc != YVEX_OK) return rc;
    if ((request->schema_version == YVEX_RUNTIME_AV_GENERATION_SCHEMA_V1 &&
         (request->conditions || request->condition_count)) ||
        request->condition_count > YVEX_RUNTIME_MEDIA_CONDITION_CAP ||
        (request->condition_count && !request->conditions))
        return generation_fail(err, YVEX_ERR_INVALID_ARG,
                               "runtime.av-generation.condition",
                               "media conditions require the bounded generation-v2 contract");
    for (index = 0ull; index < request->condition_count; ++index) {
        const yvex_runtime_media_condition *condition = request->conditions + index;
        if (condition->schema_version != YVEX_RUNTIME_MEDIA_CONDITION_SCHEMA_V1 ||
            condition->kind != YVEX_RUNTIME_MEDIA_CONDITION_IMAGE ||
            !condition->source_path || condition->source_path[0] != '/')
            return generation_fail(err, YVEX_ERR_INVALID_ARG,
                                   "runtime.av-generation.condition",
                                   "one admitted absolute image condition is required");
        if (condition->role == YVEX_RUNTIME_MEDIA_CONDITION_FIRST && !first)
            first = 1;
        else if (condition->role == YVEX_RUNTIME_MEDIA_CONDITION_LAST && !last)
            last = 1;
        else
            return generation_fail(err, YVEX_ERR_INVALID_ARG,
                                   "runtime.av-generation.condition",
                                   "media condition roles must be unique first or last anchors");
    }
    if (!request->prompt || !request->prompt[0] || !request->output_path ||
        !request->frames || !request->width || !request->height ||
        !request->inference_steps)
        return generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.av-generation",
                               "one complete bounded media request is required");
    return YVEX_OK;
}

static component_view *media_model_view(
    yvex_runtime_media_model *model, media_component_index component)
{
    if (!model) return NULL;
    if (component == MEDIA_COMPONENT_TRANSFORMER) return &model->transformer;
    if (component == MEDIA_COMPONENT_VIDEO) return &model->video;
    if (component == MEDIA_COMPONENT_AUDIO) return &model->audio;
    return NULL;
}

static int media_component_resource_release(void *opaque, yvex_error *err)
{
    media_resource_release_context *context = opaque;
    component_view *view;
    if (!context || !context->model ||
        context->component >= MEDIA_COMPONENT_COUNT)
        return generation_fail(err, YVEX_ERR_STATE, "runtime.media-resource",
                               "component release context is invalid");
    if (context->component == MEDIA_COMPONENT_TEXT)
        yvex_model_context_close(&context->model->text);
    else {
        view = media_model_view(context->model, context->component);
        component_view_close(view);
    }
    memset(&context->model->resource_handles[context->component], 0,
           sizeof(context->model->resource_handles[context->component]));
    yvex_error_clear(err);
    return YVEX_OK;
}

static int media_model_resources_open(yvex_runtime_media_model *model,
                                      yvex_error *err)
{
    yvex_engine_resource_request resource = {0};
    yvex_engine_resource_summary resources = {0};
    unsigned long long count = 0ull, index;
    int rc;
    model->summary.engine_generation =
        atomic_fetch_add_explicit(&media_engine_generation_counter, 1ull,
                                  memory_order_relaxed) + 1ull;
    if (!model->summary.engine_generation)
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.media-resource",
                               "media engine generation space is exhausted");
    rc = yvex_runtime_resource_catalog_open(
        &model->resources, model->summary.engine_generation,
        model->summary.model_identity, MEDIA_COMPONENT_COUNT + 4ull, err);
    for (index = 0ull; rc == YVEX_OK && index < MEDIA_COMPONENT_COUNT;
         ++index) {
        model->release_contexts[index].model = model;
        model->release_contexts[index].component =
            (media_component_index)index;
        memset(&resource, 0, sizeof(resource));
        resource.kind = YVEX_ENGINE_RESOURCE_COMPONENT;
        resource.owner = YVEX_ENGINE_RESOURCE_OWNER_PACKAGE;
        resource.lifetime = YVEX_ENGINE_RESOURCE_LIFETIME_ENGINE;
        resource.numeric_class =
            YVEX_ENGINE_RESOURCE_NUMERIC_CANONICAL_PACKAGE;
        resource.name = media_component_names[index];
        resource.package_identity = model->admissions[index].artifact_identity;
        resource.admission_identity = model->admissions[index].admission_identity;
        resource.bytes.mapped_package_bytes =
            model->admissions[index].file_bytes;
        resource.value = index == MEDIA_COMPONENT_TEXT
                             ? (void *)&model->text
                             : (void *)media_model_view(
                                   model, (media_component_index)index);
        resource.release = media_component_resource_release;
        resource.release_context = &model->release_contexts[index];
        resource.ready = 1;
        rc = yvex_runtime_resource_register(
            model->resources, &resource, &model->resource_handles[index], err);
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_resource_snapshot(
            model->resources, &resources, NULL, 0ull, &count, err);
    if (rc != YVEX_OK) return rc;
    model->summary.resource_count = count;
    model->summary.resource_generation = resources.generation;
    model->summary.mapped_package_bytes =
        resources.bytes.mapped_package_bytes;
    model->summary.prepared_bytes = resources.bytes.prepared_bytes;
    model->summary.resident_host_bytes =
        resources.bytes.host_resident_bytes;
    model->summary.resident_device_bytes =
        resources.bytes.device_resident_bytes;
    return YVEX_OK;
}

static int media_model_component_admit(
    yvex_runtime_media_model *model, media_component_index component,
    const char *name, const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors,
    const yvex_runtime_media_model_open_options *open_options,
    unsigned long long *artifact_bytes, yvex_error *err)
{
    yvex_artifact_admission_options admission_options = {0};
    yvex_artifact_admission_evidence *evidence =
        &model->summary.components[component].evidence;
    yvex_artifact_admission_failure failure = {0};
    yvex_complete_artifact_admission *admission = model->admissions + component;
    unsigned long long next, started = yvex_core_monotonic_ns(), completed;
    int rc;
    admission_options.schema_version = YVEX_ARTIFACT_ADMISSION_OPTIONS_SCHEMA_V1;
    admission_options.reopen_cache_root =
        open_options ? open_options->artifact_reopen_cache_root : NULL;
    rc = model->contract.component_admit(
        name, artifact, gguf, tensors, &admission_options, admission, evidence,
        &failure, err);
    completed = yvex_core_monotonic_ns();
    evidence->seconds = completed >= started
                            ? (double)(completed - started) / 1000000000.0
                            : 0.0;
    if (rc == YVEX_OK &&
        (!admission->complete ||
         !yvex_sha256_hex_valid(admission->artifact_identity) ||
         !yvex_sha256_hex_valid(admission->logical_component_identity) ||
         !yvex_sha256_hex_valid(admission->admission_identity)))
        rc = generation_fail(err, YVEX_ERR_FORMAT, "runtime.media-model",
                             "component admission did not publish exact identities");
    if (rc == YVEX_OK &&
        !yvex_core_u64_add(*artifact_bytes, yvex_artifact_size(artifact), &next))
        rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.media-model",
                             "component artifact byte accounting overflowed");
    if (rc == YVEX_OK) {
        *artifact_bytes = next;
        yvex_core_text_copy(model->summary.components[component].role,
                            sizeof(model->summary.components[component].role), name);
        if (!yvex_core_u64_add(model->summary.artifact_bytes_hashed,
                               evidence->bytes_hashed, &next))
            rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.media-model",
                                 "component verification byte accounting overflowed");
        else
            model->summary.artifact_bytes_hashed = next;
        if (rc == YVEX_OK && open_options && open_options->observe_component)
            open_options->observe_component(open_options->observer_context, name, evidence);
    }
    return rc;
}

static int media_model_identity(
    yvex_runtime_media_model *model, unsigned long long artifact_bytes,
    yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.media-model.v1") ||
        !yvex_sha256_update_text(&hash, model->contract.target) ||
        !yvex_sha256_update_text(&hash, model->contract.source_identity) ||
        !yvex_sha256_update_u64_be(&hash, MEDIA_COMPONENT_COUNT) ||
        !yvex_sha256_update_u64_be(&hash, artifact_bytes))
        return generation_fail(err, YVEX_ERR_STATE, "runtime.media-model",
                               "media model identity could not start");
    for (index = 0ull; index < MEDIA_COMPONENT_COUNT; ++index)
        if (!yvex_sha256_update_text(&hash, media_component_names[index]) ||
            !yvex_sha256_update_text(
                &hash, model->admissions[index].admission_identity))
            return generation_fail(err, YVEX_ERR_STATE, "runtime.media-model",
                                   "media model component identity could not be sealed");
    if (!yvex_sha256_final(&hash, digest))
        return generation_fail(err, YVEX_ERR_STATE, "runtime.media-model",
                               "media model identity could not finish");
    model->summary.schema_version = YVEX_RUNTIME_MEDIA_MODEL_SCHEMA_V1;
    model->summary.component_count = MEDIA_COMPONENT_COUNT;
    model->summary.artifact_bytes = artifact_bytes;
    yvex_sha256_hex(digest, model->summary.model_identity);
    yvex_core_text_copy(model->summary.source_identity,
                        sizeof(model->summary.source_identity),
                        model->contract.source_identity);
    model->summary.complete = 1;
    return YVEX_OK;
}

static int media_model_values_equal(
    const float *first, const float *second, unsigned long long count)
{
    unsigned long long index;
    if (!first || !second) return 0;
    for (index = 0ull; index < count; ++index)
        if (first[index] != second[index]) return 0;
    return 1;
}

static int media_model_contract_matches(
    const yvex_runtime_media_model *model,
    const yvex_runtime_av_generation_request *request)
{
    const yvex_runtime_av_generation_request *sealed = &model->contract;
    return !strcmp(sealed->target, request->target) &&
           !strcmp(sealed->source_identity, request->source_identity) &&
           !strcmp(sealed->text_artifact_path, request->text_artifact_path) &&
           !strcmp(sealed->transformer_artifact_path, request->transformer_artifact_path) &&
           !strcmp(sealed->video_artifact_path, request->video_artifact_path) &&
           !strcmp(sealed->audio_artifact_path, request->audio_artifact_path) &&
           sealed->fps_numerator == request->fps_numerator &&
           sealed->fps_denominator == request->fps_denominator &&
           sealed->audio_sample_rate == request->audio_sample_rate &&
           sealed->keyframe_encode_seed == request->keyframe_encode_seed &&
           sealed->conditioning_layers == request->conditioning_layers &&
           sealed->transformer_blocks == request->transformer_blocks &&
           sealed->maximum_prompt_tokens == request->maximum_prompt_tokens &&
           sealed->maximum_packed_rows == request->maximum_packed_rows &&
           sealed->maximum_host_bytes == request->maximum_host_bytes &&
           sealed->maximum_device_bytes == request->maximum_device_bytes &&
           sealed->maximum_workspace_bytes == request->maximum_workspace_bytes &&
           sealed->maximum_file_bytes == request->maximum_file_bytes &&
           sealed->component_backend == request->component_backend &&
           (!!sealed->output_semantic_domain == !!request->output_semantic_domain) &&
           (!!sealed->video_output_requirement == !!request->video_output_requirement) &&
           (!!sealed->audio_output_requirement == !!request->audio_output_requirement) &&
           !strcmp(sealed->video_output_specialization.physical_identity,
                   request->video_output_specialization.physical_identity) &&
           !strcmp(sealed->audio_output_specialization.physical_identity,
                   request->audio_output_specialization.physical_identity) &&
           sealed->video_temporal_ratio == request->video_temporal_ratio &&
           sealed->video_clip_length == request->video_clip_length &&
           sealed->video_token_drop == request->video_token_drop &&
           sealed->video_spatial_ratio == request->video_spatial_ratio &&
           sealed->video_tile_size == request->video_tile_size &&
           sealed->video_minimum_tile_overlap == request->video_minimum_tile_overlap &&
           sealed->video_channels == request->video_channels &&
           sealed->audio_channels == request->audio_channels &&
           sealed->pixel_channels == request->pixel_channels &&
           media_model_values_equal(
               sealed->video_mean, request->video_mean, sealed->video_channels) &&
           media_model_values_equal(
               sealed->video_std, request->video_std, sealed->video_channels) &&
           media_model_values_equal(
               sealed->audio_mean, request->audio_mean, sealed->audio_channels) &&
           media_model_values_equal(
               sealed->audio_std, request->audio_std, sealed->audio_channels) &&
           media_model_values_equal(
               sealed->pixel_mean, request->pixel_mean, sealed->pixel_channels) &&
           media_model_values_equal(
               sealed->pixel_std, request->pixel_std, sealed->pixel_channels) &&
           sealed->audio_output_channels == request->audio_output_channels &&
           sealed->audio_samples_per_step == request->audio_samples_per_step &&
           sealed->plan_build == request->plan_build &&
           sealed->layout_build == request->layout_build &&
           sealed->component_admit == request->component_admit &&
           sealed->condition == request->condition &&
           sealed->keyframe_encode == request->keyframe_encode &&
           sealed->latent == request->latent &&
           sealed->video_decode == request->video_decode &&
           sealed->audio_decode == request->audio_decode;
}

int yvex_runtime_media_model_open(
    yvex_runtime_media_model **out,
    const yvex_runtime_av_generation_request *request,
    const yvex_runtime_media_model_open_options *open_options,
    yvex_runtime_media_model_summary *summary, yvex_error *err)
{
    const char *paths[MEDIA_COMPONENT_COUNT];
    yvex_runtime_media_model *model = NULL;
    unsigned long long artifact_bytes = 0ull, index;
    int rc;
    if (out) *out = NULL;
    if (summary) memset(summary, 0, sizeof(*summary));
    rc = model_contract_validate(request, err);
    if (rc == YVEX_OK && open_options &&
        open_options->schema_version != YVEX_RUNTIME_MEDIA_MODEL_OPEN_SCHEMA_V1)
        rc = generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.media-model",
                             "media model-open options schema is unsupported");
    if (rc != YVEX_OK || !out || !summary) {
        if (rc == YVEX_OK)
            rc = generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.media-model",
                                 "media model and summary outputs are required");
        return rc;
    }
    model = calloc(1u, sizeof(*model));
    if (!model)
        return generation_fail(err, YVEX_ERR_NOMEM, "runtime.media-model",
                               "media model allocation failed");
    model->contract = *request;
    paths[MEDIA_COMPONENT_TEXT] = request->text_artifact_path;
    paths[MEDIA_COMPONENT_TRANSFORMER] = request->transformer_artifact_path;
    paths[MEDIA_COMPONENT_VIDEO] = request->video_artifact_path;
    paths[MEDIA_COMPONENT_AUDIO] = request->audio_artifact_path;
    rc = yvex_model_context_open(paths[MEDIA_COMPONENT_TEXT], &model->text, err);
    if (rc == YVEX_OK) {
        rc = yvex_family_tokenizer_open(&model->text.tokenizer, model->text.gguf, err);
        if (rc == YVEX_ERR_UNSUPPORTED) {
            yvex_error_clear(err);
            rc = yvex_tokenizer_from_gguf(
                &model->text.tokenizer, model->text.gguf, model->text.model, err);
        }
    }
    if (rc == YVEX_OK)
        rc = media_model_component_admit(
            model, MEDIA_COMPONENT_TEXT,
            media_component_names[MEDIA_COMPONENT_TEXT],
            model->text.artifact, model->text.gguf, model->text.table,
            open_options, &artifact_bytes, err);
    for (index = MEDIA_COMPONENT_TRANSFORMER;
         rc == YVEX_OK && index < MEDIA_COMPONENT_COUNT; ++index) {
        component_view *view = media_model_view(model, (media_component_index)index);
        rc = component_view_open(paths[index], view, err);
        if (rc == YVEX_OK)
            rc = media_model_component_admit(
                model, (media_component_index)index,
                media_component_names[index],
                view->artifact, view->gguf, view->tensors, open_options,
                &artifact_bytes, err);
    }
    if (rc == YVEX_OK) rc = media_model_identity(model, artifact_bytes, err);
    if (rc == YVEX_OK) rc = media_model_resources_open(model, err);
    if (rc != YVEX_OK) {
        yvex_runtime_media_model_close(&model);
        return rc;
    }
    *summary = model->summary;
    *out = model;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_runtime_media_model_close(yvex_runtime_media_model **model)
{
    yvex_runtime_media_model *owner;
    yvex_error cleanup = {0};
    if (!model || !*model) return;
    owner = *model;
    if (yvex_runtime_resource_catalog_close(&owner->resources, &cleanup) !=
        YVEX_OK)
        return;
    /* Unregistered components can remain after a partial open. */
    component_view_close(&owner->audio);
    component_view_close(&owner->video);
    component_view_close(&owner->transformer);
    yvex_model_context_close(&owner->text);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *model = NULL;
}

static int conditioning_component_execute(
    generation_state *state, yvex_media_conditioning_request *request, yvex_error *err)
{
    const yvex_runtime_av_generation_request *generation = state->request;
    yvex_model_context *text = &state->model->text;
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution component = {0};
    yvex_error cleanup;
    int rc, cleanup_rc;
    rc = yvex_runtime_component_session_open(
        &session, state->model->admissions + MEDIA_COMPONENT_TEXT,
        text->artifact, text->gguf, text->table, generation->component_backend,
        generation->maximum_host_bytes, generation->maximum_device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &component, err);
    request->text_component = rc == YVEX_OK ? &component : NULL;
    if (rc == YVEX_OK)
        rc = generation->condition(request, &state->conditioning_result, err);
    request->text_component = NULL;
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    return rc;
}

static int keyframe_component_execute(
    generation_state *state, yvex_media_keyframe_request *request, yvex_error *err)
{
    const yvex_runtime_av_generation_request *generation = state->request;
    component_view *video = &state->model->video;
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution component = {0};
    yvex_error cleanup;
    int rc, cleanup_rc;
    rc = yvex_runtime_component_session_open(
        &session, state->model->admissions + MEDIA_COMPONENT_VIDEO,
        video->artifact, video->gguf, video->tensors, generation->component_backend,
        generation->maximum_host_bytes, generation->maximum_device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &component, err);
    request->video_component = rc == YVEX_OK ? &component : NULL;
    if (rc == YVEX_OK)
        rc = generation->keyframe_encode(request, &state->keyframe_result, err);
    request->video_component = NULL;
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    return rc;
}

static int conditioning_execute(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    yvex_model_context *text = &state->model->text;
    yvex_media_conditioning_request conditioning = {0};
    yvex_media_keyframe_request keyframes = {0};
    unsigned long long index, latent_height = 0ull, latent_width = 0ull;
    int rc = generation_cancelled(request, err);

    for (index = 0ull; rc == YVEX_OK && index < request->condition_count; ++index)
        rc = yvex_image_decode_file(state->condition_images + index,
                                    request->conditions[index].source_path,
                                    request->maximum_file_bytes, err);
    if (rc == YVEX_OK &&
        !yvex_core_u64_mul(request->maximum_prompt_tokens, 5120ull,
                           &state->conditioning_values))
        rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.conditioning",
                             "conditioning capacity overflowed");
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->conditioning_values, sizeof(float),
                           (void **)&state->conditioning, "conditioning", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, request->maximum_prompt_tokens, sizeof(unsigned int),
                           (void **)&state->text_tags, "conditioning tags", err);
    conditioning = (yvex_media_conditioning_request){
        .schema_version = YVEX_MEDIA_CONDITIONING_SCHEMA_V2,
        .prompt = request->prompt, .tokenizer = text->tokenizer,
        .conditions = request->conditions, .condition_images = state->condition_images,
        .condition_count = request->condition_count, .width = request->width,
        .height = request->height, .layer_count = request->conditioning_layers,
        .maximum_prompt_tokens = request->maximum_prompt_tokens,
        .conditioning = state->conditioning,
        .conditioning_capacity = state->conditioning_values,
        .text_tags = state->text_tags,
        .text_tag_capacity = request->maximum_prompt_tokens,
    };
    if (rc == YVEX_OK)
        rc = conditioning_component_execute(state, &conditioning, err);
    if (rc == YVEX_OK &&
        (!state->conditioning_result.complete ||
         !state->conditioning_result.token_count ||
         state->conditioning_result.token_count > request->maximum_prompt_tokens ||
         state->conditioning_result.hidden_width != 5120ull ||
         state->conditioning_result.condition_count != request->condition_count ||
         !yvex_sha256_hex_valid(state->conditioning_result.prompt_identity) ||
         !yvex_sha256_hex_valid(state->conditioning_result.execution_identity)))
        rc = generation_fail(err, YVEX_ERR_STATE, "runtime.av-generation.conditioning",
                             "family conditioning returned incomplete execution evidence");
    if (rc == YVEX_OK)
        yvex_core_text_copy(state->prompt_identity, sizeof(state->prompt_identity),
                            state->conditioning_result.prompt_identity);
    if (rc == YVEX_OK && request->condition_count) {
        if (request->width % request->video_spatial_ratio ||
            request->height % request->video_spatial_ratio)
            rc = generation_fail(err, YVEX_ERR_FORMAT, "runtime.av-generation.keyframe",
                                 "keyframe canvas is incompatible with Visual VAE geometry");
        else {
            latent_width = request->width / request->video_spatial_ratio;
            latent_height = request->height / request->video_spatial_ratio;
        }
    }
    if (rc == YVEX_OK && request->condition_count &&
        (!yvex_core_u64_mul(request->condition_count, request->video_channels,
                            &state->condition_latent_values) ||
         !yvex_core_u64_mul(state->condition_latent_values, latent_height,
                            &state->condition_latent_values) ||
         !yvex_core_u64_mul(state->condition_latent_values, latent_width,
                            &state->condition_latent_values)))
        rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.keyframe",
                             "keyframe latent capacity overflowed");
    if (rc == YVEX_OK && request->condition_count)
        rc = host_allocate(state, state->condition_latent_values, sizeof(float),
                           (void **)&state->condition_latents, "keyframe latents", err);
    keyframes = (yvex_media_keyframe_request){
        .schema_version = YVEX_MEDIA_CONDITIONING_SCHEMA_V2,
        .conditions = request->conditions, .condition_images = state->condition_images,
        .condition_count = request->condition_count, .width = request->width,
        .height = request->height, .posterior_seed = request->keyframe_encode_seed,
        .pixel_mean = request->pixel_mean, .pixel_std = request->pixel_std,
        .latent_mean = request->video_mean, .latent_std = request->video_std,
        .pixel_channels = request->pixel_channels,
        .latent_channels = request->video_channels,
        .condition_latents = state->condition_latents,
        .condition_latent_capacity = state->condition_latent_values,
    };
    if (rc == YVEX_OK && request->condition_count)
        rc = keyframe_component_execute(state, &keyframes, err);
    if (rc == YVEX_OK && request->condition_count &&
        (!state->keyframe_result.complete ||
         state->keyframe_result.condition_count != request->condition_count ||
         state->keyframe_result.latent_values != state->condition_latent_values ||
         !yvex_sha256_hex_valid(state->keyframe_result.execution_identity)))
        rc = generation_fail(err, YVEX_ERR_STATE, "runtime.av-generation.keyframe",
                             "Visual VAE encoder returned incomplete keyframe evidence");
    if (rc == YVEX_OK && request->condition_count) {
        state->conditioning_result.condition_rows = state->keyframe_result.condition_rows;
        yvex_core_text_copy(state->conditioning_result.condition_identity,
                            sizeof(state->conditioning_result.condition_identity),
                            state->keyframe_result.execution_identity);
    }
    return rc;
}

static int plan_and_layout_build(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    yvex_media_plan_request plan_request = {
        .schema_version = YVEX_RUNTIME_AV_PLAN_SCHEMA_V1,
        .text_tokens = state->conditioning_result.token_count,
        .width = request->width, .height = request->height,
        .frames = request->frames, .inference_steps = request->inference_steps,
        .text_tags = state->text_tags, .conditions = request->conditions,
        .condition_count = request->condition_count,
        .condition_rows = state->keyframe_result.condition_rows,
    };
    yvex_media_layout_request layout_request = {
        .text_tags = state->text_tags, .conditions = request->conditions,
        .condition_count = request->condition_count,
    };
    int rc = request->plan_build(&state->plan, &plan_request, err);
    if (rc == YVEX_OK && state->plan.packed_rows > request->maximum_packed_rows)
        rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.plan",
                             "packed AV plan exceeds the admitted execution capacity");
    if (rc == YVEX_OK &&
        !yvex_core_u64_mul(state->plan.packed_rows, 3ull, &state->layout_position_values))
        rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.layout",
                             "layout position extent overflowed");
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->layout_position_values, sizeof(float),
                           (void **)&state->layout.position_ids, "layout positions", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.packed_rows, sizeof(unsigned int),
                           (void **)&state->layout.token_tags, "layout tags", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.condition_rows + state->plan.video_rows,
                           sizeof(unsigned int),
                           (void **)&state->layout.video_indices, "video indices", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.audio_rows, sizeof(unsigned int),
                           (void **)&state->layout.audio_indices, "audio indices", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.text_tokens, sizeof(unsigned int),
                           (void **)&state->layout.text_indices, "text indices", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.packed_rows, sizeof(unsigned int),
                           (void **)&state->timestep_indices, "timestep indices", err);
    state->layout.position_capacity = state->plan.packed_rows * 3ull;
    state->layout.tag_capacity = state->plan.packed_rows;
    state->layout.video_capacity = state->plan.condition_rows + state->plan.video_rows;
    state->layout.audio_capacity = state->plan.audio_rows;
    state->layout.text_capacity = state->plan.text_tokens;
    if (rc == YVEX_OK)
        layout_request.plan = &state->plan;
    if (rc == YVEX_OK)
        rc = request->layout_build(&layout_request, &state->layout,
                                   &state->layout_result, err);
    return rc;
}

static int latent_execute(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    component_view *view = &state->model->transformer;
    const yvex_complete_artifact_admission *admission =
        state->model->admissions + MEDIA_COMPONENT_TRANSFORMER;
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution component = {0};
    yvex_runtime_av_latent_context context = {0};
    yvex_error cleanup;
    int rc, cleanup_rc;
    if (!yvex_core_u64_mul(state->plan.video_rows, state->plan.video_value_width,
                           &state->video_row_values) ||
        !yvex_core_u64_mul(state->plan.audio_rows, state->plan.audio_value_width,
                           &state->audio_row_values))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.latent",
                               "paired latent extent overflowed");
    rc = host_allocate(state, state->video_row_values, sizeof(float),
                       (void **)&state->video_rows, "video latent rows", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->audio_row_values, sizeof(float),
                           (void **)&state->audio_rows, "audio latent rows", err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, admission, view->artifact, view->gguf, view->tensors,
            request->component_backend, request->maximum_host_bytes,
            request->maximum_device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &component, err);
    context.transformer_component = rc == YVEX_OK ? &component : NULL;
    context.conditioning = state->conditioning;
    context.conditioning_capacity = state->conditioning_values;
    context.condition_latents = state->condition_latents;
    context.condition_latent_capacity = state->condition_latent_values;
    context.keyframes = &state->keyframe_result;
    context.conditions = request->conditions;
    context.condition_count = request->condition_count;
    context.conditioning_identity = state->conditioning_result.execution_identity;
    context.layout = &state->layout;
    context.layout_result = &state->layout_result;
    context.video_output_specialization =
        request->output_semantic_domain
            ? &request->video_output_specialization : NULL;
    context.audio_output_specialization =
        request->output_semantic_domain
            ? &request->audio_output_specialization : NULL;
    context.timestep_indices = state->timestep_indices;
    context.timestep_capacity = state->plan.packed_rows;
    context.block_count = request->transformer_blocks;
    context.cancelled = request->cancel_requested;
    context.cancellation_context = request->cancel_context;
    context.observe = latent_progress_observe;
    context.observer_context = (void *)request;
    if (rc == YVEX_OK)
        rc = request->latent(
            &state->plan, &context, request->seed,
            request->maximum_workspace_bytes, state->video_rows, state->video_row_values,
            state->audio_rows, state->audio_row_values, &state->latent_result,
            &state->evaluator_result, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    if (rc == YVEX_OK &&
        (!state->latent_result.completed || !state->evaluator_result.complete ||
         !yvex_sha256_hex_valid(state->latent_result.execution_identity)))
        rc = generation_fail(err, YVEX_ERR_STATE, "runtime.av-generation.latent",
                             "paired latent execution returned incomplete evidence");
    return rc;
}

static int latent_unpack(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    yvex_runtime_av_unpack_request unpack = {0};
    yvex_runtime_av_unpack_output output = {0};
    int rc;
    if (!yvex_core_u64_mul(request->video_channels, state->plan.video_latent_frames,
                           &state->video_latent_values) ||
        !yvex_core_u64_mul(state->video_latent_values, state->plan.video_latent_height,
                           &state->video_latent_values) ||
        !yvex_core_u64_mul(state->video_latent_values, state->plan.video_latent_width,
                           &state->video_latent_values) ||
        !yvex_core_u64_mul(request->audio_output_channels, request->audio_channels,
                           &state->audio_latent_values) ||
        !yvex_core_u64_mul(state->audio_latent_values, state->plan.audio_latent_steps,
                           &state->audio_latent_values))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.unpack",
                               "component latent extent overflowed");
    rc = host_allocate(state, state->video_latent_values, sizeof(float),
                       (void **)&state->video_latent, "Visual VAE latent", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->audio_latent_values, sizeof(float),
                           (void **)&state->audio_latent, "Audio VAE latent", err);
    unpack.schema_version = YVEX_RUNTIME_AV_UNPACK_SCHEMA_V1;
    unpack.plan = &state->plan;
    unpack.video_rows = state->video_rows;
    unpack.audio_rows = state->audio_rows;
    unpack.video_row_capacity = state->video_row_values;
    unpack.audio_row_capacity = state->audio_row_values;
    unpack.video_channel_mean = request->video_mean;
    unpack.video_channel_std = request->video_std;
    unpack.audio_channel_mean = request->audio_mean;
    unpack.audio_channel_std = request->audio_std;
    unpack.video_channel_count = request->video_channels;
    unpack.audio_channel_count = request->audio_channels;
    unpack.maximum_workspace_bytes = request->maximum_workspace_bytes;
    unpack.latent_execution_identity = state->latent_result.execution_identity;
    output.video = state->video_latent;
    output.audio = state->audio_latent;
    output.video_capacity = state->video_latent_values;
    output.audio_capacity = state->audio_latent_values;
    if (rc == YVEX_OK) rc = yvex_runtime_av_unpack(&unpack, &output, &state->unpack_result, err);
    return rc;
}

static int video_decode(
    void *opaque, const yvex_runtime_av_video_decode_window *window,
    yvex_runtime_av_video_decode_evidence *evidence, yvex_error *err)
{
    video_decode_context *context = opaque;
    yvex_runtime_av_video_decode_options options = {0};
    yvex_runtime_av_video_decode_result result = {0};
    yvex_component_execution_failure failure = {0};
    int rc;
    options.latent = window->latent;
    options.output = window->output;
    options.batch = 1ull;
    options.latent_channels = window->latent_channels;
    options.latent_frames = window->latent_frames;
    options.latent_height = window->latent_height;
    options.latent_width = window->latent_width;
    options.output_capacity = window->output_capacity;
    options.max_workspace_bytes = context->request->maximum_workspace_bytes;
    options.cancelled = context->request->cancel_requested;
    options.cancellation_context = context->request->cancel_context;
    rc = context->request->video_decode(
        context->component, &options, &result, &failure, err);
    if (rc == YVEX_OK) {
        evidence->output_values = result.output_values;
        evidence->kernel_launches = result.kernel_launches;
        evidence->h2d_bytes = result.h2d_bytes;
        evidence->d2h_bytes = result.d2h_bytes;
        evidence->device_bytes = result.device_bytes;
        yvex_core_text_copy(evidence->execution_identity,
                            sizeof(evidence->execution_identity),
                            result.execution_identity);
        evidence->complete = result.complete;
    }
    return rc;
}

static int video_execute(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    yvex_runtime_av_video_reconstruction_request plan_request = {0};
    yvex_runtime_av_video_reconstruction_plan plan;
    yvex_runtime_av_video_reconstruction_execution execution = {0};
    component_view *view = &state->model->video;
    const yvex_complete_artifact_admission *admission =
        state->model->admissions + MEDIA_COMPONENT_VIDEO;
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution component = {0};
    video_decode_context context = {request, &component};
    yvex_error cleanup;
    int rc, cleanup_rc;
    if (!yvex_core_u64_mul(3ull, request->frames, &state->rgb_values) ||
        !yvex_core_u64_mul(state->rgb_values, request->height, &state->rgb_values) ||
        !yvex_core_u64_mul(state->rgb_values, request->width, &state->rgb_values))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.video",
                               "decoded RGB extent overflowed");
    rc = host_allocate(state, state->rgb_values, sizeof(float),
                       (void **)&state->rgb, "decoded RGB", err);
    plan_request.schema_version = YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1;
    plan_request.frames = request->frames;
    plan_request.width = request->width;
    plan_request.height = request->height;
    plan_request.latent_frames = state->plan.video_latent_frames;
    plan_request.latent_height = state->plan.video_latent_height;
    plan_request.latent_width = state->plan.video_latent_width;
    plan_request.temporal_ratio = request->video_temporal_ratio;
    plan_request.clip_length = request->video_clip_length;
    plan_request.token_drop = request->video_token_drop;
    plan_request.spatial_ratio = request->video_spatial_ratio;
    plan_request.tile_size = request->video_tile_size;
    plan_request.minimum_tile_overlap = request->video_minimum_tile_overlap;
    plan_request.source_identity = request->source_identity;
    if (rc == YVEX_OK)
        rc = yvex_runtime_av_video_reconstruction_plan_build(&plan_request, &plan, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, admission, view->artifact, view->gguf, view->tensors,
            request->component_backend, request->maximum_host_bytes,
            request->maximum_device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &component, err);
    execution.schema_version = YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1;
    execution.plan = &plan;
    execution.latent = state->video_latent;
    execution.latent_channels = request->video_channels;
    execution.latent_capacity = state->video_latent_values;
    execution.maximum_workspace_bytes = request->maximum_workspace_bytes;
    execution.output_channel_mean = request->pixel_mean;
    execution.output_channel_std = request->pixel_std;
    execution.output_channel_count = request->pixel_channels;
    execution.decode = video_decode;
    execution.decode_context = &context;
    execution.cancel_requested = request->cancel_requested;
    execution.cancel_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_av_video_reconstruct(
            &execution, state->rgb, state->rgb_values, &state->video_result, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    return rc;
}

static int audio_execute(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    component_view *view = &state->model->audio;
    const yvex_complete_artifact_admission *admission =
        state->model->admissions + MEDIA_COMPONENT_AUDIO;
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution component = {0};
    yvex_runtime_av_audio_decode_options options = {0};
    yvex_component_execution_failure failure = {0};
    unsigned long long expected_samples;
    yvex_error cleanup;
    int rc, cleanup_rc;
    if (!yvex_core_u64_mul(request->audio_output_channels,
                           state->plan.audio_latent_steps, &state->pcm_values) ||
        !yvex_core_u64_mul(state->pcm_values, request->audio_samples_per_step,
                           &state->pcm_values))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.audio",
                               "decoded PCM extent overflowed");
    rc = host_allocate(state, state->pcm_values, sizeof(float),
                       (void **)&state->pcm, "decoded PCM", err);
    options.latent = state->audio_latent;
    options.batch = request->audio_output_channels;
    options.latent_channels = request->audio_channels;
    options.latent_steps = state->plan.audio_latent_steps;
    options.output = state->pcm;
    options.output_capacity = state->pcm_values;
    options.max_workspace_bytes = request->maximum_workspace_bytes;
    options.cancelled = request->cancel_requested;
    options.cancellation_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, admission, view->artifact, view->gguf, view->tensors,
            request->component_backend, request->maximum_host_bytes,
            request->maximum_device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &component, err);
    if (rc == YVEX_OK)
        rc = request->audio_decode(
            &component, &options, &state->audio_result, &failure, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    if (rc == YVEX_OK &&
        !yvex_core_u64_mul(state->plan.audio_latent_steps,
                           request->audio_samples_per_step, &expected_samples))
        rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.audio",
                             "decoded PCM evidence extent overflowed");
    if (rc == YVEX_OK &&
        (!state->audio_result.complete ||
         state->audio_result.samples_per_channel != expected_samples))
        rc = generation_fail(err, YVEX_ERR_STATE, "runtime.av-generation.audio",
                             "Audio VAE returned incomplete decoded evidence");
    return rc;
}

static int phase_identity(generation_state *state, yvex_error *err)
{
    const char *identities[] = {
        state->prompt_identity, state->conditioning_result.execution_identity,
        state->plan.identity, state->layout_result.layout_identity,
        state->latent_result.execution_identity, state->unpack_result.input_identity,
        state->video_result.execution_identity, state->audio_result.execution_identity,
    };
    const unsigned long long facts[] = {
        state->request->frames, state->request->width, state->request->height,
        state->request->inference_steps, state->request->transformer_blocks,
        state->request->seed,
    };
    return yvex_runtime_latent_binding_identity(
        "yvex.runtime.av-generation.staged.v1", identities,
        sizeof(identities) / sizeof(identities[0]), facts,
        sizeof(facts) / sizeof(facts[0]), state->execution_identity, err);
}

static int media_publish(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    yvex_media_avi_request media = {0};
    int rc = phase_identity(state, err);
    media.schema_version = YVEX_MEDIA_AVI_SCHEMA_V1;
    media.path = request->output_path;
    media.video = state->rgb;
    media.audio = state->pcm;
    media.video_channels = 3ull;
    media.frames = request->frames;
    media.width = request->width;
    media.height = request->height;
    media.fps_numerator = request->fps_numerator;
    media.fps_denominator = request->fps_denominator;
    media.audio_channels = request->audio_output_channels;
    media.audio_samples = state->audio_result.samples_per_channel;
    media.audio_sample_rate = request->audio_sample_rate;
    media.maximum_file_bytes = request->maximum_file_bytes;
    media.video_identity = state->video_result.execution_identity;
    media.audio_identity = state->audio_result.execution_identity;
    media.execution_identity = state->execution_identity;
    media.cancel_requested = request->cancel_requested;
    media.cancel_context = request->cancel_context;
    if (rc == YVEX_OK) rc = yvex_media_avi_publish(&media, &state->media_result, err);
    return rc;
}

static int result_publish(
    const generation_state *state, yvex_runtime_av_generation_result *result,
    yvex_error *err)
{
    yvex_sha256 trajectory, rng;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    unsigned long long launches;
    result->schema_version = YVEX_RUNTIME_AV_GENERATION_SCHEMA_V2;
    result->prompt_tokens = state->conditioning_result.token_count;
    result->frames = state->request->frames;
    result->width = state->request->width;
    result->height = state->request->height;
    result->audio_samples = state->media_result.audio_samples_used;
    result->model_evaluations = state->evaluator_result.model_evaluations;
    if (!yvex_core_u64_add(state->conditioning_result.kernel_launches,
                           state->evaluator_result.kernel_launches, &launches) ||
        !yvex_core_u64_add(launches, state->video_result.kernel_launches, &launches) ||
        !yvex_core_u64_add(launches, state->audio_result.kernel_launches, &launches))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.result",
                               "kernel launch evidence overflowed");
    result->kernel_launches = launches;
    result->peak_device_bytes = state->evaluator_result.peak_device_bytes;
    if (state->video_result.peak_device_bytes > result->peak_device_bytes)
        result->peak_device_bytes = state->video_result.peak_device_bytes;
    if (state->audio_result.device_bytes > result->peak_device_bytes)
        result->peak_device_bytes = state->audio_result.device_bytes;
    result->peak_workspace_bytes = state->host_peak;
    if (state->conditioning_result.peak_workspace_bytes > result->peak_workspace_bytes)
        result->peak_workspace_bytes = state->conditioning_result.peak_workspace_bytes;
    if (state->latent_result.peak_workspace_bytes > result->peak_workspace_bytes)
        result->peak_workspace_bytes = state->latent_result.peak_workspace_bytes;
    if (state->video_result.peak_workspace_bytes > result->peak_workspace_bytes)
        result->peak_workspace_bytes = state->video_result.peak_workspace_bytes;
    if (state->audio_result.peak_workspace_bytes > result->peak_workspace_bytes)
        result->peak_workspace_bytes = state->audio_result.peak_workspace_bytes;
    result->file_bytes = state->media_result.file_bytes;
    yvex_core_text_copy(result->prompt_identity, sizeof(result->prompt_identity),
                        state->prompt_identity);
    yvex_core_text_copy(result->conditioning_identity, sizeof(result->conditioning_identity),
                        state->conditioning_result.execution_identity);
    yvex_core_text_copy(result->plan_identity, sizeof(result->plan_identity),
                        state->plan.identity);
    yvex_sha256_init(&trajectory);
    if (!yvex_sha256_update_text(&trajectory,
                                 "yvex.runtime.media-trajectory.v1") ||
        !yvex_sha256_update_u64(&trajectory, state->plan.sigma_grid_points) ||
        !yvex_sha256_update_u64(&trajectory, state->plan.model_evaluations))
        return generation_fail(err, YVEX_ERR_STATE,
                               "runtime.av-generation.result",
                               "trajectory identity could not start");
    for (index = 0ull; index < state->plan.sigma_grid_points; ++index) {
        uint32_t video_bits, audio_bits;
        memcpy(&video_bits, state->plan.video_sigmas + index,
               sizeof(video_bits));
        memcpy(&audio_bits, state->plan.audio_sigmas + index,
               sizeof(audio_bits));
        if (!yvex_sha256_update_u64(&trajectory, video_bits) ||
            !yvex_sha256_update_u64(&trajectory, audio_bits))
            return generation_fail(err, YVEX_ERR_STATE,
                                   "runtime.av-generation.result",
                                   "trajectory identity facts failed");
    }
    if (!yvex_sha256_final(&trajectory, digest))
        return generation_fail(err, YVEX_ERR_STATE,
                               "runtime.av-generation.result",
                               "trajectory identity could not finish");
    yvex_sha256_hex(digest, result->trajectory_identity);
    yvex_sha256_init(&rng);
    if (!yvex_sha256_update_text(&rng, "yvex.runtime.media-rng.v1") ||
        !yvex_sha256_update_u64(&rng, state->request->seed) ||
        !yvex_sha256_update_u64(&rng, state->request->keyframe_encode_seed) ||
        !yvex_sha256_update_u64(&rng, state->request->condition_count) ||
        !yvex_sha256_update_text(&rng, state->latent_result.initial_state_identity) ||
        !yvex_sha256_final(&rng, digest))
        return generation_fail(err, YVEX_ERR_STATE,
                               "runtime.av-generation.result",
                               "media RNG identity could not seal");
    yvex_sha256_hex(digest, result->rng_identity);
    yvex_core_text_copy(result->layout_identity, sizeof(result->layout_identity),
                        state->layout_result.layout_identity);
    yvex_core_text_copy(result->latent_identity, sizeof(result->latent_identity),
                        state->latent_result.execution_identity);
    yvex_core_text_copy(result->vae_input_identity, sizeof(result->vae_input_identity),
                        state->unpack_result.input_identity);
    yvex_core_text_copy(result->video_identity, sizeof(result->video_identity),
                        state->video_result.execution_identity);
    yvex_core_text_copy(result->audio_identity, sizeof(result->audio_identity),
                        state->audio_result.execution_identity);
    yvex_core_text_copy(result->execution_identity, sizeof(result->execution_identity),
                        state->execution_identity);
    yvex_core_text_copy(result->file_identity, sizeof(result->file_identity),
                        state->media_result.file_identity);
    yvex_core_text_copy(result->publication_identity, sizeof(result->publication_identity),
                        state->media_result.publication_identity);
    result->complete = 1;
    return YVEX_OK;
}

int yvex_runtime_media_model_generate(
    yvex_runtime_media_model *model,
    const yvex_runtime_av_generation_request *request,
    yvex_runtime_av_generation_result *result, yvex_error *err)
{
    generation_state state = {0};
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    rc = request_validate(request, err);
    if (rc == YVEX_OK && (!model || !model->summary.complete ||
                          !media_model_contract_matches(model, request)))
        rc = generation_fail(err, YVEX_ERR_FORMAT, "runtime.media-model",
                             "request differs from the opened media model contract");
    if (rc != YVEX_OK || !result) {
        if (rc == YVEX_OK)
            rc = generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.av-generation",
                                 "generation result is required");
        return rc;
    }
    state.model = model;
    state.request = request;
    rc = generation_progress(request, YVEX_RUNTIME_MEDIA_PROGRESS_CONDITIONING_START,
                             0ull, 0ull, 0ull, err);
    if (rc == YVEX_OK) rc = conditioning_execute(&state, err);
    if (rc == YVEX_OK)
        rc = generation_progress(
            request, YVEX_RUNTIME_MEDIA_PROGRESS_CONDITIONING_COMPLETE,
            state.conditioning_result.token_count,
            state.conditioning_result.token_count,
            state.conditioning_result.kernel_launches, err);
    if (rc == YVEX_OK) rc = plan_and_layout_build(&state, err);
    if (rc == YVEX_OK) rc = generation_cancelled(request, err);
    if (rc == YVEX_OK)
        rc = generation_progress(request, YVEX_RUNTIME_MEDIA_PROGRESS_LATENT_START,
                                 0ull, request->inference_steps,
                                 state.plan.packed_rows, err);
    if (rc == YVEX_OK) rc = latent_execute(&state, err);
    if (rc == YVEX_OK)
        rc = generation_progress(
            request, YVEX_RUNTIME_MEDIA_PROGRESS_LATENT_COMPLETE,
            request->inference_steps, request->inference_steps,
            state.evaluator_result.model_evaluations, err);
    if (rc == YVEX_OK) rc = latent_unpack(&state, err);
    host_release(&state, (void **)&state.video_rows, state.video_row_values, sizeof(float));
    host_release(&state, (void **)&state.audio_rows, state.audio_row_values, sizeof(float));
    host_release(&state, (void **)&state.conditioning,
                 state.conditioning_values, sizeof(float));
    host_release(&state, (void **)&state.condition_latents,
                 state.condition_latent_values, sizeof(float));
    host_release(&state, (void **)&state.text_tags,
                 request->maximum_prompt_tokens, sizeof(unsigned int));
    if (rc == YVEX_OK) rc = generation_cancelled(request, err);
    if (rc == YVEX_OK)
        rc = generation_progress(request, YVEX_RUNTIME_MEDIA_PROGRESS_VIDEO_START,
                                 0ull, request->frames, 0ull, err);
    if (rc == YVEX_OK) rc = video_execute(&state, err);
    if (rc == YVEX_OK)
        rc = generation_progress(request, YVEX_RUNTIME_MEDIA_PROGRESS_VIDEO_COMPLETE,
                                 request->frames, request->frames,
                                 state.video_result.kernel_launches, err);
    host_release(&state, (void **)&state.video_latent,
                 state.video_latent_values, sizeof(float));
    if (rc == YVEX_OK) rc = generation_cancelled(request, err);
    if (rc == YVEX_OK)
        rc = generation_progress(request, YVEX_RUNTIME_MEDIA_PROGRESS_AUDIO_START,
                                 0ull, 0ull, 0ull, err);
    if (rc == YVEX_OK) rc = audio_execute(&state, err);
    if (rc == YVEX_OK)
        rc = generation_progress(
            request, YVEX_RUNTIME_MEDIA_PROGRESS_AUDIO_COMPLETE,
            state.audio_result.samples_per_channel,
            state.audio_result.samples_per_channel,
            state.audio_result.kernel_launches, err);
    host_release(&state, (void **)&state.audio_latent,
                 state.audio_latent_values, sizeof(float));
    if (rc == YVEX_OK)
        rc = generation_progress(request, YVEX_RUNTIME_MEDIA_PROGRESS_PUBLICATION_START,
                                 0ull, 0ull, 0ull, err);
    if (rc == YVEX_OK) rc = media_publish(&state, err);
    if (rc == YVEX_OK)
        rc = generation_progress(
            request, YVEX_RUNTIME_MEDIA_PROGRESS_PUBLICATION_COMPLETE,
            state.media_result.file_bytes, state.media_result.file_bytes,
            state.media_result.video_frames, err);
    if (rc == YVEX_OK) {
        rc = result_publish(&state, result, err);
    }
    if (rc == YVEX_OK) {
        yvex_error_clear(err);
    }
    generation_state_close(&state);
    return rc;
}

int yvex_runtime_av_generate(
    const yvex_runtime_av_generation_request *request,
    yvex_runtime_av_generation_result *result, yvex_error *err)
{
    yvex_runtime_media_model_summary summary;
    yvex_runtime_media_model *model = NULL;
    yvex_error primary;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    rc = yvex_runtime_media_model_open(&model, request, NULL, &summary, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_media_model_generate(model, request, result, err);
    primary = err ? *err : (yvex_error){0};
    yvex_runtime_media_model_close(&model);
    if (rc != YVEX_OK && err) *err = primary;
    return rc;
}
