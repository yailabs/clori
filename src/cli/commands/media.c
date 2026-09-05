/* Bind operator requests to the staged runtime and native transactional publication owners. */
#include "src/cli/input/private.h"
#include "src/cli/render/private.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/io.h>
#include <yvex/internal/media.h>

static volatile sig_atomic_t media_signal_seen;

static void media_signal_handler(int signal_number)
{
    media_signal_seen = signal_number;
}

static int media_cancel_requested(void *context)
{
    (void)context;
    return media_signal_seen != 0;
}

static int media_signals_install(
    struct sigaction *old_interrupt, struct sigaction *old_terminate, yvex_error *err)
{
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = media_signal_handler;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, old_interrupt) != 0) {
        yvex_error_set(err, YVEX_ERR_IO, "media.generate.signals",
                       "generation signal handlers could not be installed");
        return YVEX_ERR_IO;
    }
    if (sigaction(SIGTERM, &action, old_terminate) != 0) {
        (void)sigaction(SIGINT, old_interrupt, NULL);
        yvex_error_set(err, YVEX_ERR_IO, "media.generate.signals",
                       "generation signal handlers could not be installed");
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}

static int media_signals_restore(
    const struct sigaction *old_interrupt, const struct sigaction *old_terminate,
    yvex_error *err)
{
    if (sigaction(SIGINT, old_interrupt, NULL) != 0 ||
        sigaction(SIGTERM, old_terminate, NULL) != 0) {
        yvex_error_set(err, YVEX_ERR_IO, "media.generate.signals",
                       "generation signal handlers could not be restored");
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}

static int media_command_error(const yvex_error *err)
{
    yvex_cli_out_writef(yvex_cli_out_stderr(), "yvex: %s: %s\n",
                        yvex_error_where(err), yvex_error_message(err));
    return exit_for_status(yvex_error_code(err));
}

int yvex_media_generate_command(const yvex_graph_args *args, yvex_error *err)
{
    const yvex_component_variant_adapter *adapter =
        args ? yvex_graph_component_variant_find(args->media.target) : NULL;
    const yvex_media_execution_recipe *execution =
        adapter ? adapter->media_execution : NULL;
    yvex_media_target_profile target = {0};
    yvex_runtime_av_generation_request request = {0};
    yvex_runtime_av_generation_result result;
    struct sigaction old_interrupt, old_terminate;
    yvex_error restore_error;
    int rc, restore_rc, render_rc, signals_installed = 0;
    if (!args || !args->media.generate || !adapter ||
        !adapter->media_target_profile || !execution) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "media.generate.cli",
                       "the requested target has no admitted media adapter");
        return media_command_error(err);
    }
    rc = adapter->media_target_profile(&target, err);
    request.schema_version = YVEX_RUNTIME_AV_GENERATION_SCHEMA_V1;
    request.target = args->media.target;
    request.prompt = args->media.prompt;
    request.output_path = args->media.output_file;
    request.text_artifact_path = args->media.text_artifact;
    request.transformer_artifact_path = args->media.transformer_artifact;
    request.video_artifact_path = args->media.video_artifact;
    request.audio_artifact_path = args->media.audio_artifact;
    request.source_identity = target.source_identity;
    request.frames = args->media.frames;
    request.width = args->media.width;
    request.height = args->media.height;
    request.fps_numerator = args->media.fps_numerator;
    request.fps_denominator = args->media.fps_denominator;
    request.audio_sample_rate = target.audio_sample_rate;
    request.inference_steps = (unsigned int)args->media.inference_steps;
    request.conditioning_layers = execution->conditioning_layers;
    request.transformer_blocks = args->media.transformer_blocks;
    request.seed = args->media.seed;
    request.keyframe_encode_seed = target.keyframe_encode_seed;
    request.maximum_prompt_tokens = execution->maximum_prompt_tokens;
    request.maximum_packed_rows = execution->maximum_packed_rows;
    request.maximum_host_bytes = args->media.maximum_host_bytes;
    request.maximum_device_bytes = args->media.maximum_device_bytes;
    request.maximum_workspace_bytes = args->media.maximum_workspace_bytes;
    request.maximum_file_bytes = args->media.maximum_output_bytes;
    request.component_backend = execution->component_backend;
    request.video_temporal_ratio = target.video_temporal_ratio;
    request.video_clip_length = target.video_clip_length;
    request.video_token_drop = target.video_token_drop;
    request.video_spatial_ratio = target.video_spatial_ratio;
    request.video_tile_size = target.video_tile_size;
    request.video_minimum_tile_overlap = target.video_minimum_tile_overlap;
    request.video_mean = target.video_mean;
    request.video_std = target.video_std;
    request.audio_mean = target.audio_mean;
    request.audio_std = target.audio_std;
    request.pixel_mean = target.pixel_mean;
    request.pixel_std = target.pixel_std;
    request.video_channels = target.video_channels;
    request.audio_channels = target.audio_channels;
    request.pixel_channels = target.pixel_channels;
    request.audio_output_channels = target.audio_output_channels;
    request.audio_samples_per_step = target.audio_samples_per_step;
    request.plan_build = execution->plan_build;
    request.layout_build = execution->layout_build;
    request.component_admit = execution->component_admit;
    request.condition = execution->condition;
    request.keyframe_encode = execution->keyframe_encode;
    request.latent = execution->latent;
    request.video_decode = execution->video_decode;
    request.audio_decode = execution->audio_decode;
    request.cancel_requested = media_cancel_requested;
    media_signal_seen = 0;
    if (rc == YVEX_OK &&
        (execution->output_semantic_domain || execution->video_output_requirement ||
         execution->audio_output_requirement))
        rc = yvex_runtime_media_request_specialize(
            &request, execution->output_semantic_domain,
            execution->video_output_requirement,
            execution->audio_output_requirement, err);
    if (rc == YVEX_OK) {
        rc = media_signals_install(&old_interrupt, &old_terminate, err);
        signals_installed = rc == YVEX_OK;
    }
    if (rc == YVEX_OK) rc = yvex_runtime_av_generate(&request, &result, err);
    if (signals_installed) {
        yvex_error_clear(&restore_error);
        restore_rc = media_signals_restore(&old_interrupt, &old_terminate, &restore_error);
        if (restore_rc != YVEX_OK) { rc = restore_rc; *err = restore_error; }
    }
    media_signal_seen = 0;
    if (rc == YVEX_OK) {
        render_rc = yvex_media_generate_render(
            yvex_cli_out_stdout(), args->render_mode, args->media.output_file, &result);
        if (render_rc != YVEX_OK) {
            yvex_error_set(err, render_rc, "media.generate.cli",
                           "generation result rendering failed after publication");
            rc = render_rc;
        }
    }
    return rc == YVEX_OK ? 0 : media_command_error(err);
}

int yvex_media_publish_command(const yvex_graph_args *args, yvex_error *err)
{
    yvex_media_avi_request request = {0};
    yvex_cli_media_report report = {0};
    yvex_core_file_result video_file_result, audio_file_result;
    unsigned char *video = NULL, *audio = NULL;
    size_t video_count = 0u, audio_count = 0u;
    unsigned long long video_values, audio_values, video_bytes, audio_bytes;
    unsigned long long input_bytes, output_budget;
    int rc, render_rc;
    if (!args || !args->media.active ||
        !yvex_core_u64_mul(3ull, args->media.frames, &video_values) ||
        !yvex_core_u64_mul(video_values, args->media.height, &video_values) ||
        !yvex_core_u64_mul(video_values, args->media.width, &video_values) ||
        !yvex_core_u64_mul(video_values, sizeof(float), &video_bytes) ||
        !yvex_core_u64_mul(args->media.audio_channels, args->media.audio_samples,
                           &audio_values) ||
        !yvex_core_u64_mul(audio_values, sizeof(float), &audio_bytes) ||
        !yvex_core_u64_add(video_bytes, audio_bytes, &input_bytes) ||
        input_bytes >= args->media.maximum_host_bytes || video_bytes > SIZE_MAX ||
        audio_bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "media.publish.cli",
                       "decoded media inputs exceed --max-host-bytes");
        return media_command_error(err);
    }
    output_budget = args->media.maximum_host_bytes - input_bytes;
    if (output_budget > args->media.maximum_output_bytes)
        output_budget = args->media.maximum_output_bytes;
    rc = yvex_core_file_read_snapshot(
        args->media.video_file, (size_t)video_bytes, &video, &video_count,
        &video_file_result, err);
    if (rc == YVEX_OK && video_count != (size_t)video_bytes) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "media.publish.cli",
                       "video file does not match planar F32 [3,frames,height,width]");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK)
        rc = yvex_core_file_read_snapshot(
            args->media.audio_file, (size_t)audio_bytes, &audio, &audio_count,
            &audio_file_result, err);
    if (rc == YVEX_OK && audio_count != (size_t)audio_bytes) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "media.publish.cli",
                       "audio file does not match planar F32 [channels,samples]");
        rc = YVEX_ERR_FORMAT;
    }
    request.schema_version = YVEX_MEDIA_AVI_SCHEMA_V1;
    request.path = args->media.output_file;
    request.video = (const float *)video;
    request.audio = (const float *)audio;
    request.video_channels = 3ull;
    request.frames = args->media.frames;
    request.width = args->media.width;
    request.height = args->media.height;
    request.fps_numerator = args->media.fps_numerator;
    request.fps_denominator = args->media.fps_denominator;
    request.audio_channels = args->media.audio_channels;
    request.audio_samples = args->media.audio_samples;
    request.audio_sample_rate = args->media.audio_sample_rate;
    request.maximum_file_bytes = output_budget;
    if (rc == YVEX_OK)
        rc = yvex_media_avi_publish(&request, &report.publication, err);
    if (rc == YVEX_OK) {
        report.output_path = args->media.output_file;
        report.width = args->media.width;
        report.height = args->media.height;
        report.fps_numerator = args->media.fps_numerator;
        report.fps_denominator = args->media.fps_denominator;
        report.audio_channels = args->media.audio_channels;
        report.audio_sample_rate = args->media.audio_sample_rate;
        render_rc = yvex_media_publish_render(
            yvex_cli_out_stdout(), args->render_mode, &report);
        if (render_rc != YVEX_OK) {
            yvex_error_set(err, render_rc, "media.publish.cli",
                           "media publication rendering failed after publication");
            rc = render_rc;
        }
    }
    free(audio);
    free(video);
    return rc == YVEX_OK ? 0 : media_command_error(err);
}
