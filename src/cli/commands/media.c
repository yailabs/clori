/* Adapt bounded decoded media files into one native transactional publication request. */
#include "src/cli/input/private.h"
#include "src/cli/render/private.h"

#include <stdlib.h>
#include <yvex/internal/core.h>
#include <yvex/internal/io.h>

static int media_command_error(const yvex_error *err)
{
    yvex_cli_out_writef(yvex_cli_out_stderr(), "yvex: %s: %s\n",
                        yvex_error_where(err), yvex_error_message(err));
    return exit_for_status(yvex_error_code(err));
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
