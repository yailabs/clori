/* Render typed synchronized-media publication facts without promoting generation support. */
#include "src/cli/render/private.h"

int yvex_media_publish_render(FILE *fp, yvex_graph_report_mode mode,
                              const yvex_cli_media_report *report)
{
    const yvex_media_avi_result *result;
    if (!fp || !report || !report->publication.complete) return YVEX_ERR_INVALID_ARG;
    result = &report->publication;
    if (mode == YVEX_GRAPH_REPORT_MODE_TABLE) {
        yvex_cli_out_writef(
            fp, "status\tcontainer\tframes\twidth\theight\taudio_samples\tfile_bytes\n"
                "media-published\tavi-bgr24-pcm-s16le\t%llu\t%llu\t%llu\t%llu\t%llu\n",
            result->video_frames, report->width, report->height,
            result->audio_samples_used, result->file_bytes);
        return YVEX_OK;
    }
    yvex_cli_out_writef(
        fp, "status: media-published\ncontainer: avi-bgr24-pcm-s16le\n"
            "output: %s\nvideo_frames: %llu\nvideo_width: %llu\nvideo_height: %llu\n"
            "video_fps: %llu/%llu\naudio_channels: %llu\naudio_sample_rate: %llu\n"
            "audio_samples_used: %llu\naudio_samples_trimmed: %llu\nfile_bytes: %llu\n"
            "file_identity: %s\npublication_identity: %s\n",
        report->output_path, result->video_frames, report->width, report->height,
        report->fps_numerator, report->fps_denominator, report->audio_channels,
        report->audio_sample_rate, result->audio_samples_used,
        result->audio_samples_trimmed, result->file_bytes, result->file_identity,
        result->publication_identity);
    if (mode == YVEX_GRAPH_REPORT_MODE_AUDIT)
        yvex_cli_out_writef(
            fp, "video_identity: %s\naudio_identity: %s\nexecution_identity: %s\n"
                "duration: %llu/%llu seconds\npeak_workspace_bytes: %llu\n"
                "production_capability_available: true\nproduction_api_available: true\n"
                "operator_command_available: true\nend_user_path_available: false\n"
                "cli_applicability: applicable\n",
            result->video_identity, result->audio_identity, result->execution_identity,
            result->video_duration_numerator, result->video_duration_denominator,
            result->peak_workspace_bytes);
    return YVEX_OK;
}

int yvex_media_generate_render(
    FILE *fp, yvex_graph_report_mode mode, const char *output_path,
    const yvex_runtime_av_generation_result *result)
{
    if (!fp || !output_path || !result || !result->complete) return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_GRAPH_REPORT_MODE_TABLE) {
        yvex_cli_out_writef(
            fp, "status\tframes\twidth\theight\taudio_samples\tfile_bytes\n"
                "generation-complete\t%llu\t%llu\t%llu\t%llu\t%llu\n",
            result->frames, result->width, result->height, result->audio_samples,
            result->file_bytes);
        return YVEX_OK;
    }
    yvex_cli_out_writef(
        fp, "status: generation-complete\ntarget: minimax-h3-fl2va\noutput: %s\n"
            "prompt_tokens: %llu\nframes: %llu\ngeometry: %llux%llu\n"
            "audio_samples: %llu\nmodel_evaluations: %llu\nkernel_launches: %llu\n"
            "peak_device_bytes: %llu\npeak_workspace_bytes: %llu\nfile_bytes: %llu\n"
            "execution_identity: %s\nfile_identity: %s\npublication_identity: %s\n",
        output_path, result->prompt_tokens, result->frames, result->width, result->height,
        result->audio_samples, result->model_evaluations, result->kernel_launches,
        result->peak_device_bytes, result->peak_workspace_bytes, result->file_bytes,
        result->execution_identity, result->file_identity, result->publication_identity);
    if (mode == YVEX_GRAPH_REPORT_MODE_AUDIT)
        yvex_cli_out_writef(
            fp, "prompt_identity: %s\nconditioning_identity: %s\nplan_identity: %s\n"
                "layout_identity: %s\nlatent_identity: %s\nvae_input_identity: %s\n"
                "video_identity: %s\naudio_identity: %s\n"
                "production_capability_available: true\nproduction_api_available: true\n"
                "operator_command_available: true\nend_user_path_available: true\n"
                "cli_applicability: applicable\n",
            result->prompt_identity, result->conditioning_identity, result->plan_identity,
            result->layout_identity, result->latent_identity, result->vae_input_identity,
            result->video_identity, result->audio_identity);
    return YVEX_OK;
}
