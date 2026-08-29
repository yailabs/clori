/* Configure one persistent server with a bounded staged audio-video family adapter. */
#ifndef INCLUDE_YVEX_INTERNAL_SERVER_MEDIA_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SERVER_MEDIA_H_INCLUDED

#include <yvex/internal/media.h>
#include <yvex/server.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_SERVER_MEDIA_SCHEMA_V2 2u
#define YVEX_SERVER_MEDIA_PROFILE_CAP YVEX_RUNTIME_MEDIA_PROFILE_CAP

typedef yvex_runtime_media_profile yvex_server_media_profile;

typedef struct {
    unsigned int schema_version;
    const char *output_root;
    const char *artifact_reopen_cache_root;
    yvex_runtime_media_execution_preset execution_preset;
    yvex_runtime_av_generation_request request_template;
    const yvex_server_media_profile *profiles;
    unsigned long long profile_count;
    unsigned long long frames_per_chunk, frame_remainder;
    unsigned long long minimum_frames, maximum_frames;
    unsigned long long minimum_inference_steps, maximum_inference_steps;
    unsigned long long released_sigma_grid_points, default_seed;
    unsigned long long canvas_multiple, canvas_short_edge;
    unsigned long long minimum_canvas_pixels, maximum_canvas_pixels;
    unsigned long long released_width, released_height;
    unsigned long long minimum_duration_milliseconds, maximum_duration_milliseconds;
    unsigned long long minimum_aspect_numerator, minimum_aspect_denominator;
    unsigned long long maximum_aspect_numerator, maximum_aspect_denominator;
} yvex_server_media_options;

int yvex_server_media_engine_load(
    yvex_server *, const yvex_server_engine_options *,
    const yvex_server_media_options *, yvex_server_engine_summary *, yvex_error *);

#ifdef __cplusplus
}
#endif
#endif
