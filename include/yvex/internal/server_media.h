/* Configure one persistent server with a bounded staged audio-video family adapter. */
#ifndef INCLUDE_YVEX_INTERNAL_SERVER_MEDIA_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SERVER_MEDIA_H_INCLUDED

#include <yvex/internal/media.h>
#include <yvex/server.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_SERVER_MEDIA_SCHEMA_V1 1u
#define YVEX_SERVER_MEDIA_PROFILE_CAP YVEX_RUNTIME_MEDIA_PROFILE_CAP

typedef yvex_runtime_media_profile yvex_server_media_profile;

typedef struct {
    unsigned int schema_version;
    const char *output_root;
    yvex_runtime_av_generation_request request_template;
    const yvex_server_media_profile *profiles;
    unsigned long long profile_count;
    unsigned long long frames_per_chunk, frame_remainder;
    unsigned long long minimum_frames, maximum_frames;
    unsigned long long minimum_inference_steps, maximum_inference_steps;
    unsigned long long canvas_multiple, maximum_canvas_pixels;
} yvex_server_media_options;

int yvex_server_media_configure(
    yvex_server *, const yvex_server_media_options *, yvex_error *);

#ifdef __cplusplus
}
#endif
#endif
