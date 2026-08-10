/* File serializers share escaping helpers here without acquiring stream ownership. */
#ifndef INCLUDE_YVEX_INTERNAL_IO_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_IO_H_INCLUDED

#include <stdio.h>
#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Escaped JSON fields. */
void yvex_file_json_write_string(FILE *fp, const char *s);
void yvex_file_json_write_field(FILE *fp,
                                const char *indent,
                                const char *name,
                                const char *value,
                                int comma);

#define YVEX_MEDIA_AVI_SCHEMA_V1 1u
#define YVEX_MEDIA_IDENTITY_CAP 65u
typedef struct {
    unsigned int schema_version;
    const char *path;
    const float *video, *audio;
    unsigned long long video_channels, frames, width, height;
    unsigned long long fps_numerator, fps_denominator;
    unsigned long long audio_channels, audio_samples, audio_sample_rate;
    unsigned long long maximum_file_bytes;
    const char *video_identity, *audio_identity, *execution_identity;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_media_avi_request;

typedef struct {
    unsigned int schema_version;
    unsigned long long file_bytes, video_frames, audio_samples_used, audio_samples_trimmed;
    unsigned long long video_duration_numerator, video_duration_denominator;
    unsigned long long peak_workspace_bytes;
    char video_identity[YVEX_MEDIA_IDENTITY_CAP];
    char audio_identity[YVEX_MEDIA_IDENTITY_CAP];
    char execution_identity[YVEX_MEDIA_IDENTITY_CAP];
    char file_identity[YVEX_MEDIA_IDENTITY_CAP];
    char publication_identity[YVEX_MEDIA_IDENTITY_CAP];
    int complete;
} yvex_media_avi_result;

int yvex_media_avi_publish(const yvex_media_avi_request *request,
                           yvex_media_avi_result *result,
                           yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_IO_H_INCLUDED */
