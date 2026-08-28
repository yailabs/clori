/* Typed decoded-image ownership and deterministic RGB8 resampling. */
#ifndef INCLUDE_YVEX_INTERNAL_IMAGE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_IMAGE_H_INCLUDED

#include <yvex/internal/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_IMAGE_SCHEMA_V1 1u
#define YVEX_IMAGE_RESIZE_SCHEMA_V1 1u

typedef enum {
    YVEX_IMAGE_RGB8 = 1
} yvex_image_pixel_format;

typedef struct yvex_image {
    unsigned int schema_version;
    yvex_image_pixel_format format;
    unsigned long long width, height, channels, row_bytes, data_bytes, source_file_bytes;
    unsigned char *data;
    char source_identity[YVEX_SHA256_HEX_BYTES];
    char content_identity[YVEX_SHA256_HEX_BYTES];
    int complete;
} yvex_image;

typedef struct {
    unsigned int schema_version;
    unsigned long long resized_width, resized_height;
    unsigned long long crop_left, crop_top, output_width, output_height;
} yvex_image_resize_request;

int yvex_image_decode_file(yvex_image *output, const char *path,
                           unsigned long long maximum_file_bytes, yvex_error *err);
int yvex_image_resize_lanczos_rgb8(const yvex_image *input,
                                   const yvex_image_resize_request *request,
                                   yvex_image *output, yvex_error *err);
int yvex_image_resize_bicubic_rgb8(const yvex_image *input,
                                   const yvex_image_resize_request *request,
                                   yvex_image *output, yvex_error *err);
void yvex_image_close(yvex_image *image);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_IMAGE_H_INCLUDED */
