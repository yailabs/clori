/* Decode immutable PNG input and reproduce Pillow-compatible RGB8 LANCZOS resampling. */
#include <yvex/internal/image.h>

#include <yvex/internal/core.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

enum { PNG_SIGNATURE_BYTES = 8u, PNG_IHDR_BYTES = 13u, RESAMPLE_PRECISION_BITS = 22u };

typedef struct {
    unsigned char *bytes;
    size_t count, capacity;
} byte_buffer;

typedef struct {
    int *bounds;
    int32_t *coefficients;
    unsigned long long output_size;
    int kernel_size;
} resample_axis;

static int image_refuse(yvex_error *err, yvex_status status, const char *where,
                        const char *message)
{
    yvex_error_set(err, status, where, message);
    return status;
}

static uint32_t read_be32(const unsigned char *bytes)
{
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}

static int byte_buffer_append(byte_buffer *buffer, const unsigned char *bytes, size_t count,
                              yvex_error *err)
{
    unsigned char *grown;
    size_t required, capacity;
    if (!buffer || (!bytes && count) || count > SIZE_MAX - buffer->count)
        return image_refuse(err, YVEX_ERR_BOUNDS, "image.png.buffer",
                            "PNG byte extent overflowed");
    required = buffer->count + count;
    if (required > buffer->capacity) {
        capacity = buffer->capacity ? buffer->capacity : 4096u;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2u)
                return image_refuse(err, YVEX_ERR_BOUNDS, "image.png.buffer",
                                    "PNG byte capacity overflowed");
            capacity *= 2u;
        }
        grown = realloc(buffer->bytes, capacity);
        if (!grown)
            return image_refuse(err, YVEX_ERR_NOMEM, "image.png.buffer",
                                "PNG byte allocation failed");
        buffer->bytes = grown;
        buffer->capacity = capacity;
    }
    if (count) memcpy(buffer->bytes + buffer->count, bytes, count);
    buffer->count = required;
    return YVEX_OK;
}

static int image_file_read(const char *path, unsigned long long maximum,
                           unsigned char **output, size_t *output_count, yvex_error *err)
{
    struct stat status;
    unsigned char *bytes = NULL;
    size_t count = 0u;
    int descriptor, rc = YVEX_OK;
    if (!path || path[0] != '/' || !maximum || !output || !output_count)
        return image_refuse(err, YVEX_ERR_INVALID_ARG, "image.file",
                            "an absolute image path and bounded output are required");
    *output = NULL;
    *output_count = 0u;
    descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0)
        return image_refuse(err, YVEX_ERR_IO, "image.file.open",
                            "image file could not be opened without following a symlink");
    if (fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0 ||
        (unsigned long long)status.st_size > maximum ||
        (uintmax_t)status.st_size > SIZE_MAX) {
        rc = image_refuse(err, YVEX_ERR_BOUNDS, "image.file.stat",
                          "image must be one bounded regular nonempty file");
    }
    if (rc == YVEX_OK) {
        count = (size_t)status.st_size;
        bytes = malloc(count);
        if (!bytes)
            rc = image_refuse(err, YVEX_ERR_NOMEM, "image.file.read",
                              "image file allocation failed");
    }
    for (size_t offset = 0u; rc == YVEX_OK && offset < count;) {
        ssize_t consumed = read(descriptor, bytes + offset, count - offset);
        if (consumed > 0) offset += (size_t)consumed;
        else if (consumed < 0 && errno == EINTR) continue;
        else rc = image_refuse(err, YVEX_ERR_IO, "image.file.read",
                               "image file could not be read completely");
    }
    if (close(descriptor) != 0 && rc == YVEX_OK)
        rc = image_refuse(err, YVEX_ERR_IO, "image.file.close",
                          "image file close failed");
    if (rc == YVEX_OK) {
        *output = bytes;
        *output_count = count;
    } else free(bytes);
    return rc;
}

static int image_identity(const char *domain, const unsigned char *bytes, size_t count,
                          unsigned long long width, unsigned long long height,
                          char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_sha256_update_u64_be(&hash, width) ||
        !yvex_sha256_update_u64_be(&hash, height) ||
        !yvex_sha256_update_u64_be(&hash, count) ||
        (count && !yvex_sha256_update(&hash, bytes, count)) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static unsigned char paeth(unsigned char left, unsigned char above, unsigned char corner)
{
    int estimate = (int)left + (int)above - (int)corner;
    int dl = abs(estimate - (int)left);
    int da = abs(estimate - (int)above);
    int dc = abs(estimate - (int)corner);
    return dl <= da && dl <= dc ? left : da <= dc ? above : corner;
}

static int png_unfilter(unsigned char *rows, const unsigned char *filtered,
                        unsigned long long height, size_t row_bytes, size_t pixel_bytes,
                        yvex_error *err)
{
    unsigned long long y;
    for (y = 0u; y < height; ++y) {
        const unsigned char *source = filtered + (size_t)y * (row_bytes + 1u) + 1u;
        unsigned char *target = rows + (size_t)y * row_bytes;
        const unsigned char *prior = y ? target - row_bytes : NULL;
        unsigned int filter = filtered[(size_t)y * (row_bytes + 1u)];
        size_t x;
        if (filter > 4u)
            return image_refuse(err, YVEX_ERR_FORMAT, "image.png.filter",
                                "PNG row uses an unsupported filter");
        for (x = 0u; x < row_bytes; ++x) {
            unsigned char left = x >= pixel_bytes ? target[x - pixel_bytes] : 0u;
            unsigned char above = prior ? prior[x] : 0u;
            unsigned char corner = prior && x >= pixel_bytes ? prior[x - pixel_bytes] : 0u;
            unsigned int predictor = filter == 0u ? 0u : filter == 1u ? left :
                                     filter == 2u ? above : filter == 3u ?
                                         ((unsigned int)left + above) / 2u :
                                         paeth(left, above, corner);
            target[x] = (unsigned char)(source[x] + predictor);
        }
    }
    return YVEX_OK;
}

static int png_to_rgb(const unsigned char *rows, unsigned long long width,
                      unsigned long long height, unsigned int color_type,
                      const unsigned char *palette, size_t palette_count,
                      unsigned char *rgb, yvex_error *err)
{
    size_t source_channels = color_type == 0u || color_type == 3u ? 1u :
                             color_type == 2u ? 3u : color_type == 4u ? 2u : 4u;
    unsigned long long pixel_count = width * height, index;
    for (index = 0u; index < pixel_count; ++index) {
        const unsigned char *source = rows + (size_t)index * source_channels;
        unsigned char *target = rgb + (size_t)index * 3u;
        if (color_type == 0u || color_type == 4u)
            target[0] = target[1] = target[2] = source[0];
        else if (color_type == 3u) {
            size_t entry = (size_t)source[0] * 3u;
            if (entry + 3u > palette_count)
                return image_refuse(err, YVEX_ERR_FORMAT, "image.png.palette",
                                    "PNG palette index exceeds the admitted palette");
            memcpy(target, palette + entry, 3u);
        } else memcpy(target, source, 3u);
    }
    return YVEX_OK;
}

static int png_chunks(const unsigned char *file, size_t file_count, byte_buffer *compressed,
                      byte_buffer *palette, unsigned long long *width,
                      unsigned long long *height, unsigned int *color_type,
                      yvex_error *err)
{
    static const unsigned char signature[PNG_SIGNATURE_BYTES] =
        {137u, 80u, 78u, 71u, 13u, 10u, 26u, 10u};
    size_t cursor = PNG_SIGNATURE_BYTES;
    int ihdr = 0, iend = 0;
    if (file_count < PNG_SIGNATURE_BYTES || memcmp(file, signature, sizeof(signature)) != 0)
        return image_refuse(err, YVEX_ERR_FORMAT, "image.png.signature",
                            "only a standard PNG image is admitted");
    while (!iend && cursor <= file_count - 12u) {
        uint32_t length = read_be32(file + cursor);
        const unsigned char *type = file + cursor + 4u;
        const unsigned char *data = type + 4u;
        uint32_t expected_crc, actual_crc;
        if ((size_t)length > file_count - cursor - 12u)
            return image_refuse(err, YVEX_ERR_FORMAT, "image.png.chunk",
                                "PNG chunk extent exceeds the file");
        expected_crc = read_be32(data + length);
        actual_crc = (uint32_t)crc32(0L, Z_NULL, 0);
        actual_crc = (uint32_t)crc32(actual_crc, type, 4u);
        actual_crc = (uint32_t)crc32(actual_crc, data, length);
        if (expected_crc != actual_crc)
            return image_refuse(err, YVEX_ERR_FORMAT, "image.png.crc",
                                "PNG chunk checksum is invalid");
        if (memcmp(type, "IHDR", 4u) == 0) {
            if (ihdr || cursor != PNG_SIGNATURE_BYTES || length != PNG_IHDR_BYTES)
                return image_refuse(err, YVEX_ERR_FORMAT, "image.png.ihdr",
                                    "PNG has a malformed IHDR owner");
            *width = read_be32(data);
            *height = read_be32(data + 4u);
            *color_type = data[9];
            if (!*width || !*height || *width > 16384u || *height > 16384u ||
                data[8] != 8u || (*color_type != 0u && *color_type != 2u &&
                                  *color_type != 3u && *color_type != 4u &&
                                  *color_type != 6u) ||
                data[10] || data[11] || data[12])
                return image_refuse(err, YVEX_ERR_UNSUPPORTED, "image.png.ihdr",
                                    "PNG must be non-interlaced 8-bit grayscale, palette, RGB, or RGBA");
            ihdr = 1;
        } else if (memcmp(type, "PLTE", 4u) == 0) {
            if (!ihdr || length < 3u || length > 768u || length % 3u || palette->count)
                return image_refuse(err, YVEX_ERR_FORMAT, "image.png.palette",
                                    "PNG palette is malformed");
            if (byte_buffer_append(palette, data, length, err) != YVEX_OK)
                return yvex_error_code(err);
        } else if (memcmp(type, "IDAT", 4u) == 0) {
            if (!ihdr || byte_buffer_append(compressed, data, length, err) != YVEX_OK)
                return yvex_error_code(err);
        } else if (memcmp(type, "IEND", 4u) == 0) {
            if (length)
                return image_refuse(err, YVEX_ERR_FORMAT, "image.png.iend",
                                    "PNG IEND chunk is malformed");
            iend = 1;
        } else if ((type[0] & 0x20u) == 0u) {
            return image_refuse(err, YVEX_ERR_UNSUPPORTED, "image.png.chunk",
                                "PNG contains an unsupported critical chunk");
        }
        cursor += (size_t)length + 12u;
    }
    if (!ihdr || !iend || !compressed->count || cursor != file_count ||
        (*color_type == 3u && !palette->count))
        return image_refuse(err, YVEX_ERR_FORMAT, "image.png.structure",
                            "PNG required chunks are absent or trailing bytes remain");
    return YVEX_OK;
}

static int png_decode(const unsigned char *file, size_t file_count, yvex_image *output,
                      yvex_error *err)
{
    byte_buffer compressed = {0}, palette = {0};
    unsigned char *filtered = NULL, *rows = NULL, *rgb = NULL;
    unsigned long long width = 0u, height = 0u, pixels, rgb_bytes;
    unsigned int color_type = 0u;
    size_t source_channels = 0u, row_bytes = 0u, filtered_bytes = 0u;
    uLongf inflated;
    int rc = png_chunks(file, file_count, &compressed, &palette, &width, &height,
                        &color_type, err);
    source_channels = color_type == 0u || color_type == 3u ? 1u :
                      color_type == 2u ? 3u : color_type == 4u ? 2u : 4u;
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(width, height, &pixels) ||
         !yvex_core_u64_mul(pixels, 3u, &rgb_bytes) || rgb_bytes > SIZE_MAX ||
         width > SIZE_MAX / source_channels ||
         (row_bytes = (size_t)width * source_channels) > SIZE_MAX - 1u ||
         height > SIZE_MAX / (row_bytes + 1u)))
        rc = image_refuse(err, YVEX_ERR_BOUNDS, "image.png.geometry",
                          "decoded PNG geometry overflowed");
    if (rc == YVEX_OK) {
        filtered_bytes = (row_bytes + 1u) * (size_t)height;
        filtered = malloc(filtered_bytes);
        rows = malloc(row_bytes * (size_t)height);
        rgb = malloc((size_t)rgb_bytes);
        if (!filtered || !rows || !rgb)
            rc = image_refuse(err, YVEX_ERR_NOMEM, "image.png.decode",
                              "decoded PNG allocation failed");
    }
    if (rc == YVEX_OK) {
        inflated = (uLongf)filtered_bytes;
        if (compressed.count > ULONG_MAX ||
            uncompress(filtered, &inflated, compressed.bytes, (uLong)compressed.count) != Z_OK ||
            inflated != filtered_bytes)
            rc = image_refuse(err, YVEX_ERR_FORMAT, "image.png.inflate",
                              "PNG compressed pixels are malformed");
    }
    if (rc == YVEX_OK)
        rc = png_unfilter(rows, filtered, height, row_bytes, source_channels, err);
    if (rc == YVEX_OK)
        rc = png_to_rgb(rows, width, height, color_type, palette.bytes, palette.count, rgb, err);
    if (rc == YVEX_OK &&
        !image_identity("yvex.image.rgb8.v1", rgb, (size_t)rgb_bytes, width, height,
                        output->content_identity))
        rc = image_refuse(err, YVEX_ERR_STATE, "image.png.identity",
                          "decoded image identity could not be sealed");
    if (rc == YVEX_OK) {
        output->schema_version = YVEX_IMAGE_SCHEMA_V1;
        output->format = YVEX_IMAGE_RGB8;
        output->width = width;
        output->height = height;
        output->channels = 3u;
        output->row_bytes = width * 3u;
        output->data_bytes = rgb_bytes;
        output->data = rgb;
        output->complete = 1;
        rgb = NULL;
    }
    free(rgb);
    free(rows);
    free(filtered);
    free(palette.bytes);
    free(compressed.bytes);
    return rc;
}

int yvex_image_decode_file(yvex_image *output, const char *path,
                           unsigned long long maximum_file_bytes, yvex_error *err)
{
    yvex_image candidate = {0};
    unsigned char *file = NULL;
    size_t file_count = 0u;
    int rc;
    if (!output)
        return image_refuse(err, YVEX_ERR_INVALID_ARG, "image.decode",
                            "decoded image output is required");
    memset(output, 0, sizeof(*output));
    rc = image_file_read(path, maximum_file_bytes, &file, &file_count, err);
    if (rc == YVEX_OK)
        rc = png_decode(file, file_count, &candidate, err);
    if (rc == YVEX_OK &&
        !image_identity("yvex.image.source-file.v1", file, file_count, 0u, 0u,
                        candidate.source_identity))
        rc = image_refuse(err, YVEX_ERR_STATE, "image.source.identity",
                          "image source identity could not be sealed");
    if (rc == YVEX_OK) {
        candidate.source_file_bytes = file_count;
        *output = candidate;
        yvex_error_clear(err);
    } else yvex_image_close(&candidate);
    free(file);
    return rc;
}

static double sinc_filter(double value)
{
    if (value == 0.0) return 1.0;
    value *= 3.14159265358979323846264338327950288;
    return sin(value) / value;
}

static double lanczos_filter(double value)
{
    return value >= -3.0 && value < 3.0
               ? sinc_filter(value) * sinc_filter(value / 3.0) : 0.0;
}

static double bicubic_filter(double value)
{
    const double coefficient = -0.5;
    value = fabs(value);
    if (value < 1.0)
        return ((coefficient + 2.0) * value - (coefficient + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return (((coefficient * value - 5.0 * coefficient) * value +
                 8.0 * coefficient) * value - 4.0 * coefficient);
    return 0.0;
}

typedef double (*resample_filter_fn)(double);

static int resample_axis_build(unsigned long long input_size, unsigned long long output_size,
                               double filter_support, resample_filter_fn filter,
                               resample_axis *axis, yvex_error *err)
{
    double scale = (double)input_size / (double)output_size;
    double filter_scale = scale < 1.0 ? 1.0 : scale;
    double support = filter_support * filter_scale;
    int kernel_size = (int)ceil(support) * 2 + 1;
    size_t outputs = (size_t)output_size, coefficient_count;
    if (!input_size || !output_size || !filter || filter_support <= 0.0 || !axis ||
        output_size > INT_MAX ||
        input_size > INT_MAX || outputs > SIZE_MAX / 2u / sizeof(int) ||
        (size_t)kernel_size > SIZE_MAX / outputs)
        return image_refuse(err, YVEX_ERR_BOUNDS, "image.resize.coefficients",
                            "LANCZOS coefficient geometry overflowed");
    coefficient_count = outputs * (size_t)kernel_size;
    axis->bounds = malloc(outputs * 2u * sizeof(*axis->bounds));
    axis->coefficients = malloc(coefficient_count * sizeof(*axis->coefficients));
    if (!axis->bounds || !axis->coefficients)
        return image_refuse(err, YVEX_ERR_NOMEM, "image.resize.coefficients",
                            "LANCZOS coefficient allocation failed");
    for (size_t x = 0u; x < outputs; ++x) {
        double center = ((double)x + 0.5) * scale;
        int minimum = (int)(center - support + 0.5);
        int maximum = (int)(center + support + 0.5);
        double sum = 0.0;
        if (minimum < 0) minimum = 0;
        if (maximum > (int)input_size) maximum = (int)input_size;
        maximum -= minimum;
        axis->bounds[x * 2u] = minimum;
        axis->bounds[x * 2u + 1u] = maximum;
        for (int k = 0; k < kernel_size; ++k) {
            double coefficient = k < maximum
                ? filter(((double)(k + minimum) - center + 0.5) / filter_scale) : 0.0;
            axis->coefficients[x * (size_t)kernel_size + (size_t)k] = 0;
            sum += coefficient;
        }
        for (int k = 0; k < maximum; ++k) {
            double coefficient = filter(
                ((double)(k + minimum) - center + 0.5) / filter_scale);
            double normalized = sum != 0.0 ? coefficient / sum : coefficient;
            axis->coefficients[x * (size_t)kernel_size + (size_t)k] =
                normalized < 0.0
                    ? (int32_t)(-0.5 + normalized * (double)(1u << RESAMPLE_PRECISION_BITS))
                    : (int32_t)(0.5 + normalized * (double)(1u << RESAMPLE_PRECISION_BITS));
        }
    }
    axis->output_size = output_size;
    axis->kernel_size = kernel_size;
    return YVEX_OK;
}

static unsigned char resample_clip(int64_t value)
{
    int64_t scale = (int64_t)1 << RESAMPLE_PRECISION_BITS;
    int64_t pixel = value >= 0 ? value / scale : -((-value + scale - 1) / scale);
    if (pixel < 0) return 0u;
    if (pixel > 255) return 255u;
    return (unsigned char)pixel;
}

static void resample_horizontal(const unsigned char *input, unsigned long long input_width,
                                unsigned long long height, const resample_axis *axis,
                                unsigned char *output)
{
    for (unsigned long long y = 0u; y < height; ++y) {
        for (unsigned long long x = 0u; x < axis->output_size; ++x) {
            int minimum = axis->bounds[x * 2u], count = axis->bounds[x * 2u + 1u];
            const int32_t *coefficients = axis->coefficients + x * (size_t)axis->kernel_size;
            int64_t sums[3] = {(int64_t)1 << (RESAMPLE_PRECISION_BITS - 1u),
                               (int64_t)1 << (RESAMPLE_PRECISION_BITS - 1u),
                               (int64_t)1 << (RESAMPLE_PRECISION_BITS - 1u)};
            for (int k = 0; k < count; ++k) {
                const unsigned char *pixel = input +
                    ((size_t)y * (size_t)input_width + (size_t)(minimum + k)) * 3u;
                for (unsigned int channel = 0u; channel < 3u; ++channel)
                    sums[channel] += (int64_t)pixel[channel] * coefficients[k];
            }
            for (unsigned int channel = 0u; channel < 3u; ++channel)
                output[((size_t)y * (size_t)axis->output_size + (size_t)x) * 3u + channel] =
                    resample_clip(sums[channel]);
        }
    }
}

static void resample_vertical(const unsigned char *input, unsigned long long width,
                              const resample_axis *axis, unsigned char *output)
{
    for (unsigned long long y = 0u; y < axis->output_size; ++y) {
        int minimum = axis->bounds[y * 2u], count = axis->bounds[y * 2u + 1u];
        const int32_t *coefficients = axis->coefficients + y * (size_t)axis->kernel_size;
        for (unsigned long long x = 0u; x < width; ++x) {
            int64_t sums[3] = {(int64_t)1 << (RESAMPLE_PRECISION_BITS - 1u),
                               (int64_t)1 << (RESAMPLE_PRECISION_BITS - 1u),
                               (int64_t)1 << (RESAMPLE_PRECISION_BITS - 1u)};
            for (int k = 0; k < count; ++k) {
                const unsigned char *pixel = input +
                    ((size_t)(minimum + k) * (size_t)width + (size_t)x) * 3u;
                for (unsigned int channel = 0u; channel < 3u; ++channel)
                    sums[channel] += (int64_t)pixel[channel] * coefficients[k];
            }
            for (unsigned int channel = 0u; channel < 3u; ++channel)
                output[((size_t)y * (size_t)width + (size_t)x) * 3u + channel] =
                    resample_clip(sums[channel]);
        }
    }
}

static int resize_geometry(const yvex_image *input, const yvex_image_resize_request *request,
                           unsigned long long *resized_bytes, unsigned long long *output_bytes,
                           yvex_error *err)
{
    unsigned long long pixels;
    if (!input || input->schema_version != YVEX_IMAGE_SCHEMA_V1 || !input->complete ||
        input->format != YVEX_IMAGE_RGB8 || input->channels != 3u || !input->data ||
        !request || request->schema_version != YVEX_IMAGE_RESIZE_SCHEMA_V1 ||
        !request->resized_width || !request->resized_height || !request->output_width ||
        !request->output_height || request->crop_left > request->resized_width ||
        request->output_width > request->resized_width - request->crop_left ||
        request->crop_top > request->resized_height ||
        request->output_height > request->resized_height - request->crop_top ||
        !yvex_core_u64_mul(request->resized_width, request->resized_height, &pixels) ||
        !yvex_core_u64_mul(pixels, 3u, resized_bytes) || *resized_bytes > SIZE_MAX ||
        !yvex_core_u64_mul(request->output_width, request->output_height, &pixels) ||
        !yvex_core_u64_mul(pixels, 3u, output_bytes) || *output_bytes > SIZE_MAX)
        return image_refuse(err, YVEX_ERR_INVALID_ARG, "image.resize.geometry",
                            "complete RGB8 input and bounded resize/crop geometry are required");
    return YVEX_OK;
}

static int image_resize_rgb8(const yvex_image *input,
                             const yvex_image_resize_request *request,
                             double support, resample_filter_fn filter,
                             yvex_image *output, yvex_error *err)
{
    yvex_image candidate = {0};
    resample_axis horizontal = {0}, vertical = {0};
    unsigned char *horizontally = NULL, *resized = NULL;
    unsigned long long horizontal_bytes = 0u, resized_bytes = 0u, output_bytes = 0u;
    int rc;
    if (!output)
        return image_refuse(err, YVEX_ERR_INVALID_ARG, "image.resize",
                            "resized image output is required");
    memset(output, 0, sizeof(*output));
    rc = resize_geometry(input, request, &resized_bytes, &output_bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(request->resized_width, input->height, &horizontal_bytes) ||
         !yvex_core_u64_mul(horizontal_bytes, 3u, &horizontal_bytes) ||
         horizontal_bytes > SIZE_MAX))
        rc = image_refuse(err, YVEX_ERR_BOUNDS, "image.resize.horizontal",
                          "horizontal resize extent overflowed");
    if (rc == YVEX_OK) {
        horizontally = malloc((size_t)horizontal_bytes);
        resized = malloc((size_t)resized_bytes);
        candidate.data = malloc((size_t)output_bytes);
        if (!horizontally || !resized || !candidate.data)
            rc = image_refuse(err, YVEX_ERR_NOMEM, "image.resize",
                              "resampled image allocation failed");
    }
    if (rc == YVEX_OK)
        rc = resample_axis_build(input->width, request->resized_width,
                                 support, filter, &horizontal, err);
    if (rc == YVEX_OK)
        rc = resample_axis_build(input->height, request->resized_height,
                                 support, filter, &vertical, err);
    if (rc == YVEX_OK) {
        resample_horizontal(input->data, input->width, input->height, &horizontal, horizontally);
        resample_vertical(horizontally, request->resized_width, &vertical, resized);
        for (unsigned long long y = 0u; y < request->output_height; ++y) {
            const unsigned char *source = resized +
                ((size_t)(request->crop_top + y) * (size_t)request->resized_width +
                 (size_t)request->crop_left) * 3u;
            memcpy(candidate.data + (size_t)y * (size_t)request->output_width * 3u,
                   source, (size_t)request->output_width * 3u);
        }
        candidate.schema_version = YVEX_IMAGE_SCHEMA_V1;
        candidate.format = YVEX_IMAGE_RGB8;
        candidate.width = request->output_width;
        candidate.height = request->output_height;
        candidate.channels = 3u;
        candidate.row_bytes = request->output_width * 3u;
        candidate.data_bytes = output_bytes;
        candidate.source_file_bytes = input->source_file_bytes;
        memcpy(candidate.source_identity, input->source_identity,
               sizeof(candidate.source_identity));
        if (!image_identity("yvex.image.rgb8.v1", candidate.data, (size_t)output_bytes,
                            candidate.width, candidate.height, candidate.content_identity))
            rc = image_refuse(err, YVEX_ERR_STATE, "image.resize.identity",
                              "resampled image identity could not be sealed");
        else candidate.complete = 1;
    }
    free(vertical.coefficients);
    free(vertical.bounds);
    free(horizontal.coefficients);
    free(horizontal.bounds);
    free(resized);
    free(horizontally);
    if (rc == YVEX_OK) {
        *output = candidate;
        yvex_error_clear(err);
    } else yvex_image_close(&candidate);
    return rc;
}

int yvex_image_resize_lanczos_rgb8(const yvex_image *input,
                                   const yvex_image_resize_request *request,
                                   yvex_image *output, yvex_error *err)
{
    return image_resize_rgb8(input, request, 3.0, lanczos_filter, output, err);
}

int yvex_image_resize_bicubic_rgb8(const yvex_image *input,
                                   const yvex_image_resize_request *request,
                                   yvex_image *output, yvex_error *err)
{
    return image_resize_rgb8(input, request, 2.0, bicubic_filter, output, err);
}

void yvex_image_close(yvex_image *image)
{
    if (!image) return;
    free(image->data);
    memset(image, 0, sizeof(*image));
}
