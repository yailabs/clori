/*
 * Publish bounded synchronized audio-video without delegating media execution.
 *
 * AVI keeps the first native path codec-free: bottom-up BGR24 frames and interleaved signed
 * PCM are independently playable, while exact rational cadence and atomic publication remain
 * owned by YVEX. Container identity never includes the operator's local destination path.
 */
#define _POSIX_C_SOURCE 200809L

#include <yvex/internal/core.h>
#include <yvex/internal/io.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define AVI_FIXED_PREFIX_BYTES 324ull
#define AVI_FIXED_FILE_BYTES 332ull
#define AVI_HEADER_LIST_SIZE 292u
#define AVI_VIDEO_LIST_SIZE 116u
#define AVI_AUDIO_LIST_SIZE 92u

typedef struct {
    unsigned long long frame_stride, frame_bytes, audio_samples_used, audio_bytes;
    unsigned long long audio_block_align, audio_chunk_max, movi_bytes, index_bytes;
    unsigned long long file_bytes, maximum_bytes_per_second, suggested_buffer_bytes;
} avi_plan;

typedef struct {
    avi_plan plan;
} avi_validation;

static int media_fail(yvex_error *err, yvex_status status, const char *message)
{
    yvex_error_set(err, status, "io.media", message);
    return status;
}

static int bytes_append(yvex_core_bytes *bytes, const void *data, size_t count)
{
    return yvex_core_bytes_append(bytes, data, count);
}

static int avi_fourcc(yvex_core_bytes *bytes, const char value[4])
{
    return bytes_append(bytes, value, 4u);
}

static int avi_u16(yvex_core_bytes *bytes, unsigned int value)
{
    unsigned char encoded[2] = {
        (unsigned char)(value & 0xffu), (unsigned char)((value >> 8u) & 0xffu)};
    return bytes_append(bytes, encoded, sizeof(encoded));
}

static int avi_u32(yvex_core_bytes *bytes, unsigned long long value)
{
    unsigned char encoded[4] = {
        (unsigned char)(value & 0xffull), (unsigned char)((value >> 8u) & 0xffull),
        (unsigned char)((value >> 16u) & 0xffull), (unsigned char)((value >> 24u) & 0xffull)};
    return value <= UINT32_MAX && bytes_append(bytes, encoded, sizeof(encoded));
}

static unsigned long long avi_audio_boundary(
    unsigned long long frame, unsigned long long samples, unsigned long long frames)
{
    /* The file-size admission bounds this product below UINT64_MAX. */
    return (frame * samples) / frames;
}

static int avi_plan_build(
    const yvex_media_avi_request *request, avi_plan *plan, yvex_error *err)
{
    unsigned long long row_bytes, duration_samples, rounded_samples, video_rate_bytes;
    unsigned long long video_chunks, audio_chunks, entries, file_without_index;
    if (!request || !plan || request->schema_version != YVEX_MEDIA_AVI_SCHEMA_V1 ||
        !request->path || !request->path[0] || !request->video || !request->audio ||
        request->video_channels != 3ull || !request->frames || !request->width ||
        !request->height || request->width > 32767ull || request->height > 32767ull ||
        !request->fps_numerator || !request->fps_denominator ||
        request->fps_numerator > UINT32_MAX || request->fps_denominator > UINT32_MAX ||
        !request->audio_channels || request->audio_channels > 2ull ||
        !request->audio_samples || !request->audio_sample_rate ||
        request->audio_sample_rate > UINT32_MAX || !request->maximum_file_bytes ||
        (request->video_identity && !yvex_sha256_hex_valid(request->video_identity)) ||
        (request->audio_identity && !yvex_sha256_hex_valid(request->audio_identity)) ||
        (request->execution_identity && !yvex_sha256_hex_valid(request->execution_identity)))
        return media_fail(err, YVEX_ERR_INVALID_ARG,
                          "exact bounded AVI publication facts are required");
    memset(plan, 0, sizeof(*plan));
    if (!yvex_core_u64_mul(request->width, 3ull, &row_bytes) ||
        !yvex_core_u64_add(row_bytes, 3ull, &plan->frame_stride))
        return media_fail(err, YVEX_ERR_BOUNDS, "AVI video row extent overflowed");
    plan->frame_stride &= ~3ull;
    if (!yvex_core_u64_mul(plan->frame_stride, request->height, &plan->frame_bytes) ||
        !yvex_core_u64_mul(request->frames, request->audio_sample_rate, &duration_samples) ||
        !yvex_core_u64_mul(duration_samples, request->fps_denominator, &duration_samples) ||
        !yvex_core_u64_add(duration_samples, request->fps_numerator / 2ull,
                           &rounded_samples))
        return media_fail(err, YVEX_ERR_BOUNDS, "AVI stream extent overflowed");
    plan->audio_samples_used = rounded_samples / request->fps_numerator;
    if (!plan->audio_samples_used || plan->audio_samples_used > request->audio_samples ||
        !yvex_core_u64_mul(request->audio_channels, 2ull, &plan->audio_block_align) ||
        !yvex_core_u64_mul(plan->audio_samples_used, plan->audio_block_align,
                           &plan->audio_bytes) ||
        !yvex_core_u64_mul(plan->frame_bytes, request->frames, &video_chunks) ||
        !yvex_core_u64_mul(request->frames, 8ull, &audio_chunks) ||
        !yvex_core_u64_add(video_chunks, audio_chunks, &plan->movi_bytes) ||
        !yvex_core_u64_add(plan->movi_bytes, audio_chunks, &plan->movi_bytes) ||
        !yvex_core_u64_add(plan->movi_bytes, plan->audio_bytes, &plan->movi_bytes))
        return media_fail(err, YVEX_ERR_BOUNDS,
                          "AVI audio duration or interleaved extent is inconsistent");
    if (!yvex_core_u64_mul(request->frames, 2ull, &entries) ||
        !yvex_core_u64_mul(entries, 16ull, &plan->index_bytes) ||
        !yvex_core_u64_add(AVI_FIXED_FILE_BYTES, plan->movi_bytes, &file_without_index) ||
        !yvex_core_u64_add(file_without_index, plan->index_bytes, &plan->file_bytes) ||
        plan->file_bytes > request->maximum_file_bytes || plan->file_bytes < 8ull ||
        plan->file_bytes - 8ull > UINT32_MAX ||
        request->frames > ULLONG_MAX / plan->audio_samples_used)
        return media_fail(err, YVEX_ERR_BOUNDS, "AVI output exceeds its file budget");
    if (!yvex_core_u64_mul(plan->frame_bytes, request->fps_numerator,
                           &video_rate_bytes) ||
        !yvex_core_u64_add(video_rate_bytes, request->fps_denominator - 1ull,
                           &video_rate_bytes))
        return media_fail(err, YVEX_ERR_BOUNDS, "AVI video rate overflowed");
    video_rate_bytes /= request->fps_denominator;
    if (!yvex_core_u64_mul(request->audio_sample_rate, plan->audio_block_align,
                           &plan->maximum_bytes_per_second) ||
        !yvex_core_u64_add(plan->maximum_bytes_per_second, video_rate_bytes,
                           &plan->maximum_bytes_per_second) ||
        !yvex_core_u64_add(plan->audio_samples_used, request->frames - 1ull,
                           &plan->audio_chunk_max) ||
        !yvex_core_u64_mul(plan->audio_chunk_max / request->frames,
                           plan->audio_block_align, &plan->audio_chunk_max) ||
        plan->maximum_bytes_per_second > UINT32_MAX)
        return media_fail(err, YVEX_ERR_BOUNDS, "AVI stream rate overflowed");
    plan->suggested_buffer_bytes = plan->frame_bytes > plan->audio_chunk_max
                                       ? plan->frame_bytes : plan->audio_chunk_max;
    return YVEX_OK;
}

static int avi_main_header(
    yvex_core_bytes *bytes, const yvex_media_avi_request *request, const avi_plan *plan)
{
    unsigned long long micros =
        (1000000ull * request->fps_denominator + request->fps_numerator / 2ull) /
        request->fps_numerator;
    unsigned int index;
    if (!avi_fourcc(bytes, "RIFF") || !avi_u32(bytes, plan->file_bytes - 8ull) ||
        !avi_fourcc(bytes, "AVI ") || !avi_fourcc(bytes, "LIST") ||
        !avi_u32(bytes, AVI_HEADER_LIST_SIZE) || !avi_fourcc(bytes, "hdrl") ||
        !avi_fourcc(bytes, "avih") || !avi_u32(bytes, 56u) ||
        !avi_u32(bytes, micros) || !avi_u32(bytes, plan->maximum_bytes_per_second) ||
        !avi_u32(bytes, 0u) || !avi_u32(bytes, 0x110u) ||
        !avi_u32(bytes, request->frames) || !avi_u32(bytes, 0u) ||
        !avi_u32(bytes, 2u) || !avi_u32(bytes, plan->suggested_buffer_bytes) ||
        !avi_u32(bytes, request->width) || !avi_u32(bytes, request->height)) return 0;
    for (index = 0u; index < 4u; ++index)
        if (!avi_u32(bytes, 0u)) return 0;
    return 1;
}

static int avi_video_header(
    yvex_core_bytes *bytes, const yvex_media_avi_request *request, const avi_plan *plan)
{
    if (!avi_fourcc(bytes, "LIST") || !avi_u32(bytes, AVI_VIDEO_LIST_SIZE) ||
        !avi_fourcc(bytes, "strl") || !avi_fourcc(bytes, "strh") ||
        !avi_u32(bytes, 56u) || !avi_fourcc(bytes, "vids") ||
        !avi_fourcc(bytes, "DIB ") || !avi_u32(bytes, 0u) || !avi_u16(bytes, 0u) ||
        !avi_u16(bytes, 0u) || !avi_u32(bytes, 0u) ||
        !avi_u32(bytes, request->fps_denominator) ||
        !avi_u32(bytes, request->fps_numerator) || !avi_u32(bytes, 0u) ||
        !avi_u32(bytes, request->frames) || !avi_u32(bytes, plan->frame_bytes) ||
        !avi_u32(bytes, UINT32_MAX) || !avi_u32(bytes, 0u) ||
        !avi_u16(bytes, 0u) || !avi_u16(bytes, 0u) ||
        !avi_u16(bytes, (unsigned int)request->width) ||
        !avi_u16(bytes, (unsigned int)request->height) ||
        !avi_fourcc(bytes, "strf") || !avi_u32(bytes, 40u) ||
        !avi_u32(bytes, 40u) || !avi_u32(bytes, request->width) ||
        !avi_u32(bytes, request->height) || !avi_u16(bytes, 1u) ||
        !avi_u16(bytes, 24u) || !avi_u32(bytes, 0u) ||
        !avi_u32(bytes, plan->frame_bytes) || !avi_u32(bytes, 0u) ||
        !avi_u32(bytes, 0u) || !avi_u32(bytes, 0u) || !avi_u32(bytes, 0u)) return 0;
    return 1;
}

static int avi_audio_header(
    yvex_core_bytes *bytes, const yvex_media_avi_request *request, const avi_plan *plan)
{
    unsigned long long byte_rate = request->audio_sample_rate * plan->audio_block_align;
    if (!avi_fourcc(bytes, "LIST") || !avi_u32(bytes, AVI_AUDIO_LIST_SIZE) ||
        !avi_fourcc(bytes, "strl") || !avi_fourcc(bytes, "strh") ||
        !avi_u32(bytes, 56u) || !avi_fourcc(bytes, "auds") ||
        !avi_u32(bytes, 0u) || !avi_u32(bytes, 0u) || !avi_u16(bytes, 0u) ||
        !avi_u16(bytes, 0u) || !avi_u32(bytes, 0u) ||
        !avi_u32(bytes, plan->audio_block_align) || !avi_u32(bytes, byte_rate) ||
        !avi_u32(bytes, 0u) || !avi_u32(bytes, plan->audio_samples_used) ||
        !avi_u32(bytes, plan->audio_chunk_max) || !avi_u32(bytes, UINT32_MAX) ||
        !avi_u32(bytes, plan->audio_block_align) || !avi_u16(bytes, 0u) ||
        !avi_u16(bytes, 0u) || !avi_u16(bytes, 0u) || !avi_u16(bytes, 0u) ||
        !avi_fourcc(bytes, "strf") || !avi_u32(bytes, 16u) ||
        !avi_u16(bytes, 1u) || !avi_u16(bytes, (unsigned int)request->audio_channels) ||
        !avi_u32(bytes, request->audio_sample_rate) || !avi_u32(bytes, byte_rate) ||
        !avi_u16(bytes, (unsigned int)plan->audio_block_align) ||
        !avi_u16(bytes, 16u)) return 0;
    return 1;
}

static unsigned char avi_pixel(float value)
{
    if (value <= 0.0f) return 0u;
    if (value >= 1.0f) return 255u;
    return (unsigned char)(value * 255.0f);
}

static int avi_frame_append(
    yvex_core_bytes *bytes, const yvex_media_avi_request *request,
    const avi_plan *plan, unsigned long long frame, yvex_error *err)
{
    unsigned long long row, column, channel, pixels = request->height * request->width;
    unsigned long long row_padding = plan->frame_stride - request->width * 3ull;
    if (!avi_fourcc(bytes, "00db") || !avi_u32(bytes, plan->frame_bytes)) return 0;
    for (row = request->height; row > 0ull; --row) {
        for (column = 0ull; column < request->width; ++column)
            for (channel = 3ull; channel > 0ull; --channel) {
                float value = request->video[
                    ((channel - 1ull) * request->frames + frame) * pixels +
                    (row - 1ull) * request->width + column];
                unsigned char pixel;
                if (!isfinite(value)) {
                    media_fail(err, YVEX_ERR_FORMAT, "AVI video contains a non-finite value");
                    return 0;
                }
                pixel = avi_pixel(value);
                if (!bytes_append(bytes, &pixel, 1u)) return 0;
            }
        if (!yvex_core_bytes_append_zero(bytes, (size_t)row_padding)) return 0;
    }
    return 1;
}

static int16_t avi_sample(float value)
{
    if (value <= -1.0f) return INT16_MIN;
    if (value >= 1.0f) return INT16_MAX;
    return value >= 0.0f ? (int16_t)(value * 32767.0f + 0.5f)
                         : (int16_t)(value * 32768.0f - 0.5f);
}

static int avi_audio_append(
    yvex_core_bytes *bytes, const yvex_media_avi_request *request,
    const avi_plan *plan, unsigned long long frame, yvex_error *err)
{
    unsigned long long start = avi_audio_boundary(frame, plan->audio_samples_used, request->frames);
    unsigned long long stop = avi_audio_boundary(frame + 1ull, plan->audio_samples_used, request->frames);
    unsigned long long sample, channel, chunk_bytes = (stop - start) * plan->audio_block_align;
    if (!avi_fourcc(bytes, "01wb") || !avi_u32(bytes, chunk_bytes)) return 0;
    for (sample = start; sample < stop; ++sample)
        for (channel = 0ull; channel < request->audio_channels; ++channel) {
            float value = request->audio[channel * request->audio_samples + sample];
            int16_t pcm;
            unsigned char encoded[2];
            if (!isfinite(value)) {
                media_fail(err, YVEX_ERR_FORMAT, "AVI audio contains a non-finite value");
                return 0;
            }
            pcm = avi_sample(value);
            encoded[0] = (unsigned char)((uint16_t)pcm & 0xffu);
            encoded[1] = (unsigned char)(((uint16_t)pcm >> 8u) & 0xffu);
            if (!bytes_append(bytes, encoded, sizeof(encoded))) return 0;
        }
    return 1;
}

static int avi_streams_append(
    yvex_core_bytes *bytes, const yvex_media_avi_request *request,
    const avi_plan *plan, yvex_error *err)
{
    unsigned long long frame;
    if (!avi_fourcc(bytes, "LIST") || !avi_u32(bytes, plan->movi_bytes + 4ull) ||
        !avi_fourcc(bytes, "movi")) return 0;
    for (frame = 0ull; frame < request->frames; ++frame) {
        if (request->cancel_requested && request->cancel_requested(request->cancel_context)) {
            media_fail(err, YVEX_ERR_CANCELLED, "AVI publication was cancelled");
            return 0;
        }
        if (!avi_frame_append(bytes, request, plan, frame, err) ||
            !avi_audio_append(bytes, request, plan, frame, err)) return 0;
    }
    return 1;
}

static int avi_index_append(
    yvex_core_bytes *bytes, const yvex_media_avi_request *request, const avi_plan *plan)
{
    unsigned long long frame, offset = 4ull;
    if (!avi_fourcc(bytes, "idx1") || !avi_u32(bytes, plan->index_bytes)) return 0;
    for (frame = 0ull; frame < request->frames; ++frame) {
        unsigned long long start = avi_audio_boundary(frame, plan->audio_samples_used, request->frames);
        unsigned long long stop = avi_audio_boundary(frame + 1ull, plan->audio_samples_used,
                                                      request->frames);
        unsigned long long audio_bytes = (stop - start) * plan->audio_block_align;
        if (!avi_fourcc(bytes, "00db") || !avi_u32(bytes, 0x10u) ||
            !avi_u32(bytes, offset) || !avi_u32(bytes, plan->frame_bytes)) return 0;
        offset += 8ull + plan->frame_bytes;
        if (!avi_fourcc(bytes, "01wb") || !avi_u32(bytes, 0u) ||
            !avi_u32(bytes, offset) || !avi_u32(bytes, audio_bytes)) return 0;
        offset += 8ull + audio_bytes;
    }
    return offset == 4ull + plan->movi_bytes;
}

static unsigned long long avi_read_u32(const unsigned char *data)
{
    return (unsigned long long)data[0] | ((unsigned long long)data[1] << 8u) |
           ((unsigned long long)data[2] << 16u) | ((unsigned long long)data[3] << 24u);
}

static int avi_validate_candidate(
    int descriptor, size_t count, void *opaque, yvex_error *err)
{
    const avi_validation *validation = opaque;
    unsigned char prefix[AVI_FIXED_PREFIX_BYTES], tail[8];
    unsigned long long index_offset = AVI_FIXED_PREFIX_BYTES + validation->plan.movi_bytes;
    if ((unsigned long long)count != validation->plan.file_bytes ||
        pread(descriptor, prefix, sizeof(prefix), 0) != (ssize_t)sizeof(prefix) ||
        pread(descriptor, tail, sizeof(tail), (off_t)index_offset) != (ssize_t)sizeof(tail) ||
        memcmp(prefix, "RIFF", 4u) || avi_read_u32(prefix + 4u) != count - 8u ||
        memcmp(prefix + 8u, "AVI ", 4u) || memcmp(prefix + 12u, "LIST", 4u) ||
        memcmp(prefix + 20u, "hdrl", 4u) || memcmp(prefix + 312u, "LIST", 4u) ||
        avi_read_u32(prefix + 316u) != validation->plan.movi_bytes + 4ull ||
        memcmp(prefix + 320u, "movi", 4u) || memcmp(tail, "idx1", 4u) ||
        avi_read_u32(tail + 4u) != validation->plan.index_bytes) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "io.media.validate",
                       "AVI publication candidate failed structural validation");
        return YVEX_ERR_FORMAT;
    }
    return YVEX_OK;
}

static int avi_identities(
    const yvex_media_avi_request *request, const avi_plan *plan,
    const unsigned char *data, size_t count, yvex_media_avi_result *result)
{
    yvex_sha256 file_hash, publication, input;
    unsigned char file_digest[YVEX_SHA256_DIGEST_BYTES], digest[YVEX_SHA256_DIGEST_BYTES];
    const unsigned long long facts[] = {
        request->frames, request->width, request->height, request->fps_numerator,
        request->fps_denominator, request->audio_channels, plan->audio_samples_used,
        request->audio_sample_rate, plan->file_bytes,
    };
    unsigned long long index, video_values, audio_values;
    uint32_t bits;
    if (!yvex_core_u64_mul(request->video_channels, request->frames, &video_values) ||
        !yvex_core_u64_mul(video_values, request->height, &video_values) ||
        !yvex_core_u64_mul(video_values, request->width, &video_values) ||
        !yvex_core_u64_mul(request->audio_channels, request->audio_samples, &audio_values))
        return 0;
    if (request->video_identity)
        yvex_core_text_copy(result->video_identity, sizeof(result->video_identity),
                            request->video_identity);
    else {
        yvex_sha256_init(&input);
        if (!yvex_sha256_update_text(&input, "yvex.media.video.planar-f32.v1") ||
            !yvex_sha256_update_u64_be(&input, video_values)) return 0;
        for (index = 0ull; index < video_values; ++index) {
            memcpy(&bits, request->video + index, sizeof(bits));
            if (!yvex_sha256_update_u64_be(&input, bits)) return 0;
        }
        if (!yvex_sha256_final(&input, digest)) return 0;
        yvex_sha256_hex(digest, result->video_identity);
    }
    if (request->audio_identity)
        yvex_core_text_copy(result->audio_identity, sizeof(result->audio_identity),
                            request->audio_identity);
    else {
        yvex_sha256_init(&input);
        if (!yvex_sha256_update_text(&input, "yvex.media.audio.planar-f32.v1") ||
            !yvex_sha256_update_u64_be(&input, audio_values)) return 0;
        for (index = 0ull; index < audio_values; ++index) {
            memcpy(&bits, request->audio + index, sizeof(bits));
            if (!yvex_sha256_update_u64_be(&input, bits)) return 0;
        }
        if (!yvex_sha256_final(&input, digest)) return 0;
        yvex_sha256_hex(digest, result->audio_identity);
    }
    if (request->execution_identity)
        yvex_core_text_copy(result->execution_identity, sizeof(result->execution_identity),
                            request->execution_identity);
    else {
        yvex_sha256_init(&input);
        if (!yvex_sha256_update_text(&input, "yvex.media.decoded-inputs.v1") ||
            !yvex_sha256_update_text(&input, result->video_identity) ||
            !yvex_sha256_update_text(&input, result->audio_identity)) return 0;
        for (index = 0ull; index < sizeof(facts) / sizeof(facts[0]); ++index)
            if (!yvex_sha256_update_u64_be(&input, facts[index])) return 0;
        if (!yvex_sha256_final(&input, digest)) return 0;
        yvex_sha256_hex(digest, result->execution_identity);
    }
    yvex_sha256_init(&file_hash);
    if (!yvex_sha256_update(&file_hash, data, count) ||
        !yvex_sha256_final(&file_hash, file_digest)) return 0;
    yvex_sha256_hex(file_digest, result->file_identity);
    yvex_sha256_init(&publication);
    if (!yvex_sha256_update_text(&publication, "yvex.media.avi.bgr24-pcm-s16le.v1") ||
        !yvex_sha256_update_text(&publication, result->video_identity) ||
        !yvex_sha256_update_text(&publication, result->audio_identity) ||
        !yvex_sha256_update_text(&publication, result->execution_identity) ||
        !yvex_sha256_update_text(&publication, result->file_identity)) return 0;
    for (index = 0ull; index < sizeof(facts) / sizeof(facts[0]); ++index)
        if (!yvex_sha256_update_u64_be(&publication, facts[index])) return 0;
    if (!yvex_sha256_final(&publication, digest)) return 0;
    yvex_sha256_hex(digest, result->publication_identity);
    return 1;
}

int yvex_media_avi_publish(
    const yvex_media_avi_request *request, yvex_media_avi_result *result, yvex_error *err)
{
    avi_plan plan;
    avi_validation validation;
    yvex_core_bytes bytes = {0};
    yvex_core_file_result file_result;
    yvex_media_avi_result staged = {0};
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!result) return media_fail(err, YVEX_ERR_INVALID_ARG, "AVI result is required");
    rc = avi_plan_build(request, &plan, err);
    if (rc != YVEX_OK) return rc;
    if (plan.file_bytes > SIZE_MAX)
        return media_fail(err, YVEX_ERR_BOUNDS, "AVI output exceeds addressable memory");
    bytes.maximum = (size_t)plan.file_bytes;
    bytes.initial_capacity = (size_t)plan.file_bytes;
    if (!avi_main_header(&bytes, request, &plan) ||
        !avi_video_header(&bytes, request, &plan) ||
        !avi_audio_header(&bytes, request, &plan) ||
        !avi_streams_append(&bytes, request, &plan, err) ||
        !avi_index_append(&bytes, request, &plan) || bytes.count != plan.file_bytes) {
        rc = yvex_error_is_set(err) ? yvex_error_code(err) : YVEX_ERR_NOMEM;
        if (!yvex_error_is_set(err))
            media_fail(err, (yvex_status)rc, "AVI serialization did not reach its exact extent");
        yvex_core_free(bytes.data);
        return rc;
    }
    staged.schema_version = YVEX_MEDIA_AVI_SCHEMA_V1;
    staged.file_bytes = plan.file_bytes; staged.video_frames = request->frames;
    staged.audio_samples_used = plan.audio_samples_used;
    staged.audio_samples_trimmed = request->audio_samples - plan.audio_samples_used;
    if (!yvex_core_u64_mul(request->frames, request->fps_denominator,
                           &staged.video_duration_numerator)) {
        yvex_core_free(bytes.data);
        return media_fail(err, YVEX_ERR_BOUNDS, "AVI duration identity overflowed");
    }
    staged.video_duration_denominator = request->fps_numerator;
    staged.peak_workspace_bytes = plan.file_bytes;
    if (!avi_identities(request, &plan, bytes.data, bytes.count, &staged))
        rc = media_fail(err, YVEX_ERR_STATE, "AVI identities could not be sealed");
    else if (request->cancel_requested && request->cancel_requested(request->cancel_context))
        rc = media_fail(err, YVEX_ERR_CANCELLED, "AVI publication was cancelled");
    else if ((rc = yvex_core_mkdir_parent(request->path, "io.media", err)) == YVEX_OK) {
        memset(&validation, 0, sizeof(validation));
        validation.plan = plan;
        rc = yvex_core_file_publish_noreplace(
            request->path, bytes.data, bytes.count, NULL, avi_validate_candidate,
            &validation, &file_result, err);
    }
    if (rc == YVEX_OK) {
        staged.complete = 1; *result = staged; yvex_error_clear(err);
    }
    yvex_core_free(bytes.data);
    return rc;
}
