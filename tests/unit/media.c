/* Exercises native synchronized media serialization and transactional publication. */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <yvex/internal/io.h>

#include "tests/test.h"

#define TEST_FRAMES 3ull
#define TEST_WIDTH 2ull
#define TEST_HEIGHT 2ull
#define TEST_AUDIO_SAMPLES 4005ull
#define TEST_FILE_BYTES 16524ull

static unsigned long long read_u32(const unsigned char *data)
{
    return (unsigned long long)data[0] | ((unsigned long long)data[1] << 8u) |
           ((unsigned long long)data[2] << 16u) | ((unsigned long long)data[3] << 24u);
}

static int16_t read_i16(const unsigned char *data)
{
    uint16_t bits = (uint16_t)data[0] | (uint16_t)((uint16_t)data[1] << 8u);
    int16_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static unsigned char expected_pixel(float value)
{
    if (value <= 0.0f) return 0u;
    if (value >= 1.0f) return 255u;
    return (unsigned char)(value * 255.0f);
}

static int16_t expected_sample(float value)
{
    if (value <= -1.0f) return INT16_MIN;
    if (value >= 1.0f) return INT16_MAX;
    return value >= 0.0f ? (int16_t)(value * 32767.0f + 0.5f)
                         : (int16_t)(value * 32768.0f - 0.5f);
}

static int verify_media_payload(
    const unsigned char *bytes, size_t count,
    const float video[3u * TEST_FRAMES * TEST_HEIGHT * TEST_WIDTH],
    const float audio[2u * TEST_AUDIO_SAMPLES])
{
    unsigned long long cursor = 324ull, frame, row, column, channel;
    unsigned long long pixels = TEST_WIDTH * TEST_HEIGHT;
    for (frame = 0ull; frame < TEST_FRAMES; ++frame) {
        unsigned long long frame_bytes, frame_end, start, stop, audio_bytes, sample;
        if (cursor + 8ull > count || memcmp(bytes + cursor, "00db", 4u) != 0) return 0;
        frame_bytes = read_u32(bytes + cursor + 4ull);
        frame_end = cursor + 8ull + frame_bytes;
        if (frame_bytes != 16ull || frame_end > count) return 0;
        cursor += 8ull;
        for (row = TEST_HEIGHT; row > 0ull; --row) {
            for (column = 0ull; column < TEST_WIDTH; ++column)
                for (channel = 3ull; channel > 0ull; --channel) {
                    float value = video[((channel - 1ull) * TEST_FRAMES + frame) * pixels +
                                        (row - 1ull) * TEST_WIDTH + column];
                    if (bytes[cursor++] != expected_pixel(value)) return 0;
                }
            if (bytes[cursor++] != 0u || bytes[cursor++] != 0u) return 0;
        }
        if (cursor != frame_end || cursor + 8ull > count ||
            memcmp(bytes + cursor, "01wb", 4u) != 0) return 0;
        start = (frame * 4000ull) / TEST_FRAMES;
        stop = ((frame + 1ull) * 4000ull) / TEST_FRAMES;
        audio_bytes = read_u32(bytes + cursor + 4ull);
        if (audio_bytes != (stop - start) * 4ull || cursor + 8ull + audio_bytes > count) return 0;
        cursor += 8ull;
        for (sample = start; sample < stop; ++sample)
            for (channel = 0ull; channel < 2ull; ++channel) {
                float value = audio[channel * TEST_AUDIO_SAMPLES + sample];
                if (read_i16(bytes + cursor) != expected_sample(value)) return 0;
                cursor += 2ull;
            }
    }
    return cursor + 8ull <= count && memcmp(bytes + cursor, "idx1", 4u) == 0 &&
           read_u32(bytes + cursor + 4ull) == TEST_FRAMES * 2ull * 16ull;
}

static int cancelled(void *context)
{
    (void)context;
    return 1;
}

static int read_file(const char *path, unsigned char **data, size_t *count)
{
    struct stat st;
    FILE *file;
    size_t read_count;
    if (!path || !data || !count || stat(path, &st) != 0 || st.st_size < 0) return 0;
    *data = malloc((size_t)st.st_size);
    if (!*data) return 0;
    file = fopen(path, "rb");
    if (!file) {
        free(*data);
        *data = NULL;
        return 0;
    }
    read_count = fread(*data, 1u, (size_t)st.st_size, file);
    if (fclose(file) != 0 || read_count != (size_t)st.st_size) {
        free(*data);
        *data = NULL;
        return 0;
    }
    *count = (size_t)st.st_size;
    return 1;
}

static void fill_inputs(float video[3u * TEST_FRAMES * TEST_HEIGHT * TEST_WIDTH],
                        float audio[2u * TEST_AUDIO_SAMPLES])
{
    unsigned long long index;
    for (index = 0ull; index < 3ull * TEST_FRAMES * TEST_HEIGHT * TEST_WIDTH; ++index)
        video[index] = (float)(index % 17ull) / 16.0f;
    for (index = 0ull; index < TEST_AUDIO_SAMPLES; ++index) {
        audio[index] = (float)((long long)(index % 41ull) - 20ll) / 20.0f;
        audio[TEST_AUDIO_SAMPLES + index] = -audio[index];
    }
}

static yvex_media_avi_request request_make(
    const char *path, const float *video, const float *audio)
{
    yvex_media_avi_request request = {
        .schema_version = YVEX_MEDIA_AVI_SCHEMA_V1,
        .path = path,
        .video = video,
        .audio = audio,
        .video_channels = 3ull,
        .frames = TEST_FRAMES,
        .width = TEST_WIDTH,
        .height = TEST_HEIGHT,
        .fps_numerator = 24ull,
        .fps_denominator = 1ull,
        .audio_channels = 2ull,
        .audio_samples = TEST_AUDIO_SAMPLES,
        .audio_sample_rate = 32000ull,
        .maximum_file_bytes = 1ull << 20u,
        .video_identity =
            "1111111111111111111111111111111111111111111111111111111111111111",
        .audio_identity =
            "2222222222222222222222222222222222222222222222222222222222222222",
        .execution_identity =
            "3333333333333333333333333333333333333333333333333333333333333333",
    };
    return request;
}

static int test_valid_and_deterministic(void)
{
    float video[3u * TEST_FRAMES * TEST_HEIGHT * TEST_WIDTH];
    float *audio = calloc(2u * TEST_AUDIO_SAMPLES, sizeof(*audio));
    char path_a[160], path_b[160];
    unsigned char *bytes_a = NULL, *bytes_b = NULL;
    size_t count_a = 0u, count_b = 0u;
    yvex_media_avi_request request;
    yvex_media_avi_result first, second;
    yvex_error err;
    int rc;
    YVEX_TEST_ASSERT(audio != NULL, "allocate media audio fixture");
    fill_inputs(video, audio);
    snprintf(path_a, sizeof(path_a), "build/tests/tmp/media-%ld-a.avi", (long)getpid());
    snprintf(path_b, sizeof(path_b), "build/tests/tmp/media-%ld-b.avi", (long)getpid());
    unlink(path_a);
    unlink(path_b);
    request = request_make(path_a, video, audio);
    rc = yvex_media_avi_publish(&request, &first, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && first.complete, "publish native AVI");
    YVEX_TEST_ASSERT(first.file_bytes == TEST_FILE_BYTES, "exact AVI byte count");
    YVEX_TEST_ASSERT(first.video_frames == TEST_FRAMES, "exact AVI frame count");
    YVEX_TEST_ASSERT(first.audio_samples_used == 4000ull, "exact synchronized PCM samples");
    YVEX_TEST_ASSERT(first.audio_samples_trimmed == 5ull, "decoded PCM tail trimmed");
    YVEX_TEST_ASSERT(first.video_duration_numerator == 3ull &&
                         first.video_duration_denominator == 24ull,
                     "exact rational video duration");
    request.path = path_b;
    rc = yvex_media_avi_publish(&request, &second, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && second.complete, "repeat native AVI publication");
    YVEX_TEST_ASSERT_STREQ(first.file_identity, second.file_identity,
                           "deterministic file identity");
    YVEX_TEST_ASSERT_STREQ(first.publication_identity, second.publication_identity,
                           "deterministic publication identity");
    YVEX_TEST_ASSERT(read_file(path_a, &bytes_a, &count_a) &&
                         read_file(path_b, &bytes_b, &count_b),
                     "read published AVIs");
    YVEX_TEST_ASSERT(count_a == TEST_FILE_BYTES && count_a == count_b &&
                         memcmp(bytes_a, bytes_b, count_a) == 0,
                     "repeat publications are byte-identical");
    YVEX_TEST_ASSERT(memcmp(bytes_a, "RIFF", 4u) == 0 &&
                         read_u32(bytes_a + 4u) == count_a - 8u &&
                         memcmp(bytes_a + 8u, "AVI ", 4u) == 0 &&
                         memcmp(bytes_a + 320u, "movi", 4u) == 0,
                     "AVI structure and extent");
    YVEX_TEST_ASSERT(verify_media_payload(bytes_a, count_a, video, audio),
                     "independent AVI readback preserves frame coordinates, BGR, stride, and PCM");
    request.path = path_a;
    rc = yvex_media_avi_publish(&request, &second, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "existing publication is never overwritten");
    free(bytes_a);
    free(bytes_b);
    free(audio);
    unlink(path_a);
    unlink(path_b);
    return 0;
}

static int test_refusals(void)
{
    float video[3u * TEST_FRAMES * TEST_HEIGHT * TEST_WIDTH];
    float *audio = calloc(2u * TEST_AUDIO_SAMPLES, sizeof(*audio));
    char path[160];
    yvex_media_avi_request request;
    yvex_media_avi_result result;
    yvex_error err;
    int rc;
    YVEX_TEST_ASSERT(audio != NULL, "allocate refusal audio fixture");
    fill_inputs(video, audio);
    snprintf(path, sizeof(path), "build/tests/tmp/media-%ld-refuse.avi", (long)getpid());
    unlink(path);
    request = request_make(path, video, audio);
    request.audio_samples = 3999ull;
    rc = yvex_media_avi_publish(&request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS && !result.complete,
                     "insufficient PCM duration refused");
    request = request_make(path, video, audio);
    request.maximum_file_bytes = TEST_FILE_BYTES - 1ull;
    rc = yvex_media_avi_publish(&request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS && !result.complete, "file budget enforced");
    request = request_make(path, video, audio);
    video[0] = NAN;
    rc = yvex_media_avi_publish(&request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !result.complete, "non-finite pixel refused");
    video[0] = 0.0f;
    audio[0] = INFINITY;
    rc = yvex_media_avi_publish(&request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !result.complete, "non-finite PCM refused");
    audio[0] = 0.0f;
    request.cancel_requested = cancelled;
    rc = yvex_media_avi_publish(&request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_CANCELLED && !result.complete,
                     "cancelled publication remains invisible");
    YVEX_TEST_ASSERT(access(path, F_OK) != 0, "refusals publish no file");
    free(audio);
    return 0;
}

int yvex_test_media(void)
{
    if (test_valid_and_deterministic() != 0) return 1;
    if (test_refusals() != 0) return 1;
    return 0;
}
