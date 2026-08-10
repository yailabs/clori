#include "tests/test.h"

#include <yvex/internal/latent.h>

#include <math.h>
#include <string.h>

typedef struct { int cancelled, evaluations; } latent_fixture;

static int latent_evaluate(
    void *context, const float *video, unsigned long long video_values,
    const float *audio, unsigned long long audio_values,
    float video_timestep, float audio_timestep,
    float *video_velocity, float *audio_velocity, yvex_error *err)
{
    latent_fixture *fixture = context;
    unsigned long long index;
    (void)video_timestep; (void)audio_timestep;
    for (index = 0ull; index < video_values; ++index) video_velocity[index] = video[index] * 0.5f;
    for (index = 0ull; index < audio_values; ++index) audio_velocity[index] = audio[index] * -0.25f;
    fixture->evaluations++;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int latent_advance(
    float *output, const float *sample, const float *velocity,
    unsigned long long values, float timestep, float sigma, float sigma_next,
    yvex_error *err)
{
    unsigned long long index;
    float ratio = sigma_next / sigma;
    for (index = 0ull; index < values; ++index) {
        float denoised = sample[index] + (1.0f - timestep) * velocity[index];
        output[index] = ratio * sample[index] + (1.0f - ratio) * denoised;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int latent_cancel(void *context)
{
    return ((latent_fixture *)context)->cancelled;
}

static yvex_runtime_latent_request latent_request(latent_fixture *fixture)
{
    static const float video_sigmas[3] = {1.0f, 0.5f, 0.0f};
    static const float audio_sigmas[3] = {1.0f, 0.25f, 0.0f};
    static const char identity[] =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    yvex_runtime_latent_request request = {
        .schema_version = YVEX_RUNTIME_LATENT_SCHEMA_V1,
        .seed = 42ull, .video_values = 3ull, .audio_values = 2ull, .step_count = 2ull,
        .maximum_workspace_bytes = 5ull * sizeof(float) * 4ull,
        .video_sigmas = video_sigmas, .audio_sigmas = audio_sigmas,
        .plan_identity = identity, .evaluate = latent_evaluate, .advance = latent_advance,
        .execution_context = fixture, .cancel_requested = latent_cancel,
        .cancel_context = fixture,
    };
    return request;
}

static int test_packed_av_layout(void)
{
    static const char plan_identity[] =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    static const unsigned int temporal_pattern[5] = {1u, 4u, 4u, 4u, 4u};
    static const unsigned long long audio_width_indices[2] = {0ull, 3ull};
    yvex_runtime_av_layout_request request = {
        .schema_version = YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1,
        .text_tag = 1u, .audio_tag = 2u, .video_tag = 0u,
        .text_rows = 2ull, .audio_steps = 3ull, .audio_channels = 2ull,
        .video_frames = 2ull, .latent_height = 4ull, .latent_width = 8ull,
        .patch_height = 2ull, .patch_width = 2ull,
        .text_start = 0ull, .audio_start = 2ull, .video_start = 8ull, .packed_rows = 24ull,
        .audio_width_indices = audio_width_indices, .temporal_pattern = temporal_pattern,
        .temporal_pattern_count = 5ull, .maximum_workspace_bytes = 24ull * 5ull * sizeof(float),
        .temporal_scale = 5.0 / 3.0, .spatial_scale = 32.0, .media_time_origin = 2.0,
        .plan_identity = plan_identity,
    };
    float positions[72], repeated_positions[72], refused_positions[72];
    unsigned int tags[24], repeated_tags[24], refused_tags[24];
    unsigned int video[16], repeated_video[16], refused_video[16];
    unsigned int audio[6], repeated_audio[6], refused_audio[6];
    unsigned int text[2], repeated_text[2], refused_text[2];
    yvex_runtime_av_layout_output first_output = {
        positions, 72ull, tags, video, audio, text, 24ull, 16ull, 6ull, 2ull,
    };
    yvex_runtime_av_layout_output repeated_output = {
        repeated_positions, 72ull, repeated_tags, repeated_video, repeated_audio, repeated_text,
        24ull, 16ull, 6ull, 2ull,
    };
    yvex_runtime_av_layout_output refused_output = {
        refused_positions, 72ull, refused_tags, refused_video, refused_audio, refused_text,
        24ull, 16ull, 6ull, 2ull,
    };
    yvex_runtime_av_layout_result first, repeated, refused;
    yvex_error err;

    YVEX_TEST_ASSERT(
        yvex_runtime_av_layout_build(&request, &first_output, &first, &err) == YVEX_OK &&
            yvex_runtime_av_layout_build(&request, &repeated_output, &repeated, &err) == YVEX_OK &&
            first.complete && first.text_rows == 2ull && first.audio_rows == 6ull &&
            first.video_rows == 16ull && first.packed_rows == 24ull &&
            first.workspace_bytes == 24ull * 5ull * sizeof(float) &&
            memcmp(positions, repeated_positions, sizeof(positions)) == 0 &&
            memcmp(tags, repeated_tags, sizeof(tags)) == 0 &&
            memcmp(video, repeated_video, sizeof(video)) == 0 &&
            memcmp(audio, repeated_audio, sizeof(audio)) == 0 &&
            memcmp(text, repeated_text, sizeof(text)) == 0 &&
            strcmp(first.layout_identity, repeated.layout_identity) == 0 &&
            strcmp(first.layout_identity,
                   "87d07a2b731bd7ebc7c48b4d3ddd3207e04102e4648dd010cb1e3fa237039eb3") == 0,
        "packed audio-video layout is byte-deterministic");
    YVEX_TEST_ASSERT(
        positions[0] == 0.0f && positions[3] == 1.0f &&
            positions[2ull * 3ull] == 2.0f && positions[4ull * 3ull] == 4.0f &&
            positions[5ull * 3ull] == 2.0f && positions[8ull * 3ull] == 2.0f &&
            fabsf(positions[16ull * 3ull] - (float)(2.0 + 5.0 / 3.0)) < 1.0e-6f &&
            fabsf(positions[2ull * 3ull + 2ull] + 6.627417f) < 1.0e-5f &&
            fabsf(positions[5ull * 3ull + 2ull] - 27.313709f) < 1.0e-5f &&
            tags[0] == 1u && tags[2] == 2u && tags[8] == 0u &&
            text[1] == 1u && audio[5] == 7u && video[15] == 23u,
        "packed layout preserves source row order, modality tags, and FP64-derived rotary coordinates");
    memset(refused_positions, 0x5a, sizeof(refused_positions));
    memset(refused_tags, 0x5a, sizeof(refused_tags));
    request.maximum_workspace_bytes--;
    YVEX_TEST_ASSERT(
        yvex_runtime_av_layout_build(&request, &refused_output, &refused, &err) == YVEX_ERR_BOUNDS &&
            !refused.complete && ((unsigned char *)refused_positions)[0] == 0x5a &&
            ((unsigned char *)refused_tags)[0] == 0x5a,
        "packed layout refuses its budget transactionally");
    return 0;
}

int yvex_test_runtime_latent(void)
{
    latent_fixture first_fixture = {0}, second_fixture = {0}, cancelled = {.cancelled = 1};
    yvex_runtime_latent_request first_request = latent_request(&first_fixture);
    yvex_runtime_latent_request second_request = latent_request(&second_fixture);
    yvex_runtime_latent_request refused_request = latent_request(&cancelled);
    yvex_runtime_latent_result first, second, refused;
    float first_video[3], first_audio[2], second_video[3], second_audio[2];
    float refused_video[3], refused_audio[2];
    yvex_error err;

    memset(refused_video, 0x5a, sizeof(refused_video));
    memset(refused_audio, 0x5a, sizeof(refused_audio));
    YVEX_TEST_ASSERT(
        yvex_runtime_latent_execute(&first_request, first_video, 3ull, first_audio, 2ull,
                                    &first, &err) == YVEX_OK &&
            yvex_runtime_latent_execute(&second_request, second_video, 3ull, second_audio, 2ull,
                                        &second, &err) == YVEX_OK &&
            first.completed && first.completed_steps == 2ull && first.model_evaluations == 2ull &&
            first.peak_workspace_bytes == 5ull * sizeof(float) * 4ull &&
            first_fixture.evaluations == 2 && second_fixture.evaluations == 2 &&
            memcmp(first_video, second_video, sizeof(first_video)) == 0 &&
            memcmp(first_audio, second_audio, sizeof(first_audio)) == 0 &&
            strcmp(first.execution_identity, second.execution_identity) == 0,
        "paired latent execution is deterministic and completes every scheduled model evaluation");
    YVEX_TEST_ASSERT(
        yvex_runtime_latent_execute(&refused_request, refused_video, 3ull, refused_audio, 2ull,
                                    &refused, &err) == YVEX_ERR_CANCELLED &&
            !refused.completed && ((unsigned char *)refused_video)[0] == 0x5a &&
            ((unsigned char *)refused_audio)[0] == 0x5a,
        "cancelled latent execution leaves both output domains unpublished");
    refused_request = latent_request(&cancelled);
    refused_request.cancel_requested = NULL;
    refused_request.maximum_workspace_bytes--;
    YVEX_TEST_ASSERT(
        yvex_runtime_latent_execute(&refused_request, refused_video, 3ull, refused_audio, 2ull,
                                    &refused, &err) == YVEX_ERR_BOUNDS && !refused.completed,
        "paired latent execution refuses an undersized workspace budget");
    if (test_packed_av_layout() != 0) return 1;
    return 0;
}
