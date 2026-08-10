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
        .plan_identity = identity, .evaluator_identity = identity,
        .evaluate = latent_evaluate, .advance = latent_advance,
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

static int test_evaluator_evidence(void)
{
    static const char identity_a[] =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    static const char identity_b[] =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    const char *identities[2] = {identity_a, identity_b};
    unsigned long long facts[2] = {50ull, 66280430144ull};
    yvex_runtime_latent_evaluator_evidence first = {0}, repeated = {0};
    yvex_runtime_latent_evaluator_result first_result, repeated_result;
    char binding[65], repeated_binding[65];
    yvex_error err;

    YVEX_TEST_ASSERT(
        yvex_runtime_latent_binding_identity(
            "test.latent.binding.v1", identities, 2ull, facts, 2ull, binding, &err) == YVEX_OK &&
            yvex_runtime_latent_binding_identity(
                "test.latent.binding.v1", identities, 2ull, facts, 2ull,
                repeated_binding, &err) == YVEX_OK && strcmp(binding, repeated_binding) == 0,
        "latent evaluator binding identity is deterministic");
    YVEX_TEST_ASSERT(
        yvex_runtime_latent_evaluator_begin(
            &first, "test.latent.chain.v1", binding, &err) == YVEX_OK &&
            yvex_runtime_latent_evaluator_record(
                &first, identity_a, identity_a, 10ull, 20ull, 30ull, 40ull, &err) == YVEX_OK &&
            yvex_runtime_latent_evaluator_record(
                &first, identity_a, identity_b, 11ull, 21ull, 31ull, 41ull, &err) == YVEX_OK &&
            yvex_runtime_latent_evaluator_finish(&first, 2ull, &first_result, &err) == YVEX_OK &&
            first_result.complete && first_result.model_evaluations == 2ull &&
            first_result.kernel_launches == 21ull && first_result.h2d_bytes == 41ull &&
            first_result.d2h_bytes == 61ull && first_result.peak_device_bytes == 41ull,
        "latent evaluator evidence seals exact execution and resource facts");
    YVEX_TEST_ASSERT(
        yvex_runtime_latent_evaluator_begin(
            &repeated, "test.latent.chain.v1", binding, &err) == YVEX_OK &&
            yvex_runtime_latent_evaluator_record(
                &repeated, identity_a, identity_a, 10ull, 20ull, 30ull, 40ull, &err) == YVEX_OK &&
            yvex_runtime_latent_evaluator_record(
                &repeated, identity_a, identity_b, 11ull, 21ull, 31ull, 41ull, &err) == YVEX_OK &&
            yvex_runtime_latent_evaluator_finish(
                &repeated, 2ull, &repeated_result, &err) == YVEX_OK &&
            strcmp(first_result.execution_chain_identity,
                   repeated_result.execution_chain_identity) == 0,
        "latent evaluator evidence is deterministic across independent executions");
    YVEX_TEST_ASSERT(
        yvex_runtime_latent_evaluator_begin(
            &repeated, "test.latent.chain.v1", binding, &err) == YVEX_OK &&
            yvex_runtime_latent_evaluator_record(
                &repeated, identity_a, identity_a, 1ull, 1ull, 1ull, 1ull, &err) == YVEX_OK &&
            yvex_runtime_latent_evaluator_record(
                &repeated, identity_b, identity_b, 1ull, 1ull, 1ull, 1ull, &err) == YVEX_ERR_STATE,
        "latent evaluator evidence refuses a residency transition within one chain");
    return 0;
}

static int test_av_unpack(void)
{
    static const char identity[] =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    yvex_runtime_av_plan plan = {
        .schema_version = YVEX_RUNTIME_AV_PLAN_SCHEMA_V1,
        .video_latent_frames = 1ull, .video_latent_height = 2ull,
        .video_latent_width = 2ull, .audio_latent_steps = 2ull,
        .audio_rows = 4ull, .video_rows = 1ull, .patch_height = 2ull,
        .patch_width = 2ull, .audio_channels = 2ull,
        .video_value_width = 8ull, .audio_value_width = 3ull,
        .complete = 1,
    };
    const float video_rows[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    const float audio_rows[12] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
    };
    const float video_mean[2] = {10.0f, 20.0f}, video_std[2] = {2.0f, 3.0f};
    const float audio_mean[3] = {100.0f, 200.0f, 300.0f};
    float audio_std[3] = {1.0f, 2.0f, 3.0f};
    const float expected_video[8] = {10.0f, 12.0f, 14.0f, 16.0f, 32.0f, 35.0f, 38.0f, 41.0f};
    const float expected_audio[12] = {
        100.0f, 103.0f, 202.0f, 208.0f, 306.0f, 315.0f,
        106.0f, 109.0f, 214.0f, 220.0f, 324.0f, 333.0f,
    };
    yvex_runtime_av_unpack_request request = {
        .schema_version = YVEX_RUNTIME_AV_UNPACK_SCHEMA_V1,
        .plan = &plan, .video_rows = video_rows, .audio_rows = audio_rows,
        .video_row_capacity = 8ull, .audio_row_capacity = 12ull,
        .video_channel_mean = video_mean, .video_channel_std = video_std,
        .audio_channel_mean = audio_mean, .audio_channel_std = audio_std,
        .video_channel_count = 2ull, .audio_channel_count = 3ull,
        .maximum_workspace_bytes = 20ull * sizeof(float),
        .latent_execution_identity = identity,
    };
    float first_video[8], first_audio[12], second_video[8], second_audio[12];
    float refused_video[8], refused_audio[12];
    yvex_runtime_av_unpack_output first_output = {first_video, first_audio, 8ull, 12ull};
    yvex_runtime_av_unpack_output second_output = {second_video, second_audio, 8ull, 12ull};
    yvex_runtime_av_unpack_output refused_output = {refused_video, refused_audio, 8ull, 12ull};
    yvex_runtime_av_unpack_result first, second, refused;
    yvex_error err;

    memcpy(plan.identity, identity, sizeof(identity));
    YVEX_TEST_ASSERT(
        yvex_runtime_av_unpack(&request, &first_output, &first, &err) == YVEX_OK &&
            yvex_runtime_av_unpack(&request, &second_output, &second, &err) == YVEX_OK &&
            first.complete && first.video_channels == 2ull && first.video_frames == 1ull &&
            first.video_height == 2ull && first.video_width == 2ull &&
            first.audio_batch == 2ull && first.audio_channels == 3ull &&
            first.audio_steps == 2ull && first.peak_workspace_bytes == 20ull * sizeof(float) &&
            memcmp(first_video, expected_video, sizeof(expected_video)) == 0 &&
            memcmp(first_audio, expected_audio, sizeof(expected_audio)) == 0 &&
            memcmp(first_video, second_video, sizeof(first_video)) == 0 &&
            memcmp(first_audio, second_audio, sizeof(first_audio)) == 0 &&
            strcmp(first.input_identity, second.input_identity) == 0,
        "packed audio-video latents unpack, denormalize, and identify deterministically");
    memset(refused_video, 0x5a, sizeof(refused_video));
    memset(refused_audio, 0x5a, sizeof(refused_audio));
    request.maximum_workspace_bytes--;
    YVEX_TEST_ASSERT(
        yvex_runtime_av_unpack(&request, &refused_output, &refused, &err) == YVEX_ERR_BOUNDS &&
            !refused.complete && ((unsigned char *)refused_video)[0] == 0x5a &&
            ((unsigned char *)refused_audio)[0] == 0x5a,
        "audio-video unpack refuses insufficient workspace without publication");
    request.maximum_workspace_bytes++;
    audio_std[1] = 0.0f;
    YVEX_TEST_ASSERT(
        yvex_runtime_av_unpack(&request, &refused_output, &refused, &err) == YVEX_ERR_FORMAT &&
            !refused.complete && ((unsigned char *)refused_video)[0] == 0x5a,
        "audio-video unpack refuses invalid source normalization");
    return 0;
}

static int test_video_reconstruction_plan(void)
{
    static const char source_identity[] =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    yvex_runtime_av_video_reconstruction_request request = {
        .schema_version = YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1,
        .frames = 124ull, .width = 768ull, .height = 768ull,
        .latent_frames = 37ull, .latent_width = 48ull, .latent_height = 48ull,
        .temporal_ratio = 4ull, .clip_length = 17ull, .token_drop = 3ull,
        .spatial_ratio = 16ull, .tile_size = 256ull, .minimum_tile_overlap = 64ull,
        .source_identity = source_identity,
    };
    yvex_runtime_av_video_reconstruction_plan first, repeated, refused;
    yvex_error err;

    YVEX_TEST_ASSERT(
        yvex_runtime_av_video_reconstruction_plan_build(&request, &first, &err) == YVEX_OK &&
            yvex_runtime_av_video_reconstruction_plan_build(&request, &repeated, &err) == YVEX_OK &&
            first.complete && first.tokens_per_chunk == 5ull && first.token_overlap == 2ull &&
            first.frame_pre_padding == 3ull && first.frame_overlap == 5ull &&
            first.temporal_chunks == 7ull && first.decode_latent_frames == 7ull &&
            first.decode_frames == 28ull && first.pad_tokens == 0ull &&
            first.tile_y_count == 4ull && first.tile_x_count == 4ull &&
            first.tile_y_start[0] == 0ull && first.tile_y_start[1] == 160ull &&
            first.tile_y_start[2] == 336ull && first.tile_y_start[3] == 512ull &&
            first.tile_y_overlap[0] == 96ull && first.tile_y_overlap[1] == 80ull &&
            first.tile_y_overlap[2] == 80ull && first.total_decode_calls == 112ull &&
            strcmp(first.identity,
                   "6c4ae7302490072c4afd87ccbf7062d5e83898050af7c32aebc70e0d18686226") == 0 &&
            strcmp(first.identity, repeated.identity) == 0,
        "video reconstruction plan preserves exact temporal chunks and spatial tiles");
    request.latent_frames = 36ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_av_video_reconstruction_plan_build(&request, &refused, &err) ==
            YVEX_ERR_FORMAT && !refused.complete,
        "video reconstruction plan refuses mismatched latent duration");
    request.latent_frames = 37ull; request.frames = 5ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_av_video_reconstruction_plan_build(&request, &refused, &err) ==
            YVEX_ERR_FORMAT && !refused.complete,
        "video reconstruction plan refuses a zero-chunk synthetic duration");
    request.frames = 124ull; request.width = 770ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_av_video_reconstruction_plan_build(&request, &refused, &err) ==
            YVEX_ERR_INVALID_ARG && !refused.complete,
        "video reconstruction plan refuses a non-latent-aligned canvas");
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
    if (test_evaluator_evidence() != 0) return 1;
    if (test_av_unpack() != 0) return 1;
    if (test_video_reconstruction_plan() != 0) return 1;
    return 0;
}
