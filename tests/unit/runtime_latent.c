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
    return 0;
}
