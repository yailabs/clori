/* Exercise the complete staged media transaction with tiny admitted fixtures. */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/internal/media.h>

#include "tests/test.h"

#define FIXTURE_PATH "tests/fixtures/gguf/valid-tokenizer-simple.gguf"
#define FIXTURE_FRAMES 124ull
#define FIXTURE_WIDTH 32ull
#define FIXTURE_HEIGHT 32ull
#define FIXTURE_AUDIO_STEPS 207ull
#define FIXTURE_BYTES 928ull
#define FIXTURE_IDENTITY "32da281a901bbee03982e7f6d9743fa43b6b9c50ae8e5393d93bf0a3be0a5d99"

typedef struct {
    unsigned long long condition_calls, latent_calls, video_calls, audio_calls;
    unsigned long long token_count, progress_mask;
    unsigned int token_ids[256];
    int fail_condition, cancel;
} media_fixture_context;

static media_fixture_context *active_fixture_context;

static int fixture_plan(
    yvex_runtime_av_plan *out, unsigned long long text_tokens,
    unsigned long long width, unsigned long long height,
    unsigned long long frames, unsigned int inference_steps, yvex_error *err)
{
    static const yvex_runtime_av_plan_policy policy = {
        .schema_version = YVEX_RUNTIME_AV_PLAN_SCHEMA_V1,
        .maximum_steps = 64u,
        .text_tag = 1u,
        .audio_tag = 2u,
        .video_tag = 0u,
        .frame_period = 17ull,
        .frame_remainder = 5ull,
        .video_latents_per_period = 5ull,
        .video_latent_remainder = 2ull,
        .spatial_ratio = 16ull,
        .patch_height = 2ull,
        .patch_width = 2ull,
        .audio_rate_numerator = 5ull,
        .audio_rate_denominator = 3ull,
        .audio_channels = 2ull,
        .video_value_width = 96ull,
        .audio_value_width = 32ull,
        .temporal_pattern = {1u, 4u, 4u, 4u, 4u},
        .temporal_pattern_count = 5u,
        .video_sigma_shift = 12.0f,
        .audio_sigma_shift = 3.0f,
        .temporal_scale = 5.0 / 3.0,
        .spatial_scale = 32.0,
        .identity_domain = "yvex.test.runtime-media.plan.v1",
        .target_identity =
            "1111111111111111111111111111111111111111111111111111111111111111",
        .source_revision = "fixture",
    };
    return yvex_runtime_av_plan_build(
        &policy, text_tokens, width, height, frames, inference_steps, out, err);
}

static int fixture_layout(
    const yvex_runtime_av_plan *plan, const yvex_runtime_av_layout_output *output,
    yvex_runtime_av_layout_result *result, yvex_error *err)
{
    return yvex_runtime_av_layout_from_plan(plan, output, result, err);
}

static int fixture_admit(
    const char *component, const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, const yvex_artifact_admission_options *options,
    yvex_complete_artifact_admission *out, yvex_artifact_admission_evidence *evidence,
    yvex_artifact_admission_failure *failure, yvex_error *err)
{
    const char *admission_identity = NULL;
    yvex_artifact_admission_evidence local_evidence;
    (void)gguf;
    (void)options;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (component && strcmp(component, "text_encoder") == 0)
        admission_identity =
            "1111111111111111111111111111111111111111111111111111111111111111";
    else if (component && strcmp(component, "transformer") == 0)
        admission_identity =
            "2222222222222222222222222222222222222222222222222222222222222222";
    else if (component && strcmp(component, "video_vae") == 0)
        admission_identity =
            "3333333333333333333333333333333333333333333333333333333333333333";
    else if (component && strcmp(component, "audio_vae") == 0)
        admission_identity =
            "4444444444444444444444444444444444444444444444444444444444444444";
    if (!admission_identity ||
        !artifact || !tensors || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "test.runtime-media.admit",
                       "fixture component admission inputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX;
    out->tensor_count = yvex_tensor_table_count(tensors);
    out->payload_bytes = 128ull;
    out->file_bytes = yvex_artifact_size(artifact);
    out->materialization_input_ready = 1;
    out->complete = 1;
    yvex_core_text_copy(out->artifact_path, sizeof(out->artifact_path),
                        yvex_artifact_path(artifact));
    yvex_core_text_copy(out->artifact_identity, sizeof(out->artifact_identity),
                        FIXTURE_IDENTITY);
    yvex_core_text_copy(out->profile_identity, sizeof(out->profile_identity),
                        "fixture-profile");
    yvex_core_text_copy(out->writer_plan_identity, sizeof(out->writer_plan_identity),
                        "fixture-writer-plan");
    yvex_core_text_copy(out->logical_component_identity,
                        sizeof(out->logical_component_identity),
                        "3333333333333333333333333333333333333333333333333333333333333333");
    yvex_core_text_copy(out->admission_identity, sizeof(out->admission_identity),
                        admission_identity);
    if (yvex_artifact_snapshot_get(artifact, &out->file_snapshot, err) != YVEX_OK)
        return yvex_error_code(err);
    return yvex_artifact_admission_authenticate(
        artifact, out, options, evidence ? evidence : &local_evidence,
        failure, err);
}

static int fixture_condition(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, yvex_backend_kind backend_kind,
    const unsigned int *tokens,
    unsigned long long token_count, unsigned long long layer_count,
    float *output, unsigned long long output_capacity,
    unsigned long long maximum_host_bytes, unsigned long long maximum_device_bytes,
    yvex_runtime_av_conditioning_result *result, yvex_error *err)
{
    media_fixture_context *context = active_fixture_context;
    unsigned long long index, expected;
    (void)artifact;
    (void)gguf;
    (void)tensors;
    (void)backend_kind;
    (void)maximum_device_bytes;
    context->condition_calls++;
    if (token_count <= sizeof(context->token_ids) / sizeof(context->token_ids[0])) {
        context->token_count = token_count;
        memcpy(context->token_ids, tokens, (size_t)token_count * sizeof(tokens[0]));
    }
    if (context->fail_condition) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.runtime-media.condition",
                       "requested conditioning refusal");
        return YVEX_ERR_STATE;
    }
    if (!tokens || !token_count || layer_count != 1ull ||
        !yvex_core_u64_mul(token_count, 5120ull, &expected) ||
        expected != output_capacity || maximum_host_bytes < expected * sizeof(float)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "test.runtime-media.condition",
                       "fixture conditioning extent is inconsistent");
        return YVEX_ERR_BOUNDS;
    }
    for (index = 0ull; index < output_capacity; ++index)
        output[index] = (float)((index + tokens[0]) % 17ull) / 16.0f;
    memset(result, 0, sizeof(*result));
    result->token_count = token_count;
    result->hidden_width = 5120ull;
    result->layer_count = 1ull;
    result->resident_bytes = 128ull;
    result->kernel_launches = 2ull;
    result->device_bytes = 128ull;
    yvex_core_text_copy(result->residency_identity, sizeof(result->residency_identity),
                        "4444444444444444444444444444444444444444444444444444444444444444");
    yvex_core_text_copy(result->execution_identity, sizeof(result->execution_identity),
                        "5555555555555555555555555555555555555555555555555555555555555555");
    result->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int fixture_latent(
    const yvex_runtime_av_plan *plan, const yvex_runtime_av_latent_context *execution,
    unsigned long long seed,
    unsigned long long maximum_workspace_bytes, float *video,
    unsigned long long video_capacity, float *audio, unsigned long long audio_capacity,
    yvex_runtime_latent_result *latent_result,
    yvex_runtime_latent_evaluator_result *evaluator_result, yvex_error *err)
{
    media_fixture_context *context = active_fixture_context;
    yvex_runtime_latent_observation observation = {0};
    unsigned long long index;
    context->latent_calls++;
    if (!execution || !execution->transformer_session || !plan || !execution->conditioning ||
        !execution->conditioning_capacity ||
        !yvex_sha256_hex_valid(execution->conditioning_identity) || !execution->layout ||
        !execution->layout_result || !execution->layout_result->complete ||
        !execution->timestep_indices || execution->timestep_capacity != plan->packed_rows ||
        execution->block_count != 50ull || seed != 42ull ||
        maximum_workspace_bytes < (1ull << 20u) ||
        video_capacity != plan->video_rows * plan->video_value_width ||
        audio_capacity != plan->audio_rows * plan->audio_value_width) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "test.runtime-media.latent",
                       "fixture latent execution inputs are inconsistent");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0ull; index < video_capacity; ++index) video[index] = 0.0f;
    for (index = 0ull; index < audio_capacity; ++index) audio[index] = 0.0f;
    memset(latent_result, 0, sizeof(*latent_result));
    latent_result->schema_version = YVEX_RUNTIME_LATENT_SCHEMA_V1;
    latent_result->video_values = video_capacity;
    latent_result->audio_values = audio_capacity;
    latent_result->completed_steps = 1ull;
    latent_result->model_evaluations = plan->model_evaluations;
    latent_result->peak_workspace_bytes = 4096ull;
    yvex_core_text_copy(latent_result->execution_identity,
                        sizeof(latent_result->execution_identity),
                        "6666666666666666666666666666666666666666666666666666666666666666");
    latent_result->completed = 1;
    memset(evaluator_result, 0, sizeof(*evaluator_result));
    evaluator_result->schema_version = YVEX_RUNTIME_LATENT_EVALUATOR_SCHEMA_V1;
    evaluator_result->model_evaluations = plan->model_evaluations;
    evaluator_result->kernel_launches = 3ull;
    evaluator_result->peak_device_bytes = 256ull;
    yvex_core_text_copy(evaluator_result->residency_identity,
                        sizeof(evaluator_result->residency_identity),
                        "7777777777777777777777777777777777777777777777777777777777777777");
    yvex_core_text_copy(evaluator_result->evaluator_identity,
                        sizeof(evaluator_result->evaluator_identity),
                        "8888888888888888888888888888888888888888888888888888888888888888");
    yvex_core_text_copy(evaluator_result->execution_chain_identity,
                        sizeof(evaluator_result->execution_chain_identity),
                        "9999999999999999999999999999999999999999999999999999999999999999");
    evaluator_result->complete = 1;
    if (execution->observe) {
        observation.schema_version = YVEX_RUNTIME_LATENT_OBSERVATION_SCHEMA_V1;
        observation.stage = YVEX_RUNTIME_LATENT_OBSERVATION_ADVANCED;
        observation.completed_steps = 1ull;
        observation.video_values = video_capacity;
        observation.audio_values = audio_capacity;
        observation.video_state = video;
        observation.audio_state = audio;
        if (execution->observe(execution->observer_context, &observation, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int fixture_video(
    yvex_runtime_component_session *session,
    const yvex_runtime_av_video_decode_options *options,
    yvex_runtime_av_video_decode_result *result,
    yvex_component_execution_failure *failure, yvex_error *err)
{
    media_fixture_context *context = active_fixture_context;
    unsigned long long index;
    (void)failure;
    context->video_calls++;
    if (!session || !options || !options->output || !options->output_capacity ||
        options->max_workspace_bytes < (1ull << 20u) ||
        (options->cancelled && options->cancelled(options->cancellation_context))) {
        yvex_error_set(err, YVEX_ERR_CANCELLED, "test.runtime-media.video",
                       "fixture video execution was cancelled or malformed");
        return YVEX_ERR_CANCELLED;
    }
    for (index = 0ull; index < options->output_capacity; ++index)
        options->output[index] = (float)(index % 251ull) / 250.0f;
    memset(result, 0, sizeof(*result));
    result->output_values = options->output_capacity;
    result->kernel_launches = 1ull;
    result->device_bytes = 512ull;
    yvex_core_text_copy(result->execution_identity, sizeof(result->execution_identity),
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    result->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int fixture_audio(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, yvex_backend_kind backend_kind,
    const yvex_runtime_av_audio_decode_options *options,
    unsigned long long maximum_device_bytes, yvex_runtime_av_audio_decode_result *result,
    yvex_component_execution_failure *failure, yvex_error *err)
{
    media_fixture_context *context = active_fixture_context;
    unsigned long long index, samples;
    (void)artifact;
    (void)gguf;
    (void)tensors;
    (void)backend_kind;
    (void)maximum_device_bytes;
    (void)failure;
    context->audio_calls++;
    if (!options || !options->latent || options->batch != 2ull ||
        options->latent_steps != FIXTURE_AUDIO_STEPS ||
        !yvex_core_u64_mul(options->latent_steps, 800ull, &samples) ||
        options->output_capacity != options->batch * samples ||
        options->max_workspace_bytes < (1ull << 20u) ||
        (options->cancelled && options->cancelled(options->cancellation_context))) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "test.runtime-media.audio",
                       "fixture audio execution inputs are inconsistent");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0ull; index < options->output_capacity; ++index) options->output[index] = 0.0f;
    memset(result, 0, sizeof(*result));
    result->batch = options->batch;
    result->samples_per_channel = samples;
    result->output_values = options->output_capacity;
    result->kernel_launches = 4ull;
    result->device_bytes = 1024ull;
    result->peak_workspace_bytes = 8192ull;
    yvex_core_text_copy(result->residency_identity, sizeof(result->residency_identity),
                        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    yvex_core_text_copy(result->execution_identity, sizeof(result->execution_identity),
                        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    result->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int fixture_cancelled(void *opaque)
{
    return ((media_fixture_context *)opaque)->cancel;
}

static int fixture_progress(void *opaque, const yvex_runtime_media_progress *progress,
                            yvex_error *err)
{
    media_fixture_context *context = opaque;
    if (!context || !progress ||
        progress->schema_version != YVEX_RUNTIME_MEDIA_PROGRESS_SCHEMA_V1 ||
        progress->kind > YVEX_RUNTIME_MEDIA_PROGRESS_PUBLICATION_COMPLETE) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "test.runtime-media.progress",
                       "typed media progress is required");
        return YVEX_ERR_INVALID_ARG;
    }
    context->progress_mask |= 1ull << progress->kind;
    yvex_error_clear(err);
    return YVEX_OK;
}

static yvex_runtime_av_generation_request fixture_request(
    media_fixture_context *context, const char *path,
    float video_mean[24], float video_std[24],
    float audio_mean[32], float audio_std[32],
    float pixel_mean[3], float pixel_std[3])
{
    yvex_runtime_av_generation_request request = {0};
    unsigned long long index;
    for (index = 0ull; index < 24ull; ++index) {
        video_mean[index] = 0.0f;
        video_std[index] = 1.0f;
    }
    for (index = 0ull; index < 32ull; ++index) {
        audio_mean[index] = 0.0f;
        audio_std[index] = 1.0f;
    }
    for (index = 0ull; index < 3ull; ++index) {
        pixel_mean[index] = 0.0f;
        pixel_std[index] = 1.0f;
    }
    request.schema_version = YVEX_RUNTIME_AV_GENERATION_SCHEMA_V1;
    request.target = "fixture-av";
    request.prompt = "hello";
    request.output_path = path;
    request.text_artifact_path = FIXTURE_PATH;
    request.transformer_artifact_path = FIXTURE_PATH;
    request.video_artifact_path = FIXTURE_PATH;
    request.audio_artifact_path = FIXTURE_PATH;
    request.source_identity =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    request.frames = FIXTURE_FRAMES;
    request.width = FIXTURE_WIDTH;
    request.height = FIXTURE_HEIGHT;
    request.fps_numerator = 24ull;
    request.fps_denominator = 1ull;
    request.audio_sample_rate = 32000ull;
    request.inference_steps = 1u;
    request.conditioning_layers = 1ull;
    request.transformer_blocks = 50ull;
    request.seed = 42ull;
    request.maximum_prompt_tokens = 256ull;
    request.maximum_packed_rows = 2048ull;
    request.maximum_host_bytes = 64ull << 20u;
    request.maximum_device_bytes = 1ull << 20u;
    request.maximum_workspace_bytes = 64ull << 20u;
    request.maximum_file_bytes = 16ull << 20u;
    request.component_backend = YVEX_BACKEND_KIND_CPU;
    request.video_temporal_ratio = 4ull;
    request.video_clip_length = 17ull;
    request.video_token_drop = 3ull;
    request.video_spatial_ratio = 16ull;
    request.video_tile_size = 32ull;
    request.video_minimum_tile_overlap = 16ull;
    request.video_mean = video_mean;
    request.video_std = video_std;
    request.audio_mean = audio_mean;
    request.audio_std = audio_std;
    request.pixel_mean = pixel_mean;
    request.pixel_std = pixel_std;
    request.video_channels = 24ull;
    request.audio_channels = 32ull;
    request.pixel_channels = 3ull;
    request.audio_output_channels = 2ull;
    request.audio_samples_per_step = 800ull;
    (void)context;
    request.plan_build = fixture_plan;
    request.layout_build = fixture_layout;
    request.component_admit = fixture_admit;
    request.condition = fixture_condition;
    request.latent = fixture_latent;
    request.video_decode = fixture_video;
    request.audio_decode = fixture_audio;
    request.cancel_requested = fixture_cancelled;
    request.cancel_context = context;
    request.observe_progress = fixture_progress;
    request.progress_context = context;
    return request;
}

static int read_file(const char *path, unsigned char **bytes, size_t *count)
{
    struct stat st;
    FILE *file;
    size_t read_count;
    if (!path || !bytes || !count || stat(path, &st) != 0 || st.st_size <= 0) return 0;
    *bytes = malloc((size_t)st.st_size);
    if (!*bytes) return 0;
    file = fopen(path, "rb");
    if (!file) {
        free(*bytes);
        *bytes = NULL;
        return 0;
    }
    read_count = fread(*bytes, 1u, (size_t)st.st_size, file);
    if (fclose(file) != 0 || read_count != (size_t)st.st_size) {
        free(*bytes);
        *bytes = NULL;
        return 0;
    }
    *count = (size_t)st.st_size;
    return 1;
}

static int fixture_copy(const char *destination)
{
    unsigned char buffer[4096];
    int input = -1, output = -1, result = 0;
    ssize_t count;
    input = open(FIXTURE_PATH, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    output = open(destination, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (input < 0 || output < 0) goto done;
    while ((count = read(input, buffer, sizeof(buffer))) > 0) {
        size_t offset = 0u;
        while (offset < (size_t)count) {
            ssize_t written = write(output, buffer + offset, (size_t)count - offset);
            if (written <= 0) goto done;
            offset += (size_t)written;
        }
    }
    result = count == 0 && fsync(output) == 0;
done:
    if (input >= 0) (void)close(input);
    if (output >= 0 && close(output) != 0) result = 0;
    return result;
}

static int fixture_lease_path(const char *path, const char *cache_root,
                              char output[YVEX_ARTIFACT_PATH_CAP])
{
    yvex_artifact_options options = {0};
    yvex_artifact_reopen_lease lease;
    yvex_artifact *artifact = NULL;
    yvex_error err;
    int rc;
    options.path = path;
    options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_reopen_lease_check(
            artifact, FIXTURE_IDENTITY, cache_root, &lease, &err);
    if (rc == YVEX_OK && lease.verified)
        yvex_core_text_copy(output, YVEX_ARTIFACT_PATH_CAP, lease.path);
    else
        rc = YVEX_ERR_STATE;
    yvex_artifact_close(artifact);
    return rc == YVEX_OK;
}

static unsigned long long summary_mode_count(
    const yvex_runtime_media_model_summary *summary,
    yvex_artifact_verification_mode mode)
{
    unsigned long long count = 0ull, index;
    for (index = 0ull; summary && index < summary->component_count; ++index)
        count += summary->components[index].evidence.verification_mode == mode;
    return count;
}

static int test_composite_verified_reopen(void)
{
    static const char *const names[] = {"text.gguf", "transformer.gguf",
                                        "video.gguf", "audio.gguf"};
    media_fixture_context context = {0};
    float video_mean[24], video_std[24], audio_mean[32], audio_std[32];
    float pixel_mean[3], pixel_std[3];
    char root[] = "/tmp/yvex-media-reopen-XXXXXX";
    char paths[4][YVEX_ARTIFACT_PATH_CAP], cache_root[YVEX_ARTIFACT_PATH_CAP];
    char leases[5][YVEX_ARTIFACT_PATH_CAP] = {{0}};
    char artifact_dir[YVEX_ARTIFACT_PATH_CAP], receipt_root[YVEX_ARTIFACT_PATH_CAP];
    yvex_runtime_av_generation_request request;
    yvex_runtime_media_model_open_options options = {0};
    yvex_runtime_media_model_summary summary;
    yvex_runtime_media_model *model = NULL;
    yvex_error err;
    struct timespec times[2] = {{0, UTIME_NOW}, {0, UTIME_NOW}};
    unsigned char changed = 0u;
    unsigned long long index;
    int descriptor, rc;

    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "composite reopen root");
    YVEX_TEST_ASSERT(snprintf(cache_root, sizeof(cache_root), "%s/cache", root) > 0,
                     "composite reopen cache root");
    for (index = 0ull; index < 4ull; ++index) {
        YVEX_TEST_ASSERT(snprintf(paths[index], sizeof(paths[index]), "%s/%s", root,
                                  names[index]) > 0 && fixture_copy(paths[index]),
                         "composite component fixture copy");
    }
    request = fixture_request(&context, "/tmp/not-generated.avi", video_mean, video_std,
                              audio_mean, audio_std, pixel_mean, pixel_std);
    request.text_artifact_path = paths[0];
    request.transformer_artifact_path = paths[1];
    request.video_artifact_path = paths[2];
    request.audio_artifact_path = paths[3];
    options.schema_version = YVEX_RUNTIME_MEDIA_MODEL_OPEN_SCHEMA_V1;
    options.artifact_reopen_cache_root = cache_root;
    rc = yvex_runtime_media_model_open(&model, &request, &options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && model && summary.component_count == 4ull &&
                         summary.engine_generation &&
                         summary.resource_count == 4ull &&
                         summary.resource_generation == 4ull &&
                         summary.mapped_package_bytes == summary.artifact_bytes &&
                         !summary.prepared_bytes && !summary.resident_host_bytes &&
                         !summary.resident_device_bytes &&
                         summary.artifact_bytes_hashed == 4ull * FIXTURE_BYTES &&
                         summary_mode_count(&summary, YVEX_ARTIFACT_VERIFICATION_FULL_HASH) == 4ull,
                     "four-component cold admission hashes and leases every component");
    yvex_runtime_media_model_close(&model);
    for (index = 0ull; index < 4ull; ++index)
        YVEX_TEST_ASSERT(fixture_lease_path(paths[index], cache_root, leases[index]),
                         "cold component lease is independently valid");
    rc = yvex_runtime_media_model_open(&model, &request, &options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && model && !summary.artifact_bytes_hashed &&
                         summary_mode_count(
                             &summary, YVEX_ARTIFACT_VERIFICATION_VERIFIED_REOPEN) == 4ull,
                     "four unchanged components reopen without full byte hashing");
    yvex_runtime_media_model_close(&model);

    YVEX_TEST_ASSERT(utimensat(AT_FDCWD, paths[1], times, 0) == 0,
                     "one component snapshot changes without changing bytes");
    rc = yvex_runtime_media_model_open(&model, &request, &options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && model && summary.artifact_bytes_hashed == FIXTURE_BYTES &&
                         summary_mode_count(&summary, YVEX_ARTIFACT_VERIFICATION_FULL_HASH) == 1ull &&
                         summary_mode_count(
                             &summary, YVEX_ARTIFACT_VERIFICATION_VERIFIED_REOPEN) == 3ull,
                     "one stale component falls back without invalidating three warm hits");
    yvex_runtime_media_model_close(&model);
    YVEX_TEST_ASSERT(fixture_lease_path(paths[1], cache_root, leases[4]),
                     "selective fallback publishes one replacement snapshot lease");

    descriptor = open(leases[2], O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    YVEX_TEST_ASSERT(descriptor >= 0 && pwrite(descriptor, &changed, 1u, 0) == 1 &&
                         fsync(descriptor) == 0 && close(descriptor) == 0,
                     "one component receipt is malformed independently");
    rc = yvex_runtime_media_model_open(&model, &request, &options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && model && summary.artifact_bytes_hashed == FIXTURE_BYTES &&
                         summary_mode_count(
                             &summary, YVEX_ARTIFACT_VERIFICATION_FALLBACK_FULL_HASH) == 1ull &&
                         summary.components[2].evidence.lease_repaired &&
                         summary_mode_count(
                             &summary, YVEX_ARTIFACT_VERIFICATION_VERIFIED_REOPEN) == 3ull,
                     "malformed component receipt repairs after full verification");
    yvex_runtime_media_model_close(&model);

    descriptor = open(paths[3], O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    changed = 0xffu;
    YVEX_TEST_ASSERT(descriptor >= 0 && pwrite(descriptor, &changed, 1u, 927) == 1 &&
                         fsync(descriptor) == 0 && close(descriptor) == 0,
                     "component mismatch fixture changes bytes without changing extent");
    YVEX_TEST_ASSERT(yvex_runtime_media_model_open(
                         &model, &request, &options, &summary, &err) != YVEX_OK && !model,
                     "changed component bytes fail closed instead of publishing a lease");

    for (index = 0ull; index < 5ull; ++index)
        YVEX_TEST_ASSERT(unlink(leases[index]) == 0, "component receipt cleanup");
    for (index = 0ull; index < 4ull; ++index)
        YVEX_TEST_ASSERT(unlink(paths[index]) == 0, "component fixture cleanup");
    YVEX_TEST_ASSERT(snprintf(artifact_dir, sizeof(artifact_dir),
                              "%s/artifact-reopen/%s", cache_root,
                              FIXTURE_IDENTITY) > 0 &&
                         snprintf(receipt_root, sizeof(receipt_root),
                                  "%s/artifact-reopen", cache_root) > 0 &&
                         rmdir(artifact_dir) == 0 && rmdir(receipt_root) == 0 &&
                         rmdir(cache_root) == 0 && rmdir(root) == 0,
                     "composite reopen fixtures clean up narrowly");
    return 0;
}

static int test_generation_transaction(void)
{
    static const char creative_prompt[] =
        "HD stars over a high desert, a seed falling near 20 draft horses, "
        "5 seconds later the MOVimento della camera begins";
    media_fixture_context first_context = {0}, second_context = {0};
    float first_video_mean[24], first_video_std[24], first_audio_mean[32];
    float first_audio_std[32], first_pixel_mean[3], first_pixel_std[3];
    float second_video_mean[24], second_video_std[24], second_audio_mean[32];
    float second_audio_std[32], second_pixel_mean[3], second_pixel_std[3];
    char first_path[160], second_path[160];
    yvex_runtime_av_generation_request first, second;
    yvex_runtime_av_generation_result first_result, second_result;
    yvex_runtime_media_model_summary model_summary, repeated_summary;
    yvex_runtime_media_model *model = NULL, *repeated_model = NULL;
    unsigned char *first_bytes = NULL, *second_bytes = NULL;
    size_t first_count = 0u, second_count = 0u;
    yvex_error err;
    int rc;
    snprintf(first_path, sizeof(first_path),
             "build/tests/tmp/runtime-media-%ld-a.avi", (long)getpid());
    snprintf(second_path, sizeof(second_path),
             "build/tests/tmp/runtime-media-%ld-b.avi", (long)getpid());
    unlink(first_path);
    unlink(second_path);
    first = fixture_request(&first_context, first_path, first_video_mean, first_video_std,
                            first_audio_mean, first_audio_std,
                            first_pixel_mean, first_pixel_std);
    second = fixture_request(&second_context, second_path, second_video_mean, second_video_std,
                             second_audio_mean, second_audio_std,
                             second_pixel_mean, second_pixel_std);
    first.prompt = creative_prompt;
    second.prompt = creative_prompt;
    rc = yvex_runtime_media_model_open(&model, &first, NULL, &model_summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && model && model_summary.complete &&
                         model_summary.component_count == 4ull &&
                         yvex_sha256_hex_valid(model_summary.model_identity),
                     "persistent media model admits four components once");
    rc = yvex_runtime_media_model_open(
        &repeated_model, &second, NULL, &repeated_summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && repeated_model &&
                         strcmp(model_summary.model_identity,
                                repeated_summary.model_identity) == 0 &&
                         model_summary.engine_generation !=
                             repeated_summary.engine_generation &&
                         model_summary.resource_count == 4ull &&
                         repeated_summary.resource_count == 4ull,
                     "package identity is stable while engine generations remain distinct");
    yvex_runtime_media_model_close(&repeated_model);
    active_fixture_context = &first_context;
    rc = yvex_runtime_media_model_generate(model, &first, &first_result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && first_result.complete,
                     "complete staged media transaction");
    YVEX_TEST_ASSERT(first_result.frames == FIXTURE_FRAMES &&
                         first_result.width == FIXTURE_WIDTH &&
                         first_result.height == FIXTURE_HEIGHT &&
                         first_result.audio_samples == 165333ull,
                     "exact synchronized media geometry");
    YVEX_TEST_ASSERT(first_context.condition_calls == 1ull &&
                         first_context.latent_calls == 1ull &&
                         first_context.video_calls == 7ull &&
                         first_context.audio_calls == 1ull,
                     "all staged component phases executed once per admitted schedule");
    YVEX_TEST_ASSERT(first_context.token_count &&
                         first_context.progress_mask ==
                             ((1ull << (YVEX_RUNTIME_MEDIA_PROGRESS_PUBLICATION_COMPLETE + 1u)) -
                              1ull),
                     "opaque creative prompt reaches conditioning and every real phase is visible");
    active_fixture_context = &second_context;
    rc = yvex_runtime_media_model_generate(model, &second, &second_result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && second_result.complete,
                     "repeat staged media transaction");
    YVEX_TEST_ASSERT_STREQ(first_result.execution_identity, second_result.execution_identity,
                           "deterministic end-to-end execution identity");
    YVEX_TEST_ASSERT_STREQ(first_result.file_identity, second_result.file_identity,
                           "deterministic end-to-end file identity");
    YVEX_TEST_ASSERT(first_context.token_count == second_context.token_count &&
                         memcmp(first_context.token_ids, second_context.token_ids,
                                (size_t)first_context.token_count * sizeof(unsigned int)) == 0,
                     "submitted creative prompt reaches conditioning byte-for-byte deterministically");
    YVEX_TEST_ASSERT(read_file(first_path, &first_bytes, &first_count) &&
                         read_file(second_path, &second_bytes, &second_count) &&
                         first_count == second_count &&
                         memcmp(first_bytes, second_bytes, first_count) == 0,
                     "repeat end-to-end outputs are byte-identical");
    YVEX_TEST_ASSERT(first_count > 12u && memcmp(first_bytes, "RIFF", 4u) == 0 &&
                         memcmp(first_bytes + 8u, "AVI ", 4u) == 0,
                     "staged transaction published a playable-container signature");
    free(first_bytes);
    free(second_bytes);
    yvex_runtime_media_model_close(&model);
    unlink(first_path);
    unlink(second_path);
    return 0;
}

static int test_generation_refusals(void)
{
    media_fixture_context context = {0};
    float video_mean[24], video_std[24], audio_mean[32], audio_std[32];
    float pixel_mean[3], pixel_std[3];
    char path[160];
    yvex_runtime_av_generation_request request;
    yvex_runtime_av_generation_result result;
    yvex_runtime_media_model_summary model_summary;
    yvex_runtime_media_model *model = NULL;
    yvex_error err;
    int rc;
    snprintf(path, sizeof(path), "build/tests/tmp/runtime-media-%ld-refuse.avi", (long)getpid());
    unlink(path);
    request = fixture_request(&context, path, video_mean, video_std, audio_mean, audio_std,
                              pixel_mean, pixel_std);
    active_fixture_context = &context;
    rc = yvex_runtime_media_model_open(&model, &request, NULL, &model_summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && model, "refusal fixture media model opens");
    request.maximum_device_bytes++;
    rc = yvex_runtime_media_model_generate(model, &request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !result.complete,
                     "request cannot mutate the opened media model contract");
    request.maximum_device_bytes--;
    yvex_runtime_media_model_close(&model);
    request.component_backend = YVEX_BACKEND_KIND_METAL;
    rc = yvex_runtime_av_generate(&request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !result.complete,
                     "unknown component placement is refused");
    request.component_backend = YVEX_BACKEND_KIND_CPU;
    request.maximum_packed_rows = 1ull;
    rc = yvex_runtime_av_generate(&request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS && !result.complete && access(path, F_OK) != 0,
                     "oversized packed media plan is refused before latent execution");
    request.maximum_packed_rows = 2048ull;
    context.fail_condition = 1;
    rc = yvex_runtime_av_generate(&request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && !result.complete && access(path, F_OK) != 0,
                     "component failure publishes no media");
    context.fail_condition = 0;
    context.cancel = 1;
    rc = yvex_runtime_av_generate(&request, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_CANCELLED && !result.complete && access(path, F_OK) != 0,
                     "cancellation publishes no media");
    return 0;
}

int yvex_test_runtime_media(void)
{
    if (test_composite_verified_reopen() != 0) return 1;
    if (test_generation_transaction() != 0) return 1;
    if (test_generation_refusals() != 0) return 1;
    return 0;
}
