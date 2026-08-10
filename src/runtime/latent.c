/*
 * Evolve paired latent state transactionally while family code supplies model and solver facts.
 * One seeded stream initializes video first and audio second; no failed step publishes output.
 */
#include <yvex/internal/latent.h>
#include <yvex/internal/core.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

static int latent_refuse(yvex_error *err, yvex_status status, const char *message)
{
    yvex_error_set(err, status, "runtime.latent", message);
    return status;
}

static int latent_hash_f32(yvex_sha256 *hash, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

static int latent_state_identity(
    const char *domain, const float *values, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_sha256_update_u64(&hash, count)) return 0;
    for (index = 0ull; index < count; ++index)
        if (!latent_hash_f32(&hash, values[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int latent_request_validate(
    const yvex_runtime_latent_request *request,
    unsigned long long video_capacity, unsigned long long audio_capacity,
    unsigned long long *total, unsigned long long *bytes, yvex_error *err)
{
    unsigned long long index, peak;
    if (!request || request->schema_version != YVEX_RUNTIME_LATENT_SCHEMA_V1 ||
        !request->video_values || !request->audio_values || !request->step_count ||
        request->video_values > video_capacity || request->audio_values > audio_capacity ||
        !request->video_sigmas || !request->audio_sigmas ||
        !request->plan_identity || !yvex_sha256_hex_valid(request->plan_identity) ||
        !request->evaluate || !request->advance ||
        !yvex_core_u64_add(request->video_values, request->audio_values, total) ||
        !yvex_core_u64_mul(*total, sizeof(float), bytes) ||
        !yvex_core_u64_mul(*bytes, 4ull, &peak) ||
        peak > request->maximum_workspace_bytes || *bytes > SIZE_MAX)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "bounded paired latent request and output are required");
    for (index = 0ull; index <= request->step_count; ++index) {
        float video = request->video_sigmas[index], audio = request->audio_sigmas[index];
        if (!isfinite(video) || !isfinite(audio) || video < 0.0f || audio < 0.0f ||
            video > 1.0f || audio > 1.0f ||
            (index && (video >= request->video_sigmas[index - 1ull] ||
                       audio >= request->audio_sigmas[index - 1ull])))
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "paired latent sigma grids must decrease to zero");
    }
    if (request->video_sigmas[request->step_count] != 0.0f ||
        request->audio_sigmas[request->step_count] != 0.0f)
        return latent_refuse(err, YVEX_ERR_FORMAT,
                             "paired latent sigma grids require terminal zero");
    return YVEX_OK;
}

static int latent_execution_identity(
    const yvex_runtime_latent_request *request,
    const yvex_runtime_latent_result *result,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.latent.execution.v1") ||
        !yvex_sha256_update_text(&hash, request->plan_identity) ||
        !yvex_sha256_update_u64(&hash, request->seed) ||
        !yvex_sha256_update_u64(&hash, request->video_values) ||
        !yvex_sha256_update_u64(&hash, request->audio_values) ||
        !yvex_sha256_update_u64(&hash, request->step_count) ||
        !yvex_sha256_update_text(&hash, result->initialization.normal_identity) ||
        !yvex_sha256_update_text(&hash, result->initial_state_identity) ||
        !yvex_sha256_update_text(&hash, result->final_state_identity)) return 0;
    for (index = 0ull; index <= request->step_count; ++index)
        if (!latent_hash_f32(&hash, request->video_sigmas[index]) ||
            !latent_hash_f32(&hash, request->audio_sigmas[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_runtime_latent_execute(
    const yvex_runtime_latent_request *request,
    float *video_output, unsigned long long video_capacity,
    float *audio_output, unsigned long long audio_capacity,
    yvex_runtime_latent_result *result, yvex_error *err)
{
    yvex_runtime_latent_result staged = {0};
    float *storage = NULL, *state, *next, *velocity, *swap;
    unsigned long long total = 0ull, bytes = 0ull, storage_bytes, step;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!video_output || !audio_output || !result)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "latent outputs and result storage are required");
    rc = latent_request_validate(request, video_capacity, audio_capacity, &total, &bytes, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_mul(bytes, 3ull, &storage_bytes) || storage_bytes > SIZE_MAX)
        return latent_refuse(err, YVEX_ERR_BOUNDS, "latent workspace extent overflowed");
    storage = yvex_core_malloc((size_t)storage_bytes);
    if (!storage) return latent_refuse(err, YVEX_ERR_NOMEM,
                                       "latent workspace allocation failed");
    state = storage;
    next = state + total;
    velocity = next + total;
    rc = yvex_runtime_sampling_normal_f32(
        state, total, total, request->seed, bytes, &staged.initialization, err);
    if (rc == YVEX_OK &&
        !latent_state_identity("yvex.runtime.latent.initial.v1", state, total,
                               staged.initial_state_identity))
        rc = latent_refuse(err, YVEX_ERR_STATE, "initial latent identity failed");
    for (step = 0ull; rc == YVEX_OK && step < request->step_count; ++step) {
        float video_timestep = 1.0f - request->video_sigmas[step];
        float audio_timestep = 1.0f - request->audio_sigmas[step];
        if (request->cancel_requested && request->cancel_requested(request->cancel_context)) {
            rc = latent_refuse(err, YVEX_ERR_CANCELLED, "latent iteration was cancelled");
            break;
        }
        rc = request->evaluate(
            request->execution_context, state, request->video_values,
            state + request->video_values, request->audio_values,
            video_timestep, audio_timestep, velocity,
            velocity + request->video_values, err);
        if (rc == YVEX_OK)
            rc = request->advance(
                next, state, velocity,
                request->video_values, video_timestep, request->video_sigmas[step],
                request->video_sigmas[step + 1ull], err);
        if (rc == YVEX_OK)
            rc = request->advance(
                next + request->video_values,
                state + request->video_values, velocity + request->video_values,
                request->audio_values, audio_timestep, request->audio_sigmas[step],
                request->audio_sigmas[step + 1ull], err);
        if (rc == YVEX_OK) { swap = state; state = next; next = swap; staged.completed_steps++; }
    }
    if (rc == YVEX_OK && request->cancel_requested &&
        request->cancel_requested(request->cancel_context))
        rc = latent_refuse(err, YVEX_ERR_CANCELLED,
                           "latent publication was cancelled");
    if (rc == YVEX_OK &&
        !latent_state_identity("yvex.runtime.latent.final.v1", state, total,
                               staged.final_state_identity))
        rc = latent_refuse(err, YVEX_ERR_STATE, "final latent identity failed");
    staged.schema_version = YVEX_RUNTIME_LATENT_SCHEMA_V1;
    staged.video_values = request->video_values;
    staged.audio_values = request->audio_values;
    staged.model_evaluations = staged.completed_steps;
    staged.peak_workspace_bytes = bytes * 4ull;
    if (rc == YVEX_OK && !latent_execution_identity(request, &staged, staged.execution_identity))
        rc = latent_refuse(err, YVEX_ERR_STATE, "latent execution identity failed");
    if (rc == YVEX_OK) {
        memcpy(video_output, state, (size_t)(request->video_values * sizeof(float)));
        memcpy(audio_output, state + request->video_values,
               (size_t)(request->audio_values * sizeof(float)));
        staged.completed = 1;
        *result = staged;
        yvex_error_clear(err);
    }
    yvex_core_free(storage);
    return rc;
}
