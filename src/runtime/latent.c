/*
 * Evolve paired latent state transactionally while family code supplies model and solver facts.
 * One seeded stream initializes video first and audio second; no failed step publishes output.
 */
#include <yvex/internal/latent.h>
#include <yvex/internal/core.h>

#include <math.h>
#include <limits.h>
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

int yvex_runtime_latent_shifted_sigmas(
    float *output, unsigned int points, float shift, yvex_error *err)
{
    unsigned int index;
    if (!output || points < 2u || points > 65u || !isfinite(shift) || shift <= 0.0f)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "a bounded positive shifted sigma grid is required");
    for (index = 0u; index < points; ++index) {
        float base = (float)(points - 1u - index) / (float)(points - 1u);
        output[index] = shift * base / (1.0f + (shift - 1.0f) * base);
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_latent_plan_identity(
    const char *domain, const char *target, const char *source_revision,
    const unsigned long long *facts, unsigned long long fact_count,
    const float *video_sigmas, const float *audio_sigmas, unsigned int points,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    if (!domain || !domain[0] || !target || !target[0] || !source_revision ||
        !source_revision[0] || !facts || !fact_count || !video_sigmas || !audio_sigmas ||
        points < 2u || points > 65u || !output)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "latent plan identity requires bounded canonical facts");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) || !yvex_sha256_update_text(&hash, target) ||
        !yvex_sha256_update_text(&hash, source_revision))
        return latent_refuse(err, YVEX_ERR_STATE, "latent plan identity initialization failed");
    for (index = 0ull; index < fact_count; ++index)
        if (!yvex_sha256_update_u64_be(&hash, facts[index]))
            return latent_refuse(err, YVEX_ERR_STATE, "latent plan fact identity failed");
    for (index = 0ull; index < points; ++index) {
        uint32_t video_bits, audio_bits;
        if (!isfinite(video_sigmas[index]) || !isfinite(audio_sigmas[index]))
            return latent_refuse(err, YVEX_ERR_FORMAT, "latent plan sigma is not finite");
        memcpy(&video_bits, &video_sigmas[index], sizeof(video_bits));
        memcpy(&audio_bits, &audio_sigmas[index], sizeof(audio_bits));
        if (!yvex_sha256_update_u64_be(&hash, video_bits) ||
            !yvex_sha256_update_u64_be(&hash, audio_bits))
            return latent_refuse(err, YVEX_ERR_STATE, "latent plan schedule identity failed");
    }
    if (!yvex_sha256_final(&hash, digest))
        return latent_refuse(err, YVEX_ERR_STATE, "latent plan identity finalization failed");
    yvex_sha256_hex(digest, output);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int packed_layout_extents(
    const yvex_runtime_av_layout_request *request,
    unsigned long long capacities[5], unsigned long long *workspace_bytes,
    yvex_error *err)
{
    unsigned long long grid_height, grid_width, frame_rows, expected, values;
    unsigned long long channel;
    if (!request || request->schema_version != YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1 ||
        !request->text_rows || !request->audio_steps || !request->audio_channels ||
        !request->video_frames || !request->latent_height || !request->latent_width ||
        !request->patch_height || !request->patch_width ||
        request->latent_height % request->patch_height ||
        request->latent_width % request->patch_width ||
        !request->audio_width_indices || !request->temporal_pattern ||
        !request->temporal_pattern_count || !isfinite(request->temporal_scale) ||
        request->temporal_scale <= 0.0 || !isfinite(request->spatial_scale) ||
        request->spatial_scale <= 0.0 || !isfinite(request->media_time_origin) ||
        !request->plan_identity || !yvex_sha256_hex_valid(request->plan_identity))
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "a complete bounded packed audio-video layout is required");
    grid_height = request->latent_height / request->patch_height;
    grid_width = request->latent_width / request->patch_width;
    if (!yvex_core_u64_mul(grid_height, grid_width, &frame_rows) ||
        !yvex_core_u64_mul(request->video_frames, frame_rows, &capacities[2]) ||
        !yvex_core_u64_mul(request->audio_steps, request->audio_channels, &capacities[3]) ||
        !yvex_core_u64_add(request->text_rows, capacities[3], &expected) ||
        !yvex_core_u64_add(expected, capacities[2], &expected) ||
        request->text_start != 0ull || request->audio_start != request->text_rows ||
        request->video_start != request->text_rows + capacities[3] ||
        request->packed_rows != expected || expected > UINT_MAX)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "packed layout geometry does not form one exact row partition");
    for (channel = 0ull; channel < request->audio_channels; ++channel)
        if (request->audio_width_indices[channel] >= grid_width)
            return latent_refuse(err, YVEX_ERR_BOUNDS,
                                 "audio channel width coordinate exceeds the video grid");
    for (channel = 0ull; channel < request->temporal_pattern_count; ++channel)
        if (!request->temporal_pattern[channel])
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "video temporal pattern requires positive spans");
    capacities[0] = request->packed_rows * 3ull;
    capacities[1] = request->packed_rows;
    capacities[4] = request->text_rows;
    if (!yvex_core_u64_mul(request->packed_rows, 5ull, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), workspace_bytes) ||
        *workspace_bytes > request->maximum_workspace_bytes || *workspace_bytes > SIZE_MAX)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "packed layout exceeds its transactional workspace budget");
    return YVEX_OK;
}

static double packed_spatial_coordinate(
    unsigned long long dimension, unsigned long long patch, unsigned long long index,
    double square_root_area, double scale)
{
    double ratio = (double)dimension / square_root_area;
    double left = (1.0 - ratio) / 2.0;
    return (left + (double)index * ratio / (double)(dimension / patch)) * scale;
}

static void packed_layout_fill(
    const yvex_runtime_av_layout_request *request, float *positions,
    unsigned int *tags, unsigned int *video_indices, unsigned int *audio_indices,
    unsigned int *text_indices)
{
    unsigned long long text, channel, step, frame, row, height, width, video_row = 0ull;
    double square_root_area = sqrt((double)request->latent_height * (double)request->latent_width);
    double video_time = request->media_time_origin;
    for (text = 0ull; text < request->text_rows; ++text) {
        row = request->text_start + text;
        positions[row * 3ull] = (float)text;
        tags[row] = request->text_tag;
        text_indices[text] = (unsigned int)row;
    }
    for (channel = 0ull; channel < request->audio_channels; ++channel) {
        double audio_width = packed_spatial_coordinate(
            request->latent_width, request->patch_width,
            request->audio_width_indices[channel], square_root_area, request->spatial_scale);
        for (step = 0ull; step < request->audio_steps; ++step) {
            unsigned long long audio_row = channel * request->audio_steps + step;
            row = request->audio_start + audio_row;
            positions[row * 3ull] = (float)(request->media_time_origin + (double)step);
            positions[row * 3ull + 2ull] = (float)audio_width;
            tags[row] = request->audio_tag;
            audio_indices[audio_row] = (unsigned int)row;
        }
    }
    for (frame = 0ull; frame < request->video_frames; ++frame) {
        for (height = 0ull; height < request->latent_height / request->patch_height; ++height)
            for (width = 0ull; width < request->latent_width / request->patch_width; ++width) {
                row = request->video_start + video_row;
                positions[row * 3ull] = (float)video_time;
                positions[row * 3ull + 1ull] = (float)packed_spatial_coordinate(
                    request->latent_height, request->patch_height, height,
                    square_root_area, request->spatial_scale);
                positions[row * 3ull + 2ull] = (float)packed_spatial_coordinate(
                    request->latent_width, request->patch_width, width,
                    square_root_area, request->spatial_scale);
                tags[row] = request->video_tag;
                video_indices[video_row++] = (unsigned int)row;
            }
        video_time += request->temporal_scale *
                      (double)request->temporal_pattern[frame % request->temporal_pattern_count];
    }
}

static int packed_layout_identity(
    const yvex_runtime_av_layout_request *request, const float *positions,
    const unsigned int *tags, const unsigned int *video_indices,
    const unsigned int *audio_indices, const unsigned int *text_indices,
    unsigned long long video_rows, unsigned long long audio_rows,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.packed-av-layout.v1") ||
        !yvex_sha256_update_text(&hash, request->plan_identity) ||
        !yvex_sha256_update_u64(&hash, request->packed_rows)) return 0;
    for (index = 0ull; index < request->packed_rows * 3ull; ++index)
        if (!latent_hash_f32(&hash, positions[index])) return 0;
    for (index = 0ull; index < request->packed_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, tags[index])) return 0;
    for (index = 0ull; index < video_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, video_indices[index])) return 0;
    for (index = 0ull; index < audio_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, audio_indices[index])) return 0;
    for (index = 0ull; index < request->text_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, text_indices[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_runtime_av_layout_build(
    const yvex_runtime_av_layout_request *request,
    const yvex_runtime_av_layout_output *output,
    yvex_runtime_av_layout_result *result, yvex_error *err)
{
    yvex_runtime_av_layout_result staged = {0};
    unsigned long long capacities[5] = {0}, workspace_bytes = 0ull;
    float *storage = NULL, *positions;
    unsigned int *tags, *video, *audio, *text;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!output || !output->position_ids || !output->token_tags || !output->video_indices ||
        !output->audio_indices || !output->text_indices || !result)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "packed layout output storage and result are required");
    rc = packed_layout_extents(request, capacities, &workspace_bytes, err);
    if (rc != YVEX_OK) return rc;
    if (output->position_capacity < capacities[0] || output->tag_capacity < capacities[1] ||
        output->video_capacity < capacities[2] || output->audio_capacity < capacities[3] ||
        output->text_capacity < capacities[4])
        return latent_refuse(err, YVEX_ERR_BOUNDS, "packed layout output capacity is insufficient");
    storage = yvex_core_malloc((size_t)workspace_bytes);
    if (!storage) return latent_refuse(err, YVEX_ERR_NOMEM, "packed layout allocation failed");
    memset(storage, 0, (size_t)workspace_bytes);
    positions = storage;
    tags = (unsigned int *)(positions + capacities[0]);
    video = tags + capacities[1]; audio = video + capacities[2]; text = audio + capacities[3];
    packed_layout_fill(request, positions, tags, video, audio, text);
    if (!packed_layout_identity(request, positions, tags, video, audio, text,
                                capacities[2], capacities[3], staged.layout_identity))
        rc = latent_refuse(err, YVEX_ERR_STATE, "packed layout identity failed");
    if (rc == YVEX_OK) {
        memcpy(output->position_ids, positions, (size_t)(capacities[0] * sizeof(float)));
        memcpy(output->token_tags, tags, (size_t)(capacities[1] * sizeof(unsigned int)));
        memcpy(output->video_indices, video, (size_t)(capacities[2] * sizeof(unsigned int)));
        memcpy(output->audio_indices, audio, (size_t)(capacities[3] * sizeof(unsigned int)));
        memcpy(output->text_indices, text, (size_t)(capacities[4] * sizeof(unsigned int)));
        staged.schema_version = YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1;
        staged.text_rows = capacities[4]; staged.audio_rows = capacities[3];
        staged.video_rows = capacities[2]; staged.packed_rows = request->packed_rows;
        staged.workspace_bytes = workspace_bytes; staged.complete = 1; *result = staged;
        yvex_error_clear(err);
    }
    yvex_core_free(storage);
    return rc;
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
