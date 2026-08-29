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

static int packed_layout_identity(
    const yvex_runtime_av_layout_request *request, const float *positions,
    const unsigned int *tags, const unsigned int *video_indices,
    const unsigned int *audio_indices, const unsigned int *text_indices,
    unsigned long long video_rows, unsigned long long audio_rows,
    char output[YVEX_SHA256_HEX_CAP]);

static int latent_hash_f32(yvex_sha256 *hash, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

/* Latent initialization owns Box-Muller publication while sampling owns the versioned PCG stream. */
int yvex_runtime_latent_normal_f32_from_offset(
    float *values, unsigned long long value_capacity, unsigned long long value_count,
    unsigned long long discarded_value_count, unsigned long long seed,
    unsigned long long maximum_workspace_bytes,
    yvex_runtime_latent_normal_result *result, yvex_error *err)
{
    yvex_runtime_latent_normal_result staged = {0};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    uint64_t state, increment;
    unsigned long long bytes, generated = 0ull, published = 0ull, total;
    float *output;
    if (result) memset(result, 0, sizeof(*result));
    if (!values || !result || !value_count || value_count > value_capacity ||
        !yvex_core_u64_add(discarded_value_count, value_count, &total) ||
        !yvex_core_u64_mul(value_count, sizeof(*output), &bytes) || bytes > SIZE_MAX ||
        bytes > maximum_workspace_bytes)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "bounded normal output and workspace are required");
    output = yvex_core_malloc((size_t)bytes);
    if (!output)
        return latent_refuse(err, YVEX_ERR_NOMEM, "normal staging allocation failed");
    yvex_runtime_sampling_pcg_seed(seed, &state, &increment);
    while (generated < total) {
        const double scale = 1.0 / 4294967296.0;
        double first = ((double)yvex_runtime_sampling_pcg_next(&state, increment) + 0.5) * scale;
        double second = ((double)yvex_runtime_sampling_pcg_next(&state, increment) + 0.5) * scale;
        double magnitude = sqrt(-2.0 * log(first));
        double angle = 6.283185307179586476925286766559 * second;
        float pair[2] = {(float)(magnitude * cos(angle)), (float)(magnitude * sin(angle))};
        unsigned int lane;
        for (lane = 0u; lane < 2u && generated < total; ++lane, ++generated)
            if (generated >= discarded_value_count) output[published++] = pair[lane];
    }
    staged.schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    staged.rng_algorithm = YVEX_SAMPLING_RNG_PCG_XSH_RR_64_32;
    staged.rng_version = YVEX_SAMPLING_RNG_VERSION_V1;
    staged.seed = seed;
    staged.discarded_value_count = discarded_value_count;
    staged.value_count = value_count;
    staged.uniform_draw_count = 2ull * ((total + 1ull) / 2ull);
    staged.workspace_bytes = bytes;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, discarded_value_count
                                            ? "yvex.runtime.sampling.normal-f32-offset.v1"
                                            : "yvex.runtime.sampling.normal-f32.v1") ||
        !yvex_sha256_update_u64(&hash, staged.rng_algorithm) ||
        !yvex_sha256_update_u64(&hash, staged.rng_version) ||
        !yvex_sha256_update_u64(&hash, seed) ||
        (discarded_value_count &&
         !yvex_sha256_update_u64(&hash, discarded_value_count)) ||
        !yvex_sha256_update_u64(&hash, value_count))
        goto identity_failed;
    for (published = 0ull; published < value_count; ++published)
        if (!latent_hash_f32(&hash, output[published])) goto identity_failed;
    if (!yvex_sha256_final(&hash, digest)) goto identity_failed;
    yvex_sha256_hex(digest, staged.normal_identity);
    staged.completed = 1;
    memcpy(values, output, (size_t)bytes);
    *result = staged;
    yvex_core_free(output);
    yvex_error_clear(err);
    return YVEX_OK;
identity_failed:
    yvex_core_free(output);
    return latent_refuse(err, YVEX_ERR_STATE, "normal identity derivation failed");
}

int yvex_runtime_latent_normal_f32(
    float *values, unsigned long long value_capacity, unsigned long long value_count,
    unsigned long long seed, unsigned long long maximum_workspace_bytes,
    yvex_runtime_latent_normal_result *result, yvex_error *err)
{
    return yvex_runtime_latent_normal_f32_from_offset(
        values, value_capacity, value_count, 0ull, seed,
        maximum_workspace_bytes, result, err);
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

static int latent_observe(
    const yvex_runtime_latent_request *request,
    yvex_runtime_latent_observation_stage stage, unsigned long long completed_steps,
    const float *state, const float *velocity, float video_timestep,
    float audio_timestep, yvex_error *err)
{
    yvex_runtime_latent_observation observation;
    int rc;
    if (!request->observe) return YVEX_OK;
    memset(&observation, 0, sizeof(observation));
    observation.schema_version = YVEX_RUNTIME_LATENT_OBSERVATION_SCHEMA_V1;
    observation.stage = stage;
    observation.completed_steps = completed_steps;
    observation.video_values = request->video_values;
    observation.audio_values = request->audio_values;
    observation.video_timestep = video_timestep;
    observation.audio_timestep = audio_timestep;
    observation.video_state = state;
    observation.audio_state = state + request->video_values;
    if (velocity) {
        observation.video_velocity = velocity;
        observation.audio_velocity = velocity + request->video_values;
    }
    rc = request->observe(request->observer_context, &observation, err);
    if (rc == YVEX_OK) return YVEX_OK;
    if (!yvex_error_is_set(err))
        return latent_refuse(err, YVEX_ERR, "latent observation failed without an error");
    return yvex_error_code(err);
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
        !request->evaluator_identity || !yvex_sha256_hex_valid(request->evaluator_identity) ||
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
        !yvex_sha256_update_text(&hash, request->evaluator_identity) ||
        !yvex_sha256_update_u64(&hash, request->seed) ||
        (request->initialization_skip_values &&
         !yvex_sha256_update_u64(&hash, request->initialization_skip_values)) ||
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

static int latent_shifted_sigmas(
    float *output, unsigned int points, float shift, yvex_error *err)
{
    float step;
    unsigned int half, index;
    if (!output || points < 2u || points > 65u || !isfinite(shift) || shift <= 0.0f)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "a bounded positive shifted sigma grid is required");
    step = -1.0f / (float)(points - 1u);
    half = points / 2u;
    for (index = 0u; index < points; ++index) {
        /* The released scheduler builds this grid with torch.linspace on CPU. Its symmetric
           FMA construction is observably different from direct rational division at a few ULPs. */
        float base = index < half
                         ? fmaf((float)index, step, 1.0f)
                         : fmaf((float)(points - index - 1u), -step, 0.0f);
        output[index] = shift * base / (1.0f + (shift - 1.0f) * base);
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int latent_plan_identity(
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

int yvex_runtime_latent_binding_identity(
    const char *domain, const char *const *identities, unsigned long long identity_count,
    const unsigned long long *facts, unsigned long long fact_count,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    if (!domain || !domain[0] || !identities || !identity_count || identity_count > 16ull ||
        (!facts && fact_count) || fact_count > 16ull || !output)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "bounded latent binding identity facts are required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.latent.binding.v1") ||
        !yvex_sha256_update_text(&hash, domain) ||
        !yvex_sha256_update_u64_be(&hash, identity_count))
        return latent_refuse(err, YVEX_ERR_STATE, "latent binding identity initialization failed");
    for (index = 0ull; index < identity_count; ++index)
        if (!yvex_sha256_hex_valid(identities[index]) ||
            !yvex_sha256_update_text(&hash, identities[index]))
            return latent_refuse(err, YVEX_ERR_FORMAT, "latent binding identity is malformed");
    if (!yvex_sha256_update_u64_be(&hash, fact_count))
        return latent_refuse(err, YVEX_ERR_STATE, "latent binding fact count failed");
    for (index = 0ull; index < fact_count; ++index)
        if (!yvex_sha256_update_u64_be(&hash, facts[index]))
            return latent_refuse(err, YVEX_ERR_STATE, "latent binding fact identity failed");
    if (!yvex_sha256_final(&hash, digest))
        return latent_refuse(err, YVEX_ERR_STATE, "latent binding identity finalization failed");
    yvex_sha256_hex(digest, output);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_latent_evaluator_begin(
    yvex_runtime_latent_evaluator_evidence *evidence, const char *domain,
    const char *evaluator_identity, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP];
    if (!evidence || !domain || !domain[0] || !yvex_sha256_hex_valid(evaluator_identity))
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "latent evaluator evidence requires an exact identity domain");
    memcpy(identity, evaluator_identity, sizeof(identity));
    memset(evidence, 0, sizeof(*evidence));
    evidence->staged.schema_version = YVEX_RUNTIME_LATENT_EVALUATOR_SCHEMA_V1;
    memcpy(evidence->staged.evaluator_identity, identity, sizeof(identity));
    yvex_sha256_init(&evidence->chain);
    if (!yvex_sha256_update_text(&evidence->chain, "yvex.runtime.latent.evaluator-chain.v1") ||
        !yvex_sha256_update_text(&evidence->chain, domain) ||
        !yvex_sha256_update_text(&evidence->chain, evaluator_identity))
        return latent_refuse(err, YVEX_ERR_STATE, "latent evaluator evidence initialization failed");
    evidence->active = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_latent_evaluator_record(
    yvex_runtime_latent_evaluator_evidence *evidence, const char *residency_identity,
    const char *execution_identity, unsigned long long kernel_launches,
    unsigned long long h2d_bytes, unsigned long long d2h_bytes,
    unsigned long long device_bytes, yvex_error *err)
{
    unsigned long long evaluations, launches, host_to_device, device_to_host;
    if (!evidence || !evidence->active || !yvex_sha256_hex_valid(residency_identity) ||
        !yvex_sha256_hex_valid(execution_identity) ||
        (evidence->staged.model_evaluations &&
         strcmp(evidence->staged.residency_identity, residency_identity) != 0))
        return latent_refuse(err, YVEX_ERR_STATE,
                             "latent evaluator evidence changed resident identity");
    if (!yvex_core_u64_add(evidence->staged.model_evaluations, 1ull, &evaluations) ||
        !yvex_core_u64_add(evidence->staged.kernel_launches, kernel_launches, &launches) ||
        !yvex_core_u64_add(evidence->staged.h2d_bytes, h2d_bytes, &host_to_device) ||
        !yvex_core_u64_add(evidence->staged.d2h_bytes, d2h_bytes, &device_to_host) ||
        !yvex_sha256_update_text(&evidence->chain, execution_identity))
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "latent evaluator evidence accounting overflowed");
    if (!evidence->staged.model_evaluations)
        memcpy(evidence->staged.residency_identity, residency_identity, YVEX_SHA256_HEX_CAP);
    evidence->staged.model_evaluations = evaluations;
    evidence->staged.kernel_launches = launches;
    evidence->staged.h2d_bytes = host_to_device;
    evidence->staged.d2h_bytes = device_to_host;
    if (device_bytes > evidence->staged.peak_device_bytes)
        evidence->staged.peak_device_bytes = device_bytes;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_latent_evaluator_finish(
    yvex_runtime_latent_evaluator_evidence *evidence,
    unsigned long long expected_evaluations,
    yvex_runtime_latent_evaluator_result *result, yvex_error *err)
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (result) memset(result, 0, sizeof(*result));
    if (!evidence || !evidence->active || !expected_evaluations || !result ||
        evidence->staged.model_evaluations != expected_evaluations ||
        !yvex_sha256_final(&evidence->chain, digest))
        return latent_refuse(err, YVEX_ERR_STATE,
                             "complete latent evaluator evidence is required");
    yvex_sha256_hex(digest, evidence->staged.execution_chain_identity);
    evidence->staged.complete = 1;
    evidence->active = 0;
    *result = evidence->staged;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_av_plan_build(
    const yvex_runtime_av_plan_policy *policy, unsigned long long text_tokens,
    unsigned long long width, unsigned long long height, unsigned long long frames,
    unsigned int inference_steps, yvex_runtime_av_plan *out, yvex_error *err)
{
    unsigned long long blocks, spatial_rows, canvas_height, canvas_width, rounded_audio;
    unsigned long long facts[32], fact_count = 0ull, index, temporal_bits, spatial_bits;
    int rc;
    if (!policy || policy->schema_version != YVEX_RUNTIME_AV_PLAN_SCHEMA_V1 || !out ||
        !text_tokens || !policy->frame_period || !policy->frame_remainder ||
        !policy->video_latents_per_period || !policy->video_latent_remainder ||
        !policy->spatial_ratio || !policy->patch_height || !policy->patch_width ||
        !policy->audio_rate_numerator || !policy->audio_rate_denominator ||
        policy->audio_channels != 2ull || !policy->video_value_width ||
        !policy->audio_value_width || !policy->temporal_pattern_count ||
        policy->temporal_pattern_count > YVEX_RUNTIME_AV_TEMPORAL_PATTERN_CAP ||
        !policy->maximum_steps || policy->maximum_steps > 64u || !inference_steps ||
        inference_steps > policy->maximum_steps || !isfinite(policy->video_sigma_shift) ||
        policy->video_sigma_shift <= 0.0f || !isfinite(policy->audio_sigma_shift) ||
        policy->audio_sigma_shift <= 0.0f || !isfinite(policy->temporal_scale) ||
        policy->temporal_scale <= 0.0 || !isfinite(policy->spatial_scale) ||
        policy->spatial_scale <= 0.0 || !policy->identity_domain || !policy->target_identity ||
        !policy->source_revision || frames < policy->frame_remainder ||
        (frames - policy->frame_remainder) % policy->frame_period ||
        !yvex_core_u64_mul(policy->spatial_ratio, policy->patch_height, &canvas_height) ||
        !yvex_core_u64_mul(policy->spatial_ratio, policy->patch_width, &canvas_width) ||
        height < canvas_height || width < canvas_width || height % canvas_height ||
        width % canvas_width || policy->text_tag == policy->audio_tag ||
        policy->text_tag == policy->video_tag || policy->audio_tag == policy->video_tag)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "a complete bounded audio-video plan policy is required");
    for (index = 0ull; index < policy->temporal_pattern_count; ++index)
        if (!policy->temporal_pattern[index])
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "audio-video plan temporal spans must be positive");
    memset(out, 0, sizeof(*out));
    out->schema_version = YVEX_RUNTIME_AV_PLAN_SCHEMA_V1;
    out->text_tokens = text_tokens; out->frames = frames; out->width = width; out->height = height;
    blocks = (frames - policy->frame_remainder) / policy->frame_period;
    if (!yvex_core_u64_mul(blocks, policy->video_latents_per_period,
                           &out->video_latent_frames) ||
        !yvex_core_u64_add(out->video_latent_frames, policy->video_latent_remainder,
                           &out->video_latent_frames) ||
        !yvex_core_u64_mul(frames, policy->audio_rate_numerator, &rounded_audio) ||
        !yvex_core_u64_add(rounded_audio, policy->audio_rate_denominator / 2ull,
                           &rounded_audio))
        return latent_refuse(err, YVEX_ERR_BOUNDS, "audio-video temporal geometry overflowed");
    out->audio_latent_steps = rounded_audio / policy->audio_rate_denominator;
    out->video_latent_height = height / policy->spatial_ratio;
    out->video_latent_width = width / policy->spatial_ratio;
    if (!yvex_core_u64_mul(out->video_latent_height / policy->patch_height,
                           out->video_latent_width / policy->patch_width, &spatial_rows) ||
        !yvex_core_u64_mul(spatial_rows, out->video_latent_frames, &out->video_rows) ||
        !yvex_core_u64_mul(out->audio_latent_steps, policy->audio_channels, &out->audio_rows) ||
        !yvex_core_u64_add(text_tokens, out->audio_rows, &out->packed_rows) ||
        !yvex_core_u64_add(out->packed_rows, out->video_rows, &out->packed_rows) ||
        out->packed_rows > UINT_MAX)
        return latent_refuse(err, YVEX_ERR_BOUNDS, "audio-video packed geometry overflowed");
    out->patch_height = policy->patch_height; out->patch_width = policy->patch_width;
    out->audio_channels = policy->audio_channels;
    out->video_value_width = policy->video_value_width;
    out->audio_value_width = policy->audio_value_width;
    out->text_tag = policy->text_tag; out->audio_tag = policy->audio_tag;
    out->video_tag = policy->video_tag; out->temporal_pattern_count = policy->temporal_pattern_count;
    out->temporal_scale = policy->temporal_scale; out->spatial_scale = policy->spatial_scale;
    memcpy(out->temporal_pattern, policy->temporal_pattern,
           policy->temporal_pattern_count * sizeof(*out->temporal_pattern));
    out->model_evaluations = inference_steps; out->sigma_grid_points = inference_steps + 1u;
    rc = latent_shifted_sigmas(
        out->video_sigmas, out->sigma_grid_points, policy->video_sigma_shift, err);
    if (rc == YVEX_OK)
        rc = latent_shifted_sigmas(
            out->audio_sigmas, out->sigma_grid_points, policy->audio_sigma_shift, err);
    facts[fact_count++] = text_tokens; facts[fact_count++] = width;
    facts[fact_count++] = height; facts[fact_count++] = frames;
    facts[fact_count++] = out->sigma_grid_points; facts[fact_count++] = out->model_evaluations;
    facts[fact_count++] = policy->frame_period; facts[fact_count++] = policy->frame_remainder;
    facts[fact_count++] = policy->video_latents_per_period;
    facts[fact_count++] = policy->video_latent_remainder;
    facts[fact_count++] = policy->spatial_ratio; facts[fact_count++] = policy->patch_height;
    facts[fact_count++] = policy->patch_width; facts[fact_count++] = policy->audio_rate_numerator;
    facts[fact_count++] = policy->audio_rate_denominator;
    facts[fact_count++] = policy->audio_channels; facts[fact_count++] = policy->video_value_width;
    facts[fact_count++] = policy->audio_value_width; facts[fact_count++] = policy->text_tag;
    facts[fact_count++] = policy->audio_tag; facts[fact_count++] = policy->video_tag;
    facts[fact_count++] = policy->temporal_pattern_count;
    for (index = 0ull; index < policy->temporal_pattern_count; ++index)
        facts[fact_count++] = policy->temporal_pattern[index];
    memcpy(&temporal_bits, &policy->temporal_scale, sizeof(temporal_bits));
    memcpy(&spatial_bits, &policy->spatial_scale, sizeof(spatial_bits));
    facts[fact_count++] = temporal_bits; facts[fact_count++] = spatial_bits;
    if (rc == YVEX_OK)
        rc = latent_plan_identity(
            policy->identity_domain, policy->target_identity, policy->source_revision,
            facts, fact_count, out->video_sigmas, out->audio_sigmas,
            out->sigma_grid_points, out->identity, err);
    if (rc != YVEX_OK) return rc;
    out->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_av_plan_add_condition_rows(
    yvex_runtime_av_plan *plan, unsigned long long condition_rows, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long packed_rows;
    if (!plan || plan->schema_version != YVEX_RUNTIME_AV_PLAN_SCHEMA_V1 ||
        !plan->complete || plan->condition_rows || !condition_rows ||
        !yvex_sha256_hex_valid(plan->identity) ||
        !yvex_core_u64_add(plan->packed_rows, condition_rows, &packed_rows) ||
        packed_rows > UINT_MAX)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "a complete unconditioned AV plan and bounded condition rows are required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.av.conditioned-plan.v1") ||
        !yvex_sha256_update_text(&hash, plan->identity) ||
        !yvex_sha256_update_u64(&hash, condition_rows) ||
        !yvex_sha256_final(&hash, digest))
        return latent_refuse(err, YVEX_ERR_STATE,
                             "conditioned AV plan identity could not be sealed");
    plan->condition_rows = condition_rows;
    plan->packed_rows = packed_rows;
    yvex_sha256_hex(digest, plan->identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_av_scheduler_step(
    float *output, const float *sample, const float *velocity,
    unsigned long long values, float timestep, float sigma, float sigma_next,
    yvex_error *err)
{
    unsigned long long index;
    if (!output || !sample || !velocity || !values || !isfinite(timestep) ||
        !isfinite(sigma) || !isfinite(sigma_next) || timestep < 0.0f || timestep >= 1.0f ||
        sigma <= 0.0f || sigma_next < 0.0f || sigma_next >= sigma)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "a finite decreasing audio-video scheduler interval is required");
    for (index = 0ull; index < values; ++index)
        if (!isfinite(sample[index]) || !isfinite(velocity[index]))
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "audio-video scheduler input contains a non-finite value");
    for (index = 0ull; index < values; ++index) {
        float denoised = sample[index] + (1.0f - timestep) * velocity[index];
        float ratio = sigma_next / sigma;
        output[index] = ratio * sample[index] + (1.0f - ratio) * denoised;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_av_latent_execute(
    const yvex_runtime_av_plan *plan, const yvex_runtime_latent_request *template,
    float *video, unsigned long long video_capacity, float *audio,
    unsigned long long audio_capacity, yvex_runtime_latent_result *result, yvex_error *err)
{
    yvex_runtime_latent_request request;
    unsigned long long video_values, audio_values;
    if (!plan || plan->schema_version != YVEX_RUNTIME_AV_PLAN_SCHEMA_V1 || !plan->complete ||
        !template || !yvex_core_u64_mul(plan->video_rows, plan->video_value_width, &video_values) ||
        !yvex_core_u64_mul(plan->audio_rows, plan->audio_value_width, &audio_values))
        return latent_refuse(err, YVEX_ERR_BOUNDS, "a complete bounded audio-video plan is required");
    request = *template; request.schema_version = YVEX_RUNTIME_LATENT_SCHEMA_V1;
    request.video_values = video_values; request.audio_values = audio_values;
    request.step_count = plan->model_evaluations; request.video_sigmas = plan->video_sigmas;
    request.audio_sigmas = plan->audio_sigmas; request.plan_identity = plan->identity;
    request.advance = yvex_runtime_av_scheduler_step;
    return yvex_runtime_latent_execute(
        &request, video, video_capacity, audio, audio_capacity, result, err);
}

int yvex_runtime_av_layout_from_plan(
    const yvex_runtime_av_plan *plan, const yvex_runtime_av_layout_output *output,
    yvex_runtime_av_layout_result *result, yvex_error *err)
{
    unsigned long long audio_width_indices[2], workspace_bytes;
    yvex_runtime_av_layout_request request = {0};
    if (!plan || plan->schema_version != YVEX_RUNTIME_AV_PLAN_SCHEMA_V1 || !plan->complete ||
        plan->audio_channels != 2ull || plan->video_latent_width < plan->patch_width ||
        !yvex_core_u64_mul(plan->packed_rows, 5ull * sizeof(float), &workspace_bytes))
        return latent_refuse(err, YVEX_ERR_INVALID_ARG, "a complete patchable audio-video plan is required");
    audio_width_indices[0] = 0ull;
    audio_width_indices[1] = plan->video_latent_width / plan->patch_width - 1ull;
    request.schema_version = YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1;
    request.text_tag = plan->text_tag; request.audio_tag = plan->audio_tag;
    request.video_tag = plan->video_tag; request.text_rows = plan->text_tokens;
    request.audio_steps = plan->audio_latent_steps; request.audio_channels = plan->audio_channels;
    request.video_frames = plan->video_latent_frames;
    request.latent_height = plan->video_latent_height; request.latent_width = plan->video_latent_width;
    request.patch_height = plan->patch_height; request.patch_width = plan->patch_width;
    request.text_start = 0ull; request.condition_start = 0ull;
    request.audio_start = plan->text_tokens;
    request.video_start = plan->text_tokens + plan->audio_rows; request.packed_rows = plan->packed_rows;
    request.audio_width_indices = audio_width_indices; request.temporal_pattern = plan->temporal_pattern;
    request.temporal_pattern_count = plan->temporal_pattern_count;
    request.temporal_scale = plan->temporal_scale; request.spatial_scale = plan->spatial_scale;
    request.media_time_origin = (double)plan->text_tokens; request.plan_identity = plan->identity;
    request.maximum_workspace_bytes = workspace_bytes;
    return yvex_runtime_av_layout_build(&request, output, result, err);
}

int yvex_runtime_av_layout_from_conditioned_plan(
    const yvex_runtime_av_plan *plan, const unsigned int *text_tags,
    const double *condition_time_origins,
    unsigned long long condition_count, const yvex_runtime_av_layout_output *output,
    yvex_runtime_av_layout_result *result, yvex_error *err)
{
    unsigned long long audio_width_indices[2], frame_rows, workspace_bytes;
    yvex_runtime_av_layout_request request = {0};
    if (!plan || plan->schema_version != YVEX_RUNTIME_AV_PLAN_SCHEMA_V1 || !plan->complete ||
        !condition_time_origins || !condition_count || !plan->condition_rows ||
        plan->condition_rows % condition_count || plan->audio_channels != 2ull ||
        !yvex_core_u64_mul(plan->video_latent_height / plan->patch_height,
                           plan->video_latent_width / plan->patch_width, &frame_rows) ||
        plan->condition_rows / condition_count != frame_rows ||
        !yvex_core_u64_mul(plan->packed_rows, 5ull * sizeof(float), &workspace_bytes))
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "a complete condition-aware AV plan is required");
    audio_width_indices[0] = 0ull;
    audio_width_indices[1] = plan->video_latent_width / plan->patch_width - 1ull;
    request.schema_version = YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V2;
    request.text_tag = plan->text_tag; request.audio_tag = plan->audio_tag;
    request.video_tag = plan->video_tag; request.text_rows = plan->text_tokens;
    request.condition_rows = plan->condition_rows; request.condition_count = condition_count;
    request.condition_rows_per_image = frame_rows;
    request.text_tags = text_tags;
    request.condition_time_origins = condition_time_origins;
    request.audio_steps = plan->audio_latent_steps; request.audio_channels = plan->audio_channels;
    request.video_frames = plan->video_latent_frames;
    request.latent_height = plan->video_latent_height; request.latent_width = plan->video_latent_width;
    request.patch_height = plan->patch_height; request.patch_width = plan->patch_width;
    request.text_start = 0ull; request.condition_start = plan->text_tokens;
    request.audio_start = request.condition_start + plan->condition_rows;
    request.video_start = request.audio_start + plan->audio_rows;
    request.packed_rows = plan->packed_rows;
    request.audio_width_indices = audio_width_indices; request.temporal_pattern = plan->temporal_pattern;
    request.temporal_pattern_count = plan->temporal_pattern_count;
    request.temporal_scale = plan->temporal_scale; request.spatial_scale = plan->spatial_scale;
    request.media_time_origin = (double)plan->text_tokens; request.plan_identity = plan->identity;
    request.maximum_workspace_bytes = workspace_bytes;
    return yvex_runtime_av_layout_build(&request, output, result, err);
}

int yvex_runtime_av_layout_matches_plan(
    const yvex_runtime_av_plan *plan, const yvex_runtime_av_layout_output *output,
    const yvex_runtime_av_layout_result *result, yvex_error *err)
{
    unsigned long long position_values;
    yvex_runtime_av_layout_request request = {0};
    char observed_identity[YVEX_SHA256_HEX_CAP];
    if (!plan || plan->schema_version != YVEX_RUNTIME_AV_PLAN_SCHEMA_V1 || !plan->complete ||
        !output || !result ||
        (result->schema_version != YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1 &&
         result->schema_version != YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V2) ||
        !result->complete || !yvex_sha256_hex_valid(result->layout_identity) ||
        result->packed_rows != plan->packed_rows ||
        result->condition_rows != plan->condition_rows ||
        result->video_rows != plan->condition_rows + plan->video_rows ||
        result->audio_rows != plan->audio_rows || result->text_rows != plan->text_tokens ||
        !output->position_ids || !output->token_tags || !output->video_indices ||
        !output->audio_indices || !output->text_indices ||
        !yvex_core_u64_mul(plan->packed_rows, 3ull, &position_values) ||
        output->position_capacity < position_values || output->tag_capacity < plan->packed_rows ||
        output->video_capacity < plan->condition_rows + plan->video_rows ||
        output->audio_capacity < plan->audio_rows ||
        output->text_capacity < plan->text_tokens)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "layout output does not cover the admitted audio-video plan");
    request.plan_identity = plan->identity;
    request.packed_rows = plan->packed_rows;
    request.text_rows = plan->text_tokens;
    if (!packed_layout_identity(
            &request, output->position_ids, output->token_tags, output->video_indices,
            output->audio_indices, output->text_indices,
            plan->condition_rows + plan->video_rows,
            plan->audio_rows, observed_identity) ||
        strcmp(observed_identity, result->layout_identity) != 0)
        return latent_refuse(err, YVEX_ERR_FORMAT,
                             "layout values do not match their admitted identity");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int av_unpack_identity(
    const yvex_runtime_av_unpack_request *request,
    const yvex_runtime_av_unpack_result *result, const float *video,
    const float *audio, char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.av.unpack.v1") ||
        !yvex_sha256_update_text(&hash, request->plan->identity) ||
        !yvex_sha256_update_text(&hash, request->latent_execution_identity) ||
        !yvex_sha256_update_u64(&hash, result->video_channels) ||
        !yvex_sha256_update_u64(&hash, result->video_frames) ||
        !yvex_sha256_update_u64(&hash, result->video_height) ||
        !yvex_sha256_update_u64(&hash, result->video_width) ||
        !yvex_sha256_update_u64(&hash, result->audio_batch) ||
        !yvex_sha256_update_u64(&hash, result->audio_channels) ||
        !yvex_sha256_update_u64(&hash, result->audio_steps))
        return 0;
    for (index = 0ull; index < request->video_channel_count; ++index)
        if (!latent_hash_f32(&hash, request->video_channel_mean[index]) ||
            !latent_hash_f32(&hash, request->video_channel_std[index]))
            return 0;
    for (index = 0ull; index < request->audio_channel_count; ++index)
        if (!latent_hash_f32(&hash, request->audio_channel_mean[index]) ||
            !latent_hash_f32(&hash, request->audio_channel_std[index]))
            return 0;
    for (index = 0ull; index < result->video_values; ++index)
        if (!latent_hash_f32(&hash, video[index])) return 0;
    for (index = 0ull; index < result->audio_values; ++index)
        if (!latent_hash_f32(&hash, audio[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int av_unpack_extents(
    const yvex_runtime_av_unpack_request *request,
    yvex_runtime_av_unpack_result *staged, unsigned long long *workspace_bytes,
    yvex_error *err)
{
    const yvex_runtime_av_plan *plan = request ? request->plan : NULL;
    unsigned long long patch_values, video_rows, audio_rows, values;
    if (!request || request->schema_version != YVEX_RUNTIME_AV_UNPACK_SCHEMA_V1 ||
        !plan || plan->schema_version != YVEX_RUNTIME_AV_PLAN_SCHEMA_V1 || !plan->complete ||
        !request->video_rows || !request->audio_rows || !request->video_channel_mean ||
        !request->video_channel_std || !request->audio_channel_mean ||
        !request->audio_channel_std || !request->latent_execution_identity ||
        !yvex_sha256_hex_valid(request->latent_execution_identity) ||
        !yvex_core_u64_mul(plan->patch_height, plan->patch_width, &patch_values) ||
        !patch_values || plan->video_value_width % patch_values ||
        request->video_channel_count != plan->video_value_width / patch_values ||
        request->audio_channel_count != plan->audio_value_width ||
        !yvex_core_u64_mul(plan->video_rows, plan->video_value_width, &video_rows) ||
        !yvex_core_u64_mul(plan->audio_rows, plan->audio_value_width, &audio_rows) ||
        request->video_row_capacity < video_rows || request->audio_row_capacity < audio_rows)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "exact packed audio-video rows and channel facts are required");
    staged->video_channels = request->video_channel_count;
    staged->video_frames = plan->video_latent_frames;
    staged->video_height = plan->video_latent_height;
    staged->video_width = plan->video_latent_width;
    staged->video_values = video_rows;
    staged->audio_batch = plan->audio_channels;
    staged->audio_channels = request->audio_channel_count;
    staged->audio_steps = plan->audio_latent_steps;
    staged->audio_values = audio_rows;
    if (!yvex_core_u64_add(video_rows, audio_rows, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), workspace_bytes) ||
        *workspace_bytes > request->maximum_workspace_bytes || *workspace_bytes > SIZE_MAX)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "audio-video unpack workspace exceeded its bound");
    return YVEX_OK;
}

int yvex_runtime_av_unpack(
    const yvex_runtime_av_unpack_request *request,
    const yvex_runtime_av_unpack_output *output,
    yvex_runtime_av_unpack_result *result, yvex_error *err)
{
    yvex_runtime_av_unpack_result staged = {0};
    const yvex_runtime_av_plan *plan = request ? request->plan : NULL;
    unsigned long long workspace_bytes = 0ull, frame, tile_y, tile_x, channel, y, x, step;
    unsigned long long patch_values, grid_height, grid_width, batch, source, destination;
    float *storage = NULL, *video, *audio;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!output || !output->video || !output->audio || !result)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "audio-video component input outputs are required");
    rc = av_unpack_extents(request, &staged, &workspace_bytes, err);
    if (rc != YVEX_OK) return rc;
    if (output->video_capacity < staged.video_values ||
        output->audio_capacity < staged.audio_values)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "audio-video component input capacity is insufficient");
    for (channel = 0ull; channel < staged.video_channels; ++channel)
        if (!isfinite(request->video_channel_mean[channel]) ||
            !isfinite(request->video_channel_std[channel]) ||
            request->video_channel_std[channel] <= 0.0f)
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "video latent normalization is invalid");
    for (channel = 0ull; channel < staged.audio_channels; ++channel)
        if (!isfinite(request->audio_channel_mean[channel]) ||
            !isfinite(request->audio_channel_std[channel]) ||
            request->audio_channel_std[channel] <= 0.0f)
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "audio latent normalization is invalid");
    storage = yvex_core_malloc((size_t)workspace_bytes);
    if (!storage) return latent_refuse(err, YVEX_ERR_NOMEM,
                                       "audio-video unpack allocation failed");
    video = storage;
    audio = video + staged.video_values;
    patch_values = plan->patch_height * plan->patch_width;
    grid_height = plan->video_latent_height / plan->patch_height;
    grid_width = plan->video_latent_width / plan->patch_width;
    for (frame = 0ull; frame < plan->video_latent_frames; ++frame)
        for (tile_y = 0ull; tile_y < grid_height; ++tile_y)
            for (tile_x = 0ull; tile_x < grid_width; ++tile_x)
                for (channel = 0ull; channel < staged.video_channels; ++channel)
                    for (y = 0ull; y < plan->patch_height; ++y)
                        for (x = 0ull; x < plan->patch_width; ++x) {
                            source = (((frame * grid_height + tile_y) * grid_width + tile_x) *
                                      plan->video_value_width) + channel * patch_values +
                                     y * plan->patch_width + x;
                            destination = ((((channel * plan->video_latent_frames + frame) *
                                             plan->video_latent_height +
                                             tile_y * plan->patch_height + y) *
                                            plan->video_latent_width) +
                                           tile_x * plan->patch_width + x);
                            video[destination] = request->video_rows[source] *
                                request->video_channel_std[channel] +
                                request->video_channel_mean[channel];
                        }
    for (batch = 0ull; batch < plan->audio_channels; ++batch)
        for (step = 0ull; step < plan->audio_latent_steps; ++step)
            for (channel = 0ull; channel < staged.audio_channels; ++channel) {
                source = (batch * plan->audio_latent_steps + step) *
                         plan->audio_value_width + channel;
                destination = (batch * staged.audio_channels + channel) *
                              plan->audio_latent_steps + step;
                audio[destination] = request->audio_rows[source] *
                    request->audio_channel_std[channel] +
                    request->audio_channel_mean[channel];
            }
    for (source = 0ull; source < staged.video_values + staged.audio_values; ++source)
        if (!isfinite(storage[source])) {
            rc = latent_refuse(err, YVEX_ERR_FORMAT,
                               "audio-video component input became non-finite");
            break;
        }
    staged.schema_version = YVEX_RUNTIME_AV_UNPACK_SCHEMA_V1;
    staged.peak_workspace_bytes = workspace_bytes;
    if (rc == YVEX_OK &&
        !av_unpack_identity(request, &staged, video, audio, staged.input_identity))
        rc = latent_refuse(err, YVEX_ERR_STATE,
                           "audio-video component input identity failed");
    if (rc == YVEX_OK) {
        memcpy(output->video, video, (size_t)(staged.video_values * sizeof(float)));
        memcpy(output->audio, audio, (size_t)(staged.audio_values * sizeof(float)));
        staged.complete = 1;
        *result = staged;
        yvex_error_clear(err);
    }
    yvex_core_free(storage);
    return rc;
}

static int av_video_split_tiles(
    unsigned long long length, unsigned long long tile_size,
    unsigned long long minimum_overlap, unsigned long long spatial_ratio,
    unsigned long long *count, unsigned long long starts[YVEX_RUNTIME_AV_TILE_CAP],
    unsigned long long lengths[YVEX_RUNTIME_AV_TILE_CAP],
    unsigned long long overlaps[YVEX_RUNTIME_AV_TILE_CAP - 1u], yvex_error *err)
{
    unsigned long long tiles, covered, remaining, index, rounded_length;
    if (tile_size >= length) {
        *count = 1ull; starts[0] = 0ull; lengths[0] = length;
        return YVEX_OK;
    }
    if (!yvex_core_u64_add(length, tile_size - 1ull, &rounded_length))
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "video reconstruction tile extent overflowed");
    tiles = rounded_length / tile_size;
    for (;;) {
        if (tiles > YVEX_RUNTIME_AV_TILE_CAP ||
            tiles - 1ull > ULLONG_MAX / minimum_overlap ||
            tiles > ULLONG_MAX / tile_size)
            return latent_refuse(err, YVEX_ERR_BOUNDS,
                                 "video reconstruction tile count exceeded its bound");
        covered = tile_size * tiles - minimum_overlap * (tiles - 1ull);
        if (covered >= length) break;
        tiles++;
    }
    remaining = covered - length;
    if (remaining % spatial_ratio)
        return latent_refuse(err, YVEX_ERR_FORMAT,
                             "video reconstruction tile slack is not latent-aligned");
    memset(overlaps, 0, (YVEX_RUNTIME_AV_TILE_CAP - 1u) * sizeof(*overlaps));
    for (index = 0ull; index + 1ull < tiles; ++index) overlaps[index] = minimum_overlap;
    for (index = 0ull; index < remaining / spatial_ratio; ++index)
        overlaps[index % (tiles - 1ull)] += spatial_ratio;
    starts[0] = 0ull;
    for (index = 0ull; index < tiles; ++index) {
        lengths[index] = tile_size;
        if (index) starts[index] = starts[index - 1ull] + tile_size - overlaps[index - 1ull];
    }
    *count = tiles;
    return YVEX_OK;
}

static int av_video_reconstruction_identity(
    const yvex_runtime_av_video_reconstruction_request *request,
    const yvex_runtime_av_video_reconstruction_plan *plan,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const unsigned long long facts[] = {
        plan->frames, plan->width, plan->height, plan->latent_frames,
        plan->latent_height, plan->latent_width, request->temporal_ratio,
        request->clip_length, request->token_drop, request->spatial_ratio,
        request->tile_size, request->minimum_tile_overlap, plan->tokens_per_chunk,
        plan->token_overlap, plan->frame_pre_padding, plan->frame_overlap,
        plan->temporal_chunks, plan->decode_latent_frames, plan->decode_frames,
        plan->pad_tokens, plan->tile_y_count, plan->tile_x_count,
        plan->total_decode_calls,
    };
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.av.video-reconstruction.v1") ||
        !yvex_sha256_update_text(&hash, request->source_identity)) return 0;
    for (index = 0ull; index < sizeof(facts) / sizeof(facts[0]); ++index)
        if (!yvex_sha256_update_u64_be(&hash, facts[index])) return 0;
    for (index = 0ull; index < plan->tile_y_count; ++index)
        if (!yvex_sha256_update_u64_be(&hash, plan->tile_y_start[index]) ||
            !yvex_sha256_update_u64_be(&hash, plan->tile_y_length[index]) ||
            (index + 1ull < plan->tile_y_count &&
             !yvex_sha256_update_u64_be(&hash, plan->tile_y_overlap[index]))) return 0;
    for (index = 0ull; index < plan->tile_x_count; ++index)
        if (!yvex_sha256_update_u64_be(&hash, plan->tile_x_start[index]) ||
            !yvex_sha256_update_u64_be(&hash, plan->tile_x_length[index]) ||
            (index + 1ull < plan->tile_x_count &&
             !yvex_sha256_update_u64_be(&hash, plan->tile_x_overlap[index]))) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_runtime_av_video_reconstruction_plan_build(
    const yvex_runtime_av_video_reconstruction_request *request,
    yvex_runtime_av_video_reconstruction_plan *plan, yvex_error *err)
{
    yvex_runtime_av_video_reconstruction_plan staged = {0};
    unsigned long long blocks, expected_latents, num_tokens, padded_tokens, calls;
    unsigned long long rounded_clip;
    int rc;
    if (plan) memset(plan, 0, sizeof(*plan));
    if (!request || request->schema_version != YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1 ||
        !plan || !request->frames || !request->width || !request->height ||
        !request->latent_frames || !request->latent_height || !request->latent_width ||
        !request->temporal_ratio || !request->clip_length || !request->token_drop ||
        !request->spatial_ratio || !request->tile_size || !request->minimum_tile_overlap ||
        request->tile_size <= request->minimum_tile_overlap ||
        request->width % request->spatial_ratio || request->height % request->spatial_ratio ||
        request->latent_width != request->width / request->spatial_ratio ||
        request->latent_height != request->height / request->spatial_ratio ||
        request->tile_size % request->spatial_ratio ||
        request->minimum_tile_overlap % request->spatial_ratio ||
        !request->source_identity || !yvex_sha256_hex_valid(request->source_identity))
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "exact bounded video reconstruction facts are required");
    if (!yvex_core_u64_add(request->clip_length, request->temporal_ratio - 1ull,
                           &rounded_clip))
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "video reconstruction temporal extent overflowed");
    staged.tokens_per_chunk = rounded_clip / request->temporal_ratio;
    if (request->token_drop >= staged.tokens_per_chunk ||
        request->frames < staged.tokens_per_chunk ||
        (request->frames - staged.tokens_per_chunk) % request->clip_length)
        return latent_refuse(err, YVEX_ERR_FORMAT,
                             "video frames do not match the admitted temporal chunk geometry");
    blocks = (request->frames - staged.tokens_per_chunk) / request->clip_length;
    if (!yvex_core_u64_mul(blocks, staged.tokens_per_chunk, &expected_latents) ||
        !yvex_core_u64_add(expected_latents,
                           staged.tokens_per_chunk - request->token_drop, &expected_latents) ||
        expected_latents != request->latent_frames ||
        !yvex_core_u64_add(request->latent_frames, request->token_drop, &num_tokens))
        return latent_refuse(err, YVEX_ERR_FORMAT,
                             "video latent frames do not match the admitted pixel frames");
    staged.token_overlap =
        (staged.tokens_per_chunk - request->token_drop % staged.tokens_per_chunk) %
        staged.tokens_per_chunk;
    staged.frame_pre_padding =
        (request->temporal_ratio - request->clip_length % request->temporal_ratio) %
        request->temporal_ratio;
    staged.frame_overlap = staged.token_overlap * request->temporal_ratio;
    staged.frame_overlap = staged.frame_overlap > staged.frame_pre_padding
                               ? staged.frame_overlap - staged.frame_pre_padding : 0ull;
    staged.pad_tokens = (staged.tokens_per_chunk - num_tokens % staged.tokens_per_chunk) %
                        staged.tokens_per_chunk;
    if (!yvex_core_u64_add(num_tokens, staged.pad_tokens, &padded_tokens) ||
        padded_tokens / staged.tokens_per_chunk <= 1ull)
        return latent_refuse(err, YVEX_ERR_FORMAT,
                             "video reconstruction requires at least one complete temporal chunk");
    staged.temporal_chunks = padded_tokens / staged.tokens_per_chunk - 1ull;
    staged.decode_latent_frames = staged.tokens_per_chunk + staged.token_overlap;
    if (!yvex_core_u64_mul(staged.decode_latent_frames, request->temporal_ratio,
                           &staged.decode_frames))
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "video reconstruction decode extent overflowed");
    rc = av_video_split_tiles(
        request->height, request->tile_size, request->minimum_tile_overlap,
        request->spatial_ratio, &staged.tile_y_count, staged.tile_y_start,
        staged.tile_y_length, staged.tile_y_overlap, err);
    if (rc == YVEX_OK)
        rc = av_video_split_tiles(
            request->width, request->tile_size, request->minimum_tile_overlap,
            request->spatial_ratio, &staged.tile_x_count, staged.tile_x_start,
            staged.tile_x_length, staged.tile_x_overlap, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_mul(staged.temporal_chunks, staged.tile_y_count, &calls) ||
        !yvex_core_u64_mul(calls, staged.tile_x_count, &staged.total_decode_calls))
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "video reconstruction call count overflowed");
    staged.schema_version = YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1;
    staged.frames = request->frames; staged.width = request->width;
    staged.height = request->height; staged.latent_frames = request->latent_frames;
    staged.latent_height = request->latent_height; staged.latent_width = request->latent_width;
    staged.spatial_ratio = request->spatial_ratio;
    if (!av_video_reconstruction_identity(request, &staged, staged.identity))
        return latent_refuse(err, YVEX_ERR_STATE,
                             "video reconstruction identity could not be sealed");
    staged.complete = 1; *plan = staged; yvex_error_clear(err);
    return YVEX_OK;
}

typedef struct {
    unsigned long long output_values, tile_values, blended_values;
    unsigned long long chunk_values, latent_tile_values, overlap_values;
    unsigned long long tile_count, maximum_tile_height, maximum_tile_width;
    unsigned long long workspace_values, workspace_bytes;
    unsigned long long temporal_ratio, main_frames, overlap_start, overlap_frames;
} av_video_reconstruction_extents;

static int av_video_reconstruction_extents_build(
    const yvex_runtime_av_video_reconstruction_execution *execution,
    unsigned long long output_capacity, av_video_reconstruction_extents *extents,
    yvex_error *err)
{
    const yvex_runtime_av_video_reconstruction_plan *plan = execution ? execution->plan : NULL;
    unsigned long long index, latent_values, tile_frame_values, values, overlap_end;
    memset(extents, 0, sizeof(*extents));
    if (!execution || execution->schema_version != YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1 ||
        !plan || plan->schema_version != YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1 ||
        !plan->complete || !yvex_sha256_hex_valid(plan->identity) || !execution->latent ||
        !execution->latent_channels || !execution->decode || !plan->frames || !plan->width ||
        !plan->height || !plan->latent_frames || !plan->latent_height || !plan->latent_width ||
        !plan->spatial_ratio || plan->height % plan->latent_height ||
        plan->width % plan->latent_width ||
        plan->height / plan->latent_height != plan->spatial_ratio ||
        plan->width / plan->latent_width != plan->spatial_ratio ||
        !plan->temporal_chunks || !plan->decode_latent_frames || !plan->decode_frames ||
        !plan->tile_y_count || !plan->tile_x_count || plan->tile_y_count > YVEX_RUNTIME_AV_TILE_CAP ||
        plan->tile_x_count > YVEX_RUNTIME_AV_TILE_CAP ||
        (!!execution->output_channel_mean != !!execution->output_channel_std) ||
        (execution->output_channel_mean && execution->output_channel_count != 3ull) ||
        plan->decode_frames % plan->decode_latent_frames)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "a complete bounded video reconstruction execution is required");
    extents->temporal_ratio = plan->decode_frames / plan->decode_latent_frames;
    if (!extents->temporal_ratio ||
        !yvex_core_u64_mul(plan->tokens_per_chunk, extents->temporal_ratio,
                           &extents->overlap_start) ||
        !yvex_core_u64_add(extents->overlap_start, plan->frame_pre_padding,
                           &overlap_end) ||
        overlap_end > plan->decode_frames || plan->frame_pre_padding >= extents->overlap_start) {
        return latent_refuse(err, YVEX_ERR_FORMAT,
                             "video reconstruction temporal splits are inconsistent");
    }
    extents->main_frames = extents->overlap_start - plan->frame_pre_padding;
    extents->overlap_start = overlap_end;
    extents->overlap_frames = plan->decode_frames - overlap_end;
    if (extents->overlap_frames != plan->frame_overlap ||
        !yvex_core_u64_mul(execution->latent_channels, plan->latent_frames, &latent_values) ||
        !yvex_core_u64_mul(latent_values, plan->latent_height, &latent_values) ||
        !yvex_core_u64_mul(latent_values, plan->latent_width, &latent_values) ||
        latent_values != execution->latent_capacity ||
        !yvex_core_u64_mul(3ull, plan->frames, &extents->output_values) ||
        !yvex_core_u64_mul(extents->output_values, plan->height, &extents->output_values) ||
        !yvex_core_u64_mul(extents->output_values, plan->width, &extents->output_values) ||
        output_capacity < extents->output_values)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "video reconstruction input or output extent is inconsistent");
    for (index = 0ull; index < plan->tile_y_count; ++index) {
        if (!plan->tile_y_length[index] || plan->tile_y_length[index] % plan->spatial_ratio ||
            plan->tile_y_start[index] % plan->spatial_ratio ||
            plan->tile_y_start[index] > plan->height ||
            plan->tile_y_length[index] > plan->height - plan->tile_y_start[index])
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "video reconstruction height tiles are invalid");
        if (plan->tile_y_length[index] > extents->maximum_tile_height)
            extents->maximum_tile_height = plan->tile_y_length[index];
    }
    for (index = 0ull; index < plan->tile_x_count; ++index) {
        if (!plan->tile_x_length[index] || plan->tile_x_length[index] % plan->spatial_ratio ||
            plan->tile_x_start[index] % plan->spatial_ratio ||
            plan->tile_x_start[index] > plan->width ||
            plan->tile_x_length[index] > plan->width - plan->tile_x_start[index])
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "video reconstruction width tiles are invalid");
        if (plan->tile_x_length[index] > extents->maximum_tile_width)
            extents->maximum_tile_width = plan->tile_x_length[index];
    }
    if (!yvex_core_u64_mul(plan->tile_y_count, plan->tile_x_count, &extents->tile_count) ||
        !yvex_core_u64_mul(3ull, plan->decode_frames, &tile_frame_values) ||
        !yvex_core_u64_mul(tile_frame_values, extents->maximum_tile_height, &values) ||
        !yvex_core_u64_mul(values, extents->maximum_tile_width, &extents->blended_values) ||
        !yvex_core_u64_mul(extents->blended_values, extents->tile_count,
                           &extents->tile_values) ||
        !yvex_core_u64_mul(tile_frame_values, plan->height, &extents->chunk_values) ||
        !yvex_core_u64_mul(extents->chunk_values, plan->width, &extents->chunk_values) ||
        !yvex_core_u64_mul(execution->latent_channels, plan->decode_latent_frames,
                           &extents->latent_tile_values) ||
        !yvex_core_u64_mul(extents->latent_tile_values,
                           extents->maximum_tile_height / plan->spatial_ratio,
                           &extents->latent_tile_values) ||
        !yvex_core_u64_mul(extents->latent_tile_values,
                           extents->maximum_tile_width / plan->spatial_ratio,
                           &extents->latent_tile_values) ||
        !yvex_core_u64_mul(3ull, extents->overlap_frames, &extents->overlap_values) ||
        !yvex_core_u64_mul(extents->overlap_values, plan->height, &extents->overlap_values) ||
        !yvex_core_u64_mul(extents->overlap_values, plan->width, &extents->overlap_values) ||
        !yvex_core_u64_add(extents->output_values, extents->tile_values,
                           &extents->workspace_values) ||
        !yvex_core_u64_add(extents->workspace_values, extents->blended_values,
                           &extents->workspace_values) ||
        !yvex_core_u64_add(extents->workspace_values, extents->chunk_values,
                           &extents->workspace_values) ||
        !yvex_core_u64_add(extents->workspace_values, extents->latent_tile_values,
                           &extents->workspace_values) ||
        !yvex_core_u64_add(extents->workspace_values, extents->overlap_values,
                           &extents->workspace_values) ||
        !yvex_core_u64_mul(extents->workspace_values, sizeof(float),
                           &extents->workspace_bytes) ||
        extents->workspace_bytes > execution->maximum_workspace_bytes ||
        extents->workspace_bytes > SIZE_MAX)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "video reconstruction exceeds its transactional workspace budget");
    if (execution->output_channel_mean)
        for (index = 0ull; index < execution->output_channel_count; ++index)
            if (!isfinite(execution->output_channel_mean[index]) ||
                !isfinite(execution->output_channel_std[index]) ||
                execution->output_channel_std[index] <= 0.0f)
                return latent_refuse(err, YVEX_ERR_FORMAT,
                                     "video output normalization facts are invalid");
    return YVEX_OK;
}

static void av_video_latent_tile_copy(
    const yvex_runtime_av_video_reconstruction_execution *execution,
    const av_video_reconstruction_extents *extents, unsigned long long chunk,
    unsigned long long tile_y, unsigned long long tile_x, float *destination)
{
    const yvex_runtime_av_video_reconstruction_plan *plan = execution->plan;
    unsigned long long latent_y = plan->tile_y_start[tile_y] / plan->spatial_ratio;
    unsigned long long latent_x = plan->tile_x_start[tile_x] / plan->spatial_ratio;
    unsigned long long height = plan->tile_y_length[tile_y] / plan->spatial_ratio;
    unsigned long long width = plan->tile_x_length[tile_x] / plan->spatial_ratio;
    unsigned long long channel, frame, row, column;
    for (channel = 0ull; channel < execution->latent_channels; ++channel)
        for (frame = 0ull; frame < plan->decode_latent_frames; ++frame) {
            unsigned long long source_frame = chunk * plan->tokens_per_chunk + frame;
            if (source_frame >= plan->latent_frames) source_frame = plan->latent_frames - 1ull;
            for (row = 0ull; row < height; ++row)
                for (column = 0ull; column < width; ++column) {
                    unsigned long long source =
                        ((channel * plan->latent_frames + source_frame) * plan->latent_height +
                         latent_y + row) * plan->latent_width + latent_x + column;
                    unsigned long long target =
                        ((channel * plan->decode_latent_frames + frame) * height + row) *
                        width + column;
                    destination[target] = execution->latent[source];
                }
        }
    (void)extents;
}

static void av_video_blend_axis(
    float *destination, const float *prior, unsigned long long channels,
    unsigned long long frames, unsigned long long height, unsigned long long width,
    unsigned long long prior_height, unsigned long long prior_width,
    unsigned long long extent, int vertical)
{
    unsigned long long channel, frame, row, column;
    if (!extent) return;
    for (channel = 0ull; channel < channels; ++channel)
        for (frame = 0ull; frame < frames; ++frame)
            for (row = 0ull; row < height; ++row)
                for (column = 0ull; column < width; ++column) {
                    unsigned long long position = vertical ? row : column;
                    unsigned long long prior_row = vertical ? prior_height - extent + row : row;
                    unsigned long long prior_column = vertical ? column : prior_width - extent + column;
                    unsigned long long destination_index, prior_index;
                    float weight;
                    if (position >= extent) continue;
                    destination_index = ((channel * frames + frame) * height + row) * width + column;
                    prior_index = ((channel * frames + frame) * prior_height + prior_row) *
                                  prior_width + prior_column;
                    weight = (float)position / (float)extent;
                    destination[destination_index] =
                        prior[prior_index] * (1.0f - weight) +
                        destination[destination_index] * weight;
                }
}

static void av_video_tile_publish(
    const yvex_runtime_av_video_reconstruction_plan *plan,
    const av_video_reconstruction_extents *extents, const float *tiles,
    float *blended, float *chunk_output, unsigned long long tile_y,
    unsigned long long tile_x)
{
    unsigned long long tile_index = tile_y * plan->tile_x_count + tile_x;
    unsigned long long height = plan->tile_y_length[tile_y];
    unsigned long long width = plan->tile_x_length[tile_x];
    unsigned long long kept_height = height;
    unsigned long long kept_width = width;
    unsigned long long values = 3ull * plan->decode_frames * height * width;
    unsigned long long channel, frame, row, column;
    const float *current = tiles + tile_index * extents->blended_values;
    memcpy(blended, current, (size_t)(values * sizeof(float)));
    if (tile_y) {
        const float *prior = tiles + ((tile_y - 1ull) * plan->tile_x_count + tile_x) *
                                    extents->blended_values;
        av_video_blend_axis(blended, prior, 3ull, plan->decode_frames, height, width,
                            plan->tile_y_length[tile_y - 1ull], width,
                            plan->tile_y_overlap[tile_y - 1ull], 1);
    }
    if (tile_x) {
        const float *prior = tiles + (tile_index - 1ull) * extents->blended_values;
        av_video_blend_axis(blended, prior, 3ull, plan->decode_frames, height, width,
                            height, plan->tile_x_length[tile_x - 1ull],
                            plan->tile_x_overlap[tile_x - 1ull], 0);
    }
    if (tile_y + 1ull < plan->tile_y_count) kept_height -= plan->tile_y_overlap[tile_y];
    if (tile_x + 1ull < plan->tile_x_count) kept_width -= plan->tile_x_overlap[tile_x];
    for (channel = 0ull; channel < 3ull; ++channel)
        for (frame = 0ull; frame < plan->decode_frames; ++frame)
            for (row = 0ull; row < kept_height; ++row)
                for (column = 0ull; column < kept_width; ++column) {
                    unsigned long long source =
                        ((channel * plan->decode_frames + frame) * height + row) * width + column;
                    unsigned long long target =
                        ((channel * plan->decode_frames + frame) * plan->height +
                         plan->tile_y_start[tile_y] + row) * plan->width +
                        plan->tile_x_start[tile_x] + column;
                    chunk_output[target] = blended[source];
                }
}

static void av_video_temporal_blend(
    float *chunk, const float *overlap,
    const yvex_runtime_av_video_reconstruction_plan *plan,
    const av_video_reconstruction_extents *extents)
{
    unsigned long long channel, frame, pixel;
    unsigned long long pixels = plan->height * plan->width;
    for (channel = 0ull; channel < 3ull; ++channel)
        for (frame = 0ull; frame < extents->overlap_frames; ++frame) {
            float weight = (float)frame / (float)extents->overlap_frames;
            for (pixel = 0ull; pixel < pixels; ++pixel) {
                unsigned long long target =
                    (channel * plan->decode_frames + plan->frame_pre_padding + frame) * pixels + pixel;
                unsigned long long source =
                    (channel * extents->overlap_frames + frame) * pixels + pixel;
                chunk[target] = overlap[source] * (1.0f - weight) + chunk[target] * weight;
            }
        }
}

static void av_video_temporal_copy(
    float *destination, unsigned long long *write_frame, const float *source,
    unsigned long long source_total_frames, unsigned long long source_start,
    unsigned long long source_frames,
    const yvex_runtime_av_video_reconstruction_plan *plan)
{
    unsigned long long channel, frame, pixel, pixels = plan->height * plan->width;
    unsigned long long copy_frames = source_frames;
    if (*write_frame >= plan->frames) copy_frames = 0ull;
    else if (copy_frames > plan->frames - *write_frame) copy_frames = plan->frames - *write_frame;
    for (channel = 0ull; channel < 3ull; ++channel)
        for (frame = 0ull; frame < copy_frames; ++frame)
            for (pixel = 0ull; pixel < pixels; ++pixel)
                destination[(channel * plan->frames + *write_frame + frame) * pixels + pixel] =
                    source[(channel * source_total_frames + source_start + frame) * pixels + pixel];
    *write_frame += copy_frames;
}

int yvex_runtime_av_video_reconstruct(
    const yvex_runtime_av_video_reconstruction_execution *execution,
    float *output, unsigned long long output_capacity,
    yvex_runtime_av_video_reconstruction_result *result, yvex_error *err)
{
    const yvex_runtime_av_video_reconstruction_plan *plan = execution ? execution->plan : NULL;
    av_video_reconstruction_extents extents;
    yvex_runtime_av_video_reconstruction_result staged = {0};
    yvex_runtime_av_video_decode_window window = {0};
    yvex_runtime_av_video_decode_evidence evidence;
    yvex_sha256 identity;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    float *workspace = NULL, *published, *tiles, *blended, *chunk_output, *latent_tile, *overlap;
    unsigned long long chunk, tile_y, tile_x, index, write_frame = 0ull;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!output || !result)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "video reconstruction output and result are required");
    rc = av_video_reconstruction_extents_build(execution, output_capacity, &extents, err);
    if (rc != YVEX_OK) return rc;
    workspace = yvex_core_malloc((size_t)extents.workspace_bytes);
    if (!workspace) return latent_refuse(err, YVEX_ERR_NOMEM,
                                         "video reconstruction workspace allocation failed");
    memset(workspace, 0, (size_t)extents.workspace_bytes);
    published = workspace;
    tiles = published + extents.output_values;
    blended = tiles + extents.tile_values;
    chunk_output = blended + extents.blended_values;
    latent_tile = chunk_output + extents.chunk_values;
    overlap = latent_tile + extents.latent_tile_values;
    yvex_sha256_init(&identity);
    if (!yvex_sha256_update_text(&identity, "yvex.runtime.av.video-reconstruction.execution.v1") ||
        !yvex_sha256_update_text(&identity, plan->identity) ||
        !yvex_sha256_update_u64_be(&identity, execution->latent_channels) ||
        !yvex_sha256_update_u64_be(&identity, execution->latent_capacity) ||
        !yvex_sha256_update_u64_be(&identity, execution->output_channel_count))
        rc = latent_refuse(err, YVEX_ERR_STATE,
                           "video reconstruction execution identity initialization failed");
    for (index = 0ull; rc == YVEX_OK && index < execution->latent_capacity; ++index)
        if (!isfinite(execution->latent[index]) || !latent_hash_f32(&identity, execution->latent[index]))
            rc = latent_refuse(err, YVEX_ERR_FORMAT,
                               "video reconstruction latent contains a non-finite value");
    for (chunk = 0ull; rc == YVEX_OK && chunk < plan->temporal_chunks; ++chunk) {
        memset(chunk_output, 0, (size_t)(extents.chunk_values * sizeof(float)));
        for (tile_y = 0ull; rc == YVEX_OK && tile_y < plan->tile_y_count; ++tile_y)
            for (tile_x = 0ull; rc == YVEX_OK && tile_x < plan->tile_x_count; ++tile_x) {
                unsigned long long tile_index = tile_y * plan->tile_x_count + tile_x;
                unsigned long long latent_height =
                    plan->tile_y_length[tile_y] / plan->spatial_ratio;
                unsigned long long latent_width =
                    plan->tile_x_length[tile_x] / plan->spatial_ratio;
                unsigned long long output_values = 3ull * plan->decode_frames *
                                                    plan->tile_y_length[tile_y] *
                                                    plan->tile_x_length[tile_x];
                float *tile_output = tiles + tile_index * extents.blended_values;
                if (execution->cancel_requested && execution->cancel_requested(execution->cancel_context)) {
                    rc = latent_refuse(err, YVEX_ERR_CANCELLED,
                                       "video reconstruction was cancelled between decode calls");
                    break;
                }
                av_video_latent_tile_copy(execution, &extents, chunk, tile_y, tile_x, latent_tile);
                memset(&evidence, 0, sizeof(evidence));
                window.latent = latent_tile; window.latent_channels = execution->latent_channels;
                window.latent_frames = plan->decode_latent_frames;
                window.latent_height = latent_height; window.latent_width = latent_width;
                window.output = tile_output; window.output_capacity = output_values;
                rc = execution->decode(execution->decode_context, &window, &evidence, err);
                if (rc == YVEX_OK && (!evidence.complete || evidence.output_values != output_values ||
                    !yvex_sha256_hex_valid(evidence.execution_identity)))
                    rc = latent_refuse(err, YVEX_ERR_STATE,
                                       "video decoder did not publish exact execution evidence");
                for (index = 0ull; rc == YVEX_OK && index < output_values; ++index)
                    if (!isfinite(tile_output[index]))
                        rc = latent_refuse(err, YVEX_ERR_FORMAT,
                                           "video decoder published a non-finite value");
                if (rc == YVEX_OK &&
                    (!yvex_sha256_update_text(&identity, evidence.execution_identity) ||
                     !yvex_core_u64_add(staged.kernel_launches, evidence.kernel_launches,
                                        &staged.kernel_launches) ||
                     !yvex_core_u64_add(staged.h2d_bytes, evidence.h2d_bytes,
                                        &staged.h2d_bytes) ||
                     !yvex_core_u64_add(staged.d2h_bytes, evidence.d2h_bytes,
                                        &staged.d2h_bytes)))
                    rc = latent_refuse(err, YVEX_ERR_BOUNDS,
                                       "video reconstruction evidence accounting overflowed");
                if (evidence.device_bytes > staged.peak_device_bytes)
                    staged.peak_device_bytes = evidence.device_bytes;
                staged.decode_calls++;
            }
        for (tile_y = 0ull; rc == YVEX_OK && tile_y < plan->tile_y_count; ++tile_y)
            for (tile_x = 0ull; tile_x < plan->tile_x_count; ++tile_x)
                av_video_tile_publish(plan, &extents, tiles, blended, chunk_output, tile_y, tile_x);
        if (rc == YVEX_OK && chunk) av_video_temporal_blend(chunk_output, overlap, plan, &extents);
        if (rc == YVEX_OK)
            av_video_temporal_copy(published, &write_frame, chunk_output,
                                   plan->decode_frames, plan->frame_pre_padding,
                                   extents.main_frames, plan);
        if (rc == YVEX_OK) {
            unsigned long long channel, frame, pixel, pixels = plan->height * plan->width;
            for (channel = 0ull; channel < 3ull; ++channel)
                for (frame = 0ull; frame < extents.overlap_frames; ++frame)
                    for (pixel = 0ull; pixel < pixels; ++pixel)
                        overlap[(channel * extents.overlap_frames + frame) * pixels + pixel] =
                            chunk_output[(channel * plan->decode_frames + extents.overlap_start + frame) *
                                         pixels + pixel];
        }
    }
    if (rc == YVEX_OK)
        av_video_temporal_copy(published, &write_frame, overlap,
                               extents.overlap_frames, 0ull, extents.overlap_frames, plan);
    if (rc == YVEX_OK && (write_frame != plan->frames || staged.decode_calls != plan->total_decode_calls))
        rc = latent_refuse(err, YVEX_ERR_STATE,
                           "video reconstruction did not reconcile its frame or call plan");
    if (rc == YVEX_OK && execution->output_channel_mean) {
        unsigned long long channel, values_per_channel = plan->frames * plan->height * plan->width;
        for (channel = 0ull; channel < execution->output_channel_count; ++channel) {
            float mean = execution->output_channel_mean[channel];
            float std = execution->output_channel_std[channel];
            if (!latent_hash_f32(&identity, mean) || !latent_hash_f32(&identity, std)) {
                rc = latent_refuse(err, YVEX_ERR_STATE,
                                   "video output normalization identity failed");
                break;
            }
            for (index = 0ull; index < values_per_channel; ++index) {
                float value = published[channel * values_per_channel + index] * std + mean;
                published[channel * values_per_channel + index] =
                    value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
            }
        }
    }
    for (index = 0ull; rc == YVEX_OK && index < extents.output_values; ++index)
        if (!latent_hash_f32(&identity, published[index]))
            rc = latent_refuse(err, YVEX_ERR_STATE,
                               "video reconstruction output identity failed");
    if (rc == YVEX_OK && execution->cancel_requested &&
        execution->cancel_requested(execution->cancel_context))
        rc = latent_refuse(err, YVEX_ERR_CANCELLED,
                           "video reconstruction publication was cancelled");
    if (rc == YVEX_OK && !yvex_sha256_final(&identity, digest))
        rc = latent_refuse(err, YVEX_ERR_STATE,
                           "video reconstruction execution identity finalization failed");
    if (rc == YVEX_OK) {
        yvex_sha256_hex(digest, staged.execution_identity);
        memcpy(output, published, (size_t)(extents.output_values * sizeof(float)));
        staged.schema_version = YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1;
        staged.output_values = extents.output_values;
        staged.peak_workspace_bytes = extents.workspace_bytes;
        staged.complete = 1; *result = staged; yvex_error_clear(err);
    }
    yvex_core_free(workspace);
    return rc;
}

static int packed_layout_extents(
    const yvex_runtime_av_layout_request *request,
    unsigned long long capacities[5], unsigned long long *workspace_bytes,
    yvex_error *err)
{
    unsigned long long grid_height, grid_width, frame_rows, condition_rows, expected, values;
    unsigned long long channel;
    if (!request ||
        (request->schema_version != YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1 &&
         request->schema_version != YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V2) ||
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
        !yvex_core_u64_add(capacities[2], request->condition_rows, &capacities[2]) ||
        !yvex_core_u64_mul(request->audio_steps, request->audio_channels, &capacities[3]) ||
        !yvex_core_u64_add(request->text_rows, request->condition_rows, &expected) ||
        !yvex_core_u64_add(expected, capacities[3], &expected) ||
        !yvex_core_u64_add(expected, capacities[2] - request->condition_rows, &expected) ||
        request->text_start != 0ull ||
        (request->schema_version == YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1 &&
         (request->condition_start || request->audio_start != request->text_rows)) ||
        (request->schema_version == YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V2 &&
         (request->condition_start != request->text_rows ||
          request->audio_start != request->text_rows + request->condition_rows)) ||
        request->video_start != request->audio_start + capacities[3] ||
        request->packed_rows != expected || expected > UINT_MAX)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "packed layout geometry does not form one exact row partition");
    if ((request->schema_version == YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1 &&
         (request->condition_rows || request->condition_count ||
          request->condition_rows_per_image || request->condition_time_origins)) ||
        (request->schema_version == YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V2 &&
         (!request->condition_rows || !request->condition_count ||
          !request->condition_rows_per_image || !request->text_tags ||
          !request->condition_time_origins ||
          !yvex_core_u64_mul(request->condition_count,
                             request->condition_rows_per_image, &condition_rows) ||
          request->condition_rows != condition_rows ||
          request->condition_rows_per_image != frame_rows)))
        return latent_refuse(err, YVEX_ERR_FORMAT,
                             "packed condition rows do not form complete image grids");
    for (channel = 0ull; channel < request->condition_count; ++channel)
        if (!isfinite(request->condition_time_origins[channel]))
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "packed condition anchors must have finite time coordinates");
    for (channel = 0ull; channel < request->text_rows; ++channel)
        if (request->text_tags && request->text_tags[channel] != request->text_tag &&
            request->text_tags[channel] != request->video_tag)
            return latent_refuse(err, YVEX_ERR_FORMAT,
                                 "packed text rows carry an unsupported modality tag");
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
    unsigned long long text, condition, channel, step, frame, row, height, width;
    unsigned long long video_row = 0ull, target_video_row = 0ull;
    double square_root_area = sqrt((double)request->latent_height * (double)request->latent_width);
    double video_time = request->media_time_origin;
    for (text = 0ull; text < request->text_rows; ++text) {
        row = request->text_start + text;
        positions[row * 3ull] = (float)text;
        tags[row] = request->text_tags ? request->text_tags[text] : request->text_tag;
        text_indices[text] = (unsigned int)row;
    }
    for (condition = 0ull; condition < request->condition_count; ++condition)
        for (height = 0ull; height < request->latent_height / request->patch_height; ++height)
            for (width = 0ull; width < request->latent_width / request->patch_width; ++width) {
                row = request->condition_start + video_row;
                positions[row * 3ull] = (float)request->condition_time_origins[condition];
                positions[row * 3ull + 1ull] = (float)packed_spatial_coordinate(
                    request->latent_height, request->patch_height, height,
                    square_root_area, request->spatial_scale);
                positions[row * 3ull + 2ull] = (float)packed_spatial_coordinate(
                    request->latent_width, request->patch_width, width,
                    square_root_area, request->spatial_scale);
                tags[row] = request->video_tag;
                video_indices[video_row++] = (unsigned int)row;
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
                row = request->video_start + target_video_row++;
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
        staged.schema_version = request->schema_version;
        staged.text_rows = capacities[4]; staged.audio_rows = capacities[3];
        staged.condition_rows = request->condition_rows;
        staged.video_rows = capacities[2]; staged.packed_rows = request->packed_rows;
        staged.workspace_bytes = workspace_bytes; staged.complete = 1; *result = staged;
        yvex_error_clear(err);
    }
    yvex_core_free(storage);
    return rc;
}

typedef struct {
    yvex_runtime_latent_result *destination, *staged;
    float *video_output, *audio_output;
    const float *state;
    unsigned long long video_values, audio_values;
} latent_transaction_publication;

typedef struct {
    const yvex_runtime_latent_request *request;
    yvex_runtime_latent_result *staged;
    float *state, *next, *velocity;
} latent_transaction_execution;

static int latent_transaction_publish(void *opaque, yvex_error *err)
{
    latent_transaction_publication *publication = opaque;
    if (!publication || !publication->destination || !publication->staged ||
        !publication->video_output || !publication->audio_output || !publication->state)
        return latent_refuse(err, YVEX_ERR_STATE,
                             "latent transaction publication state is incomplete");
    memcpy(publication->video_output, publication->state,
           (size_t)(publication->video_values * sizeof(float)));
    memcpy(publication->audio_output, publication->state + publication->video_values,
           (size_t)(publication->audio_values * sizeof(float)));
    publication->staged->completed = 1;
    *publication->destination = *publication->staged;
    yvex_error_clear(err);
    return YVEX_OK;
}

static void latent_transaction_discard(void *opaque)
{
    latent_transaction_publication *publication = opaque;
    if (publication && publication->destination)
        memset(publication->destination, 0, sizeof(*publication->destination));
}

static int latent_transaction_execute_quantum(
    void *opaque, unsigned long long step, yvex_error *err)
{
    latent_transaction_execution *execution = opaque;
    const yvex_runtime_latent_request *request;
    float video_timestep, audio_timestep, *swap;
    int rc;
    if (!execution || !execution->request || !execution->staged ||
        !execution->state || !execution->next || !execution->velocity)
        return latent_refuse(err, YVEX_ERR_STATE,
                             "latent transaction execution state is incomplete");
    request = execution->request;
    if (step >= request->step_count)
        return latent_refuse(err, YVEX_ERR_BOUNDS,
                             "latent transaction quantum exceeds its family schedule");
    video_timestep = 1.0f - request->video_sigmas[step];
    audio_timestep = 1.0f - request->audio_sigmas[step];
    rc = request->evaluate(
        request->execution_context, execution->state, request->video_values,
        execution->state + request->video_values, request->audio_values,
        video_timestep, audio_timestep, execution->velocity,
        execution->velocity + request->video_values, err);
    if (rc == YVEX_OK)
        rc = latent_observe(request, YVEX_RUNTIME_LATENT_OBSERVATION_EVALUATED, step,
                            execution->state, execution->velocity,
                            video_timestep, audio_timestep, err);
    if (rc == YVEX_OK)
        rc = request->advance(
            execution->next, execution->state, execution->velocity,
            request->video_values, video_timestep,
            request->video_sigmas[step], request->video_sigmas[step + 1ull], err);
    if (rc == YVEX_OK)
        rc = request->advance(
            execution->next + request->video_values,
            execution->state + request->video_values,
            execution->velocity + request->video_values,
            request->audio_values, audio_timestep,
            request->audio_sigmas[step], request->audio_sigmas[step + 1ull], err);
    if (rc != YVEX_OK) return rc;
    swap = execution->state;
    execution->state = execution->next;
    execution->next = swap;
    execution->staged->completed_steps++;
    return latent_observe(
        request, YVEX_RUNTIME_LATENT_OBSERVATION_ADVANCED,
        execution->staged->completed_steps, execution->state, NULL,
        video_timestep, audio_timestep, err);
}

int yvex_runtime_latent_execute(
    const yvex_runtime_latent_request *request,
    float *video_output, unsigned long long video_capacity,
    float *audio_output, unsigned long long audio_capacity,
    yvex_runtime_latent_result *result, yvex_error *err)
{
    yvex_runtime_latent_result staged = {0};
    yvex_runtime_execution_transaction *transaction = NULL;
    yvex_execution_transaction_options transaction_options = {0};
    yvex_execution_safe_point_action safe_point = 0;
    yvex_execution_transaction_summary transaction_summary = {0};
    latent_transaction_publication publication = {0};
    latent_transaction_execution execution = {0};
    float *storage = NULL;
    unsigned long long total = 0ull, bytes = 0ull, storage_bytes, step;
    yvex_error primary, cleanup;
    int rc, cleanup_rc;
    if (result) memset(result, 0, sizeof(*result));
    if (request && request->transaction_summary)
        memset(request->transaction_summary, 0, sizeof(*request->transaction_summary));
    if (!video_output || !audio_output || !result)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "latent outputs and result storage are required");
    rc = latent_request_validate(request, video_capacity, audio_capacity, &total, &bytes, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_mul(bytes, 3ull, &storage_bytes) || storage_bytes > SIZE_MAX)
        return latent_refuse(err, YVEX_ERR_BOUNDS, "latent workspace extent overflowed");
    storage = yvex_core_malloc((size_t)storage_bytes);
    if (!storage)
        return latent_refuse(err, YVEX_ERR_NOMEM, "latent workspace allocation failed");
    execution = (latent_transaction_execution){
        request, &staged, storage, storage + total, storage + total * 2ull};
    publication = (latent_transaction_publication){
        result, &staged, video_output, audio_output, NULL,
        request->video_values, request->audio_values};
    transaction_options.request_identity = request->plan_identity;
    transaction_options.quantum_count = request->step_count;
    transaction_options.resource = request->execution_resource;
    transaction_options.execute = latent_transaction_execute_quantum;
    transaction_options.execution_context = &execution;
    transaction_options.cancel_requested = request->cancel_requested;
    transaction_options.cancel_context = request->cancel_context;
    transaction_options.publish = latent_transaction_publish;
    transaction_options.discard = latent_transaction_discard;
    transaction_options.publication_context = &publication;
    rc = yvex_runtime_execution_transaction_open(&transaction, &transaction_options, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_latent_normal_f32_from_offset(
            execution.state, total, total, request->initialization_skip_values, request->seed,
            bytes, &staged.initialization, err);
    if (rc == YVEX_OK &&
        !latent_state_identity("yvex.runtime.latent.initial.v1", execution.state, total,
                               staged.initial_state_identity))
        rc = latent_refuse(err, YVEX_ERR_STATE, "initial latent identity failed");
    if (rc == YVEX_OK)
        rc = latent_observe(request, YVEX_RUNTIME_LATENT_OBSERVATION_INITIAL, 0ull,
                            execution.state, NULL, 0.0f, 0.0f, err);
    for (step = 0ull; rc == YVEX_OK && step < request->step_count; ++step) {
        rc = yvex_runtime_execution_transaction_execute_quantum(
            transaction, &safe_point, err);
        if (rc == YVEX_OK && safe_point == YVEX_EXECUTION_SAFE_POINT_CANCEL)
            rc = latent_refuse(err, YVEX_ERR_CANCELLED,
                               "latent iteration was cancelled at a safe point");
        if (rc == YVEX_OK && safe_point == YVEX_EXECUTION_SAFE_POINT_YIELD)
            rc = yvex_runtime_execution_transaction_resume(transaction, err);
        if (rc == YVEX_OK &&
            ((step + 1ull == request->step_count &&
              safe_point != YVEX_EXECUTION_SAFE_POINT_COMMIT) ||
             (step + 1ull < request->step_count &&
              safe_point != YVEX_EXECUTION_SAFE_POINT_CONTINUE)))
            rc = latent_refuse(err, YVEX_ERR_STATE,
                               "latent safe-point action does not match execution progress");
    }
    if (rc == YVEX_OK &&
        !latent_state_identity("yvex.runtime.latent.final.v1", execution.state, total,
                               staged.final_state_identity))
        rc = latent_refuse(err, YVEX_ERR_STATE, "final latent identity failed");
    if (rc == YVEX_OK)
        rc = latent_observe(request, YVEX_RUNTIME_LATENT_OBSERVATION_FINAL,
                            staged.completed_steps, execution.state, NULL, 0.0f, 0.0f, err);
    staged.schema_version = YVEX_RUNTIME_LATENT_SCHEMA_V1;
    staged.video_values = request->video_values;
    staged.audio_values = request->audio_values;
    staged.model_evaluations = staged.completed_steps;
    staged.peak_workspace_bytes = bytes * 4ull;
    if (rc == YVEX_OK && !latent_execution_identity(request, &staged, staged.execution_identity))
        rc = latent_refuse(err, YVEX_ERR_STATE, "latent execution identity failed");
    publication.state = execution.state;
    if (rc == YVEX_OK)
        rc = yvex_runtime_execution_transaction_commit(transaction, err);
    if (rc != YVEX_OK && transaction) {
        primary = err ? *err : (yvex_error){0};
        yvex_error_clear(&cleanup);
        (void)yvex_runtime_execution_transaction_abort(transaction, &cleanup);
        if (err) *err = primary;
    }
    if (transaction &&
        yvex_runtime_execution_transaction_summary_copy(
            transaction, &transaction_summary, &cleanup) == YVEX_OK) {
        if (request->transaction_summary)
            *request->transaction_summary = transaction_summary;
        if (rc == YVEX_OK) result->transaction = transaction_summary;
    }
    if (rc != YVEX_OK) primary = err ? *err : (yvex_error){0};
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_execution_transaction_close(&transaction, &cleanup);
    if (rc != YVEX_OK) {
        if (err) *err = primary;
    } else if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    } else {
        yvex_error_clear(err);
    }
    yvex_core_free(storage);
    return rc;
}
