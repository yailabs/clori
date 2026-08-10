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
    request.text_start = 0ull; request.audio_start = plan->text_tokens;
    request.video_start = plan->text_tokens + plan->audio_rows; request.packed_rows = plan->packed_rows;
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
        !output || !result || result->schema_version != YVEX_RUNTIME_AV_LAYOUT_SCHEMA_V1 ||
        !result->complete || !yvex_sha256_hex_valid(result->layout_identity) ||
        result->packed_rows != plan->packed_rows || result->video_rows != plan->video_rows ||
        result->audio_rows != plan->audio_rows || result->text_rows != plan->text_tokens ||
        !output->position_ids || !output->token_tags || !output->video_indices ||
        !output->audio_indices || !output->text_indices ||
        !yvex_core_u64_mul(plan->packed_rows, 3ull, &position_values) ||
        output->position_capacity < position_values || output->tag_capacity < plan->packed_rows ||
        output->video_capacity < plan->video_rows || output->audio_capacity < plan->audio_rows ||
        output->text_capacity < plan->text_tokens)
        return latent_refuse(err, YVEX_ERR_INVALID_ARG,
                             "layout output does not cover the admitted audio-video plan");
    request.plan_identity = plan->identity;
    request.packed_rows = plan->packed_rows;
    request.text_rows = plan->text_tokens;
    if (!packed_layout_identity(
            &request, output->position_ids, output->token_tags, output->video_indices,
            output->audio_indices, output->text_indices, plan->video_rows,
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
