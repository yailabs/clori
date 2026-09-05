/* Validate and authenticate family-neutral stateful decoder semantics. */
#include <yvex/internal/semantic_decoder.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <yvex/internal/core.h>

static int gated_delta_requirement_valid(
    const yvex_gated_delta_requirement *requirement)
{
    return requirement &&
           requirement->schema_version ==
               YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V2 &&
           requirement->query_heads && requirement->key_heads &&
           requirement->value_heads &&
           requirement->value_heads % requirement->query_heads == 0ull &&
           requirement->value_heads % requirement->key_heads == 0ull &&
           requirement->key_head_dimension && requirement->value_head_dimension &&
           requirement->convolution_kernel >= 2ull &&
           requirement->projected_dtype == YVEX_DTYPE_F32 &&
           requirement->convolution_state_dtype == YVEX_DTYPE_F32 &&
           requirement->recurrent_state_dtype == YVEX_DTYPE_F32 &&
           requirement->accumulation_dtype == YVEX_DTYPE_F32 &&
           requirement->output_dtype == YVEX_DTYPE_F32 &&
           requirement->numeric_contract ==
               YVEX_SEQUENCE_MIXER_NUMERIC_F32_RECURRENCE &&
           (requirement->output_normalization_weight_convention ==
                YVEX_NORMALIZATION_WEIGHT_DIRECT ||
            requirement->output_normalization_weight_convention ==
                YVEX_NORMALIZATION_WEIGHT_ONE_PLUS) &&
           isfinite(requirement->qk_normalization_epsilon) &&
           requirement->qk_normalization_epsilon > 0.0 &&
           isfinite(requirement->output_normalization_epsilon) &&
           requirement->output_normalization_epsilon > 0.0 &&
           isfinite(requirement->query_scale) &&
           requirement->query_scale > 0.0 && requirement->deterministic == 1;
}

int yvex_semantic_gated_delta_requirement_identity(
    const yvex_gated_delta_requirement *requirement,
    char identity[YVEX_SEMANTIC_DECODER_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long qk_epsilon_bits, output_epsilon_bits, scale_bits;

    if (identity) identity[0] = '\0';
    if (!identity || !gated_delta_requirement_valid(requirement)) return 0;
    memcpy(&qk_epsilon_bits, &requirement->qk_normalization_epsilon,
           sizeof(qk_epsilon_bits));
    memcpy(&output_epsilon_bits, &requirement->output_normalization_epsilon,
           sizeof(output_epsilon_bits));
    memcpy(&scale_bits, &requirement->query_scale, sizeof(scale_bits));
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.sequence-mixer.gated-delta.v1") ||
        !yvex_sha256_update_u64(&hash, requirement->query_heads) ||
        !yvex_sha256_update_u64(&hash, requirement->key_heads) ||
        !yvex_sha256_update_u64(&hash, requirement->value_heads) ||
        !yvex_sha256_update_u64(&hash, requirement->key_head_dimension) ||
        !yvex_sha256_update_u64(&hash, requirement->value_head_dimension) ||
        !yvex_sha256_update_u64(&hash, requirement->convolution_kernel) ||
        !yvex_sha256_update_u64(&hash, requirement->projected_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->convolution_state_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->recurrent_state_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->accumulation_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->output_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->numeric_contract) ||
        !yvex_sha256_update_u64(
            &hash, requirement->output_normalization_weight_convention) ||
        !yvex_sha256_update_u64(&hash, qk_epsilon_bits) ||
        !yvex_sha256_update_u64(&hash, output_epsilon_bits) ||
        !yvex_sha256_update_u64(&hash, scale_bits) ||
        !yvex_sha256_update_u64(
            &hash, (unsigned long long)requirement->deterministic) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, identity);
    return 1;
}

static int ssd_semantic_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "model.selective-ssd", reason);
    return status;
}

static int ssd_geometry(yvex_selective_ssd_geometry *p)
{
    const yvex_selective_ssd_requirement *r = &p->requirement;
    unsigned long long grouped, twice, bytes;

    return yvex_core_u64_mul(r->heads, r->head_dimension, &p->width) &&
        yvex_core_u64_mul(r->groups, r->state_dimension, &grouped) &&
        yvex_core_u64_mul(grouped, 2u, &twice) &&
        yvex_core_u64_add(p->width, twice, &p->convolution_width) &&
        yvex_core_u64_add(p->width, p->convolution_width, &p->projection_width) &&
        yvex_core_u64_add(p->projection_width, r->heads, &p->projection_width) &&
        yvex_core_u64_mul(p->convolution_width, r->convolution_kernel,
                          &p->convolution_state_values) &&
        yvex_core_u64_mul(p->width, r->state_dimension, &p->recurrent_state_values) &&
        yvex_core_u64_add(p->convolution_state_values, p->recurrent_state_values, &twice) &&
        yvex_core_u64_mul(twice, sizeof(float), &bytes) && bytes <= SIZE_MAX;
}

int yvex_selective_ssd_geometry_seal(
    yvex_selective_ssd_geometry *plan, const yvex_selective_ssd_requirement *r,
    yvex_error *err)
{
    char description[512];
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    int length;
    yvex_selective_ssd_requirement saved;

    if (!plan || !r || r->schema_version != YVEX_SELECTIVE_SSD_SCHEMA_V1 ||
        !r->heads || !r->head_dimension || !r->state_dimension || !r->groups ||
        r->heads % r->groups || !r->convolution_kernel || !r->normalization_groups ||
        r->heads % r->normalization_groups ||
        !isfinite((float)r->normalization_epsilon) || (float)r->normalization_epsilon <= 0.0f ||
        !isfinite((float)r->time_step_minimum) || r->time_step_minimum < 0.0 ||
        !isfinite((float)r->time_step_maximum) ||
        (r->time_step_unbounded != 0 && r->time_step_unbounded != 1) ||
        (r->norm_before_gate != 0 && r->norm_before_gate != 1) ||
        (r->time_step_unbounded ? r->time_step_maximum != 0.0 :
            r->time_step_maximum < r->time_step_minimum))
        return ssd_semantic_refuse(err, YVEX_ERR_INVALID_ARG, "selective SSD semantics are invalid");
    saved = *r;
    r = &saved;
    memset(plan, 0, sizeof(*plan));
    plan->schema_version = YVEX_SELECTIVE_SSD_SCHEMA_V1;
    plan->requirement = *r;
    if (!ssd_geometry(plan))
        return ssd_semantic_refuse(err, YVEX_ERR_BOUNDS, "selective SSD geometry overflowed");
    length = snprintf(description, sizeof(description),
        "yvex.selective-ssd.f32.v1:%llu:%llu:%llu:%llu:%llu:%llu:%.17g:%.17g:%.17g:%d:%d",
        r->heads, r->head_dimension, r->state_dimension, r->groups,
        r->convolution_kernel, r->normalization_groups, r->normalization_epsilon,
        r->time_step_minimum, r->time_step_maximum, r->time_step_unbounded, r->norm_before_gate);
    if (length < 0 || (size_t)length >= sizeof(description))
        return ssd_semantic_refuse(err, YVEX_ERR_BOUNDS, "selective SSD identity exceeds bounds");
    yvex_sha256_init(&hash);
    yvex_sha256_update(&hash, description, (size_t)length);
    yvex_sha256_final(&hash, digest);
    yvex_sha256_hex(digest, plan->identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_selective_ssd_geometry_validate(const yvex_selective_ssd_geometry *p, yvex_error *err)
{
    yvex_selective_ssd_geometry copy;

    if (!p || p->schema_version != YVEX_SELECTIVE_SSD_SCHEMA_V1 ||
        !yvex_sha256_hex_valid(p->identity) ||
        yvex_selective_ssd_geometry_seal(&copy, &p->requirement, err) != YVEX_OK ||
        strcmp(copy.identity, p->identity) || copy.width != p->width ||
        copy.convolution_width != p->convolution_width ||
        copy.projection_width != p->projection_width ||
        copy.convolution_state_values != p->convolution_state_values ||
        copy.recurrent_state_values != p->recurrent_state_values)
        return ssd_semantic_refuse(err, YVEX_ERR_FORMAT, "selective SSD geometry is malformed or stale");
    return YVEX_OK;
}
