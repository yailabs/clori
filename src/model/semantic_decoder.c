/* Validate and authenticate family-neutral stateful decoder semantics. */
#include <yvex/internal/semantic_decoder.h>

#include <math.h>
#include <string.h>

#include <yvex/internal/core.h>

static int gated_delta_requirement_valid(
    const yvex_gated_delta_requirement *requirement)
{
    return requirement &&
           requirement->schema_version ==
               YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V1 &&
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
