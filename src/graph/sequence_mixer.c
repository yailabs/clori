/* Seal backend-neutral recurrent gated-delta geometry and state economics. */
#include <yvex/internal/sequence_mixer.h>

#include <string.h>

#include <yvex/internal/core.h>

static int mixer_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "graph.sequence-mixer", reason);
    return status;
}

static int mixer_geometry(yvex_gated_delta_plan *plan)
{
    const yvex_gated_delta_requirement *r = &plan->requirement;
    unsigned long long history;

    return yvex_core_u64_mul(r->query_heads, r->key_head_dimension,
                             &plan->query_width) &&
           yvex_core_u64_mul(r->key_heads, r->key_head_dimension,
                             &plan->key_width) &&
           yvex_core_u64_mul(r->value_heads, r->value_head_dimension,
                             &plan->value_width) &&
           yvex_core_u64_add(plan->query_width, plan->key_width, &plan->qkv_width) &&
           yvex_core_u64_add(plan->qkv_width, plan->value_width, &plan->qkv_width) &&
           yvex_core_u64_mul(plan->qkv_width, r->convolution_kernel - 1ull,
                             &plan->convolution_state_values) &&
           yvex_core_u64_mul(r->value_heads, r->key_head_dimension,
                             &plan->recurrent_state_values) &&
           yvex_core_u64_mul(plan->recurrent_state_values,
                             r->value_head_dimension,
                             &plan->recurrent_state_values) &&
           yvex_core_u64_mul(plan->convolution_state_values, sizeof(float),
                             &plan->convolution_state_bytes) &&
           yvex_core_u64_mul(plan->recurrent_state_values, sizeof(float),
                             &plan->recurrent_state_bytes) &&
           yvex_core_u64_add(plan->convolution_state_values,
                             plan->recurrent_state_values, &history);
}

int yvex_gated_delta_plan_seal(
    yvex_gated_delta_plan *plan, const yvex_gated_delta_requirement *requirement,
    yvex_error *err)
{
    if (plan) memset(plan, 0, sizeof(*plan));
    if (!plan || !yvex_semantic_gated_delta_requirement_identity(
                     requirement, plan->identity))
        return mixer_refuse(err, YVEX_ERR_INVALID_ARG,
                            "one exact F32 gated-delta requirement is required");
    plan->schema_version = YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V1;
    plan->requirement = *requirement;
    if (!mixer_geometry(plan)) {
        memset(plan, 0, sizeof(*plan));
        return mixer_refuse(err, YVEX_ERR_BOUNDS,
                            "gated-delta geometry or state bytes overflowed");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_gated_delta_plan_validate(
    const yvex_gated_delta_plan *plan, yvex_error *err)
{
    yvex_gated_delta_plan copy;

    if (!plan || plan->schema_version != YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V1 ||
        !yvex_sha256_hex_valid(plan->identity) ||
        yvex_gated_delta_plan_seal(&copy, &plan->requirement, err) != YVEX_OK)
        return mixer_refuse(err, YVEX_ERR_FORMAT,
                            "sealed gated-delta plan is malformed");
    if (copy.schema_version != plan->schema_version ||
        copy.query_width != plan->query_width ||
        copy.key_width != plan->key_width ||
        copy.value_width != plan->value_width ||
        copy.qkv_width != plan->qkv_width ||
        copy.convolution_state_values != plan->convolution_state_values ||
        copy.recurrent_state_values != plan->recurrent_state_values ||
        copy.convolution_state_bytes != plan->convolution_state_bytes ||
        copy.recurrent_state_bytes != plan->recurrent_state_bytes ||
        strcmp(copy.identity, plan->identity) != 0)
        return mixer_refuse(err, YVEX_ERR_STATE,
                            "sealed gated-delta plan identity is stale");
    yvex_error_clear(err);
    return YVEX_OK;
}
