/* Coordinate logical candidate publication with the matching physical residency bank. */
#include "src/runtime/private.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static int bridge_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    if (err && yvex_error_is_set(err)) return yvex_error_code(err);
    yvex_error_set(err, status, "runtime.attention.state", reason);
    return status;
}

static int bridge_begin(
    void *context, unsigned long long layer_ordinal,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *initial_history,
    unsigned long long token_position, unsigned long long count,
    const yvex_attention_cancellation *cancellation,
    const yvex_attention_history_view **history,
    yvex_attention_failure *failure, yvex_error *err)
{
    runtime_attention_state_bridge *bridge = context;
    unsigned long long expected;
    int rc;
    if (!bridge || !bridge->provider || !bridge->provider->begin || bridge->layer_active ||
        (bridge->pending_layer_count &&
         (!yvex_core_u64_add(bridge->pending_layer_ordinal,
                             bridge->pending_layer_count, &expected) ||
          expected != layer_ordinal)))
        return bridge_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "valid ordered state bridge and bounded token range are required");
    rc = bridge->provider->begin(
        bridge->provider->context, layer_ordinal, layer, initial_history,
        token_position, count, cancellation, history, failure, err);
    if (rc == YVEX_OK && bridge->residency)
        rc = yvex_runtime_state_residency_transition(
            bridge->residency, bridge->provider, NULL, layer_ordinal, count,
            YVEX_RUNTIME_STATE_BEGIN, err);
    if (rc == YVEX_OK) {
        bridge->active_layer_ordinal = layer_ordinal;
        bridge->layer_active = 1;
    }
    return rc;
}

static int bridge_hash_output(
    runtime_attention_state_bridge *bridge,
    const yvex_attention_publication *publication)
{
    const float *values;
    unsigned long long width, count, index;
    width = bridge->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE
                ? publication->envelope_output_width
                : publication->core_output_width;
    if (!width || !yvex_core_u64_mul(publication->token_count, width, &count)) return 0;
    if (publication->evidence_level == YVEX_ATTENTION_EVIDENCE_NONE) {
        if (!yvex_sha256_update_text(&bridge->output_hash,
                                     publication->execution_identity))
            return 0;
    } else {
        values = bridge->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE
                     ? publication->envelope_output : publication->core_output;
        if (!values) return 0;
        for (index = 0ull; index < count; ++index) {
            uint32_t bits;
            if (!isfinite(values[index])) return 0;
            memcpy(&bits, &values[index], sizeof(bits));
            if (!yvex_sha256_update_u64(&bridge->output_hash,
                                        (unsigned long long)bits))
                return 0;
        }
    }
    return yvex_core_u64_add(bridge->output_values, count,
                             &bridge->output_values);
}

static int bridge_stage(
    void *context, const yvex_attention_publication *publication,
    const yvex_attention_cancellation *cancellation,
    char state_delta_identity[YVEX_SHA256_HEX_CAP],
    yvex_attention_failure *failure, yvex_error *err)
{
    runtime_attention_state_bridge *bridge = context;
    unsigned long long ordinal, next;
    int pending, completing, rc;
    if (!bridge || !bridge->provider || !bridge->provider->stage || !publication)
        return bridge_refuse(
            err, YVEX_ERR_FORMAT,
            "complete attention output publication is required");
    pending = publication->device_completion_pending;
    completing = !pending && bridge->pending_layer_count != 0ull;
    if ((pending && !bridge->layer_active) ||
        (completing && bridge->layer_active) ||
        (!pending && !completing && !bridge->layer_active) ||
        (bridge->hash_output && !completing &&
         !bridge_hash_output(bridge, publication)))
        return bridge_refuse(
            err, YVEX_ERR_FORMAT,
            "attention publication does not match its state transaction");
    ordinal = completing ? bridge->pending_layer_ordinal
                         : bridge->active_layer_ordinal;
    rc = bridge->provider->stage(
        bridge->provider->context, publication, cancellation,
        state_delta_identity, failure, err);
    if (rc != YVEX_OK) return rc;
    if (pending) {
        if ((state_delta_identity && state_delta_identity[0]) ||
            !yvex_core_u64_add(bridge->pending_layer_count, 1ull, &next))
            return bridge_refuse(
                err, YVEX_ERR_STATE,
                "pending attention state published a completed identity");
        if (!bridge->pending_layer_count)
            bridge->pending_layer_ordinal = ordinal;
        bridge->pending_layer_count = next;
        bridge->layer_active = 0;
        return YVEX_OK;
    }
    /* Prefix selection stages the physical bank; staging here would publish rejected suffixes. */
    if (bridge->residency && !publication->prefix_addressable)
        rc = yvex_runtime_state_residency_transition(
            bridge->residency, bridge->provider, publication, ordinal, 0ull,
            YVEX_RUNTIME_STATE_STAGE, err);
    if (rc != YVEX_OK) return rc;
    if (state_delta_identity)
        yvex_runtime_identity_copy(
            bridge->last_delta_identity, state_delta_identity);
    if (completing) {
        bridge->pending_layer_count--;
        bridge->pending_layer_ordinal++;
    } else {
        bridge->layer_active = 0;
    }
    return YVEX_OK;
}

int yvex_runtime_private_attention_state_abort(
    void *context, yvex_attention_failure *failure, yvex_error *err)
{
    runtime_attention_state_bridge *bridge = context;
    int rc;
    if (!bridge || !bridge->provider || !bridge->provider->abort)
        return bridge_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "attention state bridge is required for abort");
    rc = bridge->provider->abort(bridge->provider->context, failure, err);
    bridge->layer_active = 0;
    bridge->pending_layer_count = 0ull;
    return rc;
}

yvex_attention_probe_state_provider yvex_runtime_private_attention_state_provider(
    runtime_attention_state_bridge *bridge)
{
    yvex_attention_probe_state_provider provider = {0};
    provider.context = bridge;
    provider.begin = bridge_begin;
    provider.stage = bridge_stage;
    provider.abort = yvex_runtime_private_attention_state_abort;
    return provider;
}
