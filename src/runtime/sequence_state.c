/* Keep heterogeneous recurrent layer state coherent with the runtime transaction boundary. */
#include <yvex/internal/sequence_state.h>

#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>
#include <yvex/internal/backend.h>

typedef struct {
    yvex_sequence_state_binding binding;
    unsigned long long convolution_offset, recurrent_offset;
    yvex_device_tensor device_convolution[2], device_recurrent[2];
} sequence_state_layer;

struct yvex_sequence_state {
    sequence_state_layer *layers;
    unsigned char *staged;
    float *banks[2];
    yvex_device_tensor *device_banks[2];
    yvex_backend *backend;
    unsigned long long bank_values, convolution_values, recurrent_values;
    unsigned long long binding_count, committed_position, candidate_tokens;
    unsigned long long generation, staged_layers;
    unsigned int committed_bank;
    yvex_backend_kind storage_backend;
    char plan_identity[YVEX_SHA256_HEX_CAP];
    int device_attached, transaction_active, prepared, invalidated;
};

static int sequence_state_refuse(
    yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.sequence-state", reason);
    return status;
}

static int sequence_state_identity(yvex_sequence_state *state)
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.sequence-state-plan.v1") ||
        !yvex_sha256_update_u64(&hash, state->binding_count))
        return 0;
    for (index = 0ull; index < state->binding_count; ++index)
        if (!yvex_sha256_update_u64(&hash, state->layers[index].binding.layer_index) ||
            !yvex_sha256_update_text(
                &hash, state->layers[index].binding.plan.identity))
            return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, state->plan_identity);
    return 1;
}

static int sequence_state_layout(
    yvex_sequence_state *state, const yvex_sequence_state_plan *plan,
    yvex_error *err)
{
    unsigned long long index, convolution = 0ull, recurrent = 0ull;

    for (index = 0ull; index < plan->binding_count; ++index) {
        const yvex_sequence_state_binding *binding = &plan->bindings[index];
        sequence_state_layer *layer = &state->layers[index];

        if (yvex_gated_delta_plan_validate(&binding->plan, err) != YVEX_OK)
            return sequence_state_refuse(
                err, YVEX_ERR_FORMAT, "one recurrent layer plan is not sealed");
        if (index && binding->layer_index <=
                         plan->bindings[index - 1ull].layer_index)
            return sequence_state_refuse(
                err, YVEX_ERR_FORMAT,
                "recurrent layer bindings must be unique and ordered");
        layer->binding = *binding;
        layer->convolution_offset = convolution;
        layer->recurrent_offset = recurrent;
        if (!yvex_core_u64_add(
                convolution, binding->plan.convolution_state_values,
                &convolution) ||
            !yvex_core_u64_add(
                recurrent, binding->plan.recurrent_state_values, &recurrent))
            return sequence_state_refuse(
                err, YVEX_ERR_BOUNDS, "recurrent state geometry overflowed");
    }
    state->convolution_values = convolution;
    state->recurrent_values = recurrent;
    return yvex_core_u64_add(convolution, recurrent, &state->bank_values)
               ? YVEX_OK
               : sequence_state_refuse(
                     err, YVEX_ERR_BOUNDS,
                     "combined recurrent state geometry overflowed");
}

static int sequence_state_allocate_metadata(
    yvex_sequence_state *state, yvex_error *err)
{
    size_t layer_bytes;

    if (state->binding_count > SIZE_MAX / sizeof(*state->layers) ||
        state->binding_count > SIZE_MAX)
        return sequence_state_refuse(
            err, YVEX_ERR_BOUNDS, "recurrent state allocation exceeds this host");
    layer_bytes = (size_t)state->binding_count * sizeof(*state->layers);
    state->layers = calloc(1u, layer_bytes);
    state->staged = calloc((size_t)state->binding_count, 1u);
    if (!state->layers || !state->staged)
        return sequence_state_refuse(
            err, YVEX_ERR_NOMEM, "recurrent state allocation failed");
    return YVEX_OK;
}

static int sequence_state_allocate_banks(
    yvex_sequence_state *state, yvex_error *err)
{
    size_t state_bytes;

    if (!state->bank_values || state->bank_values > SIZE_MAX / sizeof(float))
        return sequence_state_refuse(
            err, YVEX_ERR_BOUNDS, "recurrent state allocation exceeds this host");
    state_bytes = (size_t)state->bank_values * sizeof(float);
    state->banks[0] = calloc(1u, state_bytes);
    state->banks[1] = calloc(1u, state_bytes);
    if (!state->banks[0] || !state->banks[1])
        return sequence_state_refuse(
            err, YVEX_ERR_NOMEM, "recurrent state bank allocation failed");
    return YVEX_OK;
}

int yvex_sequence_state_open_for_backend(
    yvex_sequence_state **out, const yvex_sequence_state_plan *plan,
    yvex_backend_kind backend, yvex_error *err)
{
    yvex_sequence_state *state;
    int rc;

    if (out) *out = NULL;
    if (!out || !plan ||
        (backend != YVEX_BACKEND_KIND_CPU &&
         backend != YVEX_BACKEND_KIND_CUDA) ||
        plan->schema_version != YVEX_SEQUENCE_STATE_SCHEMA_V1 ||
        !plan->bindings || !plan->binding_count)
        return sequence_state_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "one bounded recurrent sequence-state plan is required");
    state = calloc(1u, sizeof(*state));
    if (!state)
        return sequence_state_refuse(
            err, YVEX_ERR_NOMEM, "recurrent state owner allocation failed");
    state->binding_count = plan->binding_count;
    state->storage_backend = backend;
    rc = sequence_state_allocate_metadata(state, err);
    if (rc == YVEX_OK) rc = sequence_state_layout(state, plan, err);
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CPU)
        rc = sequence_state_allocate_banks(state, err);
    if (rc == YVEX_OK && !sequence_state_identity(state))
        rc = sequence_state_refuse(
            err, YVEX_ERR_STATE, "recurrent state plan identity failed");
    if (rc != YVEX_OK) {
        yvex_sequence_state_close(&state);
        return rc;
    }
    *out = state;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_open(
    yvex_sequence_state **out, const yvex_sequence_state_plan *plan,
    yvex_error *err)
{
    return yvex_sequence_state_open_for_backend(
        out, plan, YVEX_BACKEND_KIND_CPU, err);
}

static int sequence_state_device_views(
    yvex_sequence_state *state, yvex_error *err)
{
    unsigned long long index;
    unsigned int bank;

    for (index = 0ull; index < state->binding_count; ++index) {
        sequence_state_layer *layer = &state->layers[index];
        for (bank = 0u; bank < 2u; ++bank)
            if (!yvex_backend_tensor_f32_subview(
                    state->device_banks[bank], layer->convolution_offset,
                    layer->binding.plan.convolution_state_values,
                    &layer->device_convolution[bank]) ||
                !yvex_backend_tensor_f32_subview(
                    state->device_banks[bank],
                    state->convolution_values + layer->recurrent_offset,
                    layer->binding.plan.recurrent_state_values,
                    &layer->device_recurrent[bank]))
                return sequence_state_refuse(
                    err, YVEX_ERR_BOUNDS,
                    "recurrent device-state subview exceeds its bank");
    }
    return YVEX_OK;
}

int yvex_sequence_state_attach_device(
    yvex_sequence_state *state, yvex_backend *backend, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long bytes;
    unsigned int bank;
    int rc = YVEX_OK;

    if (!state || !backend || state->invalidated || state->transaction_active ||
        state->storage_backend != YVEX_BACKEND_KIND_CUDA ||
        yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA ||
        state->device_attached || state->backend)
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "idle unattached CUDA recurrent state and CUDA backend are required");
    if (!yvex_core_u64_mul(state->bank_values, sizeof(float), &bytes))
        return sequence_state_refuse(
            err, YVEX_ERR_BOUNDS, "recurrent device-state extent overflowed");
    descriptor.name = "runtime-sequence-state";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = state->bank_values;
    descriptor.bytes = bytes;
    state->backend = backend;
    for (bank = 0u; bank < 2u && rc == YVEX_OK; ++bank) {
        rc = yvex_backend_tensor_alloc(
            backend, &descriptor, &state->device_banks[bank], err);
        if (rc == YVEX_OK)
            rc = yvex_backend_tensor_zero(
                backend, state->device_banks[bank], err);
    }
    if (rc == YVEX_OK) rc = sequence_state_device_views(state, err);
    if (rc == YVEX_OK) {
        state->device_attached = 1;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    {
        yvex_error primary = err ? *err : (yvex_error){0}, cleanup;
        for (bank = 0u; bank < 2u; ++bank)
            if (state->device_banks[bank])
                (void)yvex_backend_tensor_release(
                    backend, &state->device_banks[bank], &cleanup);
        if (!state->device_banks[0] && !state->device_banks[1])
            state->backend = NULL;
        if (err) *err = primary;
    }
    return rc;
}

static sequence_state_layer *sequence_state_find(
    const yvex_sequence_state *state, unsigned long long layer_index,
    unsigned long long *ordinal)
{
    unsigned long long index;

    if (!state) return NULL;
    for (index = 0ull; index < state->binding_count; ++index)
        if (state->layers[index].binding.layer_index == layer_index) {
            if (ordinal) *ordinal = index;
            return &state->layers[index];
        }
    return NULL;
}

static void sequence_state_view(
    const yvex_sequence_state *state, const sequence_state_layer *layer,
    unsigned int bank, yvex_gated_delta_state_view *view)
{
    const float *base = state->banks[bank];

    view->convolution = base + layer->convolution_offset;
    view->recurrent = base + state->convolution_values + layer->recurrent_offset;
}

int yvex_sequence_state_begin(
    yvex_sequence_state *state, unsigned long long token_start,
    unsigned long long token_count, yvex_error *err)
{
    if (!state || !token_count || state->invalidated ||
        (state->storage_backend == YVEX_BACKEND_KIND_CUDA &&
         !state->device_attached))
        return sequence_state_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "one non-empty recurrent state extension is required");
    if (state->transaction_active || token_start != state->committed_position ||
        ULLONG_MAX - token_start < token_count)
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "recurrent state must extend the exact committed position");
    memset(state->staged, 0, (size_t)state->binding_count);
    state->candidate_tokens = token_count;
    state->staged_layers = 0ull;
    state->transaction_active = 1;
    state->prepared = 0;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_layer(
    yvex_sequence_state *state, unsigned long long layer_index,
    yvex_gated_delta_state_view *committed,
    yvex_gated_delta_state_output *candidate, yvex_error *err)
{
    sequence_state_layer *layer;
    unsigned long long ordinal;
    unsigned int candidate_bank;

    if (!state || !committed || !candidate || state->invalidated ||
        state->storage_backend != YVEX_BACKEND_KIND_CPU ||
        !state->transaction_active ||
        state->prepared)
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "one active unprepared recurrent state transaction is required");
    layer = sequence_state_find(state, layer_index, &ordinal);
    if (!layer || state->staged[ordinal])
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "one unstaged recurrent layer binding is required");
    sequence_state_view(state, layer, state->committed_bank, committed);
    candidate_bank = 1u - state->committed_bank;
    candidate->convolution = state->banks[candidate_bank] +
                             layer->convolution_offset;
    candidate->convolution_capacity =
        layer->binding.plan.convolution_state_values;
    candidate->recurrent = state->banks[candidate_bank] +
        state->convolution_values + layer->recurrent_offset;
    candidate->recurrent_capacity = layer->binding.plan.recurrent_state_values;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_device_layer(
    yvex_sequence_state *state, unsigned long long layer_index,
    yvex_gated_delta_device_state_view *committed,
    yvex_gated_delta_device_state_output *candidate, yvex_error *err)
{
    sequence_state_layer *layer;
    unsigned long long ordinal;
    unsigned int candidate_bank;

    if (!state || !committed || !candidate || state->invalidated ||
        state->storage_backend != YVEX_BACKEND_KIND_CUDA ||
        !state->device_attached || !state->transaction_active || state->prepared)
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "one active CUDA recurrent state transaction is required");
    layer = sequence_state_find(state, layer_index, &ordinal);
    if (!layer || state->staged[ordinal])
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "one unstaged recurrent layer binding is required");
    memset(committed, 0, sizeof(*committed));
    if (state->committed_position) {
        committed->convolution =
            &layer->device_convolution[state->committed_bank];
        committed->recurrent = &layer->device_recurrent[state->committed_bank];
    }
    candidate_bank = 1u - state->committed_bank;
    layer->device_convolution[candidate_bank].is_written = 0;
    layer->device_recurrent[candidate_bank].is_written = 0;
    candidate->convolution = &layer->device_convolution[candidate_bank];
    candidate->recurrent = &layer->device_recurrent[candidate_bank];
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_stage(
    yvex_sequence_state *state, unsigned long long layer_index,
    yvex_error *err)
{
    unsigned long long ordinal;

    if (!state || state->invalidated || !state->transaction_active ||
        state->prepared ||
        !sequence_state_find(state, layer_index, &ordinal) ||
        state->staged[ordinal])
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "one completed unstaged recurrent layer is required");
    if (state->storage_backend == YVEX_BACKEND_KIND_CUDA) {
        sequence_state_layer *layer = &state->layers[ordinal];
        unsigned int bank = 1u - state->committed_bank;
        if (!layer->device_convolution[bank].is_written ||
            !layer->device_recurrent[bank].is_written)
            return sequence_state_refuse(
                err, YVEX_ERR_STATE,
                "device recurrent layer must be fully written before staging");
    }
    state->staged[ordinal] = 1u;
    state->staged_layers++;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_committed(
    const yvex_sequence_state *state, unsigned long long layer_index,
    yvex_gated_delta_state_view *committed, yvex_error *err)
{
    sequence_state_layer *layer;

    if (!state || !committed || state->invalidated ||
        state->storage_backend != YVEX_BACKEND_KIND_CPU)
        return sequence_state_refuse(
            err, YVEX_ERR_INVALID_ARG, "committed recurrent state view is required");
    layer = sequence_state_find(state, layer_index, NULL);
    if (!layer)
        return sequence_state_refuse(
            err, YVEX_ERR_STATE, "recurrent layer binding is absent");
    sequence_state_view(state, layer, state->committed_bank, committed);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int sequence_state_prepare(void *opaque, yvex_error *err)
{
    yvex_sequence_state *state = opaque;

    if (!state || state->invalidated || !state->transaction_active ||
        state->prepared ||
        state->staged_layers != state->binding_count)
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "every recurrent layer must be staged before publication");
    state->prepared = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static void sequence_state_publish(void *opaque)
{
    yvex_sequence_state *state = opaque;

    if (!state || !state->transaction_active || !state->prepared) return;
    state->committed_bank = 1u - state->committed_bank;
    state->committed_position += state->candidate_tokens;
    state->generation++;
    state->candidate_tokens = 0ull;
    state->staged_layers = 0ull;
    state->transaction_active = 0;
    state->prepared = 0;
    memset(state->staged, 0, (size_t)state->binding_count);
}

static int sequence_state_abort(void *opaque, yvex_error *err)
{
    yvex_sequence_state *state = opaque;
    size_t bytes;

    if (!state)
        return sequence_state_refuse(
            err, YVEX_ERR_INVALID_ARG, "recurrent state owner is required");
    if (state->bank_values > SIZE_MAX / sizeof(float))
        return sequence_state_refuse(
            err, YVEX_ERR_BOUNDS, "candidate recurrent state extent overflowed");
    bytes = (size_t)state->bank_values * sizeof(float);
    if (state->storage_backend == YVEX_BACKEND_KIND_CPU)
        memset(state->banks[1u - state->committed_bank], 0, bytes);
    memset(state->staged, 0, (size_t)state->binding_count);
    state->candidate_tokens = 0ull;
    state->staged_layers = 0ull;
    state->transaction_active = 0;
    state->prepared = 0;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_participant(
    yvex_sequence_state *state,
    yvex_runtime_transaction_participant *participant, yvex_error *err)
{
    if (!state || !participant || state->invalidated ||
        !state->transaction_active)
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "an active recurrent state transaction is required");
    *participant = (yvex_runtime_transaction_participant){
        .context = state,
        .prepare = sequence_state_prepare,
        .publish = sequence_state_publish,
        .abort = sequence_state_abort};
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_reset(yvex_sequence_state *state, yvex_error *err)
{
    size_t bytes;

    if (!state || state->invalidated || state->transaction_active ||
        state->bank_values > SIZE_MAX / sizeof(float))
        return sequence_state_refuse(
            err, YVEX_ERR_STATE,
            "idle bounded recurrent state is required for reset");
    bytes = (size_t)state->bank_values * sizeof(float);
    if (state->storage_backend == YVEX_BACKEND_KIND_CPU) {
        memset(state->banks[0], 0, bytes);
        memset(state->banks[1], 0, bytes);
    } else {
        int rc;

        if (!state->device_attached)
            return sequence_state_refuse(
                err, YVEX_ERR_STATE,
                "CUDA recurrent state must be attached before reset");
        rc = yvex_backend_tensor_zero(
            state->backend, state->device_banks[0], err);
        if (rc == YVEX_OK)
            rc = yvex_backend_tensor_zero(
                state->backend, state->device_banks[1], err);
        if (rc != YVEX_OK) return rc;
        if (sequence_state_device_views(state, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    memset(state->staged, 0, (size_t)state->binding_count);
    state->committed_bank = 0u;
    state->committed_position = 0ull;
    state->generation++;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_invalidate(
    yvex_sequence_state *state, yvex_error *err)
{
    size_t bytes;

    if (!state || state->bank_values > SIZE_MAX / sizeof(float))
        return sequence_state_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "bounded recurrent state is required for invalidation");
    if (state->transaction_active && sequence_state_abort(state, err) != YVEX_OK)
        return yvex_error_code(err);
    bytes = (size_t)state->bank_values * sizeof(float);
    if (state->storage_backend == YVEX_BACKEND_KIND_CPU) {
        memset(state->banks[0], 0, bytes);
        memset(state->banks[1], 0, bytes);
    } else if (state->device_attached) {
        int rc = yvex_backend_tensor_zero(
            state->backend, state->device_banks[0], err);
        if (rc == YVEX_OK)
            rc = yvex_backend_tensor_zero(
                state->backend, state->device_banks[1], err);
        if (rc != YVEX_OK) {
            state->invalidated = 1;
            return rc;
        }
    }
    state->committed_position = 0ull;
    state->generation++;
    state->invalidated = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_fork(
    yvex_sequence_state **out, const yvex_sequence_state *source,
    yvex_error *err)
{
    yvex_sequence_state_plan plan;
    yvex_sequence_state_binding *bindings;
    yvex_sequence_state *child = NULL;
    size_t binding_bytes, state_bytes;
    unsigned long long index;
    int rc;

    if (out) *out = NULL;
    if (!out || !source || source->invalidated || source->transaction_active ||
        source->binding_count > SIZE_MAX / sizeof(*bindings) ||
        source->bank_values > SIZE_MAX / sizeof(float))
        return sequence_state_refuse(
            err, YVEX_ERR_STATE, "idle recurrent source state is required for fork");
    if (source->storage_backend != YVEX_BACKEND_KIND_CPU)
        return sequence_state_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "device-authored recurrent state fork is not implemented");
    binding_bytes = (size_t)source->binding_count * sizeof(*bindings);
    bindings = malloc(binding_bytes);
    if (!bindings)
        return sequence_state_refuse(
            err, YVEX_ERR_NOMEM, "recurrent fork plan allocation failed");
    for (index = 0ull; index < source->binding_count; ++index)
        bindings[index] = source->layers[index].binding;
    plan = (yvex_sequence_state_plan){
        .schema_version = YVEX_SEQUENCE_STATE_SCHEMA_V1,
        .bindings = bindings,
        .binding_count = source->binding_count};
    rc = yvex_sequence_state_open(&child, &plan, err);
    free(bindings);
    if (rc != YVEX_OK) return rc;
    state_bytes = (size_t)source->bank_values * sizeof(float);
    memcpy(child->banks[0], source->banks[source->committed_bank], state_bytes);
    child->committed_position = source->committed_position;
    child->generation = source->generation;
    *out = child;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_summary_copy(
    const yvex_sequence_state *state, yvex_sequence_state_summary *summary,
    yvex_error *err)
{
    unsigned long long bank_bytes, total_bytes;

    if (!state || !summary)
        return sequence_state_refuse(
            err, YVEX_ERR_INVALID_ARG, "recurrent state summary storage is required");
    if (!yvex_core_u64_mul(state->bank_values, sizeof(float), &bank_bytes) ||
        !yvex_core_u64_mul(bank_bytes, 2ull, &total_bytes))
        return sequence_state_refuse(
            err, YVEX_ERR_BOUNDS, "recurrent state byte summary overflowed");
    memset(summary, 0, sizeof(*summary));
    summary->schema_version = YVEX_SEQUENCE_STATE_SCHEMA_V1;
    summary->binding_count = state->binding_count;
    summary->committed_position = state->committed_position;
    summary->candidate_tokens = state->candidate_tokens;
    summary->generation = state->generation;
    summary->staged_layers = state->staged_layers;
    summary->convolution_state_bytes = state->convolution_values * sizeof(float);
    summary->recurrent_state_bytes = state->recurrent_values * sizeof(float);
    summary->committed_state_bytes = bank_bytes;
    summary->candidate_state_bytes = summary->committed_state_bytes;
    summary->storage_backend = state->storage_backend;
    summary->host_state_bytes = state->storage_backend == YVEX_BACKEND_KIND_CPU
                                    ? total_bytes
                                    : 0ull;
    summary->device_state_bytes = state->device_attached
                                      ? total_bytes
                                      : 0ull;
    summary->host_authoritative =
        state->storage_backend == YVEX_BACKEND_KIND_CPU;
    summary->device_authoritative =
        state->storage_backend == YVEX_BACKEND_KIND_CUDA &&
        state->device_attached;
    summary->device_attached = state->device_attached;
    summary->fork_supported = summary->host_authoritative;
    yvex_core_text_copy(summary->plan_identity, sizeof(summary->plan_identity),
                        state->plan_identity);
    summary->transaction_active = state->transaction_active;
    summary->prepared = state->prepared;
    summary->invalidated = state->invalidated;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_sequence_state_close_checked(
    yvex_sequence_state **state_pointer, yvex_error *err)
{
    yvex_sequence_state *state;
    yvex_error cleanup;
    unsigned int bank;
    int rc = YVEX_OK;

    if (!state_pointer || !*state_pointer) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    state = *state_pointer;
    for (bank = 0u; bank < 2u; ++bank)
        if (state->device_banks[bank]) {
            int release = state->backend
                              ? yvex_backend_tensor_release(
                                    state->backend, &state->device_banks[bank],
                                    &cleanup)
                              : YVEX_ERR_STATE;
            if (rc == YVEX_OK && release != YVEX_OK) {
                rc = release;
                if (state->backend && err) *err = cleanup;
                else
                    yvex_error_set(
                        err, release, "runtime.sequence-state.close",
                        "device recurrent state lost its backend owner");
            }
        }
    if (rc != YVEX_OK) return rc;
    free(state->banks[0]);
    free(state->banks[1]);
    free(state->staged);
    free(state->layers);
    memset(state, 0, sizeof(*state));
    free(state);
    *state_pointer = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_sequence_state_close(yvex_sequence_state **state_pointer)
{
    yvex_error err;
    (void)yvex_sequence_state_close_checked(state_pointer, &err);
}
