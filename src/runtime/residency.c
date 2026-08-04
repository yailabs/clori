/*
 * Retain exact encoded weights and state across warm executions.
 *
 * Immutable ranges read once; mutable state uses stable addresses and explicit generations. Models
 * share weights; sessions own persistent state and mutable staging.
 */
#include "src/runtime/private.h"

#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <yvex/artifact.h>
#include <yvex/internal/core.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/runtime.h>
typedef enum {
    RESIDENCY_BINDING_CORE = YVEX_ATTENTION_BINDING_CORE,
    RESIDENCY_BINDING_ENVELOPE = YVEX_ATTENTION_BINDING_ENVELOPE,
    RESIDENCY_BINDING_OUTPUT_HEAD = 3,
    RESIDENCY_BINDING_MODEL = 4
} residency_binding_class;
typedef struct {
    const yvex_materialized_tensor_binding *binding;
    residency_binding_class binding_class;
    unsigned long long arena_offset;
} residency_record;
struct yvex_runtime_residency {
    yvex_materialization_session *materialization;
    unsigned char *arena;
    residency_record *records;
    unsigned long long *record_index;
    unsigned long long record_index_count;
    yvex_backend *cuda_backend;
    yvex_device_tensor *cuda_weights;
    unsigned long long cuda_addressable_device_base;
    yvex_runtime_residency_summary summary;
    pthread_mutex_t access_mutex;
    int access_mutex_ready, arena_locked, arena_mapped, arena_registered;
};
typedef struct {
    const void *host[2][3];
    unsigned long long offset[3], bytes[3];
} state_resident_component;
typedef struct {
    int selected, staged;
    unsigned long long bytes;
    yvex_attention_state_recipe recipe;
    yvex_device_tensor *device[2];
    unsigned char *host[2];
    unsigned int component_count;
    state_resident_component components[YVEX_ATTENTION_STATE_COMPONENT_CAP];
} state_resident_layer;
struct yvex_runtime_state_residency {
    yvex_backend *backend;
    state_resident_layer *layers;
    unsigned long long layer_count;
    yvex_runtime_state_residency_summary summary;
};

static int state_resident_span(unsigned long long bytes, unsigned long long *cursor,
                               unsigned long long *offset)
{
    unsigned long long aligned;
    if (!cursor || !offset || *cursor > ULLONG_MAX - 7ull) return 0;
    aligned = (*cursor + 7ull) & ~7ull;
    if (bytes > ULLONG_MAX - aligned) return 0;
    *offset = aligned;
    *cursor = aligned + bytes;
    return 1;
}
/* Derive one pointer-free mutable-state residency identity. */
static int state_resident_identity(
    const yvex_runtime_state_residency *residency,
    const yvex_graph_attention_capacity_summary *capacity,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_backend_device_info device;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_error err;
    if (!residency || !capacity ||
        yvex_backend_get_device_info(residency->backend, &device, &err) != YVEX_OK)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.persistent-state-residency.v1") ||
        !yvex_sha256_update_text(&hash, capacity->identity) ||
        !yvex_sha256_update_u64(&hash, yvex_backend_kind_of(residency->backend)) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)device.device_index) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)device.compute_capability_major) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)device.compute_capability_minor) ||
        !yvex_sha256_update_u64(&hash, residency->layer_count))
        return 0;
    for (index = 0ull; index < residency->layer_count; ++index) {
        const state_resident_layer *layer = &residency->layers[index];
        if (!yvex_sha256_update_u64(&hash, index) ||
            !yvex_sha256_update_u64(&hash, (unsigned long long)layer->selected) ||
            (layer->selected &&
             !yvex_sha256_update_u64(&hash, layer->bytes)))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int state_resident_source(
    const yvex_attention_history_view *view,
    yvex_attention_state_binding binding, const void *spans[3])
{
    memset(spans, 0, 3u * sizeof(*spans));
    if (!view) return 0;
    switch (binding) {
    case YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY:
        spans[0] = view->local_kv; spans[1] = view->local_positions; return 1;
    case YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY:
        spans[0] = view->compressed_kv; spans[1] = view->compressed_positions; return 1;
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY:
        spans[0] = view->indexer_kv; spans[1] = view->indexer_positions; return 1;
    case YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING:
        spans[0] = view->main_rolling_state.kv_state;
        spans[2] = view->main_rolling_state.score_state;
        return 1;
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING:
        spans[0] = view->indexer_rolling_state.kv_state;
        spans[2] = view->indexer_rolling_state.score_state;
        return 1;
    default: return 0;
    }
}

static int state_resident_pack(
    state_resident_layer *layer, unsigned int bank,
    const yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe, yvex_error *err)
{
    unsigned int index;
    if (!layer || bank > 1u || !view || !recipe ||
        recipe->component_count != layer->component_count) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.state.residency.pack",
                       "persistent state storage inventory is incompatible");
        return YVEX_ERR_FORMAT;
    }
    if (layer->host[bank])
        memset(layer->host[bank], 0, (size_t)layer->bytes);
    for (index = 0u; index < recipe->component_count; ++index) {
        state_resident_component *target = &layer->components[index];
        const void *pointers[3];
        unsigned int span;
        if (!state_resident_source(view, recipe->components[index].binding, pointers)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.state.residency.pack",
                           "persistent state component binding is invalid");
            return YVEX_ERR_FORMAT;
        }
        for (span = 0u; span < 3u; ++span) {
            if (target->bytes[span] && !pointers[span]) {
                yvex_error_set(err, YVEX_ERR_FORMAT,
                               "runtime.state.residency.pack",
                               "persistent state component span drifted");
                return YVEX_ERR_FORMAT;
            }
            target->host[bank][span] = pointers[span];
            if (target->bytes[span] && layer->host[bank])
                memcpy(layer->host[bank] + target->offset[span],
                       pointers[span], (size_t)target->bytes[span]);
        }
    }
    return YVEX_OK;
}

static int state_resident_upload(yvex_runtime_state_residency *residency,
                                 state_resident_layer *layer, unsigned int bank,
                                 yvex_error *err)
{
    int rc;
    if (yvex_backend_kind_of(residency->backend) != YVEX_BACKEND_KIND_CUDA)
        return YVEX_OK;
    rc = yvex_backend_tensor_write(
        residency->backend, layer->device[bank], layer->host[bank],
        layer->bytes, err);
    if (rc == YVEX_OK) {
        residency->summary.upload_count++;
        if (!yvex_core_u64_add(residency->summary.upload_bytes, layer->bytes,
                               &residency->summary.upload_bytes)) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.upload",
                           "persistent state upload accounting overflowed");
            return YVEX_ERR_BOUNDS;
        }
    }
    return rc;
}

static int state_resident_layer_layout(state_resident_layer *layer,
    const yvex_attention_state_recipe *recipe, yvex_error *err)
{
    unsigned long long cursor = 0ull;
    unsigned int index;
    if (!layer || !recipe || !recipe->component_count ||
        recipe->component_count > YVEX_ATTENTION_STATE_COMPONENT_CAP) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.state.residency.layout",
                       "persistent state recipe geometry is invalid");
        return YVEX_ERR_FORMAT;
    }
    memset(layer, 0, sizeof(*layer));
    layer->selected = 1;
    layer->recipe = *recipe;
    layer->component_count = recipe->component_count;
    for (index = 0u; index < recipe->component_count; ++index) {
        const yvex_attention_state_component_recipe *component =
            &recipe->components[index];
        state_resident_component *map = &layer->components[index];
        unsigned long long values, positions = 0ull, auxiliary = 0ull;
        if (component->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
            if (!yvex_core_u64_mul(component->capacity, component->value_width,
                                   &values) ||
                !yvex_core_u64_mul(values, sizeof(float), &values) ||
                !yvex_core_u64_mul(component->capacity,
                                   sizeof(unsigned long long), &positions))
                goto overflow;
        } else if (!yvex_core_u64_mul(component->rolling.kv_state_extent,
                                      sizeof(float), &values) ||
                   !yvex_core_u64_mul(component->rolling.score_state_extent,
                                      sizeof(float), &auxiliary)) {
            goto overflow;
        }
        map->bytes[0] = values;
        map->bytes[1] = positions;
        map->bytes[2] = auxiliary;
        if (!state_resident_span(values, &cursor, &map->offset[0]) ||
            !state_resident_span(positions, &cursor, &map->offset[1]) ||
            !state_resident_span(auxiliary, &cursor, &map->offset[2]))
            goto overflow;
    }
    if (cursor > (unsigned long long)SIZE_MAX) goto overflow;
    layer->bytes = cursor;
    return YVEX_OK;
overflow:
    yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.layout",
                   "persistent state device layout overflowed");
    return YVEX_ERR_BOUNDS;
}

static int state_resident_layer_prepare(
    yvex_runtime_state_residency *residency, state_resident_layer *layer,
    unsigned long long layer_index, const yvex_attention_state_provider *provider,
    const yvex_attention_state_recipe *recipe, yvex_error *err)
{
    const yvex_attention_history_view *view;
    yvex_backend_tensor_desc descriptor = {0};
    unsigned int bank;
    int rc = state_resident_layer_layout(layer, recipe, err);
    if (rc != YVEX_OK) return rc;
    for (bank = 0u; bank < 2u; ++bank) {
        if (yvex_backend_kind_of(residency->backend) == YVEX_BACKEND_KIND_CUDA)
            layer->host[bank] = calloc(1u, (size_t)layer->bytes);
        view = provider->view(
            provider->context, layer_index,
            bank ? YVEX_ATTENTION_STATE_VIEW_CANDIDATE
                 : YVEX_ATTENTION_STATE_VIEW_COMMITTED);
        rc = view ? YVEX_OK : YVEX_ERR_STATE;
        if ((yvex_backend_kind_of(residency->backend) == YVEX_BACKEND_KIND_CUDA &&
             !layer->host[bank]) ||
            rc != YVEX_OK ||
            state_resident_pack(layer, bank, view, recipe, err) != YVEX_OK)
        {
            if (rc != YVEX_OK && !yvex_error_is_set(err))
                yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.prepare",
                               "persistent state provider view is unavailable");
            else if (rc == YVEX_OK && !yvex_error_is_set(err))
                yvex_error_set(err, YVEX_ERR_NOMEM, "runtime.state.residency.prepare",
                               "persistent state staging allocation failed");
            return rc == YVEX_OK ? yvex_error_code(err) : rc;
        }
        if (yvex_backend_kind_of(residency->backend) == YVEX_BACKEND_KIND_CUDA) {
            char name[48];
            (void)snprintf(name, sizeof(name), "persistent-state-%llu-%u",
                           layer_index, bank);
            descriptor.name = name;
            descriptor.dtype = YVEX_DTYPE_I8;
            descriptor.rank = 1u;
            descriptor.dims[0] = layer->bytes;
            descriptor.bytes = layer->bytes;
            rc = yvex_backend_tensor_alloc(
                residency->backend, &descriptor, &layer->device[bank], err);
            if (rc != YVEX_OK ||
                state_resident_upload(residency, layer, bank, err) != YVEX_OK)
                return rc == YVEX_OK ? yvex_error_code(err) : rc;
            if (!yvex_core_u64_add(residency->summary.device_bytes, layer->bytes,
                                   &residency->summary.device_bytes)) {
                yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.prepare",
                               "persistent state device-byte accounting overflowed");
                return YVEX_ERR_BOUNDS;
            }
        }
    }
    return YVEX_OK;
}

static int state_resident_bank(const state_resident_layer *layer,
    const yvex_attention_history_view *view, unsigned int *bank, yvex_error *err)
{
    int matches[2] = {1, 1};
    unsigned int component, candidate, span;
    for (component = 0u; component < layer->component_count; ++component) {
        const state_resident_component *map = &layer->components[component];
        const void *pointers[3] = {0};
        if (!state_resident_source(view, layer->recipe.components[component].binding, pointers)) {
            matches[0] = matches[1] = 0;
            continue;
        }
        for (candidate = 0u; candidate < 2u; ++candidate)
            for (span = 0u; span < 3u; ++span)
                if (map->bytes[span] && pointers[span] != map->host[candidate][span])
                    matches[candidate] = 0;
    }
    if (matches[0] == matches[1]) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.bank",
                       "candidate state does not resolve to one stable bank");
        return YVEX_ERR_STATE;
    }
    *bank = matches[1] ? 1u : 0u;
    return YVEX_OK;
}

int yvex_runtime_private_state_residency_resolve(
    const void *context, const void *host, unsigned long long bytes,
    unsigned long long *device_address)
{
    const yvex_runtime_state_residency *residency = context;
    unsigned long long layer_index;
    uintptr_t pointer = (uintptr_t)host;
    if (device_address) *device_address = 0ull;
    if (!residency || !host || !bytes || !device_address)
        return YVEX_BACKEND_RESIDENT_INVALID;
    for (layer_index = 0ull; layer_index < residency->layer_count; ++layer_index) {
        const state_resident_layer *layer = &residency->layers[layer_index];
        unsigned int component, bank, span;
        if (!layer->selected) continue;
        for (component = 0u; component < layer->component_count; ++component)
            for (bank = 0u; bank < 2u; ++bank)
                for (span = 0u; span < 3u; ++span) {
                    uintptr_t base =
                        (uintptr_t)layer->components[component].host[bank][span];
                    unsigned long long extent =
                        layer->components[component].bytes[span];
                    if (!base || pointer < base ||
                        (unsigned long long)(pointer - base) > extent ||
                        bytes > extent - (unsigned long long)(pointer - base))
                        continue;
                    if (residency->summary.invalidated)
                        return YVEX_BACKEND_RESIDENT_INVALID;
                    *device_address =
                        (unsigned long long)(uintptr_t)layer->device[bank]->data +
                        layer->components[component].offset[span] +
                        (unsigned long long)(pointer - base);
                    return YVEX_BACKEND_RESIDENT_HIT;
                }
    }
    return YVEX_BACKEND_RESIDENT_MISS;
}

static int residency_reject(yvex_runtime_residency_failure *failure,
                            yvex_runtime_residency_failure_code code,
                            const yvex_runtime_tensor_binding *binding,
                            unsigned long long expected, unsigned long long actual,
                            const char *reason, yvex_status status, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->tensor_id = binding ? binding->tensor_id : ULLONG_MAX;
        failure->layer_index = binding ? binding->layer_index : ULLONG_MAX;
        failure->role = binding ? binding->role : YVEX_TENSOR_ROLE_UNKNOWN;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
    }
    yvex_error_set(err, status, "runtime.residency", reason);
    return status;
}
/* Resolve one exact resident record without copying or exposing arena ownership. */
static int residency_resolve(const void *context,
                             const yvex_materialized_tensor_binding *binding,
                             const unsigned char **data, unsigned long long *bytes)
{
    yvex_runtime_residency *residency = (yvex_runtime_residency *)context;
    unsigned long long slot;
    const residency_record *record;
    int invalidated;
    if (data) *data = NULL;
    if (bytes) *bytes = 0ull;
    if (!residency || !binding || !data || !bytes ||
        binding->tensor_id >= residency->record_index_count)
        return YVEX_MATERIALIZATION_READ_MISS;
    if (!residency->access_mutex_ready ||
        pthread_mutex_lock(&residency->access_mutex) != 0)
        return YVEX_MATERIALIZATION_READ_INVALID;
    invalidated = residency->summary.invalidated;
    (void)pthread_mutex_unlock(&residency->access_mutex);
    if (invalidated) return YVEX_MATERIALIZATION_READ_INVALID;
    slot = residency->record_index[binding->tensor_id];
    if (!slot) return YVEX_MATERIALIZATION_READ_MISS;
    record = &residency->records[slot - 1ull];
    if (!record->binding || record->binding->tensor_id != binding->tensor_id ||
        record->binding->qtype != binding->qtype ||
        record->binding->encoded_bytes != binding->encoded_bytes ||
        record->binding->absolute_offset != binding->absolute_offset ||
        strcmp(record->binding->name, binding->name) != 0)
        return YVEX_MATERIALIZATION_READ_INVALID;
    *data = residency->arena + record->arena_offset;
    *bytes = record->binding->encoded_bytes;
    return YVEX_MATERIALIZATION_READ_HIT;
}

static int residency_note_access(const void *context, unsigned long long bytes)
{
    yvex_runtime_residency *residency = (yvex_runtime_residency *)context;
    unsigned long long next_calls, next_bytes;
    if (!residency || !residency->access_mutex_ready ||
        pthread_mutex_lock(&residency->access_mutex) != 0)
        return 0;
    if (!yvex_core_u64_add(residency->summary.resident_read_calls, 1ull,
                           &next_calls) ||
        !yvex_core_u64_add(residency->summary.resident_bytes_read, bytes,
                           &next_bytes)) {
        (void)pthread_mutex_unlock(&residency->access_mutex);
        return 0;
    }
    residency->summary.resident_read_calls = next_calls;
    residency->summary.resident_bytes_read = next_bytes;
    (void)pthread_mutex_unlock(&residency->access_mutex);
    return 1;
}

static void residency_detached(const void *context)
{
    yvex_runtime_residency *residency = (yvex_runtime_residency *)context;
    if (!residency) return;
    residency->materialization = NULL;
    residency->summary.attached = 0;
}

static void residency_storage_release(yvex_runtime_residency **owner)
{
    yvex_runtime_residency *residency = owner ? *owner : NULL;
    if (!residency) return;
    if (residency->arena_locked)
        (void)munlock(residency->arena, (size_t)residency->summary.encoded_bytes);
    if (residency->arena_mapped)
        (void)munmap(residency->arena, (size_t)residency->summary.encoded_bytes);
    free(residency->records);
    free(residency->record_index);
    if (residency->access_mutex_ready)
        (void)pthread_mutex_destroy(&residency->access_mutex);
    memset(residency, 0, sizeof(*residency));
    free(residency);
    *owner = NULL;
}
/*
 * Release the model-owned CUDA residency without touching independent host storage.
 *
 * Exclusive residency ownership.
 */
static int residency_cuda_release(yvex_runtime_residency *residency, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!residency) return YVEX_OK;
    if (residency->cuda_backend) {
        rc = yvex_backend_resident_detach(residency->cuda_backend, err);
        if (rc != YVEX_OK) return rc;
        rc = yvex_backend_close_admit(residency->cuda_backend, err);
        if (rc != YVEX_OK) return rc;
    }
    if (residency->cuda_weights) {
        rc = yvex_backend_tensor_release(
            residency->cuda_backend, &residency->cuda_weights, err);
        if (rc != YVEX_OK) return rc;
    }
    rc = yvex_backend_close_checked(&residency->cuda_backend, err);
    if (rc != YVEX_OK) return rc;
    residency->summary.cuda_ready = 0;
    residency->summary.device_resident_bytes = 0ull;
    residency->summary.cuda_addressable_bytes = 0ull;
    residency->summary.cuda_upload_bytes = 0ull;
    residency->summary.cuda_upload_count = 0ull;
    residency->summary.cuda_host_registration_count = 0ull;
    residency->summary.cuda_managed_bytes = 0ull;
    residency->summary.cuda_managed_allocation_count = 0ull;
    residency->cuda_addressable_device_base = 0ull;
    residency->arena_registered = 0;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int residency_add_record(yvex_runtime_residency *residency,
                                const yvex_runtime_tensor_binding *runtime,
                                residency_binding_class binding_class,
                                unsigned long long ordinal,
                                unsigned long long *core_bytes,
                                unsigned long long core_qtypes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP],
                                yvex_runtime_residency_failure *failure, yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding = runtime ? runtime->binding : NULL;
    unsigned long long next;
    if (!binding || binding->tensor_id >= residency->record_index_count)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                residency->record_index_count,
                                binding ? binding->tensor_id : ULLONG_MAX,
                                "resident binding tensor identity is outside the descriptor",
                                YVEX_ERR_FORMAT, err);
    if (residency->record_index[binding->tensor_id])
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_DUPLICATE_BINDING,
                                runtime, 1ull, 2ull,
                                "resident binding was selected more than once",
                                YVEX_ERR_FORMAT, err);
    if (!binding->encoded_bytes || binding->encoded_bytes > (unsigned long long)SIZE_MAX ||
        binding->qtype >= YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                (unsigned long long)SIZE_MAX, binding->encoded_bytes,
                                "resident encoded range or qtype is outside platform bounds",
                                YVEX_ERR_BOUNDS, err);
    if (!yvex_core_u64_add(residency->summary.encoded_bytes, binding->encoded_bytes, &next))
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                ULLONG_MAX, binding->encoded_bytes,
                                "resident encoded byte accounting overflowed",
                                YVEX_ERR_BOUNDS, err);
    residency->records[ordinal].binding = binding;
    residency->records[ordinal].binding_class = binding_class;
    residency->records[ordinal].arena_offset = residency->summary.encoded_bytes;
    residency->record_index[binding->tensor_id] = ordinal + 1ull;
    residency->summary.encoded_bytes = next;
    residency->summary.qtype_binding_counts[binding->qtype]++;
    if (!yvex_core_u64_add(residency->summary.qtype_bytes[binding->qtype],
                           binding->encoded_bytes,
                           &residency->summary.qtype_bytes[binding->qtype]))
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                ULLONG_MAX, binding->encoded_bytes,
                                "resident qtype byte accounting overflowed",
                                YVEX_ERR_BOUNDS, err);
    if (binding_class == RESIDENCY_BINDING_CORE) {
        residency->summary.core_binding_count++;
        core_qtypes[binding->qtype]++;
        if (!yvex_core_u64_add(*core_bytes, binding->encoded_bytes, core_bytes))
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                                    ULLONG_MAX, binding->encoded_bytes,
                                    "resident core byte accounting overflowed",
                                    YVEX_ERR_BOUNDS, err);
    } else if (binding_class == RESIDENCY_BINDING_ENVELOPE) {
        residency->summary.envelope_binding_count++;
    } else if (binding_class == RESIDENCY_BINDING_OUTPUT_HEAD) {
        residency->summary.output_head_binding_count++;
        if (!yvex_core_u64_add(residency->summary.output_head_encoded_bytes,
                               binding->encoded_bytes,
                               &residency->summary.output_head_encoded_bytes))
            return residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY, runtime,
                ULLONG_MAX, binding->encoded_bytes,
                "resident output-head byte accounting overflowed",
                YVEX_ERR_BOUNDS, err);
    } else {
        residency->summary.model_binding_count++;
    }
    residency->summary.binding_count++;
    return YVEX_OK;
}
/* Hash exact resident payload bytes and immutable range boundaries. */
static int residency_load_and_hash(yvex_runtime_residency *residency,
                                   yvex_runtime_residency_failure *failure, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char tensor_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.resident.payload.v3") ||
        !yvex_sha256_update_u64(&hash, residency->summary.binding_count))
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_LIFECYCLE, NULL,
                                1ull, 0ull, "resident payload hash initialization failed",
                                YVEX_ERR_STATE, err);
    for (index = 0ull; index < residency->summary.binding_count; ++index) {
        const residency_record *record = &residency->records[index];
        const yvex_materialized_tensor_binding *binding = record->binding;
        unsigned char *destination = residency->arena + record->arena_offset;
        int rc = yvex_materialization_session_read(
            residency->materialization, binding, 0ull, destination,
            (size_t)binding->encoded_bytes, NULL, err);
        if (rc != YVEX_OK)
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_READ, NULL,
                                    binding->encoded_bytes, 0ull,
                                    "resident encoded range cold read failed",
                                    (yvex_status)rc, err);
        if (!yvex_core_u64_add(residency->summary.cold_artifact_read_calls, 1ull,
                               &residency->summary.cold_artifact_read_calls) ||
            !yvex_core_u64_add(residency->summary.cold_artifact_bytes_read,
                               binding->encoded_bytes,
                               &residency->summary.cold_artifact_bytes_read))
            return residency_reject(
                failure, YVEX_RUNTIME_RESIDENCY_FAILURE_LIFECYCLE, NULL,
                ULLONG_MAX, binding->encoded_bytes,
                "resident cold-read accounting overflowed", YVEX_ERR_BOUNDS, err);
        rc = yvex_artifact_sha256_hex_bytes(
            destination, binding->encoded_bytes, tensor_digest, err);
        if (rc != YVEX_OK)
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_LIFECYCLE, NULL,
                                    binding->encoded_bytes, 0ull,
                                    "resident tensor payload digest failed",
                                    (yvex_status)rc, err);
        if (!yvex_sha256_update_u64(&hash, binding->tensor_id) ||
            !yvex_sha256_update_u64(&hash, binding->qtype) ||
            !yvex_sha256_update_u64(&hash, binding->encoded_bytes) ||
            !yvex_sha256_update_text(&hash, tensor_digest))
            return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_LIFECYCLE, NULL,
                                    1ull, 0ull, "resident payload hash update failed",
                                    YVEX_ERR_STATE, err);
    }
    if (!yvex_sha256_final(&hash, digest))
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_LIFECYCLE, NULL,
                                1ull, 0ull, "resident payload hash finalization failed",
                                YVEX_ERR_STATE, err);
    yvex_sha256_hex(digest, residency->summary.payload_digest);
    return YVEX_OK;
}
/*
 * Derive residency identity from semantic range order and exact encoded payload.
 *
 * Writes one canonical content identity.
 */
static int residency_identity_build(yvex_runtime_residency *residency,
                                    const yvex_runtime_model_summary *model,
                                    const yvex_attention_summary *attention,
                                    yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.residency.v4") ||
        !yvex_sha256_update_u64(&hash, YVEX_RUNTIME_RESIDENCY_SCHEMA_V4) ||
        !yvex_sha256_update_text(&hash, model->runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, model->artifact_identity) ||
        !yvex_sha256_update_text(&hash, model->materialization_identity) ||
        !yvex_sha256_update_text(&hash, attention->attention_plan_identity) ||
        !yvex_sha256_update_u64(&hash, residency->summary.model_binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.core_binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.envelope_binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.output_head_binding_count) ||
        !yvex_sha256_update_u64(
            &hash, residency->summary.accelerator_encoded_bytes) ||
        !yvex_sha256_update_u64(&hash, residency->summary.encoded_bytes))
        goto failed;
    for (index = 0ull; index < residency->summary.binding_count; ++index) {
        const residency_record *record = &residency->records[index];
        if (!yvex_sha256_update_u64(&hash, record->binding->tensor_id) ||
            !yvex_sha256_update_u64(&hash, record->binding_class) ||
            !yvex_sha256_update_u64(&hash, record->binding->qtype) ||
            !yvex_sha256_update_u64(&hash, record->binding->encoded_bytes) ||
            !yvex_sha256_update_u64(&hash, record->arena_offset))
            goto failed;
    }
    if (!yvex_sha256_update_text(&hash, residency->summary.payload_digest) ||
        !yvex_sha256_final(&hash, digest))
        goto failed;
    yvex_sha256_hex(digest, residency->summary.residency_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.output-head.residency.v1") ||
        !yvex_sha256_update_text(&hash, model->runtime_model_identity) ||
        !yvex_sha256_update_u64(&hash, residency->summary.output_head_binding_count) ||
        !yvex_sha256_update_u64(&hash, residency->summary.output_head_encoded_bytes) ||
        !yvex_sha256_update_text(&hash, residency->summary.payload_digest) ||
        !yvex_sha256_final(&hash, digest))
        goto failed;
    yvex_sha256_hex(digest, residency->summary.output_head_residency_identity);
    return YVEX_OK;
failed:
    yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.identity",
                   "resident identity encoding failed");
    return YVEX_ERR_STATE;
}

static int residency_release(yvex_runtime_residency **owner, yvex_error *err)
{
    yvex_runtime_residency *residency = owner ? *owner : NULL;
    int rc;
    if (!residency) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (residency->summary.attached && residency->materialization)
        if ((rc = yvex_materialization_session_detach_read_provider(
                 residency->materialization, residency, NULL, err)) != YVEX_OK)
            return rc;
    rc = residency_cuda_release(residency, err);
    if (rc != YVEX_OK) return rc;
    residency_storage_release(owner);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int residency_arena_prepare(yvex_runtime_residency *residency,
                                   const yvex_runtime_residency_options *options,
                                   yvex_runtime_residency_failure *failure,
                                   yvex_error *err)
{
    int rc = YVEX_OK;
    if (residency->summary.encoded_bytes > (unsigned long long)SIZE_MAX)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_BUDGET, NULL,
                                (unsigned long long)SIZE_MAX,
                                residency->summary.encoded_bytes,
                                "resident arena exceeds platform allocation range",
                                YVEX_ERR_BOUNDS, err);
    if (options && options->maximum_host_bytes &&
        residency->summary.encoded_bytes > options->maximum_host_bytes)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_BUDGET, NULL,
                                options->maximum_host_bytes,
                                residency->summary.encoded_bytes,
                                "resident arena exceeds the configured host budget",
                                YVEX_ERR_NOMEM, err);
    residency->arena = (unsigned char *)mmap(
        NULL, (size_t)residency->summary.encoded_bytes,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (residency->arena == MAP_FAILED) {
        residency->arena = NULL;
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL,
                                residency->summary.encoded_bytes, 0ull,
                                "resident encoded arena allocation failed",
                                YVEX_ERR_NOMEM, err);
    }
    residency->arena_mapped = 1;
    rc = residency_load_and_hash(residency, failure, err);
    if (rc != YVEX_OK) return rc;
    if (mlock(residency->arena, (size_t)residency->summary.encoded_bytes) != 0)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_BUDGET, NULL,
                                residency->summary.encoded_bytes, 0ull,
                                "complete resident arena could not be locked in host RAM",
                                YVEX_ERR_NOMEM, err);
    residency->arena_locked = 1;
    residency->summary.host_locked = 1;
    return YVEX_OK;
}
/*
 * Register the verified locked host arena as one immutable CUDA-addressable range.
 *
 * Registers bytes without copying or identity change.
 */
static int residency_register_cuda(yvex_runtime_residency *residency,
                                   yvex_backend *backend,
                                   yvex_device_tensor **tensor,
                                   yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned char *registered;
    int rc;
    if (!residency || !backend || !tensor || !residency->arena_mapped ||
        !residency->arena_locked || residency->arena_registered ||
        residency->summary.resident_read_calls) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.register",
                       "unconsumed first-use host residency is required");
        return YVEX_ERR_STATE;
    }
    descriptor.name = "runtime-registered-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = residency->summary.encoded_bytes;
    if (!backend->vtable || !backend->vtable->resident_alloc) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "runtime.residency.register",
                       "backend has no CUDA-addressable host placement");
        return YVEX_ERR_UNSUPPORTED;
    }
    registered = residency->arena;
    rc = backend->vtable->resident_alloc(
        backend, &descriptor, tensor, &registered, err);
    if (rc != YVEX_OK) return rc;
    if (registered != residency->arena) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.register",
                       "CUDA registration replaced the verified host address");
        return YVEX_ERR_STATE;
    }
    residency->arena_registered = 1;
    return YVEX_OK;
}
/* Build and attach one exact process-lifetime full-model residency pack. */
int yvex_runtime_residency_prepare(yvex_runtime_residency **out, yvex_runtime_model *model,
                                   const yvex_runtime_residency_options *options,
                                   yvex_runtime_residency_failure *failure, yvex_error *err)
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    yvex_runtime_model_summary model_summary;
    const yvex_runtime_descriptor *descriptor = view ? view->descriptor : NULL;
    const yvex_runtime_descriptor_summary *descriptor_summary =
        yvex_runtime_descriptor_summary_get(descriptor);
    const yvex_attention_plan *plan = view ? view->attention : NULL;
    const yvex_attention_summary *attention = yvex_attention_plan_summary(plan);
    yvex_materialization_session *materialization = view ? view->materialization : NULL;
    yvex_runtime_residency *residency = NULL;
    yvex_materialization_read_provider provider;
    unsigned long long core_qtypes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP] = {0};
    unsigned long long core_bytes = 0ull;
    unsigned long long index, ordinal = 0ull;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!out)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_INVALID_ARGUMENT,
                                NULL, 1ull, 0ull, "residency output is required",
                                YVEX_ERR_INVALID_ARG, err);
    rc = yvex_runtime_model_summary_copy(model, &model_summary, err);
    if (rc != YVEX_OK) {
        if (failure) {
            failure->code = YVEX_RUNTIME_RESIDENCY_FAILURE_LIFECYCLE;
            failure->reason = "runtime model snapshot failed during residency preparation";
        }
        return rc;
    }
    if (!model_summary.sealed || !model_summary.valid ||
        !descriptor_summary || !plan || !attention || !materialization)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_MODEL,
                                NULL, 1ull, 0ull,
                                "sealed runtime model facts are required for residency",
                                YVEX_ERR_STATE, err);
    residency = (yvex_runtime_residency *)calloc(1u, sizeof(*residency));
    if (!residency)
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION,
                                NULL, sizeof(*residency), 0ull,
                                "resident pack allocation failed", YVEX_ERR_NOMEM, err);
    if (pthread_mutex_init(&residency->access_mutex, NULL) != 0) {
        free(residency);
        return residency_reject(
            failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL, 1ull, 0ull,
            "resident access synchronization initialization failed", YVEX_ERR_STATE, err);
    }
    residency->access_mutex_ready = 1;
    residency->materialization = materialization;
    residency->record_index_count = descriptor_summary->tensor_count;
    residency->records = (residency_record *)calloc(
        (size_t)descriptor_summary->tensor_count, sizeof(*residency->records));
    residency->record_index = (unsigned long long *)calloc(
        (size_t)descriptor_summary->tensor_count, sizeof(*residency->record_index));
    if (!residency->records || !residency->record_index) {
        residency_storage_release(&residency);
        return residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION, NULL,
                                descriptor_summary->tensor_count, 0ull,
                                "resident record index allocation failed",
                                YVEX_ERR_NOMEM, err);
    }
    residency->summary.expected_core_binding_count = attention->required_binding_count;
    residency->summary.expected_envelope_binding_count =
        attention->required_envelope_binding_count;
    /* The direct CUDA pack remains a compact prefix while every other tensor is
     * retained after it in the same process-lifetime host arena. */
    for (index = 0ull; rc == YVEX_OK && index < descriptor_summary->tensor_count; ++index) {
        const yvex_runtime_tensor_binding *binding =
            yvex_runtime_descriptor_tensor_at(descriptor, index);
        yvex_attention_binding_class attention_class =
            yvex_attention_plan_binding_classify(plan, binding);
        residency_binding_class binding_class;
        int selected = 1;
        if (attention_class == YVEX_ATTENTION_BINDING_CORE)
            binding_class = RESIDENCY_BINDING_CORE;
        else if (attention_class == YVEX_ATTENTION_BINDING_ENVELOPE)
            binding_class = RESIDENCY_BINDING_ENVELOPE;
        else if (model_summary.capabilities.output_head_binding_ready && binding &&
                 binding->role == YVEX_TENSOR_ROLE_OUTPUT_HEAD &&
                 binding->scope == YVEX_TENSOR_SCOPE_GLOBAL)
            binding_class = RESIDENCY_BINDING_OUTPUT_HEAD;
        else
            selected = 0;
        if (selected)
            rc = residency_add_record(residency, binding, binding_class, ordinal++,
                                      &core_bytes, core_qtypes, failure, err);
    }
    residency->summary.accelerator_encoded_bytes =
        residency->summary.encoded_bytes;
    for (index = 0ull; rc == YVEX_OK && index < descriptor_summary->tensor_count; ++index) {
        const yvex_runtime_tensor_binding *binding =
            yvex_runtime_descriptor_tensor_at(descriptor, index);
        yvex_attention_binding_class attention_class =
            yvex_attention_plan_binding_classify(plan, binding);
        int accelerator_binding =
            attention_class == YVEX_ATTENTION_BINDING_CORE ||
            attention_class == YVEX_ATTENTION_BINDING_ENVELOPE ||
            (model_summary.capabilities.output_head_binding_ready && binding &&
             binding->role == YVEX_TENSOR_ROLE_OUTPUT_HEAD &&
             binding->scope == YVEX_TENSOR_SCOPE_GLOBAL);
        if (!accelerator_binding)
            rc = residency_add_record(
                residency, binding, RESIDENCY_BINDING_MODEL, ordinal++,
                &core_bytes, core_qtypes, failure, err);
    }
    residency->summary.expected_output_head_binding_count =
        model_summary.capabilities.output_head_binding_ready ? 1ull : 0ull;
    residency->summary.expected_model_binding_count =
        descriptor_summary->tensor_count -
        residency->summary.expected_core_binding_count -
        residency->summary.expected_envelope_binding_count -
        residency->summary.expected_output_head_binding_count;
    if (rc == YVEX_OK &&
        (residency->summary.core_binding_count != attention->required_binding_count ||
         residency->summary.envelope_binding_count != attention->required_envelope_binding_count ||
         residency->summary.output_head_binding_count !=
             residency->summary.expected_output_head_binding_count ||
         residency->summary.model_binding_count !=
             residency->summary.expected_model_binding_count ||
         residency->summary.binding_count != descriptor_summary->tensor_count ||
         residency->summary.encoded_bytes != descriptor_summary->payload_bytes ||
         core_bytes != attention->payload_bytes_bound))
        rc = residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                              descriptor_summary->tensor_count,
                              residency->summary.binding_count,
                              "full-model resident accounting differs from the descriptor",
                              YVEX_ERR_FORMAT, err);
    for (index = 0ull; rc == YVEX_OK && index < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP; ++index)
        if (core_qtypes[index] != attention->qtype_binding_counts[index])
            rc = residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN, NULL,
                                  attention->qtype_binding_counts[index], core_qtypes[index],
                                  "resident core qtype accounting differs from the plan",
                                  YVEX_ERR_FORMAT, err);
    if (rc == YVEX_OK) {
        residency->summary.model_complete =
            residency->summary.binding_count == descriptor_summary->tensor_count &&
            residency->summary.model_binding_count ==
                residency->summary.expected_model_binding_count;
        residency->summary.core_complete =
            residency->summary.core_binding_count ==
            residency->summary.expected_core_binding_count;
        residency->summary.envelope_complete =
            residency->summary.envelope_binding_count ==
            residency->summary.expected_envelope_binding_count;
        residency->summary.output_head_complete =
            residency->summary.output_head_binding_count ==
            residency->summary.expected_output_head_binding_count;
    }
    if (rc == YVEX_OK) rc = residency_arena_prepare(residency, options, failure, err);
    if (rc == YVEX_OK)
        rc = residency_identity_build(residency, &model_summary, attention, err);
    if (rc == YVEX_OK) {
        provider.context = residency;
        provider.resolve = residency_resolve;
        provider.note_access = residency_note_access;
        provider.detached = residency_detached;
        rc = yvex_materialization_session_attach_read_provider(
            materialization, &provider, NULL, err);
        if (rc != YVEX_OK)
            rc = residency_reject(failure, YVEX_RUNTIME_RESIDENCY_FAILURE_ATTACH, NULL,
                                  1ull, 0ull, "resident read provider attachment failed",
                                  (yvex_status)rc, err);
    }
    if (rc != YVEX_OK) {
        residency_storage_release(&residency);
        return rc;
    }
    residency->summary.schema_version = YVEX_RUNTIME_RESIDENCY_SCHEMA_V4;
    residency->summary.generation = 1ull;
    residency->summary.host_resident_bytes = residency->summary.encoded_bytes;
    residency->summary.sealed = 1;
    residency->summary.attached = 1;
    residency->summary.host_ready = 1;
    *out = residency;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Register one model-owned host arena and attach an isolated session backend.
 *
 * Session graph/workspace state stays private; model release owns resident bytes/context.
 */
int yvex_runtime_residency_cuda_session_attach(
    yvex_runtime_residency *residency, yvex_backend **backend,
    unsigned long long maximum_device_bytes, int *uploaded,
    yvex_runtime_residency_summary *summary, yvex_error *err)
{
    yvex_backend_options options;
    yvex_backend *candidate_backend = NULL, *session_backend = NULL;
    yvex_device_tensor *candidate_weights = NULL;
    yvex_backend_bandwidth_evidence bandwidth = {0};
    unsigned long long candidate_address = 0ull;
    yvex_error primary, cleanup;
    int rc, cleanup_rc;
    if (backend) *backend = NULL;
    if (uploaded) *uploaded = 0;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!residency || !backend || !uploaded || !summary) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.cuda",
                       "residency and complete CUDA attachment outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!residency->access_mutex_ready) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.cuda",
                       "resident access synchronization is not initialized");
        return YVEX_ERR_STATE;
    }
    if (pthread_mutex_lock(&residency->access_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.cuda",
                       "resident access synchronization could not be acquired");
        return YVEX_ERR_STATE;
    }
    if (!residency->summary.sealed || !residency->summary.host_ready ||
        !residency->summary.host_locked ||
        residency->summary.invalidated || !residency->arena ||
        !residency->summary.accelerator_encoded_bytes) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, rc, "runtime.residency.cuda",
                       "sealed valid host residency is required");
        goto done;
    }
    if ((residency->cuda_backend || residency->cuda_weights) &&
        !residency->summary.cuda_ready) {
        rc = residency_cuda_release(residency, err);
        if (rc != YVEX_OK) goto done;
    }
    if (!residency->summary.cuda_ready) {
        memset(&options, 0, sizeof(options));
        options.kind = YVEX_BACKEND_KIND_CUDA;
        options.memory_limit_bytes = maximum_device_bytes;
        rc = yvex_backend_open(&candidate_backend, &options, err);
        if (rc == YVEX_OK)
            rc = residency_register_cuda(
                residency, candidate_backend, &candidate_weights, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_resident_attach(
                candidate_backend, residency->arena, residency->summary.encoded_bytes,
                candidate_weights, residency->summary.generation, err);
        if (rc == YVEX_OK &&
            yvex_backend_resident_resolve(
                candidate_backend, residency->arena, residency->summary.encoded_bytes,
                &candidate_address) != YVEX_BACKEND_RESIDENT_HIT) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(err, rc, "runtime.residency.cuda",
                           "registered model residency did not resolve");
        }
        if (rc != YVEX_OK) {
            primary = err ? *err : (yvex_error){0};
            yvex_error_clear(&cleanup);
            cleanup_rc = yvex_backend_resident_detach(candidate_backend, &cleanup);
            if (cleanup_rc == YVEX_OK)
                cleanup_rc = candidate_weights
                             ? yvex_backend_tensor_release(
                                   candidate_backend, &candidate_weights, &cleanup)
                             : YVEX_OK;
            if (cleanup_rc == YVEX_OK) residency->arena_registered = 0;
            if (cleanup_rc == YVEX_OK)
                cleanup_rc = yvex_backend_close_checked(&candidate_backend, &cleanup);
            if (cleanup_rc != YVEX_OK) {
                residency->cuda_backend = candidate_backend;
                residency->cuda_weights = candidate_weights;
                rc = cleanup_rc;
                if (err) *err = cleanup;
            } else if (err) *err = primary;
            goto done;
        }
        residency->cuda_backend = candidate_backend;
        residency->cuda_weights = candidate_weights;
        residency->cuda_addressable_device_base = candidate_address;
        residency->summary.host_resident_bytes = residency->summary.encoded_bytes;
        residency->summary.device_resident_bytes = 0ull;
        residency->summary.cuda_addressable_bytes = residency->summary.encoded_bytes;
        residency->summary.cuda_upload_bytes = 0ull;
        residency->summary.cuda_upload_count = 0ull;
        residency->summary.cuda_host_registration_count = 1ull;
        residency->summary.cuda_managed_bytes = 0ull;
        residency->summary.cuda_managed_allocation_count = 0ull;
        residency->summary.cuda_ready = 1;
        *uploaded = 1;
    }
    rc = yvex_backend_bandwidth_probe(
        residency->cuda_backend, &bandwidth, err);
    if (rc != YVEX_OK) goto done;
    rc = yvex_backend_open_shared_cuda(
        &session_backend, residency->cuda_backend, maximum_device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_resident_attach(
            session_backend, residency->arena,
            residency->summary.encoded_bytes,
            residency->cuda_weights, residency->summary.generation, err);
    if (rc != YVEX_OK) {
        primary = err ? *err : (yvex_error){0};
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_close_checked(&session_backend, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            *backend = session_backend;
            rc = cleanup_rc;
            if (err) *err = cleanup;
        } else if (err) *err = primary;
        goto done;
    }
    *backend = session_backend;
    *summary = residency->summary;
    rc = YVEX_OK;
done:
    (void)pthread_mutex_unlock(&residency->access_mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
/* Detach and release one process-lifetime resident full-model arena. */
int yvex_runtime_residency_close(yvex_runtime_residency **residency, yvex_error *err)
{
    return residency_release(residency, err);
}
/*
 * Snapshot synchronized residency facts and optionally borrow its stable host arena.
 *
 * Copies one coherent summary and optional immutable process-lifetime span. The snapshot extends
 * no lifetime and performs no artifact read or qtype decode.
 */
int yvex_runtime_residency_snapshot(const yvex_runtime_residency *residency,
                                    yvex_runtime_residency_summary *summary,
                                    const unsigned char **arena,
                                    unsigned long long *arena_bytes,
                                    yvex_error *err)
{
    yvex_runtime_residency *mutable_residency =
        (yvex_runtime_residency *)residency;
    int borrow_arena = arena || arena_bytes;
    if (arena) *arena = NULL;
    if (arena_bytes) *arena_bytes = 0ull;
    if (!residency || !summary || borrow_arena != (arena && arena_bytes)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.snapshot",
                       "residency, summary, and paired arena outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!residency->access_mutex_ready ||
        pthread_mutex_lock(&mutable_residency->access_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.snapshot",
                       "runtime residency synchronization is unavailable");
        return YVEX_ERR_STATE;
    }
    *summary = residency->summary;
    if (borrow_arena && (!summary->sealed || !summary->host_ready ||
                         summary->invalidated || !residency->arena)) {
        (void)pthread_mutex_unlock(&mutable_residency->access_mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.snapshot",
                       "sealed available host residency is required for arena access");
        return YVEX_ERR_STATE;
    }
    if (borrow_arena) {
        *arena = residency->arena;
        *arena_bytes = summary->encoded_bytes;
    }
    (void)pthread_mutex_unlock(&mutable_residency->access_mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Borrow one exact immutable resident tensor without copying or reopening the artifact.
 *
 * Returns one model-lifetime encoded span and changes no counters. Missing, stale, or invalidated
 * bindings publish no pointer. Callers may decode or execute but never mutate the borrowed
 * resident bytes.
 */
int yvex_runtime_residency_binding_view(
    const yvex_runtime_residency *residency,
    const yvex_materialized_tensor_binding *binding,
    const unsigned char **data, unsigned long long *bytes,
    yvex_error *err)
{
    int resolved;
    if (data) *data = NULL;
    if (bytes) *bytes = 0ull;
    if (!residency || !binding || !data || !bytes) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.binding",
                       "residency, binding, and borrowed-span outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    resolved = residency_resolve(residency, binding, data, bytes);
    if (resolved != YVEX_MATERIALIZATION_READ_HIT) {
        *data = NULL;
        *bytes = 0ull;
        yvex_error_set(err, resolved == YVEX_MATERIALIZATION_READ_INVALID
                               ? YVEX_ERR_STATE : YVEX_ERR_FORMAT,
                       "runtime.residency.binding",
                       "requested tensor is not an available exact resident binding");
        return resolved == YVEX_MATERIALIZATION_READ_INVALID
                   ? YVEX_ERR_STATE : YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Invalidate one shared resident generation without releasing live arena bytes.
 *
 * Process-lifetime residency owned by an invalidated runtime model. Makes every later provider
 * resolve fail closed and advances its generation. Synchronization or generation overflow leaves
 * the pack invalidated.
 */
int yvex_runtime_residency_invalidate(yvex_runtime_residency *residency,
                                      yvex_error *err)
{
    unsigned long long next;
    if (!residency) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.residency.invalidate",
                       "runtime residency is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!residency->access_mutex_ready ||
        getenv("YVEX_TEST_RUNTIME_RESIDENCY_INVALIDATE_FAILURE") ||
        pthread_mutex_lock(&residency->access_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.residency.invalidate",
                       "runtime residency synchronization is unavailable");
        return YVEX_ERR_STATE;
    }
    if (residency->summary.invalidated) {
        (void)pthread_mutex_unlock(&residency->access_mutex);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    residency->summary.invalidated = 1;
    residency->summary.host_ready = 0;
    residency->summary.cuda_ready = 0;
    if (!yvex_core_u64_add(residency->summary.generation, 1ull, &next)) {
        (void)pthread_mutex_unlock(&residency->access_mutex);
        yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.residency.invalidate",
                       "resident generation overflowed");
        return YVEX_ERR_BOUNDS;
    }
    residency->summary.generation = next;
    (void)pthread_mutex_unlock(&residency->access_mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Release every mutable-state residency allocation through exact ownership. */
int yvex_runtime_state_residency_close(
    yvex_runtime_state_residency **owner, yvex_error *err)
{
    yvex_runtime_state_residency *residency = owner ? *owner : NULL;
    yvex_error cleanup;
    unsigned long long index;
    int result = YVEX_OK;
    if (!residency) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    for (index = 0ull; index < residency->layer_count; ++index) {
        state_resident_layer *layer = &residency->layers[index];
        unsigned int bank;
        for (bank = 0u; bank < 2u; ++bank) {
            int rc = layer->device[bank]
                         ? yvex_backend_tensor_release(
                               residency->backend, &layer->device[bank], &cleanup)
                         : YVEX_OK;
            if (result == YVEX_OK && rc != YVEX_OK) {
                result = rc;
                if (err) *err = cleanup;
            }
            if (!layer->device[bank]) {
                free(layer->host[bank]);
                layer->host[bank] = NULL;
            }
        }
    }
    if (result != YVEX_OK) return result;
    free(residency->layers);
    memset(residency, 0, sizeof(*residency));
    free(residency);
    *owner = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Prepare one session-owned persistent-state residency from sealed recipes.
 *
 * Releases all partial ownership and publishes no residency. CPU uses provider-owned host state;
 * CUDA adds no numerical fallback.
 */
int yvex_runtime_state_residency_prepare(
    yvex_runtime_state_residency **out, yvex_backend *backend,
    const yvex_graph_attention_capacity_plan *capacity,
    const yvex_attention_state_provider *provider,
    unsigned long long prior_host_bytes, unsigned long long maximum_host_bytes,
    unsigned long long prior_device_bytes,
    unsigned long long maximum_device_bytes, yvex_error *err)
{
    const yvex_graph_attention_capacity_summary *summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    yvex_runtime_state_residency *residency;
    yvex_graph_attention_state_summary state;
    unsigned long long index, required_bytes = 0ull, host_total, device_total;
    int rc = YVEX_OK;
    if (!out || *out || !backend || !summary || !provider || !provider->view ||
        !provider->summary || !summary->selected_layer_count) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.state.residency.prepare",
                       "backend, capacity, and persistent provider are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = provider->summary(provider->context, &state, err);
    if (rc != YVEX_OK) return rc;
    residency = calloc(1u, sizeof(*residency));
    if (!residency) return YVEX_ERR_NOMEM;
    residency->backend = backend;
    residency->layer_count = summary->layer_count;
    residency->layers = calloc((size_t)residency->layer_count,
                               sizeof(*residency->layers));
    if (!residency->layers) {
        free(residency);
        yvex_error_set(err, YVEX_ERR_NOMEM, "runtime.state.residency.prepare",
                       "persistent state residency allocation failed");
        return YVEX_ERR_NOMEM;
    }
    for (index = 0ull; rc == YVEX_OK && index < residency->layer_count; ++index) {
        const yvex_graph_attention_capacity_layer *layer =
            yvex_graph_attention_capacity_plan_layer(capacity, index);
        state_resident_layer layout;
        unsigned long long banks;
        if (!layer) {
            rc = YVEX_ERR_FORMAT;
            yvex_error_set(err, rc, "runtime.state.residency.prepare",
                           "persistent state capacity layer is unavailable");
        } else if (layer->selected) {
            rc = state_resident_layer_layout(&layout, &layer->recipe, err);
            if (rc == YVEX_OK &&
                (!yvex_core_u64_mul(layout.bytes, 2ull, &banks) ||
                 !yvex_core_u64_add(required_bytes, banks, &required_bytes))) {
                rc = YVEX_ERR_BOUNDS;
                yvex_error_set(err, rc, "runtime.state.residency.prepare",
                               "persistent state residency size overflowed");
            }
        }
    }
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(prior_host_bytes, state.allocated_bytes, &host_total) ||
         (yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA &&
          (!yvex_core_u64_add(prior_device_bytes, required_bytes, &device_total) ||
           !yvex_core_u64_add(host_total, required_bytes, &host_total))))) {
        rc = YVEX_ERR_BOUNDS;
        yvex_error_set(err, rc, "runtime.state.residency.prepare",
                       "persistent state residency budget arithmetic overflowed");
    }
    if (rc == YVEX_OK &&
        ((maximum_host_bytes && host_total > maximum_host_bytes) ||
         (yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA &&
          maximum_device_bytes && device_total > maximum_device_bytes))) {
        rc = YVEX_ERR_BOUNDS;
        yvex_error_set(err, rc, "runtime.state.residency.prepare",
                       "persistent state residency exceeds its session budget");
    }
    for (index = 0ull; rc == YVEX_OK && index < residency->layer_count; ++index) {
        const yvex_graph_attention_capacity_layer *layer =
            yvex_graph_attention_capacity_plan_layer(capacity, index);
        if (!layer) {
            rc = YVEX_ERR_FORMAT;
            yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.state.residency.prepare",
                           "persistent state capacity layer is unavailable");
        } else if (layer->selected) {
            rc = state_resident_layer_prepare(
                residency, &residency->layers[index], index,
                provider, &layer->recipe, err);
            if (rc == YVEX_OK) residency->summary.layer_count++;
        }
    }
    if (rc == YVEX_OK && yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA)
        residency->summary.host_bytes = required_bytes;
    residency->summary.generation = 1ull;
    residency->summary.cuda_ready =
        yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA;
    residency->summary.sealed =
        rc == YVEX_OK && state_resident_identity(
            residency, summary, residency->summary.layout_identity);
    if (rc == YVEX_OK && !residency->summary.sealed) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.prepare",
                       "persistent state residency identity failed");
    }
    if (rc != YVEX_OK) {
        yvex_error primary = *err;
        yvex_error cleanup;
        (void)yvex_runtime_state_residency_close(&residency, &cleanup);
        *err = primary;
        return rc;
    }
    *out = residency;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_state_residency_stage(
    yvex_runtime_state_residency *residency,
    const yvex_attention_state_provider *provider,
    unsigned long long layer_index, yvex_error *err)
{
    const yvex_attention_history_view *view;
    state_resident_layer *layer;
    const yvex_attention_state_recipe *recipe;
    unsigned int bank = 0u;
    int rc;
    if (!residency || !provider || !provider->view ||
        layer_index >= residency->layer_count ||
        !residency->layers[layer_index].selected ||
        residency->summary.invalidated) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.stage",
                       "valid selected persistent state layer is required");
        return YVEX_ERR_STATE;
    }
    layer = &residency->layers[layer_index];
    recipe = &layer->recipe;
    view = provider->view(
        provider->context, layer_index, YVEX_ATTENTION_STATE_VIEW_CANDIDATE);
    if (!view) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.stage",
                       "candidate persistent state view is unavailable");
        return YVEX_ERR_STATE;
    }
    rc = state_resident_bank(layer, view, &bank, err);
    if (rc == YVEX_OK) rc = state_resident_pack(layer, bank, view, recipe, err);
    if (rc != YVEX_OK ||
        state_resident_upload(residency, layer, bank, err) != YVEX_OK)
        return rc == YVEX_OK ? yvex_error_code(err) : rc;
    if (!layer->staged) {
        layer->staged = 1;
        residency->summary.staged_layer_count++;
    }
    return YVEX_OK;
}

int yvex_runtime_state_residency_publish(
    yvex_runtime_state_residency *residency, yvex_error *err)
{
    if (!residency || residency->summary.invalidated ||
        residency->summary.staged_layer_count != residency->summary.layer_count) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.publish",
                       "complete staged persistent state residency is required");
        return YVEX_ERR_STATE;
    }
    /* Graph kernels rebind current state-bank pointers on replay. Publication changes content and
     * generation, not allocation layout, so invalidating the topology registry here would force
     * one capture per committed turn without protecting any pointer lifetime. */
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_runtime_state_residency_commit(yvex_runtime_state_residency *residency)
{
    unsigned long long index;
    if (!residency) return;
    for (index = 0ull; index < residency->layer_count; ++index)
        residency->layers[index].staged = 0;
    residency->summary.staged_layer_count = 0ull;
    residency->summary.commit_count++;
    residency->summary.generation++;
    if (residency->backend)
        residency->backend->state_residency_generation =
            residency->summary.generation;
}

void yvex_runtime_state_residency_abort(yvex_runtime_state_residency *residency)
{
    unsigned long long index;
    if (!residency) return;
    for (index = 0ull; index < residency->layer_count; ++index)
        residency->layers[index].staged = 0;
    residency->summary.staged_layer_count = 0ull;
    residency->summary.abort_count++;
}

int yvex_runtime_state_residency_reset(
    yvex_runtime_state_residency *residency, yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    if (!residency || residency->summary.invalidated)
        return YVEX_ERR_STATE;
    for (index = 0ull; rc == YVEX_OK && index < residency->layer_count; ++index) {
        state_resident_layer *layer = &residency->layers[index];
        unsigned int bank;
        if (!layer->selected) continue;
        for (bank = 0u; rc == YVEX_OK && bank < 2u; ++bank) {
            if (layer->host[bank])
                memset(layer->host[bank], 0, (size_t)layer->bytes);
            if (rc == YVEX_OK)
                rc = state_resident_upload(residency, layer, bank, err);
        }
    }
    if (rc == YVEX_OK) {
        yvex_runtime_state_residency_abort(residency);
        residency->summary.generation++;
        if (residency->backend)
            residency->backend->state_residency_generation =
                residency->summary.generation;
    }
    return rc;
}
/* Fail closed one mutable-state residency and detach device resolution. */
int yvex_runtime_state_residency_invalidate(
    yvex_runtime_state_residency *residency, yvex_error *err)
{
    if (!residency) return YVEX_ERR_INVALID_ARG;
    if (!residency->summary.invalidated) {
        residency->summary.invalidated = 1;
        residency->summary.generation++;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_state_residency_summary_copy(
    const yvex_runtime_state_residency *residency,
    yvex_runtime_state_residency_summary *out, yvex_error *err)
{
    if (!residency || !out) return YVEX_ERR_INVALID_ARG;
    *out = residency->summary;
    yvex_error_clear(err);
    return YVEX_OK;
}
