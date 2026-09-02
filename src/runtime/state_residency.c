/*
 * Keep session-owned persistent state resident across transactional execution.
 *
 * Provider-visible spans retain stable device addresses while committed and candidate banks
 * advance through explicit paging, publication, rollback, reset, and invalidation.
 */
#include "src/runtime/private.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/stateful_attention.h>

typedef struct {
    const void *host[2][3];
    unsigned long long offset[3], bytes[3], visible[2][3], admitted[2][3];
    unsigned long long device_valid[2][3];
} state_resident_component;
typedef struct {
    int selected, begun, staged, staged_replaces_prefix;
    int banks_synchronized, paged, needs_upload[2];
    unsigned long long bytes, page_granularity;
    unsigned int committed_bank, staged_bank;
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
    unsigned long long prepared_generation, prepared_commit_count;
    int paging_selected, commit_prepared;
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
    const yvex_graph_attention_state_summary *state,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_backend_device_info device;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_error err;
    if (!residency || !capacity || !state ||
        yvex_backend_get_device_info(residency->backend, &device, &err) != YVEX_OK)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.persistent-state-residency.v2") ||
        !yvex_sha256_update_text(&hash, capacity->identity) ||
        !yvex_sha256_update_text(&hash, state->state_layout_identity) ||
        !yvex_sha256_update_u64(&hash, yvex_backend_kind_of(residency->backend)) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)device.device_index) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)device.compute_capability_major) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)device.compute_capability_minor) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)residency->summary.paged) ||
        !yvex_sha256_update_u64(&hash, residency->summary.page_granularity) ||
        !yvex_sha256_update_u64(&hash, residency->summary.virtual_device_bytes) ||
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

static int state_resident_history_count(
    const yvex_attention_history_view *view,
    yvex_attention_state_binding binding, unsigned long long *count)
{
    if (!view || !count) return 0;
    switch (binding) {
    case YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY:
        *count = view->local_tail_count;
        return 1;
    case YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY:
        *count = view->compressed_entry_count;
        return 1;
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY:
        *count = view->indexer_entry_count;
        return 1;
    default: return 0;
    }
}

static const yvex_attention_rolling_state_view *state_resident_rolling_source(
    const yvex_attention_history_view *view,
    yvex_attention_state_binding binding)
{
    if (!view) return NULL;
    if (binding == YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING)
        return &view->main_rolling_state;
    if (binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING)
        return &view->indexer_rolling_state;
    return NULL;
}

static int state_resident_source(
    const yvex_attention_history_view *view,
    const yvex_attention_state_component_recipe *component,
    const void *spans[3], unsigned long long visible[3], yvex_error *err)
{
    const yvex_attention_rolling_state_view *rolling;
    unsigned long long count, values;

    memset(spans, 0, 3u * sizeof(*spans));
    memset(visible, 0, 3u * sizeof(*visible));
    if (!view || !component) return 0;
    switch (component->binding) {
    case YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY:
        spans[0] = view->local_kv;
        spans[1] = view->local_positions;
        break;
    case YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY:
        spans[0] = view->compressed_kv;
        spans[1] = view->compressed_positions;
        break;
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY:
        spans[0] = view->indexer_kv;
        spans[1] = view->indexer_positions;
        break;
    case YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING:
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING:
        rolling = state_resident_rolling_source(view, component->binding);
        if (!rolling ||
            rolling->kv_state_extent != component->rolling.kv_state_extent ||
            rolling->score_state_extent != component->rolling.score_state_extent ||
            !yvex_core_u64_mul(rolling->kv_state_extent, sizeof(float), &visible[0]) ||
            !yvex_core_u64_mul(rolling->score_state_extent, sizeof(float), &visible[2])) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.state.residency.pack",
                           "persistent rolling-state extent drifted");
            return 0;
        }
        spans[0] = rolling->kv_state;
        spans[2] = rolling->score_state;
        return 1;
    default: return 0;
    }
    if (component->kind != YVEX_ATTENTION_STATE_COMPONENT_HISTORY ||
        !state_resident_history_count(view, component->binding, &count) ||
        count > component->capacity ||
        !yvex_core_u64_mul(count, component->value_width, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &visible[0]) ||
        !yvex_core_u64_mul(count, sizeof(unsigned long long), &visible[1])) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.state.residency.pack",
                       "persistent history visible extent drifted");
        return 0;
    }
    return 1;
}

static int state_resident_pack(
    state_resident_layer *layer, unsigned int bank,
    const yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe, int materialize, yvex_error *err)
{
    unsigned int index;
    if (!layer || bank > 1u || !view || !recipe ||
        recipe->component_count != layer->component_count) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.state.residency.pack",
                       "persistent state storage inventory is incompatible");
        return YVEX_ERR_FORMAT;
    }
    if (materialize && layer->host[bank])
        memset(layer->host[bank], 0, (size_t)layer->bytes);
    for (index = 0u; index < recipe->component_count; ++index) {
        state_resident_component *target = &layer->components[index];
        const void *pointers[3];
        unsigned long long visible[3];
        unsigned int span;
        if (!state_resident_source(view, &recipe->components[index], pointers,
                                   visible, err)) {
            if (!yvex_error_is_set(err))
                yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.state.residency.pack",
                               "persistent state component binding is invalid");
            return YVEX_ERR_FORMAT;
        }
        for (span = 0u; span < 3u; ++span) {
            if (visible[span] > target->bytes[span] ||
                (target->bytes[span] && !pointers[span])) {
                yvex_error_set(err, YVEX_ERR_FORMAT,
                               "runtime.state.residency.pack",
                               "persistent state component span drifted");
                return YVEX_ERR_FORMAT;
            }
            target->host[bank][span] = pointers[span];
            target->visible[bank][span] = visible[span];
            /* Future provider pages are deliberately inaccessible until admitted. */
            if (materialize && visible[span] && layer->host[bank])
                memcpy(layer->host[bank] + target->offset[span],
                       pointers[span], (size_t)visible[span]);
        }
    }
    return YVEX_OK;
}

static void state_resident_device_valid_clear(
    state_resident_layer *layer, unsigned int bank)
{
    unsigned int component;

    for (component = 0u; component < layer->component_count; ++component)
        memset(layer->components[component].device_valid[bank], 0,
               sizeof(layer->components[component].device_valid[bank]));
}

static void state_resident_device_valid_publish(
    state_resident_layer *layer, unsigned int bank)
{
    unsigned int component;

    for (component = 0u; component < layer->component_count; ++component)
        memcpy(layer->components[component].device_valid[bank],
               layer->components[component].visible[bank],
               sizeof(layer->components[component].device_valid[bank]));
}

static void state_resident_tensor_view(const yvex_device_tensor *tensor,
                                       unsigned long long offset,
                                       unsigned long long bytes,
                                       yvex_device_tensor *view)
{
    *view = *tensor;
    view->rank = 1u;
    memset(view->dims, 0, sizeof(view->dims));
    view->dims[0] = view->bytes = bytes;
    view->data += offset;
}

static int state_resident_commit_range(
    yvex_runtime_state_residency *residency, state_resident_layer *layer,
    unsigned int bank, unsigned long long offset, unsigned long long bytes,
    unsigned long long *admitted, yvex_error *err)
{
    unsigned long long delta = 0ull, device_bytes, page_count;
    int rc;
    if (!bytes) return YVEX_OK;
    if (!layer->paged) {
        if (bytes > *admitted) *admitted = bytes;
        return YVEX_OK;
    }
    rc = yvex_backend_tensor_commit_range(
        residency->backend, layer->device[bank], offset, bytes, &delta, err);
    if (rc != YVEX_OK) return rc;
    if (delta % layer->page_granularity != 0ull ||
        !yvex_core_u64_add(residency->summary.device_bytes, delta, &device_bytes) ||
        !yvex_core_u64_add(residency->summary.page_commit_count,
                           delta / layer->page_granularity, &page_count)) {
        residency->summary.invalidated = 1;
        yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.commit",
                       "persistent state page accounting overflowed");
        return YVEX_ERR_BOUNDS;
    }
    residency->summary.device_bytes = device_bytes;
    residency->summary.page_commit_count = page_count;
    if (bytes > *admitted) *admitted = bytes;
    return YVEX_OK;
}

static int state_resident_upload(yvex_runtime_state_residency *residency,
                                 state_resident_layer *layer, unsigned int bank,
                                 yvex_error *err)
{
    unsigned int component, span;
    int rc = YVEX_OK;
    if (yvex_backend_kind_of(residency->backend) != YVEX_BACKEND_KIND_CUDA)
        return YVEX_OK;
    if (!layer->paged) {
        rc = yvex_backend_tensor_write(residency->backend, layer->device[bank],
                                       layer->host[bank], layer->bytes, err);
        if (rc == YVEX_OK) {
            residency->summary.upload_count++;
            if (!yvex_core_u64_add(residency->summary.upload_bytes, layer->bytes,
                                   &residency->summary.upload_bytes))
                rc = YVEX_ERR_BOUNDS;
        }
        for (component = 0u; component < layer->component_count; ++component)
            for (span = 0u; span < 3u; ++span)
                layer->components[component].admitted[bank][span] =
                    layer->components[component].bytes[span];
    } else {
        for (component = 0u; rc == YVEX_OK && component < layer->component_count;
             ++component) {
            state_resident_component *part = &layer->components[component];
            for (span = 0u; rc == YVEX_OK && span < 3u; ++span) {
                yvex_device_tensor view;
                unsigned long long bytes = part->visible[bank][span];
                if (!bytes) continue;
                rc = state_resident_commit_range(
                    residency, layer, bank, part->offset[span], bytes,
                    &part->admitted[bank][span], err);
                if (rc == YVEX_OK) {
                    state_resident_tensor_view(
                        layer->device[bank], part->offset[span], bytes, &view);
                    rc = yvex_backend_tensor_write(
                        residency->backend, &view, part->host[bank][span], bytes, err);
                }
                if (rc == YVEX_OK &&
                    (!yvex_core_u64_add(residency->summary.upload_bytes, bytes,
                                        &residency->summary.upload_bytes) ||
                     !yvex_core_u64_add(residency->summary.upload_count, 1ull,
                                        &residency->summary.upload_count)))
                    rc = YVEX_ERR_BOUNDS;
            }
        }
    }
    if (rc == YVEX_ERR_BOUNDS && !yvex_error_is_set(err))
        yvex_error_set(err, rc, "runtime.state.residency.upload",
                       "persistent state upload accounting overflowed");
    if (rc == YVEX_OK) {
        state_resident_device_valid_publish(layer, bank);
        layer->needs_upload[bank] = 0;
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
    layer->banks_synchronized = 1;
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
    int cuda = yvex_backend_kind_of(residency->backend) == YVEX_BACKEND_KIND_CUDA;
    unsigned int bank;
    int rc = state_resident_layer_layout(layer, recipe, err);
    if (rc != YVEX_OK) return rc;
    layer->paged = cuda && residency->paging_selected;
    for (bank = 0u; bank < 2u; ++bank) {
        unsigned long long granularity = 0ull;
        if (cuda && !layer->paged)
            layer->host[bank] = calloc(1u, (size_t)layer->bytes);
        view = provider->view(
            provider->context, layer_index,
            bank ? YVEX_ATTENTION_STATE_VIEW_CANDIDATE
                 : YVEX_ATTENTION_STATE_VIEW_COMMITTED);
        rc = view ? YVEX_OK : YVEX_ERR_STATE;
        if ((cuda && !layer->paged && !layer->host[bank]) ||
            rc != YVEX_OK ||
            state_resident_pack(layer, bank, view, recipe,
                                cuda && !layer->paged, err) != YVEX_OK)
        {
            if (rc != YVEX_OK && !yvex_error_is_set(err))
                yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.prepare",
                               "persistent state provider view is unavailable");
            else if (rc == YVEX_OK && !yvex_error_is_set(err))
                yvex_error_set(err, YVEX_ERR_NOMEM, "runtime.state.residency.prepare",
                               "persistent state staging allocation failed");
            return rc == YVEX_OK ? yvex_error_code(err) : rc;
        }
        if (cuda) {
            char name[48];
            (void)snprintf(name, sizeof(name), "persistent-state-%llu-%u",
                           layer_index, bank);
            descriptor.name = name;
            descriptor.dtype = YVEX_DTYPE_I8;
            descriptor.rank = 1u;
            descriptor.dims[0] = layer->bytes;
            descriptor.bytes = layer->bytes;
            rc = layer->paged
                     ? yvex_backend_tensor_reserve(
                           residency->backend, &descriptor, &layer->device[bank],
                           &granularity, err)
                     : yvex_backend_tensor_alloc(
                           residency->backend, &descriptor, &layer->device[bank], err);
            if (rc == YVEX_OK && layer->paged &&
                ((layer->page_granularity &&
                  layer->page_granularity != granularity) ||
                 (residency->summary.page_granularity &&
                  residency->summary.page_granularity != granularity))) {
                yvex_error_set(err, YVEX_ERR_STATE,
                               "runtime.state.residency.prepare",
                               "CUDA state page granularity changed inside one residency");
                rc = YVEX_ERR_STATE;
            }
            if (rc == YVEX_OK && layer->paged) {
                layer->page_granularity = granularity;
                residency->summary.page_granularity = granularity;
                if (!yvex_core_u64_add(residency->summary.virtual_device_bytes,
                                       layer->bytes,
                                       &residency->summary.virtual_device_bytes)) {
                    yvex_error_set(err, YVEX_ERR_BOUNDS,
                                   "runtime.state.residency.prepare",
                                   "persistent state virtual-byte accounting overflowed");
                    rc = YVEX_ERR_BOUNDS;
                }
            }
            if (rc != YVEX_OK ||
                state_resident_upload(residency, layer, bank, err) != YVEX_OK)
                return rc == YVEX_OK ? yvex_error_code(err) : rc;
            if (!layer->paged &&
                (!yvex_core_u64_add(residency->summary.device_bytes, layer->bytes,
                                    &residency->summary.device_bytes) ||
                 !yvex_core_u64_add(residency->summary.host_bytes, layer->bytes,
                                    &residency->summary.host_bytes))) {
                yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.prepare",
                               "persistent state staging-byte accounting overflowed");
                return YVEX_ERR_BOUNDS;
            }
        }
    }
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
                        layer->components[component].admitted[bank][span];
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
    int cuda, rc = YVEX_OK;
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
    cuda = yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA;
    residency->paging_selected =
        cuda && yvex_backend_virtual_tensor_supported(backend);
    residency->summary.paged = residency->paging_selected;
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
    device_total = prior_device_bytes;
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(prior_host_bytes, state.allocated_bytes, &host_total) ||
         (cuda && !residency->paging_selected &&
          (!yvex_core_u64_add(prior_device_bytes, required_bytes, &device_total) ||
           !yvex_core_u64_add(host_total, required_bytes, &host_total))))) {
        rc = YVEX_ERR_BOUNDS;
        yvex_error_set(err, rc, "runtime.state.residency.prepare",
                       "persistent state residency budget arithmetic overflowed");
    }
    if (rc == YVEX_OK &&
        ((maximum_host_bytes && host_total > maximum_host_bytes) ||
         (cuda &&
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
    residency->summary.generation = 1ull;
    residency->summary.cuda_ready = cuda;
    residency->summary.sealed =
        rc == YVEX_OK && state_resident_identity(
            residency, summary, &state, residency->summary.layout_identity);
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

static int state_resident_admit_growth(
    yvex_runtime_state_residency *residency, state_resident_layer *layer,
    unsigned int bank, unsigned long long count, yvex_error *err)
{
    unsigned int component, span;
    for (component = 0u; component < layer->component_count; ++component) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[component];
        state_resident_component *part = &layer->components[component];
        unsigned long long target[3];
        memcpy(target, part->visible[bank], sizeof(target));
        if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
            unsigned long long row_bytes, current, rows;
            if (!yvex_core_u64_mul(recipe->value_width, sizeof(float), &row_bytes) ||
                !row_bytes || part->visible[bank][0] % row_bytes != 0ull ||
                !yvex_core_u64_add(part->visible[bank][0] / row_bytes, count, &rows)) {
                yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.begin",
                               "candidate state page growth overflowed");
                return YVEX_ERR_BOUNDS;
            }
            current = rows > recipe->capacity ? recipe->capacity : rows;
            if (!yvex_core_u64_mul(current, row_bytes, &target[0]) ||
                !yvex_core_u64_mul(current, sizeof(unsigned long long), &target[1])) {
                yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.begin",
                               "candidate state byte extent overflowed");
                return YVEX_ERR_BOUNDS;
            }
        }
        for (span = 0u; span < 3u; ++span) {
            int rc = state_resident_commit_range(
                residency, layer, bank, part->offset[span], target[span],
                &part->admitted[bank][span], err);
            if (rc != YVEX_OK) return rc;
        }
    }
    return YVEX_OK;
}

static int state_resident_copy_bank(
    yvex_runtime_state_residency *residency, state_resident_layer *layer,
    unsigned int target_bank, yvex_error *err)
{
    unsigned int source_bank = layer->committed_bank, component, span;
    for (component = 0u; component < layer->component_count; ++component) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[component];
        state_resident_component *part = &layer->components[component];
        for (span = 0u; span < 3u; ++span) {
            yvex_device_tensor source, target;
            unsigned long long source_bytes = part->visible[source_bank][span];
            unsigned long long source_valid =
                part->device_valid[source_bank][span];
            unsigned long long target_valid =
                part->device_valid[target_bank][span];
            unsigned long long start, bytes;
            int rc;
            if (source_valid < source_bytes) {
                yvex_error_set(err, YVEX_ERR_STATE,
                               "runtime.state.residency.begin",
                               "committed device state is not fully resident");
                return YVEX_ERR_STATE;
            }
            start = recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY &&
                            target_valid <= source_bytes
                        ? target_valid
                        : 0ull;
            bytes = source_bytes - start;
            if (!source_bytes) {
                part->device_valid[target_bank][span] = 0ull;
                continue;
            }
            rc = state_resident_commit_range(
                residency, layer, target_bank, part->offset[span], source_bytes,
                &part->admitted[target_bank][span], err);
            if (rc != YVEX_OK) return rc;
            if (!bytes) continue;
            state_resident_tensor_view(
                layer->device[source_bank], part->offset[span] + start,
                bytes, &source);
            state_resident_tensor_view(
                layer->device[target_bank], part->offset[span] + start,
                bytes, &target);
            rc = yvex_backend_tensor_copy_async(
                residency->backend, &target, &source, err);
            if (rc != YVEX_OK) return rc;
            if (!yvex_core_u64_add(residency->summary.copy_bytes, bytes,
                                   &residency->summary.copy_bytes) ||
                !yvex_core_u64_add(residency->summary.copy_count, 1ull,
                                   &residency->summary.copy_count)) {
                yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.begin",
                               "persistent state copy accounting overflowed");
                return YVEX_ERR_BOUNDS;
            }
            part->device_valid[target_bank][span] = source_bytes;
        }
    }
    layer->needs_upload[target_bank] = 0;
    return YVEX_OK;
}

static int state_residency_begin(
    yvex_runtime_state_residency *residency,
    const yvex_attention_state_provider *provider,
    unsigned long long layer_index, unsigned long long count, yvex_error *err)
{
    yvex_graph_attention_state_summary state;
    const yvex_attention_history_view *view;
    state_resident_layer *layer;
    unsigned int bank;
    int rc;
    if (!residency || !provider || !provider->view || !provider->summary ||
        layer_index >= residency->layer_count ||
        !residency->layers[layer_index].selected ||
        residency->layers[layer_index].begun ||
        residency->layers[layer_index].staged ||
        residency->summary.invalidated) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.begin",
                       "valid selected persistent state layer is required");
        return YVEX_ERR_STATE;
    }
    layer = &residency->layers[layer_index];
    bank = 1u - layer->committed_bank;
    if (layer->needs_upload[layer->committed_bank]) {
        view = provider->view(provider->context, layer_index,
                              YVEX_ATTENTION_STATE_VIEW_COMMITTED);
        rc = view ? state_resident_pack(layer, layer->committed_bank, view,
                                        &layer->recipe, !layer->paged, err)
                  : YVEX_ERR_STATE;
        if (rc == YVEX_OK)
            rc = state_resident_upload(
                residency, layer, layer->committed_bank, err);
        if (rc != YVEX_OK) return rc;
    }
    view = provider->view(
        provider->context, layer_index, YVEX_ATTENTION_STATE_VIEW_CANDIDATE);
    rc = view ? provider->summary(provider->context, &state, err)
              : YVEX_ERR_STATE;
    if (rc == YVEX_OK)
        rc = state_resident_pack(layer, bank, view, &layer->recipe, 0, err);
    if (rc == YVEX_OK && !state.extension_ready && !layer->banks_synchronized &&
        yvex_backend_kind_of(residency->backend) == YVEX_BACKEND_KIND_CUDA) {
        rc = state_resident_copy_bank(residency, layer, bank, err);
        if (rc == YVEX_OK) layer->banks_synchronized = 1;
    }
    if (rc == YVEX_OK && layer->paged)
        rc = state_resident_admit_growth(residency, layer, bank, count, err);
    if (rc == YVEX_OK) {
        layer->needs_upload[bank] = 0;
        layer->begun = 1;
    }
    /* Once admitted, the candidate device bank may change before any host publication occurs. */
    if (rc == YVEX_OK && yvex_backend_kind_of(residency->backend) == YVEX_BACKEND_KIND_CUDA)
        layer->banks_synchronized = 0;
    if (rc != YVEX_OK && !yvex_error_is_set(err))
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.begin",
                       "candidate persistent state view is unavailable");
    return rc;
}

static int state_residency_stage(
    yvex_runtime_state_residency *residency,
    const yvex_attention_state_provider *provider,
    const yvex_attention_publication *publication,
    unsigned long long layer_index, yvex_error *err)
{
    const yvex_attention_history_view *view;
    state_resident_layer *layer;
    const yvex_attention_state_recipe *recipe;
    unsigned int bank;
    unsigned long long next_bytes = 0ull, next_count = 0ull;
    int device_staged;
    int rc;
    if (!residency || !provider || !provider->view ||
        layer_index >= residency->layer_count ||
        !residency->layers[layer_index].selected ||
        (publication && publication->device_state_staged &&
         !residency->layers[layer_index].begun) ||
        residency->summary.invalidated) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.stage",
                       "valid selected persistent state layer is required");
        return YVEX_ERR_STATE;
    }
    layer = &residency->layers[layer_index];
    bank = 1u - layer->committed_bank;
    device_staged = yvex_backend_kind_of(residency->backend) == YVEX_BACKEND_KIND_CUDA &&
                    publication && publication->device_state_staged;
    recipe = &layer->recipe;
    view = provider->view(
        provider->context, layer_index, YVEX_ATTENTION_STATE_VIEW_CANDIDATE);
    if (!view) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.stage",
                       "candidate persistent state view is unavailable");
        return YVEX_ERR_STATE;
    }
    rc = state_resident_pack(layer, bank, view, recipe, !device_staged, err);
    if (rc == YVEX_OK && !device_staged)
        rc = state_resident_upload(residency, layer, bank, err);
    if (rc == YVEX_OK && device_staged && layer->paged) {
        unsigned int component, span;
        for (component = 0u; rc == YVEX_OK && component < layer->component_count;
             ++component)
            for (span = 0u; span < 3u; ++span)
                if (layer->components[component].visible[bank][span] >
                    layer->components[component].admitted[bank][span]) {
                    yvex_error_set(err, YVEX_ERR_BOUNDS,
                                   "runtime.state.residency.stage",
                                   "device state publication exceeded committed pages");
                    rc = YVEX_ERR_BOUNDS;
                    break;
                }
    }
    if (rc == YVEX_OK && device_staged) {
        if (!publication->device_state_staged_bytes ||
            publication->device_state_staged_bytes > layer->bytes) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.stage",
                           "device state publication bytes exceed the admitted bank");
            rc = YVEX_ERR_BOUNDS;
        }
    }
    if (rc == YVEX_OK && device_staged) {
        if (!yvex_core_u64_add(residency->summary.device_stage_bytes,
                               publication->device_state_staged_bytes, &next_bytes) ||
            !yvex_core_u64_add(residency->summary.device_stage_count, 1ull,
                               &next_count)) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.stage",
                           "device state stage accounting overflowed");
            rc = YVEX_ERR_BOUNDS;
        } else {
            residency->summary.device_stage_bytes = next_bytes;
            residency->summary.device_stage_count = next_count;
        }
    }
    if (rc != YVEX_OK) return rc;
    layer->needs_upload[bank] = 0;
    layer->begun = 0;
    layer->staged_bank = bank;
    if (!device_staged) layer->staged_replaces_prefix = 1;
    layer->banks_synchronized = 0;
    if (!layer->staged) {
        layer->staged = 1;
        residency->summary.staged_layer_count++;
    }
    return YVEX_OK;
}

int yvex_runtime_state_residency_candidate_history(
    yvex_runtime_state_residency *residency, unsigned long long layer_index,
    yvex_attention_state_binding binding,
    yvex_runtime_state_history_device_view *out, yvex_error *err)
{
    state_resident_layer *layer;
    state_resident_component *component = NULL;
    const yvex_attention_state_component_recipe *recipe = NULL;
    unsigned long long row_bytes, visible_positions, admitted_positions;
    unsigned int bank, index;

    if (out) memset(out, 0, sizeof(*out));
    if (!residency || !out || residency->summary.invalidated ||
        !residency->summary.cuda_ready || layer_index >= residency->layer_count ||
        !residency->layers[layer_index].selected ||
        !residency->layers[layer_index].begun) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.view",
                       "one active CUDA state candidate is required");
        return YVEX_ERR_STATE;
    }
    layer = &residency->layers[layer_index];
    for (index = 0u; index < layer->component_count; ++index)
        if (layer->recipe.components[index].binding == binding) {
            component = &layer->components[index];
            recipe = &layer->recipe.components[index];
            break;
        }
    bank = 1u - layer->committed_bank;
    if (!component || !recipe ||
        recipe->kind != YVEX_ATTENTION_STATE_COMPONENT_HISTORY ||
        !recipe->value_width ||
        !yvex_core_u64_mul(recipe->value_width, sizeof(float), &row_bytes) ||
        component->visible[bank][0] % row_bytes ||
        component->admitted[bank][0] % row_bytes ||
        component->visible[bank][1] % sizeof(unsigned long long) ||
        component->admitted[bank][1] % sizeof(unsigned long long)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.state.residency.view",
                       "candidate history geometry is incompatible");
        return YVEX_ERR_FORMAT;
    }
    out->value_width = recipe->value_width;
    out->visible_tokens = component->visible[bank][0] / row_bytes;
    out->admitted_tokens = component->admitted[bank][0] / row_bytes;
    visible_positions = component->visible[bank][1] /
                        sizeof(unsigned long long);
    admitted_positions = component->admitted[bank][1] /
                         sizeof(unsigned long long);
    if (visible_positions != out->visible_tokens ||
        admitted_positions < out->admitted_tokens ||
        out->visible_tokens > out->admitted_tokens ||
        component->device_valid[bank][0] < component->visible[bank][0] ||
        component->device_valid[bank][1] < component->visible[bank][1]) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.view",
                       "candidate history prefix is not fully resident");
        return YVEX_ERR_STATE;
    }
    state_resident_tensor_view(
        layer->device[bank], component->offset[0],
        component->admitted[bank][0], &out->values);
    state_resident_tensor_view(
        layer->device[bank], component->offset[1],
        component->admitted[bank][1], &out->positions);
    out->values.dtype = YVEX_DTYPE_F32;
    out->values.rank = 2u;
    out->values.dims[0] = out->admitted_tokens;
    out->values.dims[1] = out->value_width;
    out->values.is_written = component->visible[bank][0] != 0ull;
    out->positions.is_written = component->visible[bank][1] != 0ull;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_state_residency_transition(
    yvex_runtime_state_residency *residency,
    const yvex_attention_state_provider *provider,
    const yvex_attention_publication *publication,
    unsigned long long layer_index, unsigned long long token_count,
    yvex_runtime_state_action action,
    yvex_error *err)
{
    if (action == YVEX_RUNTIME_STATE_BEGIN)
        return state_residency_begin(
            residency, provider, layer_index, token_count, err);
    if (action == YVEX_RUNTIME_STATE_STAGE)
        return state_residency_stage(
            residency, provider, publication, layer_index, err);
    yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.state.residency.transition",
                   "persistent state residency action is invalid");
    return YVEX_ERR_INVALID_ARG;
}

int yvex_runtime_state_residency_prepare_commit(
    yvex_runtime_state_residency *residency, yvex_error *err)
{
    if (!residency || residency->summary.invalidated ||
        residency->commit_prepared ||
        residency->summary.staged_layer_count != residency->summary.layer_count) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.state.residency.prepare",
                       "complete staged persistent state residency is required");
        return YVEX_ERR_STATE;
    }
    if (!yvex_core_u64_add(residency->summary.generation, 1ull,
                           &residency->prepared_generation) ||
        !yvex_core_u64_add(residency->summary.commit_count, 1ull,
                           &residency->prepared_commit_count)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.prepare",
                       "persistent state publication counters overflowed");
        return YVEX_ERR_BOUNDS;
    }
    if (yvex_backend_state_residency_validate_generation(
            residency->backend, residency->prepared_generation, err) != YVEX_OK) {
        residency->prepared_generation = 0ull;
        residency->prepared_commit_count = 0ull;
        return yvex_error_code(err);
    }
    /* Graph kernels rebind current state-bank pointers on replay. Publication changes content and
     * generation, not allocation layout, so invalidating the topology registry here would force
     * one capture per committed turn without protecting any pointer lifetime. */
    residency->commit_prepared = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_runtime_state_residency_publish_commit(
    yvex_runtime_state_residency *residency)
{
    unsigned long long index;
    if (!residency || !residency->commit_prepared) return;
    for (index = 0ull; index < residency->layer_count; ++index) {
        state_resident_layer *layer = &residency->layers[index];
        if (layer->staged) {
            unsigned int prior_bank = layer->committed_bank;
            state_resident_device_valid_publish(layer, layer->staged_bank);
            if (layer->staged_replaces_prefix)
                state_resident_device_valid_clear(layer, prior_bank);
            layer->committed_bank = layer->staged_bank;
        }
        layer->staged_replaces_prefix = 0;
        layer->begun = 0;
        layer->staged = 0;
    }
    residency->summary.staged_layer_count = 0ull;
    residency->summary.commit_count = residency->prepared_commit_count;
    residency->summary.generation = residency->prepared_generation;
    yvex_backend_state_residency_publish_generation(
        residency->backend, residency->summary.generation);
    residency->prepared_generation = 0ull;
    residency->prepared_commit_count = 0ull;
    residency->commit_prepared = 0;
}

void yvex_runtime_state_residency_abort(yvex_runtime_state_residency *residency)
{
    unsigned long long index;
    if (!residency) return;
    for (index = 0ull; index < residency->layer_count; ++index) {
        state_resident_layer *layer = &residency->layers[index];
        if (layer->selected)
            state_resident_device_valid_clear(
                layer, 1u - layer->committed_bank);
        layer->staged_replaces_prefix = 0;
        layer->begun = 0;
        layer->staged = 0;
    }
    residency->summary.staged_layer_count = 0ull;
    residency->prepared_generation = 0ull;
    residency->prepared_commit_count = 0ull;
    residency->commit_prepared = 0;
    residency->summary.abort_count++;
}

int yvex_runtime_state_residency_reset(
    yvex_runtime_state_residency *residency, yvex_error *err)
{
    unsigned long long index, next_generation;
    int rc = YVEX_OK;
    if (!residency || residency->summary.invalidated)
        return YVEX_ERR_STATE;
    if (!yvex_core_u64_add(residency->summary.generation, 1ull,
                           &next_generation)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.state.residency.reset",
                       "persistent state generation overflowed");
        return YVEX_ERR_BOUNDS;
    }
    rc = yvex_backend_state_residency_validate_generation(
        residency->backend, next_generation, err);
    for (index = 0ull; rc == YVEX_OK && index < residency->layer_count; ++index) {
        state_resident_layer *layer = &residency->layers[index];
        unsigned int bank;
        if (!layer->selected) continue;
        for (bank = 0u; rc == YVEX_OK && bank < 2u; ++bank) {
            if (layer->paged) {
                unsigned long long released = 0ull;
                unsigned int component;
                rc = yvex_backend_tensor_decommit(
                    residency->backend, layer->device[bank], &released, err);
                if (rc == YVEX_OK &&
                    (released > residency->summary.device_bytes ||
                     released % layer->page_granularity != 0ull)) {
                    yvex_error_set(err, YVEX_ERR_STATE,
                                   "runtime.state.residency.reset",
                                   "released CUDA state pages do not match residency");
                    rc = YVEX_ERR_STATE;
                }
                if (rc == YVEX_OK) {
                    residency->summary.device_bytes -= released;
                    if (!yvex_core_u64_add(
                            residency->summary.page_release_count,
                            released / layer->page_granularity,
                            &residency->summary.page_release_count)) {
                        yvex_error_set(err, YVEX_ERR_BOUNDS,
                                       "runtime.state.residency.reset",
                                       "persistent state page-release accounting overflowed");
                        rc = YVEX_ERR_BOUNDS;
                    }
                    for (component = 0u; component < layer->component_count;
                         ++component)
                        memset(layer->components[component].admitted[bank], 0,
                               sizeof(layer->components[component].admitted[bank]));
                    layer->needs_upload[bank] = 1;
                }
            } else if (layer->host[bank]) {
                memset(layer->host[bank], 0, (size_t)layer->bytes);
            }
            if (rc == YVEX_OK && !layer->paged)
                rc = state_resident_upload(residency, layer, bank, err);
        }
    }
    if (rc == YVEX_OK) {
        yvex_runtime_state_residency_abort(residency);
        for (index = 0ull; index < residency->layer_count; ++index) {
            residency->layers[index].committed_bank = 0u;
            state_resident_device_valid_clear(&residency->layers[index], 0u);
            state_resident_device_valid_clear(&residency->layers[index], 1u);
            residency->layers[index].banks_synchronized =
                !residency->layers[index].paged;
        }
        residency->summary.generation = next_generation;
        yvex_backend_state_residency_publish_generation(
            residency->backend, residency->summary.generation);
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
