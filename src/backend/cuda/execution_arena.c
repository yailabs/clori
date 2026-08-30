/* Own one admitted CUDA execution arena and stable non-overlapping tensor views. */
#include "src/backend/cuda/private.h"
#include "src/backend/cuda/component_ops.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EXECUTION_ARENA_REGION_CAP 64u

struct yvex_cuda_execution_arena {
    yvex_backend *backend;
    yvex_device_tensor *device_storage;
    yvex_device_tensor *device_views;
    unsigned long long *device_offsets, *device_capacities, *host_offsets;
    unsigned char *host_storage;
    yvex_cuda_execution_arena_summary summary;
};

static int arena_refuse(yvex_error *err, yvex_status status,
                        const char *reason)
{
    yvex_error_set(err, status, "cuda.execution-arena", reason);
    return status;
}

static int arena_align(unsigned long long value, unsigned long long alignment,
                       unsigned long long *aligned)
{
    unsigned long long mask = alignment - 1ull;
    if (!aligned || !alignment || (alignment & mask) || value > UINT64_MAX - mask)
        return 0;
    *aligned = (value + mask) & ~mask;
    return 1;
}

static int arena_device_plan(const yvex_cuda_execution_arena_plan *plan,
                             unsigned long long *offsets,
                             unsigned long long *total)
{
    unsigned int index;
    *total = 0ull;
    for (index = 0u; index < plan->device_count; ++index) {
        const yvex_backend_tensor_desc *descriptor = plan->device + index;
        unsigned long long aligned;
        if (!descriptor->name || !descriptor->name[0] || !descriptor->bytes ||
            descriptor->rank > YVEX_TENSOR_MAX_DIMS ||
            !arena_align(*total, 256ull, &aligned) ||
            descriptor->bytes > UINT64_MAX - aligned)
            return 0;
        offsets[index] = aligned;
        *total = aligned + descriptor->bytes;
    }
    return *total != 0ull;
}

static int arena_host_plan(const yvex_cuda_execution_arena_plan *plan,
                           unsigned long long *offsets,
                           unsigned long long *total)
{
    unsigned int index;
    *total = 0ull;
    for (index = 0u; index < plan->host_count; ++index) {
        unsigned long long aligned;
        if (!plan->host_bytes[index] || !arena_align(*total, 64ull, &aligned) ||
            plan->host_bytes[index] > UINT64_MAX - aligned)
            return 0;
        offsets[index] = aligned;
        *total = aligned + plan->host_bytes[index];
    }
    return *total != 0ull && *total <= SIZE_MAX;
}

static void arena_views_bind(yvex_cuda_execution_arena *arena,
                             const yvex_cuda_execution_arena_plan *plan)
{
    unsigned int index;
    for (index = 0u; index < plan->device_count; ++index) {
        const yvex_backend_tensor_desc *descriptor = plan->device + index;
        yvex_device_tensor *view = arena->device_views + index;
        memset(view, 0, sizeof(*view));
        view->owner = arena->device_storage->owner;
        view->owner_id = arena->device_storage->owner_id;
        view->name = (char *)descriptor->name;
        view->dtype = descriptor->dtype;
        view->rank = descriptor->rank;
        memcpy(view->dims, descriptor->dims, sizeof(view->dims));
        view->bytes = descriptor->bytes;
        view->data = arena->device_storage->data + arena->device_offsets[index];
        arena->device_capacities[index] = descriptor->bytes;
    }
}

int yvex_cuda_execution_arena_open(
    yvex_cuda_execution_arena **out, yvex_backend *backend,
    const yvex_cuda_execution_arena_plan *plan,
    yvex_cuda_execution_arena_summary *summary, yvex_error *err)
{
    yvex_cuda_execution_arena *arena = NULL;
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long device_bytes = 0ull, host_bytes = 0ull;
    int rc;
    if (out) *out = NULL;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!out || !backend || !plan || !summary || !plan->device ||
        !plan->host_bytes || !plan->device_count || !plan->host_count ||
        plan->device_count > EXECUTION_ARENA_REGION_CAP ||
        plan->host_count > EXECUTION_ARENA_REGION_CAP)
        return arena_refuse(err, YVEX_ERR_INVALID_ARG,
                            "one bounded device and host lifetime plan is required");
    arena = calloc(1u, sizeof(*arena));
    if (arena) {
        arena->device_offsets = calloc(plan->device_count, sizeof(*arena->device_offsets));
        arena->device_capacities = calloc(
            plan->device_count, sizeof(*arena->device_capacities));
        arena->device_views = calloc(plan->device_count, sizeof(*arena->device_views));
        arena->host_offsets = calloc(plan->host_count, sizeof(*arena->host_offsets));
    }
    if (!arena || !arena->device_offsets || !arena->device_capacities ||
        !arena->device_views ||
        !arena->host_offsets ||
        !arena_device_plan(plan, arena->device_offsets, &device_bytes) ||
        !arena_host_plan(plan, arena->host_offsets, &host_bytes)) {
        rc = arena_refuse(err, arena ? YVEX_ERR_BOUNDS : YVEX_ERR_NOMEM,
                          arena ? "execution arena lifetime geometry overflowed"
                                : "execution arena metadata allocation failed");
        goto fail;
    }
    arena->host_storage = calloc(1u, (size_t)host_bytes);
    if (!arena->host_storage) {
        rc = arena_refuse(err, YVEX_ERR_NOMEM,
                          "execution host arena allocation failed");
        goto fail;
    }
    descriptor.name = "cuda-execution-arena";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = device_bytes;
    rc = yvex_backend_tensor_alloc(backend, &descriptor,
                                   &arena->device_storage, err);
    if (rc != YVEX_OK) goto fail;
    arena->backend = backend;
    arena->summary.device_bytes = device_bytes;
    arena->summary.host_bytes = host_bytes;
    arena->summary.allocation_count = 2ull;
    arena->summary.device_region_count = plan->device_count;
    arena->summary.host_region_count = plan->host_count;
    arena_views_bind(arena, plan);
    *summary = arena->summary;
    *out = arena;
    yvex_error_clear(err);
    return YVEX_OK;

fail:
    if (arena) {
        free(arena->host_storage);
        free(arena->host_offsets);
        free(arena->device_views);
        free(arena->device_capacities);
        free(arena->device_offsets);
        free(arena);
    }
    return rc;
}

yvex_device_tensor *yvex_cuda_execution_arena_device(
    yvex_cuda_execution_arena *arena, unsigned int index)
{
    return arena && index < arena->summary.device_region_count
               ? arena->device_views + index : NULL;
}

yvex_device_tensor *yvex_cuda_execution_arena_device_bind(
    yvex_cuda_execution_arena *arena, unsigned int index,
    const yvex_backend_tensor_desc *descriptor, yvex_error *err)
{
    yvex_device_tensor *view;
    int preserve_written;
    if (!arena || !descriptor || index >= arena->summary.device_region_count ||
        !descriptor->name || !descriptor->name[0] || !descriptor->bytes ||
        descriptor->bytes > arena->device_capacities[index] ||
        descriptor->rank > YVEX_TENSOR_MAX_DIMS) {
        arena_refuse(err, YVEX_ERR_BOUNDS,
                     "active execution view exceeds its admitted arena region");
        return NULL;
    }
    view = arena->device_views + index;
    preserve_written = view->dtype == descriptor->dtype &&
                       view->rank == descriptor->rank &&
                       view->bytes == descriptor->bytes &&
                       memcmp(view->dims, descriptor->dims,
                              sizeof(view->dims)) == 0;
    view->name = (char *)descriptor->name;
    view->dtype = descriptor->dtype;
    view->rank = descriptor->rank;
    memcpy(view->dims, descriptor->dims, sizeof(view->dims));
    view->bytes = descriptor->bytes;
    if (!preserve_written) view->is_written = 0;
    yvex_error_clear(err);
    return view;
}

void *yvex_cuda_execution_arena_host(
    yvex_cuda_execution_arena *arena, unsigned int index)
{
    return arena && index < arena->summary.host_region_count
               ? arena->host_storage + arena->host_offsets[index] : NULL;
}

int yvex_cuda_execution_arena_close(
    yvex_cuda_execution_arena **arena, yvex_error *err)
{
    yvex_cuda_execution_arena *owned;
    int rc;
    if (!arena || !*arena) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owned = *arena;
    rc = yvex_backend_tensor_release(
        owned->backend, &owned->device_storage, err);
    if (rc != YVEX_OK) return rc;
    free(owned->host_storage);
    free(owned->host_offsets);
    free(owned->device_views);
    free(owned->device_capacities);
    free(owned->device_offsets);
    memset(owned, 0, sizeof(*owned));
    free(owned);
    *arena = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
