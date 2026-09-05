/* Own bounded CUDA work allocations and admitted arenas with stable tensor views. */
#include "src/backend/cuda/private.h"
#include "src/backend/cuda/component_ops.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Initialize one stable work range without breaking active stream capture. */
int yvex_cuda_work_initialize(yvex_cuda_work *work, CUdeviceptr target,
                              size_t bytes, const void *source, int zero,
                              const char *stage, yvex_error *err)
{
    CUstream stream;
    if (!source && !zero) return YVEX_OK;
    if (!work || !work->state || !target || !bytes) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "CUDA range initialization is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    if (work->prepare_only) return YVEX_OK;
    stream = yvex_cuda_launch_stream(work->backend);
    if (stream) {
        if ((source && !work->state->driver.cuMemcpyHtoDAsync_v2) ||
            (zero && !work->state->driver.cuMemsetD8Async)) {
            yvex_error_set(err, YVEX_ERR_UNSUPPORTED, stage,
                           "captured CUDA range initialization is unavailable");
            return YVEX_ERR_UNSUPPORTED;
        }
        return yvex_cuda_status(
            &work->state->driver,
            source ? work->state->driver.cuMemcpyHtoDAsync_v2(
                         target, source, bytes, stream)
                   : work->state->driver.cuMemsetD8Async(target, 0u, bytes, stream),
            stage, err);
    }
    return yvex_cuda_status(
        &work->state->driver,
        source ? work->state->driver.cuMemcpyHtoD_v2(target, source, bytes)
               : work->state->driver.cuMemsetD8_v2(target, 0u, bytes),
        stage, err);
}

/* Own one bounded allocation transaction across reusable and temporary device storage. */
int yvex_cuda_work_allocate(yvex_cuda_work *work,
                            CUdeviceptr *out,
                            size_t bytes,
                            const void *source,
                            int zero,
                            const char *stage,
                            yvex_cuda_work_failure *failure,
                            yvex_error *err)
{
    unsigned long long address = 0ull;
    unsigned long long next;
    int acquired;
    int rc;
    if (failure) *failure = YVEX_CUDA_WORK_FAILURE_NONE;
    if (!work || !work->backend || !work->state || !out || !bytes ||
        work->count >= YVEX_CUDA_WORK_MAX_RANGES) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "CUDA work allocation request is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = backend_dispatch_admit(work->backend, stage, err);
    if (rc != YVEX_OK) {
        if (failure) *failure = YVEX_CUDA_WORK_FAILURE_ALLOCATION;
        return rc;
    }
    if (work->count == 0u) {
        rc = yvex_cuda_deferred_release_drain(work->backend, err);
        if (rc != YVEX_OK) {
            if (failure) *failure = YVEX_CUDA_WORK_FAILURE_ALLOCATION;
            return rc;
        }
    }
    if (work->current_bytes > ULLONG_MAX - (unsigned long long)bytes ||
        (work->budget && work->current_bytes + (unsigned long long)bytes > work->budget)) {
        if (failure) *failure = YVEX_CUDA_WORK_FAILURE_BUDGET;
        yvex_error_set(err, YVEX_ERR_NOMEM, stage, "CUDA work device-byte budget exceeded");
        return YVEX_ERR_NOMEM;
    }
    next = work->current_bytes + (unsigned long long)bytes;
    acquired = work->raw_only ? YVEX_BACKEND_RESIDENT_MISS
                              : yvex_backend_workspace_acquire(
                                    work->backend, bytes, 256ull, &address);
    if (acquired == YVEX_BACKEND_RESIDENT_HIT) {
        *out = (CUdeviceptr)address;
        work->workspace_owned[work->count] = 1u;
    } else if (!work->raw_only && work->backend->workspace_device_tensor) {
        if (failure) *failure = YVEX_CUDA_WORK_FAILURE_BUDGET;
        yvex_error_setf(
            err, acquired == YVEX_BACKEND_RESIDENT_INVALID
                     ? YVEX_ERR_BOUNDS : YVEX_ERR_NOMEM,
            stage,
            "CUDA reusable workspace needs %zu bytes at cursor %llu of %llu",
            bytes, work->backend->workspace_cursor, work->backend->workspace_bytes);
        return acquired == YVEX_BACKEND_RESIDENT_INVALID ? YVEX_ERR_BOUNDS
                                                         : YVEX_ERR_NOMEM;
    } else {
        rc = yvex_backend_memory_can_add(work->backend, bytes, "CUDA", stage, err);
        if (rc != YVEX_OK) {
            if (failure) *failure = YVEX_CUDA_WORK_FAILURE_BUDGET;
            return rc;
        }
        rc = yvex_cuda_status(&work->state->driver,
                              work->state->driver.cuMemAlloc_v2(out, bytes), stage, err);
        if (rc != YVEX_OK) {
            if (failure) *failure = YVEX_CUDA_WORK_FAILURE_ALLOCATION;
            return rc;
        }
        backend_memory_acquire(work->backend, bytes);
    }
    work->pointers[work->count] = *out;
    work->sizes[work->count++] = bytes;
    work->current_bytes = next;
    if (next > work->peak_bytes) work->peak_bytes = next;
    rc = yvex_cuda_work_initialize(work, *out, bytes, source, zero, stage, err);
    if (rc != YVEX_OK && failure) *failure = YVEX_CUDA_WORK_FAILURE_COPY;
    return rc;
}

/* Release an allocation transaction in reverse order while preserving failed ownership. */
int yvex_cuda_work_cleanup(yvex_cuda_work *work, yvex_error *err)
{
    yvex_error cleanup;
    int result = YVEX_OK;
    if (!work) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    while (work->count) {
        unsigned int index = work->count - 1u;
        int rc = work->workspace_owned[index]
                     ? YVEX_OK
                     : yvex_cuda_temporary_free(
                           work->backend, work->variant, &work->pointers[index],
                           work->sizes[index], 1, "cuda.work.cleanup", &cleanup);
        if (!work->workspace_owned[index] && work->pointers[index] != 0u) {
            if (result == YVEX_OK) {
                result = rc;
                if (err) *err = cleanup;
            }
            break;
        }
        work->current_bytes = work->current_bytes >= work->sizes[index]
                                  ? work->current_bytes - work->sizes[index]
                                  : 0ull;
        work->pointers[index] = 0u;
        work->workspace_owned[index] = 0u;
        work->sizes[index] = 0ull;
        work->count = index;
        if (rc != YVEX_OK && result == YVEX_OK) {
            result = rc;
            if (err) *err = cleanup;
        }
    }
    if (result == YVEX_OK) yvex_error_clear(err);
    return result;
}
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
