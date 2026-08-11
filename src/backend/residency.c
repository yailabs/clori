/*
 * Own backend-visible model residency, session state mapping, and reusable workspace lifecycles.
 *
 * Callers receive typed placement operations; concrete backend state and vtable selection remain
 * inside the backend subsystem.
 */
#include <stdint.h>
#include <stdlib.h>

#include "src/backend/private.h"

int yvex_backend_virtual_tensor_supported(const yvex_backend *backend)
{
    return backend && !backend_cleanup_only(backend) &&
           backend->status != YVEX_BACKEND_STATUS_FAILED &&
           backend->virtual_tensor_ready;
}

/*
 * Attach one stable host/device residency mapping to a backend execution context.
 *
 * Identity or lifecycle mismatch refuses.
 */
int yvex_backend_resident_attach(yvex_backend *backend, const unsigned char *host_base,
                                 unsigned long long bytes,
                                 const yvex_device_tensor *device_tensor,
                                 unsigned long long generation, yvex_error *err)
{
    yvex_backend *owner;
    unsigned long long address = 0ull;
    if (!backend || backend_cleanup_only(backend) ||
        backend->kind != YVEX_BACKEND_KIND_CUDA ||
        backend->status == YVEX_BACKEND_STATUS_FAILED || !host_base || !bytes ||
        !device_tensor ||
        (!backend_tensor_owner_is(backend, device_tensor) &&
         backend->resource_owner != device_tensor->owner) ||
        !device_tensor->bytes || !device_tensor->data || !generation) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "backend.residency.attach",
                       "same-owner or exact context-owner CUDA tensor is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (backend->resident_host_base) {
        yvex_error_set(err, YVEX_ERR_STATE, "backend.residency.attach",
                       "backend residency mapping is already attached");
        return YVEX_ERR_STATE;
    }
    owner = backend->resource_owner ? backend->resource_owner : backend;
    if (owner == backend && device_tensor->host_accessible &&
        device_tensor->bytes >= bytes && device_tensor->host_data == host_base) {
        address = (unsigned long long)(uintptr_t)device_tensor->data;
    } else if (owner != backend && owner->resident_host_base == host_base &&
               owner->resident_host_bytes >= bytes &&
               owner->resident_generation == generation && owner->resident_device_address) {
        address = owner->resident_device_address;
    } else {
        yvex_error_set(err, YVEX_ERR_STATE, "backend.residency.attach",
                       "backend has no matching CPU-visible resident arena");
        return YVEX_ERR_STATE;
    }
    backend->resident_host_base = host_base;
    backend->resident_host_bytes = bytes;
    backend->resident_device_tensor = device_tensor;
    backend->resident_device_address = address;
    backend->resident_generation = generation;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_backend_resident_detach(yvex_backend *backend, yvex_error *err)
{
    if (!backend) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    backend->resident_host_base = NULL;
    backend->resident_host_bytes = 0ull;
    backend->resident_device_tensor = NULL;
    backend->resident_device_address = 0ull;
    backend->resident_generation = 0ull;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Translate one borrowed host subrange into its stable resident device address.
 *
 * Returns miss for an unattached/outside range and invalid for arithmetic overflow. Performs no
 * allocation, transfer, or fallback.
 */
int yvex_backend_resident_resolve(const yvex_backend *backend, const unsigned char *host,
                                  unsigned long long bytes,
                                  unsigned long long *device_address)
{
    uintptr_t base, pointer;
    unsigned long long offset;
    if (device_address)
        *device_address = 0ull;
    if (!backend || backend_cleanup_only(backend) ||
        backend->status == YVEX_BACKEND_STATUS_FAILED || !host || !bytes ||
        !device_address || !backend->resident_host_base)
        return YVEX_BACKEND_RESIDENT_MISS;
    base = (uintptr_t)backend->resident_host_base;
    pointer = (uintptr_t)host;
    if (pointer < base)
        return YVEX_BACKEND_RESIDENT_MISS;
    offset = (unsigned long long)(pointer - base);
    if (offset > backend->resident_host_bytes || bytes > backend->resident_host_bytes - offset)
        return YVEX_BACKEND_RESIDENT_MISS;
    if (backend->resident_device_address > ULLONG_MAX - offset)
        return YVEX_BACKEND_RESIDENT_INVALID;
    *device_address = backend->resident_device_address + offset;
    return YVEX_BACKEND_RESIDENT_HIT;
}
/*
 * Attach one session-owned mutable-state resolver to an execution backend.
 *
 * CUDA backend, borrowed resolver/context, and nonzero layout generation. Publishes one borrowed
 * mapping used only for device-resident state reads. The runtime retains all state and device
 * allocation ownership.
 */
int yvex_backend_state_residency_attach(
    yvex_backend *backend, const void *context,
    yvex_backend_state_resolve_fn resolve, unsigned long long generation,
    yvex_error *err)
{
    if (!backend || backend_cleanup_only(backend) ||
        backend->kind != YVEX_BACKEND_KIND_CUDA || !context || !resolve ||
        !generation) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "backend.state.attach",
                       "CUDA backend and complete state resolver are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (backend->state_residency_context) {
        yvex_error_set(err, YVEX_ERR_STATE, "backend.state.attach",
                       "backend state residency is already attached");
        return YVEX_ERR_STATE;
    }
    backend->state_residency_context = context;
    backend->state_residency_resolve = resolve;
    backend->state_residency_generation = generation;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_backend_state_residency_publish_generation(
    yvex_backend *backend, unsigned long long generation, yvex_error *err)
{
    if (!backend || backend_cleanup_only(backend) ||
        backend->status == YVEX_BACKEND_STATUS_FAILED || !generation) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "backend.state.publish",
                       "attached state owner and generation are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (backend->kind != YVEX_BACKEND_KIND_CUDA) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if ((backend->state_residency_context == NULL) !=
        (backend->state_residency_resolve == NULL)) {
        yvex_error_set(err, YVEX_ERR_STATE, "backend.state.publish",
                       "state generation owner is partially attached");
        return YVEX_ERR_STATE;
    }
    if (generation < backend->state_residency_generation) {
        yvex_error_set(err, YVEX_ERR_STATE, "backend.state.publish",
                       "state generation cannot regress");
        return YVEX_ERR_STATE;
    }
    backend->state_residency_generation = generation;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_backend_state_residency_detach(yvex_backend *backend)
{
    if (!backend) return;
    backend->state_residency_context = NULL;
    backend->state_residency_resolve = NULL;
    backend->state_residency_generation = 0ull;
}

int yvex_backend_state_residency_resolve(
    const yvex_backend *backend, const void *host, unsigned long long bytes,
    unsigned long long *device_address)
{
    int result;
    if (device_address) *device_address = 0ull;
    if (!backend || backend_cleanup_only(backend) || !host || !bytes ||
        !device_address || !backend->state_residency_context ||
        !backend->state_residency_resolve)
        return YVEX_BACKEND_RESIDENT_MISS;
    result = backend->state_residency_resolve(
        backend->state_residency_context, host, bytes, device_address);
    return result == YVEX_BACKEND_RESIDENT_HIT && !*device_address
               ? YVEX_BACKEND_RESIDENT_INVALID : result;
}
/*
 * Attach one stable reusable device workspace to a backend session.
 *
 * Rejects wrong ownership, empty extent, or duplicate attachment.
 */
int yvex_backend_workspace_attach(yvex_backend *backend,
                                  const yvex_device_tensor *device_tensor,
                                  unsigned long long generation, yvex_error *err)
{
    if (!backend || backend_cleanup_only(backend) ||
        backend->kind != YVEX_BACKEND_KIND_CUDA ||
        backend->status == YVEX_BACKEND_STATUS_FAILED || !device_tensor ||
        !backend_tensor_owner_is(backend, device_tensor) || !device_tensor->data ||
        !device_tensor->bytes || !generation) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "backend.workspace.attach",
                       "same-owner CUDA workspace tensor and generation are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (backend->workspace_device_tensor) {
        yvex_error_set(err, YVEX_ERR_STATE, "backend.workspace.attach",
                       "backend workspace is already attached");
        return YVEX_ERR_STATE;
    }
    backend->workspace_device_tensor = device_tensor;
    backend->workspace_device_address = (unsigned long long)(uintptr_t)device_tensor->data;
    backend->workspace_bytes = device_tensor->bytes;
    backend->workspace_generation = generation;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_backend_workspace_detach(yvex_backend *backend)
{
    if (!backend)
        return;
    backend->workspace_device_tensor = NULL;
    backend->workspace_device_address = 0ull;
    backend->workspace_bytes = 0ull;
    backend->workspace_cursor = 0ull;
    backend->workspace_peak = 0ull;
    backend->workspace_generation = 0ull;
}
/*
 * Acquire one aligned stable subrange without allocating or resizing the workspace.
 *
 * Attached workspace, byte extent, power-of-two alignment, and address output. Returns miss for no
 * workspace/capacity and invalid for bad alignment or overflow. Caller serializes use; no implicit
 * allocation or execution-mode fallback occurs.
 */
int yvex_backend_workspace_acquire(yvex_backend *backend, unsigned long long bytes,
                                   unsigned long long alignment,
                                   unsigned long long *device_address)
{
    unsigned long long aligned, mask;
    if (device_address)
        *device_address = 0ull;
    if (!backend || backend_cleanup_only(backend) ||
        backend->status == YVEX_BACKEND_STATUS_FAILED || !device_address ||
        !backend->workspace_device_tensor)
        return YVEX_BACKEND_RESIDENT_MISS;
    if (!bytes || !alignment || (alignment & (alignment - 1ull)) != 0ull)
        return YVEX_BACKEND_RESIDENT_INVALID;
    mask = alignment - 1ull;
    if (backend->workspace_cursor > ULLONG_MAX - mask)
        return YVEX_BACKEND_RESIDENT_INVALID;
    aligned = (backend->workspace_cursor + mask) & ~mask;
    if (aligned > backend->workspace_bytes || bytes > backend->workspace_bytes - aligned ||
        backend->workspace_device_address > ULLONG_MAX - aligned)
        return YVEX_BACKEND_RESIDENT_MISS;
    *device_address = backend->workspace_device_address + aligned;
    backend->workspace_cursor = aligned + bytes;
    if (backend->workspace_cursor > backend->workspace_peak)
        backend->workspace_peak = backend->workspace_cursor;
    return YVEX_BACKEND_RESIDENT_HIT;
}
/*
 * Prepare one backend-owned cold arena for standalone execution.
 *
 * Unprepared backend and exact bounded capacity. Allocates once for backend lifetime and records
 * owned cold preparation. Duplicate preparation, overflow, or allocation failure preserves prior
 * state. Runtime sessions attach their own arena and never call this fallback.
 */
int yvex_backend_host_workspace_prepare_owned(yvex_backend *backend,
                                              unsigned long long bytes,
                                              yvex_error *err)
{
    unsigned char *base;
    int rc;
    if (!backend || backend_cleanup_only(backend) ||
        backend->status == YVEX_BACKEND_STATUS_FAILED || !bytes ||
        bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "backend.host_workspace.prepare",
                       "owned host workspace capacity is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    if (backend->host_workspace_base) {
        yvex_error_set(err, YVEX_ERR_STATE, "backend.host_workspace.prepare",
                       "backend host workspace was already prepared");
        return YVEX_ERR_STATE;
    }
    if (backend->vtable && backend->vtable->host_workspace_alloc)
        rc = backend->vtable->host_workspace_alloc(backend, (size_t)bytes, &base, err);
    else {
        base = (unsigned char *)malloc((size_t)bytes);
        rc = base ? YVEX_OK : YVEX_ERR_NOMEM;
    }
    if (!base) {
        if (rc == YVEX_OK)
            yvex_error_set(err, YVEX_ERR_NOMEM, "backend.host_workspace.prepare",
                           "failed to allocate standalone host workspace");
        return rc == YVEX_OK ? YVEX_ERR_NOMEM : rc;
    }
    backend->host_workspace_base = base;
    backend->host_workspace_bytes = bytes;
    backend->host_workspace_generation = 1ull;
    backend->host_workspace_allocation_count = 1ull;
    backend->host_workspace_owned = 1;
    backend->host_workspace_pinned = backend->vtable &&
                                     backend->vtable->host_workspace_free;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_backend_host_workspace_detach(yvex_backend *backend, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!backend) goto done;
    if (backend->host_workspace_owned && backend->host_workspace_pinned &&
        backend->vtable && backend->vtable->host_workspace_free) {
        rc = backend->vtable->host_workspace_free(
            backend, &backend->host_workspace_base, err);
    } else if (backend->host_workspace_owned) {
        free(backend->host_workspace_base);
        backend->host_workspace_base = NULL;
    }
    if (rc != YVEX_OK && backend->host_workspace_base)
        return rc;
    backend->host_workspace_base = NULL;
    backend->host_workspace_bytes = 0ull;
    backend->host_workspace_reserved = 0ull;
    backend->host_workspace_cursor = 0ull;
    backend->host_workspace_peak = 0ull;
    backend->host_workspace_generation = 0ull;
    backend->host_workspace_allocation_count = 0ull;
    backend->host_workspace_owned = 0;
    backend->host_workspace_pinned = 0;
done:
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
/*
 * Acquire one aligned stable host subrange without allocating or resizing.
 *
 * Prepared arena, exact bytes, power-of-two alignment, and output address. No implicit
 * preparation, growth, or execution-mode fallback occurs.
 */
int yvex_backend_host_workspace_acquire(yvex_backend *backend,
                                        unsigned long long bytes,
                                        unsigned long long alignment,
                                        void **address)
{
    uintptr_t base, current, aligned;
    unsigned long long offset, mask;
    if (address)
        *address = NULL;
    if (!backend || backend_cleanup_only(backend) ||
        backend->status == YVEX_BACKEND_STATUS_FAILED || !address ||
        !backend->host_workspace_base)
        return YVEX_BACKEND_RESIDENT_MISS;
    if (!bytes || !alignment || (alignment & (alignment - 1ull)) != 0ull)
        return YVEX_BACKEND_RESIDENT_INVALID;
    mask = alignment - 1ull;
    base = (uintptr_t)backend->host_workspace_base;
    if (backend->host_workspace_cursor > UINTPTR_MAX - base ||
        base + backend->host_workspace_cursor > UINTPTR_MAX - mask)
        return YVEX_BACKEND_RESIDENT_INVALID;
    current = base + backend->host_workspace_cursor;
    aligned = (current + (uintptr_t)mask) & ~(uintptr_t)mask;
    offset = (unsigned long long)(aligned - base);
    if (offset > backend->host_workspace_bytes ||
        bytes > backend->host_workspace_bytes - offset)
        return YVEX_BACKEND_RESIDENT_MISS;
    *address = (void *)aligned;
    backend->host_workspace_cursor = offset + bytes;
    if (backend->host_workspace_cursor > backend->host_workspace_peak)
        backend->host_workspace_peak = backend->host_workspace_cursor;
    return YVEX_BACKEND_RESIDENT_HIT;
}

/*
 * Seal one aligned prefix for completion records that outlive transient phase
 * acquisitions. Resets preserve this prefix and repeated binding must name the
 * same extent.
 */
int yvex_backend_host_workspace_reserve(yvex_backend *backend,
                                        unsigned long long bytes,
                                        unsigned long long alignment,
                                        void **address)
{
    unsigned long long prior_peak;
    int rc;
    if (address) *address = NULL;
    if (!backend || !address)
        return YVEX_BACKEND_RESIDENT_INVALID;
    if (backend->host_workspace_reserved) {
        if (bytes != backend->host_workspace_reserved)
            return YVEX_BACKEND_RESIDENT_MISS;
        *address = backend->host_workspace_base;
        return YVEX_BACKEND_RESIDENT_HIT;
    }
    if (backend->host_workspace_cursor)
        return YVEX_BACKEND_RESIDENT_MISS;
    prior_peak = backend->host_workspace_peak;
    rc = yvex_backend_host_workspace_acquire(backend, bytes, alignment, address);
    if (rc == YVEX_BACKEND_RESIDENT_HIT && *address == backend->host_workspace_base)
        backend->host_workspace_reserved = bytes;
    else if (rc == YVEX_BACKEND_RESIDENT_HIT) {
        backend->host_workspace_cursor = 0ull;
        backend->host_workspace_peak = prior_peak;
        *address = NULL;
        rc = YVEX_BACKEND_RESIDENT_INVALID;
    }
    return rc;
}

int yvex_backend_host_workspace_summary_get(
    const yvex_backend *backend, yvex_backend_host_workspace_summary *summary)
{
    if (!backend || !summary)
        return 0;
    *summary = (yvex_backend_host_workspace_summary){
        backend->host_workspace_base != NULL, backend->host_workspace_owned,
        backend->host_workspace_pinned,
        backend->host_workspace_bytes, backend->host_workspace_cursor,
        backend->host_workspace_peak, backend->host_workspace_generation,
        backend->host_workspace_allocation_count};
    return 1;
}
