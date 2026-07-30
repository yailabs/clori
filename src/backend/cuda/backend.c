/* Owner: src/backend/cuda.
 * Owns: Driver API discovery, device/context lifetime, backend status, vtable attachment, and coordinated
 *   context/module close.
 * Does not own: generated bundle contents, symbol policy, op geometry, CLI output, graph semantics, qtype compute,
 *   or runtime generation.
 * Invariants: context creation yields context-ready only; ready requires atomic canonical bundle admission; close
 *   clears every owned Driver API handle.
 * Boundary: an open CUDA context is not primitive or model runtime support.
 * Purpose: Construct and release the dynamically admitted CUDA backend context.
 * Inputs: Driver discovery results, device selection, and caller-owned backend result storage.
 * Effects: Creates or tears down only CUDA Driver resources owned by the backend.
 * Failure: Returns typed CUDA admission or cleanup failures without publishing partial readiness. */
#include "src/backend/cuda/private.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Purpose: Translate operator input into the canonical typed parse device index value without ambiguous aliases.
 * Inputs: Typed caller-owned outputs and immutable values declared by this subsystem ABI.
 * Effects: Updates only caller-owned result storage or lifecycle state explicitly named by the ABI.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
static int parse_device_index(const char *text, int *out, yvex_error *err)
{
    long value = 0;
    const char *p;
    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.parse_device", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = 0;
    if (!text || text[0] == '\0') {
        return YVEX_OK;
    }
    for (p = text; *p; ++p) {
        if (!isdigit((unsigned char)*p)) {
            yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "cuda.parse_device",
                            "CUDA device must be numeric: %s", text);
            return YVEX_ERR_INVALID_ARG;
        }
        value = (value * 10) + (*p - '0');
        if (value > 65535) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.parse_device",
                           "CUDA device index is too large");
            return YVEX_ERR_INVALID_ARG;
        }
    }
    *out = (int)value;
    return YVEX_OK;
}
/* Purpose: report whether deterministic timing fault injection selects one event operation. */
static int cuda_timing_failure_matches(const char *stage)
{
    const char *selected = getenv("YVEX_TEST_CUDA_EVENT_FAILURE");
    return selected && stage && strcmp(selected, stage) == 0;
}
/* Purpose: create one reusable event pair before a backend enters warm execution.
 * Inputs: admitted live backend.
 * Effects: installs context-owned start and stop events.
 * Failure: Driver or injected failure leaves timing unavailable.
 * Boundary: allocates no per-replay event. */
static int cuda_timing_open(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_driver *driver;
    int rc;
    if (!state) return YVEX_ERR_INVALID_ARG;
    driver = &state->driver;
    if (!driver->cuEventCreate || !driver->cuEventRecord ||
        !driver->cuEventSynchronize || !driver->cuEventElapsedTime_v2 ||
        !driver->cuEventDestroy_v2) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_cuda_set_current(backend, "cuda.timing.open", err);
    if (rc != YVEX_OK) return rc;
    if (cuda_timing_failure_matches("create-start"))
        rc = YVEX_ERR_BACKEND;
    else
        rc = yvex_cuda_status(driver, driver->cuEventCreate(&state->timing_start, 0u),
                              "cuda.timing.create_start", err);
    if (rc == YVEX_OK) {
        if (cuda_timing_failure_matches("create-stop"))
            rc = YVEX_ERR_BACKEND;
        else
            rc = yvex_cuda_status(driver, driver->cuEventCreate(&state->timing_stop, 0u),
                                  "cuda.timing.create_stop", err);
    }
    if (rc != YVEX_OK) {
        if (err && !err->message[0])
            yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.timing.open",
                           "injected CUDA timing event creation failure");
        return rc;
    }
    state->timing_ready = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: release the reusable event pair before its CUDA context becomes invalid.
 * Inputs: live backend context.
 * Effects: destroys only its owned events.
 * Failure: preserves remaining event ownership for checked retry.
 * Boundary: precedes context teardown. */
static int cuda_timing_close(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_driver *driver;
    int rc;
    if (!state || (!state->timing_start && !state->timing_stop)) return YVEX_OK;
    driver = &state->driver;
    rc = yvex_cuda_set_current(backend, "cuda.timing.close", err);
    if (rc != YVEX_OK) return rc;
    if (state->timing_stop) {
        if (cuda_timing_failure_matches("destroy-stop")) {
            yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.timing.destroy_stop",
                           "injected CUDA timing event cleanup failure");
            return YVEX_ERR_BACKEND;
        }
        rc = yvex_cuda_status(driver, driver->cuEventDestroy_v2(state->timing_stop),
                              "cuda.timing.destroy_stop", err);
        if (rc != YVEX_OK) return rc;
        state->timing_stop = NULL;
    }
    if (state->timing_start) {
        if (cuda_timing_failure_matches("destroy-start")) {
            yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.timing.destroy_start",
                           "injected CUDA timing event cleanup failure");
            return YVEX_ERR_BACKEND;
        }
        rc = yvex_cuda_status(driver, driver->cuEventDestroy_v2(state->timing_start),
                              "cuda.timing.destroy_start", err);
        if (rc != YVEX_OK) return rc;
        state->timing_start = NULL;
    }
    state->timing_ready = 0;
    state->timing_active = 0;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: begin, finish, or discard one interval through the reusable CUDA event pair.
 * Inputs: live backend, action, exact stream, optional elapsed output, and diagnostic stage.
 * Effects: changes only timing ownership and publishes device elapsed nanoseconds on finish.
 * Failure: invalid lifecycle, Driver, or range refusal leaves no published elapsed value.
 * Boundary: timing neither launches numerical work nor owns execution synchronization policy. */
int yvex_cuda_timing(yvex_backend *backend, CUstream stream,
                     yvex_cuda_timing_action action, unsigned long long *elapsed_ns,
                     const char *where, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    float milliseconds = 0.0f;
    double nanoseconds;
    int rc;
    if (elapsed_ns) *elapsed_ns = 0ull;
    if (!state || action > YVEX_CUDA_TIMING_DISCARD) return YVEX_ERR_INVALID_ARG;
    if (action == YVEX_CUDA_TIMING_DISCARD) {
        state->timing_active = 0;
        return YVEX_OK;
    }
    if (!where || (action == YVEX_CUDA_TIMING_FINISH && !elapsed_ns))
        return YVEX_ERR_INVALID_ARG;
    if (action == YVEX_CUDA_TIMING_BEGIN) {
        if (!state->timing_ready) return YVEX_OK;
        if (state->timing_active) {
            yvex_error_set(err, YVEX_ERR_STATE, where,
                           "CUDA timing event pair is already active");
            return YVEX_ERR_STATE;
        }
        if (cuda_timing_failure_matches("record-start")) {
            yvex_error_set(err, YVEX_ERR_BACKEND, where,
                           "injected CUDA timing start-record failure");
            return YVEX_ERR_BACKEND;
        }
        rc = yvex_cuda_status(&state->driver,
            state->driver.cuEventRecord(state->timing_start, stream), where, err);
        if (rc == YVEX_OK) state->timing_active = 1;
        return rc;
    }
    if (!state->timing_ready || !state->timing_active) return YVEX_OK;
    if (cuda_timing_failure_matches("record-stop")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "injected CUDA timing stop-record failure");
        rc = YVEX_ERR_BACKEND;
    } else {
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuEventRecord(state->timing_stop, stream), where, err);
    }
    if (rc == YVEX_OK && cuda_timing_failure_matches("synchronize")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "injected CUDA timing synchronization failure");
        rc = YVEX_ERR_BACKEND;
    } else if (rc == YVEX_OK) {
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuEventSynchronize(state->timing_stop), where, err);
    }
    if (rc == YVEX_OK && cuda_timing_failure_matches("elapsed")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "injected CUDA elapsed-time query failure");
        rc = YVEX_ERR_BACKEND;
    } else if (rc == YVEX_OK) {
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuEventElapsedTime_v2(
                &milliseconds, state->timing_start, state->timing_stop), where, err);
    }
    state->timing_active = 0;
    nanoseconds = (double)milliseconds * 1000000.0;
    if (rc == YVEX_OK && (!(nanoseconds >= 0.0) || nanoseconds > (double)ULLONG_MAX)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, where,
                       "CUDA event elapsed time is not representable");
        return YVEX_ERR_BOUNDS;
    }
    if (rc == YVEX_OK) *elapsed_ns = (unsigned long long)(nanoseconds + 0.5);
    return rc;
}
/* Purpose: discharge CUDA ownership in dependency order for checked backend close.
 * Inputs: an exclusively owned backend whose context remains live until every child is released.
 * Effects: releases deferred allocations, graphs, module, context, Driver, and implementation state.
 * Failure: the first pre-release failure preserves the remaining backend owner for retry.
 * Boundary: outer backend storage is released only by the generic checked-close owner. */
static int cuda_close(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_driver *driver;
    int rc;
    if (!backend || !state) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    driver = &state->driver;
    rc = yvex_cuda_deferred_release_drain(backend, err);
    if (rc != YVEX_OK)
        return rc;
    rc = yvex_cuda_graphs_close_all(backend, err);
    if (rc != YVEX_OK)
        return rc;
    rc = cuda_timing_close(backend, err);
    if (rc != YVEX_OK)
        return rc;
    rc = yvex_cuda_kernel_bundle_close(backend, err);
    if (rc != YVEX_OK)
        return rc;
    if (state->registered_host) {
        if (!driver->cuMemHostUnregister) {
            yvex_error_set(err, YVEX_ERR_STATE, "cuda.residency.unregister",
                           "CUDA host registration release is unavailable");
            return YVEX_ERR_STATE;
        }
        rc = yvex_cuda_status(driver,
                              driver->cuMemHostUnregister(state->registered_host),
                              "cuda.residency.unregister", err);
        if (rc != YVEX_OK) return rc;
        state->registered_host = NULL;
        state->registered_device = 0ull;
        state->registered_bytes = 0ull;
    }
    if (state->context && !state->context_borrowed) {
        if (!driver->cuCtxDestroy_v2) {
            yvex_error_set(err, YVEX_ERR_STATE, "cuda.context.destroy",
                           "CUDA context destroy function is unavailable");
            return YVEX_ERR_STATE;
        }
        rc = yvex_cuda_status(driver, driver->cuCtxDestroy_v2(state->context),
                              "cuda.context.destroy", err);
        if (rc != YVEX_OK)
            return rc;
        state->context = NULL;
    }
    if (state->context_borrowed)
        state->context = NULL;
    else
        yvex_cuda_driver_unload(driver);
    state->context_owner = NULL;
    free(state);
    backend->impl = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: Implement the canonical memory stats mechanism owned by the CUDA backend boundary. */
static int cuda_memory_stats(const yvex_backend *backend,
                             yvex_backend_memory_stats *out,
                             yvex_error *err)
{
    if (!backend || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.memory_stats",
                       "backend and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = backend->stats;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: Implement the canonical device info mechanism owned by the CUDA backend boundary. */
static int cuda_device_info(const yvex_backend *backend,
                            yvex_backend_device_info *out,
                            yvex_error *err)
{
    int rc;
    if (!backend || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.device_info",
                       "backend and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_cuda_refresh_memory_info((yvex_backend *)backend, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    *out = backend->device_info;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: Implement the canonical sync mechanism owned by the CUDA backend boundary. */
static int cuda_sync(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc;
    if (!backend || !state) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.sync", "backend is required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_cuda_set_current(backend, "cuda.sync", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    return yvex_cuda_status(&state->driver, state->driver.cuCtxSynchronize(), "cuda.sync", err);
}
/* Purpose: allocate one page-locked session staging arena for captured transfers.
 * Inputs: live CUDA backend, nonzero byte extent, and caller-owned output.
 * Effects: owns one Driver allocation until the matching backend callback releases it.
 * Failure: missing pinned-memory API or Driver failure publishes no pointer.
 * Boundary: supplies stable transfer storage only; graph capture owns no host allocation. */
static int cuda_host_workspace_alloc(yvex_backend *backend, size_t bytes,
                                     unsigned char **out, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc;
    if (out) *out = NULL;
    if (!state || !out || !bytes || !state->driver.cuMemHostAlloc) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.host_workspace.alloc",
                       "CUDA page-locked host allocation is unavailable");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = yvex_cuda_set_current(backend, "cuda.host_workspace.alloc", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuMemHostAlloc((void **)out, bytes, 0u),
                              "cuda.host_workspace.alloc", err);
    return rc;
}
/* Purpose: release only a page-locked staging arena owned by this CUDA backend.
 * Inputs: live CUDA backend and its owned page-locked base; null is idempotent.
 * Effects: returns the exact Driver allocation and owns no caller-provided storage.
 * Failure: missing API or Driver failure leaves cleanup explicitly failed.
 * Boundary: does not release device residency, workspace, or graph resources. */
static int cuda_host_workspace_free(yvex_backend *backend, unsigned char **base,
                                    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    const char *injected = getenv("YVEX_TEST_CUDA_CLEANUP_FAILURE");
    int rc;
    if (!base || !*base) return YVEX_OK;
    if (!state || !state->driver.cuMemFreeHost) {
        yvex_error_set(err, YVEX_ERR_STATE, "cuda.host_workspace.free",
                       "CUDA page-locked host release is unavailable");
        return YVEX_ERR_STATE;
    }
    if (injected && strcmp(injected, "host-workspace-pre-release") == 0) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.host_workspace.free",
                       "injected CUDA page-locked host pre-release failure");
        return YVEX_ERR_BACKEND;
    }
    rc = yvex_cuda_set_current(backend, "cuda.host_workspace.free", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver, state->driver.cuMemFreeHost(*base),
                              "cuda.host_workspace.free", err);
    if (rc == YVEX_OK)
        *base = NULL;
    if (rc == YVEX_OK && injected && strcmp(injected, "host-workspace") == 0) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.host_workspace.free",
                       "injected CUDA page-locked host cleanup failure");
        return YVEX_ERR_BACKEND;
    }
    return rc;
}
/* Purpose: make one immutable arena CUDA-addressable without duplicating imported host bytes.
 * Inputs: descriptor and optional verified host storage. Effects: imports or allocates CPU-visible CUDA storage.
 * Failure: registration/allocation refusal publishes no tensor. Boundary: runtime alone seals imported bytes. */
static int cuda_resident_alloc(yvex_backend *backend, const yvex_backend_tensor_desc *desc,
                               yvex_device_tensor **out, unsigned char **host, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_device_tensor *tensor = NULL;
    unsigned char *imported = host ? *host : NULL;
    CUdeviceptr pointer = 0ull;
    unsigned int index;
    int rc;
    if (out) *out = NULL;
    if (!backend || !state || !desc || !out || !host || !desc->name || desc->rank == 0u ||
        desc->rank > YVEX_TENSOR_MAX_DIMS || desc->bytes == 0ull ||
        desc->bytes > (unsigned long long)SIZE_MAX || state->context_borrowed ||
        !backend->device_info.unified_addressing ||
        (imported && (!state->driver.cuMemHostRegister_v2 ||
                      !state->driver.cuMemHostGetDevicePointer_v2 ||
                      !state->driver.cuMemHostUnregister || state->registered_host)) ||
        (!imported && (!backend->device_info.managed_memory ||
                       !state->driver.cuMemAllocManaged))) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.residency.alloc",
                       "one valid imported-host or managed CUDA residency is required");
        return YVEX_ERR_UNSUPPORTED;
    }
    for (index = 0u; index < desc->rank; ++index) {
        if (desc->dims[index] == 0ull) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.residency.alloc",
                           "managed allocation dimensions must be non-zero");
            return YVEX_ERR_INVALID_ARG;
        }
    }
    if (!imported) {
        rc = yvex_backend_memory_can_add(
            backend, desc->bytes, "CUDA managed", "cuda.residency.alloc", err);
        if (rc != YVEX_OK) return rc;
    }
    tensor = (yvex_device_tensor *)calloc(1u, sizeof(*tensor));
    if (tensor) tensor->name = yvex_core_strdup(desc->name);
    if (!tensor || !tensor->name) {
        free(tensor);
        yvex_error_set(err, YVEX_ERR_NOMEM, "cuda.residency.alloc",
                       "managed tensor metadata allocation failed");
        return YVEX_ERR_NOMEM;
    }
    rc = yvex_cuda_set_current(backend, "cuda.residency.alloc", err);
    if (rc == YVEX_OK && imported && getenv("YVEX_TEST_CUDA_HOST_REGISTER_FAILURE")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.residency.register",
                       "injected CUDA host registration failure");
        rc = YVEX_ERR_BACKEND;
    } else if (rc == YVEX_OK && !imported &&
               getenv("YVEX_TEST_CUDA_MANAGED_ALLOC_FAILURE")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.residency.alloc",
                       "injected managed allocation failure");
        rc = YVEX_ERR_BACKEND;
    }
    if (rc == YVEX_OK && imported) {
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemHostRegister_v2(
                imported, (size_t)desc->bytes, YVEX_CUDA_MEMHOSTREGISTER_DEVICEMAP),
            "cuda.residency.register", err);
        if (rc == YVEX_OK) {
            state->registered_host = imported;
            state->registered_bytes = desc->bytes;
            rc = yvex_cuda_status(
                &state->driver,
                state->driver.cuMemHostGetDevicePointer_v2(&pointer, imported, 0u),
                "cuda.residency.address", err);
            if (rc == YVEX_OK) state->registered_device = pointer;
        }
    } else if (rc == YVEX_OK) {
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemAllocManaged(
                &pointer, (size_t)desc->bytes, YVEX_CUDA_MEM_ATTACH_GLOBAL),
            "cuda.residency.alloc", err);
    }
    if (rc != YVEX_OK) {
        if (state->registered_host && state->driver.cuMemHostUnregister &&
            state->driver.cuMemHostUnregister(state->registered_host) == YVEX_CUDA_SUCCESS) {
            state->registered_host = NULL;
            state->registered_device = 0ull;
            state->registered_bytes = 0ull;
        }
        free(tensor->name);
        free(tensor);
        return rc;
    }
    tensor->owner = backend;
    tensor->owner_id = backend->tensor_id_next++;
    tensor->dtype = desc->dtype;
    tensor->rank = desc->rank;
    for (index = 0u; index < desc->rank; ++index) tensor->dims[index] = desc->dims[index];
    tensor->bytes = desc->bytes;
    tensor->data = (unsigned char *)(uintptr_t)pointer;
    tensor->host_data = imported ? imported : tensor->data;
    tensor->host_accessible = 1;
    tensor->is_written = imported != NULL;
    if (!imported) backend_memory_acquire(backend, desc->bytes);
    *out = tensor;
    *host = tensor->host_data;
    yvex_error_clear(err);
    return YVEX_OK;
}
static const yvex_backend_vtable cuda_vtable = {
    cuda_close,
    cuda_memory_stats,
    cuda_device_info,
    yvex_cuda_tensor_alloc,
    cuda_resident_alloc,
    yvex_cuda_tensor_free,
    yvex_cuda_tensor_write,
    yvex_cuda_tensor_read,
    yvex_cuda_tensor_copy,
    cuda_sync,
    yvex_cuda_query_capability,
    yvex_cuda_op_embed,
    yvex_cuda_op_rms_norm,
    yvex_cuda_op_rope,
    yvex_cuda_op_matmul,
    yvex_cuda_op_mlp,
    yvex_cuda_op_attention,
    cuda_host_workspace_alloc,
    cuda_host_workspace_free,
};
/* Purpose: retain one live lease before a shared CUDA backend copies owner resources.
 * Inputs: a stable outer owner reference whose lifecycle has not entered close.
 * Effects: atomically increments only the packed live-child count.
 * Failure: closing, failed, saturated, or malformed owners retain their prior lifecycle.
 * Boundary: publication owns matching release through the generic checked-close lifecycle. */
static int shared_owner_acquire(yvex_backend *owner, yvex_error *err)
{
    unsigned long long desired, observed;
    if (!owner || owner->resource_owner != owner ||
        owner->status == YVEX_BACKEND_STATUS_FAILED) {
        yvex_error_set(err, YVEX_ERR_STATE, "backend.shared.acquire",
                       "one live primary backend resource owner is required");
        return YVEX_ERR_STATE;
    }
    observed = atomic_load_explicit(&owner->lifecycle, memory_order_acquire);
    for (;;) {
        if ((observed & YVEX_BACKEND_LIFECYCLE_CLOSING) ||
            (observed & YVEX_BACKEND_LIFECYCLE_CHILD_MASK) ==
                YVEX_BACKEND_LIFECYCLE_CHILD_MASK) {
            yvex_error_set(err, YVEX_ERR_STATE, "backend.shared.acquire",
                           "backend resource owner is closing or saturated");
            return YVEX_ERR_STATE;
        }
        desired = observed + 1ull;
        if (atomic_compare_exchange_weak_explicit(
                &owner->lifecycle, &observed, desired,
                memory_order_acq_rel, memory_order_acquire)) {
            yvex_error_clear(err);
            return YVEX_OK;
        }
    }
}
/* Purpose: roll back one fully initialized CUDA-open candidate without losing retry ownership.
 * Inputs: caller output, owned candidate, primary status/error, and diagnostic output.
 * Effects: closes the candidate or returns its failed cleanup owner to the caller.
 * Failure: cleanup failure supersedes the primary error and preserves the live handle.
 * Boundary: centralizes open rollback only; normal backend close policy remains unchanged. */
static int cuda_open_rollback(yvex_backend **out, yvex_backend **backend,
                              int primary_status, yvex_error primary,
                              yvex_error *err)
{
    yvex_error cleanup;
    int cleanup_status;
    yvex_error_clear(&cleanup);
    cleanup_status = yvex_backend_close_checked(backend, &cleanup);
    if (cleanup_status != YVEX_OK) {
        *out = *backend;
        if (err) *err = cleanup;
        return cleanup_status;
    }
    if (err) *err = primary;
    return primary_status;
}
/* Purpose: Construct the admitted open impl state only after its identities and resources are valid.
 * Inputs: A validated configuration, checked resource limits, and caller-owned result storage.
 * Effects: Updates only caller-owned result storage or lifecycle state explicitly named by the ABI.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
int yvex_backend_open_cuda_impl(yvex_backend **out,
                                const char *device,
                                unsigned long long memory_limit_bytes,
                                yvex_error *err)
{
    yvex_backend *backend = NULL;
    yvex_cuda_backend_state *state = NULL;
    int device_index = 0, device_count = 0, unified = 0, managed = 0, can_map_host = 0;
    size_t global_bytes = 0;
    int rc;
    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_backend_open_cuda_impl",
                       "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    rc = parse_device_index(device, &device_index, err);
    if (rc != YVEX_OK) return rc;
    backend = (yvex_backend *)calloc(1, sizeof(*backend));
    state = (yvex_cuda_backend_state *)calloc(1, sizeof(*state));
    if (!backend || !state) {
        free(state);
        free(backend);
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_backend_open_cuda_impl",
                       "failed to allocate CUDA backend");
        return YVEX_ERR_NOMEM;
    }
    backend->kind = YVEX_BACKEND_KIND_CUDA;
    atomic_init(&backend->status, YVEX_BACKEND_STATUS_FAILED);
    backend->vtable = &cuda_vtable;
    backend->impl = state;
    backend->resource_owner = backend;
    atomic_init(&backend->lifecycle, 0ull);
    rc = yvex_cuda_driver_load(&state->driver, err);
    if (rc != YVEX_OK) goto failed;
    rc = yvex_cuda_status(&state->driver, state->driver.cuInit(0),
                          "yvex_backend_open_cuda_impl", err);
    if (rc != YVEX_OK) goto failed;
    rc = yvex_cuda_status(&state->driver, state->driver.cuDeviceGetCount(&device_count),
                          "yvex_backend_open_cuda_impl", err);
    if (rc != YVEX_OK || device_count <= 0) {
        if (rc == YVEX_OK) {
            yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "yvex_backend_open_cuda_impl",
                           "CUDA runtime is available but no CUDA device was found");
            rc = YVEX_ERR_UNSUPPORTED;
        }
        goto failed;
    }
    if (device_index < 0 || device_index >= device_count) {
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "yvex_backend_open_cuda_impl",
                        "CUDA device %d is out of range; device_count=%d",
                        device_index, device_count);
        rc = YVEX_ERR_INVALID_ARG;
        goto failed;
    }
    rc = yvex_cuda_status(&state->driver, state->driver.cuDeviceGet(&state->device, device_index),
                          "yvex_backend_open_cuda_impl", err);
    if (rc == YVEX_OK) {
        (void)state->driver.cuDeviceGetAttribute(
            &can_map_host, YVEX_CUDA_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, state->device);
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuCtxCreate_v2(
                                  &state->context,
                                  can_map_host ? YVEX_CUDA_CTX_MAP_HOST : 0u,
                                  state->device),
                              "yvex_backend_open_cuda_impl", err);
    }
    if (rc != YVEX_OK) goto failed;
    state->device_index = device_index;
    (void)state->driver.cuDriverGetVersion(&state->driver_version);
    (void)state->driver.cuDeviceGetName(backend->device_name_storage,
                                        (int)sizeof(backend->device_name_storage),
                                        state->device);
    (void)state->driver.cuDeviceComputeCapability(&backend->device_info.compute_capability_major,
                                                  &backend->device_info.compute_capability_minor,
                                                  state->device);
    (void)state->driver.cuDeviceTotalMem_v2(&global_bytes, state->device);
    (void)state->driver.cuDeviceGetAttribute(&unified,
                                             YVEX_CUDA_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                                             state->device);
    (void)state->driver.cuDeviceGetAttribute(&managed,
                                             YVEX_CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                                             state->device);
    backend->status = YVEX_BACKEND_STATUS_CONTEXT_READY;
    backend->stats.memory_limit_bytes = memory_limit_bytes;
    backend->tensor_id_next = 1;
    backend->device_info.kind = YVEX_BACKEND_KIND_CUDA;
    backend->device_info.name = backend->device_name_storage;
    backend->device_info.device_index = device_index;
    backend->device_info.global_memory_bytes = (unsigned long long)global_bytes;
    backend->device_info.unified_addressing = unified != 0;
    backend->device_info.managed_memory = managed != 0;
    (void)yvex_cuda_refresh_memory_info(backend, err);
    rc = cuda_timing_open(backend, err);
    if (rc != YVEX_OK) goto failed;
    rc = yvex_cuda_kernel_bundle_admit(backend, err);
    if (rc == YVEX_OK) {
        backend->status = YVEX_BACKEND_STATUS_READY;
    } else if (backend_cleanup_only(backend) ||
               backend->status == YVEX_BACKEND_STATUS_FAILED) {
        *out = backend;
        return rc;
    } else {
        yvex_error_clear(err);
    }
    *out = backend;
    yvex_error_clear(err);
    return YVEX_OK;
failed:
    return cuda_open_rollback(out, &backend, rc,
                              err ? *err : (yvex_error){0}, err);
}
/* Purpose: create session-local CUDA state that borrows one model-owned context.
 * Inputs: live CUDA owner and an exact session allocation budget.
 * Effects: owns a separate kernel module, graph registry, streams, and memory counters.
 * Failure: publishes only a failed cleanup owner when rollback cannot release its module.
 * Boundary: never destroys or unloads the model-owned context and Driver handle. */
int yvex_backend_open_shared_cuda(yvex_backend **out,
                                  yvex_backend *context_owner,
                                  unsigned long long memory_limit_bytes,
                                  yvex_error *err)
{
    const yvex_cuda_backend_state *owner;
    yvex_backend *backend;
    yvex_cuda_backend_state *state;
    int rc;
    if (out) *out = NULL;
    if (!out || !context_owner || context_owner->kind != YVEX_BACKEND_KIND_CUDA ||
        context_owner->resource_owner != context_owner) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.shared.open",
                       "one owning CUDA context is required");
        return YVEX_ERR_INVALID_ARG;
    }
    backend = (yvex_backend *)calloc(1u, sizeof(*backend));
    state = (yvex_cuda_backend_state *)calloc(1u, sizeof(*state));
    if (!backend || !state) {
        free(state);
        free(backend);
        yvex_error_set(err, YVEX_ERR_NOMEM, "cuda.shared.open",
                       "shared CUDA session allocation failed");
        return YVEX_ERR_NOMEM;
    }
    rc = shared_owner_acquire(context_owner, err);
    if (rc != YVEX_OK) {
        free(state);
        free(backend);
        return rc;
    }
    backend->kind = YVEX_BACKEND_KIND_CUDA;
    atomic_init(&backend->status, YVEX_BACKEND_STATUS_FAILED);
    backend->vtable = &cuda_vtable;
    backend->impl = state;
    backend->resource_owner = context_owner;
    atomic_init(&backend->lifecycle, 0ull);
    backend->shared_owner_registered = 1;
    state->context_borrowed = 1;
    owner = yvex_cuda_state(context_owner);
    if (!owner || !owner->context || owner->context_borrowed) {
        yvex_error primary;
        yvex_error_set(&primary, YVEX_ERR_STATE, "cuda.shared.open",
                       "owning CUDA context became unavailable");
        return cuda_open_rollback(out, &backend, YVEX_ERR_STATE, primary, err);
    }
    state->driver = owner->driver;
    state->context = owner->context;
    state->device = owner->device;
    state->device_index = owner->device_index;
    state->driver_version = owner->driver_version;
    state->context_owner = context_owner;
    state->context_borrowed = 1;
    backend->status = YVEX_BACKEND_STATUS_CONTEXT_READY;
    backend->stats.memory_limit_bytes = memory_limit_bytes;
    backend->tensor_id_next = 1ull;
    backend->device_info = context_owner->device_info;
    yvex_core_text_copy(backend->device_name_storage,
                        sizeof(backend->device_name_storage),
                        context_owner->device_name_storage);
    backend->device_info.name = backend->device_name_storage;
    rc = cuda_timing_open(backend, err);
    if (rc != YVEX_OK) {
        return cuda_open_rollback(out, &backend, rc,
                                  err ? *err : (yvex_error){0}, err);
    }
    rc = yvex_cuda_kernel_bundle_admit(backend, err);
    if (rc != YVEX_OK) {
        return cuda_open_rollback(out, &backend, rc,
                                  err ? *err : (yvex_error){0}, err);
    }
    backend->status = YVEX_BACKEND_STATUS_READY;
    *out = backend;
    yvex_error_clear(err);
    return YVEX_OK;
}
