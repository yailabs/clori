/*
 * Construct and release the dynamically admitted CUDA backend context.
 *
 * Context creation yields context-ready only; ready requires atomic canonical bundle admission;
 * close clears every owned Driver API handle. An open CUDA context is not primitive or model
 * runtime support.
 */
#include "src/backend/cuda/private.h"
#include <ctype.h>
#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUDA_BANDWIDTH_WORKING_SET (32ull * 1024ull * 1024ull)
#define CUDA_BANDWIDTH_ITERATIONS 8ull
#define CUDA_BANDWIDTH_BLOCK 256u

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

static int cuda_lifecycle_failure_matches(const char *variable, const char *stage)
{
    const char *selected = variable ? getenv(variable) : NULL;
    return selected && stage && strcmp(selected, stage) == 0;
}

/* Admit cuBLAS opportunistically; family consumers decide when its GEMM is mandatory. */
static int cuda_blas_open(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_blas *blas;
    if (!state) return YVEX_ERR_INVALID_ARG;
    blas = &state->blas;
    blas->library = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
    if (!blas->library) blas->library = dlopen("libcublas.so", RTLD_NOW | RTLD_LOCAL);
    if (!blas->library) goto unavailable;
    blas->create = NULL;
    *(void **)(&blas->create) = dlsym(blas->library, "cublasCreate_v2");
    *(void **)(&blas->destroy) = dlsym(blas->library, "cublasDestroy_v2");
    *(void **)(&blas->set_stream) = dlsym(blas->library, "cublasSetStream_v2");
    *(void **)(&blas->gemm_ex) = dlsym(blas->library, "cublasGemmEx");
    if (!blas->create || !blas->destroy || !blas->set_stream || !blas->gemm_ex ||
        blas->create(&blas->handle) != 0 ||
        blas->set_stream(blas->handle, state->execution_stream) != 0) {
        if (blas->handle && blas->destroy) (void)blas->destroy(blas->handle);
        dlclose(blas->library);
        goto unavailable;
    }
    blas->ready = 1;
    yvex_error_clear(err);
    return YVEX_OK;
unavailable:
    memset(blas, 0, sizeof(*blas));
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Release the library handle before its CUDA stream and context disappear. */
static int cuda_blas_close(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_blas *blas;
    if (!state) return YVEX_OK;
    blas = &state->blas;
    if (blas->handle && (!blas->destroy || blas->destroy(blas->handle) != 0)) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.blas.close",
                       "cuBLAS handle cleanup failed");
        return YVEX_ERR_BACKEND;
    }
    if (blas->library) dlclose(blas->library);
    memset(blas, 0, sizeof(*blas));
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Own one non-blocking execution stream per backend/session when the Driver exposes it. */
static int cuda_execution_stream_open(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_driver *driver;
    int rc;
    if (!state) return YVEX_ERR_INVALID_ARG;
    driver = &state->driver;
    if (!driver->cuStreamCreate || !driver->cuStreamDestroy_v2 ||
        !driver->cuStreamSynchronize || !driver->cuMemcpyHtoDAsync_v2 ||
        !driver->cuMemcpyDtoHAsync_v2 || !driver->cuMemcpyDtoDAsync_v2 ||
        !driver->cuMemsetD8Async) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_cuda_set_current(backend, "cuda.execution_stream.open", err);
    if (rc == YVEX_OK &&
        cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_STREAM_FAILURE", "create")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.execution_stream.open",
                       "injected CUDA execution stream creation failure");
        rc = YVEX_ERR_BACKEND;
    } else if (rc == YVEX_OK) {
        rc = yvex_cuda_status(
            driver,
            driver->cuStreamCreate(&state->execution_stream,
                                   YVEX_CUDA_STREAM_NON_BLOCKING),
            "cuda.execution_stream.open", err);
    }
    return rc;
}

/* Release the session stream before unloading kernels or destroying its context. */
static int cuda_execution_stream_close(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc;
    if (!state || !state->execution_stream) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_cuda_set_current(backend, "cuda.execution_stream.close", err);
    if (rc == YVEX_OK &&
        cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_STREAM_FAILURE", "destroy")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.execution_stream.close",
                       "injected CUDA execution stream cleanup failure");
        rc = YVEX_ERR_BACKEND;
    } else if (rc == YVEX_OK) {
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuStreamDestroy_v2(state->execution_stream),
            "cuda.execution_stream.close", err);
    }
    if (rc == YVEX_OK) state->execution_stream = NULL;
    return rc;
}
/*
 * Create one reusable event pair before a backend enters warm execution.
 *
 * Allocates no per-replay event.
 */
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
    if (cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_EVENT_FAILURE", "create-start"))
        rc = YVEX_ERR_BACKEND;
    else
        rc = yvex_cuda_status(driver, driver->cuEventCreate(&state->timing_start, 0u),
                              "cuda.timing.create_start", err);
    if (rc == YVEX_OK) {
        if (cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_EVENT_FAILURE", "create-stop"))
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
/*
 * Release the reusable event pair before its CUDA context becomes invalid.
 *
 * Preserves remaining event ownership for checked retry.
 */
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
        if (cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_EVENT_FAILURE", "destroy-stop")) {
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
        if (cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_EVENT_FAILURE", "destroy-start")) {
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
/*
 * Begin, finish, or discard one interval through the reusable CUDA event pair.
 *
 * Changes only timing ownership and publishes device elapsed nanoseconds on finish.
 */
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
        if (cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_EVENT_FAILURE", "record-start")) {
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
    if (cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_EVENT_FAILURE", "record-stop")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "injected CUDA timing stop-record failure");
        rc = YVEX_ERR_BACKEND;
    } else {
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuEventRecord(state->timing_stop, stream), where, err);
    }
    if (rc == YVEX_OK &&
        cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_EVENT_FAILURE", "synchronize")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, where,
                       "injected CUDA timing synchronization failure");
        rc = YVEX_ERR_BACKEND;
    } else if (rc == YVEX_OK) {
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuEventSynchronize(state->timing_stop), where, err);
    }
    if (rc == YVEX_OK &&
        cuda_lifecycle_failure_matches("YVEX_TEST_CUDA_EVENT_FAILURE", "elapsed")) {
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
/* Discharge CUDA ownership in dependency order for checked backend close. */
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
    rc = cuda_blas_close(backend, err);
    if (rc != YVEX_OK)
        return rc;
    rc = cuda_execution_stream_close(backend, err);
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

typedef struct {
    CUdeviceptr address;
    size_t logical_bytes, reserved_bytes, granularity, page_count;
    CUmemGenericAllocationHandle *handles;
    unsigned char *mapped;
} cuda_virtual_allocation;

static void cuda_virtual_properties(const yvex_cuda_backend_state *state,
                                    CUmemAllocationProp *properties,
                                    CUmemAccessDesc *access)
{
    memset(properties, 0, sizeof(*properties));
    properties->type = YVEX_CUDA_MEM_ALLOCATION_PINNED;
    properties->location.type = YVEX_CUDA_MEM_LOCATION_DEVICE;
    properties->location.id = state->device_index;
    if (access) {
        memset(access, 0, sizeof(*access));
        access->location = properties->location;
        access->flags = YVEX_CUDA_MEM_ACCESS_READ_WRITE;
    }
}

static int cuda_virtual_round(size_t bytes, size_t granularity,
                              size_t *rounded)
{
    size_t remainder;
    if (!bytes || !granularity || !rounded) return 0;
    remainder = bytes % granularity;
    if (!remainder) {
        *rounded = bytes;
        return 1;
    }
    if (bytes > SIZE_MAX - (granularity - remainder)) return 0;
    *rounded = bytes + granularity - remainder;
    return 1;
}

static int cuda_virtual_metadata_open(
    yvex_backend *backend, const yvex_backend_tensor_desc *desc,
    cuda_virtual_allocation *allocation, yvex_device_tensor **out,
    yvex_error *err)
{
    yvex_device_tensor *tensor = (yvex_device_tensor *)calloc(1u, sizeof(*tensor));
    unsigned int index;
    if (tensor) tensor->name = yvex_core_strdup(desc->name);
    if (!tensor || !tensor->name) {
        free(tensor);
        yvex_error_set(err, YVEX_ERR_NOMEM, "cuda.tensor.reserve",
                       "virtual tensor metadata allocation failed");
        return YVEX_ERR_NOMEM;
    }
    tensor->owner = backend;
    tensor->owner_id = backend->tensor_id_next++;
    tensor->dtype = desc->dtype;
    tensor->rank = desc->rank;
    for (index = 0u; index < desc->rank; ++index)
        tensor->dims[index] = desc->dims[index];
    tensor->bytes = desc->bytes;
    tensor->data = (unsigned char *)(uintptr_t)allocation->address;
    tensor->backend_allocation = allocation;
    tensor->virtual_reserved = 1;
    *out = tensor;
    return YVEX_OK;
}

/* Reserve stable CUDA addresses while leaving physical state pages uncommitted. */
static int cuda_tensor_reserve(yvex_backend *backend,
                               const yvex_backend_tensor_desc *desc,
                               yvex_device_tensor **out,
                               unsigned long long *granularity_out,
                               yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    cuda_virtual_allocation *allocation = NULL;
    CUmemAllocationProp properties;
    size_t granularity = 0u, reserved = 0u;
    CUresult status;
    int rc;
    if (!state || !desc || !out || !granularity_out ||
        desc->bytes > (unsigned long long)SIZE_MAX ||
        !state->virtual_memory_management) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.tensor.reserve",
                       "CUDA virtual-memory reservation is unavailable");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = yvex_cuda_set_current(backend, "cuda.tensor.reserve", err);
    if (rc != YVEX_OK) return rc;
    cuda_virtual_properties(state, &properties, NULL);
    status = state->driver.cuMemGetAllocationGranularity(
        &granularity, &properties, YVEX_CUDA_MEM_GRANULARITY_MINIMUM);
    if (yvex_cuda_status(&state->driver, status,
                         "cuda.tensor.reserve.granularity", err) != YVEX_OK ||
        !cuda_virtual_round((size_t)desc->bytes, granularity, &reserved)) {
        if (!yvex_error_is_set(err))
            yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.tensor.reserve",
                           "virtual tensor reservation geometry overflowed");
        return yvex_error_code(err);
    }
    allocation = (cuda_virtual_allocation *)calloc(1u, sizeof(*allocation));
    if (allocation) {
        allocation->logical_bytes = (size_t)desc->bytes;
        allocation->reserved_bytes = reserved;
        allocation->granularity = granularity;
        allocation->page_count = reserved / granularity;
        allocation->handles = (CUmemGenericAllocationHandle *)calloc(
            allocation->page_count, sizeof(*allocation->handles));
        allocation->mapped = (unsigned char *)calloc(
            allocation->page_count, sizeof(*allocation->mapped));
    }
    if (!allocation || !allocation->handles || !allocation->mapped) {
        if (allocation) {
            free(allocation->handles);
            free(allocation->mapped);
        }
        free(allocation);
        yvex_error_set(err, YVEX_ERR_NOMEM, "cuda.tensor.reserve",
                       "virtual tensor page directory allocation failed");
        return YVEX_ERR_NOMEM;
    }
    rc = yvex_cuda_status(
        &state->driver,
        state->driver.cuMemAddressReserve(
            &allocation->address, reserved, granularity, 0ull, 0ull),
        "cuda.tensor.reserve", err);
    if (rc == YVEX_OK)
        rc = cuda_virtual_metadata_open(backend, desc, allocation, out, err);
    if (rc != YVEX_OK) {
        if (allocation->address)
            (void)state->driver.cuMemAddressFree(
                allocation->address, allocation->reserved_bytes);
        free(allocation->handles);
        free(allocation->mapped);
        free(allocation);
        return rc;
    }
    *granularity_out = (unsigned long long)granularity;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int cuda_virtual_page_release(
    yvex_backend *backend, cuda_virtual_allocation *allocation,
    size_t page, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr address = allocation->address + page * allocation->granularity;
    int rc = YVEX_OK;
    if (allocation->mapped[page]) {
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemUnmap(address, allocation->granularity),
            "cuda.tensor.decommit.unmap", err);
        if (rc == YVEX_OK) allocation->mapped[page] = 0u;
    }
    if (rc == YVEX_OK && allocation->handles[page]) {
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemRelease(allocation->handles[page]),
            "cuda.tensor.decommit.release", err);
        if (rc == YVEX_OK) {
            allocation->handles[page] = 0ull;
            backend_memory_release(backend, allocation->granularity);
        }
    }
    return rc;
}

static int cuda_virtual_rollback(yvex_backend *backend,
                                 cuda_virtual_allocation *allocation,
                                 const size_t *pages, size_t count,
                                 yvex_error *err)
{
    int rc = YVEX_OK;
    while (count) {
        yvex_error cleanup;
        int released = cuda_virtual_page_release(
            backend, allocation, pages[--count], &cleanup);
        if (released != YVEX_OK && rc == YVEX_OK) {
            rc = released;
            if (err) *err = cleanup;
        }
    }
    return rc;
}

/* Map only the physical allocation granules intersecting one admitted state range. */
static int cuda_tensor_commit(yvex_backend *backend, yvex_device_tensor *tensor,
                              unsigned long long offset,
                              unsigned long long bytes,
                              unsigned long long *resident_delta,
                              yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    cuda_virtual_allocation *allocation =
        tensor ? (cuda_virtual_allocation *)tensor->backend_allocation : NULL;
    CUmemAllocationProp properties;
    CUmemAccessDesc access;
    size_t first, last, page, missing = 0u, committed = 0u;
    size_t *pages = NULL;
    unsigned long long required;
    int rc;
    if (!state || !allocation || !tensor->virtual_reserved ||
        !backend_tensor_owner_is(backend, tensor) || !bytes ||
        offset > tensor->bytes || bytes > tensor->bytes - offset) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.tensor.commit",
                       "one owned virtual tensor range is required");
        return YVEX_ERR_INVALID_ARG;
    }
    first = (size_t)offset / allocation->granularity;
    last = ((size_t)offset + (size_t)bytes - 1u) / allocation->granularity;
    for (page = first; page <= last; ++page)
        if (!allocation->handles[page]) ++missing;
    if (!missing) {
        *resident_delta = 0ull;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (missing > SIZE_MAX / allocation->granularity) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.tensor.commit",
                       "physical page commitment extent overflowed");
        return YVEX_ERR_BOUNDS;
    }
    required = (unsigned long long)(missing * allocation->granularity);
    rc = yvex_backend_memory_can_add(
        backend, required, "CUDA state pages", "cuda.tensor.commit", err);
    if (rc != YVEX_OK) return rc;
    pages = (size_t *)calloc(missing, sizeof(*pages));
    if (!pages) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "cuda.tensor.commit",
                       "physical page transaction allocation failed");
        return YVEX_ERR_NOMEM;
    }
    rc = yvex_cuda_set_current(backend, "cuda.tensor.commit", err);
    cuda_virtual_properties(state, &properties, &access);
    for (page = first; rc == YVEX_OK && page <= last; ++page) {
        CUdeviceptr address;
        if (allocation->handles[page]) continue;
        address = allocation->address + page * allocation->granularity;
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemCreate(&allocation->handles[page],
                                      allocation->granularity, &properties, 0ull),
            "cuda.tensor.commit.create", err);
        if (rc == YVEX_OK) {
            backend_memory_acquire(backend, allocation->granularity);
            pages[committed++] = page;
            rc = yvex_cuda_status(
                &state->driver,
                state->driver.cuMemMap(address, allocation->granularity, 0u,
                                       allocation->handles[page], 0ull),
                "cuda.tensor.commit.map", err);
        }
        if (rc == YVEX_OK) {
            allocation->mapped[page] = 1u;
            rc = yvex_cuda_status(
                &state->driver,
                state->driver.cuMemSetAccess(
                    address, allocation->granularity, &access, 1u),
                "cuda.tensor.commit.access", err);
        }
        if (rc == YVEX_OK)
            rc = yvex_cuda_status(
                &state->driver,
                state->driver.cuMemsetD8_v2(
                    address, 0u, allocation->granularity),
                "cuda.tensor.commit.zero", err);
    }
    if (rc != YVEX_OK) {
        yvex_error primary = *err, cleanup;
        if (cuda_virtual_rollback(
                backend, allocation, pages, committed, &cleanup) != YVEX_OK) {
            atomic_store_explicit(&backend->status, YVEX_BACKEND_STATUS_FAILED,
                                  memory_order_release);
            *err = cleanup;
            rc = yvex_error_code(err);
        } else {
            *err = primary;
        }
        free(pages);
        return rc;
    }
    if (tensor->resident_bytes > ULLONG_MAX - required) {
        yvex_error cleanup;
        rc = cuda_virtual_rollback(backend, allocation, pages, committed, &cleanup);
        free(pages);
        if (rc != YVEX_OK) {
            atomic_store_explicit(&backend->status, YVEX_BACKEND_STATUS_FAILED,
                                  memory_order_release);
            *err = cleanup;
            return rc;
        }
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.tensor.commit",
                       "resident page accounting overflowed");
        return YVEX_ERR_BOUNDS;
    }
    tensor->resident_bytes += required;
    *resident_delta = required;
    free(pages);
    (void)yvex_cuda_refresh_memory_info(backend, err);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int cuda_tensor_decommit(yvex_backend *backend,
                                yvex_device_tensor *tensor,
                                unsigned long long *released_bytes,
                                yvex_error *err)
{
    cuda_virtual_allocation *allocation =
        tensor ? (cuda_virtual_allocation *)tensor->backend_allocation : NULL;
    unsigned long long released = 0ull;
    size_t page;
    int rc;
    if (!allocation || !tensor->virtual_reserved ||
        !backend_tensor_owner_is(backend, tensor)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.tensor.decommit",
                       "one owned virtual tensor is required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_cuda_set_current(backend, "cuda.tensor.decommit", err);
    for (page = 0u; rc == YVEX_OK && page < allocation->page_count; ++page) {
        if (allocation->handles[page]) {
            rc = cuda_virtual_page_release(backend, allocation, page, err);
            if (rc == YVEX_OK) released += allocation->granularity;
        }
    }
    tensor->resident_bytes = tensor->resident_bytes >= released
                                 ? tensor->resident_bytes - released : 0ull;
    if (rc != YVEX_OK) return rc;
    *released_bytes = released;
    tensor->is_written = 0;
    (void)yvex_cuda_refresh_memory_info(backend, err);
    yvex_error_clear(err);
    return YVEX_OK;
}

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
/*
 * Own CUDA tensor allocation, transfer, copy, accounting, and release.
 *
 * Writes become visible only after synchronization; failed release preserves tensor ownership and
 * counters; unpublished allocations are cleaned. Tensor movement is not CUDA graph or generation
 * support.
 */
static int cuda_tensor_alloc(yvex_backend *backend,
                           const yvex_backend_tensor_desc *desc,
                           yvex_device_tensor **out,
                           yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work;
    yvex_device_tensor *tensor = NULL;
    yvex_error cleanup_error, primary_error;
    CUdeviceptr ptr = 0;
    unsigned int i;
    int cleanup_rc;
    int rc;
    memset(&work, 0, sizeof(work));
    if (!backend || !state || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.tensor_alloc",
                       "backend and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    rc = yvex_cuda_deferred_release_drain(backend, err);
    if (rc != YVEX_OK)
        return rc;
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_ALLOC,
                                      "cuda.tensor_alloc", err);
    if (rc == YVEX_OK) {
        rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_ZERO,
                                          "cuda.tensor_alloc", err);
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_set_current(backend, "cuda.tensor_alloc", err);
    if (rc != YVEX_OK)
        return rc;
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_TENSOR_ALLOC;
    work.raw_only = 1;
    rc = yvex_cuda_work_allocate(&work, &ptr, (size_t)desc->bytes, NULL, 1,
                                 "cuda.tensor_alloc", NULL, err);
    if (rc != YVEX_OK)
        goto allocation_failure;
    rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_TENSOR_ZERO,
                               "cuda.tensor_alloc.zero_sync", err);
    if (rc != YVEX_OK)
        goto allocation_failure;
    tensor = (yvex_device_tensor *)calloc(1, sizeof(*tensor));
    if (!tensor) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "cuda.tensor_alloc",
                       "failed to allocate CUDA tensor object");
        rc = YVEX_ERR_NOMEM;
        goto allocation_failure;
    }
    tensor->name = yvex_core_strdup(desc->name);
    if (!tensor->name) {
        free(tensor);
        yvex_error_set(err, YVEX_ERR_NOMEM, "cuda.tensor_alloc",
                       "failed to copy CUDA tensor name");
        rc = YVEX_ERR_NOMEM;
        goto allocation_failure;
    }
    tensor->owner = backend;
    tensor->owner_id = backend->tensor_id_next++;
    tensor->dtype = desc->dtype;
    tensor->rank = desc->rank;
    for (i = 0; i < desc->rank; ++i) {
        tensor->dims[i] = desc->dims[i];
    }
    tensor->bytes = desc->bytes;
    tensor->data = (unsigned char *)(uintptr_t)ptr;
    work.count = 0u;
    work.current_bytes = 0ull;
    (void)yvex_cuda_refresh_memory_info(backend, err);
    *out = tensor;
    yvex_error_clear(err);
    return YVEX_OK;
allocation_failure:
    if (err)
        primary_error = *err;
    else
        yvex_error_clear(&primary_error);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup_error);
    (void)cleanup_rc;
    if (err)
        *err = primary_error;
    return rc;
}

static int cuda_tensor_free(yvex_backend *backend,
                          yvex_device_tensor *tensor,
                          yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    cuda_virtual_allocation *allocation;
    unsigned long long released = 0ull;
    CUdeviceptr pointer;
    int rc;
    if (!backend || !state || !tensor || !backend_tensor_owner_is(backend, tensor)) {
        yvex_error_set(err, YVEX_ERR_STATE, "cuda.tensor_free",
                       "tensor does not belong to this backend");
        return YVEX_ERR_STATE;
    }
    rc = yvex_cuda_set_current(backend, "cuda.tensor_free", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    allocation = (cuda_virtual_allocation *)tensor->backend_allocation;
    if (tensor->virtual_reserved && allocation) {
        rc = cuda_tensor_decommit(backend, tensor, &released, err);
        if (rc == YVEX_OK)
            rc = yvex_cuda_status(
                &state->driver,
                state->driver.cuMemAddressFree(
                    allocation->address, allocation->reserved_bytes),
                "cuda.tensor.free.address", err);
        if (rc != YVEX_OK) return rc;
        free(allocation->handles);
        free(allocation->mapped);
        free(allocation);
        tensor->backend_allocation = NULL;
        tensor->data = NULL;
    }
    pointer = yvex_cuda_tensor_ptr(tensor);
    if (tensor->virtual_reserved) {
        rc = YVEX_OK;
    } else if (tensor->host_data && state->registered_host == tensor->host_data &&
        state->registered_device == pointer && state->registered_bytes == tensor->bytes) {
        rc = state->driver.cuMemHostUnregister
                 ? yvex_cuda_status(&state->driver,
                                    state->driver.cuMemHostUnregister(tensor->host_data),
                                    "cuda.tensor_free.registered", err)
                 : YVEX_ERR_STATE;
        if (rc == YVEX_ERR_STATE)
            yvex_error_set(err, rc, "cuda.tensor_free.registered",
                           "CUDA host registration release is unavailable");
        if (rc == YVEX_OK) {
            state->registered_host = NULL;
            state->registered_device = 0ull;
            state->registered_bytes = 0ull;
        }
    } else {
        rc = yvex_cuda_temporary_free(backend, YVEX_BACKEND_VARIANT_TENSOR_ALLOC,
                                      &pointer, tensor->bytes, 0,
                                      "cuda.tensor_free", err);
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    tensor->owner = NULL;
    tensor->owner_id = 0;
    free(tensor->name);
    free(tensor);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int cuda_tensor_write(yvex_backend *backend,
                           yvex_device_tensor *tensor,
                           const void *src,
                           unsigned long long len,
                           yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc = yvex_backend_tensor_rw_validate(
        "yvex_backend_tensor_write", backend, tensor, len, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_WRITE,
                                      "yvex_backend_tensor_write", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    tensor->is_written = 0;
    rc = yvex_cuda_set_current(backend, "yvex_backend_tensor_write", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuMemcpyHtoD_v2(yvex_cuda_tensor_ptr(tensor),
                                                         src, (size_t)len),
                          "yvex_backend_tensor_write", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_TENSOR_WRITE,
                               "yvex_backend_tensor_write", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    tensor->is_written = 1;
    return YVEX_OK;
}

static int cuda_tensor_read(yvex_backend *backend,
                          const yvex_device_tensor *tensor,
                          void *dst,
                          unsigned long long len,
                          yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc = yvex_backend_tensor_rw_validate(
        "yvex_backend_tensor_read", backend, tensor, len, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_READ,
                                      "yvex_backend_tensor_read", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_set_current(backend, "yvex_backend_tensor_read", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuMemcpyDtoH_v2(dst, yvex_cuda_tensor_ptr(tensor),
                                                         (size_t)len),
                          "yvex_backend_tensor_read", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    return yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_TENSOR_READ,
                                 "yvex_backend_tensor_read", err);
}
static int cuda_tensor_copy(yvex_backend *backend,
                          yvex_device_tensor *dst,
                          const yvex_device_tensor *src,
                          yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    int rc;
    rc = yvex_backend_tensor_copy_validate(
        backend, dst, src, "yvex_backend_tensor_copy", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_COPY,
                                      "yvex_backend_tensor_copy", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    dst->is_written = 0;
    rc = yvex_cuda_set_current(backend, "yvex_backend_tensor_copy", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuMemcpyDtoD_v2(yvex_cuda_tensor_ptr(dst),
                                                         yvex_cuda_tensor_ptr(src),
                                                         (size_t)src->bytes),
                          "yvex_backend_tensor_copy", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_TENSOR_COPY,
                               "yvex_backend_tensor_copy", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    dst->is_written = src->is_written;
    return YVEX_OK;
}

/* Enqueue a same-stream state-bank clone without introducing a layer-local synchronization. */
static int cuda_tensor_copy_async(yvex_backend *backend,
                                  yvex_device_tensor *dst,
                                  const yvex_device_tensor *src,
                                  yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUstream stream;
    CUresult copied;
    int rc = yvex_backend_tensor_copy_validate(
        backend, dst, src, "runtime.state.residency.copy", err);
    if (rc != YVEX_OK) return rc;
    if (!state || yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "runtime.state.residency.copy",
                       "CUDA state residency requires a CUDA backend");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_cuda_set_current(backend, "runtime.state.residency.copy", err);
    if (rc != YVEX_OK) return rc;
    stream = yvex_cuda_launch_stream(backend);
    copied = stream && state->driver.cuMemcpyDtoDAsync_v2
                 ? state->driver.cuMemcpyDtoDAsync_v2(
                       yvex_cuda_tensor_ptr(dst), yvex_cuda_tensor_ptr(src),
                       (size_t)src->bytes, stream)
                 : !stream ? state->driver.cuMemcpyDtoD_v2(
                       yvex_cuda_tensor_ptr(dst), yvex_cuda_tensor_ptr(src),
                       (size_t)src->bytes)
                           : (CUresult)1;
    rc = yvex_cuda_status(&state->driver, copied,
                          "runtime.state.residency.copy", err);
    if (rc == YVEX_OK) dst->is_written = src->is_written;
    return rc;
}

static int cuda_bandwidth_rate(unsigned long long bytes,
                               unsigned long long iterations,
                               unsigned long long elapsed_ns,
                               unsigned long long *rate)
{
    unsigned long long total;
    return elapsed_ns && rate && yvex_core_u64_mul(bytes, iterations, &total) &&
           yvex_core_u64_mul(total, 1000000000ull, &total)
               ? (*rate = total / elapsed_ns) != 0ull
               : 0;
}

static unsigned long long cuda_bandwidth_median(
    const unsigned long long values[YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT])
{
    unsigned long long ordered[YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT];
    unsigned int index, insert;
    memcpy(ordered, values, sizeof(ordered));
    for (index = 1u; index < YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT; ++index) {
        unsigned long long value = ordered[index];
        for (insert = index; insert && ordered[insert - 1u] > value; --insert)
            ordered[insert] = ordered[insert - 1u];
        ordered[insert] = value;
    }
    return ordered[YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT / 2u];
}

static int cuda_bandwidth_identity(yvex_backend_bandwidth_evidence *evidence)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.cuda.bandwidth-evidence.v1") ||
        !yvex_sha256_update_u64(&hash, evidence->schema_version) ||
        !yvex_sha256_update_u64(&hash, evidence->working_set_bytes) ||
        !yvex_sha256_update_u64(&hash, evidence->iterations) ||
        !yvex_sha256_update_u64(&hash, evidence->sample_count) ||
        !yvex_sha256_update_text(&hash, evidence->kernel_bundle_identity)) return 0;
    for (index = 0u; index < YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT; ++index)
        if (!yvex_sha256_update_u64(&hash, evidence->stream_elapsed_ns[index]) ||
            !yvex_sha256_update_u64(&hash, evidence->copy_elapsed_ns[index]) ||
            !yvex_sha256_update_u64(&hash,
                                    evidence->coherent_host_elapsed_ns[index])) return 0;
    if (!yvex_sha256_update_u64(&hash,
                                evidence->sustainable_read_bytes_per_second) ||
        !yvex_sha256_update_u64(&hash,
                                evidence->sustainable_copy_bytes_per_second) ||
        !yvex_sha256_update_u64(
            &hash, evidence->sustainable_coherent_host_bytes_per_second) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, evidence->identity);
    return 1;
}

static int cuda_bandwidth_host_sample(
    volatile unsigned long long *words, unsigned long long word_count,
    unsigned long long iterations, unsigned long long *elapsed_ns,
    yvex_error *err)
{
    static volatile unsigned long long retained;
    struct timespec start, finish;
    unsigned long long iteration, index, value = 0ull, seconds, nanoseconds;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        yvex_error_set(err, YVEX_ERR_IO, "cuda.bandwidth.host",
                       "monotonic host timing is unavailable");
        return YVEX_ERR_IO;
    }
    for (iteration = 0ull; iteration < iterations; ++iteration)
        for (index = 0ull; index < word_count; ++index)
            value += words[index];
    if (clock_gettime(CLOCK_MONOTONIC, &finish) != 0 ||
        finish.tv_sec < start.tv_sec) {
        yvex_error_set(err, YVEX_ERR_IO, "cuda.bandwidth.host",
                       "coherent host timing failed");
        return YVEX_ERR_IO;
    }
    seconds = (unsigned long long)(finish.tv_sec - start.tv_sec);
    nanoseconds = finish.tv_nsec >= start.tv_nsec
                      ? (unsigned long long)(finish.tv_nsec - start.tv_nsec)
                      : 1000000000ull - (unsigned long long)(start.tv_nsec - finish.tv_nsec);
    if (finish.tv_nsec < start.tv_nsec && seconds) --seconds;
    if (!yvex_core_u64_mul(seconds, 1000000000ull, elapsed_ns) ||
        !yvex_core_u64_add(*elapsed_ns, nanoseconds, elapsed_ns) || !*elapsed_ns) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.bandwidth.host",
                       "coherent host timing extent overflowed");
        return YVEX_ERR_BOUNDS;
    }
    retained ^= value;
    return YVEX_OK;
}

static int cuda_bandwidth_device_samples(
    yvex_backend *backend, yvex_device_tensor *managed,
    yvex_device_tensor *copy, yvex_device_tensor *status,
    yvex_backend_bandwidth_evidence *evidence, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr values = yvex_cuda_tensor_ptr(managed);
    CUdeviceptr destination = yvex_cuda_tensor_ptr(copy);
    CUdeviceptr device_status = yvex_cuda_tensor_ptr(status);
    CUstream stream = yvex_cuda_launch_stream(backend);
    unsigned long long elements = managed->bytes / sizeof(float), iteration;
    unsigned int grid = (unsigned int)((elements + CUDA_BANDWIDTH_BLOCK - 1ull) /
                                       CUDA_BANDWIDTH_BLOCK);
    unsigned int sample;
    int rc = YVEX_OK;
    void *params[] = {&values, &elements, &device_status};
    if (!state || !state->driver.cuMemcpyDtoDAsync_v2 ||
        yvex_cuda_capture_active(backend)) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.bandwidth.device",
                       "CUDA event, asynchronous copy, and eager execution are required");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                          state->attention_bf16_round_function, grid,
                          CUDA_BANDWIDTH_BLOCK, 0u, params,
                          "cuda.bandwidth.warmup", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuMemcpyDtoDAsync_v2(
                                  destination, values, (size_t)managed->bytes, stream),
                              "cuda.bandwidth.warmup.copy", err);
    if (rc == YVEX_OK) rc = yvex_backend_sync(backend, err);
    for (sample = 0u; rc == YVEX_OK && sample < evidence->sample_count; ++sample) {
        rc = yvex_cuda_timing(backend, stream, YVEX_CUDA_TIMING_BEGIN, NULL,
                              "cuda.bandwidth.stream.begin", err);
        for (iteration = 0ull; rc == YVEX_OK && iteration < evidence->iterations;
             ++iteration)
            rc = yvex_cuda_launch(
                backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                state->attention_bf16_round_function, grid, CUDA_BANDWIDTH_BLOCK,
                0u, params, "cuda.bandwidth.stream", err);
        if (rc == YVEX_OK)
            rc = yvex_cuda_timing(
                backend, stream, YVEX_CUDA_TIMING_FINISH,
                &evidence->stream_elapsed_ns[sample],
                "cuda.bandwidth.stream.finish", err);
        else
            (void)yvex_cuda_timing(backend, stream, YVEX_CUDA_TIMING_DISCARD,
                                   NULL, NULL, NULL);
    }
    for (sample = 0u; rc == YVEX_OK && sample < evidence->sample_count; ++sample) {
        rc = yvex_cuda_timing(backend, stream, YVEX_CUDA_TIMING_BEGIN, NULL,
                              "cuda.bandwidth.copy.begin", err);
        for (iteration = 0ull; rc == YVEX_OK && iteration < evidence->iterations;
             ++iteration)
            rc = yvex_cuda_status(
                &state->driver,
                state->driver.cuMemcpyDtoDAsync_v2(
                    destination, values, (size_t)evidence->working_set_bytes, stream),
                "cuda.bandwidth.copy", err);
        if (rc == YVEX_OK)
            rc = yvex_cuda_timing(
                backend, stream, YVEX_CUDA_TIMING_FINISH,
                &evidence->copy_elapsed_ns[sample],
                "cuda.bandwidth.copy.finish", err);
        else
            (void)yvex_cuda_timing(backend, stream, YVEX_CUDA_TIMING_DISCARD,
                                   NULL, NULL, NULL);
    }
    return rc;
}

static int cuda_bandwidth_probe(yvex_backend *backend,
                                yvex_backend_bandwidth_evidence *out,
                                yvex_error *err)
{
    yvex_backend *owner = backend && backend->resource_owner
                              ? backend->resource_owner : backend;
    yvex_cuda_backend_state *state = yvex_cuda_state(owner);
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *managed = NULL, *copy = NULL, *status = NULL;
    unsigned char *host = NULL;
    yvex_backend_bandwidth_evidence evidence = {0};
    yvex_error primary, cleanup, release_error;
    unsigned long long median, word_count;
    unsigned int sample;
    int rc, cleanup_rc;
    if (!state || !out || owner->kind != YVEX_BACKEND_KIND_CUDA) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.bandwidth",
                       "one CUDA resource owner and evidence output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (state->bandwidth_ready) {
        *out = state->bandwidth_evidence;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (state->bandwidth_active) {
        yvex_error_set(err, YVEX_ERR_STATE, "cuda.bandwidth",
                       "bandwidth measurement is already active");
        return YVEX_ERR_STATE;
    }
    state->bandwidth_active = 1;
    evidence.schema_version = YVEX_BACKEND_BANDWIDTH_SCHEMA_V1;
    evidence.working_set_bytes = CUDA_BANDWIDTH_WORKING_SET;
    evidence.iterations = CUDA_BANDWIDTH_ITERATIONS;
    evidence.sample_count = YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT;
    yvex_core_text_copy(evidence.kernel_bundle_identity,
                        sizeof(evidence.kernel_bundle_identity),
                        state->kernel_bundle_identity);
    descriptor.name = "bandwidth-managed";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = evidence.working_set_bytes;
    rc = cuda_resident_alloc(owner, &descriptor, &managed, &host, err);
    descriptor.name = "bandwidth-copy";
    if (rc == YVEX_OK) rc = yvex_backend_tensor_alloc(owner, &descriptor, &copy, err);
    descriptor.name = "bandwidth-status";
    descriptor.dtype = YVEX_DTYPE_I32;
    descriptor.dims[0] = 1ull;
    descriptor.bytes = sizeof(int);
    if (rc == YVEX_OK) rc = yvex_backend_tensor_alloc(owner, &descriptor, &status, err);
    if (rc == YVEX_OK) {
        memset(host, 0, (size_t)evidence.working_set_bytes);
        word_count = evidence.working_set_bytes / sizeof(unsigned long long);
        rc = cuda_bandwidth_host_sample(
            (volatile unsigned long long *)host, word_count, 1ull, &median, err);
    }
    if (rc == YVEX_OK)
        rc = cuda_bandwidth_device_samples(
            owner, managed, copy, status, &evidence, err);
    if (rc == YVEX_OK)
        rc = cuda_bandwidth_host_sample(
            (volatile unsigned long long *)host, word_count, 1ull, &median, err);
    for (sample = 0u; rc == YVEX_OK && sample < evidence.sample_count; ++sample)
        rc = cuda_bandwidth_host_sample(
            (volatile unsigned long long *)host, word_count, evidence.iterations,
            &evidence.coherent_host_elapsed_ns[sample], err);
    if (rc == YVEX_OK) {
        median = cuda_bandwidth_median(evidence.stream_elapsed_ns);
        rc = cuda_bandwidth_rate(2ull * evidence.working_set_bytes,
                                 evidence.iterations, median,
                                 &evidence.sustainable_read_bytes_per_second)
                 ? YVEX_OK : YVEX_ERR_BOUNDS;
    }
    if (rc == YVEX_OK) {
        median = cuda_bandwidth_median(evidence.copy_elapsed_ns);
        rc = cuda_bandwidth_rate(evidence.working_set_bytes,
                                 evidence.iterations, median,
                                 &evidence.sustainable_copy_bytes_per_second)
                 ? YVEX_OK : YVEX_ERR_BOUNDS;
    }
    if (rc == YVEX_OK) {
        median = cuda_bandwidth_median(evidence.coherent_host_elapsed_ns);
        rc = cuda_bandwidth_rate(
                 evidence.working_set_bytes, evidence.iterations, median,
                 &evidence.sustainable_coherent_host_bytes_per_second) &&
                     yvex_sha256_hex_valid(evidence.kernel_bundle_identity) &&
                     cuda_bandwidth_identity(&evidence)
                 ? YVEX_OK : YVEX_ERR_BOUNDS;
    }
    if (rc != YVEX_OK && err && !yvex_error_is_set(err))
        yvex_error_set(err, rc, "cuda.bandwidth",
                       "bandwidth evidence could not be derived");
    primary = err ? *err : (yvex_error){0};
    yvex_error_clear(&cleanup);
    cleanup_rc = YVEX_OK;
    if (status) {
        int release_rc = yvex_backend_tensor_release(owner, &status, &release_error);
        if (release_rc != YVEX_OK && cleanup_rc == YVEX_OK) {
            cleanup_rc = release_rc;
            cleanup = release_error;
        }
    }
    if (copy) {
        int release_rc = yvex_backend_tensor_release(owner, &copy, &release_error);
        if (release_rc != YVEX_OK && cleanup_rc == YVEX_OK) {
            cleanup_rc = release_rc;
            cleanup = release_error;
        }
    }
    if (managed) {
        int release_rc = yvex_backend_tensor_release(owner, &managed, &release_error);
        if (release_rc != YVEX_OK && cleanup_rc == YVEX_OK) {
            cleanup_rc = release_rc;
            cleanup = release_error;
        }
    }
    if (cleanup_rc != YVEX_OK) {
        atomic_store_explicit(&owner->status, YVEX_BACKEND_STATUS_FAILED,
                              memory_order_release);
        rc = cleanup_rc;
        primary = cleanup;
    }
    state->bandwidth_active = 0;
    if (rc == YVEX_OK) {
        state->bandwidth_evidence = evidence;
        state->bandwidth_ready = 1;
        *out = evidence;
        yvex_error_clear(err);
    } else if (err) {
        *err = primary;
    }
    return rc;
}

static const yvex_backend_vtable cuda_vtable = {
    cuda_close,
    cuda_memory_stats,
    cuda_device_info,
    cuda_bandwidth_probe,
    cuda_tensor_alloc,
    cuda_resident_alloc,
    cuda_tensor_reserve,
    cuda_tensor_commit,
    cuda_tensor_decommit,
    cuda_tensor_free,
    cuda_tensor_write,
    cuda_tensor_read,
    cuda_tensor_copy,
    cuda_tensor_copy_async,
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

int yvex_backend_open_cuda_impl(yvex_backend **out,
                                const char *device,
                                unsigned long long memory_limit_bytes,
                                yvex_error *err)
{
    yvex_backend *backend = NULL;
    yvex_cuda_backend_state *state = NULL;
    int device_index = 0, device_count = 0, unified = 0, managed = 0;
    int can_map_host = 0, virtual_memory_management = 0;
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
    (void)state->driver.cuDeviceGetAttribute(
        &virtual_memory_management,
        YVEX_CUDA_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT,
        state->device);
    state->virtual_memory_management =
        virtual_memory_management != 0 && state->driver.cuMemAddressReserve &&
        state->driver.cuMemAddressFree && state->driver.cuMemCreate &&
        state->driver.cuMemRelease && state->driver.cuMemMap &&
        state->driver.cuMemUnmap && state->driver.cuMemSetAccess &&
        state->driver.cuMemGetAllocationGranularity;
    backend->virtual_tensor_ready = state->virtual_memory_management;
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
    rc = cuda_execution_stream_open(backend, err);
    if (rc != YVEX_OK) goto failed;
    rc = cuda_timing_open(backend, err);
    if (rc != YVEX_OK) goto failed;
    rc = yvex_cuda_kernel_bundle_admit(backend, err);
    if (rc == YVEX_OK) {
        rc = cuda_blas_open(backend, err);
        if (rc == YVEX_OK) backend->status = YVEX_BACKEND_STATUS_READY;
        else goto failed;
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
/*
 * Create session-local CUDA state that borrows one model-owned context.
 *
 * Publishes only a failed cleanup owner when rollback cannot release its module.
 */
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
    state->virtual_memory_management = owner->virtual_memory_management;
    backend->virtual_tensor_ready = state->virtual_memory_management;
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
    rc = cuda_execution_stream_open(backend, err);
    if (rc != YVEX_OK) {
        return cuda_open_rollback(out, &backend, rc,
                                  err ? *err : (yvex_error){0}, err);
    }
    rc = cuda_timing_open(backend, err);
    if (rc != YVEX_OK) {
        return cuda_open_rollback(out, &backend, rc,
                                  err ? *err : (yvex_error){0}, err);
    }
    rc = cuda_blas_open(backend, err);
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
