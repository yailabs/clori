/*
 * Own imported-host and managed CUDA weight residency from allocation through device publication.
 * A managed range is not executable residency until its initialized bytes have migrated and the
 * execution stream has synchronized; failure leaves ownership with the caller for exact cleanup.
 */
#include "src/backend/cuda/private.h"

#include <stdint.h>
#include <stdlib.h>

int yvex_cuda_resident_map_supported(const yvex_backend *backend)
{
    const yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    return state && !state->context_borrowed && backend->device_info.unified_addressing &&
           backend->pageable_memory_access && backend->pageable_uses_host_page_tables &&
           state->driver.cuMemAdvise_v2;
}

int yvex_cuda_resident_map_readonly(yvex_backend *backend,
                                    const yvex_backend_tensor_desc *desc,
                                    const unsigned char *host,
                                    yvex_device_tensor **out,
                                    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_device_tensor *tensor = NULL;
    CUmemLocation location = {0};
    unsigned int index;
    int rc;
    if (out) *out = NULL;
    if (!state || !desc || !host || !out || !desc->name || !desc->rank ||
        desc->rank > YVEX_TENSOR_MAX_DIMS || !desc->bytes ||
        desc->bytes > (unsigned long long)SIZE_MAX ||
        !yvex_cuda_resident_map_supported(backend)) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.residency.map",
                       "immutable pageable CUDA mapping is unavailable");
        return YVEX_ERR_UNSUPPORTED;
    }
    for (index = 0u; index < desc->rank; ++index) {
        if (!desc->dims[index]) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.residency.map",
                           "mapped residency dimensions must be non-zero");
            return YVEX_ERR_INVALID_ARG;
        }
    }
    rc = yvex_cuda_set_current(backend, "cuda.residency.map", err);
    location.type = YVEX_CUDA_MEM_LOCATION_DEVICE;
    location.id = state->device_index;
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemAdvise_v2((CUdeviceptr)(uintptr_t)host, (size_t)desc->bytes,
                                         YVEX_CUDA_MEM_ADVISE_SET_READ_MOSTLY, location),
            "cuda.residency.map.read-mostly", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemAdvise_v2(
                (CUdeviceptr)(uintptr_t)host, (size_t)desc->bytes,
                YVEX_CUDA_MEM_ADVISE_SET_PREFERRED_LOCATION, location),
            "cuda.residency.map.preferred-location", err);
    if (rc != YVEX_OK) return rc;
    tensor = (yvex_device_tensor *)calloc(1u, sizeof(*tensor));
    if (tensor) tensor->name = yvex_core_strdup(desc->name);
    if (!tensor || !tensor->name) {
        free(tensor);
        yvex_error_set(err, YVEX_ERR_NOMEM, "cuda.residency.map",
                       "mapped residency metadata allocation failed");
        return YVEX_ERR_NOMEM;
    }
    tensor->owner = backend;
    tensor->owner_id = backend->tensor_id_next++;
    tensor->dtype = desc->dtype;
    tensor->rank = desc->rank;
    for (index = 0u; index < desc->rank; ++index) tensor->dims[index] = desc->dims[index];
    tensor->bytes = desc->bytes;
    tensor->data = (unsigned char *)(uintptr_t)host;
    tensor->host_data = tensor->data;
    tensor->host_accessible = 1;
    tensor->borrowed_host = 1;
    tensor->is_written = 1;
    *out = tensor;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_resident_alloc(yvex_backend *backend,
                             const yvex_backend_tensor_desc *desc,
                             yvex_device_tensor **out,
                             unsigned char **host,
                             yvex_error *err)
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
        (!imported && (!backend->device_info.managed_memory || !state->driver.cuMemAllocManaged))) {
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

int yvex_cuda_resident_prefetch(yvex_backend *backend,
                                yvex_device_tensor *tensor,
                                unsigned long long *prefetched_bytes,
                                yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUmemLocation location = {0};
    CUstream stream;
    int rc;

    if (prefetched_bytes) *prefetched_bytes = 0ull;
    if (!state || !tensor || !prefetched_bytes || !backend_tensor_owner_is(backend, tensor) ||
        state->context_borrowed || !tensor->host_accessible ||
        tensor->host_data != tensor->data || !state->driver.cuMemPrefetchAsync_v2 ||
        !(stream = yvex_cuda_launch_stream(backend)) || !state->driver.cuStreamSynchronize) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.residency.prefetch",
                       "one owner-local managed range and asynchronous prefetch are required");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = yvex_cuda_set_current(backend, "cuda.residency.prefetch", err);
    if (rc == YVEX_OK && getenv("YVEX_TEST_CUDA_MANAGED_PREFETCH_FAILURE")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.residency.prefetch",
                       "injected managed prefetch failure");
        rc = YVEX_ERR_BACKEND;
    }
    location.type = YVEX_CUDA_MEM_LOCATION_DEVICE;
    location.id = state->device;
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemPrefetchAsync_v2(
                yvex_cuda_tensor_ptr(tensor), (size_t)tensor->bytes, location, 0u, stream),
            "cuda.residency.prefetch", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver, state->driver.cuStreamSynchronize(stream),
                              "cuda.residency.prefetch.sync", err);
    if (rc != YVEX_OK) return rc;
    tensor->is_written = 1;
    *prefetched_bytes = tensor->bytes;
    yvex_error_clear(err);
    return YVEX_OK;
}
