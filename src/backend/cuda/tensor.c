/*
 * Own CUDA tensor allocation, transfer, copy, accounting, and release.
 *
 * Writes become visible only after synchronization; failed release preserves tensor ownership and
 * counters; unpublished allocations are cleaned. Tensor movement is not CUDA graph or generation
 * support.
 */
#include "src/backend/cuda/private.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int yvex_cuda_tensor_alloc(yvex_backend *backend,
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

int yvex_cuda_tensor_free(yvex_backend *backend,
                          yvex_device_tensor *tensor,
                          yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
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
    pointer = yvex_cuda_tensor_ptr(tensor);
    if (tensor->host_data && state->registered_host == tensor->host_data &&
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

int yvex_cuda_tensor_write(yvex_backend *backend,
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

int yvex_cuda_tensor_read(yvex_backend *backend,
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
int yvex_cuda_tensor_copy(yvex_backend *backend,
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
