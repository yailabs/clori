/*
 * Discover the CUDA Driver API and project immutable device admission facts.
 *
 * CUDA failures preserve the originating Driver status and typed operation context. This owner
 * exposes typed facts only at its admitted subsystem stage.
 */
#include "src/backend/cuda/private.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static int load_symbol(void *library, void **slot, const char *name, yvex_error *err)
{
    *slot = dlsym(library, name);
    if (!*slot) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "cuda.driver_load",
                        "CUDA driver symbol unavailable: %s", name);
        return YVEX_ERR_UNSUPPORTED;
    }
    return YVEX_OK;
}

static void load_optional_symbol(void *library,
                                 void **slot,
                                 const char *name,
                                 const char *fallback)
{
    *slot = dlsym(library, name);
    if (!*slot && fallback) {
        *slot = dlsym(library, fallback);
    }
}
#define YVEX_LOAD_REQUIRED(driver, field) \
    do { \
        if (load_symbol((driver)->library, (void **)&((driver)->field), #field, err) != YVEX_OK) { \
            return YVEX_ERR_UNSUPPORTED; \
        } \
    } while (0)

int yvex_cuda_driver_load(yvex_cuda_driver *driver, yvex_error *err)
{
    if (!driver) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.driver_load", "driver is required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(driver, 0, sizeof(*driver));
    driver->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!driver->library) {
        driver->library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!driver->library) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.driver_load",
                       "CUDA driver library is not available");
        return YVEX_ERR_UNSUPPORTED;
    }
    YVEX_LOAD_REQUIRED(driver, cuInit);
    YVEX_LOAD_REQUIRED(driver, cuDriverGetVersion);
    YVEX_LOAD_REQUIRED(driver, cuDeviceGetCount);
    YVEX_LOAD_REQUIRED(driver, cuDeviceGet);
    YVEX_LOAD_REQUIRED(driver, cuDeviceGetName);
    YVEX_LOAD_REQUIRED(driver, cuDeviceComputeCapability);
    YVEX_LOAD_REQUIRED(driver, cuDeviceTotalMem_v2);
    YVEX_LOAD_REQUIRED(driver, cuDeviceGetAttribute);
    YVEX_LOAD_REQUIRED(driver, cuCtxCreate_v2);
    YVEX_LOAD_REQUIRED(driver, cuCtxDestroy_v2);
    YVEX_LOAD_REQUIRED(driver, cuCtxSetCurrent);
    YVEX_LOAD_REQUIRED(driver, cuCtxSynchronize);
    YVEX_LOAD_REQUIRED(driver, cuMemGetInfo_v2);
    YVEX_LOAD_REQUIRED(driver, cuMemAlloc_v2);
    YVEX_LOAD_REQUIRED(driver, cuMemAllocManaged);
    load_optional_symbol(driver->library, (void **)&driver->cuMemPrefetchAsync_v2,
                         "cuMemPrefetchAsync_v2", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemAdvise_v2,
                         "cuMemAdvise_v2", NULL);
    YVEX_LOAD_REQUIRED(driver, cuMemFree_v2);
    YVEX_LOAD_REQUIRED(driver, cuMemsetD8_v2);
    YVEX_LOAD_REQUIRED(driver, cuMemcpyHtoD_v2);
    YVEX_LOAD_REQUIRED(driver, cuMemcpyDtoH_v2);
    YVEX_LOAD_REQUIRED(driver, cuMemcpyDtoD_v2);
    YVEX_LOAD_REQUIRED(driver, cuModuleLoadData);
    YVEX_LOAD_REQUIRED(driver, cuModuleGetFunction);
    YVEX_LOAD_REQUIRED(driver, cuModuleUnload);
    YVEX_LOAD_REQUIRED(driver, cuLaunchKernel);
    YVEX_LOAD_REQUIRED(driver, cuGetErrorName);
    YVEX_LOAD_REQUIRED(driver, cuGetErrorString);
    load_optional_symbol(driver->library, (void **)&driver->cuStreamCreate,
                         "cuStreamCreate", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuStreamDestroy_v2,
                         "cuStreamDestroy_v2", "cuStreamDestroy");
    load_optional_symbol(driver->library, (void **)&driver->cuStreamSynchronize,
                         "cuStreamSynchronize", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuStreamBeginCapture_v2,
                         "cuStreamBeginCapture_v2", "cuStreamBeginCapture");
    load_optional_symbol(driver->library, (void **)&driver->cuStreamEndCapture,
                         "cuStreamEndCapture", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuGraphGetNodes,
                         "cuGraphGetNodes", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuGraphGetEdges_v2,
                         "cuGraphGetEdges_v2", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuGraphNodeGetType,
                         "cuGraphNodeGetType", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuGraphInstantiateWithFlags,
                         "cuGraphInstantiateWithFlags", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuGraphUpload,
                         "cuGraphUpload", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuGraphLaunch,
                         "cuGraphLaunch", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuGraphExecUpdate_v2,
                         "cuGraphExecUpdate_v2", NULL);
    load_optional_symbol(
        driver->library, (void **)&driver->cuGraphExecKernelNodeSetParams_v2,
        "cuGraphExecKernelNodeSetParams_v2", "cuGraphExecKernelNodeSetParams");
    load_optional_symbol(driver->library, (void **)&driver->cuGraphExecDestroy,
                         "cuGraphExecDestroy", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuGraphDestroy,
                         "cuGraphDestroy", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemAllocAsync,
                         "cuMemAllocAsync", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemFreeAsync,
                         "cuMemFreeAsync", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemAddressReserve,
                         "cuMemAddressReserve", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemAddressFree,
                         "cuMemAddressFree", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemCreate,
                         "cuMemCreate", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemRelease,
                         "cuMemRelease", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemMap,
                         "cuMemMap", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemUnmap,
                         "cuMemUnmap", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemSetAccess,
                         "cuMemSetAccess", NULL);
    load_optional_symbol(
        driver->library,
        (void **)&driver->cuMemGetAllocationGranularity,
        "cuMemGetAllocationGranularity", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemcpyHtoDAsync_v2,
                         "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    load_optional_symbol(driver->library, (void **)&driver->cuMemcpyDtoDAsync_v2,
                         "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoDAsync");
    load_optional_symbol(driver->library, (void **)&driver->cuMemcpyDtoHAsync_v2,
                         "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
    load_optional_symbol(driver->library, (void **)&driver->cuMemsetD8Async,
                         "cuMemsetD8Async", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemHostAlloc,
                         "cuMemHostAlloc", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemFreeHost,
                         "cuMemFreeHost", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuMemHostRegister_v2,
                         "cuMemHostRegister_v2", "cuMemHostRegister");
    load_optional_symbol(driver->library, (void **)&driver->cuMemHostGetDevicePointer_v2,
                         "cuMemHostGetDevicePointer_v2", "cuMemHostGetDevicePointer");
    load_optional_symbol(driver->library, (void **)&driver->cuMemHostUnregister,
                         "cuMemHostUnregister", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuEventCreate,
                         "cuEventCreate", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuEventRecord,
                         "cuEventRecord", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuEventSynchronize,
                         "cuEventSynchronize", NULL);
    load_optional_symbol(driver->library, (void **)&driver->cuEventElapsedTime_v2,
                         "cuEventElapsedTime_v2", "cuEventElapsedTime");
    load_optional_symbol(driver->library, (void **)&driver->cuEventDestroy_v2,
                         "cuEventDestroy_v2", "cuEventDestroy");
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_cuda_driver_unload(yvex_cuda_driver *driver)
{
    if (!driver) {
        return;
    }
    if (driver->library) {
        dlclose(driver->library);
    }
    memset(driver, 0, sizeof(*driver));
}

yvex_cuda_backend_state *yvex_cuda_state(const yvex_backend *backend)
{
    return backend ? (yvex_cuda_backend_state *)backend->impl : NULL;
}

int yvex_cuda_set_current(const yvex_backend *backend, const char *where, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    if (!state) {
        yvex_error_set(err, YVEX_ERR_STATE, where ? where : "cuda.set_current",
                       "CUDA backend state is missing");
        return YVEX_ERR_STATE;
    }
    return yvex_cuda_status(&state->driver, state->driver.cuCtxSetCurrent(state->context),
                            where ? where : "cuda.set_current", err);
}

int yvex_cuda_refresh_memory_info(yvex_backend *backend, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    int rc;
    if (!backend || !state) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.memory_info",
                       "CUDA backend is required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = backend_dispatch_admit(backend, "cuda.memory_info", err);
    if (rc != YVEX_OK) return rc;
    rc = yvex_cuda_set_current(backend, "cuda.memory_info", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuMemGetInfo_v2(&free_bytes, &total_bytes),
                          "cuda.memory_info", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    backend->device_info.free_memory_bytes = (unsigned long long)free_bytes;
    backend->device_info.total_memory_bytes = (unsigned long long)total_bytes;
    yvex_error_clear(err);
    return YVEX_OK;
}

CUdeviceptr yvex_cuda_tensor_ptr(const yvex_device_tensor *tensor)
{
    return (CUdeviceptr)(size_t)tensor->data;
}
