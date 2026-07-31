/*
 * Translate CUDA Driver status into stable typed YVEX failures.
 *
 * CUDA failures preserve the originating Driver status and typed operation context. This owner
 * exposes typed facts only at its admitted subsystem stage.
 */

#include "src/backend/cuda/private.h"
#include <stdio.h>

int yvex_cuda_status(const yvex_cuda_driver *driver,
                     CUresult code,
                     const char *where,
                     yvex_error *err)
{
    const char *name = NULL;
    const char *message = NULL;
    yvex_status status = YVEX_ERR_BACKEND;

    if (code == YVEX_CUDA_SUCCESS) {
        yvex_error_clear(err);
        return YVEX_OK;
    }

    if (driver && driver->cuGetErrorName) {
        (void)driver->cuGetErrorName(code, &name);
    }
    if (driver && driver->cuGetErrorString) {
        (void)driver->cuGetErrorString(code, &message);
    }
    if (!name) {
        name = "CUDA_ERROR_UNKNOWN";
    }
    if (!message) {
        message = "no CUDA error string available";
    }

    switch (code) {
    case YVEX_CUDA_ERROR_OUT_OF_MEMORY:
        status = YVEX_ERR_NOMEM;
        break;
    case YVEX_CUDA_ERROR_NO_DEVICE:
    case YVEX_CUDA_ERROR_NOT_SUPPORTED:
        status = YVEX_ERR_UNSUPPORTED;
        break;
    case YVEX_CUDA_ERROR_INVALID_VALUE:
        status = YVEX_ERR_INVALID_ARG;
        break;
    default:
        status = YVEX_ERR_BACKEND;
        break;
    }

    yvex_error_setf(err, status, where ? where : "cuda",
                    "CUDA error %d (%s): %s", code, name, message);
    return status;
}
