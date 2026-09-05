/* Validate device execution views without importing runtime lifecycle state. */
#include <yvex/internal/device_view.h>

#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>

static int device_view_refuse(yvex_error *err, yvex_status status,
                              const char *reason)
{
    yvex_error_set(err, status, "runtime.execution.device-view", reason);
    return status;
}

int yvex_execution_device_view_validate(
    const yvex_execution_device_view *view, yvex_error *err)
{
    unsigned long long elements, bytes, offset_bytes, end;
    if (!view || view->schema_version != YVEX_EXECUTION_DEVICE_VIEW_SCHEMA_V1 ||
        view->kind > YVEX_EXECUTION_DEVICE_WORKSPACE || !view->backend || !view->tensor ||
        !yvex_backend_tensor_owned_by(view->backend, view->tensor) ||
        !view->resource_generation || !view->session_generation ||
        !view->state_generation || !view->rows || !view->columns ||
        !view->element_bytes ||
        view->materialization > YVEX_EXECUTION_MATERIALIZE_FORENSIC_FULL ||
        !yvex_sha256_hex_valid(view->runtime_model_identity) ||
        !yvex_sha256_hex_valid(view->execution_profile_identity) ||
        !yvex_core_u64_mul(view->rows, view->columns, &elements) ||
        !yvex_core_u64_mul(elements, view->element_bytes, &bytes) ||
        !yvex_core_u64_mul(view->element_offset, view->element_bytes,
                           &offset_bytes) ||
        !yvex_core_u64_add(offset_bytes, bytes, &end) ||
        end > view->tensor->bytes || view->dtype != view->tensor->dtype)
        return device_view_refuse(err, YVEX_ERR_FORMAT, "device value view is incomplete or has incompatible extent");
    yvex_error_clear(err);
    return YVEX_OK;
}
