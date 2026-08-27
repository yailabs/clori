/* Runtime failure cause and recovery remain one typed subsystem contract. */
#include "src/runtime/private.h"

#include <string.h>
#include <yvex/internal/core.h>

typedef struct {
    yvex_runtime_failure_origin origin;
    yvex_runtime_recovery_action recovery;
} runtime_failure_disposition;

static runtime_failure_disposition runtime_failure_default(
    yvex_model_engine_failure_code code)
{
    runtime_failure_disposition disposition = {
        YVEX_RUNTIME_FAILURE_ORIGIN_NONE, YVEX_RUNTIME_RECOVERY_NONE};
    switch (code) {
        case YVEX_MODEL_ENGINE_FAILURE_INVALID_ARGUMENT:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_EXTERNAL_REQUEST;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_REFUSE_REQUEST;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_ADAPTER:
        case YVEX_MODEL_ENGINE_FAILURE_DESCRIPTOR:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_CAPABILITY;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_REFUSE_ENGINE_OPEN;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_BINDING:
        case YVEX_MODEL_ENGINE_FAILURE_ARTIFACT:
        case YVEX_MODEL_ENGINE_FAILURE_IDENTITY:
        case YVEX_MODEL_ENGINE_FAILURE_MATERIALIZATION:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTEGRITY;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_REFUSE_ENGINE_OPEN;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_GRAPH:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_SEQUENCE;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_ABORT_TRANSACTION;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_BACKEND:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_BACKEND;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_RETRY_EQUIVALENT;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_DRIFT:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTEGRITY;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_INVALIDATE_SEQUENCE;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_BUSY:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_ENGINE;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_REFUSE_REQUEST;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_CANCELLED:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_SEQUENCE;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_ABORT_TRANSACTION;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_ALLOCATION:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_RESOURCE;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_PREPARE_OR_EVICT;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_CLEANUP:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTERNAL;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_ABORT_TRANSACTION;
            break;
        case YVEX_MODEL_ENGINE_FAILURE_NONE:
        default:
            break;
    }
    return disposition;
}

void yvex_runtime_private_failure_record(yvex_model_engine_failure *failure,
                                         yvex_model_engine_failure_code code,
                                         const char *field, unsigned long long expected,
                                         unsigned long long actual, const char *reason) {
    if (failure) {
        runtime_failure_disposition disposition = runtime_failure_default(code);
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->origin = disposition.origin;
        failure->recovery = disposition.recovery;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
        if (field) yvex_core_text_copy(failure->field, sizeof(failure->field), field);
    }
}

static void runtime_failure_record_as(
    yvex_model_engine_failure *failure, yvex_model_engine_failure_code code,
    yvex_runtime_failure_origin origin, yvex_runtime_recovery_action recovery,
    const char *field, unsigned long long expected,
    unsigned long long actual, const char *reason)
{
    yvex_runtime_private_failure_record(
        failure, code, field, expected, actual, reason);
    if (failure) {
        failure->origin = origin;
        failure->recovery = recovery;
    }
}

int yvex_runtime_private_reject(yvex_model_engine_failure *failure,
                                yvex_model_engine_failure_code code,
                                const char *field, unsigned long long expected,
                                unsigned long long actual, const char *reason,
                                yvex_error *err, yvex_status status) {
    yvex_runtime_private_failure_record(failure, code, field, expected, actual, reason);
    yvex_error_set(err, status, "runtime.model", reason);
    return status;
}

int yvex_runtime_private_reject_as(
    yvex_model_engine_failure *failure, yvex_model_engine_failure_code code,
    yvex_runtime_failure_origin origin, yvex_runtime_recovery_action recovery,
    const char *field, unsigned long long expected,
    unsigned long long actual, const char *reason, yvex_error *err,
    yvex_status status)
{
    runtime_failure_record_as(
        failure, code, origin, recovery, field, expected, actual, reason);
    yvex_error_set(err, status, "runtime.model", reason);
    return status;
}

int yvex_runtime_private_success(yvex_error *err) {
    yvex_error_clear(err);
    return YVEX_OK;
}
