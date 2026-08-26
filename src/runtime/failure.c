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

static runtime_failure_disposition runtime_refusal_disposition(
    yvex_runtime_private_refusal_id id, yvex_model_engine_failure_code code)
{
    runtime_failure_disposition disposition = runtime_failure_default(code);
    switch (id) {
        case YVEX_RUNTIME_REFUSE_MODEL_LOCK_UNAVAILABLE:
        case YVEX_RUNTIME_REFUSE_DRIFT_COUNTER:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTERNAL;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_DRAIN_ENGINE;
            break;
        case YVEX_RUNTIME_REFUSE_MODEL_LOCK_INITIALIZATION:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTERNAL;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_REFUSE_ENGINE_OPEN;
            break;
        case YVEX_RUNTIME_REFUSE_HOST_RESIDENCY:
        case YVEX_RUNTIME_REFUSE_WORKSPACE_BUDGET:
        case YVEX_RUNTIME_REFUSE_DEVICE_WORKSPACE_BUDGET:
        case YVEX_RUNTIME_REFUSE_OPEN_RESIDENCY:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_RESOURCE;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_PREPARE_OR_EVICT;
            break;
        case YVEX_RUNTIME_REFUSE_CUDA_EAGER:
        case YVEX_RUNTIME_REFUSE_DEVICE_CAPABILITY:
        case YVEX_RUNTIME_REFUSE_CUDA_CAPABILITY:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_CAPABILITY;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_RETRY_EQUIVALENT;
            break;
        case YVEX_RUNTIME_REFUSE_SESSION_LOCK_INITIALIZATION:
        case YVEX_RUNTIME_REFUSE_SESSION_CONDITION_INITIALIZATION:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTERNAL;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_REFUSE_REQUEST;
            break;
        case YVEX_RUNTIME_REFUSE_WORKSPACE_IDENTITY:
        case YVEX_RUNTIME_REFUSE_SESSION_RESOURCE_INJECTION:
        case YVEX_RUNTIME_REFUSE_WORKSPACE_CAPABILITY_INJECTION:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTERNAL;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_ABORT_TRANSACTION;
            break;
        case YVEX_RUNTIME_REFUSE_SESSION_INVALIDATED:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTEGRITY;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_INVALIDATE_SEQUENCE;
            break;
        case YVEX_RUNTIME_REFUSE_ARTIFACT_DRIFT:
        case YVEX_RUNTIME_REFUSE_OPEN_DRIFT:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTEGRITY;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_DRAIN_ENGINE;
            break;
        case YVEX_RUNTIME_REFUSE_ADAPTER_CAPABILITY_STALE:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_INTEGRITY;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_REFUSE_ENGINE_OPEN;
            break;
        case YVEX_RUNTIME_REFUSE_OPEN_SEAL:
            disposition.origin = YVEX_RUNTIME_FAILURE_ORIGIN_EXTERNAL_REQUEST;
            disposition.recovery = YVEX_RUNTIME_RECOVERY_REFUSE_ENGINE_OPEN;
            break;
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

int yvex_runtime_private_reject(yvex_model_engine_failure *failure,
                                yvex_model_engine_failure_code code,
                                const char *field, unsigned long long expected,
                                unsigned long long actual, const char *reason,
                                yvex_error *err, yvex_status status) {
    yvex_runtime_private_failure_record(failure, code, field, expected, actual, reason);
    yvex_error_set(err, status, "runtime.model", reason);
    return status;
}
typedef struct {
    yvex_model_engine_failure_code code;
    yvex_status status;
    const char *field, *reason;
} runtime_refusal_spec;
/* Ordered with yvex_runtime_private_refusal_id so one typed row owns each stable refusal contract. */
static const runtime_refusal_spec runtime_refusals[] = {
    {YVEX_MODEL_ENGINE_FAILURE_CLEANUP, YVEX_ERR_STATE, "model-lifecycle-lock", "model lock unavailable"},
    {YVEX_MODEL_ENGINE_FAILURE_BUSY, YVEX_ERR_STATE, "runtime-model-draining", "runtime model is invalid or draining"},
    {YVEX_MODEL_ENGINE_FAILURE_IDENTITY, YVEX_ERR_FORMAT, "artifact-identity", "artifact identity differs"},
    {YVEX_MODEL_ENGINE_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "request", "model request is incomplete"},
    {YVEX_MODEL_ENGINE_FAILURE_ADAPTER, YVEX_ERR_UNSUPPORTED, "family-adapter", "family adapter is incomplete"},
    {YVEX_MODEL_ENGINE_FAILURE_ALLOCATION, YVEX_ERR_NOMEM, "allocation", "runtime model allocation failed"},
    {YVEX_MODEL_ENGINE_FAILURE_CLEANUP, YVEX_ERR_STATE, "model-lifecycle-lock", "model lock init failed"},
    {YVEX_MODEL_ENGINE_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "runtime-model", "runtime model is required"},
    {YVEX_MODEL_ENGINE_FAILURE_BUSY, YVEX_ERR_STATE, "runtime-model-draining", "model is unsealed or draining"},
    {YVEX_MODEL_ENGINE_FAILURE_DRIFT, YVEX_ERR_BOUNDS, "drift-check-counter", "drift counter overflowed"},
    {YVEX_MODEL_ENGINE_FAILURE_MATERIALIZATION, YVEX_ERR_STATE, "host-residency", "host residency is required"},
    {YVEX_MODEL_ENGINE_FAILURE_BACKEND, YVEX_ERR_UNSUPPORTED, "cuda-eager-capability",
     "exact CUDA eager kernels, device, residency, and pinned workspace are required"},
    {YVEX_MODEL_ENGINE_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "session-request",
     "valid model and backend session request are required"},
    {YVEX_MODEL_ENGINE_FAILURE_ALLOCATION, YVEX_ERR_NOMEM, "session-allocation", "runtime session allocation failed"},
    {YVEX_MODEL_ENGINE_FAILURE_CLEANUP, YVEX_ERR_STATE, "session-lifecycle-lock", "session lock init failed"},
    {YVEX_MODEL_ENGINE_FAILURE_CLEANUP, YVEX_ERR_STATE, "session-idle-condition", "session condition init failed"},
    {YVEX_MODEL_ENGINE_FAILURE_BACKEND, YVEX_ERR_STATE, "workspace-identity", "workspace identity failed"},
    {YVEX_MODEL_ENGINE_FAILURE_BACKEND, YVEX_ERR_STATE, "session-open-after-resources", "injected resource failure"},
    {YVEX_MODEL_ENGINE_FAILURE_BUSY, YVEX_ERR_STATE, "runtime-model-draining", "model began draining"},
    {YVEX_MODEL_ENGINE_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG,
     "attention-workspace-plan", "open CUDA session and an exact execution mode are required"},
    {YVEX_MODEL_ENGINE_FAILURE_BUSY, YVEX_ERR_STATE, "session-lock", "runtime session workspace lock is unavailable"},
    {YVEX_MODEL_ENGINE_FAILURE_BUSY, YVEX_ERR_STATE, "session-state", "idle open session is required"},
    {YVEX_MODEL_ENGINE_FAILURE_BUSY, YVEX_ERR_STATE, "attention-state", "sealed idle state is required"},
    {YVEX_MODEL_ENGINE_FAILURE_BUSY, YVEX_ERR_STATE, "host-workspace", "host workspace is already sealed"},
    {YVEX_MODEL_ENGINE_FAILURE_GRAPH, YVEX_ERR_BOUNDS, "host-workspace-budget",
     "descriptor-bucket CUDA staging exceeds the session host budget"},
    {YVEX_MODEL_ENGINE_FAILURE_BACKEND, YVEX_ERR_STATE, "workspace-capability-publication",
     "injected runtime workspace capability publication failure"},
    {YVEX_MODEL_ENGINE_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "session", "open runtime session is required"},
    {YVEX_MODEL_ENGINE_FAILURE_DRIFT, YVEX_ERR_STATE, "session-invalidated", "artifact drift invalidated session"},
    {YVEX_MODEL_ENGINE_FAILURE_BUSY, YVEX_ERR_STATE, "session-closing", "runtime session is closing"},
    {YVEX_MODEL_ENGINE_FAILURE_BUSY, YVEX_ERR_STATE, "session-busy", "runtime session already owns an execution"},
    {YVEX_MODEL_ENGINE_FAILURE_CANCELLED, YVEX_ERR_CANCELLED, "session-cancelled", "session cancelled"},
    {YVEX_MODEL_ENGINE_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-binding-admission", "binding lacks artifact"},
    {YVEX_MODEL_ENGINE_FAILURE_ADAPTER, YVEX_ERR_FORMAT, "execution-capabilities",
     "family adapter has no typed execution capability contract"},
    {YVEX_MODEL_ENGINE_FAILURE_ADAPTER, YVEX_ERR_FORMAT, "execution-capabilities",
     "bound family execution capability contract is stale or promoted"},
    {YVEX_MODEL_ENGINE_FAILURE_DRIFT, YVEX_ERR_STATE, "artifact-snapshot", "artifact drift invalidated model"},
    {YVEX_MODEL_ENGINE_FAILURE_BACKEND, YVEX_ERR_UNSUPPORTED, "device-capability", "device admission failed"},
    {YVEX_MODEL_ENGINE_FAILURE_BACKEND, YVEX_ERR_UNSUPPORTED, "cuda-capability", "CUDA admission failed"},
    {YVEX_MODEL_ENGINE_FAILURE_CLEANUP, YVEX_ERR_STATE, "session-open-cleanup", "session cleanup failed"},
    {YVEX_MODEL_ENGINE_FAILURE_BACKEND, YVEX_ERR_BOUNDS, "device-workspace-budget",
     "descriptor-bucket CUDA workspace exceeds the session device budget"},
    {YVEX_MODEL_ENGINE_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "cleanup-lease",
     "empty cleanup lease, model request, and borrowed outputs are required"},
    {YVEX_MODEL_ENGINE_FAILURE_ALLOCATION, YVEX_ERR_NOMEM, "cleanup-lease", "lease allocation failed"},
    {YVEX_MODEL_ENGINE_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG,
     "cleanup-lease-session", "model-owning cleanup lease and session request are required"},
    {YVEX_MODEL_ENGINE_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-binding", "runtime binding open failed"},
    {YVEX_MODEL_ENGINE_FAILURE_ADAPTER, YVEX_ERR_FORMAT, "family-adapter-id", "family adapter identity differs"},
    {YVEX_MODEL_ENGINE_FAILURE_IDENTITY, YVEX_ERR_FORMAT, "logical-transform-identity",
     "runtime binding logical Transformation IR identity is stale"},
    {YVEX_MODEL_ENGINE_FAILURE_ALLOCATION, YVEX_ERR_BOUNDS, "model-host-budget",
     "configured model host budget cannot preserve the reserve after model residency"},
    {YVEX_MODEL_ENGINE_FAILURE_ALLOCATION, YVEX_ERR_BOUNDS, "process-memory-capacity",
     "process memory control cannot preserve the required system reserve after model residency"},
    {YVEX_MODEL_ENGINE_FAILURE_ALLOCATION, YVEX_ERR_BOUNDS, "system-memory-capacity",
     "available system memory cannot preserve the required system reserve after model residency"},
    {YVEX_MODEL_ENGINE_FAILURE_ALLOCATION, YVEX_ERR_BOUNDS,
     "startup-execution-capacity",
     "startup workload peak cannot preserve the required system reserve before model residency"},
    {YVEX_MODEL_ENGINE_FAILURE_ARTIFACT, YVEX_ERR_FORMAT, "artifact-open", "artifact admission failed"},
    {YVEX_MODEL_ENGINE_FAILURE_MATERIALIZATION, YVEX_ERR_FORMAT, "runtime-materialization",
     "runtime binding materialization could not be reopened"},
    {YVEX_MODEL_ENGINE_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-import",
     "runtime binding import did not reconstruct sealed runtime facts"},
    {YVEX_MODEL_ENGINE_FAILURE_DESCRIPTOR, YVEX_ERR_FORMAT, "physical-execution-ir",
     "runtime physical execution decisions could not be compiled"},
    {YVEX_MODEL_ENGINE_FAILURE_IDENTITY, YVEX_ERR_FORMAT, "imported-identity", "import identity is invalid"},
    {YVEX_MODEL_ENGINE_FAILURE_DESCRIPTOR, YVEX_ERR_FORMAT, "tokenizer-plan",
     "artifact tokenizer could not be admitted and bound to the runtime model"},
    {YVEX_MODEL_ENGINE_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-model-seal", "model seal was cancelled"},
    {YVEX_MODEL_ENGINE_FAILURE_ADAPTER, YVEX_ERR_FORMAT, "execution-capabilities",
     "runtime execution capability contract could not be admitted"},
    {YVEX_MODEL_ENGINE_FAILURE_MATERIALIZATION, YVEX_ERR_FORMAT, "model-residency",
     "runtime full-model residency could not be sealed"},
    {YVEX_MODEL_ENGINE_FAILURE_MATERIALIZATION, YVEX_ERR_FORMAT,
     "model-residency-completeness",
     "runtime full-model residency is not complete for every descriptor binding"},
    {YVEX_MODEL_ENGINE_FAILURE_DRIFT, YVEX_ERR_STATE, "artifact-snapshot", "artifact drifted before publication"}
};
_Static_assert(sizeof(runtime_refusals) / sizeof(runtime_refusals[0]) == YVEX_RUNTIME_REFUSE_COUNT,
               "runtime refusal catalog must cover every identity");

static int runtime_refuse_as(yvex_model_engine_failure *failure,
                             yvex_runtime_private_refusal_id id,
                             unsigned long long expected,
                             unsigned long long actual,
                             yvex_status status, yvex_error *err) {
    const runtime_refusal_spec *spec;
    runtime_failure_disposition disposition;
    int rc;
    if ((unsigned int)id >= YVEX_RUNTIME_REFUSE_COUNT) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.refusal",
                       "runtime refusal identity is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    spec = &runtime_refusals[id];
    disposition = runtime_refusal_disposition(id, spec->code);
    rc = yvex_runtime_private_reject(failure, spec->code, spec->field, expected, actual,
                                     spec->reason, err,
                                     status == YVEX_OK ? spec->status : status);
    if (failure) {
        failure->origin = disposition.origin;
        failure->recovery = disposition.recovery;
    }
    return rc;
}

int yvex_runtime_private_refuse(yvex_model_engine_failure *failure, yvex_runtime_private_refusal_id id,
    unsigned long long expected, unsigned long long actual, yvex_error *err) {
    return runtime_refuse_as(failure, id, expected, actual, YVEX_OK, err);
}

int yvex_runtime_private_success(yvex_error *err) {
    yvex_error_clear(err);
    return YVEX_OK;
}
