/*
 * A runtime model owns the authenticated artifact, immutable execution plans, encoded weights,
 * and reusable model resources. Sessions borrow that model but own all mutable sequence state;
 * opening or closing one session cannot alter another.
 * Construction publishes nothing until artifact admission, materialization, backend preparation,
 * and family binding all agree. Cleanup runs in reverse dependency order and retains a failed owner
 * so callers can retry without losing the resource that still needs release.
 */
#include "src/runtime/private.h"
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/logits.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/artifact.h>
void yvex_runtime_private_failure_record(yvex_runtime_model_failure *failure,
                                         yvex_runtime_model_failure_code code,
                                         const char *field, unsigned long long expected,
                                         unsigned long long actual, const char *reason) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
        if (field) yvex_core_text_copy(failure->field, sizeof(failure->field), field);
    }
}

int yvex_runtime_private_reject(yvex_runtime_model_failure *failure,
                                yvex_runtime_model_failure_code code,
                                const char *field, unsigned long long expected,
                                unsigned long long actual, const char *reason,
                                yvex_error *err, yvex_status status) {
    yvex_runtime_private_failure_record(failure, code, field, expected, actual, reason);
    yvex_error_set(err, status, "runtime.model", reason);
    return status;
}
typedef struct {
    yvex_runtime_model_failure_code code;
    yvex_status status;
    const char *field, *reason;
} runtime_refusal_spec;
/* Ordered with yvex_runtime_private_refusal_id so one typed row owns each stable refusal contract. */
static const runtime_refusal_spec runtime_refusals[] = {
    {YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, YVEX_ERR_STATE, "model-lifecycle-lock", "model lock unavailable"},
    {YVEX_RUNTIME_MODEL_FAILURE_BUSY, YVEX_ERR_STATE, "runtime-model-draining", "runtime model is invalid or draining"},
    {YVEX_RUNTIME_MODEL_FAILURE_IDENTITY, YVEX_ERR_FORMAT, "artifact-identity", "artifact identity differs"},
    {YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "request", "model request is incomplete"},
    {YVEX_RUNTIME_MODEL_FAILURE_ADAPTER, YVEX_ERR_UNSUPPORTED, "family-adapter", "family adapter is incomplete"},
    {YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION, YVEX_ERR_NOMEM, "allocation", "runtime model allocation failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, YVEX_ERR_STATE, "model-lifecycle-lock", "model lock init failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "runtime-model", "runtime model is required"},
    {YVEX_RUNTIME_MODEL_FAILURE_BUSY, YVEX_ERR_STATE, "runtime-model-draining", "model is unsealed or draining"},
    {YVEX_RUNTIME_MODEL_FAILURE_DRIFT, YVEX_ERR_BOUNDS, "drift-check-counter", "drift counter overflowed"},
    {YVEX_RUNTIME_MODEL_FAILURE_MATERIALIZATION, YVEX_ERR_STATE, "host-residency", "host residency is required"},
    {YVEX_RUNTIME_MODEL_FAILURE_BACKEND, YVEX_ERR_UNSUPPORTED, "cuda-eager-capability",
     "exact CUDA eager kernels, device, residency, and pinned workspace are required"},
    {YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "session-request",
     "valid model and backend session request are required"},
    {YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION, YVEX_ERR_NOMEM, "session-allocation", "runtime session allocation failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, YVEX_ERR_STATE, "session-lifecycle-lock", "session lock init failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, YVEX_ERR_STATE, "session-idle-condition", "session condition init failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_BACKEND, YVEX_ERR_STATE, "workspace-identity", "workspace identity failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_BACKEND, YVEX_ERR_STATE, "session-open-after-resources", "injected resource failure"},
    {YVEX_RUNTIME_MODEL_FAILURE_BUSY, YVEX_ERR_STATE, "runtime-model-draining", "model began draining"},
    {YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG,
     "attention-workspace-plan", "open CUDA session and an exact execution mode are required"},
    {YVEX_RUNTIME_MODEL_FAILURE_BUSY, YVEX_ERR_STATE, "session-lock", "runtime session workspace lock is unavailable"},
    {YVEX_RUNTIME_MODEL_FAILURE_BUSY, YVEX_ERR_STATE, "session-state", "idle open session is required"},
    {YVEX_RUNTIME_MODEL_FAILURE_BUSY, YVEX_ERR_STATE, "attention-state", "sealed idle state is required"},
    {YVEX_RUNTIME_MODEL_FAILURE_BUSY, YVEX_ERR_STATE, "host-workspace", "host workspace is already sealed"},
    {YVEX_RUNTIME_MODEL_FAILURE_GRAPH, YVEX_ERR_BOUNDS, "host-workspace-budget",
     "descriptor-bucket CUDA staging exceeds the session host budget"},
    {YVEX_RUNTIME_MODEL_FAILURE_BACKEND, YVEX_ERR_STATE, "workspace-capability-publication",
     "injected runtime workspace capability publication failure"},
    {YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "session", "open runtime session is required"},
    {YVEX_RUNTIME_MODEL_FAILURE_DRIFT, YVEX_ERR_STATE, "session-invalidated", "artifact drift invalidated session"},
    {YVEX_RUNTIME_MODEL_FAILURE_BUSY, YVEX_ERR_STATE, "session-closing", "runtime session is closing"},
    {YVEX_RUNTIME_MODEL_FAILURE_BUSY, YVEX_ERR_STATE, "session-busy", "runtime session already owns an execution"},
    {YVEX_RUNTIME_MODEL_FAILURE_CANCELLED, YVEX_ERR_CANCELLED, "session-cancelled", "session cancelled"},
    {YVEX_RUNTIME_MODEL_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-binding-admission", "binding lacks artifact"},
    {YVEX_RUNTIME_MODEL_FAILURE_ADAPTER, YVEX_ERR_FORMAT, "execution-capabilities",
     "family adapter has no typed execution capability contract"},
    {YVEX_RUNTIME_MODEL_FAILURE_ADAPTER, YVEX_ERR_FORMAT, "execution-capabilities",
     "bound family execution capability contract is stale or promoted"},
    {YVEX_RUNTIME_MODEL_FAILURE_DRIFT, YVEX_ERR_STATE, "artifact-snapshot", "artifact drift invalidated model"},
    {YVEX_RUNTIME_MODEL_FAILURE_BACKEND, YVEX_ERR_UNSUPPORTED, "device-capability", "device admission failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_BACKEND, YVEX_ERR_UNSUPPORTED, "cuda-capability", "CUDA admission failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, YVEX_ERR_STATE, "session-open-cleanup", "session cleanup failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_BACKEND, YVEX_ERR_BOUNDS, "device-workspace-budget",
     "descriptor-bucket CUDA workspace exceeds the session device budget"},
    {YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, "cleanup-lease",
     "empty cleanup lease, model request, and borrowed outputs are required"},
    {YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION, YVEX_ERR_NOMEM, "cleanup-lease", "lease allocation failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG,
     "cleanup-lease-session", "model-owning cleanup lease and session request are required"},
    {YVEX_RUNTIME_MODEL_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-binding", "runtime binding open failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_ADAPTER, YVEX_ERR_FORMAT, "family-adapter-id", "family adapter identity differs"},
    {YVEX_RUNTIME_MODEL_FAILURE_IDENTITY, YVEX_ERR_FORMAT, "logical-transform-identity",
     "runtime binding logical Transformation IR identity is stale"},
    {YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION, YVEX_ERR_BOUNDS, "model-host-budget",
     "configured model host budget cannot preserve the reserve after model residency"},
    {YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION, YVEX_ERR_BOUNDS, "process-memory-capacity",
     "process memory control cannot preserve the minimum reserve after model residency"},
    {YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION, YVEX_ERR_BOUNDS, "system-memory-capacity",
     "available system memory cannot preserve the minimum reserve after model residency"},
    {YVEX_RUNTIME_MODEL_FAILURE_ARTIFACT, YVEX_ERR_FORMAT, "artifact-open", "artifact admission failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_MATERIALIZATION, YVEX_ERR_FORMAT, "runtime-materialization",
     "runtime binding materialization could not be reopened"},
    {YVEX_RUNTIME_MODEL_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-import",
     "runtime binding import did not reconstruct sealed runtime facts"},
    {YVEX_RUNTIME_MODEL_FAILURE_DESCRIPTOR, YVEX_ERR_FORMAT, "physical-execution-ir",
     "runtime physical execution decisions could not be compiled"},
    {YVEX_RUNTIME_MODEL_FAILURE_IDENTITY, YVEX_ERR_FORMAT, "imported-identity", "import identity is invalid"},
    {YVEX_RUNTIME_MODEL_FAILURE_DESCRIPTOR, YVEX_ERR_FORMAT, "tokenizer-plan",
     "artifact tokenizer could not be admitted and bound to the runtime model"},
    {YVEX_RUNTIME_MODEL_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-model-seal", "model seal was cancelled"},
    {YVEX_RUNTIME_MODEL_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-model-build",
     "runtime model construction was not observed exactly once"},
    {YVEX_RUNTIME_MODEL_FAILURE_ADAPTER, YVEX_ERR_FORMAT, "execution-capabilities",
     "runtime execution capability contract could not be admitted"},
    {YVEX_RUNTIME_MODEL_FAILURE_MATERIALIZATION, YVEX_ERR_FORMAT, "model-residency",
     "runtime full-model residency could not be sealed"},
    {YVEX_RUNTIME_MODEL_FAILURE_MATERIALIZATION, YVEX_ERR_FORMAT,
     "model-residency-completeness",
     "runtime full-model residency is not complete for every descriptor binding"},
    {YVEX_RUNTIME_MODEL_FAILURE_DRIFT, YVEX_ERR_STATE, "artifact-snapshot", "artifact drifted before publication"}
};
_Static_assert(sizeof(runtime_refusals) / sizeof(runtime_refusals[0]) == YVEX_RUNTIME_REFUSE_COUNT,
               "runtime refusal catalog must cover every identity");

static int runtime_refuse_as(yvex_runtime_model_failure *failure,
                             yvex_runtime_private_refusal_id id,
                             unsigned long long expected,
                             unsigned long long actual,
                             yvex_status status, yvex_error *err) {
    const runtime_refusal_spec *spec;
    if ((unsigned int)id >= YVEX_RUNTIME_REFUSE_COUNT) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.refusal",
                       "runtime refusal identity is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    spec = &runtime_refusals[id];
    return yvex_runtime_private_reject(failure, spec->code, spec->field, expected, actual,
                                spec->reason, err, status == YVEX_OK ? spec->status : status);
}

int yvex_runtime_private_refuse(yvex_runtime_model_failure *failure, yvex_runtime_private_refusal_id id,
    unsigned long long expected, unsigned long long actual, yvex_error *err) {
    return runtime_refuse_as(failure, id, expected, actual, YVEX_OK, err);
}

int yvex_runtime_private_success(yvex_error *err) {
    yvex_error_clear(err);
    return YVEX_OK;
}

static int runtime_attention_state_provider_valid(const yvex_attention_state_provider *provider) {
    return provider && provider->schema_version == YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V7 &&
           provider->context && provider->configure_pages && provider->prepare &&
           provider->summary && provider->capacity && provider->recipe && provider->view &&
           provider->identity && provider->begin && provider->stage &&
           provider->select_prefix &&
           provider->prepare_commit && provider->publish_commit && provider->cancel_commit &&
           provider->commit && provider->abort &&
           provider->reset && provider->restore && provider->invalidate && provider->release;
}

static int runtime_model_once(unsigned long long *counter, const char *phase, yvex_error *err) {
    unsigned long long next;
    if (!counter || !phase || *counter != 0ull ||
        !yvex_core_u64_add(*counter, 1ull, &next)) {
        yvex_error_set(err, YVEX_ERR_STATE, phase,
                       "cold runtime construction event was not observed exactly once");
        return YVEX_ERR_STATE;
    }
    *counter = next;
    return YVEX_OK;
}

static int runtime_model_progress(const yvex_runtime_model_open_request *request,
                                  yvex_runtime_lifecycle_phase phase,
                                  unsigned long long completed, unsigned long long total,
                                  yvex_error *err) {
    if (!request->progress || request->progress(request->progress_context, phase, completed, total))
        return YVEX_OK;
    yvex_error_set(err, YVEX_ERR_CANCELLED, "runtime.model",
                   "runtime model preparation was cancelled");
    return YVEX_ERR_CANCELLED;
}

static int runtime_model_hash_progress(void *opaque, unsigned long long completed,
                                       unsigned long long total) {
    const yvex_runtime_model_open_request *request =
        (const yvex_runtime_model_open_request *)opaque;
    return !request->progress || request->progress(
               request->progress_context, YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH,
               completed, total);
}

static void runtime_model_timing(yvex_runtime_model *model, yvex_runtime_lifecycle_phase phase,
                                 unsigned long long started) {
    unsigned long long ended = yvex_core_monotonic_ns();
    if (ended >= started)
        model->summary.lifecycle_seconds[phase] += (double)(ended - started) / 1000000000.0;
}

static int runtime_model_identity_build(const yvex_runtime_binding_summary *binding,
                                        char output[YVEX_SHA256_HEX_CAP]) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!binding ||
        !yvex_sha256_update_text(&hash, "yvex.runtime.model.v1") ||
        !yvex_sha256_update_u64(&hash, YVEX_RUNTIME_MODEL_SCHEMA_V1) ||
        !yvex_sha256_update_text(&hash, binding->identity) ||
        !yvex_sha256_update_text(&hash, binding->artifact_identity) ||
        !yvex_sha256_update_text(&hash, binding->artifact_transform_identity) ||
        !yvex_sha256_update_text(&hash, binding->logical_transform_identity) ||
        !yvex_sha256_update_text(&hash, binding->materialization_identity) ||
        !yvex_sha256_update_text(&hash, binding->runtime_descriptor_identity) ||
        !yvex_sha256_update_text(&hash, binding->semantic_graph_identity) ||
        !yvex_sha256_update_text(&hash, binding->executable_graph_identity) ||
        !yvex_sha256_update_u64(&hash, binding->family_adapter_id) ||
        !yvex_sha256_update_u64(&hash, binding->family_adapter_version) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int runtime_model_session_reserve(yvex_runtime_model *model,
                                         yvex_runtime_model_failure *failure,
                                         yvex_error *err) {
    int accepted;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_MODEL_LOCK_UNAVAILABLE, 1ull, 0ull, err);
    accepted = model->summary.sealed && model->summary.valid && !model->close_requested;
    if (accepted && !yvex_core_u64_add(model->active_sessions, 1ull, &model->active_sessions))
        accepted = 0;
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    if (!accepted)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_MODEL_INVALID_OR_DRAINING, 0ull, 1ull, err);
    return YVEX_OK;
}

static void runtime_model_session_register_locked(
    yvex_runtime_model *model, yvex_runtime_execution_session *session) {
    session->model_next = model->sessions;
    if (model->sessions) model->sessions->model_previous = session;
    model->sessions = session;
    session->model_registered = 1;
}

static int runtime_model_session_unregister(yvex_runtime_model *model,
    yvex_runtime_execution_session *session, yvex_error *err) {
    if (!model || !session || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.unregister",
                       "runtime model lifecycle lock is unavailable during session close");
        return YVEX_ERR_STATE;
    }
    if (session->model_registered) {
        if (session->model_previous)
            session->model_previous->model_next = session->model_next;
        else
            model->sessions = session->model_next;
        if (session->model_next)
            session->model_next->model_previous = session->model_previous;
        session->model_previous = NULL;
        session->model_next = NULL;
        session->model_registered = 0;
    }
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    return yvex_runtime_private_success(err);
}

int yvex_runtime_private_session_invalidate(yvex_runtime_execution_session *session,
                                      int include_state, yvex_error *err) {
    unsigned long long affected;
    yvex_error cleanup;
    int graph_rc, rc = YVEX_OK;
    if (include_state && session->attention_state_provider_ready)
        rc = session->attention_state_provider.invalidate(
            session->attention_state_provider.context, err);
    if (include_state && session->state_residency) {
        graph_rc = yvex_runtime_state_residency_invalidate(
            session->state_residency, &cleanup);
        if (rc == YVEX_OK && graph_rc != YVEX_OK) {
            rc = graph_rc;
            if (err) *err = cleanup;
        }
    }
    if (include_state && session->draft_attention_state_provider_ready) {
        yvex_error_clear(&cleanup);
        graph_rc = session->draft_attention_state_provider.invalidate(
            session->draft_attention_state_provider.context, &cleanup);
        if (rc == YVEX_OK && graph_rc != YVEX_OK) {
            rc = graph_rc;
            if (err) *err = cleanup;
        }
    }
    if (include_state && session->draft_state_residency) {
        yvex_error_clear(&cleanup);
        graph_rc = yvex_runtime_state_residency_invalidate(
            session->draft_state_residency, &cleanup);
        if (rc == YVEX_OK && graph_rc != YVEX_OK) {
            rc = graph_rc;
            if (err) *err = cleanup;
        }
    }
    if (session->backend &&
        yvex_backend_kind_of(session->backend) == YVEX_BACKEND_KIND_CUDA) {
        yvex_error_clear(&cleanup);
        graph_rc = yvex_backend_cuda_attention_graph_registry_apply(
            session->backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_INVALIDATE, &affected, &cleanup);
        if (rc == YVEX_OK && graph_rc != YVEX_OK) {
            rc = graph_rc;
            if (err) *err = cleanup;
        }
    }
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

static int runtime_model_dependents_invalidate_locked(yvex_runtime_model *model, yvex_error *err) {
    yvex_runtime_execution_session *session;
    yvex_error first_error, cleanup;
    int first_rc = YVEX_OK, rc;
    if (model->residency)
        first_rc = yvex_runtime_residency_invalidate(model->residency, &first_error);
    for (session = model->sessions; session; session = session->model_next) {
        if (!session->lifecycle_mutex_ready ||
            pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
            if (first_rc == YVEX_OK) {
                first_rc = YVEX_ERR_STATE;
                yvex_error_set(&first_error, YVEX_ERR_STATE, "runtime.session.invalidate-lock",
                               "runtime session invalidation lock is unavailable");
            }
            continue;
        }
        session->summary.invalidated = session->summary.cancelled = 1;
        session->summary.residency_generation = session->summary.workspace_generation = 0ull;
        session->summary.residency_identity[0] = session->summary.workspace_identity[0] = '\0';
        session->invalidation_pending = 1;
        if (!session->summary.busy) {
            rc = yvex_runtime_private_session_invalidate(session, 1, &cleanup);
            if (rc == YVEX_OK) session->invalidation_pending = 0;
            if (rc != YVEX_OK && first_rc == YVEX_OK) first_rc = rc, first_error = cleanup;
        }
        (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    }
    if (first_rc == YVEX_OK) yvex_error_clear(err);
    else if (err) *err = first_error;
    return first_rc;
}

static int runtime_model_release(yvex_runtime_model *model, yvex_error *err) {
    int rc;
    if (!model)
        return YVEX_OK;
    if (getenv("YVEX_TEST_RUNTIME_MODEL_CLEANUP_FAILURE")) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.model.release",
                       "injected runtime model cleanup failure");
        return YVEX_ERR_STATE;
    }
    rc = yvex_runtime_residency_close(&model->residency, err);
    if (rc != YVEX_OK) return rc;
    rc = yvex_backend_close_checked(&model->opening_backend, err);
    if (rc != YVEX_OK) return rc;
    yvex_tokenizer_close(model->tokenizer);
    model->tokenizer = NULL;
    yvex_materialization_session_close(model->materialization);
    model->materialization = NULL;
    yvex_attention_plan_close(model->attention);
    yvex_attention_plan_close(model->draft_attention);
    yvex_runtime_descriptor_close(model->descriptor);
    yvex_materialization_plan_close(model->materialization_plan);
    yvex_tensor_table_close(model->tensors);
    yvex_gguf_close(model->gguf);
    yvex_artifact_close(model->artifact);
    yvex_runtime_binding_close(model->binding);
    if (model->lifecycle_mutex_ready) {
        (void)pthread_mutex_destroy(&model->lifecycle_mutex);
        model->lifecycle_mutex_ready = 0;
    }
    memset(model, 0, sizeof(*model));
    free(model);
    return yvex_runtime_private_success(err);
}

static int runtime_session_model_discharge(yvex_runtime_execution_session *session,
                                           yvex_error *err) {
    yvex_runtime_model *model = session ? session->model : NULL;
    int rc;
    if (!session || (!model &&
                     (session->model_reserved || session->model_release_pending))) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.model-discharge",
                       "retained session model ownership is required");
        return YVEX_ERR_STATE;
    }
    if (!model) {
        return yvex_runtime_private_success(err);
    }
    if (session->model_reserved) {
        if (!model->lifecycle_mutex_ready ||
            getenv("YVEX_TEST_RUNTIME_SESSION_UNRESERVE_FAILURE") ||
            pthread_mutex_lock(&model->lifecycle_mutex) != 0) {
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.model-unreserve",
                           "runtime model reservation could not be discharged");
            return YVEX_ERR_STATE;
        }
        if (!model->active_sessions) {
            (void)pthread_mutex_unlock(&model->lifecycle_mutex);
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.model-unreserve",
                           "runtime model reservation accounting is inconsistent");
            return YVEX_ERR_STATE;
        }
        model->active_sessions--;
        session->model_release_pending =
            model->close_requested && model->active_sessions == 0ull;
        session->model_reserved = 0;
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    }
    if (session->model_release_pending) {
        rc = runtime_model_release(model, err);
        if (rc != YVEX_OK) return rc;
        session->model_release_pending = 0;
    }
    session->model = NULL;
    return yvex_runtime_private_success(err);
}

static int runtime_model_open_fail(yvex_runtime_model **out, yvex_runtime_model *model,
                                   yvex_runtime_model_failure *failure,
                                   yvex_runtime_private_refusal_id refusal,
                                   unsigned long long expected, unsigned long long actual,
                                   yvex_error *err, yvex_status status) {
    const runtime_refusal_spec *spec = &runtime_refusals[refusal];
    yvex_error primary;
    int cleanup_rc;
    (void)yvex_runtime_private_reject(
        failure, spec->code, spec->field, expected, actual, spec->reason, err, status);
    primary = err ? *err : (yvex_error){0};
    cleanup_rc = runtime_model_release(model, err);
    if (cleanup_rc != YVEX_OK) {
        *out = model;
        yvex_runtime_private_failure_record(
            failure, YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, "model-open-cleanup", 0ull, 1ull,
            "runtime model candidate cleanup retained ownership for retry");
        return cleanup_rc;
    }
    if (err) *err = primary;
    return status;
}

static int runtime_model_artifact_open(
    yvex_runtime_model *model, const yvex_runtime_model_open_request *request,
    const yvex_runtime_binding_summary *binding, yvex_runtime_model_failure *failure,
    yvex_error *err) {
    yvex_artifact_admission_failure admission_failure;
    yvex_artifact_options options;
    unsigned long long started;
    int rc;
    if (!model->admission.file_bytes || !model->admission.tensor_count)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_BINDING_ADMISSION, 1ull, 0ull, err);
    memset(&options, 0, sizeof(options));
    options.path = request->artifact_path;
    options.readonly = 1;
    started = yvex_core_monotonic_ns();
    rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN,
                                0ull, 0ull, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_open(&model->artifact, &options, err);
    runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN, started);
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_ARTIFACT_ADMISSION,
                                    0ull, 0ull, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_snapshot_get(model->artifact, &model->admission.file_snapshot, err);
    if (rc == YVEX_OK)
        yvex_core_text_copy(model->admission.artifact_path,
                            sizeof(model->admission.artifact_path),
                            yvex_artifact_path(model->artifact));
    runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_ARTIFACT_ADMISSION, started);
    if (rc == YVEX_OK &&
        strcmp(model->admission.artifact_identity, binding->artifact_identity) != 0)
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_ARTIFACT_IDENTITY, 1ull, 0ull, err);
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH,
                                    0ull, yvex_artifact_size(model->artifact), err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admission_identity_verify(
            model->artifact, &model->admission, runtime_model_hash_progress,
            (void *)request, &admission_failure, err);
    if (rc == YVEX_OK)
        rc = runtime_model_once(&model->summary.artifact_hash_passes,
                                "runtime.model.artifact-hash", err);
    runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH, started);
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = yvex_gguf_open(&model->gguf, model->artifact, err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&model->tensors, model->gguf, err);
    if (rc == YVEX_OK)
        rc = runtime_model_once(&model->summary.gguf_directory_parses,
                                "runtime.model.gguf-directory", err);
    runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_ARTIFACT_ADMISSION, started);
    return rc;
}

static int runtime_model_memory_value(const char *text, unsigned long long *value)
{
    char *tail = NULL;
    unsigned long long parsed;
    if (!text || !value || !text[0] || text[0] == '-' || !strncmp(text, "max", 3u)) return 0;
    errno = 0;
    parsed = strtoull(text, &tail, 10);
    if (errno || tail == text) return 0;
    while (*tail == ' ' || *tail == '\t' || *tail == '\r' || *tail == '\n') ++tail;
    if (*tail) return 0;
    *value = parsed;
    return 1;
}

/* Return the tightest remaining cgroup-v2 memory extent across the process hierarchy. */
static int runtime_model_cgroup_available_memory(unsigned long long *bytes)
{
    const char *injected = getenv("YVEX_TEST_RUNTIME_CGROUP_AVAILABLE_MEMORY_BYTES");
    static const char *const controls[] = {"memory.max", "memory.high"};
    const char *root = "/sys/fs/cgroup";
    char group[PATH_MAX], directory[PATH_MAX], path[PATH_MAX], text[128];
    char *relative, *newline, *slash;
    unsigned long long available = ULLONG_MAX;
    size_t control, root_length = strlen(root);
    int found = 0, written;
    if (!bytes) return -1;
    if (injected)
        return runtime_model_memory_value(injected, bytes) ? 1 : -1;
    if (!yvex_core_file_read_text_prefix("/proc/self/cgroup", group, sizeof(group)) ||
        !(relative = strstr(group, "0::")) ||
        (relative != group && relative[-1] != '\n') ||
        relative[3] != '/' || strstr(relative + 3, ".."))
        return 0;
    relative += 3;
    if ((newline = strchr(relative, '\n'))) *newline = '\0';
    written = snprintf(directory, sizeof(directory), "%s%s", root, relative);
    if (written <= 0 || (size_t)written >= sizeof(directory)) return -1;
    for (;;) {
        unsigned long long current;
        int current_known;
        written = snprintf(path, sizeof(path), "%s/memory.current", directory);
        if (written <= 0 || (size_t)written >= sizeof(path)) return -1;
        current_known = yvex_core_file_read_text_prefix(path, text, sizeof(text)) &&
                        runtime_model_memory_value(text, &current);
        for (control = 0u; current_known && control < 2u; ++control) {
            unsigned long long limit, remaining;
            written = snprintf(path, sizeof(path), "%s/%s", directory, controls[control]);
            if (written <= 0 || (size_t)written >= sizeof(path)) return -1;
            if (!yvex_core_file_read_text_prefix(path, text, sizeof(text)) ||
                !runtime_model_memory_value(text, &limit)) continue;
            remaining = current < limit ? limit - current : 0ull;
            if (!found || remaining < available) available = remaining;
            found = 1;
        }
        if (!strcmp(directory, root)) break;
        slash = strrchr(directory, '/');
        if (!slash || (size_t)(slash - directory) < root_length) return -1;
        *slash = '\0';
    }
    if (found) *bytes = available;
    return found;
}

int yvex_runtime_private_available_memory(unsigned long long *bytes,
                                          int *process_limited)
{
    const char *injected = getenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES");
    char line[128];
    FILE *meminfo;
    unsigned long long value, process_available;
    long pages, page_bytes;
    int cgroup;

    if (!bytes || !process_limited) return 0;
    *process_limited = 0;
    if (injected) {
        if (!runtime_model_memory_value(injected, &value)) return 0;
    } else if ((meminfo = fopen("/proc/meminfo", "r"))) {
        value = 0ull;
        while (fgets(line, sizeof(line), meminfo)) {
            if (sscanf(line, "MemAvailable: %llu kB", &value) == 1) {
                if (!yvex_core_u64_mul(value, 1024ull, &value)) value = 0ull;
                break;
            }
        }
        if (fclose(meminfo) != 0 || !value) return 0;
    } else {
#ifdef _SC_AVPHYS_PAGES
        pages = sysconf(_SC_AVPHYS_PAGES);
        page_bytes = sysconf(_SC_PAGESIZE);
        if (pages <= 0 || page_bytes <= 0 ||
            !yvex_core_u64_mul((unsigned long long)pages,
                               (unsigned long long)page_bytes, &value)) return 0;
#else
        (void)pages;
        (void)page_bytes;
        return 0;
#endif
    }
    cgroup = runtime_model_cgroup_available_memory(&process_available);
    if (cgroup < 0) return 0;
    if (cgroup > 0 && process_available < value) {
        value = process_available;
        *process_limited = 1;
    }
    *bytes = value;
    return 1;
}

static int runtime_model_memory_preflight(
    const yvex_runtime_model_open_request *request,
    const yvex_complete_artifact_admission *admission,
    yvex_runtime_private_refusal_id *refusal,
    unsigned long long *required, unsigned long long *available)
{
    int process_limited;
    if (!request || !admission || !refusal || !required || !available ||
        !admission->payload_bytes) return YVEX_ERR_INVALID_ARG;
    *required = 0ull;
    *available = 0ull;
    if (!yvex_core_u64_add(admission->payload_bytes,
                           YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE, required))
        return YVEX_ERR_STATE;
    if (request->maximum_host_bytes &&
        *required > request->maximum_host_bytes) {
        *refusal = YVEX_RUNTIME_REFUSE_OPEN_HOST_BUDGET;
        *available = request->maximum_host_bytes;
        return YVEX_ERR_BOUNDS;
    }
    if (!yvex_runtime_private_available_memory(available, &process_limited))
        return YVEX_ERR_STATE;
    *refusal = process_limited ? YVEX_RUNTIME_REFUSE_OPEN_PROCESS_MEMORY
                               : YVEX_RUNTIME_REFUSE_OPEN_SYSTEM_MEMORY;
    return *required <= *available ? YVEX_OK : YVEX_ERR_BOUNDS;
}

static int runtime_model_capabilities_bind(
    yvex_runtime_model *model, const yvex_runtime_binding_summary *binding,
    const yvex_attention_summary *attention,
    yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_runtime_capabilities *capabilities = &model->summary.capabilities;
    char binding_identity[YVEX_SHA256_HEX_CAP];
    const int graph_ready = attention &&
        attention->history_contract_ready && attention->full_execution_ready;
    const int cpu_ready = graph_ready && attention->cpu_reference_ready;
    const int cuda_ready = graph_ready && attention->cuda_execution_ready;
    if (!model || !binding ||
        !yvex_runtime_capabilities_contract_valid(&binding->capabilities))
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_ADAPTER_CAPABILITY, 1ull, 0ull, err);
    if (!yvex_runtime_capabilities_identity(&binding->capabilities, binding_identity) ||
        strcmp(binding_identity, binding->execution_capability_identity) != 0)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_ADAPTER_CAPABILITY_STALE, 1ull, 0ull, err);
    *capabilities = binding->capabilities;
    capabilities->attention_semantics_ready &= graph_ready;
    capabilities->attention_core_ready &= graph_ready;
    capabilities->attention_envelope_ready &= graph_ready;
    capabilities->cpu_prefill_eager_ready &= cpu_ready;
    capabilities->cpu_decode_eager_ready &= cpu_ready;
    capabilities->cuda_eager_implemented &= cuda_ready;
    capabilities->cuda_piecewise_graph_implemented &= cuda_ready;
    capabilities->cuda_full_graph_implemented &= cuda_ready;
    capabilities->attention_state_delta_ready =
        capabilities->attention_state_delta_ready && attention->state_delta_contract_ready;
    return YVEX_OK;
}

static void runtime_model_summary_bind(
    yvex_runtime_model *model, const yvex_attention_summary *attention,
    const yvex_attention_summary *draft_attention)
{
    yvex_runtime_identity_copy(model->summary.runtime_binding_identity,
                               model->binding_summary.identity);
    yvex_runtime_identity_copy(model->summary.artifact_identity,
                               model->binding_summary.artifact_identity);
    yvex_runtime_identity_copy(model->summary.materialization_identity,
                               model->binding_summary.materialization_identity);
    yvex_runtime_identity_copy(model->summary.runtime_descriptor_identity,
                               model->binding_summary.runtime_descriptor_identity);
    yvex_runtime_identity_copy(model->summary.runtime_numeric_identity,
                               model->binding_summary.runtime_numeric_identity);
    yvex_runtime_identity_copy(model->summary.semantic_graph_identity,
                               model->binding_summary.semantic_graph_identity);
    yvex_runtime_identity_copy(model->summary.executable_graph_identity,
                               model->binding_summary.executable_graph_identity);
    model->summary.artifact_bytes_hashed = model->admission.artifact_bytes_hashed;
    model->summary.tensor_count = model->binding_summary.tensor_count;
    model->summary.attention_layer_count = model->binding_summary.layer_count;
    model->summary.draft_attention_layer_count =
        model->binding_summary.draft_layer_count;
    model->summary.attention_binding_count = attention->required_binding_count;
    model->summary.draft_attention_binding_count =
        draft_attention ? draft_attention->required_binding_count : 0ull;
    {
        const yvex_physical_execution_summary *physical =
            yvex_physical_execution_ir_summary(model->physical_execution);
        if (physical) {
            model->summary.physical_execution_decision_count = physical->decision_count;
            yvex_runtime_identity_copy(model->summary.physical_execution_identity,
                                       physical->identity);
        }
    }
}

static int runtime_model_residency_open(
    yvex_runtime_model *model, const yvex_runtime_model_open_request *request,
    const yvex_runtime_descriptor_summary *descriptor_summary,
    const yvex_attention_summary *attention_summary,
    yvex_runtime_private_refusal_id *refusal, yvex_error *err)
{
    yvex_runtime_residency_options options;
    yvex_runtime_residency_failure residency_failure;
    yvex_runtime_residency_summary summary;
    unsigned long long started;
    int rc;
    *refusal = YVEX_RUNTIME_REFUSE_OPEN_RESIDENCY;
    if (!attention_summary->required_binding_count) return YVEX_OK;
    started = yvex_core_monotonic_ns();
    rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_RESIDENCY,
                                0ull, attention_summary->required_binding_count, err);
    memset(&options, 0, sizeof(options));
    options.maximum_host_bytes = request->maximum_host_bytes;
    options.placement = request->residency_backend == YVEX_BACKEND_KIND_CUDA
                            ? YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED
                            : YVEX_RUNTIME_WEIGHT_PLACEMENT_HOST_LOCKED;
    memset(&residency_failure, 0, sizeof(residency_failure));
    if (rc == YVEX_OK)
        rc = yvex_runtime_residency_prepare(&model->residency, model, &options,
                                            &residency_failure, err);
    runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_RESIDENCY, started);
    if (rc != YVEX_OK) return rc;
    model->view.residency = model->residency;
    memset(&summary, 0, sizeof(summary));
    rc = yvex_runtime_residency_snapshot(model->residency, &summary, NULL, NULL, err);
    if (rc != YVEX_OK || !summary.model_complete ||
        (!summary.host_locked &&
         !(summary.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED &&
           summary.cuda_managed_allocation_count == 1ull &&
           summary.cuda_managed_bytes == summary.encoded_bytes &&
           summary.cuda_managed_prefetch_count == 1ull &&
           summary.cuda_managed_prefetch_bytes == summary.encoded_bytes)) ||
        (request->residency_backend == YVEX_BACKEND_KIND_CUDA && !summary.cuda_ready) ||
        summary.binding_count != descriptor_summary->tensor_count ||
        summary.encoded_bytes != descriptor_summary->payload_bytes ||
        !summary.core_complete || !summary.envelope_complete ||
        !yvex_runtime_logits_residency_admit(&model->summary.capabilities, &summary))
        {
            *refusal = YVEX_RUNTIME_REFUSE_OPEN_RESIDENCY_COMPLETE;
            if (rc == YVEX_OK) {
                yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.model.residency",
                               "runtime residency is incomplete for the admitted descriptor");
                rc = YVEX_ERR_FORMAT;
            }
            return rc;
        }
    model->summary.capabilities.attention_weight_residency_ready = 1;
    model->summary.capabilities.attention_envelope_ready =
        model->summary.capabilities.attention_envelope_ready && summary.envelope_complete;
    return YVEX_OK;
}

static void runtime_model_view_bind(yvex_runtime_model *model)
{
    model->view.binding = &model->binding_summary;
    model->view.compiled_binding = model->binding;
    model->view.compiled_plan = model->binding->plan;
    model->view.graph = model->graph;
    model->view.target_id = model->target_id;
    model->view.attention = model->attention;
    model->view.draft_attention = model->draft_attention;
    model->view.moe = yvex_compiled_model_plan_moe(model->binding->plan, 0);
    model->view.draft_moe = yvex_compiled_model_plan_moe(model->binding->plan, 1);
    model->view.transformer =
        yvex_compiled_model_plan_transformer(model->binding->plan, 0);
    model->view.draft_transformer =
        yvex_compiled_model_plan_transformer(model->binding->plan, 1);
    model->view.output_head =
        yvex_compiled_model_plan_output_head(model->binding->plan);
    model->view.descriptor = model->descriptor;
    model->view.physical_execution = model->physical_execution;
    model->view.tokenizer = model->tokenizer;
    model->view.materialization = model->materialization;
}

int yvex_runtime_model_open(yvex_runtime_model **out, const yvex_runtime_model_open_request *request,
                            yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_runtime_model *model = NULL;
    const yvex_runtime_descriptor_summary *descriptor_summary;
    const yvex_attention_summary *attention_summary;
    const yvex_attention_summary *draft_attention_summary;
    yvex_runtime_binding_failure binding_failure;
    yvex_materialization_options materialization_options;
    yvex_runtime_private_refusal_id capacity_refusal = YVEX_RUNTIME_REFUSE_OPEN_SYSTEM_MEMORY;
    yvex_runtime_private_refusal_id residency_refusal;
    unsigned long long total_started, phase_started, required_bytes = 0ull, available_bytes = 0ull;
    int rc;
    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!out || !request || !request->artifact_path || !request->runtime_binding_path ||
        !request->target_id || !request->target_id[0] ||
        strlen(request->target_id) >= sizeof(model->target_id))
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_MODEL_OPEN_REQUEST, 1ull, 0ull, err);
    model = (yvex_runtime_model *)calloc(1u, sizeof(*model));
    if (!model)
        return yvex_runtime_private_refuse(
            failure, YVEX_RUNTIME_REFUSE_MODEL_ALLOCATION,
            sizeof(*model), 0ull, err);
    if (pthread_mutex_init(&model->lifecycle_mutex, NULL) != 0) {
        free(model);
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_MODEL_LOCK_INITIALIZATION, 1ull, 0ull, err);
    }
    model->lifecycle_mutex_ready = 1;
    total_started = yvex_core_monotonic_ns();
    memset(&binding_failure, 0, sizeof(binding_failure));
    phase_started = yvex_core_monotonic_ns();
    rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_BINDING_OPEN, 0ull, 0ull, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_binding_open(
            &model->binding, request->runtime_binding_path, &model->binding_summary,
            &model->admission, &binding_failure, err);
    if (rc == YVEX_OK)
        rc = runtime_model_once(&model->summary.runtime_binding_parses,
                                "runtime.model.binding-parse", err);
    runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_BINDING_OPEN, phase_started);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_BINDING, 1ull, 0ull, err, (yvex_status)rc);
    model->graph = &yvex_attention_execution_api;
    yvex_core_text_copy(model->target_id, sizeof(model->target_id), request->target_id);
    rc = runtime_model_memory_preflight(
        request, &model->admission, &capacity_refusal, &required_bytes, &available_bytes);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, capacity_refusal, required_bytes, available_bytes,
            err, (yvex_status)rc);
    rc = runtime_model_artifact_open(model, request, &model->binding_summary, failure, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_ARTIFACT, 1ull, 0ull, err, (yvex_status)rc);
    if (request->residency_backend == YVEX_BACKEND_KIND_CUDA) {
        yvex_backend_options backend_options = {
            .kind = YVEX_BACKEND_KIND_CUDA,
            .memory_limit_bytes = request->maximum_device_bytes,
        };

        rc = yvex_backend_open(&model->opening_backend, &backend_options, err);
        if (rc != YVEX_OK)
            return runtime_model_open_fail(
                out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_RESIDENCY,
                1ull, 0ull, err, (yvex_status)rc);
    }
    phase_started = yvex_core_monotonic_ns();
    rc = runtime_model_progress(
        request, YVEX_RUNTIME_LIFECYCLE_MATERIALIZATION_OPEN, 0ull, 1ull, err);
    yvex_materialization_options_default(&materialization_options);
    materialization_options.require_complete_admission = 1;
    materialization_options.release_artifact_cache_after_read = 1;
    if (rc == YVEX_OK)
        rc = yvex_runtime_binding_import_materialization(
            model->binding, model->artifact, &materialization_options,
            &model->materialization_plan, &model->materialization, &binding_failure, err);
    runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_MATERIALIZATION_OPEN, phase_started);
    if (rc == YVEX_OK)
        rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_MATERIALIZATION_OPEN,
                                    1ull, 1ull, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_MATERIALIZATION, 1ull, 0ull, err,
            (yvex_status)rc);
    phase_started = yvex_core_monotonic_ns();
    rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL, 0ull, 1ull, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_binding_import_graph(
            model->binding, model->materialization, &model->descriptor,
            &model->attention, &model->draft_attention,
            &model->physical_execution, &binding_failure, err);
    if (rc == YVEX_OK)
        rc = runtime_model_once(&model->summary.runtime_descriptor_builds,
                                "runtime.model.descriptor-build", err);
    if (rc == YVEX_OK)
        rc = runtime_model_once(&model->summary.semantic_graph_builds,
                                "runtime.model.semantic-graph-build", err);
    if (rc == YVEX_OK)
        rc = runtime_model_once(&model->summary.executable_graph_builds,
                                "runtime.model.executable-graph-build", err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_IMPORT, 1ull, 0ull, err, (yvex_status)rc);
    descriptor_summary = yvex_runtime_descriptor_summary_get(model->descriptor);
    if (!descriptor_summary)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_IMPORTED_IDENTITY,
            1ull, 0ull, err, YVEX_ERR_FORMAT);
    attention_summary = yvex_attention_plan_summary(model->attention);
    draft_attention_summary = yvex_attention_plan_summary(model->draft_attention);
    if (!descriptor_summary || !attention_summary ||
        strcmp(descriptor_summary->runtime_descriptor_identity,
               model->binding_summary.runtime_descriptor_identity) != 0 ||
        strcmp(yvex_physical_execution_ir_summary(model->physical_execution)->identity,
               model->binding_summary.physical_execution_identity) != 0 ||
        strcmp(attention_summary->attention_plan_identity,
               model->binding_summary.attention_plan_identity) != 0 ||
        (model->binding_summary.draft_layer_count &&
         (!draft_attention_summary ||
          strcmp(draft_attention_summary->attention_plan_identity,
                 model->binding_summary.draft_attention_plan_identity) != 0)) ||
        !runtime_model_identity_build(&model->binding_summary,
                                      model->summary.runtime_model_identity)) {
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_IMPORTED_IDENTITY, 1ull, 0ull, err,
            YVEX_ERR_FORMAT);
    }
    rc = yvex_tokenizer_from_compiled_gguf(
        &model->tokenizer, model->gguf,
        yvex_runtime_binding_tokenizer_policy(model->binding), err);
    if (rc == YVEX_OK && yvex_tokenizer_plan_summary_get(model->tokenizer))
        rc = yvex_tokenizer_bind_runtime(
            model->tokenizer, model->binding_summary.artifact_identity,
            model->binding_summary.logical_model_identity,
            model->binding_summary.runtime_descriptor_identity, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_TOKENIZER, 1ull, 0ull, err,
            (yvex_status)rc);
    runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL, phase_started);
    rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL, 1ull, 1ull, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_SEAL, 1ull, 0ull, err, (yvex_status)rc);
    model->summary.schema_version = YVEX_RUNTIME_MODEL_SCHEMA_V1;
    model->summary.sealed = model->summary.valid = 1;
    rc = runtime_model_once(&model->summary.runtime_model_builds, "runtime.model.seal", err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_BUILD, 1ull,
            model->summary.runtime_model_builds, err, (yvex_status)rc);
    runtime_model_summary_bind(model, attention_summary, draft_attention_summary);
    rc = runtime_model_capabilities_bind(model, &model->binding_summary, attention_summary, failure, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_CAPABILITIES, 1ull, 0ull, err,
            (yvex_status)rc);
    runtime_model_view_bind(model);
    rc = runtime_model_residency_open(model, request, descriptor_summary,
                                      attention_summary, &residency_refusal, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, residency_refusal, 1ull, 0ull, err,
            (yvex_status)rc);
    rc = yvex_artifact_snapshot_validate(model->artifact, NULL, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, YVEX_RUNTIME_REFUSE_OPEN_DRIFT, 1ull, 0ull, err, (yvex_status)rc);
    model->summary.total_seconds = (double)(yvex_core_monotonic_ns() - total_started) / 1000000000.0;
    *out = model;
    if (failure) memset(failure, 0, sizeof(*failure));
    return yvex_runtime_private_success(err);
}

int yvex_runtime_model_validate(yvex_runtime_model *model,
                                yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_error cleanup;
    int cleanup_rc = YVEX_OK, counter_overflow, rc;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_MODEL_REQUIRED, 1ull, 0ull, err);
    if (!model->summary.sealed || model->close_requested) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_MODEL_UNSEALED, 0ull, 1ull, err);
    }
    counter_overflow =
        !yvex_core_u64_add(model->summary.drift_checks, 1ull, &model->summary.drift_checks);
    rc = counter_overflow ? YVEX_ERR_BOUNDS
                          : yvex_artifact_snapshot_validate(model->artifact, NULL, err);
    if (rc == YVEX_OK && getenv("YVEX_TEST_RUNTIME_MODEL_DRIFT")) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.model.snapshot",
                       "injected runtime artifact snapshot drift");
    }
    if (rc == YVEX_OK && model->summary.valid) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        if (failure)
            memset(failure, 0, sizeof(*failure));
        return YVEX_OK;
    }
    if (model->summary.valid) {
        model->summary.valid = 0;
        (void)yvex_core_u64_add(model->summary.invalidation_count, 1ull,
                                &model->summary.invalidation_count);
        model->summary.capabilities.attention_weight_residency_ready = 0;
        model->summary.capabilities.attention_envelope_ready = 0;
        yvex_runtime_logits_capabilities_invalidate(&model->summary.capabilities);
        model->dependent_invalidation_pending = 1;
    }
    if (model->dependent_invalidation_pending) {
        yvex_error_clear(&cleanup);
        cleanup_rc = runtime_model_dependents_invalidate_locked(model, &cleanup);
        if (cleanup_rc == YVEX_OK) model->dependent_invalidation_pending = 0;
    }
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    if (cleanup_rc != YVEX_OK) goto cleanup_failed;
    if (counter_overflow)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_DRIFT_COUNTER, ULLONG_MAX, 1ull, err);
    return runtime_refuse_as(
        failure, YVEX_RUNTIME_REFUSE_ARTIFACT_DRIFT, 1ull, 0ull,
        rc == YVEX_OK ? YVEX_ERR_STATE : (yvex_status)rc, err);
cleanup_failed:
    yvex_runtime_private_failure_record(failure, YVEX_RUNTIME_MODEL_FAILURE_CLEANUP,
                                 cleanup.where[0] ? cleanup.where : "dependent-invalidation",
                                 0ull, 1ull,
                                 "runtime dependent cleanup failed during model invalidation");
    if (err) *err = cleanup;
    return cleanup_rc;
}

static int runtime_summary_copy(const void *owner, const void *summary, void *out,
                                size_t size, int mutex_ready, pthread_mutex_t *mutex,
                                const char *where, const char *argument_reason,
                                const char *synchronization_reason, yvex_error *err) {
    if (!owner || !summary || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, argument_reason);
        return YVEX_ERR_INVALID_ARG;
    }
    if (!mutex_ready || !mutex || pthread_mutex_lock(mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, where, synchronization_reason);
        return YVEX_ERR_STATE;
    }
    memcpy(out, summary, size);
    (void)pthread_mutex_unlock(mutex);
    return yvex_runtime_private_success(err);
}

int yvex_runtime_model_summary_copy(const yvex_runtime_model *model,
                                    yvex_runtime_model_summary *out,
                                    yvex_error *err) {
    yvex_runtime_model *mutable_model = (yvex_runtime_model *)model;
    return runtime_summary_copy(
        model, model ? &model->summary : NULL, out, sizeof(*out),
        model ? model->lifecycle_mutex_ready : 0,
        model ? &mutable_model->lifecycle_mutex : NULL, "runtime.model.summary",
        "runtime model and summary output are required",
        "runtime model synchronization is unavailable", err);
}

void yvex_runtime_model_close(yvex_runtime_model **model_ptr) {
    yvex_runtime_model *model;
    int release;
    if (!model_ptr || !*model_ptr)
        return;
    model = *model_ptr;
    if (!model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return;
    model->close_requested = 1;
    release = model->active_sessions == 0ull;
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    if (!release || runtime_model_release(model, NULL) == YVEX_OK)
        *model_ptr = NULL;
}
/*
 * Borrow model view.
 *
 * Model lifetime.
 */
const yvex_runtime_model_view *yvex_runtime_model_view_get(const yvex_runtime_model *model) {
    return model ? &model->view : NULL;
}

static int runtime_session_attach_cuda_residency(
    yvex_runtime_execution_session *session, int *uploaded,
    yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_runtime_residency_summary summary;
    int rc = yvex_runtime_residency_cuda_session_attach(
        session->model->residency, &session->backend, session->maximum_device_bytes,
        uploaded, &summary, err);
    if (rc != YVEX_OK) {
        yvex_runtime_private_failure_record(
            failure, YVEX_RUNTIME_MODEL_FAILURE_BACKEND, "device-residency",
            session->maximum_device_bytes, summary.device_resident_bytes,
            "CUDA model residency or session attachment failed");
        return rc;
    }
    session->summary.resident_binding_count = summary.binding_count;
    session->summary.resident_encoded_bytes = summary.encoded_bytes;
    session->summary.host_resident_bytes = summary.host_resident_bytes;
    session->summary.device_resident_bytes = summary.device_resident_bytes;
    session->summary.upload_bytes = *uploaded ? summary.cuda_upload_bytes : 0ull;
    session->summary.upload_count = *uploaded ? summary.cuda_upload_count : 0ull;
    session->summary.residency_generation = summary.generation;
    yvex_runtime_identity_copy(session->summary.residency_identity, summary.residency_identity);
    session->summary.peak_device_bytes = summary.device_resident_bytes;
    return YVEX_OK;
}

int yvex_runtime_private_session_capabilities_bind(
    yvex_runtime_execution_session *session, yvex_runtime_model_failure *failure,
    int require_workspace, yvex_error *err) {
    yvex_runtime_capabilities capabilities = session->model->summary.capabilities;
    yvex_graph_attention_state_summary state = {0};
    yvex_runtime_state_residency_summary state_residency = {0};
    yvex_backend_capability_result encoded = {0};
    yvex_backend_cuda_graph_capability graph = {0};
    yvex_backend_device_info device = {0};
    yvex_backend_bandwidth_evidence bandwidth = {0};
    yvex_runtime_residency_summary residency = {0};
    int implementation_ready, workspace_ready, graph_ready, rc;
    capabilities.attention_workspace_ready =
        session->attention_workspace && session->summary.workspace_bytes;
    if (session->attention_state_provider_ready && session->state_residency &&
        session->attention_state_provider.summary(session->attention_state_provider.context,
                                                  &state, err) == YVEX_OK &&
        yvex_runtime_state_residency_summary_copy(session->state_residency, &state_residency,
                                                  err) == YVEX_OK)
        capabilities.persistent_kv_ready =
            state.sealed && state.persistent && state.position_consistent &&
            state.prepared_layer_count == state.layer_count && state_residency.sealed &&
            !state_residency.invalidated &&
            state_residency.layer_count == state.prepared_layer_count &&
            (session->summary.backend == YVEX_BACKEND_KIND_CPU || state_residency.cuda_ready);
    else
        yvex_error_clear(err);
    rc = yvex_backend_get_device_info(session->backend, &device, err);
    if (rc == YVEX_OK) {
        session->summary.device_index = device.device_index;
        session->summary.compute_capability_major = device.compute_capability_major;
        session->summary.compute_capability_minor = device.compute_capability_minor;
        session->summary.total_device_bytes = device.total_memory_bytes;
        yvex_core_text_copy(session->summary.device_name, sizeof(session->summary.device_name),
                            device.name ? device.name : "unavailable");
    }
    if (rc != YVEX_OK)
        return runtime_refuse_as(failure, YVEX_RUNTIME_REFUSE_DEVICE_CAPABILITY,
                                 1ull, 0ull, (yvex_status)rc, err);
    if (session->summary.backend == YVEX_BACKEND_KIND_CPU) {
        session->summary.capabilities = capabilities;
        return YVEX_OK;
    }
    rc = yvex_backend_bandwidth_probe(session->backend, &bandwidth, err);
    if (rc != YVEX_OK)
        return runtime_refuse_as(failure, YVEX_RUNTIME_REFUSE_DEVICE_CAPABILITY,
                                 1ull, 0ull, (yvex_status)rc, err);
    session->summary.sustainable_read_bytes_per_second =
        bandwidth.sustainable_read_bytes_per_second;
    session->summary.sustainable_copy_bytes_per_second =
        bandwidth.sustainable_copy_bytes_per_second;
    session->summary.sustainable_coherent_host_bytes_per_second =
        bandwidth.sustainable_coherent_host_bytes_per_second;
    yvex_runtime_identity_copy(session->summary.bandwidth_evidence_identity,
                               bandwidth.identity);
    rc = session->model->view.residency
             ? yvex_runtime_residency_snapshot(session->model->view.residency, &residency,
                                               NULL, NULL, err) : YVEX_OK;
    if (rc == YVEX_OK)
        rc = yvex_backend_query_capability(session->backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                           &encoded, err);
    if (rc != YVEX_OK)
        return runtime_refuse_as(failure, YVEX_RUNTIME_REFUSE_CUDA_CAPABILITY,
                                 1ull, 0ull, (yvex_status)rc, err);
    implementation_ready =
        capabilities.cuda_eager_implemented &&
        yvex_backend_status_of(session->backend) == YVEX_BACKEND_STATUS_READY &&
        encoded.state == YVEX_BACKEND_CAPABILITY_SUPPORTED &&
        device.kind == YVEX_BACKEND_KIND_CUDA && device.compute_capability_major > 0 &&
        residency.core_binding_count == session->model->summary.attention_binding_count &&
        session->summary.resident_binding_count == residency.binding_count &&
        residency.cuda_addressable_bytes > 0ull;
    workspace_ready = session->summary.host_workspace_owned && session->summary.host_workspace_pinned &&
                      session->summary.host_workspace_bytes && session->workspace &&
                      session->summary.device_workspace_bytes;
    if (require_workspace && (!implementation_ready || !workspace_ready))
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_CUDA_EAGER, 1ull, 0ull, err);
    if (!implementation_ready || !workspace_ready) {
        session->summary.capabilities = capabilities;
        return YVEX_OK;
    }
    capabilities.cuda_prefill_eager_ready = 1;
    capabilities.cuda_decode_eager_ready = 1;
    if (yvex_backend_cuda_graph_query(session->backend, &graph, err) != YVEX_OK) {
        graph = (yvex_backend_cuda_graph_capability){0};
        yvex_error_clear(err);
    }
    graph_ready = graph.state == YVEX_BACKEND_CUDA_GRAPH_OPEN && graph.stream_api_available &&
                  graph.graph_api_available && graph.update_api_available &&
                  graph.edge_inventory_available && graph.async_memory_available &&
                  graph.async_copy_available && graph.pinned_host_memory_available;
    capabilities.cuda_prefill_piecewise_graph_ready =
        capabilities.cuda_decode_piecewise_graph_ready =
        capabilities.cuda_piecewise_graph_implemented && graph_ready;
    capabilities.cuda_prefill_full_graph_ready = capabilities.cuda_decode_full_graph_ready =
        capabilities.cuda_full_graph_implemented && graph_ready;
    session->summary.capabilities = capabilities;
    return YVEX_OK;
}

int yvex_runtime_private_session_workspace_discard(yvex_runtime_execution_session *session,
                                             yvex_error *err) {
    yvex_backend_host_workspace_summary remaining;
    yvex_error cleanup, first_error;
    int first_rc, rc;
    if (!session || !session->backend) {
        return yvex_runtime_private_success(err);
    }
    yvex_error_clear(&first_error);
    first_rc = yvex_backend_host_workspace_detach(session->backend, &first_error);
    memset(&remaining, 0, sizeof(remaining));
    session->host_workspace_cleanup_pending =
        first_rc != YVEX_OK &&
        yvex_backend_host_workspace_summary_get(session->backend, &remaining) &&
        remaining.attached && remaining.owned;
    yvex_backend_workspace_detach(session->backend);
    yvex_error_clear(&cleanup);
    rc = session->workspace
             ? yvex_backend_tensor_release(session->backend, &session->workspace, &cleanup)
             : YVEX_OK;
    if (first_rc == YVEX_OK && rc != YVEX_OK)
        first_rc = rc, first_error = cleanup;
    if (first_rc == YVEX_OK) yvex_error_clear(err);
    else if (err) *err = first_error;
    return first_rc;
}

static int runtime_session_resources_release(yvex_runtime_execution_session *session,
                                             yvex_error *err) {
    yvex_error cleanup;
    int rc;
    if (!session) {
        return yvex_runtime_private_success(err);
    }
    if (getenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE")) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.resource_cleanup",
                       "injected runtime session resource cleanup failure");
        return YVEX_ERR_STATE;
    }
    if (session->backend || session->invalidation_pending) {
        rc = yvex_runtime_private_session_invalidate(session, session->invalidation_pending, &cleanup);
        if (rc != YVEX_OK) {
            if (err) *err = cleanup;
            return rc;
        }
    }
    session->invalidation_pending = 0;
    if (session->state_resolver_attached) {
        yvex_backend_state_residency_detach(session->backend);
        session->state_resolver_attached = 0;
    }
    if (session->state_residency) {
        rc = yvex_runtime_state_residency_close(
            &session->state_residency, &cleanup);
        session->view.state_residency = session->state_residency;
        if (rc != YVEX_OK) {
            if (err) *err = cleanup;
            return rc;
        }
    }
    if (session->draft_state_residency) {
        rc = yvex_runtime_state_residency_close(
            &session->draft_state_residency, &cleanup);
        session->view.draft_state_residency = session->draft_state_residency;
        if (rc != YVEX_OK) {
            if (err) *err = cleanup;
            return rc;
        }
    }
    if (session->draft_attention_state_provider_ready < 0) {
        rc = session->draft_attention_state_factory.discard(
            session->draft_attention_state_factory.context,
            &session->draft_attention_state_provider, &cleanup);
        if (rc == YVEX_OK && session->draft_attention_state_provider.context) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(&cleanup, YVEX_ERR_STATE, "runtime.session.state-provider",
                           "draft attention state factory retained ownership after discard");
        }
        if (rc != YVEX_OK) { if (err) *err = cleanup; return rc; }
        session->draft_attention_state_provider_ready = 0;
    }
    if (session->draft_attention_state_provider_ready > 0) {
        rc = session->draft_attention_state_provider.release(
            &session->draft_attention_state_provider.context, &cleanup);
        if (rc != YVEX_OK) {
            if (err) *err = cleanup;
            return rc;
        }
        memset(&session->draft_attention_state_provider, 0,
               sizeof(session->draft_attention_state_provider));
        session->draft_attention_state_provider_ready = 0;
        session->view.draft_attention_state_provider = NULL;
    }
    if (session->attention_state_provider_ready < 0) {
        rc = session->attention_state_factory.discard(session->attention_state_factory.context,
            &session->attention_state_provider, &cleanup);
        if (rc == YVEX_OK && session->attention_state_provider.context) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(&cleanup, YVEX_ERR_STATE, "runtime.session.state-provider",
                           "attention state factory retained ownership after discard");
        }
        if (rc != YVEX_OK) { if (err) *err = cleanup; return rc; }
        session->attention_state_provider_ready = 0;
    }
    if (session->attention_state_provider_ready > 0) {
        rc = session->attention_state_provider.release(&session->attention_state_provider.context,
                                                       &cleanup);
        if (rc != YVEX_OK) {
            if (err) *err = cleanup;
            return rc;
        }
        memset(&session->attention_state_provider, 0, sizeof(session->attention_state_provider));
        session->attention_state_provider_ready = 0;
        session->view.attention_state_provider = NULL;
    }
    yvex_attention_workspace_close(&session->attention_workspace);
    session->view.attention_workspace = session->attention_workspace;
    if (session->backend) {
        rc = yvex_backend_resident_detach(session->backend, err);
        if (rc != YVEX_OK) return rc;
        rc = yvex_runtime_private_session_workspace_discard(session, err);
        if (rc != YVEX_OK) return rc;
        rc = yvex_backend_close_checked(&session->backend, &cleanup);
        session->view.backend = session->backend;
        if (rc != YVEX_OK) {
            if (err) *err = cleanup;
            return rc;
        }
    }
    session->host_workspace_cleanup_pending = 0;
    return yvex_runtime_private_success(err);
}

static int runtime_session_storage_release(yvex_runtime_execution_session *session,
                                           yvex_error *err) {
    if (!session) {
        return yvex_runtime_private_success(err);
    }
    if (session->idle_condition_ready &&
        pthread_cond_destroy(&session->idle_condition) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.condition_destroy",
                       "runtime session condition cleanup failed");
        return YVEX_ERR_STATE;
    }
    session->idle_condition_ready = 0;
    if (session->lifecycle_mutex_ready &&
        pthread_mutex_destroy(&session->lifecycle_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.mutex_destroy",
                       "runtime session mutex cleanup failed");
        return YVEX_ERR_STATE;
    }
    session->lifecycle_mutex_ready = 0;
    memset(session, 0, sizeof(*session));
    free(session);
    return yvex_runtime_private_success(err);
}

static int runtime_session_open_fail(yvex_runtime_execution_session **out,
                                     yvex_runtime_execution_session *session,
                                     int status, yvex_runtime_model_failure *failure,
                                     yvex_error *err) {
    yvex_error primary = err ? *err : (yvex_error){0};
    int cleanup_rc = YVEX_OK;
    if (session) {
        session->closing = 1;
        session->summary.open = 0;
        cleanup_rc = runtime_session_resources_release(session, err);
        if (cleanup_rc == YVEX_OK)
            cleanup_rc = runtime_session_model_discharge(session, err);
        if (cleanup_rc == YVEX_OK)
            cleanup_rc = runtime_session_storage_release(session, err);
    }
    if (cleanup_rc != YVEX_OK) {
        if (out) *out = session;
        return runtime_refuse_as(
            failure, YVEX_RUNTIME_REFUSE_SESSION_OPEN_CLEANUP, 0ull, 1ull,
            (yvex_status)cleanup_rc, err);
    }
    if (err) *err = primary;
    return status;
}

static int runtime_session_state_open(
    yvex_runtime_execution_session *session, yvex_runtime_model *model,
    const yvex_attention_state_provider_factory *factory,
    unsigned long long state_budget, yvex_runtime_model_failure *failure,
    yvex_error *err)
{
    yvex_attention_failure state_failure = {0};
    int rc;

    if (factory && (!factory->open || !factory->discard)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.session.state-provider",
                       "attention state factory requires open and discard operations");
        return YVEX_ERR_INVALID_ARG;
    }
    if (factory) {
        session->attention_state_factory = *factory;
        rc = factory->open(factory->context, model->attention,
                           state_budget, &session->attention_state_provider,
                           &state_failure, err);
    } else {
        rc = yvex_attention_state_provider_open_persistent(
            model->attention, state_budget,
            &session->attention_state_provider, &state_failure, err);
    }
    if (rc != YVEX_OK ||
        !runtime_attention_state_provider_valid(
            &session->attention_state_provider)) {
        if (factory)
            session->attention_state_provider_ready = -1;
        else if (session->attention_state_provider.context &&
                 session->attention_state_provider.release)
            session->attention_state_provider_ready = 1;
        return yvex_runtime_private_reject(
            failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH, "attention-state",
            1ull, 0ull,
            state_failure.reason
                ? state_failure.reason
                : "runtime attention state provider could not open",
            err, rc == YVEX_OK ? YVEX_ERR_FORMAT : (yvex_status)rc);
    }
    session->attention_state_provider_ready = 1;
    session->view.attention_state_provider = &session->attention_state_provider;
    if (!model->draft_attention) return YVEX_OK;

    memset(&state_failure, 0, sizeof(state_failure));
    if (factory) {
        session->draft_attention_state_factory = *factory;
        rc = factory->open(factory->context, model->draft_attention,
                           state_budget,
                           &session->draft_attention_state_provider,
                           &state_failure, err);
    } else {
        rc = yvex_attention_state_provider_open_persistent(
            model->draft_attention, state_budget,
            &session->draft_attention_state_provider, &state_failure, err);
    }
    if (rc != YVEX_OK ||
        !runtime_attention_state_provider_valid(
            &session->draft_attention_state_provider)) {
        if (factory)
            session->draft_attention_state_provider_ready = -1;
        else if (session->draft_attention_state_provider.context &&
                 session->draft_attention_state_provider.release)
            session->draft_attention_state_provider_ready = 1;
        return yvex_runtime_private_reject(
            failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH,
            "draft-attention-state", 1ull, 0ull,
            state_failure.reason
                ? state_failure.reason
                : "runtime draft attention state provider could not open",
            err, rc == YVEX_OK ? YVEX_ERR_FORMAT : (yvex_status)rc);
    }
    session->draft_attention_state_provider_ready = 1;
    session->view.draft_attention_state_provider =
        &session->draft_attention_state_provider;
    return YVEX_OK;
}

int yvex_runtime_session_open(yvex_runtime_execution_session **out,
                              yvex_runtime_model *model,
                              const yvex_runtime_session_open_request *request,
                              yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_residency_summary residency_storage;
    const yvex_runtime_residency_summary *residency = NULL;
    const yvex_attention_state_provider_factory *state_factory =
        request ? request->attention_state_factory : NULL;
    const yvex_graph_execution_api *graph;
    yvex_backend_options backend_options;
    unsigned long long workspace_bytes = 0ull, draft_workspace_bytes = 0ull;
    unsigned long long admitted_host_bytes = 0ull, state_budget;
    int rc, publishable, uploaded = 0;
    if (out)
        *out = NULL;
    if (!out || !model || !request ||
        (request->backend != YVEX_BACKEND_KIND_CPU &&
         request->backend != YVEX_BACKEND_KIND_CUDA))
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_SESSION_REQUEST, 1ull, 0ull, err);
    session = (yvex_runtime_execution_session *)calloc(1u, sizeof(*session));
    if (!session)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_SESSION_ALLOCATION,
                              sizeof(*session), 0ull, err);
    session->model = model;
    session->view.model = model;
    if (pthread_mutex_init(&session->lifecycle_mutex, NULL) != 0) {
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_SESSION_LOCK_INITIALIZATION,
                            1ull, 0ull, err);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    session->lifecycle_mutex_ready = 1;
    if (pthread_cond_init(&session->idle_condition, NULL) != 0) {
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_SESSION_CONDITION_INITIALIZATION,
                            1ull, 0ull, err);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    session->idle_condition_ready = 1;
    rc = runtime_model_session_reserve(model, failure, err);
    if (rc != YVEX_OK) return runtime_session_open_fail(out, session, rc, failure, err);
    session->model_reserved = 1;
    rc = yvex_runtime_model_validate(model, failure, err);
    if (rc != YVEX_OK) return runtime_session_open_fail(out, session, rc, failure, err);
    session->maximum_host_bytes = request->maximum_host_bytes;
    session->maximum_device_bytes = request->maximum_device_bytes;
    session->summary.backend = request->backend;
    graph = model->view.graph;
    if (model->residency) {
        rc = yvex_runtime_residency_snapshot(model->residency, &residency_storage,
                                             NULL, NULL, err);
        if (rc != YVEX_OK) return runtime_session_open_fail(out, session, rc, failure, err);
        residency = &residency_storage;
    }
    rc = yvex_attention_workspace_capacity_resolve(graph, model->attention,
                                                   &workspace_bytes, err);
    if (rc == YVEX_OK && model->draft_attention)
        rc = yvex_attention_workspace_capacity_resolve(
            graph, model->draft_attention, &draft_workspace_bytes, err);
    if (draft_workspace_bytes > workspace_bytes)
        workspace_bytes = draft_workspace_bytes;
    if (rc != YVEX_OK ||
        !yvex_core_u64_add(residency ? residency->host_resident_bytes : 0ull,
                           workspace_bytes, &admitted_host_bytes) ||
        (request->maximum_host_bytes &&
         admitted_host_bytes > request->maximum_host_bytes)) {
        rc = yvex_runtime_private_reject(
            failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH, "attention-workspace",
            request->maximum_host_bytes, admitted_host_bytes,
            "attention workspace exceeds the admitted session host budget",
            err, rc == YVEX_OK ? YVEX_ERR_BOUNDS : (yvex_status)rc);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    rc = yvex_attention_workspace_open(&session->attention_workspace, workspace_bytes, err);
    if (rc != YVEX_OK) {
        rc = yvex_runtime_private_reject(
            failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH, "attention-workspace",
            workspace_bytes, 0ull, "attention workspace cold preparation failed",
            err, (yvex_status)rc);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    session->view.attention_workspace = session->attention_workspace;
    session->summary.workspace_bytes = workspace_bytes;
    session->summary.workspace_generation = 1ull;
    state_budget = request->maximum_host_bytes ? request->maximum_host_bytes - admitted_host_bytes
                                               : 0ull;
    rc = runtime_session_state_open(session, model, state_factory,
                                    state_budget, failure, err);
    if (rc != YVEX_OK)
        return runtime_session_open_fail(out, session, rc, failure, err);
    memset(&backend_options, 0, sizeof(backend_options));
    backend_options.kind = request->backend;
    backend_options.memory_limit_bytes = request->backend == YVEX_BACKEND_KIND_CUDA
                                             ? request->maximum_device_bytes
                                             : request->maximum_host_bytes;
    if (request->backend == YVEX_BACKEND_KIND_CUDA && model->residency) {
        rc = runtime_session_attach_cuda_residency(session, &uploaded, failure, err);
    } else {
        rc = yvex_backend_open(&session->backend, &backend_options, err);
    }
    session->view.backend = session->backend;
    if (rc != YVEX_OK) {
        yvex_runtime_private_failure_record(failure, YVEX_RUNTIME_MODEL_FAILURE_BACKEND,
                                     "backend-open", 1ull, 0ull,
                                     "runtime session backend could not be opened");
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    if (yvex_runtime_workspace_identity_compute(
            model->summary.runtime_model_identity, request->backend,
            request->maximum_host_bytes, request->maximum_device_bytes,
            workspace_bytes, 0ull, NULL, session->summary.workspace_identity,
            err) != YVEX_OK) {
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_IDENTITY, 1ull, 0ull, err);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    if (request->backend != YVEX_BACKEND_KIND_CUDA && residency) {
            session->summary.resident_binding_count = residency->binding_count;
            session->summary.resident_encoded_bytes = residency->encoded_bytes;
            session->summary.host_resident_bytes = residency->host_resident_bytes;
            session->summary.residency_generation = residency->generation;
            yvex_runtime_identity_copy(session->summary.residency_identity,
                                        residency->residency_identity);
    }
    if (getenv("YVEX_TEST_RUNTIME_SESSION_OPEN_FAILURE")) {
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_SESSION_RESOURCE_INJECTION,
                            0ull, 1ull, err);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    session->summary.peak_host_bytes = admitted_host_bytes;
    rc = yvex_runtime_private_session_capabilities_bind(session, failure, 0, err);
    if (rc != YVEX_OK)
        return runtime_session_open_fail(out, session, rc, failure, err);
    rc = yvex_runtime_model_validate(model, failure, err);
    if (rc != YVEX_OK)
        return runtime_session_open_fail(out, session, rc, failure, err);
    publishable = pthread_mutex_lock(&model->lifecycle_mutex) == 0;
    if (publishable) {
        publishable = model->summary.valid && !model->close_requested;
        if (publishable) {
            session->summary.open = 1;
            runtime_model_session_register_locked(model, session);
            *out = session;
        }
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    }
    if (!publishable) {
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_MODEL_DRAINING_PUBLICATION,
                            0ull, 1ull, err);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    if (failure)
        memset(failure, 0, sizeof(*failure));
    return yvex_runtime_private_success(err);
}

int yvex_runtime_session_close(yvex_runtime_execution_session **session_ptr, yvex_error *err) {
    yvex_runtime_execution_session *session;
    yvex_runtime_model *model;
    int rc;
    if (!session_ptr || !*session_ptr) {
        return yvex_runtime_private_success(err);
    }
    session = *session_ptr;
    if (!session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.close",
                       "runtime session lifecycle lock is unavailable");
        return YVEX_ERR_STATE;
    }
    session->closing = 1;
    while (session->summary.busy) {
        if (pthread_cond_wait(&session->idle_condition,
                              &session->lifecycle_mutex) != 0) {
            session->closing = 0;
            (void)pthread_mutex_unlock(&session->lifecycle_mutex);
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.close",
                           "runtime session drain wait failed");
            return YVEX_ERR_STATE;
        }
    }
    session->summary.open = 0;
    model = session->model;
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    if (session->model_registered) {
        rc = runtime_model_session_unregister(model, session, err);
        if (rc != YVEX_OK) return rc;
    }
    rc = runtime_session_resources_release(session, err);
    if (rc != YVEX_OK) return rc;
    rc = runtime_session_model_discharge(session, err);
    if (rc != YVEX_OK) return rc;
    rc = runtime_session_storage_release(session, err);
    if (rc != YVEX_OK) return rc;
    *session_ptr = NULL;
    return yvex_runtime_private_success(err);
}

int yvex_runtime_session_summary_copy(const yvex_runtime_execution_session *session,
                                      yvex_runtime_session_summary *out, yvex_error *err) {
    yvex_runtime_execution_session *mutable_session = (yvex_runtime_execution_session *)session;
    return runtime_summary_copy(
        session, session ? &session->summary : NULL, out, sizeof(*out),
        session ? session->lifecycle_mutex_ready : 0,
        session ? &mutable_session->lifecycle_mutex : NULL, "runtime.session.summary",
        "runtime session and summary output are required",
        "runtime session synchronization is unavailable", err);
}
/*
 * Borrow view.
 *
 * Session lifetime.
 */
const yvex_runtime_session_view *yvex_runtime_session_view_get(const yvex_runtime_execution_session *session) {
    return session ? &session->view : NULL;
}
/* Close one runtime lease in dependency order without losing retry ownership. */
int yvex_runtime_cleanup_lease_close(yvex_runtime_cleanup_lease **lease_ptr, yvex_error *err) {
    yvex_runtime_cleanup_lease *lease;
    int rc;
    if (!lease_ptr || !*lease_ptr) return yvex_runtime_private_success(err);
    lease = *lease_ptr;
    if (lease->dependent_context && !lease->dependent_release) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.cleanup.dependent", "dependent cleanup operation is unavailable");
        return YVEX_ERR_STATE;
    }
    if (lease->dependent_context) {
        rc = lease->dependent_release(&lease->dependent_context, err);
        if (rc != YVEX_OK) return rc;
        if (lease->dependent_context) {
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.cleanup.dependent",
                           "dependent cleanup reported success while retaining ownership");
            return YVEX_ERR_STATE;
        }
        lease->dependent_release = NULL;
    }
    rc = yvex_runtime_session_close(&lease->session, err);
    if (rc != YVEX_OK) return rc;
    yvex_runtime_model_close(&lease->model);
    if (lease->model) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.cleanup.model-close",
                       "runtime model cleanup retained ownership for retry");
        return YVEX_ERR_STATE;
    }
    free(lease);
    *lease_ptr = NULL;
    return yvex_runtime_private_success(err);
}

int yvex_runtime_cleanup_lease_adopt(yvex_runtime_cleanup_lease *lease, void *context,
    yvex_runtime_cleanup_release_fn release, yvex_error *err) {
    if (!lease || !context || !release || lease->dependent_context || lease->dependent_release) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.cleanup.adopt",
                       "one empty cleanup lease dependent slot is required");
        return YVEX_ERR_INVALID_ARG;
    }
    lease->dependent_context = context;
    lease->dependent_release = release;
    return yvex_runtime_private_success(err);
}

int yvex_runtime_cleanup_lease_acquire(
    yvex_runtime_cleanup_lease **out, const yvex_runtime_model_open_request *model_request,
    const yvex_runtime_session_open_request *session_request,
    yvex_runtime_model **borrowed_model,
    yvex_runtime_execution_session **borrowed_session,
    yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_runtime_cleanup_lease *lease;
    yvex_error primary, cleanup;
    int rc, cleanup_rc;
    if (borrowed_model) *borrowed_model = NULL;
    if (borrowed_session) *borrowed_session = NULL;
    if (!out || *out || !model_request || !borrowed_model ||
        (session_request && !borrowed_session))
        return yvex_runtime_private_refuse(
            failure, YVEX_RUNTIME_REFUSE_CLEANUP_LEASE, 1ull,
            out && !*out && model_request && borrowed_model &&
                    (!session_request || borrowed_session) ? 1ull : 0ull,
            err);
    lease = (yvex_runtime_cleanup_lease *)calloc(1u, sizeof(*lease));
    if (!lease)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_CLEANUP_LEASE_ALLOCATION, 1ull, 0ull, err);
    *out = lease;
    rc = yvex_runtime_model_open(&lease->model, model_request, failure, err);
    if (rc == YVEX_OK) *borrowed_model = lease->model;
    if (rc == YVEX_OK && session_request)
        rc = yvex_runtime_cleanup_lease_session_open(
            lease, session_request, borrowed_session, failure, err);
    if (rc == YVEX_OK) return YVEX_OK;
    *borrowed_model = NULL;
    if (borrowed_session) *borrowed_session = NULL;
    primary = err ? *err : (yvex_error){0};
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_cleanup_lease_close(out, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        yvex_runtime_private_failure_record(
            failure, YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, "cleanup-lease", 0ull, 1ull,
            "runtime acquisition cleanup retained ownership for retry");
        if (err) *err = cleanup;
        return cleanup_rc;
    }
    if (err) *err = primary;
    return rc;
}

int yvex_runtime_cleanup_lease_session_open(
    yvex_runtime_cleanup_lease *lease, const yvex_runtime_session_open_request *request,
    yvex_runtime_execution_session **borrowed_session,
    yvex_runtime_model_failure *failure, yvex_error *err) {
    int rc;
    if (borrowed_session) *borrowed_session = NULL;
    if (!lease || !lease->model || lease->session || !request || !borrowed_session)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_CLEANUP_LEASE_SESSION, 1ull, 0ull, err);
    rc = yvex_runtime_session_open(&lease->session, lease->model, request, failure, err);
    if (rc == YVEX_OK) *borrowed_session = lease->session;
    return rc;
}
