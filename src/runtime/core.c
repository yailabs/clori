/*
 * A runtime model owns the authenticated artifact, immutable execution plans, encoded weights,
 * and reusable model resources. Sessions borrow that model but own all mutable sequence state;
 * opening or closing one session cannot alter another.
 * Construction publishes nothing until artifact admission, materialization, backend preparation,
 * and family binding all agree. Cleanup runs in reverse dependency order and retains a failed owner
 * so callers can retry without losing the resource that still needs release.
 */
#include <yvex/internal/runtime.h>
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
/* One sealed runtime model owns the cold trust and compilation-free import lifecycle. */
struct yvex_runtime_model {
    const yvex_runtime_family_adapter *adapter;
    yvex_runtime_binding *binding;
    yvex_runtime_binding_summary binding_summary;
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
    yvex_complete_artifact_admission admission;
    yvex_materialization_plan *materialization_plan;
    yvex_materialization_session *materialization;
    yvex_runtime_descriptor *descriptor;
    yvex_attention_plan *attention;
    yvex_tokenizer *tokenizer;
    yvex_runtime_residency *residency;
    yvex_runtime_model_summary summary;
    yvex_runtime_model_view view;
    pthread_mutex_t lifecycle_mutex;
    struct yvex_runtime_execution_session *sessions;
    unsigned long long active_sessions;
    int lifecycle_mutex_ready, close_requested, dependent_invalidation_pending;
};
/* Each execution session owns one mutable backend context and its counters. */
struct yvex_runtime_execution_session {
    yvex_runtime_model *model;
    yvex_backend *backend;
    yvex_attention_state_provider attention_state_provider;
    yvex_attention_state_provider_factory attention_state_factory;
    yvex_attention_workspace *attention_workspace;
    yvex_runtime_state_residency *state_residency;
    yvex_device_tensor *workspace;
    yvex_runtime_session_summary summary;
    yvex_runtime_session_view view;
    pthread_mutex_t lifecycle_mutex;
    pthread_cond_t idle_condition;
    struct yvex_runtime_execution_session *model_previous, *model_next;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    int lifecycle_mutex_ready, idle_condition_ready, closing, model_registered, model_reserved;
    int model_release_pending, invalidation_pending, host_workspace_cleanup_pending, attention_state_provider_ready;
};
/* One exclusive lease keeps cold model and mutable session owners reachable across cleanup faults. */
struct yvex_runtime_cleanup_lease {
    yvex_runtime_model *model;
    yvex_runtime_execution_session *session;
    void *dependent_context;
    yvex_runtime_cleanup_release_fn dependent_release;
};

static void runtime_model_failure_record(yvex_runtime_model_failure *failure,
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

static int runtime_model_reject(yvex_runtime_model_failure *failure,
                                yvex_runtime_model_failure_code code,
                                const char *field, unsigned long long expected,
                                unsigned long long actual, const char *reason,
                                yvex_error *err, yvex_status status) {
    runtime_model_failure_record(failure, code, field, expected, actual, reason);
    yvex_error_set(err, status, "runtime.model", reason);
    return status;
}
typedef enum {
    REFUSE_MODEL_LOCK_UNAVAILABLE = 0, REFUSE_MODEL_INVALID_OR_DRAINING,
    REFUSE_ARTIFACT_IDENTITY, REFUSE_MODEL_OPEN_REQUEST, REFUSE_FAMILY_ADAPTER,
    REFUSE_MODEL_ALLOCATION, REFUSE_MODEL_LOCK_INITIALIZATION, REFUSE_MODEL_REQUIRED, REFUSE_MODEL_UNSEALED,
    REFUSE_DRIFT_COUNTER, REFUSE_HOST_RESIDENCY, REFUSE_CUDA_EAGER, REFUSE_SESSION_REQUEST,
    REFUSE_SESSION_ALLOCATION, REFUSE_SESSION_LOCK_INITIALIZATION, REFUSE_SESSION_CONDITION_INITIALIZATION,
    REFUSE_WORKSPACE_IDENTITY, REFUSE_SESSION_RESOURCE_INJECTION, REFUSE_MODEL_DRAINING_PUBLICATION,
    REFUSE_WORKSPACE_REQUEST, REFUSE_WORKSPACE_LOCK, REFUSE_WORKSPACE_SESSION_STATE, REFUSE_WORKSPACE_STATE,
    REFUSE_WORKSPACE_ALREADY_SEALED, REFUSE_WORKSPACE_BUDGET,
    REFUSE_WORKSPACE_CAPABILITY_INJECTION, REFUSE_SESSION_REQUIRED,
    REFUSE_SESSION_INVALIDATED, REFUSE_SESSION_CLOSING, REFUSE_SESSION_BUSY,
    REFUSE_SESSION_CANCELLED, REFUSE_BINDING_ADMISSION, REFUSE_ADAPTER_CAPABILITY,
    REFUSE_ADAPTER_CAPABILITY_STALE, REFUSE_ARTIFACT_DRIFT, REFUSE_DEVICE_CAPABILITY, REFUSE_CUDA_CAPABILITY,
    REFUSE_SESSION_OPEN_CLEANUP, REFUSE_DEVICE_WORKSPACE_BUDGET, REFUSE_CLEANUP_LEASE,
    REFUSE_CLEANUP_LEASE_ALLOCATION, REFUSE_CLEANUP_LEASE_SESSION, REFUSE_OPEN_BINDING, REFUSE_OPEN_ADAPTER,
    REFUSE_OPEN_LOGICAL_TRANSFORM, REFUSE_OPEN_ARTIFACT, REFUSE_OPEN_MATERIALIZATION, REFUSE_OPEN_IMPORT,
    REFUSE_OPEN_IMPORTED_IDENTITY, REFUSE_OPEN_TOKENIZER, REFUSE_OPEN_SEAL, REFUSE_OPEN_BUILD,
    REFUSE_OPEN_CAPABILITIES,
    REFUSE_OPEN_RESIDENCY, REFUSE_OPEN_RESIDENCY_COMPLETE, REFUSE_OPEN_DRIFT, REFUSE_COUNT
} runtime_refusal_id;
typedef struct {
    yvex_runtime_model_failure_code code;
    yvex_status status;
    const char *field, *reason;
} runtime_refusal_spec;
/* Ordered with runtime_refusal_id so one typed row owns each stable refusal contract. */
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
    {YVEX_RUNTIME_MODEL_FAILURE_ARTIFACT, YVEX_ERR_FORMAT, "artifact-open", "artifact admission failed"},
    {YVEX_RUNTIME_MODEL_FAILURE_MATERIALIZATION, YVEX_ERR_FORMAT, "runtime-materialization",
     "runtime binding materialization could not be reopened"},
    {YVEX_RUNTIME_MODEL_FAILURE_BINDING, YVEX_ERR_FORMAT, "runtime-import",
     "runtime binding import did not reconstruct sealed runtime facts"},
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
_Static_assert(sizeof(runtime_refusals) / sizeof(runtime_refusals[0]) == REFUSE_COUNT,
               "runtime refusal catalog must cover every identity");

static int runtime_refuse_as(yvex_runtime_model_failure *failure, runtime_refusal_id id,
    unsigned long long expected, unsigned long long actual, yvex_status status, yvex_error *err) {
    const runtime_refusal_spec *spec;
    if ((unsigned int)id >= REFUSE_COUNT) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.refusal",
                       "runtime refusal identity is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    spec = &runtime_refusals[id];
    return runtime_model_reject(failure, spec->code, spec->field, expected, actual,
                                spec->reason, err, status == YVEX_OK ? spec->status : status);
}

static int runtime_refuse(yvex_runtime_model_failure *failure, runtime_refusal_id id,
    unsigned long long expected, unsigned long long actual, yvex_error *err) {
    return runtime_refuse_as(failure, id, expected, actual, YVEX_OK, err);
}

static int runtime_success(yvex_error *err) {
    yvex_error_clear(err);
    return YVEX_OK;
}

static int runtime_attention_state_provider_valid(const yvex_attention_state_provider *provider) {
    return provider && provider->schema_version == YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V2 &&
           provider->context && provider->prepare && provider->summary && provider->view &&
           provider->identity && provider->begin && provider->stage && provider->commit && provider->abort &&
           provider->reset && provider->invalidate && provider->release;
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
                                        const yvex_runtime_family_adapter *adapter,
                                        char output[YVEX_SHA256_HEX_CAP]) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!binding || !adapter ||
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
        !yvex_sha256_update_u64(&hash, adapter->adapter_id) ||
        !yvex_sha256_update_u64(&hash, adapter->adapter_version) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

const yvex_runtime_family_adapter *yvex_runtime_family_adapter_find(const char *target_id) {
    unsigned long long index;
    if (!target_id)
        return NULL;
    for (index = 0ull;; ++index) {
        const yvex_runtime_family_adapter *adapter = yvex_graph_runtime_family_at(index);
        if (!adapter)
            break;
        if (adapter->target_id && strcmp(adapter->target_id, target_id) == 0)
            return adapter;
    }
    return NULL;
}

static int runtime_model_session_reserve(yvex_runtime_model *model,
                                         yvex_runtime_model_failure *failure,
                                         yvex_error *err) {
    int accepted;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return runtime_refuse(failure, REFUSE_MODEL_LOCK_UNAVAILABLE, 1ull, 0ull, err);
    accepted = model->summary.sealed && model->summary.valid && !model->close_requested;
    if (accepted && !yvex_core_u64_add(model->active_sessions, 1ull, &model->active_sessions))
        accepted = 0;
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    if (!accepted)
        return runtime_refuse(failure, REFUSE_MODEL_INVALID_OR_DRAINING, 0ull, 1ull, err);
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
    return runtime_success(err);
}

static int runtime_session_invalidate(yvex_runtime_execution_session *session,
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
    if (session->backend && session->backend->kind == YVEX_BACKEND_KIND_CUDA) {
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
            rc = runtime_session_invalidate(session, 1, &cleanup);
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
    yvex_tokenizer_close(model->tokenizer);
    model->tokenizer = NULL;
    yvex_materialization_session_close(model->materialization);
    model->materialization = NULL;
    if (model->adapter && model->adapter->graph && model->adapter->graph() &&
        model->adapter->graph()->plan_close)
        model->adapter->graph()->plan_close(model->attention);
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
    return runtime_success(err);
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
        return runtime_success(err);
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
    return runtime_success(err);
}

static int runtime_model_open_fail(yvex_runtime_model **out, yvex_runtime_model *model,
                                   yvex_runtime_model_failure *failure,
                                   runtime_refusal_id refusal,
                                   unsigned long long expected, unsigned long long actual,
                                   yvex_error *err, yvex_status status) {
    const runtime_refusal_spec *spec = &runtime_refusals[refusal];
    yvex_error primary;
    int cleanup_rc;
    (void)runtime_model_reject(
        failure, spec->code, spec->field, expected, actual, spec->reason, err, status);
    primary = err ? *err : (yvex_error){0};
    cleanup_rc = runtime_model_release(model, err);
    if (cleanup_rc != YVEX_OK) {
        *out = model;
        runtime_model_failure_record(
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
        return runtime_refuse(failure, REFUSE_BINDING_ADMISSION, 1ull, 0ull, err);
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
        rc = runtime_refuse(failure, REFUSE_ARTIFACT_IDENTITY, 1ull, 0ull, err);
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

static int runtime_model_capabilities_bind(
    yvex_runtime_model *model, const yvex_runtime_binding_summary *binding,
    const yvex_attention_summary *attention,
    yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_runtime_capabilities *capabilities = &model->summary.capabilities;
    yvex_runtime_capabilities declared;
    char binding_identity[YVEX_SHA256_HEX_CAP];
    const int graph_ready = attention &&
        attention->history_contract_ready && attention->full_execution_ready;
    const int cpu_ready = graph_ready && attention->cpu_reference_ready;
    const int cuda_ready = graph_ready && attention->cuda_execution_ready;
    memset(&declared, 0, sizeof(declared));
    if (!model || !binding || !model->adapter ||
        !model->adapter->execution_capabilities ||
        !model->adapter->execution_capabilities(&declared))
        return runtime_refuse(failure, REFUSE_ADAPTER_CAPABILITY, 1ull, 0ull, err);
    if (!yvex_runtime_capabilities_admitted_by(&binding->capabilities, &declared) ||
        !yvex_runtime_capabilities_identity(&binding->capabilities, binding_identity) ||
        strcmp(binding_identity, binding->execution_capability_identity) != 0)
        return runtime_refuse(failure, REFUSE_ADAPTER_CAPABILITY_STALE, 1ull, 0ull, err);
    *capabilities = binding->capabilities;
    capabilities->attention_semantics_ready = declared.attention_semantics_ready && graph_ready;
    capabilities->attention_core_ready = declared.attention_core_ready && graph_ready;
    capabilities->attention_envelope_ready = declared.attention_envelope_ready && graph_ready;
    capabilities->cpu_prefill_eager_ready = declared.cpu_prefill_eager_ready && cpu_ready;
    capabilities->cpu_decode_eager_ready = declared.cpu_decode_eager_ready && cpu_ready;
    capabilities->cuda_eager_implemented = declared.cuda_eager_implemented && cuda_ready;
    capabilities->cuda_piecewise_graph_implemented =
        declared.cuda_piecewise_graph_implemented && cuda_ready;
    capabilities->cuda_full_graph_implemented = declared.cuda_full_graph_implemented && cuda_ready;
    capabilities->attention_state_delta_ready =
        declared.attention_state_delta_ready && attention->state_delta_contract_ready;
    return YVEX_OK;
}

int yvex_runtime_model_open(yvex_runtime_model **out, const yvex_runtime_model_open_request *request,
                            yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_runtime_model *model = NULL;
    const yvex_runtime_family_adapter *adapter;
    const yvex_runtime_descriptor_summary *descriptor_summary;
    const yvex_attention_summary *attention_summary;
    yvex_runtime_binding_failure binding_failure;
    yvex_materialization_options materialization_options;
    yvex_runtime_residency_options residency_options;
    yvex_runtime_residency_failure residency_failure;
    yvex_runtime_residency_summary residency_summary;
    unsigned long long total_started, phase_started;
    int rc;
    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!out || !request || !request->artifact_path || !request->runtime_binding_path || !request->target_id)
        return runtime_refuse(failure, REFUSE_MODEL_OPEN_REQUEST, 1ull, 0ull, err);
    adapter = yvex_runtime_family_adapter_find(request->target_id);
    if (!adapter || adapter->schema_version != YVEX_RUNTIME_FAMILY_ADAPTER_SCHEMA_V1 || !adapter->adapter_id ||
        !adapter->graph || !yvex_sha256_hex_is_valid(adapter->logical_transform_identity))
        return runtime_refuse(failure, REFUSE_FAMILY_ADAPTER,
                              YVEX_RUNTIME_FAMILY_ADAPTER_SCHEMA_V1, 0ull, err);
    model = (yvex_runtime_model *)calloc(1u, sizeof(*model));
    if (!model) return runtime_refuse(failure, REFUSE_MODEL_ALLOCATION, sizeof(*model), 0ull, err);
    if (pthread_mutex_init(&model->lifecycle_mutex, NULL) != 0) {
        free(model);
        return runtime_refuse(failure, REFUSE_MODEL_LOCK_INITIALIZATION, 1ull, 0ull, err);
    }
    model->lifecycle_mutex_ready = 1;
    model->adapter = adapter;
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
            out, model, failure, REFUSE_OPEN_BINDING, 1ull, 0ull, err, (yvex_status)rc);
    if (model->binding_summary.family_adapter_id != adapter->adapter_id ||
        model->binding_summary.family_adapter_version != adapter->adapter_version)
        return runtime_model_open_fail(
            out, model, failure, REFUSE_OPEN_ADAPTER, adapter->adapter_id,
            model->binding_summary.family_adapter_id, err, YVEX_ERR_FORMAT);
    if (strcmp(model->binding_summary.logical_transform_identity,
               adapter->logical_transform_identity) != 0)
        return runtime_model_open_fail(
            out, model, failure, REFUSE_OPEN_LOGICAL_TRANSFORM, 1ull, 0ull, err,
            YVEX_ERR_FORMAT);
    rc = runtime_model_artifact_open(model, request, &model->binding_summary, failure, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, REFUSE_OPEN_ARTIFACT, 1ull, 0ull, err, (yvex_status)rc);
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
            out, model, failure, REFUSE_OPEN_MATERIALIZATION, 1ull, 0ull, err,
            (yvex_status)rc);
    phase_started = yvex_core_monotonic_ns();
    rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL, 0ull, 1ull, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_binding_import_graph(
            model->binding, model->materialization, &model->descriptor,
            &model->attention, &binding_failure, err);
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
            out, model, failure, REFUSE_OPEN_IMPORT, 1ull, 0ull, err, (yvex_status)rc);
    descriptor_summary = yvex_runtime_descriptor_summary_get(model->descriptor);
    attention_summary = model->adapter->graph()->plan_summary(model->attention);
    if (!descriptor_summary || !attention_summary ||
        strcmp(descriptor_summary->runtime_descriptor_identity,
               model->binding_summary.runtime_descriptor_identity) != 0 ||
        strcmp(attention_summary->attention_plan_identity,
               model->binding_summary.attention_plan_identity) != 0 ||
        !runtime_model_identity_build(&model->binding_summary, model->adapter,
                                      model->summary.runtime_model_identity)) {
        return runtime_model_open_fail(
            out, model, failure, REFUSE_OPEN_IMPORTED_IDENTITY, 1ull, 0ull, err,
            YVEX_ERR_FORMAT);
    }
    rc = yvex_tokenizer_from_gguf(&model->tokenizer, model->gguf, NULL, err);
    if (rc == YVEX_OK && yvex_tokenizer_plan_summary_get(model->tokenizer))
        rc = yvex_tokenizer_bind_runtime(
            model->tokenizer, model->binding_summary.artifact_identity,
            model->binding_summary.logical_model_identity,
            model->binding_summary.runtime_descriptor_identity, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, REFUSE_OPEN_TOKENIZER, 1ull, 0ull, err,
            (yvex_status)rc);
    runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL, phase_started);
    rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL, 1ull, 1ull, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, REFUSE_OPEN_SEAL, 1ull, 0ull, err, (yvex_status)rc);
    model->summary.schema_version = YVEX_RUNTIME_MODEL_SCHEMA_V1;
    model->summary.sealed = model->summary.valid = 1;
    rc = runtime_model_once(&model->summary.runtime_model_builds, "runtime.model.seal", err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, REFUSE_OPEN_BUILD, 1ull,
            model->summary.runtime_model_builds, err, (yvex_status)rc);
    yvex_runtime_identity_copy(model->summary.runtime_binding_identity, model->binding_summary.identity);
    yvex_runtime_identity_copy(model->summary.artifact_identity, model->binding_summary.artifact_identity);
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
    model->summary.attention_binding_count = attention_summary->required_binding_count;
    rc = runtime_model_capabilities_bind(model, &model->binding_summary, attention_summary, failure, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, REFUSE_OPEN_CAPABILITIES, 1ull, 0ull, err,
            (yvex_status)rc);
    model->view.binding = &model->binding_summary;
    model->view.adapter = model->adapter;
    model->view.attention = model->attention;
    model->view.descriptor = model->descriptor;
    model->view.tokenizer = model->tokenizer;
    model->view.materialization = model->materialization;
    if (attention_summary->required_binding_count) {
        phase_started = yvex_core_monotonic_ns();
        rc = runtime_model_progress(request, YVEX_RUNTIME_LIFECYCLE_RESIDENCY,
                                    0ull, attention_summary->required_binding_count, err);
        memset(&residency_options, 0, sizeof(residency_options));
        residency_options.maximum_host_bytes = request->maximum_host_bytes;
        memset(&residency_failure, 0, sizeof(residency_failure));
        if (rc == YVEX_OK)
            rc = yvex_runtime_residency_prepare(&model->residency, model, &residency_options,
                                                &residency_failure, err);
        runtime_model_timing(model, YVEX_RUNTIME_LIFECYCLE_RESIDENCY, phase_started);
        if (rc != YVEX_OK)
            return runtime_model_open_fail(
                out, model, failure, REFUSE_OPEN_RESIDENCY, 1ull, 0ull, err,
                (yvex_status)rc);
        model->view.residency = model->residency;
        memset(&residency_summary, 0, sizeof(residency_summary));
        rc = yvex_runtime_residency_snapshot(
            model->residency, &residency_summary, NULL, NULL, err);
        if (rc != YVEX_OK || !residency_summary.model_complete ||
            !residency_summary.host_locked ||
            residency_summary.binding_count != descriptor_summary->tensor_count ||
            residency_summary.encoded_bytes != descriptor_summary->payload_bytes ||
            !residency_summary.core_complete || !residency_summary.envelope_complete ||
            !yvex_runtime_logits_residency_admit(&model->summary.capabilities, &residency_summary))
            return runtime_model_open_fail(
                out, model, failure, REFUSE_OPEN_RESIDENCY_COMPLETE, 1ull, 0ull, err,
                rc == YVEX_OK ? YVEX_ERR_FORMAT : (yvex_status)rc);
        model->summary.capabilities.attention_weight_residency_ready = 1;
        model->summary.capabilities.attention_envelope_ready =
            model->summary.capabilities.attention_envelope_ready &&
            residency_summary.envelope_complete;
    }
    rc = yvex_artifact_snapshot_validate(model->artifact, NULL, err);
    if (rc != YVEX_OK)
        return runtime_model_open_fail(
            out, model, failure, REFUSE_OPEN_DRIFT, 1ull, 0ull, err, (yvex_status)rc);
    model->summary.total_seconds = (double)(yvex_core_monotonic_ns() - total_started) / 1000000000.0;
    *out = model;
    if (failure) memset(failure, 0, sizeof(*failure));
    return runtime_success(err);
}

int yvex_runtime_model_validate(yvex_runtime_model *model,
                                yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_error cleanup;
    int cleanup_rc = YVEX_OK, counter_overflow, rc;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return runtime_refuse(failure, REFUSE_MODEL_REQUIRED, 1ull, 0ull, err);
    if (!model->summary.sealed || model->close_requested) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return runtime_refuse(failure, REFUSE_MODEL_UNSEALED, 0ull, 1ull, err);
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
        return runtime_refuse(failure, REFUSE_DRIFT_COUNTER, ULLONG_MAX, 1ull, err);
    return runtime_refuse_as(
        failure, REFUSE_ARTIFACT_DRIFT, 1ull, 0ull,
        rc == YVEX_OK ? YVEX_ERR_STATE : (yvex_status)rc, err);
cleanup_failed:
    runtime_model_failure_record(failure, YVEX_RUNTIME_MODEL_FAILURE_CLEANUP,
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
    return runtime_success(err);
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
        runtime_model_failure_record(
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

static int runtime_session_capabilities_bind(
    yvex_runtime_execution_session *session, yvex_runtime_model_failure *failure,
    int require_workspace, yvex_error *err) {
    yvex_runtime_capabilities capabilities = session->model->summary.capabilities;
    yvex_graph_attention_state_summary state = {0};
    yvex_runtime_state_residency_summary state_residency = {0};
    yvex_backend_capability_result encoded = {0};
    yvex_backend_cuda_graph_capability graph = {0};
    yvex_backend_device_info device = {0};
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
        return runtime_refuse_as(failure, REFUSE_DEVICE_CAPABILITY, 1ull, 0ull,
                                 (yvex_status)rc, err);
    if (session->summary.backend == YVEX_BACKEND_KIND_CPU) {
        session->summary.capabilities = capabilities;
        return YVEX_OK;
    }
    rc = session->model->view.residency
             ? yvex_runtime_residency_snapshot(session->model->view.residency, &residency,
                                               NULL, NULL, err) : YVEX_OK;
    if (rc == YVEX_OK)
        rc = yvex_backend_query_capability(session->backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                           &encoded, err);
    if (rc != YVEX_OK)
        return runtime_refuse_as(failure, REFUSE_CUDA_CAPABILITY, 1ull, 0ull,
                                 (yvex_status)rc, err);
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
        return runtime_refuse(failure, REFUSE_CUDA_EAGER, 1ull, 0ull, err);
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

static int runtime_session_workspace_discard(yvex_runtime_execution_session *session,
                                             yvex_error *err) {
    yvex_backend_host_workspace_summary remaining;
    yvex_error cleanup, first_error;
    int first_rc, rc;
    if (!session || !session->backend) {
        return runtime_success(err);
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
        return runtime_success(err);
    }
    if (getenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE")) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.resource_cleanup",
                       "injected runtime session resource cleanup failure");
        return YVEX_ERR_STATE;
    }
    if (session->backend || session->invalidation_pending) {
        rc = runtime_session_invalidate(session, session->invalidation_pending, &cleanup);
        if (rc != YVEX_OK) {
            if (err) *err = cleanup;
            return rc;
        }
    }
    session->invalidation_pending = 0;
    if (session->state_residency) {
        rc = yvex_runtime_state_residency_close(
            &session->state_residency, &cleanup);
        session->view.state_residency = session->state_residency;
        if (rc != YVEX_OK) {
            if (err) *err = cleanup;
            return rc;
        }
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
        rc = runtime_session_workspace_discard(session, err);
        if (rc != YVEX_OK) return rc;
        rc = yvex_backend_close_checked(&session->backend, &cleanup);
        session->view.backend = session->backend;
        if (rc != YVEX_OK) {
            if (err) *err = cleanup;
            return rc;
        }
    }
    session->host_workspace_cleanup_pending = 0;
    return runtime_success(err);
}

static int runtime_session_storage_release(yvex_runtime_execution_session *session,
                                           yvex_error *err) {
    if (!session) {
        return runtime_success(err);
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
    return runtime_success(err);
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
            failure, REFUSE_SESSION_OPEN_CLEANUP, 0ull, 1ull,
            (yvex_status)cleanup_rc, err);
    }
    if (err) *err = primary;
    return status;
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
    const yvex_graph_family_api *graph;
    yvex_backend_options backend_options;
    yvex_attention_failure state_failure = {0};
    unsigned long long workspace_bytes = 0ull;
    unsigned long long admitted_host_bytes = 0ull, state_budget;
    int rc, publishable, uploaded = 0;
    if (out)
        *out = NULL;
    if (!out || !model || !request ||
        (request->backend != YVEX_BACKEND_KIND_CPU &&
         request->backend != YVEX_BACKEND_KIND_CUDA))
        return runtime_refuse(failure, REFUSE_SESSION_REQUEST, 1ull, 0ull, err);
    session = (yvex_runtime_execution_session *)calloc(1u, sizeof(*session));
    if (!session)
        return runtime_refuse(failure, REFUSE_SESSION_ALLOCATION,
                              sizeof(*session), 0ull, err);
    session->model = model;
    session->view.model = model;
    if (pthread_mutex_init(&session->lifecycle_mutex, NULL) != 0) {
        rc = runtime_refuse(failure, REFUSE_SESSION_LOCK_INITIALIZATION,
                            1ull, 0ull, err);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    session->lifecycle_mutex_ready = 1;
    if (pthread_cond_init(&session->idle_condition, NULL) != 0) {
        rc = runtime_refuse(failure, REFUSE_SESSION_CONDITION_INITIALIZATION,
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
    graph = model->view.adapter ? model->view.adapter->graph() : NULL;
    if (model->residency) {
        rc = yvex_runtime_residency_snapshot(model->residency, &residency_storage,
                                             NULL, NULL, err);
        if (rc != YVEX_OK) return runtime_session_open_fail(out, session, rc, failure, err);
        residency = &residency_storage;
    }
    rc = yvex_attention_workspace_capacity_resolve(graph, model->attention,
                                                   &workspace_bytes, err);
    if (rc != YVEX_OK ||
        !yvex_core_u64_add(residency ? residency->host_resident_bytes : 0ull,
                           workspace_bytes, &admitted_host_bytes) ||
        (request->maximum_host_bytes &&
         admitted_host_bytes > request->maximum_host_bytes)) {
        rc = runtime_model_reject(
            failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH, "attention-workspace",
            request->maximum_host_bytes, admitted_host_bytes,
            "attention workspace exceeds the admitted session host budget",
            err, rc == YVEX_OK ? YVEX_ERR_BOUNDS : (yvex_status)rc);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    rc = yvex_attention_workspace_open(&session->attention_workspace, workspace_bytes, err);
    if (rc != YVEX_OK) {
        rc = runtime_model_reject(
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
    if (state_factory && (!state_factory->open || !state_factory->discard)) {
        rc = YVEX_ERR_INVALID_ARG;
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.session.state-provider",
                       "attention state factory requires open and discard operations");
    } else if (state_factory) {
        session->attention_state_factory = *state_factory;
        rc = state_factory->open(state_factory->context, graph, model->attention,
                                 state_budget, &session->attention_state_provider,
                                 &state_failure, err);
    } else
        rc = yvex_attention_state_provider_open_persistent(
            graph, model->attention, state_budget, &session->attention_state_provider,
            &state_failure, err);
    if (rc == YVEX_OK &&
        runtime_attention_state_provider_valid(&session->attention_state_provider)) {
        session->attention_state_provider_ready = 1;
        session->view.attention_state_provider = &session->attention_state_provider;
    } else {
        if (state_factory && state_factory->open && state_factory->discard)
            session->attention_state_provider_ready = -1;
        else if (session->attention_state_provider.context &&
                 session->attention_state_provider.release)
            session->attention_state_provider_ready = 1;
        if (rc == YVEX_OK) rc = YVEX_ERR_FORMAT;
        rc = runtime_model_reject(
            failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH, "attention-state", 1ull, 0ull,
            state_failure.reason ? state_failure.reason
                                 : "runtime attention state provider could not open",
            err, (yvex_status)rc);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
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
        runtime_model_failure_record(failure, YVEX_RUNTIME_MODEL_FAILURE_BACKEND,
                                     "backend-open", 1ull, 0ull,
                                     "runtime session backend could not be opened");
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    if (yvex_runtime_workspace_identity_compute(
            model->summary.runtime_model_identity, request->backend,
            request->maximum_host_bytes, request->maximum_device_bytes,
            workspace_bytes, 0ull, NULL, session->summary.workspace_identity,
            err) != YVEX_OK) {
        rc = runtime_refuse(failure, REFUSE_WORKSPACE_IDENTITY, 1ull, 0ull, err);
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
        rc = runtime_refuse(failure, REFUSE_SESSION_RESOURCE_INJECTION,
                            0ull, 1ull, err);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    session->summary.peak_host_bytes = admitted_host_bytes;
    rc = runtime_session_capabilities_bind(session, failure, 0, err);
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
        rc = runtime_refuse(failure, REFUSE_MODEL_DRAINING_PUBLICATION,
                            0ull, 1ull, err);
        return runtime_session_open_fail(out, session, rc, failure, err);
    }
    if (failure)
        memset(failure, 0, sizeof(*failure));
    return runtime_success(err);
}
typedef struct {
    unsigned long long required, host_total, device_total, generation;
} runtime_workspace_requirements;

static int runtime_workspace_state_envelope(
    const yvex_graph_attention_capacity_summary *summary,
    const yvex_attention_state_recipe *layer,
    yvex_attention_state_recipe *envelope, yvex_error *err) {
    unsigned int index;
    if (!summary || !layer || !envelope) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.session.workspace",
                       "capture capacity envelope requires complete recipes");
        return YVEX_ERR_INVALID_ARG;
    }
    *envelope = *layer;
    for (index = 0u; index < envelope->component_count; ++index) {
        yvex_attention_state_component_recipe *component =
            &envelope->components[index];
        unsigned long long maximum;
        if (component->kind != YVEX_ATTENTION_STATE_COMPONENT_HISTORY)
            continue;
        if (component->binding >= YVEX_ATTENTION_STATE_BINDING_COUNT) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.session.workspace",
                           "capture capacity envelope contains an invalid binding");
            return YVEX_ERR_FORMAT;
        }
        maximum = summary->components[component->binding].maximum_capacity;
        if (maximum > component->capacity) component->capacity = maximum;
    }
    envelope->identity[0] = '\0';
    return yvex_attention_state_recipe_seal(envelope, err);
}

static int runtime_session_workspace_requirements(
    const yvex_runtime_execution_session *session, yvex_runtime_execution_mode mode,
    yvex_runtime_execution_scope scope, yvex_attention_evidence_level evidence_level,
    const yvex_graph_attention_capacity_plan *capacity, const yvex_graph_attention_state_summary *state,
    unsigned long long minimum_bytes, runtime_workspace_requirements *requirements,
    yvex_runtime_model_failure *failure, yvex_error *err) {
    static const yvex_attention_execution_mode graph_modes[] = {
        YVEX_ATTENTION_EXECUTION_EAGER, YVEX_ATTENTION_EXECUTION_PIECEWISE, YVEX_ATTENTION_EXECUTION_FULL};
    static const yvex_attention_operation_scope graph_scopes[] = {
        YVEX_ATTENTION_OPERATION_CORE, YVEX_ATTENTION_OPERATION_ENVELOPE, YVEX_ATTENTION_OPERATION_RELEASE_SET};
    const yvex_graph_attention_capacity_summary *summary = yvex_graph_attention_capacity_plan_summary(capacity);
    const yvex_graph_family_api *graph = session->model->adapter->graph();
    const yvex_attention_plan *attention = session->model->attention;
    yvex_attention_execution_mode graph_mode;
    yvex_attention_operation_scope graph_scope;
    unsigned long long count = yvex_attention_plan_layer_count(attention), index;
    memset(requirements, 0, sizeof(*requirements));
    if (!summary || !graph || !graph->workspace_recipe ||
        strcmp(summary->attention_plan_identity,
               yvex_attention_plan_summary(attention)->attention_plan_identity) != 0)
        return runtime_refuse(failure, REFUSE_WORKSPACE_STATE, 1ull, 0ull, err);
    if ((unsigned int)mode >= sizeof(graph_modes) / sizeof(graph_modes[0]) ||
        (unsigned int)scope >= sizeof(graph_scopes) / sizeof(graph_scopes[0]) ||
        evidence_level > YVEX_ATTENTION_EVIDENCE_FULL)
        return runtime_refuse(failure, REFUSE_WORKSPACE_REQUEST, 1ull, 0ull, err);
    graph_mode = graph_modes[mode];
    graph_scope = graph_scopes[scope];
    for (index = 0ull; index < count; ++index) {
        const yvex_attention_layer_plan *layer =
            yvex_attention_plan_layer_at(attention, index);
        const yvex_graph_attention_capacity_layer *capacity_layer =
            yvex_graph_attention_capacity_plan_layer(capacity, index);
        yvex_attention_state_recipe envelope;
        yvex_attention_workspace_recipe recipe;
        yvex_attention_failure graph_failure;
        unsigned long long layer_bytes;
        int rc;
        if (!layer || !capacity_layer) {
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.workspace",
                           "attention layer lookup failed during staging preflight");
            return YVEX_ERR_STATE;
        }
        if (!capacity_layer->selected) continue;
        rc = runtime_workspace_state_envelope(summary, &capacity_layer->recipe, &envelope, err);
        if (rc != YVEX_OK) return rc;
        memset(&recipe, 0, sizeof(recipe));
        memset(&graph_failure, 0, sizeof(graph_failure));
        rc = graph->workspace_recipe(layer, &envelope, graph_mode, graph_scope, evidence_level,
            summary->maximum_token_count, &recipe, &graph_failure, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_attention_workspace_required_from_recipe(&recipe, &layer_bytes, err);
        if (rc != YVEX_OK) return rc;
        if (layer_bytes > requirements->required) requirements->required = layer_bytes;
    }
    if (minimum_bytes > requirements->required) requirements->required = minimum_bytes;
    if (!requirements->required ||
        !yvex_core_u64_add(session->summary.host_resident_bytes, session->summary.workspace_bytes,
                           &requirements->host_total) ||
        !yvex_core_u64_add(requirements->host_total, state->allocated_bytes,
                           &requirements->host_total) ||
        !yvex_core_u64_add(requirements->host_total, requirements->required,
                           &requirements->host_total) ||
        (session->maximum_host_bytes && requirements->host_total > session->maximum_host_bytes))
        return runtime_refuse(failure, REFUSE_WORKSPACE_BUDGET, session->maximum_host_bytes,
                              requirements->host_total, err);
    if (!yvex_core_u64_add(session->summary.device_resident_bytes, requirements->required,
                           &requirements->device_total) ||
        !yvex_core_u64_add(session->summary.workspace_generation, 1ull, &requirements->generation) ||
        (session->maximum_device_bytes && requirements->device_total > session->maximum_device_bytes))
        return runtime_refuse(failure, REFUSE_DEVICE_WORKSPACE_BUDGET, session->maximum_device_bytes,
                              requirements->device_total, err);
    return YVEX_OK;
}

int yvex_runtime_session_prepare_attention_workspace(yvex_runtime_execution_session *session,
    yvex_runtime_execution_mode mode, yvex_runtime_execution_scope scope,
    yvex_attention_evidence_level evidence_level, const yvex_graph_attention_capacity_plan *capacity,
    unsigned long long minimum_bytes, yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_backend_tensor_desc device_descriptor;
    yvex_backend_host_workspace_summary workspace;
    yvex_graph_attention_state_summary state;
    yvex_runtime_session_summary summary_before;
    yvex_runtime_model_failure primary_failure;
    runtime_workspace_requirements requirements;
    yvex_error primary_error;
    char workspace_identity[YVEX_SHA256_HEX_CAP];
    int rc = YVEX_OK;
    if (!session || !yvex_graph_attention_capacity_plan_summary(capacity) ||
        (unsigned int)mode > (unsigned int)YVEX_RUNTIME_MODE_FULL || evidence_level > YVEX_ATTENTION_EVIDENCE_FULL ||
        session->summary.backend != YVEX_BACKEND_KIND_CUDA || !session->backend)
        return runtime_refuse(failure, REFUSE_WORKSPACE_REQUEST, 1ull, 0ull, err);
    if (pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return runtime_refuse(failure, REFUSE_WORKSPACE_LOCK, 1ull, 0ull, err);
    summary_before = session->summary;
    if (!session->summary.open || session->summary.busy || session->closing) {
        rc = runtime_refuse(failure, REFUSE_WORKSPACE_SESSION_STATE,
                            0ull, 1ull, err);
        goto done;
    }
    rc = session->attention_state_provider_ready
             ? session->attention_state_provider.summary(
                   session->attention_state_provider.context, &state, err)
             : YVEX_ERR_STATE;
    if (rc != YVEX_OK) {
        runtime_model_failure_record(failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH,
            "attention-state", 1ull, 0ull, "runtime attention-state capacities could not be read");
        goto done;
    }
    if (!state.sealed || state.transaction_active) {
        rc = runtime_refuse(failure, REFUSE_WORKSPACE_STATE, 0ull,
                            state.transaction_active ? 1ull : 0ull, err);
        goto done;
    }
    if (session->host_workspace_cleanup_pending ||
        (session->workspace && !session->summary.device_workspace_bytes)) {
        rc = runtime_session_workspace_discard(session, err);
        if (rc != YVEX_OK) {
            runtime_model_failure_record(
                failure, YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, "attention-workspace",
                0ull, 1ull, "prior CUDA staging candidate still requires cleanup");
            goto done;
        }
    }
    rc = runtime_session_workspace_requirements(session, mode, scope, evidence_level,
        capacity, &state, minimum_bytes, &requirements, failure, err);
    if (rc != YVEX_OK) goto done;
    rc = yvex_runtime_workspace_identity_compute(
        session->model->summary.runtime_model_identity, session->summary.backend,
        session->maximum_host_bytes, session->maximum_device_bytes,
        session->summary.workspace_bytes, requirements.required,
        yvex_graph_attention_capacity_plan_summary(capacity)->identity, workspace_identity, err);
    if (rc != YVEX_OK) goto done;
    if (session->summary.host_workspace_bytes || session->summary.device_workspace_bytes) {
        if (!session->workspace || !session->summary.host_workspace_owned ||
            !session->summary.host_workspace_pinned ||
            session->summary.host_workspace_bytes != requirements.required ||
            session->summary.device_workspace_bytes != requirements.required ||
            strcmp(session->summary.workspace_identity, workspace_identity) != 0 ||
            !yvex_backend_host_workspace_summary_get(session->backend, &workspace) ||
            !workspace.attached || !workspace.owned || !workspace.pinned ||
            workspace.capacity != requirements.required) {
            rc = runtime_refuse(failure, REFUSE_WORKSPACE_ALREADY_SEALED,
                requirements.required, session->summary.host_workspace_bytes, err);
            goto done;
        }
        rc = runtime_session_capabilities_bind(session, failure, 1, err);
        goto done;
    }
    memset(&device_descriptor, 0, sizeof(device_descriptor));
    device_descriptor.name = "runtime-attention-workspace";
    device_descriptor.dtype = YVEX_DTYPE_I8;
    device_descriptor.rank = 1u;
    device_descriptor.dims[0] = device_descriptor.bytes = requirements.required;
    rc = yvex_backend_tensor_alloc(session->backend, &device_descriptor,
                                   &session->workspace, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_workspace_attach(session->backend, session->workspace,
                                           requirements.generation, err);
    if (rc != YVEX_OK) {
        runtime_model_failure_record(
            failure, YVEX_RUNTIME_MODEL_FAILURE_BACKEND, "device-workspace",
            requirements.required, 0ull, "CUDA device workspace allocation failed");
        goto rollback;
    }
    rc = yvex_backend_host_workspace_prepare_owned(
        session->backend, requirements.required, err);
    if (rc != YVEX_OK) {
        runtime_model_failure_record(
            failure, YVEX_RUNTIME_MODEL_FAILURE_BACKEND, "pinned-host-workspace",
            requirements.required, 0ull, "CUDA pinned workspace allocation failed");
        goto rollback;
    }
    if (!yvex_backend_host_workspace_summary_get(session->backend, &workspace) ||
        !workspace.owned || !workspace.pinned ||
        workspace.capacity != requirements.required) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, rc, "runtime.session.workspace",
                       "prepared CUDA pinned workspace did not match its plan");
        goto rollback;
    }
    session->summary.host_workspace_bytes = workspace.capacity;
    session->summary.host_workspace_peak_bytes = workspace.peak;
    session->summary.host_workspace_owned = workspace.owned;
    session->summary.host_workspace_pinned = workspace.pinned;
    session->summary.device_workspace_bytes = requirements.required;
    session->summary.workspace_generation = requirements.generation;
    session->summary.peak_host_bytes = requirements.host_total;
    session->summary.peak_device_bytes = requirements.device_total;
    yvex_runtime_identity_copy(session->summary.workspace_identity,
                                workspace_identity);
    if (getenv("YVEX_TEST_RUNTIME_WORKSPACE_FAILURE"))
        rc = runtime_refuse(failure, REFUSE_WORKSPACE_CAPABILITY_INJECTION,
                            0ull, 1ull, err);
    else
        rc = runtime_session_capabilities_bind(session, failure, 1, err);
    if (rc == YVEX_OK) goto done;
rollback:
    {
        yvex_error cleanup;
        int cleanup_rc;
        primary_error = err ? *err : (yvex_error){0};
        primary_failure = failure ? *failure : (yvex_runtime_model_failure){0};
        yvex_error_clear(&cleanup);
        cleanup_rc = runtime_session_workspace_discard(session, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
            runtime_model_failure_record(
                failure, YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, "attention-workspace",
                requirements.required, 0ull,
                "failed CUDA workspace candidate could not be released");
        } else {
            if (err) *err = primary_error;
            if (failure) *failure = primary_failure;
        }
        session->summary = summary_before;
    }
done:
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    if (rc == YVEX_OK) {
        if (failure) memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_runtime_session_prepare_persistent_state(yvex_runtime_execution_session *session,
    const yvex_graph_attention_capacity_plan *capacity, yvex_runtime_model_failure *failure,
    yvex_error *err) {
    int rc;
    if (!session || !capacity || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return runtime_refuse(failure, REFUSE_WORKSPACE_LOCK, 1ull, 0ull, err);
    if (!session->summary.open || session->summary.busy || session->closing ||
        session->state_residency || !session->attention_state_provider_ready) {
        rc = runtime_refuse(failure, REFUSE_WORKSPACE_SESSION_STATE, 0ull, 1ull, err);
    } else {
        rc = yvex_runtime_state_residency_prepare(&session->state_residency, session->backend,
            capacity, &session->attention_state_provider, session->summary.peak_host_bytes,
            session->maximum_host_bytes, session->summary.device_resident_bytes,
            session->maximum_device_bytes, err);
        if (rc == YVEX_OK) {
            session->view.state_residency = session->state_residency;
            rc = runtime_session_capabilities_bind(session, failure, 0, err);
        }
        if (rc != YVEX_OK) {
            yvex_error primary = err ? *err : (yvex_error){0}, cleanup;
            int cleanup_rc =
                yvex_runtime_state_residency_close(&session->state_residency, &cleanup);
            session->view.state_residency = session->state_residency;
            if (cleanup_rc != YVEX_OK) {
                rc = cleanup_rc;
                if (err) *err = cleanup;
            } else if (err) *err = primary;
            runtime_model_failure_record(failure, cleanup_rc == YVEX_OK
                ? YVEX_RUNTIME_MODEL_FAILURE_GRAPH : YVEX_RUNTIME_MODEL_FAILURE_CLEANUP,
                "persistent-state-residency", 1ull, 0ull, cleanup_rc == YVEX_OK
                    ? "persistent state preparation failed" : "persistent state cleanup failed");
        }
    }
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}

int yvex_runtime_session_reset_persistent_state(yvex_runtime_execution_session *session,
    yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_attention_failure state_failure;
    int rc;
    if (!session || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return runtime_refuse(failure, REFUSE_WORKSPACE_LOCK, 1ull, 0ull, err);
    if (!session->summary.open || session->summary.busy || session->closing ||
        !session->state_residency || !session->attention_state_provider_ready) {
        rc = runtime_refuse(failure, REFUSE_WORKSPACE_SESSION_STATE, 0ull, 1ull, err);
    } else {
        rc = yvex_runtime_state_residency_reset(session->state_residency, err);
        if (rc == YVEX_OK)
            rc = session->attention_state_provider.reset(
                session->attention_state_provider.context, &state_failure, err);
        if (rc != YVEX_OK) {
            yvex_error cleanup;
            session->summary.invalidated = 1;
            (void)runtime_session_invalidate(session, 1, &cleanup);
            runtime_model_failure_record(failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH,
                "persistent-state-reset", 1ull, 0ull, "persistent attention state reset failed");
        }
    }
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}
/* Acquire exclusive mutable execution ownership for one session operation. */
int yvex_runtime_session_begin(yvex_runtime_execution_session *session,
                               yvex_runtime_model_failure *failure, yvex_error *err) {
    runtime_refusal_id refusal = REFUSE_COUNT;
    int rc;
    if (!session || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return runtime_refuse(failure, REFUSE_SESSION_REQUIRED, 1ull, 0ull, err);
    if (session->summary.invalidated)
        refusal = REFUSE_SESSION_INVALIDATED;
    else if (!session->summary.open || session->closing)
        refusal = REFUSE_SESSION_CLOSING;
    else if (session->summary.busy)
        refusal = REFUSE_SESSION_BUSY;
    else if (session->summary.cancelled)
        refusal = REFUSE_SESSION_CANCELLED;
    if (refusal != REFUSE_COUNT) {
        (void)pthread_mutex_unlock(&session->lifecycle_mutex);
        return runtime_refuse(failure, refusal, 0ull, 1ull, err);
    }
    session->summary.busy = 1;
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    rc = yvex_runtime_model_validate(session->model, failure, err);
    if (rc != YVEX_OK) {
        if (pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
            runtime_model_failure_record(
                failure, YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, "session-busy-release",
                0ull, 1ull, "failed model validation could not release session ownership");
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.begin",
                           "runtime session synchronization could not be reacquired");
            return YVEX_ERR_STATE;
        }
        session->summary.busy = 0;
        (void)pthread_cond_broadcast(&session->idle_condition);
        (void)pthread_mutex_unlock(&session->lifecycle_mutex);
        return rc;
    }
    return runtime_success(err);
}

int yvex_runtime_session_finish(yvex_runtime_execution_session *session, int status,
                                yvex_error *err) {
    yvex_graph_attention_state_summary state;
    const yvex_attention_workspace_summary *workspace;
    yvex_attention_failure state_failure;
    yvex_error cleanup, state_error, primary_error;
    unsigned long long *counter = NULL, counter_next = 0ull;
    int cleanup_rc = YVEX_OK, primary_status = status, state_ready = 0, rc;
    if (!session || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.finish",
                       "busy synchronized runtime session is required");
        return YVEX_ERR_STATE;
    }
    primary_error = err ? *err : (yvex_error){0};
    if (!session->summary.open || !session->summary.busy) {
        (void)pthread_mutex_unlock(&session->lifecycle_mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.finish",
                       "runtime session has no active execution to finish");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(&cleanup);
    yvex_error_clear(&state_error);
    rc = session->attention_state_provider_ready
             ? session->attention_state_provider.summary(
                   session->attention_state_provider.context, &state, &state_error)
             : YVEX_ERR_STATE;
    if (rc == YVEX_OK) state_ready = 1;
    else cleanup_rc = rc, cleanup = state_error;
    workspace = yvex_attention_workspace_summary_get(session->attention_workspace);
    if (workspace) {
        session->summary.workspace_peak_bytes = workspace->peak_bytes;
        session->summary.workspace_allocation_count = workspace->acquisition_count;
        session->summary.workspace_capacity_failure_count = workspace->capacity_failure_count;
    }
    if (state_ready && state.invalidated && cleanup_rc == YVEX_OK) {
        cleanup_rc = YVEX_ERR_STATE;
        yvex_error_set(&cleanup, YVEX_ERR_STATE, "runtime.session.state",
                       "runtime attention state is invalidated");
    }
    if (primary_status == YVEX_OK && cleanup_rc == YVEX_OK &&
        state_ready && state.candidate_active) {
        cleanup_rc = YVEX_ERR_STATE;
        yvex_error_set(&cleanup, YVEX_ERR_STATE, "runtime.session.state",
                       "successful execution left an incomplete layer candidate");
    }
    if (primary_status == YVEX_OK && cleanup_rc == YVEX_OK) {
        counter = &session->summary.execution_count;
        if (getenv("YVEX_TEST_RUNTIME_SESSION_COUNTER_OVERFLOW") ||
            !yvex_core_u64_add(*counter, 1ull, &counter_next)) {
            cleanup_rc = YVEX_ERR_BOUNDS;
            yvex_error_set(&cleanup, YVEX_ERR_BOUNDS, "runtime.session.counter",
                           "runtime session execution counter overflowed");
        }
    }
    if (session->attention_state_provider_ready && ((state_ready && state.transaction_active) ||
        primary_status != YVEX_OK || cleanup_rc != YVEX_OK || session->invalidation_pending)) {
        int committing = primary_status == YVEX_OK && cleanup_rc == YVEX_OK &&
            !session->invalidation_pending;
        yvex_error_clear(&state_error);
        rc = committing && session->state_residency
                 ? yvex_runtime_state_residency_publish(session->state_residency, &state_error)
                 : YVEX_OK;
        if (rc == YVEX_OK)
            rc = committing
                     ? session->attention_state_provider.commit(session->attention_state_provider.context,
                           &state_failure, &state_error)
                     : session->attention_state_provider.abort(session->attention_state_provider.context,
                           &state_failure, &state_error);
        if (rc == YVEX_OK && committing && session->state_residency)
            yvex_runtime_state_residency_commit(session->state_residency);
        else if (!committing && session->state_residency)
            yvex_runtime_state_residency_abort(session->state_residency);
        if (rc != YVEX_OK) {
            cleanup_rc = rc;
            cleanup = state_error;
            yvex_error_clear(&state_error);
            rc = session->attention_state_provider.abort(session->attention_state_provider.context,
                &state_failure, &state_error);
            if (rc != YVEX_OK) {
                cleanup_rc = rc;
                cleanup = state_error;
            }
            if (session->state_residency)
                yvex_runtime_state_residency_abort(session->state_residency);
        }
    }
    if (session->invalidation_pending) {
        rc = runtime_session_invalidate(session, 1, &state_error);
        if (rc == YVEX_OK) session->invalidation_pending = 0;
        else if (cleanup_rc == YVEX_OK) cleanup_rc = rc, cleanup = state_error;
        if (rc == YVEX_OK && primary_status == YVEX_OK && cleanup_rc == YVEX_OK) {
            cleanup_rc = YVEX_ERR_STATE;
            yvex_error_set(&cleanup, YVEX_ERR_STATE, "runtime.session.invalidated",
                           "runtime session was invalidated during execution");
        }
    }
    if (primary_status != YVEX_OK || cleanup_rc != YVEX_OK) {
        counter = (primary_status != YVEX_OK ? primary_status : cleanup_rc) ==
                          YVEX_ERR_CANCELLED
                      ? &session->summary.cancellation_count
                      : &session->summary.failure_count;
        if (!getenv("YVEX_TEST_RUNTIME_SESSION_COUNTER_OVERFLOW") &&
            yvex_core_u64_add(*counter, 1ull, &counter_next))
            *counter = counter_next;
        session->summary.invalidated |=
            cleanup_rc != YVEX_OK && cleanup_rc != YVEX_ERR_CANCELLED;
    } else {
        *counter = counter_next;
    }
    session->summary.busy = 0;
    (void)pthread_cond_broadcast(&session->idle_condition);
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    if (cleanup_rc != YVEX_OK) {
        if (err) *err = cleanup;
        return cleanup_rc;
    }
    if (primary_status != YVEX_OK) {
        if (err && primary_error.code != YVEX_OK)
            *err = primary_error;
        else
            yvex_error_set(err, (yvex_status)primary_status, "runtime.session.execution",
                           "execution failed before session finalization");
        return primary_status;
    }
    return runtime_success(err);
}

int yvex_runtime_session_close(yvex_runtime_execution_session **session_ptr, yvex_error *err) {
    yvex_runtime_execution_session *session;
    yvex_runtime_model *model;
    int rc;
    if (!session_ptr || !*session_ptr) {
        return runtime_success(err);
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
    return runtime_success(err);
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
    if (!lease_ptr || !*lease_ptr) return runtime_success(err);
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
    return runtime_success(err);
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
    return runtime_success(err);
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
        return runtime_refuse(
            failure, REFUSE_CLEANUP_LEASE, 1ull,
            out && !*out && model_request && borrowed_model &&
                    (!session_request || borrowed_session) ? 1ull : 0ull,
            err);
    lease = (yvex_runtime_cleanup_lease *)calloc(1u, sizeof(*lease));
    if (!lease)
        return runtime_refuse(failure, REFUSE_CLEANUP_LEASE_ALLOCATION, 1ull, 0ull, err);
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
        runtime_model_failure_record(
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
        return runtime_refuse(failure, REFUSE_CLEANUP_LEASE_SESSION, 1ull, 0ull, err);
    rc = yvex_runtime_session_open(&lease->session, lease->model, request, failure, err);
    if (rc == YVEX_OK) *borrowed_session = lease->session;
    return rc;
}
