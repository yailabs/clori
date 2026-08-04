/*
 * Mutable target and draft state are staged together and published only after every participant
 * prepares successfully. Workspace growth and reset stay under the session lifecycle lock so a
 * failed candidate cannot become visible to the next turn.
 */
#include "src/runtime/private.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/execution.h>
#include <yvex/internal/graph_state.h>

typedef struct {
    unsigned long long required, host_total, device_total, generation;
} runtime_workspace_requirements;

static int runtime_session_owned_by_current_thread(
    const yvex_runtime_execution_session *session)
{
    return session->execution_owner_ready &&
           pthread_equal(session->execution_owner, pthread_self());
}

/* The backend sees one session-owned resolver even when the session carries
 * distinct target and draft banks. Pointer membership selects the residency;
 * neither scope becomes a second backend or KV authority. */
static int runtime_session_state_resolve(
    const void *context, const void *host, unsigned long long bytes,
    unsigned long long *device_address)
{
    const yvex_runtime_execution_session *session = context;
    int result = YVEX_BACKEND_RESIDENT_MISS;
    if (!session) return YVEX_BACKEND_RESIDENT_INVALID;
    if (session->state_residency)
        result = yvex_runtime_private_state_residency_resolve(
            session->state_residency, host, bytes, device_address);
    if (result != YVEX_BACKEND_RESIDENT_MISS)
        return result;
    if (session->draft_state_residency)
        result = yvex_runtime_private_state_residency_resolve(
            session->draft_state_residency, host, bytes, device_address);
    return result;
}

int yvex_runtime_device_view_bind(
    yvex_execution_device_view *out, yvex_execution_device_value_kind kind,
    yvex_runtime_model *model, yvex_runtime_execution_session *session,
    const yvex_attention_state_provider *provider,
    const yvex_compiled_execution_profile *profile, const yvex_device_tensor *tensor,
    unsigned long long offset, unsigned long long rows, unsigned long long columns,
    yvex_error *err)
{
    const yvex_runtime_model_view *model_view = yvex_runtime_model_view_get(model);
    const yvex_runtime_session_view *session_view =
        yvex_runtime_session_view_get(session);
    yvex_runtime_model_summary model_summary;
    yvex_runtime_session_summary session_summary;
    yvex_runtime_residency_summary residency;
    yvex_graph_attention_state_summary state;
    if (!out || !model_view || !session_view || !provider || !provider->summary ||
        !profile || !tensor ||
        yvex_runtime_model_summary_copy(model, &model_summary, err) != YVEX_OK ||
        yvex_runtime_session_summary_copy(session, &session_summary, err) != YVEX_OK ||
        yvex_runtime_residency_snapshot(model_view->residency, &residency,
                                        NULL, NULL, err) != YVEX_OK ||
        provider->summary(provider->context, &state, err) != YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.execution.device-view",
                       "device value generations are unavailable");
        return YVEX_ERR_STATE;
    }
    memset(out, 0, sizeof(*out));
    out->schema_version = YVEX_EXECUTION_DEVICE_VIEW_SCHEMA_V1;
    out->kind = kind;
    out->backend = session_view->backend;
    out->tensor = tensor;
    out->element_offset = offset;
    out->model_generation = residency.generation;
    out->session_generation = session_summary.workspace_generation;
    out->state_generation = state.generation;
    out->rows = rows;
    out->columns = columns;
    out->element_bytes = sizeof(float);
    out->dtype = YVEX_DTYPE_F32;
    out->synchronization_required = 1;
    out->materialization = YVEX_EXECUTION_MATERIALIZE_NONE;
    yvex_core_text_copy(out->runtime_model_identity,
                        sizeof(out->runtime_model_identity),
                        model_summary.runtime_model_identity);
    yvex_core_text_copy(out->execution_profile_identity,
                        sizeof(out->execution_profile_identity), profile->identity);
    return yvex_execution_device_view_validate(out, err);
}

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
    const yvex_attention_summary *draft_summary =
        yvex_attention_plan_summary(session->model->draft_attention);
    const yvex_attention_plan *attention =
        draft_summary && summary &&
                strcmp(summary->attention_plan_identity,
                       draft_summary->attention_plan_identity) == 0
            ? session->model->draft_attention
            : session->model->attention;
    yvex_attention_execution_mode graph_mode;
    yvex_attention_operation_scope graph_scope;
    unsigned long long count = yvex_attention_plan_layer_count(attention), index;
    memset(requirements, 0, sizeof(*requirements));
    if (!summary || !graph || !graph->workspace_recipe ||
        strcmp(summary->attention_plan_identity,
               yvex_attention_plan_summary(attention)->attention_plan_identity) != 0)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_STATE, 1ull, 0ull, err);
    if ((unsigned int)mode >= sizeof(graph_modes) / sizeof(graph_modes[0]) ||
        (unsigned int)scope >= sizeof(graph_scopes) / sizeof(graph_scopes[0]) ||
        evidence_level > YVEX_ATTENTION_EVIDENCE_FULL)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_REQUEST, 1ull, 0ull, err);
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
    /* The session owns one physical arena for both target and draft plans. Rebinding a
     * smaller logical recipe must not resize or duplicate that arena. */
    if (session->summary.host_workspace_bytes > requirements->required)
        requirements->required = session->summary.host_workspace_bytes;
    if (!requirements->required ||
        !yvex_core_u64_add(session->summary.host_resident_bytes, session->summary.workspace_bytes,
                           &requirements->host_total) ||
        !yvex_core_u64_add(requirements->host_total, state->allocated_bytes,
                           &requirements->host_total) ||
        !yvex_core_u64_add(requirements->host_total, requirements->required,
                           &requirements->host_total) ||
        (session->maximum_host_bytes && requirements->host_total > session->maximum_host_bytes))
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_BUDGET, session->maximum_host_bytes,
                              requirements->host_total, err);
    if (!yvex_core_u64_add(session->summary.device_resident_bytes, requirements->required,
                           &requirements->device_total) ||
        !yvex_core_u64_add(session->summary.workspace_generation, 1ull, &requirements->generation) ||
        (session->maximum_device_bytes && requirements->device_total > session->maximum_device_bytes))
        return yvex_runtime_private_refuse(
            failure, YVEX_RUNTIME_REFUSE_DEVICE_WORKSPACE_BUDGET,
            session->maximum_device_bytes, requirements->device_total, err);
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
    const yvex_graph_attention_capacity_summary *capacity_summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    const yvex_attention_summary *draft_summary = session && session->model
        ? yvex_attention_plan_summary(session->model->draft_attention) : NULL;
    yvex_attention_state_provider *state_provider;
    int rc = YVEX_OK;
    if (!session || !yvex_graph_attention_capacity_plan_summary(capacity) ||
        (unsigned int)mode > (unsigned int)YVEX_RUNTIME_MODE_FULL || evidence_level > YVEX_ATTENTION_EVIDENCE_FULL ||
        session->summary.backend != YVEX_BACKEND_KIND_CUDA || !session->backend)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_REQUEST, 1ull, 0ull, err);
    if (pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_LOCK, 1ull, 0ull, err);
    summary_before = session->summary;
    if (!session->summary.open ||
        (session->summary.busy &&
         !runtime_session_owned_by_current_thread(session)) ||
        session->closing) {
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_SESSION_STATE,
                            0ull, 1ull, err);
        goto done;
    }
    state_provider = draft_summary && capacity_summary &&
                             strcmp(capacity_summary->attention_plan_identity,
                                    draft_summary->attention_plan_identity) == 0
                         ? &session->draft_attention_state_provider
                         : &session->attention_state_provider;
    rc = state_provider->context && state_provider->summary
             ? state_provider->summary(state_provider->context, &state, err)
             : YVEX_ERR_STATE;
    if (rc != YVEX_OK) {
        yvex_runtime_private_failure_record(failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH,
            "attention-state", 1ull, 0ull, "runtime attention-state capacities could not be read");
        goto done;
    }
    if (!state.sealed || state.transaction_active) {
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_STATE, 0ull,
                            state.transaction_active ? 1ull : 0ull, err);
        goto done;
    }
    if (session->host_workspace_cleanup_pending ||
        (session->workspace && !session->summary.device_workspace_bytes)) {
        rc = yvex_runtime_private_session_workspace_discard(session, err);
        if (rc != YVEX_OK) {
            yvex_runtime_private_failure_record(
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
            !yvex_backend_host_workspace_summary_get(session->backend, &workspace) ||
            !workspace.attached || !workspace.owned || !workspace.pinned ||
            workspace.capacity != requirements.required) {
            rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_ALREADY_SEALED,
                requirements.required, session->summary.host_workspace_bytes, err);
            goto done;
        }
        if (strcmp(session->summary.workspace_identity, workspace_identity) != 0) {
            unsigned long long prior_generation = session->summary.workspace_generation;
            yvex_backend_workspace_detach(session->backend);
            rc = yvex_backend_workspace_attach(session->backend, session->workspace,
                                               requirements.generation, err);
            if (rc != YVEX_OK) {
                yvex_error attach_error = err ? *err : (yvex_error){0};
                yvex_error rollback_error = {0};
                (void)yvex_backend_workspace_attach(session->backend,
                                                    session->workspace,
                                                    prior_generation,
                                                    &rollback_error);
                if (err) *err = attach_error;
                goto done;
            }
            session->summary.workspace_generation = requirements.generation;
            yvex_runtime_identity_copy(session->summary.workspace_identity,
                                       workspace_identity);
        }
        rc = yvex_runtime_private_session_capabilities_bind(session, failure, 1, err);
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
        yvex_runtime_private_failure_record(
            failure, YVEX_RUNTIME_MODEL_FAILURE_BACKEND, "device-workspace",
            requirements.required, 0ull, "CUDA device workspace allocation failed");
        goto rollback;
    }
    rc = yvex_backend_host_workspace_prepare_owned(
        session->backend, requirements.required, err);
    if (rc != YVEX_OK) {
        yvex_runtime_private_failure_record(
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
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_CAPABILITY_INJECTION,
                            0ull, 1ull, err);
    else
        rc = yvex_runtime_private_session_capabilities_bind(session, failure, 1, err);
    if (rc == YVEX_OK) goto done;
rollback:
    {
        yvex_error cleanup;
        int cleanup_rc;
        primary_error = err ? *err : (yvex_error){0};
        primary_failure = failure ? *failure : (yvex_runtime_model_failure){0};
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_runtime_private_session_workspace_discard(session, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
            yvex_runtime_private_failure_record(
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

int yvex_runtime_session_prepare_persistent_scope_state(
    yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_runtime_model_failure *failure, yvex_error *err)
{
    yvex_attention_state_provider *provider;
    yvex_runtime_state_residency **residency;
    yvex_runtime_state_residency *other;
    yvex_runtime_state_residency_summary other_summary = {0};
    unsigned long long prior_host, prior_device;
    int ready, rc;
    if (!session || !capacity ||
        (scope != YVEX_TENSOR_SCOPE_GLOBAL && scope != YVEX_TENSOR_SCOPE_DRAFT) ||
        !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_LOCK, 1ull, 0ull, err);
    provider = scope == YVEX_TENSOR_SCOPE_DRAFT
                   ? &session->draft_attention_state_provider
                   : &session->attention_state_provider;
    residency = scope == YVEX_TENSOR_SCOPE_DRAFT
                    ? &session->draft_state_residency
                    : &session->state_residency;
    other = scope == YVEX_TENSOR_SCOPE_DRAFT
                ? session->state_residency
                : session->draft_state_residency;
    ready = scope == YVEX_TENSOR_SCOPE_DRAFT
                ? session->draft_attention_state_provider_ready
                : session->attention_state_provider_ready;
    if (!session->summary.open ||
        (session->summary.busy &&
         !runtime_session_owned_by_current_thread(session)) ||
        session->closing ||
        *residency || !ready) {
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_SESSION_STATE, 0ull, 1ull, err);
    } else {
        rc = other ? yvex_runtime_state_residency_summary_copy(
                         other, &other_summary, err)
                   : YVEX_OK;
        prior_host = session->summary.peak_host_bytes;
        if (rc == YVEX_OK &&
            (!yvex_core_u64_add(session->summary.device_resident_bytes,
                                session->summary.device_workspace_bytes,
                                &prior_device) ||
             !yvex_core_u64_add(prior_host, other_summary.host_bytes,
                                &prior_host) ||
             !yvex_core_u64_add(prior_device, other_summary.device_bytes,
                                &prior_device)))
            rc = yvex_runtime_private_reject(
                failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH,
                "persistent-state-budget", ULLONG_MAX, 0ull,
                "combined target and draft state budget overflowed", err,
                YVEX_ERR_BOUNDS);
        if (rc == YVEX_OK)
            rc = yvex_runtime_state_residency_prepare(
                residency, session->backend, capacity, provider, prior_host,
                session->maximum_host_bytes, prior_device,
                session->maximum_device_bytes, err);
        if (rc == YVEX_OK &&
            session->summary.backend == YVEX_BACKEND_KIND_CUDA &&
            !session->state_resolver_attached) {
            rc = yvex_backend_state_residency_attach(
                session->backend, session, runtime_session_state_resolve,
                1ull, err);
            session->state_resolver_attached = rc == YVEX_OK;
        }
        if (rc == YVEX_OK) {
            if (scope == YVEX_TENSOR_SCOPE_DRAFT)
                session->view.draft_state_residency = *residency;
            else
                session->view.state_residency = *residency;
            rc = yvex_runtime_private_session_capabilities_bind(session, failure, 0, err);
        }
        if (rc != YVEX_OK) {
            yvex_error primary = err ? *err : (yvex_error){0}, cleanup;
            int cleanup_rc = yvex_runtime_state_residency_close(residency, &cleanup);
            if (scope == YVEX_TENSOR_SCOPE_DRAFT)
                session->view.draft_state_residency = *residency;
            else
                session->view.state_residency = *residency;
            if (cleanup_rc != YVEX_OK) {
                rc = cleanup_rc;
                if (err) *err = cleanup;
            } else if (err) {
                *err = primary;
            }
            yvex_runtime_private_failure_record(
                failure,
                cleanup_rc == YVEX_OK ? YVEX_RUNTIME_MODEL_FAILURE_GRAPH
                                      : YVEX_RUNTIME_MODEL_FAILURE_CLEANUP,
                "persistent-state-residency", 1ull, 0ull,
                cleanup_rc == YVEX_OK ? "persistent state preparation failed"
                                      : "persistent state cleanup failed");
            if (!session->state_residency &&
                !session->draft_state_residency &&
                session->state_resolver_attached) {
                yvex_backend_state_residency_detach(session->backend);
                session->state_resolver_attached = 0;
            }
        }
    }
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}

int yvex_runtime_session_prepare_persistent_state(
    yvex_runtime_execution_session *session,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_runtime_model_failure *failure, yvex_error *err)
{
    return yvex_runtime_session_prepare_persistent_scope_state(
        session, YVEX_TENSOR_SCOPE_GLOBAL, capacity, failure, err);
}

int yvex_runtime_session_reset_persistent_state(yvex_runtime_execution_session *session,
    yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_attention_failure state_failure;
    int rc;
    if (!session || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_LOCK, 1ull, 0ull, err);
    if (!session->summary.open || session->summary.busy || session->closing ||
        !session->state_residency || !session->attention_state_provider_ready) {
        rc = yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_SESSION_STATE, 0ull, 1ull, err);
    } else {
        rc = yvex_runtime_state_residency_reset(session->state_residency, err);
        if (rc == YVEX_OK)
            rc = session->attention_state_provider.reset(
                session->attention_state_provider.context, &state_failure, err);
        if (rc == YVEX_OK && session->draft_state_residency)
            rc = yvex_runtime_state_residency_reset(
                session->draft_state_residency, err);
        if (rc == YVEX_OK && session->draft_state_residency &&
            session->draft_attention_state_provider_ready)
            rc = session->draft_attention_state_provider.reset(
                session->draft_attention_state_provider.context,
                &state_failure, err);
        if (rc != YVEX_OK) {
            yvex_error cleanup;
            session->summary.invalidated = 1;
            (void)yvex_runtime_private_session_invalidate(session, 1, &cleanup);
            yvex_runtime_private_failure_record(failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH,
                "persistent-state-reset", 1ull, 0ull, "persistent attention state reset failed");
        }
    }
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}
/* Acquire exclusive mutable execution ownership for one session operation. */
int yvex_runtime_session_begin(yvex_runtime_execution_session *session,
                               yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_runtime_private_refusal_id refusal = YVEX_RUNTIME_REFUSE_COUNT;
    int rc;
    if (!session || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_SESSION_REQUIRED, 1ull, 0ull, err);
    if (session->summary.invalidated)
        refusal = YVEX_RUNTIME_REFUSE_SESSION_INVALIDATED;
    else if (!session->summary.open || session->closing)
        refusal = YVEX_RUNTIME_REFUSE_SESSION_CLOSING;
    else if (session->summary.busy)
        refusal = YVEX_RUNTIME_REFUSE_SESSION_BUSY;
    else if (session->summary.cancelled)
        refusal = YVEX_RUNTIME_REFUSE_SESSION_CANCELLED;
    if (refusal != YVEX_RUNTIME_REFUSE_COUNT) {
        (void)pthread_mutex_unlock(&session->lifecycle_mutex);
        return yvex_runtime_private_refuse(failure, refusal, 0ull, 1ull, err);
    }
    session->summary.busy = 1;
    session->execution_owner = pthread_self();
    session->execution_owner_ready = 1;
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    rc = yvex_runtime_model_validate(session->model, failure, err);
    if (rc != YVEX_OK) {
        if (pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
            yvex_runtime_private_failure_record(
                failure, YVEX_RUNTIME_MODEL_FAILURE_CLEANUP, "session-busy-release",
                0ull, 1ull, "failed model validation could not release session ownership");
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.begin",
                           "runtime session synchronization could not be reacquired");
            return YVEX_ERR_STATE;
        }
        session->execution_owner_ready = 0;
        session->summary.busy = 0;
        (void)pthread_cond_broadcast(&session->idle_condition);
        (void)pthread_mutex_unlock(&session->lifecycle_mutex);
        return rc;
    }
    return yvex_runtime_private_success(err);
}

int yvex_runtime_session_select_attention_prefix(
    yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    unsigned long long prefix_count, unsigned long long extension_count,
    yvex_runtime_state_promotion_facts *facts, yvex_error *err)
{
    yvex_runtime_state_promotion_facts candidate = {0};
    yvex_attention_state_provider *provider;
    yvex_runtime_state_residency *residency;
    yvex_runtime_state_residency_summary before = {0}, after = {0};
    yvex_graph_attention_state_summary summary;
    yvex_attention_failure failure = {0};
    unsigned long long layer;
    int ready, rc;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!session ||
        (scope != YVEX_TENSOR_SCOPE_GLOBAL &&
         scope != YVEX_TENSOR_SCOPE_DRAFT) ||
        !facts ||
        !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.prefix",
                       "a synchronized runtime session and valid scope are required");
        return YVEX_ERR_STATE;
    }
    provider = scope == YVEX_TENSOR_SCOPE_DRAFT
                   ? &session->draft_attention_state_provider
                   : &session->attention_state_provider;
    residency = scope == YVEX_TENSOR_SCOPE_DRAFT
                    ? session->draft_state_residency
                    : session->state_residency;
    ready = scope == YVEX_TENSOR_SCOPE_DRAFT
                ? session->draft_attention_state_provider_ready
                : session->attention_state_provider_ready;
    if (!session->summary.open || !session->summary.busy ||
        !runtime_session_owned_by_current_thread(session) || session->closing ||
        !ready || !provider->select_prefix || !provider->summary ||
        !provider->view) {
        (void)pthread_mutex_unlock(&session->lifecycle_mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.prefix",
                       "the active session has no prefix-addressable state owner");
        return YVEX_ERR_STATE;
    }
    if (residency)
        rc = yvex_runtime_state_residency_summary_copy(residency, &before, err);
    else
        rc = YVEX_OK;
    if (rc == YVEX_OK)
        rc = provider->select_prefix(provider->context, prefix_count,
                                     extension_count, &failure, err);
    if (rc == YVEX_OK)
        rc = provider->summary(provider->context, &summary, err);
    for (layer = 0ull; rc == YVEX_OK && residency &&
                         layer < summary.layer_count; ++layer) {
        const yvex_attention_history_view *view = provider->view(
            provider->context, layer, YVEX_ATTENTION_STATE_VIEW_CANDIDATE);
        if (view)
            rc = yvex_runtime_state_residency_stage(
                residency, provider, layer, err);
    }
    if (rc == YVEX_OK && residency)
        rc = yvex_runtime_state_residency_summary_copy(residency, &after, err);
    if (rc == YVEX_OK &&
        (after.upload_bytes < before.upload_bytes ||
         after.upload_count < before.upload_count)) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.prefix",
                       "persistent state promotion counters regressed");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        candidate.schema_version = YVEX_RUNTIME_STATE_PROMOTION_FACTS_SCHEMA_V1;
        candidate.available = 1;
        candidate.h2d_bytes = after.upload_bytes - before.upload_bytes;
        candidate.synchronization_count = after.cuda_ready
                                                ? after.upload_count - before.upload_count
                                                : 0ull;
        rc = yvex_execution_memory_facts_add(
            &candidate.memory, 0ull, candidate.h2d_bytes, 0ull, 0ull, 1ull, 0ull, err);
        if (rc == YVEX_OK) *facts = candidate;
    }
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}

int yvex_runtime_session_finish_scope(
    yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    yvex_attention_transaction_disposition disposition, int status,
    yvex_error *err) {
    yvex_graph_attention_state_summary state;
    const yvex_attention_workspace_summary *workspace;
    yvex_attention_state_provider *provider;
    yvex_runtime_state_residency *residency;
    yvex_attention_failure state_failure;
    yvex_error cleanup, state_error, primary_error;
    unsigned long long *counter = NULL, counter_next = 0ull;
    int provider_ready, cleanup_rc = YVEX_OK;
    int primary_status = status, state_ready = 0, rc;
    if (!session ||
        (scope != YVEX_TENSOR_SCOPE_GLOBAL &&
         scope != YVEX_TENSOR_SCOPE_DRAFT) ||
        disposition > YVEX_ATTENTION_TRANSACTION_ABORT ||
        !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.finish",
                       "busy synchronized runtime session is required");
        return YVEX_ERR_STATE;
    }
    provider = scope == YVEX_TENSOR_SCOPE_DRAFT
                   ? &session->draft_attention_state_provider
                   : &session->attention_state_provider;
    residency = scope == YVEX_TENSOR_SCOPE_DRAFT
                    ? session->draft_state_residency
                    : session->state_residency;
    provider_ready = scope == YVEX_TENSOR_SCOPE_DRAFT
                         ? session->draft_attention_state_provider_ready
                         : session->attention_state_provider_ready;
    primary_error = err ? *err : (yvex_error){0};
    if (!session->summary.open || !session->summary.busy ||
        !runtime_session_owned_by_current_thread(session)) {
        (void)pthread_mutex_unlock(&session->lifecycle_mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.finish",
                       "runtime session has no active execution to finish");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(&cleanup);
    yvex_error_clear(&state_error);
    rc = provider_ready
             ? provider->summary(provider->context, &state, &state_error)
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
    if (provider_ready && ((state_ready && state.transaction_active) ||
        primary_status != YVEX_OK || cleanup_rc != YVEX_OK || session->invalidation_pending)) {
        int committing = primary_status == YVEX_OK && cleanup_rc == YVEX_OK &&
            !session->invalidation_pending &&
            disposition == YVEX_ATTENTION_TRANSACTION_COMMIT;
        yvex_error_clear(&state_error);
        rc = committing && residency
                 ? yvex_runtime_state_residency_publish(residency, &state_error)
                 : YVEX_OK;
        if (rc == YVEX_OK)
            rc = committing
                     ? provider->commit(provider->context, &state_failure,
                                        &state_error)
                     : provider->abort(provider->context, &state_failure,
                                       &state_error);
        if (rc == YVEX_OK && committing && residency)
            yvex_runtime_state_residency_commit(residency);
        else if (!committing && residency)
            yvex_runtime_state_residency_abort(residency);
        if (rc != YVEX_OK) {
            cleanup_rc = rc;
            cleanup = state_error;
            yvex_error_clear(&state_error);
            rc = provider->abort(provider->context, &state_failure, &state_error);
            if (rc != YVEX_OK) {
                cleanup_rc = rc;
                cleanup = state_error;
            }
            if (residency) yvex_runtime_state_residency_abort(residency);
        }
    }
    if (session->invalidation_pending) {
        rc = yvex_runtime_private_session_invalidate(session, 1, &state_error);
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
    session->execution_owner_ready = 0;
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
    return yvex_runtime_private_success(err);
}

int yvex_runtime_session_finish_coordinated(
    yvex_runtime_execution_session *session, int status,
    const yvex_runtime_commit_participant *participant, yvex_error *err)
{
    yvex_attention_state_provider *providers[2];
    yvex_runtime_state_residency *residencies[2];
    yvex_graph_attention_state_summary summaries[2] = {{0}};
    yvex_attention_failure state_failure = {0};
    yvex_error primary = err ? *err : (yvex_error){0}, failure = {0}, cleanup = {0};
    unsigned long long counter_next = 0ull, index;
    int prepared[2] = {0, 0}, participant_prepared = 0;
    int rc = status, cleanup_rc = YVEX_OK;
    if (!session || (participant && (!participant->prepare || !participant->publish ||
                                     !participant->cancel)) ||
        !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.coordinated",
                       "busy session and complete commit participant are required");
        return YVEX_ERR_STATE;
    }
    providers[0] = &session->attention_state_provider;
    providers[1] = &session->draft_attention_state_provider;
    residencies[0] = session->state_residency;
    residencies[1] = session->draft_state_residency;
    if (!session->summary.open || !session->summary.busy ||
        !runtime_session_owned_by_current_thread(session) || session->closing ||
        !session->attention_state_provider_ready ||
        !session->draft_attention_state_provider_ready ||
        session->invalidation_pending) {
        cleanup_rc = YVEX_ERR_STATE;
        yvex_error_set(&cleanup, cleanup_rc, "runtime.session.coordinated",
                       "target and drafter state are not commit-ready");
    }
    for (index = 0ull; cleanup_rc == YVEX_OK && index < 2ull; ++index) {
        cleanup_rc = providers[index]->summary(
            providers[index]->context, &summaries[index], &cleanup);
        if (rc == YVEX_OK && cleanup_rc == YVEX_OK &&
            (!summaries[index].transaction_active || summaries[index].candidate_active ||
             !summaries[index].staged_layer_count || summaries[index].invalidated)) {
            cleanup_rc = YVEX_ERR_STATE;
            yvex_error_set(&cleanup, cleanup_rc, "runtime.session.coordinated",
                           "one coordinated state batch is incomplete");
        }
    }
    if (rc == YVEX_OK && cleanup_rc == YVEX_OK &&
        !yvex_core_u64_add(session->summary.execution_count, 1ull, &counter_next)) {
        cleanup_rc = YVEX_ERR_BOUNDS;
        yvex_error_set(&cleanup, cleanup_rc, "runtime.session.counter",
                       "runtime session execution counter overflowed");
    }

    if (rc == YVEX_OK && cleanup_rc == YVEX_OK && participant) {
        cleanup_rc = participant->prepare(participant->context, &cleanup);
        participant_prepared = cleanup_rc == YVEX_OK;
    }
    for (index = 0ull; rc == YVEX_OK && cleanup_rc == YVEX_OK && index < 2ull;
         ++index)
        if (residencies[index])
            cleanup_rc = yvex_runtime_state_residency_publish(
                residencies[index], &cleanup);
    for (index = 0ull; rc == YVEX_OK && cleanup_rc == YVEX_OK && index < 2ull;
         ++index) {
        cleanup_rc = providers[index]->prepare_commit(
            providers[index]->context, &state_failure, &cleanup);
        prepared[index] = cleanup_rc == YVEX_OK;
    }
    if (rc == YVEX_OK && cleanup_rc == YVEX_OK) {
        for (index = 0ull; index < 2ull; ++index) {
            providers[index]->publish_commit(providers[index]->context);
            if (residencies[index])
                yvex_runtime_state_residency_commit(residencies[index]);
        }
        if (participant_prepared) participant->publish(participant->context);
        session->summary.execution_count = counter_next;
    } else {
        for (index = 2ull; index-- > 0ull;)
            if (prepared[index])
                providers[index]->cancel_commit(providers[index]->context);
        if (participant) participant->cancel(participant->context);
        for (index = 0ull; index < 2ull; ++index) {
            int abort_rc = providers[index]->abort(
                providers[index]->context, &state_failure, &failure);
            if (residencies[index])
                yvex_runtime_state_residency_abort(residencies[index]);
            if (abort_rc != YVEX_OK && cleanup_rc == YVEX_OK) {
                cleanup_rc = abort_rc;
                cleanup = failure;
            }
        }
    }
    if (rc != YVEX_OK || cleanup_rc != YVEX_OK) {
        unsigned long long *counter =
            (rc != YVEX_OK ? rc : cleanup_rc) == YVEX_ERR_CANCELLED
                ? &session->summary.cancellation_count
                : &session->summary.failure_count;
        if (yvex_core_u64_add(*counter, 1ull, &counter_next)) *counter = counter_next;
        if (cleanup_rc != YVEX_OK && cleanup_rc != YVEX_ERR_CANCELLED)
            session->summary.invalidated = 1;
    }
    session->execution_owner_ready = 0;
    session->summary.busy = 0;
    (void)pthread_cond_broadcast(&session->idle_condition);
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    if (cleanup_rc != YVEX_OK) {
        if (err) *err = cleanup;
        return cleanup_rc;
    }
    if (rc != YVEX_OK) {
        if (err && primary.code != YVEX_OK)
            *err = primary;
        else
            yvex_error_set(err, (yvex_status)rc, "runtime.session.execution",
                           "coordinated execution failed before publication");
        return rc;
    }
    return yvex_runtime_private_success(err);
}

int yvex_runtime_session_finish(yvex_runtime_execution_session *session, int status,
                                yvex_error *err)
{
    return yvex_runtime_session_finish_scope(
        session, YVEX_TENSOR_SCOPE_GLOBAL,
        YVEX_ATTENTION_TRANSACTION_COMMIT, status, err);
}
