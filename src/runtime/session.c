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
    unsigned long long required, logical_required, host_total, device_total, generation;
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
    yvex_model_engine *model, yvex_runtime_execution_session *session,
    const yvex_attention_state_provider *provider,
    const yvex_runtime_execution_profile *profile, const yvex_device_tensor *tensor,
    unsigned long long offset, unsigned long long rows, unsigned long long columns,
    yvex_error *err)
{
    const yvex_model_engine_view *model_view = yvex_model_engine_view_get(model);
    const yvex_runtime_session_view *session_view =
        yvex_runtime_session_view_get(session);
    yvex_model_engine_summary model_summary;
    yvex_runtime_session_summary session_summary;
    yvex_runtime_residency_summary residency;
    yvex_graph_attention_state_summary state;
    if (!out || !model_view || !session_view || !provider || !provider->summary ||
        !profile || !tensor ||
        yvex_model_engine_summary_copy(model, &model_summary, err) != YVEX_OK ||
        yvex_runtime_session_summary_copy(session, &session_summary, err) != YVEX_OK ||
        yvex_runtime_residency_snapshot(model_view->residency, &residency,
                                        NULL, NULL, err) != YVEX_OK ||
        provider->summary(provider->context, &state, err) != YVEX_OK ||
        !runtime_execution_profile_matches(profile, &model_summary,
                                           &session_summary)) {
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
    out->resource_generation = residency.generation;
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

int yvex_runtime_private_attention_workspace_required(
    const yvex_attention_summary *summary,
    const yvex_attention_layer_plan *layers, unsigned long long layer_count,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_attention_execution_mode mode,
    yvex_attention_operation_scope scope,
    yvex_attention_evidence_level evidence_level,
    unsigned long long physical_row_capacity, int deferred,
    unsigned long long *required_bytes, yvex_error *err)
{
    const yvex_graph_attention_capacity_summary *capacity_summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    unsigned long long index, staging_bytes = 0ull;
    if (required_bytes) *required_bytes = 0ull;
    if (!summary || !layers || !layer_count || !capacity_summary ||
        !required_bytes || !physical_row_capacity ||
        mode > YVEX_ATTENTION_EXECUTION_FULL ||
        scope > YVEX_ATTENTION_OPERATION_RELEASE_SET ||
        evidence_level > YVEX_ATTENTION_EVIDENCE_FULL ||
        summary->layer_count != layer_count ||
        strcmp(summary->attention_plan_identity,
               capacity_summary->attention_plan_identity) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.session.workspace",
                       "compiled attention staging facts are incomplete");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0ull; index < layer_count; ++index) {
        const yvex_graph_attention_capacity_layer *capacity_layer =
            yvex_graph_attention_capacity_plan_layer(capacity, index);
        yvex_attention_state_recipe envelope;
        yvex_attention_workspace_recipe recipe = {0};
        yvex_attention_failure graph_failure = {0};
        unsigned long long layer_bytes;
        int rc;
        if (!capacity_layer || !capacity_layer->selected) {
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.workspace",
                           "compiled attention staging layer is unavailable");
            return YVEX_ERR_STATE;
        }
        rc = runtime_workspace_state_envelope(
            capacity_summary, &capacity_layer->recipe, &envelope, err);
        if (rc == YVEX_OK)
            rc = yvex_attention_workspace_recipe_build(
                &layers[index], &envelope, mode, scope, evidence_level,
                physical_row_capacity, &recipe, &graph_failure, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_attention_workspace_required_from_recipe(
                &recipe, &layer_bytes, err);
        if (rc != YVEX_OK) return rc;
        if (deferred) {
            if (!yvex_core_u64_add(staging_bytes, layer_bytes,
                                   &staging_bytes)) {
                yvex_error_set(err, YVEX_ERR_BOUNDS,
                               "runtime.session.workspace",
                               "deferred attention staging extent overflowed");
                return YVEX_ERR_BOUNDS;
            }
        } else if (layer_bytes > *required_bytes) {
            *required_bytes = layer_bytes;
        }
    }
    if (deferred)
        return yvex_attention_deferred_workspace_required(
            layer_count, staging_bytes, required_bytes, err);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int runtime_session_workspace_requirements(
    const yvex_runtime_execution_session *session, yvex_runtime_execution_mode mode,
    yvex_runtime_execution_scope scope, yvex_attention_evidence_level evidence_level,
    const yvex_graph_attention_capacity_plan *capacity, const yvex_graph_attention_state_summary *state,
    unsigned long long physical_row_capacity, unsigned long long minimum_bytes,
    runtime_workspace_requirements *requirements,
    yvex_model_engine_failure *failure, yvex_error *err) {
    static const yvex_attention_execution_mode graph_modes[] = {
        YVEX_ATTENTION_EXECUTION_EAGER, YVEX_ATTENTION_EXECUTION_PIECEWISE, YVEX_ATTENTION_EXECUTION_FULL};
    static const yvex_attention_operation_scope graph_scopes[] = {
        YVEX_ATTENTION_OPERATION_CORE, YVEX_ATTENTION_OPERATION_ENVELOPE, YVEX_ATTENTION_OPERATION_RELEASE_SET};
    const yvex_graph_attention_capacity_summary *summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    const yvex_runtime_binding *binding = session->engine->binding;
    const yvex_attention_summary *attention_summary;
    const yvex_attention_layer_plan *layers;
    yvex_attention_execution_mode graph_mode;
    yvex_attention_operation_scope graph_scope;
    unsigned long long count;
    int deferred;
    memset(requirements, 0, sizeof(*requirements));
    if (!summary || !binding)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_STATE, 1ull, 0ull, err);
    if ((unsigned int)mode >= sizeof(graph_modes) / sizeof(graph_modes[0]) ||
        (unsigned int)scope >= sizeof(graph_scopes) / sizeof(graph_scopes[0]) ||
        evidence_level > YVEX_ATTENTION_EVIDENCE_FULL)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_REQUEST, 1ull, 0ull, err);
    graph_mode = graph_modes[mode];
    graph_scope = graph_scopes[scope];
    deferred = yvex_backend_kind_of(session->backend) == YVEX_BACKEND_KIND_CUDA &&
               evidence_level == YVEX_ATTENTION_EVIDENCE_NONE;
    if (binding->summary.draft_layer_count && binding->draft_layers &&
        strcmp(summary->attention_plan_identity,
               binding->draft_attention.attention_plan_identity) == 0) {
        attention_summary = &binding->draft_attention;
        layers = binding->draft_layers;
        count = binding->summary.draft_layer_count;
    } else {
        attention_summary = &binding->attention;
        layers = binding->layers;
        count = binding->summary.layer_count;
    }
    if (yvex_runtime_private_attention_workspace_required(
            attention_summary, layers, count, capacity, graph_mode, graph_scope,
            evidence_level, physical_row_capacity, deferred,
            &requirements->required, err) != YVEX_OK)
        return yvex_error_code(err);
    /* Deferred layer completions retain graph publications until the whole transformer
     * transaction resolves, so the logical arena owns the same summed lifetime envelope. */
    requirements->logical_required = requirements->required;
    if (minimum_bytes > requirements->required) requirements->required = minimum_bytes;
    if (deferred) {
        const yvex_moe_plan_summary *target =
            yvex_moe_plan_summary_get(session->engine->view.moe);
        const yvex_moe_plan_summary *draft =
            yvex_moe_plan_summary_get(session->engine->view.draft_moe);
        unsigned long long moe_layer_count = target ? target->layer_count : 0ull, bytes;
        if (draft && draft->layer_count > moe_layer_count)
            moe_layer_count = draft->layer_count;
        if (moe_layer_count &&
            (!yvex_core_u64_mul(moe_layer_count,
                                sizeof(yvex_moe_device_completion_slot), &bytes) ||
             !yvex_core_u64_add(requirements->required, bytes, &requirements->required)))
            return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_WORKSPACE_BUDGET,
                                                ULLONG_MAX, 0ull, err);
    }
    /* The session owns one physical arena for both target and draft plans. Rebinding a
     * smaller logical recipe must not resize or duplicate that arena. */
    if (session->summary.host_workspace_bytes > requirements->required)
        requirements->required = session->summary.host_workspace_bytes;
    if (session->summary.workspace_bytes > requirements->logical_required)
        requirements->logical_required = session->summary.workspace_bytes;
    if (!requirements->required ||
        !yvex_core_u64_add(session->summary.host_resident_bytes, requirements->logical_required,
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

static void runtime_session_logical_workspace_publish(
    yvex_runtime_execution_session *session, yvex_attention_workspace **candidate,
    unsigned long long capacity)
{
    yvex_attention_workspace *prior = session->attention_workspace;
    session->attention_workspace = *candidate;
    session->view.attention_workspace = *candidate;
    session->summary.workspace_bytes = capacity;
    *candidate = NULL;
    yvex_attention_workspace_close(&prior);
}

int yvex_runtime_session_prepare_attention_workspace(yvex_runtime_execution_session *session,
    yvex_runtime_execution_mode mode, yvex_runtime_execution_scope scope,
    yvex_attention_evidence_level evidence_level, const yvex_graph_attention_capacity_plan *capacity,
    unsigned long long physical_row_capacity, unsigned long long minimum_bytes,
    yvex_model_engine_failure *failure, yvex_error *err) {
    yvex_backend_tensor_desc device_descriptor;
    yvex_attention_workspace *logical_candidate = NULL;
    yvex_backend_host_workspace_summary workspace;
    yvex_graph_attention_state_summary state;
    yvex_runtime_session_summary summary_before;
    yvex_model_engine_failure primary_failure;
    runtime_workspace_requirements requirements;
    yvex_error primary_error;
    char workspace_identity[YVEX_SHA256_HEX_CAP];
    const yvex_graph_attention_capacity_summary *capacity_summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    const yvex_attention_summary *draft_summary = session && session->engine
        ? yvex_attention_plan_summary(session->engine->draft_attention) : NULL;
    yvex_attention_state_provider *state_provider;
    int rc = YVEX_OK;
    if (!session || !yvex_graph_attention_capacity_plan_summary(capacity) ||
        !physical_row_capacity ||
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
        yvex_runtime_private_failure_record(failure, YVEX_MODEL_ENGINE_FAILURE_GRAPH,
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
                failure, YVEX_MODEL_ENGINE_FAILURE_CLEANUP, "attention-workspace",
                0ull, 1ull, "prior CUDA staging candidate still requires cleanup");
            goto done;
        }
    }
    rc = runtime_session_workspace_requirements(
        session, mode, scope, evidence_level, capacity, &state,
        physical_row_capacity, minimum_bytes, &requirements, failure, err);
    if (rc != YVEX_OK) goto done;
    rc = yvex_runtime_workspace_identity_compute(
        session->engine->summary.runtime_model_identity, session->summary.backend,
        session->maximum_host_bytes, session->maximum_device_bytes,
        session->summary.workspace_bytes, requirements.required,
        yvex_graph_attention_capacity_plan_summary(capacity)->identity, workspace_identity, err);
    if (rc != YVEX_OK) goto done;
    if (requirements.logical_required > session->summary.workspace_bytes) {
        rc = yvex_attention_workspace_open(
            &logical_candidate, requirements.logical_required, err);
        if (rc != YVEX_OK) {
            yvex_runtime_private_failure_record(
                failure, YVEX_MODEL_ENGINE_FAILURE_GRAPH, "attention-workspace",
                requirements.logical_required, 0ull,
                "attention logical workspace growth failed");
            goto done;
        }
    }
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
            failure, YVEX_MODEL_ENGINE_FAILURE_BACKEND, "device-workspace",
            requirements.required, 0ull, "CUDA device workspace allocation failed");
        goto rollback;
    }
    rc = yvex_backend_host_workspace_prepare_owned(
        session->backend, requirements.required, err);
    if (rc != YVEX_OK) {
        yvex_runtime_private_failure_record(
            failure, YVEX_MODEL_ENGINE_FAILURE_BACKEND, "pinned-host-workspace",
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
        primary_failure = failure ? *failure : (yvex_model_engine_failure){0};
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_runtime_private_session_workspace_discard(session, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
            yvex_runtime_private_failure_record(
                failure, YVEX_MODEL_ENGINE_FAILURE_CLEANUP, "attention-workspace",
                requirements.required, 0ull,
                "failed CUDA workspace candidate could not be released");
        } else {
            if (err) *err = primary_error;
            if (failure) *failure = primary_failure;
        }
        session->summary = summary_before;
    }
done:
    if (rc == YVEX_OK && logical_candidate)
        runtime_session_logical_workspace_publish(
            session, &logical_candidate, requirements.logical_required);
    yvex_attention_workspace_close(&logical_candidate);
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    if (rc == YVEX_OK) {
        if (failure) memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_runtime_private_session_prepare_persistent_scope_state_locked(
    yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_model_engine_failure *failure, yvex_error *err)
{
    yvex_attention_state_provider *provider;
    yvex_runtime_state_residency **residency;
    yvex_runtime_state_residency *other;
    yvex_runtime_state_residency_summary other_summary = {0};
    unsigned long long prior_host, prior_device;
    int ready, rc;
    if (!session || !capacity ||
        (scope != YVEX_TENSOR_SCOPE_GLOBAL && scope != YVEX_TENSOR_SCOPE_DRAFT))
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
                failure, YVEX_MODEL_ENGINE_FAILURE_GRAPH,
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
                cleanup_rc == YVEX_OK ? YVEX_MODEL_ENGINE_FAILURE_GRAPH
                                      : YVEX_MODEL_ENGINE_FAILURE_CLEANUP,
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
    return rc;
}

int yvex_runtime_session_prepare_persistent_scope_state(
    yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_model_engine_failure *failure, yvex_error *err)
{
    int rc;
    if (!session || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return yvex_runtime_private_refuse(
            failure, YVEX_RUNTIME_REFUSE_WORKSPACE_LOCK, 1ull, 0ull, err);
    rc = yvex_runtime_private_session_prepare_persistent_scope_state_locked(
        session, scope, capacity, failure, err);
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}

int yvex_runtime_session_prepare_persistent_state(
    yvex_runtime_execution_session *session,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_model_engine_failure *failure, yvex_error *err)
{
    return yvex_runtime_session_prepare_persistent_scope_state(
        session, YVEX_TENSOR_SCOPE_GLOBAL, capacity, failure, err);
}

int yvex_runtime_session_configure_persistent_pages(
    yvex_runtime_execution_session *session,
    const yvex_execution_capacity_plan *capacity,
    yvex_model_engine_failure *failure, yvex_error *err)
{
    yvex_attention_failure state_failure;
    int rc, target_configured = 0;
    if (!session || !capacity || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return yvex_runtime_private_refuse(
            failure, YVEX_RUNTIME_REFUSE_WORKSPACE_LOCK, 1ull, 0ull, err);
    if (!session->summary.open || session->summary.busy || session->closing ||
        !session->attention_state_provider_ready ||
        !session->attention_state_provider.configure_pages ||
        (session->draft_attention_state_provider_ready &&
         !session->draft_attention_state_provider.configure_pages)) {
        rc = yvex_runtime_private_refuse(
            failure, YVEX_RUNTIME_REFUSE_WORKSPACE_SESSION_STATE, 0ull, 1ull,
            err);
        goto done;
    }
    memset(&state_failure, 0, sizeof(state_failure));
    rc = session->attention_state_provider.configure_pages(
        session->attention_state_provider.context, capacity, &state_failure,
        err);
    target_configured = rc == YVEX_OK;
    if (rc == YVEX_OK && session->draft_attention_state_provider_ready)
        rc = session->draft_attention_state_provider.configure_pages(
            session->draft_attention_state_provider.context, capacity,
            &state_failure, err);
    if (rc != YVEX_OK) {
        if (target_configured && session->draft_attention_state_provider_ready) {
            yvex_error cleanup;
            session->summary.invalidated = 1;
            (void)yvex_runtime_private_session_invalidate(session, 1, &cleanup);
        }
        yvex_runtime_private_failure_record(
            failure, YVEX_MODEL_ENGINE_FAILURE_GRAPH,
            "persistent-state-pages", 1ull, 0ull,
            "execution capacity could not configure persistent state pages");
    }
done:
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    if (rc == YVEX_OK) {
        if (failure) memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_runtime_session_reset_persistent_state(yvex_runtime_execution_session *session,
    yvex_model_engine_failure *failure, yvex_error *err) {
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
            yvex_runtime_private_failure_record(failure, YVEX_MODEL_ENGINE_FAILURE_GRAPH,
                "persistent-state-reset", 1ull, 0ull, "persistent attention state reset failed");
        }
    }
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}
/* Acquire exclusive mutable execution ownership for one session operation. */
int yvex_runtime_session_begin(yvex_runtime_execution_session *session,
                               yvex_model_engine_failure *failure, yvex_error *err) {
    yvex_runtime_private_refusal_id refusal = YVEX_RUNTIME_REFUSE_COUNT;
    int rc;
    if (!session || !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0)
        return yvex_runtime_private_refuse(failure, YVEX_RUNTIME_REFUSE_SESSION_REQUIRED, 1ull, 0ull, err);
    if (!session->engine || !session->summary.engine_generation ||
        session->summary.engine_generation != session->engine->summary.engine_generation)
        refusal = YVEX_RUNTIME_REFUSE_SESSION_INVALIDATED;
    else if (session->summary.invalidated)
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
    rc = yvex_model_engine_validate(session->engine, failure, err);
    if (rc != YVEX_OK) {
        if (pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
            yvex_runtime_private_failure_record(
                failure, YVEX_MODEL_ENGINE_FAILURE_CLEANUP, "session-busy-release",
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
            rc = yvex_runtime_state_residency_transition(
                residency, provider, NULL, layer, 0ull,
                YVEX_RUNTIME_STATE_STAGE, err);
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
        candidate.physical.h2d_bytes = after.upload_bytes - before.upload_bytes;
        candidate.physical.synchronization_count =
            after.cuda_ready ? after.upload_count - before.upload_count : 0ull;
        rc = yvex_execution_memory_facts_add(
            &candidate.physical.memory, 0ull, candidate.physical.h2d_bytes,
            0ull, 0ull, 1ull, 0ull, err);
        if (rc == YVEX_OK) *facts = candidate;
    }
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    return rc;
}

typedef struct {
    yvex_attention_state_provider *provider;
    yvex_runtime_state_residency *residency;
    int provider_prepared;
} runtime_state_transaction_participant;

static int runtime_state_transaction_prepare(void *opaque, yvex_error *err)
{
    runtime_state_transaction_participant *participant = opaque;
    yvex_attention_failure failure = {0};
    int rc;
    if (!participant || !participant->provider)
        return YVEX_ERR_INVALID_ARG;
    rc = participant->residency
             ? yvex_runtime_state_residency_prepare_commit(
                   participant->residency, err)
             : YVEX_OK;
    if (rc == YVEX_OK)
        rc = participant->provider->prepare_commit(
            participant->provider->context, &failure, err);
    participant->provider_prepared = rc == YVEX_OK;
    return rc;
}

static void runtime_state_transaction_publish(void *opaque)
{
    runtime_state_transaction_participant *participant = opaque;
    if (!participant || !participant->provider_prepared) return;
    if (participant->residency)
        yvex_runtime_state_residency_publish_commit(participant->residency);
    participant->provider->publish_commit(participant->provider->context);
    participant->provider_prepared = 0;
}

static int runtime_state_transaction_abort(void *opaque, yvex_error *err)
{
    runtime_state_transaction_participant *participant = opaque;
    yvex_attention_failure failure = {0};
    yvex_error first = {0}, retry = {0};
    int rc, retry_rc;
    if (!participant || !participant->provider) return YVEX_ERR_INVALID_ARG;
    if (participant->provider_prepared)
        participant->provider->cancel_commit(participant->provider->context);
    participant->provider_prepared = 0;
    if (participant->residency)
        yvex_runtime_state_residency_abort(participant->residency);
    rc = participant->provider->abort(
        participant->provider->context, &failure, err);
    if (rc == YVEX_OK) return YVEX_OK;
    first = err ? *err : (yvex_error){0};
    retry_rc = participant->provider->abort(
        participant->provider->context, &failure, &retry);
    if (retry_rc != YVEX_OK) {
        if (err) *err = retry;
        return retry_rc;
    }
    if (err) *err = first;
    return rc;
}

static int runtime_transaction_resolve(
    const yvex_runtime_transaction_participant *participants,
    unsigned int participant_count, int status, int *abort_failed,
    yvex_error *err)
{
    yvex_error primary = err ? *err : (yvex_error){0};
    yvex_error failure = {0}, abort_error = {0};
    unsigned int index;
    int rc = status, abort_rc = YVEX_OK;
    if (abort_failed) *abort_failed = 0;
    if (participant_count > YVEX_RUNTIME_TRANSACTION_PARTICIPANT_CAP ||
        (participant_count && !participants)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.transaction",
                       "bounded transaction participants are required");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0u; index < participant_count; ++index)
        if (!participants[index].context || !participants[index].prepare ||
            !participants[index].publish || !participants[index].abort) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.transaction",
                           "every transaction participant must be complete");
            return YVEX_ERR_INVALID_ARG;
        }
    for (index = 0u; rc == YVEX_OK && index < participant_count; ++index)
        rc = participants[index].prepare(participants[index].context, err);
    if (rc == YVEX_OK) {
        for (index = 0u; index < participant_count; ++index)
            participants[index].publish(participants[index].context);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    failure = err ? *err : primary;
    for (index = participant_count; index-- > 0u;) {
        yvex_error current = {0};
        int current_rc = participants[index].abort(
            participants[index].context, &current);
        if (current_rc != YVEX_OK && abort_rc == YVEX_OK) {
            abort_rc = current_rc;
            abort_error = current;
        }
    }
    if (abort_rc != YVEX_OK) {
        if (abort_failed) *abort_failed = 1;
        if (err) *err = abort_error;
        return abort_rc;
    }
    if (err) {
        if (failure.code != YVEX_OK)
            *err = failure;
        else if (primary.code != YVEX_OK)
            *err = primary;
        else
            yvex_error_set(err, (yvex_status)rc, "runtime.transaction",
                           "transaction failed before publication");
    }
    return rc;
}

int yvex_runtime_transaction_resolve(
    const yvex_runtime_transaction_participant *participants,
    unsigned int participant_count, int status, yvex_error *err)
{
    return runtime_transaction_resolve(
        participants, participant_count, status, NULL, err);
}

int yvex_runtime_session_finish_scope(
    yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    yvex_attention_transaction_disposition disposition, int status,
    yvex_error *err) {
    yvex_graph_attention_state_summary state;
    const yvex_attention_workspace_summary *workspace;
    yvex_attention_state_provider *provider;
    yvex_runtime_state_residency *residency;
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
        runtime_state_transaction_participant state_participant = {
            .provider = provider, .residency = residency};
        yvex_runtime_transaction_participant transaction = {
            .context = &state_participant,
            .prepare = runtime_state_transaction_prepare,
            .publish = runtime_state_transaction_publish,
            .abort = runtime_state_transaction_abort};
        int committing = primary_status == YVEX_OK && cleanup_rc == YVEX_OK &&
            !session->invalidation_pending &&
            disposition == YVEX_ATTENTION_TRANSACTION_COMMIT;
        yvex_error_clear(&state_error);
        rc = committing
                 ? runtime_transaction_resolve(
                       &transaction, 1u, YVEX_OK, NULL, &state_error)
                 : runtime_state_transaction_abort(
                       &state_participant, &state_error);
        if (rc != YVEX_OK) {
            cleanup_rc = rc;
            cleanup = state_error;
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
    const yvex_runtime_transaction_participant *participants,
    unsigned int participant_count, yvex_error *err)
{
    yvex_attention_state_provider *providers[2];
    yvex_runtime_state_residency *residencies[2];
    int provider_ready[2];
    runtime_state_transaction_participant state_participants[2] = {{0}};
    yvex_runtime_transaction_participant transaction[
        YVEX_RUNTIME_TRANSACTION_PARTICIPANT_CAP] = {{0}};
    yvex_error primary = err ? *err : (yvex_error){0};
    yvex_error cleanup = {0}, transaction_error = {0};
    unsigned long long counter_next = 0ull;
    unsigned int index, state_count = 0u, transaction_count = 0u;
    int abort_failed = 0, boundary_failure, cleanup_rc = YVEX_OK;
    int rc = status, transaction_rc, transaction_status;
    if (!session || participant_count >
                        YVEX_RUNTIME_TRANSACTION_PARTICIPANT_CAP - 2u ||
        (participant_count && !participants) ||
        !session->lifecycle_mutex_ready ||
        pthread_mutex_lock(&session->lifecycle_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.session.coordinated",
                       "busy session and bounded transaction participants are required");
        return YVEX_ERR_STATE;
    }
    providers[0] = &session->attention_state_provider;
    providers[1] = &session->draft_attention_state_provider;
    residencies[0] = session->state_residency;
    residencies[1] = session->draft_state_residency;
    provider_ready[0] = session->attention_state_provider_ready;
    provider_ready[1] = session->draft_attention_state_provider_ready;
    if (!session->summary.open || !session->summary.busy ||
        !runtime_session_owned_by_current_thread(session) || session->closing ||
        session->invalidation_pending) {
        cleanup_rc = YVEX_ERR_STATE;
        yvex_error_set(&cleanup, cleanup_rc, "runtime.session.coordinated",
                       "session state is not commit-ready");
    }
    for (index = 0u; index < 2u; ++index) {
        yvex_graph_attention_state_summary summary = {0};
        int summary_rc;
        if (!provider_ready[index]) continue;
        summary_rc = providers[index]->summary(
            providers[index]->context, &summary, &cleanup);
        if (summary_rc != YVEX_OK && cleanup_rc == YVEX_OK)
            cleanup_rc = summary_rc;
        if (summary_rc == YVEX_OK && summary.transaction_active) {
            state_participants[state_count].provider = providers[index];
            state_participants[state_count].residency = residencies[index];
            transaction[state_count] = (yvex_runtime_transaction_participant){
                .context = &state_participants[state_count],
                .prepare = runtime_state_transaction_prepare,
                .publish = runtime_state_transaction_publish,
                .abort = runtime_state_transaction_abort};
            state_count++;
        }
        if (rc == YVEX_OK && summary_rc == YVEX_OK &&
            (!summary.transaction_active || summary.candidate_active ||
             !summary.staged_layer_count || summary.invalidated) &&
            cleanup_rc == YVEX_OK) {
            cleanup_rc = YVEX_ERR_STATE;
            yvex_error_set(&cleanup, cleanup_rc, "runtime.session.coordinated",
                           "one coordinated state participant is incomplete");
        }
    }
    transaction_count = state_count;
    if (transaction_count + participant_count >
        YVEX_RUNTIME_TRANSACTION_PARTICIPANT_CAP && cleanup_rc == YVEX_OK) {
        cleanup_rc = YVEX_ERR_BOUNDS;
        yvex_error_set(&cleanup, cleanup_rc, "runtime.session.coordinated",
                       "transaction participant capacity was exceeded");
    }
    for (index = 0u; index < participant_count; ++index) {
        if (!participants[index].context || !participants[index].prepare ||
            !participants[index].publish || !participants[index].abort) {
            if (cleanup_rc == YVEX_OK) {
                cleanup_rc = YVEX_ERR_INVALID_ARG;
                yvex_error_set(&cleanup, cleanup_rc,
                               "runtime.session.coordinated",
                               "transaction participant is incomplete");
            }
            continue;
        }
        transaction[transaction_count++] = participants[index];
    }
    if (rc == YVEX_OK && !state_count && cleanup_rc == YVEX_OK) {
        cleanup_rc = YVEX_ERR_STATE;
        yvex_error_set(&cleanup, cleanup_rc, "runtime.session.coordinated",
                       "coordinated execution has no active state participant");
    }
    if (rc == YVEX_OK && cleanup_rc == YVEX_OK &&
        !yvex_core_u64_add(session->summary.execution_count, 1ull, &counter_next)) {
        cleanup_rc = YVEX_ERR_BOUNDS;
        yvex_error_set(&cleanup, cleanup_rc, "runtime.session.counter",
                       "runtime session execution counter overflowed");
    }

    transaction_status = rc != YVEX_OK ? rc : cleanup_rc;
    transaction_error = transaction_status == rc ? primary : cleanup;
    transaction_rc = runtime_transaction_resolve(
        transaction, transaction_count, transaction_status, &abort_failed,
        &transaction_error);
    boundary_failure = cleanup_rc != YVEX_OK || abort_failed ||
                       (rc == YVEX_OK && transaction_rc != YVEX_OK);
    if (transaction_rc == YVEX_OK)
        session->summary.execution_count = counter_next;
    else {
        unsigned long long *counter =
            transaction_rc == YVEX_ERR_CANCELLED
                ? &session->summary.cancellation_count
                : &session->summary.failure_count;
        if (yvex_core_u64_add(*counter, 1ull, &counter_next)) *counter = counter_next;
        if (boundary_failure && transaction_rc != YVEX_ERR_CANCELLED)
            session->summary.invalidated = 1;
    }
    session->execution_owner_ready = 0;
    session->summary.busy = 0;
    (void)pthread_cond_broadcast(&session->idle_condition);
    (void)pthread_mutex_unlock(&session->lifecycle_mutex);
    if (transaction_rc == YVEX_OK)
        return yvex_runtime_private_success(err);
    if (err) *err = transaction_error;
    return transaction_rc;
}

int yvex_runtime_session_finish(yvex_runtime_execution_session *session, int status,
                                yvex_error *err)
{
    return yvex_runtime_session_finish_scope(
        session, YVEX_TENSOR_SCOPE_GLOBAL,
        YVEX_ATTENTION_TRANSACTION_COMMIT, status, err);
}
