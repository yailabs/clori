/* Share checked component buffers and bindings without importing family policy. */
#include <yvex/internal/component.h>

#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/convolution.h>
#include <yvex/internal/joint_transformer.h>
#include <yvex/internal/multimodal.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/transformer.h>

struct yvex_runtime_component_session {
    yvex_materialization_plan *plan;
    yvex_materialization_session *materialization;
    yvex_runtime_residency *residency;
    yvex_backend *backend;
    yvex_device_tensor *workspace;
    unsigned long long workspace_bytes, execution_transaction_leases;
    yvex_engine_resource_catalog *execution_resources;
    yvex_engine_resource_handle execution_arena_resource;
    yvex_engine_resource_handle request_prepared_resource;
    yvex_engine_resource_handle condition_prepared_resource;
    yvex_transformer_joint_prepared *joint_prepared;
    yvex_transformer_joint_prepared_summary joint_prepared_summary;
    yvex_component_resource_summary resource_summary;
    char package_identity[YVEX_SHA256_HEX_CAP];
    char admission_identity[YVEX_SHA256_HEX_CAP];
    char active_prepared_identity[YVEX_SHA256_HEX_CAP];
    int prepared_resources_borrowed;
    yvex_runtime_residency_summary summary;
};

static const yvex_materialized_tensor_binding *component_binding_find(
    const yvex_materialization_session *, const char *);
static int component_weight_bind(
    const yvex_materialization_session *, const yvex_runtime_residency *,
    const char *, yvex_component_encoded_weight *, yvex_error *);

static int component_joint_prepared_release(void *context, yvex_error *err)
{
    yvex_runtime_component_session *session = context;
    const yvex_backend_component_operations *operations;
    int rc;
    if (!session || !session->backend || !session->joint_prepared) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-resource",
                       "prepared component resource ownership is unavailable");
        return YVEX_ERR_STATE;
    }
    operations = yvex_backend_component_operations_get(session->backend);
    if (!operations || !operations->joint_transformer_prepared_release) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "runtime.component-resource",
                       "backend cannot release its prepared component resource");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = operations->joint_transformer_prepared_release(
        session->backend, &session->joint_prepared, err);
    if (rc == YVEX_OK) {
        memset(&session->joint_prepared_summary, 0,
               sizeof(session->joint_prepared_summary));
        session->resource_summary.ready = 0;
        session->resource_summary.request_ready = 0;
        session->resource_summary.condition_ready = 0;
        session->resource_summary.host_arena_bytes = 0ull;
        session->resource_summary.device_arena_bytes = 0ull;
        session->resource_summary.request_prepared_bytes = 0ull;
        session->resource_summary.condition_prepared_bytes = 0ull;
        session->resource_summary.allocation_count = 0ull;
        session->resource_summary.last_execution_allocation_events = 0ull;
        session->resource_summary.resource_count = 0ull;
        memset(session->resource_summary.prepared_identity, 0,
               sizeof(session->resource_summary.prepared_identity));
    }
    return rc;
}

int yvex_runtime_component_session_close(yvex_runtime_component_session **session,
                                         yvex_error *err)
{
    yvex_runtime_component_session *owned;
    yvex_error cleanup;
    int rc = YVEX_OK, cleanup_rc;
    if (!session || !*session) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owned = *session;
    if (owned->execution_transaction_leases) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-session",
                       "an active execution transaction still retains component resources");
        return YVEX_ERR_STATE;
    }
    if (owned->prepared_resources_borrowed) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-session",
                       "prepared execution resources remain borrowed");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_resource_catalog_close(
        &owned->execution_resources, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        if (err) *err = cleanup;
        return cleanup_rc;
    }
    *session = NULL;
    if (owned->workspace) {
        yvex_backend_workspace_detach(owned->backend);
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(owned->backend, &owned->workspace, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_backend_close_checked(&owned->backend, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_residency_close(&owned->residency, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    yvex_materialization_session_close(owned->materialization);
    yvex_materialization_plan_close(owned->plan);
    free(owned);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

static int component_session_prepare_workspace(
    yvex_runtime_component_session *session, unsigned long long bytes, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *workspace = NULL;
    yvex_error primary, cleanup;
    int rc, cleanup_rc;
    if (!session || !session->backend || !bytes || bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component-session.workspace",
                       "one bounded backend component workspace is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (session->workspace) {
        if (session->workspace_bytes >= bytes) {
            yvex_error_clear(err);
            return YVEX_OK;
        }
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-session.workspace",
                       "component workspace geometry is already sealed");
        return YVEX_ERR_STATE;
    }
    descriptor.name = "runtime-component-workspace";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = bytes;
    rc = yvex_backend_tensor_alloc(session->backend, &descriptor, &workspace, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_workspace_attach(session->backend, workspace, 1ull, err);
    if (rc != YVEX_OK) {
        primary = err ? *err : (yvex_error){0};
        yvex_error_clear(&cleanup);
        cleanup_rc = workspace
                         ? yvex_backend_tensor_release(session->backend, &workspace, &cleanup)
                         : YVEX_OK;
        if (cleanup_rc != YVEX_OK) {
            if (err) *err = cleanup;
            return cleanup_rc;
        }
        if (err) *err = primary;
        return rc;
    }
    session->workspace = workspace;
    session->workspace_bytes = bytes;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_component_session_open(
    yvex_runtime_component_session **out, const yvex_complete_artifact_admission *admission,
    const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    yvex_backend_kind backend_kind, unsigned long long maximum_host_bytes,
    unsigned long long maximum_device_bytes, yvex_error *err)
{
    yvex_runtime_component_session *session = NULL;
    yvex_backend_options backend_options = {0};
    yvex_materialization_options options;
    yvex_materialization_failure materialization_failure;
    yvex_runtime_residency_options residency_options = {0};
    yvex_runtime_residency_failure residency_failure;
    yvex_error primary, cleanup;
    int uploaded = 0, rc, cleanup_rc;
    if (out) *out = NULL;
    if (!out || !admission || !artifact || !gguf || !tensors ||
        (backend_kind != YVEX_BACKEND_KIND_CPU && backend_kind != YVEX_BACKEND_KIND_CUDA)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component-session",
                       "admitted component inputs and CPU or CUDA placement are required");
        return YVEX_ERR_INVALID_ARG;
    }
    session = (yvex_runtime_component_session *)calloc(1u, sizeof(*session));
    if (!session) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "runtime.component-session",
                       "component execution session allocation failed");
        return YVEX_ERR_NOMEM;
    }
    yvex_core_text_copy(session->package_identity,
                        sizeof(session->package_identity),
                        admission->artifact_identity);
    yvex_core_text_copy(session->admission_identity,
                        sizeof(session->admission_identity),
                        admission->admission_identity);
    yvex_materialization_options_default(&options);
    options.max_chunk_bytes = 64ull * 1024ull * 1024ull;
    if (maximum_host_bytes && maximum_host_bytes < options.max_chunk_bytes)
        options.max_chunk_bytes = (size_t)maximum_host_bytes;
    rc = yvex_materialization_plan_build(&session->plan, admission, artifact, gguf, tensors,
                                         NULL, &options, &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(&session->materialization, session->plan,
                                                artifact, &options, &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(session->materialization,
                                                  &materialization_failure, err);
    /* Context creation needs independent system headroom, so establish it before the complete
     * component payload is faulted into the locked residency arena. */
    if (rc == YVEX_OK && backend_kind == YVEX_BACKEND_KIND_CUDA) {
        backend_options.kind = YVEX_BACKEND_KIND_CUDA;
        backend_options.memory_limit_bytes = maximum_device_bytes;
        rc = yvex_backend_open(&session->backend, &backend_options, err);
    }
    residency_options.maximum_host_bytes = maximum_host_bytes;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_residency_prepare(
            &session->residency, session->materialization, admission->logical_component_identity,
            &residency_options, &residency_failure, err);
    if (rc == YVEX_OK && backend_kind == YVEX_BACKEND_KIND_CUDA)
        rc = yvex_runtime_residency_cuda_session_attach(
            session->residency, &session->backend, maximum_device_bytes, &uploaded,
            &session->summary, err);
    if (rc == YVEX_OK && backend_kind == YVEX_BACKEND_KIND_CPU)
        rc = yvex_runtime_residency_snapshot(session->residency, &session->summary,
                                             NULL, NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_resource_catalog_open(
            &session->execution_resources,
            session->summary.generation ? session->summary.generation : 1ull,
            session->summary.residency_identity, 4ull, err);
    if (rc == YVEX_OK)
        session->resource_summary.schema_version =
            YVEX_COMPONENT_RESOURCE_SUMMARY_SCHEMA_V1;
    if (rc != YVEX_OK) {
        primary = err ? *err : (yvex_error){0};
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            if (err) *err = cleanup;
            return cleanup_rc;
        }
        if (err) *err = primary;
        return rc;
    }
    *out = session;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int component_session_weight_view(
    void *context, const char *name, yvex_component_encoded_weight *weight, yvex_error *err)
{
    yvex_runtime_component_session *session = context;
    if (!session || !name || !weight || !session->materialization ||
        !session->residency || !session->summary.sealed || session->summary.invalidated) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.weight-view",
                       "one sealed component execution and weight name are required");
        return YVEX_ERR_INVALID_ARG;
    }
    return component_weight_bind(session->materialization, session->residency,
                                 name, weight, err);
}

static int component_session_workspace_reserve(
    void *context, unsigned long long bytes, yvex_error *err)
{
    return component_session_prepare_workspace(context, bytes, err);
}

static int component_resource_handle_present(yvex_engine_resource_handle handle)
{
    return handle.engine_generation && handle.slot && handle.generation;
}

static int component_prepared_resources_borrow(
    yvex_runtime_component_session *session, yvex_error *err)
{
    yvex_engine_resource_handle handles[3];
    unsigned int count = 0u, acquired = 0u;
    void *value = NULL;
    int rc = YVEX_OK;
    if (!session || !session->execution_resources || !session->joint_prepared)
        return YVEX_ERR_STATE;
    handles[count++] = session->execution_arena_resource;
    if (component_resource_handle_present(session->request_prepared_resource))
        handles[count++] = session->request_prepared_resource;
    if (component_resource_handle_present(session->condition_prepared_resource))
        handles[count++] = session->condition_prepared_resource;
    while (acquired < count && rc == YVEX_OK) {
        value = NULL;
        rc = yvex_runtime_resource_acquire(
            session->execution_resources, handles[acquired], &value, err);
        if (rc == YVEX_OK) {
            acquired++;
            if (value != session->joint_prepared) {
                yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-resource",
                               "prepared resource value disagrees with its owner");
                rc = YVEX_ERR_STATE;
            }
        }
    }
    while (rc != YVEX_OK && acquired) {
        yvex_error cleanup;
        --acquired;
        yvex_error_clear(&cleanup);
        (void)yvex_runtime_resource_drop(
            session->execution_resources, handles[acquired], &cleanup);
    }
    if (rc == YVEX_OK) {
        session->prepared_resources_borrowed = 1;
        session->resource_summary.retained_by_transaction =
            session->execution_transaction_leases != 0ull;
        yvex_error_clear(err);
    }
    return rc;
}

static int component_prepared_resources_drop(
    yvex_runtime_component_session *session, yvex_error *err)
{
    yvex_engine_resource_handle handles[3];
    unsigned int count = 0u;
    int rc = YVEX_OK;
    if (!session || !session->execution_resources ||
        !session->prepared_resources_borrowed)
        return YVEX_OK;
    handles[count++] = session->execution_arena_resource;
    if (component_resource_handle_present(session->request_prepared_resource))
        handles[count++] = session->request_prepared_resource;
    if (component_resource_handle_present(session->condition_prepared_resource))
        handles[count++] = session->condition_prepared_resource;
    while (count && rc == YVEX_OK) {
        --count;
        rc = yvex_runtime_resource_drop(
            session->execution_resources, handles[count], err);
    }
    if (rc == YVEX_OK) {
        session->prepared_resources_borrowed = 0;
        session->resource_summary.retained_by_transaction = 0;
        yvex_error_clear(err);
    }
    return rc;
}

static int component_session_transaction_retain(void *context, yvex_error *err)
{
    yvex_runtime_component_session *session = context;
    if (!session || !session->materialization || !session->residency ||
        !session->summary.sealed || session->summary.invalidated ||
        session->execution_transaction_leases == ULLONG_MAX) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-session.transaction",
                       "sealed component resources are required for execution retention");
        return YVEX_ERR_STATE;
    }
    session->execution_transaction_leases++;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int component_session_transaction_release(void *context, yvex_error *err)
{
    yvex_runtime_component_session *session = context;
    if (!session || !session->execution_transaction_leases) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-session.transaction",
                       "component execution retention is not active");
        return YVEX_ERR_STATE;
    }
    if (session->execution_transaction_leases == 1ull &&
        session->prepared_resources_borrowed &&
        component_prepared_resources_drop(session, err) != YVEX_OK)
        return yvex_error_code(err);
    session->execution_transaction_leases--;
    if (!session->execution_transaction_leases)
        memset(session->active_prepared_identity, 0,
               sizeof(session->active_prepared_identity));
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_component_session_borrow(
    yvex_runtime_component_session *session, yvex_component_execution *execution,
    yvex_error *err)
{
    if (execution) memset(execution, 0, sizeof(*execution));
    if (!session || !execution || !session->materialization || !session->residency ||
        !session->summary.sealed || session->summary.invalidated ||
        !yvex_sha256_hex_valid(session->summary.residency_identity)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component-session.borrow",
                       "one sealed runtime-owned component session is required");
        return YVEX_ERR_INVALID_ARG;
    }
    execution->schema_version = YVEX_COMPONENT_EXECUTION_SCHEMA_V1;
    execution->materialization = session->materialization;
    execution->backend = session->backend;
    execution->resident_encoded_bytes = session->summary.encoded_bytes;
    execution->owner_context = session;
    execution->weight_view = component_session_weight_view;
    execution->workspace_reserve = component_session_workspace_reserve;
    yvex_core_text_copy(execution->residency_identity,
                        sizeof(execution->residency_identity),
                        session->summary.residency_identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int component_execution_valid(const yvex_component_execution *execution)
{
    return execution &&
           execution->schema_version == YVEX_COMPONENT_EXECUTION_SCHEMA_V1 &&
           execution->materialization && execution->owner_context &&
           execution->weight_view && execution->workspace_reserve &&
           yvex_sha256_hex_valid(execution->residency_identity);
}

int yvex_component_execution_resource_lease(
    const yvex_component_execution *execution, yvex_execution_resource_lease *lease,
    yvex_error *err)
{
    if (lease) memset(lease, 0, sizeof(*lease));
    if (!component_execution_valid(execution) || !lease) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "component.execution.resource-lease",
                       "one runtime-owned component execution is required");
        return YVEX_ERR_INVALID_ARG;
    }
    lease->identity = execution->residency_identity;
    lease->context = execution->owner_context;
    lease->retain = component_session_transaction_retain;
    lease->release = component_session_transaction_release;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_component_execution_resource_summary(
    const yvex_component_execution *execution,
    yvex_component_resource_summary *summary, yvex_error *err)
{
    yvex_runtime_component_session *session;
    yvex_engine_resource_summary catalog = {0};
    unsigned long long count = 0ull;
    int rc;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!component_execution_valid(execution) || !summary) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "component.execution.resource-summary",
                       "one runtime-owned component execution is required");
        return YVEX_ERR_INVALID_ARG;
    }
    session = execution->owner_context;
    rc = yvex_runtime_resource_snapshot(
        session->execution_resources, &catalog, NULL, 0ull, &count, err);
    if (rc == YVEX_OK) {
        session->resource_summary.resource_count = count;
        session->resource_summary.resource_generation = catalog.generation;
        *summary = session->resource_summary;
    }
    return rc;
}

int yvex_component_execution_weight_view(
    const yvex_component_execution *execution, const char *name,
    yvex_component_encoded_weight *weight, yvex_error *err)
{
    if (!component_execution_valid(execution) || !name || !name[0] || !weight) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "component.execution.weight-view",
                       "one borrowed component execution and weight name are required");
        return YVEX_ERR_INVALID_ARG;
    }
    return execution->weight_view(execution->owner_context, name, weight, err);
}

static int component_text_weight_bind(
    const yvex_component_execution *execution, const char *name,
    yvex_backend_text_weight *weight, yvex_error *err)
{
    yvex_component_encoded_weight component = {0};
    int rc = yvex_component_execution_weight_view(execution, name, &component, err);
    if (rc == YVEX_OK) {
        weight->encoded = component.encoded;
        weight->encoded_bytes = component.encoded_bytes;
        weight->row_count = component.row_count;
        weight->row_width = component.row_width;
        weight->row_bytes = component.row_bytes;
        weight->qtype = component.qtype;
    }
    return rc;
}

static int component_text_weights_bind(
    const yvex_component_execution *execution, const yvex_component_text_request *request,
    yvex_backend_text_weight *weights, yvex_error *err)
{
    unsigned long long layer, slot, index;
    int rc = component_text_weight_bind(
        execution, request->embedding_weight_name, weights, err);
    for (layer = 0ull; rc == YVEX_OK && layer < request->layer_count; ++layer) {
        for (slot = 0ull; rc == YVEX_OK &&
                            slot < YVEX_COMPONENT_TEXT_LAYER_WEIGHT_COUNT; ++slot) {
            char name[256] = {0};
            index = 1ull + layer * YVEX_COMPONENT_TEXT_LAYER_WEIGHT_COUNT + slot;
            rc = request->layer_weight_name(
                request->weight_name_context, layer, (unsigned int)slot, name, err);
            if (rc == YVEX_OK)
                rc = component_text_weight_bind(execution, name, weights + index, err);
        }
    }
    return rc;
}

static void component_text_result_project(
    yvex_runtime_av_conditioning_result *out,
    const yvex_backend_text_execution_result *source)
{
    out->token_count = source->token_count;
    out->hidden_width = source->hidden_width;
    out->layer_count = source->layer_count;
    out->resident_bytes = source->resident_bytes;
    out->kernel_launches = source->kernel_launches;
    out->h2d_bytes = source->h2d_bytes;
    out->d2h_bytes = source->d2h_bytes;
    out->device_bytes = source->device_bytes;
    memcpy(out->residency_identity, source->residency_identity,
           sizeof(out->residency_identity));
    memcpy(out->execution_identity, source->execution_identity,
           sizeof(out->execution_identity));
    out->complete = source->complete;
}

static int component_text_request_validate(
    const yvex_component_text_request *request, const yvex_runtime_av_conditioning_result *result,
    unsigned long long *output_values, unsigned long long *output_bytes, yvex_error *err)
{
    if (!request || !request->recipe ||
        request->recipe->schema_version != YVEX_COMPONENT_TEXT_RECIPE_SCHEMA_V1 ||
        !request->embedding_weight_name || !request->token_ids || !request->token_count ||
        !request->output || !result || !output_values || !output_bytes ||
        request->layer_count > request->recipe->layer_capacity ||
        (request->layer_count && !request->layer_weight_name) ||
        !yvex_core_u64_mul(request->token_count, request->recipe->hidden_width, output_values) ||
        *output_values > request->output_capacity ||
        !yvex_core_u64_mul(*output_values, sizeof(float), output_bytes) ||
        *output_bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.text",
                       "one admitted bounded text-component recipe is required");
        return YVEX_ERR_INVALID_ARG;
    }
    return YVEX_OK;
}

int yvex_component_text_execute(
    const yvex_component_execution *execution, const yvex_component_text_request *request,
    yvex_runtime_av_conditioning_result *result, yvex_error *err)
{
    const yvex_backend_component_operations *operations = NULL;
    yvex_component_encoded_weight embedding = {0};
    yvex_backend_text_weight *weights = NULL;
    yvex_backend_text_execution_result backend_result = {0};
    unsigned long long output_values, output_bytes, weight_count = 0ull;
    float *staged = NULL;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    rc = component_text_request_validate(request, result, &output_values, &output_bytes, err);
    if (rc != YVEX_OK) return rc;
    if (!component_execution_valid(execution) || !execution->backend) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.text",
                       "one borrowed component execution is required");
        return YVEX_ERR_INVALID_ARG;
    }
    staged = (float *)malloc((size_t)output_bytes);
    if (!staged) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "runtime.component.text.output",
                       "transactional text-component output allocation failed");
        return YVEX_ERR_NOMEM;
    }
    if (request->layer_count &&
        (!yvex_core_u64_mul(request->layer_count,
                            YVEX_COMPONENT_TEXT_LAYER_WEIGHT_COUNT, &weight_count) ||
         !yvex_core_u64_add(weight_count, 1ull, &weight_count) ||
         weight_count > SIZE_MAX / sizeof(*weights) ||
         !(weights = (yvex_backend_text_weight *)calloc(
               (size_t)weight_count, sizeof(*weights))))) {
        free(staged);
        yvex_error_set(err, YVEX_ERR_NOMEM, "runtime.component.text.weights",
                       "bounded text-component bindings could not be allocated");
        return YVEX_ERR_NOMEM;
    }
    operations = yvex_backend_component_operations_get(execution->backend);
    if (!operations ||
        (!request->layer_count && !operations->text_embedding_execute) ||
        (request->layer_count && !operations->text_encoder_execute)) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "runtime.component.text",
                       "the selected backend lacks admitted text-component execution");
        rc = YVEX_ERR_UNSUPPORTED;
    }
    if (rc == YVEX_OK && !request->layer_count)
        rc = yvex_component_execution_weight_view(
            execution, request->embedding_weight_name, &embedding, err);
    if (rc == YVEX_OK && request->layer_count)
        rc = component_text_weights_bind(execution, request, weights, err);
    if (rc == YVEX_OK && !request->layer_count)
        rc = operations->text_embedding_execute(
            execution->backend, request->recipe, embedding.encoded, embedding.encoded_bytes,
            embedding.qtype, embedding.row_count, embedding.row_width, embedding.row_bytes,
            execution->residency_identity, execution->resident_encoded_bytes,
            request->token_ids, request->token_count, staged, output_values,
            &backend_result, err);
    if (rc == YVEX_OK && request->layer_count)
        rc = operations->text_encoder_execute(
            execution->backend, request->recipe, weights, request->layer_count,
            execution->residency_identity, execution->resident_encoded_bytes,
            request->token_ids, request->token_count, staged, output_values,
            &backend_result, err);
    if (rc == YVEX_OK) {
        memcpy(request->output, staged, (size_t)output_bytes);
        component_text_result_project(result, &backend_result);
        yvex_error_clear(err);
    }
    free(weights);
    free(staged);
    return rc;
}

int yvex_runtime_component_text_artifact_execute(
    const yvex_complete_artifact_admission *admission, const yvex_artifact *artifact,
    const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    yvex_backend_kind backend_kind, const yvex_component_text_request *request,
    yvex_runtime_av_conditioning_result *result, yvex_error *err)
{
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution execution = {0};
    unsigned long long output_values, output_bytes;
    yvex_error cleanup;
    int rc, cleanup_rc;
    if (result) memset(result, 0, sizeof(*result));
    rc = component_text_request_validate(request, result, &output_values, &output_bytes, err);
    if (rc != YVEX_OK) return rc;
    if (!admission || !artifact || !gguf || !tensors || !request->maximum_device_bytes) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.text",
                       "one admitted text-component artifact and resource budget are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_runtime_component_session_open(
        &session, admission, artifact, gguf, tensors, backend_kind,
        request->maximum_host_bytes, request->maximum_device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &execution, err);
    if (rc == YVEX_OK)
        rc = yvex_component_text_execute(&execution, request, result, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    return rc;
}

int yvex_component_vision_execute(
    const yvex_component_execution *component, const yvex_vision_request *request,
    yvex_vision_result *result, yvex_error *err)
{
    yvex_component_encoded_weight *weights = NULL;
    yvex_backend_vision_request execution = {0};
    unsigned long long count, index = 0ull, layer, slot;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!component_execution_valid(component) || !component->backend || !request ||
        !request->recipe || !request->weight_name || !result ||
        yvex_backend_kind_of(component->backend) != YVEX_BACKEND_KIND_CUDA ||
        !yvex_core_u64_mul(request->recipe->layer_count,
                           YVEX_VISION_BLOCK_WEIGHT_COUNT, &count) ||
        !yvex_core_u64_add(count, YVEX_VISION_EXTERNAL_WEIGHT_COUNT, &count) ||
        !yvex_core_u64_add(count,
                           (1ull + request->recipe->deepstack_layer_count) *
                               YVEX_VISION_MERGER_WEIGHT_COUNT,
                           &count) || count > SIZE_MAX / sizeof(*weights) ||
        !(weights = calloc((size_t)count, sizeof(*weights)))) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.vision",
                       "one sealed CUDA component and bounded vision recipe are required");
        return YVEX_ERR_INVALID_ARG;
    }
#define BIND(group, item, weight_slot) do { \
    char name[256] = {0}; \
    rc = request->weight_name(request->weight_name_context, group, item, weight_slot, name, err); \
    if (rc == YVEX_OK) rc = yvex_component_execution_weight_view( \
        component, name, weights + index, err); \
    ++index; \
} while (0)
    for (slot = 0ull; rc == YVEX_OK && slot < YVEX_VISION_EXTERNAL_WEIGHT_COUNT; ++slot)
        BIND(YVEX_VISION_WEIGHT_EXTERNAL, 0ull, (unsigned int)slot);
    for (layer = 0ull; rc == YVEX_OK && layer < request->recipe->layer_count; ++layer)
        for (slot = 0ull; rc == YVEX_OK && slot < YVEX_VISION_BLOCK_WEIGHT_COUNT; ++slot)
            BIND(YVEX_VISION_WEIGHT_BLOCK, layer, (unsigned int)slot);
    for (slot = 0ull; rc == YVEX_OK && slot < YVEX_VISION_MERGER_WEIGHT_COUNT; ++slot)
        BIND(YVEX_VISION_WEIGHT_MERGER, 0ull, (unsigned int)slot);
    for (layer = 0ull; rc == YVEX_OK && layer < request->recipe->deepstack_layer_count; ++layer)
        for (slot = 0ull; rc == YVEX_OK && slot < YVEX_VISION_MERGER_WEIGHT_COUNT; ++slot)
            BIND(YVEX_VISION_WEIGHT_DEEPSTACK, layer, (unsigned int)slot);
#undef BIND
    if (rc == YVEX_OK && index != count) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component.vision",
                       "vision weight binding count did not close");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        execution.request = request; execution.weights = weights; execution.weight_count = count;
        execution.residency_identity = component->residency_identity;
        execution.resident_bytes = component->resident_encoded_bytes;
        rc = yvex_backend_vision_execute(component->backend, &execution, result, err);
    }
    free(weights);
    return rc;
}

int yvex_component_multimodal_text_execute(
    const yvex_component_execution *component,
    const yvex_component_multimodal_text_request *request,
    yvex_runtime_av_conditioning_result *result, yvex_error *err)
{
    yvex_component_text_request binding_request = {0};
    const yvex_backend_component_operations *operations = NULL;
    yvex_backend_text_weight *weights = NULL;
    yvex_backend_text_execution_result backend_result = {0};
    unsigned long long weight_count;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!component_execution_valid(component) || !request || !request->recipe ||
        !request->embedding_weight_name ||
        !request->layer_weight_name || !request->token_ids || !request->token_count ||
        !request->layer_count || !request->multimodal || !request->output || !result ||
        !component->backend ||
        yvex_backend_kind_of(component->backend) != YVEX_BACKEND_KIND_CUDA ||
        !yvex_core_u64_mul(request->layer_count, YVEX_COMPONENT_TEXT_LAYER_WEIGHT_COUNT,
                           &weight_count) || !yvex_core_u64_add(weight_count, 1ull, &weight_count) ||
        weight_count > SIZE_MAX / sizeof(*weights) ||
        !(weights = calloc((size_t)weight_count, sizeof(*weights)))) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.multimodal-text",
                       "one sealed CUDA component and multimodal text request are required");
        return YVEX_ERR_INVALID_ARG;
    }
    binding_request.recipe = request->recipe;
    binding_request.embedding_weight_name = request->embedding_weight_name;
    binding_request.layer_weight_name = request->layer_weight_name;
    binding_request.weight_name_context = request->weight_name_context;
    binding_request.layer_count = request->layer_count;
    operations = yvex_backend_component_operations_get(component->backend);
    if (!operations || !operations->text_encoder_multimodal_execute) {
        free(weights);
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "runtime.component.multimodal-text",
                       "the admitted backend lacks multimodal text execution");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = component_text_weights_bind(component, &binding_request, weights, err);
    if (rc == YVEX_OK)
        rc = operations->text_encoder_multimodal_execute(
            component->backend, request->recipe, weights, request->layer_count,
            component->residency_identity, component->resident_encoded_bytes,
            request->token_ids, request->token_count, request->multimodal,
            request->output, request->output_capacity, &backend_result, err);
    if (rc == YVEX_OK) component_text_result_project(result, &backend_result);
    free(weights);
    return rc;
}

static int component_joint_weight_bind(
    const yvex_component_execution *execution, const char *name,
    yvex_transformer_joint_encoded_weight *weight, yvex_error *err)
{
    yvex_component_encoded_weight component = {0};
    int rc = yvex_component_execution_weight_view(execution, name, &component, err);
    if (rc == YVEX_OK) {
        weight->encoded = component.encoded;
        weight->encoded_bytes = component.encoded_bytes;
        weight->row_count = component.row_count;
        weight->row_width = component.row_width;
        weight->row_bytes = component.row_bytes;
        weight->qtype = component.qtype;
    }
    return rc;
}

static int component_joint_prepared_identity(
    const yvex_runtime_component_session *session,
    const yvex_transformer_joint_request *request,
    char identity[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!session || !request || !request->recipe ||
        !request->recipe->identity_domain ||
        !yvex_sha256_hex_valid(request->layout_identity) ||
        !yvex_sha256_hex_valid(request->condition_identity)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component-resource",
                       "layout and condition identities are required for preparation");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.joint-execution-preparation.v1") ||
        !yvex_sha256_update_text(&hash, session->summary.residency_identity) ||
        !yvex_sha256_update_text(&hash, request->recipe->identity_domain) ||
        !yvex_sha256_update_text(&hash, request->layout_identity) ||
        !yvex_sha256_update_text(&hash, request->condition_identity) ||
        !yvex_sha256_update_text(
            &hash, request->video_output_physical.physical_identity) ||
        !yvex_sha256_update_text(
            &hash, request->audio_output_physical.physical_identity) ||
        !yvex_sha256_update_u64(&hash, request->video_rows) ||
        !yvex_sha256_update_u64(&hash, request->audio_rows) ||
        !yvex_sha256_update_u64(&hash, request->text_rows) ||
        !yvex_sha256_update_u64(&hash, request->packed_rows) ||
        !yvex_sha256_update_u64(&hash, request->block_count) ||
        !yvex_sha256_final(&hash, digest)) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-resource",
                       "prepared execution identity could not be sealed");
        return YVEX_ERR_STATE;
    }
    yvex_sha256_hex(digest, identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int component_prepared_view_release(void *context, yvex_error *err)
{
    (void)context;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int component_joint_resources_evict(
    yvex_runtime_component_session *session, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!session || session->prepared_resources_borrowed) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-resource",
                       "borrowed prepared resources cannot be invalidated");
        return YVEX_ERR_STATE;
    }
    if (component_resource_handle_present(session->condition_prepared_resource))
        rc = yvex_runtime_resource_evict(
            session->execution_resources,
            &session->condition_prepared_resource, err);
    if (rc == YVEX_OK &&
        component_resource_handle_present(session->request_prepared_resource))
        rc = yvex_runtime_resource_evict(
            session->execution_resources,
            &session->request_prepared_resource, err);
    if (rc == YVEX_OK &&
        component_resource_handle_present(session->execution_arena_resource))
        rc = yvex_runtime_resource_evict(
            session->execution_resources,
            &session->execution_arena_resource, err);
    return rc;
}

static int component_joint_resource_register_one(
    yvex_runtime_component_session *session, yvex_engine_resource_kind kind,
    yvex_engine_resource_lifetime lifetime, const char *name,
    yvex_engine_resource_handle dependency,
    yvex_engine_resource_bytes bytes, unsigned long long preparation_ns,
    yvex_engine_resource_release_fn release,
    yvex_engine_resource_handle *handle, yvex_error *err)
{
    yvex_engine_resource_request resource = {0};
    resource.kind = kind;
    resource.owner = YVEX_ENGINE_RESOURCE_OWNER_EXECUTION;
    resource.lifetime = lifetime;
    resource.numeric_class = kind == YVEX_ENGINE_RESOURCE_WORKSPACE
                                 ? YVEX_ENGINE_RESOURCE_NUMERIC_STATE
                                 : YVEX_ENGINE_RESOURCE_NUMERIC_EQUIVALENT_PREPARED;
    resource.name = name;
    resource.package_identity = session->package_identity;
    resource.specialization_identity = session->summary.residency_identity;
    resource.admission_identity = session->joint_prepared_summary.identity;
    resource.dependency = dependency;
    resource.bytes = bytes;
    resource.preparation_nanoseconds = preparation_ns;
    resource.value = session->joint_prepared;
    resource.release = release;
    resource.release_context = session;
    resource.ready = 1;
    resource.evictable = 1;
    return yvex_runtime_resource_register(
        session->execution_resources, &resource, handle, err);
}

static int component_joint_resources_register(
    yvex_runtime_component_session *session, yvex_error *err)
{
    yvex_engine_resource_bytes bytes = {0};
    yvex_engine_resource_summary catalog = {0};
    unsigned long long count = 0ull, workspace_bytes;
    int rc;
    if (!yvex_core_u64_add(session->joint_prepared_summary.host_arena_bytes,
                           session->joint_prepared_summary.device_arena_bytes,
                           &workspace_bytes)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "runtime.component-resource",
                       "execution arena accounting overflowed");
        return YVEX_ERR_BOUNDS;
    }
    bytes.host_resident_bytes = session->joint_prepared_summary.host_arena_bytes;
    bytes.device_resident_bytes = session->joint_prepared_summary.device_arena_bytes;
    bytes.workspace_bytes = workspace_bytes;
    rc = component_joint_resource_register_one(
        session, YVEX_ENGINE_RESOURCE_WORKSPACE,
        YVEX_ENGINE_RESOURCE_LIFETIME_REQUEST, "joint-execution-arena",
        (yvex_engine_resource_handle){0}, bytes,
        session->joint_prepared_summary.preparation_nanoseconds,
        component_joint_prepared_release,
        &session->execution_arena_resource, err);
    memset(&bytes, 0, sizeof(bytes));
    bytes.prepared_bytes = session->joint_prepared_summary.request_prepared_bytes;
    if (rc == YVEX_OK && session->joint_prepared_summary.request_ready)
        rc = component_joint_resource_register_one(
            session, YVEX_ENGINE_RESOURCE_PREPARED_LAYOUT,
            YVEX_ENGINE_RESOURCE_LIFETIME_REQUEST, "joint-request-layout",
            session->execution_arena_resource, bytes, 0ull,
            component_prepared_view_release,
            &session->request_prepared_resource, err);
    memset(&bytes, 0, sizeof(bytes));
    bytes.prepared_bytes = session->joint_prepared_summary.condition_prepared_bytes;
    if (rc == YVEX_OK && session->joint_prepared_summary.condition_ready)
        rc = component_joint_resource_register_one(
            session, YVEX_ENGINE_RESOURCE_PREPARED_TENSOR,
            YVEX_ENGINE_RESOURCE_LIFETIME_CONDITION, "joint-condition-state",
            session->execution_arena_resource, bytes, 0ull,
            component_prepared_view_release,
            &session->condition_prepared_resource, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_resource_snapshot(
            session->execution_resources, &catalog, NULL, 0ull, &count, err);
    if (rc == YVEX_OK) {
        session->resource_summary.resource_count = count;
        session->resource_summary.resource_generation = catalog.generation;
    }
    return rc;
}

static int component_joint_resources_prepare(
    yvex_runtime_component_session *session,
    const yvex_backend_component_operations *operations,
    const yvex_transformer_joint_encoded_weight *external,
    const yvex_transformer_joint_encoded_weight *blocks,
    const yvex_transformer_joint_request *request, int *reused,
    yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP];
    unsigned long long preparation_total;
    int had_prepared, rc;
    *reused = 0;
    rc = component_joint_prepared_identity(session, request, identity, err);
    if (rc != YVEX_OK) return rc;
    if (session->active_prepared_identity[0] &&
        strcmp(session->active_prepared_identity, identity) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.component-resource",
                       "one transaction cannot change its prepared execution identity");
        return YVEX_ERR_STATE;
    }
    if (session->joint_prepared &&
        strcmp(session->joint_prepared_summary.identity, identity) == 0) {
        *reused = 1;
    } else {
        had_prepared = session->joint_prepared != NULL;
        if (had_prepared) {
            if (session->resource_summary.rebuild_count == ULLONG_MAX) {
                yvex_error_set(err, YVEX_ERR_BOUNDS,
                               "runtime.component-resource.accounting",
                               "prepared resource rebuild accounting overflowed");
                return YVEX_ERR_BOUNDS;
            }
            rc = component_joint_resources_evict(session, err);
            if (rc != YVEX_OK) return rc;
            session->resource_summary.rebuild_count++;
        }
        rc = operations->joint_transformer_prepare(
            session->backend, external, blocks, session->summary.residency_identity,
            session->summary.encoded_bytes, identity, request,
            &session->joint_prepared, &session->joint_prepared_summary, err);
        if (rc != YVEX_OK) return rc;
        if (session->resource_summary.preparation_count == ULLONG_MAX ||
            !yvex_core_u64_add(
                session->resource_summary.preparation_nanoseconds,
                session->joint_prepared_summary.preparation_nanoseconds,
                &preparation_total)) {
            yvex_error cleanup;
            int cleanup_rc;
            yvex_error_set(err, YVEX_ERR_BOUNDS,
                           "runtime.component-resource.accounting",
                           "prepared resource timing accounting overflowed");
            yvex_error_clear(&cleanup);
            cleanup_rc = operations->joint_transformer_prepared_release(
                session->backend, &session->joint_prepared, &cleanup);
            if (cleanup_rc != YVEX_OK && err) *err = cleanup;
            return cleanup_rc == YVEX_OK ? YVEX_ERR_BOUNDS : cleanup_rc;
        }
        rc = component_joint_resources_register(session, err);
        if (rc != YVEX_OK) {
            yvex_error primary = err ? *err : (yvex_error){0}, cleanup;
            int cleanup_rc;
            yvex_error_clear(&cleanup);
            cleanup_rc = component_joint_resources_evict(session, &cleanup);
            if (cleanup_rc == YVEX_OK && session->joint_prepared)
                cleanup_rc = operations->joint_transformer_prepared_release(
                    session->backend, &session->joint_prepared, &cleanup);
            if (cleanup_rc != YVEX_OK) {
                if (err) *err = cleanup;
                return cleanup_rc;
            }
            if (err) *err = primary;
            return rc;
        }
        session->resource_summary.preparation_count++;
        session->resource_summary.preparation_nanoseconds = preparation_total;
    }
    if (!session->prepared_resources_borrowed) {
        rc = component_prepared_resources_borrow(session, err);
        if (rc != YVEX_OK) return rc;
    }
    if (session->execution_transaction_leases)
        yvex_core_text_copy(session->active_prepared_identity,
                            sizeof(session->active_prepared_identity), identity);
    session->resource_summary.schema_version = YVEX_COMPONENT_RESOURCE_SUMMARY_SCHEMA_V1;
    session->resource_summary.host_arena_bytes =
        session->joint_prepared_summary.host_arena_bytes;
    session->resource_summary.device_arena_bytes =
        session->joint_prepared_summary.device_arena_bytes;
    session->resource_summary.request_prepared_bytes =
        session->joint_prepared_summary.request_prepared_bytes;
    session->resource_summary.condition_prepared_bytes =
        session->joint_prepared_summary.condition_prepared_bytes;
    session->resource_summary.allocation_count =
        session->joint_prepared_summary.allocation_count;
    session->resource_summary.ready = 1;
    session->resource_summary.request_ready =
        session->joint_prepared_summary.request_ready;
    session->resource_summary.condition_ready =
        session->joint_prepared_summary.condition_ready;
    yvex_core_text_copy(session->resource_summary.prepared_identity,
                        sizeof(session->resource_summary.prepared_identity), identity);
    return YVEX_OK;
}

int yvex_component_joint_transformer_execute(
    const yvex_component_execution *execution, const char *const *external_names,
    unsigned long long external_count, yvex_component_joint_weight_name_fn block_weight_name,
    void *weight_name_context, const yvex_transformer_joint_request *request,
    yvex_transformer_joint_result *result, yvex_error *err)
{
    const yvex_backend_transformer_operations *transformer_operations = NULL;
    const yvex_backend_component_operations *component_operations = NULL;
    yvex_runtime_component_session *session = NULL;
    yvex_transformer_joint_encoded_weight external[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {{0}};
    yvex_transformer_joint_encoded_weight *blocks = NULL;
    yvex_transformer_attention_requirement attention = {0};
    yvex_backend_memory_stats before = {0}, after = {0};
    yvex_error observation;
    unsigned long long count, index, workspace_bytes = 0ull;
    int prepared_path = 0, reused = 0, have_before = 0, rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!component_execution_valid(execution) || !external_names ||
        external_count != YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT ||
        !block_weight_name || !request || !request->recipe || !result ||
        !request->block_count || request->block_count > request->recipe->block_count ||
        !execution->backend ||
        !yvex_core_u64_mul(request->block_count,
                           YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT, &count) ||
        count > SIZE_MAX / sizeof(*blocks) ||
        !(blocks = (yvex_transformer_joint_encoded_weight *)calloc(
              (size_t)count, sizeof(*blocks)))) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.joint-transformer",
                       "one sealed backend component and complete named recipe are required");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0ull; rc == YVEX_OK && index < external_count; ++index) {
        if (!external_names[index]) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                           "runtime.component.joint-transformer.external",
                           "every external joint-transformer weight requires a name");
            rc = YVEX_ERR_INVALID_ARG;
        } else {
            rc = component_joint_weight_bind(execution, external_names[index],
                                             external + index, err);
        }
    }
    for (index = 0ull; rc == YVEX_OK && index < count; ++index) {
        char name[256] = {0};
        unsigned long long block = index / YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT;
        unsigned int slot = (unsigned int)(index % YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT);
        rc = block_weight_name(weight_name_context, block, slot, name, err);
        if (rc == YVEX_OK)
            rc = component_joint_weight_bind(execution, name, blocks + index, err);
    }
    if (rc == YVEX_OK) {
        transformer_operations = yvex_backend_transformer_operations_get(execution->backend);
        component_operations = yvex_backend_component_operations_get(execution->backend);
        if (!transformer_operations || !transformer_operations->attention_workspace_required ||
            !transformer_operations->attention_execute ||
            !transformer_operations->linear_workspace_required ||
            !transformer_operations->linear_compile ||
            !transformer_operations->linear_execute ||
            !transformer_operations->linear_summary ||
            !transformer_operations->linear_release ||
            !component_operations || !component_operations->joint_transformer_execute) {
            yvex_error_set(err, YVEX_ERR_UNSUPPORTED,
                           "runtime.component.joint-transformer",
                           "the admitted backend lacks transformer workspace planning");
            rc = YVEX_ERR_UNSUPPORTED;
        }
    }
    if (rc == YVEX_OK) {
        attention.tokens = request->packed_rows;
        attention.query_heads = request->recipe->attention_heads;
        attention.key_value_heads = request->recipe->attention_heads;
        attention.head_dimension = request->recipe->head_dimension;
        attention.query_dtype = YVEX_DTYPE_F32;
        attention.key_dtype = YVEX_DTYPE_F32;
        attention.value_dtype = YVEX_DTYPE_F32;
        attention.output_dtype = YVEX_DTYPE_F32;
        attention.layout = YVEX_TRANSFORMER_ATTENTION_LAYOUT_TOKEN_HEAD_DIM;
        attention.mask = YVEX_TRANSFORMER_ATTENTION_MASK_FULL;
        attention.numeric_contract = YVEX_TRANSFORMER_ATTENTION_NUMERIC_EXACT_F32;
        attention.deterministic = 1;
        rc = transformer_operations->attention_workspace_required(
            &attention, &workspace_bytes, err);
    }
    for (index = 0ull; rc == YVEX_OK &&
         index < YVEX_TRANSFORMER_JOINT_LINEAR_COUNT; ++index) {
        unsigned long long dense_bytes = 0ull;
        yvex_transformer_linear_requirement requirement;
        yvex_transformer_linear_compile_request compile;
        rc = yvex_transformer_joint_linear_requirement(
            request->recipe, (yvex_transformer_joint_linear_slot)index,
            &requirement, err);
        compile = (yvex_transformer_linear_compile_request){
            request->recipe->identity_domain, &requirement,
            index == YVEX_TRANSFORMER_JOINT_LINEAR_MODULATION
                ? request->timestep_count : request->packed_rows};
        if (rc == YVEX_OK)
            rc = transformer_operations->linear_workspace_required(
                &compile, &dense_bytes, err);
        if (dense_bytes > workspace_bytes) workspace_bytes = dense_bytes;
    }
    if (rc == YVEX_OK && workspace_bytes)
        rc = execution->workspace_reserve(
            execution->owner_context, workspace_bytes, err);
    session = execution->owner_context;
    prepared_path = rc == YVEX_OK &&
                    yvex_sha256_hex_valid(request->layout_identity) &&
                    yvex_sha256_hex_valid(request->condition_identity) &&
                    component_operations->joint_transformer_prepare &&
                    component_operations->joint_transformer_prepared_execute &&
                    component_operations->joint_transformer_prepared_release;
    if (prepared_path)
        rc = component_joint_resources_prepare(
            session, component_operations, external, blocks, request, &reused, err);
    if (rc == YVEX_OK) {
        yvex_error_clear(&observation);
        have_before = yvex_backend_get_memory_stats(
                          execution->backend, &before, &observation) == YVEX_OK;
    }
    if (rc == YVEX_OK) {
        rc = prepared_path
                 ? component_operations->joint_transformer_prepared_execute(
                       execution->backend, external, blocks,
                       execution->residency_identity,
                       execution->resident_encoded_bytes, session->joint_prepared,
                       request, result, err)
                 : component_operations->joint_transformer_execute(
                       execution->backend, external, blocks,
                       execution->residency_identity,
                       execution->resident_encoded_bytes, request, result, err);
    }
    if (prepared_path && rc == YVEX_OK) {
        unsigned long long events = 0ull;
        if (have_before) {
            yvex_error_clear(&observation);
            if (yvex_backend_get_memory_stats(
                    execution->backend, &after, &observation) == YVEX_OK &&
                after.allocation_events >= before.allocation_events)
                events = after.allocation_events - before.allocation_events;
        }
        if (session->resource_summary.use_count == ULLONG_MAX ||
            (reused && session->resource_summary.reuse_count == ULLONG_MAX) ||
            !yvex_core_u64_add(
                session->resource_summary.execution_allocation_events,
                events,
                &session->resource_summary.execution_allocation_events)) {
            yvex_error_set(err, YVEX_ERR_BOUNDS,
                           "runtime.component-resource.accounting",
                           "prepared execution accounting overflowed");
            rc = YVEX_ERR_BOUNDS;
        } else {
            session->resource_summary.use_count++;
            session->resource_summary.reuse_count += reused != 0;
            session->resource_summary.last_execution_allocation_events = events;
        }
    }
    if (prepared_path && !session->execution_transaction_leases &&
        session->prepared_resources_borrowed) {
        yvex_error primary = err ? *err : (yvex_error){0};
        yvex_error cleanup;
        int cleanup_rc;
        yvex_error_clear(&cleanup);
        cleanup_rc = component_prepared_resources_drop(session, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        } else if (rc != YVEX_OK && err) {
            *err = primary;
        }
    }
    free(blocks);
    return rc;
}

static int component_decoder_weight_bind(
    const yvex_component_execution *execution, const char *name,
    yvex_transformer_encoded_weight *weight, yvex_error *err)
{
    yvex_component_encoded_weight component = {0};
    unsigned long long elements, bytes;
    int rc = yvex_component_execution_weight_view(execution, name, &component, err);
    if (rc != YVEX_OK) return rc;
    if (component.qtype != YVEX_GGUF_QTYPE_F32) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.component.dense-decoder.binding",
                       "one exact F32 vector or source-ordered matrix binding is required");
        return YVEX_ERR_FORMAT;
    }
    weight->row_count = component.row_count;
    weight->row_width = component.row_width;
    if (!weight->row_count || !weight->row_width ||
        !yvex_core_u64_mul(weight->row_count, weight->row_width, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) ||
        bytes != component.encoded_bytes) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.component.dense-decoder.binding",
                       "resident F32 binding disagrees with its logical source shape");
        return YVEX_ERR_FORMAT;
    }
    weight->encoded = component.encoded;
    weight->encoded_bytes = component.encoded_bytes;
    weight->row_bytes = component.row_bytes;
    weight->qtype = component.qtype;
    return YVEX_OK;
}

int yvex_component_dense_decoder_execute(
    const yvex_component_execution *component,
    const yvex_transformer_resident_decoder_request *request,
    yvex_transformer_dense_decoder_result *result, yvex_error *err)
{
    const yvex_backend_transformer_operations *operations =
        yvex_backend_transformer_operations_get(component ? component->backend : NULL);
    yvex_transformer_dense_decoder_request execution;
    yvex_transformer_encoded_weight *blocks = NULL;
    yvex_transformer_encoded_weight final_norm = {0}, final_bias = {0};
    yvex_transformer_encoded_weight output_weight = {0}, output_bias = {0};
    unsigned long long weight_count, index;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!component_execution_valid(component) || !request || !result ||
        !request->block_weight_name ||
        !request->final_norm_weight_name || !request->final_norm_bias_name ||
        !request->output_weight_name || !request->output_bias_name ||
        !request->execution.block_count || !component->backend ||
        !operations || !operations->dense_decoder_execute ||
        !yvex_core_u64_mul(request->execution.block_count,
                           YVEX_TRANSFORMER_DENSE_DECODER_BLOCK_WEIGHT_COUNT,
                           &weight_count) || weight_count > SIZE_MAX / sizeof(*blocks)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.dense-decoder",
                       "one sealed backend component and complete named decoder recipe are required");
        return YVEX_ERR_INVALID_ARG;
    }
    blocks = (yvex_transformer_encoded_weight *)calloc(
        (size_t)weight_count, sizeof(*blocks));
    if (!blocks) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "runtime.component.dense-decoder",
                       "resident decoder binding directory allocation failed");
        return YVEX_ERR_NOMEM;
    }
    rc = YVEX_OK;
    for (index = 0ull; index < weight_count && rc == YVEX_OK; ++index) {
        char name[256];
        unsigned long long block =
            index / YVEX_TRANSFORMER_DENSE_DECODER_BLOCK_WEIGHT_COUNT;
        unsigned int slot = (unsigned int)(
            index % YVEX_TRANSFORMER_DENSE_DECODER_BLOCK_WEIGHT_COUNT);
        rc = request->block_weight_name(
            request->block_weight_name_context, block, slot, name, err);
        if (rc == YVEX_OK)
            rc = component_decoder_weight_bind(component, name, blocks + index, err);
    }
    if (rc == YVEX_OK)
        rc = component_decoder_weight_bind(
            component, request->final_norm_weight_name, &final_norm, err);
    if (rc == YVEX_OK)
        rc = component_decoder_weight_bind(
            component, request->final_norm_bias_name, &final_bias, err);
    if (rc == YVEX_OK)
        rc = component_decoder_weight_bind(
            component, request->output_weight_name, &output_weight, err);
    if (rc == YVEX_OK)
        rc = component_decoder_weight_bind(
            component, request->output_bias_name, &output_bias, err);
    execution = request->execution;
    execution.block_weights = blocks;
    execution.final_norm_weight = &final_norm;
    execution.final_norm_bias = &final_bias;
    execution.output_weight = &output_weight;
    execution.output_bias = &output_bias;
    if (rc == YVEX_OK)
        rc = operations->dense_decoder_execute(component->backend, &execution, result, err);
    free(blocks);
    return rc;
}

int yvex_component_buffer_open(
    yvex_component_f32_buffer *buffer, unsigned long long count,
    unsigned long long maximum, unsigned long long *live,
    unsigned long long *peak, const char *stage, const char *label, yvex_error *err)
{
    unsigned long long bytes, next;
    if (buffer) memset(buffer, 0, sizeof(*buffer));
    if (!buffer || !live || !peak || !stage || !label || !count ||
        !yvex_core_u64_mul(count, sizeof(float), &bytes) ||
        bytes > (unsigned long long)SIZE_MAX ||
        !yvex_core_u64_add(*live, bytes, &next)) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, stage,
                        "%s workspace extent overflowed", label);
        return YVEX_ERR_BOUNDS;
    }
    if (next > maximum) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, stage,
                        "%s workspace budget was exceeded", label);
        return YVEX_ERR_BOUNDS;
    }
    buffer->data = (float *)malloc((size_t)bytes);
    if (!buffer->data) {
        yvex_error_setf(err, YVEX_ERR_NOMEM, stage,
                        "%s workspace allocation failed", label);
        return YVEX_ERR_NOMEM;
    }
    buffer->count = count;
    *live = next;
    if (next > *peak) *peak = next;
    return YVEX_OK;
}

void yvex_component_buffer_close(yvex_component_f32_buffer *buffer,
                                 unsigned long long *live)
{
    unsigned long long bytes;
    if (!buffer || !live) return;
    bytes = buffer->count * sizeof(float);
    if (bytes <= *live) *live -= bytes;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

static const yvex_materialized_tensor_binding *component_binding_find(
    const yvex_materialization_session *session, const char *name)
{
    unsigned long long index;
    if (!session || !name) return NULL;
    for (index = 0ull;; ++index) {
        const yvex_materialized_tensor_binding *binding =
            yvex_materialization_session_tensor_at(session, index);
        if (!binding || strcmp(binding->name, name) == 0) return binding;
    }
}

static int component_weight_bind(
    const yvex_materialization_session *session,
    const yvex_runtime_residency *residency, const char *name,
    yvex_component_encoded_weight *weight, yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding =
        component_binding_find(session, name);
    if (!binding || !binding->row_count || !residency || !weight) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.component.binding",
                       "an exact resident component weight binding is unavailable");
        return YVEX_ERR_FORMAT;
    }
    memset(weight, 0, sizeof(*weight));
    if (yvex_runtime_residency_binding_view(
            residency, binding, &weight->encoded, &weight->encoded_bytes, err) != YVEX_OK)
        return yvex_error_code(err);
    weight->qtype = binding->qtype;
    weight->row_count = binding->row_count;
    weight->row_width = binding->row_width;
    weight->row_bytes = binding->encoded_bytes / binding->row_count;
    return YVEX_OK;
}

static int component_weight_bind_sized(
    void *context, const char *name, unsigned long long rows,
    unsigned long long width, yvex_component_encoded_weight *weight, yvex_error *err)
{
    const yvex_component_execution *execution = context;
    unsigned long long values, expected_bytes;
    int rc;
    if (!component_execution_valid(execution) || !weight ||
        !yvex_core_u64_mul(rows, width, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &expected_bytes)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.binding",
                       "component weight binding geometry is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_component_execution_weight_view(execution, name, weight, err);
    if (rc == YVEX_OK && weight->encoded_bytes != expected_bytes) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "runtime.component.binding",
                        "component tensor %s has %llu bytes, expected %llu",
                        name, weight->encoded_bytes, expected_bytes);
        rc = YVEX_ERR_FORMAT;
    }
    return rc;
}

int yvex_component_alias_decoder_execute(
    const yvex_component_execution *component,
    const yvex_alias_decoder_request *request,
    yvex_alias_decoder_result *result, yvex_error *err)
{
    const yvex_backend_component_operations *operations =
        yvex_backend_component_operations_get(component ? component->backend : NULL);
    yvex_alias_decoder_request execution;
    if (result) memset(result, 0, sizeof(*result));
    if (!component_execution_valid(component) || !request || !result ||
        !component->backend || !operations || !operations->alias_decoder_execute ||
        request->weight_bind ||
        request->weight_bind_context) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.alias-decoder",
                       "one sealed backend component and unbound decoder recipe are required");
        return YVEX_ERR_INVALID_ARG;
    }
    execution = *request;
    execution.weight_bind = component_weight_bind_sized;
    execution.weight_bind_context = (void *)component;
    return operations->alias_decoder_execute(component->backend, &execution, result, err);
}

static int component_load_reject(
    yvex_component_load_failure *failure, yvex_component_load_code code,
    const char *name, unsigned long long expected, unsigned long long actual,
    const char *reason, yvex_status status, const char *stage, yvex_error *err)
{
    failure->code = code;
    failure->expected = expected;
    failure->actual = actual;
    failure->reason = reason;
    yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), name);
    yvex_error_setf(err, status, stage, "component tensor contract rejected %s", name);
    return status;
}

int yvex_component_f32_load(
    yvex_materialization_session *session, const char *name, unsigned int rank,
    const unsigned long long *dims, yvex_component_f32_buffer *buffer,
    unsigned long long maximum, unsigned long long *live, unsigned long long *peak,
    unsigned long long *reads, unsigned long long *payload,
    yvex_component_load_failure *failure, const char *stage, const char *label,
    yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding;
    yvex_materialization_failure materialization_failure;
    unsigned long long count = 1ull, expected_bytes = 0ull;
    unsigned int dimension;
    int rc;
    if (!session || !name || !dims || !buffer || !live || !peak || !reads ||
        !payload || !failure || !stage || !label) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.load",
                       "complete bounded component load state is required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(failure, 0, sizeof(*failure));
    binding = component_binding_find(session, name);
    if (!binding)
        return component_load_reject(failure, YVEX_COMPONENT_LOAD_MISSING, name,
                                     1ull, 0ull, "component tensor is missing",
                                     YVEX_ERR_FORMAT, stage, err);
    if (binding->qtype != YVEX_GGUF_QTYPE_F32 || binding->rank != rank)
        return component_load_reject(failure, YVEX_COMPONENT_LOAD_CONTRACT, name,
                                     rank, binding->rank,
                                     "component tensor rank or qtype differs",
                                     YVEX_ERR_FORMAT, stage, err);
    for (dimension = 0u; dimension < rank; ++dimension)
        if (binding->dims[dimension] != dims[dimension] ||
            !yvex_core_u64_mul(count, dims[dimension], &count))
            return component_load_reject(failure, YVEX_COMPONENT_LOAD_CONTRACT, name,
                                         dims[dimension], binding->dims[dimension],
                                         "component tensor shape differs",
                                         YVEX_ERR_FORMAT, stage, err);
    if (!yvex_core_u64_mul(count, sizeof(float), &expected_bytes) ||
        binding->encoded_bytes != expected_bytes)
        return component_load_reject(failure, YVEX_COMPONENT_LOAD_CONTRACT, name,
                                     expected_bytes, binding->encoded_bytes,
                                     "component tensor byte extent differs",
                                     YVEX_ERR_FORMAT, stage, err);
    rc = yvex_component_buffer_open(
        buffer, count, maximum, live, peak, stage, label, err);
    if (rc != YVEX_OK) {
        failure->code = YVEX_COMPONENT_LOAD_BUDGET;
        failure->reason = yvex_error_message(err);
        return rc;
    }
    rc = yvex_materialization_session_read(
        session, binding, 0ull, buffer->data, (size_t)binding->encoded_bytes,
        &materialization_failure, err);
    if (rc != YVEX_OK) {
        yvex_component_buffer_close(buffer, live);
        failure->code = YVEX_COMPONENT_LOAD_MATERIALIZATION;
        failure->expected = binding->encoded_bytes;
        failure->actual = materialization_failure.actual;
        failure->reason = materialization_failure.reason;
        yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), name);
        return rc;
    }
    (*reads)++;
    *payload += binding->encoded_bytes;
    return YVEX_OK;
}

typedef struct {
    yvex_materialization_session *session;
    const yvex_alias_decoder_request *request;
    yvex_alias_decoder_result *result;
    yvex_component_execution_failure *failure;
    unsigned long long live_bytes;
    yvex_error *err;
} alias_cpu_run;

static int alias_cpu_refuse(alias_cpu_run *run, yvex_component_execution_code code,
                            const char *name, unsigned long long expected,
                            unsigned long long actual, yvex_status status,
                            const char *reason)
{
    if (run && run->failure) {
        run->failure->code = code;
        run->failure->expected = expected;
        run->failure->actual = actual;
        run->failure->reason = reason;
        yvex_core_text_copy(run->failure->tensor_name,
                            sizeof(run->failure->tensor_name), name);
    }
    yvex_error_set(run ? run->err : NULL, status, "graph.alias-decoder.cpu", reason);
    return status;
}

static int alias_cpu_cancel(alias_cpu_run *run)
{
    if (run->request->cancel_requested &&
        run->request->cancel_requested(run->request->cancel_context))
        return alias_cpu_refuse(run, YVEX_COMPONENT_EXECUTION_CANCELLED, NULL, 0ull, 1ull,
                                YVEX_ERR_CANCELLED,
                                "alias decoder was cancelled between layers");
    return YVEX_OK;
}

static int alias_cpu_buffer_open(alias_cpu_run *run, yvex_component_f32_buffer *buffer,
                                 unsigned long long count)
{
    int rc = yvex_component_buffer_open(
        buffer, count, run->request->maximum_workspace_bytes, &run->live_bytes,
        &run->result->peak_host_bytes, "graph.alias-decoder.cpu", "alias decoder", run->err);
    return rc == YVEX_OK
               ? rc
               : alias_cpu_refuse(run, YVEX_COMPONENT_EXECUTION_BUDGET, NULL,
                                  run->request->maximum_workspace_bytes, count,
                                  (yvex_status)rc, yvex_error_message(run->err));
}

static void alias_cpu_buffer_close(alias_cpu_run *run, yvex_component_f32_buffer *buffer)
{
    yvex_component_buffer_close(buffer, &run->live_bytes);
}

static int alias_cpu_name(alias_cpu_run *run, yvex_alias_decoder_weight_role role,
                          unsigned long long stage, unsigned long long block,
                          unsigned long long layer, char output[256])
{
    return run->request->weight_name(
        run->request->weight_name_context, role, stage, block, layer, output, run->err);
}

static int alias_cpu_load(alias_cpu_run *run, yvex_alias_decoder_weight_role role,
                          unsigned long long stage, unsigned long long block,
                          unsigned long long layer, unsigned int rank,
                          const unsigned long long *dims, yvex_component_f32_buffer *buffer)
{
    yvex_component_load_failure issue = {0};
    char name[256] = {0};
    int rc = alias_cpu_name(run, role, stage, block, layer, name);
    if (rc == YVEX_OK)
        rc = yvex_component_f32_load(
            run->session, name, rank, dims, buffer, run->request->maximum_workspace_bytes,
            &run->live_bytes, &run->result->peak_host_bytes, &run->result->tensor_reads,
            &run->result->payload_bytes_read, &issue, "graph.alias-decoder.cpu",
            "alias decoder", run->err);
    if (rc != YVEX_OK && run->failure && !run->failure->code) {
        run->failure->code = issue.code ? issue.code + 2u
                                        : YVEX_COMPONENT_EXECUTION_TENSOR_CONTRACT;
        run->failure->expected = issue.expected;
        run->failure->actual = issue.actual;
        run->failure->reason = issue.reason ? issue.reason : yvex_error_message(run->err);
        yvex_core_text_copy(run->failure->tensor_name,
                            sizeof(run->failure->tensor_name), name);
    }
    return rc;
}

static int alias_cpu_convolution(
    alias_cpu_run *run, yvex_alias_decoder_weight_role weight_role,
    yvex_alias_decoder_weight_role gain_role, yvex_alias_decoder_weight_role bias_role,
    int normalized, int biased, unsigned long long stage, unsigned long long block,
    unsigned long long layer, const yvex_graph_conv1d_geometry *geometry,
    const float *input, unsigned long long input_count,
    float *output, unsigned long long output_count)
{
    yvex_component_f32_buffer weight = {0}, gain = {0}, bias = {0};
    unsigned long long weight_dims[3], gain_dims[3], bias_dims[1];
    int rc = alias_cpu_cancel(run);
    weight_dims[0] = geometry->transposed ? geometry->input_channels
                                          : geometry->output_channels;
    weight_dims[1] = geometry->transposed ? geometry->output_channels
                                          : geometry->input_channels;
    weight_dims[2] = geometry->kernel_size;
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, weight_role, stage, block, layer, 3u,
                            weight_dims, &weight);
    if (normalized && rc == YVEX_OK) {
        gain_dims[0] = geometry->transposed ? geometry->input_channels
                                            : geometry->output_channels;
        gain_dims[1] = gain_dims[2] = 1ull;
        rc = alias_cpu_load(run, gain_role, stage, block, layer, 3u, gain_dims, &gain);
    }
    if (biased && rc == YVEX_OK) {
        bias_dims[0] = geometry->output_channels;
        rc = alias_cpu_load(run, bias_role, stage, block, layer, 1u, bias_dims, &bias);
    }
    if (rc == YVEX_OK)
        rc = yvex_graph_conv1d_f32(
            geometry, input, input_count, weight.data, weight.count,
            bias.data, bias.count, gain.data, gain.count, output, output_count, run->err);
    if (rc != YVEX_OK && run->failure && !run->failure->code) {
        run->failure->code = YVEX_COMPONENT_EXECUTION_NUMERIC;
        run->failure->reason = yvex_error_message(run->err);
    }
    alias_cpu_buffer_close(run, &bias);
    alias_cpu_buffer_close(run, &gain);
    alias_cpu_buffer_close(run, &weight);
    return rc;
}

static int alias_cpu_activation(
    alias_cpu_run *run, yvex_alias_decoder_weight_role alpha_role,
    yvex_alias_decoder_weight_role beta_role, yvex_alias_decoder_weight_role up_role,
    yvex_alias_decoder_weight_role down_role, unsigned long long stage,
    unsigned long long block, unsigned long long layer, const float *input,
    unsigned long long batch, unsigned long long channels, unsigned long long length,
    float *output, float *scratch, unsigned long long scratch_count)
{
    yvex_component_f32_buffer alpha = {0}, beta = {0}, up = {0}, down = {0};
    unsigned long long channel_dims[1] = {channels};
    unsigned long long filter_dims[3] = {1ull, 1ull, 12ull};
    int rc = alias_cpu_cancel(run);
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, alpha_role, stage, block, layer, 1u,
                            channel_dims, &alpha);
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, beta_role, stage, block, layer, 1u,
                            channel_dims, &beta);
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, up_role, stage, block, layer, 3u, filter_dims, &up);
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, down_role, stage, block, layer, 3u, filter_dims, &down);
    if (rc == YVEX_OK)
        rc = yvex_graph_alias_snake_f32(
            input, batch, channels, length, alpha.data, beta.data, up.data, down.data,
            output, scratch, scratch_count, run->err);
    if (rc != YVEX_OK && run->failure && !run->failure->code) {
        run->failure->code = YVEX_COMPONENT_EXECUTION_NUMERIC;
        run->failure->reason = yvex_error_message(run->err);
    }
    alias_cpu_buffer_close(run, &down);
    alias_cpu_buffer_close(run, &up);
    alias_cpu_buffer_close(run, &beta);
    alias_cpu_buffer_close(run, &alpha);
    return rc;
}

static int alias_cpu_residual(
    alias_cpu_run *run, unsigned long long stage, unsigned long long block,
    const float *input, unsigned long long batch, unsigned long long channels,
    unsigned long long length, float *output)
{
    const yvex_alias_decoder_recipe *recipe = run->request->recipe;
    yvex_component_f32_buffer activation = {0}, convolution = {0}, scratch = {0};
    yvex_graph_conv1d_geometry geometry = {0};
    unsigned long long values, layer, index;
    int rc;
    if (!yvex_core_u64_mul(batch, channels, &values) ||
        !yvex_core_u64_mul(values, length, &values))
        return alias_cpu_refuse(run, YVEX_COMPONENT_EXECUTION_BUDGET, NULL, 1ull, 0ull,
                                YVEX_ERR_BOUNDS, "alias residual extent overflowed");
    rc = alias_cpu_buffer_open(run, &activation, values);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(run, &convolution, values);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(run, &scratch, length * 2ull);
    if (rc != YVEX_OK) goto cleanup;
    memcpy(output, input, (size_t)values * sizeof(float));
    geometry = (yvex_graph_conv1d_geometry){
        batch, channels, channels, length, recipe->residual_kernels[block],
        1ull, 1ull, 0ull, 0ull, 0};
    for (layer = 0ull; layer < recipe->residual_layers && rc == YVEX_OK; ++layer) {
        rc = alias_cpu_activation(
            run, YVEX_ALIAS_DECODER_ACT_ALPHA, YVEX_ALIAS_DECODER_ACT_BETA,
            YVEX_ALIAS_DECODER_ACT_UP_FILTER, YVEX_ALIAS_DECODER_ACT_DOWN_FILTER,
            stage, block, layer * 2ull, output, batch, channels, length,
            activation.data, scratch.data, scratch.count);
        geometry.dilation = recipe->residual_dilations[layer];
        geometry.padding = ((geometry.kernel_size - 1ull) * geometry.dilation) / 2ull;
        if (rc == YVEX_OK)
            rc = alias_cpu_convolution(
                run, YVEX_ALIAS_DECODER_RES1_WEIGHT, YVEX_ALIAS_DECODER_RES1_GAIN,
                YVEX_ALIAS_DECODER_RES1_BIAS, 1, 1, stage, block, layer, &geometry,
                activation.data, activation.count, convolution.data, convolution.count);
        if (rc == YVEX_OK)
            rc = alias_cpu_activation(
                run, YVEX_ALIAS_DECODER_ACT_ALPHA, YVEX_ALIAS_DECODER_ACT_BETA,
                YVEX_ALIAS_DECODER_ACT_UP_FILTER, YVEX_ALIAS_DECODER_ACT_DOWN_FILTER,
                stage, block, layer * 2ull + 1ull, convolution.data, batch, channels,
                length, activation.data, scratch.data, scratch.count);
        geometry.dilation = 1ull;
        geometry.padding = (geometry.kernel_size - 1ull) / 2ull;
        if (rc == YVEX_OK)
            rc = alias_cpu_convolution(
                run, YVEX_ALIAS_DECODER_RES2_WEIGHT, YVEX_ALIAS_DECODER_RES2_GAIN,
                YVEX_ALIAS_DECODER_RES2_BIAS, 1, 1, stage, block, layer, &geometry,
                activation.data, activation.count, convolution.data, convolution.count);
        if (rc == YVEX_OK)
            for (index = 0ull; index < values; ++index) output[index] += convolution.data[index];
    }
cleanup:
    alias_cpu_buffer_close(run, &scratch);
    alias_cpu_buffer_close(run, &convolution);
    alias_cpu_buffer_close(run, &activation);
    return rc;
}

static int alias_cpu_stage(alias_cpu_run *run, unsigned long long stage,
                           const yvex_component_f32_buffer *input,
                           unsigned long long batch, unsigned long long input_channels,
                           unsigned long long input_length,
                           yvex_component_f32_buffer *output,
                           unsigned long long *output_channels,
                           unsigned long long *output_length)
{
    const yvex_alias_decoder_recipe *recipe = run->request->recipe;
    yvex_component_f32_buffer upsampled = {0}, sum = {0}, block = {0};
    yvex_graph_conv1d_geometry geometry = {0};
    unsigned long long values, residual, index;
    int rc;
    *output_channels = input_channels / 2ull;
    geometry = (yvex_graph_conv1d_geometry){
        batch, input_channels, *output_channels, input_length,
        recipe->upsample_kernels[stage], recipe->rates[stage], 1ull,
        (recipe->upsample_kernels[stage] - recipe->rates[stage]) / 2ull, 0ull, 1};
    rc = yvex_graph_conv1d_output_length(&geometry, output_length, run->err);
    if (rc != YVEX_OK || !yvex_core_u64_mul(batch, *output_channels, &values) ||
        !yvex_core_u64_mul(values, *output_length, &values))
        return alias_cpu_refuse(run, YVEX_COMPONENT_EXECUTION_BUDGET, NULL, 1ull, 0ull,
                                YVEX_ERR_BOUNDS, "alias decoder stage extent overflowed");
    rc = alias_cpu_buffer_open(run, &upsampled, values);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(run, &sum, values);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(run, &block, values);
    if (rc == YVEX_OK)
        rc = alias_cpu_convolution(
            run, YVEX_ALIAS_DECODER_UP_WEIGHT, YVEX_ALIAS_DECODER_UP_GAIN,
            YVEX_ALIAS_DECODER_UP_BIAS, 1, 1, stage, 0ull, 0ull, &geometry,
            input->data, input->count, upsampled.data, upsampled.count);
    if (rc == YVEX_OK) memset(sum.data, 0, (size_t)sum.count * sizeof(float));
    for (residual = 0ull; residual < recipe->residual_blocks && rc == YVEX_OK; ++residual) {
        rc = alias_cpu_residual(run, stage, residual, upsampled.data, batch,
                                *output_channels, *output_length, block.data);
        if (rc == YVEX_OK)
            for (index = 0ull; index < values; ++index) sum.data[index] += block.data[index];
    }
    if (rc == YVEX_OK) {
        for (index = 0ull; index < values; ++index)
            sum.data[index] /= (float)recipe->residual_blocks;
        *output = sum;
        memset(&sum, 0, sizeof(sum));
    }
    alias_cpu_buffer_close(run, &block);
    alias_cpu_buffer_close(run, &sum);
    alias_cpu_buffer_close(run, &upsampled);
    return rc;
}

static int alias_cpu_recipe_valid(const yvex_alias_decoder_request *request,
                                  unsigned long long *output_length)
{
    const yvex_alias_decoder_recipe *recipe = request ? request->recipe : NULL;
    unsigned long long channels, length, values, stage, index;
    if (!recipe || !request->input || !request->output || !request->batch ||
        !request->input_length || !request->maximum_workspace_bytes ||
        !recipe->input_channels || !recipe->projection_channels || !recipe->input_kernel ||
        !recipe->pre_channels || !recipe->pre_kernel || !recipe->stage_count ||
        recipe->stage_count > YVEX_ALIAS_DECODER_MAX_STAGES || !recipe->residual_blocks ||
        recipe->residual_blocks > YVEX_ALIAS_DECODER_MAX_RESBLOCKS ||
        !recipe->residual_layers || recipe->residual_layers > YVEX_ALIAS_DECODER_MAX_LAYERS ||
        !recipe->final_channels || !recipe->final_kernel || !request->weight_name ||
        !yvex_core_u64_mul(request->batch, recipe->input_channels, &values) ||
        !yvex_core_u64_mul(values, request->input_length, &values) ||
        values != request->input_count) return 0;
    channels = recipe->pre_channels;
    length = request->input_length;
    for (stage = 0ull; stage < recipe->stage_count; ++stage) {
        unsigned long long rate = recipe->rates[stage];
        unsigned long long kernel = recipe->upsample_kernels[stage];
        if (!rate || kernel < rate || ((kernel - rate) & 1ull) || channels < 2ull ||
            length > (ULLONG_MAX - kernel) / rate) return 0;
        length = (length - 1ull) * rate + kernel - (kernel - rate);
        channels /= 2ull;
    }
    for (index = 0ull; index < recipe->residual_blocks; ++index)
        if (!recipe->residual_kernels[index] || !(recipe->residual_kernels[index] & 1ull))
            return 0;
    for (index = 0ull; index < recipe->residual_layers; ++index)
        if (!recipe->residual_dilations[index]) return 0;
    if (!yvex_core_u64_mul(request->batch, recipe->final_channels, &values) ||
        !yvex_core_u64_mul(values, length, &values) || request->output_capacity < values)
        return 0;
    *output_length = length;
    return 1;
}

int yvex_runtime_alias_decoder_execute_cpu(
    yvex_materialization_session *session, const yvex_alias_decoder_request *request,
    yvex_alias_decoder_result *result, yvex_component_execution_failure *failure,
    yvex_error *err)
{
    alias_cpu_run run = {session, request, result, failure, 0ull, err};
    const yvex_alias_decoder_recipe *recipe;
    yvex_component_f32_buffer current = {0}, next = {0}, activated = {0}, scratch = {0};
    yvex_graph_conv1d_geometry geometry = {0};
    unsigned long long output_length = 0ull, channels, length, values, stage, index;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!session || !result || !alias_cpu_recipe_valid(request, &output_length))
        return alias_cpu_refuse(&run, YVEX_COMPONENT_EXECUTION_INVALID_ARGUMENT, NULL,
                                1ull, 0ull, YVEX_ERR_INVALID_ARG,
                                "one complete bounded alias decoder request is required");
    recipe = request->recipe;
    channels = recipe->projection_channels;
    length = request->input_length;
    rc = alias_cpu_buffer_open(&run, &current, request->batch * channels * length);
    geometry = (yvex_graph_conv1d_geometry){
        request->batch, recipe->input_channels, channels, length,
        recipe->input_kernel, 1ull, 1ull, 0ull, 0ull, 0};
    if (rc == YVEX_OK)
        rc = alias_cpu_convolution(
            &run, YVEX_ALIAS_DECODER_INPUT_WEIGHT, YVEX_ALIAS_DECODER_INPUT_WEIGHT,
            YVEX_ALIAS_DECODER_INPUT_BIAS, 0, 1, 0ull, 0ull, 0ull, &geometry,
            request->input, request->input_count, current.data, current.count);
    geometry.input_channels = channels;
    geometry.output_channels = recipe->pre_channels;
    geometry.kernel_size = recipe->pre_kernel;
    geometry.padding = (recipe->pre_kernel - 1ull) / 2ull;
    channels = recipe->pre_channels;
    if (rc == YVEX_OK)
        rc = alias_cpu_buffer_open(&run, &next, request->batch * channels * length);
    if (rc == YVEX_OK)
        rc = alias_cpu_convolution(
            &run, YVEX_ALIAS_DECODER_PRE_WEIGHT, YVEX_ALIAS_DECODER_PRE_GAIN,
            YVEX_ALIAS_DECODER_PRE_BIAS, 1, 1, 0ull, 0ull, 0ull, &geometry,
            current.data, current.count, next.data, next.count);
    alias_cpu_buffer_close(&run, &current);
    current = next;
    memset(&next, 0, sizeof(next));
    for (stage = 0ull; stage < recipe->stage_count && rc == YVEX_OK; ++stage) {
        unsigned long long next_channels = 0ull, next_length = 0ull;
        rc = alias_cpu_stage(&run, stage, &current, request->batch, channels, length,
                             &next, &next_channels, &next_length);
        if (rc == YVEX_OK) {
            alias_cpu_buffer_close(&run, &current);
            current = next;
            memset(&next, 0, sizeof(next));
            channels = next_channels;
            length = next_length;
        }
    }
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(&run, &activated, current.count);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(&run, &scratch, length * 2ull);
    if (rc == YVEX_OK)
        rc = alias_cpu_activation(
            &run, YVEX_ALIAS_DECODER_POST_ACT_ALPHA, YVEX_ALIAS_DECODER_POST_ACT_BETA,
            YVEX_ALIAS_DECODER_POST_ACT_UP_FILTER, YVEX_ALIAS_DECODER_POST_ACT_DOWN_FILTER,
            0ull, 0ull, 0ull, current.data, request->batch, channels, length,
            activated.data, scratch.data, scratch.count);
    geometry = (yvex_graph_conv1d_geometry){
        request->batch, channels, recipe->final_channels, length, recipe->final_kernel,
        1ull, 1ull, (recipe->final_kernel - 1ull) / 2ull, 0ull, 0};
    if (rc == YVEX_OK)
        rc = alias_cpu_convolution(
            &run, YVEX_ALIAS_DECODER_POST_WEIGHT, YVEX_ALIAS_DECODER_POST_GAIN,
            YVEX_ALIAS_DECODER_POST_GAIN, 1, 0, 0ull, 0ull, 0ull, &geometry,
            activated.data, activated.count, request->output,
            request->batch * recipe->final_channels * output_length);
    values = request->batch * recipe->final_channels * output_length;
    if (rc == YVEX_OK)
        for (index = 0ull; index < values; ++index)
            request->output[index] = fmaxf(-1.0f, fminf(1.0f, request->output[index]));
    if (rc == YVEX_OK) {
        result->output_length = output_length;
        result->output_values = values;
        result->complete = 1;
        yvex_error_clear(err);
    }
    alias_cpu_buffer_close(&run, &scratch);
    alias_cpu_buffer_close(&run, &activated);
    alias_cpu_buffer_close(&run, &next);
    alias_cpu_buffer_close(&run, &current);
    return rc;
}
