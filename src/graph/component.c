/*
 * Own bounded component workspace and immutable F32 tensor ingress independently of one model
 * family.  Family recipes retain tensor names, shapes, schedules, and failure projection.
 */
#include "src/graph/private.h"

static void component_execution_open(
    yvex_graph_component_execution *execution, yvex_materialization_session *session,
    const yvex_component_execution_request *request, yvex_component_execution_result *result,
    yvex_component_failure *failure, yvex_error *err, const char *failure_where)
{
    if (!execution)
        return;
    memset(execution, 0, sizeof(*execution));
    execution->session = session;
    execution->request = request;
    execution->result = result;
    execution->failure = failure;
    execution->err = err;
    execution->failure_where = failure_where;
    if (request && request->plan)
        execution->workspace.maximum_bytes = request->plan->workspace_bytes;
    if (result)
        memset(result, 0, sizeof(*result));
    if (failure)
        memset(failure, 0, sizeof(*failure));
}
static int component_execution_refuse(
    yvex_graph_component_execution *execution, yvex_component_failure_code code,
    const char *tensor_name, unsigned long long expected, unsigned long long actual,
    yvex_status status, const char *reason)
{
    if (execution && execution->failure) {
        memset(execution->failure, 0, sizeof(*execution->failure));
        execution->failure->code = code;
        execution->failure->expected = expected;
        execution->failure->actual = actual;
        execution->failure->reason = reason;
        if (tensor_name)
            yvex_core_text_copy(execution->failure->tensor_name,
                                sizeof(execution->failure->tensor_name), tensor_name);
    }
    yvex_error_set(execution ? execution->err : NULL, status,
                   execution && execution->failure_where ? execution->failure_where
                                                         : "graph.component.execute",
                   reason);
    return status;
}

static int component_cancel_check(yvex_graph_component_execution *execution,
                                  const char *reason)
{
    if (!execution || !execution->request)
        return component_execution_refuse(
            execution, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT, NULL, 1ull, 0ull,
            YVEX_ERR_INVALID_ARG, "component cancellation requires an execution request");
    if (execution->request->cancelled &&
        execution->request->cancelled(execution->request->cancellation_context))
        return component_execution_refuse(execution, YVEX_COMPONENT_FAILURE_CANCELLED, NULL,
                                          0ull, 1ull, YVEX_ERR_CANCELLED, reason);
    return YVEX_OK;
}

static int component_buffer_open(yvex_graph_component_execution *execution,
                                 unsigned long long count,
                                 yvex_graph_component_buffer *buffer)
{
    yvex_graph_component_workspace *workspace = execution ? &execution->workspace : NULL;
    unsigned long long bytes, next_live;

    if (buffer)
        memset(buffer, 0, sizeof(*buffer));
    if (!workspace || !buffer || !count || !workspace->maximum_bytes ||
        !yvex_core_u64_mul(count, sizeof(float), &bytes) || bytes > (unsigned long long)SIZE_MAX ||
        !yvex_core_u64_add(workspace->live_bytes, bytes, &next_live))
        return component_execution_refuse(execution, YVEX_COMPONENT_FAILURE_BUDGET, NULL, 1ull,
                                          count, YVEX_ERR_BOUNDS,
                                          "component workspace extent overflowed");
    if (next_live > workspace->maximum_bytes)
        return component_execution_refuse(execution, YVEX_COMPONENT_FAILURE_BUDGET, NULL,
                                          workspace->maximum_bytes, next_live, YVEX_ERR_BOUNDS,
                                          "component workspace budget was exceeded");
    buffer->data = (float *)malloc((size_t)bytes);
    if (!buffer->data)
        return component_execution_refuse(execution, YVEX_COMPONENT_FAILURE_BUDGET, NULL, bytes,
                                          0ull, YVEX_ERR_NOMEM,
                                          "component workspace allocation failed");
    buffer->count = count;
    workspace->live_bytes = next_live;
    if (next_live > workspace->peak_bytes)
        workspace->peak_bytes = next_live;
    if (execution->result)
        execution->result->peak_workspace_bytes = workspace->peak_bytes;
    yvex_error_clear(execution->err);
    return YVEX_OK;
}

static void component_buffer_close(yvex_graph_component_execution *execution,
                                   yvex_graph_component_buffer *buffer)
{
    yvex_graph_component_workspace *workspace = execution ? &execution->workspace : NULL;
    unsigned long long bytes;

    if (!buffer)
        return;
    bytes = buffer->count * sizeof(float);
    if (workspace && bytes <= workspace->live_bytes)
        workspace->live_bytes -= bytes;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

static const yvex_materialized_tensor_binding *component_binding_find(
    const yvex_materialization_session *session, const char *name)
{
    unsigned long long index;

    for (index = 0ull;; ++index) {
        const yvex_materialized_tensor_binding *binding =
            yvex_materialization_session_tensor_at(session, index);
        if (!binding)
            return NULL;
        if (strcmp(binding->name, name) == 0)
            return binding;
    }
}

static int component_tensor_load_f32(
    yvex_graph_component_execution *execution, const char *name, unsigned int rank,
    const unsigned long long *dims, yvex_graph_component_buffer *buffer)
{
    const yvex_materialized_tensor_binding *binding;
    yvex_materialization_failure materialization_failure;
    unsigned long long count = 1ull, expected_bytes = 0ull;
    unsigned int dimension;
    int rc;

    if (!execution || !execution->session || !execution->result || !name || !name[0] || !rank ||
        !dims || !buffer)
        return component_execution_refuse(
            execution, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT, name, 1ull, 0ull,
            YVEX_ERR_INVALID_ARG, "component tensor load requires complete typed inputs");
    binding = component_binding_find(execution->session, name);
    if (!binding)
        return component_execution_refuse(execution, YVEX_COMPONENT_FAILURE_MISSING_TENSOR, name,
                                          1ull, 0ull, YVEX_ERR_FORMAT,
                                          "component execution tensor is missing");
    if (binding->qtype != YVEX_GGUF_QTYPE_F32 || binding->rank != rank)
        return component_execution_refuse(execution, YVEX_COMPONENT_FAILURE_TENSOR_CONTRACT,
                                          name, rank, binding->rank, YVEX_ERR_FORMAT,
                                          "component tensor rank or dtype differs");
    for (dimension = 0u; dimension < rank; ++dimension)
        if (binding->dims[dimension] != dims[dimension] ||
            !yvex_core_u64_mul(count, dims[dimension], &count))
            return component_execution_refuse(execution, YVEX_COMPONENT_FAILURE_TENSOR_CONTRACT,
                                              name, dims[dimension], binding->dims[dimension],
                                              YVEX_ERR_FORMAT, "component tensor shape differs");
    if (!yvex_core_u64_mul(count, sizeof(float), &expected_bytes) ||
        binding->encoded_bytes != expected_bytes)
        return component_execution_refuse(execution, YVEX_COMPONENT_FAILURE_TENSOR_CONTRACT,
                                          name, expected_bytes, binding->encoded_bytes,
                                          YVEX_ERR_FORMAT, "component tensor byte extent differs");
    rc = component_buffer_open(execution, count, buffer);
    if (rc != YVEX_OK)
        return rc;
    memset(&materialization_failure, 0, sizeof(materialization_failure));
    rc = yvex_materialization_session_read(execution->session, binding, 0ull, buffer->data,
                                           (size_t)binding->encoded_bytes,
                                           &materialization_failure, execution->err);
    if (rc != YVEX_OK) {
        component_buffer_close(execution, buffer);
        return component_execution_refuse(
            execution, YVEX_COMPONENT_FAILURE_MATERIALIZATION, name, binding->encoded_bytes,
            materialization_failure.actual, (yvex_status)rc,
            materialization_failure.reason ? materialization_failure.reason
                                           : "component tensor read failed");
    }
    execution->result->payload_bytes_read += binding->encoded_bytes;
    execution->result->tensor_reads++;
    yvex_error_clear(execution->err);
    return YVEX_OK;
}

static int component_name_build(
    yvex_graph_component_execution *execution, char *output, size_t capacity,
    const char *prefix, const char *suffix, unsigned long long expected, const char *reason)
{
    int written;

    if (!output || !capacity || !prefix || !suffix)
        return component_execution_refuse(execution, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT,
                                          prefix, 1ull, 0ull, YVEX_ERR_INVALID_ARG,
                                          "component tensor name requires bounded inputs");
    written = snprintf(output, capacity, "%s%s", prefix, suffix);
    if (written < 0 || (size_t)written >= capacity)
        return component_execution_refuse(
            execution, YVEX_COMPONENT_FAILURE_TENSOR_CONTRACT, prefix, expected,
            written < 0 ? 0ull : (unsigned long long)written, YVEX_ERR_BOUNDS, reason);
    return YVEX_OK;
}

static int rectified_flow_step(float *output, const float *sample, const float *velocity,
                               unsigned long long values, float timestep, float sigma,
                               float sigma_next, yvex_error *err)
{
    unsigned long long index;

    if (!output || !sample || !velocity || !values || !isfinite(timestep) || !isfinite(sigma) ||
        !isfinite(sigma_next) || timestep < 0.0f || timestep >= 1.0f || sigma <= 0.0f ||
        sigma_next < 0.0f || sigma_next >= sigma) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.rectified_flow",
                       "rectified-flow step requires finite state and a decreasing sigma interval");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0ull; index < values; ++index)
        if (!isfinite(sample[index]) || !isfinite(velocity[index])) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "graph.rectified_flow",
                           "rectified-flow input contains a non-finite value");
            return YVEX_ERR_FORMAT;
        }
    for (index = 0ull; index < values; ++index) {
        float denoised = sample[index] + (1.0f - timestep) * velocity[index];
        float ratio = sigma_next / sigma;
        output[index] = ratio * sample[index] + (1.0f - ratio) * denoised;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

const yvex_graph_component_api *yvex_graph_component_api_get(void)
{
    static const yvex_graph_component_api api = {
        component_execution_open, component_execution_refuse, component_cancel_check,
        component_buffer_open, component_buffer_close, component_binding_find,
        component_tensor_load_f32, component_name_build, rectified_flow_step};
    return &api;
}
