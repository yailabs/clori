/*
 * Own bounded component workspace and immutable F32 tensor ingress independently of one model
 * family.  Family recipes retain tensor names, shapes, schedules, and failure projection.
 */
#include "src/graph/private.h"

static int component_fail(yvex_graph_component_failure *failure,
                          yvex_graph_component_failure_code code, const char *tensor_name,
                          unsigned long long expected, unsigned long long actual,
                          yvex_status status, const char *reason, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
        if (tensor_name)
            yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), tensor_name);
    }
    yvex_error_set(err, status, "graph.component", reason);
    return status;
}

static int component_buffer_open(yvex_graph_component_workspace *workspace,
                                 unsigned long long count,
                                 yvex_graph_component_buffer *buffer,
                                 yvex_graph_component_failure *failure, yvex_error *err)
{
    unsigned long long bytes, next_live;

    if (buffer)
        memset(buffer, 0, sizeof(*buffer));
    if (!workspace || !buffer || !count || !workspace->maximum_bytes ||
        !yvex_core_u64_mul(count, sizeof(float), &bytes) || bytes > (unsigned long long)SIZE_MAX ||
        !yvex_core_u64_add(workspace->live_bytes, bytes, &next_live))
        return component_fail(failure, YVEX_GRAPH_COMPONENT_FAILURE_BUDGET, NULL, 1ull, count,
                              YVEX_ERR_BOUNDS, "component workspace extent overflowed", err);
    if (next_live > workspace->maximum_bytes)
        return component_fail(failure, YVEX_GRAPH_COMPONENT_FAILURE_BUDGET, NULL,
                              workspace->maximum_bytes, next_live, YVEX_ERR_BOUNDS,
                              "component workspace budget was exceeded", err);
    buffer->data = (float *)malloc((size_t)bytes);
    if (!buffer->data)
        return component_fail(failure, YVEX_GRAPH_COMPONENT_FAILURE_BUDGET, NULL, bytes, 0ull,
                              YVEX_ERR_NOMEM, "component workspace allocation failed", err);
    buffer->count = count;
    workspace->live_bytes = next_live;
    if (next_live > workspace->peak_bytes)
        workspace->peak_bytes = next_live;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

static void component_buffer_close(yvex_graph_component_workspace *workspace,
                                   yvex_graph_component_buffer *buffer)
{
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
    yvex_materialization_session *session, const char *name, unsigned int rank,
    const unsigned long long *dims, yvex_graph_component_workspace *workspace,
    yvex_graph_component_buffer *buffer, unsigned long long *payload_bytes_read,
    yvex_graph_component_failure *failure, yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding;
    yvex_materialization_failure materialization_failure;
    unsigned long long count = 1ull, expected_bytes = 0ull;
    unsigned int dimension;
    int rc;

    if (!session || !name || !name[0] || !rank || !dims || !workspace || !buffer ||
        !payload_bytes_read)
        return component_fail(failure, YVEX_GRAPH_COMPONENT_FAILURE_INVALID_ARGUMENT, name, 1ull,
                              0ull, YVEX_ERR_INVALID_ARG,
                              "component tensor load requires complete typed inputs", err);
    binding = component_binding_find(session, name);
    if (!binding)
        return component_fail(failure, YVEX_GRAPH_COMPONENT_FAILURE_MISSING_TENSOR, name, 1ull,
                              0ull, YVEX_ERR_FORMAT, "component execution tensor is missing", err);
    if (binding->qtype != YVEX_GGUF_QTYPE_F32 || binding->rank != rank)
        return component_fail(failure, YVEX_GRAPH_COMPONENT_FAILURE_TENSOR_CONTRACT, name, rank,
                              binding->rank, YVEX_ERR_FORMAT,
                              "component tensor rank or dtype differs", err);
    for (dimension = 0u; dimension < rank; ++dimension)
        if (binding->dims[dimension] != dims[dimension] ||
            !yvex_core_u64_mul(count, dims[dimension], &count))
            return component_fail(failure, YVEX_GRAPH_COMPONENT_FAILURE_TENSOR_CONTRACT, name,
                                  dims[dimension], binding->dims[dimension], YVEX_ERR_FORMAT,
                                  "component tensor shape differs", err);
    if (!yvex_core_u64_mul(count, sizeof(float), &expected_bytes) ||
        binding->encoded_bytes != expected_bytes)
        return component_fail(failure, YVEX_GRAPH_COMPONENT_FAILURE_TENSOR_CONTRACT, name,
                              expected_bytes, binding->encoded_bytes, YVEX_ERR_FORMAT,
                              "component tensor byte extent differs", err);
    rc = component_buffer_open(workspace, count, buffer, failure, err);
    if (rc != YVEX_OK)
        return rc;
    memset(&materialization_failure, 0, sizeof(materialization_failure));
    rc = yvex_materialization_session_read(session, binding, 0ull, buffer->data,
                                           (size_t)binding->encoded_bytes,
                                           &materialization_failure, err);
    if (rc != YVEX_OK) {
        component_buffer_close(workspace, buffer);
        return component_fail(failure, YVEX_GRAPH_COMPONENT_FAILURE_MATERIALIZATION, name,
                              binding->encoded_bytes, materialization_failure.actual,
                              (yvex_status)rc,
                              materialization_failure.reason ? materialization_failure.reason
                                                             : "component tensor read failed",
                              err);
    }
    *payload_bytes_read += binding->encoded_bytes;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
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
        component_buffer_open, component_buffer_close, component_binding_find,
        component_tensor_load_f32, rectified_flow_step};
    return &api;
}
