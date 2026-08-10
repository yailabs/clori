/* Share checked component buffers and bindings without importing family policy. */
#include <yvex/internal/component.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/runtime.h>

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

const yvex_materialized_tensor_binding *yvex_component_binding_find(
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

int yvex_component_weight_bind(
    const yvex_materialization_session *session,
    const yvex_runtime_residency *residency, const char *name,
    yvex_component_encoded_weight *weight, yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding =
        yvex_component_binding_find(session, name);
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

int yvex_component_weight_bind_sized(
    void *context, const char *name, unsigned long long rows,
    unsigned long long width, yvex_component_encoded_weight *weight, yvex_error *err)
{
    const yvex_runtime_component_session *session =
        (const yvex_runtime_component_session *)context;
    unsigned long long values, expected_bytes;
    int rc;
    if (!session || !weight || !yvex_core_u64_mul(rows, width, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &expected_bytes)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.binding",
                       "component weight binding geometry is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_component_weight_bind(
        yvex_runtime_component_session_materialization(session),
        yvex_runtime_component_session_residency(session), name, weight, err);
    if (rc == YVEX_OK && weight->encoded_bytes != expected_bytes) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "runtime.component.binding",
                        "component tensor %s has %llu bytes, expected %llu",
                        name, weight->encoded_bytes, expected_bytes);
        rc = YVEX_ERR_FORMAT;
    }
    return rc;
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
    binding = yvex_component_binding_find(session, name);
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
