/* Share checked component buffers and bindings without importing family policy. */
#include <yvex/internal/component.h>

#include <math.h>
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
    unsigned long long workspace_bytes;
    yvex_runtime_residency_summary summary;
};

static const yvex_materialized_tensor_binding *component_binding_find(
    const yvex_materialization_session *, const char *);
static int component_weight_bind(
    const yvex_materialization_session *, const yvex_runtime_residency *,
    const char *, yvex_component_encoded_weight *, yvex_error *);

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
                       "one bounded CUDA component workspace is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (session->workspace) {
        if (session->workspace_bytes == bytes) {
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

yvex_materialization_session *yvex_runtime_component_session_materialization(
    const yvex_runtime_component_session *session)
{
    return session ? session->materialization : NULL;
}

int yvex_runtime_component_weight_view(
    const yvex_runtime_component_session *session, const char *name,
    yvex_component_encoded_weight *weight, yvex_error *err)
{
    if (!session || !name || !weight || !session->materialization ||
        !session->residency || !session->summary.sealed ||
        session->summary.invalidated) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.weight-view",
                       "one sealed component session and weight name are required");
        return YVEX_ERR_INVALID_ARG;
    }
    return component_weight_bind(session->materialization, session->residency,
                                 name, weight, err);
}

yvex_backend *yvex_runtime_component_session_backend(
    const yvex_runtime_component_session *session)
{
    return session && session->summary.sealed && !session->summary.invalidated
               ? session->backend : NULL;
}

static const yvex_runtime_residency *component_session_residency(
    const yvex_runtime_component_session *session)
{
    return session ? session->residency : NULL;
}

const yvex_runtime_residency_summary *yvex_runtime_component_session_summary(
    const yvex_runtime_component_session *session)
{
    return session ? &session->summary : NULL;
}

static int component_text_weight_bind(
    const yvex_runtime_component_session *session, const char *name,
    yvex_backend_text_weight *weight, yvex_error *err)
{
    yvex_component_encoded_weight component = {0};
    int rc = component_weight_bind(
        session ? session->materialization : NULL, session ? session->residency : NULL,
        name, &component, err);
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
    const yvex_runtime_component_session *session, const yvex_component_text_request *request,
    yvex_backend_text_weight *weights, yvex_error *err)
{
    unsigned long long layer, slot, index;
    int rc = component_text_weight_bind(session, request->embedding_weight_name, weights, err);
    for (layer = 0ull; rc == YVEX_OK && layer < request->layer_count; ++layer) {
        for (slot = 0ull; rc == YVEX_OK &&
                            slot < YVEX_COMPONENT_TEXT_LAYER_WEIGHT_COUNT; ++slot) {
            char name[256] = {0};
            index = 1ull + layer * YVEX_COMPONENT_TEXT_LAYER_WEIGHT_COUNT + slot;
            rc = request->layer_weight_name(
                request->weight_name_context, layer, (unsigned int)slot, name, err);
            if (rc == YVEX_OK)
                rc = component_text_weight_bind(session, name, weights + index, err);
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

int yvex_runtime_component_text_execute(
    const yvex_runtime_component_session *session, const yvex_component_text_request *request,
    yvex_runtime_av_conditioning_result *result, yvex_error *err)
{
    const yvex_backend_component_operations *operations = NULL;
    const yvex_materialized_tensor_binding *embedding = NULL;
    yvex_backend_text_weight *weights = NULL;
    yvex_backend_text_execution_result backend_result = {0};
    const unsigned char *encoded = NULL;
    unsigned long long encoded_bytes = 0ull, output_values, output_bytes, weight_count = 0ull;
    float *staged = NULL;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    rc = component_text_request_validate(request, result, &output_values, &output_bytes, err);
    if (rc != YVEX_OK) return rc;
    if (!session || !session->materialization || !session->residency || !session->backend ||
        !session->summary.sealed || session->summary.invalidated) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.text",
                       "one sealed component execution session is required");
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
    operations = yvex_backend_component_operations_get(session->backend);
    if (!operations ||
        (!request->layer_count && !operations->text_embedding_execute) ||
        (request->layer_count && !operations->text_encoder_execute)) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "runtime.component.text",
                       "the selected backend lacks admitted text-component execution");
        rc = YVEX_ERR_UNSUPPORTED;
    }
    embedding = rc == YVEX_OK
                    ? component_binding_find(
                          session->materialization, request->embedding_weight_name)
                    : NULL;
    if (rc == YVEX_OK && !embedding) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.component.text.embedding",
                       "admitted text component lacks its embedding binding");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK && request->layer_count)
        rc = component_text_weights_bind(session, request, weights, err);
    if (rc == YVEX_OK && !request->layer_count)
        rc = yvex_runtime_residency_binding_view(
            session->residency, embedding, &encoded, &encoded_bytes, err);
    if (rc == YVEX_OK && !request->layer_count)
        rc = operations->text_embedding_execute(
            session->backend, request->recipe, encoded, encoded_bytes, embedding->qtype,
            embedding->row_count, embedding->row_width,
            embedding->encoded_bytes / embedding->row_count,
            session->summary.residency_identity, session->summary.encoded_bytes,
            request->token_ids, request->token_count, staged, output_values,
            &backend_result, err);
    if (rc == YVEX_OK && request->layer_count)
        rc = operations->text_encoder_execute(
            session->backend, request->recipe, weights, request->layer_count,
            session->summary.residency_identity, session->summary.encoded_bytes,
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
        rc = yvex_runtime_component_text_execute(session, request, result, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    return rc;
}

int yvex_runtime_component_vision_cuda(
    const yvex_runtime_component_session *session, const yvex_vision_request *request,
    yvex_vision_result *result, yvex_error *err)
{
    yvex_component_encoded_weight *weights = NULL;
    yvex_backend_vision_request execution = {0};
    unsigned long long count, index = 0ull, layer, slot;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!session || !request || !request->recipe || !request->weight_name || !result ||
        !session->backend || !session->residency ||
        yvex_backend_kind_of(session->backend) != YVEX_BACKEND_KIND_CUDA ||
        !session->summary.sealed || !session->summary.cuda_ready || session->summary.invalidated ||
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
    if (rc == YVEX_OK) rc = component_weight_bind(session->materialization, session->residency, \
                                                   name, weights + index, err); \
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
        execution.residency_identity = session->summary.residency_identity;
        execution.resident_bytes = session->summary.encoded_bytes;
        rc = yvex_backend_vision_execute(session->backend, &execution, result, err);
    }
    free(weights);
    return rc;
}

int yvex_runtime_component_multimodal_text_cuda(
    const yvex_runtime_component_session *session,
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
    if (!session || !request || !request->recipe || !request->embedding_weight_name ||
        !request->layer_weight_name || !request->token_ids || !request->token_count ||
        !request->layer_count || !request->multimodal || !request->output || !result ||
        !session->backend || !session->residency ||
        yvex_backend_kind_of(session->backend) != YVEX_BACKEND_KIND_CUDA ||
        !session->summary.sealed || !session->summary.cuda_ready || session->summary.invalidated ||
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
    operations = yvex_backend_component_operations_get(session->backend);
    if (!operations || !operations->text_encoder_multimodal_execute) {
        free(weights);
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "runtime.component.multimodal-text",
                       "the admitted backend lacks multimodal text execution");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = component_text_weights_bind(session, &binding_request, weights, err);
    if (rc == YVEX_OK)
        rc = operations->text_encoder_multimodal_execute(
            session->backend, request->recipe, weights, request->layer_count,
            session->summary.residency_identity, session->summary.encoded_bytes,
            request->token_ids, request->token_count, request->multimodal,
            request->output, request->output_capacity, &backend_result, err);
    if (rc == YVEX_OK) component_text_result_project(result, &backend_result);
    free(weights);
    return rc;
}

static int component_joint_weight_bind(
    const yvex_runtime_component_session *session, const char *name,
    yvex_transformer_joint_encoded_weight *weight, yvex_error *err)
{
    yvex_component_encoded_weight component = {0};
    int rc = component_weight_bind(
        session ? session->materialization : NULL, session ? session->residency : NULL,
        name, &component, err);
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

int yvex_runtime_component_joint_transformer_execute(
    yvex_runtime_component_session *session, const char *const *external_names,
    unsigned long long external_count, yvex_component_joint_weight_name_fn block_weight_name,
    void *weight_name_context, const yvex_transformer_joint_request *request,
    yvex_transformer_joint_result *result, yvex_error *err)
{
    const yvex_backend_transformer_operations *transformer_operations = NULL;
    const yvex_backend_component_operations *component_operations = NULL;
    yvex_transformer_joint_encoded_weight external[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {{0}};
    yvex_transformer_joint_encoded_weight *blocks = NULL;
    unsigned long long count, index, workspace_bytes = 0ull;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!session || !external_names ||
        external_count != YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT ||
        !block_weight_name || !request || !request->recipe || !result ||
        !request->block_count || request->block_count > request->recipe->block_count ||
        !session->backend || !session->residency || !session->summary.sealed ||
        session->summary.invalidated ||
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
            rc = component_joint_weight_bind(session, external_names[index],
                                             external + index, err);
        }
    }
    for (index = 0ull; rc == YVEX_OK && index < count; ++index) {
        char name[256] = {0};
        unsigned long long block = index / YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT;
        unsigned int slot = (unsigned int)(index % YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT);
        rc = block_weight_name(weight_name_context, block, slot, name, err);
        if (rc == YVEX_OK)
            rc = component_joint_weight_bind(session, name, blocks + index, err);
    }
    if (rc == YVEX_OK) {
        transformer_operations = yvex_backend_transformer_operations_get(session->backend);
        component_operations = yvex_backend_component_operations_get(session->backend);
        if (!transformer_operations || !transformer_operations->gqa_workspace_required ||
            !component_operations || !component_operations->joint_transformer_execute) {
            yvex_error_set(err, YVEX_ERR_UNSUPPORTED,
                           "runtime.component.joint-transformer",
                           "the admitted backend lacks transformer workspace planning");
            rc = YVEX_ERR_UNSUPPORTED;
        }
    }
    if (rc == YVEX_OK)
        rc = transformer_operations->gqa_workspace_required(
            request->packed_rows, request->recipe->attention_heads,
            request->recipe->attention_heads, request->recipe->head_dimension,
            &workspace_bytes, err);
    if (rc == YVEX_OK && workspace_bytes)
        rc = component_session_prepare_workspace(
            session, workspace_bytes, err);
    if (rc == YVEX_OK)
        rc = component_operations->joint_transformer_execute(
            session->backend, external, blocks, session->summary.residency_identity,
            session->summary.encoded_bytes, request, result, err);
    free(blocks);
    return rc;
}

static const yvex_materialized_tensor_binding *component_tensor_find(
    const yvex_runtime_component_session *session, const char *name)
{
    unsigned long long index;
    if (!session || !session->materialization || !name || !name[0]) return NULL;
    for (index = 0ull;; ++index) {
        const yvex_materialized_tensor_binding *binding =
            yvex_materialization_session_tensor_at(session->materialization, index);
        if (!binding || strcmp(binding->name, name) == 0) return binding;
    }
}

static int component_decoder_weight_bind(
    const yvex_runtime_component_session *session, const char *name,
    yvex_transformer_encoded_weight *weight, yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding = component_tensor_find(session, name);
    unsigned long long elements, bytes;
    if (!binding || binding->qtype != YVEX_GGUF_QTYPE_F32 ||
        (binding->rank != 1u && binding->rank != 2u)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.component.dense-decoder.binding",
                       "one exact F32 vector or source-ordered matrix binding is required");
        return YVEX_ERR_FORMAT;
    }
    weight->row_count = binding->rank == 1u ? 1ull : binding->dims[0];
    weight->row_width = binding->rank == 1u ? binding->dims[0] : binding->dims[1];
    if (!weight->row_count || !weight->row_width ||
        !yvex_core_u64_mul(weight->row_count, weight->row_width, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) ||
        bytes != binding->encoded_bytes) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.component.dense-decoder.binding",
                       "resident F32 binding disagrees with its logical source shape");
        return YVEX_ERR_FORMAT;
    }
    if (yvex_runtime_residency_binding_view(
            session->residency, binding, &weight->encoded,
            &weight->encoded_bytes, err) != YVEX_OK)
        return yvex_error_code(err);
    weight->row_bytes = weight->row_width * sizeof(float);
    weight->qtype = binding->qtype;
    return YVEX_OK;
}

int yvex_runtime_component_dense_decoder_execute(
    const yvex_runtime_component_session *session,
    const yvex_transformer_resident_decoder_request *request,
    yvex_transformer_dense_decoder_result *result, yvex_error *err)
{
    const yvex_backend_transformer_operations *operations =
        yvex_backend_transformer_operations_get(session ? session->backend : NULL);
    yvex_transformer_dense_decoder_request execution;
    yvex_transformer_encoded_weight *blocks = NULL;
    yvex_transformer_encoded_weight final_norm = {0}, final_bias = {0};
    yvex_transformer_encoded_weight output_weight = {0}, output_bias = {0};
    unsigned long long weight_count, index;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!session || !request || !result || !request->block_weight_name ||
        !request->final_norm_weight_name || !request->final_norm_bias_name ||
        !request->output_weight_name || !request->output_bias_name ||
        !request->execution.block_count || !session->backend || !session->residency ||
        !operations || !operations->dense_decoder_execute ||
        !session->summary.sealed || session->summary.invalidated ||
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
            rc = component_decoder_weight_bind(session, name, blocks + index, err);
    }
    if (rc == YVEX_OK)
        rc = component_decoder_weight_bind(
            session, request->final_norm_weight_name, &final_norm, err);
    if (rc == YVEX_OK)
        rc = component_decoder_weight_bind(
            session, request->final_norm_bias_name, &final_bias, err);
    if (rc == YVEX_OK)
        rc = component_decoder_weight_bind(
            session, request->output_weight_name, &output_weight, err);
    if (rc == YVEX_OK)
        rc = component_decoder_weight_bind(
            session, request->output_bias_name, &output_bias, err);
    execution = request->execution;
    execution.block_weights = blocks;
    execution.final_norm_weight = &final_norm;
    execution.final_norm_bias = &final_bias;
    execution.output_weight = &output_weight;
    execution.output_bias = &output_bias;
    if (rc == YVEX_OK)
        rc = operations->dense_decoder_execute(
            session->backend, &execution, result, err);
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
    rc = component_weight_bind(
        yvex_runtime_component_session_materialization(session),
        component_session_residency(session), name, weight, err);
    if (rc == YVEX_OK && weight->encoded_bytes != expected_bytes) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "runtime.component.binding",
                        "component tensor %s has %llu bytes, expected %llu",
                        name, weight->encoded_bytes, expected_bytes);
        rc = YVEX_ERR_FORMAT;
    }
    return rc;
}

int yvex_runtime_component_alias_decoder_execute(
    const yvex_runtime_component_session *session,
    const yvex_alias_decoder_request *request,
    yvex_alias_decoder_result *result, yvex_error *err)
{
    const yvex_backend_component_operations *operations =
        yvex_backend_component_operations_get(session ? session->backend : NULL);
    yvex_alias_decoder_request execution;
    if (result) memset(result, 0, sizeof(*result));
    if (!session || !request || !result || !session->backend || !session->residency ||
        !operations || !operations->alias_decoder_execute || !session->summary.sealed ||
        session->summary.invalidated || request->weight_bind ||
        request->weight_bind_context) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.alias-decoder",
                       "one sealed backend component and unbound decoder recipe are required");
        return YVEX_ERR_INVALID_ARG;
    }
    execution = *request;
    execution.weight_bind = component_weight_bind_sized;
    execution.weight_bind_context = (void *)session;
    return operations->alias_decoder_execute(session->backend, &execution, result, err);
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
