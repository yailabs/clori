/* Publishes one complete finite vocabulary row from typed normalized hidden state. */
#include <yvex/internal/logits.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/execution.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>
#include "src/runtime/private.h"

struct yvex_runtime_logits_plan {
    yvex_runtime_logits_plan_summary summary;
    const yvex_materialized_tensor_binding *binding;
};
struct yvex_runtime_logits_context {
    yvex_model_engine *model;
    yvex_runtime_execution_session *session;
    const yvex_model_engine_view *model_view;
    const yvex_runtime_session_view *session_view;
    yvex_runtime_logits_plan plan;
    yvex_runtime_logits_options options;
    const unsigned char *resident_head;
    unsigned long long resident_head_bytes;
    float *candidate, *host_hidden_rows;
    yvex_device_tensor *device_hidden, *device_logits;
    /* The allocation is max-shaped; only this exact completed prefix may be published. */
    yvex_device_tensor device_logits_publication;
    pthread_mutex_t mutex;
    unsigned long long execution_count;
    char shared_draft_plan_identity[YVEX_SHA256_HEX_CAP];
    int mutex_ready, busy, invalidated, shared_draft_plan_admitted;
};
/*
 * Admit and project logits readiness from the exact immutable output-head residency.
 *
 * Narrows output-head binding readiness to resident implementation truth.
 */
int yvex_runtime_logits_residency_admit(
    yvex_runtime_capabilities *capabilities,
    const yvex_runtime_residency_summary *residency)
{
    if (!capabilities || !residency ||
        (capabilities->output_head_binding_ready && !residency->output_head_complete))
        return 0;
    capabilities->output_head_binding_ready =
        capabilities->output_head_binding_ready && residency->output_head_complete;
    return 1;
}
void yvex_runtime_logits_capabilities_invalidate(yvex_runtime_capabilities *capabilities)
{
    if (!capabilities) return;
    capabilities->output_head_binding_ready = capabilities->output_head_projection_ready =
        capabilities->logits_cpu_ready = capabilities->logits_cuda_ready =
        capabilities->logits_prefill_ready = capabilities->logits_decode_ready =
        capabilities->logits_full_vocabulary_ready = capabilities->logits_hidden_contract_ready =
        capabilities->logits_partial_progress_ready = capabilities->logits_ready = 0;
}
static int logits_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.logits", reason);
    return status;
}
static int logits_device_open(yvex_runtime_logits_context *context,
                              yvex_device_tensor **out, const char *name,
                              unsigned long long elements, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long bytes;
    if (!yvex_core_u64_mul(elements, sizeof(float), &bytes))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "logits device tensor extent overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = elements;
    descriptor.bytes = bytes;
    return yvex_backend_tensor_alloc(context->session_view->backend,
                                     &descriptor, out, err);
}
/*
 * Open one reusable logits context over borrowed model/session owners.
 *
 * Borrows resident head and allocates stable local CPU/CUDA workspaces.
 */
int yvex_runtime_logits_context_open(
    yvex_runtime_logits_context **out, yvex_model_engine *model,
    yvex_runtime_execution_session *session,
    const yvex_transformer_plan *transformer_plan,
    const yvex_runtime_logits_options *options, yvex_error *err)
{
    yvex_runtime_logits_context *context = NULL;
    yvex_model_engine_summary model_summary;
    yvex_runtime_session_summary session_summary;
    yvex_runtime_residency_summary residency;
    unsigned long long candidate_bytes, hidden_elements, hidden_bytes;
    unsigned long long logits_elements, device_elements, device_bytes, device_total, host_bytes;
    int rc;
    if (out) *out = NULL;
    if (!out || !model || !session || !transformer_plan || !options ||
        !options->maximum_rows ||
        (options->device_selection != 0 && options->device_selection != 1) ||
        options->evidence_profile > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        (options->device_selection &&
         (!options->execution_profile ||
          !yvex_sha256_hex_valid(options->execution_profile->identity))))
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "logits requires model, session, transformer plan, and row budget");
    context = (yvex_runtime_logits_context *)calloc(1u, sizeof(*context));
    if (!context) return logits_refuse(err, YVEX_ERR_NOMEM,
                                       "logits context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_model_engine_view_get(model);
    context->session_view = yvex_runtime_session_view_get(session);
    context->options = *options;
    if (!context->model_view || !context->session_view ||
        context->session_view->engine != model ||
        yvex_model_engine_summary_copy(model, &model_summary, err) != YVEX_OK ||
        yvex_runtime_session_summary_copy(session, &session_summary, err) != YVEX_OK ||
        !model_summary.sealed || !model_summary.valid ||
        (options->device_selection && session_summary.backend != YVEX_BACKEND_KIND_CUDA) ||
        (options->execution_profile && !runtime_execution_profile_matches(
             options->execution_profile, &model_summary, &session_summary))) {
        rc = logits_refuse(err, YVEX_ERR_STATE,
                           "logits model/session pairing is invalid");
        goto failure;
    }
    {
        const yvex_transformer_plan_summary *transformer =
            yvex_transformer_plan_summary_get(transformer_plan);
        const yvex_runtime_logits_plan_summary *compiled =
            context->model_view->output_head;
        const yvex_materialized_tensor_binding *binding = compiled
            ? yvex_materialization_session_tensor_at(
                  context->model_view->materialization,
                  compiled->output_head_tensor_id)
            : NULL;
        if (!transformer || !compiled || !binding ||
            strcmp(compiled->transformer_plan_identity,
                   transformer->transformer_plan_identity) != 0 ||
            strcmp(compiled->output_head_plan_identity,
                   context->model_view->binding->output_head_plan_identity) != 0 ||
            binding->tensor_id != compiled->output_head_tensor_id ||
            binding->role != compiled->role || binding->qtype != compiled->qtype ||
            binding->encoded_bytes != compiled->encoded_bytes) {
            rc = logits_refuse(err, YVEX_ERR_STATE,
                               "runtime binding output-head plan is stale");
            goto failure;
        }
        context->plan.summary = *compiled;
        context->plan.binding = binding;
    }
    rc = yvex_runtime_residency_snapshot(
        context->model_view->residency, &residency, NULL, NULL, err);
    if (rc != YVEX_OK || !residency.output_head_complete ||
        !yvex_sha256_hex_valid(residency.output_head_residency_identity)) {
        rc = logits_refuse(err, YVEX_ERR_STATE,
                           "sealed output-head residency is unavailable");
        goto failure;
    }
    rc = yvex_runtime_residency_binding_view(
        context->model_view->residency, context->plan.binding,
        &context->resident_head, &context->resident_head_bytes, err);
    if (rc != YVEX_OK ||
        context->resident_head_bytes != context->plan.summary.encoded_bytes)
        goto failure;
    if (!yvex_core_u64_mul(context->plan.summary.vocabulary_size,
                           sizeof(float), &candidate_bytes) ||
        !yvex_core_u64_mul(options->maximum_rows,
                           context->plan.summary.hidden_width, &hidden_elements) ||
        !yvex_core_u64_mul(hidden_elements, sizeof(float), &hidden_bytes) ||
        (!options->device_selection && candidate_bytes > SIZE_MAX) ||
        hidden_bytes > SIZE_MAX) {
        rc = logits_refuse(err, YVEX_ERR_NOMEM,
                           "logits host workspace extent overflowed");
        goto failure;
    }
    host_bytes = sizeof(*context);
    if ((!options->device_selection &&
         !yvex_core_u64_add(host_bytes, candidate_bytes, &host_bytes)) ||
        (options->maximum_host_bytes && host_bytes > options->maximum_host_bytes)) {
        rc = logits_refuse(err, YVEX_ERR_NOMEM,
                           "logits host workspace exceeds its budget");
        goto failure;
    }
    if (!options->device_selection)
        context->candidate = (float *)malloc((size_t)candidate_bytes);
    if (!options->device_selection && !context->candidate) {
        rc = logits_refuse(err, YVEX_ERR_NOMEM,
                           "logits candidate allocation failed");
        goto failure;
    }
    if (yvex_backend_kind_of(context->session_view->backend) ==
        YVEX_BACKEND_KIND_CUDA) {
        if (!yvex_core_u64_add(host_bytes, hidden_bytes, &host_bytes) ||
            (options->maximum_host_bytes && host_bytes > options->maximum_host_bytes) ||
            !yvex_core_u64_mul(options->maximum_rows,
                               context->plan.summary.vocabulary_size,
                               &logits_elements) ||
            !yvex_core_u64_add(hidden_elements, logits_elements, &device_elements) ||
            !yvex_core_u64_mul(device_elements, sizeof(float), &device_bytes) ||
            !yvex_core_u64_add(residency.device_resident_bytes, device_bytes,
                               &device_total) ||
            (options->maximum_device_bytes && device_total > options->maximum_device_bytes)) {
            rc = logits_refuse(err, YVEX_ERR_NOMEM,
                               "logits CUDA workspace exceeds its budget");
            goto failure;
        }
        context->host_hidden_rows = (float *)malloc((size_t)hidden_bytes);
        if (!context->host_hidden_rows) {
            rc = logits_refuse(err, YVEX_ERR_NOMEM,
                               "logits row-batch staging allocation failed");
            goto failure;
        }
        rc = logits_device_open(context, &context->device_hidden,
                                "logits.hidden", hidden_elements, err);
        if (rc == YVEX_OK)
            rc = logits_device_open(context, &context->device_logits,
                                    "logits.output", logits_elements, err);
        if (rc != YVEX_OK) goto failure;
    }
    if (pthread_mutex_init(&context->mutex, NULL) != 0) {
        rc = logits_refuse(err, YVEX_ERR_STATE,
                           "logits context synchronization initialization failed");
        goto failure;
    }
    context->mutex_ready = 1;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    (void)yvex_runtime_logits_context_close(&context, NULL);
    return rc;
}
const yvex_runtime_logits_plan_summary *yvex_runtime_logits_plan_summary_get(
    const yvex_runtime_logits_context *context)
{
    return context ? &context->plan.summary : NULL;
}
int yvex_runtime_logits_admit_shared_draft_plan(
    yvex_runtime_logits_context *context,
    const yvex_transformer_plan *draft_plan, yvex_error *err)
{
    const yvex_transformer_plan_summary *draft =
        yvex_transformer_plan_summary_get(draft_plan);
    const yvex_speculation_family_policy *policy = NULL;
    if (!context || !draft || !context->model_view ||
        !yvex_runtime_binding_policies(
            context->model_view->compiled_binding, NULL, NULL, &policy) ||
        policy->schema_version != YVEX_SPECULATION_FAMILY_POLICY_SCHEMA_V1 ||
        !policy->shares_output_head || draft->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT ||
        draft->hidden_width != context->plan.summary.hidden_width ||
        draft->vocabulary_size != context->plan.summary.vocabulary_size ||
        strcmp(draft->logical_model_identity,
               context->plan.summary.logical_model_identity) != 0 ||
        strcmp(draft->runtime_numeric_identity,
               context->plan.summary.runtime_numeric_identity) != 0 ||
        strcmp(draft->runtime_descriptor_identity,
               context->plan.summary.runtime_descriptor_identity) != 0)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "draft plan cannot consume the resident shared output head");
    yvex_runtime_identity_copy(context->shared_draft_plan_identity,
                               draft->transformer_plan_identity);
    context->shared_draft_plan_admitted = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
static int logits_source_identity(yvex_runtime_logits_source *source)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!source) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.source.v2") ||
        !yvex_sha256_update_u64(&hash, source->schema_version) ||
        !yvex_sha256_update_u64(&hash, source->source_phase) ||
        !yvex_sha256_update_u64(&hash, source->source_position) ||
        !yvex_sha256_update_u64(&hash, source->row_count) ||
        !yvex_sha256_update_u64(&hash, source->hidden_width) ||
        !yvex_sha256_update_u64(&hash, source->host_values_available) ||
        !yvex_sha256_update_u64(&hash, source->device_values_available) ||
        !yvex_sha256_update_text(&hash, source->runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, source->runtime_binding_identity) ||
        !yvex_sha256_update_text(&hash, source->transformer_plan_identity) ||
        !yvex_sha256_update_text(&hash, source->transformer_execution_identity) ||
        !yvex_sha256_update_text(&hash, source->normalized_hidden_digest) ||
        (source->device_values_available &&
         (!yvex_sha256_update_text(
              &hash, source->device_hidden.execution_profile_identity) ||
          !yvex_sha256_update_u64(&hash, source->device_hidden.resource_generation) ||
          !yvex_sha256_update_u64(&hash, source->device_hidden.session_generation) ||
          !yvex_sha256_update_u64(&hash, source->device_hidden.state_generation) ||
          !yvex_sha256_update_u64(&hash, source->device_hidden.element_offset))) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, source->source_identity);
    return 1;
}
/*
 * Initialize one source from exact borrowed model and plan identities.
 *
 * Live context, phase, position, producer identity, and finite hidden row. Publishes one
 * identity-bound borrowed source view.
 */
static int logits_source_begin(const yvex_runtime_logits_context *context,
                               yvex_runtime_logits_source *source,
                               yvex_logits_source_phase phase,
                               unsigned long long position,
                               const float *hidden,
                               const yvex_execution_device_view *device_hidden,
                               const char *producer_hidden_digest,
                               const char *plan_identity,
                               const char *producer_identity,
                               yvex_error *err)
{
    yvex_model_engine_summary model;
    if (!context || !source || (!hidden == !device_hidden) ||
        !producer_hidden_digest || !producer_identity ||
        !yvex_sha256_hex_valid(producer_hidden_digest) ||
        !yvex_sha256_hex_valid(producer_identity) ||
        (device_hidden &&
         yvex_execution_device_view_validate(device_hidden, err) != YVEX_OK) ||
        yvex_model_engine_summary_copy(context->model, &model, err) != YVEX_OK)
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "logits producer source facts are invalid");
    memset(source, 0, sizeof(*source));
    source->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    source->source_phase = phase;
    source->source_position = position;
    source->row_count = 1ull;
    source->hidden_width = context->plan.summary.hidden_width;
    source->normalized_hidden = hidden;
    source->host_values_available = hidden != NULL;
    source->device_values_available = device_hidden != NULL;
    if (device_hidden) source->device_hidden = *device_hidden;
    yvex_runtime_identity_copy(source->runtime_model_identity,
                               model.runtime_model_identity);
    yvex_runtime_identity_copy(source->runtime_binding_identity,
                               context->model_view->binding->identity);
    yvex_runtime_identity_copy(source->transformer_plan_identity, plan_identity);
    yvex_runtime_identity_copy(source->transformer_execution_identity,
                               producer_identity);
    if (hidden &&
        !yvex_execution_f32_digest("yvex.transformer.normalized-hidden.v1", hidden,
                              source->hidden_width,
                              source->normalized_hidden_digest))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden host row could not be sealed");
    if (device_hidden)
        yvex_runtime_identity_copy(source->normalized_hidden_digest,
                                   producer_hidden_digest);
    if (!logits_source_identity(source) ||
        !yvex_sha256_hex_valid(source->source_identity))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden source identity could not be sealed");
    return YVEX_OK;
}
int yvex_runtime_logits_source_from_transformer(
    const yvex_runtime_logits_context *context,
    yvex_runtime_logits_source *source,
    const yvex_runtime_transformer_result *producer,
    const float *normalized_hidden, unsigned long long hidden_capacity,
    unsigned long long row_ordinal, yvex_error *err)
{
    yvex_execution_device_view device_row;
    char complete_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long complete_values, row_offset, first_device_row;
    if (!context || !source || !producer || !producer->completed ||
        producer->phase != YVEX_TRANSFORMER_PHASE_PREFILL ||
        !producer->token_count || row_ordinal >= producer->token_count ||
        !yvex_core_u64_mul(producer->token_count,
                           context->plan.summary.hidden_width, &complete_values) ||
        !yvex_core_u64_mul(row_ordinal, context->plan.summary.hidden_width,
                           &row_offset))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "Transformer normalized-hidden publication is incompatible");
    if (producer->normalized_hidden_host_available) {
        if (!normalized_hidden || hidden_capacity < complete_values ||
            !yvex_execution_f32_digest(
                "yvex.transformer.normalized-hidden.v1", normalized_hidden,
                complete_values, complete_digest) ||
            strcmp(complete_digest, producer->normalized_hidden_digest) != 0)
            return logits_refuse(
                err, YVEX_ERR_FORMAT,
                "Transformer host hidden publication is incompatible");
        return logits_source_begin(
            context, source, YVEX_LOGITS_SOURCE_PREFILL,
            producer->token_start + row_ordinal, normalized_hidden + row_offset,
            NULL, producer->normalized_hidden_digest,
            context->plan.summary.transformer_plan_identity,
            producer->execution_identity, err);
    }
    if (!producer->normalized_hidden_device_available || normalized_hidden ||
        hidden_capacity || producer->device_hidden.rows > producer->token_count)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "Transformer device hidden publication is incompatible");
    first_device_row = producer->token_count - producer->device_hidden.rows;
    if (row_ordinal < first_device_row) return logits_refuse(
        err, YVEX_ERR_BOUNDS,
        "requested hidden row is no longer resident in the device publication");
    device_row = producer->device_hidden;
    if (!yvex_core_u64_mul(row_ordinal - first_device_row,
                           context->plan.summary.hidden_width, &row_offset) ||
        !yvex_core_u64_add(device_row.element_offset, row_offset,
                           &device_row.element_offset))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "device hidden row offset overflowed");
    device_row.rows = 1ull;
    if (yvex_execution_device_view_validate(&device_row, err) != YVEX_OK)
        return yvex_error_code(err);
    return logits_source_begin(
        context, source, YVEX_LOGITS_SOURCE_PREFILL,
        producer->token_start + row_ordinal, NULL, &device_row,
        producer->normalized_hidden_digest,
        context->plan.summary.transformer_plan_identity,
        producer->execution_identity, err);
}
int yvex_runtime_logits_source_from_decode(
    const yvex_runtime_logits_context *context,
    yvex_runtime_logits_source *source,
    const yvex_runtime_decode_step_result *producer,
    const float *normalized_hidden, unsigned long long hidden_capacity,
    yvex_error *err)
{
    char digest[YVEX_SHA256_HEX_CAP];
    if (!context || !source || !producer || !producer->completed ||
        producer->position_after != producer->position_before + 1ull)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "decode normalized-hidden publication is incompatible");
    if (producer->normalized_hidden_host_available) {
        if (!normalized_hidden ||
            hidden_capacity < context->plan.summary.hidden_width ||
            !yvex_execution_f32_digest(
                "yvex.transformer.normalized-hidden.v1", normalized_hidden,
                context->plan.summary.hidden_width, digest) ||
            strcmp(digest, producer->normalized_hidden_digest) != 0)
            return logits_refuse(
                err, YVEX_ERR_FORMAT,
                "decode host hidden publication is incompatible");
        return logits_source_begin(
            context, source, YVEX_LOGITS_SOURCE_DECODE,
            producer->position_before, normalized_hidden, NULL,
            producer->normalized_hidden_digest,
            context->plan.summary.transformer_plan_identity,
            producer->transformer_execution_identity, err);
    }
    if (!producer->normalized_hidden_device_available || normalized_hidden ||
        hidden_capacity || producer->device_hidden.rows != 1ull ||
        yvex_execution_device_view_validate(&producer->device_hidden, err) != YVEX_OK)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "decode device hidden publication is incompatible");
    return logits_source_begin(
        context, source, YVEX_LOGITS_SOURCE_DECODE,
        producer->position_before, NULL, &producer->device_hidden,
        producer->normalized_hidden_digest,
        context->plan.summary.transformer_plan_identity,
        producer->transformer_execution_identity, err);
}
int yvex_runtime_logits_source_from_draft(
    const yvex_runtime_logits_context *context, yvex_runtime_logits_source *source,
    const yvex_transformer_plan *draft_plan,
    const yvex_runtime_transformer_result *producer,
    const float *normalized_hidden, unsigned long long hidden_capacity,
    unsigned long long row_ordinal, yvex_error *err)
{
    const yvex_transformer_plan_summary *draft =
        yvex_transformer_plan_summary_get(draft_plan);
    yvex_execution_device_view device_row;
    char complete_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long complete_values, row_offset;
    if (!context || !source || !draft || !producer || !producer->completed ||
        producer->phase != YVEX_TRANSFORMER_PHASE_PREFILL ||
        !context->shared_draft_plan_admitted ||
        strcmp(draft->transformer_plan_identity,
               context->shared_draft_plan_identity) != 0 ||
        !producer->token_count || row_ordinal >= producer->token_count ||
        !yvex_core_u64_mul(producer->token_count, draft->hidden_width, &complete_values) ||
        !yvex_core_u64_mul(row_ordinal, draft->hidden_width, &row_offset))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "draft normalized-hidden publication is incompatible");
    if (producer->normalized_hidden_host_available) {
        if (!normalized_hidden || hidden_capacity < complete_values ||
            !yvex_execution_f32_digest("yvex.transformer.normalized-hidden.v1", normalized_hidden,
                                       complete_values, complete_digest) ||
            strcmp(complete_digest, producer->normalized_hidden_digest) != 0)
            return logits_refuse(err, YVEX_ERR_FORMAT,
                                 "draft host hidden publication is incompatible");
        return logits_source_begin(context, source, YVEX_LOGITS_SOURCE_DRAFT,
            producer->token_start + row_ordinal, normalized_hidden + row_offset, NULL,
            producer->normalized_hidden_digest,
            draft->transformer_plan_identity, producer->execution_identity, err);
    }
    if (!producer->normalized_hidden_device_available || normalized_hidden || hidden_capacity ||
        producer->device_hidden.rows != producer->token_count)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "draft device hidden publication is incompatible");
    device_row = producer->device_hidden;
    if (!yvex_core_u64_add(device_row.element_offset, row_offset, &device_row.element_offset))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "draft device hidden row offset overflowed");
    device_row.rows = 1ull;
    if (yvex_execution_device_view_validate(&device_row, err) != YVEX_OK)
        return yvex_error_code(err);
    return logits_source_begin(context, source, YVEX_LOGITS_SOURCE_DRAFT,
        producer->token_start + row_ordinal, NULL, &device_row,
        producer->normalized_hidden_digest, draft->transformer_plan_identity,
        producer->execution_identity, err);
}
/*
 * Validate one normalized-hidden source against exact producing identities.
 * Refuses stale model/plan/producer identity, geometry, or payload digest.
 */
static int logits_source_validate(const yvex_runtime_logits_context *context,
                                  const yvex_runtime_logits_source *source,
                                  yvex_error *err)
{
    yvex_model_engine_summary model;
    yvex_runtime_logits_source canonical;
    char digest[YVEX_SHA256_HEX_CAP];
    if (!context || !source || source->schema_version != YVEX_RUNTIME_LOGITS_SCHEMA_V1 ||
        source->source_phase > YVEX_LOGITS_SOURCE_DRAFT || source->row_count != 1ull ||
        source->hidden_width != context->plan.summary.hidden_width ||
        source->host_values_available + source->device_values_available != 1 ||
        source->host_values_available != (source->normalized_hidden != NULL) ||
        yvex_model_engine_summary_copy(context->model, &model, err) != YVEX_OK ||
        strcmp(source->runtime_model_identity, model.runtime_model_identity) != 0 ||
        strcmp(source->runtime_binding_identity,
               context->model_view->binding->identity) != 0 ||
        ((strcmp(source->transformer_plan_identity,
                 context->plan.summary.transformer_plan_identity) != 0) &&
         (!context->shared_draft_plan_admitted ||
          source->source_phase != YVEX_LOGITS_SOURCE_DRAFT ||
          strcmp(source->transformer_plan_identity,
                 context->shared_draft_plan_identity) != 0)) ||
        !yvex_sha256_hex_valid(source->transformer_execution_identity))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden source identity or geometry is stale");
    if (source->host_values_available &&
        (!yvex_execution_f32_digest("yvex.transformer.normalized-hidden.v1",
                               source->normalized_hidden, source->hidden_width,
                               digest) ||
         strcmp(digest, source->normalized_hidden_digest) != 0))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden host row digest is stale");
    if (source->device_values_available &&
        (source->device_hidden.kind != YVEX_EXECUTION_DEVICE_HIDDEN ||
         source->device_hidden.backend != context->session_view->backend ||
         source->device_hidden.rows != 1ull ||
         source->device_hidden.columns != source->hidden_width ||
         strcmp(source->device_hidden.runtime_model_identity,
                source->runtime_model_identity) != 0 ||
         !context->options.execution_profile ||
         strcmp(source->device_hidden.execution_profile_identity,
                context->options.execution_profile->identity) != 0 ||
         yvex_execution_device_view_validate(&source->device_hidden, err) != YVEX_OK))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden device row is stale");
    canonical = *source;
    canonical.source_identity[0] = '\0';
    if (!logits_source_identity(&canonical) ||
        strcmp(canonical.source_identity, source->source_identity) != 0)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden source identity is not canonical");
    return YVEX_OK;
}
static int logits_project_cpu(yvex_runtime_logits_context *context,
                              const float *hidden, yvex_error *err)
{
    unsigned long long row;
    for (row = 0ull; row < context->plan.summary.row_count; ++row) {
        yvex_quant_failure failure;
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context))
            return logits_refuse(err, YVEX_ERR_CANCELLED,
                                 "logits CPU projection was cancelled");
        memset(&failure, 0, sizeof(failure));
        if (yvex_quant_cpu_dot(
                context->plan.summary.qtype,
                context->resident_head + row * context->plan.summary.row_bytes,
                (size_t)context->plan.summary.row_bytes, hidden,
                context->plan.summary.hidden_width, &context->candidate[row],
                &failure, err) != YVEX_OK)
            return yvex_error_code(err);
        if (!isfinite(context->candidate[row]))
            return logits_refuse(err, YVEX_ERR_FORMAT,
                                 "CPU output-head projection produced non-finite logits");
    }
    return YVEX_OK;
}
/*
 * Execute one full resident output-head projection on CUDA without CPU fallback.
 *
 * CUDA context buffers, encoded resident head, and admitted hidden row. Bounded H2D, device
 * matvec, synchronization, and candidate D2H.
 */
static int logits_project_cuda(yvex_runtime_logits_context *context,
                               const yvex_runtime_logits_source *source,
                               yvex_runtime_logits_row_result *result,
                               yvex_error *err)
{
    unsigned long long hidden_bytes, logits_bytes;
    yvex_backend_cuda_operation_facts facts;
    yvex_device_tensor borrowed_hidden, staging_hidden, logits_view;
    const yvex_device_tensor *device_hidden = context->device_hidden;
    int rc;
    if (!context->device_hidden || !context->device_logits ||
        !yvex_core_u64_mul(context->plan.summary.hidden_width, sizeof(float),
                           &hidden_bytes) ||
        !yvex_core_u64_mul(context->plan.summary.vocabulary_size, sizeof(float),
                           &logits_bytes))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "stable logits CUDA buffers are unavailable");
    if (!yvex_backend_tensor_f32_subview(
            context->device_logits, 0ull,
            context->plan.summary.vocabulary_size, &logits_view))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "one-row device logits view is unavailable");
    if (source->device_values_available) {
        if (!yvex_backend_tensor_f32_subview(
                source->device_hidden.tensor,
                source->device_hidden.element_offset,
                context->plan.summary.hidden_width, &borrowed_hidden))
            return logits_refuse(err, YVEX_ERR_FORMAT,
                                 "device hidden row cannot be borrowed");
        device_hidden = &borrowed_hidden;
        rc = YVEX_OK;
    } else {
        if (!yvex_backend_tensor_f32_subview(
                context->device_hidden, 0ull,
                context->plan.summary.hidden_width, &staging_hidden))
            return logits_refuse(err, YVEX_ERR_BOUNDS,
                                 "one-row hidden staging view is unavailable");
        device_hidden = &staging_hidden;
        rc = yvex_backend_tensor_write(context->session_view->backend,
                                       &staging_hidden,
                                       source->normalized_hidden,
                                       hidden_bytes, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_cuda_encoded_matvec(
            context->session_view->backend, context->resident_head,
            context->resident_head_bytes, context->plan.summary.qtype,
            context->plan.summary.row_count, context->plan.summary.row_width,
            context->plan.summary.row_bytes, 1ull, device_hidden,
            NULL, 0ull, NULL, &logits_view, 0, &facts, err);
    if (rc == YVEX_OK)
        context->device_logits_publication = logits_view;
    if (rc == YVEX_OK && !context->options.device_selection)
        rc = yvex_backend_tensor_read(context->session_view->backend,
                                      &logits_view,
                                      context->candidate, logits_bytes, err);
    if (rc == YVEX_OK) {
        result->h2d_bytes = source->device_values_available ? 0ull : hidden_bytes;
        result->d2h_bytes = facts.d2h_bytes +
                            (context->options.device_selection ? 0ull : logits_bytes);
        result->d2d_bytes = facts.d2d_bytes;
        result->kernel_launches = facts.kernel_launches;
        result->device_synchronizations = facts.device_synchronizations +
                                          !source->device_values_available +
                                          !context->options.device_selection;
        rc = yvex_execution_memory_facts_add(
            &result->memory, facts.active_weight_bytes, facts.state_bytes,
            facts.activation_bytes, facts.temporary_bytes,
            facts.compulsory_memory_facts_available,
            !facts.compulsory_memory_facts_available, err);
    }
    return rc;
}
/*
 * Seal complete row evidence after all vocabulary values are finite.
 *
 * Any non-finite or identity failure leaves the row incomplete.
 */
static int logits_device_view_build(
    yvex_runtime_logits_context *context,
    const yvex_runtime_residency_summary *residency,
    yvex_logits_source_phase source_phase, unsigned long long element_offset,
    yvex_execution_device_view *view, yvex_error *err)
{
    const yvex_attention_state_provider *provider =
        !context ? NULL : source_phase == YVEX_LOGITS_SOURCE_DRAFT
                             ? context->session_view->draft_attention_state_provider
                             : context->session_view->attention_state_provider;
    if (!context || !residency || !residency->generation)
        return logits_refuse(err, YVEX_ERR_STATE,
                             "device logits generations are unavailable");
    return yvex_runtime_device_view_bind(
        view, YVEX_EXECUTION_DEVICE_LOGITS, context->model, context->session,
        provider, context->options.execution_profile,
        &context->device_logits_publication,
        element_offset, 1ull, context->plan.summary.vocabulary_size, err);
}
static int logits_row_identity_build(yvex_runtime_logits_row_result *result)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!result) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.row.v2") ||
        !yvex_sha256_update_u64(&hash, result->source_phase) ||
        !yvex_sha256_update_u64(&hash, result->source_position) ||
        !yvex_sha256_update_u64(&hash, result->vocabulary_size) ||
        !yvex_sha256_update_u64(&hash, result->host_values_available) ||
        !yvex_sha256_update_u64(&hash, result->device_values_available) ||
        !yvex_sha256_update_u64(&hash, result->evidence_profile) ||
        !yvex_sha256_update_text(&hash, result->source_hidden_digest) ||
        !yvex_sha256_update_text(&hash, result->output_head_plan_identity) ||
        !yvex_sha256_update_text(&hash, result->output_head_residency_identity) ||
        !yvex_sha256_update_text(&hash, result->backend_execution_identity))
        return 0;
    if (result->host_values_available &&
        !yvex_sha256_update_text(&hash, result->raw_logits_digest))
        return 0;
    if (result->device_values_available &&
        (!yvex_sha256_update_text(
             &hash, result->device_logits.execution_profile_identity) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.resource_generation) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.session_generation) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.state_generation) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.element_offset) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.rows) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.columns)))
        return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, result->logits_row_identity);
    return 1;
}
static int logits_row_finish(yvex_runtime_logits_context *context,
                             const yvex_runtime_logits_source *source,
                             yvex_backend_kind backend, const float *host_values,
                             unsigned long long device_element_offset,
                             yvex_runtime_logits_row_result *result,
                             yvex_error *err)
{
    yvex_runtime_residency_summary residency;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, scan_bytes;
    float minimum = FLT_MAX, maximum = -FLT_MAX;
    if (yvex_runtime_residency_snapshot(context->model_view->residency,
                                        &residency, NULL, NULL, err) != YVEX_OK)
        return yvex_error_code(err);
    result->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    result->evidence_profile = context->options.evidence_profile;
    result->source_phase = source->source_phase;
    result->source_position = source->source_position;
    result->vocabulary_size = context->plan.summary.vocabulary_size;
    result->hidden_width = context->plan.summary.hidden_width;
    result->logits_count = context->plan.summary.vocabulary_size;
    yvex_runtime_identity_copy(result->source_hidden_digest,
                               source->normalized_hidden_digest);
    yvex_runtime_identity_copy(result->output_head_plan_identity,
                               context->plan.summary.output_head_plan_identity);
    yvex_runtime_identity_copy(result->output_head_residency_identity,
                               residency.output_head_residency_identity);
    if (context->options.device_selection) {
        result->device_values_available = 1;
        if (logits_device_view_build(context, &residency, source->source_phase,
                                     device_element_offset,
                                     &result->device_logits, err) != YVEX_OK)
            return yvex_error_code(err);
    } else {
        if (!host_values)
            return logits_refuse(err, YVEX_ERR_STATE,
                                 "host logits values are unavailable");
        for (index = 0ull; index < result->vocabulary_size; ++index) {
            float value = host_values[index];
            if (!isfinite(value))
                return logits_refuse(
                    err, YVEX_ERR_FORMAT,
                    "output-head projection produced non-finite logits");
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        }
        if (!yvex_core_u64_mul(result->vocabulary_size, sizeof(float),
                               &scan_bytes) ||
            !yvex_execution_f32_digest("yvex.runtime.raw-logits.v1",
                                  host_values, result->logits_count,
                                  result->raw_logits_digest))
            return logits_refuse(err, YVEX_ERR_STATE,
                                 "raw logits evidence derivation failed");
        result->host_values_available = result->finite_count_available =
            result->range_available = result->raw_digest_available = 1;
        result->finite_count = result->logits_count;
        result->minimum_logit = minimum;
        result->maximum_logit = maximum;
        result->full_array_host_scan_bytes = scan_bytes;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.backend-evidence.v2") ||
        !yvex_sha256_update_u64(&hash, backend) ||
        !yvex_sha256_update_u64(&hash, result->h2d_bytes) ||
        !yvex_sha256_update_u64(&hash, result->d2h_bytes) ||
        !yvex_sha256_update_u64(&hash, result->kernel_launches) ||
        !yvex_sha256_update_u64(&hash, result->host_values_available) ||
        !yvex_sha256_update_u64(&hash, result->device_values_available) ||
        !yvex_sha256_update_u64(&hash, result->evidence_profile) ||
        (result->host_values_available &&
         !yvex_sha256_update_text(&hash, result->raw_logits_digest)) ||
        (result->device_values_available &&
         !yvex_sha256_update_text(
             &hash, result->device_logits.execution_profile_identity)) ||
        !yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "logits backend evidence identity derivation failed");
    yvex_sha256_hex(digest, result->backend_execution_identity);
    if (!logits_row_identity_build(result))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "logits row identity derivation failed");
    result->completed = 1;
    return YVEX_OK;
}
/*
 * Independently re-admit one published complete logits row for a downstream owner.
 *
 * Stale geometry, payload, plan, or field-wise row identity refuses before consumption.
 */
int yvex_runtime_logits_row_validate(
    const yvex_runtime_logits_plan_summary *plan, const float *logits,
    unsigned long long logits_capacity,
    const yvex_runtime_logits_row_result *result, yvex_error *err)
{
    yvex_runtime_logits_row_result canonical;
    char raw_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long hidden_bytes, logits_bytes, activation_bytes;
    if (!plan || !yvex_core_u64_mul(plan->hidden_width, sizeof(float), &hidden_bytes) ||
        !yvex_core_u64_mul(plan->vocabulary_size, sizeof(float), &logits_bytes) ||
        !yvex_core_u64_add(hidden_bytes, logits_bytes, &activation_bytes) ||
        !result || !result->completed ||
        result->schema_version != YVEX_RUNTIME_LOGITS_SCHEMA_V1 ||
        result->source_phase > YVEX_LOGITS_SOURCE_DRAFT ||
        result->evidence_profile > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        !plan->vocabulary_size || result->vocabulary_size != plan->vocabulary_size ||
        result->logits_count != plan->vocabulary_size ||
        result->host_values_available == result->device_values_available ||
        strcmp(result->output_head_plan_identity,
               plan->output_head_plan_identity) != 0 ||
        !yvex_sha256_hex_valid(result->source_hidden_digest) ||
        !yvex_sha256_hex_valid(result->output_head_residency_identity) ||
        !yvex_sha256_hex_valid(result->backend_execution_identity) ||
        result->memory.complete != (result->kernel_launches != 0ull) ||
        (result->memory.complete &&
         (result->memory.active_weight_bytes != plan->encoded_bytes ||
          result->memory.state_bytes || result->memory.activation_bytes != activation_bytes ||
          !result->memory.temporary_bytes)) ||
        (!result->memory.complete &&
         (result->memory.active_weight_bytes || result->memory.state_bytes ||
          result->memory.activation_bytes || result->memory.temporary_bytes)))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "complete logits row geometry or payload is stale");
    if (result->host_values_available &&
        (!logits || logits_capacity < result->logits_count ||
         !result->finite_count_available || !result->range_available ||
         !result->raw_digest_available ||
         result->finite_count != result->logits_count ||
         !yvex_execution_f32_digest("yvex.runtime.raw-logits.v1", logits,
                               result->logits_count, raw_digest) ||
         strcmp(raw_digest, result->raw_logits_digest) != 0))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "host logits evidence is stale");
    if (result->device_values_available &&
        (logits || logits_capacity || result->finite_count_available || result->range_available ||
         result->raw_digest_available || result->finite_count ||
         result->device_logits.kind != YVEX_EXECUTION_DEVICE_LOGITS ||
         result->device_logits.rows != 1ull ||
         result->device_logits.columns != result->vocabulary_size ||
         result->device_logits.materialization != YVEX_EXECUTION_MATERIALIZE_NONE ||
         yvex_execution_device_view_validate(&result->device_logits, err) !=
             YVEX_OK))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "device logits view is stale");
    canonical = *result;
    canonical.logits_row_identity[0] = '\0';
    if (!logits_row_identity_build(&canonical) ||
        strcmp(canonical.logits_row_identity, result->logits_row_identity) != 0)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "logits row identity is not canonical");
    yvex_error_clear(err);
    return YVEX_OK;
}
int yvex_runtime_logits_additive_adjust(
    const yvex_runtime_logits_context *context,
    const yvex_runtime_logits_row_result *base_result,
    const float *base_logits, const float *additive_logits, float *adjusted_logits,
    unsigned long long host_capacity, const yvex_device_tensor *adjusted_device,
    unsigned long long device_offset, const char *adjustment_identity,
    yvex_runtime_logits_row_result *result, yvex_error *err)
{
    const yvex_runtime_logits_plan_summary *plan = context ? &context->plan.summary : NULL;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char additive_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long index;
    float minimum = FLT_MAX, maximum = -FLT_MAX;
    int device = base_result && base_result->device_values_available;
    if (result) memset(result, 0, sizeof(*result));
    if (!plan || !base_result || !result ||
        base_result->source_phase != YVEX_LOGITS_SOURCE_DRAFT ||
        yvex_runtime_logits_row_validate(plan, device ? NULL : base_logits,
                                         device ? 0ull : host_capacity,
                                         base_result, err) != YVEX_OK ||
        (device && (!adjusted_device || base_logits || additive_logits || adjusted_logits ||
                    host_capacity || !adjusted_device->is_written ||
                    adjusted_device->owner != base_result->device_logits.backend ||
                    adjusted_device->dtype != YVEX_DTYPE_F32 ||
                    device_offset > adjusted_device->bytes / sizeof(float) ||
                    plan->vocabulary_size >
                        adjusted_device->bytes / sizeof(float) - device_offset ||
                    !yvex_sha256_hex_valid(adjustment_identity))) ||
        (!device && (!base_logits || !additive_logits || !adjusted_logits ||
                     host_capacity < plan->vocabulary_size || adjusted_device ||
                     device_offset || adjustment_identity)))
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "draft additive logits require one admitted base row");
    *result = *base_result;
    memset(&result->memory, 0, sizeof(result->memory));
    result->h2d_bytes = result->d2h_bytes = result->d2d_bytes = 0ull;
    result->kernel_launches = result->device_synchronizations = 0ull;
    if (device) {
        result->device_logits.tensor = adjusted_device;
        result->device_logits.element_offset = device_offset;
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.device-adjustment.v1") ||
            !yvex_sha256_update_text(&hash, base_result->logits_row_identity) ||
            !yvex_sha256_update_text(&hash, adjustment_identity) ||
            !yvex_sha256_update_u64(&hash, device_offset) || !yvex_sha256_final(&hash, digest))
            return logits_refuse(err, YVEX_ERR_STATE,
                                 "device logits adjustment identity derivation failed");
        yvex_sha256_hex(digest, result->backend_execution_identity);
        result->logits_row_identity[0] = '\0';
        if (!logits_row_identity_build(result) ||
            yvex_runtime_logits_row_validate(plan, NULL, 0ull, result, err) != YVEX_OK)
            return logits_refuse(err, YVEX_ERR_STATE,
                                 "device logits adjustment publication failed");
        yvex_error_clear(err);
        return YVEX_OK;
    }
    for (index = 0ull; index < plan->vocabulary_size; ++index) {
        double value = (double)base_logits[index] + additive_logits[index];
        if (!isfinite(additive_logits[index]) || !isfinite(value) ||
            value < -FLT_MAX || value > FLT_MAX)
            return logits_refuse(err, YVEX_ERR_FORMAT,
                                 "draft additive logits produced a non-finite value");
        adjusted_logits[index] = (float)value;
        if (adjusted_logits[index] < minimum) minimum = adjusted_logits[index];
        if (adjusted_logits[index] > maximum) maximum = adjusted_logits[index];
    }
    result->minimum_logit = minimum;
    result->maximum_logit = maximum;
    if (!yvex_core_u64_mul(plan->vocabulary_size, sizeof(float), &index) ||
        !yvex_core_u64_add(result->full_array_host_scan_bytes, index,
                           &result->full_array_host_scan_bytes))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "draft adjusted logits evidence extent overflowed");
    if (!yvex_execution_f32_digest("yvex.runtime.draft-logits-bias.v1", additive_logits,
                              plan->vocabulary_size, additive_digest) ||
        !yvex_execution_f32_digest("yvex.runtime.raw-logits.v1", adjusted_logits,
                              plan->vocabulary_size, result->raw_logits_digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "draft adjusted logits digest derivation failed");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.additive.v1") ||
        !yvex_sha256_update_text(&hash, base_result->logits_row_identity) ||
        !yvex_sha256_update_text(&hash, additive_digest) ||
        !yvex_sha256_update_text(&hash, result->raw_logits_digest) ||
        !yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "draft adjusted backend identity derivation failed");
    yvex_sha256_hex(digest, result->backend_execution_identity);
    result->logits_row_identity[0] = '\0';
    if (!logits_row_identity_build(result))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "draft adjusted row identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}
static int logits_physical_add_row(
    yvex_execution_physical_facts *physical,
    const yvex_runtime_logits_row_result *row, yvex_error *err)
{
    if (!physical || !row)
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "logits physical row is unavailable");
    return yvex_execution_physical_facts_add(
        physical, &row->memory, row->h2d_bytes, row->d2h_bytes,
        row->d2d_bytes, row->kernel_launches,
        0ull, row->device_synchronizations, err);
}
static int logits_physical_equal(
    const yvex_execution_physical_facts *left,
    const yvex_execution_physical_facts *right)
{
    return left && right &&
           left->memory.active_weight_bytes == right->memory.active_weight_bytes &&
           left->memory.state_bytes == right->memory.state_bytes &&
           left->memory.activation_bytes == right->memory.activation_bytes &&
           left->memory.temporary_bytes == right->memory.temporary_bytes &&
           left->memory.measured_operations == right->memory.measured_operations &&
           left->memory.missing_operations == right->memory.missing_operations &&
           left->memory.complete == right->memory.complete &&
           left->h2d_bytes == right->h2d_bytes && left->d2h_bytes == right->d2h_bytes &&
           left->d2d_bytes == right->d2d_bytes && left->kernel_count == right->kernel_count &&
           left->synchronization_count == right->synchronization_count;
}
int yvex_runtime_logits_result_validate(
    const yvex_runtime_logits_plan_summary *plan, const float *logits,
    unsigned long long logits_capacity,
    const yvex_runtime_logits_row_result *rows,
    unsigned long long row_capacity,
    const yvex_runtime_logits_result *result, yvex_error *err)
{
    yvex_execution_physical_facts expected = {0};
    unsigned long long valid_count, index, row_values, activation_values;
    unsigned long long activation_bytes, row_offset, expected_offset;
    int device_result;
    if (!plan || !rows || !result ||
        result->schema_version != YVEX_RUNTIME_LOGITS_SCHEMA_V1 ||
        !plan->vocabulary_size || !result->completed_rows ||
        result->completed_rows > result->requested_rows ||
        row_capacity < result->completed_rows ||
        !yvex_core_u64_mul(result->completed_rows,
                           plan->vocabulary_size, &valid_count) ||
        (rows[0].device_values_available
             ? (logits || logits_capacity)
             : (!logits || logits_capacity != valid_count)))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "repeated logits prefix extent is not canonical");
    device_result = rows[0].device_values_available;
    if ((result->completed &&
         (result->partial || result->completed_rows != result->requested_rows ||
          result->first_incomplete_row != result->requested_rows)) ||
        (result->partial &&
         (result->completed || result->completed_rows >= result->requested_rows ||
          result->first_incomplete_row != result->completed_rows)) ||
        (!result->completed && !result->partial))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "repeated logits completion directory is inconsistent");
    if (!yvex_core_u64_add(plan->hidden_width, plan->vocabulary_size,
                           &row_values) ||
        !yvex_core_u64_mul(result->grouped_rows, row_values,
                           &activation_values) ||
        !yvex_core_u64_mul(activation_values, sizeof(float),
                           &activation_bytes))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "grouped logits physical extent overflowed");
    if ((result->grouped_execution &&
         (result->grouped_rows != result->requested_rows ||
          !result->physical.memory.complete ||
          result->physical.memory.active_weight_bytes != plan->encoded_bytes ||
          result->physical.memory.state_bytes ||
          result->physical.memory.activation_bytes != activation_bytes ||
          !result->physical.memory.temporary_bytes ||
          !result->physical.kernel_count)) ||
        (!result->grouped_execution && result->grouped_rows) ||
        (device_result && !result->grouped_execution))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "grouped logits physical facts are inconsistent");
    for (index = 0ull; index < result->completed_rows; ++index) {
        if (rows[index].device_values_available != device_result ||
            (device_result &&
             (!yvex_core_u64_mul(index, plan->vocabulary_size, &row_offset) ||
              !yvex_core_u64_add(rows[0].device_logits.element_offset, row_offset,
                                 &expected_offset) ||
              rows[index].device_logits.tensor != rows[0].device_logits.tensor ||
              rows[index].device_logits.backend != rows[0].device_logits.backend ||
              rows[index].device_logits.element_offset != expected_offset ||
              rows[index].device_logits.resource_generation !=
                  rows[0].device_logits.resource_generation ||
              rows[index].device_logits.session_generation !=
                  rows[0].device_logits.session_generation ||
              rows[index].device_logits.state_generation !=
                  rows[0].device_logits.state_generation ||
              strcmp(rows[index].device_logits.execution_profile_identity,
                     rows[0].device_logits.execution_profile_identity) != 0)))
            return logits_refuse(err, YVEX_ERR_FORMAT,
                                 "logits row result classes or device layout differ");
        if (yvex_runtime_logits_row_validate(
                plan, device_result ? NULL : logits + index * plan->vocabulary_size,
                device_result ? 0ull : plan->vocabulary_size,
                &rows[index], err) != YVEX_OK)
            return yvex_error_code(err);
        if (!result->grouped_execution &&
            logits_physical_add_row(&expected, &rows[index], err) != YVEX_OK)
            return yvex_error_code(err);
    }
    if (!result->grouped_execution &&
        !logits_physical_equal(&expected, &result->physical))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "row-local logits physical facts are inconsistent");
    yvex_error_clear(err);
    return YVEX_OK;
}
static int logits_enter(yvex_runtime_logits_context *context, yvex_error *err)
{
    if (!context || !context->mutex_ready ||
        pthread_mutex_lock(&context->mutex) != 0)
        return logits_refuse(err, YVEX_ERR_STATE, "logits context lock failed");
    if (context->busy || context->invalidated) {
        (void)pthread_mutex_unlock(&context->mutex);
        return logits_refuse(err, YVEX_ERR_STATE,
                             "logits context is busy or invalidated");
    }
    context->busy = 1;
    (void)pthread_mutex_unlock(&context->mutex);
    return YVEX_OK;
}
static void logits_leave(yvex_runtime_logits_context *context, int completed)
{
    if (context && context->mutex_ready &&
        pthread_mutex_lock(&context->mutex) == 0) {
        context->busy = 0;
        if (completed) context->execution_count++;
        (void)pthread_mutex_unlock(&context->mutex);
    }
}
int yvex_runtime_logits_project(
    yvex_runtime_logits_context *context,
    const yvex_runtime_logits_source *source, yvex_backend_kind backend,
    float *logits, unsigned long long logits_capacity,
    yvex_runtime_logits_row_result *result, yvex_error *err)
{
    unsigned long long output_bytes;
    int rc = logits_enter(context, err);
    if (result) memset(result, 0, sizeof(*result));
    if (rc != YVEX_OK) return rc;
    if (!result ||
        (!context->options.device_selection &&
         (!logits || logits_capacity < context->plan.summary.vocabulary_size)) ||
        backend != yvex_backend_kind_of(context->session_view->backend))
        rc = logits_refuse(err, YVEX_ERR_INVALID_ARG,
                           "logits output capacity or backend is incompatible");
    if (rc == YVEX_OK) rc = logits_source_validate(context, source, err);
    if (rc == YVEX_OK && context->options.cancel_requested &&
        context->options.cancel_requested(context->options.cancel_context))
        rc = logits_refuse(err, YVEX_ERR_CANCELLED,
                           "logits projection was cancelled before execution");
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CPU &&
        !source->host_values_available)
        rc = logits_refuse(err, YVEX_ERR_UNSUPPORTED,
                           "CPU logits require an explicit host hidden row");
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CPU)
        rc = logits_project_cpu(context, source->normalized_hidden, err);
    else if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA)
        rc = logits_project_cuda(context, source, result, err);
    if (rc == YVEX_OK)
        rc = logits_row_finish(context, source, backend, context->candidate,
                               0ull, result, err);
    if (rc == YVEX_OK && !context->options.device_selection &&
        yvex_core_u64_mul(context->plan.summary.vocabulary_size, sizeof(float),
                          &output_bytes))
        memcpy(logits, context->candidate, (size_t)output_bytes);
    else if (rc == YVEX_OK && !context->options.device_selection)
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "logits publication extent overflowed");
    logits_leave(context, rc == YVEX_OK);
    return rc;
}
static int logits_batch_contract(
    const yvex_runtime_logits_context *context,
    const yvex_output_head_batch_request *request,
    char identity[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const yvex_runtime_logits_plan_summary *plan = context ? &context->plan.summary : NULL;
    int host_result = request &&
        request->result_class == YVEX_OUTPUT_HEAD_RESULT_HOST_LOGITS;
    int device_result = request &&
        request->result_class == YVEX_OUTPUT_HEAD_RESULT_DEVICE_LOGITS;
    if (!plan || !request || request->schema_version != YVEX_OUTPUT_HEAD_BATCH_SCHEMA_V1 ||
        !request->row_count || request->row_count > context->options.maximum_rows ||
        request->output_vocabulary != plan->vocabulary_size ||
        request->backend != yvex_backend_kind_of(context->session_view->backend) ||
        request->selection_policy != YVEX_OUTPUT_HEAD_SELECTION_RAW ||
        request->evidence_profile != context->options.evidence_profile ||
        (!host_result && !device_result) ||
        (host_result &&
         (context->options.device_selection ||
          request->execution_class != YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE)) ||
        (device_result &&
         (!context->options.device_selection || request->backend != YVEX_BACKEND_KIND_CUDA ||
          request->evidence_profile != YVEX_EXECUTION_EVIDENCE_PRODUCTION ||
          request->execution_class != YVEX_EXECUTION_CLASS_DEVICE_NATIVE)) ||
        (context->options.execution_profile &&
         (!request->execution_profile_identity ||
          strcmp(request->execution_profile_identity,
                 context->options.execution_profile->identity) != 0)) ||
        (!context->options.execution_profile && request->execution_profile_identity))
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "output-head batch contract is incompatible");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.output-head-batch.v1") ||
        !yvex_sha256_update_text(&hash, plan->output_head_plan_identity) ||
        !yvex_sha256_update_u64(&hash, request->row_count) ||
        !yvex_sha256_update_u64(&hash, request->output_vocabulary) ||
        !yvex_sha256_update_u64(&hash, request->backend) ||
        !yvex_sha256_update_u64(&hash, request->result_class) ||
        !yvex_sha256_update_u64(&hash, request->selection_policy) ||
        !yvex_sha256_update_u64(&hash, request->evidence_profile) ||
        !yvex_sha256_update_u64(&hash, request->execution_class) ||
        !yvex_sha256_update_text(
            &hash, request->execution_profile_identity
                       ? request->execution_profile_identity : "uncompiled-reference") ||
        !yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "output-head batch contract identity failed");
    yvex_sha256_hex(digest, identity);
    return YVEX_OK;
}
static int logits_cuda_batch_compatible(
    const yvex_runtime_logits_context *context,
    const yvex_runtime_logits_source *sources, unsigned long long row_count)
{
    const yvex_execution_device_view *first;
    unsigned long long index, row_offset, expected_offset;
    int all_host = 1, all_device = 1;
    yvex_error ignored;
    if (!context || !sources || !row_count ||
        yvex_backend_kind_of(context->session_view->backend) != YVEX_BACKEND_KIND_CUDA)
        return 0;
    for (index = 0ull; index < row_count; ++index) {
        yvex_error_clear(&ignored);
        if (logits_source_validate(context, &sources[index], &ignored) != YVEX_OK)
            return 0;
        all_host = all_host && sources[index].host_values_available;
        all_device = all_device && sources[index].device_values_available;
    }
    if (all_host) return 1;
    if (!all_device) return 0;
    first = &sources[0].device_hidden;
    for (index = 1ull; index < row_count; ++index) {
        const yvex_execution_device_view *current = &sources[index].device_hidden;
        if (!yvex_core_u64_mul(index, context->plan.summary.hidden_width,
                               &row_offset) ||
            !yvex_core_u64_add(first->element_offset, row_offset,
                               &expected_offset) ||
            current->tensor != first->tensor || current->backend != first->backend ||
            current->element_offset != expected_offset ||
            current->resource_generation != first->resource_generation ||
            current->session_generation != first->session_generation ||
            current->state_generation != first->state_generation ||
            strcmp(current->execution_profile_identity,
                   first->execution_profile_identity) != 0)
            return 0;
    }
    return 1;
}
static int logits_cuda_batch_physical(
    yvex_runtime_logits_result *result,
    const yvex_backend_cuda_operation_facts *facts, int device_input, int device_output,
    unsigned long long hidden_bytes, unsigned long long logits_bytes,
    yvex_error *err)
{
    yvex_execution_physical_facts physical = {0};
    if (!result || !facts ||
        !yvex_core_u64_add(facts->d2h_bytes,
                           device_output ? 0ull : logits_bytes,
                           &physical.d2h_bytes) ||
        !yvex_core_u64_add(facts->device_synchronizations,
                           (device_input ? 0ull : 1ull) +
                               (device_output ? 0ull : 1ull),
                           &physical.synchronization_count))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "grouped logits physical accounting overflowed");
    physical.h2d_bytes = device_input ? 0ull : hidden_bytes;
    physical.d2d_bytes = facts->d2d_bytes;
    physical.kernel_count = facts->kernel_launches;
    if (yvex_execution_memory_facts_add(
            &physical.memory, facts->active_weight_bytes, facts->state_bytes,
            facts->activation_bytes, facts->temporary_bytes,
            facts->compulsory_memory_facts_available,
            !facts->compulsory_memory_facts_available, err) != YVEX_OK)
        return yvex_error_code(err);
    result->physical = physical;
    return YVEX_OK;
}
/*
 * Project compatible rows through one CUDA output-head operation.
 *
 * Host rows use one bounded staging upload. Device rows borrow only one contiguous producer view;
 * incompatible layouts stay on the retained row-local path.
 */
static int logits_project_cuda_batch(
    yvex_runtime_logits_context *context,
    const yvex_runtime_logits_source *sources, unsigned long long row_count,
    int device_output, float *logits, yvex_runtime_logits_row_result *rows,
    yvex_runtime_logits_result *result, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts = {0};
    yvex_device_tensor borrowed_hidden, staging_hidden, logits_view;
    const yvex_device_tensor *device_hidden = context->device_hidden;
    unsigned long long hidden_elements, logits_elements, hidden_bytes, logits_bytes, index;
    int device_input = sources[0].device_values_available;
    int rc = logits_enter(context, err);
    if (rc != YVEX_OK) return rc;
    if (!context->device_hidden || !context->device_logits ||
        !yvex_core_u64_mul(row_count, context->plan.summary.hidden_width,
                           &hidden_elements) ||
        !yvex_core_u64_mul(row_count, context->plan.summary.vocabulary_size,
                           &logits_elements) ||
        !yvex_core_u64_mul(hidden_elements, sizeof(float), &hidden_bytes) ||
        !yvex_core_u64_mul(logits_elements, sizeof(float), &logits_bytes))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "grouped logits CUDA extent overflowed");
    if (rc == YVEX_OK && context->options.cancel_requested &&
        context->options.cancel_requested(context->options.cancel_context))
        rc = logits_refuse(err, YVEX_ERR_CANCELLED,
                           "grouped logits projection was cancelled before execution");
    if (rc == YVEX_OK &&
        !yvex_backend_tensor_f32_subview(
            context->device_logits, 0ull, logits_elements, &logits_view))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "grouped device logits view is unavailable");
    if (rc == YVEX_OK && device_input) {
        if (!yvex_backend_tensor_f32_subview(
                sources[0].device_hidden.tensor,
                sources[0].device_hidden.element_offset,
                hidden_elements, &borrowed_hidden))
            rc = logits_refuse(err, YVEX_ERR_FORMAT,
                               "grouped device hidden rows cannot be borrowed");
        else
            device_hidden = &borrowed_hidden;
    }
    if (rc == YVEX_OK && !device_input) {
        for (index = 0ull; index < row_count; ++index)
            memcpy(context->host_hidden_rows +
                       index * context->plan.summary.hidden_width,
                   sources[index].normalized_hidden,
                   (size_t)context->plan.summary.hidden_width * sizeof(float));
        if (!yvex_backend_tensor_f32_subview(
                context->device_hidden, 0ull, hidden_elements,
                &staging_hidden))
            rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                               "grouped hidden staging view is unavailable");
        else {
            device_hidden = &staging_hidden;
            rc = yvex_backend_tensor_write(
                context->session_view->backend, &staging_hidden,
                context->host_hidden_rows, hidden_bytes, err);
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_cuda_encoded_matvec(
            context->session_view->backend, context->resident_head,
            context->resident_head_bytes, context->plan.summary.qtype,
            context->plan.summary.row_count, context->plan.summary.row_width,
            context->plan.summary.row_bytes, row_count, device_hidden,
            NULL, 0ull, NULL, &logits_view, 0, &facts, err);
    if (rc == YVEX_OK)
        context->device_logits_publication = logits_view;
    if (rc == YVEX_OK && !device_output)
        rc = yvex_backend_tensor_read(context->session_view->backend,
                                      &logits_view, logits,
                                      logits_bytes, err);
    if (rc == YVEX_OK)
        rc = logits_cuda_batch_physical(result, &facts, device_input, device_output,
                                        hidden_bytes, logits_bytes, err);
    if (rc == YVEX_OK) {
        result->grouped_execution = 1;
        result->grouped_rows = row_count;
    }
    for (index = 0ull; rc == YVEX_OK && index < row_count; ++index) {
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context)) {
            rc = logits_refuse(err, YVEX_ERR_CANCELLED,
                               "grouped logits publication was cancelled");
            break;
        }
        rc = logits_row_finish(
            context, &sources[index], YVEX_BACKEND_KIND_CUDA,
            device_output ? NULL
                          : logits + index * context->plan.summary.vocabulary_size,
            index * context->plan.summary.vocabulary_size, &rows[index], err);
        if (rc == YVEX_OK) {
            result->completed_rows++;
            result->final_source_position = sources[index].source_position;
        }
    }
    logits_leave(context, rc == YVEX_OK);
    return rc;
}
/*
 * Execute an ordered output-head graph with complete-row partial-progress semantics.
 *
 * Compatible CUDA rows use one grouped operation; other layouts retain the bounded row oracle.
 */
int yvex_runtime_logits_execute_rows(
    yvex_runtime_logits_context *context,
    const yvex_output_head_batch_request *request,
    const yvex_runtime_logits_source *sources,
    float *logits, unsigned long long logits_capacity,
    yvex_runtime_logits_row_result *rows, unsigned long long row_capacity,
    yvex_runtime_logits_result *result, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, required;
    int device_output, grouped;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !request || !sources || !rows || !result)
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "ordered logits request or caller storage is invalid");
    rc = logits_batch_contract(context, request,
                               result->output_head_contract_identity, err);
    if (rc != YVEX_OK) return rc;
    device_output = request->result_class == YVEX_OUTPUT_HEAD_RESULT_DEVICE_LOGITS;
    if (row_capacity < request->row_count ||
        request->row_count > SIZE_MAX / sizeof(*rows) ||
        !yvex_core_u64_mul(request->row_count, context->plan.summary.vocabulary_size,
                           &required) ||
        (device_output ? (logits || logits_capacity)
                       : (!logits || logits_capacity < required)))
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "ordered logits result storage is incompatible");
    result->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    result->requested_rows = request->row_count;
    result->first_incomplete_row = request->row_count;
    memset(rows, 0, (size_t)request->row_count * sizeof(*rows));
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, device_output ? "yvex.runtime.logits.device-aggregate.v1"
                                 : "yvex.runtime.logits.aggregate.v1"))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "aggregate logits hash initialization failed");
    grouped = request->backend == YVEX_BACKEND_KIND_CUDA &&
              (device_output || request->row_count > 1ull) &&
              logits_cuda_batch_compatible(context, sources,
                                           request->row_count);
    if (grouped)
        rc = logits_project_cuda_batch(context, sources, request->row_count,
                                       device_output, logits, rows, result, err);
    else if (device_output)
        rc = logits_refuse(err, YVEX_ERR_UNSUPPORTED,
                           "device logits require one compatible grouped CUDA source");
    else {
        for (index = 0ull; index < request->row_count; ++index) {
            rc = yvex_runtime_logits_project(
                context, &sources[index], request->backend,
                logits + index * context->plan.summary.vocabulary_size,
                context->plan.summary.vocabulary_size, &rows[index], err);
            if (rc != YVEX_OK) break;
            rc = logits_physical_add_row(&result->physical, &rows[index], err);
            if (rc != YVEX_OK) break;
            result->completed_rows++;
            result->final_source_position = sources[index].source_position;
        }
    }
    if (rc != YVEX_OK) {
        result->partial = result->completed_rows != 0ull;
        result->first_incomplete_row = result->completed_rows;
    }
    for (index = 0ull; index < result->completed_rows; ++index)
        if (!yvex_sha256_update_text(
                &hash, device_output ? rows[index].logits_row_identity
                                     : rows[index].raw_logits_digest)) {
            rc = logits_refuse(err, YVEX_ERR_STATE,
                               "aggregate logits digest update failed");
            result->partial = index != 0ull;
            result->completed_rows = index;
            result->first_incomplete_row = index;
            result->final_source_position =
                index ? sources[index - 1ull].source_position : 0ull;
            break;
        }
    if (!yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "aggregate logits digest finalization failed");
    yvex_sha256_hex(digest, result->aggregate_logits_digest);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.execution.v2") ||
        !yvex_sha256_update_text(&hash, result->output_head_contract_identity) ||
        !yvex_sha256_update_u64(&hash, result->requested_rows) ||
        !yvex_sha256_update_u64(&hash, result->completed_rows) ||
        !yvex_sha256_update_u64(&hash, result->first_incomplete_row) ||
        !yvex_sha256_update_u64(&hash, result->final_source_position) ||
        !yvex_sha256_update_text(&hash, result->aggregate_logits_digest) ||
        !yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "logits execution identity derivation failed");
    yvex_sha256_hex(digest, result->execution_identity);
    result->completed = result->completed_rows == result->requested_rows;
    return rc;
}
static int logits_tensor_same_view(const yvex_device_tensor *left,
                                   const yvex_device_tensor *right)
{
    return left && right && left->owner == right->owner &&
           left->backend_allocation == right->backend_allocation &&
           left->data == right->data && left->bytes == right->bytes;
}
int yvex_runtime_logits_project_compatible(
    yvex_runtime_logits_context *const *contexts,
    const yvex_runtime_logits_source *const *sources,
    yvex_runtime_logits_row_result *const *rows, unsigned long long row_count, yvex_error *err)
{
    yvex_runtime_logits_context *leader;
    yvex_backend_cuda_operation_facts facts = {0};
    yvex_device_tensor input_view, output_view;
    unsigned long long hidden_elements, output_elements, row_values, row_bytes, d2d_bytes = 0ull, index, entered = 0ull;
    int rc = YVEX_OK;
    if (!contexts || !sources || !rows || row_count < 2ull || row_count >= 64ull ||
        !(leader = contexts[0]) || row_count > leader->options.maximum_rows ||
        !leader->options.device_selection ||
        !yvex_core_u64_mul(row_count, leader->plan.summary.hidden_width, &hidden_elements) ||
        !yvex_core_u64_mul(row_count, leader->plan.summary.vocabulary_size, &output_elements) ||
        !yvex_core_u64_add(leader->plan.summary.hidden_width, leader->plan.summary.vocabulary_size, &row_values) ||
        !yvex_core_u64_mul(row_values, sizeof(float), &row_bytes) ||
        !yvex_backend_tensor_f32_subview(leader->device_hidden, 0ull, hidden_elements, &input_view) ||
        !yvex_backend_tensor_f32_subview(leader->device_logits, 0ull, output_elements, &output_view))
        return logits_refuse(err, YVEX_ERR_INVALID_ARG, "compatible logits batch geometry is invalid");
    for (index = 0ull; index < row_count && rc == YVEX_OK; ++index) {
        unsigned long long prior;
        if (!contexts[index] || !sources[index] || !rows[index] ||
            !contexts[index]->options.device_selection ||
            contexts[index]->model != leader->model ||
            strcmp(contexts[index]->plan.summary.output_head_plan_identity,
                   leader->plan.summary.output_head_plan_identity) ||
            !sources[index]->device_values_available)
            rc = logits_refuse(err, YVEX_ERR_FORMAT, "compatible logits source is not device-compatible");
        for (prior = 0ull; prior < index && rc == YVEX_OK; ++prior)
            if (contexts[prior] == contexts[index])
                rc = logits_refuse(err, YVEX_ERR_FORMAT, "compatible logits session is duplicated");
        if (rc == YVEX_OK) rc = logits_enter(contexts[index], err);
        if (rc == YVEX_OK) entered++;
        if (rc == YVEX_OK) rc = logits_source_validate(contexts[index], sources[index], err);
        if (rc == YVEX_OK) memset(rows[index], 0, sizeof(*rows[index]));
    }
    for (index = 0ull; index < row_count && rc == YVEX_OK; ++index) {
        yvex_device_tensor source, destination;
        if (!yvex_backend_tensor_f32_subview(
                sources[index]->device_hidden.tensor,
                sources[index]->device_hidden.element_offset,
                leader->plan.summary.hidden_width, &source) ||
            !yvex_backend_tensor_f32_subview(
                &input_view, index * leader->plan.summary.hidden_width,
                leader->plan.summary.hidden_width, &destination))
            rc = logits_refuse(err, YVEX_ERR_BOUNDS, "compatible hidden row view is invalid");
        if (rc == YVEX_OK && !logits_tensor_same_view(&source, &destination)) {
            rc = yvex_backend_tensor_copy_shared_async(
                leader->session_view->backend, &destination, &source, err);
            if (rc == YVEX_OK && !yvex_core_u64_add(d2d_bytes, source.bytes, &d2d_bytes))
                rc = logits_refuse(err, YVEX_ERR_BOUNDS, "compatible logits D2D accounting overflowed");
        }
    }
    if (rc == YVEX_OK) {
        input_view.is_written = 1;
        rc = yvex_backend_cuda_encoded_matvec(
            leader->session_view->backend, leader->resident_head,
            leader->resident_head_bytes, leader->plan.summary.qtype,
            leader->plan.summary.row_count, leader->plan.summary.row_width,
            leader->plan.summary.row_bytes, row_count, &input_view, NULL, 0ull,
            NULL, &output_view, 0, &facts, err);
    }
    for (index = 0ull; index < row_count && rc == YVEX_OK; ++index) {
        yvex_device_tensor source, destination;
        if (!yvex_backend_tensor_f32_subview(
                &output_view, index * leader->plan.summary.vocabulary_size,
                leader->plan.summary.vocabulary_size, &source) ||
            !yvex_backend_tensor_f32_subview(
                contexts[index]->device_logits, 0ull,
                leader->plan.summary.vocabulary_size, &destination))
            rc = logits_refuse(err, YVEX_ERR_BOUNDS, "compatible output row view is invalid");
        if (rc == YVEX_OK && !logits_tensor_same_view(&source, &destination)) {
            rc = yvex_backend_tensor_copy_shared_async(
                contexts[index]->session_view->backend, &destination, &source, err);
            if (rc == YVEX_OK && !yvex_core_u64_add(d2d_bytes, source.bytes, &d2d_bytes))
                rc = logits_refuse(err, YVEX_ERR_BOUNDS, "compatible logits D2D accounting overflowed");
        }
        if (rc == YVEX_OK) destination.is_written = source.is_written;
        if (rc == YVEX_OK) contexts[index]->device_logits_publication = destination;
    }
    if (rc == YVEX_OK &&
        !yvex_core_u64_add(d2d_bytes, facts.d2d_bytes, &d2d_bytes))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS, "compatible logits D2D accounting overflowed");
    if (rc == YVEX_OK) {
        rows[0]->d2h_bytes = facts.d2h_bytes;
        rows[0]->d2d_bytes = d2d_bytes;
        rows[0]->kernel_launches = facts.kernel_launches;
        rows[0]->device_synchronizations = facts.device_synchronizations;
        rc = yvex_execution_memory_facts_add(
            &rows[0]->memory, facts.active_weight_bytes, facts.state_bytes,
            row_bytes, facts.temporary_bytes, facts.compulsory_memory_facts_available,
            !facts.compulsory_memory_facts_available, err);
    }
    for (index = 0ull; index < row_count && rc == YVEX_OK; ++index)
        rc = logits_row_finish(contexts[index], sources[index], YVEX_BACKEND_KIND_CUDA,
                               NULL, 0ull, rows[index], err);
    while (entered) logits_leave(contexts[--entered], rc == YVEX_OK);
    return rc;
}
int yvex_runtime_logits_execute(
    yvex_runtime_logits_context *context,
    const yvex_runtime_logits_source *sources, unsigned long long row_count,
    yvex_backend_kind backend, float *logits, unsigned long long logits_capacity,
    yvex_runtime_logits_row_result *rows, unsigned long long row_capacity,
    yvex_runtime_logits_result *result, yvex_error *err)
{
    yvex_output_head_batch_request request = {0};
    request.schema_version = YVEX_OUTPUT_HEAD_BATCH_SCHEMA_V1;
    request.row_count = row_count;
    request.output_vocabulary = context ? context->plan.summary.vocabulary_size : 0ull;
    request.backend = backend;
    request.result_class = YVEX_OUTPUT_HEAD_RESULT_HOST_LOGITS;
    request.selection_policy = YVEX_OUTPUT_HEAD_SELECTION_RAW;
    request.evidence_profile = context ? context->options.evidence_profile
                                       : YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    request.execution_class = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    request.execution_profile_identity = context && context->options.execution_profile
                                             ? context->options.execution_profile->identity : NULL;
    return yvex_runtime_logits_execute_rows(
        context, &request, sources, logits, logits_capacity, rows, row_capacity, result, err);
}
/*
 * Release logits-local buffers while preserving borrowed model/session owners.
 *
 * Busy or backend cleanup refusal retains retryable ownership.
 */
int yvex_runtime_logits_context_close(yvex_runtime_logits_context **context,
                                      yvex_error *err)
{
    int rc = YVEX_OK;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if ((*context)->mutex_ready && pthread_mutex_lock(&(*context)->mutex) == 0) {
        if ((*context)->busy) {
            (void)pthread_mutex_unlock(&(*context)->mutex);
            return logits_refuse(err, YVEX_ERR_STATE,
                                 "busy logits context cannot close");
        }
        (void)pthread_mutex_unlock(&(*context)->mutex);
    }
    if ((*context)->device_logits)
        rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                         &(*context)->device_logits, err);
    if (rc == YVEX_OK && (*context)->device_hidden)
        rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                         &(*context)->device_hidden, err);
    if (rc != YVEX_OK) return rc;
    if ((*context)->mutex_ready) (void)pthread_mutex_destroy(&(*context)->mutex);
    free((*context)->host_hidden_rows);
    free((*context)->candidate);
    memset(*context, 0, sizeof(**context));
    free(*context);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
static int logits_transformer_cleanup(void **opaque, yvex_error *err)
{
    return yvex_runtime_transformer_context_close(
        (yvex_runtime_transformer_context **)opaque, err);
}
static int logits_input_slice(yvex_transformer_input **out,
                              const yvex_transformer_input_summary *base,
                              const unsigned int *tokens,
                              unsigned long long offset,
                              unsigned long long count,
                              yvex_error *err)
{
    yvex_transformer_input_summary summary;
    if (!out || !base || !tokens || !count || offset > base->token_count ||
        count > base->token_count - offset)
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "logits token-input slice is invalid");
    summary = *base;
    summary.token_start = base->token_start + offset;
    summary.token_count = count;
    summary.payload_bytes = 0ull;
    summary.payload_digest[0] = summary.input_identity[0] = '\0';
    if (yvex_transformer_input_seal(&summary, tokens, err) != YVEX_OK)
        return yvex_error_code(err);
    return yvex_transformer_input_open_memory(out, &summary, tokens, err);
}
void yvex_runtime_logits_operator_result_release(yvex_logits_operator_result *result)
{
    if (!result) return;
    free(result->rows);
    free(result->raw_logits);
    result->rows = NULL;
    result->raw_logits = NULL;
    result->raw_logits_count = 0ull;
    result->row_count = 0ull;
}
static void logits_operator_refuse(yvex_logits_operator_result *result,
                                   const yvex_error *err)
{
    if (!result) return;
    yvex_core_text_copy(result->status, sizeof(result->status), "refused");
    yvex_core_text_copy(result->reason, sizeof(result->reason),
                        err && yvex_error_is_set(err)
                            ? yvex_error_message(err)
                            : "logits execution refused");
}
static void logits_operator_publish_facts(
    yvex_logits_operator_result *result,
    const yvex_transformer_plan_summary *transformer_plan,
    const yvex_model_engine_view *model_view,
    const yvex_runtime_logits_context *logits_context,
    yvex_backend_kind backend, int completed)
{
    yvex_runtime_residency_summary residency;
    yvex_error ignored;
    if (result && transformer_plan && model_view && logits_context) {
        const yvex_runtime_logits_plan_summary *logits_plan =
            yvex_runtime_logits_plan_summary_get(logits_context);
        if (logits_plan) result->plan = *logits_plan;
        yvex_core_text_copy(result->family, sizeof(result->family),
                            model_view->target_id);
        yvex_runtime_identity_copy(result->artifact_identity,
                                   model_view->binding->artifact_identity);
        yvex_runtime_identity_copy(result->runtime_binding_identity,
                                   model_view->binding->identity);
        yvex_runtime_identity_copy(result->transformer_plan_identity,
                                   transformer_plan->transformer_plan_identity);
        result->row_count = result->execution.completed_rows;
        result->prefill_logits_rows = result->row_count ? 1ull : 0ull;
        result->decode_logits_rows = result->row_count - result->prefill_logits_rows;
    }
    memset(&residency, 0, sizeof(residency));
    yvex_error_clear(&ignored);
    if (result && model_view && yvex_runtime_residency_snapshot(
            model_view->residency, &residency, NULL, NULL, &ignored) == YVEX_OK) {
        result->output_head_host_bytes = residency.output_head_encoded_bytes;
        if (backend == YVEX_BACKEND_KIND_CUDA) {
            result->output_head_device_bytes = residency.output_head_encoded_bytes;
            result->output_head_upload_bytes = residency.output_head_encoded_bytes;
            result->output_head_upload_count = 1ull;
        }
    }
    if (result && completed) {
        result->output_head_binding_ready = result->output_head_residency_ready = 1;
        result->logits_cpu_ready = result->logits_cuda_ready = 1;
        result->logits_prefill_ready = result->logits_decode_ready = 1;
        result->logits_full_vocabulary_ready = result->logits_hidden_contract_ready = 1;
        result->logits_partial_progress_ready = result->logits_ready = 1;
    }
}
static int logits_operator_finish(yvex_logits_operator_result *result, int rc,
                                  yvex_error *err)
{
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(result->status, sizeof(result->status), "complete");
        yvex_error_clear(err);
    } else {
        logits_operator_refuse(result, err);
    }
    return rc;
}
/*
 * Publish only the completed raw-logits prefix after repeated execution.
 *
 * Extent overflow preserves caller ownership and returns typed refusal.
 */
static int logits_operator_publish_raw(
    yvex_logits_operator_result *result, float **raw_logits,
    unsigned long long raw_capacity, unsigned long long row_capacity,
    const yvex_runtime_logits_plan_summary *plan, int rc, yvex_error *err)
{
    unsigned long long valid_logits_count;
    yvex_error validation_error;
    int validation_rc;
    if (!*raw_logits || !result->execution.completed_rows)
        return rc;
    if (!plan || !yvex_core_u64_mul(result->execution.completed_rows,
                                    plan->vocabulary_size,
                                    &valid_logits_count) ||
        valid_logits_count > raw_capacity) {
        if (rc != YVEX_OK) return rc;
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "completed raw logits prefix overflowed");
    }
    result->raw_logits = *raw_logits;
    result->raw_logits_count = valid_logits_count;
    *raw_logits = NULL;
    yvex_error_clear(&validation_error);
    validation_rc = yvex_runtime_logits_result_validate(
        plan, result->raw_logits, result->raw_logits_count,
        result->rows, row_capacity, &result->execution, &validation_error);
    if (validation_rc != YVEX_OK) {
        free(result->raw_logits);
        result->raw_logits = NULL;
        result->raw_logits_count = 0ull;
        if (rc == YVEX_OK) {
            if (err) *err = validation_error;
            return validation_rc;
        }
    }
    return rc;
}
/*
 * Execute one shared-context prefill/decode/logits operator workflow.
 *
 * Cleanup leases preserve exact ownership; completed raw rows remain caller-owned evidence.
 */
int yvex_runtime_logits_operator_execute(
    const yvex_logits_operator_request *request,
    yvex_logits_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err)
{
    yvex_model_engine_open_request model_request = {0};
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_transformer_options transformer_options = {0};
    yvex_runtime_decode_options decode_options = {0};
    yvex_runtime_logits_options logits_options = {0};
    yvex_model_engine_failure failure = {0};
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_model_engine *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_transformer_context *transformer = NULL;
    yvex_runtime_decode_context *decode = NULL;
    yvex_runtime_logits_context *logits_context = NULL;
    yvex_transformer_input *input = NULL, *prefill_input = NULL, *decode_input = NULL;
    const yvex_transformer_input_summary *input_summary;
    const yvex_transformer_plan_summary *plan;
    const yvex_model_engine_view *model_view;
    const unsigned int *tokens;
    yvex_transformer_input_limits limits = {0};
    yvex_runtime_transformer_request prefill_request = {0};
    yvex_runtime_transformer_output prefill_output = {0};
    yvex_runtime_transformer_result prefill_result = {0};
    yvex_runtime_decode_request decode_request = {0};
    yvex_runtime_decode_output decode_output = {0};
    yvex_runtime_decode_result decode_result = {0};
    yvex_runtime_logits_source *sources = NULL;
    yvex_runtime_decode_step_result *decode_steps = NULL;
    float *prefill_hidden = NULL, *decode_hidden = NULL, *raw_logits = NULL;
    unsigned long long prefill_count, decode_count, hidden_count, row_count, logits_count;
    yvex_error primary = {0}, cleanup_error;
    int adopted = 0, rc, close_rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !retained_cleanup || *retained_cleanup ||
        !request->target || !request->artifact_path || !request->runtime_binding_path ||
        !request->input_path || !request->prefill_tokens ||
        !request->prefill_chunk_tokens || !request->context_capacity ||
        (request->backend != YVEX_BACKEND_KIND_CPU &&
         request->backend != YVEX_BACKEND_KIND_CUDA))
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "complete logits operator arguments are required");
    yvex_core_text_copy(result->command, sizeof(result->command),
                        "execute transformer logits");
    yvex_core_text_copy(result->target, sizeof(result->target), request->target);
    yvex_core_text_copy(result->backend, sizeof(result->backend),
                        request->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    model_request.artifact_path = request->artifact_path;
    model_request.runtime_binding_path = request->runtime_binding_path;
    model_request.target_id = request->target;
    model_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.backend = request->backend;
    session_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.maximum_device_bytes = request->maximum_device_bytes;
    rc = yvex_runtime_cleanup_lease_acquire(
        &cleanup, &model_request, &session_request, &model, &session, &failure, err);
    limits.maximum_file_bytes = request->maximum_host_bytes
                                    ? request->maximum_host_bytes : 1ull << 30u;
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_file(&input, request->input_path,
                                              &limits, err);
    transformer_options.maximum_host_bytes = request->maximum_host_bytes;
    transformer_options.maximum_device_bytes = request->maximum_device_bytes;
    transformer_options.context_capacity = request->context_capacity;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_context_open(
            &transformer, model, session, &transformer_options, NULL, err);
    if (rc == YVEX_OK) {
        rc = yvex_runtime_cleanup_lease_adopt(
            cleanup, transformer, logits_transformer_cleanup, err);
        adopted = rc == YVEX_OK;
    }
    input_summary = yvex_transformer_input_summary_get(input);
    tokens = yvex_transformer_input_token_ids(input);
    plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(transformer));
    model_view = yvex_model_engine_view_get(model);
    if (rc == YVEX_OK &&
        (!input_summary || !tokens || !plan || !model_view || input_summary->token_start ||
         request->prefill_tokens >= input_summary->token_count ||
         request->context_capacity < input_summary->token_count))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "logits prefill/decode split or capacity is invalid");
    prefill_count = request->prefill_tokens;
    decode_count = input_summary ? input_summary->token_count - prefill_count : 0ull;
    row_count = decode_count + 1ull;
    if (rc == YVEX_OK)
        rc = logits_input_slice(&prefill_input, input_summary, tokens, 0ull,
                                prefill_count, err);
    if (rc == YVEX_OK)
        rc = logits_input_slice(&decode_input, input_summary, tokens + prefill_count,
                                prefill_count, decode_count, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(prefill_count, plan->hidden_width, &hidden_count) ||
         hidden_count > SIZE_MAX / sizeof(float)))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "logits prefill hidden extent overflowed");
    if (rc == YVEX_OK) {
        prefill_hidden = (float *)calloc((size_t)hidden_count, sizeof(float));
        prefill_output.normalized_hidden = prefill_hidden;
        prefill_output.capacity = hidden_count;
        if (!prefill_hidden) rc = logits_refuse(err, YVEX_ERR_NOMEM,
                                                "prefill hidden allocation failed");
    }
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(decode_count, plan->hidden_width, &hidden_count) ||
         hidden_count > SIZE_MAX / sizeof(float) || decode_count > SIZE_MAX))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "logits decode hidden extent overflowed");
    if (rc == YVEX_OK) {
        decode_hidden = (float *)calloc((size_t)hidden_count, sizeof(float));
        decode_steps = (yvex_runtime_decode_step_result *)calloc(
            (size_t)decode_count, sizeof(*decode_steps));
        sources = (yvex_runtime_logits_source *)calloc((size_t)row_count,
                                                       sizeof(*sources));
        result->rows = (yvex_runtime_logits_row_result *)calloc(
            (size_t)row_count, sizeof(*result->rows));
        if (!decode_hidden || !decode_steps || !sources || !result->rows)
            rc = logits_refuse(err, YVEX_ERR_NOMEM,
                               "logits operator directory allocation failed");
    }
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(row_count, plan->vocabulary_size, &logits_count) ||
         logits_count > SIZE_MAX / sizeof(float)))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "raw logits output extent overflowed");
    if (rc == YVEX_OK) {
        raw_logits = (float *)calloc((size_t)logits_count, sizeof(float));
        if (!raw_logits) rc = logits_refuse(err, YVEX_ERR_NOMEM,
                                            "raw logits output allocation failed");
    }
    decode_options.maximum_steps = decode_count;
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_context_open(
            &decode, transformer, session, &decode_options, err);
    logits_options.maximum_rows = row_count;
    logits_options.maximum_host_bytes = request->maximum_host_bytes;
    logits_options.maximum_device_bytes = request->maximum_device_bytes;
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_context_open(
            &logits_context, model, session,
            yvex_runtime_transformer_context_plan(transformer),
            &logits_options, err);
    prefill_request.chunk_tokens = request->prefill_chunk_tokens;
    prefill_request.backend = request->backend;
    prefill_request.phase = YVEX_TRANSFORMER_PHASE_PREFILL;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_execute(
            transformer, prefill_input, &prefill_request, &prefill_output,
            &prefill_result, err);
    decode_output.normalized_hidden = decode_hidden;
    decode_output.normalized_hidden_capacity = hidden_count;
    decode_output.steps = decode_steps;
    decode_output.step_capacity = decode_count;
    decode_request.backend = request->backend;
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_execute(decode, decode_input, &decode_request,
                                         &decode_output, &decode_result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_source_from_transformer(
            logits_context, &sources[0], &prefill_result, prefill_hidden,
            prefill_count * plan->hidden_width, prefill_count - 1ull, err);
    for (unsigned long long index = 0ull; rc == YVEX_OK && index < decode_count; ++index)
        rc = yvex_runtime_logits_source_from_decode(
            logits_context, &sources[index + 1ull], &decode_steps[index],
            decode_hidden + index * plan->hidden_width, plan->hidden_width, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_execute(
            logits_context, sources, row_count, request->backend,
            raw_logits, logits_count, result->rows, row_count,
            &result->execution, err);
    rc = logits_operator_publish_raw(
        result, &raw_logits, logits_count, row_count,
        yvex_runtime_logits_plan_summary_get(logits_context), rc, err);
    logits_operator_publish_facts(result, plan, model_view, logits_context,
                                  request->backend, rc == YVEX_OK);
    primary = err ? *err : (yvex_error){0};
    close_rc = yvex_runtime_logits_context_close(&logits_context, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    close_rc = yvex_runtime_decode_context_close(&decode, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    yvex_transformer_input_close(&decode_input);
    yvex_transformer_input_close(&prefill_input);
    yvex_transformer_input_close(&input);
    free(prefill_hidden); free(decode_hidden); free(decode_steps);
    free(sources); free(raw_logits);
    if (!adopted && transformer) {
        close_rc = yvex_runtime_transformer_context_close(&transformer, &cleanup_error);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    }
    close_rc = yvex_runtime_cleanup_lease_close(&cleanup, &cleanup_error);
    if (close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    if (cleanup) *retained_cleanup = cleanup;
    if (err && rc != YVEX_OK) *err = primary;
    return logits_operator_finish(result, rc, err);
}
