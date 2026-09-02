/*
 * Own the bounded operator workflow that composes transformer, decode, and logits.
 *
 * The logits projection core remains in logits.c; this owner only coordinates existing
 * runtime lifecycles and publishes typed operator evidence.
 */
#include <yvex/internal/logits.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "src/runtime/private.h"

static int logits_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.logits", reason);
    return status;
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
