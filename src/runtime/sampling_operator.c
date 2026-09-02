/*
 * Own the bounded operator workflow that composes logits projection and sampling.
 *
 * The sampling core remains in sampling.c; this owner publishes typed operator evidence.
 */
#include <yvex/internal/sampling.h>

#include <stdint.h>
#include <string.h>

#include <yvex/internal/core.h>

static int sampling_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.sampling", reason);
    return status;
}
/* Publish sampling readiness only after real-logits selection completes. */
static void sampling_operator_publish(
    yvex_sampling_operator_result *result, const yvex_runtime_sampling_context_summary *summary)
{
    result->sample_count = result->execution.completed_samples;
    result->prefill_samples = result->sample_count ? 1ull : 0ull;
    result->decode_samples = result->sample_count - result->prefill_samples;
    result->workspace_bytes = summary->workspace_bytes;
    result->workspace_generation = summary->workspace_generation;
    result->cold_workspace_allocations = summary->cold_workspace_allocations;
    result->warm_workspace_allocations = summary->warm_workspace_allocations;
    result->sampling_source_contract_ready = 1;
    result->sampling_policy_ready = 1;
    result->sampling_greedy_ready = 1;
    result->sampling_temperature_ready = 1;
    result->sampling_top_k_ready = 1;
    result->sampling_top_p_ready = 1;
    result->sampling_min_p_ready = 1;
    result->sampling_typical_ready = 1;
    result->sampling_stochastic_ready = 1;
    result->sampling_seed_reproducibility_ready = 1;
    result->sampling_real_logits_ready = 1;
    result->sampling_partial_progress_ready = 1;
    result->sampling_ready = 1;
    result->persistent_state_unchanged = 1;
}
/* Execute admitted logits rows without appending selected tokens to model state. */
int yvex_runtime_sampling_operator_execute(
    const yvex_sampling_operator_request *request, yvex_sampling_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err)
{
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_source *sources = NULL;
    yvex_runtime_sampling_context_summary summary = {0};
    yvex_runtime_sampling_options options = {0};
    yvex_runtime_sampling_policy policy;
    yvex_error cleanup_error;
    unsigned long long index, row_count, vocabulary_size, logits_extent;
    int rc, close_rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !retained_cleanup || *retained_cleanup)
        return sampling_refuse(err, YVEX_ERR_INVALID_ARG,
                               "sampling operator request and empty cleanup output are required");
    yvex_core_text_copy(result->command, sizeof(result->command), "execute transformer sample");
    yvex_core_text_copy(result->target, sizeof(result->target), request->logits.target);
    yvex_core_text_copy(result->logits_backend, sizeof(result->logits_backend),
                        request->logits.backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    yvex_core_text_copy(result->sampling_execution_kind,
                        sizeof(result->sampling_execution_kind), "common-host");
    rc = yvex_runtime_logits_operator_execute(
        &request->logits, &result->logits, retained_cleanup, err);
    yvex_core_text_copy(result->family, sizeof(result->family), result->logits.family);
    if (rc != YVEX_OK) goto finish;
    row_count = result->logits.row_count;
    vocabulary_size = result->logits.plan.vocabulary_size;
    if (!row_count || !vocabulary_size ||
        !yvex_core_u64_mul(row_count, vocabulary_size, &logits_extent) ||
        !result->logits.rows ||
        !result->logits.raw_logits ||
        result->logits.raw_logits_count != logits_extent) {
        rc = sampling_refuse(err, YVEX_ERR_STATE,
                             "sampling operator received no complete real logits rows");
        goto finish;
    }
    rc = yvex_runtime_logits_result_validate(
        &result->logits.plan, result->logits.raw_logits,
        result->logits.raw_logits_count, result->logits.rows,
        result->logits.row_count, &result->logits.execution, err);
    if (rc != YVEX_OK) goto finish;
    policy = request->policy;
    if (yvex_runtime_sampling_policy_seal(&policy, vocabulary_size, err) != YVEX_OK) {
        rc = yvex_error_code(err);
        goto finish;
    }
    yvex_core_text_copy(result->strategy, sizeof(result->strategy),
                        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY ? "greedy" : "stochastic");
    result->policy = policy;
    options.maximum_vocabulary_size = vocabulary_size;
    options.maximum_rows = row_count;
    options.maximum_host_bytes = request->maximum_sampling_host_bytes;
    options.cancel_requested = request->cancel_requested;
    options.cancel_context = request->cancel_context;
    rc = yvex_runtime_sampling_context_open(
        &context, &result->logits.plan, &policy, &options, err);
    if (rc != YVEX_OK) goto finish;
    if (row_count > SIZE_MAX / sizeof(*sources) ||
        row_count > SIZE_MAX / sizeof(*result->samples)) {
        rc = sampling_refuse(err, YVEX_ERR_BOUNDS,
                             "sampling operator directory extent overflowed");
        goto close_context;
    }
    sources = (yvex_runtime_sampling_source *)yvex_core_calloc(
        (size_t)row_count, sizeof(*sources));
    result->samples = (yvex_runtime_sampling_result *)yvex_core_calloc(
        (size_t)row_count, sizeof(*result->samples));
    if (!sources || !result->samples) {
        rc = sampling_refuse(err, YVEX_ERR_NOMEM,
                             "sampling operator directory allocation failed");
        goto close_context;
    }
    for (index = 0ull; index < row_count && rc == YVEX_OK; ++index)
        rc = yvex_runtime_sampling_source_from_logits(
            context, &sources[index], result->logits.raw_logits + index * vocabulary_size,
            vocabulary_size, &result->logits.rows[index], err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_execute(
            context, sources, row_count, result->samples, row_count, &result->execution, err);
    if (yvex_runtime_sampling_context_snapshot(context, &summary, &cleanup_error) == YVEX_OK &&
        (rc == YVEX_OK || result->execution.completed_samples))
        sampling_operator_publish(result, &summary);
close_context:
    yvex_error_clear(&cleanup_error);
    close_rc = yvex_runtime_sampling_context_close(&context, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        if (err) *err = cleanup_error;
    }
finish:
    yvex_core_free(sources);
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(result->status, sizeof(result->status), "complete");
        yvex_error_clear(err);
    } else {
        yvex_core_text_copy(result->status, sizeof(result->status), "refused");
        yvex_core_text_copy(result->reason, sizeof(result->reason), err && yvex_error_is_set(err)
                                ? yvex_error_message(err)
                                : "sampling execution refused");
    }
    return rc;
}
void yvex_runtime_sampling_operator_result_release(
    yvex_sampling_operator_result *result)
{
    if (!result) return;
    yvex_core_free(result->samples);
    result->samples = NULL;
    result->sample_count = 0ull;
    yvex_runtime_logits_operator_result_release(&result->logits);
}
