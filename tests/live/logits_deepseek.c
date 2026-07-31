/*
 * Exercises one prefill and two decode hidden rows project through the exact resident output
 * head. Three complete 129280-value rows are compared per backend without tracked dumps.
 * Test-only consumer of the production Transformer, decode, residency, and logits APIs.
 */
#include <yvex/internal/logits.h>
#include <yvex/internal/core.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/sampling.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tests/reference/logits.h"
#include "tests/reference/sampling.h"

#define LIVE_LOGITS_ROWS 3ull
#define LIVE_LOGITS_ABSOLUTE_TOLERANCE 2.0e-2
#define LIVE_LOGITS_RELATIVE_TOLERANCE 1.0e-2

typedef struct {
    yvex_runtime_execution_session *session;
    yvex_runtime_transformer_context *transformer;
    yvex_runtime_decode_context *decode;
    yvex_runtime_logits_context *logits;
    float *prefill_hidden, *decode_hidden, *raw_logits, *reference_logits;
    yvex_runtime_decode_step_result decode_steps[2];
    yvex_runtime_transformer_result prefill_result;
    yvex_runtime_decode_result decode_result;
    yvex_runtime_logits_source sources[LIVE_LOGITS_ROWS];
    yvex_runtime_logits_row_result rows[LIVE_LOGITS_ROWS];
    yvex_runtime_logits_result result;
    yvex_runtime_session_summary before_logits, after_logits;
} live_logits;

typedef struct {
    unsigned long long compared, finite, first_failure;
    double maximum_absolute, maximum_relative, squared;
} live_comparison;

typedef struct {
    yvex_runtime_sampling_result rows[LIVE_LOGITS_ROWS];
    yvex_runtime_sampling_execution execution;
    yvex_runtime_sampling_context_summary summary;
} live_sampling_result;

static void live_fail(const char *step, int rc, const yvex_error *err)
{
    fprintf(stderr, "logits_live step=%s status=%d where=%s reason=%s\n",
            step, rc, err ? yvex_error_where(err) : "",
            err ? yvex_error_message(err) : "");
}

static int live_input_open(yvex_transformer_input **input,
                           const yvex_transformer_plan_summary *plan,
                           const unsigned int *tokens,
                           unsigned long long start,
                           unsigned long long count,
                           const char *path, yvex_error *err)
{
    yvex_transformer_input_summary summary;
    int rc;
    memset(&summary, 0, sizeof(summary));
    summary.schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
    summary.token_start = start;
    summary.token_count = count;
    summary.vocabulary_size = plan->vocabulary_size;
    yvex_runtime_identity_copy(summary.logical_model_identity,
                               plan->logical_model_identity);
    yvex_runtime_identity_copy(summary.runtime_numeric_identity,
                               plan->runtime_numeric_identity);
    yvex_runtime_identity_copy(summary.runtime_descriptor_identity,
                               plan->runtime_descriptor_identity);
    yvex_runtime_identity_copy(summary.transformer_plan_identity,
                               plan->transformer_plan_identity);
    rc = yvex_transformer_input_seal(&summary, tokens, err);
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_memory(input, &summary, tokens, err);
    if (rc == YVEX_OK && path)
        rc = yvex_transformer_input_write(path, &summary, tokens, err);
    return rc;
}

static int live_open(live_logits *execution, yvex_runtime_model *model,
                     yvex_backend_kind backend, yvex_error *err)
{
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_transformer_options transformer_options = {.context_capacity = 3ull};
    yvex_runtime_decode_options decode_options = {.maximum_steps = 2ull};
    yvex_runtime_logits_options logits_options = {.maximum_rows = LIVE_LOGITS_ROWS};
    yvex_runtime_model_failure failure = {0};
    memset(execution, 0, sizeof(*execution));
    if (yvex_runtime_session_open(&execution->session, model, &session_options,
                                  &failure, err) != YVEX_OK)
        return yvex_error_code(err);
    if (yvex_runtime_transformer_context_open(
            &execution->transformer, model, execution->session,
            &transformer_options, err) != YVEX_OK)
        return yvex_error_code(err);
    if (yvex_runtime_decode_context_open(
            &execution->decode, execution->transformer, execution->session,
            &decode_options, err) != YVEX_OK)
        return yvex_error_code(err);
    return yvex_runtime_logits_context_open(
        &execution->logits, model, execution->session,
        yvex_runtime_transformer_context_plan(execution->transformer),
        &logits_options, err);
}

static int live_close(live_logits *execution, yvex_error *err)
{
    yvex_error cleanup;
    int rc, close_rc;
    free(execution->prefill_hidden);
    free(execution->decode_hidden);
    free(execution->raw_logits);
    free(execution->reference_logits);
    execution->prefill_hidden = execution->decode_hidden = NULL;
    execution->raw_logits = execution->reference_logits = NULL;
    rc = yvex_runtime_logits_context_close(&execution->logits, err);
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_decode_context_close(&execution->decode, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_transformer_context_close(
        &execution->transformer, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_session_close(&execution->session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    return rc;
}

static int live_execute(live_logits *execution,
                        const yvex_transformer_input *prefill,
                        const yvex_transformer_input *decode,
                        yvex_backend_kind backend, yvex_error *err)
{
    const yvex_runtime_logits_plan_summary *logits_plan =
        yvex_runtime_logits_plan_summary_get(execution->logits);
    const yvex_transformer_plan_summary *transformer_plan =
        yvex_transformer_plan_summary_get(
            yvex_runtime_transformer_context_plan(execution->transformer));
    yvex_runtime_transformer_request prefill_request = {
        .chunk_tokens = 1ull, .backend = backend,
        .phase = YVEX_TRANSFORMER_PHASE_PREFILL};
    yvex_runtime_transformer_output prefill_output;
    yvex_runtime_decode_request decode_request = {.backend = backend};
    yvex_runtime_decode_output decode_output;
    unsigned long long hidden_width, vocabulary, logits_values;
    int rc;
    if (!logits_plan || !transformer_plan) return YVEX_ERR_STATE;
    hidden_width = transformer_plan->hidden_width;
    vocabulary = logits_plan->vocabulary_size;
    logits_values = LIVE_LOGITS_ROWS * vocabulary;
    execution->prefill_hidden = (float *)calloc((size_t)hidden_width, sizeof(float));
    execution->decode_hidden = (float *)calloc((size_t)(2ull * hidden_width), sizeof(float));
    execution->raw_logits = (float *)malloc((size_t)logits_values * sizeof(float));
    execution->reference_logits = (float *)malloc((size_t)logits_values * sizeof(float));
    if (!execution->prefill_hidden || !execution->decode_hidden ||
        !execution->raw_logits || !execution->reference_logits) return YVEX_ERR_NOMEM;
    prefill_output.normalized_hidden = execution->prefill_hidden;
    prefill_output.capacity = hidden_width;
    decode_output.normalized_hidden = execution->decode_hidden;
    decode_output.normalized_hidden_capacity = 2ull * hidden_width;
    decode_output.steps = execution->decode_steps;
    decode_output.step_capacity = 2ull;
    rc = yvex_runtime_transformer_execute(
        execution->transformer, prefill, &prefill_request, &prefill_output,
        &execution->prefill_result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_execute(
            execution->decode, decode, &decode_request, &decode_output,
            &execution->decode_result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_source_from_transformer(
            execution->logits, &execution->sources[0],
            &execution->prefill_result, execution->prefill_hidden,
            hidden_width, 0ull, err);
    for (unsigned long long row = 0ull; rc == YVEX_OK && row < 2ull; ++row)
        rc = yvex_runtime_logits_source_from_decode(
            execution->logits, &execution->sources[row + 1ull],
            &execution->decode_steps[row],
            execution->decode_hidden + row * hidden_width, hidden_width, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_summary_copy(
            execution->session, &execution->before_logits, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_execute(
            execution->logits, execution->sources, LIVE_LOGITS_ROWS, backend,
            execution->raw_logits, logits_values, execution->rows,
            LIVE_LOGITS_ROWS, &execution->result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_summary_copy(
            execution->session, &execution->after_logits, err);
    if (rc == YVEX_OK &&
        (!execution->result.completed || execution->result.partial ||
         execution->result.completed_rows != LIVE_LOGITS_ROWS ||
         execution->result.final_source_position != 2ull ||
         execution->before_logits.execution_count != execution->after_logits.execution_count ||
         execution->before_logits.warm_weight_artifact_reads !=
             execution->after_logits.warm_weight_artifact_reads ||
         execution->before_logits.warm_weight_upload_bytes !=
             execution->after_logits.warm_weight_upload_bytes ||
         execution->before_logits.warm_host_allocations !=
             execution->after_logits.warm_host_allocations ||
         execution->before_logits.warm_device_allocations !=
             execution->after_logits.warm_device_allocations ||
         execution->before_logits.warm_device_frees !=
             execution->after_logits.warm_device_frees)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.logits.structure",
                       "logits completion or warm resource invariants failed");
        rc = YVEX_ERR_FORMAT;
    }
    return rc;
}

/* Compute three complete independent rows from the exact resident output head. */
static int live_reference(live_logits *execution, yvex_runtime_model *model,
                          yvex_error *err)
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    const yvex_runtime_tensor_binding *runtime_binding;
    const yvex_materialized_tensor_binding *binding;
    const yvex_runtime_logits_plan_summary *plan =
        yvex_runtime_logits_plan_summary_get(execution->logits);
    const unsigned char *encoded = NULL;
    unsigned long long encoded_bytes = 0ull;
    const float *hidden[LIVE_LOGITS_ROWS];
    int rc;
    if (!view || !plan) return YVEX_ERR_STATE;
    runtime_binding = yvex_runtime_descriptor_find_role(
        view->descriptor, YVEX_TENSOR_ROLE_OUTPUT_HEAD, YVEX_TENSOR_SCOPE_GLOBAL,
        YVEX_MATERIALIZATION_NO_INDEX, YVEX_MATERIALIZATION_NO_INDEX);
    binding = runtime_binding ? yvex_materialization_session_tensor_at(
                                   view->materialization,
                                   runtime_binding->tensor_id) : NULL;
    rc = binding ? yvex_runtime_residency_binding_view(
                       view->residency, binding, &encoded, &encoded_bytes, err)
                 : YVEX_ERR_FORMAT;
    hidden[0] = execution->prefill_hidden;
    hidden[1] = execution->decode_hidden;
    hidden[2] = execution->decode_hidden + plan->hidden_width;
    for (unsigned long long row = 0ull; rc == YVEX_OK && row < LIVE_LOGITS_ROWS; ++row)
        if (!yvex_test_logits_reference_project(
                plan->qtype, encoded, (size_t)encoded_bytes,
                plan->row_count, plan->row_width, plan->row_bytes, hidden[row],
                execution->reference_logits + row * plan->vocabulary_size,
                plan->vocabulary_size)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "test.logits.reference",
                           "independent complete-vocabulary projection failed");
            rc = YVEX_ERR_FORMAT;
        }
    return rc;
}

static int live_compare(const float *actual, const float *reference,
                        unsigned long long count, int require_exact,
                        live_comparison *comparison)
{
    memset(comparison, 0, sizeof(*comparison));
    comparison->first_failure = count;
    for (unsigned long long index = 0ull; index < count; ++index) {
        double left = actual[index], right = reference[index];
        double difference = fabs(left - right);
        double scale = fmax(fabs(left), fabs(right));
        double relative = scale ? difference / scale : difference;
        comparison->compared++;
        if (isfinite(left) && isfinite(right)) comparison->finite++;
        if (difference > comparison->maximum_absolute)
            comparison->maximum_absolute = difference;
        if (relative > comparison->maximum_relative)
            comparison->maximum_relative = relative;
        comparison->squared += difference * difference;
        if ((!isfinite(left) || !isfinite(right) ||
             (require_exact ? memcmp(&actual[index], &reference[index], sizeof(float)) != 0
                            : difference > LIVE_LOGITS_ABSOLUTE_TOLERANCE +
                                               LIVE_LOGITS_RELATIVE_TOLERANCE * scale)) &&
            comparison->first_failure == count)
            comparison->first_failure = index;
    }
    return comparison->first_failure == count;
}

/*
 * Sample three already-computed complete logits rows through one reusable context.
 *
 * Publishes bounded selections. Source, arithmetic, identity, or cleanup refusal propagates.
 */
static int live_sample_rows(
    const yvex_runtime_logits_plan_summary *plan,
    const yvex_runtime_logits_row_result rows[LIVE_LOGITS_ROWS],
    const float *logits, yvex_runtime_sampling_policy policy,
    live_sampling_result *out, yvex_error *err)
{
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_source sources[LIVE_LOGITS_ROWS];
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 129280ull,
        .maximum_rows = LIVE_LOGITS_ROWS};
    yvex_error cleanup;
    int rc, close_rc;
    memset(out, 0, sizeof(*out));
    rc = yvex_runtime_sampling_policy_seal(&policy, plan->vocabulary_size, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_open(
            &context, plan, &policy, &options, err);
    for (unsigned long long index = 0ull; rc == YVEX_OK && index < LIVE_LOGITS_ROWS; ++index)
        rc = yvex_runtime_sampling_source_from_logits(
            context, &sources[index], logits + index * plan->vocabulary_size,
            plan->vocabulary_size, &rows[index], err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_execute(
            context, sources, LIVE_LOGITS_ROWS, out->rows, LIVE_LOGITS_ROWS,
            &out->execution, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_snapshot(context, &out->summary, err);
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_sampling_context_close(&context, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    return rc;
}

static int live_sampling_policy(
    const yvex_runtime_logits_plan_summary *plan,
    const live_logits *cpu, const live_logits *cuda,
    yvex_runtime_sampling_policy policy, live_sampling_result *cpu_out,
    yvex_error *err)
{
    live_sampling_result cuda_out;
    yvex_test_sampling_rng cpu_rng, cuda_rng;
    yvex_test_sampling_reference_result cpu_ref, cuda_ref;
    int rc = live_sample_rows(plan, cpu->rows, cpu->raw_logits,
                              policy, cpu_out, err);
    if (rc == YVEX_OK)
        rc = live_sample_rows(plan, cuda->rows, cuda->raw_logits,
                              policy, &cuda_out, err);
    yvex_test_sampling_reference_seed(policy.seed, &cpu_rng);
    yvex_test_sampling_reference_seed(policy.seed, &cuda_rng);
    for (unsigned long long index = 0ull; rc == YVEX_OK && index < LIVE_LOGITS_ROWS; ++index) {
        int cpu_ok = yvex_test_sampling_reference_select(
            cpu->raw_logits + index * plan->vocabulary_size,
            plan->vocabulary_size, &policy, &cpu_rng, &cpu_ref);
        int cuda_ok = yvex_test_sampling_reference_select(
            cuda->raw_logits + index * plan->vocabulary_size,
            plan->vocabulary_size, &policy, &cuda_rng, &cuda_ref);
        if (!cpu_ok || !cuda_ok ||
            cpu_out->rows[index].selected_token_id != cpu_ref.selected_token_id ||
            cuda_out.rows[index].selected_token_id != cuda_ref.selected_token_id ||
            cpu_out->rows[index].final_candidate_count != cpu_ref.candidate_count ||
            cuda_out.rows[index].final_candidate_count != cuda_ref.candidate_count ||
            cpu_out->rows[index].selected_token_id != cuda_out.rows[index].selected_token_id ||
            strcmp(cpu_out->rows[index].candidate_set_identity,
                   cuda_out.rows[index].candidate_set_identity) != 0 ||
            strcmp(cpu_out->rows[index].selected_token_identity,
                   cuda_out.rows[index].selected_token_identity) != 0)
            rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK &&
        (cpu_out->summary.warm_workspace_allocations ||
         cuda_out.summary.warm_workspace_allocations ||
         cpu_out->summary.workspace_generation != 1ull ||
         cuda_out.summary.workspace_generation != 1ull))
        rc = YVEX_ERR_STATE;
    if (rc != YVEX_OK && !yvex_error_is_set(err))
        yvex_error_set(err, rc, "test.sampling.live",
                       "real-logits sampling/reference agreement failed");
    return rc;
}

static int live_hidden_digest(const float *hidden, unsigned long long count,
                              char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!hidden || !count || !output ||
        !yvex_sha256_update_text(&hash, "yvex.transformer.normalized-hidden.v1"))
        return 0;
    for (unsigned long long index = 0ull; index < count; ++index) {
        unsigned int bits;
        if (!isfinite(hidden[index])) return 0;
        memcpy(&bits, &hidden[index], sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int live_cancel_always(void *context)
{
    (void)context;
    return 1;
}

static int live_failure_proofs(live_logits *execution, yvex_runtime_model *model,
                               yvex_error *err)
{
    const yvex_runtime_logits_plan_summary *plan =
        yvex_runtime_logits_plan_summary_get(execution->logits);
    yvex_runtime_logits_context *cancelled = NULL;
    yvex_runtime_logits_options options = {
        .maximum_rows = 1ull, .cancel_requested = live_cancel_always};
    yvex_runtime_logits_source sources[2], mutated_source;
    yvex_runtime_logits_row_result rows[2], mutated_row;
    yvex_runtime_logits_result partial;
    yvex_runtime_transformer_result producer;
    yvex_runtime_session_summary before, after;
    live_comparison mismatch;
    yvex_error cleanup_error;
    float *scratch = NULL, *mutated_hidden = NULL;
    float saved_reference, saved_hidden;
    unsigned long long vocabulary, hidden_width;
    int rc, close_rc;
    if (!plan) return YVEX_ERR_STATE;
    vocabulary = plan->vocabulary_size;
    hidden_width = plan->hidden_width;
    scratch = (float *)malloc((size_t)(2ull * vocabulary) * sizeof(float));
    mutated_hidden = (float *)malloc((size_t)hidden_width * sizeof(float));
    if (!scratch || !mutated_hidden) { rc = YVEX_ERR_NOMEM; goto cleanup; }

    rc = yvex_runtime_logits_context_open(
        &cancelled, model, execution->session,
        yvex_runtime_transformer_context_plan(execution->transformer), &options, err);
    if (rc != YVEX_OK) goto cleanup;
    scratch[0] = 90.0f;
    memset(&mutated_row, 0, sizeof(mutated_row));
    rc = yvex_runtime_logits_project(
        cancelled, &execution->sources[0], YVEX_BACKEND_KIND_CPU,
        scratch, vocabulary, &mutated_row, err);
    if (rc != YVEX_ERR_CANCELLED || mutated_row.completed || scratch[0] != 90.0f) {
        fprintf(stderr, "logits cancellation proof rc=%d expected=%d completed=%d output=%.9g\n",
                rc, YVEX_ERR_CANCELLED, mutated_row.completed, scratch[0]);
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.logits.cancellation",
                       "cancelled logits projection published a partial row");
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    yvex_error_clear(err);

    saved_hidden = execution->prefill_hidden[0];
    execution->prefill_hidden[0] = saved_hidden + 0.5f;
    scratch[0] = 91.0f;
    memset(&mutated_row, 0, sizeof(mutated_row));
    rc = yvex_runtime_logits_project(
        execution->logits, &execution->sources[0], YVEX_BACKEND_KIND_CPU,
        scratch, vocabulary, &mutated_row, err);
    execution->prefill_hidden[0] = saved_hidden;
    if (rc != YVEX_ERR_FORMAT || mutated_row.completed || scratch[0] != 91.0f) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.logits.hidden-mutation",
                       "mutated sealed hidden input did not refuse transactionally");
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    yvex_error_clear(err);

    memcpy(mutated_hidden, execution->prefill_hidden,
           (size_t)hidden_width * sizeof(float));
    mutated_hidden[0] += 0.5f;
    producer = execution->prefill_result;
    if (!live_hidden_digest(mutated_hidden, hidden_width,
                            producer.normalized_hidden_digest)) {
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    rc = yvex_runtime_logits_source_from_transformer(
        execution->logits, &mutated_source, &producer, mutated_hidden,
        hidden_width, 0ull, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_project(
            execution->logits, &mutated_source, YVEX_BACKEND_KIND_CPU,
            scratch, vocabulary, &mutated_row, err);
    if (rc != YVEX_OK || strcmp(mutated_row.raw_logits_digest,
                                execution->rows[0].raw_logits_digest) == 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.logits.hidden-sensitivity",
                       "an independently sealed hidden mutation did not change logits");
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }

    sources[0] = execution->sources[0];
    sources[1] = execution->sources[1];
    sources[1].source_identity[0] = sources[1].source_identity[0] == '0' ? '1' : '0';
    memset(rows, 0, sizeof(rows));
    memset(&partial, 0, sizeof(partial));
    scratch[vocabulary] = 93.0f;
    if (yvex_runtime_session_summary_copy(execution->session, &before, err) != YVEX_OK) {
        rc = yvex_error_code(err);
        goto cleanup;
    }
    rc = yvex_runtime_logits_execute(
        execution->logits, sources, 2ull, YVEX_BACKEND_KIND_CPU,
        scratch, 2ull * vocabulary, rows, 2ull, &partial, err);
    if (yvex_runtime_session_summary_copy(execution->session, &after, err) != YVEX_OK) {
        rc = yvex_error_code(err);
        goto cleanup;
    }
    if (rc != YVEX_ERR_FORMAT || !partial.partial || partial.completed ||
        partial.completed_rows != 1ull || partial.first_incomplete_row != 1ull ||
        !rows[0].completed || rows[1].completed || scratch[vocabulary] != 93.0f ||
        before.execution_count != after.execution_count) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.logits.partial",
                       "ordered logits partial publication invariants failed");
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    yvex_error_clear(err);

    saved_reference = execution->reference_logits[17];
    execution->reference_logits[17] = saved_reference + 1.0f;
    if (live_compare(execution->raw_logits, execution->reference_logits,
                     LIVE_LOGITS_ROWS * vocabulary, 1, &mismatch) ||
        mismatch.first_failure != 17ull) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.logits.mismatch-coordinate",
                       "comparison did not report the exact mutated vocabulary coordinate");
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    execution->reference_logits[17] = saved_reference;
    if (strcmp(execution->rows[0].raw_logits_digest,
               execution->rows[1].raw_logits_digest) == 0 ||
        strcmp(execution->rows[1].raw_logits_digest,
               execution->rows[2].raw_logits_digest) == 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.logits.source-sensitivity",
                       "distinct admitted hidden sources produced identical logits digests");
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    rc = YVEX_OK;
cleanup:
    yvex_error_clear(&cleanup_error);
    close_rc = yvex_runtime_logits_context_close(&cancelled, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup_error; }
    free(mutated_hidden);
    free(scratch);
    return rc;
}

int main(int argc, char **argv)
{
    yvex_runtime_model_open_request request = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_model *model = NULL;
    yvex_transformer_input *stream = NULL, *prefill = NULL, *decode = NULL;
    live_logits cpu = {0}, cuda = {0};
    live_comparison cpu_reference, cuda_reference, cpu_cuda;
    live_sampling_result greedy, categorical, filtered, reproduced;
    yvex_runtime_sampling_policy greedy_policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0};
    yvex_runtime_sampling_policy categorical_policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0,
        .seed_present = 1, .seed = 42ull};
    yvex_runtime_sampling_policy filtered_policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC,
        .temperature = 0.8, .top_k = 50ull, .top_p = 0.95,
        .min_p = 0.05, .typical_p = 0.9, .seed_present = 1, .seed = 42ull};
    const yvex_transformer_plan_summary *plan = NULL;
    const yvex_runtime_logits_plan_summary *logits_plan;
    const unsigned int tokens[] = {1u, 2u, 3u};
    yvex_error err, cleanup;
    const char *step = "model-open";
    unsigned long long values;
    int rc, close_rc;
    if (argc != 4) {
        fprintf(stderr, "usage: %s ARTIFACT RUNTIME_BINDING TOKEN_STREAM_OUTPUT\n", argv[0]);
        return 2;
    }
    (void)setvbuf(stdout, NULL, _IOLBF, 0);
    request.artifact_path = argv[1];
    request.runtime_binding_path = argv[2];
    request.target_id = "deepseek4-v4-flash";
    rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    if (rc == YVEX_OK) { step = "cpu-open"; rc = live_open(&cpu, model, YVEX_BACKEND_KIND_CPU, &err); }
    plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(cpu.transformer));
    if (rc == YVEX_OK) { step = "stream-open"; rc = live_input_open(&stream, plan, tokens, 0ull, 3ull, argv[3], &err); }
    if (rc == YVEX_OK) { step = "prefill-open"; rc = live_input_open(&prefill, plan, tokens, 0ull, 1ull, NULL, &err); }
    if (rc == YVEX_OK) { step = "decode-open"; rc = live_input_open(&decode, plan, tokens + 1u, 1ull, 2ull, NULL, &err); }
    if (rc == YVEX_OK) { step = "cpu-execute"; rc = live_execute(&cpu, prefill, decode, YVEX_BACKEND_KIND_CPU, &err); }
    if (rc == YVEX_OK) { step = "cpu-reference"; rc = live_reference(&cpu, model, &err); }
    logits_plan = yvex_runtime_logits_plan_summary_get(cpu.logits);
    values = logits_plan ? LIVE_LOGITS_ROWS * logits_plan->vocabulary_size : 0ull;
    if (rc == YVEX_OK && !live_compare(cpu.raw_logits, cpu.reference_logits, values, 1, &cpu_reference)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "test.logits.cpu-reference",
                       "CPU logits differ from the independent reference");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) { step = "cpu-failure-proofs"; rc = live_failure_proofs(&cpu, model, &err); }
    if (rc == YVEX_OK) { step = "cuda-open"; rc = live_open(&cuda, model, YVEX_BACKEND_KIND_CUDA, &err); }
    if (rc == YVEX_OK) { step = "cuda-execute"; rc = live_execute(&cuda, prefill, decode, YVEX_BACKEND_KIND_CUDA, &err); }
    if (rc == YVEX_OK) { step = "cuda-reference"; rc = live_reference(&cuda, model, &err); }
    if (rc == YVEX_OK &&
        (!live_compare(cuda.raw_logits, cuda.reference_logits, values, 0, &cuda_reference) ||
         !live_compare(cuda.raw_logits, cpu.raw_logits, values, 0, &cpu_cuda))) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "test.logits.cuda-reference",
                       "CUDA full-vocabulary logits exceed the numerical contract");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        step = "sampling-greedy";
        rc = live_sampling_policy(logits_plan, &cpu, &cuda,
                                  greedy_policy, &greedy, &err);
    }
    if (rc == YVEX_OK) {
        step = "sampling-categorical";
        rc = live_sampling_policy(logits_plan, &cpu, &cuda,
                                  categorical_policy, &categorical, &err);
    }
    if (rc == YVEX_OK) {
        step = "sampling-filtered";
        rc = live_sampling_policy(logits_plan, &cpu, &cuda,
                                  filtered_policy, &filtered, &err);
    }
    for (unsigned long long row = 0ull;
         rc == YVEX_OK && row < LIVE_LOGITS_ROWS; ++row)
        if (strcmp(categorical.rows[row].candidate_set_identity,
                   filtered.rows[row].candidate_set_identity) == 0) {
            yvex_error_set(&err, YVEX_ERR_FORMAT,
                           "test.sampling.filter-sensitivity",
                           "effective filter policy did not change candidate identity");
            rc = YVEX_ERR_FORMAT;
        }
    if (rc == YVEX_OK) {
        step = "sampling-reproducibility";
        rc = live_sample_rows(logits_plan, cpu.rows, cpu.raw_logits,
                              categorical_policy, &reproduced, &err);
    }
    if (rc == YVEX_OK &&
        strcmp(categorical.execution.ordered_selected_token_digest,
               reproduced.execution.ordered_selected_token_digest) != 0) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "test.sampling.reproducibility",
                       "same seed, policy, and logits did not reproduce");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc != YVEX_OK) live_fail(step, rc, &err);
    else
        printf("logits_rows=3 vocabulary=129280 compared_per_lane=%llu "
               "cpu_ref_max_abs=%.17g cuda_ref_max_abs=%.17g "
               "cuda_ref_max_rel=%.17g cuda_ref_rmse=%.17g cpu_cuda_max_abs=%.17g\n"
               "output_head_qtype=%u output_head_bytes=%llu plan_identity=%s "
               "residency_identity=%s\n"
               "cpu_logits_digest=%s cuda_logits_digest=%s "
               "prefill_logits_rows=1 decode_logits_rows=2 warm_reuse=pass state_unchanged=pass\n"
               "sampling_greedy_tokens=%u,%u,%u sampling_stochastic_tokens=%u,%u,%u "
               "filtered_candidates=%llu,%llu,%llu sampling_reference=pass reproducibility=pass "
               "warm_sampling_allocations=0\n"
               "sampling_policy_identity=%s candidate_identity=%s selected_identity=%s "
               "aggregate_sampling_identity=%s selected_probability=%.17g "
               "selected_log_probability=%.17g rng_draws=%llu\n"
               "logits_ready=1 sampling_ready=1 token_append_ready=0 tokenizer_runtime_ready=0 "
               "generation_ready=0 cuda_sampling_ready=0\n",
               values, cpu_reference.maximum_absolute,
               cuda_reference.maximum_absolute, cuda_reference.maximum_relative,
               sqrt(cuda_reference.squared / (double)values),
               cpu_cuda.maximum_absolute, logits_plan->qtype,
               logits_plan->encoded_bytes, logits_plan->output_head_plan_identity,
               cuda.rows[0].output_head_residency_identity,
               cpu.result.aggregate_logits_digest,
               cuda.result.aggregate_logits_digest,
               greedy.rows[0].selected_token_id, greedy.rows[1].selected_token_id,
               greedy.rows[2].selected_token_id,
               categorical.rows[0].selected_token_id,
               categorical.rows[1].selected_token_id,
               categorical.rows[2].selected_token_id,
               filtered.rows[0].final_candidate_count,
               filtered.rows[1].final_candidate_count,
               filtered.rows[2].final_candidate_count,
               categorical.rows[0].policy_identity,
               categorical.rows[0].candidate_set_identity,
               categorical.rows[0].selected_token_identity,
               categorical.execution.aggregate_sampling_identity,
               categorical.rows[0].selected_probability,
               categorical.rows[0].selected_log_probability,
               categorical.execution.completed_samples);
    yvex_transformer_input_close(&decode);
    yvex_transformer_input_close(&prefill);
    yvex_transformer_input_close(&stream);
    yvex_error_clear(&cleanup);
    close_rc = live_close(&cuda, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = live_close(&cpu, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_runtime_model_close(&model);
    if (model && rc == YVEX_OK) rc = YVEX_ERR_STATE;
    return rc == YVEX_OK ? 0 : 1;
}
