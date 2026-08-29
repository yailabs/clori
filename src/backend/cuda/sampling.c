/*
 * Execute stochastic vocabulary selection on resident CUDA logits.
 *
 * The backend owns full-row filtering and categorical selection. Runtime owns the pending PCG
 * transition and publishes it only after these bounded facts pass validation.
 */
#include "src/backend/cuda/private.h"
#include "src/backend/cuda/kernel_primitives.h"

#include <yvex/internal/sampling.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define CUDA_SAMPLING_BLOCK 256u
#define CUDA_SAMPLING_ALIGNMENT 256ull

typedef struct {
    CUdeviceptr tokens, logits, probabilities, deviations;
    CUdeviceptr draft_dense, target_dense, counts, statistics, output_values;
    CUdeviceptr candidates, uniforms, committed, result, status;
} sampling_cuda_speculation_buffers;

static int sampling_cuda_refuse(yvex_error *err, yvex_status status,
                                const char *message)
{
    yvex_error_set(err, status, "cuda.sampling", message);
    return status;
}

static int sampling_cuda_padded(unsigned long long count,
                                unsigned long long *padded)
{
    unsigned long long value = 1ull;
    if (!count || count > UINT_MAX || !padded) return 0;
    while (value < count) {
        if (value > ULLONG_MAX / 2ull) return 0;
        value <<= 1ull;
    }
    *padded = value;
    return 1;
}

static int sampling_cuda_range_add(unsigned long long *cursor,
                                   unsigned long long bytes)
{
    unsigned long long aligned;
    if (!cursor || !bytes ||
        *cursor > ULLONG_MAX - (CUDA_SAMPLING_ALIGNMENT - 1ull)) return 0;
    aligned = (*cursor + CUDA_SAMPLING_ALIGNMENT - 1ull) &
              ~(CUDA_SAMPLING_ALIGNMENT - 1ull);
    return yvex_core_u64_add(aligned, bytes, cursor);
}

static int sampling_cuda_workspace_required(unsigned long long vocabulary_size,
                                            unsigned long long *bytes,
                                            yvex_error *err)
{
    unsigned long long padded, cursor = 0ull, extent;
    if (!bytes || !sampling_cuda_padded(vocabulary_size, &padded) ||
        !yvex_core_u64_mul(padded, sizeof(unsigned int), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !yvex_core_u64_mul(padded, sizeof(float), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !yvex_core_u64_mul(padded, sizeof(double), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !sampling_cuda_range_add(
            &cursor, YVEX_CUDA_SAMPLING_COUNT_FIELDS * sizeof(unsigned long long)) ||
        !sampling_cuda_range_add(
            &cursor, YVEX_CUDA_SAMPLING_STATISTIC_FIELDS * sizeof(double)) ||
        !sampling_cuda_range_add(
            &cursor, YVEX_CUDA_SAMPLING_VALUE_FIELDS * sizeof(float)) ||
        !sampling_cuda_range_add(
            &cursor, YVEX_CUDA_SAMPLING_SELECTION_FIELDS * sizeof(unsigned int)) ||
        !sampling_cuda_range_add(&cursor, sizeof(int)))
        return sampling_cuda_refuse(
            err, YVEX_ERR_BOUNDS,
            "stochastic CUDA workspace geometry exceeds addressable bounds");
    *bytes = cursor;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int sampling_cuda_speculation_workspace_required(
    unsigned long long vocabulary_size, unsigned long long candidate_count,
    unsigned long long *bytes, yvex_error *err)
{
    unsigned long long padded, cursor = 0ull, extent;
    if (!bytes || !candidate_count || candidate_count > UINT_MAX ||
        !sampling_cuda_padded(vocabulary_size, &padded) ||
        !yvex_core_u64_mul(padded, sizeof(unsigned int), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !yvex_core_u64_mul(padded, sizeof(float), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !yvex_core_u64_mul(padded, sizeof(double), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !yvex_core_u64_mul(vocabulary_size, sizeof(float), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !sampling_cuda_range_add(
            &cursor, YVEX_CUDA_SAMPLING_COUNT_FIELDS *
                         sizeof(unsigned long long)) ||
        !sampling_cuda_range_add(
            &cursor, YVEX_CUDA_SAMPLING_STATISTIC_FIELDS * sizeof(double)) ||
        !sampling_cuda_range_add(
            &cursor, YVEX_CUDA_SAMPLING_VALUE_FIELDS * sizeof(float)) ||
        !yvex_core_u64_mul(candidate_count, sizeof(unsigned int), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !yvex_core_u64_mul(candidate_count, sizeof(double), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !yvex_core_u64_mul(candidate_count + 1ull, sizeof(unsigned int), &extent) ||
        !sampling_cuda_range_add(&cursor, extent) ||
        !sampling_cuda_range_add(
            &cursor, YVEX_CUDA_SPECULATION_RESULT_FIELDS *
                         sizeof(unsigned long long)) ||
        !sampling_cuda_range_add(&cursor, sizeof(int)))
        return sampling_cuda_refuse(
            err, YVEX_ERR_BOUNDS,
            "stochastic speculation workspace geometry exceeds addressable bounds");
    *bytes = cursor;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int sampling_cuda_download(yvex_backend *backend,
                                  void *target, CUdeviceptr source,
                                  size_t bytes, const char *stage,
                                  yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUstream stream = yvex_cuda_launch_stream(backend);
    if (!state || !target || !source || !bytes) {
        return sampling_cuda_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "CUDA sampling download is invalid");
    }
    return yvex_cuda_status(
        &state->driver,
        stream ? state->driver.cuMemcpyDtoHAsync_v2(target, source, bytes, stream)
               : state->driver.cuMemcpyDtoH_v2(target, source, bytes),
        stage, err);
}

static int sampling_cuda_speculation_buffers_open(
    yvex_cuda_work *work, sampling_cuda_speculation_buffers *buffers,
    unsigned long long padded, unsigned long long vocabulary_size,
    unsigned long long candidate_bytes, const unsigned int *candidate_tokens,
    unsigned long long uniform_bytes, const double *acceptance_uniforms,
    unsigned long long committed_bytes, unsigned long long result_bytes,
    yvex_error *err)
{
    unsigned long long extent;
    int rc;
    if (!work || !buffers || !candidate_tokens || !acceptance_uniforms)
        return sampling_cuda_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "stochastic speculation workspace inputs are unavailable");
    memset(buffers, 0, sizeof(*buffers));
    extent = padded * sizeof(unsigned int);
    rc = yvex_cuda_work_allocate(
        work, &buffers->tokens, (size_t)extent, NULL, 0,
        "cuda.sampling.speculation.tokens", NULL, err);
    extent = padded * sizeof(float);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->logits, (size_t)extent, NULL, 0,
            "cuda.sampling.speculation.logits", NULL, err);
    extent = padded * sizeof(double);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->probabilities, (size_t)extent, NULL, 0,
            "cuda.sampling.speculation.probabilities", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->deviations, (size_t)extent, NULL, 0,
            "cuda.sampling.speculation.deviations", NULL, err);
    extent = vocabulary_size * sizeof(float);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->draft_dense, (size_t)extent, NULL, 0,
            "cuda.sampling.speculation.draft-dense", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->target_dense, (size_t)extent, NULL, 0,
            "cuda.sampling.speculation.target-dense", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->counts,
            YVEX_CUDA_SAMPLING_COUNT_FIELDS * sizeof(unsigned long long),
            NULL, 1, "cuda.sampling.speculation.counts", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->statistics,
            YVEX_CUDA_SAMPLING_STATISTIC_FIELDS * sizeof(double), NULL, 1,
            "cuda.sampling.speculation.statistics", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->output_values,
            YVEX_CUDA_SAMPLING_VALUE_FIELDS * sizeof(float), NULL, 1,
            "cuda.sampling.speculation.values", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->candidates, (size_t)candidate_bytes,
            candidate_tokens, 0, "cuda.sampling.speculation.candidates", NULL,
            err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->uniforms, (size_t)uniform_bytes,
            acceptance_uniforms, 0, "cuda.sampling.speculation.uniforms", NULL,
            err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->committed, (size_t)committed_bytes, NULL, 1,
            "cuda.sampling.speculation.committed", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->result, (size_t)result_bytes, NULL, 1,
            "cuda.sampling.speculation.result", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            work, &buffers->status, sizeof(int), NULL, 1,
            "cuda.sampling.speculation.status", NULL, err);
    return rc;
}

/* A failed bounded download cannot leave earlier work pending against reusable workspace. */
static int sampling_cuda_complete(yvex_backend *backend, int required,
                                  int *device_wide, const char *stage,
                                  int current, yvex_error *err)
{
    yvex_error completion;
    int rc;
    if (!required) return current;
    rc = yvex_cuda_launch_synchronize(
        backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
        device_wide, stage, &completion);
    if (rc == YVEX_OK) return current;
    if (err) *err = completion;
    return rc;
}

static int sampling_cuda_result_valid(
    unsigned long long vocabulary_size,
    const unsigned long long counts[YVEX_CUDA_SAMPLING_COUNT_FIELDS],
    const double statistics[YVEX_CUDA_SAMPLING_STATISTIC_FIELDS],
    const float values[YVEX_CUDA_SAMPLING_VALUE_FIELDS],
    const unsigned int selection[YVEX_CUDA_SAMPLING_SELECTION_FIELDS])
{
    return counts[YVEX_CUDA_SAMPLING_TOP_K_COUNT] &&
           counts[YVEX_CUDA_SAMPLING_TOP_K_COUNT] <= vocabulary_size &&
           counts[YVEX_CUDA_SAMPLING_MIN_P_COUNT] &&
           counts[YVEX_CUDA_SAMPLING_MIN_P_COUNT] <=
               counts[YVEX_CUDA_SAMPLING_TOP_K_COUNT] &&
           counts[YVEX_CUDA_SAMPLING_TYPICAL_COUNT] &&
           counts[YVEX_CUDA_SAMPLING_TYPICAL_COUNT] <=
               counts[YVEX_CUDA_SAMPLING_MIN_P_COUNT] &&
           counts[YVEX_CUDA_SAMPLING_TOP_P_COUNT] &&
           counts[YVEX_CUDA_SAMPLING_TOP_P_COUNT] <=
               counts[YVEX_CUDA_SAMPLING_TYPICAL_COUNT] &&
           selection[YVEX_CUDA_SAMPLING_SELECTED_TOKEN] < vocabulary_size &&
           selection[YVEX_CUDA_SAMPLING_NUMERIC_FALLBACK] <= 1u &&
           isfinite(values[YVEX_CUDA_SAMPLING_SELECTED_LOGIT]) &&
           isfinite(values[YVEX_CUDA_SAMPLING_MAXIMUM_LOGIT]) &&
           values[YVEX_CUDA_SAMPLING_SELECTED_LOGIT] <=
               values[YVEX_CUDA_SAMPLING_MAXIMUM_LOGIT] &&
           isfinite(statistics[YVEX_CUDA_SAMPLING_SELECTED_PROBABILITY]) &&
           statistics[YVEX_CUDA_SAMPLING_SELECTED_PROBABILITY] > 0.0 &&
           statistics[YVEX_CUDA_SAMPLING_SELECTED_PROBABILITY] <= 1.0 &&
           isfinite(statistics[YVEX_CUDA_SAMPLING_MIN_P_THRESHOLD]) &&
           statistics[YVEX_CUDA_SAMPLING_MIN_P_THRESHOLD] >= 0.0 &&
           isfinite(statistics[YVEX_CUDA_SAMPLING_ENTROPY]) &&
           statistics[YVEX_CUDA_SAMPLING_ENTROPY] >= 0.0 &&
           isfinite(statistics[YVEX_CUDA_SAMPLING_TYPICAL_MASS]) &&
           statistics[YVEX_CUDA_SAMPLING_TYPICAL_MASS] >= 0.0 &&
           isfinite(statistics[YVEX_CUDA_SAMPLING_TOP_P_MASS]) &&
           statistics[YVEX_CUDA_SAMPLING_TOP_P_MASS] >= 0.0 &&
           isfinite(statistics[YVEX_CUDA_SAMPLING_NORMALIZATION_ERROR]) &&
           statistics[YVEX_CUDA_SAMPLING_NORMALIZATION_ERROR] >= 0.0;
}

static int sampling_cuda_select_greedy_rows(
    yvex_backend *backend, const yvex_device_tensor *logits,
    unsigned long long row_count, unsigned long long row_width,
    unsigned int *selected_tokens, float *selected_values,
    unsigned long long *tie_counts, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    CUdeviceptr tokens_device = 0ull, values_device = 0ull, ties_device = 0ull;
    CUdeviceptr status_device = 0ull, input;
    unsigned long long value_count, value_bytes, token_bytes, selected_bytes, tie_bytes;
    unsigned long long temporary_bytes, row;
    int status = 0, device_wide = 0, launched = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !selected_tokens || !selected_values || !tie_counts || !facts ||
        !row_count || !row_width || row_count > UINT_MAX || row_width > UINT_MAX ||
        !yvex_core_u64_mul(row_count, row_width, &value_count) ||
        !yvex_core_u64_mul(value_count, sizeof(float), &value_bytes) ||
        !yvex_core_u64_mul(row_count, sizeof(*selected_tokens), &token_bytes) ||
        !yvex_core_u64_mul(row_count, sizeof(*selected_values), &selected_bytes) ||
        !yvex_core_u64_mul(row_count, sizeof(*tie_counts), &tie_bytes) ||
        !yvex_core_u64_add(token_bytes, selected_bytes, &temporary_bytes) ||
        !yvex_core_u64_add(temporary_bytes, tie_bytes, &temporary_bytes) ||
        !yvex_core_u64_add(temporary_bytes, sizeof(status), &temporary_bytes) ||
        value_bytes > SIZE_MAX || token_bytes > SIZE_MAX || selected_bytes > SIZE_MAX ||
        tie_bytes > SIZE_MAX || temporary_bytes > SIZE_MAX ||
        !backend_tensor_owner_is(backend, logits) || logits->dtype != YVEX_DTYPE_F32 ||
        !logits->is_written || logits->bytes < value_bytes)
        return sampling_cuda_refuse(
            err, YVEX_ERR_FORMAT,
            "greedy CUDA row selection geometry or ownership is incompatible");
    memset(selected_tokens, 0, (size_t)token_bytes);
    memset(selected_values, 0, (size_t)selected_bytes);
    memset(tie_counts, 0, (size_t)tie_bytes);
    rc = yvex_cuda_require_capability(
        backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, "cuda.sampling.greedy", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_set_current(backend, "cuda.sampling.greedy", err);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
#define ALLOC(field_, bytes_, stage_)                                              \
    if (rc == YVEX_OK)                                                            \
        rc = yvex_cuda_work_allocate(&work, &(field_), (size_t)(bytes_), NULL, 1, \
                                     (stage_), NULL, err)
    ALLOC(tokens_device, token_bytes, "cuda.sampling.greedy.tokens");
    ALLOC(values_device, selected_bytes, "cuda.sampling.greedy.values");
    ALLOC(ties_device, tie_bytes, "cuda.sampling.greedy.ties");
    ALLOC(status_device, sizeof(status), "cuda.sampling.greedy.status");
#undef ALLOC
    input = (CUdeviceptr)logits->data;
    if (rc == YVEX_OK) {
        void *params[] = {&input, &row_count, &row_width, &tokens_device,
                          &values_device, &ties_device, &status_device};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->argmax_f32_function, (unsigned int)row_count, 128u, 0u,
            params, "cuda.sampling.greedy.launch", err);
        launched = rc == YVEX_OK;
    }
#define READ(target_, source_, bytes_, stage_)                                    \
    if (rc == YVEX_OK)                                                            \
        rc = sampling_cuda_download(backend, (target_), (source_), (size_t)(bytes_), \
                                    (stage_), err)
    READ(&status, status_device, sizeof(status), "cuda.sampling.greedy.status");
    READ(selected_tokens, tokens_device, token_bytes, "cuda.sampling.greedy.tokens");
    READ(selected_values, values_device, selected_bytes, "cuda.sampling.greedy.values");
    READ(tie_counts, ties_device, tie_bytes, "cuda.sampling.greedy.ties");
#undef READ
    rc = sampling_cuda_complete(
        backend, launched || (work.count && yvex_cuda_launch_stream(backend)),
        &device_wide, "cuda.sampling.greedy.sync", rc, err);
    for (row = 0ull; rc == YVEX_OK && row < row_count; ++row)
        if (status || selected_tokens[row] >= row_width ||
            !isfinite(selected_values[row]) || !tie_counts[row])
            rc = sampling_cuda_refuse(
                err, YVEX_ERR_FORMAT,
                "greedy CUDA row selection produced invalid bounded facts");
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc == YVEX_OK) {
        facts->d2h_bytes = temporary_bytes;
        facts->kernel_launches = 1ull;
        facts->download_count = 4ull;
        facts->stream_synchronizations = (unsigned long long)!device_wide;
        facts->device_synchronizations = (unsigned long long)device_wide;
        facts->activation_bytes = value_bytes;
        facts->temporary_bytes = temporary_bytes;
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}

static int sampling_cuda_select_stochastic(
    yvex_backend *backend, const yvex_device_tensor *logits,
    unsigned long long vocabulary_size,
    const yvex_runtime_sampling_policy *policy, unsigned int random_value,
    yvex_backend_sampling_result *result,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    CUdeviceptr tokens = 0ull, values = 0ull, probabilities = 0ull;
    CUdeviceptr deviations = 0ull, counts_device = 0ull, statistics_device = 0ull;
    CUdeviceptr output_values_device = 0ull, selection_device = 0ull, status_device = 0ull;
    unsigned long long padded, required, extent, logits_bytes;
    unsigned long long counts[YVEX_CUDA_SAMPLING_COUNT_FIELDS] = {0};
    double statistics[YVEX_CUDA_SAMPLING_STATISTIC_FIELDS] = {0};
    float output_values[YVEX_CUDA_SAMPLING_VALUE_FIELDS] = {0};
    unsigned int selection[YVEX_CUDA_SAMPLING_SELECTION_FIELDS] = {0};
    int status = 0, device_wide = 0, launched = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (result) memset(result, 0, sizeof(*result));
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !result || !facts || !policy ||
        policy->strategy != YVEX_SAMPLING_STRATEGY_STOCHASTIC ||
        !backend_tensor_owner_is(backend, logits) || logits->dtype != YVEX_DTYPE_F32 ||
        !logits->is_written ||
        !yvex_core_u64_mul(vocabulary_size, sizeof(float), &logits_bytes) ||
        logits->bytes < logits_bytes ||
        !sampling_cuda_padded(vocabulary_size, &padded) ||
        sampling_cuda_workspace_required(vocabulary_size, &required, err) != YVEX_OK)
        return yvex_error_code(err) == YVEX_OK
                   ? sampling_cuda_refuse(err, YVEX_ERR_FORMAT,
                                          "stochastic CUDA sampling input is incompatible")
                   : yvex_error_code(err);
    rc = yvex_cuda_require_capability(
        backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, "cuda.sampling", err);
    if (rc == YVEX_OK) rc = yvex_cuda_set_current(backend, "cuda.sampling", err);
    if (rc == YVEX_OK &&
        (!backend->workspace_device_tensor || backend->workspace_bytes < required))
        rc = sampling_cuda_refuse(
            err, YVEX_ERR_NOMEM,
            "stochastic CUDA sampling workspace was not preflighted");
    backend_workspace_reset(backend);
    work.backend = backend; work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
#define ALLOC(target_, bytes_, zero_, stage_)                                      \
    do {                                                                           \
        if (rc == YVEX_OK)                                                         \
            rc = yvex_cuda_work_allocate(&work, &(target_), (size_t)(bytes_),     \
                                         NULL, (zero_), (stage_), NULL, err);      \
    } while (0)
    extent = padded * sizeof(unsigned int); ALLOC(tokens, extent, 0, "cuda.sampling.tokens");
    extent = padded * sizeof(float); ALLOC(values, extent, 0, "cuda.sampling.logits");
    extent = padded * sizeof(double); ALLOC(probabilities, extent, 0, "cuda.sampling.probabilities");
    ALLOC(deviations, extent, 0, "cuda.sampling.deviations");
    ALLOC(counts_device, sizeof(counts), 1, "cuda.sampling.counts");
    ALLOC(statistics_device, sizeof(statistics), 1, "cuda.sampling.statistics");
    ALLOC(output_values_device, sizeof(output_values), 1, "cuda.sampling.values");
    ALLOC(selection_device, sizeof(selection), 1, "cuda.sampling.selection");
    ALLOC(status_device, sizeof(status), 1, "cuda.sampling.status");
#undef ALLOC
    if (rc == YVEX_OK) {
        CUdeviceptr input = (CUdeviceptr)logits->data;
        double temperature = policy->temperature, top_p = policy->top_p;
        double min_p = policy->min_p, typical_p = policy->typical_p;
        void *params[] = {
            &input, &vocabulary_size, &padded, &temperature, (void *)&policy->top_k,
            &top_p, &min_p, &typical_p, &random_value, &tokens, &values,
            &probabilities, &deviations, &counts_device, &statistics_device,
            &output_values_device, &selection_device, &status_device};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->sample_stochastic_f32_function, 1u, CUDA_SAMPLING_BLOCK,
            0u, params, "cuda.sampling.launch", err);
        launched = rc == YVEX_OK;
    }
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(backend, &status, status_device,
                                    sizeof(status), "cuda.sampling.status", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(backend, counts, counts_device,
                                    sizeof(counts), "cuda.sampling.counts", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(backend, statistics, statistics_device,
                                    sizeof(statistics), "cuda.sampling.statistics", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(backend, output_values, output_values_device,
                                    sizeof(output_values), "cuda.sampling.values", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(backend, selection, selection_device,
                                    sizeof(selection), "cuda.sampling.selection", err);
    rc = sampling_cuda_complete(
        backend, launched || (work.count && yvex_cuda_launch_stream(backend)),
        &device_wide, "cuda.sampling.sync", rc, err);
    if (rc == YVEX_OK &&
        (status || !sampling_cuda_result_valid(
                       vocabulary_size, counts, statistics, output_values, selection)))
        rc = sampling_cuda_refuse(
            err, YVEX_ERR_FORMAT,
            "stochastic CUDA sampling produced invalid bounded facts");
    if (rc == YVEX_OK) {
        result->completed = 1;
        result->numeric_fallback_used =
            (int)selection[YVEX_CUDA_SAMPLING_NUMERIC_FALLBACK];
        result->selected_token_id = selection[YVEX_CUDA_SAMPLING_SELECTED_TOKEN];
        result->candidates_after_top_k = counts[YVEX_CUDA_SAMPLING_TOP_K_COUNT];
        result->candidates_after_min_p = counts[YVEX_CUDA_SAMPLING_MIN_P_COUNT];
        result->candidates_after_typical_p = counts[YVEX_CUDA_SAMPLING_TYPICAL_COUNT];
        result->candidates_after_top_p = counts[YVEX_CUDA_SAMPLING_TOP_P_COUNT];
        result->selected_logit = output_values[YVEX_CUDA_SAMPLING_SELECTED_LOGIT];
        result->maximum_logit = output_values[YVEX_CUDA_SAMPLING_MAXIMUM_LOGIT];
        result->selected_probability =
            statistics[YVEX_CUDA_SAMPLING_SELECTED_PROBABILITY];
        result->min_p_threshold = statistics[YVEX_CUDA_SAMPLING_MIN_P_THRESHOLD];
        result->entropy = statistics[YVEX_CUDA_SAMPLING_ENTROPY];
        result->typical_retained_mass = statistics[YVEX_CUDA_SAMPLING_TYPICAL_MASS];
        result->top_p_retained_mass = statistics[YVEX_CUDA_SAMPLING_TOP_P_MASS];
        result->normalization_error =
            statistics[YVEX_CUDA_SAMPLING_NORMALIZATION_ERROR];
        facts->d2h_bytes = sizeof(status) + sizeof(counts) + sizeof(statistics) +
                           sizeof(output_values) + sizeof(selection);
        facts->kernel_launches = 1ull; facts->download_count = 5ull;
        facts->stream_synchronizations = (unsigned long long)!device_wide;
        facts->device_synchronizations = (unsigned long long)device_wide;
        facts->activation_bytes = logits_bytes;
        facts->temporary_bytes = required;
        facts->compulsory_memory_facts_available = 1;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc != YVEX_OK) memset(result, 0, sizeof(*result));
    else yvex_error_clear(err);
    return rc;
}

static int sampling_cuda_speculation_result_valid(
    unsigned long long candidate_count, unsigned long long vocabulary_size,
    const unsigned int *committed,
    const unsigned long long result[YVEX_CUDA_SPECULATION_RESULT_FIELDS])
{
    unsigned long long index;
    if (result[YVEX_CUDA_SPECULATION_PROPOSED_COUNT] != candidate_count ||
        result[YVEX_CUDA_SPECULATION_ACCEPTED_COUNT] > candidate_count ||
        result[YVEX_CUDA_SPECULATION_REJECTED_COUNT] !=
            candidate_count - result[YVEX_CUDA_SPECULATION_ACCEPTED_COUNT] ||
        result[YVEX_CUDA_SPECULATION_COMMITTED_COUNT] !=
            result[YVEX_CUDA_SPECULATION_ACCEPTED_COUNT] + 1ull ||
        result[YVEX_CUDA_SPECULATION_REJECTION_INDEX] > candidate_count ||
        result[YVEX_CUDA_SPECULATION_ALL_ACCEPTED] > 1ull ||
        result[YVEX_CUDA_SPECULATION_CORRECTION_PRESENT] > 1ull ||
        result[YVEX_CUDA_SPECULATION_BONUS_PRESENT] > 1ull ||
        result[YVEX_CUDA_SPECULATION_CORRECTION_PRESENT] +
                result[YVEX_CUDA_SPECULATION_BONUS_PRESENT] !=
            1ull ||
        result[YVEX_CUDA_SPECULATION_ALL_ACCEPTED] !=
            result[YVEX_CUDA_SPECULATION_BONUS_PRESENT] ||
        (result[YVEX_CUDA_SPECULATION_CORRECTION_PRESENT] &&
         result[YVEX_CUDA_SPECULATION_REJECTION_INDEX] !=
             result[YVEX_CUDA_SPECULATION_ACCEPTED_COUNT]) ||
        (result[YVEX_CUDA_SPECULATION_BONUS_PRESENT] &&
         result[YVEX_CUDA_SPECULATION_ACCEPTED_COUNT] != candidate_count) ||
        result[YVEX_CUDA_SPECULATION_CORRECTION_TOKEN] >= vocabulary_size ||
        result[YVEX_CUDA_SPECULATION_CORRECTION_TOKEN] !=
            committed[result[YVEX_CUDA_SPECULATION_COMMITTED_COUNT] - 1ull])
        return 0;
    for (index = 0ull;
         index < result[YVEX_CUDA_SPECULATION_COMMITTED_COUNT]; ++index)
        if (committed[index] >= vocabulary_size) return 0;
    return 1;
}

static int sampling_cuda_accept_stochastic(
    yvex_backend *backend, const yvex_device_tensor *draft_logits,
    const yvex_device_tensor *target_logits, unsigned long long candidate_count,
    unsigned long long vocabulary_size,
    const yvex_runtime_sampling_policy *policy,
    const unsigned int *candidate_tokens, const double *acceptance_uniforms,
    double correction_uniform, unsigned int *committed_tokens,
    unsigned long long committed_capacity,
    yvex_backend_speculation_result *result,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    sampling_cuda_speculation_buffers buffers = {0};
    unsigned long long result_fields[YVEX_CUDA_SPECULATION_RESULT_FIELDS] = {0};
    unsigned long long padded, required, draft_values, target_values;
    unsigned long long draft_bytes, target_bytes, candidate_bytes;
    unsigned long long uniform_bytes, committed_bytes, result_bytes;
    unsigned long long activation_bytes, index;
    int status = 0, device_wide = 0, launched = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (result) memset(result, 0, sizeof(*result));
    if (facts) memset(facts, 0, sizeof(*facts));
    if (committed_tokens && candidate_count && candidate_count < ULLONG_MAX &&
        committed_capacity >= candidate_count + 1ull &&
        candidate_count + 1ull <= SIZE_MAX / sizeof(*committed_tokens))
        memset(committed_tokens, 0, (size_t)(candidate_count + 1ull) * sizeof(*committed_tokens));
    if (!state || !policy || !candidate_tokens || !acceptance_uniforms ||
        !committed_tokens || !result || !facts || !candidate_count ||
        candidate_count > UINT_MAX || !vocabulary_size ||
        vocabulary_size > UINT_MAX ||
        policy->strategy != YVEX_SAMPLING_STRATEGY_STOCHASTIC ||
        !yvex_sha256_hex_valid(policy->policy_identity) ||
        !isfinite(correction_uniform) || correction_uniform < 0.0 ||
        correction_uniform >= 1.0 ||
        committed_capacity < candidate_count + 1ull ||
        !backend_tensor_owner_is(backend, draft_logits) ||
        !backend_tensor_owner_is(backend, target_logits) ||
        draft_logits->dtype != YVEX_DTYPE_F32 ||
        target_logits->dtype != YVEX_DTYPE_F32 ||
        !draft_logits->is_written || !target_logits->is_written ||
        !sampling_cuda_padded(vocabulary_size, &padded) ||
        !yvex_core_u64_mul(candidate_count, vocabulary_size, &draft_values) ||
        !yvex_core_u64_mul(candidate_count + 1ull, vocabulary_size, &target_values) ||
        !yvex_core_u64_mul(draft_values, sizeof(float), &draft_bytes) ||
        !yvex_core_u64_mul(target_values, sizeof(float), &target_bytes) ||
        draft_logits->bytes < draft_bytes || target_logits->bytes < target_bytes ||
        !yvex_core_u64_mul(candidate_count, sizeof(*candidate_tokens), &candidate_bytes) ||
        !yvex_core_u64_mul(candidate_count, sizeof(*acceptance_uniforms), &uniform_bytes) ||
        !yvex_core_u64_mul(candidate_count + 1ull, sizeof(*committed_tokens), &committed_bytes) ||
        !yvex_core_u64_mul(YVEX_CUDA_SPECULATION_RESULT_FIELDS, sizeof(*result_fields),
                           &result_bytes) ||
        !yvex_core_u64_add(draft_bytes, target_bytes, &activation_bytes) ||
        sampling_cuda_speculation_workspace_required(
            vocabulary_size, candidate_count, &required, err) != YVEX_OK)
        return yvex_error_code(err) == YVEX_OK
            ? sampling_cuda_refuse(
                  err, YVEX_ERR_FORMAT,
                  "stochastic CUDA speculation input is incompatible")
            : yvex_error_code(err);
    for (index = 0ull; index < candidate_count; ++index)
        if (candidate_tokens[index] >= vocabulary_size ||
            !isfinite(acceptance_uniforms[index]) ||
            acceptance_uniforms[index] < 0.0 ||
            acceptance_uniforms[index] >= 1.0)
            return sampling_cuda_refuse(
                err, YVEX_ERR_FORMAT,
                "stochastic CUDA speculation draw or token is invalid");
    rc = yvex_cuda_require_capability(
        backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
        "cuda.sampling.speculation", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_set_current(backend, "cuda.sampling.speculation", err);
    if (rc == YVEX_OK &&
        (!backend->workspace_device_tensor || backend->workspace_bytes < required))
        rc = sampling_cuda_refuse(
            err, YVEX_ERR_NOMEM,
            "stochastic CUDA speculation workspace was not preflighted");
    backend_workspace_reset(backend);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    if (rc == YVEX_OK)
        rc = sampling_cuda_speculation_buffers_open(
            &work, &buffers, padded, vocabulary_size, candidate_bytes,
            candidate_tokens, uniform_bytes, acceptance_uniforms,
            committed_bytes, result_bytes, err);
    if (rc == YVEX_OK) {
        CUdeviceptr draft_input = (CUdeviceptr)draft_logits->data;
        CUdeviceptr target_input = (CUdeviceptr)target_logits->data;
        double temperature = policy->temperature, top_p = policy->top_p;
        double min_p = policy->min_p, typical_p = policy->typical_p;
        void *params[] = {
            &draft_input, &target_input, &candidate_count, &vocabulary_size,
            &padded, &temperature, (void *)&policy->top_k, &top_p, &min_p,
            &typical_p, &buffers.candidates, &buffers.uniforms,
            &correction_uniform, &buffers.tokens, &buffers.logits,
            &buffers.probabilities, &buffers.deviations, &buffers.draft_dense,
            &buffers.target_dense, &buffers.counts, &buffers.statistics,
            &buffers.output_values, &buffers.committed, &buffers.result,
            &buffers.status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->speculation_stochastic_f32_function, 1u,
            CUDA_SAMPLING_BLOCK, 0u, params,
            "cuda.sampling.speculation.launch", err);
        launched = rc == YVEX_OK;
    }
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(
            backend, &status, buffers.status, sizeof(status),
            "cuda.sampling.speculation.status", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(
            backend, committed_tokens, buffers.committed,
            (size_t)committed_bytes, "cuda.sampling.speculation.committed",
            err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(
            backend, result_fields, buffers.result, (size_t)result_bytes,
            "cuda.sampling.speculation.result", err);
    rc = sampling_cuda_complete(
        backend, launched || (work.count && yvex_cuda_launch_stream(backend)),
        &device_wide, "cuda.sampling.speculation.sync", rc, err);
    if (rc == YVEX_OK &&
        (status || !sampling_cuda_speculation_result_valid(
                       candidate_count, vocabulary_size, committed_tokens,
                       result_fields)))
        rc = sampling_cuda_refuse(
            err, YVEX_ERR_FORMAT,
            "stochastic CUDA speculation produced invalid bounded facts");
    if (rc == YVEX_OK) {
        result->completed = 1;
        result->proposed_count = result_fields[YVEX_CUDA_SPECULATION_PROPOSED_COUNT];
        result->accepted_draft_count = result_fields[YVEX_CUDA_SPECULATION_ACCEPTED_COUNT];
        result->rejected_draft_count = result_fields[YVEX_CUDA_SPECULATION_REJECTED_COUNT];
        result->committed_count = result_fields[YVEX_CUDA_SPECULATION_COMMITTED_COUNT];
        result->rejection_index = result_fields[YVEX_CUDA_SPECULATION_REJECTION_INDEX];
        result->all_candidates_accepted =
            (int)result_fields[YVEX_CUDA_SPECULATION_ALL_ACCEPTED];
        result->correction_present =
            (int)result_fields[YVEX_CUDA_SPECULATION_CORRECTION_PRESENT];
        result->bonus_present =
            (int)result_fields[YVEX_CUDA_SPECULATION_BONUS_PRESENT];
        result->correction_or_bonus_token_id =
            (unsigned int)result_fields[YVEX_CUDA_SPECULATION_CORRECTION_TOKEN];
        facts->h2d_bytes = candidate_bytes + uniform_bytes;
        facts->d2h_bytes = sizeof(status) + committed_bytes + result_bytes;
        facts->kernel_launches = 1ull;
        facts->upload_count = 2ull;
        facts->download_count = 3ull;
        facts->stream_synchronizations = (unsigned long long)!device_wide;
        facts->device_synchronizations = (unsigned long long)device_wide;
        facts->activation_bytes = activation_bytes;
        facts->temporary_bytes = required;
        facts->compulsory_memory_facts_available = 1;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc != YVEX_OK) {
        memset(result, 0, sizeof(*result));
        memset(committed_tokens, 0, (size_t)committed_bytes);
    } else {
        yvex_error_clear(err);
    }
    return rc;
}

static const yvex_backend_sampling_operations sampling_cuda_operations = {
    sampling_cuda_workspace_required,
    sampling_cuda_speculation_workspace_required,
    sampling_cuda_select_greedy_rows,
    sampling_cuda_select_stochastic,
    sampling_cuda_accept_stochastic
};

const yvex_backend_sampling_operations *yvex_cuda_sampling_operations_get(
    const yvex_backend *backend)
{
    const yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    return backend && yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA && state &&
                   state->argmax_f32_function && state->sample_stochastic_f32_function &&
                   state->speculation_stochastic_f32_function
               ? &sampling_cuda_operations
               : NULL;
}
