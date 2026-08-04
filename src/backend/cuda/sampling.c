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

static int sampling_cuda_download(yvex_cuda_backend_state *state,
                                  void *target, CUdeviceptr source,
                                  size_t bytes, const char *stage,
                                  yvex_error *err)
{
    return yvex_cuda_status(
        &state->driver,
        state->driver.cuMemcpyDtoH_v2(target, source, bytes), stage, err);
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
    int status = 0, rc, cleanup_rc;
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
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            "cuda.sampling.sync", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(state, &status, status_device,
                                    sizeof(status), "cuda.sampling.status", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(state, counts, counts_device,
                                    sizeof(counts), "cuda.sampling.counts", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(state, statistics, statistics_device,
                                    sizeof(statistics), "cuda.sampling.statistics", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(state, output_values, output_values_device,
                                    sizeof(output_values), "cuda.sampling.values", err);
    if (rc == YVEX_OK)
        rc = sampling_cuda_download(state, selection, selection_device,
                                    sizeof(selection), "cuda.sampling.selection", err);
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
        facts->device_synchronizations = 1ull;
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

static const yvex_backend_sampling_operations sampling_cuda_operations = {
    sampling_cuda_workspace_required,
    sampling_cuda_select_stochastic
};

const yvex_backend_sampling_operations *yvex_backend_sampling_operations_get(
    const yvex_backend *backend)
{
    const yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    return backend && yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA &&
                   state && state->sample_stochastic_f32_function
               ? &sampling_cuda_operations
               : NULL;
}
