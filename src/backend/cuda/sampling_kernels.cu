/*
 * Keep full-vocabulary stochastic filtering and speculative acceptance on the device.
 *
 * The portable runtime owns RNG transactions and final publication. These kernels consume
 * explicit draws and return only bounded tokens and acceptance facts; dense probability rows
 * never become host authority.
 */
#include "src/backend/cuda/kernel_primitives.h"

static __device__ void sampling_sum_add(double *sum, double *correction,
                                        double value)
{
    double next = *sum + value;
    if (fabs(*sum) >= fabs(value))
        *correction += (*sum - next) + value;
    else
        *correction += (value - next) + *sum;
    *sum = next;
}

static __device__ int sampling_normalize_device(
    double *probabilities, unsigned long long count, double *maximum_error)
{
    double sum = 0.0, correction = 0.0, verified = 0.0;
    double verified_correction = 0.0;
    for (unsigned long long index = 0ull; index < count; ++index) {
        double probability = probabilities[index];
        if (!isfinite(probability) || probability < 0.0) return 0;
        sampling_sum_add(&sum, &correction, probability);
    }
    sum += correction;
    if (!isfinite(sum) || sum <= 0.0) return 0;
    for (unsigned long long index = 0ull; index < count; ++index) {
        probabilities[index] /= sum;
        sampling_sum_add(&verified, &verified_correction, probabilities[index]);
    }
    verified += verified_correction;
    if (!isfinite(verified)) return 0;
    double error = fabs(verified - 1.0);
    if (error > *maximum_error) *maximum_error = error;
    return 1;
}

static __device__ int sampling_candidate_before(
    const unsigned int *tokens, const double *probabilities,
    const double *deviations, unsigned long long left,
    unsigned long long right, unsigned int mode)
{
    if (tokens[left] == ~0u) return 0;
    if (tokens[right] == ~0u) return 1;
    if (mode == 2u) return tokens[left] < tokens[right];
    if (mode == 1u && deviations[left] != deviations[right])
        return deviations[left] < deviations[right];
    if (probabilities[left] != probabilities[right])
        return probabilities[left] > probabilities[right];
    return tokens[left] < tokens[right];
}

static __device__ void sampling_candidate_swap(
    unsigned int *tokens, float *logits, double *probabilities,
    double *deviations, unsigned long long left, unsigned long long right)
{
    unsigned int token = tokens[left];
    float logit = logits[left];
    double probability = probabilities[left], deviation = deviations[left];
    tokens[left] = tokens[right];
    logits[left] = logits[right];
    probabilities[left] = probabilities[right];
    deviations[left] = deviations[right];
    tokens[right] = token;
    logits[right] = logit;
    probabilities[right] = probability;
    deviations[right] = deviation;
}

static __device__ void sampling_sort_device(
    unsigned int *tokens, float *logits, double *probabilities,
    double *deviations, unsigned long long count,
    unsigned long long padded, unsigned int mode)
{
    unsigned long long index, partner, width, stride;
    for (index = count + threadIdx.x; index < padded; index += blockDim.x) {
        tokens[index] = ~0u;
        logits[index] = 0.0f;
        probabilities[index] = -1.0;
        deviations[index] = 1.0 / 0.0;
    }
    __syncthreads();
    for (width = 2ull; width <= padded; width <<= 1ull) {
        for (stride = width >> 1ull; stride; stride >>= 1ull) {
            for (index = threadIdx.x; index < padded; index += blockDim.x) {
                partner = index ^ stride;
                if (partner > index) {
                    int forward = (index & width) == 0ull;
                    int swap = forward
                        ? sampling_candidate_before(tokens, probabilities, deviations,
                                                    partner, index, mode)
                        : sampling_candidate_before(tokens, probabilities, deviations,
                                                    index, partner, mode);
                    if (swap)
                        sampling_candidate_swap(tokens, logits, probabilities,
                                                deviations, index, partner);
                }
            }
            __syncthreads();
        }
        if (width == padded) break;
    }
}

static __device__ unsigned long long sampling_compact_positive(
    unsigned int *tokens, float *logits, double *probabilities,
    double *deviations, unsigned long long count)
{
    unsigned long long write = 0ull;
    for (unsigned long long read = 0ull; read < count; ++read) {
        if (probabilities[read] <= 0.0) continue;
        if (write != read) {
            tokens[write] = tokens[read];
            logits[write] = logits[read];
            probabilities[write] = probabilities[read];
            deviations[write] = deviations[read];
        }
        write++;
    }
    return write;
}

static __device__ void sampling_filter_device(
    const float *values, unsigned long long vocabulary_size,
    unsigned long long padded_size, double temperature,
    unsigned long long top_k, double top_p, double min_p,
    double typical_p, unsigned int *tokens, float *logits,
    double *probabilities, double *deviations,
    unsigned long long *count, unsigned long long *counts,
    double *statistics, float *output_values, int *status)
{
    if (!threadIdx.x) {
        double maximum_scaled = -1.7976931348623157e+308;
        float maximum_logit = -3.402823466e+38F;
        for (unsigned long long index = 0ull; index < vocabulary_size; ++index) {
            double scaled = (double)values[index] / temperature;
            if (!isfinite(values[index]) || !isfinite(scaled)) {
                atomicCAS(status, 0, 1);
                break;
            }
            tokens[index] = (unsigned int)index;
            logits[index] = values[index];
            deviations[index] = 0.0;
            probabilities[index] = scaled;
            if (scaled > maximum_scaled) maximum_scaled = scaled;
            if (values[index] > maximum_logit) maximum_logit = values[index];
        }
        for (unsigned long long index = 0ull;
             !*status && index < vocabulary_size; ++index)
            probabilities[index] = exp(probabilities[index] - maximum_scaled);
        *count = vocabulary_size;
        if (!*status &&
            !sampling_normalize_device(
                probabilities, *count,
                &statistics[YVEX_CUDA_SAMPLING_NORMALIZATION_ERROR]))
            atomicCAS(status, 0, 1);
        if (!*status)
            *count = sampling_compact_positive(
                tokens, logits, probabilities, deviations, *count);
        output_values[YVEX_CUDA_SAMPLING_MAXIMUM_LOGIT] = maximum_logit;
    }
    __syncthreads();
    if (*status || !*count) return;
    if (top_k && top_k < *count)
        sampling_sort_device(tokens, logits, probabilities, deviations,
                             *count, padded_size, 0u);
    if (!threadIdx.x) {
        if (top_k && top_k < *count) {
            *count = top_k;
            if (!sampling_normalize_device(
                    probabilities, *count,
                    &statistics[YVEX_CUDA_SAMPLING_NORMALIZATION_ERROR]))
                atomicCAS(status, 0, 1);
        }
        counts[YVEX_CUDA_SAMPLING_TOP_K_COUNT] = *count;
        if (!*status && min_p > 0.0) {
            double maximum = 0.0;
            unsigned long long write = 0ull;
            for (unsigned long long index = 0ull; index < *count; ++index)
                if (probabilities[index] > maximum) maximum = probabilities[index];
            statistics[YVEX_CUDA_SAMPLING_MIN_P_THRESHOLD] = min_p * maximum;
            for (unsigned long long read = 0ull; read < *count; ++read) {
                if (probabilities[read] <
                    statistics[YVEX_CUDA_SAMPLING_MIN_P_THRESHOLD])
                    continue;
                if (write != read) {
                    tokens[write] = tokens[read];
                    logits[write] = logits[read];
                    probabilities[write] = probabilities[read];
                    deviations[write] = deviations[read];
                }
                write++;
            }
            *count = write;
            if (!*count ||
                !sampling_normalize_device(
                    probabilities, *count,
                    &statistics[YVEX_CUDA_SAMPLING_NORMALIZATION_ERROR]))
                atomicCAS(status, 0, 1);
        }
        counts[YVEX_CUDA_SAMPLING_MIN_P_COUNT] = *count;
    }
    __syncthreads();
    if (*status || !*count) return;
    if (typical_p < 1.0) {
        if (!threadIdx.x) {
            double entropy = 0.0, correction = 0.0;
            for (unsigned long long index = 0ull; index < *count; ++index)
                sampling_sum_add(
                    &entropy, &correction,
                    -probabilities[index] * log(probabilities[index]));
            entropy += correction;
            statistics[YVEX_CUDA_SAMPLING_ENTROPY] = entropy;
            if (!isfinite(entropy)) atomicCAS(status, 0, 1);
            for (unsigned long long index = 0ull;
                 !*status && index < *count; ++index)
                deviations[index] =
                    fabs(-log(probabilities[index]) - entropy);
        }
        __syncthreads();
        if (*status) return;
        sampling_sort_device(tokens, logits, probabilities, deviations,
                             *count, padded_size, 1u);
        if (!threadIdx.x) {
            double mass = 0.0;
            unsigned long long retained = 0ull;
            while (retained < *count && mass < typical_p)
                mass += probabilities[retained++];
            if (!retained) retained = 1ull;
            statistics[YVEX_CUDA_SAMPLING_TYPICAL_MASS] = mass;
            *count = retained;
            if (!sampling_normalize_device(
                    probabilities, *count,
                    &statistics[YVEX_CUDA_SAMPLING_NORMALIZATION_ERROR]))
                atomicCAS(status, 0, 1);
        }
    }
    __syncthreads();
    if (*status || !*count) return;
    if (!threadIdx.x)
        counts[YVEX_CUDA_SAMPLING_TYPICAL_COUNT] = *count;
    if (top_p < 1.0)
        sampling_sort_device(tokens, logits, probabilities, deviations,
                             *count, padded_size, 0u);
    if (!threadIdx.x) {
        if (top_p < 1.0) {
            double mass = 0.0;
            unsigned long long retained = 0ull;
            while (retained < *count && mass < top_p)
                mass += probabilities[retained++];
            if (!retained) retained = 1ull;
            statistics[YVEX_CUDA_SAMPLING_TOP_P_MASS] = mass;
            *count = retained;
        }
        if (!sampling_normalize_device(
                probabilities, *count,
                &statistics[YVEX_CUDA_SAMPLING_NORMALIZATION_ERROR]))
            atomicCAS(status, 0, 1);
        counts[YVEX_CUDA_SAMPLING_TOP_P_COUNT] = *count;
    }
    __syncthreads();
}

static __device__ void sampling_dense_device(
    const unsigned int *tokens, const double *probabilities,
    unsigned long long count, float *dense,
    unsigned long long vocabulary_size)
{
    for (unsigned long long index = threadIdx.x;
         index < vocabulary_size; index += blockDim.x)
        dense[index] = 0.0f;
    __syncthreads();
    for (unsigned long long index = threadIdx.x; index < count;
         index += blockDim.x)
        dense[tokens[index]] = (float)probabilities[index];
    __syncthreads();
}

static __device__ unsigned int sampling_dense_draw(
    const float *target, const float *draft,
    unsigned long long vocabulary_size, double uniform, int residual)
{
    double total = 0.0, cumulative = 0.0;
    for (unsigned long long index = 0ull; index < vocabulary_size; ++index)
        total += residual ? fmax((double)target[index] - draft[index], 0.0)
                          : (double)target[index];
    if (residual && total <= 1e-8) {
        residual = 0;
        total = 0.0;
        for (unsigned long long index = 0ull; index < vocabulary_size; ++index)
            total += target[index];
    }
    if (total <= 0.0) return ~0u;
    for (unsigned long long index = 0ull; index < vocabulary_size; ++index) {
        double value = residual
            ? fmax((double)target[index] - draft[index], 0.0)
            : (double)target[index];
        cumulative += value / total;
        if (uniform < cumulative || index + 1ull == vocabulary_size)
            return (unsigned int)index;
    }
    return ~0u;
}

extern "C" __global__ void yvex_sample_stochastic_f32(
    const float *values, unsigned long long vocabulary_size,
    unsigned long long padded_size, double temperature,
    unsigned long long top_k, double top_p, double min_p,
    double typical_p, unsigned int random_value, unsigned int *tokens,
    float *logits, double *probabilities, double *deviations,
    unsigned long long *counts, double *statistics, float *output_values,
    unsigned int *selection, int *status)
{
    __shared__ unsigned long long count;
    if (!status || blockIdx.x || blockDim.x != 256u || !values ||
        !vocabulary_size || !padded_size || padded_size < vocabulary_size ||
        !tokens || !logits || !probabilities || !deviations || !counts ||
        !statistics || !output_values || !selection) {
        if (status && !blockIdx.x && !threadIdx.x) atomicCAS(status, 0, 2);
        return;
    }
    sampling_filter_device(
        values, vocabulary_size, padded_size, temperature, top_k, top_p,
        min_p, typical_p, tokens, logits, probabilities, deviations, &count,
        counts, statistics, output_values, status);
    if (*status || !count) return;
    sampling_sort_device(tokens, logits, probabilities, deviations,
                         count, padded_size, 2u);
    if (!threadIdx.x) {
        double uniform = ((double)random_value + 0.5) / 4294967296.0;
        double cumulative = 0.0;
        unsigned long long selected = count - 1ull;
        selection[YVEX_CUDA_SAMPLING_NUMERIC_FALLBACK] = 1u;
        for (unsigned long long index = 0ull; index < count; ++index) {
            cumulative += probabilities[index];
            if (uniform < cumulative) {
                selected = index;
                selection[YVEX_CUDA_SAMPLING_NUMERIC_FALLBACK] = 0u;
                break;
            }
        }
        if (!isfinite(probabilities[selected]) ||
            probabilities[selected] <= 0.0) {
            atomicCAS(status, 0, 1);
            return;
        }
        selection[YVEX_CUDA_SAMPLING_SELECTED_TOKEN] = tokens[selected];
        output_values[YVEX_CUDA_SAMPLING_SELECTED_LOGIT] = logits[selected];
        statistics[YVEX_CUDA_SAMPLING_SELECTED_PROBABILITY] =
            probabilities[selected];
    }
}

extern "C" __global__ void yvex_speculation_stochastic_f32(
    const float *draft_values, const float *target_values,
    unsigned long long candidate_count, unsigned long long vocabulary_size,
    unsigned long long padded_size, double temperature,
    unsigned long long top_k, double top_p, double min_p, double typical_p,
    const unsigned int *candidate_tokens, const double *acceptance_uniforms,
    double correction_uniform, unsigned int *tokens, float *logits,
    double *probabilities, double *deviations, float *draft_dense,
    float *target_dense, unsigned long long *counts, double *statistics,
    float *output_values, unsigned int *committed_tokens,
    unsigned long long *result, int *status)
{
    __shared__ unsigned long long count;
    if (!status || blockIdx.x || blockDim.x != 256u || !draft_values ||
        !target_values || !candidate_count || !vocabulary_size ||
        !padded_size || padded_size < vocabulary_size || !candidate_tokens ||
        !acceptance_uniforms || !tokens || !logits || !probabilities ||
        !deviations || !draft_dense || !target_dense || !counts ||
        !statistics || !output_values || !committed_tokens || !result ||
        !isfinite(correction_uniform) || correction_uniform < 0.0 ||
        correction_uniform >= 1.0) {
        if (status && !blockIdx.x && !threadIdx.x) atomicCAS(status, 0, 2);
        return;
    }
    if (!threadIdx.x) {
        result[YVEX_CUDA_SPECULATION_PROPOSED_COUNT] = candidate_count;
        result[YVEX_CUDA_SPECULATION_REJECTION_INDEX] = candidate_count;
    }
    __syncthreads();
    for (unsigned long long row = 0ull; row < candidate_count; ++row) {
        sampling_filter_device(
            draft_values + row * vocabulary_size, vocabulary_size,
            padded_size, temperature, top_k, top_p, min_p, typical_p,
            tokens, logits, probabilities, deviations, &count, counts,
            statistics, output_values, status);
        if (*status || !count) return;
        sampling_dense_device(
            tokens, probabilities, count, draft_dense, vocabulary_size);
        sampling_filter_device(
            target_values + row * vocabulary_size, vocabulary_size,
            padded_size, temperature, top_k, top_p, min_p, typical_p,
            tokens, logits, probabilities, deviations, &count, counts,
            statistics, output_values, status);
        if (*status || !count) return;
        sampling_dense_device(
            tokens, probabilities, count, target_dense, vocabulary_size);
        if (!threadIdx.x) {
            unsigned int candidate = candidate_tokens[row];
            double q, p, ratio;
            if (candidate >= vocabulary_size ||
                !isfinite(acceptance_uniforms[row]) ||
                acceptance_uniforms[row] < 0.0 ||
                acceptance_uniforms[row] >= 1.0) {
                atomicCAS(status, 0, 2);
            } else {
                q = draft_dense[candidate];
                p = target_dense[candidate];
                if (!isfinite(q) || q <= 0.0 || !isfinite(p) || p < 0.0) {
                    atomicCAS(status, 0, 1);
                } else {
                    ratio = fmin(1.0, p / q);
                    if (acceptance_uniforms[row] < ratio) {
                        committed_tokens[row] = candidate;
                        result[YVEX_CUDA_SPECULATION_ACCEPTED_COUNT] = row + 1ull;
                    } else {
                        unsigned int correction = sampling_dense_draw(
                            target_dense, draft_dense, vocabulary_size,
                            correction_uniform, 1);
                        if (correction == ~0u) {
                            atomicCAS(status, 0, 1);
                        } else {
                            committed_tokens[row] = correction;
                            result[YVEX_CUDA_SPECULATION_REJECTED_COUNT] =
                                candidate_count - row;
                            result[YVEX_CUDA_SPECULATION_COMMITTED_COUNT] =
                                row + 1ull;
                            result[YVEX_CUDA_SPECULATION_REJECTION_INDEX] = row;
                            result[YVEX_CUDA_SPECULATION_CORRECTION_PRESENT] = 1ull;
                            result[YVEX_CUDA_SPECULATION_CORRECTION_TOKEN] = correction;
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (*status) return;
        if (result[YVEX_CUDA_SPECULATION_CORRECTION_PRESENT]) return;
    }
    sampling_filter_device(
        target_values + candidate_count * vocabulary_size, vocabulary_size,
        padded_size, temperature, top_k, top_p, min_p, typical_p, tokens,
        logits, probabilities, deviations, &count, counts, statistics,
        output_values, status);
    if (*status || !count) return;
    sampling_dense_device(
        tokens, probabilities, count, target_dense, vocabulary_size);
    if (!threadIdx.x) {
        unsigned int bonus = sampling_dense_draw(
            target_dense, NULL, vocabulary_size, correction_uniform, 0);
        if (bonus == ~0u) {
            atomicCAS(status, 0, 1);
            return;
        }
        committed_tokens[candidate_count] = bonus;
        result[YVEX_CUDA_SPECULATION_ACCEPTED_COUNT] = candidate_count;
        result[YVEX_CUDA_SPECULATION_COMMITTED_COUNT] = candidate_count + 1ull;
        result[YVEX_CUDA_SPECULATION_ALL_ACCEPTED] = 1ull;
        result[YVEX_CUDA_SPECULATION_BONUS_PRESENT] = 1ull;
        result[YVEX_CUDA_SPECULATION_CORRECTION_TOKEN] = bonus;
    }
}
