/* Exact F32 physical kernels for the backend-neutral gated-delta sequence mixer. */
#include "src/backend/cuda/kernel_primitives.h"

static __device__ __forceinline__ float gated_delta_silu(float value)
{
    if (value >= 0.0f) return value / (1.0f + expf(-value));
    {
        float exponential = expf(value);
        return value * exponential / (1.0f + exponential);
    }
}

static __device__ __forceinline__ float gated_delta_sigmoid(float value)
{
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    {
        float exponential = expf(value);
        return exponential / (1.0f + exponential);
    }
}

static __device__ __forceinline__ float gated_delta_softplus(float value)
{
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return log1pf(expf(value));
}

static __device__ __forceinline__ float gated_delta_finite_or_zero(
    float value, int *status)
{
    if (isfinite(value)) return value;
    atomicCAS(status, 0, 1);
    return 0.0f;
}

extern "C" __global__ void yvex_gated_delta_convolution_f32(
    const float *projected_qkv, const float *weight,
    const float *committed_state, float *candidate_state, float *activated,
    unsigned long long token_offset, unsigned long long token_count,
    unsigned long long width, unsigned long long kernel,
    int initialize, int has_committed, int *status)
{
    unsigned long long channel =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long history, tap, token;
    float *channel_state;
    if (!projected_qkv || !weight || !candidate_state || !activated || !status ||
        !token_count || !width || kernel < 2ull ||
        (initialize != 0 && initialize != 1) ||
        (has_committed != 0 && has_committed != 1) ||
        (initialize && has_committed && !committed_state)) {
        if (blockIdx.x == 0u && threadIdx.x == 0u && status)
            atomicCAS(status, 0, 1);
        return;
    }
    if (channel >= width) return;
    history = kernel - 1ull;
    channel_state = candidate_state + channel * history;
    if (initialize) {
        const float *source = has_committed
            ? committed_state + channel * history : NULL;
        for (tap = 0ull; tap < history; ++tap)
            channel_state[tap] = source
                ? gated_delta_finite_or_zero(source[tap], status) : 0.0f;
    }
    for (token = 0ull; token < token_count; ++token) {
        float input = gated_delta_finite_or_zero(
            projected_qkv[(token_offset + token) * width + channel], status);
        float sum = gated_delta_finite_or_zero(
            weight[channel * kernel + history], status) * input;
        for (tap = 0ull; tap < history; ++tap)
            sum += gated_delta_finite_or_zero(
                       weight[channel * kernel + tap], status) *
                   channel_state[tap];
        sum = gated_delta_finite_or_zero(sum, status);
        activated[token * width + channel] = gated_delta_silu(sum);
        for (tap = 1ull; tap < history; ++tap)
            channel_state[tap - 1ull] = channel_state[tap];
        channel_state[history - 1ull] = input;
    }
}

extern "C" __global__ void yvex_gated_delta_recurrence_f32(
    const float *activated, const float *projected_output_gate,
    const float *projected_beta, const float *projected_decay,
    const float *decay_log, const float *time_bias,
    const float *normalization_weight, const float *committed_state,
    float *candidate_state, float *output,
    unsigned long long token_offset, unsigned long long token_count,
    unsigned long long query_heads, unsigned long long key_heads,
    unsigned long long value_heads, unsigned long long key_dimension,
    unsigned long long value_dimension, unsigned long long query_width,
    unsigned long long key_width, unsigned long long value_width,
    float qk_epsilon, float output_epsilon, float query_scale,
    int normalization_one_plus, int initialize, int has_committed, int *status)
{
    __shared__ float query[128], key[128], first_reduction[128];
    __shared__ float second_reduction[128], beta_value, decay_factor;
    unsigned int lane = threadIdx.x;
    unsigned long long value_head = blockIdx.x;
    unsigned long long query_repeat, key_repeat, query_head, key_head;
    unsigned long long state_base, key_lane, token;
    float *state;
    if (!activated || !projected_output_gate || !projected_beta ||
        !projected_decay || !decay_log || !time_bias ||
        !normalization_weight || !candidate_state || !output || !status ||
        blockDim.x != 128u || !token_count || !query_heads || !key_heads ||
        !value_heads || value_heads % query_heads || value_heads % key_heads ||
        !key_dimension || key_dimension > 128ull || !value_dimension ||
        value_dimension > 128ull || query_width != query_heads * key_dimension ||
        key_width != key_heads * key_dimension ||
        value_width != value_heads * value_dimension ||
        !(qk_epsilon > 0.0f) || !(output_epsilon > 0.0f) ||
        !(query_scale > 0.0f) ||
        (normalization_one_plus != 0 && normalization_one_plus != 1) ||
        (initialize != 0 && initialize != 1) ||
        (has_committed != 0 && has_committed != 1) ||
        (initialize && has_committed && !committed_state)) {
        if (blockIdx.x == 0u && lane == 0u && status)
            atomicCAS(status, 0, 1);
        return;
    }
    if (value_head >= value_heads) return;
    query_repeat = value_heads / query_heads;
    key_repeat = value_heads / key_heads;
    query_head = value_head / query_repeat;
    key_head = value_head / key_repeat;
    state_base = value_head * key_dimension * value_dimension;
    state = candidate_state + state_base;
    if (initialize && lane < value_dimension) {
        for (key_lane = 0ull; key_lane < key_dimension; ++key_lane) {
            unsigned long long index = key_lane * value_dimension + lane;
            state[index] = has_committed
                ? gated_delta_finite_or_zero(
                      committed_state[state_base + index], status)
                : 0.0f;
        }
    }
    __syncthreads();
    for (token = 0ull; token < token_count; ++token) {
        const float *token_values =
            activated + token * (query_width + key_width + value_width);
        float query_lane = lane < key_dimension
            ? gated_delta_finite_or_zero(
                  token_values[query_head * key_dimension + lane], status)
            : 0.0f;
        float key_lane_value = lane < key_dimension
            ? gated_delta_finite_or_zero(
                  token_values[query_width + key_head * key_dimension + lane],
                  status)
            : 0.0f;
        float raw = 0.0f;
        first_reduction[lane] = query_lane * query_lane;
        second_reduction[lane] = key_lane_value * key_lane_value;
        __syncthreads();
        for (unsigned int stride = 64u; stride; stride >>= 1u) {
            if (lane < stride) {
                first_reduction[lane] += first_reduction[lane + stride];
                second_reduction[lane] += second_reduction[lane + stride];
            }
            __syncthreads();
        }
        if (lane < key_dimension) {
            query[lane] = query_lane * query_scale /
                sqrtf(first_reduction[0] + qk_epsilon);
            key[lane] = key_lane_value /
                sqrtf(second_reduction[0] + qk_epsilon);
        }
        if (lane == 0u) {
            unsigned long long head_index =
                (token_offset + token) * value_heads + value_head;
            float beta_input = gated_delta_finite_or_zero(
                projected_beta[head_index], status);
            float decay_input = gated_delta_finite_or_zero(
                projected_decay[head_index], status);
            float log_value = gated_delta_finite_or_zero(
                decay_log[value_head], status);
            float bias = gated_delta_finite_or_zero(
                time_bias[value_head], status);
            beta_value = gated_delta_sigmoid(beta_input);
            decay_factor = expf(
                -expf(log_value) * gated_delta_softplus(decay_input + bias));
            decay_factor = gated_delta_finite_or_zero(decay_factor, status);
        }
        __syncthreads();
        if (lane < value_dimension) {
            float memory = 0.0f;
            float value = gated_delta_finite_or_zero(
                token_values[query_width + key_width +
                             value_head * value_dimension + lane], status);
            float delta;
            for (key_lane = 0ull; key_lane < key_dimension; ++key_lane) {
                unsigned long long index = key_lane * value_dimension + lane;
                float state_value = gated_delta_finite_or_zero(
                    state[index], status) * decay_factor;
                state[index] = state_value;
                memory += state_value * key[key_lane];
            }
            delta = (value - memory) * beta_value;
            for (key_lane = 0ull; key_lane < key_dimension; ++key_lane) {
                unsigned long long index = key_lane * value_dimension + lane;
                state[index] += key[key_lane] * delta;
                raw += query[key_lane] * state[index];
            }
            raw = gated_delta_finite_or_zero(raw, status);
        }
        first_reduction[lane] = raw * raw;
        __syncthreads();
        for (unsigned int stride = 64u; stride; stride >>= 1u) {
            if (lane < stride)
                first_reduction[lane] += first_reduction[lane + stride];
            __syncthreads();
        }
        if (lane < value_dimension) {
            unsigned long long output_index =
                (token_offset + token) * value_width +
                value_head * value_dimension + lane;
            float inverse = 1.0f /
                sqrtf(first_reduction[0] / (float)value_dimension +
                      output_epsilon);
            float gate = gated_delta_finite_or_zero(
                projected_output_gate[output_index], status);
            float weight = gated_delta_finite_or_zero(
                normalization_weight[lane], status);
            if (normalization_one_plus) weight += 1.0f;
            output[output_index] = gated_delta_finite_or_zero(
                raw * inverse * weight * gated_delta_silu(gate), status);
        }
        __syncthreads();
    }
}
