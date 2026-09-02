/* Exact portable Gated DeltaNet recurrence used as CPU execution and numerical authority. */
#include <yvex/internal/sequence_mixer.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>

static int mixer_execute_refuse(yvex_error *err, yvex_status status,
                                const char *reason)
{
    yvex_error_set(err, status, "graph.sequence-mixer.cpu", reason);
    return status;
}

static float mixer_silu(float value)
{
    return value / (1.0f + expf(-value));
}

static float mixer_softplus(float value)
{
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return log1pf(expf(value));
}

static int mixer_request_valid(const yvex_gated_delta_plan *plan,
                               const yvex_gated_delta_cpu_request *request)
{
    unsigned long long qkv_values, output_values, head_values, weight_values;

    if (!plan || !request || !request->token_count || !request->projected_qkv ||
        !request->projected_output_gate || !request->projected_beta ||
        !request->projected_decay || !request->convolution_weight ||
        !request->decay_log || !request->time_bias ||
        !request->normalization_weight || !request->next_state.convolution ||
        !request->next_state.recurrent || !request->output ||
        plan->convolution_state_bytes > SIZE_MAX ||
        plan->recurrent_state_bytes > SIZE_MAX ||
        request->next_state.convolution_capacity < plan->convolution_state_values ||
        request->next_state.recurrent_capacity < plan->recurrent_state_values ||
        !yvex_core_u64_mul(request->token_count, plan->qkv_width, &qkv_values) ||
        !yvex_core_u64_mul(request->token_count, plan->value_width, &output_values) ||
        !yvex_core_u64_mul(request->token_count,
                           plan->requirement.value_heads, &head_values) ||
        !yvex_core_u64_mul(plan->qkv_width,
                           plan->requirement.convolution_kernel,
                           &weight_values) ||
        request->projected_qkv_capacity < qkv_values ||
        request->projected_output_gate_capacity < output_values ||
        request->projected_beta_capacity < head_values ||
        request->projected_decay_capacity < head_values ||
        request->convolution_weight_capacity < weight_values ||
        request->decay_log_capacity < plan->requirement.value_heads ||
        request->time_bias_capacity < plan->requirement.value_heads ||
        request->normalization_weight_capacity <
            plan->requirement.value_head_dimension ||
        request->output_capacity < output_values)
        return 0;
    if ((request->state.convolution == NULL) != (request->state.recurrent == NULL) ||
        request->state.convolution == request->next_state.convolution ||
        request->state.recurrent == request->next_state.recurrent)
        return 0;
    return 1;
}

static void mixer_state_initialize(const yvex_gated_delta_plan *plan,
                                   const yvex_gated_delta_cpu_request *request)
{
    if (request->state.convolution) {
        memcpy(request->next_state.convolution, request->state.convolution,
               (size_t)plan->convolution_state_bytes);
        memcpy(request->next_state.recurrent, request->state.recurrent,
               (size_t)plan->recurrent_state_bytes);
    } else {
        memset(request->next_state.convolution, 0,
               (size_t)plan->convolution_state_bytes);
        memset(request->next_state.recurrent, 0,
               (size_t)plan->recurrent_state_bytes);
    }
}

static void mixer_convolution_token(const yvex_gated_delta_plan *plan,
                                    const float *input, const float *weight,
                                    float *history, float *activated)
{
    unsigned long long channel, tap;
    unsigned long long taps = plan->requirement.convolution_kernel;
    unsigned long long history_width = taps - 1ull;

    for (channel = 0ull; channel < plan->qkv_width; ++channel) {
        float sum = weight[channel * taps + history_width] * input[channel];
        float *channel_history = history + channel * history_width;

        for (tap = 0ull; tap < history_width; ++tap)
            sum += weight[channel * taps + tap] * channel_history[tap];
        if (history_width > 1ull)
            memmove(channel_history, channel_history + 1,
                    (size_t)(history_width - 1ull) * sizeof(float));
        channel_history[history_width - 1ull] = input[channel];
        activated[channel] = mixer_silu(sum);
    }
}

static void mixer_normalize(const float *input, unsigned long long width,
                            float epsilon, float scale, float *output)
{
    unsigned long long lane;
    float sum = 0.0f;

    for (lane = 0ull; lane < width; ++lane) sum += input[lane] * input[lane];
    sum = scale / sqrtf(sum + epsilon);
    for (lane = 0ull; lane < width; ++lane) output[lane] = input[lane] * sum;
}

static int mixer_delta_token(const yvex_gated_delta_plan *plan,
                             const yvex_gated_delta_cpu_request *request,
                             unsigned long long token, const float *activated,
                             float *query, float *key, float *memory,
                             float *delta, float *raw_output)
{
    const yvex_gated_delta_requirement *r = &plan->requirement;
    const float *q_values = activated;
    const float *k_values = activated + plan->query_width;
    const float *v_values = activated + plan->query_width + plan->key_width;
    const float *z_values = request->projected_output_gate + token * plan->value_width;
    const float *beta_values = request->projected_beta + token * r->value_heads;
    const float *decay_values = request->projected_decay + token * r->value_heads;
    unsigned long long value_head, key_lane, value_lane;
    unsigned long long query_repeat = r->value_heads / r->query_heads;
    unsigned long long key_repeat = r->value_heads / r->key_heads;

    for (value_head = 0ull; value_head < r->value_heads; ++value_head) {
        unsigned long long query_head = value_head / query_repeat;
        unsigned long long key_head = value_head / key_repeat;
        float *state = request->next_state.recurrent +
            value_head * r->key_head_dimension * r->value_head_dimension;
        float beta = 1.0f / (1.0f + expf(-beta_values[value_head]));
        float decay = -expf(request->decay_log[value_head]) *
                      mixer_softplus(decay_values[value_head] +
                                     request->time_bias[value_head]);
        float decay_factor = expf(decay);

        mixer_normalize(q_values + query_head * r->key_head_dimension,
                        r->key_head_dimension,
                        (float)r->qk_normalization_epsilon,
                        (float)r->query_scale, query);
        mixer_normalize(k_values + key_head * r->key_head_dimension,
                        r->key_head_dimension,
                        (float)r->qk_normalization_epsilon, 1.0f, key);
        memset(memory, 0, (size_t)r->value_head_dimension * sizeof(float));
        for (key_lane = 0ull; key_lane < r->key_head_dimension; ++key_lane) {
            float key_value = key[key_lane];
            for (value_lane = 0ull; value_lane < r->value_head_dimension;
                 ++value_lane) {
                unsigned long long index =
                    key_lane * r->value_head_dimension + value_lane;
                state[index] *= decay_factor;
                memory[value_lane] += state[index] * key_value;
            }
        }
        for (value_lane = 0ull; value_lane < r->value_head_dimension;
             ++value_lane)
            delta[value_lane] =
                (v_values[value_head * r->value_head_dimension + value_lane] -
                 memory[value_lane]) * beta;
        memset(raw_output, 0,
               (size_t)r->value_head_dimension * sizeof(float));
        for (key_lane = 0ull; key_lane < r->key_head_dimension; ++key_lane) {
            float query_value = query[key_lane];
            float key_value = key[key_lane];
            for (value_lane = 0ull; value_lane < r->value_head_dimension;
                 ++value_lane) {
                unsigned long long index =
                    key_lane * r->value_head_dimension + value_lane;
                state[index] += key_value * delta[value_lane];
                raw_output[value_lane] += query_value * state[index];
            }
        }
        {
            float rms = 0.0f;
            float *output = request->output + token * plan->value_width +
                            value_head * r->value_head_dimension;
            for (value_lane = 0ull; value_lane < r->value_head_dimension;
                 ++value_lane)
                rms += raw_output[value_lane] * raw_output[value_lane];
            rms = 1.0f / sqrtf(rms / (float)r->value_head_dimension +
                               (float)r->output_normalization_epsilon);
            for (value_lane = 0ull; value_lane < r->value_head_dimension;
                 ++value_lane) {
                unsigned long long offset =
                    value_head * r->value_head_dimension + value_lane;
                float weight = request->normalization_weight[value_lane];
                if (r->output_normalization_weight_convention ==
                    YVEX_NORMALIZATION_WEIGHT_ONE_PLUS) weight += 1.0f;
                output[value_lane] = raw_output[value_lane] * rms *
                    weight *
                    mixer_silu(z_values[offset]);
                if (!isfinite(output[value_lane])) return 0;
            }
        }
    }
    return 1;
}

int yvex_gated_delta_execute_cpu(
    const yvex_gated_delta_plan *plan, const yvex_gated_delta_cpu_request *request,
    yvex_gated_delta_cpu_result *result, yvex_error *err)
{
    const yvex_gated_delta_requirement *r;
    float *scratch = NULL, *activated, *query, *key, *memory, *delta, *raw_output;
    unsigned long long key_scratch, value_scratch, scratch_values, token;
    unsigned long long matrix_updates, accumulated_values;
    int rc = YVEX_OK;

    if (result) memset(result, 0, sizeof(*result));
    if (!result)
        return mixer_execute_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "gated-delta CPU execution request is malformed");
    rc = yvex_gated_delta_plan_validate(plan, err);
    if (rc != YVEX_OK) return rc;
    if (!mixer_request_valid(plan, request))
        return mixer_execute_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "gated-delta CPU execution request is malformed");
    r = &plan->requirement;
    if (!yvex_core_u64_mul(2ull, r->key_head_dimension, &key_scratch) ||
        !yvex_core_u64_mul(3ull, r->value_head_dimension, &value_scratch) ||
        !yvex_core_u64_add(plan->qkv_width, key_scratch, &scratch_values) ||
        !yvex_core_u64_add(scratch_values, value_scratch, &scratch_values) ||
        !yvex_core_u64_mul(request->token_count, r->value_heads,
                           &matrix_updates) ||
        !yvex_core_u64_mul(matrix_updates, r->key_head_dimension,
                           &accumulated_values) ||
        !yvex_core_u64_mul(accumulated_values, r->value_head_dimension,
                           &accumulated_values) ||
        scratch_values > SIZE_MAX / sizeof(float))
        return mixer_execute_refuse(err, YVEX_ERR_BOUNDS,
                                    "gated-delta scratch geometry overflowed");
    scratch = (float *)calloc((size_t)scratch_values, sizeof(float));
    if (!scratch)
        return mixer_execute_refuse(err, YVEX_ERR_NOMEM,
                                    "gated-delta scratch allocation failed");
    activated = scratch;
    query = activated + plan->qkv_width;
    key = query + r->key_head_dimension;
    memory = key + r->key_head_dimension;
    delta = memory + r->value_head_dimension;
    raw_output = delta + r->value_head_dimension;
    mixer_state_initialize(plan, request);
    for (token = 0ull; token < request->token_count; ++token) {
        if (request->cancel_requested && request->cancel_requested(request->cancel_context)) {
            result->cancelled = 1;
            rc = YVEX_ERR_CANCELLED;
            break;
        }
        mixer_convolution_token(plan,
                                request->projected_qkv + token * plan->qkv_width,
                                request->convolution_weight,
                                request->next_state.convolution, activated);
        if (!mixer_delta_token(plan, request, token, activated, query, key,
                               memory, delta, raw_output)) {
            rc = YVEX_ERR_FORMAT;
            break;
        }
        result->token_count++;
    }
    if (rc == YVEX_OK) {
        result->output_values = request->token_count * plan->value_width;
        result->convolution_state_values = plan->convolution_state_values;
        result->recurrent_state_values = plan->recurrent_state_values;
        result->recurrent_matrix_updates = matrix_updates;
        result->accumulated_values = accumulated_values;
        result->complete = 1;
        yvex_error_clear(err);
    } else {
        memset(request->next_state.convolution, 0,
               (size_t)plan->convolution_state_bytes);
        memset(request->next_state.recurrent, 0,
               (size_t)plan->recurrent_state_bytes);
        if (rc == YVEX_ERR_CANCELLED)
            yvex_error_set(err, rc, "graph.sequence-mixer.cpu",
                           "gated-delta execution cancelled before publication");
        else
            yvex_error_set(err, rc, "graph.sequence-mixer.cpu",
                           "gated-delta execution produced a non-finite value");
    }
    free(scratch);
    return rc;
}
