/* Portable scalar-transition selective SSD authority over caller-owned transactional state. */
#include <yvex/internal/state_space.h>

#include <yvex/internal/core.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int ssd_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "graph.selective-ssd", reason);
    return status;
}


typedef struct { const float *data; unsigned long long values; } ssd_span;

static int ssd_overlap(ssd_span left, ssd_span right)
{
    uintptr_t a = (uintptr_t)left.data, b = (uintptr_t)right.data;
    unsigned long long na, nb;

    if (!a || !b || !left.values || !right.values) return 0;
    if (!yvex_core_u64_mul(left.values, sizeof(float), &na) ||
        !yvex_core_u64_mul(right.values, sizeof(float), &nb) ||
        na > UINTPTR_MAX - a || nb > UINTPTR_MAX - b) return 1;
    return a < b + nb && b < a + na;
}

static int ssd_admit(const yvex_selective_ssd_geometry *p,
                     const yvex_selective_ssd_cpu_request *r, yvex_error *err)
{
    unsigned long long projection, output, updates;
    ssd_span writable[4], readable[9];
    size_t i, j;

    if (!r || !r->token_count ||
        !yvex_core_u64_mul(r->token_count, p->projection_width, &projection) ||
        !yvex_core_u64_mul(r->token_count, p->width, &output) ||
        !yvex_core_u64_mul(r->token_count, p->recurrent_state_values, &updates) ||
        projection > SIZE_MAX / sizeof(float) || output > SIZE_MAX / sizeof(float) ||
        !r->projection || r->projection_capacity < projection ||
        !r->convolution_weight || r->convolution_weight_capacity < p->convolution_state_values ||
        !r->convolution_bias || r->convolution_bias_capacity < p->convolution_width ||
        !r->decay_log || r->decay_log_capacity < p->requirement.heads ||
        !r->skip || r->skip_capacity < p->requirement.heads ||
        !r->time_bias || r->time_bias_capacity < p->requirement.heads ||
        !r->normalization_weight || r->normalization_weight_capacity < p->width ||
        !r->next_state.convolution || r->next_state.convolution_capacity < p->convolution_state_values ||
        !r->next_state.recurrent || r->next_state.recurrent_capacity < p->recurrent_state_values ||
        !r->workspace || r->workspace_capacity < p->convolution_width ||
        !r->output || r->output_capacity < output ||
        (!!r->state.convolution != !!r->state.recurrent))
        return ssd_refuse(err, YVEX_ERR_BOUNDS, "selective SSD buffers are incomplete");
    writable[0] = (ssd_span){r->next_state.convolution, p->convolution_state_values};
    writable[1] = (ssd_span){r->next_state.recurrent, p->recurrent_state_values};
    writable[2] = (ssd_span){r->workspace, p->convolution_width};
    writable[3] = (ssd_span){r->output, output};
    readable[0] = (ssd_span){r->projection, projection};
    readable[1] = (ssd_span){r->state.convolution, p->convolution_state_values};
    readable[2] = (ssd_span){r->state.recurrent, p->recurrent_state_values};
    readable[3] = (ssd_span){r->convolution_weight, p->convolution_state_values};
    readable[4] = (ssd_span){r->convolution_bias, p->convolution_width};
    readable[5] = (ssd_span){r->decay_log, p->requirement.heads};
    readable[6] = (ssd_span){r->skip, p->requirement.heads};
    readable[7] = (ssd_span){r->time_bias, p->requirement.heads};
    readable[8] = (ssd_span){r->normalization_weight, p->width};
    for (i = 0; i < 4u; ++i) {
        for (j = 0; j < i; ++j)
            if (ssd_overlap(writable[i], writable[j]))
                return ssd_refuse(err, YVEX_ERR_INVALID_ARG, "SSD writable buffers alias");
        for (j = 0; j < 9u; ++j)
            if (ssd_overlap(writable[i], readable[j]))
                return ssd_refuse(err, YVEX_ERR_INVALID_ARG, "SSD committed input aliases candidate output");
    }
    return YVEX_OK;
}

static float ssd_silu(float value) { return value / (1.0f + expf(-value)); }

static int ssd_convolution(const yvex_selective_ssd_geometry *p,
    const yvex_selective_ssd_cpu_request *r, const float *projected)
{
    unsigned long long channel, k, kernel = p->requirement.convolution_kernel;

    for (channel = 0; channel < p->convolution_width; ++channel) {
        float *state = r->next_state.convolution + channel * kernel;
        const float *weight = r->convolution_weight + channel * kernel;
        float value = r->convolution_bias[channel];
        for (k = 0; k + 1u < kernel; ++k) state[k] = state[k + 1u];
        state[kernel - 1u] = projected[p->width + channel];
        for (k = 0; k < kernel; ++k) value += state[k] * weight[k];
        r->workspace[channel] = ssd_silu(value);
        if (!isfinite(r->workspace[channel])) return 0;
    }
    return 1;
}

static int ssd_recurrence(const yvex_selective_ssd_geometry *p,
    const yvex_selective_ssd_cpu_request *r, const float *projected, float *output)
{
    const yvex_selective_ssd_requirement *q = &p->requirement;
    unsigned long long head, channel, n;

    for (head = 0; head < q->heads; ++head) {
        unsigned long long group = head / (q->heads / q->groups);
        const float *b = r->workspace + p->width + group * q->state_dimension;
        const float *c = b + q->groups * q->state_dimension;
        float raw_dt = projected[p->projection_width - q->heads + head] + r->time_bias[head];
        float dt = fmaxf(raw_dt, 0.0f) + log1pf(expf(-fabsf(raw_dt)));
        float decay;
        if (!isfinite(raw_dt) || !isfinite(r->decay_log[head])) return 0;
        dt = fmaxf(dt, (float)q->time_step_minimum);
        if (!q->time_step_unbounded) dt = fminf(dt, (float)q->time_step_maximum);
        decay = expf(-expf(r->decay_log[head]) * dt);
        for (channel = 0; channel < q->head_dimension; ++channel) {
            unsigned long long index = head * q->head_dimension + channel;
            float x = r->workspace[index], value = r->skip[head] * x;
            float *state = r->next_state.recurrent + index * q->state_dimension;
            for (n = 0; n < q->state_dimension; ++n) {
                state[n] = decay * state[n] + dt * b[n] * x;
                value += state[n] * c[n];
                if (!isfinite(state[n])) return 0;
            }
            output[index] = value;
            if (!isfinite(value)) return 0;
        }
    }
    return 1;
}

static int ssd_normalize(const yvex_selective_ssd_geometry *p,
    const yvex_selective_ssd_cpu_request *r, const float *gate, float *output)
{
    unsigned long long group, index, width = p->width / p->requirement.normalization_groups;
    for (group = 0; group < p->requirement.normalization_groups; ++group) {
        float sum = 0.0f, scale;
        for (index = group * width; index < (group + 1u) * width; ++index) {
            if (!p->requirement.norm_before_gate) output[index] *= ssd_silu(gate[index]);
            sum += output[index] * output[index];
        }
        scale = 1.0f / sqrtf(sum / (float)width + (float)p->requirement.normalization_epsilon);
        if (!isfinite(sum)) return 0;
        for (index = group * width; index < (group + 1u) * width; ++index) {
            output[index] *= scale * r->normalization_weight[index];
            if (p->requirement.norm_before_gate) output[index] *= ssd_silu(gate[index]);
            if (!isfinite(output[index])) return 0;
        }
    }
    return 1;
}

int yvex_selective_ssd_execute_cpu(
    const yvex_selective_ssd_geometry *p, const yvex_selective_ssd_cpu_request *r,
    yvex_selective_ssd_cpu_result *result, yvex_error *err)
{
    unsigned long long token;
    int rc;

    if (result) memset(result, 0, sizeof(*result));
    if (!result) return ssd_refuse(err, YVEX_ERR_INVALID_ARG, "SSD result is required");
    rc = yvex_selective_ssd_geometry_validate(p, err);
    if (rc == YVEX_OK) rc = ssd_admit(p, r, err);
    if (rc != YVEX_OK) return rc;
    if (r->state.convolution) {
        memcpy(r->next_state.convolution, r->state.convolution,
               (size_t)p->convolution_state_values * sizeof(float));
        memcpy(r->next_state.recurrent, r->state.recurrent,
               (size_t)p->recurrent_state_values * sizeof(float));
    } else {
        memset(r->next_state.convolution, 0, (size_t)p->convolution_state_values * sizeof(float));
        memset(r->next_state.recurrent, 0, (size_t)p->recurrent_state_values * sizeof(float));
    }
    for (token = 0; token < r->token_count; ++token) {
        const float *projection = r->projection + token * p->projection_width;
        float *output = r->output + token * p->width;
        if (r->cancel_requested && r->cancel_requested(r->cancel_context)) {
            result->cancelled = 1;
            return ssd_refuse(err, YVEX_ERR_CANCELLED, "selective SSD scan cancelled");
        }
        if (!ssd_convolution(p, r, projection) || !ssd_recurrence(p, r, projection, output) ||
            !ssd_normalize(p, r, projection, output))
            return ssd_refuse(err, YVEX_ERR_FORMAT, "selective SSD produced non-finite values");
        result->completed_tokens++;
        result->state_updates += p->recurrent_state_values;
    }
    result->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
