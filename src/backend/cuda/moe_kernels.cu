/*
 * Execute routed and shared MoE kernels from device-resident expert packs.
 *
 * This independently compiled family owns row/expert scheduling while canonical qtype
 * arithmetic remains shared through the CUDA kernel-primitives interface.
 */
#include "src/backend/cuda/kernel_primitives.h"
extern "C" __global__ void yvex_moe_route(
    const float *logits, const float *bias, const unsigned long long *hash_experts,
    unsigned int router_class, unsigned long long routed_experts,
    unsigned long long topk, int normalize, double scaling,
    float *scores, unsigned long long *selected, float *weights, int *status)
{
    if (blockIdx.x || threadIdx.x || !status || *status) return;
    if (!logits || !scores || !selected || !weights || !routed_experts ||
        routed_experts > 256ull || !topk || topk > 16ull || topk > routed_experts ||
        router_class > 1u || !isfinite(scaling) || scaling <= 0.0 ||
        (router_class == 0u && !hash_experts) || (router_class == 1u && !bias)) {
        atomicCAS(status, 0, 2);
        return;
    }
    for (unsigned long long expert = 0ull; expert < routed_experts; ++expert) {
        double value = (double)logits[expert];
        double softplus = value > 0.0 ? value + log1p(exp(-value)) : log1p(exp(value));
        double score = sqrt(softplus);
        if (!isfinite(score)) {
            atomicCAS(status, 0, 1);
            return;
        }
        scores[expert] = (float)score;
    }
    for (unsigned long long rank = 0ull; rank < topk; ++rank) {
        unsigned long long chosen = ~0ull;
        if (router_class == 0u) {
            chosen = hash_experts[rank];
        } else {
            for (unsigned long long candidate = 0ull; candidate < routed_experts; ++candidate) {
                int used = 0;
                for (unsigned long long prior = 0ull; prior < rank; ++prior)
                    if (selected[prior] == candidate) used = 1;
                double candidate_score = (double)scores[candidate] + (double)bias[candidate];
                double chosen_score = chosen == ~0ull
                    ? -INFINITY : (double)scores[chosen] + (double)bias[chosen];
                if (!used && (chosen == ~0ull || candidate_score > chosen_score ||
                              (candidate_score == chosen_score && candidate < chosen)))
                    chosen = candidate;
            }
        }
        if (chosen >= routed_experts) {
            atomicCAS(status, 0, 2);
            return;
        }
        for (unsigned long long prior = 0ull; prior < rank; ++prior)
            if (selected[prior] == chosen) {
                atomicCAS(status, 0, 2);
                return;
            }
        selected[rank] = chosen;
        weights[rank] = scores[chosen];
    }
    double total = 0.0;
    for (unsigned long long rank = 0ull; rank < topk; ++rank)
        total += (double)weights[rank];
    if (normalize && (!isfinite(total) || total <= 0.0)) {
        atomicCAS(status, 0, 1);
        return;
    }
    for (unsigned long long rank = 0ull; rank < topk; ++rank) {
        double value = (double)weights[rank];
        if (normalize) value /= total;
        value *= scaling;
        weights[rank] = (float)value;
    }
}
extern "C" __global__ void yvex_moe_grouped_up(
    const unsigned char *gate, unsigned long long gate_row_bytes,
    unsigned long long gate_expert_bytes, unsigned int gate_qtype,
    const unsigned char *up, unsigned long long up_row_bytes,
    unsigned long long up_expert_bytes, unsigned int up_qtype,
    const unsigned long long *selected, unsigned long long topk,
    unsigned long long expert_count, const unsigned char *input,
    unsigned long long input_extent, int q8_input,
    unsigned long long intermediate_width,
    double limit, float *intermediate, int *status)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned long long pair = (unsigned long long)blockIdx.x * 8ull +
                              (unsigned long long)(threadIdx.x >> 5u);
    unsigned long long rank = pair / intermediate_width;
    unsigned long long row = pair % intermediate_width;
    if (!status || *status || rank >= topk) return;
    unsigned long long expert = selected ? selected[rank] : ~0ull;
    if (!gate || !up || !input || !intermediate || expert >= expert_count ||
        !gate_row_bytes || !up_row_bytes || !input_extent || !intermediate_width ||
        !isfinite(limit) || limit <= 0.0) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    const unsigned char *gate_row = gate + expert * gate_expert_bytes + row * gate_row_bytes;
    const unsigned char *up_row = up + expert * up_expert_bytes + row * up_row_bytes;
    float g = 0.0f, u = 0.0f;
    if (q8_input) {
        if (gate_row_bytes % input_extent || up_row_bytes % input_extent) {
            if (!lane) atomicCAS(status, 0, 2);
            return;
        }
        g = q8_warp_dot(gate_row, input, input_extent,
                        gate_row_bytes / input_extent, gate_qtype);
        u = q8_warp_dot(up_row, input, input_extent,
                        up_row_bytes / input_extent, up_qtype);
    } else {
        g = qtype_warp_dot(gate_row, (const float *)input, input_extent, gate_qtype, status);
        u = qtype_warp_dot(up_row, (const float *)input, input_extent, up_qtype, status);
    }
    if (!lane && !*status) {
        g = fminf(g, (float)limit); u = fmaxf((float)-limit, fminf(u, (float)limit));
        float silu = g >= 0.0f ? g / (1.0f + expf(-g)) : g * expf(g) / (1.0f + expf(g));
        float value = float_to_bf16_rne(silu * u);
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else intermediate[rank * intermediate_width + row] = value;
    }
}

extern "C" __global__ void yvex_moe_grouped_down(
    const unsigned char *down, unsigned long long row_bytes,
    unsigned long long expert_bytes, unsigned int qtype,
    const unsigned long long *selected, const float *weights,
    unsigned long long topk, unsigned long long expert_count,
    const unsigned char *intermediate, unsigned long long intermediate_extent,
    int q8_input,
    unsigned long long hidden, float *routed, int *status)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned long long row = (unsigned long long)blockIdx.x * 8ull +
                             (unsigned long long)(threadIdx.x >> 5u);
    float total = 0.0f;
    if (!status || *status || row >= hidden) return;
    if (!down || !selected || !weights || !intermediate || !routed ||
        !row_bytes || !intermediate_extent || !topk ||
        (q8_input && row_bytes % intermediate_extent)) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    for (unsigned long long rank = 0ull; rank < topk; ++rank) {
        unsigned long long expert = selected[rank];
        if (expert >= expert_count) { if (!lane) atomicCAS(status, 0, 2); return; }
        const unsigned char *weight = down + expert * expert_bytes + row * row_bytes;
        float dot = 0.0f;
        if (q8_input) {
            const unsigned char *activation = intermediate + rank * intermediate_extent *
                                              YVEX_CUDA_Q8_K_BYTES;
            dot = q8_warp_dot(weight, activation, intermediate_extent,
                              row_bytes / intermediate_extent, qtype);
        } else
            dot = qtype_warp_dot(weight, (const float *)intermediate +
                rank * intermediate_extent, intermediate_extent, qtype, status);
        if (!lane && !*status) {
            float value = float_to_bf16_rne(dot);
            total = __fadd_rn(total, __fmul_rn(value, weights[rank]));
        }
    }
    if (!lane && !*status) routed[row] = total;
}

/* Route every row independently while retaining one launch and one bounded publication. */
extern "C" __global__ void yvex_moe_route_rows(
    const float *logits, const float *bias, const int32_t *hash_table,
    const unsigned int *token_ids, unsigned int router_class,
    unsigned long long row_count, unsigned long long routed_experts,
    unsigned long long topk, unsigned long long hash_rows,
    unsigned long long hash_columns, unsigned long long hash_row_bytes,
    int normalize, double scaling, float *scores,
    unsigned long long *selected, float *weights, int *status)
{
    unsigned long long row = (unsigned long long)blockIdx.x;
    if (!status || *status || threadIdx.x || row >= row_count) return;
    if (!logits || !scores || !selected || !weights || !row_count ||
        !routed_experts || routed_experts > 256ull || !topk || topk > 16ull ||
        topk > routed_experts || router_class > 1u || !isfinite(scaling) ||
        scaling <= 0.0 || (router_class == 0u &&
        (!hash_table || !token_ids || hash_columns < topk ||
         hash_row_bytes < hash_columns * sizeof(int32_t))) ||
        (router_class == 1u && !bias)) {
        atomicCAS(status, 0, 2);
        return;
    }
    const float *row_logits = logits + row * routed_experts;
    float *row_scores = scores + row * routed_experts;
    unsigned long long *row_selected = selected + row * topk;
    float *row_weights = weights + row * topk;
    for (unsigned long long expert = 0ull; expert < routed_experts; ++expert) {
        double value = (double)row_logits[expert];
        double softplus = value > 0.0 ? value + log1p(exp(-value)) : log1p(exp(value));
        double score = sqrt(softplus);
        if (!isfinite(score)) { atomicCAS(status, 0, 1); return; }
        row_scores[expert] = (float)score;
    }
    for (unsigned long long rank = 0ull; rank < topk; ++rank) {
        unsigned long long chosen = ~0ull;
        if (router_class == 0u) {
            unsigned int token = token_ids[row];
            if ((unsigned long long)token >= hash_rows) {
                atomicCAS(status, 0, 2);
                return;
            }
            const int32_t *hash_row = (const int32_t *)
                ((const unsigned char *)hash_table + (unsigned long long)token * hash_row_bytes);
            int32_t value = hash_row[rank];
            chosen = value < 0 ? ~0ull : (unsigned long long)value;
        } else {
            for (unsigned long long candidate = 0ull; candidate < routed_experts; ++candidate) {
                int used = 0;
                for (unsigned long long prior = 0ull; prior < rank; ++prior)
                    if (row_selected[prior] == candidate) used = 1;
                double candidate_score = (double)row_scores[candidate] + (double)bias[candidate];
                double chosen_score = chosen == ~0ull
                    ? -INFINITY : (double)row_scores[chosen] + (double)bias[chosen];
                if (!used && (chosen == ~0ull || candidate_score > chosen_score ||
                              (candidate_score == chosen_score && candidate < chosen)))
                    chosen = candidate;
            }
        }
        if (chosen >= routed_experts) { atomicCAS(status, 0, 2); return; }
        for (unsigned long long prior = 0ull; prior < rank; ++prior)
            if (row_selected[prior] == chosen) { atomicCAS(status, 0, 2); return; }
        row_selected[rank] = chosen;
        row_weights[rank] = row_scores[chosen];
    }
    double total = 0.0;
    for (unsigned long long rank = 0ull; rank < topk; ++rank)
        total += (double)row_weights[rank];
    if (normalize && (!isfinite(total) || total <= 0.0)) {
        atomicCAS(status, 0, 1);
        return;
    }
    for (unsigned long long rank = 0ull; rank < topk; ++rank) {
        double value = (double)row_weights[rank];
        if (normalize) value /= total;
        row_weights[rank] = (float)(value * scaling);
    }
}

/* Build a deterministic expert-major pair order without host routing authority. */
extern "C" __global__ void yvex_moe_pair_order(
    const unsigned long long *selected, unsigned long long row_count,
    unsigned long long topk, unsigned long long expert_count,
    unsigned long long *order, unsigned long long *unique_experts, int *status)
{
    if (!status || *status || blockIdx.x || threadIdx.x) return;
    if (!selected || !row_count || !topk || !expert_count || !order || !unique_experts) {
        atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long emitted = 0ull, unique = 0ull;
    for (unsigned long long expert = 0ull; expert < expert_count; ++expert) {
        unsigned long long before = emitted;
        for (unsigned long long pair = 0ull; pair < row_count * topk; ++pair) {
            if (selected[pair] >= expert_count) {
                atomicCAS(status, 0, 2);
                return;
            }
            if (selected[pair] == expert) order[emitted++] = pair;
        }
        if (emitted != before) unique++;
    }
    if (emitted != row_count * topk) {
        atomicCAS(status, 0, 2);
        return;
    }
    *unique_experts = unique;
}

extern "C" __global__ void yvex_moe_grouped_up_rows(
    const unsigned char *gate, unsigned long long gate_row_bytes,
    unsigned long long gate_expert_bytes, unsigned int gate_qtype,
    const unsigned char *up, unsigned long long up_row_bytes,
    unsigned long long up_expert_bytes, unsigned int up_qtype,
    const unsigned long long *selected, const unsigned long long *order,
    unsigned long long pair_count, unsigned long long topk,
    unsigned long long expert_count, const unsigned char *input,
    unsigned long long input_extent, int q8_input,
    unsigned long long intermediate_width, double limit,
    float *intermediate, int *status)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned long long task = (unsigned long long)blockIdx.x * 8ull +
                              (unsigned long long)(threadIdx.x >> 5u);
    unsigned long long ordered_pair = task / intermediate_width;
    unsigned long long output_row = task % intermediate_width;
    if (!status || *status || ordered_pair >= pair_count) return;
    unsigned long long source_pair = order ? order[ordered_pair] : ordered_pair;
    unsigned long long source_row = source_pair / topk;
    unsigned long long expert = selected ? selected[source_pair] : 0ull;
    if (!gate || !up || !input || !intermediate || expert >= expert_count ||
        source_pair >= pair_count || !topk || !gate_row_bytes || !up_row_bytes ||
        !input_extent || !intermediate_width || !isfinite(limit) || limit <= 0.0) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    const unsigned char *gate_row = gate + expert * gate_expert_bytes +
                                    output_row * gate_row_bytes;
    const unsigned char *up_row = up + expert * up_expert_bytes +
                                  output_row * up_row_bytes;
    const unsigned char *activation = input + source_row * input_extent *
                                      (q8_input ? YVEX_CUDA_Q8_K_BYTES : sizeof(float));
    float g, u;
    if (q8_input) {
        if (gate_row_bytes % input_extent || up_row_bytes % input_extent) {
            if (!lane) atomicCAS(status, 0, 2);
            return;
        }
        g = q8_warp_dot(gate_row, activation, input_extent,
                        gate_row_bytes / input_extent, gate_qtype);
        u = q8_warp_dot(up_row, activation, input_extent,
                        up_row_bytes / input_extent, up_qtype);
    } else {
        g = qtype_warp_dot(gate_row, (const float *)activation,
                           input_extent, gate_qtype, status);
        u = qtype_warp_dot(up_row, (const float *)activation,
                           input_extent, up_qtype, status);
    }
    if (!lane && !*status) {
        g = fminf(g, (float)limit);
        u = fmaxf((float)-limit, fminf(u, (float)limit));
        float silu = g >= 0.0f ? g / (1.0f + expf(-g))
                               : g * expf(g) / (1.0f + expf(g));
        float value = float_to_bf16_rne(silu * u);
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else intermediate[ordered_pair * intermediate_width + output_row] = value;
    }
}

extern "C" __global__ void yvex_moe_grouped_down_rows(
    const unsigned char *down, unsigned long long row_bytes,
    unsigned long long expert_bytes, unsigned int qtype,
    const unsigned long long *selected, const float *weights,
    const unsigned long long *order, unsigned long long pair_count,
    unsigned long long topk, unsigned long long expert_count,
    const unsigned char *intermediate, unsigned long long intermediate_extent,
    int q8_input, unsigned long long hidden, float *pair_outputs, int *status)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned long long task = (unsigned long long)blockIdx.x * 8ull +
                              (unsigned long long)(threadIdx.x >> 5u);
    unsigned long long ordered_pair = task / hidden;
    unsigned long long output_row = task % hidden;
    if (!status || *status || ordered_pair >= pair_count) return;
    unsigned long long source_pair = order ? order[ordered_pair] : ordered_pair;
    unsigned long long expert = selected ? selected[source_pair] : 0ull;
    if (!down || !intermediate || !pair_outputs || expert >= expert_count ||
        source_pair >= pair_count || !topk || !row_bytes || !intermediate_extent ||
        (q8_input && row_bytes % intermediate_extent)) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    const unsigned char *weight = down + expert * expert_bytes + output_row * row_bytes;
    const unsigned char *activation = intermediate + ordered_pair * intermediate_extent *
                                      (q8_input ? YVEX_CUDA_Q8_K_BYTES : sizeof(float));
    float dot = q8_input
        ? q8_warp_dot(weight, activation, intermediate_extent,
                      row_bytes / intermediate_extent, qtype)
        : qtype_warp_dot(weight, (const float *)activation,
                         intermediate_extent, qtype, status);
    if (!lane && !*status) {
        float route_weight = weights ? weights[source_pair] : 1.0f;
        float value = __fmul_rn(float_to_bf16_rne(dot), route_weight);
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else pair_outputs[source_pair * hidden + output_row] = value;
    }
}

extern "C" __global__ void yvex_moe_reduce_rows(
    const float *pair_outputs, unsigned long long row_count,
    unsigned long long topk, unsigned long long hidden,
    float *output, int *status)
{
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!status || *status || index >= row_count * hidden) return;
    if (!pair_outputs || !row_count || !topk || !hidden || !output) {
        atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long row = index / hidden, lane = index % hidden;
    float total = 0.0f;
    for (unsigned long long rank = 0ull; rank < topk; ++rank)
        total = __fadd_rn(total, pair_outputs[(row * topk + rank) * hidden + lane]);
    if (!isfinite(total)) atomicCAS(status, 0, 1);
    else output[index] = total;
}

extern "C" __global__ void yvex_moe_combine_rows(
    const float *routed, const float *shared, unsigned long long count,
    float *combined, int *status)
{
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!status || *status || index >= count) return;
    if (!routed || !shared || !combined || !count) {
        atomicCAS(status, 0, 2);
        return;
    }
    float value = float_to_bf16_rne(__fadd_rn(routed[index], shared[index]));
    if (!isfinite(value)) atomicCAS(status, 0, 1);
    else combined[index] = value;
}

extern "C" __global__ void yvex_moe_swiglu(
    const float *gate, const float *up, unsigned long long count,
    double limit, float *output, int *status)
{
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!status || *status || index >= count) return;
    if (!gate || !up || !output || !isfinite(limit) || limit <= 0.0) {
        atomicCAS(status, 0, 2);
        return;
    }
    double g = fmin((double)gate[index], limit);
    double u = fmax(-limit, fmin((double)up[index], limit));
    double silu = g >= 0.0 ? g / (1.0 + exp(-g)) : g * exp(g) / (1.0 + exp(g));
    float value = float_to_bf16_rne((float)(silu * u));
    if (!isfinite(value)) atomicCAS(status, 0, 1);
    else output[index] = value;
}

extern "C" __global__ void yvex_moe_accumulate(
    const float *expert, unsigned long long count, float weight,
    float *aggregate, int *status)
{
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!status || *status || index >= count) return;
    if (!expert || !aggregate || !isfinite(weight)) {
        atomicCAS(status, 0, 2);
        return;
    }
    float value = __fadd_rn(aggregate[index], __fmul_rn(expert[index], weight));
    if (!isfinite(value)) atomicCAS(status, 0, 1);
    else aggregate[index] = value;
}
