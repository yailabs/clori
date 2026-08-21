/*
 * Execute routed and shared MoE kernels from device-resident expert packs.
 *
 * This independently compiled family owns row/expert scheduling while canonical qtype
 * arithmetic remains shared through the CUDA kernel-primitives interface.
 */
#include "src/backend/cuda/kernel_primitives.h"
#include <yvex/internal/execution_batch.h>
static __device__ float moe_warp_dot(
    const unsigned char *weight, const unsigned char *activation, unsigned long long extent,
    unsigned long long row_bytes, unsigned int qtype, int q8_input, int *status)
{
    if (!q8_input)
        return qtype_warp_dot(weight, (const float *)activation, extent, qtype, status);
    float sum = q8_warp_dot(weight, activation, extent, row_bytes / extent, qtype);
    /* Only the exceptional row pays for serial FP64 recovery; finite rows retain DP4A order. */
    if (!(threadIdx.x & 31u) && !isfinite(sum)) {
        double recovered = 0.0;
        for (unsigned long long i = 0ull; i < extent * YVEX_CUDA_Q8_K_BLOCK; ++i) {
            const unsigned char *q8 = activation +
                (i / YVEX_CUDA_Q8_K_BLOCK) * YVEX_CUDA_Q8_K_BYTES;
            int quantized = (int)(signed char)q8[4ull + i % YVEX_CUDA_Q8_K_BLOCK];
            recovered += (double)qtype_value(weight, i, qtype) * __uint_as_float(qtype_load_u32(q8)) * quantized;
        }
        sum = (float)recovered;
    }
    return sum;
}
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
    const unsigned long long *selected, const float *weights,
    unsigned long long topk,
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
    }
    g = moe_warp_dot(gate_row, input, input_extent, gate_row_bytes,
                     gate_qtype, q8_input, status);
    u = moe_warp_dot(up_row, input, input_extent, up_row_bytes,
                     up_qtype, q8_input, status);
    if (!lane && !*status) {
        g = fminf(g, (float)limit); u = fmaxf((float)-limit, fminf(u, (float)limit));
        float silu = g >= 0.0f ? g / (1.0f + expf(-g)) : g * expf(g) / (1.0f + expf(g));
        float route_weight = weights ? weights[rank] : 1.0f;
        float value = float_to_bf16_rne(silu * u * route_weight);
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else intermediate[rank * intermediate_width + row] = value;
    }
}

extern "C" __global__ void yvex_moe_grouped_down(
    const unsigned char *down, unsigned long long row_bytes,
    unsigned long long expert_bytes, unsigned int qtype,
    const unsigned long long *selected,
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
    if (!down || !selected || !intermediate || !routed ||
        !row_bytes || !intermediate_extent || !topk ||
        (q8_input && row_bytes % intermediate_extent)) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    for (unsigned long long rank = 0ull; rank < topk; ++rank) {
        unsigned long long expert = selected[rank];
        if (expert >= expert_count) { if (!lane) atomicCAS(status, 0, 2); return; }
        const unsigned char *weight = down + expert * expert_bytes + row * row_bytes;
        const unsigned char *activation = intermediate + rank * intermediate_extent *
            (q8_input ? YVEX_CUDA_Q8_K_BYTES : sizeof(float));
        float dot = moe_warp_dot(weight, activation, intermediate_extent,
                                 row_bytes, qtype, q8_input, status);
        if (!lane && !*status) {
            total = __fadd_rn(total, float_to_bf16_rne(dot));
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
    __shared__ float route_scores[256], rank_scores[256];
    __shared__ unsigned long long rank_experts[256];
    unsigned long long row = (unsigned long long)blockIdx.x;
    unsigned int thread = threadIdx.x;
    if (!status || row >= row_count) return;
    if (thread == 0u) {
        if (*status == 0 && (!logits || !scores || !selected || !weights || !row_count ||
            !routed_experts || routed_experts > 256ull || !topk || topk > 16ull ||
            topk > routed_experts || router_class > 1u || !isfinite(scaling) ||
            scaling <= 0.0 || (router_class == 0u &&
            (!hash_table || !token_ids || hash_columns < topk ||
             hash_row_bytes < hash_columns * sizeof(int32_t))) ||
            (router_class == 1u && !bias))) {
            atomicCAS(status, 0, 2);
        }
    }
    __syncthreads();
    if (*status) return;
    const float *row_logits = logits + row * routed_experts;
    float *row_scores = scores + row * routed_experts, *row_weights = weights + row * topk;
    unsigned long long *row_selected = selected + row * topk;
    /* Learned top-k stays expert-parallel with source tie-breaking and weight rank order. */
    for (unsigned long long expert = thread; expert < routed_experts;
         expert += blockDim.x) {
        double value = (double)row_logits[expert];
        double softplus = value > 0.0 ? value + log1p(exp(-value)) : log1p(exp(value));
        float score = (float)sqrt(softplus);
        if (!isfinite(score)) atomicCAS(status, 0, 1);
        route_scores[expert] = score;
        row_scores[expert] = score;
    }
    __syncthreads();
    if (*status) return;
    if (router_class == 0u && thread) return;
    if (router_class == 0u) {
        for (unsigned long long rank = 0ull; rank < topk; ++rank) {
            unsigned int token = token_ids[row];
            if ((unsigned long long)token >= hash_rows) {
                atomicCAS(status, 0, 2);
                return;
            }
            const int32_t *hash_row = (const int32_t *)((const unsigned char *)hash_table +
                (unsigned long long)token * hash_row_bytes);
            unsigned long long chosen = hash_row[rank] < 0 ? ~0ull : (unsigned long long)hash_row[rank];
            if (chosen >= routed_experts) { atomicCAS(status, 0, 2); return; }
            for (unsigned long long prior = 0ull; prior < rank; ++prior)
                if (row_selected[prior] == chosen) {
                    atomicCAS(status, 0, 2);
                    return;
                }
            row_selected[rank] = chosen;
            row_weights[rank] = route_scores[chosen];
        }
    } else {
        for (unsigned long long rank = 0ull; rank < topk; ++rank) {
            int used = 0;
            if ((unsigned long long)thread < routed_experts)
                for (unsigned long long prior = 0ull; prior < rank; ++prior)
                    if (row_selected[prior] == (unsigned long long)thread) used = 1;
            rank_scores[thread] = !used && (unsigned long long)thread < routed_experts
                                      ? route_scores[thread] + bias[thread] : -INFINITY;
            rank_experts[thread] = (unsigned long long)thread;
            __syncthreads();
            for (unsigned int stride = blockDim.x >> 1u; stride; stride >>= 1u) {
                if (thread < stride) {
                    float right_score = rank_scores[thread + stride];
                    unsigned long long right_expert = rank_experts[thread + stride];
                    if (right_score > rank_scores[thread] ||
                        (right_score == rank_scores[thread] &&
                         right_expert < rank_experts[thread])) {
                        rank_scores[thread] = right_score;
                        rank_experts[thread] = right_expert;
                    }
                }
                __syncthreads();
            }
            if (thread == 0u) {
                unsigned long long chosen = rank_experts[0];
                if (chosen >= routed_experts || !isfinite(rank_scores[0]))
                    atomicCAS(status, 0, 2);
                else {
                    row_selected[rank] = chosen;
                    row_weights[rank] = route_scores[chosen];
                }
            }
            __syncthreads();
            if (*status) return;
        }
    }
    if (thread) return;
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

/* Materialize the canonical expert-major worklist from already-admitted route pairs. */
extern "C" __global__ void yvex_expert_worklist_build_cuda(
    const unsigned long long *selected, unsigned long long row_count,
    unsigned long long topk, unsigned long long expert_count,
    unsigned long long *order, unsigned long long *expert_ids,
    unsigned long long *bucket_offsets, unsigned long long *bucket_populations,
    unsigned long long *source_rows, unsigned long long *destination_rows,
    yvex_expert_worklist_observation *summary,
    unsigned long long supported_width_mask,
    unsigned long long tensor_core_minimum,
    unsigned long long admitted_width, unsigned int provenance, int *status)
{
    __shared__ unsigned long long counts[256];
    __shared__ unsigned long long offsets[256];
    __shared__ int active;
    unsigned int thread = threadIdx.x;
    unsigned long long pairs;
    if (!status || blockIdx.x) return;
    if (thread == 0u) {
        active = *status == 0;
        if (active && (!selected || !row_count || !topk || !expert_count ||
            expert_count > blockDim.x || !order || !expert_ids || !bucket_offsets ||
            !bucket_populations || !source_rows || !destination_rows || !summary ||
            !(supported_width_mask & 2ull) || supported_width_mask & 1ull ||
            row_count >= 63ull || !(supported_width_mask & (1ull << row_count)) ||
            !admitted_width || provenance > YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE ||
            row_count > ~0ull / topk)) {
            atomicCAS(status, 0, 2);
            active = 0;
        }
    }
    __syncthreads();
    if (!active) return;
    pairs = row_count * topk;
    unsigned long long count = 0ull;
    if ((unsigned long long)thread < expert_count) {
        for (unsigned long long pair = 0ull; pair < pairs; ++pair) {
            unsigned long long expert = selected[pair];
            if (expert >= expert_count) {
                atomicCAS(status, 0, 2);
                atomicExch(&active, 0);
            }
            if (expert == (unsigned long long)thread) count++;
        }
    }
    counts[thread] = count;
    __syncthreads();
    if (!active) return;
    if (thread == 0u) {
        unsigned long long emitted = 0ull, unique = 0ull;
        summary->schema_version = YVEX_EXPERT_WORKLIST_OBSERVATION_SCHEMA_V1;
        summary->worklist_count = 1ull;
        summary->pair_count = pairs;
        summary->bucket_count = 0ull;
        summary->maximum_bucket_population = 0ull;
        summary->tensor_core_eligible_pairs = 0ull;
        summary->tensor_core_executed_pairs = 0ull;
        summary->narrow_pairs = 0ull;
        summary->tail_rows = 0ull;
        for (unsigned int index = 0u; index < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP;
             ++index) {
            summary->width_histogram[index] = 0ull;
            summary->population_histogram[index] = 0ull;
        }
        for (unsigned int index = 0u;
             index <= YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE; ++index)
            summary->provenance_counts[index] = 0ull;
        summary->width_histogram[
            row_count < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP
                ? row_count : YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP - 1u] = 1ull;
        summary->provenance_counts[provenance] = 1ull;
        for (unsigned long long expert = 0ull; expert < expert_count; ++expert) {
            offsets[expert] = emitted;
            emitted += counts[expert];
            if (!counts[expert]) continue;
            expert_ids[unique] = expert;
            bucket_offsets[unique] = offsets[expert];
            bucket_populations[unique] = counts[expert];
            if (counts[expert] > summary->maximum_bucket_population)
                summary->maximum_bucket_population = counts[expert];
            unsigned long long histogram =
                counts[expert] < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP
                    ? counts[expert] : YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP - 1u;
            summary->population_histogram[histogram]++;
            if (tensor_core_minimum && counts[expert] >= tensor_core_minimum) {
                summary->tensor_core_eligible_pairs += counts[expert];
                if (counts[expert] % admitted_width)
                    summary->tail_rows += admitted_width - counts[expert] % admitted_width;
            } else summary->narrow_pairs += counts[expert];
            unique++;
        }
        if (emitted != pairs) {
            atomicCAS(status, 0, 2);
            active = 0;
        } else {
            bucket_offsets[unique] = emitted;
            summary->bucket_count = unique;
        }
    }
    __syncthreads();
    if (!active || (unsigned long long)thread >= expert_count) return;
    unsigned long long cursor = offsets[thread];
    for (unsigned long long pair = 0ull; pair < pairs; ++pair)
        if (selected[pair] == (unsigned long long)thread) {
            order[cursor] = pair;
            source_rows[cursor] = pair / topk;
            destination_rows[cursor] = pair;
            cursor++;
        }
}

static __device__ int worklist_tensor_core_pair(
    unsigned long long ordered_pair, const unsigned long long *bucket_offsets,
    const unsigned long long *bucket_populations,
    const yvex_expert_worklist_observation *summary,
    unsigned long long tensor_core_minimum, int *status)
{
    if (!tensor_core_minimum) return 0;
    if (!bucket_offsets || !bucket_populations || !summary ||
        !summary->bucket_count) {
        atomicCAS(status, 0, 2);
        return 0;
    }
    unsigned long long low = 0ull, high = summary->bucket_count;
    while (low + 1ull < high) {
        unsigned long long middle = low + (high - low) / 2ull;
        if (bucket_offsets[middle] <= ordered_pair) low = middle;
        else high = middle;
    }
    unsigned long long offset = bucket_offsets[low];
    unsigned long long population = bucket_populations[low];
    if (ordered_pair < offset || ordered_pair - offset >= population) {
        atomicCAS(status, 0, 2);
        return 0;
    }
    return population >= tensor_core_minimum;
}

extern "C" __global__ void yvex_moe_grouped_up_rows(
    const unsigned char *gate, unsigned long long gate_row_bytes,
    unsigned long long gate_expert_bytes, unsigned int gate_qtype,
    const unsigned char *up, unsigned long long up_row_bytes,
    unsigned long long up_expert_bytes, unsigned int up_qtype,
    const unsigned long long *selected, const float *weights,
    const unsigned long long *order,
    const unsigned long long *bucket_offsets,
    const unsigned long long *bucket_populations,
    const yvex_expert_worklist_observation *summary,
    unsigned long long tensor_core_minimum,
    unsigned long long pair_count, unsigned long long topk,
    unsigned long long expert_count, const unsigned char *input,
    unsigned long long input_extent, int q8_input,
    unsigned long long intermediate_width, double limit,
    float *intermediate, int *status)
{
    extern __shared__ float staged_activation[];
    unsigned int lane = threadIdx.x & 31u;
    unsigned long long warp = (unsigned long long)(threadIdx.x >> 5u);
    /* A row block stays within one ordered pair so its warps can reuse one exact activation. */
    unsigned long long rows_per_pair = intermediate_width / 8ull +
                                       (intermediate_width % 8ull != 0ull);
    unsigned long long ordered_pair = (unsigned long long)blockIdx.x / rows_per_pair;
    unsigned long long output_row = ((unsigned long long)blockIdx.x % rows_per_pair) *
                                    8ull + warp;
    if (!status || *status || ordered_pair >= pair_count) return;
    if (!gate || !up || !input || !intermediate || !topk ||
        !gate_row_bytes || !up_row_bytes || !input_extent ||
        !intermediate_width || !isfinite(limit) || limit <= 0.0) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    if (worklist_tensor_core_pair(
            ordered_pair, bucket_offsets, bucket_populations, summary,
            tensor_core_minimum, status)) return;
    if (*status) return;
    unsigned long long source_pair = order ? order[ordered_pair] : ordered_pair;
    if (source_pair >= pair_count) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long source_row = source_pair / topk;
    unsigned long long expert = selected ? selected[source_pair] : 0ull;
    if (expert >= expert_count) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    const unsigned char *activation = input + source_row * input_extent *
                                      (q8_input ? YVEX_CUDA_Q8_K_BYTES : sizeof(float));
    const unsigned char *gate_row;
    const unsigned char *up_row;
    float g, u;
    if (q8_input) {
        if (gate_row_bytes % input_extent || up_row_bytes % input_extent) {
            if (!lane) atomicCAS(status, 0, 2);
            return;
        }
        if (*status) return;
    } else {
        const float *source = (const float *)activation;
        for (unsigned long long index = threadIdx.x; index < input_extent;
             index += blockDim.x)
            staged_activation[index] = source[index];
        __syncthreads();
        if (*status) return;
        activation = (const unsigned char *)staged_activation;
    }
    if (output_row >= intermediate_width) return;
    gate_row = gate + expert * gate_expert_bytes + output_row * gate_row_bytes;
    up_row = up + expert * up_expert_bytes + output_row * up_row_bytes;
    g = moe_warp_dot(gate_row, activation, input_extent, gate_row_bytes,
                     gate_qtype, q8_input, status);
    u = moe_warp_dot(up_row, activation, input_extent, up_row_bytes,
                     up_qtype, q8_input, status);
    if (!lane && !*status) {
        g = fminf(g, (float)limit);
        u = fmaxf((float)-limit, fminf(u, (float)limit));
        float silu = g >= 0.0f ? g / (1.0f + expf(-g))
                               : g * expf(g) / (1.0f + expf(g));
        float route_weight = weights ? weights[source_pair] : 1.0f;
        float value = float_to_bf16_rne(silu * u * route_weight);
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else intermediate[ordered_pair * intermediate_width + output_row] = value;
    }
}

extern "C" __global__ void yvex_moe_grouped_down_rows(
    const unsigned char *down, unsigned long long row_bytes,
    unsigned long long expert_bytes, unsigned int qtype,
    const unsigned long long *selected,
    const unsigned long long *order,
    const unsigned long long *bucket_offsets,
    const unsigned long long *bucket_populations,
    const yvex_expert_worklist_observation *summary,
    unsigned long long tensor_core_minimum, unsigned long long pair_count,
    unsigned long long topk, unsigned long long expert_count,
    const unsigned char *intermediate, unsigned long long intermediate_extent,
    int q8_input, unsigned long long hidden, float *pair_outputs, int *status)
{
    extern __shared__ float staged_activation[];
    unsigned int lane = threadIdx.x & 31u;
    unsigned long long warp = (unsigned long long)(threadIdx.x >> 5u);
    /* The pair-local block contract also makes partial output-row groups synchronization-safe. */
    unsigned long long rows_per_pair = hidden / 8ull + (hidden % 8ull != 0ull);
    unsigned long long ordered_pair = (unsigned long long)blockIdx.x / rows_per_pair;
    unsigned long long output_row = ((unsigned long long)blockIdx.x % rows_per_pair) *
                                    8ull + warp;
    if (!status || *status || ordered_pair >= pair_count) return;
    if (!down || !intermediate || !pair_outputs || !topk || !row_bytes ||
        !intermediate_extent || (q8_input && row_bytes % intermediate_extent)) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    if (worklist_tensor_core_pair(
            ordered_pair, bucket_offsets, bucket_populations, summary,
            tensor_core_minimum, status)) return;
    if (*status) return;
    unsigned long long source_pair = order ? order[ordered_pair] : ordered_pair;
    if (source_pair >= pair_count) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long expert = selected ? selected[source_pair] : 0ull;
    if (expert >= expert_count) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    const unsigned char *activation = intermediate + ordered_pair * intermediate_extent *
                                      (q8_input ? YVEX_CUDA_Q8_K_BYTES : sizeof(float));
    const unsigned char *weight;
    if (q8_input) {
        if (*status) return;
    } else {
        const float *source = (const float *)activation;
        for (unsigned long long index = threadIdx.x; index < intermediate_extent;
             index += blockDim.x)
            staged_activation[index] = source[index];
        __syncthreads();
        if (*status) return;
        activation = (const unsigned char *)staged_activation;
    }
    if (output_row >= hidden) return;
    weight = down + expert * expert_bytes + output_row * row_bytes;
    float dot = moe_warp_dot(weight, activation, intermediate_extent,
                             row_bytes, qtype, q8_input, status);
    if (!lane && !*status) {
        float value = float_to_bf16_rne(dot);
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
    double limit, float route_weight, float *output, int *status)
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
    float value = float_to_bf16_rne((float)(silu * u * route_weight));
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
