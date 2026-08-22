/*
 * Execute exact and native online-softmax attention reduction kernels.
 *
 * The exact kernel preserves forensic arithmetic. The native kernel owns the
 * independently admitted parallel reduction used by production execution.
 */
#include "src/backend/cuda/kernel_primitives.h"

typedef struct {
    const float *local;
    const unsigned long long *local_positions;
    const float *compressed;
    const unsigned long long *compressed_positions;
    const unsigned long long *selected;
    unsigned long long initial_local_count, local_stride, compressed_stride;
    unsigned long long topk_capacity, sliding_window, ratio;
    unsigned long long phase_start_position, token_count;
    unsigned int attention_class;
    int candidate_block_visible;
} attention_reduce_rows;

/* The rolling-state contract already defines two equally shaped projections of
 * one activation.  A warp retains the activation load across both independent
 * accumulators while preserving each projection's original dot-product order. */
extern "C" __global__ void yvex_attention_bf16_pair(
    const unsigned char *first, unsigned long long first_row_bytes,
    const unsigned char *second, unsigned long long second_row_bytes,
    unsigned long long row_width, unsigned long long row_count,
    const float *input, float *first_out, float *second_out, int *status)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5u;
    unsigned long long row_index = (unsigned long long)blockIdx.x * 8ull + warp;
    float first_sum = 0.0f, second_sum = 0.0f;
    if (!status) return;
    if (!first || !second || !first_row_bytes || !second_row_bytes ||
        !row_width || !row_count || !input || !first_out || !second_out ||
        blockDim.x != 256u) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    if (*status || row_index >= row_count) return;
    first += row_index * first_row_bytes;
    second += row_index * second_row_bytes;
    for (unsigned long long i = lane; i < row_width; i += 32ull) {
        float value = input[i];
        float first_weight = bf16_bits_to_float(qtype_load_u16(first + i * 2ull));
        float second_weight = bf16_bits_to_float(qtype_load_u16(second + i * 2ull));
        if (!isfinite(value) || !isfinite(first_weight) || !isfinite(second_weight))
            atomicCAS(status, 0, 1);
        else {
            first_sum = fmaf(first_weight, value, first_sum);
            second_sum = fmaf(second_weight, value, second_sum);
        }
    }
    for (unsigned int offset = 16u; offset; offset >>= 1u) {
        first_sum += __shfl_down_sync(0xffffffffu, first_sum, offset);
        second_sum += __shfl_down_sync(0xffffffffu, second_sum, offset);
    }
    if (!lane) {
        if (!isfinite(first_sum) || !isfinite(second_sum)) atomicCAS(status, 0, 1);
        else {
            first_out[row_index] = first_sum;
            second_out[row_index] = second_sum;
        }
    }
}

static __device__ __forceinline__ const float *attention_reduce_row(
    const attention_reduce_rows *rows, unsigned long long pass,
    unsigned long long ordinal, unsigned long long candidate,
    unsigned long long local_offset, int *visible)
{
    const float *row = NULL;
    unsigned long long position = ~0ull;
    *visible = 0;
    if (pass == 0ull) {
        candidate += local_offset;
        row = rows->local + candidate * rows->local_stride;
        position = rows->local_positions[candidate];
        if (!rows->candidate_block_visible) {
            unsigned long long token_position = rows->phase_start_position + ordinal;
            unsigned long long first = token_position + 1ull > rows->sliding_window
                ? token_position + 1ull - rows->sliding_window : 0ull;
            if (position < first || position > token_position) return NULL;
        }
    } else {
        unsigned long long token_position = rows->phase_start_position + ordinal;
        unsigned long long index;
        index = rows->attention_class == 2u
            ? candidate : rows->selected[ordinal * rows->topk_capacity + candidate];
        row = rows->compressed + index * rows->compressed_stride;
        position = rows->compressed_positions[index];
        if (position > token_position ||
            position > ~0ull - rows->ratio + 1ull ||
            position + rows->ratio - 1ull > token_position) return NULL;
    }
    *visible = row != NULL;
    return row;
}

extern "C" __global__ void yvex_attention_reduce(
    const float *query,
    const float *local,
    const unsigned long long *local_positions,
    unsigned long long initial_local_count,
    unsigned long long local_stride,
    const float *compressed,
    const unsigned long long *compressed_positions,
    unsigned long long compressed_stride,
    const unsigned long long *selected,
    const unsigned long long *selected_count_ptr,
    unsigned long long topk_capacity,
    const float *sinks,
    unsigned long long query_heads,
    unsigned long long head_dim,
    unsigned long long sliding_window,
    unsigned long long ratio,
    unsigned int attention_class,
    unsigned long long phase_start_position,
    unsigned long long token_count,
    int candidate_block_visible,
    float *out,
    int *status)
{
    extern __shared__ double dot_terms[];
    __shared__ double maximum;
    __shared__ double denominator;
    __shared__ double probability;
    __shared__ double renormalization;
    __shared__ int active;
    unsigned long long task = (unsigned long long)blockIdx.x;
    unsigned long long ordinal = query_heads ? task / query_heads : token_count;
    unsigned long long head = query_heads ? task % query_heads : query_heads;
    unsigned long long token_position, local_offset, local_count, compressed_count;
    unsigned int thread = threadIdx.x;
    if (!status) return;
    if (ordinal >= token_count || head >= query_heads) return;
    if (!query || !local || !local_positions || !sinks || !out || !query_heads ||
        !head_dim || !sliding_window || !token_count || attention_class > 2u ||
        (candidate_block_visible != 0 && candidate_block_visible != 1) ||
        phase_start_position > ~0ull - ordinal || local_stride < head_dim ||
        (attention_class != 0u &&
         (!compressed || !compressed_positions || compressed_stride < head_dim)) ||
        (attention_class == 1u &&
         (!selected || !selected_count_ptr || !topk_capacity)) ||
        (attention_class == 1u && ratio != 4ull) ||
        (attention_class == 2u && ratio != 128ull) ||
        (attention_class == 0u && ratio != 0ull)) {
        if (thread == 0u) atomicCAS(status, 0, 2);
        return;
    }
    token_position = phase_start_position + ordinal;
    if (candidate_block_visible) {
        local_offset = 0ull;
        local_count = initial_local_count + token_count;
    } else {
        unsigned long long local_before = initial_local_count + ordinal;
        unsigned long long history_count = token_position < sliding_window - 1ull
            ? token_position : sliding_window - 1ull;
        local_offset = local_before - history_count;
        local_count = history_count + 1ull;
    }
    compressed_count = attention_class == 0u ? 0ull
        : attention_class == 1u ? selected_count_ptr[ordinal]
        : token_position / ratio + ((token_position + 1ull) % ratio == 0ull);
    const float *q = query + (ordinal * query_heads + head) * head_dim;
    double scale = 1.0 / sqrt((double)head_dim);
    if (thread == 0u) {
        active = *status == 0;
        maximum = (double)sinks[head];
        denominator = 1.0;
    }
    __syncthreads();
    if (!active) return;
    attention_reduce_rows rows = {
        local, local_positions, compressed, compressed_positions, selected,
        initial_local_count, local_stride, compressed_stride, topk_capacity,
        sliding_window, ratio, phase_start_position, token_count,
        attention_class, candidate_block_visible
    };
    for (unsigned long long lane = (unsigned long long)thread; lane < head_dim;
         lane += (unsigned long long)blockDim.x)
        out[(ordinal * query_heads + head) * head_dim + lane] = 0.0f;
    __syncthreads();
    /* Candidates retain source order, but a stable online softmax keeps each
       dot product single-use. When a new maximum arrives, every output lane
       and the accumulated denominator are renormalized before that candidate
       is incorporated. */
    for (unsigned long long pass = 0ull; pass < 2ull; ++pass) {
        unsigned long long count = pass == 0ull ? local_count : compressed_count;
        for (unsigned long long candidate = 0ull; candidate < count; ++candidate) {
            int visible;
            const float *row = attention_reduce_row(
                &rows, pass, ordinal, candidate, local_offset, &visible);
            if (!visible) continue;
            double dot = 0.0;
            for (unsigned long long base = 0ull; base < head_dim;
                 base += (unsigned long long)blockDim.x) {
                unsigned long long lane = base + (unsigned long long)thread;
                dot_terms[thread] = lane < head_dim
                    ? __dmul_rn((double)q[lane], (double)row[lane]) : 0.0;
                __syncthreads();
                if (thread == 0u) {
                    unsigned long long tile = head_dim - base;
                    if (tile > (unsigned long long)blockDim.x) tile = blockDim.x;
                    for (unsigned long long i = 0ull; i < tile; ++i)
                        dot = __dadd_rn(dot, dot_terms[i]);
                }
                __syncthreads();
            }
            if (thread == 0u) {
                double score = __dmul_rn(dot, scale);
                if (score > maximum) {
                    renormalization = exp(__dadd_rn(maximum, -score));
                    maximum = score;
                    probability = 1.0;
                    denominator = __dadd_rn(
                        __dmul_rn(denominator, renormalization), probability);
                } else {
                    renormalization = 1.0;
                    probability = exp(__dadd_rn(score, -maximum));
                    denominator = __dadd_rn(denominator, probability);
                }
            }
            __syncthreads();
            for (unsigned long long lane = (unsigned long long)thread; lane < head_dim;
                 lane += (unsigned long long)blockDim.x) {
                unsigned long long offset =
                    (ordinal * query_heads + head) * head_dim + lane;
                out[offset] = (float)__dadd_rn(
                    __dmul_rn((double)out[offset], renormalization),
                    __dmul_rn(probability, (double)row[lane]));
            }
            __syncthreads();
        }
    }
    if (thread == 0u && (!isfinite(denominator) || denominator <= 0.0)) {
        atomicCAS(status, 0, 1);
        active = 0;
    }
    __syncthreads();
    if (!active) return;
    for (unsigned long long lane = (unsigned long long)thread; lane < head_dim;
         lane += (unsigned long long)blockDim.x) {
        unsigned long long offset =
            (ordinal * query_heads + head) * head_dim + lane;
        float published = (float)__ddiv_rn((double)out[offset],
                                           denominator);
        if (!isfinite(published)) atomicCAS(status, 0, 1);
        else out[offset] = float_to_bf16_rne(published);
    }
}

/* Production keeps candidate order and online-softmax semantics while each
 * head uses all resident warps for its dot products. The final BF16 boundary
 * is the numerical contract; forensic evidence continues through the exact
 * kernel above. */
extern "C" __global__ void yvex_attention_reduce_native(
    const float *query,
    const float *local,
    const unsigned long long *local_positions,
    unsigned long long initial_local_count,
    unsigned long long local_stride,
    const float *compressed,
    const unsigned long long *compressed_positions,
    unsigned long long compressed_stride,
    const unsigned long long *selected,
    const unsigned long long *selected_count_ptr,
    unsigned long long topk_capacity,
    const float *sinks,
    unsigned long long query_heads,
    unsigned long long head_dim,
    unsigned long long sliding_window,
    unsigned long long ratio,
    unsigned int attention_class,
    unsigned long long phase_start_position,
    unsigned long long token_count,
    int candidate_block_visible,
    float *out,
    int *status)
{
    __shared__ float warp_sums[8];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    __shared__ float renormalization;
    __shared__ int active;
    unsigned long long task = (unsigned long long)blockIdx.x;
    unsigned long long ordinal = query_heads ? task / query_heads : token_count;
    unsigned long long head = query_heads ? task % query_heads : query_heads;
    unsigned long long token_position, local_offset, local_count, compressed_count;
    unsigned int thread = threadIdx.x;
    unsigned int lane = thread & 31u;
    unsigned int warp = thread >> 5u;
    if (!status) return;
    if (ordinal >= token_count || head >= query_heads) return;
    if (!query || !local || !local_positions || !sinks || !out || !query_heads ||
        !head_dim || !sliding_window || !token_count || blockDim.x != 256u ||
        attention_class > 2u ||
        (candidate_block_visible != 0 && candidate_block_visible != 1) ||
        phase_start_position > ~0ull - ordinal || local_stride < head_dim ||
        (attention_class != 0u &&
         (!compressed || !compressed_positions || compressed_stride < head_dim)) ||
        (attention_class == 1u &&
         (!selected || !selected_count_ptr || !topk_capacity)) ||
        (attention_class == 1u && ratio != 4ull) ||
        (attention_class == 2u && ratio != 128ull) ||
        (attention_class == 0u && ratio != 0ull)) {
        if (thread == 0u) atomicCAS(status, 0, 2);
        return;
    }
    token_position = phase_start_position + ordinal;
    if (candidate_block_visible) {
        local_offset = 0ull;
        local_count = initial_local_count + token_count;
    } else {
        unsigned long long local_before = initial_local_count + ordinal;
        unsigned long long history_count = token_position < sliding_window - 1ull
            ? token_position : sliding_window - 1ull;
        local_offset = local_before - history_count;
        local_count = history_count + 1ull;
    }
    compressed_count = attention_class == 0u ? 0ull
        : attention_class == 1u ? selected_count_ptr[ordinal]
        : token_position / ratio + ((token_position + 1ull) % ratio == 0ull);
    const float *q = query + (ordinal * query_heads + head) * head_dim;
    if (thread == 0u) {
        active = *status == 0;
        maximum = sinks[head];
        denominator = 1.0f;
    }
    for (unsigned long long i = (unsigned long long)thread; i < head_dim;
         i += (unsigned long long)blockDim.x)
        out[(ordinal * query_heads + head) * head_dim + i] = 0.0f;
    __syncthreads();
    if (!active) return;
    attention_reduce_rows rows = {
        local, local_positions, compressed, compressed_positions, selected,
        initial_local_count, local_stride, compressed_stride, topk_capacity,
        sliding_window, ratio, phase_start_position, token_count,
        attention_class, candidate_block_visible
    };
    for (unsigned long long pass = 0ull; pass < 2ull; ++pass) {
        unsigned long long count = pass == 0ull ? local_count : compressed_count;
        for (unsigned long long candidate = 0ull; candidate < count; ++candidate) {
            int visible;
            const float *row = attention_reduce_row(
                &rows, pass, ordinal, candidate, local_offset, &visible);
            if (!visible) continue;
            float dot = 0.0f;
            for (unsigned long long i = (unsigned long long)thread; i < head_dim;
                 i += (unsigned long long)blockDim.x)
                dot = fmaf(q[i], row[i], dot);
            for (unsigned int offset = 16u; offset; offset >>= 1u)
                dot += __shfl_down_sync(0xffffffffu, dot, offset);
            if (lane == 0u) warp_sums[warp] = dot;
            __syncthreads();
            if (warp == 0u) {
                dot = lane < 8u ? warp_sums[lane] : 0.0f;
                for (unsigned int offset = 16u; offset; offset >>= 1u)
                    dot += __shfl_down_sync(0xffffffffu, dot, offset);
                if (lane == 0u) {
                    float score = dot * rsqrtf((float)head_dim);
                    if (!isfinite(score)) {
                        atomicCAS(status, 0, 1);
                        active = 0;
                    } else if (score > maximum) {
                        renormalization = expf(maximum - score);
                        maximum = score;
                        probability = 1.0f;
                        denominator = denominator * renormalization + probability;
                    } else {
                        renormalization = 1.0f;
                        probability = expf(score - maximum);
                        denominator += probability;
                    }
                }
            }
            __syncthreads();
            if (!active) return;
            for (unsigned long long i = (unsigned long long)thread; i < head_dim;
                 i += (unsigned long long)blockDim.x) {
                unsigned long long offset =
                    (ordinal * query_heads + head) * head_dim + i;
                out[offset] = fmaf(probability, row[i],
                                   out[offset] * renormalization);
            }
            __syncthreads();
        }
    }
    if (thread == 0u && (!isfinite(denominator) || denominator <= 0.0f)) {
        atomicCAS(status, 0, 1);
        active = 0;
    }
    __syncthreads();
    if (!active) return;
    for (unsigned long long i = (unsigned long long)thread; i < head_dim;
         i += (unsigned long long)blockDim.x) {
        unsigned long long offset =
            (ordinal * query_heads + head) * head_dim + i;
        float published = out[offset] / denominator;
        if (!isfinite(published)) atomicCAS(status, 0, 1);
        else out[offset] = float_to_bf16_rne(published);
    }
}
