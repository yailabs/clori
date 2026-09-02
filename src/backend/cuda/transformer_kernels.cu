/*
 * Execute transformer-specific BF16 policy and packed attention primitives.
 *
 * These kernels share the transformer numerical contract with the CUDA host owner while retaining
 * an independent NVCC boundary. Generic qtype, sampling, and family kernels remain outside it.
 */
#include "src/backend/cuda/kernel_primitives.h"

extern "C" __global__ void yvex_f32_to_bf16(
    const float *input, unsigned short *output, unsigned long long count, int *status)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    union { float value; unsigned int bits; } encoded;
    unsigned int rounding;
    if (!status || *status || index >= count) return;
    if (!input || !output || !isfinite(input[index])) {
        atomicCAS(status, 0, 1);
        return;
    }
    encoded.value = input[index];
    rounding = 0x7fffu + ((encoded.bits >> 16u) & 1u);
    output[index] = (unsigned short)((encoded.bits + rounding) >> 16u);
}

extern "C" __global__ void yvex_bf16_to_f32(
    const unsigned short *input, float *output, unsigned long long count)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    union { unsigned int bits; float value; } decoded;
    if (!input || !output || index >= count) return;
    decoded.bits = (unsigned int)input[index] << 16u;
    output[index] = decoded.value;
}

/* The source CUDA RMS operation reduces four adjacent BF16-published values per vector lane,
 * then combines four warps. Both grouping and publication order are observable across a deep
 * stack. */
extern "C" __global__ void yvex_rms_norm_bf16_policy_f32(
    const float *input, const float *weight, float *output,
    unsigned long long hidden_size, unsigned long long row_count, float epsilon)
{
    extern __shared__ float scratch[];
    unsigned int thread = threadIdx.x;
    unsigned int lane = thread & 31u;
    unsigned int warp = thread >> 5u;
    unsigned long long row = (unsigned long long)blockIdx.x;
    unsigned long long offset, vector, index;
    float sum = 0.0f, inverse;
    if (!input || !weight || !output || !hidden_size || (hidden_size & 3ull) ||
        blockDim.x != 128u || row >= row_count || epsilon <= 0.0f)
        return;
    offset = row * hidden_size;
    for (vector = thread; vector < hidden_size / 4ull; vector += blockDim.x) {
#pragma unroll
        for (unsigned int element = 0u; element < 4u; ++element) {
            float value = input[offset + vector * 4ull + element];
            sum += value * value;
        }
    }
    for (unsigned int stride = 16u; stride; stride >>= 1u)
        sum += __shfl_down_sync(0xffffffffu, sum, stride);
    if (lane == 0u) scratch[warp] = sum;
    __syncthreads();
    for (unsigned int stride = 2u; stride; stride >>= 1u) {
        if (lane == 0u && warp < stride) scratch[warp] += scratch[warp + stride];
        __syncthreads();
    }
    inverse = rsqrtf(scratch[0] / (float)hidden_size + epsilon);
    for (index = thread; index < hidden_size; index += blockDim.x)
        output[offset + index] =
            float_to_bf16_rne(weight[index] * (inverse * input[offset + index]));
}

/* Apply explicit rotate-half cos/sin tables to packed token/head vectors in place. */
extern "C" __global__ void yvex_rotary_half_f32(
    float *values, const float *cosines, const float *sines,
    unsigned long long tokens, unsigned long long heads,
    unsigned long long head_dim, unsigned long long rotary_dim)
{
    unsigned long long pair =
        ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) +
        (unsigned long long)threadIdx.x;
    unsigned long long half = rotary_dim / 2ull;
    unsigned long long vector_count = tokens * heads;
    unsigned long long vector, lane, token, base;
    float first, second, cosine, sine, first_product, second_product;
    if (!values || !cosines || !sines || !tokens || !heads || !half ||
        rotary_dim > head_dim ||
        pair >= vector_count * half) return;
    vector = pair / half;
    lane = pair % half;
    token = vector / heads;
    base = vector * head_dim;
    cosine = cosines[token * rotary_dim + lane];
    sine = sines[token * rotary_dim + lane];
    first = values[base + lane];
    second = values[base + half + lane];
    first_product = float_to_bf16_rne(first * cosine);
    second_product = float_to_bf16_rne(second * sine);
    values[base + lane] = float_to_bf16_rne(first_product - second_product);
    first_product = float_to_bf16_rne(second * cosine);
    second_product = float_to_bf16_rne(first * sine);
    values[base + half + lane] = float_to_bf16_rne(first_product + second_product);
}

extern "C" __global__ void yvex_rotary_half_plain_f32(
    float *values, const float *cosines, const float *sines,
    unsigned long long tokens, unsigned long long heads,
    unsigned long long head_dim, unsigned long long rotary_dim)
{
    unsigned long long pair =
        ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) +
        (unsigned long long)threadIdx.x;
    unsigned long long half = rotary_dim / 2ull;
    unsigned long long vector_count = tokens * heads;
    unsigned long long vector, lane, token, base;
    float first, second, cosine, sine;
    if (!values || !cosines || !sines || !tokens || !heads || !half ||
        rotary_dim > head_dim || pair >= vector_count * half) return;
    vector = pair / half;
    lane = pair % half;
    token = vector / heads;
    base = vector * head_dim;
    cosine = cosines[token * rotary_dim + lane];
    sine = sines[token * rotary_dim + lane];
    first = values[base + lane];
    second = values[base + half + lane];
    values[base + lane] = first * cosine - second * sine;
    values[base + half + lane] = second * cosine + first * sine;
}

/* Four warp-owned queries reuse K/V while online softmax avoids a second Q/K traversal. */
template <unsigned int Slots>
__device__ __forceinline__ void yvex_gqa_f32_body(
    const float *query, const float *key, const float *value, float *output,
    unsigned long long query_tokens, unsigned long long key_value_tokens,
    unsigned long long query_start, unsigned long long query_heads,
    unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long query_stride, unsigned long long key_stride,
    unsigned long long value_stride, float scale, int causal, float *scratch)
{
    const unsigned int warp = threadIdx.x / warpSize, lane = threadIdx.x % warpSize;
    const unsigned int queries_per_block = blockDim.x / warpSize;
    unsigned long long query_tile = blockIdx.x / query_heads, query_head = blockIdx.x % query_heads;
    unsigned long long token = query_tile * queries_per_block + warp;
    unsigned long long position = query_start + token;
    unsigned long long kv_head = query_head / (query_heads / kv_heads);
    unsigned long long maximum_visible, visible, source, dim;
    float query_lanes[Slots] = {0.0f}, accumulated[Slots] = {0.0f};
    float maximum = -INFINITY, denominator = 0.0f;
    unsigned int slot, active;
    if (!query || !key || !value || !output || !query_tokens ||
        !key_value_tokens || !kv_heads ||
        !query_heads || query_heads % kv_heads || !head_dim ||
        head_dim > (unsigned long long)Slots * warpSize || queries_per_block > 4u) return;
    active = token < query_tokens && position < key_value_tokens;
    visible = causal ? min(key_value_tokens, position + 1ull)
                     : key_value_tokens;
    maximum_visible = causal
        ? min(key_value_tokens,
              query_start + (query_tile + 1ull) * queries_per_block)
        : key_value_tokens;
    for (slot = 0u; slot < Slots; ++slot) {
        dim = (unsigned long long)lane + (unsigned long long)slot * warpSize;
        if (active && dim < head_dim)
            query_lanes[slot] = query[token * query_stride + query_head * head_dim + dim];
    }
    for (source = 0ull; source < maximum_visible; ++source) {
        float dot = 0.0f;
        float new_maximum, old_scale, weight;
        for (dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
            scratch[dim] = key[source * key_stride + kv_head * head_dim + dim];
            scratch[head_dim + dim] =
                value[source * value_stride + kv_head * head_dim + dim];
        }
        __syncthreads();
        for (slot = 0u; slot < Slots; ++slot) {
            dim = (unsigned long long)lane + (unsigned long long)slot * warpSize;
            if (active && source < visible && dim < head_dim)
                dot += query_lanes[slot] * scratch[dim];
        }
        for (unsigned int offset = warpSize >> 1; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        dot = __shfl_sync(0xffffffffu, dot, 0) * scale;
        if (active && source < visible) {
            new_maximum = fmaxf(maximum, dot);
            old_scale = maximum == -INFINITY ? 0.0f : expf(maximum - new_maximum);
            weight = expf(dot - new_maximum);
            denominator = denominator * old_scale + weight;
            for (slot = 0u; slot < Slots; ++slot) {
                dim = (unsigned long long)lane + (unsigned long long)slot * warpSize;
                if (dim < head_dim)
                    accumulated[slot] = accumulated[slot] * old_scale + weight * scratch[head_dim + dim];
            }
            maximum = new_maximum;
        }
        __syncthreads();
    }
    for (slot = 0u; slot < Slots; ++slot) {
        dim = (unsigned long long)lane + (unsigned long long)slot * warpSize;
        if (active && dim < head_dim && denominator > 0.0f)
            output[(token * query_heads + query_head) * head_dim + dim] = accumulated[slot] / denominator;
    }
}

extern "C" __global__ void yvex_gqa_f32(
    const float *query, const float *key, const float *value, float *output,
    unsigned long long query_tokens, unsigned long long key_value_tokens,
    unsigned long long query_start, unsigned long long query_heads,
    unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long query_stride, unsigned long long key_stride,
    unsigned long long value_stride, float scale, int causal)
{
    extern __shared__ float scratch[];
    yvex_gqa_f32_body<4u>(
        query, key, value, output, query_tokens, key_value_tokens,
        query_start, query_heads, kv_heads, head_dim, query_stride, key_stride,
        value_stride, scale, causal, scratch);
}

extern "C" __global__ void yvex_gqa_wide_f32(
    const float *query, const float *key, const float *value, float *output,
    unsigned long long query_tokens, unsigned long long key_value_tokens,
    unsigned long long query_start, unsigned long long query_heads,
    unsigned long long kv_heads, unsigned long long head_dim,
    unsigned long long query_stride, unsigned long long key_stride,
    unsigned long long value_stride, float scale, int causal)
{
    extern __shared__ float scratch[];
    yvex_gqa_f32_body<8u>(
        query, key, value, output, query_tokens, key_value_tokens,
        query_start, query_heads, kv_heads, head_dim, query_stride, key_stride,
        value_stride, scale, causal, scratch);
}

/* Source attention normalizes in F32; only the completed attention output is BF16-rounded. */
extern "C" __global__ void yvex_gqa_softmax_f32(
    const float *scores, float *probabilities,
    unsigned long long query_rows, unsigned long long tokens,
    unsigned long long query_start, int causal, int *status)
{
    extern __shared__ float reduction[];
    unsigned long long row = blockIdx.x, local_query = row % query_rows;
    unsigned long long query = query_start + local_query, source;
    float maximum = -INFINITY, sum = 0.0f;
    if (!scores || !probabilities || !status || *status || !query_rows || !tokens) return;
    for (source = threadIdx.x; source < tokens; source += blockDim.x) {
        float value = scores[row * tokens + source];
        if ((!causal || source <= query) && !isfinite(value)) atomicCAS(status, 0, 1);
        if (!causal || source <= query) maximum = fmaxf(maximum, value);
    }
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x],
                                            reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = reduction[0];
    for (source = threadIdx.x; source < tokens; source += blockDim.x)
        if (!causal || source <= query) sum += expf(scores[row * tokens + source] - maximum);
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    sum = reduction[0];
    if (!isfinite(sum) || sum <= 0.0f) atomicCAS(status, 0, 1);
    for (source = threadIdx.x; source < tokens; source += blockDim.x) {
        float probability = (!causal || source <= query) && sum > 0.0f
                                ? expf(scores[row * tokens + source] - maximum) / sum
                                : 0.0f;
        probabilities[row * tokens + source] = probability;
    }
}

/* Match the source CUDA warp-softmax reduction for bounded rows up to 1024 elements. */
extern "C" __global__ void yvex_gqa_softmax_warp_f32(
    const float *scores, float *probabilities,
    unsigned long long query_rows, unsigned long long tokens,
    unsigned long long query_start, int causal, int *status)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned long long row = (unsigned long long)blockIdx.x;
    const unsigned long long local_query = query_rows ? row % query_rows : 0ull;
    const unsigned long long query = query_start + local_query;
    float elements[32], maximum, sum = 0.0f;
    unsigned int power = 1u, iterations, index;
    if (!scores || !probabilities || !status || !query_rows || !tokens || tokens > 1024ull)
        return;
    while (power < tokens) power <<= 1u;
    iterations = power >> 5u;
    if (!iterations) iterations = 1u;
    if (*status) return;
    for (index = 0u; index < iterations; ++index) {
        unsigned long long source = lane + (unsigned long long)index * 32ull;
        int valid = source < tokens && (!causal || source <= query);
        elements[index] = valid ? scores[row * tokens + source] : -INFINITY;
        if (valid && !isfinite(elements[index])) atomicCAS(status, 0, 1);
    }
    maximum = elements[0];
    for (index = 0u; index < iterations; ++index)
        maximum = maximum > elements[index] ? maximum : elements[index];
    for (unsigned int offset = 16u; offset; offset >>= 1u) {
        float other = __shfl_xor_sync(0xffffffffu, maximum, offset, 32);
        maximum = maximum < other ? other : maximum;
    }
    for (index = 0u; index < iterations; ++index) {
        elements[index] = expf(elements[index] - maximum);
        sum += elements[index];
    }
    for (unsigned int offset = 16u; offset; offset >>= 1u)
        sum += __shfl_xor_sync(0xffffffffu, sum, offset, 32);
    if (!isfinite(sum) || sum <= 0.0f) atomicCAS(status, 0, 1);
    for (index = 0u; index < iterations; ++index) {
        unsigned long long source = lane + (unsigned long long)index * 32ull;
        if (source < tokens)
            probabilities[row * tokens + source] = elements[index] / sum;
    }
}

extern "C" __global__ void yvex_attention_validate_f32(
    const float *values, unsigned long long count, int *status)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (!values || !status || *status || index >= count) return;
    if (!isfinite(values[index])) atomicCAS(status, 0, 1);
}

/* Fuse the elementwise SiLU gate product used by dense transformer MLPs. */
extern "C" __global__ void yvex_silu_product_bf16_f32(
    const float *gate, const float *up, float *output, unsigned long long count)
{
    unsigned long long index =
        ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) +
        (unsigned long long)threadIdx.x;
    float value;
    if (!gate || !up || !output || index >= count) return;
    value = gate[index];
    value = float_to_bf16_rne(value / (1.0f + expf(-value)));
    output[index] = float_to_bf16_rne(value * up[index]);
}

extern "C" __global__ void yvex_silu_f32(
    const float *input, float *output, unsigned long long count, int bf16_output)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    float value;
    if (!input || !output || index >= count) return;
    value = input[index] / (1.0f + expf(-input[index]));
    output[index] = bf16_output ? float_to_bf16_rne(value) : value;
}

/* Match the exact erf GELU used by vision towers; approximation policy is not implicit. */
extern "C" __global__ void yvex_gelu_f32(
    const float *input, float *output, unsigned long long count,
    int tanh_approximation, int bf16_output)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    float value;
    if (!input || !output || index >= count) return;
    if (tanh_approximation) {
        float cube = input[index] * input[index] * input[index];
        value = 0.5f * input[index] *
                (1.0f + tanhf(0.7978845608028654f *
                               (input[index] + 0.044715f * cube)));
    } else {
        value = 0.5f * input[index] *
                (1.0f + erff(input[index] * 0.7071067811865475f));
    }
    output[index] = bf16_output ? float_to_bf16_rne(value) : value;
}

/* Source timestep features are constructed in F32 on the execution device. Host-double
 * reconstruction crosses rounding boundaries that become observable after repeated blocks. */
extern "C" __global__ void yvex_timestep_embedding_f32(
    const float *timesteps, float *output, unsigned long long rows,
    unsigned long long half_width, float maximum_period)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row, lane;
    float exponent, angle;
    if (!timesteps || !output || !rows || !half_width || maximum_period <= 1.0f ||
        index >= rows * half_width) return;
    row = index / half_width;
    lane = index % half_width;
    exponent = -logf(maximum_period) * (float)lane / (float)half_width;
    angle = timesteps[row] * expf(exponent);
    output[row * half_width * 2ull + lane] = cosf(angle);
    output[row * half_width * 2ull + half_width + lane] = sinf(angle);
}

extern "C" __global__ void yvex_split_three_f32(
    const float *input, float *first, float *second, float *third,
    unsigned long long rows, unsigned long long width)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row, column, input_base;
    if (!input || !first || !second || !third || !width || index >= rows * width) return;
    row = index / width;
    column = index % width;
    input_base = row * 3ull * width + column;
    first[index] = input[input_base];
    second[index] = input[input_base + width];
    third[index] = input[input_base + 2ull * width];
}

extern "C" __global__ void yvex_split_interleaved_three_f32(
    const float *input, float *first, float *second, float *third,
    unsigned long long rows, unsigned long long heads,
    unsigned long long head_dim)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long width = heads * head_dim;
    unsigned long long row, within, head, lane, input_base;
    if (!input || !first || !second || !third || !heads || !head_dim ||
        index >= rows * width) return;
    row = index / width;
    within = index % width;
    head = within / head_dim;
    lane = within % head_dim;
    input_base = row * 3ull * width + head * 3ull * head_dim + lane;
    first[index] = input[input_base];
    second[index] = input[input_base + head_dim];
    third[index] = input[input_base + 2ull * head_dim];
}

extern "C" __global__ void yvex_swiglu_split_bf16_f32(
    const float *input, float *output, unsigned long long rows, unsigned long long width)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row, column, base;
    float gate, up;
    if (!input || !output || !width || index >= rows * width) return;
    row = index / width;
    column = index % width;
    base = row * 2ull * width + column;
    gate = input[base];
    up = input[base + width];
    gate = float_to_bf16_rne(gate / (1.0f + expf(-gate)));
    output[index] = float_to_bf16_rne(gate * up);
}

extern "C" __global__ void yvex_swiglu_split_f32(
    const float *input, float *output, unsigned long long rows,
    unsigned long long width, int gate_first)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row, column, base;
    float hidden, gate;
    if (!input || !output || !width || (gate_first != 0 && gate_first != 1) ||
        index >= rows * width) return;
    row = index / width;
    column = index % width;
    base = row * 2ull * width + column;
    gate = input[base + (gate_first ? 0ull : width)];
    hidden = input[base + (gate_first ? width : 0ull)];
    output[index] = gate / (1.0f + expf(-gate)) * hidden;
}

extern "C" __global__ void yvex_scaled_residual_f32(
    const float *residual, const float *update, const float *scale,
    float *output, unsigned long long rows, unsigned long long width)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!residual || !update || !scale || !output || !width ||
        index >= rows * width) return;
    output[index] = residual[index] + update[index] * scale[index % width];
}

extern "C" __global__ void yvex_layer_norm_f32(
    const float *input, const float *weight, const float *bias, float *output,
    unsigned long long rows, unsigned long long width, float epsilon)
{
    extern __shared__ float scratch[];
    unsigned int lane = threadIdx.x;
    unsigned long long row = (unsigned long long)blockIdx.x;
    unsigned long long index, offset;
    float sum = 0.0f, mean, variance = 0.0f, inverse;
    if (!input || !weight || !bias || !output || !width || row >= rows || epsilon <= 0.0f)
        return;
    offset = row * width;
    for (index = lane; index < width; index += blockDim.x) sum += input[offset + index];
    scratch[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    mean = scratch[0] / (float)width;
    for (index = lane; index < width; index += blockDim.x) {
        float centered = input[offset + index] - mean;
        variance += centered * centered;
    }
    scratch[lane] = variance;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    inverse = rsqrtf(scratch[0] / (float)width + epsilon);
    for (index = lane; index < width; index += blockDim.x)
        output[offset + index] =
            (input[offset + index] - mean) * inverse * weight[index] + bias[index];
}

extern "C" __global__ void yvex_modulation_bf16_f32(
    const float *input, const float *table, const unsigned int *row_indices,
    float *output, unsigned long long rows, unsigned long long width,
    unsigned long long table_rows, unsigned long long parameters,
    unsigned int shift_slot, unsigned int scale_slot)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row, column, table_row, base;
    float factor, value;
    if (!input || !table || !row_indices || !output || !width ||
        index >= rows * width) return;
    row = index / width;
    column = index % width;
    table_row = row_indices[row];
    if (table_row >= table_rows || shift_slot >= parameters || scale_slot >= parameters) return;
    base = table_row * parameters * width;
    factor = float_to_bf16_rne(1.0f + table[base + scale_slot * width + column]);
    value = float_to_bf16_rne(input[index] * factor);
    output[index] = float_to_bf16_rne(value + table[base + shift_slot * width + column]);
}

extern "C" __global__ void yvex_gated_residual_bf16_f32(
    const float *residual, const float *table, const unsigned int *row_indices,
    const float *update, float *output, unsigned long long rows, unsigned long long width,
    unsigned long long table_rows, unsigned long long parameters, unsigned int gate_slot)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long row, column, table_row, base;
    float value;
    if (!residual || !table || !row_indices || !update || !output || !width ||
        index >= rows * width) return;
    row = index / width;
    column = index % width;
    table_row = row_indices[row];
    if (table_row >= table_rows || gate_slot >= parameters) return;
    base = table_row * parameters * width + gate_slot * width + column;
    value = float_to_bf16_rne(table[base] * update[index]);
    output[index] = float_to_bf16_rne(residual[index] + value);
}

extern "C" __global__ void yvex_bias_f32(
    const float *input, const float *bias, float *output,
    unsigned long long rows, unsigned long long width, int bf16_output)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    float value;
    if (!input || !bias || !output || !width || index >= rows * width) return;
    value = input[index] + bias[index % width];
    output[index] = bf16_output ? float_to_bf16_rne(value) : value;
}

extern "C" __global__ void yvex_add_bf16_f32(
    const float *left, const float *right, float *output,
    unsigned long long rows, unsigned long long width)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!left || !right || !output || !width || index >= rows * width) return;
    output[index] = float_to_bf16_rne(left[index] + right[index]);
}
