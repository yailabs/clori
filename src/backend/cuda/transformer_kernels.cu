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

/* Preserve the BF16 cast between unit normalization and affine weighting. */
extern "C" __global__ void yvex_rms_norm_bf16_policy_f32(
    const float *input, const float *weight, float *output,
    unsigned long long hidden_size, unsigned long long row_count, float epsilon)
{
    extern __shared__ float scratch[];
    unsigned int lane = threadIdx.x;
    unsigned long long row = (unsigned long long)blockIdx.x;
    unsigned long long offset, index;
    float sum = 0.0f, inverse;
    if (!input || !weight || !output || !hidden_size || row >= row_count || epsilon <= 0.0f)
        return;
    offset = row * hidden_size;
    for (index = lane; index < hidden_size; index += blockDim.x) {
        float value = input[offset + index];
        sum += value * value;
    }
    scratch[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    inverse = rsqrtf(scratch[0] / (float)hidden_size + epsilon);
    for (index = lane; index < hidden_size; index += blockDim.x) {
        float normalized = float_to_bf16_rne(input[offset + index] * inverse);
        output[offset + index] = float_to_bf16_rne(normalized * weight[index]);
    }
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

/* Execute bounded grouped-query attention without materializing a score matrix. */
extern "C" __global__ void yvex_gqa_f32(
    const float *query, const float *key, const float *value, float *output,
    unsigned long long tokens, unsigned long long query_heads,
    unsigned long long kv_heads, unsigned long long head_dim, float scale, int causal)
{
    extern __shared__ float scratch[];
    unsigned long long row = (unsigned long long)blockIdx.x;
    unsigned long long token = row / query_heads;
    unsigned long long query_head = row % query_heads;
    unsigned long long kv_head = query_head / (query_heads / kv_heads);
    unsigned long long lane = (unsigned long long)threadIdx.x;
    unsigned long long visible, source, dim;
    float local, maximum, accumulated = 0.0f;
    if (!query || !key || !value || !output || !tokens || !kv_heads ||
        !query_heads || query_heads % kv_heads || !head_dim || token >= tokens) return;
    visible = causal ? token + 1ull : tokens;
    if (threadIdx.x == 0) scratch[blockDim.x] = -INFINITY;
    __syncthreads();
    for (source = 0ull; source < visible; ++source) {
        local = 0.0f;
        for (dim = lane; dim < head_dim; dim += (unsigned long long)blockDim.x)
            local += query[(token * query_heads + query_head) * head_dim + dim] *
                     key[(source * kv_heads + kv_head) * head_dim + dim];
        scratch[threadIdx.x] = local;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
            if (threadIdx.x < offset) scratch[threadIdx.x] += scratch[threadIdx.x + offset];
            __syncthreads();
        }
        if (threadIdx.x == 0 && scratch[0] * scale > scratch[blockDim.x])
            scratch[blockDim.x] = scratch[0] * scale;
        __syncthreads();
    }
    maximum = scratch[blockDim.x];
    if (threadIdx.x == 0) scratch[blockDim.x + 1u] = 0.0f;
    __syncthreads();
    for (source = 0ull; source < visible; ++source) {
        local = 0.0f;
        for (dim = lane; dim < head_dim; dim += (unsigned long long)blockDim.x)
            local += query[(token * query_heads + query_head) * head_dim + dim] *
                     key[(source * kv_heads + kv_head) * head_dim + dim];
        scratch[threadIdx.x] = local;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
            if (threadIdx.x < offset) scratch[threadIdx.x] += scratch[threadIdx.x + offset];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            scratch[0] = expf(scratch[0] * scale - maximum);
            scratch[blockDim.x + 1u] += scratch[0];
        }
        __syncthreads();
        if (lane < head_dim)
            accumulated += scratch[0] *
                value[(source * kv_heads + kv_head) * head_dim + lane];
        __syncthreads();
    }
    if (lane < head_dim && scratch[blockDim.x + 1u] > 0.0f)
        output[(token * query_heads + query_head) * head_dim + lane] =
            accumulated / scratch[blockDim.x + 1u];
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
    float hidden, gate;
    if (!input || !output || !width || index >= rows * width) return;
    row = index / width;
    column = index % width;
    base = row * 2ull * width + column;
    hidden = input[base];
    gate = input[base + width];
    gate = float_to_bf16_rne(gate / (1.0f + expf(-gate)));
    output[index] = float_to_bf16_rne(hidden * gate);
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
