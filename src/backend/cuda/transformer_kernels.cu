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

/* Preserve the BF16 cast between Qwen-style normalization and affine weighting. */
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
    unsigned long long head_dim)
{
    unsigned long long pair =
        ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) +
        (unsigned long long)threadIdx.x;
    unsigned long long half = head_dim / 2ull;
    unsigned long long vector_count = tokens * heads;
    unsigned long long vector, lane, token, base;
    float first, second, cosine, sine;
    if (!values || !cosines || !sines || !tokens || !heads || !half ||
        pair >= vector_count * half) return;
    vector = pair / half;
    lane = pair % half;
    token = vector / heads;
    base = vector * head_dim;
    cosine = cosines[token * head_dim + lane];
    sine = sines[token * head_dim + lane];
    first = values[base + lane];
    second = values[base + half + lane];
    values[base + lane] = first * cosine - second * sine;
    values[base + half + lane] = second * cosine + first * sine;
}

/* Execute bounded causal grouped-query attention without materializing a score matrix. */
extern "C" __global__ void yvex_gqa_causal_f32(
    const float *query, const float *key, const float *value, float *output,
    unsigned long long tokens, unsigned long long query_heads,
    unsigned long long kv_heads, unsigned long long head_dim, float scale)
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
    visible = token + 1ull;
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
