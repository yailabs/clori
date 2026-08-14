/*
 * Implement the admitted CUDA primitive arithmetic embedded into the generated PTX bundle.
 *
 * Every exported kernel is resolved through the generated PTX bundle and its matching host owner
 * validates byte/rank geometry first. A qtype row dot is primitive compute proof, not model
 * execution.
 */
#include "src/backend/cuda/kernel_primitives.h"
extern "C" __global__ void yvex_embed_f32(
    const float *embedding, const unsigned int *token_ids, float *out,
    unsigned long long hidden_size, unsigned long long vocab_size,
    unsigned long long token_count)
{
    unsigned long long idx =
        ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) +
        (unsigned long long)threadIdx.x;
    unsigned long long total;
    unsigned long long token_index;
    unsigned long long dim;
    unsigned int token_id;
    const unsigned long long max_ull = ~0ull;
    if (!embedding || !token_ids || !out ||
        hidden_size == 0ull || vocab_size == 0ull || token_count == 0ull ||
        hidden_size > max_ull / token_count ||
        hidden_size > max_ull / vocab_size) {
        return;
    }
    total = hidden_size * token_count;
    if (idx >= total) {
        return;
    }
    token_index = idx / hidden_size;
    dim = idx % hidden_size;
    token_id = token_ids[token_index];
    if ((unsigned long long)token_id >= vocab_size) {
        return;
    }
    out[idx] = embedding[((unsigned long long)token_id * hidden_size) + dim];
}
extern "C" __global__ void yvex_attention_bf16_round(
    float *values, unsigned long long count, int *status)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    float value;
    if (!status || index >= count || *status != 0) return;
    if (!values) {
        atomicCAS(status, 0, 2);
        return;
    }
    value = values[index];
    if (!isfinite(value)) {
        atomicCAS(status, 0, 1);
        return;
    }
    values[index] = float_to_bf16_rne(value);
}
extern "C" __global__ void yvex_argmax_f32(
    const float *values, unsigned long long row_count, unsigned long long row_width,
    unsigned int *selected_tokens, float *selected_values,
    unsigned long long *tie_counts, int *status)
{
    __shared__ float maxima[128];
    __shared__ unsigned long long indices[128], ties[128];
    unsigned int lane = threadIdx.x;
    unsigned long long row = blockIdx.x;
    float best = -3.402823466e+38F;
    unsigned long long best_index = ~0ull, tied = 0ull;
    if (!status || row >= row_count || blockDim.x != 128u || !values || !row_width ||
        !selected_tokens || !selected_values || !tie_counts) {
        if (status && !lane) atomicCAS(status, 0, 2);
        return;
    }
    values += row * row_width;
    for (unsigned long long index = lane; index < row_width; index += blockDim.x) {
        float value = values[index];
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else if (value > best) best = value, best_index = index, tied = 1ull;
        else if (value == best) {
            if (index < best_index) best_index = index;
            tied++;
        }
    }
    maxima[lane] = best;
    indices[lane] = best_index;
    ties[lane] = tied;
    __syncthreads();
    if (*status) return;
    for (unsigned int stride = 64u; stride; stride >>= 1u) {
        if (lane < stride) {
            float other = maxima[lane + stride];
            if (other > maxima[lane]) {
                maxima[lane] = other;
                indices[lane] = indices[lane + stride];
                ties[lane] = ties[lane + stride];
            } else if (other == maxima[lane]) {
                if (indices[lane + stride] < indices[lane])
                    indices[lane] = indices[lane + stride];
                ties[lane] += ties[lane + stride];
            }
        }
        __syncthreads();
    }
    if (!lane) {
        selected_tokens[row] = (unsigned int)indices[0];
        selected_values[row] = maxima[0];
        tie_counts[row] = ties[0];
    }
}

extern "C" __global__ void yvex_qtype_row_dot(
    const unsigned char *encoded, const float *vector,
    unsigned long long elements, unsigned int qtype, float *out)
{
    unsigned long long index;
    double sum = 0.0;
    if (blockIdx.x != 0u || threadIdx.x != 0u || !encoded || !vector ||
        !out || elements == 0ull) return;
    for (index = 0ull; index < elements; ++index)
        sum += (double)qtype_value(encoded, index, qtype) *
               (double)vector[index];
    out[0] = (float)sum;
}
extern "C" __global__ void yvex_embed_f16_to_f32(
    const unsigned short *embedding, const unsigned int *token_ids, float *out,
    unsigned long long hidden_size, unsigned long long vocab_size,
    unsigned long long token_count)
{
    unsigned long long idx =
        ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) +
        (unsigned long long)threadIdx.x;
    unsigned long long total;
    unsigned long long token_index;
    unsigned long long dim;
    unsigned int token_id;
    const unsigned long long max_ull = ~0ull;
    if (!embedding || !token_ids || !out ||
        hidden_size == 0ull || vocab_size == 0ull || token_count == 0ull ||
        hidden_size > max_ull / token_count ||
        hidden_size > max_ull / vocab_size) {
        return;
    }
    total = hidden_size * token_count;
    if (idx >= total) {
        return;
    }
    token_index = idx / hidden_size;
    dim = idx % hidden_size;
    token_id = token_ids[token_index];
    if ((unsigned long long)token_id >= vocab_size) {
        return;
    }
    out[idx] = f16_bits_to_float((unsigned int)embedding[((unsigned long long)token_id * hidden_size) + dim]);
}
extern "C" __global__ void yvex_rms_norm_f32_weight_f32(
    const float *input, const float *weight, float *out,
    unsigned long long hidden_size, unsigned long long row_count, float epsilon)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;
    unsigned long long row = (unsigned long long)blockIdx.x;
    unsigned long long row_offset;
    unsigned long long i;
    float sum = 0.0f;
    float inv_rms;
    if (!input || !weight || !out || hidden_size == 0ull || row >= row_count ||
        epsilon <= 0.0f) {
        return;
    }
    row_offset = row * hidden_size;
    for (i = (unsigned long long)tid; i < hidden_size; i += (unsigned long long)stride) {
        float v = input[row_offset + i];
        sum += v * v;
    }
    scratch[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        __syncthreads();
    }
    inv_rms = rsqrtf((scratch[0] / (float)hidden_size) + epsilon);
    for (i = (unsigned long long)tid; i < hidden_size; i += (unsigned long long)stride) {
        out[row_offset + i] = input[row_offset + i] * inv_rms * weight[i];
    }
}
extern "C" __global__ void yvex_rms_norm_f32_weight_f16(
    const float *input, const unsigned short *weight, float *out,
    unsigned long long hidden_size, unsigned long long row_count, float epsilon)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;
    unsigned long long row = (unsigned long long)blockIdx.x;
    unsigned long long row_offset;
    unsigned long long i;
    float sum = 0.0f;
    float inv_rms;
    if (!input || !weight || !out || hidden_size == 0ull || row >= row_count ||
        epsilon <= 0.0f) {
        return;
    }
    row_offset = row * hidden_size;
    for (i = (unsigned long long)tid; i < hidden_size; i += (unsigned long long)stride) {
        float v = input[row_offset + i];
        sum += v * v;
    }
    scratch[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        __syncthreads();
    }
    inv_rms = rsqrtf((scratch[0] / (float)hidden_size) + epsilon);
    for (i = (unsigned long long)tid; i < hidden_size; i += (unsigned long long)stride) {
        out[row_offset + i] = input[row_offset + i] * inv_rms *
                              f16_bits_to_float((unsigned int)weight[i]);
    }
}
/* Apply bounded causal RoPE to one F32 activation. */
extern "C" __global__ void yvex_rope_f32(
    const float *input, float *out, unsigned long long head_dim,
    unsigned long long position, float inverse_root)
{
    unsigned long long pair =
        ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) +
        (unsigned long long)threadIdx.x;
    unsigned long long pair_count = head_dim / 2ull;
    unsigned long long even_index;
    unsigned long long odd_index;
    unsigned long long i;
    float frequency = 1.0f;
    float angle;
    float sine;
    float cosine;
    float even;
    float odd;
    if (!input || !out || head_dim < 2ull || (head_dim & 1ull) != 0ull) {
        return;
    }
    if (pair >= pair_count) {
        return;
    }
    for (i = 0; i < pair; ++i) {
        frequency *= inverse_root;
    }
    angle = (float)position * frequency;
    sine = sinf(angle);
    cosine = cosf(angle);
    even_index = pair * 2ull;
    odd_index = even_index + 1ull;
    even = input[even_index];
    odd = input[odd_index];
    out[even_index] = (even * cosine) - (odd * sine);
    out[odd_index] = (even * sine) + (odd * cosine);
}
extern "C" __global__ void yvex_matmul_f32(
    const float *input, const float *weight, float *out,
    unsigned long long m, unsigned long long k, unsigned long long n)
{
    unsigned long long idx =
        ((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) +
        (unsigned long long)threadIdx.x;
    unsigned long long total;
    unsigned long long row;
    unsigned long long col;
    unsigned long long inner;
    float sum = 0.0f;
    const unsigned long long max_ull = ~0ull;
    if (!input || !weight || !out || m == 0ull || k == 0ull || n == 0ull ||
        m > max_ull / n || k > max_ull / n) {
        return;
    }
    total = m * n;
    if (idx >= total) {
        return;
    }
    row = idx / n;
    col = idx % n;
    for (inner = 0; inner < k; ++inner) {
        sum += input[(row * k) + inner] * weight[(inner * n) + col];
    }
    out[idx] = sum;
}
extern "C" __global__ void yvex_mlp_f32(
    const float *input, const float *gate_weight, const float *up_weight,
    const float *down_weight, float *intermediate, float *out,
    unsigned long long batch, unsigned long long hidden_dim,
    unsigned long long ffn_dim, unsigned long long expert_count,
    unsigned long long expert_id, int routed_expert_mode)
{
    unsigned long long row;
    unsigned long long j;
    unsigned long long h;
    unsigned long long gate_offset = 0ull;
    unsigned long long up_offset = 0ull;
    unsigned long long down_offset = 0ull;
    unsigned long long intermediate_total;
    unsigned long long output_total;
    unsigned long long index;
    const unsigned long long max_ull = ~0ull;
    if (blockIdx.x != 0) {
        return;
    }
    if (!input || !gate_weight || !up_weight || !down_weight || !intermediate || !out ||
        batch == 0ull || hidden_dim == 0ull || ffn_dim == 0ull ||
        batch > max_ull / ffn_dim || batch > max_ull / hidden_dim ||
        hidden_dim > max_ull / ffn_dim || ffn_dim > max_ull / hidden_dim) {
        return;
    }
    if (routed_expert_mode) {
        unsigned long long up_elements = hidden_dim * ffn_dim;
        unsigned long long down_elements = ffn_dim * hidden_dim;
        if (expert_count == 0ull || expert_id >= expert_count ||
            expert_count > max_ull / up_elements ||
            expert_count > max_ull / down_elements ||
            expert_id > max_ull / up_elements ||
            expert_id > max_ull / down_elements) {
            return;
        }
        gate_offset = expert_id * up_elements;
        up_offset = gate_offset;
        down_offset = expert_id * down_elements;
    }
    intermediate_total = batch * ffn_dim;
    output_total = batch * hidden_dim;
    for (index = (unsigned long long)threadIdx.x;
         index < intermediate_total;
         index += (unsigned long long)blockDim.x) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        float silu;
        row = index / ffn_dim;
        j = index % ffn_dim;
        for (h = 0; h < hidden_dim; ++h) {
            float x = input[(row * hidden_dim) + h];
            gate_sum += x * gate_weight[gate_offset + (h * ffn_dim) + j];
            up_sum += x * up_weight[up_offset + (h * ffn_dim) + j];
        }
        silu = gate_sum / (1.0f + expf(-gate_sum));
        intermediate[index] = silu * up_sum;
    }
    __syncthreads();
    for (index = (unsigned long long)threadIdx.x;
         index < output_total;
         index += (unsigned long long)blockDim.x) {
        float sum = 0.0f;
        row = index / hidden_dim;
        h = index % hidden_dim;
        for (j = 0; j < ffn_dim; ++j) {
            sum += intermediate[(row * ffn_dim) + j] *
                   down_weight[down_offset + (j * hidden_dim) + h];
        }
        out[index] = sum;
    }
}
extern "C" __global__ void yvex_attention_f32(
    const float *query, const float *keys, const float *values,
    float *score_scratch, float *probability_scratch, float *out,
    unsigned long long seq_len, unsigned long long position,
    unsigned long long head_dim, float scale, int causal)
{
    unsigned long long visible_count;
    unsigned long long i;
    unsigned long long d;
    float max_score = 0.0f;
    float sum_exp = 0.0f;
    __shared__ int softmax_valid;
    if (blockIdx.x != 0) {
        return;
    }
    if (!query || !keys || !values || !score_scratch || !probability_scratch || !out ||
        seq_len == 0ull || head_dim == 0ull || position >= seq_len ||
        seq_len > (~0ull) / head_dim) {
        return;
    }
    visible_count = causal ? position + 1ull : seq_len;
    for (i = (unsigned long long)threadIdx.x;
         i < seq_len;
         i += (unsigned long long)blockDim.x) {
        float score = 0.0f;
        if (causal && i > position) {
            score_scratch[i] = 0.0f;
            probability_scratch[i] = 0.0f;
            continue;
        }
        for (d = 0; d < head_dim; ++d) {
            score += query[d] * keys[(i * head_dim) + d];
        }
        score *= scale;
        score_scratch[i] = score;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        max_score = score_scratch[0];
        for (i = 1ull; i < visible_count; ++i) {
            if (score_scratch[i] > max_score) {
                max_score = score_scratch[i];
            }
        }
        for (i = 0; i < visible_count; ++i) {
            float e = expf(score_scratch[i] - max_score);
            probability_scratch[i] = e;
            sum_exp += e;
        }
        if (sum_exp == 0.0f) {
            softmax_valid = 0;
            for (i = 0; i < visible_count; ++i) {
                probability_scratch[i] = 0.0f;
            }
        } else {
            softmax_valid = 1;
            for (i = 0; i < visible_count; ++i) {
                probability_scratch[i] = probability_scratch[i] / sum_exp;
            }
        }
    }
    __syncthreads();
    for (d = (unsigned long long)threadIdx.x;
         d < head_dim;
         d += (unsigned long long)blockDim.x) {
        float value = 0.0f;
        if (softmax_valid) {
            for (i = 0; i < visible_count; ++i) {
                value += probability_scratch[i] * values[(i * head_dim) + d];
            }
        }
        out[d] = value;
    }
}
extern "C" __global__ void yvex_q8_quantize(
    unsigned char *encoded, const float *values, unsigned long long width,
    unsigned long long rows, unsigned long long groups,
    unsigned long long input_stride, int *status)
{
    __shared__ float absolute[256];
    __shared__ float signed_value[256];
    __shared__ float inverse_scale;
    unsigned long long blocks = width / YVEX_CUDA_Q8_K_BLOCK;
    unsigned long long task = blockIdx.x;
    unsigned long long row = blocks ? task / blocks : rows;
    unsigned long long block_index = blocks ? task % blocks : 0ull;
    unsigned long long input_row = groups ? row / groups : rows;
    unsigned long long group = groups ? row % groups : groups;
    unsigned int thread = threadIdx.x;
    unsigned char *block;
    float value;
    if (!status || *status || !encoded || !values || !blocks || !groups ||
        width > ~0ull / groups || input_stride < width * groups || row >= rows ||
        thread >= 256u) return;
    block = encoded + (row * blocks + block_index) * YVEX_CUDA_Q8_K_BYTES;
    value = values[input_row * input_stride + group * width +
                   block_index * YVEX_CUDA_Q8_K_BLOCK + thread];
    if (!isfinite(value)) atomicCAS(status, 0, 1);
    absolute[thread] = fabsf(value);
    signed_value[thread] = value;
    __syncthreads();
    for (unsigned int stride = 128u; stride; stride >>= 1u) {
        if (thread < stride && absolute[thread + stride] > absolute[thread]) {
            absolute[thread] = absolute[thread + stride];
            signed_value[thread] = signed_value[thread + stride];
        }
        __syncthreads();
    }
    if (!thread) {
        inverse_scale = absolute[0] == 0.0f ? 0.0f : -127.0f / signed_value[0];
        *(float *)block = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
    }
    __syncthreads();
    {
        int quantized = inverse_scale == 0.0f ? 0 : __float2int_rn(inverse_scale * value);
        quantized = quantized > 127 ? 127 : quantized < -128 ? -128 : quantized;
        block[4u + thread] = (unsigned char)(quantized & 255);
    }
    __syncthreads();
    if (thread < 16u) {
        int sum = 0;
        for (unsigned int i = 0u; i < 16u; ++i) {
            unsigned int raw = block[4u + thread * 16u + i];
            sum += raw <= 127u ? (int)raw : (int)raw - 256;
        }
        *(short *)(block + 260u + thread * 2u) = (short)sum;
    }
}
static __device__ int qtype_matvec_pair(
    unsigned long long rows, unsigned long long input_rows,
    unsigned long long *row, unsigned long long *input_row)
{
    unsigned int warp = threadIdx.x >> 5u;
    unsigned int warps = blockDim.x >> 5u;
    if (!rows || !input_rows || !warps ||
        (input_rows <= 8ull && warps < input_rows)) return 0;
    if (input_rows <= 8ull) {
        unsigned int groups = warps / (unsigned int)input_rows;
        *row = (unsigned long long)blockIdx.x * groups +
               warp / (unsigned int)input_rows;
        *input_row = warp % (unsigned int)input_rows;
    } else {
        unsigned long long tiles = (input_rows + 7ull) / 8ull;
        *row = (unsigned long long)blockIdx.x / tiles;
        *input_row = ((unsigned long long)blockIdx.x % tiles) * warps + warp;
    }
    return *row < rows && *input_row < input_rows;
}

static __device__ float qtype_dot_recover_f64(
    const unsigned char *row, const float *input, unsigned long long width,
    unsigned int qtype)
{
    double recovered = 0.0;
    for (unsigned long long i = 0ull; i < width; ++i)
        recovered += (double)qtype_value(row, i, qtype) * (double)input[i];
    return (float)recovered;
}

extern "C" __global__ void yvex_qtype_matvec(
    const unsigned char *encoded,
    unsigned long long row_bytes,
    unsigned long long row_width,
    unsigned long long start_row,
    unsigned long long row_count,
    unsigned long long input_rows,
    unsigned int qtype,
    const void *vector,
    unsigned long long input_stride,
    int q8_input,
    int block_row,
    int forensic_numeric,
    const float *additive,
    float *out,
    unsigned long long output_stride,
    int output_bf16,
    int *status)
{
    __shared__ float warp_sums[8];
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5u;
    unsigned long long input_row = input_rows, row = row_count;
    const unsigned char *row_data;
    const void *input;
    float sum;
    if (!status || *status != 0) return;
    if (!encoded || !vector || !out || !row_bytes || !row_width ||
        !row_count || !input_rows || input_stride < row_width ||
        output_stride < row_count || (q8_input && input_stride != row_width) ||
        (block_row != 0 && block_row != 1) ||
        (block_row && (qtype != YVEX_GGUF_QTYPE_F32 || q8_input ||
                       blockDim.x != 256u))) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    if (block_row) {
        unsigned long long task = blockIdx.x;
        if (row_count > ~0ull / input_rows ||
            task >= row_count * input_rows) return;
        row = task / input_rows;
        input_row = task % input_rows;
    } else if (!qtype_matvec_pair(row_count, input_rows, &row, &input_row))
        return;
    row_data = encoded + (start_row + row) * row_bytes;
    input = q8_input
        ? (const void *)((const unsigned char *)vector +
                         input_row * (row_width / YVEX_CUDA_Q8_K_BLOCK) * YVEX_CUDA_Q8_K_BYTES)
        : (const void *)((const float *)vector + input_row * input_stride);
    if (forensic_numeric) {
        if ((block_row ? threadIdx.x : lane) != 0u) return;
        sum = qtype_dot_recover_f64(
            row_data, (const float *)input, row_width, qtype);
    } else if (block_row) {
        const float *weight = (const float *)row_data;
        const float *values = (const float *)input;
        sum = 0.0f;
        for (unsigned long long i = threadIdx.x; i < row_width; i += blockDim.x) {
            float value = values[i];
            if (!isfinite(weight[i]) || !isfinite(value)) atomicCAS(status, 0, 1);
            else sum = fmaf(weight[i], value, sum);
        }
        for (unsigned int offset = 16u; offset; offset >>= 1u)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (!lane) warp_sums[warp] = sum;
        __syncthreads();
        if (!warp) {
            sum = lane < 8u ? warp_sums[lane] : 0.0f;
            for (unsigned int offset = 16u; offset; offset >>= 1u)
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (warp || lane) return;
    } else {
        if (q8_input) {
            unsigned long long blocks = row_width / YVEX_CUDA_Q8_K_BLOCK;
            if (!blocks || row_bytes % blocks) {
                if (!lane) atomicCAS(status, 0, 2);
                return;
            }
            sum = q8_warp_dot(row_data, (const unsigned char *)input, blocks,
                              row_bytes / blocks, qtype);
        } else {
            sum = qtype_warp_dot(
                row_data, (const float *)input, row_width, qtype, status);
        }
        if (lane) return;
    }
    /* A finite dot can overflow before opposite terms cancel. The exceptional row alone
       uses FP64; ordinary rows retain their parallel F32 execution order. */
    if (!q8_input && !isfinite(sum) && *status == 0)
        sum = qtype_dot_recover_f64(
            row_data, (const float *)input, row_width, qtype);
    float value = additive
        ? __fadd_rn(sum, additive[input_row * output_stride + row]) : sum;
    if (!isfinite(value)) atomicCAS(status, 0, 1);
    else out[input_row * output_stride + row] =
        output_bf16 ? float_to_bf16_rne(value) : value;
}

/* Decode keeps each group's activation distinct while one grid covers the complete
 * group-major matrix. The row dot is deliberately identical to yvex_qtype_matvec. */
extern "C" __global__ void yvex_qtype_grouped_decode(
    const unsigned char *encoded,
    unsigned long long row_bytes,
    unsigned long long row_width,
    unsigned long long group_count,
    unsigned long long group_rows,
    unsigned long long blocks_per_group,
    unsigned int qtype,
    const float *vector,
    unsigned long long input_stride,
    float *out,
    unsigned long long output_stride,
    int output_bf16,
    int *status)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5u;
    unsigned int warps = blockDim.x >> 5u;
    unsigned long long group, local_block, local_row, row;
    const unsigned char *row_data;
    const float *input;
    float sum;

    if (!status || *status != 0) return;
    if (!encoded || !vector || !out || !row_bytes || !row_width ||
        !group_count || !group_rows || !blocks_per_group || !warps ||
        (blockDim.x & 31u) != 0u ||
        group_count > ~0ull / group_rows ||
        group_count > ~0ull / row_width ||
        input_stride < group_count * row_width ||
        output_stride < group_count * group_rows) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    group = (unsigned long long)blockIdx.x / blocks_per_group;
    local_block = (unsigned long long)blockIdx.x % blocks_per_group;
    if (group >= group_count) return;
    local_row = local_block * warps + warp;
    if (local_row >= group_rows) return;
    row = group * group_rows + local_row;
    row_data = encoded + row * row_bytes;
    input = vector + group * row_width;
    sum = qtype_warp_dot(row_data, input, row_width, qtype, status);
    if (lane) return;
    if (!isfinite(sum) && *status == 0)
        sum = qtype_dot_recover_f64(row_data, input, row_width, qtype);
    if (!isfinite(sum)) atomicCAS(status, 0, 1);
    else out[row] = output_bf16 ? float_to_bf16_rne(sum) : sum;
}

extern "C" __global__ void yvex_qtype_split_matvec(
    const unsigned char *encoded,
    unsigned long long row_bytes,
    unsigned long long row_width,
    unsigned long long start_row,
    unsigned long long row_count,
    unsigned long long input_rows,
    unsigned int qtype,
    const float *head,
    const float *tail,
    unsigned long long head_width,
    const float *additive,
    float *out,
    int output_bf16,
    int *status)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned long long input_row = input_rows, row = row_count;
    float sum = 0.0f;
    if (!status || *status != 0) return;
    if (!encoded || !head || !tail || !out || !row_bytes || !row_width ||
        !row_count || !input_rows || !head_width || head_width >= row_width) {
        if (!lane) atomicCAS(status, 0, 2);
        return;
    }
    if (!qtype_matvec_pair(row_count, input_rows, &row, &input_row)) return;
    const unsigned char *row_data = encoded + (start_row + row) * row_bytes;
    unsigned long long tail_width = row_width - head_width;
    const float *head_row = head + input_row * head_width;
    const float *tail_row = tail + input_row * tail_width;
    for (unsigned long long i = lane; i < row_width; i += 32ull) {
        float weight = qtype_value(row_data, i, qtype);
        float value = float_to_bf16_rne(
            i < head_width ? head_row[i] : tail_row[i - head_width]);
        if (!isfinite(weight) || !isfinite(value)) atomicCAS(status, 0, 1);
        else sum = fmaf(weight, value, sum);
    }
    for (unsigned int offset = 16u; offset; offset >>= 1u)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0u) {
        float value = additive
            ? __fadd_rn(sum, additive[input_row * row_count + row]) : sum;
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else out[input_row * row_count + row] =
            output_bf16 ? float_to_bf16_rne(value) : value;
    }
}

extern "C" __global__ void yvex_encoded_row_decode(
    const unsigned char *encoded, unsigned long long count,
    unsigned int qtype, float *out, int *status)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (!status) return;
    if (*status != 0 || index >= count) return;
    if (!encoded || !out || !count) {
        atomicCAS(status, 0, 2);
        return;
    }
    float value = qtype_value(encoded, index, qtype);
    if (!isfinite(value)) atomicCAS(status, 0, 1);
    else out[index] = value;
}

extern "C" __global__ void yvex_qtype_gather(
    const unsigned char *encoded, unsigned long long row_bytes,
    unsigned long long row_width, unsigned long long row_count,
    const unsigned int *row_ids, unsigned long long selected_rows,
    unsigned int qtype, float *out, int *status)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long row = row_width ? index / row_width : selected_rows;
    unsigned long long column = row_width ? index % row_width : 0ull;
    unsigned long long selected;
    const unsigned char *row_data;
    float value;
    if (!status || *status != 0 || row >= selected_rows) return;
    if (!encoded || !row_bytes || !row_width || !row_count || !row_ids || !out) {
        atomicCAS(status, 0, 2);
        return;
    }
    selected = (unsigned long long)row_ids[row];
    if (selected >= row_count) {
        atomicCAS(status, 0, 2);
        return;
    }
    row_data = encoded + selected * row_bytes;
    value = qtype_value(row_data, column, qtype);
    if (!isfinite(value)) atomicCAS(status, 0, 1);
    else out[index] = value;
}

/* Recover finite BF16-range values when their ordinary F32 square sum overflows. The rare
   recovery is serial and double-precision so the established parallel fast path is unchanged. */
static __device__ double finite_square_sum_f32(
    const float *values, unsigned long long count, float *square_terms,
    int *status, int *active)
{
    unsigned int lane = threadIdx.x;
    float square_sum = 0.0f;
    for (unsigned long long i = (unsigned long long)lane; i < count;
         i += (unsigned long long)blockDim.x) {
        float value = values[i];
        if (!isfinite(value)) {
            atomicCAS(status, 0, 1);
            atomicExch(active, 0);
        } else {
            square_sum = fmaf(value, value, square_sum);
        }
    }
    square_terms[lane] = square_sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1u; offset; offset >>= 1u) {
        if (lane < offset)
            square_terms[lane] += square_terms[lane + offset];
        __syncthreads();
    }
    if (!*active) return 0.0;
    if (lane == 0u && !isfinite(square_terms[0])) {
        double recovered = 0.0;
        for (unsigned long long i = 0ull; i < count; ++i)
            recovered += (double)values[i] * (double)values[i];
        return recovered;
    }
    return (double)square_terms[0];
}

extern "C" __global__ void yvex_attention_weighted_norm(
    float *values, unsigned long long count, const unsigned char *weight,
    unsigned int weight_qtype, double epsilon, unsigned long long vectors,
    int *status)
{
    extern __shared__ float scratch_terms[];
    __shared__ double inverse;
    __shared__ int active;
    if (!status) return;
    if ((unsigned long long)blockIdx.x >= vectors) return;
    if (!values || !weight || !count || !vectors || epsilon <= 0.0) {
        if (threadIdx.x == 0u) atomicCAS(status, 0, 2);
        return;
    }
    if (threadIdx.x == 0u) active = *status == 0;
    __syncthreads();
    if (!active) return;
    values += (unsigned long long)blockIdx.x * count;
    double total = finite_square_sum_f32(
        values, count, scratch_terms, status, &active);
    if (!active) return;
    if (threadIdx.x == 0u) {
        double mean = total;
        mean = __ddiv_rn(mean, (double)count);
        inverse = __ddiv_rn(1.0, sqrt(__dadd_rn(mean, epsilon)));
        if (!isfinite(inverse)) {
            atomicCAS(status, 0, 1);
            active = 0;
        }
    }
    __syncthreads();
    if (!active) return;
    for (unsigned long long i = (unsigned long long)threadIdx.x; i < count;
         i += (unsigned long long)blockDim.x) {
        double scale = (double)qtype_value(weight, i, weight_qtype);
        double result = __dmul_rn(
            __dmul_rn((double)values[i], inverse), scale);
        float published = (float)result;
        if (!isfinite(scale) || !isfinite(result) || !isfinite(published)) {
            atomicCAS(status, 0, 1);
            return;
        }
        values[i] = float_to_bf16_rne(published);
    }
}

extern "C" __global__ void yvex_attention_unit_norm(
    float *values, unsigned long long vector_count,
    unsigned long long vector_width, double epsilon, int *status)
{
    extern __shared__ double partial[];
    unsigned long long vector_index = (unsigned long long)blockIdx.x;
    unsigned int lane = threadIdx.x;
    double sum = 0.0;
    float *vector;
    if (!status) return;
    if (*status != 0 || vector_index >= vector_count) return;
    if (!values || !vector_count || !vector_width || epsilon <= 0.0) {
        if (lane == 0u) atomicCAS(status, 0, 2);
        return;
    }
    vector = values + vector_index * vector_width;
    for (unsigned long long i = (unsigned long long)lane; i < vector_width;
         i += (unsigned long long)blockDim.x) {
        double value = (double)vector[i];
        if (!isfinite(value)) {
            atomicCAS(status, 0, 1);
            continue;
        }
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
        if (lane < offset) partial[lane] += partial[lane + offset];
        __syncthreads();
    }
    if (*status != 0) return;
    double inverse = rsqrt(partial[0] / (double)vector_width + epsilon);
    if (!isfinite(inverse)) {
        if (lane == 0u) atomicCAS(status, 0, 1);
        return;
    }
    for (unsigned long long i = (unsigned long long)lane; i < vector_width;
         i += (unsigned long long)blockDim.x) {
        float published = (float)((double)vector[i] * inverse);
        if (!isfinite(published)) atomicCAS(status, 0, 1);
        else vector[i] = float_to_bf16_rne(published);
    }
}

static __device__ double attention_yarn_frequency(
    unsigned long long pair, unsigned long long rope_dims,
    unsigned long long theta, unsigned long long scaling_factor,
    unsigned long long original_context, unsigned long long beta_fast,
    unsigned long long beta_slow)
{
    const double pi = 3.14159265358979323846264338327950288;
    double exponent = (double)(pair * 2ull) / (double)rope_dims;
    double frequency = 1.0 / pow((double)theta, exponent);
    if (original_context && scaling_factor) {
        double denominator = 2.0 * log((double)theta);
        double low = floor((double)rope_dims *
            log((double)original_context /
                ((double)beta_fast * 2.0 * pi)) / denominator);
        double high = ceil((double)rope_dims *
            log((double)original_context /
                ((double)beta_slow * 2.0 * pi)) / denominator);
        double lane = (double)pair;
        if (low < 0.0) low = 0.0;
        if (high > (double)rope_dims - 1.0)
            high = (double)rope_dims - 1.0;
        if (low == high) high += 0.001;
        double ramp = (lane - low) / (high - low);
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        double smooth = 1.0 - ramp;
        frequency = frequency / (double)scaling_factor * (1.0 - smooth) +
                    frequency * smooth;
    }
    return frequency;
}

extern "C" __global__ void yvex_attention_yarn_rope(
    float *values, unsigned long long vectors_per_token,
    unsigned long long token_count,
    unsigned long long vector_width, unsigned long long rope_dims,
    unsigned long long token_position, unsigned long long theta,
    unsigned long long scaling_factor, unsigned long long original_context,
    unsigned long long beta_fast, unsigned long long beta_slow,
    int inverse, int *status)
{
    unsigned long long pair =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long pairs_per_vector;
    unsigned long long total;
    if (!status) return;
    if (*status != 0) return;
    if (!values || !vectors_per_token || !token_count || !rope_dims ||
        rope_dims > vector_width ||
        (rope_dims & 1ull) || theta <= 1ull || !scaling_factor ||
        (original_context && (!beta_slow || beta_fast <= beta_slow))) {
        atomicCAS(status, 0, 2);
        return;
    }
    pairs_per_vector = rope_dims / 2ull;
    if (token_count > ~0ull / vectors_per_token ||
        token_count * vectors_per_token > ~0ull / pairs_per_vector) {
        atomicCAS(status, 0, 2);
        return;
    }
    total = token_count * vectors_per_token * pairs_per_vector;
    if (pair >= total) return;
    unsigned long long vector_index = pair / pairs_per_vector;
    unsigned long long token_index = vector_index / vectors_per_token;
    unsigned long long local_pair = pair % pairs_per_vector;
    unsigned long long start = vector_width - rope_dims;
    unsigned long long offset = vector_index * vector_width + start +
                                local_pair * 2ull;
    double frequency = attention_yarn_frequency(
        local_pair, rope_dims, theta, scaling_factor, original_context,
        beta_fast, beta_slow);
    double angle = (double)(token_position + token_index) * frequency;
    double c = cos(angle);
    double s = inverse ? -sin(angle) : sin(angle);
    double x = (double)values[offset];
    double y = (double)values[offset + 1ull];
    double left = x * c - y * s;
    double right = x * s + y * c;
    float published_left = (float)left;
    float published_right = (float)right;
    if (!isfinite(left) || !isfinite(right) || !isfinite(published_left) ||
        !isfinite(published_right)) atomicCAS(status, 0, 1);
    else {
        values[offset] = float_to_bf16_rne(published_left);
        values[offset + 1ull] = float_to_bf16_rne(published_right);
    }
}

static __device__ float activation_fp8_decode(unsigned int code)
{
    unsigned int sign = code & 0x80u;
    unsigned int exponent = (code >> 3u) & 0x0fu;
    unsigned int mantissa = code & 0x07u;
    float value;
    if ((code & 0x7fu) == 0u) return sign ? -0.0f : 0.0f;
    if ((code & 0x7fu) == 0x7fu) return __uint_as_float(0x7fc00000u);
    value = exponent == 0u
        ? ldexpf((float)mantissa / 8.0f, -6)
        : ldexpf(1.0f + (float)mantissa / 8.0f, (int)exponent - 7);
    return sign ? -value : value;
}

static __device__ unsigned int activation_fp8_encode(float value)
{
    float magnitude = fabsf(value);
    float best_error = INFINITY;
    unsigned int best = 0u;
    int negative = signbit(value);
    if (!isfinite(value)) return negative ? 0xffu : 0x7fu;
    if (magnitude > 448.0f) magnitude = 448.0f;
    for (unsigned int code = 0u; code < 0x7fu; ++code) {
        float error = fabsf(activation_fp8_decode(code) - magnitude);
        if (error < best_error ||
            (error == best_error && !(code & 1u) && (best & 1u))) {
            best_error = error;
            best = code;
        }
    }
    return negative ? best | 0x80u : best;
}

static __device__ float activation_fp4_decode(unsigned int code)
{
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                            2.0f, 3.0f, 4.0f, 6.0f};
    float value = table[code & 7u];
    return (code & 8u) ? -value : value;
}

static __device__ unsigned int activation_fp4_encode(float value)
{
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                            2.0f, 3.0f, 4.0f, 6.0f};
    float magnitude = fabsf(value);
    float best_error;
    unsigned int best = 0u;
    if (isnan(value)) return signbit(value) ? 8u : 0u;
    if (magnitude > 6.0f) magnitude = 6.0f;
    best_error = magnitude;
    for (unsigned int code = 1u; code < 8u; ++code) {
        float error = fabsf(magnitude - table[code]);
        if (error < best_error ||
            (error == best_error && !(code & 1u) && (best & 1u))) {
            best_error = error;
            best = code;
        }
    }
    return signbit(value) ? best | 8u : best;
}

static __device__ unsigned int activation_e8m0_encode(float value)
{
    int exponent;
    float fraction;
    if (!isfinite(value) || value <= 0.0f) return 0xffu;
    fraction = frexpf(value, &exponent);
    if (fraction > 0.5f) exponent++;
    exponent += 126;
    if (exponent < 0) return 0u;
    if (exponent > 254) return 254u;
    return (unsigned int)exponent;
}

static __device__ float activation_power_two_ceil(float value)
{
    int exponent;
    float fraction;
    if (!isfinite(value) || value <= 0.0f) return 0.0f;
    fraction = frexpf(value, &exponent);
    if (fraction > 0.5f) exponent++;
    return ldexpf(1.0f, exponent - 1);
}
/*
 * Execute Hadamard plus pinned FP8/FP4 UE8M0 fake quantization on device.
 *
 * Transforms it in deterministic stage order.
 */
extern "C" __global__ void yvex_attention_activation_quantize(
    float *values, unsigned long long vector_count,
    unsigned long long vector_width, unsigned long long vector_stride,
    unsigned long long block_width,
    unsigned int quantization, int hadamard, int *status)
{
    __shared__ unsigned int maximum_bits;
    __shared__ float quantization_scale;
    __shared__ int active;
    unsigned long long vector_index = (unsigned long long)blockIdx.x;
    unsigned int thread = threadIdx.x;
    if (!status) return;
    if (vector_index >= vector_count) return;
    if (!values || !vector_count || !vector_width ||
        vector_stride < vector_width || !block_width ||
        vector_width % block_width ||
        (quantization != 1u && quantization != 2u)) {
        if (thread == 0u) atomicCAS(status, 0, 2);
        return;
    }
    if (thread == 0u) active = *status == 0;
    __syncthreads();
    if (!active) return;
    float *vector = values + vector_index * vector_stride;
    if (hadamard) {
        if ((vector_width & (vector_width - 1ull)) != 0ull ||
            vector_width > 1024ull) {
            if (thread == 0u) atomicCAS(status, 0, 2);
            return;
        }
        /* Each butterfly pair is independent within a stage. The stage
           barrier preserves the source-authored Hadamard operation order. */
        for (unsigned long long step = 1ull; step < vector_width; step *= 2ull) {
            for (unsigned long long pair = (unsigned long long)thread;
                 pair < vector_width / 2ull;
                 pair += (unsigned long long)blockDim.x) {
                unsigned long long block = (pair / step) * step * 2ull;
                unsigned long long lane = pair % step;
                float left = vector[block + lane];
                float right = vector[block + lane + step];
                vector[block + lane] = left + right;
                vector[block + lane + step] = left - right;
            }
            __syncthreads();
        }
        float scale = rsqrtf((float)vector_width);
        for (unsigned long long i = (unsigned long long)thread; i < vector_width;
             i += (unsigned long long)blockDim.x)
            vector[i] *= scale;
        __syncthreads();
    }
    for (unsigned long long offset = 0ull; offset < vector_width;
         offset += block_width) {
        if (thread == 0u)
            maximum_bits = __float_as_uint(quantization == 1u ? 1.0e-4f : 0.0f);
        __syncthreads();
        for (unsigned long long i = (unsigned long long)thread; i < block_width;
             i += (unsigned long long)blockDim.x) {
            float value = vector[offset + i];
            if (!isfinite(value)) {
                atomicCAS(status, 0, 1);
                atomicExch(&active, 0);
                continue;
            }
            float magnitude = fabsf(value);
            atomicMax(&maximum_bits, __float_as_uint(magnitude));
        }
        __syncthreads();
        if (!active) return;
        if (thread == 0u) {
            float amax = __uint_as_float(maximum_bits);
            float minimum = 6.0f * ldexpf(1.0f, -126);
            if (quantization == 2u && amax < minimum) amax = minimum;
            float scale = activation_power_two_ceil(
                amax / (quantization == 1u ? 448.0f : 6.0f));
            unsigned int scale_code = activation_e8m0_encode(scale);
            quantization_scale = e8m0_bits_to_float(scale_code);
            if (!isfinite(quantization_scale) || quantization_scale <= 0.0f) {
                atomicCAS(status, 0, 1);
                active = 0;
            }
        }
        __syncthreads();
        if (!active) return;
        for (unsigned long long i = (unsigned long long)thread; i < block_width;
             i += (unsigned long long)blockDim.x) {
            float normalized = vector[offset + i] / quantization_scale;
            if (quantization == 1u) {
                if (normalized > 448.0f) normalized = 448.0f;
                if (normalized < -448.0f) normalized = -448.0f;
                vector[offset + i] = float_to_bf16_rne(
                    activation_fp8_decode(activation_fp8_encode(normalized)) *
                    quantization_scale);
            } else if (quantization == 2u) {
                vector[offset + i] = float_to_bf16_rne(
                    activation_fp4_decode(activation_fp4_encode(normalized)) *
                    quantization_scale);
            } else {
                atomicCAS(status, 0, 2);
                atomicExch(&active, 0);
            }
        }
        __syncthreads();
        if (!active) return;
    }
}

extern "C" __global__ void yvex_residual_mhc_pre(
    float *residual, const float *linear_mix, const float *scale,
    const float *base, unsigned long long streams,
    unsigned long long stream_width, unsigned long long mixing_rows,
    unsigned long long sinkhorn_iterations, double rms_epsilon,
    double mhc_epsilon, double post_multiplier, float *collapsed,
    float *post, float *combination, unsigned long long row_count, int *status)
{
    extern __shared__ double shared[];
    __shared__ int active;
    unsigned long long batch = (unsigned long long)blockIdx.x;
    unsigned int thread = threadIdx.x;
    if (!status || batch >= row_count) return;
    if (!residual || !linear_mix || !scale || !base || !collapsed || !post ||
        !combination || !row_count || !streams || !stream_width || !sinkhorn_iterations ||
        mixing_rows != (streams + 2ull) * streams || rms_epsilon <= 0.0 ||
        mhc_epsilon <= 0.0 || post_multiplier <= 0.0) {
        if (thread == 0u) atomicCAS(status, 0, 2);
        return;
    }
    if (thread == 0u) active = *status == 0;
    __syncthreads();
    if (!active) return;
    unsigned long long expanded = streams * stream_width;
    residual += batch * expanded;
    linear_mix += batch * mixing_rows;
    collapsed += batch * stream_width;
    post += batch * streams;
    combination += batch * streams * streams;
    double *pre_weights = shared;
    double *inverse = shared + streams;
    float *square_terms = (float *)(inverse + 1);
    for (unsigned long long lane = (unsigned long long)thread; lane < expanded;
         lane += (unsigned long long)blockDim.x) {
        float value = float_to_bf16_rne(residual[lane]);
        residual[lane] = value;
        if (!isfinite(value)) {
            atomicCAS(status, 0, 1);
            atomicExch(&active, 0);
        }
    }
    __syncthreads();
    if (!active) return;
    double total = finite_square_sum_f32(
        residual, expanded, square_terms, status, &active);
    if (!active) return;
    if (thread == 0u) {
        *inverse = 1.0 / sqrt(total / (double)expanded + rms_epsilon);
        if (!isfinite(*inverse)) {
            atomicCAS(status, 0, 1);
            active = 0;
        }
    }
    __syncthreads();
    if (!active) return;
    if ((unsigned long long)thread < streams) {
        unsigned long long stream = (unsigned long long)thread;
        double pre_arg = (double)linear_mix[stream] * *inverse *
                         (double)scale[0] + (double)base[stream];
        double pre_exp = exp(pre_arg >= 0.0 ? -pre_arg : pre_arg);
        double pre = pre_arg >= 0.0 ? 1.0 / (1.0 + pre_exp)
                                    : pre_exp / (1.0 + pre_exp);
        unsigned long long post_index = streams + stream;
        double post_arg = (double)linear_mix[post_index] * *inverse *
                          (double)scale[1] + (double)base[post_index];
        double post_exp = exp(post_arg >= 0.0 ? -post_arg : post_arg);
        double post_sigmoid = post_arg >= 0.0 ? 1.0 / (1.0 + post_exp)
                                              : post_exp / (1.0 + post_exp);
        pre_weights[stream] = pre + mhc_epsilon;
        post[stream] = (float)(post_multiplier * post_sigmoid);
        for (unsigned long long target = 0ull; target < streams; ++target) {
            unsigned long long index = 2ull * streams + stream * streams + target;
            combination[stream * streams + target] =
                (float)((double)linear_mix[index] * *inverse *
                        (double)scale[2] + (double)base[index]);
        }
    }
    __syncthreads();
    for (unsigned long long lane = (unsigned long long)thread; lane < stream_width;
         lane += (unsigned long long)blockDim.x) {
        float total = 0.0f;
        for (unsigned long long stream = 0ull; stream < streams; ++stream)
            total += (float)(pre_weights[stream] *
                (double)residual[stream * stream_width + lane]);
        collapsed[lane] = total;
    }
    if ((unsigned long long)thread < streams) {
        unsigned long long row = (unsigned long long)thread;
        double maximum = -INFINITY;
        double total = 0.0;
        for (unsigned long long column = 0ull; column < streams; ++column)
            maximum = maximum > (double)combination[row * streams + column]
                ? maximum : (double)combination[row * streams + column];
        for (unsigned long long column = 0ull; column < streams; ++column) {
            double value = exp(
                (double)combination[row * streams + column] - maximum);
            combination[row * streams + column] = (float)value;
            total += value;
        }
        if (!isfinite(total) || total <= 0.0) {
            atomicCAS(status, 0, 1);
            atomicExch(&active, 0);
        } else {
            for (unsigned long long column = 0ull; column < streams; ++column)
                combination[row * streams + column] = (float)(
                    (double)combination[row * streams + column] / total + mhc_epsilon);
        }
    }
    __syncthreads();
    if (!active) return;
    for (unsigned long long iteration = 0ull;
         iteration < sinkhorn_iterations; ++iteration) {
        if (iteration != 0ull && (unsigned long long)thread < streams) {
            unsigned long long row = (unsigned long long)thread;
            double total = 0.0;
            for (unsigned long long column = 0ull; column < streams; ++column)
                total += combination[row * streams + column];
            for (unsigned long long column = 0ull; column < streams; ++column)
                combination[row * streams + column] =
                    (float)((double)combination[row * streams + column] /
                            (total + mhc_epsilon));
        }
        __syncthreads();
        if ((unsigned long long)thread < streams) {
            unsigned long long column = (unsigned long long)thread;
            double total = 0.0;
            for (unsigned long long row = 0ull; row < streams; ++row)
                total += combination[row * streams + column];
            for (unsigned long long row = 0ull; row < streams; ++row)
                combination[row * streams + column] =
                    (float)((double)combination[row * streams + column] /
                            (total + mhc_epsilon));
        }
        __syncthreads();
    }
    for (unsigned long long lane = (unsigned long long)thread; lane < stream_width;
         lane += (unsigned long long)blockDim.x) {
        if (!isfinite(collapsed[lane]) || !isfinite(post[lane % streams])) {
            atomicCAS(status, 0, 1);
            atomicExch(&active, 0);
            continue;
        }
        collapsed[lane] = float_to_bf16_rne(collapsed[lane]);
    }
}

extern "C" __global__ void yvex_residual_mhc_post(
    const float *core, const float *residual, const float *post,
    const float *combination, unsigned long long streams,
    unsigned long long stream_width, float *output,
    unsigned long long row_count, int *status)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long expanded = streams * stream_width;
    unsigned long long total = row_count * expanded;
    if (!status || *status != 0 || index >= total) return;
    if (!core || !residual || !post || !combination || !output || !streams ||
        !stream_width || !row_count) {
        atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long batch = index / expanded;
    unsigned long long local = index % expanded;
    unsigned long long target = local / stream_width;
    unsigned long long lane = local % stream_width;
    core += batch * stream_width;
    residual += batch * expanded;
    post += batch * streams;
    combination += batch * streams * streams;
    double value = (double)post[target] * (double)core[lane];
    for (unsigned long long source = 0ull; source < streams; ++source)
        value += (double)combination[source * streams + target] *
                 (double)residual[source * stream_width + lane];
    float published = (float)value;
    if (!isfinite(published)) atomicCAS(status, 0, 1);
    else output[index] = float_to_bf16_rne(published);
}
/* Reduce a row's residual streams before an explicit target-feature materialization. */
extern "C" __global__ void yvex_transformer_feature_mean(
    const float *expanded, unsigned long long token_count,
    unsigned long long streams, unsigned long long width,
    float *output, float *resident_output,
    unsigned long long resident_row_offset,
    unsigned long long resident_row_stride,
    unsigned long long resident_column_offset, int *status)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long count = token_count * width;
    if (!status || *status || index >= count) return;
    if (!expanded || !output || !token_count || !streams || !width) {
        atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long token = index / width;
    unsigned long long lane = index % width;
    double sum = 0.0;
    for (unsigned long long stream = 0ull; stream < streams; ++stream)
        sum += (double)expanded[(token * streams + stream) * width + lane];
    float value = (float)(sum / (double)streams);
    if (!isfinite(value)) atomicCAS(status, 0, 1);
    else {
        output[index] = value;
        if (resident_output)
            resident_output[(resident_row_offset + token) * resident_row_stride +
                            resident_column_offset + lane] = value;
    }
}
/*
 * Collapse the final mHC streams and apply transformer-owned RMSNorm.
 *
 * Device-resident expanded rows and exact decoded final weights.
 */
extern "C" __global__ void yvex_transformer_final(
    const float *expanded, const float *function, const float *base,
    const float *scale, const float *norm, unsigned long long token_count,
    unsigned long long streams, unsigned long long width, double epsilon,
    double mhc_epsilon, float *pre_output, float *output, int *status)
{
    extern __shared__ double final_scratch[];
    unsigned int lane = threadIdx.x;
    unsigned long long token = blockIdx.x;
    unsigned long long expanded_width = streams * width;
    const float *input;
    float *row;
    double sum = 0.0, inverse;
    if (!status || *status || token >= token_count) return;
    if (!expanded || !function || !base || !scale || !norm || !output ||
        !streams || !width || epsilon <= 0.0 || mhc_epsilon <= 0.0) {
        if (lane == 0u) atomicCAS(status, 0, 2);
        return;
    }
    input = expanded + token * expanded_width;
    row = output + token * width;
    for (unsigned long long index = lane; index < expanded_width; index += blockDim.x)
        sum += (double)input[index] * (double)input[index];
    final_scratch[lane] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
        if (lane < offset) final_scratch[lane] += final_scratch[lane + offset];
        __syncthreads();
    }
    inverse = rsqrt(final_scratch[0] / (double)expanded_width + epsilon);
    for (unsigned long long index = lane; index < width; index += blockDim.x) row[index] = 0.0f;
    __syncthreads();
    for (unsigned long long stream = 0ull; stream < streams; ++stream) {
        sum = 0.0;
        for (unsigned long long index = lane; index < expanded_width; index += blockDim.x)
            sum += (double)function[stream * expanded_width + index] * (double)input[index];
        final_scratch[lane] = sum;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
            if (lane < offset)
                final_scratch[lane] += final_scratch[lane + offset];
            __syncthreads();
        }
        if (lane == 0u)
            final_scratch[0] =
                1.0 / (1.0 + exp(-(final_scratch[0] * inverse * (double)scale[0] +
                                   (double)base[stream]))) + mhc_epsilon;
        __syncthreads();
        for (unsigned long long index = lane; index < width; index += blockDim.x)
            row[index] +=
                (float)(final_scratch[0] * (double)input[stream * width + index]);
        __syncthreads();
    }
    sum = 0.0;
    for (unsigned long long index = lane; index < width; index += blockDim.x) {
        row[index] = float_to_bf16_rne(row[index]);
        if (pre_output) pre_output[token * width + index] = row[index];
        sum += (double)row[index] * (double)row[index];
    }
    final_scratch[lane] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset; offset >>= 1) {
        if (lane < offset) final_scratch[lane] += final_scratch[lane + offset];
        __syncthreads();
    }
    inverse = rsqrt(final_scratch[0] / (double)width + epsilon);
    for (unsigned long long index = lane; index < width; index += blockDim.x) {
        float value = float_to_bf16_rne((float)((double)row[index] * inverse * norm[index]));
        if (!isfinite(value)) atomicCAS(status, 0, 1);
        else row[index] = value;
    }
}
/*
 * Execute one admitted ratio-4 or ratio-128 compressor transition on device.
 *
 * Writes only transaction-local outputs. Host owns publication and rollback.
 */
extern "C" __global__ void yvex_attention_rolling_state(
    const float *before_kv, const float *before_score,
    const float *token_kv, const float *token_score, const float *ape,
    float *after_kv, float *after_score, float *compressed,
    unsigned long long ratio, unsigned long long head_dim,
    unsigned long long state_width, unsigned long long state_slots,
    unsigned long long cursor, int overlap, int emit, int *status)
{
    unsigned int thread = threadIdx.x;
    unsigned long long extent;
    unsigned long long insert_slot;
    if (!status) return;
    if (*status != 0 || blockIdx.x != 0u) return;
    if (!before_kv || !before_score || !token_kv || !token_score || !ape ||
        !after_kv || !after_score || !compressed || !ratio || !head_dim ||
        !state_width || !state_slots || cursor >= ratio) {
        if (thread == 0u) atomicCAS(status, 0, 2);
        return;
    }
    if (state_width > ~0ull / state_slots ||
        (overlap && ratio > ~0ull - cursor)) {
        if (thread == 0u) atomicCAS(status, 0, 1);
        return;
    }
    extent = state_width * state_slots;
    insert_slot = overlap ? ratio + cursor : cursor;
    if (insert_slot >= state_slots) {
        if (thread == 0u) atomicCAS(status, 0, 1);
        return;
    }
    for (unsigned long long i = (unsigned long long)thread; i < extent;
         i += (unsigned long long)blockDim.x) {
        after_kv[i] = before_kv[i];
        after_score[i] = before_score[i];
    }
    for (unsigned long long lane = (unsigned long long)thread;
         lane < state_width; lane += (unsigned long long)blockDim.x) {
        float kv = token_kv[lane];
        float score = token_score[lane] + ape[lane];
        if (!isfinite(kv) || !isfinite(score)) atomicCAS(status, 0, 1);
        after_kv[insert_slot * state_width + lane] = kv;
        after_score[insert_slot * state_width + lane] = score;
    }
    __syncthreads();
    if (emit) {
        for (unsigned long long lane = (unsigned long long)thread;
             lane < head_dim; lane += (unsigned long long)blockDim.x) {
            double maximum = -INFINITY;
            double denominator = 0.0;
            double value = 0.0;
            for (unsigned long long slot = 0ull; slot < ratio; ++slot) {
                double score = (double)after_score[slot * state_width + lane];
                if (score > maximum) maximum = score;
                if (overlap) {
                    score = (double)after_score[(ratio + slot) * state_width +
                                                  lane + head_dim];
                    if (score > maximum) maximum = score;
                }
            }
            for (unsigned long long slot = 0ull; slot < ratio; ++slot) {
                double score = (double)after_score[slot * state_width + lane];
                double weight = exp(__dadd_rn(score, -maximum));
                denominator = __dadd_rn(denominator, weight);
                value = __dadd_rn(
                    value,
                    __dmul_rn(weight,
                              (double)after_kv[slot * state_width + lane]));
                if (overlap) {
                    score = (double)after_score[(ratio + slot) * state_width +
                                                lane + head_dim];
                    weight = exp(__dadd_rn(score, -maximum));
                    denominator = __dadd_rn(denominator, weight);
                    value = __dadd_rn(
                        value,
                        __dmul_rn(
                            weight,
                            (double)after_kv[(ratio + slot) * state_width +
                                             lane + head_dim]));
                }
            }
            float published = (float)__ddiv_rn(value, denominator);
            if (!isfinite(denominator) || denominator <= 0.0 ||
                !isfinite(value) || !isfinite(published))
                atomicCAS(status, 0, 1);
            else compressed[lane] = float_to_bf16_rne(published);
        }
        __syncthreads();
        if (overlap) {
            for (unsigned long long i = (unsigned long long)thread;
                 i < ratio * state_width; i += (unsigned long long)blockDim.x) {
                after_kv[i] = after_kv[ratio * state_width + i];
                after_score[i] = after_score[ratio * state_width + i];
            }
        }
    }
}
/*
 * Score and rank the complete admitted CSA candidate set on device.
 *
 * Writes deterministic selected indexes and counts. Host owns transaction publication.
 */
extern "C" __global__ void yvex_attention_topk(
    const float *index_query, const float *index_weights,
    const float *history_indexer, const unsigned long long *history_positions,
    unsigned long long history_count, unsigned long long history_stride,
    const float *current_indexer, const unsigned long long *current_positions,
    unsigned long long current_count, unsigned long long current_stride,
    unsigned long long heads, unsigned long long head_dim,
    unsigned long long ratio, unsigned long long query_position,
    unsigned long long k, unsigned long long *selected,
    unsigned long long *selected_positions, unsigned long long *selected_count,
    unsigned long long *valid_count, float *scores,
    unsigned long long *valid_indexes, int *status)
{
    extern __shared__ double head_terms[];
    __shared__ unsigned long long valid;
    __shared__ int active;
    __shared__ int candidate_valid;
    unsigned long long total;
    unsigned int thread = threadIdx.x;
    if (!status) return;
    if (blockIdx.x != 0u) return;
    if (!index_query || !index_weights || !selected || !selected_positions ||
        !selected_count || !valid_count || !scores || !valid_indexes || !heads ||
        !head_dim || !ratio || !k || history_count > ~0ull - current_count ||
        (history_count && (!history_indexer || !history_positions ||
                           history_stride < head_dim)) ||
        (current_count && (!current_indexer || !current_positions ||
                           current_stride < head_dim))) {
        if (thread == 0u) atomicCAS(status, 0, 2);
        return;
    }
    if (thread == 0u) {
        active = *status == 0;
        valid = 0ull;
    }
    __syncthreads();
    if (!active) return;
    total = history_count + current_count;
    for (unsigned long long candidate = 0ull; candidate < total; ++candidate) {
        const float *row;
        unsigned long long position;
        if (candidate < history_count) {
            row = history_indexer + candidate * history_stride;
            position = history_positions[candidate];
        } else {
            unsigned long long local = candidate - history_count;
            row = current_indexer + local * current_stride;
            position = current_positions[local];
        }
        if (thread == 0u) {
            candidate_valid = row && position <= query_position &&
                position <= ~0ull - ratio + 1ull &&
                position + ratio - 1ull <= query_position;
            for (unsigned long long prior = 0ull;
                 candidate_valid && prior < valid; ++prior) {
                unsigned long long prior_candidate = valid_indexes[prior];
                unsigned long long prior_position = prior_candidate < history_count
                    ? history_positions[prior_candidate]
                    : current_positions[prior_candidate - history_count];
                if (prior_position == position) {
                    atomicCAS(status, 0, 1);
                    active = 0;
                }
            }
        }
        __syncthreads();
        if (!active) return;
        if (!candidate_valid) continue;
        /* Heads are independent; lane zero retains the source-order reduction
           across each tile so ranking and tie behavior stay bit-identical. */
        double score = 0.0;
        for (unsigned long long base = 0ull; base < heads;
             base += (unsigned long long)blockDim.x) {
            unsigned long long head = base + (unsigned long long)thread;
            double contribution = 0.0;
            if (head < heads) {
                double dot = 0.0;
                const float *query = index_query + head * head_dim;
                for (unsigned long long lane = 0ull; lane < head_dim; ++lane) {
                    double term = __dmul_rn((double)query[lane], (double)row[lane]);
                    dot = __dadd_rn(dot, term);
                }
                if (dot < 0.0) dot = 0.0;
                contribution = __dmul_rn(dot, (double)index_weights[head]);
            }
            head_terms[thread] = contribution;
            __syncthreads();
            if (thread == 0u) {
                unsigned long long tile = heads - base;
                if (tile > (unsigned long long)blockDim.x) tile = blockDim.x;
                for (unsigned long long i = 0ull; i < tile; ++i)
                    score = __dadd_rn(score, head_terms[i]);
            }
            __syncthreads();
        }
        if (thread == 0u) {
            score = __dmul_rn(score, 1.0 / sqrt((double)head_dim));
            score = __dmul_rn(score, 1.0 / sqrt((double)heads));
            if (!isfinite(score)) {
                atomicCAS(status, 0, 1);
                active = 0;
            } else {
                scores[valid] = (float)score;
                valid_indexes[valid] = candidate;
                valid++;
            }
        }
        __syncthreads();
        if (!active) return;
    }
    if (thread == 0u) {
        unsigned long long chosen = valid < k ? valid : k;
        for (unsigned long long rank = 0ull; rank < chosen; ++rank) {
            unsigned long long best = ~0ull;
            for (unsigned long long i = 0ull; i < valid; ++i) {
                unsigned long long candidate = valid_indexes[i];
                unsigned long long position = candidate < history_count
                    ? history_positions[candidate]
                    : current_positions[candidate - history_count];
                int already = 0;
                for (unsigned long long prior = 0ull; prior < rank; ++prior)
                    if (selected[prior] == candidate) already = 1;
                if (already) continue;
                if (best == ~0ull || scores[i] > scores[best] ||
                    (scores[i] == scores[best] && position <
                        (valid_indexes[best] < history_count
                            ? history_positions[valid_indexes[best]]
                            : current_positions[valid_indexes[best] - history_count])))
                    best = i;
            }
            selected[rank] = valid_indexes[best];
            selected_positions[rank] = selected[rank] < history_count
                ? history_positions[selected[rank]]
                : current_positions[selected[rank] - history_count];
        }
        *selected_count = chosen;
        *valid_count = valid;
    }
}
