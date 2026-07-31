/*
 * Implement the admitted CUDA primitive arithmetic embedded into the generated PTX bundle.
 *
 * Every exported kernel is resolved through the generated PTX bundle and its matching host owner
 * validates byte/rank geometry first. A qtype row dot is primitive compute proof, not model
 * execution.
 */
#include <yvex/qtype.h>

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

static __device__ float f16_bits_to_float(unsigned int h)
{
    unsigned int sign = (h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1fu;
    unsigned int mant = h & 0x03ffu;
    unsigned int raw;
    if (exp == 0u) {
        if (mant == 0u) {
            raw = sign;
        } else {
            unsigned int shift = 0u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x03ffu;
            raw = sign | ((113u - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        raw = sign | 0x7f800000u | (mant << 13);
    } else {
        raw = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    return __uint_as_float(raw);
}

static __device__ unsigned int qtype_load_u16(const unsigned char *bytes)
{
    return (unsigned int)bytes[0] | ((unsigned int)bytes[1] << 8);
}

static __device__ unsigned int qtype_load_u32(const unsigned char *bytes)
{
    return (unsigned int)bytes[0] |
           ((unsigned int)bytes[1] << 8) |
           ((unsigned int)bytes[2] << 16) |
           ((unsigned int)bytes[3] << 24);
}

static __device__ float bf16_bits_to_float(unsigned int bits)
{
    return __uint_as_float(bits << 16);
}

static __device__ float float_to_bf16_rne(float value)
{
    unsigned int bits = __float_as_uint(value);
    unsigned int upper = bits >> 16;
    unsigned int lower = bits & 0xffffu;
    if ((bits & 0x7f800000u) == 0x7f800000u &&
        (bits & 0x007fffffu) != 0u)
        return __uint_as_float((upper | 0x0040u) << 16);
    if (lower > 0x8000u || (lower == 0x8000u && (upper & 1u))) upper++;
    return __uint_as_float(upper << 16);
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

static __device__ float e8m0_bits_to_float(unsigned int bits)
{
    if (bits == 0xffu) return __uint_as_float(0x7fc00000u);
    return __uint_as_float(bits == 0u ? 0x00400000u : bits << 23);
}

static __device__ float mxfp4_code_to_float(unsigned int code)
{
    float magnitude;
    switch (code & 7u) {
    case 0u: magnitude = 0.0f; break;
    case 1u: magnitude = 1.0f; break;
    case 2u: magnitude = 2.0f; break;
    case 3u: magnitude = 3.0f; break;
    case 4u: magnitude = 4.0f; break;
    case 5u: magnitude = 6.0f; break;
    case 6u: magnitude = 8.0f; break;
    default: magnitude = 12.0f; break;
    }
    return code & 8u ? -magnitude : magnitude;
}
/* Pinned compatible IQ2_XXS magnitude-grid identities. */
static __device__ __constant__ unsigned short iq2_xxs_grid[256] = {
    0,     2,     5,     8,     10,    17,    20,    32,    34,    40,    42,
    65,    68,    80,    88,    97,    100,   128,   130,   138,   162,   257,
    260,   272,   277,   320,   388,   408,   512,   514,   546,   642,   1025,
    1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,
    1188,  1280,  1282,  1288,  1312,  1350,  1385,  1408,  1425,  1545,  1552,
    1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,  2116,  2128,
    2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,
    4192,  4228,  4240,  4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,
    4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,  5474,  5632,  5654,
    6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,
    8202,  8260,  8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,
    9232,  9280,  9302,  9472,  9537,  9572,  9872,  10248, 10272, 10388, 10820,
    16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513,
    16516, 16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982,
    17000, 17408, 17416, 17440, 17536, 17561, 17682, 17700, 17920, 18433, 18436,
    18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650,
    21770, 22017, 22100, 22528, 22545, 22553, 22628, 22848, 23048, 24580, 24592,
    24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874, 25988,
    26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793,
    33796, 33808, 33813, 33856, 33888, 34048, 34118, 34196, 34313, 34368, 34400,
    34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142, 37248, 37445,
    37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472,
    42008, 43088, 43268
};

static __device__ unsigned int iq2_xxs_signs(unsigned int low)
{
    unsigned int parity = 0u;
    unsigned int value = low;
    while (value) { parity ^= value & 1u; value >>= 1u; }
    return low | (parity << 7u);
}

static __device__ float qtype_value(const unsigned char *encoded,
                                         unsigned long long index,
                                         unsigned int qtype)
{
    if (qtype == YVEX_GGUF_QTYPE_F32) {
        return __uint_as_float(qtype_load_u32(encoded + index * 4ull));
    }
    if (qtype == YVEX_GGUF_QTYPE_F16) {
        return f16_bits_to_float(
            qtype_load_u16(encoded + index * 2ull));
    }
    if (qtype == YVEX_GGUF_QTYPE_BF16) {
        return bf16_bits_to_float(
            qtype_load_u16(encoded + index * 2ull));
    }
    if (qtype == YVEX_GGUF_QTYPE_I32) {
        unsigned int raw = qtype_load_u32(encoded + index * 4ull);
        int value = raw <= 0x7fffffffu
            ? (int)raw : -1 - (int)(0xffffffffu - raw);
        return (float)value;
    }
    if (qtype == YVEX_GGUF_QTYPE_Q8_0) {
        unsigned long long block = index / 32ull;
        unsigned int lane = (unsigned int)(index % 32ull);
        const unsigned char *bytes = encoded + block * 34ull;
        float scale = f16_bits_to_float(qtype_load_u16(bytes));
        int quantized = bytes[2u + lane] <= 127u
            ? (int)bytes[2u + lane] : (int)bytes[2u + lane] - 256;
        return scale * (float)quantized;
    }
    if (qtype == YVEX_GGUF_QTYPE_MXFP4) {
        unsigned long long block = index / 32ull;
        unsigned int lane = (unsigned int)(index % 32ull);
        const unsigned char *bytes = encoded + block * 17ull;
        unsigned int packed = bytes[1u + (lane & 15u)];
        unsigned int code = lane < 16u ? packed & 15u : packed >> 4;
        return mxfp4_code_to_float(code) *
               e8m0_bits_to_float(bytes[0]) * 0.5f;
    }
    if (qtype == YVEX_GGUF_QTYPE_Q2_K) {
        unsigned long long block = index / 256ull;
        unsigned int lane = (unsigned int)(index % 256ull);
        const unsigned char *bytes = encoded + block * 84ull;
        unsigned int subblock = lane / 16u;
        unsigned int half = lane / 128u;
        unsigned int local_subblock = subblock & 7u;
        unsigned int pair = local_subblock & 1u;
        unsigned int group = local_subblock / 2u;
        unsigned int packed = bytes[16u + half * 32u + pair * 16u +
                                    (lane & 15u)];
        unsigned int code = (packed >> (group * 2u)) & 3u;
        unsigned int scale_byte = bytes[subblock];
        float scale = f16_bits_to_float(
            qtype_load_u16(bytes + 80u));
        float minimum = f16_bits_to_float(
            qtype_load_u16(bytes + 82u));
        return scale * (float)(scale_byte & 15u) * (float)code -
               minimum * (float)(scale_byte >> 4);
    }
    if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        unsigned long long block = index / 256ull;
        unsigned int lane = (unsigned int)(index % 256ull);
        const unsigned char *bytes = encoded + block * 66ull;
        unsigned int group = lane / 32u;
        unsigned int subgroup = (lane & 31u) / 8u;
        unsigned int local = lane & 7u;
        unsigned int grids = qtype_load_u32(bytes + 2u + group * 8u);
        unsigned int sign_scale = qtype_load_u32(bytes + 6u + group * 8u);
        unsigned int grid = (grids >> (8u * subgroup)) & 255u;
        unsigned int signs = iq2_xxs_signs((sign_scale >> (7u * subgroup)) & 127u);
        unsigned int digit = (iq2_xxs_grid[grid] >> (2u * local)) & 3u;
        float level = digit == 0u ? 8.0f : digit == 1u ? 25.0f : 43.0f;
        float scale = f16_bits_to_float(qtype_load_u16(bytes)) *
                      (0.5f + (float)(sign_scale >> 28u)) * 0.25f;
        return scale * level * ((signs & (1u << local)) ? -1.0f : 1.0f);
    }
    return __uint_as_float(0x7fc00000u);
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
    unsigned long long hidden_size, float epsilon)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;
    unsigned long long i;
    float sum = 0.0f;
    float inv_rms;
    if (!input || !weight || !out || hidden_size == 0ull || epsilon <= 0.0f) {
        return;
    }
    for (i = (unsigned long long)tid; i < hidden_size; i += (unsigned long long)stride) {
        float v = input[i];
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
        out[i] = input[i] * inv_rms * weight[i];
    }
}

extern "C" __global__ void yvex_rms_norm_f32_weight_f16(
    const float *input, const unsigned short *weight, float *out,
    unsigned long long hidden_size, float epsilon)
{
    extern __shared__ float scratch[];
    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;
    unsigned long long i;
    float sum = 0.0f;
    float inv_rms;
    if (!input || !weight || !out || hidden_size == 0ull || epsilon <= 0.0f) {
        return;
    }
    for (i = (unsigned long long)tid; i < hidden_size; i += (unsigned long long)stride) {
        float v = input[i];
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
        out[i] = input[i] * inv_rms * f16_bits_to_float((unsigned int)weight[i]);
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

static __device__ float qtype_warp_dot(const unsigned char *row, const float *vector,
                                       unsigned long long width, unsigned int qtype,
                                       int *status)
{
    unsigned int lane = threadIdx.x & 31u;
    float sum = 0.0f;
    if (!row || !vector || !width) {
        if (!lane) atomicCAS(status, 0, 2);
    } else for (unsigned long long i = lane; i < width; i += 32ull) {
        float weight = qtype_value(row, i, qtype);
        float value = float_to_bf16_rne(vector[i]);
        if (!isfinite(weight) || !isfinite(value)) atomicCAS(status, 0, 1);
        else sum = fmaf(weight, value, sum);
    }
    for (unsigned int offset = 16u; offset; offset >>= 1u)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return sum;
}
#define YVEX_CUDA_Q8_K_BLOCK 256ull
#define YVEX_CUDA_Q8_K_BYTES 292ull

extern "C" __global__ void yvex_q8_quantize(
    unsigned char *encoded, const float *values, unsigned long long width,
    unsigned long long rows, int *status)
{
    __shared__ float absolute[256];
    __shared__ float signed_value[256];
    __shared__ float inverse_scale;
    unsigned long long blocks = width / YVEX_CUDA_Q8_K_BLOCK;
    unsigned long long task = blockIdx.x;
    unsigned long long row = blocks ? task / blocks : rows;
    unsigned long long block_index = blocks ? task % blocks : 0ull;
    unsigned int thread = threadIdx.x;
    unsigned char *block;
    float value;
    if (!status || *status || !encoded || !values || !blocks || row >= rows ||
        thread >= 256u) return;
    block = encoded + (row * blocks + block_index) * YVEX_CUDA_Q8_K_BYTES;
    value = values[row * width + block_index * YVEX_CUDA_Q8_K_BLOCK + thread];
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

static __device__ int q8_k_sum(const unsigned char *block, unsigned int index)
{
    unsigned int raw = qtype_load_u16(block + 260u + index * 2u);
    return raw <= 32767u ? (int)raw : (int)raw - 65536;
}

static __device__ int q2_k_dot16(const unsigned char *weight,
                                 const unsigned char *activation, unsigned int shift)
{
    int sum = 0;
#pragma unroll
    for (unsigned int i = 0u; i < 16u; i += 4u) {
        int packed = (int)((qtype_load_u32(weight + i) >> shift) & 0x03030303u);
        sum = __dp4a(packed, (int)qtype_load_u32(activation + i), sum);
    }
    return sum;
}

static __device__ int iq2_xxs_i8x4(unsigned short grid, unsigned int start,
                                   unsigned int signs)
{
    unsigned int packed = 0u;
#pragma unroll
    for (unsigned int i = 0u; i < 4u; ++i) {
        unsigned int code = (grid >> (2u * (start + i))) & 3u;
        int level = code == 0u ? 8 : code == 1u ? 25 : 43;
        if (signs & (1u << (start + i))) level = -level;
        packed |= ((unsigned int)level & 255u) << (8u * i);
    }
    return (int)packed;
}

static __device__ float iq2_xxs_q8_k_dot(const unsigned char *weight,
                                         const unsigned char *activation)
{
    float weight_scale = f16_bits_to_float(qtype_load_u16(weight));
    float activation_scale = __uint_as_float(qtype_load_u32(activation));
    int total = 0;
#pragma unroll
    for (unsigned int group = 0u; group < 8u; ++group) {
        unsigned int grids = qtype_load_u32(weight + 2u + group * 8u);
        unsigned int sign_scale = qtype_load_u32(weight + 6u + group * 8u);
        int group_sum = 0;
#pragma unroll
        for (unsigned int subgroup = 0u; subgroup < 4u; ++subgroup) {
            unsigned int grid_index = (grids >> (8u * subgroup)) & 255u;
            unsigned int signs = iq2_xxs_signs(
                (sign_scale >> (7u * subgroup)) & 127u);
            unsigned short grid = iq2_xxs_grid[grid_index];
            const unsigned char *q8 = activation + 4u + group * 32u + subgroup * 8u;
            group_sum = __dp4a(iq2_xxs_i8x4(grid, 0u, signs),
                                (int)qtype_load_u32(q8), group_sum);
            group_sum = __dp4a(iq2_xxs_i8x4(grid, 4u, signs),
                                (int)qtype_load_u32(q8 + 4u), group_sum);
        }
        total += group_sum * (int)(2u * (sign_scale >> 28u) + 1u);
    }
    return 0.125f * weight_scale * activation_scale * (float)total;
}

static __device__ float q2_k_q8_k_dot(const unsigned char *weight,
                                      const unsigned char *activation)
{
    const unsigned char *scales = weight;
    const unsigned char *quantized = weight + 16u;
    const unsigned char *q8 = activation + 4u;
    float activation_scale = __uint_as_float(qtype_load_u32(activation));
    int minimum_sum = 0, dot = 0, scale_index = 0;
    for (unsigned int i = 0u; i < 16u; ++i)
        minimum_sum += q8_k_sum(activation, i) * (int)(scales[i] >> 4u);
#pragma unroll
    for (unsigned int half = 0u; half < 2u; ++half) {
        unsigned int shift = 0u;
#pragma unroll
        for (unsigned int group = 0u; group < 4u; ++group) {
            dot += (int)(scales[scale_index++] & 15u) * q2_k_dot16(quantized, q8, shift);
            dot += (int)(scales[scale_index++] & 15u) *
                   q2_k_dot16(quantized + 16u, q8 + 16u, shift);
            shift += 2u;
            q8 += 32u;
        }
        quantized += 32u;
    }
    return activation_scale * f16_bits_to_float(qtype_load_u16(weight + 80u)) *
               (float)dot -
           activation_scale * f16_bits_to_float(qtype_load_u16(weight + 82u)) *
               (float)minimum_sum;
}

static __device__ float q8_0_q8_k_dot(const unsigned char *weight,
                                      const unsigned char *activation)
{
    float activation_scale = __uint_as_float(qtype_load_u32(activation));
    float total = 0.0f;
#pragma unroll
    for (unsigned int block = 0u; block < 8u; ++block) {
        const unsigned char *q8_0 = weight + block * 34u;
        int dot = 0;
#pragma unroll
        for (unsigned int i = 0u; i < 32u; i += 4u)
            dot = __dp4a((int)qtype_load_u32(q8_0 + 2u + i),
                         (int)qtype_load_u32(activation + 4u + block * 32u + i), dot);
        total = fmaf(f16_bits_to_float(qtype_load_u16(q8_0)) * activation_scale,
                     (float)dot, total);
    }
    return total;
}

static __device__ float qtype_q8_k_dot(const unsigned char *weight,
                                       const unsigned char *activation,
                                       unsigned int qtype)
{
    if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS) return iq2_xxs_q8_k_dot(weight, activation);
    if (qtype == YVEX_GGUF_QTYPE_Q2_K) return q2_k_q8_k_dot(weight, activation);
    if (qtype == YVEX_GGUF_QTYPE_Q8_0) return q8_0_q8_k_dot(weight, activation);
    return __uint_as_float(0x7fc00000u);
}

static __device__ float q8_warp_dot(const unsigned char *weight,
                                    const unsigned char *activation,
                                    unsigned long long blocks,
                                    unsigned long long weight_block,
                                    unsigned int qtype)
{
    unsigned int lane = threadIdx.x & 31u;
    float sum = 0.0f;
    for (unsigned long long block = lane; block < blocks; block += 32ull)
        sum += qtype_q8_k_dot(weight + block * weight_block,
                             activation + block * YVEX_CUDA_Q8_K_BYTES, qtype);
    for (unsigned int offset = 16u; offset; offset >>= 1u)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return sum;
}

extern "C" __global__ void yvex_qtype_matvec(
    const unsigned char *encoded,
    unsigned long long row_bytes,
    unsigned long long row_width,
    unsigned long long start_row,
    unsigned long long row_count,
    unsigned int qtype,
    const void *vector,
    int q8_input,
    float *out,
    int output_bf16,
    int *status)
{
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5u;
    unsigned long long row = (unsigned long long)blockIdx.x * 8ull + warp;
    const unsigned char *row_data;
    float sum;
    if (!status) return;
    if (*status != 0 || row >= row_count) return;
    if (!encoded || !vector || !out || !row_bytes || !row_width) {
        atomicCAS(status, 0, 2);
        return;
    }
    row_data = encoded + (start_row + row) * row_bytes;
    if (q8_input) {
        unsigned long long blocks = row_width / YVEX_CUDA_Q8_K_BLOCK;
        if (!blocks || row_bytes % blocks) {
            if (!lane) atomicCAS(status, 0, 2);
            return;
        }
        sum = q8_warp_dot(row_data, (const unsigned char *)vector, blocks,
                          row_bytes / blocks, qtype);
    } else
        sum = qtype_warp_dot(row_data, (const float *)vector, row_width, qtype, status);
    if (lane == 0u) {
        if (!isfinite(sum)) atomicCAS(status, 0, 1);
        else {
            float value = sum;
            if (!isfinite(value)) atomicCAS(status, 0, 1);
            else out[row] = output_bf16 ? float_to_bf16_rne(value) : value;
        }
    }
}

extern "C" __global__ void yvex_deepseek_decode(
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

extern "C" __global__ void yvex_deepseek_weighted_norm(
    float *values, unsigned long long count, const unsigned char *weight,
    unsigned int weight_qtype, double epsilon, int *status)
{
    unsigned int lane = threadIdx.x;
    double mean = 0.0;
    if (!status) return;
    if (*status != 0 || blockIdx.x != 0u) return;
    if (!values || !weight || !count || epsilon <= 0.0) {
        if (lane == 0u) atomicCAS(status, 0, 2);
        return;
    }
    if (lane != 0u) return;
    for (unsigned long long i = 0ull; i < count; ++i) {
        double value = (double)values[i];
        if (!isfinite(value)) {
            atomicCAS(status, 0, 1);
            return;
        }
        mean = __dadd_rn(mean, __dmul_rn(value, value));
    }
    mean = __ddiv_rn(mean, (double)count);
    double inverse = __ddiv_rn(1.0, sqrt(__dadd_rn(mean, epsilon)));
    if (!isfinite(inverse)) {
        atomicCAS(status, 0, 1);
        return;
    }
    for (unsigned long long i = 0ull; i < count; ++i) {
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

extern "C" __global__ void yvex_deepseek_unit_norm(
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

static __device__ double deepseek_yarn_frequency(
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

extern "C" __global__ void yvex_deepseek_rope(
    float *values, unsigned long long vector_count,
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
    if (!values || !vector_count || !rope_dims || rope_dims > vector_width ||
        (rope_dims & 1ull) || theta <= 1ull || !scaling_factor ||
        (original_context && (!beta_slow || beta_fast <= beta_slow))) {
        atomicCAS(status, 0, 2);
        return;
    }
    pairs_per_vector = rope_dims / 2ull;
    if (vector_count > ~0ull / pairs_per_vector) {
        atomicCAS(status, 0, 2);
        return;
    }
    total = vector_count * pairs_per_vector;
    if (pair >= total) return;
    unsigned long long vector_index = pair / pairs_per_vector;
    unsigned long long local_pair = pair % pairs_per_vector;
    unsigned long long start = vector_width - rope_dims;
    unsigned long long offset = vector_index * vector_width + start +
                                local_pair * 2ull;
    double frequency = deepseek_yarn_frequency(
        local_pair, rope_dims, theta, scaling_factor, original_context,
        beta_fast, beta_slow);
    double angle = (double)token_position * frequency;
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

static __device__ float deepseek_fp8_decode(unsigned int code)
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

static __device__ unsigned int deepseek_fp8_encode(float value)
{
    float magnitude = fabsf(value);
    float best_error = INFINITY;
    unsigned int best = 0u;
    int negative = signbit(value);
    if (!isfinite(value)) return negative ? 0xffu : 0x7fu;
    if (magnitude > 448.0f) magnitude = 448.0f;
    for (unsigned int code = 0u; code < 0x7fu; ++code) {
        float error = fabsf(deepseek_fp8_decode(code) - magnitude);
        if (error < best_error ||
            (error == best_error && !(code & 1u) && (best & 1u))) {
            best_error = error;
            best = code;
        }
    }
    return negative ? best | 0x80u : best;
}

static __device__ float deepseek_fp4_decode(unsigned int code)
{
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                            2.0f, 3.0f, 4.0f, 6.0f};
    float value = table[code & 7u];
    return (code & 8u) ? -value : value;
}

static __device__ unsigned int deepseek_fp4_encode(float value)
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

static __device__ unsigned int deepseek_e8m0_encode(float value)
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

static __device__ float deepseek_power_two_ceil(float value)
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
extern "C" __global__ void yvex_deepseek_activation(
    float *values, unsigned long long vector_count,
    unsigned long long vector_width, unsigned long long block_width,
    unsigned int quantization, int hadamard, int *status)
{
    unsigned long long vector_index = (unsigned long long)blockIdx.x;
    if (!status) return;
    if (*status != 0 || threadIdx.x != 0u || vector_index >= vector_count)
        return;
    if (!values || !vector_count || !vector_width || !block_width ||
        vector_width % block_width || (quantization != 1u && quantization != 2u)) {
        atomicCAS(status, 0, 2);
        return;
    }
    float *vector = values + vector_index * vector_width;
    if (hadamard) {
        if ((vector_width & (vector_width - 1ull)) != 0ull ||
            vector_width > 1024ull) {
            atomicCAS(status, 0, 2);
            return;
        }
        for (unsigned long long step = 1ull; step < vector_width; step *= 2ull)
            for (unsigned long long block = 0ull; block < vector_width;
                 block += step * 2ull)
                for (unsigned long long lane = 0ull; lane < step; ++lane) {
                    float left = vector[block + lane];
                    float right = vector[block + lane + step];
                    vector[block + lane] = left + right;
                    vector[block + lane + step] = left - right;
                }
        float scale = rsqrtf((float)vector_width);
        for (unsigned long long i = 0ull; i < vector_width; ++i)
            vector[i] *= scale;
    }
    for (unsigned long long offset = 0ull; offset < vector_width;
         offset += block_width) {
        float amax = quantization == 1u ? 1.0e-4f : 0.0f;
        for (unsigned long long i = 0ull; i < block_width; ++i) {
            float value = vector[offset + i];
            if (!isfinite(value)) {
                atomicCAS(status, 0, 1);
                return;
            }
            float magnitude = fabsf(value);
            if (magnitude > amax) amax = magnitude;
        }
        float minimum = 6.0f * ldexpf(1.0f, -126);
        if (quantization == 2u && amax < minimum) amax = minimum;
        float scale = deepseek_power_two_ceil(
            amax / (quantization == 1u ? 448.0f : 6.0f));
        unsigned int scale_code = deepseek_e8m0_encode(scale);
        scale = e8m0_bits_to_float(scale_code);
        if (!isfinite(scale) || scale <= 0.0f) {
            atomicCAS(status, 0, 1);
            return;
        }
        for (unsigned long long i = 0ull; i < block_width; ++i) {
            float normalized = vector[offset + i] / scale;
            if (quantization == 1u) {
                if (normalized > 448.0f) normalized = 448.0f;
                if (normalized < -448.0f) normalized = -448.0f;
                vector[offset + i] = float_to_bf16_rne(
                    deepseek_fp8_decode(deepseek_fp8_encode(normalized)) *
                    scale);
            } else if (quantization == 2u) {
                vector[offset + i] = float_to_bf16_rne(
                    deepseek_fp4_decode(deepseek_fp4_encode(normalized)) *
                    scale);
            } else {
                atomicCAS(status, 0, 2);
                return;
            }
        }
    }
}

extern "C" __global__ void yvex_deepseek_mhc_pre(
    float *residual, const float *linear_mix, const float *scale,
    const float *base, unsigned long long streams,
    unsigned long long stream_width, unsigned long long mixing_rows,
    unsigned long long sinkhorn_iterations, double rms_epsilon,
    double mhc_epsilon, double post_multiplier, float *collapsed,
    float *post, float *combination, int *status)
{
    if (!status || blockIdx.x != 0u || threadIdx.x != 0u || *status != 0)
        return;
    if (!residual || !linear_mix || !scale || !base || !collapsed || !post ||
        !combination || !streams || !stream_width || !sinkhorn_iterations ||
        mixing_rows != (streams + 2ull) * streams || rms_epsilon <= 0.0 ||
        mhc_epsilon <= 0.0 || post_multiplier <= 0.0) {
        atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long expanded = streams * stream_width;
    double squares = 0.0;
    for (unsigned long long lane = 0ull; lane < expanded; ++lane) {
        float value = float_to_bf16_rne(residual[lane]);
        residual[lane] = value;
        if (!isfinite(value)) {
            atomicCAS(status, 0, 1);
            return;
        }
        squares += (double)value * (double)value;
    }
    double inverse = 1.0 / sqrt(squares / (double)expanded + rms_epsilon);
    if (!isfinite(inverse)) {
        atomicCAS(status, 0, 1);
        return;
    }
    for (unsigned long long lane = 0ull; lane < stream_width; ++lane)
        collapsed[lane] = 0.0f;
    for (unsigned long long stream = 0ull; stream < streams; ++stream) {
        double pre_arg = (double)linear_mix[stream] * inverse * (double)scale[0] +
                         (double)base[stream];
        double pre = pre_arg >= 0.0 ? 1.0 / (1.0 + exp(-pre_arg))
                                    : exp(pre_arg) / (1.0 + exp(pre_arg));
        unsigned long long post_index = streams + stream;
        double post_arg = (double)linear_mix[post_index] * inverse * (double)scale[1] +
                          (double)base[post_index];
        double post_sigmoid = post_arg >= 0.0 ? 1.0 / (1.0 + exp(-post_arg))
                                              : exp(post_arg) / (1.0 + exp(post_arg));
        post[stream] = (float)(post_multiplier * post_sigmoid);
        for (unsigned long long target = 0ull; target < streams; ++target) {
            unsigned long long index = 2ull * streams + stream * streams + target;
            combination[stream * streams + target] =
                (float)((double)linear_mix[index] * inverse * (double)scale[2] +
                        (double)base[index]);
        }
        for (unsigned long long lane = 0ull; lane < stream_width; ++lane)
            collapsed[lane] += (float)((pre + mhc_epsilon) *
                (double)residual[stream * stream_width + lane]);
    }
    for (unsigned long long row = 0ull; row < streams; ++row) {
        double maximum = -INFINITY;
        double total = 0.0;
        for (unsigned long long column = 0ull; column < streams; ++column)
            maximum = maximum > (double)combination[row * streams + column]
                ? maximum : (double)combination[row * streams + column];
        for (unsigned long long column = 0ull; column < streams; ++column) {
            double value = exp((double)combination[row * streams + column] - maximum);
            combination[row * streams + column] = (float)value;
            total += value;
        }
        if (!isfinite(total) || total <= 0.0) {
            atomicCAS(status, 0, 1);
            return;
        }
        for (unsigned long long column = 0ull; column < streams; ++column)
            combination[row * streams + column] =
                (float)((double)combination[row * streams + column] / total + mhc_epsilon);
    }
    for (unsigned long long iteration = 0ull; iteration < sinkhorn_iterations; ++iteration) {
        if (iteration != 0ull) {
            for (unsigned long long row = 0ull; row < streams; ++row) {
                double total = 0.0;
                for (unsigned long long column = 0ull; column < streams; ++column)
                    total += combination[row * streams + column];
                for (unsigned long long column = 0ull; column < streams; ++column)
                    combination[row * streams + column] =
                        (float)((double)combination[row * streams + column] /
                                (total + mhc_epsilon));
            }
        }
        for (unsigned long long column = 0ull; column < streams; ++column) {
            double total = 0.0;
            for (unsigned long long row = 0ull; row < streams; ++row)
                total += combination[row * streams + column];
            for (unsigned long long row = 0ull; row < streams; ++row)
                combination[row * streams + column] =
                    (float)((double)combination[row * streams + column] /
                            (total + mhc_epsilon));
        }
    }
    for (unsigned long long lane = 0ull; lane < stream_width; ++lane) {
        if (!isfinite(collapsed[lane]) || !isfinite(post[lane % streams])) {
            atomicCAS(status, 0, 1);
            return;
        }
        collapsed[lane] = float_to_bf16_rne(collapsed[lane]);
    }
}

extern "C" __global__ void yvex_deepseek_mhc_post(
    const float *core, const float *residual, const float *post,
    const float *combination, unsigned long long streams,
    unsigned long long stream_width, float *output, int *status)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long expanded = streams * stream_width;
    if (!status || *status != 0 || index >= expanded) return;
    if (!core || !residual || !post || !combination || !output || !streams ||
        !stream_width) {
        atomicCAS(status, 0, 2);
        return;
    }
    unsigned long long target = index / stream_width;
    unsigned long long lane = index % stream_width;
    double value = (double)post[target] * (double)core[lane];
    for (unsigned long long source = 0ull; source < streams; ++source)
        value += (double)combination[source * streams + target] *
                 (double)residual[source * stream_width + lane];
    float published = (float)value;
    if (!isfinite(published)) atomicCAS(status, 0, 1);
    else output[index] = float_to_bf16_rne(published);
}
/*
 * Collapse the final mHC streams and apply transformer-owned RMSNorm.
 *
 * Device-resident expanded rows and exact decoded final weights.
 */
extern "C" __global__ void yvex_deepseek_transformer_final(
    const float *expanded, const float *function, const float *base,
    const float *scale, const float *norm, unsigned long long token_count,
    unsigned long long streams, unsigned long long width, double epsilon,
    double mhc_epsilon, float *output, int *status)
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
extern "C" __global__ void yvex_deepseek_rolling(
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
extern "C" __global__ void yvex_deepseek_topk(
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
    unsigned long long total;
    if (!status) return;
    if (*status != 0 || blockIdx.x != 0u || threadIdx.x != 0u) return;
    if (!index_query || !index_weights || !selected || !selected_positions ||
        !selected_count || !valid_count || !scores || !valid_indexes || !heads ||
        !head_dim || !ratio || !k || history_count > ~0ull - current_count ||
        (history_count && (!history_indexer || !history_positions ||
                           history_stride < head_dim)) ||
        (current_count && (!current_indexer || !current_positions ||
                           current_stride < head_dim))) {
        atomicCAS(status, 0, 2);
        return;
    }
    total = history_count + current_count;
    unsigned long long valid = 0ull;
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
        if (!row || position > query_position ||
            position > ~0ull - ratio + 1ull ||
            position + ratio - 1ull > query_position) continue;
        for (unsigned long long prior = 0ull; prior < valid; ++prior) {
            unsigned long long prior_candidate = valid_indexes[prior];
            unsigned long long prior_position = prior_candidate < history_count
                ? history_positions[prior_candidate]
                : current_positions[prior_candidate - history_count];
            if (prior_position == position) {
                atomicCAS(status, 0, 1);
                return;
            }
        }
        double score = 0.0;
        for (unsigned long long head = 0ull; head < heads; ++head) {
            double dot = 0.0;
            const float *query = index_query + head * head_dim;
            for (unsigned long long lane = 0ull; lane < head_dim; ++lane) {
                double term = __dmul_rn((double)query[lane], (double)row[lane]);
                dot = __dadd_rn(dot, term);
            }
            if (dot < 0.0) dot = 0.0;
            score = __dadd_rn(
                score, __dmul_rn(dot, (double)index_weights[head]));
        }
        score = __dmul_rn(score, 1.0 / sqrt((double)head_dim));
        score = __dmul_rn(score, 1.0 / sqrt((double)heads));
        if (!isfinite(score)) {
            atomicCAS(status, 0, 1);
            return;
        }
        scores[valid] = (float)score;
        valid_indexes[valid] = candidate;
        valid++;
    }
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

extern "C" __global__ void yvex_deepseek_reduce(
    const float *query,
    const float *history_local,
    const unsigned long long *history_local_positions,
    unsigned long long history_local_count,
    unsigned long long history_local_stride,
    const float *current_kv,
    unsigned long long current_kv_stride,
    const float *history_compressed,
    const unsigned long long *history_compressed_positions,
    unsigned long long history_compressed_count,
    unsigned long long history_compressed_stride,
    const float *current_compressed,
    const unsigned long long *current_compressed_positions,
    unsigned long long current_compressed_count,
    unsigned long long current_compressed_stride,
    const unsigned long long *selected,
    const unsigned long long *selected_count_ptr,
    const float *sinks,
    unsigned long long query_heads,
    unsigned long long head_dim,
    unsigned long long sliding_window,
    unsigned long long ratio,
    unsigned int attention_class,
    unsigned long long token_position,
    float *out,
    int *status)
{
    unsigned long long head = (unsigned long long)blockIdx.x;
    unsigned int thread = threadIdx.x;
    if (!status) return;
    if (*status != 0 || head >= query_heads) return;
    if (thread != 0u) return;
    if (!query || !current_kv || !sinks || !out || !query_heads || !head_dim ||
        !sliding_window || token_position == ~0ull || attention_class > 2u ||
        history_local_count == ~0ull ||
        history_compressed_count > ~0ull - current_compressed_count ||
        current_kv_stride < head_dim ||
        (history_local_count && (!history_local || !history_local_positions ||
                                 history_local_stride < head_dim)) ||
        (history_compressed_count &&
         (!history_compressed || !history_compressed_positions ||
          history_compressed_stride < head_dim)) ||
        (current_compressed_count &&
         (!current_compressed || !current_compressed_positions ||
          current_compressed_stride < head_dim)) ||
        (attention_class == 1u && (!selected || !selected_count_ptr)) ||
        (attention_class == 1u && ratio != 4ull) ||
        (attention_class == 2u && ratio != 128ull) ||
        (attention_class == 0u && ratio != 0ull)) {
        atomicCAS(status, 0, 2);
        return;
    }
    const float *q = query + head * head_dim;
    double maximum = (double)sinks[head];
    double scale = 1.0 / sqrt((double)head_dim);
    unsigned long long local_total = history_local_count + 1ull;
    unsigned long long selected_count = selected_count_ptr
        ? *selected_count_ptr : 0ull;
    unsigned long long compressed_total = attention_class == 2u
        ? history_compressed_count + current_compressed_count
        : selected_count;
    for (unsigned long long pass = 0ull; pass < 2ull; ++pass) {
        unsigned long long count = pass == 0ull ? local_total : compressed_total;
        for (unsigned long long candidate = 0ull; candidate < count; ++candidate) {
            const float *row = NULL;
            unsigned long long position = ~0ull;
            if (pass == 0ull) {
                if (candidate < history_local_count) {
                    row = history_local + candidate * history_local_stride;
                    position = history_local_positions[candidate];
                } else {
                    row = current_kv;
                    position = token_position;
                }
                unsigned long long first = token_position + 1ull > sliding_window
                    ? token_position + 1ull - sliding_window : 0ull;
                if (position < first || position > token_position) continue;
            } else {
                unsigned long long index = attention_class == 2u
                    ? candidate : selected[candidate];
                if (index < history_compressed_count) {
                    row = history_compressed + index * history_compressed_stride;
                    position = history_compressed_positions[index];
                } else {
                    unsigned long long local = index - history_compressed_count;
                    row = current_compressed + local * current_compressed_stride;
                    position = current_compressed_positions[local];
                }
                if (!row || position > token_position ||
                    position > ~0ull - ratio + 1ull ||
                    position + ratio - 1ull > token_position) continue;
            }
            double dot = 0.0;
            for (unsigned long long lane = 0ull; lane < head_dim; ++lane) {
                double term = __dmul_rn((double)q[lane], (double)row[lane]);
                dot = __dadd_rn(dot, term);
            }
            double score = __dmul_rn(dot, scale);
            if (score > maximum) maximum = score;
        }
    }
    double denominator = exp(__dadd_rn((double)sinks[head], -maximum));
    for (unsigned long long lane = 0ull; lane < head_dim; ++lane)
        out[head * head_dim + lane] = 0.0f;
    for (unsigned long long pass = 0ull; pass < 2ull; ++pass) {
        unsigned long long count = pass == 0ull ? local_total : compressed_total;
        for (unsigned long long candidate = 0ull; candidate < count; ++candidate) {
            const float *row = NULL;
            unsigned long long position = ~0ull;
            if (pass == 0ull) {
                if (candidate < history_local_count) {
                    row = history_local + candidate * history_local_stride;
                    position = history_local_positions[candidate];
                } else {
                    row = current_kv;
                    position = token_position;
                }
                unsigned long long first = token_position + 1ull > sliding_window
                    ? token_position + 1ull - sliding_window : 0ull;
                if (position < first || position > token_position) continue;
            } else {
                unsigned long long index = attention_class == 2u
                    ? candidate : selected[candidate];
                if (index < history_compressed_count) {
                    row = history_compressed + index * history_compressed_stride;
                    position = history_compressed_positions[index];
                } else {
                    unsigned long long local = index - history_compressed_count;
                    row = current_compressed + local * current_compressed_stride;
                    position = current_compressed_positions[local];
                }
                if (!row || position > token_position ||
                    position > ~0ull - ratio + 1ull ||
                    position + ratio - 1ull > token_position) continue;
            }
            double dot = 0.0;
            for (unsigned long long lane = 0ull; lane < head_dim; ++lane) {
                double term = __dmul_rn((double)q[lane], (double)row[lane]);
                dot = __dadd_rn(dot, term);
            }
            double probability = exp(__dadd_rn(__dmul_rn(dot, scale), -maximum));
            denominator = __dadd_rn(denominator, probability);
            for (unsigned long long lane = 0ull; lane < head_dim; ++lane) {
                float term = (float)__dmul_rn(probability, (double)row[lane]);
                unsigned long long offset = head * head_dim + lane;
                out[offset] = __fadd_rn(out[offset], term);
            }
        }
    }
    if (!isfinite(denominator) || denominator <= 0.0) {
        atomicCAS(status, 0, 1);
        return;
    }
    for (unsigned long long lane = 0ull; lane < head_dim; ++lane) {
        float published = (float)__ddiv_rn((double)out[head * head_dim + lane],
                                           denominator);
        if (!isfinite(published)) atomicCAS(status, 0, 1);
        else out[head * head_dim + lane] = float_to_bf16_rne(published);
    }
}
