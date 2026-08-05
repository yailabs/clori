/*
 * Share CUDA qtype decoding and dot primitives across independently compiled kernel families.
 *
 * Each CUDA module receives its own internal device definitions; no runtime symbol or model
 * policy crosses this toolchain-only interface.
 */
#ifndef SRC_BACKEND_CUDA_KERNEL_PRIMITIVES_H_INCLUDED
#define SRC_BACKEND_CUDA_KERNEL_PRIMITIVES_H_INCLUDED
#include <yvex/qtype.h>

enum {
    YVEX_CUDA_SAMPLING_TOP_K_COUNT = 0,
    YVEX_CUDA_SAMPLING_MIN_P_COUNT,
    YVEX_CUDA_SAMPLING_TYPICAL_COUNT,
    YVEX_CUDA_SAMPLING_TOP_P_COUNT,
    YVEX_CUDA_SAMPLING_COUNT_FIELDS
};
enum {
    YVEX_CUDA_SAMPLING_SELECTED_PROBABILITY = 0,
    YVEX_CUDA_SAMPLING_MIN_P_THRESHOLD,
    YVEX_CUDA_SAMPLING_ENTROPY,
    YVEX_CUDA_SAMPLING_TYPICAL_MASS,
    YVEX_CUDA_SAMPLING_TOP_P_MASS,
    YVEX_CUDA_SAMPLING_NORMALIZATION_ERROR,
    YVEX_CUDA_SAMPLING_STATISTIC_FIELDS
};
enum {
    YVEX_CUDA_SAMPLING_SELECTED_LOGIT = 0,
    YVEX_CUDA_SAMPLING_MAXIMUM_LOGIT,
    YVEX_CUDA_SAMPLING_VALUE_FIELDS
};
enum {
    YVEX_CUDA_SAMPLING_SELECTED_TOKEN = 0,
    YVEX_CUDA_SAMPLING_NUMERIC_FALLBACK,
    YVEX_CUDA_SAMPLING_SELECTION_FIELDS
};
enum {
    YVEX_CUDA_SPECULATION_PROPOSED_COUNT = 0,
    YVEX_CUDA_SPECULATION_ACCEPTED_COUNT,
    YVEX_CUDA_SPECULATION_REJECTED_COUNT,
    YVEX_CUDA_SPECULATION_COMMITTED_COUNT,
    YVEX_CUDA_SPECULATION_REJECTION_INDEX,
    YVEX_CUDA_SPECULATION_ALL_ACCEPTED,
    YVEX_CUDA_SPECULATION_CORRECTION_PRESENT,
    YVEX_CUDA_SPECULATION_BONUS_PRESENT,
    YVEX_CUDA_SPECULATION_CORRECTION_TOKEN,
    YVEX_CUDA_SPECULATION_RESULT_FIELDS
};
#ifdef __CUDACC__
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
static __device__ int mxfp4_i8x4(unsigned int packed, unsigned int high)
{
    unsigned int values = 0u;
#pragma unroll
    for (unsigned int i = 0u; i < 4u; ++i) {
        unsigned int code = (packed >> (8u * i + (high ? 4u : 0u))) & 15u;
        int magnitude = (code & 7u) == 0u ? 0 :
                        (code & 7u) < 5u ? (int)(code & 7u) :
                        (code & 7u) == 5u ? 6 :
                        (code & 7u) == 6u ? 8 : 12;
        int value = code & 8u ? -magnitude : magnitude;
        values |= ((unsigned int)value & 255u) << (8u * i);
    }
    return (int)values;
}
static __device__ float mxfp4_q8_k_dot(const unsigned char *weight,
                                       const unsigned char *activation)
{
    float activation_scale = __uint_as_float(qtype_load_u32(activation));
    float total = 0.0f;
#pragma unroll
    for (unsigned int block = 0u; block < 8u; ++block) {
        const unsigned char *mxfp4 = weight + block * 17u;
        const unsigned char *q8 = activation + 4u + block * 32u;
        int dot = 0;
#pragma unroll
        for (unsigned int i = 0u; i < 16u; i += 4u) {
            unsigned int packed = qtype_load_u32(mxfp4 + 1u + i);
            dot = __dp4a(mxfp4_i8x4(packed, 0u), (int)qtype_load_u32(q8 + i), dot);
            dot = __dp4a(mxfp4_i8x4(packed, 1u),
                         (int)qtype_load_u32(q8 + 16u + i), dot);
        }
        total = fmaf(e8m0_bits_to_float(mxfp4[0]) * 0.5f * activation_scale,
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
    if (qtype == YVEX_GGUF_QTYPE_MXFP4) return mxfp4_q8_k_dot(weight, activation);
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
#endif
#endif
