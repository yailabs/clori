/* Owner: gguf.quant block codecs (TRACK.QUANT).
 * Owns: deterministic F32/F16/BF16/I32, Q8_0, Q2_K, IQ2_XXS, and MXFP4 bytes.
 * Does not own: qtype IDs/geometry, source IO, profile selection, CUDA kernels, artifact layout, writing,
 *   materialization, or rendering.
 * Invariants: block layouts match the pinned GGUF ABI; every conversion checks arity, capacity, non-finite policy,
 *   and little-endian scalar storage.
 * Boundary: bounded encoded blocks are writer inputs, not a GGUF artifact.
 * Purpose: encode and independently reconstruct the closed qtype block set used by release plans.
 * Inputs: exact finite scalar blocks, canonical qtype identity, and caller-owned byte/value buffers.
 * Effects: writes only the requested complete block and typed diagnostic state.
 * Failure: unsupported qtypes, malformed arity, insufficient capacity, or non-finite data refuse. */
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/quant_numeric.h>

/* Pinned GGML stores the E2M1 codebook doubled and applies E8M0 / 2. */
static const float quant_mxfp4_values[16] = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  6.0f,
                                             8.0f,  12.0f, -0.0f, -1.0f, -2.0f, -3.0f,
                                             -4.0f, -6.0f, -8.0f, -12.0f};

/* Pinned IQ2_XXS magnitude-grid identities, represented as eight 2-bit digits. */
static const uint16_t quant_iq2_grid[256] = {
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97,
    100, 128, 130, 138, 162, 257, 260, 272, 277, 320, 388, 408, 512, 514, 546, 642,
    1025, 1028, 1040, 1057, 1060, 1088, 1090, 1096, 1120, 1153, 1156, 1168, 1188, 1280,
    1282, 1288, 1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600, 1668, 1700, 2048, 2053,
    2056, 2068, 2088, 2113, 2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580, 4097, 4100,
    4112, 4129, 4160, 4192, 4228, 4240, 4245, 4352, 4360, 4384, 4432, 4442, 4480, 4644,
    4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400, 5474, 5632, 5654, 6145, 6148, 6160,
    6208, 6273, 6400, 6405, 6560, 6737, 8192, 8194, 8202, 8260, 8289, 8320, 8322, 8489,
    8520, 8704, 8706, 9217, 9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872, 10248, 10272,
    10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480,
    16513, 16516, 16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982,
    17000, 17408, 17416, 17440, 17536, 17561, 17682, 17700, 17920, 18433, 18436, 18448,
    18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488, 20497, 20505,
    20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100,
    22528, 22545, 22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917,
    25112, 25184, 25600, 25605, 25872, 25874, 25988, 26690, 32768, 32770, 32778, 32833,
    32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856, 33888, 34048,
    34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928,
    37025, 37142, 37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040,
    41093, 41225, 41472, 42008, 43088, 43268,
};
static unsigned char quant_iq2_nearest[43691];
static pthread_once_t quant_iq2_once = PTHREAD_ONCE_INIT;

/* Purpose: build the immutable nearest-grid projection once with lowest-index tie breaking.
 * Inputs: pinned compact IQ2 grid table.
 * Effects: initializes one process-lifetime lookup table under pthread_once.
 * Failure: exhaustive bounded construction has no fallible operation.
 * Boundary: lookup construction does not encode model data or select policy. */
static void quant_iq2_initialize(void) {
    unsigned int packed;

    for (packed = 0u; packed < sizeof(quant_iq2_nearest); ++packed) {
        unsigned int best = 0u;
        unsigned int best_distance = UINT_MAX;
        unsigned int grid;
        for (grid = 0u; grid < 256u; ++grid) {
            unsigned int distance = 0u;
            unsigned int lane;
            for (lane = 0u; lane < 8u; ++lane) {
                int left = (int)((packed >> (2u * lane)) & 3u);
                int right = (int)((quant_iq2_grid[grid] >> (2u * lane)) & 3u);
                int delta = left - right;
                distance += (unsigned int)(delta * delta);
            }
            if (distance < best_distance) {
                best = grid;
                best_distance = distance;
            }
        }
        quant_iq2_nearest[packed] = (unsigned char)best;
    }
}

/* Purpose: publish one typed block-codec refusal with exact qtype and size facts.
 * Inputs: optional diagnostics, code, qtype, expected/actual values, status, and message.
 * Effects: replaces supplied failure and error records without modifying codec buffers.
 * Failure: represents the supplied refusal and returns no capability state.
 * Boundary: diagnostics do not own executor cleanup. */
static void quant_block_fail(yvex_quant_failure *failure, yvex_quant_failure_code code,
                             unsigned int qtype, unsigned long long expected,
                             unsigned long long actual, yvex_error *err, int status,
                             const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->terminal_ordinal = ULLONG_MAX;
        failure->source_index = ULLONG_MAX;
        failure->row_index = ULLONG_MAX;
        failure->block_index = ULLONG_MAX;
        failure->expected = expected;
        failure->actual = actual;
        failure->qtype = qtype;
        failure->operation = YVEX_TRANSFORM_OP_COUNT;
    }
    yvex_error_set(err, (yvex_status)status, "quant.block", message);
}

/* Purpose: store one unsigned 16-bit value in canonical little-endian order. */
static void quant_store_u16(unsigned char *out, unsigned short value) {
    out[0] = (unsigned char)(value & 0xffu);
    out[1] = (unsigned char)(value >> 8);
}

/* Purpose: store one unsigned 32-bit value in canonical little-endian order. */
static void quant_store_u32(unsigned char *out, unsigned int value) {
    out[0] = (unsigned char)(value & 0xffu);
    out[1] = (unsigned char)((value >> 8) & 0xffu);
    out[2] = (unsigned char)((value >> 16) & 0xffu);
    out[3] = (unsigned char)(value >> 24);
}

/* Purpose: round a bounded scalar to the nearest integer with ties resolved to even. */
static int quant_nearest_even(float value) {
    float lower = floorf(value);
    float fraction = value - lower;
    int integer = (int)lower;

    if (fraction > 0.5f || (fraction == 0.5f && (integer & 1)))
        integer++;
    return integer;
}

/* Purpose: validate that every value in one bounded block is finite.
 * Inputs: scalar block, element count, and optional bad-index output.
 * Effects: writes the first non-finite index on refusal.
 * Failure: returns false at the first NaN or infinity.
 * Boundary: the caller owns block-size and pointer admission. */
static int quant_values_finite(const float *values, unsigned long long count,
                               unsigned long long *bad) {
    unsigned long long index;
    for (index = 0u; index < count; ++index) {
        if (!isfinite(values[index])) {
            if (bad)
                *bad = index;
            return 0;
        }
    }
    return 1;
}

/* Purpose: select the nearest pinned E2M1 code with deterministic first-code tie breaking. */
static unsigned int quant_mxfp4_best(float value, float scale) {
    unsigned int best = 0u;
    float best_error = fabsf(value - quant_mxfp4_values[0] * scale);
    unsigned int index;

    for (index = 1u; index < 16u; ++index) {
        float error = fabsf(value - quant_mxfp4_values[index] * scale);
        if (error < best_error) {
            best = index;
            best_error = error;
        }
    }
    return best;
}

/* Purpose: choose the bounded E8M0 exponent that covers one MXFP4 block maximum.
 * Inputs: nonnegative finite maximum magnitude.
 * Effects: none.
 * Failure: nonpositive maxima map to the canonical zero-block exponent.
 * Boundary: exponent choice does not pack value nibbles. */
static unsigned char quant_mxfp4_exponent(float maximum) {
    int exponent;

    if (!(maximum > 0.0f))
        return 0u;
    exponent = (int)ceil(log2((double)maximum / 12.0)) + 128;
    if (exponent < 0)
        exponent = 0;
    if (exponent > 254)
        exponent = 254;
    return (unsigned char)exponent;
}

/* Purpose: encode one exact Q8_0 block using its F16 scale and signed lanes.
 * Inputs: thirty-two finite scalars and a canonical 34-byte destination.
 * Effects: writes the complete block deterministically.
 * Failure: returns false when the required scale is not representable as finite F16.
 * Boundary: caller validates qtype identity, arity, and capacity. */
static int quant_encode_q8_0(const float *source, unsigned char *encoded) {
    float maximum = 0.0f;
    float scale;
    float inverse;
    unsigned int index;

    for (index = 0u; index < YVEX_QUANT_Q8_0_ELEMENTS; ++index) {
        float magnitude = fabsf(source[index]);
        if (magnitude > maximum)
            maximum = magnitude;
    }
    scale = maximum / 127.0f;
    quant_store_u16(encoded, yvex_quant_f16_encode(scale));
    scale = yvex_quant_f16_decode(gguf_u16le_load(encoded));
    if (!isfinite(scale) || scale < 0.0f || (maximum > 0.0f && scale == 0.0f))
        return 0;
    inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    for (index = 0u; index < YVEX_QUANT_Q8_0_ELEMENTS; ++index) {
        int quantized = (int)roundf(source[index] * inverse);
        if (quantized < -127)
            quantized = -127;
        if (quantized > 127)
            quantized = 127;
        encoded[2u + index] = (unsigned char)(quantized < 0 ? 256 + quantized : quantized);
    }
    return 1;
}

/* Purpose: solve one 16-value Q2_K affine sub-block through the pinned deterministic search.
 * Inputs: sixteen finite scalars plus temporary code and minimum outputs.
 * Effects: writes temporary two-bit codes and the nonnegative minimum magnitude.
 * Failure: degenerate constant blocks return zero scale with valid zero codes.
 * Boundary: global F16 scale requantization occurs in the complete block encoder. */
static float quant_q2_subblock(const float *source, const float *calibration,
                               unsigned char *codes, float *minimum_out) {
    unsigned char candidate[16];
    float minimum = source[0];
    float maximum = source[0];
    float weight_sum = calibration ? calibration[0] : fabsf(source[0]);
    float weighted_source_sum = weight_sum * source[0];
    float scale;
    float inverse;
    float best_error = 0.0f;
    unsigned int index;
    unsigned int step;

    for (index = 1u; index < 16u; ++index) {
        float weight = calibration ? calibration[index] : fabsf(source[index]);
        if (source[index] < minimum)
            minimum = source[index];
        if (source[index] > maximum)
            maximum = source[index];
        weight_sum += weight;
        weighted_source_sum += weight * source[index];
    }
    if (minimum > 0.0f)
        minimum = 0.0f;
    if (maximum == minimum) {
        memset(codes, 0, 16u);
        *minimum_out = -minimum;
        return 0.0f;
    }
    inverse = 3.0f / (maximum - minimum);
    scale = 1.0f / inverse;
    for (index = 0u; index < 16u; ++index) {
        int code = quant_nearest_even(inverse * (source[index] - minimum));
        float difference;
        if (code < 0)
            code = 0;
        if (code > 3)
            code = 3;
        codes[index] = (unsigned char)code;
        difference = fabsf(scale * code + minimum - source[index]);
        best_error += fabsf(source[index]) * difference;
    }
    for (step = 0u; step <= 15u; ++step) {
        float code_sum = 0.0f;
        float code_squared_sum = 0.0f;
        float source_code_sum = 0.0f;
        float determinant;
        float candidate_scale;
        float candidate_minimum;
        float candidate_error = 0.0f;

        inverse = (-0.5f + 0.1f * (float)step + 3.0f) / (maximum - minimum);
        for (index = 0u; index < 16u; ++index) {
            float weight = calibration ? calibration[index] : fabsf(source[index]);
            int code = quant_nearest_even(inverse * (source[index] - minimum));
            if (code < 0)
                code = 0;
            if (code > 3)
                code = 3;
            candidate[index] = (unsigned char)code;
            code_sum += weight * code;
            code_squared_sum += weight * code * code;
            source_code_sum += weight * code * source[index];
        }
        determinant = weight_sum * code_squared_sum - code_sum * code_sum;
        if (!(determinant > 0.0f))
            continue;
        candidate_scale =
            (weight_sum * source_code_sum - weighted_source_sum * code_sum) / determinant;
        candidate_minimum =
            (code_squared_sum * weighted_source_sum - code_sum * source_code_sum) / determinant;
        if (candidate_minimum > 0.0f) {
            candidate_minimum = 0.0f;
            candidate_scale = code_squared_sum > 0.0f ? source_code_sum / code_squared_sum : 0.0f;
        }
        for (index = 0u; index < 16u; ++index) {
            float difference =
                fabsf(candidate_scale * candidate[index] + candidate_minimum - source[index]);
            candidate_error += (calibration ? calibration[index] : fabsf(source[index])) *
                               difference;
        }
        if (candidate_error < best_error) {
            memcpy(codes, candidate, 16u);
            best_error = candidate_error;
            scale = candidate_scale;
            minimum = candidate_minimum;
        }
    }
    *minimum_out = -minimum;
    return scale;
}

/* Purpose: encode one pinned MXFP4 block with E8M0 scale and low/high nibble ordering.
 * Inputs: thirty-two finite scalars and a canonical 17-byte destination.
 * Effects: writes one scale byte and sixteen paired-code bytes.
 * Failure: returns false when the derived E8M0 scale is non-finite.
 * Boundary: caller owns qtype and buffer admission. */
static int quant_encode_mxfp4(const float *source, unsigned char *encoded) {
    float maximum = 0.0f;
    float scale;
    unsigned int index;

    for (index = 0u; index < YVEX_QUANT_MXFP4_ELEMENTS; ++index) {
        float magnitude = fabsf(source[index]);
        if (magnitude > maximum)
            maximum = magnitude;
    }
    encoded[0] = quant_mxfp4_exponent(maximum);
    scale = yvex_quant_e8m0_decode(encoded[0]) * 0.5f;
    if (!isfinite(scale))
        return 0;
    for (index = 0u; index < 16u; ++index) {
        unsigned int low = quant_mxfp4_best(source[index], scale);
        unsigned int high = quant_mxfp4_best(source[index + 16u], scale);
        encoded[1u + index] = (unsigned char)(low | (high << 4));
    }
    return 1;
}

/* Purpose: encode one pinned Q2_K block with sixteen ordered affine sub-blocks.
 * Inputs: 256 finite scalars and a canonical 84-byte destination.
 * Effects: writes scale/min nibbles, packed two-bit lanes, and global F16 scales.
 * Failure: returns false when global affine scales cannot be represented as finite F16.
 * Boundary: no calibration or tensor-level policy is inferred here. */
static int quant_encode_q2_k(const float *source, const float *calibration,
                             unsigned char *encoded) {
    float scales[16];
    float minima[16];
    unsigned char quants[256];
    float maximum_scale = 0.0f;
    float maximum_minimum = 0.0f;
    float global_scale;
    float global_minimum;
    unsigned int subblock;
    unsigned int index;

    memset(encoded, 0, YVEX_QUANT_Q2_K_BYTES);
    for (subblock = 0u; subblock < 16u; ++subblock) {
        scales[subblock] = quant_q2_subblock(
            source + subblock * 16u,
            calibration ? calibration + subblock * 16u : NULL,
            quants + subblock * 16u, &minima[subblock]);
        if (scales[subblock] > maximum_scale)
            maximum_scale = scales[subblock];
        if (minima[subblock] > maximum_minimum)
            maximum_minimum = minima[subblock];
    }
    global_scale = maximum_scale / 15.0f;
    global_minimum = maximum_minimum / 15.0f;
    quant_store_u16(encoded + 80u, yvex_quant_f16_encode(global_scale));
    quant_store_u16(encoded + 82u, yvex_quant_f16_encode(global_minimum));
    global_scale = yvex_quant_f16_decode(gguf_u16le_load(encoded + 80u));
    global_minimum = yvex_quant_f16_decode(gguf_u16le_load(encoded + 82u));
    if (!isfinite(global_scale) || !isfinite(global_minimum) || global_scale < 0.0f ||
        global_minimum < 0.0f || (maximum_scale > 0.0f && global_scale == 0.0f) ||
        (maximum_minimum > 0.0f && global_minimum == 0.0f))
        return 0;
    for (subblock = 0u; subblock < 16u; ++subblock) {
        int scale_code =
            global_scale > 0.0f ? quant_nearest_even(scales[subblock] / global_scale) : 0;
        int minimum_code =
            global_minimum > 0.0f ? quant_nearest_even(minima[subblock] / global_minimum) : 0;
        float scale;
        float minimum;

        if (scale_code < 0)
            scale_code = 0;
        if (scale_code > 15)
            scale_code = 15;
        if (minimum_code < 0)
            minimum_code = 0;
        if (minimum_code > 15)
            minimum_code = 15;
        encoded[subblock] = (unsigned char)(scale_code | (minimum_code << 4));
        scale = global_scale * (float)scale_code;
        minimum = global_minimum * (float)minimum_code;
        for (index = 0u; index < 16u; ++index) {
            int code = scale > 0.0f
                           ? quant_nearest_even((source[subblock * 16u + index] + minimum) / scale)
                           : 0;
            if (code < 0)
                code = 0;
            if (code > 3)
                code = 3;
            quants[subblock * 16u + index] = (unsigned char)code;
        }
    }
    for (index = 0u; index < 256u; index += 128u) {
        unsigned int lane;
        for (lane = 0u; lane < 32u; ++lane) {
            encoded[16u + index / 4u + lane] =
                (unsigned char)(quants[index + lane] | (quants[index + lane + 32u] << 2) |
                                (quants[index + lane + 64u] << 4) |
                                (quants[index + lane + 96u] << 6));
        }
    }
    return 1;
}

/* Purpose: reconstruct the implicit even-parity eighth IQ2 sign bit. */
static unsigned int quant_iq2_sign_mask(unsigned int low) {
    unsigned int value = low;
    unsigned int parity = 0u;
    while (value) {
        parity ^= value & 1u;
        value >>= 1u;
    }
    return low | (parity << 7u);
}

/* Purpose: choose one compatible IQ2 magnitude grid for eight scaled magnitudes. */
static unsigned int quant_iq2_select_grid(const float *magnitudes, float inverse_scale) {
    unsigned int packed = 0u;
    unsigned int lane;

    for (lane = 0u; lane < 8u; ++lane) {
        int digit = quant_nearest_even(0.5f * (inverse_scale * magnitudes[lane] - 1.0f));
        if (digit < 0)
            digit = 0;
        if (digit > 2)
            digit = 2;
        packed |= (unsigned int)digit << (2u * lane);
    }
    return quant_iq2_nearest[packed];
}

/* Purpose: encode one imatrix-weighted compatible IQ2_XXS block.
 * Inputs: 256 finite source values and nonnegative finite per-column calibration weights.
 * Effects: writes one canonical 66-byte GGUF block after deterministic grid search.
 * Failure: false represents invalid weights or an unrepresentable finite scale.
 * Boundary: calibration selects codec error weighting, never tensor policy. */
static int quant_encode_iq2_xxs(const float *source, const float *calibration,
                                unsigned char *encoded) {
    float group_scales[8] = {0.0f};
    unsigned int group_words[16] = {0u};
    double square_sum = 0.0;
    float maximum_scale = 0.0f;
    unsigned int group;

    if (!calibration)
        return 0;
    pthread_once(&quant_iq2_once, quant_iq2_initialize);
    for (group = 0u; group < 256u; ++group) {
        if (!isfinite(calibration[group]) || calibration[group] < 0.0f)
            return 0;
        square_sum += (double)source[group] * (double)source[group];
    }
    memset(encoded, 0, YVEX_QUANT_IQ2_XXS_BYTES);
    for (group = 0u; group < 8u; ++group) {
        float magnitude[32];
        float weight[32];
        unsigned int signs[4];
        unsigned int grids[4] = {0u};
        float maximum = 0.0f;
        float scale;
        unsigned int lane;
        unsigned int iteration;

        for (lane = 0u; lane < 32u; ++lane) {
            unsigned int index = group * 32u + lane;
            float value = source[index];
            magnitude[lane] = fabsf(value);
            weight[lane] = calibration[index] *
                           sqrtf((float)(square_sum / 256.0) + value * value);
            if (!isfinite(weight[lane]))
                return 0;
            if (magnitude[lane] > maximum)
                maximum = magnitude[lane];
        }
        for (lane = 0u; lane < 4u; ++lane) {
            unsigned int bit;
            unsigned int mask = 0u;
            unsigned int negative_count = 0u;
            for (bit = 0u; bit < 8u; ++bit) {
                unsigned int index = lane * 8u + bit;
                if (source[group * 32u + index] < 0.0f) {
                    mask |= 1u << bit;
                    negative_count++;
                }
            }
            if (negative_count & 1u) {
                unsigned int least = 0u;
                float least_loss = weight[lane * 8u] * magnitude[lane * 8u] *
                                   magnitude[lane * 8u];
                for (bit = 1u; bit < 8u; ++bit) {
                    unsigned int index = lane * 8u + bit;
                    float loss = weight[index] * magnitude[index] * magnitude[index];
                    if (loss < least_loss) {
                        least = bit;
                        least_loss = loss;
                    }
                }
                magnitude[lane * 8u + least] = -magnitude[lane * 8u + least];
                mask ^= 1u << least;
            }
            signs[lane] = mask & 127u;
        }
        scale = maximum > 0.0f ? maximum / 5.0f : 0.0f;
        for (iteration = 0u; iteration < 3u && scale > 0.0f; ++iteration) {
            double numerator = 0.0;
            double denominator = 0.0;
            for (lane = 0u; lane < 4u; ++lane)
                grids[lane] = quant_iq2_select_grid(magnitude + lane * 8u, 1.0f / scale);
            for (lane = 0u; lane < 32u; ++lane) {
                unsigned int digit =
                    (quant_iq2_grid[grids[lane / 8u]] >> (2u * (lane & 7u))) & 3u;
                double level = (double)(2u * digit + 1u);
                numerator += (double)weight[lane] * (double)magnitude[lane] * level;
                denominator += (double)weight[lane] * level * level;
            }
            scale = denominator > 0.0 ? (float)(numerator / denominator) : 0.0f;
            if (scale < 0.0f) {
                scale = -scale;
                for (lane = 0u; lane < 4u; ++lane)
                    signs[lane] = (~signs[lane]) & 127u;
            }
        }
        for (lane = 0u; lane < 4u; ++lane) {
            group_words[2u * group] |= grids[lane] << (8u * lane);
            group_words[2u * group + 1u] |= signs[lane] << (7u * lane);
        }
        group_scales[group] = scale;
        if (scale > maximum_scale)
            maximum_scale = scale;
    }
    if (maximum_scale > 0.0f) {
        float scale = maximum_scale / 31.0f;
        unsigned short encoded_scale = yvex_quant_f16_encode(scale);
        float admitted_scale = yvex_quant_f16_decode(encoded_scale);
        if (!isfinite(admitted_scale) || !(admitted_scale > 0.0f))
            return 0;
        quant_store_u16(encoded, encoded_scale);
        for (group = 0u; group < 8u; ++group) {
            int code = quant_nearest_even(0.5f * (group_scales[group] / admitted_scale - 1.0f));
            if (code < 0)
                code = 0;
            if (code > 15)
                code = 15;
            group_words[2u * group + 1u] |= (unsigned int)code << 28u;
        }
    }
    for (group = 0u; group < 16u; ++group)
        quant_store_u32(encoded + 2u + 4u * group, group_words[group]);
    return 1;
}

/* Purpose: encode exactly one admitted scalar or block into canonical GGUF bytes.
 * Inputs: qtype, finite source values, exact arity, destination capacity, and diagnostics.
 * Effects: publishes encoded byte count only after a complete deterministic block write.
 * Failure: identity, codec, arity, capacity, finite-policy, scale, or cast refusal is typed.
 * Boundary: block encoding neither selects a profile nor emits a GGUF artifact. */
int yvex_quant_encode_block(unsigned int qtype, const float *source, unsigned long long elements,
                            unsigned char *encoded, size_t encoded_capacity, size_t *encoded_bytes,
                            yvex_quant_failure *failure, yvex_error *err) {
    const yvex_quant_numeric_capability *capability = yvex_quant_numeric_capability_at(qtype);
    const yvex_gguf_qtype_geometry *geometry = yvex_gguf_qtype_geometry_find(qtype);
    unsigned long long bad = ULLONG_MAX;
    unsigned long long required_elements;
    size_t required_bytes;

    if (encoded_bytes)
        *encoded_bytes = 0u;
    if (!source || !encoded || !encoded_bytes) {
        quant_block_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, qtype, 1u, 0u, err,
                         YVEX_ERR_INVALID_ARG,
                         "source, encoded buffer, and byte output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!capability) {
        quant_block_fail(failure, YVEX_QUANT_FAILURE_UNKNOWN_QTYPE, qtype,
                         YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID, qtype, err, YVEX_ERR_UNSUPPORTED,
                         "qtype identity is unknown");
        return YVEX_ERR_UNSUPPORTED;
    }
    if (!capability->encoder_available) {
        yvex_quant_failure_code code =
            capability->refusal == YVEX_QUANT_REFUSAL_REMOVED_IDENTITY
                ? YVEX_QUANT_FAILURE_REMOVED_QTYPE
            : capability->refusal == YVEX_QUANT_REFUSAL_OUTSIDE_PINNED_BASELINE
                ? YVEX_QUANT_FAILURE_QTYPE_OUTSIDE_BASELINE
                : YVEX_QUANT_FAILURE_ENCODER_UNAVAILABLE;
        quant_block_fail(failure, code, qtype, 1u, 0u, err, YVEX_ERR_UNSUPPORTED,
                         "qtype has no canonical encoder");
        return YVEX_ERR_UNSUPPORTED;
    }
    required_elements = geometry->block_size;
    required_bytes = geometry->bytes_per_block;
    if (elements != required_elements || encoded_capacity < required_bytes) {
        quant_block_fail(failure,
                         elements != required_elements ? YVEX_QUANT_FAILURE_ROW_DIVISIBILITY
                                                       : YVEX_QUANT_FAILURE_BYTE_OVERFLOW,
                         qtype, elements != required_elements ? required_elements : required_bytes,
                         elements != required_elements ? elements : encoded_capacity, err,
                         YVEX_ERR_BOUNDS, "qtype block arity or destination capacity mismatch");
        return YVEX_ERR_BOUNDS;
    }
    if (!quant_values_finite(source, elements, &bad)) {
        quant_block_fail(failure, YVEX_QUANT_FAILURE_NONFINITE, qtype, 0u, bad, err,
                         YVEX_ERR_FORMAT, "non-finite source value is forbidden by the profile");
        return YVEX_ERR_FORMAT;
    }
    switch (qtype) {
    case YVEX_GGUF_QTYPE_F32: {
        unsigned int bits;
        memcpy(&bits, source, sizeof(bits));
        quant_store_u32(encoded, bits);
        break;
    }
    case YVEX_GGUF_QTYPE_F16:
        quant_store_u16(encoded, yvex_quant_f16_encode(source[0]));
        break;
    case YVEX_GGUF_QTYPE_BF16:
        quant_store_u16(encoded, yvex_quant_bf16_encode(source[0]));
        break;
    case YVEX_GGUF_QTYPE_I32: {
        double rounded = nearbyint((double)source[0]);
        int32_t integer;
        if (rounded < INT32_MIN || rounded > INT32_MAX || rounded != (double)source[0]) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_CAST_RANGE, qtype, 0u, 1u, err,
                             YVEX_ERR_BOUNDS, "I32 encoding requires an exact in-range integer");
            return YVEX_ERR_BOUNDS;
        }
        integer = (int32_t)rounded;
        quant_store_u32(encoded, (unsigned int)integer);
        break;
    }
    case YVEX_GGUF_QTYPE_Q8_0:
        if (!quant_encode_q8_0(source, encoded)) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_Q8_0_BLOCK, qtype, 0u, 1u, err,
                             YVEX_ERR_BOUNDS, "Q8_0 scale is not representable as finite F16");
            return YVEX_ERR_BOUNDS;
        }
        break;
    case YVEX_GGUF_QTYPE_MXFP4:
        if (!quant_encode_mxfp4(source, encoded)) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_MXFP4_BLOCK, qtype, 0u, 1u, err,
                             YVEX_ERR_BOUNDS, "MXFP4 scale is not representable as finite E8M0");
            return YVEX_ERR_BOUNDS;
        }
        break;
    case YVEX_GGUF_QTYPE_Q2_K:
        if (!quant_encode_q2_k(source, NULL, encoded)) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_Q2_K_BLOCK, qtype, 0u, 1u, err,
                             YVEX_ERR_BOUNDS, "Q2_K affine scales are not finite F16 values");
            return YVEX_ERR_BOUNDS;
        }
        break;
    case YVEX_GGUF_QTYPE_IQ2_XXS:
        quant_block_fail(failure, YVEX_QUANT_FAILURE_CALIBRATION_REQUIRED, qtype,
                         YVEX_QUANT_IQ2_XXS_ELEMENTS, 0u, err, YVEX_ERR_INVALID_ARG,
                         "IQ2_XXS encoding requires explicit imatrix weights");
        return YVEX_ERR_INVALID_ARG;
    default:
        return YVEX_ERR_UNSUPPORTED;
    }
    *encoded_bytes = required_bytes;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: encode one admitted block with explicit per-column calibration weights.
 * Inputs: qtype, exact source/calibration block, exact destination, and diagnostics.
 * Effects: publishes one complete block and byte count only after weighted encoding succeeds.
 * Failure: invalid calibration, geometry, finite policy, or codec state publishes zero bytes.
 * Boundary: this operation consumes calibration selected by a sealed plan; it does not resolve policy. */
int yvex_quant_encode_block_weighted(unsigned int qtype, const float *source,
                                     const float *calibration_weights,
                                     unsigned long long elements, unsigned char *encoded,
                                     size_t encoded_capacity, size_t *encoded_bytes,
                                     yvex_quant_failure *failure, yvex_error *err) {
    unsigned long long bad = ULLONG_MAX;

    if (qtype != YVEX_GGUF_QTYPE_IQ2_XXS && qtype != YVEX_GGUF_QTYPE_Q2_K)
        return yvex_quant_encode_block(qtype, source, elements, encoded, encoded_capacity,
                                       encoded_bytes, failure, err);
    if (encoded_bytes)
        *encoded_bytes = 0u;
    if (!source || !calibration_weights || !encoded || !encoded_bytes) {
        quant_block_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, qtype, 1u, 0u, err,
                         YVEX_ERR_INVALID_ARG,
                         "source, calibration, destination, and byte output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (elements != YVEX_QUANT_IQ2_XXS_ELEMENTS ||
        encoded_capacity < (qtype == YVEX_GGUF_QTYPE_IQ2_XXS
                                ? YVEX_QUANT_IQ2_XXS_BYTES
                                : YVEX_QUANT_Q2_K_BYTES)) {
        quant_block_fail(failure,
                         qtype == YVEX_GGUF_QTYPE_IQ2_XXS
                             ? YVEX_QUANT_FAILURE_IQ2_XXS_BLOCK
                             : YVEX_QUANT_FAILURE_Q2_K_BLOCK,
                         qtype, YVEX_QUANT_IQ2_XXS_ELEMENTS, elements, err,
                         YVEX_ERR_BOUNDS, "weighted block arity or encoded capacity mismatch");
        return YVEX_ERR_BOUNDS;
    }
    if (!quant_values_finite(source, elements, &bad) ||
        !quant_values_finite(calibration_weights, elements, &bad)) {
        quant_block_fail(failure, YVEX_QUANT_FAILURE_NONFINITE, qtype, 0u, bad, err,
                         YVEX_ERR_FORMAT, "IQ2_XXS source and imatrix weights must be finite");
        return YVEX_ERR_FORMAT;
    }
    for (bad = 0u; bad < elements; ++bad) {
        if (calibration_weights[bad] < 0.0f) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_CALIBRATION_IDENTITY, qtype, 0u,
                             bad, err, YVEX_ERR_FORMAT,
                             "imatrix weights must be nonnegative");
            return YVEX_ERR_FORMAT;
        }
    }
    if (qtype == YVEX_GGUF_QTYPE_Q2_K) {
        if (!quant_encode_q2_k(source, calibration_weights, encoded)) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_Q2_K_BLOCK, qtype, 1u, 0u, err,
                             YVEX_ERR_FORMAT,
                             "weighted Q2_K affine scales violate the codec contract");
            return YVEX_ERR_FORMAT;
        }
        *encoded_bytes = YVEX_QUANT_Q2_K_BYTES;
        if (failure)
            memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (!quant_encode_iq2_xxs(source, calibration_weights, encoded)) {
        quant_block_fail(failure, YVEX_QUANT_FAILURE_IQ2_XXS_BLOCK, qtype, 1u, 0u, err,
                         YVEX_ERR_FORMAT,
                         "IQ2_XXS weights or derived scale violate the codec contract");
        return YVEX_ERR_FORMAT;
    }
    *encoded_bytes = YVEX_QUANT_IQ2_XXS_BYTES;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: reconstruct one pinned Q2_K block from packed affine sub-block state.
 * Inputs: canonical 84-byte block and 256-value output.
 * Effects: replaces every output scalar in deterministic sub-block order.
 * Failure: none after caller-owned exact-size admission.
 * Boundary: this primitive shares no encoder search logic. */
static void quant_decode_q2_k(const unsigned char *encoded, float *out) {
    float global_scale = yvex_quant_f16_decode(gguf_u16le_load(encoded + 80u));
    float global_minimum = yvex_quant_f16_decode(gguf_u16le_load(encoded + 82u));
    const unsigned char *quants = encoded + 16u;
    unsigned int subblock = 0u;
    unsigned int half;

    for (half = 0u; half < 2u; ++half) {
        unsigned int shift = 0u;
        unsigned int group;
        for (group = 0u; group < 4u; ++group) {
            unsigned int pair;
            for (pair = 0u; pair < 2u; ++pair) {
                unsigned char scale_byte = encoded[subblock];
                float scale = global_scale * (float)(scale_byte & 0x0fu);
                float minimum = global_minimum * (float)(scale_byte >> 4);
                unsigned int lane;
                unsigned int quant_offset = pair * 16u;
                for (lane = 0u; lane < 16u; ++lane) {
                    unsigned int code = (quants[quant_offset + lane] >> shift) & 3u;
                    out[subblock * 16u + lane] = scale * (float)code - minimum;
                }
                subblock++;
            }
            shift += 2u;
        }
        quants += 32u;
    }
}

/* Purpose: reconstruct one compatible IQ2_XXS block from grid, sign, and scale state.
 * Inputs: exact admitted 66-byte block and 256-element caller output.
 * Effects: writes the complete reconstructed F32 block.
 * Failure: caller validates geometry and scale before this infallible helper.
 * Boundary: direct reconstruction does not select qtype or allocate storage. */
static void quant_decode_iq2_xxs(const unsigned char *encoded, float *out) {
    float block_scale = yvex_quant_f16_decode(gguf_u16le_load(encoded));
    unsigned int group;

    for (group = 0u; group < 8u; ++group) {
        unsigned int grids = gguf_u32le_load(encoded + 2u + group * 8u);
        unsigned int signs_and_scale = gguf_u32le_load(encoded + 6u + group * 8u);
        float group_scale = block_scale * (0.5f + (float)(signs_and_scale >> 28u)) * 0.25f;
        unsigned int subgroup;
        for (subgroup = 0u; subgroup < 4u; ++subgroup) {
            unsigned int grid = (grids >> (8u * subgroup)) & 0xffu;
            unsigned int signs =
                quant_iq2_sign_mask((signs_and_scale >> (7u * subgroup)) & 127u);
            unsigned int lane;
            for (lane = 0u; lane < 8u; ++lane) {
                unsigned int digit = (quant_iq2_grid[grid] >> (2u * lane)) & 3u;
                float level = digit == 0u ? 8.0f : digit == 1u ? 25.0f : 43.0f;
                out[group * 32u + subgroup * 8u + lane] =
                    group_scale * level * ((signs & (1u << lane)) ? -1.0f : 1.0f);
            }
        }
    }
}

/* Purpose: reference-decode one exact admitted scalar or qtype block.
 * Inputs: qtype, exact encoded bytes, exact output arity, and diagnostics.
 * Effects: publishes the complete reconstructed block after size and codec admission.
 * Failure: unknown/decoderless qtype or malformed block size returns typed refusal.
 * Boundary: reference decoding is an oracle primitive, not dedicated backend compute. */
int yvex_quant_decode_block(unsigned int qtype, const unsigned char *encoded, size_t encoded_bytes,
                            float *out, unsigned long long out_elements,
                            yvex_quant_failure *failure, yvex_error *err) {
    const yvex_quant_numeric_capability *capability = yvex_quant_numeric_capability_at(qtype);
    const yvex_gguf_qtype_geometry *geometry = yvex_gguf_qtype_geometry_find(qtype);
    size_t required_bytes;
    unsigned long long required_elements;
    unsigned int index;

    if (!encoded || !out) {
        quant_block_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, qtype, 1u, 0u, err,
                         YVEX_ERR_INVALID_ARG, "encoded block and decode output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!capability || !capability->reference_decoder_available) {
        quant_block_fail(
            failure,
            capability ? YVEX_QUANT_FAILURE_DECODER_UNAVAILABLE : YVEX_QUANT_FAILURE_UNKNOWN_QTYPE,
            qtype, 1u, 0u, err, YVEX_ERR_UNSUPPORTED, "qtype has no canonical reference decoder");
        return YVEX_ERR_UNSUPPORTED;
    }
    required_bytes = geometry->bytes_per_block;
    required_elements = geometry->block_size;
    if (encoded_bytes != required_bytes || out_elements != required_elements) {
        quant_block_fail(failure,
                         qtype == YVEX_GGUF_QTYPE_Q8_0    ? YVEX_QUANT_FAILURE_Q8_0_BLOCK
                         : qtype == YVEX_GGUF_QTYPE_Q2_K  ? YVEX_QUANT_FAILURE_Q2_K_BLOCK
                         : qtype == YVEX_GGUF_QTYPE_IQ2_XXS
                             ? YVEX_QUANT_FAILURE_IQ2_XXS_BLOCK
                         : qtype == YVEX_GGUF_QTYPE_MXFP4 ? YVEX_QUANT_FAILURE_MXFP4_BLOCK
                                                          : YVEX_QUANT_FAILURE_BYTE_OVERFLOW,
                         qtype, required_bytes, encoded_bytes, err, YVEX_ERR_FORMAT,
                         "encoded qtype block has an exact-size mismatch");
        return YVEX_ERR_FORMAT;
    }
    switch (qtype) {
    case YVEX_GGUF_QTYPE_F32: {
        unsigned int bits = gguf_u32le_load(encoded);
        memcpy(out, &bits, sizeof(*out));
        break;
    }
    case YVEX_GGUF_QTYPE_F16:
        out[0] = yvex_quant_f16_decode(gguf_u16le_load(encoded));
        break;
    case YVEX_GGUF_QTYPE_BF16:
        out[0] = yvex_quant_bf16_decode(gguf_u16le_load(encoded));
        break;
    case YVEX_GGUF_QTYPE_I32:
        out[0] = (float)gguf_i32_from_u32(gguf_u32le_load(encoded));
        break;
    case YVEX_GGUF_QTYPE_Q8_0: {
        float scale = yvex_quant_f16_decode(gguf_u16le_load(encoded));
        if (!isfinite(scale) || scale < 0.0f) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_Q8_0_BLOCK, qtype, 0u,
                             gguf_u16le_load(encoded), err, YVEX_ERR_FORMAT,
                             "Q8_0 encoded scale is negative or non-finite");
            return YVEX_ERR_FORMAT;
        }
        for (index = 0u; index < YVEX_QUANT_Q8_0_ELEMENTS; ++index) {
            int quantized =
                encoded[2u + index] <= 127u ? encoded[2u + index] : (int)encoded[2u + index] - 256;
            out[index] = scale * (float)quantized;
        }
        break;
    }
    case YVEX_GGUF_QTYPE_MXFP4: {
        float scale = yvex_quant_e8m0_decode(encoded[0]) * 0.5f;
        if (!isfinite(scale)) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_MXFP4_BLOCK, qtype, 0xfeu, encoded[0], err,
                             YVEX_ERR_FORMAT, "MXFP4 encoded scale is non-finite");
            return YVEX_ERR_FORMAT;
        }
        for (index = 0u; index < 16u; ++index) {
            out[index] = quant_mxfp4_values[encoded[1u + index] & 0x0fu] * scale;
            out[index + 16u] = quant_mxfp4_values[encoded[1u + index] >> 4] * scale;
            if (!isfinite(out[index]) || !isfinite(out[index + 16u])) {
                quant_block_fail(failure, YVEX_QUANT_FAILURE_MXFP4_BLOCK, qtype, 0u, index, err,
                                 YVEX_ERR_FORMAT, "MXFP4 block reconstructs a non-finite value");
                return YVEX_ERR_FORMAT;
            }
        }
        break;
    }
    case YVEX_GGUF_QTYPE_Q2_K: {
        float scale = yvex_quant_f16_decode(gguf_u16le_load(encoded + 80u));
        float minimum = yvex_quant_f16_decode(gguf_u16le_load(encoded + 82u));
        if (!isfinite(scale) || !isfinite(minimum) || scale < 0.0f || minimum < 0.0f) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_Q2_K_BLOCK, qtype, 0u, 1u, err,
                             YVEX_ERR_FORMAT, "Q2_K encoded affine scales are malformed");
            return YVEX_ERR_FORMAT;
        }
        quant_decode_q2_k(encoded, out);
        break;
    }
    case YVEX_GGUF_QTYPE_IQ2_XXS: {
        float scale = yvex_quant_f16_decode(gguf_u16le_load(encoded));
        if (!isfinite(scale) || scale < 0.0f) {
            quant_block_fail(failure, YVEX_QUANT_FAILURE_IQ2_XXS_BLOCK, qtype, 0u,
                             gguf_u16le_load(encoded), err, YVEX_ERR_FORMAT,
                             "IQ2_XXS encoded scale is negative or non-finite");
            return YVEX_ERR_FORMAT;
        }
        quant_decode_iq2_xxs(encoded, out);
        break;
    }
    default:
        return YVEX_ERR_UNSUPPORTED;
    }
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}
