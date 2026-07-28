/*
 * quant_numeric.c - canonical quantization codec and capability tests.
 *
 * Owner: tests/unit.
 * Owns: scalar edge cases, block golden geometry, independent decode metrics,
 *   direct CPU qtype compute, and exhaustive capability identity coverage.
 * Does not own: release policy, payload IO, CUDA, artifact output, or claims.
 * Invariants: tests retain no model payload and use deterministic bounded data.
 * Boundary: codec arithmetic is not complete-model quantization evidence.
 */
#include "tests/test.h"

#include <yvex/internal/quant_numeric.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static const uint16_t quant_reference_iq2_grid[256] = {
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

static int quant_float_bits_equal(float left, float right)
{
    uint32_t left_bits;
    uint32_t right_bits;

    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

/* Decodes binary16 independently from the production scalar codec. */
static float quant_reference_f16(const unsigned char *encoded)
{
    unsigned int bits = encoded[0] | ((unsigned int)encoded[1] << 8);
    unsigned int sign = bits >> 15;
    unsigned int exponent = (bits >> 10) & 0x1fu;
    unsigned int mantissa = bits & 0x3ffu;
    float value;

    if (exponent == 0u)
        value = mantissa == 0u ? 0.0f : ldexpf((float)mantissa, -24);
    else if (exponent == 0x1fu)
        value = mantissa == 0u ? INFINITY : NAN;
    else
        value = ldexpf((float)(1024u + mantissa), (int)exponent - 25);
    return sign ? -value : value;
}

static unsigned int quant_reference_parity_sign(unsigned int low)
{
    unsigned int parity = 0u;
    unsigned int value = low;
    while (value) {
        parity ^= value & 1u;
        value >>= 1u;
    }
    return low | (parity << 7u);
}

/* Independently unpacks one canonical block without calling production decode. */
static int quant_reference_block(unsigned int qtype,
                                 const unsigned char *encoded,
                                 float *out)
{
    static const float mxfp4[16] = {
         0.0f,  1.0f,  2.0f,  3.0f,
         4.0f,  6.0f,  8.0f, 12.0f,
        -0.0f, -1.0f, -2.0f, -3.0f,
        -4.0f, -6.0f, -8.0f,-12.0f
    };
    unsigned int index;

    if (qtype == YVEX_GGUF_QTYPE_F32) {
        uint32_t bits = (uint32_t)encoded[0] |
                        ((uint32_t)encoded[1] << 8) |
                        ((uint32_t)encoded[2] << 16) |
                        ((uint32_t)encoded[3] << 24);
        memcpy(out, &bits, sizeof(bits));
        return 1;
    }
    if (qtype == YVEX_GGUF_QTYPE_F16) {
        out[0] = quant_reference_f16(encoded);
        return 1;
    }
    if (qtype == YVEX_GGUF_QTYPE_BF16) {
        uint32_t bits = ((uint32_t)encoded[0] |
                         ((uint32_t)encoded[1] << 8)) << 16;
        memcpy(out, &bits, sizeof(bits));
        return 1;
    }
    if (qtype == YVEX_GGUF_QTYPE_I32) {
        uint32_t bits = (uint32_t)encoded[0] |
                        ((uint32_t)encoded[1] << 8) |
                        ((uint32_t)encoded[2] << 16) |
                        ((uint32_t)encoded[3] << 24);
        int32_t value = bits <= (uint32_t)INT32_MAX
            ? (int32_t)bits : -1 - (int32_t)(UINT32_MAX - bits);
        out[0] = (float)value;
        return 1;
    }
    if (qtype == YVEX_GGUF_QTYPE_Q8_0) {
        float scale = quant_reference_f16(encoded);
        for (index = 0u; index < YVEX_QUANT_Q8_0_ELEMENTS; ++index) {
            int code = encoded[index + 2u] < 128u
                ? encoded[index + 2u]
                : (int)encoded[index + 2u] - 256;
            out[index] = scale * (float)code;
        }
        return 1;
    }
    if (qtype == YVEX_GGUF_QTYPE_MXFP4) {
        float scale = encoded[0] == 0xffu
            ? NAN : ldexpf(0.5f, (int)encoded[0] - 127);
        for (index = 0u; index < 16u; ++index) {
            out[index] = mxfp4[encoded[index + 1u] & 0x0fu] * scale;
            out[index + 16u] =
                mxfp4[encoded[index + 1u] >> 4] * scale;
        }
        return 1;
    }
    if (qtype == YVEX_GGUF_QTYPE_Q2_K) {
        float global_scale = quant_reference_f16(encoded + 80u);
        float global_minimum = quant_reference_f16(encoded + 82u);
        for (index = 0u; index < YVEX_QUANT_Q2_K_ELEMENTS; ++index) {
            unsigned int subblock = index / 16u;
            unsigned int lane = index % 16u;
            unsigned int shift = ((subblock % 8u) / 2u) * 2u;
            unsigned int byte_index = 16u + (subblock / 8u) * 32u +
                                      (subblock % 2u) * 16u + lane;
            unsigned int code = (encoded[byte_index] >> shift) & 3u;
            float scale = global_scale *
                (float)(encoded[subblock] & 0x0fu);
            float minimum = global_minimum *
                (float)(encoded[subblock] >> 4);
            out[index] = scale * (float)code - minimum;
        }
        return 1;
    }
    if (qtype == YVEX_GGUF_QTYPE_IQ2_XXS) {
        float global_scale = quant_reference_f16(encoded);
        unsigned int group;
        for (group = 0u; group < 8u; ++group) {
            unsigned int word_offset = 2u + group * 8u;
            unsigned int grids = (unsigned int)encoded[word_offset] |
                                 ((unsigned int)encoded[word_offset + 1u] << 8) |
                                 ((unsigned int)encoded[word_offset + 2u] << 16) |
                                 ((unsigned int)encoded[word_offset + 3u] << 24);
            unsigned int signs_and_scale = (unsigned int)encoded[word_offset + 4u] |
                                           ((unsigned int)encoded[word_offset + 5u] << 8) |
                                           ((unsigned int)encoded[word_offset + 6u] << 16) |
                                           ((unsigned int)encoded[word_offset + 7u] << 24);
            unsigned int group_scale = signs_and_scale >> 28;
            unsigned int lane;
            for (lane = 0u; lane < 4u; ++lane) {
                unsigned int grid = quant_reference_iq2_grid[(grids >> (8u * lane)) & 0xffu];
                unsigned int signs = quant_reference_parity_sign(
                    (signs_and_scale >> (7u * lane)) & 127u);
                unsigned int bit;
                for (bit = 0u; bit < 8u; ++bit) {
                    unsigned int digit = (grid >> (2u * bit)) & 3u;
                    float magnitude = digit == 0u ? 8.0f : digit == 1u ? 25.0f : 43.0f;
                    unsigned int output = group * 32u + lane * 8u + bit;
                    out[output] = global_scale * (0.5f + (float)group_scale) * 0.25f *
                                  ((signs >> bit) & 1u ? -magnitude : magnitude);
                }
            }
        }
        return 1;
    }
    return 0;
}

/* Exhaustively proves finite binary16 decode/nearest-even encode identity. */
static int quant_test_f16(void)
{
    unsigned int bits;

    for (bits = 0u; bits <= 0xffffu; ++bits) {
        float value = yvex_quant_f16_decode((unsigned short)bits);
        unsigned int exponent = (bits >> 10) & 0x1fu;
        unsigned int mantissa = bits & 0x3ffu;
        unsigned short encoded = yvex_quant_f16_encode(value);

        if (exponent == 0x1fu && mantissa != 0u) {
            YVEX_TEST_ASSERT(isnan(value), "binary16 NaN must decode as NaN");
            YVEX_TEST_ASSERT((encoded & 0x7c00u) == 0x7c00u &&
                                 (encoded & 0x03ffu) != 0u,
                             "binary16 NaN must remain a NaN");
        } else {
            YVEX_TEST_ASSERT(encoded == (unsigned short)bits,
                             "binary16 finite/inf roundtrip must be exact");
        }
    }
    YVEX_TEST_ASSERT(yvex_quant_f16_encode(1.00048828125f) == 0x3c00u,
                     "binary16 halfway must round to even lower value");
    YVEX_TEST_ASSERT(yvex_quant_f16_encode(1.00146484375f) == 0x3c02u,
                     "binary16 halfway must round to even upper value");
    YVEX_TEST_ASSERT(signbit(yvex_quant_f16_decode(0x8000u)),
                     "binary16 negative zero sign must survive");
    YVEX_TEST_ASSERT(yvex_quant_f16_decode(0x0001u) == ldexpf(1.0f, -24),
                     "binary16 minimum subnormal must decode exactly");
    YVEX_TEST_ASSERT(yvex_quant_f16_decode(0x7bffu) == 65504.0f,
                     "binary16 finite maximum must decode exactly");
    YVEX_TEST_ASSERT(isinf(yvex_quant_f16_decode(0x7c00u)) &&
                         yvex_quant_f16_encode(FLT_MAX) == 0x7c00u,
                     "binary16 overflow policy must produce infinity");
    return 0;
}

/* Exhaustively proves finite BF16 decode/nearest-even encode identity. */
static int quant_test_bf16(void)
{
    unsigned int bits;

    for (bits = 0u; bits <= 0xffffu; ++bits) {
        float value = yvex_quant_bf16_decode((unsigned short)bits);
        unsigned int exponent = (bits >> 7) & 0xffu;
        unsigned int mantissa = bits & 0x7fu;
        unsigned short encoded = yvex_quant_bf16_encode(value);

        if (exponent == 0xffu && mantissa != 0u) {
            YVEX_TEST_ASSERT(isnan(value), "BF16 NaN must decode as NaN");
            YVEX_TEST_ASSERT((encoded & 0x7f80u) == 0x7f80u &&
                                 (encoded & 0x007fu) != 0u,
                             "BF16 NaN must remain a NaN");
        } else {
            YVEX_TEST_ASSERT(encoded == (unsigned short)bits,
                             "BF16 finite/inf roundtrip must be exact");
        }
    }
    YVEX_TEST_ASSERT(signbit(yvex_quant_bf16_decode(0x8000u)),
                     "BF16 negative zero sign must survive");
    {
        uint32_t halfway_bits = 0x3f808000u;
        uint32_t upper_odd_bits = 0x3f818000u;
        float halfway;
        float upper_odd;
        memcpy(&halfway, &halfway_bits, sizeof(halfway));
        memcpy(&upper_odd, &upper_odd_bits, sizeof(upper_odd));
        YVEX_TEST_ASSERT(yvex_quant_bf16_encode(halfway) == 0x3f80u &&
                             yvex_quant_bf16_encode(upper_odd) == 0x3f82u,
                         "BF16 halfway cases must round to even");
    }
    return 0;
}

/* Covers every pinned FP8/E8M0 code and source packed-nibble ordering. */
static int quant_test_source_formats(void)
{
    unsigned int code;
    unsigned char packed[16];
    float out[32];
    yvex_quant_failure failure;
    yvex_error error;
    int64_t integer;
    static const unsigned char minimum_i64[8] = {
        0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x80u
    };
    static const unsigned char maximum_i64[8] = {
        0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0x7fu
    };

    for (code = 0u; code <= 0xffu; ++code) {
        float fp8 = yvex_quant_fp8_e4m3fn_decode((unsigned char)code);
        float scale = yvex_quant_e8m0_decode((unsigned char)code);
        if ((code & 0x7fu) == 0x7fu)
            YVEX_TEST_ASSERT(isnan(fp8), "E4M3FN special code must be NaN");
        else
            YVEX_TEST_ASSERT(isfinite(fp8), "E4M3FN non-special code must be finite");
        if (code == 0xffu)
            YVEX_TEST_ASSERT(isnan(scale), "E8M0 0xff must be NaN");
        else
            YVEX_TEST_ASSERT(isfinite(scale) && scale > 0.0f,
                             "E8M0 admitted scale must be positive finite");
    }
    YVEX_TEST_ASSERT(yvex_quant_fp8_e4m3fn_decode(0x7eu) == 448.0f,
                     "E4M3FN finite maximum must match pinned format");
    YVEX_TEST_ASSERT(yvex_quant_e8m0_decode(127u) == 1.0f,
                     "E8M0 unbiased scale must equal one");
    memset(packed, 0x21, sizeof(packed));
    yvex_error_clear(&error);
    YVEX_TEST_ASSERT(yvex_quant_source_mxfp4_decode(
                         packed, 127u, out, &failure, &error) == YVEX_OK,
                     "source MXFP4 block must decode");
    for (code = 0u; code < 16u; ++code) {
        YVEX_TEST_ASSERT(out[code * 2u] == 0.5f &&
                             out[code * 2u + 1u] == 1.0f,
                         "source MXFP4 must use adjacent low/high nibbles");
    }
    YVEX_TEST_ASSERT(yvex_quant_source_mxfp4_decode(
                         packed, 0xffu, out, &failure, &error) ==
                         YVEX_ERR_FORMAT &&
                         failure.code == YVEX_QUANT_FAILURE_MXFP4_BLOCK,
                     "source MXFP4 must refuse non-finite scale");
    YVEX_TEST_ASSERT(yvex_quant_source_i64_decode(
                         minimum_i64, &integer, &failure, &error) == YVEX_OK &&
                         integer == INT64_MIN,
                     "little-endian I64 minimum must decode exactly");
    YVEX_TEST_ASSERT(yvex_quant_source_i64_decode(
                         maximum_i64, &integer, &failure, &error) == YVEX_OK &&
                         integer == INT64_MAX,
                     "little-endian I64 maximum must decode exactly");
    YVEX_TEST_ASSERT(yvex_quant_source_i64_decode(
                         NULL, &integer, &failure, &error) ==
                         YVEX_ERR_INVALID_ARG &&
                         failure.code == YVEX_QUANT_FAILURE_INVALID_ARGUMENT,
                     "exact I64 decoding must reject a missing source");
    return 0;
}

static void quant_fill(float *values, unsigned int count, float scale)
{
    unsigned int index;
    for (index = 0u; index < count; ++index) {
        int centered = (int)((index * 37u + 11u) % 101u) - 50;
        values[index] = (float)centered * scale;
    }
}

/* Proves exact block geometry, deterministic packing, metrics, and CPU dot. */
static int quant_test_block(unsigned int qtype,
                            unsigned int elements,
                            size_t bytes,
                            double maximum_error_limit)
{
    float source[YVEX_QUANT_Q2_K_ELEMENTS];
    float decoded[YVEX_QUANT_Q2_K_ELEMENTS];
    float reference_decoded[YVEX_QUANT_Q2_K_ELEMENTS];
    float vector[YVEX_QUANT_Q2_K_ELEMENTS];
    unsigned char encoded[YVEX_QUANT_Q2_K_BYTES];
    unsigned char repeated[YVEX_QUANT_Q2_K_BYTES];
    yvex_quant_metrics metrics;
    yvex_quant_failure failure;
    yvex_error error;
    size_t encoded_bytes = 0u;
    size_t repeated_bytes = 0u;
    float cpu_dot = 0.0f;
    double dot_absolute_error;
    double mean_absolute_error;
    double mean_relative_error;
    double reference_dot = 0.0;
    unsigned int index;

    quant_fill(source, elements, qtype == YVEX_GGUF_QTYPE_Q2_K
                                    ? 0.075f : 0.03125f);
    for (index = 0u; index < elements; ++index)
        vector[index] = (float)((int)(index % 13u) - 6) / 7.0f;
    yvex_error_clear(&error);
    YVEX_TEST_ASSERT(yvex_quant_encode_block(
                         qtype, source, elements, encoded, sizeof(encoded),
                         &encoded_bytes, &failure, &error) == YVEX_OK,
                     "canonical qtype block must encode");
    YVEX_TEST_ASSERT(encoded_bytes == bytes,
                     "encoded block must use canonical byte geometry");
    YVEX_TEST_ASSERT(yvex_quant_encode_block(
                         qtype, source, elements, repeated, sizeof(repeated),
                         &repeated_bytes, &failure, &error) == YVEX_OK &&
                         repeated_bytes == bytes &&
                         memcmp(encoded, repeated, bytes) == 0,
                     "qtype encoding must be deterministic");
    YVEX_TEST_ASSERT(yvex_quant_decode_block(
                         qtype, encoded, encoded_bytes, decoded, elements,
                         &failure, &error) == YVEX_OK,
                     "canonical qtype block must reference-decode");
    YVEX_TEST_ASSERT(quant_reference_block(
                         qtype, encoded, reference_decoded),
                     "independent block decoder must admit the test qtype");
    for (index = 0u; index < elements; ++index)
        YVEX_TEST_ASSERT(quant_float_bits_equal(
                             decoded[index], reference_decoded[index]),
                         "production decoder must equal independent unpacking");
    yvex_quant_metrics_init(&metrics);
    YVEX_TEST_ASSERT(yvex_quant_metrics_update(
                         &metrics, source, reference_decoded, vector, elements),
                     "numeric metrics must accept bounded block");
    if (metrics.maximum_absolute_error > maximum_error_limit)
        fprintf(stderr, "quant block qtype=%u max_error=%.9g limit=%.9g\n",
                qtype, metrics.maximum_absolute_error, maximum_error_limit);
    YVEX_TEST_ASSERT(metrics.element_count == elements &&
                         metrics.nonfinite_count == 0u &&
                         metrics.maximum_absolute_error <= maximum_error_limit,
                     "qtype block error must remain within its test bound");
    mean_absolute_error = metrics.absolute_error_sum / (double)metrics.finite_count;
    mean_relative_error = metrics.relative_error_sum / (double)metrics.finite_count;
    dot_absolute_error = fabs(metrics.dot_reconstructed - metrics.dot_reference);
    YVEX_TEST_ASSERT(
        isfinite(mean_absolute_error) && isfinite(yvex_quant_metrics_rmse(&metrics)) &&
            isfinite(mean_relative_error) && isfinite(dot_absolute_error) &&
            mean_absolute_error >= 0.0 && yvex_quant_metrics_rmse(&metrics) >= 0.0 &&
            mean_relative_error >= 0.0 && dot_absolute_error >= 0.0,
        "lossy codec metrics must expose finite MAE, RMSE, relative, and dot error");
    YVEX_TEST_ASSERT(yvex_quant_cpu_dot(
                         qtype, encoded, encoded_bytes, vector, elements,
                         &cpu_dot, &failure, &error) == YVEX_OK,
                     "dedicated CPU encoded row dot must execute");
    for (index = 0u; index < elements; ++index)
        reference_dot += (double)reference_decoded[index] * vector[index];
    YVEX_TEST_ASSERT(fabs((double)cpu_dot - reference_dot) < 1e-4,
                     "CPU qtype dot must match independent decoded reference");
    YVEX_TEST_ASSERT(yvex_quant_decode_block(
                         qtype, encoded, encoded_bytes - 1u, decoded, elements,
                         &failure, &error) == YVEX_ERR_FORMAT,
                     "truncated qtype block must refuse");
    YVEX_TEST_ASSERT(yvex_quant_cpu_dot(
                         qtype, encoded, encoded_bytes, vector, elements - 1u,
                         &cpu_dot, &failure, &error) == YVEX_ERR_BOUNDS &&
                         failure.code == YVEX_QUANT_FAILURE_ROW_DIVISIBILITY,
                     "non-divisible qtype row must refuse");
    return 0;
}

/* Proves every selected exact scalar format against independent byte decoding. */
static int quant_test_exact_scalar_blocks(void)
{
    static const struct {
        unsigned int qtype;
        float value;
    } cases[] = {
        {YVEX_GGUF_QTYPE_F32, -3.25f},
        {YVEX_GGUF_QTYPE_F16, 1.00048828125f},
        {YVEX_GGUF_QTYPE_BF16, -0.333984375f},
        {YVEX_GGUF_QTYPE_I32, -2147483520.0f}
    };
    unsigned char encoded[4];
    float production;
    float reference;
    yvex_quant_failure failure;
    yvex_error error;
    unsigned int index;

    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        size_t encoded_bytes = 0u;
        YVEX_TEST_ASSERT(yvex_quant_encode_block(
                             cases[index].qtype, &cases[index].value, 1u,
                             encoded, sizeof(encoded), &encoded_bytes,
                             &failure, &error) == YVEX_OK,
                         "selected exact scalar must encode");
        YVEX_TEST_ASSERT(yvex_quant_decode_block(
                             cases[index].qtype, encoded, encoded_bytes,
                             &production, 1u, &failure, &error) == YVEX_OK &&
                             quant_reference_block(
                                 cases[index].qtype, encoded, &reference) &&
                             quant_float_bits_equal(production, reference),
                         "selected exact scalar must match independent decoding");
    }
    return 0;
}

/* Verifies the one canonical capability answer for all pinned identities. */
static int quant_test_registry(void)
{
    unsigned int qtype;

    YVEX_TEST_ASSERT(yvex_gguf_qtype_geometry_count() ==
                         YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u,
                     "numeric registry must cover every pinned identity");
    for (qtype = 0u; qtype <= YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID; ++qtype) {
        const yvex_quant_numeric_capability *capability =
            yvex_quant_numeric_capability_at(qtype);
        const yvex_gguf_qtype_geometry *geometry =
            yvex_gguf_qtype_geometry_find(qtype);
        YVEX_TEST_ASSERT(capability && capability->qtype == qtype && geometry,
                         "capability registry ordinal must equal qtype ID");
        YVEX_TEST_ASSERT(capability->storage_admitted ==
                             (geometry->identity_status ==
                              YVEX_GGUF_QTYPE_IDENTITY_ADMITTED),
                         "numeric storage projection must match geometry owner");
        if (capability->encoder_available) {
            YVEX_TEST_ASSERT(capability->reference_decoder_available &&
                                 capability->deterministic_encoding &&
                                 capability->numeric_contract_version ==
                                     YVEX_QUANT_NUMERIC_CONTRACT_VERSION,
                             "implemented encoders require deterministic reference decode");
        }
    }
    YVEX_TEST_ASSERT(yvex_quant_numeric_capability_at(
                         YVEX_GGUF_QTYPE_Q4_2_REMOVED)->refusal ==
                         YVEX_QUANT_REFUSAL_REMOVED_IDENTITY,
                     "removed qtype must have a typed refusal");
    YVEX_TEST_ASSERT(yvex_quant_numeric_capability_at(
                         YVEX_GGUF_QTYPE_NVFP4_OUTSIDE_BASELINE)->refusal ==
                         YVEX_QUANT_REFUSAL_OUTSIDE_PINNED_BASELINE,
                     "post-baseline qtype must have a typed refusal");
    YVEX_TEST_ASSERT(yvex_quant_numeric_capability_by_name("Q8_0") ==
                         yvex_quant_numeric_capability_at(
                             YVEX_GGUF_QTYPE_Q8_0),
                     "name lookup must project canonical geometry identity");
    return 0;
}

/* Proves pinned packing bytes and malformed-scale refusal independently. */
static int quant_test_golden_blocks(void)
{
    float q8_source[YVEX_QUANT_Q8_0_ELEMENTS] = {0};
    float mxfp4_source[YVEX_QUANT_MXFP4_ELEMENTS] = {0};
    float q2_source[YVEX_QUANT_Q2_K_ELEMENTS] = {0};
    float decoded[YVEX_QUANT_Q2_K_ELEMENTS];
    unsigned char encoded[YVEX_QUANT_Q2_K_BYTES];
    yvex_quant_failure failure;
    yvex_error error;
    size_t encoded_bytes = 0u;
    unsigned int index;

    q8_source[0] = 127.0f;
    q8_source[1] = -127.0f;
    q8_source[2] = 0.5f;
    q8_source[3] = -0.5f;
    YVEX_TEST_ASSERT(yvex_quant_encode_block(
                         YVEX_GGUF_QTYPE_Q8_0, q8_source,
                         YVEX_QUANT_Q8_0_ELEMENTS, encoded, sizeof(encoded),
                         &encoded_bytes, &failure, &error) == YVEX_OK &&
                         encoded_bytes == YVEX_QUANT_Q8_0_BYTES &&
                         encoded[0] == 0x00u && encoded[1] == 0x3cu &&
                         encoded[2] == 0x7fu && encoded[3] == 0x81u &&
                         encoded[4] == 0x01u && encoded[5] == 0xffu,
                     "Q8_0 golden block must store scale and deterministic halfway codes");

    memset(encoded, 0, sizeof(encoded));
    encoded[0] = 127u;
    memset(encoded + 1u, 0x21, 16u);
    YVEX_TEST_ASSERT(yvex_quant_decode_block(
                         YVEX_GGUF_QTYPE_MXFP4, encoded,
                         YVEX_QUANT_MXFP4_BYTES, decoded,
                         YVEX_QUANT_MXFP4_ELEMENTS, &failure,
                         &error) == YVEX_OK,
                     "MXFP4 golden block must decode");
    for (index = 0u; index < 16u; ++index)
        YVEX_TEST_ASSERT(decoded[index] == 0.5f &&
                             decoded[index + 16u] == 1.0f,
                         "MXFP4 must use split low/high halves and E8M0/2");
    mxfp4_source[0] = 1.9f;
    YVEX_TEST_ASSERT(yvex_quant_encode_block(
                         YVEX_GGUF_QTYPE_MXFP4, mxfp4_source,
                         YVEX_QUANT_MXFP4_ELEMENTS, encoded, sizeof(encoded),
                         &encoded_bytes, &failure, &error) == YVEX_OK &&
                         encoded[0] == 126u,
                     "MXFP4 must choose the smallest covering power-of-two scale");
    memset(encoded, 0x77, YVEX_QUANT_MXFP4_BYTES);
    encoded[0] = 254u;
    YVEX_TEST_ASSERT(yvex_quant_decode_block(
                         YVEX_GGUF_QTYPE_MXFP4, encoded,
                         YVEX_QUANT_MXFP4_BYTES, decoded,
                         YVEX_QUANT_MXFP4_ELEMENTS, &failure,
                         &error) == YVEX_ERR_FORMAT &&
                         failure.code == YVEX_QUANT_FAILURE_MXFP4_BLOCK,
                     "MXFP4 must refuse a non-finite reconstruction");

    memset(encoded, 0, sizeof(encoded));
    memset(encoded, 0x01, 16u);
    memset(encoded + 16u, 0xe4, 64u);
    encoded[80] = 0x00u;
    encoded[81] = 0x3cu;
    YVEX_TEST_ASSERT(yvex_quant_decode_block(
                         YVEX_GGUF_QTYPE_Q2_K, encoded,
                         YVEX_QUANT_Q2_K_BYTES, decoded,
                         YVEX_QUANT_Q2_K_ELEMENTS, &failure,
                         &error) == YVEX_OK,
                     "Q2_K golden packing block must decode");
    for (index = 0u; index < YVEX_QUANT_Q2_K_ELEMENTS; ++index)
        YVEX_TEST_ASSERT(decoded[index] == (float)((index % 128u) / 32u),
                         "Q2_K two-bit planes must map to ordered sub-blocks");

    memset(encoded, 0, sizeof(encoded));
    encoded[1] = 0x7cu;
    YVEX_TEST_ASSERT(yvex_quant_decode_block(
                         YVEX_GGUF_QTYPE_Q8_0, encoded,
                         YVEX_QUANT_Q8_0_BYTES, decoded,
                         YVEX_QUANT_Q8_0_ELEMENTS, &failure,
                         &error) == YVEX_ERR_FORMAT &&
                         failure.code == YVEX_QUANT_FAILURE_Q8_0_BLOCK,
                     "Q8_0 non-finite encoded scale must refuse");
    memset(encoded, 0, sizeof(encoded));
    encoded[81] = 0x7cu;
    YVEX_TEST_ASSERT(yvex_quant_decode_block(
                         YVEX_GGUF_QTYPE_Q2_K, encoded,
                         YVEX_QUANT_Q2_K_BYTES, decoded,
                         YVEX_QUANT_Q2_K_ELEMENTS, &failure,
                         &error) == YVEX_ERR_FORMAT &&
                         failure.code == YVEX_QUANT_FAILURE_Q2_K_BLOCK,
                     "Q2_K non-finite encoded scale must refuse");

    for (index = 0u; index < YVEX_QUANT_Q8_0_ELEMENTS; ++index)
        q8_source[index] = FLT_MAX;
    YVEX_TEST_ASSERT(yvex_quant_encode_block(
                         YVEX_GGUF_QTYPE_Q8_0, q8_source,
                         YVEX_QUANT_Q8_0_ELEMENTS, encoded, sizeof(encoded),
                         &encoded_bytes, &failure, &error) == YVEX_ERR_BOUNDS &&
                         failure.code == YVEX_QUANT_FAILURE_Q8_0_BLOCK,
                     "Q8_0 must refuse an unrepresentable finite scale");
    for (index = 0u; index < YVEX_QUANT_Q2_K_ELEMENTS; ++index)
        q2_source[index] = FLT_MAX;
    YVEX_TEST_ASSERT(yvex_quant_encode_block(
                         YVEX_GGUF_QTYPE_Q2_K, q2_source,
                         YVEX_QUANT_Q2_K_ELEMENTS, encoded, sizeof(encoded),
                         &encoded_bytes, &failure, &error) == YVEX_ERR_BOUNDS &&
                         failure.code == YVEX_QUANT_FAILURE_Q2_K_BLOCK,
                     "Q2_K must refuse unrepresentable finite affine scales");
    return 0;
}

static int quant_test_weighted_blocks(void)
{
    float source[YVEX_QUANT_IQ2_XXS_ELEMENTS];
    float weights[YVEX_QUANT_IQ2_XXS_ELEMENTS];
    float changed_weights[YVEX_QUANT_IQ2_XXS_ELEMENTS];
    float decoded[YVEX_QUANT_IQ2_XXS_ELEMENTS];
    float reference[YVEX_QUANT_IQ2_XXS_ELEMENTS];
    float vector[YVEX_QUANT_IQ2_XXS_ELEMENTS];
    unsigned char encoded[YVEX_QUANT_Q2_K_BYTES];
    unsigned char repeated[YVEX_QUANT_Q2_K_BYTES];
    size_t encoded_bytes = 0u;
    size_t repeated_bytes = 0u;
    yvex_quant_failure failure;
    yvex_error error;
    float cpu_dot = 0.0f;
    double reference_dot = 0.0;
    unsigned int index;

    quant_fill(source, YVEX_QUANT_IQ2_XXS_ELEMENTS, 0.04125f);
    for (index = 0u; index < YVEX_QUANT_IQ2_XXS_ELEMENTS; ++index) {
        weights[index] = 0.5f + (float)(index % 17u) * 0.125f;
        changed_weights[index] = weights[index];
        vector[index] = (float)((int)(index % 19u) - 9) / 11.0f;
    }
    changed_weights[7] = 1000.0f;
    yvex_error_clear(&error);
    YVEX_TEST_ASSERT(yvex_quant_encode_block_weighted(
                         YVEX_GGUF_QTYPE_IQ2_XXS, source, weights,
                         YVEX_QUANT_IQ2_XXS_ELEMENTS, encoded, sizeof(encoded), &encoded_bytes,
                         &failure, &error) == YVEX_OK &&
                         encoded_bytes == YVEX_QUANT_IQ2_XXS_BYTES,
                     "IQ2_XXS weighted block must encode");
    YVEX_TEST_ASSERT(yvex_quant_encode_block_weighted(
                         YVEX_GGUF_QTYPE_IQ2_XXS, source, weights,
                         YVEX_QUANT_IQ2_XXS_ELEMENTS, repeated, sizeof(repeated), &repeated_bytes,
                         &failure, &error) == YVEX_OK &&
                         repeated_bytes == encoded_bytes &&
                         memcmp(encoded, repeated, encoded_bytes) == 0,
                     "IQ2_XXS weighted encoding must be deterministic");
    YVEX_TEST_ASSERT(yvex_quant_decode_block(
                         YVEX_GGUF_QTYPE_IQ2_XXS, encoded, encoded_bytes, decoded,
                         YVEX_QUANT_IQ2_XXS_ELEMENTS, &failure, &error) == YVEX_OK &&
                         quant_reference_block(YVEX_GGUF_QTYPE_IQ2_XXS, encoded, reference),
                     "IQ2_XXS production and independent decoders must execute");
    for (index = 0u; index < YVEX_QUANT_IQ2_XXS_ELEMENTS; ++index)
        YVEX_TEST_ASSERT(quant_float_bits_equal(decoded[index], reference[index]),
                         "IQ2_XXS decoder must match independent reference");
    YVEX_TEST_ASSERT(yvex_quant_cpu_dot(
                         YVEX_GGUF_QTYPE_IQ2_XXS, encoded, encoded_bytes, vector,
                         YVEX_QUANT_IQ2_XXS_ELEMENTS, &cpu_dot, &failure, &error) == YVEX_OK,
                     "IQ2_XXS direct encoded CPU dot must execute");
    for (index = 0u; index < YVEX_QUANT_IQ2_XXS_ELEMENTS; ++index)
        reference_dot += (double)reference[index] * vector[index];
    YVEX_TEST_ASSERT(fabs((double)cpu_dot - reference_dot) < 1e-3,
                     "IQ2_XXS CPU dot must match independent decode");
    YVEX_TEST_ASSERT(yvex_quant_encode_block_weighted(
                         YVEX_GGUF_QTYPE_IQ2_XXS, source, changed_weights,
                         YVEX_QUANT_IQ2_XXS_ELEMENTS, repeated, sizeof(repeated), &repeated_bytes,
                         &failure, &error) == YVEX_OK &&
                         memcmp(encoded, repeated, encoded_bytes) != 0,
                     "imatrix mutation must influence IQ2_XXS bytes");
    YVEX_TEST_ASSERT(yvex_quant_encode_block(
                         YVEX_GGUF_QTYPE_IQ2_XXS, source, YVEX_QUANT_IQ2_XXS_ELEMENTS,
                         encoded, sizeof(encoded), &encoded_bytes, &failure, &error) ==
                         YVEX_ERR_INVALID_ARG &&
                         failure.code == YVEX_QUANT_FAILURE_CALIBRATION_REQUIRED,
                     "IQ2_XXS uncalibrated encoding must refuse");
    weights[3] = -1.0f;
    YVEX_TEST_ASSERT(yvex_quant_encode_block_weighted(
                         YVEX_GGUF_QTYPE_Q2_K, source, weights,
                         YVEX_QUANT_Q2_K_ELEMENTS, encoded, sizeof(encoded), &encoded_bytes,
                         &failure, &error) == YVEX_ERR_FORMAT &&
                         failure.code == YVEX_QUANT_FAILURE_CALIBRATION_IDENTITY,
                     "weighted Q2_K must refuse negative calibration");
    weights[3] = 1.0f;
    YVEX_TEST_ASSERT(yvex_quant_encode_block_weighted(
                         YVEX_GGUF_QTYPE_Q2_K, source, weights,
                         YVEX_QUANT_Q2_K_ELEMENTS, encoded, sizeof(encoded), &encoded_bytes,
                         &failure, &error) == YVEX_OK && encoded_bytes == YVEX_QUANT_Q2_K_BYTES,
                     "weighted Q2_K must encode");
    return 0;
}

int yvex_test_quant_numeric(void)
{
    float zero_q8[YVEX_QUANT_Q8_0_ELEMENTS] = {0};
    float decoded[YVEX_QUANT_Q8_0_ELEMENTS];
    unsigned char encoded[YVEX_QUANT_Q8_0_BYTES];
    size_t encoded_bytes = 0u;
    yvex_quant_failure failure;
    yvex_error error;
    unsigned int index;

    if (quant_test_f16() != 0 || quant_test_bf16() != 0 ||
        quant_test_source_formats() != 0 || quant_test_registry() != 0 ||
        quant_test_golden_blocks() != 0 ||
        quant_test_weighted_blocks() != 0 ||
        quant_test_exact_scalar_blocks() != 0 ||
        quant_test_block(YVEX_GGUF_QTYPE_Q8_0,
                         YVEX_QUANT_Q8_0_ELEMENTS,
                         YVEX_QUANT_Q8_0_BYTES, 0.02) != 0 ||
        quant_test_block(YVEX_GGUF_QTYPE_MXFP4,
                         YVEX_QUANT_MXFP4_ELEMENTS,
                         YVEX_QUANT_MXFP4_BYTES, 0.9) != 0 ||
        quant_test_block(YVEX_GGUF_QTYPE_Q2_K,
                         YVEX_QUANT_Q2_K_ELEMENTS,
                         YVEX_QUANT_Q2_K_BYTES, 1.5) != 0)
        return 1;
    yvex_error_clear(&error);
    YVEX_TEST_ASSERT(yvex_quant_encode_block(
                         YVEX_GGUF_QTYPE_Q8_0, zero_q8,
                         YVEX_QUANT_Q8_0_ELEMENTS, encoded, sizeof(encoded),
                         &encoded_bytes, &failure, &error) == YVEX_OK &&
                         encoded_bytes == sizeof(encoded),
                     "Q8_0 zero block must encode");
    for (index = 0u; index < encoded_bytes; ++index)
        YVEX_TEST_ASSERT(encoded[index] == 0u,
                         "Q8_0 zero block bytes must be canonical zero");
    YVEX_TEST_ASSERT(yvex_quant_decode_block(
                         YVEX_GGUF_QTYPE_Q8_0, encoded, encoded_bytes,
                         decoded, YVEX_QUANT_Q8_0_ELEMENTS, &failure,
                         &error) == YVEX_OK,
                     "Q8_0 zero block must decode");
    for (index = 0u; index < YVEX_QUANT_Q8_0_ELEMENTS; ++index)
        YVEX_TEST_ASSERT(quant_float_bits_equal(decoded[index], 0.0f),
                         "Q8_0 zero block must reconstruct exact zero");
    return 0;
}
