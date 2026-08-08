/*
 * This translation unit does not include the production attention-private header and does not
 * call production attention numeric algorithms. Independent scalar comparison is test evidence,
 * not runtime support.
 */
#include "tests/reference/deepseek_attention.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/qtype.h>

#define REF_PI 3.14159265358979323846264338327950288

typedef struct {
    float score;
    unsigned long long position;
    unsigned long long index;
} ref_ranked_candidate;

const char *yvex_test_attention_status_name(yvex_attention_status status)
{
    switch (status) {
    case YVEX_DEEPSEEK_ATTENTION_STATUS_REFUSED: return "refused";
    case YVEX_DEEPSEEK_ATTENTION_STATUS_PLANNED: return "planned";
    case YVEX_DEEPSEEK_ATTENTION_STATUS_EXECUTION_UNSUPPORTED:
        return "execution-unsupported";
    case YVEX_DEEPSEEK_ATTENTION_STATUS_EXECUTION_READY:
        return "execution-ready";
    default: return "unknown";
    }
}

const char *yvex_test_attention_failure_name(yvex_attention_failure_code code)
{
    switch (code) {
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_NONE: return "none";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT:
        return "invalid-argument";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_ARCHITECTURE: return "architecture";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_MATERIALIZATION:
        return "materialization";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_DESCRIPTOR: return "descriptor";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_MISSING_BINDING:
        return "missing-binding";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_QTYPE: return "qtype";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION: return "dimension";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY: return "history";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_EXECUTION_UNSUPPORTED:
        return "execution-unsupported";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_READ: return "read";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC: return "numeric";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA: return "state-delta";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION: return "allocation";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH: return "scratch";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_CANCELLED: return "cancelled";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_CLEANUP: return "cleanup";
    case YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND: return "backend";
    default: return "unknown";
    }
}

static int ref_mul(unsigned long long a,
                   unsigned long long b,
                   unsigned long long *out)
{
    if (!out || (a && b > ULLONG_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static void *ref_alloc(unsigned long long count, size_t width)
{
    if (!count || !width || count > SIZE_MAX / width) return NULL;
    return calloc((size_t)count, width);
}

static void ref_reason(char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP],
                       const char *message)
{
    if (!reason) return;
    (void)snprintf(reason, YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP, "%s",
                   message ? message : "reference failure");
}

static unsigned short ref_u16(const unsigned char *bytes)
{
    return (unsigned short)((unsigned short)bytes[0] |
                            ((unsigned short)bytes[1] << 8));
}

static unsigned int ref_u32(const unsigned char *bytes)
{
    return (unsigned int)bytes[0] |
           ((unsigned int)bytes[1] << 8) |
           ((unsigned int)bytes[2] << 16) |
           ((unsigned int)bytes[3] << 24);
}

static float ref_f16(unsigned short bits)
{
    unsigned int sign = (unsigned int)(bits >> 15);
    unsigned int exponent = ((unsigned int)bits >> 10) & 0x1fu;
    unsigned int fraction = (unsigned int)bits & 0x3ffu;
    double magnitude;

    if (exponent == 0u) {
        magnitude = fraction
            ? ldexp((double)fraction, -24)
            : 0.0;
    } else if (exponent == 0x1fu) {
        if (fraction) return NAN;
        return sign ? -INFINITY : INFINITY;
    } else {
        magnitude = ldexp(1.0 + (double)fraction / 1024.0,
                          (int)exponent - 15);
    }
    return (float)(sign ? -magnitude : magnitude);
}

static float ref_bf16(unsigned short bits)
{
    uint32_t value = (uint32_t)bits << 16;
    float out;
    memcpy(&out, &value, sizeof(out));
    return out;
}

static const yvex_runtime_tensor_binding *ref_binding(
    const yvex_runtime_descriptor *descriptor,
    yvex_tensor_role role,
    unsigned long long layer)
{
    return yvex_runtime_descriptor_find_role(
        descriptor, role, YVEX_TENSOR_SCOPE_MAIN_LAYER, layer,
        YVEX_DEEPSEEK_GGUF_NO_INDEX);
}

static int ref_row_bytes(const yvex_materialized_tensor_binding *binding,
                         unsigned long long *out)
{
    unsigned long long blocks;

    if (!binding || !out || !binding->row_width || !binding->block_size ||
        !binding->bytes_per_block ||
        binding->row_width % binding->block_size != 0ull)
        return 0;
    blocks = binding->row_width / binding->block_size;
    return ref_mul(blocks, binding->bytes_per_block, out) && *out > 0ull;
}

static int ref_decode_block(unsigned int qtype,
                            const unsigned char *encoded,
                            unsigned long long bytes,
                            float *out,
                            unsigned long long elements)
{
    unsigned long long lane;

    if (!encoded || !out) return 0;
    if (qtype == YVEX_GGUF_QTYPE_F32 && bytes == 4ull && elements == 1ull) {
        uint32_t bits = ref_u32(encoded);
        memcpy(out, &bits, sizeof(*out));
        return isfinite(*out);
    }
    if (qtype == YVEX_GGUF_QTYPE_F16 && bytes == 2ull && elements == 1ull) {
        *out = ref_f16(ref_u16(encoded));
        return isfinite(*out);
    }
    if (qtype == YVEX_GGUF_QTYPE_BF16 && bytes == 2ull && elements == 1ull) {
        *out = ref_bf16(ref_u16(encoded));
        return isfinite(*out);
    }
    if (qtype == YVEX_GGUF_QTYPE_Q8_0 && bytes == 34ull &&
        elements == 32ull) {
        float scale = ref_f16(ref_u16(encoded));
        if (!isfinite(scale) || scale < 0.0f) return 0;
        for (lane = 0ull; lane < 32ull; ++lane) {
            int value = encoded[2ull + lane] <= 127u
                ? (int)encoded[2ull + lane]
                : (int)encoded[2ull + lane] - 256;
            out[lane] = scale * (float)value;
        }
        return 1;
    }
    return 0;
}

static int ref_decode_row(yvex_materialization_session *session,
                          const yvex_runtime_tensor_binding *runtime,
                          unsigned long long row,
                          float *out,
                          char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP])
{
    const yvex_materialized_tensor_binding *binding;
    unsigned long long row_bytes;
    unsigned long long block;
    unsigned long long blocks;
    unsigned char *encoded;
    yvex_error error;

    if (!session || !runtime || !(binding = runtime->binding) || !out ||
        row >= binding->row_count || !ref_row_bytes(binding, &row_bytes) ||
        row_bytes > SIZE_MAX) {
        ref_reason(reason, "reference row geometry is invalid");
        return 0;
    }
    encoded = (unsigned char *)malloc((size_t)row_bytes);
    if (!encoded) {
        ref_reason(reason, "reference row allocation failed");
        return 0;
    }
    yvex_error_clear(&error);
    if (yvex_materialization_session_read(
            session, binding, row * row_bytes, encoded, (size_t)row_bytes,
            NULL, &error) != YVEX_OK) {
        free(encoded);
        ref_reason(reason, "reference materialized row read failed");
        return 0;
    }
    blocks = binding->row_width / binding->block_size;
    for (block = 0ull; block < blocks; ++block) {
        if (!ref_decode_block(
                binding->qtype,
                encoded + block * binding->bytes_per_block,
                binding->bytes_per_block,
                out + block * binding->block_size,
                binding->block_size)) {
            free(encoded);
            ref_reason(reason, "reference qtype decode failed");
            return 0;
        }
    }
    free(encoded);
    return 1;
}

static int ref_matrix_batch(
    yvex_materialization_session *session,
    const yvex_runtime_tensor_binding *runtime,
    unsigned long long row_start,
    unsigned long long rows,
    const float *input,
    unsigned long long token_count,
    unsigned long long input_stride,
    float *output,
    unsigned long long output_stride,
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP])
{
    const yvex_materialized_tensor_binding *binding;
    float *decoded;
    unsigned long long row;

    if (!session || !runtime || !(binding = runtime->binding) || !input ||
        !output || !token_count || input_stride < binding->row_width ||
        output_stride < rows || row_start > binding->row_count ||
        rows > binding->row_count - row_start) {
        ref_reason(reason, "reference matrix geometry is invalid");
        return 0;
    }
    decoded = (float *)ref_alloc(binding->row_width, sizeof(*decoded));
    if (!decoded) {
        ref_reason(reason, "reference matrix scratch allocation failed");
        return 0;
    }
    for (row = 0ull; row < rows; ++row) {
        unsigned long long token;
        if (!ref_decode_row(session, runtime, row_start + row, decoded,
                            reason)) {
            free(decoded);
            return 0;
        }
        for (token = 0ull; token < token_count; ++token) {
            unsigned long long lane;
            double sum = 0.0;
            const float *vector = input + token * input_stride;
            for (lane = 0ull; lane < binding->row_width; ++lane)
                sum += (double)decoded[lane] * (double)vector[lane];
            output[token * output_stride + row] = (float)sum;
            if (!isfinite(output[token * output_stride + row])) {
                free(decoded);
                ref_reason(reason, "reference matrix result is non-finite");
                return 0;
            }
        }
    }
    free(decoded);
    return 1;
}

static int ref_rms(float *values,
                   unsigned long long count,
                   const float *weights,
                   double epsilon)
{
    unsigned long long lane;
    double squares = 0.0;
    double inverse;

    if (!values || !count || !(epsilon > 0.0) || !isfinite(epsilon))
        return 0;
    for (lane = 0ull; lane < count; ++lane) {
        if (!isfinite(values[lane]) ||
            (weights && !isfinite(weights[lane])))
            return 0;
        squares += (double)values[lane] * (double)values[lane];
    }
    inverse = 1.0 / sqrt(squares / (double)count + epsilon);
    if (!isfinite(inverse)) return 0;
    for (lane = 0ull; lane < count; ++lane)
        values[lane] = (float)((double)values[lane] * inverse *
                               (weights ? (double)weights[lane] : 1.0));
    return 1;
}

static double ref_yarn_frequency(
    const yvex_attention_position_policy *position,
    unsigned long long pair,
    unsigned long long rope_dimensions)
{
    double base;

    if (!position || rope_dimensions < 2ull || position->theta <= 1ull)
        return 0.0;
    base = 1.0 / pow((double)position->theta,
                     (double)(pair * 2ull) /
                         (double)rope_dimensions);
    if (position->original_context && position->scaling_factor) {
        double denominator = 2.0 * log((double)position->theta);
        double lower = floor(
            (double)rope_dimensions *
            log((double)position->original_context /
                ((double)position->beta_fast * 2.0 * REF_PI)) /
            denominator);
        double upper = ceil(
            (double)rope_dimensions *
            log((double)position->original_context /
                ((double)position->beta_slow * 2.0 * REF_PI)) /
            denominator);
        double ramp;
        double smooth;
        if (lower < 0.0) lower = 0.0;
        if (upper > (double)rope_dimensions - 1.0)
            upper = (double)rope_dimensions - 1.0;
        if (lower == upper) upper += 0.001;
        ramp = ((double)pair - lower) / (upper - lower);
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        smooth = 1.0 - ramp;
        base = base / (double)position->scaling_factor * (1.0 - smooth) +
               base * smooth;
    }
    return base;
}

static int ref_rope(float *values,
                    unsigned long long width,
                    unsigned long long rope_dimensions,
                    unsigned long long position_index,
                    const yvex_attention_position_policy *position,
                    int inverse)
{
    unsigned long long lane;
    unsigned long long start;

    if (!values || rope_dimensions < 2ull || rope_dimensions > width)
        return 0;
    if (rope_dimensions & 1ull) rope_dimensions--;
    start = width - rope_dimensions;
    for (lane = 0ull; lane < rope_dimensions; lane += 2ull) {
        double angle = (double)position_index *
                       ref_yarn_frequency(position, lane / 2ull,
                                          rope_dimensions);
        double sine = sin(angle);
        double cosine = cos(angle);
        double first = values[start + lane];
        double second = values[start + lane + 1ull];
        if (inverse) sine = -sine;
        values[start + lane] = (float)(first * cosine - second * sine);
        values[start + lane + 1ull] =
            (float)(first * sine + second * cosine);
        if (!isfinite(values[start + lane]) ||
            !isfinite(values[start + lane + 1ull]))
            return 0;
    }
    return 1;
}

static float ref_power_two_ceil(float value)
{
    int exponent;
    float fraction;

    if (!(value > 0.0f) || !isfinite(value)) return 0.0f;
    fraction = frexpf(value, &exponent);
    if (fraction > 0.5f) exponent++;
    return ldexpf(1.0f, exponent - 1);
}

static unsigned char ref_ue8m0_encode(float scale)
{
    int exponent;
    float fraction;

    if (!(scale > 0.0f) || !isfinite(scale)) return 0xffu;
    fraction = frexpf(scale, &exponent);
    if (fraction > 0.5f) exponent++;
    exponent += 126;
    if (exponent < 0) return 0u;
    if (exponent > 254) return 254u;
    return (unsigned char)exponent;
}

static float ref_ue8m0_decode(unsigned char code)
{
    unsigned int bits;
    float value;

    if (code == 0xffu) return NAN;
    bits = code ? (unsigned int)code << 23 : 0x00400000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float ref_fp8_decode(unsigned char code)
{
    unsigned int negative = code & 0x80u;
    unsigned int exponent = (code >> 3u) & 0x0fu;
    unsigned int fraction = code & 0x07u;
    float magnitude;

    if ((code & 0x7fu) == 0u) return negative ? -0.0f : 0.0f;
    if ((code & 0x7fu) == 0x7fu) return NAN;
    magnitude = exponent
        ? ldexpf(1.0f + (float)fraction / 8.0f, (int)exponent - 7)
        : ldexpf((float)fraction / 8.0f, -6);
    return negative ? -magnitude : magnitude;
}

static unsigned char ref_fp8_encode(float value)
{
    unsigned int code;
    unsigned char best = 0u;
    float magnitude = fabsf(value);
    float best_error = INFINITY;
    int negative = signbit(value);

    if (!isfinite(value)) return negative ? 0xffu : 0x7fu;
    if (magnitude > 448.0f) magnitude = 448.0f;
    for (code = 0u; code < 0x7fu; ++code) {
        float decoded = ref_fp8_decode((unsigned char)code);
        float error = fabsf(decoded - magnitude);
        if (error < best_error ||
            (error == best_error && !(code & 1u) && (best & 1u))) {
            best_error = error;
            best = (unsigned char)code;
        }
    }
    return negative ? (unsigned char)(best | 0x80u) : best;
}

static float ref_fp4_decode(unsigned char code)
{
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    float value = values[code & 7u];
    return (code & 8u) ? -value : value;
}

static unsigned char ref_fp4_encode(float value)
{
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    unsigned int code;
    unsigned int best = 0u;
    float magnitude = fabsf(value);
    float error;

    if (isnan(value)) return signbit(value) ? 8u : 0u;
    if (magnitude > 6.0f) magnitude = 6.0f;
    error = fabsf(magnitude - values[0]);

    for (code = 1u; code < 8u; ++code) {
        float next = fabsf(magnitude - values[code]);
        if (next < error ||
            (next == error && !(code & 1u) && (best & 1u))) {
            best = code;
            error = next;
        }
    }
    if (signbit(value)) best |= 8u;
    return (unsigned char)best;
}

static unsigned short ref_bf16_encode(float value)
{
    unsigned int bits;
    unsigned int upper;
    unsigned int lower;

    memcpy(&bits, &value, sizeof(bits));
    upper = bits >> 16;
    lower = bits & 0xffffu;
    if ((bits & 0x7f800000u) == 0x7f800000u &&
        (bits & 0x007fffffu) != 0u)
        return (unsigned short)(upper | 0x0040u);
    if (lower > 0x8000u || (lower == 0x8000u && (upper & 1u)))
        upper++;
    return (unsigned short)upper;
}

static float ref_bf16_decode(unsigned short encoded)
{
    unsigned int bits = (unsigned int)encoded << 16;
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int ref_compute_round(yvex_attention_compute_contract contract,
                             float *values,
                             unsigned long long count)
{
    unsigned long long lane;

    if (contract != YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1 ||
        !values || !count)
        return 0;
    for (lane = 0ull; lane < count; ++lane) {
        if (!isfinite(values[lane])) return 0;
        values[lane] = ref_bf16_decode(ref_bf16_encode(values[lane]));
        if (!isfinite(values[lane])) return 0;
    }
    return 1;
}

static double ref_mhc_sigmoid(double value)
{
    if (value >= 0.0) {
        double inverse = exp(-value);
        return 1.0 / (1.0 + inverse);
    }
    {
        double direct = exp(value);
        return direct / (1.0 + direct);
    }
}

static int ref_mhc_sinkhorn(float *matrix, unsigned long long streams,
                            unsigned long long iterations, double epsilon)
{
    unsigned long long row, column, iteration;

    for (row = 0ull; row < streams; ++row) {
        double maximum = -INFINITY;
        double total = 0.0;
        for (column = 0ull; column < streams; ++column)
            if (matrix[row * streams + column] > maximum)
                maximum = matrix[row * streams + column];
        for (column = 0ull; column < streams; ++column) {
            double value = exp((double)matrix[row * streams + column] - maximum);
            matrix[row * streams + column] = (float)value;
            total += value;
        }
        if (!isfinite(total) || total <= 0.0) return 0;
        for (column = 0ull; column < streams; ++column)
            matrix[row * streams + column] =
                (float)((double)matrix[row * streams + column] / total + epsilon);
    }
    for (iteration = 0ull; iteration < iterations; ++iteration) {
        if (iteration) {
            for (row = 0ull; row < streams; ++row) {
                double total = 0.0;
                for (column = 0ull; column < streams; ++column)
                    total += matrix[row * streams + column];
                for (column = 0ull; column < streams; ++column)
                    matrix[row * streams + column] =
                        (float)((double)matrix[row * streams + column] / (total + epsilon));
            }
        }
        for (column = 0ull; column < streams; ++column) {
            double total = 0.0;
            for (row = 0ull; row < streams; ++row)
                total += matrix[row * streams + column];
            for (row = 0ull; row < streams; ++row)
                matrix[row * streams + column] =
                    (float)((double)matrix[row * streams + column] / (total + epsilon));
        }
    }
    return 1;
}

int yvex_test_attention_reference_envelope(
    const yvex_test_attention_envelope_case *test_case)
{
    const yvex_attention_layer_plan *layer = test_case ? test_case->layer : NULL;
    float *canonical_residual;
    unsigned long long token, stream, lane;
    int accepted = 0;

    if (!test_case || !layer || !test_case->residual || !test_case->linear_mixes ||
        !test_case->scale || !test_case->base || !test_case->norm_weights ||
        !test_case->core_output || !test_case->core_input || !test_case->post ||
        !test_case->combination || !test_case->envelope_output || !test_case->token_count ||
        test_case->residual_stride < layer->residual_expanded_width ||
        test_case->mix_stride < layer->mhc_mixing_rows ||
        test_case->core_stride < layer->residual_stream_width ||
        test_case->core_input_stride < layer->residual_stream_width ||
        test_case->post_stride < layer->residual_stream_count ||
        test_case->combination_stride < layer->residual_stream_count *
                                            layer->residual_stream_count ||
        test_case->envelope_stride < layer->residual_expanded_width)
        return 0;
    canonical_residual = (float *)calloc((size_t)layer->residual_expanded_width,
                                         sizeof(*canonical_residual));
    if (!canonical_residual) return 0;
    for (token = 0ull; token < test_case->token_count; ++token) {
        const float *raw_residual = test_case->residual + token * test_case->residual_stride;
        const float *residual = canonical_residual;
        const float *mix = test_case->linear_mixes + token * test_case->mix_stride;
        const float *core = test_case->core_output + token * test_case->core_stride;
        float *input = test_case->core_input + token * test_case->core_input_stride;
        float *post = test_case->post + token * test_case->post_stride;
        float *comb = test_case->combination + token * test_case->combination_stride;
        float *output = test_case->envelope_output + token * test_case->envelope_stride;
        double squares = 0.0;
        double inverse;

        memcpy(canonical_residual, raw_residual,
               (size_t)layer->residual_expanded_width * sizeof(*canonical_residual));
        if (!ref_compute_round(layer->compute_contract, canonical_residual,
                               layer->residual_expanded_width))
            goto done;
        memset(input, 0, (size_t)layer->residual_stream_width * sizeof(*input));
        for (lane = 0ull; lane < layer->residual_expanded_width; ++lane)
            squares += (double)residual[lane] * (double)residual[lane];
        inverse = 1.0 / sqrt(squares / (double)layer->residual_expanded_width +
                             layer->rms_norm_epsilon);
        for (stream = 0ull; stream < layer->residual_stream_count; ++stream) {
            unsigned long long target;
            double pre = ref_mhc_sigmoid((double)mix[stream] * inverse *
                                         (double)test_case->scale[0] +
                                         (double)test_case->base[stream]) + layer->mhc_epsilon;
            post[stream] = (float)(layer->mhc_residual_post_multiplier * ref_mhc_sigmoid(
                (double)mix[layer->residual_stream_count + stream] * inverse *
                    (double)test_case->scale[1] +
                (double)test_case->base[layer->residual_stream_count + stream]));
            for (target = 0ull; target < layer->residual_stream_count; ++target) {
                unsigned long long index = 2ull * layer->residual_stream_count +
                    stream * layer->residual_stream_count + target;
                comb[stream * layer->residual_stream_count + target] =
                    (float)((double)mix[index] * inverse * (double)test_case->scale[2] +
                            (double)test_case->base[index]);
            }
            for (lane = 0ull; lane < layer->residual_stream_width; ++lane)
                input[lane] += (float)(pre * residual[stream * layer->residual_stream_width + lane]);
        }
        if (!ref_mhc_sinkhorn(comb, layer->residual_stream_count,
                              layer->mhc_sinkhorn_iterations, layer->mhc_epsilon) ||
            !ref_compute_round(layer->compute_contract, input, layer->residual_stream_width) ||
            !ref_rms(input, layer->residual_stream_width, test_case->norm_weights,
                     layer->rms_norm_epsilon) ||
            !ref_compute_round(layer->compute_contract, input, layer->residual_stream_width))
            goto done;
        for (stream = 0ull; stream < layer->residual_stream_count; ++stream) {
            for (lane = 0ull; lane < layer->residual_stream_width; ++lane) {
                unsigned long long source;
                double value = (double)post[stream] * (double)core[lane];
                for (source = 0ull; source < layer->residual_stream_count; ++source)
                    value += (double)comb[source * layer->residual_stream_count + stream] *
                             (double)residual[source * layer->residual_stream_width + lane];
                output[stream * layer->residual_stream_width + lane] = (float)value;
            }
        }
        if (!ref_compute_round(layer->compute_contract, output,
                               layer->residual_expanded_width))
            goto done;
    }
    accepted = 1;
done:
    free(canonical_residual);
    return accepted;
}

int yvex_test_attention_reference_codec_selftest(void);

int yvex_test_attention_reference_codec_selftest(void)
{
    unsigned int code;
    float boundary[] = {1.00390625f, 1.01171875f, -0.0f};

    if (!ref_compute_round(YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
                           boundary, 3ull) ||
        boundary[0] != 1.0f || boundary[1] != 1.015625f ||
        !signbit(boundary[2]) ||
        ref_compute_round(YVEX_ATTENTION_COMPUTE_UNKNOWN, boundary, 3ull) ||
        ref_ue8m0_decode(0u) != ldexpf(1.0f, -127) ||
        !isnan(ref_ue8m0_decode(0xffu)) ||
        ref_ue8m0_encode(ldexpf(1.0f, -127)) != 0u ||
        ref_ue8m0_encode(ldexpf(1.0f, 127)) != 0xfeu ||
        ref_ue8m0_encode(0.0f) != 0xffu)
        return 0;
    for (code = 0u; code <= 0xffu; ++code) {
        float scale = ref_ue8m0_decode((unsigned char)code);
        if (code == 0xffu) {
            if (!isnan(scale)) return 0;
        } else if (!isfinite(scale) || !(scale > 0.0f) ||
                   ref_ue8m0_encode(scale) != (unsigned char)code) {
            return 0;
        }
    }
    for (code = 0u; code <= 0xffu; ++code) {
        float value = ref_fp8_decode((unsigned char)code);
        if ((code & 0x7fu) == 0x7fu) {
            if (!isnan(value)) return 0;
        } else if (!isfinite(value) ||
                   ref_fp8_encode(value) != (unsigned char)code) {
            return 0;
        }
    }
    for (code = 0u; code < 0x7eu; ++code) {
        float lower = ref_fp8_decode((unsigned char)code);
        float upper = ref_fp8_decode((unsigned char)(code + 1u));
        float midpoint = (lower + upper) * 0.5f;
        unsigned char expected = (unsigned char)((code & 1u)
            ? code + 1u : code);
        if (ref_fp8_encode(midpoint) != expected ||
            ref_fp8_encode(-midpoint) !=
                (unsigned char)(expected | 0x80u))
            return 0;
    }
    if (ref_fp8_encode(1.1875f) != 0x3au ||
        ref_fp8_encode(-0.0f) != 0x80u ||
        !isnan(ref_fp8_decode(0x7fu)) ||
        ref_bf16_decode(ref_bf16_encode(1.00390625f)) != 1.0f ||
        ref_bf16_decode(ref_bf16_encode(1.01171875f)) != 1.015625f)
        return 0;
    for (code = 0u; code < 16u; ++code)
        if (ref_fp4_encode(ref_fp4_decode((unsigned char)code)) != code)
            return 0;
    for (code = 0u; code < 7u; ++code) {
        float lower = ref_fp4_decode((unsigned char)code);
        float upper = ref_fp4_decode((unsigned char)(code + 1u));
        float midpoint = (lower + upper) * 0.5f;
        unsigned char expected = (unsigned char)((code & 1u)
            ? code + 1u : code);
        if (ref_fp4_encode(midpoint) != expected ||
            ref_fp4_encode(-midpoint) !=
                (unsigned char)(expected | 0x8u))
            return 0;
    }
    return ref_fp4_encode(1.75f) == 4u &&
           ref_fp4_encode(-0.0f) == 8u;
}

static int ref_fake_quant_block(float *values,
                                unsigned long long count,
                                int fp4)
{
    unsigned long long lane;
    float maximum = fp4 ? 0.0f : 1.0e-4f;
    float divisor = fp4 ? 6.0f : 448.0f;
    float scale;
    unsigned char scale_code;

    for (lane = 0ull; lane < count; ++lane) {
        float magnitude;
        if (!isfinite(values[lane])) return 0;
        magnitude = fabsf(values[lane]);
        if (magnitude > maximum) maximum = magnitude;
    }
    if (fp4 && maximum < 6.0f * ldexpf(1.0f, -126))
        maximum = 6.0f * ldexpf(1.0f, -126);
    scale_code = ref_ue8m0_encode(ref_power_two_ceil(maximum / divisor));
    scale = ref_ue8m0_decode(scale_code);
    if (!(scale > 0.0f) || !isfinite(scale)) return 0;
    for (lane = 0ull; lane < count; ++lane) {
        unsigned char code = fp4
            ? ref_fp4_encode(values[lane] / scale)
            : ref_fp8_encode(values[lane] / scale);
        values[lane] = ref_bf16_decode(ref_bf16_encode(
            (fp4 ? ref_fp4_decode(code) : ref_fp8_decode(code)) * scale));
    }
    return 1;
}

static int ref_activation(
    const yvex_attention_activation_policy *policy,
    float *values,
    unsigned long long count)
{
    unsigned long long offset;
    float *rotated = NULL;

    if (!policy || !policy->required) return 1;
    if (!values || !count || !policy->block_width ||
        count % policy->block_width != 0ull)
        return 0;
    if (policy->pre_transform ==
        YVEX_ATTENTION_TRANSFORM_DAO_FHT_V1_1_0_POST2) {
        rotated = (float *)ref_alloc(count, sizeof(*rotated));
        if (!rotated ||
            !yvex_test_attention_reference_hadamard(
                values, count, 1.0f / sqrtf((float)count), 1, rotated)) {
            free(rotated);
            return 0;
        }
        memcpy(values, rotated, (size_t)count * sizeof(*values));
        free(rotated);
    } else if (policy->pre_transform !=
               YVEX_ATTENTION_TRANSFORM_NONE) {
        return 0;
    }
    for (offset = 0ull; offset < count; offset += policy->block_width) {
        int fp4 = policy->quantization ==
            YVEX_ATTENTION_QUANT_FP4_E2M1_UE8M0_FAKE_DEQUANT;
        if (!fp4 && policy->quantization !=
            YVEX_ATTENTION_QUANT_FP8_E4M3_UE8M0_FAKE_DEQUANT)
            return 0;
        if (!ref_fake_quant_block(values + offset, policy->block_width,
                                  fp4))
            return 0;
    }
    return 1;
}

int yvex_test_attention_reference_hadamard(const float *input,
                                           unsigned long long length,
                                           float scale,
                                           int reject_nonfinite,
                                           float *output)
{
    unsigned long long padded = 1ull;
    unsigned long long row;

    if (!input || !output || !length || !isfinite(scale)) return 0;
    while (padded < length) {
        if (padded > ULLONG_MAX / 2ull) return 0;
        padded *= 2ull;
    }
    for (row = 0ull; row < length; ++row) {
        unsigned long long column;
        double sum = 0.0;
        for (column = 0ull; column < length; ++column) {
            unsigned long long parity = row & column;
            unsigned int odd = 0u;
            if (reject_nonfinite && !isfinite(input[column])) return 0;
            while (parity) {
                odd ^= (unsigned int)(parity & 1ull);
                parity >>= 1u;
            }
            sum += odd ? -(double)input[column] : (double)input[column];
        }
        output[row] = (float)(sum * (double)scale);
        if (!isfinite(output[row])) return 0;
    }
    return 1;
}

typedef struct {
    unsigned long long ratio;
    unsigned long long head_dimension;
    unsigned long long width;
    unsigned long long slots;
    unsigned long long next_position;
    unsigned long long cursor;
    unsigned long long previous_fill;
    unsigned long long current_fill;
    int overlap;
    float *kv;
    float *score;
} ref_rolling_state;

static void ref_rolling_release(ref_rolling_state *state)
{
    if (!state) return;
    free(state->kv);
    free(state->score);
    memset(state, 0, sizeof(*state));
}

static int ref_rolling_view_complete(
    const yvex_attention_layer_plan *layer,
    yvex_attention_rolling_kind kind,
    const yvex_attention_rolling_state_view *view,
    unsigned long long token_position)
{
    unsigned long long head_dimension;
    unsigned long long width;
    unsigned long long slots;
    unsigned long long kv_extent;
    unsigned long long score_extent;
    unsigned long long expected_cursor;
    unsigned long long expected_previous;
    int overlap;
    int rotated;

    if (!layer || !view || !view->present || !layer->compression_ratio)
        return 0;
    head_dimension = kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER
        ? layer->indexer_head_dimension : layer->head_dimension;
    overlap = kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER ||
        layer->attention_class == YVEX_ATTENTION_CLASS_CSA;
    rotated = kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER;
    if (!head_dimension ||
        !ref_mul(head_dimension, overlap ? 2ull : 1ull, &width) ||
        !ref_mul(layer->compression_ratio, overlap ? 2ull : 1ull, &slots) ||
        view->kv_state_stride < width || view->score_state_stride < width ||
        !ref_mul(slots, view->kv_state_stride, &kv_extent) ||
        !ref_mul(slots, view->score_state_stride, &score_extent))
        return 0;
    expected_cursor = token_position % layer->compression_ratio;
    expected_previous = overlap && token_position >= layer->compression_ratio
        ? layer->compression_ratio : 0ull;
    return view->schema_version ==
               YVEX_DEEPSEEK_ATTENTION_ROLLING_STATE_SCHEMA_V1 &&
        view->kind == kind &&
        view->attention_class == layer->attention_class &&
        view->layer_index == layer->layer_index &&
        view->next_token_position == token_position &&
        view->ratio == layer->compression_ratio &&
        view->head_dimension == head_dimension &&
        view->state_width == width && view->state_slots == slots &&
        view->cursor == expected_cursor &&
        view->current_fill == expected_cursor &&
        view->previous_fill == expected_previous &&
        view->overlap == overlap && view->rotated == rotated &&
        view->kv_state && view->score_state &&
        view->kv_state_extent >= kv_extent &&
        view->score_state_extent >= score_extent;
}

static int ref_rolling_init(
    ref_rolling_state *state,
    const yvex_attention_layer_plan *layer,
    yvex_attention_rolling_kind kind,
    const yvex_attention_rolling_state_view *view,
    unsigned long long token_position)
{
    unsigned long long extent;
    unsigned long long slot;

    if (!state || !layer || !layer->compression_ratio) return 0;
    memset(state, 0, sizeof(*state));
    state->ratio = layer->compression_ratio;
    state->head_dimension =
        kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER
            ? layer->indexer_head_dimension
            : layer->head_dimension;
    state->overlap =
        kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER ||
        layer->attention_class == YVEX_ATTENTION_CLASS_CSA;
    state->width = state->head_dimension * (state->overlap ? 2ull : 1ull);
    state->slots = state->ratio * (state->overlap ? 2ull : 1ull);
    if (!state->head_dimension || !state->width || !state->slots ||
        !ref_mul(state->slots, state->width, &extent))
        return 0;
    state->kv = (float *)ref_alloc(extent, sizeof(*state->kv));
    state->score = (float *)ref_alloc(extent, sizeof(*state->score));
    if (!state->kv || !state->score) {
        ref_rolling_release(state);
        return 0;
    }
    for (slot = 0ull; slot < extent; ++slot) state->score[slot] = -INFINITY;
    state->next_position = token_position;
    state->cursor = token_position % state->ratio;
    if (view && view->present) {
        if (!ref_rolling_view_complete(layer, kind, view, token_position))
            goto fail;
        state->cursor = view->cursor;
        state->previous_fill = view->previous_fill;
        state->current_fill = view->current_fill;
        for (slot = 0ull; slot < state->slots; ++slot) {
            memcpy(state->kv + slot * state->width,
                   view->kv_state + slot * view->kv_state_stride,
                   (size_t)state->width * sizeof(*state->kv));
            memcpy(state->score + slot * state->width,
                   view->score_state + slot * view->score_state_stride,
                   (size_t)state->width * sizeof(*state->score));
        }
    }
    return 1;

fail:
    ref_rolling_release(state);
    return 0;
}

/* Contract: independently applies one rolling compressor transition and
 * emits only after a complete ratio-sized group. */
static int ref_rolling_step(ref_rolling_state *state,
                            const float *token_kv,
                            const float *token_score,
                            const float *ape,
                            float *emission,
                            int *emitted)
{
    unsigned long long slot;
    unsigned long long lane;

    if (emitted) *emitted = 0;
    if (!state || !token_kv || !token_score || !ape || !emission ||
        !emitted || state->cursor != state->next_position % state->ratio)
        return 0;
    slot = state->overlap ? state->ratio + state->cursor : state->cursor;
    for (lane = 0ull; lane < state->width; ++lane) {
        float value = token_kv[lane];
        float score = token_score[lane] + ape[lane];
        if (!isfinite(value) || !isfinite(score)) return 0;
        state->kv[slot * state->width + lane] = value;
        state->score[slot * state->width + lane] = score;
    }
    if (state->current_fill < state->cursor + 1ull)
        state->current_fill = state->cursor + 1ull;
    state->next_position++;
    state->cursor = (state->cursor + 1ull) % state->ratio;
    if (state->cursor != 0ull) return 1;

    for (lane = 0ull; lane < state->head_dimension; ++lane) {
        double maximum = -HUGE_VAL;
        double denominator = 0.0;
        double sum = 0.0;
        for (slot = 0ull; slot < state->ratio; ++slot) {
            double first = state->score[slot * state->width + lane];
            if (first > maximum) maximum = first;
            if (state->overlap) {
                double second = state->score[
                    (state->ratio + slot) * state->width +
                    lane + state->head_dimension];
                if (second > maximum) maximum = second;
            }
        }
        for (slot = 0ull; slot < state->ratio; ++slot) {
            double weight = exp(state->score[slot * state->width + lane] -
                                maximum);
            denominator += weight;
            sum += weight * state->kv[slot * state->width + lane];
            if (state->overlap) {
                unsigned long long offset =
                    (state->ratio + slot) * state->width +
                    lane + state->head_dimension;
                weight = exp(state->score[offset] - maximum);
                denominator += weight;
                sum += weight * state->kv[offset];
            }
        }
        if (!(denominator > 0.0) || !isfinite(denominator) || !isfinite(sum))
            return 0;
        emission[lane] = (float)(sum / denominator);
    }
    if (state->overlap) {
        for (slot = 0ull; slot < state->ratio; ++slot) {
            memcpy(state->kv + slot * state->width,
                   state->kv + (state->ratio + slot) * state->width,
                   (size_t)state->width * sizeof(*state->kv));
            memcpy(state->score + slot * state->width,
                   state->score + (state->ratio + slot) * state->width,
                   (size_t)state->width * sizeof(*state->score));
        }
        state->previous_fill = state->ratio;
    } else {
        state->previous_fill = 0ull;
    }
    state->current_fill = 0ull;
    *emitted = 1;
    return 1;
}

static int ref_rolling_export(
    ref_rolling_state *state,
    const yvex_attention_layer_plan *layer,
    yvex_attention_rolling_kind kind,
    const yvex_attention_rolling_state_view *before,
    yvex_attention_rolling_state_output *out)
{
    unsigned long long extent;

    if (!state || !layer || !out || !state->kv || !state->score ||
        !ref_mul(state->slots, state->width, &extent))
        return 0;
    memset(out, 0, sizeof(*out));
    out->present = 1;
    out->schema_version = YVEX_DEEPSEEK_ATTENTION_ROLLING_STATE_SCHEMA_V1;
    out->kind = kind;
    out->attention_class = layer->attention_class;
    out->layer_index = layer->layer_index;
    out->next_token_position = state->next_position;
    out->ratio = state->ratio;
    out->head_dimension = state->head_dimension;
    out->state_width = state->width;
    out->state_slots = state->slots;
    out->previous_fill = state->previous_fill;
    out->current_fill = state->current_fill;
    out->cursor = state->cursor;
    out->kv_state_stride = state->width;
    out->score_state_stride = state->width;
    out->kv_state_extent = extent;
    out->score_state_extent = extent;
    out->kv_state = state->kv;
    out->score_state = state->score;
    out->overlap = state->overlap;
    out->rotated = kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER;
    if (before)
        memcpy(out->attention_plan_identity,
               before->attention_plan_identity,
               sizeof(out->attention_plan_identity));
    state->kv = NULL;
    state->score = NULL;
    return 1;
}

static int ref_rank_compare(const void *left, const void *right)
{
    const ref_ranked_candidate *a = (const ref_ranked_candidate *)left;
    const ref_ranked_candidate *b = (const ref_ranked_candidate *)right;

    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    if (a->position < b->position) return -1;
    if (a->position > b->position) return 1;
    return 0;
}

int yvex_test_attention_reference_topk(
    const float *scores,
    const unsigned long long *positions,
    unsigned long long count,
    unsigned long long k,
    unsigned long long *selected,
    unsigned long long *selected_count)
{
    ref_ranked_candidate *ranked;
    unsigned long long i;

    if (selected_count) *selected_count = 0ull;
    if (!scores || !positions || !selected || !selected_count) return 0;
    if (!count || !k) return 1;
    ranked = (ref_ranked_candidate *)ref_alloc(count, sizeof(*ranked));
    if (!ranked) return 0;
    for (i = 0ull; i < count; ++i) {
        unsigned long long prior;
        if (!isfinite(scores[i])) {
            free(ranked);
            return 0;
        }
        for (prior = 0ull; prior < i; ++prior) {
            if (positions[prior] == positions[i]) {
                free(ranked);
                return 0;
            }
        }
        ranked[i].score = scores[i];
        ranked[i].position = positions[i];
        ranked[i].index = i;
    }
    qsort(ranked, (size_t)count, sizeof(*ranked), ref_rank_compare);
    *selected_count = k < count ? k : count;
    for (i = 0ull; i < *selected_count; ++i)
        selected[i] = ranked[i].index;
    free(ranked);
    return 1;
}

static const float *ref_segmented_row(const float *history,
                                      unsigned long long history_count,
                                      unsigned long long history_stride,
                                      const float *current,
                                      unsigned long long current_count,
                                      unsigned long long current_stride,
                                      unsigned long long index)
{
    if (index < history_count)
        return history ? history + index * history_stride : NULL;
    index -= history_count;
    if (index >= current_count) return NULL;
    return current ? current + index * current_stride : NULL;
}

static unsigned long long ref_segmented_position(
    const unsigned long long *history,
    unsigned long long history_count,
    const unsigned long long *current,
    unsigned long long current_count,
    unsigned long long index)
{
    if (index < history_count) return history ? history[index] : ULLONG_MAX;
    index -= history_count;
    if (index >= current_count) return ULLONG_MAX;
    return current ? current[index] : ULLONG_MAX;
}

static int ref_decode_flat(yvex_materialization_session *session,
                           const yvex_runtime_tensor_binding *runtime,
                           float *out,
                           unsigned long long elements,
                           char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP])
{
    unsigned long long total;
    unsigned long long row;

    if (!runtime || !runtime->binding ||
        !ref_mul(runtime->binding->row_width, runtime->binding->row_count,
                 &total) || total != elements)
        return 0;
    for (row = 0ull; row < runtime->binding->row_count; ++row) {
        if (!ref_decode_row(session, runtime, row,
                            out + row * runtime->binding->row_width,
                            reason))
            return 0;
    }
    return 1;
}

static unsigned long long ref_emission_count(unsigned long long start,
                                             unsigned long long count,
                                             unsigned long long ratio)
{
    unsigned long long token;
    unsigned long long emitted = 0ull;

    if (!ratio) return ULLONG_MAX;
    for (token = 0ull; token < count; ++token) {
        if (start > ULLONG_MAX - token - 1ull) return ULLONG_MAX;
        if ((start + token + 1ull) % ratio == 0ull) emitted++;
    }
    return emitted;
}

static int ref_compressor_execute(
    yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_layer_plan *layer,
    const yvex_deepseek_v4_layer_spec *architecture,
    yvex_attention_rolling_kind kind,
    const yvex_attention_rolling_state_view *before,
    const float *hidden,
    unsigned long long token_count,
    unsigned long long hidden_stride,
    unsigned long long token_position,
    float **output,
    unsigned long long **positions,
    unsigned long long *output_count,
    yvex_attention_rolling_state_output *next_state,
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP])
{
    yvex_tensor_role kv_role =
        kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER
            ? YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV
            : YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV;
    yvex_tensor_role gate_role =
        kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER
            ? YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE
            : YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE;
    yvex_tensor_role ape_role =
        kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER
            ? YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE
            : YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE;
    yvex_tensor_role norm_role =
        kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER
            ? YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM
            : YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM;
    const yvex_runtime_tensor_binding *wkv =
        ref_binding(descriptor, kv_role, layer->layer_index);
    const yvex_runtime_tensor_binding *wgate =
        ref_binding(descriptor, gate_role, layer->layer_index);
    const yvex_runtime_tensor_binding *ape =
        ref_binding(descriptor, ape_role, layer->layer_index);
    const yvex_runtime_tensor_binding *norm =
        ref_binding(descriptor, norm_role, layer->layer_index);
    ref_rolling_state state;
    unsigned long long projected_count;
    unsigned long long emissions;
    unsigned long long token;
    unsigned long long emitted_count = 0ull;
    float *projected_kv = NULL;
    float *projected_score = NULL;
    float *ape_row = NULL;
    float *norm_weight = NULL;
    float *emitted_values = NULL;
    unsigned long long *emitted_positions = NULL;
    int ok = 0;

    if (output) *output = NULL;
    if (positions) *positions = NULL;
    if (output_count) *output_count = 0ull;
    if (next_state) memset(next_state, 0, sizeof(*next_state));
    memset(&state, 0, sizeof(state));
    if (!session || !descriptor || !layer || !architecture || !hidden ||
        !output || !positions || !output_count || !next_state || !wkv ||
        !wgate || !ape || !norm ||
        !ref_rolling_init(&state, layer, kind, before,
                                   token_position) ||
        !ref_mul(token_count, state.width, &projected_count)) {
        ref_reason(reason, "reference compressor admission failed");
        goto cleanup;
    }
    projected_kv = (float *)ref_alloc(projected_count,
                                      sizeof(*projected_kv));
    projected_score = (float *)ref_alloc(projected_count,
                                         sizeof(*projected_score));
    ape_row = (float *)ref_alloc(state.width, sizeof(*ape_row));
    norm_weight = (float *)ref_alloc(state.head_dimension,
                                     sizeof(*norm_weight));
    emissions = ref_emission_count(token_position, token_count, state.ratio);
    if (emissions == ULLONG_MAX ||
        (emissions &&
         (!(emitted_values = (float *)ref_alloc(
                emissions * state.head_dimension,
                sizeof(*emitted_values))) ||
          !(emitted_positions = (unsigned long long *)ref_alloc(
                emissions, sizeof(*emitted_positions))))) ||
        !projected_kv || !projected_score || !ape_row || !norm_weight) {
        ref_reason(reason, "reference compressor allocation failed");
        goto cleanup;
    }
    if (!ref_matrix_batch(session, wkv, 0ull, state.width, hidden,
                          token_count, hidden_stride, projected_kv,
                          state.width, reason) ||
        !ref_matrix_batch(session, wgate, 0ull, state.width, hidden,
                          token_count, hidden_stride, projected_score,
                          state.width, reason) ||
        !ref_decode_row(session, norm, 0ull, norm_weight, reason))
        goto cleanup;
    for (token = 0ull; token < token_count; ++token) {
        float scratch[1];
        float *destination = emissions
            ? emitted_values + emitted_count * state.head_dimension
            : scratch;
        int emitted = 0;
        unsigned long long position;

        if (!ref_decode_row(session, ape,
                            (token_position + token) % state.ratio,
                            ape_row, reason) ||
            !ref_rolling_step(&state,
                              projected_kv + token * state.width,
                              projected_score + token * state.width,
                              ape_row, destination, &emitted)) {
            ref_reason(reason, "reference compressor transition failed");
            goto cleanup;
        }
        if (!emitted) continue;
        if (emitted_count >= emissions ||
            !ref_compute_round(layer->compute_contract, destination,
                               state.head_dimension) ||
            !ref_rms(destination, state.head_dimension, norm_weight,
                     architecture->rms_norm_epsilon) ||
            !ref_compute_round(layer->compute_contract, destination,
                               state.head_dimension)) {
            ref_reason(reason, "reference compressor normalization failed");
            goto cleanup;
        }
        position = token_position + token + 1ull - state.ratio;
        if (!ref_rope(destination, state.head_dimension,
                      architecture->rope_head_dimension, position,
                      &architecture->position, 0) ||
            !ref_compute_round(layer->compute_contract, destination,
                               state.head_dimension)) {
            ref_reason(reason, "reference compressor RoPE failed");
            goto cleanup;
        }
        if (kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER) {
            if (!ref_activation(&layer->compressor_rotated_activation,
                                destination, state.head_dimension)) {
                ref_reason(reason,
                           "reference indexer compressor activation failed");
                goto cleanup;
            }
        } else if (state.head_dimension > architecture->rope_head_dimension &&
                   !ref_activation(
                       &layer->compressor_activation, destination,
                       state.head_dimension - architecture->rope_head_dimension)) {
            ref_reason(reason, "reference compressor activation failed");
            goto cleanup;
        }
        emitted_positions[emitted_count++] = position;
    }
    if (emitted_count != emissions) {
        ref_reason(reason, "reference compressor emission count mismatched");
        goto cleanup;
    }
    if (!ref_rolling_export(&state, layer, kind, before, next_state)) {
        ref_reason(reason, "reference compressor state export failed");
        goto cleanup;
    }
    *output = emitted_values;
    *positions = emitted_positions;
    *output_count = emitted_count;
    emitted_values = NULL;
    emitted_positions = NULL;
    ok = 1;

cleanup:
    ref_rolling_release(&state);
    free(projected_kv);
    free(projected_score);
    free(ape_row);
    free(norm_weight);
    free(emitted_values);
    free(emitted_positions);
    return ok;
}

static int ref_csa_select(
    const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *history,
    const float *current_indexer,
    unsigned long long current_count,
    unsigned long long current_stride,
    const unsigned long long *current_positions,
    const float *query,
    const float *weights,
    unsigned long long query_position,
    unsigned long long *selected,
    unsigned long long *selected_count)
{
    unsigned long long total = history->indexer_entry_count + current_count;
    ref_ranked_candidate *ranked;
    unsigned long long candidate;
    unsigned long long valid = 0ull;
    unsigned long long count;

    *selected_count = 0ull;
    if (!total) return 1;
    ranked = (ref_ranked_candidate *)ref_alloc(total, sizeof(*ranked));
    if (!ranked) return 0;
    for (candidate = 0ull; candidate < total; ++candidate) {
        const float *key = ref_segmented_row(
            history->indexer_kv, history->indexer_entry_count,
            history->indexer_kv_stride, current_indexer, current_count,
            current_stride, candidate);
        unsigned long long position = ref_segmented_position(
            history->indexer_positions, history->indexer_entry_count,
            current_positions, current_count, candidate);
        unsigned long long head;
        double score = 0.0;

        if (!key || position == ULLONG_MAX || position > query_position ||
            position > ULLONG_MAX - layer->compression_ratio + 1ull ||
            position + layer->compression_ratio - 1ull > query_position)
            continue;
        for (head = 0ull; head < layer->indexer_heads; ++head) {
            unsigned long long lane;
            double dot = 0.0;
            for (lane = 0ull; lane < layer->indexer_head_dimension; ++lane)
                dot += (double)query[
                           head * layer->indexer_head_dimension + lane] *
                       (double)key[lane];
            if (dot < 0.0) dot = 0.0;
            score += dot * (double)weights[head];
        }
        score /= sqrt((double)layer->indexer_head_dimension);
        score /= sqrt((double)layer->indexer_heads);
        if (!isfinite(score)) {
            free(ranked);
            return 0;
        }
        ranked[valid].score = (float)score;
        ranked[valid].position = position;
        ranked[valid].index = candidate;
        valid++;
    }
    qsort(ranked, (size_t)valid, sizeof(*ranked), ref_rank_compare);
    for (candidate = 1ull; candidate < valid; ++candidate) {
        if (ranked[candidate - 1ull].position == ranked[candidate].position) {
            free(ranked);
            return 0;
        }
    }
    count = valid < layer->sparse_topk.k ? valid : layer->sparse_topk.k;
    for (candidate = 0ull; candidate < count; ++candidate)
        selected[candidate] = ranked[candidate].index;
    *selected_count = count;
    free(ranked);
    return 1;
}

static int ref_candidate_score(const float *query,
                               const float *row,
                               unsigned long long width,
                               double scale,
                               double *score)
{
    unsigned long long lane;
    double sum = 0.0;

    for (lane = 0ull; lane < width; ++lane) {
        if (!isfinite(query[lane]) || !isfinite(row[lane])) return 0;
        sum += (double)query[lane] * (double)row[lane];
    }
    *score = sum * scale;
    return isfinite(*score);
}

static int ref_accumulate(const float *query,
                          const float *row,
                          unsigned long long width,
                          double scale,
                          double maximum,
                          double *denominator,
                          float *output)
{
    unsigned long long lane;
    double score;
    double weight;

    if (!ref_candidate_score(query, row, width, scale, &score)) return 0;
    weight = exp(score - maximum);
    if (!isfinite(weight)) return 0;
    *denominator += weight;
    for (lane = 0ull; lane < width; ++lane)
        output[lane] += (float)(weight * (double)row[lane]);
    return 1;
}

static int ref_history_validate(
    const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *history,
    int supplied,
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP])
{
    unsigned long long local_capacity;
    unsigned long long expected_local;
    unsigned long long local_start;
    unsigned long long expected_compressed = 0ull;
    unsigned long long i;

    if (!layer || !history || !history->immutable || !layer->sliding_window ||
        (history->local_tail_count &&
         (!history->local_kv || !history->local_positions ||
          history->local_kv_stride < layer->head_dimension)) ||
        (history->compressed_entry_count &&
         (!history->compressed_kv || !history->compressed_positions ||
          history->compressed_kv_stride < layer->head_dimension)) ||
        (history->indexer_entry_count &&
         (!history->indexer_kv || !history->indexer_positions ||
          history->indexer_kv_stride < layer->indexer_head_dimension))) {
        ref_reason(reason, "reference history geometry is invalid");
        return 0;
    }
    if (!supplied && history->token_count != 0ull) {
        ref_reason(reason,
                   "reference execution cannot omit nonempty history");
        return 0;
    }
    local_capacity = layer->sliding_window - 1ull;
    expected_local = history->token_count < local_capacity
        ? history->token_count : local_capacity;
    local_start = history->token_count - expected_local;
    if (history->local_tail_count != expected_local) {
        ref_reason(reason, "reference local history suffix is incomplete");
        return 0;
    }
    if (layer->attention_class != YVEX_ATTENTION_CLASS_SWA) {
        if (!layer->compression_ratio) {
            ref_reason(reason, "reference compression ratio is invalid");
            return 0;
        }
        expected_compressed =
            history->token_count / layer->compression_ratio;
    }
    if (history->compressed_entry_count != expected_compressed ||
        (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
         history->indexer_entry_count != expected_compressed)) {
        ref_reason(reason,
                   "reference compressed history prefix is incomplete");
        return 0;
    }
    if ((layer->attention_class == YVEX_ATTENTION_CLASS_SWA &&
         (history->compressed_entry_count ||
          history->indexer_entry_count)) ||
        (layer->attention_class == YVEX_ATTENTION_CLASS_HCA &&
         history->indexer_entry_count) ||
        (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
         history->compressed_entry_count != history->indexer_entry_count)) {
        ref_reason(reason, "reference history class contract is invalid");
        return 0;
    }
    for (i = 0ull; i < history->local_tail_count; ++i) {
        if (history->local_positions[i] != local_start + i) {
            ref_reason(reason,
                       "reference local history is not the exact suffix");
            return 0;
        }
    }
    for (i = 0ull; i < history->compressed_entry_count; ++i) {
        unsigned long long expected_position;

        if (!ref_mul(i, layer->compression_ratio, &expected_position) ||
            history->compressed_positions[i] != expected_position) {
            ref_reason(reason,
                       "reference compressed history omits a ratio group");
            return 0;
        }
    }
    for (i = 0ull; i < history->indexer_entry_count; ++i) {
        if (layer->attention_class != YVEX_ATTENTION_CLASS_CSA ||
            history->indexer_positions[i] !=
                history->compressed_positions[i]) {
            ref_reason(reason, "reference indexer ordinals are invalid");
            return 0;
        }
    }
    if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA) {
        if (history->main_rolling_state.present ||
            history->indexer_rolling_state.present) {
            ref_reason(reason, "reference SWA history carries rolling state");
            return 0;
        }
    } else if (supplied &&
               !ref_rolling_view_complete(
                   layer, YVEX_DEEPSEEK_ATTENTION_ROLLING_MAIN,
                   &history->main_rolling_state, history->token_count)) {
        ref_reason(reason, "reference main rolling history is incomplete");
        return 0;
    } else if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
               supplied &&
               !ref_rolling_view_complete(
                   layer, YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER,
                   &history->indexer_rolling_state, history->token_count)) {
        ref_reason(reason,
                   "reference indexer rolling history is incomplete");
        return 0;
    } else if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA &&
               history->indexer_rolling_state.present) {
        ref_reason(reason, "reference HCA history carries indexer state");
        return 0;
    }
    return 1;
}

/* Contract: independently evaluates local plus HCA/CSA sparse attention and
 * records deterministic CSA selections. */
static int ref_sparse_reduce(
    const yvex_attention_layer_plan *layer,
    const yvex_deepseek_v4_layer_spec *architecture,
    const float *query,
    const yvex_attention_history_view *history,
    const float *current_kv,
    unsigned long long current_kv_stride,
    const float *current_compressed,
    unsigned long long current_compressed_count,
    const unsigned long long *current_compressed_positions,
    const float *current_indexer,
    unsigned long long current_indexer_count,
    const unsigned long long *current_indexer_positions,
    const float *index_query,
    const float *index_weights,
    const float *sinks,
    unsigned long long token_count,
    unsigned long long token_position,
    float *out,
    unsigned long long *topk_counts,
    unsigned long long *topk_positions,
    unsigned long long topk_stride)
{
    unsigned long long compressed_total =
        history->compressed_entry_count + current_compressed_count;
    unsigned long long *selected = NULL;
    unsigned long long token;
    double scale = 1.0 / sqrt((double)layer->head_dimension);

    if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
        compressed_total) {
        selected = (unsigned long long *)ref_alloc(
            compressed_total < layer->sparse_topk.k
                ? compressed_total : layer->sparse_topk.k,
            sizeof(*selected));
        if (!selected) return 0;
    }
    for (token = 0ull; token < token_count; ++token) {
        unsigned long long absolute = token_position + token;
        unsigned long long selected_count = 0ull;
        unsigned long long head;

        if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
            compressed_total &&
            !ref_csa_select(
                layer, history, current_indexer, current_indexer_count,
                layer->indexer_head_dimension, current_indexer_positions,
                index_query + token * layer->indexer_heads *
                                  layer->indexer_head_dimension,
                index_weights + token * layer->indexer_heads, absolute,
                selected, &selected_count)) {
            free(selected);
            return 0;
        }
        if (topk_counts) {
            unsigned long long i;
            if (selected_count > topk_stride) {
                free(selected);
                return 0;
            }
            topk_counts[token] = selected_count;
            for (i = 0ull; i < selected_count; ++i)
                topk_positions[token * topk_stride + i] =
                    ref_segmented_position(
                        history->compressed_positions,
                        history->compressed_entry_count,
                        current_compressed_positions,
                        current_compressed_count, selected[i]);
        }
        for (head = 0ull; head < layer->query_heads; ++head) {
            const float *head_query =
                query + token * layer->query_heads * layer->head_dimension +
                head * layer->head_dimension;
            float *destination =
                out + token * layer->query_heads * layer->head_dimension +
                head * layer->head_dimension;
            unsigned long long candidate;
            unsigned long long lane;
            double maximum = sinks[head];
            double denominator;

            if (!isfinite(maximum)) goto fail;
            for (candidate = 0ull; candidate < history->local_tail_count;
                 ++candidate) {
                unsigned long long position = history->local_positions[candidate];
                unsigned long long first =
                    absolute + 1ull > layer->sliding_window
                        ? absolute + 1ull - layer->sliding_window : 0ull;
                double score;
                if (position < first || position > absolute) continue;
                if (!ref_candidate_score(
                        head_query,
                        history->local_kv + candidate * history->local_kv_stride,
                        layer->head_dimension, scale, &score))
                    goto fail;
                if (score > maximum) maximum = score;
            }
            for (candidate = 0ull; candidate <= token; ++candidate) {
                unsigned long long position = token_position + candidate;
                unsigned long long first =
                    absolute + 1ull > layer->sliding_window
                        ? absolute + 1ull - layer->sliding_window : 0ull;
                double score;
                if (position < first) continue;
                if (!ref_candidate_score(
                        head_query, current_kv + candidate * current_kv_stride,
                        layer->head_dimension, scale, &score))
                    goto fail;
                if (score > maximum) maximum = score;
            }
            if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA) {
                for (candidate = 0ull; candidate < compressed_total;
                     ++candidate) {
                    unsigned long long position = ref_segmented_position(
                        history->compressed_positions,
                        history->compressed_entry_count,
                        current_compressed_positions,
                        current_compressed_count, candidate);
                    const float *row;
                    double score;
                    if (position == ULLONG_MAX || position > absolute ||
                        position > ULLONG_MAX - layer->compression_ratio + 1ull ||
                        position + layer->compression_ratio - 1ull > absolute)
                        continue;
                    row = ref_segmented_row(
                        history->compressed_kv,
                        history->compressed_entry_count,
                        history->compressed_kv_stride, current_compressed,
                        current_compressed_count, layer->head_dimension,
                        candidate);
                    if (!ref_candidate_score(head_query, row,
                                             layer->head_dimension, scale,
                                             &score))
                        goto fail;
                    if (score > maximum) maximum = score;
                }
            } else if (layer->attention_class ==
                       YVEX_ATTENTION_CLASS_CSA) {
                for (candidate = 0ull; candidate < selected_count;
                     ++candidate) {
                    const float *row = ref_segmented_row(
                        history->compressed_kv,
                        history->compressed_entry_count,
                        history->compressed_kv_stride, current_compressed,
                        current_compressed_count, layer->head_dimension,
                        selected[candidate]);
                    double score;
                    if (!ref_candidate_score(head_query, row,
                                             layer->head_dimension, scale,
                                             &score))
                        goto fail;
                    if (score > maximum) maximum = score;
                }
            }
            denominator = exp((double)sinks[head] - maximum);
            memset(destination, 0,
                   (size_t)layer->head_dimension * sizeof(*destination));
            for (candidate = 0ull; candidate < history->local_tail_count;
                 ++candidate) {
                unsigned long long position = history->local_positions[candidate];
                unsigned long long first =
                    absolute + 1ull > layer->sliding_window
                        ? absolute + 1ull - layer->sliding_window : 0ull;
                if (position < first || position > absolute) continue;
                if (!ref_accumulate(
                        head_query,
                        history->local_kv + candidate * history->local_kv_stride,
                        layer->head_dimension, scale, maximum, &denominator,
                        destination))
                    goto fail;
            }
            for (candidate = 0ull; candidate <= token; ++candidate) {
                unsigned long long position = token_position + candidate;
                unsigned long long first =
                    absolute + 1ull > layer->sliding_window
                        ? absolute + 1ull - layer->sliding_window : 0ull;
                if (position < first) continue;
                if (!ref_accumulate(
                        head_query, current_kv + candidate * current_kv_stride,
                        layer->head_dimension, scale, maximum, &denominator,
                        destination))
                    goto fail;
            }
            if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA) {
                for (candidate = 0ull; candidate < compressed_total;
                     ++candidate) {
                    unsigned long long position = ref_segmented_position(
                        history->compressed_positions,
                        history->compressed_entry_count,
                        current_compressed_positions,
                        current_compressed_count, candidate);
                    const float *row;
                    if (position == ULLONG_MAX || position > absolute ||
                        position > ULLONG_MAX - layer->compression_ratio + 1ull ||
                        position + layer->compression_ratio - 1ull > absolute)
                        continue;
                    row = ref_segmented_row(
                        history->compressed_kv,
                        history->compressed_entry_count,
                        history->compressed_kv_stride, current_compressed,
                        current_compressed_count, layer->head_dimension,
                        candidate);
                    if (!ref_accumulate(head_query, row,
                                        layer->head_dimension, scale, maximum,
                                        &denominator, destination))
                        goto fail;
                }
            } else if (layer->attention_class ==
                       YVEX_ATTENTION_CLASS_CSA) {
                for (candidate = 0ull; candidate < selected_count;
                     ++candidate) {
                    const float *row = ref_segmented_row(
                        history->compressed_kv,
                        history->compressed_entry_count,
                        history->compressed_kv_stride, current_compressed,
                        current_compressed_count, layer->head_dimension,
                        selected[candidate]);
                    if (!ref_accumulate(head_query, row,
                                        layer->head_dimension, scale, maximum,
                                        &denominator, destination))
                        goto fail;
                }
            }
            if (!(denominator > 0.0) || !isfinite(denominator)) goto fail;
            for (lane = 0ull; lane < layer->head_dimension; ++lane)
                destination[lane] = (float)((double)destination[lane] /
                                            denominator);
            if (!ref_compute_round(layer->compute_contract, destination,
                                   layer->head_dimension) ||
                !ref_rope(destination, layer->head_dimension,
                          architecture->rope_head_dimension, absolute,
                          &architecture->position, 1) ||
                !ref_compute_round(layer->compute_contract, destination,
                                   layer->head_dimension))
                goto fail;
        }
    }
    free(selected);
    return 1;

fail:
    free(selected);
    return 0;
}

static int ref_output_projection(
    yvex_materialization_session *session,
    const yvex_runtime_tensor_binding *out_a,
    const yvex_runtime_tensor_binding *out_b,
    const yvex_attention_layer_plan *layer,
    const yvex_deepseek_v4_layer_spec *architecture,
    const float *attention,
    unsigned long long token_count,
    unsigned long long attention_stride,
    unsigned long long hidden_width,
    float *output,
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP])
{
    unsigned long long low_stride;
    unsigned long long low_count;
    unsigned long long output_count;
    unsigned long long group;
    float *low;

    if (!layer || !ref_mul(architecture->output_groups,
                 architecture->output_lora_rank, &low_stride) ||
        !ref_mul(token_count, low_stride, &low_count) ||
        !ref_mul(token_count, hidden_width, &output_count))
        return 0;
    low = (float *)ref_alloc(low_count, sizeof(*low));
    if (!low) return 0;
    for (group = 0ull; group < architecture->output_groups; ++group) {
        if (!ref_matrix_batch(
                session, out_a, group * architecture->output_lora_rank,
                architecture->output_lora_rank,
                attention + group * architecture->output_group_input_width,
                token_count, attention_stride,
                low + group * architecture->output_lora_rank, low_stride,
                reason)) {
            free(low);
            return 0;
        }
    }
    if (!ref_compute_round(layer->compute_contract, low, low_count) ||
        !ref_matrix_batch(session, out_b, 0ull, hidden_width, low,
                          token_count, low_stride, output, hidden_width,
                          reason) ||
        !ref_compute_round(layer->compute_contract, output, output_count)) {
        free(low);
        return 0;
    }
    free(low);
    return 1;
}

int yvex_test_attention_reference_execute(
    const yvex_attention_plan *plan,
    const yvex_deepseek_v4_ir *ir,
    yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_cpu_options *options,
    yvex_attention_execution_trace *trace,
    char reason[YVEX_TEST_ATTENTION_REFERENCE_REASON_CAP])
{
    const yvex_attention_cpu_options *opts = options;
    const yvex_attention_layer_plan *layer;
    const yvex_deepseek_v4_layer_spec *architecture;
    const yvex_runtime_tensor_binding *q_a;
    const yvex_runtime_tensor_binding *q_a_norm;
    const yvex_runtime_tensor_binding *q_b;
    const yvex_runtime_tensor_binding *kv;
    const yvex_runtime_tensor_binding *kv_norm;
    const yvex_runtime_tensor_binding *sinks_binding;
    const yvex_runtime_tensor_binding *out_a;
    const yvex_runtime_tensor_binding *out_b;
    yvex_attention_history_view empty_history;
    const yvex_attention_history_view *history;
    unsigned long long token_count;
    unsigned long long hidden_width;
    unsigned long long q_rank;
    unsigned long long query_width;
    unsigned long long kv_width;
    unsigned long long hidden_count;
    unsigned long long q_count;
    unsigned long long query_count;
    unsigned long long kv_count;
    unsigned long long token;
    float *input = NULL;
    float *q_low = NULL;
    float *q_norm = NULL;
    float *query = NULL;
    float *raw_kv = NULL;
    float *kv_norm_weight = NULL;
    float *sinks = NULL;
    float *compressed = NULL;
    unsigned long long *compressed_positions = NULL;
    unsigned long long compressed_count = 0ull;
    float *indexer = NULL;
    unsigned long long *indexer_positions = NULL;
    unsigned long long indexer_count = 0ull;
    float *index_query = NULL;
    float *index_weights = NULL;
    float *attention = NULL;
    float *output = NULL;
    unsigned long long *topk_counts = NULL;
    unsigned long long *topk_positions = NULL;
    yvex_attention_rolling_state_output next_main_state;
    yvex_attention_rolling_state_output next_indexer_state;
    unsigned long long topk_stride = 0ull;
    int ok = 0;

    memset(&next_main_state, 0, sizeof(next_main_state));
    memset(&next_indexer_state, 0, sizeof(next_indexer_state));
    if (reason) reason[0] = '\0';
    if (!plan || !ir || !session || !descriptor || !opts || !opts->input ||
        !trace || trace->owned) {
        ref_reason(reason, "reference execution arguments are invalid");
        return 0;
    }
    if (!yvex_test_attention_external_conformance_validate(NULL)) {
        ref_reason(reason, "pinned external conformance vectors are invalid");
        return 0;
    }
    token_count = opts->token_count ? opts->token_count : 1ull;
    layer = yvex_graph_lower_deepseek_v4()->plan_layer_at(plan, opts->layer_index);
    architecture = yvex_model_register_deepseek_v4()->ir.layer_at(ir, opts->layer_index);
    if (!layer || !architecture || !token_count) {
        ref_reason(reason, "reference execution layer is absent");
        return 0;
    }
    if (layer->compute_contract != YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1 ||
        architecture->compute_contract != layer->compute_contract) {
        ref_reason(reason, "reference compute contract is unsupported");
        return 0;
    }
    memset(&empty_history, 0, sizeof(empty_history));
    empty_history.immutable = 1;
    empty_history.token_count = opts->token_position;
    history = opts->history ? opts->history : &empty_history;
    if (history->token_count != opts->token_position) {
        ref_reason(reason, "reference history is not contiguous");
        goto cleanup;
    }
    if (!ref_history_validate(layer, history, opts->history != NULL, reason))
        goto cleanup;
    q_a = ref_binding(descriptor, YVEX_TENSOR_ROLE_ATTENTION_Q_A,
                      layer->layer_index);
    q_a_norm = ref_binding(descriptor, YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM,
                           layer->layer_index);
    q_b = ref_binding(descriptor, YVEX_TENSOR_ROLE_ATTENTION_Q_B,
                      layer->layer_index);
    kv = ref_binding(descriptor, YVEX_TENSOR_ROLE_ATTENTION_KV,
                     layer->layer_index);
    kv_norm = ref_binding(descriptor, YVEX_TENSOR_ROLE_ATTENTION_KV_NORM,
                          layer->layer_index);
    sinks_binding = ref_binding(descriptor, YVEX_TENSOR_ROLE_ATTENTION_SINKS,
                                layer->layer_index);
    out_a = ref_binding(descriptor, YVEX_TENSOR_ROLE_ATTENTION_OUT_A,
                        layer->layer_index);
    out_b = ref_binding(descriptor, YVEX_TENSOR_ROLE_ATTENTION_OUT_B,
                        layer->layer_index);
    if (!q_a || !q_a_norm || !q_b || !kv || !kv_norm || !sinks_binding ||
        !out_a || !out_b) {
        ref_reason(reason, "reference execution bindings are incomplete");
        goto cleanup;
    }
    hidden_width = q_a->binding->row_width;
    q_rank = q_a->binding->row_count;
    query_width = q_b->binding->row_count;
    kv_width = kv->binding->row_count;
    if (q_b->binding->row_width != q_rank ||
        query_width != layer->query_heads * layer->head_dimension ||
        kv_width != layer->head_dimension ||
        hidden_width != layer->hidden_dimension ||
        !ref_mul(token_count, hidden_width, &hidden_count) ||
        !ref_mul(token_count, q_rank, &q_count) ||
        !ref_mul(token_count, query_width, &query_count) ||
        !ref_mul(token_count, kv_width, &kv_count)) {
        ref_reason(reason, "reference execution dimensions are invalid");
        goto cleanup;
    }
    input = (float *)ref_alloc(hidden_count, sizeof(*input));
    q_low = (float *)ref_alloc(q_count, sizeof(*q_low));
    q_norm = (float *)ref_alloc(q_rank, sizeof(*q_norm));
    query = (float *)ref_alloc(query_count, sizeof(*query));
    raw_kv = (float *)ref_alloc(kv_count, sizeof(*raw_kv));
    kv_norm_weight = (float *)ref_alloc(kv_width, sizeof(*kv_norm_weight));
    sinks = (float *)ref_alloc(layer->query_heads, sizeof(*sinks));
    attention = (float *)ref_alloc(query_count, sizeof(*attention));
    output = (float *)ref_alloc(hidden_count, sizeof(*output));
    if (!input || !q_low || !q_norm || !query || !raw_kv ||
        !kv_norm_weight || !sinks || !attention || !output) {
        ref_reason(reason, "reference execution allocation failed");
        goto cleanup;
    }
    for (token = 0ull; token < token_count; ++token) {
        unsigned long long lane;
        if (opts->input_stride < hidden_width) {
            ref_reason(reason, "reference input stride is invalid");
            goto cleanup;
        }
        memcpy(input + token * hidden_width,
               opts->input + token * opts->input_stride,
               (size_t)hidden_width * sizeof(*input));
        for (lane = 0ull; lane < hidden_width; ++lane) {
            if (!isfinite(input[token * hidden_width + lane])) {
                ref_reason(reason, "reference input is non-finite");
                goto cleanup;
            }
        }
        if (!ref_compute_round(layer->compute_contract,
                               input + token * hidden_width,
                               hidden_width)) {
            ref_reason(reason, "reference input BF16 publication failed");
            goto cleanup;
        }
    }
    if (!ref_matrix_batch(session, q_a, 0ull, q_rank, input, token_count,
                          hidden_width, q_low, q_rank, reason) ||
        !ref_compute_round(layer->compute_contract, q_low, q_count) ||
        !ref_decode_row(session, q_a_norm, 0ull, q_norm, reason))
        goto cleanup;
    for (token = 0ull; token < token_count; ++token) {
        if (!ref_rms(q_low + token * q_rank, q_rank, q_norm,
                     architecture->rms_norm_epsilon) ||
            !ref_compute_round(layer->compute_contract,
                               q_low + token * q_rank, q_rank)) {
            ref_reason(reason, "reference Q-A norm failed");
            goto cleanup;
        }
    }
    if (!ref_matrix_batch(session, q_b, 0ull, query_width, q_low,
                          token_count, q_rank, query, query_width, reason) ||
        !ref_compute_round(layer->compute_contract, query, query_count) ||
        !ref_matrix_batch(session, kv, 0ull, kv_width, input, token_count,
                          hidden_width, raw_kv, kv_width, reason) ||
        !ref_compute_round(layer->compute_contract, raw_kv, kv_count) ||
        !ref_decode_row(session, kv_norm, 0ull, kv_norm_weight, reason) ||
        !ref_decode_flat(session, sinks_binding, sinks, layer->query_heads,
                         reason))
        goto cleanup;
    for (token = 0ull; token < token_count; ++token) {
        unsigned long long head;
        for (head = 0ull; head < layer->query_heads; ++head) {
            float *head_query = query + token * query_width +
                                head * layer->head_dimension;
            if (!ref_rms(head_query, layer->head_dimension, NULL,
                         architecture->rms_norm_epsilon) ||
                !ref_compute_round(layer->compute_contract, head_query,
                                   layer->head_dimension) ||
                !ref_rope(head_query, layer->head_dimension,
                          architecture->rope_head_dimension,
                          opts->token_position + token,
                          &architecture->position, 0) ||
                !ref_compute_round(layer->compute_contract, head_query,
                                   layer->head_dimension)) {
                ref_reason(reason, "reference query norm or RoPE failed");
                goto cleanup;
            }
        }
        if (!ref_rms(raw_kv + token * kv_width, kv_width, kv_norm_weight,
                     architecture->rms_norm_epsilon) ||
            !ref_compute_round(layer->compute_contract,
                               raw_kv + token * kv_width, kv_width) ||
            !ref_rope(raw_kv + token * kv_width, kv_width,
                      architecture->rope_head_dimension,
                      opts->token_position + token,
                      &architecture->position, 0) ||
            !ref_compute_round(layer->compute_contract,
                               raw_kv + token * kv_width, kv_width) ||
            (kv_width > architecture->rope_head_dimension &&
             !ref_activation(&layer->attention_kv_activation,
                             raw_kv + token * kv_width,
                             kv_width - architecture->rope_head_dimension))) {
            ref_reason(reason, "reference KV norm, RoPE, or activation failed");
            goto cleanup;
        }
    }
    if (layer->attention_class != YVEX_ATTENTION_CLASS_SWA &&
        !ref_compressor_execute(
            session, descriptor, layer, architecture,
            YVEX_DEEPSEEK_ATTENTION_ROLLING_MAIN,
            opts->history ? &history->main_rolling_state : NULL, input,
            token_count, hidden_width, opts->token_position, &compressed,
            &compressed_positions, &compressed_count, &next_main_state,
            reason))
        goto cleanup;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA) {
        const yvex_runtime_tensor_binding *index_q = ref_binding(
            descriptor, YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B,
            layer->layer_index);
        const yvex_runtime_tensor_binding *index_w = ref_binding(
            descriptor, YVEX_TENSOR_ROLE_INDEXER_PROJECTION,
            layer->layer_index);
        unsigned long long index_query_stride;
        unsigned long long index_query_count;
        unsigned long long index_weight_count;

        if (!ref_compressor_execute(
                session, descriptor, layer, architecture,
                YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER,
                opts->history ? &history->indexer_rolling_state : NULL,
                input, token_count, hidden_width, opts->token_position,
                &indexer, &indexer_positions, &indexer_count,
                &next_indexer_state, reason) ||
            indexer_count != compressed_count || !index_q || !index_w ||
            !ref_mul(layer->indexer_heads,
                     layer->indexer_head_dimension, &index_query_stride) ||
            !ref_mul(token_count, index_query_stride, &index_query_count) ||
            !ref_mul(token_count, layer->indexer_heads,
                     &index_weight_count))
            goto cleanup;
        index_query = (float *)ref_alloc(index_query_count,
                                         sizeof(*index_query));
        index_weights = (float *)ref_alloc(index_weight_count,
                                           sizeof(*index_weights));
        if (!index_query || !index_weights ||
            !ref_matrix_batch(session, index_q, 0ull, index_query_stride,
                              q_low, token_count, q_rank, index_query,
                              index_query_stride, reason) ||
            !ref_compute_round(layer->compute_contract, index_query,
                               index_query_count) ||
            !ref_matrix_batch(session, index_w, 0ull, layer->indexer_heads,
                              input, token_count, hidden_width, index_weights,
                              layer->indexer_heads, reason) ||
            !ref_compute_round(layer->compute_contract, index_weights,
                               index_weight_count))
            goto cleanup;
        for (token = 0ull; token < token_count; ++token) {
            unsigned long long head;
            for (head = 0ull; head < layer->indexer_heads; ++head) {
                float *head_query =
                    index_query + token * index_query_stride +
                    head * layer->indexer_head_dimension;
                if (!ref_rope(head_query, layer->indexer_head_dimension,
                              architecture->rope_head_dimension,
                              opts->token_position + token,
                              &architecture->position, 0) ||
                    !ref_compute_round(layer->compute_contract, head_query,
                                       layer->indexer_head_dimension) ||
                    !ref_activation(&layer->indexer_query_activation,
                                    head_query,
                                    layer->indexer_head_dimension)) {
                    ref_reason(reason,
                               "reference CSA query activation failed");
                    goto cleanup;
                }
            }
        }
        topk_stride = history->compressed_entry_count + compressed_count;
        if (topk_stride > layer->sparse_topk.k)
            topk_stride = layer->sparse_topk.k;
        if (topk_stride) {
            unsigned long long topk_count;
            if (!ref_mul(token_count, topk_stride, &topk_count))
                goto cleanup;
            topk_counts = (unsigned long long *)ref_alloc(
                token_count, sizeof(*topk_counts));
            topk_positions = (unsigned long long *)ref_alloc(
                topk_count, sizeof(*topk_positions));
            if (!topk_counts || !topk_positions) goto cleanup;
        }
    }
    if (!ref_sparse_reduce(
            layer, architecture, query, history, raw_kv, kv_width,
            compressed, compressed_count, compressed_positions, indexer,
            indexer_count, indexer_positions, index_query, index_weights,
            sinks, token_count, opts->token_position, attention, topk_counts,
            topk_positions, topk_stride) ||
        !ref_output_projection(session, out_a, out_b, layer, architecture,
                               attention,
                               token_count, query_width, hidden_width, output,
                               reason)) {
        ref_reason(reason, "reference attention or output projection failed");
        goto cleanup;
    }

    memset(trace, 0, sizeof(*trace));
    trace->owned = 1;
    trace->complete = 1;
    trace->layer_index = layer->layer_index;
    trace->attention_class = layer->attention_class;
    trace->token_position = opts->token_position;
    trace->token_count = token_count;
    trace->hidden_width = hidden_width;
    trace->q_rank = q_rank;
    trace->query_width = query_width;
    trace->kv_width = kv_width;
    trace->compressed_count = compressed_count;
    trace->compressed_stride = compressed_count ? layer->head_dimension : 0ull;
    trace->indexer_count = indexer_count;
    trace->indexer_stride = indexer_count ? layer->indexer_head_dimension : 0ull;
    trace->index_query_stride =
        layer->attention_class == YVEX_ATTENTION_CLASS_CSA
            ? layer->indexer_heads * layer->indexer_head_dimension : 0ull;
    trace->index_weight_stride =
        layer->attention_class == YVEX_ATTENTION_CLASS_CSA
            ? layer->indexer_heads : 0ull;
    trace->topk_stride = topk_stride;
    trace->input = input;
    trace->q_low = q_low;
    trace->query = query;
    trace->raw_kv = raw_kv;
    trace->compressed_kv = compressed;
    trace->indexer_kv = indexer;
    trace->index_query = index_query;
    trace->index_weights = index_weights;
    trace->attention_values = attention;
    trace->output = output;
    trace->compressed_positions = compressed_positions;
    trace->indexer_positions = indexer_positions;
    trace->topk_counts = topk_counts;
    trace->topk_positions = topk_positions;
    trace->next_main_rolling_state = next_main_state;
    trace->next_indexer_rolling_state = next_indexer_state;
    input = q_low = query = raw_kv = compressed = indexer = NULL;
    index_query = index_weights = attention = output = NULL;
    compressed_positions = indexer_positions = NULL;
    topk_counts = topk_positions = NULL;
    memset(&next_main_state, 0, sizeof(next_main_state));
    memset(&next_indexer_state, 0, sizeof(next_indexer_state));
    ok = 1;

cleanup:
    if (!ok && (!reason || !reason[0]))
        ref_reason(reason, "reference full-equation execution failed");
    free(input);
    free(q_low);
    free(q_norm);
    free(query);
    free(raw_kv);
    free(kv_norm_weight);
    free(sinks);
    free(compressed);
    free(compressed_positions);
    free(indexer);
    free(indexer_positions);
    free(index_query);
    free(index_weights);
    free(attention);
    free(output);
    free(topk_counts);
    free(topk_positions);
    free(next_main_state.kv_state);
    free(next_main_state.score_state);
    free(next_indexer_state.kv_state);
    free(next_indexer_state.score_state);
    return ok;
}

static int ref_compare_stage(
    const char *stage,
    yvex_test_attention_reference_stage stage_id,
    const float *production,
    const float *reference,
    unsigned long long count,
    const yvex_test_attention_reference_contract *contract,
    yvex_test_attention_reference_metrics *metrics)
{
    unsigned long long i;
    double absolute_tolerance;
    double relative_tolerance;

    if (!contract || stage_id >= YVEX_TEST_ATTENTION_STAGE_COUNT)
        return 0;
    absolute_tolerance = contract->absolute[stage_id];
    relative_tolerance = contract->relative[stage_id];

    if (count && (!production || !reference)) {
        if (!metrics->first_failed_stage) metrics->first_failed_stage = stage;
        metrics->first_failed_index = 0ull;
        return 0;
    }
    for (i = 0ull; i < count; ++i) {
        double left = production[i];
        double right = reference[i];
        double absolute = fabs(left - right);
        double denominator = fmax(fabs(left), fabs(right));
        double relative = denominator > 0.0 ? absolute / denominator : 0.0;
        double allowed = absolute_tolerance + relative_tolerance * denominator;
        metrics->compared_values++;
        metrics->squared_error_sum += absolute * absolute;
        if (absolute > metrics->maximum_absolute_error)
            metrics->maximum_absolute_error = absolute;
        if (relative > metrics->maximum_relative_error)
            metrics->maximum_relative_error = relative;
        if (!isfinite(left) || !isfinite(right) || absolute > allowed) {
            if (!metrics->first_failed_stage) {
                metrics->first_failed_stage = stage;
                metrics->first_failed_index = i;
            }
            return 0;
        }
    }
    return 1;
}

static int ref_compare_rolling_state(
    const char *stage,
    yvex_test_attention_reference_stage stage_id,
    const yvex_attention_rolling_state_output *production,
    const yvex_attention_rolling_state_output *reference,
    const yvex_test_attention_reference_contract *contract,
    yvex_test_attention_reference_metrics *metrics)
{
    unsigned long long i;

    if (!production || !reference ||
        production->present != reference->present) {
        metrics->first_failed_stage = stage;
        metrics->first_failed_index = 0ull;
        return 0;
    }
    if (!production->present) return 1;
    if (production->schema_version != reference->schema_version ||
        production->kind != reference->kind ||
        production->attention_class != reference->attention_class ||
        production->layer_index != reference->layer_index ||
        production->next_token_position != reference->next_token_position ||
        production->ratio != reference->ratio ||
        production->head_dimension != reference->head_dimension ||
        production->state_width != reference->state_width ||
        production->state_slots != reference->state_slots ||
        production->previous_fill != reference->previous_fill ||
        production->current_fill != reference->current_fill ||
        production->cursor != reference->cursor ||
        production->kv_state_stride != reference->kv_state_stride ||
        production->score_state_stride != reference->score_state_stride ||
        production->kv_state_extent != reference->kv_state_extent ||
        production->score_state_extent != reference->score_state_extent ||
        production->overlap != reference->overlap ||
        production->rotated != reference->rotated ||
        memcmp(production->attention_plan_identity,
               reference->attention_plan_identity,
               sizeof(production->attention_plan_identity)) != 0 ||
        !production->kv_state || !reference->kv_state ||
        !production->score_state || !reference->score_state) {
        metrics->first_failed_stage = stage;
        metrics->first_failed_index = 0ull;
        return 0;
    }
    if (!ref_compare_stage(stage, stage_id, production->kv_state,
                           reference->kv_state,
                           production->kv_state_extent, contract, metrics))
        return 0;
    for (i = 0ull; i < production->score_state_extent; ++i) {
        double left = production->score_state[i];
        double right = reference->score_state[i];

        metrics->compared_values++;
        if (!isfinite(left) || !isfinite(right)) {
            if (left == right) continue;
            metrics->first_failed_stage = stage;
            metrics->first_failed_index = i;
            return 0;
        }
        {
            double absolute = fabs(left - right);
            double denominator = fmax(fabs(left), fabs(right));
            double relative = denominator > 0.0
                ? absolute / denominator : 0.0;
            double allowed = contract->absolute[stage_id] +
                contract->relative[stage_id] * denominator;
            metrics->squared_error_sum += absolute * absolute;
            if (absolute > metrics->maximum_absolute_error)
                metrics->maximum_absolute_error = absolute;
            if (relative > metrics->maximum_relative_error)
                metrics->maximum_relative_error = relative;
            if (absolute > allowed) {
                metrics->first_failed_stage = stage;
                metrics->first_failed_index = i;
                return 0;
            }
        }
    }
    return 1;
}

int yvex_test_attention_reference_compare_contract(
    const yvex_attention_execution_trace *production,
    const yvex_attention_execution_trace *reference,
    const yvex_test_attention_reference_contract *contract,
    yvex_test_attention_reference_metrics *metrics)
{
    unsigned long long count;
    unsigned long long i;
    unsigned int stage;

    if (!metrics) return 0;
    memset(metrics, 0, sizeof(*metrics));
    metrics->first_failed_index = ~0ull;
    if (!contract ||
        contract->schema_version != YVEX_TEST_ATTENTION_EVIDENCE_SCHEMA_V2 ||
        !contract->path_name || !contract->compressed_positions_exact ||
        !contract->indexer_positions_exact || !contract->topk_positions_exact)
        return 0;
    for (stage = 0u; stage < YVEX_TEST_ATTENTION_STAGE_COUNT; ++stage) {
        if (!isfinite(contract->absolute[stage]) ||
            !isfinite(contract->relative[stage]) ||
            contract->absolute[stage] < 0.0 ||
            contract->relative[stage] < 0.0)
            return 0;
    }
    if (!production || !reference || !production->owned || !reference->owned ||
        !production->complete || !reference->complete ||
        production->evidence_level > YVEX_ATTENTION_EVIDENCE_FULL ||
        production->layer_index != reference->layer_index ||
        production->attention_class != reference->attention_class ||
        production->token_position != reference->token_position ||
        production->token_count != reference->token_count ||
        production->hidden_width != reference->hidden_width ||
        production->q_rank != reference->q_rank ||
        production->query_width != reference->query_width ||
        production->kv_width != reference->kv_width ||
        production->compressed_count != reference->compressed_count ||
        production->compressed_stride != reference->compressed_stride ||
        production->indexer_count != reference->indexer_count ||
        production->indexer_stride != reference->indexer_stride ||
        production->index_query_stride != reference->index_query_stride ||
        production->index_weight_stride != reference->index_weight_stride ||
        production->topk_stride != reference->topk_stride) {
        metrics->first_failed_stage = "geometry";
        return 0;
    }
#define REF_STAGE(name_, stage_, left_, right_, count_)                      \
    do {                                                                     \
        if (!ref_compare_stage((name_), (stage_), (left_), (right_),         \
                               (count_), contract, metrics))                 \
            return 0;                                                        \
    } while (0)
    if (production->evidence_level == YVEX_ATTENTION_EVIDENCE_FULL) {
        count = production->token_count * production->hidden_width;
        REF_STAGE("input", YVEX_TEST_ATTENTION_STAGE_INPUT,
                  production->input, reference->input, count);
    }
    if (production->evidence_level >= YVEX_ATTENTION_EVIDENCE_STAGES) {
        count = production->token_count * production->q_rank;
        REF_STAGE("q-low", YVEX_TEST_ATTENTION_STAGE_Q_LOW,
                  production->q_low, reference->q_low, count);
        count = production->token_count * production->query_width;
        REF_STAGE("query", YVEX_TEST_ATTENTION_STAGE_QUERY,
                  production->query, reference->query, count);
    }
    count = production->token_count * production->kv_width;
    REF_STAGE("raw-kv", YVEX_TEST_ATTENTION_STAGE_RAW_KV,
              production->raw_kv, reference->raw_kv, count);
    count = production->compressed_count * production->compressed_stride;
    REF_STAGE("compressed-kv", YVEX_TEST_ATTENTION_STAGE_COMPRESSED_KV,
              production->compressed_kv,
              reference->compressed_kv, count);
    count = production->indexer_count * production->indexer_stride;
    REF_STAGE("indexer-kv", YVEX_TEST_ATTENTION_STAGE_INDEXER_KV,
              production->indexer_kv,
              reference->indexer_kv, count);
    if (production->evidence_level >= YVEX_ATTENTION_EVIDENCE_STAGES) {
        count = production->token_count * production->index_query_stride;
        REF_STAGE("index-query", YVEX_TEST_ATTENTION_STAGE_INDEX_QUERY,
                  production->index_query,
                  reference->index_query, count);
        count = production->token_count * production->index_weight_stride;
        REF_STAGE("index-weights", YVEX_TEST_ATTENTION_STAGE_INDEX_WEIGHTS,
                  production->index_weights,
                  reference->index_weights, count);
        count = production->token_count * production->query_width;
        REF_STAGE("attention", YVEX_TEST_ATTENTION_STAGE_ATTENTION,
                  production->attention_values,
                  reference->attention_values, count);
    }
    count = production->token_count * production->hidden_width;
    REF_STAGE("output", YVEX_TEST_ATTENTION_STAGE_OUTPUT,
              production->output, reference->output, count);
#undef REF_STAGE
    if (!ref_compare_rolling_state(
            "main-rolling-state", YVEX_TEST_ATTENTION_STAGE_MAIN_STATE,
            &production->next_main_rolling_state,
            &reference->next_main_rolling_state, contract, metrics) ||
        !ref_compare_rolling_state(
            "indexer-rolling-state", YVEX_TEST_ATTENTION_STAGE_INDEXER_STATE,
            &production->next_indexer_rolling_state,
            &reference->next_indexer_rolling_state, contract, metrics))
        return 0;
    for (i = 0ull; i < production->compressed_count; ++i) {
        if (production->compressed_positions[i] !=
            reference->compressed_positions[i]) {
            metrics->discrete_mismatches++;
            metrics->first_failed_stage = "compressed-positions";
            metrics->first_failed_index = i;
            return 0;
        }
    }
    for (i = 0ull; i < production->indexer_count; ++i) {
        if (production->indexer_positions[i] !=
            reference->indexer_positions[i]) {
            metrics->discrete_mismatches++;
            metrics->first_failed_stage = "indexer-positions";
            metrics->first_failed_index = i;
            return 0;
        }
    }
    for (i = 0ull;
         production->evidence_level == YVEX_ATTENTION_EVIDENCE_FULL &&
         i < production->token_count;
         ++i) {
        unsigned long long j;
        if (production->topk_stride &&
            production->topk_counts[i] != reference->topk_counts[i]) {
            metrics->discrete_mismatches++;
            metrics->first_failed_stage = "topk-count";
            metrics->first_failed_index = i;
            return 0;
        }
        for (j = 0ull; j < production->topk_stride; ++j) {
            if (production->topk_positions[
                    i * production->topk_stride + j] !=
                reference->topk_positions[
                    i * reference->topk_stride + j]) {
                metrics->discrete_mismatches++;
                metrics->first_failed_stage = "topk-order";
                metrics->first_failed_index =
                    i * production->topk_stride + j;
                return 0;
            }
        }
    }
    return 1;
}

int yvex_test_attention_reference_compare(
    const yvex_attention_execution_trace *production,
    const yvex_attention_execution_trace *reference,
    double absolute_tolerance,
    double relative_tolerance,
    yvex_test_attention_reference_metrics *metrics)
{
    yvex_test_attention_reference_contract contract;
    unsigned int stage;

    memset(&contract, 0, sizeof(contract));
    contract.schema_version = YVEX_TEST_ATTENTION_EVIDENCE_SCHEMA_V2;
    contract.path_name = "legacy-uniform";
    contract.compressed_positions_exact = 1;
    contract.indexer_positions_exact = 1;
    contract.topk_positions_exact = 1;
    for (stage = 0u; stage < YVEX_TEST_ATTENTION_STAGE_COUNT; ++stage) {
        contract.absolute[stage] = absolute_tolerance;
        contract.relative[stage] = relative_tolerance;
    }
    return yvex_test_attention_reference_compare_contract(
        production, reference, &contract, metrics);
}

static int ref_hash_finish(yvex_sha256 *hash, char identity[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    if (!hash || !identity || !yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, identity);
    return 1;
}

static int ref_hash_float_array(yvex_sha256 *hash, const float *values,
                                unsigned long long count)
{
    unsigned long long i;

    if (!hash || (count && !values) || !yvex_sha256_update_u64(hash, count))
        return 0;
    for (i = 0ull; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, values + i, sizeof(bits));
        if (!yvex_sha256_update_u64(hash, (unsigned long long)bits)) return 0;
    }
    return 1;
}

static int ref_hash_u64_array(yvex_sha256 *hash,
                              const unsigned long long *values,
                              unsigned long long count)
{
    unsigned long long i;

    if (!hash || (count && !values) || !yvex_sha256_update_u64(hash, count))
        return 0;
    for (i = 0ull; i < count; ++i) {
        if (!yvex_sha256_update_u64(hash, values[i])) return 0;
    }
    return 1;
}

static const yvex_test_attention_external_vectors ref_external_vectors = {
    YVEX_TEST_ATTENTION_EXTERNAL_SCHEMA_V1,
    "arXiv:2606.19348v1",
    "sections-2.3.1-2.3.4;equations-9-27;section-4.2.1;section-5.2.1",
    "96a04cb13f9c3ed86028e090784a9eb059cf5318",
    "python/sglang/srt/layers/attention/dsv4/compressor.py;"
    "python/sglang/srt/layers/attention/dsv4/indexer.py;"
    "python/sglang/srt/layers/attention/dsv4/deepseek_v4_rope.py;"
    "python/sglang/srt/layers/mhc.py",
    "8df14cfc8c8a09b4e57f082e59593a3abce4ffb3",
    "vllm/models/deepseek_v4/attention.py;"
    "vllm/models/deepseek_v4/common/rope.py;"
    "vllm/models/deepseek_v4/compressor.py;"
    "vllm/model_executor/kernels/mhc/torch.py;"
    "vllm/v1/attention/backends/mla/indexer.py",
    "e7706faf8d1c3b9f241e36860640ad1dac644ede",
    "fast_hadamard_transform/fast_hadamard_transform_interface.py;"
    "tests/test_fast_hadamard_transform.py",
    {1.0f, -2.0f, 0.5f},
    {-0.25f, 1.75f, -0.75f},
    0.5f,
    {10.0f, 20.0f, 30.0f, 40.0f, 1.0f, 2.0f},
    {10.0f, 20.0f, 30.0f, 40.0f, -1.1426396369934082f,
     1.922075629234314f},
    {10.0f, 20.0f, 30.0f, 40.0f, 2.2232441902160645f,
     0.23913362622261047f},
    1ull, 131ull, 128ull,
    {0x3f80u, 0x3f82u, 0x8000u},
    {1.00390625f, 1.01171875f, -0.0f},
    {1.0f, 1.015625f, -0.0f},
    {0.0f, 0.49f, 1.01f, -1.51f, 3.99f, -8.0f},
    {0x00u, 0x00u, 0x01u, 0x0au, 0x04u, 0x0eu},
    0x80u,
    {0.0f, 0.0f, 1.0f, -2.0f, 4.0f, -8.0f},
    {0.0f, 1.0f, -2.0f, 448.0f, -512.0f},
    {0x00u, 0x30u, 0xb8u, 0x76u, 0xf8u},
    0x80u,
    {0.0f, 1.0f, -2.0f, 448.0f, -512.0f},
    {0.0f, 1.0986123085021973f, 0.6931471824645996f,
     1.3862943649291992f},
    {1.0f, 5.0f, 7.0f, 11.0f},
    7.4f,
    {0.0f, 0.6931471824645996f, 1.0986123085021973f,
     1.3862943649291992f},
    {1.0f, 5.0f, 7.0f, 11.0f},
    7.6f,
    {0.25f, 4.0f, 1.5f, 5.0f, -2.0f, 3.0f},
    {40ull, 12ull, 32ull, 8ull, 44ull, 16ull},
    {3ull, 1ull, 5ull, 2ull},
    128ull,
    {127ull, 128ull, 129ull, 384ull},
    {0ull, 1ull, 1ull, 3ull},
    {127ull, 0ull, 1ull, 0ull},
    {0.0f, 0.6931471824645996f},
    {1.0f, 2.0f, 3.0f, 4.0f},
    0.0f,
    {1.75f, 2.5f},
    {0.5f, -0.25f},
    {1.0f, 2.0f, 3.0f, 4.0f},
    {2.0f, 1.0f},
    {1.0f, 2.0f, 3.0f, 4.0f},
    {11.0f, 13.5f, 14.5f, 19.75f}
};

/* Expose immutable external literals without transferring ownership. */
const yvex_test_attention_external_vectors *
yvex_test_attention_external_vectors_get(void)
{
    return &ref_external_vectors;
}

static int ref_hash_bytes(yvex_sha256 *hash, const unsigned char *values,
                          unsigned long long count)
{
    unsigned long long index;

    if (!hash || (count && !values) || !yvex_sha256_update_u64(hash, count))
        return 0;
    for (index = 0ull; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, values[index])) return 0;
    return 1;
}

static int ref_hash_shorts(yvex_sha256 *hash, const unsigned short *values,
                           unsigned long long count)
{
    unsigned long long index;

    if (!hash || (count && !values) || !yvex_sha256_update_u64(hash, count))
        return 0;
    for (index = 0ull; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, values[index])) return 0;
    return 1;
}

static int ref_external_compress(const float *logits, const float *values,
                                 unsigned long long count, float *result)
{
    unsigned long long index;
    double maximum = -INFINITY;
    double denominator = 0.0;
    double numerator = 0.0;

    if (!logits || !values || !count || !result) return 0;
    for (index = 0ull; index < count; ++index) {
        if (!isfinite(logits[index]) || !isfinite(values[index])) return 0;
        if (logits[index] > maximum) maximum = logits[index];
    }
    for (index = 0ull; index < count; ++index) {
        double weight = exp((double)logits[index] - maximum);
        denominator += weight;
        numerator += weight * (double)values[index];
    }
    if (!(denominator > 0.0) || !isfinite(numerator)) return 0;
    *result = (float)(numerator / denominator);
    return isfinite(*result);
}

/* Build a provenance-bound identity over every external literal. */
static int ref_external_identity(char identity[YVEX_SHA256_HEX_CAP])
{
    const yvex_test_attention_external_vectors *v = &ref_external_vectors;
    yvex_sha256 hash;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.attention.external-conformance.v1") ||
        !yvex_sha256_update_u64(&hash, v->schema_version) ||
        !yvex_sha256_update_text(&hash, v->paper_revision) ||
        !yvex_sha256_update_text(&hash, v->paper_locator) ||
        !yvex_sha256_update_text(&hash, v->sglang_revision) ||
        !yvex_sha256_update_text(&hash, v->sglang_locator) ||
        !yvex_sha256_update_text(&hash, v->vllm_revision) ||
        !yvex_sha256_update_text(&hash, v->vllm_locator) ||
        !yvex_sha256_update_text(&hash, v->hadamard_revision) ||
        !yvex_sha256_update_text(&hash, v->hadamard_locator) ||
        !ref_hash_float_array(&hash, v->hadamard_input, 3ull) ||
        !ref_hash_float_array(&hash, v->hadamard_expected, 3ull) ||
        !ref_hash_float_array(&hash, &v->hadamard_scale, 1ull) ||
        !ref_hash_float_array(&hash, v->position_input, 6ull) ||
        !ref_hash_float_array(&hash, v->position_forward_expected, 6ull) ||
        !ref_hash_float_array(&hash, v->position_inverse_expected, 6ull) ||
        !yvex_sha256_update_u64(&hash, v->position) ||
        !yvex_sha256_update_u64(&hash, v->compressed_source_position) ||
        !yvex_sha256_update_u64(&hash, v->compressed_position) ||
        !ref_hash_shorts(&hash, v->bf16_expected_codes, 3ull) ||
        !ref_hash_float_array(&hash, v->bf16_input, 3ull) ||
        !ref_hash_float_array(&hash, v->bf16_expected, 3ull) ||
        !ref_hash_float_array(&hash, v->fp4_input, 6ull) ||
        !ref_hash_bytes(&hash, v->fp4_expected_codes, 6ull) ||
        !yvex_sha256_update_u64(&hash, v->fp4_expected_scale) ||
        !ref_hash_float_array(&hash, v->fp4_expected, 6ull) ||
        !ref_hash_float_array(&hash, v->fp8_input, 5ull) ||
        !ref_hash_bytes(&hash, v->fp8_expected_codes, 5ull) ||
        !yvex_sha256_update_u64(&hash, v->fp8_expected_scale) ||
        !ref_hash_float_array(&hash, v->fp8_expected, 5ull) ||
        !ref_hash_float_array(&hash, v->csa_compress_logits, 4ull) ||
        !ref_hash_float_array(&hash, v->csa_compress_values, 4ull) ||
        !ref_hash_float_array(&hash, &v->csa_compress_expected, 1ull) ||
        !ref_hash_float_array(&hash, v->hca_compress_logits, 4ull) ||
        !ref_hash_float_array(&hash, v->hca_compress_values, 4ull) ||
        !ref_hash_float_array(&hash, &v->hca_compress_expected, 1ull) ||
        !ref_hash_float_array(&hash, v->topk_scores, 6ull) ||
        !ref_hash_u64_array(&hash, v->topk_positions, 6ull) ||
        !ref_hash_u64_array(&hash, v->topk_expected, 4ull) ||
        !yvex_sha256_update_u64(&hash, v->hca_ratio) ||
        !ref_hash_u64_array(&hash, v->hca_tokens, 4ull) ||
        !ref_hash_u64_array(&hash, v->hca_emissions, 4ull) ||
        !ref_hash_u64_array(&hash, v->hca_tails, 4ull) ||
        !ref_hash_float_array(&hash, v->reduction_logits, 2ull) ||
        !ref_hash_float_array(&hash, v->reduction_values, 4ull) ||
        !ref_hash_float_array(&hash, &v->reduction_sink_logit, 1ull) ||
        !ref_hash_float_array(&hash, v->reduction_expected, 2ull) ||
        !ref_hash_float_array(&hash, v->mhc_core, 2ull) ||
        !ref_hash_float_array(&hash, v->mhc_residual, 4ull) ||
        !ref_hash_float_array(&hash, v->mhc_post, 2ull) ||
        !ref_hash_float_array(&hash, v->mhc_combination, 4ull) ||
        !ref_hash_float_array(&hash, v->mhc_expected, 4ull))
        return 0;
    return ref_hash_finish(&hash, identity);
}

int yvex_test_attention_external_conformance_validate(
    yvex_test_attention_external_summary *summary)
{
    const yvex_test_attention_external_vectors *v = &ref_external_vectors;
    yvex_attention_position_policy position = {0};
    float actual[8];
    float compressed;
    unsigned long long selected[4], selected_count = 0ull, index;
    char identity[YVEX_SHA256_HEX_CAP];
    double maximum, denominator;

    if (v->schema_version != YVEX_TEST_ATTENTION_EXTERNAL_SCHEMA_V1 ||
        strcmp(v->paper_revision, "arXiv:2606.19348v1") != 0 ||
        strcmp(v->sglang_revision,
               "96a04cb13f9c3ed86028e090784a9eb059cf5318") != 0 ||
        strcmp(v->vllm_revision,
               "8df14cfc8c8a09b4e57f082e59593a3abce4ffb3") != 0 ||
        strcmp(v->hadamard_revision,
               "e7706faf8d1c3b9f241e36860640ad1dac644ede") != 0 ||
        !ref_external_identity(identity))
        return 0;
    if (!yvex_test_attention_reference_hadamard(
            v->hadamard_input, 3ull, v->hadamard_scale, 1, actual) ||
        memcmp(actual, v->hadamard_expected, 3u * sizeof(float)) != 0)
        return 0;
    position.rope_dimension = 2ull;
    position.theta = 10000ull;
    position.partial_rope = 1;
    position.inverse_output_rotation = 1;
    memcpy(actual, v->position_input, 6u * sizeof(float));
    if (!ref_rope(actual, 6ull, 2ull, v->position, &position, 0)) return 0;
    for (index = 0ull; index < 6ull; ++index)
        if (fabsf(actual[index] - v->position_forward_expected[index]) >
            1.0e-6f) return 0;
    memcpy(actual, v->position_input, 6u * sizeof(float));
    if (!ref_rope(actual, 6ull, 2ull, v->position, &position, 1)) return 0;
    for (index = 0ull; index < 6ull; ++index)
        if (fabsf(actual[index] - v->position_inverse_expected[index]) >
            1.0e-6f) return 0;
    if (v->compressed_source_position / 128ull * 128ull !=
        v->compressed_position) return 0;
    for (index = 0ull; index < 3ull; ++index)
        if (ref_bf16_encode(v->bf16_input[index]) != v->bf16_expected_codes[index] ||
            memcmp(&v->bf16_expected[index],
                   &(float){ref_bf16_decode(v->bf16_expected_codes[index])},
                   sizeof(float)) != 0)
            return 0;
    memcpy(actual, v->fp4_input, 6u * sizeof(float));
    if (!ref_fake_quant_block(actual, 6ull, 1) ||
        memcmp(actual, v->fp4_expected, 6u * sizeof(float)) != 0 ||
        ref_ue8m0_encode(2.0f) != v->fp4_expected_scale)
        return 0;
    for (index = 0ull; index < 6ull; ++index)
        if (ref_fp4_encode(v->fp4_input[index] / 2.0f) !=
            v->fp4_expected_codes[index]) return 0;
    memcpy(actual, v->fp8_input, 5u * sizeof(float));
    if (!ref_fake_quant_block(actual, 5ull, 0) ||
        memcmp(actual, v->fp8_expected, 5u * sizeof(float)) != 0 ||
        ref_ue8m0_encode(2.0f) != v->fp8_expected_scale)
        return 0;
    for (index = 0ull; index < 5ull; ++index)
        if (ref_fp8_encode(v->fp8_input[index] / 2.0f) !=
            v->fp8_expected_codes[index]) return 0;
    if (!ref_external_compress(v->csa_compress_logits,
                               v->csa_compress_values, 4ull, &compressed) ||
        fabsf(compressed - v->csa_compress_expected) > 1.0e-6f ||
        !ref_external_compress(v->hca_compress_logits,
                               v->hca_compress_values, 4ull, &compressed) ||
        fabsf(compressed - v->hca_compress_expected) > 1.0e-6f ||
        !yvex_test_attention_reference_topk(
            v->topk_scores, v->topk_positions, 6ull, 4ull,
            selected, &selected_count) || selected_count != 4ull ||
        memcmp(selected, v->topk_expected, 4u * sizeof(*selected)) != 0)
        return 0;
    for (index = 0ull; index < 4ull; ++index)
        if (v->hca_tokens[index] / v->hca_ratio != v->hca_emissions[index] ||
            v->hca_tokens[index] % v->hca_ratio != v->hca_tails[index])
            return 0;
    maximum = v->reduction_sink_logit;
    for (index = 0ull; index < 2ull; ++index)
        if ((double)v->reduction_logits[index] > maximum)
            maximum = v->reduction_logits[index];
    denominator = exp((double)v->reduction_sink_logit - maximum);
    for (index = 0ull; index < 2ull; ++index)
        denominator += exp((double)v->reduction_logits[index] - maximum);
    for (index = 0ull; index < 2ull; ++index) {
        double value = exp((double)v->reduction_logits[0] - maximum) *
                           (double)v->reduction_values[index] +
                       exp((double)v->reduction_logits[1] - maximum) *
                           (double)v->reduction_values[2ull + index];
        if (fabs(value / denominator - (double)v->reduction_expected[index]) >
            1.0e-7) return 0;
    }
    for (index = 0ull; index < 2ull; ++index) {
        unsigned long long source, lane;
        for (lane = 0ull; lane < 2ull; ++lane) {
            double value = (double)v->mhc_post[index] * v->mhc_core[lane];
            for (source = 0ull; source < 2ull; ++source)
                value += (double)v->mhc_combination[source * 2ull + index] *
                         (double)v->mhc_residual[source * 2ull + lane];
            if ((float)value != v->mhc_expected[index * 2ull + lane]) return 0;
        }
    }
    if (summary) {
        memset(summary, 0, sizeof(*summary));
        summary->schema_version = v->schema_version;
        memcpy(summary->vector_identity, identity, sizeof(summary->vector_identity));
        summary->vector_count = 11ull;
        summary->provenance_complete = 1;
        summary->position_policy_exact = 1;
        summary->bf16_publication_exact = 1;
        summary->fp8_fake_quant_exact = 1;
        summary->fp4_fake_quant_exact = 1;
        summary->compressor_transition_exact = 1;
        summary->csa_topk_order_exact = 1;
        summary->hca_ratio_exact = 1;
        summary->local_compressed_reduction_exact = 1;
        summary->envelope_mhc_exact = 1;
        summary->hadamard_exact = 1;
    }
    return 1;
}

static int ref_hash_contract(
    const yvex_test_attention_reference_contract *contract,
    char identity[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned int index;

    if (!contract || !identity ||
        contract->schema_version != YVEX_TEST_ATTENTION_EVIDENCE_SCHEMA_V2 ||
        !contract->path_name)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.attention.reference.comparison.v2") ||
        !yvex_sha256_update_u64(&hash, contract->schema_version) ||
        !yvex_sha256_update_text(&hash, contract->path_name))
        return 0;
    for (index = 0u; index < YVEX_TEST_ATTENTION_STAGE_COUNT; ++index) {
        uint64_t absolute_bits;
        uint64_t relative_bits;

        if (!isfinite(contract->absolute[index]) ||
            !isfinite(contract->relative[index]) ||
            contract->absolute[index] < 0.0 ||
            contract->relative[index] < 0.0)
            return 0;
        memcpy(&absolute_bits, contract->absolute + index,
               sizeof(absolute_bits));
        memcpy(&relative_bits, contract->relative + index,
               sizeof(relative_bits));
        if (!yvex_sha256_update_u64(&hash, index) ||
            !yvex_sha256_update_u64(&hash, absolute_bits) ||
            !yvex_sha256_update_u64(&hash, relative_bits))
            return 0;
    }
    for (index = 0u; index < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP; ++index) {
        if (!yvex_sha256_update_u64(&hash, index) ||
            !yvex_sha256_update_u64(
                &hash, contract->qtype_binding_counts[index]))
            return 0;
    }
    if (!yvex_sha256_update_u64(
            &hash, (unsigned long long)contract->compressed_positions_exact) ||
        !yvex_sha256_update_u64(
            &hash, (unsigned long long)contract->indexer_positions_exact) ||
        !yvex_sha256_update_u64(
            &hash, (unsigned long long)contract->topk_positions_exact))
        return 0;
    return ref_hash_finish(&hash, identity);
}

static int ref_hash_rolling_output(
    yvex_sha256 *hash, const yvex_attention_rolling_state_output *state)
{
    if (!hash || !state ||
        !yvex_sha256_update_u64(hash, (unsigned long long)state->present))
        return 0;
    if (!state->present) return 1;
    return yvex_sha256_update_u64(hash, state->schema_version) &&
           yvex_sha256_update_u64(hash, (unsigned long long)state->kind) &&
           yvex_sha256_update_u64(
               hash, (unsigned long long)state->attention_class) &&
           yvex_sha256_update_u64(hash, state->layer_index) &&
           yvex_sha256_update_u64(hash, state->next_token_position) &&
           yvex_sha256_update_u64(hash, state->ratio) &&
           yvex_sha256_update_u64(hash, state->head_dimension) &&
           yvex_sha256_update_u64(hash, state->state_width) &&
           yvex_sha256_update_u64(hash, state->state_slots) &&
           yvex_sha256_update_u64(hash, state->previous_fill) &&
           yvex_sha256_update_u64(hash, state->current_fill) &&
           yvex_sha256_update_u64(hash, state->cursor) &&
           yvex_sha256_update_u64(hash, state->kv_state_stride) &&
           yvex_sha256_update_u64(hash, state->score_state_stride) &&
           yvex_sha256_update_u64(hash, state->kv_state_extent) &&
           yvex_sha256_update_u64(hash, state->score_state_extent) &&
           yvex_sha256_update_u64(hash, (unsigned long long)state->overlap) &&
           yvex_sha256_update_u64(hash, (unsigned long long)state->rotated) &&
           yvex_sha256_update_text(hash, state->attention_plan_identity) &&
           ref_hash_float_array(hash, state->kv_state,
                                state->kv_state_extent) &&
           ref_hash_float_array(hash, state->score_state,
                                state->score_state_extent);
}

static int ref_hash_rolling_view(
    yvex_sha256 *hash, const yvex_attention_rolling_state_view *state)
{
    yvex_attention_rolling_state_output projected;

    if (!hash || !state) return 0;
    memset(&projected, 0, sizeof(projected));
    projected.present = state->present;
    projected.schema_version = state->schema_version;
    projected.kind = state->kind;
    projected.attention_class = state->attention_class;
    projected.layer_index = state->layer_index;
    projected.next_token_position = state->next_token_position;
    projected.ratio = state->ratio;
    projected.head_dimension = state->head_dimension;
    projected.state_width = state->state_width;
    projected.state_slots = state->state_slots;
    projected.previous_fill = state->previous_fill;
    projected.current_fill = state->current_fill;
    projected.cursor = state->cursor;
    projected.kv_state_stride = state->kv_state_stride;
    projected.score_state_stride = state->score_state_stride;
    projected.kv_state_extent = state->kv_state_extent;
    projected.score_state_extent = state->score_state_extent;
    projected.kv_state = (float *)state->kv_state;
    projected.score_state = (float *)state->score_state;
    projected.overlap = state->overlap;
    projected.rotated = state->rotated;
    memcpy(projected.attention_plan_identity, state->attention_plan_identity,
           sizeof(projected.attention_plan_identity));
    return ref_hash_rolling_output(hash, &projected);
}

static int ref_hash_history(yvex_sha256 *hash,
                            const yvex_attention_history_view *history)
{
    unsigned long long extent;

    if (!hash ||
        !yvex_sha256_update_u64(hash, history ? 1ull : 0ull))
        return 0;
    if (!history) return 1;
    if (!ref_mul(history->local_tail_count, history->local_kv_stride,
                 &extent) ||
        !yvex_sha256_update_u64(hash, history->token_count) ||
        !yvex_sha256_update_u64(hash, history->local_tail_count) ||
        !yvex_sha256_update_u64(hash, history->local_kv_stride) ||
        !ref_hash_float_array(hash, history->local_kv, extent) ||
        !ref_hash_u64_array(hash, history->local_positions,
                            history->local_tail_count))
        return 0;
    if (!ref_mul(history->compressed_entry_count,
                 history->compressed_kv_stride, &extent) ||
        !yvex_sha256_update_u64(hash, history->compressed_entry_count) ||
        !yvex_sha256_update_u64(hash, history->compressed_kv_stride) ||
        !ref_hash_float_array(hash, history->compressed_kv, extent) ||
        !ref_hash_u64_array(hash, history->compressed_positions,
                            history->compressed_entry_count))
        return 0;
    if (!ref_mul(history->indexer_entry_count, history->indexer_kv_stride,
                 &extent) ||
        !yvex_sha256_update_u64(hash, history->indexer_entry_count) ||
        !yvex_sha256_update_u64(hash, history->indexer_kv_stride) ||
        !ref_hash_float_array(hash, history->indexer_kv, extent) ||
        !ref_hash_u64_array(hash, history->indexer_positions,
                            history->indexer_entry_count) ||
        !yvex_sha256_update_u64(hash, (unsigned long long)history->immutable) ||
        !ref_hash_rolling_view(hash, &history->main_rolling_state) ||
        !ref_hash_rolling_view(hash, &history->indexer_rolling_state))
        return 0;
    return 1;
}

int yvex_test_attention_reference_evidence_build(
    const yvex_attention_execution_trace *reference,
    const yvex_attention_history_view *history,
    const char *attention_plan_identity,
    const yvex_test_attention_reference_contract *contract,
    yvex_test_attention_reference_evidence *evidence)
{
    yvex_sha256 hash;
    unsigned long long extent;
    unsigned long long token;

    if (!evidence) return 0;
    memset(evidence, 0, sizeof(*evidence));
    if (!reference || !reference->owned || !reference->complete ||
        !attention_plan_identity || !yvex_sha256_hex_valid(
            attention_plan_identity) ||
        !ref_hash_contract(contract, evidence->comparison_contract_identity))
        return 0;
    evidence->schema_version = YVEX_TEST_ATTENTION_EVIDENCE_SCHEMA_V2;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.attention.reference.fixture.v2") ||
        !yvex_sha256_update_text(&hash, attention_plan_identity) ||
        !yvex_sha256_update_u64(&hash, reference->layer_index) ||
        !yvex_sha256_update_u64(
            &hash, (unsigned long long)reference->attention_class) ||
        !yvex_sha256_update_u64(&hash, reference->token_position) ||
        !yvex_sha256_update_u64(&hash, reference->token_count) ||
        !yvex_sha256_update_u64(&hash, reference->hidden_width) ||
        !ref_mul(reference->token_count, reference->hidden_width, &extent) ||
        !ref_hash_float_array(&hash, reference->input, extent) ||
        !ref_hash_finish(&hash, evidence->fixture_identity))
        return 0;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.attention.reference.history.v2") ||
        !ref_hash_history(&hash, history) ||
        !ref_hash_finish(&hash, evidence->history_identity))
        return 0;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.attention.reference.output.v2") ||
        !yvex_sha256_update_u64(&hash, reference->layer_index) ||
        !yvex_sha256_update_u64(&hash, reference->token_position) ||
        !yvex_sha256_update_u64(&hash, reference->token_count) ||
        !yvex_sha256_update_u64(&hash, reference->hidden_width) ||
        !ref_mul(reference->token_count, reference->hidden_width, &extent) ||
        !ref_hash_float_array(&hash, reference->output, extent) ||
        !ref_hash_finish(&hash, evidence->oracle_output_identity))
        return 0;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.attention.reference.trace.v2") ||
        !yvex_sha256_update_text(&hash, evidence->fixture_identity) ||
        !yvex_sha256_update_text(&hash, evidence->history_identity) ||
        !yvex_sha256_update_u64(&hash, reference->q_rank) ||
        !yvex_sha256_update_u64(&hash, reference->query_width) ||
        !yvex_sha256_update_u64(&hash, reference->kv_width) ||
        !yvex_sha256_update_u64(&hash, reference->compressed_count) ||
        !yvex_sha256_update_u64(&hash, reference->compressed_stride) ||
        !yvex_sha256_update_u64(&hash, reference->indexer_count) ||
        !yvex_sha256_update_u64(&hash, reference->indexer_stride) ||
        !yvex_sha256_update_u64(&hash, reference->index_query_stride) ||
        !yvex_sha256_update_u64(&hash, reference->index_weight_stride) ||
        !yvex_sha256_update_u64(&hash, reference->topk_stride))
        return 0;
#define REF_HASH_TRACE(field_, rows_, stride_)                               \
    do {                                                                     \
        if (!ref_mul((rows_), (stride_), &extent) ||                         \
            !ref_hash_float_array(&hash, reference->field_, extent))         \
            return 0;                                                        \
    } while (0)
    REF_HASH_TRACE(q_low, reference->token_count, reference->q_rank);
    REF_HASH_TRACE(query, reference->token_count, reference->query_width);
    REF_HASH_TRACE(raw_kv, reference->token_count, reference->kv_width);
    REF_HASH_TRACE(compressed_kv, reference->compressed_count,
                   reference->compressed_stride);
    REF_HASH_TRACE(indexer_kv, reference->indexer_count,
                   reference->indexer_stride);
    REF_HASH_TRACE(index_query, reference->token_count,
                   reference->index_query_stride);
    REF_HASH_TRACE(index_weights, reference->token_count,
                   reference->index_weight_stride);
    REF_HASH_TRACE(attention_values, reference->token_count,
                   reference->query_width);
#undef REF_HASH_TRACE
    if (!ref_hash_u64_array(&hash, reference->compressed_positions,
                            reference->compressed_count) ||
        !ref_hash_u64_array(&hash, reference->indexer_positions,
                            reference->indexer_count))
        return 0;
    for (token = 0ull; token < reference->token_count; ++token) {
        unsigned long long selected;
        unsigned long long selected_count = reference->topk_stride
            ? reference->topk_counts[token] : 0ull;

        if (selected_count > reference->topk_stride ||
            !yvex_sha256_update_u64(&hash, selected_count))
            return 0;
        for (selected = 0ull; selected < selected_count; ++selected) {
            if (!yvex_sha256_update_u64(
                    &hash, reference->topk_positions[
                        token * reference->topk_stride + selected]))
                return 0;
        }
        if (ULLONG_MAX - evidence->exact_topk_positions < selected_count)
            return 0;
        evidence->exact_topk_positions += selected_count;
    }
    if (!ref_hash_rolling_output(
            &hash, &reference->next_main_rolling_state) ||
        !ref_hash_rolling_output(
            &hash, &reference->next_indexer_rolling_state) ||
        !ref_hash_finish(&hash, evidence->oracle_trace_identity))
        return 0;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.attention.reference.evidence.v2") ||
        !yvex_sha256_update_text(&hash, attention_plan_identity) ||
        !yvex_sha256_update_text(&hash, evidence->oracle_trace_identity) ||
        !yvex_sha256_update_text(&hash, evidence->oracle_output_identity) ||
        !yvex_sha256_update_text(&hash, evidence->fixture_identity) ||
        !yvex_sha256_update_text(&hash, evidence->history_identity) ||
        !yvex_sha256_update_text(
            &hash, evidence->comparison_contract_identity) ||
        !yvex_sha256_update_u64(&hash, evidence->exact_topk_positions) ||
        !ref_hash_finish(&hash, evidence->evidence_identity))
        return 0;
    return 1;
}
