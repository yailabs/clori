/*
 * Exercises CUDA backend CUDA supports the same minimal F32 embedding op as the backend layer
 * CPU reference backend. Returns 77 when CUDA is unavailable.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static void make_desc(yvex_backend_tensor_desc *desc,
                      const char *name,
                      yvex_dtype dtype,
                      unsigned int rank,
                      unsigned long long d0,
                      unsigned long long d1,
                      unsigned long long bytes)
{
    memset(desc, 0, sizeof(*desc));
    desc->name = name;
    desc->dtype = dtype;
    desc->rank = rank;
    desc->dims[0] = d0;
    desc->dims[1] = d1;
    desc->bytes = bytes;
}

static void write_u16le(unsigned char *p, unsigned int value)
{
    p[0] = (unsigned char)(value & 0xffu);
    p[1] = (unsigned char)((value >> 8) & 0xffu);
}

static int float_close(float a, float b, float tolerance)
{
    float diff = a - b;
    if (diff < 0.0f) {
        diff = -diff;
    }
    return diff <= tolerance;
}

static double test_exp_double(double x)
{
    const double ln2 = 0.69314718055994530942;
    double term;
    double sum;
    int n = 0;
    unsigned int i;

    if (x < -60.0) {
        return 0.0;
    }
    if (x > 60.0) {
        x = 60.0;
    }
    while (x > 0.5) {
        x -= ln2;
        ++n;
    }
    while (x < -0.5) {
        x += ln2;
        --n;
    }
    term = 1.0;
    sum = 1.0;
    for (i = 1u; i <= 18u; ++i) {
        term *= x / (double)i;
        sum += term;
    }
    while (n > 0) {
        sum *= 2.0;
        --n;
    }
    while (n < 0) {
        sum *= 0.5;
        ++n;
    }
    return sum;
}

static double test_silu_double(double x)
{
    return x / (1.0 + test_exp_double(-x));
}

static void mlp_reference(const float *input,
                          const float *gate,
                          const float *up,
                          const float *down,
                          unsigned long long batch,
                          unsigned long long hidden_dim,
                          unsigned long long ffn_dim,
                          unsigned long long expert_id,
                          int routed,
                          float *intermediate,
                          float *out)
{
    unsigned long long gate_offset = routed ? expert_id * hidden_dim * ffn_dim : 0ull;
    unsigned long long down_offset = routed ? expert_id * ffn_dim * hidden_dim : 0ull;
    unsigned long long row;
    unsigned long long h;
    unsigned long long j;

    for (row = 0; row < batch; ++row) {
        for (j = 0; j < ffn_dim; ++j) {
            double gate_sum = 0.0;
            double up_sum = 0.0;
            for (h = 0; h < hidden_dim; ++h) {
                double x = (double)input[(row * hidden_dim) + h];
                gate_sum += x * (double)gate[gate_offset + (h * ffn_dim) + j];
                up_sum += x * (double)up[gate_offset + (h * ffn_dim) + j];
            }
            intermediate[(row * ffn_dim) + j] =
                (float)(test_silu_double(gate_sum) * up_sum);
        }
        for (h = 0; h < hidden_dim; ++h) {
            double sum = 0.0;
            for (j = 0; j < ffn_dim; ++j) {
                sum += (double)intermediate[(row * ffn_dim) + j] *
                       (double)down[down_offset + (j * hidden_dim) + h];
            }
            out[(row * hidden_dim) + h] = (float)sum;
        }
    }
}

static void attention_reference(const float *query,
                                const float *keys,
                                const float *values,
                                unsigned long long seq_len,
                                unsigned long long position,
                                unsigned long long head_dim,
                                float scale,
                                int causal,
                                float *scores,
                                float *probabilities,
                                float *out)
{
    unsigned long long visible_count = causal ? position + 1ull : seq_len;
    unsigned long long i;
    unsigned long long d;
    double max_score;
    double sum_exp = 0.0;

    for (i = 0; i < seq_len; ++i) {
        double score = 0.0;
        probabilities[i] = 0.0f;
        if (causal && i > position) {
            scores[i] = 0.0f;
            continue;
        }
        for (d = 0; d < head_dim; ++d) {
            score += (double)query[d] * (double)keys[(i * head_dim) + d];
        }
        scores[i] = (float)(score * (double)scale);
    }

    max_score = (double)scores[0];
    for (i = 1; i < visible_count; ++i) {
        if ((double)scores[i] > max_score) {
            max_score = (double)scores[i];
        }
    }
    for (i = 0; i < visible_count; ++i) {
        double e = test_exp_double((double)scores[i] - max_score);
        probabilities[i] = (float)e;
        sum_exp += e;
    }
    if (sum_exp != 0.0) {
        for (i = 0; i < visible_count; ++i) {
            probabilities[i] = (float)((double)probabilities[i] / sum_exp);
        }
    } else {
        for (i = 0; i < visible_count; ++i) {
            probabilities[i] = 0.0f;
        }
    }

    for (d = 0; d < head_dim; ++d) {
        double value = 0.0;
        for (i = 0; i < visible_count; ++i) {
            value += (double)probabilities[i] * (double)values[(i * head_dim) + d];
        }
        out[d] = (float)value;
    }
}

static int open_cuda_or_skip(yvex_backend **out)
{
    yvex_backend_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(out, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        fprintf(stderr, "SKIP: CUDA unavailable: %s\n", yvex_error_message(&err));
        return 77;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open cuda backend");
    return 0;
}

int yvex_cuda_test_ops(void)
{
    yvex_backend *cuda_backend = NULL;
    yvex_device_tensor *embedding = NULL;
    yvex_device_tensor *embedding_f16 = NULL;
    yvex_device_tensor *out_f16 = NULL;
    yvex_device_tensor *rms_input = NULL;
    yvex_device_tensor *rms_weight = NULL;
    yvex_device_tensor *rms_weight_f32 = NULL;
    yvex_device_tensor *rms_out = NULL;
    yvex_device_tensor *rope_input = NULL;
    yvex_device_tensor *rope_out = NULL;
    yvex_device_tensor *attention_query = NULL;
    yvex_device_tensor *attention_keys = NULL;
    yvex_device_tensor *attention_values = NULL;
    yvex_device_tensor *attention_scores = NULL;
    yvex_device_tensor *attention_probabilities = NULL;
    yvex_device_tensor *attention_out = NULL;
    yvex_device_tensor *matmul_input = NULL;
    yvex_device_tensor *matmul_weight = NULL;
    yvex_device_tensor *matmul_out = NULL;
    yvex_device_tensor *mlp_input = NULL;
    yvex_device_tensor *mlp_gate = NULL;
    yvex_device_tensor *mlp_up = NULL;
    yvex_device_tensor *mlp_down = NULL;
    yvex_device_tensor *mlp_intermediate = NULL;
    yvex_device_tensor *mlp_out = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_backend_capability_result capability;
    yvex_mlp_options mlp_options;
    yvex_error err;
    float embedding_data[12] = {
        0.0f, 1.0f, 2.0f, 3.0f,
        10.0f, 11.0f, 12.0f, 13.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
    };
    float out_data[8];
    float expected[8] = {
        0.0f, 1.0f, 2.0f, 3.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
    };
    unsigned int token_ids[2] = {0, 2};
    unsigned int token_zero[1] = {0};
    unsigned int bad_ids[1] = {9};
    unsigned char embedding_f16_data[12];
    unsigned int embedding_f16_bits[6] = {
        0x0000u, 0x8000u, 0x3c00u, 0xbc00u, 0x0001u, 0x7bffu,
    };
    float embedding_f16_out[6];
    float embedding_f16_expected[6] = {
        0.0f, -0.0f, 1.0f, -1.0f, 5.9604645e-8f, 65504.0f,
    };
    float embedding_f16_sentinel[6] = {
        -91.0f, -92.0f, -93.0f, -94.0f, -95.0f, -96.0f,
    };
    float rms_input_data[8] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f, -1.0f,
    };
    float rms_weight_f32_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    unsigned char rms_weight_data[8];
    float rms_out_data[8];
    float rms_expected[8] = {
        0.9999995f, 1.9999990f, 2.9999985f, 3.9999980f,
        0.9999995f, -1.9999990f, 2.9999985f, -3.9999980f,
    };
    float rope_input_data[8] = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};
    float rope_out_data[8];
    float attention_query_data[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float attention_key_data[16] = {
        0.2f, 0.1f, 0.0f, 0.3f,
        0.4f, 0.3f, 0.2f, 0.1f,
        0.1f, 0.5f, 0.3f, 0.2f,
        0.3f, 0.2f, 0.4f, 0.6f,
    };
    float attention_value_data[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        10.0f, 20.0f, 30.0f, 40.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        8.0f, 7.0f, 6.0f, 5.0f,
    };
    float attention_out_data[4];
    float attention_expected[4];
    float attention_noncausal_expected[4];
    float attention_score_data[4];
    float attention_probability_data[4];
    float attention_reference_scores[4];
    float attention_reference_probabilities[4];
    float matmul_input_data[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    float matmul_weight_data[12] = {
        0.5f, 1.0f, 1.5f,
        2.0f, 2.5f, 3.0f,
        3.5f, 4.0f, 4.5f,
        5.0f, 5.5f, 6.0f,
    };
    float matmul_out_data[6];
    float matmul_expected[6];
    float mlp_input_data[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float mlp_gate_data[12] = {
        0.01f, 0.02f, 0.03f,
        0.04f, 0.05f, 0.06f,
        0.07f, 0.08f, 0.09f,
        0.10f, 0.11f, 0.12f,
    };
    float mlp_up_data[12] = {
        0.02f, 0.03f, 0.04f,
        0.05f, 0.06f, 0.07f,
        0.08f, 0.09f, 0.10f,
        0.11f, 0.12f, 0.13f,
    };
    float mlp_down_data[12] = {
        0.03f, 0.04f, 0.05f, 0.06f,
        0.07f, 0.08f, 0.09f, 0.10f,
        0.11f, 0.12f, 0.13f, 0.14f,
    };
    float mlp_out_data[4];
    float mlp_intermediate_data[3];
    float mlp_expected[4];
    float mlp_expected_mid[3];
    float mlp_routed_gate_data[24];
    float mlp_routed_up_data[24];
    float mlp_routed_down_data[24];
    float mlp_routed_out_data[4];
    float mlp_routed_expected[4];
    float mlp_routed_expected_mid[3];
    unsigned int rms_half_values[4] = {0x3c00u, 0x4000u, 0x4200u, 0x4400u};
    unsigned int i;
    unsigned int row;
    unsigned int col;
    int rc;

    for (i = 0; i < 4u; ++i) {
        write_u16le(rms_weight_data + (i * 2u), rms_half_values[i]);
    }
    for (i = 0; i < 6u; ++i) {
        write_u16le(embedding_f16_data + (i * 2u), embedding_f16_bits[i]);
    }
    for (row = 0; row < 2u; ++row) {
        for (col = 0; col < 3u; ++col) {
            double sum = 0.0;
            for (i = 0; i < 4u; ++i) {
                sum += (double)matmul_input_data[(row * 4u) + i] *
                       (double)matmul_weight_data[(i * 3u) + col];
            }
            matmul_expected[(row * 3u) + col] = (float)sum;
        }
    }
    for (i = 0; i < 24u; ++i) {
        mlp_routed_gate_data[i] = (float)(0.01 + ((double)(i + 1u) * 0.001));
        mlp_routed_up_data[i] = (float)(0.015 + ((double)(i + 1u) * 0.0015));
        mlp_routed_down_data[i] = (float)(0.02 + ((double)(i + 1u) * 0.0008));
    }
    mlp_reference(mlp_input_data, mlp_gate_data, mlp_up_data, mlp_down_data,
                  1, 4, 3, 0, 0, mlp_expected_mid, mlp_expected);
    mlp_reference(mlp_input_data, mlp_routed_gate_data, mlp_routed_up_data,
                  mlp_routed_down_data, 1, 4, 3, 1, 1,
                  mlp_routed_expected_mid, mlp_routed_expected);
    attention_reference(attention_query_data, attention_key_data,
                        attention_value_data, 4, 2, 4, 0.5f, 1,
                        attention_reference_scores,
                        attention_reference_probabilities,
                        attention_expected);
    attention_reference(attention_query_data, attention_key_data,
                        attention_value_data, 4, 0, 4, 0.5f, 0,
                        attention_reference_scores,
                        attention_reference_probabilities,
                        attention_noncausal_expected);
    memset(&mlp_options, 0, sizeof(mlp_options));
    mlp_options.batch = 1;
    mlp_options.hidden_dim = 4;
    mlp_options.ffn_dim = 3;
    mlp_options.gated = 1;
    mlp_options.activation = "silu";

    rc = open_cuda_or_skip(&cuda_backend);
    if (rc == 77) return 77;

    YVEX_TEST_ASSERT(yvex_backend_supports(cuda_backend, YVEX_BACKEND_CAP_TENSOR_ALLOC),
                     "cuda tensor alloc supported");
    YVEX_TEST_ASSERT(yvex_backend_supports(cuda_backend, YVEX_BACKEND_CAP_TENSOR_READ_WRITE),
                     "cuda tensor rw supported");
    YVEX_TEST_ASSERT(yvex_backend_supports(cuda_backend, YVEX_BACKEND_CAP_OP_EMBED),
                     "cuda embed supported");
    YVEX_TEST_ASSERT(yvex_backend_supports(cuda_backend, YVEX_BACKEND_CAP_OP_RMS_NORM),
                     "cuda rmsnorm supported");
    YVEX_TEST_ASSERT(yvex_backend_supports(cuda_backend, YVEX_BACKEND_CAP_OP_ROPE),
                     "cuda rope supported");
    YVEX_TEST_ASSERT(yvex_backend_supports(cuda_backend, YVEX_BACKEND_CAP_OP_ATTENTION),
                     "cuda attention supported");
    YVEX_TEST_ASSERT(yvex_backend_supports(cuda_backend, YVEX_BACKEND_CAP_OP_MATMUL),
                     "cuda matmul supported");
    YVEX_TEST_ASSERT(yvex_backend_supports(cuda_backend, YVEX_BACKEND_CAP_OP_MLP),
                     "cuda mlp supported");

    make_desc(&desc, "embedding", YVEX_DTYPE_F32, 2, 4, 3, sizeof(embedding_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &embedding, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda embedding");
    rc = yvex_backend_tensor_write(cuda_backend, embedding, embedding_data, sizeof(embedding_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda embedding");

    make_desc(&desc, "out", YVEX_DTYPE_F32, 2, 2, 4, sizeof(out_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda output");
    rc = yvex_backend_op_embed(cuda_backend, embedding, token_ids, 2, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda embed succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, out, out_data, sizeof(out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda embed output");
    YVEX_TEST_ASSERT(memcmp(out_data, expected, sizeof(expected)) == 0,
                     "cuda embed output matches expected rows");

    rc = yvex_backend_op_embed(cuda_backend, embedding, bad_ids, 1, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "bad token id fails");

    make_desc(&desc, "embedding_f16", YVEX_DTYPE_F16, 2, 6, 1,
              sizeof(embedding_f16_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &embedding_f16, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate F16 CUDA embedding");
    rc = yvex_backend_tensor_write(cuda_backend, embedding_f16,
                                   embedding_f16_data,
                                   sizeof(embedding_f16_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write F16 CUDA embedding");
    make_desc(&desc, "embedding_f16_out", YVEX_DTYPE_F32, 2, 1, 6,
              sizeof(embedding_f16_out));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &out_f16, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate F16 embedding output");
    rc = yvex_backend_tensor_write(cuda_backend, out_f16,
                                   embedding_f16_sentinel,
                                   sizeof(embedding_f16_sentinel), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "seed F16 embedding output sentinel");
    rc = yvex_backend_op_embed(cuda_backend, embedding_f16, token_zero, 1,
                               out_f16, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "F16-to-F32 CUDA embedding succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, out_f16, embedding_f16_out,
                                  sizeof(embedding_f16_out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read F16-to-F32 embedding output");
    YVEX_TEST_ASSERT(memcmp(embedding_f16_out, embedding_f16_sentinel,
                            sizeof(embedding_f16_out)) != 0,
                     "F16 embedding kernel changed sentinel output");
    YVEX_TEST_ASSERT(memcmp(embedding_f16_out, embedding_f16_expected,
                            sizeof(embedding_f16_out)) == 0,
                     "F16 embedding preserves exact representative conversions");
    for (i = 0; i < 6u; ++i) {
        YVEX_TEST_ASSERT(float_close(embedding_f16_out[i],
                                     embedding_f16_expected[i], 0.000001f),
                         "F16 embedding covers signed, subnormal, and boundary values");
    }

    make_desc(&desc, "rms_input", YVEX_DTYPE_F32, 2, 2, 4, sizeof(rms_input_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &rms_input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda rms input");
    rc = yvex_backend_tensor_write(cuda_backend, rms_input, rms_input_data, sizeof(rms_input_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda rms input");

    make_desc(&desc, "rms_weight_f32", YVEX_DTYPE_F32, 1, 4, 0,
              sizeof(rms_weight_f32_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &rms_weight_f32, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate F32 CUDA rms weight");
    rc = yvex_backend_tensor_write(cuda_backend, rms_weight_f32,
                                   rms_weight_f32_data,
                                   sizeof(rms_weight_f32_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write F32 CUDA rms weight");

    make_desc(&desc, "rms_weight", YVEX_DTYPE_F16, 1, 4, 0, sizeof(rms_weight_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &rms_weight, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda rms weight");
    rc = yvex_backend_tensor_write(cuda_backend, rms_weight, rms_weight_data, sizeof(rms_weight_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda rms weight");

    make_desc(&desc, "rms_out", YVEX_DTYPE_F32, 2, 2, 4, sizeof(rms_out_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &rms_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda rms output");
    rc = yvex_backend_op_rms_norm(cuda_backend, rms_input, rms_weight_f32,
                                  0.000001f, rms_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "CUDA RMSNorm F32 weight succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, rms_out, rms_out_data,
                                  sizeof(rms_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read CUDA RMSNorm F32 output");
    for (i = 0; i < 8u; ++i) {
        YVEX_TEST_ASSERT(float_close(rms_out_data[i], rms_expected[i], 0.0001f),
                         "CUDA RMSNorm F32 output matches reference");
    }
    rc = yvex_backend_op_rms_norm(cuda_backend, rms_input, rms_weight, 0.000001f, rms_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda rms norm succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, rms_out, rms_out_data, sizeof(rms_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda rms output");
    for (i = 0; i < 8u; ++i) {
        YVEX_TEST_ASSERT(float_close(rms_out_data[i], rms_expected[i], 0.0001f),
                         "cuda rms output matches expected");
    }
    rc = yvex_backend_op_rms_norm(cuda_backend, rms_input, rms_weight,
                                  NAN, rms_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "CUDA RMSNorm rejects non-finite epsilon");

    make_desc(&desc, "rope_input", YVEX_DTYPE_F32, 1, 8, 0, sizeof(rope_input_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &rope_input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda rope input");
    rc = yvex_backend_tensor_write(cuda_backend, rope_input, rope_input_data, sizeof(rope_input_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda rope input");

    make_desc(&desc, "rope_out", YVEX_DTYPE_F32, 1, 8, 0, sizeof(rope_out_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &rope_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda rope output");
    rc = yvex_backend_op_rope(cuda_backend, rope_input, 0, 10000.0f, rope_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda rope position zero succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, rope_out, rope_out_data, sizeof(rope_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda rope zero output");
    for (i = 0; i < 8u; ++i) {
        YVEX_TEST_ASSERT(float_close(rope_out_data[i], rope_input_data[i], 0.0001f),
                         "cuda rope position zero is identity");
    }

    rc = yvex_backend_op_rope(cuda_backend, rope_input, 7, 10000.0f, rope_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda rope position seven succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, rope_out, rope_out_data, sizeof(rope_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda rope output");
    YVEX_TEST_ASSERT(!float_close(rope_out_data[0], rope_input_data[0], 0.0001f),
                     "cuda rope non-zero position changes first pair");
    rc = yvex_backend_op_rope(cuda_backend, rope_input, 7, INFINITY,
                              rope_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "CUDA RoPE rejects non-finite base");

    make_desc(&desc, "attention_query", YVEX_DTYPE_F32, 1, 4, 0, sizeof(attention_query_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &attention_query, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda attention query");
    rc = yvex_backend_tensor_write(cuda_backend, attention_query,
                                   attention_query_data, sizeof(attention_query_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda attention query");

    make_desc(&desc, "attention_keys", YVEX_DTYPE_F32, 2, 4, 4, sizeof(attention_key_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &attention_keys, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda attention keys");
    rc = yvex_backend_tensor_write(cuda_backend, attention_keys,
                                   attention_key_data, sizeof(attention_key_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda attention keys");

    make_desc(&desc, "attention_values", YVEX_DTYPE_F32, 2, 4, 4, sizeof(attention_value_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &attention_values, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda attention values");
    rc = yvex_backend_tensor_write(cuda_backend, attention_values,
                                   attention_value_data, sizeof(attention_value_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda attention values");

    make_desc(&desc, "attention_scores", YVEX_DTYPE_F32, 1, 4, 0, sizeof(attention_score_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &attention_scores, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda attention scores");
    make_desc(&desc, "attention_probabilities", YVEX_DTYPE_F32, 1, 4, 0, sizeof(attention_probability_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &attention_probabilities, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda attention probabilities");
    make_desc(&desc, "attention_out", YVEX_DTYPE_F32, 1, 4, 0, sizeof(attention_out_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &attention_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda attention output");

    rc = yvex_backend_op_attention(cuda_backend, attention_query, attention_keys,
                                   attention_values, 4, 0, 0.5f, 1,
                                   attention_scores, attention_probabilities,
                                   attention_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda attention position zero succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, attention_out,
                                  attention_out_data, sizeof(attention_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda attention output");
    for (i = 0; i < 4u; ++i) {
        YVEX_TEST_ASSERT(float_close(attention_out_data[i], attention_value_data[i], 0.0001f),
                         "cuda attention position zero selects first value");
    }

    rc = yvex_backend_op_attention(cuda_backend, attention_query, attention_keys,
                                   attention_values, 4, 2, 0.5f, 1,
                                   attention_scores, attention_probabilities,
                                   attention_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda attention causal prefix succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, attention_out,
                                  attention_out_data, sizeof(attention_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda attention causal output");
    for (i = 0; i < 4u; ++i) {
        YVEX_TEST_ASSERT(float_close(attention_out_data[i], attention_expected[i], 0.0001f),
                         "cuda attention causal output matches reference");
    }

    rc = yvex_backend_op_attention(cuda_backend, attention_query, attention_keys,
                                   attention_values, 4, 0, 0.5f, 0,
                                   attention_scores, attention_probabilities,
                                   attention_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda attention noncausal succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, attention_out,
                                  attention_out_data, sizeof(attention_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda attention noncausal output");
    for (i = 0; i < 4u; ++i) {
        YVEX_TEST_ASSERT(float_close(attention_out_data[i],
                                     attention_noncausal_expected[i],
                                     0.0001f),
                         "cuda attention noncausal output matches reference");
    }

    rc = yvex_backend_op_attention(cuda_backend, attention_query, attention_keys,
                                   attention_values, 4, 4, 0.5f, 1,
                                   attention_scores, attention_probabilities,
                                   attention_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "cuda attention rejects out-of-range position");
    rc = yvex_backend_op_attention(cuda_backend, attention_query, attention_keys,
                                   attention_values, 0, 0, 0.5f, 1,
                                   attention_scores, attention_probabilities,
                                   attention_out, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "cuda attention rejects zero seq_len");
    rc = yvex_backend_op_attention(cuda_backend, attention_query, attention_keys,
                                   attention_values, 4, 0, NAN, 1,
                                   attention_scores, attention_probabilities,
                                   attention_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "CUDA attention rejects non-finite scale");

    make_desc(&desc, "matmul_input", YVEX_DTYPE_F32, 2, 2, 4, sizeof(matmul_input_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &matmul_input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda matmul input");
    rc = yvex_backend_tensor_write(cuda_backend, matmul_input,
                                   matmul_input_data, sizeof(matmul_input_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda matmul input");
    make_desc(&desc, "matmul_weight", YVEX_DTYPE_F32, 2, 4, 3, sizeof(matmul_weight_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &matmul_weight, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda matmul weight");
    rc = yvex_backend_tensor_write(cuda_backend, matmul_weight,
                                   matmul_weight_data, sizeof(matmul_weight_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda matmul weight");
    make_desc(&desc, "matmul_out", YVEX_DTYPE_F32, 2, 2, 3, sizeof(matmul_out_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &matmul_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda matmul output");
    rc = yvex_backend_op_matmul(cuda_backend, matmul_input, matmul_weight, matmul_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda matmul succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, matmul_out,
                                  matmul_out_data, sizeof(matmul_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda matmul output");
    for (i = 0; i < 6u; ++i) {
        YVEX_TEST_ASSERT(float_close(matmul_out_data[i], matmul_expected[i], 0.0001f),
                         "cuda matmul output matches reference");
    }

    make_desc(&desc, "mlp_input", YVEX_DTYPE_F32, 2, 1, 4, sizeof(mlp_input_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda mlp input");
    rc = yvex_backend_tensor_write(cuda_backend, mlp_input,
                                   mlp_input_data, sizeof(mlp_input_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda mlp input");
    make_desc(&desc, "mlp_gate", YVEX_DTYPE_F32, 2, 4, 3, sizeof(mlp_gate_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_gate, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda mlp gate");
    rc = yvex_backend_tensor_write(cuda_backend, mlp_gate,
                                   mlp_gate_data, sizeof(mlp_gate_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda mlp gate");
    make_desc(&desc, "mlp_up", YVEX_DTYPE_F32, 2, 4, 3, sizeof(mlp_up_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_up, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda mlp up");
    rc = yvex_backend_tensor_write(cuda_backend, mlp_up,
                                   mlp_up_data, sizeof(mlp_up_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda mlp up");
    make_desc(&desc, "mlp_down", YVEX_DTYPE_F32, 2, 3, 4, sizeof(mlp_down_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_down, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda mlp down");
    rc = yvex_backend_tensor_write(cuda_backend, mlp_down,
                                   mlp_down_data, sizeof(mlp_down_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda mlp down");
    make_desc(&desc, "mlp_intermediate", YVEX_DTYPE_F32, 2, 1, 3,
              sizeof(mlp_intermediate_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_intermediate, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda mlp intermediate");
    make_desc(&desc, "mlp_out", YVEX_DTYPE_F32, 2, 1, 4, sizeof(mlp_out_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda mlp output");
    rc = yvex_backend_op_mlp(cuda_backend, mlp_input, mlp_gate, mlp_up, mlp_down,
                             &mlp_options, mlp_intermediate, mlp_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda dense mlp succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, mlp_out,
                                  mlp_out_data, sizeof(mlp_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda mlp output");
    for (i = 0; i < 4u; ++i) {
        YVEX_TEST_ASSERT(float_close(mlp_out_data[i], mlp_expected[i], 0.0001f),
                         "cuda dense mlp output matches reference");
    }

    yvex_backend_tensor_free(cuda_backend, mlp_out);
    yvex_backend_tensor_free(cuda_backend, mlp_intermediate);
    yvex_backend_tensor_free(cuda_backend, mlp_down);
    yvex_backend_tensor_free(cuda_backend, mlp_up);
    yvex_backend_tensor_free(cuda_backend, mlp_gate);
    mlp_out = NULL;
    mlp_intermediate = NULL;
    mlp_down = NULL;
    mlp_up = NULL;
    mlp_gate = NULL;

    mlp_options.routed_expert_mode = 1;
    mlp_options.expert_count = 2;
    mlp_options.expert_id = 1;
    make_desc(&desc, "mlp_routed_gate", YVEX_DTYPE_F32, 3, 2, 4,
              sizeof(mlp_routed_gate_data));
    desc.dims[2] = 3;
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_gate, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda routed mlp gate");
    rc = yvex_backend_tensor_write(cuda_backend, mlp_gate,
                                   mlp_routed_gate_data, sizeof(mlp_routed_gate_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda routed mlp gate");
    make_desc(&desc, "mlp_routed_up", YVEX_DTYPE_F32, 3, 2, 4,
              sizeof(mlp_routed_up_data));
    desc.dims[2] = 3;
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_up, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda routed mlp up");
    rc = yvex_backend_tensor_write(cuda_backend, mlp_up,
                                   mlp_routed_up_data, sizeof(mlp_routed_up_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda routed mlp up");
    make_desc(&desc, "mlp_routed_down", YVEX_DTYPE_F32, 3, 2, 3,
              sizeof(mlp_routed_down_data));
    desc.dims[2] = 4;
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_down, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda routed mlp down");
    rc = yvex_backend_tensor_write(cuda_backend, mlp_down,
                                   mlp_routed_down_data, sizeof(mlp_routed_down_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda routed mlp down");
    make_desc(&desc, "mlp_routed_intermediate", YVEX_DTYPE_F32, 2, 1, 3,
              sizeof(mlp_intermediate_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_intermediate, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda routed mlp intermediate");
    make_desc(&desc, "mlp_routed_out", YVEX_DTYPE_F32, 2, 1, 4,
              sizeof(mlp_routed_out_data));
    rc = yvex_backend_tensor_alloc(cuda_backend, &desc, &mlp_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda routed mlp output");
    rc = yvex_backend_op_mlp(cuda_backend, mlp_input, mlp_gate, mlp_up, mlp_down,
                             &mlp_options, mlp_intermediate, mlp_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cuda routed mlp succeeds");
    rc = yvex_backend_tensor_read(cuda_backend, mlp_out,
                                  mlp_routed_out_data, sizeof(mlp_routed_out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda routed mlp output");
    for (i = 0; i < 4u; ++i) {
        YVEX_TEST_ASSERT(float_close(mlp_routed_out_data[i], mlp_routed_expected[i], 0.0001f),
                         "cuda routed mlp output matches selected expert reference");
    }
    mlp_options.hidden_dim = 0;
    rc = yvex_backend_op_mlp(cuda_backend, mlp_input, mlp_gate, mlp_up, mlp_down,
                             &mlp_options, mlp_intermediate, mlp_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "cuda mlp rejects zero hidden dim");

    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_LAUNCH_FAILURE", "matmul-f32", 1) == 0,
                     "set CUDA launch failure injection");
    rc = yvex_backend_op_matmul(cuda_backend, matmul_input, matmul_weight,
                                matmul_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND,
                     "CUDA launch failure returns backend error");
    YVEX_TEST_ASSERT(!yvex_device_tensor_is_written(matmul_out),
                     "launch failure leaves output unwritten");
    rc = yvex_backend_query_capability(cuda_backend,
                                       YVEX_BACKEND_VARIANT_MATMUL_F32,
                                       &capability, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     capability.state == YVEX_BACKEND_CAPABILITY_FAILED &&
                     capability.reason == YVEX_BACKEND_CAPABILITY_REASON_LAUNCH_FAILED,
                     "launch failure demotes exact variant");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_LAUNCH_FAILURE") == 0,
                     "clear CUDA launch failure injection");

    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_SYNC_FAILURE", "rope-f32", 1) == 0,
                     "set CUDA op synchronization failure injection");
    rc = yvex_backend_op_rope(cuda_backend, rope_input, 7, 10000.0f,
                              rope_out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND,
                     "CUDA op synchronization failure returns backend error");
    YVEX_TEST_ASSERT(!yvex_device_tensor_is_written(rope_out),
                     "synchronization failure leaves output unwritten");
    rc = yvex_backend_query_capability(cuda_backend,
                                       YVEX_BACKEND_VARIANT_ROPE_F32,
                                       &capability, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     capability.state == YVEX_BACKEND_CAPABILITY_FAILED &&
                     capability.reason ==
                         YVEX_BACKEND_CAPABILITY_REASON_SYNCHRONIZATION_FAILED,
                     "synchronization failure demotes exact variant");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_SYNC_FAILURE") == 0,
                     "clear CUDA op synchronization failure injection");

    yvex_backend_tensor_free(cuda_backend, mlp_out);
    yvex_backend_tensor_free(cuda_backend, mlp_intermediate);
    yvex_backend_tensor_free(cuda_backend, mlp_down);
    yvex_backend_tensor_free(cuda_backend, mlp_up);
    yvex_backend_tensor_free(cuda_backend, mlp_gate);
    yvex_backend_tensor_free(cuda_backend, mlp_input);
    yvex_backend_tensor_free(cuda_backend, matmul_out);
    yvex_backend_tensor_free(cuda_backend, matmul_weight);
    yvex_backend_tensor_free(cuda_backend, matmul_input);
    yvex_backend_tensor_free(cuda_backend, attention_out);
    yvex_backend_tensor_free(cuda_backend, attention_probabilities);
    yvex_backend_tensor_free(cuda_backend, attention_scores);
    yvex_backend_tensor_free(cuda_backend, attention_values);
    yvex_backend_tensor_free(cuda_backend, attention_keys);
    yvex_backend_tensor_free(cuda_backend, attention_query);
    yvex_backend_tensor_free(cuda_backend, rope_out);
    yvex_backend_tensor_free(cuda_backend, rope_input);
    yvex_backend_tensor_free(cuda_backend, rms_out);
    yvex_backend_tensor_free(cuda_backend, rms_weight);
    yvex_backend_tensor_free(cuda_backend, rms_weight_f32);
    yvex_backend_tensor_free(cuda_backend, rms_input);
    yvex_backend_tensor_free(cuda_backend, out_f16);
    yvex_backend_tensor_free(cuda_backend, embedding_f16);
    yvex_backend_tensor_free(cuda_backend, out);
    yvex_backend_tensor_free(cuda_backend, embedding);
    yvex_backend_close(cuda_backend);
    return 0;
}
