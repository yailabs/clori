/*
 * Exercises backend layer CPU backend capabilities are explicit and that the minimal F32
 * embedding reference op works over backend tensors.
 */
#include <limits.h>
#include <string.h>

#include <yvex/api.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/graph_state.h>

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

static int test_capabilities(void)
{
    yvex_backend *backend = NULL;
    yvex_error err;

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu");
    YVEX_TEST_ASSERT_STREQ(yvex_backend_capability_name(YVEX_BACKEND_CAP_TENSOR_ALLOC),
                           "tensor_alloc", "capability name");
    YVEX_TEST_ASSERT_STREQ(yvex_backend_capability_name(YVEX_BACKEND_CAP_OP_ROPE),
                           "op_rope", "rope capability name");
    YVEX_TEST_ASSERT(yvex_backend_supports(backend, YVEX_BACKEND_CAP_TENSOR_ALLOC),
                     "supports tensor alloc");
    YVEX_TEST_ASSERT(yvex_backend_supports(backend, YVEX_BACKEND_CAP_TENSOR_READ_WRITE),
                     "supports tensor read/write");
    YVEX_TEST_ASSERT(yvex_backend_supports(backend, YVEX_BACKEND_CAP_OP_EMBED),
                     "supports embed");
    YVEX_TEST_ASSERT(yvex_backend_supports(backend, YVEX_BACKEND_CAP_OP_RMS_NORM),
                     "supports rms norm");
    YVEX_TEST_ASSERT(yvex_backend_supports(backend, YVEX_BACKEND_CAP_OP_ROPE),
                     "supports rope");
    YVEX_TEST_ASSERT(yvex_backend_supports(backend, YVEX_BACKEND_CAP_OP_ATTENTION),
                     "supports attention");
    YVEX_TEST_ASSERT(yvex_backend_supports(backend, YVEX_BACKEND_CAP_OP_MATMUL),
                     "supports matmul");
    YVEX_TEST_ASSERT(yvex_backend_supports(backend, YVEX_BACKEND_CAP_OP_MLP),
                     "supports mlp");
    yvex_backend_close(backend);
    return 0;
}

static int test_embed_success(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *embedding = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
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
    int rc;

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu");

    make_desc(&desc, "embedding", YVEX_DTYPE_F32, 2, 4, 3, sizeof(embedding_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &embedding, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate embedding");
    rc = yvex_backend_tensor_write(backend, embedding, embedding_data, sizeof(embedding_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write embedding");

    make_desc(&desc, "out", YVEX_DTYPE_F32, 2, 2, 4, sizeof(out_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate output");

    rc = yvex_backend_op_embed(backend, embedding, token_ids, 2, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "embed succeeds");
    rc = yvex_backend_tensor_read(backend, out, out_data, sizeof(out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read embed output");
    YVEX_TEST_ASSERT(memcmp(out_data, expected, sizeof(expected)) == 0, "embed output matches expected rows");

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, embedding);
    yvex_backend_close(backend);
    return 0;
}

static int test_embed_f16_success(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *embedding = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_error err;
    unsigned char embedding_data[24];
    float out_data[8];
    float expected[8] = {
        0.0f, 1.0f, 2.0f, 3.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
    };
    unsigned int token_ids[2] = {0, 2};
    unsigned int half_values[12] = {
        0x0000u, 0x3c00u, 0x4000u, 0x4200u,
        0x4900u, 0x4980u, 0x4a00u, 0x4a80u,
        0x4d00u, 0x4d40u, 0x4d80u, 0x4dc0u,
    };
    unsigned int i;
    int rc;

    for (i = 0; i < 12u; ++i) {
        write_u16le(embedding_data + (i * 2u), half_values[i]);
    }

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu f16");

    make_desc(&desc, "embedding", YVEX_DTYPE_F16, 2, 4, 3, sizeof(embedding_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &embedding, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate f16 embedding");
    rc = yvex_backend_tensor_write(backend, embedding, embedding_data, sizeof(embedding_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write f16 embedding");

    make_desc(&desc, "out", YVEX_DTYPE_F32, 2, 2, 4, sizeof(out_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate f16 output");

    rc = yvex_backend_op_embed(backend, embedding, token_ids, 2, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "f16 embed succeeds");
    rc = yvex_backend_tensor_read(backend, out, out_data, sizeof(out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read f16 embed output");
    YVEX_TEST_ASSERT(memcmp(out_data, expected, sizeof(expected)) == 0,
                     "f16 embed output converts expected rows");

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, embedding);
    yvex_backend_close(backend);
    return 0;
}

static int test_embed_failures(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *embedding = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_error err;
    float embedding_data[12] = {0};
    unsigned int good_ids[1] = {0};
    unsigned int bad_ids[1] = {9};
    int rc;

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu failures");

    make_desc(&desc, "embedding", YVEX_DTYPE_F32, 2, 4, 3, sizeof(embedding_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &embedding, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate embedding failures");
    rc = yvex_backend_tensor_write(backend, embedding, embedding_data, sizeof(embedding_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write embedding failures");

    make_desc(&desc, "out", YVEX_DTYPE_F32, 2, 1, 4, 4u * sizeof(float));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate output failures");

    rc = yvex_backend_op_embed(backend, embedding, bad_ids, 1, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "bad token id fails");

    yvex_backend_tensor_free(backend, out);
    make_desc(&desc, "bad_out", YVEX_DTYPE_F32, 2, 4, 1, 4u * sizeof(float));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate bad output");
    rc = yvex_backend_op_embed(backend, embedding, good_ids, 1, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "bad output shape fails");
    yvex_backend_tensor_free(backend, out);
    out = NULL;

    yvex_backend_tensor_free(backend, embedding);
    make_desc(&desc, "bad_embedding", YVEX_DTYPE_BF16, 2, 4, 3, 24);
    rc = yvex_backend_tensor_alloc(backend, &desc, &embedding, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate bf16 embedding");
    make_desc(&desc, "out", YVEX_DTYPE_F32, 2, 1, 4, 4u * sizeof(float));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate bf16 output");
    rc = yvex_backend_op_embed(backend, embedding, good_ids, 1, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_UNSUPPORTED, "unsupported embedding dtype rejected");

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, embedding);
    yvex_backend_close(backend);
    return 0;
}

static int test_rms_norm_success(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *input = NULL;
    yvex_device_tensor *weight = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_error err;
    float input_data[8] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f, -1.0f,
    };
    unsigned char weight_data[8];
    float out_data[8];
    float expected[8] = {
        0.9999995f, 1.9999990f, 2.9999985f, 3.9999980f,
        0.9999995f, -1.9999990f, 2.9999985f, -3.9999980f,
    };
    unsigned int half_values[4] = {0x3c00u, 0x4000u, 0x4200u, 0x4400u};
    unsigned int i;
    int rc;

    for (i = 0; i < 4u; ++i) {
        write_u16le(weight_data + (i * 2u), half_values[i]);
    }

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu rms");

    make_desc(&desc, "input", YVEX_DTYPE_F32, 2, 2, 4, sizeof(input_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate rms input");
    rc = yvex_backend_tensor_write(backend, input, input_data, sizeof(input_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write rms input");

    make_desc(&desc, "weight", YVEX_DTYPE_F16, 1, 4, 0, sizeof(weight_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &weight, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate rms weight");
    rc = yvex_backend_tensor_write(backend, weight, weight_data, sizeof(weight_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write rms weight");

    make_desc(&desc, "out", YVEX_DTYPE_F32, 2, 2, 4, sizeof(out_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate rms output");

    rc = yvex_backend_op_rms_norm(backend, input, weight, 0.000001f, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "rms norm succeeds");
    rc = yvex_backend_tensor_read(backend, out, out_data, sizeof(out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read rms output");
    for (i = 0; i < 8u; ++i) {
        YVEX_TEST_ASSERT(float_close(out_data[i], expected[i], 0.0001f), "rms output matches expected");
    }

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, weight);
    yvex_backend_tensor_free(backend, input);
    yvex_backend_close(backend);
    return 0;
}

static int test_rope_success(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *input = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_error err;
    float input_data[8] = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};
    float out_data[8];
    unsigned int i;
    int rc;

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu rope");

    make_desc(&desc, "rope_input", YVEX_DTYPE_F32, 1, 8, 0, sizeof(input_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate rope input");
    rc = yvex_backend_tensor_write(backend, input, input_data, sizeof(input_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write rope input");

    make_desc(&desc, "rope_out", YVEX_DTYPE_F32, 1, 8, 0, sizeof(out_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate rope output");

    rc = yvex_backend_op_rope(backend, input, 0, 10000.0f, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "rope position zero succeeds");
    rc = yvex_backend_tensor_read(backend, out, out_data, sizeof(out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read rope zero output");
    for (i = 0; i < 8u; ++i) {
        YVEX_TEST_ASSERT(float_close(out_data[i], input_data[i], 0.000001f),
                         "rope position zero is identity");
    }

    rc = yvex_backend_op_rope(backend, input, 7, 10000.0f, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "rope position seven succeeds");
    rc = yvex_backend_tensor_read(backend, out, out_data, sizeof(out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read rope output");
    YVEX_TEST_ASSERT(!float_close(out_data[0], input_data[0], 0.0001f),
                     "rope non-zero position changes first pair");

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, input);
    yvex_backend_close(backend);
    return 0;
}

static int test_rope_failures(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *input = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_error err;
    float input_data[7] = {0};
    float out_data[7] = {0};
    int rc;

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu rope failures");

    make_desc(&desc, "rope_bad_input", YVEX_DTYPE_F32, 1, 7, 0, sizeof(input_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate odd rope input");
    rc = yvex_backend_tensor_write(backend, input, input_data, sizeof(input_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write odd rope input");

    make_desc(&desc, "rope_bad_out", YVEX_DTYPE_F32, 1, 7, 0, sizeof(out_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate odd rope output");
    rc = yvex_backend_op_rope(backend, input, 7, 10000.0f, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "odd head dim rejected");

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, input);
    yvex_backend_close(backend);
    return 0;
}

static int test_attention_success_and_bounds(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *query = NULL;
    yvex_device_tensor *keys = NULL;
    yvex_device_tensor *values = NULL;
    yvex_device_tensor *scores = NULL;
    yvex_device_tensor *probabilities = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_error err;
    float query_data[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float key_data[12] = {
        0.2f, 0.1f, 0.0f, 0.3f,
        0.4f, 0.3f, 0.2f, 0.1f,
        0.1f, 0.5f, 0.3f, 0.2f,
    };
    float value_data[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        10.0f, 20.0f, 30.0f, 40.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    float out_data[4];
    float score_data[3];
    float probability_data[3];
    int rc;
    unsigned int i;

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu attention");

    make_desc(&desc, "attention_query", YVEX_DTYPE_F32, 1, 4, 0, sizeof(query_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &query, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate attention query");
    rc = yvex_backend_tensor_write(backend, query, query_data, sizeof(query_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write attention query");

    make_desc(&desc, "attention_keys", YVEX_DTYPE_F32, 2, 3, 4, sizeof(key_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &keys, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate attention keys");
    rc = yvex_backend_tensor_write(backend, keys, key_data, sizeof(key_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write attention keys");

    make_desc(&desc, "attention_values", YVEX_DTYPE_F32, 2, 3, 4, sizeof(value_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &values, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate attention values");
    rc = yvex_backend_tensor_write(backend, values, value_data, sizeof(value_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write attention values");

    make_desc(&desc, "attention_scores", YVEX_DTYPE_F32, 1, 3, 0, sizeof(score_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &scores, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate attention scores");
    make_desc(&desc, "attention_probabilities", YVEX_DTYPE_F32, 1, 3, 0, sizeof(probability_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &probabilities, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate attention probabilities");
    make_desc(&desc, "attention_out", YVEX_DTYPE_F32, 1, 4, 0, sizeof(out_data));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate attention output");

    rc = yvex_backend_op_attention(backend, query, keys, values, 3, 0, 0.5f, 1,
                                   scores, probabilities, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "attention position zero succeeds");
    rc = yvex_backend_tensor_read(backend, out, out_data, sizeof(out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read attention output");
    for (i = 0; i < 4u; ++i) {
        YVEX_TEST_ASSERT(float_close(out_data[i], value_data[i], 0.000001f),
                         "attention position zero selects first value");
    }

    rc = yvex_backend_op_attention(backend, query, keys, values, 3, 2, 0.5f, 1,
                                   scores, probabilities, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "attention full prefix succeeds");
    rc = yvex_backend_tensor_read(backend, out, out_data, sizeof(out_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read attention prefix output");
    YVEX_TEST_ASSERT(!float_close(out_data[0], value_data[0], 0.0001f),
                     "attention full prefix mixes more than first value");

    rc = yvex_backend_op_attention(backend, query, keys, values, 3, 3, 0.5f, 1,
                                   scores, probabilities, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "attention rejects out-of-range position");

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, probabilities);
    yvex_backend_tensor_free(backend, scores);
    yvex_backend_tensor_free(backend, values);
    yvex_backend_tensor_free(backend, keys);
    yvex_backend_tensor_free(backend, query);
    yvex_backend_close(backend);
    return 0;
}

static int test_matmul_success_and_failures(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *input = NULL;
    yvex_device_tensor *weight = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_error err;
    float projection_input[8] = {0.08f, 0.09f, 0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f};
    float projection_weight[64];
    float projection_out[8];
    float projection_expected[8];
    float matrix_input[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    float matrix_weight[12] = {
        0.5f, 1.0f, 1.5f,
        2.0f, 2.5f, 3.0f,
        3.5f, 4.0f, 4.5f,
        5.0f, 5.5f, 6.0f,
    };
    float matrix_out[6];
    float matrix_expected[6];
    unsigned int i;
    unsigned int row;
    unsigned int col;
    int rc;

    for (i = 0; i < 64u; ++i) {
        projection_weight[i] = (float)(0.01 + ((double)(i + 1u) * 0.001));
    }
    for (col = 0; col < 8u; ++col) {
        double sum = 0.0;
        for (i = 0; i < 8u; ++i) {
            sum += (double)projection_input[i] * (double)projection_weight[(i * 8u) + col];
        }
        projection_expected[col] = (float)sum;
    }
    for (row = 0; row < 2u; ++row) {
        for (col = 0; col < 3u; ++col) {
            double sum = 0.0;
            for (i = 0; i < 4u; ++i) {
                sum += (double)matrix_input[(row * 4u) + i] *
                       (double)matrix_weight[(i * 3u) + col];
            }
            matrix_expected[(row * 3u) + col] = (float)sum;
        }
    }

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu matmul");

    make_desc(&desc, "matmul_projection_input", YVEX_DTYPE_F32, 2, 1, 8, sizeof(projection_input));
    rc = yvex_backend_tensor_alloc(backend, &desc, &input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate projection input");
    rc = yvex_backend_tensor_write(backend, input, projection_input, sizeof(projection_input), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write projection input");
    make_desc(&desc, "matmul_projection_weight", YVEX_DTYPE_F32, 2, 8, 8, sizeof(projection_weight));
    rc = yvex_backend_tensor_alloc(backend, &desc, &weight, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate projection weight");
    rc = yvex_backend_tensor_write(backend, weight, projection_weight, sizeof(projection_weight), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write projection weight");
    make_desc(&desc, "matmul_projection_out", YVEX_DTYPE_F32, 2, 1, 8, sizeof(projection_out));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate projection output");

    rc = yvex_backend_op_matmul(backend, input, weight, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "projection matmul succeeds");
    rc = yvex_backend_tensor_read(backend, out, projection_out, sizeof(projection_out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read projection output");
    for (i = 0; i < 8u; ++i) {
        YVEX_TEST_ASSERT(float_close(projection_out[i], projection_expected[i], 0.000001f),
                         "projection matmul output matches reference");
    }

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, weight);
    yvex_backend_tensor_free(backend, input);
    input = NULL;
    weight = NULL;
    out = NULL;

    make_desc(&desc, "matmul_matrix_input", YVEX_DTYPE_F32, 2, 2, 4, sizeof(matrix_input));
    rc = yvex_backend_tensor_alloc(backend, &desc, &input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate matrix input");
    rc = yvex_backend_tensor_write(backend, input, matrix_input, sizeof(matrix_input), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write matrix input");
    make_desc(&desc, "matmul_matrix_weight", YVEX_DTYPE_F32, 2, 4, 3, sizeof(matrix_weight));
    rc = yvex_backend_tensor_alloc(backend, &desc, &weight, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate matrix weight");
    rc = yvex_backend_tensor_write(backend, weight, matrix_weight, sizeof(matrix_weight), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write matrix weight");
    make_desc(&desc, "matmul_matrix_out", YVEX_DTYPE_F32, 2, 2, 3, sizeof(matrix_out));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate matrix output");

    rc = yvex_backend_op_matmul(backend, input, weight, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "matrix matmul succeeds");
    rc = yvex_backend_tensor_read(backend, out, matrix_out, sizeof(matrix_out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read matrix output");
    for (i = 0; i < 6u; ++i) {
        YVEX_TEST_ASSERT(float_close(matrix_out[i], matrix_expected[i], 0.000001f),
                         "matrix matmul output matches reference");
    }

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, weight);
    yvex_backend_tensor_free(backend, input);
    input = NULL;
    weight = NULL;
    out = NULL;

    make_desc(&desc, "matmul_bad_input", YVEX_DTYPE_BF16, 2, 1, 2, 4);
    rc = yvex_backend_tensor_alloc(backend, &desc, &input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate unsupported matmul input dtype");
    make_desc(&desc, "matmul_bad_weight", YVEX_DTYPE_F32, 2, 2, 1, 2u * sizeof(float));
    rc = yvex_backend_tensor_alloc(backend, &desc, &weight, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate matmul bad dtype weight");
    make_desc(&desc, "matmul_bad_out", YVEX_DTYPE_F32, 2, 1, 1, sizeof(float));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate matmul bad dtype output");
    rc = yvex_backend_op_matmul(backend, input, weight, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_UNSUPPORTED, "matmul rejects unsupported dtype");

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, weight);
    yvex_backend_tensor_free(backend, input);
    yvex_backend_close(backend);
    return 0;
}

static int test_mlp_success_and_failures(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *input = NULL;
    yvex_device_tensor *gate = NULL;
    yvex_device_tensor *up = NULL;
    yvex_device_tensor *down = NULL;
    yvex_device_tensor *intermediate = NULL;
    yvex_device_tensor *out = NULL;
    yvex_backend_tensor_desc desc;
    yvex_mlp_options options;
    yvex_error err;
    float dense_input[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float dense_gate[12] = {
        0.01f, 0.02f, 0.03f,
        0.04f, 0.05f, 0.06f,
        0.07f, 0.08f, 0.09f,
        0.10f, 0.11f, 0.12f,
    };
    float dense_up[12] = {
        0.02f, 0.03f, 0.04f,
        0.05f, 0.06f, 0.07f,
        0.08f, 0.09f, 0.10f,
        0.11f, 0.12f, 0.13f,
    };
    float dense_down[12] = {
        0.03f, 0.04f, 0.05f, 0.06f,
        0.07f, 0.08f, 0.09f, 0.10f,
        0.11f, 0.12f, 0.13f, 0.14f,
    };
    float dense_intermediate[3];
    float dense_out[4];
    float dense_expected_mid[3];
    float dense_expected[4];
    float routed_gate[24];
    float routed_up[24];
    float routed_down[24];
    float routed_out[4];
    float routed_expected_mid[3];
    float routed_expected[4];
    unsigned int i;
    int rc;

    for (i = 0; i < 24u; ++i) {
        routed_gate[i] = (float)(0.01 + ((double)(i + 1u) * 0.001));
        routed_up[i] = (float)(0.015 + ((double)(i + 1u) * 0.0015));
        routed_down[i] = (float)(0.02 + ((double)(i + 1u) * 0.0008));
    }
    mlp_reference(dense_input, dense_gate, dense_up, dense_down,
                  1, 4, 3, 0, 0, dense_expected_mid, dense_expected);
    mlp_reference(dense_input, routed_gate, routed_up, routed_down,
                  1, 4, 3, 1, 1, routed_expected_mid, routed_expected);

    memset(&options, 0, sizeof(options));
    options.batch = 1;
    options.hidden_dim = 4;
    options.ffn_dim = 3;
    options.gated = 1;
    options.activation = "silu";

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu mlp");

    make_desc(&desc, "mlp_input", YVEX_DTYPE_F32, 2, 1, 4, sizeof(dense_input));
    rc = yvex_backend_tensor_alloc(backend, &desc, &input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate mlp input");
    rc = yvex_backend_tensor_write(backend, input, dense_input, sizeof(dense_input), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write mlp input");
    make_desc(&desc, "mlp_gate", YVEX_DTYPE_F32, 2, 4, 3, sizeof(dense_gate));
    rc = yvex_backend_tensor_alloc(backend, &desc, &gate, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate mlp gate");
    rc = yvex_backend_tensor_write(backend, gate, dense_gate, sizeof(dense_gate), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write mlp gate");
    make_desc(&desc, "mlp_up", YVEX_DTYPE_F32, 2, 4, 3, sizeof(dense_up));
    rc = yvex_backend_tensor_alloc(backend, &desc, &up, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate mlp up");
    rc = yvex_backend_tensor_write(backend, up, dense_up, sizeof(dense_up), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write mlp up");
    make_desc(&desc, "mlp_down", YVEX_DTYPE_F32, 2, 3, 4, sizeof(dense_down));
    rc = yvex_backend_tensor_alloc(backend, &desc, &down, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate mlp down");
    rc = yvex_backend_tensor_write(backend, down, dense_down, sizeof(dense_down), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write mlp down");
    make_desc(&desc, "mlp_intermediate", YVEX_DTYPE_F32, 2, 1, 3, sizeof(dense_intermediate));
    rc = yvex_backend_tensor_alloc(backend, &desc, &intermediate, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate mlp intermediate");
    make_desc(&desc, "mlp_out", YVEX_DTYPE_F32, 2, 1, 4, sizeof(dense_out));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate mlp out");

    rc = yvex_backend_op_mlp(backend, input, gate, up, down, &options,
                             intermediate, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "dense mlp succeeds");
    rc = yvex_backend_tensor_read(backend, intermediate, dense_intermediate,
                                  sizeof(dense_intermediate), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read dense mlp intermediate");
    rc = yvex_backend_tensor_read(backend, out, dense_out, sizeof(dense_out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read dense mlp output");
    for (i = 0; i < 3u; ++i) {
        YVEX_TEST_ASSERT(float_close(dense_intermediate[i], dense_expected_mid[i], 0.000001f),
                         "dense mlp intermediate matches reference");
    }
    for (i = 0; i < 4u; ++i) {
        YVEX_TEST_ASSERT(float_close(dense_out[i], dense_expected[i], 0.000001f),
                         "dense mlp output matches reference");
    }

    options.activation = "relu";
    rc = yvex_backend_op_mlp(backend, input, gate, up, down, &options,
                             intermediate, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_UNSUPPORTED, "mlp rejects unsupported activation");
    options.activation = "silu";

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, intermediate);
    yvex_backend_tensor_free(backend, down);
    yvex_backend_tensor_free(backend, up);
    yvex_backend_tensor_free(backend, gate);
    out = NULL;
    intermediate = NULL;
    down = NULL;
    up = NULL;
    gate = NULL;

    options.routed_expert_mode = 1;
    options.expert_count = 2;
    options.expert_id = 1;

    make_desc(&desc, "mlp_routed_gate", YVEX_DTYPE_F32, 3, 2, 4, sizeof(routed_gate));
    desc.dims[2] = 3;
    rc = yvex_backend_tensor_alloc(backend, &desc, &gate, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate routed mlp gate");
    rc = yvex_backend_tensor_write(backend, gate, routed_gate, sizeof(routed_gate), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write routed mlp gate");
    make_desc(&desc, "mlp_routed_up", YVEX_DTYPE_F32, 3, 2, 4, sizeof(routed_up));
    desc.dims[2] = 3;
    rc = yvex_backend_tensor_alloc(backend, &desc, &up, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate routed mlp up");
    rc = yvex_backend_tensor_write(backend, up, routed_up, sizeof(routed_up), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write routed mlp up");
    make_desc(&desc, "mlp_routed_down", YVEX_DTYPE_F32, 3, 2, 3, sizeof(routed_down));
    desc.dims[2] = 4;
    rc = yvex_backend_tensor_alloc(backend, &desc, &down, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate routed mlp down");
    rc = yvex_backend_tensor_write(backend, down, routed_down, sizeof(routed_down), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write routed mlp down");
    make_desc(&desc, "mlp_routed_intermediate", YVEX_DTYPE_F32, 2, 1, 3, sizeof(dense_intermediate));
    rc = yvex_backend_tensor_alloc(backend, &desc, &intermediate, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate routed mlp intermediate");
    make_desc(&desc, "mlp_routed_out", YVEX_DTYPE_F32, 2, 1, 4, sizeof(routed_out));
    rc = yvex_backend_tensor_alloc(backend, &desc, &out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate routed mlp out");

    rc = yvex_backend_op_mlp(backend, input, gate, up, down, &options,
                             intermediate, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "routed mlp succeeds");
    rc = yvex_backend_tensor_read(backend, out, routed_out, sizeof(routed_out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read routed mlp output");
    for (i = 0; i < 4u; ++i) {
        YVEX_TEST_ASSERT(float_close(routed_out[i], routed_expected[i], 0.000001f),
                         "routed mlp output matches selected expert reference");
    }
    options.expert_id = 2;
    rc = yvex_backend_op_mlp(backend, input, gate, up, down, &options,
                             intermediate, out, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "routed mlp rejects expert id out of range");

    yvex_backend_tensor_free(backend, out);
    yvex_backend_tensor_free(backend, intermediate);
    yvex_backend_tensor_free(backend, down);
    yvex_backend_tensor_free(backend, up);
    yvex_backend_tensor_free(backend, gate);
    yvex_backend_tensor_free(backend, input);
    yvex_backend_close(backend);
    return 0;
}

/* Create one sealed backend-neutral recipe with alignment and token scaling. */
static int attention_workspace_recipe_make(yvex_attention_workspace_recipe *recipe,
                                           unsigned long long token_capacity,
                                           yvex_attention_evidence_level evidence_level,
                                           yvex_error *err)
{
    yvex_attention_workspace_component *first, *second, *rolling, *evidence;

    memset(recipe, 0, sizeof(*recipe));
    recipe->schema_version = YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1;
    recipe->layer_index = 7ull;
    recipe->token_capacity = token_capacity;
    recipe->mode = YVEX_ATTENTION_EXECUTION_FULL;
    recipe->scope = YVEX_ATTENTION_OPERATION_ENVELOPE;
    recipe->evidence_level = evidence_level;
    recipe->component_count =
        evidence_level == YVEX_ATTENTION_EVIDENCE_FULL ? 4u : 3u;
    memset(recipe->state_recipe_identity, 'a', YVEX_SHA256_HEX_BYTES - 1u);
    recipe->state_recipe_identity[YVEX_SHA256_HEX_BYTES - 1u] = '\0';
    first = &recipe->components[0];
    *first = (yvex_attention_workspace_component){
        .schema_version = YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1,
        .ordinal = 0u, .kind = YVEX_ATTENTION_WORKSPACE_STATUS,
        .lifetime = YVEX_ATTENTION_WORKSPACE_EXECUTION,
        .element_count = 3ull, .element_width = 4ull, .alignment = 8ull};
    second = &recipe->components[1];
    *second = (yvex_attention_workspace_component){
        .schema_version = YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1,
        .ordinal = 1u, .kind = YVEX_ATTENTION_WORKSPACE_QUERY,
        .lifetime = YVEX_ATTENTION_WORKSPACE_EXECUTION,
        .element_count = 5ull, .element_width = 8ull, .alignment = 16ull,
        .scales_with_tokens = 1};
    rolling = &recipe->components[2];
    *rolling = (yvex_attention_workspace_component){
        .schema_version = YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1,
        .ordinal = 2u, .kind = YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_VALUES,
        .lifetime = YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE,
        .element_count = 2ull, .element_width = 4ull, .alignment = 8ull};
    if (evidence_level == YVEX_ATTENTION_EVIDENCE_FULL) {
        evidence = &recipe->components[3];
        *evidence = (yvex_attention_workspace_component){
            .schema_version = YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1,
            .ordinal = 3u, .kind = YVEX_ATTENTION_WORKSPACE_CORE_INPUT_EVIDENCE,
            .lifetime = YVEX_ATTENTION_WORKSPACE_EXECUTION,
            .element_count = 7ull, .element_width = sizeof(float),
            .alignment = 16ull, .scales_with_tokens = 1};
    }
    return yvex_attention_workspace_recipe_seal(recipe, err);
}

/* Prove recipe lowering is exact, deterministic, identity-bound, and checked. */
static int test_attention_workspace_recipe_lowering(void)
{
    yvex_attention_workspace_recipe recipe, full_recipe;
    char baseline[YVEX_ATTENTION_IDENTITY_CAP];
    unsigned long long first = 0ull, second = 0ull;
    yvex_error err;

    YVEX_TEST_ASSERT(
        attention_workspace_recipe_make(&recipe, 4ull, YVEX_ATTENTION_EVIDENCE_NONE,
                                        &err) == YVEX_OK,
                     "attention workspace fixture seals");
    (void)snprintf(baseline, sizeof(baseline), "%s", recipe.identity);
    YVEX_TEST_ASSERT(
        yvex_backend_attention_workspace_required_from_recipe(&recipe, &first, &err) == YVEX_OK &&
            first == 392ull,
        "backend reserves token-scaled execution, publication, and captured rolling extents");
    YVEX_TEST_ASSERT(
        attention_workspace_recipe_make(&full_recipe, 4ull,
                                        YVEX_ATTENTION_EVIDENCE_FULL, &err) == YVEX_OK &&
            strcmp(baseline, full_recipe.identity) != 0 &&
            yvex_backend_attention_workspace_required_from_recipe(
                &full_recipe, &second, &err) == YVEX_OK &&
            second == 624ull && second > first,
        "full evidence changes identity and reserves its execution and publication staging");
    second = 0ull;
    YVEX_TEST_ASSERT(
        yvex_backend_attention_workspace_required_from_recipe(&recipe, &second, &err) == YVEX_OK &&
            second == first,
        "workspace recipe lowering is deterministic");
    recipe.token_capacity = 5ull;
    second = 99ull;
    YVEX_TEST_ASSERT(
        yvex_backend_attention_workspace_required_from_recipe(&recipe, &second, &err) ==
            YVEX_ERR_STATE && second == 0ull,
        "backend rejects a stale mutated recipe identity");
    YVEX_TEST_ASSERT(
        yvex_backend_attention_workspace_required_from_recipe(NULL, &second, &err) ==
            YVEX_ERR_INVALID_ARG && second == 0ull,
        "backend rejects a missing workspace recipe");
    YVEX_TEST_ASSERT(
        yvex_backend_attention_workspace_required_from_recipe(&recipe, NULL, &err) ==
            YVEX_ERR_INVALID_ARG,
        "backend rejects a missing workspace result");
    YVEX_TEST_ASSERT(
        attention_workspace_recipe_make(&recipe, 2ull, YVEX_ATTENTION_EVIDENCE_NONE,
                                        &err) == YVEX_OK,
                     "overflow workspace fixture seals before lowering");
    recipe.components[1].element_count = ULLONG_MAX;
    recipe.identity[0] = '\0';
    YVEX_TEST_ASSERT(yvex_attention_workspace_recipe_seal(&recipe, &err) == YVEX_OK,
                     "overflow workspace fixture reseals deterministically");
    YVEX_TEST_ASSERT(
        yvex_backend_attention_workspace_required_from_recipe(&recipe, &second, &err) ==
            YVEX_ERR_BOUNDS && second == 0ull,
        "backend refuses workspace arithmetic overflow without publication");
    return 0;
}

int yvex_test_backend_ops(void)
{
    if (test_capabilities() != 0) return 1;
    if (test_embed_success() != 0) return 1;
    if (test_embed_f16_success() != 0) return 1;
    if (test_rms_norm_success() != 0) return 1;
    if (test_rope_success() != 0) return 1;
    if (test_rope_failures() != 0) return 1;
    if (test_attention_success_and_bounds() != 0) return 1;
    if (test_matmul_success_and_failures() != 0) return 1;
    if (test_mlp_success_and_failures() != 0) return 1;
    if (test_embed_failures() != 0) return 1;
    if (test_attention_workspace_recipe_lowering() != 0) return 1;
    return 0;
}
