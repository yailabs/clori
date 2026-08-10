/* Execute admitted MiniMax-H3 operations through generic CUDA primitives. */
#include "src/backend/cuda/private.h"

#include <yvex/backend.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/qtype.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_HIDDEN = 5120u,
    TEXT_VOCAB = 151936u,
    TEXT_MAX_TOKENS = 256u,
    TEXT_IDENTITY_CAP = 65u,
    TEXT_HEAD_DIM = 128u,
    TEXT_Q_HEADS = 64u,
    TEXT_KV_HEADS = 8u,
    TEXT_Q_WIDTH = 8192u,
    TEXT_KV_WIDTH = 1024u,
    TEXT_FFN = 25600u
};

typedef enum {
    TEXT_DEVICE_HIDDEN = 0,
    TEXT_DEVICE_NORM,
    TEXT_DEVICE_NORM_WEIGHT,
    TEXT_DEVICE_QUERY,
    TEXT_DEVICE_KEY,
    TEXT_DEVICE_VALUE,
    TEXT_DEVICE_Q_NORM,
    TEXT_DEVICE_K_NORM,
    TEXT_DEVICE_COSINE,
    TEXT_DEVICE_SINE,
    TEXT_DEVICE_ATTENTION,
    TEXT_DEVICE_RESIDUAL,
    TEXT_DEVICE_GATE,
    TEXT_DEVICE_UP,
    TEXT_DEVICE_PRODUCT,
    TEXT_DEVICE_COUNT
} text_device_slot;

typedef struct {
    yvex_backend *backend;
    const yvex_minimax_h3_encoded_weight *weights;
    yvex_device_tensor *device[TEXT_DEVICE_COUNT];
    unsigned long long tokens, values, output_bytes, device_bytes;
    unsigned long long layer_count, layer_index;
    yvex_minimax_h3_conditioning_result facts;
} text_layer_run;

enum {
    OMNI_HIDDEN = 5376u,
    OMNI_HEADS = 56u,
    OMNI_HEAD_DIM = 128u,
    OMNI_ATTENTION_WIDTH = 7168u,
    OMNI_FFN = 14336u,
    OMNI_TIME = 2688u,
    OMNI_ROTARY = 96u,
    OMNI_MODALITIES = 3u,
    OMNI_PARAMETERS = 6u,
    OMNI_BLOCKS = 50u,
    OMNI_MAX_ROWS = 2048u,
    OMNI_MAX_TIMESTEPS = 64u
};

typedef enum {
    OMNI_DEVICE_HIDDEN = 0,
    OMNI_DEVICE_NORM,
    OMNI_DEVICE_NORM_WEIGHT,
    OMNI_DEVICE_TEMB,
    OMNI_DEVICE_MODULATION,
    OMNI_DEVICE_MODULATION_BIAS,
    OMNI_DEVICE_QKV,
    OMNI_DEVICE_QUERY,
    OMNI_DEVICE_KEY,
    OMNI_DEVICE_VALUE,
    OMNI_DEVICE_Q_NORM,
    OMNI_DEVICE_K_NORM,
    OMNI_DEVICE_COSINE,
    OMNI_DEVICE_SINE,
    OMNI_DEVICE_ATTENTION,
    OMNI_DEVICE_UPDATE,
    OMNI_DEVICE_FC1,
    OMNI_DEVICE_FF,
    OMNI_DEVICE_COUNT
} omni_device_slot;

typedef struct {
    yvex_backend *backend;
    const yvex_minimax_h3_encoded_weight *weights;
    yvex_device_tensor *device[OMNI_DEVICE_COUNT];
    const float *hidden, *temb, *positions;
    const unsigned int *adaln_indices;
    unsigned long long rows, timesteps, values, output_bytes, device_bytes;
    unsigned long long block_count, block_index;
    yvex_minimax_h3_omni_result facts;
} omni_run;

static const unsigned int text_zero_row = 0u;

static int conditioning_refuse(yvex_error *err, yvex_status status, const char *stage,
                               const char *message)
{
    yvex_error_set(err, status, stage, message);
    return status;
}

static int conditioning_identity(
    const char *residency_identity, const unsigned int *token_ids,
    unsigned long long token_count, const float *values, unsigned long long value_count,
    char output[TEXT_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.text-conditioning.cuda.v1") ||
        !yvex_sha256_update_text(&hash, YVEX_MINIMAX_H3_TEXT_COMPONENT_IDENTITY) ||
        !yvex_sha256_update_text(&hash, residency_identity) ||
        !yvex_sha256_update_u64(&hash, token_count) ||
        !yvex_sha256_update_u64(&hash, TEXT_HIDDEN))
        return 0;
    for (index = 0ull; index < token_count; ++index)
        if (!yvex_sha256_update_u64(&hash, token_ids[index])) return 0;
    for (index = 0ull; index < value_count; ++index) {
        uint32_t bits;
        memcpy(&bits, values + index, sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int text_embed_validate(
    yvex_backend *backend, const unsigned char *encoded, unsigned long long encoded_bytes,
    unsigned int qtype, unsigned long long row_count, unsigned long long row_width,
    unsigned long long row_bytes, const char *residency_identity,
    unsigned long long resident_bytes, const unsigned int *token_ids,
    unsigned long long token_count, float *output, unsigned long long output_capacity,
    yvex_minimax_h3_conditioning_result *result, unsigned long long *value_count,
    unsigned long long *output_bytes, yvex_error *err)
{
    unsigned long long expected_encoded, index;

    if (!backend || !encoded || !yvex_sha256_hex_valid(residency_identity) || !resident_bytes ||
        !token_ids || !token_count || token_count > TEXT_MAX_TOKENS || !output || !result ||
        !yvex_core_u64_mul(TEXT_HIDDEN, TEXT_VOCAB * 2ull, &expected_encoded) ||
        encoded_bytes != expected_encoded || qtype != YVEX_GGUF_QTYPE_BF16 ||
        row_count != TEXT_VOCAB ||
        row_width != TEXT_HIDDEN || row_bytes != TEXT_HIDDEN * 2ull ||
        resident_bytes < encoded_bytes ||
        !yvex_core_u64_mul(token_count, TEXT_HIDDEN, value_count) ||
        *value_count > output_capacity ||
        !yvex_core_u64_mul(*value_count, sizeof(float), output_bytes) ||
        *output_bytes > SIZE_MAX)
        return conditioning_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.minimax-h3.text-embedding.validate",
            "admitted BF16 embedding, resident identity, token input, and output are required");
    for (index = 0ull; index < token_count; ++index)
        if (token_ids[index] >= TEXT_VOCAB)
            return conditioning_refuse(
                err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-embedding.token",
                "text token identifier exceeds the exact Qwen3-VL vocabulary");
    return YVEX_OK;
}

static int text_embed_cuda(
    yvex_backend *backend, const unsigned char *encoded, unsigned long long encoded_bytes,
    unsigned int qtype, unsigned long long row_count, unsigned long long row_width,
    unsigned long long row_bytes, const char *residency_identity,
    unsigned long long resident_bytes, const unsigned int *token_ids,
    unsigned long long token_count, float *output, unsigned long long output_capacity,
    yvex_minimax_h3_conditioning_result *result, yvex_error *err)
{
    yvex_backend_cuda_operation_facts operation = {0};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *device_output = NULL;
    yvex_minimax_h3_conditioning_result published = {0};
    float *staged = NULL;
    unsigned long long value_count = 0ull, output_bytes = 0ull;
    int rc, cleanup_rc;
    yvex_error cleanup;

    if (result) memset(result, 0, sizeof(*result));
    rc = text_embed_validate(
        backend, encoded, encoded_bytes, qtype, row_count, row_width, row_bytes,
        residency_identity, resident_bytes, token_ids, token_count, output,
        output_capacity, result, &value_count, &output_bytes, err);
    if (rc == YVEX_OK) {
        staged = (float *)malloc((size_t)output_bytes);
        if (!staged)
            rc = conditioning_refuse(
                err, YVEX_ERR_NOMEM, "cuda.minimax-h3.text-embedding.output",
                "transactional conditioning output allocation failed");
    }
    descriptor.name = "minimax-h3-text-conditioning";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 2u;
    descriptor.dims[0] = token_count;
    descriptor.dims[1] = TEXT_HIDDEN;
    descriptor.bytes = output_bytes;
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_alloc(backend, &descriptor, &device_output, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_cuda_encoded_gather(
            backend, encoded, encoded_bytes, YVEX_GGUF_QTYPE_BF16, row_count,
            row_width, row_bytes, token_ids, token_count, device_output, &operation, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(backend, device_output, staged, output_bytes, err);
    if (rc == YVEX_OK &&
        !conditioning_identity(
            residency_identity, token_ids, token_count, staged, value_count,
            published.execution_identity))
        rc = conditioning_refuse(
            err, YVEX_ERR_STATE, "cuda.minimax-h3.text-embedding.identity",
            "conditioning execution identity could not be sealed");
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_backend_tensor_release(backend, &device_output, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc == YVEX_OK) {
        memcpy(output, staged, (size_t)output_bytes);
        published.token_count = token_count;
        published.hidden_width = TEXT_HIDDEN;
        published.resident_bytes = resident_bytes;
        published.kernel_launches = operation.kernel_launches;
        published.h2d_bytes = operation.h2d_bytes;
        published.d2h_bytes = operation.d2h_bytes + output_bytes;
        published.device_bytes = operation.activation_bytes + operation.temporary_bytes;
        memcpy(published.residency_identity, residency_identity,
               sizeof(published.residency_identity));
        published.complete = 1;
        *result = published;
        yvex_error_clear(err);
    }
    free(staged);
    return rc;
}

static int text_layer_identity(
    const char *residency_identity, const unsigned int *token_ids,
    unsigned long long token_count, unsigned long long layer_count,
    const float *values, unsigned long long value_count,
    char output[TEXT_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.qwen-text-stack.cuda.v1") ||
        !yvex_sha256_update_text(&hash, YVEX_MINIMAX_H3_TEXT_COMPONENT_IDENTITY) ||
        !yvex_sha256_update_text(&hash, residency_identity) ||
        !yvex_sha256_update_u64(&hash, layer_count) ||
        !yvex_sha256_update_u64(&hash, token_count) ||
        !yvex_sha256_update_u64(&hash, TEXT_HIDDEN)) return 0;
    for (index = 0ull; index < token_count; ++index)
        if (!yvex_sha256_update_u64(&hash, token_ids[index])) return 0;
    for (index = 0ull; index < value_count; ++index) {
        uint32_t bits;
        memcpy(&bits, values + index, sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int text_weights_validate(
    const yvex_minimax_h3_encoded_weight *weights, unsigned long long layer_count,
    unsigned long long *weight_bytes)
{
    static const unsigned long long rows[YVEX_MINIMAX_H3_TEXT_WEIGHT_COUNT] = {
        TEXT_VOCAB, 1u, TEXT_Q_WIDTH, TEXT_KV_WIDTH, TEXT_KV_WIDTH, TEXT_HIDDEN,
        1u, 1u, 1u, TEXT_FFN, TEXT_FFN, TEXT_HIDDEN,
    };
    static const unsigned long long widths[YVEX_MINIMAX_H3_TEXT_WEIGHT_COUNT] = {
        TEXT_HIDDEN, TEXT_HIDDEN, TEXT_HIDDEN, TEXT_HIDDEN, TEXT_HIDDEN, TEXT_Q_WIDTH,
        TEXT_HEAD_DIM, TEXT_HEAD_DIM, TEXT_HIDDEN, TEXT_HIDDEN, TEXT_HIDDEN, TEXT_FFN,
    };
    unsigned long long count, index;
    if (weight_bytes) *weight_bytes = 0ull;
    if (!weights || !layer_count ||
        layer_count > YVEX_MINIMAX_H3_TEXT_CONDITIONING_LAYERS || !weight_bytes ||
        !yvex_core_u64_mul(layer_count, YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT, &count) ||
        !yvex_core_u64_add(count, 1ull, &count)) return 0;
    for (index = 0ull; index < count; ++index) {
        const yvex_minimax_h3_encoded_weight *weight = weights + index;
        unsigned long long expected, slot = index ? 1ull +
            (index - 1ull) % YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT : 0ull;
        if (!weight->encoded || weight->qtype != YVEX_GGUF_QTYPE_BF16 ||
            weight->row_count != rows[slot] || weight->row_width != widths[slot] ||
            !yvex_core_u64_mul(widths[slot], 2ull, &expected) ||
            weight->row_bytes != expected ||
            !yvex_core_u64_mul(rows[slot], expected, &expected) ||
            weight->encoded_bytes != expected ||
            !yvex_core_u64_add(*weight_bytes, weight->encoded_bytes, weight_bytes)) return 0;
    }
    return 1;
}

static int text_facts_add(yvex_minimax_h3_conditioning_result *total,
                          const yvex_backend_cuda_operation_facts *part)
{
    return total && part && part->compulsory_memory_facts_available &&
           yvex_core_u64_add(total->kernel_launches, part->kernel_launches,
                             &total->kernel_launches) &&
           yvex_core_u64_add(total->h2d_bytes, part->h2d_bytes, &total->h2d_bytes) &&
           yvex_core_u64_add(total->d2h_bytes, part->d2h_bytes, &total->d2h_bytes);
}

static int text_tensor_allocate(text_layer_run *run, text_device_slot slot,
                                const char *name, unsigned long long rows,
                                unsigned long long width, int rank_one, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long elements, bytes, next;
    if (!run || slot >= TEXT_DEVICE_COUNT || !name || !rows || !width ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) ||
        !yvex_core_u64_add(run->device_bytes, bytes, &next))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.allocate",
                                   "text activation allocation geometry overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = rank_one ? 1u : 2u;
    descriptor.dims[0] = rank_one ? width : rows;
    descriptor.dims[1] = rank_one ? 0ull : width;
    descriptor.bytes = bytes;
    if (yvex_backend_tensor_alloc(run->backend, &descriptor, &run->device[slot], err) != YVEX_OK)
        return yvex_error_code(err);
    run->device_bytes = next;
    return YVEX_OK;
}

static int text_devices_prepare(text_layer_run *run, yvex_error *err)
{
    int rc;
#define ALLOC(slot, name, rows, width, rank_one) \
    if (rc == YVEX_OK) rc = text_tensor_allocate(run, slot, name, rows, width, rank_one, err)
    rc = text_tensor_allocate(run, TEXT_DEVICE_HIDDEN, "text-hidden",
                              run->tokens, TEXT_HIDDEN, 0, err);
    ALLOC(TEXT_DEVICE_NORM, "text-norm", run->tokens, TEXT_HIDDEN, 0);
    ALLOC(TEXT_DEVICE_NORM_WEIGHT, "text-norm-weight", 1ull, TEXT_HIDDEN, 1);
    ALLOC(TEXT_DEVICE_QUERY, "text-query", run->tokens * TEXT_Q_HEADS, TEXT_HEAD_DIM, 0);
    ALLOC(TEXT_DEVICE_KEY, "text-key", run->tokens * TEXT_KV_HEADS, TEXT_HEAD_DIM, 0);
    ALLOC(TEXT_DEVICE_VALUE, "text-value", run->tokens * TEXT_KV_HEADS, TEXT_HEAD_DIM, 0);
    ALLOC(TEXT_DEVICE_Q_NORM, "text-q-norm", 1ull, TEXT_HEAD_DIM, 1);
    ALLOC(TEXT_DEVICE_K_NORM, "text-k-norm", 1ull, TEXT_HEAD_DIM, 1);
    ALLOC(TEXT_DEVICE_COSINE, "text-cosine", run->tokens, TEXT_HEAD_DIM, 0);
    ALLOC(TEXT_DEVICE_SINE, "text-sine", run->tokens, TEXT_HEAD_DIM, 0);
    ALLOC(TEXT_DEVICE_ATTENTION, "text-attention", run->tokens, TEXT_Q_WIDTH, 0);
    ALLOC(TEXT_DEVICE_RESIDUAL, "text-residual", run->tokens, TEXT_HIDDEN, 0);
    ALLOC(TEXT_DEVICE_GATE, "text-gate", run->tokens, TEXT_FFN, 0);
    ALLOC(TEXT_DEVICE_UP, "text-up", run->tokens, TEXT_FFN, 0);
    ALLOC(TEXT_DEVICE_PRODUCT, "text-product", run->tokens, TEXT_FFN, 0);
#undef ALLOC
    return rc;
}

static const yvex_minimax_h3_encoded_weight *text_weight(
    const text_layer_run *run, yvex_minimax_h3_text_weight_slot slot)
{
    if (slot == YVEX_MINIMAX_H3_TEXT_EMBEDDING) return run->weights;
    return run->weights + 1ull +
           run->layer_index * YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT + slot - 1ull;
}

static int text_weight_gather(text_layer_run *run, yvex_minimax_h3_text_weight_slot slot,
                              yvex_device_tensor *output, const unsigned int *rows,
                              unsigned long long row_count, yvex_error *err)
{
    const yvex_minimax_h3_encoded_weight *weight = text_weight(run, slot);
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_gather(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes,
        rows, row_count, output, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text gather accounting overflowed");
    return rc;
}

static int text_weight_project(text_layer_run *run, yvex_minimax_h3_text_weight_slot slot,
                               const yvex_device_tensor *input,
                               const yvex_device_tensor *additive,
                               yvex_device_tensor *output, yvex_error *err)
{
    const yvex_minimax_h3_encoded_weight *weight = text_weight(run, slot);
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_matvec(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes, run->tokens,
        input, NULL, 0ull, additive, output, 0, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text projection accounting overflowed");
    return rc;
}

static int text_round(text_layer_run *run, text_device_slot slot,
                      unsigned long long count, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_cuda_transformer_bf16_round(
        run->backend, run->device[slot], count, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text rounding accounting overflowed");
    return rc;
}

static int text_norm(text_layer_run *run, text_device_slot input,
                     yvex_minimax_h3_text_weight_slot weight,
                     text_device_slot output, unsigned long long count,
                     yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    text_device_slot weight_device =
        weight == YVEX_MINIMAX_H3_TEXT_Q_NORM ? TEXT_DEVICE_Q_NORM
        : weight == YVEX_MINIMAX_H3_TEXT_K_NORM ? TEXT_DEVICE_K_NORM
                                                : TEXT_DEVICE_NORM_WEIGHT;
    unsigned long long width =
        weight == YVEX_MINIMAX_H3_TEXT_Q_NORM || weight == YVEX_MINIMAX_H3_TEXT_K_NORM
            ? TEXT_HEAD_DIM : TEXT_HIDDEN;
    int rc = text_weight_gather(
        run, weight, run->device[weight_device], &text_zero_row, 1ull, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rms_norm_bf16(
            run->backend, run->device[input], run->device[weight_device],
            run->device[output], count / width, width, 1.0e-6f, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text normalization accounting overflowed");
    return rc;
}

static float text_bf16_value(float value)
{
    return yvex_quant_bf16_decode(yvex_quant_bf16_encode(value));
}

static int text_rope_tables(text_layer_run *run, yvex_error *err)
{
    float *cosines = NULL, *sines = NULL;
    unsigned long long token, coordinate, elements, bytes;
    int rc = YVEX_OK;
    if (!yvex_core_u64_mul(run->tokens, TEXT_HEAD_DIM, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) || bytes > SIZE_MAX)
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.rope",
                                   "text rotary table geometry overflowed");
    cosines = (float *)malloc((size_t)bytes);
    sines = (float *)malloc((size_t)bytes);
    if (!cosines || !sines)
        rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.minimax-h3.text-layer.rope",
                                 "text rotary table allocation failed");
    for (token = 0ull; rc == YVEX_OK && token < run->tokens; ++token) {
        for (coordinate = 0ull; coordinate < TEXT_HEAD_DIM; ++coordinate) {
            unsigned long long pair = coordinate % (TEXT_HEAD_DIM / 2ull);
            double frequency = pow(5000000.0, -(double)(2ull * pair) / TEXT_HEAD_DIM);
            double angle = (double)token * frequency;
            unsigned long long index = token * TEXT_HEAD_DIM + coordinate;
            cosines[index] = text_bf16_value((float)cos(angle));
            sines[index] = text_bf16_value((float)sin(angle));
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[TEXT_DEVICE_COSINE],
                                       cosines, bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[TEXT_DEVICE_SINE],
                                       sines, bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->facts.h2d_bytes, bytes, &run->facts.h2d_bytes) ||
         !yvex_core_u64_add(run->facts.h2d_bytes, bytes, &run->facts.h2d_bytes)))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text rotary upload accounting overflowed");
    free(sines);
    free(cosines);
    return rc;
}

static int text_attention(text_layer_run *run, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    unsigned long long q_values = run->tokens * TEXT_Q_WIDTH;
    unsigned long long kv_values = run->tokens * TEXT_KV_WIDTH;
    int rc = text_weight_project(run, YVEX_MINIMAX_H3_TEXT_Q_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_QUERY], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_QUERY, q_values, err);
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_MINIMAX_H3_TEXT_K_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_KEY], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_KEY, kv_values, err);
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_MINIMAX_H3_TEXT_V_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_VALUE], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_VALUE, kv_values, err);
    if (rc == YVEX_OK)
        rc = text_norm(run, TEXT_DEVICE_QUERY, YVEX_MINIMAX_H3_TEXT_Q_NORM,
                       TEXT_DEVICE_QUERY, q_values, err);
    if (rc == YVEX_OK)
        rc = text_norm(run, TEXT_DEVICE_KEY, YVEX_MINIMAX_H3_TEXT_K_NORM,
                       TEXT_DEVICE_KEY, kv_values, err);
    if (rc == YVEX_OK) rc = text_rope_tables(run, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half(
            run->backend, run->device[TEXT_DEVICE_QUERY], run->device[TEXT_DEVICE_COSINE],
            run->device[TEXT_DEVICE_SINE], run->tokens, TEXT_Q_HEADS, TEXT_HEAD_DIM,
            TEXT_HEAD_DIM, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text rotary accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half(
            run->backend, run->device[TEXT_DEVICE_KEY], run->device[TEXT_DEVICE_COSINE],
            run->device[TEXT_DEVICE_SINE], run->tokens, TEXT_KV_HEADS, TEXT_HEAD_DIM,
            TEXT_HEAD_DIM, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text rotary accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gqa(
            run->backend, run->device[TEXT_DEVICE_QUERY], run->device[TEXT_DEVICE_KEY],
            run->device[TEXT_DEVICE_VALUE], run->device[TEXT_DEVICE_ATTENTION],
            run->tokens, TEXT_Q_HEADS, TEXT_KV_HEADS, TEXT_HEAD_DIM, 1, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text attention accounting overflowed");
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_ATTENTION, q_values, err);
    return rc;
}

static int text_mlp(text_layer_run *run, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    unsigned long long ffn_values = run->tokens * TEXT_FFN;
    int rc = text_norm(run, TEXT_DEVICE_RESIDUAL, YVEX_MINIMAX_H3_TEXT_POST_NORM,
                       TEXT_DEVICE_NORM, run->values, err);
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_MINIMAX_H3_TEXT_GATE_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_GATE], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_GATE, ffn_values, err);
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_MINIMAX_H3_TEXT_UP_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_UP], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_UP, ffn_values, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_silu_product_bf16(
            run->backend, run->device[TEXT_DEVICE_GATE], run->device[TEXT_DEVICE_UP],
            run->device[TEXT_DEVICE_PRODUCT], ffn_values, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text MLP accounting overflowed");
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_MINIMAX_H3_TEXT_DOWN_PROJECTION,
                                 run->device[TEXT_DEVICE_PRODUCT],
                                 run->device[TEXT_DEVICE_RESIDUAL],
                                 run->device[TEXT_DEVICE_HIDDEN], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_HIDDEN, run->values, err);
    return rc;
}

static int text_layer_compute(text_layer_run *run, const unsigned int *token_ids,
                              float *staged, yvex_error *err)
{
    int rc = text_devices_prepare(run, err);
    if (rc == YVEX_OK)
        rc = text_weight_gather(run, YVEX_MINIMAX_H3_TEXT_EMBEDDING,
                                run->device[TEXT_DEVICE_HIDDEN], token_ids,
                                run->tokens, err);
    for (run->layer_index = 0ull; rc == YVEX_OK && run->layer_index < run->layer_count;
         ++run->layer_index) {
        rc = text_norm(run, TEXT_DEVICE_HIDDEN, YVEX_MINIMAX_H3_TEXT_INPUT_NORM,
                       TEXT_DEVICE_NORM, run->values, err);
        if (rc == YVEX_OK) rc = text_attention(run, err);
        if (rc == YVEX_OK)
            rc = text_weight_project(run, YVEX_MINIMAX_H3_TEXT_O_PROJECTION,
                                     run->device[TEXT_DEVICE_ATTENTION],
                                     run->device[TEXT_DEVICE_HIDDEN],
                                     run->device[TEXT_DEVICE_RESIDUAL], err);
        if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_RESIDUAL, run->values, err);
        if (rc == YVEX_OK) rc = text_mlp(run, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(run->backend, run->device[TEXT_DEVICE_HIDDEN],
                                      staged, run->output_bytes, err);
    if (rc == YVEX_OK &&
        !yvex_core_u64_add(run->facts.d2h_bytes, run->output_bytes, &run->facts.d2h_bytes))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.text-layer.facts",
                                 "text output accounting overflowed");
    return rc;
}

static int text_devices_release(text_layer_run *run, int rc, yvex_error *err)
{
    int slot;
    for (slot = TEXT_DEVICE_COUNT - 1; slot >= 0; --slot) {
        yvex_error cleanup;
        int cleanup_rc;
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(run->backend, &run->device[slot], &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    return rc;
}

static int text_layer_cuda(
    yvex_backend *backend,
    const yvex_minimax_h3_encoded_weight *weights, unsigned long long layer_count,
    const char *residency_identity, unsigned long long resident_bytes,
    const unsigned int *token_ids, unsigned long long token_count, float *output,
    unsigned long long output_capacity, yvex_minimax_h3_conditioning_result *result,
    yvex_error *err)
{
    text_layer_run run = {0};
    yvex_minimax_h3_conditioning_result published = {0};
    float *staged = NULL;
    unsigned long long index, weight_bytes = 0ull;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!backend || !text_weights_validate(weights, layer_count, &weight_bytes) ||
        !yvex_sha256_hex_valid(residency_identity) || resident_bytes < weight_bytes || !token_ids ||
        !token_count || token_count > TEXT_MAX_TOKENS || !output || !result ||
        !yvex_core_u64_mul(token_count, TEXT_HIDDEN, &run.values) ||
        run.values > output_capacity ||
        !yvex_core_u64_mul(run.values, sizeof(float), &run.output_bytes) ||
        run.output_bytes > SIZE_MAX)
        return conditioning_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.minimax-h3.text-layer.validate",
            "exact Qwen BF16 stack weights, tokens, residency, and output are required");
    for (index = 0ull; index < token_count; ++index)
        if (token_ids[index] >= TEXT_VOCAB)
            return conditioning_refuse(err, YVEX_ERR_BOUNDS,
                                       "cuda.minimax-h3.text-layer.token",
                                       "text token exceeds the exact Qwen3-VL vocabulary");
    staged = (float *)malloc((size_t)run.output_bytes);
    if (!staged)
        return conditioning_refuse(err, YVEX_ERR_NOMEM,
                                   "cuda.minimax-h3.text-layer.output",
                                   "transactional layer output allocation failed");
    run.backend = backend;
    run.weights = weights;
    run.tokens = token_count;
    run.layer_count = layer_count;
    rc = text_layer_compute(&run, token_ids, staged, err);
    if (rc == YVEX_OK &&
        !text_layer_identity(residency_identity, token_ids, token_count, layer_count, staged,
                             run.values, published.execution_identity))
        rc = conditioning_refuse(err, YVEX_ERR_STATE,
                                 "cuda.minimax-h3.text-layer.identity",
                                 "text layer execution identity could not be sealed");
    rc = text_devices_release(&run, rc, err);
    if (rc == YVEX_OK) {
        memcpy(output, staged, (size_t)run.output_bytes);
        published.token_count = token_count;
        published.hidden_width = TEXT_HIDDEN;
        published.layer_count = layer_count;
        published.resident_bytes = resident_bytes;
        published.kernel_launches = run.facts.kernel_launches;
        published.h2d_bytes = run.facts.h2d_bytes;
        published.d2h_bytes = run.facts.d2h_bytes;
        published.device_bytes = run.device_bytes;
        memcpy(published.residency_identity, residency_identity,
               sizeof(published.residency_identity));
        published.complete = 1;
        *result = published;
        yvex_error_clear(err);
    }
    free(staged);
    return rc;
}

static int omni_facts_add(yvex_minimax_h3_omni_result *total,
                          const yvex_backend_cuda_operation_facts *part)
{
    return total && part && part->compulsory_memory_facts_available &&
           yvex_core_u64_add(total->kernel_launches, part->kernel_launches,
                             &total->kernel_launches) &&
           yvex_core_u64_add(total->h2d_bytes, part->h2d_bytes, &total->h2d_bytes) &&
           yvex_core_u64_add(total->d2h_bytes, part->d2h_bytes, &total->d2h_bytes);
}

static const yvex_minimax_h3_encoded_weight *omni_weight(
    const omni_run *run, yvex_minimax_h3_omni_weight_slot slot)
{
    return run->weights + run->block_index * YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT + slot;
}

static int omni_weights_validate(const yvex_minimax_h3_encoded_weight *weights,
                                 unsigned long long block_count,
                                 unsigned long long *weight_bytes)
{
    static const unsigned long long rows[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT] = {
        1u, 3u * OMNI_ATTENTION_WIDTH, 1u, 1u, OMNI_HIDDEN,
        1u, 2u * OMNI_FFN, OMNI_HIDDEN, OMNI_PARAMETERS * OMNI_MODALITIES * OMNI_HIDDEN, 1u,
    };
    static const unsigned long long widths[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT] = {
        OMNI_HIDDEN, OMNI_HIDDEN, OMNI_HEAD_DIM, OMNI_HEAD_DIM, OMNI_ATTENTION_WIDTH,
        OMNI_HIDDEN, OMNI_HIDDEN, OMNI_FFN, OMNI_TIME,
        OMNI_PARAMETERS * OMNI_MODALITIES * OMNI_HIDDEN,
    };
    unsigned long long count, index;
    if (weight_bytes) *weight_bytes = 0ull;
    if (!weights || !block_count || block_count > OMNI_BLOCKS || !weight_bytes ||
        !yvex_core_u64_mul(block_count, YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT, &count))
        return 0;
    for (index = 0ull; index < count; ++index) {
        const yvex_minimax_h3_encoded_weight *weight = weights + index;
        unsigned long long expected, slot = index % YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT;
        if (!weight->encoded || weight->qtype != YVEX_GGUF_QTYPE_BF16 ||
            weight->row_count != rows[slot] || weight->row_width != widths[slot] ||
            !yvex_core_u64_mul(widths[slot], 2ull, &expected) ||
            weight->row_bytes != expected ||
            !yvex_core_u64_mul(rows[slot], expected, &expected) ||
            weight->encoded_bytes != expected ||
            !yvex_core_u64_add(*weight_bytes, expected, weight_bytes)) return 0;
    }
    return 1;
}

static int omni_validate(omni_run *run, const char *residency_identity,
                         unsigned long long resident_bytes, float *output,
                         unsigned long long output_capacity,
                         yvex_minimax_h3_omni_result *result,
                         unsigned long long *weight_bytes, yvex_error *err)
{
    unsigned long long row;
    if (!run || !run->backend ||
        !omni_weights_validate(run->weights, run->block_count, weight_bytes) ||
        !yvex_sha256_hex_valid(residency_identity) || resident_bytes < *weight_bytes ||
        !run->hidden || !run->temb || !run->positions || !run->adaln_indices ||
        !run->rows || run->rows > OMNI_MAX_ROWS || !run->timesteps ||
        run->timesteps > OMNI_MAX_TIMESTEPS || !output || !result ||
        !yvex_core_u64_mul(run->rows, OMNI_HIDDEN, &run->values) ||
        run->values > output_capacity ||
        !yvex_core_u64_mul(run->values, sizeof(float), &run->output_bytes) ||
        run->output_bytes > SIZE_MAX)
        return conditioning_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.minimax-h3.omni.validate",
            "exact Omni BF16 block weights, packed inputs, residency, and output are required");
    for (row = 0ull; row < run->rows; ++row)
        if ((unsigned long long)run->adaln_indices[row] >= run->timesteps * OMNI_MODALITIES)
            return conditioning_refuse(err, YVEX_ERR_BOUNDS,
                                       "cuda.minimax-h3.omni.adaln-index",
                                       "packed row selects an unavailable timestep/modality table row");
    return YVEX_OK;
}

static int omni_tensor_allocate(omni_run *run, omni_device_slot slot,
                                const char *name, unsigned long long rows,
                                unsigned long long width, int rank_one, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long elements, bytes, next;
    if (!run || slot >= OMNI_DEVICE_COUNT || !name || !rows || !width ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) ||
        !yvex_core_u64_add(run->device_bytes, bytes, &next))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.allocate",
                                   "Omni activation allocation geometry overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = rank_one ? 1u : 2u;
    descriptor.dims[0] = rank_one ? width : rows;
    descriptor.dims[1] = rank_one ? 0ull : width;
    descriptor.bytes = bytes;
    if (yvex_backend_tensor_alloc(run->backend, &descriptor, &run->device[slot], err) != YVEX_OK)
        return yvex_error_code(err);
    run->device_bytes = next;
    return YVEX_OK;
}

static int omni_devices_prepare(omni_run *run, yvex_error *err)
{
    int rc;
#define OMNI_ALLOC(slot, name, rows, width, rank_one) \
    if (rc == YVEX_OK) rc = omni_tensor_allocate(run, slot, name, rows, width, rank_one, err)
    rc = omni_tensor_allocate(run, OMNI_DEVICE_HIDDEN, "omni-hidden",
                              run->rows, OMNI_HIDDEN, 0, err);
    OMNI_ALLOC(OMNI_DEVICE_NORM, "omni-norm", run->rows, OMNI_HIDDEN, 0);
    OMNI_ALLOC(OMNI_DEVICE_NORM_WEIGHT, "omni-norm-weight", 1ull, OMNI_HIDDEN, 1);
    OMNI_ALLOC(OMNI_DEVICE_TEMB, "omni-temb", run->timesteps, OMNI_TIME, 0);
    OMNI_ALLOC(OMNI_DEVICE_MODULATION, "omni-modulation", run->timesteps,
               OMNI_PARAMETERS * OMNI_MODALITIES * OMNI_HIDDEN, 0);
    OMNI_ALLOC(OMNI_DEVICE_MODULATION_BIAS, "omni-modulation-bias", 1ull,
               OMNI_PARAMETERS * OMNI_MODALITIES * OMNI_HIDDEN, 1);
    OMNI_ALLOC(OMNI_DEVICE_QKV, "omni-qkv", run->rows, 3ull * OMNI_ATTENTION_WIDTH, 0);
    OMNI_ALLOC(OMNI_DEVICE_QUERY, "omni-query", run->rows, OMNI_ATTENTION_WIDTH, 0);
    OMNI_ALLOC(OMNI_DEVICE_KEY, "omni-key", run->rows, OMNI_ATTENTION_WIDTH, 0);
    OMNI_ALLOC(OMNI_DEVICE_VALUE, "omni-value", run->rows, OMNI_ATTENTION_WIDTH, 0);
    OMNI_ALLOC(OMNI_DEVICE_Q_NORM, "omni-q-norm", 1ull, OMNI_HEAD_DIM, 1);
    OMNI_ALLOC(OMNI_DEVICE_K_NORM, "omni-k-norm", 1ull, OMNI_HEAD_DIM, 1);
    OMNI_ALLOC(OMNI_DEVICE_COSINE, "omni-cosine", run->rows, OMNI_ROTARY, 0);
    OMNI_ALLOC(OMNI_DEVICE_SINE, "omni-sine", run->rows, OMNI_ROTARY, 0);
    OMNI_ALLOC(OMNI_DEVICE_ATTENTION, "omni-attention", run->rows, OMNI_ATTENTION_WIDTH, 0);
    OMNI_ALLOC(OMNI_DEVICE_UPDATE, "omni-update", run->rows, OMNI_HIDDEN, 0);
    OMNI_ALLOC(OMNI_DEVICE_FC1, "omni-fc1", run->rows, 2ull * OMNI_FFN, 0);
    OMNI_ALLOC(OMNI_DEVICE_FF, "omni-ff", run->rows, OMNI_FFN, 0);
#undef OMNI_ALLOC
    return rc;
}

static int omni_weight_gather(omni_run *run, yvex_minimax_h3_omni_weight_slot slot,
                              yvex_device_tensor *output, yvex_error *err)
{
    const yvex_minimax_h3_encoded_weight *weight = omni_weight(run, slot);
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_gather(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes,
        &text_zero_row, 1ull, output, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni gather accounting overflowed");
    return rc;
}

static int omni_weight_project(omni_run *run, yvex_minimax_h3_omni_weight_slot slot,
                               unsigned long long rows, const yvex_device_tensor *input,
                               yvex_device_tensor *output, yvex_error *err)
{
    const yvex_minimax_h3_encoded_weight *weight = omni_weight(run, slot);
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_matvec(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes, rows,
        input, NULL, 0ull, NULL, output, 0, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni projection accounting overflowed");
    return rc;
}

static int omni_round(omni_run *run, omni_device_slot slot,
                      unsigned long long count, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_cuda_transformer_bf16_round(
        run->backend, run->device[slot], count, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni rounding accounting overflowed");
    return rc;
}

static int omni_norm(omni_run *run, omni_device_slot input,
                     yvex_minimax_h3_omni_weight_slot weight,
                     omni_device_slot weight_device, omni_device_slot output,
                     unsigned long long rows, unsigned long long width,
                     float epsilon, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    int rc = omni_weight_gather(run, weight, run->device[weight_device], err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rms_norm_bf16(
            run->backend, run->device[input], run->device[weight_device],
            run->device[output], rows, width, epsilon, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni normalization accounting overflowed");
    return rc;
}

static int omni_upload_inputs(omni_run *run, yvex_error *err)
{
    unsigned long long temb_values, temb_bytes;
    int rc;
    if (!yvex_core_u64_mul(run->timesteps, OMNI_TIME, &temb_values) ||
        !yvex_core_u64_mul(temb_values, sizeof(float), &temb_bytes))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.input",
                                   "Omni packed input byte geometry overflowed");
    rc = yvex_backend_tensor_write(run->backend, run->device[OMNI_DEVICE_HIDDEN],
                                   run->hidden, run->output_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[OMNI_DEVICE_TEMB],
                                       run->temb, temb_bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->facts.h2d_bytes, run->output_bytes,
                            &run->facts.h2d_bytes) ||
         !yvex_core_u64_add(run->facts.h2d_bytes, temb_bytes, &run->facts.h2d_bytes)))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni input upload accounting overflowed");
    return rc;
}

static int omni_rope_tables(omni_run *run, yvex_error *err)
{
    float *cosines = NULL, *sines = NULL;
    unsigned long long row, axis, frequency_index, half_index, elements, bytes;
    int rc = YVEX_OK;
    if (!yvex_core_u64_mul(run->rows, OMNI_ROTARY, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) || bytes > SIZE_MAX)
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.rope",
                                   "Omni MM-RoPE table geometry overflowed");
    cosines = (float *)malloc((size_t)bytes);
    sines = (float *)malloc((size_t)bytes);
    if (!cosines || !sines)
        rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.minimax-h3.omni.rope",
                                 "Omni MM-RoPE table allocation failed");
    for (row = 0ull; rc == YVEX_OK && row < run->rows; ++row) {
        for (axis = 0ull; axis < 3ull; ++axis) {
            for (frequency_index = 0ull; frequency_index < 16ull; ++frequency_index) {
                double frequency = pow(10000.0, -(double)(2ull * frequency_index) / 32.0);
                double angle = (double)run->positions[row * 3ull + axis] * frequency;
                unsigned long long coordinate = axis * 16ull + frequency_index;
                for (half_index = 0ull; half_index < 2ull; ++half_index) {
                    unsigned long long index =
                        row * OMNI_ROTARY + half_index * (OMNI_ROTARY / 2ull) + coordinate;
                    cosines[index] = text_bf16_value((float)cos(angle));
                    sines[index] = text_bf16_value((float)sin(angle));
                }
            }
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[OMNI_DEVICE_COSINE],
                                       cosines, bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[OMNI_DEVICE_SINE],
                                       sines, bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->facts.h2d_bytes, bytes, &run->facts.h2d_bytes) ||
         !yvex_core_u64_add(run->facts.h2d_bytes, bytes, &run->facts.h2d_bytes)))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni MM-RoPE upload accounting overflowed");
    free(sines);
    free(cosines);
    return rc;
}

static int omni_modulation(omni_run *run, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    unsigned long long temb_values = run->timesteps * OMNI_TIME;
    unsigned long long table_width = OMNI_PARAMETERS * OMNI_MODALITIES * OMNI_HIDDEN;
    int rc = yvex_cuda_transformer_silu(
        run->backend, run->device[OMNI_DEVICE_TEMB], run->device[OMNI_DEVICE_TEMB],
        temb_values, 1, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni timestep activation accounting overflowed");
    if (rc == YVEX_OK)
        rc = omni_weight_project(run, YVEX_MINIMAX_H3_OMNI_ADALN_WEIGHT,
                                 run->timesteps, run->device[OMNI_DEVICE_TEMB],
                                 run->device[OMNI_DEVICE_MODULATION], err);
    if (rc == YVEX_OK) rc = omni_round(run, OMNI_DEVICE_MODULATION,
                                       run->timesteps * table_width, err);
    if (rc == YVEX_OK)
        rc = omni_weight_gather(run, YVEX_MINIMAX_H3_OMNI_ADALN_BIAS,
                                run->device[OMNI_DEVICE_MODULATION_BIAS], err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(
            run->backend, run->device[OMNI_DEVICE_MODULATION],
            run->device[OMNI_DEVICE_MODULATION_BIAS],
            run->device[OMNI_DEVICE_MODULATION], run->timesteps, table_width,
            1, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni AdaLN bias accounting overflowed");
    return rc;
}

static int omni_modulate(omni_run *run, unsigned int shift_slot,
                         unsigned int scale_slot, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_cuda_transformer_modulate_bf16(
        run->backend, run->device[OMNI_DEVICE_NORM],
        run->device[OMNI_DEVICE_MODULATION], run->adaln_indices,
        run->device[OMNI_DEVICE_NORM], run->rows, OMNI_HIDDEN,
        run->timesteps * OMNI_MODALITIES, OMNI_PARAMETERS,
        shift_slot, scale_slot, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni modulation accounting overflowed");
    return rc;
}

static int omni_residual(omni_run *run, unsigned int gate_slot, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_cuda_transformer_gated_residual_bf16(
        run->backend, run->device[OMNI_DEVICE_HIDDEN],
        run->device[OMNI_DEVICE_MODULATION], run->adaln_indices,
        run->device[OMNI_DEVICE_UPDATE], run->device[OMNI_DEVICE_HIDDEN],
        run->rows, OMNI_HIDDEN, run->timesteps * OMNI_MODALITIES,
        OMNI_PARAMETERS, gate_slot, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni gated residual accounting overflowed");
    return rc;
}

static int omni_attention(omni_run *run, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    unsigned long long attention_values = run->rows * OMNI_ATTENTION_WIDTH;
    int rc = omni_weight_project(run, YVEX_MINIMAX_H3_OMNI_QKV, run->rows,
                                 run->device[OMNI_DEVICE_NORM],
                                 run->device[OMNI_DEVICE_QKV], err);
    if (rc == YVEX_OK) rc = omni_round(run, OMNI_DEVICE_QKV, 3ull * attention_values, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_split_three(
            run->backend, run->device[OMNI_DEVICE_QKV],
            run->device[OMNI_DEVICE_QUERY], run->device[OMNI_DEVICE_KEY],
            run->device[OMNI_DEVICE_VALUE], run->rows, OMNI_ATTENTION_WIDTH,
            &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni QKV split accounting overflowed");
    if (rc == YVEX_OK)
        rc = omni_norm(run, OMNI_DEVICE_QUERY, YVEX_MINIMAX_H3_OMNI_Q_NORM,
                       OMNI_DEVICE_Q_NORM, OMNI_DEVICE_QUERY,
                       run->rows * OMNI_HEADS, OMNI_HEAD_DIM, 1.0e-5f, err);
    if (rc == YVEX_OK)
        rc = omni_norm(run, OMNI_DEVICE_KEY, YVEX_MINIMAX_H3_OMNI_K_NORM,
                       OMNI_DEVICE_K_NORM, OMNI_DEVICE_KEY,
                       run->rows * OMNI_HEADS, OMNI_HEAD_DIM, 1.0e-5f, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half(
            run->backend, run->device[OMNI_DEVICE_QUERY], run->device[OMNI_DEVICE_COSINE],
            run->device[OMNI_DEVICE_SINE], run->rows, OMNI_HEADS, OMNI_HEAD_DIM,
            OMNI_ROTARY, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni query MM-RoPE accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half(
            run->backend, run->device[OMNI_DEVICE_KEY], run->device[OMNI_DEVICE_COSINE],
            run->device[OMNI_DEVICE_SINE], run->rows, OMNI_HEADS, OMNI_HEAD_DIM,
            OMNI_ROTARY, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni key MM-RoPE accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gqa(
            run->backend, run->device[OMNI_DEVICE_QUERY], run->device[OMNI_DEVICE_KEY],
            run->device[OMNI_DEVICE_VALUE], run->device[OMNI_DEVICE_ATTENTION],
            run->rows, OMNI_HEADS, OMNI_HEADS, OMNI_HEAD_DIM, 0, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni full-attention accounting overflowed");
    if (rc == YVEX_OK) rc = omni_round(run, OMNI_DEVICE_ATTENTION, attention_values, err);
    if (rc == YVEX_OK)
        rc = omni_weight_project(run, YVEX_MINIMAX_H3_OMNI_ATTENTION_OUT, run->rows,
                                 run->device[OMNI_DEVICE_ATTENTION],
                                 run->device[OMNI_DEVICE_UPDATE], err);
    if (rc == YVEX_OK) rc = omni_round(run, OMNI_DEVICE_UPDATE, run->values, err);
    return rc;
}

static int omni_mlp(omni_run *run, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    unsigned long long ffn_values = run->rows * OMNI_FFN;
    int rc = omni_weight_project(run, YVEX_MINIMAX_H3_OMNI_FC1, run->rows,
                                 run->device[OMNI_DEVICE_NORM],
                                 run->device[OMNI_DEVICE_FC1], err);
    if (rc == YVEX_OK) rc = omni_round(run, OMNI_DEVICE_FC1, 2ull * ffn_values, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_swiglu_split_bf16(
            run->backend, run->device[OMNI_DEVICE_FC1], run->device[OMNI_DEVICE_FF],
            run->rows, OMNI_FFN, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni SwiGLU accounting overflowed");
    if (rc == YVEX_OK)
        rc = omni_weight_project(run, YVEX_MINIMAX_H3_OMNI_FC2, run->rows,
                                 run->device[OMNI_DEVICE_FF],
                                 run->device[OMNI_DEVICE_UPDATE], err);
    if (rc == YVEX_OK) rc = omni_round(run, OMNI_DEVICE_UPDATE, run->values, err);
    return rc;
}

static int omni_block(omni_run *run, yvex_error *err)
{
    int rc = omni_modulation(run, err);
    if (rc == YVEX_OK)
        rc = omni_norm(run, OMNI_DEVICE_HIDDEN, YVEX_MINIMAX_H3_OMNI_NORM1,
                       OMNI_DEVICE_NORM_WEIGHT, OMNI_DEVICE_NORM,
                       run->rows, OMNI_HIDDEN, 1.0e-5f, err);
    if (rc == YVEX_OK) rc = omni_modulate(run, 0u, 1u, err);
    if (rc == YVEX_OK) rc = omni_attention(run, err);
    if (rc == YVEX_OK) rc = omni_residual(run, 2u, err);
    if (rc == YVEX_OK)
        rc = omni_norm(run, OMNI_DEVICE_HIDDEN, YVEX_MINIMAX_H3_OMNI_NORM2,
                       OMNI_DEVICE_NORM_WEIGHT, OMNI_DEVICE_NORM,
                       run->rows, OMNI_HIDDEN, 1.0e-5f, err);
    if (rc == YVEX_OK) rc = omni_modulate(run, 3u, 4u, err);
    if (rc == YVEX_OK) rc = omni_mlp(run, err);
    if (rc == YVEX_OK) rc = omni_residual(run, 5u, err);
    return rc;
}

static int omni_compute(omni_run *run, float *staged, yvex_error *err)
{
    int rc = omni_devices_prepare(run, err);
    if (rc == YVEX_OK) rc = omni_upload_inputs(run, err);
    if (rc == YVEX_OK) rc = omni_rope_tables(run, err);
    for (run->block_index = 0ull;
         rc == YVEX_OK && run->block_index < run->block_count; ++run->block_index)
        rc = omni_block(run, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(run->backend, run->device[OMNI_DEVICE_HIDDEN],
                                      staged, run->output_bytes, err);
    if (rc == YVEX_OK &&
        !yvex_core_u64_add(run->facts.d2h_bytes, run->output_bytes, &run->facts.d2h_bytes))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni output accounting overflowed");
    return rc;
}

static int omni_devices_release(omni_run *run, int rc, yvex_error *err)
{
    int slot;
    for (slot = OMNI_DEVICE_COUNT - 1; slot >= 0; --slot) {
        yvex_error cleanup;
        int cleanup_rc;
        if (!run->device[slot]) continue;
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(run->backend, &run->device[slot], &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    return rc;
}

static int omni_identity(const omni_run *run, const char *residency_identity,
                         const float *output, char identity[TEXT_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, temb_values = run->timesteps * OMNI_TIME;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.omni-block-stack.cuda.v1") ||
        !yvex_sha256_update_text(&hash, "minimax-h3-fl2va-omni-transformer") ||
        !yvex_sha256_update_text(&hash, residency_identity) ||
        !yvex_sha256_update_u64(&hash, run->block_count) ||
        !yvex_sha256_update_u64(&hash, run->rows) ||
        !yvex_sha256_update_u64(&hash, run->timesteps)) return 0;
#define HASH_FLOATS(values, count) \
    for (index = 0ull; index < (count); ++index) { \
        uint32_t bits; \
        memcpy(&bits, (values) + index, sizeof(bits)); \
        if (!yvex_sha256_update_u64(&hash, bits)) return 0; \
    }
    HASH_FLOATS(run->positions, run->rows * 3ull)
    HASH_FLOATS(run->temb, temb_values)
    HASH_FLOATS(output, run->values)
#undef HASH_FLOATS
    for (index = 0ull; index < run->rows; ++index)
        if (!yvex_sha256_update_u64(&hash, run->adaln_indices[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, identity);
    return 1;
}

static int omni_blocks_cuda(
    yvex_backend *backend, const yvex_minimax_h3_encoded_weight *weights,
    unsigned long long block_count, const char *residency_identity,
    unsigned long long resident_bytes, const float *hidden, const float *temb,
    unsigned long long timestep_count, const float *position_ids,
    const unsigned int *adaln_indices, unsigned long long packed_rows,
    float *output, unsigned long long output_capacity,
    yvex_minimax_h3_omni_result *result, yvex_error *err)
{
    omni_run run = {0};
    yvex_minimax_h3_omni_result published = {0};
    float *staged = NULL;
    unsigned long long weight_bytes = 0ull;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    run.backend = backend;
    run.weights = weights;
    run.hidden = hidden;
    run.temb = temb;
    run.positions = position_ids;
    run.adaln_indices = adaln_indices;
    run.rows = packed_rows;
    run.timesteps = timestep_count;
    run.block_count = block_count;
    rc = omni_validate(&run, residency_identity, resident_bytes, output,
                       output_capacity, result, &weight_bytes, err);
    if (rc == YVEX_OK) {
        staged = (float *)malloc((size_t)run.output_bytes);
        if (!staged)
            rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.minimax-h3.omni.output",
                                     "transactional Omni output allocation failed");
    }
    if (rc == YVEX_OK) rc = omni_compute(&run, staged, err);
    if (rc == YVEX_OK && !omni_identity(&run, residency_identity, staged,
                                        published.execution_identity))
        rc = conditioning_refuse(err, YVEX_ERR_STATE, "cuda.minimax-h3.omni.identity",
                                 "Omni execution identity could not be sealed");
    rc = omni_devices_release(&run, rc, err);
    if (rc == YVEX_OK) {
        memcpy(output, staged, (size_t)run.output_bytes);
        published.packed_rows = packed_rows;
        published.block_count = block_count;
        published.resident_bytes = resident_bytes;
        published.kernel_launches = run.facts.kernel_launches;
        published.h2d_bytes = run.facts.h2d_bytes;
        published.d2h_bytes = run.facts.d2h_bytes;
        published.device_bytes = run.device_bytes;
        memcpy(published.residency_identity, residency_identity,
               sizeof(published.residency_identity));
        published.complete = 1;
        *result = published;
        yvex_error_clear(err);
    }
    free(staged);
    return rc;
}

const yvex_minimax_h3_backend_api *yvex_backend_register_minimax_h3(void)
{
    static const yvex_minimax_h3_backend_api api = {
        text_embed_cuda, text_layer_cuda, omni_blocks_cuda,
    };
    return &api;
}
