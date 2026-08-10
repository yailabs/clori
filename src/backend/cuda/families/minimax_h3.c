/* Execute admitted MiniMax-H3 operations through generic CUDA primitives. */
#include "src/backend/cuda/private.h"
#include <yvex/backend.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/component.h>
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
    OMNI_DEVICE_TEMB, OMNI_DEVICE_TEMB_ACTIVATED,
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
    const float *hidden, *temb, *positions, *inv_freq;
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
    OMNI_ALLOC(OMNI_DEVICE_TEMB_ACTIVATED, "omni-temb-activated", run->timesteps, OMNI_TIME, 0);
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
                float frequency = run->inv_freq ? run->inv_freq[frequency_index]
                                                : powf(10000.0f, -(float)(2ull * frequency_index) / 32.0f);
                float angle = run->positions[row * 3ull + axis] * frequency;
                unsigned long long coordinate = axis * 16ull + frequency_index;
                for (half_index = 0ull; half_index < 2ull; ++half_index) {
                    unsigned long long index =
                        row * OMNI_ROTARY + half_index * (OMNI_ROTARY / 2ull) + coordinate;
                    cosines[index] = text_bf16_value(cosf(angle));
                    sines[index] = text_bf16_value(sinf(angle));
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
        run->backend, run->device[OMNI_DEVICE_TEMB], run->device[OMNI_DEVICE_TEMB_ACTIVATED],
        temb_values, 1, &facts, err);
    if (rc == YVEX_OK && !omni_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.omni.facts",
                                 "Omni timestep activation accounting overflowed");
    if (rc == YVEX_OK)
        rc = omni_weight_project(run, YVEX_MINIMAX_H3_OMNI_ADALN_WEIGHT, run->timesteps,
                                 run->device[OMNI_DEVICE_TEMB_ACTIVATED],
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

static int omni_blocks_execute(
    yvex_backend *backend, const yvex_minimax_h3_encoded_weight *weights,
    unsigned long long block_count, const char *residency_identity,
    unsigned long long resident_bytes, const float *hidden, const float *temb,
    unsigned long long timestep_count, const float *position_ids,
    const unsigned int *adaln_indices, unsigned long long packed_rows,
    float *output, unsigned long long output_capacity,
    yvex_minimax_h3_omni_result *result, const float *inv_freq, yvex_error *err)
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
    run.inv_freq = inv_freq;
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
static int omni_blocks_cuda(yvex_backend *backend, const yvex_minimax_h3_encoded_weight *weights,
    unsigned long long block_count, const char *residency_identity, unsigned long long resident_bytes,
    const float *hidden, const float *temb, unsigned long long timestep_count, const float *position_ids,
    const unsigned int *adaln_indices, unsigned long long packed_rows, float *output,
    unsigned long long output_capacity, yvex_minimax_h3_omni_result *result, yvex_error *err)
{
    return omni_blocks_execute(backend, weights, block_count, residency_identity, resident_bytes,
        hidden, temb, timestep_count, position_ids, adaln_indices, packed_rows, output,
        output_capacity, result, NULL, err);
}

typedef enum {
    REFINER_HIDDEN = 0, REFINER_NORM, REFINER_NORM_WEIGHT, REFINER_QKV,
    REFINER_QUERY, REFINER_KEY, REFINER_VALUE, REFINER_Q_NORM,
    REFINER_K_NORM, REFINER_ATTENTION, REFINER_UPDATE, REFINER_FC1,
    REFINER_FF, REFINER_DEVICE_COUNT
} refiner_device_slot;

typedef struct {
    yvex_backend *backend;
    const yvex_minimax_h3_encoded_weight *weights;
    yvex_device_tensor *device[REFINER_DEVICE_COUNT];
    unsigned long long rows, values, output_bytes, device_bytes;
    yvex_minimax_h3_omni_transformer_result *facts;
} refiner_run;

static int transformer_facts_add(yvex_minimax_h3_omni_transformer_result *total,
                                 const yvex_backend_cuda_operation_facts *part)
{
    return total && part && part->compulsory_memory_facts_available &&
           yvex_core_u64_add(total->kernel_launches, part->kernel_launches,
                             &total->kernel_launches) &&
           yvex_core_u64_add(total->h2d_bytes, part->h2d_bytes, &total->h2d_bytes) &&
           yvex_core_u64_add(total->d2h_bytes, part->d2h_bytes, &total->d2h_bytes);
}
static int transformer_fact_bytes(unsigned long long *total, unsigned long long bytes,
                                  yvex_error *err)
{
    if (yvex_core_u64_add(*total, bytes, total)) return YVEX_OK;
    return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.transformer.facts",
                               "transformer transfer accounting overflowed");
}
static int transformer_weight_valid(const yvex_minimax_h3_encoded_weight *weight,
                                    unsigned int qtype, unsigned long long rows,
                                    unsigned long long width)
{
    unsigned long long row_bytes, bytes;
    return weight && weight->encoded && weight->qtype == qtype &&
           weight->row_count == rows && weight->row_width == width &&
           yvex_core_u64_mul(width, qtype == YVEX_GGUF_QTYPE_F32 ? 4ull : 2ull,
                             &row_bytes) &&
           yvex_core_u64_mul(rows, row_bytes, &bytes) && weight->row_bytes == row_bytes &&
           weight->encoded_bytes == bytes;
}

static int transformer_external_valid(const yvex_minimax_h3_encoded_weight *weights,
                                      unsigned long long *bytes)
{
    static const unsigned int qtypes[YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT] = {
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
    };
    static const unsigned long long rows[YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT] = {
        5376u, 1u, 5376u, 1u, 5376u, 1u, 5376u, 1u, 2688u, 1u,
        1u, 21504u, 1u, 1u, 5376u, 1u, 28672u, 5376u,
        1u, 21504u, 1u, 1u, 5376u, 1u, 28672u, 5376u, 1u,
        1u, 1u, 10752u, 1u, 96u, 1u, 32u, 1u,
    };
    static const unsigned long long widths[YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT] = {
        32u, 5376u, 96u, 5376u, 5120u, 5376u, 256u, 5376u, 5376u, 2688u,
        5376u, 5376u, 128u, 128u, 7168u, 5376u, 5376u, 14336u,
        5376u, 5376u, 128u, 128u, 7168u, 5376u, 5376u, 14336u, 5376u,
        16u, 5376u, 2688u, 10752u, 5376u, 96u, 5376u, 32u,
    };
    unsigned long long index, next;
    if (bytes) *bytes = 0ull;
    if (!weights || !bytes) return 0;
    for (index = 0ull; index < YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT; ++index) {
        if (!transformer_weight_valid(weights + index, qtypes[index], rows[index], widths[index]) ||
            !yvex_core_u64_add(*bytes, weights[index].encoded_bytes, &next)) return 0;
        *bytes = next;
    }
    return 1;
}

static int transformer_tensor(yvex_backend *backend, const char *name,
                              unsigned long long rows, unsigned long long width,
                              yvex_device_tensor **out, unsigned long long *bytes,
                              yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long elements, allocation, next;
    if (!backend || !name || !rows || !width || !out || !bytes ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &allocation) ||
        !yvex_core_u64_add(*bytes, allocation, &next))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.transformer.allocate",
                                   "transformer activation geometry overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = rows == 1ull ? 1u : 2u;
    descriptor.dims[0] = rows == 1ull ? width : rows;
    descriptor.dims[1] = rows == 1ull ? 0ull : width;
    descriptor.bytes = allocation;
    if (yvex_backend_tensor_alloc(backend, &descriptor, out, err) != YVEX_OK)
        return yvex_error_code(err);
    *bytes = next;
    return YVEX_OK;
}

static int transformer_devices_release(yvex_backend *backend, yvex_device_tensor **device,
                                       unsigned int count, int rc, yvex_error *err)
{
    while (count) {
        yvex_error cleanup;
        int cleanup_rc;
        --count;
        if (!device[count]) continue;
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(backend, &device[count], &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    return rc;
}

static int transformer_gather(yvex_backend *backend,
                              const yvex_minimax_h3_encoded_weight *weight,
                              yvex_device_tensor *output,
                              yvex_minimax_h3_omni_transformer_result *facts,
                              yvex_error *err)
{
    yvex_backend_cuda_operation_facts part;
    int rc = yvex_backend_cuda_encoded_gather(
        backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes,
        &text_zero_row, 1ull, output, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.transformer.facts",
                                 "transformer gather accounting overflowed");
    return rc;
}

static int transformer_project(yvex_backend *backend,
                               const yvex_minimax_h3_encoded_weight *weight,
                               unsigned long long rows, const yvex_device_tensor *input,
                               const yvex_device_tensor *additive, yvex_device_tensor *output,
                               yvex_minimax_h3_omni_transformer_result *facts,
                               yvex_error *err)
{
    yvex_backend_cuda_operation_facts part;
    int rc = yvex_backend_cuda_encoded_matvec(
        backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes, rows,
        input, NULL, 0ull, additive, output, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.transformer.facts",
                                 "transformer projection accounting overflowed");
    return rc;
}

static int transformer_round(yvex_backend *backend, yvex_device_tensor *tensor,
                             unsigned long long values,
                             yvex_minimax_h3_omni_transformer_result *facts,
                             yvex_error *err)
{
    yvex_backend_cuda_operation_facts part;
    int rc = yvex_cuda_transformer_bf16_round(backend, tensor, values, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.transformer.facts",
                                 "transformer rounding accounting overflowed");
    return rc;
}

static int transformer_linear_host(
    yvex_backend *backend, const yvex_minimax_h3_encoded_weight *weight,
    const yvex_minimax_h3_encoded_weight *bias, const float *input,
    unsigned long long rows, float *output, int bf16_output,
    yvex_minimax_h3_omni_transformer_result *facts, yvex_error *err)
{
    enum { INPUT = 0, OUTPUT, BIAS, COUNT };
    yvex_device_tensor *device[COUNT] = {0};
    yvex_backend_cuda_operation_facts part;
    unsigned long long device_bytes = 0ull, input_values, input_bytes, output_values, output_bytes;
    int rc;
    if (!yvex_core_u64_mul(rows, weight->row_width, &input_values) ||
        !yvex_core_u64_mul(input_values, sizeof(float), &input_bytes) ||
        !yvex_core_u64_mul(rows, weight->row_count, &output_values) ||
        !yvex_core_u64_mul(output_values, sizeof(float), &output_bytes))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.transformer.linear",
                                   "transformer projection byte geometry overflowed");
    rc = transformer_tensor(backend, "transformer-linear-input", rows, weight->row_width,
                            &device[INPUT], &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "transformer-linear-output", rows, weight->row_count,
                                &device[OUTPUT], &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "transformer-linear-bias", 1ull, weight->row_count,
                                &device[BIAS], &device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(backend, device[INPUT], input, input_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, input_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_project(backend, weight, rows, device[INPUT], NULL,
                                 device[OUTPUT], facts, err);
    if (rc == YVEX_OK) rc = transformer_gather(backend, bias, device[BIAS], facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(backend, device[OUTPUT], device[BIAS], device[OUTPUT],
                                        rows, weight->row_count, bf16_output, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.transformer.facts",
                                 "transformer bias accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(backend, device[OUTPUT], output, output_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->d2h_bytes, output_bytes, err);
    if (device_bytes > facts->device_bytes) facts->device_bytes = device_bytes;
    return transformer_devices_release(backend, device, COUNT, rc, err);
}

static const yvex_minimax_h3_encoded_weight *refiner_weight(
    const refiner_run *run, unsigned long long block, unsigned long long slot)
{
    return run->weights + block * 8ull + slot;
}

static int refiner_norm(refiner_run *run, refiner_device_slot input,
                        const yvex_minimax_h3_encoded_weight *weight,
                        refiner_device_slot weight_device, refiner_device_slot output,
                        unsigned long long rows, unsigned long long width, yvex_error *err)
{
    yvex_backend_cuda_operation_facts part;
    int rc = transformer_gather(run->backend, weight, run->device[weight_device], run->facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rms_norm_bf16(
            run->backend, run->device[input], run->device[weight_device], run->device[output],
            rows, width, 1.0e-5f, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(run->facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.refiner.facts",
                                 "token-refiner norm accounting overflowed");
    return rc;
}

static int refiner_devices_prepare(refiner_run *run, yvex_error *err)
{
    int rc;
#define REFINE_ALLOC(slot, name, rows, width) \
    if (rc == YVEX_OK) rc = transformer_tensor(run->backend, name, rows, width, \
                                                &run->device[slot], &run->device_bytes, err)
    rc = transformer_tensor(run->backend, "refiner-hidden", run->rows, OMNI_HIDDEN,
                            &run->device[REFINER_HIDDEN], &run->device_bytes, err);
    REFINE_ALLOC(REFINER_NORM, "refiner-norm", run->rows, OMNI_HIDDEN);
    REFINE_ALLOC(REFINER_NORM_WEIGHT, "refiner-norm-weight", 1ull, OMNI_HIDDEN);
    REFINE_ALLOC(REFINER_QKV, "refiner-qkv", run->rows, 3ull * OMNI_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_QUERY, "refiner-query", run->rows, OMNI_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_KEY, "refiner-key", run->rows, OMNI_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_VALUE, "refiner-value", run->rows, OMNI_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_Q_NORM, "refiner-q-norm", 1ull, OMNI_HEAD_DIM);
    REFINE_ALLOC(REFINER_K_NORM, "refiner-k-norm", 1ull, OMNI_HEAD_DIM);
    REFINE_ALLOC(REFINER_ATTENTION, "refiner-attention", run->rows, OMNI_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_UPDATE, "refiner-update", run->rows, OMNI_HIDDEN);
    REFINE_ALLOC(REFINER_FC1, "refiner-fc1", run->rows, 2ull * OMNI_FFN);
    REFINE_ALLOC(REFINER_FF, "refiner-ff", run->rows, OMNI_FFN);
#undef REFINE_ALLOC
    return rc;
}

static int refiner_block(refiner_run *run, unsigned long long block, yvex_error *err)
{
    const yvex_minimax_h3_encoded_weight *weight = refiner_weight(run, block, 0ull);
    yvex_backend_cuda_operation_facts part;
    unsigned long long attention_values = run->rows * OMNI_ATTENTION_WIDTH;
    unsigned long long ffn_values = run->rows * OMNI_FFN;
    int rc = refiner_norm(run, REFINER_HIDDEN, weight, REFINER_NORM_WEIGHT,
                          REFINER_NORM, run->rows, OMNI_HIDDEN, err);
    if (rc == YVEX_OK)
        rc = transformer_project(run->backend, weight + 1, run->rows,
                                 run->device[REFINER_NORM], NULL,
                                 run->device[REFINER_QKV], run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_QKV],
                               3ull * attention_values, run->facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_split_three(
            run->backend, run->device[REFINER_QKV], run->device[REFINER_QUERY],
            run->device[REFINER_KEY], run->device[REFINER_VALUE], run->rows,
            OMNI_ATTENTION_WIDTH, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(run->facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = refiner_norm(run, REFINER_QUERY, weight + 2, REFINER_Q_NORM,
                          REFINER_QUERY, run->rows * OMNI_HEADS, OMNI_HEAD_DIM, err);
    if (rc == YVEX_OK)
        rc = refiner_norm(run, REFINER_KEY, weight + 3, REFINER_K_NORM,
                          REFINER_KEY, run->rows * OMNI_HEADS, OMNI_HEAD_DIM, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gqa(
            run->backend, run->device[REFINER_QUERY], run->device[REFINER_KEY],
            run->device[REFINER_VALUE], run->device[REFINER_ATTENTION], run->rows,
            OMNI_HEADS, OMNI_HEADS, OMNI_HEAD_DIM, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(run->facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_ATTENTION],
                               attention_values, run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_project(run->backend, weight + 4, run->rows,
                                 run->device[REFINER_ATTENTION], run->device[REFINER_HIDDEN],
                                 run->device[REFINER_UPDATE], run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_UPDATE], run->values,
                               run->facts, err);
    if (rc == YVEX_OK)
        rc = refiner_norm(run, REFINER_UPDATE, weight + 5, REFINER_NORM_WEIGHT,
                          REFINER_NORM, run->rows, OMNI_HIDDEN, err);
    if (rc == YVEX_OK)
        rc = transformer_project(run->backend, weight + 6, run->rows,
                                 run->device[REFINER_NORM], NULL,
                                 run->device[REFINER_FC1], run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_FC1],
                               2ull * ffn_values, run->facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_swiglu_split_bf16(
            run->backend, run->device[REFINER_FC1], run->device[REFINER_FF],
            run->rows, OMNI_FFN, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(run->facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = transformer_project(run->backend, weight + 7, run->rows,
                                 run->device[REFINER_FF], run->device[REFINER_UPDATE],
                                 run->device[REFINER_HIDDEN], run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_HIDDEN], run->values,
                               run->facts, err);
    if (rc == YVEX_ERR_BOUNDS && yvex_error_code(err) == YVEX_OK)
        rc = conditioning_refuse(err, rc, "cuda.minimax-h3.refiner.facts",
                                 "token-refiner accounting overflowed");
    return rc;
}

static int transformer_refine(yvex_backend *backend,
                              const yvex_minimax_h3_encoded_weight *weights,
                              const float *input, unsigned long long rows, float *output,
                              yvex_minimax_h3_omni_transformer_result *facts,
                              yvex_error *err)
{
    refiner_run run = {0};
    unsigned long long block, input_bytes;
    int rc;
    run.backend = backend;
    run.weights = weights;
    run.rows = rows;
    run.facts = facts;
    if (!yvex_core_u64_mul(rows, OMNI_HIDDEN, &run.values) ||
        !yvex_core_u64_mul(run.values, sizeof(float), &run.output_bytes))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.refiner",
                                   "token-refiner geometry overflowed");
    input_bytes = run.output_bytes;
    rc = refiner_devices_prepare(&run, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(backend, run.device[REFINER_HIDDEN], input,
                                       input_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, input_bytes, err);
    for (block = 0ull; rc == YVEX_OK && block < 2ull; ++block)
        rc = refiner_block(&run, block, err);
    if (rc == YVEX_OK)
        rc = refiner_norm(&run, REFINER_HIDDEN, weights + 16,
                          REFINER_NORM_WEIGHT, REFINER_NORM,
                          rows, OMNI_HIDDEN, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(backend, run.device[REFINER_NORM], output,
                                      run.output_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->d2h_bytes, run.output_bytes, err);
    if (run.device_bytes > facts->device_bytes) facts->device_bytes = run.device_bytes;
    return transformer_devices_release(backend, run.device, REFINER_DEVICE_COUNT, rc, err);
}

static int transformer_time_embed(
    yvex_backend *backend, const yvex_minimax_h3_encoded_weight *weights,
    const float *timesteps, unsigned long long rows, float *output,
    yvex_minimax_h3_omni_transformer_result *facts, yvex_error *err)
{
    enum { INPUT = 0, HIDDEN, OUTPUT, BIAS_IN, BIAS_OUT, COUNT };
    yvex_device_tensor *device[COUNT] = {0};
    yvex_backend_cuda_operation_facts part;
    float *embedding = NULL;
    unsigned long long row, lane, input_values, input_bytes, output_bytes;
    unsigned long long device_bytes = 0ull;
    int rc = YVEX_OK;
    if (!yvex_core_u64_mul(rows, 256ull, &input_values) ||
        !yvex_core_u64_mul(input_values, sizeof(float), &input_bytes) ||
        !yvex_core_u64_mul(rows * OMNI_TIME, sizeof(float), &output_bytes) ||
        input_bytes > SIZE_MAX || !(embedding = (float *)malloc((size_t)input_bytes)))
        return conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.minimax-h3.time-embed",
                                   "bounded timestep embedding allocation failed");
    for (row = 0ull; row < rows; ++row)
        for (lane = 0ull; lane < 128ull; ++lane) {
            double angle = timesteps[row] * exp(-log(10000.0) * (double)lane / 128.0);
            embedding[row * 256ull + lane] = (float)cos(angle);
            embedding[row * 256ull + 128ull + lane] = (float)sin(angle);
        }
    rc = transformer_tensor(backend, "time-input", rows, 256ull, &device[INPUT],
                            &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "time-hidden", rows, 5376ull, &device[HIDDEN],
                                &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "time-output", rows, OMNI_TIME, &device[OUTPUT],
                                &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "time-bias-in", 1ull, 5376ull, &device[BIAS_IN],
                                &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "time-bias-out", 1ull, OMNI_TIME,
                                &device[BIAS_OUT], &device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(backend, device[INPUT], embedding, input_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, input_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_project(backend, weights + YVEX_MINIMAX_H3_OMNI_TIME_IN_WEIGHT,
                                 rows, device[INPUT], NULL, device[HIDDEN], facts, err);
    if (rc == YVEX_OK)
        rc = transformer_gather(backend, weights + YVEX_MINIMAX_H3_OMNI_TIME_IN_BIAS,
                                device[BIAS_IN], facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(backend, device[HIDDEN], device[BIAS_IN], device[HIDDEN],
                                        rows, 5376ull, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_silu(backend, device[HIDDEN], device[HIDDEN], rows * 5376ull,
                                        0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = transformer_project(backend, weights + YVEX_MINIMAX_H3_OMNI_TIME_OUT_WEIGHT,
                                 rows, device[HIDDEN], NULL, device[OUTPUT], facts, err);
    if (rc == YVEX_OK)
        rc = transformer_gather(backend, weights + YVEX_MINIMAX_H3_OMNI_TIME_OUT_BIAS,
                                device[BIAS_OUT], facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(backend, device[OUTPUT], device[BIAS_OUT], device[OUTPUT],
                                        rows, OMNI_TIME, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(backend, device[OUTPUT], output, output_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->d2h_bytes, output_bytes, err);
    if (rc == YVEX_ERR_BOUNDS && yvex_error_code(err) == YVEX_OK)
        rc = conditioning_refuse(err, rc, "cuda.minimax-h3.time-embed.facts",
                                 "timestep embedding accounting overflowed");
    if (device_bytes > facts->device_bytes) facts->device_bytes = device_bytes;
    free(embedding);
    return transformer_devices_release(backend, device, COUNT, rc, err);
}

static int transformer_final_norm(
    yvex_backend *backend, const yvex_minimax_h3_encoded_weight *weights,
    const float *hidden, const float *temb, const unsigned int *timestep_indices,
    unsigned long long rows, unsigned long long timesteps, float *output,
    yvex_minimax_h3_omni_transformer_result *facts, yvex_error *err)
{
    enum { HIDDEN = 0, NORM, NORM_WEIGHT, TEMB, TABLE, BIAS, COUNT };
    yvex_device_tensor *device[COUNT] = {0};
    yvex_backend_cuda_operation_facts part;
    unsigned long long device_bytes = 0ull, hidden_bytes = rows * OMNI_HIDDEN * 4ull;
    unsigned long long temb_bytes = timesteps * OMNI_TIME * 4ull;
    int rc = transformer_tensor(backend, "final-hidden", rows, OMNI_HIDDEN,
                                &device[HIDDEN], &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-norm", rows, OMNI_HIDDEN,
                                                &device[NORM], &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-norm-weight", 1ull, OMNI_HIDDEN,
                                                &device[NORM_WEIGHT], &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-temb", timesteps, OMNI_TIME,
                                                &device[TEMB], &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-table", timesteps,
                                                2ull * OMNI_HIDDEN, &device[TABLE],
                                                &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-bias", 1ull,
                                                2ull * OMNI_HIDDEN, &device[BIAS],
                                                &device_bytes, err);
    if (rc == YVEX_OK) rc = yvex_backend_tensor_write(backend, device[HIDDEN], hidden,
                                                       hidden_bytes, err);
    if (rc == YVEX_OK) rc = yvex_backend_tensor_write(backend, device[TEMB], temb,
                                                       temb_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, hidden_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, temb_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_gather(backend, weights + YVEX_MINIMAX_H3_OMNI_FINAL_NORM,
                                device[NORM_WEIGHT], facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rms_norm_bf16(backend, device[HIDDEN], device[NORM_WEIGHT],
                                                  device[NORM], rows, OMNI_HIDDEN, 1.0e-5f,
                                                  &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_silu(backend, device[TEMB], device[TEMB],
                                        timesteps * OMNI_TIME, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = transformer_project(backend, weights + YVEX_MINIMAX_H3_OMNI_FINAL_ADALN_WEIGHT,
                                 timesteps, device[TEMB], NULL, device[TABLE], facts, err);
    if (rc == YVEX_OK) rc = transformer_round(backend, device[TABLE],
                                               timesteps * 2ull * OMNI_HIDDEN, facts, err);
    if (rc == YVEX_OK)
        rc = transformer_gather(backend, weights + YVEX_MINIMAX_H3_OMNI_FINAL_ADALN_BIAS,
                                device[BIAS], facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(backend, device[TABLE], device[BIAS], device[TABLE],
                                        timesteps, 2ull * OMNI_HIDDEN, 1, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_modulate_bf16(
            backend, device[NORM], device[TABLE], timestep_indices, device[NORM], rows,
            OMNI_HIDDEN, timesteps, 2ull, 0u, 1u, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(backend, device[NORM], output, hidden_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->d2h_bytes, hidden_bytes, err);
    if (rc == YVEX_ERR_BOUNDS && yvex_error_code(err) == YVEX_OK)
        rc = conditioning_refuse(err, rc, "cuda.minimax-h3.final.facts",
                                 "final normalization accounting overflowed");
    if (device_bytes > facts->device_bytes) facts->device_bytes = device_bytes;
    return transformer_devices_release(backend, device, COUNT, rc, err);
}

static int transformer_request_valid(
    const yvex_minimax_h3_omni_transformer_request *request,
    unsigned long long *video_values, unsigned long long *audio_values,
    yvex_error *err)
{
    unsigned char seen[OMNI_MAX_ROWS] = {0};
    unsigned long long kind, row, total;
    const unsigned int *indices[3];
    unsigned long long counts[3];
    if (!request || !request->video || !request->audio || !request->conditioning ||
        !request->timesteps || !request->position_ids || !request->video_indices ||
        !request->audio_indices || !request->text_indices || !request->timestep_indices ||
        !request->token_tags || !request->video_output || !request->audio_output ||
        !request->video_rows || !request->audio_rows || !request->text_rows ||
        !request->timestep_count || request->timestep_count > OMNI_MAX_TIMESTEPS ||
        !request->packed_rows || request->packed_rows > OMNI_MAX_ROWS ||
        !request->block_count || request->block_count > OMNI_BLOCKS ||
        !yvex_core_u64_add(request->video_rows, request->audio_rows, &total) ||
        !yvex_core_u64_add(total, request->text_rows, &total) || total != request->packed_rows ||
        !yvex_core_u64_mul(request->video_rows, 96ull, video_values) ||
        !yvex_core_u64_mul(request->audio_rows, 32ull, audio_values) ||
        *video_values > request->video_output_capacity ||
        *audio_values > request->audio_output_capacity)
        return conditioning_refuse(err, YVEX_ERR_INVALID_ARG, "cuda.minimax-h3.transformer.request",
                                   "one complete bounded packed FL2VA request is required");
    indices[0] = request->video_indices; counts[0] = request->video_rows;
    indices[1] = request->text_indices; counts[1] = request->text_rows;
    indices[2] = request->audio_indices; counts[2] = request->audio_rows;
    for (kind = 0ull; kind < 3ull; ++kind)
        for (row = 0ull; row < counts[kind]; ++row) {
            unsigned int packed = indices[kind][row];
            if (packed >= request->packed_rows || seen[packed] || request->token_tags[packed] != kind)
                return conditioning_refuse(err, YVEX_ERR_FORMAT,
                                           "cuda.minimax-h3.transformer.layout",
                                           "packed modality indices must form one exact tagged partition");
            seen[packed] = 1u;
        }
    for (row = 0ull; row < request->packed_rows; ++row)
        if (!seen[row] || request->timestep_indices[row] >= request->timestep_count ||
            !isfinite(request->position_ids[row * 3ull]) ||
            !isfinite(request->position_ids[row * 3ull + 1ull]) ||
            !isfinite(request->position_ids[row * 3ull + 2ull]))
            return conditioning_refuse(err, YVEX_ERR_FORMAT, "cuda.minimax-h3.transformer.layout",
                                       "packed rows require finite positions and admitted timesteps");
    for (row = 0ull; row < request->timestep_count; ++row)
        if (!isfinite(request->timesteps[row]) || request->timesteps[row] < 0.0f ||
            request->timesteps[row] > 1.0f)
            return conditioning_refuse(err, YVEX_ERR_FORMAT, "cuda.minimax-h3.transformer.timestep",
                                       "distinct timesteps must be finite values in [0,1]");
    return YVEX_OK;
}

static int transformer_hash_floats(yvex_sha256 *hash, const float *values,
                                   unsigned long long count)
{
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        memcpy(&bits, values + index, sizeof(bits));
        if (!yvex_sha256_update_u64(hash, bits)) return 0;
    }
    return 1;
}

static int transformer_execution_identity(
    const yvex_minimax_h3_omni_transformer_request *request,
    const char *residency_identity, const char *block_identity,
    const float *video, unsigned long long video_values,
    const float *audio, unsigned long long audio_values, char output[65])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.omni-transformer.cuda.v1") ||
        !yvex_sha256_update_text(&hash, residency_identity) ||
        !yvex_sha256_update_text(&hash, block_identity) ||
        !yvex_sha256_update_u64(&hash, request->video_rows) ||
        !yvex_sha256_update_u64(&hash, request->audio_rows) ||
        !yvex_sha256_update_u64(&hash, request->text_rows) ||
        !yvex_sha256_update_u64(&hash, request->timestep_count) ||
        !yvex_sha256_update_u64(&hash, request->packed_rows) ||
        !yvex_sha256_update_u64(&hash, request->block_count) ||
        !transformer_hash_floats(&hash, request->video, request->video_rows * 96ull) ||
        !transformer_hash_floats(&hash, request->audio, request->audio_rows * 32ull) ||
        !transformer_hash_floats(&hash, request->conditioning,
                                 request->text_rows * 5120ull) ||
        !transformer_hash_floats(&hash, request->timesteps, request->timestep_count) ||
        !transformer_hash_floats(&hash, request->position_ids,
                                 request->packed_rows * 3ull)) return 0;
    for (index = 0ull; index < request->video_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->video_indices[index])) return 0;
    for (index = 0ull; index < request->audio_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->audio_indices[index])) return 0;
    for (index = 0ull; index < request->text_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->text_indices[index])) return 0;
    for (index = 0ull; index < request->packed_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->timestep_indices[index]) ||
            !yvex_sha256_update_u64(&hash, request->token_tags[index])) return 0;
    if (!transformer_hash_floats(&hash, video, video_values) ||
        !transformer_hash_floats(&hash, audio, audio_values) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int omni_transformer_cuda(
    yvex_backend *backend, const yvex_minimax_h3_encoded_weight *external_weights,
    const yvex_minimax_h3_encoded_weight *block_weights, const char *residency_identity,
    unsigned long long resident_bytes, const yvex_minimax_h3_omni_transformer_request *request,
    yvex_minimax_h3_omni_transformer_result *result, yvex_error *err)
{
    yvex_minimax_h3_omni_transformer_result published = {0};
    yvex_minimax_h3_omni_result blocks = {0};
    float *video_embed = NULL, *audio_embed = NULL, *text_embed = NULL, *text_refined = NULL;
    float *packed = NULL, *block_output = NULL, *temb = NULL, *normalized = NULL;
    float *all_video = NULL, *all_audio = NULL, *staged_video = NULL, *staged_audio = NULL;
    unsigned int *adaln = NULL;
    unsigned long long external_bytes = 0ull, block_bytes = 0ull, required_bytes;
    unsigned long long video_values = 0ull, audio_values = 0ull, hidden_values, row, lane;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    rc = transformer_request_valid(request, &video_values, &audio_values, err);
    if (rc == YVEX_OK &&
        (!backend || !result || !yvex_sha256_hex_valid(residency_identity) ||
         !transformer_external_valid(external_weights, &external_bytes) ||
         !omni_weights_validate(block_weights, request->block_count, &block_bytes) ||
         !yvex_core_u64_add(external_bytes, block_bytes, &required_bytes) ||
         resident_bytes < required_bytes ||
         !yvex_core_u64_mul(request->packed_rows, OMNI_HIDDEN, &hidden_values) ||
         hidden_values > SIZE_MAX / sizeof(float)))
        rc = conditioning_refuse(err, YVEX_ERR_INVALID_ARG, "cuda.minimax-h3.transformer",
                                 "exact resident Transformer weights and output state are required");
#define HOST_FLOATS(target, count) \
    if (rc == YVEX_OK && (!(target = (float *)malloc((size_t)(count) * sizeof(float))))) \
        rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.minimax-h3.transformer.host", \
                                 "transactional Transformer host allocation failed")
    HOST_FLOATS(video_embed, request ? request->video_rows * OMNI_HIDDEN : 0ull);
    HOST_FLOATS(audio_embed, request ? request->audio_rows * OMNI_HIDDEN : 0ull);
    HOST_FLOATS(text_embed, request ? request->text_rows * OMNI_HIDDEN : 0ull);
    HOST_FLOATS(text_refined, request ? request->text_rows * OMNI_HIDDEN : 0ull);
    HOST_FLOATS(packed, hidden_values); HOST_FLOATS(block_output, hidden_values);
    HOST_FLOATS(temb, request ? request->timestep_count * OMNI_TIME : 0ull);
    HOST_FLOATS(normalized, hidden_values);
    HOST_FLOATS(all_video, request ? request->packed_rows * 96ull : 0ull);
    HOST_FLOATS(all_audio, request ? request->packed_rows * 32ull : 0ull);
    HOST_FLOATS(staged_video, video_values); HOST_FLOATS(staged_audio, audio_values);
#undef HOST_FLOATS
    if (rc == YVEX_OK && !(adaln = (unsigned int *)malloc(
                               (size_t)request->packed_rows * sizeof(*adaln))))
        rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.minimax-h3.transformer.host",
                                 "packed AdaLN selection allocation failed");
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_MINIMAX_H3_OMNI_VIDEO_WEIGHT,
                                     external_weights + YVEX_MINIMAX_H3_OMNI_VIDEO_BIAS,
                                     request->video, request->video_rows, video_embed, 0,
                                     &published, err);
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_MINIMAX_H3_OMNI_AUDIO_WEIGHT,
                                     external_weights + YVEX_MINIMAX_H3_OMNI_AUDIO_BIAS,
                                     request->audio, request->audio_rows, audio_embed, 0,
                                     &published, err);
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_MINIMAX_H3_OMNI_CONDITION_WEIGHT,
                                     external_weights + YVEX_MINIMAX_H3_OMNI_CONDITION_BIAS,
                                     request->conditioning, request->text_rows, text_embed, 1,
                                     &published, err);
    if (rc == YVEX_OK)
        rc = transformer_refine(backend, external_weights + YVEX_MINIMAX_H3_OMNI_REFINER_WEIGHTS,
                                text_embed, request->text_rows, text_refined, &published, err);
    if (rc == YVEX_OK)
        rc = transformer_time_embed(backend, external_weights, request->timesteps,
                                    request->timestep_count, temb, &published, err);
    if (rc == YVEX_OK) memset(packed, 0, (size_t)hidden_values * sizeof(float));
    for (row = 0ull; rc == YVEX_OK && row < request->video_rows; ++row)
        for (lane = 0ull; lane < OMNI_HIDDEN; ++lane)
            packed[request->video_indices[row] * OMNI_HIDDEN + lane] =
                text_bf16_value(video_embed[row * OMNI_HIDDEN + lane]);
    for (row = 0ull; rc == YVEX_OK && row < request->audio_rows; ++row)
        for (lane = 0ull; lane < OMNI_HIDDEN; ++lane)
            packed[request->audio_indices[row] * OMNI_HIDDEN + lane] =
                text_bf16_value(audio_embed[row * OMNI_HIDDEN + lane]);
    for (row = 0ull; rc == YVEX_OK && row < request->text_rows; ++row)
        memcpy(packed + request->text_indices[row] * OMNI_HIDDEN,
               text_refined + row * OMNI_HIDDEN, OMNI_HIDDEN * sizeof(float));
    for (row = 0ull; rc == YVEX_OK && row < request->packed_rows; ++row)
        adaln[row] = request->timestep_indices[row] * OMNI_MODALITIES + request->token_tags[row];
    if (rc == YVEX_OK)
        rc = omni_blocks_execute(backend, block_weights, request->block_count, residency_identity,
            resident_bytes, packed, temb, request->timestep_count, request->position_ids, adaln,
            request->packed_rows, block_output, hidden_values, &blocks,
            (const float *)external_weights[YVEX_MINIMAX_H3_OMNI_ROPE_INV_FREQ].encoded, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(published.kernel_launches, blocks.kernel_launches,
                            &published.kernel_launches) ||
         !yvex_core_u64_add(published.h2d_bytes, blocks.h2d_bytes, &published.h2d_bytes) ||
         !yvex_core_u64_add(published.d2h_bytes, blocks.d2h_bytes, &published.d2h_bytes)))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.minimax-h3.transformer.facts",
                                 "block-stack accounting overflowed");
    if (blocks.device_bytes > published.device_bytes) published.device_bytes = blocks.device_bytes;
    if (rc == YVEX_OK)
        rc = transformer_final_norm(backend, external_weights, block_output, temb,
                                    request->timestep_indices, request->packed_rows,
                                    request->timestep_count, normalized, &published, err);
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_MINIMAX_H3_OMNI_VIDEO_OUT_WEIGHT,
                                     external_weights + YVEX_MINIMAX_H3_OMNI_VIDEO_OUT_BIAS,
                                     normalized, request->packed_rows, all_video, 0, &published, err);
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_MINIMAX_H3_OMNI_AUDIO_OUT_WEIGHT,
                                     external_weights + YVEX_MINIMAX_H3_OMNI_AUDIO_OUT_BIAS,
                                     normalized, request->packed_rows, all_audio, 0, &published, err);
    for (row = 0ull; rc == YVEX_OK && row < request->video_rows; ++row)
        memcpy(staged_video + row * 96ull, all_video + request->video_indices[row] * 96ull,
               96ull * sizeof(float));
    for (row = 0ull; rc == YVEX_OK && row < request->audio_rows; ++row)
        memcpy(staged_audio + row * 32ull, all_audio + request->audio_indices[row] * 32ull,
               32ull * sizeof(float));
    if (rc == YVEX_OK && !transformer_execution_identity(
            request, residency_identity, blocks.execution_identity, staged_video, video_values,
            staged_audio, audio_values, published.execution_identity))
        rc = conditioning_refuse(err, YVEX_ERR_STATE, "cuda.minimax-h3.transformer.identity",
                                 "Transformer execution identity could not be sealed");
    if (rc == YVEX_OK) {
        memcpy(request->video_output, staged_video, (size_t)video_values * sizeof(float));
        memcpy(request->audio_output, staged_audio, (size_t)audio_values * sizeof(float));
        published.video_rows = request->video_rows; published.audio_rows = request->audio_rows;
        published.text_rows = request->text_rows; published.packed_rows = request->packed_rows;
        published.block_count = request->block_count; published.resident_bytes = resident_bytes;
        memcpy(published.residency_identity, residency_identity, 65u);
        published.complete = 1; *result = published; yvex_error_clear(err);
    }
    free(adaln); free(staged_audio); free(staged_video); free(all_audio); free(all_video);
    free(normalized); free(temb); free(block_output); free(packed); free(text_refined);
    free(text_embed); free(audio_embed); free(video_embed);
    return rc;
}
const yvex_minimax_h3_backend_api *yvex_backend_register_minimax_h3(void)
{
    static const yvex_minimax_h3_backend_api api = {
        text_embed_cuda, text_layer_cuda, omni_blocks_cuda, omni_transformer_cuda,
    };
    return &api;
}
