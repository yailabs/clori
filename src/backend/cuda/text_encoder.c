/* Execute an admitted text-encoder operation through generic CUDA primitives. */
#include "src/backend/cuda/private.h"

#include <yvex/backend.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/component.h>
#include <yvex/internal/core.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/qtype.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_IDENTITY_CAP = 65u
};

typedef struct {
    unsigned long long hidden, vocabulary, ffn, query_heads, kv_heads;
    unsigned long long head_dimension, query_width, kv_width, rope_theta, layer_capacity;
    const char *semantic_identity, *embedding_identity_domain, *encoder_identity_domain;
    float normalization_epsilon;
} text_geometry;

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
    const yvex_backend_text_weight *weights;
    yvex_device_tensor *device[TEXT_DEVICE_COUNT];
    text_geometry geometry;
    unsigned long long tokens, values, output_bytes, device_bytes;
    unsigned long long layer_count, layer_index;
    yvex_backend_text_execution_result facts;
} text_layer_run;

static const unsigned int text_zero_row = 0u;

static int text_geometry_build(const yvex_component_text_recipe *source,
                               text_geometry *out)
{
    text_geometry geometry = {0};
    if (!source || !out ||
        source->schema_version != YVEX_COMPONENT_TEXT_RECIPE_SCHEMA_V1 ||
        !yvex_sha256_hex_valid(source->semantic_identity) ||
        !source->embedding_identity_domain || !*source->embedding_identity_domain ||
        !source->encoder_identity_domain || !*source->encoder_identity_domain ||
        !source->layer_capacity ||
        !source->hidden_width || !source->vocabulary_size || !source->ffn_width ||
        !source->query_heads || !source->kv_heads || !source->head_dimension ||
        source->query_heads % source->kv_heads || source->head_dimension % 2ull ||
        !source->rope_theta || source->normalization_epsilon <= 0.0f ||
        !yvex_core_u64_mul(source->query_heads, source->head_dimension,
                          &geometry.query_width) ||
        !yvex_core_u64_mul(source->kv_heads, source->head_dimension,
                          &geometry.kv_width))
        return 0;
    geometry.hidden = source->hidden_width;
    geometry.vocabulary = source->vocabulary_size;
    geometry.ffn = source->ffn_width;
    geometry.query_heads = source->query_heads;
    geometry.kv_heads = source->kv_heads;
    geometry.head_dimension = source->head_dimension;
    geometry.rope_theta = source->rope_theta;
    geometry.layer_capacity = source->layer_capacity;
    geometry.semantic_identity = source->semantic_identity;
    geometry.embedding_identity_domain = source->embedding_identity_domain;
    geometry.encoder_identity_domain = source->encoder_identity_domain;
    geometry.normalization_epsilon = source->normalization_epsilon;
    *out = geometry;
    return 1;
}

static int conditioning_refuse(yvex_error *err, yvex_status status, const char *stage,
                               const char *message)
{
    yvex_error_set(err, status, stage, message);
    return status;
}

static int conditioning_identity(
    const text_geometry *geometry, const char *residency_identity, const unsigned int *token_ids,
    unsigned long long token_count, const float *values, unsigned long long value_count,
    char output[TEXT_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, geometry->embedding_identity_domain) ||
        !yvex_sha256_update_text(&hash, geometry->semantic_identity) ||
        !yvex_sha256_update_text(&hash, residency_identity) ||
        !yvex_sha256_update_u64(&hash, token_count) ||
        !yvex_sha256_update_u64(&hash, geometry->hidden) ||
        !yvex_sha256_update_u64(&hash, geometry->vocabulary))
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
    yvex_backend *backend, const text_geometry *geometry, const unsigned char *encoded,
    unsigned long long encoded_bytes,
    unsigned int qtype, unsigned long long row_count, unsigned long long row_width,
    unsigned long long row_bytes, const char *residency_identity,
    unsigned long long resident_bytes, const unsigned int *token_ids,
    unsigned long long token_count, float *output, unsigned long long output_capacity,
    yvex_backend_text_execution_result *result, unsigned long long *value_count,
    unsigned long long *output_bytes, yvex_error *err)
{
    unsigned long long expected_encoded, row_bytes_total, index;

    if (!backend || !encoded || !yvex_sha256_hex_valid(residency_identity) || !resident_bytes ||
        !geometry || !token_ids || !token_count || !output || !result ||
        !yvex_core_u64_mul(geometry->hidden, 2ull, &row_bytes_total) ||
        !yvex_core_u64_mul(row_bytes_total, geometry->vocabulary, &expected_encoded) ||
        encoded_bytes != expected_encoded || qtype != YVEX_GGUF_QTYPE_BF16 ||
        row_count != geometry->vocabulary ||
        row_width != geometry->hidden || row_bytes != row_bytes_total ||
        resident_bytes < encoded_bytes ||
        !yvex_core_u64_mul(token_count, geometry->hidden, value_count) ||
        *value_count > output_capacity ||
        !yvex_core_u64_mul(*value_count, sizeof(float), output_bytes) ||
        *output_bytes > SIZE_MAX)
        return conditioning_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.text-embedding.validate",
            "admitted BF16 embedding, resident identity, token input, and output are required");
    for (index = 0ull; index < token_count; ++index)
        if (token_ids[index] >= geometry->vocabulary)
            return conditioning_refuse(
                err, YVEX_ERR_BOUNDS, "cuda.text-embedding.token",
                "text token identifier exceeds the compiled vocabulary");
    return YVEX_OK;
}

int yvex_backend_text_embedding_execute(
    yvex_backend *backend, const yvex_component_text_recipe *source,
    const unsigned char *encoded, unsigned long long encoded_bytes,
    unsigned int qtype, unsigned long long row_count, unsigned long long row_width,
    unsigned long long row_bytes, const char *residency_identity,
    unsigned long long resident_bytes, const unsigned int *token_ids,
    unsigned long long token_count, float *output, unsigned long long output_capacity,
    yvex_backend_text_execution_result *result, yvex_error *err)
{
    text_geometry geometry;
    yvex_backend_cuda_operation_facts operation = {0};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *device_output = NULL;
    yvex_backend_text_execution_result published = {0};
    float *staged = NULL;
    unsigned long long value_count = 0ull, output_bytes = 0ull;
    int rc, cleanup_rc;
    yvex_error cleanup;

    if (result) memset(result, 0, sizeof(*result));
    if (!text_geometry_build(source, &geometry))
        return conditioning_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.text-geometry",
            "an admitted text encoder geometry is required");
    rc = text_embed_validate(
        backend, &geometry, encoded, encoded_bytes, qtype, row_count, row_width, row_bytes,
        residency_identity, resident_bytes, token_ids, token_count, output,
        output_capacity, result, &value_count, &output_bytes, err);
    if (rc == YVEX_OK) {
        staged = (float *)malloc((size_t)output_bytes);
        if (!staged)
            rc = conditioning_refuse(
                err, YVEX_ERR_NOMEM, "cuda.text-embedding.output",
                "transactional conditioning output allocation failed");
    }
    descriptor.name = "text-encoder-conditioning";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 2u;
    descriptor.dims[0] = token_count;
    descriptor.dims[1] = geometry.hidden;
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
            &geometry, residency_identity, token_ids, token_count, staged, value_count,
            published.execution_identity))
        rc = conditioning_refuse(
            err, YVEX_ERR_STATE, "cuda.text-embedding.identity",
            "conditioning execution identity could not be sealed");
    if (device_output) {
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(backend, &device_output, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    if (rc == YVEX_OK) {
        memcpy(output, staged, (size_t)output_bytes);
        published.token_count = token_count;
        published.hidden_width = geometry.hidden;
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
    const text_geometry *geometry, const char *residency_identity,
    const unsigned int *token_ids,
    unsigned long long token_count, unsigned long long layer_count,
    const float *values, unsigned long long value_count,
    char output[TEXT_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, geometry->encoder_identity_domain) ||
        !yvex_sha256_update_text(&hash, geometry->semantic_identity) ||
        !yvex_sha256_update_text(&hash, residency_identity) ||
        !yvex_sha256_update_u64(&hash, layer_count) ||
        !yvex_sha256_update_u64(&hash, token_count) ||
        !yvex_sha256_update_u64(&hash, geometry->hidden) ||
        !yvex_sha256_update_u64(&hash, geometry->query_heads) ||
        !yvex_sha256_update_u64(&hash, geometry->kv_heads) ||
        !yvex_sha256_update_u64(&hash, geometry->head_dimension) ||
        !yvex_sha256_update_u64(&hash, geometry->ffn)) return 0;
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
    const text_geometry *geometry, const yvex_backend_text_weight *weights,
    unsigned long long layer_count,
    unsigned long long *weight_bytes)
{
    unsigned long long rows[YVEX_BACKEND_TEXT_WEIGHT_COUNT] = {
        geometry ? geometry->vocabulary : 0ull, 1u,
        geometry ? geometry->query_width : 0ull,
        geometry ? geometry->kv_width : 0ull,
        geometry ? geometry->kv_width : 0ull,
        geometry ? geometry->hidden : 0ull,
        1u, 1u, 1u, geometry ? geometry->ffn : 0ull,
        geometry ? geometry->ffn : 0ull, geometry ? geometry->hidden : 0ull,
    };
    unsigned long long widths[YVEX_BACKEND_TEXT_WEIGHT_COUNT] = {
        geometry ? geometry->hidden : 0ull, geometry ? geometry->hidden : 0ull,
        geometry ? geometry->hidden : 0ull, geometry ? geometry->hidden : 0ull,
        geometry ? geometry->hidden : 0ull, geometry ? geometry->query_width : 0ull,
        geometry ? geometry->head_dimension : 0ull,
        geometry ? geometry->head_dimension : 0ull,
        geometry ? geometry->hidden : 0ull, geometry ? geometry->hidden : 0ull,
        geometry ? geometry->hidden : 0ull, geometry ? geometry->ffn : 0ull,
    };
    unsigned long long count, index;
    if (weight_bytes) *weight_bytes = 0ull;
    if (!geometry || !weights || !layer_count || !weight_bytes ||
        !yvex_core_u64_mul(layer_count, YVEX_BACKEND_TEXT_LAYER_WEIGHT_COUNT, &count) ||
        !yvex_core_u64_add(count, 1ull, &count)) return 0;
    for (index = 0ull; index < count; ++index) {
        const yvex_backend_text_weight *weight = weights + index;
        unsigned long long expected, slot = index ? 1ull +
            (index - 1ull) % YVEX_BACKEND_TEXT_LAYER_WEIGHT_COUNT : 0ull;
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

static int text_facts_add(yvex_backend_text_execution_result *total,
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
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.allocate",
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
    const text_geometry *geometry = &run->geometry;
    int rc;
#define ALLOC(slot, name, rows, width, rank_one) \
    if (rc == YVEX_OK) rc = text_tensor_allocate(run, slot, name, rows, width, rank_one, err)
    rc = text_tensor_allocate(run, TEXT_DEVICE_HIDDEN, "text-hidden",
                              run->tokens, geometry->hidden, 0, err);
    ALLOC(TEXT_DEVICE_NORM, "text-norm", run->tokens, geometry->hidden, 0);
    ALLOC(TEXT_DEVICE_NORM_WEIGHT, "text-norm-weight", 1ull, geometry->hidden, 1);
    ALLOC(TEXT_DEVICE_QUERY, "text-query", run->tokens * geometry->query_heads,
          geometry->head_dimension, 0);
    ALLOC(TEXT_DEVICE_KEY, "text-key", run->tokens * geometry->kv_heads,
          geometry->head_dimension, 0);
    ALLOC(TEXT_DEVICE_VALUE, "text-value", run->tokens * geometry->kv_heads,
          geometry->head_dimension, 0);
    ALLOC(TEXT_DEVICE_Q_NORM, "text-q-norm", 1ull, geometry->head_dimension, 1);
    ALLOC(TEXT_DEVICE_K_NORM, "text-k-norm", 1ull, geometry->head_dimension, 1);
    ALLOC(TEXT_DEVICE_COSINE, "text-cosine", run->tokens, geometry->head_dimension, 0);
    ALLOC(TEXT_DEVICE_SINE, "text-sine", run->tokens, geometry->head_dimension, 0);
    ALLOC(TEXT_DEVICE_ATTENTION, "text-attention", run->tokens, geometry->query_width, 0);
    ALLOC(TEXT_DEVICE_RESIDUAL, "text-residual", run->tokens, geometry->hidden, 0);
    ALLOC(TEXT_DEVICE_GATE, "text-gate", run->tokens, geometry->ffn, 0);
    ALLOC(TEXT_DEVICE_UP, "text-up", run->tokens, geometry->ffn, 0);
    ALLOC(TEXT_DEVICE_PRODUCT, "text-product", run->tokens, geometry->ffn, 0);
#undef ALLOC
    return rc;
}

static const yvex_backend_text_weight *text_weight(
    const text_layer_run *run, yvex_backend_text_weight_slot slot)
{
    if (slot == YVEX_BACKEND_TEXT_EMBEDDING) return run->weights;
    return run->weights + 1ull +
           run->layer_index * YVEX_BACKEND_TEXT_LAYER_WEIGHT_COUNT + slot - 1ull;
}

static int text_weight_gather(text_layer_run *run, yvex_backend_text_weight_slot slot,
                              yvex_device_tensor *output, const unsigned int *rows,
                              unsigned long long row_count, yvex_error *err)
{
    const yvex_backend_text_weight *weight = text_weight(run, slot);
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_gather(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes,
        rows, row_count, output, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
                                 "text gather accounting overflowed");
    return rc;
}

static int text_weight_project(text_layer_run *run, yvex_backend_text_weight_slot slot,
                               const yvex_device_tensor *input,
                               const yvex_device_tensor *additive,
                               yvex_device_tensor *output, yvex_error *err)
{
    const yvex_backend_text_weight *weight = text_weight(run, slot);
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_matvec(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes, run->tokens,
        input, NULL, 0ull, additive, output, 0, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
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
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
                                 "text rounding accounting overflowed");
    return rc;
}

static int text_norm(text_layer_run *run, text_device_slot input,
                     yvex_backend_text_weight_slot weight,
                     text_device_slot output, unsigned long long count,
                     yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    text_device_slot weight_device =
        weight == YVEX_BACKEND_TEXT_Q_NORM ? TEXT_DEVICE_Q_NORM
        : weight == YVEX_BACKEND_TEXT_K_NORM ? TEXT_DEVICE_K_NORM
                                                : TEXT_DEVICE_NORM_WEIGHT;
    unsigned long long width =
        weight == YVEX_BACKEND_TEXT_Q_NORM || weight == YVEX_BACKEND_TEXT_K_NORM
            ? run->geometry.head_dimension : run->geometry.hidden;
    int rc = text_weight_gather(
        run, weight, run->device[weight_device], &text_zero_row, 1ull, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rms_norm_bf16(
            run->backend, run->device[input], run->device[weight_device],
            run->device[output], count / width, width,
            run->geometry.normalization_epsilon, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
                                 "text normalization accounting overflowed");
    return rc;
}

static float text_bf16_value(float value)
{
    return yvex_quant_bf16_decode(yvex_quant_bf16_encode(value));
}

static int text_rope_tables(text_layer_run *run, yvex_error *err)
{
    const text_geometry *geometry = &run->geometry;
    float *cosines = NULL, *sines = NULL;
    unsigned long long token, coordinate, elements, bytes;
    int rc = YVEX_OK;
    if (!yvex_core_u64_mul(run->tokens, geometry->head_dimension, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) || bytes > SIZE_MAX)
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.rope",
                                   "text rotary table geometry overflowed");
    cosines = (float *)malloc((size_t)bytes);
    sines = (float *)malloc((size_t)bytes);
    if (!cosines || !sines)
        rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.text-layer.rope",
                                 "text rotary table allocation failed");
    for (token = 0ull; rc == YVEX_OK && token < run->tokens; ++token) {
        for (coordinate = 0ull; coordinate < geometry->head_dimension; ++coordinate) {
            unsigned long long pair = coordinate % (geometry->head_dimension / 2ull);
            double frequency = pow((double)geometry->rope_theta,
                                   -(double)(2ull * pair) / geometry->head_dimension);
            double angle = (double)token * frequency;
            unsigned long long index = token * geometry->head_dimension + coordinate;
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
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
                                 "text rotary upload accounting overflowed");
    free(sines);
    free(cosines);
    return rc;
}

static int text_attention(text_layer_run *run, yvex_error *err)
{
    const text_geometry *geometry = &run->geometry;
    yvex_backend_cuda_operation_facts facts;
    unsigned long long q_values = run->tokens * geometry->query_width;
    unsigned long long kv_values = run->tokens * geometry->kv_width;
    int rc = text_weight_project(run, YVEX_BACKEND_TEXT_Q_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_QUERY], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_QUERY, q_values, err);
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_BACKEND_TEXT_K_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_KEY], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_KEY, kv_values, err);
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_BACKEND_TEXT_V_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_VALUE], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_VALUE, kv_values, err);
    if (rc == YVEX_OK)
        rc = text_norm(run, TEXT_DEVICE_QUERY, YVEX_BACKEND_TEXT_Q_NORM,
                       TEXT_DEVICE_QUERY, q_values, err);
    if (rc == YVEX_OK)
        rc = text_norm(run, TEXT_DEVICE_KEY, YVEX_BACKEND_TEXT_K_NORM,
                       TEXT_DEVICE_KEY, kv_values, err);
    if (rc == YVEX_OK) rc = text_rope_tables(run, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half(
            run->backend, run->device[TEXT_DEVICE_QUERY], run->device[TEXT_DEVICE_COSINE],
            run->device[TEXT_DEVICE_SINE], run->tokens, geometry->query_heads,
            geometry->head_dimension, geometry->head_dimension,
            &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
                                 "text rotary accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half(
            run->backend, run->device[TEXT_DEVICE_KEY], run->device[TEXT_DEVICE_COSINE],
            run->device[TEXT_DEVICE_SINE], run->tokens, geometry->kv_heads,
            geometry->head_dimension, geometry->head_dimension,
            &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
                                 "text rotary accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gqa(
            run->backend, run->device[TEXT_DEVICE_QUERY], run->device[TEXT_DEVICE_KEY],
            run->device[TEXT_DEVICE_VALUE], run->device[TEXT_DEVICE_ATTENTION],
            run->tokens, geometry->query_heads, geometry->kv_heads,
            geometry->head_dimension, 1, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
                                 "text attention accounting overflowed");
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_ATTENTION, q_values, err);
    return rc;
}

static int text_mlp(text_layer_run *run, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    unsigned long long ffn_values = run->tokens * run->geometry.ffn;
    int rc = text_norm(run, TEXT_DEVICE_RESIDUAL, YVEX_BACKEND_TEXT_POST_NORM,
                       TEXT_DEVICE_NORM, run->values, err);
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_BACKEND_TEXT_GATE_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_GATE], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_GATE, ffn_values, err);
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_BACKEND_TEXT_UP_PROJECTION,
                                 run->device[TEXT_DEVICE_NORM], NULL,
                                 run->device[TEXT_DEVICE_UP], err);
    if (rc == YVEX_OK) rc = text_round(run, TEXT_DEVICE_UP, ffn_values, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_silu_product_bf16(
            run->backend, run->device[TEXT_DEVICE_GATE], run->device[TEXT_DEVICE_UP],
            run->device[TEXT_DEVICE_PRODUCT], ffn_values, &facts, err);
    if (rc == YVEX_OK && !text_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
                                 "text MLP accounting overflowed");
    if (rc == YVEX_OK)
        rc = text_weight_project(run, YVEX_BACKEND_TEXT_DOWN_PROJECTION,
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
        rc = text_weight_gather(run, YVEX_BACKEND_TEXT_EMBEDDING,
                                run->device[TEXT_DEVICE_HIDDEN], token_ids,
                                run->tokens, err);
    for (run->layer_index = 0ull; rc == YVEX_OK && run->layer_index < run->layer_count;
         ++run->layer_index) {
        rc = text_norm(run, TEXT_DEVICE_HIDDEN, YVEX_BACKEND_TEXT_INPUT_NORM,
                       TEXT_DEVICE_NORM, run->values, err);
        if (rc == YVEX_OK) rc = text_attention(run, err);
        if (rc == YVEX_OK)
            rc = text_weight_project(run, YVEX_BACKEND_TEXT_O_PROJECTION,
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
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.text-layer.facts",
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

int yvex_backend_text_encoder_execute(
    yvex_backend *backend, const yvex_component_text_recipe *source,
    const yvex_backend_text_weight *weights, unsigned long long layer_count,
    const char *residency_identity, unsigned long long resident_bytes,
    const unsigned int *token_ids, unsigned long long token_count, float *output,
    unsigned long long output_capacity, yvex_backend_text_execution_result *result,
    yvex_error *err)
{
    text_layer_run run = {0};
    yvex_backend_text_execution_result published = {0};
    float *staged = NULL;
    unsigned long long index, weight_bytes = 0ull;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!text_geometry_build(source, &run.geometry) || !backend ||
        layer_count > source->layer_capacity ||
        !text_weights_validate(&run.geometry, weights, layer_count, &weight_bytes) ||
        !yvex_sha256_hex_valid(residency_identity) || resident_bytes < weight_bytes || !token_ids ||
        !token_count || !output || !result ||
        !yvex_core_u64_mul(token_count, run.geometry.hidden, &run.values) ||
        run.values > output_capacity ||
        !yvex_core_u64_mul(run.values, sizeof(float), &run.output_bytes) ||
        run.output_bytes > SIZE_MAX)
        return conditioning_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.text-layer.validate",
            "exact compiled BF16 stack weights, tokens, residency, and output are required");
    for (index = 0ull; index < token_count; ++index)
        if (token_ids[index] >= run.geometry.vocabulary)
            return conditioning_refuse(err, YVEX_ERR_BOUNDS,
                                       "cuda.text-layer.token",
                                       "text token exceeds the compiled vocabulary");
    staged = (float *)malloc((size_t)run.output_bytes);
    if (!staged)
        return conditioning_refuse(err, YVEX_ERR_NOMEM,
                                   "cuda.text-layer.output",
                                   "transactional layer output allocation failed");
    run.backend = backend;
    run.weights = weights;
    run.tokens = token_count;
    run.layer_count = layer_count;
    rc = text_layer_compute(&run, token_ids, staged, err);
    if (rc == YVEX_OK &&
        !text_layer_identity(&run.geometry, residency_identity, token_ids, token_count,
                             layer_count, staged, run.values, published.execution_identity))
        rc = conditioning_refuse(err, YVEX_ERR_STATE,
                                 "cuda.text-layer.identity",
                                 "text layer execution identity could not be sealed");
    rc = text_devices_release(&run, rc, err);
    if (rc == YVEX_OK) {
        memcpy(output, staged, (size_t)run.output_bytes);
        published.token_count = token_count;
        published.hidden_width = run.geometry.hidden;
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
