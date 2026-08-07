/* Execute admitted MiniMax-H3 operations through generic CUDA primitives. */
#include <yvex/backend.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/qtype.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_HIDDEN = 5120u,
    TEXT_VOCAB = 151936u,
    TEXT_MAX_TOKENS = 256u,
    TEXT_IDENTITY_CAP = 65u
};

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

const yvex_minimax_h3_backend_api *yvex_backend_register_minimax_h3(void)
{
    static const yvex_minimax_h3_backend_api api = {text_embed_cuda};
    return &api;
}
