/* Project the backend-neutral exact-attention contract onto CUDA GQA execution. */
#include <yvex/internal/transformer.h>

#include "src/backend/cuda/transformer_ops.h"

enum { YVEX_CUDA_EXACT_ATTENTION_HEAD_MAX = 256u };

static int exact_attention_validate(
    const yvex_transformer_attention_requirement *requirement,
    const char *stage, yvex_error *err)
{
    unsigned long long query_end, query_width, key_value_width;
    if (!requirement || !requirement->query_tokens ||
        !requirement->key_value_tokens ||
        !yvex_core_u64_add(requirement->query_start,
                           requirement->query_tokens, &query_end) ||
        query_end > requirement->key_value_tokens ||
        !requirement->query_heads || !requirement->key_value_heads ||
        requirement->query_heads % requirement->key_value_heads ||
        !requirement->head_dimension ||
        !yvex_core_u64_mul(requirement->query_heads,
                           requirement->head_dimension, &query_width) ||
        !yvex_core_u64_mul(requirement->key_value_heads,
                           requirement->head_dimension, &key_value_width) ||
        (requirement->query_token_stride &&
         requirement->query_token_stride < query_width) ||
        (requirement->key_token_stride &&
         requirement->key_token_stride < key_value_width) ||
        (requirement->value_token_stride &&
         requirement->value_token_stride < key_value_width)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "complete exact-attention geometry is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (requirement->layout != YVEX_TRANSFORMER_ATTENTION_LAYOUT_TOKEN_HEAD_DIM ||
        requirement->mask < YVEX_TRANSFORMER_ATTENTION_MASK_FULL ||
        requirement->mask > YVEX_TRANSFORMER_ATTENTION_MASK_CAUSAL ||
        requirement->numeric_contract !=
            YVEX_TRANSFORMER_ATTENTION_NUMERIC_EXACT_F32 ||
        requirement->head_dimension > YVEX_CUDA_EXACT_ATTENTION_HEAD_MAX ||
        requirement->query_dtype != YVEX_DTYPE_F32 ||
        requirement->key_dtype != YVEX_DTYPE_F32 ||
        requirement->value_dtype != YVEX_DTYPE_F32 ||
        requirement->output_dtype != YVEX_DTYPE_F32 ||
        !requirement->deterministic) {
        yvex_error_set(
            err, YVEX_ERR_UNSUPPORTED, stage,
            "CUDA requires deterministic token/head/dimension exact F32 attention");
        return YVEX_ERR_UNSUPPORTED;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_transformer_attention_workspace_required(
    const yvex_transformer_attention_requirement *requirement,
    unsigned long long *bytes, yvex_error *err)
{
    unsigned long long query_width, key_value_width;
    int rc;
    if (bytes) *bytes = 0ull;
    rc = exact_attention_validate(
        requirement, "cuda.transformer.attention.workspace", err);
    if (rc != YVEX_OK) return rc;
    query_width = requirement->query_heads * requirement->head_dimension;
    key_value_width = requirement->key_value_heads * requirement->head_dimension;
    if ((requirement->query_token_stride &&
         requirement->query_token_stride != query_width) ||
        (requirement->key_token_stride &&
         requirement->key_token_stride != key_value_width) ||
        (requirement->value_token_stride &&
         requirement->value_token_stride != key_value_width)) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    return yvex_cuda_transformer_gqa_workspace_required(
        requirement->query_tokens, requirement->key_value_tokens,
        requirement->query_heads, requirement->key_value_heads,
        requirement->head_dimension, bytes, err);
}

int yvex_cuda_transformer_attention_execute(
    yvex_backend *backend, const yvex_transformer_attention_request *request,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    int rc;
    if (!request || !request->query || !request->key || !request->value ||
        !request->output || !facts) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.transformer.attention",
                       "complete exact-attention tensors and result facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = exact_attention_validate(
        &request->requirement, "cuda.transformer.attention", err);
    if (rc != YVEX_OK) return rc;
    return yvex_cuda_transformer_gqa_strided(
        backend, request->query, request->key, request->value, request->output,
        request->requirement.query_tokens,
        request->requirement.key_value_tokens,
        request->requirement.query_start, request->requirement.query_heads,
        request->requirement.key_value_heads,
        request->requirement.head_dimension,
        request->requirement.query_token_stride,
        request->requirement.key_token_stride,
        request->requirement.value_token_stride,
        request->requirement.mask == YVEX_TRANSFORMER_ATTENTION_MASK_CAUSAL,
        facts, err);
}
