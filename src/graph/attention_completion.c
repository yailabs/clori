/* Resolve ordered device publications only after their shared CUDA barrier is observable. */
#include <yvex/internal/graph.h>

#include <string.h>

#include "src/graph/private.h"

static yvex_attention_failure_code completion_failure_code(
    yvex_backend_attention_failure_code code)
{
    switch (code) {
    case YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT:
        return YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT;
    case YVEX_BACKEND_ATTENTION_FAILURE_BUDGET:
        return YVEX_ATTENTION_FAILURE_SCRATCH;
    case YVEX_BACKEND_ATTENTION_FAILURE_ALLOCATION:
        return YVEX_ATTENTION_FAILURE_ALLOCATION;
    case YVEX_BACKEND_ATTENTION_FAILURE_NUMERIC:
        return YVEX_ATTENTION_FAILURE_NUMERIC;
    case YVEX_BACKEND_ATTENTION_FAILURE_CANCELLED:
        return YVEX_ATTENTION_FAILURE_CANCELLED;
    case YVEX_BACKEND_ATTENTION_FAILURE_CLEANUP:
        return YVEX_ATTENTION_FAILURE_CLEANUP;
    default:
        return YVEX_ATTENTION_FAILURE_BACKEND;
    }
}

static int completion_reject(
    yvex_attention_failure *failure, yvex_attention_failure_code code,
    const yvex_attention_publication *publication,
    unsigned long long expected, unsigned long long actual,
    yvex_status status, const char *reason, yvex_error *err)
{
    yvex_attention_reject(
        failure, code, NULL,
        publication ? publication->layer_index : YVEX_ATTENTION_NO_LAYER,
        YVEX_TENSOR_ROLE_UNKNOWN, expected, actual, err, status, reason);
    if (publication)
        yvex_error_setf(
            err, status, "graph.attention.completion",
            "%s: layer=%llu expected=%llu actual=%llu", reason,
            publication->layer_index, expected, actual);
    return status;
}

static int completion_publication_finalize(
    yvex_backend_attention_completion *completion,
    yvex_attention_publication *publication,
    yvex_attention_cpu_result *evidence,
    yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_backend_attention_output *output = &completion->output;
    unsigned long long token, storage_stride, semantic_stride;
    if (!publication || !evidence || !publication->complete ||
        !publication->device_completion_pending ||
        publication->evidence_level != YVEX_ATTENTION_EVIDENCE_NONE ||
        output->tokens_executed != publication->token_count ||
        output->compressed_count != publication->compressed_count ||
        output->indexer_count != publication->indexer_count)
        return completion_reject(
            failure, YVEX_ATTENTION_FAILURE_BACKEND, publication,
            publication ? publication->token_count : 1ull,
            output->tokens_executed, YVEX_ERR_STATE,
            "deferred attention publication disagrees with completed CUDA work", err);
    storage_stride = publication->topk_stride;
    semantic_stride = output->topk_count;
    if (semantic_stride > storage_stride)
        return completion_reject(
            failure, YVEX_ATTENTION_FAILURE_BACKEND, publication,
            storage_stride, semantic_stride, YVEX_ERR_BOUNDS,
            "completed attention top-k exceeds its publication storage", err);
    if (publication->topk_positions && semantic_stride < storage_stride)
        for (token = 1ull; token < publication->token_count; ++token)
            memmove(publication->topk_positions + token * semantic_stride,
                    publication->topk_positions + token * storage_stride,
                    (size_t)semantic_stride * sizeof(*publication->topk_positions));
    publication->topk_stride = semantic_stride;
    publication->device_completion_pending = 0;
    evidence->topk_candidates = output->valid_candidate_count;
    evidence->topk_selected = semantic_stride;
    evidence->cuda_stream_synchronizations = output->stream_synchronizations;
    evidence->cuda_device_synchronizations = output->device_synchronizations;
    evidence->cuda_device_execution_elapsed_ns = output->device_execution_elapsed_ns;
    return YVEX_OK;
}

int yvex_attention_device_completion_resolve(
    yvex_backend *backend, yvex_backend_attention_completion *completion,
    yvex_attention_publication *publication, yvex_attention_cpu_result *evidence,
    const yvex_attention_probe_state_provider *provider,
    const yvex_attention_cancellation *cancellation, int barrier_observed,
    char state_delta_identity[YVEX_SHA256_HEX_CAP],
    yvex_attention_failure *failure, yvex_error *err)
{
    int rc;
    if (state_delta_identity) state_delta_identity[0] = '\0';
    if (!backend || !completion || !publication || !evidence || !provider ||
        !provider->context || !provider->stage || !state_delta_identity)
        return completion_reject(
            failure, YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT, publication,
            1ull, 0ull, YVEX_ERR_INVALID_ARG,
            "deferred attention completion contract is incomplete", err);
    rc = yvex_backend_attention_complete(
        backend, completion, barrier_observed, err);
    if (rc != YVEX_OK)
        return completion_reject(
            failure, completion_failure_code(completion->failure.code), publication,
            completion->failure.expected, completion->failure.actual,
            (yvex_status)rc,
            completion->failure.stage
                ? completion->failure.stage
                : "CUDA attention completion failed", err);
    rc = completion_publication_finalize(
        completion, publication, evidence, failure, err);
    if (rc == YVEX_OK)
        rc = provider->stage(
            provider->context, publication, cancellation,
            state_delta_identity, failure, err);
    if (rc == YVEX_OK && !yvex_sha256_hex_valid(state_delta_identity))
        rc = completion_reject(
            failure, YVEX_ATTENTION_FAILURE_STATE_DELTA, publication,
            1ull, 0ull, YVEX_ERR_STATE,
            "completed attention state delta identity is invalid", err);
    if (rc == YVEX_OK) {
        if (failure) memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_attention_state_provider_abort(
    const yvex_attention_probe_state_provider *provider, int primary_status,
    yvex_attention_failure *failure, yvex_error *err)
{
    yvex_attention_failure primary_failure = failure
        ? *failure : (yvex_attention_failure){0};
    yvex_attention_failure cleanup_failure = {0};
    yvex_error primary_error = err ? *err : (yvex_error){0};
    yvex_error cleanup_error = {0};
    int rc;
    if (!provider) return YVEX_OK;
    if (!provider->context || !provider->abort)
        return completion_reject(
            failure, YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            1ull, 0ull, YVEX_ERR_INVALID_ARG,
            "attention state abort owner is incomplete", err);
    rc = provider->abort(provider->context, &cleanup_failure, &cleanup_error);
    if (rc != YVEX_OK) {
        if (failure) *failure = cleanup_failure;
        if (err) *err = cleanup_error;
        return rc;
    }
    if (primary_status != YVEX_OK) {
        if (failure) *failure = primary_failure;
        if (err) {
            if (yvex_error_is_set(&primary_error))
                *err = primary_error;
            else
                yvex_error_set(
                    err, (yvex_status)primary_status,
                    "graph.attention.execute",
                    primary_failure.reason
                        ? primary_failure.reason
                        : "attention execution failed before rollback");
        }
    } else {
        if (failure) memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    return YVEX_OK;
}

int yvex_attention_publication_identity_build(
    const char *plan_identity, const char *logical_model_identity,
    const char *input_identity, yvex_attention_operation_scope scope,
    int candidate_visible, int retain_prefix,
    yvex_attention_publication *publication)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!plan_identity || !logical_model_identity || !input_identity ||
        !publication || !publication->token_count)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.publication.v1") ||
        !yvex_sha256_update_text(&hash, plan_identity) ||
        !yvex_sha256_update_text(&hash, logical_model_identity) ||
        !yvex_sha256_update_text(&hash, input_identity) ||
        !yvex_sha256_update_u64(&hash, publication->layer_index) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)scope) ||
        !yvex_sha256_update_u64(&hash, publication->token_position) ||
        !yvex_sha256_update_u64(&hash, publication->token_count) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)candidate_visible) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)retain_prefix) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, publication->execution_identity);
    return 1;
}

void yvex_attention_comparison_failure_publish(
    yvex_attention_probe_result *result,
    const yvex_attention_probe_result *candidate)
{
    size_t identities = offsetof(yvex_attention_probe_result, cuda_device) -
                        offsetof(yvex_attention_probe_result, cpu_output_digest);
    size_t counts = offsetof(yvex_attention_probe_result, cuda_compute_capability_major) -
                    offsetof(yvex_attention_probe_result, comparison_values);
    size_t metrics = sizeof(*result) -
                     offsetof(yvex_attention_probe_result,
                              comparison_maximum_absolute_error);
    memcpy(&result->cpu_output_digest, &candidate->cpu_output_digest, identities);
    memcpy(&result->comparison_values, &candidate->comparison_values, counts);
    memcpy(&result->comparison_maximum_absolute_error,
           &candidate->comparison_maximum_absolute_error, metrics);
    result->comparison_passed = 0;
}
