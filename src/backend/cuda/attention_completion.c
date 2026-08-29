/* Complete one or more ordered CUDA attention publications after a shared stream barrier. */
#include <yvex/internal/backend.h>

#include <limits.h>
#include <string.h>

#include "src/backend/cuda/private.h"
#include "src/backend/private.h"

static int completion_refuse(
    yvex_backend_attention_completion *completion,
    yvex_backend_attention_failure_code code, const char *stage,
    unsigned long long expected, unsigned long long actual,
    yvex_status status, const char *reason, yvex_error *err)
{
    if (completion) {
        completion->failure.code = code;
        completion->failure.stage = stage;
        completion->failure.expected = expected;
        completion->failure.actual = actual;
        completion->pending = 0;
    }
    yvex_error_set(err, status, stage, reason);
    return status;
}

static int completion_transfers_publish(
    yvex_backend_attention_completion *completion, yvex_error *err)
{
    unsigned int index;
    for (index = 0u; index < completion->transfer_count; ++index) {
        const yvex_backend_attention_completion_transfer *transfer =
            &completion->transfers[index];
        size_t bytes;
        if (!transfer->output || !transfer->staged || !transfer->width ||
            transfer->used > transfer->capacity ||
            transfer->used > transfer->output_capacity ||
            !yvex_cuda_work_checked_bytes(transfer->used, transfer->width, &bytes))
            return completion_refuse(
                completion, YVEX_BACKEND_ATTENTION_FAILURE_COPY,
                transfer->stage ? transfer->stage : "cuda.attention.complete.copy",
                transfer->output_capacity, transfer->used, YVEX_ERR_BOUNDS,
                "CUDA attention completion span is invalid", err);
        if (bytes) memcpy(transfer->output, transfer->staged, bytes);
    }
    return YVEX_OK;
}

static int completion_topk_publish(
    yvex_backend_attention_completion *completion, yvex_error *err)
{
    unsigned long long token, maximum_selected = 0ull, maximum_valid = 0ull;
    if (completion->attention_class != YVEX_BACKEND_ATTENTION_CSA) return YVEX_OK;
    if (!completion->host_selected_counts || !completion->host_candidate_counts ||
        (completion->output.topk_counts.data &&
         completion->output.topk_counts.capacity < completion->token_count) ||
        (completion->output.valid_candidate_counts.data &&
         completion->output.valid_candidate_counts.capacity < completion->token_count))
        return completion_refuse(
            completion, YVEX_BACKEND_ATTENTION_FAILURE_COPY,
            "cuda.attention.complete.topk", completion->token_count, 0ull,
            YVEX_ERR_BOUNDS, "CUDA attention completion counts are unavailable", err);
    for (token = 0ull; token < completion->token_count; ++token) {
        unsigned long long selected = completion->host_selected_counts[token];
        unsigned long long valid = completion->host_candidate_counts[token];
        unsigned long long expected = valid < completion->indexer_topk
                                          ? valid : completion->indexer_topk;
        if (valid > completion->candidate_capacity || selected != expected)
            return completion_refuse(
                completion, YVEX_BACKEND_ATTENTION_FAILURE_NUMERIC,
                "cuda.attention.complete.topk", expected, selected,
                YVEX_ERR_BOUNDS,
                "CUDA attention top-k counts violate candidate geometry", err);
        if (selected > maximum_selected) maximum_selected = selected;
        if (valid > maximum_valid) maximum_valid = valid;
    }
    if (completion->output.topk_counts.data)
        memcpy(completion->output.topk_counts.data, completion->host_selected_counts,
               (size_t)completion->token_count * sizeof(*completion->host_selected_counts));
    if (completion->output.valid_candidate_counts.data)
        memcpy(completion->output.valid_candidate_counts.data,
               completion->host_candidate_counts,
               (size_t)completion->token_count * sizeof(*completion->host_candidate_counts));
    completion->output.topk_count = maximum_selected;
    completion->output.valid_candidate_count = maximum_valid;
    return YVEX_OK;
}

static int completion_account(
    yvex_backend *backend, yvex_backend_attention_completion *completion,
    yvex_error *err)
{
    unsigned long long h2d, d2h;
    if (!yvex_core_u64_add(backend->stats.h2d_bytes,
                           completion->output.h2d_bytes, &h2d) ||
        !yvex_core_u64_add(backend->stats.d2h_bytes,
                           completion->output.d2h_bytes, &d2h))
        return completion_refuse(
            completion, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.attention.complete.account", ULLONG_MAX,
            completion->output.h2d_bytes > completion->output.d2h_bytes
                ? completion->output.h2d_bytes : completion->output.d2h_bytes,
            YVEX_ERR_BOUNDS,
            "CUDA attention cumulative transfer accounting overflowed", err);
    backend->stats.h2d_bytes = h2d;
    backend->stats.d2h_bytes = d2h;
    return YVEX_OK;
}

int yvex_backend_attention_complete(
    yvex_backend *backend, yvex_backend_attention_completion *completion,
    int barrier_observed, yvex_error *err)
{
    int device_wide = 0, rc;
    if (!backend || !completion || !completion->pending ||
        (barrier_observed != 0 && barrier_observed != 1) ||
        yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA)
        return completion_refuse(
            completion, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.complete", 1ull, 0ull, YVEX_ERR_INVALID_ARG,
            "CUDA attention completion owner is invalid", err);
    rc = barrier_observed
             ? YVEX_OK
             : yvex_cuda_launch_synchronize(
                   backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                   &device_wide, "cuda.attention.complete.synchronize", err);
    if (rc != YVEX_OK)
        return completion_refuse(
            completion, YVEX_BACKEND_ATTENTION_FAILURE_SYNCHRONIZE,
            "cuda.attention.complete.synchronize", 1ull, 0ull,
            (yvex_status)rc, "CUDA attention completion barrier failed", err);
    completion->barrier_observed = 1;
    completion->output.queue_synchronizations +=
        (unsigned long long)(!barrier_observed && !device_wide);
    completion->output.device_synchronizations +=
        (unsigned long long)(!barrier_observed && device_wide);
    if (!completion->host_status || *completion->host_status != 0)
        return completion_refuse(
            completion, YVEX_BACKEND_ATTENTION_FAILURE_NUMERIC,
            "cuda.attention.complete.numeric", 0ull,
            completion->host_status ? (unsigned long long)*completion->host_status : ULLONG_MAX,
            YVEX_ERR_FORMAT,
            "CUDA attention device numerical stage refused its input", err);
    rc = completion_topk_publish(completion, err);
    if (rc == YVEX_OK) rc = completion_transfers_publish(completion, err);
    if (rc == YVEX_OK) rc = completion_account(backend, completion, err);
    if (rc != YVEX_OK) return rc;
    completion->pending = 0;
    memset(&completion->failure, 0, sizeof(completion->failure));
    yvex_error_clear(err);
    return YVEX_OK;
}
