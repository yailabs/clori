/*
 * Validate and launch CUDA graph primitives through admitted generated-kernel variants.
 *
 * Every op requires an exact admitted variant; launch and final synchronization must succeed
 * before any output is marked written. Bounded primitive execution is not transformer or model
 * runtime.
 */
#include "src/backend/cuda/private.h"
#include <yvex/internal/graph_state.h>
#include <yvex/internal/transformer.h>
#include <yvex/quant.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define CUDA_ATTENTION_BLOCK 256u
static int cuda_work_launch(yvex_cuda_work *work,
                            CUfunction function,
                            unsigned int grid,
                            unsigned int block,
                            unsigned int shared_bytes,
                            void **params,
                            const char *stage,
                            yvex_error *err)
{
    int rc;
    if (!work || !work->backend) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage, "CUDA work owner is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (work->prepare_only)
        return yvex_cuda_graph_kernel_update(
            work->backend, work->variant, function, grid, block, shared_bytes,
            params, stage, err);
    rc = yvex_cuda_launch(work->backend, work->variant, function, grid, block,
                          shared_bytes, params, stage, err);
    if (rc == YVEX_OK && !yvex_cuda_capture_active(work->backend))
        work->launches++;
    return rc;
}
static int attention_fail(yvex_backend_attention_failure *failure,
                             yvex_backend_attention_failure_code code,
                             const char *stage,
                             unsigned long long expected,
                             unsigned long long actual,
                             yvex_error *err,
                             yvex_status status,
                             const char *message)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->stage = stage;
        failure->expected = expected;
        failure->actual = actual;
    }
    if (!yvex_error_is_set(err)) yvex_error_set(err, status, stage, message);
    return status;
}
static int attention_account_transfer(
    unsigned long long count, size_t width, unsigned long long *total,
    const char *stage, yvex_backend_attention_failure *failure, yvex_error *err)
{
    size_t bytes;
    unsigned long long next;
    if (!total ||
        !yvex_cuda_work_checked_bytes(count, (unsigned long long)width, &bytes) ||
        !yvex_core_u64_add(*total, (unsigned long long)bytes, &next))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, stage, ULLONG_MAX,
            count, err, YVEX_ERR_BOUNDS,
            "CUDA attention transfer accounting overflowed");
    *total = next;
    return YVEX_OK;
}
/* Acquire one bounded, family-neutral attention transaction range. */
static int attention_allocate(yvex_cuda_work *work,
                                 CUdeviceptr *out,
                                 size_t bytes,
                                 const void *source,
                                 int zero,
                                 const char *stage,
                                 yvex_backend_attention_failure *failure,
                                 yvex_error *err)
{
    const char *injected = getenv("YVEX_TEST_CUDA_ATTENTION_FAILURE");
    yvex_cuda_work_failure work_failure = YVEX_CUDA_WORK_FAILURE_NONE;
    yvex_backend_attention_failure_code code;
    const char *message;
    int rc;
    if (injected && strcmp(injected, "allocation") == 0)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_ALLOCATION, stage, bytes,
            0ull, err, YVEX_ERR_NOMEM, "injected CUDA attention allocation failure");
    if (injected && source && strcmp(injected, "copy-input") == 0)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
            err, YVEX_ERR_BACKEND, "injected CUDA attention input copy failure");
    rc = yvex_cuda_work_allocate(work, out, bytes, source, zero, stage,
                                 &work_failure, err);
    if (rc == YVEX_OK) return YVEX_OK;
    code = work_failure == YVEX_CUDA_WORK_FAILURE_BUDGET
        ? YVEX_BACKEND_ATTENTION_FAILURE_BUDGET
        : work_failure == YVEX_CUDA_WORK_FAILURE_COPY
              ? YVEX_BACKEND_ATTENTION_FAILURE_COPY
              : YVEX_BACKEND_ATTENTION_FAILURE_ALLOCATION;
    message = code == YVEX_BACKEND_ATTENTION_FAILURE_BUDGET
        ? "CUDA attention reusable device budget is insufficient"
        : code == YVEX_BACKEND_ATTENTION_FAILURE_COPY
              ? "CUDA attention device initialization failed"
              : "CUDA attention device allocation failed";
    rc = attention_fail(failure, code, stage, bytes, 0ull, err,
                        (yvex_status)rc, message);
    if (code == YVEX_BACKEND_ATTENTION_FAILURE_BUDGET &&
        work->backend->workspace_device_tensor)
        yvex_error_setf(
            err, (yvex_status)rc, stage,
            "CUDA reusable workspace needs %zu bytes at cursor %llu of %llu",
            bytes, work->backend->workspace_cursor,
            work->backend->workspace_bytes);
    return rc;
}
static int attention_initialize(yvex_cuda_work *work, CUdeviceptr target,
                                   size_t bytes, const void *source, int zero,
                                   const char *stage,
                                   yvex_backend_attention_failure *failure,
                                   yvex_error *err)
{
    const char *injected = getenv("YVEX_TEST_CUDA_ATTENTION_FAILURE");
    int rc;
    if (!work || !work->backend)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
            err, YVEX_ERR_INVALID_ARG, "CUDA attention initialization is invalid");
    rc = backend_dispatch_admit(work->backend, stage, err);
    if (rc != YVEX_OK)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
            err, (yvex_status)rc, "CUDA attention backend is cleanup-only");
    if (injected && source && strcmp(injected, "copy-input") == 0)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
            err, YVEX_ERR_BACKEND, "injected CUDA attention input copy failure");
    rc = yvex_cuda_work_initialize(work, target, bytes, source, zero, stage, err);
    return rc == YVEX_OK ? YVEX_OK : attention_fail(
        failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull, err,
        (yvex_status)rc, "CUDA attention range initialization failed");
}
static int attention_download(yvex_cuda_work *work, void *target,
                                 CUdeviceptr source, size_t bytes,
                                 const char *stage,
                                 yvex_backend_attention_failure *failure,
                                 yvex_error *err)
{
    const char *injected = getenv("YVEX_TEST_CUDA_ATTENTION_FAILURE");
    CUstream stream;
    int rc;
    if (!work || !work->state || !target || !source || !bytes)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
            err, YVEX_ERR_INVALID_ARG, "CUDA attention download is invalid");
    rc = backend_dispatch_admit(work->backend, stage, err);
    if (rc != YVEX_OK)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
            err, (yvex_status)rc, "CUDA attention backend is cleanup-only");
    if (work->prepare_only) return YVEX_OK;
    if (injected && (strcmp(injected, "copy-output") == 0 ||
                     strcmp(injected, stage) == 0))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
            err, YVEX_ERR_BACKEND, "injected CUDA attention output copy failure");
    stream = yvex_cuda_launch_stream(work->backend);
    if (stream && !work->state->driver.cuMemcpyDtoHAsync_v2)
        rc = YVEX_ERR_UNSUPPORTED;
    else
        rc = yvex_cuda_status(
            &work->state->driver,
            stream ? work->state->driver.cuMemcpyDtoHAsync_v2(
                         target, source, bytes, stream)
                   : work->state->driver.cuMemcpyDtoH_v2(target, source, bytes),
            stage, err);
    return rc == YVEX_OK ? YVEX_OK : attention_fail(
        failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull, err,
        (yvex_status)rc, "CUDA attention staged output copy failed");
}
/* Launch only; the family transaction retains synchronization and publication. */
static int attention_launch(yvex_cuda_work *work,
                               CUfunction function,
                               unsigned int grid,
                               unsigned int block,
                               unsigned int shared_bytes,
                               void **params,
                               const char *stage,
                               yvex_backend_attention_failure *failure,
                               yvex_error *err)
{
    const char *injected = getenv("YVEX_TEST_CUDA_ATTENTION_FAILURE");
    int rc;
    if (!work || !work->backend)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_LAUNCH, stage, 1ull, 0ull,
            err, YVEX_ERR_INVALID_ARG, "CUDA attention launch is invalid");
    rc = backend_dispatch_admit(work->backend, stage, err);
    if (rc != YVEX_OK)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_LAUNCH, stage, 1ull, 0ull,
            err, (yvex_status)rc, "CUDA attention backend is cleanup-only");
    if (injected && strcmp(injected, stage) == 0)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_LAUNCH, stage, 1ull, 0ull,
            err, YVEX_ERR_BACKEND, "injected CUDA attention kernel launch failure");
    rc = cuda_work_launch(work, function, grid, block, shared_bytes,
                          params, stage, err);
    if (rc != YVEX_OK)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_LAUNCH, stage, 1ull, 0ull,
            err, (yvex_status)rc, "CUDA attention kernel launch failed");
    return YVEX_OK;
}
static int attention_round_bf16(
    yvex_cuda_work *work, CUdeviceptr values, unsigned long long count,
    CUdeviceptr status, const char *stage,
    yvex_backend_attention_failure *failure, yvex_error *err)
{
    unsigned int grid;
    if (!count || count > UINT_MAX * (unsigned long long)CUDA_ATTENTION_BLOCK)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            UINT_MAX, count, err, YVEX_ERR_BOUNDS,
            "CUDA attention BF16-round geometry is invalid");
    grid = (unsigned int)((count + CUDA_ATTENTION_BLOCK - 1ull) /
                          CUDA_ATTENTION_BLOCK);
    {
        void *params[] = {&values, &count, &status};
        return attention_launch(
            work, work->state->attention_bf16_round_function, grid,
            CUDA_ATTENTION_BLOCK, 0u, params, stage, failure, err);
    }
}
static int attention_matvec(yvex_cuda_work *work,
                               const yvex_backend_attention_weight *weight,
                               CUdeviceptr device_weight,
                               unsigned long long start_row,
                               unsigned long long rows,
                               unsigned long long input_rows,
                               CUdeviceptr vector,
                               CUdeviceptr out,
                               int output_bf16,
                               CUdeviceptr status,
                               const char *stage,
                               yvex_backend_attention_failure *failure,
                               yvex_error *err)
{
    CUdeviceptr additive = 0ull;
    int block_row = 0, q8_path, q8_input = 0;
    unsigned long long tensorcore_grid = 0ull;
    unsigned int matvec_grid, matvec_block;
    q8_path = weight && work->activation_q8 && !work->forensic_numeric &&
              weight->row_width % 256ull == 0ull &&
              yvex_cuda_q8_activation_eligible(weight->qtype) &&
              work->state->qtype_tensorcore_rows_function;
    if (!weight || !weight->present || !device_weight || !vector || !out ||
        !rows || !input_rows || start_row > weight->row_count ||
        rows > weight->row_count - start_row ||
        !yvex_cuda_qtype_matvec_geometry(
            rows, weight ? weight->row_width : 0ull, input_rows,
            weight ? weight->qtype : 0u, 1, &matvec_grid, &matvec_block,
            &block_row))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            weight ? weight->row_count : 0ull, start_row + rows, err,
            YVEX_ERR_BOUNDS, "CUDA attention matvec geometry is invalid");
    if (q8_path &&
        (!yvex_core_u64_mul((rows + 15ull) / 16ull,
                            (input_rows + 15ull) / 16ull, &tensorcore_grid) ||
         tensorcore_grid > UINT_MAX))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            UINT_MAX, tensorcore_grid, err, YVEX_ERR_BOUNDS,
            "CUDA Tensor Core attention grid exceeds launch bounds");
    if (q8_path) {
        unsigned long long blocks = weight->row_width / 256ull;
        unsigned long long quantize_tasks, quantized_bytes;
        CUdeviceptr quantized;
        int rc;
        if (!yvex_core_u64_mul(blocks, input_rows, &quantize_tasks) ||
            !yvex_core_u64_mul(quantize_tasks, 292ull, &quantized_bytes) ||
            quantize_tasks > UINT_MAX)
            return attention_fail(
                failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
                UINT_MAX, quantize_tasks, err, YVEX_ERR_BOUNDS,
                "CUDA Q8 activation geometry is invalid");
        rc = YVEX_OK;
        if (quantized_bytes > work->q8_capacity) {
            rc = yvex_cuda_work_allocate(work, &work->q8_input,
                                         (size_t)quantized_bytes, NULL, 0,
                                         "cuda.q8-activation", NULL, err);
            if (rc == YVEX_OK) work->q8_capacity = quantized_bytes;
        }
        quantized = work->q8_input;
        if (rc == YVEX_OK) {
            void *params[] = {&quantized, &vector, (void *)&weight->row_width,
                              &input_rows, &status};
            rc = attention_launch(work, work->state->q8_quantize_function,
                                  (unsigned int)quantize_tasks, CUDA_ATTENTION_BLOCK, 0u,
                                  params, "cuda.q8-activation", failure, err);
        }
        if (rc == YVEX_OK) {
            void *params[] = {
                &device_weight, (void *)&weight->row_bytes,
                (void *)&weight->row_width, &start_row, &rows, &input_rows,
                (void *)&weight->qtype, &quantized, &additive, &out,
                &output_bf16, &status};
            rc = attention_launch(
                work, work->state->qtype_tensorcore_rows_function,
                (unsigned int)tensorcore_grid, 32u, 0u, params, stage,
                failure, err);
            if (rc == YVEX_OK) work->tensor_core_launches++;
        }
        return rc;
    }
    {
        void *params[] = {&device_weight, (void *)&weight->row_bytes,
            (void *)&weight->row_width, &start_row, &rows,
            &input_rows, (void *)&weight->qtype, &vector, &q8_input,
            &block_row, &work->forensic_numeric, &additive, &out,
            &output_bf16, &status
        };
        return attention_launch(
            work, work->state->qtype_matvec_function,
            matvec_grid, matvec_block, 0u, params, stage, failure, err);
    }
}
static int attention_decode(yvex_cuda_work *work,
                               const yvex_backend_attention_weight *weight,
                               CUdeviceptr device_weight,
                               unsigned long long row,
                               unsigned long long count,
                               CUdeviceptr out,
                               CUdeviceptr status,
                               const char *stage,
                               yvex_backend_attention_failure *failure,
                               yvex_error *err)
{
    CUdeviceptr encoded;
    unsigned int grid;
    if (!weight || !weight->present || row >= weight->row_count ||
        count != weight->row_width ||
        count > UINT_MAX * (unsigned long long)CUDA_ATTENTION_BLOCK)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            weight ? weight->row_width : 0ull, count, err, YVEX_ERR_BOUNDS,
            "CUDA attention decode geometry is invalid");
    encoded = device_weight + row * weight->row_bytes;
    grid = (unsigned int)((count + CUDA_ATTENTION_BLOCK - 1ull) /
                          CUDA_ATTENTION_BLOCK);
    {
        void *params[] = {
            &encoded, &count, (void *)&weight->qtype, &out, &status
        };
        return attention_launch(
            work, work->state->encoded_row_decode_function, grid,
            CUDA_ATTENTION_BLOCK, 0u, params, stage, failure, err);
    }
}
static int attention_weighted_norm(
    yvex_cuda_work *work, CUdeviceptr values, unsigned long long count,
    unsigned long long vectors,
    const yvex_backend_attention_weight *weight, CUdeviceptr device_weight,
    double epsilon, CUdeviceptr status, const char *stage,
    yvex_backend_attention_failure *failure, yvex_error *err)
{
    if (!weight || !weight->present || !vectors || vectors > UINT_MAX ||
        weight->row_count != 1ull || weight->row_width != count)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            count, weight ? weight->row_width : 0ull, err, YVEX_ERR_FORMAT,
            "CUDA attention normalization weight shape is invalid");
    {
        void *params[] = {
            &values, &count, &device_weight, (void *)&weight->qtype,
            &epsilon, &vectors, &status
        };
        return attention_launch(
            work, work->state->attention_weighted_norm_function, (unsigned int)vectors,
            CUDA_ATTENTION_BLOCK, CUDA_ATTENTION_BLOCK * sizeof(double),
            params, stage, failure, err);
    }
}
static int attention_unit_norm(yvex_cuda_work *work,
                                  CUdeviceptr values,
                                  unsigned long long vectors,
                                  unsigned long long width,
                                  double epsilon,
                                  CUdeviceptr status,
                                  const char *stage,
                                  yvex_backend_attention_failure *failure,
                                  yvex_error *err)
{
    if (!vectors || vectors > UINT_MAX || !width)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            1ull, vectors, err, YVEX_ERR_BOUNDS,
            "CUDA attention unit-norm geometry is invalid");
    {
        void *params[] = {&values, &vectors, &width, &epsilon, &status};
        return attention_launch(
            work, work->state->attention_unit_norm_function,
            (unsigned int)vectors, CUDA_ATTENTION_BLOCK,
            CUDA_ATTENTION_BLOCK * (unsigned int)sizeof(double), params, stage,
            failure, err);
    }
}
static int attention_rope(yvex_cuda_work *work,
                             CUdeviceptr values,
                             unsigned long long vectors,
                             unsigned long long width,
                             unsigned long long token_position,
                             const yvex_backend_attention_position *position,
                             int inverse,
                             CUdeviceptr status,
                             const char *stage,
                             yvex_backend_attention_failure *failure,
                             yvex_error *err)
{
    unsigned long long total;
    unsigned int grid;
    if (!position || !position->rope_dimensions ||
        position->rope_dimensions > width ||
        vectors > ULLONG_MAX / (position->rope_dimensions / 2ull))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            width, position ? position->rope_dimensions : 0ull, err,
            YVEX_ERR_BOUNDS, "CUDA attention RoPE geometry is invalid");
    total = vectors * (position->rope_dimensions / 2ull);
    if (!total || total > UINT_MAX * (unsigned long long)CUDA_ATTENTION_BLOCK)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            UINT_MAX, total, err, YVEX_ERR_BOUNDS,
            "CUDA attention RoPE launch extent is invalid");
    grid = (unsigned int)((total + CUDA_ATTENTION_BLOCK - 1ull) /
                          CUDA_ATTENTION_BLOCK);
    {
        void *params[] = {
            &values, &vectors, &width, (void *)&position->rope_dimensions,
            &token_position, (void *)&position->theta,
            (void *)&position->scaling_factor,
            (void *)&position->original_context, (void *)&position->beta_fast,
            (void *)&position->beta_slow, &inverse, &status
        };
        return attention_launch(
            work, work->state->attention_yarn_rope_function, grid,
            CUDA_ATTENTION_BLOCK, 0u, params, stage, failure, err);
    }
}
static int attention_activation(
    yvex_cuda_work *work, CUdeviceptr values, unsigned long long vectors,
    unsigned long long width, const yvex_backend_attention_activation *policy,
    CUdeviceptr status, const char *stage,
    yvex_backend_attention_failure *failure, yvex_error *err)
{
    if (!policy || !policy->required) return YVEX_OK;
    if (!vectors || vectors > UINT_MAX || !width || !policy->block_width ||
        width % policy->block_width)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            policy->block_width, width, err, YVEX_ERR_BOUNDS,
            "CUDA attention activation geometry is invalid");
    {
        void *params[] = {
            &values, &vectors, &width, (void *)&policy->block_width,
            (void *)&policy->quantization, (void *)&policy->hadamard, &status
        };
        return attention_launch(
            work, work->state->attention_activation_quantize_function,
            (unsigned int)vectors, CUDA_ATTENTION_BLOCK, 0u, params, stage,
            failure, err);
    }
}
static int attention_validate_job(yvex_backend_attention_job *job,
                                  yvex_backend_attention_output *output,
                                  yvex_backend_attention_failure *failure,
                                  yvex_error *err)
{
    unsigned long long input_width =
        job && job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
            ? job->residual_expanded_width : job ? job->hidden_width : 0ull;
    if (job && !job->local_count && !job->local_stride)
        job->local_stride = job->head_dimension;
    if (job && job->attention_class != YVEX_BACKEND_ATTENTION_SWA &&
        !job->compressed_count && !job->compressed_stride)
        job->compressed_stride = job->head_dimension;
    if (job && job->attention_class == YVEX_BACKEND_ATTENTION_CSA &&
        !job->indexer_count && !job->indexer_stride)
        job->indexer_stride = job->indexer_head_dimension;
    if (!job || !output || job->schema != YVEX_BACKEND_ATTENTION_JOB_SCHEMA ||
        (!job->input && !job->device_input) || !job->token_count || !input_width ||
        job->input_stride < input_width ||
        job->token_position > ULLONG_MAX - job->token_count ||
        (job->phase == YVEX_BACKEND_ATTENTION_PHASE_DECODE &&
         job->token_count != 1ull) ||
        (job->phase != YVEX_BACKEND_ATTENTION_PHASE_DECODE &&
         job->phase != YVEX_BACKEND_ATTENTION_PHASE_PREFILL &&
         job->phase != YVEX_BACKEND_ATTENTION_PHASE_SPECULATIVE_DRAFT &&
         job->phase != YVEX_BACKEND_ATTENTION_PHASE_SPECULATIVE_VERIFY) ||
        (job->candidate_block_visible != 0 &&
         job->candidate_block_visible != 1) ||
        (job->candidate_block_visible &&
         (job->phase != YVEX_BACKEND_ATTENTION_PHASE_SPECULATIVE_DRAFT ||
          job->attention_class != YVEX_BACKEND_ATTENTION_SWA ||
          job->token_count < 2ull)) ||
        !job->hidden_width || !job->q_rank || !job->query_heads ||
        !job->head_dimension || !job->kv_width || !job->max_device_bytes ||
        job->query_heads > ULLONG_MAX / job->head_dimension ||
        job->query_heads * job->head_dimension >
            (unsigned long long)SIZE_MAX / sizeof(float) ||
        (job->cancellation && !job->cancellation->requested))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate", 1ull, 0ull, err, YVEX_ERR_INVALID_ARG,
            "CUDA attention job and output geometry are required");
    if (job->operation_scope != YVEX_BACKEND_ATTENTION_SCOPE_CORE &&
        job->operation_scope != YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.scope", YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE,
            job->operation_scope, err, YVEX_ERR_FORMAT,
            "CUDA attention operation scope is invalid");
    if (job->local_stride != job->head_dimension ||
        (job->local_count && (!job->local_kv || !job->local_positions)))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.local_history", job->head_dimension,
            job->local_stride, err, YVEX_ERR_FORMAT,
            "CUDA attention local history is incomplete");
    if (job->compressed_count &&
        (!job->compressed_kv || !job->compressed_positions ||
         job->compressed_stride != job->head_dimension))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.compressed_history", job->head_dimension,
            job->compressed_stride, err, YVEX_ERR_FORMAT,
            "CUDA attention compressed history is incomplete");
    if (job->compute_contract != YVEX_BACKEND_ATTENTION_COMPUTE_BF16_F32_RNE_V1)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.compute_contract",
            YVEX_BACKEND_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
            job->compute_contract, err, YVEX_ERR_UNSUPPORTED,
            "CUDA attention compute contract is unavailable");
    if (job->attention_class < YVEX_BACKEND_ATTENTION_SWA ||
        job->attention_class > YVEX_BACKEND_ATTENTION_HCA)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.class", YVEX_BACKEND_ATTENTION_HCA,
            job->attention_class, err, YVEX_ERR_FORMAT,
            "CUDA attention class is not admitted");
    if (job->attention_class == YVEX_BACKEND_ATTENTION_CSA &&
        job->indexer_count != job->compressed_count)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.index_history", job->compressed_count,
            job->indexer_count, err, YVEX_ERR_FORMAT,
            "CUDA CSA indexer/compressed history cardinality differs");
    if (job->evidence_level > 3u)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.evidence", 3ull, job->evidence_level,
            err, YVEX_ERR_FORMAT, "CUDA attention evidence level is invalid");
    if (job->native_execution != 0 && job->native_execution != 1)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.native_execution", 1ull,
            (unsigned long long)job->native_execution, err, YVEX_ERR_FORMAT,
            "CUDA attention native-execution admission is invalid");
    return YVEX_OK;
}
static int attention_validate_weight(const yvex_backend_attention_weight *weight,
                                     unsigned long long rows,
                                     unsigned long long width,
                                     yvex_backend_attention_failure *failure,
                                     yvex_error *err)
{
    const yvex_quant_numeric_capability *capability;
    unsigned long long row_bytes = 0ull, total_bytes = 0ull;
    const char *reason = NULL;
    if (!weight || !weight->present || !weight->encoded ||
        weight->row_count != rows || weight->row_width != width)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.weight_shape", rows,
            weight ? weight->row_count : 0ull, err, YVEX_ERR_FORMAT,
            "CUDA attention encoded weight shape is invalid");
    capability = yvex_quant_numeric_capability_at(weight->qtype);
    if (!capability || !capability->dedicated_cuda_compute_available ||
        !yvex_gguf_qtype_storage_bytes(weight->qtype, width, &row_bytes, &reason) ||
        row_bytes != weight->row_bytes ||
        !yvex_core_u64_mul(rows, row_bytes, &total_bytes) ||
        total_bytes != (unsigned long long)weight->encoded_bytes)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.attention.validate.weight_encoding", row_bytes,
            weight->row_bytes, err, YVEX_ERR_UNSUPPORTED,
            reason ? reason : "CUDA attention encoded weight capability is unavailable");
    return YVEX_OK;
}
static int attention_validate_activation(
    const yvex_backend_attention_activation *policy, unsigned long long width,
    const char *stage, yvex_backend_attention_failure *failure, yvex_error *err)
{
    if (!policy || !policy->required) return YVEX_OK;
    if (!width || !policy->block_width || width % policy->block_width ||
        (policy->quantization != 1u && policy->quantization != 2u) ||
        (policy->hadamard &&
         ((width & (width - 1ull)) != 0ull || width > 1024ull)))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            policy->block_width, width, err, YVEX_ERR_FORMAT,
            "CUDA attention activation policy and width are incompatible");
    return YVEX_OK;
}
static int attention_validate_rolling(
    const yvex_backend_attention_job *job,
    const yvex_backend_attention_rolling *rolling, unsigned long long ratio,
    unsigned long long head_dimension, int overlap, unsigned long long *extent,
    const char *stage, yvex_backend_attention_failure *failure, yvex_error *err)
{
    unsigned long long factor = overlap ? 2ull : 1ull, state_width, state_slots;
    if (!job || !rolling || !extent ||
        !yvex_core_u64_mul(head_dimension, factor, &state_width) ||
        !yvex_core_u64_mul(ratio, factor, &state_slots) ||
        !yvex_core_u64_mul(state_width, state_slots, extent) ||
        !rolling->present || rolling->next_token_position != job->token_position ||
        rolling->ratio != ratio || rolling->head_dimension != head_dimension ||
        rolling->state_width != state_width || rolling->state_slots != state_slots ||
        rolling->overlap != overlap || rolling->cursor != job->token_position % ratio ||
        rolling->current_fill != job->token_position % ratio ||
        rolling->previous_fill !=
            (overlap && job->token_position >= ratio ? ratio : 0ull) ||
        !rolling->kv_state || rolling->kv_state_capacity < *extent ||
        !rolling->score_state || rolling->score_state_capacity < *extent)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            ratio, rolling ? rolling->ratio : 0ull, err, YVEX_ERR_FORMAT,
            "CUDA attention rolling-state geometry is invalid");
    return YVEX_OK;
}
static int attention_spans_disjoint(const yvex_cuda_host_span *writes,
                                    size_t write_count,
                                    const yvex_cuda_host_span *reads,
                                    size_t read_count)
{
    uintptr_t write_first, write_last, other_first, other_last;
    size_t i, j, bytes;
    for (i = 0u; i < write_count; ++i) {
        if (!writes[i].data ||
            !yvex_cuda_work_checked_bytes(writes[i].count, writes[i].width, &bytes) ||
            (uintptr_t)writes[i].data > UINTPTR_MAX - bytes)
            return 0;
        write_first = (uintptr_t)writes[i].data;
        write_last = write_first + bytes;
        for (j = i + 1u; j < write_count; ++j) {
            if (!writes[j].data ||
                !yvex_cuda_work_checked_bytes(writes[j].count, writes[j].width, &bytes) ||
                (uintptr_t)writes[j].data > UINTPTR_MAX - bytes)
                return 0;
            other_first = (uintptr_t)writes[j].data;
            other_last = other_first + bytes;
            if (write_first < other_last && other_first < write_last) return -1;
        }
        for (j = 0u; j < read_count; ++j) {
            if (!reads[j].count) continue;
            if (!reads[j].data ||
                !yvex_cuda_work_checked_bytes(reads[j].count, reads[j].width, &bytes) ||
                (uintptr_t)reads[j].data > UINTPTR_MAX - bytes)
                return 0;
            other_first = (uintptr_t)reads[j].data;
            other_last = other_first + bytes;
            if (write_first < other_last && other_first < write_last) return -1;
        }
    }
    return 1;
}
static int attention_validate_alias(
    const yvex_backend_attention_job *job,
    const yvex_cuda_attention_transfer *transfers, size_t transfer_count,
    unsigned long long local_extent, unsigned long long compressed_extent,
    unsigned long long index_extent, unsigned long long main_rolling_extent,
    unsigned long long index_rolling_extent)
{
    yvex_cuda_host_span writes[YVEX_CUDA_WORK_MAX_RANGES], reads[40];
    unsigned long long input_width, input_count;
    size_t i, read_count = 0u;
    if (!job || transfer_count > YVEX_CUDA_WORK_MAX_RANGES) return 0;
    input_width = job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
                      ? job->residual_expanded_width : job->hidden_width;
    if (!yvex_core_u64_mul(job->token_count - 1ull, job->input_stride, &input_count) ||
        !yvex_core_u64_add(input_count, input_width, &input_count))
        return 0;
    for (i = 0u; i < transfer_count; ++i)
        writes[i] = (yvex_cuda_host_span){transfers[i].output,
                                         transfers[i].output_capacity,
                                         transfers[i].width};
#define READ(data_, count_, width_) \
    (reads[read_count++] = (yvex_cuda_host_span){(data_), (count_), (width_)})
    if (!job->device_input) READ(job->input, input_count, sizeof(float));
    READ(job->local_kv, local_extent, sizeof(float));
    READ(job->local_positions, job->local_count, sizeof(unsigned long long));
    READ(job->compressed_kv, compressed_extent, sizeof(float));
    READ(job->compressed_positions, job->compressed_count, sizeof(unsigned long long));
    READ(job->indexer_kv, index_extent, sizeof(float));
    READ(job->indexer_positions, job->indexer_count, sizeof(unsigned long long));
    READ(job->main_rolling.kv_state, main_rolling_extent, sizeof(float));
    READ(job->main_rolling.score_state, main_rolling_extent, sizeof(float));
    READ(job->indexer_rolling.kv_state, index_rolling_extent, sizeof(float));
    READ(job->indexer_rolling.score_state, index_rolling_extent, sizeof(float));
    for (i = 0u; i < YVEX_BACKEND_ATTENTION_WEIGHT_COUNT; ++i)
        READ(job->weights[i].encoded,
             job->weights[i].present ? job->weights[i].encoded_bytes : 0u,
             sizeof(unsigned char));
#undef READ
    return attention_spans_disjoint(writes, transfer_count, reads, read_count);
}
static int attention_cancel(yvex_backend *backend,
                            const yvex_backend_attention_job *job,
                            const char *stage, int pending,
                            yvex_backend_attention_failure *failure,
                            yvex_error *err)
{
    int rc;
    if (!job || !job->cancellation ||
        !job->cancellation->requested(job->cancellation->context))
        return YVEX_OK;
    if (pending) {
        rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, stage, err);
        if (rc != YVEX_OK)
            return attention_fail(
                failure, YVEX_BACKEND_ATTENTION_FAILURE_SYNCHRONIZE, stage,
                1ull, 0ull, err, (yvex_status)rc,
                "CUDA attention cancellation synchronization failed");
    }
    return attention_fail(
        failure, YVEX_BACKEND_ATTENTION_FAILURE_CANCELLED, stage, 0ull, 1ull,
        err, YVEX_ERR_CANCELLED,
        "CUDA attention execution was cancelled before publication");
}
/*
 * Acquire one exact reusable host-staging span for attention execution.
 *
 * Generic staging ownership; family layout remains caller-defined.
 */
static int attention_stage_acquire(
    yvex_backend *backend, size_t bytes, int require_pinned, int injected,
    unsigned char **out, int *reused, yvex_backend_attention_failure *failure,
    yvex_error *err)
{
    yvex_backend_host_workspace_summary summary;
    int acquired;
    if (!backend || !bytes || !out || !reused || injected)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_ALLOCATION,
            "cuda.attention.host_stage", bytes, 0ull, err,
            injected ? YVEX_ERR_NOMEM : YVEX_ERR_INVALID_ARG,
            "CUDA attention host-staging acquisition failed");
    if (!yvex_backend_host_workspace_summary_get(backend, &summary))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_ALLOCATION,
            "cuda.attention.host_stage", bytes, 0ull, err, YVEX_ERR_STATE,
            "CUDA attention host workspace state is unavailable");
    *reused = summary.peak != 0ull;
    acquired = yvex_backend_host_workspace_acquire(
        backend, bytes, 8ull, (void **)out);
    if (acquired != YVEX_BACKEND_RESIDENT_HIT || !*out)
        return attention_fail(
            failure, acquired == YVEX_BACKEND_RESIDENT_MISS
                         ? YVEX_BACKEND_ATTENTION_FAILURE_BUDGET
                         : YVEX_BACKEND_ATTENTION_FAILURE_ALLOCATION,
            "cuda.attention.host_stage", bytes, summary.capacity, err,
            acquired == YVEX_BACKEND_RESIDENT_INVALID ? YVEX_ERR_BOUNDS
                                                      : YVEX_ERR_NOMEM,
            "CUDA attention host workspace capacity is insufficient");
    if (require_pinned &&
        (!yvex_backend_host_workspace_summary_get(backend, &summary) ||
         !summary.pinned))
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.attention.host_stage.pinned", 1ull, 0ull, err,
            YVEX_ERR_UNSUPPORTED,
            "CUDA graph attention requires page-locked stable host staging");
    return YVEX_OK;
}

static int attention_stage_range(unsigned char *base, size_t *cursor,
                                 unsigned long long count, size_t width, void **out)
{
    size_t aligned, bytes;
    if (!cursor || !out) return 0;
    *out = NULL;
    if (!count) return 1;
    if (*cursor > SIZE_MAX - 7u ||
        !yvex_cuda_work_checked_bytes(count, (unsigned long long)width, &bytes))
        return 0;
    aligned = (*cursor + 7u) & ~(size_t)7u;
    if (aligned > SIZE_MAX - bytes) return 0;
    if (base) *out = base + aligned;
    *cursor = aligned + bytes;
    return 1;
}

static int attention_stage_layout(
    unsigned char *base, yvex_cuda_attention_upload *uploads, size_t upload_count,
    yvex_cuda_attention_transfer *transfers, size_t transfer_count,
    unsigned long long csa_tokens, int **status, unsigned long long **selected,
    unsigned long long **candidates, size_t *total, size_t *download_total)
{
    size_t cursor = 0u, i;
    if (!uploads || !transfers || !status || !selected || !candidates || !total ||
        !download_total)
        return 0;
    if (!attention_stage_range(base, &cursor, 1ull, sizeof(int), (void **)status) ||
        !attention_stage_range(base, &cursor, csa_tokens, sizeof(**selected),
                               (void **)selected) ||
        !attention_stage_range(base, &cursor, csa_tokens, sizeof(**candidates),
                               (void **)candidates))
        return 0;
    for (i = 0u; i < transfer_count; ++i)
        if (!attention_stage_range(base, &cursor, transfers[i].capacity,
                                   transfers[i].width, &transfers[i].staged))
            return 0;
    *download_total = cursor;
    for (i = 0u; i < upload_count; ++i)
        if (!attention_stage_range(base, &cursor, uploads[i].count,
                                   uploads[i].width, &uploads[i].staged))
            return 0;
    *total = cursor;
    return 1;
}

typedef struct {
    const void *host;
    CUdeviceptr source, target;
    unsigned long long count;
    size_t width;
} cuda_state_span;

static int cuda_state_source_offset(CUdeviceptr base, unsigned long long count,
                                    size_t width, CUdeviceptr *out)
{
    unsigned long long bytes;
    if (!out || !yvex_core_u64_mul(count, (unsigned long long)width, &bytes) ||
        base > ULLONG_MAX - bytes)
        return 0;
    *out = base + bytes;
    return 1;
}

/* Publish complete non-prefix state into the already admitted candidate bank. */
static int attention_state_stage(
    yvex_backend *backend, const yvex_backend_attention_job *job,
    const yvex_cuda_attention_state_sources *sources,
    size_t *copied_bytes, int *staged, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    cuda_state_span spans[10];
    CUstream stream;
    unsigned long long total_local, local_count, local_offset;
    unsigned long long compressed_count, indexer_count;
    CUdeviceptr local, local_positions, main_kv, main_score, index_kv, index_score;
    size_t bytes = 0u, index;
    unsigned int hits = 0u, misses = 0u;
    int rc;
    if (copied_bytes) *copied_bytes = 0u;
    if (staged) *staged = 0;
    if (!backend || !job || !sources || !copied_bytes || !staged || !state) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.attention.state.stage",
                       "CUDA attention state publication arguments are incomplete");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_core_u64_add(sources->initial_local, job->token_count, &total_local) ||
        !yvex_core_u64_add(sources->initial_compressed,
                           sources->emitted_compressed, &compressed_count) ||
        !yvex_core_u64_add(sources->initial_indexer,
                           sources->emitted_indexer, &indexer_count) ||
        job->local_stride > SIZE_MAX / sizeof(float) ||
        job->compressed_stride > SIZE_MAX / sizeof(float) ||
        job->indexer_stride > SIZE_MAX / sizeof(float)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.attention.state.stage",
                       "CUDA attention state publication extent overflowed");
        return YVEX_ERR_BOUNDS;
    }
    local_count = total_local < sources->local_capacity
                      ? total_local : sources->local_capacity;
    local_offset = total_local - local_count;
    if (!cuda_state_source_offset(
            sources->local, local_offset,
            sizeof(float) * (size_t)job->local_stride, &local) ||
        !cuda_state_source_offset(
            sources->local_positions, local_offset,
            sizeof(unsigned long long), &local_positions) ||
        !cuda_state_source_offset(
            sources->main_kv, job->token_count,
            sizeof(float) * (size_t)sources->main_extent, &main_kv) ||
        !cuda_state_source_offset(
            sources->main_score, job->token_count,
            sizeof(float) * (size_t)sources->main_extent, &main_score) ||
        !cuda_state_source_offset(
            sources->index_kv, job->token_count,
            sizeof(float) * (size_t)sources->index_extent, &index_kv) ||
        !cuda_state_source_offset(
            sources->index_score, job->token_count,
            sizeof(float) * (size_t)sources->index_extent, &index_score)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.attention.state.stage",
                       "CUDA attention state publication extent overflowed");
        return YVEX_ERR_BOUNDS;
    }
    spans[0] = (cuda_state_span){job->local_kv, local, 0ull, local_count,
                                 sizeof(float) * (size_t)job->local_stride};
    spans[1] = (cuda_state_span){job->local_positions, local_positions, 0ull,
                                 local_count, sizeof(unsigned long long)};
    spans[2] = (cuda_state_span){job->compressed_kv, sources->compressed, 0ull,
                                 compressed_count,
                                 sizeof(float) * (size_t)job->compressed_stride};
    spans[3] = (cuda_state_span){job->compressed_positions,
                                 sources->compressed_positions, 0ull,
                                 compressed_count, sizeof(unsigned long long)};
    spans[4] = (cuda_state_span){job->indexer_kv, sources->indexer, 0ull,
                                 indexer_count,
                                 sizeof(float) * (size_t)job->indexer_stride};
    spans[5] = (cuda_state_span){job->indexer_positions, sources->indexer_positions, 0ull,
                                 indexer_count, sizeof(unsigned long long)};
    spans[6] = (cuda_state_span){job->main_rolling.kv_state, main_kv, 0ull,
                                 job->main_rolling.kv_state_capacity, sizeof(float)};
    spans[7] = (cuda_state_span){job->main_rolling.score_state, main_score, 0ull,
                                 job->main_rolling.score_state_capacity, sizeof(float)};
    spans[8] = (cuda_state_span){job->indexer_rolling.kv_state, index_kv, 0ull,
                                 job->indexer_rolling.kv_state_capacity, sizeof(float)};
    spans[9] = (cuda_state_span){job->indexer_rolling.score_state, index_score, 0ull,
                                 job->indexer_rolling.score_state_capacity, sizeof(float)};
    for (index = 0u; index < sizeof(spans) / sizeof(spans[0]); ++index) {
        size_t span_bytes;
        unsigned long long target = 0ull;
        int resolved;
        if (!spans[index].count) continue;
        if (!spans[index].host || !spans[index].source ||
            !yvex_cuda_work_checked_bytes(
                spans[index].count, spans[index].width, &span_bytes)) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.attention.state.stage",
                           "CUDA attention state span is invalid");
            return YVEX_ERR_BOUNDS;
        }
        resolved = yvex_backend_state_residency_resolve(
            backend, spans[index].host, span_bytes, &target);
        if (resolved == YVEX_BACKEND_RESIDENT_INVALID) {
            yvex_error_set(err, YVEX_ERR_STATE, "cuda.attention.state.stage",
                           "CUDA attention state residency is invalidated");
            return YVEX_ERR_STATE;
        }
        if (resolved == YVEX_BACKEND_RESIDENT_HIT) ++hits;
        else ++misses;
        spans[index].target = (CUdeviceptr)target;
        if (span_bytes > SIZE_MAX - bytes) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.attention.state.stage",
                           "CUDA attention state byte accounting overflowed");
            return YVEX_ERR_BOUNDS;
        }
        bytes += span_bytes;
    }
    if (!hits) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (misses) {
        yvex_error_set(err, YVEX_ERR_STATE, "cuda.attention.state.stage",
                       "CUDA attention state residency resolved only a partial publication");
        return YVEX_ERR_STATE;
    }
    rc = yvex_cuda_set_current(backend, "cuda.attention.state.stage", err);
    if (rc != YVEX_OK) return rc;
    stream = yvex_cuda_launch_stream(backend);
    for (index = 0u; index < sizeof(spans) / sizeof(spans[0]); ++index) {
        size_t span_bytes = 0u;
        CUresult copied;
        if (!spans[index].count) continue;
        if (!yvex_cuda_work_checked_bytes(
                spans[index].count, spans[index].width, &span_bytes)) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.attention.state.stage",
                           "CUDA attention state span drifted after admission");
            return YVEX_ERR_BOUNDS;
        }
        copied = stream && state->driver.cuMemcpyDtoDAsync_v2
                     ? state->driver.cuMemcpyDtoDAsync_v2(
                           spans[index].target, spans[index].source,
                           span_bytes, stream)
                     : !stream ? state->driver.cuMemcpyDtoD_v2(
                           spans[index].target, spans[index].source, span_bytes)
                               : (CUresult)1;
        rc = yvex_cuda_status(&state->driver, copied,
                              "cuda.attention.state.stage", err);
        if (rc != YVEX_OK) return rc;
    }
    *copied_bytes = bytes;
    *staged = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Expose the single private encoded-attention operation boundary.
 *
 * None; returns immutable process-lifetime methods.
 */
const yvex_cuda_attention_operations *yvex_cuda_attention_operations_get(void)
{
    static const yvex_cuda_attention_operations operations = {
        attention_fail, attention_account_transfer, attention_validate_job,
        attention_validate_weight,
        attention_validate_activation, attention_validate_rolling,
        attention_validate_alias, attention_cancel, attention_stage_acquire,
        attention_stage_layout,
        attention_allocate, attention_initialize, attention_download,
        attention_launch, attention_round_bf16, attention_matvec, attention_decode,
        attention_weighted_norm, attention_unit_norm, attention_rope,
        attention_activation, attention_state_stage
    };
    return &operations;
}

static int cuda_grid_1d(unsigned long long elements,
                        unsigned int block_size,
                        unsigned int *out,
                        const char *where,
                        yvex_error *err)
{
    unsigned long long blocks;
    if (!out || block_size == 0u || elements == 0ull) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where,
                       "CUDA launch extent and block size must be non-zero");
        return YVEX_ERR_INVALID_ARG;
    }
    blocks = ((elements - 1ull) / (unsigned long long)block_size) + 1ull;
    if (blocks > (unsigned long long)UINT_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, where,
                       "CUDA grid dimension exceeds Driver API range");
        return YVEX_ERR_BOUNDS;
    }
    *out = (unsigned int)blocks;
    return YVEX_OK;
}

int yvex_cuda_op_embed(yvex_backend *backend,
                       const yvex_device_tensor *embedding,
                       const unsigned int *token_ids,
                       unsigned long long token_count,
                       yvex_device_tensor *out,
                       yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work;
    yvex_cuda_work_failure work_failure;
    CUdeviceptr token_ids_device = 0;
    CUdeviceptr embedding_ptr;
    CUdeviceptr out_ptr;
    unsigned long long hidden_size;
    unsigned long long vocab_size;
    unsigned long long total_elements;
    size_t token_bytes;
    unsigned int block_size = 128;
    unsigned int grid_size;
    void *params[6];
    yvex_backend_operation_variant variant;
    yvex_error cleanup_error;
    int cleanup_rc;
    int rc;
    memset(&work, 0, sizeof(work));
    if (!state) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_embed",
                       "CUDA backend state is missing");
        return YVEX_ERR_STATE;
    }
    if (!backend_tensor_owner_is(backend, embedding) ||
        !backend_tensor_owner_is(backend, out)) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_embed",
                       "embedding and output tensors must belong to this backend");
        return YVEX_ERR_STATE;
    }
    if ((embedding->dtype != YVEX_DTYPE_F32 && embedding->dtype != YVEX_DTYPE_F16) ||
        out->dtype != YVEX_DTYPE_F32) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "yvex_backend_op_embed",
                       "CUDA backend embed supports F32 and F16 embeddings with F32 output");
        return YVEX_ERR_UNSUPPORTED;
    }
    variant = embedding->dtype == YVEX_DTYPE_F16
                  ? YVEX_BACKEND_VARIANT_EMBED_F16_TO_F32
                  : YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32;
    out->is_written = 0;
    rc = yvex_cuda_require_capability(backend, variant,
                                      "yvex_backend_op_embed", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_backend_validate_embed(
        backend, embedding, token_ids, token_count, out, &hidden_size, &vocab_size,
        "CUDA backend embed supports F32 and F16 embeddings with F32 output",
        "yvex_backend_op_embed", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    total_elements = token_count * hidden_size;
    if (token_count > (unsigned long long)(SIZE_MAX / sizeof(unsigned int))) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_backend_op_embed",
                       "token id buffer is too large");
        return YVEX_ERR_BOUNDS;
    }
    rc = yvex_cuda_set_current(backend, "yvex_backend_op_embed", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    token_bytes = (size_t)(token_count * sizeof(unsigned int));
    work.backend = backend;
    work.state = state;
    work.variant = variant;
    rc = yvex_cuda_work_allocate(&work, &token_ids_device, token_bytes, token_ids, 0,
                                 "cuda.embed.token_alloc", &work_failure, err);
    if (rc != YVEX_OK)
        goto cleanup;
    rc = cuda_grid_1d(total_elements, block_size, &grid_size,
                      "cuda.embed.grid", err);
    if (rc != YVEX_OK)
        goto cleanup;
    embedding_ptr = yvex_cuda_tensor_ptr(embedding);
    out_ptr = yvex_cuda_tensor_ptr(out);
    params[0] = &embedding_ptr;
    params[1] = &token_ids_device;
    params[2] = &out_ptr;
    params[3] = &hidden_size;
    params[4] = &vocab_size;
    params[5] = &token_count;
    rc = yvex_cuda_launch(backend, variant,
                          embedding->dtype == YVEX_DTYPE_F16
                              ? state->embed_f16_function : state->embed_function,
                          grid_size, block_size, 0, params,
                          "cuda.embed.launch", err);
    if (rc == YVEX_OK) {
        rc = yvex_cuda_synchronize(backend, variant, "cuda.embed.sync", err);
    }
cleanup:
    yvex_error_clear(&cleanup_error);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup_error);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        if (err)
            *err = cleanup_error;
        return cleanup_rc;
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    out->is_written = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Lower one sealed family-neutral workspace recipe to a checked byte extent.
 *
 * Pointer-free semantic components with explicit alignment and token scaling. Malformed identity
 * or arithmetic overflow leaves required bytes zero. Backend owns alignment lowering, while
 * graph/family owners select components.
 */
int yvex_backend_attention_workspace_required_from_recipe(
    const struct yvex_attention_workspace_recipe *recipe,
    unsigned long long *required_bytes, yvex_error *err)
{
    yvex_attention_workspace_recipe candidate;
    unsigned long long cursor = 0ull;
    unsigned int index;
    if (required_bytes) *required_bytes = 0ull;
    if (!recipe || !required_bytes || !yvex_sha256_hex_valid(recipe->identity)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.attention.workspace",
                       "one sealed attention workspace recipe is required");
        return YVEX_ERR_INVALID_ARG;
    }
    candidate = *recipe;
    if (yvex_attention_workspace_recipe_seal(&candidate, err) != YVEX_OK)
        return err ? yvex_error_code(err) : YVEX_ERR_FORMAT;
    if (strcmp(candidate.identity, recipe->identity) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "cuda.attention.workspace",
                       "attention workspace recipe identity is stale");
        return YVEX_ERR_STATE;
    }
    for (index = 0u; index < recipe->component_count; ++index) {
        const yvex_attention_workspace_component *component = &recipe->components[index];
        unsigned long long count = component->element_count, bytes, aligned;
        unsigned long long scale =
            component->scales_with_tokens ? recipe->token_capacity : 1ull;
        unsigned long long mask = component->alignment - 1ull;
        if (!component->scales_with_tokens &&
            component->kind >= YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_VALUES &&
                 component->kind <= YVEX_ATTENTION_WORKSPACE_INDEXER_ROLLING_SCORES &&
                 !yvex_core_u64_add(recipe->token_capacity, 1ull, &scale))
            goto overflow;
        if (!yvex_core_u64_mul(count, scale, &count) ||
            !yvex_core_u64_mul(count, component->element_width, &bytes) ||
            (component->lifetime != YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE &&
             !yvex_core_u64_add(bytes, bytes, &bytes)) ||
            cursor > ULLONG_MAX - mask) goto overflow;
        aligned = (cursor + mask) & ~mask;
        if (aligned > ULLONG_MAX - bytes) goto overflow;
        cursor = aligned + bytes;
    }
    *required_bytes = cursor;
    yvex_error_clear(err);
    return YVEX_OK;
overflow:
    yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.attention.workspace",
                   "attention workspace recipe overflowed backend address space");
    return YVEX_ERR_BOUNDS;
}

int yvex_cuda_op_rms_norm(yvex_backend *backend,
                          const yvex_device_tensor *input,
                          const yvex_device_tensor *weight,
                          float epsilon,
                          yvex_device_tensor *out,
                          yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr;
    CUdeviceptr weight_ptr;
    CUdeviceptr out_ptr;
    unsigned long long hidden_size;
    unsigned long long row_count;
    unsigned int block_size = 256u;
    unsigned int shared_bytes = block_size * (unsigned int)sizeof(float);
    void *params[6];
    yvex_backend_operation_variant variant;
    int rc;
    if (!state) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_rms_norm",
                       "CUDA backend state is missing");
        return YVEX_ERR_STATE;
    }
    rc = yvex_backend_validate_rms_norm(
        backend, input, weight, epsilon, out, &hidden_size,
        "CUDA RMSNorm supports F32 input/output with F16 or F32 weight",
        "yvex_backend_op_rms_norm", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    row_count = input->rank == 2 ? input->dims[0] : 1ull;
    if (row_count > UINT_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_backend_op_rms_norm",
                       "CUDA RMSNorm row count exceeds launch geometry");
        return YVEX_ERR_BOUNDS;
    }
    variant = weight->dtype == YVEX_DTYPE_F16
                  ? YVEX_BACKEND_VARIANT_RMS_NORM_F32_WEIGHT_F16
                  : YVEX_BACKEND_VARIANT_RMS_NORM_F32_WEIGHT_F32;
    out->is_written = 0;
    rc = yvex_cuda_require_capability(backend, variant,
                                      "yvex_backend_op_rms_norm", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_set_current(backend, "yvex_backend_op_rms_norm", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    input_ptr = yvex_cuda_tensor_ptr(input);
    weight_ptr = yvex_cuda_tensor_ptr(weight);
    out_ptr = yvex_cuda_tensor_ptr(out);
    params[0] = &input_ptr;
    params[1] = &weight_ptr;
    params[2] = &out_ptr;
    params[3] = &hidden_size;
    params[4] = &row_count;
    params[5] = &epsilon;
    rc = yvex_cuda_launch(backend, variant,
                          weight->dtype == YVEX_DTYPE_F16
                              ? state->rms_norm_f16_function
                              : state->rms_norm_f32_function,
                          (unsigned int)row_count, block_size, shared_bytes, params,
                          "cuda.rms_norm.launch", err);
    if (rc == YVEX_OK) {
        rc = yvex_cuda_synchronize(backend, variant, "cuda.rms_norm.sync", err);
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    out->is_written = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_op_rope(yvex_backend *backend,
                      const yvex_device_tensor *input,
                      unsigned long long position,
                      float rope_base,
                      yvex_device_tensor *out,
                      yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr;
    CUdeviceptr out_ptr;
    unsigned long long head_dim;
    unsigned long long pair_count;
    unsigned int block_size = 128u;
    unsigned int grid_size;
    float inverse_root;
    void *params[5];
    int rc;
    if (!state) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_rope",
                       "CUDA backend state is missing");
        return YVEX_ERR_STATE;
    }
    if (!backend_tensor_owner_is(backend, input) ||
        !backend_tensor_owner_is(backend, out)) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_rope",
                       "input and output tensors must belong to this backend");
        return YVEX_ERR_STATE;
    }
    if (input->dtype != YVEX_DTYPE_F32 || out->dtype != YVEX_DTYPE_F32) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "yvex_backend_op_rope",
                       "CUDA RoPE supports F32 input/output");
        return YVEX_ERR_UNSUPPORTED;
    }
    out->is_written = 0;
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_ROPE_F32,
                                      "yvex_backend_op_rope", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    if (!yvex_backend_tensor_same_shape(input, out)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_backend_op_rope",
                       "RoPE output shape must match input shape");
        return YVEX_ERR_FORMAT;
    }
    if (!isfinite(rope_base) || rope_base <= 1.0f) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_backend_op_rope",
                       "rope_base must be finite and greater than 1");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_backend_validate_rope(input, &head_dim, "yvex_backend_op_rope", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    if (!backend_tensor_f32_elements(input, head_dim) ||
        !backend_tensor_f32_elements(out, head_dim)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_backend_op_rope",
                       "RoPE input/output bytes must match F32 head_dim");
        return YVEX_ERR_BOUNDS;
    }
    rc = yvex_cuda_set_current(backend, "yvex_backend_op_rope", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    pair_count = head_dim / 2ull;
    inverse_root = (float)(1.0 / yvex_backend_nth_root((double)rope_base, pair_count));
    rc = cuda_grid_1d(pair_count, block_size, &grid_size, "cuda.rope.grid", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    input_ptr = yvex_cuda_tensor_ptr(input);
    out_ptr = yvex_cuda_tensor_ptr(out);
    params[0] = &input_ptr;
    params[1] = &out_ptr;
    params[2] = &head_dim;
    params[3] = &position;
    params[4] = &inverse_root;
    rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ROPE_F32,
                          state->rope_function, grid_size, block_size, 0,
                          params, "cuda.rope.launch", err);
    if (rc == YVEX_OK) {
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_ROPE_F32,
                                   "cuda.rope.sync", err);
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    out->is_written = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_op_matmul(yvex_backend *backend,
                        const yvex_device_tensor *input,
                        const yvex_device_tensor *weight,
                        yvex_device_tensor *out,
                        yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr;
    CUdeviceptr weight_ptr;
    CUdeviceptr out_ptr;
    unsigned long long m;
    unsigned long long k;
    unsigned long long n;
    unsigned long long output_elements;
    unsigned int block_size = 128u;
    unsigned int grid_size;
    void *params[6];
    int rc;
    if (!state) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_matmul",
                       "CUDA backend state is missing");
        return YVEX_ERR_STATE;
    }
    rc = yvex_backend_validate_matmul(backend, input, weight, out, &m, &k, &n,
                                      "yvex_backend_op_matmul", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    out->is_written = 0;
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_MATMUL_F32,
                                      "yvex_backend_op_matmul", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_set_current(backend, "yvex_backend_op_matmul", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    output_elements = m * n;
    rc = cuda_grid_1d(output_elements, block_size, &grid_size,
                      "cuda.matmul.grid", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    input_ptr = yvex_cuda_tensor_ptr(input);
    weight_ptr = yvex_cuda_tensor_ptr(weight);
    out_ptr = yvex_cuda_tensor_ptr(out);
    params[0] = &input_ptr;
    params[1] = &weight_ptr;
    params[2] = &out_ptr;
    params[3] = &m;
    params[4] = &k;
    params[5] = &n;
    rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_MATMUL_F32,
                          state->matmul_function, grid_size, block_size, 0,
                          params, "cuda.matmul.launch", err);
    if (rc == YVEX_OK) {
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_MATMUL_F32,
                                   "cuda.matmul.sync", err);
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    out->is_written = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_op_mlp(yvex_backend *backend,
                     const yvex_device_tensor *input,
                     const yvex_device_tensor *gate_weight,
                     const yvex_device_tensor *up_weight,
                     const yvex_device_tensor *down_weight,
                     const yvex_mlp_options *options,
                     yvex_device_tensor *intermediate,
                     yvex_device_tensor *out,
                     yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr input_ptr;
    CUdeviceptr gate_ptr;
    CUdeviceptr up_ptr;
    CUdeviceptr down_ptr;
    CUdeviceptr intermediate_ptr;
    CUdeviceptr out_ptr;
    unsigned long long batch;
    unsigned long long hidden_dim;
    unsigned long long ffn_dim;
    unsigned long long expert_count;
    unsigned long long expert_id;
    unsigned int block_size = 128u;
    int routed;
    void *params[12];
    yvex_backend_operation_variant variant;
    int rc;
    if (!state) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_mlp",
                       "CUDA backend state is missing");
        return YVEX_ERR_STATE;
    }
    rc = yvex_backend_validate_mlp(
        backend, input, gate_weight, up_weight, down_weight, options,
        intermediate, out, &batch, &hidden_dim, &ffn_dim, NULL, NULL, NULL,
        "yvex_backend_op_mlp", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    variant = options->routed_expert_mode
                  ? YVEX_BACKEND_VARIANT_MLP_ROUTED_F32
                  : YVEX_BACKEND_VARIANT_MLP_DENSE_F32;
    intermediate->is_written = 0;
    out->is_written = 0;
    rc = yvex_cuda_require_capability(backend, variant,
                                      "yvex_backend_op_mlp", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_set_current(backend, "yvex_backend_op_mlp", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    input_ptr = yvex_cuda_tensor_ptr(input);
    gate_ptr = yvex_cuda_tensor_ptr(gate_weight);
    up_ptr = yvex_cuda_tensor_ptr(up_weight);
    down_ptr = yvex_cuda_tensor_ptr(down_weight);
    intermediate_ptr = yvex_cuda_tensor_ptr(intermediate);
    out_ptr = yvex_cuda_tensor_ptr(out);
    expert_count = options->expert_count;
    expert_id = options->expert_id;
    routed = options->routed_expert_mode ? 1 : 0;
    params[0] = &input_ptr;
    params[1] = &gate_ptr;
    params[2] = &up_ptr;
    params[3] = &down_ptr;
    params[4] = &intermediate_ptr;
    params[5] = &out_ptr;
    params[6] = &batch;
    params[7] = &hidden_dim;
    params[8] = &ffn_dim;
    params[9] = &expert_count;
    params[10] = &expert_id;
    params[11] = &routed;
    rc = yvex_cuda_launch(backend, variant, state->mlp_function,
                          1, block_size, 0, params, "cuda.mlp.launch", err);
    if (rc == YVEX_OK) {
        rc = yvex_cuda_synchronize(backend, variant, "cuda.mlp.sync", err);
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    intermediate->is_written = 1;
    out->is_written = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_op_attention(yvex_backend *backend,
                           const yvex_device_tensor *query,
                           const yvex_device_tensor *keys,
                           const yvex_device_tensor *values,
                           unsigned long long seq_len,
                           unsigned long long position,
                           float scale,
                           int causal,
                           yvex_device_tensor *score_scratch,
                           yvex_device_tensor *probability_scratch,
                           yvex_device_tensor *out,
                           yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUdeviceptr query_ptr;
    CUdeviceptr keys_ptr;
    CUdeviceptr values_ptr;
    CUdeviceptr score_ptr;
    CUdeviceptr probability_ptr;
    CUdeviceptr out_ptr;
    unsigned long long head_dim;
    unsigned long long kv_elements;
    unsigned int block_size = 128u;
    int causal_flag = causal ? 1 : 0;
    void *params[11];
    yvex_backend_operation_variant variant;
    int rc;
    if (!state) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_attention",
                       "CUDA backend state is missing");
        return YVEX_ERR_STATE;
    }
    if (!isfinite(scale) || scale <= 0.0f) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_backend_op_attention",
                       "scale must be finite and positive");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_backend_validate_attention(
        backend, query, keys, values, seq_len, position, score_scratch,
        probability_scratch, out, &head_dim, &kv_elements,
        "yvex_backend_op_attention", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    (void)kv_elements;
    variant = causal ? YVEX_BACKEND_VARIANT_ATTENTION_CAUSAL_F32
                     : YVEX_BACKEND_VARIANT_ATTENTION_NONCAUSAL_F32;
    score_scratch->is_written = 0;
    probability_scratch->is_written = 0;
    out->is_written = 0;
    rc = yvex_cuda_require_capability(backend, variant,
                                      "yvex_backend_op_attention", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_cuda_set_current(backend, "yvex_backend_op_attention", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    query_ptr = yvex_cuda_tensor_ptr(query);
    keys_ptr = yvex_cuda_tensor_ptr(keys);
    values_ptr = yvex_cuda_tensor_ptr(values);
    score_ptr = yvex_cuda_tensor_ptr(score_scratch);
    probability_ptr = yvex_cuda_tensor_ptr(probability_scratch);
    out_ptr = yvex_cuda_tensor_ptr(out);
    params[0] = &query_ptr;
    params[1] = &keys_ptr;
    params[2] = &values_ptr;
    params[3] = &score_ptr;
    params[4] = &probability_ptr;
    params[5] = &out_ptr;
    params[6] = &seq_len;
    params[7] = &position;
    params[8] = &head_dim;
    params[9] = &scale;
    params[10] = &causal_flag;
    rc = yvex_cuda_launch(backend, variant, state->attention_function,
                          1, block_size, 0, params,
                          "cuda.attention.launch", err);
    if (rc == YVEX_OK) {
        rc = yvex_cuda_synchronize(backend, variant, "cuda.attention.sync", err);
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    score_scratch->is_written = 1;
    probability_scratch->is_written = 1;
    out->is_written = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int cuda_transformer_refuse(yvex_error *err, yvex_status status,
                                   const char *where, const char *reason)
{
    yvex_error_set(err, status, where, reason);
    return status;
}
/*
 * Decode selected encoded embedding rows and initialize repeated mHC streams on CUDA.
 *
 * Writes device embedding and expanded tensors; allocates only transaction scratch. Ownership,
 * launch, copy, status, sync, or cleanup refusal leaves output unadmitted. Transformer embedding
 * initialization only; no tokenizer or host numerical fallback.
 */
int yvex_backend_transformer_cuda_initial(
    yvex_backend *backend, const yvex_device_tensor *encoded, unsigned int qtype,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, yvex_device_tensor *embedding,
    yvex_device_tensor *expanded, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    const yvex_gguf_qtype_geometry *geometry = yvex_gguf_qtype_geometry_find(qtype);
    yvex_cuda_work work = {0};
    CUstream stream = yvex_cuda_launch_stream(backend);
    CUdeviceptr status = 0ull, encoded_ptr, embedding_ptr, expanded_ptr;
    unsigned long long count, expanded_count, encoded_required, token, residual_stream;
    size_t embedding_bytes, expanded_bytes, row_bytes;
    unsigned long long activation_bytes;
    unsigned int grid;
    int host_status = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !geometry || !geometry->block_size || !geometry->bytes_per_block || !facts ||
        !encoded || !embedding || !expanded || !token_count || !hidden_width ||
        !residual_streams || !yvex_core_u64_mul(token_count, hidden_width, &count) ||
        !yvex_core_u64_mul(count, residual_streams, &expanded_count) ||
        !yvex_core_u64_mul(count / geometry->block_size,
                           geometry->bytes_per_block, &encoded_required) ||
        !yvex_cuda_work_checked_bytes(count, sizeof(float), &embedding_bytes) ||
        !yvex_cuda_work_checked_bytes(expanded_count, sizeof(float), &expanded_bytes) ||
        !yvex_cuda_work_checked_bytes(hidden_width, sizeof(float), &row_bytes) ||
        !yvex_core_u64_add((unsigned long long)embedding_bytes,
                           (unsigned long long)expanded_bytes, &activation_bytes) ||
        !backend_tensor_owner_is(backend, encoded) ||
        count % geometry->block_size ||
        encoded->bytes < encoded_required ||
        !backend_tensor_owner_is(backend, embedding) || embedding->dtype != YVEX_DTYPE_F32 ||
        embedding->bytes < embedding_bytes ||
        !backend_tensor_owner_is(backend, expanded) || expanded->dtype != YVEX_DTYPE_F32 ||
        expanded->bytes < expanded_bytes ||
        count > UINT_MAX * (unsigned long long)CUDA_ATTENTION_BLOCK)
        return cuda_transformer_refuse(err, YVEX_ERR_FORMAT, "cuda.transformer.initial",
                                       "CUDA transformer embedding geometry is incompatible");
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    rc = yvex_cuda_work_allocate(&work, &status, sizeof(int), NULL, 1,
                                 "cuda.transformer.initial.status", NULL, err);
    encoded_ptr = (CUdeviceptr)encoded->data;
    embedding_ptr = (CUdeviceptr)embedding->data;
    expanded_ptr = (CUdeviceptr)expanded->data;
    grid = (unsigned int)((count + CUDA_ATTENTION_BLOCK - 1ull) / CUDA_ATTENTION_BLOCK);
    if (rc == YVEX_OK) {
        void *params[] = {&encoded_ptr, &count, &qtype, &embedding_ptr, &status};
        rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                              state->encoded_row_decode_function, grid, CUDA_ATTENTION_BLOCK,
                              0u, params, "cuda.transformer.embedding", err);
    }
    for (token = 0ull; rc == YVEX_OK && token < token_count; ++token)
        for (residual_stream = 0ull;
             rc == YVEX_OK && residual_stream < residual_streams; ++residual_stream) {
            CUdeviceptr source = embedding_ptr + token * hidden_width * sizeof(float);
            CUdeviceptr target = expanded_ptr +
                (token * residual_streams + residual_stream) * hidden_width * sizeof(float);
            CUresult copied = stream && state->driver.cuMemcpyDtoDAsync_v2
                                  ? state->driver.cuMemcpyDtoDAsync_v2(
                                        target, source, row_bytes, stream)
                                  : !stream ? state->driver.cuMemcpyDtoD_v2(
                                        target, source, row_bytes) : (CUresult)1;
            rc = yvex_cuda_status(&state->driver, copied,
                                  "cuda.transformer.initial.repeat", err);
        }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   "cuda.transformer.initial.sync", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuMemcpyDtoH_v2(&host_status, status, sizeof(host_status)),
                              "cuda.transformer.initial.status", err);
    if (rc == YVEX_OK && host_status)
        rc = cuda_transformer_refuse(err, YVEX_ERR_FORMAT, "cuda.transformer.initial",
                                     "CUDA transformer embedding produced invalid numerics");
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    if (rc == YVEX_OK) {
        embedding->is_written = 1;
        expanded->is_written = 1;
        facts->d2h_bytes = sizeof(host_status);
        facts->d2d_bytes = expanded_bytes;
        facts->kernel_launches = 1ull;
        facts->download_count = 1ull;
        facts->device_synchronizations = 1ull;
        facts->active_weight_bytes = encoded_required;
        facts->activation_bytes = activation_bytes;
        facts->temporary_bytes = sizeof(host_status);
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}

/*
 * Collapse target-feature residual streams on CUDA and publish compact and optional resident rows.
 *
 * The caller supplies reusable device storage and may request compact host evidence. Bounded
 * status and resident rows become visible together; expanded input never leaves the device.
 */
int yvex_backend_transformer_cuda_feature_mean(
    yvex_backend *backend, const yvex_device_tensor *expanded,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, yvex_device_tensor *device_output,
    yvex_device_tensor *resident_output, unsigned long long resident_row_offset,
    unsigned long long resident_row_stride, unsigned long long resident_column_offset,
    float *host_output, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    CUdeviceptr status = 0ull, input_ptr, output_ptr, resident_ptr = 0ull;
    unsigned long long input_count, output_count, activation_count, resident_rows, resident_elements;
    size_t output_bytes, activation_bytes;
    unsigned int grid;
    int host_status = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !state->transformer_feature_mean_function || !facts ||
        !token_count || !hidden_width || !residual_streams ||
        !yvex_core_u64_mul(token_count, hidden_width, &output_count) ||
        !yvex_core_u64_mul(output_count, residual_streams, &input_count) ||
        !yvex_core_u64_add(input_count, output_count, &activation_count) ||
        (resident_output &&
         (!yvex_core_u64_add(activation_count, output_count, &activation_count) ||
          !resident_row_stride || resident_column_offset > resident_row_stride ||
          hidden_width > resident_row_stride - resident_column_offset ||
          !yvex_core_u64_add(resident_row_offset, token_count, &resident_rows) ||
          !yvex_core_u64_mul(resident_rows, resident_row_stride, &resident_elements) ||
          resident_elements > ULLONG_MAX / sizeof(float) ||
          !backend_tensor_owner_is(backend, resident_output) ||
          resident_output->dtype != YVEX_DTYPE_F32 ||
          resident_output->data == expanded->data || resident_output->data == device_output->data ||
          resident_output->bytes < resident_elements * sizeof(float))) ||
        (!resident_output &&
         (resident_row_offset || resident_row_stride || resident_column_offset)) ||
        !yvex_cuda_work_checked_bytes(output_count, sizeof(float), &output_bytes) ||
        !yvex_cuda_work_checked_bytes(activation_count, sizeof(float), &activation_bytes) ||
        output_count > UINT_MAX * (unsigned long long)CUDA_ATTENTION_BLOCK ||
        !backend_tensor_f32_elements(expanded, input_count) ||
        !backend_tensor_f32_elements(device_output, output_count))
        return cuda_transformer_refuse(
            err, YVEX_ERR_FORMAT, "cuda.transformer.feature-mean",
            "CUDA transformer feature geometry is incompatible");
    device_output->is_written = 0;
    if (resident_output) resident_output->is_written = 0;
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    rc = yvex_cuda_work_allocate(&work, &status, sizeof(int), NULL, 1,
                                 "cuda.transformer.feature-mean.status", NULL, err);
    input_ptr = (CUdeviceptr)expanded->data;
    output_ptr = (CUdeviceptr)device_output->data;
    if (resident_output) resident_ptr = (CUdeviceptr)resident_output->data;
    grid = (unsigned int)((output_count + CUDA_ATTENTION_BLOCK - 1ull) /
                          CUDA_ATTENTION_BLOCK);
    if (rc == YVEX_OK) {
        void *params[] = {
            &input_ptr, &token_count, &residual_streams, &hidden_width,
            &output_ptr, &resident_ptr, &resident_row_offset,
            &resident_row_stride, &resident_column_offset, &status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->transformer_feature_mean_function, grid, CUDA_ATTENTION_BLOCK,
            0u, params, "cuda.transformer.feature-mean", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   "cuda.transformer.feature-mean.sync", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemcpyDtoH_v2(&host_status, status, sizeof(host_status)),
            "cuda.transformer.feature-mean.status", err);
    if (rc == YVEX_OK && host_status)
        rc = cuda_transformer_refuse(
            err, YVEX_ERR_FORMAT, "cuda.transformer.feature-mean",
            "CUDA transformer feature reduction produced invalid numerics");
    if (rc == YVEX_OK && host_output)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemcpyDtoH_v2(host_output, output_ptr, output_bytes),
            "cuda.transformer.feature-mean.output", err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc == YVEX_OK) {
        device_output->is_written = 1;
        if (resident_output) resident_output->is_written = 1;
        facts->d2h_bytes = sizeof(host_status) + (host_output ? output_bytes : 0u);
        facts->kernel_launches = 1ull;
        facts->download_count = 1ull + (host_output != NULL);
        facts->device_synchronizations = 1ull;
        facts->activation_bytes = activation_bytes;
        facts->temporary_bytes = sizeof(host_status);
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_backend_transformer_cuda_final(
    yvex_backend *backend, const yvex_device_tensor *expanded,
    const yvex_device_tensor *function, const yvex_device_tensor *base,
    const yvex_device_tensor *scale, const yvex_device_tensor *norm,
    unsigned long long token_count, unsigned long long hidden_width,
    unsigned long long residual_streams, double epsilon, double mhc_epsilon,
    yvex_device_tensor *pre_normalized, yvex_device_tensor *output,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    CUdeviceptr status = 0ull, input_ptr, function_ptr, base_ptr, scale_ptr, norm_ptr;
    CUdeviceptr pre_output_ptr = 0ull, output_ptr;
    unsigned long long expanded_width, expanded_count, function_count, output_count;
    unsigned long long weight_count, activation_count;
    size_t weight_bytes, activation_bytes;
    int host_status = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !facts || !token_count || !hidden_width || !residual_streams ||
        !yvex_core_u64_mul(hidden_width, residual_streams, &expanded_width) ||
        !yvex_core_u64_mul(token_count, expanded_width, &expanded_count) ||
        !yvex_core_u64_mul(residual_streams, expanded_width, &function_count) ||
        !yvex_core_u64_mul(token_count, hidden_width, &output_count) ||
        !yvex_core_u64_add(function_count, residual_streams, &weight_count) ||
        !yvex_core_u64_add(weight_count, 1ull, &weight_count) ||
        !yvex_core_u64_add(weight_count, hidden_width, &weight_count) ||
        !yvex_core_u64_add(expanded_count, output_count, &activation_count) ||
        (pre_normalized &&
         !yvex_core_u64_add(activation_count, output_count, &activation_count)) ||
        !yvex_cuda_work_checked_bytes(weight_count, sizeof(float), &weight_bytes) ||
        !yvex_cuda_work_checked_bytes(activation_count, sizeof(float), &activation_bytes) ||
        token_count > UINT_MAX || !backend_tensor_f32_elements(expanded, expanded_count) ||
        !backend_tensor_f32_elements(function, function_count) ||
        !backend_tensor_f32_elements(base, residual_streams) ||
        !backend_tensor_f32_elements(scale, 1ull) ||
        !backend_tensor_f32_elements(norm, hidden_width) ||
        !backend_tensor_f32_elements(output, output_count) ||
        (pre_normalized &&
         (!backend_tensor_f32_elements(pre_normalized, output_count) ||
          pre_normalized->data == output->data)))
        return cuda_transformer_refuse(err, YVEX_ERR_FORMAT, "cuda.transformer.final",
                                       "CUDA transformer final geometry is incompatible");
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    rc = yvex_cuda_work_allocate(&work, &status, sizeof(int), NULL, 1,
                                 "cuda.transformer.final.status", NULL, err);
    input_ptr = (CUdeviceptr)expanded->data; function_ptr = (CUdeviceptr)function->data;
    base_ptr = (CUdeviceptr)base->data; scale_ptr = (CUdeviceptr)scale->data;
    norm_ptr = (CUdeviceptr)norm->data;
    if (pre_normalized) pre_output_ptr = (CUdeviceptr)pre_normalized->data;
    output_ptr = (CUdeviceptr)output->data;
    if (rc == YVEX_OK) {
        void *params[] = {&input_ptr, &function_ptr, &base_ptr, &scale_ptr, &norm_ptr,
                          &token_count, &residual_streams, &hidden_width, &epsilon,
                          &mhc_epsilon, &pre_output_ptr, &output_ptr, &status};
        rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->transformer_final_function, (unsigned int)token_count,
            CUDA_ATTENTION_BLOCK, CUDA_ATTENTION_BLOCK * (unsigned int)sizeof(double),
            params, "cuda.transformer.final", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   "cuda.transformer.final.sync", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuMemcpyDtoH_v2(&host_status, status, sizeof(host_status)),
                              "cuda.transformer.final.status", err);
    if (rc == YVEX_OK && host_status)
        rc = cuda_transformer_refuse(err, YVEX_ERR_FORMAT, "cuda.transformer.final",
                                     "CUDA transformer final stage produced invalid numerics");
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    if (rc == YVEX_OK) {
        if (pre_normalized) pre_normalized->is_written = 1;
        output->is_written = 1;
        facts->d2h_bytes = sizeof(host_status);
        facts->kernel_launches = 1ull;
        facts->download_count = 1ull;
        facts->device_synchronizations = 1ull;
        facts->active_weight_bytes = weight_bytes;
        facts->activation_bytes = activation_bytes;
        facts->temporary_bytes = sizeof(host_status);
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_cuda_activation_views_valid(yvex_backend *backend,
    const yvex_device_tensor *input, unsigned long long input_elements,
    const yvex_device_tensor *output, unsigned long long output_elements)
{
    return backend_tensor_owner_is(backend, input) &&
           backend_tensor_owner_is(backend, output) &&
           backend_tensor_f32_elements(input, input_elements) &&
           backend_tensor_f32_elements(output, output_elements);
}
/*
 * Resolve one admitted activation tensor.
 *
 * Returns zero for foreign ownership.
 */
CUdeviceptr yvex_cuda_activation_pointer(
    yvex_backend *backend, const yvex_device_tensor *tensor)
{
    return backend_tensor_owner_is(backend, tensor) ? (CUdeviceptr)tensor->data : 0ull;
}
/*
 * Copy a completed F32 activation into one stable backend-owned view.
 *
 * Ownership, extent, Driver API, or copy error leaves output unpublished. Transport only; no
 * synchronization or numerical fallback.
 */
int yvex_cuda_activation_copy(yvex_backend *backend, CUdeviceptr source,
    yvex_device_tensor *output, unsigned long long elements,
    const char *stage, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    CUstream stream = yvex_cuda_launch_stream(backend);
    size_t bytes;
    CUresult copied;
    if (!state || !source || !stage || !backend_tensor_f32_elements(output, elements) ||
        !yvex_cuda_work_checked_bytes(elements, sizeof(float), &bytes)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, stage, "CUDA activation copy extent is invalid");
        return YVEX_ERR_BOUNDS;
    }
    copied = stream && state->driver.cuMemcpyDtoDAsync_v2
                 ? state->driver.cuMemcpyDtoDAsync_v2(
                       (CUdeviceptr)output->data, source, bytes, stream)
                 : !stream ? state->driver.cuMemcpyDtoD_v2(
                       (CUdeviceptr)output->data, source, bytes) : (CUresult)1;
    {
        int rc = yvex_cuda_status(&state->driver, copied, stage, err);
        if (rc != YVEX_OK) return rc;
    }
    output->is_written = 1;
    return YVEX_OK;
}
