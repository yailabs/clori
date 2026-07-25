/* Owner: src/backend/cuda.
 * Owns: exact dtype/shape/parameter validation, Driver API launch parameters, bounded grid arithmetic,
 *   synchronization, and output-written transitions.
 * Does not own: bundle admission, device kernel source, CPU references, graph semantics, model-family behavior, CLI
 *   output, qtype compute, or generation.
 * Invariants: every op requires an exact admitted variant; launch and final synchronization must succeed before any
 *   output is marked written.
 * Boundary: bounded primitive execution is not transformer or model runtime.
 * Purpose: Validate and launch CUDA graph primitives through admitted generated-kernel variants.
 * Inputs: Owned CUDA tensors, checked operation geometry, and immutable numeric parameters.
 * Effects: Launches work and marks output written only after successful synchronization.
 * Failure: Any admission, launch, or sync failure leaves output uncommitted. */

#include "src/backend/cuda/private.h"
#include <yvex/internal/graph_state.h>
#include <yvex/quant.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_ATTENTION_BLOCK 256u

/* Purpose: initialize one stable device range synchronously or on the active capture stream.
 * Inputs: live work owner, exact device range, and either source bytes or zero policy.
 * Effects: performs eager initialization or enqueues one capturable copy/memset node.
 * Failure: malformed ranges or unavailable Driver entrypoints publish typed failure.
 * Boundary: never allocates, synchronizes, or substitutes host numerical work. */
static int cuda_work_initialize(yvex_cuda_work *work, CUdeviceptr target,
                                size_t bytes, const void *source, int zero,
                                const char *stage, yvex_error *err)
{
    CUstream stream;

    if (!source && !zero) return YVEX_OK;
    if (!work || !work->state || !target || !bytes) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "CUDA range initialization is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    if (work->prepare_only) return YVEX_OK;
    stream = yvex_cuda_launch_stream(work->backend);
    if (stream) {
        if ((source && !work->state->driver.cuMemcpyHtoDAsync_v2) ||
            (zero && !work->state->driver.cuMemsetD8Async)) {
            yvex_error_set(err, YVEX_ERR_UNSUPPORTED, stage,
                           "captured CUDA range initialization is unavailable");
            return YVEX_ERR_UNSUPPORTED;
        }
        return yvex_cuda_status(
            &work->state->driver,
            source ? work->state->driver.cuMemcpyHtoDAsync_v2(
                         target, source, bytes, stream)
                   : work->state->driver.cuMemsetD8Async(target, 0u, bytes, stream),
            stage, err);
    }
    return yvex_cuda_status(
        &work->state->driver,
        source ? work->state->driver.cuMemcpyHtoD_v2(target, source, bytes)
               : work->state->driver.cuMemsetD8_v2(target, 0u, bytes),
        stage, err);
}

/* Purpose: acquire and initialize one range from a stable workspace or owned device allocation.
 * Inputs: initialized work owner, byte extent, initialization policy, and typed failure output.
 * Effects: tracks one range and advances exact current/peak device-byte accounting.
 * Failure: budget, workspace, allocation, or copy error retains ownership for cleanup.
 * Boundary: generic CUDA transaction resource; it does not infer family geometry. */
int yvex_cuda_work_allocate(yvex_cuda_work *work,
                            CUdeviceptr *out,
                            size_t bytes,
                            const void *source,
                            int zero,
                            const char *stage,
                            yvex_cuda_work_failure *failure,
                            yvex_error *err)
{
    unsigned long long address = 0ull;
    unsigned long long next;
    int acquired;
    int rc;

    if (failure)
        *failure = YVEX_CUDA_WORK_FAILURE_NONE;
    if (!work || !work->backend || !work->state || !out || !bytes ||
        work->count >= YVEX_CUDA_WORK_MAX_RANGES) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, stage,
                       "CUDA work allocation request is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = backend_dispatch_admit(work->backend, stage, err);
    if (rc != YVEX_OK) {
        if (failure) *failure = YVEX_CUDA_WORK_FAILURE_ALLOCATION;
        return rc;
    }
    if (work->count == 0u) {
        rc = yvex_cuda_deferred_release_drain(work->backend, err);
        if (rc != YVEX_OK) {
            if (failure)
                *failure = YVEX_CUDA_WORK_FAILURE_ALLOCATION;
            return rc;
        }
    }
    if (work->current_bytes > ULLONG_MAX - (unsigned long long)bytes ||
        (work->budget && work->current_bytes + (unsigned long long)bytes > work->budget)) {
        if (failure)
            *failure = YVEX_CUDA_WORK_FAILURE_BUDGET;
        yvex_error_set(err, YVEX_ERR_NOMEM, stage, "CUDA work device-byte budget exceeded");
        return YVEX_ERR_NOMEM;
    }
    next = work->current_bytes + (unsigned long long)bytes;
    acquired = work->raw_only ? YVEX_BACKEND_RESIDENT_MISS
                              : yvex_backend_workspace_acquire(
                                    work->backend, bytes, 256ull, &address);
    if (acquired == YVEX_BACKEND_RESIDENT_HIT) {
        *out = (CUdeviceptr)address;
        work->workspace_owned[work->count] = 1u;
    } else if (!work->raw_only && work->backend->workspace_device_tensor) {
        if (failure)
            *failure = YVEX_CUDA_WORK_FAILURE_BUDGET;
        yvex_error_setf(
            err, acquired == YVEX_BACKEND_RESIDENT_INVALID
                     ? YVEX_ERR_BOUNDS : YVEX_ERR_NOMEM,
            stage,
            "CUDA reusable workspace needs %zu bytes at cursor %llu of %llu",
            bytes, work->backend->workspace_cursor,
            work->backend->workspace_bytes);
        return acquired == YVEX_BACKEND_RESIDENT_INVALID ? YVEX_ERR_BOUNDS
                                                         : YVEX_ERR_NOMEM;
    } else {
        rc = yvex_backend_memory_can_add(work->backend, bytes, "CUDA", stage, err);
        if (rc != YVEX_OK) {
            if (failure)
                *failure = YVEX_CUDA_WORK_FAILURE_BUDGET;
            return rc;
        }
        rc = yvex_cuda_status(&work->state->driver,
                              work->state->driver.cuMemAlloc_v2(out, bytes), stage, err);
        if (rc != YVEX_OK) {
            if (failure)
                *failure = YVEX_CUDA_WORK_FAILURE_ALLOCATION;
            return rc;
        }
        backend_memory_acquire(work->backend, bytes);
    }
    work->pointers[work->count] = *out;
    work->sizes[work->count++] = bytes;
    work->current_bytes = next;
    if (next > work->peak_bytes)
        work->peak_bytes = next;
    rc = cuda_work_initialize(work, *out, bytes, source, zero, stage, err);
    if (rc != YVEX_OK && failure)
        *failure = YVEX_CUDA_WORK_FAILURE_COPY;
    return rc;
}

/* Purpose: release every owned CUDA work range in reverse acquisition order.
 * Inputs: initialized or partially initialized work owner.
 * Effects: frees only non-workspace ranges and clears exact ownership/accounting.
 * Failure: returns the first Driver cleanup error while attempting every range.
 * Boundary: stable session workspace remains attached and caller-owned. */
int yvex_cuda_work_cleanup(yvex_cuda_work *work, yvex_error *err)
{
    yvex_error cleanup;
    int result = YVEX_OK;

    if (!work) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    while (work->count) {
        unsigned int index = work->count - 1u;
        int rc = work->workspace_owned[index]
                     ? YVEX_OK
                     : yvex_cuda_temporary_free(work->backend, work->variant,
                                                &work->pointers[index],
                                                work->sizes[index], 1,
                                                "cuda.work.cleanup", &cleanup);
        if (!work->workspace_owned[index] && work->pointers[index] != 0u) {
            if (result == YVEX_OK) {
                result = rc;
                if (err)
                    *err = cleanup;
            }
            break;
        }
        work->current_bytes = work->current_bytes >= work->sizes[index]
                                  ? work->current_bytes - work->sizes[index] : 0ull;
        work->pointers[index] = 0u;
        work->workspace_owned[index] = 0u;
        work->sizes[index] = 0ull;
        work->count = index;
        if (rc != YVEX_OK && result == YVEX_OK) {
            result = rc;
            if (err)
                *err = cleanup;
        }
    }
    if (result == YVEX_OK)
        yvex_error_clear(err);
    return result;
}

/* Purpose: launch one admitted kernel through eager or active graph-capture stream ownership.
 * Inputs: initialized work owner, admitted function, geometry, and kernel parameter array.
 * Effects: enqueues one kernel and counts direct launches outside capture.
 * Failure: Driver/capability failure leaves launch accounting unchanged.
 * Boundary: synchronization and family-specific error mapping remain caller-owned. */
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

/* Purpose: record one typed CUDA attention primitive refusal.
 * Inputs: optional failure storage, code, stage, evidence, status, and message.
 * Effects: resets failure detail and writes the canonical error.
 * Failure: returns the supplied status.
 * Boundary: generic CUDA attention primitives; no family topology or cleanup. */
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
    yvex_error_set(err, status, stage, message);
    return status;
}

/* Purpose: advance one exact attention transfer counter with checked byte arithmetic. */
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

/* Purpose: acquire one bounded CUDA attention work range.
 * Inputs: work owner, size, initialization policy, stage, and failure storage.
 * Effects: allocates, tracks, and initializes one range.
 * Failure: typed injected, budget, allocation, or copy refusal.
 * Boundary: generic transaction resource; no family-specific geometry. */
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

/* Purpose: initialize an already allocated attention range inside eager or captured execution.
 * Inputs: live work owner, exact range, initialization policy, and typed failure storage.
 * Effects: performs or captures one H2D copy/memset without allocating the range.
 * Failure: injected, validation, or Driver failures preserve explicit copy failure.
 * Boundary: preparation only; no attention math or synchronization is performed. */
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
    rc = cuda_work_initialize(work, target, bytes, source, zero, stage, err);
    return rc == YVEX_OK ? YVEX_OK : attention_fail(
        failure, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull, err,
        (yvex_status)rc, "CUDA attention range initialization failed");
}

/* Purpose: copy one device result into stable host staging inside eager or captured execution.
 * Inputs: live work owner, pinned/stable host target, exact device range, and failure storage.
 * Effects: performs eager D2H or captures one D2H node into the active graph stream.
 * Failure: invalid ranges, injected faults, or missing Driver APIs return typed copy failure.
 * Boundary: stages bytes only; caller synchronizes, validates, and publishes transactionally. */
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

/* Purpose: launch one admitted generic attention kernel.
 * Inputs: work owner, function, launch geometry, parameters, and failure storage.
 * Effects: enqueues work and advances eager launch accounting.
 * Failure: typed injected or Driver refusal.
 * Boundary: launch only; family transaction owns synchronization and publication. */
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

/* Purpose: publish one finite activation vector at the admitted BF16 RNE boundary.
 * Inputs: device values, exact count, shared status, and typed failure storage.
 * Effects: enqueues one capturable in-place rounding kernel.
 * Failure: invalid geometry or launch refusal remains typed and uncommitted.
 * Boundary: generic attention numeric ingress; family owners select its semantic placement. */
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

/* Purpose: execute one encoded attention matrix-vector product directly on device.
 * Inputs: admitted weight slice, vector, output, rounding policy, and status range.
 * Effects: enqueues direct encoded computation.
 * Failure: typed geometry or launch refusal.
 * Boundary: generic encoded attention primitive; no host decode fallback. */
static int attention_matvec(yvex_cuda_work *work,
                               const yvex_backend_attention_weight *weight,
                               CUdeviceptr device_weight,
                               unsigned long long start_row,
                               unsigned long long rows,
                               CUdeviceptr vector,
                               CUdeviceptr out,
                               int output_bf16,
                               CUdeviceptr status,
                               const char *stage,
                               yvex_backend_attention_failure *failure,
                               yvex_error *err)
{
    if (!weight || !weight->present || !device_weight || !vector || !out ||
        !rows || start_row > weight->row_count ||
        rows > weight->row_count - start_row || rows > UINT_MAX)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            weight ? weight->row_count : 0ull, start_row + rows, err,
            YVEX_ERR_BOUNDS, "CUDA attention matvec geometry is invalid");
    {
        void *params[] = {
            &device_weight, (void *)&weight->row_bytes,
            (void *)&weight->row_width, &start_row, &rows,
            (void *)&weight->qtype, &vector, &out, &output_bf16, &status
        };
        return attention_launch(
            work, work->state->deepseek_qtype_matvec_function,
            (unsigned int)rows, CUDA_ATTENTION_BLOCK,
            CUDA_ATTENTION_BLOCK * (unsigned int)sizeof(double), params, stage,
            failure, err);
    }
}

/* Purpose: decode one admitted encoded attention row on device.
 * Inputs: weight, row, element count, output, status, and failure storage.
 * Effects: enqueues bounded reference decoding.
 * Failure: typed row geometry or launch refusal.
 * Boundary: device primitive; never publishes host output. */
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
            work, work->state->deepseek_decode_function, grid,
            CUDA_ATTENTION_BLOCK, 0u, params, stage, failure, err);
    }
}

/* Purpose: apply one learned RMS normalization to device values.
 * Inputs: values, exact weight row, epsilon, status, and failure storage.
 * Effects: enqueues in-place normalization.
 * Failure: typed shape or launch refusal.
 * Boundary: generic attention normalization primitive. */
static int attention_weighted_norm(
    yvex_cuda_work *work, CUdeviceptr values, unsigned long long count,
    const yvex_backend_attention_weight *weight, CUdeviceptr device_weight,
    double epsilon, CUdeviceptr status, const char *stage,
    yvex_backend_attention_failure *failure, yvex_error *err)
{
    if (!weight || !weight->present || weight->row_count != 1ull ||
        weight->row_width != count)
        return attention_fail(
            failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT, stage,
            count, weight ? weight->row_width : 0ull, err, YVEX_ERR_FORMAT,
            "CUDA attention normalization weight shape is invalid");
    {
        void *params[] = {
            &values, &count, &device_weight, (void *)&weight->qtype,
            &epsilon, &status
        };
        return attention_launch(
            work, work->state->deepseek_weighted_norm_function, 1u,
            1u, 0u, params, stage, failure, err);
    }
}

/* Purpose: unit-normalize one or more device vectors.
 * Inputs: values, vector geometry, epsilon, status, and failure storage.
 * Effects: enqueues in-place normalization.
 * Failure: typed geometry or launch refusal.
 * Boundary: generic attention numeric primitive. */
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
            work, work->state->deepseek_unit_norm_function,
            (unsigned int)vectors, CUDA_ATTENTION_BLOCK,
            CUDA_ATTENTION_BLOCK * (unsigned int)sizeof(double), params, stage,
            failure, err);
    }
}

/* Purpose: apply or invert one admitted RoPE/YaRN position transform.
 * Inputs: vector geometry, absolute position, immutable policy, and status.
 * Effects: enqueues in-place rotation.
 * Failure: typed geometry or launch refusal.
 * Boundary: generic device position primitive; family supplies the policy. */
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
            work, work->state->deepseek_rope_function, grid,
            CUDA_ATTENTION_BLOCK, 0u, params, stage, failure, err);
    }
}

/* Purpose: apply one admitted activation fake-quantization policy on device.
 * Inputs: vectors, geometry, policy, status, and failure storage.
 * Effects: conditionally enqueues in-place fake quantization.
 * Failure: typed geometry or launch refusal.
 * Boundary: no-op only when the immutable policy marks quantization unnecessary. */
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
            work, work->state->deepseek_activation_function,
            (unsigned int)vectors, 1u, 0u, params, stage, failure, err);
    }
}

/* Purpose: validate the family-neutral encoded-attention request envelope.
 * Inputs: immutable request, caller output views, and failure storage.
 * Effects: reads only.
 * Failure: malformed phase, geometry, history, or evidence facts refuse explicitly.
 * Boundary: family-specific ratios and tensor-role requirements remain with the adapter. */
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
        !job->input || !job->token_count || !input_width ||
        job->input_stride < input_width ||
        job->token_position > ULLONG_MAX - job->token_count ||
        (job->phase == YVEX_BACKEND_ATTENTION_PHASE_DECODE &&
         job->token_count != 1ull) ||
        (job->phase != YVEX_BACKEND_ATTENTION_PHASE_DECODE &&
         job->phase != YVEX_BACKEND_ATTENTION_PHASE_PREFILL) ||
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
    return YVEX_OK;
}

/* Purpose: validate one encoded matrix against exact CUDA qtype geometry.
 * Inputs: immutable encoded weight and expected logical rows and width.
 * Effects: reads canonical qtype capability only.
 * Failure: missing bytes, shape drift, or unavailable CUDA compute refuses.
 * Boundary: validates physical compute admission without inferring tensor roles. */
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

/* Purpose: validate one reusable activation-quantization policy for CUDA execution.
 * Inputs: immutable policy, exact vector width, stage, and failure storage.
 * Effects: reads only.
 * Failure: incompatible block, encoding, or Hadamard geometry refuses.
 * Boundary: family adapters select policy; generic CUDA owns variant admission. */
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

/* Purpose: validate one immutable rolling-state view against exact request geometry.
 * Inputs: request, prior state, required ratio/head/overlap, and extent output.
 * Effects: publishes the checked state extent only.
 * Failure: absent, stale, undersized, or incompatible state refuses explicitly.
 * Boundary: validates generic recurrence storage without selecting family policy. */
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

/* Purpose: prove disjoint representable host spans.
 * Inputs: immutable write/read span inventories.
 * Effects: none.
 * Failure: zero denotes an invalid range; negative denotes overlap.
 * Boundary: pure host address admission without transfer or publication. */
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

/* Purpose: prove disjoint representable attention input and output spans.
 * Inputs: immutable job, transfer inventory, and checked history extents.
 * Effects: none.
 * Failure: zero denotes an invalid range; negative denotes overlap.
 * Boundary: generic host alias admission performs no transfer or publication. */
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
    READ(job->input, input_count, sizeof(float));
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

/* Purpose: honor request cancellation without abandoning pending device work.
 * Inputs: backend, immutable request, stage, pending-work fact, and failure storage.
 * Effects: synchronizes only when cancellation observes pending CUDA work.
 * Failure: synchronization or cancellation publishes one typed refusal.
 * Boundary: never publishes output or candidate state. */
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

/* Purpose: acquire one exact reusable host-staging span for attention execution.
 * Inputs: backend, extent, graph pinning policy, injected-failure fact, and outputs.
 * Effects: borrows one session-owned span and reports whether prior use exists.
 * Failure: capacity, allocation, or pinning refusal leaves no borrowed span.
 * Boundary: generic staging ownership; family layout remains caller-defined. */
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

/* Purpose: bind one aligned host-staging span and advance its checked cursor. */
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

/* Purpose: derive and bind the complete generic attention host-staging layout.
 * Inputs: upload/transfer catalogs, CSA count slots, optional base, and extent output.
 * Effects: binds every staging pointer and publishes the total aligned extent.
 * Failure: false on malformed catalogs or unrepresentable layout.
 * Boundary: layout only; allocation, transfer, and family input generation remain separate. */
static int attention_stage_layout(
    unsigned char *base, yvex_cuda_attention_upload *uploads, size_t upload_count,
    yvex_cuda_attention_transfer *transfers, size_t transfer_count,
    unsigned long long csa_tokens, int **status, unsigned long long **selected,
    unsigned long long **candidates, size_t *total)
{
    size_t cursor = 0u, i;
    if (!uploads || !transfers || !status || !selected || !candidates || !total)
        return 0;
    for (i = 0u; i < upload_count; ++i)
        if (!attention_stage_range(base, &cursor, uploads[i].count,
                                   uploads[i].width, &uploads[i].staged))
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
    *total = cursor;
    return 1;
}

/* Purpose: expose the single private encoded-attention operation boundary.
 * Inputs: none.
 * Effects: none; returns immutable process-lifetime methods.
 * Failure: none.
 * Boundary: one ABI replaces per-stage globals without moving family policy here. */
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
        attention_activation
    };
    return &operations;
}

/* Contract: converts a non-zero one-dimensional launch extent without truncation. */
/* Purpose: Implement the canonical grid 1d mechanism owned by the CUDA backend boundary.
 * Inputs: Typed caller-owned outputs and immutable values declared by this subsystem ABI.
 * Effects: Updates only caller-owned result storage or lifecycle state explicitly named by the ABI.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Purpose: Execute the typed op embed operation over already admitted buffers.
 * Inputs: Typed admitted handles, immutable source ranges, checked dimensions, and an explicit destination.
 * Effects: Mutates only the admitted destination or transaction after every precondition passes.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Purpose: lower one sealed family-neutral workspace recipe to a checked byte extent.
 * Inputs: pointer-free semantic components with explicit alignment and token scaling.
 * Effects: publishes one exact upper bound; performs no allocation or family inference.
 * Failure: malformed identity or arithmetic overflow leaves required bytes zero.
 * Boundary: backend owns alignment lowering, while graph/family owners select components. */
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

/* Purpose: Execute the typed op rms norm operation over already admitted buffers.
 * Inputs: Typed admitted handles, immutable source ranges, checked dimensions, and an explicit destination.
 * Effects: Mutates only the admitted destination or transaction after every precondition passes.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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
    unsigned int block_size = 256u;
    unsigned int shared_bytes = block_size * (unsigned int)sizeof(float);
    void *params[5];
    yvex_backend_operation_variant variant;
    int rc;

    if (!state) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_rms_norm",
                       "CUDA backend state is missing");
        return YVEX_ERR_STATE;
    }
    if (!backend_tensor_owner_is(backend, input) ||
        !backend_tensor_owner_is(backend, weight) ||
        !backend_tensor_owner_is(backend, out)) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_backend_op_rms_norm",
                       "input, weight, and output tensors must belong to this backend");
        return YVEX_ERR_STATE;
    }
    if (input->dtype != YVEX_DTYPE_F32 || out->dtype != YVEX_DTYPE_F32 ||
        (weight->dtype != YVEX_DTYPE_F32 && weight->dtype != YVEX_DTYPE_F16)) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "yvex_backend_op_rms_norm",
                       "CUDA RMSNorm supports F32 input/output with F16 or F32 weight");
        return YVEX_ERR_UNSUPPORTED;
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
    rc = yvex_backend_validate_rms_norm(
        backend, input, weight, epsilon, out, &hidden_size,
        "CUDA RMSNorm supports F32 input/output with F16 or F32 weight",
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
    params[4] = &epsilon;

    rc = yvex_cuda_launch(backend, variant,
                          weight->dtype == YVEX_DTYPE_F16
                              ? state->rms_norm_f16_function
                              : state->rms_norm_f32_function,
                          1, block_size, shared_bytes, params,
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

/* Purpose: Execute the typed op rope operation over already admitted buffers.
 * Inputs: Typed admitted handles, immutable source ranges, checked dimensions, and an explicit destination.
 * Effects: Mutates only the admitted destination or transaction after every precondition passes.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Purpose: Execute the typed op matmul operation over already admitted buffers.
 * Inputs: Typed admitted handles, immutable source ranges, checked dimensions, and an explicit destination.
 * Effects: Mutates only the admitted destination or transaction after every precondition passes.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Purpose: Execute the typed op mlp operation over already admitted buffers.
 * Inputs: Typed admitted handles, immutable source ranges, checked dimensions, and an explicit destination.
 * Effects: Mutates only the admitted destination or transaction after every precondition passes.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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

/* Purpose: Execute the typed op attention operation over already admitted buffers.
 * Inputs: Typed admitted handles, immutable source ranges, checked dimensions, and an explicit destination.
 * Effects: Mutates only the admitted destination or transaction after every precondition passes.
 * Failure: Returns a typed CUDA refusal and publishes no partial success state.
 * Boundary: CUDA execution; does not infer model topology, profile policy, or runtime support. */
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
