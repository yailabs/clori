/*
 * Project canonical qtype capability and execute bounded encoded row-dot proofs on CUDA.
 *
 * Qtype compute support must be present in TRACK.QUANT and proven by the dedicated generated-PTX
 * row-dot variant before this owner reports it. CUDA qtype facts do not make CUDA runtime
 * generation available.
 */
#include "src/backend/cuda/private.h"
#include <yvex/internal/quant_numeric.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define CUDA_QTYPE_MATVEC_BLOCK 256u
#define CUDA_QTYPE_MATVEC_ROWS 8u

static int cuda_quant_fail(yvex_quant_failure *failure,
                           yvex_quant_failure_code code,
                           unsigned int qtype,
                           unsigned long long expected,
                           unsigned long long actual,
                           yvex_error *err,
                           int status,
                           const char *message)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->terminal_ordinal = ULLONG_MAX;
        failure->source_index = ULLONG_MAX;
        failure->row_index = 0u;
        failure->block_index = ULLONG_MAX;
        failure->expected = expected;
        failure->actual = actual;
        failure->qtype = qtype;
        failure->operation = YVEX_TRANSFORM_OP_COUNT;
    }
    yvex_error_set(err, (yvex_status)status, "cuda.quant.row_dot", message);
    return status;
}
/*
 * Project one resident encoded matrix through the generic CUDA qtype matvec.
 *
 * Exact resident span/geometry and stable backend-owned F32 input/output tensors.
 */
int yvex_backend_cuda_encoded_matvec(
    yvex_backend *backend, const unsigned char *resident_encoded,
    unsigned long long encoded_bytes, unsigned int qtype,
    unsigned long long row_count, unsigned long long row_width,
    unsigned long long row_bytes, const yvex_device_tensor *input,
    yvex_device_tensor *output, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    unsigned long long device_address = 0ull, input_bytes, output_bytes;
    CUdeviceptr encoded_ptr, input_ptr, output_ptr, status = 0ull, quantized = 0ull;
    unsigned long long start_row = 0ull, launches = 0ull;
    int output_bf16 = 0, host_status = 0, rc, cleanup_rc, q8_path, q8_input = 0;
    int forensic_numeric = 0;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !resident_encoded || !encoded_bytes || !row_count ||
        !row_width || !row_bytes || row_count > UINT_MAX || !facts ||
        !yvex_core_u64_mul(row_width, sizeof(float), &input_bytes) ||
        !yvex_core_u64_mul(row_count, sizeof(float), &output_bytes) ||
        row_count > ULLONG_MAX / row_bytes || row_count * row_bytes != encoded_bytes ||
        !backend_tensor_owner_is(backend, input) || input->dtype != YVEX_DTYPE_F32 ||
        input->bytes < input_bytes || !backend_tensor_owner_is(backend, output) ||
        output->dtype != YVEX_DTYPE_F32 || output->bytes < output_bytes ||
        yvex_backend_resident_resolve(backend, resident_encoded, encoded_bytes,
                                      &device_address) != YVEX_BACKEND_RESIDENT_HIT) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.encoded-matvec",
                       "resident encoded matvec geometry or ownership is incompatible");
        return YVEX_ERR_FORMAT;
    }
    output->is_written = 0;
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                      "cuda.encoded-matvec", err);
    if (rc == YVEX_OK) rc = yvex_cuda_set_current(backend, "cuda.encoded-matvec", err);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &status, sizeof(int), NULL, 1,
                                     "cuda.encoded-matvec.status", NULL, err);
    encoded_ptr = (CUdeviceptr)device_address;
    input_ptr = (CUdeviceptr)input->data;
    output_ptr = (CUdeviceptr)output->data;
    q8_path = row_width % 256ull == 0ull &&
              (qtype == YVEX_GGUF_QTYPE_IQ2_XXS || qtype == YVEX_GGUF_QTYPE_Q2_K ||
               qtype == YVEX_GGUF_QTYPE_Q8_0);
    if (rc == YVEX_OK && q8_path) {
        unsigned long long blocks = row_width / 256ull;
        if (blocks > UINT_MAX || blocks > ULLONG_MAX / 292ull) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.encoded-matvec",
                           "Q8 activation workspace exceeds launch bounds");
            rc = YVEX_ERR_BOUNDS;
        } else
            rc = yvex_cuda_work_allocate(&work, &quantized, (size_t)(blocks * 292ull),
                                         NULL, 0, "cuda.encoded-matvec.q8", NULL, err);
        if (rc == YVEX_OK) {
            unsigned long long one = 1ull;
            void *params[] = {&quantized, &input_ptr, &row_width, &one, &status};
            rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                  state->q8_quantize_function, (unsigned int)blocks,
                                  CUDA_QTYPE_MATVEC_BLOCK, 0u, params,
                                  "cuda.encoded-matvec.q8", err);
            if (rc == YVEX_OK) launches = 1ull;
        }
    }
    if (rc == YVEX_OK) {
        void *params[] = {&encoded_ptr, &row_bytes, &row_width, &start_row,
                          &row_count, &qtype, &input_ptr, &q8_input,
                          &forensic_numeric, &output_ptr,
                          &output_bf16, &status};
        void *q8_params[] = {&encoded_ptr, &row_bytes, &row_width, &start_row,
                             &row_count, &qtype, &quantized, &q8_input,
                             &forensic_numeric, &output_ptr,
                             &output_bf16, &status};
        unsigned int grid = (unsigned int)((row_count + CUDA_QTYPE_MATVEC_ROWS - 1ull) /
                                           CUDA_QTYPE_MATVEC_ROWS);
        q8_input = q8_path;
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, state->qtype_matvec_function,
            grid, CUDA_QTYPE_MATVEC_BLOCK, 0u, q8_path ? q8_params : params,
            "cuda.encoded-matvec.launch", err);
        if (rc == YVEX_OK) launches++;
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   "cuda.encoded-matvec.sync", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemcpyDtoH_v2(&host_status, status, sizeof(host_status)),
            "cuda.encoded-matvec.status", err);
    if (rc == YVEX_OK && host_status) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.encoded-matvec",
                       "encoded CUDA projection produced invalid numerics");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc == YVEX_OK) {
        output->is_written = 1;
        facts->d2h_bytes = sizeof(host_status);
        facts->kernel_launches = launches;
        facts->download_count = 1ull;
        facts->device_synchronizations = 1ull;
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_backend_cuda_argmax_f32(
    yvex_backend *backend, const yvex_device_tensor *values,
    unsigned long long count, unsigned int *selected_token, float *selected_value,
    unsigned long long *tie_count, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    CUdeviceptr device_token = 0ull, device_value = 0ull, device_ties = 0ull;
    CUdeviceptr status = 0ull, input;
    unsigned int token = 0u;
    unsigned long long ties = 0ull;
    float value = 0.0f;
    int host_status = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (selected_token) *selected_token = 0u;
    if (selected_value) *selected_value = 0.0f;
    if (tie_count) *tie_count = 0ull;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !selected_token || !selected_value || !tie_count || !count ||
        !facts || count > UINT_MAX || !backend_tensor_owner_is(backend, values) ||
        values->dtype != YVEX_DTYPE_F32 || !values->is_written ||
        values->bytes < count * sizeof(float)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.argmax",
                       "device argmax geometry or ownership is incompatible");
        return YVEX_ERR_FORMAT;
    }
    rc = yvex_cuda_require_capability(
        backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, "cuda.argmax", err);
    if (rc == YVEX_OK) rc = yvex_cuda_set_current(backend, "cuda.argmax", err);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
#define ALLOC(field, type, stage)                                                   \
    if (rc == YVEX_OK)                                                             \
        rc = yvex_cuda_work_allocate(&work, &field, sizeof(type), NULL, 1, stage,  \
                                     NULL, err)
    ALLOC(device_token, unsigned int, "cuda.argmax.token");
    ALLOC(device_value, float, "cuda.argmax.value");
    ALLOC(device_ties, unsigned long long, "cuda.argmax.ties");
    ALLOC(status, int, "cuda.argmax.status");
#undef ALLOC
    input = (CUdeviceptr)values->data;
    if (rc == YVEX_OK) {
        void *params[] = {&input, &count, &device_token, &device_value,
                          &device_ties, &status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->argmax_f32_function, 1u, 128u, 0u, params,
            "cuda.argmax.launch", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            "cuda.argmax.sync", err);
#define READ(target, source, stage)                                                \
    if (rc == YVEX_OK)                                                            \
        rc = yvex_cuda_status(                                                    \
            &state->driver,                                                       \
            state->driver.cuMemcpyDtoH_v2(&target, source, sizeof(target)), stage, err)
    READ(host_status, status, "cuda.argmax.status");
    READ(token, device_token, "cuda.argmax.token");
    READ(value, device_value, "cuda.argmax.value");
    READ(ties, device_ties, "cuda.argmax.ties");
#undef READ
    if (rc == YVEX_OK && (host_status || token >= count || !isfinite(value) || !ties)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.argmax",
                       "device argmax produced invalid bounded facts");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc == YVEX_OK) {
        *selected_token = token;
        *selected_value = value;
        *tie_count = ties;
        facts->d2h_bytes = sizeof(host_status) + sizeof(token) +
                           sizeof(value) + sizeof(ties);
        facts->kernel_launches = 1ull;
        facts->download_count = 4ull;
        facts->device_synchronizations = 1ull;
        yvex_error_clear(err);
    }
    return rc;
}
/*
 * Executes one encoded row dot directly on CUDA. Host inputs are borrowed,
 * device temporaries are always released, and no decoded tensor is retained. */

int yvex_cuda_quant_row_dot(yvex_backend *backend,
                            unsigned int qtype,
                            const unsigned char *encoded,
                            size_t encoded_bytes,
                            const float *vector,
                            unsigned long long elements,
                            float *out,
                            yvex_quant_failure *failure,
                            yvex_error *err)
{
    const yvex_quant_numeric_capability *capability =
        yvex_quant_numeric_capability_at(qtype);
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work;
    yvex_gguf_qtype_storage_result storage;
    unsigned long long dims[1];
    CUdeviceptr device_encoded = 0u;
    CUdeviceptr device_vector = 0u;
    CUdeviceptr device_output = 0u;
    size_t vector_bytes;
    void *params[5];
    const char *copy_failure;
    yvex_error cleanup_error;
    yvex_error primary_error;
    int rc;
    int cleanup_rc;
    memset(&work, 0, sizeof(work));
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (!backend || !state || !encoded || !vector || !out || !elements ||
        ((uintptr_t)vector % _Alignof(float)) != 0u ||
        ((uintptr_t)out % _Alignof(float)) != 0u ||
        ((qtype == YVEX_GGUF_QTYPE_F32 ||
          qtype == YVEX_GGUF_QTYPE_I32) &&
         ((uintptr_t)encoded % 4u) != 0u) ||
        ((qtype == YVEX_GGUF_QTYPE_F16 ||
          qtype == YVEX_GGUF_QTYPE_BF16) &&
         ((uintptr_t)encoded % 2u) != 0u)) {
        return cuda_quant_fail(
            failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, qtype, 1u, 0u,
            err, YVEX_ERR_INVALID_ARG,
            "CUDA encoded row, aligned vector/result, and backend are required");
    }
    rc = backend_dispatch_admit(backend, "cuda.quant.row_dot", err);
    if (rc != YVEX_OK) return rc;
    if (!capability || !capability->dedicated_cuda_compute_available) {
        return cuda_quant_fail(
            failure, YVEX_QUANT_FAILURE_CUDA_COMPUTE_UNAVAILABLE, qtype,
            1u, 0u, err, YVEX_ERR_UNSUPPORTED,
            "qtype has no dedicated CUDA numeric contract");
    }
    dims[0] = elements;
    if (yvex_gguf_qtype_tensor_storage(qtype, dims, 1u, &storage) !=
            YVEX_GGUF_QTYPE_STORAGE_OK) {
        return cuda_quant_fail(
            failure, YVEX_QUANT_FAILURE_ROW_DIVISIBILITY, qtype,
            yvex_gguf_qtype_geometry_find(qtype)->block_size, elements,
            err, YVEX_ERR_BOUNDS,
            "CUDA qtype row does not satisfy canonical block geometry");
    }
    if (storage.total_bytes != encoded_bytes ||
        elements > SIZE_MAX / sizeof(float)) {
        return cuda_quant_fail(
            failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, qtype,
            storage.total_bytes, encoded_bytes, err, YVEX_ERR_BOUNDS,
            "CUDA encoded row byte geometry is inconsistent");
    }
    vector_bytes = (size_t)elements * sizeof(float);
    rc = yvex_cuda_deferred_release_drain(backend, err);
    if (rc != YVEX_OK) {
        return cuda_quant_fail(
            failure, YVEX_QUANT_FAILURE_CLEANUP, qtype, 0u, 0u, err, rc,
            "prior CUDA qtype temporary cleanup remains incomplete");
    }
    rc = yvex_cuda_require_capability(
        backend, YVEX_BACKEND_VARIANT_QTYPE_ROW_DOT,
        "cuda.quant.row_dot.capability", err);
    if (rc != YVEX_OK) {
        return cuda_quant_fail(
            failure, YVEX_QUANT_FAILURE_CUDA_COMPUTE_UNAVAILABLE, qtype,
            1u, 0u, err, rc,
            "CUDA qtype row-dot kernel is not admitted");
    }
    rc = yvex_cuda_set_current(backend, "cuda.quant.row_dot.context", err);
    if (rc != YVEX_OK) goto execution_failure;
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_QTYPE_ROW_DOT;
    work.raw_only = 1;
    rc = yvex_cuda_work_allocate(&work, &device_encoded, encoded_bytes, NULL, 0,
                                 "cuda.quant.row_dot.alloc_encoded", NULL, err);
    if (rc != YVEX_OK) goto execution_failure;
    rc = yvex_cuda_work_allocate(&work, &device_vector, vector_bytes, NULL, 0,
                                 "cuda.quant.row_dot.alloc_vector", NULL, err);
    if (rc != YVEX_OK) goto execution_failure;
    rc = yvex_cuda_work_allocate(&work, &device_output, sizeof(float), NULL, 1,
                                 "cuda.quant.row_dot.alloc_output", NULL, err);
    if (rc != YVEX_OK) goto execution_failure;
    copy_failure = getenv("YVEX_TEST_CUDA_QTYPE_COPY_FAILURE");
    if (copy_failure && strcmp(copy_failure, "input") == 0) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.quant.row_dot.copy_input",
                       "injected CUDA qtype input copy failure");
        rc = YVEX_ERR_BACKEND;
        goto execution_failure;
    }
    rc = yvex_cuda_status(
        &state->driver,
        state->driver.cuMemcpyHtoD_v2(device_encoded, encoded, encoded_bytes),
        "cuda.quant.row_dot.copy_encoded", err);
    if (rc != YVEX_OK) goto execution_failure;
    rc = yvex_cuda_status(
        &state->driver,
        state->driver.cuMemcpyHtoD_v2(device_vector, vector, vector_bytes),
        "cuda.quant.row_dot.copy_vector", err);
    if (rc != YVEX_OK) goto execution_failure;
    params[0] = &device_encoded;
    params[1] = &device_vector;
    params[2] = &elements;
    params[3] = &qtype;
    params[4] = &device_output;
    rc = yvex_cuda_launch(
        backend, YVEX_BACKEND_VARIANT_QTYPE_ROW_DOT,
        state->qtype_row_dot_function, 1u, 1u, 0u, params,
        "cuda.quant.row_dot.launch", err);
    if (rc != YVEX_OK) goto execution_failure;
    rc = yvex_cuda_synchronize(
        backend, YVEX_BACKEND_VARIANT_QTYPE_ROW_DOT,
        "cuda.quant.row_dot.synchronize", err);
    if (rc != YVEX_OK) goto execution_failure;
    if (copy_failure && strcmp(copy_failure, "output") == 0) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.quant.row_dot.copy_output",
                       "injected CUDA qtype output copy failure");
        rc = YVEX_ERR_BACKEND;
        goto execution_failure;
    }
    rc = yvex_cuda_status(
        &state->driver,
        state->driver.cuMemcpyDtoH_v2(out, device_output, sizeof(float)),
        "cuda.quant.row_dot.copy_output", err);
    if (rc != YVEX_OK) goto execution_failure;
    cleanup_rc = yvex_cuda_work_cleanup(&work, err);
    if (cleanup_rc != YVEX_OK) {
        return cuda_quant_fail(
            failure, YVEX_QUANT_FAILURE_CLEANUP, qtype, 3u, 0u, err,
            cleanup_rc, "CUDA qtype temporary cleanup failed");
    }
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
execution_failure:
    if (err)
        primary_error = *err;
    else
        yvex_error_clear(&primary_error);
    yvex_error_clear(&cleanup_error);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup_error);
    (void)cleanup_rc;
    rc = cuda_quant_fail(
        failure, YVEX_QUANT_FAILURE_WORKER, qtype, 1u, 0u, err,
        rc == YVEX_OK ? YVEX_ERR_BACKEND : rc,
        "CUDA qtype encoded-row execution failed");
    if (err)
        *err = primary_error;
    return rc;
}
