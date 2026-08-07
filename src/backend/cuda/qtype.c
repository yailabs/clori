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
#define CUDA_BLAS_OP_N 0
#define CUDA_BLAS_OP_T 1
#define CUDA_BLAS_R_32F 0
#define CUDA_BLAS_R_16BF 14
#define CUDA_BLAS_COMPUTE_32F 68
#define CUDA_BLAS_DEFAULT -1

/* Wide, narrow F32 matrices need one block per row/input pair to expose enough
 * independent reduction work; ordinary encoded rows remain warp-owned so
 * their blockwise numerical order and shared activation reuse do not change. */
int yvex_cuda_qtype_matvec_geometry(
    unsigned long long rows, unsigned long long row_width,
    unsigned long long input_rows, unsigned int qtype,
    int block_row_eligible, unsigned int *grid, unsigned int *block,
    int *block_row)
{
    unsigned long long blocks, groups, tiles;
    unsigned int warps;
    if (!rows || !row_width || !input_rows || !grid || !block || !block_row)
        return 0;
    *block_row = 0;
    if (block_row_eligible && qtype == YVEX_GGUF_QTYPE_F32 &&
        rows <= 32ull && row_width >= 4096ull && input_rows <= 8ull) {
        if (rows > UINT_MAX / input_rows) return 0;
        *grid = (unsigned int)(rows * input_rows);
        *block = CUDA_QTYPE_MATVEC_BLOCK;
        *block_row = 1;
        return 1;
    }
    if (input_rows <= 8ull) {
        groups = 8ull / input_rows;
        blocks = (rows + groups - 1ull) / groups;
        warps = (unsigned int)(groups * input_rows);
    } else {
        tiles = (input_rows + 7ull) / 8ull;
        if (rows > ULLONG_MAX / tiles) return 0;
        blocks = rows * tiles;
        warps = 8u;
    }
    if (!blocks || blocks > UINT_MAX) return 0;
    *grid = (unsigned int)blocks;
    *block = warps * 32u;
    return 1;
}

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

/* Execute the source-faithful BF16 row-major projection through one mixed GEMM. */
static int cuda_blas_bf16_projection(
    yvex_backend *backend, yvex_cuda_backend_state *state,
    CUdeviceptr encoded, unsigned long long encoded_bytes,
    unsigned long long row_count, unsigned long long row_width,
    unsigned long long input_rows, const yvex_device_tensor *input,
    const yvex_device_tensor *additive, yvex_device_tensor *output,
    unsigned long long activation_bytes, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    const float alpha = 1.0f, beta = additive ? 1.0f : 0.0f;
    yvex_cuda_work work = {0};
    CUdeviceptr input_ptr = (CUdeviceptr)input->data;
    CUdeviceptr output_ptr = (CUdeviceptr)output->data;
    CUdeviceptr packed = 0ull, device_status = 0ull;
    unsigned long long input_elements, packed_bytes, output_bytes, grid;
    int host_status = 0, rc = YVEX_OK, cleanup_rc, blas_status;
    yvex_error cleanup;

    if (!state->blas.ready || row_count > INT_MAX || row_width > INT_MAX ||
        input_rows > INT_MAX ||
        !yvex_core_u64_mul(row_width, input_rows, &input_elements) ||
        !yvex_core_u64_mul(input_elements, sizeof(unsigned short), &packed_bytes) ||
        !yvex_core_u64_mul(row_count, input_rows, &output_bytes) ||
        !yvex_core_u64_mul(output_bytes, sizeof(float), &output_bytes) ||
        !yvex_core_u64_add(input_elements, CUDA_QTYPE_MATVEC_BLOCK - 1ull, &grid) ||
        grid / CUDA_QTYPE_MATVEC_BLOCK > UINT_MAX || packed_bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.encoded-gemm",
                       "BF16 projection exceeds cuBLAS integer geometry");
        return YVEX_ERR_BOUNDS;
    }
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    rc = yvex_cuda_work_allocate(&work, &packed, (size_t)packed_bytes, NULL, 0,
                                 "cuda.encoded-gemm.input", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &device_status, sizeof(int), NULL, 1,
                                     "cuda.encoded-gemm.status", NULL, err);
    if (rc == YVEX_OK) {
        void *params[] = {&input_ptr, &packed, &input_elements, &device_status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->bf16_pack_function,
            (unsigned int)(grid / CUDA_QTYPE_MATVEC_BLOCK),
            CUDA_QTYPE_MATVEC_BLOCK, 0u, params,
            "cuda.encoded-gemm.pack", err);
    }
    if (rc == YVEX_OK && additive) {
        CUdeviceptr additive_ptr = (CUdeviceptr)additive->data;
        int driver_status = state->driver.cuMemcpyDtoDAsync_v2
                                ? state->driver.cuMemcpyDtoDAsync_v2(
                                      output_ptr, additive_ptr, (size_t)output_bytes,
                                      state->execution_stream)
                                : state->driver.cuMemcpyDtoD_v2(
                                      output_ptr, additive_ptr, (size_t)output_bytes);
        if (driver_status != YVEX_CUDA_SUCCESS)
            rc = yvex_cuda_status(&state->driver, driver_status,
                                  "cuda.encoded-gemm.additive", err);
    }
    blas_status = rc == YVEX_OK
        ? state->blas.gemm_ex(
              state->blas.handle, CUDA_BLAS_OP_T, CUDA_BLAS_OP_N,
              (int)row_count, (int)input_rows, (int)row_width, &alpha,
              (const void *)(uintptr_t)encoded, CUDA_BLAS_R_16BF, (int)row_width,
              (const void *)(uintptr_t)packed, CUDA_BLAS_R_16BF, (int)row_width,
              &beta, (void *)(uintptr_t)output_ptr, CUDA_BLAS_R_32F, (int)row_count,
              CUDA_BLAS_COMPUTE_32F, CUDA_BLAS_DEFAULT)
        : 0;
    if (rc == YVEX_OK && blas_status != 0) {
        yvex_error_setf(err, YVEX_ERR_BACKEND, "cuda.encoded-gemm",
                        "cuBLAS BF16 projection failed with status %d", blas_status);
        rc = YVEX_ERR_BACKEND;
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            "cuda.encoded-gemm.sync", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemcpyDtoH_v2(
                &host_status, device_status, sizeof(host_status)),
            "cuda.encoded-gemm.status", err);
    if (rc == YVEX_OK && host_status) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.encoded-gemm",
                       "BF16 projection input contains invalid numerics");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc != YVEX_OK) return rc;
    output->is_written = 1;
    facts->d2h_bytes = sizeof(host_status);
    facts->d2d_bytes = additive ? output_bytes : 0ull;
    facts->kernel_launches = 2ull;
    facts->download_count = 1ull;
    facts->device_synchronizations = 1ull;
    facts->active_weight_bytes = encoded_bytes;
    facts->activation_bytes = activation_bytes;
    facts->temporary_bytes = packed_bytes + sizeof(host_status);
    facts->compulsory_memory_facts_available = 1;
    yvex_error_clear(err);
    return YVEX_OK;
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
    unsigned long long row_bytes, unsigned long long input_rows,
    const yvex_device_tensor *input, const yvex_device_tensor *input_tail,
    unsigned long long input_head_width, const yvex_device_tensor *additive,
    yvex_device_tensor *output, int activation_q8,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    unsigned long long device_address = 0ull, input_bytes, output_bytes, activation_bytes;
    unsigned long long input_elements, input_head_elements, input_tail_elements;
    unsigned long long input_head_bytes, input_tail_bytes, output_elements;
    unsigned long long temporary_bytes = sizeof(int), q8_workspace_bytes = 0ull;
    CUdeviceptr encoded_ptr, input_ptr, input_tail_ptr = 0ull, additive_ptr = 0ull, output_ptr;
    CUdeviceptr status = 0ull, quantized = 0ull;
    unsigned long long start_row = 0ull, launches = 0ull;
    int output_bf16 = 0, host_status = 0, rc, cleanup_rc, q8_path, q8_input = 0;
    int block_row = 0;
    int forensic_numeric = 0, split_input = input_tail != NULL;
    unsigned int matvec_grid, matvec_block;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (activation_q8 != 0 && activation_q8 != 1) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.encoded-matvec.activation",
                       "activation Q8 policy must be explicitly disabled or enabled");
        return YVEX_ERR_INVALID_ARG;
    }
    q8_path = activation_q8 && !split_input && row_width % 256ull == 0ull &&
              yvex_cuda_q8_activation_eligible(qtype);
    if (!state || !resident_encoded || !encoded_bytes || !row_count || !input_rows ||
        !row_width || !row_bytes || !facts || split_input != (input_head_width != 0ull) ||
        (split_input && input_head_width >= row_width) ||
        !yvex_core_u64_mul(input_rows, row_width, &input_elements) ||
        !yvex_core_u64_mul(input_rows, split_input ? input_head_width : row_width,
                           &input_head_elements) ||
        !yvex_core_u64_mul(input_rows, split_input ? row_width - input_head_width : 0ull,
                           &input_tail_elements) ||
        !yvex_core_u64_mul(input_rows, row_count, &output_elements) ||
        !yvex_core_u64_mul(input_elements, sizeof(float), &input_bytes) ||
        !yvex_core_u64_mul(input_head_elements, sizeof(float), &input_head_bytes) ||
        !yvex_core_u64_mul(input_tail_elements, sizeof(float), &input_tail_bytes) ||
        !yvex_core_u64_mul(output_elements, sizeof(float), &output_bytes) ||
        !yvex_core_u64_add(input_bytes, output_bytes, &activation_bytes) ||
        (additive && !yvex_core_u64_add(activation_bytes, output_bytes,
                                        &activation_bytes)) ||
        !yvex_cuda_qtype_matvec_geometry(
            row_count, row_width, input_rows, qtype, !split_input,
            &matvec_grid, &matvec_block, &block_row) ||
        row_count > ULLONG_MAX / row_bytes || row_count * row_bytes != encoded_bytes ||
        !backend_tensor_owner_is(backend, input) || !input->is_written ||
        input->dtype != YVEX_DTYPE_F32 || input->bytes < input_head_bytes ||
        (split_input && (!backend_tensor_owner_is(backend, input_tail) ||
                         !input_tail->is_written || input_tail->dtype != YVEX_DTYPE_F32 ||
                         input_tail->bytes < input_tail_bytes || input_tail == output)) ||
        !backend_tensor_owner_is(backend, output) ||
        output->dtype != YVEX_DTYPE_F32 || output->bytes < output_bytes ||
        (additive && (!backend_tensor_owner_is(backend, additive) ||
                      additive->dtype != YVEX_DTYPE_F32 || !additive->is_written ||
                      additive->bytes < output_bytes || additive == output)) ||
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
    encoded_ptr = (CUdeviceptr)device_address;
    if (rc == YVEX_OK && !split_input && input_rows > 1ull &&
        qtype == YVEX_GGUF_QTYPE_BF16 && state->blas.ready)
        return cuda_blas_bf16_projection(
            backend, state, encoded_ptr, encoded_bytes, row_count, row_width,
            input_rows, input, additive, output, activation_bytes, facts, err);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &status, sizeof(int), NULL, 1,
                                     "cuda.encoded-matvec.status", NULL, err);
    input_ptr = (CUdeviceptr)input->data;
    if (input_tail) input_tail_ptr = (CUdeviceptr)input_tail->data;
    if (additive) additive_ptr = (CUdeviceptr)additive->data;
    output_ptr = (CUdeviceptr)output->data;
    if (rc == YVEX_OK && q8_path) {
        unsigned long long blocks = row_width / 256ull, quantize_tasks;
        if (!yvex_core_u64_mul(blocks, input_rows, &quantize_tasks) ||
            quantize_tasks > UINT_MAX ||
            !yvex_core_u64_mul(quantize_tasks, 292ull, &q8_workspace_bytes) ||
            q8_workspace_bytes > SIZE_MAX ||
            !yvex_core_u64_add(temporary_bytes, q8_workspace_bytes, &temporary_bytes)) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.encoded-matvec",
                           "Q8 activation workspace exceeds launch bounds");
            rc = YVEX_ERR_BOUNDS;
        } else
            rc = yvex_cuda_work_allocate(&work, &quantized, (size_t)q8_workspace_bytes,
                                         NULL, 0, "cuda.encoded-matvec.q8", NULL, err);
        if (rc == YVEX_OK) {
            void *params[] = {&quantized, &input_ptr, &row_width, &input_rows, &status};
            rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                  state->q8_quantize_function, (unsigned int)quantize_tasks,
                                  CUDA_QTYPE_MATVEC_BLOCK, 0u, params,
                                  "cuda.encoded-matvec.q8", err);
            if (rc == YVEX_OK) launches = 1ull;
        }
    }
    if (rc == YVEX_OK) {
        void *params[] = {&encoded_ptr, &row_bytes, &row_width, &start_row,
                          &row_count, &input_rows, &qtype, &input_ptr, &q8_input,
                          &block_row, &forensic_numeric, &additive_ptr, &output_ptr,
                          &output_bf16, &status};
        void *q8_params[] = {&encoded_ptr, &row_bytes, &row_width, &start_row,
                             &row_count, &input_rows, &qtype, &quantized, &q8_input,
                             &block_row, &forensic_numeric, &additive_ptr, &output_ptr,
                             &output_bf16, &status};
        void *split_params[] = {&encoded_ptr, &row_bytes, &row_width, &start_row,
                                &row_count, &input_rows, &qtype, &input_ptr,
                                &input_tail_ptr, &input_head_width, &additive_ptr,
                                &output_ptr, &output_bf16, &status};
        q8_input = q8_path;
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            split_input ? state->qtype_split_matvec_function : state->qtype_matvec_function,
            matvec_grid, matvec_block, 0u,
            split_input ? split_params : q8_path ? q8_params : params,
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
        facts->active_weight_bytes = encoded_bytes;
        facts->activation_bytes = activation_bytes;
        facts->temporary_bytes = temporary_bytes;
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}

/*
 * Decode selected rows from one admitted resident matrix into a backend-owned F32 view.
 * Row identifiers are bounded host control facts; encoded values never leave device residency.
 */
int yvex_backend_cuda_encoded_gather(
    yvex_backend *backend, const unsigned char *resident_encoded,
    unsigned long long encoded_bytes, unsigned int qtype,
    unsigned long long row_count, unsigned long long row_width,
    unsigned long long row_bytes, const unsigned int *row_ids,
    unsigned long long selected_rows, yvex_device_tensor *output,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    const yvex_gguf_qtype_geometry *geometry = yvex_gguf_qtype_geometry_find(qtype);
    const yvex_quant_numeric_capability *capability =
        yvex_quant_numeric_capability_at(qtype);
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    unsigned long long device_address = 0ull, id_bytes, output_bytes;
    unsigned long long output_elements, launch_elements, active_weight_bytes;
    CUdeviceptr encoded_ptr, device_ids = 0ull, output_ptr, status = 0ull;
    unsigned long long index;
    int host_status = 0, rc, cleanup_rc;
    yvex_error cleanup;

    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !geometry || !capability || !capability->dedicated_cuda_compute_available ||
        !geometry->block_size || !geometry->bytes_per_block ||
        !resident_encoded || !encoded_bytes || !row_count ||
        !row_width || !row_bytes || !row_ids || !selected_rows || !facts ||
        row_width % geometry->block_size != 0ull ||
        row_width / geometry->block_size > ULLONG_MAX / geometry->bytes_per_block ||
        row_width / geometry->block_size * geometry->bytes_per_block != row_bytes ||
        row_count > ULLONG_MAX / row_bytes || row_count * row_bytes != encoded_bytes ||
        !yvex_core_u64_mul(selected_rows, sizeof(*row_ids), &id_bytes) ||
        !yvex_core_u64_mul(selected_rows, row_width, &output_elements) ||
        !yvex_core_u64_mul(output_elements, sizeof(float), &output_bytes) ||
        !yvex_core_u64_add(output_elements, CUDA_QTYPE_MATVEC_BLOCK - 1ull,
                           &launch_elements) ||
        launch_elements / CUDA_QTYPE_MATVEC_BLOCK > UINT_MAX || id_bytes > SIZE_MAX ||
        !yvex_core_u64_mul(selected_rows, row_bytes, &active_weight_bytes) ||
        !backend_tensor_owner_is(backend, output) || output->dtype != YVEX_DTYPE_F32 ||
        output->bytes < output_bytes ||
        yvex_backend_resident_resolve(backend, resident_encoded, encoded_bytes,
                                      &device_address) != YVEX_BACKEND_RESIDENT_HIT) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.encoded-gather",
                       "resident encoded gather geometry or ownership is incompatible");
        return YVEX_ERR_FORMAT;
    }
    output->is_written = 0;
    for (index = 0ull; index < selected_rows; ++index) {
        if ((unsigned long long)row_ids[index] >= row_count) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.encoded-gather",
                           "encoded gather row identifier exceeds the resident matrix");
            return YVEX_ERR_BOUNDS;
        }
    }

    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                      "cuda.encoded-gather", err);
    if (rc == YVEX_OK) rc = yvex_cuda_set_current(backend, "cuda.encoded-gather", err);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &device_ids, (size_t)id_bytes, row_ids, 0,
                                     "cuda.encoded-gather.ids", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &status, sizeof(int), NULL, 1,
                                     "cuda.encoded-gather.status", NULL, err);
    encoded_ptr = (CUdeviceptr)device_address;
    output_ptr = (CUdeviceptr)output->data;
    if (rc == YVEX_OK) {
        unsigned int grid = (unsigned int)(launch_elements / CUDA_QTYPE_MATVEC_BLOCK);
        void *params[] = {&encoded_ptr, &row_bytes, &row_width, &row_count,
                          &device_ids, &selected_rows, &qtype, &output_ptr, &status};
        rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                              state->qtype_gather_function, grid,
                              CUDA_QTYPE_MATVEC_BLOCK, 0u, params,
                              "cuda.encoded-gather.launch", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   "cuda.encoded-gather.sync", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemcpyDtoH_v2(&host_status, status, sizeof(host_status)),
            "cuda.encoded-gather.status", err);
    if (rc == YVEX_OK && host_status) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.encoded-gather",
                       "encoded CUDA gather produced invalid numerics");
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
        facts->h2d_bytes = id_bytes;
        facts->d2h_bytes = sizeof(host_status);
        facts->kernel_launches = 1ull;
        facts->upload_count = 1ull;
        facts->download_count = 1ull;
        facts->device_synchronizations = 1ull;
        facts->active_weight_bytes = active_weight_bytes;
        facts->activation_bytes = output_bytes;
        facts->temporary_bytes = id_bytes + sizeof(host_status);
        facts->compulsory_memory_facts_available = 1;
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
