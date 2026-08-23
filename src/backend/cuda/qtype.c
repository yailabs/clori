/*
 * Project canonical qtype capability and execute bounded encoded row-dot proofs on CUDA.
 *
 * Qtype compute support must be present in TRACK.QUANT and proven by the dedicated generated-PTX
 * row-dot variant before this owner reports it. CUDA qtype facts do not make CUDA runtime
 * generation available.
 */
#include "src/backend/cuda/private.h"
#include <yvex/internal/quant_numeric.h>
#include <dlfcn.h>
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
#define CUDA_BLAS_LT_DESC_TRANSA 3
#define CUDA_BLAS_LT_DESC_TRANSB 4
#define CUDA_BLAS_LT_DESC_EPILOGUE 7
#define CUDA_BLAS_LT_DESC_BIAS_POINTER 8
#define CUDA_BLAS_LT_EPILOGUE_BIAS 4
#define CUDA_BLAS_LT_PREF_WORKSPACE 1
#define CUDA_BLAS_LT_PREF_ALIGNMENT_A 5
#define CUDA_BLAS_LT_PREF_ALIGNMENT_B 6
#define CUDA_BLAS_LT_PREF_ALIGNMENT_C 7
#define CUDA_BLAS_LT_PREF_ALIGNMENT_D 8
#define CUDA_BLAS_LT_ALGO_TILE 1
#define CUDA_BLAS_LT_ALGO_SPLIT_K 2
#define CUDA_BLAS_LT_ALGO_REDUCTION 3
#define CUDA_BLAS_LT_ALGO_STAGES 6
#define CUDA_BLAS_LT_TILE_32X32 11u
#define CUDA_BLAS_LT_TILE_128X32 16u
#define CUDA_BLAS_LT_STAGES_8X5 28u

typedef struct { uint64_t data[8]; } cuda_blas_lt_algo;
typedef struct {
    cuda_blas_lt_algo algo;
    size_t workspace_size;
    int state;
    float waves;
    int reserved[4];
} cuda_blas_lt_result;
typedef struct {
    void *library, *handle;
    int (*create)(void **), (*destroy)(void *);
    int (*operation_create)(void **, int, int), (*operation_destroy)(void *);
    int (*operation_set)(void *, int, const void *, size_t);
    int (*layout_create)(void **, int, uint64_t, uint64_t, int64_t);
    int (*layout_destroy)(void *), (*preference_create)(void **);
    int (*preference_destroy)(void *), (*preference_set)(void *, int, const void *, size_t);
    int (*heuristic)(void *, void *, void *, void *, void *, void *, void *,
                     int, cuda_blas_lt_result *, int *);
    int (*algo_init)(void *, int, int, int, int, int, int, int, cuda_blas_lt_algo *);
    int (*algo_set)(cuda_blas_lt_algo *, int, const void *, size_t);
    int (*algo_check)(void *, void *, void *, void *, void *, void *,
                      const cuda_blas_lt_algo *, cuda_blas_lt_result *);
    int (*matmul)(void *, void *, const void *, const void *, void *,
                  const void *, void *, const void *, void *, const void *,
                  void *, void *, const cuda_blas_lt_algo *, void *, size_t, CUstream);
} cuda_blas_lt;

static unsigned int cuda_blas_lt_alignment(CUdeviceptr pointer)
{
    unsigned int alignment = 256u;
    while (alignment > 1u && pointer % alignment) alignment >>= 1u;
    return alignment;
}

static int cuda_blas_lt_open(cuda_blas_lt *lt, yvex_error *err)
{
    if (!lt) return YVEX_ERR_INVALID_ARG;
    memset(lt, 0, sizeof(*lt));
    lt->library = dlopen("libcublasLt.so.13", RTLD_NOW | RTLD_LOCAL);
    if (!lt->library) lt->library = dlopen("libcublasLt.so", RTLD_NOW | RTLD_LOCAL);
    if (!lt->library) goto unavailable;
#define LOAD_LT(field, symbol) *(void **)(&lt->field) = dlsym(lt->library, symbol)
    LOAD_LT(create, "cublasLtCreate");
    LOAD_LT(destroy, "cublasLtDestroy");
    LOAD_LT(operation_create, "cublasLtMatmulDescCreate");
    LOAD_LT(operation_destroy, "cublasLtMatmulDescDestroy");
    LOAD_LT(operation_set, "cublasLtMatmulDescSetAttribute");
    LOAD_LT(layout_create, "cublasLtMatrixLayoutCreate");
    LOAD_LT(layout_destroy, "cublasLtMatrixLayoutDestroy");
    LOAD_LT(preference_create, "cublasLtMatmulPreferenceCreate");
    LOAD_LT(preference_destroy, "cublasLtMatmulPreferenceDestroy");
    LOAD_LT(preference_set, "cublasLtMatmulPreferenceSetAttribute");
    LOAD_LT(heuristic, "cublasLtMatmulAlgoGetHeuristic");
    LOAD_LT(algo_init, "cublasLtMatmulAlgoInit");
    LOAD_LT(algo_set, "cublasLtMatmulAlgoConfigSetAttribute");
    LOAD_LT(algo_check, "cublasLtMatmulAlgoCheck");
    LOAD_LT(matmul, "cublasLtMatmul");
#undef LOAD_LT
    if (!lt->create || !lt->destroy || !lt->operation_create ||
        !lt->operation_destroy || !lt->operation_set || !lt->layout_create ||
        !lt->layout_destroy || !lt->preference_create || !lt->preference_destroy ||
        !lt->preference_set || !lt->heuristic || !lt->algo_init || !lt->algo_set ||
        !lt->algo_check || !lt->matmul || lt->create(&lt->handle) != 0)
        goto unavailable;
    return YVEX_OK;
unavailable:
    if (lt->handle && lt->destroy) (void)lt->destroy(lt->handle);
    if (lt->library) dlclose(lt->library);
    memset(lt, 0, sizeof(*lt));
    yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.encoded-linear.lt",
                   "an admitted cuBLASLt bias epilogue is required");
    return YVEX_ERR_UNSUPPORTED;
}

static int cuda_blas_lt_close(cuda_blas_lt *lt, int rc, yvex_error *err)
{
    int cleanup = 0;
    if (!lt) return rc;
    if (lt->handle && lt->destroy) cleanup = lt->destroy(lt->handle);
    if (lt->library) dlclose(lt->library);
    memset(lt, 0, sizeof(*lt));
    if (rc == YVEX_OK && cleanup != 0) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.encoded-linear.lt-close",
                       "cuBLASLt handle cleanup failed");
        return YVEX_ERR_BACKEND;
    }
    return rc;
}

static int cuda_blas_lt_bias(
    cuda_blas_lt *lt, CUdeviceptr weight, CUdeviceptr input, CUdeviceptr bias,
    CUdeviceptr output, unsigned long long rows, unsigned long long columns,
    unsigned long long batches, int dtype, size_t preference_workspace,
    CUdeviceptr workspace, size_t workspace_capacity,
    const yvex_backend_linear_numeric_policy *numeric_policy,
    CUstream stream, yvex_error *err)
{
    void *operation = NULL, *weight_layout = NULL, *input_layout = NULL;
    void *output_layout = NULL, *preference = NULL;
    cuda_blas_lt_result result;
    float alpha = 1.0f, beta = 0.0f;
    unsigned int alignment_a, alignment_b, alignment_c;
    int transa = CUDA_BLAS_OP_T, transb = CUDA_BLAS_OP_N, split_k = 0, algorithm_id = 0;
    int epilogue = CUDA_BLAS_LT_EPILOGUE_BIAS, returned = 0, status = 0;
    unsigned int tile = 0u, reduction = 0u, stages = 0u;
    if (!lt || !lt->handle || (dtype != CUDA_BLAS_R_16BF && dtype != CUDA_BLAS_R_32F) ||
        rows > INT_MAX || columns > INT_MAX || batches > INT_MAX)
        return YVEX_ERR_BOUNDS;
    if (numeric_policy) {
        split_k = (int)numeric_policy->split_k;
        reduction = (unsigned int)numeric_policy->reduction;
        if (numeric_policy->tile_rows == 32u &&
            numeric_policy->reduction == YVEX_BACKEND_LINEAR_REDUCTION_INPLACE) {
            algorithm_id = 10;
            tile = CUDA_BLAS_LT_TILE_32X32;
        } else if (numeric_policy->tile_rows == 128u &&
                   numeric_policy->reduction ==
                       YVEX_BACKEND_LINEAR_REDUCTION_COMPUTE_TYPE) {
            algorithm_id = 20;
            tile = CUDA_BLAS_LT_TILE_128X32;
            stages = CUDA_BLAS_LT_STAGES_8X5;
        } else if (numeric_policy->tile_rows || numeric_policy->split_k ||
                   numeric_policy->reduction != YVEX_BACKEND_LINEAR_REDUCTION_DEFAULT) {
            yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.encoded-linear.lt-policy",
                           "the requested F32 reduction policy is not admitted");
            return YVEX_ERR_UNSUPPORTED;
        }
    }
    memset(&result, 0, sizeof(result));
    alignment_a = cuda_blas_lt_alignment(weight);
    alignment_b = cuda_blas_lt_alignment(input);
    alignment_c = cuda_blas_lt_alignment(output);
    status = lt->operation_create(&operation, CUDA_BLAS_COMPUTE_32F, CUDA_BLAS_R_32F);
    if (!status) status = lt->operation_set(operation, CUDA_BLAS_LT_DESC_TRANSA,
                                             &transa, sizeof(transa));
    if (!status) status = lt->operation_set(operation, CUDA_BLAS_LT_DESC_TRANSB,
                                             &transb, sizeof(transb));
    if (!status) status = lt->operation_set(operation, CUDA_BLAS_LT_DESC_EPILOGUE,
                                             &epilogue, sizeof(epilogue));
    if (!status) status = lt->operation_set(operation, CUDA_BLAS_LT_DESC_BIAS_POINTER,
                                             &bias, sizeof(bias));
    if (!status) status = lt->layout_create(&weight_layout, dtype,
                                             columns, rows, (int64_t)columns);
    if (!status) status = lt->layout_create(&input_layout, dtype,
                                             columns, batches, (int64_t)columns);
    if (!status) status = lt->layout_create(&output_layout, dtype,
                                             rows, batches, (int64_t)rows);
    if (!status) status = lt->preference_create(&preference);
    if (!status) status = lt->preference_set(preference, CUDA_BLAS_LT_PREF_WORKSPACE,
                                              &preference_workspace,
                                              sizeof(preference_workspace));
    if (!status) status = lt->preference_set(preference, CUDA_BLAS_LT_PREF_ALIGNMENT_A,
                                              &alignment_a, sizeof(alignment_a));
    if (!status) status = lt->preference_set(preference, CUDA_BLAS_LT_PREF_ALIGNMENT_B,
                                              &alignment_b, sizeof(alignment_b));
    if (!status) status = lt->preference_set(preference, CUDA_BLAS_LT_PREF_ALIGNMENT_C,
                                              &alignment_c, sizeof(alignment_c));
    if (!status) status = lt->preference_set(preference, CUDA_BLAS_LT_PREF_ALIGNMENT_D,
                                              &alignment_c, sizeof(alignment_c));
    if (!status && algorithm_id) {
        status = lt->algo_init(lt->handle, CUDA_BLAS_COMPUTE_32F, CUDA_BLAS_R_32F,
                               dtype, dtype, dtype, dtype, algorithm_id, &result.algo);
        if (!status)
            status = lt->algo_set(&result.algo, CUDA_BLAS_LT_ALGO_TILE,
                                  &tile, sizeof(tile));
        if (!status)
            status = lt->algo_set(&result.algo, CUDA_BLAS_LT_ALGO_SPLIT_K,
                                  &split_k, sizeof(split_k));
        if (!status)
            status = lt->algo_set(&result.algo, CUDA_BLAS_LT_ALGO_REDUCTION,
                                  &reduction, sizeof(reduction));
        if (!status && stages)
            status = lt->algo_set(&result.algo, CUDA_BLAS_LT_ALGO_STAGES,
                                  &stages, sizeof(stages));
        if (!status)
            status = lt->algo_check(lt->handle, operation, weight_layout, input_layout,
                                    output_layout, output_layout, &result.algo, &result);
        returned = status ? 0 : 1;
    } else if (!status) {
        status = lt->heuristic(lt->handle, operation, weight_layout, input_layout,
                               output_layout, output_layout, preference, 1,
                               &result, &returned);
    }
    if (!status && (returned != 1 || result.state != 0 ||
                    result.workspace_size > workspace_capacity)) status = 1;
    if (!status)
        status = lt->matmul(lt->handle, operation, &alpha,
                            (const void *)(uintptr_t)weight, weight_layout,
                            (const void *)(uintptr_t)input, input_layout, &beta,
                            (void *)(uintptr_t)output, output_layout,
                            (void *)(uintptr_t)output, output_layout, &result.algo,
                            (void *)(uintptr_t)workspace, workspace_capacity, stream);
    if (preference) (void)lt->preference_destroy(preference);
    if (output_layout) (void)lt->layout_destroy(output_layout);
    if (input_layout) (void)lt->layout_destroy(input_layout);
    if (weight_layout) (void)lt->layout_destroy(weight_layout);
    if (operation) (void)lt->operation_destroy(operation);
    if (status) {
        yvex_error_setf(err, YVEX_ERR_BACKEND, "cuda.encoded-linear.lt",
                        "cuBLASLt bias epilogue failed with status %d", status);
        return YVEX_ERR_BACKEND;
    }
    return YVEX_OK;
}

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

int yvex_cuda_qtype_tensorcore_geometry(
    unsigned long long rows, unsigned long long input_rows,
    unsigned int *grid, unsigned int *block)
{
    unsigned long long row_tiles, input_tiles, input_groups, blocks;
    unsigned int warps;
    if (!rows || !input_rows || !grid || !block ||
        rows > ULLONG_MAX - 15ull || input_rows > ULLONG_MAX - 15ull) return 0;
    row_tiles = (rows + 15ull) / 16ull;
    input_tiles = (input_rows + 15ull) / 16ull;
    warps = input_tiles > 4ull ? 4u : (unsigned int)input_tiles;
    input_groups = (input_tiles + warps - 1ull) / warps;
    if (!yvex_core_u64_mul(row_tiles, input_groups, &blocks) ||
        !blocks || blocks > UINT_MAX) return 0;
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
    rc = yvex_cuda_blas_bind_launch_stream(backend, "cuda.encoded-gemm.stream", err);
    if (rc == YVEX_OK)
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
                                      yvex_cuda_launch_stream(backend))
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

int yvex_backend_cuda_encoded_linear_bf16(
    yvex_backend *backend, const unsigned char *resident_weight,
    unsigned long long weight_bytes, const unsigned char *resident_bias,
    unsigned long long bias_bytes, unsigned long long output_width,
    unsigned long long input_width, unsigned long long input_rows,
    const yvex_device_tensor *input, yvex_device_tensor *output,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    cuda_blas_lt lt = {0};
    CUdeviceptr resident_weight_device = 0ull, resident_bias_device = 0ull;
    CUdeviceptr weight = 0ull, bias = 0ull, packed_input = 0ull, packed_output = 0ull;
    CUdeviceptr status_device = 0ull, input_ptr, output_ptr;
    unsigned long long input_values, output_values, packed_input_bytes, packed_output_bytes;
    unsigned long long input_grid, output_grid, expected_weight, expected_bias;
    int status_host = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !facts || !resident_weight || !resident_bias || !output_width ||
        !input_width || !input_rows ||
        !yvex_core_u64_mul(output_width, input_width, &expected_weight) ||
        !yvex_core_u64_mul(expected_weight, sizeof(unsigned short), &expected_weight) ||
        !yvex_core_u64_mul(output_width, sizeof(unsigned short), &expected_bias) ||
        weight_bytes != expected_weight || bias_bytes != expected_bias ||
        !yvex_core_u64_mul(input_rows, input_width, &input_values) ||
        !yvex_core_u64_mul(input_rows, output_width, &output_values) ||
        !yvex_core_u64_mul(input_values, sizeof(unsigned short), &packed_input_bytes) ||
        !yvex_core_u64_mul(output_values, sizeof(unsigned short), &packed_output_bytes) ||
        !yvex_core_u64_add(input_values, CUDA_QTYPE_MATVEC_BLOCK - 1ull, &input_grid) ||
        !yvex_core_u64_add(output_values, CUDA_QTYPE_MATVEC_BLOCK - 1ull, &output_grid) ||
        input_grid / CUDA_QTYPE_MATVEC_BLOCK > UINT_MAX ||
        output_grid / CUDA_QTYPE_MATVEC_BLOCK > UINT_MAX ||
        !backend_tensor_owner_is(backend, input) || input->dtype != YVEX_DTYPE_F32 ||
        input->bytes < input_values * sizeof(float) ||
        !backend_tensor_owner_is(backend, output) || output->dtype != YVEX_DTYPE_F32 ||
        output->bytes < output_values * sizeof(float) ||
        yvex_backend_resident_resolve(backend, resident_weight, weight_bytes,
                                      &resident_weight_device) != YVEX_BACKEND_RESIDENT_HIT ||
        yvex_backend_resident_resolve(backend, resident_bias, bias_bytes,
                                      &resident_bias_device) != YVEX_BACKEND_RESIDENT_HIT ||
        weight_bytes > SIZE_MAX || bias_bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.encoded-linear-bf16",
                       "resident BF16 weight, bias, and bounded F32 tensors are required");
        return YVEX_ERR_FORMAT;
    }
    output->is_written = 0;
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                      "cuda.encoded-linear-bf16", err);
    if (rc == YVEX_OK) rc = yvex_cuda_set_current(backend, "cuda.encoded-linear-bf16", err);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    work.raw_only = 1;
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &weight, (size_t)weight_bytes,
                                     resident_weight, 0,
                                     "cuda.encoded-linear-bf16.weight", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &bias, (size_t)bias_bytes,
                                     resident_bias, 0,
                                     "cuda.encoded-linear-bf16.bias", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &packed_input, (size_t)packed_input_bytes,
                                     NULL, 0, "cuda.encoded-linear-bf16.input", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &packed_output, (size_t)packed_output_bytes,
                                     NULL, 1, "cuda.encoded-linear-bf16.output", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &status_device, sizeof(status_host),
                                     NULL, 1, "cuda.encoded-linear-bf16.status", NULL, err);
    input_ptr = input ? (CUdeviceptr)input->data : 0ull;
    output_ptr = output ? (CUdeviceptr)output->data : 0ull;
    if (rc == YVEX_OK) {
        void *parameters[] = {&input_ptr, &packed_input, &input_values, &status_device};
        rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                              state->bf16_pack_function,
                              (unsigned int)(input_grid / CUDA_QTYPE_MATVEC_BLOCK),
                              CUDA_QTYPE_MATVEC_BLOCK, 0u, parameters,
                              "cuda.encoded-linear-bf16.pack", err);
    }
    if (rc == YVEX_OK) rc = cuda_blas_lt_open(&lt, err);
    if (rc == YVEX_OK)
        rc = cuda_blas_lt_bias(&lt, weight, packed_input, bias, packed_output,
                               output_width, input_width, input_rows, CUDA_BLAS_R_16BF,
                               256u * 1024u * 1024u, 0ull, 0u,
                               NULL,
                               yvex_cuda_launch_stream(backend), err);
    if (rc == YVEX_OK) {
        void *parameters[] = {&packed_output, &output_ptr, &output_values};
        rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                              state->bf16_unpack_function,
                              (unsigned int)(output_grid / CUDA_QTYPE_MATVEC_BLOCK),
                              CUDA_QTYPE_MATVEC_BLOCK, 0u, parameters,
                              "cuda.encoded-linear-bf16.unpack", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   "cuda.encoded-linear-bf16.sync", err);
    rc = cuda_blas_lt_close(&lt, rc, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemcpyDtoH_v2(&status_host, status_device, sizeof(status_host)),
            "cuda.encoded-linear-bf16.status", err);
    if (rc == YVEX_OK && status_host) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.encoded-linear-bf16",
                       "BF16 linear input contains invalid numerics");
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
    facts->d2h_bytes = sizeof(status_host);
    facts->h2d_bytes = weight_bytes + bias_bytes;
    facts->kernel_launches = 3ull;
    facts->download_count = 1ull;
    facts->device_synchronizations = 1ull;
    facts->active_weight_bytes = weight_bytes + bias_bytes;
    facts->activation_bytes = input_values * sizeof(float) + output_values * sizeof(float);
    facts->temporary_bytes = weight_bytes + bias_bytes + packed_input_bytes +
                             packed_output_bytes + sizeof(status_host);
    facts->compulsory_memory_facts_available = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_backend_cuda_encoded_linear_f32(
    yvex_backend *backend, const unsigned char *resident_weight,
    unsigned long long weight_bytes, const unsigned char *resident_bias,
    unsigned long long bias_bytes, unsigned long long output_width,
    unsigned long long input_width, unsigned long long input_rows,
    const yvex_device_tensor *input, yvex_device_tensor *output,
    const yvex_backend_linear_numeric_policy *numeric_policy,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    enum { WORKSPACE_BYTES = 1024u * 1024u };
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    cuda_blas_lt lt = {0};
    CUdeviceptr resident_weight_device = 0ull, resident_bias_device = 0ull;
    CUdeviceptr weight = 0ull, bias = 0ull, workspace = 0ull;
    unsigned long long input_values, output_values, expected_weight, expected_bias;
    int rc, cleanup_rc;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !facts || !numeric_policy || !resident_weight || !resident_bias ||
        !output_width || numeric_policy->split_k == 0u || numeric_policy->split_k > INT_MAX ||
        !input_width || !input_rows ||
        !yvex_core_u64_mul(output_width, input_width, &expected_weight) ||
        !yvex_core_u64_mul(expected_weight, sizeof(float), &expected_weight) ||
        !yvex_core_u64_mul(output_width, sizeof(float), &expected_bias) ||
        weight_bytes != expected_weight || bias_bytes != expected_bias ||
        !yvex_core_u64_mul(input_rows, input_width, &input_values) ||
        !yvex_core_u64_mul(input_rows, output_width, &output_values) ||
        !backend_tensor_owner_is(backend, input) || !input->is_written ||
        input->dtype != YVEX_DTYPE_F32 || input->bytes < input_values * sizeof(float) ||
        !backend_tensor_owner_is(backend, output) || output->dtype != YVEX_DTYPE_F32 ||
        output->bytes < output_values * sizeof(float) ||
        yvex_backend_resident_resolve(backend, resident_weight, weight_bytes,
                                      &resident_weight_device) != YVEX_BACKEND_RESIDENT_HIT ||
        yvex_backend_resident_resolve(backend, resident_bias, bias_bytes,
                                      &resident_bias_device) != YVEX_BACKEND_RESIDENT_HIT ||
        weight_bytes > SIZE_MAX || bias_bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.encoded-linear-f32",
                       "resident F32 weight, bias, and bounded F32 tensors are required");
        return YVEX_ERR_FORMAT;
    }
    output->is_written = 0;
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                      "cuda.encoded-linear-f32", err);
    if (rc == YVEX_OK) rc = yvex_cuda_set_current(backend, "cuda.encoded-linear-f32", err);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    work.raw_only = 1;
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &weight, (size_t)weight_bytes,
                                     resident_weight, 0, "cuda.encoded-linear-f32.weight",
                                     NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &bias, (size_t)bias_bytes,
                                     resident_bias, 0, "cuda.encoded-linear-f32.bias", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(&work, &workspace, WORKSPACE_BYTES, NULL, 1,
                                     "cuda.encoded-linear-f32.workspace", NULL, err);
    if (rc == YVEX_OK) rc = cuda_blas_lt_open(&lt, err);
    if (rc == YVEX_OK)
        rc = cuda_blas_lt_bias(&lt, weight, (CUdeviceptr)input->data, bias,
                               (CUdeviceptr)output->data, output_width, input_width,
                               input_rows, CUDA_BLAS_R_32F, WORKSPACE_BYTES,
                               workspace, WORKSPACE_BYTES,
                               numeric_policy,
                               yvex_cuda_launch_stream(backend), err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   "cuda.encoded-linear-f32.sync", err);
    rc = cuda_blas_lt_close(&lt, rc, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc != YVEX_OK) return rc;
    output->is_written = 1;
    facts->h2d_bytes = weight_bytes + bias_bytes;
    facts->kernel_launches = 1ull;
    facts->upload_count = 2ull;
    facts->device_synchronizations = 1ull;
    facts->active_weight_bytes = weight_bytes + bias_bytes;
    facts->activation_bytes = (input_values + output_values) * sizeof(float);
    facts->temporary_bytes = weight_bytes + bias_bytes + WORKSPACE_BYTES;
    facts->tensor_core_launches = 1ull;
    facts->compulsory_memory_facts_available = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Execute one admitted row-major F32 matrix batch through cuBLAS. */
static int cuda_blas_f32_projection(
    yvex_backend *backend, yvex_cuda_backend_state *state,
    CUdeviceptr encoded, unsigned long long encoded_bytes,
    unsigned long long row_count, unsigned long long row_width,
    unsigned long long input_rows, const yvex_device_tensor *input,
    const yvex_device_tensor *additive, yvex_device_tensor *output,
    unsigned long long activation_bytes, yvex_backend_cuda_operation_facts *facts,
    yvex_error *err)
{
    const float alpha = 1.0f, beta = additive ? 1.0f : 0.0f;
    CUdeviceptr input_ptr = (CUdeviceptr)input->data;
    CUdeviceptr output_ptr = (CUdeviceptr)output->data;
    unsigned long long output_bytes;
    int rc = YVEX_OK, blas_status;

    if (!state->blas.ready || row_count > INT_MAX || row_width > INT_MAX ||
        input_rows > INT_MAX ||
        !yvex_core_u64_mul(row_count, input_rows, &output_bytes) ||
        !yvex_core_u64_mul(output_bytes, sizeof(float), &output_bytes) ||
        output_bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.encoded-gemm",
                       "F32 projection exceeds cuBLAS integer geometry");
        return YVEX_ERR_BOUNDS;
    }
    rc = yvex_cuda_blas_bind_launch_stream(backend, "cuda.encoded-gemm.stream", err);
    if (rc == YVEX_OK && additive) {
        CUdeviceptr additive_ptr = (CUdeviceptr)additive->data;
        int driver_status = state->driver.cuMemcpyDtoDAsync_v2
                                ? state->driver.cuMemcpyDtoDAsync_v2(
                                      output_ptr, additive_ptr, (size_t)output_bytes,
                                      yvex_cuda_launch_stream(backend))
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
              (const void *)(uintptr_t)encoded, CUDA_BLAS_R_32F, (int)row_width,
              (const void *)(uintptr_t)input_ptr, CUDA_BLAS_R_32F, (int)row_width,
              &beta, (void *)(uintptr_t)output_ptr, CUDA_BLAS_R_32F, (int)row_count,
              CUDA_BLAS_COMPUTE_32F, CUDA_BLAS_DEFAULT)
        : 0;
    if (rc == YVEX_OK && blas_status != 0) {
        yvex_error_setf(err, YVEX_ERR_BACKEND, "cuda.encoded-gemm",
                        "cuBLAS F32 projection failed with status %d", blas_status);
        rc = YVEX_ERR_BACKEND;
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            "cuda.encoded-gemm.sync", err);
    if (rc != YVEX_OK) return rc;
    output->is_written = 1;
    facts->d2d_bytes = additive ? output_bytes : 0ull;
    facts->kernel_launches = 1ull;
    facts->device_synchronizations = 1ull;
    facts->active_weight_bytes = encoded_bytes;
    facts->activation_bytes = activation_bytes;
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
    unsigned long long group_count = 1ull, group_rows = row_count;
    int output_bf16 = 0, host_status = 0, rc, cleanup_rc, q8_path, q8_input = 0;
    int tensorcore_path;
    int block_row = 0;
    int forensic_numeric = 0, split_input = input_tail != NULL;
    unsigned int matvec_grid, matvec_block, tensorcore_grid = 0u, tensorcore_block = 0u;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (activation_q8 != 0 && activation_q8 != 1) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.encoded-matvec.activation",
                       "activation Q8 policy must be explicitly disabled or enabled");
        return YVEX_ERR_INVALID_ARG;
    }
    q8_path = activation_q8 && !split_input && row_width % 256ull == 0ull &&
              yvex_cuda_q8_activation_eligible(qtype);
    tensorcore_path = q8_path && state && state->qtype_tensorcore_rows_function &&
                      cuda_qtype_tensorcore_eligible(input_rows);
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
    if (tensorcore_path && !yvex_cuda_qtype_tensorcore_geometry(
                               row_count, input_rows,
                               &tensorcore_grid, &tensorcore_block)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.encoded-matvec.tensorcore",
                       "Tensor Core row-batch grid exceeds launch bounds");
        return YVEX_ERR_BOUNDS;
    }
    output->is_written = 0;
    rc = yvex_cuda_require_capability(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                      "cuda.encoded-matvec", err);
    if (rc == YVEX_OK) rc = yvex_cuda_set_current(backend, "cuda.encoded-matvec", err);
    encoded_ptr = (CUdeviceptr)device_address;
    if (rc == YVEX_OK && !split_input &&
        qtype == YVEX_GGUF_QTYPE_BF16 && state->blas.ready)
        return cuda_blas_bf16_projection(
            backend, state, encoded_ptr, encoded_bytes, row_count, row_width,
            input_rows, input, additive, output, activation_bytes, facts, err);
    if (rc == YVEX_OK && !split_input &&
        qtype == YVEX_GGUF_QTYPE_F32 && state->blas.ready)
        return cuda_blas_f32_projection(
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
            void *params[] = {&quantized, &input_ptr, &row_width, &input_rows,
                              &group_count, &row_width, &status};
            rc = yvex_cuda_launch(backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                  state->q8_quantize_function, (unsigned int)quantize_tasks,
                                  CUDA_QTYPE_MATVEC_BLOCK, 0u, params,
                                  "cuda.encoded-matvec.q8", err);
            if (rc == YVEX_OK) launches = 1ull;
        }
    }
    if (rc == YVEX_OK) {
        void *params[] = {&encoded_ptr, &row_bytes, &row_width, &start_row,
                          &row_count, &input_rows, &qtype, &input_ptr, &row_width, &q8_input,
                          &block_row, &forensic_numeric, &additive_ptr, &output_ptr,
                          &row_count, &output_bf16, &status};
        void *q8_params[] = {&encoded_ptr, &row_bytes, &row_width, &start_row,
                             &row_count, &input_rows, &qtype, &quantized, &row_width, &q8_input,
                             &block_row, &forensic_numeric, &additive_ptr, &output_ptr,
                             &row_count, &output_bf16, &status};
        void *tensorcore_params[] = {
            &encoded_ptr, &row_bytes, &row_width, &start_row, &row_count,
            &input_rows, &group_count, &group_rows, &qtype, &quantized,
            &additive_ptr, &output_ptr, &output_bf16, &status};
        void *split_params[] = {&encoded_ptr, &row_bytes, &row_width, &start_row,
                                &row_count, &input_rows, &qtype, &input_ptr,
                                &input_tail_ptr, &input_head_width, &additive_ptr,
                                &output_ptr, &output_bf16, &status};
        q8_input = q8_path;
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            tensorcore_path ? state->qtype_tensorcore_rows_function
                            : split_input ? state->qtype_split_matvec_function
                                          : state->qtype_matvec_function,
            tensorcore_path ? tensorcore_grid : matvec_grid,
            tensorcore_path ? tensorcore_block : matvec_block, 0u,
            tensorcore_path ? tensorcore_params
                            : split_input ? split_params : q8_path ? q8_params : params,
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
        facts->tensor_core_launches = tensorcore_path ? 1ull : 0ull;
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
