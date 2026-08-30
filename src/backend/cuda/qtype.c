/*
 * Project canonical qtype capability and execute bounded encoded row-dot proofs on CUDA.
 *
 * Qtype compute support must be present in TRACK.QUANT and proven by the dedicated generated-PTX
 * row-dot variant before this owner reports it. CUDA qtype facts do not make CUDA runtime
 * generation available.
 */
#include "src/backend/cuda/private.h"
#include "src/backend/cuda/transformer_ops.h"
#include <yvex/internal/component.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/transformer.h>
#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

static int cuda_encoded_matvec(
    yvex_backend *, const unsigned char *, unsigned long long, unsigned int,
    unsigned long long, unsigned long long, unsigned long long, unsigned long long,
    const yvex_device_tensor *, const yvex_device_tensor *, unsigned long long,
    const yvex_device_tensor *, yvex_device_tensor *, int,
    yvex_backend_operation_facts *, yvex_error *);
static int cuda_encoded_gather(
    yvex_backend *, const unsigned char *, unsigned long long, unsigned int,
    unsigned long long, unsigned long long, unsigned long long,
    const unsigned int *, unsigned long long, yvex_device_tensor *,
    yvex_backend_operation_facts *, yvex_error *);
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
#define CUDA_BLAS_LT_REDUCTION_INPLACE 1u
#define CUDA_BLAS_LT_REDUCTION_COMPUTE_TYPE 2u
#define CUDA_DENSE_WORKSPACE_LIMIT (64ull * 1024ull * 1024ull)
#define CUDA_DENSE_ALIGNMENT 256ull
#define CUDA_DENSE_POINTER_ALIGNMENT 16u
#define CUDA_DENSE_HEURISTIC_COUNT 8

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

struct yvex_transformer_linear_executable {
    yvex_backend *backend;
    cuda_blas_lt lt;
    void *operation, *weight_layout, *input_layout, *output_layout;
    cuda_blas_lt_algo algorithm;
    yvex_transformer_linear_requirement requirement;
    yvex_transformer_linear_executable_summary summary;
    size_t algorithm_workspace;
    int in_use;
};

typedef struct {
    yvex_transformer_linear_operation operation;
    unsigned long long input_width, output_width, workspace_bytes;
    int algorithm_id, split_k, compute_major, compute_minor;
    unsigned int tile, reduction, stages;
} cuda_linear_implementation;

/* These exact algorithms are CUDA implementation facts. The generic plan seals the operation,
 * numerical contract, geometry, backend, and workspace; this owner alone maps that contract to
 * cuBLASLt algorithm attributes and the device generation on which they were qualified. */
static const cuda_linear_implementation cuda_linear_implementations[] = {
    {YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_VIDEO_OUTPUT,
     5376ull, 96ull, 1024ull * 1024ull, 10, 10, 12, 1,
     CUDA_BLAS_LT_TILE_32X32, CUDA_BLAS_LT_REDUCTION_INPLACE, 0u},
    {YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_AUDIO_OUTPUT,
     5376ull, 32ull, 1024ull * 1024ull, 20, 3, 12, 1,
     CUDA_BLAS_LT_TILE_128X32, CUDA_BLAS_LT_REDUCTION_COMPUTE_TYPE,
    CUDA_BLAS_LT_STAGES_8X5},
};

static const yvex_backend_encoded_operations cuda_encoded_operations = {
    .matvec = cuda_encoded_matvec,
    .gather = cuda_encoded_gather,
};

const yvex_backend_encoded_operations *yvex_cuda_encoded_operations_get(
    const yvex_backend *backend)
{
    return backend && yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA
               ? &cuda_encoded_operations
               : NULL;
}

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
    const cuda_linear_implementation *implementation,
    CUstream stream, int *physical_geometry_refused, yvex_error *err)
{
    void *operation = NULL, *weight_layout = NULL, *input_layout = NULL;
    void *output_layout = NULL, *preference = NULL;
    cuda_blas_lt_result result;
    float alpha = 1.0f, beta = 0.0f;
    unsigned int alignment_a, alignment_b, alignment_c;
    int transa = CUDA_BLAS_OP_T, transb = CUDA_BLAS_OP_N, split_k = 0, algorithm_id = 0;
    int epilogue = CUDA_BLAS_LT_EPILOGUE_BIAS, returned = 0, status = 0;
    unsigned int tile = 0u, reduction = 0u, stages = 0u;
    if (physical_geometry_refused) *physical_geometry_refused = 0;
    if (!lt || !lt->handle || (dtype != CUDA_BLAS_R_16BF && dtype != CUDA_BLAS_R_32F) ||
        rows > INT_MAX || columns > INT_MAX || batches > INT_MAX)
        return YVEX_ERR_BOUNDS;
    if (implementation) {
        algorithm_id = implementation->algorithm_id;
        split_k = implementation->split_k;
        tile = implementation->tile;
        reduction = implementation->reduction;
        stages = implementation->stages;
    } else if (dtype == CUDA_BLAS_R_32F) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.encoded-linear.lt-policy",
                       "F32 linear execution requires one compiler-sealed physical plan");
        return YVEX_ERR_UNSUPPORTED;
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
        if (status && physical_geometry_refused) *physical_geometry_refused = 1;
        returned = status ? 0 : 1;
    } else if (!status) {
        status = lt->heuristic(lt->handle, operation, weight_layout, input_layout,
                               output_layout, output_layout, preference, 1,
                               &result, &returned);
    }
    if (!status && (returned != 1 || result.state != 0 ||
                    result.workspace_size > workspace_capacity)) {
        if (implementation && physical_geometry_refused) *physical_geometry_refused = 1;
        status = 1;
    }
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

/* A sealed split-K algorithm may require workspace proportional to the independent batch-row
 * extent even though each output element reduces over the same immutable K dimension. Retry a
 * refused batch geometry at smaller row extents and preserve the exact per-element reduction,
 * bias epilogue, and source order. Completed chunks never overlap and share one ordered stream. */
static int cuda_blas_lt_bias_f32_batches(
    cuda_blas_lt *lt, CUdeviceptr weight, CUdeviceptr input, CUdeviceptr bias,
    CUdeviceptr output, unsigned long long rows, unsigned long long columns,
    unsigned long long batches, size_t preference_workspace, CUdeviceptr workspace,
    size_t workspace_capacity,
    const cuda_linear_implementation *implementation,
    CUstream stream, unsigned long long *launches, yvex_error *err)
{
    unsigned long long completed = 0ull, chunk = batches;
    unsigned long long input_row_bytes, output_row_bytes;
    int rc;
    if (launches) *launches = 0ull;
    if (!lt || !implementation || !launches ||
        !yvex_core_u64_mul(columns, sizeof(float), &input_row_bytes) ||
        !yvex_core_u64_mul(rows, sizeof(float), &output_row_bytes)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.encoded-linear.lt-batches",
                       "bounded F32 batch-row geometry is required");
        return YVEX_ERR_BOUNDS;
    }
    while (completed < batches) {
        unsigned long long remaining = batches - completed;
        unsigned long long input_offset, output_offset;
        int geometry_refused = 0;
        if (chunk > remaining) chunk = remaining;
        if (!yvex_core_u64_mul(completed, input_row_bytes, &input_offset) ||
            !yvex_core_u64_mul(completed, output_row_bytes, &output_offset) ||
            input_offset > UINT64_MAX - input || output_offset > UINT64_MAX - output) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.encoded-linear.lt-batches",
                           "F32 batch-row device offsets overflowed");
            return YVEX_ERR_BOUNDS;
        }
        rc = cuda_blas_lt_bias(
            lt, weight, input + input_offset, bias, output + output_offset,
            rows, columns, chunk, CUDA_BLAS_R_32F, preference_workspace,
            workspace, workspace_capacity, implementation, stream,
            &geometry_refused, err);
        if (rc == YVEX_OK) {
            completed += chunk;
            (*launches)++;
            continue;
        }
        if (!geometry_refused || chunk == 1ull) return rc;
        chunk /= 2ull;
        yvex_error_clear(err);
    }
    return YVEX_OK;
}

static int cuda_linear_physical_validate(
    yvex_backend *backend, const yvex_transformer_linear_physical_plan *plan,
    unsigned long long input_width, unsigned long long output_width,
    const cuda_linear_implementation **selected, yvex_error *err)
{
    yvex_backend_device_info device;
    const cuda_linear_implementation *match = NULL;
    size_t index;
    int rc;
    if (selected) *selected = NULL;
    if (!backend || !plan || plan->input_width != input_width ||
        plan->output_width != output_width || plan->backend != YVEX_BACKEND_KIND_CUDA ||
        plan->implementation != YVEX_TRANSFORMER_LINEAR_IMPLEMENTATION_DEVICE_F32_BIAS ||
        !selected) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.encoded-linear-f32.physical",
                       "compiled linear operation and runtime geometry do not match");
        return YVEX_ERR_FORMAT;
    }
    rc = yvex_transformer_linear_physical_validate(plan, err);
    if (rc != YVEX_OK) return rc;
    for (index = 0u;
         index < sizeof(cuda_linear_implementations) / sizeof(cuda_linear_implementations[0]);
         ++index) {
        const cuda_linear_implementation *candidate = cuda_linear_implementations + index;
        if (candidate->operation == plan->operation &&
            candidate->input_width == plan->input_width &&
            candidate->output_width == plan->output_width &&
            candidate->workspace_bytes == plan->workspace_bytes) {
            if (match) {
                yvex_error_set(err, YVEX_ERR_STATE, "cuda.encoded-linear-f32.physical",
                               "linear operation has ambiguous CUDA implementations");
                return YVEX_ERR_STATE;
            }
            match = candidate;
        }
    }
    if (!match || match->workspace_bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.encoded-linear-f32.physical",
                       "compiled linear implementation is unavailable in this CUDA executor");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = yvex_backend_get_device_info(backend, &device, err);
    if (rc != YVEX_OK) return rc;
    if (device.compute_capability_major != match->compute_major ||
        device.compute_capability_minor != match->compute_minor) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.encoded-linear-f32.hardware",
                       "compiled linear algorithm is incompatible with this CUDA device");
        return YVEX_ERR_UNSUPPORTED;
    }
    *selected = match;
    return YVEX_OK;
}

static int cuda_dense_fault(const char *phase)
{
    const char *value = getenv("YVEX_TEST_CUDA_LINEAR_PLAN_FAILURE");
    return value && phase && strcmp(value, phase) == 0;
}

static unsigned long long cuda_dense_elapsed_ns(const struct timespec *start,
                                                 const struct timespec *finish)
{
    unsigned long long seconds, nanoseconds;
    if (!start || !finish || finish->tv_sec < start->tv_sec) return 0ull;
    seconds = (unsigned long long)(finish->tv_sec - start->tv_sec);
    nanoseconds = (unsigned long long)finish->tv_nsec;
    if (finish->tv_nsec < start->tv_nsec) {
        if (!seconds) return 0ull;
        seconds--;
        nanoseconds += 1000000000ull;
    }
    nanoseconds -= (unsigned long long)start->tv_nsec;
    return seconds <= ULLONG_MAX / 1000000000ull
               ? seconds * 1000000000ull + nanoseconds
               : 0ull;
}

static int cuda_dense_align(unsigned long long value, unsigned long long *out)
{
    unsigned long long expanded;
    if (!out || !yvex_core_u64_add(value, CUDA_DENSE_ALIGNMENT - 1ull, &expanded))
        return 0;
    *out = expanded & ~(CUDA_DENSE_ALIGNMENT - 1ull);
    return 1;
}

static int cuda_dense_workspace_geometry(
    const yvex_transformer_linear_compile_request *request,
    unsigned long long algorithm_bytes, unsigned long long *pack_bytes,
    unsigned long long *total_bytes, yvex_error *err)
{
    unsigned long long elements, packed, status, algorithm, total;
    int rc = request && request->requirement
                 ? yvex_transformer_linear_requirement_validate(request->requirement, err)
                 : YVEX_ERR_INVALID_ARG;
    if (rc == YVEX_OK &&
        (request->requirement->operation < YVEX_TRANSFORMER_LINEAR_OPERATION_MODULATION ||
         request->requirement->operation > YVEX_TRANSFORMER_LINEAR_OPERATION_DOWN)) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.dense-plan.workspace",
                       "the CUDA compiled dense plan does not implement this linear class");
        return YVEX_ERR_UNSUPPORTED;
    }
    if (rc != YVEX_OK || !request->semantic_domain || !request->semantic_domain[0] ||
        !request->input_rows || request->input_rows > INT_MAX ||
        request->requirement->input_width > INT_MAX ||
        request->requirement->output_width > INT_MAX ||
        !yvex_core_u64_mul(request->requirement->input_width, request->input_rows,
                           &elements) ||
        !yvex_core_u64_mul(elements, sizeof(unsigned short), &packed) ||
        !cuda_dense_align(packed, &packed) ||
        !cuda_dense_align(sizeof(int), &status) ||
        !cuda_dense_align(algorithm_bytes, &algorithm) ||
        !yvex_core_u64_add(packed, status, &total) ||
        !yvex_core_u64_add(total, algorithm, &total)) {
        if (rc == YVEX_OK)
            yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.dense-plan.workspace",
                           "dense plan workspace geometry is not representable");
        return rc == YVEX_OK ? YVEX_ERR_BOUNDS : rc;
    }
    if (pack_bytes) *pack_bytes = packed;
    if (total_bytes) *total_bytes = total;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int cuda_dense_identity(
    const yvex_transformer_linear_compile_request *request,
    const yvex_backend_device_info *device, char output[YVEX_SHA256_HEX_CAP])
{
    const yvex_transformer_linear_requirement *requirement = request->requirement;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.cuda.dense-executable.v1") ||
        !yvex_sha256_update_text(&hash, request->semantic_domain) ||
        !yvex_sha256_update_u64(&hash, requirement->operation) ||
        !yvex_sha256_update_u64(&hash, requirement->publication_contract) ||
        !yvex_sha256_update_u64(&hash, requirement->source_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->input_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->accumulation_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->output_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->publication_dtype) ||
        !yvex_sha256_update_u64(&hash, requirement->input_width) ||
        !yvex_sha256_update_u64(&hash, requirement->output_width) ||
        !yvex_sha256_update_u64(&hash, request->input_rows) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)device->kind) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)device->device_index) ||
        !yvex_sha256_update_u64(&hash,
                                (unsigned long long)device->compute_capability_major) ||
        !yvex_sha256_update_u64(&hash,
                                (unsigned long long)device->compute_capability_minor) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static void cuda_dense_plan_discard(yvex_transformer_linear_executable *plan)
{
    if (!plan) return;
    if (plan->output_layout) (void)plan->lt.layout_destroy(plan->output_layout);
    if (plan->input_layout) (void)plan->lt.layout_destroy(plan->input_layout);
    if (plan->weight_layout) (void)plan->lt.layout_destroy(plan->weight_layout);
    if (plan->operation) (void)plan->lt.operation_destroy(plan->operation);
    (void)cuda_blas_lt_close(&plan->lt, YVEX_OK, NULL);
    free(plan);
}

int yvex_cuda_transformer_linear_workspace_required(
    const yvex_transformer_linear_compile_request *request,
    unsigned long long *required, yvex_error *err)
{
    if (required) *required = 0ull;
    if (!required) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.dense-plan.workspace",
                       "dense plan workspace output is required");
        return YVEX_ERR_INVALID_ARG;
    }
    return cuda_dense_workspace_geometry(
        request, CUDA_DENSE_WORKSPACE_LIMIT, NULL, required, err);
}

int yvex_cuda_transformer_linear_compile(
    yvex_backend *backend, const yvex_transformer_linear_compile_request *request,
    yvex_transformer_linear_executable **out,
    yvex_transformer_linear_executable_summary *summary, yvex_error *err)
{
    yvex_transformer_linear_executable *plan = NULL;
    cuda_blas_lt_result candidates[CUDA_DENSE_HEURISTIC_COUNT];
    yvex_backend_device_info device;
    struct timespec started = {0}, finished = {0};
    void *preference = NULL;
    unsigned long long pack_bytes, total_bytes;
    size_t workspace_limit = (size_t)CUDA_DENSE_WORKSPACE_LIMIT;
    unsigned int alignment = CUDA_DENSE_POINTER_ALIGNMENT;
    int transa = CUDA_BLAS_OP_T, transb = CUDA_BLAS_OP_N;
    int returned = 0, selected = -1, status = 0, rc, index;
    if (out) *out = NULL;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!backend || !out || !summary || yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.dense-plan.compile",
                       "one CUDA backend and plan output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = cuda_dense_workspace_geometry(
        request, CUDA_DENSE_WORKSPACE_LIMIT, &pack_bytes, &total_bytes, err);
    if (rc != YVEX_OK) return rc;
    if (cuda_dense_fault("compile")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.dense-plan.compile",
                       "injected dense plan compilation failure");
        return YVEX_ERR_BACKEND;
    }
    if (cuda_dense_fault("unsupported")) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "cuda.dense-plan.compile",
                       "injected unsupported dense plan capability");
        return YVEX_ERR_UNSUPPORTED;
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &started);
    plan = calloc(1u, sizeof(*plan));
    if (!plan) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "cuda.dense-plan.compile",
                       "dense executable plan allocation failed");
        return YVEX_ERR_NOMEM;
    }
    plan->backend = backend;
    plan->requirement = *request->requirement;
    rc = yvex_backend_get_device_info(backend, &device, err);
    if (rc == YVEX_OK) rc = cuda_blas_lt_open(&plan->lt, err);
    if (rc == YVEX_OK)
        status = plan->lt.operation_create(
            &plan->operation, CUDA_BLAS_COMPUTE_32F, CUDA_BLAS_R_32F);
    if (!status && rc == YVEX_OK)
        status = plan->lt.operation_set(
            plan->operation, CUDA_BLAS_LT_DESC_TRANSA, &transa, sizeof(transa));
    if (!status && rc == YVEX_OK)
        status = plan->lt.operation_set(
            plan->operation, CUDA_BLAS_LT_DESC_TRANSB, &transb, sizeof(transb));
    if (!status && rc == YVEX_OK)
        status = plan->lt.layout_create(
            &plan->weight_layout, CUDA_BLAS_R_16BF,
            request->requirement->input_width, request->requirement->output_width,
            (int64_t)request->requirement->input_width);
    if (!status && rc == YVEX_OK)
        status = plan->lt.layout_create(
            &plan->input_layout, CUDA_BLAS_R_16BF,
            request->requirement->input_width, request->input_rows,
            (int64_t)request->requirement->input_width);
    if (!status && rc == YVEX_OK)
        status = plan->lt.layout_create(
            &plan->output_layout, CUDA_BLAS_R_32F,
            request->requirement->output_width, request->input_rows,
            (int64_t)request->requirement->output_width);
    if (!status && rc == YVEX_OK) status = plan->lt.preference_create(&preference);
    if (!status && rc == YVEX_OK)
        status = plan->lt.preference_set(
            preference, CUDA_BLAS_LT_PREF_WORKSPACE, &workspace_limit, sizeof(workspace_limit));
    if (!status && rc == YVEX_OK)
        status = plan->lt.preference_set(
            preference, CUDA_BLAS_LT_PREF_ALIGNMENT_A, &alignment, sizeof(alignment));
    if (!status && rc == YVEX_OK)
        status = plan->lt.preference_set(
            preference, CUDA_BLAS_LT_PREF_ALIGNMENT_B, &alignment, sizeof(alignment));
    if (!status && rc == YVEX_OK)
        status = plan->lt.preference_set(
            preference, CUDA_BLAS_LT_PREF_ALIGNMENT_C, &alignment, sizeof(alignment));
    if (!status && rc == YVEX_OK)
        status = plan->lt.preference_set(
            preference, CUDA_BLAS_LT_PREF_ALIGNMENT_D, &alignment, sizeof(alignment));
    memset(candidates, 0, sizeof(candidates));
    if (!status && rc == YVEX_OK)
        status = plan->lt.heuristic(
            plan->lt.handle, plan->operation, plan->weight_layout, plan->input_layout,
            plan->output_layout, plan->output_layout, preference,
            CUDA_DENSE_HEURISTIC_COUNT, candidates, &returned);
    for (index = 0; !status && index < returned; ++index)
        if (candidates[index].state == 0 &&
            candidates[index].workspace_size <= CUDA_DENSE_WORKSPACE_LIMIT) {
            selected = index;
            break;
        }
    if (preference) (void)plan->lt.preference_destroy(preference);
    if (rc == YVEX_OK && (status || selected < 0)) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "cuda.dense-plan.compile",
                        "cuBLASLt has no compatible exact BF16 plan (status %d)", status);
        rc = YVEX_ERR_UNSUPPORTED;
    }
    if (rc == YVEX_OK) {
        plan->algorithm = candidates[selected].algo;
        plan->algorithm_workspace = candidates[selected].workspace_size;
        rc = cuda_dense_workspace_geometry(
            request, plan->algorithm_workspace, &pack_bytes, &total_bytes, err);
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &finished);
    if (rc == YVEX_OK && !cuda_dense_identity(request, &device, plan->summary.identity)) {
        yvex_error_set(err, YVEX_ERR_STATE, "cuda.dense-plan.compile",
                       "dense executable identity derivation failed");
        rc = YVEX_ERR_STATE;
    }
    if (rc != YVEX_OK) {
        cuda_dense_plan_discard(plan);
        return rc;
    }
    plan->summary.schema_version = YVEX_TRANSFORMER_LINEAR_EXECUTABLE_SCHEMA_V1;
    plan->summary.input_rows = request->input_rows;
    plan->summary.workspace_bytes = total_bytes;
    plan->summary.input_pack_bytes = pack_bytes;
    plan->summary.plan_host_bytes = sizeof(*plan);
    plan->summary.preparation_nanoseconds = cuda_dense_elapsed_ns(&started, &finished);
    plan->summary.algorithm_selection_count = 1ull;
    plan->summary.accelerated_matrix = 1;
    plan->summary.exact = 1;
    *summary = plan->summary;
    *out = plan;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_transformer_linear_execute(
    yvex_backend *backend, const yvex_transformer_linear_execution_request *request,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_transformer_linear_executable *plan = request ? request->executable : NULL;
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    const yvex_component_encoded_weight *weight = request ? request->weight : NULL;
    yvex_cuda_work work = {0};
    CUdeviceptr weight_ptr = 0ull, packed = 0ull, device_status = 0ull, algorithm = 0ull;
    CUdeviceptr input_ptr, output_ptr;
    unsigned long long expected_weight, input_values, output_values, input_bytes, output_bytes;
    unsigned long long activation_bytes, pack_bytes, total_bytes;
    unsigned long long input_tasks, output_tasks;
    const float alpha = 1.0f, beta = 0.0f;
    int host_status = 0, rc = YVEX_OK, cleanup_rc, blas_status = 0, submitted = 0;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!backend || !state || !request || !plan || !facts || plan->backend != backend ||
        plan->in_use || !weight || weight->qtype != YVEX_GGUF_QTYPE_BF16 ||
        weight->row_width != plan->requirement.input_width ||
        weight->row_count != plan->requirement.output_width ||
        !yvex_core_u64_mul(weight->row_width, weight->row_count, &expected_weight) ||
        !yvex_core_u64_mul(expected_weight, sizeof(unsigned short), &expected_weight) ||
        weight->encoded_bytes != expected_weight ||
        yvex_backend_resident_resolve(backend, weight->encoded, weight->encoded_bytes,
                                      &weight_ptr) != YVEX_BACKEND_RESIDENT_HIT ||
        !backend_tensor_owner_is(backend, request->input) || !request->input->is_written ||
        request->input->dtype != YVEX_DTYPE_F32 ||
        !backend_tensor_owner_is(backend, request->output) ||
        request->output->dtype != YVEX_DTYPE_F32 ||
        !yvex_core_u64_mul(plan->summary.input_rows, plan->requirement.input_width,
                           &input_values) ||
        !yvex_core_u64_mul(plan->summary.input_rows, plan->requirement.output_width,
                           &output_values) ||
        !yvex_core_u64_mul(input_values, sizeof(float), &input_bytes) ||
        !yvex_core_u64_mul(output_values, sizeof(float), &output_bytes) ||
        !yvex_core_u64_add(input_bytes, output_bytes, &activation_bytes) ||
        request->input->bytes < input_bytes || request->output->bytes < output_bytes ||
        !yvex_core_u64_add(input_values, CUDA_QTYPE_MATVEC_BLOCK - 1ull, &input_tasks) ||
        !yvex_core_u64_add(output_values, CUDA_QTYPE_MATVEC_BLOCK - 1ull, &output_tasks) ||
        input_tasks / CUDA_QTYPE_MATVEC_BLOCK > UINT_MAX ||
        output_tasks / CUDA_QTYPE_MATVEC_BLOCK > UINT_MAX ||
        cuda_dense_workspace_geometry(
            &(yvex_transformer_linear_compile_request){
                plan->summary.identity, &plan->requirement, plan->summary.input_rows},
            plan->algorithm_workspace, &pack_bytes, &total_bytes, err) != YVEX_OK) {
        if (!err || err->code == YVEX_OK)
            yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.dense-plan.execute",
                           "dense executable bindings do not match their compiled geometry");
        return err && err->code != YVEX_OK ? err->code : YVEX_ERR_FORMAT;
    }
    request->output->is_written = 0;
    if (cuda_dense_fault("execute")) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.dense-plan.execute",
                       "injected dense plan execution failure");
        return YVEX_ERR_BACKEND;
    }
    plan->in_use = 1;
    backend_workspace_reset(backend);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    rc = yvex_cuda_set_current(backend, "cuda.dense-plan.execute", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_blas_bind_launch_stream(backend, "cuda.dense-plan.stream", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            &work, &packed, (size_t)pack_bytes, NULL, 0, "cuda.dense-plan.pack", NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_work_allocate(
            &work, &device_status, (size_t)CUDA_DENSE_ALIGNMENT, NULL, 1,
            "cuda.dense-plan.status", NULL, err);
    if (rc == YVEX_OK && plan->algorithm_workspace)
        rc = yvex_cuda_work_allocate(
            &work, &algorithm, plan->algorithm_workspace, NULL, 0,
            "cuda.dense-plan.algorithm", NULL, err);
    input_ptr = yvex_cuda_tensor_ptr(request->input);
    output_ptr = yvex_cuda_tensor_ptr(request->output);
    if (rc == YVEX_OK) {
        void *parameters[] = {&input_ptr, &packed, &input_values, &device_status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->bf16_pack_function,
            (unsigned int)(input_tasks / CUDA_QTYPE_MATVEC_BLOCK),
            CUDA_QTYPE_MATVEC_BLOCK, 0u, parameters, "cuda.dense-plan.pack", err);
        if (rc == YVEX_OK) submitted = 1;
    }
    if (rc == YVEX_OK) {
        blas_status = plan->lt.matmul(
            plan->lt.handle, plan->operation, &alpha,
            (const void *)(uintptr_t)weight_ptr, plan->weight_layout,
            (const void *)(uintptr_t)packed, plan->input_layout, &beta,
            (void *)(uintptr_t)output_ptr, plan->output_layout,
            (void *)(uintptr_t)output_ptr, plan->output_layout, &plan->algorithm,
            (void *)(uintptr_t)algorithm, plan->algorithm_workspace,
            yvex_cuda_launch_stream(backend));
        if (!blas_status) submitted = 1;
        else {
            yvex_error_setf(err, YVEX_ERR_BACKEND, "cuda.dense-plan.matmul",
                            "compiled cuBLASLt submission failed with status %d", blas_status);
            rc = YVEX_ERR_BACKEND;
        }
    }
    if (rc == YVEX_OK && plan->requirement.publication_dtype == YVEX_DTYPE_BF16) {
        void *parameters[] = {&output_ptr, &output_values, &device_status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->attention_bf16_round_function,
            (unsigned int)(output_tasks / CUDA_QTYPE_MATVEC_BLOCK),
            CUDA_QTYPE_MATVEC_BLOCK, 0u, parameters, "cuda.dense-plan.publish", err);
        if (rc == YVEX_OK) submitted = 1;
    }
    if (submitted) {
        int sync_rc = yvex_cuda_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            "cuda.dense-plan.complete", rc == YVEX_OK ? err : NULL);
        if (rc == YVEX_OK) rc = sync_rc;
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(
            &state->driver,
            state->driver.cuMemcpyDtoH_v2(&host_status, device_status, sizeof(host_status)),
            "cuda.dense-plan.status", err);
    if (rc == YVEX_OK && host_status) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "cuda.dense-plan.execute",
                       "dense executable produced or consumed invalid numerics");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_cuda_work_cleanup(&work, &cleanup);
    backend_workspace_reset(backend);
    plan->in_use = 0;
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    if (rc != YVEX_OK) return rc;
    if (plan->summary.use_count == ULLONG_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.dense-plan.execute",
                       "dense executable use accounting overflowed");
        return YVEX_ERR_BOUNDS;
    }
    request->output->is_written = 1;
    plan->summary.use_count++;
    facts->d2h_bytes = sizeof(host_status);
    facts->kernel_launches = plan->requirement.publication_dtype == YVEX_DTYPE_BF16 ? 3ull : 2ull;
    facts->download_count = 1ull;
    facts->device_synchronizations = 1ull;
    facts->active_weight_bytes = weight->encoded_bytes;
    facts->activation_bytes = activation_bytes;
    facts->temporary_bytes = total_bytes;
    facts->accelerated_matrix_launches = 1ull;
    facts->compulsory_memory_facts_available = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_transformer_linear_summary(
    const yvex_transformer_linear_executable *plan,
    yvex_transformer_linear_executable_summary *summary, yvex_error *err)
{
    if (!plan || !summary ||
        plan->summary.schema_version != YVEX_TRANSFORMER_LINEAR_EXECUTABLE_SCHEMA_V1) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda.dense-plan.summary",
                       "one admitted dense executable is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *summary = plan->summary;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_cuda_transformer_linear_release(
    yvex_backend *backend, yvex_transformer_linear_executable **owner, yvex_error *err)
{
    yvex_transformer_linear_executable *plan = owner ? *owner : NULL;
    int status = 0;
    if (!owner || (plan && (!backend || plan->backend != backend || plan->in_use))) {
        yvex_error_set(err, YVEX_ERR_STATE, "cuda.dense-plan.release",
                       "idle dense executable ownership is required");
        return YVEX_ERR_STATE;
    }
    if (!plan) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (plan->output_layout) status |= plan->lt.layout_destroy(plan->output_layout);
    if (plan->input_layout) status |= plan->lt.layout_destroy(plan->input_layout);
    if (plan->weight_layout) status |= plan->lt.layout_destroy(plan->weight_layout);
    if (plan->operation) status |= plan->lt.operation_destroy(plan->operation);
    status |= cuda_blas_lt_close(&plan->lt, YVEX_OK, NULL) != YVEX_OK;
    memset(plan, 0, sizeof(*plan));
    free(plan);
    *owner = NULL;
    if (status) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.dense-plan.release",
                       "dense executable cleanup reported a backend failure");
        return YVEX_ERR_BACKEND;
    }
    yvex_error_clear(err);
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
    unsigned long long activation_bytes, yvex_backend_operation_facts *facts,
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

int yvex_cuda_transformer_linear_bf16(
    yvex_backend *backend, const unsigned char *resident_weight,
    unsigned long long weight_bytes, const unsigned char *resident_bias,
    unsigned long long bias_bytes, unsigned long long output_width,
    unsigned long long input_width, unsigned long long input_rows,
    const yvex_device_tensor *input, yvex_device_tensor *output,
    yvex_backend_operation_facts *facts, yvex_error *err)
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
                               yvex_cuda_launch_stream(backend), NULL, err);
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

int yvex_cuda_transformer_linear_f32(
    yvex_backend *backend, const unsigned char *resident_weight,
    unsigned long long weight_bytes, const unsigned char *resident_bias,
    unsigned long long bias_bytes, unsigned long long output_width,
    unsigned long long input_width, unsigned long long input_rows,
    const yvex_device_tensor *input, yvex_device_tensor *output,
    const yvex_transformer_linear_physical_plan *physical_plan,
    yvex_backend_operation_facts *facts, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_cuda_work work = {0};
    cuda_blas_lt lt = {0};
    CUdeviceptr resident_weight_device = 0ull, resident_bias_device = 0ull;
    CUdeviceptr weight = 0ull, bias = 0ull, workspace = 0ull;
    const cuda_linear_implementation *implementation = NULL;
    unsigned long long input_values, output_values, expected_weight, expected_bias, temporary_bytes;
    unsigned long long launches = 0ull;
    int rc, cleanup_rc;
    yvex_error cleanup;
    if (facts) memset(facts, 0, sizeof(*facts));
    if (!state || !facts || !physical_plan || !resident_weight || !resident_bias ||
        !output_width || !input_width || !input_rows ||
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
    rc = cuda_linear_physical_validate(
        backend, physical_plan, input_width, output_width, &implementation, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_add(weight_bytes, bias_bytes, &temporary_bytes) ||
        !yvex_core_u64_add(temporary_bytes, implementation->workspace_bytes, &temporary_bytes)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "cuda.encoded-linear-f32.physical",
                       "compiled linear workspace accounting overflowed");
        return YVEX_ERR_BOUNDS;
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
        rc = yvex_cuda_work_allocate(&work, &workspace, (size_t)implementation->workspace_bytes,
                                     NULL, 1,
                                     "cuda.encoded-linear-f32.workspace", NULL, err);
    if (rc == YVEX_OK) rc = cuda_blas_lt_open(&lt, err);
    if (rc == YVEX_OK)
        rc = cuda_blas_lt_bias_f32_batches(
            &lt, weight, (CUdeviceptr)input->data, bias,
            (CUdeviceptr)output->data, output_width, input_width, input_rows,
            (size_t)implementation->workspace_bytes, workspace,
            (size_t)implementation->workspace_bytes, implementation,
            yvex_cuda_launch_stream(backend), &launches, err);
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
    facts->kernel_launches = launches;
    facts->upload_count = 2ull;
    facts->device_synchronizations = 1ull;
    facts->active_weight_bytes = weight_bytes + bias_bytes;
    facts->activation_bytes = (input_values + output_values) * sizeof(float);
    facts->temporary_bytes = temporary_bytes;
    facts->accelerated_matrix_launches = launches;
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
    unsigned long long activation_bytes, yvex_backend_operation_facts *facts,
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
static int cuda_encoded_matvec(
    yvex_backend *backend, const unsigned char *resident_encoded,
    unsigned long long encoded_bytes, unsigned int qtype,
    unsigned long long row_count, unsigned long long row_width,
    unsigned long long row_bytes, unsigned long long input_rows,
    const yvex_device_tensor *input, const yvex_device_tensor *input_tail,
    unsigned long long input_head_width, const yvex_device_tensor *additive,
    yvex_device_tensor *output, int activation_q8,
    yvex_backend_operation_facts *facts, yvex_error *err)
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
        facts->accelerated_matrix_launches = tensorcore_path ? 1ull : 0ull;
        facts->compulsory_memory_facts_available = 1;
        yvex_error_clear(err);
    }
    return rc;
}

/*
 * Decode selected rows from one admitted resident matrix into a backend-owned F32 view.
 * Row identifiers are bounded host control facts; encoded values never leave device residency.
 */
static int cuda_encoded_gather(
    yvex_backend *backend, const unsigned char *resident_encoded,
    unsigned long long encoded_bytes, unsigned int qtype,
    unsigned long long row_count, unsigned long long row_width,
    unsigned long long row_bytes, const unsigned int *row_ids,
    unsigned long long selected_rows, yvex_device_tensor *output,
    yvex_backend_operation_facts *facts, yvex_error *err)
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
