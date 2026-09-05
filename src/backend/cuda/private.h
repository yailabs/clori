/* CUDA owners share admitted Driver handles, kernels, tensors, and launch resources.
 * Family topology cannot be reconstructed from this platform boundary. */
#ifndef SRC_BACKEND_CUDA_PRIVATE_H_INCLUDED
#define SRC_BACKEND_CUDA_PRIVATE_H_INCLUDED
#include <stddef.h>
#include <yvex/backend.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/quant_numeric.h>
#include "src/backend/private.h"
#define YVEX_CUDA_Q8_K_BLOCK 256ull
#define YVEX_CUDA_Q8_K_BYTES 292ull
#ifdef __cplusplus
extern "C" {
#endif
typedef int CUresult, CUdevice;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream, *CUevent, *CUgraph;
typedef void *CUgraphExec, *CUgraphNode;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef struct { int type, id; } CUmemLocation;
typedef struct {
    int type, requested_handle_types;
    CUmemLocation location;
    void *win32_handle_metadata;
    struct {
        unsigned char compression_type, gpu_direct_rdma_capable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocation_flags;
} CUmemAllocationProp;
typedef struct { CUmemLocation location; int flags; } CUmemAccessDesc;
typedef struct {
    CUfunction function;
    unsigned int grid_x, grid_y, grid_z;
    unsigned int block_x, block_y, block_z;
    unsigned int shared_bytes;
    void **parameters;
    void **extra;
    void *kernel;
    CUcontext context;
} yvex_cuda_kernel_node_params;
typedef struct {
    unsigned char from_port;
    unsigned char to_port;
    unsigned char type;
    unsigned char reserved[5];
} CUgraphEdgeData;
typedef struct {
    int result;
    CUgraphNode error_node;
    CUgraphNode error_from_node;
} CUgraphExecUpdateResultInfo;
typedef enum {
    YVEX_CUDA_KERNEL_BUNDLE_ABSENT = 0,
    YVEX_CUDA_KERNEL_BUNDLE_ADMITTED,
    YVEX_CUDA_KERNEL_BUNDLE_REJECTED
} yvex_cuda_kernel_bundle_state;
#define YVEX_CUDA_WORK_MAX_RANGES 96u
#define YVEX_CUDA_DEFERRED_RELEASE_MAX (YVEX_CUDA_WORK_MAX_RANGES + 8u)
#if YVEX_CUDA_DEFERRED_RELEASE_MAX < YVEX_CUDA_WORK_MAX_RANGES
#error "deferred release capacity must cover one complete CUDA work transaction"
#endif
typedef struct {
    CUdeviceptr pointer;
    unsigned long long bytes;
    yvex_backend_operation_variant variant;
} yvex_cuda_deferred_release;
#define YVEX_CUDA_ATTENTION_CONFIGURATION_CAP 128u
typedef struct {
    int configured;
    yvex_backend_attention_phase phase;
    yvex_backend_cuda_attention_mode mode;
    unsigned long long local_capacity;
    unsigned long long compressed_capacity;
    unsigned long long indexer_capacity;
    char compatibility_identity[YVEX_BACKEND_CUDA_GRAPH_IDENTITY_CAP];
    char capture_bucket[YVEX_BACKEND_CUDA_CAPTURE_BUCKET_CAP];
} yvex_cuda_attention_configuration;
#define YVEX_CUDA_SUCCESS 0
#define YVEX_CUDA_ERROR_INVALID_VALUE 1
#define YVEX_CUDA_ERROR_OUT_OF_MEMORY 2
#define YVEX_CUDA_ERROR_NOT_INITIALIZED 3
#define YVEX_CUDA_ERROR_NO_DEVICE 100
#define YVEX_CUDA_ERROR_NOT_FOUND 500
#define YVEX_CUDA_ERROR_NOT_SUPPORTED 801
#define YVEX_CUDA_KERNEL_MODULE_MAX 8u
#define YVEX_CUDA_CTX_MAP_HOST 0x08u
#define YVEX_CUDA_STREAM_NON_BLOCKING 0x01u
#define YVEX_CUDA_MEM_ATTACH_GLOBAL 0x01u
#define YVEX_CUDA_MEMHOSTREGISTER_DEVICEMAP 0x02u
#define YVEX_CUDA_MEMHOSTREGISTER_READ_ONLY 0x08u
#define YVEX_CUDA_MEM_ACCESS_READ_WRITE 0x03
#define YVEX_CUDA_MEM_LOCATION_DEVICE 0x01
#define YVEX_CUDA_MEM_ALLOCATION_PINNED 0x01
#define YVEX_CUDA_MEM_GRANULARITY_MINIMUM 0x00
#define YVEX_CUDA_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY 19
#define YVEX_CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_READ_ONLY_SUPPORTED 113
#define YVEX_CUDA_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING 41
#define YVEX_CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY 83
#define YVEX_CUDA_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT 102
typedef struct {
    void *library;
    CUresult (*cuInit)(unsigned int flags);
    CUresult (*cuDriverGetVersion)(int *driverVersion);
    CUresult (*cuDeviceGetCount)(int *count);
    CUresult (*cuDeviceGet)(CUdevice *device, int ordinal);
    CUresult (*cuDeviceGetName)(char *name, int len, CUdevice dev);
    CUresult (*cuDeviceComputeCapability)(int *major, int *minor, CUdevice dev);
    CUresult (*cuDeviceTotalMem_v2)(size_t *bytes, CUdevice dev);
    CUresult (*cuDeviceGetAttribute)(int *pi, int attrib, CUdevice dev);
    CUresult (*cuCtxCreate_v2)(CUcontext *pctx, unsigned int flags, CUdevice dev);
    CUresult (*cuCtxDestroy_v2)(CUcontext ctx);
    CUresult (*cuCtxSetCurrent)(CUcontext ctx);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuMemGetInfo_v2)(size_t *free_bytes, size_t *total_bytes);
    CUresult (*cuMemAlloc_v2)(CUdeviceptr *dptr, size_t bytesize);
    CUresult (*cuMemAllocManaged)(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
    CUresult (*cuMemPrefetchAsync_v2)(CUdeviceptr, size_t, CUmemLocation, unsigned int, CUstream);
    CUresult (*cuMemAdvise_v2)(CUdeviceptr, size_t, int, CUmemLocation);
    CUresult (*cuMemAddressReserve)(CUdeviceptr *, size_t, size_t, CUdeviceptr,
                                   unsigned long long);
    CUresult (*cuMemAddressFree)(CUdeviceptr ptr, size_t size);
    CUresult (*cuMemCreate)(CUmemGenericAllocationHandle *, size_t,
                            const CUmemAllocationProp *, unsigned long long);
    CUresult (*cuMemRelease)(CUmemGenericAllocationHandle handle);
    CUresult (*cuMemMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                         unsigned long long);
    CUresult (*cuMemUnmap)(CUdeviceptr ptr, size_t size);
    CUresult (*cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
    CUresult (*cuMemGetAllocationGranularity)(size_t *, const CUmemAllocationProp *, int);
    CUresult (*cuMemHostRegister_v2)(void *ptr, size_t bytes, unsigned int flags);
    CUresult (*cuMemHostGetDevicePointer_v2)(CUdeviceptr *device, void *host,
                                             unsigned int flags);
    CUresult (*cuMemHostUnregister)(void *ptr);
    CUresult (*cuMemFree_v2)(CUdeviceptr dptr);
    CUresult (*cuMemsetD8_v2)(CUdeviceptr dstDevice, unsigned char uc, size_t n);
    CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
    CUresult (*cuMemcpyDtoH_v2)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
    CUresult (*cuMemcpyDtoD_v2)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
    CUresult (*cuModuleLoadData)(CUmodule *module, const void *image);
    CUresult (*cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
    CUresult (*cuModuleUnload)(CUmodule hmod);
    CUresult (*cuLaunchKernel)(CUfunction f,
                               unsigned int gridDimX,
                               unsigned int gridDimY,
                               unsigned int gridDimZ,
                               unsigned int blockDimX,
                               unsigned int blockDimY,
                               unsigned int blockDimZ,
                               unsigned int sharedMemBytes,
                               void *hStream,
                               void **kernelParams,
                               void **extra);
    CUresult (*cuStreamCreate)(CUstream *stream, unsigned int flags);
    CUresult (*cuStreamDestroy_v2)(CUstream stream);
    CUresult (*cuStreamSynchronize)(CUstream stream);
    CUresult (*cuStreamWaitEvent)(CUstream stream, CUevent event, unsigned int flags);
    CUresult (*cuStreamBeginCapture_v2)(CUstream stream, int mode);
    CUresult (*cuStreamEndCapture)(CUstream stream, CUgraph *graph);
    CUresult (*cuGraphGetNodes)(CUgraph graph, CUgraphNode *nodes, size_t *count);
    CUresult (*cuGraphGetEdges_v2)(CUgraph graph, CUgraphNode *from,
                                   CUgraphNode *to, CUgraphEdgeData *data,
                                   size_t *count);
    CUresult (*cuGraphNodeGetType)(CUgraphNode node, int *type);
    CUresult (*cuGraphInstantiateWithFlags)(CUgraphExec *exec,
                                           CUgraph graph,
                                           unsigned long long flags);
    CUresult (*cuGraphUpload)(CUgraphExec exec, CUstream stream);
    CUresult (*cuGraphLaunch)(CUgraphExec exec, CUstream stream);
    CUresult (*cuGraphExecUpdate_v2)(CUgraphExec exec,
                                    CUgraph graph,
                                    CUgraphExecUpdateResultInfo *result);
    CUresult (*cuGraphExecKernelNodeSetParams_v2)(
        CUgraphExec exec, CUgraphNode node,
        const yvex_cuda_kernel_node_params *params);
    CUresult (*cuGraphExecDestroy)(CUgraphExec exec);
    CUresult (*cuGraphDestroy)(CUgraph graph);
    CUresult (*cuMemAllocAsync)(CUdeviceptr *ptr, size_t bytes, CUstream stream);
    CUresult (*cuMemFreeAsync)(CUdeviceptr ptr, CUstream stream);
    CUresult (*cuMemcpyHtoDAsync_v2)(CUdeviceptr dst,
                                    const void *src,
                                    size_t bytes,
                                    CUstream stream);
    CUresult (*cuMemcpyDtoDAsync_v2)(CUdeviceptr dst,
                                    CUdeviceptr src,
                                    size_t bytes,
                                    CUstream stream);
    CUresult (*cuMemcpyDtoHAsync_v2)(void *dst,
                                    CUdeviceptr src,
                                    size_t bytes,
                                    CUstream stream);
    CUresult (*cuMemsetD8Async)(CUdeviceptr dst,
                               unsigned char value,
                               size_t bytes,
                               CUstream stream);
    CUresult (*cuMemHostAlloc)(void **ptr, size_t bytes, unsigned int flags);
    CUresult (*cuMemFreeHost)(void *ptr);
    CUresult (*cuEventCreate)(CUevent *event, unsigned int flags);
    CUresult (*cuEventRecord)(CUevent event, CUstream stream);
    CUresult (*cuEventSynchronize)(CUevent event);
    CUresult (*cuEventElapsedTime_v2)(float *milliseconds, CUevent start, CUevent end);
    CUresult (*cuEventDestroy_v2)(CUevent event);
    CUresult (*cuGetErrorName)(CUresult error, const char **pStr);
    CUresult (*cuGetErrorString)(CUresult error, const char **pStr);
} yvex_cuda_driver;
typedef struct {
    void *library, *handle;
    int (*create)(void **handle), (*destroy)(void *handle);
    int (*set_stream)(void *handle, CUstream stream);
    int (*set_workspace)(void *handle, void *workspace, size_t bytes);
    int (*gemm_ex)(void *handle, int transa, int transb, int m, int n, int k,
                   const void *alpha, const void *a, int atype, int lda,
                   const void *b, int btype, int ldb, const void *beta,
                   void *c, int ctype, int ldc, int compute_type, int algorithm);
    int (*gemm_strided_batched_ex)(
        void *handle, int transa, int transb, int m, int n, int k,
        const void *alpha, const void *a, int atype, int lda, long long stride_a,
        const void *b, int btype, int ldb, long long stride_b, const void *beta,
        void *c, int ctype, int ldc, long long stride_c, int batches,
        int compute_type, int algorithm);
    int ready;
} yvex_cuda_blas;
typedef struct {
    yvex_cuda_driver driver;
    yvex_cuda_blas blas;
    CUcontext context;
    CUdevice device;
    int device_index;
    int driver_version;
    CUmodule modules[YVEX_CUDA_KERNEL_MODULE_MAX];
    unsigned int module_count;
    CUfunction embed_function;
    CUfunction embed_f16_function;
    CUfunction rms_norm_f32_function;
    CUfunction rms_norm_f16_function;
    CUfunction rms_norm_bf16_policy_function;
    CUfunction rope_function;
    CUfunction matmul_function;
    CUfunction qtype_row_dot_function;
    CUfunction attention_bf16_round_function;
    CUfunction bf16_pack_function, bf16_unpack_function;
    CUfunction qtype_matvec_function, qtype_grouped_rows_function, mxfp4_q8_rows_function,
        attention_bf16_pair_function;
    CUfunction qtype_split_matvec_function, qtype_tensorcore_rows_function;
    CUfunction qtype_gather_function, argmax_f32_function;
    CUfunction sample_stochastic_f32_function;
    CUfunction speculation_stochastic_f32_function;
    CUfunction q8_quantize_function;
    CUfunction encoded_row_decode_function;
    CUfunction attention_weighted_norm_function;
    CUfunction attention_unit_norm_function;
    CUfunction attention_yarn_rope_function;
    CUfunction attention_activation_quantize_function;
    CUfunction residual_mhc_pre_function;
    CUfunction residual_mhc_post_function;
    CUfunction transformer_feature_mean_function;
    CUfunction transformer_final_function;
    CUfunction attention_rolling_state_function;
    CUfunction attention_topk_function;
    CUfunction attention_reduce_function, attention_reduce_native_function;
    CUfunction moe_route_function;
    CUfunction moe_route_rows_function;
    CUfunction expert_worklist_build_cuda_function;
    CUfunction moe_grouped_up_function;
    CUfunction moe_grouped_down_function;
    CUfunction moe_grouped_up_rows_function, moe_grouped_down_rows_function;
    CUfunction moe_grouped_up_tensorcore_function, moe_grouped_down_tensorcore_function;
    CUfunction moe_reduce_rows_function;
    CUfunction moe_combine_rows_function;
    CUfunction moe_swiglu_function;
    CUfunction moe_accumulate_function;
    CUfunction mlp_function;
    CUfunction attention_function;
    CUfunction rotary_half_function, rotary_half_plain_function, gqa_function,
        gqa_wide_function;
    CUfunction gqa_softmax_function, gqa_softmax_warp_function, attention_validate_function;
    CUfunction silu_product_function, sigmoid_product_function, silu_function, gelu_function;
    CUfunction timestep_embedding_function, split_three_function, split_interleaved_function;
    CUfunction split_interleaved_two_function, swiglu_split_function, swiglu_split_f32_function;
    CUfunction modulation_function, gated_residual_function, bias_function, add_bf16_function;
    CUfunction scaled_residual_f32_function, layer_norm_f32_function;
    CUfunction conv_scale_function, conv1d_function, conv1d_transposed_function;
    CUfunction conv2d_function, group_norm_silu_function;
    CUfunction alias_up_function, alias_down_function;
    CUfunction vector_update_function, clamp_function;
    CUfunction gated_delta_convolution_function, gated_delta_recurrence_function;
    yvex_cuda_kernel_bundle_state kernel_bundle_state;
    yvex_backend_capability_reason kernel_bundle_reason;
    yvex_backend_operation_variant kernel_bundle_failure_variant;
    yvex_backend_capability_reason backend_failure_reason;
    yvex_backend_capability_reason variant_failures[YVEX_BACKEND_VARIANT_COUNT];
    yvex_backend_cuda_graph *graphs;
    yvex_backend_cuda_graph *capture_owner;
    yvex_backend_cuda_graph *parameter_update_owner;
    CUstream capture_stream, execution_stream;
    int shared_stream_in_flight;
    CUevent timing_start;
    CUevent timing_stop;
    int timing_ready;
    int timing_active;
    yvex_cuda_attention_configuration
        attention_configurations[YVEX_CUDA_ATTENTION_CONFIGURATION_CAP];
    unsigned int attention_configuration_count;
    unsigned int attention_active_configuration;
    unsigned long long attention_configuration_hits;
    unsigned long long attention_configuration_misses;
    yvex_cuda_deferred_release deferred_releases[YVEX_CUDA_DEFERRED_RELEASE_MAX];
    unsigned int deferred_release_count;
    unsigned long long deferred_release_bytes;
    void *registered_host;
    CUdeviceptr registered_device, transformer_status;
    unsigned long long registered_bytes;
    int kernel_bundle_native, status_transaction_active;
    char kernel_bundle_identity[YVEX_SHA256_HEX_BYTES];
    char kernel_bundle_architecture[16];
    yvex_backend_bandwidth_evidence bandwidth_evidence;
    int bandwidth_ready, bandwidth_active, virtual_memory_management, can_map_host, host_register_readonly;
    const yvex_backend *context_owner;
    int context_borrowed;
} yvex_cuda_backend_state;
const yvex_cuda_attention_configuration *yvex_cuda_attention_configuration_active(
    const yvex_cuda_backend_state *state, yvex_backend_attention_phase phase);
static inline unsigned long long yvex_cuda_attention_local_capacity(
    const yvex_cuda_attention_configuration *shape,
    const yvex_backend_attention_job *job, int publication) {
    return shape->local_capacity + (!publication && shape->local_capacity < job->sliding_window);
}
/* These formats can consume the canonical Q8_K activation workspace. Runtime
 * admission remains a separate explicit decision because weight qtype alone
 * cannot establish whole-stack numerical compatibility. */
static inline int yvex_cuda_q8_activation_eligible(unsigned int qtype) {
    return qtype == YVEX_GGUF_QTYPE_IQ2_XXS || qtype == YVEX_GGUF_QTYPE_Q2_K ||
           qtype == YVEX_GGUF_QTYPE_Q8_0 || qtype == YVEX_GGUF_QTYPE_MXFP4;
}
typedef int (*yvex_cuda_graph_enqueue_fn)(void *context, int enqueue_kernels, yvex_error *err);
typedef int (*yvex_cuda_graph_prepare_fn)(void *context, yvex_error *err);
/* Canonical contiguous stages in the admitted CUDA attention launch schedule. */
typedef enum {
    YVEX_CUDA_ATTENTION_STAGE_ENVELOPE_PRE = 0,
    YVEX_CUDA_ATTENTION_STAGE_PROJECT,
    YVEX_CUDA_ATTENTION_STAGE_COMPRESS,
    YVEX_CUDA_ATTENTION_STAGE_REDUCE,
    YVEX_CUDA_ATTENTION_STAGE_ENVELOPE_POST,
    YVEX_CUDA_ATTENTION_STAGE_COUNT
} yvex_cuda_attention_stage;
/* Admit only semantically active pieces to the piecewise launch graph. */
static inline int cuda_attention_piece_active(
    yvex_backend_attention_scope scope,
    yvex_backend_attention_class attention_class,
    yvex_cuda_attention_stage stage) {
    if (stage >= YVEX_CUDA_ATTENTION_STAGE_COUNT)
        return 0;
    if (scope == YVEX_BACKEND_ATTENTION_SCOPE_CORE &&
        (stage == YVEX_CUDA_ATTENTION_STAGE_ENVELOPE_PRE ||
         stage == YVEX_CUDA_ATTENTION_STAGE_ENVELOPE_POST))
        return 0;
    return stage != YVEX_CUDA_ATTENTION_STAGE_COMPRESS ||
           attention_class != YVEX_BACKEND_ATTENTION_SWA;
}
typedef struct {
    yvex_backend *backend;
    yvex_cuda_backend_state *state;
    yvex_backend_operation_variant variant;
    CUdeviceptr pointers[YVEX_CUDA_WORK_MAX_RANGES], q8_input, status;
    unsigned long long sizes[YVEX_CUDA_WORK_MAX_RANGES];
    unsigned char workspace_owned[YVEX_CUDA_WORK_MAX_RANGES], status_deferred;
    int prepare_only, raw_only, forensic_numeric, activation_q8;
    unsigned int count;
    unsigned long long current_bytes, peak_bytes, budget, launches, q8_capacity;
    unsigned long long tensor_core_launches;
} yvex_cuda_work;
typedef enum {
    YVEX_CUDA_WORK_FAILURE_NONE = 0,
    YVEX_CUDA_WORK_FAILURE_BUDGET,
    YVEX_CUDA_WORK_FAILURE_ALLOCATION,
    YVEX_CUDA_WORK_FAILURE_COPY
} yvex_cuda_work_failure;
typedef struct {
    const void *data;
    unsigned long long count;
    size_t width;
} yvex_cuda_host_span;
typedef struct {
    CUdeviceptr *device;
    void *output, *staged;
    unsigned long long capacity, output_capacity, *used;
    size_t width;
    const char *stage;
} yvex_cuda_attention_transfer;
typedef struct {
    CUdeviceptr *device;
    const void *source;
    void *staged;
    unsigned long long count, used;
    size_t width;
    int generated;
    const char *stage;
} yvex_cuda_attention_upload;
int yvex_cuda_work_allocate(yvex_cuda_work *, CUdeviceptr *, size_t, const void *, int,
                            const char *, yvex_cuda_work_failure *, yvex_error *);
int yvex_cuda_work_initialize(yvex_cuda_work *work, CUdeviceptr target,
                              size_t bytes, const void *source, int zero,
                              const char *stage, yvex_error *err);
int yvex_cuda_driver_load(yvex_cuda_driver *driver, yvex_error *err);
void yvex_cuda_driver_unload(yvex_cuda_driver *driver);
int yvex_cuda_status(const yvex_cuda_driver *, CUresult, const char *, yvex_error *);
yvex_cuda_backend_state *yvex_cuda_state(const yvex_backend *backend);
int yvex_cuda_set_current(const yvex_backend *backend, const char *where, yvex_error *err);
int yvex_cuda_refresh_memory_info(yvex_backend *backend, yvex_error *err);
CUdeviceptr yvex_cuda_tensor_ptr(const yvex_device_tensor *tensor);
int yvex_cuda_blas_bind_launch_stream(yvex_backend *backend, const char *where, yvex_error *err);
CUstream yvex_cuda_launch_stream(const yvex_backend *backend);
int yvex_cuda_resident_alloc(yvex_backend *, const yvex_backend_tensor_desc *,
                             yvex_device_tensor **, unsigned char **, yvex_error *);
int yvex_cuda_resident_map_supported(const yvex_backend *);
int yvex_cuda_resident_map_readonly(yvex_backend *, const yvex_backend_tensor_desc *,
                                    const unsigned char *, yvex_device_tensor **, yvex_error *);
int yvex_cuda_resident_prefetch_supported(const yvex_backend *);
int yvex_cuda_resident_prefetch(yvex_backend *, yvex_device_tensor *, unsigned long long *, yvex_error *);
typedef enum {
    YVEX_CUDA_TIMING_BEGIN = 0,
    YVEX_CUDA_TIMING_FINISH,
    YVEX_CUDA_TIMING_DISCARD
} yvex_cuda_timing_action;
int yvex_cuda_timing(yvex_backend *backend, CUstream stream,
                     yvex_cuda_timing_action action, unsigned long long *elapsed_ns,
                     const char *where, yvex_error *err);
int yvex_cuda_capture_active(const yvex_backend *backend);
int yvex_cuda_graphs_close_all(yvex_backend *backend, yvex_error *err);
typedef enum {
    YVEX_CUDA_GRAPH_EXECUTION_MEASURE_DEVICE_TIME = 1u << 0,
    YVEX_CUDA_GRAPH_EXECUTION_SHARED_LAUNCH_STREAM = 1u << 1,
    YVEX_CUDA_GRAPH_EXECUTION_DEFER_COMPLETION = 1u << 2
} yvex_cuda_graph_execution_flag;
int yvex_cuda_graph_execute(yvex_backend *, const char *, yvex_cuda_graph_prepare_fn,
    yvex_cuda_graph_enqueue_fn, void *, unsigned int, yvex_backend_cuda_graph_info *, yvex_error *);
int yvex_cuda_graph_kernel_capture(yvex_backend *, yvex_backend_operation_variant, CUfunction,
    unsigned int, unsigned int, unsigned int, const char *, yvex_error *);
int yvex_cuda_graph_kernel_update(yvex_backend *, yvex_backend_operation_variant, CUfunction,
    unsigned int, unsigned int, unsigned int, void **, const char *, yvex_error *);
int yvex_cuda_attention_graph_key(const yvex_backend *, const yvex_backend_attention_job *,
                                  unsigned int, unsigned int, char[160], yvex_error *);
static inline int yvex_cuda_work_checked_bytes(unsigned long long count,
                                               unsigned long long width,
                                               size_t *out) {
    if (!out || !width || count > (unsigned long long)SIZE_MAX / width)
        return 0;
    *out = (size_t)(count * width);
    return 1;
}
int yvex_cuda_work_cleanup(yvex_cuda_work *work, yvex_error *err);
int yvex_cuda_activation_views_valid(yvex_backend *backend,
    const yvex_device_tensor *input, unsigned long long input_elements,
    const yvex_device_tensor *output, unsigned long long output_elements);
CUdeviceptr yvex_cuda_activation_pointer(yvex_backend *backend, const yvex_device_tensor *tensor);
int yvex_cuda_activation_copy(yvex_backend *backend, CUdeviceptr source,
    yvex_device_tensor *output, unsigned long long elements,
    const char *stage, yvex_error *err);
typedef struct {
    CUdeviceptr local, local_positions, compressed, compressed_positions;
    CUdeviceptr indexer, indexer_positions, main_kv, main_score, index_kv, index_score;
    unsigned long long initial_local, initial_compressed, initial_indexer;
    unsigned long long emitted_compressed, emitted_indexer, local_capacity;
    unsigned long long main_extent, index_extent;
} yvex_cuda_attention_state_sources;
int yvex_cuda_qtype_matvec_geometry(
    unsigned long long rows, unsigned long long row_width, unsigned long long input_rows,
    unsigned int qtype, int block_row_eligible, unsigned int *grid, unsigned int *block,
    int *block_row);
int yvex_cuda_qtype_tensorcore_geometry(unsigned long long, unsigned long long, unsigned int *, unsigned int *);
#define YVEX_CUDA_TENSORCORE_MIN_ROWS 16ull
static inline int cuda_qtype_tensorcore_eligible(unsigned long long input_rows) {
    return input_rows >= YVEX_CUDA_TENSORCORE_MIN_ROWS;
}
typedef struct {
    int (*fail)(yvex_backend_attention_failure *, yvex_backend_attention_failure_code,
                const char *, unsigned long long, unsigned long long, yvex_error *,
                yvex_status, const char *);
    int (*account_transfer)(unsigned long long, size_t, unsigned long long *,
                            const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*validate_job)(yvex_backend_attention_job *, yvex_backend_attention_output *,
                        yvex_backend_attention_failure *, yvex_error *);
    int (*validate_weight)(const yvex_backend_attention_weight *, unsigned long long, unsigned long long,
                           yvex_backend_attention_failure *, yvex_error *);
    int (*validate_activation)(const yvex_backend_attention_activation *, unsigned long long,
                               const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*validate_rolling)(const yvex_backend_attention_job *,
                            const yvex_backend_attention_rolling *, unsigned long long,
                            unsigned long long, int, unsigned long long *, const char *,
                            yvex_backend_attention_failure *, yvex_error *);
    int (*validate_alias)(const yvex_backend_attention_job *,
                          const yvex_cuda_attention_transfer *, size_t, unsigned long long,
                          unsigned long long, unsigned long long, unsigned long long,
                          unsigned long long);
    int (*cancel)(yvex_backend *, const yvex_backend_attention_job *,
                  const char *, int, yvex_backend_attention_failure *, yvex_error *);
    int (*stage_acquire)(yvex_backend *, size_t, int, int, unsigned char **, int *,
                         yvex_backend_attention_failure *, yvex_error *);
    int (*stage_layout)(unsigned char *, yvex_cuda_attention_upload *, size_t,
                        yvex_cuda_attention_transfer *, size_t,
                        unsigned long long, int **, unsigned long long **,
                        unsigned long long **, size_t *, size_t *);
    int (*allocate)(yvex_cuda_work *, CUdeviceptr *, size_t, const void *, int,
                    const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*initialize)(yvex_cuda_work *, CUdeviceptr, size_t, const void *, int,
                      const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*download)(yvex_cuda_work *, void *, CUdeviceptr, size_t, const char *,
                    yvex_backend_attention_failure *, yvex_error *);
    int (*launch)(yvex_cuda_work *, CUfunction, unsigned int, unsigned int, unsigned int,
                  void **, const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*round_bf16)(yvex_cuda_work *, CUdeviceptr, unsigned long long, CUdeviceptr,
                      const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*matvec)(yvex_cuda_work *, const yvex_backend_attention_weight *, CUdeviceptr,
                  unsigned long long, unsigned long long, unsigned long long, CUdeviceptr,
                  CUdeviceptr, int, CUdeviceptr, const char *,
                  yvex_backend_attention_failure *, yvex_error *);
    int (*matvec_grouped)(yvex_cuda_work *, const yvex_backend_attention_weight *, CUdeviceptr,
                  unsigned long long, unsigned long long, unsigned long long, CUdeviceptr,
                  unsigned long long, CUdeviceptr, unsigned long long, int, CUdeviceptr,
                  const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*decode)(yvex_cuda_work *, const yvex_backend_attention_weight *, CUdeviceptr,
                  unsigned long long, unsigned long long, CUdeviceptr, CUdeviceptr,
                  const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*weighted_norm)(yvex_cuda_work *, CUdeviceptr, unsigned long long, unsigned long long,
                         const yvex_backend_attention_weight *, CUdeviceptr, double,
                         CUdeviceptr, const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*unit_norm)(yvex_cuda_work *, CUdeviceptr, unsigned long long, unsigned long long,
                     double, CUdeviceptr, const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*rope)(yvex_cuda_work *, CUdeviceptr, unsigned long long, unsigned long long,
                unsigned long long, unsigned long long,
                const yvex_backend_attention_position *, int, CUdeviceptr, const char *,
                yvex_backend_attention_failure *, yvex_error *);
    int (*activation)(yvex_cuda_work *, CUdeviceptr, unsigned long long, unsigned long long,
                      unsigned long long,
                      const yvex_backend_attention_activation *, CUdeviceptr,
                      const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*state_stage)(yvex_backend *, const yvex_backend_attention_job *,
                       const yvex_cuda_attention_state_sources *, size_t *, int *, yvex_error *);
} yvex_cuda_attention_operations;
const yvex_cuda_attention_operations *yvex_cuda_attention_operations_get(void);
const struct yvex_backend_sampling_operations *yvex_cuda_sampling_operations_get(
    const yvex_backend *);
const struct yvex_backend_moe_operations *yvex_cuda_moe_operations_get(
    const yvex_backend *);
const struct yvex_backend_transformer_operations *yvex_cuda_transformer_operations_get(
    const yvex_backend *);
const struct yvex_backend_encoded_operations *yvex_cuda_encoded_operations_get(
    const yvex_backend *);
int yvex_cuda_kernel_bundle_admit(yvex_backend *backend, yvex_error *err);
const char *yvex_cuda_kernel_function_identity(const yvex_cuda_backend_state *, CUfunction);
int yvex_cuda_kernel_bundle_close(yvex_backend *backend, yvex_error *err);
int yvex_cuda_query_capability(const yvex_backend *, yvex_backend_operation_variant,
                               yvex_backend_capability_result *, yvex_error *);
int yvex_cuda_require_capability(yvex_backend *, yvex_backend_operation_variant,
                                 const char *, yvex_error *);
int yvex_cuda_launch(yvex_backend *, yvex_backend_operation_variant, CUfunction, unsigned int,
                     unsigned int, unsigned int, void **, const char *, yvex_error *);
int yvex_cuda_synchronize(yvex_backend *, yvex_backend_operation_variant,
                          const char *, yvex_error *);
int yvex_cuda_launch_synchronize(yvex_backend *, yvex_backend_operation_variant, int *,
                                 const char *, yvex_error *);
int yvex_cuda_temporary_free(yvex_backend *, yvex_backend_operation_variant, CUdeviceptr *,
                             unsigned long long, int, const char *, yvex_error *);
int yvex_cuda_deferred_release_drain(yvex_backend *backend, yvex_error *err);
int yvex_cuda_op_embed(yvex_backend *backend, const yvex_device_tensor *embedding,
                       const unsigned int *token_ids, unsigned long long token_count,
                       yvex_device_tensor *out, yvex_error *err);
int yvex_cuda_op_rms_norm(yvex_backend *backend, const yvex_device_tensor *input,
                          const yvex_device_tensor *weight, float epsilon,
                          yvex_device_tensor *out, yvex_error *err);
int yvex_cuda_op_rope(yvex_backend *backend, const yvex_device_tensor *input,
                      unsigned long long position, float rope_base,
                      yvex_device_tensor *out, yvex_error *err);
int yvex_cuda_op_matmul(yvex_backend *backend,
                        const yvex_device_tensor *input,
                        const yvex_device_tensor *weight,
                        yvex_device_tensor *out, yvex_error *err);
int yvex_cuda_op_mlp(yvex_backend *backend, const yvex_device_tensor *input,
                     const yvex_device_tensor *gate_weight, const yvex_device_tensor *up_weight,
                     const yvex_device_tensor *down_weight, const yvex_mlp_options *options,
                     yvex_device_tensor *intermediate, yvex_device_tensor *out, yvex_error *err);
int yvex_cuda_op_attention(yvex_backend *backend, const yvex_device_tensor *query,
                           const yvex_device_tensor *keys, const yvex_device_tensor *values,
                           unsigned long long seq_len, unsigned long long position, float scale,
                           int causal, yvex_device_tensor *score_scratch,
                           yvex_device_tensor *probability_scratch, yvex_device_tensor *out,
                           yvex_error *err);
int yvex_cuda_quant_row_dot(yvex_backend *backend,
                            unsigned int qtype,
                            const unsigned char *encoded,
                            size_t encoded_bytes,
                            const float *vector,
                            unsigned long long elements,
                            float *out,
                            yvex_quant_failure *failure, yvex_error *err);
#ifdef __cplusplus
}
#endif
#endif /* SRC_BACKEND_CUDA_PRIVATE_H_INCLUDED */
