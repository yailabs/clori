/* Owner: backend.cuda.private (backend.cuda).
 * Owns: CUDA driver ABI, generated kernel bundle, and qtype launch contracts.
 * Does not own: model policy, graph topology, or CPU fallback.
 * Invariants: declarations have one owner, stable ordering, and no hidden capability promotion.
 * Boundary: CUDA platform-private execution.
 * Purpose: provide the canonical CUDA platform-private execution contract.
 * Inputs: typed immutable facts and explicitly owned mutable lifecycle objects.
 * Effects: only declared lifecycle, allocation, I/O, and publication operations mutate state.
 * Failure: typed refusals leave outputs defined and preserve caller-owned state. */
#ifndef SRC_BACKEND_CUDA_PRIVATE_H_INCLUDED
#define SRC_BACKEND_CUDA_PRIVATE_H_INCLUDED

#include <stddef.h>
#include <yvex/backend.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/quant_numeric.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Driver contract. */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef void *CUgraph;
typedef void *CUgraphExec;
typedef void *CUgraphNode;
typedef unsigned long long CUdeviceptr;
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
#define YVEX_CUDA_SUCCESS 0
#define YVEX_CUDA_ERROR_INVALID_VALUE 1
#define YVEX_CUDA_ERROR_OUT_OF_MEMORY 2
#define YVEX_CUDA_ERROR_NOT_INITIALIZED 3
#define YVEX_CUDA_ERROR_NO_DEVICE 100
#define YVEX_CUDA_ERROR_NOT_SUPPORTED 801
#define YVEX_CUDA_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING 41
#define YVEX_CUDA_DEVICE_ATTRIBUTE_MANAGED_MEMORY 83
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
    CUresult (*cuStreamBeginCapture_v2)(CUstream stream, int mode);
    CUresult (*cuStreamEndCapture)(CUstream stream, CUgraph *graph);
    CUresult (*cuGraphGetNodes)(CUgraph graph, CUgraphNode *nodes, size_t *count);
    CUresult (*cuGraphGetEdges_v2)(CUgraph graph,
                                  CUgraphNode *from,
                                  CUgraphNode *to,
                                  CUgraphEdgeData *data,
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
    yvex_cuda_driver driver;
    CUcontext context;
    CUdevice device;
    int device_index;
    int driver_version;
    CUmodule module;
    CUfunction embed_function;
    CUfunction embed_f16_function;
    CUfunction rms_norm_f32_function;
    CUfunction rms_norm_f16_function;
    CUfunction rope_function;
    CUfunction matmul_function;
    CUfunction qtype_row_dot_function;
    CUfunction attention_bf16_round_function;
    CUfunction deepseek_qtype_matvec_function;
    CUfunction deepseek_decode_function;
    CUfunction deepseek_weighted_norm_function;
    CUfunction deepseek_unit_norm_function;
    CUfunction deepseek_rope_function;
    CUfunction deepseek_activation_function;
    CUfunction deepseek_mhc_pre_function;
    CUfunction deepseek_mhc_post_function;
    CUfunction deepseek_rolling_function;
    CUfunction deepseek_topk_function;
    CUfunction deepseek_reduce_function;
    CUfunction mlp_function;
    CUfunction attention_function;
    int module_loaded;
    yvex_cuda_kernel_bundle_state kernel_bundle_state;
    yvex_backend_capability_reason kernel_bundle_reason;
    yvex_backend_operation_variant kernel_bundle_failure_variant;
    yvex_backend_capability_reason backend_failure_reason;
    yvex_backend_capability_reason variant_failures[YVEX_BACKEND_VARIANT_COUNT];
    yvex_backend_cuda_graph *graphs;
    yvex_backend_cuda_graph *capture_owner;
    yvex_backend_cuda_graph *parameter_update_owner;
    CUstream capture_stream;
    CUevent timing_start;
    CUevent timing_stop;
    int timing_ready;
    int timing_active;
    int attention_graph_configured;
    yvex_backend_cuda_attention_mode attention_mode;
    unsigned long long attention_local_capacity;
    unsigned long long attention_compressed_capacity;
    unsigned long long attention_indexer_capacity;
    yvex_cuda_deferred_release deferred_releases[YVEX_CUDA_DEFERRED_RELEASE_MAX];
    unsigned int deferred_release_count;
    unsigned long long deferred_release_bytes;
    char kernel_bundle_identity[YVEX_SHA256_HEX_BYTES];
    char attention_compatibility_identity[YVEX_BACKEND_CUDA_GRAPH_IDENTITY_CAP];
    char attention_capture_bucket[YVEX_BACKEND_CUDA_CAPTURE_BUCKET_CAP];
    const yvex_backend *context_owner;
    int context_borrowed;
} yvex_cuda_backend_state;

typedef int (*yvex_cuda_graph_enqueue_fn)(void *context, int enqueue_kernels,
                                          yvex_error *err);

/* Canonical contiguous stages in the admitted CUDA attention launch schedule. */
typedef enum {
    YVEX_CUDA_ATTENTION_STAGE_ENVELOPE_PRE = 0,
    YVEX_CUDA_ATTENTION_STAGE_PROJECT,
    YVEX_CUDA_ATTENTION_STAGE_COMPRESS,
    YVEX_CUDA_ATTENTION_STAGE_REDUCE,
    YVEX_CUDA_ATTENTION_STAGE_ENVELOPE_POST,
    YVEX_CUDA_ATTENTION_STAGE_COUNT
} yvex_cuda_attention_stage;

/* Purpose: admit only semantically active pieces to the piecewise launch graph. */
static inline int cuda_attention_piece_active(
    yvex_backend_attention_scope scope,
    yvex_backend_attention_class attention_class,
    yvex_cuda_attention_stage stage)
{
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
    CUdeviceptr pointers[YVEX_CUDA_WORK_MAX_RANGES];
    unsigned long long sizes[YVEX_CUDA_WORK_MAX_RANGES];
    unsigned char workspace_owned[YVEX_CUDA_WORK_MAX_RANGES];
    int prepare_only, raw_only;
    unsigned int count;
    unsigned long long current_bytes, peak_bytes, budget, launches;
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
    const char *stage;
} yvex_cuda_attention_upload;
int yvex_cuda_work_allocate(yvex_cuda_work *work,
                            CUdeviceptr *out,
                            size_t bytes,
                            const void *source,
                            int zero,
                            const char *stage,
                            yvex_cuda_work_failure *failure,
                            yvex_error *err);
int yvex_cuda_driver_load(yvex_cuda_driver *driver, yvex_error *err);
void yvex_cuda_driver_unload(yvex_cuda_driver *driver);
int yvex_cuda_status(const yvex_cuda_driver *driver,
                     CUresult code,
                     const char *where,
                     yvex_error *err);
yvex_cuda_backend_state *yvex_cuda_state(const yvex_backend *backend);
int yvex_cuda_set_current(const yvex_backend *backend, const char *where, yvex_error *err);
int yvex_cuda_refresh_memory_info(yvex_backend *backend, yvex_error *err);
CUdeviceptr yvex_cuda_tensor_ptr(const yvex_device_tensor *tensor);
CUstream yvex_cuda_launch_stream(const yvex_backend *backend);
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
int yvex_cuda_graph_execute(yvex_backend *backend,
                            const char *compatibility_identity,
                            yvex_cuda_graph_enqueue_fn enqueue,
                            void *context,
                            yvex_backend_cuda_graph_info *info,
                            yvex_error *err);
int yvex_cuda_graph_kernel_capture(yvex_backend *backend,
                                   yvex_backend_operation_variant variant,
                                   CUfunction function,
                                   unsigned int grid,
                                   unsigned int block,
                                   unsigned int shared_bytes,
                                   const char *stage,
                                   yvex_error *err);
int yvex_cuda_graph_kernel_update(yvex_backend *backend,
                                  yvex_backend_operation_variant variant,
                                  CUfunction function,
                                  unsigned int grid,
                                  unsigned int block,
                                  unsigned int shared_bytes,
                                  void **params,
                                  const char *stage,
                                  yvex_error *err);
int yvex_cuda_attention_graph_key(const yvex_backend *backend,
                                  const yvex_backend_attention_job *job,
                                  unsigned int first,
                                  unsigned int last,
                                  char output[160],
                                  yvex_error *err);
/* Purpose: derive one representable host byte extent for a CUDA work range. */
static inline int yvex_cuda_work_checked_bytes(unsigned long long count,
                                               unsigned long long width,
                                               size_t *out)
{
    if (!out || !width || count > (unsigned long long)SIZE_MAX / width)
        return 0;
    *out = (size_t)(count * width);
    return 1;
}
int yvex_cuda_work_cleanup(yvex_cuda_work *work, yvex_error *err);
typedef struct {
    int (*fail)(yvex_backend_attention_failure *, yvex_backend_attention_failure_code,
                const char *, unsigned long long, unsigned long long, yvex_error *,
                yvex_status, const char *);
    int (*account_transfer)(unsigned long long, size_t, unsigned long long *,
                            const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*validate_job)(yvex_backend_attention_job *,
                        yvex_backend_attention_output *,
                        yvex_backend_attention_failure *, yvex_error *);
    int (*validate_weight)(const yvex_backend_attention_weight *,
                           unsigned long long, unsigned long long,
                           yvex_backend_attention_failure *, yvex_error *);
    int (*validate_activation)(const yvex_backend_attention_activation *,
                               unsigned long long, const char *,
                               yvex_backend_attention_failure *, yvex_error *);
    int (*validate_rolling)(const yvex_backend_attention_job *,
                            const yvex_backend_attention_rolling *,
                            unsigned long long, unsigned long long, int,
                            unsigned long long *, const char *,
                            yvex_backend_attention_failure *, yvex_error *);
    int (*validate_alias)(const yvex_backend_attention_job *,
                          const yvex_cuda_attention_transfer *, size_t,
                          unsigned long long, unsigned long long,
                          unsigned long long, unsigned long long,
                          unsigned long long);
    int (*cancel)(yvex_backend *, const yvex_backend_attention_job *,
                  const char *, int, yvex_backend_attention_failure *, yvex_error *);
    int (*stage_acquire)(yvex_backend *, size_t, int, int,
                         unsigned char **, int *,
                         yvex_backend_attention_failure *, yvex_error *);
    int (*stage_layout)(unsigned char *, yvex_cuda_attention_upload *, size_t,
                        yvex_cuda_attention_transfer *, size_t,
                        unsigned long long, int **, unsigned long long **,
                        unsigned long long **, size_t *);
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
                  unsigned long long, unsigned long long, CUdeviceptr, CUdeviceptr, int,
                  CUdeviceptr, const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*decode)(yvex_cuda_work *, const yvex_backend_attention_weight *, CUdeviceptr,
                  unsigned long long, unsigned long long, CUdeviceptr, CUdeviceptr,
                  const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*weighted_norm)(yvex_cuda_work *, CUdeviceptr, unsigned long long,
                         const yvex_backend_attention_weight *, CUdeviceptr, double,
                         CUdeviceptr, const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*unit_norm)(yvex_cuda_work *, CUdeviceptr, unsigned long long, unsigned long long,
                     double, CUdeviceptr, const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*rope)(yvex_cuda_work *, CUdeviceptr, unsigned long long, unsigned long long,
                unsigned long long, const yvex_backend_attention_position *, int, CUdeviceptr,
                const char *, yvex_backend_attention_failure *, yvex_error *);
    int (*activation)(yvex_cuda_work *, CUdeviceptr, unsigned long long, unsigned long long,
                      const yvex_backend_attention_activation *, CUdeviceptr, const char *,
                      yvex_backend_attention_failure *, yvex_error *);
} yvex_cuda_attention_operations;
const yvex_cuda_attention_operations *yvex_cuda_attention_operations_get(void);
int yvex_cuda_kernel_bundle_admit(yvex_backend *backend, yvex_error *err);
int yvex_cuda_kernel_bundle_close(yvex_backend *backend, yvex_error *err);
int yvex_cuda_query_capability(const yvex_backend *backend,
                               yvex_backend_operation_variant variant,
                               yvex_backend_capability_result *out,
                               yvex_error *err);
int yvex_cuda_require_capability(yvex_backend *backend,
                                 yvex_backend_operation_variant variant,
                                 const char *where,
                                 yvex_error *err);
int yvex_cuda_launch(yvex_backend *backend,
                     yvex_backend_operation_variant variant,
                     CUfunction function,
                     unsigned int grid_x,
                     unsigned int block_x,
                     unsigned int shared_bytes,
                     void **params,
                     const char *where,
                     yvex_error *err);
int yvex_cuda_synchronize(yvex_backend *backend,
                          yvex_backend_operation_variant variant,
                          const char *where,
                          yvex_error *err);
int yvex_cuda_temporary_free(yvex_backend *backend,
                             yvex_backend_operation_variant variant,
                             CUdeviceptr *ptr,
                             unsigned long long bytes,
                             int defer_on_failure,
                             const char *where,
                             yvex_error *err);
int yvex_cuda_deferred_release_drain(yvex_backend *backend, yvex_error *err);
int yvex_cuda_tensor_alloc(yvex_backend *backend,
                           const yvex_backend_tensor_desc *desc,
                           yvex_device_tensor **out,
                           yvex_error *err);
int yvex_cuda_tensor_free(yvex_backend *backend,
                          yvex_device_tensor *tensor,
                          yvex_error *err);
int yvex_cuda_tensor_write(yvex_backend *backend,
                           yvex_device_tensor *tensor,
                           const void *src,
                           unsigned long long len,
                           yvex_error *err);
int yvex_cuda_tensor_read(yvex_backend *backend,
                          const yvex_device_tensor *tensor,
                          void *dst,
                          unsigned long long len,
                          yvex_error *err);
int yvex_cuda_tensor_copy(yvex_backend *backend,
                          yvex_device_tensor *dst,
                          const yvex_device_tensor *src,
                          yvex_error *err);
int yvex_cuda_op_embed(yvex_backend *backend,
                       const yvex_device_tensor *embedding,
                       const unsigned int *token_ids,
                       unsigned long long token_count,
                       yvex_device_tensor *out,
                       yvex_error *err);
int yvex_cuda_op_rms_norm(yvex_backend *backend,
                          const yvex_device_tensor *input,
                          const yvex_device_tensor *weight,
                          float epsilon,
                          yvex_device_tensor *out,
                          yvex_error *err);
int yvex_cuda_op_rope(yvex_backend *backend,
                      const yvex_device_tensor *input,
                      unsigned long long position,
                      float rope_base,
                      yvex_device_tensor *out,
                      yvex_error *err);
int yvex_cuda_op_matmul(yvex_backend *backend,
                        const yvex_device_tensor *input,
                        const yvex_device_tensor *weight,
                        yvex_device_tensor *out,
                        yvex_error *err);
int yvex_cuda_op_mlp(yvex_backend *backend,
                     const yvex_device_tensor *input,
                     const yvex_device_tensor *gate_weight,
                     const yvex_device_tensor *up_weight,
                     const yvex_device_tensor *down_weight,
                     const yvex_mlp_options *options,
                     yvex_device_tensor *intermediate,
                     yvex_device_tensor *out,
                     yvex_error *err);
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
                           yvex_error *err);

/* Qtype contract. */
int yvex_cuda_quant_row_dot(yvex_backend *backend,
                            unsigned int qtype,
                            const unsigned char *encoded,
                            size_t encoded_bytes,
                            const float *vector,
                            unsigned long long elements,
                            float *out,
                            yvex_quant_failure *failure,
                            yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* SRC_BACKEND_CUDA_PRIVATE_H_INCLUDED */
