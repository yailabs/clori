/* Backends execute already-admitted operations; they neither reconstruct model policy nor fall
 * back across backend kinds. Owned resources remain associated with the backend that created
 * them until checked release succeeds. */
#ifndef YVEX_BACKEND_H
#define YVEX_BACKEND_H

#include <yvex/model.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Backend execution. */
typedef struct yvex_backend yvex_backend;
typedef struct yvex_device_tensor yvex_device_tensor;

typedef enum {
    YVEX_BACKEND_KIND_CPU = 0,
    YVEX_BACKEND_KIND_CUDA,
    YVEX_BACKEND_KIND_METAL,
    YVEX_BACKEND_KIND_ROCM
} yvex_backend_kind;

typedef enum {
    YVEX_BACKEND_STATUS_READY = 0,
    YVEX_BACKEND_STATUS_CONTEXT_READY,
    YVEX_BACKEND_STATUS_UNSUPPORTED,
    YVEX_BACKEND_STATUS_FAILED
} yvex_backend_status;

typedef struct {
    yvex_backend_kind kind;
    const char *device;
    unsigned long long memory_limit_bytes;
} yvex_backend_options;

typedef struct {
    unsigned long long allocated_bytes;
    unsigned long long allocation_count, allocation_events, release_events;
    unsigned long long h2d_bytes, d2h_bytes;
    unsigned long long peak_allocated_bytes;
    unsigned long long memory_limit_bytes;
} yvex_backend_memory_stats;

typedef struct {
    const char *name;
    yvex_dtype dtype;
    unsigned int rank;
    unsigned long long dims[YVEX_TENSOR_MAX_DIMS];
    unsigned long long bytes;
} yvex_backend_tensor_desc;

typedef struct {
    yvex_backend_kind kind;
    const char *name;
    int device_index;
    int compute_capability_major;
    int compute_capability_minor;
    unsigned long long global_memory_bytes;
    unsigned long long free_memory_bytes;
    unsigned long long total_memory_bytes;
    int unified_addressing;
    int managed_memory;
} yvex_backend_device_info;

typedef enum {
    YVEX_BACKEND_CAP_TENSOR_ALLOC = 0,
    YVEX_BACKEND_CAP_TENSOR_READ_WRITE,
    YVEX_BACKEND_CAP_OP_EMBED,
    YVEX_BACKEND_CAP_OP_MATMUL,
    YVEX_BACKEND_CAP_OP_MLP,
    YVEX_BACKEND_CAP_OP_RMS_NORM,
    YVEX_BACKEND_CAP_OP_ROPE,
    YVEX_BACKEND_CAP_OP_ATTENTION
} yvex_backend_capability;

typedef enum {
    YVEX_BACKEND_VARIANT_TENSOR_ALLOC = 0,
    YVEX_BACKEND_VARIANT_TENSOR_ZERO,
    YVEX_BACKEND_VARIANT_TENSOR_WRITE,
    YVEX_BACKEND_VARIANT_TENSOR_READ,
    YVEX_BACKEND_VARIANT_TENSOR_COPY,
    YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32,
    YVEX_BACKEND_VARIANT_EMBED_F16_TO_F32,
    YVEX_BACKEND_VARIANT_RMS_NORM_F32_WEIGHT_F32,
    YVEX_BACKEND_VARIANT_RMS_NORM_F32_WEIGHT_F16,
    YVEX_BACKEND_VARIANT_ROPE_F32,
    YVEX_BACKEND_VARIANT_MATMUL_F32,
    YVEX_BACKEND_VARIANT_MLP_DENSE_F32,
    YVEX_BACKEND_VARIANT_MLP_ROUTED_F32,
    YVEX_BACKEND_VARIANT_ATTENTION_CAUSAL_F32,
    YVEX_BACKEND_VARIANT_ATTENTION_NONCAUSAL_F32,
    YVEX_BACKEND_VARIANT_QTYPE_ROW_DOT,
    YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
    YVEX_BACKEND_VARIANT_COUNT
} yvex_backend_operation_variant;

typedef enum {
    YVEX_BACKEND_CAPABILITY_UNSUPPORTED = 0,
    YVEX_BACKEND_CAPABILITY_SUPPORTED,
    YVEX_BACKEND_CAPABILITY_FAILED
} yvex_backend_capability_state;

typedef enum {
    YVEX_BACKEND_CAPABILITY_REASON_NONE = 0,
    YVEX_BACKEND_CAPABILITY_REASON_DRIVER_UNAVAILABLE,
    YVEX_BACKEND_CAPABILITY_REASON_DEVICE_UNAVAILABLE,
    YVEX_BACKEND_CAPABILITY_REASON_CONTEXT_UNAVAILABLE,
    YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_ABSENT,
    YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_REJECTED,
    YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING,
    YVEX_BACKEND_CAPABILITY_REASON_VARIANT_UNSUPPORTED,
    YVEX_BACKEND_CAPABILITY_REASON_LAUNCH_FAILED,
    YVEX_BACKEND_CAPABILITY_REASON_SYNCHRONIZATION_FAILED,
    YVEX_BACKEND_CAPABILITY_REASON_CLEANUP_FAILED
} yvex_backend_capability_reason;

typedef struct {
    yvex_backend_kind backend_kind;
    yvex_backend_operation_variant variant;
    yvex_backend_capability_state state;
    yvex_backend_capability_reason reason;
    yvex_dtype input_dtype;
    yvex_dtype weight_dtype;
    yvex_dtype output_dtype;
    int context_available;
    int kernel_bundle_available;
    int function_available;
} yvex_backend_capability_result;

typedef struct {
    unsigned long long batch;
    unsigned long long hidden_dim;
    unsigned long long ffn_dim;
    unsigned long long expert_count;
    unsigned long long expert_id;
    int routed_expert_mode;
    int gated;
    const char *activation;
} yvex_mlp_options;

/* A cleanup failure may return an error with a non-null owner for checked close retry. */
int yvex_backend_open(yvex_backend **out,
                      const yvex_backend_options *options,
                      yvex_error *err);
int yvex_backend_open_cpu(yvex_backend **out, yvex_error *err);
int yvex_backend_cuda_context_available(void);
int yvex_backend_cuda_available(void);
/* Checked close retains a non-null owner when resource discharge must be retried. */
int yvex_backend_close_checked(yvex_backend **backend, yvex_error *err);
/* Compatibility close is best-effort; completion-sensitive owners use checked close. */
void yvex_backend_close(yvex_backend *backend);

yvex_backend_kind yvex_backend_kind_of(const yvex_backend *backend);
const char *yvex_backend_kind_name(yvex_backend_kind kind);
int yvex_backend_kind_parse(const char *name,
                            yvex_backend_kind *out,
                            yvex_error *err);
yvex_backend_status yvex_backend_status_of(const yvex_backend *backend);
const char *yvex_backend_status_name(yvex_backend_status status);

int yvex_backend_get_memory_stats(const yvex_backend *backend,
                                  yvex_backend_memory_stats *out,
                                  yvex_error *err);
int yvex_backend_get_device_info(const yvex_backend *backend,
                                 yvex_backend_device_info *out,
                                 yvex_error *err);

int yvex_backend_tensor_alloc(yvex_backend *backend,
                              const yvex_backend_tensor_desc *desc,
                              yvex_device_tensor **out,
                              yvex_error *err);
void yvex_backend_tensor_free(yvex_backend *backend,
                              yvex_device_tensor *tensor);
int yvex_backend_tensor_release(yvex_backend *backend,
                                yvex_device_tensor **tensor,
                                yvex_error *err);

const char *yvex_device_tensor_name(const yvex_device_tensor *tensor);
unsigned long long yvex_device_tensor_bytes(const yvex_device_tensor *tensor);
int yvex_device_tensor_is_written(const yvex_device_tensor *tensor);

int yvex_backend_tensor_write(yvex_backend *backend,
                              yvex_device_tensor *tensor,
                              const void *src,
                              unsigned long long len,
                              yvex_error *err);
int yvex_backend_tensor_read(yvex_backend *backend,
                             const yvex_device_tensor *tensor,
                             void *dst,
                             unsigned long long len,
                             yvex_error *err);
int yvex_backend_tensor_copy(yvex_backend *backend,
                             yvex_device_tensor *dst,
                             const yvex_device_tensor *src,
                             yvex_error *err);
int yvex_backend_sync(yvex_backend *backend, yvex_error *err);

int yvex_backend_supports(const yvex_backend *backend,
                          yvex_backend_capability capability);
const char *yvex_backend_capability_name(yvex_backend_capability capability);
int yvex_backend_query_capability(const yvex_backend *backend,
                                  yvex_backend_operation_variant variant,
                                  yvex_backend_capability_result *out,
                                  yvex_error *err);
const char *yvex_backend_operation_variant_name(yvex_backend_operation_variant variant);
const char *yvex_backend_capability_state_name(yvex_backend_capability_state state);
const char *yvex_backend_capability_reason_name(yvex_backend_capability_reason reason);
int yvex_backend_op_embed(yvex_backend *backend,
                          const yvex_device_tensor *embedding,
                          const unsigned int *token_ids,
                          unsigned long long token_count,
                          yvex_device_tensor *out,
                          yvex_error *err);

int yvex_backend_op_rms_norm(yvex_backend *backend,
                             const yvex_device_tensor *input,
                             const yvex_device_tensor *weight,
                             float epsilon,
                             yvex_device_tensor *out,
                             yvex_error *err);

int yvex_backend_op_rope(yvex_backend *backend,
                         const yvex_device_tensor *input,
                         unsigned long long position,
                         float rope_base,
                         yvex_device_tensor *out,
                         yvex_error *err);

int yvex_backend_op_matmul(yvex_backend *backend,
                           const yvex_device_tensor *input,
                           const yvex_device_tensor *weight,
                           yvex_device_tensor *out,
                           yvex_error *err);

int yvex_backend_op_mlp(yvex_backend *backend,
                        const yvex_device_tensor *input,
                        const yvex_device_tensor *gate_weight,
                        const yvex_device_tensor *up_weight,
                        const yvex_device_tensor *down_weight,
                        const yvex_mlp_options *options,
                        yvex_device_tensor *intermediate,
                        yvex_device_tensor *out,
                        yvex_error *err);

int yvex_backend_op_attention(yvex_backend *backend,
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

#ifdef __cplusplus
}
#endif

#endif /* YVEX_BACKEND_H */
