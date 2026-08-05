/* Backend implementations share lifecycle, capability, tensor, and report types here;
 * model topology and family policy remain outside this non-installed ABI. */
#ifndef INCLUDE_YVEX_INTERNAL_BACKEND_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_BACKEND_H_INCLUDED
#include <limits.h>
#include <stdatomic.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/internal/core.h>
#include <yvex/model.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Encoded attention is a private graph/backend ABI, never installed capability surface. */
#define YVEX_BACKEND_ATTENTION_JOB_SCHEMA 3u
typedef enum {
    YVEX_BACKEND_ATTENTION_WEIGHT_Q_A = 0,
    YVEX_BACKEND_ATTENTION_WEIGHT_Q_A_NORM,
    YVEX_BACKEND_ATTENTION_WEIGHT_Q_B,
    YVEX_BACKEND_ATTENTION_WEIGHT_KV,
    YVEX_BACKEND_ATTENTION_WEIGHT_KV_NORM,
    YVEX_BACKEND_ATTENTION_WEIGHT_SINKS,
    YVEX_BACKEND_ATTENTION_WEIGHT_OUT_A,
    YVEX_BACKEND_ATTENTION_WEIGHT_OUT_B,
    YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_KV,
    YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_GATE,
    YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_APE,
    YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_NORM,
    YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_KV,
    YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_GATE,
    YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_APE,
    YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_NORM,
    YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_Q,
    YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_PROJECTION,
    YVEX_BACKEND_ATTENTION_WEIGHT_MHC_FUNCTION,
    YVEX_BACKEND_ATTENTION_WEIGHT_MHC_BASE,
    YVEX_BACKEND_ATTENTION_WEIGHT_MHC_SCALE,
    YVEX_BACKEND_ATTENTION_WEIGHT_INPUT_NORM,
    YVEX_BACKEND_ATTENTION_WEIGHT_COUNT
} yvex_backend_attention_weight_slot;
typedef struct {
    const unsigned char *encoded;
    size_t encoded_bytes;
    unsigned long long row_bytes, row_width, row_count;
    unsigned int qtype;
    int present;
} yvex_backend_attention_weight;
typedef struct {
    unsigned long long theta, scaling_factor, original_context;
    unsigned long long beta_fast, beta_slow, rope_dimensions;
} yvex_backend_attention_position;
typedef struct {
    int required;
    unsigned long long block_width;
    unsigned int quantization;
    int hadamard;
} yvex_backend_attention_activation;
typedef enum {
    YVEX_BACKEND_ATTENTION_SWA = 0,
    YVEX_BACKEND_ATTENTION_CSA,
    YVEX_BACKEND_ATTENTION_HCA
} yvex_backend_attention_class;
typedef enum {
    YVEX_BACKEND_ATTENTION_COMPUTE_UNKNOWN = 0,
    YVEX_BACKEND_ATTENTION_COMPUTE_BF16_F32_RNE_V1
} yvex_backend_attention_compute;
typedef enum {
    YVEX_BACKEND_ATTENTION_SCOPE_CORE = 0,
    YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
} yvex_backend_attention_scope;
typedef enum {
    YVEX_BACKEND_ATTENTION_PHASE_DECODE = 0,
    YVEX_BACKEND_ATTENTION_PHASE_PREFILL,
    YVEX_BACKEND_ATTENTION_PHASE_MIXED,
    YVEX_BACKEND_ATTENTION_PHASE_SPECULATIVE_DRAFT,
    YVEX_BACKEND_ATTENTION_PHASE_SPECULATIVE_VERIFY
} yvex_backend_attention_phase;
typedef int (*yvex_backend_cancelled_fn)(void *context);
typedef struct { yvex_backend_cancelled_fn requested; void *context; } yvex_backend_cancellation;
typedef struct { float *data; unsigned long long capacity; } yvex_backend_float_span;
typedef struct { unsigned long long *data, capacity; } yvex_backend_u64_span;
typedef struct {
    int present, overlap;
    unsigned long long next_token_position, ratio, head_dimension, state_width, state_slots;
    unsigned long long cursor, previous_fill, current_fill;
    const float *kv_state, *score_state;
    unsigned long long kv_state_capacity, score_state_capacity;
} yvex_backend_attention_rolling;
typedef struct {
    unsigned int schema;
    yvex_backend_attention_phase phase;
    yvex_backend_attention_class attention_class;
    yvex_backend_attention_compute compute_contract;
    yvex_backend_attention_scope operation_scope;
    unsigned long long token_position, token_count, input_stride;
    unsigned long long hidden_width, q_rank, query_heads, head_dimension, kv_width;
    unsigned long long sliding_window, compression_ratio;
    unsigned long long output_groups, output_group_input_width, output_rank;
    unsigned long long indexer_heads, indexer_head_dimension, indexer_topk;
    unsigned long long residual_stream_count, residual_stream_width, residual_expanded_width;
    unsigned long long mhc_mixing_rows, mhc_sinkhorn_iterations;
    double rms_epsilon, mhc_epsilon, mhc_residual_post_multiplier;
    yvex_backend_attention_position position;
    yvex_backend_attention_activation attention_kv_activation, compressor_activation;
    yvex_backend_attention_activation compressor_rotated_activation, indexer_query_activation;
    yvex_backend_attention_weight weights[YVEX_BACKEND_ATTENTION_WEIGHT_COUNT];
    const float *input, *local_kv, *compressed_kv, *indexer_kv;
    const yvex_device_tensor *device_input; yvex_device_tensor *device_output;
    const unsigned long long *local_positions, *compressed_positions, *indexer_positions;
    unsigned long long local_count, local_stride, compressed_count, compressed_stride;
    unsigned long long indexer_count, indexer_stride;
    yvex_backend_attention_rolling main_rolling, indexer_rolling;
    const yvex_backend_cancellation *cancellation;
    unsigned int evidence_level;
    int candidate_block_visible, retain_prefix_checkpoints;
    unsigned long long max_host_bytes, max_device_bytes;
} yvex_backend_attention_job;
typedef struct {
    yvex_backend_float_span core_input, q_low, query, raw_kv;
    yvex_backend_float_span compressed_kv, indexer_kv, index_query, index_weights;
    yvex_backend_float_span attention_values, output, envelope_output;
    yvex_backend_u64_span compressed_positions, indexer_positions, topk_positions;
    yvex_backend_u64_span topk_counts, valid_candidate_counts;
    yvex_backend_float_span main_kv_state, main_score_state;
    yvex_backend_float_span indexer_kv_state, indexer_score_state;
    unsigned long long tokens_executed, compressed_count, indexer_count;
    unsigned long long topk_count, valid_candidate_count;
    unsigned long long host_bytes, peak_host_bytes, device_bytes, peak_device_bytes;
    unsigned long long kernel_launches, h2d_bytes, d2h_bytes, d2d_bytes;
    unsigned long long stream_synchronizations, device_synchronizations;
    unsigned long long device_execution_elapsed_ns, host_workspace_capacity;
    unsigned long long host_workspace_used, host_workspace_peak, host_workspace_allocation_count;
    int host_workspace_reused;
} yvex_backend_attention_output;
typedef enum {
    YVEX_BACKEND_ATTENTION_FAILURE_NONE = 0,
    YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
    YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
    YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
    YVEX_BACKEND_ATTENTION_FAILURE_ALLOCATION,
    YVEX_BACKEND_ATTENTION_FAILURE_COPY,
    YVEX_BACKEND_ATTENTION_FAILURE_LAUNCH,
    YVEX_BACKEND_ATTENTION_FAILURE_SYNCHRONIZE,
    YVEX_BACKEND_ATTENTION_FAILURE_NUMERIC,
    YVEX_BACKEND_ATTENTION_FAILURE_CANCELLED,
    YVEX_BACKEND_ATTENTION_FAILURE_CLEANUP
} yvex_backend_attention_failure_code;
typedef struct {
    yvex_backend_attention_failure_code code;
    const char *stage;
    unsigned long long expected, actual;
} yvex_backend_attention_failure;
#define YVEX_BACKEND_BANDWIDTH_SCHEMA_V1 1u
#define YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT 5u
typedef struct {
    unsigned int schema_version;
    unsigned long long working_set_bytes, iterations, sample_count;
    unsigned long long stream_elapsed_ns[YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT];
    unsigned long long copy_elapsed_ns[YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT];
    unsigned long long coherent_host_elapsed_ns[YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT];
    unsigned long long sustainable_read_bytes_per_second, sustainable_copy_bytes_per_second;
    unsigned long long sustainable_coherent_host_bytes_per_second;
    char kernel_bundle_identity[YVEX_SHA256_HEX_BYTES], identity[YVEX_SHA256_HEX_BYTES];
} yvex_backend_bandwidth_evidence;
int yvex_backend_attention_execute(yvex_backend *backend, const yvex_backend_attention_job *job,
                                   yvex_backend_attention_output *output,
                                   yvex_backend_attention_failure *failure, yvex_error *err);
typedef struct yvex_backend_vtable {
    int (*close)(yvex_backend *backend, yvex_error *err);
    int (*memory_stats)(const yvex_backend *backend, yvex_backend_memory_stats *out, yvex_error *err);
    int (*device_info)(const yvex_backend *backend, yvex_backend_device_info *out, yvex_error *err);
    int (*bandwidth_probe)(yvex_backend *backend, yvex_backend_bandwidth_evidence *out,
                           yvex_error *err);
    int (*tensor_alloc)(yvex_backend *backend, const yvex_backend_tensor_desc *desc,
                        yvex_device_tensor **out, yvex_error *err);
    int (*resident_alloc)(yvex_backend *backend, const yvex_backend_tensor_desc *desc,
                          yvex_device_tensor **out, unsigned char **host, yvex_error *err);
    int (*tensor_free)(yvex_backend *backend, yvex_device_tensor *tensor, yvex_error *err);
    int (*tensor_write)(yvex_backend *backend, yvex_device_tensor *tensor, const void *src,
                        unsigned long long len, yvex_error *err);
    int (*tensor_read)(yvex_backend *backend, const yvex_device_tensor *tensor, void *dst,
                       unsigned long long len, yvex_error *err);
    int (*tensor_copy)(yvex_backend *backend, yvex_device_tensor *dst,
                       const yvex_device_tensor *src, yvex_error *err);
    int (*sync)(yvex_backend *backend, yvex_error *err);
    int (*query_capability)(const yvex_backend *backend,
                            yvex_backend_operation_variant variant,
                            yvex_backend_capability_result *out, yvex_error *err);
    int (*op_embed)(yvex_backend *backend, const yvex_device_tensor *embedding,
                    const unsigned int *token_ids, unsigned long long token_count,
                    yvex_device_tensor *out, yvex_error *err);
    int (*op_rms_norm)(yvex_backend *backend, const yvex_device_tensor *input,
                       const yvex_device_tensor *weight, float epsilon,
                       yvex_device_tensor *out, yvex_error *err);
    int (*op_rope)(yvex_backend *backend, const yvex_device_tensor *input,
                   unsigned long long position, float rope_base,
                   yvex_device_tensor *out, yvex_error *err);
    int (*op_matmul)(yvex_backend *backend, const yvex_device_tensor *input,
                     const yvex_device_tensor *weight, yvex_device_tensor *out, yvex_error *err);
    int (*op_mlp)(yvex_backend *backend, const yvex_device_tensor *input,
                  const yvex_device_tensor *gate_weight, const yvex_device_tensor *up_weight,
                  const yvex_device_tensor *down_weight, const yvex_mlp_options *options,
                  yvex_device_tensor *intermediate, yvex_device_tensor *out, yvex_error *err);
    int (*op_attention)(yvex_backend *backend, const yvex_device_tensor *query,
                        const yvex_device_tensor *keys, const yvex_device_tensor *values,
                        unsigned long long seq_len, unsigned long long position, float scale, int causal,
                        yvex_device_tensor *score_scratch, yvex_device_tensor *probability_scratch,
                        yvex_device_tensor *out, yvex_error *err);
    int (*host_workspace_alloc)(yvex_backend *backend, size_t bytes,
                                unsigned char **out, yvex_error *err);
    int (*host_workspace_free)(yvex_backend *backend, unsigned char **base,
                               yvex_error *err);
} yvex_backend_vtable;
typedef int (*yvex_backend_state_resolve_fn)(
    const void *context, const void *host, unsigned long long bytes,
    unsigned long long *device_address);
struct yvex_backend {
    yvex_backend_kind kind;
    _Atomic(yvex_backend_status) status;
    const yvex_backend_vtable *vtable;
    yvex_backend_memory_stats stats;
    yvex_backend_device_info device_info;
    char device_name_storage[128];
    void *impl;
    struct yvex_backend *resource_owner;
    _Atomic unsigned long long lifecycle;
    unsigned long long tensor_id_next, resident_host_bytes;
    const unsigned char *resident_host_base;
    const yvex_device_tensor *resident_device_tensor;
    unsigned long long resident_device_address, resident_generation;
    const void *state_residency_context;
    yvex_backend_state_resolve_fn state_residency_resolve;
    unsigned long long state_residency_generation;
    const yvex_device_tensor *workspace_device_tensor;
    unsigned long long workspace_device_address, workspace_bytes, workspace_cursor;
    unsigned long long workspace_peak, workspace_generation;
    unsigned char *host_workspace_base;
    unsigned long long host_workspace_bytes, host_workspace_cursor, host_workspace_peak;
    unsigned long long host_workspace_generation, host_workspace_allocation_count;
    int host_workspace_owned, host_workspace_pinned, shared_owner_registered;
};
#define YVEX_BACKEND_LIFECYCLE_CLOSING (1ull << 63)
#define YVEX_BACKEND_LIFECYCLE_CHILD_MASK (YVEX_BACKEND_LIFECYCLE_CLOSING - 1ull)
/* Report whether checked close has made one backend cleanup-only. */
static inline int backend_cleanup_only(const yvex_backend *backend)
{
    return backend &&
           (atomic_load_explicit(&backend->lifecycle, memory_order_acquire) &
            YVEX_BACKEND_LIFECYCLE_CLOSING) != 0ull;
}
/* Refuse numerical or allocating dispatch through a terminal backend owner. */
static inline int backend_dispatch_admit(const yvex_backend *backend,
                                         const char *where, yvex_error *err)
{
    if (!backend || backend_cleanup_only(backend) ||
        backend->status == YVEX_BACKEND_STATUS_FAILED) {
        yvex_error_set(err, YVEX_ERR_STATE, where,
                       "backend is retained for cleanup only");
        return YVEX_ERR_STATE;
    }
    return YVEX_OK;
}
int yvex_backend_close_admit(yvex_backend *backend, yvex_error *err);
/* Project one exact admitted variant without promoting an aggregate capability. */
static inline int backend_variant_supported(const yvex_backend *backend,
                                            yvex_backend_operation_variant variant)
{
    yvex_backend_capability_result result;
    yvex_error err;
    yvex_error_clear(&err);
    return yvex_backend_query_capability(backend, variant, &result, &err) == YVEX_OK &&
           result.state == YVEX_BACKEND_CAPABILITY_SUPPORTED;
}
/* Admission failure publishes only a FAILED cleanup owner when checked rollback also fails. */
int yvex_backend_open_shared_cuda(yvex_backend **out, yvex_backend *context_owner,
                                  unsigned long long memory_limit_bytes, yvex_error *err);
struct yvex_device_tensor {
    yvex_backend *owner;
    unsigned long long owner_id;
    char *name;
    yvex_dtype dtype;
    unsigned int rank;
    unsigned long long dims[YVEX_TENSOR_MAX_DIMS];
    unsigned long long bytes;
    unsigned char *data, *host_data;
    int is_written, host_accessible;
};
/* Prove exact F32 storage geometry for one backend-owned tensor view. */
static inline int backend_tensor_f32_elements(const yvex_device_tensor *tensor,
                                              unsigned long long elements)
{
    return tensor && elements <= ULLONG_MAX / sizeof(float) &&
           tensor->bytes == elements * (unsigned long long)sizeof(float);
}
/*
 * Commit one admitted allocation to the backend counters.
 *
 * None after the caller's overflow and capacity admission. Concrete storage ownership remains with
 * the backend implementation.
 */
static inline void backend_memory_acquire(yvex_backend *backend, unsigned long long bytes)
{
    backend->stats.allocated_bytes += bytes;
    backend->stats.allocation_count += 1ull;
    backend->stats.allocation_events += 1ull;
    if (backend->stats.allocated_bytes > backend->stats.peak_allocated_bytes)
        backend->stats.peak_allocated_bytes = backend->stats.allocated_bytes;
}
/* Release one owned allocation from the backend counters. */
static inline void backend_memory_release(yvex_backend *backend, unsigned long long bytes)
{
    backend->stats.allocated_bytes =
        backend->stats.allocated_bytes >= bytes ? backend->stats.allocated_bytes - bytes : 0ull;
    if (backend->stats.allocation_count > 0ull) {
        backend->stats.allocation_count -= 1ull;
        backend->stats.release_events += 1ull;
    }
}
/* Admit one tensor only when its live owner identity matches the backend. */
static inline int backend_tensor_owner_is(const yvex_backend *backend,
                                          const yvex_device_tensor *tensor)
{
    return backend && tensor && tensor->owner == backend && tensor->owner_id != 0ull;
}
int yvex_backend_tensor_f32_subview(
    const yvex_device_tensor *source, unsigned long long offset,
    unsigned long long count, yvex_device_tensor *view);
int yvex_backend_bandwidth_probe(yvex_backend *backend, yvex_backend_bandwidth_evidence *out, yvex_error *err);
/* Rewind a serialized device workspace while preserving its stable address and peak. */
static inline void backend_workspace_reset(yvex_backend *backend)
{
    if (backend) backend->workspace_cursor = 0ull;
}
/* Rewind a serialized host workspace while preserving its storage and peak. */
static inline void backend_host_workspace_reset(yvex_backend *backend)
{
    if (backend) backend->host_workspace_cursor = 0ull;
}
double yvex_backend_nth_root(double value, unsigned long long degree);
int yvex_backend_memory_can_add(const yvex_backend *backend, unsigned long long bytes,
                                const char *backend_name, const char *where, yvex_error *err);
int yvex_backend_tensor_rw_validate(const char *where, const yvex_backend *backend,
                                    const yvex_device_tensor *tensor, unsigned long long len,
                                    yvex_error *err);
int yvex_backend_tensor_copy_validate(const yvex_backend *backend,
                                      const yvex_device_tensor *dst, const yvex_device_tensor *src,
                                      const char *where, yvex_error *err);
int yvex_backend_tensor_same_shape(const yvex_device_tensor *a, const yvex_device_tensor *b);
typedef enum {
    YVEX_BACKEND_RESIDENT_INVALID = -1,
    YVEX_BACKEND_RESIDENT_MISS = 0,
    YVEX_BACKEND_RESIDENT_HIT = 1
} yvex_backend_resident_result;
typedef struct {
    int attached, owned, pinned;
    unsigned long long capacity, used, peak, generation, allocation_count;
} yvex_backend_host_workspace_summary;
int yvex_backend_resident_attach(yvex_backend *backend, const unsigned char *host_base,
                                 unsigned long long bytes, const yvex_device_tensor *device_tensor,
                                 unsigned long long generation, yvex_error *err);
int yvex_backend_resident_detach(yvex_backend *backend, yvex_error *err);
int yvex_backend_resident_resolve(const yvex_backend *backend, const unsigned char *host,
                                  unsigned long long bytes, unsigned long long *device_address);
typedef struct yvex_backend_cuda_operation_facts {
    unsigned long long h2d_bytes, d2h_bytes, d2d_bytes, kernel_launches, upload_count, download_count;
    unsigned long long stream_synchronizations, device_synchronizations, active_weight_bytes, state_bytes;
    unsigned long long activation_bytes, temporary_bytes;
    int compulsory_memory_facts_available;
} yvex_backend_cuda_operation_facts;
int yvex_backend_cuda_encoded_matvec(yvex_backend *backend, const unsigned char *resident_encoded,
    unsigned long long encoded_bytes, unsigned int qtype, unsigned long long row_count,
    unsigned long long row_width, unsigned long long row_bytes, unsigned long long input_rows,
    const yvex_device_tensor *input, const yvex_device_tensor *additive,
    yvex_device_tensor *output,
    yvex_backend_cuda_operation_facts *facts, yvex_error *err);
int yvex_backend_state_residency_attach(yvex_backend *backend, const void *context,
    yvex_backend_state_resolve_fn resolve, unsigned long long generation, yvex_error *err);
void yvex_backend_state_residency_detach(yvex_backend *backend);
int yvex_backend_state_residency_resolve(
    const yvex_backend *backend, const void *host, unsigned long long bytes,
    unsigned long long *device_address);
int yvex_backend_workspace_attach(yvex_backend *backend,
                                  const yvex_device_tensor *device_tensor, unsigned long long generation,
                                  yvex_error *err);
void yvex_backend_workspace_detach(yvex_backend *backend);
int yvex_backend_workspace_acquire(yvex_backend *backend, unsigned long long bytes,
                                   unsigned long long alignment, unsigned long long *device_address);
int yvex_backend_host_workspace_prepare_owned(yvex_backend *backend, unsigned long long bytes,
                                              yvex_error *err);
int yvex_backend_host_workspace_detach(yvex_backend *backend, yvex_error *err);
int yvex_backend_host_workspace_acquire(yvex_backend *backend,
                                        unsigned long long bytes, unsigned long long alignment,
                                        void **address);
int yvex_backend_host_workspace_summary_get(
    const yvex_backend *backend, yvex_backend_host_workspace_summary *summary);
int yvex_backend_validate_rope(const yvex_device_tensor *tensor, unsigned long long *head_dim,
                               const char *where, yvex_error *err);
int yvex_backend_validate_embed(const yvex_backend *backend,
                                const yvex_device_tensor *embedding, const unsigned int *token_ids,
                                unsigned long long token_count, const yvex_device_tensor *out,
                                unsigned long long *hidden_size, unsigned long long *vocab_size,
                                const char *unsupported_message, const char *where, yvex_error *err);
int yvex_backend_validate_rms_norm(const yvex_backend *backend,
                                   const yvex_device_tensor *input, const yvex_device_tensor *weight,
                                   float epsilon, const yvex_device_tensor *out,
                                   unsigned long long *hidden_size, const char *unsupported_message,
                                   const char *where, yvex_error *err);
int yvex_backend_validate_matmul(const yvex_backend *backend,
                                 const yvex_device_tensor *input, const yvex_device_tensor *weight,
                                 const yvex_device_tensor *out, unsigned long long *m,
                                 unsigned long long *k, unsigned long long *n,
                                 const char *where, yvex_error *err);
int yvex_backend_validate_attention(const yvex_backend *backend,
                                    const yvex_device_tensor *query, const yvex_device_tensor *keys,
                                    const yvex_device_tensor *values, unsigned long long seq_len,
                                    unsigned long long position, yvex_device_tensor *score_scratch,
                                    yvex_device_tensor *probability_scratch, yvex_device_tensor *out,
                                    unsigned long long *head_dim, unsigned long long *kv_elements,
                                    const char *where, yvex_error *err);
int yvex_backend_validate_mlp(const yvex_backend *backend,
                              const yvex_device_tensor *input, const yvex_device_tensor *gate_weight,
                              const yvex_device_tensor *up_weight, const yvex_device_tensor *down_weight,
                              const yvex_mlp_options *options, const yvex_device_tensor *intermediate,
                              const yvex_device_tensor *out, unsigned long long *batch,
                              unsigned long long *hidden_dim, unsigned long long *ffn_dim,
                              unsigned long long *gate_offset, unsigned long long *up_offset,
                              unsigned long long *down_offset, const char *where, yvex_error *err);
int yvex_backend_open_cuda_impl(yvex_backend **out, const char *device,
                                unsigned long long memory_limit_bytes, yvex_error *err);
/* CUDA launch-graph lifecycle shared by runtime execution sessions. */
#define YVEX_BACKEND_CUDA_GRAPH_SCHEMA 1u
#define YVEX_BACKEND_CUDA_GRAPH_IDENTITY_CAP 65u
typedef struct yvex_backend_cuda_graph yvex_backend_cuda_graph;
typedef enum {
    YVEX_BACKEND_CUDA_GRAPH_UNAVAILABLE = 0,
    YVEX_BACKEND_CUDA_GRAPH_OPEN,
    YVEX_BACKEND_CUDA_GRAPH_CAPTURING,
    YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED,
    YVEX_BACKEND_CUDA_GRAPH_INVALIDATED,
    YVEX_BACKEND_CUDA_GRAPH_FAILED
} yvex_backend_cuda_graph_state;
typedef enum {
    YVEX_BACKEND_CUDA_GRAPH_REASON_NONE = 0,
    YVEX_BACKEND_CUDA_GRAPH_REASON_NOT_CUDA,
    YVEX_BACKEND_CUDA_GRAPH_REASON_CONTEXT_UNAVAILABLE,
    YVEX_BACKEND_CUDA_GRAPH_REASON_BACKEND_FAILED,
    YVEX_BACKEND_CUDA_GRAPH_REASON_STREAM_API_UNAVAILABLE,
    YVEX_BACKEND_CUDA_GRAPH_REASON_GRAPH_API_UNAVAILABLE,
    YVEX_BACKEND_CUDA_GRAPH_REASON_UPDATE_API_UNAVAILABLE,
    YVEX_BACKEND_CUDA_GRAPH_REASON_BUSY,
    YVEX_BACKEND_CUDA_GRAPH_REASON_INVALID_STATE,
    YVEX_BACKEND_CUDA_GRAPH_REASON_CAPTURE_FAILED,
    YVEX_BACKEND_CUDA_GRAPH_REASON_EMPTY_CAPTURE,
    YVEX_BACKEND_CUDA_GRAPH_REASON_INSTANTIATE_FAILED,
    YVEX_BACKEND_CUDA_GRAPH_REASON_UPLOAD_FAILED,
    YVEX_BACKEND_CUDA_GRAPH_REASON_UPDATE_INCOMPATIBLE,
    YVEX_BACKEND_CUDA_GRAPH_REASON_LAUNCH_FAILED,
    YVEX_BACKEND_CUDA_GRAPH_REASON_SYNCHRONIZE_FAILED,
    YVEX_BACKEND_CUDA_GRAPH_REASON_CLEANUP_FAILED
} yvex_backend_cuda_graph_reason;
typedef enum {
    YVEX_BACKEND_CUDA_CAPTURE_GLOBAL = 0,
    YVEX_BACKEND_CUDA_CAPTURE_THREAD_LOCAL,
    YVEX_BACKEND_CUDA_CAPTURE_RELAXED
} yvex_backend_cuda_capture_mode;
typedef struct {
    unsigned int schema;
    yvex_backend_cuda_graph_state state;
    yvex_backend_cuda_graph_reason reason;
    int stream_api_available, graph_api_available, update_api_available;
    int edge_inventory_available, async_memory_available, async_copy_available;
    int pinned_host_memory_available, event_timing_available;
} yvex_backend_cuda_graph_capability;
typedef struct {
    unsigned int schema;
    yvex_backend_cuda_capture_mode capture_mode;
    const char *compatibility_identity;
} yvex_backend_cuda_graph_options;
typedef struct {
    unsigned long long node_count, edge_count, kernel_node_count, memcpy_node_count;
    unsigned long long memset_node_count, host_node_count, child_graph_node_count;
    unsigned long long event_node_count, memory_node_count, other_node_count;
} yvex_backend_cuda_graph_inventory;
typedef struct {
    unsigned int schema;
    yvex_backend_cuda_graph_state state;
    yvex_backend_cuda_graph_reason reason;
    yvex_backend_cuda_capture_mode capture_mode;
    int uploaded;
    yvex_backend_cuda_graph_inventory inventory;
    unsigned long long capture_count, instantiate_count, upload_count, update_count;
    unsigned long long launch_count, replay_count, synchronize_count, invalidation_count;
    unsigned long long capture_elapsed_ns, instantiate_elapsed_ns, last_update_elapsed_ns;
    unsigned long long last_replay_elapsed_ns, last_device_elapsed_ns;
    char launch_graph_identity[YVEX_SHA256_HEX_BYTES], graph_exec_identity[YVEX_SHA256_HEX_BYTES];
} yvex_backend_cuda_graph_info;
/* Session-selected CUDA attention execution over the graph lifecycle above. */
#define YVEX_BACKEND_CUDA_ATTENTION_GRAPH_SCHEMA 1u
#define YVEX_BACKEND_CUDA_CAPTURE_BUCKET_CAP 64u
#define YVEX_BACKEND_CUDA_GRAPH_KEY_CAP 160u
typedef enum {
    YVEX_BACKEND_CUDA_ATTENTION_EAGER = 0,
    YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE,
    YVEX_BACKEND_CUDA_ATTENTION_FULL
} yvex_backend_cuda_attention_mode;
struct yvex_attention_workspace_recipe;
int yvex_backend_attention_workspace_required_from_recipe(
    const struct yvex_attention_workspace_recipe *recipe,
    unsigned long long *required_bytes, yvex_error *err);
typedef struct {
    unsigned int schema;
    int configured;
    int kernel_bundle_native;
    yvex_backend_cuda_attention_mode selected_mode;
    unsigned long long graph_count, piece_count;
    unsigned long long capture_count, instantiate_count, replay_count;
    unsigned long long upload_count, update_count, update_pending_count, launch_count;
    unsigned long long node_count, kernel_node_count;
    unsigned long long memcpy_node_count, memset_node_count;
    unsigned long long capture_elapsed_ns, instantiate_elapsed_ns, last_update_elapsed_ns;
    unsigned long long last_replay_elapsed_ns, last_device_elapsed_ns;
    unsigned long long invalidation_count;
    int driver_version;
    char compatibility_identity[YVEX_BACKEND_CUDA_GRAPH_IDENTITY_CAP];
    char capture_bucket[YVEX_BACKEND_CUDA_CAPTURE_BUCKET_CAP], kernel_bundle_architecture[16];
    char cuda_build_identity[YVEX_SHA256_HEX_BYTES];
    char launch_graph_identity[YVEX_SHA256_HEX_BYTES], graph_exec_identity[YVEX_SHA256_HEX_BYTES];
} yvex_backend_cuda_attention_graph_summary;
typedef struct {
    unsigned int schema;
    unsigned long long index;
    int update_requested;
    char compatibility_identity[YVEX_BACKEND_CUDA_GRAPH_KEY_CAP];
    yvex_backend_cuda_graph_info graph;
} yvex_backend_cuda_attention_graph_entry;
int yvex_backend_cuda_graph_query(const yvex_backend *backend,
                                  yvex_backend_cuda_graph_capability *out, yvex_error *err);
int yvex_backend_cuda_attention_configure(
    yvex_backend *backend, yvex_backend_cuda_attention_mode mode,
    const char *compatibility_identity, const char *capture_bucket,
    unsigned long long local_capacity, unsigned long long compressed_capacity,
    unsigned long long indexer_capacity, yvex_error *err);
int yvex_backend_cuda_attention_graph_summary_get(
    const yvex_backend *backend, yvex_backend_cuda_attention_graph_summary *out, yvex_error *err);
int yvex_backend_cuda_attention_graph_registry_count(
    const yvex_backend *backend, unsigned long long *count, yvex_error *err);
int yvex_backend_cuda_attention_graph_registry_get(
    const yvex_backend *backend, unsigned long long index,
    yvex_backend_cuda_attention_graph_entry *out, yvex_error *err);
typedef enum {
    YVEX_BACKEND_CUDA_GRAPH_REGISTRY_UPDATE = 0,
    YVEX_BACKEND_CUDA_GRAPH_REGISTRY_INVALIDATE,
    YVEX_BACKEND_CUDA_GRAPH_REGISTRY_RELEASE
} yvex_backend_cuda_graph_registry_action;
int yvex_backend_cuda_attention_graph_registry_apply(
    yvex_backend *backend, yvex_backend_cuda_graph_registry_action action,
    unsigned long long *affected, yvex_error *err);
/* Immutable backend reports. */
typedef enum {
    YVEX_BACKEND_REPORT_CAPABILITIES = 0,
    YVEX_BACKEND_REPORT_CUDA_INFO,
    YVEX_BACKEND_REPORT_CUDA_BANDWIDTH
} yvex_backend_report_kind;
typedef enum {
    YVEX_BACKEND_BUNDLE_NOT_APPLICABLE = 0,
    YVEX_BACKEND_BUNDLE_ABSENT,
    YVEX_BACKEND_BUNDLE_ADMITTED,
    YVEX_BACKEND_BUNDLE_REJECTED
} yvex_backend_bundle_admission;
typedef struct {
    yvex_backend_report_kind kind;
    yvex_backend_kind backend_kind;
} yvex_backend_report_request;
typedef struct {
    yvex_backend_report_kind kind;
    yvex_backend_kind backend_kind;
    yvex_backend_status backend_status;
    int exit_code, available, has_device_info;
    yvex_backend_device_info device_info;
    char device_name[128], reason[256];
    yvex_backend_memory_stats memory;
    int capabilities[YVEX_BACKEND_CAP_OP_ATTENTION + 1];
    yvex_backend_capability_result variants[YVEX_BACKEND_VARIANT_COUNT];
    unsigned int variant_count, context_available;
    yvex_backend_bundle_admission bundle_admission;
    yvex_backend_capability_reason bundle_reason;
    yvex_backend_bandwidth_evidence bandwidth;
} yvex_backend_report;
typedef struct {
    const char *kind, *status, *reason, *next_row;
} yvex_backend_report_fact;
int yvex_backend_report_build(const yvex_backend_report_request *request,
                              yvex_backend_report *report, yvex_error *err);
const char *yvex_backend_bundle_admission_name(yvex_backend_bundle_admission admission);
#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_BACKEND_H_INCLUDED */
