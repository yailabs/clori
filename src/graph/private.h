/*
 * Share the minimum typed contracts required by independent graph translation units.
 *
 * Declarations are graph-local and preserve immutable identity and transaction boundaries. This
 * header is never installed or included outside graph production and focused tests.
 */
#ifndef YVEX_GRAPH_PRIVATE_H
#define YVEX_GRAPH_PRIVATE_H
#include <stddef.h>
#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/graph.h>
#include <yvex/model.h>
#include <yvex/registry.h>
#include <yvex/tokenizer.h>
#include <yvex/internal/core.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/compiler.h>
#include <yvex/internal/convolution.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/quant_numeric.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    unsigned int input_ids[4], output_ids[4];
} yvex_graph_op_edges;
typedef struct {
    float *data;
    unsigned long long count;
} yvex_graph_component_buffer;
typedef struct {
    unsigned long long maximum_bytes, live_bytes, peak_bytes;
} yvex_graph_component_workspace;
typedef struct {
    yvex_materialization_session *session;
    const yvex_component_execution_request *request;
    yvex_component_execution_result *result;
    yvex_component_failure *failure;
    yvex_error *err;
    yvex_graph_component_workspace workspace;
    const char *failure_where;
} yvex_graph_component_execution;
typedef struct {
    void (*execution_open)(yvex_graph_component_execution *, yvex_materialization_session *,
                           const yvex_component_execution_request *,
                           yvex_component_execution_result *, yvex_component_failure *,
                           yvex_error *, const char *);
    int (*execution_refuse)(yvex_graph_component_execution *, yvex_component_failure_code,
                            const char *, unsigned long long, unsigned long long, yvex_status,
                            const char *);
    int (*cancel_check)(yvex_graph_component_execution *, const char *);
    int (*buffer_open)(yvex_graph_component_execution *, unsigned long long,
                       yvex_graph_component_buffer *);
    void (*buffer_close)(yvex_graph_component_execution *, yvex_graph_component_buffer *);
    const yvex_materialized_tensor_binding *(*binding_find)(
        const yvex_materialization_session *, const char *);
    int (*tensor_load_f32)(yvex_graph_component_execution *, const char *, unsigned int,
                           const unsigned long long *, yvex_graph_component_buffer *);
    int (*name_build)(yvex_graph_component_execution *, char *, size_t, const char *,
                      const char *, unsigned long long, const char *);
    int (*rectified_flow_step)(float *, const float *, const float *, unsigned long long,
                               float, float, float, yvex_error *);
} yvex_graph_component_api;
const yvex_graph_component_api *yvex_graph_component_api_get(void);
const yvex_physical_variant_api *yvex_graph_physical_variant_api_get(void);
/* Stable virtual spans commit physical state pages only as graph publication reaches them. */
typedef struct yvex_graph_state_page_pool yvex_graph_state_page_pool;
typedef struct yvex_graph_state_page_store yvex_graph_state_page_store;
typedef struct {
    yvex_attention_state_component_recipe recipe;
    float *values;
    unsigned long long *positions;
    float *auxiliary;
    unsigned long long start, allocated_rows;
    yvex_graph_state_page_store *value_pages;
    yvex_graph_state_page_store *position_pages;
    yvex_graph_state_page_store *auxiliary_pages;
} yvex_graph_state_component_storage;
typedef struct {
    unsigned long long allocated_bytes, metadata_bytes, virtual_bytes;
    unsigned long long resident_bytes, page_count, resident_page_count;
    unsigned long long page_commit_count, page_release_count;
} yvex_graph_state_page_summary;
typedef struct {
    const float *values;
    const unsigned long long *positions;
    unsigned long long count, width;
} yvex_graph_state_history_span;
int yvex_graph_state_history_project(
    const yvex_attention_history_view *view,
    const yvex_attention_state_component_recipe *component,
    yvex_graph_state_history_span *out);
const yvex_attention_rolling_state_view *yvex_graph_state_rolling_view(
    const yvex_attention_history_view *view,
    yvex_attention_state_binding binding);
int yvex_graph_state_page_pool_open(
    yvex_graph_state_page_pool **out, unsigned long long maximum_bytes,
    yvex_error *err);
int yvex_graph_state_page_pool_bind_capacity(
    yvex_graph_state_page_pool *pool,
    const yvex_execution_capacity_plan *capacity, yvex_error *err);
void yvex_graph_state_page_pool_release(
    yvex_graph_state_page_pool *pool, unsigned long long bytes);
int yvex_graph_state_page_pool_summary(
    const yvex_graph_state_page_pool *pool,
    yvex_graph_state_page_summary *out, yvex_error *err);
void yvex_graph_state_page_pool_close(yvex_graph_state_page_pool **pool);
int yvex_graph_state_capacity_plan_valid(
    const yvex_execution_capacity_plan *capacity);
int yvex_graph_state_bank_pages_open(
    yvex_graph_state_page_pool *pool,
    const yvex_execution_capacity_plan *capacity,
    const yvex_attention_summary *summary,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_recipe *recipe,
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *view, yvex_error *err);
int yvex_graph_state_bank_pages_reset(
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe, yvex_error *err);
void yvex_graph_state_bank_pages_bind(
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe);
int yvex_graph_state_bank_pages_transfer(
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe,
    const yvex_attention_history_view *source,
    int validate_storage, yvex_error *err);
void yvex_graph_state_bank_pages_close(
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT]);
int yvex_graph_state_pages_prepare_publications(
    yvex_attention_history_view *view,
    yvex_graph_state_component_storage
        components[YVEX_ATTENTION_STATE_BINDING_COUNT],
    const yvex_attention_state_recipe *recipe,
    const yvex_attention_publication *const *publications,
    unsigned long long publication_count, yvex_error *err);
int yvex_graph_state_pointer_table_reserve(
    yvex_graph_state_page_pool *pool, void ***table,
    unsigned long long *capacity, unsigned long long *bytes,
    unsigned long long limit, yvex_error *err);
int yvex_graph_state_initial_identity(
    const yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe, const char *plan_identity,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_graph_state_advance_identity(
    const char *prior_identity, const yvex_attention_state_recipe *recipe,
    const char *plan_identity, const yvex_attention_publication *publication,
    char output[YVEX_SHA256_HEX_CAP]);
struct yvex_graph {
    yvex_graph_status status;
    char *architecture, *model_name;
    unsigned long long sequence_length, context_length;
    yvex_graph_value_info *values;
    unsigned long long value_count, value_cap;
    yvex_graph_op_info *ops;
    yvex_graph_op_edges *edges;
    unsigned long long op_count, op_cap;
    yvex_graph_missing_required *missing;
    unsigned long long missing_count, missing_cap;
};
struct yvex_memory_plan {
    yvex_memory_plan_status status;
    yvex_memory_plan_summary summary;
};
struct yvex_plan {
    char *backend_name, *backend_status;
    int backend_tensor_alloc, backend_tensor_read_write, backend_op_embed, backend_op_matmul, backend_op_mlp,
        backend_op_rms_norm, backend_op_rope, backend_op_attention;
    yvex_graph *graph;
    yvex_memory_plan *memory;
};

typedef enum {
    YVEX_ATTENTION_COMPONENT_ATTENTION_OUTPUT = 0, YVEX_ATTENTION_COMPONENT_RAW_LOCAL_KV,
    YVEX_ATTENTION_COMPONENT_COMPRESSED_MAIN_KV, YVEX_ATTENTION_COMPONENT_INDEXER_KV,
    YVEX_ATTENTION_COMPONENT_MAIN_KV_STATE, YVEX_ATTENTION_COMPONENT_MAIN_SCORE_STATE,
    YVEX_ATTENTION_COMPONENT_INDEXER_KV_STATE, YVEX_ATTENTION_COMPONENT_INDEXER_SCORE_STATE,
    YVEX_ATTENTION_COMPONENT_ENVELOPE_OUTPUT,
    YVEX_ATTENTION_COMPONENT_COUNT
} yvex_attention_component_kind;
typedef enum {
    YVEX_ATTENTION_COMPONENT_STORAGE_F32 = 1
} yvex_attention_component_storage;
typedef enum {
    YVEX_ATTENTION_TRANSACTION_EMPTY = 0, YVEX_ATTENTION_TRANSACTION_BEGUN,
    YVEX_ATTENTION_TRANSACTION_ABORTED, YVEX_ATTENTION_TRANSACTION_COMMITTED
} yvex_attention_transaction_status;
typedef struct {
    yvex_attention_component_kind kind;
    yvex_attention_component_storage storage;
    unsigned int rank;
    unsigned long long dims[4], stride, expected_elements, produced_elements, byte_extent, position_start,
        position_count;
    void *data;
    yvex_attention_workspace *workspace;
    int required, acquired, written, sealed;
} yvex_attention_component_span;
typedef struct {
    yvex_attention_component_kind fail_acquire_kind;
    yvex_attention_component_kind fail_seal_kind;
    yvex_attention_workspace *workspace;
    int fail_begin, fail_commit, fail_abort;
} yvex_attention_memory_sink_options;
typedef struct {
    int initialized, has_committed;
    yvex_attention_workspace *workspace;
    yvex_attention_memory_sink_options options;
    yvex_attention_component_span committed[
        YVEX_ATTENTION_COMPONENT_COUNT];
    char committed_identity[YVEX_ATTENTION_IDENTITY_CAP];
} yvex_attention_memory_sink;
typedef struct {
    yvex_attention_transaction_status status;
    yvex_attention_memory_sink *sink;
    unsigned long long layer_index;
    yvex_attention_class attention_class;
    unsigned long long token_position, token_count;
    char previous_state_identity[YVEX_ATTENTION_IDENTITY_CAP],
        transaction_identity[YVEX_ATTENTION_IDENTITY_CAP];
    yvex_attention_component_span components[
        YVEX_ATTENTION_COMPONENT_COUNT];
} yvex_attention_state_transaction;
typedef struct {
    unsigned long long limit_bytes;
    size_t live_bytes, peak_bytes;
    yvex_attention_workspace *workspace;
} yvex_attention_scratch_budget;
typedef struct {
    unsigned char *owned[YVEX_BACKEND_ATTENTION_WEIGHT_COUNT];
    unsigned long long payload_bytes_read;
} attention_cuda_weights;
typedef struct {
    const yvex_attention_plan *plan;
    const void *family_ir;
    yvex_materialization_session *session;
    const yvex_runtime_descriptor *descriptor;
    yvex_backend *backend;
    yvex_attention_cpu_options defaults;
    const yvex_attention_cpu_options *opts;
    yvex_attention_cpu_result *result;
    yvex_attention_failure *failure;
    yvex_error *err;
    const yvex_attention_layer_plan *layer;
    yvex_attention_history_view empty_history;
    const yvex_attention_history_view *history;
    attention_cuda_weights weights;
    yvex_backend_cancellation cancellation;
    yvex_backend_attention_job job;
    yvex_backend_attention_output cuda_output;
    yvex_backend_attention_failure cuda_failure;
    yvex_attention_execution_trace trace;
    unsigned long long trace_bytes, token_count, compressed_capacity, indexer_capacity;
    unsigned int i, role_mask;
    int rc;
} attention_cuda_context;
int yvex_attention_cuda_reject(
    attention_cuda_context *context, yvex_attention_failure_code code,
    unsigned long long expected, unsigned long long actual,
    yvex_status status, const char *reason);
int yvex_attention_cancel_check(const yvex_attention_cancellation *cancellation,
    unsigned long long layer_index, const char *safe_point, yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_class_geometry_validate(
    const yvex_attention_layer_plan *layer,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_history_validate(const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *history, yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_rolling_state_step_cpu(const yvex_attention_layer_plan *layer,
    const yvex_attention_rolling_state_view *before, const float *token_kv, const float *token_score,
    const float *ape_row, yvex_attention_rolling_state_output *after, float *compressed_out,
    unsigned long long compressed_out_count, int *emitted, yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_memory_sink_init(yvex_attention_memory_sink *sink,
    const yvex_attention_memory_sink_options *options, yvex_attention_failure *failure, yvex_error *err);
void yvex_attention_memory_sink_release(yvex_attention_memory_sink *sink);
int yvex_attention_state_transaction_begin_scope(yvex_attention_memory_sink *sink,
    const yvex_attention_layer_plan *layer, const yvex_attention_history_view *history,
    yvex_attention_operation_scope scope, unsigned long long token_position,
    unsigned long long token_count, yvex_attention_state_transaction *transaction,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_transaction_scratch_elements(const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *history, yvex_attention_operation_scope scope,
    unsigned long long token_position, unsigned long long token_count,
    unsigned long long *elements);
int yvex_attention_state_transaction_acquire(yvex_attention_state_transaction *transaction,
    yvex_attention_component_kind kind, yvex_attention_component_span *out, yvex_attention_failure *failure,
    yvex_error *err);
int yvex_attention_state_transaction_seal(yvex_attention_state_transaction *transaction,
    yvex_attention_component_kind kind, unsigned long long produced_elements, yvex_attention_failure *failure,
    yvex_error *err);
int yvex_attention_state_transaction_commit(yvex_attention_state_transaction *transaction,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_state_transaction_abort(yvex_attention_state_transaction *transaction,
    yvex_attention_failure *failure, yvex_error *err);
const yvex_attention_component_span *
yvex_attention_memory_sink_committed_component(const yvex_attention_memory_sink *sink,
    yvex_attention_component_kind kind);
const char *yvex_attention_memory_sink_identity(const yvex_attention_memory_sink *sink);

/* Attention internals share this single graph-private contract. */
struct yvex_attention_plan {
    yvex_attention_layer_plan *layers;
    unsigned long long layer_count;
    yvex_attention_summary summary;
};
const yvex_attention_summary *yvex_attention_plan_summary(const yvex_attention_plan *plan);
unsigned long long yvex_attention_plan_layer_count(const yvex_attention_plan *plan);
const yvex_attention_layer_plan *yvex_attention_plan_layer_at(const yvex_attention_plan *plan,
    unsigned long long index);
const yvex_attention_layer_plan *yvex_attention_plan_layer_find(
    const yvex_attention_plan *plan, unsigned long long layer_index);
int yvex_attention_reject(yvex_attention_failure *failure, yvex_attention_failure_code code,
    const yvex_runtime_tensor_binding *binding, unsigned long long layer_index, yvex_tensor_role role,
    unsigned long long expected, unsigned long long actual, yvex_error *err, yvex_status err_code,
    const char *reason);
int yvex_attention_accept(yvex_attention_failure *failure, yvex_error *err);

#define attention_hash_u64 yvex_sha256_update_u64_be

/* Append one nullable text field to a graph identity stream. */
static inline int attention_hash_text(yvex_sha256 *hash, const char *text)
{
    return yvex_sha256_update(hash, text ? text : "", text ? strlen(text) : 0u);
}

/* Select the lesser of two admitted unsigned extents. */
static inline unsigned long long attention_min_u64(unsigned long long left,
                                                   unsigned long long right)
{
    return left < right ? left : right;
}

/* Release one previously admitted live CPU scratch extent. */
static inline void attention_scratch_release(yvex_attention_scratch_budget *budget,
                                             size_t bytes)
{
    if (budget && bytes <= budget->live_bytes)
        budget->live_bytes -= bytes;
}

/* Clear one caller-owned attention result before transactional execution. */
static inline void attention_result_reset(yvex_attention_cpu_result *result)
{
    if (result)
        memset(result, 0, sizeof(*result));
}

int yvex_attention_checked_size(unsigned long long count, unsigned long long width, size_t *out);
void *yvex_attention_calloc_array(unsigned long long count, unsigned long long width);
void *yvex_attention_scratch_calloc(yvex_attention_scratch_budget *budget,
    unsigned long long count, unsigned long long width);
void yvex_attention_scratch_free(yvex_attention_scratch_budget *budget, void *allocation);
int yvex_attention_scratch_reserve(yvex_attention_scratch_budget *budget,
    unsigned long long count, size_t element_size, size_t *bytes_out);
int yvex_attention_context_validate(const yvex_attention_plan *plan, const char *logical_identity,
    const yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_execution_admit(
    const yvex_attention_plan *plan, const char *logical_identity,
    yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    const yvex_attention_cpu_options *options, const char *cancel_stage,
    const yvex_attention_layer_plan **layer, yvex_attention_failure *failure,
    yvex_error *err);
const yvex_runtime_tensor_binding *yvex_attention_binding_find(const yvex_runtime_descriptor *descriptor,
    yvex_tensor_role role, const yvex_attention_layer_plan *layer);
int yvex_attention_decode_row(yvex_materialization_session *session, const yvex_runtime_tensor_binding *binding,
    unsigned long long row, float *out, unsigned long long elements, yvex_attention_scratch_budget *scratch,
    yvex_attention_cpu_result *result, yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_dot_batch(yvex_materialization_session *session, const yvex_runtime_tensor_binding *binding,
    unsigned long long start_row, const float *vectors, unsigned long long token_count,
    unsigned long long vector_stride, unsigned long long vector_len, unsigned long long max_rows, float *out,
    unsigned long long output_stride, unsigned long long *rows, yvex_attention_scratch_budget *scratch,
    yvex_attention_cpu_result *result,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_decode_flat(yvex_materialization_session *session, const yvex_runtime_tensor_binding *binding,
    float *out, unsigned long long elements, yvex_attention_scratch_budget *scratch,
    yvex_attention_cpu_result *result, yvex_attention_failure *failure, yvex_error *err);
double yvex_attention_checksum(const float *values, unsigned long long count);
int yvex_attention_rms_norm(float *values, unsigned long long count, const float *weights, double epsilon);
int yvex_attention_unit_rms_norm(float *values, unsigned long long count, double epsilon);
typedef struct {
    const yvex_attention_layer_plan *layer;
    const float *residual, *linear_mixes, *scale, *base;
    unsigned long long token_count, residual_stride, mix_stride;
    float *collapsed, *post, *combination;
    unsigned long long collapsed_stride, post_stride, combination_stride;
} yvex_attention_mhc_pre_args;
typedef struct {
    const yvex_attention_layer_plan *layer;
    const float *core_output, *residual, *post, *combination;
    unsigned long long token_count, core_stride, residual_stride;
    unsigned long long post_stride, combination_stride;
    float *envelope_output;
    unsigned long long envelope_stride;
} yvex_attention_mhc_post_args;
typedef struct {
    float *residual, *linear_mixes, *scale, *base, *post, *combination, *norm_weights;
    unsigned long long residual_elements, residual_stride;
    unsigned long long mix_elements, post_elements, combination_elements;
    yvex_attention_workspace *workspace;
} yvex_attention_envelope_workspace;
int yvex_attention_mhc_pre(const yvex_attention_mhc_pre_args *args,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_mhc_post(const yvex_attention_mhc_post_args *args,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_envelope_prepare(yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor, const yvex_attention_layer_plan *layer,
    const float *expanded_input, unsigned long long token_count,
    unsigned long long input_stride, float *core_input,
    unsigned long long core_stride, yvex_attention_envelope_workspace *workspace,
    yvex_attention_scratch_budget *scratch, yvex_attention_cpu_result *result,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_envelope_finish(const yvex_attention_layer_plan *layer,
    const float *core_output, unsigned long long core_stride,
    unsigned long long token_count, const yvex_attention_envelope_workspace *workspace,
    float *envelope_output, unsigned long long envelope_stride,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_envelope_scratch_elements(const yvex_attention_layer_plan *layer,
    unsigned long long token_count, unsigned long long *elements);
void yvex_attention_envelope_workspace_release(yvex_attention_envelope_workspace *workspace);
int yvex_attention_compute_round(yvex_attention_compute_contract contract,
    float *values, unsigned long long count);
int yvex_attention_rope_apply(float *values, unsigned long long width, unsigned long long rope_width,
    unsigned long long position, const yvex_attention_position_policy *position_spec, int inverse);
int yvex_attention_activation_apply(const yvex_attention_activation_policy *policy, float *values,
    unsigned long long count, unsigned long long layer_index, yvex_tensor_role role,
    yvex_attention_scratch_budget *scratch, yvex_attention_failure *failure, yvex_error *err);
/* Project one checked row across the immutable history/current segment boundary. */
static inline const float *attention_segment_row(
    const float *history, unsigned long long history_count,
    unsigned long long history_stride, const float *current,
    unsigned long long current_count, unsigned long long current_stride,
    unsigned long long index)
{
    if (index < history_count)
        return history + index * history_stride;
    index -= history_count;
    return index < current_count ? current + index * current_stride : NULL;
}

/* Project one checked position across the immutable history/current boundary. */
static inline unsigned long long attention_segment_position(
    const unsigned long long *history, unsigned long long history_count,
    const unsigned long long *current, unsigned long long current_count,
    unsigned long long index)
{
    if (index < history_count)
        return history[index];
    index -= history_count;
    return index < current_count ? current[index] : ULLONG_MAX;
}
int yvex_attention_csa_select(const yvex_attention_layer_plan *layer, const yvex_attention_history_view *history,
    const float *current_indexer, unsigned long long current_count, unsigned long long current_stride,
    const unsigned long long *current_positions, const float *query, const float *weights,
    unsigned long long absolute, unsigned long long *selected, unsigned long long *selected_count,
    unsigned long long *valid_count, yvex_attention_scratch_budget *scratch,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_reduce_chunk(const yvex_attention_layer_plan *layer,
    const float *query, const yvex_attention_history_view *history, const float *current_kv,
    unsigned long long current_kv_stride, const float *current_compressed,
    unsigned long long current_compressed_count, unsigned long long current_compressed_stride,
    const unsigned long long *current_compressed_positions, const float *current_indexer,
    unsigned long long current_indexer_count, unsigned long long current_indexer_stride,
    const unsigned long long *current_indexer_positions, const float *index_query,
    unsigned long long index_query_stride, const float *index_weights, unsigned long long index_weight_stride,
    const float *sinks, unsigned long long token_count, unsigned long long token_position, float *out,
    int candidate_block_visible,
    unsigned long long *trace_topk_counts, unsigned long long *trace_topk_positions,
    unsigned long long trace_topk_stride, yvex_attention_scratch_budget *scratch,
    yvex_attention_cpu_result *result, yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_output_project(yvex_materialization_session *session, const yvex_runtime_tensor_binding *out_a,
    const yvex_runtime_tensor_binding *out_b, const float *values, unsigned long long token_count,
    unsigned long long value_stride, unsigned long long groups, unsigned long long group_width,
    unsigned long long rank, unsigned long long hidden_width,
    yvex_attention_compute_contract compute_contract, float *out,
    unsigned long long output_stride,
    yvex_attention_scratch_budget *scratch, yvex_attention_cpu_result *result,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_rolling_geometry(const yvex_attention_layer_plan *layer, yvex_attention_rolling_kind kind,
    unsigned long long *ratio, unsigned long long *head_dimension, unsigned long long *state_width,
    unsigned long long *state_slots, int *overlap, int *rotated);
void yvex_attention_execution_trace_release(yvex_attention_execution_trace *trace);
int yvex_attention_trace_capture(yvex_attention_execution_trace *trace, unsigned long long layer_index,
    yvex_attention_class attention_class, unsigned long long token_position,
    unsigned long long token_count, unsigned long long hidden_width, unsigned long long q_rank,
    unsigned long long query_width, unsigned long long kv_width, const float *input, const float *q_low,
    const float *query, const float *raw_kv, const float *compressed_kv, unsigned long long compressed_count,
    unsigned long long compressed_stride, const unsigned long long *compressed_positions, const float *indexer_kv,
    unsigned long long indexer_count, unsigned long long indexer_stride, const unsigned long long *indexer_positions,
    const float *index_query, unsigned long long index_query_stride, const float *index_weights,
    unsigned long long index_weight_stride, const float *attention_values, const float *output,
    const unsigned long long *topk_counts, const unsigned long long *topk_positions, unsigned long long topk_stride,
    const yvex_attention_rolling_state_output *main_state, const float *main_kv, const float *main_score,
    const yvex_attention_rolling_state_output *index_state, const float *index_kv,
    const float *index_score, unsigned long long rolling_checkpoint_count,
    const float *main_checkpoint_kv, const float *main_checkpoint_score,
    const float *index_checkpoint_kv, const float *index_checkpoint_score,
    yvex_attention_evidence_level evidence_level,
    yvex_attention_workspace *workspace);
int yvex_attention_candidate_checkpoints_open(
    yvex_attention_scratch_budget *scratch, unsigned long long rows,
    const yvex_attention_rolling_state_view *rolling, float **kv,
    float **score);
void yvex_attention_candidate_checkpoint_capture(
    float *kv, float *score, unsigned long long row,
    const yvex_attention_rolling_state_output *rolling);
int yvex_attention_rolling_storage_acquire(const yvex_attention_layer_plan *layer,
    yvex_attention_rolling_kind kind, unsigned long long token_position,
    yvex_attention_workspace *workspace, float **kv_state, float **score_state,
    yvex_attention_rolling_state_view *view, yvex_attention_failure *failure,
    yvex_error *err);
int yvex_attention_trace_outputs_attach(yvex_attention_execution_trace *trace,
    const yvex_attention_component_span *core,
    const yvex_attention_component_span *envelope);
void yvex_attention_result_outputs_publish(yvex_attention_cpu_result *result,
    yvex_attention_operation_scope scope, const yvex_attention_component_span *core,
    const yvex_attention_component_span *envelope);
void yvex_attention_cuda_weights_release(attention_cuda_weights *weights);
int yvex_attention_cuda_role_load(yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    const yvex_attention_layer_plan *layer, yvex_tensor_role role, yvex_backend_attention_weight_slot slot,
    attention_cuda_weights *owned, yvex_backend_attention_job *job, yvex_attention_failure *failure,
    yvex_error *err);
void yvex_attention_cuda_activation_project(const yvex_attention_activation_policy *source,
    yvex_backend_attention_activation *out);
int yvex_attention_cuda_rolling_project(const yvex_attention_rolling_state_view *source,
    yvex_backend_attention_rolling *out);
int yvex_attention_cuda_trace_open(yvex_attention_publication *trace,
    const yvex_attention_layer_plan *layer, yvex_attention_operation_scope scope,
    const yvex_attention_history_view *history, unsigned long long token_position,
    unsigned long long token_count, yvex_attention_evidence_level evidence_level,
    int retain_prefix_checkpoints, yvex_attention_workspace *workspace,
    unsigned long long limit_bytes,
    unsigned long long *owned_bytes, yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_cuda_publish(attention_cuda_context *context);
int yvex_attention_hadamard_cpu(const float *input, unsigned long long length, float scale, int reject_nonfinite,
    float *output, yvex_attention_scratch_budget *scratch, yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_topk_select(const float *scores, const unsigned long long *ordinals,
    unsigned long long candidate_count, unsigned long long k, unsigned long long *selected_indices,
    unsigned long long *selected_count, yvex_attention_scratch_budget *scratch,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_fp8_fake_quant_block(const float *input, unsigned long long count, float *dequantized,
    unsigned char *codes, unsigned char *scale_code, yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_fp4_fake_quant_block(const float *input, unsigned long long count, float *dequantized,
    unsigned char *codes, unsigned char *scale_code, yvex_attention_failure *failure, yvex_error *err);

int yvex_graph_linear_source_f32(
    const float *input, unsigned long long input_count, unsigned long long rows,
    unsigned long long input_width, const float *weight,
    unsigned long long weight_count, const float *bias,
    unsigned long long bias_count, unsigned long long output_width,
    float *output, unsigned long long output_count, yvex_error *err);
int yvex_graph_layer_norm_f32(float *values, unsigned long long rows,
                              unsigned long long width, const float *weight,
                              const float *bias, double epsilon, yvex_error *err);
int yvex_graph_silu_gate_f32(const float *fused, unsigned long long rows,
                             unsigned long long width, float *output,
                             yvex_error *err);
int yvex_graph_full_attention_f32(
    const float *qkv, unsigned long long rows, unsigned long long heads,
    unsigned long long head_width, float *output, float *scratch,
    unsigned long long scratch_count, yvex_error *err);
int yvex_graph_f32_execution_identity(
    const char *domain, const char *artifact_identity,
    const unsigned long long *geometry, unsigned long long geometry_count,
    const float *input, unsigned long long input_count,
    const float *output, unsigned long long output_count, char identity[65]);
int yvex_graph_interleaved_qk_norm_f32(
    float *qkv, unsigned long long rows, unsigned long long heads,
    unsigned long long head_width, double epsilon, yvex_error *err);
int yvex_graph_rope_3d_row_f32(
    unsigned long long token, unsigned long long frames,
    unsigned long long height, unsigned long long width,
    unsigned long long frequencies, float base,
    float *cosines, float *sines, yvex_error *err);
int yvex_graph_rope_3d_interleaved_qk_f32(
    float *qkv, unsigned long long rows, unsigned long long frames,
    unsigned long long height, unsigned long long width,
    unsigned long long heads, unsigned long long head_width,
    unsigned long long frequencies, float base, yvex_error *err);
int yvex_graph_scaled_residual_f32(
    float *hidden, const float *delta, const float *scale,
    unsigned long long rows, unsigned long long width, yvex_error *err);
#endif
