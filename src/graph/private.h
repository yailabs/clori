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
#include <yvex/internal/graph.h>
#include <yvex/internal/quant_numeric.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    unsigned int input_ids[4], output_ids[4];
} yvex_graph_op_edges;
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
    YVEX_DEEPSEEK_ATTENTION_COMPONENT_ATTENTION_OUTPUT = 0, YVEX_DEEPSEEK_ATTENTION_COMPONENT_RAW_LOCAL_KV,
    YVEX_DEEPSEEK_ATTENTION_COMPONENT_COMPRESSED_MAIN_KV, YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV,
    YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_KV_STATE, YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_SCORE_STATE,
    YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV_STATE, YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_SCORE_STATE,
    YVEX_DEEPSEEK_ATTENTION_COMPONENT_ENVELOPE_OUTPUT,
    YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT
} yvex_attention_component_kind;
typedef enum {
    YVEX_DEEPSEEK_ATTENTION_COMPONENT_STORAGE_F32 = 1
} yvex_attention_component_storage;
typedef enum {
    YVEX_DEEPSEEK_ATTENTION_TRANSACTION_EMPTY = 0, YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN,
    YVEX_DEEPSEEK_ATTENTION_TRANSACTION_ABORTED, YVEX_DEEPSEEK_ATTENTION_TRANSACTION_COMMITTED
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
        YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT];
    char committed_identity[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP];
} yvex_attention_memory_sink;
typedef struct {
    yvex_attention_transaction_status status;
    yvex_attention_memory_sink *sink;
    unsigned long long layer_index;
    yvex_attention_class attention_class;
    unsigned long long token_position, token_count;
    char previous_state_identity[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP],
        transaction_identity[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP];
    yvex_attention_component_span components[
        YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT];
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
int yvex_attention_class_geometry_validate(const yvex_attention_layer_plan *layer,
    unsigned long long csa_ratio, unsigned long long hca_ratio,
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
int yvex_attention_plan_build(yvex_attention_plan **out, const yvex_attention_recipe *recipe,
    const yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err);
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
    unsigned long long csa_ratio, unsigned long long hca_ratio,
    const yvex_attention_layer_plan **layer, yvex_attention_failure *failure,
    yvex_error *err);
int yvex_attention_state_recipe_build(
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_recipe_request *request,
    yvex_attention_state_recipe *recipe, yvex_attention_failure *failure,
    yvex_error *err);
int yvex_attention_workspace_recipe_build(
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_recipe *state, yvex_attention_execution_mode mode,
    yvex_attention_operation_scope scope, yvex_attention_evidence_level evidence_level,
    unsigned long long token_capacity,
    yvex_attention_workspace_recipe *recipe, yvex_attention_failure *failure,
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
    const float *index_score, yvex_attention_evidence_level evidence_level,
    yvex_attention_workspace *workspace);
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
    yvex_attention_workspace *workspace, unsigned long long limit_bytes,
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

#endif
