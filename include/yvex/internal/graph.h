/*
 * Expose the minimum graph contract required by runtime, backend, and operator adapters.
 *
 * Cross-owner consumers use explicit immutable contracts and never graph-private implementation
 * state. Graph execution is not persistent runtime KV, transformer composition, or generation.
 */
#ifndef INCLUDE_YVEX_INTERNAL_GRAPH_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_GRAPH_H_INCLUDED

#include <stddef.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/model.h>
#include <yvex/registry.h>

#define YVEX_ATTENTION_NO_LAYER (~0ull)
#define YVEX_ATTENTION_NO_TENSOR_INDEX (~0ull)

typedef enum {
    YVEX_GRAPH_REPORT_MODE_NORMAL = 0,
    YVEX_GRAPH_REPORT_MODE_TABLE,
    YVEX_GRAPH_REPORT_MODE_AUDIT,
    YVEX_GRAPH_REPORT_MODE_JSON,
    YVEX_GRAPH_REPORT_MODE_CSV
} yvex_graph_report_mode;

typedef struct {
    unsigned long long ordinal, layer_index, predictor_index;
    yvex_tensor_scope tensor_scope;
    yvex_attention_class attention_class;
    yvex_attention_compute_contract compute_contract;
    unsigned long long compression_ratio, sliding_window, query_heads, kv_heads;
    unsigned long long head_dimension, rope_head_dimension, query_lora_rank;
    unsigned long long output_lora_rank, output_groups, output_group_input_width;
    unsigned long long hidden_dimension;
    unsigned long long indexer_heads, indexer_head_dimension, indexer_topk;
    unsigned long long compressor_ape_columns, indexer_ape_columns;
    double rms_norm_epsilon;
    unsigned long long residual_stream_count, residual_stream_width, residual_expanded_width;
    unsigned long long mhc_mixing_rows, mhc_mixing_columns, mhc_base_width, mhc_scale_width;
    unsigned long long mhc_sinkhorn_iterations, attention_input_norm_width;
    double mhc_epsilon, mhc_residual_post_multiplier;
    unsigned int mhc_entry_policy;
    int mhc_attention_pre_and_post, attention_input_norm_required;
    yvex_tensor_role attention_input_norm_role, mhc_function_role, mhc_base_role, mhc_scale_role;
    int compressor_required, indexer_required;
    yvex_attention_position_policy position;
    yvex_attention_activation_policy attention_kv_activation;
    yvex_attention_activation_policy compressor_activation;
    yvex_attention_activation_policy compressor_rotated_activation;
    yvex_attention_activation_policy indexer_query_activation;
    yvex_attention_topk_policy sparse_topk;
    unsigned long long required_binding_count, qtype_compute_refusal_count, payload_bytes_bound;
} yvex_attention_layer_plan;

typedef int (*yvex_attention_recipe_identity_fn)(const void *context, char output[65]);
typedef int (*yvex_attention_recipe_layer_fn)(const void *context, unsigned long long index,
                                              yvex_attention_layer_plan *output);

typedef struct {
    const void *context;
    unsigned long long layer_count, auxiliary_layer_count;
    unsigned long long swa_layer_count, csa_layer_count, hca_layer_count;
    yvex_tensor_scope tensor_scope;
    yvex_attention_recipe_identity_fn identity;
    yvex_attention_recipe_layer_fn layer;
} yvex_attention_recipe;

/* Attention execution and publication are an internal cross-owner ABI. */
#define YVEX_ATTENTION_IDENTITY_CAP 65u
#define YVEX_ATTENTION_ROLLING_STATE_SCHEMA_V1 1u
/* Legacy spellings remain until the in-flight CUDA owner lands its atomic rename. */
#define YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP YVEX_ATTENTION_IDENTITY_CAP
#define YVEX_DEEPSEEK_ATTENTION_ROLLING_STATE_SCHEMA_V1 \
    YVEX_ATTENTION_ROLLING_STATE_SCHEMA_V1

typedef enum {
    YVEX_ATTENTION_STATUS_REFUSED = 0,
    YVEX_ATTENTION_STATUS_PLANNED,
    YVEX_ATTENTION_STATUS_EXECUTION_UNSUPPORTED,
    YVEX_ATTENTION_STATUS_EXECUTION_READY,
    YVEX_DEEPSEEK_ATTENTION_STATUS_REFUSED = YVEX_ATTENTION_STATUS_REFUSED,
    YVEX_DEEPSEEK_ATTENTION_STATUS_PLANNED = YVEX_ATTENTION_STATUS_PLANNED,
    YVEX_DEEPSEEK_ATTENTION_STATUS_EXECUTION_UNSUPPORTED =
        YVEX_ATTENTION_STATUS_EXECUTION_UNSUPPORTED,
    YVEX_DEEPSEEK_ATTENTION_STATUS_EXECUTION_READY = YVEX_ATTENTION_STATUS_EXECUTION_READY
} yvex_attention_status;

typedef enum {
    YVEX_ATTENTION_FAILURE_NONE = 0,
    YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT,
    YVEX_ATTENTION_FAILURE_ARCHITECTURE,
    YVEX_ATTENTION_FAILURE_MATERIALIZATION,
    YVEX_ATTENTION_FAILURE_DESCRIPTOR,
    YVEX_ATTENTION_FAILURE_MISSING_BINDING,
    YVEX_ATTENTION_FAILURE_QTYPE,
    YVEX_ATTENTION_FAILURE_DIMENSION,
    YVEX_ATTENTION_FAILURE_HISTORY,
    YVEX_ATTENTION_FAILURE_EXECUTION_UNSUPPORTED,
    YVEX_ATTENTION_FAILURE_READ,
    YVEX_ATTENTION_FAILURE_NUMERIC,
    YVEX_ATTENTION_FAILURE_STATE_DELTA,
    YVEX_ATTENTION_FAILURE_ALLOCATION,
    YVEX_ATTENTION_FAILURE_SCRATCH,
    YVEX_ATTENTION_FAILURE_CANCELLED,
    YVEX_ATTENTION_FAILURE_CLEANUP,
    YVEX_ATTENTION_FAILURE_BACKEND,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_NONE = YVEX_ATTENTION_FAILURE_NONE,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT = YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_ARCHITECTURE = YVEX_ATTENTION_FAILURE_ARCHITECTURE,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_MATERIALIZATION = YVEX_ATTENTION_FAILURE_MATERIALIZATION,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_DESCRIPTOR = YVEX_ATTENTION_FAILURE_DESCRIPTOR,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_MISSING_BINDING = YVEX_ATTENTION_FAILURE_MISSING_BINDING,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_QTYPE = YVEX_ATTENTION_FAILURE_QTYPE,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION = YVEX_ATTENTION_FAILURE_DIMENSION,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY = YVEX_ATTENTION_FAILURE_HISTORY,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_EXECUTION_UNSUPPORTED =
        YVEX_ATTENTION_FAILURE_EXECUTION_UNSUPPORTED,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_READ = YVEX_ATTENTION_FAILURE_READ,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC = YVEX_ATTENTION_FAILURE_NUMERIC,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA = YVEX_ATTENTION_FAILURE_STATE_DELTA,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION = YVEX_ATTENTION_FAILURE_ALLOCATION,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH = YVEX_ATTENTION_FAILURE_SCRATCH,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_CANCELLED = YVEX_ATTENTION_FAILURE_CANCELLED,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_CLEANUP = YVEX_ATTENTION_FAILURE_CLEANUP,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND = YVEX_ATTENTION_FAILURE_BACKEND
} yvex_attention_failure_code;

typedef enum {
    YVEX_ATTENTION_ROLLING_NONE = 0,
    YVEX_ATTENTION_ROLLING_MAIN,
    YVEX_ATTENTION_ROLLING_INDEXER,
    YVEX_DEEPSEEK_ATTENTION_ROLLING_NONE = YVEX_ATTENTION_ROLLING_NONE,
    YVEX_DEEPSEEK_ATTENTION_ROLLING_MAIN = YVEX_ATTENTION_ROLLING_MAIN,
    YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER = YVEX_ATTENTION_ROLLING_INDEXER
} yvex_attention_rolling_kind;

typedef struct {
    yvex_attention_failure_code code;
    unsigned long long layer_index;
    yvex_tensor_role role;
    unsigned long long expected, actual;
    char tensor_name[YVEX_MATERIALIZATION_NAME_CAP];
    const char *reason;
} yvex_attention_failure;

typedef struct {
    int present;
    unsigned int schema_version;
    yvex_attention_rolling_kind kind;
    yvex_attention_class attention_class;
    unsigned long long layer_index, next_token_position, ratio, head_dimension;
    unsigned long long state_width, state_slots, previous_fill, current_fill, cursor;
    unsigned long long kv_state_stride, score_state_stride, kv_state_extent, score_state_extent;
    const float *kv_state, *score_state;
    int overlap, rotated;
    char attention_plan_identity[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP];
} yvex_attention_rolling_state_view;

typedef struct {
    int present;
    unsigned int schema_version;
    yvex_attention_rolling_kind kind;
    yvex_attention_class attention_class;
    unsigned long long layer_index, next_token_position, ratio, head_dimension;
    unsigned long long state_width, state_slots, previous_fill, current_fill, cursor;
    unsigned long long kv_state_stride, score_state_stride, kv_state_extent, score_state_extent;
    float *kv_state, *score_state;
    int overlap, rotated;
    char attention_plan_identity[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP];
} yvex_attention_rolling_state_output;

typedef struct {
    unsigned long long token_count, local_tail_count, compressed_entry_count, indexer_entry_count;
    const float *local_kv;
    unsigned long long local_kv_stride;
    const unsigned long long *local_positions;
    const float *compressed_kv;
    unsigned long long compressed_kv_stride;
    const unsigned long long *compressed_positions;
    const float *indexer_kv;
    unsigned long long indexer_kv_stride;
    const unsigned long long *indexer_positions;
    yvex_attention_rolling_state_view main_rolling_state;
    yvex_attention_rolling_state_view indexer_rolling_state;
    int immutable;
} yvex_attention_history_view;

typedef enum {
    YVEX_ATTENTION_EXECUTION_EAGER = 0,
    YVEX_ATTENTION_EXECUTION_PIECEWISE,
    YVEX_ATTENTION_EXECUTION_FULL
} yvex_attention_execution_mode;
typedef struct yvex_attention_state_recipe yvex_attention_state_recipe;
typedef struct yvex_attention_state_recipe_request yvex_attention_state_recipe_request;
typedef struct yvex_attention_workspace_recipe yvex_attention_workspace_recipe;
int yvex_attention_state_recipe_seal(yvex_attention_state_recipe *recipe,
                                     yvex_error *err);

typedef struct yvex_attention_publication {
    int owned, complete;
    struct yvex_attention_workspace *workspace;
    unsigned int evidence_level;
    unsigned long long layer_index;
    yvex_attention_class attention_class;
    unsigned long long token_position, token_count, hidden_width, q_rank;
    unsigned long long core_output_width, envelope_output_width;
    unsigned long long query_width, kv_width, compressed_count, compressed_stride;
    unsigned long long indexer_count, indexer_stride, index_query_stride;
    unsigned long long index_weight_stride, topk_stride;
    float *input, *q_low, *query, *raw_kv, *compressed_kv;
    float *indexer_kv, *index_query, *index_weights, *attention_values, *output;
    float *core_output, *envelope_output;
    unsigned long long *compressed_positions, *indexer_positions;
    unsigned long long *topk_counts, *topk_positions;
    yvex_attention_rolling_state_output next_main_rolling_state;
    yvex_attention_rolling_state_output next_indexer_rolling_state;
} yvex_attention_publication;

/* Historical tests use the same owned object as an execution trace. */
typedef yvex_attention_publication yvex_attention_execution_trace;

typedef struct {
    yvex_attention_status status;
    yvex_tensor_scope tensor_scope;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char materialization_plan_identity[YVEX_MATERIALIZATION_IDENTITY_CAP];
    char logical_model_identity[YVEX_RUNTIME_DESCRIPTOR_IDENTITY_CAP];
    char runtime_descriptor_identity[YVEX_RUNTIME_DESCRIPTOR_IDENTITY_CAP];
    char runtime_numeric_identity[YVEX_RUNTIME_DESCRIPTOR_IDENTITY_CAP];
    char attention_plan_identity[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP];
    unsigned long long layer_count, auxiliary_layer_count;
    unsigned long long swa_layer_count, csa_layer_count, hca_layer_count;
    unsigned long long required_binding_count, required_envelope_binding_count;
    unsigned long long missing_binding_count;
    unsigned long long qtype_compute_refusal_count, payload_bytes_bound;
    unsigned long long qtype_binding_counts[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    int history_contract_ready, state_delta_contract_ready, cpu_reference_ready;
    int cuda_execution_ready, full_execution_ready;
} yvex_attention_summary;

typedef struct yvex_attention_plan yvex_attention_plan;
typedef enum {
    YVEX_ATTENTION_BINDING_NOT_REQUIRED = 0,
    YVEX_ATTENTION_BINDING_CORE,
    YVEX_ATTENTION_BINDING_ENVELOPE
} yvex_attention_binding_class;
const yvex_attention_summary *yvex_attention_plan_summary(const yvex_attention_plan *plan);
unsigned long long yvex_attention_plan_layer_count(const yvex_attention_plan *plan);
void yvex_attention_plan_close(yvex_attention_plan *plan);
const yvex_attention_layer_plan *
yvex_attention_plan_layer_at(const yvex_attention_plan *plan, unsigned long long index);
yvex_attention_binding_class yvex_attention_plan_binding_classify(
    const yvex_attention_plan *plan, const yvex_runtime_tensor_binding *binding);
int yvex_attention_plan_identity_compute(const yvex_attention_summary *summary,
                                         const yvex_attention_layer_plan *layers,
                                         unsigned long long layer_count,
                                         char output[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP]);
typedef enum {
    YVEX_ATTENTION_OPERATION_CORE = 0,
    YVEX_ATTENTION_OPERATION_ENVELOPE,
    YVEX_ATTENTION_OPERATION_RELEASE_SET
} yvex_attention_operation_scope;
typedef enum {
    YVEX_ATTENTION_EVIDENCE_NONE = 0,
    YVEX_ATTENTION_EVIDENCE_SUMMARY,
    YVEX_ATTENTION_EVIDENCE_STAGES,
    YVEX_ATTENTION_EVIDENCE_FULL
} yvex_attention_evidence_level;
typedef struct yvex_attention_workspace yvex_attention_workspace;
typedef struct {
    unsigned long long capacity_bytes, used_bytes, peak_bytes;
    unsigned long long acquisition_count, rewind_count, capacity_failure_count;
    unsigned long long generation;
    int open, active;
} yvex_attention_workspace_summary;
int yvex_attention_workspace_open(yvex_attention_workspace **out,
                                  unsigned long long capacity_bytes,
                                  yvex_error *err);
int yvex_attention_workspace_begin(yvex_attention_workspace *workspace,
                                   yvex_error *err);
unsigned long long yvex_attention_workspace_mark(
    const yvex_attention_workspace *workspace);
int yvex_attention_workspace_rewind(yvex_attention_workspace *workspace,
                                    unsigned long long mark, yvex_error *err);
int yvex_attention_workspace_finish(yvex_attention_workspace *workspace,
                                    yvex_error *err);
void *yvex_attention_workspace_calloc(yvex_attention_workspace *workspace,
                                      unsigned long long count,
                                      unsigned long long width);
const yvex_attention_workspace_summary *yvex_attention_workspace_summary_get(
    const yvex_attention_workspace *workspace);
void yvex_attention_workspace_close(yvex_attention_workspace **workspace);
typedef int (*yvex_attention_cancel_predicate)(void *context);
typedef struct {
    yvex_attention_cancel_predicate requested;
    void *context;
} yvex_attention_cancellation;

typedef struct {
    yvex_attention_operation_scope operation_scope;
    const char *logical_model_identity;
    unsigned long long layer_index, token_position, token_count;
    unsigned long long local_history_tokens, compressed_history_tokens;
    unsigned long long max_q_b_rows, max_kv_rows, max_compressor_rows, max_indexer_rows;
    unsigned long long scratch_limit_bytes;
    yvex_attention_evidence_level evidence_level;
    yvex_attention_workspace *workspace;
    const float *input;
    unsigned long long input_stride;
    const yvex_device_tensor *device_input; yvex_device_tensor *device_output;
    const yvex_attention_history_view *history;
    yvex_attention_publication *publication;
    yvex_attention_execution_trace *trace;
    const yvex_attention_cancellation *cancellation;
    int candidate_block_visible;
} yvex_attention_cpu_options;

typedef struct {
    int executed, full_attention, cuda_executed;
    yvex_attention_operation_scope operation_scope;
    unsigned long long layer_index;
    yvex_attention_class attention_class;
    unsigned long long token_position, q_a_rows, q_b_rows, kv_rows;
    unsigned long long compressor_rows, indexer_rows, topk_candidates, topk_selected;
    unsigned long long local_entries, compressed_entries, payload_bytes_read;
    unsigned long long state_raw_entries, state_compressed_entries, state_indexer_entries;
    unsigned long long cuda_kernel_launches, cuda_peak_host_bytes, cuda_peak_device_bytes;
    unsigned long long cuda_h2d_bytes, cuda_d2h_bytes;
    unsigned long long cuda_device_execution_elapsed_ns;
    unsigned long long cuda_host_workspace_capacity, cuda_host_workspace_used;
    unsigned long long cuda_host_workspace_peak, cuda_host_workspace_allocations;
    int cuda_host_workspace_reused;
    double q_projection_checksum, kv_projection_checksum, rope_checksum;
    double attention_checksum, core_output_checksum, envelope_output_checksum, output_checksum;
    char output_identity[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP];
} yvex_attention_cpu_result;

void yvex_attention_rolling_output_bind(
    const yvex_attention_rolling_state_output *output,
    yvex_attention_rolling_state_view *view);

typedef struct yvex_graph_family_api {
    int (*plan_build)(yvex_attention_plan **out, const void *family_ir,
                      const yvex_materialization_session *session,
                      const yvex_runtime_descriptor *descriptor, yvex_attention_failure *failure,
                      yvex_error *err);
    int (*draft_plan_build)(yvex_attention_plan **out, const void *family_ir,
                            const yvex_materialization_session *session,
                            const yvex_runtime_descriptor *descriptor,
                            yvex_attention_failure *failure, yvex_error *err);
    void (*plan_close)(yvex_attention_plan *plan);
    const yvex_attention_summary *(*plan_summary)(const yvex_attention_plan *plan);
    unsigned long long (*plan_layer_count)(const yvex_attention_plan *plan);
    const yvex_attention_layer_plan *(*plan_layer_at)(const yvex_attention_plan *plan,
                                                      unsigned long long index);
    int (*selection_key_resolve)(const char *operator_token,
                                 unsigned long long *selection_key,
                                 yvex_error *err);
    int (*state_recipe)(const yvex_attention_layer_plan *layer,
                        const yvex_attention_state_recipe_request *request,
                        yvex_attention_state_recipe *recipe,
                        yvex_attention_failure *failure, yvex_error *err);
    int (*workspace_recipe)(const yvex_attention_layer_plan *layer,
                            const yvex_attention_state_recipe *state_recipe,
                            yvex_attention_execution_mode mode,
                            yvex_attention_operation_scope scope,
                            yvex_attention_evidence_level evidence_level,
                            unsigned long long token_capacity,
                            yvex_attention_workspace_recipe *recipe,
                            yvex_attention_failure *failure, yvex_error *err);
    int (*history_validate)(const yvex_attention_layer_plan *layer,
                            const yvex_attention_history_view *history,
                            yvex_attention_failure *failure, yvex_error *err);
    int (*rolling_state_step_cpu)(const yvex_attention_layer_plan *layer,
                                  const yvex_attention_rolling_state_view *before,
                                  const float *token_kv, const float *token_score,
                                  const float *ape_row, yvex_attention_rolling_state_output *after,
                                  float *compressed_out, unsigned long long compressed_out_count,
                                  int *emitted, yvex_attention_failure *failure, yvex_error *err);
    void (*cpu_options_default)(yvex_attention_cpu_options *options);
    void (*publication_release)(yvex_attention_publication *publication);
    void (*execution_trace_release)(yvex_attention_execution_trace *trace);
    int (*cuda_token_execute)(const yvex_attention_plan *plan, const void *family_ir,
                              yvex_materialization_session *session,
                              const yvex_runtime_descriptor *descriptor, yvex_backend *backend,
                              const yvex_attention_cpu_options *options,
                              yvex_attention_cpu_result *result, yvex_attention_failure *failure,
                              yvex_error *err);
    int (*cpu_chunk_execute)(const yvex_attention_plan *plan, const void *family_ir,
                             yvex_materialization_session *session,
                             const yvex_runtime_descriptor *descriptor,
                             const yvex_attention_cpu_options *options,
                             yvex_attention_cpu_result *result, yvex_attention_failure *failure,
                             yvex_error *err);
} yvex_graph_family_api;

/* Resolve the reusable graph numerical arena independently of backend staging. */
int yvex_attention_workspace_capacity_resolve(
    const yvex_graph_family_api *family, const yvex_attention_plan *plan,
    unsigned long long *arena_bytes, yvex_error *err);

struct yvex_model_family_api;
struct yvex_compilation_runtime_binding_request;
struct yvex_compilation_runtime_binding_result;
typedef struct {
    const char *target_id, *source_manifest_filename;
    const struct yvex_model_family_api *(*model)(void);
    int (*admit_artifact)(const yvex_artifact *artifact, yvex_complete_artifact_admission *out,
                          yvex_artifact_admission_failure *failure, yvex_error *err);
    int (*prepare_runtime_binding)(
        const struct yvex_compilation_runtime_binding_request *request,
        struct yvex_compilation_runtime_binding_result *result, yvex_error *err);
} yvex_graph_family_preparation;
const yvex_graph_family_preparation *
yvex_graph_family_preparation_at(unsigned long long index);

int yvex_attention_plan_import(yvex_attention_plan **out,
                               const yvex_attention_summary *summary,
                               const yvex_attention_layer_plan *layers,
                               unsigned long long layer_count,
                               const yvex_materialization_session *session,
                               const yvex_runtime_descriptor *descriptor,
                               yvex_attention_failure *failure, yvex_error *err);

typedef enum {
    YVEX_ATTENTION_PROBE_UNSPECIFIED = 0,
    YVEX_ATTENTION_PROBE_CANONICAL_V2 = 2
} yvex_attention_probe_kind;
typedef enum {
    YVEX_ATTENTION_PROBE_SCOPE_QUICK = 0,
    YVEX_ATTENTION_PROBE_SCOPE_FULL
} yvex_attention_probe_scope;
typedef struct {
    void *context;
    int (*begin)(void *context, unsigned long long layer_ordinal,
                 const yvex_attention_layer_plan *layer,
                 const yvex_attention_history_view *initial_history,
                 unsigned long long token_position, unsigned long long token_count,
                 const yvex_attention_cancellation *cancellation,
                 const yvex_attention_history_view **history,
                 yvex_attention_failure *failure, yvex_error *err);
    int (*stage)(void *context, const yvex_attention_publication *publication,
                 const yvex_attention_cancellation *cancellation,
                 char state_delta_identity[YVEX_SHA256_HEX_CAP],
                 yvex_attention_failure *failure, yvex_error *err);
    int (*abort)(void *context, yvex_attention_failure *failure, yvex_error *err);
} yvex_attention_probe_state_provider;
typedef int (*yvex_attention_probe_evidence_fn)(
    void *context, yvex_backend_kind backend,
    const yvex_attention_publication *publication, yvex_error *err);
typedef int (*yvex_attention_activation_view_fn)(
    void *context, unsigned long long layer_ordinal,
    unsigned long long token_count, const float **input,
    unsigned long long *input_stride, yvex_error *err);
typedef int (*yvex_attention_device_view_fn)(
    void *context, unsigned long long layer_ordinal, unsigned long long token_count,
    const yvex_device_tensor **input, yvex_device_tensor **output, yvex_error *err);
typedef enum {
    YVEX_ATTENTION_TRANSACTION_COMMIT = 0,
    YVEX_ATTENTION_TRANSACTION_ABORT,
    YVEX_ATTENTION_TRANSACTION_STAGE
} yvex_attention_transaction_disposition;
typedef struct {
    yvex_backend_kind backend;
    yvex_tensor_scope tensor_scope;
    yvex_backend *backend_context;
    const char *logical_model_identity;
    yvex_attention_probe_kind probe;
    yvex_attention_probe_scope scope;
    yvex_attention_operation_scope operation_scope;
    unsigned long long token_count, layer_ordinal, token_position;
    int select_layer, select_position, perturb_input;
    int compare_backends;
    int (*cancel_requested)(void *context);
    void *cancel_context;
    const char *input_identity;
    yvex_attention_activation_view_fn activation_view;
    yvex_attention_device_view_fn device_view;
    void *activation_context;
    const yvex_attention_probe_state_provider *state_provider;
    yvex_attention_workspace *workspace;
    yvex_attention_evidence_level evidence_level;
    yvex_attention_probe_evidence_fn evidence;
    void *evidence_context;
    yvex_attention_transaction_disposition transaction_disposition;
    int candidate_block_visible;
} yvex_attention_probe_request;
typedef yvex_attention_probe_request yvex_attention_execution_request;
typedef struct {
    unsigned long long value_count;
    unsigned long long finite_value_count;
    unsigned long long nonfinite_value_count;
    unsigned long long first_failing_coordinate;
    double maximum_absolute_error;
    double maximum_relative_error;
    double squared_error_sum;
    int within_tolerance;
    int bitwise_equal;
} yvex_graph_f32_comparison;
int yvex_graph_f32_compare(const float *left, const float *right,
                           unsigned long long count, double absolute_tolerance,
                           double relative_tolerance, yvex_graph_f32_comparison *out,
                           yvex_error *err);
typedef enum {
    YVEX_ATTENTION_COMPARISON_STAGE_NONE = 0,
    YVEX_ATTENTION_COMPARISON_STAGE_OUTPUT,
    YVEX_ATTENTION_COMPARISON_STAGE_PUBLICATION,
    YVEX_ATTENTION_COMPARISON_STAGE_RAW_KV,
    YVEX_ATTENTION_COMPARISON_STAGE_COMPRESSED_GEOMETRY,
    YVEX_ATTENTION_COMPARISON_STAGE_COMPRESSED_KV,
    YVEX_ATTENTION_COMPARISON_STAGE_COMPRESSED_POSITIONS,
    YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_EMISSION_GEOMETRY,
    YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_EMISSION_KV,
    YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_EMISSION_POSITIONS,
    YVEX_ATTENTION_COMPARISON_STAGE_MAIN_GEOMETRY,
    YVEX_ATTENTION_COMPARISON_STAGE_MAIN_KV,
    YVEX_ATTENTION_COMPARISON_STAGE_MAIN_SCORE,
    YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_ROLLING_GEOMETRY,
    YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_ROLLING_KV,
    YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_ROLLING_SCORE
} yvex_attention_comparison_stage;
typedef struct {
    yvex_graph_f32_comparison numeric;
    yvex_attention_comparison_stage first_failing_stage;
    int geometry_equal;
} yvex_attention_state_comparison;
int yvex_attention_state_compare(const yvex_attention_publication *left,
                                 const yvex_attention_publication *right,
                                 double absolute_tolerance, double relative_tolerance,
                                 yvex_attention_state_comparison *out, yvex_error *err);
int yvex_attention_publication_hash_update(
    yvex_sha256 *output_hash, yvex_sha256 *state_hash,
    const yvex_attention_publication *publication);
typedef struct {
    char attention_execution_identity[YVEX_SHA256_HEX_CAP];
    char tensor_output_digest[YVEX_SHA256_HEX_CAP];
    char cpu_output_digest[YVEX_SHA256_HEX_CAP], cuda_output_digest[YVEX_SHA256_HEX_CAP];
    char state_delta_digest[YVEX_SHA256_HEX_CAP];
    char cpu_state_delta_digest[YVEX_SHA256_HEX_CAP];
    char cuda_state_delta_digest[YVEX_SHA256_HEX_CAP];
    char comparison_contract_identity[YVEX_SHA256_HEX_CAP];
    char cuda_device[128];
    unsigned long long layers_executed, bindings_executed;
    unsigned long long swa_layers_executed, csa_layers_executed, hca_layers_executed;
    unsigned long long topk_selected, hca_ratio, payload_bytes_read;
    unsigned long long kernel_launches, peak_device_bytes, comparison_values;
    unsigned long long h2d_bytes, d2h_bytes, cuda_device_execution_elapsed_ns;
    unsigned long long comparison_output_values, comparison_state_values;
    unsigned long long comparison_finite_values, comparison_nonfinite_values;
    unsigned long long first_failing_layer, first_failing_coordinate;
    yvex_attention_comparison_stage first_failing_stage;
    int cuda_compute_capability_major, cuda_compute_capability_minor;
    double comparison_maximum_absolute_error, comparison_maximum_relative_error;
    double comparison_rmse;
    int comparison_available, comparison_passed, output_digest_equal, state_delta_digest_equal;
    int bitwise_equality_observed, bitwise_equality_required;
} yvex_attention_probe_result;
int yvex_attention_publication_compare(
    const yvex_attention_publication *left,
    const yvex_attention_publication *right,
    double absolute_tolerance, double relative_tolerance,
    yvex_attention_probe_result *result, double *squared_error,
    yvex_error *err);

typedef struct yvex_attention_probe_history yvex_attention_probe_history;
int yvex_attention_probe_position_resolve(const yvex_attention_layer_plan *layer, int class_selected,
    unsigned long long offset, unsigned long long *position, yvex_error *err);
int yvex_attention_probe_history_open(yvex_attention_probe_history **out,
    const yvex_attention_layer_plan *layer, const yvex_attention_summary *summary, unsigned long long position,
    const yvex_attention_history_view **view, yvex_error *err);
void yvex_attention_probe_history_close(yvex_attention_probe_history **history);

int yvex_attention_probe_execute(const yvex_graph_family_api *family,
                                 const yvex_attention_plan *plan, const void *family_ir,
                                 yvex_materialization_session *session,
                                 const yvex_runtime_descriptor *descriptor,
                                 const yvex_attention_probe_request *request,
                                 yvex_attention_probe_result *result,
                                 yvex_attention_failure *failure, yvex_error *err);
int yvex_attention_execute(
    const yvex_graph_family_api *family, const yvex_attention_plan *plan,
    const void *family_ir, yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_execution_request *request,
    yvex_attention_probe_result *result,
    yvex_attention_failure *failure, yvex_error *err);

#endif
