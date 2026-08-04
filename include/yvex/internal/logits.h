/*
 * Expose complete phase-neutral vocabulary projection to sampling and operators.
 *
 * One exact resident encoded head produces one complete F32 vocabulary row or nothing. Internal
 * runtime/operator ABI from normalized hidden state to raw vocabulary logits.
 */
#ifndef INCLUDE_YVEX_INTERNAL_LOGITS_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_LOGITS_H_INCLUDED

#include <yvex/internal/decode.h>
#include <yvex/internal/runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_LOGITS_SCHEMA_V3 3u
#define YVEX_RUNTIME_LOGITS_SCHEMA_V2 YVEX_RUNTIME_LOGITS_SCHEMA_V3
#define YVEX_RUNTIME_LOGITS_SCHEMA_V1 YVEX_RUNTIME_LOGITS_SCHEMA_V3
#define YVEX_OUTPUT_HEAD_BATCH_SCHEMA_V1 1u

typedef enum {
    YVEX_LOGITS_SOURCE_PREFILL = 0,
    YVEX_LOGITS_SOURCE_DECODE = 1,
    YVEX_LOGITS_SOURCE_DRAFT = 2
} yvex_logits_source_phase;

typedef struct yvex_logits_family_policy {
    unsigned int schema_version;
    int separate_output_head, tied_output_head, output_head_bias;
} yvex_logits_family_policy;

typedef struct {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long output_head_tensor_id, row_width, row_count, row_bytes;
    unsigned long long encoded_bytes, vocabulary_size, hidden_width;
    yvex_tensor_role role;
    unsigned int qtype;
    int separate_output_head, output_head_bias;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char materialization_identity[YVEX_SHA256_HEX_CAP];
    char logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    char output_head_plan_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_logits_plan_summary;

typedef struct yvex_runtime_logits_plan yvex_runtime_logits_plan;

typedef struct {
    unsigned int schema_version;
    int host_values_available, device_values_available;
    yvex_logits_source_phase source_phase;
    unsigned long long source_position, row_count, hidden_width;
    const float *normalized_hidden;
    yvex_execution_device_view device_hidden;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    char transformer_execution_identity[YVEX_SHA256_HEX_CAP];
    char normalized_hidden_digest[YVEX_SHA256_HEX_CAP];
    char source_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_logits_source;

typedef struct {
    unsigned int schema_version;
    int completed, host_values_available, device_values_available;
    int finite_count_available, range_available, raw_digest_available;
    int compulsory_memory_facts_available;
    yvex_execution_evidence_profile evidence_profile;
    yvex_logits_source_phase source_phase;
    unsigned long long source_position, vocabulary_size, hidden_width;
    unsigned long long logits_count, finite_count;
    float minimum_logit, maximum_logit;
    unsigned long long active_weight_bytes, state_bytes, activation_bytes, temporary_bytes;
    unsigned long long h2d_bytes, d2h_bytes, d2d_bytes, kernel_launches, device_synchronizations;
    unsigned long long full_array_host_scan_bytes;
    yvex_execution_device_view device_logits;
    char source_hidden_digest[YVEX_SHA256_HEX_CAP];
    char output_head_plan_identity[YVEX_SHA256_HEX_CAP];
    char output_head_residency_identity[YVEX_SHA256_HEX_CAP];
    char raw_logits_digest[YVEX_SHA256_HEX_CAP];
    char backend_execution_identity[YVEX_SHA256_HEX_CAP];
    char logits_row_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_logits_row_result;

typedef enum {
    YVEX_OUTPUT_HEAD_RESULT_HOST_LOGITS = 0,
    YVEX_OUTPUT_HEAD_RESULT_DEVICE_SELECTION
} yvex_output_head_result_class;

typedef enum {
    YVEX_OUTPUT_HEAD_SELECTION_RAW = 0,
    YVEX_OUTPUT_HEAD_SELECTION_GREEDY,
    YVEX_OUTPUT_HEAD_SELECTION_STOCHASTIC_REFERENCE
} yvex_output_head_selection_policy;

/*
 * Output width is an admitted graph shape. Rows remain ordered through one operation even while
 * the portable backend projects them independently.
 */
typedef struct {
    unsigned int schema_version;
    unsigned long long row_count, output_vocabulary;
    yvex_backend_kind backend;
    yvex_output_head_result_class result_class;
    yvex_output_head_selection_policy selection_policy;
    yvex_execution_evidence_profile evidence_profile;
    yvex_execution_class execution_class;
    const char *execution_profile_identity;
} yvex_output_head_batch_request;

typedef struct {
    unsigned int schema_version;
    int completed, partial;
    unsigned long long requested_rows, completed_rows, first_incomplete_row;
    unsigned long long final_source_position;
    char aggregate_logits_digest[YVEX_SHA256_HEX_CAP];
    char output_head_contract_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_logits_result;

typedef struct {
    unsigned long long maximum_rows, maximum_host_bytes, maximum_device_bytes;
    yvex_execution_evidence_profile evidence_profile;
    int device_greedy_selection;
    const yvex_compiled_execution_profile *execution_profile;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_runtime_logits_options;

typedef struct yvex_runtime_logits_context yvex_runtime_logits_context;

int yvex_runtime_logits_residency_admit(
    yvex_runtime_capabilities *capabilities,
    const yvex_runtime_residency_summary *residency);
void yvex_runtime_logits_capabilities_invalidate(yvex_runtime_capabilities *capabilities);
int yvex_runtime_logits_context_open(
    yvex_runtime_logits_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session,
    const yvex_transformer_plan *transformer_plan,
    const yvex_runtime_logits_options *options, yvex_error *err);
const yvex_runtime_logits_plan_summary *yvex_runtime_logits_plan_summary_get(
    const yvex_runtime_logits_context *context);
int yvex_runtime_logits_admit_shared_draft_plan(
    yvex_runtime_logits_context *context,
    const yvex_transformer_plan *draft_plan, yvex_error *err);
int yvex_runtime_logits_source_from_transformer(
    const yvex_runtime_logits_context *context,
    yvex_runtime_logits_source *source,
    const yvex_runtime_transformer_result *producer,
    const float *normalized_hidden, unsigned long long hidden_capacity,
    unsigned long long row_ordinal, yvex_error *err);
int yvex_runtime_logits_source_from_decode(
    const yvex_runtime_logits_context *context,
    yvex_runtime_logits_source *source,
    const yvex_runtime_decode_step_result *producer,
    const float *normalized_hidden, unsigned long long hidden_capacity,
    yvex_error *err);
int yvex_runtime_logits_source_from_draft(
    const yvex_runtime_logits_context *context,
    yvex_runtime_logits_source *source,
    const yvex_transformer_plan *draft_plan,
    const yvex_runtime_transformer_result *producer,
    const float *normalized_hidden, unsigned long long hidden_capacity,
    unsigned long long row_ordinal, yvex_error *err);
int yvex_runtime_logits_project(
    yvex_runtime_logits_context *context,
    const yvex_runtime_logits_source *source, yvex_backend_kind backend,
    float *logits, unsigned long long logits_capacity,
    yvex_runtime_logits_row_result *result, yvex_error *err);
int yvex_runtime_logits_additive_adjust(
    const yvex_runtime_logits_context *context, const float *base_logits,
    unsigned long long base_capacity,
    const yvex_runtime_logits_row_result *base_result,
    const float *additive_logits, unsigned long long additive_capacity,
    float *adjusted_logits, unsigned long long adjusted_capacity,
    yvex_runtime_logits_row_result *result, yvex_error *err);
int yvex_runtime_logits_execute(
    yvex_runtime_logits_context *context,
    const yvex_runtime_logits_source *sources, unsigned long long row_count,
    yvex_backend_kind backend, float *logits, unsigned long long logits_capacity,
    yvex_runtime_logits_row_result *rows, unsigned long long row_capacity,
    yvex_runtime_logits_result *result, yvex_error *err);
int yvex_runtime_logits_execute_rows(
    yvex_runtime_logits_context *context,
    const yvex_output_head_batch_request *request,
    const yvex_runtime_logits_source *sources,
    float *logits, unsigned long long logits_capacity,
    yvex_runtime_logits_row_result *rows, unsigned long long row_capacity,
    yvex_runtime_logits_result *result, yvex_error *err);
int yvex_runtime_logits_row_validate(
    const yvex_runtime_logits_plan_summary *plan, const float *logits,
    unsigned long long logits_capacity,
    const yvex_runtime_logits_row_result *result, yvex_error *err);
int yvex_runtime_logits_result_validate(
    const yvex_runtime_logits_plan_summary *plan, const float *logits,
    unsigned long long logits_capacity,
    const yvex_runtime_logits_row_result *rows,
    unsigned long long row_capacity,
    const yvex_runtime_logits_result *result, yvex_error *err);
int yvex_runtime_logits_context_close(yvex_runtime_logits_context **context,
                                      yvex_error *err);

typedef struct {
    const char *target, *artifact_path, *runtime_binding_path, *input_path;
    yvex_backend_kind backend;
    unsigned long long prefill_tokens, prefill_chunk_tokens, context_capacity;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
} yvex_logits_operator_request;

typedef struct {
    int completed;
    char status[32], command[64], target[128], family[32], backend[16], reason[256];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    yvex_runtime_logits_plan_summary plan;
    yvex_runtime_logits_result execution;
    yvex_runtime_logits_row_result *rows;
    float *raw_logits;
    unsigned long long raw_logits_count;
    unsigned long long row_count, prefill_logits_rows, decode_logits_rows;
    unsigned long long output_head_host_bytes, output_head_device_bytes;
    unsigned long long output_head_upload_bytes, output_head_upload_count;
    int output_head_binding_ready, output_head_residency_ready;
    int logits_cpu_ready, logits_cuda_ready, logits_prefill_ready, logits_decode_ready;
    int logits_full_vocabulary_ready, logits_hidden_contract_ready;
    int logits_partial_progress_ready, logits_ready;
    int sampling_ready, tokenizer_runtime_ready, generation_ready;
    int model_behavior_evaluation_ready, full_model_benchmark_ready;
    int release_qualification_ready;
} yvex_logits_operator_result;

int yvex_runtime_logits_operator_execute(
    const yvex_logits_operator_request *request,
    yvex_logits_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err);
void yvex_runtime_logits_operator_result_release(yvex_logits_operator_result *result);

#ifdef __cplusplus
}
#endif
#endif
