/*
 * Expose reusable greedy and seeded stochastic selection to tokenizer and generation consumers.
 *
 * One complete logits row yields one canonical token ID or no sampling/RNG publication.
 * Family-neutral internal runtime/operator ABI from admitted raw logits to one vocabulary token
 * ID.
 */
#ifndef INCLUDE_YVEX_INTERNAL_SAMPLING_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SAMPLING_H_INCLUDED

#include <yvex/internal/logits.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_SAMPLING_SCHEMA_V2 2u
#define YVEX_RUNTIME_SAMPLING_SCHEMA_V1 YVEX_RUNTIME_SAMPLING_SCHEMA_V2
#define YVEX_SAMPLING_FILTER_ORDER_V1 1u
#define YVEX_SAMPLING_FILTER_ORDER_V2 2u
#define YVEX_SAMPLING_RNG_PCG_XSH_RR_64_32 1u
#define YVEX_SAMPLING_RNG_VERSION_V1 1u
#define YVEX_SAMPLING_GREEDY_LOWEST_TOKEN_ID 1u

typedef enum {
    YVEX_SAMPLING_STRATEGY_GREEDY = 0,
    YVEX_SAMPLING_STRATEGY_STOCHASTIC = 1
} yvex_sampling_strategy;

typedef struct yvex_runtime_sampling_policy {
    unsigned int schema_version;
    yvex_sampling_strategy strategy;
    double temperature;
    unsigned long long top_k;
    double top_p, min_p, typical_p;
    int seed_present;
    unsigned long long seed;
    unsigned int rng_algorithm, rng_version, filter_order_version;
    char policy_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_sampling_policy;

typedef struct {
    unsigned int schema_version;
    int host_values_available, device_values_available;
    yvex_logits_source_phase source_phase;
    unsigned long long source_position, vocabulary_size, logits_capacity;
    const float *logits;
    yvex_execution_device_view device_logits;
    char raw_logits_digest[YVEX_SHA256_HEX_CAP];
    char logits_row_identity[YVEX_SHA256_HEX_CAP];
    char output_head_plan_identity[YVEX_SHA256_HEX_CAP];
    char source_hidden_digest[YVEX_SHA256_HEX_CAP];
    char backend_execution_identity[YVEX_SHA256_HEX_CAP];
    char source_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_sampling_source;

typedef struct {
    unsigned int schema_version;
    int completed, numeric_fallback_used, device_selection;
    yvex_sampling_strategy strategy;
    yvex_logits_source_phase source_phase;
    unsigned long long source_position, vocabulary_size, values_considered;
    unsigned long long candidates_before, candidates_after_top_k;
    unsigned long long candidates_after_min_p, candidates_after_typical_p;
    unsigned long long candidates_after_top_p, final_candidate_count;
    unsigned int selected_token_id;
    unsigned int greedy_tie_policy;
    float selected_logit, maximum_logit;
    double temperature, selected_probability, selected_log_probability;
    unsigned long long tied_maximum_count, effective_top_k, rng_draw_count;
    unsigned long long d2h_bytes, kernel_launches, device_synchronizations;
    unsigned long long full_array_host_scan_bytes;
    double effective_top_p, effective_min_p, effective_typical_p;
    double min_p_threshold, entropy, typical_retained_mass, top_p_retained_mass;
    double normalization_error;
    char rng_state_before_identity[YVEX_SHA256_HEX_CAP];
    char rng_state_after_identity[YVEX_SHA256_HEX_CAP];
    char policy_identity[YVEX_SHA256_HEX_CAP];
    char source_identity[YVEX_SHA256_HEX_CAP];
    char candidate_set_identity[YVEX_SHA256_HEX_CAP];
    char selected_token_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_sampling_result;

typedef struct {
    unsigned int schema_version;
    int completed, partial;
    unsigned long long requested_samples, completed_samples, first_incomplete_sample;
    char initial_rng_state_identity[YVEX_SHA256_HEX_CAP];
    char final_rng_state_identity[YVEX_SHA256_HEX_CAP];
    char ordered_selected_token_digest[YVEX_SHA256_HEX_CAP];
    char aggregate_sampling_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_sampling_execution;

typedef struct {
    unsigned long long maximum_vocabulary_size, maximum_rows, maximum_host_bytes;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_runtime_sampling_options;

typedef struct {
    unsigned int schema_version;
    unsigned long long vocabulary_size, maximum_rows, workspace_bytes;
    unsigned long long workspace_generation, cold_workspace_allocations;
    unsigned long long warm_workspace_allocations, successful_samples;
    unsigned long long stochastic_draws, failure_count, cancellation_count;
    char output_head_plan_identity[YVEX_SHA256_HEX_CAP];
    char policy_identity[YVEX_SHA256_HEX_CAP];
    char rng_state_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_sampling_context_summary;

typedef struct {
    unsigned int schema_version;
    int completed;
    yvex_sampling_strategy strategy;
    unsigned long long vocabulary_size, positive_probability_count;
    char policy_identity[YVEX_SHA256_HEX_CAP];
    char source_identity[YVEX_SHA256_HEX_CAP];
    char distribution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_sampling_distribution_result;

typedef struct {
    unsigned int schema_version;
    int completed;
    unsigned long long draw_count;
    char rng_state_before_identity[YVEX_SHA256_HEX_CAP];
    char rng_state_after_identity[YVEX_SHA256_HEX_CAP];
    char draw_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_sampling_uniform_result;

typedef struct yvex_runtime_sampling_context yvex_runtime_sampling_context;
typedef struct yvex_runtime_sampling_transaction yvex_runtime_sampling_transaction;

int yvex_runtime_sampling_policy_seal(
    yvex_runtime_sampling_policy *policy, unsigned long long vocabulary_size,
    yvex_error *err);
int yvex_runtime_sampling_context_open(
    yvex_runtime_sampling_context **out,
    const yvex_runtime_logits_plan_summary *logits_plan,
    const yvex_runtime_sampling_policy *policy,
    const yvex_runtime_sampling_options *options, yvex_error *err);
int yvex_runtime_sampling_source_from_logits(
    const yvex_runtime_sampling_context *context,
    yvex_runtime_sampling_source *source, const float *logits,
    unsigned long long logits_capacity,
    const yvex_runtime_logits_row_result *row, yvex_error *err);
int yvex_runtime_sampling_select(
    yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source,
    yvex_runtime_sampling_result *result, yvex_error *err);
int yvex_runtime_sampling_distribution(
    yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *source,
    float *probabilities, unsigned long long probability_capacity,
    yvex_runtime_sampling_distribution_result *result, yvex_error *err);
int yvex_runtime_sampling_transaction_begin(
    yvex_runtime_sampling_context *context,
    yvex_runtime_sampling_transaction **transaction, yvex_error *err);
int yvex_runtime_sampling_transaction_uniforms(
    yvex_runtime_sampling_transaction *transaction, double *values,
    unsigned long long value_count,
    yvex_runtime_sampling_uniform_result *result, yvex_error *err);
int yvex_runtime_sampling_transaction_prepare_commit(
    yvex_runtime_sampling_transaction *transaction, yvex_error *err);
void yvex_runtime_sampling_transaction_publish_commit(
    yvex_runtime_sampling_transaction **transaction);
int yvex_runtime_sampling_transaction_abort(
    yvex_runtime_sampling_transaction **transaction, yvex_error *err);
int yvex_runtime_sampling_execute(
    yvex_runtime_sampling_context *context,
    const yvex_runtime_sampling_source *sources, unsigned long long source_count,
    yvex_runtime_sampling_result *results, unsigned long long result_capacity,
    yvex_runtime_sampling_execution *execution, yvex_error *err);
int yvex_runtime_sampling_context_snapshot(
    const yvex_runtime_sampling_context *context,
    yvex_runtime_sampling_context_summary *summary, yvex_error *err);
int yvex_runtime_sampling_result_validate(
    const yvex_runtime_sampling_result *result, yvex_error *err);
int yvex_runtime_sampling_context_close(
    yvex_runtime_sampling_context **context, yvex_error *err);

typedef struct {
    yvex_logits_operator_request logits;
    yvex_runtime_sampling_policy policy;
    unsigned long long maximum_sampling_host_bytes;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_sampling_operator_request;

typedef struct {
    int completed;
    char status[32], command[64], target[128], family[32], logits_backend[16];
    char sampling_execution_kind[32], strategy[16], reason[256];
    yvex_logits_operator_result logits;
    yvex_runtime_sampling_policy policy;
    yvex_runtime_sampling_execution execution;
    yvex_runtime_sampling_result *samples;
    unsigned long long sample_count, prefill_samples, decode_samples;
    unsigned long long workspace_bytes, workspace_generation;
    unsigned long long cold_workspace_allocations, warm_workspace_allocations;
    int sampling_source_contract_ready, sampling_policy_ready;
    int sampling_greedy_ready, sampling_temperature_ready, sampling_top_k_ready;
    int sampling_top_p_ready, sampling_min_p_ready, sampling_typical_ready;
    int sampling_stochastic_ready, sampling_seed_reproducibility_ready;
    int sampling_real_logits_ready, sampling_partial_progress_ready, sampling_ready;
    int persistent_state_unchanged;
    int token_append_ready, tokenizer_runtime_ready, eos_policy_ready, stop_policy_ready;
    int detokenization_ready, generation_ready, cuda_sampling_ready;
    int model_behavior_evaluation_ready, full_model_benchmark_ready;
    int release_qualification_ready;
} yvex_sampling_operator_result;

int yvex_runtime_sampling_operator_execute(
    const yvex_sampling_operator_request *request,
    yvex_sampling_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err);
void yvex_runtime_sampling_operator_result_release(
    yvex_sampling_operator_result *result);

#ifdef __cplusplus
}
#endif
#endif
