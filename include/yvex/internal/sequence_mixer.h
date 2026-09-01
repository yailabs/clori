/*
 * Stateful sequence mixers expose semantic geometry and transactional state without backend
 * representation details. Families own projections and parameters; backends execute this
 * admitted recurrence over caller-owned candidate state.
 */
#ifndef INCLUDE_YVEX_INTERNAL_SEQUENCE_MIXER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SEQUENCE_MIXER_H_INCLUDED

#include <yvex/core.h>
#include <yvex/model.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_SEQUENCE_MIXER_IDENTITY_CAP 65u
#define YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V1 1u

typedef enum {
    YVEX_SEQUENCE_MIXER_NUMERIC_UNKNOWN = 0,
    YVEX_SEQUENCE_MIXER_NUMERIC_F32_RECURRENCE
} yvex_sequence_mixer_numeric_contract;

typedef struct {
    unsigned int schema_version;
    unsigned long long query_heads, key_heads, value_heads;
    unsigned long long key_head_dimension, value_head_dimension;
    unsigned long long convolution_kernel;
    yvex_dtype projected_dtype, convolution_state_dtype;
    yvex_dtype recurrent_state_dtype, accumulation_dtype, output_dtype;
    yvex_sequence_mixer_numeric_contract numeric_contract;
    double qk_normalization_epsilon, output_normalization_epsilon;
    double query_scale;
    int deterministic;
} yvex_gated_delta_requirement;

typedef struct {
    unsigned int schema_version;
    yvex_gated_delta_requirement requirement;
    unsigned long long query_width, key_width, value_width, qkv_width;
    unsigned long long convolution_state_values, recurrent_state_values;
    unsigned long long convolution_state_bytes, recurrent_state_bytes;
    char identity[YVEX_SEQUENCE_MIXER_IDENTITY_CAP];
} yvex_gated_delta_plan;

typedef struct {
    const float *convolution;
    const float *recurrent;
} yvex_gated_delta_state_view;

typedef struct {
    float *convolution;
    unsigned long long convolution_capacity;
    float *recurrent;
    unsigned long long recurrent_capacity;
} yvex_gated_delta_state_output;

typedef struct {
    unsigned long long token_count;
    const float *projected_qkv;
    unsigned long long projected_qkv_capacity;
    const float *projected_output_gate;
    unsigned long long projected_output_gate_capacity;
    const float *projected_beta;
    unsigned long long projected_beta_capacity;
    const float *projected_decay;
    unsigned long long projected_decay_capacity;
    const float *convolution_weight;
    unsigned long long convolution_weight_capacity;
    const float *decay_log;
    unsigned long long decay_log_capacity;
    const float *time_bias;
    unsigned long long time_bias_capacity;
    const float *normalization_weight;
    unsigned long long normalization_weight_capacity;
    yvex_gated_delta_state_view state;
    yvex_gated_delta_state_output next_state;
    float *output;
    unsigned long long output_capacity;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_gated_delta_cpu_request;

typedef struct {
    unsigned long long token_count, output_values;
    unsigned long long convolution_state_values, recurrent_state_values;
    unsigned long long recurrent_matrix_updates, accumulated_values;
    int complete, cancelled;
} yvex_gated_delta_cpu_result;

int yvex_gated_delta_plan_seal(
    yvex_gated_delta_plan *plan, const yvex_gated_delta_requirement *requirement,
    yvex_error *err);
int yvex_gated_delta_plan_validate(
    const yvex_gated_delta_plan *plan, yvex_error *err);
int yvex_gated_delta_execute_cpu(
    const yvex_gated_delta_plan *plan, const yvex_gated_delta_cpu_request *request,
    yvex_gated_delta_cpu_result *result, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
