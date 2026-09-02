/*
 * Stateful sequence mixers expose semantic geometry and transactional state without backend
 * representation details. Families own projections and parameters; backends execute this
 * admitted recurrence over caller-owned candidate state.
 */
#ifndef INCLUDE_YVEX_INTERNAL_SEQUENCE_MIXER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SEQUENCE_MIXER_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/semantic_decoder.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_SEQUENCE_MIXER_IDENTITY_CAP 65u
#define YVEX_SEQUENCE_STATE_SCHEMA_V1 1u

typedef struct {
    unsigned int schema_version;
    yvex_gated_delta_requirement requirement;
    unsigned long long query_width, key_width, value_width, qkv_width;
    unsigned long long convolution_state_values, recurrent_state_values;
    unsigned long long convolution_state_bytes, recurrent_state_bytes;
    char identity[YVEX_SEQUENCE_MIXER_IDENTITY_CAP];
} yvex_gated_delta_plan;

typedef struct {
    unsigned long long layer_index;
    yvex_gated_delta_plan plan;
} yvex_sequence_state_binding;

typedef struct {
    unsigned int schema_version;
    const yvex_sequence_state_binding *bindings;
    unsigned long long binding_count;
} yvex_sequence_state_plan;

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

/* Device execution consumes family-owned projections and parameters while preserving the same
 * sealed recurrence and caller-owned candidate-state contract as the portable CPU authority.
 * A backend may mutate only next_state/output; committed state is always read-only. */
typedef struct {
    unsigned long long token_count;
    const yvex_device_tensor *projected_qkv;
    const yvex_device_tensor *projected_output_gate;
    const yvex_device_tensor *projected_beta;
    const yvex_device_tensor *projected_decay;
    const yvex_device_tensor *convolution_weight;
    const yvex_device_tensor *decay_log;
    const yvex_device_tensor *time_bias;
    const yvex_device_tensor *normalization_weight;
    const yvex_device_tensor *convolution_state;
    const yvex_device_tensor *recurrent_state;
    yvex_device_tensor *next_convolution_state;
    yvex_device_tensor *next_recurrent_state;
    yvex_device_tensor *output;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_gated_delta_device_request;

typedef struct {
    unsigned long long token_count, output_values;
    unsigned long long convolution_state_values, recurrent_state_values;
    unsigned long long recurrent_matrix_updates, accumulated_values;
    unsigned long long execution_chunks;
    int complete, cancelled;
} yvex_gated_delta_device_result;

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
