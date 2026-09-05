/* Execute exact causal attention over session-owned transactional K/V history. */
#ifndef INCLUDE_YVEX_INTERNAL_STATEFUL_ATTENTION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_STATEFUL_ATTENTION_H_INCLUDED

#include <yvex/internal/runtime.h>
#include <yvex/internal/transformer.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_STATEFUL_ATTENTION_RESULT_SCHEMA_V1 1u

typedef struct {
    yvex_device_tensor values, positions;
    unsigned long long value_width, visible_tokens, admitted_tokens;
} yvex_runtime_state_history_device_view;

int yvex_runtime_state_residency_candidate_history(
    yvex_runtime_state_residency *, unsigned long long,
    yvex_attention_state_binding, yvex_runtime_state_history_device_view *,
    yvex_error *);

typedef struct {
    yvex_backend *backend;
    const yvex_attention_state_provider *state;
    yvex_runtime_state_residency *residency;
    const yvex_attention_layer_plan *layer;
    const char *attention_plan_identity, *input_identity;
    unsigned long long layer_ordinal, token_position, token_count;
    unsigned long long query_token_stride;
    const yvex_device_tensor *query, *key, *value;
    yvex_device_tensor *output;
    float *host_workspace;
    unsigned long long host_workspace_values;
    unsigned long long *host_positions;
    unsigned long long host_position_capacity;
    yvex_attention_cancellation cancellation;
} yvex_runtime_stateful_attention_request;

typedef struct {
    unsigned int schema_version;
    int completed;
    unsigned long long layer_ordinal, token_position, token_count;
    unsigned long long history_tokens, state_staged_bytes;
    unsigned long long h2d_bytes, d2h_bytes, d2d_bytes;
    yvex_backend_operation_facts attention;
    char state_delta_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_stateful_attention_result;

int yvex_runtime_stateful_attention_execute(
    const yvex_runtime_stateful_attention_request *,
    yvex_runtime_stateful_attention_result *, yvex_attention_failure *,
    yvex_error *);

#ifdef __cplusplus
}
#endif
#endif
