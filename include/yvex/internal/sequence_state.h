/* Session-owned transactional state for backend-neutral recurrent sequence mixers. */
#ifndef INCLUDE_YVEX_INTERNAL_SEQUENCE_STATE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SEQUENCE_STATE_H_INCLUDED

#include <yvex/internal/graph_state.h>
#include <yvex/internal/sequence_mixer.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_SEQUENCE_STATE_SCHEMA_V1 1u

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
    unsigned int schema_version;
    unsigned long long binding_count, committed_position, candidate_tokens;
    unsigned long long generation, staged_layers;
    unsigned long long convolution_state_bytes, recurrent_state_bytes;
    unsigned long long committed_state_bytes, candidate_state_bytes;
    char plan_identity[YVEX_SHA256_HEX_CAP];
    int transaction_active, prepared, invalidated;
} yvex_sequence_state_summary;

typedef struct yvex_sequence_state yvex_sequence_state;

int yvex_sequence_state_open(
    yvex_sequence_state **out, const yvex_sequence_state_plan *plan,
    yvex_error *err);
int yvex_sequence_state_fork(
    yvex_sequence_state **out, const yvex_sequence_state *source,
    yvex_error *err);
int yvex_sequence_state_begin(
    yvex_sequence_state *state, unsigned long long token_start,
    unsigned long long token_count, yvex_error *err);
int yvex_sequence_state_layer(
    yvex_sequence_state *state, unsigned long long layer_index,
    yvex_gated_delta_state_view *committed,
    yvex_gated_delta_state_output *candidate, yvex_error *err);
int yvex_sequence_state_stage(
    yvex_sequence_state *state, unsigned long long layer_index,
    yvex_error *err);
int yvex_sequence_state_committed(
    const yvex_sequence_state *state, unsigned long long layer_index,
    yvex_gated_delta_state_view *committed, yvex_error *err);
int yvex_sequence_state_participant(
    yvex_sequence_state *state,
    yvex_runtime_transaction_participant *participant, yvex_error *err);
int yvex_sequence_state_reset(yvex_sequence_state *state, yvex_error *err);
int yvex_sequence_state_invalidate(
    yvex_sequence_state *state, yvex_error *err);
int yvex_sequence_state_summary_copy(
    const yvex_sequence_state *state, yvex_sequence_state_summary *summary,
    yvex_error *err);
void yvex_sequence_state_close(yvex_sequence_state **state);

#ifdef __cplusplus
}
#endif
#endif
