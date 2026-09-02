/* Prove recurrent state participates in the runtime session commit boundary. */
#include "tests/test.h"

#include <pthread.h>
#include <string.h>

#include <yvex/internal/sequence_state.h>

#include "src/runtime/private.h"

static int session_state_plan(yvex_gated_delta_plan *plan, yvex_error *err)
{
    const yvex_gated_delta_requirement requirement = {
        .schema_version = YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V2,
        .output_normalization_weight_convention =
            YVEX_NORMALIZATION_WEIGHT_DIRECT,
        .query_heads = 1ull, .key_heads = 1ull, .value_heads = 1ull,
        .key_head_dimension = 2ull, .value_head_dimension = 2ull,
        .convolution_kernel = 2ull,
        .projected_dtype = YVEX_DTYPE_F32,
        .convolution_state_dtype = YVEX_DTYPE_F32,
        .recurrent_state_dtype = YVEX_DTYPE_F32,
        .accumulation_dtype = YVEX_DTYPE_F32,
        .output_dtype = YVEX_DTYPE_F32,
        .numeric_contract = YVEX_SEQUENCE_MIXER_NUMERIC_F32_RECURRENCE,
        .qk_normalization_epsilon = 1e-6,
        .output_normalization_epsilon = 1e-6,
        .query_scale = 0.7071067811865476,
        .deterministic = 1};

    return yvex_gated_delta_plan_seal(plan, &requirement, err);
}

static int session_state_stage(
    yvex_sequence_state *state, unsigned long long start, float value,
    yvex_error *err)
{
    yvex_gated_delta_state_view committed;
    yvex_gated_delta_state_output candidate;
    unsigned long long index;

    if (yvex_sequence_state_begin(state, start, 1ull, err) != YVEX_OK ||
        yvex_sequence_state_layer(
            state, 2ull, &committed, &candidate, err) != YVEX_OK)
        return 0;
    for (index = 0ull; index < candidate.convolution_capacity; ++index)
        candidate.convolution[index] = committed.convolution[index] + value;
    for (index = 0ull; index < candidate.recurrent_capacity; ++index)
        candidate.recurrent[index] = committed.recurrent[index] + value;
    return yvex_sequence_state_stage(state, 2ull, err) == YVEX_OK;
}

static void session_state_execution_begin(
    yvex_runtime_execution_session *session)
{
    session->summary.busy = 1;
    session->execution_owner = pthread_self();
    session->execution_owner_ready = 1;
}

int yvex_test_sequence_state_session(void)
{
    yvex_gated_delta_plan mixer;
    yvex_sequence_state_binding binding;
    yvex_sequence_state_plan plan;
    yvex_sequence_state *state = NULL;
    yvex_sequence_state_summary state_summary;
    yvex_runtime_execution_session session;
    yvex_error err;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(session_state_plan(&mixer, &err) == YVEX_OK,
                     "seal session recurrent plan");
    binding = (yvex_sequence_state_binding){
        .layer_index = 2ull, .plan = mixer};
    plan = (yvex_sequence_state_plan){
        .schema_version = YVEX_SEQUENCE_STATE_SCHEMA_V1,
        .bindings = &binding, .binding_count = 1ull};
    YVEX_TEST_ASSERT(yvex_sequence_state_open(&state, &plan, &err) == YVEX_OK,
                     "open session-owned recurrent state");
    memset(&session, 0, sizeof(session));
    YVEX_TEST_ASSERT(pthread_mutex_init(&session.lifecycle_mutex, NULL) == 0,
                     "initialize synthetic session lock");
    YVEX_TEST_ASSERT(pthread_cond_init(&session.idle_condition, NULL) == 0,
                     "initialize synthetic session condition");
    session.lifecycle_mutex_ready = 1;
    session.idle_condition_ready = 1;
    session.summary.open = 1;
    session.sequence_state = state;
    session.view.sequence_state = state;

    session_state_execution_begin(&session);
    YVEX_TEST_ASSERT(
        session_state_stage(state, 0ull, 4.0f, &err) &&
            yvex_runtime_session_finish_coordinated(
                &session, YVEX_OK, NULL, 0u, &err) == YVEX_OK &&
            yvex_sequence_state_summary_copy(state, &state_summary, &err) ==
                YVEX_OK &&
            state_summary.committed_position == 1ull &&
            session.summary.execution_count == 1ull &&
            session.summary.sequence_state_generation == 1ull &&
            !session.summary.busy,
        "session commit publishes recurrent state and execution together");

    session_state_execution_begin(&session);
    YVEX_TEST_ASSERT(session_state_stage(state, 1ull, 8.0f, &err),
                     "stage cancellable recurrent extension");
    yvex_error_set(&err, YVEX_ERR_CANCELLED, "test.sequence-state",
                   "injected cancellation");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_finish_coordinated(
            &session, YVEX_ERR_CANCELLED, NULL, 0u, &err) ==
                YVEX_ERR_CANCELLED &&
            yvex_sequence_state_summary_copy(state, &state_summary, &err) ==
                YVEX_OK &&
            state_summary.committed_position == 1ull &&
            session.summary.cancellation_count == 1ull &&
            !session.summary.invalidated && !session.summary.busy,
        "session cancellation aborts recurrent state without invalidation");

    session.sequence_state = NULL;
    session.view.sequence_state = NULL;
    yvex_sequence_state_close(&state);
    YVEX_TEST_ASSERT(pthread_cond_destroy(&session.idle_condition) == 0 &&
                         pthread_mutex_destroy(&session.lifecycle_mutex) == 0 &&
                         !state,
                     "synthetic session and recurrent owner close cleanly");
    return 0;
}
