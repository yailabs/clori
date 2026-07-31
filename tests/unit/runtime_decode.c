/*
 * Exercises decode evidence identities and malformed lifecycle requests fail closed. Tests
 * exercise field-wise public internal ABI and never enter production objects. Focused
 * decode-contract evidence over deterministic caller-owned facts.
 */
#include "tests/test.h"

#include <string.h>

#include <yvex/internal/decode.h>

static void decode_test_identity(char output[YVEX_SHA256_HEX_CAP], unsigned int value)
{
    (void)snprintf(output, YVEX_SHA256_HEX_CAP, "%064x", value);
}

static void decode_test_step(yvex_runtime_decode_step_result *step)
{
    memset(step, 0, sizeof(*step));
    step->schema_version = YVEX_RUNTIME_DECODE_SCHEMA_V1;
    step->completed = 1;
    step->step_ordinal = 0ull;
    step->token_id = 7u;
    step->position_before = 2ull;
    step->position_after = 3ull;
    step->generation_before = 4ull;
    step->generation_after = 5ull;
    decode_test_identity(step->embedding_digest, 1u);
    decode_test_identity(step->routing_digest, 2u);
    decode_test_identity(step->layer_digest, 3u);
    decode_test_identity(step->normalized_hidden_digest, 4u);
    decode_test_identity(step->persistent_state_digest, 5u);
    decode_test_identity(step->transformer_execution_identity, 6u);
}

static int decode_test_step_identity(void)
{
    yvex_runtime_decode_step_result step;
    char first[YVEX_SHA256_HEX_CAP], second[YVEX_SHA256_HEX_CAP];
    decode_test_step(&step);
    YVEX_TEST_ASSERT(yvex_runtime_decode_step_identity(&step, first),
                     "complete decode step seals field-by-field");
    step.token_id++;
    YVEX_TEST_ASSERT(yvex_runtime_decode_step_identity(&step, second) &&
                         strcmp(first, second) != 0,
                     "token mutation changes decode step identity");
    step.token_id--;
    step.layers_executed++;
    YVEX_TEST_ASSERT(yvex_runtime_decode_step_identity(&step, second) &&
                         strcmp(first, second) != 0,
                     "structural counter mutation changes decode step identity");
    step.layers_executed--;
    step.position_after++;
    YVEX_TEST_ASSERT(!yvex_runtime_decode_step_identity(&step, second),
                     "non-unit position transition refuses identity");
    return 0;
}

/* Prove ordered partial progress has one stable aggregate identity. */
static int decode_test_result_identity(void)
{
    yvex_runtime_decode_step_result step;
    yvex_runtime_decode_result result;
    char first[YVEX_SHA256_HEX_CAP], second[YVEX_SHA256_HEX_CAP];
    decode_test_step(&step);
    YVEX_TEST_ASSERT(yvex_runtime_decode_step_identity(
                         &step, step.decode_step_identity),
                     "step identity is available to repeated evidence");
    memset(&result, 0, sizeof(result));
    result.schema_version = YVEX_RUNTIME_DECODE_SCHEMA_V1;
    result.status = YVEX_RUNTIME_DECODE_STATUS_PARTIAL;
    result.partial = result.has_incomplete_step = 1;
    result.requested_steps = 2ull;
    result.completed_steps = 1ull;
    result.first_incomplete_step = 1ull;
    result.initial_committed_prefix = 2ull;
    result.final_committed_prefix = 3ull;
    result.generation_before = 4ull;
    result.generation_after = 5ull;
    decode_test_identity(result.input_identity, 7u);
    decode_test_identity(result.aggregate_hidden_digest, 8u);
    decode_test_identity(result.aggregate_state_digest, 9u);
    YVEX_TEST_ASSERT(yvex_runtime_decode_result_identity(&result, &step, first),
                     "partial decode result remains identity-bearing");
    step.decode_step_identity[0] = step.decode_step_identity[0] == 'a' ? 'b' : 'a';
    YVEX_TEST_ASSERT(yvex_runtime_decode_result_identity(&result, &step, second) &&
                         strcmp(first, second) != 0,
                     "ordered step mutation changes aggregate decode identity");
    decode_test_step(&step);
    YVEX_TEST_ASSERT(yvex_runtime_decode_step_identity(
                         &step, step.decode_step_identity),
                     "step identity resets before aggregate-field mutation");
    result.first_incomplete_step = 0ull;
    YVEX_TEST_ASSERT(yvex_runtime_decode_result_identity(&result, &step, second) &&
                         strcmp(first, second) != 0,
                     "partial-status mutation changes aggregate decode identity");
    result.first_incomplete_step = 1ull;
    result.final_committed_prefix++;
    YVEX_TEST_ASSERT(!yvex_runtime_decode_result_identity(&result, &step, second),
                     "inconsistent partial prefix refuses identity");
    return 0;
}

static int decode_test_lifecycle_refusal(void)
{
    yvex_runtime_decode_context *context = NULL;
    yvex_runtime_decode_options options = {.maximum_steps = 2ull};
    yvex_runtime_decode_step_result result;
    yvex_error err;
    YVEX_TEST_ASSERT(yvex_runtime_decode_context_open(
                         &context, NULL, NULL, &options, &err) == YVEX_ERR_INVALID_ARG &&
                         context == NULL,
                     "decode context refuses absent paired transformer/session owners");
    memset(&result, 0, sizeof(result));
    YVEX_TEST_ASSERT(yvex_runtime_decode_step(
                         NULL, 0ull, 1ull, 1u, YVEX_BACKEND_KIND_CPU,
                         NULL, 0ull, &result, &err) == YVEX_ERR_INVALID_ARG &&
                         !result.completed,
                     "decode step refuses absent lifecycle and output owners");
    YVEX_TEST_ASSERT(yvex_runtime_decode_context_close(&context, &err) == YVEX_OK,
                     "empty decode close is idempotent");
    return 0;
}

int yvex_test_runtime_decode(void)
{
    if (decode_test_step_identity() != 0) return 1;
    if (decode_test_result_identity() != 0) return 1;
    if (decode_test_lifecycle_refusal() != 0) return 1;
    return 0;
}
