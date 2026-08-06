/*
 * Keep the common generation owner in focused and sanitizer aggregates. Focused tests never
 * fabricate a production model/session or teacher-forced generation path. Sanitizer-visible
 * contract proof; complete composition belongs to the live runner.
 */
#include "tests/test.h"

#include <string.h>

#include <yvex/internal/generation.h>

static int generation_test_stop_taxonomy(void)
{
    unsigned int reason;
    for (reason = YVEX_GENERATION_STOP_NONE;
         reason <= YVEX_GENERATION_STOP_OUTPUT_FAILURE; ++reason)
        YVEX_TEST_ASSERT(
            strcmp(yvex_runtime_generation_stop_reason_name(
                       (yvex_runtime_generation_stop_reason)reason),
                   "unknown") != 0,
            "generation stop reason must have a stable name");
    YVEX_TEST_ASSERT(strcmp(yvex_runtime_generation_stop_reason_name(
                               (yvex_runtime_generation_stop_reason)99),
                           "unknown") == 0,
                     "unknown generation stop reason must remain explicit");
    return 0;
}

static int generation_test_refusals(void)
{
    yvex_runtime_generation_context *context = NULL;
    yvex_runtime_generation_result result;
    yvex_runtime_generation_options options;
    yvex_generation_operator_result operator_result;
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_error err;
    memset(&result, 0, sizeof(result));
    memset(&options, 0, sizeof(options));
    memset(&operator_result, 0, sizeof(operator_result));
    YVEX_TEST_ASSERT(yvex_runtime_generation_context_open(
                         &context, NULL, NULL, &options, &err) ==
                         YVEX_ERR_INVALID_ARG && !context,
                     "generation context must refuse absent model/session");
    YVEX_TEST_ASSERT(yvex_runtime_generation_execute(
                         NULL, NULL, NULL, 0ull, NULL, 0ull, &result, &err) ==
                         YVEX_ERR_INVALID_ARG,
                     "generation execute must refuse absent context");
    YVEX_TEST_ASSERT(yvex_runtime_generation_result_validate(
                         NULL, NULL, 0ull, NULL, 0ull, &result, &err) ==
                         YVEX_ERR_FORMAT,
                     "generation result validation must fail closed");
    YVEX_TEST_ASSERT(yvex_runtime_generation_operator_execute(
                         NULL, &operator_result, &cleanup, &err) ==
                         YVEX_ERR_INVALID_ARG && !cleanup,
                     "generation operator must refuse absent request");
    YVEX_TEST_ASSERT(yvex_runtime_generation_context_close(&context, &err) ==
                         YVEX_OK && !context,
                     "generation close must be idempotent for empty ownership");
    yvex_runtime_generation_operator_result_release(&operator_result);
    return 0;
}

static int generation_test_execution_identity_excludes_measurement(void)
{
    yvex_runtime_generation_result result;
    char before[YVEX_SHA256_HEX_CAP], after[YVEX_SHA256_HEX_CAP];
    memset(&result, 0, sizeof(result));
    result.schema_version = YVEX_RUNTIME_GENERATION_RESULT_SCHEMA_V4;
    result.execution_mode = YVEX_GENERATION_MODE_DSPARK;
    result.draft_cycle_count = 2ull;
    result.proposed_token_count = 8ull;
    result.accepted_draft_token_count = 6ull;
    result.rejected_draft_token_count = 2ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_execution_identity(&result, NULL, before),
        "generation execution identity must admit a bounded result");
    result.draft_ns = 123456ull;
    result.verification_ns = 654321ull;
    result.speculative_commit_ns = 777ull;
    result.mean_accepted_prefix = 3.0;
    result.effective_committed_tokens_per_second = 0.75;
    result.roofline_available = 1;
    memset(result.roofline.identity, 'a', YVEX_SHA256_HEX_CAP - 1u);
    result.roofline.identity[YVEX_SHA256_HEX_CAP - 1u] = '\0';
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_execution_identity(&result, NULL, after) &&
            strcmp(before, after) == 0,
        "measurement values must not alter semantic execution identity");
    result.proposed_token_count++;
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_execution_identity(&result, NULL, after) &&
            strcmp(before, after) != 0,
        "semantic speculative counters must alter execution identity");
    return 0;
}

static int generation_test_plan_binds_workload_profile(void)
{
    yvex_runtime_generation_plan_summary plan;
    char before[YVEX_SHA256_HEX_CAP], after[YVEX_SHA256_HEX_CAP];
    memset(&plan, 0, sizeof(plan));
    plan.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5;
    strcpy(plan.workload_profile_identity,
           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    YVEX_TEST_ASSERT(yvex_runtime_generation_plan_identity(&plan, before),
                     "generation plan must bind a workload profile");
    plan.workload_profile_identity[0] = 'b';
    YVEX_TEST_ASSERT(yvex_runtime_generation_plan_identity(&plan, after) &&
                         strcmp(before, after) != 0,
                     "workload profile changes must alter generation plan identity");
    return 0;
}

int yvex_test_runtime_generation(void)
{
    if (generation_test_stop_taxonomy() != 0) return 1;
    if (generation_test_refusals() != 0) return 1;
    if (generation_test_execution_identity_excludes_measurement() != 0) return 1;
    if (generation_test_plan_binds_workload_profile() != 0) return 1;
    return 0;
}
