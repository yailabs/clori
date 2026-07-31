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

int yvex_test_runtime_generation(void)
{
    if (generation_test_stop_taxonomy() != 0) return 1;
    if (generation_test_refusals() != 0) return 1;
    return 0;
}
