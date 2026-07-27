/* Owner: tests/unit runtime logits.
 * Owns: output-head orientation, independent CPU arithmetic, policy, and lifecycle refusals.
 * Does not own: target-scale artifact execution or CUDA live qualification.
 * Invariants: synthetic matrices are test-only and production APIs remain the execution authority.
 * Boundary: focused contract tests over logits, qtype, and family adapter ABIs. */
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>

#include <math.h>
#include <string.h>

#include "tests/reference/logits.h"
#include "tests/test.h"

/* Purpose: encode one exactly represented finite BF16 test value. */
static void logits_test_bf16(unsigned char out[2], float value)
{
    unsigned int bits;
    memcpy(&bits, &value, sizeof(bits));
    out[0] = (unsigned char)(bits >> 16u);
    out[1] = (unsigned char)(bits >> 24u);
}

/* Purpose: prove independent row-major reference and production direct dots agree exactly. */
static int logits_test_orientation(void)
{
    static const float weights[3][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f, 0.0f},
        {1.0f, -1.0f, 2.0f, 0.5f}};
    const float hidden[4] = {1.0f, 2.0f, -1.0f, 0.5f};
    unsigned char encoded[24], mutated[24];
    float reference[3], production[3], changed[3];
    unsigned long long row, column;
    yvex_error err;
    for (row = 0ull; row < 3ull; ++row)
        for (column = 0ull; column < 4ull; ++column)
            logits_test_bf16(encoded + row * 8ull + column * 2ull,
                             weights[row][column]);
    YVEX_TEST_ASSERT(yvex_test_logits_reference_project(
                         YVEX_GGUF_QTYPE_BF16, encoded, sizeof(encoded),
                         3ull, 4ull, 8ull, hidden, reference, 3ull),
                     "independent BF16 reference projects exact rows");
    for (row = 0ull; row < 3ull; ++row) {
        yvex_quant_failure failure;
        memset(&failure, 0, sizeof(failure));
        YVEX_TEST_ASSERT(yvex_quant_cpu_dot(
                             YVEX_GGUF_QTYPE_BF16, encoded + row * 8ull, 8u,
                             hidden, 4ull, &production[row], &failure, &err) == YVEX_OK,
                         "production direct encoded row dot succeeds");
        YVEX_TEST_ASSERT(production[row] == reference[row],
                         "production and independent row orientation agree");
    }
    memcpy(mutated, encoded, sizeof(mutated));
    logits_test_bf16(mutated + 16u, 2.0f);
    YVEX_TEST_ASSERT(yvex_test_logits_reference_project(
                         YVEX_GGUF_QTYPE_BF16, mutated, sizeof(mutated),
                         3ull, 4ull, 8ull, hidden, changed, 3ull) &&
                         changed[2] != reference[2] &&
                         changed[0] == reference[0] && changed[1] == reference[1],
                     "one encoded output-head mutation affects only its vocabulary row");
    return 0;
}

/* Purpose: prove DeepSeek selects one separate unbiased output head and truthful capabilities. */
static int logits_test_policy(void)
{
    const yvex_runtime_family_adapter *adapter =
        yvex_runtime_family_adapter_find("deepseek4-v4-flash");
    yvex_logits_family_policy policy;
    yvex_runtime_capabilities capabilities;
    YVEX_TEST_ASSERT(adapter && adapter->logits_policy &&
                         adapter->logits_policy(&policy),
                     "DeepSeek adapter exposes typed logits policy");
    YVEX_TEST_ASSERT(policy.schema_version == YVEX_RUNTIME_LOGITS_SCHEMA_V1 &&
                         policy.separate_output_head && !policy.tied_output_head &&
                         !policy.output_head_bias,
                     "DeepSeek policy requires one separate unbiased head");
    memset(&capabilities, 0, sizeof(capabilities));
    YVEX_TEST_ASSERT(adapter->execution_capabilities(&capabilities) &&
                         capabilities.output_head_binding_ready &&
                         capabilities.output_head_projection_ready &&
                         capabilities.logits_cpu_ready && capabilities.logits_cuda_ready &&
                         capabilities.logits_prefill_ready && capabilities.logits_decode_ready &&
                         capabilities.logits_full_vocabulary_ready &&
                         capabilities.logits_hidden_contract_ready &&
                         capabilities.logits_partial_progress_ready &&
                         capabilities.logits_ready && !capabilities.generation_ready &&
                         yvex_runtime_capabilities_contract_valid(&capabilities),
                     "adapter capability lattice exposes logits without generation");
    return 0;
}

/* Purpose: prove absent owners and caller storage refuse without output publication. */
static int logits_test_lifecycle_refusal(void)
{
    yvex_runtime_logits_context *context = NULL;
    yvex_runtime_logits_options options = {.maximum_rows = 1ull};
    yvex_runtime_logits_row_result row;
    yvex_runtime_logits_result aggregate;
    yvex_runtime_logits_source source;
    float output = 91.0f;
    yvex_error err;
    memset(&row, 0, sizeof(row));
    memset(&aggregate, 0, sizeof(aggregate));
    memset(&source, 0, sizeof(source));
    YVEX_TEST_ASSERT(yvex_runtime_logits_context_open(
                         &context, NULL, NULL, NULL, &options, &err) ==
                         YVEX_ERR_INVALID_ARG && !context,
                     "logits context refuses missing production owners");
    YVEX_TEST_ASSERT(yvex_runtime_logits_project(
                         NULL, &source, YVEX_BACKEND_KIND_CPU, &output, 1ull,
                         &row, &err) == YVEX_ERR_STATE && output == 91.0f && !row.completed,
                     "missing context publishes no logits row");
    YVEX_TEST_ASSERT(yvex_runtime_logits_execute(
                         NULL, &source, 1ull, YVEX_BACKEND_KIND_CPU,
                         &output, 1ull, &row, 1ull, &aggregate, &err) ==
                         YVEX_ERR_INVALID_ARG && output == 91.0f &&
                         !aggregate.completed,
                     "missing repeated lifecycle publishes no partial output");
    YVEX_TEST_ASSERT(yvex_runtime_logits_context_close(&context, &err) == YVEX_OK,
                     "empty logits close is idempotent");
    return 0;
}

int yvex_test_runtime_logits(void)
{
    if (logits_test_orientation() != 0) return 1;
    if (logits_test_policy() != 0) return 1;
    if (logits_test_lifecycle_refusal() != 0) return 1;
    return 0;
}
