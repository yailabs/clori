/* CUDA sampling conformance against analytically bounded stochastic fixtures. */
#include <yvex/api.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/decode.h>
#include <yvex/internal/sampling.h>

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tests/test.h"

static void sampling_tensor_desc(yvex_backend_tensor_desc *desc,
                                 const char *name, yvex_dtype dtype,
                                 unsigned long long bytes)
{
    memset(desc, 0, sizeof(*desc));
    desc->name = name;
    desc->dtype = dtype;
    desc->rank = 1u;
    desc->dims[0] = dtype == YVEX_DTYPE_F32 ? bytes / sizeof(float) : bytes;
    desc->bytes = bytes;
}

static int sampling_open_cuda(yvex_backend **out)
{
    yvex_backend_options options = {0};
    yvex_error err;
    int rc;
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(out, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        fprintf(stderr, "SKIP: CUDA unavailable: %s\n", yvex_error_message(&err));
        return 77;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open CUDA backend for stochastic sampling");
    return 0;
}

static yvex_runtime_sampling_policy sampling_policy(void)
{
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC,
        .temperature = 1.0,
        .top_p = 1.0,
        .typical_p = 1.0,
        .seed_present = 1,
        .seed = 42ull};
    return policy;
}

static void sampling_identity(char output[YVEX_SHA256_HEX_CAP], char digit)
{
    memset(output, digit, YVEX_SHA256_HEX_CAP - 1u);
    output[YVEX_SHA256_HEX_CAP - 1u] = '\0';
}

static int sampling_device_row(
    yvex_backend *backend, const yvex_device_tensor *tensor,
    const yvex_runtime_logits_plan_summary *plan,
    yvex_runtime_logits_row_result *row)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    memset(row, 0, sizeof(*row));
    row->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    row->completed = row->device_values_available = 1;
    row->evidence_profile = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    row->source_phase = YVEX_LOGITS_SOURCE_DECODE;
    row->source_position = 1ull;
    row->vocabulary_size = row->logits_count = plan->vocabulary_size;
    row->hidden_width = plan->hidden_width;
    sampling_identity(row->source_hidden_digest, 'b');
    sampling_identity(row->output_head_residency_identity, 'c');
    sampling_identity(row->backend_execution_identity, 'd');
    yvex_runtime_identity_copy(row->output_head_plan_identity,
                               plan->output_head_plan_identity);
    row->device_logits.schema_version = YVEX_EXECUTION_DEVICE_VIEW_SCHEMA_V1;
    row->device_logits.kind = YVEX_EXECUTION_DEVICE_LOGITS;
    row->device_logits.backend = backend;
    row->device_logits.tensor = tensor;
    row->device_logits.resource_generation = row->device_logits.session_generation =
        row->device_logits.state_generation = 1ull;
    row->device_logits.rows = 1ull;
    row->device_logits.columns = plan->vocabulary_size;
    row->device_logits.element_bytes = sizeof(float);
    row->device_logits.dtype = YVEX_DTYPE_F32;
    row->device_logits.synchronization_required = 1;
    row->device_logits.materialization = YVEX_EXECUTION_MATERIALIZE_NONE;
    sampling_identity(row->device_logits.runtime_model_identity, 'e');
    sampling_identity(row->device_logits.execution_profile_identity, 'f');
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.row.v2") ||
        !yvex_sha256_update_u64(&hash, row->source_phase) ||
        !yvex_sha256_update_u64(&hash, row->source_position) ||
        !yvex_sha256_update_u64(&hash, row->vocabulary_size) ||
        !yvex_sha256_update_u64(&hash, row->host_values_available) ||
        !yvex_sha256_update_u64(&hash, row->device_values_available) ||
        !yvex_sha256_update_u64(&hash, row->evidence_profile) ||
        !yvex_sha256_update_text(&hash, row->source_hidden_digest) ||
        !yvex_sha256_update_text(&hash, row->output_head_plan_identity) ||
        !yvex_sha256_update_text(&hash, row->output_head_residency_identity) ||
        !yvex_sha256_update_text(&hash, row->backend_execution_identity) ||
        !yvex_sha256_update_text(&hash, row->device_logits.execution_profile_identity) ||
        !yvex_sha256_update_u64(&hash, row->device_logits.resource_generation) ||
        !yvex_sha256_update_u64(&hash, row->device_logits.session_generation) ||
        !yvex_sha256_update_u64(&hash, row->device_logits.state_generation) ||
        !yvex_sha256_update_u64(&hash, row->device_logits.element_offset) ||
        !yvex_sha256_update_u64(&hash, row->device_logits.rows) ||
        !yvex_sha256_update_u64(&hash, row->device_logits.columns) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, row->logits_row_identity);
    return 1;
}

static int sampling_transactional_device(
    yvex_backend *backend, const yvex_device_tensor *device_logits)
{
    yvex_runtime_logits_plan_summary plan = {0};
    yvex_runtime_logits_row_result row;
    yvex_runtime_sampling_policy policy = sampling_policy();
    yvex_runtime_sampling_options options = {
        .maximum_vocabulary_size = 4ull, .maximum_rows = 2ull, .device_selection = 1};
    yvex_runtime_sampling_context *context = NULL;
    yvex_runtime_sampling_transaction *transaction = NULL;
    yvex_runtime_sampling_context_summary before, after;
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_result aborted, committed;
    yvex_error err;
    plan.schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    plan.vocabulary_size = plan.row_count = 4ull;
    plan.row_width = plan.hidden_width = 4ull;
    sampling_identity(plan.output_head_plan_identity, 'a');
    YVEX_TEST_ASSERT(
        sampling_device_row(backend, device_logits, &plan, &row) &&
            yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_open(
                &context, &plan, &policy, &options, &err) == YVEX_OK &&
            yvex_runtime_sampling_source_from_logits(
                context, &source, NULL, 0ull, &row, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_snapshot(context, &before, &err) == YVEX_OK,
        "admit transaction-owned stochastic CUDA selection");
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_transaction_begin(context, &transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_select(
                context, transaction, &source, &aborted, &err) == YVEX_OK &&
            yvex_runtime_sampling_transaction_abort(&transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
            after.successful_samples == before.successful_samples &&
            after.stochastic_draws == before.stochastic_draws &&
            yvex_runtime_sampling_transaction_begin(context, &transaction, &err) == YVEX_OK &&
            yvex_runtime_sampling_select(
                context, transaction, &source, &committed, &err) == YVEX_OK &&
            strcmp(aborted.execution_identity, committed.execution_identity) == 0 &&
            yvex_runtime_sampling_transaction_prepare_commit(transaction, &err) == YVEX_OK,
        "aborted CUDA selection retries the same staged token and RNG transition");
    yvex_runtime_sampling_transaction_publish_commit(&transaction);
    YVEX_TEST_ASSERT(
        !transaction && committed.device_selection && !committed.full_array_host_scan_bytes &&
            yvex_runtime_sampling_context_snapshot(context, &after, &err) == YVEX_OK &&
            after.successful_samples == before.successful_samples + 1ull &&
            after.stochastic_draws == before.stochastic_draws + 1ull &&
            strcmp(after.rng_state_identity, committed.rng_state_after_identity) == 0 &&
            yvex_runtime_sampling_context_close(&context, &err) == YVEX_OK,
        "CUDA sampling transaction publishes one bounded device result and one RNG draw");
    return 0;
}

static int sampling_greedy_rows(
    yvex_backend *backend, const yvex_backend_sampling_operations *operations)
{
    const float values[] = {
        -4.0f, 3.5f, 1.0f, 3.5f, -2.0f, 0.0f, 2.0f,
         9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f,
        -1.0f, 0.0f, 2.0f, 1.0f, 2.0f, 2.0f, 0.0f};
    yvex_backend_tensor_desc descriptor;
    yvex_device_tensor *device = NULL;
    yvex_backend_operation_facts facts;
    unsigned long long ties[3] = {0ull};
    unsigned int tokens[3] = {UINT_MAX, UINT_MAX, UINT_MAX};
    float selected[3] = {0.0f};
    yvex_error err;
    sampling_tensor_desc(&descriptor, "sampling-greedy-rows", YVEX_DTYPE_F32,
                         sizeof(values));
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &device, &err) == YVEX_OK &&
            yvex_backend_tensor_write(
                backend, device, values, sizeof(values), &err) == YVEX_OK &&
            operations->select_greedy_rows(
                backend, device, 3ull, 7ull, tokens, selected, ties, &facts, &err) == YVEX_OK,
        "execute one CUDA argmax over three resident rows");
    YVEX_TEST_ASSERT(
        tokens[0] == 1u && selected[0] == 3.5f && ties[0] == 2ull &&
            tokens[1] == 0u && selected[1] == 9.0f && ties[1] == 1ull &&
            tokens[2] == 2u && selected[2] == 2.0f && ties[2] == 3ull &&
            facts.kernel_launches == 1ull && facts.queue_synchronizations == 1ull &&
            facts.device_synchronizations == 0ull &&
            facts.d2h_bytes == sizeof(int) + sizeof(tokens) + sizeof(selected) + sizeof(ties),
        "batched CUDA argmax preserves lowest-token tie policy and aggregate physical facts");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &device, &err) == YVEX_OK,
        "release CUDA greedy row fixture");
    return 0;
}

static int sampling_full_vocabulary(
    yvex_backend *backend, const yvex_backend_sampling_operations *operations)
{
    const unsigned long long vocabulary = 129280ull;
    yvex_device_tensor *logits = NULL, *workspace = NULL;
    yvex_backend_tensor_desc descriptor;
    yvex_runtime_sampling_policy policy = sampling_policy();
    yvex_backend_sampling_result result;
    yvex_backend_operation_facts facts;
    yvex_error err;
    unsigned long long workspace_bytes;
    float *uniform = calloc((size_t)vocabulary, sizeof(*uniform));
    YVEX_TEST_ASSERT(uniform != NULL, "allocate full-vocabulary CUDA sampling fixture");
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_policy_seal(&policy, vocabulary, &err) == YVEX_OK &&
            operations->workspace_required(vocabulary, &workspace_bytes, &err) == YVEX_OK,
        "derive full-vocabulary stochastic workspace");
    sampling_tensor_desc(&descriptor, "sampling-full-logits", YVEX_DTYPE_F32,
                         vocabulary * sizeof(*uniform));
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &logits, &err) == YVEX_OK &&
            yvex_backend_tensor_write(
                backend, logits, uniform, vocabulary * sizeof(*uniform), &err) == YVEX_OK,
        "upload full-vocabulary uniform logits");
    sampling_tensor_desc(&descriptor, "sampling-full-workspace", YVEX_DTYPE_I8,
                         workspace_bytes);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &workspace, &err) == YVEX_OK &&
            yvex_backend_workspace_attach(backend, workspace, 3ull, &err) == YVEX_OK &&
            operations->select_stochastic(
                backend, logits, vocabulary, &policy, 0u,
                &result, &facts, &err) == YVEX_OK &&
            result.selected_token_id == 0u &&
            result.candidates_after_top_p == vocabulary && facts.d2h_bytes == 100ull,
        "full vocabulary selects on device with constant bounded download");
    yvex_backend_workspace_detach(backend);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &workspace, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &logits, &err) == YVEX_OK,
        "release full-vocabulary stochastic fixture");
    free(uniform);
    return 0;
}

static void sampling_softmax_row(const float *logits, unsigned long long count,
                                 float *probabilities)
{
    double maximum = logits[0], total = 0.0;
    unsigned long long index;
    for (index = 1ull; index < count; ++index)
        if (logits[index] > maximum) maximum = logits[index];
    for (index = 0ull; index < count; ++index) {
        probabilities[index] = (float)exp((double)logits[index] - maximum);
        total += probabilities[index];
    }
    for (index = 0ull; index < count; ++index)
        probabilities[index] = (float)(probabilities[index] / total);
}

static int sampling_speculation_device(
    yvex_backend *backend, const yvex_backend_sampling_operations *operations)
{
    static const float draft_logits[] = {
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f};
    static const float target_cases[][12] = {
        {0.0f, 2.0f, 0.0f, 0.0f,
         0.0f, 2.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 2.0f, 0.0f},
        {2.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 2.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 2.0f, 0.0f},
        {2.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 2.0f, 0.0f,
         0.0f, 0.0f, 2.0f, 0.0f}};
    const unsigned int candidates[] = {0u, 1u};
    static const double uniform_cases[][2] = {
        {0.5, 0.0}, {0.5, 0.5}, {0.5, 0.5}};
    static const double correction_cases[] = {0.25, 0.5, 0.25};
    static const unsigned long long accepted_cases[] = {0ull, 2ull, 1ull};
    float draft_probabilities[8], target_probabilities[12];
    unsigned int device_tokens[3] = {UINT_MAX, UINT_MAX, UINT_MAX};
    unsigned int reference_tokens[3] = {UINT_MAX, UINT_MAX, UINT_MAX};
    yvex_runtime_sampling_policy policy = sampling_policy();
    yvex_speculation_acceptance_request request = {0};
    yvex_speculation_acceptance_result reference = {0};
    yvex_backend_speculation_result result = {0};
    yvex_backend_operation_facts facts = {0};
    yvex_backend_tensor_desc descriptor;
    yvex_device_tensor *draft = NULL, *target = NULL, *workspace = NULL;
    unsigned long long workspace_bytes, row, case_index;
    yvex_error err;
    for (row = 0ull; row < 2ull; ++row)
        sampling_softmax_row(
            draft_logits + row * 4ull, 4ull,
            draft_probabilities + row * 4ull);
    request.schema_version = YVEX_SPECULATION_SCHEMA_V1;
    request.kind = YVEX_SPECULATION_ACCEPT_STOCHASTIC;
    request.candidate_count = 2ull;
    request.vocabulary_size = request.distribution_stride = 4ull;
    request.candidate_token_ids = candidates;
    request.draft_probabilities = draft_probabilities;
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK &&
            operations->speculation_workspace_required(
                4ull, 2ull, &workspace_bytes, &err) == YVEX_OK,
        "derive stochastic speculation reference and workspace");
    sampling_tensor_desc(
        &descriptor, "speculation-draft-logits", YVEX_DTYPE_F32,
        sizeof(draft_logits));
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &draft, &err) == YVEX_OK &&
            yvex_backend_tensor_write(
                backend, draft, draft_logits, sizeof(draft_logits), &err) == YVEX_OK,
        "upload resident draft distributions");
    sampling_tensor_desc(
        &descriptor, "speculation-target-logits", YVEX_DTYPE_F32,
        sizeof(target_cases[0]));
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &target, &err) == YVEX_OK,
        "allocate resident target distributions");
    sampling_tensor_desc(
        &descriptor, "speculation-small-workspace", YVEX_DTYPE_I8,
        workspace_bytes - 1ull);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_write(
            backend, target, target_cases[0], sizeof(target_cases[0]), &err) == YVEX_OK &&
            yvex_backend_tensor_alloc(backend, &descriptor, &workspace, &err) == YVEX_OK &&
            yvex_backend_workspace_attach(backend, workspace, 4ull, &err) == YVEX_OK &&
            operations->accept_stochastic(
                backend, draft, target, 2ull, 4ull, &policy, candidates,
                uniform_cases[0], correction_cases[0], device_tokens, 3ull,
                &result, &facts, &err) == YVEX_ERR_NOMEM &&
            !result.completed && !facts.kernel_launches,
        "speculative acceptance refuses a workspace below its derived capacity");
    yvex_backend_workspace_detach(backend);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &workspace, &err) == YVEX_OK,
        "release refused stochastic speculation workspace");
    sampling_tensor_desc(
        &descriptor, "speculation-workspace", YVEX_DTYPE_I8, workspace_bytes);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &workspace, &err) == YVEX_OK &&
            yvex_backend_workspace_attach(backend, workspace, 4ull, &err) == YVEX_OK,
        "attach stochastic speculation workspace");
    {
        const unsigned int invalid_candidates[] = {4u, 1u};
        memset(device_tokens, 0xff, sizeof(device_tokens));
        YVEX_TEST_ASSERT(
            operations->accept_stochastic(
                backend, draft, target, 2ull, 4ull, &policy,
                invalid_candidates, uniform_cases[0], correction_cases[0],
                device_tokens, 3ull, &result, &facts, &err) == YVEX_ERR_FORMAT &&
                !result.completed && !facts.kernel_launches &&
                device_tokens[0] == 0u && device_tokens[1] == 0u &&
                device_tokens[2] == 0u,
            "invalid device candidates refuse before launch or token publication");
    }
    for (case_index = 0ull; case_index < 3ull; ++case_index) {
        memset(device_tokens, 0xff, sizeof(device_tokens));
        memset(reference_tokens, 0xff, sizeof(reference_tokens));
        for (row = 0ull; row < 3ull; ++row)
            sampling_softmax_row(
                target_cases[case_index] + row * 4ull, 4ull,
                target_probabilities + row * 4ull);
        request.target_probabilities = target_probabilities;
        request.acceptance_uniforms = uniform_cases[case_index];
        request.correction_uniform = correction_cases[case_index];
        YVEX_TEST_ASSERT(
            yvex_speculation_accept(
                &request, reference_tokens, 3ull, &reference, &err) == YVEX_OK &&
                yvex_backend_tensor_write(
                    backend, target, target_cases[case_index],
                    sizeof(target_cases[case_index]), &err) == YVEX_OK &&
                operations->accept_stochastic(
                    backend, draft, target, 2ull, 4ull, &policy, candidates,
                    uniform_cases[case_index], correction_cases[case_index],
                    device_tokens, 3ull, &result, &facts, &err) == YVEX_OK,
            "execute CUDA acceptance at each speculative prefix length");
        YVEX_TEST_ASSERT(
            result.completed &&
                result.accepted_draft_count == accepted_cases[case_index] &&
                result.accepted_draft_count == reference.accepted_draft_count &&
                result.rejected_draft_count == reference.rejected_draft_count &&
                result.committed_count == reference.committed_count &&
                result.rejection_index == reference.rejection_index &&
                result.correction_present == reference.correction_present &&
                result.bonus_present == reference.bonus_present &&
                memcmp(device_tokens, reference_tokens,
                       (size_t)reference.committed_count *
                           sizeof(*device_tokens)) == 0 &&
                facts.h2d_bytes ==
                    sizeof(candidates) + sizeof(uniform_cases[case_index]) &&
                facts.d2h_bytes <= 128ull && facts.kernel_launches == 1ull &&
                facts.queue_synchronizations == 1ull &&
                facts.device_synchronizations == 0ull &&
                facts.activation_bytes ==
                    sizeof(draft_logits) + sizeof(target_cases[case_index]) &&
                facts.temporary_bytes == workspace_bytes,
            "device p/q acceptance matches the independent bounded oracle");
    }
    yvex_backend_workspace_detach(backend);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &workspace, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &target, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &draft, &err) == YVEX_OK,
        "release stochastic speculation device fixtures");
    return 0;
}

int yvex_cuda_test_sampling(void)
{
    static const float logits[] = {0.0f, 1.0f, 2.0f, 3.0f};
    yvex_backend *backend = NULL;
    yvex_device_tensor *device_logits = NULL, *workspace = NULL;
    yvex_backend_tensor_desc descriptor;
    const yvex_backend_sampling_operations *operations;
    yvex_runtime_sampling_policy policy = sampling_policy();
    yvex_backend_sampling_result result;
    yvex_backend_operation_facts facts;
    yvex_error err;
    unsigned long long workspace_bytes = 0ull;
    double denominator = 1.0 + exp(1.0) + exp(2.0) + exp(3.0);
    int rc = sampling_open_cuda(&backend);
    if (rc != 0) return rc;
    operations = yvex_backend_sampling_operations_get(backend);
    YVEX_TEST_ASSERT(
        operations && operations->workspace_required &&
            operations->speculation_workspace_required &&
            operations->select_greedy_rows && operations->select_stochastic &&
            operations->accept_stochastic &&
            yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK &&
            operations->workspace_required(4ull, &workspace_bytes, &err) == YVEX_OK &&
            workspace_bytes > sizeof(logits),
        "CUDA stochastic sampling admits sealed policy and derived workspace");
    YVEX_TEST_ASSERT(sampling_greedy_rows(backend, operations) == 0,
                     "CUDA sampling owns batched greedy selection");
    YVEX_TEST_ASSERT(sampling_speculation_device(backend, operations) == 0,
                     "CUDA sampling owns stochastic speculative acceptance");
    sampling_tensor_desc(&descriptor, "sampling-logits", YVEX_DTYPE_F32,
                         sizeof(logits));
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &device_logits, &err) == YVEX_OK &&
            yvex_backend_tensor_write(
                backend, device_logits, logits, sizeof(logits), &err) == YVEX_OK,
        "upload bounded stochastic logits fixture");
    YVEX_TEST_ASSERT(
        operations->select_stochastic(
            backend, device_logits, 4ull, &policy, 0u,
            &result, &facts, &err) == YVEX_ERR_NOMEM && !result.completed,
        "sampling refuses implicit CUDA allocation without admitted workspace");
    sampling_tensor_desc(&descriptor, "sampling-small-workspace", YVEX_DTYPE_I8,
                         workspace_bytes - 1ull);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &workspace, &err) == YVEX_OK &&
            yvex_backend_workspace_attach(backend, workspace, 1ull, &err) == YVEX_OK &&
            operations->select_stochastic(
                backend, device_logits, 4ull, &policy, 0u,
                &result, &facts, &err) == YVEX_ERR_NOMEM,
        "sampling refuses an attached workspace one byte below derived capacity");
    yvex_backend_workspace_detach(backend);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &workspace, &err) == YVEX_OK,
        "release refused sampling workspace");
    sampling_tensor_desc(&descriptor, "sampling-workspace", YVEX_DTYPE_I8,
                         workspace_bytes);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &workspace, &err) == YVEX_OK &&
            yvex_backend_workspace_attach(backend, workspace, 2ull, &err) == YVEX_OK &&
            operations->select_stochastic(
                backend, device_logits, 4ull, &policy, 0u,
                &result, &facts, &err) == YVEX_OK,
        "execute neutral stochastic sampling on resident logits");
    YVEX_TEST_ASSERT(
        result.completed && result.selected_token_id == 0u &&
            result.candidates_after_top_k == 4ull &&
            result.candidates_after_min_p == 4ull &&
            result.candidates_after_typical_p == 4ull &&
            result.candidates_after_top_p == 4ull &&
            fabs(result.selected_probability - 1.0 / denominator) < 1.0e-14 &&
            facts.kernel_launches == 1ull && facts.queue_synchronizations == 1ull &&
            facts.device_synchronizations == 0ull &&
            facts.download_count == 5ull && facts.d2h_bytes == 100ull &&
            facts.activation_bytes == sizeof(logits) &&
            facts.temporary_bytes == workspace_bytes,
        "device categorical selection matches analytic softmax with bounded facts");
    YVEX_TEST_ASSERT(
        operations->select_stochastic(
            backend, device_logits, 4ull, &policy, 0xffffffffu,
            &result, &facts, &err) == YVEX_OK &&
            result.selected_token_id == 3u,
        "categorical endpoint preserves token-order draw semantics");
    policy = sampling_policy();
    policy.top_k = 4ull;
    policy.min_p = 0.2;
    policy.typical_p = 0.7;
    policy.top_p = 0.6;
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_policy_seal(&policy, 4ull, &err) == YVEX_OK &&
            operations->select_stochastic(
                backend, device_logits, 4ull, &policy, 0u,
                &result, &facts, &err) == YVEX_OK &&
            result.candidates_after_top_k == 4ull &&
            result.candidates_after_min_p == 2ull &&
            result.candidates_after_typical_p == 1ull &&
            result.candidates_after_top_p == 1ull &&
            result.selected_token_id == 3u && result.selected_probability == 1.0,
        "device filters preserve canonical top-k min-p typical-p top-p order");
    YVEX_TEST_ASSERT(sampling_transactional_device(backend, device_logits) == 0,
                     "runtime transaction owns CUDA stochastic selection");
    {
        const float invalid[4] = {0.0f, 1.0f, NAN, 3.0f};
        YVEX_TEST_ASSERT(
            yvex_backend_tensor_write(
                backend, device_logits, invalid, sizeof(invalid), &err) == YVEX_OK &&
                operations->select_stochastic(
                    backend, device_logits, 4ull, &policy, 0u,
                    &result, &facts, &err) == YVEX_ERR_FORMAT &&
                !result.completed,
            "device stochastic selection refuses non-finite logits without publication");
    }
    yvex_backend_workspace_detach(backend);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &workspace, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &device_logits, &err) == YVEX_OK,
        "release bounded stochastic sampling tensors");
    if (sampling_full_vocabulary(backend, operations)) return 1;
    sampling_tensor_desc(&descriptor, "sampling-sync-fault", YVEX_DTYPE_F32,
                         sizeof(logits));
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &device_logits, &err) == YVEX_OK &&
            yvex_backend_tensor_write(
                backend, device_logits, logits, sizeof(logits), &err) == YVEX_OK,
        "prepare CUDA selection completion fault fixture");
    {
        unsigned int token = UINT_MAX;
        unsigned long long ties = 0ull;
        float selected = 0.0f;
        YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_SYNC_FAILURE", "encoded-attention", 1) == 0,
                         "inject CUDA selection completion failure");
        rc = operations->select_greedy_rows(
            backend, device_logits, 1ull, 4ull, &token, &selected, &ties, &facts, &err);
        YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_SYNC_FAILURE") == 0,
                         "clear CUDA selection completion failure");
        YVEX_TEST_ASSERT(
            rc == YVEX_ERR_BACKEND &&
                strstr(yvex_error_message(&err), "synchronization failure") != NULL,
            "CUDA selection fails closed when stream completion fails");
        YVEX_TEST_ASSERT(
            facts.kernel_launches == 0ull && facts.d2h_bytes == 0ull,
            "failed stream completion publishes no physical facts");
    }
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &device_logits, &err) == YVEX_OK,
        "release CUDA selection completion fault fixture");
    YVEX_TEST_ASSERT(
        yvex_backend_close_checked(&backend, &err) == YVEX_OK && !backend,
        "release stochastic sampling backend ownership");
    return 0;
}
