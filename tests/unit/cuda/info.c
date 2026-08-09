/*
 * Exercises the CUDA backend CUDA device probe opens a real CUDA backend when the local
 * driver/device is available. Returns 77 when CUDA is unavailable.
 */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <yvex/api.h>
#include <yvex/qtype.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/moe.h>
#include "tests/test.h"

static int assert_supported_variant(const yvex_backend *backend,
                                    yvex_backend_operation_variant variant)
{
    yvex_backend_capability_result result;
    yvex_error err;
    int rc = yvex_backend_query_capability(backend, variant, &result, &err);

    YVEX_TEST_ASSERT(rc == YVEX_OK, "query exact CUDA capability");
    YVEX_TEST_ASSERT(result.state == YVEX_BACKEND_CAPABILITY_SUPPORTED,
                     "exact CUDA variant supported");
    YVEX_TEST_ASSERT(result.reason == YVEX_BACKEND_CAPABILITY_REASON_NONE,
                     "supported CUDA variant has no refusal reason");
    YVEX_TEST_ASSERT(result.context_available, "CUDA variant has context");
    if (variant >= YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) {
        YVEX_TEST_ASSERT(result.kernel_bundle_available,
                         "kernel CUDA variant has admitted generated bundle");
        YVEX_TEST_ASSERT(result.function_available,
                         "kernel CUDA variant has resolved function");
    }
    return 0;
}

/* Contract: proves atomic bundle rejection, cleared handles, and clean retry. */
static int assert_bundle_rollback(const char *failure,
                                  yvex_backend_capability_reason expected_reason,
                                  yvex_backend_operation_variant variant)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_backend_capability_result result;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_BUNDLE_FAILURE", failure, 1) == 0,
                     "set bundle failure injection");
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "context survives rejected kernel bundle");
    YVEX_TEST_ASSERT(yvex_backend_status_of(backend) == YVEX_BACKEND_STATUS_CONTEXT_READY,
                     "rejected bundle leaves context-only status");
    rc = yvex_backend_query_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_ALLOC,
                                       &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     result.state == YVEX_BACKEND_CAPABILITY_SUPPORTED,
                     "Driver memory capability survives bundle rejection");
    rc = yvex_backend_query_capability(backend, variant, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "query rejected kernel capability");
    YVEX_TEST_ASSERT(result.state == YVEX_BACKEND_CAPABILITY_FAILED,
                     "rejected bundle cannot support kernel variant");
    YVEX_TEST_ASSERT(result.reason == expected_reason,
                     "rejected bundle exposes typed reason");
    YVEX_TEST_ASSERT(!result.kernel_bundle_available && !result.function_available,
                     "rejected bundle leaves no admitted handle");
    yvex_backend_close(backend);
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_BUNDLE_FAILURE") == 0,
                     "clear bundle failure injection");
    return 0;
}

typedef struct {
    float *arena;
    unsigned long long used, device_base;
} moe_fixture_weights;

static void moe_fixture_weight(moe_fixture_weights *fixture,
                               yvex_moe_weight_view *view, yvex_tensor_role role,
                               unsigned long long rows, unsigned long long width,
                               const float *values)
{
    unsigned long long count = rows * width;
    float *destination = fixture->arena + fixture->used;
    memcpy(destination, values, (size_t)count * sizeof(*values));
    memset(view, 0, sizeof(*view));
    view->tensor_id = fixture->used + 1ull;
    view->expert_index = YVEX_MOE_NO_TENSOR;
    view->role = role;
    view->qtype = YVEX_GGUF_QTYPE_F32;
    view->encoded = (const unsigned char *)destination;
    view->encoded_bytes = (size_t)count * sizeof(*values);
    view->row_bytes = width * sizeof(*values);
    view->row_width = width;
    view->row_count = rows;
    view->device_address = fixture->device_base + fixture->used * sizeof(*values);
    fixture->used += count;
}

static yvex_moe_weight_view moe_fixture_expert(const yvex_moe_weight_view *aggregate,
                                               unsigned long long expert,
                                               unsigned long long rows)
{
    yvex_moe_weight_view view = *aggregate;
    unsigned long long bytes = rows * aggregate->row_bytes;
    view.expert_index = expert;
    view.encoded += expert * bytes;
    view.device_address += expert * bytes;
    view.encoded_bytes = (size_t)bytes;
    view.row_count = rows;
    return view;
}

static void moe_fixture_result(yvex_moe_layer_result *result, float combined[2],
                               float routed[2], float shared[2], float post[1],
                               float combination[1])
{
    memset(result, 0, sizeof(*result));
    result->combined_output = combined;
    result->combined_capacity = 2ull;
    result->routed_output = routed;
    result->routed_capacity = 2ull;
    result->shared_output = shared;
    result->shared_capacity = 2ull;
    result->post = post;
    result->post_capacity = 1ull;
    result->combination = combination;
    result->combination_capacity = 1ull;
}

static int assert_grouped_moe(yvex_backend *backend)
{
    static const float mhc_function[] = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f};
    static const float three_ones[] = {1.0f, 1.0f, 1.0f};
    static const float three_zeroes[] = {0.0f, 0.0f, 0.0f};
    static const float identity[] = {1.0f, 0.0f, 0.0f, 1.0f};
    static const float routed[] = {1.0f, 0.0f, 0.0f, 1.0f,
                                   0.0f, 1.0f, 1.0f, 0.0f};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *anchor = NULL, *input = NULL, *normal_output = NULL;
    yvex_device_tensor *audit_output = NULL, *batch_input = NULL;
    yvex_device_tensor *batch_output = NULL, *reference_output = NULL, *workspace = NULL;
    yvex_backend_moe_execution *execution = NULL;
    yvex_moe_layer_plan layer = {0};
    yvex_moe_layer_job job = {0};
    yvex_moe_layer_result normal, audit;
    yvex_moe_weight_view selected[3];
    const yvex_backend_moe_operations *row_operations;
    yvex_moe_row_batch row_batch = {0};
    yvex_moe_row_batch_output row_output = {0};
    yvex_moe_row_batch_result row_result, completion_result;
    yvex_moe_device_completion device_completion = {0};
    moe_fixture_weights fixture = {0};
    yvex_error err;
    float host_input[2] = {1.0f, 0.5f};
    float host_batch[4] = {1.0f, 0.5f, 0.5f, 1.0f};
    float normal_device[2], audit_device[2];
    float batch_device[4], reference_device[4];
    float combined[2], routed_output[2], shared_output[2], post[1], combination[1];
    float batch_combined[4] = {0}, batch_routed[4] = {0}, batch_shared[4] = {0};
    float batch_post[2] = {0}, batch_combination[2] = {0}, batch_weights[2] = {0};
    unsigned long long batch_selected[2] = {0};
    unsigned long long deferred_unique = 0ull;
    unsigned long long address = 0ull, expert, workspace_bytes = 0ull, slot;
    unsigned long long immediate_active_bytes;
    int deferred_status = 0, rc;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.name = "grouped-moe-anchor";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = 128ull * sizeof(*fixture.arena);
    YVEX_TEST_ASSERT(backend->vtable->resident_alloc(
                         backend, &descriptor, &anchor,
                         (unsigned char **)&fixture.arena, &err) == YVEX_OK,
                     "allocate grouped MoE managed fixture");
    memset(fixture.arena, 0, (size_t)descriptor.bytes);
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, (const unsigned char *)fixture.arena,
                         128ull * sizeof(*fixture.arena), anchor, 7ull, &err) == YVEX_OK &&
                         yvex_backend_resident_resolve(
                             backend, (const unsigned char *)fixture.arena,
                             128ull * sizeof(*fixture.arena), &address) ==
                             YVEX_BACKEND_RESIDENT_HIT,
                     "register grouped MoE fixture once");
    fixture.device_base = address;
    layer.schema_version = YVEX_MOE_PLAN_SCHEMA_V1;
    layer.router_class = YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE;
    layer.scoring = YVEX_MOE_SCORING_SQRT_SOFTPLUS;
    layer.topk_policy = YVEX_MOE_TOPK_NOAUX_TC;
    layer.activation = YVEX_MOE_ACTIVATION_SILU;
    layer.hidden_width = layer.expanded_width = 2ull;
    layer.residual_streams = 1ull;
    layer.mhc_mixing_rows = 3ull;
    layer.mhc_sinkhorn_iterations = 1ull;
    layer.routed_experts = 2ull;
    layer.shared_experts = layer.experts_per_token = 1ull;
    layer.expert_intermediate_width = layer.shared_intermediate_width = 2ull;
    layer.correction_bias_width = 2ull;
    layer.rms_epsilon = layer.mhc_epsilon = 0.00001;
    layer.mhc_post_multiplier = layer.routed_scaling_factor = 1.0;
    layer.activation_limit = 10.0;
    layer.requires_correction_bias = layer.normalize_topk_probabilities = 1;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        layer.tensor_ids[slot] = YVEX_MOE_NO_TENSOR;
    job.layer = &layer;
    job.expanded_input = host_input;
    job.token_id_present = 1;
    job.evidence_level = YVEX_ATTENTION_EVIDENCE_SUMMARY;
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_MHC_FUNCTION],
                       YVEX_TENSOR_ROLE_HC_FFN_FUNCTION, 3ull, 2ull, mhc_function);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_MHC_SCALE],
                       YVEX_TENSOR_ROLE_HC_FFN_SCALE, 1ull, 3ull, three_ones);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_MHC_BASE],
                       YVEX_TENSOR_ROLE_HC_FFN_BASE, 1ull, 3ull, three_zeroes);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_FFN_NORM],
                       YVEX_TENSOR_ROLE_FFN_NORM, 1ull, 2ull, three_ones);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTER],
                       YVEX_TENSOR_ROLE_MOE_ROUTER, 2ull, 2ull, identity);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTER_BIAS],
                       YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS, 1ull, 2ull, three_zeroes);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE],
                       YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, 4ull, 2ull, routed);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTED_UP],
                       YVEX_TENSOR_ROLE_MOE_EXPERT_UP, 4ull, 2ull, routed);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTED_DOWN],
                       YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN, 4ull, 2ull, routed);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
                       YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE, 2ull, 2ull, identity);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
                       YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP, 2ull, 2ull, identity);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN],
                       YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN, 2ull, 2ull, identity);
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        if (job.weights[slot].device_address)
            layer.tensor_ids[slot] = job.weights[slot].tensor_id;
    row_operations = yvex_backend_moe_operations_get(backend);
    YVEX_TEST_ASSERT(row_operations &&
                         row_operations->workspace_required(
                             &layer, 2ull, &workspace_bytes, &err) == YVEX_OK &&
                         workspace_bytes != 0ull,
                     "derive grouped MoE workspace from width and layer geometry");
#define ALLOCATE_TENSOR(OWNER_, NAME_, BYTES_)                                             \
    do {                                                                                   \
        memset(&descriptor, 0, sizeof(descriptor));                                        \
        descriptor.name = (NAME_);                                                         \
        descriptor.dtype = YVEX_DTYPE_F32;                                                 \
        descriptor.rank = 1u;                                                              \
        descriptor.dims[0] = (BYTES_) / sizeof(float);                                    \
        descriptor.bytes = (BYTES_);                                                       \
        YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &(OWNER_), &err) == \
                             YVEX_OK,                                                       \
                         "allocate grouped MoE device tensor");                           \
    } while (0)
    ALLOCATE_TENSOR(input, "grouped-moe-input", sizeof(host_input));
    ALLOCATE_TENSOR(normal_output, "grouped-moe-normal", sizeof(host_input));
    ALLOCATE_TENSOR(audit_output, "grouped-moe-audit", sizeof(host_input));
    ALLOCATE_TENSOR(batch_input, "grouped-moe-batch-input", sizeof(host_batch));
    ALLOCATE_TENSOR(batch_output, "grouped-moe-batch-output", sizeof(host_batch));
    ALLOCATE_TENSOR(reference_output, "grouped-moe-reference-output", sizeof(host_batch));
    ALLOCATE_TENSOR(workspace, "grouped-moe-workspace", workspace_bytes);
#undef ALLOCATE_TENSOR
    YVEX_TEST_ASSERT(yvex_backend_tensor_write(
                         backend, input, host_input, sizeof(host_input), &err) == YVEX_OK &&
                         yvex_backend_tensor_write(
                             backend, batch_input, host_batch, sizeof(host_batch), &err) == YVEX_OK &&
                         yvex_backend_workspace_attach(backend, workspace, 1ull, &err) == YVEX_OK,
                     "prepare grouped MoE device input and workspace");
    job.device_input = input;
    job.device_output = normal_output;
    moe_fixture_result(&normal, combined, routed_output, shared_output, post, combination);
    rc = yvex_backend_moe_begin(&execution, backend, &job, &normal, &err);
    if (rc == YVEX_OK)
        rc = yvex_backend_moe_add_expert(
            execution, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
            &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
            &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN], 1.0f, 1, &err);
    if (rc == YVEX_OK) rc = yvex_backend_moe_finish(execution, &normal, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         yvex_backend_moe_close(&execution, &err) == YVEX_OK &&
                         yvex_backend_tensor_read(backend, normal_output, normal_device,
                                                  sizeof(normal_device), &err) == YVEX_OK,
                     "execute grouped direct-address MoE");
    job.evidence_level = YVEX_ATTENTION_EVIDENCE_FULL;
    job.device_output = audit_output;
    moe_fixture_result(&audit, combined, routed_output, shared_output, post, combination);
    rc = yvex_backend_moe_begin(&execution, backend, &job, &audit, &err);
    expert = audit.router.selected_experts[0];
    selected[0] = moe_fixture_expert(
        &job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE], expert, 2ull);
    selected[1] = moe_fixture_expert(
        &job.weights[YVEX_MOE_WEIGHT_ROUTED_UP], expert, 2ull);
    selected[2] = moe_fixture_expert(
        &job.weights[YVEX_MOE_WEIGHT_ROUTED_DOWN], expert, 2ull);
    if (rc == YVEX_OK)
        rc = yvex_backend_moe_add_expert(execution, &selected[0], &selected[1],
                                         &selected[2], audit.router.selected_weights[0],
                                         0, &err);
    if (rc == YVEX_OK)
        rc = yvex_backend_moe_add_expert(
            execution, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
            &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
            &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN], 1.0f, 1, &err);
    if (rc == YVEX_OK) rc = yvex_backend_moe_finish(execution, &audit, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         yvex_backend_moe_close(&execution, &err) == YVEX_OK &&
                         yvex_backend_tensor_read(backend, audit_output, audit_device,
                                                  sizeof(audit_device), &err) == YVEX_OK,
                     "execute audit selected-expert MoE");
    YVEX_TEST_ASSERT(memcmp(normal_device, audit_device, sizeof(normal_device)) == 0 &&
                         normal.router.selected_experts[0] ==
                             audit.router.selected_experts[0] &&
                         normal.memory.activation_bytes ==
                             2ull * layer.expanded_width * sizeof(float) &&
                         normal.memory.temporary_bytes != 0ull &&
                         normal.upload_count == 0ull &&
                         normal.device_to_host_bytes < audit.device_to_host_bytes &&
                         normal.device_synchronizations == 1ull,
                     "grouped and audit MoE agree with reduced movement and synchronization");
    for (slot = 0ull; slot < 2ull; ++slot) {
        yvex_device_tensor input_view, output_view;
        yvex_moe_layer_result reference;
        YVEX_TEST_ASSERT(
            yvex_backend_tensor_f32_subview(batch_input, slot * 2ull, 2ull, &input_view) &&
                yvex_backend_tensor_f32_subview(reference_output, slot * 2ull, 2ull,
                                                &output_view),
            "form one-row MoE reference views");
        job.device_input = &input_view;
        job.device_output = &output_view;
        job.expanded_input = host_batch + slot * 2ull;
        job.evidence_level = YVEX_ATTENTION_EVIDENCE_SUMMARY;
        moe_fixture_result(&reference, combined, routed_output, shared_output, post, combination);
        rc = yvex_backend_moe_begin(&execution, backend, &job, &reference, &err);
        if (rc == YVEX_OK)
            rc = yvex_backend_moe_add_expert(
                execution, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
                &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
                &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN], 1.0f, 1, &err);
        if (rc == YVEX_OK) rc = yvex_backend_moe_finish(execution, &reference, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK &&
                             yvex_backend_moe_close(&execution, &err) == YVEX_OK,
                         "execute one-row MoE reference for width-N comparison");
    }
    memset(&row_batch, 0, sizeof(row_batch));
    row_batch.schema_version = YVEX_MOE_ROW_BATCH_SCHEMA_V1;
    row_batch.row_count = 2ull;
    row_batch.row_width = row_batch.row_stride = 2ull;
    row_batch.expanded_rows = host_batch;
    row_batch.device_rows = batch_input;
    row_batch.device_outputs = batch_output;
    row_batch.token_ids = (const unsigned int[]){7u, 11u};
    row_batch.token_ids_present = 1;
    row_batch.execution_class = YVEX_EXECUTION_CLASS_DEVICE_NATIVE;
    row_output.combined_rows = batch_combined;
    row_output.combined_capacity = 4ull;
    row_output.routed_rows = batch_routed;
    row_output.routed_capacity = 4ull;
    row_output.shared_rows = batch_shared;
    row_output.shared_capacity = 4ull;
    row_output.post_rows = batch_post;
    row_output.post_capacity = 2ull;
    row_output.combination_rows = batch_combination;
    row_output.combination_capacity = 2ull;
    row_output.selected_experts = batch_selected;
    row_output.selected_weights = batch_weights;
    row_output.selection_capacity = 2ull;
    job.device_input = batch_input;
    job.device_output = batch_output;
    job.expanded_input = host_batch;
    YVEX_TEST_ASSERT(!yvex_device_tensor_is_written(batch_output),
                     "width-N MoE output begins unpublished");
    rc = row_operations->execute_rows(
        backend, &job, &row_batch, &row_output, &row_result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && row_result.completed == 1 && row_result.row_count == 2ull &&
            row_result.row_expert_pairs == 2ull && row_result.unique_experts >= 1ull &&
            yvex_device_tensor_is_written(batch_output) &&
            row_result.kernel_launches < 2ull * normal.kernel_launches &&
            row_result.stream_synchronizations == 1ull &&
            row_result.device_synchronizations == 0ull &&
            yvex_backend_tensor_read(backend, batch_output, batch_device,
                                     sizeof(batch_device), &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, reference_output, reference_device,
                                     sizeof(reference_device), &err) == YVEX_OK &&
            memcmp(batch_device, reference_device, sizeof(batch_device)) == 0,
        "width-N MoE equals two one-row oracles with one grouped transaction");
    immediate_active_bytes = row_result.encoded_bytes_read;
    batch_selected[0] = batch_selected[1] = ULLONG_MAX;
    batch_weights[0] = batch_weights[1] = -99.0f;
    device_completion.defer = 1;
    device_completion.host_status = &deferred_status;
    device_completion.host_unique_experts = &deferred_unique;
    job.device_completion = &device_completion;
    batch_output->is_written = 0;
    rc = row_operations->execute_rows(
        backend, &job, &row_batch, &row_output, &row_result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && !row_result.completed &&
            row_result.device_completion_pending &&
            yvex_device_tensor_is_written(batch_output) &&
            !row_result.stream_synchronizations &&
            !row_result.device_synchronizations &&
            row_result.d2h_bytes == sizeof(deferred_status) + sizeof(deferred_unique) &&
            !row_result.memory.complete && row_result.memory.activation_bytes != 0ull &&
            batch_selected[0] == ULLONG_MAX && batch_selected[1] == ULLONG_MAX &&
            batch_weights[0] == -99.0f && batch_weights[1] == -99.0f,
        "deferred width-N MoE keeps routing evidence and completion off the layer path");
    rc = row_operations->complete_rows(
        backend, 2, &completion_result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "deferred MoE refuses an unproved completion barrier");
    rc = row_operations->complete_rows(
        backend, 0, &completion_result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && completion_result.completed && deferred_status == 0 &&
            deferred_unique >= 1ull &&
            completion_result.stream_synchronizations == 1ull &&
            completion_result.device_synchronizations == 0ull &&
            row_result.active_weight_base_bytes +
                    row_result.active_weight_per_unique_expert_bytes * deferred_unique ==
                immediate_active_bytes,
        "one phase completion validates deferred MoE and restores exact active bytes");
    rc = row_operations->complete_rows(backend, 1, &completion_result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && completion_result.completed &&
            !completion_result.stream_synchronizations &&
            !completion_result.device_synchronizations,
        "a proved same-stream barrier adds no redundant MoE synchronization");
    job.device_completion = NULL;
    yvex_backend_workspace_detach(backend);
    YVEX_TEST_ASSERT(yvex_backend_tensor_release(backend, &workspace, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &reference_output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &batch_output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &batch_input, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &audit_output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &normal_output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
                         yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &anchor, &err) == YVEX_OK,
                     "release grouped MoE fixture ownership");
    return 0;
}
int yvex_cuda_test_info(void)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_backend_device_info info;
    yvex_backend_tensor_desc descriptor;
    yvex_device_tensor *resident = NULL;
    yvex_backend_cuda_attention_graph_summary kernel_summary = {0};
    yvex_backend_bandwidth_evidence bandwidth = {0}, repeated = {0};
    const char *required_native = getenv("YVEX_REQUIRE_NATIVE_CUDA_TEST");
    yvex_error err;
    static const char *attention_symbols[] = {
        "yvex_attention_bf16_round",
        "yvex_qtype_matvec",
        "yvex_encoded_row_decode",
        "yvex_attention_weighted_norm",
        "yvex_attention_unit_norm",
        "yvex_attention_yarn_rope",
        "yvex_attention_activation_quantize",
        "yvex_attention_rolling_state",
        "yvex_attention_topk",
        "yvex_attention_reduce"
    };
    size_t symbol_index;
    unsigned char *imported = NULL, *mapped = NULL;
    unsigned long long mapped_address = 0ull;
    int rc;
    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        fprintf(stderr, "SKIP: CUDA unavailable: %s\n", yvex_error_message(&err));
        return 77;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open cuda backend");
    rc = yvex_backend_get_device_info(backend, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "get cuda device info");
    YVEX_TEST_ASSERT(info.kind == YVEX_BACKEND_KIND_CUDA, "device info kind");
    YVEX_TEST_ASSERT(info.name && info.name[0] != '\0', "device name non-empty");
    YVEX_TEST_ASSERT(info.compute_capability_major >= 1, "compute capability major");
    YVEX_TEST_ASSERT(info.global_memory_bytes > 0, "global memory nonzero");
    YVEX_TEST_ASSERT(info.total_memory_bytes > 0, "total memory nonzero");
    YVEX_TEST_ASSERT(yvex_backend_cuda_attention_graph_summary_get(
                         backend, &kernel_summary, &err) == YVEX_OK &&
                         kernel_summary.kernel_bundle_architecture[0] &&
                         yvex_sha256_hex_valid(kernel_summary.cuda_build_identity),
                     "query admitted CUDA kernel image identity");
    if (required_native && required_native[0]) {
        YVEX_TEST_ASSERT(kernel_summary.kernel_bundle_native,
                         "native CUDA validation refuses a PTX-only bundle");
        YVEX_TEST_ASSERT(strcmp(kernel_summary.kernel_bundle_architecture,
                                required_native) == 0,
                         "native CUDA image targets the required architecture");
        fprintf(stderr, "native CUDA kernel bundle: architecture=%s identity=%s\n",
                kernel_summary.kernel_bundle_architecture,
                kernel_summary.cuda_build_identity);
    }
    YVEX_TEST_ASSERT(yvex_backend_bandwidth_probe(backend, &bandwidth, &err) == YVEX_OK &&
                         bandwidth.schema_version ==
                             YVEX_BACKEND_BANDWIDTH_SCHEMA_V1 &&
                         bandwidth.sample_count == YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT &&
                         bandwidth.working_set_bytes > 0ull && bandwidth.iterations > 0ull &&
                         bandwidth.sustainable_read_bytes_per_second > 0ull &&
                         bandwidth.sustainable_copy_bytes_per_second > 0ull &&
                         bandwidth.sustainable_coherent_host_bytes_per_second > 0ull &&
                         yvex_sha256_hex_valid(bandwidth.identity) &&
                         strcmp(bandwidth.kernel_bundle_identity,
                                kernel_summary.cuda_build_identity) == 0,
                     "measure identity-bound CUDA bandwidth evidence");
    YVEX_TEST_ASSERT(yvex_backend_bandwidth_probe(backend, &repeated, &err) == YVEX_OK &&
                         memcmp(&bandwidth, &repeated, sizeof(bandwidth)) == 0,
                     "CUDA backend reuses one immutable bandwidth measurement");

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.name = "cuda-addressable-host-fixture";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = 4096ull;
    YVEX_TEST_ASSERT(posix_memalign((void **)&imported, 4096u, 4096u) == 0,
                     "allocate page-aligned imported host residency");
    memset(imported, 0x3c, 4096u);
    YVEX_TEST_ASSERT(mlock(imported, 4096u) == 0,
                     "lock imported host residency before CUDA registration");
    mapped = imported;
    YVEX_TEST_ASSERT(backend->vtable->resident_alloc(
                         backend, &descriptor, &resident, &mapped, &err) == YVEX_OK &&
                         mapped == imported,
                     "register existing host residency without replacement");
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, mapped, 4096ull, resident, 1ull, &err) == YVEX_OK &&
                         yvex_backend_resident_resolve(
                             backend, mapped, 4096ull, &mapped_address) ==
                             YVEX_BACKEND_RESIDENT_HIT && mapped_address != 0ull,
                     "resolve registered host residency to its CUDA address");
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
                     "unregister imported host residency exactly once");
    YVEX_TEST_ASSERT(munlock(imported, 4096u) == 0,
                     "unlock imported host residency after CUDA unregister");
    free(imported);
    imported = mapped = NULL;
    YVEX_TEST_ASSERT(backend->vtable->resident_alloc(
                         backend, &descriptor, &resident, &mapped, &err) == YVEX_OK,
                     "allocate exact managed residency");
    memset(mapped, 0x5a, 4096u);
    rc = yvex_backend_resident_attach(backend, mapped, 4096ull, resident, 1ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         yvex_backend_resident_resolve(
                             backend, mapped, 4096ull, &mapped_address) ==
                             YVEX_BACKEND_RESIDENT_HIT && mapped_address != 0ull,
                     "attach one direct managed CUDA range");
    rc = yvex_backend_resident_attach(backend, mapped, 4096ull, resident, 1ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE, "duplicate managed attachment refuses");
    rc = yvex_backend_resident_detach(backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
                     "release exact CUDA managed residency");
    mapped = NULL;
    YVEX_TEST_ASSERT(assert_grouped_moe(backend) == 0,
                     "grouped direct-address MoE matches audit execution");
    for (rc = 0; rc < (int)YVEX_BACKEND_VARIANT_COUNT; ++rc) {
        YVEX_TEST_ASSERT(assert_supported_variant(
                             backend, (yvex_backend_operation_variant)rc) == 0,
                         "all advertised CUDA variants are exact");
    }
    {
        const yvex_backend_moe_operations *operations =
            yvex_backend_moe_operations_get(backend);
        yvex_moe_row_batch_result completion = {0};
        YVEX_TEST_ASSERT(
            operations && operations->complete_rows &&
                setenv("YVEX_TEST_CUDA_SYNC_FAILURE", "encoded-attention", 1) == 0,
            "inject deferred MoE completion failure");
        rc = operations->complete_rows(backend, 0, &completion, &err);
        YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_SYNC_FAILURE") == 0,
                         "clear deferred MoE completion failure");
        YVEX_TEST_ASSERT(
            rc == YVEX_ERR_BACKEND && !completion.completed &&
                strstr(yvex_error_message(&err), "synchronization failure") != NULL,
            "deferred MoE completion fails closed before physical facts publish");
    }
    yvex_backend_close(backend);
    YVEX_TEST_ASSERT(assert_bundle_rollback(
                         "module",
                         YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_REJECTED,
                         YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) == 0,
                     "module failure rolls back atomically");
    YVEX_TEST_ASSERT(assert_bundle_rollback(
                         "symbol",
                         YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING,
                         YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) == 0,
                     "symbol failure rolls back atomically");
    for (symbol_index = 0u;
         symbol_index < sizeof(attention_symbols) / sizeof(attention_symbols[0]);
         ++symbol_index) {
        YVEX_TEST_ASSERT(assert_bundle_rollback(
                             attention_symbols[symbol_index],
                             YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING,
                             YVEX_BACKEND_VARIANT_ATTENTION_ENCODED) == 0,
                         "each encoded-attention symbol is atomically required");
    }
    backend = NULL;
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "CUDA bundle admission retries after rollback");
    YVEX_TEST_ASSERT(assert_supported_variant(
                         backend, YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) == 0,
                     "retry resolves canonical function");
    yvex_backend_close(backend);
    return 0;
}
