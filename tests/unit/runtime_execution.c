#include "tests/test.h"

#include <string.h>

#include <yvex/internal/backend.h>
#include <yvex/internal/execution.h>

static void execution_test_identity(char output[YVEX_SHA256_HEX_CAP], char digit)
{
    memset(output, digit, YVEX_SHA256_HEX_CAP - 1u);
    output[YVEX_SHA256_HEX_CAP - 1u] = '\0';
}

static int execution_test_profile(void)
{
    char identity[YVEX_SHA256_HEX_CAP];
    yvex_compiled_execution_profile_request request = {0};
    yvex_compiled_execution_profile first, second;
    yvex_error err;

    execution_test_identity(identity, 'a');
    request.schema_version = YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V1;
    request.logical_model_identity = identity;
    request.physical_variant_identity = identity;
    request.physical_execution_identity = identity;
    request.artifact_identity = identity;
    request.materialization_identity = identity;
    request.runtime_binding_identity = identity;
    request.kernel_bundle_identity = identity;
    request.hardware_profile = "portable-cpu";
    request.backend = YVEX_BACKEND_KIND_CPU;
    request.context_capacity = 4096ull;
    request.generation_mode = YVEX_EXECUTION_GENERATION_TARGET_ONLY;
    request.workload = YVEX_EXECUTION_WORKLOAD_INTERACTIVE;
    request.evidence = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    request.execution_class = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    request.host_stochastic_reference = 1;
    request.token_local_moe_reference = 1;
    request.eager_attention_reference = 1;
    YVEX_TEST_ASSERT(yvex_compiled_execution_profile_seal(
                         &request, &first, &err) == YVEX_OK,
                     "compiled execution profile should seal");
    YVEX_TEST_ASSERT(yvex_compiled_execution_profile_seal(
                         &request, &second, &err) == YVEX_OK,
                     "equal compiled execution profile should seal");
    YVEX_TEST_ASSERT(strcmp(first.identity, second.identity) == 0,
                     "compiled execution identity should be deterministic");
    request.evidence = YVEX_EXECUTION_EVIDENCE_FORENSIC;
    YVEX_TEST_ASSERT(yvex_compiled_execution_profile_seal(
                         &request, &second, &err) == YVEX_OK &&
                         strcmp(first.identity, second.identity) != 0,
                     "evidence profile should change execution identity");
    return 0;
}

static int execution_test_shape(void)
{
    yvex_execution_shape_registry *registry = NULL;
    yvex_execution_shape configured = {0}, required;
    yvex_execution_shape_failure failure;
    yvex_execution_shape_registry_summary summary;
    const yvex_execution_shape *selected = NULL;
    yvex_error err;

    configured.schema_version = YVEX_EXECUTION_SHAPE_SCHEMA_V1;
    configured.target_scope = YVEX_EXECUTION_SCOPE_TARGET;
    configured.phase = YVEX_EXECUTION_PHASE_VERIFY;
    configured.operation_scope = YVEX_EXECUTION_OPERATION_ENVELOPE;
    configured.token_width = 5ull;
    configured.candidate_visible = 1;
    configured.context_band = YVEX_EXECUTION_CONTEXT_SHORT;
    configured.context_capacity = 32ull;
    configured.local_capacity = 16ull;
    configured.compressed_capacity = 8ull;
    configured.indexer_capacity = 8ull;
    configured.rolling_capacity = 2ull;
    configured.candidate_capacity = 6ull;
    configured.workspace_generation = 1ull;
    configured.evidence = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    execution_test_identity(configured.execution_profile_identity, '1');
    execution_test_identity(configured.attention_plan_identity, '2');
    execution_test_identity(configured.state_layout_identity, '3');
    execution_test_identity(configured.kernel_bundle_identity, '4');
    execution_test_identity(configured.workspace_identity, '5');
    YVEX_TEST_ASSERT(yvex_execution_shape_seal(&configured, &err) == YVEX_OK,
                     "execution shape should seal");
    YVEX_TEST_ASSERT(yvex_execution_shape_registry_open(
                         &registry, 2ull, &err) == YVEX_OK,
                     "shape registry should open");
    YVEX_TEST_ASSERT(yvex_execution_shape_registry_register(
                         registry, &configured, &err) == YVEX_OK,
                     "execution shape should register");
    required = configured;
    required.position = 10ull;
    required.local_capacity = 12ull;
    YVEX_TEST_ASSERT(yvex_execution_shape_seal(&required, &err) == YVEX_OK,
                     "shape requirement should seal");
    YVEX_TEST_ASSERT(yvex_execution_shape_registry_select(
                         registry, &required, &selected, &failure, &err) == YVEX_OK &&
                         selected != NULL,
                     "compatible execution shape should select");
    required.local_capacity = 17ull;
    YVEX_TEST_ASSERT(yvex_execution_shape_seal(&required, &err) == YVEX_OK,
                     "oversized shape requirement should seal");
    YVEX_TEST_ASSERT(yvex_execution_shape_registry_select(
                         registry, &required, &selected, &failure, &err) ==
                         YVEX_ERR_BOUNDS &&
                         failure.component == YVEX_EXECUTION_CAPACITY_LOCAL &&
                         failure.configured == 16ull && failure.required == 17ull &&
                         failure.position == 10ull,
                     "shape capacity refusal should identify the exact component");
    YVEX_TEST_ASSERT(yvex_execution_shape_registry_summary_copy(
                         registry, &summary, &err) == YVEX_OK &&
                         summary.count == 1ull && summary.hit_count == 1ull &&
                         summary.miss_count == 1ull,
                     "shape registry should publish hit and miss accounting");
    yvex_execution_shape_registry_close(&registry);
    return 0;
}

static int execution_test_device_view(void)
{
    yvex_backend backend = {0};
    yvex_device_tensor tensor = {0};
    yvex_execution_device_view view = {0};
    yvex_error err;

    tensor.owner = &backend;
    tensor.owner_id = 1ull;
    tensor.dtype = YVEX_DTYPE_F32;
    tensor.bytes = 48ull;
    view.schema_version = YVEX_EXECUTION_DEVICE_VIEW_SCHEMA_V1;
    view.kind = YVEX_EXECUTION_DEVICE_LOGITS;
    view.backend = &backend;
    view.tensor = &tensor;
    view.element_offset = 4ull;
    view.model_generation = 1ull;
    view.session_generation = 1ull;
    view.state_generation = 1ull;
    view.rows = 2ull;
    view.columns = 4ull;
    view.element_bytes = 4ull;
    view.dtype = YVEX_DTYPE_F32;
    view.materialization = YVEX_EXECUTION_MATERIALIZE_NONE;
    execution_test_identity(view.runtime_model_identity, '6');
    execution_test_identity(view.execution_profile_identity, '7');
    YVEX_TEST_ASSERT(yvex_execution_device_view_validate(&view, &err) == YVEX_OK,
                     "exact device view should validate");
    view.columns = 5ull;
    YVEX_TEST_ASSERT(yvex_execution_device_view_validate(&view, &err) ==
                         YVEX_ERR_FORMAT,
                     "device view with a mismatched extent should refuse");
    return 0;
}

int yvex_test_runtime_execution(void)
{
    if (execution_test_profile() != 0) return 1;
    if (execution_test_shape() != 0) return 1;
    return execution_test_device_view();
}
