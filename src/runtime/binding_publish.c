/*
 * Publish compiler-owned products through the runtime-binding codec.
 *
 * Runtime consumes only the typed compilation lease and never imports source, quant, or writer
 * planning. The lease cleanup runs on every publication path.
 */
#include <yvex/internal/runtime.h>

#include <string.h>

int yvex_runtime_binding_compile_publish(
    const yvex_family_compiler_adapter *adapter,
    const struct yvex_compilation_runtime_binding_request *request,
    char path[YVEX_PATH_CAP], int *published, yvex_error *err)
{
    yvex_family_compilation_products products = {0};
    yvex_runtime_binding_prepare_request prepare = {0};
    yvex_runtime_binding_prepare_result result = {0};
    yvex_runtime_binding_failure failure = {0};
    yvex_runtime_binding_failure reopen_failure = {0};
    yvex_runtime_binding_summary reopened_summary = {0};
    yvex_runtime_binding *reopened = NULL;
    void *owner = NULL;
    int rc;

    if (path) memset(path, 0, YVEX_PATH_CAP);
    if (published) *published = 0;
    if (!adapter || !request || !path || !published ||
        adapter->schema_version != YVEX_FAMILY_COMPILER_SCHEMA_V2 ||
        !adapter->adapter_id || !adapter->adapter_version || !adapter->binding_pipeline ||
        !adapter->binding_compile) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "runtime.binding.compile-publish",
                       "one exact family compiler adapter and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = adapter->binding_compile(adapter, request, &products, &owner, err);
    if (rc == YVEX_OK &&
        (products.family_adapter_id != adapter->adapter_id ||
         products.family_adapter_version != adapter->adapter_version)) {
        yvex_error_set(err, YVEX_ERR_STATE,
                       "runtime.binding.compile-publish",
                       "compiled binding inputs disagree with the family adapter identity");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK &&
        (!products.admission || !products.physical_compatibility ||
         !products.materialization || !products.runtime_descriptor ||
         !products.operator_graph || !products.attention_plan ||
         !products.physical_execution || !products.compiled_plan ||
         !products.capabilities ||
         !products.transformer_policy || !products.logits_policy ||
         !products.speculation_policy || !products.tokenizer_policy)) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.binding.compile-publish",
                       "compiler products are incomplete");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        prepare.directory = products.directory;
        prepare.admission = products.admission;
        prepare.physical_compatibility = products.physical_compatibility;
        prepare.materialization = products.materialization;
        prepare.runtime_descriptor = products.runtime_descriptor;
        prepare.operator_graph = products.operator_graph;
        prepare.physical_execution = products.physical_execution;
        prepare.compiled_plan = products.compiled_plan;
        prepare.attention_plan = products.attention_plan;
        prepare.draft_attention_plan = products.draft_attention_plan;
        prepare.family_adapter_id = products.family_adapter_id;
        prepare.family_adapter_version = products.family_adapter_version;
        prepare.artifact_format = products.artifact_format;
        prepare.artifact_format_version = products.artifact_format_version;
        prepare.logical_transform_identity = products.logical_transform_identity;
        prepare.capabilities = *products.capabilities;
        prepare.transformer_policy = *products.transformer_policy;
        prepare.logits_policy = *products.logits_policy;
        prepare.speculation_policy = *products.speculation_policy;
        prepare.tokenizer_policy = *products.tokenizer_policy;
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_binding_prepare(&prepare, &result, &failure, err);
    if (rc != YVEX_OK &&
        failure.code == YVEX_RUNTIME_BINDING_FAILURE_CONFLICT &&
        failure.path[0]) {
        rc = yvex_runtime_binding_open(
            &reopened, failure.path, &reopened_summary, NULL,
            &reopen_failure, err);
        if (rc == YVEX_OK) {
            yvex_core_text_copy(result.path, sizeof(result.path), failure.path);
            result.summary = reopened_summary;
            result.published = 0;
        }
    }
    if (rc == YVEX_OK) {
        memcpy(path, result.path, YVEX_PATH_CAP);
        *published = result.published;
    }
    yvex_runtime_binding_close(reopened);
    if (products.release) products.release(owner);
    return rc;
}
