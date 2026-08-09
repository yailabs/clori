/* Instantiate authenticated binding records without reopening family compilation. */
#include "src/runtime/private.h"

#include <stdlib.h>
#include <string.h>

int yvex_runtime_binding_policies(
    const yvex_runtime_binding *binding,
    const yvex_transformer_family_policy **transformer,
    const yvex_logits_family_policy **logits,
    const yvex_speculation_family_policy **speculation)
{
    if (transformer) *transformer = NULL;
    if (logits) *logits = NULL;
    if (speculation) *speculation = NULL;
    if (!binding || (!transformer && !logits && !speculation)) return 0;
    if (transformer) *transformer = &binding->transformer_policy;
    if (logits) *logits = &binding->logits_policy;
    if (speculation) *speculation = &binding->speculation_policy;
    return 1;
}

int yvex_runtime_binding_import_materialization(
    const yvex_runtime_binding *binding, const yvex_artifact *artifact,
    const yvex_materialization_options *options, yvex_materialization_plan **plan_out,
    yvex_materialization_session **session_out, yvex_runtime_binding_failure *failure,
    yvex_error *err)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_snapshot snapshot = {0};
    yvex_materialization_failure material_failure = {0};
    int rc;
    if (plan_out) *plan_out = NULL;
    if (session_out) *session_out = NULL;
    if (!binding || !artifact || !plan_out || !session_out)
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT,
            "materialization-import", NULL, 0ull, 1ull, 0ull,
            YVEX_ERR_INVALID_ARG,
            "runtime binding materialization import arguments are required", err);
    if (yvex_artifact_snapshot_get(artifact, &snapshot, err) != YVEX_OK ||
        yvex_artifact_snapshot_validate(artifact, NULL, err) != YVEX_OK ||
        snapshot.size != binding->admission.file_bytes)
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_ARTIFACT, "artifact-snapshot", NULL,
            0ull, binding->admission.file_bytes, snapshot.size, YVEX_ERR_STATE,
            "runtime binding artifact snapshot is stale or mismatched", err);
    admission = binding->admission;
    admission.file_snapshot = snapshot;
    yvex_core_text_copy(admission.artifact_path, sizeof(admission.artifact_path),
                        yvex_artifact_path(artifact));
    rc = yvex_materialization_plan_import(
        plan_out, &admission, &binding->materialization, binding->materialized,
        binding->summary.tensor_count, &material_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(session_out, *plan_out, artifact, options,
                                               &material_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(*session_out, &material_failure, err);
    if (rc != YVEX_OK) {
        yvex_materialization_session_close(*session_out);
        yvex_materialization_plan_close(*plan_out);
        *session_out = NULL;
        *plan_out = NULL;
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_MATERIALIZATION,
            yvex_materialization_failure_name(material_failure.code), NULL,
            material_failure.tensor_index, material_failure.expected,
            material_failure.actual, (yvex_status)rc,
            "runtime binding materialization import was refused", err);
    }
    if (failure) memset(failure, 0, sizeof(*failure));
    return YVEX_OK;
}

int yvex_runtime_binding_import_graph(
    const yvex_runtime_binding *binding, const yvex_materialization_session *session,
    yvex_runtime_descriptor **descriptor_out, yvex_attention_plan **attention_out,
    yvex_attention_plan **draft_attention_out,
    const yvex_physical_execution_ir **physical_execution_out,
    yvex_runtime_binding_failure *failure, yvex_error *err)
{
    yvex_runtime_descriptor_failure descriptor_failure = {0};
    yvex_attention_failure attention_failure = {0};
    yvex_runtime_descriptor *descriptor = NULL;
    yvex_attention_plan *attention = NULL, *draft_attention = NULL;
    int rc;
    if (descriptor_out) *descriptor_out = NULL;
    if (attention_out) *attention_out = NULL;
    if (draft_attention_out) *draft_attention_out = NULL;
    if (physical_execution_out) *physical_execution_out = NULL;
    if (!binding || !session || !descriptor_out || !attention_out ||
        !draft_attention_out || !physical_execution_out)
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT,
            "runtime-graph-import", NULL, 0ull, 1ull, 0ull,
            YVEX_ERR_INVALID_ARG,
            "runtime binding graph import arguments are required", err);
    rc = yvex_runtime_descriptor_import(
        &descriptor, &binding->descriptor, binding->runtime,
        binding->summary.tensor_count, session, &descriptor_failure, err);
    if (rc != YVEX_OK)
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_DESCRIPTOR,
            yvex_runtime_descriptor_failure_name(descriptor_failure.code), NULL,
            descriptor_failure.tensor_index, descriptor_failure.expected,
            descriptor_failure.actual, (yvex_status)rc,
            "runtime binding descriptor import was refused", err);
    rc = yvex_attention_plan_import(
        &attention, &binding->attention, binding->layers,
        binding->summary.layer_count, session, descriptor, &attention_failure, err);
    if (rc == YVEX_OK && binding->summary.draft_layer_count)
        rc = yvex_attention_plan_import(
            &draft_attention, &binding->draft_attention, binding->draft_layers,
            binding->summary.draft_layer_count, session, descriptor,
            &attention_failure, err);
    if (rc != YVEX_OK) {
        yvex_attention_plan_close(draft_attention);
        yvex_attention_plan_close(attention);
        yvex_runtime_descriptor_close(descriptor);
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_ATTENTION,
            attention_failure.reason ? attention_failure.reason : "attention", NULL,
            attention_failure.layer_index, attention_failure.expected,
            attention_failure.actual, (yvex_status)rc,
            "runtime binding attention import was refused", err);
    }
    *descriptor_out = descriptor;
    *attention_out = attention;
    *draft_attention_out = draft_attention;
    *physical_execution_out = binding->physical_execution;
    if (failure) memset(failure, 0, sizeof(*failure));
    return YVEX_OK;
}
