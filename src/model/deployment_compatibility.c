/* Reconcile durable deployment profiles with the current typed execution registry. */
#define _POSIX_C_SOURCE 200809L
#include <yvex/internal/deployment_compatibility.h>

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/compiler.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/media_target.h>
#include <yvex/internal/runtime.h>
#include <yvex/model.h>

static int compatibility_finish(
    yvex_deployment_compatibility *result,
    yvex_deployment_compatibility_status status, const char *reason,
    yvex_error *err)
{
    result->status = status;
    result->current = status == YVEX_DEPLOYMENT_COMPATIBILITY_CURRENT;
    (void)snprintf(result->reason, sizeof(result->reason), "%s",
                   reason ? reason : "deployment compatibility unavailable");
    yvex_error_clear(err);
    return YVEX_OK;
}

static yvex_deployment_compatibility_status binding_failure_status(
    const yvex_runtime_binding_failure *failure)
{
    if (!failure) return YVEX_DEPLOYMENT_COMPATIBILITY_MALFORMED_BINDING;
    if (failure->code == YVEX_RUNTIME_BINDING_FAILURE_OPEN ||
        failure->code == YVEX_RUNTIME_BINDING_FAILURE_DIRECTORY)
        return YVEX_DEPLOYMENT_COMPATIBILITY_MISSING_DEPENDENCY;
    if (failure->code == YVEX_RUNTIME_BINDING_FAILURE_SCHEMA ||
        failure->code == YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY)
        return YVEX_DEPLOYMENT_COMPATIBILITY_STALE_BINDING;
    return YVEX_DEPLOYMENT_COMPATIBILITY_MALFORMED_BINDING;
}

static int composite_component_path(
    const yvex_model_registry_entry *entry, const char *leaf,
    char path[YVEX_PATH_CAP])
{
    struct stat status;
    int written;
    if (!leaf || !leaf[0]) return 0;
    written = snprintf(path, YVEX_PATH_CAP, "%s/%s",
                       entry->runtime_installation, leaf);
    return written > 0 && written < (int)YVEX_PATH_CAP &&
           lstat(path, &status) == 0 && S_ISREG(status.st_mode) &&
           !S_ISLNK(status.st_mode) && access(path, R_OK) == 0;
}

typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
} compatibility_component_view;

static void compatibility_component_close(compatibility_component_view *view)
{
    if (!view) return;
    yvex_tensor_table_close(view->tensors);
    yvex_gguf_close(view->gguf);
    yvex_artifact_close(view->artifact);
    memset(view, 0, sizeof(*view));
}

static int compatibility_component_open(
    const char *path, compatibility_component_view *view, yvex_error *err)
{
    yvex_artifact_options options = {.path = path, .readonly = 1};
    int rc;

    memset(view, 0, sizeof(*view));
    rc = yvex_artifact_open(&view->artifact, &options, err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&view->gguf, view->artifact, err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&view->tensors, view->gguf, err);
    if (rc != YVEX_OK) compatibility_component_close(view);
    return rc;
}

static int compatibility_composite(
    const yvex_model_registry_entry *entry,
    yvex_deployment_compatibility *result, yvex_error *err)
{
    const yvex_component_variant_adapter *adapter =
        yvex_graph_component_variant_find(entry->runtime_target);
    yvex_media_target_profile profile = {0};
    static const char *const roles[] = {
        "text_encoder", "transformer", "video_vae", "audio_vae"};
    const char *components[4];
    char path[YVEX_PATH_CAP];
    size_t index;
    int rc;

    if (!adapter ||
        adapter->schema_version != YVEX_COMPONENT_VARIANT_ADAPTER_SCHEMA_V2 ||
        !adapter->media_target_profile || !adapter->component_contract ||
        !adapter->media_execution)
        return compatibility_finish(
            result, YVEX_DEPLOYMENT_COMPATIBILITY_UNSUPPORTED_TARGET,
            "current composite execution adapter is unavailable", err);
    rc = adapter->media_target_profile(&profile, err);
    if (rc != YVEX_OK ||
        profile.schema_version != YVEX_MEDIA_TARGET_PROFILE_SCHEMA_V2 ||
        !profile.target || strcmp(profile.target, entry->runtime_target))
        return compatibility_finish(
            result, YVEX_DEPLOYMENT_COMPATIBILITY_MALFORMED_BINDING,
            "current composite deployment profile is malformed", err);
    components[0] = profile.text_artifact;
    components[1] = profile.transformer_artifact;
    components[2] = profile.video_artifact;
    components[3] = profile.audio_artifact;
    for (index = 0u; index < sizeof(components) / sizeof(components[0]); ++index) {
        compatibility_component_view view = {0};
        yvex_complete_artifact_admission admission = {0};
        yvex_artifact_admission_failure failure = {0};

        if (!composite_component_path(entry, components[index], path))
            return compatibility_finish(
                result, YVEX_DEPLOYMENT_COMPATIBILITY_MISSING_DEPENDENCY,
                "a required composite deployment component is unavailable", err);
        const yvex_artifact_catalog_contract *contract =
            adapter->component_contract(roles[index]);

        if (!contract)
            return compatibility_finish(
                result, YVEX_DEPLOYMENT_COMPATIBILITY_MALFORMED_BINDING,
                "current composite component contract is unavailable", err);
        rc = compatibility_component_open(path, &view, err);
        if (rc == YVEX_OK)
            rc = yvex_artifact_catalog_compatible(
                view.artifact, view.gguf, view.tensors, contract,
                &admission, &failure, err);
        compatibility_component_close(&view);
        if (rc != YVEX_OK)
            return compatibility_finish(
                result, YVEX_DEPLOYMENT_COMPATIBILITY_ARTIFACT_MISMATCH,
                "a composite component does not match the current family execution contract",
                err);
        if (!admission.complete ||
            !yvex_sha256_hex_valid(admission.artifact_identity))
            return compatibility_finish(
                result, YVEX_DEPLOYMENT_COMPATIBILITY_MALFORMED_BINDING,
                "current composite component admission published incomplete identity",
                err);
    }
    return compatibility_finish(
        result, YVEX_DEPLOYMENT_COMPATIBILITY_CURRENT,
        "composite components match the current family execution contract", err);
}

static int compatibility_single(
    const yvex_model_registry_entry *entry,
    yvex_deployment_compatibility *result, yvex_error *err)
{
    const yvex_graph_execution_binding *execution =
        yvex_graph_execution_find(0ull, 0ull, entry->runtime_target);
    yvex_runtime_binding_summary summary = {0};
    yvex_runtime_binding_failure failure = {0};
    yvex_runtime_binding *binding = NULL;
    char capability_identity[YVEX_SHA256_HEX_CAP];
    int rc;

    /* A registered family supplies independent current semantic authority.
     * A portable compiled binding has no family compiler to consult, so its
     * current generic binding schema is the strongest available authority. */
    if (execution &&
        (execution->schema_version != YVEX_GRAPH_EXECUTION_BINDING_SCHEMA_V1 ||
         !execution->compiler))
        return compatibility_finish(
            result, YVEX_DEPLOYMENT_COMPATIBILITY_UNSUPPORTED_TARGET,
            "current execution adapter is incomplete for this target", err);
    if (execution) {
        result->family_adapter_id = execution->adapter_id;
        result->family_adapter_version = execution->adapter_version;
    }
    rc = yvex_runtime_binding_open_compatible(
        &binding, entry->runtime_binding,
        execution ? execution->adapter_id : 0ull,
        execution ? execution->adapter_version : 0ull,
        execution ? execution->logical_transform_identity : NULL,
        &summary, NULL, &failure, err);
    if (rc != YVEX_OK) {
        const char *reason = failure.code == YVEX_RUNTIME_BINDING_FAILURE_SCHEMA
                                 ? "runtime binding predates the current canonical schema"
                                 : yvex_error_message(err);
        yvex_runtime_binding_close(binding);
        return compatibility_finish(result, binding_failure_status(&failure),
                                    reason, err);
    }
    if (!entry->sha256 || !yvex_sha256_hex_valid(entry->sha256)) {
        yvex_runtime_binding_close(binding);
        return compatibility_finish(
            result, YVEX_DEPLOYMENT_COMPATIBILITY_INCOMPLETE,
            "deployment profile does not name an immutable artifact identity",
            err);
    }
    if (strcmp(summary.artifact_identity, entry->sha256)) {
        yvex_runtime_binding_close(binding);
        return compatibility_finish(
            result, YVEX_DEPLOYMENT_COMPATIBILITY_ARTIFACT_MISMATCH,
            "runtime binding names a different immutable artifact", err);
    }
    if (!yvex_runtime_capabilities_identity(&summary.capabilities,
                                             capability_identity) ||
        strcmp(capability_identity, summary.execution_capability_identity)) {
        yvex_runtime_binding_close(binding);
        return compatibility_finish(
            result, YVEX_DEPLOYMENT_COMPATIBILITY_MALFORMED_BINDING,
            "runtime binding capability contract is malformed", err);
    }
    result->family_adapter_id = summary.family_adapter_id;
    result->family_adapter_version = summary.family_adapter_version;
    (void)snprintf(result->runtime_binding_identity,
                   sizeof(result->runtime_binding_identity), "%s", summary.identity);
    (void)snprintf(result->artifact_identity,
                   sizeof(result->artifact_identity), "%s", summary.artifact_identity);
    (void)snprintf(result->semantic_graph_identity,
                   sizeof(result->semantic_graph_identity), "%s",
                   summary.semantic_graph_identity);
    (void)snprintf(result->executable_graph_identity,
                   sizeof(result->executable_graph_identity), "%s",
                   summary.executable_graph_identity);
    yvex_runtime_binding_close(binding);
    return compatibility_finish(
        result, YVEX_DEPLOYMENT_COMPATIBILITY_CURRENT,
        execution
            ? "runtime binding matches the current semantic execution adapter"
            : "portable runtime binding matches the current generic execution schema",
        err);
}

int yvex_deployment_compatibility_evaluate(
    const yvex_model_registry_entry *entry,
    yvex_deployment_compatibility *result, yvex_error *err)
{
    yvex_error structural;
    int rc;
    if (!result) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "deployment.compatibility",
                       "compatibility result is required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(result, 0, sizeof(*result));
    result->schema_version = YVEX_DEPLOYMENT_COMPATIBILITY_SCHEMA_V1;
    yvex_error_clear(&structural);
    rc = yvex_model_registry_startup_validate(entry, &structural);
    if (rc != YVEX_OK)
        return compatibility_finish(
            result,
            rc == YVEX_ERR_IO
                ? YVEX_DEPLOYMENT_COMPATIBILITY_MISSING_DEPENDENCY
                : YVEX_DEPLOYMENT_COMPATIBILITY_INCOMPLETE,
            yvex_error_message(&structural), err);
    if (entry->runtime_profile && !strcmp(entry->runtime_profile, "composite"))
        return compatibility_composite(entry, result, err);
    return compatibility_single(entry, result, err);
}

const char *yvex_deployment_compatibility_status_name(
    yvex_deployment_compatibility_status status)
{
    static const char *const names[] = {
        "current", "incomplete", "missing-dependency", "unsupported-target",
        "stale-binding", "malformed-binding", "artifact-mismatch"};
    return status <= YVEX_DEPLOYMENT_COMPATIBILITY_ARTIFACT_MISMATCH
               ? names[status] : "invalid";
}
