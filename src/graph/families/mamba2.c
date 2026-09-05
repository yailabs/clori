/* Source-only Mamba2 promotion gate. No executable descriptor is published by an audit. */
#include <yvex/internal/family_catalog.h>
#include <yvex/internal/compiler.h>
#include <yvex/internal/families/mamba2.h>
#include <yvex/internal/source_catalog.h>

#include <string.h>

static int mamba_source_compile(yvex_family_source_products *out,
    const yvex_compilation_runtime_binding_request *request, yvex_error *err)
{
    const yvex_mamba2_api *family = yvex_model_register_mamba2();
    yvex_source_verify_options options = {0};
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_mamba2_architecture architecture;
    yvex_mamba2_inventory inventory;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    if (!out || !request || !request->source_path || !request->models_root) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "mamba2.source-gate", "exact source is required");
        return YVEX_ERR_INVALID_ARG;
    }
    options.identity = yvex_source_target_identity_find(YVEX_MAMBA2_TARGET);
    options.source_path = request->source_path;
    options.models_root = request->models_root;
    options.manifest_path = request->source_manifest_path;
    options.promote_manifest = 1;
    rc = yvex_source_verify_with_snapshot(&options, &verification, &snapshot, err);
    if (rc == YVEX_OK) rc = family->open(&verification, &architecture, err);
    if (rc == YVEX_OK) rc = family->snapshot_audit(&architecture, snapshot, &inventory, err);
    yvex_source_tensor_snapshot_release(snapshot);
    if (rc != YVEX_OK) return rc;
    yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "mamba2.source-gate",
        "Mamba2 source roles complete (%llu tensors, %llu bytes); "
        "normalization_authority_conflict=%d token_authority_conflict=%d; "
        "no admitted complete-artifact/SSD-decoder binding; source inspection is not READY",
        inventory.tensors, inventory.tensor_bytes, architecture.normalization_policy_conflict,
        architecture.token_policy_conflict);
    return YVEX_ERR_UNSUPPORTED;
}

static const yvex_family_source_adapter *mamba_source(void)
{
    static const yvex_family_source_adapter adapter = {
        .schema_version = YVEX_FAMILY_SOURCE_ADAPTER_SCHEMA_V1,
        .target_id = YVEX_MAMBA2_TARGET, .family = "mamba2",
        .tokenizer_architecture = "mamba2", .tokenizer_pre = "sentencepiece",
        .compile = mamba_source_compile};
    return &adapter;
}

const yvex_family_descriptor yvex_graph_family_descriptor_mamba2 = {
    .schema_version = YVEX_FAMILY_DESCRIPTOR_SCHEMA_V1,
    .target_id = YVEX_MAMBA2_TARGET, .family = "mamba2",
    .tokenizer_architecture = "mamba2", .tokenizer_pre = "sentencepiece",
    .source = mamba_source};
