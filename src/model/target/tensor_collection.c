/*
 * Bind canonical family tensor recipes to retained source inventory rows and expose deterministic
 * lookup/accounting to compilation consumers.
 *
 * Lexical collection facts never become role mapping. Exact source coverage belongs to sealed
 * transformation planning rather than a parallel family-specific inventory authority.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/core.h>
#include <yvex/internal/source.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/source.h>

typedef struct {
    const char *status;
    const char *family;
    const char *target;
    const char *source;
    const char *moe_status;
    unsigned long long tensors;
    unsigned long long layers;
    unsigned long long embed;
    unsigned long long attention_q;
    unsigned long long attention_k;
    unsigned long long attention_v;
    unsigned long long attention_o;
    unsigned long long attention_complete;
    unsigned long long mlp_gate;
    unsigned long long mlp_up;
    unsigned long long mlp_down;
    unsigned long long mlp_complete;
    unsigned long long norm;
    unsigned long long head;
    unsigned long long moe;
} collection_audit_facts;

#define COLLECTION_LITERAL(text) \
    { YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u }
#define COLLECTION_STRING(field, format) \
    { YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(collection_audit_facts, field) }
#define COLLECTION_U64(field, format) \
    { YVEX_MODEL_TARGET_ROW_U64, (format), offsetof(collection_audit_facts, field) }

static const yvex_model_target_row_spec collection_audit_rows[] = {
    COLLECTION_STRING(status, "tensor_collection_status: %s"),
    COLLECTION_STRING(family, "tensor_collection_family: %s"),
    COLLECTION_STRING(target, "tensor_collection_target_id: %s"),
    COLLECTION_LITERAL("tensor_collection_stage: header-collection-inventory"),
    COLLECTION_LITERAL("tensor_collection_evidence_basis: header-metadata-only"),
    COLLECTION_STRING(source, "tensor_collection_source_status: %s"),
    COLLECTION_LITERAL("tensor_collection_manifest_status: not-checked"),
    COLLECTION_STRING(source, "tensor_collection_config_status: %s"),
    COLLECTION_STRING(source, "tensor_collection_tokenizer_status: %s"),
    COLLECTION_U64(tensors, "tensor_collection_tensor_count: %llu"),
    COLLECTION_U64(layers, "tensor_collection_layer_count_observed: %llu"),
    COLLECTION_LITERAL("tensor_collection_embedding_status: candidate"),
    COLLECTION_U64(embed, "tensor_collection_embedding_tensor_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_attention_status: candidate"),
    COLLECTION_U64(attention_q, "tensor_collection_attention_q_count: %llu"),
    COLLECTION_U64(attention_k, "tensor_collection_attention_k_count: %llu"),
    COLLECTION_U64(attention_v, "tensor_collection_attention_v_count: %llu"),
    COLLECTION_U64(attention_o, "tensor_collection_attention_o_count: %llu"),
    COLLECTION_U64(attention_complete,
                   "tensor_collection_attention_complete_qkvo_layer_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_mlp_status: candidate"),
    COLLECTION_U64(mlp_gate, "tensor_collection_mlp_gate_count: %llu"),
    COLLECTION_U64(mlp_up, "tensor_collection_mlp_up_count: %llu"),
    COLLECTION_U64(mlp_down, "tensor_collection_mlp_down_count: %llu"),
    COLLECTION_U64(mlp_complete, "tensor_collection_mlp_complete_gud_layer_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_norm_status: candidate"),
    COLLECTION_U64(norm, "tensor_collection_norm_tensor_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_output_head_status: candidate"),
    COLLECTION_U64(head, "tensor_collection_output_head_tensor_count: %llu"),
    COLLECTION_STRING(moe_status, "tensor_collection_moe_status: %s"),
    COLLECTION_U64(moe, "tensor_collection_moe_router_count: %llu"),
    COLLECTION_U64(moe, "tensor_collection_moe_expert_count: %llu"),
    COLLECTION_LITERAL("tensor_collection_tokenizer_collection_status: sidecar-observed"),
    COLLECTION_LITERAL(
        "tensor_collection_kv_runtime_state_status: runtime-state-required-not-implemented"),
    COLLECTION_LITERAL("tensor_collection_validation_status: lexical-and-header-only"),
    COLLECTION_LITERAL("tensor_collection_role_mapping_status: not-implemented"),
    COLLECTION_LITERAL("tensor_collection_runtime_descriptor_status: not-implemented"),
    COLLECTION_LITERAL("tensor_collection_graph_consumer_status: not-implemented")
};

#undef COLLECTION_LITERAL
#undef COLLECTION_STRING
#undef COLLECTION_U64

static void collection_family_facts(const char *family,
                                    const char **top_blocker,
                                    const char **source_blocker)
{
    if (strcmp(family, "gemma") == 0) {
        *top_blocker = "missing-gemma-tensor-role-map";
        *source_blocker = "missing-gemma-source-path";
    } else {
        *top_blocker = "missing-qwen-tensor-role-map";
        *source_blocker = "missing-qwen-source-path";
    }
}

static void collection_audit_common(yvex_model_target_report *report,
                                    const yvex_model_target_request *request,
                                    const char *family,
                                    const yvex_model_target_source_scan *scan,
                                    const char *status)
{
    collection_audit_facts facts = {
        status, family, request->target_id,
        scan->source_present ? "present" : "missing",
        scan->moe == 0 ? "not-observed" : "candidate",
        scan->tensors, scan->layers, scan->embed,
        scan->attn >= 1 ? 1ull : 0ull, scan->attn >= 2 ? 1ull : 0ull,
        scan->attn >= 3 ? 1ull : 0ull, scan->attn >= 4 ? 1ull : 0ull,
        scan->attn >= 4 ? 1ull : 0ull, scan->mlp >= 1 ? 1ull : 0ull,
        scan->mlp >= 2 ? 1ull : 0ull, scan->mlp >= 3 ? 1ull : 0ull,
        scan->mlp >= 3 ? 1ull : 0ull, scan->norm, scan->head, scan->moe
    };

    yvex_model_target_report_project_rows(
        report, collection_audit_rows,
        sizeof(collection_audit_rows) / sizeof(collection_audit_rows[0]), &facts);
    yvex_model_target_report_common_tail(report);
    yvex_model_target_report_add_row(report, "next_required_rows: V010.MAP.8");
}

int yvex_tensor_collection_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const char *family;
    const char *top_blocker;
    const char *source_blocker;
    yvex_model_target_source_scan scan;
    const char *status;

    if (!request || !report ||
        request->kind != YVEX_MODEL_TARGET_COMMAND_TENSOR_COLLECTION) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tensor_collection",
                       "tensor collection report requires tensor-collection command kind");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_model_target_validate_supported(
            request, report, "tensor-collection", 0)) {
        return YVEX_OK;
    }
    if (yvex_source_is_release_target(request->target_id)) {
        return yvex_model_target_report_release_coverage(
            request, report, "tensor-collection", "tensor_collection",
            "exact-source-tensor-covered",
            "exact source coverage consumed by the canonical map; payload, artifact, "
            "runtime, and generation remain blocked",
            err);
    }
    family = yvex_model_target_family_key(request->target_id);
    collection_family_facts(family, &top_blocker, &source_blocker);
    yvex_model_target_scan_source(request, family, &scan);
    status = scan.source_present ? "collection-profiled" : "source-missing";
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report, "TENSOR COLLECTION INVENTORY");
        yvex_model_target_report_add_row(
            report, "FAMILY  TARGET  STATUS  EMBED  ATTN_QKVO  MLP_GUD  NORM  "
                    "HEAD  MOE  LAYERS  NEXT");
        yvex_model_target_report_add_row(report, "%s  %s  %s  %llu  %llu  %llu  %llu  %llu  %llu  %llu  V010.MAP.8",
                                         family, request->target_id, status,
                                         scan.embed, scan.attn >= 4 ? 1ull : 0ull,
                                         scan.mlp >= 3 ? 1ull : 0ull,
                                         scan.norm, scan.head, scan.moe,
                                         scan.layers);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        collection_audit_common(report, request, family, &scan, status);
        return YVEX_OK;
    }
    yvex_model_target_report_add_row(report, "tensor-collection: %s", family);
    yvex_model_target_report_add_row(report, "target: %s", request->target_id);
    yvex_model_target_report_add_row(report, "status: %s", status);
    yvex_model_target_report_add_row(report, "stage: header-collection-inventory");
    yvex_model_target_report_add_row(report, "evidence: header-metadata-only");
    yvex_model_target_report_add_row(report,
                                     "collections: embedding=%llu attention_qkvo=%llu "
                                     "mlp_gud=%llu norm=%llu head=%llu moe=%llu",
                                     scan.embed, scan.attn >= 4 ? 1ull : 0ull,
                                     scan.mlp >= 3 ? 1ull : 0ull, scan.norm,
                                     scan.head, scan.moe);
    yvex_model_target_report_add_row(report, "layers_observed: %llu", scan.layers);
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     scan.source_present ? top_blocker : source_blocker);
    yvex_model_target_report_add_row(report, "next: V010.MAP.8");
    yvex_model_target_report_add_row(
        report, "boundary: tensor collection inventory only; no role "
                "mapping/runtime/generation");
    return YVEX_OK;
}
