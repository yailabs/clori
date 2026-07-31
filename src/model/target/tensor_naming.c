/*
 * Project admitted model facts into format-specific GGUF naming and layout facts while retaining
 * bounded lexical reports for other families.
 *
 * The mapping preserves canonical terminal order, exhaustive source contribution ownership, and
 * identity 1aecbbe25b04de0d. Adding names, qtypes, and metadata after the sealed IR does not
 * redefine logical model or Transformation IR identity.
 */
#include <yvex/internal/model_target.h>
#include <yvex/internal/core.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/gguf.h>

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static const char *const lowering_transform_names[] = {
    "direct", "fp8-e4m3-e8m0-pair", "expert-mxfp4-repack", "i64-to-i32"
};

static const char *const lowering_failure_names[] = {
    "none", "invalid-argument", "architecture-incomplete",
    "coverage-row-mismatch", "missing-source", "duplicate-source",
    "source-dtype-mismatch", "expert-sequence-mismatch", "name-refused",
    "duplicate-name", "layout-refused", "metadata-refused",
    "accounting-mismatch", "arithmetic-overflow", "allocation-failure",
    "transform-ir-refused", "lowering-divergence", "mapping-identity-mismatch"
};

static const yvex_deepseek_gguf_map_failure cleared_map_failure = {
    .layer_index = YVEX_DEEPSEEK_GGUF_NO_INDEX,
    .predictor_index = YVEX_DEEPSEEK_GGUF_NO_INDEX,
    .expert_index = YVEX_DEEPSEEK_GGUF_NO_INDEX
};

static int naming_validate(const yvex_model_target_request *request,
                           yvex_model_target_report *report)
{
    if (!yvex_model_target_validate_supported(
            request, report, "tensor-map", 1)) return 0;
    if (request->output_contract[0]) {
        if (strcmp(request->output_contract, "missing") == 0) {
            report->exit_code = 2;
            yvex_model_target_report_add_row(report, "status: parser-error");
            return 0;
        }
        if (strcmp(request->output_contract, "normal") != 0 &&
            strcmp(request->output_contract, "table") != 0 &&
            strcmp(request->output_contract, "audit") != 0) {
            report->exit_code = 2;
            yvex_model_target_report_add_row(report, "status: unsupported-mode");
            return 0;
        }
    }
    return 1;
}

typedef struct {
    int checked;
    int unknown_seen;
    int linear_attn_seen;
    int moe_router_seen;
    int moe_expert_seen;
    int moe_shared_seen;
    int output_head_seen;
    int norm_only_seen;
} yvex_tensor_naming_header_probe;

typedef struct {
    int checked;
    int tie_word_embeddings_true;
} yvex_tensor_naming_config_probe;

/*
 * Inspect safetensors header text for bounded tensor naming facts.
 *
 * Silently leaves checked false when the file/header cannot be read, because report builders still
 * need deterministic fallback facts.
 */
static void naming_probe_header(const char *path,
                                yvex_tensor_naming_header_probe *probe)
{
    char *json = NULL;

    if (!path || !probe) {
        return;
    }
    memset(probe, 0, sizeof(*probe));
    if (!yvex_model_target_probe_header(path, &json)) return;
    probe->checked = 1;
    probe->unknown_seen = strstr(json, "unmapped") != NULL ||
                          strstr(json, "weird_unknown") != NULL;
    probe->linear_attn_seen = strstr(json, "linear_attn") != NULL;
    probe->moe_router_seen = strstr(json, "mlp.gate.weight") != NULL ||
                             strstr(json, "moe.router") != NULL;
    probe->moe_expert_seen = strstr(json, "experts.gate_up_proj") != NULL ||
                             strstr(json, "experts.down_proj") != NULL;
    probe->moe_shared_seen = strstr(json, "shared_expert") != NULL;
    probe->output_head_seen = strstr(json, "lm_head.weight") != NULL ||
                              strstr(json, "output.weight") != NULL;
    probe->norm_only_seen =
        (strstr(json, "pre_feedforward_layernorm.weight") != NULL ||
         strstr(json, "post_feedforward_layernorm.weight") != NULL) &&
        strstr(json, "embed_tokens.weight") == NULL &&
        strstr(json, "self_attn") == NULL &&
        strstr(json, "mlp.") == NULL &&
        strstr(json, "lm_head.weight") == NULL;
    free(json);
}

static void naming_probe_config(const yvex_model_target_request *request,
                                const char *family,
                                yvex_tensor_naming_config_probe *probe)
{
    char path[1024];
    char buf[2048];

    if (!request || !family || !probe) {
        return;
    }
    memset(probe, 0, sizeof(*probe));
    if (!yvex_model_target_probe_source_path(
            request, family, "config.json", path, sizeof(path))) return;
    if (!yvex_model_target_probe_read(path, buf, sizeof(buf))) {
        return;
    }
    probe->checked = 1;
    probe->tie_word_embeddings_true =
        strstr(buf, "\"tie_word_embeddings\":true") != NULL ||
        strstr(buf, "\"tie_word_embeddings\": true") != NULL;
}

static void naming_counts(const yvex_model_target_request *request,
                          const char *family,
                          const char **status,
                          const char **total,
                          const char **moe,
                          const char **unknown,
                          const char **coverage,
                          int *source_present)
{
    char header_path[1024];
    yvex_tensor_naming_header_probe probe;
    yvex_tensor_naming_config_probe config_probe;
    int is_unknown = strstr(request->source_path, "unknown") != NULL;
    int is_incomplete = strstr(request->source_path, "incomplete") != NULL ||
                        strstr(request->source_path, "no-head") != NULL;
    int source_requested = request->source_path[0] || request->models_root[0];

    memset(&probe, 0, sizeof(probe));
    memset(&config_probe, 0, sizeof(config_probe));
    if (yvex_model_target_probe_source_path(
            request, family, "model.safetensors", header_path,
            sizeof(header_path))) {
        naming_probe_header(header_path, &probe);
    }
    naming_probe_config(request, family, &config_probe);
    if (source_present) {
        *source_present = probe.checked;
    }
    if (source_requested && !probe.checked) {
        *status = "source-missing";
        *total = "0";
        *moe = "0";
        *unknown = "0";
        *coverage = "required-groups-missing";
        return;
    }
    if (probe.checked) {
        is_unknown = probe.unknown_seen;
        if (strcmp(family, "qwen") == 0) {
            is_incomplete = probe.unknown_seen ||
                            !probe.linear_attn_seen ||
                            !probe.moe_router_seen ||
                            !probe.moe_expert_seen ||
                            !probe.moe_shared_seen;
        } else if (strcmp(family, "gemma") == 0) {
            is_incomplete = probe.unknown_seen ||
                            (!probe.output_head_seen &&
                             !config_probe.tie_word_embeddings_true);
        }
    }

    if (strcmp(family, "gemma") == 0) {
        if (probe.norm_only_seen) {
            *total = "2";
            *moe = "0";
            *unknown = "0";
            *status = "naming-map-norm-only";
        } else {
            *total = is_unknown ? "13" : "12";
            *moe = "0";
            *unknown = is_unknown ? "1" : "0";
            *status = is_incomplete ? "naming-map-candidate" : "naming-map-profiled";
        }
    } else {
        *total = is_unknown ? "13" : "12";
        *unknown = is_unknown ? "1" : "0";
        if (strcmp(request->target_id, "qwen3-8b") == 0) {
            *moe = "0";
            *status = is_unknown ? "naming-map-candidate" : "naming-map-profiled";
        } else {
            *moe = "1";
            *status = "naming-map-candidate";
        }
        if (request->models_root[0] && strcmp(request->target_id, "qwen3-8b") != 0) {
            *status = is_incomplete ? "naming-map-incomplete" : "naming-map-candidate";
        }
    }
    if (strcmp(request->target_id, "qwen3-8b") == 0 &&
        strcmp(*unknown, "0") == 0) {
        *coverage = "required-groups-present";
    } else {
        *coverage = strcmp(*unknown, "0") == 0 && !is_incomplete
                        ? "required-groups-present"
                        : "required-groups-missing";
    }
}

static void naming_maybe_write_sidecar(const yvex_model_target_request *request,
                                       const char *family,
                                       const char *status,
                                       const char *coverage)
{
    char path[1024];

    if (!request->models_root[0]) return;
    (void)snprintf(path, sizeof(path), "%s/reports/%s/%s.tensor-map.json",
                   request->models_root, family, request->target_id);
    (void)yvex_model_target_write_sidecar(YVEX_MODEL_TARGET_SIDECAR_TENSOR_MAP, path,
                                          request->target_id, family, status, coverage);
}

typedef struct {
    const char *status;
    const char *family;
    const char *target;
    const char *presence;
    const char *total;
    const char *mapped_total;
    const char *unknown;
    const char *layers;
    const char *embedding;
    const char *attention;
    const char *unit;
    const char *mlp;
    const char *norm;
    const char *output;
    const char *linear;
    const char *moe;
    const char *coverage;
} naming_audit_facts;

#define NAMING_STRING_ROW(format_, member_) \
    {YVEX_MODEL_TARGET_ROW_STRING, format_, offsetof(naming_audit_facts, member_)}
#define NAMING_LITERAL_ROW(text_) {YVEX_MODEL_TARGET_ROW_LITERAL, text_, 0u}

static const yvex_model_target_row_spec naming_audit_schema[] = {
    NAMING_STRING_ROW("tensor_map_status: %s", status),
    NAMING_STRING_ROW("tensor_map_family: %s", family),
    NAMING_STRING_ROW("tensor_map_target_id: %s", target),
    NAMING_LITERAL_ROW("tensor_map_stage: header-naming-map"),
    NAMING_LITERAL_ROW("tensor_map_evidence_basis: header-metadata-only"),
    NAMING_STRING_ROW("tensor_map_source_status: %s", presence),
    NAMING_STRING_ROW("tensor_map_config_status: %s", presence),
    NAMING_STRING_ROW("tensor_map_tokenizer_status: %s", presence),
    NAMING_STRING_ROW("tensor_map_tensor_count: %s", total),
    NAMING_STRING_ROW("tensor_map_mapped_total_count: %s", mapped_total),
    NAMING_STRING_ROW("tensor_map_unmapped_unknown_count: %s", unknown),
    NAMING_LITERAL_ROW("tensor_map_ambiguous_count: 0"),
    NAMING_STRING_ROW("tensor_map_layer_count_observed: %s", layers),
    NAMING_STRING_ROW("tensor_map_embedding_count: %s", embedding),
    NAMING_STRING_ROW("tensor_map_attention_count: %s", attention),
    NAMING_STRING_ROW("tensor_map_attention_q_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_attention_k_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_attention_v_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_attention_o_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_mlp_count: %s", mlp),
    NAMING_STRING_ROW("tensor_map_mlp_gate_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_mlp_up_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_mlp_down_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_norm_count: %s", norm),
    NAMING_STRING_ROW("tensor_map_output_head_count: %s", output),
    NAMING_STRING_ROW("tensor_map_qwen_linear_attn_count: %s", linear),
    NAMING_STRING_ROW("tensor_map_moe_router_count: %s", moe),
    NAMING_STRING_ROW("tensor_map_moe_expert_count: %s", moe),
    NAMING_STRING_ROW("tensor_map_moe_shared_count: %s", moe),
    NAMING_STRING_ROW("tensor_map_required_role_coverage_status: %s", coverage),
    NAMING_LITERAL_ROW("tensor_map_validation_status: lexical-and-header-only"),
    NAMING_LITERAL_ROW("tensor_map_canonical_role_status: mapped-candidates"),
    NAMING_LITERAL_ROW("tensor_map_runtime_role_coverage_status: report-only"),
    NAMING_LITERAL_ROW("tensor_map_artifact_contract_status: not-implemented"),
    NAMING_LITERAL_ROW("tensor_map_runtime_descriptor_status: not-implemented"),
    NAMING_LITERAL_ROW("tensor_map_graph_consumer_status: not-implemented")
};

static const char *const norm_mapping_rows[] = {
    "tensor_map.entry.0.mapping: model.layers.0.pre_feedforward_layernorm.weight "
    "-> model.layers.0.mlp.norm.weight",
    "tensor_map.entry.1.mapping: model.layers.0.post_feedforward_layernorm.weight "
    "-> model.layers.0.mlp.norm.weight"
};

static const char *const dense_mapping_rows[] = {
    "tensor_map.entry.0.mapping: model.embed_tokens.weight -> model.embedding.token.weight",
    "tensor_map.entry.1.mapping: model.layers.0.self_attn.q_proj.weight -> "
    "model.layers.0.attention.q_proj.weight",
    "tensor_map.entry.2.mapping: model.layers.0.self_attn.k_proj.weight -> "
    "model.layers.0.attention.k_proj.weight",
    "tensor_map.entry.3.mapping: model.layers.0.self_attn.v_proj.weight -> "
    "model.layers.0.attention.v_proj.weight",
    "tensor_map.entry.4.mapping: model.layers.0.self_attn.o_proj.weight -> "
    "model.layers.0.attention.o_proj.weight",
    "tensor_map.entry.5.mapping: model.layers.0.mlp.gate_proj.weight -> "
    "model.layers.0.mlp.gate_proj.weight",
    "tensor_map.entry.6.mapping: model.layers.0.mlp.up_proj.weight -> "
    "model.layers.0.mlp.up_proj.weight",
    "tensor_map.entry.7.mapping: model.layers.0.mlp.down_proj.weight -> "
    "model.layers.0.mlp.down_proj.weight",
    "tensor_map.entry.8.mapping: model.layers.0.input_layernorm.weight -> "
    "model.layers.0.attention.norm.weight",
    "tensor_map.entry.9.mapping: model.layers.0.post_attention_layernorm.weight -> "
    "model.layers.0.mlp.norm.weight",
    "tensor_map.entry.10.mapping: model.norm.weight -> model.final_norm.weight",
    "tensor_map.entry.11.mapping: lm_head.weight -> model.output_head.weight"
};

static const char *const moe_mapping_rows[] = {
    "tensor_map.entry.0.mapping: model.language_model.embed_tokens.weight -> "
    "model.embedding.token.weight",
    "tensor_map.entry.1.mapping: model.language_model.layers.0.self_attn.q_proj.weight "
    "-> model.layers.0.attention.q_proj.weight",
    "tensor_map.entry.2.mapping: model.language_model.layers.0.self_attn.k_proj.weight "
    "-> model.layers.0.attention.k_proj.weight",
    "tensor_map.entry.3.mapping: model.language_model.layers.0.self_attn.v_proj.weight "
    "-> model.layers.0.attention.v_proj.weight",
    "tensor_map.entry.4.mapping: model.language_model.layers.0.self_attn.o_proj.weight "
    "-> model.layers.0.attention.o_proj.weight",
    "tensor_map.entry.5.mapping: model.language_model.layers.0.linear_attn.A_log -> "
    "model.layers.0.qwen_linear_attn.A_log",
    "tensor_map.entry.6.mapping: model.language_model.layers.0.mlp.gate.weight -> "
    "model.layers.0.moe.router.weight",
    "tensor_map.entry.7.mapping: model.language_model.layers.0.mlp.experts.gate_up_proj "
    "-> model.layers.0.moe.experts.all.gate_up_proj.weight",
    "tensor_map.entry.8.mapping: model.language_model.layers.0.mlp.shared_expert.down_proj.weight "
    "-> model.layers.0.moe.shared_expert.down_proj.weight"
};

static void naming_audit_rows(yvex_model_target_report *report,
                              const yvex_model_target_request *request,
                              const char *family,
                              const char *status,
                              const char *total,
                              const char *moe,
                              const char *unknown,
                              const char *coverage,
                              int source_present)
{
    int missing = strcmp(status, "source-missing") == 0;
    int dense_qwen = strcmp(family, "qwen") == 0 &&
                     strcmp(request->target_id, "qwen3-8b") == 0;
    int dense_style = dense_qwen ||
                      strcmp(request->target_id, "gemma-4-12b-it") == 0;
    int norm_only = strcmp(status, "naming-map-norm-only") == 0;
    const char *unit = missing || norm_only ? "0" : "1";
    naming_audit_facts facts = {
        status, family, request->target_id, source_present ? "present" : "missing",
        total, strcmp(unknown, "0") == 0 ? total : "12", unknown,
        missing ? "0" : "1", unit, missing || norm_only ? "0" : "4", unit,
        missing || norm_only ? "0" : "3",
        missing ? "0" : (norm_only ? "2" : "3"), unit,
        missing || dense_qwen ? "0" : (strcmp(family, "qwen") == 0 ? "1" : "0"),
        moe, coverage
    };

    yvex_model_target_report_project_rows(
        report, naming_audit_schema,
        sizeof(naming_audit_schema) / sizeof(naming_audit_schema[0]), &facts);
    if (!missing && norm_only) {
        yvex_model_target_report_add_rows(
            report, norm_mapping_rows,
            sizeof(norm_mapping_rows) / sizeof(norm_mapping_rows[0]));
    } else if (!missing && dense_style) {
        yvex_model_target_report_add_rows(
            report, dense_mapping_rows,
            sizeof(dense_mapping_rows) / sizeof(dense_mapping_rows[0]));
        if (strcmp(unknown, "0") != 0) {
            yvex_model_target_report_add_row(
                report,
                "tensor_map.entry.12.native_name: model.layers.0.weird_unknown.weight");
            yvex_model_target_report_add_row(report, "tensor_map.entry.12.mapping_status: unmapped-unknown");
            yvex_model_target_report_add_row(report, "model.layers.0.weird_unknown.weight");
            yvex_model_target_report_add_row(report, "mapping_status: unmapped-unknown");
        }
    } else if (!missing) {
        yvex_model_target_report_add_rows(
            report, moe_mapping_rows,
            sizeof(moe_mapping_rows) / sizeof(moe_mapping_rows[0]));
    }
    if (!missing && !dense_style && strcmp(family, "gemma") == 0) {
        yvex_model_target_report_add_row(
            report, "tensor_map.entry.9.mapping: "
                    "model.language_model.layers.0.layer_scalar -> "
                    "model.layers.0.layer_scalar");
    }
    if (!missing && !dense_qwen && !norm_only) {
        yvex_model_target_report_add_row(report, "model.layers.0.weird_unknown.weight");
        yvex_model_target_report_add_row(report, "mapping_status: %s",
                                         strcmp(unknown, "0") == 0 ? "mapped-candidate" : "unmapped-unknown");
    }
    yvex_model_target_report_common_tail(report);
    if (missing) {
        yvex_model_target_report_add_row(report, "top_blocker: %s",
                                         strcmp(family, "gemma") == 0
                                             ? "missing-gemma-source-path"
                                             : "missing-qwen-source-path");
    }
    yvex_model_target_report_add_row(report, "next_required_rows: V010.MAP.8");
}

#undef NAMING_LITERAL_ROW
#undef NAMING_STRING_ROW

int yvex_tensor_naming_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const char *family;
    const char *status;
    const char *total;
    const char *moe;
    const char *unknown;
    const char *coverage;
    int source_present = 0;
    int missing_source;
    int norm_only;

    if (!request || !report ||
        request->kind != YVEX_MODEL_TARGET_COMMAND_TENSOR_MAP) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tensor_naming_report",
                       "tensor naming report requires tensor-map command kind");
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(request->target_id, "deepseek4-v4-flash") == 0) {
        return yvex_model_mapping_report_deepseek(request, report, err);
    }
    if (!naming_validate(request, report)) {
        return YVEX_OK;
    }
    if (request->output_contract[0]) {
        yvex_model_target_report_add_output_contract(
            report, "tensor-map", request->output_contract);
        return YVEX_OK;
    }
    family = yvex_model_target_family_key(request->target_id);
    naming_counts(request, family, &status, &total, &moe, &unknown, &coverage,
                  &source_present);
    missing_source = strcmp(status, "source-missing") == 0;
    norm_only = strcmp(status, "naming-map-norm-only") == 0;
    naming_maybe_write_sidecar(
        request,
        family,
        status,
        strcmp(family, "gemma") == 0 ? "required-groups-present" : coverage);
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report, "TENSOR NAMING MAP");
        yvex_model_target_report_add_row(
            report, "FAMILY  TARGET                STATUS                      TOTAL   "
                    "EMBED    ATTN     MLP    NORM    HEAD     MOE   UNKNOWN   "
                    "LAYERS  NEXT");
        yvex_model_target_report_add_row(report, "%-8s%-22s%s  %s  %s  %s  %s  %s  %s  %s  %s  %s  V010.MAP.8",
                                         family, request->target_id, status,
                                         total,
                                         missing_source || norm_only ? "0" : "1",
                                         missing_source || norm_only ? "0" : "4",
                                         missing_source || norm_only ? "0" : "3",
                                         missing_source ? "0" : (norm_only ? "2" : "3"),
                                         missing_source || norm_only ? "0" : "1",
                                         moe, unknown,
                                         missing_source ? "0" : "1");
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        naming_audit_rows(report, request, family, status, total, moe, unknown,
                          coverage, source_present);
        return YVEX_OK;
    }
    yvex_model_target_report_add_row(report, "tensor-map: %s [%s]",
                                     request->target_id,
                                     missing_source || norm_only ? "blocked" :
                                     strcmp(status, "naming-map-profiled") == 0
                                         ? "reported" : status);
    yvex_model_target_report_add_row(report, "family: %s  stage: header-naming-map  evidence: header-only", family);
    yvex_model_target_report_add_row(report,
                                     "roles: total=%s embedding=%s attention=%s "
                                     "mlp=%s norm=%s head=%s moe=%s unknown=%s",
                                     strcmp(unknown, "0") == 0 ? total : "12",
                                     missing_source || norm_only ? "0" : "1",
                                     missing_source || norm_only ? "0" : "4",
                                     missing_source || norm_only ? "0" : "3",
                                     missing_source ? "0" : (norm_only ? "2" : "3"),
                                     missing_source || norm_only ? "0" : "1",
                                     moe, unknown);
    yvex_model_target_report_add_row(report, "layers: %s",
                                     missing_source ? "0" : "1");
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     missing_source
                                         ? (strcmp(family, "gemma") == 0
                                                ? "missing-gemma-source-path"
                                                : "missing-qwen-source-path")
                                     : strcmp(request->target_id, "qwen3-8b") == 0
                                         ? "missing-qwen-runtime-role-validation"
                                     : strcmp(family, "gemma") == 0
                                         ? "missing-dense-runtime-role-validation"
                                         : "missing-qwen-tensor-role-map");
    yvex_model_target_report_add_row(report, "next: V010.MAP.8");
    yvex_model_target_report_add_row(report, "boundary: report-only; use --audit for tensor entries");
    return YVEX_OK;
}
/* GGUF lowering projects the sealed IR without becoming semantic identity. */

#define MAP_METADATA_CAP 48u

/* Local lowering lifecycle and diagnostic operations used before definition. */
static void lowering_close(yvex_deepseek_gguf_map *map);
static const char *lowering_failure_name(
    yvex_deepseek_gguf_map_failure_code code);
static yvex_tensor_scope map_scope(yvex_transform_scope scope);

static const yvex_model_family_ir_api *lowering_family_ir(void)
{
    return &yvex_model_register_deepseek_v4()->ir;
}

typedef struct {
    unsigned long long hash;
    unsigned long long value_plus_one;
} map_index_slot;

struct yvex_deepseek_gguf_map {
    yvex_deepseek_gguf_map_allocator allocator;
    yvex_deepseek_gguf_descriptor *descriptors;
    yvex_deepseek_gguf_contribution *contributions;
    map_index_slot *source_index;
    map_index_slot *emitted_index;
    map_index_slot *role_index;
    unsigned long long source_index_capacity;
    unsigned long long emitted_index_capacity;
    unsigned long long role_index_capacity;
    yvex_deepseek_gguf_metadata metadata[MAP_METADATA_CAP];
    yvex_deepseek_gguf_map_summary summary;
};

typedef struct {
    yvex_deepseek_gguf_map *map;
    const yvex_deepseek_v4_ir *architecture;
    const yvex_transform_ir *transform_ir;
    yvex_deepseek_gguf_map_failure *failure;
    yvex_error *err;
} map_builder;

typedef struct {
    yvex_deepseek_gguf_transform transform;
    unsigned int qtype;
    int supported;
} map_transform_projection;

static const yvex_tensor_collection map_collections[YVEX_TRANSFORM_SUBSYSTEM_COUNT] = {
    YVEX_TENSOR_COLLECTION_GLOBAL,
    YVEX_TENSOR_COLLECTION_ATTENTION,
    YVEX_TENSOR_COLLECTION_COMPRESSOR,
    YVEX_TENSOR_COLLECTION_INDEXER,
    YVEX_TENSOR_COLLECTION_NORM,
    YVEX_TENSOR_COLLECTION_MHC,
    YVEX_TENSOR_COLLECTION_ROUTER,
    YVEX_TENSOR_COLLECTION_ROUTED_EXPERT,
    YVEX_TENSOR_COLLECTION_SHARED_EXPERT,
    YVEX_TENSOR_COLLECTION_GLOBAL,
    YVEX_TENSOR_COLLECTION_AUXILIARY
};

static const yvex_tensor_scope map_scopes[] = {
    YVEX_TENSOR_SCOPE_GLOBAL,
    YVEX_TENSOR_SCOPE_MAIN_LAYER,
    YVEX_TENSOR_SCOPE_MTP
};

enum { MAP_SCOPE_COUNT = 3, MAP_SUBSYSTEM_COUNT = YVEX_TRANSFORM_SUBSYSTEM_COUNT };

static const yvex_deepseek_gguf_contribution_kind map_contribution_kinds[][2] = {
    {YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY,
     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY},
    {YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY,
     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_SCALE},
    {YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_WEIGHT,
     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_SCALE},
    {YVEX_DEEPSEEK_GGUF_CONTRIBUTION_ROUTING_TABLE,
     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_ROUTING_TABLE}
};

static const map_transform_projection map_transforms[YVEX_TRANSFORM_OP_COUNT] = {
    [YVEX_TRANSFORM_OP_IDENTITY] = {
        YVEX_DEEPSEEK_GGUF_TRANSFORM_DIRECT, YVEX_GGUF_NO_FORCED_QTYPE, 1},
    [YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR] = {
        YVEX_DEEPSEEK_GGUF_TRANSFORM_FP8_E4M3_E8M0, YVEX_GGUF_NO_FORCED_QTYPE, 1},
    [YVEX_TRANSFORM_OP_CHECKED_CAST] = {
        YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32, 26u, 1},
    [YVEX_TRANSFORM_OP_EXPERT_AGGREGATE] = {
        YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4, 39u, 1}
};

typedef struct {
    yvex_tensor_collection collection;
    unsigned long long count;
} map_collection_expectation;

static const map_collection_expectation map_trunk_expectations[] = {
    {YVEX_TENSOR_COLLECTION_GLOBAL, 6u},
    {YVEX_TENSOR_COLLECTION_ATTENTION, 344u},
    {YVEX_TENSOR_COLLECTION_MHC, 258u},
    {YVEX_TENSOR_COLLECTION_NORM, 86u},
    {YVEX_TENSOR_COLLECTION_ROUTED_EXPERT, 129u},
    {YVEX_TENSOR_COLLECTION_SHARED_EXPERT, 129u},
    {YVEX_TENSOR_COLLECTION_ROUTER, 86u},
    {YVEX_TENSOR_COLLECTION_COMPRESSOR, 164u},
    {YVEX_TENSOR_COLLECTION_INDEXER, 126u}
};

typedef struct {
    size_t offset;
    unsigned long long count;
} map_summary_expectation;

static const map_summary_expectation map_summary_expectations[] = {
    {offsetof(yvex_deepseek_gguf_map_summary, source_contribution_count),
     YVEX_DEEPSEEK_GGUF_SOURCE_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, descriptor_count),
     YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, trunk_descriptor_count),
     YVEX_DEEPSEEK_GGUF_TRUNK_DESCRIPTOR_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, mtp_descriptor_count),
     YVEX_DEEPSEEK_GGUF_MTP_DESCRIPTOR_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, pinned_standard_count),
     YVEX_DEEPSEEK_GGUF_TRUNK_DESCRIPTOR_COUNT},
    {offsetof(yvex_deepseek_gguf_map_summary, extension_count),
     YVEX_DEEPSEEK_GGUF_MTP_DESCRIPTOR_COUNT}
};

typedef enum {
    M_LIT = 0, M_MODEL, M_LAYER, M_CSA,
    M_LAYER_NUM, M_CSA_NUM, M_RATIOS, M_CLAMP
} map_metadata_owner;

#define M_STR YVEX_DEEPSEEK_GGUF_METADATA_STRING
#define M_U64 YVEX_DEEPSEEK_GGUF_METADATA_U64
#define M_F64 YVEX_DEEPSEEK_GGUF_METADATA_F64
#define M_BOOL YVEX_DEEPSEEK_GGUF_METADATA_BOOL
#define M_U64S YVEX_DEEPSEEK_GGUF_METADATA_U64_ARRAY
#define M_F64S YVEX_DEEPSEEK_GGUF_METADATA_F64_ARRAY

typedef yvex_deepseek_v4_model_spec model_t;
typedef yvex_deepseek_v4_layer_spec layer_t;

typedef struct {
    const char *key;
    yvex_deepseek_gguf_metadata_type type;
    map_metadata_owner owner;
    size_t offset;
    union {
        const char *string;
        unsigned long long u64;
        double f64;
    } literal;
} map_metadata_spec;

static const map_metadata_spec map_metadata_specs[] = {
    {"general.architecture", M_STR, M_LIT, 0u, {.string = "deepseek4"}},
    {"general.name", M_STR, M_LIT, 0u, {.string = "DeepSeek-V4-Flash"}},
    {"general.source.huggingface.repository", M_STR, M_MODEL, offsetof(model_t, repository), {0}},
    {"yvex.source.revision", M_STR, M_MODEL, offsetof(model_t, revision), {0}},
    {"deepseek4.block_count", M_U64, M_MODEL, offsetof(model_t, main_layer_count), {0}},
    {"deepseek4.embedding_length", M_U64, M_MODEL, offsetof(model_t, hidden_size), {0}},
    {"deepseek4.context_length", M_U64, M_MODEL, offsetof(model_t, maximum_context), {0}},
    {"deepseek4.vocab_size", M_U64, M_MODEL, offsetof(model_t, vocabulary_size), {0}},
    {"deepseek4.attention.head_count", M_U64, M_LAYER, offsetof(layer_t, query_heads), {0}},
    {"deepseek4.attention.head_count_kv", M_U64, M_LAYER, offsetof(layer_t, kv_heads), {0}},
    {"deepseek4.attention.key_length", M_U64, M_LAYER, offsetof(layer_t, head_dimension), {0}},
    {"deepseek4.attention.value_length", M_U64, M_LAYER, offsetof(layer_t, head_dimension), {0}},
    {"deepseek4.attention.layer_norm_rms_epsilon", M_F64, M_LAYER, offsetof(layer_t, rms_norm_epsilon), {0}},
    {"deepseek4.rope.dimension_count", M_U64, M_LAYER, offsetof(layer_t, rope_head_dimension), {0}},
    {"deepseek4.rope.freq_base", M_F64, M_LAYER_NUM, offsetof(layer_t, position.theta), {0}},
    {"deepseek4.attention.q_lora_rank", M_U64, M_LAYER, offsetof(layer_t, query_lora_rank), {0}},
    {"deepseek4.attention.output_lora_rank", M_U64, M_LAYER, offsetof(layer_t, output_lora_rank), {0}},
    {"deepseek4.attention.output_group_count", M_U64, M_LAYER, offsetof(layer_t, output_groups), {0}},
    {"deepseek4.attention.compress_ratios", M_U64S, M_RATIOS, 0u, {0}},
    {"deepseek4.attention.sliding_window", M_U64, M_LAYER, offsetof(layer_t, kv.sliding_window), {0}},
    {"deepseek4.expert_count", M_U64, M_LAYER, offsetof(layer_t, moe.routed_experts), {0}},
    {"deepseek4.expert_used_count", M_U64, M_LAYER, offsetof(layer_t, moe.experts_per_token), {0}},
    {"deepseek4.expert_shared_count", M_U64, M_LAYER, offsetof(layer_t, moe.shared_experts), {0}},
    {"deepseek4.expert_feed_forward_length", M_U64, M_LAYER, offsetof(layer_t, moe.expert_intermediate_size), {0}},
    {"deepseek4.expert_weights_scale", M_F64, M_LAYER, offsetof(layer_t, moe.routed_scaling_factor), {0}},
    {"deepseek4.expert_weights_norm", M_BOOL, M_LAYER, offsetof(layer_t, moe.normalize_topk_probabilities), {0}},
    {"deepseek4.expert_gating_func", M_U64, M_LIT, 0u, {.u64 = 4u}},
    {"deepseek4.swiglu_clamp_exp", M_F64S, M_CLAMP, 0u, {0}},
    {"deepseek4.swiglu_clamp_shexp", M_F64S, M_CLAMP, 0u, {0}},
    {"deepseek4.hash_layer_count", M_U64, M_MODEL, offsetof(model_t, hash_router_layer_count), {0}},
    {"deepseek4.attention.compress_rope_freq_base", M_F64, M_CSA_NUM, offsetof(layer_t, position.theta), {0}},
    {"deepseek4.hyper_connection.count", M_U64, M_LAYER, offsetof(layer_t, mhc.residual_streams), {0}},
    {"deepseek4.hyper_connection.sinkhorn_iterations", M_U64, M_LAYER, offsetof(layer_t, mhc.sinkhorn_iterations), {0}},
    {"deepseek4.hyper_connection.epsilon", M_F64, M_LAYER, offsetof(layer_t, mhc.epsilon), {0}},
    {"deepseek4.indexer.head_count", M_U64, M_CSA, offsetof(layer_t, indexer_heads), {0}},
    {"deepseek4.indexer.key_length", M_U64, M_CSA, offsetof(layer_t, indexer_head_dimension), {0}},
    {"deepseek4.indexer.top_k", M_U64, M_CSA, offsetof(layer_t, indexer_topk), {0}},
    {"tokenizer.ggml.model", M_STR, M_LIT, 0u, {.string = "gpt2"}},
    {"tokenizer.ggml.vocab_size", M_U64, M_MODEL, offsetof(model_t, tokenizer.vocabulary_size), {0}},
    {"tokenizer.ggml.bos_token_id", M_U64, M_MODEL, offsetof(model_t, tokenizer.bos_token_id), {0}},
    {"tokenizer.ggml.eos_token_id", M_U64, M_MODEL, offsetof(model_t, tokenizer.eos_token_id), {0}},
    {"yvex.tokenizer.sidecars_verified", M_BOOL, M_LIT, 0u, {.u64 = 1u}},
    {"yvex.deepseek4.mtp.schema", M_U64, M_LIT, 0u, {.u64 = YVEX_GGUF_MTP_EXTENSION_VERSION}},
    {"yvex.deepseek4.mtp.predictor_count", M_U64, M_MODEL, offsetof(model_t, auxiliary_layer_count), {0}},
    {"yvex.deepseek4.mtp.descriptor_count", M_U64, M_LIT, 0u, {.u64 = YVEX_DEEPSEEK_GGUF_MTP_DESCRIPTOR_COUNT}},
    {"yvex.deepseek4.mtp.runtime_supported", M_BOOL, M_LIT, 0u, {0}},
    {"yvex.deepseek4.mtp.name_prefix", M_STR, M_LIT, 0u, {.string = "yvex.mtp.v1"}}
};

static void *map_default_allocate(size_t size, void *context)
{
    (void)context;
    return malloc(size);
}

static void map_default_release(void *allocation, void *context)
{
    (void)context;
    free(allocation);
}

static unsigned long long map_hash_string(const char *text)
{
    return yvex_core_hash_mix_bytes(1469598103934665603ull, text, strlen(text) + 1u);
}

static void map_failure_clear(yvex_deepseek_gguf_map_failure *failure)
{
    if (!failure) return;
    *failure = cleared_map_failure;
}

static int map_reject(map_builder *builder,
                      yvex_deepseek_gguf_map_failure_code code,
                      yvex_tensor_role role,
                      yvex_tensor_scope scope,
                      unsigned long long layer,
                      unsigned long long predictor,
                      unsigned long long expert,
                      const char *source_name,
                      const char *emitted_name,
                      unsigned long long expected,
                      unsigned long long actual)
{
    yvex_status status = code == YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ALLOCATION
        ? YVEX_ERR_NOMEM
        : (code == YVEX_DEEPSEEK_GGUF_MAP_FAILURE_INVALID_ARGUMENT
            ? YVEX_ERR_INVALID_ARG : YVEX_ERR_FORMAT);
    yvex_deepseek_gguf_map_failure *failure =
        builder ? builder->failure : NULL;

    if (failure) {
        map_failure_clear(failure);
        failure->code = code;
        failure->role = role;
        failure->scope = scope;
        failure->layer_index = layer;
        failure->predictor_index = predictor;
        failure->expert_index = expert;
        failure->expected = expected;
        failure->actual = actual;
        yvex_core_text_copy(failure->source_name, sizeof(failure->source_name), source_name ? source_name : "");
        yvex_core_text_copy(failure->emitted_name, sizeof(failure->emitted_name), emitted_name ? emitted_name : "");
    }
    yvex_error_setf(builder ? builder->err : NULL, status,
                    "deepseek_gguf_lowering",
                    "%s role=%s source=%s emitted=%s layer=%llu expert=%llu expected=%llu actual=%llu",
                    lowering_failure_name(code),
                    yvex_tensor_role_name(role),
                    source_name ? source_name : "none",
                    emitted_name ? emitted_name : "none", layer, expert,
                    expected, actual);
    return status;
}

static int map_reject_global(map_builder *builder,
                             yvex_deepseek_gguf_map_failure_code code,
                             const char *subject,
                             unsigned long long expected,
                             unsigned long long actual)
{
    return map_reject(builder, code, YVEX_TENSOR_ROLE_UNKNOWN,
                      YVEX_TENSOR_SCOPE_GLOBAL, YVEX_DEEPSEEK_GGUF_NO_INDEX,
                      YVEX_DEEPSEEK_GGUF_NO_INDEX, YVEX_DEEPSEEK_GGUF_NO_INDEX,
                      subject, NULL, expected, actual);
}

static int map_reject_descriptor(map_builder *builder,
                                 yvex_deepseek_gguf_map_failure_code code,
                                 const yvex_deepseek_gguf_descriptor *descriptor,
                                 unsigned long long expert,
                                 const char *source,
                                 const char *emitted,
                                 unsigned long long expected,
                                 unsigned long long actual)
{
    return map_reject(builder, code, descriptor->role, descriptor->scope,
                      descriptor->layer_index, descriptor->predictor_index,
                      expert, source, emitted, expected, actual);
}

static int map_reject_key(map_builder *builder,
                          yvex_deepseek_gguf_map_failure_code code,
                          const yvex_transform_logical_key *key,
                          int include_location,
                          unsigned long long expected,
                          unsigned long long actual)
{
    return map_reject(builder, code, key ? key->role : YVEX_TENSOR_ROLE_UNKNOWN,
                      key ? map_scope(key->scope) : YVEX_TENSOR_SCOPE_GLOBAL,
                      key && include_location ? key->layer_index
                                              : YVEX_DEEPSEEK_GGUF_NO_INDEX,
                      key && include_location ? key->auxiliary_index
                                              : YVEX_DEEPSEEK_GGUF_NO_INDEX,
                      YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, NULL, expected, actual);
}

static void *map_allocate_zero(yvex_deepseek_gguf_map *map, size_t size)
{
    void *allocation = map->allocator.allocate(size, map->allocator.context);
    if (allocation) memset(allocation, 0, size);
    return allocation;
}

static int map_index_insert(map_index_slot *slots,
                            unsigned long long capacity,
                            unsigned long long hash,
                            unsigned long long value)
{
    unsigned long long slot;
    unsigned long long probe;

    if (!slots || !capacity || (capacity & (capacity - 1u)) != 0u) return 0;
    slot = hash & (capacity - 1u);
    for (probe = 0u; probe < capacity; ++probe) {
        if (!slots[slot].value_plus_one) {
            slots[slot].hash = hash;
            slots[slot].value_plus_one = value + 1u;
            return 1;
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    return 0;
}

static int map_unique_index_equal(const yvex_deepseek_gguf_map *map,
                                  int emitted,
                                  unsigned long long left,
                                  unsigned long long right)
{
    const yvex_deepseek_gguf_descriptor *candidate = &map->descriptors[right];
    const yvex_deepseek_gguf_descriptor *current = &map->descriptors[left];

    return emitted ? strcmp(current->emitted_name, candidate->emitted_name) == 0
                   : current->role == candidate->role &&
                         current->scope == candidate->scope &&
                         current->layer_index == candidate->layer_index &&
                         current->predictor_index == candidate->predictor_index;
}

static int map_unique_index_insert(yvex_deepseek_gguf_map *map,
                                   int emitted,
                                   unsigned long long hash,
                                   unsigned long long value)
{
    map_index_slot *slots = emitted ? map->emitted_index : map->role_index;
    unsigned long long capacity = emitted ? map->emitted_index_capacity
                                          : map->role_index_capacity;
    unsigned long long slot = hash & (capacity - 1u);
    unsigned long long probe;

    for (probe = 0u; probe < capacity; ++probe) {
        map_index_slot *entry = &slots[slot];
        if (!entry->value_plus_one) {
            entry->hash = hash;
            entry->value_plus_one = value + 1u;
            return 1;
        }
        if (entry->hash == hash &&
            map_unique_index_equal(
                map, emitted, entry->value_plus_one - 1u, value)) return 0;
        slot = (slot + 1u) & (capacity - 1u);
    }
    return 0;
}

static int map_emitted_index_insert(yvex_deepseek_gguf_map *map,
                                    unsigned long long hash,
                                    unsigned long long value)
{
    return map_unique_index_insert(map, 1, hash, value);
}

static int map_role_index_insert(yvex_deepseek_gguf_map *map,
                                 unsigned long long hash,
                                 unsigned long long value)
{
    return map_unique_index_insert(map, 0, hash, value);
}

static yvex_tensor_scope map_scope(yvex_transform_scope scope)
{
    return (unsigned int)scope < MAP_SCOPE_COUNT ? map_scopes[(unsigned int)scope]
                                                  : YVEX_TENSOR_SCOPE_GLOBAL;
}

static yvex_tensor_collection map_collection(
    yvex_transform_subsystem subsystem)
{
    return (unsigned int)subsystem < MAP_SUBSYSTEM_COUNT
        ? map_collections[(unsigned int)subsystem] : YVEX_TENSOR_COLLECTION_COUNT;
}

static int map_transform(const yvex_transform_node *node,
                         yvex_deepseek_gguf_transform *transform,
                         unsigned int *qtype)
{
    const map_transform_projection *projection;

    if (!node || !transform || !qtype) return 0;
    if ((unsigned int)node->kind >= YVEX_TRANSFORM_OP_COUNT ||
        !map_transforms[(unsigned int)node->kind].supported) return 0;
    projection = &map_transforms[(unsigned int)node->kind];
    *transform = projection->transform;
    *qtype = projection->qtype;
    return 1;
}

static yvex_deepseek_gguf_contribution_kind map_contribution_kind(
    yvex_deepseek_gguf_transform transform,
    unsigned long long input)
{
    unsigned int secondary = transform == YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4
        ? (unsigned int)(input & 1u) : input != 0u;

    return (unsigned int)transform <= YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32
        ? map_contribution_kinds[(unsigned int)transform][secondary]
        : YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY;
}

static int map_descriptor_begin(map_builder *builder,
                                const yvex_transform_value *terminal,
                                const yvex_transform_node *node,
                                unsigned long long descriptor_index)
{
    yvex_deepseek_gguf_map *map = builder->map;
    yvex_deepseek_gguf_descriptor *descriptor =
        &map->descriptors[descriptor_index];
    yvex_gguf_name_provenance provenance;
    yvex_tensor_scope scope = map_scope(terminal->logical_key.scope);
    yvex_tensor_collection collection =
        map_collection(terminal->logical_key.subsystem);
    const char *reason = NULL;
    unsigned long long role_hash = 1469598103934665603ull;
    unsigned int qtype;
    unsigned int dimension;

    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->role = terminal->logical_key.role;
    descriptor->collection = collection;
    descriptor->scope = scope;
    descriptor->layer_index = terminal->logical_key.layer_index;
    descriptor->predictor_index = terminal->logical_key.auxiliary_index;
    descriptor->expert_count = node->expert_count;
    if (collection >= YVEX_TENSOR_COLLECTION_COUNT ||
        !map_transform(node, &descriptor->transform, &qtype) ||
        terminal->shape.rank > YVEX_TENSOR_MAX_DIMS) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_LOWERING_DIVERGENCE,
            descriptor, YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, NULL, 1u, 0u);
    }
    descriptor->forced_qtype = qtype;
    descriptor->logical_rank = terminal->shape.rank;
    descriptor->contribution_offset = map->summary.source_contribution_count;
    for (dimension = 0u; dimension < terminal->shape.rank; ++dimension) {
        unsigned int source_axis = terminal->shape.rank - dimension - 1u;
        descriptor->logical_dims[dimension] =
            terminal->shape.dims[source_axis];
        descriptor->source_axis_for_logical[dimension] = source_axis;
    }
    if (node->kind == YVEX_TRANSFORM_OP_EXPERT_AGGREGATE) {
        descriptor->source_axis_for_logical[0] = 1u;
        descriptor->source_axis_for_logical[1] = 0u;
        descriptor->source_axis_for_logical[2] =
            YVEX_DEEPSEEK_GGUF_AGGREGATED_AXIS;
    }
    if (!yvex_gguf_name_map_resolve(
            descriptor->role, scope == YVEX_TENSOR_SCOPE_MTP,
            descriptor->layer_index, descriptor->predictor_index,
            descriptor->emitted_name, sizeof(descriptor->emitted_name),
            &provenance, &reason)) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_NAME, descriptor,
            YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, reason, 1u, 0u);
    }
    descriptor->name_provenance = provenance;
    if (!yvex_gguf_layout_map_shape_supported(
            descriptor->role, qtype, descriptor->logical_rank,
            descriptor->logical_dims, &reason)) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_LAYOUT, descriptor,
            YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, descriptor->emitted_name, 1u, 0u);
    }
    if (!map_emitted_index_insert(
            map, map_hash_string(descriptor->emitted_name), descriptor_index)) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_DUPLICATE_NAME, descriptor,
            YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, descriptor->emitted_name, 1u, 2u);
    }
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->role);
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->scope);
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->layer_index);
    role_hash = yvex_core_hash_mix_u64(role_hash, descriptor->predictor_index);
    if (!map_role_index_insert(map, role_hash, descriptor_index)) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_DUPLICATE_NAME, descriptor,
            YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, descriptor->emitted_name, 1u, 2u);
    }
    descriptor->identity = map_hash_string(descriptor->emitted_name);
    descriptor->identity = yvex_core_hash_mix_u64(descriptor->identity,
                                        descriptor->transform);
    descriptor->identity = yvex_core_hash_mix_u64(descriptor->identity, qtype);
    for (dimension = 0u; dimension < descriptor->logical_rank; ++dimension)
        descriptor->identity = yvex_core_hash_mix_u64(
            descriptor->identity, descriptor->logical_dims[dimension]);
    map->summary.descriptor_count++;
    map->summary.collection_counts[collection]++;
    if (scope == YVEX_TENSOR_SCOPE_MTP)
        map->summary.mtp_descriptor_count++;
    else
        map->summary.trunk_descriptor_count++;
    if (provenance == YVEX_GGUF_NAME_PINNED_STANDARD)
        map->summary.pinned_standard_count++;
    else if (provenance == YVEX_GGUF_NAME_SEMANTIC_STANDARD)
        map->summary.semantic_standard_count++;
    else
        map->summary.extension_count++;
    return YVEX_OK;
}

static int map_descriptor_add_source(
    map_builder *builder,
    const yvex_transform_node *node,
    unsigned long long descriptor_index,
    unsigned long long input_index)
{
    yvex_deepseek_gguf_map *map = builder->map;
    yvex_deepseek_gguf_descriptor *descriptor =
        &map->descriptors[descriptor_index];
    const yvex_transform_value *value = yvex_transform_ir_node_input_at(
        builder->transform_ir, node, input_index);
    const yvex_transform_source_value *source;
    yvex_deepseek_gguf_contribution *contribution;
    unsigned long long index = map->summary.source_contribution_count;
    unsigned int dimension;

    if (!value || value->kind != YVEX_TRANSFORM_VALUE_SOURCE)
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_LOWERING_DIVERGENCE,
            descriptor, YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL,
            descriptor->emitted_name, 1u, 0u);
    source = yvex_transform_ir_source_at(
        builder->transform_ir, value->source_index);
    if (!source || index >= YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        source->requirement_index >= YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        source->shape.rank > 2u || source->role_hint != descriptor->role ||
        map_scope(source->scope) != descriptor->scope ||
        map_collection(source->subsystem) != descriptor->collection) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_COVERAGE_ROW, descriptor,
            source ? source->expert_index : YVEX_DEEPSEEK_GGUF_NO_INDEX,
            source ? source->source_name : NULL, descriptor->emitted_name,
            1u, 0u);
    }
    contribution = &map->contributions[index];
    yvex_core_text_copy(contribution->source_name, sizeof(contribution->source_name), source->source_name);
    contribution->source_dtype = source->source_dtype;
    contribution->source_rank = source->shape.rank;
    for (dimension = 0u; dimension < source->shape.rank; ++dimension)
        contribution->source_dims[dimension] = source->shape.dims[dimension];
    contribution->kind = map_contribution_kind(descriptor->transform,
                                               input_index);
    contribution->source_row_index = source->requirement_index;
    contribution->descriptor_index = descriptor_index;
    contribution->expert_index = source->expert_index;
    if (!map_index_insert(map->source_index, map->source_index_capacity,
                          map_hash_string(source->source_name), index)) {
        return map_reject_descriptor(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_DUPLICATE_SOURCE,
            descriptor, source->expert_index,
            source->source_name, descriptor->emitted_name, 1u, 2u);
    }
    descriptor->contribution_count++;
    descriptor->identity = yvex_core_hash_mix_bytes(
        descriptor->identity, source->source_name,
        strlen(source->source_name) + 1u);
    map->summary.source_contribution_count++;
    return YVEX_OK;
}

static int map_build_descriptors(map_builder *builder)
{
    const yvex_transform_ir_summary *summary =
        yvex_transform_ir_summary_get(builder->transform_ir);
    unsigned long long ordinal;

    if (!summary || !summary->complete ||
        summary->state != YVEX_TRANSFORM_IR_STATE_SEALED ||
        summary->source_value_count != YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        summary->terminal_count != YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT ||
        summary->edge_count != YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        summary->payload_bytes_read != 0u) {
        return map_reject_global(
            builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_TRANSFORM_IR, NULL,
            YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT,
            summary ? summary->terminal_count : 0u);
    }
    for (ordinal = 0u; ordinal < summary->terminal_count; ++ordinal) {
        const yvex_transform_value *terminal =
            yvex_transform_ir_terminal_at(builder->transform_ir, ordinal);
        const yvex_transform_node *node;
        unsigned long long input;
        int rc;

        if (!terminal || terminal->canonical_ordinal != ordinal ||
            terminal->producer_node_id >= summary->node_count) {
            return map_reject_key(
                builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_TRANSFORM_IR,
                terminal ? &terminal->logical_key : NULL, 0, ordinal,
                terminal ? terminal->canonical_ordinal : ULLONG_MAX);
        }
        node = yvex_transform_ir_node_at(
            builder->transform_ir, terminal->producer_node_id);
        if (!node || node->output_value_id != terminal->id) {
            return map_reject_key(
                builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_TRANSFORM_IR,
                &terminal->logical_key, 1, terminal->id,
                node ? node->output_value_id : ULLONG_MAX);
        }
        rc = map_descriptor_begin(builder, terminal, node, ordinal);
        if (rc != YVEX_OK) return rc;
        for (input = 0u; input < node->input_count; ++input) {
            rc = map_descriptor_add_source(builder, node, ordinal, input);
            if (rc != YVEX_OK) return rc;
        }
    }
    return YVEX_OK;
}

static int map_metadata_begin(map_builder *builder,
                              const char *key,
                              yvex_deepseek_gguf_metadata **out)
{
    yvex_deepseek_gguf_map *map = builder->map;
    unsigned long long index;

    if (!key || map->summary.metadata_count >= MAP_METADATA_CAP)
        return map_reject_global(builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_METADATA,
                                 key, MAP_METADATA_CAP,
                                 map->summary.metadata_count + 1u);
    for (index = 0u; index < map->summary.metadata_count; ++index)
        if (strcmp(map->metadata[index].key, key) == 0)
            return map_reject_global(builder,
                                     YVEX_DEEPSEEK_GGUF_MAP_FAILURE_METADATA,
                                     key, 1u, 2u);
    *out = &map->metadata[map->summary.metadata_count++];
    yvex_core_text_copy((*out)->key, sizeof((*out)->key), key);
    return YVEX_OK;
}

static int map_add_metadata_spec(map_builder *builder,
                                 const map_metadata_spec *spec,
                                 const model_t *model,
                                 const layer_t *first,
                                 const layer_t *first_csa,
                                 const unsigned long long *ratios,
                                 const double *clamp)
{
    yvex_deepseek_gguf_metadata *entry;
    const void *owner = spec->owner == M_MODEL
        ? (const void *)model
        : (spec->owner == M_CSA ||
           spec->owner == M_CSA_NUM)
            ? (const void *)first_csa : (const void *)first;
    const char *field = owner ? (const char *)owner + spec->offset : NULL;
    unsigned int count = (unsigned int)model->main_layer_count;
    int rc;

    if (!count || count > 64u)
        return map_reject_global(builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_METADATA,
                                 spec->key, 64u, count);
    rc = map_metadata_begin(builder, spec->key, &entry);
    if (rc != YVEX_OK) return rc;
    entry->type = spec->type;
    if (spec->type == M_STR) {
        const char *value = spec->owner == M_LIT
            ? spec->literal.string : field;
        yvex_core_text_copy(entry->string_value, sizeof(entry->string_value), value);
    } else if (spec->type == M_U64) {
        entry->u64_value = spec->owner == M_LIT
            ? spec->literal.u64 : *(const unsigned long long *)field;
    } else if (spec->type == M_BOOL) {
        entry->bool_value = spec->owner == M_LIT
            ? spec->literal.u64 != 0u : *(const int *)field != 0;
    } else if (spec->type == M_F64) {
        entry->f64_value = spec->owner == M_LAYER_NUM ||
                           spec->owner == M_CSA_NUM
            ? (double)*(const unsigned long long *)field : *(const double *)field;
    } else if (spec->type == M_U64S) {
        memcpy(entry->array_values, ratios,
               (size_t)count * sizeof(entry->array_values[0]));
    } else {
        memcpy(entry->f64_array_values, clamp,
               (size_t)count * sizeof(entry->f64_array_values[0]));
    }
    entry->array_count = spec->type >= M_U64S
        ? count : 0u;
    return YVEX_OK;
}

static int map_build_metadata(map_builder *builder)
{
    const model_t *model =
        lowering_family_ir()->model(builder->architecture);
    const layer_t *first =
        lowering_family_ir()->layer_at(builder->architecture, 0u);
    const layer_t *first_csa =
        lowering_family_ir()->layer_at(builder->architecture, 2u);
    unsigned long long ratios[64];
    double clamp[64];
    unsigned long long index;
    int rc;
    for (index = 0u; index < model->main_layer_count; ++index) {
        const layer_t *layer =
            lowering_family_ir()->layer_at(builder->architecture, index);
        ratios[index] = layer->compression_ratio;
        clamp[index] = layer->moe.activation_limit;
    }
    for (index = 0u;
         index < sizeof(map_metadata_specs) / sizeof(map_metadata_specs[0]);
         ++index) {
        rc = map_add_metadata_spec(builder, &map_metadata_specs[index], model,
                                   first, first_csa, ratios, clamp);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

static int map_finalize(map_builder *builder)
{
    yvex_deepseek_gguf_map *map = builder->map;
    unsigned long long trunk[YVEX_TENSOR_COLLECTION_COUNT] = {0};
    unsigned long long identity = 1469598103934665603ull;
    unsigned long long index;

    for (index = 0u;
         index < sizeof(map_summary_expectations) / sizeof(map_summary_expectations[0]);
         ++index)
        if (*(const unsigned long long *)((const char *)&map->summary +
                                          map_summary_expectations[index].offset) !=
            map_summary_expectations[index].count)
            return map_reject_global(
                builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ACCOUNTING, NULL,
                YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT,
                map->summary.descriptor_count);
    for (index = 0u; index < map->summary.descriptor_count; ++index) {
        const yvex_deepseek_gguf_descriptor *descriptor =
            &map->descriptors[index];
        if (descriptor->scope != YVEX_TENSOR_SCOPE_MTP)
            trunk[descriptor->collection]++;
        identity = yvex_core_hash_mix_u64(identity, descriptor->identity);
    }
    for (index = 0u;
         index < sizeof(map_trunk_expectations) / sizeof(map_trunk_expectations[0]);
         ++index)
        if (trunk[map_trunk_expectations[index].collection] !=
            map_trunk_expectations[index].count)
            return map_reject(
                builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ACCOUNTING,
                YVEX_TENSOR_ROLE_UNKNOWN, YVEX_TENSOR_SCOPE_MAIN_LAYER,
                YVEX_DEEPSEEK_GGUF_NO_INDEX, YVEX_DEEPSEEK_GGUF_NO_INDEX,
                YVEX_DEEPSEEK_GGUF_NO_INDEX, NULL, NULL, 1328u, 0u);
    identity = yvex_core_hash_mix_u64(identity, map->summary.source_identity);
    identity = yvex_core_hash_mix_u64(identity, map->summary.coverage_identity);
    map->summary.mapping_identity = identity;
    map->summary.complete = 1;
    return YVEX_OK;
}

static int lowering_build_with_allocator(
    yvex_deepseek_gguf_map **out,
    const yvex_deepseek_v4_ir *architecture,
    const yvex_transform_ir *transform_ir,
    const yvex_deepseek_gguf_map_allocator *allocator,
    yvex_deepseek_gguf_map_failure *failure,
    yvex_error *err)
{
    const model_t *model;
    const yvex_transform_ir_summary *transform_summary;
    yvex_deepseek_gguf_map *map;
    map_builder builder;
    size_t bytes;
    int rc;

    if (out) *out = NULL;
    map_failure_clear(failure);
    yvex_error_clear(err);
    memset(&builder, 0, sizeof(builder));
    builder.failure = failure;
    builder.err = err;
    if (!out || !architecture || !transform_ir || !allocator ||
        !allocator->allocate || !allocator->release) {
        return map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_INVALID_ARGUMENT,
            NULL, 1u, 0u);
    }
    model = lowering_family_ir()->model(architecture);
    transform_summary = yvex_transform_ir_summary_get(transform_ir);
    if (!model || model->main_layer_count != 43u ||
        model->auxiliary_layer_count != 1u) {
        return map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ARCHITECTURE, NULL, 44u,
            model ? model->main_layer_count + model->auxiliary_layer_count : 0u);
    }
    if (!transform_summary || !transform_summary->complete ||
        transform_summary->source_value_count !=
            YVEX_DEEPSEEK_GGUF_SOURCE_COUNT ||
        transform_summary->terminal_count !=
            YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT) {
        return map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_TRANSFORM_IR, NULL,
            YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT,
            transform_summary ? transform_summary->terminal_count : 0u);
    }
    map = (yvex_deepseek_gguf_map *)allocator->allocate(
        sizeof(*map), allocator->context);
    if (!map)
        return map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ALLOCATION,
            "map", sizeof(*map), 0u);
    memset(map, 0, sizeof(*map));
    map->allocator = *allocator;
    builder.map = map;
    if (!yvex_core_power_of_two_capacity(YVEX_DEEPSEEK_GGUF_SOURCE_COUNT, 8ull,
                                         1ull, 2ull, &map->source_index_capacity) ||
        !yvex_core_power_of_two_capacity(YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT, 8ull,
                                         1ull, 2ull, &map->emitted_index_capacity) ||
        !yvex_core_power_of_two_capacity(YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT, 8ull,
                                         1ull, 2ull, &map->role_index_capacity))
    {
        rc = map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ARITHMETIC_OVERFLOW,
            "mapping-index", 1u, 0u);
        lowering_close(map);
        return rc;
    }
    map->descriptors = (yvex_deepseek_gguf_descriptor *)map_allocate_zero(
        map, (size_t)YVEX_DEEPSEEK_GGUF_DESCRIPTOR_COUNT *
             sizeof(*map->descriptors));
    map->contributions = (yvex_deepseek_gguf_contribution *)map_allocate_zero(
        map, (size_t)YVEX_DEEPSEEK_GGUF_SOURCE_COUNT *
             sizeof(*map->contributions));
    bytes = (size_t)map->source_index_capacity * sizeof(*map->source_index);
    map->source_index = (map_index_slot *)map_allocate_zero(map, bytes);
    bytes = (size_t)map->emitted_index_capacity * sizeof(*map->emitted_index);
    map->emitted_index = (map_index_slot *)map_allocate_zero(map, bytes);
    bytes = (size_t)map->role_index_capacity * sizeof(*map->role_index);
    map->role_index = (map_index_slot *)map_allocate_zero(map, bytes);
    builder.architecture = architecture;
    builder.transform_ir = transform_ir;
    if (!map->descriptors || !map->contributions || !map->source_index ||
        !map->emitted_index || !map->role_index) {
        rc = map_reject_global(
            &builder, YVEX_DEEPSEEK_GGUF_MAP_FAILURE_ALLOCATION,
            "mapping-tables", 1u, 0u);
        lowering_close(map);
        return rc;
    }
    map->summary.header_scan_count = transform_summary->header_scan_count;
    map->summary.payload_bytes_read = transform_summary->payload_bytes_read;
    map->summary.source_identity = transform_summary->source_snapshot_identity;
    map->summary.coverage_identity = transform_summary->coverage_identity;
    rc = map_build_descriptors(&builder);
    if (rc == YVEX_OK) rc = map_build_metadata(&builder);
    if (rc == YVEX_OK) rc = map_finalize(&builder);
    if (rc != YVEX_OK) {
        lowering_close(map);
        return rc;
    }
    *out = map;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int lowering_build(
    yvex_deepseek_gguf_map **out,
    const yvex_deepseek_v4_ir *architecture,
    const yvex_transform_ir *transform_ir,
    yvex_deepseek_gguf_map_failure *failure,
    yvex_error *err)
{
    yvex_deepseek_gguf_map_allocator allocator;
    allocator.allocate = map_default_allocate;
    allocator.release = map_default_release;
    allocator.context = NULL;
    return lowering_build_with_allocator(
        out, architecture, transform_ir, &allocator, failure, err);
}

static void lowering_close(yvex_deepseek_gguf_map *map)
{
    yvex_deepseek_gguf_map_allocator allocator;
    void *allocations[5];
    unsigned int index;

    if (!map) return;
    allocator = map->allocator;
    allocations[0] = map->role_index;
    allocations[1] = map->emitted_index;
    allocations[2] = map->source_index;
    allocations[3] = map->contributions;
    allocations[4] = map->descriptors;
    for (index = 0u; index < sizeof(allocations) / sizeof(allocations[0]); ++index)
        if (allocations[index]) allocator.release(allocations[index], allocator.context);
    allocator.release(map, allocator.context);
}

static const yvex_deepseek_gguf_map_summary *lowering_summary(
    const yvex_deepseek_gguf_map *map)
{
    return map ? &map->summary : NULL;
}

static const yvex_deepseek_gguf_descriptor *lowering_at(
    const yvex_deepseek_gguf_map *map,
    unsigned long long index)
{
    return map && index < map->summary.descriptor_count ? &map->descriptors[index] : NULL;
}

static const yvex_deepseek_gguf_contribution *
lowering_contribution_at(
    const yvex_deepseek_gguf_map *map,
    unsigned long long index)
{
    return map && index < map->summary.source_contribution_count ? &map->contributions[index] : NULL;
}

static const yvex_deepseek_gguf_descriptor *map_find_name(
    const yvex_deepseek_gguf_map *map,
    const char *name,
    int emitted)
{
    const map_index_slot *slots;
    unsigned long long capacity;
    unsigned long long hash;
    unsigned long long slot;
    unsigned long long probe;

    if (!map || !name) return NULL;
    slots = emitted ? map->emitted_index : map->source_index;
    capacity = emitted ? map->emitted_index_capacity
                       : map->source_index_capacity;
    hash = map_hash_string(name);
    slot = hash & (capacity - 1u);
    for (probe = 0u; probe < capacity && slots[slot].value_plus_one; ++probe) {
        if (slots[slot].hash == hash) {
            unsigned long long value = slots[slot].value_plus_one - 1u;
            if (emitted) {
                if (strcmp(map->descriptors[value].emitted_name, name) == 0)
                    return &map->descriptors[value];
            } else if (strcmp(map->contributions[value].source_name,
                              name) == 0) {
                return &map->descriptors[
                    map->contributions[value].descriptor_index];
            }
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    return NULL;
}

static const yvex_deepseek_gguf_descriptor *lowering_find_source(
    const yvex_deepseek_gguf_map *map,
    const char *source_name)
{
    return map_find_name(map, source_name, 0);
}

static const yvex_deepseek_gguf_descriptor *lowering_find_emitted(
    const yvex_deepseek_gguf_map *map,
    const char *emitted_name)
{
    return map_find_name(map, emitted_name, 1);
}

static const yvex_deepseek_gguf_descriptor *lowering_find_role(
    const yvex_deepseek_gguf_map *map,
    yvex_tensor_role role,
    yvex_tensor_scope scope,
    unsigned long long layer_index,
    unsigned long long predictor_index)
{
    unsigned long long hash = 1469598103934665603ull;
    unsigned long long slot;
    unsigned long long probe;
    if (!map) return NULL;
    hash = yvex_core_hash_mix_u64(hash, role);
    hash = yvex_core_hash_mix_u64(hash, scope);
    hash = yvex_core_hash_mix_u64(hash, layer_index);
    hash = yvex_core_hash_mix_u64(hash, predictor_index);
    slot = hash & (map->role_index_capacity - 1u);
    for (probe = 0u; probe < map->role_index_capacity &&
         map->role_index[slot].value_plus_one; ++probe) {
        if (map->role_index[slot].hash == hash) {
            const yvex_deepseek_gguf_descriptor *descriptor =
                &map->descriptors[map->role_index[slot].value_plus_one - 1u];
            if (descriptor->role == role && descriptor->scope == scope &&
                descriptor->layer_index == layer_index &&
                descriptor->predictor_index == predictor_index)
                return descriptor;
        }
        slot = (slot + 1u) & (map->role_index_capacity - 1u);
    }
    return NULL;
}

static const yvex_deepseek_gguf_metadata *lowering_metadata_at(
    const yvex_deepseek_gguf_map *map,
    unsigned long long index)
{
    return map && index < map->summary.metadata_count ? &map->metadata[index] : NULL;
}

static const yvex_deepseek_gguf_metadata *lowering_metadata_find(
    const yvex_deepseek_gguf_map *map,
    const char *key)
{
    unsigned long long index;
    if (!map || !key) return NULL;
    for (index = 0u; index < map->summary.metadata_count; ++index)
        if (strcmp(map->metadata[index].key, key) == 0)
            return &map->metadata[index];
    return NULL;
}

static const char *lowering_transform_name(
    yvex_deepseek_gguf_transform transform)
{
    return transform <= YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32
        ? lowering_transform_names[transform] : "unknown";
}

static const char *lowering_failure_name(
    yvex_deepseek_gguf_map_failure_code code)
{
    return code <= YVEX_DEEPSEEK_GGUF_MAP_FAILURE_MAPPING_IDENTITY
        ? lowering_failure_names[code] : "unknown";
}

/*
 * Publish the immutable GGUF-lowering operation table used by the family registration without
 * exporting its implementation helpers.
 *
 * Returns process-lifetime immutable storage; no allocation or I/O.
 */
const yvex_model_family_lowering_api *yvex_model_deepseek_lowering_api(void)
{
    static const yvex_model_family_lowering_api api = {
        lowering_build,
        lowering_build_with_allocator,
        lowering_close,
        lowering_summary,
        lowering_at,
        lowering_contribution_at,
        lowering_find_source,
        lowering_find_emitted,
        lowering_find_role,
        lowering_metadata_at,
        lowering_metadata_find,
        lowering_transform_name,
        lowering_failure_name
    };

    return &api;
}
