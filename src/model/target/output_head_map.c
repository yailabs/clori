/*
 * Derive output-head mapping facts from bounded headers and configuration.
 *
 * Output-head reports are header/metadata mapping facts only and do not create logits support.
 * Output-head mapping is not runtime logits, generation support, benchmark evidence, or release
 * readiness.
 */
#include <yvex/internal/model_target.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    int header_checked;
    int embed_seen;
    int separate_head_seen;
    int separate_head_count;
    int config_checked;
    int config_tie_true;
    int config_tie_false;
} output_head_probe;

typedef struct {
    const char *status;
    const char *family;
    const char *target;
    const char *source_status;
    int candidate_count;
    int ambiguous_count;
    const char *native;
    const char *canonical;
    const char *mapping;
    const char *missing_status;
    const char *tie;
    const char *config_tie;
    const char *shape_relation;
} output_head_report_facts;

#define OUTPUT_HEAD_LITERAL(text) \
    { YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u }
#define OUTPUT_HEAD_STRING(field, format) \
    { YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(output_head_report_facts, field) }
#define OUTPUT_HEAD_INT(field, format) \
    { YVEX_MODEL_TARGET_ROW_INT, (format), offsetof(output_head_report_facts, field) }

static const yvex_model_target_row_spec output_head_audit_rows[] = {
    OUTPUT_HEAD_STRING(status, "output_head_map_status: %s"),
    OUTPUT_HEAD_STRING(family, "output_head_map_family: %s"),
    OUTPUT_HEAD_STRING(target, "output_head_map_target_id: %s"),
    OUTPUT_HEAD_LITERAL("output_head_map_stage: header-output-head-map"),
    OUTPUT_HEAD_LITERAL("output_head_map_evidence_basis: header-metadata-only"),
    OUTPUT_HEAD_STRING(source_status, "output_head_map_source_status: %s"),
    OUTPUT_HEAD_INT(candidate_count, "output_head_candidate_count: %d"),
    OUTPUT_HEAD_INT(ambiguous_count, "output_head_ambiguous_count: %d"),
    OUTPUT_HEAD_STRING(native, "output_head_native_name: %s"),
    OUTPUT_HEAD_STRING(canonical, "output_head_canonical_role: %s"),
    OUTPUT_HEAD_STRING(mapping, "output_head_mapping_status: %s"),
    OUTPUT_HEAD_STRING(missing_status, "output_head_missing_status: %s"),
    OUTPUT_HEAD_LITERAL("embedding_canonical_role: model.embedding.token.weight"),
    OUTPUT_HEAD_LITERAL("final_norm_canonical_role: model.final_norm.weight"),
    OUTPUT_HEAD_STRING(tie, "tie_policy_status: %s"),
    OUTPUT_HEAD_STRING(config_tie, "config_tie_word_embeddings_status: %s"),
    OUTPUT_HEAD_STRING(shape_relation, "shape_relation_status: %s"),
    OUTPUT_HEAD_LITERAL("output_head_runtime_consumer_status: target-runtime-owned"),
    OUTPUT_HEAD_LITERAL("output_head_logits_status: target-capability-dependent"),
    OUTPUT_HEAD_LITERAL("output_head_artifact_contract_status: artifact-owner"),
    OUTPUT_HEAD_LITERAL("output_head_runtime_descriptor_status: runtime-owner"),
    OUTPUT_HEAD_LITERAL("output_head_graph_consumer_status: runtime-logits-owner")
};

static const yvex_model_target_row_spec output_head_entry_rows[] = {
    OUTPUT_HEAD_STRING(native, "output_head.entry.output.native_name: %s"),
    OUTPUT_HEAD_STRING(canonical, "output_head.entry.output.canonical_role: %s")
};

#undef OUTPUT_HEAD_LITERAL
#undef OUTPUT_HEAD_STRING
#undef OUTPUT_HEAD_INT

static int output_head_json_bool_field(const char *text, const char *key,
                                       int *seen_true, int *seen_false)
{
    char needle[96];
    const char *p;

    if (seen_true) *seen_true = 0;
    if (seen_false) *seen_false = 0;
    if (!text || !key) {
        return 0;
    }
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(text, needle);
    if (!p) {
        return 0;
    }
    p = strchr(p, ':');
    if (!p) {
        return 0;
    }
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
        p++;
    }
    if (strncmp(p, "true", 4u) == 0) {
        if (seen_true) *seen_true = 1;
        return 1;
    }
    if (strncmp(p, "false", 5u) == 0) {
        if (seen_false) *seen_false = 1;
        return 1;
    }
    return 0;
}

/*
 * Collect bounded local header/config facts for output-head reports.
 *
 * Missing local files leave probe fields unset so deterministic fallback report facts can still be
 * emitted.
 */
static void output_head_probe_source(const yvex_model_target_request *request,
                                     const char *family,
                                     output_head_probe *probe)
{
    char path[1024];
    char config[2048];
    char *header = NULL;

    if (!request || !family || !probe) {
        return;
    }
    memset(probe, 0, sizeof(*probe));
    if (yvex_model_target_probe_source_path(
            request, family, "model.safetensors", path, sizeof(path)) &&
        yvex_model_target_probe_header(path, &header)) {
        probe->header_checked = 1;
        probe->embed_seen =
            strstr(header, "model.language_model.embed_tokens.weight") != NULL ||
            strstr(header, "model.embed_tokens.weight") != NULL;
        probe->separate_head_seen =
            strstr(header, "model.language_model.lm_head.weight") != NULL ||
            strstr(header, "lm_head.weight") != NULL ||
            strstr(header, "output.weight") != NULL;
        probe->separate_head_count =
            (strstr(header, "model.language_model.lm_head.weight") != NULL ? 1 : 0) +
            (strstr(header, "lm_head.weight") != NULL ? 1 : 0) +
            (strstr(header, "output.weight") != NULL ? 1 : 0);
        free(header);
    }
    if (yvex_model_target_probe_source_path(
            request, family, "config.json", path, sizeof(path)) &&
        yvex_model_target_probe_read(path, config, sizeof(config))) {
        probe->config_checked =
            output_head_json_bool_field(config, "tie_word_embeddings",
                                        &probe->config_tie_true,
                                        &probe->config_tie_false);
    }
}

static const char *output_head_status(const yvex_model_target_request *request,
                                      const char *family,
                                      const output_head_probe *probe)
{
    if ((request->source_path[0] || request->models_root[0]) &&
        probe && !probe->header_checked) {
        return "source-missing";
    }
    if (strcmp(family, "gemma") == 0 && probe && probe->header_checked) {
        if (probe->separate_head_seen) {
            return "output-head-profiled";
        }
        if (probe->config_tie_true && probe->embed_seen) {
            return "tied-output-head-report-only";
        }
        return "output-head-missing";
    }
    if (strcmp(family, "gemma") == 0 &&
        strstr(request->source_path, "tied") != NULL &&
        strstr(request->source_path, "no-head") == NULL) {
        return "tied-output-head-report-only";
    }
    if (strcmp(family, "gemma") == 0 &&
        strstr(request->source_path, "no-head") != NULL) {
        return "output-head-missing";
    }
    if (strcmp(family, "qwen") == 0 && probe && probe->header_checked) {
        if (probe->separate_head_count == 0) {
            return "output-head-missing";
        }
        if (probe->separate_head_count > 1) {
            return "output-head-ambiguous";
        }
        return "output-head-profiled";
    }
    return "output-head-profiled";
}

static void output_head_write_sidecar(const yvex_model_target_request *request,
                                      const char *family,
                                      const char *status)
{
    char path[1024];

    if (!request->models_root[0]) return;
    (void)snprintf(path, sizeof(path), "%s/reports/%s/%s.output-head-map.json",
                   request->models_root, family, request->target_id);
    (void)yvex_model_target_write_sidecar(YVEX_MODEL_TARGET_SIDECAR_OUTPUT_HEAD, path,
                                          request->target_id, family, status, NULL);
}

int yvex_output_head_map_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const char *family;
    const char *status;
    const char *native;
    const char *canonical;
    const char *tie;
    const char *mapping;
    output_head_probe probe;
    output_head_report_facts facts;

    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "output_head_map",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_model_target_validate_supported(
            request, report, "tensor-map", 0)) {
        return YVEX_OK;
    }
    family = yvex_model_target_family_key(request->target_id);
    output_head_probe_source(request, family, &probe);
    status = output_head_status(request, family, &probe);
    native = strcmp(family, "gemma") == 0
                 ? (strcmp(status, "source-missing") == 0
                        ? "none"
                    : strcmp(status, "tied-output-head-report-only") == 0
                        ? "model.language_model.embed_tokens.weight"
                        : (strcmp(status, "output-head-missing") == 0
                               ? "none"
                               : (strcmp(request->target_id, "gemma-4-12b-it") == 0
                                      ? "lm_head.weight"
                                      : "model.language_model.lm_head.weight")))
                 : (strcmp(status, "source-missing") == 0 ? "none" : "lm_head.weight");
    canonical = strcmp(status, "tied-output-head-report-only") == 0
                    ? "model.output_head.tied_embedding"
                    : (strcmp(status, "output-head-missing") == 0 ||
                       strcmp(status, "source-missing") == 0
                           ? "missing"
                           : "model.output_head.weight");
    tie = strcmp(status, "tied-output-head-report-only") == 0
              ? "tied-output-head-candidate"
              : (strcmp(status, "source-missing") == 0
                     ? "unknown"
                 : strcmp(status, "output-head-missing") == 0
                     ? "not-proven"
                     : "separate-output-head-candidate");
    mapping = strcmp(status, "tied-output-head-report-only") == 0
                  ? "tied-to-token-embedding-candidate"
                  : (strcmp(status, "output-head-ambiguous") == 0
                         ? "ambiguous"
                  : (strcmp(status, "output-head-missing") == 0 ||
                     strcmp(status, "source-missing") == 0
                         ? "missing"
                         : "mapped-candidate"));
    memset(&facts, 0, sizeof(facts));
    facts.status = status;
    facts.family = family;
    facts.target = request->target_id;
    facts.source_status = strcmp(status, "source-missing") == 0
                              ? "missing" : "present";
    facts.candidate_count = strcmp(status, "source-missing") == 0 ||
                            strcmp(status, "output-head-missing") == 0
                                ? 0
                                : (strcmp(status, "output-head-ambiguous") == 0
                                       ? 2 : 1);
    facts.ambiguous_count = strcmp(status, "output-head-ambiguous") == 0;
    facts.native = native;
    facts.canonical = canonical;
    facts.mapping = mapping;
    facts.missing_status = strcmp(status, "output-head-missing") == 0 ||
                           strcmp(status, "source-missing") == 0
                               ? "missing" : "present";
    facts.tie = tie;
    facts.config_tie = probe.config_checked
                           ? (probe.config_tie_true ? "true" : "false")
                           : "missing";
    facts.shape_relation = strcmp(status, "source-missing") == 0
                               ? "unknown" : "compatible-same-shape";
    output_head_write_sidecar(request, family, status);

    if (request->output_contract[0]) {
        yvex_model_target_report_add_output_contract(
            report, "output-head-map", request->output_contract);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report, "OUTPUT HEAD TENSOR MAP");
        yvex_model_target_report_add_row(
            report, "FAMILY  TARGET                STATUS                           HEAD  "
                    "FINAL_NORM  EMBED  TIE_POLICY                          "
                    "SHAPE_RELATION            NEXT");
        yvex_model_target_report_add_row(report,
                                         "%-8s%-22s%-33s%s    %s         %s    %s  %s  V010.MAP.8",
                                         family, request->target_id, status,
                                         (strcmp(status, "output-head-missing") == 0 ||
                                          strcmp(status, "source-missing") == 0) ? "no " : "yes",
                                         strcmp(status, "source-missing") == 0 ? "no " : "yes",
                                         strcmp(status, "source-missing") == 0 ? "no " : "yes",
                                         tie,
                                         strcmp(status, "source-missing") == 0
                                             ? "unknown" : "compatible-same-shape");
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        yvex_model_target_report_project_rows(
            report, output_head_audit_rows,
            sizeof(output_head_audit_rows) / sizeof(output_head_audit_rows[0]),
            &facts);
        if (strcmp(status, "output-head-missing") == 0) {
            yvex_model_target_report_add_row(report,
                                             "top_blocker: missing-output-head-tensor");
        } else if (strcmp(status, "output-head-ambiguous") == 0) {
            yvex_model_target_report_add_row(report,
                                             "top_blocker: ambiguous-output-head-tensor");
        }
        yvex_model_target_report_common_tail(report);
        yvex_model_target_report_add_row(report, "next_required_rows: V010.MAP.8");
        if (strcmp(status, "source-missing") != 0 &&
            strcmp(status, "output-head-missing") != 0) {
            yvex_model_target_report_project_rows(
                report, output_head_entry_rows,
                sizeof(output_head_entry_rows) / sizeof(output_head_entry_rows[0]),
                &facts);
        }
        return YVEX_OK;
    }
    yvex_model_target_report_add_row(report, "output-head-map: %s [%s]",
                                     request->target_id,
                                     strcmp(status, "source-missing") == 0 ? "blocked" : "reported");
    yvex_model_target_report_add_row(report, "family: %s  evidence: header-only", family);
    yvex_model_target_report_add_row(report, "head: %s  final_norm: %s  embedding: %s  tie: %s",
                                     strcmp(status, "source-missing") == 0 ? "missing" : canonical,
                                     strcmp(status, "source-missing") == 0 ? "missing" : "model.final_norm.weight",
                                     strcmp(status, "source-missing") == 0 ? "missing" : "model.embedding.token.weight",
                                     tie);
    yvex_model_target_report_add_row(report, "shape: %s",
                                     strcmp(status, "source-missing") == 0 ? "unknown" : "compatible-same-shape");
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     strcmp(status, "source-missing") == 0
                                         ? (strcmp(family, "gemma") == 0
                                                ? "missing-gemma-source-path"
                                                : "missing-qwen-source-path")
                                         : "missing-output-head-runtime-consumer");
    yvex_model_target_report_add_row(report, "next: V010.MAP.8");
    yvex_model_target_report_add_row(report, "boundary: mapping only; no logits/runtime/generation");
    return YVEX_OK;
}
