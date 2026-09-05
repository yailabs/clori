/*
 * Project missing-role accounting and blockers from canonical facts.
 *
 * Missing-role reports describe coverage blockers only and never promote a lexical tensor map to
 * runtime readiness. Missing-role reporting is not tensor materialization, runtime support,
 * generation readiness, benchmark evidence, or release readiness.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/source.h>

#include <stdio.h>
#include <string.h>

typedef struct {
    const char *name;
    const char *status;
    const char *blocker;
} missing_role_fact;

typedef struct {
    yvex_model_target_source_profile source;
    int tensor_map_incomplete;
    int tokenizer_map_present;
    int output_head_map_present;
    int output_head_map_missing;
    int output_head_map_tied;
} missing_role_state;

typedef struct {
    const char *family;
    const char *target;
    const char *attention_k;
    const char *output_head;
    const char *tokenizer;
    const char *config;
    const char *generation;
    const char *specials;
    const char *top;
    int source_observed;
    int source_missing;
    int source_ambiguous;
    int metadata_observed;
    int metadata_missing;
} missing_role_source_facts;

typedef struct {
    const char *target;
    const char *family;
    const char *tensor;
    const char *tensor_path;
    const char *head;
    const char *tokenizer;
    const char *artifact_path;
    const char *top;
    const char *next;
    const char *status;
    const char *role_status;
    const char *output_head;
    const char *tied_head;
} missing_role_dynamic_facts;

#define MISSING_LITERAL(text) \
    { YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u }
#define MISSING_SOURCE_STRING(field, format) \
    { YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(missing_role_source_facts, field) }
#define MISSING_SOURCE_INT(field, format) \
    { YVEX_MODEL_TARGET_ROW_INT, (format), offsetof(missing_role_source_facts, field) }
#define MISSING_DYNAMIC_STRING(field, format) \
    { YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(missing_role_dynamic_facts, field) }

static const yvex_model_target_row_spec missing_role_source_rows[] = {
    MISSING_LITERAL("missing_role_report_status: missing-role-report-blocked"),
    MISSING_SOURCE_STRING(family, "missing_role_report_family: %s"),
    MISSING_SOURCE_STRING(target, "missing_role_report_target_id: %s"),
    MISSING_LITERAL("missing_role_report_stage: missing-role-blocker-report"),
    MISSING_LITERAL(
        "missing_role_report_evidence_basis: header-and-sidecar-metadata-only"),
    MISSING_LITERAL("missing_role_source_role_required_count: 12"),
    MISSING_SOURCE_INT(source_observed, "missing_role_source_role_observed_count: %d"),
    MISSING_SOURCE_INT(source_missing, "missing_role_source_role_missing_count: %d"),
    MISSING_LITERAL("missing_role_metadata_required_count: 4"),
    MISSING_SOURCE_INT(metadata_observed, "missing_role_metadata_observed_count: %d"),
    MISSING_SOURCE_INT(metadata_missing, "missing_role_metadata_missing_count: %d"),
    MISSING_LITERAL("missing_role_embedding_status: present"),
    MISSING_LITERAL("missing_role_attention_norm_status: present"),
    MISSING_LITERAL("missing_role_attention_q_status: present"),
    MISSING_SOURCE_STRING(attention_k, "missing_role_attention_k_status: %s"),
    MISSING_LITERAL("missing_role_attention_v_status: present"),
    MISSING_LITERAL("missing_role_attention_o_status: present"),
    MISSING_LITERAL("missing_role_mlp_norm_status: present"),
    MISSING_LITERAL("missing_role_mlp_gate_status: present"),
    MISSING_LITERAL("missing_role_mlp_up_status: present"),
    MISSING_LITERAL("missing_role_mlp_down_status: present"),
    MISSING_LITERAL("missing_role_final_norm_status: present"),
    MISSING_SOURCE_STRING(output_head, "missing_role_output_head_status: %s"),
    MISSING_SOURCE_STRING(tokenizer, "missing_role_tokenizer_metadata_status: %s"),
    MISSING_SOURCE_STRING(config, "missing_role_config_metadata_status: %s"),
    MISSING_SOURCE_STRING(generation, "missing_role_generation_metadata_status: %s"),
    MISSING_SOURCE_STRING(specials, "missing_role_special_tokens_status: %s"),
    MISSING_LITERAL("missing_role_artifact_contract_status: missing"),
    MISSING_LITERAL("missing_role_runtime_descriptor_status: missing"),
    MISSING_LITERAL("missing_role_graph_consumer_status: missing"),
    MISSING_LITERAL("missing_role_logits_runtime_status: missing"),
    MISSING_LITERAL("missing_role_tokenizer_runtime_status: missing"),
    MISSING_SOURCE_STRING(top, "missing_role_top_blocker: %s"),
    MISSING_LITERAL("missing_role_next_required_row: V010.MAP.9")
};

static const yvex_model_target_row_spec missing_role_dynamic_prefix[] = {
    MISSING_DYNAMIC_STRING(target, "target_id: %s"),
    MISSING_DYNAMIC_STRING(family, "family: %s"),
    MISSING_LITERAL("source_status: present"),
    MISSING_DYNAMIC_STRING(tensor, "tensor_map_status: %s"),
    MISSING_DYNAMIC_STRING(tensor_path, "tensor_map_path: %s")
};

static const yvex_model_target_row_spec missing_role_dynamic_suffix[] = {
    MISSING_DYNAMIC_STRING(head, "output_head_map_status: %s"),
    MISSING_DYNAMIC_STRING(tokenizer, "tokenizer_map_status: %s"),
    MISSING_LITERAL("artifact_status: missing"),
    MISSING_DYNAMIC_STRING(artifact_path, "expected_artifact_path: %s"),
    MISSING_LITERAL("artifact_emission_status: not-performed"),
    MISSING_LITERAL("artifact_identity_status: missing"),
    MISSING_LITERAL("prepare_blocker_count: 3"),
    MISSING_DYNAMIC_STRING(top, "top_blocker: %s"),
    MISSING_DYNAMIC_STRING(next, "next: %s"),
    MISSING_LITERAL("runtime_execution: not-performed"),
    MISSING_LITERAL("generation: unsupported"),
    MISSING_LITERAL("benchmark_status: not-measured")
};

static const yvex_model_target_row_spec missing_role_normal_prefix[] = {
    MISSING_DYNAMIC_STRING(status, "status: %s"),
    MISSING_DYNAMIC_STRING(family, "family: %s"),
    MISSING_DYNAMIC_STRING(top, "top_blocker: %s")
};

static const yvex_model_target_row_spec missing_role_normal_suffix[] = {
    MISSING_DYNAMIC_STRING(role_status, "moe-router: %s"),
    MISSING_DYNAMIC_STRING(role_status, "moe-experts: %s"),
    MISSING_DYNAMIC_STRING(role_status, "shared-expert: %s"),
    MISSING_DYNAMIC_STRING(output_head, "output-head: %s"),
    MISSING_DYNAMIC_STRING(tied_head, "tied-head-policy: %s"),
    MISSING_LITERAL("unknown-tensors: unclassified-header-name"),
    MISSING_DYNAMIC_STRING(tokenizer, "tokenizer: %s"),
    MISSING_LITERAL("artifact: missing"),
    MISSING_DYNAMIC_STRING(next, "next: %s")
};

#undef MISSING_LITERAL
#undef MISSING_SOURCE_STRING
#undef MISSING_SOURCE_INT
#undef MISSING_DYNAMIC_STRING

static const missing_role_fact qwen_missing_roles[] = {
    {"token_embedding", "covered-report-only", "none"},
    {"attention_q", "covered-report-only", "none"},
    {"attention_k", "covered-report-only", "none"},
    {"attention_v", "covered-report-only", "none"},
    {"attention_o", "covered-report-only", "none"},
    {"mlp_gate", "covered-report-only", "none"},
    {"mlp_up", "covered-report-only", "none"},
    {"mlp_down", "covered-report-only", "none"},
    {"output_head", "covered-report-only", "none"},
    {"qwen_linear_attn", "present-report-only", "quant-policy-or-artifact-emitter"},
    {"moe_router", "present-report-only", "quant-policy-or-artifact-emitter"},
    {"moe_experts", "present-report-only", "quant-policy-or-artifact-emitter"},
    {"shared_expert", "present-report-only", "quant-policy-or-artifact-emitter"},
};

static const missing_role_fact gemma_missing_roles[] = {
    {"token_embedding", "covered-report-only", "none"},
    {"attention_q", "covered-report-only", "none"},
    {"attention_k", "covered-report-only", "none"},
    {"attention_v", "covered-report-only", "none"},
    {"attention_o", "covered-report-only", "none"},
    {"attention_q_norm", "present-report-only", "quant-policy-or-artifact-emitter"},
    {"attention_k_norm", "present-report-only", "quant-policy-or-artifact-emitter"},
    {"mlp_gate", "covered-report-only", "none"},
    {"mlp_up", "covered-report-only", "none"},
    {"mlp_down", "covered-report-only", "none"},
    {"output_head", "covered-report-only", "none"},
};

static const missing_role_source_facts missing_source_ready = {
    NULL, NULL, "present", "present", "present", "present", "present",
    "present", "missing-artifact-contract", 12, 0, 0, 4, 0
};
static const missing_role_source_facts missing_source_qwen_absent = {
    NULL, NULL, "present", "present", "missing", "missing", "missing",
    "missing", "missing-qwen-source-path", 0, 12, 0, 0, 4
};
static const missing_role_source_facts missing_source_gemma_absent = {
    NULL, NULL, "present", "present", "missing", "missing", "missing",
    "missing", "missing-gemma-source-path", 0, 12, 0, 0, 4
};
static const missing_role_source_facts missing_source_attention_k = {
    NULL, NULL, "missing", "present", "present", "present", "present",
    "present", "missing-source-role-attention-k", 11, 1, 0, 4, 0
};
static const missing_role_source_facts missing_source_output_ambiguous = {
    NULL, NULL, "present", "ambiguous", "present", "present", "present",
    "present", "ambiguous-source-role-output-head", 11, 0, 1, 4, 0
};
static const missing_role_source_facts missing_source_output_absent = {
    NULL, NULL, "present", "missing", "present", "present", "present",
    "present", "missing-source-role-output-head", 11, 1, 0, 4, 0
};
static const missing_role_source_facts missing_source_metadata_absent = {
    NULL, NULL, "present", "present", "missing", "missing", "missing",
    "missing", "missing-tokenizer-metadata", 12, 0, 0, 0, 4
};

static const missing_role_dynamic_facts missing_dynamic_ready = {
    NULL, NULL, "present-report-only", NULL, "present-report-only", NULL, NULL,
    "quant-policy-or-artifact-emitter", "V010.QUANT.1", "missing-role-report",
    NULL, NULL, NULL
};
static const missing_role_dynamic_facts missing_dynamic_head_absent = {
    NULL, NULL, "present-report-only", NULL, "missing-in-report", NULL, NULL,
    "missing-output-head-map", "V010.MAP.8", "blocked", NULL, NULL, NULL
};
static const missing_role_dynamic_facts missing_dynamic_tensor_incomplete = {
    NULL, NULL, "incomplete-report-only", NULL, "present-report-only", NULL,
    NULL, "incomplete-tensor-map", "V010.MAP.8", "blocked", NULL, NULL, NULL
};

static void missing_role_sidecar_path(const yvex_model_target_request *request,
                                      const char *family,
                                      const char *suffix,
                                      char *out,
                                      size_t cap)
{
    if (!out || cap == 0u) {
        return;
    }
    out[0] = '\0';
    if (request->models_root[0]) {
        (void)snprintf(out, cap, "%s/evidence/build/%s/%s.%s",
                       request->models_root, family, request->target_id,
                       suffix);
    }
}

static void missing_role_build_state(const yvex_model_target_request *request,
                                     const char *family,
                                     missing_role_state *state)
{
    char sidecar[1200];
    char buf[2048];

    if (!request || !family || !state) {
        return;
    }
    memset(state, 0, sizeof(*state));
    yvex_model_target_probe_source_profile(request, family, &state->source);
    missing_role_sidecar_path(request, family, "tensor-map.json",
                              sidecar, sizeof(sidecar));
    if (yvex_model_target_probe_read(sidecar, buf, sizeof(buf))) {
        state->tensor_map_incomplete =
            strstr(buf, "naming-map-incomplete") != NULL ||
            strstr(buf, "required-groups-missing") != NULL;
    }
    missing_role_sidecar_path(request, family, "tokenizer-map.json",
                              sidecar, sizeof(sidecar));
    state->tokenizer_map_present = yvex_model_target_probe_file(sidecar);
    missing_role_sidecar_path(request, family, "output-head-map.json",
                              sidecar, sizeof(sidecar));
    state->output_head_map_present =
        yvex_model_target_probe_read(sidecar, buf, sizeof(buf));
    state->output_head_map_missing =
        state->output_head_map_present &&
        strstr(buf, "output-head-missing") != NULL;
    state->output_head_map_tied =
        state->output_head_map_present &&
        strstr(buf, "tied-output-head-report-only") != NULL;
}

/*
 * Derive the single source-role view shared by normal and audit reports.
 *
 * Absent evidence remains represented by deterministic missing fields.
 */
static missing_role_source_facts missing_role_source_view(
    const char *family,
    const missing_role_state *state,
    const char *target)
{
    const missing_role_source_facts *selected = &missing_source_ready;
    missing_role_source_facts facts;

    if (!state || !state->source.header_present) {
        selected = strcmp(family, "gemma") == 0
                       ? &missing_source_gemma_absent
                       : &missing_source_qwen_absent;
    } else if (!state->source.attention_k_present) {
        selected = &missing_source_attention_k;
    } else if (state->source.output_head_ambiguous) {
        selected = &missing_source_output_ambiguous;
    } else if (!state->source.output_head_present) {
        selected = &missing_source_output_absent;
    } else if (!state->source.metadata_present) {
        selected = &missing_source_metadata_absent;
    }
    facts = *selected;
    facts.family = family;
    facts.target = target;
    return facts;
}

static missing_role_dynamic_facts missing_role_dynamic_view(
    const yvex_model_target_request *request,
    const char *family,
    const missing_role_state *state,
    const yvex_model_target_report *report,
    char tensor_path[1200],
    char artifact_path[1200])
{
    const missing_role_dynamic_facts *selected =
        state->output_head_map_missing
            ? &missing_dynamic_head_absent
            : (state->tensor_map_incomplete
                   ? &missing_dynamic_tensor_incomplete : &missing_dynamic_ready);
    missing_role_dynamic_facts facts = *selected;

    facts.target = report->target_id;
    facts.family = family;
    facts.tokenizer = state->output_head_map_missing || state->tokenizer_map_present
                          ? "present-report-only" : "missing";
    facts.role_status = state->tensor_map_incomplete
                            ? "missing"
                            : (strcmp(family, "qwen") == 0
                                   ? "present" : "not-applicable");
    facts.output_head = state->output_head_map_missing ? "missing" : "present";
    facts.tied_head = state->output_head_map_missing
                          ? "not-proven" : "present-report-only";
    missing_role_sidecar_path(request, family, "tensor-map.json",
                              tensor_path, 1200u);
    (void)snprintf(artifact_path, 1200u, "%s/artifacts/%s/%s.gguf",
                   request->models_root, family, report->target_id);
    facts.tensor_path = tensor_path;
    facts.artifact_path = artifact_path;
    return facts;
}

static const missing_role_fact *missing_role_rows(const char *family,
                                                 unsigned long *count)
{
    if (count) {
        *count = sizeof(qwen_missing_roles) / sizeof(qwen_missing_roles[0]);
    }
    if (family && strcmp(family, "gemma") == 0) {
        if (count) {
            *count = sizeof(gemma_missing_roles) / sizeof(gemma_missing_roles[0]);
        }
        return gemma_missing_roles;
    }
    return qwen_missing_roles;
}

static void missing_role_prepare(const yvex_model_target_request *request,
                                 yvex_model_target_report *report)
{
    const char *target = request->target_id[0] ? request->target_id : "qwen3-8b";
    const char *family = yvex_model_target_family_key(target);
    const yvex_model_target_report_profile profile = {
        .status = "missing-role-report", .target_id = target, .family = family,
        .stage = "report-only", .tensor_map_status = "present-report-only",
        .runtime_status = "unsupported", .generation_status = "unsupported-full-model",
        .benchmark_status = "not-measured", .next_row = "V010.QUANT.1",
        .boundary = "missing-role report only; no artifact emission or runtime execution",
        .reason = "quant-policy-or-artifact-emitter"
    };

    yvex_model_target_report_prepare(report, request, &profile);
}

static int missing_role_validate(const yvex_model_target_request *request,
                                 yvex_model_target_report *report)
{
    const char *target = request->target_id[0] ? request->target_id : "qwen3-8b";

    if (request->kind != YVEX_MODEL_TARGET_COMMAND_TENSOR_MAP &&
        request->kind != YVEX_MODEL_TARGET_COMMAND_MISSING_ROLES) {
        report->status = "missing-role-report-fail";
        report->exit_code = 2;
        yvex_model_target_report_add_error(report,
                                           "missing-role report requires tensor-map command kind");
        return 1;
    }
    if (request->kind == YVEX_MODEL_TARGET_COMMAND_MISSING_ROLES &&
        !request->target_id[0]) {
        report->status = "missing-role-report-fail";
        report->exit_code = 2;
        yvex_model_target_report_add_error(report,
                                           "model-target missing-roles: requires TARGET");
        return 1;
    }
    if (!yvex_model_target_supported_source_target(target)) {
        report->status = "unsupported-target";
        report->exit_code = 2;
        yvex_model_target_report_add_row(report, "status: unsupported-target");
        yvex_model_target_report_add_row(report, "target_id: %s", target);
        yvex_model_target_report_add_error(report, "unsupported target: %s", target);
        return 1;
    }
    return 0;
}

static void missing_role_render_source_table(const char *family,
                                             yvex_model_target_report *report)
{
    yvex_model_target_report_add_row(report, "MISSING ROLE BLOCKER REPORT");
    yvex_model_target_report_add_row(
        report,
        "FAMILY  TARGET  STATUS  OBS_SRC  MISS_SRC  AMBIG_SRC  OBS_META  MISS_META  TOP_BLOCKER  NEXT");
    yvex_model_target_report_add_row(
        report,
        "%s  %s  missing-role-report-blocked  12  0  0  4  0  missing-artifact-contract  V010.MAP.9",
        family, report->target_id);
}

static void missing_role_render_dynamic_table(const missing_role_dynamic_facts *facts,
                                              yvex_model_target_report *report)
{
    yvex_model_target_report_add_row(report,
                                     "TARGET  FAMILY  STATUS  TOP_BLOCKER  PREPARE_BLOCKERS  "
                                     "TOKENIZER  ARTIFACT  NEXT");
    yvex_model_target_report_add_row(report, "%s  %s  blocked  %s  %s  %s  %s  %s",
                                     facts->target, facts->family, facts->top, "3",
                                     facts->tokenizer, "missing", facts->next);
}

static void missing_role_render_audit_facts(const char *family,
                                           const missing_role_state *state,
                                           yvex_model_target_report *report)
{
    const missing_role_fact *rows;
    unsigned long count;
    unsigned long i;

    yvex_model_target_report_add_row(report, "report: missing-runtime-roles");
    yvex_model_target_report_add_row(report, "status: missing-role-report");
    yvex_model_target_report_add_row(report, "target_id: %s", report->target_id);
    yvex_model_target_report_add_row(report, "family: %s", family);
    yvex_model_target_report_add_row(report, "tensor_map_status: present-report-only");
    yvex_model_target_report_add_row(report, "output_head_map_status: present-report-only");
    yvex_model_target_report_add_row(report, "tokenizer_map_status: present-report-only");
    rows = missing_role_rows(family, &count);
    for (i = 0; i < count; ++i) {
        yvex_model_target_report_add_row(report, "role.%lu.name: %s", i, rows[i].name);
        yvex_model_target_report_add_row(report, "role.%lu.status: %s", i, rows[i].status);
        yvex_model_target_report_add_row(report, "role.%lu.blocker: %s", i, rows[i].blocker);
    }
    yvex_model_target_report_add_row(report, "role_group.qwen_linear_attn.status: %s",
                                     strcmp(family, "qwen") == 0
                                         ? "present"
                                         : "not-applicable");
    yvex_model_target_report_add_row(report, "role_group.moe_router.status: %s",
                                     strcmp(family, "qwen") == 0
                                         ? "present"
                                         : "not-applicable");
    yvex_model_target_report_add_row(report, "role_group.moe_experts.status: %s",
                                     strcmp(family, "qwen") == 0
                                         ? "present"
                                         : "not-applicable");
    yvex_model_target_report_add_row(report, "role_group.shared_expert.status: %s",
                                     strcmp(family, "qwen") == 0
                                         ? "present"
                                         : "not-applicable");
    yvex_model_target_report_add_row(report,
                                     "role_group.unknown_tensors.status: unclassified-header-name");
    yvex_model_target_report_add_row(report, "role_group.output_head.status: present");
    yvex_model_target_report_add_row(
        report,
        "role_group.tied_head_policy.status: %s",
        state && state->output_head_map_tied
            ? "tied-output-head-candidate"
            : (state && state->output_head_map_missing ? "not-proven" : "not-applicable"));
    yvex_model_target_report_add_row(report, "top_blocker: quant-policy-or-artifact-emitter");
    yvex_model_target_report_add_row(report, "next: V010.QUANT.1");
    yvex_model_target_report_common_tail(report);
}

static void missing_role_render_source_audit(const missing_role_source_facts *facts,
                                             yvex_model_target_report *report)
{
    yvex_model_target_report_project_rows(
        report, missing_role_source_rows,
        sizeof(missing_role_source_rows) / sizeof(missing_role_source_rows[0]),
        facts);
    if (strcmp(facts->top, "missing-source-role-attention-k") == 0) {
        yvex_model_target_report_add_row(report, "missing_role.entry.0.role: attention_k");
        yvex_model_target_report_add_row(report, "missing_role.entry.0.blocker_class: source-role-missing");
    } else if (strcmp(facts->top, "missing-source-role-output-head") == 0) {
        yvex_model_target_report_add_row(report, "missing_role.entry.0.role: output_head");
        yvex_model_target_report_add_row(report, "missing_role.entry.0.blocker_class: source-role-missing");
    } else if (strcmp(facts->top, "ambiguous-source-role-output-head") == 0) {
        yvex_model_target_report_add_row(report, "missing_role.entry.0.role: output_head");
        yvex_model_target_report_add_row(report, "missing_role.entry.0.blocker_class: source-role-ambiguous");
    }
    yvex_model_target_report_common_tail(report);
}

static void missing_role_render_dynamic_audit(const missing_role_dynamic_facts *facts,
                                              const missing_role_state *state,
                                              yvex_model_target_report *report)
{
    yvex_model_target_report_project_rows(
        report, missing_role_dynamic_prefix,
        sizeof(missing_role_dynamic_prefix) / sizeof(missing_role_dynamic_prefix[0]),
        facts);
    if (state && state->tensor_map_incomplete) {
        yvex_model_target_report_add_row(report, "tensor_map_unmapped_unknown_count: 1");
    }
    yvex_model_target_report_project_rows(
        report, missing_role_dynamic_suffix,
        sizeof(missing_role_dynamic_suffix) / sizeof(missing_role_dynamic_suffix[0]),
        facts);
}

int yvex_missing_role_report_build(const yvex_model_target_request *request,
                                   yvex_model_target_report *report,
                                   yvex_error *err)
{
    const char *family;
    missing_role_state state;
    missing_role_dynamic_facts dynamic;
    char tensor_path[1200];
    char artifact_path[1200];
    int source_style;

    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "missing_role_report",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    missing_role_prepare(request, report);
    if (missing_role_validate(request, report)) {
        return YVEX_OK;
    }
    if (yvex_source_is_release_target(request->target_id)) {
        return yvex_model_target_report_release_coverage(
            request, report, "missing-roles", "missing_role_report",
            "no-missing-source-tensor-requirements",
            "zero missing source requirements and mapping is complete; payload and all "
            "higher capabilities remain blocked",
            err);
    }
    family = report->family[0] ? report->family : "qwen";
    missing_role_build_state(request, family, &state);
    source_style = request->source_path[0] ||
                   request->kind == YVEX_MODEL_TARGET_COMMAND_TENSOR_MAP;
    if (source_style) {
        missing_role_source_facts facts =
            missing_role_source_view(family, &state, report->target_id);

        if (request->output_contract[0]) {
            yvex_model_target_report_add_output_contract(
                report, "missing-roles", request->output_contract);
            return YVEX_OK;
        }
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
            missing_role_render_source_table(family, report);
            return YVEX_OK;
        }
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
            missing_role_render_source_audit(&facts, report);
            return YVEX_OK;
        }
        yvex_model_target_report_add_row(report, "missing-roles: %s [blocked]",
                                         report->target_id);
        yvex_model_target_report_add_row(report,
                                         "family: %s  evidence: header+sidecar-only",
                                         family);
        yvex_model_target_report_add_row(
            report, "source_roles: %d/12 present, %d missing, %d ambiguous",
            facts.source_observed, facts.source_missing, facts.source_ambiguous);
        yvex_model_target_report_add_row(
            report, "metadata_roles: %d/4 present, %d missing, 0 ambiguous",
            facts.metadata_observed, facts.metadata_missing);
        if (!state.source.attention_k_present && state.source.header_present) {
            yvex_model_target_report_add_row(report, "missing_source: attention_k");
        }
        if (strcmp(facts.output_head, "missing") == 0 &&
            state.source.header_present) {
            yvex_model_target_report_add_row(report, "missing_source: output_head");
        }
        if (!state.source.metadata_present && state.source.header_present) {
            yvex_model_target_report_add_row(
                report,
                "missing_metadata: tokenizer_metadata,config_metadata,generation_metadata,special_tokens");
        }
        yvex_model_target_report_add_row(report, "top_blocker: %s", facts.top);
        yvex_model_target_report_add_row(report, "next: V010.MAP.9");
        yvex_model_target_report_add_row(report,
                                         "boundary: report-only; use --audit for role details");
        return YVEX_OK;
    }

    dynamic = missing_role_dynamic_view(
        request, family, &state, report, tensor_path, artifact_path);
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        missing_role_render_dynamic_table(&dynamic, report);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        if (state.tensor_map_incomplete || state.output_head_map_missing) {
            missing_role_render_dynamic_audit(&dynamic, &state, report);
        } else {
            missing_role_render_audit_facts(family, &state, report);
        }
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        yvex_model_target_report_add_row(
            report,
            "{\"status\":\"missing-role-report\",\"target_id\":\"%s\","
            "\"top_blocker\":\"%s\",\"qwen_linear_attn\":\"%s\","
            "\"shared_expert\":\"%s\",\"tokenizer\":\"%s\","
            "\"next\":\"%s\"}",
            report->target_id,
            dynamic.top,
            state.tensor_map_incomplete ? "missing" : "present",
            state.tensor_map_incomplete ? "missing" : "present",
            state.tokenizer_map_present ? "present-report-only" : "missing",
            dynamic.next);
        return YVEX_OK;
    }
    yvex_model_target_report_add_row(report, "missing-roles: %s", report->target_id);
    yvex_model_target_report_project_rows(
        report, missing_role_normal_prefix,
        sizeof(missing_role_normal_prefix) / sizeof(missing_role_normal_prefix[0]),
        &dynamic);
    yvex_model_target_report_add_row(report, "%s-linear-attn: %s",
                                     strcmp(family, "qwen") == 0 ? "qwen" : "gemma",
                                     dynamic.role_status);
    yvex_model_target_report_project_rows(
        report, missing_role_normal_suffix,
        sizeof(missing_role_normal_suffix) / sizeof(missing_role_normal_suffix[0]),
        &dynamic);
    if (strcmp(dynamic.top, "quant-policy-or-artifact-emitter") == 0) {
        yvex_model_target_report_common_tail(report);
    } else {
        yvex_model_target_report_add_row(report,
                                         "boundary: missing-role report only; no GGUF/runtime/generation");
    }
    return YVEX_OK;
}
