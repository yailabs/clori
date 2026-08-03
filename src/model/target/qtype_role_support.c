/*
 * Project canonical qtype capability facts by tensor role.
 *
 * Role-support matrices are report-only facts and hand off incomplete quantization work to later
 * rows. Qtype role-support reporting is not qtype support completion, quantization, artifact
 * emission, runtime readiness, generation readiness, benchmark evidence, or release readiness.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/quant_numeric.h>

#include <stdio.h>
#include <string.h>

typedef struct {
    const char *role_name;
    const char *source_dtype;
    const char *storage_status;
    const char *blocker;
} qtype_role_fact;

typedef struct {
    const char *family;
    const char *target_id;
    const char *status;
    const char *top_blocker;
    const char *next_row;
} qtype_gate_family_fact;

typedef struct {
    const char *target;
    const char *family;
    const char *source_dtype;
    const char *top_blocker;
    const char *next_row;
    unsigned long role_count;
} qtype_role_summary;

#define QTYPE_ROLE_LITERAL(text) \
    { YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u }
#define QTYPE_ROLE_STRING(field, format) \
    { YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(qtype_role_summary, field) }
#define QTYPE_ROLE_ULONG(field, format) \
    { YVEX_MODEL_TARGET_ROW_ULONG, (format), offsetof(qtype_role_summary, field) }

static const yvex_model_target_row_spec qtype_role_summary_rows[] = {
    QTYPE_ROLE_STRING(target, "qtype-role-support: %s"),
    QTYPE_ROLE_LITERAL("status: blocked"),
    QTYPE_ROLE_STRING(family, "family: %s"),
    QTYPE_ROLE_STRING(source_dtype, "source_dtype: %s"),
    QTYPE_ROLE_LITERAL("preferred_artifact_qtype: unresolved"),
    QTYPE_ROLE_ULONG(role_count, "supported_roles: %lu"),
    QTYPE_ROLE_ULONG(role_count, "blocked_roles: %lu"),
    QTYPE_ROLE_STRING(top_blocker, "top_blocker: %s"),
    QTYPE_ROLE_STRING(next_row, "next: %s"),
    QTYPE_ROLE_LITERAL(
        "boundary: qtype role report only; no quantization/GGUF/runtime/generation")
};

static const char *const qtype_gate_normal_rows[] = {
    "qtype-role-support-gate: v0.1.0",
    "status: qtype-role-support-gate-blocked",
    "family_count: 3",
    "top_blocker: artifact-materialization-unimplemented",
    "next: V010.ARTIFACT.MATERIALIZE.0"
};

static const char *const qtype_gate_audit_prefix[] = {
    "report: qtype-role-support-gate",
    "status: qtype-role-support-gate-blocked",
    "release: v0.1.0"
};

static const char *const qtype_role_table_prefix[] = {
    "QTYPE ROLE SUPPORT",
    "ROLE  SRC_DTYPE  ARTIFACT_QTYPE  STORAGE  COMPUTE  CALIBRATION  STATUS"
};

static const char *const qtype_role_audit_suffix[] = {
    "payload_bytes_read: false",
    "quantization_performed: false",
    "gguf_emitted: false"
};

static const char *const qtype_role_selected_rows[] = {
    "selected_slice_evidence_only: true",
    "full_family_artifact_status: missing"
};

static const yvex_model_target_request_rules qtype_role_rules = {
    YVEX_MODEL_TARGET_COMMAND_QUANT_POLICY,
    "qtype-role-support-fail",
    "qtype role-support report requires quant-policy command kind",
    NULL,
    1
};

#undef QTYPE_ROLE_LITERAL
#undef QTYPE_ROLE_STRING
#undef QTYPE_ROLE_ULONG

static const qtype_role_fact qwen_role_facts[] = {
    {"token_embedding", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_q", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_k", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_v", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_o", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"qwen_linear_attn_A_log", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"moe_expert_gate_up", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"output_head", "BF16", "source-native", "artifact-emitter-missing"},
    {"tokenizer_metadata", "metadata", "metadata-sidecar", "artifact-emitter-missing"},
};

static const qtype_role_fact gemma_role_facts[] = {
    {"token_embedding", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_q_norm", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_k_norm", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_q", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_k", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_v", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"attention_o", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"pre_feedforward_layernorm", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"layer_scalar", "BF16", "source-native", "family-quantization-plan-unimplemented"},
    {"output_head_tied_embedding", "BF16", "source-native", "artifact-emitter-missing"},
    {"tokenizer_metadata", "metadata", "metadata-sidecar", "artifact-emitter-missing"},
};

static const qtype_gate_family_fact qtype_gate_rows[] = {
    {"deepseek", "deepseek4-v4-flash-dspark", "blocked",
     "artifact-materialization-unimplemented", "V010.ARTIFACT.MATERIALIZE.0"},
    {"qwen", "qwen3-6-35b-a3b", "blocked", "family-quantization-plan-unimplemented", "not-scheduled"},
    {"gemma", "gemma-4-31b-it", "blocked", "family-quantization-plan-unimplemented", "not-scheduled"},
};

static const char *qtype_role_compute_status(const char *source_dtype)
{
    const yvex_quant_numeric_capability *capability;
    unsigned int qtype;

    if (source_dtype && strcmp(source_dtype, "metadata") == 0)
        return "not-applicable";
    if (source_dtype && strcmp(source_dtype, "F32") == 0)
        qtype = YVEX_GGUF_QTYPE_F32;
    else if (source_dtype && strcmp(source_dtype, "F16") == 0)
        qtype = YVEX_GGUF_QTYPE_F16;
    else if (source_dtype && strcmp(source_dtype, "BF16") == 0)
        qtype = YVEX_GGUF_QTYPE_BF16;
    else
        return "unresolved-source-dtype";
    capability = yvex_quant_numeric_capability_at(qtype);
    return capability && capability->dedicated_cpu_compute_available &&
           capability->dedicated_cuda_compute_available
        ? "cpu-cuda-available" : "unavailable";
}

static const qtype_role_fact *qtype_role_rows(const char *family,
                                             unsigned long *count)
{
    if (family && strcmp(family, "gemma") == 0) {
        if (count) {
            *count = sizeof(gemma_role_facts) / sizeof(gemma_role_facts[0]);
        }
        return gemma_role_facts;
    }
    if (count) {
        *count = sizeof(qwen_role_facts) / sizeof(qwen_role_facts[0]);
    }
    return qwen_role_facts;
}

static void qtype_role_prepare(const yvex_model_target_request *request,
                               yvex_model_target_report *report)
{
    const char *target = request->target_id[0] ? request->target_id : "qwen3-8b";
    const char *family = yvex_model_target_family_key(target);
    int deepseek = strcmp(family, "deepseek") == 0;
    const yvex_model_target_report_profile profile = {
        .status = request->gate[0] ? "qtype-role-support-gate-blocked"
                                   : "qtype-role-support-blocked",
        .target_id = target, .family = family, .stage = "report-only",
        .qtype_policy_status = "blocked", .artifact_status = "missing",
        .runtime_status = "unsupported", .generation_status = "unsupported-full-model",
        .benchmark_status = "not-measured",
        .next_row = deepseek ? "V010.ARTIFACT.MATERIALIZE.0" : "not-scheduled",
        .reason = deepseek ? "complete-artifact-admission-required"
                           : "family-quantization-plan-unimplemented",
        .boundary = "qtype role-support report only; no quantization or artifact emission"
    };

    yvex_model_target_report_prepare(report, request, &profile);
}

static int qtype_role_validate(const yvex_model_target_request *request,
                               yvex_model_target_report *report)
{
    if (!yvex_model_target_validate_request_shape(
            request, report, &qtype_role_rules, request->gate)) {
        return 1;
    }
    return 0;
}

static void qtype_gate_add_table(yvex_model_target_report *report)
{
    unsigned long i;

    yvex_model_target_report_add_row(report, "QTYPE ROLE SUPPORT GATE");
    yvex_model_target_report_add_table_row(report, 7u, "FAMILY", "TARGET",
                                           "STATUS", "ROLES", "BLOCKED",
                                           "TOP_BLOCKER", "NEXT", NULL);
    for (i = 0; i < sizeof(qtype_gate_rows) / sizeof(qtype_gate_rows[0]); ++i) {
        yvex_model_target_report_add_table_row(report, 7u, qtype_gate_rows[i].family,
                                               qtype_gate_rows[i].target_id,
                                               qtype_gate_rows[i].status,
                                               "1",
                                               "1",
                                               qtype_gate_rows[i].top_blocker,
                                               qtype_gate_rows[i].next_row,
                                               NULL);
    }
}

static void qtype_gate_add_audit(yvex_model_target_report *report)
{
    unsigned long i;

    yvex_model_target_report_add_rows(
        report, qtype_gate_audit_prefix,
        sizeof(qtype_gate_audit_prefix) / sizeof(qtype_gate_audit_prefix[0]));
    for (i = 0; i < sizeof(qtype_gate_rows) / sizeof(qtype_gate_rows[0]); ++i) {
        yvex_model_target_report_add_row(report, "family.%lu.name: %s", i,
                                         qtype_gate_rows[i].family);
        yvex_model_target_report_add_row(report, "family.%lu.target_id: %s", i,
                                         qtype_gate_rows[i].target_id);
        yvex_model_target_report_add_row(report, "family.%lu.status: %s", i,
                                         qtype_gate_rows[i].status);
        yvex_model_target_report_add_row(report, "family.%lu.top_blocker: %s", i,
                                         qtype_gate_rows[i].top_blocker);
        yvex_model_target_report_add_row(report, "family.%lu.next: %s", i,
                                         qtype_gate_rows[i].next_row);
    }
    yvex_model_target_report_common_tail(report);
}

static void qtype_role_add_table(const char *family,
                                 yvex_model_target_report *report)
{
    const qtype_role_fact *rows;
    unsigned long count;
    unsigned long i;

    yvex_model_target_report_add_rows(
        report, qtype_role_table_prefix,
        sizeof(qtype_role_table_prefix) / sizeof(qtype_role_table_prefix[0]));
    rows = qtype_role_rows(family, &count);
    for (i = 0; i < count; ++i) {
        yvex_model_target_report_add_row(report,
                                         "%s  %s  unresolved  header-storage-profiled  %s  deferred  present",
                                         rows[i].role_name,
                                         rows[i].source_dtype,
                                         qtype_role_compute_status(
                                             rows[i].source_dtype));
    }
}

static void qtype_role_add_audit(const char *family,
                                 yvex_model_target_report *report)
{
    const qtype_role_fact *rows;
    unsigned long count;
    unsigned long i;
    int selected_slice = strcmp(family, "deepseek") == 0;

    yvex_model_target_report_add_row(report, "report: qtype-role-support");
    yvex_model_target_report_add_row(report, "status: qtype-role-support-blocked");
    yvex_model_target_report_add_row(report, "target_id: %s", report->target_id);
    yvex_model_target_report_add_row(report, "family: %s", family);
    yvex_model_target_report_add_row(report, "source_dtype: %s",
                                     selected_slice ? "selected-slice" : "BF16");
    if (selected_slice) {
        yvex_model_target_report_add_rows(report, qtype_role_selected_rows, 2u);
    }
    rows = qtype_role_rows(family, &count);
    for (i = 0; i < count; ++i) {
        yvex_model_target_report_add_row(report, "role.%lu.role_name: %s", i,
                                         rows[i].role_name);
        yvex_model_target_report_add_row(
            report, "role.%lu.source_dtype: %s", i,
            selected_slice ? "selected-slice" : rows[i].source_dtype);
        if (selected_slice) {
            yvex_model_target_report_add_row(
                report,
                "role.%lu.role_status: selected-slice-evidence-only", i);
        }
        yvex_model_target_report_add_row(report,
                                         "role.%lu.compute_support_status: %s", i,
                                         qtype_role_compute_status(
                                             rows[i].source_dtype));
        yvex_model_target_report_add_row(report,
                                         "role.%lu.artifact_emission_allowed: false", i);
        yvex_model_target_report_add_row(report,
                                         "role.%lu.artifact_emission_blocker: %s", i,
                                         selected_slice
                                             ? "complete-artifact-admission-required"
                                             : rows[i].blocker);
    }
    yvex_model_target_report_add_rows(
        report, qtype_role_audit_suffix,
        sizeof(qtype_role_audit_suffix) / sizeof(qtype_role_audit_suffix[0]));
    yvex_model_target_report_common_tail(report);
}

int yvex_qtype_role_support_report_build(const yvex_model_target_request *request,
                                         yvex_model_target_report *report,
                                         yvex_error *err)
{
    const char *family;
    unsigned long role_count = 0;

    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "qtype_role_support_report",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    qtype_role_prepare(request, report);
    if (qtype_role_validate(request, report)) {
        return YVEX_OK;
    }
    family = report->family[0] ? report->family : "qwen";
    (void)qtype_role_rows(family, &role_count);
    if (request->gate[0]) {
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
            qtype_gate_add_table(report);
        } else if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
            qtype_gate_add_audit(report);
        } else {
            yvex_model_target_report_add_rows(
                report, qtype_gate_normal_rows,
                sizeof(qtype_gate_normal_rows) / sizeof(qtype_gate_normal_rows[0]));
            yvex_model_target_report_common_tail(report);
        }
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        qtype_role_add_table(family, report);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        qtype_role_add_audit(family, report);
        return YVEX_OK;
    }
    {
        int deepseek = strcmp(family, "deepseek") == 0;
        const qtype_role_summary summary = {
            report->target_id, family, deepseek ? "selected-slice" : "BF16",
            deepseek ? "complete-artifact-admission-required"
                     : "family-quantization-plan-unimplemented",
            deepseek ? "V010.ARTIFACT.MATERIALIZE.0" : "not-scheduled",
            role_count
        };

        yvex_model_target_report_project_rows(
            report, qtype_role_summary_rows,
            sizeof(qtype_role_summary_rows) / sizeof(qtype_role_summary_rows[0]),
            &summary);
    }
    return YVEX_OK;
}
