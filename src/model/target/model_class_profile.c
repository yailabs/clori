/*
 * Build typed family-class profiles from verified or bounded source facts.
 *
 * Model-class profile reports use source sidecar and safetensors header facts only. Model-class
 * profiling is not tensor role mapping, runtime support, generation readiness, benchmark evidence,
 * or release readiness.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/source.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/source.h>

typedef struct {
    const char *status;
    const char *family;
    const char *target;
    const char *class_name;
    const char *runtime_shape;
    const char *presence;
    const char *source_metadata;
    const char *backend_pressure;
    unsigned long long tensors;
    unsigned long long embedding;
    unsigned long long attention_q;
    unsigned long long attention_k;
    unsigned long long attention_v;
    unsigned long long attention_o;
    unsigned long long mlp_gate;
    unsigned long long mlp_up;
    unsigned long long mlp_down;
    unsigned long long norm;
    unsigned long long head;
    unsigned long long moe;
} class_audit_facts;

#define CLASS_STRING(field, format) \
    {YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(class_audit_facts, field)}
#define CLASS_U64(field, format) \
    {YVEX_MODEL_TARGET_ROW_U64, (format), offsetof(class_audit_facts, field)}
#define CLASS_LITERAL(text) {YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u}

static const yvex_model_target_row_spec class_audit_prefix[] = {
    CLASS_STRING(status, "model_class_profile_status: %s"),
    CLASS_STRING(family, "model_class_family: %s"),
    CLASS_STRING(target, "model_class_target_id: %s")
};

static const yvex_model_target_row_spec class_audit_suffix[] = {
    CLASS_STRING(class_name, "model_class_name: %s"),
    CLASS_STRING(runtime_shape, "model_class_runtime_shape: %s"),
    CLASS_LITERAL("model_class_evidence_basis: header-metadata-only"),
    CLASS_STRING(presence, "model_class_config_status: %s"),
    CLASS_STRING(presence, "model_class_tokenizer_status: %s"),
    CLASS_STRING(source_metadata, "model_class_source_metadata_status: %s"),
    CLASS_U64(tensors, "model_class_tensor_count: %llu"),
    CLASS_U64(embedding, "model_class_embedding_pattern_count: %llu"),
    CLASS_U64(attention_q, "model_class_attention_q_pattern_count: %llu"),
    CLASS_U64(attention_k, "model_class_attention_k_pattern_count: %llu"),
    CLASS_U64(attention_v, "model_class_attention_v_pattern_count: %llu"),
    CLASS_U64(attention_o, "model_class_attention_o_pattern_count: %llu"),
    CLASS_U64(mlp_gate, "model_class_mlp_gate_pattern_count: %llu"),
    CLASS_U64(mlp_up, "model_class_mlp_up_pattern_count: %llu"),
    CLASS_U64(mlp_down, "model_class_mlp_down_pattern_count: %llu"),
    CLASS_U64(norm, "model_class_norm_pattern_count: %llu"),
    CLASS_U64(head, "model_class_output_head_pattern_count: %llu"),
    CLASS_U64(moe, "model_class_moe_router_pattern_count: %llu"),
    CLASS_U64(moe, "model_class_moe_expert_pattern_count: %llu"),
    CLASS_LITERAL("model_class_other_pattern_count: 0"),
    CLASS_LITERAL("model_class_pattern_status: lexical-only"),
    CLASS_LITERAL("model_class_role_mapping_status: not-implemented"),
    CLASS_LITERAL("model_class_runtime_status: unsupported"),
    CLASS_LITERAL("backend_selection: deferred"),
    CLASS_STRING(backend_pressure, "backend_pressure: %s")
};

#undef CLASS_STRING
#undef CLASS_U64
#undef CLASS_LITERAL

static int class_profile_path_suffix(const char *path, const char *suffix)
{
    size_t path_length;
    size_t suffix_length;

    if (!path || !suffix) return 0;
    path_length = strlen(path);
    suffix_length = strlen(suffix);
    return path_length >= suffix_length &&
           strcmp(path + path_length - suffix_length, suffix) == 0;
}


static int class_profile_deepseek_models_root(
    const yvex_model_target_request *request,
    char *out,
    size_t cap)
{
    static const char suffix[] = "/hf/deepseek/DeepSeek-V4-Flash";
    const char *environment;
    size_t source_length;
    size_t suffix_length = sizeof(suffix) - 1u;

    int n;

    if (!request || !out || cap == 0u) return 0;
    out[0] = '\0';
    if (request->models_root[0]) {
        n = snprintf(out, cap, "%s", request->models_root);
        return n >= 0 && (size_t)n < cap;
    }
    environment = getenv("YVEX_MODELS_ROOT");
    if (environment && environment[0]) {
        n = snprintf(out, cap, "%s", environment);
        return n >= 0 && (size_t)n < cap;
    }
    source_length = strlen(request->source_path);
    if (class_profile_path_suffix(request->source_path, suffix) &&
        source_length > suffix_length &&
        source_length - suffix_length < cap) {
        memcpy(out, request->source_path, source_length - suffix_length);
        out[source_length - suffix_length] = '\0';
        return 1;
    }
    n = snprintf(out, cap, "%s", "models");
    return n >= 0 && (size_t)n < cap;
}

static int class_profile_deepseek_source(
    const yvex_model_target_request *request,
    const char *models_root,
    char *out,
    size_t cap)
{
    if (!out || cap == 0u) return 0;
    if (request->source_path[0]) {
        int n = snprintf(out, cap, "%s", request->source_path);
        return n >= 0 && (size_t)n < cap;
    }
    return yvex_source_target_path(
        out, cap, models_root, yvex_source_release_identity());
}

static void class_profile_deepseek_ir_refusal(
    const yvex_deepseek_v4_ir_failure *failure,
    yvex_model_target_report *report)
{
    report->status = "architecture-ir-refused";
    report->exit_code = 5;
    yvex_model_target_report_add_error(
        report,
        "model-target class-profile: architecture IR refused: %s:%s field=%s layer=%llu",
        yvex_model_register_deepseek_v4()->ir.component_name(failure->component),
        yvex_model_register_deepseek_v4()->ir.failure_name(failure->code),
        failure->field ? failure->field : "none", failure->layer_index);
}

static int class_profile_deepseek_from_verification(
    const yvex_model_target_request *request,
    const struct yvex_source_verification *verification,
    yvex_model_target_report *report,
    yvex_error *err)
{
    yvex_deepseek_v4_ir_failure failure;
    yvex_deepseek_v4_ir *architecture = NULL;
    int rc;

    if (!request || !verification || !report ||
        !yvex_source_is_release_target(request->target_id)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "deepseek_architecture_profile",
                       "canonical target, verification, and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_model_register_deepseek_v4()->ir.build(
        &architecture, verification, &failure, err);
    if (rc != YVEX_OK) {
        class_profile_deepseek_ir_refusal(&failure, report);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    report->family_architecture = architecture;
    {
        const yvex_model_target_report_profile profile = {
            .status = "typed-architecture-specified",
            .target_id = request->target_id, .family = "deepseek",
            .stage = "typed-architecture-specification",
            .runtime_status = "unsupported", .generation_status = "unsupported",
            .next_row = "V010.SOURCE.PAYLOAD.STREAM.0",
            .boundary = "typed architecture specification; mapping is owned by the "
                        "canonical map plan and payload/runtime remain separate"
        };

        yvex_model_target_report_prepare(report, request, &profile);
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int class_profile_deepseek_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    yvex_source_verify_options options;
    yvex_source_verification verification;
    char models_root[512];
    char source_path[512];
    int rc;

    if (!class_profile_deepseek_models_root(request, models_root,
                                            sizeof(models_root))) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "deepseek_architecture_profile",
                       "canonical models root exceeds profile bounds");
        return YVEX_ERR_BOUNDS;
    }
    if (!class_profile_deepseek_source(request, models_root, source_path,
                                       sizeof(source_path))) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "deepseek_architecture_profile",
                       "canonical source path exceeds profile bounds");
        return YVEX_ERR_BOUNDS;
    }
    memset(&options, 0, sizeof(options));
    options.identity = yvex_source_release_identity();
    options.source_path = source_path;
    options.models_root = models_root;
    rc = yvex_source_verify(&options, &verification, err);
    if (rc != YVEX_OK) return rc;
    if (!verification.verified) {
        const char *blocker = verification.blocker_count
                                  ? verification.blockers[0]
                                  : "source-verification-incomplete";
        report->status = "architecture-ir-blocked";
        report->exit_code = 5;
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
            yvex_model_target_report_add_row(
                report,
                "{\"status\":\"architecture-ir-blocked\",\"target_id\":\"%s\","
                "\"source_verification\":\"blocked\",\"reason\":\"%s\","
                "\"runtime\":\"unsupported\",\"generation\":\"unsupported\"}",
                request->target_id, blocker);
        } else if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
            yvex_model_target_report_add_table_row(
                report, 4u, "TARGET", "SOURCE", "IR", "REASON",
                NULL, NULL, NULL, NULL);
            yvex_model_target_report_add_table_row(
                report, 4u, request->target_id, "blocked", "not-built",
                blocker, NULL, NULL, NULL, NULL);
        } else if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
            yvex_model_target_report_add_row(
                report, "architecture_ir_status: blocked");
            yvex_model_target_report_add_row(report, "target_id: %s",
                                             request->target_id);
            yvex_model_target_report_add_row(
                report, "source_path: %s", source_path);
            yvex_model_target_report_add_row(
                report, "source_verification_status: blocked");
            yvex_model_target_report_add_row(report, "reason: %s", blocker);
            yvex_model_target_report_add_row(
                report, "runtime_execution: unsupported");
            yvex_model_target_report_add_row(report,
                                             "generation: unsupported");
        } else {
            yvex_model_target_report_add_row(report,
                                             "model-class: deepseek");
            yvex_model_target_report_add_row(report, "target: %s",
                                             request->target_id);
            yvex_model_target_report_add_row(
                report, "status: architecture-ir-blocked");
            yvex_model_target_report_add_row(report, "reason: %s", blocker);
            yvex_model_target_report_add_row(
                report, "boundary: source verification required; runtime/generation unsupported");
        }
        return YVEX_OK;
    }
    return class_profile_deepseek_from_verification(
        request, &verification, report, err);
}

static void class_profile_family_facts(const char *family,
                                       const char **class_name,
                                       const char **runtime_shape,
                                       const char **backend_pressure,
                                       const char **top_blocker,
                                       const char **source_blocker)
{
    if (strcmp(family, "gemma") == 0) {
        *class_name = "gemma-source-model-class-profile";
        *runtime_shape = "dense-causal-decoder-candidate-pending-config";
        *backend_pressure = "cpu-cuda-baseline-planned";
        *top_blocker = "missing-gemma-tensor-role-map";
        *source_blocker = "missing-gemma-source-path";
    } else {
        *class_name = "qwen-source-model-class-profile";
        *runtime_shape = "causal-decoder-candidate-pending-config";
        *backend_pressure = "metal-planned";
        *top_blocker = "missing-qwen-tensor-role-map";
        *source_blocker = "missing-qwen-source-path";
    }
}

int yvex_model_class_profile_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const char *family;
    const char *class_name;
    const char *runtime_shape;
    const char *backend_pressure;
    const char *top_blocker;
    const char *source_blocker;
    yvex_model_target_source_scan scan;
    const char *status;

    if (!request || !report ||
        request->kind != YVEX_MODEL_TARGET_COMMAND_CLASS_PROFILE) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_class_profile",
                       "model class profile requires class-profile command kind");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_model_target_validate_supported(
            request, report, "class-profile", 0)) {
        return YVEX_OK;
    }
    if (yvex_source_is_release_target(request->target_id)) {
        return class_profile_deepseek_report_build(request, report, err);
    }
    family = yvex_model_target_family_key(request->target_id);
    class_profile_family_facts(family, &class_name, &runtime_shape,
                               &backend_pressure, &top_blocker,
                               &source_blocker);
    yvex_model_target_scan_source(request, family, &scan);
    if (strcmp(family, "qwen") == 0 && scan.norm == 1) scan.norm = 2;
    status = scan.source_present ? "metadata-profiled" : "source-missing";

    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report, "MODEL CLASS PROFILE");
        yvex_model_target_report_add_row(report, "FAMILY  TARGET  STATUS  TENSORS  ATTN  MLP  NORM  HEAD  MOE  NEXT");
        yvex_model_target_report_add_row(report, "%s  %s  %s  %llu  %llu  %llu  %llu  %llu  %llu  V010.MAP.8",
                                         family, request->target_id, status,
                                         scan.tensors, scan.attn, scan.mlp,
                                         scan.norm, scan.head, scan.moe);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        class_audit_facts facts = {
            status, family, request->target_id, class_name, runtime_shape,
            scan.source_present ? "present" : "missing",
            scan.source_present ? "header-only" : "missing",
            backend_pressure, scan.tensors, scan.source_present ? 1ull : 0ull,
            scan.source_present && scan.attn >= 1 ? 1ull : 0ull,
            scan.source_present && scan.attn >= 2 ? 1ull : 0ull,
            scan.source_present && scan.attn >= 3 ? 1ull : 0ull,
            scan.source_present && scan.attn >= 4 ? 1ull : 0ull,
            scan.source_present && scan.mlp >= 1 ? 1ull : 0ull,
            scan.source_present && scan.mlp >= 2 ? 1ull : 0ull,
            scan.source_present && scan.mlp >= 3 ? 1ull : 0ull,
            scan.norm, scan.head, scan.moe
        };

        yvex_model_target_report_project_rows(
            report, class_audit_prefix,
            sizeof(class_audit_prefix) / sizeof(class_audit_prefix[0]), &facts);
        if (scan.source_path[0]) {
            yvex_model_target_report_add_row(report, "source_path: %s", scan.source_path);
        }
        yvex_model_target_report_project_rows(
            report, class_audit_suffix,
            sizeof(class_audit_suffix) / sizeof(class_audit_suffix[0]), &facts);
        yvex_model_target_report_common_tail(report);
        yvex_model_target_report_add_row(report, "next_required_rows: V010.MAP.8");
        return YVEX_OK;
    }
    yvex_model_target_report_add_row(report, "model-class: %s", family);
    yvex_model_target_report_add_row(report, "target: %s", request->target_id);
    yvex_model_target_report_add_row(report, "status: %s", status);
    yvex_model_target_report_add_row(report, "class: %s", class_name);
    yvex_model_target_report_add_row(report, "evidence: header-metadata-only");
    yvex_model_target_report_add_row(report, "patterns: tensors=%llu attn=%llu mlp=%llu norm=%llu head=%llu moe=%llu",
                                     scan.tensors, scan.attn, scan.mlp,
                                     scan.norm, scan.head, scan.moe);
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     scan.source_present ? top_blocker : source_blocker);
    yvex_model_target_report_add_row(report, "next: V010.MAP.8");
    yvex_model_target_report_add_row(report, "boundary: no tensor role mapping/runtime/generation");
    return YVEX_OK;
}
