/* Owner: src/cli/model_artifacts
 * Owns: existing moe command-family parsing and output behavior.
 * Does not own: runtime implementation, graph execution, backend algorithms, artifact emission, eval, benchmark, or
 *   release claims.
 * Invariants: CLI-only and excluded from libyvex.a; preserves existing command behavior.
 * Boundary: moe reports are report-only diagnostics.
 * Purpose: provide existing moe command-family parsing and output behavior.
 * Inputs: typed domain facts, requested output mode, and caller-owned render state.
 * Effects: formats admitted facts through CLI I/O without changing domain state.
 * Failure: formatting or I/O refusal cannot alter capability facts. */
#include "src/cli/model_artifacts/private.h"

#include <string.h>

static const char *const literal_pair_0[] = { "\nmoe report:",
    "  classifies the model as MoE/source-only/unsupported-family and reports router facts, expert tensor-"
        "role facts, shared-expert facts, storage and residency pressure, blockers, and next rows."
};

static const char *const literal_pair_1[] = { "cleanup_attempted: false",
    "cleanup_status: not-needed"};

static const char *const literal_lines_0[] = {
    "  report-only boundary: it does not execute router logits, does not perform top-k routing, does not "
        "activate experts, does not dispatch experts, does not accumulate expert outputs, does not integrate "
        "MoE into graph/prefill/decode, does not generate, and does not benchmark.",
    "  selected-runtime-slice targets may return ok-partial when the family is MoE but router or expert "
        "tensors are not present in the selected artifact.",
    "  source-only targets are reported without opening huge source shards or downloading artifacts.",
    "Boundary: no MoE runtime execution, no full transformer block, no full transformer prefill, no "
        "attention-backed KV writes, no decode, no logits, no vocabulary sampling, no generation, no serving, "
        "no eval, no benchmark, no throughput."
};

typedef struct {
    const char *model;
    const char *backend;
    const char *family;
    const char *registry_path;
    int include_tensors;
    int include_residency;
    int include_blockers;
    yvex_models_output_mode output_mode;
} yvex_cli_moe_options;

typedef struct {
    const char *status;
    const char *target_id;
    const char *target_class;
    const char *model;
    const char *model_resolved_path;
    const char *family;
    const char *family_detected;
    const char *family_requested;
    const char *backend;
    const char *source_class;
    const char *artifact_class;
    const char *implementation_stage;
    const char *model_is_moe;
    const char *moe_class_status;
    const char *expert_count_status;
    unsigned long long expert_count;
    const char *active_expert_count_status;
    unsigned long long active_expert_count;
    const char *router_status;
    const char *router_tensor_status;
    const char *router_tensor_name;
    const char *router_dtype;
    const char *router_logits_status;
    const char *top_k_policy_status;
    unsigned long long top_k;
    const char *shared_expert_status;
    unsigned long long shared_expert_count;
    const char *expert_tensor_collection_status;
    const char *expert_tensor_roles;
    unsigned long long expert_tensor_count;
    const char *expert_qtype_summary;
    const char *expert_storage_pressure;
    const char *expert_residency_pressure;
    const char *expert_dispatch_status;
    const char *expert_activation_status;
    const char *expert_accumulation_status;
    const char *prefill_integration_status;
    const char *decode_integration_status;
    const char *graph_integration_status;
    const char *runtime_claim;
    const char *generation;
    const char *benchmark_status;
    const char *blockers;
    const char *next_required_rows;
    yvex_models_output_mode output_mode;
    int include_tensors;
    int include_residency;
    int include_blockers;
} yvex_moe_class_report;

#define MOE_TEXT(member, default_text) \
    {#member, YVEX_CLI_FIELD_TEXT, offsetof(yvex_moe_class_report, member), default_text}
#define MOE_U64(member) \
    {#member, YVEX_CLI_FIELD_U64, offsetof(yvex_moe_class_report, member), NULL}
#define MOE_BOOL(key_name, member) \
    {key_name, YVEX_CLI_FIELD_BOOL, offsetof(yvex_moe_class_report, member), NULL}
static const yvex_cli_field_spec moe_audit_fields[] = {
    MOE_TEXT(status, "internal-error"), MOE_TEXT(target_id, "unknown"),
    MOE_TEXT(target_class, "unknown"), MOE_TEXT(model, ""),
    MOE_TEXT(model_resolved_path, "unknown"), MOE_TEXT(family, "unknown"),
    MOE_TEXT(family_detected, "unknown"), MOE_TEXT(family_requested, "auto"),
    MOE_TEXT(backend, "cpu"), MOE_TEXT(source_class, "unknown"),
    MOE_TEXT(artifact_class, "unknown"), MOE_TEXT(implementation_stage, "report-only"),
    MOE_TEXT(model_is_moe, "unknown"), MOE_TEXT(moe_class_status, "unknown"),
    MOE_TEXT(expert_count_status, "unknown"), MOE_U64(expert_count),
    MOE_TEXT(active_expert_count_status, "unknown"), MOE_U64(active_expert_count),
    MOE_TEXT(router_status, "unknown"), MOE_TEXT(router_tensor_status, "unknown"),
    MOE_TEXT(router_tensor_name, "none"), MOE_TEXT(router_dtype, "unknown"),
    MOE_TEXT(router_logits_status, "unsupported"),
    MOE_TEXT(top_k_policy_status, "unknown"), MOE_U64(top_k),
    MOE_TEXT(shared_expert_status, "unknown"), MOE_U64(shared_expert_count),
    MOE_TEXT(expert_tensor_collection_status, "unknown"),
    MOE_TEXT(expert_tensor_roles, "unknown"), MOE_U64(expert_tensor_count),
    MOE_TEXT(expert_qtype_summary, "unknown"),
    MOE_TEXT(expert_storage_pressure, "unknown"),
    MOE_TEXT(expert_residency_pressure, "unknown"),
    MOE_TEXT(expert_dispatch_status, "unsupported"),
    MOE_TEXT(expert_activation_status, "unsupported"),
    MOE_TEXT(expert_accumulation_status, "unsupported"),
    MOE_TEXT(prefill_integration_status, "unsupported"),
    MOE_TEXT(decode_integration_status, "unsupported"),
    MOE_TEXT(graph_integration_status, "unsupported"),
    MOE_TEXT(runtime_claim, "unsupported"),
    MOE_TEXT(generation, "unsupported-full-model"),
    MOE_TEXT(benchmark_status, "not-measured"), MOE_TEXT(blockers, "unknown"),
    MOE_TEXT(next_required_rows, "V010.CLASS.3"),
    MOE_BOOL("report_options.include_tensors", include_tensors),
    MOE_BOOL("report_options.include_residency", include_residency),
    MOE_BOOL("report_options.include_blockers", include_blockers),
};
static const yvex_moe_class_report moe_report_defaults = {
    .status = "internal-error", .target_id = "unknown", .target_class = "unknown",
    .model = "", .model_resolved_path = "unknown", .family = "auto",
    .family_detected = "unknown", .family_requested = "auto", .backend = "cpu",
    .source_class = "unknown", .artifact_class = "unknown",
    .implementation_stage = "report-only", .model_is_moe = "unknown",
    .moe_class_status = "unknown", .expert_count_status = "unknown",
    .active_expert_count_status = "unknown", .router_status = "unknown",
    .router_tensor_status = "unknown", .router_tensor_name = "none",
    .router_dtype = "unknown", .router_logits_status = "unsupported",
    .top_k_policy_status = "unknown", .shared_expert_status = "unknown",
    .expert_tensor_collection_status = "unknown", .expert_tensor_roles = "unknown",
    .expert_qtype_summary = "unknown", .expert_storage_pressure = "unknown",
    .expert_residency_pressure = "unknown", .expert_dispatch_status = "unsupported",
    .expert_activation_status = "unsupported", .expert_accumulation_status = "unsupported",
    .prefill_integration_status = "unsupported", .decode_integration_status = "unsupported",
    .graph_integration_status = "unsupported", .runtime_claim = "unsupported",
    .generation = "unsupported-full-model", .benchmark_status = "not-measured",
    .blockers = "MoE model-class report failed before blocker classification",
    .next_required_rows = "V010.CLASS.3", .output_mode = YVEX_MODELS_OUTPUT_AUDIT,
};
#undef MOE_BOOL
#undef MOE_U64
#undef MOE_TEXT

static const yvex_models_option_spec moe_option_specs[] = {
    {"--model", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_moe_options, model)},
    {"--backend", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_moe_options, backend)},
    {"--family", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_moe_options, family)},
    {"--registry", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_moe_options, registry_path)},
    {"--include-tensors", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_moe_options, include_tensors)},
    {"--include-residency", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_moe_options, include_residency)},
    {"--include-blockers", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_moe_options, include_blockers)},
    {"--output", YVEX_MODELS_OPTION_OUTPUT, offsetof(yvex_cli_moe_options, output_mode)},
};

/* Purpose: Parse parse moe options into typed CLI state (`parse_moe_options`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int parse_moe_options(int arg_count,
                             char **args,
                             yvex_cli_moe_options *options)
{
    int i;

    if (!options) return 2;
    memset(options, 0, sizeof(*options));
    options->backend = "cpu";
    options->family = "auto";
    options->output_mode = YVEX_MODELS_OUTPUT_NORMAL;

    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_model_artifacts_surface_moe_help(stdout);
        return 1;
    }
    if (arg_count < 3 || strcmp(args[2], "report") != 0) {
        yvex_cli_out_writef(stderr, "yvex: moe requires report\n");
        yvex_cli_out_writef(stderr,
            "usage: yvex evidence moe report --model FILE_OR_ALIAS [--family auto|deepseek|glm|qwen] [--backend cpu|"
                "cuda] [--registry FILE] [--include-tensors] [--include-residency] [--include-blockers]\n");
        return 2;
    }

    for (i = 3; i < arg_count; ++i) {
        const char *flag = args[i];
        int handled = 0;
        int rc = parse_models_bound_option("moe", arg_count, args, &i,
                                           options, moe_option_specs,
                                           sizeof(moe_option_specs) /
                                               sizeof(moe_option_specs[0]),
                                           &handled);
        if (rc != 0) return rc;
        if (handled) {
            if (strcmp(flag, "--backend") == 0 &&
                strcmp(options->backend, "cpu") != 0 &&
                strcmp(options->backend, "cuda") != 0) {
                yvex_cli_out_writef(stderr, "yvex: moe --backend must be cpu or cuda\n");
                return 2;
            }
            continue;
        }
        if (strcmp(flag, "--audit") == 0) {
            options->output_mode = YVEX_MODELS_OUTPUT_AUDIT;
        } else if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) {
            yvex_model_artifacts_surface_moe_help(stdout);
            return 1;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown moe option: %s\n", flag);
            return 2;
        }
    }

    if (!options->model || !options->model[0]) {
        yvex_cli_out_writef(stderr, "yvex: moe report requires --model FILE_OR_ALIAS\n");
        return 2;
    }
    return 0;
}

/* Purpose: Construct the owned moe init report state (`moe_init_report`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static void moe_init_report(yvex_moe_class_report *report,
                            const yvex_cli_moe_options *options)
{
    if (!report) return;
    *report = moe_report_defaults;
    report->model = options && options->model ? options->model : "";
    report->family = model_requested_family(options ? options->family : NULL);
    report->family_requested = model_requested_family(options ? options->family : NULL);
    report->backend = options && options->backend ? options->backend : "cpu";
    report->output_mode = options ? options->output_mode : YVEX_MODELS_OUTPUT_AUDIT;
    report->include_tensors = options ? options->include_tensors : 0;
    report->include_residency = options ? options->include_residency : 0;
    report->include_blockers = options ? options->include_blockers : 0;
}

/* Purpose: Render moe print report from typed facts (`moe_print_report`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static void moe_print_report(const yvex_moe_class_report *report)
{
    if (report && report->output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "report: moe\n");
        yvex_cli_out_writef(stdout, "model: %s\n", report->model ? report->model : "");
        yvex_cli_out_writef(stdout, "family: %s\n", report->family ? report->family : "unknown");
        yvex_cli_out_writef(stdout, "status: %s\n", report->status ? report->status : "report-only");
        yvex_cli_out_writef(stdout, "top_blocker: %s\n",
            report->blockers ? report->blockers : "router/expert runtime unsupported");
        yvex_cli_out_writef(stdout, "next: %s\n",
            report->next_required_rows ? report->next_required_rows : "V010.MOE.*");
        yvex_cli_out_writef(stdout, "boundary: report-only, no runtime execution\n");
        return;
    }

    yvex_cli_out_line(stdout, "moe: report");
    if (!report) {
        return;
    }
    (void)yvex_cli_out_fields(stdout, report, moe_audit_fields,
                              sizeof(moe_audit_fields) / sizeof(moe_audit_fields[0]));
    yvex_cli_out_lines(stdout, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
}

/* Purpose: Render moe print source only report from typed facts (`moe_print_source_only_report`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int moe_print_source_only_report(const yvex_cli_moe_options *options,
                                        const char *target)
{
    yvex_moe_class_report report;

    moe_init_report(&report, options);
    report.status = "unsupported-source-only";
    report.target_id = target ? target : "glm-5.2-official-safetensors";
    report.target_class = "huge-source-pressure";
    report.model_resolved_path = "source-only-target";
    report.family = "glm";
    report.family_detected = "glm";
    report.source_class = "official safetensors";
    report.artifact_class = "future YVEX-produced GGUF";
    report.model_is_moe = "true";
    report.moe_class_status = "source-only";
    report.expert_count_status = "unknown";
    report.active_expert_count_status = "unknown";
    report.router_status = "unsupported";
    report.router_tensor_status = "unsupported";
    report.router_logits_status = "unsupported";
    report.top_k_policy_status = "unknown";
    report.shared_expert_status = "unknown";
    report.expert_tensor_collection_status = "unsupported";
    report.expert_tensor_roles = "source-only-not-inspected";
    report.expert_qtype_summary = "unknown";
    report.expert_storage_pressure = "source-only-huge-pressure";
    report.expert_residency_pressure = "source-only-huge-pressure";
    report.blockers = "source-only target has no YVEX-produced GGUF tensor inventory; GLM runtime unsupported";
    report.next_required_rows = "OWI.HUGE.0,MODEL.CLASS.3,TENSOR.COLLECTION.2,V010.TENSOR.14,"
        "V010.TENSOR.15,V010.TENSOR.16,V010.TENSOR.17,V010.STORAGE.18,"
        "V010.RESIDENCY.14,GLM-YVEX-produced-GGUF";
    moe_print_report(&report);
    return 5;
}

/* Purpose: Render moe print missing model report from typed facts (`moe_print_missing_model_report`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int moe_print_missing_model_report(const yvex_cli_moe_options *options,
                                          const char *reason)
{
    yvex_moe_class_report report;

    moe_init_report(&report, options);
    report.status = "missing-model";
    report.target_id = options && options->model ? options->model : "missing";
    report.target_class = "unknown";
    report.model_resolved_path = "missing";
    report.family_detected = "unknown";
    report.source_class = "unknown";
    report.artifact_class = "unknown";
    report.model_is_moe = "unknown";
    report.moe_class_status = "unknown";
    report.router_status = "unknown";
    report.router_tensor_status = "unknown";
    report.expert_tensor_collection_status = "unknown";
    report.expert_tensor_roles = "unknown";
    report.blockers = reason && reason[0] ? reason : "model or alias could not be resolved";
    report.next_required_rows = "V010.TARGET.9,V010.CLASS.3";
    moe_print_report(&report);
    return 5;
}

/* Purpose: Render moe print unsupported family report from typed facts (`moe_print_unsupported_family_report`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int moe_print_unsupported_family_report(const yvex_cli_moe_options *options,
                                               yvex_model_ref *ref,
                                               const char *target_id,
                                               const char *target_class,
                                               const char *detected)
{
    yvex_moe_class_report report;

    moe_init_report(&report, options);
    report.status = "unsupported-family";
    report.target_id = target_id ? target_id : "path";
    report.target_class = target_class ? target_class : "candidate-GGUF-path";
    report.model_resolved_path = ref && ref->path ? ref->path : "unknown";
    report.family = model_requested_family(options ? options->family : NULL);
    report.family_detected = detected ? detected : "unknown";
    report.source_class = "GGUF artifact";
    report.artifact_class = "GGUF artifact";
    report.model_is_moe = "unknown";
    report.moe_class_status = "unsupported-family";
    report.router_status = "unknown";
    report.router_tensor_status = "unknown";
    report.expert_tensor_collection_status = "unknown";
    report.expert_tensor_roles = "unknown";
    report.blockers = "requested family is not supported by MoE model-class report";
    report.next_required_rows = "V010.CLASS.0,V010.CLASS.3,family-specific-MoE-adapter";
    moe_print_report(&report);
    return 5;
}

/* Purpose: Render moe print model report from typed facts (`moe_print_model_report`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int moe_print_model_report(const yvex_cli_moe_options *options,
                                  yvex_model_ref *ref,
                                  yvex_model_context *ctx,
                                  const char *target_id,
                                  const char *target_class,
                                  yvex_arch arch,
                                  const yvex_fullmodel_collections *collections)
{
    yvex_moe_class_report report;
    yvex_cli_fullmodel_options family_probe;
    const yvex_tensor_info *router;
    const yvex_tensor_info *expert_gate;
    const yvex_tensor_info *expert_up;
    const yvex_tensor_info *expert_down;
    const char *requested;
    const char *detected;
    char expert_roles[160];
    char expert_qtypes[192];
    unsigned int present_roles = 0u;

    memset(&family_probe, 0, sizeof(family_probe));
    family_probe.model = options ? options->model : NULL;
    family_probe.family = options ? options->family : NULL;
    requested = model_requested_family(options ? options->family : NULL);
    detected = fullmodel_detect_family(&family_probe, arch, target_id);
    if (!fullmodel_family_request_matches(requested, detected) || strcmp(detected, "deepseek") != 0) {
        return moe_print_unsupported_family_report(options, ref, target_id, target_class, detected);
    }

    router = fullmodel_descriptor_find_tensor(ctx, "moe_router");
    expert_gate = fullmodel_descriptor_find_tensor(ctx, "moe_expert_gate");
    expert_up = fullmodel_descriptor_find_tensor(ctx, "moe_expert_up");
    expert_down = fullmodel_descriptor_find_tensor(ctx, "moe_expert_down");

    expert_roles[0] = '\0';
    expert_qtypes[0] = '\0';
    if (expert_gate) {
        present_roles++;
        model_artifact_append_role(expert_roles, sizeof(expert_roles), "moe-expert-gate");
        snprintf(expert_qtypes + strlen(expert_qtypes),
                 strlen(expert_qtypes) < sizeof(expert_qtypes) ? sizeof(expert_qtypes) - strlen(expert_qtypes) : 0u,
                 "%smoe-expert-gate:%s",
                 expert_qtypes[0] ? "," : "",
                 yvex_dtype_name(expert_gate->dtype));
    }
    if (expert_up) {
        present_roles++;
        model_artifact_append_role(expert_roles, sizeof(expert_roles), "moe-expert-up");
        snprintf(expert_qtypes + strlen(expert_qtypes),
                 strlen(expert_qtypes) < sizeof(expert_qtypes) ? sizeof(expert_qtypes) - strlen(expert_qtypes) : 0u,
                 "%smoe-expert-up:%s",
                 expert_qtypes[0] ? "," : "",
                 yvex_dtype_name(expert_up->dtype));
    }
    if (expert_down) {
        present_roles++;
        model_artifact_append_role(expert_roles, sizeof(expert_roles), "moe-expert-down");
        snprintf(expert_qtypes + strlen(expert_qtypes),
                 strlen(expert_qtypes) < sizeof(expert_qtypes) ? sizeof(expert_qtypes) - strlen(expert_qtypes) : 0u,
                 "%smoe-expert-down:%s",
                 expert_qtypes[0] ? "," : "",
                 yvex_dtype_name(expert_down->dtype));
    }
    if (!expert_roles[0]) snprintf(expert_roles, sizeof(expert_roles), "missing");
    if (!expert_qtypes[0]) snprintf(expert_qtypes, sizeof(expert_qtypes), "unknown");

    moe_init_report(&report, options);
    report.status = "ok-partial";
    report.target_id = target_id ? target_id : "path";
    report.target_class = target_class ? target_class : "candidate-GGUF-path";
    report.model_resolved_path = ref && ref->path ? ref->path : "unknown";
    report.family = "deepseek";
    report.family_detected = detected;
    report.family_requested = requested;
    report.source_class = strcmp(report.target_class, "selected-runtime-slice") == 0
                              ? "YVEX-produced selected GGUF"
                              : "GGUF artifact";
    report.artifact_class = strcmp(report.target_class, "selected-runtime-slice") == 0
                                ? "YVEX-produced selected GGUF"
                                : "GGUF artifact";
    report.model_is_moe = "true";
    report.moe_class_status = "partial";
    report.expert_count_status = "unknown";
    report.active_expert_count_status = "unknown";
    report.router_status = router ? "known" : "missing";
    report.router_tensor_status = router ? "known" : "missing";
    report.router_tensor_name = router && router->name ? router->name : "none";
    report.router_dtype = router ? yvex_dtype_name(router->dtype) : "unknown";
    report.router_logits_status = "unsupported";
    report.top_k_policy_status = "unknown";
    report.shared_expert_status = "unknown";
    report.expert_tensor_collection_status = present_roles == 3u ? "known" :
                                             (present_roles > 0u ? "partial" : "missing");
    report.expert_tensor_roles = expert_roles;
    report.expert_tensor_count = present_roles;
    report.expert_qtype_summary = expert_qtypes;
    report.expert_storage_pressure = present_roles > 0u ? "planned" : "missing";
    report.expert_residency_pressure = present_roles > 0u ? "planned" : "missing";
    report.blockers = router
                          ? "router tensor is classified but router logits, top-k routing, expert "
                              "activation, dispatch, accumulation, graph integration, prefill integration, "
                              "and decode integration are unsupported"
                          : "router tensor missing; expert tensor collection missing; router logits "
                              "unsupported; top-k routing unsupported; expert activation unsupported; expert "
                              "dispatch unsupported; expert accumulation unsupported";
    report.next_required_rows = router || present_roles > 0u
                                    ? "V010.MOE.4,V010.MOE.5,V010.MOE.6,V010.MOE.7,V010.STORAGE.18,V010.RESIDENCY.14"
                                    : "V010.TENSOR.14,V010.TENSOR.15,V010.TENSOR.16,V010.TENSOR.17,"
                                        "V010.STORAGE.18,V010.RESIDENCY.14,V010.MOE.4";
    if (collections && collections->has_moe_expert && present_roles == 0u) {
        report.expert_tensor_collection_status = "partial";
        report.expert_tensor_count = collections->moe;
        report.expert_tensor_roles = "expert-pattern-detected";
        report.expert_storage_pressure = "planned";
        report.expert_residency_pressure = "planned";
    }
    moe_print_report(&report);
    return 0;
}

/* Purpose: Orchestrate the typed model artifacts surface moe command request
 * (`yvex_model_artifacts_surface_moe_command`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_model_artifacts_surface_moe_command(int arg_count, char **args)
{
    yvex_cli_moe_options options;
    yvex_model_ref_options ref_options;
    yvex_model_ref ref;
    yvex_model_context ctx;
    yvex_fullmodel_collections collections;
    yvex_error err;
    const char *target_id;
    const char *target_class;
    yvex_arch arch;
    unsigned long long artifact_bytes = 0ull;
    unsigned long long tensor_count;
    unsigned long long i;
    int selected_target;
    int rc;

    yvex_error_clear(&err);
    memset(&options, 0, sizeof(options));
    memset(&ref, 0, sizeof(ref));
    memset(&ctx, 0, sizeof(ctx));
    memset(&collections, 0, sizeof(collections));

    rc = parse_moe_options(arg_count, args, &options);
    if (rc == 1) return 0;
    if (rc != 0) return rc;

    if (strcmp(options.model, "glm-5.2-official-safetensors") == 0) {
        return moe_print_source_only_report(&options, "glm-5.2-official-safetensors");
    }

    memset(&ref_options, 0, sizeof(ref_options));
    ref_options.allow_registry = 1;
    ref_options.registry_path = options.registry_path;
    rc = yvex_model_ref_resolve(&ref, options.model, &ref_options, &err);
    if (rc != YVEX_OK) {
        yvex_error_clear(&err);
        return moe_print_missing_model_report(&options, "model or alias could not be resolved");
    }
    if (!fullmodel_file_size(ref.path, &artifact_bytes)) {
        rc = moe_print_missing_model_report(&options, "resolved artifact path does not exist");
        yvex_model_ref_clear(&ref);
        return rc;
    }

    rc = yvex_model_context_open(ref.path, &ctx, &err);
    if (rc != YVEX_OK) {
        const char *reason = yvex_error_message(&err);
        rc = moe_print_missing_model_report(&options,
            reason && reason[0] ? reason : "GGUF metadata or tensor directory parse failed");
        yvex_error_clear(&err);
        yvex_model_ref_clear(&ref);
        return rc;
    }

    tensor_count = yvex_tensor_table_count(ctx.table);
    for (i = 0; i < tensor_count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(ctx.table, i);
        if (!tensor) continue;
        fullmodel_classify_tensor(tensor, &collections);
    }
    arch = yvex_model_arch(ctx.model);
    target_id = ref.alias && ref.alias[0] ? ref.alias : "path";
    selected_target = fullmodel_is_selected_target(target_id) ||
                      fullmodel_is_selected_target(options.model);
    target_class = selected_target ? "selected-runtime-slice" : "candidate-GGUF-path";
    rc = moe_print_model_report(&options,
                                &ref,
                                &ctx,
                                target_id,
                                target_class,
                                arch,
                                &collections);
    yvex_model_context_close(&ctx);
    yvex_model_ref_clear(&ref);
    return rc;
}

/* Purpose: Render model artifacts surface moe help from typed facts (`yvex_model_artifacts_surface_moe_help`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
void yvex_model_artifacts_surface_moe_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex evidence moe report --model FILE_OR_ALIAS [--family auto|deepseek|glm|qwen] [--backend cpu|"
            "cuda] [--registry FILE] [--audit | --output normal|table|audit] [--include-tensors] [--include-"
            "residency] [--include-blockers]\n");
    yvex_cli_out_writef(fp, "\nExamples:\n");
    yvex_cli_out_writef(fp,
        "  yvex evidence moe report --model deepseek4-v4-flash-selected-embed-rmsnorm --family deepseek --backend "
            "cpu --include-tensors --include-blockers\n");
    yvex_cli_out_writef(fp,
        "  yvex evidence moe report --model deepseek4-v4-flash-selected-embed-rmsnorm --family deepseek --backend "
            "cuda --include-residency\n");
    yvex_cli_out_writef(fp,
        "  yvex evidence moe report --model glm-5.2-official-safetensors --family glm "
        "--backend cpu --include-blockers\n");
    yvex_cli_out_lines(fp, literal_pair_0, sizeof(literal_pair_0) / sizeof(literal_pair_0[0]));
    yvex_cli_out_writef(fp, "  Default output is compact. Use --audit for full diagnostic fields.\n");
    yvex_cli_out_lines(fp, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
}
