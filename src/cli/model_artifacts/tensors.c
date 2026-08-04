/*
 * Present tensor-collection coverage and geometry without materializing payloads or executing the
 * model.
 */
#include "src/cli/model_artifacts/private.h"

#include <string.h>

static const char *const literal_pair_0[] = { "\ntensor-collection report:",
    "  reports inspect tensor collection requirements and coverage for the requested collection. The current "
        "implemented collection is moe."
};

static const char *const literal_pair_1[] = { "cleanup_attempted: false",
    "cleanup_status: not-needed"};

static const char *const literal_lines_0[] = {
    "  MoE collection reports classify router, expert gate/up/down, shared expert, dispatch metadata, "
        "indexing, storage pressure, residency pressure, blockers, and next rows.",
    "  report-only boundary: it does not materialize tensors, does not execute router logits, does not "
        "select experts, does not dispatch experts, does not accumulate expert outputs, does not run MoE "
        "blocks, does not run prefill, does not run decode, does not produce logits, does not sample, does not "
        "generate, does not evaluate, and does not benchmark.",
    "  selected-runtime-slice targets may return ok-partial when the family is MoE but router or expert "
        "tensors are not present in the selected artifact.",
    "  source-only targets are reported without opening huge source shards or downloading artifacts.",
    "Boundary: no MoE runtime execution, no router logits, no top-k routing, no expert activation, no "
        "expert dispatch, no expert accumulation, no full transformer block, no full transformer prefill, no "
        "decode, no logits, no vocabulary sampling, no generation, no eval, no benchmark, no throughput."
};

typedef struct {
    const char *model;
    const char *backend;
    const char *family;
    const char *collection;
    const char *registry_path;
    int include_router;
    int include_experts;
    int include_shared;
    int include_dispatch;
    int include_storage;
    int include_residency;
    int include_blockers;
    yvex_models_output_mode output_mode;
} yvex_cli_tensor_collection_options;

typedef struct {
    const char *tensor_collection;
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
    const char *collection_stage;
    const char *model_is_moe;
    const char *moe_class_status;
    const char *router_collection_status;
    const char *router_role_status;
    const char *router_tensor_name;
    const char *router_tensor_present;
    const char *router_tensor_status;
    const char *router_dtype;
    const char *router_qtype;
    const char *router_shape;
    const char *router_source;
    const char *router_logits_status;
    const char *top_k_policy_status;
    unsigned long long top_k;
    const char *expert_collection_status;
    const char *expert_role_status;
    const char *expert_tensor_roles;
    const char *expert_gate_role_status;
    const char *expert_up_role_status;
    const char *expert_down_role_status;
    const char *expert_tensor_name_pattern;
    const char *expert_tensor_count_status;
    unsigned long long expert_tensor_count;
    const char *expert_count_status;
    unsigned long long expert_count;
    const char *active_expert_count_status;
    unsigned long long active_expert_count;
    const char *shared_expert_collection_status;
    const char *shared_expert_status;
    const char *shared_expert_roles;
    const char *shared_expert_count_status;
    unsigned long long shared_expert_count;
    const char *dispatch_metadata_status;
    const char *dispatch_metadata_roles;
    const char *expert_indexing_policy_status;
    const char *expert_storage_pressure_status;
    const char *expert_storage_pressure;
    const char *expert_residency_pressure_status;
    const char *expert_residency_pressure;
    const char *present_roles;
    const char *missing_roles;
    const char *unknown_roles;
    const char *not_applicable_roles;
    const char *blocked_rows;
    const char *next_required_rows;
    const char *runtime_claim;
    const char *generation;
    const char *benchmark_status;
    yvex_models_output_mode output_mode;
    int include_router;
    int include_experts;
    int include_shared;
    int include_dispatch;
    int include_storage;
    int include_residency;
    int include_blockers;
} yvex_tensor_collection_report;

#define TENSOR_TEXT(member, default_text) \
    {#member, YVEX_CLI_FIELD_TEXT, offsetof(yvex_tensor_collection_report, member), default_text}
#define TENSOR_U64(member) \
    {#member, YVEX_CLI_FIELD_U64, offsetof(yvex_tensor_collection_report, member), NULL}
#define TENSOR_BOOL(key_name, member) \
    {key_name, YVEX_CLI_FIELD_BOOL, offsetof(yvex_tensor_collection_report, member), NULL}

static const yvex_cli_field_spec tensor_collection_audit_fields[] = {
    TENSOR_TEXT(tensor_collection, "unknown"), TENSOR_TEXT(status, "internal-error"),
    TENSOR_TEXT(target_id, "unknown"), TENSOR_TEXT(target_class, "unknown"),
    TENSOR_TEXT(model, ""), TENSOR_TEXT(model_resolved_path, "unknown"),
    TENSOR_TEXT(family, "unknown"), TENSOR_TEXT(family_detected, "unknown"),
    TENSOR_TEXT(family_requested, "auto"), TENSOR_TEXT(backend, "cpu"),
    TENSOR_TEXT(source_class, "unknown"), TENSOR_TEXT(artifact_class, "unknown"),
    TENSOR_TEXT(implementation_stage, "report-only"),
    TENSOR_TEXT(collection_stage, "report-only"), TENSOR_TEXT(model_is_moe, "unknown"),
    TENSOR_TEXT(moe_class_status, "unknown"),
    TENSOR_TEXT(router_collection_status, "unknown"),
    TENSOR_TEXT(router_role_status, "unknown"), TENSOR_TEXT(router_tensor_name, "none"),
    TENSOR_TEXT(router_tensor_present, "false"),
    TENSOR_TEXT(router_tensor_status, "unknown"), TENSOR_TEXT(router_dtype, "unknown"),
    TENSOR_TEXT(router_qtype, "unknown"), TENSOR_TEXT(router_shape, "unknown"),
    TENSOR_TEXT(router_source, "unknown"),
    TENSOR_TEXT(router_logits_status, "unsupported"),
    TENSOR_TEXT(top_k_policy_status, "unknown"), TENSOR_U64(top_k),
    TENSOR_TEXT(expert_collection_status, "unknown"),
    TENSOR_TEXT(expert_role_status, "unknown"), TENSOR_TEXT(expert_tensor_roles, "unknown"),
    TENSOR_TEXT(expert_gate_role_status, "unknown"),
    TENSOR_TEXT(expert_up_role_status, "unknown"),
    TENSOR_TEXT(expert_down_role_status, "unknown"),
    TENSOR_TEXT(expert_tensor_name_pattern, "unknown"),
    TENSOR_TEXT(expert_tensor_count_status, "unknown"), TENSOR_U64(expert_tensor_count),
    TENSOR_TEXT(expert_count_status, "unknown"), TENSOR_U64(expert_count),
    TENSOR_TEXT(active_expert_count_status, "unknown"), TENSOR_U64(active_expert_count),
    TENSOR_TEXT(shared_expert_collection_status, "unknown"),
    TENSOR_TEXT(shared_expert_status, "unknown"),
    TENSOR_TEXT(shared_expert_roles, "unknown"),
    TENSOR_TEXT(shared_expert_count_status, "unknown"), TENSOR_U64(shared_expert_count),
    TENSOR_TEXT(dispatch_metadata_status, "unknown"),
    TENSOR_TEXT(dispatch_metadata_roles, "unknown"),
    TENSOR_TEXT(expert_indexing_policy_status, "unknown"),
    TENSOR_TEXT(expert_storage_pressure_status, "unknown"),
    TENSOR_TEXT(expert_storage_pressure, "unknown"),
    TENSOR_TEXT(expert_residency_pressure_status, "unknown"),
    TENSOR_TEXT(expert_residency_pressure, "unknown"),
    TENSOR_TEXT(present_roles, "none"), TENSOR_TEXT(missing_roles, "unknown"),
    TENSOR_TEXT(unknown_roles, "unknown"), TENSOR_TEXT(not_applicable_roles, "none"),
    TENSOR_TEXT(blocked_rows, "unknown"), TENSOR_TEXT(next_required_rows, "unknown"),
    TENSOR_TEXT(runtime_claim, "unsupported"),
    TENSOR_TEXT(generation, "unsupported-full-model"),
    TENSOR_TEXT(benchmark_status, "not-measured"),
    TENSOR_BOOL("report_options.include_router", include_router),
    TENSOR_BOOL("report_options.include_experts", include_experts),
    TENSOR_BOOL("report_options.include_shared", include_shared),
    TENSOR_BOOL("report_options.include_dispatch", include_dispatch),
    TENSOR_BOOL("report_options.include_storage", include_storage),
    TENSOR_BOOL("report_options.include_residency", include_residency),
    TENSOR_BOOL("report_options.include_blockers", include_blockers),
};

static const yvex_tensor_collection_report tensor_collection_defaults = {
    .tensor_collection = "unknown", .status = "internal-error", .target_id = "unknown",
    .target_class = "unknown", .model = "", .model_resolved_path = "unknown",
    .family = "auto", .family_detected = "unknown", .family_requested = "auto",
    .backend = "cpu", .source_class = "unknown", .artifact_class = "unknown",
    .implementation_stage = "report-only", .collection_stage = "report-only",
    .model_is_moe = "unknown", .moe_class_status = "unknown",
    .router_collection_status = "unknown", .router_role_status = "unknown",
    .router_tensor_name = "none", .router_tensor_present = "false",
    .router_tensor_status = "unknown", .router_dtype = "unknown",
    .router_qtype = "unknown", .router_shape = "unknown", .router_source = "unknown",
    .router_logits_status = "unsupported", .top_k_policy_status = "unknown",
    .expert_collection_status = "unknown", .expert_role_status = "unknown",
    .expert_tensor_roles = "unknown", .expert_gate_role_status = "unknown",
    .expert_up_role_status = "unknown", .expert_down_role_status = "unknown",
    .expert_tensor_name_pattern = "unknown", .expert_tensor_count_status = "unknown",
    .expert_count_status = "unknown", .active_expert_count_status = "unknown",
    .shared_expert_collection_status = "unknown", .shared_expert_status = "unknown",
    .shared_expert_roles = "unknown", .shared_expert_count_status = "unknown",
    .dispatch_metadata_status = "unknown", .dispatch_metadata_roles = "unknown",
    .expert_indexing_policy_status = "unknown", .expert_storage_pressure_status = "unknown",
    .expert_storage_pressure = "unknown", .expert_residency_pressure_status = "unknown",
    .expert_residency_pressure = "unknown", .present_roles = "none",
    .missing_roles = "unknown", .unknown_roles = "unknown", .not_applicable_roles = "none",
    .blocked_rows = "V010.TENSOR.14,V010.TENSOR.15,V010.TENSOR.16,V010.TENSOR.17",
    .next_required_rows = "V010.TENSOR.14,V010.TENSOR.15,V010.TENSOR.16,V010.TENSOR.17",
    .runtime_claim = "unsupported", .generation = "unsupported-full-model",
    .benchmark_status = "not-measured", .output_mode = YVEX_MODELS_OUTPUT_AUDIT,
};

#undef TENSOR_BOOL
#undef TENSOR_U64
#undef TENSOR_TEXT

static const yvex_models_option_spec tensor_option_specs[] = {
    {"--model", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_tensor_collection_options, model)},
    {"--backend", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_tensor_collection_options, backend)},
    {"--family", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_tensor_collection_options, family)},
    {"--collection", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_tensor_collection_options, collection)},
    {"--registry", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_tensor_collection_options, registry_path)},
    {"--include-router", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_tensor_collection_options, include_router)},
    {"--include-experts", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_tensor_collection_options, include_experts)},
    {"--include-shared", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_tensor_collection_options, include_shared)},
    {"--include-dispatch", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_tensor_collection_options, include_dispatch)},
    {"--include-storage", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_tensor_collection_options, include_storage)},
    {"--include-residency", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_tensor_collection_options, include_residency)},
    {"--include-blockers", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_tensor_collection_options, include_blockers)},
    {"--output", YVEX_MODELS_OPTION_OUTPUT,
     offsetof(yvex_cli_tensor_collection_options, output_mode)},
};

static int parse_tensor_collection_options(int arg_count,
                                           char **args,
                                           yvex_cli_tensor_collection_options *options)
{
    int i;

    if (!options) return 2;
    memset(options, 0, sizeof(*options));
    options->backend = "cpu";
    options->family = "auto";
    options->output_mode = YVEX_MODELS_OUTPUT_NORMAL;

    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_tensor_collection_help(stdout);
        return 1;
    }
    if (arg_count < 3 || strcmp(args[2], "report") != 0) {
        yvex_cli_out_writef(stderr, "yvex: tensor-collection requires report\n");
        yvex_cli_out_writef(stderr,
            "usage: yvex inspect tensor collection report --model FILE_OR_ALIAS --collection moe [--family auto|"
                "deepseek|glm|qwen] [--backend cpu|cuda]\n");
        return 2;
    }

    for (i = 3; i < arg_count; ++i) {
        const char *flag = args[i];
        int handled = 0;
        int rc = parse_models_bound_option("tensor-collection", arg_count,
                                           args, &i, options,
                                           tensor_option_specs,
                                           sizeof(tensor_option_specs) /
                                               sizeof(tensor_option_specs[0]),
                                           &handled);
        if (rc != 0) return rc;
        if (handled) continue;
        if (strcmp(flag, "--audit") == 0) {
            options->output_mode = YVEX_MODELS_OUTPUT_AUDIT;
        } else if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) {
            yvex_tensor_collection_help(stdout);
            return 1;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown tensor-collection option: %s\n", flag);
            return 2;
        }
    }

    if (!options->model || !options->model[0]) {
        yvex_cli_out_writef(stderr, "yvex: tensor-collection report requires --model FILE_OR_ALIAS\n");
        return 2;
    }
    return 0;
}

static const char *tensor_collection_requested_collection(const yvex_cli_tensor_collection_options *options)
{
    return options && options->collection && options->collection[0] ? options->collection : "unknown";
}

static void tensor_collection_init_report(yvex_tensor_collection_report *report,
    const yvex_cli_tensor_collection_options *options)
{
    if (!report) return;
    *report = tensor_collection_defaults;
    report->tensor_collection = tensor_collection_requested_collection(options);
    report->model = options && options->model ? options->model : "";
    report->family = model_requested_family(options ? options->family : NULL);
    report->family_requested = model_requested_family(options ? options->family : NULL);
    report->backend = options && options->backend ? options->backend : "cpu";
    report->output_mode = options ? options->output_mode : YVEX_MODELS_OUTPUT_AUDIT;
    report->include_router = options ? options->include_router : 0;
    report->include_experts = options ? options->include_experts : 0;
    report->include_shared = options ? options->include_shared : 0;
    report->include_dispatch = options ? options->include_dispatch : 0;
    report->include_storage = options ? options->include_storage : 0;
    report->include_residency = options ? options->include_residency : 0;
    report->include_blockers = options ? options->include_blockers : 0;
}

static void tensor_collection_print_report(const yvex_tensor_collection_report *report)
{
    if (report && report->output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "report: tensor-collection\n");
        yvex_cli_out_writef(stdout, "model: %s\n", report->model ? report->model : "");
        yvex_cli_out_writef(stdout, "family: %s collection=%s\n",
               report->family ? report->family : "unknown",
               report->tensor_collection ? report->tensor_collection : "unknown");
        yvex_cli_out_writef(stdout, "status: %s\n", report->status ? report->status : "report-only");
        yvex_cli_out_writef(stdout, "top_blocker: %s\n",
            report->blocked_rows ? report->blocked_rows : "tensor roles incomplete");
        yvex_cli_out_writef(stdout, "next: %s\n",
            report->next_required_rows ? report->next_required_rows : "V010.TENSOR.*");
        yvex_cli_out_writef(stdout, "boundary: report-only, no runtime execution\n");
        return;
    }
    if (!report) {
        return;
    }
    (void)yvex_cli_out_fields(stdout, report, tensor_collection_audit_fields,
                              sizeof(tensor_collection_audit_fields) /
                                  sizeof(tensor_collection_audit_fields[0]));
    yvex_cli_out_line(stdout, "generation_ready: false");
    yvex_cli_out_lines(stdout, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
}

static int tensor_collection_print_invalid_argument_report(const yvex_cli_tensor_collection_options *options,
                                                           const char *status,
                                                           const char *reason)
{
    yvex_tensor_collection_report report;

    tensor_collection_init_report(&report, options);
    report.status = status && status[0] ? status : "invalid-argument";
    report.target_id = options && options->model ? options->model : "unknown";
    report.model_resolved_path = "not-resolved";
    report.collection_stage = "report-only";
    report.router_logits_status = "unsupported";
    report.blocked_rows = reason && reason[0] ? reason : "invalid inspect tensor collection report argument";
    report.next_required_rows = "valid-collection-moe";
    tensor_collection_print_report(&report);
    return 2;
}

static int tensor_collection_print_source_only_report(const yvex_cli_tensor_collection_options *options,
                                                      const char *target)
{
    yvex_tensor_collection_report report;

    tensor_collection_init_report(&report, options);
    report.tensor_collection = "moe";
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
    report.router_collection_status = "unsupported";
    report.router_role_status = "unsupported";
    report.router_tensor_status = "unsupported";
    report.router_source = "source-only-not-inspected";
    report.expert_collection_status = "unsupported";
    report.expert_role_status = "unsupported";
    report.expert_tensor_roles = "source-only-not-inspected";
    report.expert_gate_role_status = "unsupported";
    report.expert_up_role_status = "unsupported";
    report.expert_down_role_status = "unsupported";
    report.expert_tensor_count_status = "unknown";
    report.shared_expert_collection_status = "unknown";
    report.shared_expert_status = "unknown";
    report.shared_expert_count_status = "unknown";
    report.dispatch_metadata_status = "unknown";
    report.expert_storage_pressure_status = "unknown";
    report.expert_storage_pressure = "source-only-unmeasured";
    report.expert_residency_pressure_status = "unknown";
    report.expert_residency_pressure = "requires-storage-stream-plan";
    report.present_roles = "none";
    report.missing_roles = "source-only-not-inspected";
    report.unknown_roles = "moe_router,moe_expert_gate,moe_expert_up,moe_expert_down,moe_shared_expert,"
        "moe_dispatch_metadata,expert_indexing_policy";
    report.blocked_rows = "OWI.HUGE.0,V010.MAP.4,V010.TENSOR.14,V010.TENSOR.15,V010.TENSOR.16,"
        "V010.TENSOR.17,V010.STORAGE.18,V010.RESIDENCY.14";
    report.next_required_rows = "OWI.HUGE.0,V010.MAP.4,V010.TENSOR.14,V010.TENSOR.15,V010.TENSOR.16,"
        "V010.TENSOR.17,V010.STORAGE.18,V010.RESIDENCY.14,GLM-YVEX-produced-GGUF";
    tensor_collection_print_report(&report);
    return 5;
}

static int tensor_collection_print_missing_model_report(const yvex_cli_tensor_collection_options *options,
                                                        const char *reason)
{
    yvex_tensor_collection_report report;

    tensor_collection_init_report(&report, options);
    report.tensor_collection = "moe";
    report.status = "missing-model";
    report.target_id = options && options->model ? options->model : "missing";
    report.model_resolved_path = "missing";
    report.router_collection_status = "unknown";
    report.router_role_status = "unknown";
    report.router_tensor_status = "unknown";
    report.expert_collection_status = "unknown";
    report.expert_role_status = "unknown";
    report.blocked_rows = reason && reason[0] ? reason : "model or alias could not be resolved";
    report.next_required_rows = "V010.TARGET.9,V010.TENSOR.14,V010.TENSOR.15,V010.TENSOR.16,V010.TENSOR.17";
    tensor_collection_print_report(&report);
    return 5;
}

static int tensor_collection_print_unsupported_family_report(const yvex_cli_tensor_collection_options *options,
                                                            yvex_model_ref *ref,
                                                            const char *target_id,
                                                            const char *target_class,
                                                            const char *detected)
{
    yvex_tensor_collection_report report;

    tensor_collection_init_report(&report, options);
    report.tensor_collection = "moe";
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
    report.router_collection_status = "unknown";
    report.router_role_status = "unknown";
    report.router_tensor_status = "unknown";
    report.expert_collection_status = "unknown";
    report.expert_role_status = "unknown";
    report.blocked_rows = "requested family is not supported by MoE inspect tensor collection report";
    report.next_required_rows = "V010.CLASS.0,V010.TENSOR.14,V010.TENSOR.15,V010.TENSOR.16,V010.TENSOR.17,"
        "family-specific-tensor-adapter";
    tensor_collection_print_report(&report);
    return 5;
}

static const char *tensor_collection_role_status(const yvex_tensor_info *tensor)
{
    return tensor ? "present" : "missing";
}

static int tensor_collection_print_model_report(const yvex_cli_tensor_collection_options *options,
                                                yvex_model_ref *ref,
                                                yvex_model_context *ctx,
                                                const char *target_id,
                                                const char *target_class,
                                                yvex_arch arch)
{
    yvex_tensor_collection_report report;
    yvex_cli_fullmodel_options family_probe;
    const yvex_tensor_info *router;
    const yvex_tensor_info *expert_gate;
    const yvex_tensor_info *expert_up;
    const yvex_tensor_info *expert_down;
    const char *requested;
    const char *detected;
    char router_shape[128];
    char present_roles[192];
    char missing_roles[256];
    char expert_roles[192];
    unsigned int present_expert_roles = 0u;

    memset(&family_probe, 0, sizeof(family_probe));
    family_probe.model = options ? options->model : NULL;
    family_probe.family = options ? options->family : NULL;
    requested = model_requested_family(options ? options->family : NULL);
    detected = fullmodel_detect_family(&family_probe, arch, target_id);
    if (!fullmodel_family_request_matches(requested, detected) || strcmp(detected, "deepseek") != 0) {
        return tensor_collection_print_unsupported_family_report(options, ref, target_id, target_class, detected);
    }

    router = fullmodel_descriptor_find_tensor(ctx, "moe_router");
    expert_gate = fullmodel_descriptor_find_tensor(ctx, "moe_expert_gate");
    expert_up = fullmodel_descriptor_find_tensor(ctx, "moe_expert_up");
    expert_down = fullmodel_descriptor_find_tensor(ctx, "moe_expert_down");

    router_shape[0] = '\0';
    if (router) dims_to_text(router->dims, router->rank, router_shape, sizeof(router_shape));
    else snprintf(router_shape, sizeof(router_shape), "unknown");

    present_roles[0] = '\0';
    missing_roles[0] = '\0';
    expert_roles[0] = '\0';
    if (router) model_artifact_append_role(present_roles, sizeof(present_roles), "moe_router");
    else model_artifact_append_role(missing_roles, sizeof(missing_roles), "moe_router");
    if (expert_gate) {
        present_expert_roles++;
        model_artifact_append_role(present_roles, sizeof(present_roles), "moe_expert_gate");
        model_artifact_append_role(expert_roles, sizeof(expert_roles), "moe_expert_gate");
    } else {
        model_artifact_append_role(missing_roles, sizeof(missing_roles), "moe_expert_gate");
    }
    if (expert_up) {
        present_expert_roles++;
        model_artifact_append_role(present_roles, sizeof(present_roles), "moe_expert_up");
        model_artifact_append_role(expert_roles, sizeof(expert_roles), "moe_expert_up");
    } else {
        model_artifact_append_role(missing_roles, sizeof(missing_roles), "moe_expert_up");
    }
    if (expert_down) {
        present_expert_roles++;
        model_artifact_append_role(present_roles, sizeof(present_roles), "moe_expert_down");
        model_artifact_append_role(expert_roles, sizeof(expert_roles), "moe_expert_down");
    } else {
        model_artifact_append_role(missing_roles, sizeof(missing_roles), "moe_expert_down");
    }
    if (!present_roles[0]) snprintf(present_roles, sizeof(present_roles), "none");
    if (!missing_roles[0]) snprintf(missing_roles, sizeof(missing_roles), "none");
    if (!expert_roles[0]) snprintf(expert_roles, sizeof(expert_roles), "missing");

    tensor_collection_init_report(&report, options);
    report.tensor_collection = "moe";
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
    report.router_collection_status = router ? "present" : "missing";
    report.router_role_status = tensor_collection_role_status(router);
    report.router_tensor_name = router && router->name ? router->name : "none";
    report.router_tensor_present = router ? "true" : "false";
    report.router_tensor_status = tensor_collection_role_status(router);
    report.router_dtype = router ? yvex_dtype_name(router->dtype) : "unknown";
    report.router_qtype = router ? yvex_dtype_name(router->dtype) : "unknown";
    report.router_shape = router_shape;
    report.router_source = router ? "artifact-directory" : "missing";
    report.router_logits_status = "unsupported";
    report.top_k_policy_status = "unknown";
    report.top_k = 0ull;
    report.expert_collection_status = present_expert_roles == 3u ? "present" :
                                      (present_expert_roles > 0u ? "partial" : "missing");
    report.expert_role_status = present_expert_roles == 3u ? "present" :
                                (present_expert_roles > 0u ? "partial" : "missing");
    report.expert_tensor_roles = expert_roles;
    report.expert_gate_role_status = tensor_collection_role_status(expert_gate);
    report.expert_up_role_status = tensor_collection_role_status(expert_up);
    report.expert_down_role_status = tensor_collection_role_status(expert_down);
    report.expert_tensor_name_pattern = present_expert_roles > 0u ? "expert.*.(gate|up|down)" : "unknown";
    report.expert_tensor_count_status = present_expert_roles > 0u ? "known" : "missing";
    report.expert_tensor_count = present_expert_roles;
    report.expert_count_status = "unknown";
    report.active_expert_count_status = "unknown";
    report.shared_expert_collection_status = "unknown";
    report.shared_expert_status = "unknown";
    report.shared_expert_roles = "unknown";
    report.shared_expert_count_status = "unknown";
    report.dispatch_metadata_status = "unknown";
    report.dispatch_metadata_roles = "unknown";
    report.expert_indexing_policy_status = "unknown";
    report.expert_storage_pressure_status = "unknown";
    report.expert_storage_pressure = strcmp(report.target_class, "selected-runtime-slice") == 0
                                         ? "selected-slice-missing"
                                         : "expert-dominant-expected";
    report.expert_residency_pressure_status = "unknown";
    report.expert_residency_pressure = strcmp(report.target_class, "selected-runtime-slice") == 0
                                           ? "selected-slice-missing"
                                           : "requires-expert-residency-plan";
    report.present_roles = present_roles;
    report.missing_roles = missing_roles;
    report.unknown_roles = "moe_shared_expert,moe_dispatch_metadata,expert_indexing_policy";
    report.not_applicable_roles = "none";
    report.blocked_rows = "V010.TENSOR.14,V010.TENSOR.15,V010.TENSOR.16,V010.TENSOR.17,V010.MOE.5,"
        "V010.MOE.6,V010.MOE.7,V010.PREFILL.7,V010.DECODE.6";
    report.next_required_rows = router || present_expert_roles > 0u
                                    ? "V010.STORAGE.18,V010.RESIDENCY.14,V010.MOE.5,V010.MOE.6,V010.MOE.7"
                                    : "V010.TENSOR.14,V010.TENSOR.15,V010.TENSOR.16,V010.TENSOR.17,"
                                        "V010.TARGET.9,V010.MAP.2,V010.FULLMODEL.6";
    tensor_collection_print_report(&report);
    return 0;
}

int yvex_tensor_collection_command(int arg_count, char **args)
{
    yvex_cli_tensor_collection_options options;
    yvex_model_ref_options ref_options;
    yvex_model_ref ref;
    yvex_model_context ctx;
    yvex_error err;
    const char *target_id;
    const char *target_class;
    yvex_arch arch;
    unsigned long long artifact_bytes = 0ull;
    int selected_target;
    int rc;

    yvex_error_clear(&err);
    memset(&options, 0, sizeof(options));
    memset(&ref, 0, sizeof(ref));
    memset(&ctx, 0, sizeof(ctx));

    rc = parse_tensor_collection_options(arg_count, args, &options);
    if (rc == 1) return 0;
    if (rc != 0) return rc;

    if (strcmp(options.backend, "cpu") != 0 && strcmp(options.backend, "cuda") != 0) {
        return tensor_collection_print_invalid_argument_report(&options,
                                                              "unsupported-backend",
                                                              "backend must be cpu or cuda");
    }
    if (!options.collection || strcmp(options.collection, "moe") != 0) {
        return tensor_collection_print_invalid_argument_report(&options,
                                                              "invalid-argument",
                                                              "only --collection moe is implemented");
    }
    if (strcmp(options.model, "glm-5.2-official-safetensors") == 0) {
        return tensor_collection_print_source_only_report(&options, "glm-5.2-official-safetensors");
    }

    memset(&ref_options, 0, sizeof(ref_options));
    ref_options.allow_registry = 1;
    ref_options.registry_path = options.registry_path;
    rc = yvex_model_ref_resolve(&ref, options.model, &ref_options, &err);
    if (rc != YVEX_OK) {
        yvex_error_clear(&err);
        return tensor_collection_print_missing_model_report(&options, "model or alias could not be resolved");
    }
    if (!fullmodel_file_size(ref.path, &artifact_bytes)) {
        rc = tensor_collection_print_missing_model_report(&options, "resolved artifact path does not exist");
        yvex_model_ref_clear(&ref);
        return rc;
    }

    rc = yvex_model_context_open(ref.path, &ctx, &err);
    if (rc != YVEX_OK) {
        const char *reason = yvex_error_message(&err);
        rc = tensor_collection_print_missing_model_report(&options,
            reason && reason[0] ? reason : "GGUF metadata or tensor directory parse failed");
        yvex_error_clear(&err);
        yvex_model_ref_clear(&ref);
        return rc;
    }

    arch = yvex_model_arch(ctx.model);
    target_id = ref.alias && ref.alias[0] ? ref.alias : "path";
    selected_target = fullmodel_is_selected_target(target_id) ||
                      fullmodel_is_selected_target(options.model);
    target_class = selected_target ? "selected-runtime-slice" : "candidate-GGUF-path";
    rc = tensor_collection_print_model_report(&options, &ref, &ctx, target_id, target_class, arch);
    yvex_model_context_close(&ctx);
    yvex_model_ref_clear(&ref);
    return rc;
}

void yvex_tensor_collection_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex inspect tensor collection report --model FILE_OR_ALIAS --collection moe [--family auto|"
            "deepseek|glm|qwen] [--backend cpu|cuda] [--registry FILE] [--audit | --output normal|table|audit] "
            "[--include-router] [--include-experts] [--include-shared] [--include-dispatch] [--include-"
            "storage] [--include-residency] [--include-blockers]\n");
    yvex_cli_out_writef(fp, "\nExamples:\n");
    yvex_cli_out_writef(fp,
        "  yvex inspect tensor collection report --model deepseek4-v4-flash-dspark-selected-embed-rmsnorm --family "
            "deepseek --collection moe --backend cpu --include-router --include-experts --include-blockers\n");
    yvex_cli_out_writef(fp,
        "  yvex inspect tensor collection report --model glm-5.2-official-safetensors --family glm --collection "
            "moe --backend cpu --include-blockers\n");
    yvex_cli_out_lines(fp, literal_pair_0, sizeof(literal_pair_0) / sizeof(literal_pair_0[0]));
    yvex_cli_out_writef(fp, "  Default output is compact. Use --audit for full diagnostic fields.\n");
    yvex_cli_out_lines(fp, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
}
