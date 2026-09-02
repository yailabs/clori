/* Orchestrate one admitted source-to-ready model workflow over existing domain owners. */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/model_artifacts/private.h"

#include <yvex/internal/compilation.h>
#include <yvex/internal/deployment.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/source_catalog.h>
#include <yvex/internal/source_payload.h>
#include <yvex/quant.h>

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

typedef struct {
    const char *model, *models_root, *registry_path, *quant, *imatrix;
    int dry_run, json;
} model_prepare_options;

typedef struct {
    yvex_operator_paths operator_paths;
    const yvex_model_library_entry *model;
    const yvex_local_source_record *source;
    const yvex_source_target_identity *source_identity;
    const yvex_graph_execution_binding *execution;
    const yvex_model_deployment_defaults *deployment;
    const char *preset;
    char registry_path[YVEX_PATH_CAP];
    char manifest_path[YVEX_PATH_CAP];
    char plan_path[YVEX_PATH_CAP];
    char artifact_path[YVEX_PATH_CAP];
    char binding_dir[YVEX_PATH_CAP];
    char binding_path[YVEX_PATH_CAP];
    char profile_alias[YVEX_MODEL_LIBRARY_NAME_CAP];
} model_prepare_plan;

#define MODEL_PREPARE_DEFAULT_CONTEXT_CAPACITY 4096ull

static int prepare_value(int argc, char **argv, int *index, const char **out)
{
    if (*index + 1 >= argc || !argv[*index + 1][0]) {
        yvex_cli_out_writef(stderr, "yvex: model prepare %s requires a value\n",
                            argv[*index]);
        return 0;
    }
    *out = argv[++*index];
    return 1;
}

static int prepare_options_parse(int argc, char **argv, model_prepare_options *out)
{
    int index;

    memset(out, 0, sizeof(*out));
    for (index = 3; index < argc; ++index) {
        const char *flag = argv[index];
        const char **field = NULL;

        if (!strcmp(flag, "--models-root")) field = &out->models_root;
        else if (!strcmp(flag, "--registry")) field = &out->registry_path;
        else if (!strcmp(flag, "--quant")) field = &out->quant;
        else if (!strcmp(flag, "--imatrix")) field = &out->imatrix;
        if (field) {
            if (!prepare_value(argc, argv, &index, field)) return 2;
        } else if (!strcmp(flag, "--dry-run")) out->dry_run = 1;
        else if (!strcmp(flag, "--json")) out->json = 1;
        else if (flag[0] == '-') {
            yvex_cli_out_writef(stderr, "yvex: unknown model prepare option: %s\n", flag);
            return 2;
        } else if (!out->model) out->model = flag;
        else {
            yvex_cli_out_writef(stderr, "yvex: model prepare received extra model: %s\n",
                                flag);
            return 2;
        }
    }
    if (!out->model) {
        yvex_cli_out_fputs("yvex: model prepare requires MODEL\n", stderr);
        return 2;
    }
    return 0;
}

static int prepare_blocked(const model_prepare_options *options,
                           const char *model, const char *reason, int status)
{
    if (options->json) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.model.prepare.v1\",\"model\":", stdout);
        yvex_cli_out_json_string(stdout, model);
        yvex_cli_out_fputs(",\"state\":\"BLOCKED\",\"changed\":false,\"blocker\":",
                           stdout);
        yvex_cli_out_json_string(stdout, reason);
        yvex_cli_out_fputs("}\n", stdout);
    } else {
        yvex_cli_out_writef(stderr, "yvex: %s\n", reason);
    }
    return status;
}

static int prepare_library_open(const model_prepare_options *options,
                                yvex_model_library **library,
                                unsigned long long *model_index)
{
    yvex_local_catalog_options open = {options->models_root, options->registry_path};
    yvex_error err;
    int matches, rc;

    yvex_error_clear(&err);
    rc = yvex_model_library_open(library, &open, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    matches = yvex_cli_model_find(*library, options->model, model_index);
    if (matches == 1) return 0;
    yvex_cli_out_writef(stderr, "yvex: model selector %s: %s\n", options->model,
                        matches ? "ambiguous" : "not found; use `yvex model pull` first");
    yvex_model_library_close(*library);
    *library = NULL;
    return 2;
}

static const yvex_local_source_record *prepare_exact_source(
    const yvex_model_library *library, unsigned long long model_index,
    const yvex_source_target_identity **identity_out)
{
    const yvex_local_source_record *selected = NULL;
    unsigned long long index;

    *identity_out = NULL;
    for (index = 0u; index < yvex_model_library_source_count(library, model_index); ++index) {
        const yvex_local_source_record *source =
            yvex_model_library_source_at(library, model_index, index);
        const yvex_source_target_identity *identity = source
            ? yvex_source_target_identity_find_repository(source->repository) : NULL;

        if (!source || !identity || strcmp(source->revision, identity->upstream_revision) ||
            strcmp(source->provider, "huggingface") ||
            strcmp(source->acquisition_state, "source-acquired") || !source->path[0] ||
            access(source->path, R_OK | X_OK) != 0)
            continue;
        if (selected) return NULL;
        selected = source;
        *identity_out = identity;
    }
    return selected;
}

static int prepare_text_path(char out[YVEX_PATH_CAP], const char *directory,
                             const char *name, const char *where, yvex_error *err)
{
    return path_join2(out, YVEX_PATH_CAP, directory, name, err, where);
}

static int prepare_alias(char out[YVEX_MODEL_LIBRARY_NAME_CAP],
                         const char *target, const char *preset, const char *backend)
{
    char source[YVEX_MODEL_LIBRARY_NAME_CAP * 2u];
    size_t input, output = 0u;
    int written = snprintf(source, sizeof(source), "%s-%s-%s", target, preset, backend);

    if (written < 0 || (size_t)written >= sizeof(source)) return 0;
    for (input = 0u; source[input] && output + 1u < YVEX_MODEL_LIBRARY_NAME_CAP; ++input) {
        unsigned char value = (unsigned char)source[input];
        char normalized = (char)(isalnum(value) ? tolower(value) : '-');

        if (normalized == '-' && (!output || out[output - 1u] == '-')) continue;
        out[output++] = normalized;
    }
    while (output && out[output - 1u] == '-') output--;
    out[output] = '\0';
    return output != 0u && source[input] == '\0';
}

static int prepare_plan_paths(const model_prepare_options *options,
                              model_prepare_plan *plan, yvex_error *err)
{
    yvex_paths paths = {0};
    char reports_family[YVEX_PATH_CAP], artifacts_family[YVEX_PATH_CAP];
    char registry_family[YVEX_PATH_CAP], binding_leaf[YVEX_MODEL_LIBRARY_NAME_CAP + 16u];
    char file[YVEX_MODEL_LIBRARY_NAME_CAP + 32u];
    int rc;

    rc = yvex_operator_paths_resolve(&paths, options->models_root,
                                     &plan->operator_paths, err);
    if (rc == YVEX_OK)
        rc = prepare_text_path(reports_family, plan->operator_paths.reports_root,
                               plan->execution->operator_family_key,
                               "model.prepare.paths", err);
    if (rc == YVEX_OK)
        rc = prepare_text_path(artifacts_family, plan->operator_paths.gguf_root,
                               plan->execution->operator_family_key,
                               "model.prepare.paths", err);
    if (rc == YVEX_OK)
        rc = prepare_text_path(registry_family, plan->operator_paths.registry_root,
                               plan->execution->operator_family_key,
                               "model.prepare.paths", err);
    snprintf(file, sizeof(file), "%s.source-manifest.json", plan->source->name);
    if (rc == YVEX_OK)
        rc = prepare_text_path(plan->manifest_path, reports_family, file,
                               "model.prepare.paths", err);
    snprintf(file, sizeof(file), "%s.quant-plan", plan->preset);
    if (rc == YVEX_OK)
        rc = prepare_text_path(plan->plan_path, reports_family, file,
                               "model.prepare.paths", err);
    snprintf(file, sizeof(file), "%s.gguf", plan->preset);
    if (rc == YVEX_OK)
        rc = prepare_text_path(plan->artifact_path, artifacts_family, file,
                               "model.prepare.paths", err);
    snprintf(binding_leaf, sizeof(binding_leaf), "%s-bindings", plan->execution->target_id);
    if (rc == YVEX_OK)
        rc = prepare_text_path(plan->binding_dir, registry_family, binding_leaf,
                               "model.prepare.paths", err);
    if (rc == YVEX_OK && options->registry_path)
        rc = expand_operator_path(options->registry_path, plan->registry_path,
                                  sizeof(plan->registry_path), err, "model.prepare.paths");
    else if (rc == YVEX_OK)
        rc = yvex_model_registry_default_path(plan->registry_path,
                                              sizeof(plan->registry_path), err);
    if (rc == YVEX_OK && !prepare_alias(plan->profile_alias, plan->execution->target_id,
                                        plan->preset, plan->deployment->backend)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "model.prepare.paths",
                       "derived deployment profile alias exceeds capacity");
        rc = YVEX_ERR_BOUNDS;
    }
    return rc;
}

static int prepare_plan_build(const model_prepare_options *options,
                              const yvex_model_library *library,
                              unsigned long long model_index,
                              model_prepare_plan *plan, yvex_error *err)
{
    yvex_quant_policy *policy = NULL;
    yvex_quant_policy_summary summary;
    const char *target;
    int rc;

    memset(plan, 0, sizeof(*plan));
    plan->model = yvex_model_library_at(library, model_index);
    plan->source = prepare_exact_source(library, model_index, &plan->source_identity);
    target = plan->model && plan->model->runtime_target[0]
                 ? plan->model->runtime_target
                 : plan->source_identity ? plan->source_identity->target_id : NULL;
    plan->execution = target ? yvex_graph_execution_find(0u, 0u, target) : NULL;
    plan->deployment = plan->execution ? plan->execution->deployment_defaults : NULL;
    if (!plan->source || !plan->source_identity || !plan->execution ||
        !plan->execution->compiler || !plan->deployment) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "model.prepare",
                       "model has no exact acquired source-to-ready compiler binding");
        return YVEX_ERR_UNSUPPORTED;
    }
    plan->preset = options->quant ? options->quant : plan->deployment->quant_preset;
    memset(&summary, 0, sizeof(summary));
    rc = yvex_quant_policy_preset_open(&policy, plan->preset, err);
    if (rc == YVEX_OK) rc = yvex_quant_policy_get_summary(policy, &summary, err);
    yvex_quant_policy_close(policy);
    if (rc != YVEX_OK) return rc;
    if (summary.requires_imatrix_count && !options->imatrix) {
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "model.prepare",
                        "quant preset %s requires --imatrix FILE", plan->preset);
        return YVEX_ERR_INVALID_ARG;
    }
    return prepare_plan_paths(options, plan, err);
}

static void prepare_render_plan(const model_prepare_options *options,
                                const model_prepare_plan *plan)
{
    const char *selector = yvex_cli_model_selector(plan->model);

    if (options->json) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.model.prepare.v1\",\"model\":", stdout);
        yvex_cli_out_json_string(stdout, selector);
        yvex_cli_out_fputs(",\"state\":\"PLANNED\",\"changed\":false,\"source\":", stdout);
        yvex_cli_out_json_string(stdout, plan->source->path);
        yvex_cli_out_fputs(",\"revision\":", stdout);
        yvex_cli_out_json_string(stdout, plan->source_identity->upstream_revision);
        yvex_cli_out_fputs(",\"target\":", stdout);
        yvex_cli_out_json_string(stdout, plan->execution->target_id);
        yvex_cli_out_fputs(",\"quant\":", stdout);
        yvex_cli_out_json_string(stdout, plan->preset);
        yvex_cli_out_fputs(",\"backend\":", stdout);
        yvex_cli_out_json_string(stdout, plan->deployment->backend);
        yvex_cli_out_fputs(",\"artifact\":", stdout);
        yvex_cli_out_json_string(stdout, plan->artifact_path);
        yvex_cli_out_fputs(",\"profile\":", stdout);
        yvex_cli_out_json_string(stdout, plan->profile_alias);
        yvex_cli_out_fputs("}\n", stdout);
        return;
    }
    yvex_cli_out_writef(stdout,
                        "PREPARE PLAN\n"
                        "  Model      %s\n"
                        "  Source     %s\n"
                        "  Revision   %s\n"
                        "  Target     %s\n"
                        "  Quant      %s\n"
                        "  Backend    %s\n"
                        "  Artifact   %s\n"
                        "  Profile    %s\n",
                        selector, plan->source->path,
                        plan->source_identity->upstream_revision,
                        plan->execution->target_id, plan->preset,
                        plan->deployment->backend, plan->artifact_path,
                        plan->profile_alias);
}

static int prepare_parents(const model_prepare_plan *plan, yvex_error *err)
{
    const char *paths[] = {plan->manifest_path, plan->plan_path, plan->artifact_path,
                           plan->registry_path};
    char binding_anchor[YVEX_PATH_CAP];
    size_t index;
    int rc = YVEX_OK;

    for (index = 0u; rc == YVEX_OK && index < sizeof(paths) / sizeof(paths[0]); ++index)
        rc = yvex_core_mkdir_parent(paths[index], "model.prepare", err);
    if (rc == YVEX_OK)
        rc = prepare_text_path(binding_anchor, plan->binding_dir, ".binding",
                               "model.prepare", err);
    if (rc == YVEX_OK)
        rc = yvex_core_mkdir_parent(binding_anchor, "model.prepare", err);
    return rc;
}

static int prepare_source_verify(const model_prepare_plan *plan,
                                 yvex_source_payload_verification_result *result,
                                 yvex_error *err)
{
    yvex_source_manifest_options manifest = {0};
    yvex_source_manifest_summary summary = {0};
    yvex_source_verify_options verification = {0};
    yvex_source_payload_budget budget;
    yvex_source_payload_failure failure = {0};
    int rc;

    if (access(plan->manifest_path, F_OK) != 0) {
        manifest.repo = plan->source_identity->upstream_repo_id;
        manifest.revision = plan->source_identity->upstream_revision;
        manifest.local_path = plan->source->path;
        manifest.node_name = plan->source->name;
        manifest.download_command = "yvex model pull";
        manifest.status = YVEX_SOURCE_STATUS_IN_PROGRESS;
        manifest.include_files = 1;
        rc = yvex_source_manifest_write_json(plan->manifest_path, &manifest, &summary, err);
        if (rc != YVEX_OK) return rc;
    }
    verification.identity = plan->source_identity;
    verification.source_path = plan->source->path;
    verification.models_root = plan->operator_paths.models_root;
    verification.manifest_path = plan->manifest_path;
    verification.promote_manifest = 1;
    yvex_source_payload_budget_default(&budget);
    budget.allow_local_snapshot_seal = 0;
    memset(result, 0, sizeof(*result));
    return yvex_source_payload_verify_snapshot(&verification, &budget, result, &failure, err);
}

static int prepare_quant_plan(const model_prepare_options *options,
                              const model_prepare_plan *plan, yvex_error *err)
{
    char *argv[24];
    int argc = 0, rc;

    argv[argc++] = "yvex"; argv[argc++] = "quant";
    argv[argc++] = access(plan->plan_path, F_OK) == 0 ? "summarize" : "plan";
    argv[argc++] = "--target"; argv[argc++] = (char *)plan->execution->target_id;
    argv[argc++] = "--source"; argv[argc++] = (char *)plan->source->path;
    argv[argc++] = "--models-root";
    argv[argc++] = (char *)plan->operator_paths.models_root;
    argv[argc++] = "--source-manifest"; argv[argc++] = (char *)plan->manifest_path;
    argv[argc++] = "--preset"; argv[argc++] = (char *)plan->preset;
    if (options->imatrix) {
        argv[argc++] = "--imatrix-manifest"; argv[argc++] = (char *)options->imatrix;
    }
    argv[argc++] = "--backend"; argv[argc++] = (char *)plan->deployment->backend;
    if (!strcmp(argv[2], "summarize")) {
        argv[argc++] = "--plan"; argv[argc++] = (char *)plan->plan_path;
    } else {
        argv[argc++] = "--out-plan"; argv[argc++] = (char *)plan->plan_path;
    }
    rc = yvex_quant_command_execute(argc, argv, 0);
    if (rc) yvex_error_set(err, YVEX_ERR_STATE, "model.prepare", "quant plan failed");
    return rc ? YVEX_ERR_STATE : YVEX_OK;
}

static int prepare_quant_emit(const model_prepare_options *options,
                              const model_prepare_plan *plan, yvex_error *err)
{
    char *argv[24];
    int argc = 0, rc;

    if (access(plan->artifact_path, F_OK) == 0) return YVEX_OK;
    argv[argc++] = "yvex"; argv[argc++] = "quant"; argv[argc++] = "emit";
    argv[argc++] = "--target"; argv[argc++] = (char *)plan->execution->target_id;
    argv[argc++] = "--source"; argv[argc++] = (char *)plan->source->path;
    argv[argc++] = "--models-root";
    argv[argc++] = (char *)plan->operator_paths.models_root;
    argv[argc++] = "--source-manifest"; argv[argc++] = (char *)plan->manifest_path;
    argv[argc++] = "--preset"; argv[argc++] = (char *)plan->preset;
    if (options->imatrix) {
        argv[argc++] = "--imatrix-manifest"; argv[argc++] = (char *)options->imatrix;
    }
    argv[argc++] = "--backend"; argv[argc++] = (char *)plan->deployment->backend;
    argv[argc++] = "--plan"; argv[argc++] = (char *)plan->plan_path;
    argv[argc++] = "--out"; argv[argc++] = (char *)plan->artifact_path;
    rc = yvex_quant_command_execute(argc, argv, 0);
    if (rc) yvex_error_set(err, YVEX_ERR_STATE, "model.prepare", "artifact emission failed");
    return rc ? YVEX_ERR_STATE : YVEX_OK;
}

static int prepare_binding(model_prepare_plan *plan,
                           const model_prepare_options *options,
                           int *published, yvex_error *err)
{
    yvex_compilation_runtime_binding_request request = {0};

    request.source_path = plan->source->path;
    request.models_root = plan->operator_paths.models_root;
    request.source_manifest_path = plan->manifest_path;
    request.artifact_path = plan->artifact_path;
    request.directory = plan->binding_dir;
    request.quant_preset_name = plan->preset;
    request.imatrix_path = options->imatrix;
    request.physical_variant_plan_path = plan->plan_path;
    request.family_adapter_id = plan->execution->adapter_id;
    request.family_adapter_version = plan->execution->adapter_version;
    return yvex_runtime_binding_compile_publish(plan->execution->compiler, &request,
                                                plan->binding_path, published, err);
}

static int prepare_profile(const model_prepare_options *options,
                           const model_prepare_plan *plan)
{
    char context[32];
    char *argv[40];
    int argc = 0;

    snprintf(context, sizeof(context), "%llu", MODEL_PREPARE_DEFAULT_CONTEXT_CAPACITY);
    argv[argc++] = "yvex"; argv[argc++] = "models"; argv[argc++] = "add";
    argv[argc++] = "--path"; argv[argc++] = (char *)plan->artifact_path;
    argv[argc++] = "--alias"; argv[argc++] = (char *)plan->profile_alias;
    argv[argc++] = "--family"; argv[argc++] = (char *)plan->deployment->logical_family;
    argv[argc++] = "--model"; argv[argc++] = (char *)plan->deployment->logical_model;
    argv[argc++] = "--scope"; argv[argc++] = "runtime";
    argv[argc++] = "--class"; argv[argc++] = "prepared";
    argv[argc++] = "--qprofile"; argv[argc++] = (char *)plan->preset;
    argv[argc++] = "--calibration"; argv[argc++] = options->imatrix ? "imatrix" : "none";
    argv[argc++] = "--support-level"; argv[argc++] = "generation-ready";
    argv[argc++] = "--startup-profile"; argv[argc++] = "single-artifact";
    argv[argc++] = "--runtime-binding"; argv[argc++] = (char *)plan->binding_path;
    argv[argc++] = "--target"; argv[argc++] = (char *)plan->execution->target_id;
    argv[argc++] = "--backend"; argv[argc++] = (char *)plan->deployment->backend;
    argv[argc++] = "--execution-strategy";
    argv[argc++] = (char *)plan->deployment->execution_strategy;
    argv[argc++] = "--ctx"; argv[argc++] = context;
    argv[argc++] = "--registry"; argv[argc++] = (char *)plan->registry_path;
    return yvex_model_profile_create_adapter(argc, argv, 0, 1);
}

static void prepare_render_ready(const model_prepare_options *options,
                                 const model_prepare_plan *plan,
                                 int changed, int binding_published)
{
    const char *selector = yvex_cli_model_selector(plan->model);

    if (options->json) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.model.prepare.v1\",\"model\":", stdout);
        yvex_cli_out_json_string(stdout, selector);
        yvex_cli_out_writef(stdout,
                            ",\"state\":\"READY\",\"changed\":%s,"
                            "\"binding_published\":%s,\"target\":",
                            changed ? "true" : "false",
                            binding_published ? "true" : "false");
        yvex_cli_out_json_string(stdout, plan->execution->target_id);
        yvex_cli_out_fputs(",\"quant\":", stdout);
        yvex_cli_out_json_string(stdout, plan->preset);
        yvex_cli_out_fputs(",\"artifact\":", stdout);
        yvex_cli_out_json_string(stdout, plan->artifact_path);
        yvex_cli_out_fputs(",\"runtime_binding\":", stdout);
        yvex_cli_out_json_string(stdout, plan->binding_path);
        yvex_cli_out_fputs(",\"profile\":", stdout);
        yvex_cli_out_json_string(stdout, plan->profile_alias);
        yvex_cli_out_fputs("}\n", stdout);
    } else {
        yvex_cli_out_writef(stdout,
                            "READY\n"
                            "  Model      %s\n"
                            "  Target     %s\n"
                            "  Format     GGUF\n"
                            "  Quant      %s\n"
                            "  Artifact   %s\n"
                            "  Backend    %s\n"
                            "  Load       yvex model load %s\n",
                            selector, plan->execution->target_id, plan->preset,
                            plan->artifact_path, plan->deployment->backend, selector);
    }
}

static int prepare_already_ready(const model_prepare_options *options,
                                 const yvex_model_library_entry *model)
{
    const char *selector = yvex_cli_model_selector(model);

    if (options->json) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.model.prepare.v1\",\"model\":", stdout);
        yvex_cli_out_json_string(stdout, selector);
        yvex_cli_out_fputs(",\"state\":\"READY\",\"changed\":false}\n", stdout);
    } else {
        yvex_cli_out_writef(stdout,
                            "READY\n  Model      %s\n  Action     none; a launchable representation already exists\n",
                            selector);
    }
    return 0;
}

int yvex_model_prepare_command(int arg_count, char **args)
{
    model_prepare_options options;
    model_prepare_plan plan;
    yvex_model_library *library = NULL;
    yvex_source_payload_verification_result verification;
    yvex_error err;
    unsigned long long model_index = 0u;
    int binding_published = 0, changed = 0, rc;

    rc = prepare_options_parse(arg_count, args, &options);
    if (rc) return rc;
    rc = prepare_library_open(&options, &library, &model_index);
    if (rc) return rc;
    if (yvex_model_library_at(library, model_index)->profile_launchable &&
        !options.quant && !options.imatrix) {
        rc = prepare_already_ready(&options, yvex_model_library_at(library, model_index));
        yvex_model_library_close(library);
        return rc;
    }
    yvex_error_clear(&err);
    rc = prepare_plan_build(&options, library, model_index, &plan, &err);
    if (rc != YVEX_OK) {
        const yvex_model_library_entry *model = yvex_model_library_at(library, model_index);
        const yvex_local_source_record *source =
            yvex_model_library_source_count(library, model_index)
                ? yvex_model_library_source_at(library, model_index, 0u) : NULL;
        const char *reason = source && !strcasecmp(source->format, "GGUF")
            ? "existing GGUF is preserved without requantization, but this "
              "representation has no admitted runtime binding"
            : yvex_error_message(&err);
        rc = prepare_blocked(&options, yvex_cli_model_selector(model), reason,
                             rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
        yvex_model_library_close(library);
        return rc;
    }
    if (options.dry_run) {
        prepare_render_plan(&options, &plan);
        yvex_model_library_close(library);
        return 0;
    }
    if (!options.json) prepare_render_plan(&options, &plan);
    rc = prepare_parents(&plan, &err);
    if (rc == YVEX_OK) {
        if (!options.json) yvex_cli_out_fputs("\n[verify] authenticating source payload\n", stdout);
        rc = prepare_source_verify(&plan, &verification, &err);
    }
    if (rc == YVEX_OK) {
        if (!options.json) yvex_cli_out_fputs("[plan] compiling physical representation\n", stdout);
        changed |= access(plan.plan_path, F_OK) != 0;
        rc = prepare_quant_plan(&options, &plan, &err);
    }
    if (rc == YVEX_OK) {
        if (!options.json) yvex_cli_out_fputs("[materialize] emitting admitted artifact\n", stdout);
        changed |= access(plan.artifact_path, F_OK) != 0;
        rc = prepare_quant_emit(&options, &plan, &err);
    }
    if (rc == YVEX_OK) {
        if (!options.json) yvex_cli_out_fputs("[profile] compiling runtime binding\n", stdout);
        rc = prepare_binding(&plan, &options, &binding_published, &err);
        changed |= binding_published;
    }
    if (rc == YVEX_OK) {
        int profile_rc = prepare_profile(&options, &plan);

        if (profile_rc) {
            yvex_model_library_close(library);
            return profile_rc;
        }
        changed = 1;
    }
    if (rc != YVEX_OK) {
        yvex_model_library_close(library);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    prepare_render_ready(&options, &plan, changed, binding_published);
    yvex_model_library_close(library);
    return 0;
}
