/*
 * Coordinate registry resolution and artifact-preparation diagnostics for the finite CLI lane.
 * Successful checks do not execute or host a model.
 */
#include "src/cli/model_artifacts/private.h"

#include <string.h>
#include <yvex/internal/source.h>

#include <yvex/artifact.h>
#include <yvex/graph.h>

static const char *const literal_pair_0[] = { "boundary: report-only refusal, generation unsupported",
    "status: model-check-unsupported"};

static const char *const literal_pair_1[] = { "reason: artifact exists; pass --overwrite to replace it",
    "status: model-prepare-refused"};

static const char *const literal_pair_2[] = { "reason: source path does not exist",
    "status: model-prepare-fail"};

static const char *const literal_pair_3[] = { "reason: unknown model prepare target",
    "status: model-prepare-unknown-target"};

static const char *const literal_lines_0[] = { "execution_ready: false",
    "graph_execution_ready: false",
    "prefill_ready: false",
    "logits_ready: false",
    "generation: unsupported"};

typedef enum {
    PREPARE_SOURCE_PRESENT = 1u << 0,
    PREPARE_TENSOR_MAP_PRESENT = 1u << 1,
    PREPARE_TENSOR_MAP_INCOMPLETE = 1u << 2,
    PREPARE_OUTPUT_MAP_PRESENT = 1u << 3,
    PREPARE_OUTPUT_MAP_MISSING = 1u << 4,
    PREPARE_TOKENIZER_MAP_PRESENT = 1u << 5,
    PREPARE_ARTIFACT_PRESENT = 1u << 6,
    PREPARE_FAMILY_DEEPSEEK = 1u << 7
} prepare_source_fact;

typedef struct {
    unsigned int required;
    unsigned int forbidden;
    const char *reason;
} prepare_reason_rule;

static const prepare_reason_rule prepare_reason_rules[] = {
    {0u, PREPARE_TENSOR_MAP_PRESENT, "missing tensor map / model class / artifact path"},
    {PREPARE_TENSOR_MAP_INCOMPLETE, PREPARE_TOKENIZER_MAP_PRESENT,
     "incomplete tensor map / tokenizer metadata mapping / artifact path missing"},
    {PREPARE_TENSOR_MAP_INCOMPLETE | PREPARE_TOKENIZER_MAP_PRESENT, 0u,
     "incomplete tensor map / artifact path missing"},
    {0u, PREPARE_OUTPUT_MAP_PRESENT | PREPARE_TOKENIZER_MAP_PRESENT,
     "missing output head map / tokenizer metadata mapping / artifact path missing"},
    {PREPARE_TOKENIZER_MAP_PRESENT, PREPARE_OUTPUT_MAP_PRESENT,
     "missing output head map / artifact path missing"},
    {PREPARE_OUTPUT_MAP_MISSING, PREPARE_TOKENIZER_MAP_PRESENT,
     "output head mapping missing / tokenizer metadata mapping / artifact path missing"},
    {PREPARE_OUTPUT_MAP_MISSING | PREPARE_TOKENIZER_MAP_PRESENT, 0u,
     "output head mapping missing / artifact path missing"},
    {0u, PREPARE_TOKENIZER_MAP_PRESENT, "tokenizer metadata mapping / artifact path missing"},
    {PREPARE_FAMILY_DEEPSEEK, 0u, "complete artifact admission not bound to this request"},
    {0u, 0u, "family quantization plan unimplemented"},
};

typedef struct {
    const char *target;
    const char *source;
    const char *out;
    const char *out_dir;
    const char *models_root;
    const char *registry_path;
    yvex_models_output_mode output_mode;
    int overwrite;
    int dry_run;
    int register_alias;
} yvex_cli_models_prepare_options;

static const yvex_models_option_spec prepare_options[] = {
    {"--source", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_models_prepare_options, source)},
    {"--out", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_models_prepare_options, out)},
    {"--out-dir", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_models_prepare_options, out_dir)},
    {"--models-root", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_models_prepare_options, models_root)},
    {"--registry", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_models_prepare_options, registry_path)},
    {"--overwrite", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_models_prepare_options, overwrite)},
    {"--dry-run", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_models_prepare_options, dry_run)},
    {"--output", YVEX_MODELS_OPTION_OUTPUT,
     offsetof(yvex_cli_models_prepare_options, output_mode)},
};

static int parse_models_prepare_options(int arg_count,
                                        char **args,
                                        yvex_cli_models_prepare_options *options)
{
    int i;

    memset(options, 0, sizeof(*options));
    options->register_alias = 1;
    options->output_mode = YVEX_MODELS_OUTPUT_NORMAL;
    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "yvex: models prepare requires TARGET\n");
        return 2;
    }
    options->target = args[3];
    if (!cli_arg_value_valid(options->target)) {
        yvex_cli_out_writef(stderr, "yvex: models prepare target is empty or invalid\n");
        return 2;
    }
    for (i = 4; i < arg_count; ++i) {
        int handled = 0;
        int rc = parse_models_bound_option("models prepare", arg_count, args, &i,
                                           options, prepare_options,
                                           sizeof(prepare_options) /
                                               sizeof(prepare_options[0]),
                                           &handled);
        if (rc != 0) return rc;
        if (handled) continue;
        if (strcmp(args[i], "--audit") == 0) {
            options->output_mode = YVEX_MODELS_OUTPUT_AUDIT;
        } else if (strcmp(args[i], "--no-register") == 0) {
            options->register_alias = 0;
        } else if (strcmp(args[i], "--json") == 0) {
            yvex_cli_out_writef(stderr,
                "yvex: models prepare JSON output is unsupported; use --output normal|table|audit\n");
            return 2;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown models prepare option: %s\n", args[i]);
            return 2;
        }
    }
    if (options->out && options->out_dir) {
        yvex_cli_out_writef(stderr, "yvex: models prepare --out and --out-dir are mutually exclusive\n");
        return 2;
    }
    return 0;
}

static void print_prepare_common(const yvex_cli_models_prepare_options *options,
                                 const yvex_operator_paths *operator_paths,
                                 const char *source_path,
                                 const char *artifact_path,
                                 const char *manifest_path,
                                 const char *plan_path,
                                 const char *registry_path)
{
    yvex_cli_out_writef(stdout, "models: prepare\n");
    yvex_cli_out_writef(stdout, "target_id: %s\n", options->target);
    yvex_cli_out_writef(stdout, "models_root_source: %s\n", operator_paths->models_root_source);
    yvex_cli_out_writef(stdout, "models_root: %s\n", operator_paths->models_root);
    yvex_cli_out_writef(stdout, "source_path: %s\n", source_path);
    yvex_cli_out_writef(stdout, "artifact_path: %s\n", artifact_path);
    yvex_cli_out_writef(stdout, "source_manifest_path: %s\n", manifest_path);
    yvex_cli_out_writef(stdout, "conversion_plan_path: %s\n", plan_path);
    yvex_cli_out_writef(stdout, "registry_path: %s\n",
        registry_path && registry_path[0] ? registry_path : "unavailable");
    yvex_cli_out_writef(stdout, "alias: deepseek4-v4-flash-dspark-selected-embed\n");
    yvex_cli_out_writef(stdout, "overwrite: %s\n", options->overwrite ? "true" : "false");
    yvex_cli_out_writef(stdout, "dry_run: %s\n", options->dry_run ? "true" : "false");
    yvex_cli_out_writef(stdout, "register: %s\n", options->register_alias ? "true" : "false");
}

static void print_prepare_dry_run_stages(int register_alias)
{
    static const char *planned[] = {
        "resolve-paths",
        "source-manifest",
        "native-weights",
        "tensor-map",
        "convert-plan",
        "convert-emit",
        "inspect",
        "tensors",
        "metadata"
    };
    unsigned long i;

    for (i = 0; i < sizeof(planned) / sizeof(planned[0]); ++i) {
        model_stage_print(planned[i], "planned");
    }
    model_stage_print("registry-remove-existing", register_alias ? "planned" : "skipped");
    model_stage_print("registry-add", register_alias ? "planned" : "skipped");
    model_stage_print("registry-verify", register_alias ? "planned" : "skipped");
}

static const char *prepare_unsupported_reason(const char *target)
{
    if (strcmp(target, "deepseek4-v4-flash-dspark-selected-embed-rmsnorm") == 0) {
        return "segment prepare is planned, not implemented by this preset";
    }
    if (strcmp(target, "glm-5.2-official-safetensors") == 0) {
        return "YVEX-produced GGUF emission for this target is planned, not implemented";
    }
    return NULL;
}

static int print_prepare_unsupported(const char *target)
{
    const char *reason = prepare_unsupported_reason(target);

    yvex_cli_out_writef(stdout, "models: prepare\n");
    yvex_cli_out_writef(stdout, "target_id: %s\n", target);
    model_stage_print("target", "unsupported");
    model_print_runtime_generation("not-performed");
    if (!reason) {
        yvex_cli_out_lines(stdout, literal_pair_3, sizeof(literal_pair_3) / sizeof(literal_pair_3[0]));
        return 2;
    }
    yvex_cli_out_writef(stdout, "reason: %s\n", reason);
    yvex_cli_out_writef(stdout, "status: model-prepare-unsupported\n");
    return exit_for_status(YVEX_ERR_UNSUPPORTED);
}

void prepare_probe_map_sidecar_status(const char *tensor_map_path,
                                             const char *output_head_map_path,
                                             int *tensor_map_incomplete,
                                             int *output_head_map_missing)
{
    char buf[16384];
    char status[64];
    char coverage[64];
    long long unmapped;

    if (tensor_map_incomplete) *tensor_map_incomplete = 0;
    if (output_head_map_missing) *output_head_map_missing = 0;
    if (tensor_map_path && tensor_map_path[0] &&
        yvex_core_file_read_text_prefix(tensor_map_path, buf, sizeof(buf))) {
        if (yvex_json_probe_string_field(buf,
                                         "required_role_coverage_status",
                                         coverage,
                                         sizeof(coverage))) {
            if (strcmp(coverage, "required-groups-present") != 0 &&
                tensor_map_incomplete) {
                *tensor_map_incomplete = 1;
            }
        } else {
            unmapped = model_download_json_i64_field(buf,
                                                     "unmapped_unknown_count");
            if (unmapped > 0 && tensor_map_incomplete) {
                *tensor_map_incomplete = 1;
            }
        }
    }
    if (output_head_map_path && output_head_map_path[0] &&
        yvex_core_file_read_text_prefix(output_head_map_path, buf, sizeof(buf)) &&
        yvex_json_probe_string_field(buf, "output_head_status", status, sizeof(status)) &&
        strcmp(status, "present") != 0 &&
        output_head_map_missing) {
        *output_head_map_missing = 1;
    }
}

static void prepare_report_text(char *out, size_t cap, const char *value, const char *fallback)
{
    snprintf(out, cap, "%s", value && value[0] ? value : fallback);
}

static void prepare_report_path(char *out,
                                size_t cap,
                                const char *root,
                                const char *family,
                                const char *target,
                                const char *suffix)
{
    int n = snprintf(out, cap, "%s/%s/%s.%s", root, family, target, suffix);

    if (n < 0 || (size_t)n >= cap) out[0] = '\0';
}

static const char *prepare_report_reason(unsigned int facts)
{
    size_t index;

    for (index = 0u; index < sizeof(prepare_reason_rules) / sizeof(prepare_reason_rules[0]); ++index) {
        const prepare_reason_rule *rule = &prepare_reason_rules[index];
        if ((facts & rule->required) == rule->required && !(facts & rule->forbidden)) return rule->reason;
    }
    return "family quantization plan unimplemented";
}

static void prepare_source_report_build(
    const yvex_cli_models_prepare_options *options,
    const yvex_operator_paths *operator_paths,
    const yvex_model_download_resolved_target *target,
    yvex_models_prepare_source_report *report)
{
    unsigned int facts = 0u;
    int tensor_map_incomplete = 0;
    int output_head_map_missing = 0;
    int has_identity = operator_paths && target && target->family[0] && target->target_id[0];

    memset(report, 0, sizeof(*report));
    prepare_report_text(report->target_id, sizeof(report->target_id),
                        target ? target->target_id : NULL, options ? options->target : "unknown");
    prepare_report_text(report->family, sizeof(report->family), target ? target->family : NULL, "unknown");
    prepare_report_text(report->provider, sizeof(report->provider), target ? target->provider : NULL, "huggingface");
    prepare_report_text(report->repo_id, sizeof(report->repo_id), target ? target->repo_id : NULL, "unknown");
    prepare_report_text(report->revision, sizeof(report->revision), target ? target->revision : NULL, "main");
    prepare_report_text(report->models_root, sizeof(report->models_root),
                        operator_paths ? operator_paths->models_root : NULL, "unknown");
    prepare_report_text(report->source_path, sizeof(report->source_path),
                        target ? target->local_source_dir : NULL, "unknown");
    prepare_report_text(report->source_manifest_path, sizeof(report->source_manifest_path),
                        target ? target->manifest_path : NULL, "unknown");
    prepare_report_text(report->native_inventory_path, sizeof(report->native_inventory_path),
                        target ? target->native_inventory_path : NULL, "unknown");
    prepare_report_text(report->download_registry_path, sizeof(report->download_registry_path),
                        target ? target->registry_path : NULL, "unknown");
    prepare_report_text(report->download_report_path, sizeof(report->download_report_path),
                        target ? target->download_report_path : NULL, "unknown");

    if (target && target->local_source_dir[0] && path_exists(target->local_source_dir)) {
        facts |= PREPARE_SOURCE_PRESENT;
    }
    if (target && strcmp(target->family, "deepseek") == 0) facts |= PREPARE_FAMILY_DEEPSEEK;
    if (has_identity) {
        prepare_report_path(report->expected_artifact_path, sizeof(report->expected_artifact_path),
                            operator_paths->gguf_root, target->family, target->target_id, "gguf");
        prepare_report_path(report->tensor_map_path, sizeof(report->tensor_map_path),
                            operator_paths->reports_root, target->family, target->target_id, "tensor-map.json");
        prepare_report_path(report->output_head_map_path, sizeof(report->output_head_map_path),
                            operator_paths->reports_root, target->family, target->target_id, "output-head-map.json");
        prepare_report_path(report->tokenizer_map_path, sizeof(report->tokenizer_map_path),
                            operator_paths->reports_root, target->family, target->target_id, "tokenizer-map.json");
        if (report->expected_artifact_path[0] && path_exists(report->expected_artifact_path)) {
            facts |= PREPARE_ARTIFACT_PRESENT;
        }
        if (report->tensor_map_path[0] && path_exists(report->tensor_map_path)) facts |= PREPARE_TENSOR_MAP_PRESENT;
        if (report->output_head_map_path[0] && path_exists(report->output_head_map_path)) {
            facts |= PREPARE_OUTPUT_MAP_PRESENT;
        }
        if (report->tokenizer_map_path[0] && path_exists(report->tokenizer_map_path)) {
            facts |= PREPARE_TOKENIZER_MAP_PRESENT;
        }
        prepare_probe_map_sidecar_status(report->tensor_map_path, report->output_head_map_path,
                                         &tensor_map_incomplete, &output_head_map_missing);
        if (tensor_map_incomplete) facts |= PREPARE_TENSOR_MAP_INCOMPLETE;
        if (output_head_map_missing) facts |= PREPARE_OUTPUT_MAP_MISSING;
    } else {
        prepare_report_text(report->expected_artifact_path, sizeof(report->expected_artifact_path), NULL, "unknown");
        snprintf(report->tensor_map_path, sizeof(report->tensor_map_path), "unknown");
        snprintf(report->output_head_map_path, sizeof(report->output_head_map_path), "unknown");
        snprintf(report->tokenizer_map_path, sizeof(report->tokenizer_map_path), "unknown");
    }

    if (!(facts & PREPARE_SOURCE_PRESENT)) report->blocker_count++;
    if (!(facts & PREPARE_TENSOR_MAP_PRESENT) || (facts & PREPARE_TENSOR_MAP_INCOMPLETE)) report->blocker_count++;
    if (!(facts & PREPARE_OUTPUT_MAP_PRESENT) || (facts & PREPARE_OUTPUT_MAP_MISSING)) report->blocker_count++;
    if (!(facts & PREPARE_TOKENIZER_MAP_PRESENT)) report->blocker_count++;
    if (!(facts & PREPARE_ARTIFACT_PRESENT)) report->blocker_count++;
    report->blocker_count++; /* full artifact emitter/identity is not implemented for dynamic source targets. */

    report->source_status = facts & PREPARE_SOURCE_PRESENT ? "present" : "missing";
    report->model_class_status = report->source_status;
    report->tensor_map_status = !(facts & PREPARE_TENSOR_MAP_PRESENT) ? "missing" :
        facts & PREPARE_TENSOR_MAP_INCOMPLETE ? "incomplete-report-only" : "present-report-only";
    report->output_head_map_status = !(facts & PREPARE_OUTPUT_MAP_PRESENT) ? "missing" :
        facts & PREPARE_OUTPUT_MAP_MISSING ? "missing-in-report" : "present-report-only";
    report->tokenizer_map_status = facts & PREPARE_TOKENIZER_MAP_PRESENT ? "present-report-only" : "missing";
    report->artifact_status = facts & PREPARE_ARTIFACT_PRESENT ? "present" : "missing";
    report->artifact_plan_status = "planned-full-gguf";
    report->artifact_emission_status = "not-performed";
    report->artifact_identity_status = facts & PREPARE_ARTIFACT_PRESENT ? "not-checked" : "missing";
    if (!(facts & PREPARE_TENSOR_MAP_PRESENT) || (facts & PREPARE_TENSOR_MAP_INCOMPLETE) ||
        !(facts & PREPARE_OUTPUT_MAP_PRESENT) || (facts & PREPARE_OUTPUT_MAP_MISSING)) {
        report->next = "V010.MAP.8";
    } else if (!(facts & PREPARE_TOKENIZER_MAP_PRESENT)) {
        report->next = "V010.MAP.7";
    } else {
        report->next = facts & PREPARE_FAMILY_DEEPSEEK ? "V010.ARTIFACT.MATERIALIZE.0" : "not-scheduled";
    }
    report->final_status = "model-prepare-unsupported";
    report->downloaded_target_resolved = 1;

    if (!(facts & PREPARE_SOURCE_PRESENT)) {
        report->top_blocker = "missing-source";
    } else if (!(facts & PREPARE_OUTPUT_MAP_PRESENT) || (facts & PREPARE_OUTPUT_MAP_MISSING)) {
        report->top_blocker = "missing-output-head-map";
    } else if (!(facts & PREPARE_TENSOR_MAP_PRESENT) || (facts & PREPARE_TENSOR_MAP_INCOMPLETE)) {
        report->top_blocker = "incomplete-tensor-map";
    } else if (!(facts & PREPARE_TOKENIZER_MAP_PRESENT)) {
        report->top_blocker = "missing-tokenizer-map";
    } else {
        report->top_blocker = facts & PREPARE_FAMILY_DEEPSEEK ? "complete-artifact-admission-required" :
            "family-quantization-plan-unimplemented";
    }
    report->reason = prepare_report_reason(facts);
}

static int print_prepare_downloaded_source_unsupported(
    const yvex_cli_models_prepare_options *options,
    const yvex_operator_paths *operator_paths,
    const yvex_model_download_resolved_target *target)
{
    yvex_models_prepare_source_report report;

    prepare_source_report_build(options, operator_paths, target, &report);
    model_prepare_source_report_render(
        &report,
        options ? options->output_mode : YVEX_MODELS_OUTPUT_NORMAL);
    return exit_for_status(YVEX_ERR_UNSUPPORTED);
}

/*
 * Reopen one registered artifact and compare its physical identity and metadata snapshot.
 *
 * Reads the artifact identity and metadata without modifying the registry or artifact. Propagates
 * read failures or returns state refusal on identity/metadata drift.
 */
static int registry_snapshot_verify(const yvex_model_registry_entry *entry,
                                    const char *path,
                                    const char *where,
                                    const char *identity_failure,
                                    const char *metadata_failure,
                                    yvex_error *err)
{
    yvex_artifact_file_identity identity;
    yvex_model_metadata_snapshot current_metadata;
    yvex_model_metadata_drift_report metadata_report;
    int rc;

    memset(&identity, 0, sizeof(identity));
    memset(&current_metadata, 0, sizeof(current_metadata));
    memset(&metadata_report, 0, sizeof(metadata_report));
    rc = yvex_artifact_identity_read(path, &identity, err);
    if (rc != YVEX_OK) return rc;
    if (!entry->sha256 || strcmp(entry->sha256, identity.sha256) != 0 ||
        (entry->file_size && entry->file_size != identity.file_size)) {
        yvex_error_set(err, YVEX_ERR_STATE, where, identity_failure);
        return YVEX_ERR_STATE;
    }
    rc = yvex_model_metadata_snapshot_read(&current_metadata, path, err);
    if (rc != YVEX_OK) return rc;
    rc = yvex_model_registry_compare_metadata(entry, &current_metadata.entry, &metadata_report, err);
    if (rc != YVEX_OK) return rc;
    if (strcmp(metadata_report.metadata_status, "pass") != 0 ||
        strcmp(metadata_report.readiness_status, "pass") != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, where, metadata_failure);
        return YVEX_ERR_STATE;
    }
    return YVEX_OK;
}

static int prepare_registry_verify(const yvex_model_registry_entry *entry, yvex_error *err)
{
    if (!entry) {
        yvex_error_set(err, YVEX_ERR_STATE, "model_prepare_registry",
                       "prepared alias was not found after registration");
        return YVEX_ERR_STATE;
    }
    return registry_snapshot_verify(entry, entry->path, "model_prepare_registry",
                                    "registered identity does not match emitted artifact",
                                    "registered metadata drifted immediately after prepare", err);
}

typedef struct {
    yvex_model_registry *registry;
    yvex_model_registry_entry derived;
    yvex_model_registry_entry entry;
    char sha256[YVEX_SHA256_HEX_CAP];
    char format[16];
    char architecture[64];
    char tensor_name[128];
    char tensor_role[64];
    char tensor_dtype[32];
    char tensor_dims[128];
} prepare_registry_state;

static int prepare_register_artifact(const char *artifact_path,
                                     const char *registry_path,
                                     const char *target_alias,
                                     yvex_error *err)
{
    prepare_registry_state state;
    const yvex_model_registry_entry *registered;
    int rc;

    memset(&state, 0, sizeof(state));
    rc = yvex_model_registry_entry_derive_from_path(&state.derived, artifact_path, err);
    if (rc != YVEX_OK) return rc;
    state.entry = state.derived;
    state.entry.support_level = "selected-tensor-materialized";
    rc = populate_registry_identity(&state.entry, state.sha256, state.format,
                                    state.architecture, state.tensor_name,
                                    state.tensor_role, state.tensor_dtype,
                                    state.tensor_dims, err);
    if (rc != YVEX_OK) return rc;
    rc = models_registry_open(&state.registry, registry_path, 1, err);
    if (rc != YVEX_OK) return rc;
    if (yvex_model_registry_find(state.registry, target_alias)) {
        rc = yvex_model_registry_remove(state.registry, target_alias, err);
        model_stage_print("registry-remove-existing", rc == YVEX_OK ? "pass" : "fail");
    } else {
        model_stage_print("registry-remove-existing", "not-found");
    }
    if (rc == YVEX_OK) rc = yvex_model_registry_add(state.registry, &state.entry, err);
    if (rc == YVEX_OK) model_stage_print("registry-add", "pass");
    if (rc == YVEX_OK) rc = yvex_model_registry_save(state.registry, registry_path, err);
    if (rc == YVEX_OK) {
        registered = yvex_model_registry_find(state.registry, target_alias);
        rc = prepare_registry_verify(registered, err);
        if (rc == YVEX_OK) model_stage_print("registry-verify", "pass");
    }
    yvex_model_registry_close(state.registry);
    return rc;
}

static int prepare_complete(void)
{
    model_print_runtime_generation("not-performed");
    yvex_cli_out_writef(stdout, "status: model-prepare\n");
    return 0;
}

static int prepare_create_parents(const char *artifact_path,
                                  const char *manifest_path,
                                  const char *plan_path,
                                  const char *registry_path,
                                  int register_alias,
                                  yvex_error *err)
{
    const char *paths[] = {artifact_path, manifest_path, plan_path, registry_path};
    size_t count = register_alias ? 4u : 3u;
    size_t index;

    for (index = 0u; index < count; ++index) {
        int rc = yvex_core_mkdir_parent(paths[index], "model_registry_json", err);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

int yvex_models_prepare_surface_command(int arg_count, char **args)
{
    static const char *target_alias = "deepseek4-v4-flash-dspark-selected-embed";
    static const char *artifact_name = "deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf";
    yvex_cli_models_prepare_options options;
    yvex_paths paths;
    yvex_operator_paths operator_paths;
    yvex_source_manifest_options manifest_options;
    yvex_source_manifest_summary manifest_summary;
    yvex_native_weight_options native_options;
    yvex_native_weight_table *native_table = NULL;
    yvex_native_weight_summary native_summary;
    yvex_conversion_options conversion_options;
    yvex_conversion_summary conversion_summary;
    yvex_model_metadata_snapshot metadata_snapshot;
    yvex_error err;
    char source_path[YVEX_PATH_CAP];
    char out_dir[YVEX_PATH_CAP];
    char artifact_path[YVEX_PATH_CAP];
    char manifest_path[YVEX_PATH_CAP];
    char plan_path[YVEX_PATH_CAP];
    char registry_path_buf[YVEX_PATH_CAP];
    const char *registry_path = NULL;
    int source_exists;
    int artifact_exists;
    int rc;

    rc = parse_models_prepare_options(arg_count, args, &options);
    if (rc != 0) return rc;

    memset(&paths, 0, sizeof(paths));
    memset(&operator_paths, 0, sizeof(operator_paths));
    yvex_error_clear(&err);
    rc = yvex_operator_paths_resolve(&paths, options.models_root, &operator_paths, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));

    if (strcmp(options.target, target_alias) != 0) {
        yvex_model_download_resolved_target downloaded;

        yvex_error_clear(&err);
        if (model_download_resolve_downloaded_target(options.target,
                                                     &operator_paths,
                                                     &downloaded,
                                                     &err)) {
            return print_prepare_downloaded_source_unsupported(&options,
                                                               &operator_paths,
                                                               &downloaded);
        }
        return print_prepare_unsupported(options.target);
    }

    rc = yvex_operator_paths_resolve_target(&operator_paths, "deepseek", "source",
                                            source_path, sizeof(source_path),
                                            &source_exists, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    if (options.source) {
        rc = expand_operator_path(options.source, source_path, sizeof(source_path), &err, "models_prepare");
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        source_exists = path_exists(source_path);
    }

    if (options.out) {
        rc = expand_operator_path(options.out, artifact_path, sizeof(artifact_path), &err, "models_prepare");
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        if (!path_parent_dir(artifact_path, out_dir, sizeof(out_dir))) {
            yvex_error_set(&err, YVEX_ERR_BOUNDS, "models_prepare", "output parent path is too long");
            return print_yvex_error(&err, exit_for_status(YVEX_ERR_BOUNDS));
        }
    } else {
        if (options.out_dir) {
            rc = expand_operator_path(options.out_dir, out_dir, sizeof(out_dir), &err, "models_prepare");
            if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        } else {
            rc = yvex_operator_paths_resolve_target(&operator_paths, "deepseek", "gguf",
                                                    out_dir, sizeof(out_dir),
                                                    &artifact_exists, &err);
            if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        }
        rc = path_join2(artifact_path, sizeof(artifact_path), out_dir, artifact_name, &err, "models_prepare");
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    }
    artifact_exists = path_exists(artifact_path);

    rc = path_join2(manifest_path, sizeof(manifest_path), out_dir,
                    YVEX_SOURCE_RELEASE_MANIFEST_LEAF, &err, "models_prepare");
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    rc = path_join2(plan_path, sizeof(plan_path), out_dir, "deepseek-selected-plan.json", &err, "models_prepare");
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));

    if (options.registry_path) {
        rc = expand_operator_path(options.registry_path, registry_path_buf, sizeof(registry_path_buf), &err,
            "models_prepare");
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        registry_path = registry_path_buf;
    } else {
        rc = yvex_model_registry_default_path(registry_path_buf, sizeof(registry_path_buf), &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        registry_path = registry_path_buf;
    }

    print_prepare_common(&options, &operator_paths, source_path, artifact_path,
                         manifest_path, plan_path, registry_path);

    if (options.dry_run) {
        print_prepare_dry_run_stages(options.register_alias);
        model_print_runtime_generation("not-performed");
        yvex_cli_out_writef(stdout, "status: model-prepare-dry-run\n");
        return 0;
    }

    if (!source_exists) {
        model_stage_print("source-path", "fail");
        model_print_runtime_generation("not-performed");
        yvex_cli_out_lines(stdout, literal_pair_2, sizeof(literal_pair_2) / sizeof(literal_pair_2[0]));
        return exit_for_status(YVEX_ERR_IO);
    }
    if (artifact_exists && !options.overwrite) {
        model_stage_print("convert-emit", "refused");
        model_print_runtime_generation("not-performed");
        yvex_cli_out_lines(stdout, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
        return exit_for_status(YVEX_ERR_STATE);
    }

    rc = prepare_create_parents(artifact_path, manifest_path, plan_path,
                                registry_path, options.register_alias, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));

    memset(&manifest_options, 0, sizeof(manifest_options));
    memset(&manifest_summary, 0, sizeof(manifest_summary));
    manifest_options.repo = yvex_source_release_identity()->upstream_repo_id;
    manifest_options.revision = "main";
    manifest_options.local_path = source_path;
    manifest_options.status = YVEX_SOURCE_STATUS_IN_PROGRESS;
    rc = yvex_source_manifest_write_json(manifest_path, &manifest_options, &manifest_summary, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    model_stage_print("source-manifest", "pass");

    memset(&native_options, 0, sizeof(native_options));
    memset(&native_summary, 0, sizeof(native_summary));
    native_options.source_dir = source_path;
    native_options.recursive = 1;
    rc = yvex_native_weight_table_open(&native_table, &native_options, &err);
    if (rc == YVEX_OK) rc = yvex_native_weight_table_summary(native_table, &native_summary, &err);
    if (rc == YVEX_OK && !yvex_native_weight_table_find(native_table, "embed.weight")) {
        yvex_error_set(&err, YVEX_ERR_STATE, "models_prepare", "required native tensor is missing: embed.weight");
        rc = YVEX_ERR_STATE;
    }
    if (rc != YVEX_OK) {
        yvex_native_weight_table_close(native_table);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    model_stage_print("native-weights", "pass");
    yvex_cli_out_writef(stdout, "native_tensor_count: %llu\n", native_summary.tensor_count);
    model_stage_print("tensor-map", "pass");
    yvex_native_weight_table_close(native_table);
    native_table = NULL;

    memset(&conversion_options, 0, sizeof(conversion_options));
    memset(&conversion_summary, 0, sizeof(conversion_summary));
    conversion_options.architecture = "deepseek4";
    conversion_options.source_manifest_path = manifest_path;
    conversion_options.native_source_dir = source_path;
    conversion_options.tensor_name = "embed.weight";
    conversion_options.target_qtype = "F16";
    conversion_options.out_path = artifact_path;
    conversion_options.overwrite = options.overwrite;
    rc = yvex_conversion_plan_write_json(&conversion_options, plan_path, &conversion_summary, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    model_stage_print("convert-plan", "pass");

    memset(&conversion_summary, 0, sizeof(conversion_summary));
    rc = yvex_conversion_emit_gguf(&conversion_options, &conversion_summary, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    model_stage_print("convert-emit", "pass");
    yvex_cli_out_writef(stdout, "bytes_written: %llu\n", conversion_summary.bytes_written);

    memset(&metadata_snapshot, 0, sizeof(metadata_snapshot));
    rc = yvex_model_metadata_snapshot_read(&metadata_snapshot, artifact_path, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    model_stage_print("inspect", "pass");
    model_stage_print("tensors", "pass");
    model_stage_print("metadata", "pass");
    yvex_cli_out_writef(stdout, "artifact_architecture: %s\n", metadata_snapshot.architecture);
    yvex_cli_out_writef(stdout, "artifact_tensor_count: %llu\n", metadata_snapshot.entry.tensor_count);

    if (options.register_alias) {
        rc = prepare_register_artifact(artifact_path, registry_path,
                                       target_alias, &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    } else {
        model_stage_print("registry-remove-existing", "skipped");
        model_stage_print("registry-add", "skipped");
        model_stage_print("registry-verify", "skipped");
    }

    return prepare_complete();
}

/* Native source tensor download lane. */

#define YVEX_MODEL_DOWNLOAD_PATTERN_CAP 32u

typedef enum {
    YVEX_CLI_MODEL_CHECK_QUICK = 0,
    YVEX_CLI_MODEL_CHECK_RUNTIME,
    YVEX_CLI_MODEL_CHECK_FULL
} yvex_cli_model_check_level;

/* Selected artifact check preset. */

typedef struct {
    const char *target;
    const char *backend_name;
    const char *level_name;
    const char *models_root;
    const char *registry_path;
    const char *report_dir;
    int no_materialize;
    int no_graph;
    yvex_cli_model_check_level level;
    yvex_models_output_mode output_mode;
} yvex_cli_models_check_options;

static const yvex_models_option_spec check_options[] = {
    {"--backend", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_models_check_options, backend_name)},
    {"--level", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_models_check_options, level_name)},
    {"--models-root", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_models_check_options, models_root)},
    {"--registry", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_models_check_options, registry_path)},
    {"--report-dir", YVEX_MODELS_OPTION_TEXT,
     offsetof(yvex_cli_models_check_options, report_dir)},
    {"--no-materialize", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_models_check_options, no_materialize)},
    {"--no-graph", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_models_check_options, no_graph)},
    {"--output", YVEX_MODELS_OPTION_OUTPUT,
     offsetof(yvex_cli_models_check_options, output_mode)},
};

typedef struct {
    char target_id[256];
    char model_input_kind[32];
    char backend_name[16];
    char level_name[16];
    char artifact_path[YVEX_PATH_CAP];
    char registry_path[YVEX_PATH_CAP];
    char report_path[YVEX_PATH_CAP];
    char error[256];
    char graph_skip_reason[128];
    const char *stage_resolve_target;
    const char *stage_resolve_artifact;
    const char *stage_inspect;
    const char *stage_tensors;
    const char *stage_metadata;
    const char *stage_registry_identity;
    const char *stage_integrity_check;
    const char *stage_integrity_report;
    const char *stage_materialize;
    const char *stage_engine;
    const char *stage_session;
    const char *stage_plan;
    const char *stage_graph_partial;
    const char *stage_model_gate;
    const char *stage_materialize_gate;
    const char *runtime_execution;
    const char *final_status;
} yvex_cli_model_check_report;

typedef struct {
    yvex_cli_models_check_options options;
    yvex_cli_model_check_report report;
    yvex_model_ref ref;
    yvex_model_context context;
    yvex_model_metadata_snapshot metadata;
    yvex_artifact_integrity_report integrity;
    yvex_error error;
    char registry_path[YVEX_PATH_CAP];
} model_check_run;

typedef struct model_check_stage_spec {
    const char *name;
    size_t status_offset;
} model_check_stage_spec;

#define CHECK_STAGE(name_, member_) \
    {name_, offsetof(yvex_cli_model_check_report, member_)}

static const yvex_cli_model_check_report model_check_defaults = {
    .model_input_kind = "unknown",
    .stage_resolve_target = "not-run",
    .stage_resolve_artifact = "not-run",
    .stage_inspect = "not-run",
    .stage_tensors = "not-run",
    .stage_metadata = "not-run",
    .stage_registry_identity = "skipped",
    .stage_integrity_check = "not-run",
    .stage_integrity_report = "skipped",
    .stage_materialize = "skipped",
    .stage_engine = "skipped",
    .stage_session = "skipped",
    .stage_plan = "skipped",
    .stage_graph_partial = "skipped",
    .stage_model_gate = "skipped",
    .stage_materialize_gate = "skipped",
    .runtime_execution = "not-performed",
    .final_status = "model-check-fail",
};

static const model_check_stage_spec model_check_stages[] = {
    CHECK_STAGE("resolve-target", stage_resolve_target),
    CHECK_STAGE("resolve-artifact", stage_resolve_artifact),
    CHECK_STAGE("inspect", stage_inspect),
    CHECK_STAGE("tensors", stage_tensors),
    CHECK_STAGE("metadata", stage_metadata),
    CHECK_STAGE("registry-identity", stage_registry_identity),
    CHECK_STAGE("integrity-check", stage_integrity_check),
    CHECK_STAGE("integrity-report", stage_integrity_report),
    CHECK_STAGE("materialize", stage_materialize),
    CHECK_STAGE("engine", stage_engine),
    CHECK_STAGE("session", stage_session),
    CHECK_STAGE("plan", stage_plan),
    CHECK_STAGE("graph-partial", stage_graph_partial),
};

#undef CHECK_STAGE

static void model_check_report_init(yvex_cli_model_check_report *report,
                                    const yvex_cli_models_check_options *options)
{
    *report = model_check_defaults;
    snprintf(report->target_id, sizeof(report->target_id), "%s",
             options->target ? options->target : "");
    snprintf(report->backend_name, sizeof(report->backend_name), "%s",
             options->backend_name ? options->backend_name : "cpu");
    snprintf(report->level_name, sizeof(report->level_name), "%s",
             options->level_name ? options->level_name : "quick");
}

static void model_check_report_print(FILE *fp,
                                     const yvex_cli_model_check_report *report)
{
    size_t i;

    yvex_cli_out_writef(fp, "status: model-check\n");
    yvex_cli_out_writef(fp, "target_id: %s\n", report->target_id);
    yvex_cli_out_writef(fp, "model_input_kind: %s\n", report->model_input_kind);
    yvex_cli_out_writef(fp, "backend: %s\n", report->backend_name);
    yvex_cli_out_writef(fp, "level: %s\n", report->level_name);
    yvex_cli_out_writef(fp, "artifact_path: %s\n", report->artifact_path);
    yvex_cli_out_writef(fp, "registry_path: %s\n", report->registry_path);
    if (report->report_path[0]) {
        yvex_cli_out_writef(fp, "report_path: %s\n", report->report_path);
    }
    for (i = 0; i < sizeof(model_check_stages) / sizeof(model_check_stages[0]); ++i) {
        const model_check_stage_spec *stage = &model_check_stages[i];
        const char *status = *(const char *const *)((const unsigned char *)report +
                                                   stage->status_offset);

        yvex_cli_out_writef(fp, "stage: %s %s\n", stage->name, status);
    }
    if (report->graph_skip_reason[0]) {
        yvex_cli_out_writef(fp, "graph_partial_reason: %s\n", report->graph_skip_reason);
    }
    yvex_cli_out_writef(fp, "stage: model-gate %s\n", report->stage_model_gate);
    yvex_cli_out_writef(fp, "stage: materialize-gate %s\n", report->stage_materialize_gate);
    if (report->error[0]) {
        yvex_cli_out_writef(fp, "error: %s\n", report->error);
    }
    yvex_cli_out_writef(fp, "runtime_execution: %s\n", report->runtime_execution);
    yvex_cli_out_lines(fp, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
    yvex_cli_out_writef(fp, "status: %s\n", report->final_status);
}

static int model_check_write_report(const yvex_cli_models_check_options *options,
                                    yvex_cli_model_check_report *report,
                                    yvex_error *err)
{
    FILE *fp;
    char report_dir[YVEX_PATH_CAP];
    char report_name[128];
    int rc;
    int n;

    if (!options->report_dir) {
        return YVEX_OK;
    }
    rc = expand_operator_path(options->report_dir, report_dir, sizeof(report_dir),
                              err, "models_check");
    if (rc != YVEX_OK) {
        return rc;
    }
    n = snprintf(report_name, sizeof(report_name),
                 "model-check-deepseek4-v4-flash-dspark-selected-embed-%s-%s.txt",
                 report->backend_name, report->level_name);
    if (n < 0 || (size_t)n >= sizeof(report_name)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "models_check", "report filename is too long");
        return YVEX_ERR_BOUNDS;
    }
    rc = path_join2(report->report_path, sizeof(report->report_path),
                    report_dir, report_name, err, "models_check");
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_core_mkdir_parent(report->report_path, "model_registry_json", err);
    if (rc != YVEX_OK) {
        return rc;
    }
    fp = fopen(report->report_path, "w");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_check",
                        "failed to open report: %s", report->report_path);
        return YVEX_ERR_IO;
    }
    model_check_report_print(fp, report);
    if (fclose(fp) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_check",
                        "failed to close report: %s", report->report_path);
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}

static void model_check_report_print_normal(FILE *fp,
                                            const yvex_cli_model_check_report *report)
{
    if (!fp || !report) return;
    yvex_cli_out_writef(fp, "model-check: %s target=%s level=%s\n",
            strcmp(report->final_status ? report->final_status : "", "model-check-pass") == 0
                ? "pass"
                : "fail",
            report->target_id,
            report->level_name);
    yvex_cli_out_writef(fp, "backend: %s\n", report->backend_name);
    yvex_cli_out_writef(fp, "checked: inspect, metadata, registry, integrity%s%s\n",
            strcmp(report->stage_materialize ? report->stage_materialize : "", "pass") == 0
                ? ", materialize"
                : "",
            strcmp(report->stage_graph_partial ? report->stage_graph_partial : "", "pass") == 0
                ? ", selected graph"
                : "");
    yvex_cli_out_writef(fp, "runtime: %s\n", report->runtime_execution ? report->runtime_execution : "not-performed");
    if (report->error[0]) {
        yvex_cli_out_writef(fp, "top_blocker: %s\n", report->error);
    } else {
        yvex_cli_out_writef(fp, "top_blocker: none\n");
    }
    yvex_cli_out_writef(fp, "boundary: selected-slice check only, generation unsupported\n");
    yvex_cli_out_writef(fp, "status: %s\n", report->final_status ? report->final_status : "model-check-fail");
}

static int parse_models_check_options(int arg_count,
                                      char **args,
                                      yvex_cli_models_check_options *options)
{
    int i;

    memset(options, 0, sizeof(*options));
    options->backend_name = "cpu";
    options->level_name = "quick";
    options->level = YVEX_CLI_MODEL_CHECK_QUICK;
    options->output_mode = YVEX_MODELS_OUTPUT_NORMAL;
    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "yvex: models check requires TARGET\n");
        return 2;
    }
    options->target = args[3];
    if (!cli_arg_value_valid(options->target)) {
        yvex_cli_out_writef(stderr, "yvex: models check target is empty or invalid\n");
        return 2;
    }
    for (i = 4; i < arg_count; ++i) {
        int handled = 0;
        int rc = parse_models_bound_option("models check", arg_count, args, &i,
                                           options, check_options,
                                           sizeof(check_options) / sizeof(check_options[0]),
                                           &handled);
        if (rc != 0) return rc;
        if (handled) continue;
        if (strcmp(args[i], "--audit") == 0) {
            options->output_mode = YVEX_MODELS_OUTPUT_AUDIT;
        } else if (args[i][0] == '-') {
            yvex_cli_out_writef(stderr, "yvex: unknown models check option: %s\n", args[i]);
            return 2;
        } else {
            yvex_cli_out_writef(stderr, "yvex: models check accepts only one TARGET\n");
            return 2;
        }
    }
    if (strcmp(options->backend_name, "cpu") != 0 &&
        strcmp(options->backend_name, "cuda") != 0) {
        yvex_cli_out_writef(stderr, "yvex: unknown backend kind: %s\n", options->backend_name);
        return 2;
    }
    if (strcmp(options->level_name, "quick") == 0) {
        options->level = YVEX_CLI_MODEL_CHECK_QUICK;
    } else if (strcmp(options->level_name, "runtime") == 0) {
        options->level = YVEX_CLI_MODEL_CHECK_RUNTIME;
    } else if (strcmp(options->level_name, "full") == 0) {
        options->level = YVEX_CLI_MODEL_CHECK_FULL;
    } else {
        yvex_cli_out_writef(stderr, "yvex: unknown models check level: %s\n", options->level_name);
        return 2;
    }
    return 0;
}

static int print_model_check_unsupported(const yvex_cli_models_check_options *options)
{
    const char *target = options && options->target ? options->target : "";
    if (options && options->output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "model-check: unsupported target=%s level=%s\n",
               target,
               options->level_name ? options->level_name : "quick");
        yvex_cli_out_writef(stdout, "backend: %s\n", options->backend_name ? options->backend_name : "cpu");
        yvex_cli_out_writef(stdout, "top_blocker: %s\n",
               strcmp(target, "deepseek4-v4-flash-dspark-selected-embed-rmsnorm") == 0
                   ? "segment check is planned"
                   : "source-only target cannot be checked as a YVEX-produced runtime artifact yet");
        yvex_cli_out_lines(stdout, literal_pair_0, sizeof(literal_pair_0) / sizeof(literal_pair_0[0]));
    } else {
        yvex_cli_out_writef(stdout, "status: model-check-unsupported\n");
        yvex_cli_out_writef(stdout, "target_id: %s\n", target);
        if (strcmp(target, "deepseek4-v4-flash-dspark-selected-embed-rmsnorm") == 0) {
            yvex_cli_out_writef(stdout, "reason: segment check is planned, not implemented by this preset\n");
        } else {
            yvex_cli_out_writef(stdout,
                "reason: source-only target cannot be checked as a YVEX-produced runtime artifact yet\n");
        }
        model_print_runtime_generation("unsupported");
    }
    return exit_for_status(YVEX_ERR_UNSUPPORTED);
}

static int model_check_resolve_registry_path(const yvex_cli_models_check_options *options,
                                             char *registry_path,
                                             size_t registry_path_cap,
                                             yvex_error *err)
{
    if (options->registry_path) {
        return expand_operator_path(options->registry_path, registry_path,
                                    registry_path_cap, err, "models_check");
    }
    return yvex_model_registry_default_path(registry_path,
                                            (unsigned long long)registry_path_cap,
                                            err);
}

static int model_check_resolve_canonical_path(
    const yvex_cli_models_check_options *options,
    yvex_model_ref *ref,
    yvex_cli_model_check_report *report,
    yvex_error *err)
{
    static const char *artifact_name =
        "deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf";
    yvex_paths paths;
    yvex_operator_paths operator_paths;
    char gguf_dir[YVEX_PATH_CAP];
    char artifact_path[YVEX_PATH_CAP];
    int exists = 0;
    int rc;

    memset(&paths, 0, sizeof(paths));
    memset(&operator_paths, 0, sizeof(operator_paths));
    rc = yvex_operator_paths_resolve(&paths, options->models_root, &operator_paths, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_operator_paths_resolve_target(&operator_paths, "deepseek", "gguf",
                                            gguf_dir, sizeof(gguf_dir), &exists, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = path_join2(artifact_path, sizeof(artifact_path), gguf_dir, artifact_name,
                    err, "models_check");
    if (rc != YVEX_OK) {
        return rc;
    }
    if (!path_exists(artifact_path)) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_check",
                        "canonical selected artifact does not exist: %s",
                        artifact_path);
        return YVEX_ERR_IO;
    }
    rc = yvex_model_ref_resolve(ref, artifact_path, NULL, err);
    if (rc == YVEX_OK) {
        snprintf(report->model_input_kind, sizeof(report->model_input_kind), "target");
    }
    return rc;
}

static int model_check_resolve_ref(const yvex_cli_models_check_options *options,
                                   const char *registry_path,
                                   yvex_model_ref *ref,
                                   yvex_cli_model_check_report *report,
                                   yvex_error *err)
{
    yvex_model_ref_options ref_options;
    int rc;

    if (options->models_root &&
        strcmp(options->target, "deepseek4-v4-flash-dspark-selected-embed") == 0 &&
        !is_path_like_reference(options->target)) {
        return model_check_resolve_canonical_path(options, ref, report, err);
    }

    memset(&ref_options, 0, sizeof(ref_options));
    ref_options.registry_path = registry_path;
    ref_options.allow_registry = 1;
    rc = yvex_model_ref_resolve(ref, options->target, &ref_options, err);
    if (rc == YVEX_OK) {
        if (is_path_like_reference(options->target)) {
            snprintf(report->model_input_kind, sizeof(report->model_input_kind), "path");
        } else if (strcmp(options->target, "deepseek4-v4-flash-dspark-selected-embed") == 0) {
            snprintf(report->model_input_kind, sizeof(report->model_input_kind), "target");
        } else {
            snprintf(report->model_input_kind, sizeof(report->model_input_kind), "alias");
        }
        return YVEX_OK;
    }
    if (strcmp(options->target, "deepseek4-v4-flash-dspark-selected-embed") == 0) {
        yvex_error_clear(err);
        yvex_model_ref_clear(ref);
        return model_check_resolve_canonical_path(options, ref, report, err);
    }
    return rc;
}

static int model_check_verify_registry_identity(const yvex_model_ref *ref,
                                                yvex_error *err)
{
    yvex_model_registry_entry registered_metadata;

    if (!ref || ref->kind != YVEX_MODEL_REF_ALIAS) return YVEX_OK;
    memset(&registered_metadata, 0, sizeof(registered_metadata));
    yvex_model_ref_registry_entry_view(ref, &registered_metadata);
    return registry_snapshot_verify(&registered_metadata, ref->path, "models_check",
                                    "registry identity drift", "registry identity drift", err);
}

static int model_check_integrity(const yvex_model_ref *ref,
                                 int require_embedding,
                                 yvex_artifact_integrity_report *report,
                                 yvex_error *err)
{
    yvex_artifact_integrity_options integrity_options;

    memset(&integrity_options, 0, sizeof(integrity_options));
    integrity_options.require_token_embedding = require_embedding;
    integrity_options.token_id = 0u;
    if (ref->kind == YVEX_MODEL_REF_ALIAS && ref->sha256 && ref->sha256[0]) {
        integrity_options.registered_sha256 = ref->sha256;
    }
    return yvex_artifact_integrity_check_path(ref->path, &integrity_options,
                                             report, err);
}

static int model_check_backend_probe(const char *backend_name, yvex_error *err)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    int rc;

    memset(&options, 0, sizeof(options));
    (void)model_backend_kind_from_name(backend_name, &options.kind);
    rc = yvex_backend_open(&backend, &options, err);
    if (rc == YVEX_OK) {
        yvex_backend_close(backend);
    }
    return rc;
}

static int model_check_materialize(const char *path,
                                   const char *backend_name,
                                   yvex_error *err)
{
    yvex_model_context ctx;
    yvex_backend *backend = NULL;
    yvex_weight_table *weights = NULL;
    yvex_backend_options backend_options;
    yvex_materialize_options materialize_options;
    yvex_materialize_summary summary;
    int rc;

    memset(&ctx, 0, sizeof(ctx));
    memset(&backend_options, 0, sizeof(backend_options));
    memset(&materialize_options, 0, sizeof(materialize_options));
    memset(&summary, 0, sizeof(summary));
    rc = yvex_model_context_open(path, &ctx, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    (void)model_backend_kind_from_name(backend_name, &backend_options.kind);
    rc = yvex_backend_open(&backend, &backend_options, err);
    if (rc != YVEX_OK) {
        yvex_model_context_close(&ctx);
        return rc;
    }
    materialize_options.backend_name = backend_name;
    materialize_options.require_all_tensors = 1;
    rc = yvex_weight_table_materialize(&weights,
                                       ctx.artifact,
                                       ctx.gguf,
                                       ctx.table,
                                       backend,
                                       &materialize_options,
                                       err);
    if (rc == YVEX_OK) {
        rc = yvex_weight_table_get_summary(weights, &summary, err);
    }
    if (rc == YVEX_OK && summary.status != YVEX_WEIGHT_STATUS_MATERIALIZED) {
        yvex_error_set(err, YVEX_ERR_STATE, "models_check",
                       "materialization did not reach weights-materialized");
        rc = YVEX_ERR_STATE;
    }
    yvex_weight_table_close(weights);
    yvex_backend_close(backend);
    yvex_model_context_close(&ctx);
    return rc;
}

static int model_check_plan(const char *path,
                            const char *backend_name,
                            yvex_error *err)
{
    yvex_model_context ctx;
    yvex_plan *plan = NULL;
    yvex_plan_options options;
    int rc;

    memset(&ctx, 0, sizeof(ctx));
    memset(&options, 0, sizeof(options));
    options.sequence_length = 1ull;
    options.context_length = 16ull;
    options.backend_name = backend_name;
    rc = yvex_model_context_open(path, &ctx, err);
    if (rc == YVEX_OK) {
        rc = yvex_plan_create(&plan, ctx.model, ctx.table, &options, err);
    }
    yvex_plan_close(plan);
    yvex_model_context_close(&ctx);
    return rc;
}

static int model_check_is_real_selected_embedding(
    const yvex_model_metadata_snapshot *metadata)
{
    return metadata &&
           strcmp(metadata->entry.primary_tensor_name, "token_embd.weight") == 0 &&
           strcmp(metadata->entry.primary_tensor_dtype, "F16") == 0 &&
           metadata->entry.primary_tensor_rank == 2u &&
           strcmp(metadata->entry.primary_tensor_dims, "[4096,129280]") == 0 &&
           metadata->entry.primary_tensor_bytes == 1059061760ull;
}

static int model_check_run_gate(const yvex_model_ref *ref,
                                const char *backend_name,
                                int materialize,
                                yvex_error *err)
{
    static const yvex_model_gate_expected_tensor model_expected = {
        .name = "token_embd.weight", .dtype = "F16", .rank = 2u,
        .dims = {4096ull, 129280ull}, .bytes = 1059061760ull};
    static const yvex_materialize_expected_tensor materialize_expected = {
        .name = "token_embd.weight", .dtype = "F16", .rank = 2u,
        .dims = {4096ull, 129280ull}, .bytes = 1059061760ull};
    int cpu = strcmp(backend_name, "cpu") == 0;
    int cuda = strcmp(backend_name, "cuda") == 0;
    int rc;

    if (materialize) {
        yvex_materialize_gate_options options = {0};
        yvex_materialize_gate_summary summary = {0};
        options.model_path = ref->path;
        options.label = "deepseek-v4-flash-dspark-selected-embedding";
        options.family = "deepseek4";
        options.sha256 = ref->kind == YVEX_MODEL_REF_ALIAS ? ref->sha256 : NULL;
        options.metadata_status = "pass";
        options.scope = YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR;
        options.expected_tensors = &materialize_expected;
        options.expected_tensor_count = 1ull;
        options.check_cpu = options.require_cpu = cpu;
        options.check_cuda = options.require_cuda = cuda;
        options.repeat_count = options.check_cleanup = 1u;
        rc = yvex_materialize_gate_check(&options, &summary, err);
        if (rc == YVEX_OK && summary.status != YVEX_MATERIALIZE_GATE_PASS) {
            yvex_error_set(err, YVEX_ERR_STATE, "models_check", "materialize gate did not pass");
            rc = YVEX_ERR_STATE;
        }
    } else {
        yvex_model_gate_options options = {0};
        yvex_model_gate_summary summary = {0};
        options.model_path = ref->path;
        options.model_label = "deepseek-v4-flash-dspark-selected-embedding";
        options.family = "deepseek4";
        options.artifact_sha256 = ref->kind == YVEX_MODEL_REF_ALIAS ? ref->sha256 : NULL;
        options.expected_tensors = &model_expected;
        options.expected_tensor_count = 1ull;
        options.check_cpu = options.require_cpu = cpu;
        options.check_cuda = options.require_cuda = cuda;
        rc = yvex_model_gate_check(&options, &summary, err);
        if (rc == YVEX_OK && summary.status != YVEX_MODEL_GATE_PASS) {
            yvex_error_set(err, YVEX_ERR_STATE, "models_check", "model gate did not pass");
            rc = YVEX_ERR_STATE;
        }
    }
    return rc;
}

static int model_check_finish(model_check_run *run, int exit_code)
{
    int rc;

    rc = model_check_write_report(&run->options, &run->report, &run->error);
    if (rc != YVEX_OK && exit_code == 0) {
        snprintf(run->report.error, sizeof(run->report.error), "%s", yvex_error_message(&run->error));
        run->report.final_status = "model-check-fail";
        exit_code = exit_for_status(rc);
    }
    yvex_model_context_close(&run->context);
    yvex_model_ref_clear(&run->ref);
    if (run->options.output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
        model_check_report_print_normal(stdout, &run->report);
    } else {
        model_check_report_print(stdout, &run->report);
    }
    return exit_code;
}

static int model_check_fail(model_check_run *run, const char **stage, int rc, const char *message)
{
    if (stage) *stage = "fail";
    snprintf(run->report.error, sizeof(run->report.error), "%s",
             message ? message : yvex_error_message(&run->error));
    return model_check_finish(run, rc == YVEX_ERR_UNSUPPORTED ? 5 : exit_for_status(rc));
}

static int model_check_runtime_pipeline(model_check_run *run)
{
    yvex_cli_models_check_options *options = &run->options;
    yvex_cli_model_check_report *report = &run->report;
    int rc;

    rc = model_check_backend_probe(options->backend_name, &run->error);
    if (rc != YVEX_OK) return model_check_fail(run, &report->stage_integrity_report, rc, NULL);
    report->stage_integrity_report = "pass";
    if (options->no_materialize) {
        report->stage_materialize = "skipped";
        report->stage_engine = "skipped";
        report->stage_session = "skipped";
        report->stage_plan = "skipped";
        report->stage_graph_partial = "skipped";
        snprintf(report->graph_skip_reason, sizeof(report->graph_skip_reason),
                 "disabled by --no-materialize");
        report->stage_model_gate = "skipped";
        report->stage_materialize_gate = "skipped";
        report->runtime_execution = "not-performed";
        report->final_status = "model-check-pass";
        return model_check_finish(run, 0);
    }
    rc = model_check_materialize(run->ref.path, options->backend_name, &run->error);
    if (rc != YVEX_OK) return model_check_fail(run, &report->stage_materialize, rc, NULL);
    report->stage_materialize = "pass";
    rc = model_check_plan(run->ref.path, options->backend_name, &run->error);
    if (rc != YVEX_OK) return model_check_fail(run, &report->stage_plan, rc, NULL);
    report->stage_plan = "pass";
    report->stage_graph_partial = "skipped";
    snprintf(report->graph_skip_reason, sizeof(report->graph_skip_reason),
             "%s", options->no_graph ? "disabled by --no-graph"
                                      : "legacy selected-graph diagnostic retired");
    report->runtime_execution = "not-performed";
    if (options->level == YVEX_CLI_MODEL_CHECK_FULL &&
        model_check_is_real_selected_embedding(&run->metadata)) {
        rc = model_check_run_gate(&run->ref, options->backend_name, 0, &run->error);
        if (rc != YVEX_OK) return model_check_fail(run, &report->stage_model_gate, rc, NULL);
        report->stage_model_gate = "pass";
        rc = model_check_run_gate(&run->ref, options->backend_name, 1, &run->error);
        if (rc != YVEX_OK) return model_check_fail(run, &report->stage_materialize_gate, rc, NULL);
        report->stage_materialize_gate = "pass";
    } else if (options->level == YVEX_CLI_MODEL_CHECK_FULL) {
        report->stage_model_gate = "skipped";
        report->stage_materialize_gate = "skipped";
    }
    report->final_status = "model-check-pass";
    return model_check_finish(run, 0);
}

int yvex_models_check_surface_command(int arg_count, char **args)
{
    model_check_run run = {0};
    int rc;

    yvex_error_clear(&run.error);
    rc = parse_models_check_options(arg_count, args, &run.options);
    if (rc != 0) return rc;
    if (strcmp(run.options.target, "deepseek4-v4-flash-dspark-selected-embed-rmsnorm") == 0 ||
        strcmp(run.options.target, "glm-5.2-official-safetensors") == 0) {
        return print_model_check_unsupported(&run.options);
    }
    if (strcmp(run.options.target, "deepseek4-v4-flash-dspark-selected-embed") != 0 &&
        !is_path_like_reference(run.options.target)) {
        yvex_model_ref_options ref_options = {0};
        ref_options.allow_registry = 1;
        ref_options.registry_path = run.options.registry_path;
        rc = yvex_model_ref_resolve(&run.ref, run.options.target, &ref_options, &run.error);
        if (rc != YVEX_OK) return print_yvex_error(&run.error, exit_for_status(rc));
        yvex_model_ref_clear(&run.ref);
    }

    model_check_report_init(&run.report, &run.options);
    rc = model_check_resolve_registry_path(&run.options, run.registry_path,
                                           sizeof(run.registry_path), &run.error);
    if (rc != YVEX_OK) {
        return model_check_fail(&run, NULL, rc, NULL);
    }
    snprintf(run.report.registry_path, sizeof(run.report.registry_path), "%s", run.registry_path);

    rc = model_check_resolve_ref(&run.options, run.registry_path, &run.ref, &run.report, &run.error);
    if (rc != YVEX_OK) return model_check_fail(&run, &run.report.stage_resolve_target, rc, NULL);
    run.report.stage_resolve_target = "pass";
    snprintf(run.report.artifact_path, sizeof(run.report.artifact_path), "%s", run.ref.path);
    if (!path_exists(run.ref.path)) {
        return model_check_fail(&run, &run.report.stage_resolve_artifact, YVEX_ERR_IO,
                                "artifact path does not exist");
    }
    run.report.stage_resolve_artifact = "pass";

    rc = yvex_model_context_open(run.ref.path, &run.context, &run.error);
    if (rc != YVEX_OK) return model_check_fail(&run, &run.report.stage_inspect, rc, NULL);
    run.report.stage_inspect = run.report.stage_tensors = "pass";

    rc = yvex_model_metadata_snapshot_read(&run.metadata, run.ref.path, &run.error);
    if (rc != YVEX_OK) return model_check_fail(&run, &run.report.stage_metadata, rc, NULL);
    run.report.stage_metadata = "pass";

    if (run.ref.kind == YVEX_MODEL_REF_ALIAS) {
        rc = model_check_verify_registry_identity(&run.ref, &run.error);
        if (rc != YVEX_OK) {
            return model_check_fail(&run, &run.report.stage_registry_identity,
                                    YVEX_ERR_STATE, "registry identity drift");
        }
        run.report.stage_registry_identity = "pass";
    } else {
        run.report.stage_registry_identity = "unregistered";
    }

    rc = model_check_integrity(&run.ref, 0, &run.integrity, &run.error);
    if (rc != YVEX_OK || !run.integrity.passed) {
        return model_check_fail(&run, &run.report.stage_integrity_check,
                                rc != YVEX_OK ? rc : YVEX_ERR_FORMAT,
                                rc == YVEX_OK ? "artifact integrity failed" : NULL);
    }
    run.report.stage_integrity_check = "pass";

    if (run.options.level == YVEX_CLI_MODEL_CHECK_QUICK) {
        if (run.options.no_graph) {
            snprintf(run.report.graph_skip_reason, sizeof(run.report.graph_skip_reason),
                     "quick level does not run graph");
        }
        run.report.final_status = "model-check-pass";
        return model_check_finish(&run, 0);
    }
    return model_check_runtime_pipeline(&run);
}
