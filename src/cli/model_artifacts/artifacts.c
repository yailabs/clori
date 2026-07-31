/*
 * Scan registry and artifact metadata for finite CLI reports. Discovery never performs artifact
 * admission or opens a runtime model.
 */
#include "src/cli/model_artifacts/private.h"

#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *const literal_lines_0[] = { "source_payload_loaded: false",
    "hash_performed: false",
    "runtime_ready: false",
    "generation: unsupported",
    "benchmark_status: not-measured"};

static const char *const literal_lines_1[] = { "source_payload_loaded: false",
    "hash_performed: false",
    "runtime_ready: false",
    "generation: unsupported",
    "benchmark_status: not-measured"};

typedef enum {
    YVEX_ARTIFACTS_OUTPUT_NORMAL = 0,
    YVEX_ARTIFACTS_OUTPUT_TABLE,
    YVEX_ARTIFACTS_OUTPUT_AUDIT,
    YVEX_ARTIFACTS_OUTPUT_JSON
} yvex_artifacts_output_mode;

typedef struct {
    const char *action;
    const char *target;
    const char *models_root;
    const char *family;
    yvex_artifacts_output_mode output_mode;
} yvex_cli_models_artifacts_options;

typedef struct {
    char target_id[128];
    char family[32];
    char artifact_class[64];
    char artifact_status[32];
    char source_status[32];
    char prepare_status[32];
    char top_blocker[64];
    char detail[256];
    char path[YVEX_PATH_CAP];
    char expected_path[YVEX_PATH_CAP];
    char display_path[YVEX_PATH_CAP];
    char registry_path[YVEX_PATH_CAP];
    char download_report_path[YVEX_PATH_CAP];
    char source_manifest_path[YVEX_PATH_CAP];
    char native_inventory_path[YVEX_PATH_CAP];
    char tensor_map_path[YVEX_PATH_CAP];
    char output_head_map_path[YVEX_PATH_CAP];
    char tokenizer_map_path[YVEX_PATH_CAP];
    unsigned long long size_bytes;
    int source_present;
    int tensor_map_present;
    int tensor_map_incomplete;
    int output_head_map_present;
    int output_head_missing;
    int tokenizer_map_present;
    int dynamic_source;
} yvex_models_artifact_row;

typedef struct {
    yvex_models_artifact_row rows[YVEX_MODELS_ARTIFACT_ROWS_CAP];
    unsigned int count;
} yvex_models_artifact_rows;

static int artifacts_parse_output_mode(const char *value,
                                       yvex_artifacts_output_mode *mode)
{
    if (!value || !mode) return 0;
    if (strcmp(value, "normal") == 0) {
        *mode = YVEX_ARTIFACTS_OUTPUT_NORMAL;
        return 1;
    }
    if (strcmp(value, "table") == 0) {
        *mode = YVEX_ARTIFACTS_OUTPUT_TABLE;
        return 1;
    }
    if (strcmp(value, "audit") == 0) {
        *mode = YVEX_ARTIFACTS_OUTPUT_AUDIT;
        return 1;
    }
    if (strcmp(value, "json") == 0) {
        *mode = YVEX_ARTIFACTS_OUTPUT_JSON;
        return 1;
    }
    return 0;
}

static int parse_models_artifacts_options(int arg_count,
                                          char **args,
                                          yvex_cli_models_artifacts_options *options)
{
    int i;

    memset(options, 0, sizeof(*options));
    options->output_mode = YVEX_ARTIFACTS_OUTPUT_NORMAL;
    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "yvex: models artifacts requires list or status\n");
        return 2;
    }
    options->action = args[3];
    if (strcmp(options->action, "list") != 0 &&
        strcmp(options->action, "status") != 0) {
        yvex_cli_out_writef(stderr, "yvex: unknown models artifacts action: %s\n", options->action);
        return 2;
    }
    i = 4;
    if (strcmp(options->action, "status") == 0) {
        if (i >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: models artifacts status requires TARGET\n");
            return 2;
        }
        options->target = args[i++];
        if (!cli_arg_value_valid(options->target)) {
            yvex_cli_out_writef(stderr, "yvex: models artifacts status target is empty or invalid\n");
            return 2;
        }
    }
    for (; i < arg_count; ++i) {
        if (strcmp(args[i], "--audit") == 0) {
            options->output_mode = YVEX_ARTIFACTS_OUTPUT_AUDIT;
        } else if (strcmp(args[i], "--models-root") == 0 ||
                   strcmp(args[i], "--family") == 0 ||
                   strcmp(args[i], "--output") == 0) {
            const char *flag = args[i];
            const char *value = NULL;
            int rc = parse_models_value_option("models artifacts", flag,
                                               arg_count, args, &i, &value);
            if (rc != 0) return rc;
            if (strcmp(flag, "--models-root") == 0) {
                options->models_root = value;
            } else if (strcmp(flag, "--family") == 0) {
                if (strcmp(value, "deepseek") != 0 &&
                    strcmp(value, "glm") != 0 &&
                    strcmp(value, "qwen") != 0 &&
                    strcmp(value, "gemma") != 0) {
                    yvex_cli_out_writef(stderr, "yvex: models artifacts --family requires deepseek|glm|qwen|gemma\n");
                    return 2;
                }
                options->family = value;
            } else if (!artifacts_parse_output_mode(value, &options->output_mode)) {
                yvex_cli_out_writef(stderr, "yvex: models artifacts unsupported output mode: %s\n", value);
                return 2;
            }
        } else if (strcmp(args[i], "--json") == 0) {
            options->output_mode = YVEX_ARTIFACTS_OUTPUT_JSON;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown models artifacts option: %s\n", args[i]);
            return 2;
        }
    }
    return 0;
}

static int artifacts_family_allowed(const yvex_cli_models_artifacts_options *options,
                                    const char *family)
{
    return !options || !options->family || strcmp(options->family, family) == 0;
}

static int artifacts_rows_find(yvex_models_artifact_rows *rows,
                               const char *target,
                               const char *family)
{
    unsigned int i;

    if (!rows || !target || !family) return -1;
    for (i = 0; i < rows->count; ++i) {
        if (strcmp(rows->rows[i].target_id, target) == 0 &&
            strcmp(rows->rows[i].family, family) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static yvex_models_artifact_row *artifacts_rows_append(
    yvex_models_artifact_rows *rows,
    const char *target,
    const char *family)
{
    yvex_models_artifact_row *row;

    if (!rows || !target || !family || !target[0] || !family[0]) return NULL;
    if (artifacts_rows_find(rows, target, family) >= 0) {
        return &rows->rows[artifacts_rows_find(rows, target, family)];
    }
    if (rows->count >= YVEX_MODELS_ARTIFACT_ROWS_CAP) return NULL;
    row = &rows->rows[rows->count++];
    memset(row, 0, sizeof(*row));
    snprintf(row->target_id, sizeof(row->target_id), "%s", target);
    snprintf(row->family, sizeof(row->family), "%s", family);
    snprintf(row->artifact_status, sizeof(row->artifact_status), "missing");
    snprintf(row->source_status, sizeof(row->source_status), "not-applicable");
    snprintf(row->prepare_status, sizeof(row->prepare_status), "unknown");
    snprintf(row->top_blocker, sizeof(row->top_blocker), "none");
    snprintf(row->detail, sizeof(row->detail), "artifact discovery only");
    return row;
}

static void artifacts_relative_path(const yvex_operator_paths *operator_paths,
                                    const char *path,
                                    char *out,
                                    size_t out_cap)
{
    size_t root_len;

    if (!out || out_cap == 0u) return;
    out[0] = '\0';
    if (!path || !path[0]) {
        snprintf(out, out_cap, "-");
        return;
    }
    if (operator_paths && operator_paths->models_root[0]) {
        root_len = strlen(operator_paths->models_root);
        if (strncmp(path, operator_paths->models_root, root_len) == 0 &&
            path[root_len] == '/') {
            snprintf(out, out_cap, "%s", path + root_len + 1u);
            return;
        }
    }
    snprintf(out, out_cap, "%s", path);
}

static void artifacts_strip_suffix(char *text, const char *suffix)
{
    if (!yvex_source_ends_with(text, suffix))
        return;
    text[strlen(text) - strlen(suffix)] = '\0';
}

static void artifacts_target_from_gguf_name(const char *file_name,
                                            char *out,
                                            size_t out_cap)
{
    if (!out || out_cap == 0u) return;
    out[0] = '\0';
    if (!file_name || !file_name[0]) return;
    snprintf(out, out_cap, "%s", file_name);
    artifacts_strip_suffix(out, ".gguf");
    artifacts_strip_suffix(out, "-F16-noimatrix-yvex-v1");
}

static const char *artifacts_class_from_name(const char *file_name)
{
    if (file_name && strstr(file_name, "controlled")) return "yvex-controlled-gguf";
    if (file_name && strstr(file_name, "selected")) return "yvex-selected-gguf";
    return "unknown-gguf";
}

static int artifacts_stat_file(const char *path, unsigned long long *size_out)
{
    struct stat st;

    if (size_out) *size_out = 0ull;
    if (!path || stat(path, &st) != 0 || !S_ISREG(st.st_mode)) return 0;
    if (size_out) *size_out = (unsigned long long)st.st_size;
    return 1;
}

static void artifacts_classify_dynamic_row(const yvex_operator_paths *operator_paths,
                                           yvex_models_artifact_row *row)
{
    int n;

    if (!operator_paths || !row) return;
    n = snprintf(row->expected_path, sizeof(row->expected_path),
                 "%s/%s/%s.gguf",
                 operator_paths->gguf_root,
                 row->family,
                 row->target_id);
    if (n < 0 || (size_t)n >= sizeof(row->expected_path)) row->expected_path[0] = '\0';
    if (row->expected_path[0] &&
        artifacts_stat_file(row->expected_path, &row->size_bytes)) {
        snprintf(row->path, sizeof(row->path), "%s", row->expected_path);
        snprintf(row->artifact_status, sizeof(row->artifact_status), "present");
    } else {
        snprintf(row->artifact_status, sizeof(row->artifact_status), "missing");
        row->size_bytes = 0ull;
    }
    artifacts_relative_path(operator_paths,
                            row->path[0] ? row->path : row->expected_path,
                            row->display_path,
                            sizeof(row->display_path));
    snprintf(row->artifact_class, sizeof(row->artifact_class), "planned-full-gguf");
    row->source_present = row->source_status[0] &&
                          strcmp(row->source_status, "present") == 0;
    row->tensor_map_present = row->tensor_map_path[0] && path_exists(row->tensor_map_path);
    row->output_head_map_present =
        row->output_head_map_path[0] && path_exists(row->output_head_map_path);
    row->tokenizer_map_present =
        row->tokenizer_map_path[0] && path_exists(row->tokenizer_map_path);
    prepare_probe_map_sidecar_status(row->tensor_map_path,
                                     row->output_head_map_path,
                                     &row->tensor_map_incomplete,
                                     &row->output_head_missing);
    snprintf(row->prepare_status, sizeof(row->prepare_status), "blocked");
    if (!row->source_present) {
        snprintf(row->top_blocker, sizeof(row->top_blocker), "missing-source");
        snprintf(row->detail, sizeof(row->detail), "downloaded source path is missing");
    } else if (!row->output_head_map_present || row->output_head_missing) {
        snprintf(row->top_blocker, sizeof(row->top_blocker), "missing-output-head-map");
        snprintf(row->detail, sizeof(row->detail),
            "output-head/tokenizer mapping missing; full GGUF emission not performed");
    } else if (!row->tensor_map_present || row->tensor_map_incomplete) {
        snprintf(row->top_blocker, sizeof(row->top_blocker), "incomplete-tensor-map");
        snprintf(row->detail, sizeof(row->detail), "tensor map incomplete; full GGUF emission not performed");
    } else if (!row->tokenizer_map_present) {
        snprintf(row->top_blocker, sizeof(row->top_blocker), "missing-tokenizer-map");
        snprintf(row->detail, sizeof(row->detail),
            "tokenizer metadata mapping missing; full GGUF emission not performed");
    } else if (strcmp(row->artifact_status, "missing") == 0) {
        snprintf(row->top_blocker, sizeof(row->top_blocker), "%s",
                 strcmp(row->family, "deepseek") == 0
                     ? "complete-artifact-admission-required"
                     : "family-quantization-plan-unimplemented");
        snprintf(row->detail, sizeof(row->detail), "%s",
                 strcmp(row->family, "deepseek") == 0
                     ? "artifact discovery did not bind canonical complete-artifact admission"
                     : "this engineering family has no complete quantization plan");
    } else {
        snprintf(row->top_blocker, sizeof(row->top_blocker), "missing-artifact-identity");
        snprintf(row->detail, sizeof(row->detail),
            "artifact exists but identity/admission is not checked by this discovery command");
    }
}

static const char *artifacts_row_next(const yvex_models_artifact_row *row)
{
    if (!row || strcmp(row->prepare_status, "blocked") != 0) return "none";
    if (strcmp(row->top_blocker, "missing-tokenizer-map") == 0) {
        return "V010.MAP.7";
    }
    if (strcmp(row->top_blocker,
               "complete-artifact-admission-required") == 0) {
        return "V010.ARTIFACT.MATERIALIZE.0";
    }
    if (strcmp(row->top_blocker,
               "family-quantization-plan-unimplemented") == 0) {
        return "not-scheduled";
    }
    return "V010.MAP.8";
}

static void artifacts_add_dynamic_target(
    const yvex_operator_paths *operator_paths,
    yvex_models_artifact_rows *rows,
    const char *target,
    const char *family,
    const yvex_model_download_resolved_target *resolved)
{
    yvex_models_artifact_row *row;
    char family_dir[YVEX_PATH_CAP];

    row = artifacts_rows_append(rows, target, family);
    if (!row || !operator_paths) return;
    row->dynamic_source = 1;
    snprintf(row->artifact_class, sizeof(row->artifact_class), "planned-full-gguf");
    if (resolved) {
        snprintf(row->registry_path, sizeof(row->registry_path), "%s", resolved->registry_path);
        snprintf(row->download_report_path, sizeof(row->download_report_path), "%s", resolved->download_report_path);
        snprintf(row->source_manifest_path, sizeof(row->source_manifest_path), "%s", resolved->manifest_path);
        snprintf(row->native_inventory_path, sizeof(row->native_inventory_path), "%s", resolved->native_inventory_path);
        if (resolved->local_source_dir[0] && path_exists(resolved->local_source_dir)) {
            snprintf(row->source_status, sizeof(row->source_status), "present");
        } else {
            snprintf(row->source_status, sizeof(row->source_status), "missing");
        }
    }
    if (!row->source_status[0] || strcmp(row->source_status, "not-applicable") == 0) {
        snprintf(row->source_status, sizeof(row->source_status), "missing");
    }
    if (path_join2(family_dir, sizeof(family_dir), operator_paths->reports_root,
                   family, NULL, "models_artifacts") == YVEX_OK) {
        char file_name[256];
        snprintf(file_name, sizeof(file_name), "%s.tensor-map.json", target);
        (void)path_join2(row->tensor_map_path, sizeof(row->tensor_map_path),
                         family_dir, file_name, NULL, "models_artifacts");
        snprintf(file_name, sizeof(file_name), "%s.output-head-map.json", target);
        (void)path_join2(row->output_head_map_path, sizeof(row->output_head_map_path),
                         family_dir, file_name, NULL, "models_artifacts");
        snprintf(file_name, sizeof(file_name), "%s.tokenizer-map.json", target);
        (void)path_join2(row->tokenizer_map_path, sizeof(row->tokenizer_map_path),
                         family_dir, file_name, NULL, "models_artifacts");
    }
    artifacts_classify_dynamic_row(operator_paths, row);
}

static void artifacts_scan_gguf_family(const yvex_operator_paths *operator_paths,
                                       yvex_models_artifact_rows *rows,
                                       const char *family)
{
    DIR *dir;
    struct dirent *ent;
    char family_dir[YVEX_PATH_CAP];
    yvex_error err;

    yvex_error_clear(&err);
    if (!operator_paths || !rows || !family) return;
    if (path_join2(family_dir, sizeof(family_dir), operator_paths->gguf_root,
                   family, &err, "models_artifacts") != YVEX_OK) {
        return;
    }
    dir = opendir(family_dir);
    if (!dir) return;
    while ((ent = readdir(dir)) != NULL) {
        char path[YVEX_PATH_CAP];
        char target[128];
        yvex_models_artifact_row *row;

        if (ent->d_name[0] == '.') continue;
        if (!yvex_source_ends_with(ent->d_name, ".gguf")) continue;
        if (path_join2(path, sizeof(path), family_dir, ent->d_name, &err,
                       "models_artifacts") != YVEX_OK) {
            continue;
        }
        if (!artifacts_stat_file(path, NULL)) continue;
        artifacts_target_from_gguf_name(ent->d_name, target, sizeof(target));
        row = artifacts_rows_append(rows, target, family);
        if (!row) continue;
        snprintf(row->artifact_class, sizeof(row->artifact_class), "%s",
                 artifacts_class_from_name(ent->d_name));
        snprintf(row->artifact_status, sizeof(row->artifact_status), "present");
        snprintf(row->source_status, sizeof(row->source_status), "not-applicable");
        snprintf(row->prepare_status, sizeof(row->prepare_status), "ready");
        snprintf(row->top_blocker, sizeof(row->top_blocker), "none");
        snprintf(row->detail, sizeof(row->detail), "GGUF artifact present");
        snprintf(row->path, sizeof(row->path), "%s", path);
        snprintf(row->expected_path, sizeof(row->expected_path), "%s", path);
        artifacts_stat_file(path, &row->size_bytes);
        artifacts_relative_path(operator_paths, path, row->display_path,
                                sizeof(row->display_path));
    }
    closedir(dir);
}

static void artifacts_scan_dynamic_sidecar_dir(
    const yvex_operator_paths *operator_paths,
    yvex_models_artifact_rows *rows,
    const char *family,
    const char *dir_path,
    const char *suffix)
{
    DIR *dir;
    struct dirent *ent;
    yvex_error err;

    if (!operator_paths || !rows || !family || !dir_path || !suffix) return;
    dir = opendir(dir_path);
    if (!dir) return;
    yvex_error_clear(&err);
    while ((ent = readdir(dir)) != NULL) {
        char target[128];
        char path[YVEX_PATH_CAP];
        yvex_model_download_resolved_target resolved;
        size_t name_len;

        if (ent->d_name[0] == '.') continue;
        if (!yvex_source_ends_with(ent->d_name, suffix)) continue;
        name_len = strlen(ent->d_name);
        if (name_len >= sizeof(target)) continue;
        memcpy(target, ent->d_name, name_len + 1u);
        artifacts_strip_suffix(target, suffix);
        if (!model_download_identity_paths(target, family, operator_paths,
                                           &resolved, &err)) {
            yvex_error_clear(&err);
            continue;
        }
        if (path_join2(path, sizeof(path), dir_path, ent->d_name, &err,
                       "models_artifacts") != YVEX_OK) {
            yvex_error_clear(&err);
            continue;
        }
        if (model_download_read_identity_file(path, target, family, &resolved) ||
            model_download_resolve_downloaded_target(target, operator_paths,
                                                     &resolved, &err)) {
            artifacts_add_dynamic_target(operator_paths, rows,
                                         resolved.target_id[0] ? resolved.target_id : target,
                                         resolved.family[0] ? resolved.family : family,
                                         &resolved);
        }
        yvex_error_clear(&err);
    }
    closedir(dir);
}

static int artifacts_collect(const yvex_cli_models_artifacts_options *options,
                             const yvex_operator_paths *operator_paths,
                             yvex_models_artifact_rows *rows,
                             yvex_error *err)
{
    static const char *families[] = { "deepseek", "qwen", "gemma", "glm" };
    unsigned long i;

    if (!options || !operator_paths || !rows) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "models_artifacts",
                       "options, operator paths, and rows are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(rows, 0, sizeof(*rows));
    for (i = 0; i < sizeof(families) / sizeof(families[0]); ++i) {
        char registry_family_dir[YVEX_PATH_CAP];
        char reports_family_dir[YVEX_PATH_CAP];
        const char *family = families[i];

        if (!artifacts_family_allowed(options, family)) continue;
        artifacts_scan_gguf_family(operator_paths, rows, family);
        if (path_join2(registry_family_dir, sizeof(registry_family_dir),
                       operator_paths->registry_root, family, err,
                       "models_artifacts") == YVEX_OK) {
            artifacts_scan_dynamic_sidecar_dir(operator_paths, rows, family,
                                               registry_family_dir,
                                               ".download.json");
        }
        yvex_error_clear(err);
        if (path_join2(reports_family_dir, sizeof(reports_family_dir),
                       operator_paths->reports_root, family, err,
                       "models_artifacts") == YVEX_OK) {
            artifacts_scan_dynamic_sidecar_dir(operator_paths, rows, family,
                                               reports_family_dir,
                                               ".download-report.json");
            artifacts_scan_dynamic_sidecar_dir(operator_paths, rows, family,
                                               reports_family_dir,
                                               ".source-manifest.json");
        }
        yvex_error_clear(err);
    }
    return YVEX_OK;
}

static void artifacts_print_table_header(void)
{
    yvex_cli_out_writef(stdout, "artifacts\n\n");
    yvex_cli_out_writef(stdout, "%-42s  %-9s  %-18s  %-8s  %-8s  %-8s  %s\n",
           "TARGET", "FAMILY", "CLASS", "STATUS", "PREPARE", "SIZE", "PATH");
}

static void artifacts_print_row_table(const yvex_models_artifact_row *row)
{
    char size_text[32];

    if (!row) return;
    if (row->size_bytes > 0ull) {
        model_download_format_bytes(size_text, sizeof(size_text), row->size_bytes);
    } else {
        snprintf(size_text, sizeof(size_text), "-");
    }
    yvex_cli_out_writef(stdout, "%-42s  %-9s  %-18s  %-8s  %-8s  %-8s  %s\n",
           row->target_id,
           row->family,
           row->artifact_class[0] ? row->artifact_class : "unknown-gguf",
           row->artifact_status,
           row->prepare_status,
           size_text,
           row->display_path[0] ? row->display_path : "-");
}

static void artifacts_json_string(const char *value)
{
    const unsigned char *p = (const unsigned char *)(value ? value : "");

    yvex_cli_out_char(stdout, '"');
    while (*p) {
        if (*p == '"' || *p == '\\') {
            yvex_cli_out_char(stdout, '\\');
            yvex_cli_out_char(stdout, (int)*p);
        } else if (*p == '\n') {
            yvex_cli_out_fputs("\\n", stdout);
        } else if (*p == '\r') {
            yvex_cli_out_fputs("\\r", stdout);
        } else {
            yvex_cli_out_char(stdout, (int)*p);
        }
        p++;
    }
    yvex_cli_out_char(stdout, '"');
}

static void artifacts_print_list(const yvex_cli_models_artifacts_options *options,
                                 const yvex_operator_paths *operator_paths,
                                 const yvex_models_artifact_rows *rows)
{
    unsigned int i;

    if (options->output_mode == YVEX_ARTIFACTS_OUTPUT_JSON) {
        yvex_cli_out_writef(stdout, "{\"status\":\"artifacts-list\",\"artifacts\":[");
        for (i = 0; i < rows->count; ++i) {
            const yvex_models_artifact_row *row = &rows->rows[i];
            if (i) yvex_cli_out_char(stdout, ',');
            yvex_cli_out_writef(stdout, "{\"target_id\":");
            artifacts_json_string(row->target_id);
            yvex_cli_out_writef(stdout, ",\"family\":");
            artifacts_json_string(row->family);
            yvex_cli_out_writef(stdout, ",\"artifact_class\":");
            artifacts_json_string(row->artifact_class);
            yvex_cli_out_writef(stdout, ",\"artifact_status\":");
            artifacts_json_string(row->artifact_status);
            yvex_cli_out_writef(stdout, ",\"prepare_status\":");
            artifacts_json_string(row->prepare_status);
            yvex_cli_out_writef(stdout, ",\"top_blocker\":");
            artifacts_json_string(row->top_blocker);
            yvex_cli_out_writef(stdout, ",\"path\":");
            artifacts_json_string(row->path[0] ? row->path : row->expected_path);
            yvex_cli_out_writef(stdout, "}");
        }
        yvex_cli_out_writef(stdout, "]}\n");
        return;
    }
    artifacts_print_table_header();
    for (i = 0; i < rows->count; ++i) {
        artifacts_print_row_table(&rows->rows[i]);
    }
    if (options->output_mode == YVEX_ARTIFACTS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "\nmodels_root: %s\n", operator_paths->models_root);
        yvex_cli_out_writef(stdout, "gguf_root: %s\n", operator_paths->gguf_root);
        yvex_cli_out_writef(stdout, "reports_root: %s\n", operator_paths->reports_root);
        yvex_cli_out_writef(stdout, "registry_root: %s\n", operator_paths->registry_root);
        for (i = 0; i < rows->count; ++i) {
            const yvex_models_artifact_row *row = &rows->rows[i];
            yvex_cli_out_writef(stdout, "artifact.%u.target_id: %s\n", i, row->target_id);
            yvex_cli_out_writef(stdout, "artifact.%u.family: %s\n", i, row->family);
            yvex_cli_out_writef(stdout, "artifact.%u.source_status: %s\n", i, row->source_status);
            yvex_cli_out_writef(stdout, "artifact.%u.expected_artifact_path: %s\n", i,
                   row->expected_path[0] ? row->expected_path : row->path);
            yvex_cli_out_writef(stdout, "artifact.%u.download_registry_path: %s\n", i,
                   row->registry_path[0] ? row->registry_path : "none");
            yvex_cli_out_writef(stdout, "artifact.%u.source_manifest_path: %s\n", i,
                   row->source_manifest_path[0] ? row->source_manifest_path : "none");
            yvex_cli_out_writef(stdout, "artifact.%u.native_inventory_path: %s\n", i,
                   row->native_inventory_path[0] ? row->native_inventory_path : "none");
            yvex_cli_out_writef(stdout, "artifact.%u.tensor_map_path: %s\n", i,
                   row->tensor_map_path[0] ? row->tensor_map_path : "none");
            yvex_cli_out_writef(stdout, "artifact.%u.output_head_map_path: %s\n", i,
                   row->output_head_map_path[0] ? row->output_head_map_path : "none");
            yvex_cli_out_writef(stdout, "artifact.%u.tokenizer_map_path: %s\n", i,
                   row->tokenizer_map_path[0] ? row->tokenizer_map_path : "none");
            yvex_cli_out_writef(stdout, "artifact.%u.top_blocker: %s\n", i, row->top_blocker);
            yvex_cli_out_writef(stdout, "artifact.%u.detail: %s\n", i, row->detail);
        }
        yvex_cli_out_lines(stdout, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
    }
    yvex_cli_out_writef(stdout, "\nstatus: artifacts-list\n");
}

static const yvex_models_artifact_row *artifacts_find_target(
    const yvex_models_artifact_rows *rows,
    const char *target)
{
    unsigned int i;

    if (!rows || !target) return NULL;
    for (i = 0; i < rows->count; ++i) {
        if (strcmp(rows->rows[i].target_id, target) == 0) {
            return &rows->rows[i];
        }
    }
    return NULL;
}

static void artifacts_print_status(const yvex_cli_models_artifacts_options *options,
                                   const yvex_operator_paths *operator_paths,
                                   const yvex_models_artifact_row *row)
{
    char size_text[32];

    if (!row) return;
    if (row->size_bytes > 0ull) {
        model_download_format_bytes(size_text, sizeof(size_text), row->size_bytes);
    } else {
        snprintf(size_text, sizeof(size_text), "-");
    }
    if (options->output_mode == YVEX_ARTIFACTS_OUTPUT_JSON) {
        yvex_cli_out_writef(stdout, "{\"status\":\"artifacts-status\",\"target_id\":");
        artifacts_json_string(row->target_id);
        yvex_cli_out_writef(stdout, ",\"family\":");
        artifacts_json_string(row->family);
        yvex_cli_out_writef(stdout, ",\"source_status\":");
        artifacts_json_string(row->source_status);
        yvex_cli_out_writef(stdout, ",\"artifact_status\":");
        artifacts_json_string(row->artifact_status);
        yvex_cli_out_writef(stdout, ",\"expected_artifact_path\":");
        artifacts_json_string(row->expected_path[0] ? row->expected_path : row->path);
        yvex_cli_out_writef(stdout, ",\"prepare_status\":");
        artifacts_json_string(row->prepare_status);
        yvex_cli_out_writef(stdout, ",\"top_blocker\":");
        artifacts_json_string(row->top_blocker);
        yvex_cli_out_writef(stdout, "}\n");
        return;
    }
    yvex_cli_out_writef(stdout, "artifact: %s\n", row->target_id);
    yvex_cli_out_writef(stdout, "family: %s\n", row->family);
    yvex_cli_out_writef(stdout, "class: %s\n", row->artifact_class[0] ? row->artifact_class : "unknown-gguf");
    yvex_cli_out_writef(stdout, "source: %s\n", row->source_status);
    yvex_cli_out_writef(stdout, "artifact_status: %s\n", row->artifact_status);
    yvex_cli_out_writef(stdout, "expected: %s\n", row->expected_path[0] ? row->expected_path : row->path);
    yvex_cli_out_writef(stdout, "prepare: %s\n", row->prepare_status);
    yvex_cli_out_writef(stdout, "top_blocker: %s\n", row->top_blocker);
    if (strcmp(row->prepare_status, "blocked") == 0) {
        yvex_cli_out_writef(stdout, "next: %s\n", artifacts_row_next(row));
    }
    yvex_cli_out_writef(stdout, "detail: %s\n", row->detail);
    yvex_cli_out_writef(stdout, "boundary: artifact discovery only; no runtime/generation\n");
    if (options->output_mode == YVEX_ARTIFACTS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "models_root: %s\n", operator_paths->models_root);
        yvex_cli_out_writef(stdout, "gguf_root: %s\n", operator_paths->gguf_root);
        yvex_cli_out_writef(stdout, "artifact_class: %s\n", row->artifact_class);
        yvex_cli_out_writef(stdout, "artifact_path: %s\n", row->path[0] ? row->path : "missing");
        yvex_cli_out_writef(stdout, "artifact_size: %s\n", size_text);
        yvex_cli_out_writef(stdout, "source_manifest_path: %s\n",
               row->source_manifest_path[0] ? row->source_manifest_path : "none");
        yvex_cli_out_writef(stdout, "native_inventory_path: %s\n",
               row->native_inventory_path[0] ? row->native_inventory_path : "none");
        yvex_cli_out_writef(stdout, "tensor_map_path: %s\n",
               row->tensor_map_path[0] ? row->tensor_map_path : "none");
        yvex_cli_out_writef(stdout, "tensor_map_status: %s\n",
               row->tensor_map_present
                   ? (row->tensor_map_incomplete ? "incomplete-report-only" : "present-report-only")
                   : "missing");
        yvex_cli_out_writef(stdout, "output_head_map_path: %s\n",
               row->output_head_map_path[0] ? row->output_head_map_path : "none");
        yvex_cli_out_writef(stdout, "output_head_map_status: %s\n",
               row->output_head_map_present
                   ? (row->output_head_missing ? "missing-in-report" : "present-report-only")
                   : "missing");
        yvex_cli_out_writef(stdout, "tokenizer_map_path: %s\n",
               row->tokenizer_map_path[0] ? row->tokenizer_map_path : "none");
        yvex_cli_out_writef(stdout, "tokenizer_map_status: %s\n",
               row->tokenizer_map_present ? "present-report-only" : "missing");
        yvex_cli_out_lines(stdout, literal_lines_1, sizeof(literal_lines_1) / sizeof(literal_lines_1[0]));
    }
    yvex_cli_out_writef(stdout, "status: artifacts-status\n");
}

int yvex_models_artifacts_surface_command(int arg_count, char **args)
{
    yvex_cli_models_artifacts_options options;
    yvex_paths paths;
    yvex_operator_paths operator_paths;
    yvex_models_artifact_rows *rows = NULL;
    yvex_error err;
    int rc;

    rc = parse_models_artifacts_options(arg_count, args, &options);
    if (rc != 0) return rc;
    memset(&paths, 0, sizeof(paths));
    memset(&operator_paths, 0, sizeof(operator_paths));
    yvex_error_clear(&err);
    rc = yvex_operator_paths_resolve(&paths, options.models_root,
                                     &operator_paths, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    rows = (yvex_models_artifact_rows *)calloc(1u, sizeof(*rows));
    if (!rows) {
        yvex_error_set(&err, YVEX_ERR_NOMEM, "models_artifacts",
                       "artifact row allocation failed");
        return print_yvex_error(&err, exit_for_status(YVEX_ERR_NOMEM));
    }
    rc = artifacts_collect(&options, &operator_paths, rows, &err);
    if (rc != YVEX_OK) {
        free(rows);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    if (strcmp(options.action, "list") == 0) {
        artifacts_print_list(&options, &operator_paths, rows);
        free(rows);
        return 0;
    }
    {
        const yvex_models_artifact_row *row = artifacts_find_target(rows, options.target);
        if (!row) {
            free(rows);
            yvex_cli_out_writef(stderr, "yvex: unknown models artifact target: %s\n", options.target);
            return 2;
        }
        artifacts_print_status(&options, &operator_paths, row);
    }
    free(rows);
    return 0;
}
