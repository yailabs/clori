/* Owner: CLI paths command.
 * Owns: path argv validation, dispatch, help, and compatibility rendering.
 * Does not own: filesystem policy, path resolution, or directory lifecycle.
 * Invariants: bytes are written only through the CLI IO owner.
 * Boundary: consumes typed filesystem APIs and returns process exit status.
 * Purpose: provide path argv validation, dispatch, help, and compatibility rendering.
 * Inputs: typed command arguments and borrowed domain APIs.
 * Effects: dispatches domain calls and routes operator bytes only through CLI I/O.
 * Failure: returns a stable CLI status while preserving domain ownership. */
#include "src/cli/input/private.h"
#include <yvex/internal/source.h>
#include "src/cli/io/private.h"
#include <yvex/core.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *const literal_lines_0[] = { "       yvex evidence paths [--project DIR] --run [--create]",
    "       yvex evidence paths [--project DIR] configure --models-root DIR [--create]",
    "       yvex evidence paths [--project DIR] configure --reset",
    "       yvex evidence paths [--project DIR] resolve --family deepseek|glm|qwen|gemma --kind source|gguf|reports|"
        "reference|registry\n",
    "Path configuration records operator-local storage only; it does not download models, create artifacts,"
        " register aliases, or claim runtime support."
};

/* Domain-owned command surface moved out of core.c. */

typedef enum {
    YVEX_PATHS_OUTPUT_NORMAL = 0,
    YVEX_PATHS_OUTPUT_AUDIT
} yvex_paths_output_mode;

/* Format one CLI-owned target path with the historical typed bounds refusal. */
/* Purpose: Compute target path format for its CLI invariant (`target_path_format`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int target_path_format(char *out,
                              size_t cap,
                              yvex_error *err,
                              const char *format,
                              ...)
{
    va_list args;
    int length;

    if (!out || cap == 0u || !format) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths",
                       "invalid path format argument");
        return YVEX_ERR_INVALID_ARG;
    }
    va_start(args, format);
    length = vsnprintf(out, cap, format, args);
    va_end(args);
    if (length < 0 || (size_t)length >= cap) {
        out[cap - 1u] = '\0';
        yvex_error_setf(err, YVEX_ERR_BOUNDS, "operator_paths",
                        "path exceeds capacity %lu", (unsigned long)cap);
        return YVEX_ERR_BOUNDS;
    }
    return YVEX_OK;
}

/* Resolve a CLI family/kind target without mutating or admitting the path. */
/* Purpose: Construct the owned operator paths resolve target state (`yvex_operator_paths_resolve_target`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_operator_paths_resolve_target(const yvex_operator_paths *operator_paths,
                                       const char *family,
                                       const char *kind,
                                       char *out,
                                       size_t cap,
                                       int *out_exists,
                                       yvex_error *err)
{
    struct stat status;
    int rc;

    if (!operator_paths || !family || !kind || !out || !out_exists) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths",
                       "family, kind and outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(family, "deepseek") != 0 && strcmp(family, "glm") != 0 &&
        strcmp(family, "qwen") != 0 && strcmp(family, "gemma") != 0) {
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "operator_paths",
                        "unknown family: %s", family);
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(kind, "source") == 0 && strcmp(family, "deepseek") == 0) {
        if (!yvex_source_target_path(out, cap, operator_paths->models_root,
                                     yvex_source_release_identity())) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "operator_paths",
                           "DeepSeek source path exceeds output capacity");
            return YVEX_ERR_BOUNDS;
        }
        rc = YVEX_OK;
    } else if (strcmp(kind, "source") == 0) {
        const char *name = strcmp(family, "glm") == 0 ? "GLM-5.2" :
                           strcmp(family, "qwen") == 0 ? "qwen3-8b" : "gemma-4-12b-it";
        rc = target_path_format(out, cap, err, "%s/hf/%s/%s",
                                operator_paths->models_root, family, name);
    } else if (strcmp(kind, "gguf") == 0) {
        rc = target_path_format(out, cap, err, "%s/gguf/%s",
                                operator_paths->models_root, family);
    } else if (strcmp(kind, "reports") == 0) {
        rc = target_path_format(out, cap, err, "%s/reports/%s",
                                operator_paths->models_root, family);
    } else if (strcmp(kind, "reference") == 0) {
        rc = target_path_format(out, cap, err, "%s/reference/%s",
                                operator_paths->models_root, family);
    } else if (strcmp(kind, "registry") == 0) {
        rc = target_path_format(out, cap, err, "%s", operator_paths->registry_root);
    } else {
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "operator_paths",
                        "unknown kind: %s", kind);
        return YVEX_ERR_INVALID_ARG;
    }
    if (rc != YVEX_OK) return rc;
    *out_exists = stat(out, &status) == 0 ? 1 : 0;
    return YVEX_OK;
}

/* Purpose: Parse parse paths output mode into typed CLI state (`parse_paths_output_mode`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int parse_paths_output_mode(const char *value, yvex_paths_output_mode *mode)
{
    if (!value || !mode) {
        return 0;
    }
    if (strcmp(value, "normal") == 0) {
        *mode = YVEX_PATHS_OUTPUT_NORMAL;
        return 1;
    }
    if (strcmp(value, "audit") == 0) {
        *mode = YVEX_PATHS_OUTPUT_AUDIT;
        return 1;
    }
    return 0;
}

/* Purpose: Render print operator paths normal from typed facts (`print_operator_paths_normal`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int print_operator_paths_normal(const yvex_operator_paths *operator_paths,
                                       const char *status,
                                       yvex_error *err)
{
    if (!operator_paths || !status) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths",
                       "operator paths and status are required");
        return YVEX_ERR_INVALID_ARG;
    }

    yvex_cli_out_writef(stdout, "paths: %s\n", status);
    yvex_cli_out_writef(stdout, "models_root_source: %s\n", operator_paths->models_root_source);
    yvex_cli_out_writef(stdout, "models_root: %s\n", operator_paths->models_root);
    yvex_cli_out_writef(stdout, "hf_root: %s\n", operator_paths->hf_root);
    yvex_cli_out_writef(stdout, "gguf_root: %s\n", operator_paths->gguf_root);
    yvex_cli_out_writef(stdout, "reports_root: %s\n", operator_paths->reports_root);
    yvex_cli_out_writef(stdout, "registry_root: %s\n", operator_paths->registry_root);
    yvex_cli_out_writef(stdout, "hint: use --audit for project/cache/state paths\n");
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: Render print paths audit from typed facts (`print_paths_audit`). */
static int print_paths_audit(const yvex_paths *paths, yvex_error *err)
{
    if (!paths) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "paths_render",
                       "path facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_cli_out_writef(stdout, "config: %s\n", paths->config_dir);
    yvex_cli_out_writef(stdout, "cache: %s\n", paths->cache_dir);
    yvex_cli_out_writef(stdout, "state: %s\n", paths->state_dir);
    yvex_cli_out_writef(stdout, "data: %s\n", paths->data_dir);
    yvex_cli_out_writef(stdout, "project: %s\n", paths->project_dir);
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: Render print operator paths audit from typed facts (`print_operator_paths_audit`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int print_operator_paths_audit(const yvex_operator_paths *paths,
                                      const char *status,
                                      int created,
                                      int include_created,
                                      yvex_error *err)
{
    if (!paths || !status) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths_render",
                       "operator path facts and status are required");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_cli_out_writef(stdout, "status: %s\n", status);
    yvex_cli_out_writef(stdout, "models_root_source: %s\n", paths->models_root_source);
    yvex_cli_out_writef(stdout, "models_root: %s\n", paths->models_root);
    yvex_cli_out_writef(stdout, "hf_root: %s\n", paths->hf_root);
    yvex_cli_out_writef(stdout, "gguf_root: %s\n", paths->gguf_root);
    yvex_cli_out_writef(stdout, "reports_root: %s\n", paths->reports_root);
    yvex_cli_out_writef(stdout, "reference_root: %s\n", paths->reference_root);
    yvex_cli_out_writef(stdout, "registry_root: %s\n", paths->registry_root);
    yvex_cli_out_writef(stdout, "operator_config_path: %s\n", paths->config_path);
    if (include_created) {
        yvex_cli_out_writef(stdout, "created: %s\n", created ? "true" : "false");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: Render print run dir from typed facts (`print_run_dir`). */
static int print_run_dir(const yvex_run_dir *run, yvex_error *err)
{
    if (!run) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "run_dir_render",
                       "run directory facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_cli_out_writef(stdout, "run_id: %s\n", run->run_id);
    yvex_cli_out_writef(stdout, "root: %s\n", run->root);
    yvex_cli_out_writef(stdout, "command: %s\n", run->command_path);
    yvex_cli_out_writef(stdout, "stdout: %s\n", run->stdout_path);
    yvex_cli_out_writef(stdout, "stderr: %s\n", run->stderr_path);
    yvex_cli_out_writef(stdout, "metrics: %s\n", run->metrics_path);
    yvex_cli_out_writef(stdout, "trace: %s\n", run->trace_path);
    yvex_cli_out_writef(stdout, "receipt: %s\n", run->receipt_path);
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: Orchestrate the typed command paths default request (`command_paths_default`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int command_paths_default(const yvex_paths *paths,
                                 int want_create,
                                 yvex_paths_output_mode output_mode)
{
    yvex_operator_paths operator_paths;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_operator_paths_resolve(paths, NULL, &operator_paths, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
    }
    if (want_create) {
        rc = yvex_operator_paths_create(&operator_paths, &err);
        if (rc != YVEX_OK) {
            return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
        }
        if (output_mode == YVEX_PATHS_OUTPUT_AUDIT) {
            rc = print_paths_audit(paths, &err);
            if (rc != YVEX_OK) {
                return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
            }
            return print_operator_paths_audit(&operator_paths, "paths-created", 1, 1, &err);
        }
        return print_operator_paths_normal(&operator_paths, "created", &err);
    }
    if (output_mode == YVEX_PATHS_OUTPUT_AUDIT) {
        rc = print_paths_audit(paths, &err);
        if (rc == YVEX_OK) {
            rc = print_operator_paths_audit(&operator_paths, "paths", 0, 0, &err);
        }
    } else {
        rc = print_operator_paths_normal(&operator_paths, "normal", &err);
    }
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
    }
    return 0;
}

/* Execute the optional run-directory phase after path configuration. */
/* Purpose: Orchestrate the typed command paths run request (`command_paths_run`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int command_paths_run(const yvex_paths *paths, int want_create)
{
    yvex_run_dir run;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_run_dir_prepare(&run, paths, NULL, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
    }
    if (want_create) {
        rc = yvex_run_dir_create(&run, &err);
        if (rc != YVEX_OK) {
            return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
        }
    }
    rc = print_run_dir(&run, &err);
    return rc == YVEX_OK
               ? 0
               : print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
}

/* Purpose: Orchestrate the typed command paths request (`command_paths`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int command_paths(int arg_count, char **args)
{
    const char *project_root = NULL;
    const char *models_root = NULL;
    const char *family = NULL;
    const char *kind = NULL;
    int want_run = 0;
    int want_create = 0;
    int want_reset = 0;
    yvex_paths_output_mode output_mode = YVEX_PATHS_OUTPUT_NORMAL;
    int i;
    int rc;
    int removed = 0;
    int exists = 0;
    char resolved_path[YVEX_PATH_CAP];
    yvex_paths paths;
    yvex_operator_paths operator_paths;
    yvex_error err;

    yvex_error_clear(&err);

    i = 2;
    while (i < arg_count) {
        if (strcmp(args[i], "--project") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: --project requires a path\n");
                return 2;
            }
            project_root = args[++i];
            ++i;
        } else {
            break;
        }
    }

    rc = project_root ? yvex_paths_project(&paths, project_root, &err) : yvex_paths_default(&paths, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
    }

    if (i < arg_count && strcmp(args[i], "configure") == 0) {
        ++i;
        while (i < arg_count) {
            if (strcmp(args[i], "--models-root") == 0) {
                if (i + 1 >= arg_count) {
                    yvex_cli_out_writef(stderr, "yvex: --models-root requires a path\n");
                    return 2;
                }
                models_root = args[++i];
            } else if (strcmp(args[i], "--create") == 0) {
                want_create = 1;
            } else if (strcmp(args[i], "--reset") == 0) {
                want_reset = 1;
            } else if (strcmp(args[i], "--help") == 0 || strcmp(args[i], "-h") == 0) {
                yvex_paths_help(stdout);
                return 0;
            } else {
                yvex_cli_out_writef(stderr, "yvex: unknown paths configure option: %s\n", args[i]);
                yvex_cli_out_writef(stderr, "Try 'yvex help paths' for usage.\n");
                return 2;
            }
            ++i;
        }
        if (want_reset && (models_root || want_create)) {
            yvex_cli_out_writef(stderr, "yvex: paths configure --reset does not accept --models-root or --create\n");
            return 2;
        }
        if (!want_reset && !models_root) {
            yvex_cli_out_writef(stderr, "yvex: paths configure requires --models-root DIR or --reset\n");
            return 2;
        }

        if (want_reset) {
            rc = yvex_operator_paths_reset(&paths, &removed, &operator_paths, &err);
            if (rc != YVEX_OK) {
                return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
            }
            yvex_cli_out_writef(stdout, "status: paths-reset\n");
            yvex_cli_out_writef(stdout, "config_path: %s\n", operator_paths.config_path);
            yvex_cli_out_writef(stdout, "removed: %s\n", removed ? "true" : "false");
            yvex_cli_out_writef(stdout, "models_root_source: %s\n", operator_paths.models_root_source);
            yvex_cli_out_writef(stdout, "models_root: %s\n", operator_paths.models_root);
            return 0;
        }

        rc = yvex_operator_paths_configure(&paths, models_root, want_create, &operator_paths, &err);
        if (rc != YVEX_OK) {
            return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
        }
        yvex_cli_out_writef(stdout, "status: paths-configured\n");
        yvex_cli_out_writef(stdout, "models_root_source: %s\n", operator_paths.models_root_source);
        yvex_cli_out_writef(stdout, "models_root: %s\n", operator_paths.models_root);
        yvex_cli_out_writef(stdout, "hf_root: %s\n", operator_paths.hf_root);
        yvex_cli_out_writef(stdout, "gguf_root: %s\n", operator_paths.gguf_root);
        yvex_cli_out_writef(stdout, "reports_root: %s\n", operator_paths.reports_root);
        yvex_cli_out_writef(stdout, "reference_root: %s\n", operator_paths.reference_root);
        yvex_cli_out_writef(stdout, "registry_root: %s\n", operator_paths.registry_root);
        yvex_cli_out_writef(stdout, "config_path: %s\n", operator_paths.config_path);
        yvex_cli_out_writef(stdout, "created: %s\n", want_create ? "true" : "false");
        return 0;
    }

    if (i < arg_count && strcmp(args[i], "resolve") == 0) {
        ++i;
        while (i < arg_count) {
            if (strcmp(args[i], "--family") == 0) {
                if (i + 1 >= arg_count) {
                    yvex_cli_out_writef(stderr, "yvex: --family requires a value\n");
                    return 2;
                }
                family = args[++i];
            } else if (strcmp(args[i], "--kind") == 0) {
                if (i + 1 >= arg_count) {
                    yvex_cli_out_writef(stderr, "yvex: --kind requires a value\n");
                    return 2;
                }
                kind = args[++i];
            } else if (strcmp(args[i], "--help") == 0 || strcmp(args[i], "-h") == 0) {
                yvex_paths_help(stdout);
                return 0;
            } else {
                yvex_cli_out_writef(stderr, "yvex: unknown paths resolve option: %s\n", args[i]);
                yvex_cli_out_writef(stderr, "Try 'yvex help paths' for usage.\n");
                return 2;
            }
            ++i;
        }
        if (!family) {
            yvex_cli_out_writef(stderr, "yvex: paths resolve requires --family\n");
            return 2;
        }
        if (!kind) {
            yvex_cli_out_writef(stderr, "yvex: paths resolve requires --kind\n");
            return 2;
        }
        rc = yvex_operator_paths_resolve(&paths, NULL, &operator_paths, &err);
        if (rc != YVEX_OK) {
            return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
        }
        rc = yvex_operator_paths_resolve_target(&operator_paths, family, kind,
                                                resolved_path, sizeof(resolved_path), &exists, &err);
        if (rc != YVEX_OK) {
            return print_yvex_error(&err, rc == YVEX_ERR_INVALID_ARG ? 2 : 3);
        }
        yvex_cli_out_writef(stdout, "status: paths-resolve\n");
        yvex_cli_out_writef(stdout, "models_root_source: %s\n", operator_paths.models_root_source);
        yvex_cli_out_writef(stdout, "family: %s\n", family);
        yvex_cli_out_writef(stdout, "kind: %s\n", kind);
        yvex_cli_out_writef(stdout, "path: %s\n", resolved_path);
        yvex_cli_out_writef(stdout, "exists: %s\n", exists ? "true" : "false");
        return 0;
    }

    for (; i < arg_count; ++i) {
        if (strcmp(args[i], "--run") == 0) {
            want_run = 1;
        } else if (strcmp(args[i], "--create") == 0) {
            want_create = 1;
        } else if (strcmp(args[i], "--audit") == 0) {
            output_mode = YVEX_PATHS_OUTPUT_AUDIT;
        } else if (strcmp(args[i], "--output") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex paths: --output requires normal|audit\n");
                return 2;
            }
            if (!parse_paths_output_mode(args[++i], &output_mode)) {
                yvex_cli_out_writef(stderr, "yvex paths: unsupported output mode: %s\n", args[i]);
                return 2;
            }
        } else if (strcmp(args[i], "--help") == 0 || strcmp(args[i], "-h") == 0) {
            yvex_paths_help(stdout);
            return 0;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown paths option: %s\n", args[i]);
            yvex_cli_out_writef(stderr, "Try 'yvex help paths' for usage.\n");
            return 2;
        }
    }

    if (!want_run) {
        return command_paths_default(&paths, want_create, output_mode);
    }

    return command_paths_run(&paths, want_create);
}

/* Purpose: Orchestrate the typed paths command request (`yvex_paths_command`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_paths_command(int arg_count, char **args)
{
    return command_paths(arg_count, args);
}

/* Purpose: Render paths help from typed facts (`yvex_paths_help`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
void yvex_paths_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex evidence paths [--project DIR] [--create] [--audit | --output normal|audit]\n");
    yvex_cli_out_lines(fp, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
}
