/* Owner: src/cli/commands
 * Owns: public command symbols for models/fullmodel/context/moe and tensor-collection dispatch.
 * Does not own: model registry facts, model reference construction, gate execution, artifact inspection,
 *   source/native-weight inspection, output formatting, JSON/table formatting, direct stdout/stderr
 *   writing, artifact emission, runtime generation, eval, benchmark, or release decisions.
 * Invariants: this file stays a thin adapter over typed input/report/render contracts and the transitional CLI
 *   surface implementation.
 * Boundary: command dispatch does not imply quantization, artifact emission, runtime generation, benchmark
 *   evidence, or release readiness.
 * Purpose: provide public command symbols for models/fullmodel/context/moe and tensor-collection
 *   dispatch.
 * Inputs: typed command arguments and borrowed domain APIs.
 * Effects: dispatches domain calls and routes operator bytes only through CLI I/O.
 * Failure: returns a stable CLI status while preserving domain ownership. */
#include "src/cli/input/private.h"
#include "src/cli/render/private.h"
#include "src/cli/model_artifacts/private.h"
#include <yvex/internal/model_artifact.h>

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *const literal_pair_0[] = { "deleted: 0",
    "status: model-download-cleanup-blocked"};

static const char *const literal_pair_1[] = { "process: none",
    "action: none"};

static const char *const literal_lines_0[] = { "cleanup: preserved-partial-source",
    "lock_cleanup: not-attempted",
    "partial_source_preserved: true",
    "lock_files_deleted: false",
    "boundary: partial source files may exist; runtime unsupported"};

static const char *const literal_lines_1[] = { "provider_process_alive: true",
    "deleted: 0",
    "status: model-download-cleanup-blocked"};

typedef struct {
    const char *target_id;
    const char *family;
    const char *repo_id;
    const char *local_name;
    const char *revision;
    const char *provider_name;
    const char *token_value;
    yvex_model_download_resolved_target resolved;
    int resolved_dynamic;
    int token_present;
} download_identity;

static int download_identity_resolve(const yvex_cli_models_download_options *options,
    yvex_model_download_report *report, yvex_operator_paths *operator_paths,
    yvex_account_provider *provider_kind, download_identity *identity,
    yvex_error *err, int control_mode);
static int download_paths_prepare(const yvex_cli_models_download_options *options,
    yvex_model_download_report *report, const yvex_operator_paths *operator_paths,
    yvex_account_provider provider_kind, const download_identity *identity,
    yvex_error *err, int create_paths);

/* Purpose: Orchestrate the typed models command request (`yvex_models_command`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_models_command(int arg_count, char **args)
{
    return yvex_model_artifacts_surface_models_command(arg_count, args);
}

/* Purpose: Render models help from typed facts (`yvex_models_help`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
void yvex_models_help(FILE *fp)
{
    yvex_model_artifacts_surface_models_help(fp);
}

/* Purpose: Orchestrate the typed fullmodel command request (`yvex_fullmodel_command`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_fullmodel_command(int arg_count, char **args)
{
    return yvex_model_artifacts_surface_fullmodel_command(arg_count, args);
}

/* Purpose: Render fullmodel help from typed facts (`yvex_fullmodel_help`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
void yvex_fullmodel_help(FILE *fp)
{
    yvex_model_artifacts_surface_fullmodel_help(fp);
}

/* Purpose: Orchestrate the typed context command request (`yvex_context_command`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_context_command(int arg_count, char **args)
{
    return yvex_model_artifacts_surface_context_command(arg_count, args);
}

/* Purpose: Render context help from typed facts (`yvex_context_help`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
void yvex_context_help(FILE *fp)
{
    yvex_model_artifacts_surface_context_help(fp);
}

/* Purpose: Orchestrate the typed moe command request (`yvex_moe_command`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_moe_command(int arg_count, char **args)
{
    return yvex_model_artifacts_surface_moe_command(arg_count, args);
}

/* Purpose: Render moe help from typed facts (`yvex_moe_help`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
void yvex_moe_help(FILE *fp)
{
    yvex_model_artifacts_surface_moe_help(fp);
}

/* Purpose: Orchestrate the typed tensor collection command request (`yvex_tensor_collection_command`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_tensor_collection_command(int arg_count, char **args)
{
    return yvex_model_artifacts_surface_tensor_collection_command(arg_count, args);
}

/* Purpose: Render tensor collection help from typed facts (`yvex_tensor_collection_help`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
void yvex_tensor_collection_help(FILE *fp)
{
    yvex_model_artifacts_surface_tensor_collection_help(fp);
}

/* Purpose: Transfer bounded model download read receipt status data (`model_download_read_receipt_status`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int model_download_read_receipt_status(const char *path, char *status, size_t status_cap)
{
    char buf[8192];

    if (status && status_cap > 0u) status[0] = '\0';
    if (!yvex_core_file_read_text_prefix(path, buf, sizeof(buf))) return 0;
    if (!yvex_json_probe_string_field(buf, "status", status, status_cap)) {
        if (status && status_cap > 0u) snprintf(status, status_cap, "unknown");
    }
    return 1;
}

/* Purpose: Transfer bounded model download read active process data (`model_download_read_active_process`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int model_download_read_active_process(const char *active_path, pid_t *pid_out,
                                              pid_t *pgid_out)
{
    char buf[8192];
    long long pid;
    long long pgid;

    if (pid_out) *pid_out = -1;
    if (pgid_out) *pgid_out = -1;
    if (!yvex_core_file_read_text_prefix(active_path, buf, sizeof(buf))) return 0;
    pid = model_download_json_i64_field(buf, "provider_pid");
    pgid = model_download_json_i64_field(buf, "provider_pgid");
    if (pid_out) *pid_out = (pid_t)pid;
    if (pgid_out) *pgid_out = (pid_t)pgid;
    return 1;
}

/* Purpose: Transfer bounded model download find provider processes data (`model_download_find_provider_processes`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static void model_download_find_provider_processes(
    const char *local_source_dir, yvex_model_download_process_match *match)
{
    DIR *proc;
    struct dirent *ent;

    if (!match) return;
    memset(match, 0, sizeof(*match));
    match->first_pid = -1;
    match->first_pgid = -1;
    if (!local_source_dir || !local_source_dir[0]) return;
    proc = opendir("/proc");
    if (!proc) return;
    while ((ent = readdir(proc)) != NULL) {
        char *endptr = NULL;
        long pid_long;
        char cmdline_path[128];
        char cmdline[8192];
        int fd;
        ssize_t got;
        ssize_t i;
        pid_t pid;
        pid_t pgid;

        if (!isdigit((unsigned char)ent->d_name[0])) continue;
        pid_long = strtol(ent->d_name, &endptr, 10);
        if (!endptr || *endptr || pid_long <= 0) continue;
        pid = (pid_t)pid_long;
        snprintf(cmdline_path, sizeof(cmdline_path), "/proc/%ld/cmdline", pid_long);
        fd = open(cmdline_path, O_RDONLY);
        if (fd < 0) continue;
        got = read(fd, cmdline, sizeof(cmdline) - 1u);
        close(fd);
        if (got <= 0) continue;
        for (i = 0; i < got; ++i) {
            if (cmdline[i] == '\0') cmdline[i] = ' ';
        }
        cmdline[got] = '\0';
        if (!strstr(cmdline, local_source_dir)) continue;
        if (!strstr(cmdline, "hf") && !strstr(cmdline, "huggingface") &&
            !strstr(cmdline, "gh") && !strstr(cmdline, "fake-hf") &&
            !strstr(cmdline, "fake-gh")) {
            continue;
        }
        pgid = getpgid(pid);
        match->count++;
        if (match->first_pid <= 0) {
            match->first_pid = pid;
            match->first_pgid = pgid > 0 ? pgid : pid;
        }
    }
    closedir(proc);
}

/* Purpose: Construct the owned model download resolve for control state (`model_download_resolve_for_control`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int model_download_resolve_for_control(int arg_count, char **args, int start_index,
    yvex_cli_models_download_options *options, yvex_model_download_report *report,
    yvex_operator_paths *operator_paths, yvex_account_provider *provider_kind,
    yvex_error *err)
{
    download_identity identity;
    int rc;

    rc = parse_models_download_options_from(arg_count, args, start_index, options);
    if (rc != 0) return rc;
    model_download_report_init(report);
    yvex_core_timestamp_utc(report->created_at, sizeof(report->created_at));
    rc = download_identity_resolve(options, report, operator_paths, provider_kind,
                                   &identity, err, 1);
    if (rc != YVEX_OK) return rc;
    return download_paths_prepare(options, report, operator_paths, *provider_kind,
                                  &identity, err, 0);
}

/* Purpose: Transfer bounded command models download status data (`command_models_download_status`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int command_models_download_status(int arg_count, char **args)
{
    yvex_cli_models_download_options options;
    yvex_model_download_report report;
    yvex_operator_paths operator_paths;
    yvex_account_provider provider_kind;
    yvex_model_download_process_match match;
    yvex_model_download_safetensors_check safe_check;
    yvex_error err;
    pid_t receipt_pid = -1;
    pid_t receipt_pgid = -1;
    int active_present;
    int active_stale;
    int last_present;
    char last_status[64];
    int rc;

    yvex_error_clear(&err);
    rc = model_download_resolve_for_control(arg_count, args, 4, &options, &report,
                                            &operator_paths, &provider_kind, &err);
    if (rc == 1) {
        yvex_model_artifacts_surface_models_help(stdout);
        return 0;
    }
    if (rc != YVEX_OK) return rc == 2 ? 2 : print_yvex_error(&err, exit_for_status(rc));
    (void)operator_paths;
    (void)provider_kind;
    active_present = model_download_read_active_process(report.active_receipt_path,
                                                        &receipt_pid,
                                                        &receipt_pgid);
    active_stale = active_present &&
                   !(model_download_pid_alive(receipt_pid) ||
                     model_download_pgid_alive(receipt_pgid));
    last_present = model_download_read_receipt_status(report.last_receipt_path,
                                                      last_status,
                                                      sizeof(last_status));
    yvex_error_clear(&err);
    (void)model_download_scan_source(report.local_source_dir, &report.source_scan, &err);
    yvex_error_clear(&err);
    (void)model_download_check_safetensors_source(report.local_source_dir, &safe_check, &err);
    model_download_find_provider_processes(report.local_source_dir, &match);
    model_download_print_status_report(&options, &report, &match, &safe_check,
                                       active_present, active_stale,
                                       last_present, last_status,
                                       receipt_pid, receipt_pgid);
    return 0;
}

/* Purpose: Transfer bounded model download wait group gone data (`model_download_wait_group_gone`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int model_download_wait_group_gone(pid_t pgid, unsigned long long timeout_seconds)
{
    time_t start = time(NULL);

    while (model_download_pgid_alive(pgid)) {
        time_t now = time(NULL);
        if (start != (time_t)-1 && now != (time_t)-1 &&
            now >= start + (time_t)timeout_seconds) {
            return 0;
        }
        (void)poll(NULL, 0, 100);
    }
    return 1;
}

/* Purpose: Transfer bounded model download wait path gone data (`model_download_wait_path_gone`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static void model_download_wait_path_gone(const char *path,
                                          unsigned long long timeout_seconds)
{
    time_t start = time(NULL);

    if (!path || !path[0]) return;
    while (access(path, F_OK) == 0) {
        time_t now = time(NULL);
        if (start != (time_t)-1 && now != (time_t)-1 &&
            now >= start + (time_t)timeout_seconds) {
            return;
        }
        (void)poll(NULL, 0, 100);
    }
}

/* Purpose: Transfer bounded command models download stop data (`command_models_download_stop`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int command_models_download_stop(int arg_count, char **args)
{
    yvex_cli_models_download_options options;
    yvex_model_download_report report;
    yvex_operator_paths operator_paths;
    yvex_account_provider provider_kind;
    yvex_model_download_process_match match;
    yvex_error err;
    pid_t pid = -1;
    pid_t pgid = -1;
    int active_present;
    int signaled = 0;
    int killed = 0;
    int stopped = 0;
    int rc;

    yvex_error_clear(&err);
    rc = model_download_resolve_for_control(arg_count, args, 4, &options, &report,
                                            &operator_paths, &provider_kind, &err);
    if (rc != YVEX_OK) return rc == 2 ? 2 : print_yvex_error(&err, exit_for_status(rc));
    (void)operator_paths;
    (void)provider_kind;
    active_present = model_download_read_active_process(report.active_receipt_path, &pid, &pgid);
    if (!active_present || !model_download_pgid_alive(pgid)) {
        if (options.match_provider_process) {
            model_download_find_provider_processes(report.local_source_dir, &match);
            if (match.count > 0u) {
                pid = match.first_pid;
                pgid = match.first_pgid;
            }
        }
    }
    if (pgid <= 0 || !model_download_pgid_alive(pgid)) {
        yvex_cli_out_writef(stdout, "model-download-stop: target=%s\n", report.target_id);
        yvex_cli_out_lines(stdout, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
        yvex_cli_out_writef(stdout, "next: yvex evidence models download status %s --models-root %s\n",
               report.target_id, report.models_root);
        yvex_cli_out_writef(stdout, "status: model-download-stop-none\n");
        return 0;
    }
    if (!options.dry_run) {
        if (kill(-pgid, SIGTERM) == 0 || errno == ESRCH) {
            signaled = 1;
        }
        stopped = model_download_wait_group_gone(pgid, options.timeout_seconds);
        if (!stopped && options.force) {
            if (kill(-pgid, SIGKILL) == 0 || errno == ESRCH) killed = 1;
            stopped = model_download_wait_group_gone(pgid, options.timeout_seconds);
        }
        snprintf(report.status, sizeof(report.status), "model-download-stopped");
        report.provider_pid = pid;
        report.provider_process_group = pgid;
        report.signal_forwarded = signaled;
        report.child_killed_after_timeout = killed;
        report.partial_source_preserved = 1;
        report.lock_files_deleted = 0;
        model_download_wait_path_gone(report.active_receipt_path, options.timeout_seconds);
        (void)model_download_finalize_control_receipt(&options, &report, "stopped");
    }
    yvex_cli_out_writef(stdout, "model-download-stop: target=%s\n", report.target_id);
    yvex_cli_out_writef(stdout, "provider: %s\n", report.provider);
    yvex_cli_out_writef(stdout, "signal: SIGTERM\n");
    yvex_cli_out_writef(stdout, "process_group: %lld\n", (long long)pgid);
    yvex_cli_out_writef(stdout, "child_signal_forwarded: %s\n", signaled ? "true" : "false");
    yvex_cli_out_writef(stdout, "child_killed_after_timeout: %s\n", killed ? "true" : "false");
    yvex_cli_out_lines(stdout, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
    yvex_cli_out_writef(stdout, "status: %s\n",
        options.dry_run ? "model-download-stop-dry-run" : "model-download-stopped");
    return signaled || stopped || options.dry_run ? 0 : 1;
}

/* Purpose: Transfer bounded model download delete lock paths data (`model_download_delete_lock_paths`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int model_download_delete_lock_paths(const yvex_model_download_report *report,
    int dry_run, int yes, unsigned long long *candidate_index,
    unsigned long long *deleted_out)
{
    unsigned long long i;
    unsigned long long deleted = 0ull;

    if (deleted_out) *deleted_out = 0ull;
    if (!report) return 0;
    for (i = 0; i < report->source_scan.lock_count && i < YVEX_MODEL_DOWNLOAD_PATTERN_CAP; ++i) {
        char abs_lock[YVEX_PATH_CAP];
        yvex_error err;
        if (!model_download_name_contains(report->source_scan.lock_paths[i],
                                          ".cache/huggingface/download/")) {
            continue;
        }
        yvex_error_clear(&err);
        if (path_join2(abs_lock, sizeof(abs_lock), report->local_source_dir,
                       report->source_scan.lock_paths[i], &err,
                       "models_download_cleanup") != YVEX_OK) {
            continue;
        }
        yvex_cli_out_writef(stdout, "delete_candidate.%llu: %s\n",
               candidate_index ? *candidate_index : i,
               abs_lock);
        if (candidate_index) (*candidate_index)++;
        if (!dry_run && yes) {
            if (unlink(abs_lock) == 0) deleted++;
        }
    }
    if (deleted_out) *deleted_out = deleted;
    return 1;
}

/* Purpose: Transfer bounded model download remove tree path data (`model_download_remove_tree_path`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int model_download_remove_tree_path(const char *path,
                                           unsigned long long *removed_out)
{
    struct stat st;
    unsigned long long removed = 0ull;
    int rc = 0;

    if (removed_out) *removed_out = 0ull;
    if (!path || !path[0]) return -1;
    if (lstat(path, &st) != 0) {
        return errno == ENOENT ? 0 : -1;
    }
    if (!S_ISDIR(st.st_mode)) {
        if (unlink(path) != 0) return -1;
        if (removed_out) *removed_out = 1ull;
        return 0;
    }
    {
        DIR *dir = opendir(path);
        struct dirent *ent;
        if (!dir) return -1;
        while ((ent = readdir(dir)) != NULL) {
            char child[YVEX_PATH_CAP];
            unsigned long long child_removed = 0ull;
            int n;

            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
                continue;
            }
            n = snprintf(child, sizeof(child), "%s/%s", path, ent->d_name);
            if (n < 0 || (size_t)n >= sizeof(child)) {
                rc = -1;
                continue;
            }
            if (model_download_remove_tree_path(child, &child_removed) != 0) {
                rc = -1;
            } else {
                removed += child_removed;
            }
        }
        closedir(dir);
    }
    if (rmdir(path) != 0) return -1;
    removed++;
    if (removed_out) *removed_out = removed;
    return rc;
}

/* Purpose: Transfer bounded model download path exists data (`model_download_path_exists`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int model_download_path_exists(const char *path)
{
    struct stat st;
    return path && path[0] && lstat(path, &st) == 0;
}

/* Purpose: Transfer bounded model download delete path candidate data (`model_download_delete_path_candidate`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int model_download_delete_path_candidate(const char *path, int recursive,
    int dry_run, int yes, unsigned long long *candidate_index,
    unsigned long long *deleted_out, unsigned long long *missing_out)
{
    unsigned long long removed = 0ull;

    if (deleted_out) *deleted_out = 0ull;
    if (missing_out) *missing_out = 0ull;
    if (!path || !path[0]) return 0;
    yvex_cli_out_writef(stdout, "delete_candidate.%llu: %s\n",
           candidate_index ? *candidate_index : 0ull,
           path);
    if (candidate_index) (*candidate_index)++;
    if (!model_download_path_exists(path)) {
        if (missing_out) *missing_out = 1ull;
        return 1;
    }
    if (dry_run || !yes) return 1;
    if (recursive) {
        if (model_download_remove_tree_path(path, &removed) != 0) return 0;
    } else {
        if (unlink(path) != 0) return 0;
        removed = 1ull;
    }
    if (deleted_out) *deleted_out = removed;
    return 1;
}

/* Purpose: Release or reset owned model download cleanup sidecars state (`model_download_cleanup_sidecars`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static void model_download_cleanup_paths(const char *const *paths, size_t path_count,
    int dry_run, int yes, unsigned long long *candidate_index,
    unsigned long long *deleted_inout, unsigned long long *missing_inout,
    unsigned long long *failed_inout)
{
    size_t i;

    for (i = 0u; i < path_count; ++i) {
        unsigned long long deleted = 0ull;
        unsigned long long missing = 0ull;

        if (model_download_delete_path_candidate(paths[i], 0, dry_run, yes,
                                                 candidate_index, &deleted,
                                                 &missing)) {
            if (deleted_inout) *deleted_inout += deleted;
            if (missing_inout) *missing_inout += missing;
        } else if (failed_inout) {
            (*failed_inout)++;
        }
    }
}

/* Purpose: Release or reset owned command models download cleanup state (`command_models_download_cleanup`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int command_models_download_cleanup(int arg_count, char **args)
{
    yvex_cli_models_download_options options;
    yvex_model_download_report report;
    yvex_operator_paths operator_paths;
    yvex_account_provider provider_kind;
    yvex_model_download_process_match match;
    yvex_error err;
    unsigned long long deleted = 0ull;
    unsigned long long candidates = 0ull;
    unsigned long long missing = 0ull;
    unsigned long long source_deleted = 0ull;
    unsigned long long sidecar_deleted = 0ull;
    unsigned long long log_deleted = 0ull;
    unsigned long long cache_deleted = 0ull;
    unsigned long long lock_deleted = 0ull;
    unsigned long long failed_deletes = 0ull;
    const char *cleanup_status;
    const char *sidecar_paths[7];
    const char *log_paths[2];
    int sidecars_requested;
    int logs_requested;
    int rc;

    yvex_error_clear(&err);
    rc = model_download_resolve_for_control(arg_count, args, 4, &options, &report,
                                            &operator_paths, &provider_kind, &err);
    if (rc != YVEX_OK) return rc == 2 ? 2 : print_yvex_error(&err, exit_for_status(rc));
    (void)provider_kind;
    if (!model_download_source_path_allowed(&operator_paths, report.local_source_dir, &report)) {
        yvex_cli_out_writef(stdout, "model-download-cleanup: target=%s\n", report.target_id);
        yvex_cli_out_writef(stdout, "top_blocker: %s\n",
               report.top_blocker[0] ? report.top_blocker : "unsafe-source-path");
        yvex_cli_out_writef(stdout, "reason: %s\n",
               report.error[0] ? report.error : "resolved local source path is unsafe");
        yvex_cli_out_lines(stdout, literal_pair_0, sizeof(literal_pair_0) / sizeof(literal_pair_0[0]));
        return exit_for_status(YVEX_ERR_UNSUPPORTED);
    }
    yvex_error_clear(&err);
    (void)model_download_scan_source(report.local_source_dir, &report.source_scan, &err);
    model_download_find_provider_processes(report.local_source_dir, &match);
    yvex_cli_out_writef(stdout, "model-download-cleanup: target=%s\n", report.target_id);
    if (match.count > 0u) {
        yvex_cli_out_lines(stdout, literal_lines_1, sizeof(literal_lines_1) / sizeof(literal_lines_1[0]));
        return exit_for_status(YVEX_ERR_UNSUPPORTED);
    }
    sidecars_requested = options.cleanup_receipts || options.cleanup_failed_partials;
    logs_requested = options.cleanup_logs || options.cleanup_failed_partials;
    sidecar_paths[0] = report.receipt_path;
    sidecar_paths[1] = report.active_receipt_path;
    sidecar_paths[2] = report.last_receipt_path;
    sidecar_paths[3] = report.download_report_path;
    sidecar_paths[4] = report.manifest_path;
    sidecar_paths[5] = report.native_inventory_path;
    sidecar_paths[6] = report.registry_path;
    log_paths[0] = report.stdout_log_path;
    log_paths[1] = report.stderr_log_path;
    if (options.cleanup_stale_locks) {
        (void)model_download_delete_lock_paths(&report, options.dry_run, options.yes,
                                               &candidates, &lock_deleted);
        deleted += lock_deleted;
    }
    if (options.cleanup_all_provider_cache) {
        char cache_path[YVEX_PATH_CAP];
        unsigned long long cache_missing = 0ull;
        yvex_error path_err;

        yvex_error_clear(&path_err);
        if (path_join2(cache_path, sizeof(cache_path), report.local_source_dir,
                       ".cache", &path_err, "models_download_cleanup") == YVEX_OK) {
            if (model_download_delete_path_candidate(cache_path, 1,
                                                     options.dry_run,
                                                     options.yes,
                                                     &candidates,
                                                     &cache_deleted,
                                                     &cache_missing)) {
                deleted += cache_deleted;
                missing += cache_missing;
            } else {
                failed_deletes++;
            }
        }
    }
    if (options.cleanup_failed_partials) {
        unsigned long long source_missing = 0ull;

        if (model_download_delete_path_candidate(report.local_source_dir, 1,
                                                 options.dry_run,
                                                 options.yes,
                                                 &candidates,
                                                 &source_deleted,
                                                 &source_missing)) {
            deleted += source_deleted;
            missing += source_missing;
        } else {
            failed_deletes++;
        }
    }
    if (sidecars_requested) {
        model_download_cleanup_paths(sidecar_paths, sizeof(sidecar_paths) / sizeof(sidecar_paths[0]),
                                     options.dry_run, options.yes, &candidates,
                                     &sidecar_deleted, &missing, &failed_deletes);
        deleted += sidecar_deleted;
    }
    if (logs_requested) {
        model_download_cleanup_paths(log_paths, sizeof(log_paths) / sizeof(log_paths[0]),
                                     options.dry_run, options.yes, &candidates,
                                     &log_deleted, &missing, &failed_deletes);
        deleted += log_deleted;
    }
    cleanup_status = options.dry_run ? "model-download-cleanup-dry-run" :
                     failed_deletes > 0ull ? "model-download-cleanup-partial" :
                     "model-download-cleanup";
    yvex_cli_out_writef(stdout, "stale_locks: %llu\n", report.source_scan.lock_count);
    yvex_cli_out_writef(stdout, "cleanup_stale_locks: %s\n", options.cleanup_stale_locks ? "true" : "false");
    yvex_cli_out_writef(stdout, "cleanup_failed_partials: %s\n", options.cleanup_failed_partials ? "true" : "false");
    yvex_cli_out_writef(stdout, "cleanup_sidecars: %s\n", sidecars_requested ? "true" : "false");
    yvex_cli_out_writef(stdout, "cleanup_logs: %s\n", logs_requested ? "true" : "false");
    yvex_cli_out_writef(stdout, "cleanup_provider_cache: %s\n", options.cleanup_all_provider_cache ? "true" : "false");
    yvex_cli_out_writef(stdout, "delete_candidates: %llu\n", candidates);
    yvex_cli_out_writef(stdout, "missing: %llu\n", missing);
    yvex_cli_out_writef(stdout, "failed_deletes: %llu\n", failed_deletes);
    yvex_cli_out_writef(stdout, "deleted_locks: %llu\n", lock_deleted);
    yvex_cli_out_writef(stdout, "deleted_source_entries: %llu\n", source_deleted);
    yvex_cli_out_writef(stdout, "deleted_sidecars: %llu\n", sidecar_deleted);
    yvex_cli_out_writef(stdout, "deleted_logs: %llu\n", log_deleted);
    yvex_cli_out_writef(stdout, "deleted_provider_cache_entries: %llu\n", cache_deleted);
    yvex_cli_out_writef(stdout, "deleted: %llu\n", deleted);
    yvex_cli_out_writef(stdout, "dry_run: %s\n", options.dry_run ? "true" : "false");
    yvex_cli_out_writef(stdout, "yes: %s\n", options.yes ? "true" : "false");
    yvex_cli_out_writef(stdout,
        "boundary: cleanup removes only target download state under models-root; runtime unsupported\n");
    yvex_cli_out_writef(stdout, "status: %s\n", cleanup_status);
    return failed_deletes > 0ull ? 1 : 0;
}

/* Resolve provider, catalog identity, operator roots, and token provenance. */
/* Purpose: Construct the owned download identity resolve state (`download_identity_resolve`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int download_identity_resolve(const yvex_cli_models_download_options *options,
    yvex_model_download_report *report, yvex_operator_paths *operator_paths,
    yvex_account_provider *provider_kind, download_identity *identity,
    yvex_error *err, int control_mode)
{
    const yvex_model_download_catalog_row *row;
    yvex_paths paths;
    int rc;

    memset(&paths, 0, sizeof(paths));
    memset(operator_paths, 0, sizeof(*operator_paths));
    memset(identity, 0, sizeof(*identity));
    if (!yvex_account_provider_from_name(options->provider, provider_kind)) {
        yvex_cli_out_writef(stderr,
            "yvex: models download --provider requires hf|huggingface|gh|github\n");
        return 2;
    }
    identity->provider_name = yvex_account_provider_name(*provider_kind);
    rc = yvex_operator_paths_resolve(&paths, options->models_root, operator_paths, err);
    if (rc != YVEX_OK) {
        if (control_mode) return rc;
        snprintf(report->status, sizeof(report->status), "model-download-fail");
        snprintf(report->error, sizeof(report->error), "%s", yvex_error_message(err));
        return print_yvex_error(err, exit_for_status(rc));
    }
    if (options->repo) {
        identity->target_id = options->name;
        identity->family = *provider_kind == YVEX_ACCOUNT_PROVIDER_GITHUB
            ? "github" : options->family;
        identity->repo_id = options->repo;
        identity->local_name = options->name;
        identity->revision = *provider_kind == YVEX_ACCOUNT_PROVIDER_GITHUB
            ? (options->release ? options->release : "latest")
            : (options->revision ? options->revision : "main");
    } else {
        identity->resolved_dynamic = model_download_resolve_downloaded_target(
            options->target, operator_paths, &identity->resolved, err);
        row = model_download_find_catalog(options->target);
        if (identity->resolved_dynamic) {
            identity->target_id = identity->resolved.target_id;
            identity->family = identity->resolved.family;
            identity->repo_id = identity->resolved.repo_id;
            identity->local_name = identity->resolved.local_name;
            identity->revision = options->revision ? options->revision
                                                   : identity->resolved.revision;
            if (!yvex_account_provider_from_name(identity->resolved.provider,
                                                  provider_kind)) {
                *provider_kind = YVEX_ACCOUNT_PROVIDER_HUGGINGFACE;
            }
        } else if (row) {
            identity->target_id = row->target_id;
            identity->family = row->family;
            identity->repo_id = row->repo_id;
            identity->local_name = row->local_name;
            identity->revision = options->revision ? options->revision : row->revision_default;
            if (!yvex_account_provider_from_name(row->provider, provider_kind)) {
                *provider_kind = YVEX_ACCOUNT_PROVIDER_HUGGINGFACE;
            }
        } else {
            if (control_mode) {
                yvex_cli_out_writef(stderr, "yvex: unknown models download target: %s\n",
                                    options->target ? options->target : "");
                return 2;
            }
            yvex_cli_out_writef(stdout, "models: download\ntarget_id: %s\n",
                                options->target ? options->target : "");
            model_stage_print("resolve-target", "fail");
            yvex_cli_out_writef(stdout,
                "reason: unknown models download target\nstatus: model-download-unknown-target\n");
            return 2;
        }
        identity->provider_name = yvex_account_provider_name(*provider_kind);
    }
    snprintf(report->stage_resolve_target, sizeof(report->stage_resolve_target), "pass");
    snprintf(report->target_id, sizeof(report->target_id), "%s", identity->target_id);
    snprintf(report->family, sizeof(report->family), "%s", identity->family);
    snprintf(report->provider, sizeof(report->provider), "%s", identity->provider_name);
    snprintf(report->repo_id, sizeof(report->repo_id), "%s", identity->repo_id);
    snprintf(report->revision, sizeof(report->revision), "%s", identity->revision);
    snprintf(report->local_name, sizeof(report->local_name), "%s", identity->local_name);
    snprintf(report->token_env_name, sizeof(report->token_env_name), "%s",
        options->token_env ? options->token_env : yvex_account_default_token_env(*provider_kind));
    identity->token_value = getenv(report->token_env_name);
    identity->token_present = identity->token_value && identity->token_value[0];
    snprintf(report->auth_state, sizeof(report->auth_state), "%s",
             identity->token_present ? "env-token-present" : "not-provided");
    snprintf(report->models_root, sizeof(report->models_root), "%s",
             operator_paths->models_root);
    snprintf(report->models_root_source, sizeof(report->models_root_source), "%s",
             operator_paths->models_root_source);
    return 0;
}

/* Build all source, receipt, report, registry, and log paths before execution. */
/* Purpose: Construct the owned download paths prepare state (`download_paths_prepare`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int download_paths_prepare(const yvex_cli_models_download_options *options,
    yvex_model_download_report *report, const yvex_operator_paths *operator_paths,
    yvex_account_provider provider_kind, const download_identity *identity,
    yvex_error *err, int create_paths)
{
    char provider_root[YVEX_PATH_CAP];
    char family_dir[YVEX_PATH_CAP];
    char repo_dir[YVEX_PATH_CAP];
    char reports_dir[YVEX_PATH_CAP];
    char registry_dir[YVEX_PATH_CAP];
    char logs_dir[YVEX_PATH_CAP];
    char file_name[256];
    int rc;

    if (provider_kind == YVEX_ACCOUNT_PROVIDER_GITHUB) {
        rc = path_join2(provider_root, sizeof(provider_root), operator_paths->models_root,
                        "github", err, "models_download");
        if (rc == YVEX_OK) rc = path_join2(repo_dir, sizeof(repo_dir), provider_root,
                                            identity->repo_id, err, "models_download");
        if (rc == YVEX_OK) rc = path_join2(report->local_source_dir,
            sizeof(report->local_source_dir), repo_dir, identity->revision, err, "models_download");
    } else {
        rc = path_join2(family_dir, sizeof(family_dir), operator_paths->hf_root,
                        identity->family, err, "models_download");
        if (rc == YVEX_OK) rc = path_join2(report->local_source_dir,
            sizeof(report->local_source_dir), family_dir, identity->local_name,
            err, "models_download");
    }
    if (rc == YVEX_OK && identity->resolved_dynamic &&
        identity->resolved.local_source_dir[0]) {
        snprintf(report->local_source_dir, sizeof(report->local_source_dir), "%s",
                 identity->resolved.local_source_dir);
    }
    if (rc == YVEX_OK) rc = path_join2(reports_dir, sizeof(reports_dir),
        operator_paths->reports_root, identity->family, err, "models_download");
    if (rc == YVEX_OK) rc = path_join2(registry_dir, sizeof(registry_dir),
        operator_paths->registry_root, identity->family, err, "models_download");
    if (rc == YVEX_OK) rc = path_join2(logs_dir, sizeof(logs_dir),
        operator_paths->models_root, "logs", err, "models_download");
    if (rc != YVEX_OK) {
        return create_paths ? print_yvex_error(err, exit_for_status(rc)) : rc;
    }
    snprintf(report->reports_dir, sizeof(report->reports_dir), "%s", reports_dir);
    snprintf(report->registry_dir, sizeof(report->registry_dir), "%s", registry_dir);
#define DOWNLOAD_PATH(field, suffix, root) do { \
    snprintf(file_name, sizeof(file_name), "%s%s", identity->target_id, suffix); \
    if (rc == YVEX_OK) rc = path_join2(report->field, sizeof(report->field), \
                                       root, file_name, err, "models_download"); \
} while (0)
    DOWNLOAD_PATH(receipt_path, ".download.receipt", reports_dir);
    DOWNLOAD_PATH(active_receipt_path, ".download.active.json", reports_dir);
    DOWNLOAD_PATH(last_receipt_path, ".download.last.json", reports_dir);
    DOWNLOAD_PATH(download_report_path, ".download-report.json", reports_dir);
    DOWNLOAD_PATH(manifest_path, ".source-manifest.json", reports_dir);
    DOWNLOAD_PATH(native_inventory_path, ".native-inventory.json", reports_dir);
    DOWNLOAD_PATH(registry_path, ".download.json", registry_dir);
    DOWNLOAD_PATH(stdout_log_path, ".download.stdout.log", logs_dir);
    DOWNLOAD_PATH(stderr_log_path, ".download.stderr.log", logs_dir);
#undef DOWNLOAD_PATH
    if (rc == YVEX_OK && identity->resolved_dynamic) {
        if (identity->resolved.registry_path[0]) snprintf(report->registry_path,
            sizeof(report->registry_path), "%s", identity->resolved.registry_path);
        if (identity->resolved.download_report_path[0]) snprintf(report->download_report_path,
            sizeof(report->download_report_path), "%s", identity->resolved.download_report_path);
        if (identity->resolved.manifest_path[0]) snprintf(report->manifest_path,
            sizeof(report->manifest_path), "%s", identity->resolved.manifest_path);
        if (identity->resolved.native_inventory_path[0]) snprintf(report->native_inventory_path,
            sizeof(report->native_inventory_path), "%s", identity->resolved.native_inventory_path);
    }
    if (rc != YVEX_OK) return print_yvex_error(err, exit_for_status(rc));
    if (!create_paths) return YVEX_OK;
    if (!model_download_source_path_allowed(operator_paths, report->local_source_dir, report)) {
        snprintf(report->status, sizeof(report->status), "model-download-blocked");
        snprintf(report->stage_resolve_paths, sizeof(report->stage_resolve_paths), "fail");
        return model_download_finish(options, report);
    }
    snprintf(report->stage_resolve_paths, sizeof(report->stage_resolve_paths), "pass");
    {
        const char *paths[] = { report->local_source_dir, report->receipt_path,
            report->active_receipt_path, report->last_receipt_path,
            report->download_report_path, report->registry_path,
            report->stdout_log_path, report->stderr_log_path };
        size_t i;

        for (i = 0u; rc == YVEX_OK && i < sizeof(paths) / sizeof(paths[0]); ++i) {
            rc = yvex_core_mkdir_parent(paths[i], "model_registry_json", err);
        }
    }
    if (rc != YVEX_OK) {
        snprintf(report->status, sizeof(report->status), "model-download-fail");
        snprintf(report->stage_prepare_dirs, sizeof(report->stage_prepare_dirs), "fail");
        snprintf(report->error, sizeof(report->error), "%s", yvex_error_message(err));
        return model_download_finish(options, report);
    }
    snprintf(report->stage_prepare_dirs, sizeof(report->stage_prepare_dirs), "pass");
    return 0;
}

/* Admit resume state and provider credentials before writing the active receipt. */
/* Purpose: Validate download account admit before downstream use (`download_account_admit`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int download_account_admit(const yvex_cli_models_download_options *options,
    yvex_model_download_report *report, yvex_account_provider provider_kind,
    yvex_account_observation *observation, int token_present,
    yvex_error *err, int *admitted)
{
    int rc;

    *admitted = 0;
    if (options->resume) {
        yvex_model_download_process_match match;
        pid_t active_pid = -1;
        pid_t active_pgid = -1;
        int active_present = model_download_read_active_process(report->active_receipt_path,
                                                                &active_pid, &active_pgid);
        model_download_find_provider_processes(report->local_source_dir, &match);
        if ((active_present && (model_download_pid_alive(active_pid) ||
             model_download_pgid_alive(active_pgid))) || match.count > 0u) {
            snprintf(report->status, sizeof(report->status), "model-download-resume-blocked");
            snprintf(report->stage_download, sizeof(report->stage_download), "blocked");
            snprintf(report->top_blocker, sizeof(report->top_blocker), "active-provider-process");
            snprintf(report->error, sizeof(report->error),
                     "another provider process is active for this source directory");
            return model_download_finish(options, report);
        }
        yvex_error_clear(err);
        (void)model_download_scan_source(report->local_source_dir, &report->source_scan, err);
        if (report->source_scan.lock_count > 0ull && !options->clear_stale_locks) {
            snprintf(report->status, sizeof(report->status), "model-download-resume-blocked");
            snprintf(report->stage_download, sizeof(report->stage_download), "blocked");
            snprintf(report->top_blocker, sizeof(report->top_blocker), "stale-lock-candidates");
            snprintf(report->error, sizeof(report->error),
                     "stale lock candidates require --clear-stale-locks");
            return model_download_finish(options, report);
        }
        if (report->source_scan.lock_count > 0ull) {
            unsigned long long deleted = 0ull;
            (void)model_download_delete_lock_paths(report, 0, 1, NULL, &deleted);
            report->lock_files_deleted = deleted > 0ull;
            memset(&report->source_scan, 0, sizeof(report->source_scan));
        }
    }
    if (options->auth_mode == YVEX_MODEL_DOWNLOAD_AUTH_NEVER) {
        yvex_account_observe_options observe;
        memset(&observe, 0, sizeof(observe));
        observe.provider = provider_kind;
        observe.cli_override = options->cli;
        observe.token_env_name = report->token_env_name;
        rc = yvex_account_observe(&observe, observation, err);
    } else {
        yvex_account_ensure_options ensure;
        memset(&ensure, 0, sizeof(ensure));
        ensure.provider = provider_kind;
        ensure.cli_override = options->cli;
        ensure.token_env_name = report->token_env_name;
        ensure.interactive = options->auth_mode == YVEX_MODEL_DOWNLOAD_AUTH_REQUIRED
            ? YVEX_ACCOUNT_INTERACTIVE_NEVER : YVEX_ACCOUNT_INTERACTIVE_AUTO;
        ensure.required = options->auth_mode == YVEX_MODEL_DOWNLOAD_AUTH_REQUIRED;
        rc = yvex_account_ensure(&ensure, observation, err);
    }
    if (rc != YVEX_OK) {
        snprintf(report->status, sizeof(report->status), "model-download-fail");
        snprintf(report->stage_account_provider, sizeof(report->stage_account_provider), "fail");
        snprintf(report->error, sizeof(report->error), "%s", yvex_error_message(err));
        return model_download_finish(options, report);
    }
    snprintf(report->provider_cli_path, sizeof(report->provider_cli_path), "%s",
             observation->cli_path);
    snprintf(report->provider_cli_source, sizeof(report->provider_cli_source), "%s",
             observation->cli_source);
    snprintf(report->provider_cli_status, sizeof(report->provider_cli_status), "%.*s",
             (int)sizeof(report->provider_cli_status) - 1, observation->cli_status);
    snprintf(report->auth_state, sizeof(report->auth_state), "%s", observation->auth_state);
    snprintf(report->credential_source, sizeof(report->credential_source), "%s",
             observation->credential_source);
    snprintf(report->account_hint, sizeof(report->account_hint), "%s", observation->account_hint);
    snprintf(report->accounts_state_path, sizeof(report->accounts_state_path), "%s",
             observation->state_path);
    if (provider_kind == YVEX_ACCOUNT_PROVIDER_HUGGINGFACE) {
        snprintf(report->hf_cli_path, sizeof(report->hf_cli_path), "%s", observation->cli_path);
        snprintf(report->hf_cli_source, sizeof(report->hf_cli_source), "%.*s",
                 (int)sizeof(report->hf_cli_source) - 1, observation->cli_source);
    }
    if (!observation->cli_present) {
        snprintf(report->status, sizeof(report->status), "model-download-blocked");
        snprintf(report->stage_account_provider, sizeof(report->stage_account_provider), "blocked");
        snprintf(report->stage_provider_cli, sizeof(report->stage_provider_cli), "blocked");
        if (provider_kind == YVEX_ACCOUNT_PROVIDER_HUGGINGFACE) {
            snprintf(report->stage_hf_cli, sizeof(report->stage_hf_cli), "blocked");
        }
        snprintf(report->top_blocker, sizeof(report->top_blocker), "%s",
                 observation->top_blocker);
        snprintf(report->error, sizeof(report->error), "%s", observation->next);
        return model_download_finish(options, report);
    }
    snprintf(report->stage_provider_cli, sizeof(report->stage_provider_cli), "pass");
    if (provider_kind == YVEX_ACCOUNT_PROVIDER_HUGGINGFACE) {
        snprintf(report->stage_hf_cli, sizeof(report->stage_hf_cli), "pass");
    }
    if (options->auth_mode == YVEX_MODEL_DOWNLOAD_AUTH_NEVER) {
        snprintf(report->stage_account_provider, sizeof(report->stage_account_provider), "skipped");
    } else if (strcmp(observation->auth_state, "logged-in") == 0 ||
               strcmp(observation->auth_state, "env-token-present") == 0) {
        snprintf(report->stage_account_provider, sizeof(report->stage_account_provider), "pass");
    } else {
        snprintf(report->status, sizeof(report->status), "model-download-blocked");
        snprintf(report->stage_account_provider, sizeof(report->stage_account_provider), "blocked");
        snprintf(report->top_blocker, sizeof(report->top_blocker), "provider-login-required");
        snprintf(report->error, sizeof(report->error), "%s",
                 observation->next[0] ? observation->next : "yvex evidence accounts login provider");
        return model_download_finish(options, report);
    }
    rc = model_download_write_receipt(report->receipt_path, options, report,
                                      token_present, err);
    if (rc != YVEX_OK) {
        snprintf(report->status, sizeof(report->status), "model-download-fail");
        snprintf(report->error, sizeof(report->error), "%s", yvex_error_message(err));
        return model_download_finish(options, report);
    }
    *admitted = 1;
    return 0;
}

/* Publish source manifest, native inventory, registry sidecar, and final receipt. */
/* Purpose: Transfer bounded download source finalize data (`download_source_finalize`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int download_source_finalize(const yvex_cli_models_download_options *options,
                                    yvex_model_download_report *report,
                                    yvex_account_provider provider_kind,
                                    yvex_error *err)
{
    yvex_source_manifest_options manifest_options;
    yvex_source_manifest_summary manifest_summary;
    yvex_native_weight_options native_options;
    yvex_native_weight_table *native_table = NULL;
    int rc;

    rc = model_download_scan_source(report->local_source_dir, &report->source_scan, err);
    if (rc != YVEX_OK) {
        snprintf(report->status, sizeof(report->status), "model-download-fail");
        snprintf(report->stage_source_scan, sizeof(report->stage_source_scan), "fail");
        snprintf(report->error, sizeof(report->error), "%s", yvex_error_message(err));
        return model_download_finish(options, report);
    }
    snprintf(report->stage_source_scan, sizeof(report->stage_source_scan), "pass");
    if (options->no_manifest) {
        snprintf(report->stage_source_manifest, sizeof(report->stage_source_manifest), "skipped");
    } else {
        memset(&manifest_options, 0, sizeof(manifest_options));
        memset(&manifest_summary, 0, sizeof(manifest_summary));
        manifest_options.repo = report->repo_id;
        manifest_options.revision = report->revision;
        manifest_options.local_path = report->local_source_dir;
        manifest_options.node_name = report->local_name;
        manifest_options.download_log = report->stdout_log_path;
        manifest_options.dry_run_log = "";
        manifest_options.pid_file = "";
        manifest_options.download_command = provider_kind == YVEX_ACCOUNT_PROVIDER_GITHUB
            ? "gh release download (see receipt; token redacted)"
            : "hf download (see receipt; token redacted)";
        manifest_options.status = YVEX_SOURCE_STATUS_IN_PROGRESS;
        manifest_options.include_files = 1;
        rc = yvex_source_manifest_write_json(report->manifest_path, &manifest_options,
                                             &manifest_summary, err);
        if (rc != YVEX_OK) {
            snprintf(report->status, sizeof(report->status), "model-download-fail");
            snprintf(report->stage_source_manifest, sizeof(report->stage_source_manifest), "fail");
            snprintf(report->error, sizeof(report->error), "%s", yvex_error_message(err));
            return model_download_finish(options, report);
        }
        report->source_manifest_written = 1;
        snprintf(report->stage_source_manifest, sizeof(report->stage_source_manifest), "pass");
    }
    if (options->no_native_inventory) {
        snprintf(report->stage_native_inventory, sizeof(report->stage_native_inventory), "skipped");
    } else {
        memset(&native_options, 0, sizeof(native_options));
        memset(&report->native_summary, 0, sizeof(report->native_summary));
        native_options.source_dir = report->local_source_dir;
        native_options.recursive = 1;
        rc = yvex_native_weight_table_open(&native_table, &native_options, err);
        if (rc == YVEX_OK) rc = yvex_native_weight_table_summary(native_table,
                                                                 &report->native_summary, err);
        if (rc == YVEX_OK) rc = model_download_write_native_inventory_json(
            report->native_inventory_path, report->local_source_dir, native_table, err);
        yvex_native_weight_table_close(native_table);
        if (rc != YVEX_OK) {
            snprintf(report->status, sizeof(report->status), "model-download-fail");
            snprintf(report->stage_native_inventory, sizeof(report->stage_native_inventory), "fail");
            snprintf(report->error, sizeof(report->error), "%s", yvex_error_message(err));
            return model_download_finish(options, report);
        }
        report->native_inventory_written = 1;
        snprintf(report->stage_native_inventory, sizeof(report->stage_native_inventory), "pass");
    }
    snprintf(report->status, sizeof(report->status), "%s",
             options->resume ? "model-download-resume-pass" : "model-download-pass");
    rc = model_download_write_json_sidecar(report->download_report_path,
        "yvex.model_download.report.v1", options, report, err);
    if (rc == YVEX_OK) {
        report->report_written = 1;
        rc = model_download_write_json_sidecar(report->registry_path,
            "yvex.model_download.registry.v1", options, report, err);
    }
    if (rc != YVEX_OK) {
        snprintf(report->status, sizeof(report->status), "model-download-fail");
        snprintf(report->stage_sidecar, sizeof(report->stage_sidecar), "fail");
        snprintf(report->error, sizeof(report->error), "%s", yvex_error_message(err));
        return model_download_finish(options, report);
    }
    report->registry_written = 1;
    snprintf(report->stage_sidecar, sizeof(report->stage_sidecar), "pass");
    (void)model_download_finalize_control_receipt(options, report, "pass");
    return model_download_finish(options, report);
}

/* Purpose: Transfer bounded command models download execute data (`command_models_download_execute`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int command_models_download_execute(int arg_count, char **args, int start_index, int resume_mode)
{
    yvex_cli_models_download_options options;
    yvex_model_download_report report;
    yvex_operator_paths operator_paths;
    yvex_error err;
    yvex_account_provider provider_kind = YVEX_ACCOUNT_PROVIDER_UNKNOWN;
    yvex_account_observation account_obs;
    download_identity identity;
    int rc;

    rc = parse_models_download_options_from(arg_count, args, start_index, &options);
    if (rc == 1) {
        yvex_model_artifacts_surface_models_help(stdout);
        return 0;
    }
    if (rc != 0) return rc;
    options.resume = resume_mode;

    model_download_report_init(&report);
    yvex_error_clear(&err);
    memset(&account_obs, 0, sizeof(account_obs));
    yvex_core_timestamp_utc(report.created_at, sizeof(report.created_at));

    rc = download_identity_resolve(&options, &report, &operator_paths,
                                   &provider_kind, &identity, &err, 0);
    if (rc != 0) return rc;
    rc = download_paths_prepare(&options, &report, &operator_paths,
                                provider_kind, &identity, &err, 1);
    if (rc != 0) return rc;

    {
        int admitted = 0;
        rc = download_account_admit(&options, &report, provider_kind, &account_obs,
                                    identity.token_present, &err, &admitted);
        if (!admitted) return rc;
    }

    if (provider_kind == YVEX_ACCOUNT_PROVIDER_GITHUB) {
        report.provider_exit_code = model_download_run_github(&options, &report, &err);
        report.hf_exit_code = report.provider_exit_code;
    } else {
        report.provider_exit_code = model_download_run_hf(&options, &report,
                                                          identity.token_present
                                                              ? identity.token_value : NULL,
                                                          &err);
        report.hf_exit_code = report.provider_exit_code;
    }
    if (provider_kind == YVEX_ACCOUNT_PROVIDER_HUGGINGFACE &&
        model_download_effective_progress_mode(options.progress_mode) !=
            YVEX_MODEL_DOWNLOAD_PROGRESS_OFF) {
        snprintf(report.stage_progress_stream, sizeof(report.stage_progress_stream),
                 report.provider_exit_code < 0 ? "fail" : "pass");
        snprintf(report.stage_progress_ticks, sizeof(report.stage_progress_ticks),
                 report.tick_count > 0ull ? "pass" : "skipped");
    } else {
        snprintf(report.stage_progress_stream, sizeof(report.stage_progress_stream), "skipped");
        snprintf(report.stage_progress_ticks, sizeof(report.stage_progress_ticks), "skipped");
    }
    if (report.interrupted || report.provider_exit_code == 130 || report.provider_exit_code == 143) {
        int write_rc;
        if (!report.interrupted) {
            report.interrupted = 1;
            report.interrupt_signal = report.provider_exit_code == 143 ? SIGTERM : SIGINT;
            report.partial_source_preserved = 1;
            report.lock_files_deleted = 0;
            snprintf(report.child_exit_status, sizeof(report.child_exit_status),
                     "interrupted");
        }
        (void)model_download_scan_source(report.local_source_dir, &report.source_scan, &err);
        snprintf(report.status, sizeof(report.status), "model-download-interrupted");
        snprintf(report.stage_download, sizeof(report.stage_download), "interrupted");
        snprintf(report.stage_source_scan, sizeof(report.stage_source_scan), "partial");
        snprintf(report.top_blocker, sizeof(report.top_blocker), "provider-download-interrupted");
        snprintf(report.error, sizeof(report.error), "provider download interrupted");
        write_rc = model_download_write_json_sidecar(report.download_report_path,
                                                    "yvex.model_download.report.v1",
                                                    &options,
                                                    &report,
                                                    &err);
        if (write_rc == YVEX_OK) {
            report.report_written = 1;
            snprintf(report.stage_sidecar, sizeof(report.stage_sidecar), "pass");
        } else {
            snprintf(report.stage_sidecar, sizeof(report.stage_sidecar), "fail");
        }
        (void)model_download_finalize_control_receipt(&options, &report, "interrupted");
        return model_download_finish(&options, &report);
    }
    if (options.dry_run && report.provider_exit_code == 0) {
        snprintf(report.status, sizeof(report.status), "model-download-dry-run");
        snprintf(report.stage_download, sizeof(report.stage_download), "dry-run");
        snprintf(report.stage_source_scan, sizeof(report.stage_source_scan), "skipped");
        snprintf(report.stage_source_manifest, sizeof(report.stage_source_manifest), "skipped");
        snprintf(report.stage_native_inventory, sizeof(report.stage_native_inventory), "skipped");
        snprintf(report.stage_sidecar, sizeof(report.stage_sidecar), "skipped");
        return model_download_finish(&options, &report);
    }
    if (report.provider_exit_code != 0) {
        int write_rc;
        snprintf(report.status, sizeof(report.status), "model-download-fail");
        snprintf(report.stage_download, sizeof(report.stage_download), "fail");
        snprintf(report.top_blocker, sizeof(report.top_blocker), "provider-download-failed");
        if (report.provider_exit_code < 0 && yvex_error_message(&err)[0]) {
            snprintf(report.error, sizeof(report.error), "%s", yvex_error_message(&err));
        } else {
            snprintf(report.error, sizeof(report.error), "%s download exited nonzero",
                     report.provider);
        }
        write_rc = model_download_write_json_sidecar(report.download_report_path,
                                                    "yvex.model_download.report.v1",
                                                    &options,
                                                    &report,
                                                    &err);
        if (write_rc == YVEX_OK) {
            report.report_written = 1;
            snprintf(report.stage_sidecar, sizeof(report.stage_sidecar), "pass");
        } else {
            snprintf(report.stage_sidecar, sizeof(report.stage_sidecar), "fail");
        }
        (void)model_download_finalize_control_receipt(&options, &report, "failed");
        return model_download_finish(&options, &report);
    }
    snprintf(report.stage_download, sizeof(report.stage_download), "pass");

    return download_source_finalize(&options, &report, provider_kind, &err);
}

/* Purpose: Transfer bounded models download surface command data (`yvex_models_download_surface_command`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_models_download_surface_command(int arg_count, char **args)
{
    if (arg_count >= 4) {
        if (strcmp(args[3], "status") == 0) {
            return command_models_download_status(arg_count, args);
        }
        if (strcmp(args[3], "stop") == 0) {
            return command_models_download_stop(arg_count, args);
        }
        if (strcmp(args[3], "resume") == 0) {
            return command_models_download_execute(arg_count, args, 4, 1);
        }
        if (strcmp(args[3], "cleanup") == 0) {
            return command_models_download_cleanup(arg_count, args);
        }
    }
    return command_models_download_execute(arg_count, args, 3, 0);
}
