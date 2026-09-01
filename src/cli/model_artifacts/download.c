/*
 * Provide typed option parsing, provider process lifecycle, bounded source intake checks,
 * progress/file output, control routing, and report rendering.
 *
 * CLI-only and excluded from libyvex.a; existing command syntax is preserved. Source download
 * facts do not make artifacts generation-capable.
 */
#include "src/cli/model_artifacts/private.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <yvex/core.h>

static volatile sig_atomic_t model_download_provider_signal_seen;

static int model_download_write_all_fd(int fd, const void *buf, size_t len);
static void model_download_mirror_provider_bytes(int fd,
                                                 const char *buf,
                                                 size_t len,
                                                 int normalize_cr);
static int model_download_install_provider_signal_handlers(
    struct sigaction *old_int,
    struct sigaction *old_term,
    yvex_error *err);
static void model_download_restore_provider_signal_handlers(
    const struct sigaction *old_int,
    const struct sigaction *old_term);
static void model_download_print_start_progress(
    const yvex_model_download_report *report,
    yvex_model_download_progress_mode effective_mode);
static void model_download_print_tick_progress(
    const char *source_dir,
    time_t started_at,
    yvex_model_download_report *report,
    yvex_model_download_progress_mode effective_mode);
static void model_download_reset_child_signal_handlers(void);
static int model_download_set_nonblocking(int fd);
static void model_download_record_child_exit_status(
    yvex_model_download_report *report,
    int status);
static void model_download_mark_provider_interrupted(
    yvex_model_download_report *report,
    int signo,
    pid_t pgid);
static void model_download_orphan_check(yvex_model_download_report *report);
static const char *model_download_safetensors_file_status(const char *path);

static const char *const model_download_default_includes[] = { "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
    "*.jinja",
    "*.md"};

static const char *const model_download_default_excludes[] = { "*.bin",
    "*.pt",
    "*.onnx",
    "*.msgpack",
    "*.tflite",
    "*.h5",
    "*.ckpt",
    "*.tar",
    "*.zip"};

int model_download_family_valid(const char *family)
{
    const unsigned char *cursor = (const unsigned char *)family;
    size_t length;

    if (!family || !family[0]) return 0;
    length = strlen(family);
    if (length >= YVEX_REMOTE_FAMILY_CAP || !islower(*cursor)) return 0;
    while (*cursor) {
        if (!(islower(*cursor) || isdigit(*cursor) || *cursor == '-' ||
              *cursor == '_' || *cursor == '.'))
            return 0;
        ++cursor;
    }
    return 1;
}

int model_download_local_name_valid(const char *name)
{
    const unsigned char *p = (const unsigned char *)name;

    if (!name || !name[0]) return 0;
    while (*p) {
        if (!(isalnum(*p) || *p == '-' || *p == '_' || *p == '.')) {
            return 0;
        }
        ++p;
    }
    return 1;
}

int model_download_repo_valid(const char *repo)
{
    const unsigned char *p = (const unsigned char *)repo;
    int slash_count = 0;

    if (!repo || !repo[0] || repo[0] == '/' || strstr(repo, "..")) return 0;
    while (*p) {
        if (*p == '/') {
            slash_count++;
            if (p[1] == '\0') return 0;
        } else if (!(isalnum(*p) || *p == '-' || *p == '_' || *p == '.')) {
            return 0;
        }
        ++p;
    }
    return slash_count == 1;
}

unsigned int model_download_effective_include_count(const yvex_cli_models_download_options *options)
{
    return options && options->include_count
        ? options->include_count
        : (unsigned int)(sizeof(model_download_default_includes) /
                         sizeof(model_download_default_includes[0]));
}

unsigned int model_download_effective_exclude_count(const yvex_cli_models_download_options *options)
{
    return options && options->exclude_count
        ? options->exclude_count
        : (unsigned int)(sizeof(model_download_default_excludes) /
                         sizeof(model_download_default_excludes[0]));
}

const char *model_download_effective_include_at(const yvex_cli_models_download_options *options,
                                                       unsigned int index)
{
    if (options && options->include_count) return options->include_patterns[index];
    return model_download_default_includes[index];
}

const char *model_download_effective_exclude_at(const yvex_cli_models_download_options *options,
                                                       unsigned int index)
{
    if (options && options->exclude_count) return options->exclude_patterns[index];
    return model_download_default_excludes[index];
}

static const yvex_model_download_report model_download_report_defaults = {
    .status = "model-download-fail", .provider = "huggingface",
    .auth_state = "not-provided", .credential_source = "unknown",
    .account_hint = "unknown", .provider_cli_status = "unknown",
    .stage_resolve_target = "fail", .stage_resolve_paths = "fail",
    .stage_prepare_dirs = "skipped", .stage_account_provider = "skipped",
    .stage_provider_cli = "skipped", .stage_hf_cli = "skipped",
    .stage_download = "skipped", .stage_progress_stream = "skipped",
    .stage_progress_ticks = "skipped", .stage_source_scan = "skipped",
    .stage_source_manifest = "skipped", .stage_native_inventory = "skipped",
    .stage_sidecar = "skipped", .hf_exit_code = -1, .provider_exit_code = -1,
    .provider_pid = -1, .provider_process_group = -1,
    .child_exit_status = "unknown", .orphan_check_status = "unknown",
};

void model_download_report_init(yvex_model_download_report *report)
{
    if (!report) return;
    *report = model_download_report_defaults;
}

static int model_download_path_under(const char *path, const char *dir)
{
    size_t dir_len;

    if (!path || !dir || !path[0] || !dir[0]) return 0;
    dir_len = strlen(dir);
    while (dir_len > 1u && dir[dir_len - 1u] == '/') {
        --dir_len;
    }
    return strncmp(path, dir, dir_len) == 0 &&
           (path[dir_len] == '\0' || path[dir_len] == '/');
}

int model_download_source_path_allowed(const yvex_operator_paths *operator_paths,
                                              const char *local_source_dir,
                                              yvex_model_download_report *report)
{
    char repo_root[YVEX_PATH_CAP];

    if (!operator_paths || !local_source_dir) return 0;
    if (strcmp(operator_paths->models_root_source, "explicit") == 0) {
        return 1;
    }
    if (!getcwd(repo_root, sizeof(repo_root))) {
        snprintf(report->top_blocker, sizeof(report->top_blocker),
                 "repository-path-check-unavailable");
        snprintf(report->error, sizeof(report->error), "getcwd failed while checking local source path");
        return 0;
    }
    if (model_download_path_under(local_source_dir, repo_root)) {
        snprintf(report->top_blocker, sizeof(report->top_blocker),
                 "unsafe-repository-source-path");
        snprintf(report->error, sizeof(report->error),
                 "resolved local source path is inside the repository tree");
        return 0;
    }
    return 1;
}

int model_download_name_starts_with(const char *name, const char *prefix)
{
    size_t n;

    if (!name || !prefix) return 0;
    n = strlen(prefix);
    return strncmp(name, prefix, n) == 0;
}

int model_download_name_contains(const char *name, const char *needle)
{
    return name && needle && strstr(name, needle) != NULL;
}

static unsigned long long model_download_lock_age_seconds(const struct stat *st)
{
    time_t now;

    if (!st) return 0ull;
    now = time(NULL);
    if (now == (time_t)-1 || st->st_mtime > now) return 0ull;
    return (unsigned long long)(now - st->st_mtime);
}

typedef struct {
    yvex_model_download_source_scan *scan;
    yvex_model_download_safetensors_check *safetensors;
    const char *owner;
    int strict_io;
} model_download_tree_walk;

static int model_download_copy_path(char *out,
                                    size_t cap,
                                    const char *value,
                                    const char *owner,
                                    const char *message,
                                    yvex_error *err)
{
    int written = snprintf(out, cap, "%s", value);
    if (written >= 0 && (size_t)written < cap) return YVEX_OK;
    yvex_error_set(err, YVEX_ERR_BOUNDS, owner, message);
    return YVEX_ERR_BOUNDS;
}

static int model_download_walk_tree(const char *root,
                                    const char *rel_dir,
                                    model_download_tree_walk *walk,
                                    yvex_error *err)
{
    char abs_dir[YVEX_PATH_CAP];
    DIR *dir;
    struct dirent *ent;
    int rc = YVEX_OK;

    if (rel_dir && rel_dir[0]) {
        rc = path_join2(abs_dir, sizeof(abs_dir), root, rel_dir, err, walk->owner);
        if (rc != YVEX_OK) return rc;
    } else {
        rc = model_download_copy_path(abs_dir, sizeof(abs_dir), root, walk->owner,
                                      "source path is too long", err);
        if (rc != YVEX_OK) return rc;
    }

    dir = opendir(abs_dir);
    if (!dir) {
        if (walk->strict_io) {
            yvex_error_setf(err, YVEX_ERR_IO, walk->owner,
                            "cannot open source directory: %s", abs_dir);
            return YVEX_ERR_IO;
        }
        return YVEX_OK;
    }

    while ((ent = readdir(dir)) != NULL) {
        char rel_path[YVEX_PATH_CAP];
        char abs_path[YVEX_PATH_CAP];
        struct stat st;
        const char *base = ent->d_name;

        if (strcmp(base, ".") == 0 || strcmp(base, "..") == 0) {
            continue;
        }
        if (rel_dir && rel_dir[0]) {
            rc = path_join2(rel_path, sizeof(rel_path), rel_dir, base, err, walk->owner);
        } else {
            rc = model_download_copy_path(rel_path, sizeof(rel_path), base, walk->owner,
                                          "relative source path is too long", err);
        }
        if (rc != YVEX_OK) break;
        rc = path_join2(abs_path, sizeof(abs_path), root, rel_path, err, walk->owner);
        if (rc != YVEX_OK) break;
        if (lstat(abs_path, &st) != 0) {
            if (!walk->strict_io) continue;
            yvex_error_setf(err, YVEX_ERR_IO, walk->owner,
                            "cannot stat source path: %s", abs_path);
            rc = YVEX_ERR_IO;
            break;
        }
        if (S_ISDIR(st.st_mode)) {
            rc = model_download_walk_tree(root, rel_path, walk, err);
            if (rc != YVEX_OK) break;
        } else if (S_ISREG(st.st_mode)) {
            unsigned long long bytes = st.st_size > 0 ? (unsigned long long)st.st_size : 0ull;
            if (walk->scan) {
                yvex_model_download_source_scan *scan = walk->scan;

                scan->file_count++;
                scan->total_regular_file_bytes += bytes;
                if (yvex_source_ends_with(rel_path, ".safetensors")) scan->safetensors_count++;
                if (yvex_source_ends_with(rel_path, ".gguf")) scan->gguf_count++;
                if (yvex_source_ends_with(rel_path, ".lock")) {
                    unsigned long long idx = scan->lock_count;
                    if (idx < YVEX_MODEL_DOWNLOAD_PATTERN_CAP) {
                        snprintf(scan->lock_paths[idx], sizeof(scan->lock_paths[idx]), "%s",
                                 rel_path);
                        scan->lock_age_seconds[idx] = model_download_lock_age_seconds(&st);
                    }
                    scan->lock_count++;
                }
                if (yvex_source_ends_with(rel_path, ".partial") ||
                    yvex_source_ends_with(rel_path, ".incomplete") ||
                    yvex_source_ends_with(rel_path, ".tmp") ||
                    model_download_name_contains(rel_path, ".part"))
                    scan->partial_file_count++;
                if (model_download_name_starts_with(rel_path, ".cache/") ||
                    model_download_name_contains(rel_path, "/.cache/"))
                    scan->cache_file_count++;
                if (strcmp(base, "config.json") == 0) scan->config_present = 1;
                if (strcmp(base, "tokenizer.json") == 0 ||
                    strcmp(base, "tokenizer.model") == 0 ||
                    strcmp(base, "tokenizer_config.json") == 0 ||
                    model_download_name_starts_with(base, "tokenizer."))
                    scan->tokenizer_present = 1;
                if (bytes > scan->largest_file_bytes) {
                    scan->largest_file_bytes = bytes;
                    snprintf(scan->largest_file_name, sizeof(scan->largest_file_name), "%s",
                             rel_path);
                }
            }
            if (walk->safetensors && yvex_source_ends_with(rel_path, ".safetensors")) {
                const char *status = model_download_safetensors_file_status(abs_path);
                yvex_model_download_safetensors_check *check = walk->safetensors;

                check->checked = 1;
                if (strcmp(status, "ok") == 0) check->ok_count++;
                else if (strcmp(status, "truncated") == 0) check->truncated_count++;
                else check->invalid_count++;
            }
        }
    }

    closedir(dir);
    return rc;
}

int model_download_scan_source(const char *root,
                                      yvex_model_download_source_scan *scan,
                                      yvex_error *err)
{
    model_download_tree_walk walk = {scan, NULL, "models_download", 1};

    if (!root || !scan) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "models_download", "source root and scan output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(scan, 0, sizeof(*scan));
    return model_download_walk_tree(root, "", &walk, err);
}

static int model_download_read_u64_le(FILE *fp, unsigned long long *out)
{
    unsigned char raw[8];
    size_t got;
    unsigned int i;
    unsigned long long v = 0ull;

    if (!fp || !out) return 0;
    got = fread(raw, 1u, sizeof(raw), fp);
    if (got != sizeof(raw)) return 0;
    for (i = 0; i < 8u; ++i) {
        v |= ((unsigned long long)raw[i]) << (8u * i);
    }
    *out = v;
    return 1;
}

static int model_download_parse_data_offsets(const char *header,
                                             unsigned long long *max_end,
                                             unsigned long long *count)
{
    const char *p = header;
    const char *needle = "data_offsets";

    if (!header || !max_end || !count) return 0;
    *max_end = 0ull;
    *count = 0ull;
    while ((p = strstr(p, needle)) != NULL) {
        const char *bracket = strchr(p, '[');
        char *endptr = NULL;
        unsigned long long start;
        unsigned long long end;

        if (!bracket) return 0;
        errno = 0;
        start = strtoull(bracket + 1, &endptr, 10);
        (void)start;
        if (errno != 0 || !endptr || *endptr != ',') return 0;
        errno = 0;
        end = strtoull(endptr + 1, &endptr, 10);
        if (errno != 0 || !endptr || *endptr != ']') return 0;
        if (end > *max_end) *max_end = end;
        (*count)++;
        p = endptr + 1;
    }
    return *count > 0ull;
}

static const char *model_download_safetensors_file_status(const char *path)
{
    FILE *fp;
    struct stat st;
    unsigned long long header_len = 0ull;
    unsigned long long max_end = 0ull;
    unsigned long long tensor_count = 0ull;
    unsigned long long required_size;
    char *header = NULL;
    size_t header_size;
    const char *status = "unknown";

    if (!path || stat(path, &st) != 0 || !S_ISREG(st.st_mode)) return "unknown";
    fp = fopen(path, "rb");
    if (!fp) return "unknown";
    if (!model_download_read_u64_le(fp, &header_len)) {
        status = "invalid-header";
        goto done;
    }
    if (header_len == 0ull || header_len > 128ull * 1024ull * 1024ull ||
        header_len > (unsigned long long)SIZE_MAX - 1ull) {
        status = "invalid-header";
        goto done;
    }
    header_size = (size_t)header_len;
    header = (char *)malloc(header_size + 1u);
    if (!header) goto done;
    if (fread(header, 1u, header_size, fp) != header_size) {
        status = "invalid-header";
        goto done;
    }
    header[header_size] = '\0';
    if (!model_download_parse_data_offsets(header, &max_end, &tensor_count)) {
        status = "invalid-header";
    } else {
        required_size = 8ull + header_len + max_end;
        status = (unsigned long long)st.st_size >= required_size ? "ok" : "truncated";
    }
done:
    free(header);
    fclose(fp);
    return status;
}

int model_download_check_safetensors_source(const char *root,
                                                   yvex_model_download_safetensors_check *check,
                                                   yvex_error *err)
{
    model_download_tree_walk walk = {NULL, check, "models_download_status", 0};
    const char *status;
    int rc;

    if (!check) return YVEX_ERR_INVALID_ARG;
    memset(check, 0, sizeof(*check));
    snprintf(check->status, sizeof(check->status), "not-checked");
    if (!root || access(root, F_OK) != 0) return YVEX_OK;
    rc = model_download_walk_tree(root, "", &walk, err);
    if (rc != YVEX_OK) return rc;
    status = !check->checked          ? "not-checked"
             : check->truncated_count ? "truncated"
             : check->invalid_count   ? "invalid-header"
                                      : "ok";
    snprintf(check->status, sizeof(check->status), "%s", status);
    return YVEX_OK;
}
typedef struct {
    int stdout_pipe[2];
    int stderr_pipe[2];
    int stdout_log_fd;
    int stderr_log_fd;
    int stdout_open;
    int stderr_open;
    int child_exited;
    int shutdown_signal_sent;
    int kill_signal_sent;
    int child_status;
    int mirror_provider;
    int normalize_cr;
    time_t started_at;
    time_t next_tick;
    time_t shutdown_deadline;
    pid_t pid;
    pid_t pgid;
    struct sigaction old_int;
    struct sigaction old_term;
    int signal_handlers_installed;
    unsigned long long shutdown_timeout_seconds;
    yvex_model_download_progress_mode effective_mode;
    unsigned long long tick_seconds;
    const char *local_source_dir;
    yvex_model_download_report *report;
    yvex_error *err;
} provider_stream_state;

static const provider_stream_state provider_stream_initial = {
    .stdout_pipe = {-1, -1},
    .stderr_pipe = {-1, -1},
    .stdout_log_fd = -1,
    .stderr_log_fd = -1,
    .stdout_open = 1,
    .stderr_open = 1,
    .shutdown_deadline = (time_t)-1,
    .pgid = -1,
};

static const size_t provider_stream_fd_offsets[] = {
    offsetof(provider_stream_state, stdout_pipe[0]),
    offsetof(provider_stream_state, stdout_pipe[1]),
    offsetof(provider_stream_state, stderr_pipe[0]),
    offsetof(provider_stream_state, stderr_pipe[1]),
    offsetof(provider_stream_state, stdout_log_fd),
    offsetof(provider_stream_state, stderr_log_fd),
};

static void provider_stream_close(provider_stream_state *state)
{
    unsigned int i;

    for (i = 0; i < sizeof(provider_stream_fd_offsets) / sizeof(provider_stream_fd_offsets[0]);
         ++i) {
        int *fd = (int *)((unsigned char *)state + provider_stream_fd_offsets[i]);
        if (*fd >= 0) close(*fd);
        *fd = -1;
    }
    if (state->signal_handlers_installed) {
        model_download_restore_provider_signal_handlers(&state->old_int, &state->old_term);
        state->signal_handlers_installed = 0;
    }
}

static int provider_stream_fail(provider_stream_state *state,
                                const char *message,
                                const char *detail)
{
    provider_stream_close(state);
    yvex_error_setf(state->err, YVEX_ERR_IO, "provider_process", "%s: %s", message, detail);
    return -1;
}

static int provider_stream_poll_failure(provider_stream_state *state)
{
    int saved_errno = errno;

    if (!state->child_exited && state->pgid > 0) {
        time_t deadline;
        pid_t waited;

        (void)kill(-state->pgid, SIGTERM);
        deadline = time(NULL) + (time_t)state->shutdown_timeout_seconds;
        while (1) {
            waited = waitpid(state->pid, &state->child_status, WNOHANG);
            if (waited == state->pid) {
                model_download_record_child_exit_status(state->report, state->child_status);
                state->child_exited = 1;
                break;
            }
            if (waited < 0 && errno != EINTR) {
                state->child_exited = 1;
                break;
            }
            if (time(NULL) >= deadline) {
                (void)kill(-state->pgid, SIGKILL);
                state->report->child_killed_after_timeout = 1;
                while (waitpid(state->pid, &state->child_status, 0) < 0 && errno == EINTR) {
                }
                model_download_record_child_exit_status(state->report, state->child_status);
                state->child_exited = 1;
                break;
            }
            (void)poll(NULL, 0, 100);
        }
    }
    provider_stream_close(state);
    model_download_orphan_check(state->report);
    yvex_error_setf(state->err, YVEX_ERR_IO, "provider_process",
                    "poll failed: %s", strerror(saved_errno));
    return -1;
}

static void provider_stream_drain(provider_stream_state *state,
                                  struct pollfd fds[2],
                                  const int which[2],
                                  nfds_t count)
{
    nfds_t i;

    for (i = 0; i < count; ++i) {
        char buf[4096];
        int stream_kind = which[i];
        int read_fd = stream_kind == 1 ? state->stdout_pipe[0] : state->stderr_pipe[0];
        int log_fd = stream_kind == 1 ? state->stdout_log_fd : state->stderr_log_fd;
        int mirror_fd = stream_kind == 1 ? STDOUT_FILENO : STDERR_FILENO;
        int *open = stream_kind == 1 ? &state->stdout_open : &state->stderr_open;
        int *streamed = stream_kind == 1 ? &state->report->stdout_streamed
                                         : &state->report->stderr_streamed;
        unsigned long long *bytes = stream_kind == 1 ? &state->report->stdout_bytes
                                                     : &state->report->stderr_bytes;
        ssize_t got;

        if ((fds[i].revents & (POLLIN | POLLHUP | POLLERR)) == 0) continue;
        got = read(read_fd, buf, sizeof(buf));
        if (got > 0) {
            (void)model_download_write_all_fd(log_fd, buf, (size_t)got);
            *bytes += (unsigned long long)got;
            *streamed = 1;
            if (state->mirror_provider) {
                model_download_mirror_provider_bytes(mirror_fd, buf, (size_t)got,
                                                     state->normalize_cr);
                fflush(stream_kind == 1 ? stdout : stderr);
            }
        } else if (got == 0 ||
                   (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)) {
            if (*open) {
                close(read_fd);
                if (stream_kind == 1) state->stdout_pipe[0] = -1;
                else state->stderr_pipe[0] = -1;
                *open = 0;
            }
        }
    }
}

static int provider_stream_iteration(provider_stream_state *state)
{
    struct pollfd fds[2];
    int which[2];
    nfds_t count = 0;
    time_t now;
    int timeout_ms = 1000;
    int poll_rc;
    pid_t waited;

    if (!state->child_exited) {
        waited = waitpid(state->pid, &state->child_status, WNOHANG);
        if (waited == state->pid) {
            state->child_exited = 1;
            model_download_record_child_exit_status(state->report, state->child_status);
        } else if (waited < 0 && errno == ECHILD) {
            state->child_exited = 1;
            snprintf(state->report->child_exit_status,
                     sizeof(state->report->child_exit_status), "unknown");
        } else if (waited < 0 && errno != EINTR) {
            state->child_exited = 1;
        }
    }
    now = time(NULL);
    if (model_download_provider_signal_seen != 0 && !state->report->interrupted &&
        !state->child_exited) {
        model_download_mark_provider_interrupted(state->report,
            (int)model_download_provider_signal_seen, state->pgid);
        state->shutdown_signal_sent = 1;
        state->shutdown_deadline = now == (time_t)-1 ? (time_t)-1
            : now + (time_t)state->shutdown_timeout_seconds;
    }
    if (state->report->interrupted && state->shutdown_signal_sent &&
        !state->kill_signal_sent && state->shutdown_deadline != (time_t)-1 &&
        now != (time_t)-1 && now >= state->shutdown_deadline &&
        (!state->child_exited || state->stdout_open || state->stderr_open)) {
        if (state->pgid > 0 && kill(-state->pgid, SIGKILL) == 0) {
            state->report->child_killed_after_timeout = 1;
        }
        state->kill_signal_sent = 1;
    }
    if (state->effective_mode != YVEX_MODEL_DOWNLOAD_PROGRESS_OFF &&
        state->tick_seconds > 0ull && now != (time_t)-1 &&
        state->next_tick != (time_t)-1 && now >= state->next_tick &&
        !state->child_exited) {
        model_download_print_tick_progress(state->local_source_dir, state->started_at,
                                           state->report, state->effective_mode);
        state->next_tick = now + (time_t)state->tick_seconds;
    }
    if (state->tick_seconds > 0ull && now != (time_t)-1 &&
        state->next_tick != (time_t)-1 && state->next_tick > now) {
        time_t diff = state->next_tick - now;
        timeout_ms = diff > 1 ? 1000 : (int)(diff * 1000);
        if (timeout_ms <= 0) timeout_ms = 100;
    }
    if (state->report->interrupted && timeout_ms > 100) timeout_ms = 100;
    if (state->stdout_open) {
        fds[count] = (struct pollfd){ state->stdout_pipe[0], POLLIN | POLLHUP | POLLERR, 0 };
        which[count++] = 1;
    }
    if (state->stderr_open) {
        fds[count] = (struct pollfd){ state->stderr_pipe[0], POLLIN | POLLHUP | POLLERR, 0 };
        which[count++] = 2;
    }
    poll_rc = poll(count ? fds : NULL, count, timeout_ms);
    if (poll_rc < 0) {
        if (errno == EINTR) return 0;
        return provider_stream_poll_failure(state);
    }
    if (poll_rc > 0) provider_stream_drain(state, fds, which, count);
    return 0;
}

/* Run the bounded provider event loop until both pipes and the child are closed. */

static int provider_stream_loop(provider_stream_state *state)
{
    while (state->stdout_open || state->stderr_open || !state->child_exited) {
        if (provider_stream_iteration(state) != 0) return -1;
    }
    return 0;
}

static int provider_process_run_streaming(const char *const *args,
                                          const char *stdout_log_path,
                                          const char *stderr_log_path,
                                          const yvex_cli_models_download_options *options,
                                          yvex_model_download_progress_mode effective_mode,
                                          unsigned long long tick_seconds,
                                          const char *local_source_dir,
                                          yvex_model_download_report *report,
                                          yvex_error *err)
{
    provider_stream_state state;

    state = provider_stream_initial;
    state.effective_mode = effective_mode;
    state.tick_seconds = tick_seconds;
    state.local_source_dir = local_source_dir;
    state.report = report;
    state.err = err;

    if (!args || !args[0] || !stdout_log_path || !stderr_log_path || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "provider_process",
                       "provider args, log paths, and report are required");
        return -1;
    }
    state.shutdown_timeout_seconds = options && options->timeout_seconds
        ? options->timeout_seconds
        : (unsigned long long)YVEX_MODEL_DOWNLOAD_INTERRUPT_TIMEOUT_SECONDS;

    state.stdout_log_fd = open(stdout_log_path, O_WRONLY | O_CREAT | O_TRUNC, 0664);
    if (state.stdout_log_fd < 0)
        return provider_stream_fail(&state, "cannot open stdout log", stdout_log_path);
    state.stderr_log_fd = open(stderr_log_path, O_WRONLY | O_CREAT | O_TRUNC, 0664);
    if (state.stderr_log_fd < 0)
        return provider_stream_fail(&state, "cannot open stderr log", stderr_log_path);
    if (pipe(state.stdout_pipe) != 0 || pipe(state.stderr_pipe) != 0)
        return provider_stream_fail(&state, "pipe failed", strerror(errno));
    if (!model_download_set_nonblocking(state.stdout_pipe[0]) ||
        !model_download_set_nonblocking(state.stderr_pipe[0]))
        return provider_stream_fail(&state, "cannot make provider pipes nonblocking",
                                    strerror(errno));

    state.mirror_provider = effective_mode != YVEX_MODEL_DOWNLOAD_PROGRESS_OFF &&
                            effective_mode != YVEX_MODEL_DOWNLOAD_PROGRESS_LOG;
    state.normalize_cr = effective_mode != YVEX_MODEL_DOWNLOAD_PROGRESS_LIVE;
    state.started_at = time(NULL);
    state.next_tick = state.started_at == (time_t)-1
        ? (time_t)-1
        : state.started_at + (time_t)(tick_seconds ? tick_seconds : 1ull);

    if (!model_download_install_provider_signal_handlers(&state.old_int, &state.old_term, err)) {
        provider_stream_close(&state);
        return -1;
    }
    state.signal_handlers_installed = 1;

    state.pid = fork();
    if (state.pid < 0) return provider_stream_fail(&state, "fork failed", strerror(errno));
    if (state.pid == 0) {
        model_download_reset_child_signal_handlers();
        if (setpgid(0, 0) != 0) _exit(127);
        close(state.stdout_pipe[0]);
        close(state.stderr_pipe[0]);
        if (dup2(state.stdout_pipe[1], STDOUT_FILENO) < 0) _exit(127);
        if (dup2(state.stderr_pipe[1], STDERR_FILENO) < 0) _exit(127);
        close(state.stdout_pipe[1]);
        close(state.stderr_pipe[1]);
        close(state.stdout_log_fd);
        close(state.stderr_log_fd);
        execv(args[0], (char *const *)args);
        _exit(127);
    }

    report->provider_pid = state.pid;
    (void)setpgid(state.pid, state.pid);
    state.pgid = getpgid(state.pid);
    if (state.pgid <= 0) state.pgid = state.pid;
    report->provider_process_group = state.pgid;
    if ((!options || !options->dry_run) && report->active_receipt_path[0]) {
        yvex_error receipt_err;
        yvex_error_clear(&receipt_err);
        (void)model_download_write_control_receipt(report->active_receipt_path,
                                                   options,
                                                   report,
                                                   "running",
                                                   &receipt_err);
    }

    close(state.stdout_pipe[1]);
    close(state.stderr_pipe[1]);
    state.stdout_pipe[1] = -1;
    state.stderr_pipe[1] = -1;
    if (provider_stream_loop(&state) != 0) return -1;

    if (!state.child_exited) {
        while (waitpid(state.pid, &state.child_status, 0) < 0) {
            if (errno == EINTR) continue;
            return provider_stream_fail(&state, "waitpid failed", strerror(errno));
        }
        model_download_record_child_exit_status(report, state.child_status);
    }
    provider_stream_close(&state);
    model_download_orphan_check(report);
    if (WIFEXITED(state.child_status)) return WEXITSTATUS(state.child_status);
    if (WIFSIGNALED(state.child_status)) return 128 + WTERMSIG(state.child_status);
    return 1;
}

int model_download_run_hf(const yvex_cli_models_download_options *options,
                                 yvex_model_download_report *report,
                                 const char *token_value,
                                 yvex_error *err)
{
    const char *args[180];
    char max_workers_buf[32];
    unsigned int i;
    unsigned int n = 0;
    yvex_model_download_progress_mode effective_mode;

    if (!options || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "models_download_hf",
                       "download options and report are required");
        return -1;
    }
    snprintf(max_workers_buf, sizeof(max_workers_buf), "%llu", options->max_workers);
    args[n++] = report->provider_cli_path;
    args[n++] = "download";
    args[n++] = report->repo_id;
    args[n++] = "--revision";
    args[n++] = report->revision;
    if (!options->dry_run) {
        args[n++] = "--local-dir";
        args[n++] = report->local_source_dir;
    }
    for (i = 0; i < model_download_effective_include_count(options); ++i) {
        args[n++] = "--include";
        args[n++] = model_download_effective_include_at(options, i);
    }
    for (i = 0; i < model_download_effective_exclude_count(options); ++i) {
        args[n++] = "--exclude";
        args[n++] = model_download_effective_exclude_at(options, i);
    }
    args[n++] = "--max-workers";
    args[n++] = max_workers_buf;
    if (options->dry_run) {
        args[n++] = "--dry-run";
    }
    if (token_value && token_value[0]) {
        args[n++] = "--token";
        args[n++] = token_value;
    }
    args[n] = NULL;

    effective_mode = model_download_effective_progress_mode(options->progress_mode);
    model_download_print_start_progress(report, effective_mode);
    return provider_process_run_streaming(
        args,
        options->dry_run ? "/dev/null" : report->stdout_log_path,
        options->dry_run ? "/dev/null" : report->stderr_log_path,
        options, effective_mode, options->dry_run ? 0ull : options->tick_seconds,
        options->dry_run ? "" : report->local_source_dir, report, err);
}

int model_download_run_github(const yvex_cli_models_download_options *options,
                                     const yvex_model_download_report *report,
                                     yvex_error *err)
{
    yvex_account_command_options command;
    unsigned int n = 0u;

    if (!options || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "models_download_github",
                       "download options and report are required");
        return -1;
    }
    memset(&command, 0, sizeof(command));
    command.args[n++] = report->provider_cli_path;
    command.args[n++] = "release";
    command.args[n++] = "download";
    if (options->release && options->release[0]) {
        command.args[n++] = options->release;
    }
    command.args[n++] = "--repo";
    command.args[n++] = report->repo_id;
    command.args[n++] = "--pattern";
    command.args[n++] = options->asset;
    command.args[n++] = "--dir";
    command.args[n++] = report->local_source_dir;
    command.args[n++] = "--skip-existing";
    command.args[n] = NULL;
    command.stdout_path = report->stdout_log_path;
    command.stderr_path = report->stderr_log_path;
    return yvex_accounts_run_provider_command(&command, err);
}

/* GGUF artifact discovery and prepare preflight UX. */

#define YVEX_MODELS_ARTIFACT_ROWS_CAP 256u

static int model_download_write_all_fd(int fd, const void *buf, size_t len)
{
    const unsigned char *p = (const unsigned char *)buf;

    while (len > 0u) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static void model_download_mirror_provider_bytes(int fd,
                                                 const char *buf,
                                                 size_t len,
                                                 int normalize_cr)
{
    size_t i;

    if (fd < 0 || !buf || len == 0u) return;
    if (!normalize_cr) {
        (void)model_download_write_all_fd(fd, buf, len);
        return;
    }
    for (i = 0; i < len; ++i) {
        char c = buf[i] == '\r' ? '\n' : buf[i];
        (void)model_download_write_all_fd(fd, &c, 1u);
    }
}

static void model_download_provider_signal_handler(int signo)
{
    if ((signo == SIGINT || signo == SIGTERM) &&
        model_download_provider_signal_seen == 0) {
        model_download_provider_signal_seen = signo;
    }
}

static int model_download_install_provider_signal_handlers(struct sigaction *old_int,
                                                           struct sigaction *old_term,
                                                           yvex_error *err)
{
    struct sigaction action;

    memset(&action, 0, sizeof(action));
    action.sa_handler = model_download_provider_signal_handler;
    sigemptyset(&action.sa_mask);

    model_download_provider_signal_seen = 0;
    if (sigaction(SIGINT, &action, old_int) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "provider_process",
                        "cannot install SIGINT handler: %s", strerror(errno));
        return 0;
    }
    if (sigaction(SIGTERM, &action, old_term) != 0) {
        (void)sigaction(SIGINT, old_int, NULL);
        yvex_error_setf(err, YVEX_ERR_IO, "provider_process",
                        "cannot install SIGTERM handler: %s", strerror(errno));
        return 0;
    }
    return 1;
}

static void model_download_restore_provider_signal_handlers(const struct sigaction *old_int,
                                                            const struct sigaction *old_term)
{
    if (old_int) (void)sigaction(SIGINT, old_int, NULL);
    if (old_term) (void)sigaction(SIGTERM, old_term, NULL);
    model_download_provider_signal_seen = 0;
}

static void model_download_reset_child_signal_handlers(void)
{
    struct sigaction action;

    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    (void)sigaction(SIGINT, &action, NULL);
    (void)sigaction(SIGTERM, &action, NULL);
}

static int model_download_set_nonblocking(int fd)
{
    int flags;

    flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return 0;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static void model_download_record_child_exit_status(yvex_model_download_report *report,
                                                    int child_status)
{
    const char *status = "unknown";

    if (!report) return;
    report->child_terminated = 1;
    if (WIFEXITED(child_status)) {
        int code = WEXITSTATUS(child_status);

        if (report->interrupted &&
            (code == 0 ||
             (report->interrupt_signal > 0 && code == 128 + report->interrupt_signal)))
            status = "interrupted";
        else
            status = code == 0 ? "exited" : "terminated";
    } else if (WIFSIGNALED(child_status)) {
        int sig = WTERMSIG(child_status);

        status = report->interrupted &&
                         (sig == report->interrupt_signal || sig == SIGINT || sig == SIGTERM)
                     ? "interrupted"
                     : "terminated";
    }
    snprintf(report->child_exit_status, sizeof(report->child_exit_status), "%s", status);
}

static void model_download_mark_provider_interrupted(yvex_model_download_report *report,
                                                     int signo,
                                                     pid_t pgid)
{
    int kill_rc;

    if (!report || report->interrupted) return;
    report->interrupted = 1;
    report->interrupt_signal = signo;
    report->partial_source_preserved = 1;
    report->lock_files_deleted = 0;
    if (pgid > 0) {
        kill_rc = kill(-pgid, signo);
        if (kill_rc == 0 || errno == ESRCH) {
            report->signal_forwarded = 1;
        }
    }
}

static void model_download_orphan_check(yvex_model_download_report *report)
{
    const char *status;
    pid_t pgid;

    if (!report) return;
    report->orphan_check_performed = 1;
    pgid = report->provider_process_group;
    status = pgid <= 0        ? "unknown"
             : kill(-pgid, 0) == 0 ? "fail"
             : errno == ESRCH      ? "pass"
                                  : "unknown";
    snprintf(report->orphan_check_status, sizeof(report->orphan_check_status), "%s", status);
}

int model_download_pid_alive(pid_t pid)
{
    if (pid <= 0) return 0;
    if (kill(pid, 0) == 0) return 1;
    return errno == EPERM;
}

int model_download_pgid_alive(pid_t pgid)
{
    if (pgid <= 0) return 0;
    if (kill(-pgid, 0) == 0) return 1;
    return errno == EPERM;
}

static const char *const download_status_pair[] = {
    "boundary: source state only, runtime unsupported", "status: model-download-status"};

static const char *const download_status_lines[] = {
    "runtime_ready: false", "generation: unsupported", "benchmark_status: not-measured",
    "status: model-download-status"};

static void model_download_print_start_progress(
    const yvex_model_download_report *report,
    yvex_model_download_progress_mode effective_mode)
{
    if (!report || effective_mode == YVEX_MODEL_DOWNLOAD_PROGRESS_OFF) {
        return;
    }
    yvex_cli_out_writef(stdout, "model-download: start target=%s\n", report->target_id);
    yvex_cli_out_writef(stdout, "provider: %s\n", report->provider);
    yvex_cli_out_writef(stdout, "repo: %s\n", report->repo_id);
    yvex_cli_out_writef(stdout, "source: %s\n", report->local_source_dir);
    yvex_cli_out_writef(stdout, "stage: account-provider %s\n", report->stage_account_provider);
    yvex_cli_out_writef(stdout, "stage: download running\n");
    fflush(stdout);
}

void model_download_format_bytes(char *out,
                                        size_t cap,
                                        unsigned long long bytes)
{
    static const char *const units[] = { "B", "KiB", "MiB", "GiB", "TiB" };
    double value = (double)bytes;
    unsigned int unit = 0u;

    if (!out || cap == 0u) return;
    while (value >= 1024.0 && unit + 1u < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }
    if (unit == 0u) {
        snprintf(out, cap, "%llu B", bytes);
    } else if (value >= 100.0) {
        snprintf(out, cap, "%.0f %s", value, units[unit]);
    } else {
        snprintf(out, cap, "%.1f %s", value, units[unit]);
    }
}

static void model_download_format_elapsed(char *out,
                                          size_t cap,
                                          unsigned long long seconds)
{
    unsigned long long hours = seconds / 3600ull;
    unsigned long long minutes = (seconds / 60ull) % 60ull;
    unsigned long long secs = seconds % 60ull;

    if (!out || cap == 0u) return;
    if (hours > 0ull) {
        snprintf(out, cap, "%lluh%02llum%02llus", hours, minutes, secs);
    } else if (minutes > 0ull) {
        snprintf(out, cap, "%llum%02llus", minutes, secs);
    } else {
        snprintf(out, cap, "%llus", secs);
    }
}

void model_download_short_file_name(char *out,
                                           size_t cap,
                                           const char *path)
{
    const char *name;
    size_t len;
    size_t head;
    size_t tail;

    if (!out || cap == 0u) return;
    if (!path || !path[0]) {
        snprintf(out, cap, "none");
        return;
    }
    name = strrchr(path, '/');
    name = name ? name + 1 : path;
    len = strlen(name);
    if (len + 1u <= cap && len <= 48u) {
        snprintf(out, cap, "%s", name);
        return;
    }
    if (cap < 16u) {
        snprintf(out, cap, "%.*s", (int)(cap - 1u), name);
        return;
    }
    head = 18u;
    tail = 22u;
    if (head + tail + 4u > cap) {
        head = (cap - 4u) / 2u;
        tail = cap - 4u - head;
    }
    if (tail > len) tail = len;
    snprintf(out, cap, "%.*s...%s", (int)head, name, name + len - tail);
}

static int model_download_tick_scan_sync(yvex_model_download_report *report,
                                         const yvex_model_download_source_scan *scan,
                                         int store)
{
    unsigned long long facts[7] = {
        scan->file_count, scan->safetensors_count, scan->gguf_count, scan->partial_file_count,
        scan->cache_file_count, scan->total_regular_file_bytes, scan->largest_file_bytes};
    int changed = report->tick_count == 0ull ||
                  memcmp(&report->tick_last_file_count, facts, sizeof(facts)) != 0 ||
                  strcmp(report->tick_last_largest_file_name, scan->largest_file_name) != 0;

    if (store) {
        memcpy(&report->tick_last_file_count, facts, sizeof(facts));
        snprintf(report->tick_last_largest_file_name,
                 sizeof(report->tick_last_largest_file_name), "%s", scan->largest_file_name);
    }
    return changed;
}

static void model_download_print_tick_progress(const char *source_dir,
                                               time_t started_at,
                                               yvex_model_download_report *report,
                                               yvex_model_download_progress_mode effective_mode)
{
    yvex_model_download_source_scan scan;
    yvex_error err;
    time_t now;
    unsigned long long elapsed = 0ull;
    unsigned long long delta_bytes = 0ull;
    char elapsed_text[32];
    char total_text[32];
    char delta_text[32];
    char largest_text[32];
    char largest_name[64];
    const char *delta_sign = "+";

    if (!report || effective_mode == YVEX_MODEL_DOWNLOAD_PROGRESS_OFF) {
        return;
    }
    memset(&scan, 0, sizeof(scan));
    yvex_error_clear(&err);
    (void)model_download_scan_source(source_dir, &scan, &err);
    now = time(NULL);
    if (now != (time_t)-1 && started_at != (time_t)-1 && now >= started_at) {
        elapsed = (unsigned long long)(now - started_at);
    }
    if (!model_download_tick_scan_sync(report, &scan, 0) &&
        elapsed < report->tick_last_elapsed_seconds + 60ull) {
        return;
    }
    if (report->tick_count > 0ull) {
        if (scan.total_regular_file_bytes >= report->tick_last_total_regular_file_bytes) {
            delta_bytes = scan.total_regular_file_bytes - report->tick_last_total_regular_file_bytes;
        } else {
            delta_sign = "-";
            delta_bytes = report->tick_last_total_regular_file_bytes - scan.total_regular_file_bytes;
        }
    }
    model_download_format_elapsed(elapsed_text, sizeof(elapsed_text), elapsed);
    model_download_format_bytes(total_text, sizeof(total_text), scan.total_regular_file_bytes);
    model_download_format_bytes(delta_text, sizeof(delta_text), delta_bytes);
    model_download_format_bytes(largest_text, sizeof(largest_text), scan.largest_file_bytes);
    model_download_short_file_name(largest_name, sizeof(largest_name),
                                   scan.largest_file_name[0] ? scan.largest_file_name : "none");
    yvex_cli_out_writef(stdout,
        "tick: elapsed=%s files=%llu partial=%llu safetensors=%llu gguf=%llu bytes=%s delta=%s%s largest=%s (%s)\n",
           elapsed_text,
           scan.file_count,
           scan.partial_file_count,
           scan.safetensors_count,
           scan.gguf_count,
           total_text,
           delta_sign,
           delta_text,
           largest_name,
           largest_text);
    fflush(stdout);
    report->tick_last_elapsed_seconds = elapsed;
    (void)model_download_tick_scan_sync(report, &scan, 1);
    report->tick_count++;
}

void model_download_print_status_report(
    const yvex_cli_models_download_options *options,
    const yvex_model_download_report *report,
    const yvex_model_download_process_match *match,
    const yvex_model_download_safetensors_check *safe_check,
    int active_receipt_present,
    int active_receipt_stale,
    int last_receipt_present,
    const char *last_receipt_status,
    pid_t provider_pid,
    pid_t provider_pgid)
{
    int provider_alive = provider_pid > 0 && model_download_pid_alive(provider_pid);
    int process_alive = provider_alive || (match && match->count > 0u);
    int resume_available = !process_alive &&
                           report->source_scan.file_count > 0ull &&
                           strcmp(safe_check->status, "ok") != 0;
    int stop_available = process_alive;
    unsigned long long i;
    char bytes_text[32];
    char largest_text[32];
    char largest_name[64];

    if (options && options->output_mode == YVEX_MODELS_OUTPUT_JSON) {
        yvex_cli_out_fputs(
            "{\"schema\":\"yvex.model.acquisition.status.v1\",\"model\":",
            stdout);
        yvex_cli_out_json_string(stdout, report->target_id);
        yvex_cli_out_fputs(",\"provider\":", stdout);
        yvex_cli_out_json_string(stdout, report->provider);
        yvex_cli_out_fputs(",\"repository\":", stdout);
        yvex_cli_out_json_string(stdout, report->repo_id);
        yvex_cli_out_fputs(",\"revision\":", stdout);
        yvex_cli_out_json_string(stdout, report->revision);
        yvex_cli_out_fputs(",\"location\":", stdout);
        yvex_cli_out_json_string(stdout, report->local_source_dir);
        yvex_cli_out_writef(
            stdout,
            ",\"active\":%s,\"resume_available\":%s,\"stop_available\":%s,"
            "\"files\":%llu,\"partial_files\":%llu,\"bytes\":%llu,"
            "\"provider_pid\":%lld,\"provider_pgid\":%lld,\"receipt\":",
            process_alive ? "true" : "false",
            resume_available ? "true" : "false",
            stop_available ? "true" : "false", report->source_scan.file_count,
            report->source_scan.partial_file_count,
            report->source_scan.total_regular_file_bytes,
            (long long)provider_pid, (long long)provider_pgid);
        yvex_cli_out_json_string(
            stdout, active_receipt_present ? report->active_receipt_path
                                           : report->last_receipt_path);
        yvex_cli_out_fputs("}\n", stdout);
        return;
    }
    if (options && options->output_mode == YVEX_MODELS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "models: download status\n");
        yvex_cli_out_writef(stdout, "target_id: %s\n", report->target_id);
        yvex_cli_out_writef(stdout, "family: %s\n", report->family);
        yvex_cli_out_writef(stdout, "provider: %s\n", report->provider);
        yvex_cli_out_writef(stdout, "repo_id: %s\n", report->repo_id);
        yvex_cli_out_writef(stdout, "revision: %s\n", report->revision);
        yvex_cli_out_writef(stdout, "local_source_dir: %s\n", report->local_source_dir);
        yvex_cli_out_writef(stdout, "receipt_path: %s\n", report->receipt_path);
        yvex_cli_out_writef(stdout, "active_receipt_path: %s\n", report->active_receipt_path);
        yvex_cli_out_writef(stdout, "receipt_status: %s\n",
               active_receipt_present ? (active_receipt_stale ? "stale-active-receipt" : "active") : "none");
        yvex_cli_out_writef(stdout, "last_receipt_path: %s\n", report->last_receipt_path);
        yvex_cli_out_writef(stdout, "last_receipt_status: %s\n",
               last_receipt_present ? (last_receipt_status && last_receipt_status[0]
                                       ? last_receipt_status : "unknown") : "none");
        yvex_cli_out_writef(stdout, "provider_pid: %lld\n", (long long)provider_pid);
        yvex_cli_out_writef(stdout, "provider_pgid: %lld\n", (long long)provider_pgid);
        yvex_cli_out_writef(stdout, "provider_process_alive: %s\n", process_alive ? "true" : "false");
        yvex_cli_out_writef(stdout, "matching_provider_process_count: %u\n", match ? match->count : 0u);
        yvex_cli_out_writef(stdout, "lock_count: %llu\n", report->source_scan.lock_count);
        for (i = 0; i < report->source_scan.lock_count && i < YVEX_MODEL_DOWNLOAD_PATTERN_CAP; ++i) {
            char abs_lock[YVEX_PATH_CAP];
            yvex_error err;
            yvex_error_clear(&err);
            if (path_join2(abs_lock, sizeof(abs_lock), report->local_source_dir,
                           report->source_scan.lock_paths[i], &err,
                           "models_download_status") == YVEX_OK) {
                yvex_cli_out_writef(stdout, "lock.%llu.path: %s\n", i, abs_lock);
            }
            yvex_cli_out_writef(stdout, "lock.%llu.age_seconds: %llu\n", i, report->source_scan.lock_age_seconds[i]);
        }
        yvex_cli_out_writef(stdout, "source_file_count: %llu\n", report->source_scan.file_count);
        yvex_cli_out_writef(stdout, "safetensors_count: %llu\n", report->source_scan.safetensors_count);
        yvex_cli_out_writef(stdout, "gguf_count: %llu\n", report->source_scan.gguf_count);
        yvex_cli_out_writef(stdout, "total_regular_file_bytes: %llu\n", report->source_scan.total_regular_file_bytes);
        yvex_cli_out_writef(stdout, "largest_file_name: %s\n",
               report->source_scan.largest_file_name[0] ? report->source_scan.largest_file_name : "none");
        yvex_cli_out_writef(stdout, "largest_file_bytes: %llu\n", report->source_scan.largest_file_bytes);
        yvex_cli_out_writef(stdout, "partial_file_count: %llu\n", report->source_scan.partial_file_count);
        yvex_cli_out_writef(stdout, "cache_file_count: %llu\n", report->source_scan.cache_file_count);
        yvex_cli_out_writef(stdout, "stdout_log: %s\n", report->stdout_log_path);
        yvex_cli_out_writef(stdout, "stderr_log: %s\n", report->stderr_log_path);
        yvex_cli_out_writef(stdout, "source_manifest_path: %s\n", report->manifest_path);
        yvex_cli_out_writef(stdout, "native_inventory_path: %s\n", report->native_inventory_path);
        yvex_cli_out_writef(stdout, "safetensors_header_checked: %s\n", safe_check->checked ? "true" : "false");
        yvex_cli_out_writef(stdout, "safetensors_size_status: %s\n", safe_check->status);
        yvex_cli_out_writef(stdout, "resume_available: %s\n", resume_available ? "true" : "false");
        yvex_cli_out_writef(stdout, "stop_available: %s\n", stop_available ? "true" : "false");
        yvex_cli_out_lines(stdout, download_status_lines,
            sizeof(download_status_lines) / sizeof(download_status_lines[0]));
        return;
    }

    model_download_format_bytes(bytes_text, sizeof(bytes_text),
                                report->source_scan.total_regular_file_bytes);
    model_download_format_bytes(largest_text, sizeof(largest_text),
                                report->source_scan.largest_file_bytes);
    model_download_short_file_name(largest_name, sizeof(largest_name),
                                   report->source_scan.largest_file_name);

    yvex_cli_out_writef(stdout, "model-download-status: target=%s\n", report->target_id);
    yvex_cli_out_writef(stdout, "provider: %s\n", report->provider);
    yvex_cli_out_writef(stdout, "repo: %s\n", report->repo_id);
    yvex_cli_out_writef(stdout, "source: %s\n", report->local_source_dir);
    yvex_cli_out_writef(stdout, "process: %s\n", process_alive ? "active" : "none");
    yvex_cli_out_writef(stdout, "last_status: %s\n",
           last_receipt_present ? (last_receipt_status && last_receipt_status[0]
                                   ? last_receipt_status : "unknown") : "none");
    yvex_cli_out_writef(stdout, "locks: %llu\n", report->source_scan.lock_count);
    yvex_cli_out_writef(stdout, "files: %llu partial=%llu safetensors=%llu gguf=%llu bytes=%s\n",
           report->source_scan.file_count,
           report->source_scan.partial_file_count,
           report->source_scan.safetensors_count,
           report->source_scan.gguf_count,
           bytes_text);
    yvex_cli_out_writef(stdout, "largest: %s (%s)\n", largest_name, largest_text);
    yvex_cli_out_writef(stdout, "safetensors_size_status: %s\n", safe_check->status);
    yvex_cli_out_writef(stdout, "resume: %s\n", resume_available ? "available" : "not-needed");
    yvex_cli_out_writef(stdout, "stop: %s\n", stop_available ? "available" : "not-needed");
    yvex_cli_out_lines(stdout, download_status_pair, sizeof(download_status_pair) / sizeof(download_status_pair[0]));
}
