/*
 * Resolve canonical local paths and create explicitly requested run directories.
 *
 * Target-specific path facts are consumed from their canonical owner and path construction
 * performs checked capacity handling. Resolving a local path does not verify the source or support
 * a model.
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <yvex/core.h>
#include <yvex/internal/core.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif

#define CORE_FILE_NAME_CAP 256u

typedef struct {
    size_t offset;
    size_t capacity;
    const char *suffix;
} path_projection;

typedef struct {
    size_t offset;
    const char *environment;
    const char *home_suffix;
} default_path_projection;

static const default_path_projection default_path_fields[] = {
    {offsetof(yvex_paths, config_dir), "YVEX_CONFIG_DIR", "/.config/yvex"},
    {offsetof(yvex_paths, cache_dir), "YVEX_CACHE_DIR", "/.cache/yvex"},
    {offsetof(yvex_paths, state_dir), "YVEX_STATE_DIR", "/.local/state/yvex"},
    {offsetof(yvex_paths, data_dir), "YVEX_DATA_DIR", "/.local/share/yvex"},
};

static const path_projection project_path_fields[] = {
    {offsetof(yvex_paths, config_dir), YVEX_PATH_CAP, ""},
    {offsetof(yvex_paths, cache_dir), YVEX_PATH_CAP, "/cache"},
    {offsetof(yvex_paths, state_dir), YVEX_PATH_CAP, "/state"},
    {offsetof(yvex_paths, data_dir), YVEX_PATH_CAP, "/data"},
};

static const path_projection operator_root_fields[] = {
    {offsetof(yvex_operator_paths, models_root), YVEX_PATH_CAP, ""},
    {offsetof(yvex_operator_paths, hf_root), YVEX_PATH_CAP, "/hf"},
    {offsetof(yvex_operator_paths, gguf_root), YVEX_PATH_CAP, "/gguf"},
    {offsetof(yvex_operator_paths, reports_root), YVEX_PATH_CAP, "/reports"},
    {offsetof(yvex_operator_paths, reference_root), YVEX_PATH_CAP, "/reference"},
    {offsetof(yvex_operator_paths, registry_root), YVEX_PATH_CAP, "/registry"},
};

static const path_projection run_file_fields[] = {
    {offsetof(yvex_run_dir, command_path), YVEX_PATH_CAP, "/command.txt"},
    {offsetof(yvex_run_dir, stdout_path), YVEX_PATH_CAP, "/stdout.log"},
    {offsetof(yvex_run_dir, stderr_path), YVEX_PATH_CAP, "/stderr.log"},
    {offsetof(yvex_run_dir, metrics_path), YVEX_PATH_CAP, "/metrics.json"},
    {offsetof(yvex_run_dir, trace_path), YVEX_PATH_CAP, "/trace.jsonl"},
    {offsetof(yvex_run_dir, receipt_path), YVEX_PATH_CAP, "/receipt.json"},
};

static int path_format(char *dst,
                            size_t cap,
                            yvex_error *err,
                            const char *where,
                            const char *fmt,
                            ...)
{
    va_list ap;
    int n;

    if (!dst || cap == 0 || !fmt) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, "invalid path format argument");
        return YVEX_ERR_INVALID_ARG;
    }

    va_start(ap, fmt);
    n = vsnprintf(dst, cap, fmt, ap);
    va_end(ap);

    if (n < 0 || (size_t)n >= cap) {
        dst[cap - 1] = '\0';
        yvex_error_setf(err, YVEX_ERR_BOUNDS, where, "path exceeds capacity %lu", (unsigned long)cap);
        return YVEX_ERR_BOUNDS;
    }

    return YVEX_OK;
}

static const char *env_or_null(const char *name)
{
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        return NULL;
    }
    return value;
}

static int copy_path(char *dst, const char *value, yvex_error *err, const char *where)
{
    return path_format(dst, YVEX_PATH_CAP, err, where, "%s", value);
}

static int path_fields_project(void *owner, const path_projection *fields,
                               size_t count, const char *root,
                               const char *where, yvex_error *err)
{
    size_t index;

    for (index = 0u; index < count; ++index) {
        char *destination = (char *)owner + fields[index].offset;
        int rc = path_format(destination, fields[index].capacity, err, where,
                             "%s%s", root, fields[index].suffix);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

int yvex_paths_default(yvex_paths *out, yvex_error *err)
{
    const char *home;
    size_t index;
    int rc;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_paths_default", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    home = getenv("HOME");
    if (!home || home[0] == '\0') {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_paths_default", "HOME is not set; cannot resolve default YVEX paths");
        return YVEX_ERR_STATE;
    }

    for (index = 0u; index < sizeof(default_path_fields) / sizeof(default_path_fields[0]);
         ++index) {
        const default_path_projection *field = &default_path_fields[index];
        const char *environment = env_or_null(field->environment);
        char *destination = (char *)out + field->offset;
        rc = environment ? copy_path(destination, environment, err, "yvex_paths_default")
                         : path_format(destination, YVEX_PATH_CAP, err,
                                       "yvex_paths_default", "%s%s", home,
                                       field->home_suffix);
        if (rc != YVEX_OK) return rc;
    }

    out->project_dir[0] = '\0';
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_paths_project(yvex_paths *out, const char *project_root, yvex_error *err)
{
    int rc;

    if (!out || !project_root || project_root[0] == '\0') {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_paths_project", "out and project_root are required");
        return YVEX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    rc = path_format(out->project_dir, YVEX_PATH_CAP, err, "yvex_paths_project", "%s/.yvex", project_root);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = path_fields_project(out, project_path_fields,
                             sizeof(project_path_fields) / sizeof(project_path_fields[0]),
                             out->project_dir, "yvex_paths_project", err);
    if (rc != YVEX_OK) return rc;

    yvex_error_clear(err);
    return YVEX_OK;
}

static int path_is_dir(const char *path, int *out_is_dir, yvex_error *err, const char *where)
{
    struct stat st;

    if (stat(path, &st) != 0) {
        if (errno == ENOENT) {
            *out_is_dir = 0;
            return YVEX_OK;
        }
        yvex_error_setf(err, YVEX_ERR_IO, where, "stat failed for %s: %s", path, strerror(errno));
        return YVEX_ERR_IO;
    }

    *out_is_dir = S_ISDIR(st.st_mode) ? 1 : -1;
    return YVEX_OK;
}

static int mkdir_one(const char *path, const char *where, yvex_error *err)
{
    int is_dir;
    int rc;

    if (mkdir(path, 0777) == 0) {
        return YVEX_OK;
    }

    if (errno != EEXIST) {
        yvex_error_setf(err, YVEX_ERR_IO, where, "mkdir failed for %s: %s",
                        path, strerror(errno));
        return YVEX_ERR_IO;
    }

    rc = path_is_dir(path, &is_dir, err, where);
    if (rc != YVEX_OK) {
        return rc;
    }
    if (is_dir == 1) {
        return YVEX_OK;
    }

    yvex_error_setf(err, YVEX_ERR_IO, where,
                    "path exists and is not a directory: %s", path);
    return YVEX_ERR_IO;
}

static int mkdir_p(const char *path, const char *where, yvex_error *err)
{
    char tmp[YVEX_PATH_CAP];
    char *p;
    size_t len;
    int rc;

    if (!path || path[0] == '\0') {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where,
                       "directory path is required");
        return YVEX_ERR_INVALID_ARG;
    }

    rc = path_format(tmp, sizeof(tmp), err, where, "%s", path);
    if (rc != YVEX_OK) {
        return rc;
    }

    len = strlen(tmp);
    while (len > 1 && tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
        --len;
    }

    for (p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (tmp[0] != '\0') {
                rc = mkdir_one(tmp, where, err);
                if (rc != YVEX_OK) {
                    return rc;
                }
            }
            *p = '/';
        }
    }

    return mkdir_one(tmp, where, err);
}

int yvex_core_mkdir_parent(const char *path, const char *where, yvex_error *err)
{
    char parent[YVEX_PATH_CAP];
    char *slash;
    size_t length;

    where = where ? where : "core.mkdir_parent";
    if (!path || !path[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, "file path is required");
        return YVEX_ERR_INVALID_ARG;
    }
    length = strlen(path);
    if (length >= sizeof(parent)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, where, "path exceeds capacity");
        return YVEX_ERR_BOUNDS;
    }
    memcpy(parent, path, length + 1u);
    slash = strrchr(parent, '/');
    if (!slash || slash == parent) {
        return YVEX_OK;
    }
    *slash = '\0';
    return mkdir_p(parent, where, err);
}

/*
 * Format the current UTC instant for bounded operator metadata.
 *
 * Diagnostic time never contributes to semantic identity.
 */
void yvex_core_timestamp_utc(char *out, size_t capacity)
{
    time_t now;
    struct tm tm_utc;

    if (!out || capacity == 0u) return;
    out[0] = '\0';
    now = time(NULL);
    if (now == (time_t)-1 || !gmtime_r(&now, &tm_utc)) {
        snprintf(out, capacity, "unknown");
        return;
    }
    if (strftime(out, capacity, "%Y-%m-%dT%H:%M:%SZ", &tm_utc) == 0)
        snprintf(out, capacity, "unknown");
}

int yvex_core_file_read_text_prefix(const char *path, char *buffer, size_t capacity)
{
    FILE *file;
    size_t count;

    if (!path || !buffer || capacity == 0u)
        return 0;
    buffer[0] = '\0';
    file = fopen(path, "rb");
    if (!file)
        return 0;
    count = fread(buffer, 1u, capacity - 1u, file);
    buffer[count] = '\0';
    fclose(file);
    return 1;
}

static int core_file_fail(yvex_core_file_result *result, yvex_core_file_stage stage,
                          int system_error, unsigned long long expected,
                          unsigned long long actual, int status, const char *reason,
                          yvex_error *err)
{
    if (result) {
        memset(result, 0, sizeof(*result));
        result->stage = stage;
        result->system_error = system_error;
        result->expected = expected;
        result->actual = actual;
    }
    yvex_error_set(err, (yvex_status)status, "core.file", reason);
    return status;
}

static int core_file_parent_open(const char *path, int *directory_fd,
                                 char name[CORE_FILE_NAME_CAP],
                                 yvex_core_file_result *result, yvex_error *err)
{
    char copy[YVEX_PATH_CAP];
    char *slash, *cursor, *next;
    int fd = -1;

    if (!path || !path[0] || strnlen(path, sizeof(copy)) >= sizeof(copy))
        return core_file_fail(result, YVEX_CORE_FILE_STAGE_ARGUMENT, EINVAL, 1ull, 0ull,
                              YVEX_ERR_INVALID_ARG, "file path is empty or exceeds capacity", err);
    yvex_core_text_copy(copy, sizeof(copy), path);
    slash = strrchr(copy, '/');
    if (!slash) {
        if (snprintf(name, CORE_FILE_NAME_CAP, "%s", copy) >= (int)CORE_FILE_NAME_CAP)
            goto unsafe;
        cursor = NULL;
        fd = open(".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    } else {
        if (!slash[1] || snprintf(name, CORE_FILE_NAME_CAP, "%s", slash + 1) >=
                             (int)CORE_FILE_NAME_CAP)
            goto unsafe;
        *slash = '\0';
        fd = open(path[0] == '/' ? "/" : ".",
                  O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        cursor = slash == copy ? NULL : copy + (path[0] == '/');
    }
    if (!strcmp(name, ".") || !strcmp(name, "..") || !name[0]) goto unsafe;
    while (fd >= 0 && cursor && *cursor) {
        int child;
        next = strchr(cursor, '/');
        if (next) *next = '\0';
        if (!cursor[0] || !strcmp(cursor, ".") || !strcmp(cursor, "..")) goto unsafe;
        child = openat(fd, cursor, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (child < 0) goto unsafe;
        (void)close(fd);
        fd = child;
        cursor = next ? next + 1 : NULL;
    }
    if (fd < 0) goto unsafe;
    *directory_fd = fd;
    return YVEX_OK;

unsafe:
    if (fd >= 0) (void)close(fd);
    return core_file_fail(result, YVEX_CORE_FILE_STAGE_PATH, errno, 1ull, 0ull,
                          YVEX_ERR_IO, "file path or parent directory is unsafe", err);
}

static int core_file_write_exact(int fd, const void *source, size_t count)
{
    const unsigned char *data = (const unsigned char *)source;
    size_t offset = 0u;

    while (offset < count) {
        ssize_t wrote = write(fd, data + offset, count - offset);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) return 0;
        offset += (size_t)wrote;
    }
    return 1;
}

static int core_file_read_exact(int fd, unsigned char *data, size_t count)
{
    size_t offset = 0u;

    while (offset < count) {
        ssize_t got = pread(fd, data + offset, count - offset, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 0;
        offset += (size_t)got;
    }
    return 1;
}

int yvex_core_file_read_descriptor_snapshot(
    int descriptor, size_t expected_count, unsigned char **data,
    yvex_core_file_result *result, yvex_error *err)
{
    struct stat before = {0}, after = {0};
    unsigned char *buffer = NULL;
    int rc;
    if (data) *data = NULL;
    if (result) memset(result, 0, sizeof(*result));
    if (!data || !result || descriptor < 0 || !expected_count)
        return core_file_fail(result, YVEX_CORE_FILE_STAGE_ARGUMENT, EINVAL, 1ull, 0ull,
                              YVEX_ERR_INVALID_ARG,
                              "descriptor snapshot arguments are incomplete", err);
    if (fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) || before.st_size < 0 ||
        (unsigned long long)before.st_size != expected_count)
        return core_file_fail(result, YVEX_CORE_FILE_STAGE_BOUNDS, EFBIG, expected_count,
                              before.st_size > 0 ? (unsigned long long)before.st_size : 0ull,
                              YVEX_ERR_BOUNDS,
                              "descriptor snapshot size is inconsistent", err);
    buffer = malloc(expected_count + 1u);
    if (!buffer)
        return core_file_fail(result, YVEX_CORE_FILE_STAGE_ALLOCATION, ENOMEM,
                              expected_count + 1ull, 0ull, YVEX_ERR_NOMEM,
                              "descriptor snapshot allocation failed", err);
    if (!core_file_read_exact(descriptor, buffer, expected_count))
        rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_READ, errno, expected_count, 0ull,
                            YVEX_ERR_IO, "descriptor snapshot read was incomplete", err);
    else if (fstat(descriptor, &after) != 0 || before.st_dev != after.st_dev ||
             before.st_ino != after.st_ino || before.st_size != after.st_size ||
             before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
             before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
             before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
             before.st_ctim.tv_nsec != after.st_ctim.tv_nsec)
        rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_DRIFT, errno, expected_count,
                            after.st_size > 0 ? (unsigned long long)after.st_size : 0ull,
                            YVEX_ERR_IO, "descriptor snapshot drifted during read", err);
    else {
        buffer[expected_count] = '\0';
        *data = buffer;
        result->actual = expected_count;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    free(buffer);
    return rc;
}

static void core_file_cleanup_note(yvex_core_file_result *result,
                                   yvex_core_file_cleanup_stage stage,
                                   int system_error)
{
    if (!result || result->cleanup_stage != YVEX_CORE_FILE_CLEANUP_NONE) return;
    result->cleanup_stage = stage;
    result->cleanup_system_error = system_error ? system_error : EIO;
}

static int core_file_unlink_owned(int directory_fd, const char *name)
{
    int rc;

    do {
        rc = unlinkat(directory_fd, name, 0);
    } while (rc != 0 && errno == EINTR);
    return rc == 0 || errno == ENOENT;
}

int yvex_core_file_publish_noreplace(
    const char *path, const void *data, size_t count,
    const yvex_core_file_faults *faults, yvex_core_file_validator validator,
    void *validator_context, yvex_core_file_result *result, yvex_error *err)
{
    char name[CORE_FILE_NAME_CAP], temporary[CORE_FILE_NAME_CAP];
    int directory_fd = -1, file_fd = -1, validation_fd = -1, destination_exists = 0;
    int temporary_exists = 0, cleanup_fault_consumed = 0, rc = YVEX_ERR_IO;
    unsigned int attempt;

    if (result) memset(result, 0, sizeof(*result));
    if (!result || !data || !count)
        return core_file_fail(result, YVEX_CORE_FILE_STAGE_ARGUMENT, EINVAL, 1ull, 0ull,
                              YVEX_ERR_INVALID_ARG, "file publication arguments are incomplete", err);
    rc = core_file_parent_open(path, &directory_fd, name, result, err);
    if (rc != YVEX_OK) return rc;
    for (attempt = 0u; attempt < 32u; ++attempt) {
        if (snprintf(temporary, sizeof(temporary), ".%s.%llu.%u.tmp", name,
                     (unsigned long long)getpid(), attempt) >= (int)sizeof(temporary)) {
            rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_BOUNDS, ENAMETOOLONG,
                                sizeof(temporary), strlen(name), YVEX_ERR_BOUNDS,
                                "temporary filename exceeds capacity", err);
            goto done;
        }
        file_fd = openat(directory_fd, temporary,
                         O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (file_fd >= 0 || errno != EEXIST) break;
    }
    if (file_fd < 0) {
        rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_CREATE, errno, 1ull, 0ull,
                            YVEX_ERR_IO, "temporary file could not be created", err);
        goto done;
    }
    temporary_exists = 1;
    if (!core_file_write_exact(file_fd, data, count)) {
        rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_WRITE, errno, count, 0ull,
                            YVEX_ERR_IO, "file write was incomplete", err);
        goto done;
    }
    if (fsync(file_fd) != 0) {
        rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_FILE_SYNC, errno, 1ull, 0ull,
                            YVEX_ERR_IO, "file synchronization failed", err);
        goto done;
    }
    {
        int close_failed = close(file_fd) != 0;
        int close_error = close_failed ? errno : EIO;
        file_fd = -1;
        if (close_failed || (faults && faults->inject_file_close_failure)) {
            rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_FILE_CLOSE, close_error, 1ull, 0ull,
                            YVEX_ERR_IO, "file close failed", err);
            goto done;
        }
    }
    if (validator) {
        int validation_rc;
        int close_failed;
        int close_error;

        validation_fd = openat(directory_fd, temporary, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        if (validation_fd < 0) {
            rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_VALIDATE, errno, count, 0ull,
                                YVEX_ERR_IO, "publication candidate could not be reopened", err);
            goto done;
        }
        validation_rc = validator(validation_fd, count, validator_context, err);
        close_failed = close(validation_fd) != 0;
        close_error = close_failed ? errno : EIO;
        validation_fd = -1;
        if (validation_rc != YVEX_OK) {
            memset(result, 0, sizeof(*result));
            result->stage = YVEX_CORE_FILE_STAGE_VALIDATE;
            result->expected = count;
            if (close_failed)
                core_file_cleanup_note(result, YVEX_CORE_FILE_CLEANUP_FILE_CLOSE,
                                       close_error);
            if (!yvex_error_is_set(err))
                yvex_error_set(err, (yvex_status)validation_rc, "core.file",
                               "publication candidate validation failed");
            rc = validation_rc;
            goto done;
        }
        if (close_failed) {
            rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_FILE_CLOSE, close_error,
                                1ull, 0ull, YVEX_ERR_IO,
                                "validation descriptor close failed", err);
            goto done;
        }
    }
    if (linkat(directory_fd, temporary, directory_fd, name, 0) != 0) {
        int saved = errno;
        rc = core_file_fail(result,
                            saved == EEXIST ? YVEX_CORE_FILE_STAGE_CONFLICT
                                            : YVEX_CORE_FILE_STAGE_PUBLISH,
                            saved, 1ull, 0ull,
                            saved == EEXIST ? YVEX_ERR_STATE : YVEX_ERR_IO,
                            saved == EEXIST ? "destination already exists"
                                            : "atomic file publication failed", err);
        goto done;
    }
    destination_exists = 1;
    if ((faults && faults->inject_temporary_unlink_failure) ||
        !core_file_unlink_owned(directory_fd, temporary)) {
        int saved = errno;
        if (faults && faults->inject_temporary_unlink_failure) saved = EIO;
        rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_TEMPORARY_UNLINK, saved, 1ull, 0ull,
                            YVEX_ERR_IO, "temporary publication name could not be removed", err);
        goto done;
    }
    temporary_exists = 0;
    if (fsync(directory_fd) != 0) {
        rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_DIRECTORY_SYNC, errno, 1ull, 0ull,
                            YVEX_ERR_IO, "directory publication synchronization failed", err);
        goto done;
    }
    memset(result, 0, sizeof(*result));
    result->actual = count;
    rc = YVEX_OK;
    yvex_error_clear(err);

done:
    if (validation_fd >= 0) {
        if (close(validation_fd) != 0)
            core_file_cleanup_note(result, YVEX_CORE_FILE_CLEANUP_FILE_CLOSE, errno);
        validation_fd = -1;
    }
    if (file_fd >= 0) {
        if (close(file_fd) != 0)
            core_file_cleanup_note(result, YVEX_CORE_FILE_CLEANUP_FILE_CLOSE, errno);
        file_fd = -1;
    }
    if (directory_fd >= 0) {
        if (rc != YVEX_OK && destination_exists) {
            int removed = core_file_unlink_owned(directory_fd, name);
            int saved = removed ? EIO : errno;
            destination_exists = 0;
            if (!removed || (faults && faults->inject_cleanup_unlink_failure &&
                             !cleanup_fault_consumed)) {
                core_file_cleanup_note(result,
                                       YVEX_CORE_FILE_CLEANUP_DESTINATION_UNLINK, saved);
                cleanup_fault_consumed = 1;
            }
        }
        if (rc != YVEX_OK && temporary_exists) {
            int removed = core_file_unlink_owned(directory_fd, temporary);
            int saved = removed ? EIO : errno;
            temporary_exists = 0;
            if (!removed || (faults && faults->inject_cleanup_unlink_failure &&
                             !cleanup_fault_consumed)) {
                core_file_cleanup_note(result,
                                       YVEX_CORE_FILE_CLEANUP_TEMPORARY_UNLINK, saved);
                cleanup_fault_consumed = 1;
            }
        }
        if (rc != YVEX_OK && fsync(directory_fd) != 0)
            core_file_cleanup_note(result, YVEX_CORE_FILE_CLEANUP_DIRECTORY_SYNC, errno);
        (void)close(directory_fd);
    }
    return rc;
}

/*
 * Read one exact stable regular-file snapshot through a no-symlink path.
 *
 * Core validates only file mechanics; domain parsing and identity remain upstream.
 */
int yvex_core_file_read_snapshot(const char *path, size_t maximum_bytes,
                                 unsigned char **data, size_t *count,
                                 yvex_core_file_result *result, yvex_error *err)
{
    char name[CORE_FILE_NAME_CAP];
    int directory_fd = -1, fd = -1, rc;
    struct stat before;
    unsigned char *buffer = NULL;

    if (result) memset(result, 0, sizeof(*result));
    if (data) *data = NULL;
    if (count) *count = 0u;
    if (!data || !count || !result || !maximum_bytes)
        return core_file_fail(result, YVEX_CORE_FILE_STAGE_ARGUMENT, EINVAL, 1ull, 0ull,
                              YVEX_ERR_INVALID_ARG, "file snapshot arguments are incomplete", err);
    rc = core_file_parent_open(path, &directory_fd, name, result, err);
    if (rc != YVEX_OK) return rc;
    fd = openat(directory_fd, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0 || fstat(fd, &before) != 0 || !S_ISREG(before.st_mode)) {
        rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_OPEN, errno, 1ull, 0ull,
                            YVEX_ERR_IO, "file could not be opened safely", err);
        goto done;
    }
    if (before.st_size <= 0 || (unsigned long long)before.st_size > maximum_bytes) {
        rc = core_file_fail(result, YVEX_CORE_FILE_STAGE_BOUNDS, EFBIG, maximum_bytes,
                            before.st_size > 0 ? (unsigned long long)before.st_size : 0ull,
                            YVEX_ERR_BOUNDS, "file size exceeds its bound", err);
        goto done;
    }
    rc = yvex_core_file_read_descriptor_snapshot(
        fd, (size_t)before.st_size, &buffer, result, err);
    if (rc != YVEX_OK) goto done;
    *data = buffer;
    *count = (size_t)before.st_size;
    buffer = NULL;

done:
    free(buffer);
    if (fd >= 0) (void)close(fd);
    if (directory_fd >= 0) (void)close(directory_fd);
    return rc;
}

static int operator_config_location(const yvex_paths *paths,
                                    int include_file,
                                    char *out,
                                    size_t cap,
                                    yvex_error *err)
{
    const char *suffix = include_file ? "/operator-paths.conf" : "";

    if (!paths || !out || cap == 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths", "paths and output are required");
        return YVEX_ERR_INVALID_ARG;
    }

    if (paths->project_dir[0] != '\0') {
        return path_format(out, cap, err, "operator_paths", "%s%s", paths->project_dir, suffix);
    }
    return path_format(out, cap, err, "operator_paths", ".yvex%s", suffix);
}

static int path_has_newline(const char *value)
{
    return value && (strchr(value, '\n') || strchr(value, '\r'));
}

static int operator_normalize_root(const char *value,
                                        char *out,
                                        size_t cap,
                                        yvex_error *err,
                                        const char *where)
{
    const char *home;
    char cwd[YVEX_PATH_CAP];

    if (!value || value[0] == '\0') {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, "models root is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (path_has_newline(value)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, "models root must not contain a newline");
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(value, "/") == 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, "models root must not be filesystem root");
        return YVEX_ERR_INVALID_ARG;
    }

    if (value[0] == '/') {
        return path_format(out, cap, err, where, "%s", value);
    }

    if (value[0] == '~' && value[1] == '/') {
        home = getenv("HOME");
        if (!home || home[0] == '\0') {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, "HOME is required to expand ~/ paths");
            return YVEX_ERR_INVALID_ARG;
        }
        return path_format(out, cap, err, where, "%s/%s", home, value + 2);
    }

    if (!getcwd(cwd, sizeof(cwd))) {
        yvex_error_setf(err, YVEX_ERR_IO, where, "getcwd failed: %s", strerror(errno));
        return YVEX_ERR_IO;
    }
    return path_format(out, cap, err, where, "%s/%s", cwd, value);
}

static int operator_fill(yvex_operator_paths *out,
                              const char *source,
                              const char *models_root,
                              const char *config_path,
                              yvex_error *err)
{
    int rc;

    if (!out || !source || !models_root || !config_path) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths", "operator path fields are required");
        return YVEX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));
    rc = path_format(out->models_root_source, sizeof(out->models_root_source), err,
                          "operator_paths", "%s", source);
    if (rc != YVEX_OK) return rc;
    rc = path_fields_project(out, operator_root_fields,
                             sizeof(operator_root_fields) / sizeof(operator_root_fields[0]),
                             models_root, "operator_paths", err);
    if (rc != YVEX_OK) return rc;
    return path_format(out->config_path, sizeof(out->config_path), err,
                            "operator_paths", "%s", config_path);
}

static int operator_read_config(const yvex_paths *paths,
                                     char *out_root,
                                     size_t cap,
                                     int *out_found,
                                     char *out_config_path,
                                     size_t config_cap,
                                     yvex_error *err)
{
    FILE *fp;
    char line[YVEX_PATH_CAP + 32];
    char *value;
    size_t len;
    int rc;

    if (!out_root || !out_found || !out_config_path) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths", "config outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }

    *out_found = 0;
    out_root[0] = '\0';
    rc = operator_config_location(paths, 1, out_config_path, config_cap, err);
    if (rc != YVEX_OK) {
        return rc;
    }

    fp = fopen(out_config_path, "r");
    if (!fp) {
        if (errno == ENOENT) {
            return YVEX_OK;
        }
        yvex_error_setf(err, YVEX_ERR_IO, "operator_paths", "cannot read %s: %s",
                        out_config_path, strerror(errno));
        return YVEX_ERR_IO;
    }

    if (!fgets(line, sizeof(line), fp)) {
        if (ferror(fp)) {
            yvex_error_setf(err, YVEX_ERR_IO, "operator_paths", "cannot read %s: %s",
                            out_config_path, strerror(errno));
            fclose(fp);
            return YVEX_ERR_IO;
        }
        fclose(fp);
        return YVEX_OK;
    }
    fclose(fp);

    value = line;
    if (strncmp(value, "models_root=", 12) != 0) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "operator_paths", "invalid operator path config: %s", out_config_path);
        return YVEX_ERR_FORMAT;
    }
    value += 12;
    len = strlen(value);
    while (len > 0 && (value[len - 1] == '\n' || value[len - 1] == '\r')) {
        value[--len] = '\0';
    }
    if (value[0] == '\0' || path_has_newline(value)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "operator_paths", "configured models root is invalid");
        return YVEX_ERR_FORMAT;
    }

    rc = path_format(out_root, cap, err, "operator_paths", "%s", value);
    if (rc != YVEX_OK) {
        return rc;
    }
    *out_found = 1;
    return YVEX_OK;
}

int yvex_operator_paths_resolve(const yvex_paths *paths,
                                const char *explicit_models_root,
                                yvex_operator_paths *out,
                                yvex_error *err)
{
    char config_path[YVEX_PATH_CAP];
    char configured[YVEX_PATH_CAP];
    char resolved[YVEX_PATH_CAP];
    char builtin[YVEX_PATH_CAP];
    const char *env;
    const char *home;
    int found;
    int rc;

    if (!paths || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths", "paths and output are required");
        return YVEX_ERR_INVALID_ARG;
    }

    rc = operator_config_location(paths, 1, config_path, sizeof(config_path), err);
    if (rc != YVEX_OK) {
        return rc;
    }

    if (explicit_models_root) {
        rc = operator_normalize_root(explicit_models_root, resolved, sizeof(resolved), err, "operator_paths");
        if (rc != YVEX_OK) {
            return rc;
        }
        return operator_fill(out, "explicit", resolved, config_path, err);
    }

    rc = operator_read_config(paths, configured, sizeof(configured), &found,
                                   config_path, sizeof(config_path), err);
    if (rc != YVEX_OK) {
        return rc;
    }
    if (found) {
        return operator_fill(out, "configured", configured, config_path, err);
    }

    env = env_or_null("YVEX_MODELS_ROOT");
    if (env) {
        rc = operator_normalize_root(env, resolved, sizeof(resolved), err, "operator_paths");
        if (rc != YVEX_OK) {
            return rc;
        }
        return operator_fill(out, "environment", resolved, config_path, err);
    }

    home = getenv("HOME");
    if (home && home[0] != '\0') {
        rc = path_format(builtin, sizeof(builtin), err, "operator_paths", "%s/lab/models", home);
    } else {
        rc = operator_normalize_root("./models", builtin, sizeof(builtin), err, "operator_paths");
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    return operator_fill(out, "builtin", builtin, config_path, err);
}

int yvex_operator_paths_configure(const yvex_paths *paths,
                                  const char *models_root,
                                  int create_dirs,
                                  yvex_operator_paths *out,
                                  yvex_error *err)
{
    char config_dir[YVEX_PATH_CAP];
    FILE *fp;
    int rc;

    rc = yvex_operator_paths_resolve(paths, models_root, out, err);
    if (rc != YVEX_OK) {
        return rc;
    }

    rc = operator_config_location(paths, 0, config_dir, sizeof(config_dir), err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = mkdir_p(config_dir, "yvex_run_dir_create", err);
    if (rc != YVEX_OK) {
        return rc;
    }

    fp = fopen(out->config_path, "w");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "operator_paths", "cannot write %s: %s",
                        out->config_path, strerror(errno));
        return YVEX_ERR_IO;
    }
    if (fprintf(fp, "models_root=%s\n", out->models_root) < 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "operator_paths", "cannot write %s: %s",
                        out->config_path, strerror(errno));
        fclose(fp);
        return YVEX_ERR_IO;
    }
    if (fclose(fp) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "operator_paths", "cannot close %s: %s",
                        out->config_path, strerror(errno));
        return YVEX_ERR_IO;
    }

    if (create_dirs) {
        rc = yvex_operator_paths_create(out, err);
        if (rc != YVEX_OK) {
            return rc;
        }
    }

    return YVEX_OK;
}

int yvex_operator_paths_reset(const yvex_paths *paths,
                              int *out_removed,
                              yvex_operator_paths *out,
                              yvex_error *err)
{
    char config_path[YVEX_PATH_CAP];
    int rc;

    if (!out_removed) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths", "removed output is required");
        return YVEX_ERR_INVALID_ARG;
    }

    rc = operator_config_location(paths, 1, config_path, sizeof(config_path), err);
    if (rc != YVEX_OK) {
        return rc;
    }

    if (unlink(config_path) == 0) {
        *out_removed = 1;
    } else if (errno == ENOENT) {
        *out_removed = 0;
    } else {
        yvex_error_setf(err, YVEX_ERR_IO, "operator_paths", "cannot remove %s: %s",
                        config_path, strerror(errno));
        return YVEX_ERR_IO;
    }

    return yvex_operator_paths_resolve(paths, NULL, out, err);
}

int yvex_operator_paths_create(const yvex_operator_paths *operator_paths, yvex_error *err)
{
    const char *const family_names[] = {"deepseek", "glm", "qwen", "gemma"};
    const char *roots[4];
    const char *dirs[22];
    char family_dirs[4][4][YVEX_PATH_CAP];
    size_t root, family, dir_count = 0u;
    int rc;

    if (!operator_paths || operator_paths->models_root[0] == '\0') {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "operator_paths", "operator paths are required");
        return YVEX_ERR_INVALID_ARG;
    }
    roots[0] = operator_paths->hf_root;
    roots[1] = operator_paths->gguf_root;
    roots[2] = operator_paths->reports_root;
    roots[3] = operator_paths->reference_root;
    dirs[dir_count++] = operator_paths->models_root;
    for (root = 0u; root < 4u; ++root) {
        dirs[dir_count++] = roots[root];
        for (family = 0u; family < 4u; ++family) {
            rc = path_format(family_dirs[root][family], YVEX_PATH_CAP, err,
                             "operator_paths", "%s/%s", roots[root],
                             family_names[family]);
            if (rc != YVEX_OK) return rc;
            dirs[dir_count++] = family_dirs[root][family];
        }
    }
    dirs[dir_count++] = operator_paths->registry_root;
    for (root = 0u; root < dir_count; ++root) {
        rc = mkdir_p(dirs[root], "yvex_run_dir_create", err);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

int yvex_run_id_make(char *out, unsigned long cap, yvex_error *err)
{
    time_t now;
    struct tm tm_buf;
    char stamp[32];
    int n;

    if (!out || cap == 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_run_id_make", "out and cap are required");
        return YVEX_ERR_INVALID_ARG;
    }

    now = time(NULL);
    if (now == (time_t)-1) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_run_id_make", "time() failed");
        return YVEX_ERR_STATE;
    }

    if (!localtime_r(&now, &tm_buf)) {
        yvex_error_set(err, YVEX_ERR_STATE, "yvex_run_id_make", "localtime_r() failed");
        return YVEX_ERR_STATE;
    }

    if (strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_buf) == 0) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_run_id_make", "timestamp does not fit run id buffer");
        return YVEX_ERR_BOUNDS;
    }

    n = snprintf(out, (size_t)cap, "run_%s_%lu", stamp, (unsigned long)getpid());
    if (n < 0 || (unsigned long)n >= cap) {
        out[cap - 1] = '\0';
        yvex_error_setf(err, YVEX_ERR_BOUNDS, "yvex_run_id_make", "run id exceeds capacity %lu", cap);
        return YVEX_ERR_BOUNDS;
    }

    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_run_dir_prepare(yvex_run_dir *out, const yvex_paths *paths, const char *run_id, yvex_error *err)
{
    char generated[YVEX_RUN_ID_CAP];
    char run_root[YVEX_PATH_CAP];
    const char *resolved_run_id;
    const char *env_run_dir;
    int rc;

    if (!out || !paths) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_run_dir_prepare", "out and paths are required");
        return YVEX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    if (run_id && run_id[0] != '\0') {
        resolved_run_id = run_id;
    } else {
        rc = yvex_run_id_make(generated, sizeof(generated), err);
        if (rc != YVEX_OK) {
            return rc;
        }
        resolved_run_id = generated;
    }

    rc = path_format(out->run_id, sizeof(out->run_id), err, "yvex_run_dir_prepare", "%s", resolved_run_id);
    if (rc != YVEX_OK) {
        return rc;
    }

    env_run_dir = env_or_null("YVEX_RUN_DIR");
    if (env_run_dir) {
        rc = path_format(run_root, sizeof(run_root), err, "yvex_run_dir_prepare", "%s", env_run_dir);
    } else if (paths->project_dir[0] != '\0') {
        rc = path_format(run_root, sizeof(run_root), err, "yvex_run_dir_prepare", "%s/runs", paths->project_dir);
    } else {
        rc = path_format(run_root, sizeof(run_root), err, "yvex_run_dir_prepare", "%s/runs", paths->state_dir);
    }
    if (rc != YVEX_OK) {
        return rc;
    }

    rc = path_format(out->root, sizeof(out->root), err, "yvex_run_dir_prepare", "%s/%s", run_root, out->run_id);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = path_fields_project(out, run_file_fields,
                             sizeof(run_file_fields) / sizeof(run_file_fields[0]),
                             out->root, "yvex_run_dir_prepare", err);
    if (rc != YVEX_OK) return rc;

    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_run_dir_create(const yvex_run_dir *run, yvex_error *err)
{
    int rc;

    if (!run || run->root[0] == '\0') {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_run_dir_create", "run root is required");
        return YVEX_ERR_INVALID_ARG;
    }

    rc = mkdir_p(run->root, "yvex_run_dir_create", err);
    if (rc != YVEX_OK) {
        return rc;
    }

    yvex_error_clear(err);
    return YVEX_OK;
}
