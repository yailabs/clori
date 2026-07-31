/*
 * Own immutable artifact snapshots and bounded byte access.
 *
 * Every read remains bound to one revalidated regular-file snapshot. Byte access does not
 * establish artifact completeness or runtime support.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <yvex/artifact.h>
#include <yvex/internal/artifact.h>

#ifdef __linux__
#include <linux/openat2.h>
#endif

struct yvex_artifact {
    char path[YVEX_ARTIFACT_PATH_CAP];
    unsigned long long size;
    yvex_artifact_snapshot snapshot;
    int fd;
    unsigned char *mapping;
    size_t mapping_len;
};

static void snapshot_from_stat(const struct stat *st, yvex_artifact_snapshot *out) {
    memset(out, 0, sizeof(*out));
    out->device = (unsigned long long)st->st_dev;
    out->inode = (unsigned long long)st->st_ino;
    out->size = (unsigned long long)st->st_size;
    out->mtime_seconds = (long long)st->st_mtim.tv_sec;
    out->mtime_nanoseconds = (long long)st->st_mtim.tv_nsec;
    out->ctime_seconds = (long long)st->st_ctim.tv_sec;
    out->ctime_nanoseconds = (long long)st->st_ctim.tv_nsec;
}

static int artifact_path_open(const char *path) {
#if defined(__linux__) && defined(SYS_openat2)
    struct open_how how;

    memset(&how, 0, sizeof(how));
    how.flags = O_RDONLY | O_CLOEXEC;
    how.resolve = RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
    return (int)syscall(SYS_openat2, AT_FDCWD, path, &how, sizeof(how));
#else
    (void)path;
    errno = ENOTSUP;
    return -1;
#endif
}

int yvex_artifact_open(yvex_artifact **out, const yvex_artifact_options *options, yvex_error *err) {
    struct stat st;
    yvex_artifact *artifact;
    int path_length;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_open", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;

    if (!options || !options->path || options->path[0] == '\0') {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_open", "options.path is required");
        return YVEX_ERR_INVALID_ARG;
    }

    artifact = (yvex_artifact *)calloc(1, sizeof(*artifact));
    if (!artifact) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_artifact_open", "failed to allocate artifact");
        return YVEX_ERR_NOMEM;
    }

    artifact->fd = -1;
    path_length = snprintf(artifact->path, sizeof(artifact->path), "%s", options->path);
    if (path_length < 0 || path_length >= (int)sizeof(artifact->path)) {
        yvex_error_set(
            err, YVEX_ERR_BOUNDS, "yvex_artifact_open", "artifact path exceeds capacity");
        free(artifact);
        return YVEX_ERR_BOUNDS;
    }

    if (!options->readonly) {
        yvex_error_set(err,
                       YVEX_ERR_UNSUPPORTED,
                       "yvex_artifact_open",
                       "artifact opens require readonly mode");
        free(artifact);
        return YVEX_ERR_UNSUPPORTED;
    }

    artifact->fd = artifact_path_open(options->path);
    if (artifact->fd < 0) {
        yvex_error_setf(err,
                        YVEX_ERR_IO,
                        "yvex_artifact_open",
                        "failed to open %s: %s",
                        options->path,
                        strerror(errno));
        free(artifact);
        return YVEX_ERR_IO;
    }

    if (fstat(artifact->fd, &st) != 0) {
        yvex_error_setf(err,
                        YVEX_ERR_IO,
                        "yvex_artifact_open",
                        "failed to stat %s: %s",
                        options->path,
                        strerror(errno));
        yvex_artifact_close(artifact);
        return YVEX_ERR_IO;
    }
    if (!S_ISREG(st.st_mode) || st.st_size < 0) {
        yvex_error_set(
            err, YVEX_ERR_FORMAT, "yvex_artifact_open", "artifact path must name a regular file");
        yvex_artifact_close(artifact);
        return YVEX_ERR_FORMAT;
    }
    artifact->size = (unsigned long long)st.st_size;
    snapshot_from_stat(&st, &artifact->snapshot);

    if (options->map && artifact->size > 0ull) {
        if (artifact->size > (unsigned long long)SIZE_MAX) {
            yvex_error_set(err,
                           YVEX_ERR_BOUNDS,
                           "yvex_artifact_open",
                           "artifact is too large for this address space mapping");
            yvex_artifact_close(artifact);
            return YVEX_ERR_BOUNDS;
        }
        artifact->mapping_len = (size_t)artifact->size;
        artifact->mapping = (unsigned char *)mmap(
            NULL, artifact->mapping_len, PROT_READ, MAP_PRIVATE, artifact->fd, 0);
        if (artifact->mapping == MAP_FAILED) {
            artifact->mapping = NULL;
            artifact->mapping_len = 0u;
            yvex_error_setf(err,
                            YVEX_ERR_IO,
                            "yvex_artifact_open",
                            "failed to map %s: %s",
                            options->path,
                            strerror(errno));
            yvex_artifact_close(artifact);
            return YVEX_ERR_IO;
        }
    }
    *out = artifact;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Release resources owned by one artifact snapshot object and clear its observable state.
 *
 * Releases only resources owned by artifact byte access; cleanup remains deterministic.
 */
void yvex_artifact_close(yvex_artifact *artifact) {
    if (!artifact) {
        return;
    }

    if (artifact->mapping) {
        (void)munmap(artifact->mapping, artifact->mapping_len);
    }
    if (artifact->fd >= 0) {
        (void)close(artifact->fd);
    }
    free(artifact);
}

const char *yvex_artifact_path(const yvex_artifact *artifact) {
    if (!artifact) {
        return "";
    }
    return artifact->path;
}

unsigned long long yvex_artifact_size(const yvex_artifact *artifact) {
    if (!artifact) {
        return 0;
    }
    return artifact->size;
}

/*
 * Release cached pages for one verified range on the retained artifact handle.
 *
 * Cache residency changes neither artifact bytes nor snapshot identity.
 */
int yvex_artifact_cache_release(const yvex_artifact *artifact,
                                unsigned long long offset,
                                unsigned long long byte_count,
                                yvex_error *err)
{
    int advice_rc;

    if (!artifact || artifact->fd < 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "artifact.cache-release",
                       "an open artifact is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (offset > (unsigned long long)INT64_MAX ||
        byte_count > (unsigned long long)INT64_MAX ||
        yvex_range_check(artifact->size, offset, byte_count, err) != YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "artifact.cache-release",
                       "cache-release range exceeds the artifact or platform offset contract");
        return YVEX_ERR_BOUNDS;
    }
    advice_rc = posix_fadvise(artifact->fd, (off_t)offset, (off_t)byte_count,
                              POSIX_FADV_DONTNEED);
    if (advice_rc != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "artifact.cache-release",
                        "kernel cache release failed: %s", strerror(advice_rc));
        return YVEX_ERR_IO;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_artifact_is_mapped(const yvex_artifact *artifact) {
    return artifact && artifact->mapping ? 1 : 0;
}

const unsigned char *yvex_artifact_data(const yvex_artifact *artifact) {
    if (!artifact) {
        return NULL;
    }
    return artifact->mapping;
}

int yvex_artifact_read_at(const yvex_artifact *artifact,
                          unsigned long long offset,
                          void *dst,
                          size_t len,
                          yvex_error *err) {
    unsigned char *out = (unsigned char *)dst;
    size_t done = 0u;

    if (!artifact || (!dst && len != 0u)) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "yvex_artifact_read_at",
                       "artifact and destination are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (yvex_range_check(artifact->size, offset, (unsigned long long)len, err) != YVEX_OK) {
        return YVEX_ERR_BOUNDS;
    }

    while (done < len) {
        unsigned long long current = offset + (unsigned long long)done;
        ssize_t n;
        if (current > (unsigned long long)INT64_MAX) {
            yvex_error_set(err,
                           YVEX_ERR_BOUNDS,
                           "yvex_artifact_read_at",
                           "artifact offset exceeds positioned IO range");
            return YVEX_ERR_BOUNDS;
        }
        n = pread(artifact->fd, out + done, len - done, (off_t)current);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            yvex_error_setf(err,
                            YVEX_ERR_IO,
                            "yvex_artifact_read_at",
                            "positioned read failed at offset %llu: %s",
                            current,
                            strerror(errno));
            return YVEX_ERR_IO;
        }
        if (n == 0) {
            yvex_error_setf(
                err, YVEX_ERR_IO, "yvex_artifact_read_at", "short read at offset %llu", current);
            return YVEX_ERR_IO;
        }
        done += (size_t)n;
    }

    yvex_error_clear(err);
    return YVEX_OK;
}

/* Copies open-time identity and performs no IO or allocation. */
int yvex_artifact_snapshot_get(const yvex_artifact *artifact,
                               yvex_artifact_snapshot *out,
                               yvex_error *err) {
    if (!artifact || !out) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "yvex_artifact_snapshot_get",
                       "artifact and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = artifact->snapshot;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_artifact_snapshot_validate(const yvex_artifact *artifact,
                                    yvex_artifact_snapshot *current,
                                    yvex_error *err) {
    struct stat open_st;
    struct stat path_st;
    yvex_artifact_snapshot open_snapshot;
    yvex_artifact_snapshot path_snapshot;
    int path_fd;

    if (!artifact) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "yvex_artifact_snapshot_validate", "artifact is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (fstat(artifact->fd, &open_st) != 0 || !S_ISREG(open_st.st_mode)) {
        yvex_error_set(err,
                       YVEX_ERR_IO,
                       "yvex_artifact_snapshot_validate",
                       "artifact-identity-drift: opened file cannot be restated");
        return YVEX_ERR_IO;
    }
    snapshot_from_stat(&open_st, &open_snapshot);
    if (current)
        *current = open_snapshot;
    if (!yvex_artifact_snapshot_equal(&artifact->snapshot, &open_snapshot)) {
        yvex_error_set(err,
                       YVEX_ERR_FORMAT,
                       "yvex_artifact_snapshot_validate",
                       "artifact-identity-drift: opened file changed after admission");
        return YVEX_ERR_FORMAT;
    }
    path_fd = artifact_path_open(artifact->path);
    if (path_fd < 0 || fstat(path_fd, &path_st) != 0 || !S_ISREG(path_st.st_mode)) {
        if (path_fd >= 0)
            (void)close(path_fd);
        yvex_error_set(err,
                       YVEX_ERR_FORMAT,
                       "yvex_artifact_snapshot_validate",
                       "artifact-identity-drift: artifact path is unavailable or unsafe");
        return YVEX_ERR_FORMAT;
    }
    (void)close(path_fd);
    snapshot_from_stat(&path_st, &path_snapshot);
    if (!yvex_artifact_snapshot_equal(&artifact->snapshot, &path_snapshot)) {
        yvex_error_set(err,
                       YVEX_ERR_FORMAT,
                       "yvex_artifact_snapshot_validate",
                       "artifact-identity-drift: artifact path identity changed");
        return YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_range_check(unsigned long long file_size,
                     unsigned long long offset,
                     unsigned long long len,
                     yvex_error *err) {
    if (offset > file_size) {
        yvex_error_setf(err,
                        YVEX_ERR_BOUNDS,
                        "yvex_range_check",
                        "offset %llu exceeds file size %llu",
                        offset,
                        file_size);
        return YVEX_ERR_BOUNDS;
    }

    if (len > file_size - offset) {
        yvex_error_setf(err,
                        YVEX_ERR_BOUNDS,
                        "yvex_range_check",
                        "range offset=%llu len=%llu exceeds file size %llu",
                        offset,
                        len,
                        file_size);
        return YVEX_ERR_BOUNDS;
    }

    yvex_error_clear(err);
    return YVEX_OK;
}
