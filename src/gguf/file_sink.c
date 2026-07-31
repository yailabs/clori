/*
 * Transactionally persist planned tensor bytes into one external artifact.
 *
 * Writes stay inside disjoint preplanned ranges; one terminal is monotonic; no incomplete/aborted
 * session publishes; existing files survive. This sink writes bytes but cannot declare artifact
 * completeness.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/gguf_writer.h>

#ifdef __linux__
#include <linux/openat2.h>
#endif

#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1u << 0)
#endif

typedef enum {
    FILE_TERMINAL_EMPTY = 0,
    FILE_TERMINAL_BEGINNING,
    FILE_TERMINAL_ACTIVE,
    FILE_TERMINAL_CHUNKING,
    FILE_TERMINAL_COMMITTING,
    FILE_TERMINAL_COMMITTED,
    FILE_TERMINAL_ABORTED
} file_terminal_state;

typedef struct {
    file_terminal_state state;
    unsigned long long delivered_bytes;
    unsigned long long chunks;
} file_terminal_record;

struct yvex_gguf_file_sink {
    pthread_mutex_t mutex;
    int mutex_initialized;
    const yvex_gguf_writer_plan *writer_plan;
    const yvex_quant_plan *quant_plan;
    yvex_quant_digest_sink *digest;
    yvex_quant_output_sink digest_adapter;
    file_terminal_record *records;
    unsigned long long terminal_count;
    int directory_fd;
    int file_fd;
    char *directory_path;
    char *destination_name;
    char *temporary_name;
    char *destination_path;
    char *temporary_path;
    yvex_gguf_file_sink_options options;
    yvex_gguf_file_sink_summary summary;
    int failed;
    int finalized;
    int published;
};

static void file_sink_poison(yvex_gguf_file_sink *sink) {
    if (!sink || !sink->mutex_initialized)
        return;
    pthread_mutex_lock(&sink->mutex);
    sink->failed = 1;
    pthread_mutex_unlock(&sink->mutex);
}

/*
 * Account immutable source ranges consumed by one terminal operation.
 *
 * Returns false for invalid binding, missing range, or overflow.
 */
static int file_terminal_source_account(const yvex_gguf_file_sink *sink,
                                        const yvex_quant_decision *decision,
                                        unsigned long long *range_count,
                                        unsigned long long *byte_count) {
    const yvex_transform_ir *ir;
    const yvex_transform_binding *binding;
    const yvex_transform_node *node;
    unsigned long long ranges = 0u;
    unsigned long long bytes = 0u;
    unsigned long long input;

    if (!sink || !decision || !range_count || !byte_count)
        return 0;
    ir = yvex_quant_plan_transform_ir(sink->quant_plan);
    binding = yvex_quant_plan_binding(sink->quant_plan);
    node = binding ? yvex_transform_binding_terminal_operation(binding, decision->terminal_ordinal)
                   : NULL;
    if (!ir || !binding || !node || !node->input_count)
        return 0;
    for (input = 0u; input < node->input_count; ++input) {
        const yvex_transform_value *value = yvex_transform_ir_node_input_at(ir, node, input);
        const yvex_source_payload_range *range =
            value && value->kind == YVEX_TRANSFORM_VALUE_SOURCE
                ? yvex_transform_binding_range_at(binding, value->source_index)
                : NULL;
        if (!range || ranges == ULLONG_MAX || bytes > ULLONG_MAX - range->byte_length)
            return 0;
        ranges++;
        bytes += range->byte_length;
    }
    *range_count = ranges;
    *byte_count = bytes;
    return 1;
}

static int file_sink_fail(yvex_gguf_file_failure *failure, yvex_gguf_file_code code,
                          int system_error, unsigned long long terminal,
                          unsigned long long expected, unsigned long long actual,
                          unsigned long long offset, yvex_error *err, yvex_status status,
                          const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->system_error = system_error;
        failure->terminal_ordinal = terminal;
        failure->expected = expected;
        failure->actual = actual;
        failure->file_offset = offset;
    }
    yvex_error_set(err, status, "gguf.file.sink", message);
    return status;
}

static int file_stat_core_matches(const yvex_gguf_file_sink_summary *summary,
                                  const struct stat *file_stat) {
    return summary && file_stat && S_ISREG(file_stat->st_mode) && file_stat->st_size >= 0 &&
           (unsigned long long)file_stat->st_dev == summary->file_device &&
           (unsigned long long)file_stat->st_ino == summary->file_inode &&
           (unsigned long long)file_stat->st_size == summary->file_size;
}

static void file_stat_capture_validated(yvex_gguf_file_sink_summary *summary,
                                        const struct stat *file_stat) {
    summary->validated_mtime_seconds = (long long)file_stat->st_mtim.tv_sec;
    summary->validated_mtime_nanoseconds = (long long)file_stat->st_mtim.tv_nsec;
    summary->validated_ctime_seconds = (long long)file_stat->st_ctim.tv_sec;
    summary->validated_ctime_nanoseconds = (long long)file_stat->st_ctim.tv_nsec;
}

static int file_stat_validated_matches(const yvex_gguf_file_sink_summary *summary,
                                       const struct stat *file_stat) {
    return file_stat_core_matches(summary, file_stat) &&
           (long long)file_stat->st_mtim.tv_sec == summary->validated_mtime_seconds &&
           (long long)file_stat->st_mtim.tv_nsec == summary->validated_mtime_nanoseconds &&
           (long long)file_stat->st_ctim.tv_sec == summary->validated_ctime_seconds &&
           (long long)file_stat->st_ctim.tv_nsec == summary->validated_ctime_nanoseconds;
}

static void file_stat_capture_published(yvex_gguf_file_sink_summary *summary,
                                        const struct stat *file_stat) {
    summary->published_mtime_seconds = (long long)file_stat->st_mtim.tv_sec;
    summary->published_mtime_nanoseconds = (long long)file_stat->st_mtim.tv_nsec;
    summary->published_ctime_seconds = (long long)file_stat->st_ctim.tv_sec;
    summary->published_ctime_nanoseconds = (long long)file_stat->st_ctim.tv_nsec;
}

static int file_stat_published_matches(const yvex_gguf_file_sink_summary *summary,
                                       const struct stat *file_stat) {
    return file_stat_core_matches(summary, file_stat) &&
           (long long)file_stat->st_mtim.tv_sec == summary->published_mtime_seconds &&
           (long long)file_stat->st_mtim.tv_nsec == summary->published_mtime_nanoseconds &&
           (long long)file_stat->st_ctim.tv_sec == summary->published_ctime_seconds &&
           (long long)file_stat->st_ctim.tv_nsec == summary->published_ctime_nanoseconds;
}

static char *file_string_copy(const char *text) {
    size_t length;
    char *copy;
    if (!text)
        return NULL;
    length = strlen(text);
    if (length == SIZE_MAX)
        return NULL;
    copy = (char *)malloc(length + 1u);
    if (copy)
        memcpy(copy, text, length + 1u);
    return copy;
}

static int file_destination_path_safe(const char *path) {
    const char *cursor;

    if (!path || !path[0])
        return 0;
    cursor = path;
    if (*cursor == '/')
        cursor++;
    while (*cursor) {
        const char *begin = cursor;
        size_t length;
        while (*cursor && *cursor != '/') {
            if (*cursor == '\\' || *cursor == '\n' || *cursor == '\r')
                return 0;
            cursor++;
        }
        length = (size_t)(cursor - begin);
        if (!length || (length == 1u && begin[0] == '.') ||
            (length == 2u && begin[0] == '.' && begin[1] == '.'))
            return 0;
        if (*cursor == '/')
            cursor++;
    }
    return cursor > path && cursor[-1] != '/';
}

/*
 * Split a safe destination into owned directory and basename fields.
 *
 * Allocates directory/name/full-path strings under sink ownership. Returns false for invalid
 * basename, overflow, or allocation failure.
 */
static int file_destination_split(yvex_gguf_file_sink *sink, const char *path) {
    const char *slash;
    size_t directory_length;
    if (!sink || !file_destination_path_safe(path) || strlen(path) >= YVEX_ARTIFACT_PATH_CAP)
        return 0;
    slash = strrchr(path, '/');
    if (!slash) {
        sink->directory_path = file_string_copy(".");
        sink->destination_name = file_string_copy(path);
    } else {
        directory_length = slash == path ? 1u : (size_t)(slash - path);
        sink->directory_path = (char *)malloc(directory_length + 1u);
        if (sink->directory_path) {
            memcpy(sink->directory_path, path, directory_length);
            sink->directory_path[directory_length] = '\0';
        }
        sink->destination_name = file_string_copy(slash + 1u);
    }
    if (!sink->directory_path || !sink->destination_name || !sink->destination_name[0] ||
        strcmp(sink->destination_name, ".") == 0 || strcmp(sink->destination_name, "..") == 0 ||
        strchr(sink->destination_name, '/') || strchr(sink->destination_name, '\\'))
        return 0;
    sink->destination_path = file_string_copy(path);
    return sink->destination_path != NULL;
}

static int file_temp_create(yvex_gguf_file_sink *sink) {
    unsigned int attempt;
    size_t cap = strlen(sink->destination_name) + 96u;
    sink->temporary_name = (char *)malloc(cap);
    if (!sink->temporary_name)
        return 0;
    for (attempt = 0u; attempt < 128u; ++attempt) {
        int written = snprintf(sink->temporary_name, cap, ".%s.yvex-tmp-%ld-%u",
                               sink->destination_name, (long)getpid(), attempt);
        if (written < 0 || (size_t)written >= cap)
            return 0;
        sink->file_fd = openat(sink->directory_fd, sink->temporary_name,
                               O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (sink->file_fd >= 0)
            break;
        if (errno != EEXIST)
            return 0;
    }
    if (sink->file_fd < 0)
        return 0;
    {
        size_t dir_length = strlen(sink->directory_path);
        size_t name_length = strlen(sink->temporary_name);
        int needs_slash = dir_length && sink->directory_path[dir_length - 1u] != '/';
        if (dir_length > SIZE_MAX - name_length - (size_t)needs_slash - 1u)
            return 0;
        sink->temporary_path = (char *)malloc(dir_length + name_length + (size_t)needs_slash + 1u);
        if (!sink->temporary_path)
            return 0;
        memcpy(sink->temporary_path, sink->directory_path, dir_length);
        if (needs_slash)
            sink->temporary_path[dir_length++] = '/';
        memcpy(sink->temporary_path + dir_length, sink->temporary_name, name_length + 1u);
    }
    return 1;
}

static int file_directory_open(const char *path) {
#if defined(__linux__) && defined(SYS_openat2)
    struct open_how how;
    int fd;

    memset(&how, 0, sizeof(how));
    how.flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
    how.resolve = RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
    fd = (int)syscall(SYS_openat2, AT_FDCWD, path, &how, sizeof(how));
    if (fd >= 0)
        return fd;
    if (errno != ENOSYS && errno != EINVAL)
        return -1;
#endif
    return open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
}

static int file_pwrite_exact(yvex_gguf_file_sink *sink, const unsigned char *bytes,
                             size_t byte_count, unsigned long long offset) {
    size_t delivered = 0u;
    if (!sink || (!bytes && byte_count) || offset > (unsigned long long)LLONG_MAX)
        return 0;
    while (delivered < byte_count) {
        size_t remaining = byte_count - delivered;
        size_t request = remaining > (size_t)SSIZE_MAX ? (size_t)SSIZE_MAX : remaining;
        ssize_t wrote;
        unsigned long long current;
        if (offset > ULLONG_MAX - delivered ||
            (current = offset + delivered) > (unsigned long long)LLONG_MAX)
            return 0;
        pthread_mutex_lock(&sink->mutex);
        sink->summary.write_calls++;
        if (sink->options.injected_write_failure_call &&
            sink->summary.write_calls == sink->options.injected_write_failure_call) {
            pthread_mutex_unlock(&sink->mutex);
            errno = EIO;
            return 0;
        }
        if (sink->options.injected_write_eintr_call &&
            sink->summary.write_calls == sink->options.injected_write_eintr_call) {
            pthread_mutex_unlock(&sink->mutex);
            errno = EINTR;
            continue;
        }
        pthread_mutex_unlock(&sink->mutex);
        if (sink->options.injected_write_max_bytes &&
            request > sink->options.injected_write_max_bytes)
            request = sink->options.injected_write_max_bytes;
        wrote = pwrite(sink->file_fd, bytes + delivered, request, (off_t)current);
        if (wrote < 0 && errno == EINTR)
            continue;
        if (wrote <= 0)
            return 0;
        delivered += (size_t)wrote;
    }
    pthread_mutex_lock(&sink->mutex);
    if (!yvex_core_u64_add(sink->summary.physical_write_bytes, byte_count,
                      &sink->summary.physical_write_bytes)) {
        sink->failed = 1;
        pthread_mutex_unlock(&sink->mutex);
        errno = EOVERFLOW;
        return 0;
    }
    pthread_mutex_unlock(&sink->mutex);
    return 1;
}

/*
 * Begin one terminal transaction in both file and digest sinks.
 *
 * Poisons session and returns refusal for identity or lifecycle mismatch.
 */
static int file_sink_begin(void *opaque, const yvex_quant_decision *decision) {
    yvex_gguf_file_sink *sink = (yvex_gguf_file_sink *)opaque;
    file_terminal_record *record;
    int refused = 0;
    if (!sink)
        return 1;
    if (!decision || decision->terminal_ordinal >= sink->terminal_count) {
        file_sink_poison(sink);
        return 1;
    }
    pthread_mutex_lock(&sink->mutex);
    record = &sink->records[decision->terminal_ordinal];
    if (sink->failed || sink->finalized || record->state != FILE_TERMINAL_EMPTY ||
        decision != yvex_quant_plan_decision_at(sink->quant_plan, decision->terminal_ordinal)) {
        refused = 1;
    } else {
        record->state = FILE_TERMINAL_BEGINNING;
    }
    if (refused)
        sink->failed = 1;
    pthread_mutex_unlock(&sink->mutex);
    if (!refused) {
        int digest_refused =
            sink->digest_adapter.begin_terminal(sink->digest_adapter.context, decision);
        pthread_mutex_lock(&sink->mutex);
        if (digest_refused || record->state != FILE_TERMINAL_BEGINNING) {
            if (record->state != FILE_TERMINAL_ABORTED) {
                record->state = FILE_TERMINAL_ABORTED;
                sink->summary.aborted_terminals++;
            }
            sink->failed = 1;
            refused = 1;
        } else {
            record->state = FILE_TERMINAL_ACTIVE;
        }
        pthread_mutex_unlock(&sink->mutex);
    }
    return refused;
}

/*
 * Deliver one monotonic terminal chunk to file and digest evidence.
 *
 * Poisons session for bounds, order, write, digest, or overflow refusal.
 */
static int file_sink_chunk(void *opaque, const yvex_quant_decision *decision,
                           unsigned long long output_offset, const unsigned char *bytes,
                           size_t byte_count) {
    yvex_gguf_file_sink *sink = (yvex_gguf_file_sink *)opaque;
    const yvex_gguf_writer_tensor *tensor;
    file_terminal_record *record;
    unsigned long long file_offset;
    int refused = 0;
    if (!sink)
        return 1;
    if (!decision || !bytes || !byte_count || decision->terminal_ordinal >= sink->terminal_count ||
        decision != yvex_quant_plan_decision_at(sink->quant_plan, decision->terminal_ordinal)) {
        file_sink_poison(sink);
        return 1;
    }
    tensor = yvex_gguf_writer_plan_tensor_at(sink->writer_plan, decision->terminal_ordinal);
    if (!tensor || output_offset > ULLONG_MAX - byte_count ||
        output_offset + byte_count > tensor->raw_bytes ||
        tensor->absolute_offset > ULLONG_MAX - output_offset) {
        file_sink_poison(sink);
        return 1;
    }
    file_offset = tensor->absolute_offset + output_offset;
    pthread_mutex_lock(&sink->mutex);
    record = &sink->records[decision->terminal_ordinal];
    if (sink->failed || sink->finalized || record->state != FILE_TERMINAL_ACTIVE ||
        output_offset != record->delivered_bytes ||
        record->delivered_bytes > ULLONG_MAX - byte_count) {
        refused = 1;
    } else {
        record->state = FILE_TERMINAL_CHUNKING;
    }
    if (refused)
        sink->failed = 1;
    pthread_mutex_unlock(&sink->mutex);
    if (refused || !file_pwrite_exact(sink, bytes, byte_count, file_offset)) {
        pthread_mutex_lock(&sink->mutex);
        sink->failed = 1;
        pthread_mutex_unlock(&sink->mutex);
        return 1;
    }
    if (sink->digest_adapter.deliver_chunk(sink->digest_adapter.context, decision, output_offset,
                                           bytes, byte_count)) {
        pthread_mutex_lock(&sink->mutex);
        sink->failed = 1;
        pthread_mutex_unlock(&sink->mutex);
        return 1;
    }
    pthread_mutex_lock(&sink->mutex);
    if (record->state != FILE_TERMINAL_CHUNKING ||
        !yvex_core_u64_add(record->delivered_bytes, byte_count, &record->delivered_bytes) ||
        record->chunks == ULLONG_MAX ||
        !yvex_core_u64_add(sink->summary.encoded_bytes_written, byte_count,
                      &sink->summary.encoded_bytes_written) ||
        sink->summary.output_chunks == ULLONG_MAX) {
        sink->failed = 1;
        pthread_mutex_unlock(&sink->mutex);
        return 1;
    }
    record->chunks++;
    record->state = FILE_TERMINAL_ACTIVE;
    sink->summary.output_chunks++;
    pthread_mutex_unlock(&sink->mutex);
    return 0;
}

static int file_sink_commit(void *opaque, const yvex_quant_decision *decision,
                            unsigned long long delivered_bytes) {
    yvex_gguf_file_sink *sink = (yvex_gguf_file_sink *)opaque;
    file_terminal_record *record;
    unsigned long long source_ranges;
    unsigned long long source_bytes;
    int refused = 0;
    if (!sink)
        return 1;
    if (!decision || decision->terminal_ordinal >= sink->terminal_count ||
        decision != yvex_quant_plan_decision_at(sink->quant_plan, decision->terminal_ordinal)) {
        file_sink_poison(sink);
        return 1;
    }
    if (!file_terminal_source_account(sink, decision, &source_ranges, &source_bytes)) {
        file_sink_poison(sink);
        return 1;
    }
    pthread_mutex_lock(&sink->mutex);
    record = &sink->records[decision->terminal_ordinal];
    if (sink->failed || sink->finalized || record->state != FILE_TERMINAL_ACTIVE ||
        record->delivered_bytes != delivered_bytes || delivered_bytes != decision->encoded_bytes) {
        refused = 1;
    } else {
        record->state = FILE_TERMINAL_COMMITTING;
    }
    pthread_mutex_unlock(&sink->mutex);
    if (!refused && sink->digest_adapter.commit_terminal(sink->digest_adapter.context, decision,
                                                         delivered_bytes))
        refused = 1;
    pthread_mutex_lock(&sink->mutex);
    if (refused || record->state != FILE_TERMINAL_COMMITTING ||
        !yvex_core_u64_add(sink->summary.source_ranges_committed, source_ranges,
                      &sink->summary.source_ranges_committed) ||
        !yvex_core_u64_add(sink->summary.source_bytes_committed, source_bytes,
                      &sink->summary.source_bytes_committed)) {
        sink->failed = 1;
        refused = 1;
    } else {
        record->state = FILE_TERMINAL_COMMITTED;
        sink->summary.committed_terminals++;
    }
    pthread_mutex_unlock(&sink->mutex);
    return refused;
}

static void file_sink_abort(void *opaque, const yvex_quant_decision *decision,
                            const yvex_quant_failure *failure, unsigned long long delivered_bytes) {
    yvex_gguf_file_sink *sink = (yvex_gguf_file_sink *)opaque;
    file_terminal_record *record;
    if (!sink)
        return;
    if (!decision || decision->terminal_ordinal >= sink->terminal_count ||
        decision != yvex_quant_plan_decision_at(sink->quant_plan, decision->terminal_ordinal)) {
        file_sink_poison(sink);
        return;
    }
    sink->digest_adapter.abort_terminal(sink->digest_adapter.context, decision, failure,
                                        delivered_bytes);
    pthread_mutex_lock(&sink->mutex);
    record = &sink->records[decision->terminal_ordinal];
    if (record->state != FILE_TERMINAL_ABORTED && record->state != FILE_TERMINAL_COMMITTED) {
        record->state = FILE_TERMINAL_ABORTED;
        sink->summary.aborted_terminals++;
    }
    sink->failed = 1;
    pthread_mutex_unlock(&sink->mutex);
}

void yvex_gguf_file_sink_options_default(yvex_gguf_file_sink_options *options) {
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->safety_margin_bytes = 1024ull * 1024ull * 1024ull;
}

static int file_sink_prepare_destination(yvex_gguf_file_sink **sink_address,
                                         const yvex_gguf_writer_plan_summary *writer,
                                         yvex_gguf_file_failure *failure, yvex_error *err) {
    yvex_gguf_file_sink *sink = *sink_address;
    struct stat destination_stat;
    struct statvfs filesystem;
    unsigned long long available;
    unsigned long long filesystem_bytes;
    unsigned long long required;

    if (!file_destination_split(sink, sink->options.destination_path)) {
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_UNSAFE_DESTINATION, 0, ULLONG_MAX, 1u, 0u, 0u,
                              err, YVEX_ERR_INVALID_ARG, "destination path or basename is unsafe");
    }
    sink->directory_fd = file_directory_open(sink->directory_path);
    if (sink->directory_fd < 0) {
        int saved = errno;
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_DIRECTORY_OPEN, saved, ULLONG_MAX, 1u, 0u, 0u,
                              err, YVEX_ERR_IO, "destination directory cannot be opened safely");
    }
    if (fstatat(sink->directory_fd, sink->destination_name, &destination_stat,
                AT_SYMLINK_NOFOLLOW) == 0 ||
        errno != ENOENT) {
        int saved = errno;
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_DESTINATION_EXISTS, saved, ULLONG_MAX, 0u, 1u,
                              0u, err, YVEX_ERR_STATE,
                              "destination already exists or cannot be admitted");
    }
    if (fstatvfs(sink->directory_fd, &filesystem) != 0 ||
        !yvex_core_u64_mul(filesystem.f_bavail, filesystem.f_frsize, &filesystem_bytes)) {
        int saved = errno;
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_INSUFFICIENT_SPACE, saved, ULLONG_MAX,
                              writer->final_file_bytes, 0u, 0u, err, YVEX_ERR_IO,
                              "filesystem capacity cannot be determined safely");
    }
    available = filesystem_bytes;
    if (writer->final_file_bytes > ULLONG_MAX - sink->options.safety_margin_bytes) {
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_INSUFFICIENT_SPACE, 0, ULLONG_MAX, ULLONG_MAX,
                              available, 0u, err, YVEX_ERR_BOUNDS,
                              "required safe filesystem capacity overflowed");
    }
    required = writer->final_file_bytes + sink->options.safety_margin_bytes;
    sink->summary.available_bytes = available;
    sink->summary.required_safe_bytes = required;
    if (available < required) {
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_INSUFFICIENT_SPACE, ENOSPC, ULLONG_MAX,
                              required, available, 0u, err, YVEX_ERR_BOUNDS,
                              "filesystem has insufficient safe capacity for artifact");
    }
    return YVEX_OK;
}

static int file_sink_prepare_temporary(yvex_gguf_file_sink **sink_address,
                                       const yvex_gguf_writer_plan_summary *writer,
                                       yvex_gguf_file_failure *failure, yvex_error *err) {
    yvex_gguf_file_sink *sink = *sink_address;
    struct stat file_stat;
    const unsigned char *prefix;
    size_t prefix_bytes = 0u;
    int fallocate_status;

    if (pthread_mutex_init(&sink->mutex, NULL) != 0) {
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_ALLOCATION, 0, ULLONG_MAX, 1u, 0u, 0u, err,
                              YVEX_ERR_NOMEM, "file sink mutex creation failed");
    }
    sink->mutex_initialized = 1;
    sink->records =
        (file_terminal_record *)calloc((size_t)sink->terminal_count, sizeof(*sink->records));
    if (sink->options.inject_temp_create_failure)
        errno = EACCES;
    if (!sink->records || sink->options.inject_temp_create_failure || !file_temp_create(sink) ||
        fstat(sink->file_fd, &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) {
        int saved = errno;
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_TEMP_CREATE, saved, ULLONG_MAX, 1u, 0u, 0u,
                              err, YVEX_ERR_IO,
                              "unique regular temporary artifact creation failed");
    }
    if (writer->final_file_bytes > (unsigned long long)LLONG_MAX) {
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_PREALLOCATE, EOVERFLOW, ULLONG_MAX, LLONG_MAX,
                              writer->final_file_bytes, 0u, err, YVEX_ERR_BOUNDS,
                              "planned file size is not representable by off_t");
    }
    fallocate_status = sink->options.inject_preallocate_failure
                           ? ENOSPC
                           : posix_fallocate(sink->file_fd, 0, (off_t)writer->final_file_bytes);
    if (fallocate_status != 0) {
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_PREALLOCATE, fallocate_status, ULLONG_MAX,
                              writer->final_file_bytes, 0u, 0u, err,
                              fallocate_status == ENOSPC ? YVEX_ERR_BOUNDS : YVEX_ERR_IO,
                              "complete temporary artifact preallocation failed");
    }
    if (fstat(sink->file_fd, &file_stat) != 0 || !S_ISREG(file_stat.st_mode) ||
        file_stat.st_size < 0 ||
        (unsigned long long)file_stat.st_size != writer->final_file_bytes) {
        int saved = errno;
        unsigned long long actual =
            file_stat.st_size < 0 ? 0u : (unsigned long long)file_stat.st_size;
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_PREALLOCATE, saved, ULLONG_MAX,
                              writer->final_file_bytes, actual, 0u, err, YVEX_ERR_IO,
                              "preallocated artifact size or regular-file identity diverged");
    }
    sink->summary.preallocated = 1;
    sink->summary.file_device = (unsigned long long)file_stat.st_dev;
    sink->summary.file_inode = (unsigned long long)file_stat.st_ino;
    sink->summary.file_size = (unsigned long long)file_stat.st_size;
    prefix = yvex_gguf_writer_plan_prefix(sink->writer_plan, &prefix_bytes);
    if (!prefix || prefix_bytes != writer->structural_bytes + writer->pre_data_padding_bytes ||
        !file_pwrite_exact(sink, prefix, prefix_bytes, 0u)) {
        int saved = errno;
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_WRITE, saved, ULLONG_MAX, prefix_bytes, 0u,
                              0u, err, YVEX_ERR_IO, "GGUF structural prefix exact write failed");
    }
    sink->summary.prefix_bytes_written = prefix_bytes;
    return YVEX_OK;
}

/*
 * Install digest fan-out and prove complete owned-memory accounting.
 *
 * Releases complete sink for digest allocation or accounting overflow.
 */
static int file_sink_prepare_digest(yvex_gguf_file_sink **sink_address,
                                    const yvex_quant_plan_summary *quant,
                                    yvex_gguf_file_failure *failure, yvex_error *err) {
    yvex_gguf_file_sink *sink = *sink_address;
    const char *owned_strings[5];
    unsigned long long records_bytes;
    unsigned long long own_bytes = sizeof(*sink);
    size_t digest_bytes;
    size_t string_index;
    int accounting_ok;

    if (yvex_quant_digest_sink_create(&sink->digest, sink->quant_plan,
                                      quant->required_payload_identity, NULL, err) != YVEX_OK) {
        unsigned long long terminal_count = sink->terminal_count;
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_ALLOCATION, 0, ULLONG_MAX, terminal_count, 0u,
                              0u, err, YVEX_ERR_NOMEM, "quant digest fan-out sink creation failed");
    }
    yvex_quant_digest_sink_adapter(sink->digest, &sink->digest_adapter);
    owned_strings[0] = sink->directory_path;
    owned_strings[1] = sink->destination_name;
    owned_strings[2] = sink->temporary_name;
    owned_strings[3] = sink->destination_path;
    owned_strings[4] = sink->temporary_path;
    digest_bytes = yvex_quant_digest_sink_owned_bytes(sink->digest);
    accounting_ok = yvex_core_u64_mul(sink->terminal_count, sizeof(*sink->records), &records_bytes) &&
                    yvex_core_u64_add(own_bytes, records_bytes, &own_bytes);
    for (string_index = 0u; accounting_ok && string_index < 5u; ++string_index) {
        size_t length = strlen(owned_strings[string_index]);
        accounting_ok = length < SIZE_MAX &&
                        yvex_core_u64_add(own_bytes, (unsigned long long)length + 1u, &own_bytes);
    }
    if (!accounting_ok || !digest_bytes || own_bytes > ULLONG_MAX - digest_bytes) {
        yvex_gguf_file_sink_release(sink_address);
        return file_sink_fail(failure, YVEX_GGUF_FILE_ALLOCATION, EOVERFLOW, ULLONG_MAX, ULLONG_MAX,
                              own_bytes, 0u, err, YVEX_ERR_BOUNDS,
                              "file and digest sink ownership accounting overflowed");
    }
    sink->summary.peak_owned_bytes = own_bytes + digest_bytes;
    if (sink->temporary_path)
        yvex_core_text_copy(sink->summary.temporary_path, sizeof(sink->summary.temporary_path), sink->temporary_path);
    return YVEX_OK;
}

int yvex_gguf_file_sink_create(yvex_gguf_file_sink **out, const yvex_gguf_writer_plan *writer_plan,
                               const yvex_quant_plan *quant_plan,
                               const yvex_gguf_file_sink_options *options,
                               yvex_gguf_file_failure *failure, yvex_error *err) {
    const yvex_gguf_writer_plan_summary *writer = yvex_gguf_writer_plan_summary_get(writer_plan);
    const yvex_quant_plan_summary *quant = yvex_quant_plan_summary_get(quant_plan);
    yvex_gguf_file_sink_options local;
    yvex_gguf_file_sink *sink;
    int rc;

    if (out)
        *out = NULL;
    if (!out || !writer || !quant || !writer->complete || !quant->complete ||
        writer->profile_identity[0] == '\0' ||
        quant->terminal_count > SIZE_MAX / sizeof(file_terminal_record))
        return file_sink_fail(failure, YVEX_GGUF_FILE_INVALID_ARGUMENT, 0, ULLONG_MAX, 1u, 0u, 0u,
                              err, YVEX_ERR_INVALID_ARG,
                              "sealed writer and quant plans are required");
    yvex_gguf_file_sink_options_default(&local);
    if (options)
        local = *options;
    if (!local.destination_path || !local.destination_path[0] ||
        strcmp(writer->profile_identity, quant->profile_identity) != 0 ||
        writer->tensor_payload_bytes != quant->encoded_bytes)
        return file_sink_fail(failure, YVEX_GGUF_FILE_INVALID_ARGUMENT, 0, ULLONG_MAX,
                              writer->tensor_payload_bytes, quant->encoded_bytes, 0u, err,
                              YVEX_ERR_INVALID_ARG,
                              "destination and matching writer/quant identities are required");
    sink = (yvex_gguf_file_sink *)calloc(1u, sizeof(*sink));
    if (!sink)
        return file_sink_fail(failure, YVEX_GGUF_FILE_ALLOCATION, 0, ULLONG_MAX, sizeof(*sink), 0u,
                              0u, err, YVEX_ERR_NOMEM, "file sink allocation failed");
    sink->directory_fd = -1;
    sink->file_fd = -1;
    sink->writer_plan = writer_plan;
    sink->quant_plan = quant_plan;
    sink->terminal_count = quant->terminal_count;
    sink->options = local;
    sink->summary.planned_file_bytes = writer->final_file_bytes;
    rc = file_sink_prepare_destination(&sink, writer, failure, err);
    if (rc == YVEX_OK)
        rc = file_sink_prepare_temporary(&sink, writer, failure, err);
    if (rc == YVEX_OK)
        rc = file_sink_prepare_digest(&sink, quant, failure, err);
    if (rc != YVEX_OK)
        return rc;
    *out = sink;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Project a file sink into the quant executor's transactional ABI.
 *
 * Installs synchronous begin/chunk/commit/abort callbacks. Adapter borrows sink lifetime and
 * retains no callback arguments.
 */
void yvex_gguf_file_sink_adapter(yvex_gguf_file_sink *sink, yvex_quant_output_sink *out) {
    if (!out)
        return;
    memset(out, 0, sizeof(*out));
    if (!sink)
        return;
    out->begin_terminal = file_sink_begin;
    out->deliver_chunk = file_sink_chunk;
    out->commit_terminal = file_sink_commit;
    out->abort_terminal = file_sink_abort;
    out->context = sink;
}

/*
 * Seal execution identity and flush an exact unpublished artifact.
 *
 * Writes zero padding, finalizes digest, fsyncs, and snapshots inode.
 */
int yvex_gguf_file_sink_finalize(yvex_gguf_file_sink *sink, yvex_gguf_file_sink_summary *out,
                                 yvex_gguf_file_failure *failure, yvex_error *err) {
    const yvex_gguf_writer_plan_summary *writer;
    yvex_quant_digest_summary digest;
    yvex_quant_failure quant_failure;
    unsigned char zeros[32] = {0};
    struct stat final_stat;
    unsigned long long ordinal;
    int rc;

    if (out)
        memset(out, 0, sizeof(*out));
    if (!sink || !out || sink->finalized)
        return file_sink_fail(failure, YVEX_GGUF_FILE_INVALID_ARGUMENT, 0, ULLONG_MAX, 1u, 0u, 0u,
                              err, YVEX_ERR_INVALID_ARG,
                              "active file sink and summary output are required");
    writer = yvex_gguf_writer_plan_summary_get(sink->writer_plan);
    pthread_mutex_lock(&sink->mutex);
    if (sink->failed || sink->summary.committed_terminals != sink->terminal_count ||
        sink->summary.aborted_terminals != 0u ||
        sink->summary.encoded_bytes_written != writer->tensor_payload_bytes) {
        unsigned long long committed = sink->summary.committed_terminals;
        sink->failed = 1;
        pthread_mutex_unlock(&sink->mutex);
        return file_sink_fail(failure, YVEX_GGUF_FILE_INCOMPLETE, 0, ULLONG_MAX,
                              sink->terminal_count, committed, 0u, err, YVEX_ERR_STATE,
                              "every terminal must commit exact bytes before file finalization");
    }
    pthread_mutex_unlock(&sink->mutex);
    for (ordinal = 0u; ordinal < sink->terminal_count; ++ordinal) {
        const yvex_gguf_writer_tensor *tensor =
            yvex_gguf_writer_plan_tensor_at(sink->writer_plan, ordinal);
        unsigned long long padding;
        if (!tensor || tensor->padded_bytes < tensor->raw_bytes)
            return file_sink_fail(failure, YVEX_GGUF_FILE_TERMINAL_PROTOCOL, 0, ordinal, 1u, 0u, 0u,
                                  err, YVEX_ERR_FORMAT,
                                  "writer tensor padding geometry is unavailable");
        padding = tensor->padded_bytes - tensor->raw_bytes;
        if (padding && (padding > sizeof(zeros) ||
                        !file_pwrite_exact(sink, zeros, (size_t)padding, tensor->absolute_end)))
            return file_sink_fail(failure, YVEX_GGUF_FILE_WRITE, errno, ordinal, padding, 0u,
                                  tensor ? tensor->absolute_end : 0u, err, YVEX_ERR_IO,
                                  "tensor zero-padding write failed");
    }
    memset(&digest, 0, sizeof(digest));
    rc = yvex_quant_digest_sink_finalize(sink->digest, &digest, &quant_failure, err);
    if (rc != YVEX_OK)
        return file_sink_fail(failure, YVEX_GGUF_FILE_EXECUTION_IDENTITY, 0,
                              quant_failure.terminal_ordinal, sink->terminal_count,
                              digest.committed_terminals, 0u, err, (yvex_status)rc,
                              "quant execution digest could not finalize");
    if (writer->required_execution_identity[0] &&
        strcmp(writer->required_execution_identity, digest.execution_identity) != 0)
        return file_sink_fail(failure, YVEX_GGUF_FILE_EXECUTION_IDENTITY, 0, ULLONG_MAX, 1u, 0u, 0u,
                              err, YVEX_ERR_FORMAT,
                              "observed quant execution identity differs from writer plan");
    memset(&final_stat, 0, sizeof(final_stat));
    if (sink->options.inject_fsync_failure || fsync(sink->file_fd) != 0)
        return file_sink_fail(failure, YVEX_GGUF_FILE_FLUSH,
                              sink->options.inject_fsync_failure ? EIO : errno, ULLONG_MAX, 1u, 0u,
                              0u, err, YVEX_ERR_IO, "temporary artifact data flush failed");
    if (fstat(sink->file_fd, &final_stat) != 0 ||
        !file_stat_core_matches(&sink->summary, &final_stat))
        return file_sink_fail(
            failure, YVEX_GGUF_FILE_SNAPSHOT_DRIFT, errno, ULLONG_MAX, sink->summary.file_size,
            final_stat.st_size < 0 ? 0u : (unsigned long long)final_stat.st_size, 0u, err,
            YVEX_ERR_STATE, "temporary artifact identity drifted during finalization");
    file_stat_capture_validated(&sink->summary, &final_stat);
    memcpy(sink->summary.execution_identity, digest.execution_identity,
           sizeof(sink->summary.execution_identity));
    memcpy(sink->summary.payload_byte_identity, digest.payload_byte_identity,
           sizeof(sink->summary.payload_byte_identity));
    sink->summary.finalized = 1;
    sink->finalized = 1;
    *out = sink->summary;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

static int file_rename_noreplace(yvex_gguf_file_sink *sink) {
#ifdef SYS_renameat2
    if (syscall(SYS_renameat2, sink->directory_fd, sink->temporary_name, sink->directory_fd,
                sink->destination_name, RENAME_NOREPLACE) == 0)
        return 1;
    if (errno != ENOSYS && errno != EINVAL)
        return 0;
#endif
    if (linkat(sink->directory_fd, sink->temporary_name, sink->directory_fd, sink->destination_name,
               0) != 0)
        return 0;
    if (unlinkat(sink->directory_fd, sink->temporary_name, 0) != 0) {
        int saved = errno;
        (void)unlinkat(sink->directory_fd, sink->destination_name, 0);
        errno = saved;
        return 0;
    }
    return 1;
}

/* Publish a finalized temp only after exact native validation. */
int yvex_gguf_file_sink_publish(yvex_gguf_file_sink *sink,
                                const yvex_gguf_roundtrip_summary *roundtrip,
                                yvex_gguf_file_sink_summary *out, yvex_gguf_file_failure *failure,
                                yvex_error *err) {
    struct stat validated_stat;
    struct stat published_stat;
    memset(&validated_stat, 0, sizeof(validated_stat));
    memset(&published_stat, 0, sizeof(published_stat));
    if (out)
        memset(out, 0, sizeof(*out));
    if (!sink || !out || !sink->finalized || sink->published)
        return file_sink_fail(failure, YVEX_GGUF_FILE_INVALID_ARGUMENT, 0, ULLONG_MAX, 1u, 0u, 0u,
                              err, YVEX_ERR_STATE,
                              "one finalized unpublished artifact is required");
    if (fstat(sink->file_fd, &validated_stat) != 0 ||
        !file_stat_validated_matches(&sink->summary, &validated_stat))
        return file_sink_fail(
            failure, YVEX_GGUF_FILE_SNAPSHOT_DRIFT, errno, ULLONG_MAX, sink->summary.file_size,
            validated_stat.st_size < 0 ? 0u : (unsigned long long)validated_stat.st_size, 0u, err,
            YVEX_ERR_STATE, "validated temporary artifact changed before publication");
    if (!roundtrip || !roundtrip->complete || !roundtrip->reader_accepted ||
        !roundtrip->layout_accepted || !roundtrip->payload_accepted ||
        !roundtrip->snapshot_stable || roundtrip->file_bytes != sink->summary.file_size ||
        roundtrip->bytes_hashed != sink->summary.file_size ||
        roundtrip->file_device != sink->summary.file_device ||
        roundtrip->file_inode != sink->summary.file_inode ||
        roundtrip->file_mtime_seconds != sink->summary.validated_mtime_seconds ||
        roundtrip->file_mtime_nanoseconds != sink->summary.validated_mtime_nanoseconds ||
        roundtrip->file_ctime_seconds != sink->summary.validated_ctime_seconds ||
        roundtrip->file_ctime_nanoseconds != sink->summary.validated_ctime_nanoseconds ||
        strlen(roundtrip->artifact_identity) != 64u ||
        strlen(roundtrip->payload_byte_identity) != 64u ||
        strcmp(roundtrip->payload_byte_identity, sink->summary.payload_byte_identity) != 0)
        return file_sink_fail(failure, YVEX_GGUF_FILE_VALIDATION_REQUIRED, 0, ULLONG_MAX,
                              sink->summary.file_size, roundtrip ? roundtrip->bytes_hashed : 0u, 0u,
                              err, YVEX_ERR_STATE,
                              "exact native roundtrip for this temporary inode is required");
    if (sink->options.inject_publish_failure || !file_rename_noreplace(sink))
        return file_sink_fail(failure, YVEX_GGUF_FILE_PUBLICATION,
                              sink->options.inject_publish_failure ? EIO : errno, ULLONG_MAX, 1u,
                              0u, 0u, err, YVEX_ERR_IO, "no-replace artifact publication failed");
    if (sink->options.inject_directory_fsync_failure || fsync(sink->directory_fd) != 0) {
        int saved = sink->options.inject_directory_fsync_failure ? EIO : errno;
        int removed = unlinkat(sink->directory_fd, sink->destination_name, 0) == 0;
        if (removed)
            (void)fsync(sink->directory_fd);
        sink->temporary_name[0] = '\0';
        return file_sink_fail(failure,
                              removed ? YVEX_GGUF_FILE_DIRECTORY_FLUSH : YVEX_GGUF_FILE_CLEANUP,
                              saved, ULLONG_MAX, 1u, 0u, 0u, err, YVEX_ERR_IO,
                              removed ? "publication was rolled back after directory flush failure"
                                      : "publication directory flush failed and rollback failed");
    }
    if (fstatat(sink->directory_fd, sink->destination_name, &published_stat, AT_SYMLINK_NOFOLLOW) !=
            0 ||
        !file_stat_core_matches(&sink->summary, &published_stat)) {
        int saved = errno;
        int removed = unlinkat(sink->directory_fd, sink->destination_name, 0) == 0;
        if (removed)
            (void)fsync(sink->directory_fd);
        sink->temporary_name[0] = '\0';
        return file_sink_fail(
            failure, removed ? YVEX_GGUF_FILE_PUBLICATION : YVEX_GGUF_FILE_CLEANUP, saved,
            ULLONG_MAX, sink->summary.file_size,
            published_stat.st_size < 0 ? 0u : (unsigned long long)published_stat.st_size, 0u, err,
            YVEX_ERR_STATE,
            removed ? "published path identity diverged and was rolled back"
                    : "published path identity diverged and cleanup failed");
    }
    sink->temporary_name[0] = '\0';
    sink->published = 1;
    sink->summary.published = 1;
    file_stat_capture_published(&sink->summary, &published_stat);
    yvex_core_text_copy(sink->summary.published_path, sizeof(sink->summary.published_path), sink->destination_path);
    *out = sink->summary;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Withdraw only this session's unchanged publication after gate refusal.
 *
 * Refuses cleanup when path identity drifted or durability fails.
 */
int yvex_gguf_file_sink_withdraw(yvex_gguf_file_sink *sink, yvex_gguf_file_failure *failure,
                                 yvex_error *err) {
    struct stat current;
    memset(&current, 0, sizeof(current));
    if (!sink || !sink->published || sink->directory_fd < 0 ||
        fstatat(sink->directory_fd, sink->destination_name, &current, AT_SYMLINK_NOFOLLOW) != 0 ||
        !file_stat_published_matches(&sink->summary, &current) ||
        unlinkat(sink->directory_fd, sink->destination_name, 0) != 0 ||
        fsync(sink->directory_fd) != 0)
        return file_sink_fail(failure, YVEX_GGUF_FILE_CLEANUP, errno, ULLONG_MAX,
                              sink ? sink->summary.file_size : 0u,
                              current.st_size < 0 ? 0u : (unsigned long long)current.st_size, 0u,
                              err, YVEX_ERR_IO, "owned published artifact withdrawal failed");
    sink->published = 0;
    sink->summary.published = 0;
    sink->summary.published_path[0] = '\0';
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Borrow the quant digest sink paired with file delivery.
 *
 * Digest lifetime remains owned by the file sink.
 */
yvex_quant_digest_sink *yvex_gguf_file_sink_digest(yvex_gguf_file_sink *sink) {
    return sink ? sink->digest : NULL;
}

const char *yvex_gguf_file_sink_temporary_path(const yvex_gguf_file_sink *sink) {
    return sink && !sink->published ? sink->temporary_path : NULL;
}

int yvex_gguf_file_sink_summary_get(yvex_gguf_file_sink *sink, yvex_gguf_file_sink_summary *out) {
    if (!sink || !out || !sink->mutex_initialized)
        return YVEX_ERR_INVALID_ARG;
    pthread_mutex_lock(&sink->mutex);
    *out = sink->summary;
    pthread_mutex_unlock(&sink->mutex);
    return YVEX_OK;
}

void yvex_gguf_file_sink_release(yvex_gguf_file_sink **sink_address) {
    yvex_gguf_file_sink *sink;
    if (!sink_address || !*sink_address)
        return;
    sink = *sink_address;
    *sink_address = NULL;
    if (sink->file_fd >= 0)
        close(sink->file_fd);
    if (!sink->published && sink->directory_fd >= 0 && sink->temporary_name &&
        sink->temporary_name[0])
        (void)unlinkat(sink->directory_fd, sink->temporary_name, 0);
    if (sink->directory_fd >= 0)
        close(sink->directory_fd);
    yvex_quant_digest_sink_release(&sink->digest);
    if (sink->mutex_initialized)
        pthread_mutex_destroy(&sink->mutex);
    free(sink->records);
    free(sink->directory_path);
    free(sink->destination_name);
    free(sink->temporary_name);
    free(sink->destination_path);
    free(sink->temporary_path);
    memset(sink, 0, sizeof(*sink));
    free(sink);
}
