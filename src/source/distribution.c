/* Own deterministic locator parsing and local source distribution mechanics. */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <yvex/internal/source_distribution.h>

#include <yvex/internal/core.h>
#include <yvex/internal/io.h>
#include <yvex/internal/source.h>
#include <yvex/internal/source_catalog.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#include <sys/types.h>
#include <unistd.h>

#define SOURCE_COPY_BUFFER_BYTES (1024u * 1024u)
#define SOURCE_RECORD_BYTES (YVEX_PATH_CAP * 3u + 2048u)

static int distribution_refuse(yvex_error *err, yvex_status status,
                               const char *where, const char *reason)
{
    yvex_error_set(err, status, where, reason);
    return status;
}

static int distribution_copy(char *out, size_t cap, const char *value,
                             yvex_error *err, const char *where)
{
    size_t length = value ? strlen(value) : 0u;
    if (!out || !cap || length >= cap)
        return distribution_refuse(err, YVEX_ERR_BOUNDS, where,
                                   "source locator field exceeds its bound");
    memcpy(out, value ? value : "", length + 1u);
    return YVEX_OK;
}

static int repository_valid(const char *text)
{
    const unsigned char *cursor = (const unsigned char *)text;
    unsigned int slash = 0u;
    if (!text || !text[0]) return 0;
    while (*cursor) {
        if (*cursor == '/') slash++;
        else if (!((*cursor >= 'a' && *cursor <= 'z') ||
                   (*cursor >= 'A' && *cursor <= 'Z') ||
                   (*cursor >= '0' && *cursor <= '9') || *cursor == '-' ||
                   *cursor == '_' || *cursor == '.'))
            return 0;
        cursor++;
    }
    return slash == 1u && text[0] != '/' && cursor[-1] != '/';
}

static int locator_huggingface(const char *body, yvex_source_locator *out,
                               yvex_error *err)
{
    char repository[YVEX_SOURCE_LOCATOR_REPOSITORY_CAP];
    const char *at = strrchr(body, '@');
    size_t extent = at ? (size_t)(at - body) : strlen(body);

    if (!extent || extent >= sizeof(repository) || (at && !at[1]))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.locator",
                                   "hf locator requires ORG/REPOSITORY[@REVISION]");
    memcpy(repository, body, extent);
    repository[extent] = '\0';
    if (!repository_valid(repository))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.locator",
                                   "hf locator repository is invalid");
    if (distribution_copy(out->repository, sizeof(out->repository), repository,
                          err, "source.locator") != YVEX_OK ||
        (at && distribution_copy(out->revision, sizeof(out->revision), at + 1,
                                 err, "source.locator") != YVEX_OK))
        return yvex_error_code(err);
    out->kind = YVEX_SOURCE_LOCATOR_HUGGINGFACE;
    out->revision_present = at != NULL;
    out->readable = 1;
    out->writable = 0;
    (void)distribution_copy(out->scheme, sizeof(out->scheme), "hf", err,
                            "source.locator");
    (void)distribution_copy(out->provider, sizeof(out->provider), "huggingface",
                            err, "source.locator");
    if (snprintf(out->canonical, sizeof(out->canonical), "hf://%s%s%s",
                 repository, at ? "@" : "", at ? at + 1 : "") >=
        (int)sizeof(out->canonical))
        return distribution_refuse(err, YVEX_ERR_BOUNDS, "source.locator",
                                   "canonical hf locator is too long");
    return YVEX_OK;
}

static int locator_local(const char *path, yvex_source_locator *out,
                         yvex_error *err)
{
    char resolved[YVEX_PATH_CAP];
    if (!path || !path[0])
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.locator",
                                   "local source path is empty");
    if (!realpath(path, resolved)) {
        yvex_error_setf(err, YVEX_ERR_IO, "source.locator",
                        "local source path is unavailable: %s", path);
        return YVEX_ERR_IO;
    }
    if (distribution_copy(out->path, sizeof(out->path), resolved, err,
                          "source.locator") != YVEX_OK)
        return yvex_error_code(err);
    out->kind = YVEX_SOURCE_LOCATOR_LOCAL_PATH;
    out->readable = 1;
    out->writable = 1;
    (void)distribution_copy(out->scheme, sizeof(out->scheme), "file", err,
                            "source.locator");
    if (snprintf(out->canonical, sizeof(out->canonical), "file://%s", resolved) >=
        (int)sizeof(out->canonical))
        return distribution_refuse(err, YVEX_ERR_BOUNDS, "source.locator",
                                   "canonical file locator is too long");
    return YVEX_OK;
}

int yvex_source_locator_parse(const char *text, yvex_source_locator *out,
                              yvex_error *err)
{
    const char *separator;
    size_t scheme_length;
    if (!text || !out)
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.locator",
                                   "source locator and output are required");
    memset(out, 0, sizeof(*out));
    if (!strncmp(text, "hf://", 5u)) return locator_huggingface(text + 5u, out, err);
    if (!strncmp(text, "file://", 7u)) {
        if (text[7] != '/')
            return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.locator",
                                       "file locator requires an absolute path");
        return locator_local(text + 7u, out, err);
    }
    separator = strstr(text, "://");
    if (!separator) return locator_local(text, out, err);
    scheme_length = (size_t)(separator - text);
    if (!scheme_length || scheme_length >= sizeof(out->scheme))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.locator",
                                   "source locator scheme is invalid");
    memcpy(out->scheme, text, scheme_length);
    out->scheme[scheme_length] = '\0';
    out->kind = !strcmp(out->scheme, "ssh") ? YVEX_SOURCE_LOCATOR_SSH
              : !strcmp(out->scheme, "oci") ? YVEX_SOURCE_LOCATOR_OCI
              : !strcmp(out->scheme, "yvex") ? YVEX_SOURCE_LOCATOR_YVEX
                                               : YVEX_SOURCE_LOCATOR_UNSUPPORTED;
    (void)distribution_copy(out->canonical, sizeof(out->canonical), text, err,
                            "source.locator");
    return YVEX_OK;
}

const char *yvex_source_locator_kind_name(yvex_source_locator_kind kind)
{
    switch (kind) {
    case YVEX_SOURCE_LOCATOR_LOCAL_PATH: return "file";
    case YVEX_SOURCE_LOCATOR_HUGGINGFACE: return "huggingface";
    case YVEX_SOURCE_LOCATOR_SSH: return "ssh";
    case YVEX_SOURCE_LOCATOR_OCI: return "oci";
    case YVEX_SOURCE_LOCATOR_YVEX: return "yvex";
    case YVEX_SOURCE_LOCATOR_UNSUPPORTED: return "unsupported";
    }
    return "unsupported";
}

static int hash_regular_file(const char *path, char digest_hex[65],
                             unsigned long long *bytes, yvex_error *err)
{
    unsigned char *buffer = NULL, digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    struct stat before, after;
    unsigned long long total = 0u;
    ssize_t count;
    int fd = -1, rc = YVEX_OK;

    fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0 || fstat(fd, &before) != 0 || !S_ISREG(before.st_mode)) {
        yvex_error_setf(err, YVEX_ERR_IO, "source.distribution.hash",
                        "cannot open regular source file: %s", path);
        rc = YVEX_ERR_IO;
        goto done;
    }
    buffer = malloc(SOURCE_COPY_BUFFER_BYTES);
    if (!buffer) {
        rc = distribution_refuse(err, YVEX_ERR_NOMEM, "source.distribution.hash",
                                 "source hash buffer allocation failed");
        goto done;
    }
    yvex_sha256_init(&hash);
    while ((count = read(fd, buffer, SOURCE_COPY_BUFFER_BYTES)) > 0) {
        if (!yvex_sha256_update(&hash, buffer, (size_t)count) ||
            ULLONG_MAX - total < (unsigned long long)count) {
            rc = distribution_refuse(err, YVEX_ERR_BOUNDS, "source.distribution.hash",
                                     "source file identity exceeds its bound");
            goto done;
        }
        total += (unsigned long long)count;
    }
    if (count < 0 || fstat(fd, &after) != 0 ||
        before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_size != after.st_size || before.st_mtime != after.st_mtime ||
        before.st_ctime != after.st_ctime) {
        rc = distribution_refuse(err, YVEX_ERR_IO, "source.distribution.hash",
                                 "source file changed while computing identity");
        goto done;
    }
    if (!yvex_sha256_final(&hash, digest)) {
        rc = distribution_refuse(err, YVEX_ERR_STATE, "source.distribution.hash",
                                 "source digest finalization failed");
        goto done;
    }
    yvex_sha256_hex(digest, digest_hex);
    if (bytes) *bytes = total;
done:
    free(buffer);
    if (fd >= 0 && close(fd) != 0 && rc == YVEX_OK)
        rc = distribution_refuse(err, YVEX_ERR_IO, "source.distribution.hash",
                                 "cannot close source file");
    return rc;
}

static const char *path_basename(const char *path)
{
    const char *slash = strrchr(path, '/');
    return slash && slash[1] ? slash + 1 : path;
}

static int ends_with(const char *text, const char *suffix)
{
    size_t left = strlen(text), right = strlen(suffix);
    return left >= right && !strcasecmp(text + left - right, suffix);
}

static void representation_name(char *out, size_t cap, const char *path)
{
    const char *base = path_basename(path);
    size_t length = strlen(base);
    if (ends_with(base, ".gguf") && length > 5u) length -= 5u;
    if (length >= cap) length = cap - 1u;
    memcpy(out, base, length);
    out[length] = '\0';
}

static int inspect_directory(const char *path, yvex_source_representation_fact *out,
                             yvex_error *err)
{
    yvex_source_manifest_file_list files;
    yvex_sha256 tree;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long gguf = 0u;
    size_t index;
    int rc;

    yvex_source_manifest_file_list_init(&files);
    rc = yvex_source_manifest_scan_files(path, 1, &files, err);
    if (rc != YVEX_OK) goto done;
    yvex_sha256_init(&tree);
    if (!yvex_sha256_update_text(&tree, "yvex.local-source.v1")) {
        rc = YVEX_ERR_STATE;
        goto done;
    }
    for (index = 0u; index < files.count; ++index) {
        char absolute[YVEX_PATH_CAP], file_digest[65];
        unsigned long long bytes = 0u;
        if (!yvex_source_path_join(absolute, sizeof(absolute), path,
                                   files.items[index].path)) {
            rc = distribution_refuse(err, YVEX_ERR_BOUNDS,
                                     "source.distribution.inspect",
                                     "source member path is too long");
            goto done;
        }
        rc = hash_regular_file(absolute, file_digest, &bytes, err);
        if (rc != YVEX_OK) goto done;
        if (bytes != files.items[index].size_bytes ||
            !yvex_sha256_update_text(&tree, files.items[index].path) ||
            !yvex_sha256_update_u64(&tree, bytes) ||
            !yvex_sha256_update_text(&tree, file_digest)) {
            rc = distribution_refuse(err, YVEX_ERR_STATE,
                                     "source.distribution.inspect",
                                     "source member identity changed");
            goto done;
        }
        if (ends_with(files.items[index].path, ".gguf")) gguf++;
    }
    if (!files.count || !yvex_sha256_final(&tree, digest)) {
        rc = distribution_refuse(err, YVEX_ERR_FORMAT,
                                 "source.distribution.inspect",
                                 "local source directory has no regular representation files");
        goto done;
    }
    yvex_sha256_hex(digest, out->digest);
    out->size_bytes = files.summary.total_size_bytes;
    out->file_count = files.summary.file_count;
    out->directory = 1;
    if (files.summary.safetensors_count && gguf)
        (void)distribution_copy(out->format, sizeof(out->format), "mixed", err,
                                "source.distribution.inspect");
    else if (files.summary.safetensors_count)
        (void)distribution_copy(out->format, sizeof(out->format), "safetensors", err,
                                "source.distribution.inspect");
    else if (gguf)
        (void)distribution_copy(out->format, sizeof(out->format), "gguf", err,
                                "source.distribution.inspect");
    else
        (void)distribution_copy(out->format, sizeof(out->format), "source", err,
                                "source.distribution.inspect");
done:
    yvex_source_manifest_file_list_free(&files);
    return rc;
}

int yvex_source_representation_inspect_local(
    const yvex_source_locator *locator, yvex_source_representation_fact *out,
    yvex_error *err)
{
    struct stat status;
    int rc;
    if (!locator || !out || locator->kind != YVEX_SOURCE_LOCATOR_LOCAL_PATH)
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "source.distribution.inspect",
                                   "a local source locator is required");
    memset(out, 0, sizeof(*out));
    if (lstat(locator->path, &status) != 0 || S_ISLNK(status.st_mode))
        return distribution_refuse(err, YVEX_ERR_IO, "source.distribution.inspect",
                                   "local source is unavailable or symbolic");
    if (distribution_copy(out->path, sizeof(out->path), locator->path, err,
                          "source.distribution.inspect") != YVEX_OK)
        return yvex_error_code(err);
    representation_name(out->name, sizeof(out->name), locator->path);
    if (S_ISDIR(status.st_mode)) rc = inspect_directory(locator->path, out, err);
    else if (S_ISREG(status.st_mode)) {
        rc = hash_regular_file(locator->path, out->digest, &out->size_bytes, err);
        if (rc == YVEX_OK) {
            out->file_count = 1u;
            (void)distribution_copy(out->format, sizeof(out->format),
                                    ends_with(locator->path, ".gguf") ? "gguf"
                                    : ends_with(locator->path, ".safetensors")
                                          ? "safetensors" : "file",
                                    err, "source.distribution.inspect");
        }
    } else {
        rc = distribution_refuse(err, YVEX_ERR_UNSUPPORTED,
                                 "source.distribution.inspect",
                                 "local source must be a regular file or directory");
    }
    return rc;
}

static int safe_name(char *out, size_t cap, const char *input)
{
    size_t index, count = 0u;
    if (!input || !input[0]) return 0;
    for (index = 0u; input[index] && count + 1u < cap; ++index) {
        unsigned char value = (unsigned char)input[index];
        if ((value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
            (value >= '0' && value <= '9') || value == '-' || value == '_' || value == '.')
            out[count++] = (char)value;
        else if (count && out[count - 1u] != '-') out[count++] = '-';
    }
    while (count && (out[count - 1u] == '-' || out[count - 1u] == '.')) count--;
    out[count] = '\0';
    return count != 0u;
}

static int remove_tree(const char *path)
{
    struct stat status;
    DIR *dir;
    struct dirent *entry;
    if (lstat(path, &status) != 0) return errno == ENOENT ? 0 : -1;
    if (!S_ISDIR(status.st_mode)) return unlink(path);
    dir = opendir(path);
    if (!dir) return -1;
    while ((entry = readdir(dir)) != NULL) {
        char child[YVEX_PATH_CAP];
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        if (snprintf(child, sizeof(child), "%s/%s", path, entry->d_name) >=
            (int)sizeof(child) || remove_tree(child) != 0) {
            (void)closedir(dir);
            return -1;
        }
    }
    if (closedir(dir) != 0) return -1;
    return rmdir(path);
}

static int copy_file_direct(const char *source, const char *destination,
                            yvex_error *err)
{
    unsigned char *buffer = NULL;
    struct stat before, after;
    int in = -1, out = -1, rc = YVEX_OK;
    ssize_t count;

    in = open(source, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (in < 0 || fstat(in, &before) != 0 || !S_ISREG(before.st_mode)) {
        rc = distribution_refuse(err, YVEX_ERR_IO, "source.distribution.copy",
                                 "cannot open regular source file");
        goto done;
    }
    rc = yvex_core_mkdir_parent(destination, "source.distribution.copy", err);
    if (rc != YVEX_OK) goto done;
    out = open(destination, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (out < 0) {
        rc = distribution_refuse(err, errno == EEXIST ? YVEX_ERR_STATE : YVEX_ERR_IO,
                                 "source.distribution.copy",
                                 "cannot create destination file");
        goto done;
    }
    buffer = malloc(SOURCE_COPY_BUFFER_BYTES);
    if (!buffer) {
        rc = distribution_refuse(err, YVEX_ERR_NOMEM, "source.distribution.copy",
                                 "copy buffer allocation failed");
        goto done;
    }
    while ((count = read(in, buffer, SOURCE_COPY_BUFFER_BYTES)) > 0) {
        size_t offset = 0u;
        while (offset < (size_t)count) {
            ssize_t written = write(out, buffer + offset, (size_t)count - offset);
            if (written <= 0) {
                rc = distribution_refuse(err, YVEX_ERR_IO, "source.distribution.copy",
                                         "cannot write destination file");
                goto done;
            }
            offset += (size_t)written;
        }
    }
    if (count < 0 || fsync(out) != 0 || fstat(in, &after) != 0 ||
        before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_size != after.st_size || before.st_mtime != after.st_mtime ||
        before.st_ctime != after.st_ctime)
        rc = distribution_refuse(err, YVEX_ERR_IO, "source.distribution.copy",
                                 "source changed or destination sync failed during copy");
done:
    free(buffer);
    if (in >= 0) (void)close(in);
    if (out >= 0 && close(out) != 0 && rc == YVEX_OK)
        rc = distribution_refuse(err, YVEX_ERR_IO, "source.distribution.copy",
                                 "cannot close destination file");
    if (rc != YVEX_OK) (void)unlink(destination);
    return rc;
}

static int publish_temporary(const char *temporary, const char *destination,
                             int directory, yvex_error *err)
{
    if (!directory) {
        if (link(temporary, destination) == 0) {
            if (unlink(temporary) == 0) return YVEX_OK;
            (void)unlink(destination);
            return distribution_refuse(err, YVEX_ERR_IO,
                                       "source.distribution.copy",
                                       "cannot remove temporary copy name");
        }
        return distribution_refuse(err, errno == EEXIST ? YVEX_ERR_STATE : YVEX_ERR_IO,
                                   "source.distribution.copy",
                                   errno == EEXIST ? "destination already exists"
                                                    : "cannot publish copied file");
    }
#ifdef __linux__
    if (syscall(SYS_renameat2, AT_FDCWD, temporary, AT_FDCWD, destination,
                1u /* RENAME_NOREPLACE */) == 0)
        return YVEX_OK;
    return distribution_refuse(err, errno == EEXIST ? YVEX_ERR_STATE : YVEX_ERR_IO,
                               "source.distribution.copy",
                               errno == EEXIST ? "destination already exists"
                                                : "cannot publish copied directory");
#else
    (void)temporary;
    (void)destination;
    return distribution_refuse(err, YVEX_ERR_UNSUPPORTED,
                               "source.distribution.copy",
                               "atomic no-replace directory publication is unavailable");
#endif
}

static int copy_representation(const yvex_source_representation_fact *source,
                               const char *destination, const char *scratch,
                               yvex_error *err)
{
    char temporary[YVEX_PATH_CAP];
    yvex_source_manifest_file_list files;
    size_t index;
    struct stat status;
    int rc;

    if (access(destination, F_OK) == 0)
        return distribution_refuse(err, YVEX_ERR_STATE,
                                   "source.distribution.copy",
                                   "destination already exists");
    if (snprintf(temporary, sizeof(temporary), "%s.partial.%ld", scratch ? scratch : destination,
                 (long)getpid()) >= (int)sizeof(temporary))
        return distribution_refuse(err, YVEX_ERR_BOUNDS, "source.distribution.copy",
                                   "temporary destination path is too long");
    if (lstat(temporary, &status) == 0 || errno != ENOENT)
        return distribution_refuse(err, YVEX_ERR_STATE, "source.distribution.copy",
                                   "temporary copy path already exists");
    rc = yvex_core_mkdir_parent(destination, "source.distribution.copy", err);
    if (rc != YVEX_OK) return rc;
    if (!source->directory) {
        rc = copy_file_direct(source->path, temporary, err);
    } else {
        yvex_source_manifest_file_list_init(&files);
        rc = yvex_core_mkdir_parent(temporary, "source.distribution.copy", err);
        if (rc == YVEX_OK && mkdir(temporary, 0755) != 0)
            rc = distribution_refuse(err, YVEX_ERR_IO, "source.distribution.copy",
                                     "cannot create temporary destination directory");
        if (rc == YVEX_OK)
            rc = yvex_source_manifest_scan_files(source->path, 1, &files, err);
        for (index = 0u; rc == YVEX_OK && index < files.count; ++index) {
            char input[YVEX_PATH_CAP], output[YVEX_PATH_CAP];
            if (!yvex_source_path_join(input, sizeof(input), source->path,
                                       files.items[index].path) ||
                !yvex_source_path_join(output, sizeof(output), temporary,
                                       files.items[index].path))
                rc = distribution_refuse(err, YVEX_ERR_BOUNDS,
                                         "source.distribution.copy",
                                         "source member path is too long");
            else
                rc = copy_file_direct(input, output, err);
        }
        yvex_source_manifest_file_list_free(&files);
    }
    if (rc == YVEX_OK)
        rc = publish_temporary(temporary, destination, source->directory, err);
    if (rc != YVEX_OK) (void)remove_tree(temporary);
    return rc;
}

static int record_write(const char *path, const yvex_source_import_options *options,
                        const yvex_source_import_result *result, const char *name,
                        const char *family, yvex_error *err)
{
    char *json = NULL;
    size_t length = 0u;
    FILE *stream;
    yvex_core_file_result publication;
    int rc;

    stream = open_memstream(&json, &length);
    if (!stream)
        return distribution_refuse(err, YVEX_ERR_IO, "source.distribution.record",
                                   "cannot allocate source record stream");
    fputs("{\n  \"schema\": \"yvex.model-source.registry.v1\",\n  \"name\": ", stream);
    yvex_file_json_write_string(stream, name);
    fputs(",\n  \"family\": ", stream); yvex_file_json_write_string(stream, family);
    fputs(",\n  \"provider\": \"local\",\n  \"repository\": ", stream);
    yvex_file_json_write_string(stream, result->representation.digest);
    fputs(",\n  \"revision\": ", stream);
    yvex_file_json_write_string(stream, result->representation.digest);
    fputs(",\n  \"origin_uri\": ", stream); yvex_file_json_write_string(stream, result->origin_uri);
    fputs(",\n  \"source_path\": ", stream); yvex_file_json_write_string(stream, result->source_path);
    fputs(",\n  \"storage\": ", stream); yvex_file_json_write_string(stream, result->storage);
    fputs(",\n  \"format\": ", stream); yvex_file_json_write_string(stream, result->representation.format);
    fputs(",\n  \"precision\": ", stream); yvex_file_json_write_string(stream, result->representation.precision);
    fputs(",\n  \"digest\": ", stream); yvex_file_json_write_string(stream, result->representation.digest);
    fprintf(stream, ",\n  \"size_bytes\": %llu,\n  \"file_count\": %llu,\n"
                    "  \"directory\": %s,\n  \"status\": \"complete\",\n"
                    "  \"verification\": \"payload-verified\"\n}\n",
            result->representation.size_bytes, result->representation.file_count,
            result->representation.directory ? "true" : "false");
    if (fclose(stream) != 0 || !json) {
        free(json);
        return distribution_refuse(err, YVEX_ERR_IO, "source.distribution.record",
                                   "cannot finalize source record");
    }
    (void)options;
    rc = yvex_core_mkdir_parent(path, "source.distribution.record", err);
    if (rc == YVEX_OK)
        rc = yvex_core_file_publish_noreplace(path, json, length, NULL, NULL,
                                              NULL, &publication, err);
    if (rc == YVEX_ERR_STATE) {
        unsigned char *existing = NULL;
        size_t existing_length = 0u;
        yvex_error prior;
        yvex_error_clear(&prior);
        if (yvex_core_file_read_snapshot(path, SOURCE_RECORD_BYTES, &existing,
                                         &existing_length, &publication, &prior) == YVEX_OK &&
            existing_length == length && !memcmp(existing, json, length)) {
            yvex_error_clear(err);
            rc = YVEX_OK;
        }
        free(existing);
    }
    free(json);
    return rc;
}

int yvex_source_register_reference(const yvex_source_reference_options *options,
                                   yvex_source_reference_result *out,
                                   yvex_error *err)
{
    unsigned char identity_digest[YVEX_SHA256_DIGEST_BYTES];
    char identity_hex[65], name[YVEX_SOURCE_DISTRIBUTION_NAME_CAP];
    char source_path[YVEX_PATH_CAP];
    char family[64], *json = NULL;
    size_t length = 0u;
    FILE *stream;
    yvex_sha256 identity;
    yvex_core_file_result publication;
    int rc;

    if (!options || !options->locator || !options->models_root || !out ||
        options->locator->kind != YVEX_SOURCE_LOCATOR_HUGGINGFACE ||
        !yvex_source_provider_path(source_path, sizeof(source_path),
                                   options->models_root, options->locator->repository,
                                   options->resolved_revision))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "source.distribution.reference",
                                   "an immutable Hugging Face reference is required");
    memset(out, 0, sizeof(*out));
    if (!safe_name(name, sizeof(name), options->name && options->name[0]
                                             ? options->name
                                             : path_basename(options->locator->repository)) ||
        !safe_name(family, sizeof(family), options->family && options->family[0]
                                               ? options->family : "unknown"))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "source.distribution.reference",
                                   "model name or family is invalid");
    if (snprintf(out->immutable_uri, sizeof(out->immutable_uri), "hf://%s@%s",
                 options->locator->repository, options->resolved_revision) >=
        (int)sizeof(out->immutable_uri))
        return distribution_refuse(err, YVEX_ERR_BOUNDS,
                                   "source.distribution.reference",
                                   "immutable provider locator is too long");
    yvex_sha256_init(&identity);
    if (!yvex_sha256_update_text(&identity, "yvex.remote-source-reference.v1") ||
        !yvex_sha256_update_text(&identity, out->immutable_uri) ||
        !yvex_sha256_final(&identity, identity_digest))
        return distribution_refuse(err, YVEX_ERR_STATE,
                                   "source.distribution.reference",
                                   "cannot derive reference record key");
    yvex_sha256_hex(identity_digest, identity_hex);
    if (snprintf(out->record_path, sizeof(out->record_path),
                 "%s/registry/sources/%s-%.16s.source.json",
                 options->models_root, name, identity_hex) >= (int)sizeof(out->record_path))
        return distribution_refuse(err, YVEX_ERR_BOUNDS,
                                   "source.distribution.reference",
                                   "source reference record path is too long");
    stream = open_memstream(&json, &length);
    if (!stream)
        return distribution_refuse(err, YVEX_ERR_IO,
                                   "source.distribution.reference",
                                   "cannot allocate source reference stream");
    fputs("{\n  \"schema\": \"yvex.model-source.registry.v1\",\n  \"name\": ", stream);
    yvex_file_json_write_string(stream, name);
    fputs(",\n  \"family\": ", stream); yvex_file_json_write_string(stream, family);
    fputs(",\n  \"provider\": \"huggingface\",\n  \"repository\": ", stream);
    yvex_file_json_write_string(stream, options->locator->repository);
    fputs(",\n  \"revision\": ", stream);
    yvex_file_json_write_string(stream, options->resolved_revision);
    fputs(",\n  \"origin_uri\": ", stream); yvex_file_json_write_string(stream, out->immutable_uri);
    fputs(",\n  \"source_path\": \"\",\n  \"storage\": \"remote\",\n  \"format\": ", stream);
    yvex_file_json_write_string(stream, options->format ? options->format : "unknown");
    fputs(",\n  \"precision\": ", stream);
    yvex_file_json_write_string(stream, options->precision ? options->precision : "");
    fprintf(stream, ",\n  \"digest\": \"\",\n  \"size_bytes\": %llu,\n"
                    "  \"file_count\": 0,\n  \"directory\": false,\n"
                    "  \"status\": \"reference\",\n"
                    "  \"verification\": \"revision-verified\"\n}\n",
            options->size_known ? options->size_bytes : 0u);
    if (fclose(stream) != 0 || !json) {
        free(json);
        return distribution_refuse(err, YVEX_ERR_IO,
                                   "source.distribution.reference",
                                   "cannot finalize source reference record");
    }
    rc = yvex_core_mkdir_parent(out->record_path, "source.distribution.reference", err);
    if (rc == YVEX_OK)
        rc = yvex_core_file_publish_noreplace(out->record_path, json, length,
                                              NULL, NULL, NULL, &publication, err);
    if (rc == YVEX_ERR_STATE) {
        unsigned char *existing = NULL;
        size_t existing_length = 0u;
        yvex_error prior;
        yvex_error_clear(&prior);
        if (yvex_core_file_read_snapshot(out->record_path, SOURCE_RECORD_BYTES,
                                         &existing, &existing_length,
                                         &publication, &prior) == YVEX_OK &&
            existing_length == length && !memcmp(existing, json, length)) {
            yvex_error_clear(err);
            rc = YVEX_OK;
        }
        free(existing);
    }
    free(json);
    if (rc == YVEX_OK) out->registered = 1;
    return rc;
}

int yvex_source_import_local(const yvex_source_import_options *options,
                             yvex_source_import_result *out, yvex_error *err)
{
    char name[YVEX_SOURCE_DISTRIBUTION_NAME_CAP];
    char family[64], leaf[YVEX_PATH_CAP], scratch[YVEX_PATH_CAP];
    yvex_source_representation_fact verified;
    int rc;

    if (!options || !options->locator || !options->models_root || !out ||
        options->locator->kind != YVEX_SOURCE_LOCATOR_LOCAL_PATH)
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "source.distribution.import",
                                   "local locator, model root, and output are required");
    memset(out, 0, sizeof(*out));
    rc = yvex_source_representation_inspect_local(options->locator,
                                                  &out->representation, err);
    if (rc != YVEX_OK) return rc;
    if (!safe_name(name, sizeof(name), options->name && options->name[0]
                                             ? options->name
                                             : out->representation.name) ||
        !safe_name(family, sizeof(family), options->family && options->family[0]
                                               ? options->family : "unknown"))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "source.distribution.import",
                                   "model name or family is invalid");
    if (distribution_copy(out->origin_uri, sizeof(out->origin_uri),
                          options->locator->canonical, err,
                          "source.distribution.import") != YVEX_OK)
        return yvex_error_code(err);
    if (options->storage == YVEX_SOURCE_STORAGE_EXTERNAL) {
        (void)distribution_copy(out->storage, sizeof(out->storage), "external", err,
                                "source.distribution.import");
        rc = distribution_copy(out->source_path, sizeof(out->source_path),
                               options->locator->path, err,
                               "source.distribution.import");
    } else {
        const char *base = out->representation.directory ? "source" :
                          !strcmp(out->representation.format, "gguf") ? "model.gguf" :
                          "model.safetensors";
        (void)distribution_copy(out->storage, sizeof(out->storage), "managed", err,
                                "source.distribution.import");
        if (snprintf(leaf, sizeof(leaf), "%s/source/local/%s/%s",
                     options->models_root, out->representation.digest, base) >=
            (int)sizeof(leaf))
            return distribution_refuse(err, YVEX_ERR_BOUNDS,
                                       "source.distribution.import",
                                       "managed source path is too long");
        if (snprintf(scratch, sizeof(scratch), "%s/tmp/imports/%s",
                     options->models_root, out->representation.digest) >= (int)sizeof(scratch))
            return distribution_refuse(err, YVEX_ERR_BOUNDS, "source.distribution.import",
                                       "import temporary path is too long");
        rc = copy_representation(&out->representation, leaf, scratch, err);
        if (rc == YVEX_ERR_STATE) {
            yvex_source_locator existing_locator;
            yvex_error_clear(err);
            memset(&existing_locator, 0, sizeof(existing_locator));
            existing_locator.kind = YVEX_SOURCE_LOCATOR_LOCAL_PATH;
            (void)distribution_copy(existing_locator.path, sizeof(existing_locator.path),
                                    leaf, err, "source.distribution.import");
            rc = yvex_source_representation_inspect_local(&existing_locator,
                                                           &verified, err);
            if (rc == YVEX_OK && strcmp(verified.digest,
                                        out->representation.digest))
                rc = distribution_refuse(err, YVEX_ERR_STATE,
                                         "source.distribution.import",
                                         "managed destination identity conflicts");
        } else if (rc == YVEX_OK) {
            out->copied = 1;
        }
        if (rc == YVEX_OK)
            rc = distribution_copy(out->source_path, sizeof(out->source_path),
                                   leaf, err, "source.distribution.import");
    }
    if (rc != YVEX_OK) return rc;
    (void)distribution_copy(out->representation.path,
                            sizeof(out->representation.path), out->source_path,
                            err, "source.distribution.import");
    if (snprintf(out->record_path, sizeof(out->record_path),
                 "%s/registry/sources/%s-%.16s.source.json",
                 options->models_root, name, out->representation.digest) >=
        (int)sizeof(out->record_path))
        return distribution_refuse(err, YVEX_ERR_BOUNDS,
                                   "source.distribution.import",
                                   "source record path is too long");
    rc = record_write(out->record_path, options, out, name, family, err);
    if (rc == YVEX_OK) out->registered = 1;
    return rc;
}

int yvex_source_export_local(const yvex_source_export_options *options,
                             yvex_source_export_result *out, yvex_error *err)
{
    yvex_source_locator source_locator, destination_locator;
    yvex_source_representation_fact source, copied;
    char destination[YVEX_PATH_CAP], parent[YVEX_PATH_CAP];
    struct stat status;
    int rc;

    if (!options || !options->source_path || !options->destination_path || !out)
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "source.distribution.export",
                                   "source, destination, and output are required");
    memset(out, 0, sizeof(*out));
    memset(&source_locator, 0, sizeof(source_locator));
    source_locator.kind = YVEX_SOURCE_LOCATOR_LOCAL_PATH;
    if (!realpath(options->source_path, source_locator.path))
        return distribution_refuse(err, YVEX_ERR_IO, "source.distribution.export",
                                   "export source is unavailable");
    rc = yvex_source_representation_inspect_local(&source_locator, &source, err);
    if (rc != YVEX_OK) return rc;
    if (options->expected_digest && options->expected_digest[0] &&
        strcmp(options->expected_digest, source.digest))
        return distribution_refuse(err, YVEX_ERR_STATE, "source.distribution.export",
                                   "export source digest does not match its registry identity");
    if (!strncmp(options->destination_path, "file://", 7u)) {
        if (options->destination_path[7] != '/')
            return distribution_refuse(err, YVEX_ERR_INVALID_ARG,
                                       "source.distribution.export",
                                       "file destination requires an absolute path");
        if (distribution_copy(destination, sizeof(destination),
                              options->destination_path + 7u, err,
                              "source.distribution.export") != YVEX_OK)
            return yvex_error_code(err);
    } else if (strstr(options->destination_path, "://")) {
        return distribution_refuse(err, YVEX_ERR_UNSUPPORTED,
                                   "source.distribution.export",
                                   "destination transport is unavailable");
    } else if (options->destination_path[0] == '/') {
        if (distribution_copy(destination, sizeof(destination),
                              options->destination_path, err,
                              "source.distribution.export") != YVEX_OK)
            return yvex_error_code(err);
    } else {
        if (!getcwd(parent, sizeof(parent)) ||
            snprintf(destination, sizeof(destination), "%s/%s", parent,
                     options->destination_path) >= (int)sizeof(destination))
            return distribution_refuse(err, YVEX_ERR_BOUNDS,
                                       "source.distribution.export",
                                       "destination path is too long");
    }
    if (lstat(destination, &status) == 0 && S_ISDIR(status.st_mode)) {
        size_t used = strlen(destination);
        if (snprintf(destination + used, sizeof(destination) - used, "/%s",
                     path_basename(source.path)) >= (int)(sizeof(destination) - used))
            return distribution_refuse(err, YVEX_ERR_BOUNDS,
                                       "source.distribution.export",
                                       "destination member path is too long");
    }
    rc = copy_representation(&source, destination, NULL, err);
    if (rc != YVEX_OK) return rc;
    memset(&destination_locator, 0, sizeof(destination_locator));
    destination_locator.kind = YVEX_SOURCE_LOCATOR_LOCAL_PATH;
    (void)distribution_copy(destination_locator.path, sizeof(destination_locator.path),
                            destination, err, "source.distribution.export");
    rc = yvex_source_representation_inspect_local(&destination_locator, &copied, err);
    if (rc != YVEX_OK || strcmp(source.digest, copied.digest)) {
        (void)remove_tree(destination);
        return rc != YVEX_OK ? rc
                             : distribution_refuse(err, YVEX_ERR_STATE,
                                                   "source.distribution.export",
                                                   "published representation digest changed");
    }
    (void)distribution_copy(out->source_path, sizeof(out->source_path),
                            source.path, err, "source.distribution.export");
    (void)distribution_copy(out->destination_path, sizeof(out->destination_path),
                            destination, err, "source.distribution.export");
    (void)distribution_copy(out->digest, sizeof(out->digest), source.digest, err,
                            "source.distribution.export");
    out->bytes = source.size_bytes;
    out->copied = 1;
    return YVEX_OK;
}
