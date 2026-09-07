/* Own deterministic locator parsing and local source distribution mechanics. */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <yvex/internal/source_distribution.h>

#include <yvex/internal/core.h>
#include <yvex/internal/artifact_storage.h>
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
#include <sys/file.h>
#include <sys/ioctl.h>
#ifdef __linux__
#include <sys/syscall.h>
#include <linux/fs.h>
#include <linux/openat2.h>
#endif
#include <sys/types.h>
#include <unistd.h>

#define SOURCE_COPY_BUFFER_BYTES (1024u * 1024u)
#define SOURCE_RECORD_BYTES (YVEX_PATH_CAP * 3u + 2048u)

static int representation_receipt(const char *root,
                                   const yvex_source_representation_fact *fact,
                                   int publish, yvex_error *err);
static int directory_receipt_hit(const char *root, const char *path, const char *expected,
                                  yvex_source_representation_fact *out, yvex_error *err);

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
    char resolved[YVEX_PATH_CAP], parent[YVEX_PATH_CAP], logical[YVEX_PATH_CAP];
    const char *base;
    char *slash;
    size_t length;
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
    (void)distribution_copy(parent, sizeof(parent), path, err, "source.locator");
    length = strlen(parent);
    while (length > 1u && parent[length - 1u] == '/') parent[--length] = '\0';
    slash = strrchr(parent, '/');
    base = slash ? slash + 1 : parent;
    if (slash) {
        *slash = '\0';
        if (!realpath(parent[0] ? parent : "/", logical)) return YVEX_ERR_IO;
    } else if (!getcwd(logical, sizeof(logical))) return YVEX_ERR_IO;
    if (snprintf(out->canonical, sizeof(out->canonical), "file://%s%s%s",
                 logical, !strcmp(logical, "/") ? "" : "/", base) >=
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
                             unsigned long long *bytes,
                             yvex_artifact_snapshot *snapshot, yvex_error *err)
{
    yvex_artifact *artifact = NULL;
    char physical[YVEX_PATH_CAP], current[YVEX_PATH_CAP];
    yvex_artifact_options options = {physical, 1, 0};
    yvex_artifact_file_identity identity;
    int rc;
    if (!realpath(path, physical))
        return distribution_refuse(err, YVEX_ERR_IO, "source.hash", "source file target is unavailable");
    rc = yvex_artifact_open(&artifact, &options, err);
    if (rc == YVEX_OK) rc = yvex_artifact_identity_read_open(artifact, &identity, err);
    if (rc == YVEX_OK && (!realpath(path, current) || strcmp(current, physical)))
        rc = distribution_refuse(err, YVEX_ERR_STATE, "source.hash", "source link changed during verification");
    if (rc == YVEX_OK && snapshot)
        rc = yvex_artifact_snapshot_get(artifact, snapshot, err);
    if (rc == YVEX_OK) {
        memcpy(digest_hex, identity.sha256, sizeof(identity.sha256));
        if (bytes) *bytes = identity.file_size;
    }
    yvex_artifact_close(artifact);
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

static int inspect_directory(const char *path, const char *root,
                             yvex_source_representation_fact *out, yvex_error *err)
{
    yvex_source_manifest_file_list files;
    yvex_sha256 tree;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long gguf = 0u;
    size_t index, record_bytes = 0u;
    char *record = NULL;
    FILE *stream = root ? open_memstream(&record, &record_bytes) : NULL;
    int rc;
    if (root && !stream)
        return distribution_refuse(err, YVEX_ERR_NOMEM, "source.inspect", "verification index allocation failed");
    if (stream) fputs("{\"files\":[", stream);
    yvex_source_manifest_file_list_init(&files);
    rc = yvex_source_manifest_scan_import_files(path, &files, err);
    if (rc != YVEX_OK) goto done;
    yvex_sha256_init(&tree);
    if (!yvex_sha256_update_text(&tree, "yvex.local-source.v1")) {
        rc = YVEX_ERR_STATE;
        goto done;
    }
    for (index = 0u; index < files.count; ++index) {
        char absolute[YVEX_PATH_CAP], file_digest[65];
        unsigned long long bytes = 0u;
        yvex_source_representation_fact member = {0};
        if (!yvex_source_path_join(absolute, sizeof(absolute), path,
                                   files.items[index].path)) {
            rc = distribution_refuse(err, YVEX_ERR_BOUNDS,
                                     "source.distribution.inspect",
                                     "source member path is too long");
            goto done;
        }
        rc = hash_regular_file(absolute, file_digest, &bytes, &member.snapshot, err);
        if (rc != YVEX_OK) goto done;
        if (stream) {
            member.snapshot_verified = 1;
            (void)distribution_copy(member.path, sizeof(member.path), absolute, err, "source.inspect");
            (void)distribution_copy(member.digest, sizeof(member.digest), file_digest, err, "source.inspect");
            rc = representation_receipt(root, &member, 1, err);
            if (rc != YVEX_OK) goto done;
            if (index) fputs(",", stream);
            fputs("{\"path\":", stream); yvex_file_json_write_string(stream, files.items[index].path);
            fprintf(stream, ",\"sha256\":\"%s\",\"size_bytes\":%llu}", file_digest, bytes);
        }
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
    if (stream) {
        char cache_path[YVEX_PATH_CAP];
        yvex_core_file_result publication;
        fputs("]}\n", stream);
        if (fclose(stream) != 0 && rc == YVEX_OK) rc = YVEX_ERR_IO;
        if (rc == YVEX_OK && (record_bytes > 16u * 1024u * 1024u ||
            snprintf(cache_path, sizeof(cache_path), "%s/cache/verification/source-reopen/%s.json",
                     root, out->digest) >= (int)sizeof(cache_path))) rc = YVEX_ERR_BOUNDS;
        if (rc == YVEX_OK) rc = yvex_core_mkdir_parent(cache_path, "source.inspect", err);
        if (rc == YVEX_OK)
            rc = yvex_core_file_publish_replace(cache_path, record, record_bytes, NULL, NULL, &publication, err);
        free(record);
    }
    yvex_source_manifest_file_list_free(&files);
    return rc;
}

static int representation_inspect_local(
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
    representation_name(out->name, sizeof(out->name),
                         locator->canonical[0] ? locator->canonical : locator->path);
    if (S_ISDIR(status.st_mode)) rc = inspect_directory(locator->path, NULL, out, err);
    else if (S_ISREG(status.st_mode)) {
        rc = hash_regular_file(locator->path, out->digest, &out->size_bytes, &out->snapshot, err);
        if (rc == YVEX_OK) {
            out->snapshot_verified = 1;
            out->file_count = 1u;
            (void)distribution_copy(out->format, sizeof(out->format),
                                    (ends_with(locator->path, ".gguf") ||
                                     ends_with(locator->canonical, ".gguf")) ? "gguf"
                                    : (ends_with(locator->path, ".safetensors") ||
                                       ends_with(locator->canonical, ".safetensors"))
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

/* Receipts are rebuildable verification evidence, never model identity or ownership. */
static int representation_receipt(const char *root,
                                   const yvex_source_representation_fact *fact,
                                   int publish, yvex_error *err)
{
    char cache[YVEX_PATH_CAP], physical[YVEX_PATH_CAP], current[YVEX_PATH_CAP];
    yvex_artifact *artifact = NULL;
    yvex_artifact_options options = {physical, 1, 0};
    yvex_artifact_snapshot snapshot;
    yvex_artifact_reopen_lease lease;
    int rc;
    if (fact->directory || !realpath(fact->path, physical)) return YVEX_ERR_STATE;
    if (snprintf(cache, sizeof(cache), "%s/cache/verification", root) >= (int)sizeof(cache))
        return distribution_refuse(err, YVEX_ERR_BOUNDS, "source.receipt", "cache path is too long");
    rc = yvex_artifact_open(&artifact, &options, err);
    if (rc == YVEX_OK && publish) {
        rc = yvex_artifact_snapshot_get(artifact, &snapshot, err);
        if (rc == YVEX_OK && (!fact->snapshot_verified ||
            !yvex_artifact_snapshot_equal(&snapshot, &fact->snapshot)))
            rc = distribution_refuse(err, YVEX_ERR_STATE, "source.receipt",
                                       "source snapshot changed after verification");
        if (rc == YVEX_OK)
            rc = yvex_artifact_reopen_lease_publish(artifact, fact->digest, cache, &lease, err);
    } else if (rc == YVEX_OK) {
        rc = yvex_artifact_reopen_lease_check(artifact, fact->digest, cache, &lease, err);
        if (rc == YVEX_OK && !lease.verified) rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK && (!realpath(fact->path, current) || strcmp(current, physical)))
        rc = distribution_refuse(err, YVEX_ERR_STATE, "source.receipt", "source link changed during reuse");
    yvex_artifact_close(artifact);
    return rc;
}

int yvex_source_file_reopen(const char *models_root, const char *path,
                             const char *digest, yvex_error *err)
{
    yvex_source_representation_fact fact = {0};
    if (!models_root || !path || !yvex_sha256_hex_is_valid(digest))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.reopen", "exact file identity required");
    if (distribution_copy(fact.path, sizeof(fact.path), path, err, "source.reopen") != YVEX_OK ||
        distribution_copy(fact.digest, sizeof(fact.digest), digest, err, "source.reopen") != YVEX_OK)
        return yvex_error_code(err);
    return representation_receipt(models_root, &fact, 0, err);
}

static int receipt_member_parse(yvex_json *json, char path[YVEX_PATH_CAP], char digest[65],
                                 unsigned long long *bytes)
{
    yvex_json_iter object;
    yvex_json_item item;
    char key[32];
    unsigned int seen = 0u;
    if (!yvex_json_iter_begin(json, &object, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&object, key, sizeof(key))) == YVEX_JSON_ITEM_READY) {
        unsigned int bit = !strcmp(key, "path") ? 1u : !strcmp(key, "sha256") ? 2u :
                           !strcmp(key, "size_bytes") ? 4u : 0u;
        if (!bit || (seen & bit)) return 0;
        seen |= bit;
        if (bit == 1u && !yvex_json_string(json, path, YVEX_PATH_CAP)) return 0;
        if (bit == 2u && !yvex_json_string(json, digest, 65u)) return 0;
        if (bit == 4u && !yvex_json_u64(json, bytes)) return 0;
    }
    return item == YVEX_JSON_ITEM_END && seen == 7u && yvex_sha256_hex_is_valid(digest);
}

static int directory_receipt_read(const char *root, const char *path, const char *expected,
                                   const char *member_digest, yvex_source_representation_fact *out,
                                   yvex_error *err)
{
    char cache[YVEX_PATH_CAP], digest_hex[65], *record;
    const char *array;
    yvex_source_manifest_file_list files;
    yvex_sha256 tree;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_json json;
    yvex_json_iter iterator;
    yvex_json_item item = YVEX_JSON_ITEM_END;
    size_t length, index = 0u, gguf = 0u;
    yvex_source_representation_fact selected = {0};
    int rc = YVEX_ERR_STATE;
    if (!yvex_sha256_hex_is_valid(expected) ||
        snprintf(cache, sizeof(cache), "%s/cache/verification/source-reopen/%s.json", root, expected) >=
            (int)sizeof(cache)) return YVEX_ERR_STATE;
    record = yvex_read_bounded_file(cache, 16u * 1024u * 1024u, &length, err);
    if (!record) return YVEX_ERR_STATE;
    yvex_source_manifest_file_list_init(&files);
    if (yvex_source_manifest_scan_import_files(path, &files, err) != YVEX_OK) goto done;
    array = yvex_json_probe_field_value(record, "files");
    if (!array) goto done;
    yvex_json_init(&json, array, strlen(array));
    if (!yvex_json_iter_begin(&json, &iterator, YVEX_JSON_COLLECTION_ARRAY)) goto done;
    yvex_sha256_init(&tree);
    (void)yvex_sha256_update_text(&tree, "yvex.local-source.v1");
    while ((item = yvex_json_array_value(&iterator)) == YVEX_JSON_ITEM_READY) {
        char relative[YVEX_PATH_CAP];
        yvex_source_representation_fact member = {0};
        if (index >= files.count || !receipt_member_parse(&json, relative, member.digest, &member.size_bytes) ||
            strcmp(relative, files.items[index].path) || member.size_bytes != files.items[index].size_bytes ||
            !yvex_source_path_join(member.path, sizeof(member.path), path, relative) ||
            representation_receipt(root, &member, 0, err) != YVEX_OK ||
            !yvex_sha256_update_text(&tree, relative) || !yvex_sha256_update_u64(&tree, member.size_bytes) ||
            !yvex_sha256_update_text(&tree, member.digest)) goto done;
        if (ends_with(relative, ".gguf")) {
            gguf++;
            if (member_digest && !strcmp(member_digest, member.digest)) selected = member;
        }
        index++;
    }
    if (item != YVEX_JSON_ITEM_END || !index || index != files.count || !yvex_sha256_final(&tree, digest))
        goto done;
    yvex_sha256_hex(digest, digest_hex);
    if (strcmp(expected, digest_hex)) goto done;
    if (member_digest) {
        char resolved[YVEX_PATH_CAP];
        if (gguf != 1u || files.summary.safetensors_count || !selected.path[0] ||
            !realpath(selected.path, resolved) || strcmp(resolved, selected.path)) goto done;
        *out = selected;
        out->file_count = 1u;
        (void)distribution_copy(out->format, sizeof(out->format), "gguf", err, "source.receipt");
        rc = YVEX_OK;
        yvex_error_clear(err);
        goto done;
    }
    memset(out, 0, sizeof(*out));
    (void)distribution_copy(out->path, sizeof(out->path), path, err, "source.receipt");
    (void)distribution_copy(out->digest, sizeof(out->digest), expected, err, "source.receipt");
    out->directory = 1;
    out->size_bytes = files.summary.total_size_bytes;
    out->file_count = files.count;
    (void)distribution_copy(out->format, sizeof(out->format),
                            files.summary.safetensors_count ? (gguf ? "mixed" : "safetensors") :
                            gguf ? "gguf" : "source", err, "source.receipt");
    rc = YVEX_OK;
    yvex_error_clear(err);
done:
    yvex_source_manifest_file_list_free(&files);
    free(record);
    return rc;
}

static int directory_receipt_hit(const char *root, const char *path, const char *expected,
                                  yvex_source_representation_fact *out, yvex_error *err)
{
    return directory_receipt_read(root, path, expected, NULL, out, err);
}

int yvex_source_gguf_member_reopen(const char *models_root, const char *path, const char *tree_digest,
                                    const char *member_digest, yvex_source_representation_fact *out,
                                    yvex_error *err)
{
    if (!models_root || !path || !out || !yvex_sha256_hex_is_valid(member_digest))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.receipt", "exact member identity required");
    return directory_receipt_read(models_root, path, tree_digest, member_digest, out, err);
}

int yvex_source_representation_verify_local(
    const yvex_source_locator *locator, const char *models_root, const char *expected_digest,
    yvex_source_representation_fact *out, yvex_error *err)
{
    struct stat status;
    int rc;
    if (!locator || !models_root || !out || lstat(locator->path, &status) != 0)
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.verify", "existing source and root required");
    if (S_ISDIR(status.st_mode)) {
        rc = expected_digest ? directory_receipt_hit(models_root, locator->path, expected_digest, out, err)
                             : YVEX_ERR_STATE;
        if (rc != YVEX_OK) {
            yvex_error_clear(err);
            memset(out, 0, sizeof(*out));
            (void)distribution_copy(out->path, sizeof(out->path), locator->path, err, "source.verify");
            rc = inspect_directory(locator->path, models_root, out, err);
        }
        representation_name(out->name, sizeof(out->name), locator->path);
    } else {
        rc = yvex_source_representation_resolve_local(locator, models_root, out, err);
        if (rc == YVEX_OK) rc = representation_receipt(models_root, out, 1, err);
    }
    if (rc == YVEX_OK && expected_digest && strcmp(expected_digest, out->digest))
        rc = distribution_refuse(err, YVEX_ERR_STATE, "source.verify", "source content identity changed");
    return rc;
}

static int representation_record_hit(const char *record, const char *root,
                                      const yvex_source_locator *locator,
                                      yvex_source_representation_fact *out)
{
    char source[YVEX_PATH_CAP], origin[YVEX_PATH_CAP], digest[65];
    char *json;
    size_t length;
    yvex_error ignored;
    yvex_artifact *artifact = NULL;
    yvex_artifact_options options = {locator->path, 1, 0};
    yvex_artifact_reopen_lease lease;
    char cache[YVEX_PATH_CAP];
    int hit = 0;
    yvex_error_clear(&ignored);
    json = yvex_read_bounded_file(record, SOURCE_RECORD_BYTES, &length, &ignored);
    if (!json) return 0;
    if (!yvex_json_probe_string_field(json, "source_path", source, sizeof(source)) ||
        !yvex_json_probe_string_field(json, "origin_uri", origin, sizeof(origin)) ||
        (strcmp(source, locator->path) && strcmp(origin, locator->canonical)) ||
        !yvex_json_probe_string_field(json, "digest", digest, sizeof(digest)) ||
        !yvex_sha256_hex_is_valid(digest)) goto done;
    memset(out, 0, sizeof(*out));
    {
        struct stat status;
        if (lstat(locator->path, &status) == 0 && S_ISDIR(status.st_mode)) {
            hit = directory_receipt_hit(root, locator->path, digest, out, &ignored) == YVEX_OK;
            if (hit) {
                representation_name(out->name, sizeof(out->name), locator->path);
                (void)yvex_json_probe_string_field(json, "format", out->format, sizeof(out->format));
            }
            goto done;
        }
    }
    (void)distribution_copy(out->path, sizeof(out->path), locator->path, &ignored, "source.resolve");
    (void)distribution_copy(out->digest, sizeof(out->digest), digest, &ignored, "source.resolve");
    if (snprintf(cache, sizeof(cache), "%s/cache/verification", root) >= (int)sizeof(cache) ||
        yvex_artifact_open(&artifact, &options, &ignored) != YVEX_OK ||
        yvex_artifact_reopen_lease_check(artifact, digest, cache, &lease, &ignored) != YVEX_OK ||
        !lease.verified) goto done;
    out->snapshot = lease.snapshot;
    out->snapshot_verified = 1;
    out->size_bytes = out->snapshot.size;
    out->file_count = 1u;
    representation_name(out->name, sizeof(out->name),
                         locator->canonical[0] ? locator->canonical : locator->path);
    (void)yvex_json_probe_string_field(json, "format", out->format, sizeof(out->format));
    (void)yvex_json_probe_string_field(json, "precision", out->precision, sizeof(out->precision));
    hit = 1;
done:
    yvex_artifact_close(artifact);
    free(json);
    return hit;
}

int yvex_source_representation_resolve_local(
    const yvex_source_locator *locator, const char *models_root,
    yvex_source_representation_fact *out, yvex_error *err)
{
    char directory[YVEX_PATH_CAP];
    DIR *dir;
    struct dirent *entry;
    if (!locator || !models_root || !out)
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.resolve", "local source and root required");
    if (snprintf(directory, sizeof(directory), "%s/registry/sources", models_root) >= (int)sizeof(directory))
        return distribution_refuse(err, YVEX_ERR_BOUNDS, "source.resolve", "source registry path is too long");
    dir = opendir(directory);
    if (dir) {
        while ((entry = readdir(dir)) != NULL) {
            char path[YVEX_PATH_CAP];
            if (!ends_with(entry->d_name, ".source.json") ||
                snprintf(path, sizeof(path), "%s/%s", directory, entry->d_name) >= (int)sizeof(path)) continue;
            if (representation_record_hit(path, models_root, locator, out)) {
                (void)closedir(dir);
                yvex_error_clear(err);
                return YVEX_OK;
            }
        }
        (void)closedir(dir);
    }
    return representation_inspect_local(locator, out, err);
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
    char physical[YVEX_PATH_CAP], current[YVEX_PATH_CAP];
    int in = -1, out = -1, rc = YVEX_OK;
    ssize_t count;

    if (!realpath(source, physical))
        return distribution_refuse(err, YVEX_ERR_IO, "source.copy", "source file target is unavailable");
    in = open(physical, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
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
#ifdef __linux__
    if (ioctl(out, FICLONE, in) == 0) {
        count = 0;
        goto verify;
    }
#endif
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
verify:
    if (count < 0 || fsync(out) != 0 || fstat(in, &after) != 0 ||
        !realpath(source, current) || strcmp(current, physical) ||
        before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_size != after.st_size || before.st_mtime != after.st_mtime ||
        before.st_ctime != after.st_ctime ||
        before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
        before.st_ctim.tv_nsec != after.st_ctim.tv_nsec)
        rc = distribution_refuse(err, YVEX_ERR_IO, "source.distribution.copy",
                                 "source changed or destination sync failed during copy");
done:
    free(buffer);
    if (in >= 0) (void)close(in);
    if (out >= 0 && close(out) != 0 && rc == YVEX_OK)
        rc = distribution_refuse(err, YVEX_ERR_IO, "source.distribution.copy",
                                 "cannot close destination file");
    if (rc != YVEX_OK && out >= 0) (void)unlink(destination);
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
            rc = yvex_source_manifest_scan_import_files(source->path, &files, err);
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

int yvex_source_stage_file(const char *source, const char *destination, yvex_error *err)
{
    yvex_source_representation_fact file = {0};
    if (!source || !destination || strlen(source) >= sizeof(file.path))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.stage", "bounded file paths required");
    snprintf(file.path, sizeof(file.path), "%s", source);
    return copy_representation(&file, destination, NULL, err);
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
    if (options->local && (!options->remote_filename || !options->remote_filename[0] ||
        strchr(options->remote_filename, '/') || strstr(options->remote_filename, "..") ||
        options->local->directory || strcmp(options->local->format, "gguf") ||
        yvex_source_file_reopen(options->models_root, options->local->path, options->local->digest, err) != YVEX_OK))
        return distribution_refuse(err, YVEX_ERR_STATE, "source.reference", "verified local GGUF identity required");
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
        (options->local && !yvex_sha256_update_text(&identity, options->remote_filename)) ||
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
    fputs(",\n  \"source_path\": ", stream);
    yvex_file_json_write_string(stream, options->local ? options->local->path : "");
    fputs(",\n  \"storage\": ", stream);
    yvex_file_json_write_string(stream, options->local ? "managed" : "remote");
    fputs(",\n  \"remote_filename\": ", stream);
    yvex_file_json_write_string(stream, options->remote_filename ? options->remote_filename : "");
    fputs(",\n  \"format\": ", stream);
    yvex_file_json_write_string(stream, options->format ? options->format : "unknown");
    fputs(",\n  \"precision\": ", stream);
    yvex_file_json_write_string(stream, options->precision ? options->precision : "");
    fputs(",\n  \"digest\": ", stream);
    yvex_file_json_write_string(stream, options->local ? options->local->digest : "");
    fprintf(stream, ",\n  \"size_bytes\": %llu,\n  \"file_count\": %u,\n  \"directory\": false,\n",
            options->size_known ? options->size_bytes : 0u, options->local ? 1u : 0u);
    fprintf(stream, "  \"status\": \"%s\",\n  \"verification\": \"%s\"\n}\n",
            options->local ? "complete" : "reference", options->local ? "payload-verified" : "revision-verified");
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

static int source_import_locked(const yvex_source_import_options *options,
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
    if (options->inspected && !options->inspected->directory &&
        !strcmp(options->inspected->path, options->locator->path)) {
        out->representation = *options->inspected;
        rc = representation_receipt(options->models_root, &out->representation, 1, err);
    } else {
        rc = options->inspected && options->inspected->directory
            ? yvex_source_representation_verify_local(options->locator, options->models_root,
                                                       options->inspected->digest, &out->representation, err)
            : yvex_source_representation_resolve_local(options->locator, options->models_root,
                                                        &out->representation, err);
        if (rc == YVEX_OK && !out->representation.directory)
            rc = representation_receipt(options->models_root, &out->representation, 1, err);
    }
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
            rc = yvex_source_representation_resolve_local(&existing_locator, options->models_root,
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
    if (out->representation.directory) {
        yvex_source_locator destination = {0};
        destination.kind = YVEX_SOURCE_LOCATOR_LOCAL_PATH;
        (void)distribution_copy(destination.path, sizeof(destination.path), out->source_path, err, "source.import");
        rc = yvex_source_representation_verify_local(&destination, options->models_root,
                                                      out->representation.digest, &verified, err);
        if (rc != YVEX_OK) return rc;
    }
    if (!out->representation.directory && options->storage == YVEX_SOURCE_STORAGE_MANAGED) {
        yvex_source_locator destination = {0};
        destination.kind = YVEX_SOURCE_LOCATOR_LOCAL_PATH;
        (void)distribution_copy(destination.path, sizeof(destination.path), out->source_path,
                                err, "source.distribution.import");
        rc = yvex_source_representation_resolve_local(&destination, options->models_root, &verified, err);
        if (rc == YVEX_OK && strcmp(verified.digest, out->representation.digest))
            rc = distribution_refuse(err, YVEX_ERR_STATE, "source.distribution.import",
                                       "copied representation digest does not match source");
        if (rc == YVEX_OK) rc = representation_receipt(options->models_root, &verified, 1, err);
        if (rc != YVEX_OK) return rc;
    }
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

int yvex_source_import_local(const yvex_source_import_options *options,
                             yvex_source_import_result *out, yvex_error *err)
{
    yvex_source_import_options effective;
    yvex_source_representation_fact inspected;
    char lock_path[YVEX_PATH_CAP];
    int fd, rc;
    if (!options || !options->locator || !options->models_root || !out)
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.import", "source and root required");
    effective = *options;
    if (!effective.inspected || effective.inspected->directory) {
        rc = yvex_source_representation_verify_local(options->locator, options->models_root,
                                                      effective.inspected ? effective.inspected->digest : NULL,
                                                      &inspected, err);
        if (rc != YVEX_OK) return rc;
        effective.inspected = &inspected;
    }
    if (snprintf(lock_path, sizeof(lock_path), "%s/tmp/imports/%s.lock", options->models_root,
                 effective.inspected->digest) >= (int)sizeof(lock_path))
        return distribution_refuse(err, YVEX_ERR_BOUNDS, "source.import", "lock path is too long");
    rc = yvex_core_mkdir_parent(lock_path, "source.import", err);
    if (rc != YVEX_OK) return rc;
    fd = open(lock_path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return distribution_refuse(err, YVEX_ERR_IO, "source.import", "cannot open content lock");
    do { rc = flock(fd, LOCK_EX); } while (rc < 0 && errno == EINTR);
    if (rc == 0) rc = source_import_locked(&effective, out, err);
    else rc = distribution_refuse(err, YVEX_ERR_IO, "source.import", "cannot lock content");
    (void)close(fd);
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
    rc = representation_inspect_local(&source_locator, &source, err);
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
    rc = representation_inspect_local(&destination_locator, &copied, err);
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

static int selection_compare(const void *left, const void *right)
{
    return strcmp(*(const char *const *)left, *(const char *const *)right);
}

int yvex_source_selection_identity(const char *const *includes, size_t include_count,
                                    const char *const *excludes, size_t exclude_count,
                                    char out[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const char *patterns[256];
    size_t index, group;
    if (!out || include_count > 256u || exclude_count > 256u ||
        (include_count && !includes) || (exclude_count && !excludes))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.selection",
            "bounded selection patterns required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.source.selection.v1")) return YVEX_ERR_BOUNDS;
    for (group = 0u; group < 2u; ++group) {
        size_t count = group ? exclude_count : include_count;
        const char *const *input = group ? excludes : includes;
        if (!yvex_sha256_update_text(&hash, group ? "exclude" : "include")) return YVEX_ERR_BOUNDS;
        for (index = 0u; index < count; ++index) {
            if (!input[index]) return YVEX_ERR_INVALID_ARG;
            patterns[index] = input[index];
        }
        qsort(patterns, count, sizeof(patterns[0]), selection_compare);
        for (index = 0u; index < count; ++index) {
            if (index && !strcmp(patterns[index], patterns[index - 1u])) continue;
            if (!yvex_sha256_update_text(&hash, patterns[index])) return YVEX_ERR_BOUNDS;
        }
    }
    if (!yvex_sha256_final(&hash, digest)) return YVEX_ERR_BOUNDS;
    yvex_sha256_hex(digest, out);
    return YVEX_OK;
}

int yvex_source_acquisition_lock(const char *models_root, const char *repository,
                                  const char *revision, int *descriptor, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char identity[65], path[YVEX_PATH_CAP];
    int fd, rc;
    if (!models_root || !repository || !revision || !descriptor)
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.acquire", "exact acquisition identity required");
    *descriptor = -1;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.source.acquisition-lock.v1") ||
        !yvex_sha256_update_text(&hash, repository) || !yvex_sha256_update_text(&hash, revision) ||
        !yvex_sha256_final(&hash, digest)) return YVEX_ERR_STATE;
    yvex_sha256_hex(digest, identity);
    if (snprintf(path, sizeof(path), "%s/tmp/acquisitions/%s.lock", models_root, identity) >= (int)sizeof(path))
        return distribution_refuse(err, YVEX_ERR_BOUNDS, "source.acquire", "acquisition lock path exceeds bound");
    rc = yvex_core_mkdir_parent(path, "source.acquire", err);
    if (rc != YVEX_OK) return rc;
    fd = open(path, O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
    if (fd < 0) return distribution_refuse(err, YVEX_ERR_IO, "source.acquire", "cannot open acquisition lease");
    do { rc = flock(fd, LOCK_EX); } while (rc < 0 && errno == EINTR);
    if (rc != 0) {
        (void)close(fd);
        return distribution_refuse(err, YVEX_ERR_IO, "source.acquire", "cannot lock acquisition");
    }
    *descriptor = fd;
    return YVEX_OK;
}

static int acquisition_patterns_match(const char *record, const char *key,
                                       const char *const *expected, unsigned int count)
{
    const char *value = yvex_json_probe_field_value(record, key);
    yvex_json json;
    yvex_json_iter array;
    yvex_json_item item;
    unsigned int index, seen[256] = {0};
    if (!value || count > 256u) return 0;
    yvex_json_init(&json, value, strlen(value));
    if (!yvex_json_iter_begin(&json, &array, YVEX_JSON_COLLECTION_ARRAY)) return 0;
    while ((item = yvex_json_array_value(&array)) == YVEX_JSON_ITEM_READY) {
        char pattern[YVEX_PATH_CAP];
        int matched = 0;
        if (!yvex_json_string(&json, pattern, sizeof(pattern))) return 0;
        for (index = 0u; index < count; ++index)
            if (!strcmp(pattern, expected[index])) { seen[index] = 1u; matched = 1; }
        if (!matched) return 0;
    }
    for (index = 0u; index < count; ++index) if (!seen[index]) return 0;
    return item == YVEX_JSON_ITEM_END;
}

int yvex_source_acquisition_reopen(const char *record_path, const char *models_root,
                                    const char *repository, const char *revision,
                                    const char *const *includes, unsigned int include_count,
                                    const char *const *excludes, unsigned int exclude_count,
                                    yvex_source_representation_fact *out, yvex_error *err)
{
    char repo[256], pinned[128], status[64], path[YVEX_PATH_CAP], digest[65];
    char *record;
    size_t length;
    int rc = YVEX_ERR_STATE;
    if (!record_path || !models_root || !repository || !revision || !out)
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.reopen", "acquisition identity required");
    record = yvex_read_bounded_file(record_path, 4u * 1024u * 1024u, &length, err);
    if (!record) return YVEX_ERR_STATE;
    if (!yvex_json_probe_string_field(record, "status", status, sizeof(status)) ||
        (strcmp(status, "model-download-pass") && strcmp(status, "model-download-resume-pass")) ||
        !yvex_json_probe_string_field(record, "repo_id", repo, sizeof(repo)) || strcmp(repo, repository) ||
        !yvex_json_probe_string_field(record, "revision", pinned, sizeof(pinned)) || strcmp(pinned, revision) ||
        !yvex_json_probe_string_field(record, "local_source_dir", path, sizeof(path)) ||
        !yvex_json_probe_string_field(record, "source_payload_digest", digest, sizeof(digest)) ||
        !acquisition_patterns_match(record, "include_patterns", includes, include_count) ||
        !acquisition_patterns_match(record, "exclude_patterns", excludes, exclude_count)) goto done;
    rc = directory_receipt_hit(models_root, path, digest, out, err);
    if (rc == YVEX_OK) {
        (void)yvex_json_probe_string_field(record, "representation_format", out->format, sizeof(out->format));
        (void)yvex_json_probe_string_field(record, "representation_precision", out->precision, sizeof(out->precision));
    }
done:
    free(record);
    return rc;
}

static int source_directory_open(const char *path)
{
#ifdef __linux__
    struct open_how how = {0};
    how.flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
    how.resolve = RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
    return (int)syscall(SYS_openat2, AT_FDCWD, path, &how, sizeof(how));
#else
    (void)path;
    errno = ENOTSUP;
    return -1;
#endif
}

static int source_eviction_pins(const char *root, const char *path, const char *digest,
                                 yvex_artifact ***out, size_t *count, yvex_error *err)
{
    char receipt[YVEX_PATH_CAP], cache[YVEX_PATH_CAP];
    char *record;
    const char *value;
    size_t length;
    yvex_json json;
    yvex_json_iter array;
    yvex_json_item item;
    int rc = YVEX_ERR_STATE;
    *out = NULL;
    *count = 0u;
    if (snprintf(cache, sizeof(cache), "%s/cache/verification", root) >= (int)sizeof(cache) ||
        snprintf(receipt, sizeof(receipt), "%s/source-reopen/%s.json", cache, digest) >= (int)sizeof(receipt))
        return YVEX_ERR_BOUNDS;
    record = yvex_read_bounded_file(receipt, 16u * 1024u * 1024u, &length, err);
    if (!record) return YVEX_ERR_STATE;
    value = yvex_json_probe_field_value(record, "files");
    if (!value) goto done;
    yvex_json_init(&json, value, strlen(value));
    if (!yvex_json_iter_begin(&json, &array, YVEX_JSON_COLLECTION_ARRAY)) goto done;
    while ((item = yvex_json_array_value(&array)) == YVEX_JSON_ITEM_READY) {
        char relative[YVEX_PATH_CAP], member[YVEX_PATH_CAP], physical[YVEX_PATH_CAP], sha256[65];
        unsigned long long bytes;
        yvex_artifact_options options = {physical, 1, 0};
        yvex_artifact_reopen_lease lease;
        yvex_artifact **grown;
        if (*count >= 65536u || !receipt_member_parse(&json, relative, sha256, &bytes) ||
            !yvex_source_path_join(member, sizeof(member), path, relative) || !realpath(member, physical)) goto done;
        grown = realloc(*out, (*count + 1u) * sizeof(**out));
        if (!grown) { rc = YVEX_ERR_NOMEM; goto done; }
        *out = grown;
        (*out)[*count] = NULL;
        rc = yvex_artifact_open(&(*out)[*count], &options, err);
        if (rc != YVEX_OK) goto done;
        (*count)++;
        rc = yvex_artifact_reopen_lease_check((*out)[*count - 1u], sha256, cache, &lease, err);
        if (rc != YVEX_OK || !lease.verified || lease.snapshot.size != bytes) { rc = YVEX_ERR_STATE; goto done; }
        rc = yvex_artifact_pin_exclusive((*out)[*count - 1u], err);
        if (rc != YVEX_OK) goto done;
    }
    rc = item == YVEX_JSON_ITEM_END && *count ? YVEX_OK : YVEX_ERR_STATE;
done:
    free(record);
    return rc;
}

/* Walk only the directory detached into a private operation directory; never follow links. */
static int source_erase_at(int parent, const char *name, unsigned int depth,
                            unsigned long long *allocated)
{
    struct stat status;
    int fd, rc = 0;
    DIR *directory;
    struct dirent *entry;
    if (depth > 64u || fstatat(parent, name, &status, AT_SYMLINK_NOFOLLOW) != 0) return -1;
    if (!S_ISDIR(status.st_mode)) {
        if (unlinkat(parent, name, 0) != 0) return -1;
        if (status.st_nlink == 1u) *allocated += (unsigned long long)status.st_blocks * 512ull;
        return 0;
    }
    fd = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return -1;
    directory = fdopendir(fd);
    if (!directory) { (void)close(fd); return -1; }
    while (rc == 0) {
        errno = 0;
        entry = readdir(directory);
        if (!entry) { if (errno) rc = -1; break; }
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        rc = source_erase_at(fd, entry->d_name, depth + 1u, allocated);
    }
    if (closedir(directory) != 0) rc = -1;
    if (rc == 0) rc = unlinkat(parent, name, AT_REMOVEDIR);
    if (rc == 0) *allocated += (unsigned long long)status.st_blocks * 512ull;
    return rc;
}

int yvex_source_evict_local(const char *models_root, const char *repository, const char *revision,
                             const char *path, const char *digest, int dry_run,
                             yvex_source_eviction_result *out, yvex_error *err)
{
    char expected[YVEX_PATH_CAP], operation[YVEX_PATH_CAP], parent[YVEX_PATH_CAP], *leaf;
    yvex_source_representation_fact verified;
    yvex_artifact **pins = NULL;
    size_t count = 0u, index, prefix;
    struct stat opened, named;
    int lock = -1, root_fd = -1, parent_fd = -1, temporary_fd = -1, rc;
    if (!out || !models_root || !repository || !revision || !path || !yvex_sha256_hex_is_valid(digest) ||
        !yvex_source_provider_path(expected, sizeof(expected), models_root, repository, revision))
        return distribution_refuse(err, YVEX_ERR_INVALID_ARG, "source.evict",
                                   "exact provider source identity required");
    memset(out, 0, sizeof(*out));
    prefix = strlen(expected);
    if (strncmp(path, expected, prefix) ||
        (path[prefix] && (path[prefix] != '-' || !yvex_sha256_hex_is_valid(path + prefix + 1u))))
        return distribution_refuse(err, YVEX_ERR_STATE, "source.evict",
                                   "path is not the canonical provider acquisition");
    rc = yvex_source_acquisition_lock(models_root, repository, revision, &lock, err);
    if (rc != YVEX_OK) return rc;
    if (lstat(path, &named) != 0 && errno == ENOENT) { rc = YVEX_OK; goto done; }
    root_fd = source_directory_open(path);
    if (root_fd < 0 || flock(root_fd, LOCK_EX | LOCK_NB) != 0) {
        rc = distribution_refuse(err, YVEX_ERR_STATE, "source.evict", "active source operation prevents eviction");
        goto done;
    }
    out->local = 1;
    rc = directory_receipt_hit(models_root, path, digest, &verified, err);
    if (rc == YVEX_OK) rc = source_eviction_pins(models_root, path, digest, &pins, &count, err);
    if (rc != YVEX_OK) goto done;
    out->logical_bytes = verified.size_bytes;
    if (dry_run) goto done;
    if (snprintf(operation, sizeof(operation), "%s/tmp/evictions/%s.XXXXXX", models_root, digest) >=
        (int)sizeof(operation)) { rc = YVEX_ERR_BOUNDS; goto done; }
    rc = yvex_core_mkdir_parent(operation, "source.evict", err);
    if (rc != YVEX_OK || !mkdtemp(operation)) { rc = YVEX_ERR_IO; goto done; }
    (void)distribution_copy(out->pending_path, sizeof(out->pending_path), operation, err, "source.evict");
    (void)distribution_copy(parent, sizeof(parent), path, err, "source.evict");
    leaf = strrchr(parent, '/');
    *leaf++ = '\0';
    parent_fd = source_directory_open(parent);
    temporary_fd = source_directory_open(operation);
    if (parent_fd < 0 || temporary_fd < 0 || fstat(root_fd, &opened) != 0 ||
        fstatat(parent_fd, leaf, &named, AT_SYMLINK_NOFOLLOW) != 0 ||
        opened.st_dev != named.st_dev || opened.st_ino != named.st_ino ||
        renameat(parent_fd, leaf, temporary_fd, "payload") != 0) { rc = YVEX_ERR_STATE; goto done; }
    out->changed = 1;
    out->local = 0;
    if (fsync(parent_fd) != 0 || fsync(temporary_fd) != 0 ||
        source_erase_at(temporary_fd, "payload", 0u, &out->allocated_bytes) != 0 ||
        rmdir(operation) != 0) { rc = YVEX_ERR_IO; goto done; }
    out->pending_path[0] = '\0';
done:
    for (index = 0u; index < count; ++index) yvex_artifact_close(pins[index]);
    free(pins);
    if (root_fd >= 0) (void)close(root_fd);
    if (parent_fd >= 0) (void)close(parent_fd);
    if (temporary_fd >= 0) (void)close(temporary_fd);
    if (lock >= 0) (void)close(lock);
    if (rc != YVEX_OK && out->changed)
        yvex_error_setf(err, (yvex_status)rc, "source.evict",
                        "source detached; cleanup remains at %s", out->pending_path);
    else if (rc != YVEX_OK && yvex_error_code(err) == YVEX_OK)
        yvex_error_set(err, (yvex_status)rc, "source.evict", "source verification or exclusive ownership unavailable");
    return rc;
}
