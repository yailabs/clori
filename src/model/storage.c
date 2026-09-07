/* Explicit storage inspection projects catalog ownership and filesystem allocation. */
#define _XOPEN_SOURCE 700
#include <yvex/internal/model_lifecycle.h>
#include <yvex/internal/core.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { dev_t device; ino_t inode; } storage_inode;
typedef struct {
    storage_inode *inodes;
    size_t count, capacity;
    yvex_model_storage_report *report;
} storage_inspection;

static int storage_error(yvex_error *err, yvex_status status, const char *message)
{
    yvex_error_set(err, status, "model.storage", message);
    return status;
}

static int storage_count(storage_inspection *inspection, yvex_model_storage_row *row,
                          const struct stat *status, int payload, yvex_error *err)
{
    size_t index;
    if (payload) {
        row->logical_bytes += (unsigned long long)status->st_size;
        row->files++;
    }
    for (index = 0u; index < inspection->count; ++index) {
        if (inspection->inodes[index].device == status->st_dev &&
            inspection->inodes[index].inode == status->st_ino) {
            if (payload) row->shared_files++;
            return YVEX_OK;
        }
    }
    if (inspection->count == inspection->capacity) {
        size_t capacity = inspection->capacity ? inspection->capacity * 2u : 128u;
        storage_inode *grown = realloc(inspection->inodes, capacity * sizeof(*grown));
        if (!grown) return storage_error(err, YVEX_ERR_NOMEM, "inode accounting allocation failed");
        inspection->inodes = grown;
        inspection->capacity = capacity;
    }
    inspection->inodes[inspection->count++] = (storage_inode){status->st_dev, status->st_ino};
    row->allocated_bytes += (unsigned long long)status->st_blocks * 512ull;
    return YVEX_OK;
}

static int storage_walk(storage_inspection *inspection, yvex_model_storage_row *row,
                         const char *path, unsigned depth, yvex_error *err)
{
    struct stat status;
    DIR *directory;
    struct dirent *entry;
    int rc;
    if (depth > 64u) return storage_error(err, YVEX_ERR_BOUNDS, "storage directory depth exceeds bound");
    if (lstat(path, &status) != 0) {
        if (errno == ENOENT) return YVEX_OK;
        return storage_error(err, YVEX_ERR_IO, "cannot inspect storage path");
    }
    if (!depth) row->exists = 1;
    if (S_ISLNK(status.st_mode)) {
        rc = storage_count(inspection, row, &status, 0, err);
        if (rc != YVEX_OK) return rc;
        if (stat(path, &status) != 0) {
            if (errno == ENOENT) return YVEX_OK;
            return storage_error(err, YVEX_ERR_IO, "cannot inspect storage link target");
        }
        /* File aliases refer to physical blobs; never traverse directory links. */
        return S_ISREG(status.st_mode) ? storage_count(inspection, row, &status, 1, err) : YVEX_OK;
    }
    rc = storage_count(inspection, row, &status, S_ISREG(status.st_mode), err);
    if (rc != YVEX_OK || !S_ISDIR(status.st_mode)) return rc;
    row->recursive = 1;
    directory = opendir(path);
    if (!directory) return storage_error(err, YVEX_ERR_IO, "cannot read storage directory");
    while (rc == YVEX_OK) {
        char child[YVEX_PATH_CAP];
        errno = 0;
        entry = readdir(directory);
        if (!entry) {
            if (errno) rc = storage_error(err, YVEX_ERR_IO, "storage directory changed during inspection");
            break;
        }
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        if (snprintf(child, sizeof(child), "%s/%s", path, entry->d_name) >= (int)sizeof(child)) {
            rc = storage_error(err, YVEX_ERR_BOUNDS, "storage path exceeds bound");
            break;
        }
        rc = storage_walk(inspection, row, child, depth + 1u, err);
    }
    closedir(directory);
    return rc;
}

static int storage_add(storage_inspection *inspection, const char *path,
                        const char *role, int attributable, yvex_error *err)
{
    yvex_model_storage_report *report = inspection->report;
    yvex_model_storage_row *rows, *row;
    size_t index;
    int rc;
    if (!path || !path[0]) return YVEX_OK;
    for (index = 0u; index < report->count; ++index)
        if (!strcmp(report->rows[index].path, path)) return YVEX_OK;
    rows = realloc(report->rows, (report->count + 1u) * sizeof(*rows));
    if (!rows) return storage_error(err, YVEX_ERR_NOMEM, "storage rows allocation failed");
    report->rows = rows;
    row = &rows[report->count++];
    memset(row, 0, sizeof(*row));
    yvex_core_text_copy(row->path, sizeof(row->path), path);
    yvex_core_text_copy(row->role, sizeof(row->role), role);
    row->attributable = attributable;
    rc = storage_walk(inspection, row, path, 0u, err);
    report->logical_bytes += row->logical_bytes;
    report->allocated_bytes += row->allocated_bytes;
    return rc;
}

static int storage_child(storage_inspection *inspection, const char *root,
                          const char *suffix, const char *role, yvex_error *err)
{
    char path[YVEX_PATH_CAP];
    if (snprintf(path, sizeof(path), "%s/%s", root, suffix) >= (int)sizeof(path))
        return storage_error(err, YVEX_ERR_BOUNDS, "storage root exceeds bound");
    return storage_add(inspection, path, role, 0, err);
}

int yvex_model_storage_inspect(const yvex_model_library *library,
                                unsigned long long model_index,
                                const char *models_root, int include_caches,
                                yvex_model_storage_report *out, yvex_error *err)
{
    storage_inspection inspection = {0};
    unsigned long long index;
    int rc = YVEX_OK;
    if (!library || !models_root || !out || model_index >= yvex_model_library_count(library))
        return storage_error(err, YVEX_ERR_INVALID_ARG, "known model and configured root required");
    memset(out, 0, sizeof(*out));
    inspection.report = out;
    for (index = 0u; rc == YVEX_OK && index < yvex_model_library_artifact_count(library, model_index); ++index) {
        const yvex_model_artifact_fact *artifact = yvex_model_library_artifact_at(library, model_index, index);
        int managed = yvex_model_artifact_is_managed(library, model_index, artifact, models_root);
        rc = storage_add(&inspection, artifact->path,
                          managed ? "managed-representation" : "external-representation", 1, err);
    }
    for (index = 0u; rc == YVEX_OK && index < yvex_model_library_source_count(library, model_index); ++index) {
        const yvex_local_source_record *source = yvex_model_library_source_at(library, model_index, index);
        rc = storage_add(&inspection, source->path,
                          !strcmp(source->storage_kind, "external") ? "external-source" : "managed-source", 1, err);
    }
    if (include_caches && rc == YVEX_OK) {
        yvex_paths paths;
        const char *home = getenv("HF_HOME"), *hub = getenv("HF_HUB_CACHE"), *xet = getenv("HF_XET_CACHE");
        char default_home[YVEX_PATH_CAP];
        rc = storage_child(&inspection, models_root, "cache/hf", "provider-acquisition-cache", err);
        if (rc == YVEX_OK)
            rc = storage_child(&inspection, models_root, "cache/provider-hub", "provider-hub-xet-cache", err);
        if (rc == YVEX_OK)
            rc = storage_child(&inspection, models_root, "tmp", "temporary-preparation-acquisition", err);
        if (rc == YVEX_OK) rc = yvex_paths_default(&paths, err);
        if (rc == YVEX_OK) rc = storage_add(&inspection, paths.cache_dir, "runtime-cache", 0, err);
        if (!home || !home[0]) {
            const char *xdg = getenv("XDG_CACHE_HOME"), *user_home = getenv("HOME");
            int length = xdg && xdg[0] ? snprintf(default_home, sizeof(default_home), "%s/huggingface", xdg) :
                snprintf(default_home, sizeof(default_home), "%s/.cache/huggingface", user_home ? user_home : "");
            if (length >= (int)sizeof(default_home))
                rc = storage_error(err, YVEX_ERR_BOUNDS, "HF cache root exceeds bound");
            home = default_home;
        }
        /* Never inspect HF_HOME itself: its authentication files are not storage evidence. */
        if (rc == YVEX_OK) rc = hub && hub[0] ? storage_add(&inspection, hub, "provider-hub-cache", 0, err) :
            storage_child(&inspection, home, "hub", "provider-hub-cache", err);
        if (rc == YVEX_OK) rc = xet && xet[0] ? storage_add(&inspection, xet, "provider-xet-cache", 0, err) :
            storage_child(&inspection, home, "xet", "provider-xet-cache", err);
    }
    free(inspection.inodes);
    if (rc != YVEX_OK) yvex_model_storage_report_free(out);
    return rc;
}

void yvex_model_storage_report_free(yvex_model_storage_report *report)
{
    if (!report) return;
    free(report->rows);
    memset(report, 0, sizeof(*report));
}
