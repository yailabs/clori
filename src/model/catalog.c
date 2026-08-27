/*
 * Compose local source-acquisition and package records without opening model payloads or engines.
 *
 * Registry membership, recorded source verification, package readiness, and engine residency stay
 * separate so list projections cannot silently promote one lifecycle stage into another.
 */

#define _POSIX_C_SOURCE 200809L

#include <yvex/model.h>

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/internal/core.h>
#include <yvex/registry.h>

#define LOCAL_CATALOG_FILE_CAP (1024u * 1024u)
#define LOCAL_CATALOG_ENTRY_CAP 4096u

struct yvex_local_model_catalog {
    yvex_local_model *entries;
    unsigned long long count;
    unsigned long long capacity;
};

static int local_refuse(yvex_error *err, yvex_status status, const char *message)
{
    yvex_error_set(err, status, "local_model_catalog", message);
    return status;
}

static void local_copy(char *out, size_t capacity, const char *value)
{
    if (!out || !capacity) return;
    snprintf(out, capacity, "%s", value ? value : "");
}

static int local_catalog_reserve(yvex_local_model_catalog *catalog,
                                 unsigned long long required)
{
    yvex_local_model *entries;
    unsigned long long capacity;

    if (required <= catalog->capacity) return 1;
    capacity = catalog->capacity ? catalog->capacity * 2u : 32u;
    while (capacity < required) capacity *= 2u;
    if (capacity > LOCAL_CATALOG_ENTRY_CAP) capacity = LOCAL_CATALOG_ENTRY_CAP;
    if (capacity < required) return 0;
    entries = realloc(catalog->entries, (size_t)capacity * sizeof(*entries));
    if (!entries) return 0;
    catalog->entries = entries;
    catalog->capacity = capacity;
    return 1;
}

static yvex_local_model *local_catalog_add(yvex_local_model_catalog *catalog)
{
    yvex_local_model *entry;

    if (!local_catalog_reserve(catalog, catalog->count + 1u)) return NULL;
    entry = &catalog->entries[catalog->count++];
    memset(entry, 0, sizeof(*entry));
    return entry;
}

static int local_json_u64(const char *text, const char *key, unsigned long long *out)
{
    const char *value = yvex_json_probe_field_value(text, key);
    yvex_json json;

    if (!value) return 0;
    yvex_json_init(&json, value, strlen(value));
    return yvex_json_u64(&json, out);
}

static int local_json_bool(const char *text, const char *key, int *out)
{
    const char *value = yvex_json_probe_field_value(text, key);
    yvex_json json;

    if (!value) return 0;
    yvex_json_init(&json, value, strlen(value));
    return yvex_json_bool(&json, out);
}

static int local_revision_immutable(const char *revision)
{
    size_t index;
    size_t length;

    if (!revision) return 0;
    length = strlen(revision);
    if (length != 40u && length != 64u) return 0;
    for (index = 0u; index < length; ++index)
        if (!isxdigit((unsigned char)revision[index])) return 0;
    return 1;
}

static int local_source_record_add(yvex_local_model_catalog *catalog,
                                   const char *path,
                                   yvex_error *err)
{
    char *text;
    size_t length;
    char schema[64];
    char status[64];
    char target[YVEX_REMOTE_NAME_CAP];
    char family[YVEX_REMOTE_FAMILY_CAP];
    char provider[YVEX_ACCOUNT_PROVIDER_CAP];
    char repository[YVEX_REMOTE_REPOSITORY_CAP];
    char revision[YVEX_REMOTE_REVISION_CAP];
    char source_path[YVEX_PATH_CAP];
    unsigned long long size = 0u;
    unsigned long long safetensors = 0u;
    unsigned long long gguf = 0u;
    int upstream_verified = 0;
    int payload_verified = 0;
    struct stat source;
    yvex_local_model *entry;

    text = yvex_read_bounded_file(path, LOCAL_CATALOG_FILE_CAP, &length, err);
    if (!text) return yvex_error_code(err);
    memset(schema, 0, sizeof(schema));
    memset(status, 0, sizeof(status));
    memset(target, 0, sizeof(target));
    memset(family, 0, sizeof(family));
    memset(provider, 0, sizeof(provider));
    memset(repository, 0, sizeof(repository));
    memset(revision, 0, sizeof(revision));
    memset(source_path, 0, sizeof(source_path));
    (void)yvex_json_probe_string_field(text, "schema", schema, sizeof(schema));
    if (strcmp(schema, "yvex.model_download.registry.v1") != 0) {
        free(text);
        return YVEX_OK;
    }
    if (!yvex_json_probe_string_field(text, "target_id", target, sizeof(target)) ||
        !yvex_json_probe_string_field(text, "family", family, sizeof(family)) ||
        !yvex_json_probe_string_field(text, "repo_id", repository, sizeof(repository)) ||
        !yvex_json_probe_string_field(text, "revision", revision, sizeof(revision)) ||
        !yvex_json_probe_string_field(text, "local_source_dir", source_path,
                                      sizeof(source_path))) {
        free(text);
        return local_refuse(err, YVEX_ERR_FORMAT, "local acquisition record is incomplete");
    }
    (void)yvex_json_probe_string_field(text, "status", status, sizeof(status));
    (void)yvex_json_probe_string_field(text, "provider", provider, sizeof(provider));
    (void)local_json_u64(text, "total_regular_file_bytes", &size);
    (void)local_json_u64(text, "safetensors_count", &safetensors);
    (void)local_json_u64(text, "gguf_count", &gguf);
    (void)local_json_bool(text, "upstream_identity_verified", &upstream_verified);
    (void)local_json_bool(text, "payload_hash_verified", &payload_verified);
    free(text);
    entry = local_catalog_add(catalog);
    if (!entry) return local_refuse(err, YVEX_ERR_NOMEM, "local catalog allocation failed");
    entry->kind = YVEX_LOCAL_MODEL_ACQUIRED_SOURCE;
    local_copy(entry->name, sizeof(entry->name), target);
    local_copy(entry->family, sizeof(entry->family), family);
    local_copy(entry->provider, sizeof(entry->provider), provider);
    local_copy(entry->repository, sizeof(entry->repository), repository);
    local_copy(entry->revision, sizeof(entry->revision), revision);
    local_copy(entry->path, sizeof(entry->path), source_path);
    local_copy(entry->representation, sizeof(entry->representation),
               safetensors && gguf ? "mixed-provider-source"
                                   : (safetensors ? "safetensors-source"
                                                  : (gguf ? "gguf" : "provider-source")));
    entry->size_bytes = size;
    entry->size_known = size != 0u;
    if (stat(source_path, &source) != 0 || !S_ISDIR(source.st_mode)) {
        local_copy(entry->package_state, sizeof(entry->package_state), "source-missing");
        local_copy(entry->blocker, sizeof(entry->blocker), "recorded source directory is unavailable");
    } else if (strcmp(status, "model-download-pass") == 0) {
        local_copy(entry->package_state, sizeof(entry->package_state), "source-acquired");
    } else {
        local_copy(entry->package_state, sizeof(entry->package_state), "source-partial");
        local_copy(entry->blocker, sizeof(entry->blocker), "acquisition did not complete successfully");
    }
    if (payload_verified)
        local_copy(entry->verification_state, sizeof(entry->verification_state), "payload-verified");
    else if (upstream_verified)
        local_copy(entry->verification_state, sizeof(entry->verification_state), "revision-verified");
    else
        local_copy(entry->verification_state, sizeof(entry->verification_state), "recorded-unverified");
    if (!local_revision_immutable(revision)) {
        local_copy(entry->verification_state, sizeof(entry->verification_state), "moving-reference");
        local_copy(entry->blocker, sizeof(entry->blocker),
                   "resolve an immutable provider revision before package preparation");
    } else if (strcmp(entry->package_state, "source-acquired") == 0) {
        local_copy(entry->blocker, sizeof(entry->blocker),
                   safetensors && gguf
                       ? "select one acquired representation before package preparation"
                       : (safetensors
                              ? "compile or prepare an admitted YVEX package"
                              : (gguf ? "inspect GGUF compatibility, then admit or repack"
                                      : "classify acquired files before package preparation")));
    }
    return YVEX_OK;
}

static int local_source_manifest_add(yvex_local_model_catalog *catalog,
                                     const char *path,
                                     yvex_error *err)
{
    char *text;
    size_t length;
    char schema[64] = "";
    char repository[YVEX_REMOTE_REPOSITORY_CAP] = "";
    char revision[YVEX_REMOTE_REVISION_CAP] = "";
    char source_path[YVEX_PATH_CAP];
    const char *name;
    char *slash;
    unsigned long long size = 0u;
    int complete = 0;
    struct stat source;
    yvex_local_model *entry;

    text = yvex_read_bounded_file(path, LOCAL_CATALOG_FILE_CAP, &length, err);
    if (!text) return yvex_error_code(err);
    (void)yvex_json_probe_string_field(text, "schema", schema, sizeof(schema));
    if (strcmp(schema, "yvex.source-acquisition.v1") != 0) {
        free(text);
        return YVEX_OK;
    }
    if (!yvex_json_probe_string_field(text, "repository", repository, sizeof(repository)) ||
        !yvex_json_probe_string_field(text, "revision", revision, sizeof(revision)) ||
        !local_revision_immutable(revision) ||
        !local_json_bool(text, "acquisition_complete", &complete)) {
        free(text);
        return local_refuse(err, YVEX_ERR_FORMAT, "source acquisition provenance is incomplete");
    }
    (void)local_json_u64(text, "source_bytes", &size);
    free(text);
    local_copy(source_path, sizeof(source_path), path);
    slash = strrchr(source_path, '/');
    if (!slash || slash == source_path) return local_refuse(err, YVEX_ERR_FORMAT,
                                                             "source provenance path is invalid");
    *slash = '\0';
    entry = local_catalog_add(catalog);
    if (!entry) return local_refuse(err, YVEX_ERR_NOMEM, "local catalog allocation failed");
    entry->kind = YVEX_LOCAL_MODEL_ACQUIRED_SOURCE;
    name = strrchr(repository, '/');
    local_copy(entry->name, sizeof(entry->name), name ? name + 1 : repository);
    local_copy(entry->provider, sizeof(entry->provider), "huggingface");
    local_copy(entry->repository, sizeof(entry->repository), repository);
    local_copy(entry->revision, sizeof(entry->revision), revision);
    local_copy(entry->representation, sizeof(entry->representation), "safetensors-source");
    local_copy(entry->path, sizeof(entry->path), source_path);
    entry->size_bytes = size;
    entry->size_known = size != 0u;
    if (lstat(source_path, &source) != 0 || !S_ISDIR(source.st_mode) || S_ISLNK(source.st_mode)) {
        local_copy(entry->package_state, sizeof(entry->package_state), "source-missing");
        local_copy(entry->verification_state, sizeof(entry->verification_state), "recorded-unverified");
        local_copy(entry->blocker, sizeof(entry->blocker), "recorded source directory is unavailable");
    } else if (complete) {
        local_copy(entry->package_state, sizeof(entry->package_state), "source-acquired");
        local_copy(entry->verification_state, sizeof(entry->verification_state), "payload-verified");
        local_copy(entry->blocker, sizeof(entry->blocker),
                   "compile or prepare an admitted YVEX package");
    } else {
        local_copy(entry->package_state, sizeof(entry->package_state), "source-partial");
        local_copy(entry->verification_state, sizeof(entry->verification_state), "revision-verified");
        local_copy(entry->blocker, sizeof(entry->blocker), "source acquisition is incomplete");
    }
    return YVEX_OK;
}

static int local_name_ends_with(const char *name, const char *suffix)
{
    size_t name_length = strlen(name);
    size_t suffix_length = strlen(suffix);
    return name_length >= suffix_length &&
           strcmp(name + name_length - suffix_length, suffix) == 0;
}

static int local_scan_acquisitions(yvex_local_model_catalog *catalog,
                                   const char *directory,
                                   unsigned int depth,
                                   int source_manifest,
                                   yvex_error *err)
{
    DIR *dir;
    struct dirent *item;
    int result = YVEX_OK;

    if (depth > (source_manifest ? 5u : 3u)) return YVEX_OK;
    dir = opendir(directory);
    if (!dir) return errno == ENOENT ? YVEX_OK : local_refuse(err, YVEX_ERR_IO,
                                                               "cannot scan acquisition registry");
    while ((item = readdir(dir)) != NULL) {
        char path[YVEX_PATH_CAP];
        struct stat status;
        int written;

        if (strcmp(item->d_name, ".") == 0 || strcmp(item->d_name, "..") == 0) continue;
        written = snprintf(path, sizeof(path), "%s/%s", directory, item->d_name);
        if (written < 0 || (size_t)written >= sizeof(path) || lstat(path, &status) != 0) {
            result = local_refuse(err, YVEX_ERR_IO, "cannot inspect acquisition registry path");
            break;
        }
        if (S_ISLNK(status.st_mode)) continue;
        if (S_ISDIR(status.st_mode)) {
            result = local_scan_acquisitions(catalog, path, depth + 1u,
                                             source_manifest, err);
        } else if (S_ISREG(status.st_mode) &&
                   ((source_manifest && strcmp(item->d_name, "yvex-source-acquisition.json") == 0) ||
                    (!source_manifest && local_name_ends_with(item->d_name, ".download.json")))) {
            result = source_manifest ? local_source_manifest_add(catalog, path, err)
                                     : local_source_record_add(catalog, path, err);
        }
        if (result != YVEX_OK || catalog->count >= LOCAL_CATALOG_ENTRY_CAP) break;
    }
    (void)closedir(dir);
    return result;
}

static void local_package_provenance(yvex_local_model *entry,
                                     const yvex_model_registry_entry *registered)
{
    const char *root = registered->runtime_installation;
    char path[YVEX_PATH_CAP];
    char repository[YVEX_REMOTE_REPOSITORY_CAP] = "";
    char revision[YVEX_REMOTE_REVISION_CAP] = "";
    char *text;
    size_t length;
    struct stat status;
    yvex_error ignored;
    int written;

    if (!root || !root[0]) return;
    written = snprintf(path, sizeof(path), "%s/repository.json", root);
    if (written < 0 || (size_t)written >= sizeof(path) || lstat(path, &status) != 0 ||
        !S_ISREG(status.st_mode) || S_ISLNK(status.st_mode))
        return;
    yvex_error_clear(&ignored);
    text = yvex_read_bounded_file(path, 16384u, &length, &ignored);
    if (!text) return;
    if (yvex_json_probe_string_field(text, "repository", repository, sizeof(repository)) &&
        yvex_json_probe_string_field(text, "resolved_revision", revision, sizeof(revision)) &&
        local_revision_immutable(revision)) {
        local_copy(entry->repository, sizeof(entry->repository), repository);
        local_copy(entry->revision, sizeof(entry->revision), revision);
    }
    free(text);
}

static void local_package_add(yvex_local_model_catalog *catalog,
                              const yvex_model_registry_entry *registered)
{
    yvex_local_model *entry = local_catalog_add(catalog);
    yvex_error startup_error;
    int startup_ready;

    if (!entry) return;
    yvex_error_clear(&startup_error);
    startup_ready = yvex_model_registry_startup_validate(registered, &startup_error) == YVEX_OK;
    entry->kind = YVEX_LOCAL_MODEL_PACKAGE;
    local_copy(entry->name, sizeof(entry->name), registered->alias);
    local_copy(entry->family, sizeof(entry->family), registered->family);
    local_copy(entry->path, sizeof(entry->path),
               registered->runtime_installation && registered->runtime_installation[0]
                   ? registered->runtime_installation
                   : registered->path);
    if (registered->format && registered->format[0])
        local_copy(entry->representation, sizeof(entry->representation), registered->format);
    else if (registered->artifact_class)
        local_copy(entry->representation, sizeof(entry->representation), registered->artifact_class);
    local_copy(entry->backend, sizeof(entry->backend), registered->runtime_backend);
    local_copy(entry->package_state, sizeof(entry->package_state),
               startup_ready ? "package-ready" : "artifact-registered");
    local_copy(entry->verification_state, sizeof(entry->verification_state),
               registered->sha256 && registered->sha256[0]
                   ? "identity-recorded"
                   : "identity-missing");
    if (!startup_ready)
        local_copy(entry->blocker, sizeof(entry->blocker), yvex_error_message(&startup_error));
    entry->size_bytes = registered->file_size;
    entry->size_known = registered->file_size != 0u;
    entry->package_ready = startup_ready;
    local_package_provenance(entry, registered);
}

static int local_scan_packages(yvex_local_model_catalog *catalog,
                               const char *registry_path,
                               yvex_error *err)
{
    yvex_model_registry_options options;
    yvex_model_registry *registry = NULL;
    unsigned long long index;
    int rc;

    if (!registry_path || !registry_path[0]) return YVEX_OK;
    memset(&options, 0, sizeof(options));
    options.registry_path = registry_path;
    options.create_if_missing = 0;
    rc = yvex_model_registry_open(&registry, &options, err);
    if (rc != YVEX_OK) {
        if (access(registry_path, F_OK) != 0) {
            yvex_error_clear(err);
            return YVEX_OK;
        }
        return rc;
    }
    for (index = 0u; index < yvex_model_registry_count(registry); ++index) {
        const yvex_model_registry_entry *entry = yvex_model_registry_at(registry, index);
        if (catalog->count >= LOCAL_CATALOG_ENTRY_CAP) {
            yvex_model_registry_close(registry);
            return local_refuse(err, YVEX_ERR_BOUNDS, "local catalog entry limit exceeded");
        }
        local_package_add(catalog, entry);
    }
    yvex_model_registry_close(registry);
    return YVEX_OK;
}

static int local_compare(const void *left, const void *right)
{
    const yvex_local_model *a = left;
    const yvex_local_model *b = right;
    int family = strcmp(a->family, b->family);
    int name;

    if (family) return family;
    name = strcmp(a->name, b->name);
    if (name) return name;
    return (int)a->kind - (int)b->kind;
}

static void local_reconcile_entries(yvex_local_model_catalog *catalog)
{
    unsigned long long source_index;
    unsigned long long package_index;

    for (source_index = 0u; source_index < catalog->count; ++source_index) {
        yvex_local_model *source = &catalog->entries[source_index];
        if (source->kind != YVEX_LOCAL_MODEL_ACQUIRED_SOURCE || !source->repository[0] ||
            !source->revision[0])
            continue;
        for (package_index = 0u; package_index < catalog->count; ++package_index) {
            yvex_local_model *package = &catalog->entries[package_index];
            if (package->kind != YVEX_LOCAL_MODEL_PACKAGE ||
                strcmp(source->repository, package->repository) != 0 ||
                strcmp(source->revision, package->revision) != 0)
                continue;
            if (!source->family[0])
                local_copy(source->family, sizeof(source->family), package->family);
            if (!package->provider[0])
                local_copy(package->provider, sizeof(package->provider), source->provider);
            if (package->package_ready) source->blocker[0] = '\0';
        }
    }
}

int yvex_local_model_catalog_open(yvex_local_model_catalog **out,
                                  const yvex_local_catalog_options *options,
                                  yvex_error *err)
{
    yvex_local_model_catalog *catalog;
    yvex_operator_paths operator_paths;
    yvex_paths paths;
    char registry_path[YVEX_PATH_CAP];
    const char *explicit_root = options ? options->models_root : NULL;
    const char *explicit_registry = options ? options->registry_path : NULL;
    int rc;

    if (!out) return local_refuse(err, YVEX_ERR_INVALID_ARG, "catalog output is required");
    *out = NULL;
    yvex_error_clear(err);
    rc = yvex_paths_default(&paths, err);
    if (rc != YVEX_OK) return rc;
    rc = yvex_operator_paths_resolve(&paths, explicit_root, &operator_paths, err);
    if (rc != YVEX_OK) return rc;
    if (explicit_registry && explicit_registry[0])
        local_copy(registry_path, sizeof(registry_path), explicit_registry);
    else {
        rc = yvex_model_registry_default_path(registry_path, sizeof(registry_path), err);
        if (rc != YVEX_OK) return rc;
    }
    catalog = calloc(1u, sizeof(*catalog));
    if (!catalog) return local_refuse(err, YVEX_ERR_NOMEM, "local catalog allocation failed");
    rc = local_scan_acquisitions(catalog, operator_paths.registry_root, 0u, 0, err);
    if (rc == YVEX_OK)
        rc = local_scan_acquisitions(catalog, operator_paths.hf_root, 0u, 1, err);
    if (rc == YVEX_OK) rc = local_scan_packages(catalog, registry_path, err);
    if (rc != YVEX_OK) {
        yvex_local_model_catalog_close(catalog);
        return rc;
    }
    local_reconcile_entries(catalog);
    qsort(catalog->entries, (size_t)catalog->count, sizeof(*catalog->entries), local_compare);
    *out = catalog;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_local_model_catalog_close(yvex_local_model_catalog *catalog)
{
    if (!catalog) return;
    free(catalog->entries);
    free(catalog);
}

unsigned long long yvex_local_model_catalog_count(const yvex_local_model_catalog *catalog)
{
    return catalog ? catalog->count : 0u;
}

const yvex_local_model *yvex_local_model_catalog_at(const yvex_local_model_catalog *catalog,
                                                    unsigned long long index)
{
    return catalog && index < catalog->count ? &catalog->entries[index] : NULL;
}
