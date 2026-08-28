/*
 * Compose local source-acquisition and package records without opening model payloads or engines.
 *
 * Registry membership, recorded source verification, package readiness, and engine residency stay
 * separate so list projections cannot silently promote one lifecycle stage into another.
 */

#define _POSIX_C_SOURCE 200809L

#include <yvex/catalog.h>

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

struct yvex_local_catalog {
    yvex_local_source_record *sources;
    unsigned long long source_count;
    unsigned long long source_capacity;
    yvex_local_package_record *packages;
    unsigned long long package_count;
    unsigned long long package_capacity;
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

static int local_source_reserve(yvex_local_catalog *catalog,
                                unsigned long long required)
{
    yvex_local_source_record *entries;
    unsigned long long capacity;

    if (required <= catalog->source_capacity) return 1;
    capacity = catalog->source_capacity ? catalog->source_capacity * 2u : 32u;
    while (capacity < required) capacity *= 2u;
    if (capacity > LOCAL_CATALOG_ENTRY_CAP) capacity = LOCAL_CATALOG_ENTRY_CAP;
    if (capacity < required) return 0;
    entries = realloc(catalog->sources, (size_t)capacity * sizeof(*entries));
    if (!entries) return 0;
    catalog->sources = entries;
    catalog->source_capacity = capacity;
    return 1;
}

static yvex_local_source_record *local_source_add(yvex_local_catalog *catalog)
{
    yvex_local_source_record *entry;

    if (!local_source_reserve(catalog, catalog->source_count + 1u)) return NULL;
    entry = &catalog->sources[catalog->source_count++];
    memset(entry, 0, sizeof(*entry));
    return entry;
}

static int local_package_reserve(yvex_local_catalog *catalog,
                                 unsigned long long required)
{
    yvex_local_package_record *entries;
    unsigned long long capacity;

    if (required <= catalog->package_capacity) return 1;
    capacity = catalog->package_capacity ? catalog->package_capacity * 2u : 32u;
    while (capacity < required) capacity *= 2u;
    if (capacity > LOCAL_CATALOG_ENTRY_CAP) capacity = LOCAL_CATALOG_ENTRY_CAP;
    if (capacity < required) return 0;
    entries = realloc(catalog->packages, (size_t)capacity * sizeof(*entries));
    if (!entries) return 0;
    catalog->packages = entries;
    catalog->package_capacity = capacity;
    return 1;
}

static yvex_local_package_record *local_package_add(yvex_local_catalog *catalog)
{
    yvex_local_package_record *entry;

    if (!local_package_reserve(catalog, catalog->package_count + 1u)) return NULL;
    entry = &catalog->packages[catalog->package_count++];
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

static int local_source_record_add(yvex_local_catalog *catalog,
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
    yvex_local_source_record *entry;

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
    entry = local_source_add(catalog);
    if (!entry) return local_refuse(err, YVEX_ERR_NOMEM, "local catalog allocation failed");
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
        local_copy(entry->acquisition_state, sizeof(entry->acquisition_state), "source-missing");
        local_copy(entry->blocker, sizeof(entry->blocker), "recorded source directory is unavailable");
    } else if (strcmp(status, "model-download-pass") == 0) {
        local_copy(entry->acquisition_state, sizeof(entry->acquisition_state), "source-acquired");
    } else {
        local_copy(entry->acquisition_state, sizeof(entry->acquisition_state), "source-partial");
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
    }
    return YVEX_OK;
}

static int local_source_manifest_add(yvex_local_catalog *catalog,
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
    yvex_local_source_record *entry;

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
    entry = local_source_add(catalog);
    if (!entry) return local_refuse(err, YVEX_ERR_NOMEM, "local catalog allocation failed");
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
        local_copy(entry->acquisition_state, sizeof(entry->acquisition_state), "source-missing");
        local_copy(entry->verification_state, sizeof(entry->verification_state), "recorded-unverified");
        local_copy(entry->blocker, sizeof(entry->blocker), "recorded source directory is unavailable");
    } else if (complete) {
        local_copy(entry->acquisition_state, sizeof(entry->acquisition_state), "source-acquired");
        local_copy(entry->verification_state, sizeof(entry->verification_state), "payload-verified");
    } else {
        local_copy(entry->acquisition_state, sizeof(entry->acquisition_state), "source-partial");
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

static int local_scan_acquisitions(yvex_local_catalog *catalog,
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
        if (result != YVEX_OK || catalog->source_count >= LOCAL_CATALOG_ENTRY_CAP) break;
    }
    (void)closedir(dir);
    return result;
}

static void local_package_provenance(yvex_local_package_record *entry,
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

static void local_package_record_add(yvex_local_catalog *catalog,
                              const yvex_model_registry_entry *registered)
{
    yvex_local_package_record *entry = local_package_add(catalog);
    yvex_error startup_error;
    int startup_ready;

    if (!entry) return;
    yvex_error_clear(&startup_error);
    startup_ready = yvex_model_registry_startup_validate(registered, &startup_error) == YVEX_OK;
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
    entry->ready = startup_ready;
    local_package_provenance(entry, registered);
}

static int local_scan_packages(yvex_local_catalog *catalog,
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
        if (catalog->package_count >= LOCAL_CATALOG_ENTRY_CAP) {
            yvex_model_registry_close(registry);
            return local_refuse(err, YVEX_ERR_BOUNDS, "local catalog entry limit exceeded");
        }
        local_package_record_add(catalog, entry);
    }
    yvex_model_registry_close(registry);
    return YVEX_OK;
}

static int local_source_compare(const void *left, const void *right)
{
    const yvex_local_source_record *a = left;
    const yvex_local_source_record *b = right;
    int family = strcmp(a->family, b->family);

    if (family) return family;
    return strcmp(a->name, b->name);
}

static int local_package_compare(const void *left, const void *right)
{
    const yvex_local_package_record *a = left;
    const yvex_local_package_record *b = right;
    int family = strcmp(a->family, b->family);

    if (family) return family;
    return strcmp(a->name, b->name);
}

int yvex_local_catalog_open(yvex_local_catalog **out,
                            const yvex_local_catalog_options *options,
                            yvex_error *err)
{
    yvex_local_catalog *catalog;
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
        yvex_local_catalog_close(catalog);
        return rc;
    }
    if (catalog->source_count > 1u)
        qsort(catalog->sources, (size_t)catalog->source_count, sizeof(*catalog->sources),
              local_source_compare);
    if (catalog->package_count > 1u)
        qsort(catalog->packages, (size_t)catalog->package_count, sizeof(*catalog->packages),
              local_package_compare);
    *out = catalog;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_local_catalog_close(yvex_local_catalog *catalog)
{
    if (!catalog) return;
    free(catalog->sources);
    free(catalog->packages);
    free(catalog);
}

unsigned long long yvex_local_catalog_source_count(const yvex_local_catalog *catalog)
{
    return catalog ? catalog->source_count : 0u;
}

const yvex_local_source_record *yvex_local_catalog_source_at(
    const yvex_local_catalog *catalog, unsigned long long index)
{
    return catalog && index < catalog->source_count ? &catalog->sources[index] : NULL;
}

unsigned long long yvex_local_catalog_package_count(const yvex_local_catalog *catalog)
{
    return catalog ? catalog->package_count : 0u;
}

const yvex_local_package_record *yvex_local_catalog_package_at(
    const yvex_local_catalog *catalog, unsigned long long index)
{
    return catalog && index < catalog->package_count ? &catalog->packages[index] : NULL;
}

typedef struct {
    yvex_model_library_entry summary;
    yvex_local_source_record *sources;
    unsigned long long source_count, source_capacity;
    yvex_model_artifact_fact *artifacts;
    unsigned long long artifact_count, artifact_capacity;
    yvex_model_runtime_profile_fact *profiles;
    unsigned long long profile_count, profile_capacity;
} library_model;

struct yvex_model_library {
    library_model *models;
    unsigned long long count, capacity;
};

static const char *library_text(const char *text)
{
    return text ? text : "";
}

static int library_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "model.library", reason);
    return status;
}

static int library_reserve(yvex_model_library *library, unsigned long long required,
                           yvex_error *err)
{
    library_model *models;
    unsigned long long capacity;

    if (required <= library->capacity) return YVEX_OK;
    capacity = library->capacity ? library->capacity * 2u : 8u;
    while (capacity < required) capacity *= 2u;
    if (capacity > LOCAL_CATALOG_ENTRY_CAP)
        return library_refuse(err, YVEX_ERR_BOUNDS, "logical model limit exceeded");
    models = realloc(library->models, (size_t)capacity * sizeof(*models));
    if (!models)
        return library_refuse(err, YVEX_ERR_NOMEM, "logical model allocation failed");
    memset(models + library->capacity, 0,
           (size_t)(capacity - library->capacity) * sizeof(*models));
    library->models = models;
    library->capacity = capacity;
    return YVEX_OK;
}

static library_model *library_find(yvex_model_library *library, const char *identity)
{
    unsigned long long index;

    for (index = 0u; index < library->count; ++index)
        if (strcmp(library->models[index].summary.identity, identity) == 0)
            return &library->models[index];
    return NULL;
}

static int library_identity_registry(const yvex_model_registry_entry *entry,
                                     yvex_model_library_entry *summary,
                                     yvex_error *err)
{
    const char *family = library_text(entry->family);
    const char *model = library_text(entry->model);
    const char *target = library_text(entry->runtime_target);
    const char *alias = library_text(entry->alias);
    int written;

    memset(summary, 0, sizeof(*summary));
    if (family[0] && model[0] && target[0]) {
        written = snprintf(summary->identity, sizeof(summary->identity),
                           "family:%s/model:%s/target:%s", family, model, target);
        summary->identity_kind = YVEX_MODEL_IDENTITY_FAMILY_MODEL_TARGET;
    } else if (target[0]) {
        written = snprintf(summary->identity, sizeof(summary->identity),
                           "target:%s", target);
        summary->identity_kind = YVEX_MODEL_IDENTITY_TARGET;
    } else if (alias[0]) {
        written = snprintf(summary->identity, sizeof(summary->identity),
                           "alias:%s", alias);
        summary->identity_kind = YVEX_MODEL_IDENTITY_ALIAS;
    } else {
        return library_refuse(err, YVEX_ERR_FORMAT,
                              "registry row has no exact logical identity");
    }
    if (written < 0 || (size_t)written >= sizeof(summary->identity))
        return library_refuse(err, YVEX_ERR_BOUNDS, "logical model identity is too long");
    local_copy(summary->family, sizeof(summary->family), family);
    local_copy(summary->model, sizeof(summary->model), model);
    local_copy(summary->runtime_target, sizeof(summary->runtime_target), target);
    local_copy(summary->display_name, sizeof(summary->display_name),
               model[0] ? model : target[0] ? target : alias);
    return YVEX_OK;
}

static int library_artifact_add(library_model *model,
                                const yvex_model_registry_entry *entry,
                                yvex_error *err)
{
    yvex_model_artifact_fact *facts, *fact;
    const char *identity = library_text(entry->sha256);
    const char *path = library_text(entry->path);
    unsigned long long index, capacity;

    if (!path[0]) return YVEX_OK;
    for (index = 0u; index < model->artifact_count; ++index) {
        fact = &model->artifacts[index];
        if ((identity[0] && strcmp(fact->identity, identity) == 0) ||
            (!identity[0] && !fact->identity[0] && strcmp(fact->path, path) == 0)) {
            if (entry->execution_ready) fact->execution_ready = 1;
            if (fact->execution_ready) model->summary.artifact_ready = 1;
            return YVEX_OK;
        }
    }
    if (model->artifact_count == model->artifact_capacity) {
        capacity = model->artifact_capacity ? model->artifact_capacity * 2u : 4u;
        facts = realloc(model->artifacts, (size_t)capacity * sizeof(*facts));
        if (!facts)
            return library_refuse(err, YVEX_ERR_NOMEM, "artifact aggregate allocation failed");
        model->artifacts = facts;
        model->artifact_capacity = capacity;
    }
    fact = &model->artifacts[model->artifact_count++];
    memset(fact, 0, sizeof(*fact));
    local_copy(fact->identity, sizeof(fact->identity), identity);
    local_copy(fact->path, sizeof(fact->path), path);
    local_copy(fact->artifact_class, sizeof(fact->artifact_class), entry->artifact_class);
    local_copy(fact->format, sizeof(fact->format), entry->format);
    local_copy(fact->physical_variant, sizeof(fact->physical_variant), entry->qprofile);
    fact->file_size = entry->file_size;
    fact->tensor_count = entry->tensor_count;
    fact->execution_ready = entry->execution_ready != 0;
    model->summary.artifact_count = model->artifact_count;
    if (fact->execution_ready) model->summary.artifact_ready = 1;
    return YVEX_OK;
}

static int library_profile_present(const yvex_model_registry_entry *entry)
{
    return library_text(entry->runtime_profile)[0] ||
           library_text(entry->runtime_installation)[0] ||
           library_text(entry->runtime_binding)[0] ||
           library_text(entry->runtime_target)[0] ||
           library_text(entry->runtime_backend)[0] ||
           library_text(entry->runtime_mode)[0] || entry->runtime_context;
}

static int library_profile_add(library_model *model,
                               const yvex_model_registry_entry *entry,
                               yvex_error *err)
{
    yvex_model_runtime_profile_fact *profiles, *fact;
    unsigned long long capacity;
    yvex_error admission;

    if (!library_profile_present(entry)) return YVEX_OK;
    if (model->profile_count == model->profile_capacity) {
        capacity = model->profile_capacity ? model->profile_capacity * 2u : 4u;
        profiles = realloc(model->profiles, (size_t)capacity * sizeof(*profiles));
        if (!profiles)
            return library_refuse(err, YVEX_ERR_NOMEM, "profile aggregate allocation failed");
        model->profiles = profiles;
        model->profile_capacity = capacity;
    }
    fact = &model->profiles[model->profile_count++];
    memset(fact, 0, sizeof(*fact));
    local_copy(fact->alias, sizeof(fact->alias), entry->alias);
    local_copy(fact->profile, sizeof(fact->profile), entry->runtime_profile);
    local_copy(fact->installation, sizeof(fact->installation), entry->runtime_installation);
    local_copy(fact->artifact_path, sizeof(fact->artifact_path), entry->path);
    local_copy(fact->artifact_identity, sizeof(fact->artifact_identity), entry->sha256);
    local_copy(fact->artifact_class, sizeof(fact->artifact_class), entry->artifact_class);
    local_copy(fact->runtime_binding, sizeof(fact->runtime_binding), entry->runtime_binding);
    local_copy(fact->runtime_target, sizeof(fact->runtime_target), entry->runtime_target);
    local_copy(fact->backend, sizeof(fact->backend), entry->runtime_backend);
    local_copy(fact->generation_mode, sizeof(fact->generation_mode), entry->runtime_mode);
    fact->context_capacity = entry->runtime_context;
    yvex_error_clear(&admission);
    /* Startup readiness belongs to the canonical profile validator.  The legacy
     * execution_ready bit describes an older artifact capability and is false in
     * valid v5 launch records that the server host accepts. */
    fact->launchable =
        yvex_model_registry_startup_validate(entry, &admission) == YVEX_OK;
    if (!fact->launchable)
        local_copy(fact->blocker, sizeof(fact->blocker),
                   yvex_error_message(&admission));
    model->summary.profile_count = model->profile_count;
    if (fact->launchable) {
        model->summary.launchable_profile_count++;
        model->summary.profile_launchable = 1;
    }
    return YVEX_OK;
}

static int library_registry_add(yvex_model_library *library,
                                const yvex_model_registry_entry *entry,
                                yvex_error *err)
{
    yvex_model_library_entry identity;
    yvex_local_package_record provenance;
    library_model *model;
    int rc;

    rc = library_identity_registry(entry, &identity, err);
    if (rc != YVEX_OK) return rc;
    model = library_find(library, identity.identity);
    if (!model) {
        rc = library_reserve(library, library->count + 1u, err);
        if (rc != YVEX_OK) return rc;
        model = &library->models[library->count++];
        model->summary = identity;
    }
    memset(&provenance, 0, sizeof(provenance));
    local_package_provenance(&provenance, entry);
    if (provenance.repository[0] && provenance.revision[0]) {
        local_copy(model->summary.provider, sizeof(model->summary.provider), "huggingface");
        local_copy(model->summary.repository, sizeof(model->summary.repository),
                   provenance.repository);
        local_copy(model->summary.revision, sizeof(model->summary.revision),
                   provenance.revision);
    }
    rc = library_artifact_add(model, entry, err);
    return rc == YVEX_OK ? library_profile_add(model, entry, err) : rc;
}

static int library_source_equal(const yvex_local_source_record *left,
                                const yvex_local_source_record *right)
{
    return strcmp(left->provider, right->provider) == 0 &&
           strcmp(left->repository, right->repository) == 0 &&
           strcmp(left->revision, right->revision) == 0;
}

static library_model *library_source_model(yvex_model_library *library,
                                           const yvex_local_source_record *source)
{
    unsigned long long index;

    for (index = 0u; index < library->count; ++index) {
        const yvex_model_library_entry *summary = &library->models[index].summary;
        if (source->repository[0] && source->revision[0] && summary->repository[0] &&
            strcmp(source->provider, summary->provider) == 0 &&
            strcmp(source->repository, summary->repository) == 0 &&
            strcmp(source->revision, summary->revision) == 0)
            return &library->models[index];
    }
    return NULL;
}

static int library_source_add(yvex_model_library *library,
                              const yvex_local_source_record *source,
                              yvex_error *err)
{
    yvex_local_source_record *sources;
    yvex_model_library_entry identity;
    library_model *model = library_source_model(library, source);
    unsigned long long index, capacity;
    int written, rc;

    if (!model) {
        memset(&identity, 0, sizeof(identity));
        written = snprintf(identity.identity, sizeof(identity.identity),
                           "provider:%s/repository:%s/revision:%s", source->provider,
                           source->repository, source->revision);
        if (written < 0 || (size_t)written >= sizeof(identity.identity))
            return library_refuse(err, YVEX_ERR_BOUNDS, "source identity is too long");
        identity.identity_kind = YVEX_MODEL_IDENTITY_PROVIDER_REPOSITORY_REVISION;
        local_copy(identity.display_name, sizeof(identity.display_name), source->name);
        local_copy(identity.family, sizeof(identity.family), source->family);
        local_copy(identity.provider, sizeof(identity.provider), source->provider);
        local_copy(identity.repository, sizeof(identity.repository), source->repository);
        local_copy(identity.revision, sizeof(identity.revision), source->revision);
        model = library_find(library, identity.identity);
        if (!model) {
            rc = library_reserve(library, library->count + 1u, err);
            if (rc != YVEX_OK) return rc;
            model = &library->models[library->count++];
            model->summary = identity;
        }
    }
    for (index = 0u; index < model->source_count; ++index)
        if (library_source_equal(&model->sources[index], source)) return YVEX_OK;
    if (model->source_count == model->source_capacity) {
        capacity = model->source_capacity ? model->source_capacity * 2u : 2u;
        sources = realloc(model->sources, (size_t)capacity * sizeof(*sources));
        if (!sources)
            return library_refuse(err, YVEX_ERR_NOMEM, "source aggregate allocation failed");
        model->sources = sources;
        model->source_capacity = capacity;
    }
    model->sources[model->source_count++] = *source;
    model->summary.source_count = model->source_count;
    if (strcmp(source->acquisition_state, "source-acquired") == 0)
        model->summary.source_local = 1;
    return YVEX_OK;
}

static int library_model_compare(const void *left, const void *right)
{
    const library_model *a = left;
    const library_model *b = right;
    int family = strcmp(a->summary.family, b->summary.family);

    if (family) return family;
    return strcmp(a->summary.display_name, b->summary.display_name);
}

int yvex_model_library_open(yvex_model_library **out,
                            const yvex_local_catalog_options *options,
                            yvex_error *err)
{
    yvex_model_library *library;
    yvex_local_catalog *local = NULL;
    yvex_model_registry *registry = NULL;
    yvex_model_registry_options registry_options;
    char default_registry[YVEX_PATH_CAP];
    const char *registry_path = options ? options->registry_path : NULL;
    unsigned long long index;
    int rc;

    if (!out) return library_refuse(err, YVEX_ERR_INVALID_ARG, "library output is required");
    *out = NULL;
    library = calloc(1u, sizeof(*library));
    if (!library) return library_refuse(err, YVEX_ERR_NOMEM, "model library allocation failed");
    if (!registry_path || !registry_path[0]) {
        rc = yvex_model_registry_default_path(default_registry, sizeof(default_registry), err);
        if (rc != YVEX_OK) goto fail;
        registry_path = default_registry;
    }
    memset(&registry_options, 0, sizeof(registry_options));
    registry_options.registry_path = registry_path;
    rc = yvex_model_registry_open(&registry, &registry_options, err);
    if (rc != YVEX_OK && access(registry_path, F_OK) == 0) goto fail;
    if (rc != YVEX_OK) yvex_error_clear(err);
    for (index = 0u; registry && index < yvex_model_registry_count(registry); ++index) {
        rc = library_registry_add(library, yvex_model_registry_at(registry, index), err);
        if (rc != YVEX_OK) goto fail;
    }
    yvex_model_registry_close(registry);
    registry = NULL;
    rc = yvex_local_catalog_open(&local, options, err);
    if (rc != YVEX_OK) goto fail;
    for (index = 0u; index < yvex_local_catalog_source_count(local); ++index) {
        rc = library_source_add(library, yvex_local_catalog_source_at(local, index), err);
        if (rc != YVEX_OK) goto fail;
    }
    yvex_local_catalog_close(local);
    if (library->count > 1u)
        qsort(library->models, (size_t)library->count, sizeof(*library->models),
              library_model_compare);
    *out = library;
    yvex_error_clear(err);
    return YVEX_OK;
fail:
    yvex_model_registry_close(registry);
    yvex_local_catalog_close(local);
    yvex_model_library_close(library);
    return rc;
}

void yvex_model_library_close(yvex_model_library *library)
{
    unsigned long long index;

    if (!library) return;
    for (index = 0u; index < library->count; ++index) {
        free(library->models[index].sources);
        free(library->models[index].artifacts);
        free(library->models[index].profiles);
    }
    free(library->models);
    free(library);
}

unsigned long long yvex_model_library_count(const yvex_model_library *library)
{
    return library ? library->count : 0u;
}

const yvex_model_library_entry *yvex_model_library_at(
    const yvex_model_library *library, unsigned long long index)
{
    return library && index < library->count ? &library->models[index].summary : NULL;
}

unsigned long long yvex_model_library_artifact_count(
    const yvex_model_library *library, unsigned long long model_index)
{
    return library && model_index < library->count
               ? library->models[model_index].artifact_count : 0u;
}

const yvex_model_artifact_fact *yvex_model_library_artifact_at(
    const yvex_model_library *library, unsigned long long model_index,
    unsigned long long artifact_index)
{
    if (!library || model_index >= library->count ||
        artifact_index >= library->models[model_index].artifact_count)
        return NULL;
    return &library->models[model_index].artifacts[artifact_index];
}

unsigned long long yvex_model_library_profile_count(
    const yvex_model_library *library, unsigned long long model_index)
{
    return library && model_index < library->count
               ? library->models[model_index].profile_count : 0u;
}

const yvex_model_runtime_profile_fact *yvex_model_library_profile_at(
    const yvex_model_library *library, unsigned long long model_index,
    unsigned long long profile_index)
{
    if (!library || model_index >= library->count ||
        profile_index >= library->models[model_index].profile_count)
        return NULL;
    return &library->models[model_index].profiles[profile_index];
}

unsigned long long yvex_model_library_source_count(
    const yvex_model_library *library, unsigned long long model_index)
{
    return library && model_index < library->count
               ? library->models[model_index].source_count : 0u;
}

const yvex_local_source_record *yvex_model_library_source_at(
    const yvex_model_library *library, unsigned long long model_index,
    unsigned long long source_index)
{
    if (!library || model_index >= library->count ||
        source_index >= library->models[model_index].source_count)
        return NULL;
    return &library->models[model_index].sources[source_index];
}

int yvex_model_library_remote_match(const yvex_model_library *library,
                                    const yvex_remote_model *remote,
                                    unsigned long long *model_index)
{
    unsigned long long index;
    const char *revision;

    if (model_index) *model_index = 0u;
    if (!library || !remote || !remote->provider[0] || !remote->repository[0]) return 0;
    revision = remote->resolved_revision[0] ? remote->resolved_revision
                                           : remote->revision_reference;
    if (!revision[0]) return 0;
    for (index = 0u; index < library->count; ++index) {
        const yvex_model_library_entry *summary = &library->models[index].summary;
        if (strcmp(summary->provider, remote->provider) == 0 &&
            strcmp(summary->repository, remote->repository) == 0 &&
            strcmp(summary->revision, revision) == 0) {
            if (model_index) *model_index = index;
            return 1;
        }
    }
    return 0;
}
