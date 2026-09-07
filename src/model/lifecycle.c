/* Own catalog-authorized local eviction; remote and logical records outlive local bytes. */
#define _GNU_SOURCE
#include <yvex/internal/model_lifecycle.h>
#include <yvex/internal/core.h>
#include <yvex/internal/source_distribution.h>
#include <yvex/internal/source_catalog.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/openat2.h>
#endif

static int lifecycle_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "model.lifecycle", reason);
    return status;
}

int yvex_model_artifact_local_verify(const yvex_model_artifact_fact *fact,
                                     const char *models_root,
                                     yvex_artifact_snapshot *snapshot, yvex_error *err)
{
    yvex_artifact *artifact = NULL;
    yvex_artifact_options options = {0};
    yvex_artifact_reopen_lease lease;
    yvex_paths paths;
    char cache[YVEX_PATH_CAP];
    int rc;
    if (!fact || !models_root || !snapshot || !yvex_sha256_hex_is_valid(fact->identity))
        return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "exact artifact and model root required");
    options.path = fact->path;
    options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &options, err);
    if (rc != YVEX_OK) return rc;
    if (fact->file_size && yvex_artifact_size(artifact) != fact->file_size) {
        yvex_artifact_close(artifact);
        return lifecycle_refuse(err, YVEX_ERR_STATE, "local artifact size differs from catalog identity");
    }
    rc = yvex_paths_default(&paths, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_reopen_lease_check(artifact, fact->identity, paths.cache_dir, &lease, err);
    if (rc == YVEX_OK && !lease.verified) {
        if (snprintf(cache, sizeof(cache), "%s/cache/verification", models_root) >= (int)sizeof(cache))
            rc = lifecycle_refuse(err, YVEX_ERR_BOUNDS, "verification cache path exceeds bound");
        else rc = yvex_artifact_reopen_lease_check(artifact, fact->identity, cache, &lease, err);
    }
    if (rc == YVEX_OK && !lease.verified)
        rc = lifecycle_refuse(err, YVEX_ERR_STATE,
                               "local snapshot lacks current verification; explicit integrity verification required");
    if (rc == YVEX_OK) *snapshot = lease.snapshot;
    yvex_artifact_close(artifact);
    return rc;
}

int yvex_model_artifact_is_managed(const yvex_model_library *library, unsigned long long model_index,
                                    const yvex_model_artifact_fact *artifact, const char *models_root)
{
    char managed[YVEX_PATH_CAP];
    unsigned long long index;
    if (!library || !artifact || !models_root || strstr(artifact->path, "/../") ||
        snprintf(managed, sizeof(managed), "%s/representations/", models_root) >= (int)sizeof(managed) ||
        strncmp(artifact->path, managed, strlen(managed))) return 0;
    for (index = 0u; index < yvex_model_library_source_count(library, model_index); ++index) {
        const yvex_local_source_record *source = yvex_model_library_source_at(library, model_index, index);
        size_t length = strlen(source->path);
        if (!strcmp(source->storage_kind, "external") && length &&
            !strncmp(source->path, artifact->path, length) &&
            (!artifact->path[length] || artifact->path[length] == '/')) return 0;
    }
    return 1;
}

int yvex_model_local_content_resolve(const char *models_root, const char *digest,
                                      yvex_model_remote_selection *out, yvex_error *err)
{
    yvex_model_library *library = NULL;
    yvex_local_catalog_options options = {models_root, NULL};
    unsigned long long model, index;
    int rc;
    if (!out || !models_root || !yvex_sha256_hex_is_valid(digest))
        return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "exact local content identity required");
    memset(out, 0, sizeof(*out));
    rc = yvex_model_library_open(&library, &options, err);
    if (rc != YVEX_OK) return rc;
    for (model = 0u; model < yvex_model_library_count(library); ++model) {
        for (index = 0u; index < yvex_model_library_artifact_count(library, model); ++index) {
            const yvex_model_artifact_fact *fact = yvex_model_library_artifact_at(library, model, index);
            const yvex_model_library_entry *logical = yvex_model_library_at(library, model);
            yvex_artifact_snapshot snapshot;
            if (strcmp(fact->identity, digest) || !yvex_model_library_artifact_is_local(library, model, index) ||
                !yvex_model_artifact_is_managed(library, model, fact, models_root)) continue;
            rc = yvex_model_artifact_local_verify(fact, models_root, &snapshot, err);
            if (rc != YVEX_OK) goto done;
            if (out->found && strcmp(out->logical_identity, logical->identity)) {
                rc = lifecycle_refuse(err, YVEX_ERR_STATE,
                    "content has multiple logical owners; select an explicit model");
                goto done;
            }
            out->artifact = *fact;
            out->found = out->local = out->verified = 1;
            yvex_core_text_copy(out->logical_identity, sizeof(out->logical_identity), logical->identity);
            yvex_core_text_copy(out->model, sizeof(out->model), logical->display_name);
            yvex_core_text_copy(out->family, sizeof(out->family), logical->family);
        }
        for (index = 0u; index < yvex_model_library_source_count(library, model); ++index) {
            const yvex_local_source_record *source = yvex_model_library_source_at(library, model, index);
            const yvex_model_library_entry *logical = yvex_model_library_at(library, model);
            yvex_source_representation_fact member;
            yvex_error ignored;
            char managed[YVEX_PATH_CAP];
            if (strcmp(source->storage_kind, "managed") || strcmp(source->format, "gguf") ||
                snprintf(managed, sizeof(managed), "%s/source/", models_root) >= (int)sizeof(managed) ||
                strncmp(source->path, managed, strlen(managed)) || strstr(source->path, "/../") ||
                !yvex_sha256_hex_is_valid(source->digest)) continue;
            if (!strcmp(source->digest, digest)) {
                if (yvex_source_file_reopen(models_root, source->path, digest, &ignored) != YVEX_OK) continue;
                memset(&member, 0, sizeof(member));
                yvex_core_text_copy(member.path, sizeof(member.path), source->path);
                member.size_bytes = source->size_bytes;
            } else if (yvex_source_gguf_member_reopen(models_root, source->path, source->digest, digest,
                                                       &member, &ignored) != YVEX_OK) continue;
            if (out->found && strcmp(out->logical_identity, logical->identity)) {
                rc = lifecycle_refuse(err, YVEX_ERR_STATE,
                    "content has multiple logical owners; select an explicit model");
                goto done;
            }
            if (out->found) continue;
            out->found = out->local = out->verified = 1;
            yvex_core_text_copy(out->artifact.path, sizeof(out->artifact.path), member.path);
            yvex_core_text_copy(out->artifact.identity, sizeof(out->artifact.identity), digest);
            yvex_core_text_copy(out->artifact.format, sizeof(out->artifact.format), "gguf");
            out->artifact.file_size = member.size_bytes;
            yvex_core_text_copy(out->logical_identity, sizeof(out->logical_identity), logical->identity);
            yvex_core_text_copy(out->model, sizeof(out->model), logical->display_name);
            yvex_core_text_copy(out->family, sizeof(out->family), logical->family);
        }
    }
done:
    yvex_model_library_close(library);
    return rc;
}

int yvex_model_remote_selection_resolve(const char *models_root, const char *repository,
                                         const char *revision, const char *variant,
                                         const char *format,
                                         yvex_model_remote_selection *out, yvex_error *err)
{
    yvex_model_library *library = NULL;
    yvex_local_catalog_options options = {models_root, NULL};
    yvex_artifact_snapshot snapshot;
    unsigned long long model, publication, artifact;
    int rc;
    if (!models_root || !repository || !out)
        return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "model root, repository and result required");
    memset(out, 0, sizeof(*out));
    rc = yvex_model_library_open(&library, &options, err);
    if (rc != YVEX_OK) return rc;
    for (model = 0u; model < yvex_model_library_count(library); ++model) {
        for (publication = 0u; publication < yvex_model_library_publication_count(library, model); ++publication) {
            const yvex_model_publication *remote = yvex_model_library_publication_at(library, model, publication);
            if (strcmp(remote->provider, "huggingface") || strcmp(remote->repository, repository) ||
                (revision && revision[0] && strcmp(remote->revision, revision))) continue;
            for (artifact = 0u; artifact < yvex_model_library_artifact_count(library, model); ++artifact) {
                const yvex_model_artifact_fact *fact = yvex_model_library_artifact_at(library, model, artifact);
                const yvex_model_library_entry *logical = yvex_model_library_at(library, model);
                if (strcmp(fact->identity, remote->artifact_identity) ||
                    (variant && strcmp(variant, fact->identity) && strcmp(variant, remote->filename) &&
                     strcmp(variant, fact->physical_variant)) ||
                    (format && strcasecmp(format, fact->format))) continue;
                if (out->found) {
                    rc = lifecycle_refuse(err, YVEX_ERR_INVALID_ARG,
                                           "multiple known representations match; select --variant FILENAME or SHA256");
                    goto done;
                }
                out->found = 1;
                out->artifact = *fact;
                out->remote = *remote;
                yvex_core_text_copy(out->logical_identity, sizeof(out->logical_identity), logical->identity);
                yvex_core_text_copy(out->model, sizeof(out->model), logical->display_name);
                out->local = yvex_model_library_artifact_is_local(library, model, artifact);
            }
        }
    }
    if (out->local) {
        rc = yvex_model_artifact_local_verify(&out->artifact, models_root, &snapshot, err);
        out->verified = rc == YVEX_OK;
    }
done:
    yvex_model_library_close(library);
    return rc;
}

int yvex_model_remote_adopt_existing(const yvex_remote_catalog *catalog,
                                      const yvex_model_representation *representation,
                                      const char *models_root, int dry_run,
                                      yvex_model_remote_selection *out, yvex_error *err)
{
    const yvex_remote_model *remote = yvex_remote_catalog_at(catalog, 0u);
    unsigned int index;
    int rc = YVEX_OK;
    if (!remote || !representation || !models_root || !out)
        return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "remote representation required");
    memset(out, 0, sizeof(*out));
    if (strcmp(representation->format, "gguf") || representation->file_count != 1u) return YVEX_OK;
    for (index = 0u; index < remote->available_file_count; ++index) {
        const yvex_remote_file *file = yvex_remote_catalog_file_at(catalog, 0u, index);
        const char *digest = yvex_model_remote_file_sha256(catalog, 0u, index);
        yvex_source_reference_options reference = {0};
        yvex_source_reference_result registered;
        yvex_source_locator locator = {0};
        yvex_source_representation_fact local = {0};
        if (strcmp(file->path, representation->file_pattern) || !file->size_known ||
            strchr(file->path, '/') || !yvex_sha256_hex_is_valid(digest)) continue;
        rc = yvex_model_local_content_resolve(models_root, digest, out, err);
        if (rc != YVEX_OK || !out->found) return rc;
        if (out->artifact.file_size != file->size_bytes)
            return lifecycle_refuse(err, YVEX_ERR_STATE, "provider and local content sizes disagree");
        locator.kind = YVEX_SOURCE_LOCATOR_HUGGINGFACE;
        yvex_core_text_copy(locator.repository, sizeof(locator.repository), remote->repository);
        yvex_core_text_copy(local.path, sizeof(local.path), out->artifact.path);
        yvex_core_text_copy(local.digest, sizeof(local.digest), digest);
        yvex_core_text_copy(local.format, sizeof(local.format), "gguf");
        local.size_bytes = file->size_bytes;
        reference.locator = &locator;
        reference.local = &local;
        reference.models_root = models_root;
        reference.name = out->model;
        reference.family = out->family;
        reference.resolved_revision = remote->resolved_revision;
        reference.format = "gguf";
        reference.remote_filename = file->path;
        reference.size_bytes = file->size_bytes;
        reference.size_known = 1;
        yvex_core_text_copy(out->remote.revision, sizeof(out->remote.revision), remote->resolved_revision);
        return dry_run ? YVEX_OK : yvex_source_register_reference(&reference, &registered, err);
    }
    return rc;
}

int yvex_model_remote_revision_resolve(const char *models_root, const char *repository,
                                         char revision[YVEX_REMOTE_REVISION_CAP], yvex_error *err)
{
    yvex_local_catalog *catalog = NULL;
    yvex_local_catalog_options options = {models_root, NULL};
    unsigned long long index;
    int rc;
    if (!models_root || !repository || !revision)
        return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "repository identity required");
    revision[0] = '\0';
    rc = yvex_local_catalog_open(&catalog, &options, err);
    if (rc != YVEX_OK) return rc;
    for (index = 0u; index < yvex_local_catalog_source_count(catalog); ++index) {
        const yvex_local_source_record *source = yvex_local_catalog_source_at(catalog, index);
        char canonical[YVEX_PATH_CAP];
        if (strcmp(source->provider, "huggingface") || strcmp(source->repository, repository) ||
            !yvex_source_provider_path(canonical, sizeof(canonical), models_root, repository, source->revision))
            continue;
        if (revision[0] && strcmp(revision, source->revision)) {
            rc = lifecycle_refuse(err, YVEX_ERR_INVALID_ARG,
                "multiple retained revisions match; select --revision or deliberately --refresh");
            break;
        }
        yvex_core_text_copy(revision, YVEX_REMOTE_REVISION_CAP, source->revision);
    }
    yvex_local_catalog_close(catalog);
    return rc;
}

int yvex_model_remote_source_resolve(const char *models_root, const char *repository,
                                      const char *revision, const char *variant, const char *format,
                                      yvex_local_source_record *out, int *found, yvex_error *err)
{
    yvex_local_catalog *catalog = NULL;
    yvex_local_catalog_options options = {models_root, NULL};
    yvex_source_locator locator = {0};
    yvex_source_representation_fact verified;
    unsigned long long index;
    int rc;
    if (!out || !found) return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "source result required");
    *found = 0;
    memset(out, 0, sizeof(*out));
    rc = yvex_local_catalog_open(&catalog, &options, err);
    if (rc != YVEX_OK) return rc;
    for (index = 0u; index < yvex_local_catalog_source_count(catalog); ++index) {
        const yvex_local_source_record *source = yvex_local_catalog_source_at(catalog, index);
        char canonical[YVEX_PATH_CAP];
        if (strcmp(source->provider, "huggingface") || strcmp(source->repository, repository) ||
            (revision && strcmp(source->revision, revision)) ||
            (format && strcasecmp(source->format, format)) ||
            (strcmp(source->acquisition_state, "source-acquired") &&
             strcmp(source->acquisition_state, "source-missing")) ||
            !yvex_sha256_hex_is_valid(source->digest)) continue;
        if (!yvex_source_provider_path(canonical, sizeof(canonical), models_root, repository, source->revision)) {
            rc = lifecycle_refuse(err, YVEX_ERR_STATE, "acquired source lacks an immutable upstream revision");
            goto done;
        }
        if (variant && strcmp(variant, source->precision) && strcmp(variant, source->digest) &&
            strcmp(variant, source->representation)) {
            char path[YVEX_PATH_CAP];
            struct stat status;
            if (strchr(variant, '/') || strstr(variant, "..") ||
                snprintf(path, sizeof(path), "%s/%s", source->path, variant) >= (int)sizeof(path) ||
                lstat(path, &status) != 0 || !S_ISREG(status.st_mode)) continue;
        }
        if (*found && (strcmp(out->revision, source->revision) || strcmp(out->digest, source->digest))) {
            rc = lifecycle_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "multiple acquired revisions match; select an immutable --revision");
            goto done;
        }
        *out = *source;
        *found = 1;
    }
    if (*found && !strcmp(out->acquisition_state, "source-acquired")) {
        rc = yvex_source_locator_parse(out->path, &locator, err);
        if (rc == YVEX_OK)
            rc = yvex_source_representation_verify_local(&locator, models_root, out->digest, &verified, err);
    }
done:
    yvex_local_catalog_close(catalog);
    return rc;
}

static int publication_exact(const yvex_model_library *library, unsigned long long model,
                              const yvex_model_artifact_fact *artifact)
{
    unsigned long long index;
    for (index = 0u; index < yvex_model_library_publication_count(library, model); ++index) {
        const yvex_model_publication *remote = yvex_model_library_publication_at(library, model, index);
        if (remote && !strcmp(remote->artifact_identity, artifact->identity) &&
            !strcmp(remote->remote_sha256, artifact->identity) && remote->size_bytes == artifact->file_size)
            return 1;
    }
    return 0;
}

static int snapshot_matches_stat(const yvex_artifact_snapshot *snapshot, const struct stat *status)
{
    return S_ISREG(status->st_mode) && snapshot->device == (unsigned long long)status->st_dev &&
           snapshot->inode == (unsigned long long)status->st_ino &&
           snapshot->size == (unsigned long long)status->st_size &&
           snapshot->mtime_seconds == status->st_mtim.tv_sec &&
           snapshot->mtime_nanoseconds == status->st_mtim.tv_nsec &&
           snapshot->ctime_seconds == status->st_ctim.tv_sec &&
           snapshot->ctime_nanoseconds == status->st_ctim.tv_nsec;
}

static int evict_verified(const char *path, const yvex_artifact_snapshot *snapshot,
                           int dry_run, yvex_model_storage_result *out, yvex_error *err)
{
    char parent[YVEX_PATH_CAP], *leaf;
    struct stat opened, named;
    int directory = -1, fd = -1, rc = YVEX_OK;
#ifdef __linux__
    struct open_how how = {0};
    how.flags = O_RDONLY | O_CLOEXEC | O_DIRECTORY;
    how.resolve = RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
#endif
    yvex_core_text_copy(parent, sizeof(parent), path);
    leaf = strrchr(parent, '/');
    if (!leaf || !leaf[1]) return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "artifact parent required");
    *leaf++ = '\0';
#ifdef __linux__
    directory = (int)syscall(SYS_openat2, AT_FDCWD, parent, &how, sizeof(how));
#else
    return lifecycle_refuse(err, YVEX_ERR_UNSUPPORTED, "safe local eviction requires openat2");
#endif
    if (directory < 0 || (fd = openat(directory, leaf, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)) < 0) {
        rc = lifecycle_refuse(err, YVEX_ERR_IO, "cannot open canonical eviction target");
        goto done;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        rc = lifecycle_refuse(err, YVEX_ERR_STATE, "loaded model or active artifact operation prevents eviction");
        goto done;
    }
    if (fstat(fd, &opened) != 0 || fstatat(directory, leaf, &named, AT_SYMLINK_NOFOLLOW) != 0 ||
        !snapshot_matches_stat(snapshot, &opened) || !snapshot_matches_stat(snapshot, &named)) {
        rc = lifecycle_refuse(err, YVEX_ERR_STATE, "artifact changed before eviction; nothing removed");
        goto done;
    }
    out->local = 1;
    out->verified = 1;
    out->logical_bytes = (unsigned long long)opened.st_size;
    out->allocated_bytes = opened.st_nlink == 1 ? (unsigned long long)opened.st_blocks * 512u : 0u;
    if (!dry_run) {
        if (unlinkat(directory, leaf, 0) != 0)
            rc = lifecycle_refuse(err, YVEX_ERR_IO, "cannot remove verified local name");
        else {
            out->changed = 1;
            out->local = 0;
            if (fsync(directory) != 0)
                rc = lifecycle_refuse(err, YVEX_ERR_IO, "local name removed but directory sync failed");
        }
    }
done:
    if (fd >= 0) (void)close(fd);
    if (directory >= 0) (void)close(directory);
    return rc;
}

int yvex_model_source_evict(const yvex_model_library *library,
                             unsigned long long model_index, unsigned long long source_index,
                             const char *models_root, int dry_run,
                             yvex_model_storage_result *out, yvex_error *err)
{
    const yvex_local_source_record *source = yvex_model_library_source_at(library, model_index, source_index);
    yvex_source_eviction_result result;
    char upstream[YVEX_PATH_CAP];
    int rc;
    if (!out || !source || strcmp(source->provider, "huggingface") ||
        strcmp(source->storage_kind, "managed") || !yvex_sha256_hex_is_valid(source->digest) ||
        !yvex_source_provider_path(upstream, sizeof(upstream), models_root, source->repository, source->revision))
        return lifecycle_refuse(err, YVEX_ERR_STATE, "only exact managed upstream acquisitions may be evicted");
    memset(out, 0, sizeof(*out));
    if (!strcmp(source->format, "gguf") && strcmp(source->representation, "gguf")) {
        yvex_model_artifact_fact fact = {0};
        yvex_artifact_snapshot snapshot;
        struct stat status;
        char canonical[YVEX_PATH_CAP];
        if (strchr(source->representation, '/') || strstr(source->representation, "..") ||
            snprintf(canonical, sizeof(canonical), "%s/source/local/%s/model.gguf", models_root, source->digest) >=
                (int)sizeof(canonical) || strcmp(source->path, canonical))
            return lifecycle_refuse(err, YVEX_ERR_STATE, "source file is not a canonical managed content object");
        yvex_core_text_copy(out->path, sizeof(out->path), source->path);
        yvex_core_text_copy(out->identity, sizeof(out->identity), source->digest);
        if (lstat(source->path, &status) != 0)
            return errno == ENOENT ? YVEX_OK : lifecycle_refuse(err, YVEX_ERR_IO, "cannot inspect source file");
        yvex_core_text_copy(fact.path, sizeof(fact.path), source->path);
        yvex_core_text_copy(fact.identity, sizeof(fact.identity), source->digest);
        fact.file_size = source->size_bytes;
        rc = yvex_model_artifact_local_verify(&fact, models_root, &snapshot, err);
        return rc == YVEX_OK ? evict_verified(source->path, &snapshot, dry_run, out, err) : rc;
    }
    rc = yvex_source_evict_local(models_root, source->repository, source->revision,
                                 source->path, source->digest, dry_run, &result, err);
    yvex_core_text_copy(out->path, sizeof(out->path), source->path);
    yvex_core_text_copy(out->identity, sizeof(out->identity), source->digest);
    if (rc == YVEX_OK) {
        out->changed = result.changed;
        out->local = result.local;
        out->logical_bytes = result.logical_bytes;
        out->allocated_bytes = result.allocated_bytes;
        out->verified = 1;
    }
    return rc;
}

int yvex_model_local_evict(const yvex_model_library *library,
                            unsigned long long model_index, unsigned long long artifact_index,
                            const char *models_root, int dry_run,
                            yvex_model_storage_result *out, yvex_error *err)
{
    const yvex_model_artifact_fact *artifact;
    yvex_artifact_snapshot snapshot;
    struct stat status;
    int rc;
    if (!library || !models_root || !out)
        return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "catalog, model root and result required");
    memset(out, 0, sizeof(*out));
    artifact = yvex_model_library_artifact_at(library, model_index, artifact_index);
    if (!artifact || !publication_exact(library, model_index, artifact))
        return lifecycle_refuse(err, YVEX_ERR_STATE,
                               "exact verified remote artifact identity is required; unique bytes are retained");
    if (!yvex_model_artifact_is_managed(library, model_index, artifact, models_root))
        return lifecycle_refuse(err, YVEX_ERR_STATE, "external artifact location is not owned by YVEX");
    yvex_core_text_copy(out->path, sizeof(out->path), artifact->path);
    yvex_core_text_copy(out->identity, sizeof(out->identity), artifact->identity);
    if (lstat(artifact->path, &status) != 0) {
        if (errno == ENOENT) { yvex_error_clear(err); return YVEX_OK; }
        return lifecycle_refuse(err, YVEX_ERR_IO, "cannot inspect local artifact availability");
    }
    rc = yvex_model_artifact_local_verify(artifact, models_root, &snapshot, err);
    if (rc != YVEX_OK) return rc;
    return evict_verified(artifact->path, &snapshot, dry_run, out, err);
}

static int materialize_commit(yvex_model_remote_selection *selected, const char *root,
                               const char *temporary, yvex_model_storage_result *out, yvex_error *err)
{
    yvex_artifact *artifact = NULL;
    yvex_artifact_options options = {temporary, 1, 0};
    yvex_artifact_file_identity identity;
    yvex_artifact_snapshot verified, committed;
    yvex_artifact_reopen_lease lease;
    char cache[YVEX_PATH_CAP], parent[YVEX_PATH_CAP], *slash;
    int directory = -1, rc = yvex_artifact_open(&artifact, &options, err);
    if (rc == YVEX_OK && yvex_artifact_size(artifact) != selected->remote.size_bytes)
        rc = lifecycle_refuse(err, YVEX_ERR_FORMAT, "download size differs from exact remote artifact");
    if (rc == YVEX_OK) rc = yvex_artifact_identity_read_open(artifact, &identity, err);
    if (rc == YVEX_OK && strcmp(identity.sha256, selected->remote.remote_sha256))
        rc = lifecycle_refuse(err, YVEX_ERR_FORMAT, "download SHA-256 differs from qualified remote artifact");
    if (rc == YVEX_OK) rc = yvex_artifact_snapshot_validate(artifact, &verified, err);
    if (rc == YVEX_OK) rc = yvex_core_mkdir_parent(selected->artifact.path, "model.materialize", err);
    if (rc == YVEX_OK && syscall(SYS_renameat2, AT_FDCWD, temporary, AT_FDCWD,
                                 selected->artifact.path, 1u) != 0)
        rc = lifecycle_refuse(err, YVEX_ERR_IO, "cannot atomically commit acquired artifact without replacement");
    yvex_artifact_close(artifact);
    artifact = NULL;
    if (rc != YVEX_OK) return rc;
    out->changed = 1;
    /* Rename changes ctime. Bind the verified inode at its durable name before admitting reuse. */
    options.path = selected->artifact.path;
    rc = yvex_artifact_open(&artifact, &options, err);
    if (rc == YVEX_OK) rc = yvex_artifact_snapshot_get(artifact, &committed, err);
    if (rc == YVEX_OK && (verified.device != committed.device || verified.inode != committed.inode ||
        verified.size != committed.size || verified.mtime_seconds != committed.mtime_seconds ||
        verified.mtime_nanoseconds != committed.mtime_nanoseconds))
        rc = lifecycle_refuse(err, YVEX_ERR_STATE, "verified artifact changed during durable publication");
    if (rc == YVEX_OK && snprintf(cache, sizeof(cache), "%s/cache/verification", root) >= (int)sizeof(cache))
        rc = lifecycle_refuse(err, YVEX_ERR_BOUNDS, "verification cache path exceeds bound");
    if (rc == YVEX_OK)
        rc = yvex_artifact_reopen_lease_publish(artifact, identity.sha256, cache, &lease, err);
    if (rc == YVEX_OK) {
        yvex_paths paths;
        rc = yvex_paths_default(&paths, err);
        if (rc == YVEX_OK)
            rc = yvex_artifact_reopen_lease_publish(artifact, identity.sha256, paths.cache_dir, &lease, err);
    }
    yvex_artifact_close(artifact);
    yvex_core_text_copy(parent, sizeof(parent), selected->artifact.path);
    slash = strrchr(parent, '/');
    if (slash) *slash = '\0';
    directory = open(parent, O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
    if (directory < 0 || fsync(directory) != 0)
        rc = lifecycle_refuse(err, YVEX_ERR_IO, "artifact committed but durable directory sync failed");
    if (directory >= 0) (void)close(directory);
    if (rc == YVEX_OK) {
        selected->local = selected->verified = 1;
        out->local = out->verified = 1;
        out->logical_bytes = identity.file_size;
    }
    return rc;
}

int yvex_model_remote_materialize(yvex_model_remote_selection *selected,
                                   const char *models_root, int anonymous,
                                   yvex_model_storage_result *out, yvex_error *err)
{
    char directory[YVEX_PATH_CAP], lock[YVEX_PATH_CAP], temporary[YVEX_PATH_CAP], managed[YVEX_PATH_CAP];
    yvex_artifact_snapshot snapshot;
    struct stat status;
    int fd, rc;
    if (!selected || !selected->found || !models_root || !out ||
        !yvex_sha256_hex_is_valid(selected->artifact.identity) ||
        strcmp(selected->artifact.identity, selected->remote.remote_sha256) ||
        strcmp(selected->remote.provider, "huggingface") ||
        !yvex_source_provider_path(managed, sizeof(managed), models_root,
                                    selected->remote.repository, selected->remote.revision))
        return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "exact catalog remote representation required");
    memset(out, 0, sizeof(*out));
    if (snprintf(managed, sizeof(managed), "%s/representations/", models_root) >= (int)sizeof(managed))
        return lifecycle_refuse(err, YVEX_ERR_BOUNDS, "managed path exceeds bound");
    if (strncmp(selected->artifact.path, managed, strlen(managed)) &&
        (snprintf(managed, sizeof(managed), "%s/source/local/%s/model.gguf", models_root,
                   selected->artifact.identity) >= (int)sizeof(managed) || strcmp(selected->artifact.path, managed)))
        return lifecycle_refuse(err, YVEX_ERR_STATE, "rehydration cannot replace an external location");
    if (strstr(selected->artifact.path, "/../"))
        return lifecycle_refuse(err, YVEX_ERR_STATE, "rehydration cannot replace an external location");
    if (snprintf(directory, sizeof(directory), "%s/tmp/acquisitions/%s", models_root,
                 selected->artifact.identity) >= (int)sizeof(directory) ||
        snprintf(lock, sizeof(lock), "%s.lock", directory) >= (int)sizeof(lock) ||
        snprintf(temporary, sizeof(temporary), "%s/%s", directory,
                 selected->remote.filename) >= (int)sizeof(temporary))
        return lifecycle_refuse(err, YVEX_ERR_BOUNDS, "acquisition path exceeds bound");
    rc = yvex_core_mkdir_parent(temporary, "model.materialize", err);
    if (rc != YVEX_OK) return rc;
    /* The provider inherits this lease: parent failure cannot start a competing transfer. */
    fd = open(lock, O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
    if (fd < 0) return lifecycle_refuse(err, YVEX_ERR_IO, "cannot open acquisition lock");
    do { rc = flock(fd, LOCK_EX); } while (rc < 0 && errno == EINTR);
    if (rc != 0) rc = lifecycle_refuse(err, YVEX_ERR_IO, "cannot lock acquisition");
    else if (lstat(selected->artifact.path, &status) == 0) {
        rc = yvex_model_artifact_local_verify(&selected->artifact, models_root, &snapshot, err);
        if (rc == YVEX_OK) {
            selected->local = selected->verified = 1;
            out->local = out->verified = 1;
            out->logical_bytes = snapshot.size;
        }
    } else if (errno != ENOENT) rc = lifecycle_refuse(err, YVEX_ERR_IO, "cannot inspect destination");
    else {
        int complete_pending = lstat(temporary, &status) == 0 && S_ISREG(status.st_mode) &&
            (unsigned long long)status.st_size == selected->remote.size_bytes;
        rc = complete_pending ? YVEX_OK :
            yvex_model_remote_file_download(&selected->remote, directory, anonymous, err);
        if (rc == YVEX_OK) rc = materialize_commit(selected, models_root, temporary, out, err);
    }
    (void)close(fd);
    yvex_core_text_copy(out->path, sizeof(out->path), selected->artifact.path);
    yvex_core_text_copy(out->identity, sizeof(out->identity), selected->artifact.identity);
    return rc;
}

int yvex_model_source_file_materialize(const yvex_local_source_record *source,
                                        const char *models_root, int anonymous,
                                        yvex_model_storage_result *out, yvex_error *err)
{
    yvex_model_remote_selection selected = {0};
    if (!source || strcmp(source->provider, "huggingface") || strcmp(source->storage_kind, "managed") ||
        strcmp(source->format, "gguf") || !strcmp(source->representation, "gguf") ||
        !source->representation[0] || strchr(source->representation, '/') || strstr(source->representation, ".."))
        return lifecycle_refuse(err, YVEX_ERR_INVALID_ARG, "exact managed remote file association required");
    selected.found = 1;
    yvex_core_text_copy(selected.artifact.path, sizeof(selected.artifact.path), source->path);
    yvex_core_text_copy(selected.artifact.identity, sizeof(selected.artifact.identity), source->digest);
    selected.artifact.file_size = source->size_bytes;
    yvex_core_text_copy(selected.remote.provider, sizeof(selected.remote.provider), "huggingface");
    yvex_core_text_copy(selected.remote.repository, sizeof(selected.remote.repository), source->repository);
    yvex_core_text_copy(selected.remote.revision, sizeof(selected.remote.revision), source->revision);
    yvex_core_text_copy(selected.remote.filename, sizeof(selected.remote.filename), source->representation);
    yvex_core_text_copy(selected.remote.remote_sha256, sizeof(selected.remote.remote_sha256), source->digest);
    selected.remote.size_bytes = source->size_bytes;
    return yvex_model_remote_materialize(&selected, models_root, anonymous, out, err);
}
