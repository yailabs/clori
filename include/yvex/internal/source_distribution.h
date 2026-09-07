/* Exact model-source locators, local representation inspection, and bounded distribution. */
#ifndef INCLUDE_YVEX_INTERNAL_SOURCE_DISTRIBUTION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SOURCE_DISTRIBUTION_H_INCLUDED

#include <yvex/core.h>
#include <yvex/artifact.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_SOURCE_LOCATOR_SCHEME_CAP 16u
#define YVEX_SOURCE_LOCATOR_PROVIDER_CAP 32u
#define YVEX_SOURCE_LOCATOR_REPOSITORY_CAP 256u
#define YVEX_SOURCE_LOCATOR_REVISION_CAP 128u
#define YVEX_SOURCE_DISTRIBUTION_NAME_CAP 128u
#define YVEX_SOURCE_DISTRIBUTION_FORMAT_CAP 32u
#define YVEX_SOURCE_DISTRIBUTION_PRECISION_CAP 96u
#define YVEX_SOURCE_DISTRIBUTION_DIGEST_CAP 65u

typedef enum {
    YVEX_SOURCE_LOCATOR_LOCAL_PATH = 0,
    YVEX_SOURCE_LOCATOR_HUGGINGFACE,
    YVEX_SOURCE_LOCATOR_SSH,
    YVEX_SOURCE_LOCATOR_OCI,
    YVEX_SOURCE_LOCATOR_YVEX,
    YVEX_SOURCE_LOCATOR_UNSUPPORTED
} yvex_source_locator_kind;

typedef struct {
    yvex_source_locator_kind kind;
    char scheme[YVEX_SOURCE_LOCATOR_SCHEME_CAP];
    char provider[YVEX_SOURCE_LOCATOR_PROVIDER_CAP];
    char repository[YVEX_SOURCE_LOCATOR_REPOSITORY_CAP];
    char revision[YVEX_SOURCE_LOCATOR_REVISION_CAP];
    char path[YVEX_PATH_CAP];
    char canonical[YVEX_PATH_CAP];
    int revision_present;
    int readable;
    int writable;
} yvex_source_locator;

typedef enum {
    YVEX_SOURCE_STORAGE_MANAGED = 0,
    YVEX_SOURCE_STORAGE_EXTERNAL
} yvex_source_storage_kind;

typedef struct {
    char name[YVEX_SOURCE_DISTRIBUTION_NAME_CAP];
    char format[YVEX_SOURCE_DISTRIBUTION_FORMAT_CAP];
    char precision[YVEX_SOURCE_DISTRIBUTION_PRECISION_CAP];
    char digest[YVEX_SOURCE_DISTRIBUTION_DIGEST_CAP];
    char path[YVEX_PATH_CAP];
    unsigned long long size_bytes;
    unsigned long long file_count;
    int directory;
    yvex_artifact_snapshot snapshot;
    int snapshot_verified;
} yvex_source_representation_fact;

typedef struct {
    const yvex_source_locator *locator;
    const char *models_root;
    const char *name;
    const char *family;
    yvex_source_storage_kind storage;
    const yvex_source_representation_fact *inspected;
} yvex_source_import_options;

typedef struct {
    yvex_source_representation_fact representation;
    char origin_uri[YVEX_PATH_CAP];
    char source_path[YVEX_PATH_CAP];
    char record_path[YVEX_PATH_CAP];
    char storage[24];
    int copied;
    int registered;
} yvex_source_import_result;

typedef struct {
    const yvex_source_locator *locator;
    const char *models_root;
    const char *name;
    const char *family;
    const char *resolved_revision;
    const char *format;
    const char *precision;
    unsigned long long size_bytes;
    int size_known;
    /* Optional exact remote-file association to already verified managed bytes. */
    const yvex_source_representation_fact *local;
    const char *remote_filename;
} yvex_source_reference_options;

typedef struct {
    char record_path[YVEX_PATH_CAP];
    char immutable_uri[YVEX_PATH_CAP];
    int registered;
} yvex_source_reference_result;

typedef struct {
    const char *source_path;
    const char *destination_path;
    const char *expected_digest;
} yvex_source_export_options;

typedef struct {
    char source_path[YVEX_PATH_CAP];
    char destination_path[YVEX_PATH_CAP];
    char digest[YVEX_SOURCE_DISTRIBUTION_DIGEST_CAP];
    unsigned long long bytes;
    int copied;
} yvex_source_export_result;

int yvex_source_locator_parse(const char *text,
                              yvex_source_locator *out,
                              yvex_error *err);
const char *yvex_source_locator_kind_name(yvex_source_locator_kind kind);
/* Resolve prior verified local bytes from source records; a stale receipt falls back to inspection. */
int yvex_source_representation_resolve_local(
    const yvex_source_locator *locator, const char *models_root,
    yvex_source_representation_fact *out, yvex_error *err);
/* Check a previously established regular-file receipt without reading payload bytes. */
int yvex_source_file_reopen(const char *models_root, const char *path,
                             const char *digest, yvex_error *err);
/* Match a sole GGUF member through the authenticated source tree and current file receipts.
 * Explicit adoption only: inspects the selected source directory, never hashes payload bytes. */
int yvex_source_gguf_member_reopen(const char *models_root, const char *path, const char *tree_digest,
                                    const char *member_digest, yvex_source_representation_fact *out,
                                    yvex_error *err);
/* Establish or reuse byte verification; emits only rebuildable verification receipts. */
int yvex_source_representation_verify_local(
    const yvex_source_locator *locator, const char *models_root, const char *expected_digest,
    yvex_source_representation_fact *out, yvex_error *err);
int yvex_source_acquisition_reopen(const char *record_path, const char *models_root,
                                    const char *repository, const char *revision,
                                    const char *const *includes, unsigned int include_count,
                                    const char *const *excludes, unsigned int exclude_count,
                                    yvex_source_representation_fact *out, yvex_error *err);
/* Provider processes inherit the lease so a parent crash cannot permit a competing acquisition. */
/* Stage bytes without establishing identity; the caller must verify before catalog publication. */
int yvex_source_stage_file(const char *source, const char *destination, yvex_error *err);
int yvex_source_selection_identity(const char *const *includes, size_t include_count,
                                    const char *const *excludes, size_t exclude_count,
                                    char out[YVEX_SHA256_HEX_CAP], yvex_error *err);
int yvex_source_acquisition_lock(const char *models_root, const char *repository,
                                  const char *revision, int *descriptor, yvex_error *err);
typedef struct {
    unsigned long long logical_bytes, allocated_bytes;
    int changed, local;
    char pending_path[YVEX_PATH_CAP];
} yvex_source_eviction_result;
/* Only an exact catalog-owned immutable provider acquisition is eligible. */
int yvex_source_evict_local(const char *models_root, const char *repository, const char *revision,
                             const char *path, const char *digest, int dry_run,
                             yvex_source_eviction_result *out, yvex_error *err);
int yvex_source_import_local(const yvex_source_import_options *options,
                             yvex_source_import_result *out,
                             yvex_error *err);
int yvex_source_register_reference(const yvex_source_reference_options *options,
                                   yvex_source_reference_result *out,
                                   yvex_error *err);
int yvex_source_export_local(const yvex_source_export_options *options,
                             yvex_source_export_result *out,
                             yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_SOURCE_DISTRIBUTION_H_INCLUDED */
