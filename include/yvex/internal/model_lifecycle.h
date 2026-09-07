/* Model catalog projects local storage operations without replacing identity owners. */
#ifndef INCLUDE_YVEX_INTERNAL_MODEL_LIFECYCLE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MODEL_LIFECYCLE_H_INCLUDED
#include <yvex/catalog.h>
#include <yvex/internal/artifact_storage.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char path[YVEX_PATH_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
    unsigned long long logical_bytes, allocated_bytes;
    int local, verified, changed;
} yvex_model_storage_result;

typedef struct {
    char path[YVEX_PATH_CAP];
    char role[40];
    unsigned long long logical_bytes, allocated_bytes, files, shared_files;
    int exists, recursive, attributable;
} yvex_model_storage_row;
typedef struct {
    yvex_model_storage_row *rows;
    size_t count;
    unsigned long long logical_bytes, allocated_bytes;
} yvex_model_storage_report;
/* Explicit inspection only: walks selected roots, counts each inode allocation once.
 * Reflink extent sharing and historical peaks are not inferred from st_blocks. */
int yvex_model_storage_inspect(const yvex_model_library *library,
                                unsigned long long model_index,
                                const char *models_root, int include_caches,
                                yvex_model_storage_report *out, yvex_error *err);
void yvex_model_storage_report_free(yvex_model_storage_report *report);

typedef struct {
    yvex_model_artifact_fact artifact;
    yvex_model_publication remote;
    char logical_identity[YVEX_MODEL_LIBRARY_ID_CAP];
    char model[YVEX_MODEL_LIBRARY_NAME_CAP];
    char family[YVEX_REMOTE_FAMILY_CAP];
    int found, local, verified;
} yvex_model_remote_selection;
int yvex_model_local_content_resolve(const char *models_root, const char *digest,
                                      yvex_model_remote_selection *out, yvex_error *err);
int yvex_model_remote_selection_resolve(const char *models_root, const char *repository,
                                         const char *revision, const char *variant,
                                         const char *format,
                                         yvex_model_remote_selection *out, yvex_error *err);

int yvex_model_remote_inspect_policy(yvex_remote_catalog **out,
                                      const yvex_remote_inspect_options *options,
                                      int anonymous, yvex_error *err);
const char *yvex_model_remote_file_sha256(const yvex_remote_catalog *catalog,
                                          unsigned long long model_index, unsigned int file_index);
int yvex_model_remote_adopt_existing(const yvex_remote_catalog *catalog,
                                      const yvex_model_representation *representation,
                                      const char *models_root, int dry_run,
                                      yvex_model_remote_selection *out, yvex_error *err);
int yvex_model_remote_file_download(const yvex_model_publication *remote,
                                    const char *directory, int anonymous, yvex_error *err);
int yvex_model_remote_materialize(yvex_model_remote_selection *selected,
                                   const char *models_root, int anonymous,
                                   yvex_model_storage_result *out, yvex_error *err);
int yvex_model_source_file_materialize(const yvex_local_source_record *source,
                                        const char *models_root, int anonymous,
                                        yvex_model_storage_result *out, yvex_error *err);

int yvex_model_remote_revision_resolve(const char *models_root, const char *repository,
                                         char revision[YVEX_REMOTE_REVISION_CAP], yvex_error *err);
int yvex_model_remote_source_resolve(const char *models_root, const char *repository,
                                      const char *revision, const char *variant, const char *format,
                                      yvex_local_source_record *out, int *found, yvex_error *err);

int yvex_model_artifact_is_managed(const yvex_model_library *library, unsigned long long model_index,
                                    const yvex_model_artifact_fact *artifact, const char *models_root);

/* Metadata-only verification: absence of a current receipt is explicit, never a hidden full hash. */
int yvex_model_artifact_local_verify(const yvex_model_artifact_fact *artifact,
                                     const char *models_root,
                                     yvex_artifact_snapshot *snapshot, yvex_error *err);
int yvex_model_source_evict(const yvex_model_library *library,
                             unsigned long long model_index, unsigned long long source_index,
                             const char *models_root, int dry_run,
                             yvex_model_storage_result *out, yvex_error *err);
int yvex_model_local_evict(const yvex_model_library *library,
                            unsigned long long model_index, unsigned long long artifact_index,
                            const char *models_root, int dry_run,
                            yvex_model_storage_result *out, yvex_error *err);
#ifdef __cplusplus
}
#endif
#endif
