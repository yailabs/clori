/* Local verification receipts share artifact identity without compiler/runtime dependencies. */
#ifndef INCLUDE_YVEX_INTERNAL_ARTIFACT_STORAGE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_ARTIFACT_STORAGE_H_INCLUDED
#include <yvex/artifact.h>
#ifdef __cplusplus
extern "C" {
#endif
int yvex_artifact_cache_release(const yvex_artifact *artifact,
                                unsigned long long offset,
                                unsigned long long byte_count,
                                yvex_error *err);
/* Upgrade a live byte pin for an exclusive storage operation; refuses concurrent readers. */
int yvex_artifact_pin_exclusive(yvex_artifact *artifact, yvex_error *err);
int yvex_artifact_snapshot_equal(const yvex_artifact_snapshot *left,
                                 const yvex_artifact_snapshot *right);
typedef struct {
    int verified, receipt_present, receipt_valid;
    yvex_artifact_snapshot snapshot;
    char lease_identity[YVEX_SHA256_HEX_CAP], path[YVEX_ARTIFACT_PATH_CAP];
} yvex_artifact_reopen_lease;
int yvex_artifact_reopen_lease_check(
    const yvex_artifact *artifact, const char *artifact_identity, const char *cache_root,
    yvex_artifact_reopen_lease *out, yvex_error *err);
int yvex_artifact_reopen_lease_publish(
    const yvex_artifact *artifact, const char *artifact_identity, const char *cache_root,
    yvex_artifact_reopen_lease *out, yvex_error *err);
#ifdef __cplusplus
}
#endif
#endif
