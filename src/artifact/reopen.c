/*
 * Persist a rebuildable local lease for one fully verified artifact snapshot.
 *
 * The receipt is content-addressed by the expected artifact identity and every stable filesystem
 * snapshot fact. A hit proves only that the same local inode still carries bytes verified by an
 * earlier complete admission; the runtime binding remains the semantic authority. Missing,
 * malformed, or stale cache data must fall back to a complete byte verification.
 */
#include <yvex/internal/artifact.h>
#include <yvex/internal/core.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REOPEN_LEASE_SCHEMA 1ull
#define REOPEN_LEASE_MAGIC "YVARL001"
#define REOPEN_LEASE_MAGIC_BYTES 8u
#define REOPEN_LEASE_IDENTITY_BYTES 64u
#define REOPEN_LEASE_SNAPSHOT_FIELDS 7u
#define REOPEN_LEASE_BYTES                                                        \
    (REOPEN_LEASE_MAGIC_BYTES + 8u + 2u * REOPEN_LEASE_IDENTITY_BYTES +          \
     REOPEN_LEASE_SNAPSHOT_FIELDS * 8u)

typedef struct {
    const char *artifact_identity;
    const yvex_artifact_reopen_lease *lease;
} reopen_validation;

static void reopen_put_u64(unsigned char *bytes, size_t offset, unsigned long long value)
{
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        bytes[offset + index] = (unsigned char)(value >> (index * 8u));
}

static unsigned long long reopen_get_u64(const unsigned char *bytes, size_t offset)
{
    unsigned long long value = 0ull;
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        value |= (unsigned long long)bytes[offset + index] << (index * 8u);
    return value;
}

static int reopen_identity_build(const char *artifact_identity,
                                 const yvex_artifact_snapshot *snapshot,
                                 char identity[YVEX_SHA256_HEX_CAP])
{
    const unsigned long long fields[] = {
        snapshot ? snapshot->device : 0ull,
        snapshot ? snapshot->inode : 0ull,
        snapshot ? snapshot->size : 0ull,
        snapshot ? (unsigned long long)snapshot->mtime_seconds : 0ull,
        snapshot ? (unsigned long long)snapshot->mtime_nanoseconds : 0ull,
        snapshot ? (unsigned long long)snapshot->ctime_seconds : 0ull,
        snapshot ? (unsigned long long)snapshot->ctime_nanoseconds : 0ull,
    };
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    size_t index;

    if (!yvex_sha256_hex_valid(artifact_identity) || !snapshot || !identity)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.artifact.reopen-lease.v1") ||
        !yvex_sha256_update_u64(&hash, REOPEN_LEASE_SCHEMA) ||
        !yvex_sha256_update_text(&hash, artifact_identity))
        return 0;
    for (index = 0u; index < sizeof(fields) / sizeof(fields[0]); ++index)
        if (!yvex_sha256_update_u64(&hash, fields[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, identity);
    return 1;
}

static int reopen_path_build(const char *cache_root, const char *artifact_identity,
                             const char *lease_identity, char path[YVEX_ARTIFACT_PATH_CAP])
{
    int count;
    if (!cache_root || !cache_root[0] || !yvex_sha256_hex_valid(artifact_identity) ||
        !yvex_sha256_hex_valid(lease_identity) || !path)
        return 0;
    count = snprintf(path, YVEX_ARTIFACT_PATH_CAP,
                     "%s/artifact-reopen/%s/%s.lease", cache_root,
                     artifact_identity, lease_identity);
    return count > 0 && count < YVEX_ARTIFACT_PATH_CAP;
}

static int reopen_lease_seed(const yvex_artifact *artifact, const char *artifact_identity,
                             const char *cache_root, yvex_artifact_reopen_lease *out,
                             yvex_error *err)
{
    if (!artifact || !yvex_sha256_hex_valid(artifact_identity) || !cache_root ||
        !cache_root[0] || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "artifact.reopen-lease",
                       "artifact, identity, cache root, and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    if (yvex_artifact_snapshot_get(artifact, &out->snapshot, err) != YVEX_OK ||
        yvex_artifact_snapshot_validate(artifact, NULL, err) != YVEX_OK)
        return yvex_error_code(err);
    if (!reopen_identity_build(artifact_identity, &out->snapshot, out->lease_identity) ||
        !reopen_path_build(cache_root, artifact_identity, out->lease_identity, out->path)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "artifact.reopen-lease",
                       "verified-reopen lease identity or path exceeds its bound");
        return YVEX_ERR_BOUNDS;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static void reopen_encode(unsigned char bytes[REOPEN_LEASE_BYTES],
                          const char *artifact_identity,
                          const yvex_artifact_reopen_lease *lease)
{
    const yvex_artifact_snapshot *snapshot = &lease->snapshot;
    const unsigned long long fields[] = {
        snapshot->device, snapshot->inode, snapshot->size,
        (unsigned long long)snapshot->mtime_seconds,
        (unsigned long long)snapshot->mtime_nanoseconds,
        (unsigned long long)snapshot->ctime_seconds,
        (unsigned long long)snapshot->ctime_nanoseconds,
    };
    size_t index, offset = 0u;

    memset(bytes, 0, REOPEN_LEASE_BYTES);
    memcpy(bytes + offset, REOPEN_LEASE_MAGIC, REOPEN_LEASE_MAGIC_BYTES);
    offset += REOPEN_LEASE_MAGIC_BYTES;
    reopen_put_u64(bytes, offset, REOPEN_LEASE_SCHEMA);
    offset += 8u;
    memcpy(bytes + offset, artifact_identity, REOPEN_LEASE_IDENTITY_BYTES);
    offset += REOPEN_LEASE_IDENTITY_BYTES;
    memcpy(bytes + offset, lease->lease_identity, REOPEN_LEASE_IDENTITY_BYTES);
    offset += REOPEN_LEASE_IDENTITY_BYTES;
    for (index = 0u; index < sizeof(fields) / sizeof(fields[0]); ++index) {
        reopen_put_u64(bytes, offset, fields[index]);
        offset += 8u;
    }
}

static int reopen_decode(const unsigned char *bytes, size_t count,
                         const reopen_validation *expected)
{
    yvex_artifact_snapshot snapshot;
    char artifact_identity[YVEX_SHA256_HEX_CAP] = {0};
    char lease_identity[YVEX_SHA256_HEX_CAP] = {0};
    char derived[YVEX_SHA256_HEX_CAP];
    size_t offset = REOPEN_LEASE_MAGIC_BYTES;

    if (!bytes || count != REOPEN_LEASE_BYTES || !expected || !expected->lease ||
        memcmp(bytes, REOPEN_LEASE_MAGIC, REOPEN_LEASE_MAGIC_BYTES) != 0 ||
        reopen_get_u64(bytes, offset) != REOPEN_LEASE_SCHEMA)
        return 0;
    offset += 8u;
    memcpy(artifact_identity, bytes + offset, REOPEN_LEASE_IDENTITY_BYTES);
    offset += REOPEN_LEASE_IDENTITY_BYTES;
    memcpy(lease_identity, bytes + offset, REOPEN_LEASE_IDENTITY_BYTES);
    offset += REOPEN_LEASE_IDENTITY_BYTES;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.device = reopen_get_u64(bytes, offset);
    offset += 8u;
    snapshot.inode = reopen_get_u64(bytes, offset);
    offset += 8u;
    snapshot.size = reopen_get_u64(bytes, offset);
    offset += 8u;
    snapshot.mtime_seconds = (long long)reopen_get_u64(bytes, offset);
    offset += 8u;
    snapshot.mtime_nanoseconds = (long long)reopen_get_u64(bytes, offset);
    offset += 8u;
    snapshot.ctime_seconds = (long long)reopen_get_u64(bytes, offset);
    offset += 8u;
    snapshot.ctime_nanoseconds = (long long)reopen_get_u64(bytes, offset);
    return yvex_sha256_hex_valid(artifact_identity) &&
           yvex_sha256_hex_valid(lease_identity) &&
           strcmp(artifact_identity, expected->artifact_identity) == 0 &&
           yvex_artifact_snapshot_equal(&snapshot, &expected->lease->snapshot) &&
           reopen_identity_build(artifact_identity, &snapshot, derived) &&
           strcmp(lease_identity, derived) == 0 &&
           strcmp(lease_identity, expected->lease->lease_identity) == 0;
}

static int reopen_candidate_validate(int descriptor, size_t count, void *context,
                                     yvex_error *err)
{
    unsigned char *bytes = NULL;
    yvex_core_file_result result;
    int rc = yvex_core_file_read_descriptor_snapshot(
        descriptor, count, &bytes, &result, err);
    if (rc == YVEX_OK && !reopen_decode(bytes, count, context)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "artifact.reopen-lease",
                       "verified-reopen lease candidate is not canonical");
        rc = YVEX_ERR_FORMAT;
    }
    free(bytes);
    return rc;
}

int yvex_artifact_reopen_lease_check(
    const yvex_artifact *artifact, const char *artifact_identity, const char *cache_root,
    yvex_artifact_reopen_lease *out, yvex_error *err)
{
    yvex_core_file_result result;
    reopen_validation validation;
    unsigned char *bytes = NULL;
    size_t count = 0u;
    int rc = reopen_lease_seed(artifact, artifact_identity, cache_root, out, err);

    if (rc != YVEX_OK) return rc;
    validation.artifact_identity = artifact_identity;
    validation.lease = out;
    rc = yvex_core_file_read_snapshot(
        out->path, REOPEN_LEASE_BYTES, &bytes, &count, &result, err);
    if (rc != YVEX_OK) {
        out->receipt_present = result.system_error != ENOENT;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    out->receipt_present = 1;
    out->receipt_valid = reopen_decode(bytes, count, &validation);
    out->verified = out->receipt_valid;
    free(bytes);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_artifact_reopen_lease_publish(
    const yvex_artifact *artifact, const char *artifact_identity, const char *cache_root,
    yvex_artifact_reopen_lease *out, yvex_error *err)
{
    yvex_core_file_result result;
    reopen_validation validation;
    unsigned char bytes[REOPEN_LEASE_BYTES];
    int rc = reopen_lease_seed(artifact, artifact_identity, cache_root, out, err);

    if (rc != YVEX_OK) return rc;
    memset(&result, 0, sizeof(result));
    reopen_encode(bytes, artifact_identity, out);
    validation.artifact_identity = artifact_identity;
    validation.lease = out;
    rc = yvex_core_mkdir_parent(out->path, "artifact.reopen-lease", err);
    if (rc == YVEX_OK)
        rc = yvex_core_file_publish_noreplace(
            out->path, bytes, sizeof(bytes), NULL, reopen_candidate_validate,
            &validation, &result, err);
    if (rc == YVEX_ERR_STATE && result.stage == YVEX_CORE_FILE_STAGE_CONFLICT) {
        yvex_error_clear(err);
        rc = yvex_artifact_reopen_lease_check(
            artifact, artifact_identity, cache_root, out, err);
        if (rc == YVEX_OK && !out->verified) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "artifact.reopen-lease",
                           "existing verified-reopen lease is invalid");
            return YVEX_ERR_FORMAT;
        }
        return rc;
    }
    if (rc != YVEX_OK) return rc;
    out->receipt_present = out->receipt_valid = out->verified = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
