/*
 * Derive payload identity and publish completed trust atomically.
 *
 * Identity excludes paths, timestamps, and mutable file facts. Local sealing remains distinct from
 * upstream verification.
 */
#include <yvex/internal/core.h>
#include <yvex/internal/source.h>
#include <yvex/internal/source_payload.h>

#include <limits.h>
#include <string.h>

/* Computes stable identity only after every shard owns a verified/sealed digest. */
int yvex_source_payload_identity_compute(yvex_source_payload_session *session,
                                         yvex_source_payload_failure *failure,
                                         yvex_error *err) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char identity[YVEX_SOURCE_PAYLOAD_IDENTITY_CAP];
    yvex_source_payload_trust_class aggregate = YVEX_SOURCE_PAYLOAD_TRUST_UPSTREAM_VERIFIED;
    unsigned long long index;

    if (!session || !session->target_id[0] || !session->family_key[0] ||
        !session->repository_id[0]) {
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT,
                    err, YVEX_ERR_INVALID_ARG,
                               "source_payload_identity", "payload session identity facts are required");
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.source_manifest.v3") ||
        !yvex_sha256_update_text(&hash, session->target_id) ||
        !yvex_sha256_update_text(&hash, session->repository_id) ||
        !yvex_sha256_update_text(&hash, session->verification.revision) ||
        !yvex_sha256_update_u64(&hash, session->verification.source_snapshot_identity) ||
        !yvex_sha256_update_u64(&hash, session->shard_count))
        goto overflow;
    for (index = 0u; index < session->shard_count; ++index) {
        yvex_source_payload_owned_shard *shard = &session->shards[index];
        const char *trust = yvex_source_payload_trust_class_name(shard->public_fact.trust_class);

        if (!shard->observed_digest[0] ||
            shard->public_fact.trust_class == YVEX_SOURCE_PAYLOAD_TRUST_NONE) {
                        return yvex_source_payload_refuse_at(
                            failure,
                            YVEX_SOURCE_PAYLOAD_FAILURE_PAYLOAD_NOT_TRUSTED,
                            index,
                                      ULLONG_MAX, 0u, 0u, 0, err, YVEX_ERR_STATE, "source_payload_identity",
                                      "all shard digests must complete before identity");
        }
        if (shard->public_fact.trust_class == YVEX_SOURCE_PAYLOAD_TRUST_LOCAL_SNAPSHOT_SEALED)
            aggregate = YVEX_SOURCE_PAYLOAD_TRUST_LOCAL_SNAPSHOT_SEALED;
        if (!yvex_sha256_update_u64(&hash, index) || !yvex_sha256_update_text(&hash, shard->name) ||
            !yvex_sha256_update_u64(&hash, shard->public_fact.file_bytes) ||
            !yvex_sha256_update_text(&hash, shard->digest_algorithm) ||
            !yvex_sha256_update_text(&hash, shard->observed_digest) ||
            !yvex_sha256_update_text(&hash, trust))
            goto overflow;
    }
    if (!yvex_sha256_final(&hash, digest))
        goto overflow;
    yvex_sha256_hex(digest, identity);
    if (session->verification.manifest_payload_trusted &&
        session->verification.manifest_payload_identity[0] &&
        strcmp(identity, session->verification.manifest_payload_identity) != 0) {
        pthread_mutex_lock(&session->mutex);
        session->state = YVEX_SOURCE_PAYLOAD_STATE_POISONED;
        session->facts.state = session->state;
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_PAYLOAD_IDENTITY_MISMATCH, err,
                               YVEX_ERR_FORMAT, "source_payload_identity",
                               "observed payload set does not match the published snapshot identity");
    }
    memcpy(session->facts.payload_identity, identity, sizeof(identity));
    session->facts.trust_class = aggregate;
    session->facts.trusted_shard_count = session->shard_count;
    for (index = 0u; index < session->tensor_count; ++index) {
        session->ranges[index].payload_identity = session->facts.payload_identity;
        session->ranges[index].trust_class = aggregate;
    }
    yvex_error_clear(err);
    return YVEX_OK;
overflow:
        return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW, err, YVEX_ERR_BOUNDS,
                           "source_payload_identity", "canonical payload identity input overflow");
}

/*
 * Delegates atomic v3 publication to the canonical source writer owner.
 *
 * Writes only the explicit source payload identity destination through its transaction.
 * Serialization or I/O failure publishes no partial source payload identity result.
 */
int yvex_source_payload_manifest_publish(const yvex_source_payload_session *session,
                                         yvex_error *err) {
    if (!session || !session->manifest_path) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "source_payload_manifest",
                       "payload session and manifest path are required");
        return YVEX_ERR_INVALID_ARG;
    }
    return yvex_source_publish(&(yvex_source_publication_request){
                                   .kind = YVEX_SOURCE_PUBLICATION_PAYLOAD_MANIFEST,
                                   .out_path = session->manifest_path,
                                   .verification = &session->verification,
                                   .payload_session = session,
                               },
                               err);
}
