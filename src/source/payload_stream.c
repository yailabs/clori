/*
 * Deliver planned ranges transactionally and verify shard trust.
 *
 * Commit follows exact completion and post-read identity validation. Streamed source bytes are not
 * emitted artifact tensors.
 */
#define _GNU_SOURCE
#include <yvex/internal/core.h>
#include <yvex/internal/source_payload.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <time.h>

static int payload_stream_identity_matches(const yvex_source_payload_file_identity *identity,
                                           const struct stat *status) {
    return identity->device == status->st_dev && identity->inode == status->st_ino &&
           identity->size == status->st_size && identity->mtime.tv_sec == status->st_mtim.tv_sec &&
           identity->mtime.tv_nsec == status->st_mtim.tv_nsec &&
           identity->ctime.tv_sec == status->st_ctim.tv_sec &&
           identity->ctime.tv_nsec == status->st_ctim.tv_nsec;
}

static int payload_sink_valid(const yvex_source_payload_sink *sink) {
    return sink && sink->begin && sink->chunk && sink->commit && sink->abort;
}

static int payload_stream_enter(yvex_source_payload_session *session,
                                size_t bytes,
                                yvex_source_payload_state required,
                                yvex_source_payload_state transition,
                                yvex_source_payload_failure *failure,
                                yvex_error *err) {
    int was_cancelled;

    pthread_mutex_lock(&session->mutex);
    if (session->state != required || session->cancelled) {
        yvex_source_payload_failure_code code;

        was_cancelled = session->cancelled;
        code = was_cancelled ? YVEX_SOURCE_PAYLOAD_FAILURE_CANCELLED
               : required == YVEX_SOURCE_PAYLOAD_STATE_READY
                   ? YVEX_SOURCE_PAYLOAD_FAILURE_PAYLOAD_NOT_TRUSTED
                   : YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_STATE;
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(failure, code, ULLONG_MAX, ULLONG_MAX, bytes, 0u, 0, err,
                                was_cancelled ? YVEX_ERR_CANCELLED : YVEX_ERR_STATE, "source_payload_stream",
                                "payload session is not admissible for this stream");
    }
    if (session->facts.active_streams >= session->budget.maximum_streams ||
        bytes > session->budget.maximum_inflight_host_bytes - session->inflight_host_bytes) {
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_RESOURCE_BUDGET,
                    ULLONG_MAX, ULLONG_MAX,
                                bytes, 0u, 0, err, YVEX_ERR_BOUNDS, "source_payload_stream",
                                "payload stream resource budget is exhausted");
    }
    session->facts.active_streams++;
    if (session->facts.active_streams > session->facts.peak_active_streams)
        session->facts.peak_active_streams = session->facts.active_streams;
    session->inflight_host_bytes += bytes;
    if (session->inflight_host_bytes > session->facts.peak_inflight_host_bytes)
        session->facts.peak_inflight_host_bytes = session->inflight_host_bytes;
    if (transition != required) {
        session->state = transition;
        session->facts.state = transition;
    }
    pthread_mutex_unlock(&session->mutex);
    return YVEX_OK;
}

static void payload_stream_leave(yvex_source_payload_session *session, size_t bytes, int failed) {
    pthread_mutex_lock(&session->mutex);
    if (session->facts.active_streams)
        session->facts.active_streams--;
    if (session->inflight_host_bytes >= bytes)
        session->inflight_host_bytes -= bytes;
    else
        session->inflight_host_bytes = 0u;
    if (failed)
        session->facts.failed_streams++;
    pthread_mutex_unlock(&session->mutex);
}

static void payload_stream_abort(const yvex_source_payload_sink *sink,
                                 yvex_source_payload_failure *failure,
                                 yvex_source_payload_stream_result *result) {
    result->complete = 0;
    result->committed = 0;
    result->aborted = 1;
    if (sink && sink->abort)
        sink->abort(sink->context, failure, result);
}

static int payload_stream_validate_after(yvex_source_payload_session *session,
                                         unsigned long long shard_index,
                                         int fd,
                                         yvex_source_payload_failure *failure,
                                         yvex_error *err) {
    struct stat status;
    struct stat path_status;

    if (session->ops.fstat_fn(fd, &status) != 0) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_STAT,
                    shard_index, ULLONG_MAX, 0u, 0u,
                                errno, err, YVEX_ERR_IO, "source_payload_stream",
                                "post-read source shard stat failed");
    }
    if (session->ops.fstatat_fn(session->root_fd,
                                session->shards[shard_index].name,
                                &path_status,
                                AT_SYMLINK_NOFOLLOW) != 0 ||
        !S_ISREG(path_status.st_mode) ||
        !payload_stream_identity_matches(&session->shards[shard_index].admitted_identity,
                                         &status) ||
        status.st_dev != path_status.st_dev || status.st_ino != path_status.st_ino) {
        pthread_mutex_lock(&session->mutex);
        session->state = YVEX_SOURCE_PAYLOAD_STATE_POISONED;
        session->facts.state = session->state;
        session->facts.identity_drifts++;
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_DRIFT,
                    shard_index, ULLONG_MAX, 0u,
                                0u, 0, err, YVEX_ERR_STATE, "source_payload_stream",
                                "source shard drifted during payload read");
    }
    return YVEX_OK;
}

static int payload_stream_deliver(yvex_source_payload_session *session,
                                  const yvex_source_payload_sink *sink,
                                  const yvex_source_payload_chunk *chunk,
                                  const unsigned char *buffer,
                                  yvex_source_payload_stream_result *result,
                                  yvex_source_payload_failure *failure,
                                  yvex_error *err) {
    result->chunks_attempted++;
    if (sink->chunk(sink->context, chunk, buffer) != 0) {
                return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_CONSUMER, chunk->shard_index,
                                chunk->source_tensor_index, chunk->byte_length, result->delivered_logical_bytes,
                                0, err, YVEX_ERR_STATE, "source_payload_sink",
                                "payload consumer refused a bounded chunk");
    }
    if (!yvex_core_u64_add(result->delivered_logical_bytes,
                           (unsigned long long)chunk->byte_length,
                           &result->delivered_logical_bytes)) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                    chunk->shard_index,
                                chunk->source_tensor_index, chunk->byte_length, result->delivered_logical_bytes,
                                0, err, YVEX_ERR_BOUNDS, "source_payload_sink",
                                "payload delivery accounting overflow");
    }
    result->chunks_completed++;
    pthread_mutex_lock(&session->mutex);
    if (session->cancelled) {
        session->facts.cancellations++;
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_CANCELLED, chunk->shard_index,
                                chunk->source_tensor_index, chunk->byte_length, result->delivered_logical_bytes,
                                0, err, YVEX_ERR_CANCELLED, "source_payload_sink",
                                "source payload stream cancelled after chunk delivery");
    }
    pthread_mutex_unlock(&session->mutex);
    return YVEX_OK;
}

int yvex_source_payload_session_stream(yvex_source_payload_session *session,
                                       const yvex_source_payload_plan *plan,
                                       const yvex_source_payload_sink *sink,
                                       yvex_source_payload_stream_result *result,
                                       yvex_source_payload_failure *failure,
                                       yvex_error *err) {
    unsigned char *buffer = NULL;
    unsigned long long ordinal;
    unsigned long long pinned_shard = ULLONG_MAX;
    int fd = -1;
    int rc;
    int entered = 0;
    int began = 0;
    unsigned int buffer_slot = UINT_MAX;

    if (result)
        memset(result, 0, sizeof(*result));
    if (!session || !plan || plan->session != session || !payload_sink_valid(sink) || !result) {
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT,
                    err, YVEX_ERR_INVALID_ARG,
                             "source_payload_stream",
                             "session, owned plan, transactional sink, and result are required");
    }
    result->requested_logical_bytes = plan->summary.logical_bytes;
    rc = payload_stream_enter(session,
                              plan->summary.chunk_bytes,
                              YVEX_SOURCE_PAYLOAD_STATE_READY,
                              YVEX_SOURCE_PAYLOAD_STATE_READY,
                              failure,
                              err);
    if (rc != YVEX_OK)
        return rc;
    entered = 1;
    rc = yvex_source_payload_buffer_acquire(
        session, plan->summary.chunk_bytes, &buffer, &buffer_slot, failure, err);
    if (rc != YVEX_OK)
        goto fail;
    began = 1;
    if (sink->begin(sink->context, &plan->summary) != 0) {
        yvex_source_payload_fail(failure,
                                 YVEX_SOURCE_PAYLOAD_FAILURE_CONSUMER,
                                 ULLONG_MAX,
                                 ULLONG_MAX,
                                 plan->summary.logical_bytes,
                                 0u,
                                 0,
                                 err,
                                 YVEX_ERR_STATE,
                                 "source_payload_sink",
                                 "payload consumer refused stream begin");
        rc = YVEX_ERR_STATE;
        goto fail;
    }
    for (ordinal = 0u; ordinal < plan->summary.chunk_count; ++ordinal) {
        const yvex_source_payload_chunk *chunk = &plan->chunks[ordinal];

        if (chunk->shard_index != pinned_shard) {
            if (pinned_shard != ULLONG_MAX) {
                rc = payload_stream_validate_after(session, pinned_shard, fd, failure, err);
                yvex_source_payload_handle_release(session, pinned_shard);
                pinned_shard = ULLONG_MAX;
                fd = -1;
                if (rc != YVEX_OK)
                    goto fail;
            }
            rc = yvex_source_payload_handle_acquire(session, chunk->shard_index, &fd, failure, err);
            if (rc != YVEX_OK)
                goto fail;
            pinned_shard = chunk->shard_index;
        }
        rc = yvex_source_payload_exact_read(session,
                                            chunk->shard_index,
                                            fd,
                                            chunk->absolute_offset,
                                            buffer,
                                            chunk->byte_length,
                                            result,
                                            failure,
                                            err);
        if (rc != YVEX_OK)
            goto fail;
        rc = payload_stream_deliver(session, sink, chunk, buffer, result, failure, err);
        if (rc != YVEX_OK)
            goto fail;
    }
    if (pinned_shard != ULLONG_MAX) {
        rc = payload_stream_validate_after(session, pinned_shard, fd, failure, err);
        yvex_source_payload_handle_release(session, pinned_shard);
        pinned_shard = ULLONG_MAX;
        fd = -1;
        if (rc != YVEX_OK)
            goto fail;
    }
    if (result->delivered_logical_bytes != result->requested_logical_bytes ||
        result->chunks_completed != plan->summary.chunk_count) {
        yvex_source_payload_fail(failure,
                                 YVEX_SOURCE_PAYLOAD_FAILURE_SHORT_READ,
                                 ULLONG_MAX,
                                 ULLONG_MAX,
                                 result->requested_logical_bytes,
                                 result->delivered_logical_bytes,
                                 0,
                                 err,
                                 YVEX_ERR_IO,
                                 "source_payload_stream",
                                 "payload plan did not complete exactly");
        rc = YVEX_ERR_IO;
        goto fail;
    }
    result->complete = 1;
    if (sink->commit(sink->context, result) != 0) {
        yvex_source_payload_fail(failure,
                                 YVEX_SOURCE_PAYLOAD_FAILURE_CONSUMER,
                                 ULLONG_MAX,
                                 ULLONG_MAX,
                                 result->requested_logical_bytes,
                                 result->delivered_logical_bytes,
                                 0,
                                 err,
                                 YVEX_ERR_STATE,
                                 "source_payload_sink",
                                 "payload consumer refused stream commit");
        rc = YVEX_ERR_STATE;
        goto fail;
    }
    result->committed = 1;
    yvex_source_payload_buffer_release(session, buffer_slot);
    payload_stream_leave(session, plan->summary.chunk_bytes, 0);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
fail:
    if (pinned_shard != ULLONG_MAX)
        yvex_source_payload_handle_release(session, pinned_shard);
    if (began)
        payload_stream_abort(sink, failure, result);
    if (buffer_slot != UINT_MAX)
        yvex_source_payload_buffer_release(session, buffer_slot);
    if (entered)
        payload_stream_leave(session, plan->summary.chunk_bytes, 1);
    return rc;
}

static int payload_verify_hash_span(yvex_source_payload_session *session,
                                    unsigned long long shard_index,
                                    int fd,
                                    unsigned long long begin,
                                    unsigned long long end,
                                    unsigned char *buffer,
                                    size_t buffer_bytes,
                                    yvex_sha256 *hash,
                                    yvex_source_payload_stream_result *result,
                                    yvex_source_payload_failure *failure,
                                    yvex_error *err) {
    unsigned long long offset = begin;

    while (offset < end) {
        unsigned long long remaining = end - offset;
        size_t length =
            remaining > (unsigned long long)buffer_bytes ? buffer_bytes : (size_t)remaining;
        int rc = yvex_source_payload_exact_read(
            session, shard_index, fd, offset, buffer, length, result, failure, err);
        if (rc != YVEX_OK)
            return rc;
        if (!yvex_sha256_update(hash, buffer, length)) {
                        return yvex_source_payload_refuse_at(
                            failure,
                            YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                            shard_index,
                                    ULLONG_MAX, length, 0u, 0, err, YVEX_ERR_BOUNDS, "source_payload_verify",
                                    "payload digest length overflow");
        }
        offset += (unsigned long long)length;
    }
    return YVEX_OK;
}

static int payload_verify_shard(yvex_source_payload_session *session,
                                const yvex_source_payload_plan *plan,
                                const yvex_source_payload_sink *sink,
                                yvex_source_payload_stream_result *result,
                                yvex_source_payload_failure *failure,
                                yvex_error *err,
                                unsigned char *buffer,
                                size_t buffer_bytes,
                                unsigned long long shard_index,
                                unsigned long long *chunk_ordinal) {
    yvex_source_payload_owned_shard *shard = &session->shards[shard_index];
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long cursor = 0u;
    int fd = -1;
    int rc = yvex_source_payload_handle_acquire(session, shard_index, &fd, failure, err);

    if (rc != YVEX_OK)
        return rc;
    yvex_sha256_init(&hash);
    while (plan && *chunk_ordinal < plan->summary.chunk_count &&
           plan->chunks[*chunk_ordinal].shard_index == shard_index) {
        const yvex_source_payload_chunk *chunk = &plan->chunks[*chunk_ordinal];

        if (chunk->absolute_offset < cursor) {
            yvex_source_payload_fail(failure,
                                     YVEX_SOURCE_PAYLOAD_FAILURE_DUPLICATE_TENSOR,
                                     shard_index,
                                     chunk->source_tensor_index,
                                     chunk->byte_length,
                                     result->delivered_logical_bytes,
                                     0,
                                     err,
                                     YVEX_ERR_FORMAT,
                                     "source_payload_verify",
                                     "delivery plan overlaps physical source ranges");
            rc = YVEX_ERR_FORMAT;
            goto done;
        }
        rc = payload_verify_hash_span(session,
                                      shard_index,
                                      fd,
                                      cursor,
                                      chunk->absolute_offset,
                                      buffer,
                                      buffer_bytes,
                                      &hash,
                                      result,
                                      failure,
                                      err);
        if (rc != YVEX_OK)
            goto done;
        rc = yvex_source_payload_exact_read(session,
                                            shard_index,
                                            fd,
                                            chunk->absolute_offset,
                                            buffer,
                                            chunk->byte_length,
                                            result,
                                            failure,
                                            err);
        if (rc != YVEX_OK)
            goto done;
        if (!yvex_sha256_update(&hash, buffer, chunk->byte_length)) {
            yvex_source_payload_fail(failure,
                                     YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                                     shard_index,
                                     chunk->source_tensor_index,
                                     chunk->byte_length,
                                     result->delivered_logical_bytes,
                                     0,
                                     err,
                                     YVEX_ERR_BOUNDS,
                                     "source_payload_verify",
                                     "payload digest length overflow");
            rc = YVEX_ERR_BOUNDS;
            goto done;
        }
        rc = payload_stream_deliver(session, sink, chunk, buffer, result, failure, err);
        if (rc != YVEX_OK)
            goto done;
        cursor = chunk->absolute_offset + (unsigned long long)chunk->byte_length;
        (*chunk_ordinal)++;
    }
    rc = payload_verify_hash_span(session,
                                  shard_index,
                                  fd,
                                  cursor,
                                  shard->public_fact.file_bytes,
                                  buffer,
                                  buffer_bytes,
                                  &hash,
                                  result,
                                  failure,
                                  err);
    if (rc == YVEX_OK)
        rc = payload_stream_validate_after(session, shard_index, fd, failure, err);
done:
    yvex_source_payload_handle_release(session, shard_index);
    if (rc != YVEX_OK)
        return rc;
    if (!yvex_sha256_final(&hash, digest)) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                    shard_index, ULLONG_MAX,
                                shard->public_fact.file_bytes, 0u, 0, err, YVEX_ERR_BOUNDS,
                                "source_payload_verify", "payload digest finalization failed");
    }
    yvex_sha256_hex(digest, shard->observed_digest);
    if (shard->expected_digest[0] && strcmp(shard->expected_digest, shard->observed_digest) != 0) {
        pthread_mutex_lock(&session->mutex);
        session->facts.digest_mismatches++;
        session->state = YVEX_SOURCE_PAYLOAD_STATE_POISONED;
        session->facts.state = session->state;
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_DIGEST_MISMATCH,
                    shard_index, ULLONG_MAX,
                                shard->public_fact.file_bytes, 0u, 0, err, YVEX_ERR_FORMAT,
                                "source_payload_verify",
                                "source shard SHA-256 does not match authoritative digest");
    }
    shard->public_fact.trust_class = shard->expected_digest[0]
                                         ? YVEX_SOURCE_PAYLOAD_TRUST_UPSTREAM_VERIFIED
                                         : YVEX_SOURCE_PAYLOAD_TRUST_LOCAL_SNAPSHOT_SEALED;
    return YVEX_OK;
}

static void payload_verify_reset_trust(yvex_source_payload_session *session) {
    unsigned long long index;

    session->facts.trust_class = YVEX_SOURCE_PAYLOAD_TRUST_NONE;
    session->facts.trusted_shard_count = 0u;
    session->facts.payload_identity[0] = '\0';
    for (index = 0u; index < session->shard_count; ++index)
        session->shards[index].public_fact.trust_class = YVEX_SOURCE_PAYLOAD_TRUST_NONE;
    for (index = 0u; index < session->tensor_count; ++index) {
        session->ranges[index].payload_identity = session->facts.payload_identity;
        session->ranges[index].trust_class = YVEX_SOURCE_PAYLOAD_TRUST_NONE;
    }
}

int yvex_source_payload_session_verify(yvex_source_payload_session *session,
                                       const yvex_source_payload_plan *delivery_plan,
                                       const yvex_source_payload_sink *sink,
                                       yvex_source_payload_stream_result *result,
                                       yvex_source_payload_failure *failure,
                                       yvex_error *err) {
    unsigned char *buffer = NULL;
    unsigned long long shard_index;
    unsigned long long chunk_ordinal = 0u;
    size_t buffer_bytes;
    int began = 0;
    int entered = 0;
    int rc;
    unsigned int buffer_slot = UINT_MAX;

    if (result)
        memset(result, 0, sizeof(*result));
    if (!session || !result ||
        ((delivery_plan || sink) &&
         (!delivery_plan || delivery_plan->session != session || !payload_sink_valid(sink)))) {
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT,
                    err, YVEX_ERR_INVALID_ARG,
                             "source_payload_verify", "session and paired delivery plan/sink are required");
    }
    buffer_bytes = delivery_plan ? delivery_plan->summary.chunk_bytes : session->budget.chunk_bytes;
    result->requested_logical_bytes = delivery_plan ? delivery_plan->summary.logical_bytes : 0u;
    rc = payload_stream_enter(session,
                              buffer_bytes,
                              YVEX_SOURCE_PAYLOAD_STATE_UNTRUSTED,
                              YVEX_SOURCE_PAYLOAD_STATE_VERIFYING,
                              failure,
                              err);
    if (rc != YVEX_OK)
        return rc;
    entered = 1;
    rc = yvex_source_payload_buffer_acquire(
        session, buffer_bytes, &buffer, &buffer_slot, failure, err);
    if (rc != YVEX_OK)
        goto fail;
    if (delivery_plan) {
        began = 1;
        if (sink->begin(sink->context, &delivery_plan->summary) != 0) {
            yvex_source_payload_fail(failure,
                                     YVEX_SOURCE_PAYLOAD_FAILURE_CONSUMER,
                                     ULLONG_MAX,
                                     ULLONG_MAX,
                                     delivery_plan->summary.logical_bytes,
                                     0u,
                                     0,
                                     err,
                                     YVEX_ERR_STATE,
                                     "source_payload_sink",
                                     "payload consumer refused trust-stream begin");
            rc = YVEX_ERR_STATE;
            goto fail;
        }
    }
    for (shard_index = 0u; shard_index < session->shard_count; ++shard_index) {
        rc = payload_verify_shard(session,
                                  delivery_plan,
                                  sink,
                                  result,
                                  failure,
                                  err,
                                  buffer,
                                  buffer_bytes,
                                  shard_index,
                                  &chunk_ordinal);
        if (rc != YVEX_OK)
            goto fail;
    }
    if (delivery_plan &&
        (chunk_ordinal != delivery_plan->summary.chunk_count ||
         result->delivered_logical_bytes != delivery_plan->summary.logical_bytes)) {
        yvex_source_payload_fail(failure,
                                 YVEX_SOURCE_PAYLOAD_FAILURE_SHORT_READ,
                                 ULLONG_MAX,
                                 ULLONG_MAX,
                                 delivery_plan->summary.logical_bytes,
                                 result->delivered_logical_bytes,
                                 0,
                                 err,
                                 YVEX_ERR_IO,
                                 "source_payload_verify",
                                 "trust stream did not cover its complete delivery plan");
        rc = YVEX_ERR_IO;
        goto fail;
    }
    rc = yvex_source_payload_identity_compute(session, failure, err);
    if (rc != YVEX_OK)
        goto fail;
    rc = yvex_source_payload_manifest_publish(session, err);
    if (rc != YVEX_OK) {
        yvex_source_payload_fail(failure,
                                 YVEX_SOURCE_PAYLOAD_FAILURE_IO,
                                 ULLONG_MAX,
                                 ULLONG_MAX,
                                 result->physical_bytes_read,
                                 result->delivered_logical_bytes,
                                 0,
                                 err,
                                 rc,
                                 "source_payload_manifest",
                                 "trusted payload manifest publication failed");
        goto fail;
    }
    pthread_mutex_lock(&session->mutex);
    session->state = YVEX_SOURCE_PAYLOAD_STATE_READY;
    session->facts.state = session->state;
    pthread_mutex_unlock(&session->mutex);
    result->trust_bytes_read = result->physical_bytes_read;
    result->complete = 1;
    if (delivery_plan) {
        if (sink->commit(sink->context, result) != 0) {
            yvex_source_payload_fail(failure,
                                     YVEX_SOURCE_PAYLOAD_FAILURE_CONSUMER,
                                     ULLONG_MAX,
                                     ULLONG_MAX,
                                     result->requested_logical_bytes,
                                     result->delivered_logical_bytes,
                                     0,
                                     err,
                                     YVEX_ERR_STATE,
                                     "source_payload_sink",
                                     "payload consumer refused trust-stream commit");
            rc = YVEX_ERR_STATE;
            goto fail_after_trust;
        }
        result->committed = 1;
    }
    yvex_source_payload_buffer_release(session, buffer_slot);
    payload_stream_leave(session, buffer_bytes, 0);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
fail:
    result->trust_bytes_read = result->physical_bytes_read;
    pthread_mutex_lock(&session->mutex);
    if (session->state == YVEX_SOURCE_PAYLOAD_STATE_VERIFYING) {
        session->state = YVEX_SOURCE_PAYLOAD_STATE_UNTRUSTED;
        session->facts.state = session->state;
    }
    if (session->state != YVEX_SOURCE_PAYLOAD_STATE_READY) {
        payload_verify_reset_trust(session);
    }
    pthread_mutex_unlock(&session->mutex);
fail_after_trust:
    if (began)
        payload_stream_abort(sink, failure, result);
    if (buffer_slot != UINT_MAX)
        yvex_source_payload_buffer_release(session, buffer_slot);
    if (entered)
        payload_stream_leave(session, buffer_bytes, 1);
    return rc;
}

static int payload_probe_begin(void *context, const yvex_source_payload_plan_summary *summary) {
    (void)context;
    return summary ? 0 : 1;
}

static int payload_probe_chunk(void *context,
                               const yvex_source_payload_chunk *chunk,
                               const unsigned char *bytes) {
    (void)context;
    return chunk && bytes ? 0 : 1;
}

static int payload_probe_commit(void *context, const yvex_source_payload_stream_result *result) {
    (void)context;
    return result && result->complete ? 0 : 1;
}

static void payload_probe_abort(void *context,
                                const yvex_source_payload_failure *failure,
                                const yvex_source_payload_stream_result *result) {
    (void)context;
    (void)failure;
    (void)result;
}

int yvex_source_payload_probe(yvex_source_payload_session *session,
                              const yvex_source_payload_plan *plan,
                              yvex_source_payload_probe_mode mode,
                              unsigned int repetitions,
                              yvex_source_payload_probe_result *out,
                              yvex_source_payload_failure *failure,
                              yvex_error *err) {
    yvex_source_payload_sink sink;
    yvex_source_payload_session_facts before;
    yvex_source_payload_session_facts after;
    struct timespec begin;
    struct timespec end;
    unsigned int repetition;
    int rc;

    if (!session || !plan || plan->session != session || !out || repetitions == 0u ||
        mode < YVEX_SOURCE_PAYLOAD_PROBE_COLD_ADVISORY || mode > YVEX_SOURCE_PAYLOAD_PROBE_STAGED) {
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT,
                    err, YVEX_ERR_INVALID_ARG,
                             "source_payload_probe",
                             "valid probe session, plan, mode, and repetition count are required");
    }
    memset(out, 0, sizeof(*out));
    out->mode = mode;
    out->repetitions = repetitions;
    if (yvex_source_payload_session_facts_get(session, &before, err) != YVEX_OK)
        return yvex_error_code(err);
    if (mode == YVEX_SOURCE_PAYLOAD_PROBE_COLD_ADVISORY) {
        unsigned long long range_index;
        out->page_cache_advice_requested = 1;
        out->page_cache_advice_accepted = 1;
        for (range_index = 0u; range_index < plan->summary.range_count; ++range_index) {
            const yvex_source_payload_range *range = &plan->ranges[range_index];
            int fd;
            rc = yvex_source_payload_handle_acquire(session, range->shard_index, &fd, failure, err);
            if (rc != YVEX_OK)
                return rc;
            if (posix_fadvise(fd,
                              (off_t)range->absolute_begin,
                              (off_t)range->byte_length,
                              POSIX_FADV_DONTNEED) != 0)
                out->page_cache_advice_accepted = 0;
            yvex_source_payload_handle_release(session, range->shard_index);
        }
    }
    memset(&sink, 0, sizeof(sink));
    sink.begin = payload_probe_begin;
    sink.chunk = payload_probe_chunk;
    sink.commit = payload_probe_commit;
    sink.abort = payload_probe_abort;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    for (repetition = 0u; repetition < repetitions; ++repetition) {
        if (mode == YVEX_SOURCE_PAYLOAD_PROBE_STAGED) {
            unsigned long long range_index;

            for (range_index = 0u; range_index < plan->summary.range_count; ++range_index) {
                unsigned long long tensor_index = plan->ranges[range_index].source_tensor_index;
                yvex_source_payload_plan *stage = NULL;
                yvex_source_payload_stream_result stream;

                rc = yvex_source_payload_plan_build(&stage,
                                                    session,
                                                    &tensor_index,
                                                    1u,
                                                    plan->summary.chunk_bytes,
                                                    plan->summary.page_bytes,
                                                    failure,
                                                    err);
                if (rc != YVEX_OK)
                    return rc;
                rc = yvex_source_payload_session_stream(
                    session, stage, &sink, &stream, failure, err);
                yvex_source_payload_plan_close(stage);
                if (rc != YVEX_OK)
                    return rc;
                out->logical_bytes += stream.delivered_logical_bytes;
                out->physical_bytes += stream.physical_bytes_read;
                out->chunks += stream.chunks_completed;
            }
        } else {
            yvex_source_payload_stream_result stream;

            rc = yvex_source_payload_session_stream(session, plan, &sink, &stream, failure, err);
            if (rc != YVEX_OK)
                return rc;
            out->logical_bytes += stream.delivered_logical_bytes;
            out->physical_bytes += stream.physical_bytes_read;
            out->chunks += stream.chunks_completed;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    if (end.tv_nsec >= begin.tv_nsec) {
        out->elapsed_nanoseconds = (unsigned long long)(end.tv_sec - begin.tv_sec) * 1000000000ull +
                                   (unsigned long long)(end.tv_nsec - begin.tv_nsec);
    } else {
        out->elapsed_nanoseconds =
            (unsigned long long)(end.tv_sec - begin.tv_sec - 1) * 1000000000ull + 1000000000ull -
            (unsigned long long)(begin.tv_nsec - end.tv_nsec);
    }
    if (yvex_source_payload_session_facts_get(session, &after, err) != YVEX_OK)
        return yvex_error_code(err);
    out->handle_hits = after.handle_cache_hits - before.handle_cache_hits;
    out->handle_misses = after.handle_cache_misses - before.handle_cache_misses;
    out->buffer_reuses = after.buffer_reuses - before.buffer_reuses;
    yvex_error_clear(err);
    return YVEX_OK;
}
