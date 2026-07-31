/*
 * Admit immutable shards and expose exact bounded payload ranges.
 *
 * Every read stays under the admitted root and bound to file identity. Payload readability
 * performs no numeric transformation.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <yvex/internal/core.h>
#include <yvex/internal/source.h>
#include <yvex/internal/source_payload.h>

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

static int payload_openat(int directory, const char *name, int flags) {
    return openat(directory, name, flags);
}

static int payload_fstat(int fd, struct stat *status) {
    return fstat(fd, status);
}

static int payload_fstatat(int directory, const char *name, struct stat *status, int flags) {
    return fstatat(directory, name, status, flags);
}

static ssize_t payload_pread(int fd, void *buffer, size_t length, off_t offset) {
    return pread(fd, buffer, length, offset);
}

static int payload_allocation_fits(unsigned long long count, size_t element_size) {
    return element_size != 0u && count <= (unsigned long long)(SIZE_MAX / element_size);
}

void yvex_source_payload_default_ops(yvex_source_payload_ops *ops) {
    if (!ops)
        return;
    memset(ops, 0, sizeof(*ops));
    ops->openat_fn = payload_openat;
    ops->fstat_fn = payload_fstat;
    ops->fstatat_fn = payload_fstatat;
    ops->pread_fn = payload_pread;
    ops->close_fn = close;
    ops->malloc_fn = malloc;
    ops->calloc_fn = calloc;
    ops->free_fn = free;
}

void yvex_source_payload_fail(yvex_source_payload_failure *failure,
                              yvex_source_payload_failure_code code,
                              unsigned long long shard_index,
                              unsigned long long tensor_index,
                              unsigned long long requested,
                              unsigned long long delivered,
                              int system_error,
                              yvex_error *err,
                              int status,
                              const char *where,
                              const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->shard_index = shard_index;
        failure->tensor_index = tensor_index;
        failure->requested_bytes = requested;
        failure->delivered_bytes = delivered;
        failure->system_error = system_error;
    }
    yvex_error_set(err, (yvex_status)status, where, message);
}

int yvex_source_payload_refuse(yvex_source_payload_failure *failure,
                               yvex_source_payload_failure_code code,
                               yvex_error *err,
                               int status,
                               const char *where,
                               const char *message) {
    yvex_source_payload_fail(
        failure, code, ULLONG_MAX, ULLONG_MAX, 0u, 0u, 0, err, status, where, message);
    return status;
}

/*
 * Publish one fully contextualized payload refusal and return its legacy status.
 *
 * Failure accounting does not mutate admitted source or payload identity.
 */
int yvex_source_payload_refuse_at(yvex_source_payload_failure *failure,
                                  yvex_source_payload_failure_code code,
                                  unsigned long long shard,
                                  unsigned long long tensor,
                                  unsigned long long requested,
                                  unsigned long long delivered,
                                  int system_error,
                                  yvex_error *err,
                                  int status,
                                  const char *where,
                                  const char *message) {
    yvex_source_payload_fail(failure,
                             code,
                             shard,
                             tensor,
                             requested,
                             delivered,
                             system_error,
                             err,
                             status,
                             where,
                             message);
    return status;
}

void yvex_source_payload_budget_default(yvex_source_payload_budget *budget) {
    if (!budget)
        return;
    memset(budget, 0, sizeof(*budget));
    budget->maximum_shards = 4096u;
    budget->maximum_tensors = 1000000u;
    budget->maximum_plan_chunks = 2000000u;
    budget->maximum_open_handles = 16u;
    budget->maximum_streams = 4u;
    budget->chunk_bytes = YVEX_SOURCE_PAYLOAD_DEFAULT_CHUNK_BYTES;
    budget->page_bytes = 4096u;
    budget->maximum_inflight_host_bytes = YVEX_SOURCE_PAYLOAD_DEFAULT_CHUNK_BYTES * 4u;
    budget->allow_local_snapshot_seal = 1;
}

const char *yvex_source_payload_trust_class_name(yvex_source_payload_trust_class trust_class) {
    switch (trust_class) {
    case YVEX_SOURCE_PAYLOAD_TRUST_LOCAL_SNAPSHOT_SEALED:
        return "local_payload_snapshot_sealed";
    case YVEX_SOURCE_PAYLOAD_TRUST_UPSTREAM_VERIFIED:
        return "upstream_payload_verified";
    default:
        return "payload_not_trusted";
    }
}

static char *payload_strdup(yvex_source_payload_session *session, const char *text) {
    size_t length;
    char *copy;

    if (!text)
        return NULL;
    length = strlen(text);
    if (length == SIZE_MAX)
        return NULL;
    copy = (char *)session->ops.malloc_fn(length + 1u);
    if (copy)
        memcpy(copy, text, length + 1u);
    return copy;
}

static void payload_identity_from_stat(yvex_source_payload_file_identity *out,
                                       const struct stat *status) {
    out->device = status->st_dev;
    out->inode = status->st_ino;
    out->size = status->st_size;
    out->mtime = status->st_mtim;
    out->ctime = status->st_ctim;
}

static int payload_identity_equal(const yvex_source_payload_file_identity *left,
                                  const struct stat *right) {
    return left->device == right->st_dev && left->inode == right->st_ino &&
           left->size == right->st_size && left->mtime.tv_sec == right->st_mtim.tv_sec &&
           left->mtime.tv_nsec == right->st_mtim.tv_nsec &&
           left->ctime.tv_sec == right->st_ctim.tv_sec &&
           left->ctime.tv_nsec == right->st_ctim.tv_nsec;
}

typedef struct {
    unsigned long long tensor_index;
    unsigned long long shard_index;
    unsigned long long begin;
    unsigned long long end;
} payload_physical_row;

static int payload_physical_row_compare(const void *left, const void *right) {
    const payload_physical_row *a = (const payload_physical_row *)left;
    const payload_physical_row *b = (const payload_physical_row *)right;

    if (a->shard_index != b->shard_index)
        return a->shard_index < b->shard_index ? -1 : 1;
    if (a->begin != b->begin)
        return a->begin < b->begin ? -1 : 1;
    if (a->end != b->end)
        return a->end < b->end ? -1 : 1;
    return a->tensor_index < b->tensor_index ? -1 : a->tensor_index > b->tensor_index ? 1 : 0;
}

static int payload_tensor_storage(const yvex_native_weight_info *tensor,
                                  unsigned long long *bytes) {
    unsigned long long elements = 1u;
    unsigned long long width = 0u;
    unsigned int dimension;

    if (!tensor || !bytes)
        return 0;
    for (dimension = 0u; dimension < tensor->rank; ++dimension) {
        if (tensor->dims[dimension] == 0u ||
            !yvex_core_u64_mul(elements, tensor->dims[dimension], &elements))
            return 0;
    }
    switch (tensor->dtype) {
    case YVEX_NATIVE_DTYPE_F64:
    case YVEX_NATIVE_DTYPE_I64:
        width = 8u;
        break;
    case YVEX_NATIVE_DTYPE_F32:
    case YVEX_NATIVE_DTYPE_I32:
        width = 4u;
        break;
    case YVEX_NATIVE_DTYPE_F16:
    case YVEX_NATIVE_DTYPE_BF16:
    case YVEX_NATIVE_DTYPE_I16:
        width = 2u;
        break;
    case YVEX_NATIVE_DTYPE_I8:
    case YVEX_NATIVE_DTYPE_U8:
    case YVEX_NATIVE_DTYPE_BOOL:
    case YVEX_NATIVE_DTYPE_F8_E4M3:
    case YVEX_NATIVE_DTYPE_F8_E5M2:
    case YVEX_NATIVE_DTYPE_F8_E8M0:
        width = 1u;
        break;
    case YVEX_NATIVE_DTYPE_FP4:
        if ((elements & 1u) != 0u)
            return 0;
        *bytes = elements / 2u;
        return 1;
    default:
        return 0;
    }
    return yvex_core_u64_mul(elements, width, bytes);
}

static int payload_budget_valid(const yvex_source_payload_budget *budget) {
    return budget && budget->maximum_shards != 0u && budget->maximum_tensors != 0u &&
           budget->maximum_plan_chunks != 0u && budget->maximum_open_handles != 0u &&
           budget->maximum_streams != 0u &&
           budget->chunk_bytes >= YVEX_SOURCE_PAYLOAD_MIN_CHUNK_BYTES &&
           budget->chunk_bytes <= YVEX_SOURCE_PAYLOAD_MAX_CHUNK_BYTES && budget->page_bytes != 0u &&
           (budget->page_bytes & (budget->page_bytes - 1u)) == 0u &&
           budget->maximum_inflight_host_bytes >= budget->chunk_bytes;
}

/* Admits one root-relative regular shard and records the exact opened-file identity. */
static int payload_admit_shard(yvex_source_payload_session *session,
                               unsigned long long index,
                               yvex_source_payload_failure *failure,
                               yvex_error *err) {
    yvex_source_payload_owned_shard *shard = &session->shards[index];
    struct stat status;
    int fd;
    int saved;

    fd = session->ops.openat_fn(session->root_fd, shard->name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        saved = errno;
                return yvex_source_payload_refuse_at(failure,
                                 saved == ELOOP ? YVEX_SOURCE_PAYLOAD_FAILURE_SYMLINK_REFUSED
                                     : YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_OPEN,
                                 index, ULLONG_MAX, 0u, 0u, saved, err, YVEX_ERR_IO, "source_payload_admit",
                                 saved == ELOOP ? "source shard symlink refused" : "source shard open failed");
    }
    session->facts.handle_opens++;
    if (session->ops.fstat_fn(fd, &status) != 0) {
        saved = errno;
        session->ops.close_fn(fd);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_STAT,
                    index, ULLONG_MAX, 0u, 0u, saved,
                                 err, YVEX_ERR_IO, "source_payload_admit", "source shard fstat failed");
    }
    if (!S_ISREG(status.st_mode)) {
        session->ops.close_fn(fd);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_NON_REGULAR_SHARD,
                    index, ULLONG_MAX, 0u,
                                 0u, 0, err, YVEX_ERR_FORMAT, "source_payload_admit",
                                 "source shard is not a regular file");
    }
    if (status.st_size < 0 || (unsigned long long)status.st_size != shard->public_fact.file_bytes) {
        session->ops.close_fn(fd);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_SIZE_MISMATCH,
                    index, ULLONG_MAX,
                                 shard->public_fact.file_bytes,
                                 status.st_size < 0 ? 0u : (unsigned long long)status.st_size, 0, err,
                                 YVEX_ERR_FORMAT, "source_payload_admit",
                                 "source shard size differs from verified snapshot");
    }
    payload_identity_from_stat(&shard->admitted_identity, &status);
    if (index < session->budget.maximum_open_handles) {
        yvex_source_payload_handle *handle = &session->handles[index];

        handle->fd = fd;
        handle->shard_index = index;
        handle->use_tick = ++session->use_tick;
        session->facts.open_handles++;
        if (session->facts.open_handles > session->facts.peak_open_handles)
            session->facts.peak_open_handles = session->facts.open_handles;
        return YVEX_OK;
    }
    if (session->ops.close_fn(fd) != 0) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_CLEANUP,
                    index, ULLONG_MAX, 0u, 0u, errno,
                                 err, YVEX_ERR_IO, "source_payload_admit",
                                 "source shard admission close failed");
    }
    return YVEX_OK;
}

static int payload_allocate_indexes(yvex_source_payload_session *session,
                                    yvex_source_payload_failure *failure,
                                    yvex_error *err) {
    unsigned long long index;

    session->shards = (yvex_source_payload_owned_shard *)session->ops.calloc_fn(
        (size_t)session->shard_count, sizeof(session->shards[0]));
    session->shard_index_entries = (yvex_shard_index_entry *)session->ops.calloc_fn(
        (size_t)session->shard_count, sizeof(session->shard_index_entries[0]));
    session->ranges = (yvex_source_payload_range *)session->ops.calloc_fn(
        (size_t)session->tensor_count, sizeof(session->ranges[0]));
    session->handles = (yvex_source_payload_handle *)session->ops.calloc_fn(
        session->budget.maximum_open_handles, sizeof(session->handles[0]));
    session->buffers = (yvex_source_payload_buffer *)session->ops.calloc_fn(
        session->budget.maximum_streams, sizeof(session->buffers[0]));
    if (!session->shards || !session->shard_index_entries || !session->ranges ||
        !session->handles || !session->buffers) {
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION, err, YVEX_ERR_NOMEM,
                              "source_payload_index", "payload index allocation failed");
    }
    for (index = 0u; index < session->budget.maximum_open_handles; ++index)
        session->handles[index].fd = -1;
    return YVEX_OK;
}

static int payload_index_shard(yvex_source_payload_session *session,
                               unsigned long long index,
                               yvex_source_payload_failure *failure,
                               yvex_error *err) {
    const yvex_source_shard_snapshot *source =
        yvex_source_tensor_snapshot_shard_at(session->snapshot, index);
    yvex_source_payload_digest_fact digest;
    yvex_source_payload_owned_shard *target = &session->shards[index];
    int rc;

    if (!source || source->canonical_id != index ||
        !yvex_source_payload_name_is_canonical(source->canonical_name)) {
                return yvex_source_payload_refuse_at(failure,
                                 source && source->canonical_name
                                     ? YVEX_SOURCE_PAYLOAD_FAILURE_PATH_ESCAPE
                                     : YVEX_SOURCE_PAYLOAD_FAILURE_DUPLICATE_SHARD,
                                 index, ULLONG_MAX, 0u, 0u, 0, err, YVEX_ERR_FORMAT, "source_payload_index",
                                 "invalid canonical source shard catalog");
    }
    target->name = payload_strdup(session, source->canonical_name);
    if (!target->name) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION,
                    index, ULLONG_MAX, 0u, 0u, 0,
                                 err, YVEX_ERR_NOMEM, "source_payload_index",
                                 "source shard name allocation failed");
    }
    target->public_fact.canonical_id = index;
    target->public_fact.canonical_name = target->name;
    session->shard_index_entries[index].canonical_id = index;
    session->shard_index_entries[index].canonical_key = target->name;
    target->public_fact.file_bytes = source->file_bytes;
    target->public_fact.data_region_offset = source->data_region_offset;
    target->public_fact.payload_bytes = source->payload_bytes;
    rc = yvex_source_provenance_payload_digest(
        &session->verification, source->canonical_name, &digest, err);
    if (rc != YVEX_OK) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_DIGEST_ALGORITHM_UNSUPPORTED,
                    index,
                                 ULLONG_MAX, 0u, 0u, 0, err, rc, "source_payload_index",
                                 "invalid provider payload digest fact");
    }
    if (digest.available) {
        memcpy(target->expected_digest, digest.expected_digest, sizeof(target->expected_digest));
        snprintf(
            target->digest_algorithm, sizeof(target->digest_algorithm), "%s", digest.algorithm);
        snprintf(
            target->digest_authority, sizeof(target->digest_authority), "%s", digest.authority);
    } else if (session->budget.allow_local_snapshot_seal) {
        snprintf(target->digest_algorithm, sizeof(target->digest_algorithm), "%s", "sha256");
        snprintf(target->digest_authority,
                 sizeof(target->digest_authority),
                 "%s",
                 "local-snapshot-seal");
    } else {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_EXPECTED_DIGEST_UNAVAILABLE,
                    index,
                                 ULLONG_MAX, 0u, 0u, 0, err, YVEX_ERR_UNSUPPORTED, "source_payload_index",
                                 "authoritative payload digest unavailable");
    }
    target->public_fact.digest_algorithm = target->digest_algorithm;
    target->public_fact.digest_authority = target->digest_authority;
    target->public_fact.expected_digest = target->expected_digest;
    target->public_fact.observed_digest = target->observed_digest;
    return payload_admit_shard(session, index, failure, err);
}

static int payload_index_range(yvex_source_payload_session *session,
                               unsigned long long index,
                               payload_physical_row *physical,
                               yvex_source_payload_failure *failure,
                               yvex_error *err) {
    const yvex_native_weight_info *tensor =
        yvex_source_tensor_snapshot_at(session->snapshot, index);
    const yvex_source_shard_snapshot *source_shard;
    yvex_source_payload_range *range = &session->ranges[index];
    unsigned long long expected_bytes = 0u;
    unsigned long long absolute_begin;
    unsigned long long absolute_end;
    unsigned int dimension;

    if (!tensor) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_TENSOR_NOT_INDEXED,
                    ULLONG_MAX, index, 0u,
                                 0u, 0, err, YVEX_ERR_FORMAT, "source_payload_range",
                                 "retained tensor row is absent");
    }
    source_shard = yvex_source_tensor_snapshot_shard_find(session->snapshot, tensor->shard_path);
    session->facts.shard_lookup_count++;
    if (!source_shard) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_NOT_INDEXED,
                    ULLONG_MAX, index, 0u,
                                 0u, 0, err, YVEX_ERR_FORMAT, "source_payload_range",
                                 "tensor refers to an absent source shard");
    }
    if (tensor->data_end <= tensor->data_start || tensor->data_end > source_shard->payload_bytes) {
                return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OUTSIDE_DATA_REGION,
                                 source_shard->canonical_id, index, tensor->data_bytes, 0u, 0, err,
                                 YVEX_ERR_BOUNDS, "source_payload_range",
                                 "tensor range lies outside source data region");
    }
    if (!payload_tensor_storage(tensor, &expected_bytes) || expected_bytes != tensor->data_bytes) {
                return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_TENSOR_LENGTH_MISMATCH,
                                 source_shard->canonical_id, index, tensor->data_bytes, expected_bytes, 0, err,
                                 YVEX_ERR_FORMAT, "source_payload_range",
                                 "tensor dtype and shape disagree with verified byte range");
    }
    if (!yvex_core_u64_add(
            source_shard->data_region_offset, tensor->data_start, &absolute_begin) ||
        !yvex_core_u64_add(source_shard->data_region_offset, tensor->data_end, &absolute_end)) {
                return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                                 source_shard->canonical_id, index, tensor->data_bytes, 0u, 0, err,
                                 YVEX_ERR_BOUNDS, "source_payload_range", "tensor absolute range overflow");
    }
    if (absolute_end > source_shard->file_bytes || absolute_end < absolute_begin) {
                return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OUTSIDE_FILE,
                                 source_shard->canonical_id, index, tensor->data_bytes, 0u, 0, err,
                                 YVEX_ERR_BOUNDS, "source_payload_range", "tensor range exceeds source file");
    }
    range->source_tensor_index = index;
    range->source_tensor_name = tensor->name;
    range->shard_index = source_shard->canonical_id;
    range->data_region_offset = source_shard->data_region_offset;
    range->relative_begin = tensor->data_start;
    range->relative_end = tensor->data_end;
    range->absolute_begin = absolute_begin;
    range->absolute_end = absolute_end;
    range->byte_length = tensor->data_bytes;
    range->dtype = tensor->dtype;
    range->rank = tensor->rank;
    for (dimension = 0u; dimension < tensor->rank; ++dimension)
        range->dims[dimension] = tensor->dims[dimension];
    range->source_snapshot_identity = session->verification.source_snapshot_identity;
    range->payload_identity = session->facts.payload_identity;
    if (!yvex_core_u64_add(
            session->logical_tensor_bytes, range->byte_length, &session->logical_tensor_bytes)) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                    range->shard_index, index,
                                 range->byte_length, 0u, 0, err, YVEX_ERR_BOUNDS, "source_payload_range",
                                 "aggregate tensor byte count overflow");
    }
    physical->tensor_index = index;
    physical->shard_index = range->shard_index;
    physical->begin = range->absolute_begin;
    physical->end = range->absolute_end;
    return YVEX_OK;
}

static int payload_build_indexes(yvex_source_payload_session *session,
                                 yvex_source_payload_failure *failure,
                                 yvex_error *err) {
    unsigned long long index;
    payload_physical_row *physical_order = NULL;
    int rc = payload_allocate_indexes(session, failure, err);

    if (rc != YVEX_OK)
        return rc;
    for (index = 0u; index < session->shard_count; ++index) {
        rc = payload_index_shard(session, index, failure, err);
        if (rc != YVEX_OK)
            return rc;
    }
    {
        yvex_shard_index_result index_result =
            yvex_shard_index_init(&session->shard_index,
                                  session->shard_index_entries,
                                  session->shard_count,
                                  session->budget.maximum_shards);

        if (index_result != YVEX_SHARD_INDEX_OK) {
                        return yvex_source_payload_refuse_at(failure,
                                     index_result == YVEX_SHARD_INDEX_BUDGET
                                         ? YVEX_SOURCE_PAYLOAD_FAILURE_RESOURCE_BUDGET
                                         : YVEX_SOURCE_PAYLOAD_FAILURE_DUPLICATE_SHARD,
                                     ULLONG_MAX, ULLONG_MAX, session->shard_count, 0u, 0, err,
                                     index_result == YVEX_SHARD_INDEX_BUDGET ? YVEX_ERR_BOUNDS : YVEX_ERR_FORMAT,
                                     "source_payload_index", "canonical source shard index admission failed");
        }
    }
    physical_order = (payload_physical_row *)session->ops.malloc_fn((size_t)session->tensor_count *
                                                                    sizeof(physical_order[0]));
    if (!physical_order) {
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION, err, YVEX_ERR_NOMEM,
                              "source_payload_index", "range-order allocation failed");
    }
    for (index = 0u; index < session->tensor_count; ++index) {
        rc = payload_index_range(session, index, &physical_order[index], failure, err);
        if (rc != YVEX_OK)
            goto cleanup;
    }
    qsort(physical_order,
          (size_t)session->tensor_count,
          sizeof(physical_order[0]),
          payload_physical_row_compare);
    for (index = 1u; index < session->tensor_count; ++index) {
        const yvex_source_payload_range *prior =
            &session->ranges[physical_order[index - 1u].tensor_index];
        const yvex_source_payload_range *next =
            &session->ranges[physical_order[index].tensor_index];
        if (prior->shard_index == next->shard_index && prior->absolute_end > next->absolute_begin) {
            rc = YVEX_ERR_FORMAT;
            yvex_source_payload_fail(failure,
                                     YVEX_SOURCE_PAYLOAD_FAILURE_DUPLICATE_TENSOR,
                                     next->shard_index,
                                     next->source_tensor_index,
                                     next->byte_length,
                                     0u,
                                     0,
                                     err,
                                     rc,
                                     "source_payload_range",
                                     "source tensor payload ranges overlap");
            goto cleanup;
        }
    }
cleanup:
    session->ops.free_fn(physical_order);
    return rc;
}

/*
 * Restore an upstream-verified v3 seal after current file admission reproduces every published
 * provider digest and the aggregate payload identity. Local-only seals remain untrusted because
 * their observed per-shard digests are not provider authority.
 *
 * Writes only the explicit source payload session destination through its transaction.
 */
static int payload_restore_published_upstream_trust(yvex_source_payload_session *session,
                                                    yvex_source_payload_failure *failure,
                                                    yvex_error *err) {
    unsigned long long index;
    int rc;

    if (!session->verification.manifest_payload_trusted)
        return YVEX_OK;
    if (strcmp(session->verification.manifest_payload_trust_class, "upstream_payload_verified") !=
        0)
        return YVEX_OK;
    if (session->verification.manifest_payload_shard_count != session->shard_count ||
        session->verification.manifest_payload_tensor_count != session->tensor_count ||
        session->verification.manifest_payload_logical_tensor_bytes !=
            session->logical_tensor_bytes ||
        !yvex_sha256_hex_valid(session->verification.manifest_payload_identity)) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_PAYLOAD_IDENTITY_MISMATCH,
                    ULLONG_MAX,
                                 ULLONG_MAX, session->tensor_count,
                                 session->verification.manifest_payload_tensor_count, 0, err, YVEX_ERR_FORMAT,
                                 "source_payload_restore_trust",
                                 "published payload aggregate facts do not match the admitted snapshot");
    }
    for (index = 0u; index < session->shard_count; ++index) {
        yvex_source_payload_owned_shard *shard = &session->shards[index];

        if (!yvex_sha256_hex_valid(shard->expected_digest) ||
            strcmp(shard->digest_algorithm, "sha256") != 0 ||
            strcmp(shard->digest_authority, "huggingface-git-lfs-etag") != 0) {
                        return yvex_source_payload_refuse_at(failure,
                            YVEX_SOURCE_PAYLOAD_FAILURE_EXPECTED_DIGEST_UNAVAILABLE, index,
                                     ULLONG_MAX, 1u, 0u, 0, err, YVEX_ERR_UNSUPPORTED,
                                     "source_payload_restore_trust",
                                     "published upstream trust requires an authoritative shard digest");
        }
        memcpy(shard->observed_digest, shard->expected_digest, sizeof(shard->observed_digest));
        shard->public_fact.trust_class = YVEX_SOURCE_PAYLOAD_TRUST_UPSTREAM_VERIFIED;
    }
    rc = yvex_source_payload_identity_compute(session, failure, err);
    if (rc != YVEX_OK)
        return rc;
    session->state = YVEX_SOURCE_PAYLOAD_STATE_READY;
    session->facts.state = session->state;
    return YVEX_OK;
}

int yvex_source_payload_session_open_with_ops(yvex_source_payload_session **out,
                                              const yvex_source_payload_open_options *options,
                                              const yvex_source_payload_ops *ops,
                                              yvex_source_payload_failure *failure,
                                              yvex_error *err) {
    yvex_source_payload_session *session;
    yvex_source_tensor_snapshot_facts snapshot_facts;
    yvex_source_payload_ops defaults;
    const yvex_source_target_identity *identity;
    int rc;

    if (out)
        *out = NULL;
    if (!out || !options || !options->verification_options || !options->verification ||
        !options->snapshot || !options->verification_options->identity) {
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT,
                    err, YVEX_ERR_INVALID_ARG,
                              "source_payload_open",
                              "verification, retained snapshot, and output are required");
    }
    if (!options->verification->verified || !options->verification->manifest_verified ||
        options->verification->header_scan_count != 1u) {
        unsigned int blocker;
        yvex_source_payload_failure_code code = YVEX_SOURCE_PAYLOAD_FAILURE_METADATA_NOT_VERIFIED;

        for (blocker = 0u; blocker < options->verification->blocker_count; ++blocker) {
            if (strcmp(options->verification->blockers[blocker],
                       "unsupported-source-manifest-version") == 0) {
                code = YVEX_SOURCE_PAYLOAD_FAILURE_MANIFEST_VERSION_UNSUPPORTED;
                break;
            }
        }
                return yvex_source_payload_refuse(failure, code, err, YVEX_ERR_STATE, "source_payload_open",
                              "exact source metadata/header verification is required");
    }
    identity = options->verification_options->identity;
    if (!identity->target_id || !identity->family_key || !identity->upstream_repo_id ||
        !identity->upstream_revision || strlen(identity->target_id) >= sizeof(session->target_id) ||
        strlen(identity->family_key) >= sizeof(session->family_key) ||
        strlen(identity->upstream_repo_id) >= sizeof(session->repository_id) ||
        strcmp(options->verification->repository_id, identity->upstream_repo_id) != 0 ||
        strcmp(options->verification->revision, identity->upstream_revision) != 0 ||
        (options->verification->manifest_target_id[0] &&
         strcmp(options->verification->manifest_target_id, identity->target_id) != 0)) {
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_SOURCE_IDENTITY_MISMATCH, err,
                              YVEX_ERR_FORMAT, "source_payload_open",
                              "verified source and target identity do not match");
    }
    yvex_error_clear(err);
    rc = yvex_source_tensor_snapshot_facts_get(options->snapshot, &snapshot_facts, err);
    if (rc != YVEX_OK)
        return rc;
    if (!yvex_source_tensor_snapshot_has_shard_catalog(options->snapshot) ||
        snapshot_facts.identity != options->verification->source_snapshot_identity) {
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_SOURCE_IDENTITY_MISMATCH, err,
                              YVEX_ERR_FORMAT, "source_payload_open",
                              "retained source snapshot identity mismatch");
    }
    if (!payload_budget_valid(&options->budget) ||
        snapshot_facts.shard_count > options->budget.maximum_shards ||
        snapshot_facts.tensor_count > options->budget.maximum_tensors ||
        !payload_allocation_fits(snapshot_facts.shard_count,
                                 sizeof(yvex_source_payload_owned_shard)) ||
        !payload_allocation_fits(snapshot_facts.shard_count, sizeof(yvex_shard_index_entry)) ||
        !payload_allocation_fits(snapshot_facts.tensor_count, sizeof(yvex_source_payload_range)) ||
        !payload_allocation_fits(snapshot_facts.tensor_count, sizeof(payload_physical_row)) ||
        !payload_allocation_fits(options->budget.maximum_open_handles,
                                 sizeof(yvex_source_payload_handle)) ||
        !payload_allocation_fits(options->budget.maximum_streams,
                                 sizeof(yvex_source_payload_buffer))) {
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_RESOURCE_BUDGET,
                    err, YVEX_ERR_BOUNDS,
                              "source_payload_open", "source payload resource budget exceeded");
    }
    yvex_source_payload_default_ops(&defaults);
    if (!ops)
        ops = &defaults;
    if (!ops->openat_fn || !ops->fstat_fn || !ops->fstatat_fn || !ops->pread_fn || !ops->close_fn ||
        !ops->malloc_fn || !ops->calloc_fn || !ops->free_fn) {
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT,
                    err, YVEX_ERR_INVALID_ARG,
                              "source_payload_open", "payload operation table is incomplete");
    }
    session = (yvex_source_payload_session *)ops->calloc_fn(1u, sizeof(*session));
    if (!session) {
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION, err, YVEX_ERR_NOMEM,
                              "source_payload_open", "payload session allocation failed");
    }
    session->root_fd = -1;
    session->ops = *ops;
    session->state = YVEX_SOURCE_PAYLOAD_STATE_CONSTRUCTING;
    session->budget = options->budget;
    session->verification = *options->verification;
    memcpy(session->target_id, identity->target_id, strlen(identity->target_id) + 1u);
    memcpy(session->family_key, identity->family_key, strlen(identity->family_key) + 1u);
    memcpy(session->repository_id,
           identity->upstream_repo_id,
           strlen(identity->upstream_repo_id) + 1u);
    session->snapshot = options->snapshot;
    session->shard_count = snapshot_facts.shard_count;
    session->tensor_count = snapshot_facts.tensor_count;
    session->facts.source_snapshot_identity = snapshot_facts.identity;
    session->facts.header_scan_count = snapshot_facts.header_scan_count;
    if (options->verification->manifest_payload_trusted &&
        options->verification->manifest_payload_source_snapshot_identity ==
            snapshot_facts.identity &&
        yvex_sha256_hex_valid(options->verification->manifest_payload_identity)) {
        yvex_core_text_copy(session->facts.admitted_payload_identity,
                            sizeof(session->facts.admitted_payload_identity),
                            options->verification->manifest_payload_identity);
    }
    if (pthread_mutex_init(&session->mutex, NULL) != 0) {
        ops->free_fn(session);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_RESOURCE_BUDGET,
                    ULLONG_MAX, ULLONG_MAX,
                                 0u, 0u, errno, err, YVEX_ERR_STATE, "source_payload_open",
                                 "payload session mutex initialization failed");
    }
    session->mutex_initialized = 1;
    yvex_source_tensor_snapshot_retain(session->snapshot);
    session->manifest_path = payload_strdup(
        session,
        options->manifest_path ? options->manifest_path : options->verification->manifest_path);
    if (!session->manifest_path) {
        rc = YVEX_ERR_NOMEM;
        yvex_source_payload_fail(failure,
                                 YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION,
                                 ULLONG_MAX,
                                 ULLONG_MAX,
                                 0u,
                                 0u,
                                 0,
                                 err,
                                 rc,
                                 "source_payload_open",
                                 "payload manifest path allocation failed");
        goto fail;
    }
    session->root_fd = ops->openat_fn(AT_FDCWD,
                                      options->verification->resolved_source_path,
                                      O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
    if (session->root_fd < 0) {
        rc = YVEX_ERR_IO;
        yvex_source_payload_fail(failure,
                                 errno == ELOOP ? YVEX_SOURCE_PAYLOAD_FAILURE_SYMLINK_REFUSED
                                                : YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_OPEN,
                                 ULLONG_MAX,
                                 ULLONG_MAX,
                                 0u,
                                 0u,
                                 errno,
                                 err,
                                 rc,
                                 "source_payload_open",
                                 "verified source root cannot be opened safely");
        goto fail;
    }
    rc = payload_build_indexes(session, failure, err);
    if (rc != YVEX_OK)
        goto fail;
    session->state = YVEX_SOURCE_PAYLOAD_STATE_UNTRUSTED;
    session->facts.state = session->state;
    session->facts.shard_count = session->shard_count;
    session->facts.tensor_count = session->tensor_count;
    session->facts.logical_tensor_bytes = session->logical_tensor_bytes;
    rc = payload_restore_published_upstream_trust(session, failure, err);
    if (rc != YVEX_OK)
        goto fail;
    *out = session;
    yvex_error_clear(err);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    return YVEX_OK;
fail:
    (void)yvex_source_payload_session_release(&session, NULL, NULL);
    return rc;
}

int yvex_source_payload_session_open(yvex_source_payload_session **out,
                                     const yvex_source_payload_open_options *options,
                                     yvex_source_payload_failure *failure,
                                     yvex_error *err) {
    return yvex_source_payload_session_open_with_ops(out, options, NULL, failure, err);
}

int yvex_source_payload_handle_acquire(yvex_source_payload_session *session,
                                       unsigned long long shard_index,
                                       int *fd,
                                       yvex_source_payload_failure *failure,
                                       yvex_error *err) {
    unsigned int slot;
    unsigned int candidate = UINT_MAX;
    unsigned long long oldest = ULLONG_MAX;
    yvex_source_payload_owned_shard *shard;
    struct stat descriptor_status;
    struct stat path_status;
    int opened;
    int saved;

    if (!session || !fd || shard_index >= session->shard_count)
        return YVEX_ERR_INVALID_ARG;
    pthread_mutex_lock(&session->mutex);
    for (slot = 0u; slot < session->budget.maximum_open_handles; ++slot) {
        if (session->handles[slot].fd >= 0 && session->handles[slot].shard_index == shard_index) {
            struct stat cached_status;
            struct stat cached_path_status;
            yvex_source_payload_owned_shard *cached_shard = &session->shards[shard_index];
            if (session->ops.fstat_fn(session->handles[slot].fd, &cached_status) != 0 ||
                session->ops.fstatat_fn(session->root_fd,
                                        cached_shard->name,
                                        &cached_path_status,
                                        AT_SYMLINK_NOFOLLOW) != 0 ||
                !S_ISREG(cached_path_status.st_mode) ||
                !payload_identity_equal(&cached_shard->admitted_identity, &cached_status) ||
                cached_status.st_dev != cached_path_status.st_dev ||
                cached_status.st_ino != cached_path_status.st_ino) {
                session->state = YVEX_SOURCE_PAYLOAD_STATE_POISONED;
                session->facts.state = session->state;
                session->facts.identity_drifts++;
                pthread_mutex_unlock(&session->mutex);
                                return yvex_source_payload_refuse_at(
                                    failure,
                                    YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_DRIFT,
                                    shard_index,
                                         ULLONG_MAX, 0u, 0u, errno, err, YVEX_ERR_STATE,
                                         "source_payload_handle",
                                         "cached source shard path or descriptor drifted");
            }
            session->handles[slot].pins++;
            session->handles[slot].use_tick = ++session->use_tick;
            session->facts.handle_cache_hits++;
            *fd = session->handles[slot].fd;
            pthread_mutex_unlock(&session->mutex);
            return YVEX_OK;
        }
        if (session->handles[slot].fd < 0 && candidate == UINT_MAX)
            candidate = slot;
    }
    if (candidate == UINT_MAX) {
        for (slot = 0u; slot < session->budget.maximum_open_handles; ++slot) {
            if (session->handles[slot].pins == 0u && session->handles[slot].use_tick < oldest) {
                candidate = slot;
                oldest = session->handles[slot].use_tick;
            }
        }
    }
    if (candidate == UINT_MAX) {
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_HANDLE_CACHE_EXHAUSTED,
                    shard_index,
                                 ULLONG_MAX, 0u, 0u, 0, err, YVEX_ERR_STATE, "source_payload_handle",
                                 "all source shard handles are pinned");
    }
    if (session->handles[candidate].fd >= 0) {
        int close_status = session->ops.close_fn(session->handles[candidate].fd);
        session->handles[candidate].fd = -1;
        session->facts.handle_evictions++;
        if (session->facts.open_handles)
            session->facts.open_handles--;
        if (close_status != 0) {
            saved = errno;
            pthread_mutex_unlock(&session->mutex);
                        return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_CLEANUP,
                            shard_index, ULLONG_MAX, 0u,
                                     0u, saved, err, YVEX_ERR_IO, "source_payload_handle",
                                     "evicted source shard handle close failed");
        }
    }
    shard = &session->shards[shard_index];
    opened =
        session->ops.openat_fn(session->root_fd, shard->name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (opened < 0) {
        saved = errno;
        session->facts.handle_cache_misses++;
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(failure,
                                 saved == ELOOP ? YVEX_SOURCE_PAYLOAD_FAILURE_SYMLINK_REFUSED
                                     : YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_OPEN,
                                 shard_index, ULLONG_MAX, 0u, 0u, saved, err, YVEX_ERR_IO,
                                 "source_payload_handle", "source shard reopen failed");
    }
    session->facts.handle_cache_misses++;
    session->facts.handle_opens++;
    session->facts.handle_reopens++;
    if (session->ops.fstat_fn(opened, &descriptor_status) != 0 ||
        session->ops.fstatat_fn(session->root_fd, shard->name, &path_status, AT_SYMLINK_NOFOLLOW) !=
            0) {
        saved = errno;
        session->ops.close_fn(opened);
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_STAT,
                    shard_index, ULLONG_MAX, 0u, 0u,
                                 saved, err, YVEX_ERR_IO, "source_payload_handle",
                                 "source shard identity stat failed");
    }
    if (!S_ISREG(path_status.st_mode) ||
        !payload_identity_equal(&shard->admitted_identity, &descriptor_status) ||
        descriptor_status.st_dev != path_status.st_dev ||
        descriptor_status.st_ino != path_status.st_ino) {
        session->ops.close_fn(opened);
        session->state = YVEX_SOURCE_PAYLOAD_STATE_POISONED;
        session->facts.state = session->state;
        session->facts.identity_drifts++;
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_DRIFT,
                    shard_index, ULLONG_MAX, 0u,
                                 0u, 0, err, YVEX_ERR_STATE, "source_payload_handle",
                                 "source shard replacement or drift detected");
    }
    session->handles[candidate].fd = opened;
    session->handles[candidate].shard_index = shard_index;
    session->handles[candidate].pins = 1u;
    session->handles[candidate].use_tick = ++session->use_tick;
    session->facts.open_handles++;
    if (session->facts.open_handles > session->facts.peak_open_handles)
        session->facts.peak_open_handles = session->facts.open_handles;
    *fd = opened;
    pthread_mutex_unlock(&session->mutex);
    return YVEX_OK;
}

/*
 * Releases one pin without closing or invalidating the reusable cached handle.
 *
 * Releases only resources owned by source payload session; cleanup remains deterministic.
 */
void yvex_source_payload_handle_release(yvex_source_payload_session *session,
                                        unsigned long long shard_index) {
    unsigned int slot;

    if (!session)
        return;
    pthread_mutex_lock(&session->mutex);
    for (slot = 0u; slot < session->budget.maximum_open_handles; ++slot) {
        if (session->handles[slot].fd >= 0 && session->handles[slot].shard_index == shard_index &&
            session->handles[slot].pins != 0u) {
            session->handles[slot].pins--;
            break;
        }
    }
    pthread_mutex_unlock(&session->mutex);
}

int yvex_source_payload_buffer_acquire(yvex_source_payload_session *session,
                                       size_t bytes,
                                       unsigned char **buffer,
                                       unsigned int *slot,
                                       yvex_source_payload_failure *failure,
                                       yvex_error *err) {
    unsigned int index;
    unsigned int candidate = UINT_MAX;

    if (!session || !buffer || !slot || bytes == 0u || bytes > session->budget.chunk_bytes) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT,
                    ULLONG_MAX, ULLONG_MAX,
                                 bytes, 0u, 0, err, YVEX_ERR_INVALID_ARG, "source_payload_buffer",
                                 "bounded payload buffer request is invalid");
    }
    *buffer = NULL;
    *slot = UINT_MAX;
    pthread_mutex_lock(&session->mutex);
    for (index = 0u; index < session->budget.maximum_streams; ++index) {
        if (!session->buffers[index].in_use && session->buffers[index].bytes &&
            session->buffers[index].capacity >= bytes) {
            session->buffers[index].in_use = 1;
            session->facts.buffer_reuses++;
            *buffer = session->buffers[index].bytes;
            *slot = index;
            pthread_mutex_unlock(&session->mutex);
            return YVEX_OK;
        }
        if (!session->buffers[index].in_use && candidate == UINT_MAX)
            candidate = index;
    }
    if (candidate == UINT_MAX) {
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_RESOURCE_BUDGET,
                    ULLONG_MAX, ULLONG_MAX,
                                 bytes, 0u, 0, err, YVEX_ERR_BOUNDS, "source_payload_buffer",
                                 "all bounded payload buffers are active");
    }
    if (session->buffers[candidate].bytes) {
        session->ops.free_fn(session->buffers[candidate].bytes);
        session->buffers[candidate].bytes = NULL;
        session->buffers[candidate].capacity = 0u;
        session->facts.buffer_releases++;
    }
    session->buffers[candidate].bytes = (unsigned char *)session->ops.malloc_fn(bytes);
    if (!session->buffers[candidate].bytes) {
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION,
                    ULLONG_MAX, ULLONG_MAX, bytes,
                                 0u, 0, err, YVEX_ERR_NOMEM, "source_payload_buffer",
                                 "bounded payload buffer allocation failed");
    }
    session->buffers[candidate].capacity = bytes;
    session->buffers[candidate].in_use = 1;
    session->facts.buffer_allocations++;
    *buffer = session->buffers[candidate].bytes;
    *slot = candidate;
    pthread_mutex_unlock(&session->mutex);
    return YVEX_OK;
}

/*
 * Returns one buffer to the session pool without retaining consumer pointers.
 *
 * Releases only resources owned by source payload session; cleanup remains deterministic.
 */
void yvex_source_payload_buffer_release(yvex_source_payload_session *session, unsigned int slot) {
    if (!session || slot >= session->budget.maximum_streams)
        return;
    pthread_mutex_lock(&session->mutex);
    session->buffers[slot].in_use = 0;
    pthread_mutex_unlock(&session->mutex);
}

int yvex_source_payload_exact_read(yvex_source_payload_session *session,
                                   unsigned long long shard_index,
                                   int fd,
                                   unsigned long long offset,
                                   unsigned char *buffer,
                                   size_t length,
                                   yvex_source_payload_stream_result *result,
                                   yvex_source_payload_failure *failure,
                                   yvex_error *err) {
    size_t completed = 0u;

    if (!session || fd < 0 || (!buffer && length != 0u) || !result ||
        offset > (unsigned long long)LLONG_MAX || (unsigned long long)(off_t)offset != offset) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                    shard_index, ULLONG_MAX,
                                 length, 0u, 0, err, YVEX_ERR_BOUNDS, "source_payload_read",
                                 "source payload offset is not representable");
    }
    while (completed < length) {
        size_t remaining = length - completed;
        size_t request = remaining > (size_t)SSIZE_MAX ? (size_t)SSIZE_MAX : remaining;
        unsigned long long current;
        ssize_t got;

        pthread_mutex_lock(&session->mutex);
        if (session->cancelled) {
            session->facts.cancellations++;
            pthread_mutex_unlock(&session->mutex);
                        return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_CANCELLED,
                            shard_index, ULLONG_MAX,
                                     length, result->delivered_logical_bytes, 0, err, YVEX_ERR_CANCELLED,
                                     "source_payload_read", "source payload stream cancelled");
        }
        pthread_mutex_unlock(&session->mutex);
        if (!yvex_core_u64_add(offset, (unsigned long long)completed, &current) ||
            current > (unsigned long long)LLONG_MAX) {
                        return yvex_source_payload_refuse_at(
                            failure,
                            YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                            shard_index,
                                     ULLONG_MAX, length, result->delivered_logical_bytes, 0, err,
                                     YVEX_ERR_BOUNDS, "source_payload_read",
                                     "source payload offset progression overflow");
        }
        got = session->ops.pread_fn(fd, buffer + completed, request, (off_t)current);
        if (got < 0 && errno == EINTR)
            continue;
        if (got < 0) {
                        return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_IO,
                            shard_index, ULLONG_MAX, length,
                                     result->delivered_logical_bytes, errno, err, YVEX_ERR_IO,
                                     "source_payload_read", "positioned source payload read failed");
        }
        if (got == 0) {
            pthread_mutex_lock(&session->mutex);
            session->facts.short_reads++;
            pthread_mutex_unlock(&session->mutex);
                        return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_SHORT_READ,
                            shard_index, ULLONG_MAX,
                                     length, result->delivered_logical_bytes, 0, err, YVEX_ERR_IO,
                                     "source_payload_read",
                                     "source payload ended before exact range completion");
        }
        completed += (size_t)got;
        if (!yvex_core_u64_add(result->physical_bytes_read,
                               (unsigned long long)got,
                               &result->physical_bytes_read)) {
                        return yvex_source_payload_refuse_at(
                            failure,
                            YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                            shard_index,
                                     ULLONG_MAX, length, result->delivered_logical_bytes, 0, err,
                                     YVEX_ERR_BOUNDS, "source_payload_read",
                                     "source payload physical byte accounting overflow");
        }
    }
    yvex_core_execution_observation_record(
        YVEX_CORE_OBSERVE_SOURCE_PAYLOAD_BYTES,
        (unsigned long long)completed);
    return YVEX_OK;
}

const yvex_source_payload_shard *yvex_source_payload_shard_at(yvex_source_payload_session *session,
                                                              unsigned long long index) {
    const yvex_shard_index_entry *entry;

    if (!session)
        return NULL;
    entry = yvex_shard_index_at(&session->shard_index, index);
    return entry ? &session->shards[entry->canonical_id].public_fact : NULL;
}

const yvex_source_payload_range *yvex_source_payload_range_at(yvex_source_payload_session *session,
                                                              unsigned long long tensor_index) {
    if (!session || tensor_index >= session->tensor_count)
        return NULL;
    pthread_mutex_lock(&session->mutex);
    session->facts.range_lookup_count++;
    pthread_mutex_unlock(&session->mutex);
    return &session->ranges[tensor_index];
}

const yvex_source_payload_range *
yvex_source_payload_range_find(yvex_source_payload_session *session, const char *tensor_name) {
    unsigned long long index;

    if (!session || !tensor_name ||
        !yvex_source_tensor_snapshot_find_index(session->snapshot, tensor_name, &index))
        return NULL;
    return yvex_source_payload_range_at(session, index);
}

int yvex_source_payload_session_cancel(yvex_source_payload_session *session,
                                       yvex_source_payload_failure *failure,
                                       yvex_error *err) {
    if (!session)
        return YVEX_ERR_INVALID_ARG;
    pthread_mutex_lock(&session->mutex);
    if (session->state == YVEX_SOURCE_PAYLOAD_STATE_CLOSED) {
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_STATE,
                    err, YVEX_ERR_STATE,
                              "source_payload_cancel", "closed payload session cannot be cancelled");
    }
    session->cancelled = 1;
    pthread_mutex_unlock(&session->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_source_payload_session_reset_cancel(yvex_source_payload_session *session,
                                             yvex_source_payload_failure *failure,
                                             yvex_error *err) {
    if (!session)
        return YVEX_ERR_INVALID_ARG;
    pthread_mutex_lock(&session->mutex);
    if (session->facts.active_streams != 0u ||
        (session->state != YVEX_SOURCE_PAYLOAD_STATE_READY &&
         session->state != YVEX_SOURCE_PAYLOAD_STATE_UNTRUSTED)) {
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_STATE,
                    err, YVEX_ERR_STATE,
                              "source_payload_cancel", "cancellation can reset only on an idle usable session");
    }
    session->cancelled = 0;
    pthread_mutex_unlock(&session->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Closes all unpinned resources only when no stream owns session state.
 *
 * Releases only resources owned by source payload session; cleanup remains deterministic.
 */
int yvex_source_payload_session_close(yvex_source_payload_session *session,
                                      yvex_source_payload_failure *failure,
                                      yvex_error *err) {
    unsigned int slot;
    unsigned int buffer_slot;
    int cleanup_failed = 0;

    if (!session)
        return YVEX_OK;
    pthread_mutex_lock(&session->mutex);
    if (session->state == YVEX_SOURCE_PAYLOAD_STATE_CLOSED) {
        pthread_mutex_unlock(&session->mutex);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (session->facts.active_streams != 0u || session->active_plans != 0u) {
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_CLOSE_BUSY, err, YVEX_ERR_STATE,
                              "source_payload_close", "payload session has active streams or plans");
    }
    for (slot = 0u; session->handles && slot < session->budget.maximum_open_handles; ++slot) {
        if (session->handles[slot].fd >= 0) {
            if (session->ops.close_fn(session->handles[slot].fd) != 0)
                cleanup_failed = 1;
            session->handles[slot].fd = -1;
        }
    }
    session->facts.open_handles = 0u;
    for (buffer_slot = 0u; session->buffers && buffer_slot < session->budget.maximum_streams;
         ++buffer_slot) {
        if (session->buffers[buffer_slot].bytes) {
            session->ops.free_fn(session->buffers[buffer_slot].bytes);
            session->buffers[buffer_slot].bytes = NULL;
            session->buffers[buffer_slot].capacity = 0u;
            session->facts.buffer_releases++;
        }
    }
    if (session->root_fd >= 0) {
        if (session->ops.close_fn(session->root_fd) != 0)
            cleanup_failed = 1;
        session->root_fd = -1;
    }
    session->state = YVEX_SOURCE_PAYLOAD_STATE_CLOSED;
    session->facts.state = session->state;
    pthread_mutex_unlock(&session->mutex);
    if (cleanup_failed) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_CLEANUP,
                    ULLONG_MAX, ULLONG_MAX, 0u, 0u,
                                 errno, err, YVEX_ERR_IO, "source_payload_close",
                                 "payload session descriptor cleanup failed");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Idempotently closes and releases all owned memory through a pointer-to-pointer.
 *
 * Releases only resources owned by source payload session; cleanup remains deterministic.
 */
int yvex_source_payload_session_release(yvex_source_payload_session **session_pointer,
                                        yvex_source_payload_failure *failure,
                                        yvex_error *err) {
    yvex_source_payload_session *session;
    unsigned long long index;
    int rc;

    if (!session_pointer || !*session_pointer) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    session = *session_pointer;
    pthread_mutex_lock(&session->mutex);
    if (session->active_plans != 0u) {
        pthread_mutex_unlock(&session->mutex);
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_CLOSE_BUSY, err, YVEX_ERR_STATE,
                              "source_payload_release", "payload plans must be released before their session");
    }
    pthread_mutex_unlock(&session->mutex);
    rc = yvex_source_payload_session_close(session, failure, err);
    if (rc == YVEX_ERR_STATE)
        return rc;
    for (index = 0u; session->shards && index < session->shard_count; ++index)
        session->ops.free_fn(session->shards[index].name);
    session->ops.free_fn(session->shards);
    yvex_shard_index_reset(&session->shard_index);
    session->ops.free_fn(session->shard_index_entries);
    session->ops.free_fn(session->ranges);
    session->ops.free_fn(session->handles);
    session->ops.free_fn(session->buffers);
    session->ops.free_fn(session->manifest_path);
    yvex_source_tensor_snapshot_release(session->snapshot);
    if (session->mutex_initialized)
        pthread_mutex_destroy(&session->mutex);
    session->ops.free_fn(session);
    *session_pointer = NULL;
    return rc;
}

int yvex_source_payload_session_facts_get(const yvex_source_payload_session *session,
                                          yvex_source_payload_session_facts *out,
                                          yvex_error *err) {
    if (!session || !out) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "source_payload_facts",
                       "payload session and facts output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    pthread_mutex_lock((pthread_mutex_t *)&session->mutex);
    *out = session->facts;
    pthread_mutex_unlock((pthread_mutex_t *)&session->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
