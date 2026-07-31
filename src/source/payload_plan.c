/*
 * Project immutable ranges into deterministic bounded chunk plans.
 *
 * Chunks never cross requested tensor ranges and order is physical. Planning reads zero payload
 * bytes.
 */
#include <yvex/internal/core.h>
#include <yvex/internal/source_payload.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int payload_plan_range_compare(const void *left, const void *right) {
    const yvex_source_payload_range *a = (const yvex_source_payload_range *)left;
    const yvex_source_payload_range *b = (const yvex_source_payload_range *)right;

    if (a->shard_index != b->shard_index)
        return a->shard_index < b->shard_index ? -1 : 1;
    if (a->absolute_begin != b->absolute_begin)
        return a->absolute_begin < b->absolute_begin ? -1 : 1;
    if (a->absolute_end != b->absolute_end)
        return a->absolute_end < b->absolute_end ? -1 : 1;
    return a->source_tensor_index < b->source_tensor_index   ? -1
           : a->source_tensor_index > b->source_tensor_index ? 1
                                                             : 0;
}

int yvex_source_payload_plan_build(yvex_source_payload_plan **out,
                                   yvex_source_payload_session *session,
                                   const unsigned long long *tensor_indices,
                                   unsigned long long tensor_count,
                                   size_t chunk_bytes,
                                   size_t page_bytes,
                                   yvex_source_payload_failure *failure,
                                   yvex_error *err) {
    yvex_source_payload_plan *plan;
    unsigned long long range_index;
    unsigned long long chunk_count = 0u;
    unsigned long long logical_bytes = 0u;
    unsigned long long page_count = 0u;
    unsigned long long chunk_ordinal = 0u;

    if (out)
        *out = NULL;
    if (!out || !session || tensor_count == 0u ||
        (!tensor_indices && tensor_count != session->tensor_count)) {
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT,
                    err, YVEX_ERR_INVALID_ARG,
                           "source_payload_plan", "session and a nonempty tensor selection are required");
    }
    if (chunk_bytes == 0u)
        chunk_bytes = session->budget.chunk_bytes;
    if (page_bytes == 0u)
        page_bytes = session->budget.page_bytes;
    if (chunk_bytes < YVEX_SOURCE_PAYLOAD_MIN_CHUNK_BYTES ||
        chunk_bytes > YVEX_SOURCE_PAYLOAD_MAX_CHUNK_BYTES ||
        chunk_bytes > session->budget.chunk_bytes || page_bytes == 0u ||
        (page_bytes & (page_bytes - 1u)) != 0u ||
        tensor_count > (unsigned long long)(SIZE_MAX / sizeof(yvex_source_payload_range))) {
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_CHUNK,
                    ULLONG_MAX, ULLONG_MAX,
                              chunk_bytes, 0u, 0, err, YVEX_ERR_BOUNDS, "source_payload_plan",
                              "chunk or page configuration is outside session bounds");
    }
    plan = (yvex_source_payload_plan *)session->ops.calloc_fn(1u, sizeof(*plan));
    if (!plan) {
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION, err, YVEX_ERR_NOMEM,
                           "source_payload_plan", "payload plan allocation failed");
    }
    plan->session = session;
    plan->ranges = (yvex_source_payload_range *)session->ops.calloc_fn((size_t)tensor_count,
                                                                       sizeof(plan->ranges[0]));
    if (!plan->ranges) {
        session->ops.free_fn(plan);
                return yvex_source_payload_refuse(failure, YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION, err, YVEX_ERR_NOMEM,
                           "source_payload_plan", "payload range plan allocation failed");
    }
    pthread_mutex_lock(&session->mutex);
    if (session->state != YVEX_SOURCE_PAYLOAD_STATE_UNTRUSTED &&
        session->state != YVEX_SOURCE_PAYLOAD_STATE_READY) {
        pthread_mutex_unlock(&session->mutex);
        session->ops.free_fn(plan->ranges);
        session->ops.free_fn(plan);
                return yvex_source_payload_refuse(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_STATE,
                    err, YVEX_ERR_STATE,
                           "source_payload_plan", "payload plans require an idle usable session");
    }
    for (range_index = 0u; range_index < tensor_count; ++range_index) {
        unsigned long long tensor_index =
            tensor_indices ? tensor_indices[range_index] : range_index;

        if (tensor_index >= session->tensor_count) {
            pthread_mutex_unlock(&session->mutex);
            session->ops.free_fn(plan->ranges);
            session->ops.free_fn(plan);
                        return yvex_source_payload_refuse_at(
                            failure,
                            YVEX_SOURCE_PAYLOAD_FAILURE_TENSOR_NOT_INDEXED,
                            ULLONG_MAX,
                                  tensor_index, 0u, 0u, 0, err, YVEX_ERR_BOUNDS, "source_payload_plan",
                                  "selected tensor index is absent");
        }
        plan->ranges[range_index] = session->ranges[tensor_index];
    }
    session->active_plans++;
    session->facts.active_plans = session->active_plans;
    plan->registered = 1;
    pthread_mutex_unlock(&session->mutex);
    for (range_index = 0u; range_index < tensor_count; ++range_index) {
        const yvex_source_payload_range *range = &plan->ranges[range_index];
        unsigned long long chunks;
        unsigned long long pages;

        chunks = range->byte_length / (unsigned long long)chunk_bytes;
        if (range->byte_length % (unsigned long long)chunk_bytes)
            chunks++;
        pages = (range->absolute_end - 1u) / (unsigned long long)page_bytes -
                range->absolute_begin / (unsigned long long)page_bytes + 1u;
        if (!yvex_core_u64_add(chunk_count, chunks, &chunk_count) ||
            !yvex_core_u64_add(logical_bytes, range->byte_length, &logical_bytes) ||
            !yvex_core_u64_add(page_count, pages, &page_count)) {
            yvex_source_payload_plan_close(plan);
                        return yvex_source_payload_refuse_at(
                            failure,
                            YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
                            range->shard_index,
                                  range->source_tensor_index, range->byte_length, 0u, 0, err, YVEX_ERR_BOUNDS,
                                  "source_payload_plan", "payload plan accounting overflow");
        }
    }
    qsort(plan->ranges, (size_t)tensor_count, sizeof(plan->ranges[0]), payload_plan_range_compare);
    for (range_index = 1u; range_index < tensor_count; ++range_index) {
        if (plan->ranges[range_index - 1u].source_tensor_index ==
            plan->ranges[range_index].source_tensor_index) {
            unsigned long long duplicate = plan->ranges[range_index].source_tensor_index;
            yvex_source_payload_plan_close(plan);
                        return yvex_source_payload_refuse_at(failure, YVEX_SOURCE_PAYLOAD_FAILURE_DUPLICATE_TENSOR,
                            ULLONG_MAX, duplicate,
                                  0u, 0u, 0, err, YVEX_ERR_FORMAT, "source_payload_plan",
                                  "payload plan repeats a source tensor");
        }
    }
    if (chunk_count > (unsigned long long)(SIZE_MAX / sizeof(yvex_source_payload_chunk)) ||
        chunk_count > session->budget.maximum_plan_chunks) {
        yvex_source_payload_plan_close(plan);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_RESOURCE_BUDGET,
                    ULLONG_MAX, ULLONG_MAX,
                              chunk_count, 0u, 0, err, YVEX_ERR_BOUNDS, "source_payload_plan",
                              "payload chunk map exceeds addressable memory");
    }
    plan->chunks = (yvex_source_payload_chunk *)session->ops.calloc_fn((size_t)chunk_count,
                                                                       sizeof(plan->chunks[0]));
    if (!plan->chunks) {
        yvex_source_payload_plan_close(plan);
                return yvex_source_payload_refuse_at(
                    failure,
                    YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION,
                    ULLONG_MAX, ULLONG_MAX,
                              chunk_count, 0u, 0, err, YVEX_ERR_NOMEM, "source_payload_plan",
                              "payload chunk map allocation failed");
    }
    for (range_index = 0u; range_index < tensor_count; ++range_index) {
        const yvex_source_payload_range *range = &plan->ranges[range_index];
        unsigned long long logical_offset = 0u;

        while (logical_offset < range->byte_length) {
            unsigned long long remaining = range->byte_length - logical_offset;
            size_t length =
                remaining > (unsigned long long)chunk_bytes ? chunk_bytes : (size_t)remaining;
            yvex_source_payload_chunk *chunk = &plan->chunks[chunk_ordinal];
            unsigned long long absolute = range->absolute_begin + logical_offset;

            chunk->ordinal = chunk_ordinal;
            chunk->range_ordinal = range_index;
            chunk->source_tensor_index = range->source_tensor_index;
            chunk->source_tensor_name = range->source_tensor_name;
            chunk->shard_index = range->shard_index;
            chunk->logical_offset = logical_offset;
            chunk->absolute_offset = absolute;
            chunk->byte_length = length;
            chunk->first_page = absolute / (unsigned long long)page_bytes;
            chunk->last_page =
                (absolute + (unsigned long long)length - 1u) / (unsigned long long)page_bytes;
            logical_offset += (unsigned long long)length;
            chunk_ordinal++;
        }
    }
    plan->summary.range_count = tensor_count;
    plan->summary.chunk_count = chunk_count;
    plan->summary.logical_bytes = logical_bytes;
    plan->summary.page_count = page_count;
    plan->summary.chunk_bytes = chunk_bytes;
    plan->summary.page_bytes = page_bytes;
    *out = plan;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_source_payload_plan_build_all(yvex_source_payload_plan **out,
                                       yvex_source_payload_session *session,
                                       size_t chunk_bytes,
                                       size_t page_bytes,
                                       yvex_source_payload_failure *failure,
                                       yvex_error *err) {
    if (!session)
        return YVEX_ERR_INVALID_ARG;
    return yvex_source_payload_plan_build(
        out, session, NULL, session->tensor_count, chunk_bytes, page_bytes, failure, err);
}

const yvex_source_payload_plan_summary *
yvex_source_payload_plan_summary_get(const yvex_source_payload_plan *plan) {
    return plan ? &plan->summary : NULL;
}

const yvex_source_payload_range *
yvex_source_payload_plan_range_at(const yvex_source_payload_plan *plan,
                                  unsigned long long ordinal) {
    const yvex_source_payload_range *planned;
    int trusted;

    if (!plan || !plan->session || ordinal >= plan->summary.range_count)
        return NULL;
    planned = &plan->ranges[ordinal];
    pthread_mutex_lock(&plan->session->mutex);
    trusted = plan->session->state == YVEX_SOURCE_PAYLOAD_STATE_READY;
    pthread_mutex_unlock(&plan->session->mutex);
    return trusted ? &plan->session->ranges[planned->source_tensor_index] : planned;
}

const yvex_source_payload_chunk *
yvex_source_payload_plan_chunk_at(const yvex_source_payload_plan *plan,
                                  unsigned long long ordinal) {
    return plan && ordinal < plan->summary.chunk_count ? &plan->chunks[ordinal] : NULL;
}

/*
 * Releases only plan arrays; the borrowed session and all source facts survive.
 *
 * Releases only resources owned by source payload planning; cleanup remains deterministic.
 */
void yvex_source_payload_plan_close(yvex_source_payload_plan *plan) {
    yvex_source_payload_session *session;

    if (!plan)
        return;
    session = plan->session;
    if (session) {
        if (plan->registered) {
            pthread_mutex_lock(&session->mutex);
            if (session->active_plans != 0u)
                session->active_plans--;
            session->facts.active_plans = session->active_plans;
            pthread_mutex_unlock(&session->mutex);
        }
        session->ops.free_fn(plan->chunks);
        session->ops.free_fn(plan->ranges);
        session->ops.free_fn(plan);
    }
}
