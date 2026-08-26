/* Form bounded FIFO physical batches only from compiler-sealed compatible runtime work. */
#include "src/runtime/private.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    COMPATIBLE_COALESCING_NS = 5000000,
    COMPATIBLE_LOGITS_COALESCING_NS = 1000000,
    COMPATIBLE_RENDEZVOUS_NS = 50000000
};

struct runtime_compatible_batcher {
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_cond_t completed;
    pthread_t worker;
    runtime_compatible_batch_ticket **queue;
    unsigned long long queue_capacity, queue_count, producer_count;
    yvex_runtime_execution_batch_summary summary;
    int mutex_ready, ready_condition, completed_condition;
    int worker_started, stopping;
};

static unsigned long long batching_elapsed_ns(const struct timespec *start,
                                              const struct timespec *end)
{
    unsigned long long seconds, nanoseconds;
    if (end->tv_sec < start->tv_sec ||
        (end->tv_sec == start->tv_sec && end->tv_nsec < start->tv_nsec))
        return 0ull;
    seconds = (unsigned long long)(end->tv_sec - start->tv_sec);
    if (end->tv_nsec >= start->tv_nsec) {
        nanoseconds = (unsigned long long)(end->tv_nsec - start->tv_nsec);
    } else {
        if (!seconds) return 0ull;
        seconds--;
        nanoseconds = 1000000000ull + (unsigned long long)end->tv_nsec -
                      (unsigned long long)start->tv_nsec;
    }
    if (seconds > (ULLONG_MAX - nanoseconds) / 1000000000ull)
        return ULLONG_MAX;
    return seconds * 1000000000ull + nanoseconds;
}

static void batching_observation_add(unsigned long long *total,
                                     unsigned long long value)
{
    *total = ULLONG_MAX - *total < value ? ULLONG_MAX : *total + value;
}

static unsigned long long batching_compatible_count_locked(
    const runtime_compatible_batcher *batcher)
{
    const runtime_compatible_batch_ticket *first;
    unsigned long long count = 0ull, index, rows = 0ull;
    if (!batcher || !batcher->queue_count) return 0ull;
    first = batcher->queue[0];
    for (index = 0ull; index < batcher->queue_count; ++index) {
        const runtime_compatible_batch_ticket *candidate =
            batcher->queue[index];
        if (!yvex_execution_compatibility_keys_match(
                &first->key, &candidate->key, NULL) ||
            candidate->row_count > first->key.admitted_width - rows)
            continue;
        rows += candidate->row_count;
        count++;
    }
    return count;
}

static unsigned long long batching_possible_compatible_count_locked(
    const runtime_compatible_batcher *batcher, unsigned long long expected)
{
    unsigned long long compatible, available, potential_producers;
    compatible = batching_compatible_count_locked(batcher);
    potential_producers = batcher->producer_count > expected
                              ? batcher->producer_count
                              : expected;
    available = potential_producers > batcher->queue_count
                    ? potential_producers - batcher->queue_count
                    : 0ull;
    return compatible > ULLONG_MAX - available
               ? ULLONG_MAX
               : compatible + available;
}

static void batching_coalesce_locked(runtime_compatible_batcher *batcher)
{
    struct timespec deadline, finish, start;
    unsigned long long elapsed, expected, limit;
    int wait_status = 0;
    if (!batcher->queue_count || batcher->stopping)
        return;
    expected = batcher->producer_count;
    if (expected < 2ull ||
        batching_compatible_count_locked(batcher) >= expected ||
        batching_possible_compatible_count_locked(batcher, expected) < expected)
        return;
    limit = batcher->queue[0]->coalescing_limit_ns
                ? batcher->queue[0]->coalescing_limit_ns
                : COMPATIBLE_COALESCING_NS;
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0 ||
        clock_gettime(CLOCK_MONOTONIC, &start) != 0)
        return;
    deadline.tv_sec += (time_t)(limit / 1000000000ull);
    deadline.tv_nsec += (long)(limit % 1000000000ull);
    if (deadline.tv_nsec >= 1000000000l) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000l;
    }
    batcher->summary.coalescing_waits++;
    while (!batcher->stopping &&
           batching_compatible_count_locked(batcher) < expected &&
           batching_possible_compatible_count_locked(batcher, expected) >= expected &&
           wait_status != ETIMEDOUT)
        wait_status = pthread_cond_timedwait(&batcher->ready, &batcher->mutex,
                                             &deadline);
    if (wait_status == ETIMEDOUT) batcher->summary.coalescing_timeouts++;
    if (clock_gettime(CLOCK_MONOTONIC, &finish) == 0) {
        elapsed = batching_elapsed_ns(&start, &finish);
        if (ULLONG_MAX - batcher->summary.coalescing_ns < elapsed)
            batcher->summary.coalescing_ns = ULLONG_MAX;
        else
            batcher->summary.coalescing_ns += elapsed;
    }
}

static int batching_refuse(yvex_error *err, yvex_status status,
                           const char *reason)
{
    yvex_error_set(err, status, "runtime.compatible-batching", reason);
    return status;
}

static int ticket_cancelled(const runtime_compatible_batch_ticket *ticket)
{
    return ticket && ticket->cancel_requested &&
           ticket->cancel_requested(ticket->cancel_context);
}

static void batching_remove(runtime_compatible_batcher *batcher,
                            unsigned long long index)
{
    for (; index + 1ull < batcher->queue_count; ++index)
        batcher->queue[index] = batcher->queue[index + 1ull];
    batcher->queue_count--;
    batcher->queue[batcher->queue_count] = NULL;
}

static void batching_complete_cancelled(runtime_compatible_batcher *batcher,
                                        unsigned long long index)
{
    runtime_compatible_batch_ticket *ticket = batcher->queue[index];
    batching_remove(batcher, index);
    ticket->status = YVEX_ERR_CANCELLED;
    ticket->done = 1;
    yvex_error_set(&ticket->failure, YVEX_ERR_CANCELLED,
                   "runtime.compatible-batching",
                   "compatible execution was cancelled before dispatch");
    batcher->summary.cancellations++;
}

static int batching_keys_match(runtime_compatible_batcher *batcher,
                               const yvex_execution_compatibility_key *left,
                               const yvex_execution_compatibility_key *right)
{
    batcher->summary.compatibility_candidates++;
    if (yvex_execution_compatibility_keys_match(left, right, NULL)) return 1;
    batcher->summary.compatibility_mismatches++;
    if (left->phase != right->phase) {
        batcher->summary.phase_mismatches++;
    } else if (left->operation != right->operation) {
        batcher->summary.operation_mismatches++;
    } else if (left->layer_ordinal != right->layer_ordinal) {
        batcher->summary.layer_mismatches++;
    } else if (left->tensor_scope != right->tensor_scope ||
               left->row_width != right->row_width ||
               left->admitted_width != right->admitted_width) {
        batcher->summary.geometry_mismatches++;
    } else if (left->backend_kind != right->backend_kind ||
               left->execution_class != right->execution_class) {
        batcher->summary.profile_mismatches++;
    } else {
        batcher->summary.identity_mismatches++;
    }
    return 0;
}

static unsigned long long batching_select_locked(
    runtime_compatible_batcher *batcher,
    runtime_compatible_batch_ticket **selected)
{
    runtime_compatible_batch_ticket *first;
    unsigned long long index, count = 0ull, rows = 0ull;
    while (batcher->queue_count && ticket_cancelled(batcher->queue[0]))
        batching_complete_cancelled(batcher, 0ull);
    if (!batcher->queue_count) return 0ull;
    first = batcher->queue[0];
    selected[count++] = first;
    rows = first->row_count;
    batching_remove(batcher, 0ull);
    for (index = 0ull; index < batcher->queue_count;) {
        runtime_compatible_batch_ticket *candidate = batcher->queue[index];
        if (ticket_cancelled(candidate)) {
            batching_complete_cancelled(batcher, index);
            continue;
        }
        if (batching_keys_match(batcher, &first->key, &candidate->key) &&
            candidate->row_count <= first->key.admitted_width - rows) {
            selected[count++] = candidate;
            rows += candidate->row_count;
            batching_remove(batcher, index);
            continue;
        }
        index++;
    }
    for (index = 0ull; index < count; ++index) {
        selected[index]->actual_width = rows;
        selected[index]->group_size = count;
    }
    batcher->summary.active = count;
    return count;
}

static void batching_finish_locked(
    runtime_compatible_batcher *batcher,
    runtime_compatible_batch_ticket **selected, unsigned long long count,
    int status, const yvex_error *failure)
{
    unsigned long long index, width = count ? selected[0]->actual_width : 0ull;
    if (count && selected[0]->kind == RUNTIME_COMPATIBLE_BATCH_RENDEZVOUS) {
        batcher->summary.rendezvous_steps++;
        batcher->summary.multi_source_rendezvous += count > 1ull;
        if (width > batcher->summary.maximum_rendezvous_width)
            batcher->summary.maximum_rendezvous_width = width;
    } else {
        batcher->summary.physical_batches++;
        batcher->summary.executed_rows += width;
        if (width > batcher->summary.maximum_width)
            batcher->summary.maximum_width = width;
        if (count > 1ull) {
            unsigned long long bucket;
            const yvex_expert_worklist_observation *worklists =
                &selected[0]->worklists;
            batcher->summary.multi_source_batches++;
            batcher->summary.multi_source_rows += width;
            if (width > batcher->summary.maximum_multi_source_width)
                batcher->summary.maximum_multi_source_width = width;
            if (count > batcher->summary.maximum_source_count)
                batcher->summary.maximum_source_count = count;
            if (worklists->worklist_count) {
                batching_observation_add(&batcher->summary.multi_source_worklists,
                                         worklists->worklist_count);
                batching_observation_add(&batcher->summary.multi_source_expert_pairs,
                                         worklists->pair_count);
                batching_observation_add(
                    &batcher->summary.multi_source_tensor_core_eligible_pairs,
                    worklists->tensor_core_eligible_pairs);
                batching_observation_add(
                    &batcher->summary.multi_source_tensor_core_executed_pairs,
                    worklists->tensor_core_executed_pairs);
                batching_observation_add(&batcher->summary.multi_source_narrow_pairs,
                                         worklists->narrow_pairs);
                if (worklists->maximum_bucket_population >
                    batcher->summary.maximum_multi_source_bucket_population)
                    batcher->summary.maximum_multi_source_bucket_population =
                        worklists->maximum_bucket_population;
                for (bucket = 0ull;
                     bucket < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP; ++bucket)
                    batching_observation_add(
                        &batcher->summary.multi_source_population_histogram[bucket],
                        worklists->population_histogram[bucket]);
            }
        }
    }
    batcher->summary.failures += status != YVEX_OK;
    batcher->summary.active = 0ull;
    for (index = 0ull; index < count; ++index) {
        selected[index]->status = status;
        if (status != YVEX_OK && failure) selected[index]->failure = *failure;
        selected[index]->done = 1;
    }
    (void)pthread_cond_broadcast(&batcher->completed);
}

static void *batching_worker(void *opaque)
{
    runtime_compatible_batcher *batcher = opaque;
    runtime_compatible_batch_ticket **selected = calloc(
        (size_t)batcher->queue_capacity, sizeof(*selected));
    if (!selected) {
        (void)pthread_mutex_lock(&batcher->mutex);
        batcher->stopping = 1;
        while (batcher->queue_count)
            batching_complete_cancelled(batcher, 0ull);
        (void)pthread_cond_broadcast(&batcher->completed);
        (void)pthread_mutex_unlock(&batcher->mutex);
        return NULL;
    }
    for (;;) {
        yvex_error failure;
        unsigned long long count;
        int status;
        (void)pthread_mutex_lock(&batcher->mutex);
        while (!batcher->queue_count && !batcher->stopping)
            (void)pthread_cond_wait(&batcher->ready, &batcher->mutex);
        if (batcher->stopping && !batcher->queue_count) {
            (void)pthread_mutex_unlock(&batcher->mutex);
            break;
        }
        batching_coalesce_locked(batcher);
        count = batching_select_locked(batcher, selected);
        batcher->summary.queued = batcher->queue_count;
        (void)pthread_cond_broadcast(&batcher->completed);
        (void)pthread_mutex_unlock(&batcher->mutex);
        if (!count) continue;
        yvex_error_clear(&failure);
        status = selected[0]->execute(selected, count, &failure);
        if (status != YVEX_OK && !yvex_error_is_set(&failure))
            yvex_error_set(&failure, (yvex_status)status,
                           "runtime.compatible-batching",
                           "compatible physical execution failed");
        (void)pthread_mutex_lock(&batcher->mutex);
        batching_finish_locked(batcher, selected, count, status, &failure);
        (void)pthread_mutex_unlock(&batcher->mutex);
    }
    free(selected);
    return NULL;
}

int yvex_runtime_private_batcher_set_producers(
    runtime_compatible_batcher *batcher, unsigned long long producers,
    yvex_error *err)
{
    if (!batcher || producers > batcher->queue_capacity ||
        pthread_mutex_lock(&batcher->mutex) != 0)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "bounded producer population is required");
    batcher->producer_count = producers;
    batcher->summary.registered_producers = producers;
    (void)pthread_cond_broadcast(&batcher->ready);
    (void)pthread_mutex_unlock(&batcher->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_batcher_open(
    runtime_compatible_batcher **out, unsigned long long queue_capacity,
    yvex_error *err)
{
    runtime_compatible_batcher *batcher;
    int rc;
    if (out) *out = NULL;
    if (!out || queue_capacity < 2ull || queue_capacity >= 64ull ||
        queue_capacity > SIZE_MAX / sizeof(*batcher->queue))
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "bounded compatible batch capacity is required");
    batcher = calloc(1u, sizeof(*batcher));
    if (!batcher)
        return batching_refuse(err, YVEX_ERR_NOMEM,
                               "compatible batch owner allocation failed");
    batcher->queue = calloc((size_t)queue_capacity, sizeof(*batcher->queue));
    batcher->queue_capacity = queue_capacity;
    batcher->summary.coalescing_limit_ns = COMPATIBLE_COALESCING_NS;
    batcher->summary.rendezvous_limit_ns = COMPATIBLE_RENDEZVOUS_NS;
    if (!batcher->queue || pthread_mutex_init(&batcher->mutex, NULL) != 0) {
        rc = batching_refuse(err, YVEX_ERR_NOMEM,
                             "compatible batch queue allocation failed");
        goto failure;
    }
    batcher->mutex_ready = 1;
    if (pthread_cond_init(&batcher->ready, NULL) != 0) {
        rc = batching_refuse(err, YVEX_ERR_STATE,
                             "compatible batch conditions are unavailable");
        goto failure;
    }
    batcher->ready_condition = 1;
    if (pthread_cond_init(&batcher->completed, NULL) != 0) {
        rc = batching_refuse(err, YVEX_ERR_STATE,
                             "compatible batch conditions are unavailable");
        goto failure;
    }
    batcher->completed_condition = 1;
    *out = batcher;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    (void)yvex_runtime_private_batcher_close(&batcher, NULL);
    return rc;
}

int yvex_runtime_private_batcher_start(
    runtime_compatible_batcher *batcher, yvex_error *err)
{
    if (!batcher || !batcher->mutex_ready ||
        pthread_mutex_lock(&batcher->mutex) != 0)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "open compatible batch owner is required");
    if (batcher->worker_started || batcher->stopping) {
        (void)pthread_mutex_unlock(&batcher->mutex);
        return batching_refuse(err, YVEX_ERR_STATE,
                               "compatible batch owner cannot start twice");
    }
    if (pthread_create(&batcher->worker, NULL, batching_worker, batcher) != 0) {
        (void)pthread_mutex_unlock(&batcher->mutex);
        return batching_refuse(err, YVEX_ERR_STATE,
                               "compatible batch worker creation failed");
    }
    batcher->worker_started = 1;
    (void)pthread_mutex_unlock(&batcher->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_batcher_submit(
    runtime_compatible_batcher *batcher,
    runtime_compatible_batch_ticket *ticket, yvex_error *err)
{
    int status;
    if (!batcher || !ticket || !ticket->execute || !ticket->row_count ||
        yvex_execution_compatibility_key_validate(&ticket->key, err) != YVEX_OK ||
        ticket->row_count > ticket->key.admitted_width)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "one admitted compatible batch ticket is required");
    if (pthread_mutex_lock(&batcher->mutex) != 0)
        return batching_refuse(err, YVEX_ERR_STATE,
                               "compatible batch lock is unavailable");
    if (batcher->stopping || batcher->queue_count == batcher->queue_capacity) {
        status = batcher->queue_count == batcher->queue_capacity
                     ? YVEX_ERR_BOUNDS : YVEX_ERR_STATE;
        (void)pthread_mutex_unlock(&batcher->mutex);
        return batching_refuse(
            err, (yvex_status)status,
            status == YVEX_ERR_BOUNDS ? "compatible batch queue is full"
                                      : "compatible batch owner is stopping");
    }
    ticket->actual_width = ticket->group_size = 0ull;
    ticket->status = YVEX_OK;
    ticket->done = 0;
    yvex_error_clear(&ticket->failure);
    batcher->queue[batcher->queue_count++] = ticket;
    if (ticket->kind == RUNTIME_COMPATIBLE_BATCH_RENDEZVOUS) {
        batcher->summary.rendezvous_submissions++;
    } else {
        batcher->summary.submissions++;
        batcher->summary.submitted_rows += ticket->row_count;
    }
    batcher->summary.queued = batcher->queue_count;
    (void)pthread_cond_signal(&batcher->ready);
    while (!ticket->done && !batcher->stopping)
        (void)pthread_cond_wait(&batcher->completed, &batcher->mutex);
    status = ticket->done ? ticket->status : YVEX_ERR_STATE;
    if (status != YVEX_OK) {
        if (ticket->done && yvex_error_is_set(&ticket->failure)) {
            if (err) *err = ticket->failure;
        } else {
            batching_refuse(err, YVEX_ERR_STATE,
                            "compatible batch stopped before completion");
        }
    } else {
        yvex_error_clear(err);
    }
    (void)pthread_mutex_unlock(&batcher->mutex);
    return status;
}

int yvex_runtime_private_batcher_snapshot(
    const runtime_compatible_batcher *batcher,
    yvex_runtime_execution_batch_summary *summary, yvex_error *err)
{
    runtime_compatible_batcher *owner = (runtime_compatible_batcher *)batcher;
    if (!owner || !summary || !owner->mutex_ready ||
        pthread_mutex_lock(&owner->mutex) != 0)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible batch summary owner is unavailable");
    *summary = owner->summary;
    summary->queued = owner->queue_count;
    (void)pthread_mutex_unlock(&owner->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_batcher_close(
    runtime_compatible_batcher **batcher, yvex_error *err)
{
    runtime_compatible_batcher *owner;
    if (!batcher || !*batcher) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owner = *batcher;
    if (owner->mutex_ready) {
        (void)pthread_mutex_lock(&owner->mutex);
        owner->stopping = 1;
        if (owner->ready_condition) (void)pthread_cond_broadcast(&owner->ready);
        if (owner->completed_condition)
            (void)pthread_cond_broadcast(&owner->completed);
        (void)pthread_mutex_unlock(&owner->mutex);
    }
    if (owner->worker_started) (void)pthread_join(owner->worker, NULL);
    if (owner->completed_condition) (void)pthread_cond_destroy(&owner->completed);
    if (owner->ready_condition) (void)pthread_cond_destroy(&owner->ready);
    if (owner->mutex_ready) (void)pthread_mutex_destroy(&owner->mutex);
    free(owner->queue);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *batcher = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

typedef struct {
    runtime_compatible_batch_ticket ticket;
    const runtime_compatible_moe_request *request;
    yvex_execution_batch_source source;
} compatible_moe_ticket;

static int compatible_moe_request_valid(
    const runtime_compatible_moe_request *request)
{
    return request && request->model && request->session && request->moe &&
           request->backend && request->transformer && request->layer &&
           request->attention && request->token_ids && request->row_count &&
           request->row_capacity && request->admitted_width &&
           request->row_count <= request->row_capacity &&
           request->admitted_width < 64ull &&
           request->attention->complete &&
           request->attention->token_count == request->row_count &&
           request->attention->envelope_output && request->expanded_rows &&
           request->combined_rows && request->routed_rows &&
           request->shared_rows && request->post_rows &&
           request->combination_rows && request->batch_token_ids &&
           request->batch_sources && request->batch_rows && request->result &&
           request->transformer_result &&
           request->provenance <= YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE &&
           request->phase < YVEX_EXECUTION_PHASE_COUNT;
}

static void compatible_moe_transformer_result(
    const runtime_compatible_moe_request *request)
{
    const yvex_moe_row_batch_result *source = request->result;
    yvex_runtime_transformer_block_result *result = request->transformer_result;
    result->hash_routers = request->row_count *
        (request->layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID);
    result->learned_routers = request->row_count *
        (request->layer->router_class == YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE);
    result->routed_experts = source->row_expert_pairs;
    result->shared_experts = source->shared_expert_operations;
    result->row_expert_pairs = source->row_expert_pairs;
    result->unique_experts = source->unique_experts;
    result->expert_worklists = source->worklists;
    result->grouped_expert_operations = source->grouped_expert_operations;
    result->expert_subviews_accessed = source->expert_subviews_accessed;
    result->expert_weight_bytes = source->encoded_bytes_read;
    result->memory = source->memory;
    result->h2d_bytes = source->h2d_bytes;
    result->d2h_bytes = source->d2h_bytes;
    result->d2d_bytes = source->d2d_bytes;
    result->kernel_launches = source->kernel_launches;
    result->tensor_core_launches = source->tensor_core_launches;
    result->graph_launches = source->graph_launches;
    result->graph_captures = source->graph_captures;
    result->graph_replays = source->graph_replays;
    result->upload_count = source->upload_count;
    result->download_count = source->download_count;
    result->cache_hits = source->cache_hits;
    result->cache_misses = source->cache_misses;
    result->stream_synchronizations = source->stream_synchronizations;
    result->device_synchronizations = source->device_synchronizations;
    result->moe_ns = source->total_ns;
    result->synchronization_ns = source->synchronization_ns;
    yvex_runtime_identity_copy(result->routing_digest, source->routing_digest);
}

static int compatible_moe_source_prepare(compatible_moe_ticket *ticket,
                                         yvex_error *err)
{
    const runtime_compatible_moe_request *request = ticket->request;
    const yvex_runtime_session_view *view =
        yvex_runtime_session_view_get(request->session);
    const yvex_attention_state_provider *provider = view
        ? (request->layer->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
               ? view->draft_attention_state_provider
               : view->attention_state_provider)
        : NULL;
    yvex_runtime_session_summary session;
    yvex_graph_attention_state_summary state;
    if (!provider || !provider->summary ||
        yvex_runtime_session_summary_copy(request->session, &session, err) !=
            YVEX_OK ||
        provider->summary(provider->context, &state, err) != YVEX_OK ||
        !session.busy ||
        !yvex_core_u64_add(session.execution_count, 1ull,
                           &ticket->source.execution_generation) ||
        !yvex_sha256_hex_valid(request->session->batch_source_identity))
        return batching_refuse(
            err, YVEX_ERR_STATE,
            "compatible MoE source generation is unavailable");
    ticket->source.state_generation = state.generation;
    yvex_runtime_identity_copy(ticket->source.identity,
                               request->session->batch_source_identity);
    return YVEX_OK;
}

static int compatible_moe_key_prepare(compatible_moe_ticket *ticket,
                                      yvex_error *err)
{
    const runtime_compatible_moe_request *request = ticket->request;
    yvex_execution_compatibility_key *key = &ticket->ticket.key;
    if (request->session->engine != request->model ||
        !request->session->summary.engine_generation)
        return batching_refuse(err, YVEX_ERR_STATE,
                               "compatible MoE engine handle is unavailable");
    memset(key, 0, sizeof(*key));
    key->schema_version = YVEX_EXECUTION_COMPATIBILITY_SCHEMA_V2;
    key->phase = request->phase;
    key->operation = YVEX_EXECUTION_COMPATIBILITY_MOE;
    key->backend_kind = yvex_backend_kind_of(request->backend);
    key->tensor_scope = request->layer->tensor_scope;
    key->execution_class = request->execution_class;
    key->engine_generation = request->session->summary.engine_generation;
    key->layer_ordinal = request->layer_ordinal;
    key->row_width = request->transformer->expanded_width;
    key->admitted_width = request->admitted_width;
    return yvex_execution_compatibility_key_validate(key, err);
}

static int compatible_tensor_same_view(const yvex_device_tensor *left,
                                       const yvex_device_tensor *right)
{
    return left && right && left->owner == right->owner &&
           left->backend_allocation == right->backend_allocation &&
           left->data == right->data && left->bytes == right->bytes;
}

static int compatible_moe_copy(yvex_backend *executor,
                               yvex_device_tensor *destination,
                               const yvex_device_tensor *source,
                               yvex_error *err)
{
    if (compatible_tensor_same_view(destination, source)) {
        destination->is_written = source->is_written;
        return YVEX_OK;
    }
    return yvex_backend_tensor_copy_shared_async(
        executor, destination, source, err);
}

static int compatible_moe_direct(
    const runtime_compatible_moe_request *request, yvex_error *err)
{
    yvex_execution_batch_source source = {0};
    yvex_moe_row_batch batch = {0};
    yvex_moe_row_batch_output output = {0};
    unsigned long long row;
    batch.schema_version = YVEX_MOE_ROW_BATCH_SCHEMA_V1;
    batch.row_count = request->row_count;
    batch.row_width = batch.row_stride = request->transformer->expanded_width;
    batch.expanded_rows = request->attention->envelope_output;
    batch.device_rows = request->device_rows;
    batch.device_outputs = request->device_outputs;
    batch.token_ids = request->token_ids;
    batch.token_ids_present = 1;
    batch.provenance = request->provenance;
    batch.phase = request->phase;
    batch.execution_class = request->execution_class;
    batch.execution_profile_identity = request->execution_profile
                                           ? request->execution_profile->identity
                                           : NULL;
    yvex_runtime_identity_copy(source.identity,
                               request->attention->execution_identity);
    for (row = 0ull; row < request->row_count; ++row) {
        yvex_execution_batch_row *owner = &request->batch_rows[row];
        if (!yvex_core_u64_add(request->attention->token_position, row,
                               &owner->sequence_position))
            return batching_refuse(err, YVEX_ERR_BOUNDS,
                                   "MoE sequence position overflowed");
        owner->source_index = 0ull;
        owner->source_row = row;
        owner->candidate_present = request->phase == YVEX_EXECUTION_PHASE_VERIFY;
        owner->candidate_ordinal = owner->candidate_present ? row : 0ull;
        owner->publication_ordinal = row;
    }
    batch.execution_sources = &source;
    batch.execution_source_count = 1ull;
    batch.execution_rows = request->batch_rows;
    output.combined_rows = request->combined_rows;
    output.combined_capacity = request->row_count * request->transformer->hidden_width;
    output.routed_rows = request->routed_rows;
    output.routed_capacity = output.combined_capacity;
    output.shared_rows = request->shared_rows;
    output.shared_capacity = output.combined_capacity;
    output.post_rows = request->post_rows;
    output.post_capacity = request->row_count * request->transformer->residual_streams;
    output.combination_rows = request->combination_rows;
    output.combination_capacity = output.post_capacity * request->transformer->residual_streams;
    return yvex_runtime_moe_rows(
        request->moe,
        &(yvex_moe_rows_request){YVEX_MOE_ROWS_EXECUTE,
                                 request->layer_ordinal, &batch, &output, 0},
        request->result, err);
}

static int compatible_moe_ticket_compare(const void *left, const void *right)
{
    const compatible_moe_ticket *a =
        *(compatible_moe_ticket *const *)left;
    const compatible_moe_ticket *b =
        *(compatible_moe_ticket *const *)right;
    return strcmp(a->source.identity, b->source.identity);
}

static int compatible_moe_local_result(
    compatible_moe_ticket *ticket, const yvex_moe_row_batch_result *physical,
    int physical_owner, yvex_error *err)
{
    const runtime_compatible_moe_request *request = ticket->request;
    yvex_moe_row_batch local = {0};
    yvex_moe_row_batch_result *result = request->result;
    if (physical_owner) *result = *physical;
    else memset(result, 0, sizeof(*result));
    result->schema_version = YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V4;
    result->completed = 1;
    result->device_completion_pending = 0;
    result->execution_class = request->execution_class;
    result->execution_profile_available = 1;
    result->row_count = request->row_count;
    result->row_expert_pairs = request->row_count * request->layer->experts_per_token;
    result->shared_expert_operations =
        request->row_count * request->layer->shared_experts;
    yvex_runtime_identity_copy(result->execution_profile_identity,
                               request->execution_profile->identity);
    local.schema_version = YVEX_MOE_ROW_BATCH_SCHEMA_V1;
    local.row_count = request->row_count;
    local.row_width = local.row_stride = request->transformer->expanded_width;
    local.expanded_rows = request->attention->envelope_output;
    local.token_ids = request->token_ids;
    local.token_ids_present = 1;
    local.execution_profile_identity = request->execution_profile->identity;
    return yvex_runtime_moe_row_routing_identity(
        request->moe, request->layer_ordinal, &local,
        result->routing_digest, err);
}

static int compatible_moe_batch_execute(
    runtime_compatible_batch_ticket *const *tickets,
    unsigned long long ticket_count, yvex_error *err)
{
    compatible_moe_ticket **ordered = (compatible_moe_ticket **)tickets;
    compatible_moe_ticket *leader;
    const runtime_compatible_moe_request *owner;
    yvex_moe_row_batch batch = {0};
    yvex_moe_row_batch_output output = {0};
    yvex_moe_row_batch_result physical = {0}, completion = {0};
    yvex_device_tensor batch_rows_view, batch_outputs_view;
    unsigned long long source_index, row_index, row_next = 0ull;
    unsigned long long expanded_values, hidden_values, post_values;
    unsigned long long d2d_bytes = 0ull;
    int rc = YVEX_OK;
    if (!ticket_count)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible MoE batch is empty");
    qsort(ordered, (size_t)ticket_count, sizeof(*ordered),
          compatible_moe_ticket_compare);
    leader = ordered[0];
    owner = leader->request;
    if (!yvex_core_u64_mul(leader->ticket.actual_width,
                           owner->transformer->expanded_width,
                           &expanded_values) ||
        !yvex_core_u64_mul(leader->ticket.actual_width,
                           owner->transformer->hidden_width,
                           &hidden_values) ||
        !yvex_core_u64_mul(leader->ticket.actual_width,
                           owner->transformer->residual_streams,
                           &post_values) ||
        leader->ticket.actual_width > owner->row_capacity ||
        !yvex_backend_tensor_f32_subview(
            owner->batch_device_rows, 0ull, expanded_values,
            &batch_rows_view) ||
        !yvex_backend_tensor_f32_subview(
            owner->batch_device_outputs, 0ull, expanded_values,
            &batch_outputs_view))
        return batching_refuse(err, YVEX_ERR_BOUNDS,
                               "compatible MoE batch exceeds sealed capacity");
    for (source_index = 0ull; source_index < ticket_count && rc == YVEX_OK;
         ++source_index) {
        compatible_moe_ticket *entry = ordered[source_index];
        const runtime_compatible_moe_request *request = entry->request;
        yvex_device_tensor destination;
        unsigned long long values;
        if (!yvex_core_u64_mul(request->row_count,
                               owner->transformer->expanded_width, &values) ||
            !yvex_backend_tensor_f32_subview(
                owner->batch_device_rows,
                row_next * owner->transformer->expanded_width, values,
                &destination))
            rc = batching_refuse(err, YVEX_ERR_BOUNDS,
                                 "compatible MoE input view is invalid");
        if (rc == YVEX_OK)
            rc = compatible_moe_copy(owner->backend, &destination,
                                     request->device_rows, err);
        if (rc == YVEX_OK &&
            !compatible_tensor_same_view(&destination, request->device_rows))
            d2d_bytes += destination.bytes;
        if (rc != YVEX_OK) break;
        owner->batch_sources[source_index] = entry->source;
        memcpy(owner->expanded_rows +
                   row_next * owner->transformer->expanded_width,
               request->attention->envelope_output,
               (size_t)values * sizeof(float));
        memcpy(owner->batch_token_ids + row_next, request->token_ids,
               (size_t)request->row_count * sizeof(*request->token_ids));
        for (row_index = 0ull; row_index < request->row_count; ++row_index) {
            yvex_execution_batch_row *row =
                &owner->batch_rows[row_next + row_index];
            if (!yvex_core_u64_add(request->attention->token_position,
                                   row_index, &row->sequence_position)) {
                rc = batching_refuse(err, YVEX_ERR_BOUNDS,
                                     "compatible MoE row position overflowed");
                break;
            }
            row->source_index = source_index;
            row->source_row = row_index;
            row->candidate_present =
                request->phase == YVEX_EXECUTION_PHASE_VERIFY;
            row->candidate_ordinal = row->candidate_present ? row_index : 0ull;
            row->publication_ordinal = row_index;
        }
        row_next += request->row_count;
    }
    if (rc != YVEX_OK) return rc;
    batch_rows_view.is_written = 1;
    batch.schema_version = YVEX_MOE_ROW_BATCH_SCHEMA_V1;
    batch.row_count = leader->ticket.actual_width;
    batch.row_width = batch.row_stride = owner->transformer->expanded_width;
    batch.expanded_rows = owner->expanded_rows;
    batch.device_rows = &batch_rows_view;
    batch.device_outputs = &batch_outputs_view;
    batch.token_ids = owner->batch_token_ids;
    batch.token_ids_present = 1;
    batch.provenance = ticket_count > 1ull
                           ? YVEX_EXECUTION_BATCH_MULTI_SESSION
                           : owner->provenance;
    batch.phase = owner->phase;
    batch.execution_class = owner->execution_class;
    batch.execution_profile_identity = owner->execution_profile->identity;
    batch.execution_sources = owner->batch_sources;
    batch.execution_source_count = ticket_count;
    batch.execution_rows = owner->batch_rows;
    batch.complete_after_operation = 1;
    output.combined_rows = owner->combined_rows;
    output.combined_capacity = hidden_values;
    output.routed_rows = owner->routed_rows;
    output.routed_capacity = hidden_values;
    output.shared_rows = owner->shared_rows;
    output.shared_capacity = hidden_values;
    output.post_rows = owner->post_rows;
    output.post_capacity = post_values;
    output.combination_rows = owner->combination_rows;
    output.combination_capacity = post_values * owner->transformer->residual_streams;
    rc = yvex_runtime_moe_rows(
        owner->moe,
        &(yvex_moe_rows_request){YVEX_MOE_ROWS_EXECUTE,
                                 owner->layer_ordinal, &batch, &output, 0},
        &physical, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_moe_rows(
            owner->moe,
            &(yvex_moe_rows_request){YVEX_MOE_ROWS_COMPLETE, 0ull, NULL,
                                     NULL, 0},
            &completion, err);
    if (rc == YVEX_OK) {
        physical.completed = 1;
        physical.device_completion_pending = 0;
        batch_outputs_view.is_written = 1;
        physical.d2d_bytes += d2d_bytes;
        physical.stream_synchronizations += completion.stream_synchronizations;
        physical.device_synchronizations += completion.device_synchronizations;
        physical.synchronization_ns += completion.synchronization_ns;
        physical.total_ns += completion.synchronization_ns;
        if (completion.worklists.worklist_count)
            physical.worklists = completion.worklists;
        leader->ticket.worklists = physical.worklists;
    }
    row_next = 0ull;
    for (source_index = 0ull; source_index < ticket_count && rc == YVEX_OK;
         ++source_index) {
        compatible_moe_ticket *entry = ordered[source_index];
        const runtime_compatible_moe_request *request = entry->request;
        yvex_device_tensor source;
        unsigned long long values;
        if (!yvex_core_u64_mul(request->row_count,
                               owner->transformer->expanded_width, &values) ||
            !yvex_backend_tensor_f32_subview(
                &batch_outputs_view,
                row_next * owner->transformer->expanded_width, values,
                &source))
            rc = batching_refuse(err, YVEX_ERR_BOUNDS,
                                 "compatible MoE output view is invalid");
        if (rc == YVEX_OK)
            rc = compatible_moe_copy(request->backend,
                                     request->device_outputs, &source, err);
        if (rc == YVEX_OK &&
            !compatible_tensor_same_view(request->device_outputs, &source))
            physical.d2d_bytes += source.bytes;
        row_next += request->row_count;
    }
    for (source_index = 0ull; source_index < ticket_count && rc == YVEX_OK;
         ++source_index)
        rc = compatible_moe_local_result(
            ordered[source_index], &physical, source_index == 0ull, err);
    return rc;
}

int yvex_runtime_private_compatible_moe_execute(
    const runtime_compatible_moe_request *request, yvex_error *err)
{
    compatible_moe_ticket ticket;
    int rc;
    if (!compatible_moe_request_valid(request))
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible MoE request is incomplete");
    if (!request->model->compatible_batcher || !request->execution_profile ||
        request->row_count > request->admitted_width ||
        yvex_backend_kind_of(request->backend) != YVEX_BACKEND_KIND_CUDA ||
        request->execution_class != YVEX_EXECUTION_CLASS_DEVICE_NATIVE) {
        rc = compatible_moe_direct(request, err);
    } else {
        memset(&ticket, 0, sizeof(ticket));
        ticket.request = request;
        ticket.ticket.row_count = request->row_count;
        ticket.ticket.execute = compatible_moe_batch_execute;
        ticket.ticket.context = &ticket;
        ticket.ticket.cancel_requested = request->cancel_requested;
        ticket.ticket.cancel_context = request->cancel_context;
        rc = compatible_moe_source_prepare(&ticket, err);
        if (rc == YVEX_OK) rc = compatible_moe_key_prepare(&ticket, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_private_batcher_submit(
                request->model->compatible_batcher, &ticket.ticket, err);
    }
    if (rc == YVEX_OK) compatible_moe_transformer_result(request);
    return rc;
}

typedef struct {
    runtime_compatible_batch_ticket ticket;
    yvex_model_engine *model;
    yvex_runtime_execution_session *session;
    yvex_runtime_logits_context *logits;
    yvex_backend *backend;
    const yvex_runtime_execution_profile *execution_profile;
    const yvex_runtime_logits_source *source;
    yvex_runtime_logits_row_result *result;
    float *host_logits;
    unsigned long long host_logits_capacity;
    unsigned long long admitted_width;
} compatible_logits_ticket;

static int compatible_logits_direct(
    const compatible_logits_ticket *ticket, yvex_error *err)
{
    return yvex_runtime_logits_project(
        ticket->logits, ticket->source, yvex_backend_kind_of(ticket->backend),
        ticket->host_logits, ticket->host_logits_capacity, ticket->result, err);
}

static yvex_execution_phase compatible_logits_phase(yvex_logits_source_phase phase)
{
    if (phase == YVEX_LOGITS_SOURCE_PREFILL) return YVEX_EXECUTION_PHASE_PREFILL;
    if (phase == YVEX_LOGITS_SOURCE_DRAFT) return YVEX_EXECUTION_PHASE_DRAFT;
    return YVEX_EXECUTION_PHASE_DECODE;
}

static int compatible_logits_key_prepare(compatible_logits_ticket *ticket,
                                         yvex_error *err)
{
    const yvex_runtime_logits_plan_summary *plan =
        yvex_runtime_logits_plan_summary_get(ticket->logits);
    yvex_execution_compatibility_key *key = &ticket->ticket.key;
    if (!plan || ticket->session->engine != ticket->model ||
        ticket->admitted_width >= 64ull ||
        !ticket->session->summary.engine_generation)
        return batching_refuse(err, YVEX_ERR_STATE,
                               "compatible output-head engine handle is unavailable");
    key->schema_version = YVEX_EXECUTION_COMPATIBILITY_SCHEMA_V2;
    key->phase = compatible_logits_phase(ticket->source->source_phase);
    key->operation = YVEX_EXECUTION_COMPATIBILITY_OUTPUT_HEAD;
    key->backend_kind = yvex_backend_kind_of(ticket->backend);
    key->tensor_scope = YVEX_TENSOR_SCOPE_GLOBAL;
    key->execution_class = ticket->execution_profile->execution_class;
    key->engine_generation = ticket->session->summary.engine_generation;
    key->row_width = plan->hidden_width;
    key->admitted_width = ticket->admitted_width;
    return yvex_execution_compatibility_key_validate(key, err);
}

static int compatible_logits_ticket_compare(const void *left,
                                            const void *right)
{
    const compatible_logits_ticket *a = *(compatible_logits_ticket *const *)left;
    const compatible_logits_ticket *b = *(compatible_logits_ticket *const *)right;
    return strcmp(a->session->batch_source_identity,
                  b->session->batch_source_identity);
}

static int compatible_logits_batch_execute(
    runtime_compatible_batch_ticket *const *tickets,
    unsigned long long ticket_count, yvex_error *err)
{
    compatible_logits_ticket **ordered = (compatible_logits_ticket **)tickets;
    yvex_runtime_logits_context *contexts[63];
    const yvex_runtime_logits_source *sources[63];
    yvex_runtime_logits_row_result *rows[63];
    unsigned long long index;
    if (!ticket_count || ticket_count >= 64ull)
        return batching_refuse(err, YVEX_ERR_BOUNDS,
                               "compatible output-head batch exceeds capacity");
    if (ticket_count == 1ull) return compatible_logits_direct(ordered[0], err);
    qsort(ordered, (size_t)ticket_count, sizeof(*ordered),
          compatible_logits_ticket_compare);
    for (index = 0ull; index < ticket_count; ++index) {
        contexts[index] = ordered[index]->logits;
        sources[index] = ordered[index]->source;
        rows[index] = ordered[index]->result;
    }
    return yvex_runtime_logits_project_compatible(
        contexts, sources, rows, ticket_count, err);
}

int yvex_runtime_private_generation_logits_project(
    yvex_runtime_generation_context *context,
    const yvex_runtime_logits_source *source,
    yvex_runtime_logits_row_result *result, yvex_error *err)
{
    const yvex_runtime_session_view *session = context
        ? yvex_runtime_session_view_get(context->session) : NULL;
    compatible_logits_ticket ticket = {0};
    int producer_active = 0, rc;
    if (!context || !session)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "generation output-head owner is unavailable");
    ticket.model = context->model;
    ticket.session = context->session;
    ticket.logits = context->logits;
    ticket.backend = session->backend;
    ticket.execution_profile = &context->execution_profile;
    ticket.source = source;
    ticket.result = result;
    ticket.host_logits = context->logits_row;
    ticket.host_logits_capacity = context->logits_row ? context->logits_count : 0ull;
    ticket.admitted_width = context->options.continuous_batching
                                ? context->options.concurrent_sequences : 1ull;
    if (!source || !result || !ticket.admitted_width)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible output-head request is incomplete");
    if (!ticket.model->compatible_batcher || ticket.admitted_width < 2ull ||
        yvex_backend_kind_of(ticket.backend) != YVEX_BACKEND_KIND_CUDA ||
        ticket.execution_profile->execution_class != YVEX_EXECUTION_CLASS_DEVICE_NATIVE ||
        !source->device_values_available)
        return compatible_logits_direct(&ticket, err);
    ticket.ticket.row_count = 1ull;
    ticket.ticket.coalescing_limit_ns = COMPATIBLE_LOGITS_COALESCING_NS;
    ticket.ticket.execute = compatible_logits_batch_execute;
    ticket.ticket.context = &ticket;
    ticket.ticket.cancel_requested = context->options.cancel_requested;
    ticket.ticket.cancel_context = context->options.cancel_context;
    rc = compatible_logits_key_prepare(&ticket, err);
    if (rc == YVEX_OK) {
        rc = yvex_runtime_private_model_batcher_producer_enter(ticket.model, err);
        producer_active = rc == YVEX_OK;
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_private_batcher_submit(
            ticket.model->compatible_batcher, &ticket.ticket, err);
    return yvex_runtime_private_batcher_producer_finish(
        ticket.model, &producer_active, rc, err);
}

static int compatible_step_execute(
    runtime_compatible_batch_ticket *const *tickets,
    unsigned long long ticket_count, yvex_error *err)
{
    unsigned long long index;
    for (index = 0ull; index < ticket_count; ++index)
        if (!tickets[index] ||
            tickets[index]->kind != RUNTIME_COMPATIBLE_BATCH_RENDEZVOUS ||
            tickets[index]->execute != compatible_step_execute)
            return batching_refuse(
                err, YVEX_ERR_STATE,
                "non-rendezvous work entered a compatible step barrier");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int compatible_step_rendezvous(
    const runtime_compatible_step_request *request, yvex_error *err)
{
    runtime_compatible_batch_ticket ticket = {0};
    yvex_execution_compatibility_key *key = &ticket.key;
    if (!request || !request->model || !request->session || !request->backend ||
        !request->transformer || !request->execution_profile ||
        request->maximum_width < 2ull || request->maximum_width >= 64ull ||
        request->phase >= YVEX_EXECUTION_PHASE_COUNT)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible execution step is incomplete");
    if (!request->model->compatible_batcher) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (request->session->engine != request->model ||
        !request->session->summary.engine_generation)
        return batching_refuse(err, YVEX_ERR_STATE,
                               "compatible execution step engine handle is unavailable");
    key->schema_version = YVEX_EXECUTION_COMPATIBILITY_SCHEMA_V2;
    key->phase = request->phase;
    key->operation = YVEX_EXECUTION_COMPATIBILITY_TRANSFORMER_STEP;
    key->backend_kind = yvex_backend_kind_of(request->backend);
    key->tensor_scope = request->tensor_scope;
    key->execution_class = request->execution_class;
    key->engine_generation = request->session->summary.engine_generation;
    key->row_width = request->transformer->expanded_width;
    key->admitted_width = request->maximum_width;
    if (yvex_execution_compatibility_key_validate(key, err) != YVEX_OK)
        return yvex_error_code(err);
    ticket.row_count = 1ull;
    ticket.coalescing_limit_ns = COMPATIBLE_RENDEZVOUS_NS;
    ticket.execute = compatible_step_execute;
    ticket.cancel_requested = request->cancel_requested;
    ticket.cancel_context = request->cancel_context;
    ticket.kind = RUNTIME_COMPATIBLE_BATCH_RENDEZVOUS;
    return yvex_runtime_private_batcher_submit(
        request->model->compatible_batcher, &ticket, err);
}

int yvex_runtime_private_model_batcher_acquire(
    yvex_model_engine *model, unsigned long long maximum_width,
    yvex_error *err)
{
    runtime_compatible_batcher *created = NULL;
    int rc = YVEX_OK;
    if (!model || maximum_width < 2ull || maximum_width >= 64ull ||
        !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "runtime model and compatible width are required");
    if (model->close_requested ||
        (model->compatible_batcher &&
         model->compatible_batch_width != maximum_width)) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return batching_refuse(
            err, YVEX_ERR_STATE,
            model->close_requested
                ? "draining runtime model cannot admit compatible batching"
                : "runtime model compatible width is already sealed");
    }
    if (!model->compatible_batcher) {
        rc = yvex_runtime_private_batcher_open(
            &created, maximum_width, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_private_batcher_start(created, err);
        if (rc == YVEX_OK) {
            model->compatible_batcher = created;
            model->compatible_batch_width = maximum_width;
            created = NULL;
        }
    }
    if (rc == YVEX_OK &&
        model->compatible_batcher_references == ULLONG_MAX)
        rc = batching_refuse(err, YVEX_ERR_BOUNDS,
                             "runtime model batching references overflowed");
    if (rc == YVEX_OK) model->compatible_batcher_references++;
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    if (created) (void)yvex_runtime_private_batcher_close(&created, NULL);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_runtime_private_model_batcher_release(
    yvex_model_engine *model, yvex_error *err)
{
    runtime_compatible_batcher *owner = NULL;
    int rc;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "runtime model batching ownership is required");
    if (!model->compatible_batcher || !model->compatible_batcher_references) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return batching_refuse(err, YVEX_ERR_STATE,
                               "runtime model batching ownership is inconsistent");
    }
    if (model->compatible_batcher_references == 1ull &&
        model->compatible_batcher_producers) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return batching_refuse(
            err, YVEX_ERR_STATE,
            "active compatible producers prevent final ownership release");
    }
    model->compatible_batcher_references--;
    if (!model->compatible_batcher_references) {
        owner = model->compatible_batcher;
        model->compatible_batcher = NULL;
        model->compatible_batch_width = 0ull;
    }
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    rc = yvex_runtime_private_batcher_close(&owner, err);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_runtime_private_model_batcher_finish(
    yvex_model_engine *model, int *acquired, yvex_error *err)
{
    int rc;
    if (!acquired || !*acquired) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_runtime_private_model_batcher_release(model, err);
    if (rc == YVEX_OK) *acquired = 0;
    return rc;
}

int yvex_runtime_private_model_batcher_producer_enter(
    yvex_model_engine *model, yvex_error *err)
{
    int rc;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "runtime model batching producer is required");
    if (!model->compatible_batcher || !model->compatible_batcher_references ||
        model->compatible_batcher_producers >= model->compatible_batch_width) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return batching_refuse(
            err, YVEX_ERR_BOUNDS,
            "active compatible producer population exceeds admitted width");
    }
    model->compatible_batcher_producers++;
    rc = yvex_runtime_private_batcher_set_producers(
        model->compatible_batcher, model->compatible_batcher_producers, err);
    if (rc != YVEX_OK) model->compatible_batcher_producers--;
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    return rc;
}

int yvex_runtime_private_model_batcher_producer_leave(
    yvex_model_engine *model, yvex_error *err)
{
    int rc;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "runtime model batching producer is required");
    if (!model->compatible_batcher || !model->compatible_batcher_producers) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return batching_refuse(err, YVEX_ERR_STATE,
                               "runtime model batching producer is inconsistent");
    }
    model->compatible_batcher_producers--;
    rc = yvex_runtime_private_batcher_set_producers(
        model->compatible_batcher, model->compatible_batcher_producers, err);
    if (rc != YVEX_OK) model->compatible_batcher_producers++;
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    return rc;
}

int yvex_runtime_private_batcher_producer_finish(
    yvex_model_engine *model, int *active, int status, yvex_error *err)
{
    int leave_status;
    if (!active || !*active) return status;
    leave_status = yvex_runtime_private_model_batcher_producer_leave(
        model, status == YVEX_OK ? err : NULL);
    if (leave_status == YVEX_OK) *active = 0;
    return status == YVEX_OK ? leave_status : status;
}

int yvex_model_engine_compatible_batch_width_copy(
    const yvex_model_engine *model, unsigned long long *width,
    yvex_error *err)
{
    yvex_model_engine *owner = (yvex_model_engine *)model;
    const yvex_physical_execution_summary *summary;
    unsigned long long common = 0ull, consumers = 0ull, backend, index, candidate;
    int initialized = 0;
    if (width) *width = 0ull;
    if (!owner || !width || !owner->lifecycle_mutex_ready ||
        pthread_mutex_lock(&owner->lifecycle_mutex) != 0)
        return batching_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "runtime model compatible width is unavailable");
    summary = yvex_physical_execution_ir_summary(owner->physical_execution);
    for (backend = YVEX_BACKEND_KIND_CPU;
         summary && backend <= YVEX_BACKEND_KIND_CUDA; ++backend) {
        const yvex_engine_specialization *specialization =
            owner->specializations[backend];
        if (!specialization) continue;
        for (index = 0ull; index < summary->decision_count; ++index) {
            const yvex_physical_execution_decision *package =
                yvex_physical_execution_ir_decision_at(
                    owner->physical_execution, index);
            const yvex_engine_implementation_record *decision =
                runtime_specialization_decision(specialization, index);
            unsigned long long admitted;
            if (!package || !decision ||
                (package->consumer != YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP &&
                 package->consumer != YVEX_EXECUTION_CONSUMER_ROUTED_DOWN))
                continue;
            admitted = decision->supported_width_mask &
                       decision->worklist_width_mask;
            common = initialized ? common & admitted : admitted;
            initialized = 1;
            consumers |= 1ull << (unsigned int)package->consumer;
        }
    }
    *width = 1ull;
    if (initialized &&
        (consumers & (1ull << YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP)) &&
        (consumers & (1ull << YVEX_EXECUTION_CONSUMER_ROUTED_DOWN)))
        for (candidate = 2ull; candidate < 63ull; ++candidate)
            if (common & (1ull << candidate)) *width = candidate;
    (void)pthread_mutex_unlock(&owner->lifecycle_mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_compatible_step_enter(
    const runtime_compatible_step_request *request, int *active,
    yvex_error *err)
{
    yvex_error primary, cleanup;
    int rc;
    if (!request || !request->model || !active || *active)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible transformer step is required");
    rc = yvex_runtime_private_model_batcher_producer_enter(request->model, err);
    if (rc != YVEX_OK) return rc;
    rc = compatible_step_rendezvous(request, err);
    if (rc == YVEX_OK) {
        *active = 1;
        return YVEX_OK;
    }
    primary = err ? *err : (yvex_error){0};
    yvex_error_clear(&cleanup);
    (void)yvex_runtime_private_model_batcher_producer_leave(request->model,
                                                            &cleanup);
    if (err) *err = primary;
    return rc;
}

int yvex_model_engine_execution_batch_summary_copy(
    const yvex_model_engine *model, yvex_runtime_execution_batch_summary *out,
    yvex_error *err)
{
    yvex_model_engine *owner = (yvex_model_engine *)model;
    int rc = YVEX_OK;
    if (!owner || !out || !owner->lifecycle_mutex_ready ||
        pthread_mutex_lock(&owner->lifecycle_mutex) != 0)
        return batching_refuse(err, YVEX_ERR_INVALID_ARG,
                               "runtime model batching summary is unavailable");
    memset(out, 0, sizeof(*out));
    if (owner->compatible_batcher)
        rc = yvex_runtime_private_batcher_snapshot(owner->compatible_batcher,
                                                   out, err);
    if (rc == YVEX_OK) {
        out->enabled = owner->compatible_batcher != NULL;
        out->admitted_maximum_width = owner->compatible_batch_width;
    }
    (void)pthread_mutex_unlock(&owner->lifecycle_mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
