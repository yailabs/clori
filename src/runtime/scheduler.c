/* Schedule bounded engine work and form physical batches only from sealed compatibility keys. */
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
    COMPATIBLE_PRODUCER_ARRIVAL_NS = 1000000,
    COMPATIBLE_RENDEZVOUS_NS = 50000000
};

typedef struct {
    unsigned long long generation, submission_ordinal;
    unsigned long long engine_generation, sequence_handle;
    yvex_engine_progress_kind kind;
    int (*cancel_requested)(void *context);
    void *cancel_context;
    int in_use, queued, active, status;
} runtime_engine_progress_work;

struct runtime_engine_scheduler {
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_cond_t completed;
    pthread_t worker;
    runtime_engine_work **queue;
    unsigned long long queue_capacity, queue_count, producer_count;
    unsigned long long producer_arrival_floor;
    runtime_engine_progress_work *progress;
    unsigned long long progress_capacity, progress_in_use_count;
    unsigned long long progress_ready_count, progress_active_count;
    unsigned long long next_progress_generation, next_progress_ordinal;
    yvex_engine_scheduler_summary summary;
    int mutex_ready, ready_condition, completed_condition;
    int worker_started, stopping;
};

static int scheduler_refuse(yvex_error *err, yvex_status status,
                            const char *reason);

static unsigned long long scheduler_elapsed_ns(const struct timespec *start,
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

static void scheduler_observation_add(unsigned long long *total,
                                      unsigned long long value)
{
    *total = ULLONG_MAX - *total < value ? ULLONG_MAX : *total + value;
}

static int progress_cancelled(const runtime_engine_progress_work *work)
{
    return work && work->cancel_requested &&
           work->cancel_requested(work->cancel_context);
}

static void progress_summary_locked(runtime_engine_scheduler *scheduler)
{
    scheduler->summary.ready_sequence_work = scheduler->progress_ready_count;
    scheduler->summary.active_sequences = scheduler->progress_active_count;
    if (scheduler->progress_ready_count >
        scheduler->summary.maximum_ready_sequence_work)
        scheduler->summary.maximum_ready_sequence_work =
            scheduler->progress_ready_count;
    if (scheduler->progress_active_count >
        scheduler->summary.maximum_active_sequences)
        scheduler->summary.maximum_active_sequences =
            scheduler->progress_active_count;
}

static runtime_engine_progress_work *progress_slot_locked(
    runtime_engine_scheduler *scheduler,
    const runtime_engine_progress_lease *lease)
{
    runtime_engine_progress_work *work;
    if (!scheduler || !lease || lease->scheduler != scheduler ||
        lease->slot >= scheduler->progress_capacity)
        return NULL;
    work = &scheduler->progress[lease->slot];
    return work->in_use && work->generation == lease->generation
               ? work : NULL;
}

static int progress_active_conflict_locked(
    const runtime_engine_scheduler *scheduler,
    const runtime_engine_progress_work *candidate)
{
    unsigned long long index;
    for (index = 0ull; index < scheduler->progress_capacity; ++index) {
        const runtime_engine_progress_work *active =
            &scheduler->progress[index];
        if (active != candidate && active->in_use && active->active &&
            active->engine_generation == candidate->engine_generation &&
            active->sequence_handle == candidate->sequence_handle)
            return 1;
    }
    return 0;
}

static void progress_active_remove_locked(
    runtime_engine_scheduler *scheduler,
    runtime_engine_progress_work *work)
{
    if (!work || !work->active) return;
    work->active = 0;
    scheduler->progress_active_count--;
}

static void progress_complete_kind_locked(
    runtime_engine_scheduler *scheduler,
    const runtime_engine_progress_work *work)
{
    scheduler_observation_add(&scheduler->summary.progress_completions, 1ull);
    scheduler_observation_add(
        &scheduler->summary.progress_completions_by_kind[work->kind], 1ull);
}

static void progress_cancel_ready_locked(
    runtime_engine_scheduler *scheduler,
    runtime_engine_progress_work *work)
{
    if (!work || !work->queued) return;
    work->queued = 0;
    work->status = YVEX_ERR_CANCELLED;
    scheduler->progress_ready_count--;
    scheduler_observation_add(&scheduler->summary.progress_cancellations, 1ull);
}

static runtime_engine_progress_work *progress_next_ready_locked(
    runtime_engine_scheduler *scheduler)
{
    runtime_engine_progress_work *selected = NULL;
    unsigned long long index;
    for (index = 0ull; index < scheduler->progress_capacity; ++index) {
        runtime_engine_progress_work *candidate =
            &scheduler->progress[index];
        if (!candidate->in_use || !candidate->queued) continue;
        if (progress_cancelled(candidate)) {
            progress_cancel_ready_locked(scheduler, candidate);
            continue;
        }
        if (progress_active_conflict_locked(scheduler, candidate)) {
            scheduler_observation_add(
                &scheduler->summary.sequence_conflicts, 1ull);
            continue;
        }
        if (!selected ||
            candidate->submission_ordinal < selected->submission_ordinal)
            selected = candidate;
    }
    return selected;
}

static void progress_grant_locked(runtime_engine_scheduler *scheduler)
{
    while (!scheduler->stopping &&
           scheduler->progress_active_count < scheduler->progress_capacity) {
        runtime_engine_progress_work *candidate =
            progress_next_ready_locked(scheduler);
        if (!candidate) break;
        candidate->queued = 0;
        candidate->active = 1;
        scheduler->progress_ready_count--;
        scheduler->progress_active_count++;
    }
    progress_summary_locked(scheduler);
    (void)pthread_cond_broadcast(&scheduler->completed);
}

static void progress_enqueue_locked(
    runtime_engine_scheduler *scheduler,
    runtime_engine_progress_work *work)
{
    scheduler->next_progress_ordinal++;
    if (!scheduler->next_progress_ordinal)
        scheduler->next_progress_ordinal++;
    work->submission_ordinal = scheduler->next_progress_ordinal;
    work->queued = 1;
    work->active = 0;
    work->status = YVEX_OK;
    scheduler->progress_ready_count++;
    scheduler_observation_add(&scheduler->summary.progress_submissions, 1ull);
    scheduler_observation_add(
        &scheduler->summary.progress_submissions_by_kind[work->kind], 1ull);
    progress_grant_locked(scheduler);
}

static int progress_wait_locked(
    runtime_engine_scheduler *scheduler,
    runtime_engine_progress_work *work, yvex_error *err)
{
    while (work->queued && !scheduler->stopping) {
        struct timespec deadline;
        if (clock_gettime(CLOCK_REALTIME, &deadline) == 0) {
            deadline.tv_nsec += 10000000l;
            if (deadline.tv_nsec >= 1000000000l) {
                deadline.tv_sec++;
                deadline.tv_nsec -= 1000000000l;
            }
            (void)pthread_cond_timedwait(
                &scheduler->completed, &scheduler->mutex, &deadline);
        } else {
            (void)pthread_cond_wait(&scheduler->completed, &scheduler->mutex);
        }
        progress_grant_locked(scheduler);
    }
    if (work->active) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (work->status == YVEX_ERR_CANCELLED)
        return scheduler_refuse(
            err, YVEX_ERR_CANCELLED,
            "ready sequence work was cancelled before admission");
    return scheduler_refuse(
        err, YVEX_ERR_STATE,
        "engine scheduler stopped before sequence work admission");
}

static void progress_release_locked(
    runtime_engine_scheduler *scheduler, runtime_engine_progress_work *work)
{
    if (!work || !work->in_use) return;
    if (work->queued) {
        work->queued = 0;
        scheduler->progress_ready_count--;
    }
    progress_active_remove_locked(scheduler, work);
    memset(work, 0, sizeof(*work));
    scheduler->progress_in_use_count--;
    progress_summary_locked(scheduler);
    (void)pthread_cond_broadcast(&scheduler->completed);
}

static unsigned long long scheduler_compatible_count_locked(
    const runtime_engine_scheduler *scheduler)
{
    const runtime_engine_work *first;
    unsigned long long count = 0ull, index, rows = 0ull;
    if (!scheduler || !scheduler->queue_count) return 0ull;
    first = scheduler->queue[0];
    for (index = 0ull; index < scheduler->queue_count; ++index) {
        const runtime_engine_work *candidate =
            scheduler->queue[index];
        if (!yvex_execution_compatibility_keys_match(
                &first->key, &candidate->key, NULL) ||
            candidate->row_count > first->key.admitted_width - rows)
            continue;
        rows += candidate->row_count;
        count++;
    }
    return count;
}

static unsigned long long scheduler_possible_compatible_count_locked(
    const runtime_engine_scheduler *scheduler, unsigned long long expected)
{
    unsigned long long compatible, available, potential_producers;
    compatible = scheduler_compatible_count_locked(scheduler);
    potential_producers = scheduler->producer_count > expected
                              ? scheduler->producer_count
                              : expected;
    available = potential_producers > scheduler->queue_count
                    ? potential_producers - scheduler->queue_count
                    : 0ull;
    return compatible > ULLONG_MAX - available
               ? ULLONG_MAX
               : compatible + available;
}

static void scheduler_coalesce_locked(runtime_engine_scheduler *scheduler)
{
    struct timespec deadline, finish, start;
    unsigned long long elapsed, expected, limit;
    int wait_status = 0;
    if (!scheduler->queue_count || scheduler->stopping)
        return;
    expected = scheduler->producer_count > scheduler->producer_arrival_floor
                   ? scheduler->producer_count
                   : scheduler->producer_arrival_floor;
    if (expected < 2ull ||
        scheduler_compatible_count_locked(scheduler) >= expected ||
        scheduler_possible_compatible_count_locked(scheduler, expected) < expected) {
        scheduler->producer_arrival_floor = 0ull;
        return;
    }
    limit = scheduler->queue[0]->coalescing_limit_ns
                ? scheduler->queue[0]->coalescing_limit_ns
                : COMPATIBLE_COALESCING_NS;
    if (scheduler->producer_arrival_floor > scheduler->producer_count &&
        limit > COMPATIBLE_PRODUCER_ARRIVAL_NS)
        limit = COMPATIBLE_PRODUCER_ARRIVAL_NS;
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0 ||
        clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        scheduler->producer_arrival_floor = 0ull;
        return;
    }
    deadline.tv_sec += (time_t)(limit / 1000000000ull);
    deadline.tv_nsec += (long)(limit % 1000000000ull);
    if (deadline.tv_nsec >= 1000000000l) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000l;
    }
    scheduler->summary.coalescing_waits++;
    while (!scheduler->stopping &&
           scheduler_compatible_count_locked(scheduler) < expected &&
           scheduler_possible_compatible_count_locked(scheduler, expected) >= expected &&
           wait_status != ETIMEDOUT)
        wait_status = pthread_cond_timedwait(&scheduler->ready, &scheduler->mutex,
                                             &deadline);
    if (wait_status == ETIMEDOUT) scheduler->summary.coalescing_timeouts++;
    if (clock_gettime(CLOCK_MONOTONIC, &finish) == 0) {
        elapsed = scheduler_elapsed_ns(&start, &finish);
        if (ULLONG_MAX - scheduler->summary.coalescing_ns < elapsed)
            scheduler->summary.coalescing_ns = ULLONG_MAX;
        else
            scheduler->summary.coalescing_ns += elapsed;
    }
    scheduler->producer_arrival_floor = 0ull;
}

static int scheduler_refuse(yvex_error *err, yvex_status status,
                           const char *reason)
{
    yvex_error_set(err, status, "runtime.engine-scheduler", reason);
    return status;
}

static int ticket_cancelled(const runtime_engine_work *ticket)
{
    return ticket && ticket->cancel_requested &&
           ticket->cancel_requested(ticket->cancel_context);
}

static void scheduler_remove(runtime_engine_scheduler *scheduler,
                            unsigned long long index)
{
    for (; index + 1ull < scheduler->queue_count; ++index)
        scheduler->queue[index] = scheduler->queue[index + 1ull];
    scheduler->queue_count--;
    scheduler->queue[scheduler->queue_count] = NULL;
}

static void scheduler_complete_cancelled(runtime_engine_scheduler *scheduler,
                                        unsigned long long index)
{
    runtime_engine_work *ticket = scheduler->queue[index];
    scheduler_remove(scheduler, index);
    ticket->status = YVEX_ERR_CANCELLED;
    ticket->done = 1;
    yvex_error_set(&ticket->failure, YVEX_ERR_CANCELLED,
                   "runtime.engine-scheduler",
                   "compatible execution was cancelled before dispatch");
    scheduler->summary.cancellations++;
}

static int scheduler_keys_match(runtime_engine_scheduler *scheduler,
                               const yvex_execution_compatibility_key *left,
                               const yvex_execution_compatibility_key *right)
{
    scheduler->summary.compatibility_candidates++;
    if (yvex_execution_compatibility_keys_match(left, right, NULL)) return 1;
    scheduler->summary.compatibility_mismatches++;
    if (left->phase != right->phase) {
        scheduler->summary.phase_mismatches++;
    } else if (left->operation != right->operation) {
        scheduler->summary.operation_mismatches++;
    } else if (left->layer_ordinal != right->layer_ordinal) {
        scheduler->summary.layer_mismatches++;
    } else if (left->tensor_scope != right->tensor_scope ||
               left->row_width != right->row_width ||
               left->admitted_width != right->admitted_width) {
        scheduler->summary.geometry_mismatches++;
    } else if (left->backend_kind != right->backend_kind ||
               left->execution_class != right->execution_class) {
        scheduler->summary.profile_mismatches++;
    } else {
        scheduler->summary.identity_mismatches++;
    }
    return 0;
}

static unsigned long long scheduler_select_locked(
    runtime_engine_scheduler *scheduler,
    runtime_engine_work **selected)
{
    runtime_engine_work *first;
    unsigned long long index, count = 0ull, rows = 0ull;
    while (scheduler->queue_count && ticket_cancelled(scheduler->queue[0]))
        scheduler_complete_cancelled(scheduler, 0ull);
    if (!scheduler->queue_count) return 0ull;
    first = scheduler->queue[0];
    selected[count++] = first;
    rows = first->row_count;
    scheduler_remove(scheduler, 0ull);
    for (index = 0ull; index < scheduler->queue_count;) {
        runtime_engine_work *candidate = scheduler->queue[index];
        if (ticket_cancelled(candidate)) {
            scheduler_complete_cancelled(scheduler, index);
            continue;
        }
        if (scheduler_keys_match(scheduler, &first->key, &candidate->key) &&
            candidate->row_count <= first->key.admitted_width - rows) {
            selected[count++] = candidate;
            rows += candidate->row_count;
            scheduler_remove(scheduler, index);
            continue;
        }
        index++;
    }
    for (index = 0ull; index < count; ++index) {
        selected[index]->actual_width = rows;
        selected[index]->group_size = count;
    }
    scheduler->summary.active = count;
    return count;
}

static void scheduler_finish_locked(
    runtime_engine_scheduler *scheduler,
    runtime_engine_work **selected, unsigned long long count,
    int status, const yvex_error *failure)
{
    unsigned long long index, phase = count ? selected[0]->key.phase : 0ull;
    unsigned long long width = count ? selected[0]->actual_width : 0ull;
    if (count && selected[0]->kind == RUNTIME_ENGINE_WORK_RENDEZVOUS) {
        scheduler->summary.rendezvous_steps++;
        scheduler_observation_add(
            &scheduler->summary.rendezvous_steps_by_phase[phase], 1ull);
        scheduler->summary.multi_source_rendezvous += count > 1ull;
        if (width > scheduler->summary.maximum_rendezvous_width)
            scheduler->summary.maximum_rendezvous_width = width;
    } else {
        scheduler->summary.physical_batches++;
        scheduler->summary.executed_rows += width;
        scheduler_observation_add(
            &scheduler->summary.physical_batches_by_phase[phase], 1ull);
        scheduler_observation_add(
            &scheduler->summary.executed_rows_by_phase[phase], width);
        if (width > scheduler->summary.maximum_width)
            scheduler->summary.maximum_width = width;
        if (count > 1ull) {
            unsigned long long bucket;
            const yvex_expert_worklist_observation *worklists =
                &selected[0]->worklists;
            scheduler->summary.multi_source_batches++;
            scheduler->summary.multi_source_rows += width;
            if (width > scheduler->summary.maximum_multi_source_width)
                scheduler->summary.maximum_multi_source_width = width;
            if (count > scheduler->summary.maximum_source_count)
                scheduler->summary.maximum_source_count = count;
            if (worklists->worklist_count) {
                scheduler_observation_add(&scheduler->summary.multi_source_worklists,
                                          worklists->worklist_count);
                scheduler_observation_add(&scheduler->summary.multi_source_expert_pairs,
                                          worklists->pair_count);
                scheduler_observation_add(
                    &scheduler->summary.multi_source_matrix_tile_eligible_pairs,
                    worklists->matrix_tile_eligible_pairs);
                scheduler_observation_add(
                    &scheduler->summary.multi_source_matrix_tile_executed_pairs,
                    worklists->matrix_tile_executed_pairs);
                scheduler_observation_add(&scheduler->summary.multi_source_narrow_pairs,
                                          worklists->narrow_pairs);
                if (worklists->maximum_bucket_population >
                    scheduler->summary.maximum_multi_source_bucket_population)
                    scheduler->summary.maximum_multi_source_bucket_population =
                        worklists->maximum_bucket_population;
                for (bucket = 0ull;
                     bucket < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP; ++bucket)
                    scheduler_observation_add(
                        &scheduler->summary.multi_source_population_histogram[bucket],
                        worklists->population_histogram[bucket]);
            }
        }
    }
    scheduler->summary.failures += status != YVEX_OK;
    scheduler->summary.active = 0ull;
    for (index = 0ull; index < count; ++index) {
        selected[index]->status = status;
        if (status != YVEX_OK && failure) selected[index]->failure = *failure;
        selected[index]->done = 1;
    }
    (void)pthread_cond_broadcast(&scheduler->completed);
}

static void *scheduler_worker(void *opaque)
{
    runtime_engine_scheduler *scheduler = opaque;
    runtime_engine_work **selected = calloc(
        (size_t)scheduler->queue_capacity, sizeof(*selected));
    if (!selected) {
        (void)pthread_mutex_lock(&scheduler->mutex);
        scheduler->stopping = 1;
        while (scheduler->queue_count)
            scheduler_complete_cancelled(scheduler, 0ull);
        (void)pthread_cond_broadcast(&scheduler->completed);
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return NULL;
    }
    for (;;) {
        yvex_error failure;
        unsigned long long count;
        int status;
        (void)pthread_mutex_lock(&scheduler->mutex);
        while (!scheduler->queue_count && !scheduler->stopping)
            (void)pthread_cond_wait(&scheduler->ready, &scheduler->mutex);
        if (scheduler->stopping && !scheduler->queue_count) {
            (void)pthread_mutex_unlock(&scheduler->mutex);
            break;
        }
        scheduler_coalesce_locked(scheduler);
        count = scheduler_select_locked(scheduler, selected);
        scheduler->summary.queued = scheduler->queue_count;
        (void)pthread_cond_broadcast(&scheduler->completed);
        (void)pthread_mutex_unlock(&scheduler->mutex);
        if (!count) continue;
        yvex_error_clear(&failure);
        status = selected[0]->execute(selected, count, &failure);
        if (status != YVEX_OK && !yvex_error_is_set(&failure))
            yvex_error_set(&failure, (yvex_status)status,
                           "runtime.engine-scheduler",
                           "compatible physical execution failed");
        (void)pthread_mutex_lock(&scheduler->mutex);
        scheduler_finish_locked(scheduler, selected, count, status, &failure);
        (void)pthread_mutex_unlock(&scheduler->mutex);
    }
    free(selected);
    return NULL;
}

int yvex_runtime_private_engine_scheduler_set_producers(
    runtime_engine_scheduler *scheduler, unsigned long long producers,
    yvex_error *err)
{
    if (!scheduler || producers > scheduler->queue_capacity ||
        pthread_mutex_lock(&scheduler->mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "bounded producer population is required");
    if (!scheduler->producer_count && producers == 1ull &&
        scheduler->queue_capacity >= 2ull)
        scheduler->producer_arrival_floor = 2ull;
    else if (!producers)
        scheduler->producer_arrival_floor = 0ull;
    scheduler->producer_count = producers;
    scheduler->summary.registered_producers = producers;
    (void)pthread_cond_broadcast(&scheduler->ready);
    (void)pthread_mutex_unlock(&scheduler->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_engine_scheduler_open(
    runtime_engine_scheduler **out, unsigned long long queue_capacity,
    yvex_error *err)
{
    runtime_engine_scheduler *scheduler;
    int rc;
    if (out) *out = NULL;
    if (!out || !queue_capacity || queue_capacity >= 64ull ||
        queue_capacity > SIZE_MAX / sizeof(*scheduler->queue))
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                                "bounded engine scheduler capacity is required");
    scheduler = calloc(1u, sizeof(*scheduler));
    if (!scheduler)
        return scheduler_refuse(err, YVEX_ERR_NOMEM,
                                "engine scheduler allocation failed");
    scheduler->queue = calloc((size_t)queue_capacity, sizeof(*scheduler->queue));
    scheduler->progress = calloc((size_t)queue_capacity,
                                 sizeof(*scheduler->progress));
    scheduler->queue_capacity = queue_capacity;
    scheduler->progress_capacity = queue_capacity;
    scheduler->summary.sequence_capacity = queue_capacity;
    scheduler->summary.coalescing_limit_ns = COMPATIBLE_COALESCING_NS;
    scheduler->summary.rendezvous_limit_ns = COMPATIBLE_RENDEZVOUS_NS;
    if (!scheduler->queue || !scheduler->progress ||
        pthread_mutex_init(&scheduler->mutex, NULL) != 0) {
        rc = scheduler_refuse(err, YVEX_ERR_NOMEM,
                              "engine scheduler queue allocation failed");
        goto failure;
    }
    scheduler->mutex_ready = 1;
    if (pthread_cond_init(&scheduler->ready, NULL) != 0) {
        rc = scheduler_refuse(err, YVEX_ERR_STATE,
                              "engine scheduler conditions are unavailable");
        goto failure;
    }
    scheduler->ready_condition = 1;
    if (pthread_cond_init(&scheduler->completed, NULL) != 0) {
        rc = scheduler_refuse(err, YVEX_ERR_STATE,
                              "engine scheduler conditions are unavailable");
        goto failure;
    }
    scheduler->completed_condition = 1;
    *out = scheduler;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    (void)yvex_runtime_private_engine_scheduler_close(&scheduler, NULL);
    return rc;
}

int yvex_runtime_private_engine_scheduler_start(
    runtime_engine_scheduler *scheduler, yvex_error *err)
{
    if (!scheduler || !scheduler->mutex_ready ||
        pthread_mutex_lock(&scheduler->mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                                "open engine scheduler is required");
    if (scheduler->worker_started || scheduler->stopping) {
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(err, YVEX_ERR_STATE,
                                "engine scheduler cannot start twice");
    }
    if (pthread_create(&scheduler->worker, NULL, scheduler_worker, scheduler) != 0) {
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(err, YVEX_ERR_STATE,
                                "engine scheduler worker creation failed");
    }
    scheduler->worker_started = 1;
    (void)pthread_mutex_unlock(&scheduler->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_engine_scheduler_submit(
    runtime_engine_scheduler *scheduler,
    runtime_engine_work *ticket, yvex_error *err)
{
    int status;
    if (!scheduler || !ticket || !ticket->execute || !ticket->row_count ||
        yvex_execution_compatibility_key_validate(&ticket->key, err) != YVEX_OK ||
        ticket->row_count > ticket->key.admitted_width)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                                "one admitted executable work item is required");
    if (pthread_mutex_lock(&scheduler->mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_STATE,
                                "engine scheduler lock is unavailable");
    if (scheduler->stopping || scheduler->queue_count == scheduler->queue_capacity) {
        status = scheduler->queue_count == scheduler->queue_capacity
                     ? YVEX_ERR_BOUNDS : YVEX_ERR_STATE;
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(
            err, (yvex_status)status,
            status == YVEX_ERR_BOUNDS ? "engine scheduler queue is full"
                                      : "engine scheduler is stopping");
    }
    ticket->actual_width = ticket->group_size = 0ull;
    ticket->status = YVEX_OK;
    ticket->done = 0;
    yvex_error_clear(&ticket->failure);
    scheduler->queue[scheduler->queue_count++] = ticket;
    scheduler_observation_add(
        &scheduler->summary.submissions_by_phase[ticket->key.phase], 1ull);
    if (ticket->kind == RUNTIME_ENGINE_WORK_RENDEZVOUS) {
        scheduler->summary.rendezvous_submissions++;
    } else {
        scheduler->summary.submissions++;
        scheduler->summary.submitted_rows += ticket->row_count;
    }
    scheduler->summary.queued = scheduler->queue_count;
    (void)pthread_cond_signal(&scheduler->ready);
    while (!ticket->done && !scheduler->stopping)
        (void)pthread_cond_wait(&scheduler->completed, &scheduler->mutex);
    status = ticket->done ? ticket->status : YVEX_ERR_STATE;
    if (status != YVEX_OK) {
        if (ticket->done && yvex_error_is_set(&ticket->failure)) {
            if (err) *err = ticket->failure;
        } else {
            scheduler_refuse(err, YVEX_ERR_STATE,
                             "scheduled work stopped before completion");
        }
    } else {
        yvex_error_clear(err);
    }
    (void)pthread_mutex_unlock(&scheduler->mutex);
    return status;
}

int yvex_runtime_private_engine_scheduler_snapshot(
    const runtime_engine_scheduler *scheduler,
    yvex_engine_scheduler_summary *summary, yvex_error *err)
{
    runtime_engine_scheduler *owner = (runtime_engine_scheduler *)scheduler;
    if (!owner || !summary || !owner->mutex_ready ||
        pthread_mutex_lock(&owner->mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                                "engine scheduler summary is unavailable");
    *summary = owner->summary;
    summary->queued = owner->queue_count;
    (void)pthread_mutex_unlock(&owner->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_engine_scheduler_close(
    runtime_engine_scheduler **scheduler, yvex_error *err)
{
    runtime_engine_scheduler *owner;
    unsigned long long index;
    if (!scheduler || !*scheduler) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owner = *scheduler;
    if (owner->mutex_ready) {
        (void)pthread_mutex_lock(&owner->mutex);
        owner->stopping = 1;
        for (index = 0ull; index < owner->progress_capacity; ++index)
            if (owner->progress[index].in_use &&
                owner->progress[index].queued)
                progress_cancel_ready_locked(
                    owner, &owner->progress[index]);
        progress_summary_locked(owner);
        if (owner->ready_condition) (void)pthread_cond_broadcast(&owner->ready);
        if (owner->completed_condition)
            (void)pthread_cond_broadcast(&owner->completed);
        while (owner->progress_in_use_count && owner->completed_condition)
            (void)pthread_cond_wait(&owner->completed, &owner->mutex);
        (void)pthread_mutex_unlock(&owner->mutex);
    }
    if (owner->worker_started) (void)pthread_join(owner->worker, NULL);
    if (owner->completed_condition) (void)pthread_cond_destroy(&owner->completed);
    if (owner->ready_condition) (void)pthread_cond_destroy(&owner->ready);
    if (owner->mutex_ready) (void)pthread_mutex_destroy(&owner->mutex);
    free(owner->progress);
    free(owner->queue);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *scheduler = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_private_engine_progress_enter(
    yvex_model_engine *model, yvex_runtime_execution_session *session,
    yvex_engine_progress_kind kind,
    int (*cancel_requested)(void *context), void *cancel_context,
    runtime_engine_progress_lease *lease, yvex_error *err)
{
    runtime_engine_scheduler *scheduler;
    runtime_engine_progress_work *work = NULL;
    unsigned long long index;
    int rc;
    if (lease) memset(lease, 0, sizeof(*lease));
    if (!model || !session || !lease ||
        kind >= YVEX_ENGINE_PROGRESS_KIND_COUNT ||
        !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return scheduler_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "engine-owned ready sequence work is required");
    scheduler = model->engine_scheduler;
    if (!scheduler || !model->engine_scheduler_references ||
        model->close_requested || session->engine != model ||
        !session->summary.engine_generation ||
        session->summary.engine_generation != model->summary.engine_generation ||
        !session->batch_source_ordinal ||
        pthread_mutex_lock(&scheduler->mutex) != 0) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return scheduler_refuse(
            err, YVEX_ERR_STATE,
            "ready sequence work has no live engine scheduler generation");
    }
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    if (scheduler->stopping) {
        (void)pthread_mutex_unlock(&scheduler->mutex);
        memset(lease, 0, sizeof(*lease));
        return scheduler_refuse(
            err, YVEX_ERR_STATE,
            "draining engine scheduler cannot admit sequence work");
    }
    for (index = 0ull; index < scheduler->progress_capacity; ++index)
        if (!scheduler->progress[index].in_use) {
            work = &scheduler->progress[index];
            break;
        }
    if (!work) {
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(
            err, YVEX_ERR_BOUNDS,
            "engine ready-work capacity is exhausted");
    }
    scheduler->next_progress_generation++;
    if (!scheduler->next_progress_generation)
        scheduler->next_progress_generation++;
    memset(work, 0, sizeof(*work));
    work->generation = scheduler->next_progress_generation;
    work->engine_generation = session->summary.engine_generation;
    work->sequence_handle = session->batch_source_ordinal;
    work->kind = kind;
    work->cancel_requested = cancel_requested;
    work->cancel_context = cancel_context;
    work->in_use = 1;
    scheduler->progress_in_use_count++;
    lease->scheduler = scheduler;
    lease->slot = index;
    lease->generation = work->generation;
    progress_enqueue_locked(scheduler, work);
    rc = progress_wait_locked(scheduler, work, err);
    if (rc != YVEX_OK) {
        progress_release_locked(scheduler, work);
        memset(lease, 0, sizeof(*lease));
    }
    (void)pthread_mutex_unlock(&scheduler->mutex);
    return rc;
}

int yvex_runtime_private_engine_progress_transition(
    runtime_engine_progress_lease *lease, yvex_engine_progress_kind kind,
    yvex_error *err)
{
    runtime_engine_scheduler *scheduler;
    runtime_engine_progress_work *work;
    int rc;
    if (!lease || !lease->scheduler ||
        kind >= YVEX_ENGINE_PROGRESS_KIND_COUNT)
        return scheduler_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "active engine sequence work and next kind are required");
    scheduler = lease->scheduler;
    if (pthread_mutex_lock(&scheduler->mutex) != 0)
        return scheduler_refuse(
            err, YVEX_ERR_STATE,
            "engine progress transition lock is unavailable");
    work = progress_slot_locked(scheduler, lease);
    if (!work || !work->active) {
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(
            err, YVEX_ERR_STATE,
            "engine progress transition has a stale sequence lease");
    }
    progress_complete_kind_locked(scheduler, work);
    progress_active_remove_locked(scheduler, work);
    scheduler_observation_add(&scheduler->summary.progress_transitions, 1ull);
    if (scheduler->stopping) {
        scheduler_observation_add(
            &scheduler->summary.progress_failures, 1ull);
        progress_release_locked(scheduler, work);
        memset(lease, 0, sizeof(*lease));
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(
            err, YVEX_ERR_STATE,
            "draining engine scheduler rejected a phase transition");
    }
    work->kind = kind;
    progress_enqueue_locked(scheduler, work);
    rc = progress_wait_locked(scheduler, work, err);
    if (rc != YVEX_OK) {
        progress_release_locked(scheduler, work);
        memset(lease, 0, sizeof(*lease));
    }
    (void)pthread_mutex_unlock(&scheduler->mutex);
    return rc;
}

int yvex_runtime_private_engine_progress_leave(
    runtime_engine_progress_lease *lease, int status, yvex_error *err)
{
    runtime_engine_scheduler *scheduler;
    runtime_engine_progress_work *work;
    if (!lease || !lease->scheduler)
        return status == YVEX_OK
                   ? scheduler_refuse(
                         err, YVEX_ERR_INVALID_ARG,
                         "engine progress lease is required")
                   : status;
    scheduler = lease->scheduler;
    if (pthread_mutex_lock(&scheduler->mutex) != 0)
        return status == YVEX_OK
                   ? scheduler_refuse(
                         err, YVEX_ERR_STATE,
                         "engine progress completion lock is unavailable")
                   : status;
    work = progress_slot_locked(scheduler, lease);
    if (!work) {
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return status == YVEX_OK
                   ? scheduler_refuse(
                         err, YVEX_ERR_STATE,
                         "engine progress lease is stale")
                   : status;
    }
    if (work->active) {
        progress_complete_kind_locked(scheduler, work);
        progress_active_remove_locked(scheduler, work);
    }
    if (status == YVEX_ERR_CANCELLED &&
        work->status != YVEX_ERR_CANCELLED)
        scheduler_observation_add(
            &scheduler->summary.progress_cancellations, 1ull);
    else if (status != YVEX_OK)
        scheduler_observation_add(
            &scheduler->summary.progress_failures, 1ull);
    progress_release_locked(scheduler, work);
    progress_grant_locked(scheduler);
    (void)pthread_mutex_unlock(&scheduler->mutex);
    memset(lease, 0, sizeof(*lease));
    if (status == YVEX_OK) yvex_error_clear(err);
    return status;
}

typedef struct {
    runtime_engine_work ticket;
    const runtime_engine_moe_request *request;
    yvex_execution_batch_source source;
} compatible_moe_ticket;

static int compatible_moe_request_valid(
    const runtime_engine_moe_request *request)
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
    const runtime_engine_moe_request *request)
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
    result->accelerated_matrix_launches = source->accelerated_matrix_launches;
    result->graph_launches = source->graph_launches;
    result->graph_captures = source->graph_captures;
    result->graph_replays = source->graph_replays;
    result->upload_count = source->upload_count;
    result->download_count = source->download_count;
    result->cache_hits = source->cache_hits;
    result->cache_misses = source->cache_misses;
    result->queue_synchronizations = source->queue_synchronizations;
    result->device_synchronizations = source->device_synchronizations;
    result->moe_ns = source->total_ns;
    result->synchronization_ns = source->synchronization_ns;
    yvex_runtime_identity_copy(result->routing_digest, source->routing_digest);
}

static int compatible_moe_source_prepare(compatible_moe_ticket *ticket,
                                         yvex_error *err)
{
    const runtime_engine_moe_request *request = ticket->request;
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
        return scheduler_refuse(
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
    const runtime_engine_moe_request *request = ticket->request;
    yvex_execution_compatibility_key *key = &ticket->ticket.key;
    if (request->session->engine != request->model ||
        !request->session->summary.engine_generation)
        return scheduler_refuse(err, YVEX_ERR_STATE,
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
    const runtime_engine_moe_request *request, yvex_error *err)
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
            return scheduler_refuse(err, YVEX_ERR_BOUNDS,
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
    const runtime_engine_moe_request *request = ticket->request;
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
    runtime_engine_work *const *tickets,
    unsigned long long ticket_count, yvex_error *err)
{
    compatible_moe_ticket **ordered = (compatible_moe_ticket **)tickets;
    compatible_moe_ticket *leader;
    const runtime_engine_moe_request *owner;
    yvex_moe_row_batch batch = {0};
    yvex_moe_row_batch_output output = {0};
    yvex_moe_row_batch_result physical = {0}, completion = {0};
    yvex_device_tensor batch_rows_view, batch_outputs_view;
    unsigned long long source_index, row_index, row_next = 0ull;
    unsigned long long expanded_values, hidden_values, post_values;
    unsigned long long d2d_bytes = 0ull;
    int rc = YVEX_OK;
    if (!ticket_count)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
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
        return scheduler_refuse(err, YVEX_ERR_BOUNDS,
                               "compatible MoE batch exceeds sealed capacity");
    for (source_index = 0ull; source_index < ticket_count && rc == YVEX_OK;
         ++source_index) {
        compatible_moe_ticket *entry = ordered[source_index];
        const runtime_engine_moe_request *request = entry->request;
        yvex_device_tensor destination;
        unsigned long long values;
        if (!yvex_core_u64_mul(request->row_count,
                               owner->transformer->expanded_width, &values) ||
            !yvex_backend_tensor_f32_subview(
                owner->batch_device_rows,
                row_next * owner->transformer->expanded_width, values,
                &destination))
            rc = scheduler_refuse(err, YVEX_ERR_BOUNDS,
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
                rc = scheduler_refuse(err, YVEX_ERR_BOUNDS,
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
        physical.queue_synchronizations += completion.queue_synchronizations;
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
        const runtime_engine_moe_request *request = entry->request;
        yvex_device_tensor source;
        unsigned long long values;
        if (!yvex_core_u64_mul(request->row_count,
                               owner->transformer->expanded_width, &values) ||
            !yvex_backend_tensor_f32_subview(
                &batch_outputs_view,
                row_next * owner->transformer->expanded_width, values,
                &source))
            rc = scheduler_refuse(err, YVEX_ERR_BOUNDS,
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

int yvex_runtime_private_engine_scheduler_moe_execute(
    const runtime_engine_moe_request *request, yvex_error *err)
{
    compatible_moe_ticket ticket;
    int rc;
    if (!compatible_moe_request_valid(request))
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible MoE request is incomplete");
    if (!request->compatible_scheduling || !request->model->engine_scheduler ||
        !request->execution_profile ||
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
            rc = yvex_runtime_private_engine_scheduler_submit(
                request->model->engine_scheduler, &ticket.ticket, err);
    }
    if (rc == YVEX_OK) compatible_moe_transformer_result(request);
    return rc;
}

typedef struct {
    runtime_engine_work ticket;
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
        return scheduler_refuse(err, YVEX_ERR_STATE,
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
    runtime_engine_work *const *tickets,
    unsigned long long ticket_count, yvex_error *err)
{
    compatible_logits_ticket **ordered = (compatible_logits_ticket **)tickets;
    yvex_runtime_logits_context *contexts[63];
    const yvex_runtime_logits_source *sources[63];
    yvex_runtime_logits_row_result *rows[63];
    unsigned long long index;
    if (!ticket_count || ticket_count >= 64ull)
        return scheduler_refuse(err, YVEX_ERR_BOUNDS,
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
    int rc;
    if (!context || !session)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
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
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible output-head request is incomplete");
    if (!ticket.model->engine_scheduler || ticket.admitted_width < 2ull ||
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
    if (rc == YVEX_OK)
        rc = yvex_runtime_private_engine_scheduler_submit(
            ticket.model->engine_scheduler, &ticket.ticket, err);
    return rc;
}

static int compatible_step_execute(
    runtime_engine_work *const *tickets,
    unsigned long long ticket_count, yvex_error *err)
{
    unsigned long long index;
    for (index = 0ull; index < ticket_count; ++index)
        if (!tickets[index] ||
            tickets[index]->kind != RUNTIME_ENGINE_WORK_RENDEZVOUS ||
            tickets[index]->execute != compatible_step_execute)
            return scheduler_refuse(
                err, YVEX_ERR_STATE,
                "non-rendezvous work entered a compatible step barrier");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int compatible_step_rendezvous(
    const runtime_engine_step_request *request, yvex_error *err)
{
    runtime_engine_work ticket = {0};
    yvex_execution_compatibility_key *key = &ticket.key;
    if (!request || !request->model || !request->session || !request->backend ||
        !request->transformer || !request->execution_profile ||
        request->maximum_width < 2ull || request->maximum_width >= 64ull ||
        request->phase >= YVEX_EXECUTION_PHASE_COUNT)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible execution step is incomplete");
    if (!request->model->engine_scheduler) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (request->session->engine != request->model ||
        !request->session->summary.engine_generation)
        return scheduler_refuse(err, YVEX_ERR_STATE,
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
    ticket.kind = RUNTIME_ENGINE_WORK_RENDEZVOUS;
    return yvex_runtime_private_engine_scheduler_submit(
        request->model->engine_scheduler, &ticket, err);
}

int yvex_runtime_private_model_scheduler_acquire(
    yvex_model_engine *model, unsigned long long sequence_capacity,
    unsigned long long maximum_width,
    yvex_error *err)
{
    runtime_engine_scheduler *created = NULL;
    int rc = YVEX_OK;
    if (!model || !sequence_capacity || sequence_capacity >= 64ull ||
        !maximum_width || maximum_width > sequence_capacity ||
        !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "bounded sequence and compatible widths are required");
    if (model->close_requested ||
        (model->engine_scheduler &&
         (model->scheduler_sequence_capacity != sequence_capacity ||
          model->scheduler_maximum_width != maximum_width))) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return scheduler_refuse(
            err, YVEX_ERR_STATE,
            model->close_requested
                ? "draining model engine cannot admit scheduling"
                : "runtime model scheduling envelope is already sealed");
    }
    if (!model->engine_scheduler) {
        rc = yvex_runtime_private_engine_scheduler_open(
            &created, sequence_capacity, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_private_engine_scheduler_start(created, err);
        if (rc == YVEX_OK) {
            model->engine_scheduler = created;
            model->scheduler_sequence_capacity = sequence_capacity;
            model->scheduler_maximum_width = maximum_width;
            created = NULL;
        }
    }
    if (rc == YVEX_OK &&
        model->engine_scheduler_references == ULLONG_MAX)
        rc = scheduler_refuse(err, YVEX_ERR_BOUNDS,
                             "model engine scheduler references overflowed");
    if (rc == YVEX_OK) model->engine_scheduler_references++;
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    if (created) (void)yvex_runtime_private_engine_scheduler_close(&created, NULL);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_runtime_private_model_scheduler_release(
    yvex_model_engine *model, yvex_error *err)
{
    runtime_engine_scheduler *owner = NULL;
    int rc;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "model engine scheduler ownership is required");
    if (!model->engine_scheduler || !model->engine_scheduler_references) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return scheduler_refuse(err, YVEX_ERR_STATE,
                               "model engine scheduler ownership is inconsistent");
    }
    if (model->engine_scheduler_references == 1ull &&
        model->engine_scheduler_producers) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return scheduler_refuse(
            err, YVEX_ERR_STATE,
            "active compatible producers prevent final ownership release");
    }
    model->engine_scheduler_references--;
    if (!model->engine_scheduler_references) {
        owner = model->engine_scheduler;
        model->engine_scheduler = NULL;
        model->scheduler_sequence_capacity = 0ull;
        model->scheduler_maximum_width = 0ull;
    }
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    rc = yvex_runtime_private_engine_scheduler_close(&owner, err);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_runtime_private_model_scheduler_finish(
    yvex_model_engine *model, int *acquired, yvex_error *err)
{
    int rc;
    if (!acquired || !*acquired) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_runtime_private_model_scheduler_release(model, err);
    if (rc == YVEX_OK) *acquired = 0;
    return rc;
}

int yvex_runtime_private_model_scheduler_producer_enter(
    yvex_model_engine *model, yvex_error *err)
{
    int rc;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "active engine scheduler producer is required");
    if (!model->engine_scheduler || !model->engine_scheduler_references ||
        model->engine_scheduler_producers >=
            model->scheduler_sequence_capacity) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return scheduler_refuse(
            err, YVEX_ERR_BOUNDS,
            "active compatible producer population exceeds admitted width");
    }
    model->engine_scheduler_producers++;
    rc = yvex_runtime_private_engine_scheduler_set_producers(
        model->engine_scheduler, model->engine_scheduler_producers, err);
    if (rc != YVEX_OK) model->engine_scheduler_producers--;
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    return rc;
}

int yvex_runtime_private_model_scheduler_producer_leave(
    yvex_model_engine *model, yvex_error *err)
{
    int rc;
    if (!model || !model->lifecycle_mutex_ready ||
        pthread_mutex_lock(&model->lifecycle_mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "active engine scheduler producer is required");
    if (!model->engine_scheduler || !model->engine_scheduler_producers) {
        (void)pthread_mutex_unlock(&model->lifecycle_mutex);
        return scheduler_refuse(err, YVEX_ERR_STATE,
                               "engine scheduler producer state is inconsistent");
    }
    model->engine_scheduler_producers--;
    rc = yvex_runtime_private_engine_scheduler_set_producers(
        model->engine_scheduler, model->engine_scheduler_producers, err);
    if (rc != YVEX_OK) model->engine_scheduler_producers++;
    (void)pthread_mutex_unlock(&model->lifecycle_mutex);
    return rc;
}

int yvex_runtime_private_engine_scheduler_producer_finish(
    yvex_model_engine *model, int *active, int status, yvex_error *err)
{
    int leave_status;
    if (!active || !*active) return status;
    leave_status = yvex_runtime_private_model_scheduler_producer_leave(
        model, status == YVEX_OK ? err : NULL);
    if (leave_status == YVEX_OK) *active = 0;
    return status == YVEX_OK ? leave_status : status;
}

int yvex_model_engine_scheduler_maximum_width_copy(
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
        return scheduler_refuse(
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

int yvex_runtime_private_engine_scheduler_step_rendezvous(
    const runtime_engine_step_request *request, yvex_error *err)
{
    if (!request || !request->model)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "compatible transformer step is required");
    return compatible_step_rendezvous(request, err);
}

int yvex_model_engine_scheduler_summary_copy(
    const yvex_model_engine *model, yvex_engine_scheduler_summary *out,
    yvex_error *err)
{
    yvex_model_engine *owner = (yvex_model_engine *)model;
    int rc = YVEX_OK;
    if (!owner || !out || !owner->lifecycle_mutex_ready ||
        pthread_mutex_lock(&owner->lifecycle_mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                               "model engine scheduler summary is unavailable");
    memset(out, 0, sizeof(*out));
    if (owner->engine_scheduler)
        rc = yvex_runtime_private_engine_scheduler_snapshot(owner->engine_scheduler,
                                                   out, err);
    if (rc == YVEX_OK) {
        out->enabled = owner->engine_scheduler != NULL;
        out->admitted_maximum_width = owner->scheduler_maximum_width;
        out->sequence_capacity = owner->scheduler_sequence_capacity;
    }
    (void)pthread_mutex_unlock(&owner->lifecycle_mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
