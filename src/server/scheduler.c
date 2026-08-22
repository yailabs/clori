/*
 * Serialize request-state mutation while independent workers execute compatible sessions.
 *
 * The scheduler mutex is the sole authority for queue order and active serialization keys.
 * Workers may execute concurrently, but a key remains active until its callback returns, so two
 * operations can never mutate the same session simultaneously.
 */
#define _POSIX_C_SOURCE 200809L

#include "src/server/private.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SCHEDULER_EXECUTION_READY_NS 3000000000ull

typedef struct {
    void *work;
    char key[YVEX_SERVER_SESSION_NAME_CAP];
    int compatible_batch_candidate;
} scheduler_entry;

typedef struct {
    char key[YVEX_SERVER_SESSION_NAME_CAP];
    int compatible_batch_candidate;
    int execution_ready;
} scheduler_active;

struct server_scheduler {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    scheduler_entry *queue;
    scheduler_active *active;
    pthread_t *workers;
    server_scheduler_execute execute;
    server_scheduler_observe observe;
    void *context;
    unsigned long long queue_capacity, queue_count;
    unsigned long long worker_count, worker_started, active_count;
    unsigned long long execution_ready_waits, execution_ready_timeouts;
    unsigned long long execution_ready_ns, maximum_execution_ready_width;
    int mutex_ready, condition_ready, started, stopping, finished;
};

static int scheduler_refuse(yvex_error *err, yvex_status status,
                            const char *reason)
{
    yvex_error_set(err, status, "server.scheduler", reason);
    return status;
}

static int scheduler_key_active(const server_scheduler *scheduler,
                                const char *key)
{
    unsigned long long index;
    for (index = 0ull; index < scheduler->active_count; ++index)
        if (strcmp(scheduler->active[index].key, key) == 0) return 1;
    return 0;
}

static unsigned long long scheduler_active_candidates(
    const server_scheduler *scheduler)
{
    unsigned long long index, count = 0ull;
    for (index = 0ull; index < scheduler->active_count; ++index)
        count += scheduler->active[index].compatible_batch_candidate != 0;
    return count;
}

static unsigned long long scheduler_ready_candidates(
    const server_scheduler *scheduler)
{
    unsigned long long index, count = 0ull;
    for (index = 0ull; index < scheduler->active_count; ++index)
        count += scheduler->active[index].compatible_batch_candidate &&
                 scheduler->active[index].execution_ready;
    return count;
}

static unsigned long long scheduler_active_find(
    const server_scheduler *scheduler, const char *key)
{
    unsigned long long index;
    for (index = 0ull; index < scheduler->active_count; ++index)
        if (strcmp(scheduler->active[index].key, key) == 0) break;
    return index;
}

static unsigned long long scheduler_batch_candidates(
    const server_scheduler *scheduler)
{
    unsigned long long index, count = 0ull;
    for (index = 0ull; index < scheduler->queue_count; ++index)
        count += scheduler->queue[index].compatible_batch_candidate != 0;
    return count;
}

static unsigned long long scheduler_elapsed_ns(const struct timespec *start,
                                               const struct timespec *finish)
{
    unsigned long long seconds, nanoseconds;
    if (finish->tv_sec < start->tv_sec ||
        (finish->tv_sec == start->tv_sec && finish->tv_nsec < start->tv_nsec))
        return 0ull;
    seconds = (unsigned long long)(finish->tv_sec - start->tv_sec);
    if (finish->tv_nsec >= start->tv_nsec) {
        nanoseconds = (unsigned long long)(finish->tv_nsec - start->tv_nsec);
    } else {
        if (!seconds) return 0ull;
        seconds--;
        nanoseconds = 1000000000ull + (unsigned long long)finish->tv_nsec -
                      (unsigned long long)start->tv_nsec;
    }
    if (seconds > (ULLONG_MAX - nanoseconds) / 1000000000ull)
        return ULLONG_MAX;
    return seconds * 1000000000ull + nanoseconds;
}

static int scheduler_select_locked(server_scheduler *scheduler,
                                   scheduler_entry *entry)
{
    unsigned long long index;
    for (index = 0ull; index < scheduler->queue_count; ++index)
        if (!scheduler_key_active(scheduler, scheduler->queue[index].key))
            break;
    if (index == scheduler->queue_count) return 0;
    *entry = scheduler->queue[index];
    for (; index + 1ull < scheduler->queue_count; ++index)
        scheduler->queue[index] = scheduler->queue[index + 1ull];
    scheduler->queue_count--;
    memset(&scheduler->queue[scheduler->queue_count], 0,
           sizeof(*scheduler->queue));
    memcpy(scheduler->active[scheduler->active_count].key, entry->key,
           sizeof(entry->key));
    scheduler->active[scheduler->active_count].compatible_batch_candidate =
        entry->compatible_batch_candidate;
    scheduler->active[scheduler->active_count].execution_ready = 0;
    scheduler->active_count++;
    return 1;
}

static void scheduler_release_locked(server_scheduler *scheduler,
                                     const char *key)
{
    unsigned long long index;
    for (index = 0ull; index < scheduler->active_count; ++index)
        if (strcmp(scheduler->active[index].key, key) == 0) break;
    if (index == scheduler->active_count) return;
    for (; index + 1ull < scheduler->active_count; ++index)
        scheduler->active[index] = scheduler->active[index + 1ull];
    scheduler->active_count--;
    memset(&scheduler->active[scheduler->active_count], 0,
           sizeof(*scheduler->active));
}

static void scheduler_observe_locked(server_scheduler *scheduler)
{
    if (scheduler->observe)
        scheduler->observe(scheduler->context, scheduler->queue_count,
                           scheduler->queue_capacity, scheduler->active_count);
}

static void *scheduler_worker(void *opaque)
{
    server_scheduler *scheduler = opaque;
    for (;;) {
        scheduler_entry entry = {0};
        (void)pthread_mutex_lock(&scheduler->mutex);
        for (;;) {
            if (scheduler->stopping && !scheduler->queue_count) {
                (void)pthread_mutex_unlock(&scheduler->mutex);
                return NULL;
            }
            while (!scheduler->queue_count && !scheduler->stopping)
                (void)pthread_cond_wait(&scheduler->condition,
                                        &scheduler->mutex);
            if (scheduler->stopping && !scheduler->queue_count) {
                (void)pthread_mutex_unlock(&scheduler->mutex);
                return NULL;
            }
            if (scheduler_select_locked(scheduler, &entry)) break;
            (void)pthread_cond_wait(&scheduler->condition, &scheduler->mutex);
        }
        scheduler_observe_locked(scheduler);
        (void)pthread_mutex_unlock(&scheduler->mutex);
        scheduler->execute(scheduler->context, entry.work);
        (void)pthread_mutex_lock(&scheduler->mutex);
        scheduler_release_locked(scheduler, entry.key);
        scheduler_observe_locked(scheduler);
        (void)pthread_cond_broadcast(&scheduler->condition);
        (void)pthread_mutex_unlock(&scheduler->mutex);
    }
}

int yvex_server_scheduler_open(
    server_scheduler **out, unsigned long long queue_capacity,
    unsigned long long worker_count, server_scheduler_execute execute,
    server_scheduler_observe observe, void *context, yvex_error *err)
{
    server_scheduler *scheduler;
    if (out) *out = NULL;
    if (!out || !queue_capacity || !worker_count || !execute ||
        queue_capacity > SIZE_MAX / sizeof(scheduler_entry) ||
        worker_count > SIZE_MAX / sizeof(pthread_t) ||
        worker_count > SIZE_MAX / sizeof(scheduler_active))
        return scheduler_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "bounded queue, workers, execution callback, and output are required");
    scheduler = calloc(1u, sizeof(*scheduler));
    if (!scheduler)
        return scheduler_refuse(err, YVEX_ERR_NOMEM,
                                "scheduler allocation failed");
    scheduler->queue = calloc((size_t)queue_capacity, sizeof(*scheduler->queue));
    scheduler->active = calloc((size_t)worker_count,
                               sizeof(*scheduler->active));
    scheduler->workers = calloc((size_t)worker_count,
                                sizeof(*scheduler->workers));
    if (!scheduler->queue || !scheduler->active || !scheduler->workers) {
        yvex_server_scheduler_close(&scheduler);
        return scheduler_refuse(err, YVEX_ERR_NOMEM,
                                "scheduler bounded storage allocation failed");
    }
    scheduler->queue_capacity = queue_capacity;
    scheduler->worker_count = worker_count;
    scheduler->execute = execute;
    scheduler->observe = observe;
    scheduler->context = context;
    if (pthread_mutex_init(&scheduler->mutex, NULL) != 0) {
        yvex_server_scheduler_close(&scheduler);
        return scheduler_refuse(err, YVEX_ERR_STATE,
                                "scheduler mutex initialization failed");
    }
    scheduler->mutex_ready = 1;
    if (pthread_cond_init(&scheduler->condition, NULL) != 0) {
        yvex_server_scheduler_close(&scheduler);
        return scheduler_refuse(err, YVEX_ERR_STATE,
                                "scheduler condition initialization failed");
    }
    scheduler->condition_ready = 1;
    *out = scheduler;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_scheduler_start(server_scheduler *scheduler, yvex_error *err)
{
    unsigned long long index;
    if (!scheduler || !scheduler->mutex_ready ||
        pthread_mutex_lock(&scheduler->mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                                "open scheduler is required");
    if (scheduler->started || scheduler->stopping) {
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(err, YVEX_ERR_STATE,
                                "scheduler cannot start twice or after stop");
    }
    scheduler->started = 1;
    (void)pthread_mutex_unlock(&scheduler->mutex);
    for (index = 0ull; index < scheduler->worker_count; ++index) {
        if (pthread_create(&scheduler->workers[index], NULL,
                           scheduler_worker, scheduler) != 0) {
            yvex_server_scheduler_request_stop(scheduler);
            (void)yvex_server_scheduler_finish(scheduler, NULL);
            return scheduler_refuse(err, YVEX_ERR_STATE,
                                    "scheduler worker creation failed");
        }
        scheduler->worker_started++;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_scheduler_submit(server_scheduler *scheduler, void *work,
                                 const char *serialization_key,
                                 int compatible_batch_candidate,
                                 unsigned long long *queued, yvex_error *err)
{
    scheduler_entry *entry;
    unsigned long long depth;
    if (queued) *queued = 0ull;
    if (!scheduler || !work || !serialization_key ||
        (compatible_batch_candidate != 0 && compatible_batch_candidate != 1) ||
        strlen(serialization_key) >= YVEX_SERVER_SESSION_NAME_CAP ||
        !scheduler->mutex_ready || pthread_mutex_lock(&scheduler->mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                                "scheduler work and bounded key are required");
    if (!scheduler->started || scheduler->stopping ||
        scheduler->queue_count == scheduler->queue_capacity) {
        int status = scheduler->queue_count == scheduler->queue_capacity
                         ? YVEX_ERR_BOUNDS : YVEX_ERR_STATE;
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(
            err, (yvex_status)status,
            status == YVEX_ERR_BOUNDS ? "bounded scheduler queue is full"
                                      : "scheduler is not accepting work");
    }
    entry = &scheduler->queue[scheduler->queue_count++];
    entry->work = work;
    memcpy(entry->key, serialization_key, strlen(serialization_key) + 1u);
    entry->compatible_batch_candidate = compatible_batch_candidate;
    depth = scheduler->queue_count;
    if (queued) *queued = depth;
    scheduler_observe_locked(scheduler);
    (void)pthread_cond_broadcast(&scheduler->condition);
    (void)pthread_mutex_unlock(&scheduler->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_scheduler_execution_ready(
    server_scheduler *scheduler, const char *serialization_key,
    unsigned long long *compatible_width, unsigned long long *wait_ns,
    int *timed_out, yvex_error *err)
{
    struct timespec deadline, finish, start;
    unsigned long long active_index, elapsed = 0ull, width;
    int cohort_pending, wait_status = 0;
    if (compatible_width) *compatible_width = 0ull;
    if (wait_ns) *wait_ns = 0ull;
    if (timed_out) *timed_out = 0;
    if (!scheduler || !serialization_key ||
        strlen(serialization_key) >= YVEX_SERVER_SESSION_NAME_CAP ||
        !scheduler->mutex_ready || pthread_mutex_lock(&scheduler->mutex) != 0)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                                "open scheduler and bounded active key are required");
    active_index = scheduler_active_find(scheduler, serialization_key);
    if (active_index == scheduler->active_count ||
        !scheduler->active[active_index].compatible_batch_candidate) {
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(err, YVEX_ERR_STATE,
                                "active compatible generation work is required");
    }
    scheduler->active[active_index].execution_ready = 1;
    width = scheduler_active_candidates(scheduler);
    cohort_pending = scheduler->worker_count > 1ull &&
                     (width > 1ull || scheduler->active_count > width ||
                      scheduler_batch_candidates(scheduler));
    (void)pthread_cond_broadcast(&scheduler->condition);
    if (cohort_pending && !scheduler->stopping &&
        clock_gettime(CLOCK_REALTIME, &deadline) == 0 &&
        clock_gettime(CLOCK_MONOTONIC, &start) == 0) {
        deadline.tv_sec += (time_t)(SCHEDULER_EXECUTION_READY_NS / 1000000000ull);
        deadline.tv_nsec += (long)(SCHEDULER_EXECUTION_READY_NS % 1000000000ull);
        if (deadline.tv_nsec >= 1000000000l) {
            deadline.tv_sec++;
            deadline.tv_nsec -= 1000000000l;
        }
        scheduler->execution_ready_waits++;
        while (!scheduler->stopping &&
               (width < 2ull ||
                scheduler_ready_candidates(scheduler) < width) &&
               wait_status != ETIMEDOUT) {
            wait_status = pthread_cond_timedwait(
                &scheduler->condition, &scheduler->mutex, &deadline);
            width = scheduler_active_candidates(scheduler);
        }
        if (wait_status == ETIMEDOUT) {
            scheduler->execution_ready_timeouts++;
            if (timed_out) *timed_out = 1;
        } else if (wait_status != 0) {
            (void)pthread_mutex_unlock(&scheduler->mutex);
            return scheduler_refuse(err, YVEX_ERR_STATE,
                                    "execution-ready rendezvous wait failed");
        }
        if (clock_gettime(CLOCK_MONOTONIC, &finish) == 0)
            elapsed = scheduler_elapsed_ns(&start, &finish);
        if (ULLONG_MAX - scheduler->execution_ready_ns < elapsed)
            scheduler->execution_ready_ns = ULLONG_MAX;
        else
            scheduler->execution_ready_ns += elapsed;
    }
    if (scheduler->stopping) {
        (void)pthread_mutex_unlock(&scheduler->mutex);
        return scheduler_refuse(err, YVEX_ERR_STATE,
                                "scheduler stopped before execution readiness");
    }
    if (width > scheduler->maximum_execution_ready_width)
        scheduler->maximum_execution_ready_width = width;
    if (compatible_width) *compatible_width = width;
    if (wait_ns) *wait_ns = elapsed;
    (void)pthread_mutex_unlock(&scheduler->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_server_scheduler_request_stop(server_scheduler *scheduler)
{
    if (!scheduler || !scheduler->mutex_ready ||
        pthread_mutex_lock(&scheduler->mutex) != 0)
        return;
    scheduler->stopping = 1;
    (void)pthread_cond_broadcast(&scheduler->condition);
    (void)pthread_mutex_unlock(&scheduler->mutex);
}

int yvex_server_scheduler_finish(server_scheduler *scheduler, yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    if (!scheduler)
        return scheduler_refuse(err, YVEX_ERR_INVALID_ARG,
                                "scheduler is required");
    yvex_server_scheduler_request_stop(scheduler);
    if (scheduler->finished) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    for (index = 0ull; index < scheduler->worker_started; ++index)
        if (pthread_join(scheduler->workers[index], NULL) != 0 && rc == YVEX_OK)
            rc = scheduler_refuse(err, YVEX_ERR_STATE,
                                  "scheduler worker join failed");
    scheduler->worker_started = 0ull;
    scheduler->finished = 1;
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

void yvex_server_scheduler_snapshot(const server_scheduler *scheduler,
                                    server_scheduler_summary *summary)
{
    server_scheduler *mutable = (server_scheduler *)scheduler;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!scheduler || !summary || !scheduler->mutex_ready ||
        pthread_mutex_lock(&mutable->mutex) != 0)
        return;
    summary->queued = scheduler->queue_count;
    summary->capacity = scheduler->queue_capacity;
    summary->active = scheduler->active_count;
    summary->workers = scheduler->worker_count;
    summary->execution_ready_limit_ns = SCHEDULER_EXECUTION_READY_NS;
    summary->execution_ready_waits = scheduler->execution_ready_waits;
    summary->execution_ready_timeouts = scheduler->execution_ready_timeouts;
    summary->execution_ready_ns = scheduler->execution_ready_ns;
    summary->maximum_execution_ready_width =
        scheduler->maximum_execution_ready_width;
    (void)pthread_mutex_unlock(&mutable->mutex);
}

void yvex_server_scheduler_close(server_scheduler **scheduler)
{
    server_scheduler *owner;
    if (!scheduler || !*scheduler) return;
    owner = *scheduler;
    (void)yvex_server_scheduler_finish(owner, NULL);
    if (owner->condition_ready) (void)pthread_cond_destroy(&owner->condition);
    if (owner->mutex_ready) (void)pthread_mutex_destroy(&owner->mutex);
    free(owner->workers);
    free(owner->active);
    free(owner->queue);
    free(owner);
    *scheduler = NULL;
}
