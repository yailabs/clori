/*
 * Serialize request-state mutation while independent workers execute compatible sessions.
 *
 * The scheduler mutex is the sole authority for queue order and active serialization keys.
 * Workers may execute concurrently, but a key remains active until its callback returns, so two
 * operations can never mutate the same session simultaneously.
 */
#define _POSIX_C_SOURCE 200809L

#include "src/server/private.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *work;
    char key[YVEX_SERVER_SESSION_NAME_CAP];
} scheduler_entry;

struct server_scheduler {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    scheduler_entry *queue;
    char (*active_keys)[YVEX_SERVER_SESSION_NAME_CAP];
    pthread_t *workers;
    server_scheduler_execute execute;
    server_scheduler_observe observe;
    void *context;
    unsigned long long queue_capacity, queue_count;
    unsigned long long worker_count, worker_started, active_count;
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
        if (strcmp(scheduler->active_keys[index], key) == 0) return 1;
    return 0;
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
    memcpy(scheduler->active_keys[scheduler->active_count], entry->key,
           sizeof(entry->key));
    scheduler->active_count++;
    return 1;
}

static void scheduler_release_locked(server_scheduler *scheduler,
                                     const char *key)
{
    unsigned long long index;
    for (index = 0ull; index < scheduler->active_count; ++index)
        if (strcmp(scheduler->active_keys[index], key) == 0) break;
    if (index == scheduler->active_count) return;
    for (; index + 1ull < scheduler->active_count; ++index)
        memcpy(scheduler->active_keys[index], scheduler->active_keys[index + 1ull],
               YVEX_SERVER_SESSION_NAME_CAP);
    scheduler->active_count--;
    memset(scheduler->active_keys[scheduler->active_count], 0,
           YVEX_SERVER_SESSION_NAME_CAP);
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
        while (!scheduler_select_locked(scheduler, &entry)) {
            if (scheduler->stopping && !scheduler->queue_count) {
                (void)pthread_mutex_unlock(&scheduler->mutex);
                return NULL;
            }
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
        worker_count > SIZE_MAX / YVEX_SERVER_SESSION_NAME_CAP)
        return scheduler_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "bounded queue, workers, execution callback, and output are required");
    scheduler = calloc(1u, sizeof(*scheduler));
    if (!scheduler)
        return scheduler_refuse(err, YVEX_ERR_NOMEM,
                                "scheduler allocation failed");
    scheduler->queue = calloc((size_t)queue_capacity, sizeof(*scheduler->queue));
    scheduler->active_keys = calloc((size_t)worker_count,
                                    sizeof(*scheduler->active_keys));
    scheduler->workers = calloc((size_t)worker_count,
                                sizeof(*scheduler->workers));
    if (!scheduler->queue || !scheduler->active_keys || !scheduler->workers) {
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
                                 unsigned long long *queued, yvex_error *err)
{
    scheduler_entry *entry;
    unsigned long long depth;
    if (queued) *queued = 0ull;
    if (!scheduler || !work || !serialization_key ||
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
    depth = scheduler->queue_count;
    if (queued) *queued = depth;
    scheduler_observe_locked(scheduler);
    (void)pthread_cond_broadcast(&scheduler->condition);
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
                                    unsigned long long *queued,
                                    unsigned long long *active)
{
    server_scheduler *mutable = (server_scheduler *)scheduler;
    if (queued) *queued = 0ull;
    if (active) *active = 0ull;
    if (!scheduler || !scheduler->mutex_ready ||
        pthread_mutex_lock(&mutable->mutex) != 0)
        return;
    if (queued) *queued = scheduler->queue_count;
    if (active) *active = scheduler->active_count;
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
    free(owner->active_keys);
    free(owner->queue);
    free(owner);
    *scheduler = NULL;
}
