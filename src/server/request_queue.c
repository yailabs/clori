/*
 * Serialize request-state mutation while independent workers execute compatible sessions.
 *
 * The request-queue mutex is the sole authority for queue order and active serialization keys.
 * Workers may execute concurrently, but a key remains active until its callback returns, so two
 * operations can never mutate the same session simultaneously.
 */
#define _POSIX_C_SOURCE 200809L

#include "src/server/private.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *work;
    char key[SERVER_REQUEST_QUEUE_KEY_CAP];
} request_queue_entry;

typedef struct {
    char key[SERVER_REQUEST_QUEUE_KEY_CAP];
} request_queue_active;

struct server_request_queue {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    request_queue_entry *queue;
    request_queue_active *active;
    pthread_t *workers;
    server_request_queue_execute execute;
    server_request_queue_observe observe;
    void *context;
    unsigned long long queue_capacity, queue_count;
    unsigned long long worker_count, worker_started, active_count;
    int mutex_ready, condition_ready, started, stopping, finished;
};

static int request_queue_refuse(yvex_error *err, yvex_status status,
                                const char *reason)
{
    yvex_error_set(err, status, "server.request-queue", reason);
    return status;
}

static int request_queue_key_active(const server_request_queue *request_queue,
                                    const char *key)
{
    unsigned long long index;
    for (index = 0ull; index < request_queue->active_count; ++index)
        if (strcmp(request_queue->active[index].key, key) == 0) return 1;
    return 0;
}

static int request_queue_select_locked(server_request_queue *request_queue,
                                       request_queue_entry *entry)
{
    unsigned long long index;
    for (index = 0ull; index < request_queue->queue_count; ++index)
        if (!request_queue_key_active(request_queue,
                                      request_queue->queue[index].key))
            break;
    if (index == request_queue->queue_count) return 0;
    *entry = request_queue->queue[index];
    for (; index + 1ull < request_queue->queue_count; ++index)
        request_queue->queue[index] = request_queue->queue[index + 1ull];
    request_queue->queue_count--;
    memset(&request_queue->queue[request_queue->queue_count], 0,
           sizeof(*request_queue->queue));
    memcpy(request_queue->active[request_queue->active_count].key, entry->key,
           sizeof(entry->key));
    request_queue->active_count++;
    return 1;
}

static void request_queue_release_locked(server_request_queue *request_queue,
                                         const char *key)
{
    unsigned long long index;
    for (index = 0ull; index < request_queue->active_count; ++index)
        if (strcmp(request_queue->active[index].key, key) == 0) break;
    if (index == request_queue->active_count) return;
    for (; index + 1ull < request_queue->active_count; ++index)
        request_queue->active[index] = request_queue->active[index + 1ull];
    request_queue->active_count--;
    memset(&request_queue->active[request_queue->active_count], 0,
           sizeof(*request_queue->active));
}

static void request_queue_observe_locked(server_request_queue *request_queue)
{
    if (request_queue->observe)
        request_queue->observe(request_queue->context, request_queue->queue_count,
                               request_queue->queue_capacity,
                               request_queue->active_count);
}

static void *request_queue_worker(void *opaque)
{
    server_request_queue *request_queue = opaque;
    for (;;) {
        request_queue_entry entry = {0};
        (void)pthread_mutex_lock(&request_queue->mutex);
        for (;;) {
            if (request_queue->stopping && !request_queue->queue_count) {
                (void)pthread_mutex_unlock(&request_queue->mutex);
                return NULL;
            }
            while (!request_queue->queue_count && !request_queue->stopping)
                (void)pthread_cond_wait(&request_queue->condition,
                                        &request_queue->mutex);
            if (request_queue->stopping && !request_queue->queue_count) {
                (void)pthread_mutex_unlock(&request_queue->mutex);
                return NULL;
            }
            if (request_queue_select_locked(request_queue, &entry)) break;
            (void)pthread_cond_wait(&request_queue->condition,
                                    &request_queue->mutex);
        }
        request_queue_observe_locked(request_queue);
        (void)pthread_mutex_unlock(&request_queue->mutex);
        request_queue->execute(request_queue->context, entry.work);
        (void)pthread_mutex_lock(&request_queue->mutex);
        request_queue_release_locked(request_queue, entry.key);
        request_queue_observe_locked(request_queue);
        (void)pthread_cond_broadcast(&request_queue->condition);
        (void)pthread_mutex_unlock(&request_queue->mutex);
    }
}

int yvex_server_request_queue_open(
    server_request_queue **out, unsigned long long queue_capacity,
    unsigned long long worker_count, server_request_queue_execute execute,
    server_request_queue_observe observe, void *context, yvex_error *err)
{
    server_request_queue *request_queue;
    if (out) *out = NULL;
    if (!out || !queue_capacity || !worker_count || !execute ||
        queue_capacity > SIZE_MAX / sizeof(request_queue_entry) ||
        worker_count > SIZE_MAX / sizeof(pthread_t) ||
        worker_count > SIZE_MAX / sizeof(request_queue_active))
        return request_queue_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "bounded queue, workers, execution callback, and output are required");
    request_queue = calloc(1u, sizeof(*request_queue));
    if (!request_queue)
        return request_queue_refuse(err, YVEX_ERR_NOMEM,
                                    "request queue allocation failed");
    request_queue->queue =
        calloc((size_t)queue_capacity, sizeof(*request_queue->queue));
    request_queue->active = calloc((size_t)worker_count,
                                   sizeof(*request_queue->active));
    request_queue->workers = calloc((size_t)worker_count,
                                    sizeof(*request_queue->workers));
    if (!request_queue->queue || !request_queue->active || !request_queue->workers) {
        yvex_server_request_queue_close(&request_queue);
        return request_queue_refuse(
            err, YVEX_ERR_NOMEM,
            "request queue bounded storage allocation failed");
    }
    request_queue->queue_capacity = queue_capacity;
    request_queue->worker_count = worker_count;
    request_queue->execute = execute;
    request_queue->observe = observe;
    request_queue->context = context;
    if (pthread_mutex_init(&request_queue->mutex, NULL) != 0) {
        yvex_server_request_queue_close(&request_queue);
        return request_queue_refuse(err, YVEX_ERR_STATE,
                                    "request queue mutex initialization failed");
    }
    request_queue->mutex_ready = 1;
    if (pthread_cond_init(&request_queue->condition, NULL) != 0) {
        yvex_server_request_queue_close(&request_queue);
        return request_queue_refuse(
            err, YVEX_ERR_STATE,
            "request queue condition initialization failed");
    }
    request_queue->condition_ready = 1;
    *out = request_queue;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_request_queue_start(server_request_queue *request_queue,
                                    yvex_error *err)
{
    unsigned long long index;
    if (!request_queue || !request_queue->mutex_ready ||
        pthread_mutex_lock(&request_queue->mutex) != 0)
        return request_queue_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "open request queue is required");
    if (request_queue->started || request_queue->stopping) {
        (void)pthread_mutex_unlock(&request_queue->mutex);
        return request_queue_refuse(
            err, YVEX_ERR_STATE,
            "request queue cannot start twice or after stop");
    }
    request_queue->started = 1;
    (void)pthread_mutex_unlock(&request_queue->mutex);
    for (index = 0ull; index < request_queue->worker_count; ++index) {
        if (pthread_create(&request_queue->workers[index], NULL,
                           request_queue_worker, request_queue) != 0) {
            yvex_server_request_queue_request_stop(request_queue);
            (void)yvex_server_request_queue_finish(request_queue, NULL);
            return request_queue_refuse(err, YVEX_ERR_STATE,
                                        "request queue worker creation failed");
        }
        request_queue->worker_started++;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_request_queue_key(
    char output[SERVER_REQUEST_QUEUE_KEY_CAP],
    unsigned long long engine_generation, const char *session_name,
    yvex_error *err)
{
    int written;
    if (output) output[0] = '\0';
    if (!output || !engine_generation || !session_name || !session_name[0])
        return request_queue_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "engine generation, session, and key output are required");
    written = snprintf(output, SERVER_REQUEST_QUEUE_KEY_CAP, "%llu:%s",
                       engine_generation, session_name);
    if (written < 0 || (unsigned int)written >= SERVER_REQUEST_QUEUE_KEY_CAP) {
        output[0] = '\0';
        return request_queue_refuse(err, YVEX_ERR_BOUNDS,
                                    "engine-session request queue key exceeds its bound");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_request_queue_submit(server_request_queue *request_queue,
                                     void *work,
                                     const char *serialization_key,
                                     unsigned long long *queued, yvex_error *err)
{
    request_queue_entry *entry;
    unsigned long long depth;
    if (queued) *queued = 0ull;
    if (!request_queue || !work || !serialization_key ||
        strlen(serialization_key) >= SERVER_REQUEST_QUEUE_KEY_CAP ||
        !request_queue->mutex_ready ||
        pthread_mutex_lock(&request_queue->mutex) != 0)
        return request_queue_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "request queue work and bounded key are required");
    if (!request_queue->started || request_queue->stopping ||
        request_queue->queue_count == request_queue->queue_capacity) {
        int status = request_queue->queue_count == request_queue->queue_capacity
                         ? YVEX_ERR_BOUNDS : YVEX_ERR_STATE;
        (void)pthread_mutex_unlock(&request_queue->mutex);
        return request_queue_refuse(
            err, (yvex_status)status,
            status == YVEX_ERR_BOUNDS ? "bounded request queue is full"
                                      : "request queue is not accepting work");
    }
    entry = &request_queue->queue[request_queue->queue_count++];
    entry->work = work;
    memcpy(entry->key, serialization_key, strlen(serialization_key) + 1u);
    depth = request_queue->queue_count;
    if (queued) *queued = depth;
    request_queue_observe_locked(request_queue);
    (void)pthread_cond_broadcast(&request_queue->condition);
    (void)pthread_mutex_unlock(&request_queue->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_server_request_queue_request_stop(server_request_queue *request_queue)
{
    if (!request_queue || !request_queue->mutex_ready ||
        pthread_mutex_lock(&request_queue->mutex) != 0)
        return;
    request_queue->stopping = 1;
    (void)pthread_cond_broadcast(&request_queue->condition);
    (void)pthread_mutex_unlock(&request_queue->mutex);
}

int yvex_server_request_queue_finish(server_request_queue *request_queue,
                                     yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    if (!request_queue)
        return request_queue_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "request queue is required");
    yvex_server_request_queue_request_stop(request_queue);
    if (request_queue->finished) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    for (index = 0ull; index < request_queue->worker_started; ++index)
        if (pthread_join(request_queue->workers[index], NULL) != 0 && rc == YVEX_OK)
            rc = request_queue_refuse(err, YVEX_ERR_STATE,
                                      "request queue worker join failed");
    request_queue->worker_started = 0ull;
    request_queue->finished = 1;
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

void yvex_server_request_queue_snapshot(const server_request_queue *request_queue,
                                        server_request_queue_summary *summary)
{
    server_request_queue *mutable = (server_request_queue *)request_queue;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!request_queue || !summary || !request_queue->mutex_ready ||
        pthread_mutex_lock(&mutable->mutex) != 0)
        return;
    summary->queued = request_queue->queue_count;
    summary->capacity = request_queue->queue_capacity;
    summary->active = request_queue->active_count;
    summary->workers = request_queue->worker_count;
    (void)pthread_mutex_unlock(&mutable->mutex);
}

void yvex_server_request_queue_close(server_request_queue **request_queue)
{
    server_request_queue *owner;
    if (!request_queue || !*request_queue) return;
    owner = *request_queue;
    (void)yvex_server_request_queue_finish(owner, NULL);
    if (owner->condition_ready) (void)pthread_cond_destroy(&owner->condition);
    if (owner->mutex_ready) (void)pthread_mutex_destroy(&owner->mutex);
    free(owner->workers);
    free(owner->active);
    free(owner->queue);
    free(owner);
    *request_queue = NULL;
}
