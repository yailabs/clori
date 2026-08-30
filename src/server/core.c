/*
 * Keep one persistent host alive while independently managed model engines serve typed work.
 *
 * The host owns transport and request admission. Engine generations own model resources and
 * sessions; queued work holds a generation lease through completion.
 */
#define _GNU_SOURCE
#include "src/server/private.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include <yvex/internal/core.h>
#include <yvex/internal/engine_scheduler.h>
#define SERVER_TELEMETRY_CAPACITY 4096u
#define SERVER_CLIENT_CAPACITY 64u
typedef struct server_work_item {
    yvex_client_request request;
    server_engine_lease engine;
    yvex_server_engine_summary engine_summary;
    server_event_scope event_scope;
    char request_id[YVEX_SERVER_ID_CAP];
    unsigned char *prompt;
    yvex_provider_request *provider;
    unsigned long long enqueued_ns;
    int fd, done, response_sent, status;
    yvex_client_failure_class failure_class;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    yvex_error error;
} server_work_item;
typedef struct {
    struct yvex_server *server;
    int fd, active, subscriber;
} server_client_slot;
struct yvex_server {
    pthread_mutex_t state_mutex, clients_mutex;
    pthread_cond_t clients_condition;
    yvex_server_options options;
    yvex_server_summary summary;
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    char lock_path[YVEX_SERVER_SOCKET_PATH_CAP];
    server_telemetry *telemetry;
    server_engine_manager *engines;
    server_openai_listener *openai;
    unsigned long long next_request_id;
    server_client_slot *clients;
    unsigned long long client_capacity, active_clients;
    int listen_fd, lock_fd, lock_owned;
    atomic_int stopping;
    int state_mutex_ready;
    int clients_mutex_ready, clients_condition_ready;
    int finish_started, finish_completed;
};

static void model_work_execute(void *context, void *work);

static int server_refuse(yvex_error *err, yvex_status status,
                         const char *reason)
{
    yvex_error_set(err, status, "server.host", reason);
    return status;
}

yvex_client_failure_class yvex_server_failure_class_from_status(int status)
{
    switch (status) {
    case YVEX_ERR_FORMAT:
    case YVEX_ERR_INVALID_ARG: return YVEX_CLIENT_FAILURE_INVALID_REQUEST;
    case YVEX_ERR_UNSUPPORTED:
        return YVEX_CLIENT_FAILURE_UNSUPPORTED_PARAMETER;
    case YVEX_ERR_BOUNDS: return YVEX_CLIENT_FAILURE_REQUEST_TOO_LARGE;
    case YVEX_ERR_STATE: return YVEX_CLIENT_FAILURE_INCOMPATIBLE_STATE;
    case YVEX_ERR_CANCELLED: return YVEX_CLIENT_FAILURE_CLIENT_CANCELLED;
    case YVEX_ERR_IO:
    case YVEX_ERR_BACKEND: return YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE;
    case YVEX_ERR_TIMEOUT: return YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT;
    default: return YVEX_CLIENT_FAILURE_INTERNAL;
    }
}

static int peer_validate(int fd, yvex_error *err)
{
#ifdef SO_PEERCRED
    struct ucred credentials;
    socklen_t count = sizeof(credentials);
    memset(&credentials, 0, sizeof(credentials));
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &count) != 0 ||
        count != sizeof(credentials) || credentials.uid != geteuid())
        return server_refuse(err, YVEX_ERR_STATE,
                             "local client UID does not own the runtime");
    return YVEX_OK;
#else
    (void)fd;
    return server_refuse(err, YVEX_ERR_UNSUPPORTED,
                         "local peer credential validation is unavailable");
#endif
}

static unsigned long long server_monotonic_ns(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0u;
    return (unsigned long long)value.tv_sec * 1000000000ull +
           (unsigned long long)value.tv_nsec;
}

static int server_options_admit(yvex_server *server,
                                const yvex_server_options *options,
                                yvex_error *err)
{
    char canonical[YVEX_SERVER_SOCKET_PATH_CAP];
    if (!options)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "runtime-host options are required");
    /* Classify the public layout before reading any field absent from legacy v3. */
    if (options->schema_version != YVEX_SERVER_OPTIONS_SCHEMA_CURRENT)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "unsupported server-options schema");
    if (!options->request_queue_capacity || !options->worker_count ||
        options->worker_count > SERVER_CLIENT_CAPACITY ||
        options->maximum_engines > YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES ||
        options->trace_level > YVEX_SERVER_TRACE_FULL ||
        options->console > YVEX_SERVER_CONSOLE_HUMAN ||
        (options->openai_enabled &&
         (!options->openai_port || options->openai_timeout_ms < 100u ||
          options->openai_timeout_ms > 86400000u)))
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "complete bounded runtime-host options are required");
    server->options = *options;
    if (!server->options.maximum_engines)
        server->options.maximum_engines =
            YVEX_SERVER_DEFAULT_MAXIMUM_ENGINES;
    if (options->socket_path)
        yvex_core_text_copy(server->socket_path, sizeof(server->socket_path),
                            options->socket_path);
    else {
        if (yvex_server_socket_path(canonical, err) != YVEX_OK)
            return yvex_error_code(err);
        yvex_core_text_copy(server->socket_path, sizeof(server->socket_path),
                            canonical);
    }
    if (!server->socket_path[0] || server->socket_path[0] != '/' ||
        strlen(server->socket_path) >= sizeof(((struct sockaddr_un *)0)->sun_path))
        return server_refuse(err, YVEX_ERR_BOUNDS,
                             "local socket path must be absolute and fit AF_UNIX");
    if (snprintf(server->lock_path, sizeof(server->lock_path), "%s.lock",
                 server->socket_path) < 0 ||
        strlen(server->lock_path) >= sizeof(server->lock_path) - 1u)
        return server_refuse(err, YVEX_ERR_BOUNDS,
                             "daemon lock path exceeds its bound");
    server->options.socket_path = server->socket_path;
    return YVEX_OK;
}

static int server_synchronization_open(yvex_server *server, yvex_error *err)
{
    if (pthread_mutex_init(&server->state_mutex, NULL) != 0)
        return server_refuse(err, YVEX_ERR_STATE,
                             "host state mutex initialization failed");
    server->state_mutex_ready = 1;
    if (pthread_mutex_init(&server->clients_mutex, NULL) != 0)
        return server_refuse(err, YVEX_ERR_STATE,
                             "client mutex initialization failed");
    server->clients_mutex_ready = 1;
    if (pthread_cond_init(&server->clients_condition, NULL) != 0)
        return server_refuse(err, YVEX_ERR_STATE,
                             "client condition initialization failed");
    server->clients_condition_ready = 1;
    return YVEX_OK;
}

int yvex_server_create(yvex_server **out, const yvex_server_options *options,
                       yvex_error *err)
{
    yvex_server *server;
    const yvex_server_options *admitted;
    int rc;
    if (out) *out = NULL;
    if (!out)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "server output is required");
    server = calloc(1u, sizeof(*server));
    if (!server)
        return server_refuse(err, YVEX_ERR_NOMEM,
                             "runtime host allocation failed");
    server->listen_fd = -1;
    server->lock_fd = -1;
    atomic_init(&server->stopping, 0);
    rc = server_options_admit(server, options, err);
    admitted = &server->options;
    if (rc == YVEX_OK) rc = server_synchronization_open(server, err);
    if (rc == YVEX_OK) {
        server->client_capacity = SERVER_CLIENT_CAPACITY;
        server->clients = calloc((size_t)server->client_capacity,
                                 sizeof(*server->clients));
        if (!server->clients)
            rc = server_refuse(err, YVEX_ERR_NOMEM,
                               "host connection allocation failed");
    }
    if (rc == YVEX_OK)
        rc = yvex_server_telemetry_open(
            &server->telemetry, SERVER_TELEMETRY_CAPACITY, err);
    if (rc == YVEX_OK)
        rc = yvex_server_engine_manager_open(
            &server->engines, admitted->maximum_engines,
            admitted->request_queue_capacity, admitted->worker_count,
            model_work_execute, server, server->telemetry, err);
    if (rc == YVEX_OK)
        yvex_server_telemetry_queue(server->telemetry, 0u,
                                    0u);
    if (rc == YVEX_OK && admitted->openai_enabled) {
        server_openai_options openai = {
            .yvex_socket = server->socket_path,
            .port = admitted->openai_port,
            .timeout_ms = admitted->openai_timeout_ms,
            .maximum_connections = admitted->worker_count
        };
        rc = yvex_server_openai_prepare(&server->openai, &openai,
                                        server->telemetry, err);
    }
    if (rc != YVEX_OK) {
        yvex_server_close(&server);
        return rc;
    }
    memset(&server->summary, 0, sizeof(server->summary));
    server->summary.schema_version = YVEX_SERVER_SUMMARY_SCHEMA_V1;
    server->summary.status = YVEX_SERVER_STATUS_CONFIGURED;
    server->summary.request_queue_capacity = admitted->request_queue_capacity;
    server->summary.maximum_engines = admitted->maximum_engines;
    server->summary.worker_count = admitted->worker_count;
    server->summary.openai_timeout_ms = admitted->openai_timeout_ms;
    server->summary.trace_level = admitted->trace_level;
    server->summary.openai_listener_enabled = admitted->openai_enabled;
    server->summary.openai_port = admitted->openai_enabled
                                      ? admitted->openai_port : 0u;
    yvex_core_text_copy(server->summary.socket_path,
                        sizeof(server->summary.socket_path),
                        server->socket_path);
    (void)yvex_server_telemetry_emit(
        server->telemetry, NULL, YVEX_SERVER_EVENT_PROCESS_START,
        YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "process",
        (unsigned long long)getpid(), admitted->worker_count, 0u, 0.0, 0.0, err);
    (void)yvex_server_telemetry_emit(
        server->telemetry, NULL, YVEX_SERVER_EVENT_TELEMETRY_READY,
        YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "telemetry",
        SERVER_TELEMETRY_CAPACITY, 0u, 0u, 0.0, 0.0, err);
    *out = server;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int socket_directory_prepare(const char *socket_path, yvex_error *err)
{
    char directory[YVEX_SERVER_SOCKET_PATH_CAP];
    char *slash;
    struct stat info;
    yvex_core_text_copy(directory, sizeof(directory), socket_path);
    slash = strrchr(directory, '/');
    if (!slash || slash == directory)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "socket path requires one private parent directory");
    *slash = '\0';
    if (mkdir(directory, 0700) != 0 && errno != EEXIST)
        return server_refuse(err, YVEX_ERR_IO,
                             "cannot create local runtime directory");
    if (lstat(directory, &info) != 0 || !S_ISDIR(info.st_mode) ||
        info.st_uid != geteuid() || (info.st_mode & 0077u) != 0u)
        return server_refuse(err, YVEX_ERR_IO,
                             "local runtime directory is not private to this user");
    return YVEX_OK;
}
/*
 * Remove only an owner-validated stale socket and refuse live or foreign endpoints.
 *
 * Lock ownership is external.
 */
static int stale_socket_clear(const char *path, yvex_error *err)
{
    struct stat info;
    int fd;
    struct sockaddr_un address;
    if (lstat(path, &info) != 0)
        return errno == ENOENT ? YVEX_OK
                              : server_refuse(err, YVEX_ERR_IO,
                                              "socket path inspection failed");
    if (!S_ISSOCK(info.st_mode) || info.st_uid != geteuid())
        return server_refuse(err, YVEX_ERR_IO,
                             "existing socket path is not an owned socket");
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd >= 0) {
        memset(&address, 0, sizeof(address));
        address.sun_family = AF_UNIX;
        memcpy(address.sun_path, path, strlen(path) + 1u);
        if (connect(fd, (struct sockaddr *)&address, sizeof(address)) == 0) {
            (void)close(fd);
            return server_refuse(err, YVEX_ERR_STATE,
                                 "another YVEX server already owns the socket");
        }
        (void)close(fd);
    }
    if (unlink(path) != 0)
        return server_refuse(err, YVEX_ERR_IO,
                             "owned stale socket could not be removed");
    return YVEX_OK;
}
/*
 * Publish one private singleton Unix socket after acquiring its lock.
 *
 * Removes partial socket ownership and preserves the causal refusal.
 */
static int listener_open(yvex_server *server, yvex_error *err)
{
    struct sockaddr_un address;
    struct stat lock_info;
    char pending[YVEX_SERVER_SOCKET_PATH_CAP];
    int fd;
    if (socket_directory_prepare(server->socket_path, err) != YVEX_OK)
        return yvex_error_code(err);
    server->lock_fd = open(server->lock_path, O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
    if (server->lock_fd < 0 || fstat(server->lock_fd, &lock_info) != 0 ||
        !S_ISREG(lock_info.st_mode) || lock_info.st_uid != geteuid() ||
        flock(server->lock_fd, LOCK_EX | LOCK_NB) != 0)
        return server_refuse(err, YVEX_ERR_STATE,
                             "another daemon instance owns the runtime lock");
    server->lock_owned = 1;
    if (fchmod(server->lock_fd, 0600) != 0 ||
        ftruncate(server->lock_fd, 0) != 0 ||
        dprintf(server->lock_fd, "%llu\n",
                (unsigned long long)getpid()) < 0)
        return server_refuse(err, YVEX_ERR_IO,
                             "runtime lock publication failed");
    if (stale_socket_clear(server->socket_path, err) != YVEX_OK)
        return yvex_error_code(err);
    if (snprintf(pending, sizeof(pending), "%s.pending",
                 server->socket_path) < 0 ||
        strlen(pending) >= sizeof(address.sun_path))
        return server_refuse(err, YVEX_ERR_BOUNDS,
                             "atomic socket publication path exceeds its bound");
    if (stale_socket_clear(pending, err) != YVEX_OK)
        return yvex_error_code(err);
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return server_refuse(err, YVEX_ERR_IO,
                             "local listener socket creation failed");
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, pending, strlen(pending) + 1u);
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) != 0 ||
        chmod(pending, 0600) != 0 || listen(fd, 32) != 0 ||
        rename(pending, server->socket_path) != 0) {
        (void)close(fd);
        (void)unlink(pending);
        return server_refuse(err, YVEX_ERR_IO,
                             "local listener publication failed");
    }
    server->listen_fd = fd;
    return YVEX_OK;
}

static int work_emit(void *opaque, const yvex_client_message *message,
                     yvex_error *err)
{
    server_work_item *item = opaque;
    int rc = yvex_server_protocol_send(item->fd, message, err);
    if (rc == YVEX_OK) item->response_sent = 1;
    return rc;
}

static int protocol_error(int fd, const yvex_client_request *request,
                          int status, yvex_client_failure_class failure_class,
                          const char *reason, yvex_error *err)
{
    yvex_client_message message;
    memset(&message, 0, sizeof(message));
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_ERROR;
    message.status = status;
    message.stream_channel = YVEX_CLIENT_STREAM_ERROR;
    message.failure_class = failure_class != YVEX_CLIENT_FAILURE_NONE
                                ? failure_class
                                : yvex_server_failure_class_from_status(status);
    message.request_number = request ? request->request_number : 0u;
    if (request)
        yvex_core_text_copy(message.session_name, sizeof(message.session_name),
                            request->session_name);
    yvex_core_text_copy(message.reason, sizeof(message.reason),
                        reason ? reason : "request failed");
    return yvex_server_protocol_send(fd, &message, err);
}

static void model_work_execute(void *context, void *work)
{
    yvex_server *server = context;
    server_work_item *item = work;
    int rc;
    yvex_server_telemetry_request(server->telemetry, 1, 0, 0, 0);
    if (atomic_load_explicit(&server->stopping, memory_order_acquire)) {
        rc = server_refuse(&item->error, YVEX_ERR_CANCELLED,
                           "queued request cancelled by server shutdown");
    } else {
        unsigned long long started = server_monotonic_ns();
        double queue_seconds = started >= item->enqueued_ns
                                   ? (double)(started - item->enqueued_ns) /
                                         1000000000.0
                                   : 0.0;
        rc = yvex_server_engine_lease_execute(
            &item->engine, &item->request, item->request_id, queue_seconds,
            work_emit, item, &item->error);
    }
    if (rc != YVEX_OK && !item->response_sent) {
        yvex_error send_error;
        item->failure_class = yvex_server_failure_class_from_status(rc);
        if (protocol_error(item->fd, &item->request, rc,
                           item->failure_class,
                           yvex_error_message(&item->error),
                           &send_error) == YVEX_OK)
            item->response_sent = 1;
    }
    yvex_server_telemetry_request(server->telemetry, -1, rc == YVEX_OK,
                             rc != YVEX_OK && rc != YVEX_ERR_CANCELLED,
                             rc == YVEX_ERR_CANCELLED);
    yvex_server_engine_manager_release(server->engines, &item->engine);
    (void)pthread_mutex_lock(&item->mutex);
    item->status = rc;
    item->done = 1;
    (void)pthread_cond_broadcast(&item->condition);
    (void)pthread_mutex_unlock(&item->mutex);
}

int yvex_server_engine_load(
    yvex_server *server, const yvex_server_engine_options *options,
    yvex_server_engine_summary *summary, yvex_error *err)
{
    if (!server || !options || !summary || !server->engines ||
        server->summary.status != YVEX_SERVER_STATUS_READY)
        return server_refuse(err, YVEX_ERR_STATE,
                             "ready host and text-engine options are required");
    if (options->schema_version != YVEX_SERVER_ENGINE_SCHEMA_CURRENT)
        return server_refuse(err, YVEX_ERR_UNSUPPORTED,
                             "server engine options schema is unsupported");
    if (options->engine_kind != YVEX_SERVER_ENGINE_TEXT)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "media engine requires its composite profile");
    return yvex_server_engine_manager_load(
        server->engines, options, NULL, summary, err);
}

int yvex_server_media_engine_load(
    yvex_server *server, const yvex_server_engine_options *options,
    const yvex_server_media_options *media,
    yvex_server_engine_summary *summary, yvex_error *err)
{
    if (!server || !options || !media || !summary || !server->engines ||
        server->summary.status != YVEX_SERVER_STATUS_READY)
        return server_refuse(err, YVEX_ERR_STATE,
                             "ready host and media-engine profile are required");
    if (options->schema_version != YVEX_SERVER_ENGINE_SCHEMA_CURRENT)
        return server_refuse(err, YVEX_ERR_UNSUPPORTED,
                             "server engine options schema is unsupported");
    if (options->engine_kind != YVEX_SERVER_ENGINE_MEDIA)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "media engine requires its composite profile");
    return yvex_server_engine_manager_load(
        server->engines, options, media, summary, err);
}

int yvex_server_engine_unload(
    yvex_server *server, const char *alias, unsigned long long generation,
    yvex_server_engine_summary *summary, yvex_error *err)
{
    if (!server || !server->engines ||
        server->summary.status != YVEX_SERVER_STATUS_READY)
        return server_refuse(err, YVEX_ERR_STATE,
                             "ready host is required for engine unload");
    return yvex_server_engine_manager_unload(
        server->engines, alias, generation, summary, err);
}

int yvex_server_engine_snapshot(
    const yvex_server *server, yvex_server_engine_summary *engines,
    unsigned long long capacity, unsigned long long *count, yvex_error *err)
{
    if (!server || !server->engines)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "open host is required for engine snapshot");
    return yvex_server_engine_manager_snapshot(
        server->engines, engines, capacity, count, err);
}

/*
 * Publish the persistent host before any model is loaded.
 */
int yvex_server_start(yvex_server *server, yvex_error *err)
{
    int rc;
    if (!server || !server->state_mutex_ready ||
        (rc = socket_directory_prepare(server->socket_path, err)) != YVEX_OK ||
        pthread_mutex_lock(&server->state_mutex) != 0)
        return rc != YVEX_OK ? rc : server_refuse(
            err, YVEX_ERR_INVALID_ARG, "configured host is required");
    if (server->summary.status != YVEX_SERVER_STATUS_CONFIGURED) {
        (void)pthread_mutex_unlock(&server->state_mutex);
        return server_refuse(err, YVEX_ERR_STATE,
                             "host has already started or failed");
    }
    server->summary.status = YVEX_SERVER_STATUS_STARTING;
    (void)pthread_mutex_unlock(&server->state_mutex);
    rc = listener_open(server, err);
    if (rc == YVEX_OK)
        rc = yvex_server_telemetry_emit(
            server->telemetry, NULL, YVEX_SERVER_EVENT_LISTENER_READY,
            YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "listener",
            0600u, server->options.request_queue_capacity,
            server->options.maximum_engines, 0.0, 0.0, err);
    if (rc == YVEX_OK && server->openai)
        rc = yvex_server_openai_start(server->openai, err);
    if (pthread_mutex_lock(&server->state_mutex) == 0) {
        server->summary.status = rc == YVEX_OK ? YVEX_SERVER_STATUS_READY
                                              : YVEX_SERVER_STATUS_FAILED;
        server->summary.host_ready = rc == YVEX_OK;
        (void)pthread_mutex_unlock(&server->state_mutex);
    }
    if (rc != YVEX_OK) return rc;
    if (server->openai) yvex_server_openai_activate(server->openai);
    rc = yvex_server_telemetry_emit(
        server->telemetry, NULL, YVEX_SERVER_EVENT_RUNTIME_READY,
        YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "host",
        0u, server->options.maximum_engines, 0u, 0.0, 0.0, err);
    return rc;
}

static int request_enqueue(yvex_server *server, server_work_item *item,
                           yvex_error *err)
{
    const char *serialization_scope = item->request.session_name;
    unsigned long long depth = 0ull;
    int rc;
    /* Engine-scoped catalog work has no sequence identity. Keep it out of a
     * fabricated session while retaining one deterministic queue lane. */
    if (item->request.operation == YVEX_CLIENT_OP_SESSION_LIST)
        serialization_scope = "@engine.sessions";
    rc = yvex_server_engine_manager_acquire(
        server->engines, item->request.model_alias,
        item->request.engine_generation, &item->engine,
        &item->engine_summary, err);
    if (rc != YVEX_OK) {
        item->failure_class = YVEX_CLIENT_FAILURE_MODEL_NOT_FOUND;
        return rc;
    }
    server_event_scope_from_engine(&item->event_scope, &item->engine_summary);
    if (pthread_mutex_lock(&server->state_mutex) != 0) {
        rc = server_refuse(err, YVEX_ERR_STATE,
                           "request identity lock failed");
        goto failed;
    }
    server->next_request_id++;
    (void)snprintf(item->request_id, sizeof(item->request_id), "r%llu",
                   server->next_request_id);
    (void)pthread_mutex_unlock(&server->state_mutex);
    if (yvex_server_telemetry_emit_provider(
            server->telemetry, &item->event_scope,
            YVEX_SERVER_EVENT_REQUEST_RECEIVED,
            YVEX_SERVER_SEVERITY_INFO, item->request.session_name,
            item->request_id, NULL, "queue", item->request.prompt_bytes,
            0u, 0u, 0.0, 0.0, NULL, item->request.provider_request, NULL,
            err) != YVEX_OK)
        goto failed;
    item->enqueued_ns = server_monotonic_ns();
    rc = yvex_server_engine_lease_submit(
        &item->engine, item, serialization_scope, &depth, err);
    if (rc != YVEX_OK) {
        item->failure_class = rc == YVEX_ERR_BOUNDS
                                  ? YVEX_CLIENT_FAILURE_QUEUE_FULL
                                  : YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE;
        goto failed;
    }
    (void)yvex_server_telemetry_emit_provider(
        server->telemetry, &item->event_scope,
        YVEX_SERVER_EVENT_REQUEST_QUEUED,
        YVEX_SERVER_SEVERITY_INFO, item->request.session_name,
        item->request_id, NULL, "queue", depth,
        server->options.request_queue_capacity,
        0ull, 0.0, 0.0,
        NULL, item->request.provider_request, NULL, err);
    return YVEX_OK;
failed:
    yvex_server_engine_manager_release(server->engines, &item->engine);
    return rc != YVEX_OK ? rc : yvex_error_code(err);
}

static int status_message(yvex_server *server,
                          const yvex_client_request *request,
                          yvex_client_message *message, yvex_error *err)
{
    memset(message, 0, sizeof(*message));
    message->schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message->kind = YVEX_CLIENT_MESSAGE_STATUS;
    message->status = YVEX_OK;
    message->request_number = request->request_number;
    return yvex_server_get_summary(server, &message->runtime, err);
}

static int console_status_message(yvex_server *server,
                                  const yvex_client_request *request,
                                  yvex_client_message *message,
                                  yvex_error *err)
{
    yvex_server_summary summary;
    yvex_server_engine_summary engine;
    server_engine_lease lease = {0};
    int rc;
    memset(message, 0, sizeof(*message));
    rc = yvex_server_get_summary(server, &summary, err);
    if (rc != YVEX_OK) return rc;
    message->schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message->kind = YVEX_CLIENT_MESSAGE_CONSOLE_STATUS;
    message->status = YVEX_OK;
    message->request_number = request->request_number;
    message->runtime = summary;
    rc = yvex_server_engine_manager_acquire(
        server->engines, request->model_alias, request->engine_generation,
        &lease, &engine, err);
    if (rc != YVEX_OK) return rc;
    message->console.schema_version = YVEX_CONSOLE_STATUS_SCHEMA_V1;
    message->console.runtime_ready = summary.host_ready && engine.execution_ready;
    message->console.backend = engine.backend;
    message->console.context_capacity = engine.context_capacity;
    message->console.engine_generation = engine.generation;
    message->engine_kind = engine.engine_kind;
    message->execution_strategy = engine.execution_strategy;
    message->console.selected_model_available = 0;
    message->console.explicit_reasoning_channel_supported =
        engine.explicit_reasoning_channel_supported;
    yvex_core_text_copy(message->console.model_alias,
                        sizeof(message->console.model_alias), engine.alias);
    yvex_core_text_copy(message->console.live_model_identity,
                        sizeof(message->console.live_model_identity),
                        engine.runtime_model_identity);
    yvex_core_text_copy(message->console.physical_variant_identity,
                        sizeof(message->console.physical_variant_identity),
                        engine.specialization_identity);
    rc = yvex_server_engine_lease_console_status(
        &lease, request->session_name, &message->console,
        &message->partial_turn, err);
    yvex_server_engine_manager_release(server->engines, &lease);
    return rc;
}

static int event_subscription(yvex_server *server, int fd,
                              const yvex_client_request *request,
                              yvex_error *err)
{
    unsigned long long cursor = request->event_after_sequence;
    int rc = YVEX_OK;
    while (rc == YVEX_OK) {
        yvex_client_message message;
        memset(&message, 0, sizeof(message));
        message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
        message.kind = YVEX_CLIENT_MESSAGE_EVENT;
        message.status = YVEX_OK;
        message.request_number = request->request_number;
        rc = yvex_server_telemetry_next(server->telemetry, cursor, 1,
                                   &message.event, err);
        if (rc == YVEX_OK) {
            cursor = message.event.sequence;
            if (request->trace_level < YVEX_SERVER_TRACE_TOKENS &&
                (message.event.kind == YVEX_SERVER_EVENT_GENERATION_FRAGMENT ||
                 message.event.kind == YVEX_SERVER_EVENT_GENERATION_PROGRESS ||
                 message.event.kind == YVEX_SERVER_EVENT_PREFILL_PROGRESS))
                continue;
            if (request->trace_level == YVEX_SERVER_TRACE_SUMMARY &&
                message.event.severity < YVEX_SERVER_SEVERITY_WARNING &&
                message.event.kind != YVEX_SERVER_EVENT_RUNTIME_READY &&
                message.event.kind != YVEX_SERVER_EVENT_GENERATION_COMPLETED &&
                message.event.kind != YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_START &&
                message.event.kind != YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE)
                continue;
            rc = yvex_server_protocol_send(fd, &message, err);
            if (rc == YVEX_OK &&
                message.event.kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE)
                break;
        }
    }
    return rc;
}

static int client_wait_work(server_work_item *item, int fd, yvex_error *err)
{
    int cancel_sent = 0;
    if (pthread_mutex_lock(&item->mutex) != 0)
        return server_refuse(err, YVEX_ERR_STATE,
                             "request completion gate lock failed");
    while (!item->done) {
        struct timespec deadline;
        unsigned char byte;
        ssize_t peeked;
        (void)clock_gettime(CLOCK_REALTIME, &deadline);
        deadline.tv_nsec += 100000000L;
        if (deadline.tv_nsec >= 1000000000L) {
            deadline.tv_sec++;
            deadline.tv_nsec -= 1000000000L;
        }
        (void)pthread_cond_timedwait(&item->condition, &item->mutex,
                                     &deadline);
        if (item->done || cancel_sent) continue;
        peeked = recv(fd, &byte, sizeof(byte), MSG_PEEK | MSG_DONTWAIT);
        if (peeked == 0 ||
            (peeked < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
             errno != EINTR)) {
            yvex_error cancel_error;
            (void)pthread_mutex_unlock(&item->mutex);
            int cancel_rc = yvex_server_engine_lease_cancel(
                &item->engine, item->request.session_name, &cancel_error);
            if (cancel_rc == YVEX_OK)
                cancel_sent = 1;
            (void)pthread_mutex_lock(&item->mutex);
        }
    }
    (void)pthread_mutex_unlock(&item->mutex);
    if (item->status != YVEX_OK) *err = item->error;
    return item->status;
}

static int engine_message_send(int fd, unsigned long long request_number,
                               const yvex_server_engine_summary *engine,
                               yvex_error *err)
{
    yvex_client_message message;
    memset(&message, 0, sizeof(message));
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_ENGINE;
    message.status = YVEX_OK;
    message.request_number = request_number;
    message.engine = *engine;
    return yvex_server_protocol_send(fd, &message, err);
}

static int engine_list_send(yvex_server *server, int fd,
                            const yvex_client_request *request,
                            yvex_error *err)
{
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    yvex_client_message complete;
    unsigned long long count = 0ull, index;
    int rc = yvex_server_engine_snapshot(
        server, engines, YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES, &count, err);
    for (index = 0ull; index < count && rc == YVEX_OK; ++index)
        rc = engine_message_send(fd, request->request_number,
                                 &engines[index], err);
    if (rc != YVEX_OK) return rc;
    memset(&complete, 0, sizeof(complete));
    complete.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    complete.kind = YVEX_CLIENT_MESSAGE_ACK;
    complete.status = YVEX_OK;
    complete.request_number = request->request_number;
    return yvex_server_protocol_send(fd, &complete, err);
}

static int engine_load_control(yvex_server *server, int fd,
                               const yvex_client_request *request,
                               yvex_error *err)
{
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    unsigned long long count = 0ull, index;
    int rc;
    if (!request->model_alias[0] || !server->options.model_loader)
        return server_refuse(err, YVEX_ERR_UNSUPPORTED,
                             "host has no registry-backed model loader");
    rc = server->options.model_loader(
        server->options.model_loader_context, server,
        request->model_alias, err);
    if (rc == YVEX_OK)
        rc = yvex_server_engine_snapshot(
            server, engines, YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES, &count, err);
    for (index = 0ull; index < count && rc == YVEX_OK; ++index)
        if (!strcmp(engines[index].alias, request->model_alias))
            return engine_message_send(fd, request->request_number,
                                       &engines[index], err);
    return rc != YVEX_OK
               ? rc
               : server_refuse(err, YVEX_ERR_STATE,
                               "model loader did not publish the requested engine");
}

static void *client_main(void *opaque)
{
    server_client_slot *slot = opaque;
    yvex_server *server = slot->server;
    int fd = slot->fd, done = 0;
    while (!done &&
           !atomic_load_explicit(&server->stopping, memory_order_acquire)) {
        yvex_client_request request;
        unsigned char *prompt = NULL;
        yvex_provider_request *provider = NULL;
        yvex_error err;
        int response_sent = 0;
        yvex_client_failure_class failure_class = YVEX_CLIENT_FAILURE_NONE;
        int rc = yvex_server_protocol_receive(
            fd, &request, &prompt, &provider, &err);
        if (rc != YVEX_OK) break;
        if (request.operation == YVEX_CLIENT_OP_RUNTIME_STATUS) {
            yvex_client_message message;
            rc = status_message(server, &request, &message, &err);
            if (rc == YVEX_OK)
                rc = yvex_server_protocol_send(fd, &message, &err);
        } else if (request.operation == YVEX_CLIENT_OP_ENGINE_LIST) {
            rc = engine_list_send(server, fd, &request, &err);
        } else if (request.operation == YVEX_CLIENT_OP_ENGINE_LOAD) {
            rc = engine_load_control(server, fd, &request, &err);
        } else if (request.operation == YVEX_CLIENT_OP_ENGINE_UNLOAD) {
            yvex_server_engine_summary engine;
            rc = yvex_server_engine_unload(
                server, request.model_alias, request.engine_generation,
                &engine, &err);
            if (rc == YVEX_OK)
                rc = engine_message_send(fd, request.request_number,
                                         &engine, &err);
        } else if (request.operation == YVEX_CLIENT_OP_CONSOLE_STATUS) {
            yvex_client_message message;
            rc = console_status_message(server, &request, &message, &err);
            if (rc == YVEX_OK)
                rc = yvex_server_protocol_send(fd, &message, &err);
        } else if (request.operation == YVEX_CLIENT_OP_RUNTIME_WATCH ||
                   request.operation == YVEX_CLIENT_OP_RUNTIME_TRACE) {
            (void)pthread_mutex_lock(&server->clients_mutex);
            slot->subscriber = 1;
            (void)pthread_mutex_unlock(&server->clients_mutex);
            rc = event_subscription(server, fd, &request, &err);
            done = 1;
        } else if (request.operation == YVEX_CLIENT_OP_GENERATION_CANCEL) {
            server_engine_lease lease = {0};
            yvex_server_engine_summary engine;
            rc = yvex_server_engine_manager_acquire(
                server->engines, request.model_alias,
                request.engine_generation, &lease, &engine, &err);
            if (rc == YVEX_OK)
                rc = yvex_server_engine_lease_cancel(
                    &lease, request.session_name, &err);
            yvex_server_engine_manager_release(server->engines, &lease);
            if (rc == YVEX_OK) {
                yvex_client_message message;
                memset(&message, 0, sizeof(message));
                message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
                message.kind = YVEX_CLIENT_MESSAGE_ACK;
                message.status = YVEX_OK;
                message.request_number = request.request_number;
                rc = yvex_server_protocol_send(fd, &message, &err);
            }
        } else if (request.operation == YVEX_CLIENT_OP_RUNTIME_STOP) {
            yvex_client_message message;
            memset(&message, 0, sizeof(message));
            message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
            message.kind = YVEX_CLIENT_MESSAGE_ACK;
            message.status = YVEX_OK;
            message.request_number = request.request_number;
            rc = yvex_server_protocol_send(fd, &message, &err);
            if (rc == YVEX_OK) (void)yvex_server_stop(server, &err);
            done = 1;
        } else if (request.operation == YVEX_CLIENT_OP_HANDSHAKE) {
            yvex_client_message message;
            memset(&message, 0, sizeof(message));
            message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
            message.kind = YVEX_CLIENT_MESSAGE_ACK;
            message.status = YVEX_OK;
            message.request_number = request.request_number;
            (void)snprintf(message.reason, sizeof(message.reason), "protocol-v%u",
                           YVEX_LOCAL_PROTOCOL_VERSION);
            rc = yvex_server_protocol_send(fd, &message, &err);
        } else {
            server_work_item item;
            memset(&item, 0, sizeof(item));
            item.request = request;
            item.prompt = prompt;
            item.provider = provider;
            item.request.prompt = prompt;
            item.request.provider_request = provider;
            item.fd = fd;
            prompt = NULL;
            provider = NULL;
            {
                int mutex_ready = 0, condition_ready = 0;
                if (pthread_mutex_init(&item.mutex, NULL) == 0)
                    mutex_ready = 1;
                if (mutex_ready &&
                    pthread_cond_init(&item.condition, NULL) == 0)
                    condition_ready = 1;
                if (!mutex_ready || !condition_ready) {
                rc = server_refuse(&err, YVEX_ERR_STATE,
                                   "request completion gate initialization failed");
                } else {
                    rc = request_enqueue(server, &item, &err);
                    if (rc == YVEX_OK)
                        rc = client_wait_work(&item, fd, &err);
                }
                if (condition_ready) (void)pthread_cond_destroy(&item.condition);
                if (mutex_ready) (void)pthread_mutex_destroy(&item.mutex);
            }
            free(item.prompt);
            yvex_provider_request_close(&item.provider);
            response_sent = item.response_sent;
            failure_class = item.failure_class;
        }
        if (rc != YVEX_OK && !done && !response_sent) {
            yvex_error send_error;
            (void)protocol_error(fd, &request, rc, failure_class,
                                 yvex_error_message(&err), &send_error);
        }
        free(prompt);
        yvex_provider_request_close(&provider);
    }
    (void)yvex_server_telemetry_emit(
        server->telemetry, NULL, YVEX_SERVER_EVENT_CLIENT_DISCONNECTED,
        YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "transport",
        0u, 0u, 0u, 0.0, 0.0, NULL);
    (void)close(fd);
    (void)pthread_mutex_lock(&server->clients_mutex);
    slot->fd = -1;
    slot->active = 0;
    slot->subscriber = 0;
    if (server->active_clients) server->active_clients--;
    (void)pthread_cond_broadcast(&server->clients_condition);
    (void)pthread_mutex_unlock(&server->clients_mutex);
    return NULL;
}
/*
 * Reserve one bounded connection slot and launch a detached transport thread.
 *
 * Records ownership and creates one thread.
 */
static int client_start(yvex_server *server, int fd, yvex_error *err)
{
    unsigned long long index;
    pthread_t thread;
    struct timeval timeout;
    if (pthread_mutex_lock(&server->clients_mutex) != 0)
        return server_refuse(err, YVEX_ERR_STATE,
                             "client registry lock failed");
    for (index = 0u; index < server->client_capacity; ++index)
        if (!server->clients[index].active) break;
    if (index == server->client_capacity) {
        (void)pthread_mutex_unlock(&server->clients_mutex);
        return server_refuse(err, YVEX_ERR_BOUNDS,
                             "client connection capacity is exhausted");
    }
    server->clients[index].server = server;
    server->clients[index].fd = fd;
    server->clients[index].active = 1;
    server->clients[index].subscriber = 0;
    server->active_clients++;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    if (pthread_create(&thread, NULL, client_main, &server->clients[index]) != 0) {
        server->clients[index].active = 0;
        server->active_clients--;
        (void)pthread_mutex_unlock(&server->clients_mutex);
        return server_refuse(err, YVEX_ERR_STATE,
                             "client thread creation failed");
    }
    (void)pthread_detach(thread);
    (void)pthread_mutex_unlock(&server->clients_mutex);
    return YVEX_OK;
}

/*
 * Accept local connections until graceful stop while model work stays on the worker.
 *
 * May start the host and create bounded client threads. Returns listener/start refusal and leaves
 * ownership for close.
 */
int yvex_server_serve(yvex_server *server, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!server)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "runtime host is required");
    if (server->summary.status == YVEX_SERVER_STATUS_CONFIGURED) {
        rc = yvex_server_start(server, err);
        if (rc != YVEX_OK) return rc;
    }
    while (!atomic_load_explicit(&server->stopping, memory_order_acquire)) {
        int fd = accept(server->listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (atomic_load_explicit(&server->stopping, memory_order_acquire) ||
                errno == EBADF || errno == EINVAL)
                break;
            rc = server_refuse(err, YVEX_ERR_IO,
                               "local listener accept failed");
            break;
        }
        if (peer_validate(fd, err) != YVEX_OK) {
            (void)close(fd);
            continue;
        }
        rc = client_start(server, fd, err);
        if (rc != YVEX_OK) {
            (void)close(fd);
            if (rc != YVEX_ERR_BOUNDS) break;
            rc = YVEX_OK;
        }
    }
    return rc;
}

int yvex_server_stop(yvex_server *server, yvex_error *err)
{
    yvex_error openai_error = {0};
    server_request_queue_summary request_queue = {0};
    int rc = YVEX_OK;
    if (!server || !server->state_mutex_ready ||
        pthread_mutex_lock(&server->state_mutex) != 0)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "runtime host is required");
    if (atomic_load_explicit(&server->stopping, memory_order_acquire) ||
        server->summary.status == YVEX_SERVER_STATUS_STOPPING ||
        server->summary.status == YVEX_SERVER_STATUS_STOPPED) {
        (void)pthread_mutex_unlock(&server->state_mutex);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    server->summary.status = YVEX_SERVER_STATUS_STOPPING;
    server->summary.openai_listener_ready = 0;
    (void)pthread_mutex_unlock(&server->state_mutex);
    if (server->openai) {
        yvex_server_openai_request_stop(server->openai);
        rc = yvex_server_openai_finish(server->openai, &openai_error);
    }
    atomic_store_explicit(&server->stopping, 1, memory_order_release);
    (void)yvex_server_engine_manager_request_queue_snapshot(
        server->engines, &request_queue, NULL);
    (void)yvex_server_telemetry_emit(
        server->telemetry, NULL, YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_START,
        YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "shutdown",
        request_queue.queued, server->active_clients, request_queue.active,
        0.0, 0.0, err);
    if (server->listen_fd >= 0) {
        (void)shutdown(server->listen_fd, SHUT_RDWR);
        (void)close(server->listen_fd);
        server->listen_fd = -1;
    }
    yvex_server_engine_manager_cancel_all(server->engines);
    if (rc != YVEX_OK) {
        if (err) *err = openai_error;
        return rc;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Drain graph work and close sessions/model while telemetry remains readable.
 *
 * Joins request-queue workers, closes runtime ownership, emits shutdown.complete, and leaves
 * protocol/telemetry storage alive for final subscribers. Concurrent finish refuses and cleanup
 * preserves the first causal error.
 */
int yvex_server_finish(yvex_server *server, yvex_error *err)
{
    yvex_error primary = {0}, cleanup = {0};
    yvex_server_metrics metrics = {0};
    int rc, cleanup_rc;
    if (!server || !server->state_mutex_ready ||
        pthread_mutex_lock(&server->state_mutex) != 0)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "runtime host is required");
    if (server->finish_completed) {
        (void)pthread_mutex_unlock(&server->state_mutex);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (server->finish_started) {
        (void)pthread_mutex_unlock(&server->state_mutex);
        return server_refuse(err, YVEX_ERR_STATE,
                             "runtime host shutdown is already being finalized");
    }
    server->finish_started = 1;
    (void)pthread_mutex_unlock(&server->state_mutex);
    rc = yvex_server_stop(server, &primary);
    cleanup_rc = yvex_server_engine_manager_close(&server->engines, &cleanup);
    if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
        rc = cleanup_rc;
        primary = cleanup;
    }
    (void)yvex_server_telemetry_metrics_copy(server->telemetry, &metrics,
                                             &cleanup);
    cleanup_rc = yvex_server_telemetry_emit(
        server->telemetry, NULL, YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE,
        rc == YVEX_OK ? YVEX_SERVER_SEVERITY_INFO : YVEX_SERVER_SEVERITY_ERROR,
        NULL, NULL, NULL, "shutdown", metrics.model_close_count,
        metrics.model_open_count, metrics.active_sessions, 0.0, 0.0,
        &cleanup);
    if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
        rc = cleanup_rc;
        primary = cleanup;
    }
    if (pthread_mutex_lock(&server->state_mutex) == 0) {
        server->summary.status = YVEX_SERVER_STATUS_STOPPED;
        server->finish_completed = 1;
        (void)pthread_mutex_unlock(&server->state_mutex);
    } else if (rc == YVEX_OK) {
        rc = server_refuse(&primary, YVEX_ERR_STATE,
                           "runtime final state publication failed");
    }
    if (err) *err = primary;
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

/*
 * Copy one authoritative host and metrics snapshot without exposing owners.
 *
 * Refuses absent ownership.
 */
int yvex_server_get_summary(const yvex_server *server,
                            yvex_server_summary *out, yvex_error *err)
{
    yvex_server *mutable = (yvex_server *)server;
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    server_request_queue_summary request_queue = {0};
    unsigned long long count = 0u, index;
    if (!server || !out || pthread_mutex_lock(&mutable->state_mutex) != 0)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "host and summary output are required");
    *out = server->summary;
    (void)pthread_mutex_unlock(&mutable->state_mutex);
    if (server->engines &&
        yvex_server_engine_manager_snapshot(
            server->engines, engines, YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES,
            &count, err) !=
            YVEX_OK)
        return yvex_error_code(err);
    if (server->engines &&
        yvex_server_engine_manager_request_queue_snapshot(
            server->engines, &request_queue, err) != YVEX_OK)
        return yvex_error_code(err);
    if (server->telemetry) {
        yvex_server_telemetry_queue(server->telemetry, request_queue.queued,
                                    request_queue.capacity);
        (void)yvex_server_telemetry_metrics_copy(server->telemetry,
                                                 &out->metrics, err);
    }
    out->engine_count = count;
    out->loaded_engine_count = 0ull;
    out->draining_engine_count = 0ull;
    out->session_count = 0ull;
    out->metrics.mapped_artifact_bytes = 0ull;
    out->metrics.resident_host_bytes = 0ull;
    out->metrics.resident_device_bytes = 0ull;
    for (index = 0ull; index < count; ++index) {
        if (!yvex_core_u64_add(out->metrics.mapped_artifact_bytes,
                               engines[index].mapped_package_bytes,
                               &out->metrics.mapped_artifact_bytes) ||
            !yvex_core_u64_add(out->metrics.resident_host_bytes,
                               engines[index].resident_host_bytes,
                               &out->metrics.resident_host_bytes) ||
            !yvex_core_u64_add(out->metrics.resident_device_bytes,
                               engines[index].resident_device_bytes,
                               &out->metrics.resident_device_bytes))
            return server_refuse(err, YVEX_ERR_BOUNDS,
                                 "host engine resource total overflowed");
        out->session_count += engines[index].session_count;
        out->loaded_engine_count +=
            engines[index].state == YVEX_SERVER_ENGINE_LOADED;
        out->draining_engine_count +=
            engines[index].state == YVEX_SERVER_ENGINE_DRAINING ||
            engines[index].state == YVEX_SERVER_ENGINE_UNLOADING;
    }
    out->request_count = out->metrics.completed_requests +
                         out->metrics.failed_requests +
                         out->metrics.cancelled_requests;
    if (server->openai) {
        server_openai_snapshot openai = {0};
        yvex_server_openai_snapshot(server->openai, &openai);
        out->openai_listener_enabled = openai.enabled;
        out->openai_listener_ready = openai.ready;
        out->openai_port = openai.port;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_event_next(yvex_server *server,
                           unsigned long long after_sequence, int wait,
                           yvex_server_event *event, yvex_error *err)
{
    if (!server || !server->telemetry)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "open host telemetry is required");
    return yvex_server_telemetry_next(server->telemetry, after_sequence,
                                 wait, event, err);
}

void yvex_server_close(yvex_server **server)
{
    yvex_server *owner;
    yvex_error err;
    unsigned long long index;
    if (!server || !*server) return;
    owner = *server;
    (void)yvex_server_finish(owner, &err);
    if (owner->clients_mutex_ready &&
        pthread_mutex_lock(&owner->clients_mutex) == 0) {
        for (index = 0u; index < owner->client_capacity; ++index)
            if (owner->clients[index].active && !owner->clients[index].subscriber &&
                owner->clients[index].fd >= 0)
                (void)shutdown(owner->clients[index].fd, SHUT_RDWR);
        while (owner->active_clients && owner->clients_condition_ready) {
            int subscribers = 0;
            for (index = 0u; index < owner->client_capacity; ++index)
                if (owner->clients[index].active &&
                    owner->clients[index].subscriber)
                    subscribers = 1;
            if (!subscribers) break;
            (void)pthread_cond_wait(&owner->clients_condition,
                                    &owner->clients_mutex);
        }
        for (index = 0u; index < owner->client_capacity; ++index)
            if (owner->clients[index].active && owner->clients[index].fd >= 0)
                (void)shutdown(owner->clients[index].fd, SHUT_RDWR);
        while (owner->active_clients && owner->clients_condition_ready)
            (void)pthread_cond_wait(&owner->clients_condition,
                                    &owner->clients_mutex);
        (void)pthread_mutex_unlock(&owner->clients_mutex);
    }
    /* A contender can have the canonical path configured without ever acquiring
       publication ownership.  Its refusal cleanup must not unlink the live
       listener owned by the process that holds the singleton lock. */
    if (owner->lock_owned && owner->socket_path[0])
        (void)unlink(owner->socket_path);
    if (owner->lock_fd >= 0) (void)close(owner->lock_fd);
    if (owner->lock_owned && owner->lock_path[0])
        (void)unlink(owner->lock_path);
    yvex_server_openai_close(&owner->openai);
    yvex_server_telemetry_close(&owner->telemetry);
    if (owner->clients_condition_ready)
        (void)pthread_cond_destroy(&owner->clients_condition);
    if (owner->clients_mutex_ready)
        (void)pthread_mutex_destroy(&owner->clients_mutex);
    if (owner->state_mutex_ready)
        (void)pthread_mutex_destroy(&owner->state_mutex);
    free(owner->clients);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *server = NULL;
}
