/*
 * Keep immutable model resources resident while local clients submit typed work.
 *
 * One host opens at most one model. Its scheduler serializes per-session mutation while bounded
 * workers execute independent sessions over admitted runtime/session/protocol owners.
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
#define SERVER_SCHEMA_V1 1u
#define SERVER_TELEMETRY_CAPACITY 4096u
#define SERVER_CLIENT_CAPACITY 64u
typedef struct server_work_item {
    yvex_client_request request;
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
    char artifact_path[YVEX_PATH_CAP];
    char runtime_binding_path[YVEX_PATH_CAP];
    char target_id[128];
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    char lock_path[YVEX_SERVER_SOCKET_PATH_CAP];
    yvex_runtime_model *model;
    yvex_runtime_generation_context_summary capacity_summary;
    yvex_runtime_generation_context *warm_generation;
    yvex_runtime_execution_session *warm_session;
    server_telemetry *telemetry;
    server_session_registry *sessions;
    server_scheduler *scheduler;
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
static void scheduler_observe(void *context, unsigned long long queued,
                              unsigned long long capacity,
                              unsigned long long active);

static int server_refuse(yvex_error *err, yvex_status status,
                         const char *reason)
{
    yvex_error_set(err, status, "server.host", reason);
    return status;
}

static void server_report_model_refusal(
    int status, const yvex_runtime_model_failure *failure, yvex_error *err)
{
    char where[YVEX_ERROR_WHERE_CAP];
    char reason[YVEX_ERROR_MESSAGE_CAP];
    if (!err || !failure || failure->code == YVEX_RUNTIME_MODEL_FAILURE_NONE)
        return;
    yvex_core_text_copy(where, sizeof(where), yvex_error_where(err));
    yvex_core_text_copy(reason, sizeof(reason),
                        failure->reason ? failure->reason : yvex_error_message(err));
    if (failure->code == YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION)
        yvex_error_setf(
            err, (yvex_status)status, where,
            "model admission refused: field=%s required=%llu available=%llu reason=%s",
            failure->field, failure->expected, failure->actual, reason);
    else
        yvex_error_setf(
            err, (yvex_status)status, where,
            "model admission refused: failure=%u field=%s expected=%llu actual=%llu reason=%s",
            (unsigned int)failure->code, failure->field, failure->expected,
            failure->actual, reason);
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

static double server_elapsed_seconds(unsigned long long start,
                                     unsigned long long end)
{
    if (!start || end < start) return 0.0;
    return (double)(end - start) / 1000000000.0;
}

static int server_options_admit(yvex_server *server,
                                const yvex_server_options *options,
                                yvex_error *err)
{
    char canonical[YVEX_SERVER_SOCKET_PATH_CAP];
    if (!options || options->schema_version != YVEX_SERVER_OPTIONS_SCHEMA_V2 ||
        !options->artifact_path || !options->runtime_binding_path ||
        !options->target_id ||
        (options->backend != YVEX_BACKEND_KIND_CPU &&
         options->backend != YVEX_BACKEND_KIND_CUDA) ||
        options->generation_mode > YVEX_SERVER_GENERATION_DSPARK ||
        !options->context_capacity || !options->prefill_chunk_tokens ||
        !options->maximum_new_tokens || !options->maximum_output_bytes ||
        !options->maximum_sessions || !options->request_queue_capacity ||
        !options->concurrent_sequences ||
        options->concurrent_sequences > options->maximum_sessions ||
        options->maximum_sessions > SERVER_CLIENT_CAPACITY ||
        (options->openai_enabled &&
         (!options->openai_port || options->openai_timeout_ms < 100u ||
          options->openai_timeout_ms > 86400000u)))
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "complete bounded runtime-host options are required");
    server->options = *options;
    yvex_core_text_copy(server->artifact_path, sizeof(server->artifact_path),
                        options->artifact_path);
    yvex_core_text_copy(server->runtime_binding_path,
                        sizeof(server->runtime_binding_path),
                        options->runtime_binding_path);
    yvex_core_text_copy(server->target_id, sizeof(server->target_id),
                        options->target_id);
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
    server->options.artifact_path = server->artifact_path;
    server->options.runtime_binding_path = server->runtime_binding_path;
    server->options.target_id = server->target_id;
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
        rc = yvex_server_scheduler_open(
            &server->scheduler, options->request_queue_capacity,
            options->concurrent_sequences,
            model_work_execute, scheduler_observe, server, err);
    if (rc == YVEX_OK)
        rc = yvex_server_telemetry_open(&server->telemetry,
                                   SERVER_TELEMETRY_CAPACITY,
                                   options->generation_mode,
                                   NULL, NULL, NULL, err);
    if (rc == YVEX_OK)
        yvex_server_telemetry_queue(server->telemetry, 0u,
                                    options->request_queue_capacity);
    if (rc == YVEX_OK && options->openai_enabled) {
        server_openai_options openai = {
            .yvex_socket = server->socket_path,
            .port = options->openai_port,
            .timeout_ms = options->openai_timeout_ms
        };
        rc = yvex_server_openai_prepare(&server->openai, &openai,
                                        server->telemetry, err);
    }
    if (rc != YVEX_OK) {
        yvex_server_close(&server);
        return rc;
    }
    memset(&server->summary, 0, sizeof(server->summary));
    server->summary.schema_version = SERVER_SCHEMA_V1;
    server->summary.status = YVEX_SERVER_STATUS_CONFIGURED;
    server->summary.backend = options->backend;
    server->summary.context_capacity = options->context_capacity;
    server->summary.prefill_chunk_tokens = options->prefill_chunk_tokens;
    server->summary.generation_mode = options->generation_mode;
    server->summary.maximum_new_tokens = options->maximum_new_tokens;
    server->summary.maximum_output_bytes = options->maximum_output_bytes;
    server->summary.maximum_sessions = options->maximum_sessions;
    server->summary.request_queue_capacity = options->request_queue_capacity;
    server->summary.concurrent_sequences = options->concurrent_sequences;
    server->summary.openai_timeout_ms = options->openai_timeout_ms;
    server->summary.trace_level = options->trace_level;
    server->summary.explicit_reasoning_channel_supported = 0;
    server->summary.openai_listener_enabled = options->openai_enabled;
    server->summary.openai_port = options->openai_enabled
                                      ? options->openai_port : 0u;
    yvex_core_text_copy(server->summary.socket_path,
                        sizeof(server->summary.socket_path),
                        server->socket_path);
    yvex_core_text_copy(server->summary.target_id,
                        sizeof(server->summary.target_id),
                        server->target_id);
    (void)yvex_server_telemetry_emit(
        server->telemetry, YVEX_SERVER_EVENT_PROCESS_START,
        YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "process",
        (unsigned long long)getpid(), options->backend, 0u, 0.0, 0.0, err);
    (void)yvex_server_telemetry_emit(
        server->telemetry, YVEX_SERVER_EVENT_TELEMETRY_READY,
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
        info.st_uid != getuid() || (info.st_mode & 0077u) != 0u)
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
    if (!S_ISSOCK(info.st_mode) || info.st_uid != getuid())
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
        !S_ISREG(lock_info.st_mode) || lock_info.st_uid != getuid() ||
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

static void scheduler_observe(void *context, unsigned long long queued,
                              unsigned long long capacity,
                              unsigned long long active)
{
    yvex_server *server = context;
    (void)active;
    if (server && server->telemetry)
        yvex_server_telemetry_queue(server->telemetry, queued, capacity);
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
        rc = yvex_server_sessions_execute(
            server->sessions, &item->request, item->request_id, queue_seconds,
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
    (void)pthread_mutex_lock(&item->mutex);
    item->status = rc;
    item->done = 1;
    (void)pthread_cond_broadcast(&item->condition);
    (void)pthread_mutex_unlock(&item->mutex);
}

static void server_generation_options(
    const yvex_server *server, yvex_runtime_generation_options *options)
{
    memset(options, 0, sizeof(*options));
    options->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5;
    options->backend = server->options.backend;
    options->mode = server->options.generation_mode == YVEX_SERVER_GENERATION_DSPARK
                        ? YVEX_GENERATION_MODE_DSPARK
                        : YVEX_GENERATION_MODE_TARGET_ONLY;
    options->workload_kind = server->options.concurrent_sequences > 1ull
                                 ? YVEX_EXECUTION_WORKLOAD_BALANCED_SERVING
                                 : YVEX_EXECUTION_WORKLOAD_INTERACTIVE_LATENCY;
    options->context_capacity = server->options.context_capacity;
    options->prefill_chunk_tokens = server->options.prefill_chunk_tokens;
    options->maximum_new_tokens = server->options.maximum_new_tokens;
    options->maximum_output_bytes = server->options.maximum_output_bytes;
    options->maximum_host_bytes = server->options.maximum_host_bytes;
    options->maximum_device_bytes = server->options.maximum_device_bytes;
    options->concurrent_sequences = server->options.concurrent_sequences;
    options->trace_policy = server->options.trace_level == YVEX_SERVER_TRACE_FULL
                                ? YVEX_RUNTIME_TRACE_FULL
                                : YVEX_RUNTIME_TRACE_STAGES;
    options->evidence_profile = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    options->sampling_policy.schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    options->sampling_policy.strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC;
    options->sampling_policy.temperature = 1.0;
    options->sampling_policy.top_p = 1.0;
    options->sampling_policy.typical_p = 1.0;
    options->sampling_policy.seed_present = 1;
    options->sampling_policy.seed = server->options.sampling_seed;
}

static int server_execution_prepare(yvex_server *server,
                                    yvex_runtime_residency_summary *summary,
                                    yvex_error *err)
{
    const yvex_runtime_model_view *view = server && server->model
                                              ? yvex_runtime_model_view_get(server->model)
                                              : NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *generation = NULL;
    yvex_runtime_session_open_request request;
    yvex_runtime_generation_options options;
    yvex_runtime_model_failure failure;
    yvex_error primary, cleanup;
    int rc, generation_cleanup_rc, session_cleanup_rc;
    if (!server || !summary || !view || !view->residency)
        return server_refuse(err, YVEX_ERR_STATE,
                             "runtime residency is unavailable during startup");
    memset(&request, 0, sizeof(request));
    memset(&failure, 0, sizeof(failure));
    request.backend = server->options.backend;
    request.maximum_host_bytes = server->options.maximum_host_bytes;
    request.maximum_device_bytes = server->options.maximum_device_bytes;
    rc = yvex_runtime_session_open(&session, server->model, &request,
                                   &failure, err);
    server_generation_options(server, &options);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &generation, server->model, session, &options, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_summary_copy(
            generation, &server->capacity_summary, err);
    if (rc == YVEX_OK &&
        (server->capacity_summary.concurrent_sequences !=
             server->options.concurrent_sequences ||
         server->capacity_summary.continuous_batching))
        rc = server_refuse(
            err, YVEX_ERR_STATE,
            "startup capacity plan does not match independent-session scheduling");
    if (rc == YVEX_OK)
        rc = yvex_runtime_residency_snapshot(view->residency, summary,
                                             NULL, NULL, err);
    primary = err ? *err : (yvex_error){0};
    yvex_error_clear(&cleanup);
    generation_cleanup_rc = yvex_runtime_generation_context_close(
        &generation, &cleanup);
    if (generation_cleanup_rc != YVEX_OK) {
        server->warm_generation = generation;
        server->warm_session = session;
        if (err) *err = cleanup;
        return generation_cleanup_rc;
    }
    session_cleanup_rc = yvex_runtime_session_close(&session, &cleanup);
    if (session_cleanup_rc != YVEX_OK) {
        server->warm_session = session;
        if (err) *err = cleanup;
        return session_cleanup_rc;
    }
    if (rc != YVEX_OK) {
        if (err) *err = primary;
        return rc;
    }
    if (server->options.backend == YVEX_BACKEND_KIND_CUDA &&
        (!summary->cuda_ready ||
        (!summary->cuda_upload_count && !summary->cuda_host_registration_count &&
         !summary->cuda_managed_prefetch_count)))
        return server_refuse(err, YVEX_ERR_STATE,
                             "CUDA residency did not complete before readiness");
    if (err) *err = primary;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Open the model once, then sessions, scheduler workers, and listener before READY.
 * Failed starts retain acquired owners for deterministic close.
 */
int yvex_server_start(yvex_server *server, yvex_error *err)
{
    yvex_runtime_model_open_request request;
    yvex_runtime_generation_options startup_options;
    yvex_runtime_model_failure failure;
    yvex_runtime_model_summary model;
    yvex_runtime_residency_summary residency;
    yvex_paths paths;
    yvex_error path_error;
    const yvex_runtime_model_view *view;
    unsigned long long startup_started, cuda_started = 0u, startup_completed;
    double cuda_seconds = 0.0, startup_seconds;
    int rc = YVEX_OK;
    if (!server || !server->state_mutex_ready ||
        (rc = socket_directory_prepare(server->socket_path, err)) != YVEX_OK ||
        pthread_mutex_lock(&server->state_mutex) != 0)
        return rc != YVEX_OK ? rc : server_refuse(
            err, YVEX_ERR_INVALID_ARG, "configured host is required");
    if (server->summary.status != YVEX_SERVER_STATUS_CONFIGURED || server->model) {
        (void)pthread_mutex_unlock(&server->state_mutex);
        return server_refuse(err, YVEX_ERR_STATE,
                             "host has already started or failed");
    }
    server->summary.status = YVEX_SERVER_STATUS_STARTING;
    (void)pthread_mutex_unlock(&server->state_mutex);
    startup_started = server_monotonic_ns();
    (void)yvex_server_telemetry_emit(
        server->telemetry, YVEX_SERVER_EVENT_ARTIFACT_OPEN_START,
        YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "startup",
        0u, 0u, 0u, 0.0, 0.0, err);
    (void)yvex_server_telemetry_emit(
        server->telemetry, YVEX_SERVER_EVENT_MATERIALIZATION_START,
        YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "startup",
        0u, 0u, 0u, 0.0, 0.0, err);
    memset(&request, 0, sizeof(request));
    memset(&failure, 0, sizeof(failure));
    request.artifact_path = server->artifact_path;
    request.runtime_binding_path = server->runtime_binding_path;
    request.target_id = server->target_id;
    yvex_error_clear(&path_error);
    if (yvex_paths_default(&paths, &path_error) == YVEX_OK)
        request.artifact_reopen_cache_root = paths.cache_dir;
    server_generation_options(server, &startup_options);
    request.startup_generation = &startup_options;
    request.residency_backend = server->options.backend;
    request.maximum_host_bytes = server->options.maximum_host_bytes;
    request.maximum_device_bytes = server->options.maximum_device_bytes;
    rc = yvex_runtime_model_open(&server->model, &request, &failure, err);
    if (rc != YVEX_OK) server_report_model_refusal(rc, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_model_summary_copy(server->model, &model, err);
    view = rc == YVEX_OK ? yvex_runtime_model_view_get(server->model) : NULL;
    memset(&residency, 0, sizeof(residency));
    if (rc == YVEX_OK && (!view || !view->residency))
        rc = server_refuse(err, YVEX_ERR_STATE,
                           "runtime model did not publish immutable residency");
    if (rc == YVEX_OK) {
        const yvex_tokenizer_plan_summary *tokenizer =
            yvex_tokenizer_plan_summary_get(view->tokenizer);
        server->summary.explicit_reasoning_channel_supported =
            tokenizer && tokenizer->explicit_reasoning_supported;
        cuda_started = server_monotonic_ns();
        rc = server_execution_prepare(server, &residency, err);
        if (rc == YVEX_OK)
            cuda_seconds = server_elapsed_seconds(cuda_started,
                                                  server_monotonic_ns());
    }
    if (rc == YVEX_OK) {
        yvex_server_telemetry_identities(server->telemetry,
                                    model.runtime_model_identity,
                                    model.artifact_identity,
                                    view->binding->profile_identity);
        yvex_server_telemetry_model_opened(
            server->telemetry, model.artifact_bytes_hashed,
            residency.host_resident_bytes, residency.device_resident_bytes,
            residency.cuda_upload_count);
        rc = yvex_server_telemetry_emit(
            server->telemetry, YVEX_SERVER_EVENT_BINDING_ADMITTED,
            YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "startup",
            1u, 0u, 0u, 0.0, 0.0, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_server_telemetry_emit(
            server->telemetry, YVEX_SERVER_EVENT_MATERIALIZATION_COMPLETE,
            YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "startup",
            residency.host_resident_bytes, residency.device_resident_bytes,
            residency.binding_count,
            model.lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_MATERIALIZATION_OPEN],
            0.0, err);
    if (rc == YVEX_OK)
        rc = yvex_server_telemetry_emit(
            server->telemetry, YVEX_SERVER_EVENT_RESIDENCY_READY,
            YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "startup",
            residency.host_resident_bytes, residency.device_resident_bytes,
            residency.cuda_upload_count,
            model.lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_RESIDENCY] + cuda_seconds,
            0.0, err);
    if (rc == YVEX_OK) {
        startup_completed = server_monotonic_ns();
        startup_seconds = server_elapsed_seconds(startup_started,
                                                 startup_completed);
        yvex_runtime_identity_copy(server->summary.runtime_model_identity,
                                   model.runtime_model_identity);
        yvex_runtime_identity_copy(server->summary.runtime_binding_identity,
                                   model.runtime_binding_identity);
        yvex_runtime_identity_copy(server->summary.artifact_identity,
                                   model.artifact_identity);
        yvex_runtime_identity_copy(server->summary.physical_variant_identity,
                                   view->binding->profile_identity);
        yvex_runtime_identity_copy(server->summary.capacity_plan_identity,
                                   server->capacity_summary.capacity_plan_identity);
        server->summary.capacity_required_bytes =
            server->capacity_summary.capacity_required_bytes;
        server->summary.capacity_unreserved_bytes =
            server->capacity_summary.capacity_unreserved_bytes;
        rc = yvex_server_telemetry_emit(
            server->telemetry, YVEX_SERVER_EVENT_ARTIFACT_OPEN_COMPLETE,
            YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "startup",
            model.artifact_bytes_hashed, residency.host_resident_bytes,
            residency.device_resident_bytes, startup_seconds, 0.0, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_server_sessions_open(
            &server->sessions, server->model, &server->options,
            server->telemetry, err);
    if (rc == YVEX_OK)
        rc = yvex_server_scheduler_start(server->scheduler, err);
    if (rc == YVEX_OK) rc = listener_open(server, err);
    if (rc == YVEX_OK)
        rc = yvex_server_telemetry_emit(
            server->telemetry, YVEX_SERVER_EVENT_LISTENER_READY,
            YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "listener",
            0600u, server->options.request_queue_capacity,
            server->options.maximum_sessions,
            0.0, 0.0, err);
    if (rc == YVEX_OK && server->openai)
        rc = yvex_server_openai_start(server->openai, err);
    if (rc == YVEX_OK)
        rc = yvex_server_telemetry_emit(
            server->telemetry, YVEX_SERVER_EVENT_RUNTIME_READY,
            YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "runtime",
            1u, server->options.context_capacity, server->options.backend,
            server_elapsed_seconds(startup_started, server_monotonic_ns()),
            0.0, err);
    if (rc != YVEX_OK && server->openai) {
        yvex_error cleanup;
        yvex_server_openai_request_stop(server->openai);
        (void)yvex_server_openai_finish(server->openai, &cleanup);
    }
    (void)pthread_mutex_lock(&server->state_mutex);
    if (rc == YVEX_OK) {
        server_openai_snapshot openai = {0};
        yvex_server_openai_snapshot(server->openai, &openai);
        server->summary.status = YVEX_SERVER_STATUS_READY;
        server->summary.runtime_ready = 1;
        server->summary.generation_ready = 1;
        server->summary.public_server_ready = 0;
        server->summary.independent_session_scheduling_ready =
            server->options.concurrent_sequences > 1ull;
        server->summary.continuous_batching_ready = 0;
        server->summary.openai_listener_enabled = openai.enabled;
        server->summary.openai_listener_ready = openai.ready;
        server->summary.openai_port = openai.port;
    } else {
        server->summary.status = YVEX_SERVER_STATUS_FAILED;
    }
    (void)pthread_mutex_unlock(&server->state_mutex);
    if (rc == YVEX_OK && server->openai) {
        yvex_server_openai_activate(server->openai);
        (void)pthread_mutex_lock(&server->state_mutex);
        server->summary.openai_listener_ready = 1;
        (void)pthread_mutex_unlock(&server->state_mutex);
    }
    return rc;
}
/*
 * Enqueue one request pointer while its transport thread retains stack ownership.
 *
 * The session name is the serialization key. Global operations use the empty key and therefore
 * remain ordered with one another without preventing independent named sessions from executing.
 */
static int request_enqueue(yvex_server *server, server_work_item *item,
                           yvex_error *err)
{
    unsigned long long depth = 0ull;
    int rc;
    if (pthread_mutex_lock(&server->state_mutex) != 0)
        return server_refuse(err, YVEX_ERR_STATE,
                             "request identity lock failed");
    server->next_request_id++;
    (void)snprintf(item->request_id, sizeof(item->request_id), "r%llu",
                   server->next_request_id);
    (void)pthread_mutex_unlock(&server->state_mutex);
    if (yvex_server_telemetry_emit_provider(
            server->telemetry, YVEX_SERVER_EVENT_REQUEST_RECEIVED,
            YVEX_SERVER_SEVERITY_INFO, item->request.session_name,
            item->request_id, NULL, "queue", item->request.prompt_bytes,
            0u, 0u, 0.0, 0.0, NULL, item->request.provider_request, NULL,
            err) != YVEX_OK)
        return yvex_error_code(err);
    item->enqueued_ns = server_monotonic_ns();
    rc = yvex_server_scheduler_submit(
        server->scheduler, item, item->request.session_name, &depth, err);
    if (rc != YVEX_OK) {
        item->failure_class = rc == YVEX_ERR_BOUNDS
                                  ? YVEX_CLIENT_FAILURE_QUEUE_FULL
                                  : YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE;
        return rc;
    }
    (void)yvex_server_telemetry_emit_provider(
        server->telemetry, YVEX_SERVER_EVENT_REQUEST_QUEUED,
        YVEX_SERVER_SEVERITY_INFO, item->request.session_name,
        item->request_id, NULL, "queue", depth,
        server->options.request_queue_capacity, 0u, 0.0, 0.0,
        NULL, item->request.provider_request, NULL, err);
    return YVEX_OK;
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
    int rc;
    memset(message, 0, sizeof(*message));
    rc = yvex_server_get_summary(server, &summary, err);
    if (rc != YVEX_OK) return rc;
    message->schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message->kind = YVEX_CLIENT_MESSAGE_CONSOLE_STATUS;
    message->status = YVEX_OK;
    message->request_number = request->request_number;
    message->runtime = summary;
    message->console.schema_version = 1u;
    message->console.runtime_ready = summary.runtime_ready;
    message->console.backend = summary.backend;
    message->console.context_capacity = summary.context_capacity;
    message->console.selected_model_available = 0;
    message->console.explicit_reasoning_channel_supported =
        summary.explicit_reasoning_channel_supported;
    yvex_core_text_copy(message->console.live_model_identity,
                        sizeof(message->console.live_model_identity),
                        summary.runtime_model_identity);
    yvex_core_text_copy(message->console.physical_variant_identity,
                        sizeof(message->console.physical_variant_identity),
                        summary.physical_variant_identity);
    return yvex_server_sessions_console_status(server->sessions,
                                               request->session_name,
                                               &message->console,
                                               &message->partial_turn, err);
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

static int client_wait_work(yvex_server *server, server_work_item *item,
                            int fd, yvex_error *err)
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
            if (yvex_server_sessions_cancel(server->sessions,
                                            item->request.session_name,
                                            &cancel_error) == YVEX_OK)
                cancel_sent = 1;
            (void)pthread_mutex_lock(&item->mutex);
        }
    }
    (void)pthread_mutex_unlock(&item->mutex);
    if (item->status != YVEX_OK) *err = item->error;
    return item->status;
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
            rc = yvex_server_sessions_cancel(server->sessions,
                                                request.session_name, &err);
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
                        rc = client_wait_work(server, &item, fd, &err);
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
        server->telemetry, YVEX_SERVER_EVENT_CLIENT_DISCONNECTED,
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
    unsigned long long queued = 0ull, active = 0ull;
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
    yvex_server_scheduler_snapshot(server->scheduler, &queued, &active);
    (void)yvex_server_telemetry_emit(
        server->telemetry, YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_START,
        YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, "shutdown",
        queued, server->active_clients, active, 0.0, 0.0, err);
    if (server->listen_fd >= 0) {
        (void)shutdown(server->listen_fd, SHUT_RDWR);
        (void)close(server->listen_fd);
        server->listen_fd = -1;
    }
    yvex_server_sessions_cancel_all(server->sessions);
    yvex_server_scheduler_request_stop(server->scheduler);
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
 * Joins scheduler workers, closes runtime ownership, emits shutdown.complete, and leaves
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
    cleanup_rc = yvex_server_scheduler_finish(server->scheduler, &cleanup);
    if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
        rc = cleanup_rc;
        primary = cleanup;
    }
    cleanup_rc = yvex_server_sessions_close(&server->sessions, &cleanup);
    if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
        rc = cleanup_rc;
        primary = cleanup;
    }
    cleanup_rc = yvex_runtime_generation_context_close(
        &server->warm_generation, &cleanup);
    if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
        rc = cleanup_rc;
        primary = cleanup;
    }
    if (!server->warm_generation) {
        cleanup_rc = yvex_runtime_session_close(&server->warm_session, &cleanup);
        if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
            rc = cleanup_rc;
            primary = cleanup;
        }
    }
    if (server->model && !server->warm_generation && !server->warm_session) {
        yvex_runtime_model_close(&server->model);
        yvex_server_telemetry_model_closed(server->telemetry);
    }
    (void)yvex_server_telemetry_metrics_copy(server->telemetry, &metrics,
                                             &cleanup);
    cleanup_rc = yvex_server_telemetry_emit(
        server->telemetry, YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE,
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
    unsigned long long sessions = 0u;
    if (!server || !out || pthread_mutex_lock(&mutable->state_mutex) != 0)
        return server_refuse(err, YVEX_ERR_INVALID_ARG,
                             "host and summary output are required");
    *out = server->summary;
    (void)pthread_mutex_unlock(&mutable->state_mutex);
    if (server->telemetry)
        (void)yvex_server_telemetry_metrics_copy(server->telemetry,
                                            &out->metrics, err);
    if (server->sessions)
        (void)yvex_server_sessions_count(server->sessions, &sessions, err);
    out->session_count = sessions;
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
    if (owner->socket_path[0]) (void)unlink(owner->socket_path);
    if (owner->lock_fd >= 0) (void)close(owner->lock_fd);
    if (owner->lock_owned && owner->lock_path[0])
        (void)unlink(owner->lock_path);
    yvex_server_openai_close(&owner->openai);
    yvex_server_scheduler_close(&owner->scheduler);
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
