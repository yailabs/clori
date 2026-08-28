/* Run the persistent foreground host and resolve explicit engine loads from the local registry. */
#define _POSIX_C_SOURCE 200809L

#include <yvex/registry.h>
#include <yvex/server.h>
#include <yvex/internal/core.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/server_media.h>

#include "src/cli/private.h"
#include "src/cli/io/private.h"

#include <errno.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

typedef struct {
    char name[256];
    char family[64];
    char profile_kind[32];
    char installation[PATH_MAX];
    char artifact[PATH_MAX];
    char binding[PATH_MAX];
    char target[128];
    char backend[8];
    char mode[16];
    unsigned long long context_capacity;
} cli_server_profile;

typedef struct {
    const char *artifact_root;
    const char *output_root;
    const char *artifact_reopen_cache_root;
    char default_output_root[PATH_MAX];
    char default_cache_root[PATH_MAX];
} cli_server_media_configuration;

typedef struct {
    yvex_server_options host;
    pthread_mutex_t registry_mutex;
} cli_server_loader_context;

typedef struct {
    yvex_server *server;
    cli_server_loader_context *loader;
    const char *socket_path;
    int attached;
    int leave_console;
} cli_server_thread_state;

static int parse_u64(const char *text, unsigned long long *value)
{
    char *end = NULL;
    unsigned long long parsed;
    if (!text || !value || !text[0] || text[0] == '-') return 0;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno || !end || *end || !parsed) return 0;
    *value = parsed;
    return 1;
}

static int profile_copy(cli_server_profile *profile,
                        const yvex_model_registry_entry *entry)
{
    const char *profile_kind;
    const char *installation;
    const char *artifact;
    const char *binding;
    if (!profile || !entry || !entry->alias || !entry->family ||
        !entry->runtime_target || !entry->runtime_backend || !entry->runtime_mode ||
        strlen(entry->alias) >= sizeof(profile->name) ||
        strlen(entry->family) >= sizeof(profile->family) ||
        strlen(entry->runtime_target) >= sizeof(profile->target) ||
        strlen(entry->runtime_backend) >= sizeof(profile->backend) ||
        strlen(entry->runtime_mode) >= sizeof(profile->mode))
        return 0;
    profile_kind = entry->runtime_profile && entry->runtime_profile[0]
                       ? entry->runtime_profile : "single-artifact";
    installation = entry->runtime_installation ? entry->runtime_installation : "";
    artifact = entry->path ? entry->path : "";
    binding = entry->runtime_binding ? entry->runtime_binding : "";
    if (strlen(profile_kind) >= sizeof(profile->profile_kind) ||
        strlen(installation) >= sizeof(profile->installation) ||
        strlen(artifact) >= sizeof(profile->artifact) ||
        strlen(binding) >= sizeof(profile->binding))
        return 0;
    return snprintf(profile->name, sizeof(profile->name), "%s", entry->alias) > 0 &&
           snprintf(profile->family, sizeof(profile->family), "%s", entry->family) >= 0 &&
           snprintf(profile->profile_kind, sizeof(profile->profile_kind), "%s",
                    profile_kind) > 0 &&
           snprintf(profile->installation, sizeof(profile->installation), "%s",
                    installation) >= 0 &&
           snprintf(profile->artifact, sizeof(profile->artifact), "%s", artifact) >= 0 &&
           snprintf(profile->binding, sizeof(profile->binding), "%s", binding) >= 0 &&
           snprintf(profile->target, sizeof(profile->target), "%s",
                    entry->runtime_target) >= 0 &&
           snprintf(profile->backend, sizeof(profile->backend), "%s",
                    entry->runtime_backend) >= 0 &&
           snprintf(profile->mode, sizeof(profile->mode), "%s",
                    entry->runtime_mode) >= 0;
}

static int profile_resolve(const char *name, cli_server_profile *profile,
                           int *media_requested, yvex_error *err)
{
    yvex_model_registry_options options;
    yvex_model_registry *registry = NULL;
    const yvex_model_registry_entry *entry;
    int rc;
    memset(&options, 0, sizeof(options));
    memset(profile, 0, sizeof(*profile));
    rc = yvex_model_registry_open(&registry, &options, err);
    if (rc != YVEX_OK) return rc;
    entry = yvex_model_registry_find(registry, name);
    if (!entry) {
        yvex_error_setf(err, YVEX_ERR_STATE, "server.model-loader",
                        "model is not registered: %s", name);
        yvex_model_registry_close(registry);
        return YVEX_ERR_STATE;
    }
    *media_requested = entry->runtime_profile &&
                       !strcmp(entry->runtime_profile, "composite") &&
                       entry->runtime_mode && !strcmp(entry->runtime_mode, "media");
    rc = yvex_model_registry_startup_validate(entry, err);
    if (rc != YVEX_OK) {
        yvex_model_registry_close(registry);
        return rc;
    }
    if (!profile_copy(profile, entry)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.model-loader",
                       "registered startup profile exceeds command limits");
        yvex_model_registry_close(registry);
        return YVEX_ERR_BOUNDS;
    }
    profile->context_capacity = entry->runtime_context;
    yvex_model_registry_close(registry);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int server_error(const yvex_error *err, int status)
{
    fprintf(stderr, "yvex server: %s: %s\n", yvex_error_where(err),
            yvex_error_message(err));
    return status;
}

static void *signal_main(void *opaque)
{
    cli_server_thread_state *state = opaque;
    sigset_t set;
    int signal_number;
    yvex_error err;
    (void)sigemptyset(&set);
    (void)sigaddset(&set, SIGINT);
    (void)sigaddset(&set, SIGTERM);
    if (sigwait(&set, &signal_number) == 0)
        (void)yvex_server_stop(state->server, &err);
    return NULL;
}

static void *raw_console_main(void *opaque)
{
    cli_server_thread_state *state = opaque;
    unsigned long long cursor = 0u;
    char line[2048];
    for (;;) {
        yvex_server_event event;
        yvex_error err;
        if (yvex_server_event_next(state->server, cursor, 1, &event, &err) != YVEX_OK)
            continue;
        cursor = event.sequence;
        if (yvex_server_event_json(&event, line, sizeof(line), &err) == YVEX_OK) {
            (void)fwrite(line, 1u, strlen(line), stdout);
            (void)fflush(stdout);
        }
        if (event.kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE) break;
    }
    return NULL;
}

static void *human_console_main(void *opaque)
{
    cli_server_thread_state *state = opaque;
    yvex_cli_watch_renderer renderer;
    unsigned long long cursor = 0u;
    yvex_cli_watch_renderer_open(&renderer, 0);
    for (;;) {
        yvex_server_event event;
        yvex_error err;
        if (yvex_server_event_next(state->server, cursor, 1, &event, &err) != YVEX_OK)
            continue;
        cursor = event.sequence;
        (void)yvex_cli_watch_renderer_event(&renderer, &event);
        if (event.kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE) break;
    }
    yvex_cli_watch_renderer_finish(&renderer);
    return NULL;
}

static void engine_profile_defaults(yvex_server_engine_options *options,
                                    const cli_server_profile *profile,
                                    int media_requested)
{
    memset(options, 0, sizeof(*options));
    options->schema_version = YVEX_SERVER_ENGINE_SCHEMA_CURRENT;
    options->alias = profile->name;
    options->artifact_path = profile->artifact;
    options->runtime_binding_path = profile->binding;
    options->target_id = profile->target;
    options->backend = media_requested || !strcmp(profile->backend, "cuda")
                           ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU;
    options->engine_kind = media_requested ? YVEX_SERVER_ENGINE_MEDIA
                                           : YVEX_SERVER_ENGINE_TEXT;
    options->execution_strategy =
        media_requested ? YVEX_SERVER_EXECUTION_NOT_APPLICABLE
                        : (!strcmp(profile->mode, "dspark")
                               ? YVEX_SERVER_EXECUTION_SPECULATIVE
                               : YVEX_SERVER_EXECUTION_TARGET_ONLY);
    options->context_capacity = media_requested ? 0u : profile->context_capacity;
    options->prefill_chunk_tokens = 0u;
    options->maximum_new_tokens = 0u;
    options->maximum_output_bytes = 1048576u;
    options->maximum_sessions = 8u;
    options->concurrent_sequences = 1u;
    options->trace_level = YVEX_SERVER_TRACE_STAGES;
}

static int option_parse(yvex_server_options *host, const char *flag,
                        const char *value)
{
    if (!strcmp(flag, "--socket")) host->socket_path = value;
    else if (!strcmp(flag, "--workers"))
        return parse_u64(value, &host->worker_count) &&
               host->worker_count <= 64ull;
    else if (!strcmp(flag, "--max-engines"))
        return parse_u64(value, &host->maximum_engines) &&
               host->maximum_engines <=
                   YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES;
    else if (!strcmp(flag, "--console")) {
        if (!strcmp(value, "human")) host->console = YVEX_SERVER_CONSOLE_HUMAN;
        else if (!strcmp(value, "raw")) host->console = YVEX_SERVER_CONSOLE_RAW;
        else host->console = YVEX_SERVER_CONSOLE_OFF;
    }
    else if (!strcmp(flag, "--trace-level")) {
        if (!strcmp(value, "summary")) host->trace_level = YVEX_SERVER_TRACE_SUMMARY;
        else if (!strcmp(value, "stages")) host->trace_level = YVEX_SERVER_TRACE_STAGES;
        else if (!strcmp(value, "tokens")) host->trace_level = YVEX_SERVER_TRACE_TOKENS;
        else host->trace_level = YVEX_SERVER_TRACE_FULL;
    } else if (!strcmp(flag, "--openai"))
        host->openai_enabled = !strcmp(value, "on");
    else if (!strcmp(flag, "--openai-port")) {
        unsigned long long port;
        if (!parse_u64(value, &port) || port > 65535u) return 0;
        host->openai_port = (unsigned short)port;
    } else if (!strcmp(flag, "--openai-timeout-ms"))
        return parse_u64(value, &host->openai_timeout_ms);
    else return 0;
    return 1;
}

static int command_options_parse(cli_server_loader_context *context,
                                 int argc, char **argv, size_t consumed)
{
    int index;
    for (index = (int)consumed + 1; index < argc; ++index) {
        const char *flag = argv[index];
        if (!strcmp(flag, "--trace-content")) {
            context->host.trace_content = 1;
            continue;
        }
        if (index + 1 >= argc ||
            !option_parse(&context->host, flag, argv[index + 1])) {
            fprintf(stderr, "yvex server: invalid option: %s\n", flag);
            return 0;
        }
        index++;
    }
    return 1;
}

static int media_configuration_defaults(
    const cli_server_profile *profile, cli_server_media_configuration *configuration,
    yvex_error *err)
{
    yvex_paths paths;
    char admission_path[YVEX_PATH_CAP];
    int admission_written, written, rc;
    if (!configuration->artifact_root)
        configuration->artifact_root = profile->installation;
    rc = yvex_paths_default(&paths, err);
    if (rc == YVEX_OK) {
        yvex_core_text_copy(configuration->default_cache_root,
                            sizeof(configuration->default_cache_root), paths.cache_dir);
        configuration->artifact_reopen_cache_root = configuration->default_cache_root;
    }
    if (rc != YVEX_OK || configuration->output_root) return rc;
    written = rc == YVEX_OK
                  ? snprintf(configuration->default_output_root,
                             sizeof(configuration->default_output_root), "%s/media",
                             paths.data_dir)
                  : -1;
    if (rc != YVEX_OK) return rc;
    admission_written = written >= 0 &&
                                (size_t)written < sizeof(configuration->default_output_root)
                            ? snprintf(admission_path, sizeof(admission_path),
                                       "%s/.publication",
                                       configuration->default_output_root)
                            : -1;
    if (written < 0 || (size_t)written >= sizeof(configuration->default_output_root) ||
        admission_written < 0 || (size_t)admission_written >= sizeof(admission_path)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.media-output",
                       "default media output path exceeds capacity");
        return YVEX_ERR_BOUNDS;
    }
    rc = yvex_core_mkdir_parent(admission_path, "server.media-output", err);
    if (rc == YVEX_OK)
        configuration->output_root = configuration->default_output_root;
    return rc;
}

static int media_prepare(const cli_server_profile *profile,
                         const cli_server_media_configuration *configuration,
                         yvex_runtime_media_host_profile *host,
                         yvex_server_media_options *media,
                         yvex_server_engine_options *options, yvex_error *err)
{
    const yvex_component_variant_adapter *adapter;
    yvex_media_target_profile target = {0};
    int rc;

    if (!configuration->artifact_root || !configuration->artifact_root[0] ||
        !configuration->output_root || !configuration->output_root[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.media-profile",
                       "media mode requires a composite installation and publication root");
        return YVEX_ERR_INVALID_ARG;
    }
    adapter = yvex_graph_component_variant_find_family(profile->family);
    if (!adapter || !adapter->media_target_profile || !adapter->media_execution) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "server.media-profile",
                        "family has no conversational media adapter: %s", profile->family);
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = adapter->media_target_profile(&target, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_media_host_profile_build(
            host, &target, adapter->media_execution, configuration->artifact_root,
            configuration->output_root, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_media_execution_preset_build(
            host, &media->execution_preset, err);
    if (rc != YVEX_OK) return rc;
    if (options->backend != host->request_template.component_backend) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.media-profile",
                       "media mode requires the admitted CUDA backend");
        return YVEX_ERR_INVALID_ARG;
    }
    options->artifact_path = NULL;
    options->runtime_binding_path = NULL;
    options->target_id = host->target;
    media->schema_version = YVEX_SERVER_MEDIA_SCHEMA_V1;
    media->output_root = host->output_root;
    media->artifact_reopen_cache_root = configuration->artifact_reopen_cache_root;
    media->request_template = host->request_template;
    media->profiles = host->profiles;
    media->profile_count = host->profile_count;
    media->frames_per_chunk = host->frames_per_chunk;
    media->frame_remainder = host->frame_remainder;
    media->minimum_frames = host->minimum_frames;
    media->maximum_frames = host->maximum_frames;
    media->minimum_inference_steps = host->minimum_inference_steps;
    media->maximum_inference_steps = host->maximum_inference_steps;
    media->canvas_multiple = host->canvas_multiple;
    media->maximum_canvas_pixels = host->maximum_canvas_pixels;
    return YVEX_OK;
}

static void host_options_defaults(yvex_server_options *options)
{
    memset(options, 0, sizeof(*options));
    options->schema_version = YVEX_SERVER_OPTIONS_SCHEMA_CURRENT;
    options->request_queue_capacity = 16u;
    options->worker_count = 1u;
    options->maximum_engines = YVEX_SERVER_DEFAULT_MAXIMUM_ENGINES;
    options->trace_level = YVEX_SERVER_TRACE_STAGES;
    options->console = YVEX_SERVER_CONSOLE_HUMAN;
    options->openai_enabled = 1;
    options->openai_port = 8001u;
    options->openai_timeout_ms = 600000u;
}

static int registered_model_load(void *opaque, yvex_server *server,
                                 const char *alias, yvex_error *err)
{
    cli_server_loader_context *context = opaque;
    cli_server_profile profile;
    cli_server_media_configuration media_configuration;
    yvex_runtime_media_host_profile media_host = {0};
    yvex_server_media_options media = {0};
    yvex_server_engine_options selected;
    yvex_server_engine_summary summary;
    int media_requested = 0, rc;
    if (pthread_mutex_lock(&context->registry_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.model-loader",
                       "model registry lock failed");
        return YVEX_ERR_STATE;
    }
    rc = profile_resolve(alias, &profile, &media_requested, err);
    (void)pthread_mutex_unlock(&context->registry_mutex);
    if (rc != YVEX_OK) return rc;
    engine_profile_defaults(&selected, &profile, media_requested);
    selected.maximum_new_tokens = media_requested ? 0ull
                                                  : selected.context_capacity;
    selected.trace_level = context->host.trace_level;
    memset(&media_configuration, 0, sizeof(media_configuration));
    if (media_requested) {
        rc = media_configuration_defaults(
            &profile, &media_configuration, err);
        if (rc == YVEX_OK)
            rc = media_prepare(&profile, &media_configuration, &media_host,
                               &media, &selected, err);
    } else {
        rc = YVEX_OK;
    }
    if (rc != YVEX_OK) return rc;
    return media_requested
               ? yvex_server_media_engine_load(
                     server, &selected, &media, &summary, err)
               : yvex_server_engine_load(server, &selected, &summary, err);
}

static const char *console_host_state_name(yvex_server_status status)
{
    switch (status) {
    case YVEX_SERVER_STATUS_CONFIGURED: return "configured";
    case YVEX_SERVER_STATUS_STARTING: return "starting";
    case YVEX_SERVER_STATUS_READY: return "ready";
    case YVEX_SERVER_STATUS_STOPPING: return "stopping";
    case YVEX_SERVER_STATUS_STOPPED: return "stopped";
    case YVEX_SERVER_STATUS_FAILED: return "failed";
    }
    return "unknown";
}

static const char *console_engine_state_name(yvex_server_engine_state state)
{
    switch (state) {
    case YVEX_SERVER_ENGINE_UNLOADED: return "unloaded";
    case YVEX_SERVER_ENGINE_LOADING: return "loading";
    case YVEX_SERVER_ENGINE_LOADED: return "loaded";
    case YVEX_SERVER_ENGINE_DRAINING: return "draining";
    case YVEX_SERVER_ENGINE_UNLOADING: return "unloading";
    case YVEX_SERVER_ENGINE_FAILED: return "failed";
    }
    return "unknown";
}

static int console_remote_connect(
    const char *socket_path, yvex_client_operation operation,
    yvex_client_request *request, yvex_client **client, yvex_error *err)
{
    yvex_cli_client_request_init(request, operation);
    return yvex_client_connect(client, socket_path, err);
}

static int console_remote_message_error(
    const yvex_client_message *message, const char *where, yvex_error *err)
{
    if (message->kind != YVEX_CLIENT_MESSAGE_ERROR) return YVEX_OK;
    yvex_error_set(err, (yvex_status)message->status, where,
                   message->reason[0] ? message->reason : "server request failed");
    return message->status;
}

static int console_remote_summary(
    const char *socket_path, yvex_server_summary *summary, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc = console_remote_connect(
        socket_path, YVEX_CLIENT_OP_RUNTIME_STATUS, &request, &client, err);

    if (rc == YVEX_OK) rc = yvex_client_send(client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK) rc = console_remote_message_error(&message, "client.status", err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_STATUS)
        *summary = message.runtime;
    else if (rc == YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "client.status",
                       "server returned an unexpected status response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc;
}

static int console_remote_engine_snapshot(
    const char *socket_path,
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES],
    unsigned long long *count, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    unsigned long long used = 0u;
    int rc = console_remote_connect(
        socket_path, YVEX_CLIENT_OP_ENGINE_LIST, &request, &client, err);

    if (rc == YVEX_OK) rc = yvex_client_send(client, &request, err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, err);
        if (rc != YVEX_OK) break;
        rc = console_remote_message_error(&message, "client.engines", err);
        if (rc != YVEX_OK || message.kind == YVEX_CLIENT_MESSAGE_ACK) break;
        if (message.kind != YVEX_CLIENT_MESSAGE_ENGINE) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "client.engines",
                           "server returned an unexpected engine-list response");
            rc = YVEX_ERR_FORMAT;
            break;
        }
        if (used == YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "client.engines",
                           "server engine catalog exceeds the client bound");
            rc = YVEX_ERR_BOUNDS;
            break;
        }
        engines[used++] = message.engine;
    }
    yvex_client_close(&client);
    if (rc == YVEX_OK) *count = used;
    return rc;
}

static int console_remote_engine_control(
    const char *socket_path, yvex_client_operation operation,
    const char *alias, unsigned long long generation,
    yvex_server_engine_summary *summary, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc = console_remote_connect(
        socket_path, operation, &request, &client, err);

    if (rc == YVEX_OK) {
        (void)snprintf(request.model_alias, sizeof(request.model_alias), "%s", alias);
        request.engine_generation = generation;
        rc = yvex_client_send(client, &request, err);
    }
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK) rc = console_remote_message_error(&message, "client.engine", err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ENGINE)
        *summary = message.engine;
    else if (rc == YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "client.engine",
                       "server returned an unexpected engine response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc;
}

static int console_remote_stop(const char *socket_path, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc = console_remote_connect(
        socket_path, YVEX_CLIENT_OP_RUNTIME_STOP, &request, &client, err);

    if (rc == YVEX_OK) rc = yvex_client_send(client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK) rc = console_remote_message_error(&message, "client.stop", err);
    if (rc == YVEX_OK && message.kind != YVEX_CLIENT_MESSAGE_ACK) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "client.stop",
                       "server returned an unexpected stop response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc;
}

static int console_summary(const cli_server_thread_state *state,
                           yvex_server_summary *summary, yvex_error *err)
{
    return state->attached
               ? console_remote_summary(state->socket_path, summary, err)
               : yvex_server_get_summary(state->server, summary, err);
}

static int console_engine_snapshot(
    const cli_server_thread_state *state,
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES],
    unsigned long long *count, yvex_error *err)
{
    return state->attached
               ? console_remote_engine_snapshot(state->socket_path, engines,
                                                count, err)
               : yvex_server_engine_snapshot(
                     state->server, engines,
                     YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES, count, err);
}

static void console_status_render(const cli_server_thread_state *state)
{
    yvex_server_summary summary;
    yvex_error err;
    if (console_summary(state, &summary, &err) != YVEX_OK) {
        fprintf(stderr, "yvex server: status: %s\n", yvex_error_message(&err));
        return;
    }
    printf("host %s · engines %llu loaded/%llu known/%llu max · workers %llu"
           " · sessions %llu · queue %llu/%llu\n",
           console_host_state_name(summary.status), summary.loaded_engine_count,
           summary.engine_count, summary.maximum_engines, summary.worker_count,
           summary.session_count, summary.metrics.queue_depth,
           summary.metrics.queue_capacity);
}

static void console_models_render(const cli_server_thread_state *state)
{
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    unsigned long long count = 0u, index;
    yvex_error err;
    if (console_engine_snapshot(state, engines, &count, &err) != YVEX_OK) {
        fprintf(stderr, "yvex server: models: %s\n", yvex_error_message(&err));
        return;
    }
    if (!count) {
        puts("no model engines known to this host");
        return;
    }
    for (index = 0u; index < count; ++index)
        printf("  %s · %s · generation %llu · work %llu · sessions %llu\n",
               engines[index].alias, console_engine_state_name(engines[index].state),
               engines[index].generation, engines[index].active_work,
               engines[index].session_count);
}

static int console_profiles(cli_server_loader_context *context,
                            const char *selector,
                            char alias[YVEX_SERVER_MODEL_ALIAS_CAP], int render,
                            yvex_error *err)
{
    yvex_model_registry_options options;
    yvex_model_registry *registry = NULL;
    char target_alias[YVEX_SERVER_MODEL_ALIAS_CAP] = "";
    unsigned long long index, visible = 0u, requested = 0u, target_matches = 0u;
    int numeric = selector && parse_u64(selector, &requested);
    int direct_match = 0, rc;
    memset(&options, 0, sizeof(options));
    if (alias) alias[0] = '\0';
    if (pthread_mutex_lock(&context->registry_mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.console-profile",
                       "model registry lock failed");
        return YVEX_ERR_STATE;
    }
    rc = yvex_model_registry_open(&registry, &options, err);
    if (rc != YVEX_OK) {
        (void)pthread_mutex_unlock(&context->registry_mutex);
        return rc;
    }
    if (render)
        puts("structurally complete local profiles · load authenticates artifact + binding");
    for (index = 0u; index < yvex_model_registry_count(registry); ++index) {
        const yvex_model_registry_entry *entry =
            yvex_model_registry_at(registry, index);
        char entry_alias[YVEX_SERVER_MODEL_ALIAS_CAP];
        char target[128], backend[8], mode[16];
        unsigned long long file_size;
        yvex_error ignored;
        if (!entry ||
            yvex_model_registry_startup_validate(entry, &ignored) != YVEX_OK)
            continue;
        (void)snprintf(entry_alias, sizeof(entry_alias), "%s", entry->alias);
        (void)snprintf(target, sizeof(target), "%s", entry->runtime_target);
        (void)snprintf(backend, sizeof(backend), "%s", entry->runtime_backend);
        (void)snprintf(mode, sizeof(mode), "%s", entry->runtime_mode);
        file_size = entry->file_size;
        visible++;
        if (render)
            printf("  [%llu] %s\n"
                   "      %s · %s/%s · %.2f GiB\n",
                   visible, entry_alias, target, backend, mode,
                   (double)file_size / 1073741824.0);
        if (alias && selector &&
            ((numeric && visible == requested) ||
             (!numeric && !strcmp(selector, entry_alias)))) {
            (void)snprintf(alias, YVEX_SERVER_MODEL_ALIAS_CAP, "%s", entry_alias);
            direct_match = 1;
        } else if (alias && selector && !numeric && !strcmp(selector, target)) {
            target_matches++;
            (void)snprintf(target_alias, sizeof(target_alias), "%s", entry_alias);
        }
    }
    yvex_model_registry_close(registry);
    (void)pthread_mutex_unlock(&context->registry_mutex);
    if (render && !visible) puts("  no structurally complete profile is registered");
    if (alias && !direct_match && target_matches == 1u)
        (void)snprintf(alias, YVEX_SERVER_MODEL_ALIAS_CAP, "%s", target_alias);
    if (alias && !direct_match && target_matches > 1u) {
        yvex_error_setf(err, YVEX_ERR_STATE, "server.console-profile",
                        "target matches %llu profiles; use an exact alias or number",
                        target_matches);
        return YVEX_ERR_STATE;
    }
    if (alias && !alias[0]) {
        yvex_error_setf(err, YVEX_ERR_STATE, "server.console-profile",
                        "no startup profile matches: %s",
                        selector && selector[0] ? selector : "(missing)");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static void console_prompt(const cli_server_thread_state *state)
{
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    unsigned long long count = 0u, index, loaded = 0u;
    const char *label = "yvex[host]";
    yvex_cli_terminal_style style;
    yvex_error err;
    if (console_engine_snapshot(state, engines, &count, &err) == YVEX_OK)
        for (index = 0u; index < count; ++index)
            if (engines[index].state == YVEX_SERVER_ENGINE_LOADED) {
                loaded++;
                label = engines[index].target_id[0]
                            ? engines[index].target_id : engines[index].alias;
            }
    if (loaded != 1u) label = loaded ? "yvex[multi-engine]" : "yvex[host]";
    yvex_cli_terminal_style_get(stdout, &style);
    printf("%s%s%s > ", style.strong, label, style.reset);
    (void)fflush(stdout);
}

static int console_engine_load(cli_server_thread_state *state,
                               const char *selector)
{
    char alias[YVEX_SERVER_MODEL_ALIAS_CAP];
    yvex_error err;
    int rc = console_profiles(state->loader, selector, alias, 0, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "load: %s\n", yvex_error_message(&err));
        return rc;
    }
    printf("loading %s ...\n", alias);
    (void)fflush(stdout);
    if (state->attached) {
        yvex_server_engine_summary summary;
        rc = console_remote_engine_control(
            state->socket_path, YVEX_CLIENT_OP_ENGINE_LOAD, alias, 0u,
            &summary, &err);
    } else {
        rc = registered_model_load(state->loader, state->server, alias, &err);
    }
    if (rc != YVEX_OK) {
        fprintf(stderr, "load failed at %s: %s\n", yvex_error_where(&err),
                yvex_error_message(&err));
        fputs("load hint: registry readiness is structural; use `profiles` to select another "
              "exact alias or repair this profile\n", stderr);
    } else
        printf("loaded %s\n", alias);
    return rc;
}

static int console_engine_unload(cli_server_thread_state *state,
                                 const char *selector)
{
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    yvex_server_engine_summary result;
    unsigned long long count = 0u, index, matches = 0u, selected = 0u;
    yvex_error err;
    int rc = console_engine_snapshot(state, engines, &count, &err);
    if (rc != YVEX_OK) return rc;
    for (index = 0u; index < count; ++index)
        if (engines[index].state == YVEX_SERVER_ENGINE_LOADED &&
            (!selector || !selector[0] || !strcmp(selector, engines[index].alias) ||
             !strcmp(selector, engines[index].target_id))) {
            selected = index;
            matches++;
        }
    if (matches != 1u) {
        fprintf(stderr, "unload: %s\n",
                matches ? "model selection is ambiguous"
                        : "no matching loaded model engine");
        return YVEX_ERR_STATE;
    }
    rc = state->attached
             ? console_remote_engine_control(
                   state->socket_path, YVEX_CLIENT_OP_ENGINE_UNLOAD,
                   engines[selected].alias, engines[selected].generation,
                   &result, &err)
             : yvex_server_engine_unload(
                   state->server, engines[selected].alias,
                   engines[selected].generation, &result, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "unload failed at %s: %s\n", yvex_error_where(&err),
                yvex_error_message(&err));
    else
        printf("unloaded %s · host remains ready\n", engines[selected].alias);
    return rc;
}

static void console_help(int attached)
{
    puts("commands\n"
         "  profiles              list structurally complete local runtime profiles\n"
         "  load MODEL|N          load an exact alias, target, or profile number\n"
         "  models                list engines owned by this host\n"
         "  unload [MODEL]        unload one exact engine; omit when only one is loaded\n"
         "  status                show host and resource state\n"
         "  help                  show these commands\n"
         "  stop                  stop the host and leave the console");
    if (attached)
        puts("  exit                  detach this console; keep the host online");
}

static void console_line_run(cli_server_thread_state *state, char *line)
{
    char *command = line, *argument = NULL, *end;
    while (*command == ' ' || *command == '\t') command++;
    end = command + strlen(command);
    while (end > command &&
           (end[-1] == '\n' || end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t'))
        *--end = '\0';
    if (*command == '/') command++;
    argument = strpbrk(command, " \t");
    if (argument) {
        *argument++ = '\0';
        while (*argument == ' ' || *argument == '\t') argument++;
        if (!argument[0]) argument = NULL;
    }
    if (!command[0]) return;
    if (!strcmp(command, "help") || !strcmp(command, "?"))
        console_help(state->attached);
    else if (!strcmp(command, "profiles")) {
        yvex_error err;
        if (console_profiles(state->loader, NULL, NULL, 1, &err) != YVEX_OK)
            fprintf(stderr, "profiles: %s\n", yvex_error_message(&err));
    }
    else if (!strcmp(command, "models")) console_models_render(state);
    else if (!strcmp(command, "status")) console_status_render(state);
    else if (!strcmp(command, "load")) (void)console_engine_load(state, argument);
    else if (!strcmp(command, "unload")) (void)console_engine_unload(state, argument);
    else if (!strcmp(command, "stop")) {
        yvex_error err;
        int rc = state->attached
                     ? console_remote_stop(state->socket_path, &err)
                     : yvex_server_stop(state->server, &err);
        if (rc != YVEX_OK)
            fprintf(stderr, "stop: %s\n", yvex_error_message(&err));
        else
            state->leave_console = 1;
    } else if (!strcmp(command, "quit") || !strcmp(command, "exit")) {
        if (state->attached) {
            puts("console detached · YVEX host remains online");
            state->leave_console = 1;
        } else {
            yvex_error err;
            (void)yvex_server_stop(state->server, &err);
        }
    } else {
        fprintf(stderr, "unknown server-console command: %s (type `help`)\n", command);
    }
}

static void console_ready_announce(int attached)
{
    yvex_cli_terminal_style style;

    yvex_cli_terminal_style_get(stdout, &style);
    printf("\n  %s● %s%s  %sInteractive host console %s%s\n",
           style.success, attached ? "ATTACHED" : "READY", style.reset,
           style.strong, attached ? "connected" : "ready", style.reset);
    printf("  %sLIFECYCLE%s  profiles  ·  load MODEL  ·  models  ·  unload [MODEL]\n",
           style.dim, style.reset);
    printf("  %sOBSERVE%s    status    ·  help        %sSHUTDOWN%s  stop\n",
           style.dim, style.reset, style.dim, style.reset);
    printf("  %sExternal Unix and OpenAI clients remain online while you operate here.%s\n",
           style.dim, style.reset);
    if (attached) {
        printf("  %sserver already active · this console uses its existing engine manager%s\n",
               style.dim, style.reset);
        printf("  %s`exit` detaches this console; `stop` shuts down the shared host.%s\n",
               style.dim, style.reset);
    }
    fputc('\n', stdout);
}

static void *operator_console_main(void *opaque)
{
    cli_server_thread_state *state = opaque;
    struct pollfd input = {.fd = STDIN_FILENO, .events = POLLIN | POLLHUP};
    char line[1024];
    console_ready_announce(state->attached);
    console_prompt(state);
    for (;;) {
        yvex_server_summary summary;
        yvex_error err;
        int ready = poll(&input, 1u, state->attached ? -1 : 250);
        if (state->leave_console)
            break;
        if (!state->attached &&
            (console_summary(state, &summary, &err) != YVEX_OK ||
             summary.status != YVEX_SERVER_STATUS_READY))
            break;
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0 || (input.revents & (POLLERR | POLLNVAL))) break;
        if (ready > 0 && (input.revents & POLLIN)) {
            if (!fgets(line, sizeof(line), stdin)) {
                if (!state->attached)
                    (void)yvex_server_stop(state->server, &err);
                break;
            }
            console_line_run(state, line);
            if (!state->leave_console &&
                console_summary(state, &summary, &err) == YVEX_OK &&
                summary.status == YVEX_SERVER_STATUS_READY)
                console_prompt(state);
        } else if (ready > 0 && (input.revents & POLLHUP)) {
            if (!state->attached)
                (void)yvex_server_stop(state->server, &err);
            break;
        }
    }
    return NULL;
}

static unsigned int startup_terminal_columns(void)
{
    struct winsize window = {0};
    unsigned long long configured;

    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &window) == 0 && window.ws_col)
        return window.ws_col;
    if (parse_u64(getenv("COLUMNS"), &configured) && configured <= UINT_MAX)
        return (unsigned int)configured;
    return 80u;
}

static void startup_tail_fit(char *output, size_t capacity, const char *text,
                             size_t maximum)
{
    size_t length, retained;

    if (!output || !capacity) return;
    text = text ? text : "unavailable";
    length = strlen(text);
    if (maximum >= capacity) maximum = capacity - 1u;
    if (length <= maximum) {
        (void)snprintf(output, capacity, "%s", text);
        return;
    }
    if (maximum <= 3u) {
        (void)snprintf(output, capacity, "%.*s", (int)maximum, text);
        return;
    }
    retained = maximum - 3u;
    (void)snprintf(output, capacity, "...%s", text + length - retained);
}

static size_t startup_text_columns(const char *text)
{
    const unsigned char *byte = (const unsigned char *)(text ? text : "");
    size_t columns = 0u;

    for (; *byte; ++byte)
        if ((*byte & 0xc0u) != 0x80u) columns++;
    return columns;
}

static const char *startup_logo_line(size_t index)
{
    /* Generated from the canonical traced mark at 88x64 dots, then packed into
     * Unicode Braille cells.  The star is strengthened for terminal legibility. */
    static const char *const logo[] = {
        "                     ✦",
        " ⢲⣤⣀          " "   ⡀   ⢸⡇   ⢀ " "            ⣀⣤" "⡖",
        "  ⠙⢿⣿⣶⣤⣀      " "   ⠘⢆  ⢸⡇  ⡴⠃ " "        ⣀⣤⣶⣿⡿⠋",
        "   ⠈⠻⣿⣿⣿⣿⣶⣄⡀  " "    ⠈⢷⡀  ⢀⡞⠁  " "    ⢀⣠⣶⣿⣿⣿⣿⠟⠁",
        "     ⠈⢿⣿⣿⣿⣿⣿⣷⣦" "⣄     ⠻⣄⣠⠟    " " ⣠⣴⣾⣿⣿⣿⣿⣿⡿⠁",
        "       ⠙⢿⣿⣿⣿⣿⣿" "⣿⣿⣶⣄⡀  ⣿⣿  ⢀⣠⣶" "⣿⣿⣿⣿⣿⣿⣿⡿⠋",
        "         ⠹⣿⣿⣿⣿" "⣿⣿⣿⣟⣛⠲⢴⣿⣿⡤⠞⣛⣿⣿" "⣿⣿⣿⣿⣿⣿⠏",
        "          ⠈⢻⣿⣿" "⣿⣿⣿⠿⢿⣻⣿⣿⣿⣿⣟⡿⠿⣿" "⣿⣿⣿⣿⡟⠁",
        "            ⠙⠛" "⠉⢁⣤⣶⣿⣿⢻⣿⣿⡟⣿⣿⣶⣤" "⡈⠉⠛⠋",
        "              " "⣰⣿⣿⣿⣿⠏⢀⣿⣿⡀⠹⣿⣿⣿" "⣿⣆",
        "             ⢠" "⣿⣿⣿⣿⡟ ⠈⣿⣿⠃ ⢻⣿⣿" "⣿⣿⡄",
        "            ⢠⣿" "⣿⣿⡿⠛⠁  ⣿⣿  ⠈⠻⢿" "⣿⣿⣿⡄",
        "           ⢀⣾⡿" "⠛⠁     ⣿⣿     " "⠉⠛⢿⣷⡀",
        "           ⠘⠁ " "       ⢸⡇     " "   ⠈⠃",
        "                     ⢸⡇",
        "                     ⠘⠃",
        "",
        "                Y  V  E  X",
    };

    return index < sizeof(logo) / sizeof(logo[0]) ? logo[index] : "";
}

static void startup_hero_row(const yvex_cli_terminal_style *style,
                             const char *art, const char *label,
                             const char *value, const char *tone)
{
    size_t width;

    art = art ? art : "";
    printf("  %s%s%s", style->strong, art, style->reset);
    width = startup_text_columns(art);
    while (width++ < 46u) fputc(' ', stdout);
    fputs("  ", stdout);
    if (label && label[0])
        printf("%s%-10s%s ", style->dim, label, style->reset);
    if (value && value[0])
        printf("%s%s%s", tone ? tone : "", value, style->reset);
    fputc('\n', stdout);
}

static void startup_logo_render(const yvex_cli_terminal_style *style)
{
    size_t index;

    fputc('\n', stdout);
    for (index = 0u; index < 18u; ++index)
        printf("  %s%s%s\n", style->strong, startup_logo_line(index),
               style->reset);
    fputc('\n', stdout);
}

static void startup_announce_wide(const yvex_server_options *options,
                                  const yvex_server_summary *summary,
                                  const char *endpoint,
                                  const yvex_cli_terminal_style *style,
                                  unsigned int columns)
{
    char engines[96], workers[64], protocol[96];
    char local[YVEX_SERVER_SOCKET_PATH_CAP], openai[64];
    const char *state = summary ? "● READY · ATTACHED" : "● STARTING";
    int openai_enabled = summary ? summary->openai_listener_enabled
                                 : options->openai_enabled;
    unsigned int openai_port = summary ? summary->openai_port
                                       : options->openai_port;
    size_t endpoint_width = columns > 64u ? columns - 64u : 24u;

    if (summary)
        (void)snprintf(engines, sizeof(engines),
                       "%llu loaded / %llu known / %llu capacity",
                       summary->loaded_engine_count, summary->engine_count,
                       summary->maximum_engines);
    else
        (void)snprintf(engines, sizeof(engines), "0 / %u capacity",
                       (unsigned int)options->maximum_engines);
    (void)snprintf(workers, sizeof(workers), "%llu parallel",
                   summary ? summary->worker_count : options->worker_count);
    (void)snprintf(protocol, sizeof(protocol), "v%u · YVEX %s",
                   YVEX_LOCAL_PROTOCOL_VERSION, yvex_version_string());
    startup_tail_fit(local, sizeof(local), endpoint, endpoint_width);
    if (openai_enabled)
        (void)snprintf(openai, sizeof(openai), "127.0.0.1:%u · loopback",
                       openai_port);
    else
        (void)snprintf(openai, sizeof(openai), "disabled");

    fputc('\n', stdout);
    startup_hero_row(style, startup_logo_line(0u), NULL,
                     "YVEX SERVER · PERSISTENT HOST",
                     style->accent);
    startup_hero_row(style, startup_logo_line(1u), NULL,
                     "native verified inference command center",
                     style->dim);
    startup_hero_row(style, startup_logo_line(2u), NULL, NULL, NULL);
    startup_hero_row(style, startup_logo_line(3u), "STATE", state,
                     summary ? style->success : style->warning);
    startup_hero_row(style, startup_logo_line(4u), "ENGINES", engines,
                     style->strong);
    startup_hero_row(style, startup_logo_line(5u), "WORKERS", workers,
                     style->strong);
    startup_hero_row(style, startup_logo_line(6u), "PROTOCOL", protocol,
                     style->strong);
    startup_hero_row(style, startup_logo_line(7u), "CONSOLE",
                     summary ? "human · attached" : "human · interactive",
                     style->strong);
    startup_hero_row(style, startup_logo_line(8u), NULL, NULL, NULL);
    startup_hero_row(style, startup_logo_line(9u), "LOCAL IPC", local,
                     style->strong);
    startup_hero_row(style, startup_logo_line(10u), "OPENAI", openai,
                     openai_enabled ? style->success : style->dim);
    startup_hero_row(style, startup_logo_line(11u), "ACCESS",
                     "external clients enabled", style->success);
    startup_hero_row(style, startup_logo_line(12u), NULL, NULL, NULL);
    startup_hero_row(style, startup_logo_line(13u), "CONTROL",
                     "profiles · load · models",
                     style->strong);
    startup_hero_row(style, startup_logo_line(14u), "OPERATE",
                     "status · help · unload",
                     style->strong);
    startup_hero_row(style, startup_logo_line(15u), "SHUTDOWN", "stop",
                     style->warning);
    startup_hero_row(style, startup_logo_line(16u), NULL, NULL, NULL);
    startup_hero_row(style, startup_logo_line(17u), "HELP",
                     "type `help` for the full command map", style->dim);
}

static void startup_announce_compact(const yvex_server_options *options,
                                     const yvex_server_summary *summary,
                                     const char *endpoint, int interactive,
                                     const yvex_cli_terminal_style *style)
{
    printf("%sYVEX server%s · persistent host\n"
           "%snative verified inference%s · YVEX %s · protocol %u\n",
           style->strong, style->reset, style->dim, style->reset,
           yvex_version_string(), YVEX_LOCAL_PROTOCOL_VERSION);
    if (summary)
        printf("  engines %llu loaded/%llu known/%llu max · parallel workers=%llu\n",
               summary->loaded_engine_count, summary->engine_count,
               summary->maximum_engines, summary->worker_count);
    else
        printf("  engines 0/%llu · parallel workers=%llu\n",
               options->maximum_engines, options->worker_count);
    printf("  local endpoint %s", endpoint ? endpoint : "unavailable");
    if (summary ? summary->openai_listener_enabled : options->openai_enabled)
        printf(" · OpenAI 127.0.0.1:%u",
               summary ? (unsigned int)summary->openai_port
                       : (unsigned int)options->openai_port);
    else
        printf(" · OpenAI disabled");
    if (summary && interactive)
        printf("\n  server already active · attaching this console\n");
    else if (summary)
        printf("\n  server already active · host configuration is unchanged\n");
    else if (interactive)
        printf("\n  manage this host below · external clients remain available\n");
    else
        printf("\n  load with `yvex server load MODEL`"
               " · stop with Ctrl-C or `yvex server stop`\n");
}

static void startup_announce(const yvex_server_options *options,
                             const yvex_server_summary *summary,
                             int interactive)
{
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    yvex_cli_terminal_style style;
    unsigned int columns;
    yvex_error err;
    const char *endpoint = summary ? summary->socket_path : options->socket_path;
    if (!endpoint && yvex_server_socket_path(socket_path, &err) == YVEX_OK)
        endpoint = socket_path;
    yvex_cli_terminal_style_get(stdout, &style);
    columns = startup_terminal_columns();
    if (interactive && columns >= 104u)
        startup_announce_wide(options, summary, endpoint, &style, columns);
    else {
        if (interactive) startup_logo_render(&style);
        startup_announce_compact(options, summary, endpoint, interactive, &style);
    }
    (void)fflush(stdout);
}

int yvex_cli_server_dispatch(int argc, char **argv, size_t consumed)
{
    cli_server_loader_context loader = {0};
    yvex_server_summary attached_summary;
    yvex_server *server = NULL;
    cli_server_thread_state thread_state;
    pthread_t signal_thread, console_thread, operator_thread;
    sigset_t signals;
    yvex_error err;
    int rc, signal_ready = 0, console_ready = 0, operator_ready = 0;
    int interactive;
    host_options_defaults(&loader.host);
    if (!command_options_parse(&loader, argc, argv, consumed))
        return 2;
    if (pthread_mutex_init(&loader.registry_mutex, NULL) != 0) {
        fprintf(stderr, "yvex server: model registry coordinator creation failed\n");
        return 1;
    }
    loader.host.model_loader = registered_model_load;
    loader.host.model_loader_context = &loader;
    interactive = loader.host.console == YVEX_SERVER_CONSOLE_HUMAN &&
                  isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
    memset(&attached_summary, 0, sizeof(attached_summary));
    yvex_error_clear(&err);
    rc = console_remote_summary(
        loader.host.socket_path, &attached_summary, &err);
    if (rc == YVEX_OK) {
        startup_announce(&loader.host, &attached_summary, interactive);
        if (interactive) {
            memset(&thread_state, 0, sizeof(thread_state));
            thread_state.loader = &loader;
            thread_state.socket_path = attached_summary.socket_path;
            thread_state.attached = 1;
            (void)operator_console_main(&thread_state);
        }
        (void)pthread_mutex_destroy(&loader.registry_mutex);
        return 0;
    }
    if (rc != YVEX_ERR_IO) {
        (void)pthread_mutex_destroy(&loader.registry_mutex);
        return server_error(&err, 1);
    }
    startup_announce(&loader.host, NULL, interactive);
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGTERM);
    (void)pthread_sigmask(SIG_BLOCK, &signals, NULL);
    yvex_error_clear(&err);
    rc = yvex_server_create(&server, &loader.host, &err);
    if (rc == YVEX_OK) rc = yvex_server_start(server, &err);
    if (rc != YVEX_OK) {
        yvex_server_close(&server);
        (void)pthread_mutex_destroy(&loader.registry_mutex);
        return server_error(&err, 1);
    }
    memset(&thread_state, 0, sizeof(thread_state));
    thread_state.server = server;
    thread_state.loader = &loader;
    thread_state.socket_path = loader.host.socket_path;
    if (pthread_create(&signal_thread, NULL, signal_main, &thread_state) == 0)
        signal_ready = 1;
    else {
        yvex_server_close(&server);
        (void)pthread_mutex_destroy(&loader.registry_mutex);
        fprintf(stderr, "yvex server: signal coordinator creation failed\n");
        return 1;
    }
    if (loader.host.console != YVEX_SERVER_CONSOLE_OFF) {
        void *(*console_main)(void *) =
            loader.host.console == YVEX_SERVER_CONSOLE_RAW
                ? raw_console_main : human_console_main;
        if (pthread_create(&console_thread, NULL, console_main, &thread_state) == 0)
            console_ready = 1;
        else {
            (void)yvex_server_stop(server, &err);
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc == YVEX_OK && interactive) {
        if (pthread_create(&operator_thread, NULL, operator_console_main,
                           &thread_state) == 0)
            operator_ready = 1;
        else {
            (void)yvex_server_stop(server, &err);
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc == YVEX_OK) rc = yvex_server_serve(server, &err);
    (void)yvex_server_stop(server, &err);
    if (signal_ready) {
        (void)pthread_kill(signal_thread, SIGTERM);
        (void)pthread_join(signal_thread, NULL);
    }
    if (operator_ready) (void)pthread_join(operator_thread, NULL);
    if (yvex_server_finish(server, &err) != YVEX_OK && rc == YVEX_OK)
        rc = yvex_error_code(&err);
    if (console_ready) (void)pthread_join(console_thread, NULL);
    yvex_server_close(&server);
    (void)pthread_mutex_destroy(&loader.registry_mutex);
    return rc == YVEX_OK ? 0 : server_error(&err, 1);
}
