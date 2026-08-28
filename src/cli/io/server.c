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
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    yvex_server *server;
} cli_server_thread_state;

typedef struct {
    yvex_server_options host;
} cli_server_loader_context;

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
    options->schema_version = YVEX_SERVER_ENGINE_SCHEMA_V1;
    options->alias = profile->name;
    options->artifact_path = profile->artifact;
    options->runtime_binding_path = profile->binding;
    options->target_id = profile->target;
    options->backend = media_requested || !strcmp(profile->backend, "cuda")
                           ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU;
    options->generation_mode = media_requested
                                   ? YVEX_SERVER_GENERATION_MEDIA
                                   : (!strcmp(profile->mode, "dspark")
                                          ? YVEX_SERVER_GENERATION_DSPARK
                                          : YVEX_SERVER_GENERATION_TARGET_ONLY);
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
    rc = profile_resolve(alias, &profile, &media_requested, err);
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

static void startup_announce(const yvex_server_options *options)
{
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    yvex_cli_terminal_style style;
    yvex_error err;
    const char *endpoint = options->socket_path;
    if (!endpoint && yvex_server_socket_path(socket_path, &err) == YVEX_OK)
        endpoint = socket_path;
    yvex_cli_terminal_style_get(stdout, &style);
    printf("%s"
           "__   __  __     __  ______  __   __\n"
           "\\ \\ / /  \\ \\   / / |  ____| \\ \\ / /\n"
           " \\ V /    \\ \\ / /  | |__     \\ V /\n"
           "  | |      \\ V /   |  __|     > <\n"
           "  | |       \\ /    | |____   / . \\\n"
           "  |_|        V     |______| /_/ \\_\\\n"
           "%snative verified inference · YVEX %s · protocol %u\n\n"
           "YVEX server · persistent host\n"
           "  engines 0/%u · parallel workers=%llu\n"
           "  local endpoint %s",
           style.strong, style.reset, yvex_version_string(),
           YVEX_LOCAL_PROTOCOL_VERSION, (unsigned int)options->maximum_engines,
           options->worker_count,
           endpoint ? endpoint : "unavailable");
    if (options->openai_enabled)
        printf(" · OpenAI 127.0.0.1:%u", (unsigned int)options->openai_port);
    else
        printf(" · OpenAI disabled");
    printf("\n  load with `yvex server load MODEL`"
           " · stop with Ctrl-C or `yvex server stop`\n");
    (void)fflush(stdout);
}

int yvex_cli_server_dispatch(int argc, char **argv, size_t consumed)
{
    cli_server_loader_context loader = {0};
    yvex_server *server = NULL;
    cli_server_thread_state thread_state;
    pthread_t signal_thread, console_thread;
    sigset_t signals;
    yvex_error err;
    int rc, signal_ready = 0, console_ready = 0;
    host_options_defaults(&loader.host);
    if (!command_options_parse(&loader, argc, argv, consumed))
        return 2;
    loader.host.model_loader = registered_model_load;
    loader.host.model_loader_context = &loader;
    startup_announce(&loader.host);
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGTERM);
    (void)pthread_sigmask(SIG_BLOCK, &signals, NULL);
    yvex_error_clear(&err);
    rc = yvex_server_create(&server, &loader.host, &err);
    if (rc == YVEX_OK) rc = yvex_server_start(server, &err);
    if (rc != YVEX_OK) {
        yvex_server_close(&server);
        return server_error(&err, 1);
    }
    memset(&thread_state, 0, sizeof(thread_state));
    thread_state.server = server;
    if (pthread_create(&signal_thread, NULL, signal_main, &thread_state) == 0)
        signal_ready = 1;
    else {
        yvex_server_close(&server);
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
    if (rc == YVEX_OK) rc = yvex_server_serve(server, &err);
    (void)yvex_server_stop(server, &err);
    if (signal_ready) {
        (void)pthread_kill(signal_thread, SIGTERM);
        (void)pthread_join(signal_thread, NULL);
    }
    if (yvex_server_finish(server, &err) != YVEX_OK && rc == YVEX_OK)
        rc = yvex_error_code(&err);
    if (console_ready) (void)pthread_join(console_thread, NULL);
    yvex_server_close(&server);
    return rc == YVEX_OK ? 0 : server_error(&err, 1);
}
