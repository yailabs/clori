/* Run the persistent foreground host and resolve explicit engine loads from the local registry. */
#define _POSIX_C_SOURCE 200809L

#include <yvex/registry.h>
#include <yvex/server.h>
#include <yvex/internal/core.h>
#include <yvex/internal/deployment_compatibility.h>
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
    char engine_kind[16];
    char execution_strategy[32];
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
        !entry->runtime_target || !entry->runtime_backend ||
        !entry->runtime_engine_kind || !entry->runtime_execution_strategy ||
        strlen(entry->alias) >= sizeof(profile->name) ||
        strlen(entry->family) >= sizeof(profile->family) ||
        strlen(entry->runtime_target) >= sizeof(profile->target) ||
        strlen(entry->runtime_backend) >= sizeof(profile->backend) ||
        strlen(entry->runtime_engine_kind) >= sizeof(profile->engine_kind) ||
        strlen(entry->runtime_execution_strategy) >=
            sizeof(profile->execution_strategy))
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
           snprintf(profile->engine_kind, sizeof(profile->engine_kind), "%s",
                    entry->runtime_engine_kind) >= 0 &&
           snprintf(profile->execution_strategy,
                    sizeof(profile->execution_strategy), "%s",
                    entry->runtime_execution_strategy) >= 0;
}

static int profile_resolve(const char *name, cli_server_profile *profile,
                           yvex_error *err)
{
    yvex_model_registry_options options;
    yvex_model_registry *registry = NULL;
    const yvex_model_registry_entry *entry;
    yvex_deployment_compatibility compatibility = {0};
    int rc;
    memset(&options, 0, sizeof(options));
    memset(profile, 0, sizeof(*profile));
    rc = yvex_model_registry_open(&registry, &options, err);
    if (rc != YVEX_OK) return rc;
    entry = yvex_model_registry_find(registry, name);
    if (!entry) {
        yvex_error_setf(err, YVEX_ERR_STATE, "server.model-loader",
                        "profile is not registered: %s", name);
        yvex_model_registry_close(registry);
        return YVEX_ERR_STATE;
    }
    rc = yvex_deployment_compatibility_evaluate(entry, &compatibility, err);
    if (rc == YVEX_OK && !compatibility.current) {
        yvex_error_setf(err, YVEX_ERR_STATE, "server.model-loader",
                        "deployment is not current (%s): %s",
                        yvex_deployment_compatibility_status_name(
                            compatibility.status),
                        compatibility.reason);
        rc = YVEX_ERR_STATE;
    }
    if (rc != YVEX_OK) {
        yvex_model_registry_close(registry);
        return rc;
    }
    if (!profile_copy(profile, entry)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.model-loader",
                       "registered profile exceeds command limits");
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
    fprintf(stderr, "yvex serve: %s: %s\n", yvex_error_where(err),
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

static void *raw_log_main(void *opaque)
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

static void *human_log_main(void *opaque)
{
    cli_server_thread_state *state = opaque;
    yvex_cli_watch_renderer renderer;
    unsigned long long cursor = 0u;
    yvex_cli_watch_renderer_open(&renderer, 0);
    for (;;) {
        yvex_server_event event;
        yvex_server_summary summary;
        const yvex_server_summary *live = NULL;
        yvex_error err;
        if (yvex_server_event_next(state->server, cursor, 1, &event, &err) != YVEX_OK)
            continue;
        cursor = event.sequence;
        if (event.kind == YVEX_SERVER_EVENT_GENERATION_PROGRESS &&
            yvex_server_get_summary(state->server, &summary, &err) == YVEX_OK)
            live = &summary;
        (void)yvex_cli_watch_renderer_event(&renderer, &event, live);
        if (event.kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE) break;
    }
    yvex_cli_watch_renderer_finish(&renderer);
    return NULL;
}

static void engine_profile_defaults(yvex_server_engine_options *options,
                                    const cli_server_profile *profile)
{
    int media_requested = !strcmp(profile->engine_kind, "media");

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
                        : (!strcmp(profile->execution_strategy, "speculative")
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
    else if (!strcmp(flag, "--logs")) {
        if (!strcmp(value, "human")) host->console = YVEX_SERVER_CONSOLE_HUMAN;
        else if (!strcmp(value, "json")) host->console = YVEX_SERVER_CONSOLE_RAW;
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
            fprintf(stderr, "yvex serve: invalid option: %s\n", flag);
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
    adapter = yvex_graph_component_variant_find(profile->target);
    if (!adapter || !adapter->media_target_profile || !adapter->media_execution) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "server.media-profile",
                        "target has no conversational media adapter: %s", profile->target);
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
    media->schema_version = YVEX_SERVER_MEDIA_SCHEMA_V2;
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
    media->released_sigma_grid_points = host->released_sigma_grid_points;
    media->default_seed = host->default_seed;
    media->canvas_multiple = host->canvas_multiple;
    media->canvas_short_edge = host->canvas_short_edge;
    media->minimum_canvas_pixels = host->minimum_canvas_pixels;
    media->maximum_canvas_pixels = host->maximum_canvas_pixels;
    media->released_width = host->released_width;
    media->released_height = host->released_height;
    media->minimum_duration_milliseconds = host->minimum_duration_milliseconds;
    media->maximum_duration_milliseconds = host->maximum_duration_milliseconds;
    media->minimum_aspect_numerator = host->minimum_aspect_numerator;
    media->minimum_aspect_denominator = host->minimum_aspect_denominator;
    media->maximum_aspect_numerator = host->maximum_aspect_numerator;
    media->maximum_aspect_denominator = host->maximum_aspect_denominator;
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
    rc = profile_resolve(alias, &profile, err);
    (void)pthread_mutex_unlock(&context->registry_mutex);
    if (rc != YVEX_OK) return rc;
    media_requested = !strcmp(profile.engine_kind, "media");
    engine_profile_defaults(&selected, &profile);
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

static int server_remote_summary(
    const char *socket_path, yvex_server_summary *summary, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc;

    yvex_cli_client_request_init(&request, YVEX_CLIENT_OP_RUNTIME_STATUS);
    rc = yvex_client_connect(&client, socket_path, err);
    if (rc == YVEX_OK) rc = yvex_client_send(client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
        yvex_error_set(err, (yvex_status)message.status, "client.status",
                       message.reason[0] ? message.reason : "server request failed");
        rc = message.status;
    }
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
    /* High-contrast silhouette sampled from the canonical butterfly mark. */
    static const char *const logo[] = {
        " ██▄▄            ▄   ██   ▄            ▄▄██",
        "   ▀███▄▄         █▄ ▄  ▄█         ▄▄███▀",
        "    ▀███████▄      ▀▄▀▀▄▀      ▄███████▀",
        "       ▀███████▄▄    ██▀   ▄▄███████▀▀",
        "        ▀██████████▄▄██▄▄██████████▀",
        "          ▀██████████████████████▀",
        "            ▀▀▀██████████████▀▀▀",
        "              ▄██ ████ ██▄",
        "            ▄███▀  ██  ▀███▄",
        "          ▄███▀    ██    ▀███▄",
        "        ▄██▀       ██       ▀██▄",
        "       ▀            ██            ▀",
        "                    ▀▀",
        "                 Y V E X",
    };

    return index < sizeof(logo) / sizeof(logo[0]) ? logo[index] : "";
}

static const char *startup_logo_wide_line(size_t index)
{
    return startup_logo_line(index);
}

static const char *startup_logo_compact_line(size_t index)
{
    /* Smaller solid silhouette for narrow terminals. */
    static const char *const logo[] = {
        " ▄                              ▄",
        " ▀██▄▄       ▀▄ ▀▀  ▀       ▄▄██▀",
        "   ▀████▄▄    ▀▄  ▄▀    ▄▄████▀",
        "     ▀█████▄▄   ██   ▄▄█████▀",
        "      ▀████████████████████▀",
        "        ▀████████████████▀",
        "           ▄█ ████ █▄",
        "         ▄██▀  ██  ▀██▄",
        "       ▄██▀    ██    ▀██▄",
        "      ▀        ██        ▀",
        "               ▀▀",
        "            Y V E X",
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
    for (index = 0u; index < 12u; ++index)
        printf("  %s%s%s\n", style->strong,
               startup_logo_compact_line(index),
               style->reset);
    fputc('\n', stdout);
}

static void startup_announce_wide(const yvex_server_options *options,
                                  const char *endpoint,
                                  const yvex_cli_terminal_style *style,
                                  unsigned int columns)
{
    char capacity[96], protocol[96];
    char local[YVEX_SERVER_SOCKET_PATH_CAP], openai[64];
    size_t endpoint_width = columns > 64u ? columns - 64u : 24u;

    (void)snprintf(capacity, sizeof(capacity), "0/%u engines · %llu worker%s",
                   (unsigned int)options->maximum_engines, options->worker_count,
                   options->worker_count == 1ull ? "" : "s");
    (void)snprintf(protocol, sizeof(protocol), "v%u · YVEX %s",
                   YVEX_LOCAL_PROTOCOL_VERSION, yvex_version_string());
    startup_tail_fit(local, sizeof(local), endpoint, endpoint_width);
    if (options->openai_enabled)
        (void)snprintf(openai, sizeof(openai), "127.0.0.1:%u · loopback",
                       options->openai_port);
    else
        (void)snprintf(openai, sizeof(openai), "disabled");

    fputc('\n', stdout);
    startup_hero_row(style, startup_logo_wide_line(0u), NULL,
                     "YVEX HOST", style->accent);
    startup_hero_row(style, startup_logo_wide_line(1u), NULL,
                     "verified inference runtime", style->dim);
    startup_hero_row(style, startup_logo_wide_line(2u), "STATE", "● STARTING",
                     style->warning);
    startup_hero_row(style, startup_logo_wide_line(3u), "CAPACITY", capacity,
                     style->strong);
    startup_hero_row(style, startup_logo_wide_line(4u), "PROTOCOL", protocol,
                     style->strong);
    startup_hero_row(style, startup_logo_wide_line(5u), "NATIVE", local,
                     style->strong);
    startup_hero_row(style, startup_logo_wide_line(6u), "OPENAI", openai,
                     options->openai_enabled ? style->success : style->dim);
    startup_hero_row(style, startup_logo_wide_line(7u), "EVENTS",
                     "lifecycle · progress · resources", style->strong);
    startup_hero_row(style, startup_logo_wide_line(8u), NULL, NULL, NULL);
    startup_hero_row(style, startup_logo_wide_line(9u), NULL, NULL, NULL);
    startup_hero_row(style, startup_logo_wide_line(10u), NULL, NULL, NULL);
    startup_hero_row(style, startup_logo_wide_line(11u), NULL, NULL, NULL);
    startup_hero_row(style, startup_logo_wide_line(12u), NULL, NULL, NULL);
    startup_hero_row(style, startup_logo_wide_line(13u), NULL, NULL, NULL);
}

static void startup_announce_compact(const yvex_server_options *options,
                                     const char *endpoint, int human_terminal,
                                     const yvex_cli_terminal_style *style)
{
    printf("%sYVEX HOST%s · verified inference runtime\n"
           "%sYVEX %s · protocol %u%s\n",
           style->strong, style->reset, style->dim, yvex_version_string(),
           YVEX_LOCAL_PROTOCOL_VERSION, style->reset);
    printf("  0/%llu engines · %llu worker%s\n", options->maximum_engines,
           options->worker_count, options->worker_count == 1ull ? "" : "s");
    printf("  native %s", endpoint ? endpoint : "unavailable");
    if (options->openai_enabled)
        printf(" · OpenAI 127.0.0.1:%u",
               (unsigned int)options->openai_port);
    else
        printf(" · OpenAI disabled");
    printf("\n  events lifecycle · progress · resources");
    if (!human_terminal) printf(" · Ctrl-C to stop");
    putchar('\n');
}

static void startup_announce(const yvex_server_options *options,
                             int human_terminal)
{
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    yvex_cli_terminal_style style;
    unsigned int columns;
    yvex_error err;
    const char *endpoint = options->socket_path;
    if (!endpoint && yvex_server_socket_path(socket_path, &err) == YVEX_OK)
        endpoint = socket_path;
    yvex_cli_terminal_style_get(stdout, &style);
    columns = startup_terminal_columns();
    if (human_terminal && columns >= 104u)
        startup_announce_wide(options, endpoint, &style, columns);
    else {
        if (human_terminal) startup_logo_render(&style);
        startup_announce_compact(options, endpoint, human_terminal, &style);
    }
    (void)fflush(stdout);
}

int yvex_cli_server_dispatch(int argc, char **argv, size_t consumed)
{
    cli_server_loader_context loader = {0};
    yvex_server_summary attached_summary;
    yvex_server *server = NULL;
    cli_server_thread_state thread_state;
    pthread_t signal_thread, log_thread;
    sigset_t signals;
    yvex_error err;
    int rc, signal_ready = 0, log_ready = 0;
    int human_terminal;
    host_options_defaults(&loader.host);
    if (!command_options_parse(&loader, argc, argv, consumed))
        return 2;
    if (pthread_mutex_init(&loader.registry_mutex, NULL) != 0) {
        fprintf(stderr, "yvex serve: profile registry coordinator creation failed\n");
        return 1;
    }
    loader.host.model_loader = registered_model_load;
    loader.host.model_loader_context = &loader;
    human_terminal = loader.host.console == YVEX_SERVER_CONSOLE_HUMAN &&
                     isatty(STDOUT_FILENO);
    memset(&attached_summary, 0, sizeof(attached_summary));
    yvex_error_clear(&err);
    rc = server_remote_summary(
        loader.host.socket_path, &attached_summary, &err);
    if (rc == YVEX_OK) {
        fprintf(stderr, "yvex: host already running\n\n"
                        "  socket:  %s\n\n"
                        "  inspect:\n"
                        "    yvex host status\n",
                attached_summary.socket_path[0] ? attached_summary.socket_path
                                                : "authoritative default");
        (void)pthread_mutex_destroy(&loader.registry_mutex);
        return 1;
    }
    if (rc != YVEX_ERR_IO) {
        (void)pthread_mutex_destroy(&loader.registry_mutex);
        return server_error(&err, 1);
    }
    startup_announce(&loader.host, human_terminal);
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
    puts("host ready · Ctrl-C to stop");
    (void)fflush(stdout);
    memset(&thread_state, 0, sizeof(thread_state));
    thread_state.server = server;
    if (pthread_create(&signal_thread, NULL, signal_main, &thread_state) == 0)
        signal_ready = 1;
    else {
        yvex_server_close(&server);
        (void)pthread_mutex_destroy(&loader.registry_mutex);
        fprintf(stderr, "yvex serve: signal coordinator creation failed\n");
        return 1;
    }
    if (loader.host.console != YVEX_SERVER_CONSOLE_OFF) {
        void *(*log_main)(void *) =
            loader.host.console == YVEX_SERVER_CONSOLE_RAW
                ? raw_log_main : human_log_main;
        if (pthread_create(&log_thread, NULL, log_main, &thread_state) == 0)
            log_ready = 1;
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
    if (log_ready) (void)pthread_join(log_thread, NULL);
    yvex_server_close(&server);
    (void)pthread_mutex_destroy(&loader.registry_mutex);
    return rc == YVEX_OK ? 0 : server_error(&err, 1);
}
