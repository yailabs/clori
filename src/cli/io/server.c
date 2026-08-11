/*
 * Run the foreground model server selected by one explicit registry profile.
 *
 * The public command and the process lifecycle are the same operation: this owner resolves the
 * named startup profile, admits bounded overrides, starts the server resources, and remains in
 * the foreground until shutdown. It never consults implicit persisted model selection.
 */
#define _POSIX_C_SOURCE 200809L

#include <yvex/registry.h>
#include <yvex/server.h>

#include "src/cli/private.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char name[256];
    char artifact[PATH_MAX];
    char binding[PATH_MAX];
    char target[128];
    char backend[8];
    char mode[16];
    unsigned long long context_capacity;
} cli_server_profile;

typedef struct {
    yvex_server *server;
} cli_server_thread_state;

typedef struct {
    atomic_int done;
    pthread_t thread;
    struct timespec started;
    int thread_ready;
} cli_server_startup_progress;

static unsigned long long startup_elapsed_seconds(
    const cli_server_startup_progress *progress)
{
    struct timespec now;
    if (!progress || clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
        now.tv_sec < progress->started.tv_sec)
        return 0u;
    return (unsigned long long)(now.tv_sec - progress->started.tv_sec);
}

static void *startup_progress_main(void *opaque)
{
    cli_server_startup_progress *progress = opaque;
    const struct timespec interval = {0, 100000000};
    unsigned long long last_report = 0u;
    while (!atomic_load_explicit(&progress->done, memory_order_relaxed)) {
        unsigned long long elapsed;
        (void)nanosleep(&interval, NULL);
        elapsed = startup_elapsed_seconds(progress);
        if (!atomic_load_explicit(&progress->done, memory_order_relaxed) &&
            elapsed >= last_report + 10u) {
            fprintf(stderr,
                    "yvex server: model admission still in progress (elapsed %llu s)\n",
                    elapsed);
            (void)fflush(stderr);
            last_report = elapsed;
        }
    }
    return NULL;
}

static void startup_progress_begin(cli_server_startup_progress *progress)
{
    memset(progress, 0, sizeof(*progress));
    atomic_init(&progress->done, 0);
    fprintf(stderr,
            "yvex server: model admission in progress (elapsed 0 s); "
            "readiness follows verification and residency\n");
    (void)fflush(stderr);
    if (clock_gettime(CLOCK_MONOTONIC, &progress->started) == 0 &&
        pthread_create(&progress->thread, NULL, startup_progress_main, progress) == 0)
        progress->thread_ready = 1;
    else {
        fprintf(stderr,
                "yvex server: periodic startup progress unavailable; admission continues\n");
        (void)fflush(stderr);
    }
}

static void startup_progress_end(cli_server_startup_progress *progress, int status)
{
    unsigned long long elapsed;
    atomic_store_explicit(&progress->done, 1, memory_order_relaxed);
    if (progress->thread_ready) (void)pthread_join(progress->thread, NULL);
    elapsed = startup_elapsed_seconds(progress);
    fprintf(stderr, "yvex server: model admission %s (elapsed %llu s)%s\n",
            status == YVEX_OK ? "complete" : "failed", elapsed,
            status == YVEX_OK ? "; local server ready" : "");
    (void)fflush(stderr);
}

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
    if (!profile || !entry || strlen(entry->alias) >= sizeof(profile->name) ||
        strlen(entry->path) >= sizeof(profile->artifact) ||
        strlen(entry->runtime_binding) >= sizeof(profile->binding) ||
        strlen(entry->runtime_target) >= sizeof(profile->target) ||
        strlen(entry->runtime_backend) >= sizeof(profile->backend) ||
        strlen(entry->runtime_mode) >= sizeof(profile->mode))
        return 0;
    return snprintf(profile->name, sizeof(profile->name), "%s", entry->alias) > 0 &&
           snprintf(profile->artifact, sizeof(profile->artifact), "%s", entry->path) > 0 &&
           snprintf(profile->binding, sizeof(profile->binding), "%s",
                    entry->runtime_binding) > 0 &&
           snprintf(profile->target, sizeof(profile->target), "%s",
                    entry->runtime_target) > 0 &&
           snprintf(profile->backend, sizeof(profile->backend), "%s",
                    entry->runtime_backend) > 0 &&
           snprintf(profile->mode, sizeof(profile->mode), "%s",
                    entry->runtime_mode) > 0;
}

static int profile_resolve(const char *name, cli_server_profile *profile)
{
    static const char *const controls[] = {
        "status", "model", "memory", "log", "stop"
    };
    yvex_model_registry_options options;
    yvex_model_registry *registry = NULL;
    const yvex_model_registry_entry *entry;
    yvex_error err;
    size_t control;
    int rc;
    memset(&options, 0, sizeof(options));
    memset(profile, 0, sizeof(*profile));
    yvex_error_clear(&err);
    rc = yvex_model_registry_open(&registry, &options, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "yvex server: model registry is unavailable: %s\n"
                "hint: register a startup profile with `yvex model registry add --help`\n",
                yvex_error_message(&err));
        return 1;
    }
    entry = yvex_model_registry_find(registry, name);
    if (!entry) {
        for (control = 0u; control < sizeof(controls) / sizeof(controls[0]); ++control) {
            if (yvex_cli_command_distance(name, controls[control]) <= 1u) {
                fprintf(stderr,
                        "yvex: unknown command: server %s\n"
                        "hint: did you mean `yvex server %s`?\n",
                        name, controls[control]);
                yvex_model_registry_close(registry);
                return 2;
            }
        }
        fprintf(stderr,
                "yvex server: model is not registered: %s\n"
                "hint: inspect available profiles with `yvex model list`\n",
                name);
        yvex_model_registry_close(registry);
        return 1;
    }
    rc = yvex_model_registry_startup_validate(entry, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "yvex server: model cannot start: %s\n"
                "hint: `yvex model show %s` reports its startup profile\n",
                yvex_error_message(&err), name);
        yvex_model_registry_close(registry);
        return 1;
    }
    if (!profile_copy(profile, entry)) {
        fprintf(stderr, "yvex server: registered startup profile exceeds command limits\n");
        yvex_model_registry_close(registry);
        return 1;
    }
    profile->context_capacity = entry->runtime_context;
    yvex_model_registry_close(registry);
    return 0;
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

static void options_defaults(yvex_server_options *options,
                             const cli_server_profile *profile)
{
    memset(options, 0, sizeof(*options));
    options->schema_version = YVEX_SERVER_OPTIONS_SCHEMA_V2;
    options->artifact_path = profile->artifact;
    options->runtime_binding_path = profile->binding;
    options->target_id = profile->target;
    options->backend = !strcmp(profile->backend, "cuda")
                           ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU;
    options->generation_mode = !strcmp(profile->mode, "dspark")
                                   ? YVEX_SERVER_GENERATION_DSPARK
                                   : YVEX_SERVER_GENERATION_TARGET_ONLY;
    options->context_capacity = profile->context_capacity;
    options->prefill_chunk_tokens = 64u;
    options->maximum_new_tokens = 256u;
    options->maximum_output_bytes = 1048576u;
    options->maximum_sessions = 8u;
    options->request_queue_capacity = 16u;
    options->concurrent_sequences = 1u;
    options->trace_level = YVEX_SERVER_TRACE_STAGES;
    options->openai_enabled = 1;
    options->openai_port = 8001u;
    options->openai_timeout_ms = 600000u;
}

static int option_parse(yvex_server_options *options, const char *flag,
                        const char *value)
{
    if (!strcmp(flag, "--socket")) options->socket_path = value;
    else if (!strcmp(flag, "--backend"))
        options->backend = !strcmp(value, "cuda") ? YVEX_BACKEND_KIND_CUDA
                                                    : YVEX_BACKEND_KIND_CPU;
    else if (!strcmp(flag, "--generation-mode"))
        options->generation_mode = !strcmp(value, "dspark")
                                       ? YVEX_SERVER_GENERATION_DSPARK
                                       : YVEX_SERVER_GENERATION_TARGET_ONLY;
    else if (!strcmp(flag, "--ctx"))
        return parse_u64(value, &options->context_capacity);
    else if (!strcmp(flag, "--prefill-chunk"))
        return parse_u64(value, &options->prefill_chunk_tokens);
    else if (!strcmp(flag, "--max-new-tokens"))
        return parse_u64(value, &options->maximum_new_tokens);
    else if (!strcmp(flag, "--parallel")) {
        if (!parse_u64(value, &options->concurrent_sequences)) return 0;
        if (options->maximum_sessions < options->concurrent_sequences)
            options->maximum_sessions = options->concurrent_sequences;
    }
    else if (!strcmp(flag, "--console"))
        options->console = !strcmp(value, "raw") ? YVEX_SERVER_CONSOLE_RAW
                                                  : YVEX_SERVER_CONSOLE_OFF;
    else if (!strcmp(flag, "--trace-level")) {
        if (!strcmp(value, "summary")) options->trace_level = YVEX_SERVER_TRACE_SUMMARY;
        else if (!strcmp(value, "stages")) options->trace_level = YVEX_SERVER_TRACE_STAGES;
        else if (!strcmp(value, "tokens")) options->trace_level = YVEX_SERVER_TRACE_TOKENS;
        else options->trace_level = YVEX_SERVER_TRACE_FULL;
    } else if (!strcmp(flag, "--openai"))
        options->openai_enabled = !strcmp(value, "on");
    else if (!strcmp(flag, "--openai-port")) {
        unsigned long long port;
        if (!parse_u64(value, &port) || port > 65535u) return 0;
        options->openai_port = (unsigned short)port;
    } else if (!strcmp(flag, "--openai-timeout-ms"))
        return parse_u64(value, &options->openai_timeout_ms);
    else return 0;
    return 1;
}

static int command_options_parse(yvex_server_options *options, int argc,
                                 char **argv, size_t consumed)
{
    int index;
    for (index = (int)consumed + 2; index < argc; ++index) {
        const char *flag = argv[index];
        if (!strcmp(flag, "--trace-content")) {
            options->trace_content = 1;
            continue;
        }
        if (index + 1 >= argc || !option_parse(options, flag, argv[index + 1])) {
            fprintf(stderr, "yvex server: invalid option: %s\n", flag);
            return 0;
        }
        index++;
    }
    return 1;
}

static void startup_announce(const cli_server_profile *profile,
                             const yvex_server_options *options)
{
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    yvex_error err;
    const char *endpoint = options->socket_path;
    if (!endpoint && yvex_server_socket_path(socket_path, &err) == YVEX_OK)
        endpoint = socket_path;
    printf("YVEX server · foreground\n"
           "  profile %s\n"
           "  target %s · backend=%s · mode=%s · requested ctx=%llu · parallel=%llu\n"
           "  artifact %s\n"
           "  binding %s\n"
           "  local endpoint %s",
           profile->name, profile->target,
           options->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu",
           options->generation_mode == YVEX_SERVER_GENERATION_DSPARK
               ? "dspark" : "target-only",
           options->context_capacity, options->concurrent_sequences,
           profile->artifact, profile->binding,
           endpoint ? endpoint : "unavailable");
    if (options->openai_enabled)
        printf(" · OpenAI 127.0.0.1:%u", (unsigned int)options->openai_port);
    else
        printf(" · OpenAI disabled");
    printf("\n  stop with Ctrl-C or `yvex server stop`\n");
    (void)fflush(stdout);
}

int yvex_cli_server_dispatch(int argc, char **argv, size_t consumed)
{
    const char *profile_name;
    cli_server_profile profile;
    yvex_server_options options;
    yvex_server *server = NULL;
    cli_server_thread_state thread_state;
    cli_server_startup_progress startup_progress;
    pthread_t signal_thread, console_thread;
    sigset_t signals;
    yvex_error err;
    int rc, signal_ready = 0, console_ready = 0;
    if (consumed + 1u >= (size_t)argc) return 2;
    profile_name = argv[consumed + 1u];
    rc = profile_resolve(profile_name, &profile);
    if (rc != 0) return rc;
    options_defaults(&options, &profile);
    if (!command_options_parse(&options, argc, argv, consumed)) return 2;
    startup_announce(&profile, &options);
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGTERM);
    (void)pthread_sigmask(SIG_BLOCK, &signals, NULL);
    yvex_error_clear(&err);
    rc = yvex_server_create(&server, &options, &err);
    if (rc == YVEX_OK) {
        startup_progress_begin(&startup_progress);
        rc = yvex_server_start(server, &err);
        startup_progress_end(&startup_progress, rc);
    }
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
    if (options.console == YVEX_SERVER_CONSOLE_RAW) {
        if (pthread_create(&console_thread, NULL, raw_console_main, &thread_state) == 0)
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
