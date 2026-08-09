/*
 * Configure, start, observe, serve, and gracefully close yvexd.
 *
 * One invocation creates one host and writes raw stdout only from typed event records. Process
 * entrypoint for the long-lived local runtime host.
 */
#define _POSIX_C_SOURCE 200809L

#include <yvex/server.h>
#include <yvex/internal/source.h>

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    yvex_server *server;
} daemon_thread_state;

typedef struct {
    atomic_int done;
    pthread_t thread;
    struct timespec started;
    int thread_ready;
} daemon_startup_progress;

static unsigned long long startup_elapsed_seconds(const daemon_startup_progress *progress)
{
    struct timespec now;
    if (!progress || clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
        now.tv_sec < progress->started.tv_sec)
        return 0u;
    return (unsigned long long)(now.tv_sec - progress->started.tv_sec);
}

static void *startup_progress_main(void *opaque)
{
    daemon_startup_progress *progress = opaque;
    const struct timespec interval = {0, 100000000};
    unsigned long long last_report = 0u;
    while (!atomic_load_explicit(&progress->done, memory_order_relaxed)) {
        unsigned long long elapsed;
        (void)nanosleep(&interval, NULL);
        elapsed = startup_elapsed_seconds(progress);
        if (!atomic_load_explicit(&progress->done, memory_order_relaxed) &&
            elapsed >= last_report + 10u) {
            fprintf(stderr,
                    "yvexd: model admission still in progress (elapsed %llu s)\n",
                    elapsed);
            (void)fflush(stderr);
            last_report = elapsed;
        }
    }
    return NULL;
}

static void startup_progress_begin(daemon_startup_progress *progress)
{
    memset(progress, 0, sizeof(*progress));
    atomic_init(&progress->done, 0);
    fprintf(stderr,
            "yvexd: model admission in progress (elapsed 0 s); "
            "readiness follows verification and residency\n");
    (void)fflush(stderr);
    if (clock_gettime(CLOCK_MONOTONIC, &progress->started) == 0 &&
        pthread_create(&progress->thread, NULL, startup_progress_main, progress) == 0)
        progress->thread_ready = 1;
    else {
        fprintf(stderr,
                "yvexd: periodic startup progress unavailable; model admission continues\n");
        (void)fflush(stderr);
    }
}

static void startup_progress_end(daemon_startup_progress *progress, int status)
{
    unsigned long long elapsed;
    atomic_store_explicit(&progress->done, 1, memory_order_relaxed);
    if (progress->thread_ready) (void)pthread_join(progress->thread, NULL);
    elapsed = startup_elapsed_seconds(progress);
    fprintf(stderr, "yvexd: model admission %s (elapsed %llu s)%s\n",
            status == YVEX_OK ? "complete" : "failed", elapsed,
            status == YVEX_OK ? "; runtime listener ready" : "");
    (void)fflush(stderr);
}

static void print_help(FILE *output)
{
    fprintf(output,
            "usage: yvexd --model ARTIFACT --runtime-binding FILE "
            "[--target ID] [--backend cpu|cuda] "
            "[--generation-mode target-only|dspark] [--socket PATH]\n"
            "             [--context TOKENS] [--prefill-chunk TOKENS] "
            "[--max-new-tokens N] [--console off|raw]\n"
            "             [--trace-level summary|stages|tokens|full] "
            "[--trace-content]\n"
            "             [--openai on|off] [--openai-port PORT] "
            "[--openai-timeout-ms MS]\n\n"
            "Hosts one process-resident model, a private Unix socket, and an optional "
            "loopback OpenAI listener.\n");
}

static int parse_u64(const char *text, unsigned long long *value)
{
    char *end = NULL;
    unsigned long long parsed;
    if (!text || !value) return 0;
    parsed = strtoull(text, &end, 10);
    if (!end || *end || !parsed) return 0;
    *value = parsed;
    return 1;
}

static int print_error(const yvex_error *err, int status)
{
    fprintf(stderr, "yvexd: %s: %s\n", yvex_error_where(err),
            yvex_error_message(err));
    return status;
}

static void *signal_main(void *opaque)
{
    daemon_thread_state *state = opaque;
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
    daemon_thread_state *state = opaque;
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

/* Own daemon startup through deterministic model, session, and socket shutdown. */
int main(int argument_count, char **arguments)
{
    yvex_server_options options;
    yvex_server *server = NULL;
    daemon_thread_state thread_state;
    daemon_startup_progress startup_progress;
    pthread_t signal_thread, console_thread;
    sigset_t signals;
    yvex_error err;
    int index, rc, signal_ready = 0, console_ready = 0;
    int mode_seen = 0, openai_seen = 0, openai_port_seen = 0;
    int openai_timeout_seen = 0;
    memset(&options, 0, sizeof(options));
    options.target_id = "deepseek4-v4-flash-dspark";
    options.backend = YVEX_BACKEND_KIND_CPU;
    options.generation_mode = YVEX_SERVER_GENERATION_DSPARK;
    options.context_capacity = 4096u;
    options.prefill_chunk_tokens = 64u;
    options.maximum_new_tokens = 256u;
    options.maximum_output_bytes = 1048576u;
    options.maximum_sessions = 8u;
    options.request_queue_capacity = 16u;
    options.trace_level = YVEX_SERVER_TRACE_STAGES;
    options.openai_enabled = 1;
    options.openai_port = 8001u;
    options.openai_timeout_ms = 600000u;
    for (index = 1; index < argument_count; ++index) {
        const char *argument = arguments[index];
        if (!strcmp(argument, "--help") || !strcmp(argument, "-h")) {
            print_help(stdout);
            return 0;
        } else if (!strcmp(argument, "--version")) {
            fprintf(stdout, "%s protocol=%u\n", yvex_version_string(),
                    YVEX_LOCAL_PROTOCOL_VERSION);
            return 0;
        } else if ((!strcmp(argument, "--model") ||
                    !strcmp(argument, "--runtime-binding") ||
                    !strcmp(argument, "--target") ||
                    !strcmp(argument, "--backend") ||
                    !strcmp(argument, "--generation-mode") ||
                    !strcmp(argument, "--socket") ||
                    !strcmp(argument, "--context") ||
                    !strcmp(argument, "--prefill-chunk") ||
                    !strcmp(argument, "--max-new-tokens") ||
                    !strcmp(argument, "--console") ||
                    !strcmp(argument, "--trace-level") ||
                    !strcmp(argument, "--openai") ||
                    !strcmp(argument, "--openai-port") ||
                    !strcmp(argument, "--openai-timeout-ms")) &&
                   index + 1 >= argument_count) {
            fprintf(stderr, "yvexd: %s requires a value\n", argument);
            return 2;
        } else if (!strcmp(argument, "--model")) {
            options.artifact_path = arguments[++index];
        } else if (!strcmp(argument, "--runtime-binding")) {
            options.runtime_binding_path = arguments[++index];
        } else if (!strcmp(argument, "--target")) {
            options.target_id = arguments[++index];
        } else if (!strcmp(argument, "--socket")) {
            options.socket_path = arguments[++index];
        } else if (!strcmp(argument, "--backend")) {
            const char *backend = arguments[++index];
            if (!strcmp(backend, "cpu")) options.backend = YVEX_BACKEND_KIND_CPU;
            else if (!strcmp(backend, "cuda")) options.backend = YVEX_BACKEND_KIND_CUDA;
            else {
                fprintf(stderr, "yvexd: --backend requires cpu or cuda\n");
                return 2;
            }
        } else if (!strcmp(argument, "--generation-mode")) {
            const char *mode;
            if (mode_seen++) {
                fprintf(stderr, "yvexd: duplicate --generation-mode option\n");
                return 2;
            }
            mode = arguments[++index];
            if (!strcmp(mode, "target-only"))
                options.generation_mode = YVEX_SERVER_GENERATION_TARGET_ONLY;
            else if (!strcmp(mode, "dspark"))
                options.generation_mode = YVEX_SERVER_GENERATION_DSPARK;
            else {
                fprintf(stderr,
                        "yvexd: --generation-mode requires target-only or dspark\n");
                return 2;
            }
        } else if (!strcmp(argument, "--context")) {
            if (!parse_u64(arguments[++index], &options.context_capacity)) return 2;
        } else if (!strcmp(argument, "--prefill-chunk")) {
            if (!parse_u64(arguments[++index], &options.prefill_chunk_tokens)) return 2;
        } else if (!strcmp(argument, "--max-new-tokens")) {
            if (!parse_u64(arguments[++index], &options.maximum_new_tokens)) return 2;
        } else if (!strcmp(argument, "--console")) {
            const char *console = arguments[++index];
            if (!strcmp(console, "off")) options.console = YVEX_SERVER_CONSOLE_OFF;
            else if (!strcmp(console, "raw")) options.console = YVEX_SERVER_CONSOLE_RAW;
            else return 2;
        } else if (!strcmp(argument, "--trace-level")) {
            const char *level = arguments[++index];
            if (!strcmp(level, "summary")) options.trace_level = YVEX_SERVER_TRACE_SUMMARY;
            else if (!strcmp(level, "stages")) options.trace_level = YVEX_SERVER_TRACE_STAGES;
            else if (!strcmp(level, "tokens")) options.trace_level = YVEX_SERVER_TRACE_TOKENS;
            else if (!strcmp(level, "full")) options.trace_level = YVEX_SERVER_TRACE_FULL;
            else return 2;
        } else if (!strcmp(argument, "--trace-content")) {
            options.trace_content = 1;
        } else if (!strcmp(argument, "--openai")) {
            const char *enabled;
            if (openai_seen++) {
                fprintf(stderr, "yvexd: duplicate --openai option\n");
                return 2;
            }
            enabled = arguments[++index];
            if (!strcmp(enabled, "on")) options.openai_enabled = 1;
            else if (!strcmp(enabled, "off")) options.openai_enabled = 0;
            else return 2;
        } else if (!strcmp(argument, "--openai-port")) {
            unsigned long long port;
            if (openai_port_seen++ ||
                !parse_u64(arguments[++index], &port) || port > 65535u) {
                fprintf(stderr, "yvexd: invalid or duplicate --openai-port\n");
                return 2;
            }
            options.openai_port = (unsigned short)port;
        } else if (!strcmp(argument, "--openai-timeout-ms")) {
            unsigned long long timeout;
            if (openai_timeout_seen++ ||
                !parse_u64(arguments[++index], &timeout) || timeout < 100u ||
                timeout > 86400000u) {
                fprintf(stderr, "yvexd: invalid or duplicate --openai-timeout-ms\n");
                return 2;
            }
            options.openai_timeout_ms = timeout;
        } else {
            fprintf(stderr, "yvexd: unknown option: %s\n", argument);
            return 2;
        }
    }
    if (!options.artifact_path || !options.runtime_binding_path) {
        fprintf(stderr, "yvexd: --model and --runtime-binding are required\n");
        return 2;
    }
    if (!strcmp(options.target_id, YVEX_SOURCE_RETIRED_TARGET_ID)) {
        fprintf(stderr, "yvexd: target %s was replaced; use %s\n",
                YVEX_SOURCE_RETIRED_TARGET_ID, YVEX_SOURCE_RELEASE_TARGET_ID);
        return 2;
    }
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGTERM);
    (void)pthread_sigmask(SIG_BLOCK, &signals, NULL);
    rc = yvex_server_create(&server, &options, &err);
    if (rc == YVEX_OK) {
        startup_progress_begin(&startup_progress);
        rc = yvex_server_start(server, &err);
        startup_progress_end(&startup_progress, rc);
    }
    if (rc != YVEX_OK) {
        yvex_server_close(&server);
        return print_error(&err, 1);
    }
    memset(&thread_state, 0, sizeof(thread_state));
    thread_state.server = server;
    if (pthread_create(&signal_thread, NULL, signal_main, &thread_state) == 0)
        signal_ready = 1;
    else {
        yvex_server_close(&server);
        fprintf(stderr, "yvexd: signal coordinator creation failed\n");
        return 1;
    }
    if (options.console == YVEX_SERVER_CONSOLE_RAW) {
        if (pthread_create(&console_thread, NULL, raw_console_main,
                           &thread_state) == 0)
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
    return rc == YVEX_OK ? 0 : print_error(&err, 1);
}
