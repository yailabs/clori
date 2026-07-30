/* Owner: daemon.yvexd.
 * Owns: daemon argument admission, signals, raw-console projection, and host lifecycle.
 * Does not own: model admission, session semantics, protocol framing, event identity, or client UX.
 * Invariants: one invocation creates one host and writes raw stdout only from typed event records.
 * Boundary: process entrypoint for the long-lived local runtime host.
 * Purpose: configure, start, observe, serve, and gracefully close yvexd.
 * Inputs: explicit artifact/binding/backend/budgets and operating-system signals.
 * Effects: owns process threads and delegates all runtime/socket/session resources to yvex_server.
 * Failure: concise stderr diagnostics preserve nonzero process status and close the host once. */
#define _POSIX_C_SOURCE 200809L

#include <yvex/server.h>

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    yvex_server *server;
} daemon_thread_state;

/* Purpose: render the incompatible daemon product contract. */
static void print_help(FILE *output)
{
    fprintf(output,
            "usage: yvexd --model ARTIFACT --runtime-binding FILE "
            "[--target ID] [--backend cpu|cuda] [--socket PATH]\n"
            "             [--context TOKENS] [--prefill-chunk TOKENS] "
            "[--max-new-tokens N] [--console off|raw]\n"
            "             [--trace-level summary|stages|tokens|full] "
            "[--trace-content]\n"
            "             [--openai on|off] [--openai-port PORT] "
            "[--openai-timeout-ms MS]\n\n"
            "Hosts one process-resident model, a private Unix socket, and an optional "
            "loopback OpenAI listener.\n");
}

/* Purpose: parse one positive unsigned daemon option without trailing bytes.
 * Inputs: terminated text and output. Effects: writes output only on success.
 * Failure: returns false for zero or malformed text. Boundary: owner-specific range checks follow. */
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

/* Purpose: render one typed process failure. */
static int print_error(const yvex_error *err, int status)
{
    fprintf(stderr, "yvexd: %s: %s\n", yvex_error_where(err),
            yvex_error_message(err));
    return status;
}

/* Purpose: synchronously own SIGINT/SIGTERM outside async-signal context. */
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

/* Purpose: project the canonical typed event stream to parseable JSONL stdout. */
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

/* Purpose: own daemon startup through deterministic model, session, and socket shutdown.
 * Inputs: admitted process argv and process signals. Effects: starts, serves, stops, and closes one host.
 * Failure: prints one fatal diagnostic and returns nonzero after cleanup. Boundary: runtime work is delegated. */
int main(int argument_count, char **arguments)
{
    yvex_server_options options;
    yvex_server *server = NULL;
    daemon_thread_state thread_state;
    pthread_t signal_thread, console_thread;
    sigset_t signals;
    yvex_error err;
    int index, rc, signal_ready = 0, console_ready = 0;
    int openai_seen = 0, openai_port_seen = 0, openai_timeout_seen = 0;
    memset(&options, 0, sizeof(options));
    options.target_id = "deepseek4-v4-flash";
    options.backend = YVEX_BACKEND_KIND_CPU;
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
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGTERM);
    (void)pthread_sigmask(SIG_BLOCK, &signals, NULL);
    rc = yvex_server_create(&server, &options, &err);
    if (rc == YVEX_OK) rc = yvex_server_start(server, &err);
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
