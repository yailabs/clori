/* Owner: client.yvex runtime-client lane.
 * Owns: product CLI grammar, thin protocol requests, one-shot streaming, REPL, and compact views.
 * Does not own: model/artifact opening, generation, sessions, telemetry truth, or offline tooling.
 * Invariants: no runtime-client route invokes an engine API and all generation flows through yvexd.
 * Boundary: runtime-facing command lane over the public local client protocol.
 * Purpose: provide chat, run, runtime, session, model selection, help, and version surfaces.
 * Inputs: argv, terminal input, explicit prompt bytes, and protocol messages.
 * Effects: writes client stdout/stderr, connects local sockets, and may exec yvexd.
 * Failure: concise errors preserve stable parser/runtime exit classes and one actionable hint. */
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700

#include "src/cli/private.h"

#include <yvex/server.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define CLIENT_REPL_LINE_MAX 65536u
#define CLIENT_REPL_HISTORY_MAX 64u

typedef struct {
    unsigned long long maximum_new_tokens, seed, top_k;
    double temperature, top_p, min_p, typical_p;
    int stochastic, seed_present;
} client_turn_options;

typedef struct {
    char session[YVEX_SERVER_SESSION_NAME_CAP];
    atomic_int done, interrupts, force_exit;
    pthread_t thread;
    sigset_t previous_mask;
    int ready;
} client_turn_signals;

typedef struct {
    char *entry[CLIENT_REPL_HISTORY_MAX];
    size_t count;
} client_repl_history;

typedef struct {
    char name[YVEX_SERVER_SESSION_NAME_CAP];
    char artifact[PATH_MAX];
    char binding[PATH_MAX];
    char target[128];
    char backend[8];
    unsigned long long context;
} client_model_config;

static volatile sig_atomic_t repl_signal_state;

/* Purpose: print the complete compact product grammar without diagnostic catalog walls. */
static void print_help(FILE *output)
{
    fprintf(output,
            "YVEX local inference\n\n"
            "  yvex                         enter chat\n"
            "  yvex chat [--session NAME] [--max-new-tokens N]\n"
            "                               interactive client\n"
            "  yvex run [options] TEXT     one streamed turn\n"
            "  yvex runtime start|stop|status|watch|trace\n"
            "  yvex session new|list|show|attach|detach|reset|close\n"
            "  yvex model list|use|show\n"
            "  yvex artifact show|verify|metadata|tensors|materialize|emit ...\n"
            "  yvex graph ...\n"
            "  yvex quant preset|plan|emit|summarize|explain|policy|imatrix ...\n"
            "  yvex tokenizer show|encode|decode|prompt ...\n"
            "  yvex source manifest|native ...\n"
            "  yvex tensor map|collection ...\n"
            "  yvex evidence target|model|moe|backend|cuda ...\n"
            "  yvex help | version\n");
}

/* Purpose: report one missing/refusing daemon consistently. */
static int client_error(const yvex_error *err)
{
    fprintf(stderr, "yvex: %s\n", yvex_error_message(err));
    if (yvex_error_code(err) == YVEX_ERR_IO)
        fprintf(stderr, "hint: start it with `yvex runtime start --model ARTIFACT "
                        "--runtime-binding BINDING`\n");
    return 1;
}

/* Purpose: initialize neutral greedy facts and a client-local correlation number.
 * Inputs: request storage and operation. Effects: resets and populates caller storage.
 * Failure: none. Boundary: the daemon remains authority for execution identities. */
static void request_init(yvex_client_request *request,
                         yvex_client_operation operation)
{
    static unsigned long long next_request = 1u;
    memset(request, 0, sizeof(*request));
    request->schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request->operation = operation;
    request->request_number = next_request++;
    request->maximum_new_tokens = 128u;
    request->temperature = 1.0;
    request->top_p = 1.0;
    request->typical_p = 1.0;
}

/* Purpose: initialize the explicit neutral greedy product policy.
 * Inputs: caller-owned policy storage. Effects: replaces its prior contents.
 * Failure: none. Boundary: does not seal or execute a sampling policy. */
static void turn_options_init(client_turn_options *options)
{
    memset(options, 0, sizeof(*options));
    options->maximum_new_tokens = 128u;
    options->temperature = 1.0;
    options->top_p = 1.0;
    options->typical_p = 1.0;
}

/* Purpose: parse one unsigned client option with explicit zero admission.
 * Inputs: terminated text, output, and zero policy. Effects: writes output on success.
 * Failure: returns false on range, syntax, or policy error. Boundary: no domain admission. */
static int parse_u64(const char *text, unsigned long long *value, int allow_zero)
{
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno || !end || *end || (!allow_zero && !parsed)) return 0;
    *value = parsed;
    return 1;
}

/* Purpose: parse one finite binary64 client policy value.
 * Inputs: terminated text and output. Effects: writes output on success.
 * Failure: returns false for malformed or non-finite text. Boundary: policy ranges are checked later. */
static int parse_double(const char *text, double *value)
{
    char *end = NULL;
    double parsed;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno || !end || *end || !isfinite(parsed)) return 0;
    *value = parsed;
    return 1;
}

/* Purpose: connect, send one typed request, and retain the socket for streaming.
 * Inputs: client output, immutable request, and error output. Effects: opens a local connection.
 * Failure: closes partial ownership and returns the first protocol error. Boundary: no engine call. */
static int request_open(yvex_client **client,
                        const yvex_client_request *request, yvex_error *err)
{
    int rc = yvex_client_connect(client, NULL, err);
    if (rc == YVEX_OK) rc = yvex_client_send(*client, request, err);
    if (rc != YVEX_OK) yvex_client_close(client);
    return rc;
}

/* Purpose: request cancellation over a separate connection while the stream remains owned. */
static int cancellation_request(const char *session)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_GENERATION_CANCEL);
    (void)snprintf(request.session_name, sizeof(request.session_name), "%s",
                   session);
    rc = request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    yvex_client_close(&client);
    return rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ACK;
}

/* Purpose: translate the first SIGINT into cancellation and later SIGINT into REPL exit.
 * Inputs: one live turn signal state. Effects: sends cancellation and updates atomic flags.
 * Failure: retries transient cancellation connection failure until the turn ends. Boundary: signal thread only. */
static void *turn_signal_main(void *opaque)
{
    client_turn_signals *state = opaque;
    sigset_t signals;
    int number;
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGUSR1);
    while (sigwait(&signals, &number) == 0) {
        if (number == SIGUSR1) break;
        if (atomic_fetch_add_explicit(&state->interrupts, 1,
                                      memory_order_acq_rel) > 0) {
            atomic_store_explicit(&state->force_exit, 1, memory_order_release);
            continue;
        }
        while (!atomic_load_explicit(&state->done, memory_order_acquire)) {
            struct timespec delay = {0, 10000000L};
            if (cancellation_request(state->session)) break;
            (void)nanosleep(&delay, NULL);
        }
    }
    return NULL;
}

/* Purpose: transfer SIGINT ownership to one bounded cancellation coordinator.
 * Inputs: state storage and exact session name. Effects: blocks signals and starts one thread.
 * Failure: leaves coordination disabled if mask or thread setup fails. Boundary: no generation mutation. */
static void turn_signals_open(client_turn_signals *state,
                              const char *session)
{
    sigset_t signals;
    memset(state, 0, sizeof(*state));
    (void)snprintf(state->session, sizeof(state->session), "%s", session);
    atomic_init(&state->done, 0);
    atomic_init(&state->interrupts, 0);
    atomic_init(&state->force_exit, 0);
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGUSR1);
    if (pthread_sigmask(SIG_BLOCK, &signals, &state->previous_mask) != 0)
        return;
    if (pthread_create(&state->thread, NULL, turn_signal_main, state) == 0)
        state->ready = 1;
    else
        (void)pthread_sigmask(SIG_SETMASK, &state->previous_mask, NULL);
}

/* Purpose: finish cancellation coordination and restore the caller signal policy.
 * Inputs: initialized coordinator state. Effects: joins its thread and restores the mask.
 * Failure: returns the observed interrupt class. Boundary: owns no daemon cancellation state. */
static int turn_signals_close(client_turn_signals *state)
{
    int result = 0;
    if (!state->ready) return 0;
    atomic_store_explicit(&state->done, 1, memory_order_release);
    (void)pthread_kill(state->thread, SIGUSR1);
    (void)pthread_join(state->thread, NULL);
    (void)pthread_sigmask(SIG_SETMASK, &state->previous_mask, NULL);
    if (atomic_load_explicit(&state->force_exit, memory_order_acquire))
        result = 2;
    else if (atomic_load_explicit(&state->interrupts, memory_order_acquire))
        result = 1;
    return result;
}

/* Purpose: render one compact runtime status or stable JSON projection.
 * Inputs: authoritative protocol snapshot and output choice. Effects: writes product stdout.
 * Failure: none after snapshot admission. Boundary: rendering cannot change readiness. */
static void render_status(const yvex_server_summary *status, int json)
{
    if (json) {
        printf("{\"protocol\":%u,\"status\":%u,\"target\":\"%s\","
               "\"backend\":%u,\"ready\":%s,\"uptime_ns\":%llu,"
               "\"model_open_count\":%llu,\"model_close_count\":%llu,"
               "\"artifact_open_count\":%llu,\"binding_open_count\":%llu,"
               "\"materialization_count\":%llu,\"residency_build_count\":%llu,"
               "\"output_head_upload_count\":%llu,\"sessions\":%llu,"
               "\"active_sessions\":%llu,\"total_sessions\":%llu,"
               "\"queue_depth\":%llu,\"queue_capacity\":%llu,"
               "\"active_requests\":%llu,\"completed_requests\":%llu,"
               "\"failed_requests\":%llu,\"cancelled_requests\":%llu,"
               "\"openai_enabled\":%s,\"openai_ready\":%s,"
               "\"openai_port\":%u,\"active_http_requests\":%llu,"
               "\"completed_http_requests\":%llu,"
               "\"failed_http_requests\":%llu,"
               "\"cancelled_http_requests\":%llu,"
               "\"telemetry_dropped\":%llu,\"rss_bytes\":%llu,"
               "\"peak_rss_bytes\":%llu,\"mapped_artifact_bytes\":%llu,"
               "\"resident_host_bytes\":%llu,\"resident_device_bytes\":%llu,"
               "\"model_identity\":\"%s\",\"binding_identity\":\"%s\","
               "\"artifact_identity\":\"%s\",\"variant_identity\":\"%s\"}\n",
               YVEX_LOCAL_PROTOCOL_VERSION, (unsigned int)status->status,
               status->target_id, (unsigned int)status->backend,
               status->runtime_ready ? "true" : "false",
               status->metrics.uptime_ns, status->metrics.model_open_count,
               status->metrics.model_close_count,
               status->metrics.artifact_open_count,
               status->metrics.binding_open_count,
               status->metrics.materialization_count,
               status->metrics.residency_build_count,
               status->metrics.output_head_upload_count, status->session_count,
               status->metrics.active_sessions, status->metrics.total_sessions,
               status->metrics.queue_depth, status->metrics.queue_capacity,
               status->metrics.active_requests,
               status->metrics.completed_requests,
               status->metrics.failed_requests,
               status->metrics.cancelled_requests,
               status->openai_listener_enabled ? "true" : "false",
               status->openai_listener_ready ? "true" : "false",
               (unsigned int)status->openai_port,
               status->metrics.active_http_requests,
               status->metrics.completed_http_requests,
               status->metrics.failed_http_requests,
               status->metrics.cancelled_http_requests,
               status->metrics.telemetry_dropped,
               status->metrics.current_rss_bytes,
               status->metrics.peak_rss_bytes,
               status->metrics.mapped_artifact_bytes,
               status->metrics.resident_host_bytes,
               status->metrics.resident_device_bytes,
               status->runtime_model_identity,
               status->runtime_binding_identity,
               status->artifact_identity,
               status->physical_variant_identity);
        return;
    }
    printf("YVEX runtime · %s · %s\n",
           status->target_id[0] ? status->target_id : "no model",
           status->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    printf("  state      %s\n",
           status->status == YVEX_SERVER_STATUS_READY ? "ready" : "not ready");
    printf("  sessions   %llu active\n", status->session_count);
    printf("  queue      %llu/%llu\n", status->metrics.queue_depth,
           status->metrics.queue_capacity);
    printf("  model      opened %llu time%s\n", status->metrics.model_open_count,
           status->metrics.model_open_count == 1u ? "" : "s");
    if (status->openai_listener_enabled)
        printf("  openai     %s · 127.0.0.1:%u · %llu active · %llu completed\n",
               status->openai_listener_ready ? "ready" : "starting",
               (unsigned int)status->openai_port,
               status->metrics.active_http_requests,
               status->metrics.completed_http_requests);
    else
        printf("  openai     disabled\n");
    printf("  memory     host %.2f GiB · device %.2f GiB · RSS %.2f GiB\n",
           (double)status->metrics.resident_host_bytes / 1073741824.0,
           (double)status->metrics.resident_device_bytes / 1073741824.0,
           (double)status->metrics.current_rss_bytes / 1073741824.0);
}

/* Purpose: request and render one runtime status response. */
static int runtime_status(int json)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_RUNTIME_STATUS);
    rc = request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_STATUS)
        render_status(&message.runtime, json);
    else if (rc == YVEX_OK) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "client.status",
                       "daemon returned an unexpected response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc == YVEX_OK ? 0 : client_error(&err);
}

/* Purpose: render the compact operational projection of one typed event. */
static void render_engine_event(const yvex_server_event *event)
{
    printf("%-30s", yvex_server_event_kind_name(event->kind));
    if (event->request_id[0]) printf(" request=%s", event->request_id);
    if (event->session_id[0]) printf(" session=%s", event->session_id);
    if (event->value_a) printf(" a=%llu", event->value_a);
    if (event->value_b) printf(" b=%llu", event->value_b);
    if (event->seconds > 0.0) printf(" elapsed=%.3fs", event->seconds);
    if (event->rate > 0.0) printf(" rate=%.1f tok/s", event->rate);
    putchar('\n');
    fflush(stdout);
}

/* Purpose: subscribe to raw JSONL or the compact operational event view.
 * Inputs: raw-versus-engine choice. Effects: opens a subscription and streams stdout.
 * Failure: returns concise protocol refusal after closing the client. Boundary: one event authority. */
static int runtime_events(int raw)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    char json[2048];
    int rc;
    request_init(&request, raw ? YVEX_CLIENT_OP_RUNTIME_TRACE
                               : YVEX_CLIENT_OP_RUNTIME_WATCH);
    request.trace_level = raw ? YVEX_SERVER_TRACE_FULL
                              : YVEX_SERVER_TRACE_STAGES;
    rc = request_open(&client, &request, &err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind != YVEX_CLIENT_MESSAGE_EVENT) continue;
        if (!raw) render_engine_event(&message.event);
        else if (yvex_server_event_json(&message.event, json, sizeof(json),
                                        &err) == YVEX_OK) {
            fputs(json, stdout);
            fflush(stdout);
        }
        if (message.event.kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE)
            break;
    }
    yvex_client_close(&client);
    return rc == YVEX_OK ? 0 : client_error(&err);
}

/* Purpose: send one administration request and render bounded response rows.
 * Inputs: typed operation, optional session, and row mode. Effects: performs one protocol exchange.
 * Failure: reports daemon errors without mutating local state. Boundary: no direct session access. */
static int administration(yvex_client_operation operation,
                          const char *session_name, int render_mode)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, operation);
    if (session_name)
        snprintf(request.session_name, sizeof(request.session_name), "%s",
                 session_name);
    rc = request_open(&client, &request, &err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            yvex_error_set(&err, (yvex_status)message.status, "client.request",
                           message.reason);
            rc = message.status;
            break;
        }
        if (message.kind == YVEX_CLIENT_MESSAGE_SESSION)
            printf("%-20s %-10s position=%llu turns=%llu\n",
                   message.session_name,
                   yvex_server_session_state_name(message.session_state),
                   message.final_position, message.generated_tokens);
        else if (message.kind == YVEX_CLIENT_MESSAGE_ACK) {
            if (!render_mode)
                printf("%s\n", message.reason[0] ? message.reason : "ok");
            break;
        }
        if (render_mode <= 0) break;
    }
    yvex_client_close(&client);
    return rc == YVEX_OK ? 0 : client_error(&err);
}

/* Purpose: stream one turn and render only committed fragments plus final metrics.
 * Inputs: session, explicit prompt bytes, policy, and conversation mode. Effects: sends or cancels a turn.
 * Failure: preserves daemon progress and reports exact stream status. Boundary: never opens the engine. */
static int generation_turn(const char *session_name,
                           const unsigned char *prompt,
                           unsigned long long prompt_bytes,
                           const client_turn_options *options,
                           int conversation)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    client_turn_signals signals;
    yvex_error err;
    int rc, started = 0;
    request_init(&request, YVEX_CLIENT_OP_GENERATION_TURN);
    snprintf(request.session_name, sizeof(request.session_name), "%s",
             session_name);
    request.prompt = prompt;
    request.prompt_bytes = prompt_bytes;
    request.maximum_new_tokens = options->maximum_new_tokens;
    request.stochastic = options->stochastic;
    request.seed_present = options->seed_present;
    request.seed = options->seed;
    request.temperature = options->temperature;
    request.top_k = options->top_k;
    request.top_p = options->top_p;
    request.min_p = options->min_p;
    request.typical_p = options->typical_p;
    turn_signals_open(&signals, session_name);
    rc = request_open(&client, &request, &err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_TURN_STARTED) {
            if (conversation) {
                fputs("assistant> ", stdout);
                fflush(stdout);
            }
            started = 1;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
            if (!started && conversation) fputs("assistant> ", stdout);
            if (message.byte_count)
                (void)fwrite(message.bytes, 1u, (size_t)message.byte_count,
                             stdout);
            fflush(stdout);
            started = 1;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE) {
            if (started) putchar('\n');
            printf("%llu prompt · %llu reused · TTFT %.2fs · %llu generated · %.1f tok/s\n",
                   message.prompt_tokens, message.reused_tokens,
                   message.first_token_seconds, message.generated_tokens,
                   message.decode_rate);
            break;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            if (started) putchar('\n');
            yvex_error_set(&err, (yvex_status)message.status, "client.turn",
                           message.reason);
            rc = message.status;
            break;
        }
    }
    yvex_client_close(&client);
    {
        int interrupted = turn_signals_close(&signals);
        if (interrupted) {
            if (conversation) puts(interrupted == 2 ? "[cancelled; leaving chat]"
                                                    : "[cancelled]");
            return interrupted == 2 ? 131 : 130;
        }
    }
    return rc == YVEX_OK ? 0 : client_error(&err);
}

/* Purpose: ensure a named session exists without treating duplicate creation as success. */
static int session_ensure(const char *name)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_SESSION_SHOW);
    snprintf(request.session_name, sizeof(request.session_name), "%s", name);
    rc = request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    yvex_client_close(&client);
    if (rc == YVEX_OK && message.kind != YVEX_CLIENT_MESSAGE_ERROR) return 0;
    return administration(YVEX_CLIENT_OP_SESSION_NEW, name, -1);
}

/* Purpose: retain one bounded in-memory prompt history.
 * Inputs: history and immutable line. Effects: copies a nonduplicate entry and evicts the oldest.
 * Failure: allocation failure leaves history unchanged. Boundary: prompt content is never persisted. */
static void repl_history_push(client_repl_history *history, const char *line)
{
    char *copy;
    if (!line[0] || (history->count &&
                     !strcmp(history->entry[history->count - 1u], line)))
        return;
    copy = strdup(line);
    if (!copy) return;
    if (history->count == CLIENT_REPL_HISTORY_MAX) {
        free(history->entry[0]);
        memmove(history->entry, history->entry + 1,
                (CLIENT_REPL_HISTORY_MAX - 1u) * sizeof(history->entry[0]));
        history->count--;
    }
    history->entry[history->count++] = copy;
}

/* Purpose: release prompt history without persisting conversation content.
 * Inputs: history owner. Effects: frees all entries and clears the owner.
 * Failure: none. Boundary: does not alter daemon transcript state. */
static void repl_history_close(client_repl_history *history)
{
    size_t index;
    for (index = 0u; index < history->count; ++index) free(history->entry[index]);
    memset(history, 0, sizeof(*history));
}

/* Purpose: make resize and prompt-interrupt state observable to the bounded TTY reader.
 * Inputs: SIGWINCH or SIGINT. Effects: records only signal-safe scalar state.
 * Failure: excess interrupts saturate. Boundary: generation owns SIGINT while a turn executes. */
static void repl_signal_handler(int number)
{
    sig_atomic_t interrupts = repl_signal_state & 3;
    if (number == SIGWINCH)
        repl_signal_state |= 4;
    else if (interrupts < 2)
        repl_signal_state = (repl_signal_state & ~3) | (interrupts + 1);
}

/* Purpose: redraw one prompt after history navigation or terminal resize.
 * Inputs: prompt and explicit line bytes. Effects: emits only terminal-control and caller bytes.
 * Failure: terminal write failure is observed by the next input operation. Boundary: TTY only. */
static void repl_redraw(const char *prompt, const char *line, size_t count)
{
    fputs("\r\033[2K", stdout);
    fputs(prompt, stdout);
    if (count) (void)fwrite(line, 1u, count, stdout);
    fflush(stdout);
}

/* Purpose: replace the editable line with one bounded history entry.
 * Inputs: storage owner, extent, selected text, and prompt. Effects: grows and redraws the line.
 * Failure: allocation failure preserves the prior line. Boundary: no daemon state changes. */
static int repl_replace_line(char **line, size_t *count, size_t *capacity,
                             const char *replacement, const char *prompt)
{
    size_t needed = strlen(replacement) + 1u;
    char *grown;
    if (needed > CLIENT_REPL_LINE_MAX + 1u) return 0;
    if (needed > *capacity) {
        grown = realloc(*line, needed);
        if (!grown) return 0;
        *line = grown;
        *capacity = needed;
    }
    memcpy(*line, replacement, needed);
    *count = needed - 1u;
    repl_redraw(prompt, *line, *count);
    return 1;
}

/* Purpose: append one byte to a bounded editable prompt.
 * Inputs: line owner, extent, capacity, and byte. Effects: grows and terminates the line.
 * Failure: returns false at the prompt budget or allocation failure. Boundary: byte-oriented UTF-8 input. */
static int repl_append_byte(char **line, size_t *count, size_t *capacity,
                            unsigned char byte)
{
    char *grown;
    size_t next;
    if (*count >= CLIENT_REPL_LINE_MAX) return 0;
    if (*count + 1u >= *capacity) {
        next = *capacity ? *capacity * 2u : 256u;
        if (next > CLIENT_REPL_LINE_MAX + 1u) next = CLIENT_REPL_LINE_MAX + 1u;
        grown = realloc(*line, next);
        if (!grown) return 0;
        *line = grown;
        *capacity = next;
    }
    (*line)[(*count)++] = (char)byte;
    (*line)[*count] = '\0';
    return 1;
}

/* Purpose: remove the final complete UTF-8 code-unit sequence from one editable line.
 * Inputs: line bytes and extent. Effects: truncates to the preceding code-point boundary.
 * Failure: malformed trailing continuation bytes are removed conservatively. Boundary: display editing only. */
static void repl_backspace(char *line, size_t *count)
{
    if (!*count) return;
    (*count)--;
    while (*count && (((unsigned char)line[*count] & 0xc0u) == 0x80u))
        (*count)--;
    line[*count] = '\0';
}

/* Purpose: read one bounded TTY line with history, resize, UTF-8 deletion, and bracketed paste.
 * Inputs: prompt, in-memory history, and result owners. Effects: temporarily enables raw input.
 * Failure: returns -1 on terminal/allocation failure, -2 on first SIGINT, and 0 on EOF/second SIGINT.
 * Boundary: restores terminal state before generation or return and never persists prompt content. */
static int repl_read_line(const char *prompt, const client_repl_history *history,
                          char **output, size_t *output_count)
{
    struct termios saved, raw;
    char *line = NULL;
    size_t count = 0u, capacity = 0u, selected = history->count;
    int paste = 0, result = -1;
    unsigned char byte;
    if (tcgetattr(STDIN_FILENO, &saved) != 0) return -1;
    raw = saved;
    raw.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    raw.c_iflag &= (tcflag_t)~(ICRNL | IXON);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) return -1;
    fputs("\033[?2004h", stdout);
    repl_redraw(prompt, "", 0u);
    for (;;) {
        ssize_t got = read(STDIN_FILENO, &byte, 1u);
        if (got < 0 && errno == EINTR) {
            if ((repl_signal_state & 3) >= 2) {
                result = 0;
                break;
            }
            if ((repl_signal_state & 3) == 1) {
                fputs("^C\r\n", stdout);
                result = -2;
                break;
            }
            if (repl_signal_state & 4) {
                struct winsize window;
                (void)ioctl(STDOUT_FILENO, TIOCGWINSZ, &window);
                repl_signal_state &= ~4;
                repl_redraw(prompt, line ? line : "", count);
            }
            continue;
        }
        if (got <= 0) {
            result = 0;
            break;
        }
        if (byte == '\r' || byte == '\n') {
            if (paste) {
                if (!repl_append_byte(&line, &count, &capacity, '\n')) break;
                fputs("\r\n... ", stdout);
                fflush(stdout);
                continue;
            }
            fputs("\r\n", stdout);
            result = 1;
            break;
        }
        if (byte == 4u && !count) {
            result = 0;
            break;
        }
        if (byte == 8u || byte == 127u) {
            repl_backspace(line, &count);
            repl_redraw(prompt, line ? line : "", count);
            continue;
        }
        if (byte == 27u) {
            unsigned char sequence[5];
            size_t length = 0u;
            while (length < sizeof(sequence) && read(STDIN_FILENO, &sequence[length], 1u) == 1) {
                length++;
                if ((length == 2u && (sequence[1] == 'A' || sequence[1] == 'B')) ||
                    (length == 5u && sequence[4] == '~'))
                    break;
            }
            if (length == 5u && !memcmp(sequence, "[200~", 5u)) paste = 1;
            else if (length == 5u && !memcmp(sequence, "[201~", 5u)) paste = 0;
            else if (!paste && length == 2u && sequence[0] == '[' && sequence[1] == 'A' && selected) {
                selected--;
                if (!repl_replace_line(&line, &count, &capacity,
                                       history->entry[selected], prompt))
                    break;
            } else if (!paste && length == 2u && sequence[0] == '[' &&
                       sequence[1] == 'B' && selected < history->count) {
                selected++;
                if (!repl_replace_line(&line, &count, &capacity,
                                       selected == history->count ? "" : history->entry[selected],
                                       prompt))
                    break;
            }
            continue;
        }
        if (!repl_append_byte(&line, &count, &capacity, byte)) break;
        (void)fwrite(&byte, 1u, 1u, stdout);
        fflush(stdout);
    }
    fputs("\033[?2004l", stdout);
    (void)tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved);
    if (result == 1) {
        if (!line) {
            line = calloc(1u, 1u);
            if (!line) result = -1;
        }
        *output = line;
        *output_count = count;
    } else {
        free(line);
    }
    return result;
}

/* Purpose: move one REPL attachment without transferring session authority to the client.
 * Inputs: current storage, requested name, and creation policy. Effects: creates/attaches then detaches old.
 * Failure: preserves the current attachment when admission fails. Boundary: daemon owns both sessions. */
static int repl_switch_session(char current[YVEX_SERVER_SESSION_NAME_CAP],
                               const char *next, int create)
{
    if (!next || !next[0] || strlen(next) >= YVEX_SERVER_SESSION_NAME_CAP) return 0;
    if (!strcmp(current, next)) return 1;
    if (create && administration(YVEX_CLIENT_OP_SESSION_NEW, next, -1) != 0) return 0;
    if (administration(YVEX_CLIENT_OP_SESSION_ATTACH, next, -1) != 0) return 0;
    (void)administration(YVEX_CLIENT_OP_SESSION_DETACH, current, -1);
    (void)snprintf(current, YVEX_SERVER_SESSION_NAME_CAP, "%s", next);
    printf("session: %s\n", current);
    return 1;
}

/* Purpose: run the bounded terminal REPL over one daemon-owned session.
 * Inputs: exact session name and terminal input. Effects: attaches, streams turns, then detaches.
 * Failure: refuses non-TTY use and preserves daemon session state. Boundary: slash commands stay small. */
static int chat(const char *session_name, unsigned long long maximum_new_tokens)
{
    client_turn_options options;
    client_repl_history history;
    struct sigaction action, prior_interrupt, prior_resize;
    char current[YVEX_SERVER_SESSION_NAME_CAP];
    unsigned long long generated_session = 1u;
    int closed = 0;
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        fprintf(stderr, "yvex: chat requires a terminal; use `yvex run TEXT`\n");
        return 2;
    }
    memset(&history, 0, sizeof(history));
    memset(&action, 0, sizeof(action));
    action.sa_handler = repl_signal_handler;
    (void)sigemptyset(&action.sa_mask);
    repl_signal_state = 0;
    if (sigaction(SIGINT, &action, &prior_interrupt) != 0) return 1;
    if (sigaction(SIGWINCH, &action, &prior_resize) != 0) {
        (void)sigaction(SIGINT, &prior_interrupt, NULL);
        return 1;
    }
    turn_options_init(&options);
    options.maximum_new_tokens = maximum_new_tokens;
    (void)snprintf(current, sizeof(current), "%s", session_name);
    if (session_ensure(current) != 0) {
        (void)sigaction(SIGINT, &prior_interrupt, NULL);
        (void)sigaction(SIGWINCH, &prior_resize, NULL);
        return 1;
    }
    (void)administration(YVEX_CLIENT_OP_SESSION_ATTACH, current, 0);
    printf("YVEX · local runtime · session %s\n\n", current);
    for (;;) {
        char *line = NULL;
        size_t count = 0u;
        int input = repl_read_line("you> ", &history, &line, &count);
        if (input == -2) continue;
        if (input <= 0) break;
        repl_signal_state &= ~3;
        if (!count) {
            free(line);
            continue;
        }
        if (!strcmp(line, "/quit")) {
            free(line);
            break;
        }
        if (!strcmp(line, "/help")) {
            puts("/new [name]  /sessions  /attach NAME  /status  /reset  /cancel  /detach  /close  /quit");
            free(line);
            continue;
        }
        if (!strcmp(line, "/sessions")) {
            (void)administration(YVEX_CLIENT_OP_SESSION_LIST, NULL, 1);
            free(line);
            continue;
        }
        if (!strcmp(line, "/status")) {
            (void)runtime_status(0);
            free(line);
            continue;
        }
        if (!strcmp(line, "/reset")) {
            (void)administration(YVEX_CLIENT_OP_SESSION_RESET, current, 0);
            free(line);
            continue;
        }
        if (!strcmp(line, "/cancel")) {
            puts(cancellation_request(current) ? "cancel requested" : "no active turn");
            free(line);
            continue;
        }
        if (!strncmp(line, "/new", 4u) && (!line[4] || line[4] == ' ')) {
            char generated[YVEX_SERVER_SESSION_NAME_CAP];
            const char *next = line[4] ? line + 5 : generated;
            if (!line[4])
                (void)snprintf(generated, sizeof(generated), "chat-%llu", generated_session++);
            (void)repl_switch_session(current, next, 1);
            free(line);
            continue;
        }
        if (!strncmp(line, "/attach ", 8u)) {
            (void)repl_switch_session(current, line + 8, 0);
            free(line);
            continue;
        }
        if (!strcmp(line, "/detach")) {
            free(line);
            break;
        }
        if (!strcmp(line, "/close")) {
            (void)administration(YVEX_CLIENT_OP_SESSION_CLOSE, current, 0);
            free(line);
            closed = 1;
            break;
        }
        repl_history_push(&history, line);
        if (generation_turn(current, (const unsigned char *)line,
                            (unsigned long long)count, &options, 1) == 131) {
            free(line);
            break;
        }
        free(line);
    }
    repl_history_close(&history);
    if (!closed) (void)administration(YVEX_CLIENT_OP_SESSION_DETACH, current, 0);
    (void)sigaction(SIGINT, &prior_interrupt, NULL);
    (void)sigaction(SIGWINCH, &prior_resize, NULL);
    return 0;
}

/* Purpose: parse the bounded product-chat options before entering terminal mode.
 * Inputs: product argv. Effects: invokes one REPL with explicit session and turn bound.
 * Failure: malformed, duplicate, zero, or unknown options refuse before daemon attachment.
 * Boundary: sampling remains the daemon generation owner's policy. */
static int chat_command(int argc, char **argv)
{
    const char *session = "main";
    unsigned long long maximum_new_tokens = 128u;
    int index, saw_session = 0, saw_maximum = 0;
    for (index = 2; index < argc; ++index) {
        if (!strcmp(argv[index], "--session") && !saw_session && index + 1 < argc) {
            session = argv[++index];
            saw_session = 1;
        } else if (!strcmp(argv[index], "--max-new-tokens") && !saw_maximum &&
                   index + 1 < argc) {
            if (!parse_u64(argv[++index], &maximum_new_tokens, 0)) return 2;
            saw_maximum = 1;
        } else {
            return 2;
        }
    }
    return chat(session, maximum_new_tokens);
}

/* Purpose: dispatch the compact session namespace without exposing protocol details.
 * Inputs: process argv. Effects: sends one typed administration operation.
 * Failure: returns parser or daemon status. Boundary: no session registry exists in the client. */
static int session_command(int argc, char **argv)
{
    const char *action = argc > 2 ? argv[2] : NULL;
    const char *name = argc > 3 ? argv[3] : NULL;
    if (!action) return 2;
    if (!strcmp(action, "new")) {
        if (argc > 4) return 2;
        return administration(YVEX_CLIENT_OP_SESSION_NEW, name, 0);
    }
    if (!strcmp(action, "list")) {
        if (argc != 3) return 2;
        return administration(YVEX_CLIENT_OP_SESSION_LIST, NULL, 1);
    }
    if (!name || argc != 4) {
        fprintf(stderr, "yvex: session %s requires NAME\n", action);
        return 2;
    }
    if (!strcmp(action, "show")) return administration(YVEX_CLIENT_OP_SESSION_SHOW, name, 0);
    if (!strcmp(action, "attach")) return administration(YVEX_CLIENT_OP_SESSION_ATTACH, name, 0);
    if (!strcmp(action, "detach")) return administration(YVEX_CLIENT_OP_SESSION_DETACH, name, 0);
    if (!strcmp(action, "reset")) return administration(YVEX_CLIENT_OP_SESSION_RESET, name, 0);
    if (!strcmp(action, "close")) return administration(YVEX_CLIENT_OP_SESSION_CLOSE, name, 0);
    fprintf(stderr, "yvex: unknown session action: %s\n", action);
    return 2;
}

/* Purpose: parse and execute one complete one-shot policy without inferring strategy.
 * Inputs: product argv. Effects: creates and closes an ephemeral session or uses an explicit one.
 * Failure: rejects malformed policy before connecting and preserves daemon state on stream failure.
 * Boundary: product generation always crosses the local protocol. */
static int run_command(int argc, char **argv)
{
    client_turn_options options;
    char ephemeral[YVEX_SERVER_SESSION_NAME_CAP];
    const char *session = NULL, *prompt = NULL;
    int index, owns_session = 0;
    turn_options_init(&options);
    for (index = 2; index < argc; ++index) {
        const char *argument = argv[index];
        if (!strcmp(argument, "--session") && index + 1 < argc)
            session = argv[++index];
        else if (!strcmp(argument, "--max-new-tokens") && index + 1 < argc) {
            if (!parse_u64(argv[++index], &options.maximum_new_tokens, 0)) return 2;
        } else if (!strcmp(argument, "--strategy") && index + 1 < argc) {
            const char *strategy = argv[++index];
            if (!strcmp(strategy, "greedy")) options.stochastic = 0;
            else if (!strcmp(strategy, "stochastic")) options.stochastic = 1;
            else return 2;
        } else if (!strcmp(argument, "--seed") && index + 1 < argc) {
            if (!parse_u64(argv[++index], &options.seed, 1)) return 2;
            options.seed_present = 1;
        } else if (!strcmp(argument, "--temperature") && index + 1 < argc) {
            if (!parse_double(argv[++index], &options.temperature)) return 2;
        } else if (!strcmp(argument, "--top-k") && index + 1 < argc) {
            if (!parse_u64(argv[++index], &options.top_k, 1)) return 2;
        } else if (!strcmp(argument, "--top-p") && index + 1 < argc) {
            if (!parse_double(argv[++index], &options.top_p)) return 2;
        } else if (!strcmp(argument, "--min-p") && index + 1 < argc) {
            if (!parse_double(argv[++index], &options.min_p)) return 2;
        } else if (!strcmp(argument, "--typical-p") && index + 1 < argc) {
            if (!parse_double(argv[++index], &options.typical_p)) return 2;
        } else if (argument[0] == '-') {
            fprintf(stderr, "yvex: unknown run option: %s\n", argument);
            return 2;
        } else if (!prompt)
            prompt = argument;
        else {
            fprintf(stderr, "yvex: run accepts one prompt argument\n");
            return 2;
        }
    }
    if (!prompt || options.temperature <= 0.0 || options.top_p <= 0.0 ||
        options.top_p > 1.0 || options.min_p < 0.0 || options.min_p > 1.0 ||
        options.typical_p <= 0.0 || options.typical_p > 1.0 ||
        (options.stochastic && !options.seed_present) ||
        (!options.stochastic &&
         (options.seed_present || options.temperature != 1.0 || options.top_k ||
          options.top_p != 1.0 || options.min_p != 0.0 ||
          options.typical_p != 1.0))) {
        fprintf(stderr,
                "yvex: run requires one prompt and an explicit valid strategy policy\n");
        return 2;
    }
    if (!session) {
        (void)snprintf(ephemeral, sizeof(ephemeral), "run-%lu",
                       (unsigned long)getpid());
        session = ephemeral;
        owns_session = 1;
        if (administration(YVEX_CLIENT_OP_SESSION_NEW, session, -1) != 0) return 1;
    }
    {
        int status = generation_turn(session, (const unsigned char *)prompt,
                                     (unsigned long long)strlen(prompt),
                                     &options, 0);
        if (owns_session)
            (void)administration(YVEX_CLIENT_OP_SESSION_CLOSE, session, -1);
        return status;
    }
}

/* Purpose: resolve the private XDG model-selection file and its owning directory.
 * Inputs: caller path buffers. Effects: writes terminated absolute paths only.
 * Failure: missing/relative environment or overflow refuses. Boundary: no directory is created. */
static int model_config_paths(char directory[PATH_MAX], char path[PATH_MAX])
{
    const char *base = getenv("XDG_CONFIG_HOME");
    char fallback[PATH_MAX];
    int count;
    if (!base || !base[0]) {
        const char *home = getenv("HOME");
        count = home ? snprintf(fallback, sizeof(fallback), "%s/.config", home) : -1;
        if (!home || home[0] != '/' || count <= 0 || (size_t)count >= sizeof(fallback))
            return 0;
        base = fallback;
    }
    count = snprintf(directory, PATH_MAX, "%s/yvex", base);
    if (base[0] != '/' || count <= 0 || count >= PATH_MAX) return 0;
    count = snprintf(path, PATH_MAX, "%s/model.conf", directory);
    if (count <= 0 || count >= PATH_MAX)
        return 0;
    return 1;
}

/* Purpose: validate or create one private client configuration directory.
 * Inputs: absolute directory. Effects: may create the final yvex component mode 0700.
 * Failure: symlink, foreign owner, non-directory, or unsafe permissions refuse.
 * Boundary: parent XDG configuration directory must already exist. */
static int model_config_directory(const char *directory)
{
    struct stat status;
    char parent[PATH_MAX], resolved[PATH_MAX], *slash;
    (void)snprintf(parent, sizeof(parent), "%s", directory);
    slash = strrchr(parent, '/');
    if (!slash || slash == parent) return 0;
    *slash = '\0';
    if (lstat(parent, &status) != 0 || !S_ISDIR(status.st_mode) ||
        status.st_uid != geteuid() || !realpath(parent, resolved) ||
        strcmp(parent, resolved))
        return 0;
    if (mkdir(directory, 0700) != 0 && errno != EEXIST) return 0;
    if (lstat(directory, &status) != 0 || !S_ISDIR(status.st_mode) ||
        S_ISLNK(status.st_mode) || status.st_uid != geteuid() ||
        (status.st_mode & 0077u) != 0u)
        return 0;
    return 1;
}

/* Purpose: atomically persist one selected model alias without reading model payloads.
 * Inputs: sealed bounded config. Effects: replaces one mode-0600 XDG file through rename.
 * Failure: unsafe paths or I/O remove the temporary file and preserve the old selection.
 * Boundary: selection is client configuration, not artifact or runtime admission. */
static int model_config_write(const client_model_config *config)
{
    char directory[PATH_MAX], path[PATH_MAX], temporary[PATH_MAX];
    FILE *output = NULL;
    int fd = -1, ok = 0, count;
    count = model_config_paths(directory, path)
                ? snprintf(temporary, sizeof(temporary), "%s/.model.%lu", directory,
                           (unsigned long)getpid())
                : -1;
    if (!model_config_paths(directory, path) || !model_config_directory(directory) ||
        count <= 0 || (size_t)count >= sizeof(temporary))
        return 0;
    fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) return 0;
    output = fdopen(fd, "w");
    if (!output) {
        (void)close(fd);
        (void)unlink(temporary);
        return 0;
    }
    fd = -1;
    ok = fprintf(output,
                 "name\t%s\nartifact\t%s\nbinding\t%s\ntarget\t%s\nbackend\t%s\ncontext\t%llu\n",
                 config->name, config->artifact, config->binding, config->target,
                 config->backend, config->context) > 0 &&
         fflush(output) == 0 && fsync(fileno(output)) == 0;
    if (fclose(output) != 0) ok = 0;
    output = NULL;
    if (ok) ok = rename(temporary, path) == 0;
    if (!ok) (void)unlink(temporary);
    return ok;
}

/* Purpose: load one owner-validated selected-model configuration.
 * Inputs: empty output. Effects: reads one bounded regular mode-0600 file.
 * Failure: malformed, duplicate/missing, foreign, symlinked, or oversized input refuses.
 * Boundary: paths remain inert until yvexd independently authenticates them. */
static int model_config_read(client_model_config *config)
{
    char directory[PATH_MAX], path[PATH_MAX], line[PATH_MAX + 32u];
    struct stat status;
    FILE *input;
    int fd, fields = 0;
    if (!config || !model_config_paths(directory, path) ||
        lstat(path, &status) != 0 || !S_ISREG(status.st_mode) ||
        S_ISLNK(status.st_mode) || status.st_uid != geteuid() ||
        (status.st_mode & 0077u) != 0u)
        return 0;
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &status) != 0 || !S_ISREG(status.st_mode)) {
        if (fd >= 0) (void)close(fd);
        return 0;
    }
    input = fdopen(fd, "r");
    if (!input) {
        (void)close(fd);
        return 0;
    }
    memset(config, 0, sizeof(*config));
    while (fgets(line, sizeof(line), input)) {
        char *value = strchr(line, '\t'), *newline;
        if (!value || !(newline = strchr(value + 1, '\n')) || newline[1]) {
            fields = -1;
            break;
        }
        *value++ = '\0';
        *newline = '\0';
        if (!strcmp(line, "name") && !config->name[0] && value[0] &&
            strlen(value) < sizeof(config->name) &&
            snprintf(config->name, sizeof(config->name), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "artifact") && !config->artifact[0] && value[0] &&
                 strlen(value) < sizeof(config->artifact) &&
                 snprintf(config->artifact, sizeof(config->artifact), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "binding") && !config->binding[0] && value[0] &&
                 strlen(value) < sizeof(config->binding) &&
                 snprintf(config->binding, sizeof(config->binding), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "target") && !config->target[0] && value[0] &&
                 strlen(value) < sizeof(config->target) &&
                 snprintf(config->target, sizeof(config->target), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "backend") && !config->backend[0] && value[0] &&
                 strlen(value) < sizeof(config->backend) &&
                 snprintf(config->backend, sizeof(config->backend), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "context") && !config->context &&
                 parse_u64(value, &config->context, 0))
            fields++;
        else {
            fields = -1;
            break;
        }
    }
    if (ferror(input) || fclose(input) != 0) fields = -1;
    return fields == 6 && config->artifact[0] == '/' && config->binding[0] == '/' &&
           (!strcmp(config->backend, "cpu") || !strcmp(config->backend, "cuda"));
}

/* Purpose: parse and atomically select one explicit model alias for the next daemon start.
 * Inputs: product model-use argv. Effects: writes only private client configuration.
 * Failure: incomplete or malformed facts refuse without replacing the prior selection.
 * Boundary: does not open, verify, materialize, or switch a running model. */
static int model_use_command(int argc, char **argv)
{
    client_model_config config;
    int index;
    memset(&config, 0, sizeof(config));
    if (argc < 8 || !argv[3][0] || strlen(argv[3]) >= sizeof(config.name) ||
        snprintf(config.name, sizeof(config.name), "%s", argv[3]) <= 0)
        return 2;
    (void)snprintf(config.target, sizeof(config.target), "%s", "deepseek4-v4-flash");
    (void)snprintf(config.backend, sizeof(config.backend), "%s", "cuda");
    config.context = 4096u;
    for (index = 4; index < argc; ++index) {
        if (!strcmp(argv[index], "--artifact") && index + 1 < argc &&
            strlen(argv[index + 1]) < sizeof(config.artifact))
            (void)snprintf(config.artifact, sizeof(config.artifact), "%s", argv[++index]);
        else if (!strcmp(argv[index], "--runtime-binding") && index + 1 < argc &&
                 strlen(argv[index + 1]) < sizeof(config.binding))
            (void)snprintf(config.binding, sizeof(config.binding), "%s", argv[++index]);
        else if (!strcmp(argv[index], "--target") && index + 1 < argc &&
                 strlen(argv[index + 1]) < sizeof(config.target))
            (void)snprintf(config.target, sizeof(config.target), "%s", argv[++index]);
        else if (!strcmp(argv[index], "--backend") && index + 1 < argc &&
                 strlen(argv[index + 1]) < sizeof(config.backend))
            (void)snprintf(config.backend, sizeof(config.backend), "%s", argv[++index]);
        else if (!strcmp(argv[index], "--context") && index + 1 < argc) {
            if (!parse_u64(argv[++index], &config.context, 0)) return 2;
        } else return 2;
    }
    if (!config.name[0] || config.artifact[0] != '/' || config.binding[0] != '/' ||
        (strcmp(config.backend, "cpu") && strcmp(config.backend, "cuda")) ||
        !config.target[0])
        return 2;
    if (!model_config_write(&config)) {
        fprintf(stderr, "yvex: selected model configuration could not be written safely\n");
        return 1;
    }
    printf("selected model: %s (restart runtime to apply)\n", config.name);
    return 0;
}

/* Purpose: render one selected model configuration without opening its artifact.
 * Inputs: none. Effects: reads private XDG configuration and prints compact facts.
 * Failure: missing selection returns one actionable refusal. Boundary: yvexd remains admission authority. */
static int model_config_show(void)
{
    client_model_config config;
    if (!model_config_read(&config)) {
        fprintf(stderr,
                "yvex: no selected model\n"
                "hint: use `yvex model use NAME --artifact FILE "
                "--runtime-binding FILE`\n");
        return 1;
    }
    printf("%-20s backend=%s context=%llu\n  artifact=%s\n  binding=%s\n",
           config.name, config.backend, config.context, config.artifact, config.binding);
    return 0;
}

/* Purpose: exec one exact argument vector through a colocated product binary or PATH fallback.
 * Inputs: binary name and null-terminated vector. Effects: replaces the client process on success.
 * Failure: reports both colocated and PATH lookup failure. Boundary: no compatibility dispatch. */
static int exec_sibling_vector(const char *binary, char *const arguments[])
{
    char executable[PATH_MAX], sibling[PATH_MAX];
    ssize_t count;
    count = readlink("/proc/self/exe", executable, sizeof(executable) - 1u);
    if (count > 0 && (size_t)count < sizeof(executable)) {
        char *slash;
        executable[count] = '\0';
        slash = strrchr(executable, '/');
        if (slash) {
            *slash = '\0';
            if (snprintf(sibling, sizeof(sibling), "%s/%s", executable,
                         binary) > 0)
                execv(sibling, arguments);
        }
    }
    execvp(binary, arguments);
    fprintf(stderr, "yvex: cannot execute %s: %s\n", binary, strerror(errno));
    return 1;
}

/* Purpose: project a product subcommand tail into one sibling argument vector.
 * Inputs: binary name, argv, and prefix count. Effects: allocates then delegates exec.
 * Failure: reports allocation or sibling failure and frees temporary argv. Boundary: no old alias mapping. */
static int exec_sibling(const char *binary, int argc, char **argv, int skip)
{
    char **arguments = calloc((size_t)argc + 1u, sizeof(*arguments));
    int index, out = 0, status;
    if (!arguments) return 1;
    arguments[out++] = (char *)binary;
    for (index = skip; index < argc; ++index) arguments[out++] = argv[index];
    arguments[out] = NULL;
    status = exec_sibling_vector(binary, arguments);
    free(arguments);
    return status;
}

/* Purpose: start yvexd from explicit argv or the selected model configuration.
 * Inputs: product runtime-start argv. Effects: replaces the client with one foreground daemon.
 * Failure: missing/unsafe selection refuses with one configuration hint. Boundary: yvexd revalidates all facts. */
static int runtime_start(int argc, char **argv)
{
    client_model_config config;
    char context[32];
    char *arguments[14];
    int count = 0;
    if (argc > 3) return exec_sibling("yvexd", argc, argv, 3);
    if (!model_config_read(&config)) {
        fprintf(stderr,
                "yvex: no selected model\nhint: use `yvex model use NAME --artifact FILE --runtime-binding FILE`\n");
        return 1;
    }
    (void)snprintf(context, sizeof(context), "%llu", config.context);
    arguments[count++] = "yvexd";
    arguments[count++] = "--model";
    arguments[count++] = config.artifact;
    arguments[count++] = "--runtime-binding";
    arguments[count++] = config.binding;
    arguments[count++] = "--target";
    arguments[count++] = config.target;
    arguments[count++] = "--backend";
    arguments[count++] = config.backend;
    arguments[count++] = "--context";
    arguments[count++] = context;
    arguments[count] = NULL;
    return exec_sibling_vector("yvexd", arguments);
}

/* Purpose: dispatch only the runtime-client lane of the unified yvex grammar.
 * Inputs: process argv not claimed by the offline route table. Effects: selects one protocol/admin path.
 * Failure: returns stable parser or runtime status. Boundary: it cannot enter an offline engine handler. */
int yvex_client_dispatch(int argc, char **argv)
{
    const char *command = argc > 1 ? argv[1] : "chat";
    if (!strcmp(command, "help") || !strcmp(command, "--help") ||
        !strcmp(command, "-h")) {
        if (!strcmp(command, "help") && argc != 2) {
            fprintf(stderr, "yvex: unknown help topic: %s\n", argv[2]);
            return 2;
        }
        print_help(stdout);
        return 0;
    }
    if (!strcmp(command, "version") || !strcmp(command, "--version")) {
        printf("yvex %s protocol=%u\n", yvex_version_string(),
               YVEX_LOCAL_PROTOCOL_VERSION);
        return 0;
    }
    if (!strcmp(command, "chat")) {
        return chat_command(argc, argv);
    }
    if (!strcmp(command, "run")) {
        return run_command(argc, argv);
    }
    if (!strcmp(command, "runtime")) {
        const char *action = argc > 2 ? argv[2] : NULL;
        if (!action) return 2;
        if (!strcmp(action, "start")) return runtime_start(argc, argv);
        if (!strcmp(action, "status")) {
            if (argc != 3 && !(argc == 4 && !strcmp(argv[3], "--json"))) return 2;
            return runtime_status(argc == 4);
        }
        if (!strcmp(action, "watch")) {
            if (argc != 3) return 2;
            return runtime_events(0);
        }
        if (!strcmp(action, "trace")) {
            if (argc != 3 && !(argc == 4 && !strcmp(argv[3], "--follow"))) return 2;
            return runtime_events(1);
        }
        if (!strcmp(action, "stop")) {
            if (argc != 3) return 2;
            return administration(YVEX_CLIENT_OP_RUNTIME_STOP, NULL, 0);
        }
        return 2;
    }
    if (!strcmp(command, "session")) return session_command(argc, argv);
    if (!strcmp(command, "model")) {
        if (argc == 3 && (!strcmp(argv[2], "show") || !strcmp(argv[2], "list")))
            return model_config_show();
        if (argc >= 4 && !strcmp(argv[2], "use")) return model_use_command(argc, argv);
        return 2;
    }
    fprintf(stderr, "yvex: unknown command: %s\nhint: use `yvex help`\n", command);
    return 2;
}
