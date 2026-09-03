/*
 * Runtime-facing commands are deliberately thin local-protocol clients. Even though the yvex ELF
 * also contains finite offline-engine adapters, this lane cannot open artifacts, initialize CUDA,
 * or call generation directly; every hosted client operation crosses the server protocol boundary.
 * The file also owns the linear interactive console. Operation identity and argument schemas come
 * from the compiled registry; terminal state and rendering remain client-owned projections.
 */
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <build_commit.h>
#include <operator/registry.h>
#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include "src/cli/private.h"
#include "src/cli/render/private.h"
#include <yvex/internal/core.h>
#include <yvex/server.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#define CLIENT_REPL_LINE_MAX 65536u
#define CLIENT_REPL_HISTORY_MAX 64u
typedef struct {
    unsigned long long maximum_new_tokens;
    yvex_provider_sampling sampling;
    char first_image[YVEX_SERVER_STATE_PATH_CAP], last_image[YVEX_SERVER_STATE_PATH_CAP];
    yvex_client_media_execution media_execution;
    int text_policy_explicit;
    yvex_reasoning_policy reasoning_policy;
} client_turn_options;
typedef yvex_cli_engine_binding client_engine_binding;
typedef struct {
    client_engine_binding engine;
    const char *positionals[3];
    size_t positional_count;
    int json;
} client_session_arguments;
typedef struct {
    client_engine_binding engine;
    char session[YVEX_SERVER_SESSION_NAME_CAP];
    atomic_int done, interrupts, force_exit;
    pthread_t thread;
    sigset_t previous_mask;
    int ready;
} client_turn_signals;
typedef struct {
    struct termios saved;
    int active;
} client_turn_terminal;
typedef struct {
    char *entry[CLIENT_REPL_HISTORY_MAX];
    size_t count;
} client_repl_history;
static volatile sig_atomic_t repl_signal_state;
static int console_status(const client_engine_binding *engine, const char *session_name);
static int console_status_fetch(const client_engine_binding *engine,
                                const char *session_name, yvex_client_message *message,
                                yvex_error *err);
static void render_console_status(
    const yvex_client_message *message,
    const yvex_cli_model_profile_selection *product, int startup);
static int client_error(const yvex_error *err)
{
    fprintf(stderr, "yvex: %s\n", yvex_error_message(err));
    if (yvex_error_code(err) == YVEX_ERR_IO)
        fprintf(stderr, "hint: start one with `yvex serve`; then use `yvex model load`\n");
    return 1;
}
static const char *reasoning_policy_name(yvex_reasoning_policy policy)
{
    switch (policy) {
    case YVEX_REASONING_DISABLED: return "none";
    case YVEX_REASONING_LOW: return "low";
    case YVEX_REASONING_ENABLED: return "medium";
    case YVEX_REASONING_MAXIMUM: return "xhigh";
    case YVEX_REASONING_SOURCE_DEFAULT: return "source-default";
    case YVEX_REASONING_POLICY_COUNT: break;
    }
    return "unknown";
}
void yvex_cli_client_request_init(yvex_client_request *request, yvex_client_operation operation)
{
    static unsigned long long next_request = 1u;
    yvex_provider_request defaults;
    yvex_provider_request_default(&defaults);
    memset(request, 0, sizeof(*request));
    request->schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request->operation = operation;
    request->request_number = next_request++;
    request->maximum_new_tokens = defaults.maximum_output_tokens;
    request->stochastic = defaults.sampling.stochastic;
    request->seed_present = defaults.sampling.seed_present;
    request->seed = defaults.sampling.seed;
    request->temperature = defaults.sampling.temperature;
    request->top_k = defaults.sampling.top_k;
    request->top_p = defaults.sampling.top_p;
    request->min_p = defaults.sampling.min_p;
    request->typical_p = defaults.sampling.typical_p;
    request->reasoning_policy = YVEX_REASONING_DISABLED;
}
#define request_init yvex_cli_client_request_init
static void request_engine_bind(yvex_client_request *request, const client_engine_binding *engine)
{
    if (!engine) return;
    (void)snprintf(request->model_alias, sizeof(request->model_alias), "%s",
                   engine->alias);
    request->engine_generation = engine->generation;
}
static int engine_binding_capture(client_engine_binding *engine, const yvex_console_status *status,
                                  yvex_error *err)
{
    if (!engine || !status || !status->model_alias[0] ||
        !status->engine_generation ||
        (engine->alias[0] && strcmp(engine->alias, status->model_alias)) ||
        (engine->generation && engine->generation != status->engine_generation)) {
        yvex_error_set(err, YVEX_ERR_STATE, "client.engine-binding",
                       "session belongs to another or stale engine generation");
        return YVEX_ERR_STATE;
    }
    (void)snprintf(engine->alias, sizeof(engine->alias), "%s",
                   status->model_alias);
    engine->generation = status->engine_generation;
    return YVEX_OK;
}
static void turn_options_init(client_turn_options *options)
{
    memset(options, 0, sizeof(*options));
    yvex_provider_sampling_default(&options->sampling);
    options->reasoning_policy = YVEX_REASONING_POLICY_COUNT;
}
static int turn_condition_path(char output[YVEX_SERVER_STATE_PATH_CAP], const char *source) {
    char *resolved; if (!output || !source || !source[0] || !(resolved = realpath(source, NULL))) return 0;
    if (strlen(resolved) >= YVEX_SERVER_STATE_PATH_CAP) {
        free(resolved);
        return 0;
    }
    (void)snprintf(output, YVEX_SERVER_STATE_PATH_CAP, "%s", resolved);
    free(resolved);
    return 1;
}
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
static int parse_duration_milliseconds(const char *text,
                                       unsigned long long *milliseconds)
{
    const char *cursor = text;
    unsigned long long seconds = 0ull, fraction = 0ull, scale = 100ull;
    int digits = 0;
    if (!text || !text[0] || !milliseconds) return 0;
    while (*cursor >= '0' && *cursor <= '9') {
        if (!yvex_core_u64_mul(seconds, 10ull, &seconds) ||
            !yvex_core_u64_add(seconds, (unsigned long long)(*cursor - '0'),
                               &seconds))
            return 0;
        cursor++;
        digits = 1;
    }
    if (!digits) return 0;
    if (*cursor == '.') {
        cursor++;
        digits = 0;
        while (*cursor >= '0' && *cursor <= '9' && scale) {
            fraction += (unsigned long long)(*cursor - '0') * scale;
            scale /= 10ull;
            cursor++;
            digits = 1;
        }
        if (!digits || (*cursor >= '0' && *cursor <= '9')) return 0;
    }
    if (*cursor || !yvex_core_u64_mul(seconds, 1000ull, milliseconds) ||
        !yvex_core_u64_add(*milliseconds, fraction, milliseconds))
        return 0;
    return *milliseconds != 0ull;
}
int yvex_cli_client_request_open(yvex_client **client, const yvex_client_request *request, yvex_error *err)
{
    int rc = yvex_client_connect(client, NULL, err);
    if (rc == YVEX_OK) rc = yvex_client_send(*client, request, err);
    if (rc != YVEX_OK) yvex_client_close(client);
    return rc;
}
#define request_open yvex_cli_client_request_open
static int cancellation_request(const client_engine_binding *engine,
                                const char *session)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_GENERATION_CANCEL);
    request_engine_bind(&request, engine);
    (void)snprintf(request.session_name, sizeof(request.session_name), "%s",
                   session);
    rc = request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    yvex_client_close(&client);
    return rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ACK;
}
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
            if (cancellation_request(&state->engine, state->session)) break;
            (void)nanosleep(&delay, NULL);
        }
    }
    return NULL;
}
static void turn_signals_open(client_turn_signals *state,
                              const client_engine_binding *engine,
                              const char *session)
{
    sigset_t signals;
    memset(state, 0, sizeof(*state));
    if (engine) state->engine = *engine;
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
static void turn_terminal_open(client_turn_terminal *terminal)
{
    struct termios quiet;
    memset(terminal, 0, sizeof(*terminal));
    if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &terminal->saved) != 0)
        return;
    quiet = terminal->saved;
    quiet.c_lflag &= (tcflag_t)~(ECHO | ECHONL);
    if (tcsetattr(STDIN_FILENO, TCSANOW, &quiet) == 0) terminal->active = 1;
}
static void turn_terminal_close(client_turn_terminal *terminal)
{
    if (!terminal || !terminal->active) return;
    /* Chat deliberately does not draft the next turn while generation owns the
     * output stream. Discard queued editing bytes before restoring the editor. */
    (void)tcflush(STDIN_FILENO, TCIFLUSH);
    (void)tcsetattr(STDIN_FILENO, TCSANOW, &terminal->saved);
    terminal->active = 0;
}
static const char *backend_name(yvex_backend_kind backend)
{
    return backend == YVEX_BACKEND_KIND_CUDA ? "CUDA" : "CPU";
}
static const char *engine_kind_name(yvex_server_engine_kind kind)
{
    if (kind == YVEX_SERVER_ENGINE_TEXT) return "text";
    if (kind == YVEX_SERVER_ENGINE_MEDIA) return "media";
    return "none";
}
static const char *execution_strategy_name(
    yvex_server_execution_strategy strategy)
{
    if (strategy == YVEX_SERVER_EXECUTION_TARGET_ONLY) return "target-only";
    if (strategy == YVEX_SERVER_EXECUTION_SPECULATIVE) return "speculative";
    return "n/a";
}
static const char *engine_execution_name(
    yvex_server_engine_kind kind, yvex_server_execution_strategy strategy)
{
    return kind == YVEX_SERVER_ENGINE_MEDIA ? "media"
                                             : execution_strategy_name(strategy);
}
static int runtime_summary_fetch(yvex_server_summary *summary, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_RUNTIME_STATUS);
    rc = request_open(&client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_STATUS)
        *summary = message.runtime;
    else if (rc == YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "client.status",
                       "server returned an unexpected response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc;
}
static int host_status(int json)
{
    yvex_server_summary summary;
    yvex_error err;
    int rc = runtime_summary_fetch(&summary, &err);
    if (rc == YVEX_OK) yvex_cli_host_status_render(stdout, &summary, json);
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static const char *engine_state_name(yvex_server_engine_state state)
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
static void render_engine(const yvex_server_engine_summary *engine, int json)
{
    if (json) {
        fputs("{\"alias\":", stdout);
        yvex_cli_out_json_string(stdout, engine->alias);
        printf(",\"generation\":%llu,\"state\":", engine->generation);
        yvex_cli_out_json_string(stdout, engine_state_name(engine->state));
        printf(",\"backend\":%u,\"engine_kind\":", (unsigned int)engine->backend);
        yvex_cli_out_json_string(stdout, engine_kind_name(engine->engine_kind));
        fputs(",\"execution_strategy\":", stdout);
        yvex_cli_out_json_string(
            stdout, execution_strategy_name(engine->execution_strategy));
        printf(",\"execution_ready\":%s,\"continuous_batching\":%s,"
               "\"context_capacity\":%llu,\"prefill_chunk_tokens\":%llu,"
               "\"maximum_new_tokens\":%llu,\"maximum_sessions\":%llu,"
               "\"concurrent_sequences\":%llu,\"active_work\":%llu,\"sessions\":%llu,"
               "\"mapped_package_bytes\":%llu,\"resident_host_bytes\":%llu,"
               "\"resident_device_bytes\":%llu,\"prepared_bytes\":%llu,\"target\":",
               engine->execution_ready ? "true" : "false",
               engine->continuous_batching_ready ? "true" : "false",
               engine->context_capacity, engine->prefill_chunk_tokens,
               engine->maximum_new_tokens, engine->maximum_sessions,
               engine->concurrent_sequences, engine->active_work,
               engine->session_count, engine->mapped_package_bytes,
               engine->resident_host_bytes, engine->resident_device_bytes,
               engine->prepared_bytes);
        yvex_cli_out_json_string(stdout, engine->target_id);
        fputs(",\"model_identity\":", stdout);
        yvex_cli_out_json_string(stdout, engine->runtime_model_identity);
        fputs(",\"specialization_identity\":", stdout);
        yvex_cli_out_json_string(stdout, engine->specialization_identity);
        fputc('}', stdout);
        return;
    }
    printf("%-24s generation=%llu state=%-10s %s/%s/%s work=%llu sessions=%llu "
           "mapped=%.2f GiB prepared=%.2f GiB device=%.2f GiB\n",
           engine->alias, engine->generation, engine_state_name(engine->state),
           backend_name(engine->backend), engine_kind_name(engine->engine_kind),
           execution_strategy_name(engine->execution_strategy),
           engine->active_work, engine->session_count,
           (double)engine->mapped_package_bytes / 1073741824.0,
           (double)engine->prepared_bytes / 1073741824.0,
           (double)engine->resident_device_bytes / 1073741824.0);
}
static int engine_generation_resolve(const char *alias,
                                     unsigned long long *generation,
                                     yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int found = 0, rc;
    if (!alias || !alias[0] || !generation) return YVEX_ERR_INVALID_ARG;
    request_init(&request, YVEX_CLIENT_OP_ENGINE_LIST);
    rc = request_open(&client, &request, err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, err);
        if (rc != YVEX_OK || message.kind == YVEX_CLIENT_MESSAGE_ACK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            rc = message.status;
            yvex_error_set(err, (yvex_status)rc, "client.engine-generation",
                           message.reason);
            break;
        }
        if (message.kind != YVEX_CLIENT_MESSAGE_ENGINE) {
            rc = YVEX_ERR_FORMAT;
            yvex_error_set(err, YVEX_ERR_FORMAT, "client.engine-generation",
                           "server returned an invalid engine catalog");
            break;
        }
        if (!strcmp(message.engine.alias, alias) &&
            message.engine.state == YVEX_SERVER_ENGINE_LOADED) {
            *generation = message.engine.generation;
            found = 1;
        }
    }
    yvex_client_close(&client);
    if (rc == YVEX_OK && !found) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, YVEX_ERR_STATE, "client.engine-generation",
                       "requested engine is not loaded");
    }
    return rc;
}
static int session_arguments_parse(int argc, char **argv, size_t consumed,
                                   client_session_arguments *arguments,
                                   yvex_error *err)
{
    size_t index;
    memset(arguments, 0, sizeof(*arguments));
    for (index = consumed + 1u; index < (size_t)argc; ++index) {
        if (!strcmp(argv[index], "--model")) {
            if (index + 1u >= (size_t)argc) {
                yvex_error_set(err, YVEX_ERR_INVALID_ARG, "client.session-route",
                               "--model requires a model alias");
                return YVEX_ERR_INVALID_ARG;
            }
            const char *alias = argv[++index];
            if (!alias[0] || strlen(alias) >= sizeof(arguments->engine.alias)) {
                yvex_error_set(err, YVEX_ERR_INVALID_ARG, "client.session-route",
                               "model alias exceeds the protocol bound");
                return YVEX_ERR_INVALID_ARG;
            }
            (void)snprintf(arguments->engine.alias,
                           sizeof(arguments->engine.alias), "%s", alias);
            continue;
        }
        if (!strcmp(argv[index], "--json")) {
            arguments->json = 1;
            continue;
        }
        if (arguments->positional_count ==
            sizeof(arguments->positionals) / sizeof(arguments->positionals[0])) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "client.session-route",
                           "session command positional bound exceeded");
            return YVEX_ERR_BOUNDS;
        }
        arguments->positionals[arguments->positional_count++] = argv[index];
    }
    if (arguments->engine.alias[0])
        return engine_generation_resolve(arguments->engine.alias,
                                         &arguments->engine.generation, err);
    yvex_error_clear(err);
    return YVEX_OK;
}
static int runtime_adapter_session_bound(yvex_operator_runtime_adapter adapter)
{
    switch (adapter) {
    case YVEX_OPERATOR_RUNTIME_SESSION_ATTACH:
    case YVEX_OPERATOR_RUNTIME_SESSION_CANCEL:
    case YVEX_OPERATOR_RUNTIME_SESSION_CLOSE:
    case YVEX_OPERATOR_RUNTIME_SESSION_DETACH:
    case YVEX_OPERATOR_RUNTIME_SESSION_FORK:
    case YVEX_OPERATOR_RUNTIME_SESSION_LIST:
    case YVEX_OPERATOR_RUNTIME_SESSION_NEW:
    case YVEX_OPERATOR_RUNTIME_SESSION_RESET:
    case YVEX_OPERATOR_RUNTIME_SESSION_STATE_RESTORE:
    case YVEX_OPERATOR_RUNTIME_SESSION_STATE_SAVE:
    case YVEX_OPERATOR_RUNTIME_SESSION_SHOW: return 1;
    default: return 0;
    }
}
static int engine_control(yvex_client_operation operation, const char *alias)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, operation);
    if (!alias || !alias[0]) {
        fputs(operation == YVEX_CLIENT_OP_ENGINE_LOAD
                  ? "yvex: engine load requires PROFILE\n"
                    "hint: normal workflow uses `yvex model load [MODEL]`\n"
                  : "yvex: engine unload requires ENGINE\n",
              stderr);
        return 2;
    }
    snprintf(request.model_alias, sizeof(request.model_alias), "%s", alias);
    if (operation == YVEX_CLIENT_OP_ENGINE_UNLOAD) {
        rc = engine_generation_resolve(alias, &request.engine_generation, &err);
        if (rc != YVEX_OK) return client_error(&err);
    }
    rc = request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
        yvex_error_set(&err, (yvex_status)message.status, "client.engine",
                       message.reason);
        rc = message.status;
    } else if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ENGINE) {
        render_engine(&message.engine, 0);
    } else if (rc == YVEX_OK) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "client.engine",
                       "server returned an unexpected engine response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int engine_catalog(const char *filter, int json)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    unsigned long long count = 0ull;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_ENGINE_LIST);
    rc = request_open(&client, &request, &err);
    if (json && !filter)
        fputs("{\"schema\":\"yvex.engine.list.v1\",\"engines\":[", stdout);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            yvex_error_set(&err, (yvex_status)message.status, "client.engines",
                           message.reason);
            rc = message.status;
            break;
        }
        if (message.kind == YVEX_CLIENT_MESSAGE_ACK) break;
        if (message.kind != YVEX_CLIENT_MESSAGE_ENGINE) {
            yvex_error_set(&err, YVEX_ERR_FORMAT, "client.engines",
                           "server returned an unexpected engine-list response");
            rc = YVEX_ERR_FORMAT;
            break;
        }
        if (filter && strcmp(filter, message.engine.alias)) continue;
        if (filter && count) continue;
        if (json && filter)
            fputs("{\"schema\":\"yvex.engine.v1\",\"engine\":", stdout);
        else if (json && count)
            fputc(',', stdout);
        render_engine(&message.engine, json);
        count++;
    }
    if (json && (!filter || count)) fputs(filter ? "}\n" : "]}\n", stdout);
    else if (rc == YVEX_OK && !count && !filter) puts("no engines known to this host");
    if (rc == YVEX_OK && filter && !count) {
        yvex_error_set(&err, YVEX_ERR_STATE, "client.engine",
                       "requested engine is not known to the host");
        rc = YVEX_ERR_STATE;
    }
    yvex_client_close(&client);
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int host_memory(int json)
{
    yvex_server_summary summary;
    yvex_error err;
    int rc = runtime_summary_fetch(&summary, &err);
    if (rc == YVEX_OK) yvex_cli_host_memory_render(stdout, &summary, json);
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int host_logs(int json_output, int detailed, int follow)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    yvex_cli_terminal_style style;
    yvex_cli_watch_renderer watch;
    char json[2048];
    int rc;
    if (!json_output) {
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%shost logs%s · %s\n\n", style.accent, style.reset,
               follow ? "recent history, then live events · Ctrl-C to stop"
                      : "recent retained history");
        yvex_cli_watch_renderer_open(&watch, detailed);
    }
    request_init(&request, follow ? YVEX_CLIENT_OP_RUNTIME_WATCH
                                 : YVEX_CLIENT_OP_RUNTIME_TRACE);
    request.trace_level = json_output || detailed ? YVEX_SERVER_TRACE_FULL
                                                  : YVEX_SERVER_TRACE_STAGES;
    rc = request_open(&client, &request, &err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            yvex_error_set(&err, (yvex_status)message.status, "client.host.logs",
                           message.reason);
            rc = message.status;
            break;
        }
        if (message.kind == YVEX_CLIENT_MESSAGE_ACK) break;
        if (message.kind != YVEX_CLIENT_MESSAGE_EVENT) {
            yvex_error_set(&err, YVEX_ERR_FORMAT, "client.host.logs",
                           "server returned an unexpected log response");
            rc = YVEX_ERR_FORMAT;
            break;
        }
        if (!json_output)
            (void)yvex_cli_watch_renderer_event(&watch, &message.event, NULL);
        else if (yvex_server_event_json(&message.event, json, sizeof(json), &err) == YVEX_OK) {
            fputs(json, stdout);
            fflush(stdout);
        }
        if (message.event.kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE)
            break;
    }
    if (!json_output) yvex_cli_watch_renderer_finish(&watch);
    yvex_client_close(&client);
    return rc == YVEX_OK ? 0 : client_error(&err);
}

static int administration_request(yvex_client_request *request,
                                  int render_mode)
{
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_cli_session_table_fact *sessions = NULL;
    size_t session_count = 0u, session_capacity = 0u;
    yvex_error err;
    int rc;
    rc = request_open(&client, request, &err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            yvex_error_set(&err, (yvex_status)message.status, "client.request",
                           message.reason);
            rc = message.status;
            break;
        }
        if (render_mode > 0 && message.kind == YVEX_CLIENT_MESSAGE_SESSION) {
            if (session_count == session_capacity) {
                size_t next = session_capacity ? session_capacity * 2u : 8u;
                void *grown = realloc(sessions, next * sizeof(*sessions));
                if (!grown) {
                    yvex_error_set(&err, YVEX_ERR_NOMEM, "client.sessions",
                                   "session table allocation failed");
                    rc = YVEX_ERR_NOMEM;
                    break;
                }
                sessions = grown;
                session_capacity = next;
            }
            yvex_cli_session_table_fact_set(&sessions[session_count++], &message);
            if (render_mode >= 3) break;
        }
        else if (render_mode == 0 &&
                 message.kind == YVEX_CLIENT_MESSAGE_SESSION) {
            printf("%-20s %-10s position=%llu turns=%llu\n",
                   message.session_name,
                   yvex_server_session_state_name(message.session_state),
                   message.final_position, message.turn_count);
        }
        else if (message.kind == YVEX_CLIENT_MESSAGE_ACK) {
            if (!render_mode && message.state_checkpoint.schema_version)
                printf("%s position=%llu bytes=%llu digest=%s\n",
                       message.reason, message.state_checkpoint.committed_sequence_length,
                       message.state_checkpoint.file_bytes,
                       message.state_checkpoint.file_digest);
            else if (!render_mode)
                printf("%s\n", message.reason[0] ? message.reason : "ok");
            break;
        }
        if (render_mode <= 0) break;
    }
    if (rc == YVEX_OK && render_mode > 0) {
        if (render_mode == 1)
            rc = yvex_cli_session_table_render(stdout, sessions, session_count);
        else
            rc = yvex_cli_session_json_render(stdout, sessions, session_count,
                                               render_mode == 2);
        if (rc != YVEX_OK)
            yvex_error_set(&err, (yvex_status)rc, "client.sessions",
                           "cannot render session catalog");
    }
    free(sessions);
    yvex_client_close(&client);
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int administration_bound(yvex_client_operation operation,
                                const client_engine_binding *engine,
                                const char *session_name, int render_mode)
{
    yvex_client_request request;
    request_init(&request, operation);
    request_engine_bind(&request, engine);
    if (session_name)
        snprintf(request.session_name, sizeof(request.session_name), "%s",
                 session_name);
    return administration_request(&request, render_mode);
}
static int state_checkpoint(yvex_client_operation operation,
                            const client_engine_binding *engine,
                            const char *session_name, const char *path,
                            unsigned long long maximum_file_bytes)
{
    yvex_client_request request;
    request_init(&request, operation);
    request_engine_bind(&request, engine);
    snprintf(request.session_name, sizeof(request.session_name), "%s",
             session_name);
    snprintf(request.state_path, sizeof(request.state_path), "%s", path);
    request.maximum_state_file_bytes = maximum_file_bytes;
    return administration_request(&request, 0);
}
static int session_fork(const client_engine_binding *engine,
                        const char *source, const char *child,
                        unsigned long long maximum_prefix_bytes)
{
    yvex_client_request request;
    request_init(&request, YVEX_CLIENT_OP_SESSION_FORK);
    request_engine_bind(&request, engine);
    snprintf(request.session_name, sizeof(request.session_name), "%s", source);
    snprintf(request.fork_session_name, sizeof(request.fork_session_name),
             "%s", child);
    request.maximum_prefix_bytes = maximum_prefix_bytes;
    return administration_request(&request, 0);
}
static void generation_progress_finish(int *active, int terminate_line)
{
    if (!active || !*active) return;
    fputs("\r\033[2K", stdout);
    if (terminate_line) fputc('\n', stdout);
    fflush(stdout);
    *active = 0;
}
static void generation_progress_event(const yvex_server_event *event,
                                      int conversation,
                                      const yvex_cli_terminal_style *style,
                                      int *active)
{
    if (!conversation) return;
    if (event->engine_kind == YVEX_SERVER_ENGINE_MEDIA) {
        printf("\r\033[2K%smedia · %s", style->accent,
               event->phase[0] ? event->phase : "executing");
        if (event->value_b)
            printf(" · %llu/%llu", event->value_a, event->value_b);
        printf("%s", style->reset);
        if (event->kind == YVEX_SERVER_EVENT_GENERATION_COMPLETED ||
            event->kind == YVEX_SERVER_EVENT_GENERATION_CANCELLED ||
            event->kind == YVEX_SERVER_EVENT_GENERATION_FAILED ||
            strstr(event->phase, "complete"))
            putchar('\n');
        fflush(stdout);
        *active = event->kind != YVEX_SERVER_EVENT_GENERATION_COMPLETED &&
                  event->kind != YVEX_SERVER_EVENT_GENERATION_CANCELLED &&
                  event->kind != YVEX_SERVER_EVENT_GENERATION_FAILED;
    } else if (event->kind == YVEX_SERVER_EVENT_PREFILL_STARTED) {
        printf("\r\033[2K%sprocessing %llu input tokens · 0/%llu · 0%%%s",
               style->accent, event->value_a, event->value_a, style->reset);
        fflush(stdout);
        *active = 1;
    } else if (event->kind == YVEX_SERVER_EVENT_PREFILL_PROGRESS) {
        printf("\r\033[2K%sprocessing %llu input tokens · %llu/%llu · %.1f%%%s",
               style->accent, event->value_b, event->value_a, event->value_b,
               event->value_b ? 100.0 * (double)event->value_a /
                                      (double)event->value_b : 0.0,
               style->reset);
        fflush(stdout);
    } else if (event->kind == YVEX_SERVER_EVENT_PREFILL_COMPLETED) {
        printf("\r\033[2K%sprocessing %llu input tokens · %llu/%llu · 100%%%s\n",
               style->success, event->value_a, event->value_a, event->value_a,
               style->reset);
        fflush(stdout);
        *active = 0;
    }
}
static int generation_turn(const client_engine_binding *engine,
                           const char *session_name,
                           const unsigned char *prompt,
                           unsigned long long prompt_bytes,
                           const client_turn_options *options, int conversation,
                           yvex_server_engine_kind engine_kind,
                           unsigned long long context_capacity,
                           int *connection_lost) {
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    client_turn_signals signals;
    client_turn_terminal terminal;
    yvex_cli_stream_renderer renderer;
    yvex_error err;
    yvex_cli_terminal_style style;
    FILE *status_output = conversation ? stdout : stderr;
    yvex_reasoning_policy reasoning_policy = options->reasoning_policy;
    int rc, started = 0, progress_active = 0, renderer_finished = 0;
    int terminal_output = isatty(fileno(stdout));
    if (connection_lost) *connection_lost = 0;
    if (!yvex_reasoning_policy_valid(reasoning_policy) &&
        console_status_fetch(engine, session_name, &message, &err) != YVEX_OK)
        return client_error(&err);
    if (!yvex_reasoning_policy_valid(reasoning_policy))
        reasoning_policy = message.console.reasoning_policy;
    yvex_cli_terminal_style_get(status_output, &style);
    yvex_cli_stream_renderer_open(&renderer, stdout, terminal_output);
    request_init(&request, YVEX_CLIENT_OP_GENERATION_TURN);
    request_engine_bind(&request, engine);
    snprintf(request.session_name, sizeof(request.session_name), "%s",
             session_name);
    request.prompt = prompt;
    request.prompt_bytes = prompt_bytes;
    {
        const char *paths[] = {options->first_image, options->last_image};
        const yvex_client_media_condition_role roles[] = {
            YVEX_CLIENT_MEDIA_CONDITION_FIRST, YVEX_CLIENT_MEDIA_CONDITION_LAST};
        size_t index;
        for (index = 0u; index < sizeof(paths) / sizeof(paths[0]); ++index) {
            yvex_client_media_condition *condition;
            if (!paths[index][0]) continue;
            condition = request.media_conditions + request.media_condition_count++;
            condition->schema_version = YVEX_CLIENT_MEDIA_CONDITION_SCHEMA_V1;
            condition->kind = YVEX_CLIENT_MEDIA_CONDITION_IMAGE;
            condition->role = roles[index];
            (void)snprintf(condition->source_path, sizeof(condition->source_path),
                           "%s", paths[index]);
        }
    }
    request.maximum_new_tokens = options->maximum_new_tokens;
    if (engine_kind == YVEX_SERVER_ENGINE_MEDIA) {
        request.media_execution = options->media_execution;
        request.media_execution.schema_version =
            YVEX_CLIENT_MEDIA_EXECUTION_SCHEMA_V1;
        if (options->sampling.seed_present) {
            request.media_execution.present |= YVEX_CLIENT_MEDIA_EXECUTION_SEED;
            request.media_execution.seed = options->sampling.seed;
        }
        request.stochastic = 0;
        request.seed_present = 0;
        request.seed = 0ull;
        request.temperature = 1.0;
        request.top_k = 0ull;
        request.top_p = 1.0;
        request.min_p = 0.0;
        request.typical_p = 1.0;
    } else {
        request.stochastic = options->sampling.stochastic;
        request.seed_present = options->sampling.seed_present;
        request.seed = options->sampling.seed;
        request.temperature = options->sampling.temperature;
        request.top_k = options->sampling.top_k;
        request.top_p = options->sampling.top_p;
        request.min_p = options->sampling.min_p;
        request.typical_p = options->sampling.typical_p;
    }
    request.reasoning_policy = reasoning_policy;
    turn_signals_open(&signals, engine, session_name);
    turn_terminal_open(&terminal);
    rc = request_open(&client, &request, &err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_TURN_STARTED) {
            continue;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_EVENT) {
            generation_progress_event(&message.event, conversation, &style,
                                      &progress_active);
        } else if (message.kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
            generation_progress_finish(&progress_active, 0);
            rc = yvex_cli_stream_renderer_write(
                &renderer, message.stream_channel, message.bytes,
                message.byte_count);
            if (rc == YVEX_OK) rc = yvex_cli_out_flush(stdout);
            if (rc != YVEX_OK) {
                yvex_error_set(&err, YVEX_ERR_IO, "client.turn.render",
                               "terminal stream rendering failed");
                break;
            }
            started = 1;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE) {
            generation_progress_finish(&progress_active, 0);
            rc = yvex_cli_stream_renderer_finish(&renderer,
                                                  conversation || terminal_output);
            renderer_finished = 1;
            if (rc == YVEX_OK) rc = yvex_cli_out_flush(stdout);
            if (rc != YVEX_OK) {
                yvex_error_set(&err, YVEX_ERR_IO, "client.turn.render",
                               "terminal stream finalization failed");
                break;
            }
            yvex_cli_out_turn_complete(status_output, &message,
                                       context_capacity, &style);
            break;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            generation_progress_finish(&progress_active, 1);
            rc = yvex_cli_stream_renderer_finish(&renderer,
                                                  conversation || terminal_output);
            renderer_finished = 1;
            if (rc == YVEX_OK) rc = yvex_cli_out_flush(stdout);
            if (rc != YVEX_OK) {
                yvex_error_set(&err, YVEX_ERR_IO, "client.turn.render",
                               "terminal stream finalization failed");
                break;
            }
            if (message.partial_turn.available)
                fprintf(status_output,
                        "%spartial%s · %llu committed token%s · position %llu · %s\n",
                        style.warning, style.reset,
                        message.partial_turn.committed_token_count,
                        message.partial_turn.committed_token_count == 1u ? "" : "s",
                        message.partial_turn.final_committed_position,
                        message.partial_turn.reset_required
                            ? "reset required (/reset)"
                            : "recovery unavailable");
            yvex_error_set(&err, (yvex_status)message.status, "client.turn",
                           message.reason);
            rc = message.status;
            break;
        }
    }
    if (rc != YVEX_OK) generation_progress_finish(&progress_active, 1);
    if (!renderer_finished && started) {
        (void)yvex_cli_stream_renderer_finish(&renderer,
                                              conversation || terminal_output);
        (void)yvex_cli_out_flush(stdout);
    }
    yvex_client_close(&client);
    turn_terminal_close(&terminal);
    {
        int interrupted = turn_signals_close(&signals);
        if (interrupted) {
            if (conversation) {
                const char *text = interrupted == 2 ? "cancelled · leaving chat"
                                   : message.session_state == YVEX_SERVER_SESSION_PARTIAL
                                       ? "cancelled · session partial · use /reset"
                                       : "cancelled";
                printf("%s%s%s\n", style.warning, text, style.reset);
            }
            return interrupted == 2 ? 131 : 130;
        }
    }
    if (connection_lost && rc != YVEX_OK &&
        yvex_error_code(&err) == YVEX_ERR_IO &&
        !strcmp(yvex_error_where(&err), "server.protocol"))
        *connection_lost = 1;
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int session_ensure(const client_engine_binding *engine, const char *name)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_SESSION_SHOW);
    request_engine_bind(&request, engine);
    snprintf(request.session_name, sizeof(request.session_name), "%s", name);
    rc = request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    yvex_client_close(&client);
    if (rc == YVEX_OK && message.kind != YVEX_CLIENT_MESSAGE_ERROR) return 0;
    return administration_bound(YVEX_CLIENT_OP_SESSION_NEW, engine, name, -1);
}
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
static void repl_history_close(client_repl_history *history)
{
    size_t index;
    for (index = 0u; index < history->count; ++index) free(history->entry[index]);
    memset(history, 0, sizeof(*history));
}
static void repl_signal_handler(int number)
{
    sig_atomic_t interrupts = repl_signal_state & 3;
    if (number == SIGWINCH)
        repl_signal_state |= 4;
    else if (interrupts < 2)
        repl_signal_state = (repl_signal_state & ~3) | (interrupts + 1);
}
static size_t repl_previous(const char *line, size_t cursor)
{
    if (!cursor) return 0u;
    cursor--;
    while (cursor && (((unsigned char)line[cursor] & 0xc0u) == 0x80u)) cursor--;
    return cursor;
}
static size_t repl_next(const char *line, size_t count, size_t cursor)
{
    if (cursor >= count) return count;
    cursor++;
    while (cursor < count && (((unsigned char)line[cursor] & 0xc0u) == 0x80u)) cursor++;
    return cursor;
}
static size_t repl_columns(const char *line, size_t start, size_t count)
{
    size_t columns = 0u;
    while (start < count) {
        if (((unsigned char)line[start] & 0xc0u) != 0x80u) columns++;
        start++;
    }
    return columns;
}
static void repl_redraw(const char *prompt, const char *line, size_t count,
                        size_t cursor)
{
    fputs("\r\033[2K", stdout);
    fputs(prompt, stdout);
    if (count) (void)fwrite(line, 1u, count, stdout);
    if (cursor < count)
        fprintf(stdout, "\033[%zuD", repl_columns(line, cursor, count));
    fflush(stdout);
}
static int repl_replace_line(char **line, size_t *count, size_t *capacity,
                             size_t *cursor, const char *replacement,
                             const char *prompt)
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
    *cursor = *count;
    repl_redraw(prompt, *line, *count, *cursor);
    return 1;
}
static int repl_complete_slash(char **line, size_t *count, size_t *capacity,
                               size_t *cursor, const char *prompt)
{
    const yvex_operator_descriptor *match = NULL;
    size_t index, matches = 0u;
    if (!*line || !*count || (*line)[0] != '/' || strchr(*line, ' ')) return 0;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *candidate = &yvex_operator_descriptors[index];
        if (strcmp(candidate->slash_projection, "none") &&
            !strncmp(candidate->slash_projection, *line, *count)) {
            match = candidate;
            matches++;
        }
    }
    if (matches == 1u) {
        char replacement[128];
        (void)snprintf(replacement, sizeof(replacement), "%s%s",
                       match->slash_projection, match->argument_count ? " " : "");
        return repl_replace_line(line, count, capacity, cursor, replacement, prompt);
    }
    if (matches > 1u) return 1;
    return 0;
}
static int repl_insert_byte(char **line, size_t *count, size_t *capacity,
                            size_t *cursor, unsigned char byte)
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
    memmove(*line + *cursor + 1u, *line + *cursor, *count - *cursor + 1u);
    (*line)[(*cursor)++] = (char)byte;
    (*count)++;
    (*line)[*count] = '\0';
    return 1;
}
static void repl_erase(char *line, size_t *count, size_t *cursor, int backward)
{
    size_t start = backward ? repl_previous(line, *cursor) : *cursor;
    size_t end = backward ? *cursor : repl_next(line, *count, *cursor);
    if (start == end) return;
    memmove(line + start, line + end, *count - end + 1u);
    *count -= end - start;
    *cursor = start;
}
static size_t repl_escape_read(unsigned char sequence[8])
{
    size_t length = 0u;
    while (length < 8u) {
        struct pollfd input = {.fd = STDIN_FILENO, .events = POLLIN};
        if (poll(&input, 1u, 25) <= 0 || !(input.revents & POLLIN) ||
            read(STDIN_FILENO, &sequence[length], 1u) != 1)
            break;
        length++;
        if ((sequence[0] == '[' && length >= 2u && sequence[length - 1u] >= '@') ||
            (sequence[0] == 'O' && length == 2u) ||
            (sequence[0] != '[' && sequence[0] != 'O'))
            break;
    }
    return length;
}
static int repl_read_line(const char *prompt, const char *initial,
                          const client_repl_history *history,
                          char **output, size_t *output_count)
{
    struct termios saved, raw;
    char *line = NULL;
    size_t count = 0u, capacity = 0u, cursor = 0u, selected = history->count;
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
    if (initial && !repl_replace_line(&line, &count, &capacity, &cursor,
                                      initial, prompt))
        goto done;
    if (!initial) repl_redraw(prompt, "", 0u, 0u);
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
                repl_redraw(prompt, line ? line : "", count, cursor);
            }
            continue;
        }
        if (got <= 0) {
            result = 0;
            break;
        }
        if (byte == '\r' || byte == '\n') {
            if (paste) {
                if (!repl_insert_byte(&line, &count, &capacity, &cursor, '\n')) break;
                fputs("\r\n... ", stdout);
                fflush(stdout);
                continue;
            }
            fputs("\r\n", stdout);
            result = 1;
            break;
        }
        if (byte == 4u && !paste) {
            fputs("\r\n", stdout);
            result = 0;
            break;
        }
        if ((byte == 1u || byte == 5u) && !paste) {
            cursor = byte == 1u ? 0u : count;
            repl_redraw(prompt, line ? line : "", count, cursor);
            continue;
        }
        if (byte == '\t' && !paste) {
            if (!repl_complete_slash(&line, &count, &capacity, &cursor, prompt))
                fputc('\a', stdout);
            fflush(stdout);
            continue;
        }
        if (byte == '\f' && !paste)
            fputs("\033[2J\033[H", stdout);
        if ((byte == '\f' && !paste) || byte == 8u || byte == 127u) {
            if (byte != '\f') repl_erase(line, &count, &cursor, 1);
            repl_redraw(prompt, line ? line : "", count, cursor);
            continue;
        }
        if (byte == 27u) {
            unsigned char sequence[8];
            size_t length = repl_escape_read(sequence);
            if (length == 5u && !memcmp(sequence, "[200~", 5u)) paste = 1;
            else if (length == 5u && !memcmp(sequence, "[201~", 5u)) paste = 0;
            else if (!paste && length == 2u && sequence[0] == '[' && sequence[1] == 'A' && selected) {
                selected--;
                if (!repl_replace_line(&line, &count, &capacity, &cursor,
                                       history->entry[selected], prompt))
                    break;
            } else if (!paste && length == 2u && sequence[0] == '[' &&
                       sequence[1] == 'B' && selected < history->count) {
                selected++;
                if (!repl_replace_line(&line, &count, &capacity, &cursor,
                                       selected == history->count ? "" : history->entry[selected],
                                       prompt))
                    break;
            } else if (!paste && length == 2u && sequence[0] == '[' &&
                       sequence[1] == 'C') {
                cursor = repl_next(line ? line : "", count, cursor);
                repl_redraw(prompt, line ? line : "", count, cursor);
            } else if (!paste && length == 2u && sequence[0] == '[' &&
                       sequence[1] == 'D') {
                cursor = repl_previous(line ? line : "", cursor);
                repl_redraw(prompt, line ? line : "", count, cursor);
            } else if (!paste && ((length == 2u &&
                        ((sequence[0] == '[' && sequence[1] == 'H') ||
                         (sequence[0] == 'O' && sequence[1] == 'H'))) ||
                       (length == 3u && (!memcmp(sequence, "[1~", 3u) ||
                                        !memcmp(sequence, "[7~", 3u))))) {
                cursor = 0u;
                repl_redraw(prompt, line ? line : "", count, cursor);
            } else if (!paste && ((length == 2u &&
                        ((sequence[0] == '[' && sequence[1] == 'F') ||
                         (sequence[0] == 'O' && sequence[1] == 'F'))) ||
                       (length == 3u && (!memcmp(sequence, "[4~", 3u) ||
                                        !memcmp(sequence, "[8~", 3u))))) {
                cursor = count;
                repl_redraw(prompt, line ? line : "", count, cursor);
            } else if (!paste && length == 3u && !memcmp(sequence, "[3~", 3u)) {
                repl_erase(line, &count, &cursor, 0);
                repl_redraw(prompt, line ? line : "", count, cursor);
            }
            continue;
        }
        {
            size_t old_count = count;
            if (!repl_insert_byte(&line, &count, &capacity, &cursor, byte)) break;
            if (cursor == count && old_count + 1u == count) {
                (void)fwrite(&byte, 1u, 1u, stdout);
                fflush(stdout);
            } else repl_redraw(prompt, line, count, cursor);
        }
    }
done:
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
static int repl_switch_session(const client_engine_binding *engine,
                               char current[YVEX_SERVER_SESSION_NAME_CAP],
                               const char *next, int create)
{
    yvex_cli_terminal_style style;
    if (!next || !next[0] || strlen(next) >= YVEX_SERVER_SESSION_NAME_CAP) return 0;
    if (!strcmp(current, next)) return 1;
    if (create && administration_bound(YVEX_CLIENT_OP_SESSION_NEW, engine, next,
                                       -1) != 0)
        return 0;
    if (administration_bound(YVEX_CLIENT_OP_SESSION_ATTACH, engine, next, -1) != 0)
        return 0;
    (void)administration_bound(YVEX_CLIENT_OP_SESSION_DETACH, engine, current, -1);
    (void)snprintf(current, YVEX_SERVER_SESSION_NAME_CAP, "%s", next);
    yvex_cli_terminal_style_get(stdout, &style);
    printf("%ssession%s · %s\n", style.success, style.reset, current);
    return 1;
}
static int slash_alias_matches(const char *aliases, const char *line,
                               size_t extent)
{
    const char *cursor = aliases;
    if (!aliases || !strcmp(aliases, "none")) return 0;
    while (*cursor) {
        const char *end = strchr(cursor, ',');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (length == extent && !memcmp(cursor, line, extent)) return 1;
        if (!end) break;
        cursor = end + 1;
    }
    return 0;
}
static const yvex_operator_descriptor *slash_descriptor(const char *line,
                                                         const char **argument)
{
    const char *end = strchr(line, ' ');
    size_t extent = end ? (size_t)(end - line) : strlen(line), index;
    *argument = end ? end + 1 : NULL;
    while (*argument && **argument == ' ') (*argument)++;
    if (*argument && !**argument) *argument = NULL;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
        if ((strcmp(descriptor->slash_projection, "none") &&
             strlen(descriptor->slash_projection) == extent &&
             !memcmp(descriptor->slash_projection, line, extent)) ||
            slash_alias_matches(descriptor->slash_aliases, line, extent))
            return descriptor;
    }
    return NULL;
}
static void repl_reasoning_policy(
    const client_engine_binding *engine, const char *session,
    client_turn_options *options,
    yvex_reasoning_policy policy)
{
    yvex_client_message status;
    yvex_cli_terminal_style style;
    yvex_error err;
    const char *name = reasoning_policy_name(policy);
    yvex_cli_terminal_style_get(stdout, &style);
    if (console_status_fetch(engine, session, &status, &err) != YVEX_OK) {
        (void)client_error(&err);
        return;
    }
    if (!status.console.explicit_reasoning_channel_supported &&
        policy != YVEX_REASONING_DISABLED) {
        printf("%sreasoning unavailable%s · active model has no explicit channel\n",
               style.warning, style.reset);
        return;
    }
    options->reasoning_policy = policy;
    printf("%sreasoning%s · %s until changed\n", style.accent,
           style.reset, name);
}
static int repl_command(const char *line, const client_engine_binding *engine,
                        char current[YVEX_SERVER_SESSION_NAME_CAP],
                        unsigned long long *generated_session,
                        client_turn_options *options)
{
    const yvex_operator_descriptor *descriptor;
    yvex_cli_operator_invocation invocation;
    const char *argument;
    char generated[YVEX_SERVER_SESSION_NAME_CAP];
    int result = 1, status;
    if (line[0] != '/') return 0;
    descriptor = slash_descriptor(line, &argument);
    if (!descriptor) {
        yvex_cli_terminal_style style;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%sunknown command:%s %.*s\n", style.error, style.reset,
               (int)(strchr(line, ' ') ? (size_t)(strchr(line, ' ') - line) : strlen(line)),
               line);
        return 1;
    }
    status = yvex_cli_operator_slash_parse(descriptor, argument, &invocation);
    if (status) {
        yvex_cli_terminal_style style;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%sinvalid arguments for %s:%s %s\n", style.error,
               descriptor->slash_projection, style.reset, invocation.message);
        yvex_cli_operator_invocation_close(&invocation);
        return 1;
    }
    argument = invocation.argument_count ? invocation.arguments[0] : NULL;
    if (descriptor->lane == YVEX_OPERATOR_LANE_REPL_LOCAL) {
        result = descriptor->repl_adapter == YVEX_OPERATOR_REPL_QUIT ? 2 : 1;
        yvex_cli_operator_invocation_close(&invocation);
        return result;
    }
    switch (descriptor->runtime_adapter) {
    case YVEX_OPERATOR_RUNTIME_HELP:
        if (invocation.argument_count)
            (void)yvex_client_render_help_path(invocation.argument_count,
                                                invocation.arguments, 0, 0);
        else
            yvex_cli_out_repl_catalog();
        break;
    case YVEX_OPERATOR_RUNTIME_CONSOLE_STATUS:
        (void)console_status(engine, current);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_LIST:
        (void)administration_bound(YVEX_CLIENT_OP_SESSION_LIST, engine, NULL, 1);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_SHOW:
        (void)administration_bound(YVEX_CLIENT_OP_SESSION_SHOW, engine,
                                   argument ? argument : current, 0);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_NEW:
        if (!argument) {
            (void)snprintf(generated, sizeof(generated), "chat-%llu", (*generated_session)++);
            argument = generated;
        }
        (void)repl_switch_session(engine, current, argument, 1);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_ATTACH:
        (void)repl_switch_session(engine, current, argument, 0);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_DETACH:
        result = 2;
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_FORK:
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_RESET:
        (void)administration_bound(YVEX_CLIENT_OP_SESSION_RESET, engine, current, 0);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_CLOSE:
        (void)administration_bound(YVEX_CLIENT_OP_SESSION_CLOSE, engine, current, 0);
        result = 3;
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_CANCEL:
        {
            yvex_cli_terminal_style style;
            int cancelled = cancellation_request(engine, current);
            yvex_cli_terminal_style_get(stdout, &style);
            printf("%s%s%s\n", cancelled ? style.warning : style.dim,
                   cancelled ? "cancel requested" : "no active turn", style.reset);
        }
        break;
    case YVEX_OPERATOR_RUNTIME_REASONING_DISABLED:
        repl_reasoning_policy(engine, current, options, YVEX_REASONING_DISABLED);
        break;
    case YVEX_OPERATOR_RUNTIME_REASONING_LOW:
        repl_reasoning_policy(engine, current, options, YVEX_REASONING_LOW);
        break;
    case YVEX_OPERATOR_RUNTIME_REASONING_ENABLED:
        repl_reasoning_policy(engine, current, options, YVEX_REASONING_ENABLED);
        break;
    case YVEX_OPERATOR_RUNTIME_REASONING_MAXIMUM:
        repl_reasoning_policy(engine, current, options, YVEX_REASONING_MAXIMUM);
        break;
    default:
        {
            yvex_cli_terminal_style style;
            yvex_cli_terminal_style_get(stdout, &style);
            printf("%scommand unavailable in chat%s\n", style.warning, style.reset);
        }
        break;
    }
    yvex_cli_operator_invocation_close(&invocation);
    return result;
}
static int repl_reconnect(client_engine_binding *engine, const char *session,
                          yvex_client_message *status)
{
    yvex_cli_terminal_style style;
    yvex_error err;
    if (session_ensure(engine, session) != 0 ||
        administration_bound(YVEX_CLIENT_OP_SESSION_ATTACH, engine, session,
                             -1) != 0 ||
        console_status_fetch(engine, session, status, &err) != YVEX_OK ||
        engine_binding_capture(engine, &status->console, &err) != YVEX_OK)
        return 0;
    yvex_cli_terminal_style_get(stdout, &style);
    printf("%sreconnected%s · session %s\n", style.success, style.reset,
           session);
    return 1;
}
static int chat(const client_engine_binding *selected_engine,
                const yvex_cli_model_profile_selection *selected_model,
                const char *session_name,
                unsigned long long maximum_new_tokens,
                const client_turn_options *initial_options)
{
    client_engine_binding engine = {0};
    client_turn_options options;
    client_repl_history history;
    yvex_client_message status;
    yvex_error err;
    yvex_cli_terminal_style style;
    struct sigaction action, prior_interrupt, prior_resize;
    char current[YVEX_SERVER_SESSION_NAME_CAP];
    char prompt[YVEX_MODEL_LIBRARY_NAME_CAP + 128u];
    char *draft = NULL;
    unsigned long long generated_session = 1u;
    int closed = 0, connected = 1, attached = 0, result = 0;
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
    options = *initial_options;
    options.maximum_new_tokens = maximum_new_tokens;
    if (selected_engine) engine = *selected_engine;
    (void)snprintf(current, sizeof(current), "%s", session_name);
    if (session_ensure(&engine, current) != 0) {
        result = 1;
        goto cleanup;
    }
    if (administration_bound(YVEX_CLIENT_OP_SESSION_ATTACH, &engine, current,
                             -1) != 0) {
        result = 1;
        goto cleanup;
    }
    attached = 1;
    if (console_status_fetch(&engine, current, &status, &err) != YVEX_OK ||
        engine_binding_capture(&engine, &status.console, &err) != YVEX_OK) {
        result = client_error(&err);
        goto cleanup;
    }
    if ((status.engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
         options.maximum_new_tokens) ||
        (status.engine_kind != YVEX_SERVER_ENGINE_MEDIA &&
         (options.first_image[0] || options.last_image[0] ||
          options.media_execution.trajectory !=
              YVEX_CLIENT_MEDIA_TRAJECTORY_DEFAULT ||
          options.media_execution.present || options.sampling.seed_present))) {
        fprintf(stderr, "yvex: selected options do not apply to the attached engine\n");
        result = 2;
        goto cleanup;
    }
    render_console_status(&status, selected_model, 1);
    options.reasoning_policy = status.console.reasoning_policy;
    yvex_cli_terminal_style_get(stdout, &style);
    for (;;) {
        char *line = NULL;
        size_t count = 0u;
        int input;
        const char *prompt_model =
            selected_model && selected_model->model_name[0]
                ? selected_model->model_name
            : selected_model && selected_model->model_selector[0]
                ? selected_model->model_selector
            : status.console.model_alias[0] ? status.console.model_alias : "yvex";
        (void)snprintf(prompt, sizeof(prompt), "%.31s%.127s%s>%.31s ", style.accent,
                       prompt_model, connected ? "" : " [disconnected]",
                       style.reset);
        input = repl_read_line(prompt, draft, &history, &line, &count);
        free(draft);
        draft = NULL;
        if (input == -2) continue;
        if (input <= 0) break;
        repl_signal_state &= ~3;
        if (!count) {
            free(line);
            continue;
        }
        if (line[0] == '/') {
            client_turn_terminal terminal;
            const char *argument = NULL;
            const yvex_operator_descriptor *descriptor =
                slash_descriptor(line, &argument);
            int local = descriptor &&
                (descriptor->lane == YVEX_OPERATOR_LANE_REPL_LOCAL ||
                 descriptor->runtime_adapter == YVEX_OPERATOR_RUNTIME_HELP);
            turn_terminal_open(&terminal);
            if (!connected && descriptor && !local &&
                !repl_reconnect(&engine, current, &status)) {
                turn_terminal_close(&terminal);
                draft = line;
                continue;
            }
            if (!connected && descriptor && !local) connected = 1;
            int command = repl_command(line, &engine, current, &generated_session,
                                       &options);
            turn_terminal_close(&terminal);
            free(line);
            if (command == 3) {
                closed = 1;
                break;
            }
            if (command == 2) break;
            continue;
        }
        if (!connected) {
            client_turn_terminal terminal;
            int reconnected;
            turn_terminal_open(&terminal);
            reconnected = repl_reconnect(&engine, current, &status);
            turn_terminal_close(&terminal);
            if (!reconnected) {
                draft = line;
                continue;
            }
        }
        connected = 1;
        repl_history_push(&history, line);
        {
            int connection_lost = 0;
            int turn = generation_turn(
                &engine, current, (const unsigned char *)line,
                (unsigned long long)count, &options, 1,
                status.engine_kind,
                status.console.context_capacity, &connection_lost);
            if (connection_lost) connected = 0;
            if (turn == 131) {
                free(line);
                break;
            }
        }
        free(line);
    }
cleanup:
    free(draft);
    repl_history_close(&history);
    if (attached && !closed && connected)
        (void)administration_bound(YVEX_CLIENT_OP_SESSION_DETACH, &engine,
                                   current, -1);
    (void)sigaction(SIGINT, &prior_interrupt, NULL);
    (void)sigaction(SIGWINCH, &prior_resize, NULL);
    return result;
}
static int chat_command(int argc, char **argv, size_t consumed)
{
    const char *model = NULL, *session = "main";
    client_engine_binding engine = {0};
    yvex_cli_model_profile_selection selection = {0};
    client_turn_options options;
    unsigned long long maximum_new_tokens = 0u;
    int index, saw_model = 0, saw_session = 0, saw_maximum = 0;
    turn_options_init(&options);
    for (index = (int)consumed + 1; index < argc; ++index) {
        if (!strcmp(argv[index], "--model") && !saw_model && index + 1 < argc) {
            model = argv[++index];
            saw_model = 1;
        } else if (!strcmp(argv[index], "--session") && !saw_session && index + 1 < argc) {
            session = argv[++index];
            saw_session = 1;
        } else if (!strcmp(argv[index], "--max-new-tokens") && !saw_maximum &&
                   index + 1 < argc) {
            if (!parse_u64(argv[++index], &maximum_new_tokens, 0)) return 2;
            saw_maximum = 1;
        } else if (!strcmp(argv[index], "--first-image") && !options.first_image[0] &&
                   index + 1 < argc) {
            if (!turn_condition_path(options.first_image, argv[++index])) return 2;
        } else if (!strcmp(argv[index], "--last-image") && !options.last_image[0] &&
                   index + 1 < argc) {
            if (!turn_condition_path(options.last_image, argv[++index])) return 2;
        } else if (!strcmp(argv[index], "--trajectory") &&
                   options.media_execution.trajectory ==
                       YVEX_CLIENT_MEDIA_TRAJECTORY_DEFAULT &&
                   index + 1 < argc) {
            const char *value = argv[++index];
            if (!strcmp(value, "preview"))
                options.media_execution.trajectory =
                    YVEX_CLIENT_MEDIA_TRAJECTORY_PREVIEW;
            else if (!strcmp(value, "released"))
                options.media_execution.trajectory =
                    YVEX_CLIENT_MEDIA_TRAJECTORY_RELEASED;
            else return 2;
        } else if (!strcmp(argv[index], "--width") &&
                   !(options.media_execution.present &
                     YVEX_CLIENT_MEDIA_EXECUTION_WIDTH) && index + 1 < argc) {
            if (!parse_u64(argv[++index], &options.media_execution.width, 0)) return 2;
            options.media_execution.present |= YVEX_CLIENT_MEDIA_EXECUTION_WIDTH;
        } else if (!strcmp(argv[index], "--height") &&
                   !(options.media_execution.present &
                     YVEX_CLIENT_MEDIA_EXECUTION_HEIGHT) && index + 1 < argc) {
            if (!parse_u64(argv[++index], &options.media_execution.height, 0)) return 2;
            options.media_execution.present |= YVEX_CLIENT_MEDIA_EXECUTION_HEIGHT;
        } else if (!strcmp(argv[index], "--duration") &&
                   !(options.media_execution.present &
                     YVEX_CLIENT_MEDIA_EXECUTION_DURATION) && index + 1 < argc) {
            if (!parse_duration_milliseconds(
                    argv[++index], &options.media_execution.duration_milliseconds))
                return 2;
            options.media_execution.present |= YVEX_CLIENT_MEDIA_EXECUTION_DURATION;
        } else if (!strcmp(argv[index], "--seed") &&
                   !options.sampling.seed_present && index + 1 < argc) {
            if (!parse_u64(argv[++index], &options.sampling.seed, 1)) return 2;
            options.sampling.seed_present = 1;
        } else {
            return 2;
        }
    }
    if (model && (!model[0] || strlen(model) >= YVEX_SERVER_MODEL_ALIAS_CAP)) return 2;
    if (!!(options.media_execution.present & YVEX_CLIENT_MEDIA_EXECUTION_WIDTH) !=
        !!(options.media_execution.present & YVEX_CLIENT_MEDIA_EXECUTION_HEIGHT))
        return 2;
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        fputs("yvex: chat requires a terminal\n"
              "programmatic inference: use the configured provider API\n",
              stderr);
        return 2;
    }
    {
        int media = options.first_image[0] || options.last_image[0] ||
                    options.media_execution.present ||
                    options.media_execution.trajectory !=
                        YVEX_CLIENT_MEDIA_TRAJECTORY_DEFAULT;
        int selected = yvex_cli_model_loaded_select(model, NULL, !media,
                                                     &engine, &selection);
        if (selected) return selected;
    }
    return chat(&engine, &selection, session, maximum_new_tokens, &options);
}
static int help_command(int argc, char **argv, size_t consumed)
{
    const char *path[16];
    size_t count = 0u, index;
    int advanced = 0, json = 0;
    for (index = consumed + 1u; index < (size_t)argc; ++index) {
        if (!strcmp(argv[index], "--advanced")) advanced = 1;
        else if (!strcmp(argv[index], "--json")) json = 1;
        else if (strcmp(argv[index], "--help") && strcmp(argv[index], "-h")) {
            if (count == sizeof(path) / sizeof(path[0])) return 2;
            path[count++] = argv[index];
        }
    }
    if (json && (advanced || count)) {
        fprintf(stderr, "yvex: help --json is the complete deterministic discovery document\n");
        return 2;
    }
    return yvex_client_render_help_path(count, path, advanced, json);
}
static int console_status_fetch(const client_engine_binding *engine,
                                const char *session_name,
                                yvex_client_message *message,
                                yvex_error *err)
{
    yvex_client_request request;
    yvex_client *client = NULL;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_CONSOLE_STATUS);
    request_engine_bind(&request, engine);
    (void)snprintf(request.session_name, sizeof(request.session_name), "%s", session_name);
    rc = request_open(&client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, message, err);
    if (rc == YVEX_OK && message->kind == YVEX_CLIENT_MESSAGE_ERROR) {
        yvex_error_set(err, (yvex_status)message->status, "client.console-status",
                       message->reason);
        rc = message->status;
    } else if (rc == YVEX_OK && message->kind != YVEX_CLIENT_MESSAGE_CONSOLE_STATUS) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "client.console-status",
                       "server returned an unexpected console status response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc;
}
static void render_console_status(
    const yvex_client_message *message,
    const yvex_cli_model_profile_selection *product, int startup)
{
    const yvex_console_status *status = &message->console;
    const char *target = product && product->model_name[0]
                             ? product->model_name
                         : product && product->model_selector[0]
                             ? product->model_selector
                         : status->model_alias[0] ? status->model_alias
                                                  : status->live_model_identity;
    const char *variant = product && product->quant_precision[0]
                              ? product->quant_precision
                          : product && product->variant[0] ? product->variant
                                                           : status->physical_variant_identity;
    const char *reasoning = reasoning_policy_name(status->reasoning_policy);
    yvex_cli_terminal_style style;
    yvex_cli_terminal_style_get(stdout, &style);
    if (startup) {
        printf("%sYVEX %s%s · %s%s%s\n", style.strong, yvex_version_string(),
               style.reset, style.accent, target, style.reset);
        printf("  %s%s%s · %s\n",
               status->runtime_ready ? style.success : style.warning,
               status->runtime_ready ? "● ready" : "● not ready", style.reset,
               status->attached ? "attached to resident runtime"
                                : "detached from runtime");
        printf("  %s · %s · %s", backend_name(status->backend),
               engine_execution_name(message->engine_kind,
                                     message->execution_strategy), variant);
        if (message->engine_kind == YVEX_SERVER_ENGINE_MEDIA)
            printf(" · direct media generation");
        else
            printf(" · context %llu/%llu", status->context_used,
                   status->context_capacity);
        printf("\n  session %s · position %llu · turns %llu · reasoning %s\n",
               status->session_name, status->position, status->turn_count,
               reasoning);
        if (message->partial_turn.available)
            printf("  %sPARTIAL%s · %llu committed token%s · reset required\n",
                   style.warning, style.reset,
                   message->partial_turn.committed_token_count,
                   message->partial_turn.committed_token_count == 1u ? "" : "s");
        printf("  %s/help%s commands · Ctrl-C cancels · Ctrl-D exits\n\n",
               style.accent, style.reset);
        return;
    }
    printf("%schat%s · ", style.strong, style.reset);
    printf("%s · %s · %s · variant %s · %s%s%s · %s · "
           "session %s · position %llu · turns %llu",
           target, backend_name(status->backend),
           engine_execution_name(message->engine_kind,
                                 message->execution_strategy),
           variant,
           status->runtime_ready ? style.success : style.warning,
           status->runtime_ready ? "● ready" : "● not ready", style.reset,
           status->attached ? "attached to resident runtime" : "detached from runtime",
           status->session_name, status->position, status->turn_count);
    if (message->engine_kind == YVEX_SERVER_ENGINE_MEDIA)
        printf(" · direct media generation");
    else
        printf(" · context %llu/%llu", status->context_used,
               status->context_capacity);
    if (status->kv_used_available) printf(" · KV %.2f MiB", (double)status->kv_used_bytes / 1048576.0);
    printf(" · reasoning %s · live %.12s", reasoning, status->live_model_identity);
    if (status->selected_model_available) printf(" · selected %.12s", status->selected_model_identity);
    if (message->partial_turn.available)
        printf(" · %sPARTIAL%s · %llu committed · reset required", style.warning,
               style.reset, message->partial_turn.committed_token_count);
    putchar('\n');
}
static int console_status(const client_engine_binding *engine,
                          const char *session_name)
{
    yvex_client_message message;
    yvex_error err;
    int rc = console_status_fetch(engine, session_name, &message, &err);
    if (rc == YVEX_OK) render_console_status(&message, NULL, 0);
    return rc == YVEX_OK ? 0 : client_error(&err);
}
int yvex_client_dispatch(const yvex_operator_descriptor *operation, int argc,
                         char **argv, size_t consumed)
{
    client_session_arguments session_arguments;
    yvex_error session_error;
    const client_engine_binding *session_engine = NULL;
    const char *name = consumed + 1u < (size_t)argc ? argv[consumed + 1u] : NULL;
    int session_bound = runtime_adapter_session_bound(operation->runtime_adapter);
    if (session_bound) {
        int rc = session_arguments_parse(argc, argv, consumed,
                                         &session_arguments, &session_error);
        if (rc != YVEX_OK) return client_error(&session_error);
        session_engine = &session_arguments.engine;
        name = session_arguments.positional_count
                   ? session_arguments.positionals[0] : NULL;
    }
    switch (operation->runtime_adapter) {
    case YVEX_OPERATOR_RUNTIME_CHAT: return chat_command(argc, argv, consumed);
    case YVEX_OPERATOR_RUNTIME_HOST_STATUS: {
        int json = 0, index;
        for (index = (int)consumed + 1; index < argc; ++index)
            json |= !strcmp(argv[index], "--json");
        return host_status(json);
    }
    case YVEX_OPERATOR_RUNTIME_ENGINE_LOAD:
        return engine_control(YVEX_CLIENT_OP_ENGINE_LOAD, name);
    case YVEX_OPERATOR_RUNTIME_ENGINE_UNLOAD:
        return engine_control(YVEX_CLIENT_OP_ENGINE_UNLOAD, name);
    case YVEX_OPERATOR_RUNTIME_MODEL_LOAD:
        return yvex_cli_model_load_command(argc, argv, consumed);
    case YVEX_OPERATOR_RUNTIME_MODEL_UNLOAD:
        return yvex_cli_model_unload_command(argc, argv, consumed);
    case YVEX_OPERATOR_RUNTIME_ENGINE_CATALOG: {
        int json = 0, index;
        const char *filter = !strcmp(operation->operation_id, "engine.show")
                                 ? name : NULL;
        for (index = (int)consumed + 1; index < argc; ++index)
            json |= !strcmp(argv[index], "--json");
        return engine_catalog(filter, json);
    }
    case YVEX_OPERATOR_RUNTIME_HOST_MEMORY: {
        int json = 0, index;
        for (index = (int)consumed + 1; index < argc; ++index)
            json |= !strcmp(argv[index], "--json");
        return host_memory(json);
    }
    case YVEX_OPERATOR_RUNTIME_HOST_LOGS: {
        int json = 0, detailed = 0, follow = 0, index;
        for (index = (int)consumed + 1; index < argc; ++index) {
            json |= !strcmp(argv[index], "--json");
            detailed |= !strcmp(argv[index], "--verbose");
            follow |= !strcmp(argv[index], "--follow");
        }
        if (json && detailed) {
            fputs("yvex: host logs accepts either --verbose or --json, not both\n",
                  stderr);
            return 2;
        }
        return host_logs(json, detailed, follow);
    }
    case YVEX_OPERATOR_RUNTIME_HOST_STOP:
        return administration_bound(YVEX_CLIENT_OP_RUNTIME_STOP, NULL, NULL, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_NEW:
        return administration_bound(YVEX_CLIENT_OP_SESSION_NEW, session_engine,
                                    name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_LIST:
        return administration_bound(YVEX_CLIENT_OP_SESSION_LIST, session_engine,
                                    NULL, session_arguments.json ? 2 : 1);
    case YVEX_OPERATOR_RUNTIME_SESSION_SHOW:
        return administration_bound(YVEX_CLIENT_OP_SESSION_SHOW, session_engine,
                                    name, session_arguments.json ? 3 : 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_ATTACH:
        return administration_bound(YVEX_CLIENT_OP_SESSION_ATTACH, session_engine,
                                    name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_DETACH:
        return administration_bound(YVEX_CLIENT_OP_SESSION_DETACH, session_engine,
                                    name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_RESET:
        return administration_bound(YVEX_CLIENT_OP_SESSION_RESET, session_engine,
                                    name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_FORK: {
        unsigned long long maximum_prefix_bytes;
        if (session_arguments.positional_count != 3u ||
            !parse_u64(session_arguments.positionals[2],
                       &maximum_prefix_bytes, 0)) {
            fputs("yvex: session fork requires a positive shared-prefix byte bound\n",
                  stderr);
            return 2;
        }
        return session_fork(session_engine, name,
                            session_arguments.positionals[1],
                            maximum_prefix_bytes);
    }
    case YVEX_OPERATOR_RUNTIME_SESSION_STATE_SAVE:
        if (session_arguments.positional_count != 2u) {
            fputs("yvex: state save requires SESSION PATH\n", stderr);
            return 2;
        }
        return state_checkpoint(YVEX_CLIENT_OP_SESSION_STATE_SAVE,
                                session_engine, name,
                                session_arguments.positionals[1], 0u);
    case YVEX_OPERATOR_RUNTIME_SESSION_STATE_RESTORE: {
        unsigned long long maximum_file_bytes;
        if (session_arguments.positional_count != 3u ||
            !parse_u64(session_arguments.positionals[2],
                       &maximum_file_bytes, 0)) {
            fputs("yvex: state restore requires a positive maximum file byte bound\n",
                  stderr);
            return 2;
        }
        return state_checkpoint(YVEX_CLIENT_OP_SESSION_STATE_RESTORE,
                                session_engine, name,
                                session_arguments.positionals[1],
                                maximum_file_bytes);
    }
    case YVEX_OPERATOR_RUNTIME_SESSION_CLOSE:
        return administration_bound(YVEX_CLIENT_OP_SESSION_CLOSE, session_engine,
                                    name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_CANCEL:
        puts(cancellation_request(session_engine, name)
                 ? "cancel requested" : "no active turn");
        return 0;
    case YVEX_OPERATOR_RUNTIME_HELP: return help_command(argc, argv, consumed);
    case YVEX_OPERATOR_RUNTIME_COMPLETION:
        return yvex_cli_completion_command(argc, argv, consumed);
    case YVEX_OPERATOR_RUNTIME_VERSION:
        printf("yvex %s protocol=%u registry=%s commit=%s\n", yvex_version_string(),
               YVEX_LOCAL_PROTOCOL_VERSION, yvex_operator_registry_identity,
               YVEX_BUILD_COMMIT);
        return 0;
    case YVEX_OPERATOR_RUNTIME_CONSOLE_STATUS:
        return console_status(NULL, name ? name : "main");
    case YVEX_OPERATOR_RUNTIME_REASONING_DISABLED:
    case YVEX_OPERATOR_RUNTIME_REASONING_LOW:
    case YVEX_OPERATOR_RUNTIME_REASONING_ENABLED:
    case YVEX_OPERATOR_RUNTIME_REASONING_MAXIMUM:
    case YVEX_OPERATOR_RUNTIME_COUNT: break;
    }
    fprintf(stderr, "yvex: unbound runtime adapter: %s\n", operation->adapter_id);
    return 2;
}
