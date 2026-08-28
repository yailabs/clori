/* Own the deterministic application loop and compose local UI intent with typed client I/O. */
#define _POSIX_C_SOURCE 200809L

#include "src/cli/input/private.h"
#include "src/cli/tui/private.h"

#include <errno.h>
#include <fcntl.h>
#include <locale.h>
#include <operator/registry.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>

#define TUI_FRAME_CAP 2097152u
#define TUI_REFRESH_NS 2000000000ull
#define TUI_LAUNCH_REFRESH_NS 500000000ull
#define TUI_REMOTE_WORKER_SCHEMA_V1 1u

typedef struct {
    unsigned int schema_version;
    int operation_status;
    unsigned int result_count;
    yvex_remote_model results[YVEX_TUI_REMOTE_CAP];
    char reason[YVEX_SERVER_REASON_CAP];
} tui_remote_message;

typedef struct {
    pid_t pid;
    int result_fd, cancel_fd, running;
    size_t received;
    unsigned long long started_ns;
    tui_remote_message message;
} tui_remote_worker;

typedef struct {
    pid_t pid;
    int diagnostic_fd, running, exit_known, exit_status, truncated;
    size_t diagnostic_count;
    char diagnostic[512];
} tui_acquisition_worker;

static unsigned long long monotonic_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0u;
    return (unsigned long long)now.tv_sec * 1000000000ull +
           (unsigned long long)now.tv_nsec;
}

static int descriptor_nonblocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static void worker_stop(pid_t pid, int process_group)
{
    unsigned int attempt;
    int status;
    (void)kill(process_group ? -pid : pid, SIGTERM);
    for (attempt = 0u; attempt < 4u; ++attempt) {
        pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid || (waited < 0 && errno == ECHILD)) return;
        (void)poll(NULL, 0u, 25);
    }
    (void)kill(process_group ? -pid : pid, SIGKILL);
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
}

static void remote_worker_init(tui_remote_worker *worker)
{
    memset(worker, 0, sizeof(*worker));
    worker->pid = -1;
    worker->result_fd = -1;
    worker->cancel_fd = -1;
}

static int remote_worker_start(tui_remote_worker *worker, const char *query,
                               yvex_error *err)
{
    int result_pipe[2] = {-1, -1}, cancel_pipe[2] = {-1, -1};
    pid_t pid;
    if (!worker || worker->running || !query || !query[0]) {
        yvex_error_set(err, YVEX_ERR_STATE, "cli.tui.discover",
                       "one non-empty remote search may be active");
        return YVEX_ERR_STATE;
    }
    if (pipe(result_pipe) != 0 || pipe(cancel_pipe) != 0) {
        if (result_pipe[0] >= 0) {
            (void)close(result_pipe[0]);
            (void)close(result_pipe[1]);
        }
        yvex_error_set(err, YVEX_ERR_IO, "cli.tui.discover",
                       "remote search handoff pipe could not be created");
        return YVEX_ERR_IO;
    }
    (void)fcntl(result_pipe[1], F_SETFD, FD_CLOEXEC);
    (void)fcntl(cancel_pipe[0], F_SETFD, FD_CLOEXEC);
    pid = fork();
    if (pid == 0) {
        tui_remote_message message;
        yvex_remote_search_options options;
        yvex_remote_catalog *catalog = NULL;
        yvex_error child_error;
        const unsigned char *bytes;
        size_t written = 0u;
        (void)close(result_pipe[0]);
        (void)close(cancel_pipe[1]);
        memset(&message, 0, sizeof(message));
        memset(&options, 0, sizeof(options));
        yvex_error_clear(&child_error);
        message.schema_version = TUI_REMOTE_WORKER_SCHEMA_V1;
        options.provider = YVEX_ACCOUNT_PROVIDER_HUGGINGFACE;
        options.query = query;
        options.page = 1u;
        options.page_size = 20u;
        message.operation_status = yvex_remote_model_search(&catalog, &options, &child_error);
        if (message.operation_status != YVEX_OK) {
            (void)snprintf(message.reason, sizeof(message.reason), "%.190s",
                           yvex_error_message(&child_error));
        } else {
            unsigned long long count = yvex_remote_catalog_count(catalog), index;
            if (count > YVEX_TUI_REMOTE_CAP) count = YVEX_TUI_REMOTE_CAP;
            message.result_count = (unsigned int)count;
            for (index = 0u; index < count; ++index)
                message.results[index] = *yvex_remote_catalog_at(catalog, index);
        }
        yvex_remote_catalog_close(catalog);
        bytes = (const unsigned char *)&message;
        while (written < sizeof(message)) {
            ssize_t count = write(result_pipe[1], bytes + written,
                                  sizeof(message) - written);
            if (count > 0) written += (size_t)count;
            else if (count < 0 && errno == EINTR) continue;
            else break;
        }
        (void)close(cancel_pipe[0]);
        (void)close(result_pipe[1]);
        _exit(written == sizeof(message) ? 0 : 1);
    }
    (void)close(result_pipe[1]);
    (void)close(cancel_pipe[0]);
    if (pid < 0 || !descriptor_nonblocking(result_pipe[0])) {
        if (pid > 0) {
            (void)kill(pid, SIGTERM);
            (void)waitpid(pid, NULL, 0);
        }
        (void)close(result_pipe[0]);
        (void)close(cancel_pipe[1]);
        yvex_error_set(err, YVEX_ERR_IO, "cli.tui.discover",
                       "remote search worker could not be started");
        return YVEX_ERR_IO;
    }
    worker->pid = pid;
    worker->result_fd = result_pipe[0];
    worker->cancel_fd = cancel_pipe[1];
    worker->running = 1;
    worker->started_ns = monotonic_ns();
    worker->received = 0u;
    memset(&worker->message, 0, sizeof(worker->message));
    yvex_error_clear(err);
    return YVEX_OK;
}

static void remote_worker_finish(tui_remote_worker *worker,
                                 yvex_tui_state *state)
{
    int status;
    if (!worker || !worker->running) return;
    if (monotonic_ns() - worker->started_ns > 10000000000ull) {
        worker_stop(worker->pid, 0);
        if (worker->result_fd >= 0) (void)close(worker->result_fd);
        if (worker->cancel_fd >= 0) (void)close(worker->cancel_fd);
        worker->result_fd = worker->cancel_fd = -1;
        worker->running = 0;
        yvex_tui_remote_search_publish(state, NULL, 0u,
                                       "Hugging Face discovery timed out after 10 seconds");
        return;
    }
    while (worker->result_fd >= 0 && worker->received < sizeof(worker->message)) {
        ssize_t count = read(worker->result_fd,
                             (unsigned char *)&worker->message + worker->received,
                             sizeof(worker->message) - worker->received);
        if (count > 0) worker->received += (size_t)count;
        else if (count == 0) {
            (void)close(worker->result_fd);
            worker->result_fd = -1;
        } else if (errno == EINTR) continue;
        else if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        else {
            (void)close(worker->result_fd);
            worker->result_fd = -1;
        }
    }
    if (worker->received != sizeof(worker->message) && worker->result_fd >= 0) return;
    if (waitpid(worker->pid, &status, WNOHANG) == 0) return;
    if (worker->result_fd >= 0) (void)close(worker->result_fd);
    if (worker->cancel_fd >= 0) (void)close(worker->cancel_fd);
    worker->result_fd = worker->cancel_fd = -1;
    worker->running = 0;
    if (worker->received == sizeof(worker->message) &&
        worker->message.schema_version == TUI_REMOTE_WORKER_SCHEMA_V1)
        yvex_tui_remote_search_publish(state, worker->message.results,
                                       worker->message.result_count,
                                       worker->message.reason);
    else
        yvex_tui_remote_search_publish(
            state, NULL, 0u,
            "remote search worker ended without a complete typed result");
}

static void remote_worker_close(tui_remote_worker *worker)
{
    unsigned int attempt;
    if (!worker) return;
    if (worker->running && worker->cancel_fd >= 0) {
        const char cancel = 'x';
        ssize_t cancel_count = write(worker->cancel_fd, &cancel, 1u);
        (void)cancel_count;
    }
    for (attempt = 0u; worker->running && attempt < 20u; ++attempt) {
        int status;
        if (waitpid(worker->pid, &status, WNOHANG) == worker->pid)
            worker->running = 0;
        else
            (void)poll(NULL, 0u, 10);
    }
    if (worker->running) {
        worker_stop(worker->pid, 0);
    }
    if (worker->result_fd >= 0) (void)close(worker->result_fd);
    if (worker->cancel_fd >= 0) (void)close(worker->cancel_fd);
    remote_worker_init(worker);
}

static void acquisition_worker_init(tui_acquisition_worker *worker)
{
    memset(worker, 0, sizeof(*worker));
    worker->pid = -1;
    worker->diagnostic_fd = -1;
}

static void diagnostic_redact(char *text)
{
    char *cursor;
    if (!text) return;
    for (cursor = text; (cursor = strstr(cursor, "hf_")) != NULL;) {
        char *end = cursor + 3u;
        while (*end && *end != ' ' && *end != '\t' && *end != '\n') end++;
        while (cursor < end) *cursor++ = '*';
    }
    for (cursor = text; (cursor = strstr(cursor, "Bearer ")) != NULL;) {
        char *end = cursor + 7u;
        while (*end && *end != ' ' && *end != '\t' && *end != '\n') end++;
        while (cursor < end) *cursor++ = '*';
    }
}

static int acquisition_worker_start(tui_acquisition_worker *worker,
                                    const char *executable,
                                    const yvex_remote_model *remote,
                                    yvex_error *err)
{
    yvex_tui_acquire_command command;
    int diagnostic_pipe[2] = {-1, -1}, null_fd, rc;
    pid_t pid;
    if (!worker || worker->running) {
        yvex_error_set(err, YVEX_ERR_STATE, "cli.tui.acquire",
                       "one model acquisition may be active");
        return YVEX_ERR_STATE;
    }
    rc = yvex_tui_acquire_prepare(executable, remote, &command, err);
    if (rc != YVEX_OK) return rc;
    if (pipe(diagnostic_pipe) != 0) {
        yvex_error_set(err, YVEX_ERR_IO, "cli.tui.acquire",
                       "acquisition diagnostic pipe could not be created");
        return YVEX_ERR_IO;
    }
    (void)fcntl(diagnostic_pipe[1], F_SETFD, FD_CLOEXEC);
    null_fd = open("/dev/null", O_RDWR);
    if (null_fd < 0) {
        (void)close(diagnostic_pipe[0]);
        (void)close(diagnostic_pipe[1]);
        yvex_error_set(err, YVEX_ERR_IO, "cli.tui.acquire",
                       "acquisition null terminal could not be opened");
        return YVEX_ERR_IO;
    }
    pid = fork();
    if (pid == 0) {
        (void)close(diagnostic_pipe[0]);
        if (setsid() < 0 || dup2(null_fd, STDIN_FILENO) < 0 ||
            dup2(null_fd, STDOUT_FILENO) < 0 ||
            dup2(diagnostic_pipe[1], STDERR_FILENO) < 0)
            _exit(127);
        if (null_fd > STDERR_FILENO) (void)close(null_fd);
        if (diagnostic_pipe[1] > STDERR_FILENO) (void)close(diagnostic_pipe[1]);
        execv(command.argv[0], command.argv);
        _exit(127);
    }
    (void)close(null_fd);
    (void)close(diagnostic_pipe[1]);
    if (pid < 0 || !descriptor_nonblocking(diagnostic_pipe[0])) {
        if (pid > 0) {
            (void)kill(pid, SIGTERM);
            (void)waitpid(pid, NULL, 0);
        }
        (void)close(diagnostic_pipe[0]);
        yvex_error_set(err, YVEX_ERR_IO, "cli.tui.acquire",
                       "acquisition process could not be started");
        return YVEX_ERR_IO;
    }
    worker->pid = pid;
    worker->diagnostic_fd = diagnostic_pipe[0];
    worker->running = 1;
    worker->exit_known = 0;
    worker->exit_status = 0;
    worker->truncated = 0;
    worker->diagnostic_count = 0u;
    worker->diagnostic[0] = '\0';
    yvex_error_clear(err);
    return YVEX_OK;
}

static int acquisition_worker_poll(tui_acquisition_worker *worker,
                                   yvex_tui_state *state)
{
    char buffer[512];
    if (!worker || !worker->running) return 0;
    while (worker->diagnostic_fd >= 0) {
        ssize_t count = read(worker->diagnostic_fd, buffer, sizeof(buffer));
        if (count > 0) {
            size_t take = (size_t)count;
            if (worker->diagnostic_count + take >= sizeof(worker->diagnostic)) {
                take = worker->diagnostic_count + 1u < sizeof(worker->diagnostic)
                           ? sizeof(worker->diagnostic) - worker->diagnostic_count - 1u : 0u;
                worker->truncated = 1;
            }
            if (take) memcpy(worker->diagnostic + worker->diagnostic_count, buffer, take);
            worker->diagnostic_count += take;
            worker->diagnostic[worker->diagnostic_count] = '\0';
        } else if (count == 0) {
            (void)close(worker->diagnostic_fd);
            worker->diagnostic_fd = -1;
        } else if (errno == EINTR) continue;
        else if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        else {
            (void)close(worker->diagnostic_fd);
            worker->diagnostic_fd = -1;
        }
    }
    if (!worker->exit_known) {
        int status;
        if (waitpid(worker->pid, &status, WNOHANG) == worker->pid) {
            worker->exit_known = 1;
            worker->exit_status = WIFEXITED(status) ? WEXITSTATUS(status) :
                                  WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1;
        }
    }
    if (!worker->exit_known || worker->diagnostic_fd >= 0) return 0;
    worker->running = 0;
    state->acquisition_running = 0;
    state->acquisition_exit_known = 1;
    state->acquisition_exit_status = worker->exit_status;
    if (worker->truncated)
        (void)snprintf(state->acquisition_diagnostic,
                       sizeof(state->acquisition_diagnostic), "%.480s [truncated]",
                       worker->diagnostic);
    else
        (void)snprintf(state->acquisition_diagnostic,
                       sizeof(state->acquisition_diagnostic), "%s",
                       worker->diagnostic);
    diagnostic_redact(state->acquisition_diagnostic);
    state->redraw = 1;
    return 1;
}

static void acquisition_worker_close(tui_acquisition_worker *worker)
{
    if (!worker) return;
    if (worker->running) {
        worker_stop(worker->pid, 1);
    }
    if (worker->diagnostic_fd >= 0) (void)close(worker->diagnostic_fd);
    acquisition_worker_init(worker);
}

static int slash_alias_matches(const char *aliases, const char *line,
                               size_t extent)
{
    const char *cursor = aliases;
    if (!aliases || !strcmp(aliases, "none")) return 0;
    while (*cursor) {
        const char *end = strchr(cursor, ',');
        size_t count = end ? (size_t)(end - cursor) : strlen(cursor);
        if (count == extent && !memcmp(cursor, line, extent)) return 1;
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
        if ((!strcmp(descriptor->slash_projection, "none") ? 0
              : strlen(descriptor->slash_projection) == extent &&
                    !memcmp(descriptor->slash_projection, line, extent)) ||
            slash_alias_matches(descriptor->slash_aliases, line, extent))
            return descriptor;
    }
    return NULL;
}

static void local_notice(yvex_tui_state *state, yvex_tui_severity severity,
                         const char *text)
{
    yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_SYSTEM, severity,
                          severity == YVEX_TUI_SEVERITY_ERROR
                              ? YVEX_CLIENT_STREAM_ERROR
                              : YVEX_CLIENT_STREAM_CONTROL_EVENT,
                          text);
}

static int runtime_operation(yvex_tui_state *state,
                             yvex_cli_interactive *interactive,
                             const yvex_operator_descriptor *descriptor,
                             const char *argument)
{
    char generated[YVEX_SERVER_SESSION_NAME_CAP];
    const char *session = argument && argument[0] ? argument : state->active_session;
    switch (descriptor->runtime_adapter) {
    case YVEX_OPERATOR_RUNTIME_HELP:
        state->overlay = YVEX_TUI_OVERLAY_HELP;
        return 1;
    case YVEX_OPERATOR_RUNTIME_CONSOLE_STATUS:
        state->surface = YVEX_TUI_SURFACE_HOME;
        return yvex_cli_interactive_refresh(interactive, &state->active_engine,
                                            state->active_session);
    case YVEX_OPERATOR_RUNTIME_SERVER_STATUS:
    case YVEX_OPERATOR_RUNTIME_SERVER_MEMORY:
        state->surface = YVEX_TUI_SURFACE_RUNTIME;
        return yvex_cli_interactive_refresh(interactive, &state->active_engine,
                                            state->active_session);
    case YVEX_OPERATOR_RUNTIME_SERVER_STOP:
        if (!yvex_cli_interactive_request(interactive,
                                          YVEX_CLIENT_OP_RUNTIME_STOP, NULL, NULL))
            return 0;
        yvex_tui_runtime_stop_requested(state, 0);
        return 1;
    case YVEX_OPERATOR_RUNTIME_SERVER_LOAD:
        if (!argument || !argument[0]) {
            const yvex_model_runtime_profile_fact *profile =
                yvex_tui_launch_profile(state);
            argument = profile ? profile->alias : NULL;
        }
        if (!argument || !argument[0]) {
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Select a launchable model profile before loading");
            return 1;
        }
        if (!yvex_cli_interactive_request(interactive, YVEX_CLIENT_OP_ENGINE_LOAD,
                                          NULL, argument))
            return 0;
        state->engine_load_requested = 1;
        state->runtime_lifecycle = YVEX_TUI_RUNTIME_ENGINE_LOADING;
        return 1;
    case YVEX_OPERATOR_RUNTIME_SERVER_UNLOAD:
        if (!state->active_engine.alias[0] || !state->active_engine.generation) {
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "No loaded model is active in this client");
            return 1;
        }
        if (argument && argument[0] &&
            strcmp(argument, state->active_engine.alias)) {
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Select the requested loaded model before unloading it");
            return 1;
        }
        return yvex_cli_interactive_request(interactive,
                                            YVEX_CLIENT_OP_ENGINE_UNLOAD,
                                            &state->active_engine, NULL);
    case YVEX_OPERATOR_RUNTIME_SESSION_LIST:
        state->surface = YVEX_TUI_SURFACE_SESSIONS;
        return yvex_cli_interactive_request(interactive,
                                            YVEX_CLIENT_OP_SESSION_LIST,
                                            &state->active_engine, NULL);
    case YVEX_OPERATOR_RUNTIME_SESSION_SHOW:
        state->surface = YVEX_TUI_SURFACE_SESSIONS;
        return yvex_cli_interactive_request(interactive,
                                            YVEX_CLIENT_OP_SESSION_SHOW,
                                            &state->active_engine, session);
    case YVEX_OPERATOR_RUNTIME_SESSION_NEW:
        if (!argument || !argument[0]) {
            (void)snprintf(generated, sizeof(generated), "tui-%lu",
                           (unsigned long)getpid());
            session = generated;
        }
        return yvex_cli_interactive_request(interactive,
                                            YVEX_CLIENT_OP_SESSION_NEW,
                                            &state->active_engine, session);
    case YVEX_OPERATOR_RUNTIME_SESSION_ATTACH:
        return yvex_cli_interactive_request(interactive,
                                            YVEX_CLIENT_OP_SESSION_ATTACH,
                                            &state->active_engine, session);
    case YVEX_OPERATOR_RUNTIME_SESSION_DETACH:
        return yvex_cli_interactive_request(interactive,
                                            YVEX_CLIENT_OP_SESSION_DETACH,
                                            &state->active_engine, session);
    case YVEX_OPERATOR_RUNTIME_SESSION_RESET:
        return yvex_cli_interactive_request(interactive,
                                            YVEX_CLIENT_OP_SESSION_RESET,
                                            &state->active_engine, session);
    case YVEX_OPERATOR_RUNTIME_SESSION_CLOSE:
        return yvex_cli_interactive_request(interactive,
                                            YVEX_CLIENT_OP_SESSION_CLOSE,
                                            &state->active_engine, session);
    case YVEX_OPERATOR_RUNTIME_SESSION_CANCEL:
        return yvex_cli_interactive_cancel(&state->active_engine, session);
    case YVEX_OPERATOR_RUNTIME_REASONING_DISABLED:
        state->reasoning_policy = YVEX_REASONING_DISABLED;
        local_notice(state, YVEX_TUI_SEVERITY_INFO,
                     "Explicit reasoning disabled for the next turn");
        return 1;
    case YVEX_OPERATOR_RUNTIME_REASONING_ENABLED:
        if (state->console_available &&
            !state->console.explicit_reasoning_channel_supported) {
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Active model exposes no explicit reasoning channel");
            return 1;
        }
        state->reasoning_policy = YVEX_REASONING_ENABLED;
        local_notice(state, YVEX_TUI_SEVERITY_INFO,
                     "Explicit reasoning enabled for the next turn");
        return 1;
    case YVEX_OPERATOR_RUNTIME_REASONING_MAXIMUM:
        if (state->console_available &&
            !state->console.explicit_reasoning_channel_supported) {
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Active model exposes no explicit reasoning channel");
            return 1;
        }
        state->reasoning_policy = YVEX_REASONING_MAXIMUM;
        local_notice(state, YVEX_TUI_SEVERITY_INFO,
                     "Maximum source-authored reasoning selected for the next turn");
        return 1;
    default:
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "This registry operation is not projected in Wave 1");
        return 1;
    }
}

static void palette_submit(yvex_tui_state *state,
                           yvex_cli_interactive *interactive)
{
    const yvex_operator_descriptor *descriptor;
    yvex_cli_operator_invocation invocation;
    const char *argument = NULL, *parsed_argument = NULL;
    int status, accepted = 1;
    descriptor = slash_descriptor((const char *)state->command.bytes, &argument);
    if (!descriptor) {
        local_notice(state, YVEX_TUI_SEVERITY_ERROR,
                     "Unknown command; palette entries come from the operator registry");
        return;
    }
    status = yvex_cli_operator_slash_parse(descriptor, argument, &invocation);
    if (status) {
        local_notice(state, YVEX_TUI_SEVERITY_ERROR, invocation.message);
        yvex_cli_operator_invocation_close(&invocation);
        return;
    }
    if (invocation.argument_count) parsed_argument = invocation.arguments[0];
    if (descriptor->lane == YVEX_OPERATOR_LANE_REPL_LOCAL) {
        if (descriptor->repl_adapter == YVEX_OPERATOR_REPL_QUIT)
            state->shutdown_requested = 1;
    } else if (!strcmp(descriptor->daemon_requirement, "required") &&
               state->connection != YVEX_TUI_CONNECTION_CONNECTED) {
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "Operation unavailable while the resident runtime is disconnected");
    } else if (descriptor->lane == YVEX_OPERATOR_LANE_RUNTIME_CLIENT) {
        accepted = runtime_operation(state, interactive, descriptor, parsed_argument);
        if (!accepted)
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Interactive request queue is busy");
    } else {
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "Operation is discoverable but not executable inside this client context");
    }
    yvex_cli_operator_invocation_close(&invocation);
    state->overlay = YVEX_TUI_OVERLAY_NONE;
    state->focus = YVEX_TUI_FOCUS_COMPOSER;
    yvex_tui_composer_clear(&state->command);
    state->redraw = 1;
}

static void generation_submit(yvex_tui_state *state,
                              yvex_cli_interactive *interactive)
{
    yvex_provider_request defaults;
    yvex_cli_interactive_turn options;
    if (!state->composer.count) return;
    if (state->connection != YVEX_TUI_CONNECTION_CONNECTED) {
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "Runtime unavailable; composer draft preserved");
        return;
    }
    if (state->generation_active) {
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "A generation is already active; composer draft preserved");
        return;
    }
    yvex_provider_request_default(&defaults);
    memset(&options, 0, sizeof(options));
    options.maximum_new_tokens = state->maximum_new_tokens;
    options.stochastic = defaults.sampling.stochastic;
    options.seed_present = defaults.sampling.seed_present;
    options.seed = defaults.sampling.seed;
    options.temperature = defaults.sampling.temperature;
    options.top_k = defaults.sampling.top_k;
    options.top_p = defaults.sampling.top_p;
    options.min_p = defaults.sampling.min_p;
    options.typical_p = defaults.sampling.typical_p;
    options.reasoning_policy = state->reasoning_policy;
    if (!state->active_engine.alias[0] || !state->active_engine.generation) {
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "Load a model engine before sending a message");
        return;
    }
    if (!yvex_cli_interactive_generate(interactive, &state->active_engine,
                                       state->active_session,
                                       state->composer.bytes,
                                       state->composer.count, &options)) {
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "Interactive request queue is busy; composer draft preserved");
        return;
    }
    yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_USER,
                          YVEX_TUI_SEVERITY_INFO,
                          YVEX_CLIENT_STREAM_UNSPECIFIED,
                          (const char *)state->composer.bytes);
    yvex_tui_composer_history_push(&state->composer);
    yvex_tui_composer_clear(&state->composer);
    state->generation_active = 1;
    state->generation_phase = YVEX_CLIENT_PHASE_IDLE;
}

static void error_notice(yvex_tui_state *state, const char *operation,
                         const yvex_error *err)
{
    char message[YVEX_SERVER_REASON_CAP];
    (void)snprintf(message, sizeof(message), "%s: %.200s", operation,
                   yvex_error_message(err));
    local_notice(state, YVEX_TUI_SEVERITY_ERROR, message);
}

static int launch_configure(yvex_tui_state *state)
{
    const yvex_model_runtime_profile_fact *profile;
    if (!state->model_count || state->launch_selected_model >= state->model_count)
        return 0;
    profile = yvex_tui_launch_profile(state);
    if (!profile || !profile->launchable) {
        yvex_tui_runtime_launch_failed(
            state, YVEX_TUI_LAUNCH_FAILURE_PREFLIGHT,
            profile && profile->blocker[0]
                ? profile->blocker : "selected runtime profile is not launchable",
            0, 0, NULL);
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     profile && profile->blocker[0]
                         ? profile->blocker : "Selected profile cannot start a runtime");
        return 0;
    }
    (void)snprintf(state->launch_request.profile,
                   sizeof(state->launch_request.profile), "%s", profile->alias);
    return 1;
}

static int runtime_launch(yvex_tui_state *state, yvex_tui_launcher *launcher)
{
    yvex_error err;
    unsigned long long started = monotonic_ns();
    if (!launch_configure(state)) return 0;
    yvex_error_clear(&err);
    if (yvex_tui_launcher_start(launcher, started, &err) !=
        YVEX_OK) {
        yvex_tui_runtime_launch_failed(
            state, YVEX_TUI_LAUNCH_FAILURE_SPAWN,
            yvex_error_message(&err), 0, 0, NULL);
        error_notice(state, "Runtime launch failed", &err);
        return 0;
    }
    yvex_tui_runtime_launch_started(state, launcher->pid, started);
    state->engine_load_requested = 0;
    state->restart_pending = 0;
    local_notice(state, YVEX_TUI_SEVERITY_INFO,
                 "Canonical server process started; waiting for protocol");
    return 1;
}

static void launch_submit(yvex_tui_state *state,
                          yvex_cli_interactive *interactive,
                          yvex_tui_launcher *launcher)
{
    if (state->restart_pending &&
        state->connection == YVEX_TUI_CONNECTION_CONNECTED) {
        if (!launch_configure(state)) return;
        state->overlay = YVEX_TUI_OVERLAY_NONE;
        state->focus = YVEX_TUI_FOCUS_CONTENT;
        if (yvex_cli_interactive_request(interactive,
                                         YVEX_CLIENT_OP_RUNTIME_STOP, NULL, NULL)) {
            yvex_tui_runtime_stop_requested(state, 1);
            local_notice(state, YVEX_TUI_SEVERITY_INFO,
                         "Restart requested; waiting for canonical shutdown");
        } else {
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Interactive request queue is busy");
        }
        return;
    }
    if (!launch_configure(state)) return;
    state->overlay = YVEX_TUI_OVERLAY_NONE;
    state->focus = YVEX_TUI_FOCUS_CONTENT;
    if (state->connection == YVEX_TUI_CONNECTION_CONNECTED) {
        if (yvex_cli_interactive_request(interactive, YVEX_CLIENT_OP_ENGINE_LOAD,
                                         NULL, state->launch_request.profile)) {
            state->engine_load_requested = 1;
            state->runtime_lifecycle = YVEX_TUI_RUNTIME_ENGINE_LOADING;
            local_notice(state, YVEX_TUI_SEVERITY_INFO,
                         "Model engine load requested through the resident host");
        } else {
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Interactive request queue is busy");
        }
    } else if (runtime_launch(state, launcher)) {
        (void)yvex_cli_interactive_refresh(interactive, &state->active_engine,
                                           state->active_session);
    }
}

static void runtime_action_submit(yvex_tui_state *state,
                                  yvex_cli_interactive *interactive)
{
    if (state->connection != YVEX_TUI_CONNECTION_CONNECTED) return;
    if (state->runtime_action == 1u) {
        if (!yvex_tui_startup_model_count(state)) {
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Restart requires a startup-ready registry model");
            return;
        }
        yvex_tui_runtime_launch_open(state, state->selected_model, 1);
        return;
    }
    if (yvex_cli_interactive_request(interactive, YVEX_CLIENT_OP_RUNTIME_STOP,
                                     NULL, NULL)) {
        yvex_tui_runtime_stop_requested(state, 0);
        local_notice(state, YVEX_TUI_SEVERITY_INFO,
                     "Canonical runtime shutdown requested");
    } else {
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "Interactive request queue is busy");
    }
}

static void submit_action(yvex_tui_state *state,
                          yvex_cli_interactive *interactive,
                          yvex_tui_launcher *launcher)
{
    if (state->overlay == YVEX_TUI_OVERLAY_PALETTE) {
        palette_submit(state, interactive);
    } else if (state->overlay == YVEX_TUI_OVERLAY_RUNTIME_LAUNCH) {
        launch_submit(state, interactive, launcher);
    } else if (state->surface == YVEX_TUI_SURFACE_RUNTIME &&
               state->focus == YVEX_TUI_FOCUS_CONTENT &&
               state->connection == YVEX_TUI_CONNECTION_CONNECTED) {
        runtime_action_submit(state, interactive);
    } else if (state->focus == YVEX_TUI_FOCUS_CONTENT &&
               state->surface == YVEX_TUI_SURFACE_SESSIONS &&
               !state->composer.count && state->session_count) {
        const char *session = state->sessions[state->selected_session].name;
        if (!yvex_cli_interactive_request(interactive,
                                          YVEX_CLIENT_OP_SESSION_ATTACH,
                                          &state->active_engine, session))
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Interactive request queue is busy");
    } else {
        generation_submit(state, interactive);
    }
}

static void remote_search_submit(yvex_tui_state *state,
                                 tui_remote_worker *worker)
{
    yvex_error err;
    if (!state->discover_query_count) {
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "Enter a Hugging Face model query first");
        return;
    }
    yvex_error_clear(&err);
    if (remote_worker_start(worker, state->discover_query, &err) != YVEX_OK) {
        error_notice(state, "Hugging Face discovery failed", &err);
        return;
    }
    yvex_tui_remote_search_started(state);
    local_notice(state, YVEX_TUI_SEVERITY_INFO,
                 "Hugging Face discovery requested through the provider boundary");
}

static void acquisition_submit(yvex_tui_state *state,
                               const yvex_tui_launcher *launcher,
                               tui_acquisition_worker *worker)
{
    const yvex_remote_model *remote;
    yvex_error err;
    if (state->selected_remote >= state->remote_count) return;
    remote = &state->remote_models[state->selected_remote];
    if (remote->support_stage < YVEX_MODEL_SUPPORT_SOURCE_INGEST ||
        !remote->resolved_revision[0]) {
        local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                     "Remote model needs an admitted source path and immutable revision");
        return;
    }
    yvex_error_clear(&err);
    if (acquisition_worker_start(worker, launcher->executable,
                                 remote, &err) != YVEX_OK) {
        error_notice(state, "Model acquisition failed to start", &err);
        return;
    }
    state->acquisition_running = 1;
    state->acquisition_exit_known = 0;
    state->acquisition_exit_status = 0;
    state->acquisition_diagnostic[0] = '\0';
    state->redraw = 1;
    local_notice(state, YVEX_TUI_SEVERITY_INFO,
                 "Canonical model acquisition started in a separate process");
}

static void acquisition_events(tui_acquisition_worker *worker,
                               yvex_tui_state *state)
{
    yvex_error err;
    if (!acquisition_worker_poll(worker, state)) return;
    if (state->acquisition_exit_status == 0) {
        yvex_error_clear(&err);
        if (yvex_tui_models_load(state, NULL, &err) == YVEX_OK)
            local_notice(state, YVEX_TUI_SEVERITY_SUCCESS,
                         "Model source acquisition completed; Library refreshed");
        else
            error_notice(state, "Library refresh after acquisition failed", &err);
    } else {
        local_notice(state, YVEX_TUI_SEVERITY_ERROR,
                     "Canonical model acquisition failed; inspect the bounded diagnostic");
    }
}

static void input_read(yvex_tui_terminal *terminal, yvex_tui_input *input,
                       yvex_tui_state *state,
                       yvex_cli_interactive *interactive,
                       yvex_tui_launcher *launcher,
                       tui_remote_worker *remote_worker,
                       tui_acquisition_worker *acquisition_worker)
{
    unsigned char bytes[256];
    ssize_t count = read(terminal->input_fd, bytes, sizeof(bytes));
    ssize_t index;
    if (count == 0) {
        state->shutdown_requested = 1;
        return;
    }
    if (count < 0) {
        if (errno != EINTR && errno != EAGAIN)
            state->shutdown_requested = 1;
        return;
    }
    for (index = 0; index < count && !state->shutdown_requested; ++index) {
        yvex_tui_input_action action = yvex_tui_input_byte(input, state, bytes[index]);
        if (action == YVEX_TUI_INPUT_SUBMIT)
            submit_action(state, interactive, launcher);
        else if (action == YVEX_TUI_INPUT_REMOTE_SEARCH)
            remote_search_submit(state, remote_worker);
        else if (action == YVEX_TUI_INPUT_ACQUIRE)
            acquisition_submit(state, launcher, acquisition_worker);
        else if (action == YVEX_TUI_INPUT_EXIT)
            state->shutdown_requested = 1;
        else if (action == YVEX_TUI_INPUT_REFRESH) {
            if (state->surface == YVEX_TUI_SURFACE_MODELS ||
                (!state->runtime_available &&
                 state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR)) {
                yvex_error err;
                if (yvex_tui_models_load(state, NULL, &err) != YVEX_OK)
                    error_notice(state, "Model registry refresh failed", &err);
            }
            (void)yvex_cli_interactive_refresh(interactive,
                                               &state->active_engine,
                                               state->active_session);
        }
    }
}

static void signal_events(yvex_tui_terminal *terminal, yvex_tui_state *state)
{
    int resize = 0, interrupt = 0, terminate = 0;
    yvex_tui_terminal_take_signals(terminal, &resize, &interrupt, &terminate);
    if (resize) {
        unsigned int rows, columns;
        if (yvex_tui_terminal_dimensions(terminal, &rows, &columns))
            yvex_tui_state_resize(state, rows, columns);
    }
    if (terminate) state->shutdown_requested = 1;
    if (!interrupt) return;
    if (state->generation_active) {
        if (yvex_cli_interactive_cancel(&state->active_engine,
                                        state->active_session))
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "Cancellation requested");
        else
            local_notice(state, YVEX_TUI_SEVERITY_WARNING,
                         "No active server request accepted cancellation");
    } else if (state->composer.count || state->overlay != YVEX_TUI_OVERLAY_NONE) {
        yvex_tui_composer_clear(&state->composer);
        state->overlay = YVEX_TUI_OVERLAY_NONE;
        state->focus = YVEX_TUI_FOCUS_COMPOSER;
        local_notice(state, YVEX_TUI_SEVERITY_INFO, "Composer cleared");
    } else {
        state->shutdown_requested = 1;
    }
}

static void protocol_events(yvex_cli_interactive *interactive,
                            yvex_tui_state *state,
                            yvex_tui_launcher *launcher)
{
    yvex_cli_interactive_event event;
    int had_engine = state->active_engine.alias[0] != '\0';
    int engine_snapshot_complete = 0;
    (void)launcher;
    while (yvex_cli_interactive_event_take(interactive, &event)) {
        if (event.kind == YVEX_CLI_INTERACTIVE_MESSAGE &&
            event.operation == YVEX_CLIENT_OP_ENGINE_LIST &&
            event.message.kind == YVEX_CLIENT_MESSAGE_ACK)
            engine_snapshot_complete = 1;
        yvex_tui_state_message(state, &event);
    }
    if (engine_snapshot_complete &&
        state->connection == YVEX_TUI_CONNECTION_CONNECTED &&
        state->launch_request.profile[0] && !state->active_engine.alias[0] &&
        !state->engine_load_requested &&
        (state->runtime_lifecycle == YVEX_TUI_RUNTIME_CONNECTED_EXTERNAL ||
         state->runtime_lifecycle == YVEX_TUI_RUNTIME_CONNECTED_OWNED)) {
        if (yvex_cli_interactive_request(interactive, YVEX_CLIENT_OP_ENGINE_LOAD,
                                         NULL, state->launch_request.profile)) {
            state->engine_load_requested = 1;
            state->runtime_lifecycle = YVEX_TUI_RUNTIME_ENGINE_LOADING;
            local_notice(state, YVEX_TUI_SEVERITY_INFO,
                         "Host ready; loading the selected model engine");
        }
    }
    if (!had_engine && state->active_engine.alias[0])
        (void)yvex_cli_interactive_refresh(interactive, &state->active_engine,
                                           state->active_session);
}

static int endpoint_absent(void)
{
    char path[YVEX_SERVER_SOCKET_PATH_CAP];
    yvex_error err;
    return yvex_server_socket_path(path, &err) == YVEX_OK &&
           access(path, F_OK) != 0;
}

static void launcher_events(yvex_tui_state *state,
                            yvex_cli_interactive *interactive,
                            yvex_tui_launcher *launcher)
{
    (void)yvex_tui_launcher_diagnostic_take(launcher);
    if (launcher->exec_read_fd >= 0 && yvex_tui_launcher_exec_take(launcher)) {
        if (launcher->exec_error) {
            yvex_tui_runtime_launch_failed(
                state, YVEX_TUI_LAUNCH_FAILURE_SPAWN,
                "canonical server executable could not be entered",
                launcher->exec_error, 127, launcher->diagnostic);
            local_notice(state, YVEX_TUI_SEVERITY_ERROR,
                         "Canonical server executable could not be entered");
        } else if (state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCHING) {
            state->runtime_lifecycle = YVEX_TUI_RUNTIME_WAITING_PROTOCOL;
            state->redraw = 1;
        }
    }
    if (yvex_tui_launcher_reap(launcher) &&
        state->runtime_lifecycle != YVEX_TUI_RUNTIME_CONNECTED_EXTERNAL &&
        state->runtime_lifecycle != YVEX_TUI_RUNTIME_CONNECTED_OWNED &&
        state->runtime_lifecycle != YVEX_TUI_RUNTIME_SHUTDOWN_REQUESTED &&
        state->runtime_lifecycle != YVEX_TUI_RUNTIME_STOPPED) {
        diagnostic_redact(launcher->diagnostic);
        yvex_tui_runtime_launch_failed(
            state, YVEX_TUI_LAUNCH_FAILURE_BOOTSTRAP,
            "server process exited before protocol readiness",
            launcher->exec_error, launcher->exit_status,
            launcher->diagnostic);
        local_notice(state, YVEX_TUI_SEVERITY_ERROR,
                     "Server process exited before protocol readiness");
    }
    if (state->runtime_lifecycle == YVEX_TUI_RUNTIME_STOPPED &&
        state->restart_pending && !launcher->running && endpoint_absent()) {
        if (runtime_launch(state, launcher))
            (void)yvex_cli_interactive_refresh(interactive,
                                               &state->active_engine,
                                               state->active_session);
    }
}

static int render_state(yvex_tui_terminal *terminal, yvex_tui_state *state,
                        char *frame)
{
    size_t count = 0u;
    unsigned int cursor_row = 1u, cursor_column = 1u;
    int rc = yvex_tui_render(state, frame, TUI_FRAME_CAP, &count,
                             &cursor_row, &cursor_column);
    if (rc != YVEX_OK || !yvex_tui_terminal_write(terminal, frame, count))
        return 0;
    state->redraw = 0;
    return 1;
}

static int event_loop(yvex_tui_terminal *terminal, yvex_tui_state *state,
                      yvex_cli_interactive *interactive,
                      yvex_tui_launcher *launcher,
                      tui_remote_worker *remote_worker,
                      tui_acquisition_worker *acquisition_worker, char *frame)
{
    yvex_tui_input input;
    unsigned long long next_refresh = monotonic_ns() + TUI_REFRESH_NS;
    memset(&input, 0, sizeof(input));
    while (!state->shutdown_requested) {
        struct pollfd descriptors[7];
        unsigned long long now = monotonic_ns();
        int timeout = input.escape && !input.sequence_count ? 50 :
                      now >= next_refresh ? 0 :
                      (int)((next_refresh - now) / 1000000ull);
        int result;
        if (timeout > 1000) timeout = 1000;
        if (state->redraw && !render_state(terminal, state, frame)) return 0;
        descriptors[0].fd = terminal->input_fd;
        descriptors[0].events = POLLIN | POLLHUP;
        descriptors[0].revents = 0;
        descriptors[1].fd = yvex_tui_terminal_signal_fd(terminal);
        descriptors[1].events = POLLIN;
        descriptors[1].revents = 0;
        descriptors[2].fd = yvex_cli_interactive_event_fd(interactive);
        descriptors[2].events = POLLIN;
        descriptors[2].revents = 0;
        descriptors[3].fd = yvex_tui_launcher_exec_fd(launcher);
        descriptors[3].events = POLLIN | POLLHUP;
        descriptors[3].revents = 0;
        descriptors[4].fd = remote_worker->result_fd;
        descriptors[4].events = POLLIN | POLLHUP;
        descriptors[4].revents = 0;
        descriptors[5].fd = acquisition_worker->diagnostic_fd;
        descriptors[5].events = POLLIN | POLLHUP;
        descriptors[5].revents = 0;
        descriptors[6].fd = yvex_tui_launcher_diagnostic_fd(launcher);
        descriptors[6].events = POLLIN | POLLHUP;
        descriptors[6].revents = 0;
        result = poll(descriptors, 7u, timeout);
        if (result < 0 && errno != EINTR) return 0;
        if (descriptors[1].revents & POLLIN)
            signal_events(terminal, state);
        if (descriptors[2].revents & POLLIN)
            protocol_events(interactive, state, launcher);
        if (descriptors[3].revents & (POLLIN | POLLHUP))
            launcher_events(state, interactive, launcher);
        if (descriptors[4].revents & (POLLIN | POLLHUP))
            remote_worker_finish(remote_worker, state);
        if (descriptors[5].revents & (POLLIN | POLLHUP))
            acquisition_events(acquisition_worker, state);
        if (descriptors[6].revents & (POLLIN | POLLHUP)) {
            (void)yvex_tui_launcher_diagnostic_take(launcher);
            state->redraw = 1;
        }
        if (descriptors[0].revents & (POLLIN | POLLHUP))
            input_read(terminal, &input, state, interactive, launcher,
                       remote_worker, acquisition_worker);
        launcher_events(state, interactive, launcher);
        remote_worker_finish(remote_worker, state);
        acquisition_events(acquisition_worker, state);
        now = monotonic_ns();
        if (input.escape && !input.sequence_count && timeout <= 50)
            (void)yvex_tui_input_flush(&input, state);
        if (now >= next_refresh) {
            (void)yvex_cli_interactive_refresh(interactive,
                                               &state->active_engine,
                                               state->active_session);
            next_refresh = now +
                (state->runtime_lifecycle >= YVEX_TUI_RUNTIME_LAUNCH_REQUESTED &&
                 state->runtime_lifecycle <= YVEX_TUI_RUNTIME_WAITING_PROTOCOL
                     ? TUI_LAUNCH_REFRESH_NS : TUI_REFRESH_NS);
        }
    }
    return 1;
}

static void requested_model_select(yvex_tui_state *state, const char *alias)
{
    size_t model_index;
    int found = 0;
    if (!alias || !alias[0]) return;
    for (model_index = 0u; model_index < state->model_count; ++model_index) {
        unsigned long long profile_index;
        unsigned long long profile_count = yvex_model_library_profile_count(
            state->model_library, model_index);
        for (profile_index = 0u; profile_index < profile_count; ++profile_index) {
            const yvex_model_runtime_profile_fact *profile =
                yvex_model_library_profile_at(state->model_library, model_index,
                                              profile_index);
            if (!profile || strcmp(profile->alias, alias)) continue;
            state->selected_model = model_index;
            state->launch_selected_model = model_index;
            state->launch_selected_profile = (size_t)profile_index;
            found = 1;
            break;
        }
        if (found) break;
    }
    (void)snprintf(state->launch_request.profile,
                   sizeof(state->launch_request.profile), "%s", alias);
}

int yvex_tui_run(const char *executable, const char *model, const char *session,
                 unsigned long long maximum_new_tokens)
{
    yvex_tui_terminal terminal;
    yvex_tui_state *state = NULL;
    yvex_cli_interactive *interactive = NULL;
    yvex_tui_launcher launcher;
    tui_remote_worker remote_worker;
    tui_acquisition_worker acquisition_worker;
    yvex_error err;
    char *frame = NULL;
    unsigned int rows = 24u, columns = 80u;
    int status = 1, terminal_open = 0, launcher_open = 0;
    (void)setlocale(LC_CTYPE, "");
    remote_worker_init(&remote_worker);
    acquisition_worker_init(&acquisition_worker);
    yvex_error_clear(&err);
    if (yvex_tui_terminal_open(&terminal, STDIN_FILENO, STDOUT_FILENO, &err) != YVEX_OK) {
        yvex_cli_out_writef(stderr, "yvex: %s; use `yvex run TEXT` for noninteractive use\n",
                            yvex_error_message(&err));
        return 2;
    }
    terminal_open = 1;
    (void)yvex_tui_terminal_dimensions(&terminal, &rows, &columns);
    state = calloc(1u, sizeof(*state));
    frame = malloc(TUI_FRAME_CAP);
    if (!state || !frame) goto done;
    yvex_tui_state_init(state, rows, columns, session);
    state->maximum_new_tokens = maximum_new_tokens;
    if (yvex_tui_models_load(state, NULL, &err) != YVEX_OK) {
        char notice[YVEX_SERVER_REASON_CAP];
        (void)snprintf(notice, sizeof(notice), "Model registry unavailable: %.200s",
                       yvex_error_message(&err));
        local_notice(state, YVEX_TUI_SEVERITY_WARNING, notice);
    }
    requested_model_select(state, model);
    if (yvex_tui_launcher_open(&launcher, executable, &err) != YVEX_OK) goto done;
    launcher_open = 1;
    if (yvex_cli_interactive_open(&interactive, &err) != YVEX_OK) goto done;
    (void)yvex_cli_interactive_refresh(interactive, &state->active_engine,
                                       state->active_session);
    status = event_loop(&terminal, state, interactive, &launcher,
                        &remote_worker, &acquisition_worker, frame) ? 0 : 1;
done:
    if (interactive && state && state->connection == YVEX_TUI_CONNECTION_CONNECTED)
        (void)yvex_cli_interactive_detach(&state->active_engine,
                                          state->active_session);
    yvex_cli_interactive_close(&interactive);
    remote_worker_close(&remote_worker);
    acquisition_worker_close(&acquisition_worker);
    if (launcher_open) yvex_tui_launcher_close(&launcher);
    free(frame);
    yvex_tui_state_close(state);
    free(state);
    if (terminal_open) yvex_tui_terminal_close(&terminal);
    if (status)
        yvex_cli_out_writef(stderr, "yvex: interactive terminal session failed\n");
    return status;
}

int yvex_tui_dispatch(int argc, char **argv)
{
    const char *model = NULL, *session = "main";
    unsigned long long maximum_new_tokens = 0u;
    int index;
    for (index = 1; index < argc; ++index) {
        char *end = NULL;
        unsigned long long parsed;
        if (!strcmp(argv[index], "chat")) continue;
        if (!strcmp(argv[index], "--model") && index + 1 < argc) {
            model = argv[++index];
        } else if (!strcmp(argv[index], "--session") && index + 1 < argc) {
            session = argv[++index];
        } else if (!strcmp(argv[index], "--max-new-tokens") && index + 1 < argc) {
            errno = 0;
            parsed = strtoull(argv[++index], &end, 10);
            if (errno || !end || *end) return 2;
            maximum_new_tokens = parsed;
        }
    }
    return yvex_tui_run(argv[0], model, session, maximum_new_tokens);
}
