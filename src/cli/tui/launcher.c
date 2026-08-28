/* Own the bounded local process used to enter the canonical server host operation. */
#define _POSIX_C_SOURCE 200809L

#include "src/cli/input/private.h"
#include "src/cli/tui/private.h"

#include <errno.h>
#include <fcntl.h>
#include <operator/registry.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int launcher_refuse(yvex_error *err, yvex_status status,
                           const char *message)
{
    yvex_error_set(err, status, "cli.tui.launcher", message);
    return status;
}

static int executable_candidate(const char *path,
                                char output[YVEX_TUI_EXECUTABLE_CAP])
{
    char resolved[YVEX_TUI_EXECUTABLE_CAP];
    if (!path || !path[0] || access(path, X_OK) != 0 || !realpath(path, resolved))
        return 0;
    if (strlen(resolved) >= YVEX_TUI_EXECUTABLE_CAP) return 0;
    memcpy(output, resolved, strlen(resolved) + 1u);
    return 1;
}

static int executable_resolve(const char *hint,
                              char output[YVEX_TUI_EXECUTABLE_CAP])
{
    const char *path, *cursor;
    if (!hint || !hint[0]) return 0;
    if (strchr(hint, '/')) return executable_candidate(hint, output);
    path = getenv("PATH");
    if (!path) return 0;
    cursor = path;
    while (1) {
        const char *end = strchr(cursor, ':');
        size_t directory_count = end ? (size_t)(end - cursor) : strlen(cursor);
        char candidate[YVEX_TUI_EXECUTABLE_CAP];
        int count;
        if (directory_count) {
            count = directory_count < sizeof(candidate)
                        ? snprintf(candidate, sizeof(candidate), "%.*s/%s",
                                   (int)directory_count, cursor, hint)
                        : -1;
        } else {
            count = snprintf(candidate, sizeof(candidate), "./%s", hint);
        }
        if (count > 0 && (size_t)count < sizeof(candidate) &&
            executable_candidate(candidate, output))
            return 1;
        if (!end) break;
        cursor = end + 1u;
    }
    return 0;
}

int yvex_tui_launcher_open(yvex_tui_launcher *launcher,
                           const char *executable_hint, yvex_error *err)
{
    if (!launcher)
        return launcher_refuse(err, YVEX_ERR_INVALID_ARG,
                               "launcher state is required");
    memset(launcher, 0, sizeof(*launcher));
    launcher->pid = -1;
    launcher->exec_read_fd = -1;
    launcher->diagnostic_read_fd = -1;
    if (!executable_resolve(executable_hint, launcher->executable))
        return launcher_refuse(err, YVEX_ERR_IO,
                               "current yvex executable could not be resolved");
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_tui_launcher_close(yvex_tui_launcher *launcher)
{
    if (!launcher) return;
    if (launcher->exec_read_fd >= 0) (void)close(launcher->exec_read_fd);
    if (launcher->diagnostic_read_fd >= 0)
        (void)close(launcher->diagnostic_read_fd);
    if (launcher->running) (void)yvex_tui_launcher_reap(launcher);
    launcher->exec_read_fd = -1;
    launcher->diagnostic_read_fd = -1;
}

static const yvex_operator_descriptor *host_descriptor(void)
{
    size_t index;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *descriptor =
            &yvex_operator_descriptors[index];
        if (descriptor->lane == YVEX_OPERATOR_LANE_DAEMON_ENTRYPOINT &&
            descriptor->daemon_adapter == YVEX_OPERATOR_DAEMON_HOST)
            return descriptor;
    }
    return NULL;
}

static const yvex_operator_descriptor *acquire_descriptor(void)
{
    size_t index;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index)
        if (!strcmp(yvex_operator_descriptors[index].operation_id, "model.acquire"))
            return &yvex_operator_descriptors[index];
    return NULL;
}

int yvex_tui_acquire_prepare(const char *executable,
                             const yvex_remote_model *remote,
                             yvex_tui_acquire_command *command,
                             yvex_error *err)
{
    const yvex_operator_descriptor *descriptor = acquire_descriptor();
    yvex_cli_operator_invocation invocation;
    size_t word, output = 0u;
    int status;
    if (!executable || !executable[0] || !remote || !remote->repository[0] ||
        !remote->family[0] || !remote->resolved_revision[0] || !command || !descriptor)
        return launcher_refuse(err, YVEX_ERR_INVALID_ARG,
                               "acquisition requires repository, family, and immutable revision");
    if (remote->support_stage < YVEX_MODEL_SUPPORT_SOURCE_INGEST)
        return launcher_refuse(err, YVEX_ERR_UNSUPPORTED,
                               "remote model has no admitted YVEX source-ingest path");
    memset(command, 0, sizeof(*command));
    if (strlen(executable) >= sizeof(command->executable) ||
        strlen(remote->repository) >= sizeof(command->repository) ||
        strlen(remote->family) >= sizeof(command->family) ||
        strlen(remote->resolved_revision) >= sizeof(command->revision))
        return launcher_refuse(err, YVEX_ERR_BOUNDS,
                               "model acquisition identity exceeds its bound");
    memcpy(command->executable, executable, strlen(executable) + 1u);
    memcpy(command->repository, remote->repository, strlen(remote->repository) + 1u);
    memcpy(command->family, remote->family, strlen(remote->family) + 1u);
    memcpy(command->revision, remote->resolved_revision,
           strlen(remote->resolved_revision) + 1u);
    (void)snprintf(command->name, sizeof(command->name), "%s",
                   strrchr(remote->repository, '/')
                       ? strrchr(remote->repository, '/') + 1u : remote->repository);
    command->argv[output++] = command->executable;
    for (word = 0u; word < descriptor->command_word_count; ++word)
        command->argv[output++] = (char *)descriptor->command_words[word];
    command->argv[output++] = (char *)"--repo";
    command->argv[output++] = command->repository;
    command->argv[output++] = (char *)"--family";
    command->argv[output++] = command->family;
    command->argv[output++] = (char *)"--name";
    command->argv[output++] = command->name;
    command->argv[output++] = (char *)"--revision";
    command->argv[output++] = command->revision;
    command->argv[output++] = (char *)"--auth";
    command->argv[output++] = (char *)"auto";
    command->argv[output++] = (char *)"--progress";
    command->argv[output++] = (char *)"off";
    command->argv[output] = NULL;
    command->argc = (int)output;
    status = yvex_cli_operator_argv_parse(descriptor, command->argc, command->argv,
                                          descriptor->command_word_count, &invocation);
    if (status) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cli.tui.acquire", invocation.message);
        yvex_cli_operator_invocation_close(&invocation);
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_cli_operator_invocation_close(&invocation);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_tui_launch_prepare(const char *executable,
                            yvex_tui_launch_command *command, yvex_error *err)
{
    const yvex_operator_descriptor *descriptor = host_descriptor();
    yvex_cli_operator_invocation invocation;
    size_t word, output = 0u;
    int status;
    if (!executable || !executable[0] || !command || !descriptor)
        return launcher_refuse(err, YVEX_ERR_INVALID_ARG,
                               "canonical server launch inputs are required");
    memset(command, 0, sizeof(*command));
    if (strlen(executable) >= sizeof(command->executable))
        return launcher_refuse(err, YVEX_ERR_BOUNDS,
                               "server executable identity exceeds its bound");
    memcpy(command->executable, executable, strlen(executable) + 1u);
    command->argv[output++] = command->executable;
    for (word = 0u; word < descriptor->command_word_count; ++word) {
        if (output + 1u >= YVEX_TUI_LAUNCH_ARG_CAP)
            return launcher_refuse(err, YVEX_ERR_BOUNDS,
                                   "server command path exceeds launcher capacity");
        command->argv[output++] = (char *)descriptor->command_words[word];
    }
    memcpy(command->console, "off", 4u);
    command->argv[output++] = (char *)"--console";
    command->argv[output++] = command->console;
    if (output >= YVEX_TUI_LAUNCH_ARG_CAP)
        return launcher_refuse(err, YVEX_ERR_BOUNDS,
                               "server launch arguments exceed their bound");
    command->argv[output] = NULL;
    command->argc = (int)output;
    status = yvex_cli_operator_argv_parse(descriptor, command->argc,
                                          command->argv,
                                          descriptor->command_word_count,
                                          &invocation);
    if (status) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cli.tui.launcher",
                       invocation.message);
        yvex_cli_operator_invocation_close(&invocation);
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_cli_operator_invocation_close(&invocation);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int descriptor_flags(int descriptors[2])
{
    int flags;
    if (pipe(descriptors) != 0) return 0;
    flags = fcntl(descriptors[0], F_GETFL);
    if (flags < 0 ||
        fcntl(descriptors[0], F_SETFL, flags | O_NONBLOCK) != 0 ||
        fcntl(descriptors[0], F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(descriptors[1], F_SETFD, FD_CLOEXEC) != 0) {
        (void)close(descriptors[0]);
        (void)close(descriptors[1]);
        return 0;
    }
    return 1;
}

int yvex_tui_launcher_start(yvex_tui_launcher *launcher,
                            unsigned long long started_ns, yvex_error *err)
{
    yvex_tui_launch_command command;
    int descriptors[2], diagnostics[2], null_fd, rc;
    pid_t pid;
    if (!launcher || launcher->running)
        return launcher_refuse(err, YVEX_ERR_STATE,
                               "one local server launch may be active");
    rc = yvex_tui_launch_prepare(launcher->executable, &command, err);
    if (rc != YVEX_OK) return rc;
    if (!descriptor_flags(descriptors))
        return launcher_refuse(err, YVEX_ERR_IO,
                               "launcher status pipe could not be created");
    if (!descriptor_flags(diagnostics)) {
        (void)close(descriptors[0]);
        (void)close(descriptors[1]);
        return launcher_refuse(err, YVEX_ERR_IO,
                               "launcher diagnostic pipe could not be created");
    }
    null_fd = open("/dev/null", O_RDWR);
    if (null_fd < 0) {
        (void)close(descriptors[0]);
        (void)close(descriptors[1]);
        (void)close(diagnostics[0]);
        (void)close(diagnostics[1]);
        return launcher_refuse(err, YVEX_ERR_IO,
                               "launcher null terminal could not be opened");
    }
    pid = fork();
    if (pid == 0) {
        int child_error;
        (void)close(descriptors[0]);
        (void)close(diagnostics[0]);
        (void)signal(SIGPIPE, SIG_IGN);
        if (setsid() < 0 || dup2(null_fd, STDIN_FILENO) < 0 ||
            dup2(null_fd, STDOUT_FILENO) < 0 ||
            dup2(diagnostics[1], STDERR_FILENO) < 0) {
            ssize_t written;
            child_error = errno;
            written = write(descriptors[1], &child_error, sizeof(child_error));
            (void)written;
            _exit(127);
        }
        if (null_fd > STDERR_FILENO) (void)close(null_fd);
        if (diagnostics[1] > STDERR_FILENO) (void)close(diagnostics[1]);
        execv(command.argv[0], command.argv);
        {
            ssize_t written;
            child_error = errno;
            written = write(descriptors[1], &child_error, sizeof(child_error));
            (void)written;
        }
        _exit(127);
    }
    (void)close(null_fd);
    (void)close(descriptors[1]);
    (void)close(diagnostics[1]);
    if (pid < 0) {
        (void)close(descriptors[0]);
        (void)close(diagnostics[0]);
        return launcher_refuse(err, YVEX_ERR_IO,
                               "server process could not be forked");
    }
    launcher->pid = pid;
    launcher->exec_read_fd = descriptors[0];
    launcher->diagnostic_read_fd = diagnostics[0];
    launcher->running = 1;
    launcher->exec_confirmed = 0;
    launcher->exec_error = 0;
    launcher->exit_known = 0;
    launcher->exit_status = 0;
    launcher->diagnostic_count = 0u;
    launcher->diagnostic_truncated = 0;
    launcher->diagnostic[0] = '\0';
    launcher->started_ns = started_ns;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_tui_launcher_exec_fd(const yvex_tui_launcher *launcher)
{
    return launcher ? launcher->exec_read_fd : -1;
}

int yvex_tui_launcher_exec_take(yvex_tui_launcher *launcher)
{
    int child_error = 0;
    ssize_t count;
    if (!launcher || launcher->exec_read_fd < 0) return 0;
    count = read(launcher->exec_read_fd, &child_error, sizeof(child_error));
    if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
        return 0;
    if (count == (ssize_t)sizeof(child_error)) launcher->exec_error = child_error;
    if (count == 0 || count == (ssize_t)sizeof(child_error) || count < 0) {
        (void)close(launcher->exec_read_fd);
        launcher->exec_read_fd = -1;
        launcher->exec_confirmed = child_error == 0;
        return 1;
    }
    return 0;
}

int yvex_tui_launcher_diagnostic_fd(const yvex_tui_launcher *launcher)
{
    return launcher ? launcher->diagnostic_read_fd : -1;
}

int yvex_tui_launcher_diagnostic_take(yvex_tui_launcher *launcher)
{
    char bytes[512];
    int changed = 0;
    if (!launcher || launcher->diagnostic_read_fd < 0) return 0;
    for (;;) {
        ssize_t count = read(launcher->diagnostic_read_fd, bytes, sizeof(bytes));
        if (count > 0) {
            size_t index, take = (size_t)count;
            if (launcher->diagnostic_count + take >= sizeof(launcher->diagnostic)) {
                take = launcher->diagnostic_count + 1u < sizeof(launcher->diagnostic)
                           ? sizeof(launcher->diagnostic) - launcher->diagnostic_count - 1u : 0u;
                launcher->diagnostic_truncated = 1;
            }
            for (index = 0u; index < take; ++index) {
                unsigned char byte = (unsigned char)bytes[index];
                launcher->diagnostic[launcher->diagnostic_count++] =
                    byte == 0x1bu || (byte < 0x20u && byte != '\n' && byte != '\t')
                        ? '?' : (char)byte;
            }
            launcher->diagnostic[launcher->diagnostic_count] = '\0';
            changed = 1;
            continue;
        }
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        (void)close(launcher->diagnostic_read_fd);
        launcher->diagnostic_read_fd = -1;
        break;
    }
    return changed;
}

int yvex_tui_launcher_reap(yvex_tui_launcher *launcher)
{
    int status;
    pid_t result;
    if (!launcher || !launcher->running) return 0;
    result = waitpid(launcher->pid, &status, WNOHANG);
    if (result <= 0) return 0;
    launcher->running = 0;
    launcher->exit_known = 1;
    launcher->exit_status = WIFEXITED(status)
                                ? WEXITSTATUS(status)
                                : WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1;
    if (launcher->exec_read_fd >= 0) (void)yvex_tui_launcher_exec_take(launcher);
    if (launcher->diagnostic_read_fd >= 0)
        (void)yvex_tui_launcher_diagnostic_take(launcher);
    return 1;
}
