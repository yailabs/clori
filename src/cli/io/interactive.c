/*
 * Own the long-lived interactive client's bounded protocol handoff.
 * The worker emits typed messages only; application state and rendering stay with the TUI.
 */
#define _POSIX_C_SOURCE 200809L

#include "src/cli/io/private.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define INTERACTIVE_COMMAND_CAP 8u
#define INTERACTIVE_EVENT_CAP 128u
#define INTERACTIVE_PROMPT_CAP 65536u

typedef enum {
    INTERACTIVE_REFRESH = 0,
    INTERACTIVE_REQUEST,
    INTERACTIVE_GENERATE
} interactive_command_kind;

typedef struct {
    interactive_command_kind kind;
    yvex_client_operation operation;
    yvex_cli_engine_binding engine;
    char session[YVEX_SERVER_SESSION_NAME_CAP];
    unsigned char prompt[INTERACTIVE_PROMPT_CAP];
    size_t prompt_count;
    yvex_cli_interactive_turn options;
} interactive_command;

struct yvex_cli_interactive {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    interactive_command commands[INTERACTIVE_COMMAND_CAP];
    size_t command_start, command_count;
    yvex_cli_interactive_event events[INTERACTIVE_EVENT_CAP];
    size_t event_start, event_count;
    int wake_read_fd, wake_write_fd;
    int thread_started, stopping, busy;
    yvex_cli_engine_binding active_engine;
    char active_session[YVEX_SERVER_SESSION_NAME_CAP];
};

static void interactive_text(char *output, size_t capacity, const char *input)
{
    size_t count;
    if (!capacity) return;
    if (!input) input = "";
    count = strlen(input);
    if (count >= capacity) count = capacity - 1u;
    memcpy(output, input, count);
    output[count] = '\0';
}

static int wake_pipe_open(yvex_cli_interactive *interactive)
{
    int descriptors[2];
    if (pipe(descriptors) != 0) return 0;
    interactive->wake_read_fd = descriptors[0];
    interactive->wake_write_fd = descriptors[1];
    if (fcntl(descriptors[0], F_SETFL, O_NONBLOCK) != 0 ||
        fcntl(descriptors[1], F_SETFL, O_NONBLOCK) != 0 ||
        fcntl(descriptors[0], F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(descriptors[1], F_SETFD, FD_CLOEXEC) != 0) {
        (void)close(descriptors[0]);
        (void)close(descriptors[1]);
        interactive->wake_read_fd = -1;
        interactive->wake_write_fd = -1;
        return 0;
    }
    return 1;
}

static void event_push(yvex_cli_interactive *interactive,
                       const yvex_cli_interactive_event *event)
{
    unsigned char wake = 1u;
    ssize_t wake_result;
    size_t slot;
    (void)pthread_mutex_lock(&interactive->mutex);
    while (!interactive->stopping &&
           interactive->event_count == INTERACTIVE_EVENT_CAP)
        (void)pthread_cond_wait(&interactive->changed, &interactive->mutex);
    if (interactive->stopping) {
        (void)pthread_mutex_unlock(&interactive->mutex);
        return;
    }
    slot = (interactive->event_start + interactive->event_count) %
           INTERACTIVE_EVENT_CAP;
    interactive->events[slot] = *event;
    interactive->event_count++;
    (void)pthread_mutex_unlock(&interactive->mutex);
    wake_result = write(interactive->wake_write_fd, &wake, sizeof(wake));
    (void)wake_result;
}

static void connection_push(yvex_cli_interactive *interactive,
                            yvex_cli_interactive_connection connection,
                            const yvex_error *err)
{
    yvex_cli_interactive_event event;
    memset(&event, 0, sizeof(event));
    event.kind = YVEX_CLI_INTERACTIVE_CONNECTION;
    event.connection = connection;
    if (err) interactive_text(event.reason, sizeof(event.reason),
                              yvex_error_message(err));
    event_push(interactive, &event);
}

static void message_push(yvex_cli_interactive *interactive,
                         yvex_client_operation operation,
                         const yvex_cli_engine_binding *engine,
                         const char *session,
                         const yvex_client_message *message)
{
    yvex_cli_interactive_event event;
    memset(&event, 0, sizeof(event));
    event.kind = YVEX_CLI_INTERACTIVE_MESSAGE;
    event.operation = operation;
    if (engine) event.engine = *engine;
    event.message = *message;
    interactive_text(event.session, sizeof(event.session), session);
    event_push(interactive, &event);
}

static void transport_failure(yvex_cli_interactive *interactive,
                              const yvex_error *err, int connected)
{
    yvex_status status = err ? yvex_error_code(err) : YVEX_ERR;
    yvex_cli_interactive_connection connection;
    if (status == YVEX_ERR_FORMAT || status == YVEX_ERR_UNSUPPORTED)
        connection = YVEX_CLI_INTERACTIVE_INCOMPATIBLE;
    else
        connection = connected ? YVEX_CLI_INTERACTIVE_DISCONNECTED
                               : YVEX_CLI_INTERACTIVE_UNAVAILABLE;
    connection_push(interactive, connection, err);
}

static int request_open(yvex_client **client, yvex_client_request *request,
                        yvex_error *err)
{
    return yvex_cli_client_request_open(client, request, err);
}

static void request_bind(yvex_client_request *request,
                         const yvex_cli_engine_binding *engine)
{
    if (!engine) return;
    interactive_text(request->model_alias, sizeof(request->model_alias), engine->alias);
    request->engine_generation = engine->generation;
}

static int request_run(yvex_cli_interactive *interactive,
                       yvex_client_request *request,
                       const yvex_cli_engine_binding *engine, const char *session,
                       yvex_client_message *last)
{
    yvex_client *client = NULL;
    yvex_client_message message;
    yvex_error err;
    int rc, done = 0, connected = 0;
    yvex_error_clear(&err);
    rc = request_open(&client, request, &err);
    if (rc == YVEX_OK) {
        connected = 1;
        connection_push(interactive, YVEX_CLI_INTERACTIVE_CONNECTED, NULL);
    }
    while (rc == YVEX_OK && !done) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (last) *last = message;
        message_push(interactive, request->operation, engine, session, &message);
        if (request->operation == YVEX_CLIENT_OP_SESSION_LIST ||
            request->operation == YVEX_CLIENT_OP_ENGINE_LIST)
            done = message.kind == YVEX_CLIENT_MESSAGE_ACK ||
                   message.kind == YVEX_CLIENT_MESSAGE_SESSION_LIST ||
                   message.kind == YVEX_CLIENT_MESSAGE_ERROR;
        else if (request->operation == YVEX_CLIENT_OP_GENERATION_TURN)
            done = message.kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE ||
                   message.kind == YVEX_CLIENT_MESSAGE_ERROR;
        else
            done = 1;
    }
    yvex_client_close(&client);
    if (rc != YVEX_OK) transport_failure(interactive, &err, connected);
    return rc;
}

static int simple_request(yvex_cli_interactive *interactive,
                          yvex_client_operation operation,
                          const yvex_cli_engine_binding *engine,
                          const char *subject, yvex_client_message *last)
{
    yvex_client_request request;
    yvex_cli_client_request_init(&request, operation);
    request_bind(&request, engine);
    if (operation == YVEX_CLIENT_OP_ENGINE_LOAD)
        interactive_text(request.model_alias, sizeof(request.model_alias), subject);
    else
        interactive_text(request.session_name, sizeof(request.session_name), subject);
    return request_run(interactive, &request, engine, subject, last);
}

static void refresh_run(yvex_cli_interactive *interactive,
                        const yvex_cli_engine_binding *engine,
                        const char *session)
{
    yvex_client_message status;
    int rc;
    memset(&status, 0, sizeof(status));
    rc = simple_request(interactive, YVEX_CLIENT_OP_RUNTIME_STATUS,
                        NULL, NULL, NULL);
    if (rc != YVEX_OK) return;
    if (simple_request(interactive, YVEX_CLIENT_OP_ENGINE_LIST,
                       NULL, NULL, NULL) != YVEX_OK || !engine ||
        !engine->alias[0] || !engine->generation)
        return;
    rc = simple_request(interactive, YVEX_CLIENT_OP_CONSOLE_STATUS,
                        engine, session, &status);
    if (rc != YVEX_OK) return;
    if (status.kind == YVEX_CLIENT_MESSAGE_ERROR) {
        if (simple_request(interactive, YVEX_CLIENT_OP_SESSION_NEW,
                           engine, session, NULL) != YVEX_OK ||
            simple_request(interactive, YVEX_CLIENT_OP_SESSION_ATTACH,
                           engine, session, NULL) != YVEX_OK)
            return;
        (void)simple_request(interactive, YVEX_CLIENT_OP_CONSOLE_STATUS,
                             engine, session, &status);
    } else if (status.kind == YVEX_CLIENT_MESSAGE_CONSOLE_STATUS &&
               !status.console.attached) {
        if (simple_request(interactive, YVEX_CLIENT_OP_SESSION_ATTACH,
                           engine, session, NULL) != YVEX_OK)
            return;
        (void)simple_request(interactive, YVEX_CLIENT_OP_CONSOLE_STATUS,
                             engine, session, &status);
    }
    (void)simple_request(interactive, YVEX_CLIENT_OP_SESSION_LIST,
                         engine, NULL, NULL);
}

static void generate_run(yvex_cli_interactive *interactive,
                         const interactive_command *command)
{
    yvex_client_request request;
    yvex_cli_client_request_init(&request, YVEX_CLIENT_OP_GENERATION_TURN);
    request_bind(&request, &command->engine);
    interactive_text(request.session_name, sizeof(request.session_name),
                     command->session);
    request.prompt = command->prompt;
    request.prompt_bytes = command->prompt_count;
    request.maximum_new_tokens = command->options.maximum_new_tokens;
    request.stochastic = command->options.stochastic;
    request.seed_present = command->options.seed_present;
    request.seed = command->options.seed;
    request.temperature = command->options.temperature;
    request.top_k = command->options.top_k;
    request.top_p = command->options.top_p;
    request.min_p = command->options.min_p;
    request.typical_p = command->options.typical_p;
    request.reasoning_policy = command->options.reasoning_policy;
    (void)request_run(interactive, &request, &command->engine,
                      command->session, NULL);
}

static int command_take(yvex_cli_interactive *interactive,
                        interactive_command *command)
{
    (void)pthread_mutex_lock(&interactive->mutex);
    while (!interactive->stopping && !interactive->command_count)
        (void)pthread_cond_wait(&interactive->changed, &interactive->mutex);
    if (interactive->stopping) {
        (void)pthread_mutex_unlock(&interactive->mutex);
        return 0;
    }
    *command = interactive->commands[interactive->command_start];
    interactive->command_start = (interactive->command_start + 1u) %
                                 INTERACTIVE_COMMAND_CAP;
    interactive->command_count--;
    interactive->busy = 1;
    interactive->active_engine = command->engine;
    interactive_text(interactive->active_session,
                     sizeof(interactive->active_session), command->session);
    (void)pthread_cond_broadcast(&interactive->changed);
    (void)pthread_mutex_unlock(&interactive->mutex);
    return 1;
}

static void command_finish(yvex_cli_interactive *interactive)
{
    (void)pthread_mutex_lock(&interactive->mutex);
    interactive->busy = 0;
    memset(&interactive->active_engine, 0, sizeof(interactive->active_engine));
    interactive->active_session[0] = '\0';
    (void)pthread_cond_broadcast(&interactive->changed);
    (void)pthread_mutex_unlock(&interactive->mutex);
}

static void *interactive_main(void *opaque)
{
    yvex_cli_interactive *interactive = opaque;
    interactive_command command;
    while (command_take(interactive, &command)) {
        if (command.kind == INTERACTIVE_REFRESH)
            refresh_run(interactive, &command.engine, command.session);
        else if (command.kind == INTERACTIVE_GENERATE)
            generate_run(interactive, &command);
        else
            (void)simple_request(interactive, command.operation,
                                 &command.engine, command.session, NULL);
        command_finish(interactive);
    }
    return NULL;
}

static int command_push(yvex_cli_interactive *interactive,
                        const interactive_command *command)
{
    size_t slot;
    int accepted = 0;
    if (!interactive || !command) return 0;
    (void)pthread_mutex_lock(&interactive->mutex);
    if (!interactive->stopping &&
        interactive->command_count < INTERACTIVE_COMMAND_CAP) {
        slot = (interactive->command_start + interactive->command_count) %
               INTERACTIVE_COMMAND_CAP;
        interactive->commands[slot] = *command;
        interactive->command_count++;
        accepted = 1;
        (void)pthread_cond_signal(&interactive->changed);
    }
    (void)pthread_mutex_unlock(&interactive->mutex);
    return accepted;
}

int yvex_cli_interactive_open(yvex_cli_interactive **out, yvex_error *err)
{
    yvex_cli_interactive *interactive;
    int mutex_ready = 0, condition_ready = 0;
    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cli.interactive.open",
                       "interactive output is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    interactive = calloc(1u, sizeof(*interactive));
    if (!interactive) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "cli.interactive.open",
                       "interactive client allocation failed");
        return YVEX_ERR_NOMEM;
    }
    interactive->wake_read_fd = -1;
    interactive->wake_write_fd = -1;
    if (pthread_mutex_init(&interactive->mutex, NULL) == 0) mutex_ready = 1;
    if (mutex_ready && pthread_cond_init(&interactive->changed, NULL) == 0)
        condition_ready = 1;
    if (!condition_ready || !wake_pipe_open(interactive) ||
        pthread_create(&interactive->thread, NULL, interactive_main,
                       interactive) != 0) {
        if (interactive->wake_read_fd >= 0) (void)close(interactive->wake_read_fd);
        if (interactive->wake_write_fd >= 0) (void)close(interactive->wake_write_fd);
        if (condition_ready) (void)pthread_cond_destroy(&interactive->changed);
        if (mutex_ready) (void)pthread_mutex_destroy(&interactive->mutex);
        free(interactive);
        yvex_error_set(err, YVEX_ERR_IO, "cli.interactive.open",
                       "interactive worker initialization failed");
        return YVEX_ERR_IO;
    }
    interactive->thread_started = 1;
    *out = interactive;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_cli_interactive_close(yvex_cli_interactive **owner)
{
    yvex_cli_interactive *interactive;
    yvex_cli_engine_binding active_engine;
    char active[YVEX_SERVER_SESSION_NAME_CAP];
    int busy;
    if (!owner || !*owner) return;
    interactive = *owner;
    (void)pthread_mutex_lock(&interactive->mutex);
    interactive->stopping = 1;
    busy = interactive->busy;
    active_engine = interactive->active_engine;
    interactive_text(active, sizeof(active), interactive->active_session);
    (void)pthread_cond_broadcast(&interactive->changed);
    (void)pthread_mutex_unlock(&interactive->mutex);
    if (busy && active[0])
        (void)yvex_cli_interactive_cancel(&active_engine, active);
    if (interactive->thread_started) (void)pthread_join(interactive->thread, NULL);
    if (interactive->wake_read_fd >= 0) (void)close(interactive->wake_read_fd);
    if (interactive->wake_write_fd >= 0) (void)close(interactive->wake_write_fd);
    (void)pthread_cond_destroy(&interactive->changed);
    (void)pthread_mutex_destroy(&interactive->mutex);
    memset(interactive, 0, sizeof(*interactive));
    free(interactive);
    *owner = NULL;
}

int yvex_cli_interactive_event_fd(const yvex_cli_interactive *interactive)
{
    return interactive ? interactive->wake_read_fd : -1;
}

int yvex_cli_interactive_event_take(yvex_cli_interactive *interactive,
                                    yvex_cli_interactive_event *event)
{
    unsigned char wake;
    ssize_t wake_result;
    if (!interactive || !event) return 0;
    (void)pthread_mutex_lock(&interactive->mutex);
    if (!interactive->event_count) {
        (void)pthread_mutex_unlock(&interactive->mutex);
        return 0;
    }
    *event = interactive->events[interactive->event_start];
    interactive->event_start = (interactive->event_start + 1u) %
                               INTERACTIVE_EVENT_CAP;
    interactive->event_count--;
    (void)pthread_cond_broadcast(&interactive->changed);
    (void)pthread_mutex_unlock(&interactive->mutex);
    wake_result = read(interactive->wake_read_fd, &wake, sizeof(wake));
    (void)wake_result;
    return 1;
}

int yvex_cli_interactive_refresh(yvex_cli_interactive *interactive,
                                 const yvex_cli_engine_binding *engine,
                                 const char *session)
{
    interactive_command command;
    memset(&command, 0, sizeof(command));
    command.kind = INTERACTIVE_REFRESH;
    if (engine) command.engine = *engine;
    interactive_text(command.session, sizeof(command.session), session);
    return command_push(interactive, &command);
}

int yvex_cli_interactive_request(yvex_cli_interactive *interactive,
                                 yvex_client_operation operation,
                                 const yvex_cli_engine_binding *engine,
                                 const char *subject)
{
    interactive_command command;
    memset(&command, 0, sizeof(command));
    command.kind = INTERACTIVE_REQUEST;
    command.operation = operation;
    if (engine) command.engine = *engine;
    interactive_text(command.session, sizeof(command.session), subject);
    return command_push(interactive, &command);
}

int yvex_cli_interactive_generate(yvex_cli_interactive *interactive,
                                  const yvex_cli_engine_binding *engine,
                                  const char *session,
                                  const unsigned char *prompt, size_t prompt_count,
                                  const yvex_cli_interactive_turn *options)
{
    interactive_command command;
    if (!engine || !engine->alias[0] || !engine->generation ||
        (!prompt && prompt_count) || !options ||
        prompt_count >= sizeof(command.prompt))
        return 0;
    memset(&command, 0, sizeof(command));
    command.kind = INTERACTIVE_GENERATE;
    command.engine = *engine;
    interactive_text(command.session, sizeof(command.session), session);
    if (prompt_count) memcpy(command.prompt, prompt, prompt_count);
    command.prompt_count = prompt_count;
    command.options = *options;
    return command_push(interactive, &command);
}

static int direct_session_request(yvex_client_operation operation,
                                  const yvex_cli_engine_binding *engine,
                                  const char *session)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    if (!engine || !engine->alias[0] || !engine->generation || !session || !session[0])
        return 0;
    yvex_cli_client_request_init(&request, operation);
    request_bind(&request, engine);
    interactive_text(request.session_name, sizeof(request.session_name), session);
    rc = yvex_cli_client_request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    yvex_client_close(&client);
    return rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ACK;
}

int yvex_cli_interactive_cancel(const yvex_cli_engine_binding *engine,
                                const char *session)
{
    return direct_session_request(YVEX_CLIENT_OP_GENERATION_CANCEL, engine, session);
}

int yvex_cli_interactive_detach(const yvex_cli_engine_binding *engine,
                                const char *session)
{
    return direct_session_request(YVEX_CLIENT_OP_SESSION_DETACH, engine, session);
}
