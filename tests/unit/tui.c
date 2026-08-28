#define _XOPEN_SOURCE 700

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

#include <yvex/registry.h>

#include "src/cli/tui/private.h"
#include "tests/test.h"

static int write_file(const char *path, const char *text)
{
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    if (fputs(text, file) < 0) {
        (void)fclose(file);
        return 0;
    }
    return fclose(file) == 0;
}

static void registry_entry(yvex_model_registry_entry *entry, const char *alias,
                           const char *artifact, const char *binding,
                           const char *identity)
{
    memset(entry, 0, sizeof(*entry));
    entry->alias = alias;
    entry->family = "deepseek4";
    entry->model = "v4-flash-dspark";
    entry->scope = "complete";
    entry->artifact_class = "transformer";
    entry->qprofile = "Q4_K_M";
    entry->calibration = "noimatrix";
    entry->producer = "yvex";
    entry->schema_version = "v1";
    entry->path = artifact;
    entry->sha256 = identity;
    entry->file_size = 4096u;
    entry->format = "gguf";
    entry->architecture = "deepseek";
    entry->tensor_count = 128u;
    entry->support_level = "generation-ready";
    entry->execution_ready = 0;
    entry->runtime_profile = "single-artifact";
    entry->runtime_binding = binding;
    entry->runtime_target = "deepseek4-v4-flash-dspark";
    entry->runtime_backend = "cuda";
    entry->runtime_mode = "dspark";
    entry->runtime_context = 8192u;
}

static int registry_fixture(const char *root, const char *registry_path)
{
    static const char *const aliases[] = {
        "deepseek4-v4-flash-profile-a",
        "deepseek4-v4-flash-profile-b",
    };
    static const char *const identities[] = {
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    };
    yvex_model_registry_options options = {0};
    yvex_model_registry_entry entry;
    yvex_model_registry *registry = NULL;
    char artifact[YVEX_PATH_CAP], binding[YVEX_PATH_CAP];
    char absolute_artifact[YVEX_PATH_CAP], absolute_binding[YVEX_PATH_CAP];
    yvex_error err;
    size_t index;

    if (mkdir(root, 0700) != 0 && errno != EEXIST) return 0;
    (void)snprintf(artifact, sizeof(artifact), "%s/artifact.gguf", root);
    (void)snprintf(binding, sizeof(binding), "%s/runtime.binding", root);
    if (!write_file(artifact, "fixture artifact\n") ||
        !write_file(binding, "fixture binding\n") ||
        !realpath(artifact, absolute_artifact) ||
        !realpath(binding, absolute_binding))
        return 0;
    options.registry_path = registry_path;
    options.create_if_missing = 1;
    if (yvex_model_registry_open(&registry, &options, &err) != YVEX_OK) return 0;
    for (index = 0u; index < 2u; ++index) {
        registry_entry(&entry, aliases[index], absolute_artifact,
                       absolute_binding, identities[index]);
        if (yvex_model_registry_add(registry, &entry, &err) != YVEX_OK) {
            yvex_model_registry_close(registry);
            return 0;
        }
    }
    if (yvex_model_registry_save(registry, registry_path, &err) != YVEX_OK) {
        yvex_model_registry_close(registry);
        return 0;
    }
    yvex_model_registry_close(registry);
    return 1;
}

static int render_frame(yvex_tui_state *state, char *output, size_t capacity)
{
    size_t count = 0u;
    unsigned int row = 0u, column = 0u;
    return yvex_tui_render(state, output, capacity, &count, &row, &column) == YVEX_OK &&
           count && count < capacity && row && column;
}

static int test_state_render_and_input(void)
{
    const char *root = "build/tests/tui-models";
    const char *registry_path = "build/tests/tui-models/models.local.json";
    const char *old_root = getenv("YVEX_MODELS_ROOT");
    const char *old_color = getenv("NO_COLOR");
    char *saved_root = old_root ? strdup(old_root) : NULL;
    char *saved_color = old_color ? strdup(old_color) : NULL;
    static const unsigned char paste[] =
        "\033[200~hello\nworld \xf0\x9f\x8c\x8d\033[201~";
    yvex_cli_interactive_event event;
    yvex_tui_launch_command command;
    yvex_tui_input input;
    yvex_tui_state state;
    yvex_error err;
    char frame[262144];
    size_t index;

    YVEX_TEST_ASSERT((!old_root || saved_root) && (!old_color || saved_color),
                     "save TUI test environment");
    YVEX_TEST_ASSERT(system("rm -rf build/tests/tui-models") == 0 &&
                         registry_fixture(root, registry_path) &&
                         setenv("YVEX_MODELS_ROOT", root, 1) == 0 &&
                         setenv("NO_COLOR", "1", 1) == 0,
                     "prepare isolated TUI model library");
    yvex_tui_state_init(&state, 30u, 120u, "main");
    YVEX_TEST_ASSERT(yvex_tui_models_load(&state, registry_path, &err) == YVEX_OK &&
                         state.model_count == 1u && state.models[0].profile_count == 2u,
                     "TUI consumes one logical model with subordinate profiles");
    YVEX_TEST_ASSERT(render_frame(&state, frame, sizeof(frame)) &&
                         strstr(frame, "RUNTIME OFFLINE") &&
                         strstr(frame, "Enter  Start Runtime") &&
                         !strstr(frame, "\033[48;") && !strstr(frame, "\033[40m"),
                     "offline Home is actionable and uses terminal-default background");
    memset(&input, 0, sizeof(input));
    YVEX_TEST_ASSERT(yvex_tui_input_byte(&input, &state, '\r') == YVEX_TUI_INPUT_NONE &&
                         state.overlay == YVEX_TUI_OVERLAY_RUNTIME_LAUNCH,
                     "Home Enter opens runtime launch selection");
    YVEX_TEST_ASSERT(yvex_tui_launch_prepare("/opt/yvex", &command, &err) == YVEX_OK &&
                         command.argc == 4 && !strcmp(command.argv[0], "/opt/yvex") &&
                         !strcmp(command.argv[1], "server") &&
                         !strcmp(command.argv[2], "--console") &&
                         !strcmp(command.argv[3], "off") && command.argv[4] == NULL,
                     "launcher enters the model-neutral canonical host with structured argv");

    state.overlay = YVEX_TUI_OVERLAY_NONE;
    state.focus = YVEX_TUI_FOCUS_CONTENT;
    yvex_tui_state_connection(&state, YVEX_TUI_CONNECTION_CONNECTED, "");
    memset(&event, 0, sizeof(event));
    event.kind = YVEX_CLI_INTERACTIVE_MESSAGE;
    event.message.kind = YVEX_CLIENT_MESSAGE_STATUS;
    event.message.runtime.status = YVEX_SERVER_STATUS_READY;
    event.message.runtime.host_ready = 1;
    memcpy(event.message.runtime.socket_path, "/tmp/yvex.sock", 15u);
    yvex_tui_state_message(&state, &event);
    YVEX_TEST_ASSERT(render_frame(&state, frame, sizeof(frame)) &&
                         strstr(frame, "HOST READY") && strstr(frame, "Enter  Load Model"),
                     "connected host without an engine collapses around model loading");
    YVEX_TEST_ASSERT(yvex_tui_input_byte(&input, &state, '\r') == YVEX_TUI_INPUT_NONE &&
                         state.overlay == YVEX_TUI_OVERLAY_RUNTIME_LAUNCH,
                     "host-ready Home can load a model without leaving the TUI");

    state.overlay = YVEX_TUI_OVERLAY_NONE;
    memset(&event, 0, sizeof(event));
    event.kind = YVEX_CLI_INTERACTIVE_MESSAGE;
    event.operation = YVEX_CLIENT_OP_ENGINE_LIST;
    event.message.kind = YVEX_CLIENT_MESSAGE_ENGINE;
    event.message.engine.schema_version = YVEX_SERVER_ENGINE_SCHEMA_V1;
    event.message.engine.state = YVEX_SERVER_ENGINE_LOADED;
    event.message.engine.backend = YVEX_BACKEND_KIND_CUDA;
    event.message.engine.generation_mode = YVEX_SERVER_GENERATION_DSPARK;
    event.message.engine.generation = 7u;
    (void)snprintf(event.message.engine.alias, sizeof(event.message.engine.alias),
                   "%s", "deepseek4-v4-flash-profile-a");
    (void)snprintf(event.message.engine.target_id,
                   sizeof(event.message.engine.target_id), "%s",
                   "deepseek4-v4-flash-dspark");
    yvex_tui_state_message(&state, &event);
    YVEX_TEST_ASSERT(state.active_engine.generation == 7u && state.models[0].resident &&
                         render_frame(&state, frame, sizeof(frame)) &&
                         strstr(frame, "ACTIVITY") && strstr(frame, "deepseek4-v4-flash-profile-a"),
                     "typed engine summary establishes connected Home and residency");

    memset(&state.active_engine, 0, sizeof(state.active_engine));
    memset(&event, 0, sizeof(event));
    event.kind = YVEX_CLI_INTERACTIVE_MESSAGE;
    event.operation = YVEX_CLIENT_OP_ENGINE_LOAD;
    event.message.kind = YVEX_CLIENT_MESSAGE_ERROR;
    memcpy(event.message.reason, "runtime binding rejected", 25u);
    yvex_tui_state_message(&state, &event);
    YVEX_TEST_ASSERT(state.launch_failure == YVEX_TUI_LAUNCH_FAILURE_ENGINE_LOAD &&
                         render_frame(&state, frame, sizeof(frame)) &&
                         strstr(frame, "MODEL LOAD FAILURE") &&
                         strstr(frame, "runtime binding rejected"),
                     "model-load refusal remains distinct from process exit metadata");

    yvex_tui_composer_clear(&state.composer);
    state.surface = YVEX_TUI_SURFACE_HOME;
    state.focus = YVEX_TUI_FOCUS_COMPOSER;
    memset(&input, 0, sizeof(input));
    for (index = 0u; index < sizeof(paste) - 1u; ++index)
        (void)yvex_tui_input_byte(&input, &state, paste[index]);
    YVEX_TEST_ASSERT(!strcmp((const char *)state.composer.bytes,
                             "hello\nworld \xf0\x9f\x8c\x8d") &&
                         state.composer.multiline,
                     "composer retains bracketed multiline UTF-8 paste");
    yvex_tui_state_resize(&state, 18u, 72u);
    YVEX_TEST_ASSERT(state.terminal.layout == YVEX_TUI_LAYOUT_COMPACT &&
                         !strcmp((const char *)state.composer.bytes,
                                 "hello\nworld \xf0\x9f\x8c\x8d"),
                     "compact resize preserves composer and application state");
    yvex_tui_state_close(&state);
    if (saved_root) (void)setenv("YVEX_MODELS_ROOT", saved_root, 1);
    else (void)unsetenv("YVEX_MODELS_ROOT");
    if (saved_color) (void)setenv("NO_COLOR", saved_color, 1);
    else (void)unsetenv("NO_COLOR");
    free(saved_root);
    free(saved_color);
    return 0;
}

static int termios_equal(const struct termios *left, const struct termios *right)
{
    return left->c_iflag == right->c_iflag && left->c_oflag == right->c_oflag &&
           left->c_cflag == right->c_cflag && left->c_lflag == right->c_lflag &&
           !memcmp(left->c_cc, right->c_cc, sizeof(left->c_cc));
}

static int terminal_output(int fd, char *output, size_t capacity)
{
    size_t used = 0u;
    while (used + 1u < capacity) {
        ssize_t count = read(fd, output + used, capacity - used - 1u);
        if (count > 0) used += (size_t)count;
        else if (count < 0 && errno == EINTR) continue;
        else break;
    }
    output[used] = '\0';
    return (int)used;
}

static int test_terminal_transaction(void)
{
    yvex_tui_terminal terminal;
    yvex_error err;
    struct termios before, during, after;
    struct winsize dimensions = {0};
    char output[1024], *slave_name;
    unsigned int rows = 0u, columns = 0u;
    int master, slave, resize = 0, interrupt = 0, terminate = 0;

    master = posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK);
    YVEX_TEST_ASSERT(master >= 0 && grantpt(master) == 0 && unlockpt(master) == 0,
                     "open terminal fixture");
    slave_name = ptsname(master);
    slave = slave_name ? open(slave_name, O_RDWR | O_NOCTTY) : -1;
    YVEX_TEST_ASSERT(slave >= 0 && tcgetattr(slave, &before) == 0,
                     "capture terminal fixture attributes");
    dimensions.ws_row = 31u;
    dimensions.ws_col = 137u;
    YVEX_TEST_ASSERT(ioctl(slave, TIOCSWINSZ, &dimensions) == 0 &&
                         yvex_tui_terminal_open(&terminal, slave, slave, &err) == YVEX_OK,
                     "enter full-screen terminal transaction");
    YVEX_TEST_ASSERT(tcgetattr(slave, &during) == 0 && !(during.c_lflag & ICANON) &&
                         !(during.c_lflag & ECHO) &&
                         yvex_tui_terminal_dimensions(&terminal, &rows, &columns) &&
                         rows == 31u && columns == 137u,
                     "terminal uses cbreak input and authoritative dimensions");
    YVEX_TEST_ASSERT(raise(SIGWINCH) == 0, "observe terminal resize signal");
    yvex_tui_terminal_take_signals(&terminal, &resize, &interrupt, &terminate);
    YVEX_TEST_ASSERT(resize && !interrupt && !terminate,
                     "signal handler publishes resize without rendering");
    YVEX_TEST_ASSERT(terminal_output(master, output, sizeof(output)) > 0 &&
                         strstr(output, "\033[?1049h") && strstr(output, "\033[?25l") &&
                         strstr(output, "\033[?2004h"),
                     "terminal enters alternate screen cursor and paste modes");
    yvex_tui_terminal_close(&terminal);
    YVEX_TEST_ASSERT(tcgetattr(slave, &after) == 0 && termios_equal(&before, &after),
                     "terminal attributes restore exactly");
    YVEX_TEST_ASSERT(terminal_output(master, output, sizeof(output)) > 0 &&
                         strstr(output, "\033[?1049l") && strstr(output, "\033[?25h") &&
                         strstr(output, "\033[?2004l"),
                     "terminal restores screen cursor and paste modes");
    (void)close(slave);
    (void)close(master);
    return 0;
}

static int test_terminal_partial_init_rollback(void)
{
    yvex_tui_terminal terminal;
    yvex_error err;
    struct termios before, after;
    char *slave_name;
    int master, input, read_only_output;

    master = posix_openpt(O_RDWR | O_NOCTTY);
    YVEX_TEST_ASSERT(master >= 0 && grantpt(master) == 0 && unlockpt(master) == 0,
                     "open terminal rollback fixture");
    slave_name = ptsname(master);
    input = slave_name ? open(slave_name, O_RDWR | O_NOCTTY) : -1;
    read_only_output = slave_name ? open(slave_name, O_RDONLY | O_NOCTTY) : -1;
    YVEX_TEST_ASSERT(input >= 0 && read_only_output >= 0 &&
                         tcgetattr(input, &before) == 0,
                     "capture rollback terminal attributes");
    YVEX_TEST_ASSERT(yvex_tui_terminal_open(&terminal, input, read_only_output,
                                             &err) != YVEX_OK &&
                         tcgetattr(input, &after) == 0 &&
                         termios_equal(&before, &after) &&
                         terminal.signal_read_fd < 0 &&
                         terminal.signal_write_fd < 0,
                     "failed alternate-screen entry rolls back terminal transaction");
    (void)close(read_only_output);
    (void)close(input);
    (void)close(master);
    return 0;
}

static int test_launcher_bootstrap_failure(void)
{
    yvex_tui_launcher launcher;
    yvex_error err;
    struct timespec pause = {0, 1000000};
    unsigned int attempt;

    YVEX_TEST_ASSERT(yvex_tui_launcher_open(&launcher, "/bin/false", &err) == YVEX_OK &&
                         yvex_tui_launcher_start(&launcher, 1u, &err) == YVEX_OK,
                     "launch canonical command through a separate process");
    for (attempt = 0u; attempt < 1000u && !yvex_tui_launcher_reap(&launcher);
         ++attempt)
        (void)nanosleep(&pause, NULL);
    YVEX_TEST_ASSERT(launcher.exit_known && launcher.exit_status == 1 &&
                         launcher.diagnostic_count < sizeof(launcher.diagnostic),
                     "pre-handshake child exit remains a bounded bootstrap failure");
    yvex_tui_launcher_close(&launcher);
    return 0;
}

int yvex_test_tui(void)
{
    if (test_state_render_and_input()) return 1;
    if (test_terminal_transaction()) return 1;
    if (test_terminal_partial_init_rollback()) return 1;
    if (test_launcher_bootstrap_failure()) return 1;
    return 0;
}
