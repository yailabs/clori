/* Own the reversible POSIX terminal transaction and signal-safe wakeup path. */
#define _POSIX_C_SOURCE 200809L

#include "src/cli/tui/private.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

static yvex_tui_terminal *active_terminal;

static int terminal_refuse(yvex_error *err, yvex_status status,
                           const char *message)
{
    yvex_error_set(err, status, "cli.tui.terminal", message);
    return status;
}

static int write_all(int fd, const char *bytes, size_t count)
{
    size_t written = 0u;
    while (written < count) {
        ssize_t result = write(fd, bytes + written, count - written);
        if (result > 0) {
            written += (size_t)result;
            continue;
        }
        if (result < 0 && errno == EINTR) continue;
        return 0;
    }
    return 1;
}

static void terminal_signal(int number)
{
    yvex_tui_terminal *terminal = active_terminal;
    unsigned char byte = (unsigned char)number;
    if (!terminal) return;
    if (number == SIGWINCH)
        terminal->resize_pending = 1;
    else if (number == SIGINT)
        terminal->interrupt_pending = 1;
    else if (number == SIGTERM)
        terminal->terminate_pending = 1;
    if (terminal->signal_write_fd >= 0) {
        ssize_t ignored = write(terminal->signal_write_fd, &byte, sizeof(byte));
        (void)ignored;
    }
}

static int terminal_pipe_open(yvex_tui_terminal *terminal)
{
    int descriptors[2];
    if (pipe(descriptors) != 0) return 0;
    terminal->signal_read_fd = descriptors[0];
    terminal->signal_write_fd = descriptors[1];
    if (fcntl(descriptors[0], F_SETFL, O_NONBLOCK) != 0 ||
        fcntl(descriptors[1], F_SETFL, O_NONBLOCK) != 0 ||
        fcntl(descriptors[0], F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(descriptors[1], F_SETFD, FD_CLOEXEC) != 0) {
        (void)close(descriptors[0]);
        (void)close(descriptors[1]);
        terminal->signal_read_fd = -1;
        terminal->signal_write_fd = -1;
        return 0;
    }
    return 1;
}

static int terminal_signals_open(yvex_tui_terminal *terminal)
{
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = terminal_signal;
    (void)sigemptyset(&action.sa_mask);
    if (sigaction(SIGWINCH, &action, &terminal->saved_winch) != 0) return 0;
    if (sigaction(SIGINT, &action, &terminal->saved_interrupt) != 0) {
        (void)sigaction(SIGWINCH, &terminal->saved_winch, NULL);
        return 0;
    }
    if (sigaction(SIGTERM, &action, &terminal->saved_terminate) != 0) {
        (void)sigaction(SIGINT, &terminal->saved_interrupt, NULL);
        (void)sigaction(SIGWINCH, &terminal->saved_winch, NULL);
        return 0;
    }
    terminal->signals_installed = 1;
    active_terminal = terminal;
    return 1;
}

int yvex_tui_terminal_open(yvex_tui_terminal *terminal, int input_fd,
                           int output_fd, yvex_error *err)
{
    static const char enter[] = "\033[?1049h\033[?25l\033[?2004h";
    struct termios mode;
    if (!terminal)
        return terminal_refuse(err, YVEX_ERR_INVALID_ARG,
                               "terminal state is required");
    memset(terminal, 0, sizeof(*terminal));
    terminal->input_fd = input_fd;
    terminal->output_fd = output_fd;
    terminal->signal_read_fd = -1;
    terminal->signal_write_fd = -1;
    if (!isatty(input_fd) || !isatty(output_fd))
        return terminal_refuse(err, YVEX_ERR_STATE, "chat requires a terminal");
    if (active_terminal)
        return terminal_refuse(err, YVEX_ERR_STATE,
                               "another TUI terminal transaction is active");
    if (tcgetattr(input_fd, &terminal->saved_termios) != 0)
        return terminal_refuse(err, YVEX_ERR_IO,
                               "cannot capture terminal attributes");
    terminal->termios_saved = 1;
    if (!terminal_pipe_open(terminal)) {
        yvex_tui_terminal_close(terminal);
        return terminal_refuse(err, YVEX_ERR_IO,
                               "cannot create signal wakeup pipe");
    }
    if (!terminal_signals_open(terminal)) {
        yvex_tui_terminal_close(terminal);
        return terminal_refuse(err, YVEX_ERR_IO,
                               "cannot install terminal signal observers");
    }
    mode = terminal->saved_termios;
    mode.c_iflag &= (tcflag_t)~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    mode.c_lflag &= (tcflag_t)~(ECHO | ICANON | IEXTEN);
    mode.c_cc[VMIN] = 0;
    mode.c_cc[VTIME] = 0;
    if (tcsetattr(input_fd, TCSAFLUSH, &mode) != 0) {
        yvex_tui_terminal_close(terminal);
        return terminal_refuse(err, YVEX_ERR_IO,
                               "cannot enter terminal input mode");
    }
    terminal->termios_changed = 1;
    /* A short terminal write may already have activated one or more modes. Mark the
     * whole transaction before the batch so rollback always emits the inverse. */
    terminal->alternate_screen = 1;
    terminal->cursor_hidden = 1;
    terminal->paste_enabled = 1;
    if (!write_all(output_fd, enter, sizeof(enter) - 1u)) {
        yvex_tui_terminal_close(terminal);
        return terminal_refuse(err, YVEX_ERR_IO,
                               "cannot enter alternate terminal screen");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_tui_terminal_close(yvex_tui_terminal *terminal)
{
    static const char leave[] = "\033[0m\033[?2004l\033[?25h\033[?1049l";
    if (!terminal) return;
    if (terminal->alternate_screen || terminal->cursor_hidden ||
        terminal->paste_enabled)
        (void)write_all(terminal->output_fd, leave, sizeof(leave) - 1u);
    terminal->alternate_screen = 0;
    terminal->cursor_hidden = 0;
    terminal->paste_enabled = 0;
    if (terminal->termios_changed && terminal->termios_saved)
        (void)tcsetattr(terminal->input_fd, TCSAFLUSH,
                        &terminal->saved_termios);
    terminal->termios_changed = 0;
    if (active_terminal == terminal) active_terminal = NULL;
    if (terminal->signals_installed) {
        (void)sigaction(SIGTERM, &terminal->saved_terminate, NULL);
        (void)sigaction(SIGINT, &terminal->saved_interrupt, NULL);
        (void)sigaction(SIGWINCH, &terminal->saved_winch, NULL);
        terminal->signals_installed = 0;
    }
    if (terminal->signal_read_fd >= 0) (void)close(terminal->signal_read_fd);
    if (terminal->signal_write_fd >= 0) (void)close(terminal->signal_write_fd);
    terminal->signal_read_fd = -1;
    terminal->signal_write_fd = -1;
}

int yvex_tui_terminal_dimensions(yvex_tui_terminal *terminal,
                                 unsigned int *rows, unsigned int *columns)
{
    struct winsize size;
    if (!terminal || !rows || !columns) return 0;
    memset(&size, 0, sizeof(size));
    if (ioctl(terminal->output_fd, TIOCGWINSZ, &size) != 0 &&
        ioctl(terminal->input_fd, TIOCGWINSZ, &size) != 0) {
        size.ws_row = 24u;
        size.ws_col = 80u;
    }
    *rows = size.ws_row ? size.ws_row : 24u;
    *columns = size.ws_col ? size.ws_col : 80u;
    if (*rows > 256u) *rows = 256u;
    if (*columns > 512u) *columns = 512u;
    return 1;
}

int yvex_tui_terminal_signal_fd(const yvex_tui_terminal *terminal)
{
    return terminal ? terminal->signal_read_fd : -1;
}

void yvex_tui_terminal_take_signals(yvex_tui_terminal *terminal, int *resize,
                                    int *interrupt, int *terminate)
{
    unsigned char bytes[64];
    if (!terminal) return;
    while (read(terminal->signal_read_fd, bytes, sizeof(bytes)) > 0) {}
    if (resize) *resize = terminal->resize_pending != 0;
    if (interrupt) *interrupt = terminal->interrupt_pending != 0;
    if (terminate) *terminate = terminal->terminate_pending != 0;
    terminal->resize_pending = 0;
    terminal->interrupt_pending = 0;
    terminal->terminate_pending = 0;
}

int yvex_tui_terminal_write(yvex_tui_terminal *terminal,
                            const char *bytes, size_t count)
{
    return terminal && bytes && write_all(terminal->output_fd, bytes, count);
}
