/* POSIX terminal mechanics below the platform-neutral client contract.
 * REPLAI owns editing. This adapter owns observation, temporary quiet output,
 * process interrupt capture and its bounded worker lifetime, never requests. */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/io/terminal/private.h"
#include <replai.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "interrupt capture requires lock-free unsigned atomics");

struct yvex_cli_output_scope { struct termios saved; };
struct yvex_cli_interrupt {
    struct sigaction previous;
    int wake[2], watching;
    pthread_t worker;
    atomic_int stopping;
    int (*handle)(void *context);
    void *context;
};

/* A CLI process has one terminal interaction owner. The handler never touches
 * the owner allocation; its lock-free count and pipe remain valid until the
 * worker joins and the prior handler is restored. */
static pthread_mutex_t capture_mutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_uint captured;
static volatile sig_atomic_t wake_descriptor = -1;
static int capture_owned;

static int terminal_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "client.terminal", reason);
    return status;
}

int yvex_cli_terminal_interactive(FILE *stream)
{
    int descriptor = stream ? fileno(stream) : -1;
    return descriptor >= 0 && isatty(descriptor);
}

unsigned int yvex_cli_terminal_width(FILE *stream)
{
    struct winsize size = {0};
    int descriptor = stream ? fileno(stream) : -1;
    return descriptor >= 0 && ioctl(descriptor, TIOCGWINSZ, &size) == 0
               ? size.ws_col : 0u;
}

int yvex_cli_terminal_editor_open(struct replai_handle *editor)
{
    return replai_open(editor, STDIN_FILENO, STDOUT_FILENO);
}

int yvex_cli_output_scope_open(yvex_cli_output_scope **out, yvex_error *err)
{
    yvex_cli_output_scope *scope;
    struct termios quiet;
    if (!out) return terminal_refuse(err, YVEX_ERR_INVALID_ARG, "output scope is required");
    *out = NULL;
    if (!yvex_cli_terminal_interactive(stdin)) return YVEX_OK;
    scope = calloc(1u, sizeof(*scope));
    if (!scope) return terminal_refuse(err, YVEX_ERR_NOMEM, "output scope allocation failed");
    if (tcgetattr(STDIN_FILENO, &scope->saved) != 0) {
        free(scope);
        return terminal_refuse(err, YVEX_ERR_IO, "terminal state capture failed");
    }
    quiet = scope->saved;
    quiet.c_lflag &= (tcflag_t)~(ECHO | ECHONL);
    if (tcsetattr(STDIN_FILENO, TCSANOW, &quiet) != 0) {
        free(scope);
        return terminal_refuse(err, YVEX_ERR_IO, "quiet output admission failed");
    }
    *out = scope;
    return YVEX_OK;
}

int yvex_cli_output_scope_close(yvex_cli_output_scope **owned, yvex_error *err)
{
    yvex_cli_output_scope *scope;
    int flushed, restored;
    if (!owned || !*owned) return YVEX_OK;
    scope = *owned;
    /* Product does not admit a draft while a request owns output. */
    flushed = tcflush(STDIN_FILENO, TCIFLUSH);
    restored = tcsetattr(STDIN_FILENO, TCSANOW, &scope->saved);
    free(scope);
    *owned = NULL;
    return flushed == 0 && restored == 0 ? YVEX_OK
        : terminal_refuse(err, YVEX_ERR_IO, "terminal output restoration failed");
}

static void interrupt_record(void)
{
    unsigned int value = atomic_load_explicit(&captured, memory_order_relaxed);
    while (value != UINT_MAX && !atomic_compare_exchange_weak_explicit(
        &captured, &value, value + 1u, memory_order_relaxed, memory_order_relaxed)) {}
}

static void wake_send(int descriptor)
{
    const unsigned char byte = 1u;
    ssize_t written;
    do { written = write(descriptor, &byte, 1u); } while (written < 0 && errno == EINTR);
    /* EAGAIN means a wake is already queued. Counts, not pipe bytes, own events. */
}

static void interrupt_handler(int number)
{
    int saved_errno = errno;
    (void)number;
    interrupt_record();
    if (wake_descriptor >= 0) wake_send(wake_descriptor);
    errno = saved_errno;
}

static int wake_pipe_open(int descriptors[2])
{
    size_t index;
    if (pipe(descriptors) != 0) return 0;
    for (index = 0u; index < 2u; ++index)
        if (fcntl(descriptors[index], F_SETFD, FD_CLOEXEC) != 0 ||
            fcntl(descriptors[index], F_SETFL, O_NONBLOCK) != 0) {
            (void)close(descriptors[0]);
            (void)close(descriptors[1]);
            return 0;
        }
    return 1;
}

int yvex_cli_interrupt_open(yvex_cli_interrupt **out, yvex_error *err)
{
    yvex_cli_interrupt *scope;
    struct sigaction action = {0};
    int rc = YVEX_OK;
    if (!out) return terminal_refuse(err, YVEX_ERR_INVALID_ARG, "interrupt scope is required");
    *out = NULL;
    if (pthread_mutex_lock(&capture_mutex) != 0)
        return terminal_refuse(err, YVEX_ERR_STATE, "interrupt capture lock failed");
    if (capture_owned) {
        (void)pthread_mutex_unlock(&capture_mutex);
        return terminal_refuse(err, YVEX_ERR_STATE, "interrupt capture already has an owner");
    }
    scope = calloc(1u, sizeof(*scope));
    if (!scope || !wake_pipe_open(scope->wake)) {
        free(scope);
        (void)pthread_mutex_unlock(&capture_mutex);
        return terminal_refuse(err, YVEX_ERR_NOMEM, "interrupt capture allocation failed");
    }
    action.sa_handler = interrupt_handler;
    (void)sigemptyset(&action.sa_mask);
    atomic_store_explicit(&captured, 0u, memory_order_relaxed);
    wake_descriptor = scope->wake[1];
    if (sigaction(SIGINT, &action, &scope->previous) != 0) {
        wake_descriptor = -1;
        (void)close(scope->wake[0]);
        (void)close(scope->wake[1]);
        free(scope);
        rc = terminal_refuse(err, YVEX_ERR_IO, "interrupt handler admission failed");
    } else {
        atomic_init(&scope->stopping, 0);
        capture_owned = 1;
        *out = scope;
    }
    (void)pthread_mutex_unlock(&capture_mutex);
    return rc;
}

unsigned int yvex_cli_interrupt_count(const yvex_cli_interrupt *scope)
{
    return scope ? atomic_load_explicit(&captured, memory_order_relaxed) : 0u;
}

void yvex_cli_interrupt_clear(yvex_cli_interrupt *scope)
{
    if (scope && !scope->watching) atomic_store_explicit(&captured, 0u, memory_order_relaxed);
}

void yvex_cli_interrupt_record(yvex_cli_interrupt *scope)
{
    if (scope) interrupt_record();
}

static void *interrupt_worker(void *opaque)
{
    yvex_cli_interrupt *scope = opaque;
    struct pollfd wake = {.fd = scope->wake[0], .events = POLLIN};
    int handled = 0;
    while (!atomic_load_explicit(&scope->stopping, memory_order_acquire)) {
        unsigned char bytes[64];
        int pending = yvex_cli_interrupt_count(scope) != 0u;
        if (pending && !handled) handled = scope->handle(scope->context);
        if (atomic_load_explicit(&scope->stopping, memory_order_acquire)) break;
        if (poll(&wake, 1u, pending && !handled ? 10 : -1) < 0 && errno != EINTR) break;
        while (read(scope->wake[0], bytes, sizeof(bytes)) > 0) {}
    }
    return NULL;
}

int yvex_cli_interrupt_watch(yvex_cli_interrupt *scope,
    int (*handle)(void *context), void *context, yvex_error *err)
{
    if (!scope || scope->watching || !handle)
        return terminal_refuse(err, YVEX_ERR_STATE, "one idle interrupt scope and callback are required");
    scope->handle = handle;
    scope->context = context;
    atomic_store_explicit(&scope->stopping, 0, memory_order_release);
    if (pthread_create(&scope->worker, NULL, interrupt_worker, scope) != 0)
        return terminal_refuse(err, YVEX_ERR_IO, "interrupt watch admission failed");
    scope->watching = 1;
    return YVEX_OK;
}

unsigned int yvex_cli_interrupt_unwatch(yvex_cli_interrupt *scope)
{
    if (!scope) return 0u;
    if (scope->watching) {
        atomic_store_explicit(&scope->stopping, 1, memory_order_release);
        wake_send(scope->wake[1]);
        (void)pthread_join(scope->worker, NULL);
        scope->watching = 0;
        scope->handle = NULL;
        scope->context = NULL;
    }
    return yvex_cli_interrupt_count(scope);
}

int yvex_cli_interrupt_close(yvex_cli_interrupt **owned, yvex_error *err)
{
    yvex_cli_interrupt *scope;
    if (!owned || !*owned) return YVEX_OK;
    scope = *owned;
    (void)yvex_cli_interrupt_unwatch(scope);
    if (sigaction(SIGINT, &scope->previous, NULL) != 0)
        return terminal_refuse(err, YVEX_ERR_IO, "interrupt handler restoration failed");
    wake_descriptor = -1;
    (void)close(scope->wake[0]);
    (void)close(scope->wake[1]);
    free(scope);
    *owned = NULL;
    (void)pthread_mutex_lock(&capture_mutex);
    capture_owned = 0;
    (void)pthread_mutex_unlock(&capture_mutex);
    return YVEX_OK;
}
