/* Platform lifetime oracle. No editor, model, host or protocol is simulated. */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/io/terminal/private.h"
#include <assert.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <time.h>

static volatile sig_atomic_t prior_interrupts;
static atomic_int attempts;

static void prior_handler(int number)
{
    (void)number;
    prior_interrupts++;
}

static int handle_interrupt(void *context)
{
    atomic_int *count = context;
    return atomic_fetch_add(count, 1) >= 2;
}

static void wait_attempts(int wanted)
{
    struct timespec delay = {0, 1000000L};
    unsigned int wait;
    for (wait = 0u; wait < 2000u && atomic_load(&attempts) < wanted; ++wait)
        (void)nanosleep(&delay, NULL);
    assert(atomic_load(&attempts) >= wanted);
}

int main(void)
{
    struct sigaction action = {0}, previous, restored;
    yvex_cli_interrupt *scope = NULL, *other = NULL;
    yvex_cli_output_scope *output = NULL;
    yvex_error err;
    unsigned int cycle;
    FILE *file = tmpfile();
    assert(file && !yvex_cli_terminal_interactive(file) && !yvex_cli_terminal_width(file));
    assert(!yvex_cli_terminal_interactive(NULL) && !yvex_cli_terminal_width(NULL));
    assert(yvex_cli_output_scope_open(NULL, &err) == YVEX_ERR_INVALID_ARG);
    assert(yvex_cli_output_scope_close(&output, &err) == YVEX_OK);
    assert(yvex_cli_interrupt_open(NULL, &err) == YVEX_ERR_INVALID_ARG);
    assert(yvex_cli_interrupt_watch(NULL, handle_interrupt, &attempts, &err) == YVEX_ERR_STATE);
    action.sa_handler = prior_handler;
    (void)sigemptyset(&action.sa_mask);
    assert(sigaction(SIGINT, &action, &previous) == 0);
    for (cycle = 0u; cycle < 32u; ++cycle) {
        int finished;
        assert(yvex_cli_interrupt_open(&scope, &err) == YVEX_OK && scope);
        assert(yvex_cli_interrupt_open(&other, &err) == YVEX_ERR_STATE && !other);
        assert(!yvex_cli_interrupt_count(scope));
        atomic_store(&attempts, 0);
        assert(yvex_cli_interrupt_watch(scope, handle_interrupt, &attempts, &err) == YVEX_OK);
        assert(yvex_cli_interrupt_watch(scope, handle_interrupt, &attempts, &err) == YVEX_ERR_STATE);
        assert(raise(SIGINT) == 0);
        wait_attempts(1);
        assert(raise(SIGINT) == 0);
        wait_attempts(3);
        assert(yvex_cli_interrupt_unwatch(scope) == 2u);
        finished = atomic_load(&attempts);
        assert(finished == 3); /* acknowledged interrupts do not repeat */
        yvex_cli_interrupt_clear(scope);
        assert(!yvex_cli_interrupt_count(scope));
        yvex_cli_interrupt_record(scope); /* editor-authored generic interruption */
        assert(yvex_cli_interrupt_count(scope) == 1u);
        yvex_cli_interrupt_clear(scope);
        assert(yvex_cli_interrupt_watch(scope, handle_interrupt, &attempts, &err) == YVEX_OK);
        assert(yvex_cli_interrupt_unwatch(scope) == 0u); /* idle close wakes and joins */
        assert(atomic_load(&attempts) == finished);
        assert(yvex_cli_interrupt_close(&scope, &err) == YVEX_OK && !scope);
        assert(yvex_cli_interrupt_close(&scope, &err) == YVEX_OK);
        assert(sigaction(SIGINT, NULL, &restored) == 0 && restored.sa_handler == prior_handler);
    }
    assert(raise(SIGINT) == 0 && prior_interrupts == 1);
    assert(sigaction(SIGINT, &previous, NULL) == 0);
    assert(fclose(file) == 0);
    puts("terminal scope: 32 capture/watch/retry/close cycles, prior handler restored, no request semantics");
    return 0;
}
