/* Terminal platform adaptation keeps descriptor, signal and console layouts local.
 * This is not a public terminal API or a replacement editor. */
#ifndef SRC_CLI_IO_TERMINAL_PRIVATE_H_INCLUDED
#define SRC_CLI_IO_TERMINAL_PRIVATE_H_INCLUDED

#include <stdio.h>
#include <yvex/core.h>

struct replai_handle;
typedef struct yvex_cli_interrupt yvex_cli_interrupt;
typedef struct yvex_cli_output_scope yvex_cli_output_scope;

int yvex_cli_terminal_interactive(FILE *stream);
/* Zero means unavailable; renderers own fallback/override policy. */
unsigned int yvex_cli_terminal_width(FILE *stream);
/* Adapts the pinned editor's platform entrypoint; returns its status code. */
int yvex_cli_terminal_editor_open(struct replai_handle *editor);
int yvex_cli_output_scope_open(yvex_cli_output_scope **out, yvex_error *err);
int yvex_cli_output_scope_close(yvex_cli_output_scope **scope, yvex_error *err);

/* One process-local capture scope; nested capture refuses. Editing and work
 * watches share it. Counts have no application meaning at this boundary. */
int yvex_cli_interrupt_open(yvex_cli_interrupt **out, yvex_error *err);
unsigned int yvex_cli_interrupt_count(const yvex_cli_interrupt *scope);
void yvex_cli_interrupt_clear(yvex_cli_interrupt *scope);
void yvex_cli_interrupt_record(yvex_cli_interrupt *scope);
/* Called outside signal context after an interrupt. Return nonzero when the
 * application has handled it, zero to retry after a bounded wait. A callback
 * must finish in bounded time. Stop joins before borrowed context expires. */
int yvex_cli_interrupt_watch(yvex_cli_interrupt *scope,
    int (*handle)(void *context), void *context, yvex_error *err);
unsigned int yvex_cli_interrupt_unwatch(yvex_cli_interrupt *scope);
int yvex_cli_interrupt_close(yvex_cli_interrupt **scope, yvex_error *err);

#endif
