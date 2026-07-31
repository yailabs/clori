/*
 * Declare runtime dispatch and registry-driven discovery rendering.
 *
 * The resolved descriptor has runtime-client lane identity and cannot fall back. Source-local
 * interface shared only by cli.main and cli.io.client.
 */
#ifndef SRC_CLI_PRIVATE_H_INCLUDED
#define SRC_CLI_PRIVATE_H_INCLUDED

#include <stddef.h>

struct yvex_operator_descriptor;

int yvex_client_dispatch(const struct yvex_operator_descriptor *operation,
                         int argc, char **argv, size_t consumed);
int yvex_client_render_help_path(size_t path_count, const char *const *path,
                                 int advanced, int json);
void yvex_client_render_usage_error(
    const struct yvex_operator_descriptor *operation);

#endif
