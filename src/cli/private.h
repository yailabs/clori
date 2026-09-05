/*
 * Declare runtime dispatch and registry-driven discovery rendering.
 *
 * The resolved descriptor selects one non-overlapping client, foreground-server, or offline lane.
 * This source-local interface keeps those product entry paths under the one registry dispatcher.
 */
#ifndef SRC_CLI_PRIVATE_H_INCLUDED
#define SRC_CLI_PRIVATE_H_INCLUDED

#include <stddef.h>

struct yvex_operator_descriptor;

int yvex_client_dispatch(const struct yvex_operator_descriptor *operation,
                         int argc, char **argv, size_t consumed);
size_t yvex_cli_command_distance(const char *left, const char *right);
int yvex_cli_server_dispatch(int argc, char **argv, size_t consumed);

#endif
