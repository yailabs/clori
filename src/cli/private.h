/* Owner: client.dispatch source-local interface.
 * Owns: typed handoff from canonical command resolution into the runtime-client lane.
 * Does not own: registry data, domain adapters, protocol serialization, or engine capability.
 * Invariants: the resolved descriptor has runtime-client lane identity and cannot fall back.
 * Boundary: source-local interface shared only by cli.main and cli.io.client.
 * Purpose: declare runtime dispatch and registry-driven discovery rendering.
 * Inputs: compiled descriptors and admitted argv. Effects: declarations only.
 * Failure: implementations return stable process status. */
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
