/* Owner: client.dispatch source-local interface.
 * Owns: the bounded handoff between the sole yvex entrypoint and its runtime-client lane.
 * Does not own: command grammar, domain adapters, rendering, protocol, or engine capability.
 * Invariants: one process entrypoint delegates unclaimed routes exactly once to the client lane.
 * Boundary: source-local interface shared only by cli.main and cli.io.client.
 * Purpose: declare the unified executable dispatch seam without importing engine headers.
 * Inputs: process argc/argv. Effects: declarations only.
 * Failure: implementations return stable process status. */
#ifndef SRC_CLI_PRIVATE_H_INCLUDED
#define SRC_CLI_PRIVATE_H_INCLUDED

int yvex_client_dispatch(int argc, char **argv);

#endif
