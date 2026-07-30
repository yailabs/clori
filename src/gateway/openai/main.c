/* Owner: gateway.openai entrypoint.
 * Owns: yvex-openai process arguments, signals, local defaults, and gateway lifecycle invocation.
 * Does not own: HTTP parsing, OpenAI translation, YVEX sessions, runtime models, or tool execution.
 * Invariants: only 127.0.0.1 is admitted and the process never starts yvexd implicitly.
 * Boundary: one lightweight packaged executable over the private local-protocol client objects.
 * Purpose: expose the bounded OpenAI compatibility profile as a local application provider.
 * Inputs: host/port/YVEX socket arguments and process termination signals.
 * Effects: runs one loopback listener and writes only entrypoint diagnostics to stderr.
 * Failure: exits nonzero without opening artifacts, models, CUDA, or runtime engine state. */
#define _POSIX_C_SOURCE 200809L
#include "src/gateway/openai/private.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t stop_requested;

/* Purpose: request graceful listener shutdown from admitted process signals. */
static void stop_signal(int signal_number)
{
    (void)signal_number;
    stop_requested = 1;
}

/* Purpose: print the complete compact gateway grammar. */
static void usage(FILE *output)
{
    fprintf(output,
            "Usage: yvex-openai [--host 127.0.0.1] [--port 8001] "
            "[--yvex-socket PATH] [--timeout-ms 600000]\n"
            "\nBounded local OpenAI compatibility profile: %s\n",
            OPENAI_COMPAT_PROFILE);
}

/* Purpose: parse one nonzero TCP port without integer coercion.
 * Inputs: CLI text and port output. Effects: writes one nonzero 16-bit port.
 * Failure: returns false for malformed, zero, or out-of-range input.
 * Boundary: process configuration only. */
static int port_parse(const char *text, unsigned short *port)
{
    char *end = NULL;
    unsigned long value;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno || !end || *end || !value || value > 65535u) return 0;
    *port = (unsigned short)value;
    return 1;
}

/* Purpose: parse one bounded nonzero gateway-to-daemon timeout in milliseconds.
 * Inputs: decimal CLI text and caller-owned timeout output.
 * Effects: publishes only values between 100 ms and one day.
 * Failure: returns false without modifying the output for malformed or out-of-range text.
 * Boundary: process configuration only; runtime waiting remains protocol-owned. */
static int timeout_parse(const char *text, unsigned long long *milliseconds)
{
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno || !end || *end || value < 100u || value > 86400000u)
        return 0;
    *milliseconds = value;
    return 1;
}

/* Purpose: admit process configuration and run the lightweight compatibility gateway.
 * Inputs: process arguments and environment-derived socket default.
 * Effects: installs signal handlers, runs gateway lifecycle, and emits fatal diagnostics.
 * Failure: exits nonzero after concise configuration/runtime error reporting.
 * Boundary: entrypoint owns process I/O but links no inference engine. */
int main(int argc, char **argv)
{
    openai_gateway gateway;
    struct sigaction action;
    yvex_error err;
    int index, rc;
    memset(&gateway, 0, sizeof(gateway));
    strcpy(gateway.host, "127.0.0.1");
    gateway.port = 8001u;
    gateway.yvex_timeout_ms = 600000u;
    if (yvex_server_socket_path(gateway.yvex_socket, &err) != YVEX_OK) {
        fprintf(stderr, "yvex-openai: %s\n", yvex_error_message(&err));
        return 1;
    }
    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--help") == 0) {
            usage(stdout);
            return 0;
        }
        if (strcmp(argv[index], "--version") == 0) {
            puts(OPENAI_COMPAT_PROFILE " protocol-v2");
            return 0;
        }
        if (strcmp(argv[index], "--host") == 0 && index + 1 < argc) {
            if (strlen(argv[++index]) >= sizeof(gateway.host)) goto invalid;
            strcpy(gateway.host, argv[index]);
        } else if (strcmp(argv[index], "--port") == 0 && index + 1 < argc) {
            if (!port_parse(argv[++index], &gateway.port)) goto invalid;
        } else if (strcmp(argv[index], "--yvex-socket") == 0 && index + 1 < argc) {
            if (strlen(argv[++index]) >= sizeof(gateway.yvex_socket)) goto invalid;
            strcpy(gateway.yvex_socket, argv[index]);
        } else if (strcmp(argv[index], "--timeout-ms") == 0 &&
                   index + 1 < argc) {
            if (!timeout_parse(argv[++index], &gateway.yvex_timeout_ms))
                goto invalid;
        } else goto invalid;
    }
    if (strcmp(gateway.host, "127.0.0.1") != 0) {
        fprintf(stderr, "yvex-openai: non-loopback bind addresses are refused\n");
        return 1;
    }
    memset(&action, 0, sizeof(action));
    action.sa_handler = stop_signal;
    sigemptyset(&action.sa_mask);
    (void)sigaction(SIGINT, &action, NULL);
    (void)sigaction(SIGTERM, &action, NULL);
    fprintf(stderr, "yvex-openai: listening on http://%s:%u/v1 (%s)\n",
            gateway.host, gateway.port, OPENAI_COMPAT_PROFILE);
    rc = openai_gateway_run(&gateway, &stop_requested, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "yvex-openai: %s\n", yvex_error_message(&err));
    return rc == YVEX_OK ? 0 : 1;
invalid:
    usage(stderr);
    return 2;
}
