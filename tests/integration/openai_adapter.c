/*
 * Test-only lifecycle driver for the production in-process OpenAI adapter. It supplies no HTTP,
 * JSON, provider, protocol, or runtime semantics of its own.
 */
#define _POSIX_C_SOURCE 200809L
#include "src/server/openai/private.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *output)
{
    fprintf(output,
            "Usage: openai_adapter [--host 127.0.0.1] [--port 8001] "
            "[--yvex-socket PATH] [--timeout-ms 600000]\n");
}

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

static int timeout_parse(const char *text, unsigned long long *timeout)
{
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno || !end || *end || value < 100u || value > 86400000u)
        return 0;
    *timeout = value;
    return 1;
}

int main(int argc, char **argv)
{
    server_openai_listener *listener = NULL;
    server_openai_options options;
    yvex_error err;
    sigset_t signals;
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    const char *host = "127.0.0.1";
    int index, signal_number, rc;

    memset(&options, 0, sizeof(options));
    if (yvex_server_socket_path(socket_path, &err) != YVEX_OK) {
        fprintf(stderr, "openai_adapter: %s\n", yvex_error_message(&err));
        return 1;
    }
    options.yvex_socket = socket_path;
    options.port = 8001u;
    options.timeout_ms = 600000u;
    options.maximum_connections = 8ull;
    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--help") == 0) {
            usage(stdout);
            return 0;
        }
        if (strcmp(argv[index], "--host") == 0 && index + 1 < argc) {
            host = argv[++index];
        } else if (strcmp(argv[index], "--port") == 0 && index + 1 < argc) {
            if (!port_parse(argv[++index], &options.port)) goto invalid;
        } else if (strcmp(argv[index], "--yvex-socket") == 0 &&
                   index + 1 < argc) {
            if (strlen(argv[++index]) >= sizeof(socket_path)) goto invalid;
            strcpy(socket_path, argv[index]);
        } else if (strcmp(argv[index], "--timeout-ms") == 0 &&
                   index + 1 < argc) {
            if (!timeout_parse(argv[++index], &options.timeout_ms))
                goto invalid;
        } else {
            goto invalid;
        }
    }
    if (strcmp(host, "127.0.0.1") != 0) {
        fprintf(stderr, "openai_adapter: non-loopback bind refused\n");
        return 1;
    }
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    if (pthread_sigmask(SIG_BLOCK, &signals, NULL) != 0) return 1;
    rc = yvex_server_openai_prepare(&listener, &options, NULL, &err);
    if (rc == YVEX_OK) rc = yvex_server_openai_start(listener, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "openai_adapter: %s\n", yvex_error_message(&err));
        yvex_server_openai_close(&listener);
        return 1;
    }
    yvex_server_openai_activate(listener);
    fprintf(stderr, "openai_adapter: test fixture listening on 127.0.0.1:%u\n",
            options.port);
    if (sigwait(&signals, &signal_number) != 0) rc = YVEX_ERR_STATE;
    (void)signal_number;
    yvex_server_openai_request_stop(listener);
    if (yvex_server_openai_finish(listener, &err) != YVEX_OK) rc = YVEX_ERR_STATE;
    yvex_server_openai_close(&listener);
    return rc == YVEX_OK ? 0 : 1;

invalid:
    usage(stderr);
    return 2;
}
