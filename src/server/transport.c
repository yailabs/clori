/*
 * Own the local Unix transport lifecycle around the stateless protocol codec.
 *
 * Framing, socket privacy, timeouts, and connection ownership stay here so the protocol owner
 * remains a deterministic byte codec with no file-descriptor lifecycle.
 */
#define _POSIX_C_SOURCE 200809L
#include "src/server/private.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#define FRAME_HEADER_BYTES 12u
#define FRAME_KIND_REQUEST 1u
#define FRAME_KIND_MESSAGE 2u

struct yvex_client {
    int fd;
};

static int transport_refuse(yvex_error *err, yvex_status status,
                            const char *reason)
{
    yvex_error_set(err, status, "server.protocol", reason);
    return status;
}

static void put_u16(unsigned char *out, uint16_t value)
{
    out[0] = (unsigned char)(value >> 8u);
    out[1] = (unsigned char)value;
}

static void put_u32(unsigned char *out, uint32_t value)
{
    out[0] = (unsigned char)(value >> 24u);
    out[1] = (unsigned char)(value >> 16u);
    out[2] = (unsigned char)(value >> 8u);
    out[3] = (unsigned char)value;
}

static uint16_t get_u16(const unsigned char *input)
{
    return (uint16_t)(((uint16_t)input[0] << 8u) | input[1]);
}

static uint32_t get_u32(const unsigned char *input)
{
    return ((uint32_t)input[0] << 24u) | ((uint32_t)input[1] << 16u) |
           ((uint32_t)input[2] << 8u) | input[3];
}

static int transfer_all(int fd, void *buffer, size_t count, int writing,
                        yvex_error *err)
{
    unsigned char *bytes = buffer;
    size_t offset = 0u;
    while (offset < count) {
        ssize_t moved = writing ? send(fd, bytes + offset, count - offset, MSG_NOSIGNAL)
                                : recv(fd, bytes + offset, count - offset, 0);
        if (moved < 0 && errno == EINTR) continue;
        if (moved < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            return transport_refuse(err, YVEX_ERR_TIMEOUT,
                                    "local protocol operation timed out");
        if (moved <= 0)
            return transport_refuse(err, YVEX_ERR_IO,
                                    writing ? "local socket write failed"
                                            : "local socket closed during frame read");
        offset += (size_t)moved;
    }
    return YVEX_OK;
}

int yvex_client_timeout_set(yvex_client *client,
                            unsigned long long milliseconds,
                            yvex_error *err)
{
    struct timeval timeout;
    if (!client || client->fd < 0 || milliseconds > 86400000u)
        return transport_refuse(err, YVEX_ERR_INVALID_ARG,
                                "connected client and bounded timeout are required");
    timeout.tv_sec = (time_t)(milliseconds / 1000u);
    timeout.tv_usec = (suseconds_t)((milliseconds % 1000u) * 1000u);
    if (setsockopt(client->fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                   sizeof(timeout)) != 0 ||
        setsockopt(client->fd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                   sizeof(timeout)) != 0)
        return transport_refuse(err, YVEX_ERR_IO,
                                "local protocol timeout configuration failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int frame_send(int fd, unsigned int kind, const unsigned char *payload,
                      unsigned long long count, yvex_error *err)
{
    unsigned char header[FRAME_HEADER_BYTES] = {'Y', 'V', 'X', 'P'};
    if (count > YVEX_SERVER_FRAME_MAX_BYTES)
        return transport_refuse(err, YVEX_ERR_BOUNDS,
                                "local protocol frame exceeds maximum size");
    put_u16(header + 4u, YVEX_LOCAL_PROTOCOL_VERSION);
    put_u16(header + 6u, (uint16_t)kind);
    put_u32(header + 8u, (uint32_t)count);
    if (transfer_all(fd, header, sizeof(header), 1, err) != YVEX_OK)
        return yvex_error_code(err);
    return count ? transfer_all(fd, (void *)payload, (size_t)count, 1, err)
                 : YVEX_OK;
}

static int frame_receive(int fd, unsigned int expected_kind,
                         unsigned char **payload, unsigned long long *count,
                         yvex_error *err)
{
    unsigned char header[FRAME_HEADER_BYTES], *bytes = NULL;
    uint32_t length;
    int rc;
    *payload = NULL;
    *count = 0u;
    rc = transfer_all(fd, header, sizeof(header), 0, err);
    if (rc != YVEX_OK) return rc;
    length = get_u32(header + 8u);
    if (memcmp(header, "YVXP", 4u) != 0 ||
        get_u16(header + 6u) != expected_kind ||
        length > YVEX_SERVER_FRAME_MAX_BYTES)
        return transport_refuse(err, YVEX_ERR_FORMAT,
                                "local protocol frame header is invalid");
    if (get_u16(header + 4u) != YVEX_LOCAL_PROTOCOL_VERSION)
        return transport_refuse(
            err, YVEX_ERR_FORMAT,
            "local protocol version is incompatible; version 8 is required");
    if (length) {
        bytes = malloc(length);
        if (!bytes)
            return transport_refuse(err, YVEX_ERR_NOMEM,
                                    "local protocol frame allocation failed");
        rc = transfer_all(fd, bytes, length, 0, err);
        if (rc != YVEX_OK) {
            free(bytes);
            return rc;
        }
    }
    *payload = bytes;
    *count = length;
    return YVEX_OK;
}

int yvex_server_socket_path(char output[YVEX_SERVER_SOCKET_PATH_CAP],
                            yvex_error *err)
{
    const char *runtime = getenv("XDG_RUNTIME_DIR");
    int length;
    if (!output)
        return transport_refuse(err, YVEX_ERR_INVALID_ARG,
                                "socket path output is required");
    if (runtime && runtime[0] == '/')
        length = snprintf(output, YVEX_SERVER_SOCKET_PATH_CAP,
                          "%s/yvex/yvexd.sock", runtime);
    else
        length = snprintf(output, YVEX_SERVER_SOCKET_PATH_CAP,
                          "/tmp/yvex-%lu/yvexd.sock", (unsigned long)getuid());
    if (length < 0 || length >= (int)YVEX_SERVER_SOCKET_PATH_CAP)
        return transport_refuse(err, YVEX_ERR_BOUNDS,
                                "canonical socket path exceeds its bound");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_client_connect(yvex_client **out, const char *socket_path,
                        yvex_error *err)
{
    yvex_client_request handshake;
    yvex_client_message response;
    yvex_provider_sampling sampling;
    yvex_client *client;
    struct sockaddr_un address;
    struct stat info;
    char canonical[YVEX_SERVER_SOCKET_PATH_CAP];
    const char *path = socket_path;
    int fd;
    if (out) *out = NULL;
    if (!out)
        return transport_refuse(err, YVEX_ERR_INVALID_ARG,
                                "client output is required");
    if (!path) {
        if (yvex_server_socket_path(canonical, err) != YVEX_OK)
            return yvex_error_code(err);
        path = canonical;
    }
    if (strlen(path) >= sizeof(address.sun_path) || lstat(path, &info) != 0 ||
        !S_ISSOCK(info.st_mode) || info.st_uid != getuid() ||
        (info.st_mode & 0077u) != 0u)
        return transport_refuse(
            err, YVEX_ERR_IO,
            "local runtime socket is absent or not private to this user");
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return transport_refuse(err, YVEX_ERR_IO,
                                "local client socket creation failed");
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path) + 1u);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        (void)close(fd);
        return transport_refuse(err, YVEX_ERR_IO,
                                "cannot connect to the local YVEX runtime");
    }
    client = calloc(1u, sizeof(*client));
    if (!client) {
        (void)close(fd);
        return transport_refuse(err, YVEX_ERR_NOMEM,
                                "local client allocation failed");
    }
    client->fd = fd;
    if (yvex_client_timeout_set(client, 30000u, err) != YVEX_OK) {
        (void)close(client->fd);
        free(client);
        return yvex_error_code(err);
    }
    memset(&handshake, 0, sizeof(handshake));
    yvex_provider_sampling_default(&sampling);
    handshake.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    handshake.operation = YVEX_CLIENT_OP_HANDSHAKE;
    handshake.temperature = sampling.temperature;
    handshake.top_p = sampling.top_p;
    handshake.typical_p = sampling.typical_p;
    if (yvex_client_send(client, &handshake, err) != YVEX_OK ||
        yvex_client_receive(client, &response, err) != YVEX_OK ||
        response.kind != YVEX_CLIENT_MESSAGE_ACK ||
        response.status != YVEX_OK || strcmp(response.reason, "protocol-v9") != 0) {
        (void)close(client->fd);
        memset(client, 0, sizeof(*client));
        free(client);
        if (yvex_error_code(err) == YVEX_OK)
            yvex_error_set(err, YVEX_ERR_FORMAT, "server.protocol.handshake",
                           "daemon did not admit local protocol version 8");
        return yvex_error_code(err);
    }
    if (yvex_client_timeout_set(client, 0u, err) != YVEX_OK) {
        (void)close(client->fd);
        free(client);
        return yvex_error_code(err);
    }
    *out = client;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_client_send(yvex_client *client, const yvex_client_request *request,
                     yvex_error *err)
{
    unsigned char *payload;
    unsigned long long capacity, count;
    int rc;
    if (!client || client->fd < 0 || !request)
        return transport_refuse(err, YVEX_ERR_INVALID_ARG,
                                "connected client and request are required");
    capacity = request->provider_request ? YVEX_SERVER_FRAME_MAX_BYTES
                                         : request->prompt_bytes + 512u;
    if (capacity > YVEX_SERVER_FRAME_MAX_BYTES)
        return transport_refuse(err, YVEX_ERR_BOUNDS,
                                "client request exceeds frame capacity");
    payload = malloc((size_t)capacity);
    if (!payload)
        return transport_refuse(err, YVEX_ERR_NOMEM,
                                "client request frame allocation failed");
    rc = yvex_protocol_request_encode(request, payload, capacity, &count, err);
    if (rc == YVEX_OK)
        rc = frame_send(client->fd, FRAME_KIND_REQUEST, payload, count, err);
    free(payload);
    return rc;
}

int yvex_client_receive(yvex_client *client, yvex_client_message *message,
                        yvex_error *err)
{
    unsigned char *payload;
    unsigned long long count;
    int rc;
    if (!client || client->fd < 0 || !message)
        return transport_refuse(err, YVEX_ERR_INVALID_ARG,
                                "connected client and message output are required");
    rc = frame_receive(client->fd, FRAME_KIND_MESSAGE, &payload, &count, err);
    if (rc == YVEX_OK)
        rc = yvex_protocol_message_decode(payload, count, message, err);
    free(payload);
    return rc;
}

void yvex_client_close(yvex_client **client)
{
    if (!client || !*client) return;
    if ((*client)->fd >= 0) (void)close((*client)->fd);
    memset(*client, 0, sizeof(**client));
    free(*client);
    *client = NULL;
}

int yvex_server_protocol_receive(int fd, yvex_client_request *request,
                                 unsigned char **owned_prompt,
                                 yvex_provider_request **owned_provider,
                                 yvex_error *err)
{
    unsigned char *payload;
    unsigned long long count;
    int rc = frame_receive(fd, FRAME_KIND_REQUEST, &payload, &count, err);
    if (rc == YVEX_OK)
        rc = yvex_protocol_request_decode(payload, count, request,
                                          owned_prompt, owned_provider, err);
    free(payload);
    return rc;
}

int yvex_server_protocol_send(int fd, const yvex_client_message *message,
                              yvex_error *err)
{
    unsigned char payload[16384];
    unsigned long long count;
    int rc = yvex_protocol_message_encode(message, payload, sizeof(payload),
                                          &count, err);
    if (rc == YVEX_OK)
        rc = frame_send(fd, FRAME_KIND_MESSAGE, payload, count, err);
    return rc;
}
