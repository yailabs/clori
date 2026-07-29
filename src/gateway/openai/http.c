/* Owner: gateway.openai.http.
 * Owns: bounded loopback HTTP/1.1 admission, complete body reads, responses, and SSE framing.
 * Does not own: JSON semantics, OpenAI translation, YVEX sessions, model work, or public networking.
 * Invariants: Content-Length is unique, transfer encoding is refused, and each connection serves one request.
 * Boundary: raw socket bytes become one bounded HTTP request or one typed refusal.
 * Purpose: provide a small request-smuggling-resistant local transport for the compatibility gateway.
 * Inputs: accepted loopback socket descriptors and explicit response bytes.
 * Effects: reads/writes one connection and allocates at most the admitted body extent.
 * Failure: malformed, oversized, timed-out, or incomplete messages publish no partial request. */

#define _POSIX_C_SOURCE 200809L

#include "src/gateway/openai/private.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* Purpose: write one complete response span without SIGPIPE.
 * Inputs: connected descriptor, explicit bytes/count, and error output.
 * Effects: advances the socket until every admitted byte is sent.
 * Failure: reports the first non-interrupt write failure and never claims completion.
 * Boundary: HTTP transport only; it does not interpret JSON or provider facts. */
static int write_all(int fd, const void *bytes, size_t count,
                     yvex_error *err)
{
    const unsigned char *cursor = bytes;
    while (count) {
        ssize_t written = send(fd, cursor, count, MSG_NOSIGNAL);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            yvex_error_set(err, YVEX_ERR_IO, "gateway.http.write",
                           "HTTP client disconnected during response");
            return YVEX_ERR_IO;
        }
        cursor += (size_t)written;
        count -= (size_t)written;
    }
    return YVEX_OK;
}

/* Purpose: parse one bounded decimal Content-Length without coercion. */
static int content_length(const char *text, unsigned long long *value)
{
    unsigned long long parsed = 0u;
    const unsigned char *cursor = (const unsigned char *)text;
    if (!*cursor) return 0;
    while (*cursor) {
        if (*cursor < '0' || *cursor > '9' ||
            parsed > (ULLONG_MAX - (*cursor - '0')) / 10u)
            return 0;
        parsed = parsed * 10u + (*cursor++ - '0');
    }
    *value = parsed;
    return 1;
}

/* Purpose: compare one HTTP header name without accepting prefix aliases. */
static int header_name(const char *line, size_t name_count, const char *name)
{
    size_t index, count = strlen(name);
    if (name_count != count) return 0;
    for (index = 0u; index < count; ++index) {
        unsigned char left = (unsigned char)line[index];
        unsigned char right = (unsigned char)name[index];
        if (left >= 'A' && left <= 'Z') left = (unsigned char)(left + 32u);
        if (left != right) return 0;
    }
    return 1;
}

/* Purpose: admit one exact HTTP/1.1 request line and unique bounded headers.
 * Inputs: mutable complete header bytes and request output.
 * Effects: splits validated lines and records method/path/content length.
 * Failure: refuses ambiguity, transfer encoding, duplicates, or bound excess.
 * Boundary: header grammar only; body bytes are read by the caller. */
static int parse_headers(char *header, size_t header_count,
                         openai_http_request *request,
                         unsigned long long *declared, yvex_error *err)
{
    char *line, *next, *space, *version;
    unsigned int fields = 0u;
    int length_seen = 0;
    if (header_count < 4u || memchr(header, '\0', header_count)) goto malformed;
    line = header;
    next = strstr(line, "\r\n");
    if (!next) goto malformed;
    *next = '\0';
    space = strchr(line, ' ');
    if (!space || space == line || (size_t)(space - line) >= sizeof(request->method))
        goto malformed;
    memcpy(request->method, line, (size_t)(space - line));
    request->method[space - line] = '\0';
    line = space + 1;
    version = strchr(line, ' ');
    if (!version || version == line ||
        (size_t)(version - line) >= sizeof(request->path)) goto malformed;
    memcpy(request->path, line, (size_t)(version - line));
    request->path[version - line] = '\0';
    if (strcmp(version + 1, "HTTP/1.1") != 0 || request->path[0] != '/')
        goto malformed;
    line = next + 2;
    while (*line) {
        char *colon, *value, *end;
        next = strstr(line, "\r\n");
        if (!next) goto malformed;
        *next = '\0';
        if (!*line) break;
        if (++fields > OPENAI_HTTP_HEADER_COUNT_MAX) goto too_large;
        colon = strchr(line, ':');
        if (!colon || colon == line) goto malformed;
        value = colon + 1;
        while (*value == ' ' || *value == '\t') value++;
        end = line + strlen(line);
        while (end > value && (end[-1] == ' ' || end[-1] == '\t')) *--end = '\0';
        if (header_name(line, (size_t)(colon - line), "content-length")) {
            if (length_seen++ || !content_length(value, declared)) goto malformed;
        } else if (header_name(line, (size_t)(colon - line), "transfer-encoding")) {
            yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "gateway.http.headers",
                           "Transfer-Encoding requests are unsupported");
            return YVEX_ERR_UNSUPPORTED;
        }
        line = next + 2;
    }
    if (strcmp(request->method, "POST") == 0 && !length_seen) goto malformed;
    if (*declared > OPENAI_HTTP_BODY_MAX) goto too_large;
    return YVEX_OK;
too_large:
    yvex_error_set(err, YVEX_ERR_BOUNDS, "gateway.http.headers",
                   "HTTP headers or body exceed the gateway limit");
    return YVEX_ERR_BOUNDS;
malformed:
    yvex_error_set(err, YVEX_ERR_FORMAT, "gateway.http.headers",
                   "malformed bounded HTTP/1.1 request");
    return YVEX_ERR_FORMAT;
}

/* Purpose: read one complete bounded request, including bytes already received after the headers.
 * Inputs: connected descriptor, cleared request output, and error output.
 * Effects: allocates and publishes one complete admitted request body.
 * Failure: frees partial storage and refuses timeouts, truncation, or extent overflow.
 * Boundary: one HTTP request per connection with explicit Content-Length. */
int openai_http_read(int fd, openai_http_request *request, yvex_error *err)
{
    char header[OPENAI_HTTP_HEADER_MAX + 1u];
    size_t used = 0u, body_offset;
    unsigned long long declared = 0u, copied;
    int rc;
    if (!request || fd < 0) return YVEX_ERR_INVALID_ARG;
    memset(request, 0, sizeof(*request));
    while (used < OPENAI_HTTP_HEADER_MAX) {
        char *end;
        ssize_t count = recv(fd, header + used, OPENAI_HTTP_HEADER_MAX - used, 0);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) goto incomplete;
        used += (size_t)count;
        header[used] = '\0';
        end = strstr(header, "\r\n\r\n");
        if (!end) continue;
        body_offset = (size_t)(end - header) + 4u;
        rc = parse_headers(header, body_offset, request, &declared, err);
        if (rc != YVEX_OK) return rc;
        if ((unsigned long long)(used - body_offset) > declared) goto malformed;
        if (declared) {
            request->body = malloc((size_t)declared);
            if (!request->body) {
                yvex_error_set(err, YVEX_ERR_NOMEM, "gateway.http.body",
                               "HTTP body allocation failed");
                return YVEX_ERR_NOMEM;
            }
        }
        copied = (unsigned long long)(used - body_offset);
        if (copied) memcpy(request->body, header + body_offset, (size_t)copied);
        while (copied < declared) {
            ssize_t body_read = recv(fd, request->body + copied,
                                     (size_t)(declared - copied), 0);
            if (body_read < 0 && errno == EINTR) continue;
            if (body_read <= 0) goto incomplete;
            copied += (unsigned long long)body_read;
        }
        request->body_count = declared;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    yvex_error_set(err, YVEX_ERR_BOUNDS, "gateway.http.headers",
                   "HTTP header extent exceeds the gateway limit");
    return YVEX_ERR_BOUNDS;
incomplete:
    openai_http_request_clear(request);
    yvex_error_set(err, YVEX_ERR_IO, "gateway.http.read",
                   "HTTP client closed before the request completed");
    return YVEX_ERR_IO;
malformed:
    openai_http_request_clear(request);
    yvex_error_set(err, YVEX_ERR_FORMAT, "gateway.http.body",
                   "bytes beyond Content-Length are refused");
    return YVEX_ERR_FORMAT;
}

/* Purpose: release one admitted request body and erase parser facts.
 * Inputs: one request that may own body storage.
 * Effects: frees the body and clears all parsed fields.
 * Failure: none; null and cleared requests are accepted.
 * Boundary: does not close the associated socket descriptor. */
void openai_http_request_clear(openai_http_request *request)
{
    if (!request) return;
    free(request->body);
    memset(request, 0, sizeof(*request));
}

/* Purpose: map one admitted status code to a fixed HTTP reason. */
static const char *status_reason(int status)
{
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 409: return "Conflict";
    case 413: return "Content Too Large";
    case 422: return "Unprocessable Content";
    case 429: return "Too Many Requests";
    case 499: return "Client Closed Request";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout";
    default: return "Error";
    }
}

/* Purpose: send one complete JSON response with explicit close semantics.
 * Inputs: client descriptor, admitted status, JSON bytes/count, and error output.
 * Effects: writes headers and the complete response body.
 * Failure: returns the first transport error without substituting another status.
 * Boundary: frames already-rendered JSON and performs no semantic translation. */
int openai_http_json(int fd, int status, const unsigned char *body,
                     unsigned long long count, yvex_error *err)
{
    char header[512];
    int length;
    if ((!body && count) || count > SIZE_MAX) return YVEX_ERR_INVALID_ARG;
    length = snprintf(header, sizeof(header),
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %llu\r\n"
                      "Connection: close\r\n"
                      "X-YVEX-OpenAI-Profile: %s\r\n\r\n",
                      status, status_reason(status), count, OPENAI_COMPAT_PROFILE);
    if (length < 0 || (size_t)length >= sizeof(header)) return YVEX_ERR_BOUNDS;
    if (write_all(fd, header, (size_t)length, err) != YVEX_OK)
        return yvex_error_code(err);
    return write_all(fd, body, (size_t)count, err);
}

/* Purpose: start one non-buffered local SSE response.
 * Inputs: client descriptor and error output.
 * Effects: commits one HTTP 200 event-stream header block.
 * Failure: reports transport failure before any event claim.
 * Boundary: SSE framing only; stream event schemas remain renderer-owned. */
int openai_http_sse_begin(int fd, yvex_error *err)
{
    static const char header[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "X-Accel-Buffering: no\r\n"
        "X-YVEX-OpenAI-Profile: " OPENAI_COMPAT_PROFILE "\r\n\r\n";
    return write_all(fd, header, sizeof(header) - 1u, err);
}

/* Purpose: write one exact SSE record without mixing terminal prose.
 * Inputs: descriptor, optional event name, explicit JSON bytes/count, and error output.
 * Effects: writes one complete event/data record in order.
 * Failure: stops on the first socket failure and emits no success sentinel.
 * Boundary: consumes already-rendered event JSON without inspecting model content. */
int openai_http_sse_event(int fd, const char *event,
                          const unsigned char *json,
                          unsigned long long count, yvex_error *err)
{
    if (event && *event) {
        if (write_all(fd, "event: ", 7u, err) != YVEX_OK ||
            write_all(fd, event, strlen(event), err) != YVEX_OK ||
            write_all(fd, "\n", 1u, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    if (write_all(fd, "data: ", 6u, err) != YVEX_OK ||
        write_all(fd, json, (size_t)count, err) != YVEX_OK ||
        write_all(fd, "\n\n", 2u, err) != YVEX_OK)
        return yvex_error_code(err);
    return YVEX_OK;
}

/* Purpose: terminate a Chat Completions stream with the specified protocol sentinel.
 * Inputs: descriptor and error output.
 * Effects: writes exactly one data [DONE] SSE record.
 * Failure: reports transport failure and never retries past caller ownership.
 * Boundary: used only where the admitted endpoint requires this sentinel. */
int openai_http_sse_done(int fd, yvex_error *err)
{
    return write_all(fd, "data: [DONE]\n\n", 14u, err);
}
