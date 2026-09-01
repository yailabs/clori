/*
 * Render typed model stream channels as bounded line-oriented terminal text.
 *
 * This owner changes presentation only. Canonical response bytes and protocol
 * channel identity remain untouched.
 */
#include "src/cli/io/private.h"

#include <ctype.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define STREAM_PREFERRED_PROSE_WIDTH 112u

static const char *stream_style_code(const yvex_cli_stream_renderer *renderer,
                                     yvex_cli_stream_style style)
{
    switch (style) {
    case YVEX_CLI_STREAM_STYLE_DIM: return renderer->style.dim;
    case YVEX_CLI_STREAM_STYLE_ACCENT: return renderer->style.accent;
    case YVEX_CLI_STREAM_STYLE_STRONG: return renderer->style.strong;
    default: return "";
    }
}

static int stream_style_set(yvex_cli_stream_renderer *renderer,
                            yvex_cli_stream_style style)
{
    if (renderer->active_style == style) return 1;
    if (renderer->style.reset[0] &&
        fputs(renderer->style.reset, renderer->output) == EOF)
        return 0;
    if (stream_style_code(renderer, style)[0] &&
        fputs(stream_style_code(renderer, style), renderer->output) == EOF)
        return 0;
    renderer->active_style = style;
    return 1;
}

static yvex_cli_stream_style stream_channel_style(
    const yvex_cli_stream_renderer *renderer, yvex_cli_stream_style requested)
{
    if (renderer->channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING)
        return YVEX_CLI_STREAM_STYLE_DIM;
    if (renderer->channel == YVEX_CLIENT_STREAM_TOOL_CALL ||
        renderer->channel == YVEX_CLIENT_STREAM_TOOL_RESULT)
        return YVEX_CLI_STREAM_STYLE_ACCENT;
    return requested;
}

static int stream_output(yvex_cli_stream_renderer *renderer,
                         const unsigned char *bytes, size_t count,
                         yvex_cli_stream_style style)
{
    if (!stream_style_set(renderer, stream_channel_style(renderer, style)) ||
        (count && fwrite(bytes, 1u, count, renderer->output) != count))
        return 0;
    if (count) {
        renderer->wrote_bytes = 1;
        renderer->last_newline = bytes[count - 1u] == '\n';
    }
    return 1;
}

static int stream_ascii(yvex_cli_stream_renderer *renderer, const char *text,
                        yvex_cli_stream_style style)
{
    return stream_output(renderer, (const unsigned char *)text, strlen(text),
                         style);
}

static int stream_newline(yvex_cli_stream_renderer *renderer)
{
    if (!stream_style_set(renderer, stream_channel_style(
                                       renderer, YVEX_CLI_STREAM_STYLE_NORMAL)) ||
        fputc('\n', renderer->output) == EOF)
        return 0;
    renderer->column = 0u;
    renderer->wrote_bytes = renderer->last_newline = 1;
    return 1;
}

static unsigned int stream_terminal_columns(FILE *output)
{
    struct winsize size;
    const char *configured = getenv("COLUMNS");
    char *end = NULL;
    unsigned long value;
    int fd = fileno(output);
    if (fd >= 0 && ioctl(fd, TIOCGWINSZ, &size) == 0 && size.ws_col)
        return size.ws_col;
    if (!configured || !configured[0]) return STREAM_PREFERRED_PROSE_WIDTH;
    value = strtoul(configured, &end, 10);
    if (!end || *end || value > UINT_MAX || value < 1u)
        return STREAM_PREFERRED_PROSE_WIDTH;
    return (unsigned int)value;
}

static void stream_width_refresh(yvex_cli_stream_renderer *renderer)
{
    unsigned int columns = stream_terminal_columns(renderer->output);
    renderer->prose_width =
        columns < STREAM_PREFERRED_PROSE_WIDTH
            ? columns : STREAM_PREFERRED_PROSE_WIDTH;
}

static int stream_escape_byte(yvex_cli_stream_renderer *renderer,
                              unsigned char byte,
                              yvex_cli_stream_style style)
{
    char escaped[5];
    int count = snprintf(escaped, sizeof(escaped), "\\x%02x", byte);
    if (count != 4 ||
        !stream_output(renderer, (const unsigned char *)escaped, 4u, style))
        return 0;
    renderer->column += 4u;
    return 1;
}

static unsigned int stream_utf8_extent(unsigned char byte)
{
    if (byte >= 0xc2u && byte <= 0xdfu) return 2u;
    if (byte >= 0xe0u && byte <= 0xefu) return 3u;
    if (byte >= 0xf0u && byte <= 0xf4u) return 4u;
    return 0u;
}

static int stream_utf8_valid(const unsigned char *bytes, size_t available,
                             unsigned int extent)
{
    unsigned int index;
    if (!extent || available < extent) return 0;
    for (index = 1u; index < extent; ++index)
        if ((bytes[index] & 0xc0u) != 0x80u) return 0;
    if (extent == 3u && bytes[0] == 0xe0u && bytes[1] < 0xa0u) return 0;
    if (extent == 3u && bytes[0] == 0xedu && bytes[1] >= 0xa0u) return 0;
    if (extent == 4u && bytes[0] == 0xf0u && bytes[1] < 0x90u) return 0;
    if (extent == 4u && bytes[0] == 0xf4u && bytes[1] >= 0x90u) return 0;
    return 1;
}

static size_t stream_visible_word(const unsigned char *bytes, size_t count)
{
    size_t index = 0u, visible = 0u;
    while (index < count && bytes[index] != ' ' && bytes[index] != '\t') {
        unsigned int extent;
        if ((bytes[index] == '*' || bytes[index] == '_' ||
             bytes[index] == '`') &&
            memchr(bytes + index + 1u, bytes[index], count - index - 1u)) {
            index++;
            continue;
        }
        extent = stream_utf8_extent(bytes[index]);
        if (extent && stream_utf8_valid(bytes + index, count - index, extent))
            index += extent;
        else
            index++;
        visible++;
    }
    return visible;
}

static int stream_pair_remains(const unsigned char *bytes, size_t count,
                               unsigned char first, unsigned char second)
{
    size_t index;
    for (index = 0u; index + 1u < count; ++index)
        if (bytes[index] == first && bytes[index + 1u] == second) return 1;
    return 0;
}

static int stream_prefix(yvex_cli_stream_renderer *renderer,
                         const char *prefix, yvex_cli_stream_style style)
{
    if (renderer->channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING) {
        if (!stream_ascii(renderer, "│ ", YVEX_CLI_STREAM_STYLE_DIM)) return 0;
        renderer->column += 2u;
    }
    if (prefix && prefix[0]) {
        if (!stream_ascii(renderer, prefix, style)) return 0;
        renderer->column += (unsigned int)strlen(prefix);
    }
    return 1;
}

static int stream_continuation(yvex_cli_stream_renderer *renderer,
                               unsigned int indent)
{
    unsigned int index;
    if (!stream_newline(renderer) ||
        !stream_prefix(renderer, NULL, YVEX_CLI_STREAM_STYLE_NORMAL))
        return 0;
    for (index = 0u; index < indent; ++index)
        if (!stream_ascii(renderer, " ", YVEX_CLI_STREAM_STYLE_NORMAL))
            return 0;
    renderer->column += indent;
    return 1;
}

static int stream_text_unit(yvex_cli_stream_renderer *renderer,
                            const unsigned char *bytes, size_t available,
                            yvex_cli_stream_style style, size_t *consumed)
{
    unsigned int extent;
    static const unsigned char replacement[] = {0xefu, 0xbfu, 0xbdu};
    *consumed = 1u;
    if (bytes[0] < 0x20u || bytes[0] == 0x7fu) {
        if (bytes[0] == '\t') {
            if (!stream_output(renderer, bytes, 1u, style)) return 0;
            renderer->column += 4u;
            return 1;
        }
        return stream_escape_byte(renderer, bytes[0], style);
    }
    if (bytes[0] < 0x80u) {
        if (!stream_output(renderer, bytes, 1u, style)) return 0;
        renderer->column++;
        return 1;
    }
    extent = stream_utf8_extent(bytes[0]);
    if (!stream_utf8_valid(bytes, available, extent)) {
        if (!stream_output(renderer, replacement, sizeof(replacement), style))
            return 0;
        renderer->column++;
        return 1;
    }
    if (!stream_output(renderer, bytes, extent, style)) return 0;
    renderer->column++;
    *consumed = extent;
    return 1;
}

static int stream_inline(yvex_cli_stream_renderer *renderer,
                         const unsigned char *bytes, size_t count,
                         yvex_cli_stream_style base, unsigned int indent)
{
    size_t index = 0u;
    int strong = 0, emphasis = 0, code = 0;
    while (index < count) {
        yvex_cli_stream_style style;
        size_t consumed;
        if (bytes[index] == ' ' || bytes[index] == '\t') {
            size_t next = index;
            while (next < count &&
                   (bytes[next] == ' ' || bytes[next] == '\t')) next++;
            if (next < count &&
                renderer->column + 1u +
                        stream_visible_word(bytes + next, count - next) >
                    renderer->prose_width &&
                renderer->column > indent) {
                if (!stream_continuation(renderer, indent)) return 0;
            } else if (!stream_ascii(renderer, " ", base)) {
                return 0;
            } else {
                renderer->column++;
            }
            index = next;
            continue;
        }
        if (index + 1u < count &&
            ((bytes[index] == '*' && bytes[index + 1u] == '*') ||
             (bytes[index] == '_' && bytes[index + 1u] == '_'))) {
            if (strong || stream_pair_remains(
                              bytes + index + 2u, count - index - 2u,
                              bytes[index], bytes[index + 1u])) {
                strong = !strong;
                index += 2u;
                continue;
            }
        }
        if (bytes[index] == '*' || bytes[index] == '_') {
            if (emphasis || memchr(bytes + index + 1u, bytes[index],
                                   count - index - 1u)) {
                emphasis = !emphasis;
                index++;
                continue;
            }
        }
        if (bytes[index] == '`') {
            if (code || memchr(bytes + index + 1u, '`', count - index - 1u)) {
                code = !code;
                index++;
                continue;
            }
        }
        style = code ? YVEX_CLI_STREAM_STYLE_ACCENT
                     : (strong || emphasis ? YVEX_CLI_STREAM_STYLE_STRONG
                                           : base);
        if (!stream_text_unit(renderer, bytes + index, count - index,
                              style, &consumed))
            return 0;
        index += consumed;
    }
    return stream_style_set(renderer, stream_channel_style(renderer, base));
}

static size_t stream_fence_offset(const unsigned char *bytes, size_t count)
{
    size_t offset = 0u;
    while (offset < count && offset < 3u && bytes[offset] == ' ') offset++;
    return count - offset >= 3u && bytes[offset] == '`' &&
                   bytes[offset + 1u] == '`' && bytes[offset + 2u] == '`'
               ? offset : SIZE_MAX;
}

static int stream_code_line(yvex_cli_stream_renderer *renderer,
                            const unsigned char *bytes, size_t count)
{
    size_t index = 0u;
    if (!stream_prefix(renderer, "  ", YVEX_CLI_STREAM_STYLE_ACCENT)) return 0;
    while (index < count) {
        size_t consumed;
        if (!stream_text_unit(renderer, bytes + index, count - index,
                              YVEX_CLI_STREAM_STYLE_ACCENT, &consumed))
            return 0;
        index += consumed;
    }
    return 1;
}

static int stream_render_line(yvex_cli_stream_renderer *renderer,
                              const unsigned char *bytes, size_t count)
{
    char ordered[32];
    size_t offset = 0u, fence, digits;
    const char *prefix = NULL;
    yvex_cli_stream_style base = YVEX_CLI_STREAM_STYLE_NORMAL;
    stream_width_refresh(renderer);
    fence = stream_fence_offset(bytes, count);
    if (fence != SIZE_MAX) {
        size_t language = fence + 3u;
        while (language < count && bytes[language] == ' ') language++;
        if (renderer->in_fence) {
            renderer->in_fence = 0;
            return 1;
        }
        renderer->in_fence = 1;
        if (!stream_prefix(renderer, NULL, YVEX_CLI_STREAM_STYLE_DIM) ||
            !stream_ascii(renderer, "code", YVEX_CLI_STREAM_STYLE_DIM))
            return 0;
        renderer->column += 4u;
        if (language < count) {
            if (!stream_ascii(renderer, " · ", YVEX_CLI_STREAM_STYLE_DIM) ||
                !stream_output(renderer, bytes + language, count - language,
                               YVEX_CLI_STREAM_STYLE_DIM))
                return 0;
            renderer->column += 3u + (unsigned int)(count - language);
        }
        return 1;
    }
    if (renderer->in_fence) return stream_code_line(renderer, bytes, count);
    if (!count) return 1;
    while (offset < count && bytes[offset] == '#') offset++;
    if (offset && offset <= 6u && offset < count && bytes[offset] == ' ') {
        base = YVEX_CLI_STREAM_STYLE_STRONG;
        while (offset < count && bytes[offset] == ' ') offset++;
    } else {
        offset = 0u;
    }
    if (count - offset >= 2u &&
        (bytes[offset] == '-' || bytes[offset] == '*' ||
         bytes[offset] == '+') && bytes[offset + 1u] == ' ') {
        prefix = "• ";
        offset += 2u;
    } else if (count - offset >= 2u && bytes[offset] == '>' &&
               bytes[offset + 1u] == ' ') {
        prefix = "│ ";
        base = YVEX_CLI_STREAM_STYLE_DIM;
        offset += 2u;
    } else {
        digits = offset;
        while (digits < count && isdigit(bytes[digits])) digits++;
        if (digits > offset && digits + 1u < count && bytes[digits] == '.' &&
            bytes[digits + 1u] == ' ') {
            size_t length = digits - offset;
            if (length + 3u < sizeof(ordered)) {
                memcpy(ordered, bytes + offset, length);
                ordered[length] = '.';
                ordered[length + 1u] = ' ';
                ordered[length + 2u] = '\0';
                prefix = ordered;
                offset = digits + 2u;
            }
        }
    }
    if (!stream_prefix(renderer, prefix, YVEX_CLI_STREAM_STYLE_ACCENT))
        return 0;
    return stream_inline(renderer, bytes + offset, count - offset, base,
                         renderer->column);
}

static int stream_flush_line(yvex_cli_stream_renderer *renderer, int newline)
{
    int ok = stream_render_line(renderer, renderer->line, renderer->line_count);
    renderer->line_count = 0u;
    return ok && (!newline || stream_newline(renderer));
}

static const char *stream_channel_label(yvex_client_stream_channel channel)
{
    if (channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING) return "reasoning";
    if (channel == YVEX_CLIENT_STREAM_FINAL_TEXT) return "answer";
    if (channel == YVEX_CLIENT_STREAM_TOOL_CALL) return "tool call";
    if (channel == YVEX_CLIENT_STREAM_TOOL_RESULT) return "tool result";
    if (channel == YVEX_CLIENT_STREAM_ERROR) return "error";
    return "output";
}

static int stream_channel_begin(yvex_cli_stream_renderer *renderer)
{
    yvex_cli_stream_style style =
        renderer->channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING
            ? YVEX_CLI_STREAM_STYLE_DIM : YVEX_CLI_STREAM_STYLE_STRONG;
    if (renderer->channel_announced) return 1;
    if (renderer->wrote_bytes && !renderer->last_newline &&
        !stream_newline(renderer))
        return 0;
    if (renderer->wrote_bytes && !stream_newline(renderer)) return 0;
    if (!stream_ascii(renderer, stream_channel_label(renderer->channel), style) ||
        !stream_newline(renderer))
        return 0;
    renderer->channel_announced = 1;
    return 1;
}

void yvex_cli_stream_renderer_open(yvex_cli_stream_renderer *renderer,
                                   FILE *output, int enhanced)
{
    if (!renderer) return;
    memset(renderer, 0, sizeof(*renderer));
    renderer->output = output ? output : stdout;
    renderer->enhanced = enhanced != 0;
    renderer->last_newline = 1;
    renderer->channel = YVEX_CLIENT_STREAM_FINAL_TEXT;
    yvex_cli_terminal_style_get(renderer->output, &renderer->style);
    stream_width_refresh(renderer);
}

int yvex_cli_stream_renderer_write(yvex_cli_stream_renderer *renderer,
                                   yvex_client_stream_channel channel,
                                   const unsigned char *bytes,
                                   unsigned long long count)
{
    unsigned long long index;
    if (!renderer || !renderer->output || (!bytes && count) || count > SIZE_MAX)
        return YVEX_ERR_INVALID_ARG;
    if (!renderer->enhanced) {
        if (count && fwrite(bytes, 1u, (size_t)count, renderer->output) != count)
            return YVEX_ERR_IO;
        if (count) {
            renderer->wrote_bytes = 1;
            renderer->last_newline = bytes[count - 1u] == '\n';
        }
        return YVEX_OK;
    }
    if (channel != renderer->channel) {
        if (renderer->line_count && !stream_flush_line(renderer, 1))
            return YVEX_ERR_IO;
        renderer->channel = channel;
        renderer->channel_announced = 0;
    }
    if (count && !stream_channel_begin(renderer)) return YVEX_ERR_IO;
    for (index = 0u; index < count; ++index) {
        unsigned char byte = bytes[index];
        if (renderer->pending_cr) {
            renderer->pending_cr = 0;
            if (!stream_flush_line(renderer, 1)) return YVEX_ERR_IO;
            if (byte == '\n') continue;
        }
        if (byte == '\r') {
            renderer->pending_cr = 1;
            continue;
        }
        if (byte == '\n') {
            if (!stream_flush_line(renderer, 1)) return YVEX_ERR_IO;
            continue;
        }
        if (renderer->line_count == sizeof(renderer->line) &&
            !stream_flush_line(renderer, 1))
            return YVEX_ERR_IO;
        renderer->line[renderer->line_count++] = byte;
    }
    return ferror(renderer->output) ? YVEX_ERR_IO : YVEX_OK;
}

int yvex_cli_stream_renderer_finish(yvex_cli_stream_renderer *renderer,
                                    int separate_terminal_status)
{
    int ok = 1;
    if (!renderer || !renderer->output) return YVEX_ERR_INVALID_ARG;
    if (!renderer->enhanced) {
        if (separate_terminal_status && renderer->wrote_bytes &&
            !renderer->last_newline)
            renderer->last_newline = fputc('\n', renderer->output) != EOF;
        return !ferror(renderer->output) ? YVEX_OK : YVEX_ERR_IO;
    }
    if (renderer->pending_cr || renderer->line_count) {
        renderer->pending_cr = 0;
        ok = stream_flush_line(renderer, separate_terminal_status);
    }
    if (ok) ok = stream_style_set(renderer, YVEX_CLI_STREAM_STYLE_NORMAL);
    if (ok && separate_terminal_status && renderer->wrote_bytes &&
        !renderer->last_newline)
        ok = stream_newline(renderer);
    renderer->in_fence = renderer->channel_announced = 0;
    return ok && !ferror(renderer->output) ? YVEX_OK : YVEX_ERR_IO;
}
