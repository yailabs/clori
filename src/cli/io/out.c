/*
 * Provide approved direct text output calls for operator normal/table/audit text.
 *
 * Direct stdio output stays in this file; callers provide target streams; wrappers preserve stdio
 * return behavior where legacy code checks it. Writer calls serialize existing facts only and do
 * not create capability.
 */
#include "src/cli/io/private.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <operator/registry.h>
#include <yvex/internal/core.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { COMPLETION_CANDIDATE_CAP = 256, COMPLETION_TEXT_CAP = 128 };

typedef struct {
    char text[COMPLETION_CANDIDATE_CAP][COMPLETION_TEXT_CAP];
    size_t count;
} completion_candidates;

static int completion_visible(const yvex_operator_descriptor *descriptor)
{
    return descriptor->cli_projection &&
           descriptor->visibility != YVEX_OPERATOR_VISIBILITY_REMOVED &&
           descriptor->visibility != YVEX_OPERATOR_VISIBILITY_API_ONLY &&
           descriptor->visibility != YVEX_OPERATOR_VISIBILITY_TEST_ONLY;
}

static int completion_prefix_matches(const yvex_operator_descriptor *descriptor,
                                     size_t count, const char *const *words)
{
    size_t index;
    if (count > descriptor->command_word_count) return 0;
    for (index = 0u; index < count; ++index)
        if (strcmp(descriptor->command_words[index], words[index])) return 0;
    return 1;
}

static void completion_add(completion_candidates *candidates, const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;
    size_t index;
    if (!value[0] || strlen(value) >= COMPLETION_TEXT_CAP) return;
    while (*cursor) {
        if (!(isalnum(*cursor) || strchr("._:/@+-", *cursor))) return;
        cursor++;
    }
    for (index = 0u; index < candidates->count; ++index)
        if (!strcmp(candidates->text[index], value)) return;
    if (candidates->count >= COMPLETION_CANDIDATE_CAP) return;
    (void)snprintf(candidates->text[candidates->count], COMPLETION_TEXT_CAP, "%s", value);
    candidates->count++;
}

static void completion_add_metadata(completion_candidates *candidates, const char *values)
{
    const char *cursor = values;
    if (!strcmp(values, "none")) return;
    while (*cursor) {
        const char *end = strchr(cursor, '|');
        size_t extent = end ? (size_t)(end - cursor) : strlen(cursor);
        char item[COMPLETION_TEXT_CAP];
        if (extent < sizeof(item)) {
            memcpy(item, cursor, extent);
            item[extent] = '\0';
            completion_add(candidates, item);
        }
        if (!end) break;
        cursor = end + 1;
    }
}

static completion_candidates completion_collect(size_t prefix_count,
                                                const char *const *prefix)
{
    completion_candidates candidates = {{{0}}, 0u};
    size_t descriptor_index;
    for (descriptor_index = 0u; descriptor_index < yvex_operator_descriptor_count;
         ++descriptor_index) {
        const yvex_operator_descriptor *descriptor =
            &yvex_operator_descriptors[descriptor_index];
        size_t index;
        if (!completion_visible(descriptor) ||
            !completion_prefix_matches(descriptor, prefix_count, prefix))
            continue;
        if (descriptor->command_word_count > prefix_count) {
            completion_add(&candidates, descriptor->command_words[prefix_count]);
            continue;
        }
        for (index = 0u; index < descriptor->flag_count; ++index) {
            completion_add(&candidates, descriptor->flags[index].name);
            completion_add_metadata(&candidates, descriptor->flags[index].aliases);
        }
        for (index = 0u; index < descriptor->argument_count; ++index)
            completion_add_metadata(&candidates, descriptor->arguments[index].enum_values);
    }
    return candidates;
}

static void completion_emit_case(FILE *output, const char *shell,
                                 size_t prefix_count, const char *const *prefix)
{
    completion_candidates candidates = completion_collect(prefix_count, prefix);
    size_t index;
    if (!candidates.count) return;
    if (!strcmp(shell, "fish")) fputs("    case '", output);
    else fputs("    '", output);
    for (index = 0u; index < prefix_count; ++index)
        fprintf(output, "%s%s", index ? " " : "", prefix[index]);
    if (!strcmp(shell, "fish")) fputs("'\n      set candidates", output);
    else fputs("') candidates='", output);
    for (index = 0u; index < candidates.count; ++index)
        fprintf(output, " %s", candidates.text[index]);
    if (!strcmp(shell, "fish")) fputc('\n', output);
    else fputs(" ' ;;\n", output);
}

static void completion_emit_cases(FILE *output, const char *shell)
{
    size_t descriptor_index, prefix_count, prior;
    for (descriptor_index = 0u; descriptor_index < yvex_operator_descriptor_count;
         ++descriptor_index) {
        const yvex_operator_descriptor *descriptor =
            &yvex_operator_descriptors[descriptor_index];
        if (!completion_visible(descriptor)) continue;
        for (prefix_count = 0u; prefix_count <= descriptor->command_word_count;
             ++prefix_count) {
            int seen = 0;
            for (prior = 0u; prior < descriptor_index && !seen; ++prior) {
                const yvex_operator_descriptor *candidate =
                    &yvex_operator_descriptors[prior];
                if (completion_visible(candidate) &&
                    candidate->command_word_count >= prefix_count &&
                    completion_prefix_matches(candidate, prefix_count,
                                              descriptor->command_words))
                    seen = 1;
            }
            if (!seen)
                completion_emit_case(output, shell, prefix_count,
                                     descriptor->command_words);
        }
    }
}

int yvex_cli_completion_command(int argc, char **argv, size_t consumed)
{
    const char *shell = consumed + 1u < (size_t)argc ? argv[consumed + 1u] : NULL;
    if (!shell) return 2;
    if (!strcmp(shell, "bash")) {
        fputs("_yvex_complete() {\n"
              "  local cur=${COMP_WORDS[COMP_CWORD]} path='' candidates=''\n"
              "  if (( COMP_CWORD > 1 )); then "
              "path=${COMP_WORDS[*]:1:$((COMP_CWORD-1))}; fi\n"
              "  case \"$path\" in\n", stdout);
        completion_emit_cases(stdout, shell);
        fputs("  esac\n  COMPREPLY=( $(compgen -W \"$candidates\" -- \"$cur\") )\n"
              "}\ncomplete -F _yvex_complete yvex\n", stdout);
        return 0;
    }
    if (!strcmp(shell, "zsh")) {
        fputs("#compdef yvex\n_yvex_complete() {\n"
              "  local path='' candidates=''\n"
              "  if (( CURRENT > 2 )); then path=${(j: :)words[2,$((CURRENT-1))]}; fi\n"
              "  case \"$path\" in\n", stdout);
        completion_emit_cases(stdout, shell);
        fputs("  esac\n  compadd -- ${(z)candidates}\n}\ncompdef _yvex_complete yvex\n", stdout);
        return 0;
    }
    if (!strcmp(shell, "fish")) {
        fputs("function __yvex_candidates\n"
              "  set -l tokens (commandline -opc)\n"
              "  set -e tokens[1]\n"
              "  set -l path (string join ' ' $tokens)\n"
              "  set -l candidates\n"
              "  switch $path\n", stdout);
        completion_emit_cases(stdout, shell);
        fputs("  end\n  printf '%s\\n' $candidates\nend\n"
              "complete -c yvex -f -a '(__yvex_candidates)'\n", stdout);
        return 0;
    }
    fprintf(stderr, "yvex: completion shell must be bash, zsh, or fish\n");
    return 2;
}

int yvex_cli_out_vwritef(FILE *fp, const char *fmt, va_list ap)
{
    return vfprintf(fp ? fp : stdout, fmt ? fmt : "", ap);
}

int yvex_cli_out_writef(FILE *fp, const char *fmt, ...)
{
    va_list ap;
    int rc;

    va_start(ap, fmt);
    rc = yvex_cli_out_vwritef(fp, fmt, ap);
    va_end(ap);
    return rc;
}

int yvex_cli_out_puts(FILE *fp, const char *text)
{
    return fputs(text ? text : "", fp ? fp : stdout);
}

int yvex_cli_out_fputs(const char *text, FILE *fp)
{
    return yvex_cli_out_puts(fp, text);
}

int yvex_cli_out_char(FILE *fp, int ch)
{
    return fputc(ch, fp ? fp : stdout);
}

int yvex_cli_out_flush(FILE *fp)
{
    FILE *stream = fp ? fp : stdout;

    return fflush(stream) == 0 && !ferror(stream) ? YVEX_OK : YVEX_ERR_IO;
}

FILE *yvex_cli_out_stdout(void)
{
    return stdout;
}

FILE *yvex_cli_out_stderr(void)
{
    return stderr;
}

void yvex_cli_terminal_style_get(FILE *fp, yvex_cli_terminal_style *style)
{
    FILE *stream = fp ? fp : stdout;
    const char *terminal;
    int fd;

    if (!style) return;
    memset(style, 0, sizeof(*style));
    style->reset = "";
    style->strong = "";
    style->accent = "";
    style->dim = "";
    style->success = "";
    style->warning = "";
    style->error = "";
    fd = fileno(stream);
    terminal = getenv("TERM");
    if (fd < 0 || !isatty(fd) || getenv("NO_COLOR") ||
        (terminal && !strcmp(terminal, "dumb")))
        return;
    style->reset = "\033[0m";
    style->strong = "\033[1;38;5;250m";
    style->accent = "\033[38;5;81m";
    style->dim = "\033[38;5;245m";
    style->success = "\033[38;5;114m";
    style->warning = "\033[38;5;179m";
    style->error = "\033[38;5;203m";
}

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
    if (renderer->style.reset[0] && fputs(renderer->style.reset, renderer->output) == EOF)
        return 0;
    if (stream_style_code(renderer, style)[0] &&
        fputs(stream_style_code(renderer, style), renderer->output) == EOF)
        return 0;
    renderer->active_style = style;
    return 1;
}

static yvex_cli_stream_style stream_context_style(
    const yvex_cli_stream_renderer *renderer)
{
    if (renderer->in_fence || renderer->in_inline_code)
        return YVEX_CLI_STREAM_STYLE_ACCENT;
    if (renderer->line_style != YVEX_CLI_STREAM_STYLE_NORMAL)
        return renderer->line_style;
    if (renderer->channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING)
        return YVEX_CLI_STREAM_STYLE_DIM;
    if (renderer->channel == YVEX_CLIENT_STREAM_TOOL_CALL ||
        renderer->channel == YVEX_CLIENT_STREAM_TOOL_RESULT)
        return YVEX_CLI_STREAM_STYLE_ACCENT;
    return YVEX_CLI_STREAM_STYLE_NORMAL;
}

static int stream_bytes(yvex_cli_stream_renderer *renderer,
                        const unsigned char *bytes, size_t count)
{
    if (!stream_style_set(renderer, stream_context_style(renderer)) ||
        (count && fwrite(bytes, 1u, count, renderer->output) != count))
        return 0;
    if (count) {
        renderer->wrote_bytes = 1;
        renderer->last_newline = bytes[count - 1u] == '\n';
    }
    return 1;
}

static int stream_escape_byte(yvex_cli_stream_renderer *renderer,
                              unsigned char byte)
{
    char escaped[5];
    int count = snprintf(escaped, sizeof(escaped), "\\x%02x", byte);
    return count == 4 && stream_bytes(renderer, (const unsigned char *)escaped, 4u);
}

static int stream_language_finish(yvex_cli_stream_renderer *renderer)
{
    int ok = 1;
    if (renderer->fence_language_count) {
        ok = stream_style_set(renderer, YVEX_CLI_STREAM_STYLE_DIM) &&
             fputc('[', renderer->output) != EOF &&
             fwrite(renderer->fence_language, 1u,
                    renderer->fence_language_count, renderer->output) ==
                 renderer->fence_language_count &&
             fputc(']', renderer->output) != EOF;
        renderer->wrote_bytes = 1;
    }
    renderer->collecting_language = renderer->fence_language_count = 0u;
    return ok;
}

static int stream_newline(yvex_cli_stream_renderer *renderer)
{
    if (renderer->collecting_language && !stream_language_finish(renderer)) return 0;
    renderer->closing_fence = 0;
    if (!stream_style_set(renderer, stream_context_style(renderer)) ||
        fputc('\n', renderer->output) == EOF)
        return 0;
    renderer->line_start = renderer->wrote_bytes = renderer->last_newline = 1;
    renderer->line_style = YVEX_CLI_STREAM_STYLE_NORMAL;
    return 1;
}

static int stream_backticks_flush(yvex_cli_stream_renderer *renderer)
{
    unsigned int count = renderer->backtick_count;
    static const unsigned char ticks[] = "````````";
    renderer->backtick_count = 0u;
    if (!count) return 1;
    if (renderer->line_start && count >= 3u) {
        if (renderer->in_fence) {
            renderer->in_fence = 0;
            renderer->closing_fence = 1;
        } else {
            renderer->in_fence = 1;
            renderer->collecting_language = 1;
            renderer->fence_language_count = 0u;
        }
        count -= 3u;
    } else if (!renderer->in_fence && count == 1u) {
        renderer->in_inline_code = !renderer->in_inline_code;
        count = 0u;
    }
    while (count) {
        size_t extent = count > sizeof(ticks) - 1u ? sizeof(ticks) - 1u : count;
        if (!stream_bytes(renderer, ticks, extent)) return 0;
        count -= (unsigned int)extent;
    }
    return stream_style_set(renderer, stream_context_style(renderer));
}

static int stream_terminal_ascii(yvex_cli_stream_renderer *renderer,
                                 unsigned char byte)
{
    if (renderer->pending_cr) {
        renderer->pending_cr = 0;
        if (!stream_backticks_flush(renderer) || !stream_newline(renderer)) return 0;
        if (byte == '\n') return 1;
    }
    if (byte == '\r') {
        renderer->pending_cr = 1;
        return 1;
    }
    if (byte == '\n')
        return stream_backticks_flush(renderer) && stream_newline(renderer);
    if (byte == '`') {
        if (renderer->backtick_count < UINT_MAX) renderer->backtick_count++;
        return 1;
    }
    if (!stream_backticks_flush(renderer)) return 0;
    if (renderer->collecting_language) {
        if (byte >= 0x20u && byte < 0x7fu &&
            renderer->fence_language_count + 1u < sizeof(renderer->fence_language))
            renderer->fence_language[renderer->fence_language_count++] = (char)byte;
        return 1;
    }
    if (renderer->closing_fence && (byte == ' ' || byte == '\t')) return 1;
    renderer->closing_fence = 0;
    if (byte < 0x20u || byte == 0x7fu) {
        if (byte == '\t') return stream_bytes(renderer, &byte, 1u);
        return stream_escape_byte(renderer, byte);
    }
    if (renderer->line_start && !renderer->in_fence) {
        if (byte == '#') {
            renderer->line_style = YVEX_CLI_STREAM_STYLE_STRONG;
        } else if (byte == '>') {
            renderer->line_style = YVEX_CLI_STREAM_STYLE_DIM;
        } else if (byte == '-' || byte == '*') {
            renderer->line_style = YVEX_CLI_STREAM_STYLE_ACCENT;
        }
    }
    if (byte != ' ' && byte != '\t') renderer->line_start = 0;
    return stream_bytes(renderer, &byte, 1u);
}

static unsigned int stream_utf8_extent(unsigned char byte)
{
    if (byte >= 0xc2u && byte <= 0xdfu) return 2u;
    if (byte >= 0xe0u && byte <= 0xefu) return 3u;
    if (byte >= 0xf0u && byte <= 0xf4u) return 4u;
    return 0u;
}

static int stream_utf8_valid(const unsigned char *bytes, unsigned int count)
{
    if (count == 3u && bytes[0] == 0xe0u && bytes[1] < 0xa0u) return 0;
    if (count == 3u && bytes[0] == 0xedu && bytes[1] >= 0xa0u) return 0;
    if (count == 4u && bytes[0] == 0xf0u && bytes[1] < 0x90u) return 0;
    if (count == 4u && bytes[0] == 0xf4u && bytes[1] >= 0x90u) return 0;
    return 1;
}

static int stream_replacement(yvex_cli_stream_renderer *renderer)
{
    static const unsigned char replacement[] = {0xefu, 0xbfu, 0xbdu};
    return stream_bytes(renderer, replacement, sizeof(replacement));
}

static int stream_terminal_byte(yvex_cli_stream_renderer *renderer,
                                unsigned char byte)
{
    if (renderer->utf8_expected) {
        if ((byte & 0xc0u) != 0x80u) {
            renderer->utf8_count = renderer->utf8_expected = 0u;
            return stream_replacement(renderer) &&
                   stream_terminal_byte(renderer, byte);
        }
        renderer->utf8[renderer->utf8_count++] = byte;
        if (renderer->utf8_count < renderer->utf8_expected) return 1;
        if (!stream_utf8_valid(renderer->utf8, renderer->utf8_count)) {
            renderer->utf8_count = renderer->utf8_expected = 0u;
            return stream_replacement(renderer);
        }
        if (!stream_bytes(renderer, renderer->utf8, renderer->utf8_count)) return 0;
        renderer->line_start = 0;
        renderer->utf8_count = renderer->utf8_expected = 0u;
        return 1;
    }
    if (byte < 0x80u) return stream_terminal_ascii(renderer, byte);
    renderer->utf8_expected = stream_utf8_extent(byte);
    if (!renderer->utf8_expected) return stream_replacement(renderer);
    renderer->utf8[0] = byte;
    renderer->utf8_count = 1u;
    return 1;
}

void yvex_cli_stream_renderer_open(yvex_cli_stream_renderer *renderer,
                                   FILE *output, int enhanced)
{
    if (!renderer) return;
    memset(renderer, 0, sizeof(*renderer));
    renderer->output = output ? output : stdout;
    renderer->enhanced = enhanced != 0;
    renderer->line_start = renderer->last_newline = 1;
    renderer->channel = YVEX_CLIENT_STREAM_FINAL_TEXT;
    yvex_cli_terminal_style_get(renderer->output, &renderer->style);
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
        if (!stream_backticks_flush(renderer) ||
            (renderer->wrote_bytes && !renderer->last_newline &&
             !stream_newline(renderer)) ||
            (renderer->channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING &&
             channel == YVEX_CLIENT_STREAM_FINAL_TEXT && renderer->wrote_bytes && !stream_newline(renderer)))
            return YVEX_ERR_IO;
        renderer->channel = channel;
        if (!stream_style_set(renderer, stream_context_style(renderer)) ||
            (channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING &&
             (!stream_bytes(renderer, (const unsigned char *)"thinking", 8u) || !stream_newline(renderer))))
            return YVEX_ERR_IO;
    }
    for (index = 0u; index < count; ++index)
        if (!stream_terminal_byte(renderer, bytes[index])) return YVEX_ERR_IO;
    return ferror(renderer->output) ? YVEX_ERR_IO : YVEX_OK;
}

int yvex_cli_stream_renderer_finish(yvex_cli_stream_renderer *renderer,
                                    int separate_terminal_status)
{
    int ok;
    if (!renderer || !renderer->output) return YVEX_ERR_INVALID_ARG;
    if (!renderer->enhanced) {
        if (separate_terminal_status && renderer->wrote_bytes && !renderer->last_newline)
            renderer->last_newline = fputc('\n', renderer->output) != EOF;
        return !ferror(renderer->output) ? YVEX_OK : YVEX_ERR_IO;
    }
    ok = stream_backticks_flush(renderer);
    if (ok && renderer->utf8_expected) ok = stream_replacement(renderer);
    renderer->utf8_count = renderer->utf8_expected = 0u;
    if (ok && renderer->pending_cr) ok = stream_newline(renderer);
    renderer->pending_cr = renderer->collecting_language =
        renderer->closing_fence = renderer->in_fence = renderer->in_inline_code = 0;
    if (ok) ok = stream_style_set(renderer, YVEX_CLI_STREAM_STYLE_NORMAL);
    if (ok && separate_terminal_status && renderer->wrote_bytes &&
        !renderer->last_newline) {
        ok = fputc('\n', renderer->output) != EOF;
        renderer->last_newline = ok;
    }
    return ok && !ferror(renderer->output) ? YVEX_OK : YVEX_ERR_IO;
}

static int server_event_watch_visible(const yvex_server_event *event)
{
    return event && event->kind != YVEX_SERVER_EVENT_CLIENT_DISCONNECTED &&
           event->kind != YVEX_SERVER_EVENT_REQUEST_RECEIVED &&
           !(event->kind == YVEX_SERVER_EVENT_REQUEST_QUEUED && event->value_a <= 1u) &&
           !(event->kind >= YVEX_SERVER_EVENT_DRAFT_STARTED &&
             event->kind <= YVEX_SERVER_EVENT_CANDIDATE_REJECTED) &&
           !(event->kind >= YVEX_SERVER_EVENT_GENERATION_FRAGMENT &&
             event->kind <= YVEX_SERVER_EVENT_GENERATION_PROFILE);
}

static const char *server_event_category(yvex_server_event_kind kind)
{
    if (kind == YVEX_SERVER_EVENT_RUNTIME_READY) return "READY";
    if (kind <= YVEX_SERVER_EVENT_LISTENER_READY) return "STARTUP";
    if (kind <= YVEX_SERVER_EVENT_SESSION_CLOSED) return "SESSION";
    if (kind <= YVEX_SERVER_EVENT_REQUEST_STARTED) return "REQUEST";
    if (kind <= YVEX_SERVER_EVENT_PREFILL_COMPLETED) return "PREFILL";
    if (kind <= YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED) return "DSPARK";
    if (kind <= YVEX_SERVER_EVENT_GENERATION_FAILED) return "GENERATE";
    if (kind == YVEX_SERVER_EVENT_TELEMETRY_DROPPED) return "WARNING";
    return "RUNTIME";
}

static const char *server_event_color(const yvex_server_event *event,
                                      const yvex_cli_terminal_style *style)
{
    if (event->severity >= YVEX_SERVER_SEVERITY_ERROR) return style->error;
    if (event->severity == YVEX_SERVER_SEVERITY_WARNING) return style->warning;
    switch (event->kind) {
    case YVEX_SERVER_EVENT_RUNTIME_READY:
    case YVEX_SERVER_EVENT_LISTENER_READY:
    case YVEX_SERVER_EVENT_PREFILL_COMPLETED:
    case YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED:
    case YVEX_SERVER_EVENT_GENERATION_COMPLETED:
    case YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE:
        return style->success;
    case YVEX_SERVER_EVENT_GENERATION_CANCELLED:
    case YVEX_SERVER_EVENT_TELEMETRY_DROPPED:
        return style->warning;
    case YVEX_SERVER_EVENT_PREFILL_STARTED:
    case YVEX_SERVER_EVENT_PREFILL_PROGRESS:
    case YVEX_SERVER_EVENT_GENERATION_FIRST_TOKEN:
        return style->accent;
    default:
        return style->strong;
    }
}

const char *yvex_cli_out_stop_reason(unsigned long long reason)
{
    static const char *const names[] = {
        "none", "EOS", "tokenizer stop", "maximum tokens", "context capacity",
        "cancelled", "model failure", "tokenizer failure", "output failure"};
    return reason < sizeof(names) / sizeof(names[0]) ? names[reason] : "unknown";
}

void yvex_cli_out_turn_metrics(FILE *output, const yvex_client_message *message,
                               unsigned long long context_capacity,
                               const yvex_cli_terminal_style *style)
{
    if (message->generation_mode == YVEX_SERVER_GENERATION_MEDIA) {
        fprintf(output, "%smedia%s · %.2f s", style->success, style->reset,
                message->decode_seconds);
        return;
    }
    fprintf(output,
            "%sprefill%s %llu new/%llu prompt/%llu reused · %.2f s · %.2f tok/s · "
            "%sgeneration%s %llu tokens · %.2f s · %.2f tok/s · TTFT %.2f s",
            style->accent, style->reset, message->prefill_tokens, message->prompt_tokens,
            message->reused_tokens, message->prefill_seconds, message->prefill_rate,
            style->success, style->reset, message->generated_tokens, message->decode_seconds,
            message->decode_rate, message->first_token_seconds);
    if (message->generation_mode == YVEX_SERVER_GENERATION_DSPARK)
        fprintf(output, " · %sDSpark%s %llu proposed/%llu accepted/%llu rejected/%llu verified",
                style->accent, style->reset, message->proposed_tokens,
                message->accepted_draft_tokens, message->rejected_draft_tokens,
                message->target_verification_count);
    if (context_capacity)
        fprintf(output, " · context %llu/%llu", message->context_used, context_capacity);
    else
        fprintf(output, " · context %llu", message->context_used);
}

static const char *server_backend_name(unsigned long long backend)
{
    return backend == YVEX_BACKEND_KIND_CUDA ? "CUDA" : "CPU";
}

static void server_event_name(const yvex_server_event *event)
{
    const char *name = yvex_server_event_kind_name(event->kind);
    while (*name) {
        int byte = *name++;
        putchar(byte == '.' || byte == '_' ? ' ' : byte);
    }
}

static void server_event_bytes(const char *name, unsigned long long bytes)
{
    if (bytes >= 1073741824u)
        printf(" · %s %.2f GiB", name, (double)bytes / 1073741824.0);
    else
        printf(" · %s %.2f MiB", name, (double)bytes / 1048576.0);
}

static void server_event_values(const yvex_server_event *event, int detailed)
{
    if (event->kind >= YVEX_SERVER_EVENT_DRAFT_STARTED &&
        event->kind <= YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED) {
        printf(" · cycle %llu", event->speculative_cycle);
        if (event->proposed_tokens)
            printf(" · accepted %llu/%llu", event->accepted_tokens,
                   event->proposed_tokens);
        if (event->selected_verification_tokens)
            printf(" · verified %llu", event->selected_verification_tokens);
        if (event->rejected_tokens) printf(" · rejected %llu", event->rejected_tokens);
        if (event->discarded_tokens) printf(" · discarded %llu", event->discarded_tokens);
        if (detailed && event->confidence_logit_count)
            printf(" · confidence %.4g..%.4g mean %.4g",
                   event->confidence_logit_minimum, event->confidence_logit_maximum,
                   event->confidence_logit_mean);
        return;
    }
    switch (event->kind) {
    case YVEX_SERVER_EVENT_MATERIALIZATION_COMPLETE:
        server_event_bytes("host", event->value_a);
        server_event_bytes("device", event->value_b);
        printf(" · %llu tensor binding%s", event->value_c,
               event->value_c == 1u ? "" : "s");
        break;
    case YVEX_SERVER_EVENT_RESIDENCY_READY:
        server_event_bytes("host", event->value_a);
        server_event_bytes("device", event->value_b);
        printf(" · %llu upload%s", event->value_c, event->value_c == 1u ? "" : "s");
        break;
    case YVEX_SERVER_EVENT_ARTIFACT_OPEN_COMPLETE:
        server_event_bytes("hashed", event->value_a);
        server_event_bytes("host", event->value_b);
        server_event_bytes("device", event->value_c);
        break;
    case YVEX_SERVER_EVENT_LISTENER_READY:
        printf(" · socket %04llo · queue %llu · sessions %llu", event->value_a,
               event->value_b, event->value_c);
        break;
    case YVEX_SERVER_EVENT_RUNTIME_READY:
        printf(" · %s · context %llu", server_backend_name(event->value_c), event->value_b);
        break;
    case YVEX_SERVER_EVENT_SESSION_CREATED:
    case YVEX_SERVER_EVENT_SESSION_CLOSED:
        printf(" · %llu active", event->value_b);
        break;
    case YVEX_SERVER_EVENT_SESSION_ATTACHED:
    case YVEX_SERVER_EVENT_SESSION_DETACHED:
        printf(" · %llu client%s", event->value_a, event->value_a == 1u ? "" : "s");
        break;
    case YVEX_SERVER_EVENT_REQUEST_RECEIVED:
        if (detailed) printf(" · prompt %llu bytes", event->value_a);
        break;
    case YVEX_SERVER_EVENT_REQUEST_QUEUED:
        printf(" · depth %llu/%llu", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_REQUEST_STARTED:
        printf(" · input %llu · prefix %llu · limit %llu", event->value_a,
               event->value_b, event->value_c);
        break;
    case YVEX_SERVER_EVENT_TOKENIZER_COMPLETED:
        printf(" · prompt %llu · reused %llu", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_PREFILL_STARTED:
        printf(" · %llu new · chunk %llu", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_PREFILL_PROGRESS:
        printf(" · %llu/%llu tokens", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_PREFILL_COMPLETED:
        printf(" · %llu tokens · %llu chunk%s", event->value_a, event->value_b,
               event->value_b == 1u ? "" : "s");
        break;
    case YVEX_SERVER_EVENT_GENERATION_FIRST_TOKEN:
        if (detailed) printf(" · ordinal %llu · token %llu", event->value_a, event->value_b);
        break;
    case YVEX_SERVER_EVENT_GENERATION_COMPLETED:
    case YVEX_SERVER_EVENT_GENERATION_CANCELLED:
    case YVEX_SERVER_EVENT_GENERATION_FAILED:
        printf(" · %llu token%s · position %llu · stop %s", event->value_a,
               event->value_a == 1u ? "" : "s", event->value_b,
               yvex_cli_out_stop_reason(event->value_c));
        break;
    case YVEX_SERVER_EVENT_GENERATION_PROFILE:
        if (!strcmp(event->phase, "movement"))
            printf(" · H2D %llu · D2H %llu · D2D %llu bytes", event->value_a,
                   event->value_b, event->value_c);
        else if (!strcmp(event->phase, "transfers"))
            printf(" · uploads %llu · downloads %llu · expert subviews %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "launches"))
            printf(" · kernel launches %llu · stream syncs %llu · device syncs %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "target"))
            printf(" · target forwards %llu · rows %llu · replayed %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "speculation"))
            printf(" · draft forwards %llu · verified rows %llu · promoted rows %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "candidate"))
            printf(" · accepted %llu · discarded %llu · extensions %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "shape"))
            printf(" · shape hits %llu · misses %llu · host scan %llu bytes",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "moe"))
            printf(" · row/expert pairs %llu · subviews %llu · bytes %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "output"))
            printf(" · rows %llu · D2H %llu bytes · committed %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "prefill"))
            printf(" · prompt %llu · reused %llu · new %llu", event->value_a,
                   event->value_b, event->value_c);
        else if (!strcmp(event->phase, "decode"))
            printf(" · first decode %llu · later decode %llu · tokens %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batches"))
            printf(" · physical %llu · multi-source %llu · max width %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-rows"))
            printf(" · submitted %llu · executed %llu · admitted width %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-multi-source"))
            printf(" · physical %llu · rows %llu · max real width %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-sources"))
            printf(" · max sources %llu · active producers %llu",
                   event->value_a, event->value_b);
        else if (!strcmp(event->phase, "execution-batch-experts"))
            printf(" · worklists %llu · pairs %llu · max population %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-expert-rows"))
            printf(" · TC eligible %llu · TC executed %llu · narrow %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "batch-expert-population-1-3"))
            printf(" · population 1 %llu · 2 %llu · 3 %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "batch-expert-population-4-6"))
            printf(" · population 4 %llu · 5 %llu · 6 %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-coalescing"))
            printf(" · waits %llu · timeouts %llu · producers %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-policy"))
            printf(" · coalescing limit %llu ns · width %llu · producers %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-step-rendezvous"))
            printf(" · submissions %llu · multi-source %llu · max width %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-step-policy"))
            printf(" · limit %llu ns · steps %llu · producers %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-mismatch"))
            printf(" · phase %llu · layer %llu · operation %llu",
                   event->value_a, event->value_b, event->value_c);
        else if (!strcmp(event->phase, "execution-batch-mismatch-other"))
            printf(" · geometry %llu · profile %llu · identity %llu",
                   event->value_a, event->value_b, event->value_c);
        break;
    case YVEX_SERVER_EVENT_TELEMETRY_DROPPED:
        printf(" · %llu dropped · capacity %llu", event->value_a, event->value_b);
        break;
    default:
        break;
    }
}

int yvex_cli_out_server_event(const yvex_server_event *event, int detailed)
{
    static const char *const severity[] = {"debug", "info", "warning", "error", "fatal"};
    yvex_cli_terminal_style style;
    const char *color;
    unsigned int level;
    if (!event || (!detailed && !server_event_watch_visible(event))) return 0;
    yvex_cli_terminal_style_get(stdout, &style);
    color = server_event_color(event, &style);
    level = event->severity <= YVEX_SERVER_SEVERITY_FATAL
                ? (unsigned int)event->severity : YVEX_SERVER_SEVERITY_FATAL;
    if (detailed) {
        const char *severity_color = event->severity >= YVEX_SERVER_SEVERITY_ERROR
                                         ? style.error
                                         : event->severity == YVEX_SERVER_SEVERITY_WARNING
                                               ? style.warning
                                               : style.dim;
        printf("%s#%llu%s %s%-7s%s ", style.dim, event->sequence, style.reset,
               severity_color, severity[level], style.reset);
    } else {
        time_t seconds = (time_t)(event->wall_time_ns / 1000000000u);
        struct tm clock;
        char stamp[16] = "--:--:--";
        if (event->wall_time_ns && localtime_r(&seconds, &clock))
            (void)strftime(stamp, sizeof(stamp), "%H:%M:%S", &clock);
        printf("%s%s%s  %s%-8s%s ", style.dim, stamp, style.reset, color,
               server_event_category(event->kind), style.reset);
    }
    fputs(color, stdout);
    server_event_name(event);
    fputs(style.reset, stdout);
    if (!detailed && event->session_id[0] && event->request_id[0])
        printf(" · %s/%s", event->session_id, event->request_id);
    else {
        if (event->session_id[0]) printf(" · session %s", event->session_id);
        if (event->request_id[0]) printf(" · request %s", event->request_id);
    }
    if (detailed && event->turn_id[0]) printf(" · turn %s", event->turn_id);
    if (detailed && event->phase[0]) printf(" · phase %s", event->phase);
    server_event_values(event, detailed);
    if (event->seconds > 0.0) printf(" · %.3f s", event->seconds);
    if (event->rate > 0.0) printf(" · %.2f tok/s", event->rate);
    putchar('\n');
    fflush(stdout);
    return 1;
}

static void watch_stamp(const yvex_server_event *event, char stamp[16])
{
    time_t seconds = (time_t)(event->wall_time_ns / 1000000000u);
    struct tm clock;
    memcpy(stamp, "--:--:--", 9u);
    if (event->wall_time_ns && localtime_r(&seconds, &clock))
        (void)strftime(stamp, 16u, "%H:%M:%S", &clock);
}

static void watch_line_begin(const yvex_cli_watch_renderer *renderer,
                             const yvex_server_event *event,
                             const char *color, const char *label)
{
    char stamp[16];
    watch_stamp(event, stamp);
    printf("%s%s%s  %s%-9s%s ", renderer->style.dim, stamp,
           renderer->style.reset, color, label, renderer->style.reset);
}

static void watch_request_id(const yvex_server_event *event)
{
    if (event->session_id[0] && event->request_id[0])
        printf("%s/%s", event->session_id, event->request_id);
    else if (event->session_id[0])
        fputs(event->session_id, stdout);
    else if (event->request_id[0])
        fputs(event->request_id, stdout);
    else
        fputs("runtime", stdout);
}

static void watch_request_begin(yvex_cli_watch_renderer *renderer,
                                const yvex_server_event *event)
{
    if (renderer->request_open &&
        (!event->session_id[0] || !event->request_id[0] ||
         strcmp(renderer->session_id, event->session_id) ||
         strcmp(renderer->request_id, event->request_id))) {
        printf("%s          previous request has no terminal event%s\n\n",
               renderer->style.warning, renderer->style.reset);
        renderer->request_open = 0;
    }
    if (renderer->request_open) return;
    renderer->request_open = 1;
    renderer->cycles = renderer->proposed = renderer->accepted = 0ull;
    renderer->rejected = renderer->discarded = 0ull;
    yvex_core_text_copy(renderer->session_id, sizeof(renderer->session_id),
                        event->session_id);
    yvex_core_text_copy(renderer->request_id, sizeof(renderer->request_id),
                        event->request_id);
    putchar('\n');
    watch_line_begin(renderer, event, renderer->style.accent, "REQUEST");
    watch_request_id(event);
    if (event->kind == YVEX_SERVER_EVENT_REQUEST_STARTED)
        printf("  %sinput %llu · prefix %llu · limit %llu%s",
               renderer->style.dim, event->value_a, event->value_b,
               event->value_c, renderer->style.reset);
    putchar('\n');
}

static void watch_session(const yvex_cli_watch_renderer *renderer,
                          const yvex_server_event *event)
{
    const char *verb = yvex_server_event_kind_name(event->kind);
    const char *dot = strrchr(verb, '.');
    watch_line_begin(renderer, event, server_event_color(event, &renderer->style),
                     "SESSION");
    printf("%-20s %s", event->session_id, dot ? dot + 1 : verb);
    if (event->kind == YVEX_SERVER_EVENT_SESSION_ATTACHED ||
        event->kind == YVEX_SERVER_EVENT_SESSION_DETACHED)
        printf(" · %llu client%s", event->value_a, event->value_a == 1ull ? "" : "s");
    else if (event->kind == YVEX_SERVER_EVENT_SESSION_CREATED ||
             event->kind == YVEX_SERVER_EVENT_SESSION_CLOSED)
        printf(" · %llu active", event->value_b);
    putchar('\n');
}

static void watch_cycle(yvex_cli_watch_renderer *renderer,
                        const yvex_server_event *event)
{
    renderer->cycles++;
    renderer->proposed += event->proposed_tokens;
    renderer->accepted += event->accepted_tokens;
    renderer->rejected += event->rejected_tokens;
    renderer->discarded += event->discarded_tokens;
    if (!renderer->detailed) return;
    watch_line_begin(renderer, event, renderer->style.success, "DSPARK");
    printf("cycle %-3llu %llu/%llu accepted", event->speculative_cycle,
           event->accepted_tokens, event->proposed_tokens);
    if (event->selected_verification_tokens)
        printf(" · verify %llu", event->selected_verification_tokens);
    if (event->rejected_tokens) printf(" · %llu rejected", event->rejected_tokens);
    if (event->discarded_tokens) printf(" · %llu stop-discarded", event->discarded_tokens);
    if (event->seconds > 0.0) printf(" · %.3f s", event->seconds);
    putchar('\n');
}

static void watch_request_end(yvex_cli_watch_renderer *renderer,
                              const yvex_server_event *event)
{
    const char *label = event->kind == YVEX_SERVER_EVENT_GENERATION_COMPLETED
                            ? "COMPLETE"
                        : event->kind == YVEX_SERVER_EVENT_GENERATION_CANCELLED
                            ? "CANCELLED"
                            : "FAILED";
    const char *color = server_event_color(event, &renderer->style);
    watch_line_begin(renderer, event, color, label);
    printf("%llu token%s · position %llu · %s", event->value_a,
           event->value_a == 1ull ? "" : "s", event->value_b,
           yvex_cli_out_stop_reason(event->value_c));
    if (event->seconds > 0.0) printf(" · %.3f s", event->seconds);
    if (event->rate > 0.0) printf(" · %.2f tok/s", event->rate);
    if (renderer->cycles)
        printf("\n          %sDSPARK %llu cycle%s · %llu/%llu accepted · "
               "%llu rejected · %llu discarded%s",
               renderer->style.dim, renderer->cycles,
               renderer->cycles == 1ull ? "" : "s", renderer->accepted,
               renderer->proposed, renderer->rejected, renderer->discarded,
               renderer->style.reset);
    puts("\n");
    renderer->request_open = 0;
}

void yvex_cli_watch_renderer_open(yvex_cli_watch_renderer *renderer, int detailed)
{
    if (!renderer) return;
    memset(renderer, 0, sizeof(*renderer));
    renderer->detailed = detailed != 0;
    yvex_cli_terminal_style_get(stdout, &renderer->style);
}

int yvex_cli_watch_renderer_event(yvex_cli_watch_renderer *renderer,
                                  const yvex_server_event *event)
{
    if (!renderer || !event) return 0;
    if (event->kind <= YVEX_SERVER_EVENT_LISTENER_READY ||
        event->kind == YVEX_SERVER_EVENT_CLIENT_DISCONNECTED ||
        event->kind == YVEX_SERVER_EVENT_REQUEST_RECEIVED ||
        event->kind == YVEX_SERVER_EVENT_GENERATION_FRAGMENT ||
        event->kind == YVEX_SERVER_EVENT_GENERATION_PROGRESS ||
        event->kind == YVEX_SERVER_EVENT_GENERATION_PROFILE ||
        (event->kind >= YVEX_SERVER_EVENT_DRAFT_STARTED &&
         event->kind <= YVEX_SERVER_EVENT_CANDIDATE_REJECTED))
        return 0;
    if (event->kind >= YVEX_SERVER_EVENT_SESSION_CREATED &&
        event->kind <= YVEX_SERVER_EVENT_SESSION_CLOSED) {
        watch_session(renderer, event);
        return 1;
    }
    if (event->kind == YVEX_SERVER_EVENT_REQUEST_QUEUED) {
        if (event->value_a <= 1ull) return 0;
        watch_line_begin(renderer, event, renderer->style.warning, "QUEUE");
        watch_request_id(event);
        printf(" · depth %llu/%llu\n", event->value_a, event->value_b);
        return 1;
    }
    if (event->kind == YVEX_SERVER_EVENT_REQUEST_STARTED) {
        watch_request_begin(renderer, event);
        return 1;
    }
    if (event->kind >= YVEX_SERVER_EVENT_TOKENIZER_COMPLETED &&
        event->kind <= YVEX_SERVER_EVENT_GENERATION_FAILED)
        watch_request_begin(renderer, event);
    if (event->kind == YVEX_SERVER_EVENT_TOKENIZER_COMPLETED) {
        watch_line_begin(renderer, event, renderer->style.strong, "INPUT");
        printf("%llu prompt tokens · %llu reused\n", event->value_a, event->value_b);
    } else if (event->kind == YVEX_SERVER_EVENT_PREFILL_COMPLETED) {
        watch_line_begin(renderer, event, renderer->style.success, "PREFILL");
        printf("%llu token%s · %llu chunk%s", event->value_a,
               event->value_a == 1ull ? "" : "s", event->value_b,
               event->value_b == 1ull ? "" : "s");
        if (event->seconds > 0.0) printf(" · %.3f s", event->seconds);
        if (event->rate > 0.0) printf(" · %.2f tok/s", event->rate);
        putchar('\n');
    } else if (event->kind == YVEX_SERVER_EVENT_GENERATION_FIRST_TOKEN) {
        watch_line_begin(renderer, event, renderer->style.accent, "FIRST");
        printf("first committed token");
        if (event->seconds > 0.0) printf(" · TTFT %.3f s", event->seconds);
        putchar('\n');
    } else if (event->kind == YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED) {
        watch_cycle(renderer, event);
    } else if (event->kind >= YVEX_SERVER_EVENT_GENERATION_COMPLETED &&
               event->kind <= YVEX_SERVER_EVENT_GENERATION_FAILED) {
        watch_request_end(renderer, event);
    } else if (event->kind == YVEX_SERVER_EVENT_TELEMETRY_DROPPED) {
        watch_line_begin(renderer, event, renderer->style.warning, "WARNING");
        printf("telemetry dropped %llu event%s · capacity %llu\n", event->value_a,
               event->value_a == 1ull ? "" : "s", event->value_b);
    } else if (event->kind >= YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_START) {
        watch_line_begin(renderer, event, server_event_color(event, &renderer->style),
                         "RUNTIME");
        server_event_name(event);
        putchar('\n');
    } else {
        return 0;
    }
    fflush(stdout);
    return 1;
}

void yvex_cli_watch_renderer_finish(yvex_cli_watch_renderer *renderer)
{
    if (!renderer || !renderer->request_open) return;
    printf("%s          request stream ended without a terminal event%s\n",
           renderer->style.warning, renderer->style.reset);
    renderer->request_open = 0;
}

void yvex_cli_out_repl_catalog(void)
{
    yvex_cli_terminal_style style;
    size_t index, pass;
    yvex_cli_terminal_style_get(stdout, &style);
    printf("%scommands%s", style.strong, style.reset);
    for (pass = 0u; pass < 3u; ++pass) {
        for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
            const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
            int priority = descriptor->runtime_adapter == YVEX_OPERATOR_RUNTIME_HELP ? 0
                         : descriptor->repl_adapter == YVEX_OPERATOR_REPL_QUIT ? 2 : 1;
            if (!strcmp(descriptor->slash_projection, "none") || priority != (int)pass) continue;
            printf("\n  %s%-12s%s %s%s%s", style.accent,
                   descriptor->slash_projection, style.reset, style.dim,
                   descriptor->summary, style.reset);
        }
    }
    printf("\n\n  %s%-12s%s %scancel an active turn or clear input; press again to exit%s",
           style.warning, "Ctrl-C", style.reset, style.dim, style.reset);
    printf("\n  %s%-12s%s %sexit and discard an unfinished line%s", style.accent,
           "Ctrl-D", style.reset, style.dim, style.reset);
    printf("\n  %s%-12s%s %sclear and redraw input%s", style.accent, "Ctrl-L",
           style.reset, style.dim, style.reset);
    puts("\n");
}

void yvex_cli_out_line(FILE *fp, const char *text)
{
    (void)yvex_cli_out_puts(fp, text);
    (void)yvex_cli_out_char(fp, '\n');
}

void yvex_cli_out_lines(FILE *fp,
                        const char *const *lines,
                        size_t line_count)
{
    size_t i;

    if (!lines) {
        return;
    }
    for (i = 0; i < line_count; ++i) {
        yvex_cli_out_line(fp, lines[i]);
    }
}

void yvex_cli_out_kv_str(FILE *fp, const char *key, const char *value)
{
    (void)yvex_cli_out_writef(fp, "%s: %s\n", key ? key : "", value ? value : "");
}

void yvex_cli_out_kv_bool(FILE *fp, const char *key, int value)
{
    yvex_cli_out_kv_str(fp, key, value ? "true" : "false");
}

int yvex_cli_out_fields(FILE *fp,
                        const void *object,
                        const yvex_cli_field_spec *fields,
                        size_t field_count)
{
    const unsigned char *base = object;
    size_t i;

    if (!object || (!fields && field_count != 0u)) {
        return -1;
    }
    for (i = 0; i < field_count; ++i) {
        const yvex_cli_field_spec *field = &fields[i];
        const void *value = base + field->offset;
        const char *text;
        int rc;

        switch (field->kind) {
        case YVEX_CLI_FIELD_TEXT:
            text = *(const char *const *)value;
            rc = yvex_cli_out_writef(fp, "%s: %s\n", field->key,
                                     text && text[0]
                                         ? text
                                         : (field->fallback ? field->fallback : "unknown"));
            break;
        case YVEX_CLI_FIELD_TEXT_ARRAY:
            text = value;
            rc = yvex_cli_out_writef(fp, "%s: %s\n", field->key,
                                     text[0]
                                         ? text
                                         : (field->fallback ? field->fallback : "unknown"));
            break;
        case YVEX_CLI_FIELD_U64:
            rc = yvex_cli_out_writef(fp, "%s: %llu\n", field->key,
                                     *(const unsigned long long *)value);
            break;
        case YVEX_CLI_FIELD_U32:
            rc = yvex_cli_out_writef(fp, "%s: %u\n", field->key,
                                     *(const unsigned int *)value);
            break;
        case YVEX_CLI_FIELD_I32:
            rc = yvex_cli_out_writef(fp, "%s: %d\n", field->key, *(const int *)value);
            break;
        case YVEX_CLI_FIELD_BOOL:
            rc = yvex_cli_out_writef(fp, "%s: %s\n", field->key,
                                     *(const int *)value ? "true" : "false");
            break;
        case YVEX_CLI_FIELD_DOUBLE:
            rc = yvex_cli_out_writef(fp, "%s: %.17g\n", field->key,
                                     *(const double *)value);
            break;
        case YVEX_CLI_FIELD_FLOAT9:
            rc = yvex_cli_out_writef(fp, "%s: %.9g\n", field->key,
                                     *(const double *)value);
            break;
        case YVEX_CLI_FIELD_HEX64:
            rc = yvex_cli_out_writef(fp, "%s: %016llx\n", field->key,
                                     *(const unsigned long long *)value);
            break;
        default:
            return -1;
        }
        if (rc < 0) {
            return -1;
        }
    }
    return 0;
}

int print_yvex_error(const yvex_error *err, int exit_code)
{
    yvex_cli_out_writef(stderr, "yvex: %s: %s\n", yvex_error_where(err),
                        yvex_error_message(err));
    return exit_code;
}

int exit_for_status(int status)
{
    switch (status) {
    case YVEX_ERR_INVALID_ARG:
        return 2;
    case YVEX_ERR_IO:
        return 3;
    case YVEX_ERR_FORMAT:
    case YVEX_ERR_BOUNDS:
        return 4;
    case YVEX_ERR_UNSUPPORTED:
        return 5;
    default:
        return 1;
    }
}

/*
 * Parse one complete unsigned integer without accepting signs or suffixes.
 *
 * Returns false and leaves result ownership with the caller.
 */
int parse_ull_allow_zero(const char *text, unsigned long long *out)
{
    char *end = NULL;
    unsigned long long value;

    if (!text || !out || text[0] == '\0' || text[0] == '-') return 0;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *out = value;
    return 1;
}

int parse_positive_ull(const char *text, unsigned long long *out)
{
    return parse_ull_allow_zero(text, out) && *out != 0;
}

int parse_uint_allow_zero(const char *text, unsigned int *out)
{
    unsigned long long value;

    if (!out || !parse_ull_allow_zero(text, &value) || value > UINT32_MAX) return 0;
    *out = (unsigned int)value;
    return 1;
}

void print_quoted_bytes(const char *data, unsigned long long len)
{
    unsigned long long i;

    yvex_cli_out_writef(stdout, "\"");
    for (i = 0; i < len; ++i) {
        unsigned char ch = (unsigned char)data[i];
        if (ch == '"' || ch == '\\') {
            yvex_cli_out_writef(stdout, "\\%c", (int)ch);
        } else if (ch == '\n') {
            yvex_cli_out_writef(stdout, "\\n");
        } else if (ch == '\r') {
            yvex_cli_out_writef(stdout, "\\r");
        } else if (ch == '\t') {
            yvex_cli_out_writef(stdout, "\\t");
        } else if (ch < 32 || ch > 126) {
            yvex_cli_out_writef(stdout, "\\x%02x", (unsigned int)ch);
        } else {
            yvex_cli_out_writef(stdout, "%c", (int)ch);
        }
    }
    yvex_cli_out_writef(stdout, "\"");
}

int open_artifact_for_gguf(const char *path, yvex_artifact **artifact, yvex_error *err)
{
    yvex_artifact_options options;
    yvex_model_ref ref;
    int rc;

    memset(&options, 0, sizeof(options));
    memset(&ref, 0, sizeof(ref));
    rc = yvex_model_ref_resolve(&ref, path, NULL, err);
    if (rc != YVEX_OK) return rc;
    options.path = ref.path;
    options.readonly = 1;
    rc = yvex_artifact_open(artifact, &options, err);
    yvex_model_ref_clear(&ref);
    return rc;
}

void print_tensor_dims(const unsigned long long *dims, unsigned int rank)
{
    unsigned int i;

    yvex_cli_out_writef(stdout, "[");
    for (i = 0; i < rank; ++i) {
        if (i > 0) yvex_cli_out_writef(stdout, ",");
        yvex_cli_out_writef(stdout, "%llu", dims[i]);
    }
    yvex_cli_out_writef(stdout, "]");
}

void print_native_dims(const unsigned long long *dims, unsigned int rank)
{
    print_tensor_dims(dims, rank);
}

void print_token_ids(const yvex_tokens *tokens)
{
    unsigned long long i;

    yvex_cli_out_writef(stdout, "ids:");
    for (i = 0; i < tokens->len; ++i) yvex_cli_out_writef(stdout, " %u", tokens->ids[i]);
    yvex_cli_out_writef(stdout, "\n");
}

int parse_id_list(const char *text, unsigned int **out_ids, unsigned long long *out_len)
{
    unsigned int *ids = NULL;
    unsigned long long len = 0;
    unsigned long long capacity = 0;
    const char *cursor = text;

    if (!text || !out_ids || !out_len) return 0;
    *out_ids = NULL;
    *out_len = 0;
    while (*cursor) {
        char *end = NULL;
        unsigned long value = strtoul(cursor, &end, 10);
        unsigned int *next;

        if (end == cursor || value > UINT32_MAX) goto fail;
        if (len == capacity) {
            unsigned long long next_capacity = capacity == 0 ? 8 : capacity * 2u;
            if (next_capacity > (unsigned long long)(SIZE_MAX / sizeof(*ids))) goto fail;
            next = realloc(ids, (size_t)next_capacity * sizeof(*ids));
            if (!next) goto fail;
            ids = next;
            capacity = next_capacity;
        }
        ids[len++] = (unsigned int)value;
        if (*end == ',') {
            cursor = end + 1;
        } else if (*end == '\0') {
            cursor = end;
        } else {
            goto fail;
        }
    }
    if (len == 0) goto fail;
    *out_ids = ids;
    *out_len = len;
    return 1;

fail:
    free(ids);
    return 0;
}

/*
 * Parse an exact-rank comma-separated tensor shape.
 *
 * Rejects invalid rank, malformed fields, zero dimensions, and overflow.
 */
int parse_dims_csv(const char *text, unsigned int rank, unsigned long long dims[4])
{
    const char *cursor = text;
    char *end = NULL;
    unsigned int i;

    if (!text || !dims || rank == 0 || rank > 4u) return 0;
    memset(dims, 0, 4u * sizeof(*dims));
    for (i = 0; i < rank; ++i) {
        errno = 0;
        dims[i] = strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor || dims[i] == 0) return 0;
        if (i + 1u < rank) {
            if (*end != ',') return 0;
            cursor = end + 1;
        } else if (*end != '\0') {
            return 0;
        }
    }
    return 1;
}

/*
 * Provide approved direct JSON text output for CLI plumbing surfaces.
 *
 * Helpers serialize only caller-provided fields and do not claim uniform JSON. JSON writer
 * primitives are not command-level JSON support by themselves.
 */
static void json_string(FILE *fp, const char *text) {
    const unsigned char *p = (const unsigned char *)(text ? text : "");

    (void)yvex_cli_out_char(fp, '"');
    while (*p) {
        if (*p == '"' || *p == '\\') {
            (void)yvex_cli_out_char(fp, '\\');
            (void)yvex_cli_out_char(fp, *p);
        } else if (*p == '\n') {
            (void)yvex_cli_out_puts(fp, "\\n");
        } else if (*p == '\r') {
            (void)yvex_cli_out_puts(fp, "\\r");
        } else if (*p == '\t') {
            (void)yvex_cli_out_puts(fp, "\\t");
        } else if (*p < 0x20u) {
            (void)yvex_cli_out_writef(fp, "\\u%04x", (unsigned int)*p);
        } else {
            (void)yvex_cli_out_char(fp, *p);
        }
        ++p;
    }
    (void)yvex_cli_out_char(fp, '"');
}

void yvex_cli_json_begin(FILE *fp) {
    yvex_cli_out_line(fp, "{");
}

void yvex_cli_json_end(FILE *fp) {
    yvex_cli_out_line(fp, "}");
}

static int json_field(FILE *fp, const char *key, yvex_cli_field_kind kind, const void *value,
                      int comma) {
    (void)yvex_cli_out_puts(fp, "  ");
    json_string(fp, key);
    (void)yvex_cli_out_puts(fp, ": ");
    switch (kind) {
    case YVEX_CLI_FIELD_TEXT:
    case YVEX_CLI_FIELD_TEXT_ARRAY:
        json_string(fp, value);
        break;
    case YVEX_CLI_FIELD_U64:
        (void)yvex_cli_out_writef(fp, "%llu", *(const unsigned long long *)value);
        break;
    case YVEX_CLI_FIELD_U32:
        (void)yvex_cli_out_writef(fp, "%u", *(const unsigned int *)value);
        break;
    case YVEX_CLI_FIELD_I32:
        (void)yvex_cli_out_writef(fp, "%d", *(const int *)value);
        break;
    case YVEX_CLI_FIELD_BOOL:
        (void)yvex_cli_out_puts(fp, *(const int *)value ? "true" : "false");
        break;
    case YVEX_CLI_FIELD_DOUBLE:
        if (isfinite(*(const double *)value))
            (void)yvex_cli_out_writef(fp, "%.17g", *(const double *)value);
        else
            (void)yvex_cli_out_puts(fp, "null");
        break;
    default:
        return YVEX_ERR_UNSUPPORTED;
    }
    (void)yvex_cli_out_writef(fp, "%s\n", comma ? "," : "");
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}

void yvex_cli_json_field_str(FILE *fp, const char *key, const char *value, int comma) {
    (void)json_field(fp, key, YVEX_CLI_FIELD_TEXT_ARRAY, value ? value : "", comma);
}

void yvex_cli_json_field_u64(FILE *fp, const char *key, unsigned long long value, int comma) {
    (void)json_field(fp, key, YVEX_CLI_FIELD_U64, &value, comma);
}

void yvex_cli_json_field_bool(FILE *fp, const char *key, int value, int comma) {
    (void)json_field(fp, key, YVEX_CLI_FIELD_BOOL, &value, comma);
}

int yvex_cli_json_fields(FILE *fp, const void *object, const yvex_cli_field_spec *fields,
                         size_t field_count, int comma) {
    const unsigned char *base = object;
    size_t index;

    if (!fp || !object || (!fields && field_count))
        return YVEX_ERR_INVALID_ARG;
    for (index = 0; index < field_count; ++index) {
        const yvex_cli_field_spec *field = &fields[index];
        const void *value = base + field->offset;
        int separator = comma || index + 1u < field_count;
        if (field->kind == YVEX_CLI_FIELD_TEXT)
            value = *(const char *const *)value;
        if (json_field(fp, field->key, field->kind, value, separator) != YVEX_OK)
            return ferror(fp) ? YVEX_ERR_IO : YVEX_ERR_UNSUPPORTED;
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
